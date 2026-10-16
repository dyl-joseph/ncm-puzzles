"""Budgeted weight-warm-start training blocks with verified immutable archives."""

from __future__ import annotations

import argparse
import codecs
from dataclasses import asdict, dataclass
import hashlib
import json
import lzma
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Literal

from .verify_training import verify as verify_training


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_FILES = ("generator.safetensors", "tokenizer.safetensors", "metadata.json",
                    "provenance.json", "inference_controls.json")
RUNTIME_FILES = ("colab_training.py", "train_gameplay.py", "gameplay_worker.py", "gpu_selfcheck.py",
                 "mlx_ncm.py", "mlx_worker.py", "encoding.py", "run_agent.py", "verify_training.py",
                 "verify_score.py", "public-baselines.json", "fixtures/replay_controls.json")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def checkpoint_hashes(path):
    return {name: digest(path / name) for name in CHECKPOINT_FILES}


def run_process(command, *, timeout, log=None, status=None):
    print("Running " + " ".join(map(str, command)), flush=True)
    with log.open("w+b") if log else tempfile.TemporaryFile() as stream:
        process = subprocess.Popen(list(map(str, command)), cwd=ROOT, start_new_session=True,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   env={**os.environ, "MLX_ENABLE_TF32": "0", "MPLBACKEND": "Agg"})
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        offset = 0

        def forward_output(*, final=False):
            nonlocal offset
            if log:
                return
            end = os.fstat(stream.fileno()).st_size
            if not final:
                end = min(end, offset + 1024 * 1024)
            while offset < end:
                chunk = os.pread(stream.fileno(), min(65536, end - offset), offset)
                offset += len(chunk)
                sys.stdout.write(decoder.decode(chunk))
            if final:
                sys.stdout.write(decoder.decode(b"", final=True))
            sys.stdout.flush()

        try:
            deadline = time.monotonic() + timeout
            next_status = time.monotonic() + 30
            while True:
                forward_output()
                try:
                    return_code = process.wait(timeout=min(0.1, max(0.001, deadline - time.monotonic())))
                    break
                except subprocess.TimeoutExpired:
                    now = time.monotonic()
                    if now >= deadline:
                        raise
                    if now >= next_status:
                        if status and status.exists():
                            print(json.dumps({"training_progress": json.loads(status.read_text())}), flush=True)
                        next_status = now + 30
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if process.poll() is None:
                process.wait()
            forward_output(final=True)
        if return_code:
            end = os.fstat(stream.fileno()).st_size
            tail = os.pread(stream.fileno(), min(8000, end), max(0, end - 8000)).decode(errors="replace")
            raise RuntimeError(f"Process failed with exit {return_code}. Log {log}\n{tail}")


@dataclass(frozen=True)
class RunSpec:
    revision: str
    total_seconds: int
    block_seconds: int
    base_seed: int
    initial_checkpoint: dict
    source_sha256: dict
    game_source_sha256: dict
    baseline_sha256: str
    resume_kind: str = "weights only; Adam, RNG, EWMA baseline and policy counters reset each block"


@dataclass
class Attempt:
    index: int
    state: Literal["running", "completed", "failed"]
    reserved_seconds: int
    seed: int
    initial_checkpoint: dict
    completion_sha256: str | None = None
    error: str | None = None


class TrainingRun:
    def __init__(self, store, local, spec, initial_checkpoint, runner=run_process, verifier=verify_training):
        self.store, self.local, self.spec = store.resolve(), local.resolve(), spec
        self.initial_checkpoint = initial_checkpoint.resolve()
        self.runner, self.verifier = runner, verifier
        self.spec_hash = object_digest(asdict(spec))
        self.store.mkdir(parents=True, exist_ok=True)
        self.local.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.store / "ledger.json"
        if self.ledger_path.exists():
            ledger = json.loads(self.ledger_path.read_text())
            if ledger["schema"] != 1 or ledger["spec"] != asdict(spec) or ledger["spec_sha256"] != self.spec_hash:
                raise ValueError("Run configuration changed. Restore it or choose a new RUN_NAME.")
            self.attempts = [Attempt(**row) for row in ledger["attempts"]]
        else:
            if any(self.store.iterdir()):
                raise ValueError("Run store has files but no ledger. Choose a new RUN_NAME.")
            self.attempts = []
            self.save_ledger()
        self.checkpoint = self.initial_checkpoint
        self.recover()

    @property
    def charged_seconds(self):
        return sum(attempt.reserved_seconds for attempt in self.attempts)

    @property
    def can_evaluate(self):
        return (self.spec.total_seconds - self.charged_seconds < 30 and
                any(attempt.state == "completed" for attempt in self.attempts))

    def save_ledger(self):
        write_json(self.ledger_path, {"schema": 1, "spec": asdict(self.spec), "spec_sha256": self.spec_hash,
                                    "attempts": [asdict(attempt) for attempt in self.attempts]})

    def bundle_path(self, attempt):
        return self.store / "blocks" / f"{attempt.index:04d}"

    def validate_bundle(self, directory, attempt):
        manifest_path = directory / "completion.json"
        manifest = json.loads(manifest_path.read_text())
        if (manifest["spec_sha256"] != self.spec_hash or manifest["attempt_index"] != attempt.index or
                manifest["initial_checkpoint"] != attempt.initial_checkpoint or manifest["seed"] != attempt.seed or
                manifest["reserved_seconds"] != attempt.reserved_seconds):
            raise ValueError("Completed block belongs to a different run or attempt")
        required = {"config.json", "summary.json", "deadline-result.json", "verification.json",
                    "game-trace.jsonl.xz", "training.jsonl.xz"}
        required.update("checkpoint/" + name for name in CHECKPOINT_FILES)
        if not required.issubset(manifest["files"]):
            raise ValueError("Completed block is missing required evidence")
        actual_files = {str(path.relative_to(directory)) for path in directory.rglob("*")
                        if path.is_file() and path != manifest_path}
        if actual_files != set(manifest["files"]):
            raise ValueError("Completed block file inventory differs")
        for name, expected in manifest["files"].items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or digest(directory / relative) != expected:
                raise ValueError(f"Completed block checksum differs for {name}")
        if json.loads((directory / "verification.json").read_text())["status"] != "passed":
            raise ValueError("Completed block has no passing independent verification")
        if checkpoint_hashes(directory / "checkpoint") != manifest["checkpoint"]:
            raise ValueError("Completed block checkpoint differs")
        if attempt.completion_sha256 and digest(manifest_path) != attempt.completion_sha256:
            raise ValueError("Completed block publication manifest changed")
        return directory / "checkpoint"

    def recover(self):
        current = checkpoint_hashes(self.initial_checkpoint)
        if current != self.spec.initial_checkpoint:
            raise ValueError("Starting checkpoint changed")
        for index, attempt in enumerate(self.attempts):
            if (attempt.index != index or attempt.seed != self.spec.base_seed + index * 100000 or
                    not 30 <= attempt.reserved_seconds <= self.spec.block_seconds or
                    attempt.initial_checkpoint != current or attempt.state not in ("running", "completed", "failed")):
                raise ValueError("Ledger attempt order, seed range, budget or checkpoint chain differs")
            directory = self.bundle_path(attempt)
            if attempt.state in ("running", "failed"):
                if (directory / "completion.json").exists():
                    self.checkpoint = self.validate_bundle(directory, attempt)
                    attempt.state = "completed"
                    attempt.error = None
                    attempt.completion_sha256 = digest(directory / "completion.json")
                elif attempt.state == "running":
                    attempt.state = "failed"
                    attempt.error = "Abandoned attempt. Its full reserved time remains charged."
                    print(attempt.error, flush=True)
                self.save_ledger()
            if attempt.state == "completed":
                self.checkpoint = self.validate_bundle(directory, attempt)
                current = checkpoint_hashes(self.checkpoint)
        if self.charged_seconds > self.spec.total_seconds:
            raise ValueError("Ledger reserved more than the run budget")

    def reserve(self):
        remaining = self.spec.total_seconds - self.charged_seconds
        if remaining < 30:
            return None
        index = len(self.attempts)
        attempt = Attempt(index, "running", min(remaining, self.spec.block_seconds),
                          self.spec.base_seed + index * 100000, checkpoint_hashes(self.checkpoint))
        self.attempts.append(attempt)
        self.save_ledger()
        return attempt

    def archive(self, output, attempt, initial_checkpoint):
        report = self.verifier(output, initial_checkpoint)
        summary = json.loads((output / "summary.json").read_text())
        if any(not 0 <= episode["episode_index"] < 100000 for episode in summary["episodes"]):
            raise ValueError("A block exhausted its unique seed range")
        write_json(output / "verification.json", report)
        staging = self.local / f"archive-{attempt.index:04d}"
        staging.mkdir()
        for name in ("config.json", "summary.json", "deadline-result.json", "verification.json"):
            shutil.copyfile(output / name, staging / name)
        shutil.copytree(output / "checkpoint", staging / "checkpoint")
        for name in ("game-trace.jsonl", "training.jsonl"):
            with (output / name).open("rb") as source, lzma.open(staging / (name + ".xz"), "wb", preset=3) as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
        write_json(staging / "verification.json", self.verifier(staging, initial_checkpoint))
        manifest = {"schema": 1, "spec_sha256": self.spec_hash, "attempt_index": attempt.index,
                    "seed": attempt.seed, "reserved_seconds": attempt.reserved_seconds,
                    "initial_checkpoint": attempt.initial_checkpoint,
                    "checkpoint": checkpoint_hashes(staging / "checkpoint"),
                    "files": {str(path.relative_to(staging)): digest(path)
                              for path in sorted(staging.rglob("*")) if path.is_file()}}
        destination = self.bundle_path(attempt)
        if destination.exists():
            raise ValueError("Block archive already exists without a completed ledger entry")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staging, destination)
        for name, expected in manifest["files"].items():
            if digest(destination / name) != expected:
                raise ValueError("Archive copy failed its checksum")
        write_json(destination / "completion.json", manifest)
        checkpoint = self.validate_bundle(destination, attempt)
        attempt.completion_sha256 = digest(destination / "completion.json")
        attempt.state = "completed"
        self.save_ledger()
        self.checkpoint = checkpoint
        for name in ("game-trace.jsonl", "training.jsonl"):
            (output / name).unlink()
        shutil.rmtree(staging)
        return report

    def train(self, python, environments):
        while (attempt := self.reserve()) is not None:
            output = self.local / f"attempt-{attempt.index:04d}"
            initial_checkpoint = self.checkpoint
            print(f"Block {attempt.index + 1}. Reserved {self.charged_seconds}/{self.spec.total_seconds}s. "
                  f"Seed {attempt.seed}. Warm start {initial_checkpoint}.", flush=True)
            archiving = False
            try:
                self.runner([python, "-m", "arc3.train_gameplay", "--output", output,
                             "--checkpoint", initial_checkpoint, "--mlx-python", python,
                             "--environments-dir", environments, "--max-seconds", attempt.reserved_seconds,
                             "--seed", attempt.seed], timeout=attempt.reserved_seconds + 20,
                            log=self.local / f"attempt-{attempt.index:04d}.log", status=output / "status.json")
                archiving = True
                report = self.archive(output, attempt, initial_checkpoint)
            except BaseException as error:
                if attempt.state != "completed":
                    attempt.state = "failed"
                    attempt.error = (f"Block stopped. Reserved budget remains charged. "
                                     f"No verified learning signal or optimizer update was published. "
                                     f"{type(error).__name__}: {error}")
                    self.save_ledger()
                    print(attempt.error, flush=True)
                if (archiving and attempt.state == "failed" and isinstance(error, AssertionError) and
                        error.args == ("No optimizer work happened",)):
                    print("Continuing with the same checkpoint after a block without an optimizer update.", flush=True)
                    continue
                raise
            print(f"Verified checkpoint {attempt.index + 1}. Updates {report['optimizer_updates']}. "
                  f"Level rewards {report['counts'].get('level_completion_rewards', 0)}. "
                  f"Saved {self.bundle_path(attempt)}.", flush=True)
        if not any(attempt.state == "completed" for attempt in self.attempts):
            raise RuntimeError("Budget ended without a verified trained checkpoint")
        write_json(self.store / "latest.json", {"spec_sha256": self.spec_hash,
                                              "checkpoint": str(self.checkpoint),
                                              "checkpoint_sha256": checkpoint_hashes(self.checkpoint),
                                              "charged_seconds": self.charged_seconds})
        print(f"Training budget finished. Final verified checkpoint {self.checkpoint}", flush=True)
        return self.checkpoint


def build_spec(revision, hours, block_seconds, base_seed, checkpoint, manifest_path):
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("An immutable 40-character source revision is required")
    actual_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if actual_revision != revision:
        raise ValueError("Checked-out source differs from the pinned notebook revision")
    if not math.isfinite(hours) or not 1 / 120 <= hours <= 24 or not 30 <= block_seconds <= 600:
        raise ValueError("Use 30 seconds to 24 hours total, with 30 to 600 second blocks")
    manifest = json.loads(manifest_path.read_text())
    games = {row["game_id"]: row["source_sha256"] for row in manifest["games"]}
    baselines = ROOT / "arc3/public-baselines.json"
    if len(games) != 25 or manifest["baselines_sha256"] != digest(baselines):
        raise ValueError("Download all 25 pinned official public games before training")
    return RunSpec(revision, round(hours * 3600), block_seconds, base_seed, checkpoint_hashes(checkpoint),
                   {name: digest(ROOT / "arc3" / name) for name in RUNTIME_FILES}, games, digest(baselines))


def verify_evaluation(output, checkpoint):
    from .verify_score import inspect_run, verify_weights

    baselines = json.loads((ROOT / "arc3/public-baselines.json").read_text())["games"]
    weights = verify_weights(checkpoint)
    report = inspect_run(output, baselines)
    summary = json.loads((output / "summary.json").read_text())
    if (summary["subset_smoke"] or summary["policy"] != "ncm" or
            not summary["all_created_games_headless"] or summary["worker_stats"]["weight_sha256"] != weights):
        raise ValueError("Evaluation was not the frozen headless all-game NCM test")
    return {"status": "passed", "frozen_weight_sha256": weights, "runs": [report]}


def frozen_evaluation(run, python, environments, runner=run_process, verifier=verify_evaluation):
    output = run.store / "frozen-evaluation"
    marker = output / "completion.json"
    expected = checkpoint_hashes(run.checkpoint)
    if marker.exists():
        manifest = json.loads(marker.read_text())
        if manifest["checkpoint_sha256"] != expected or manifest["spec_sha256"] != run.spec_hash:
            raise ValueError("Frozen evaluation belongs to a different checkpoint")
        for name, checksum in manifest["files"].items():
            if digest(output / name) != checksum:
                raise ValueError("Frozen evaluation evidence changed")
        verifier(output, run.checkpoint)
    else:
        if output.exists():
            index = 0
            while (run.store / f"incomplete-frozen-evaluation-{index:04d}").exists():
                index += 1
            preserved = run.store / f"incomplete-frozen-evaluation-{index:04d}"
            output.rename(preserved)
            print(f"Preserved an interrupted evaluation at {preserved}. Reusing the trained checkpoint.", flush=True)
        index = 0
        while ((run.store / f"evaluation-attempt-{index:04d}.json").exists() or
               (run.local / f"evaluation-attempt-{index:04d}").exists()):
            index += 1
        attempt_file = run.store / f"evaluation-attempt-{index:04d}.json"
        local_output = run.local / f"evaluation-attempt-{index:04d}"
        attempt = {"state": "running", "checkpoint_sha256": expected, "local_output": str(local_output)}
        write_json(attempt_file, attempt)
        try:
            runner([python, "-m", "arc3.run_agent", "--policy", "ncm", "--output", local_output,
                    "--checkpoint", run.checkpoint, "--mlx-python", python, "--environments-dir", environments],
                   timeout=25 * 12 * 3600 + 60)
            verifier(local_output, run.checkpoint)
            with (local_output / "trace.jsonl").open("rb") as source, lzma.open(
                    local_output / "trace.jsonl.xz", "wb", preset=3) as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            (local_output / "trace.jsonl").unlink()
            write_json(local_output / "verification.json", verifier(local_output, run.checkpoint))
            shutil.copytree(local_output, output)
            for path in local_output.rglob("*"):
                if path.is_file() and digest(path) != digest(output / path.relative_to(local_output)):
                    raise ValueError("Frozen evaluation archive copy failed its checksum")
            write_json(output / "verification.json", verifier(output, run.checkpoint))
            write_json(marker, {"spec_sha256": run.spec_hash, "checkpoint_sha256": expected,
                                "files": {str(path.relative_to(output)): digest(path)
                                          for path in sorted(output.rglob("*")) if path.is_file()}})
            attempt["state"] = "completed"
            write_json(attempt_file, attempt)
        except BaseException as error:
            attempt.update(state="failed", error=f"{type(error).__name__}: {error}")
            write_json(attempt_file, attempt)
            raise
    summary = json.loads((output / "summary.json").read_text())
    ls20 = next(row for row in summary["episodes"] if row["game_id"].startswith("ls20-"))
    result = {"all_25_public_games": summary["all_games"],
              "training_game_ls20": {key: ls20[key] for key in ("game_id", "completed_levels", "score", "sdk_actions")},
              "other_24_public_games": summary["excluding_training_game"],
              "scope": "Public development games. This is not a Kaggle competition score."}
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--hours", type=float, default=4)
    parser.add_argument("--block-seconds", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "arc3/results/gameplay-training/checkpoint")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/arcagi3/evaluation_public/manifest.json")
    parser.add_argument("--environments-dir", type=Path, default=ROOT / "data/arcagi3/evaluation_public/environment_files")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    spec = build_spec(args.revision, args.hours, args.block_seconds, args.seed, args.checkpoint, args.manifest)
    run = TrainingRun(args.store, args.local, spec, args.checkpoint)
    if args.evaluate:
        if not run.can_evaluate:
            raise RuntimeError("Finish the reserved training budget before the frozen evaluation")
        frozen_evaluation(run, Path(sys.executable), args.environments_dir)
    else:
        run.train(Path(sys.executable), args.environments_dir)


if __name__ == "__main__":
    main()
