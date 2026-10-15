#!/usr/bin/env python3
"""Train a fresh Latin generator for at most ten minutes using the saved tokenizer."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from omegaconf import OmegaConf

from evaluate_latin import ROOT, verify_local_export
from latin_data import LatinCorpus
from ncm_attention import ATTENTION_KERNELS
from ncm_area_pool import POOLING_KERNELS
from ncm_resource import PACING_METHODS
from ncm_checkpoint import CHECKPOINT_COPY_METHODS
from resource_runner import execution_descriptor
from train_latin import atomic_json, stage_spec

SOURCE = ROOT / "runs/latin9-scratch/tokenizer/export"
CONFIG = ROOT / "configs/latin9-10min.yaml"
DATA = ROOT / "data/latin9"
TOKENIZER_SHA256 = "e7a37a8e46549632314d9031b6bef102521224f94da6b0263936d0391c071b47"


@contextmanager
def deadline_lock(run: Path):
    if ROOT / "runs" not in run.parents or run == SOURCE.parents[1] or SOURCE.parents[1] in run.parents:
        raise ValueError("Choose a new directory under this project's runs directory.")
    run.mkdir(parents=True, exist_ok=True)
    with (run / ".deadline.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another ten-minute wrapper owns this run.") from error
        yield


def prepare_run(run: Path, resume: bool):
    provenance = verify_local_export(SOURCE, stage="tokenizer")["checkpoint"]
    recipe = OmegaConf.load(CONFIG)
    if (provenance["training"]["tokenizer_updates"] != 213
            or provenance["files"]["tokenizer.pt"]["sha256"] != TOKENIZER_SHA256):
        raise ValueError("The original 213-update Latin tokenizer changed.")
    if provenance["training"]["stage_spec"] != asdict(stage_spec(recipe, "tokenizer")):
        raise ValueError("The tokenizer schedule must match its original training recipe.")
    if provenance["data"]["manifest_sha256"] != LatinCorpus.open(DATA).manifest_sha256:
        raise ValueError("The tokenizer and current corpus have different manifests.")
    if (run / "generator/state.pt").is_file() != resume:
        raise ValueError("Use --resume only when this run has a generator checkpoint.")
    destination = run / "tokenizer/export"
    if destination.exists():
        if verify_local_export(destination, stage="tokenizer")["checkpoint"] != provenance:
            raise ValueError("This run contains a different tokenizer export.")
    elif resume:
        raise ValueError("The resumed run is missing its copied tokenizer export.")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".tokenizer-", dir=run) as directory:
            copied = Path(directory) / "export"
            shutil.copytree(SOURCE, copied)
            if verify_local_export(copied, stage="tokenizer")["checkpoint"] != provenance:
                raise ValueError("The source tokenizer changed during copying.")
            copied.rename(destination)


def run_child(command: list[str], started: float, seconds: float) -> tuple[int, str]:
    process = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
    reason = "completed"
    try:
        try:
            process.wait(timeout=max(0, started + seconds * 0.9 - time.monotonic()))
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            reason = "deadline" if isinstance(error, subprocess.TimeoutExpired) else "interrupted"
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=max(0, started + seconds * (590 / 600) - time.monotonic()))
            except subprocess.TimeoutExpired:
                reason = "killed"
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    return process.returncode, reason


def prepare_resource_recipe(run: Path, resume: bool, execution: dict) -> Path:
    recipe = OmegaConf.to_container(OmegaConf.load(CONFIG), resolve=True)
    recipe["generator"]["microbatch_size"] = 16
    recipe["local_choices"]["generator_batch"] = "global_32_with_physical_microbatch_16_and_two_accumulation_steps"
    recipe["local_choices"]["platform"] = (
        "mps_float32_parameters_generator_bfloat16_autocast" if execution["generator_precision"] == "bf16"
        else "float32_eager_on_all_backends_bf16_compile_not_enabled")
    recipe["local_choices"]["execution"] = execution
    config = run / "resource-recipe.yaml"
    if resume:
        if (not config.is_file()
                or OmegaConf.to_container(OmegaConf.load(config), resolve=True) != recipe):
            raise ValueError("The resource recipe changed or this is an older unguarded run. "
                             "Use continue_latin.py to create a new continuation.")
    else:
        OmegaConf.save(OmegaConf.create(recipe), config)
    return config


def verify_ready(run: Path, launched_at: datetime, config: Path = CONFIG) -> int:
    status = json.loads((run / "generator/status.json").read_text())
    if (status["status"] not in ("stopped", "complete")
            or datetime.fromisoformat(status["finished_at"]) < launched_at):
        raise ValueError("This invocation did not save a successful generator status.")
    checkpoint = verify_local_export(run / "generator/export")["checkpoint"]
    updates = checkpoint["training"]["generator_updates"]
    if (updates < 1 or updates != status["last_saved_updates"] or updates != status["completed_updates_in_process"]
            or checkpoint["training"]["tokenizer_updates"] != 213
            or checkpoint["files"]["tokenizer.pt"]["sha256"] != TOKENIZER_SHA256):
        raise ValueError("The generator export does not match its saved training boundary.")
    exported = json.loads((run / "generator/export/run-config.json").read_text())
    if exported["recipe"] != OmegaConf.to_container(OmegaConf.load(config), resolve=True):
        raise ValueError("The generator export does not use the ten-minute recipe.")
    return updates


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/latin9-10min-resource")
    parser.add_argument("--stop-after", type=int, default=400, help="Maximum new generator updates, capped at 400.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seconds", type=float, default=600, help="Wall-clock budget, greater than zero and at most 600.")
    parser.add_argument("--gpu-duty-cycle", type=float, default=0.8)
    parser.add_argument("--memory-gb", type=float, default=8)
    parser.add_argument("--pacing-method", choices=PACING_METHODS, default="idle-credit",
                        help="Count CPU telemetry after GPU synchronization toward required idle time.")
    parser.add_argument("--checkpoint-copies", choices=CHECKPOINT_COPY_METHODS, default="batched",
                        help="Complete checkpoint transfers together before saving their exact CPU values.")
    parser.add_argument("--generator-precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--attention-kernel", choices=ATTENTION_KERNELS,
                        help="Defaults to bf16-math for BF16 generator precision, reference for FP32.")
    parser.add_argument("--frozen-pooling", choices=POOLING_KERNELS, default="metal",
                        help="Frozen tokenizer area pooling kernel. Generator operations are unchanged.")
    args = parser.parse_args(argv)
    if (args.stop_after < 1 or not math.isfinite(args.seconds) or not 0 < args.seconds <= 600
            or not math.isfinite(args.memory_gb) or not 0 < args.memory_gb <= 8):
        parser.error("Use a positive --stop-after, 0 < --seconds <= 600, and 0 < --memory-gb <= 8.")
    started, launched_at = time.monotonic(), datetime.now(timezone.utc)
    run = args.run.resolve()
    try:
        from ncm_resource import ResourcePolicy
        policy = ResourcePolicy(gpu_duty_cycle=args.gpu_duty_cycle,
                                memory_limit_bytes=int(args.memory_gb * 1024 ** 3),
                                rss_limit_bytes=int(args.memory_gb * 1024 ** 3),
                                pacing_method=args.pacing_method)
        execution = execution_descriptor(policy, generator_precision=args.generator_precision,
                                         attention_kernel=args.attention_kernel, frozen_pooling=args.frozen_pooling,
                                         checkpoint_copies=args.checkpoint_copies)
        with deadline_lock(run):
            result = {"ready": False, "source_tokenizer_sha256": TOKENIZER_SHA256}
            try:
                prepare_run(run, args.resume)
                config = prepare_resource_recipe(run, args.resume, execution)
                if time.monotonic() >= started + args.seconds * 0.9:
                    raise TimeoutError("The time budget expired during preparation. Training was not started.")
                report_path = run / "resources-train.json"
                command = [sys.executable, str(ROOT / "resource_runner.py"), "train",
                           "--gpu-duty-cycle", str(args.gpu_duty_cycle), "--memory-gb", str(args.memory_gb),
                           "--pacing-method", args.pacing_method,
                           "--checkpoint-copies", args.checkpoint_copies,
                           "--generator-precision", args.generator_precision,
                           "--attention-kernel", execution["attention_kernel"],
                           "--frozen-pooling", args.frozen_pooling,
                           "--resource-report", str(report_path), "generator", "--config", str(config),
                           "--data", str(DATA), "--run", str(run), "--device", "mps",
                           "--allow-partial-tokenizer", "--stop-after", str(min(args.stop_after, 400))]
                if args.resume:
                    command.append("--resume")
                print(f"Training the fresh generator. Graceful stop at {args.seconds * 0.9:g} seconds; "
                      f"hard stop before {args.seconds:g} seconds.", flush=True)
                code, reason = run_child(command, started, args.seconds)
                result.update(returncode=code, stop_reason=reason, resource_report=str(report_path))
                resource_reason = None
                if report_path.is_file():
                    resource_reason = json.loads(report_path.read_text()).get("resources", {}).get("stop_reason")
                    if resource_reason:
                        result.update(stop_reason="resource_guard", resource_stop_reason=resource_reason)
                if reason == "killed" or (code != 0 and not resource_reason):
                    raise RuntimeError("The trainer did not exit cleanly. Resume its last saved state; do not evaluate an older export.")
                result.update(completed_updates=verify_ready(run, launched_at, config), ready=True,
                              export=str(run / "generator/export"))
            except (Exception, KeyboardInterrupt) as error:
                result["error"] = str(error) or "Interrupted before training completed."
            result["elapsed_seconds"] = time.monotonic() - started
            atomic_json(run / "deadline-result.json", result)
            print(json.dumps(result), flush=True)
            return 0 if result["ready"] and result.get("stop_reason") != "resource_guard" else 1
    except (Exception, KeyboardInterrupt) as error:
        print(f"Ten-minute run failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
