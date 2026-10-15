#!/usr/bin/env python3
"""Continue the saved Latin generator for up to ten minutes, then optionally evaluate it."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from omegaconf import OmegaConf
import torch

from evaluate_latin import ROOT, sha256_file, verify_local_export
from ncm_attention import ATTENTION_KERNELS
from ncm_area_pool import POOLING_KERNELS
from ncm_resource import PACING_METHODS
from ncm_checkpoint import CHECKPOINT_COPY_METHODS
from run_latin_10min import run_child
from resource_runner import execution_descriptor
from train_latin import (atomic_json, atomic_torch_save, canonical_hash, learning_rate,
                         run_context, stage_spec)

DATA = ROOT / "data/latin9"
DEFAULT_ADDITIONAL_UPDATES = 600
MAX_UPDATES_PER_INVOCATION = 600


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def tensors_equal(first: dict, second: dict) -> bool:
    return (first.keys() == second.keys()
            and all(torch.equal(first[name], second[name]) for name in first))


@contextmanager
def continuation_lock(source: Path, run: Path):
    if (ROOT / "runs" not in source.parents or ROOT / "runs" not in run.parents
            or run == source or source in run.parents or run in source.parents):
        raise ValueError("Choose separate source and destination directories under this project's runs directory.")
    run.parent.mkdir(parents=True, exist_ok=True)
    with (run.parent / f".{run.name}.continuation.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another continuation wrapper owns this destination.") from error
        yield


def validate_state(state: dict, context: dict, snapshot: dict, corpus) -> int:
    recipe = OmegaConf.create(context["recipe"])
    expected = {"run_config_sha256": canonical_hash(context),
                "stage": asdict(stage_spec(recipe, "generator")), "tokenizer_snapshot": snapshot}
    completed = state.get("completed_updates")
    if (state.get("format") != "ncm-latin-training-state-v1" or state.get("immutable") != expected
            or type(completed) is not int or not 1 <= completed <= expected["stage"]["updates"]
            or state.get("fine_drop_step") is not None or not state.get("model")
            or not state.get("optimizer", {}).get("state")):
        raise ValueError("Generator state does not match its saved run, schedule, or tokenizer.")
    stream = corpus.training_stream(seed=recipe.seed, clue_count=recipe.clue_count, augment=recipe.augment)
    stream.load_state_dict(state["sampler"])
    rng = state["rng"]
    random.Random().setstate(rng["python"])
    numpy_state = rng["numpy"]
    np.random.RandomState().set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                                      numpy_state[2], numpy_state[3], numpy_state[4]))
    torch.Generator().set_state(rng["torch_cpu"])
    if (not isinstance(rng["torch_mps"], torch.Tensor) or rng["torch_mps"].dtype != torch.uint8
            or not rng["torch_mps"].numel()):
        raise ValueError("The saved MPS RNG state is missing or malformed.")
    groups = state["optimizer"]["param_groups"]
    lr = learning_rate(stage_spec(recipe, "generator"), completed)
    if not groups or any(not math.isfinite(group["lr"]) or group["lr"] != lr for group in groups):
        raise ValueError("Saved optimizer learning rates do not match the last completed update.")
    return completed


def prepare_continuation(source: Path, run: Path, resume: bool, execution: dict | None = None,
                         microbatch_size: int | None = None,
                         additional_updates: int | None = None) -> tuple[dict, int]:
    if additional_updates is not None and (type(additional_updates) is not int or additional_updates < 1):
        raise ValueError("Additional updates must be a positive integer.")
    if resume:
        saved = read_json(run / "continuation.json")
        start, target = saved.get("start_update"), saved.get("target_update")
        if type(start) is not int or type(target) is not int or start < 0 or target <= start:
            raise ValueError("The saved continuation update range is invalid.")
        recorded_updates = target - start
        if additional_updates is not None and additional_updates != recorded_updates:
            raise ValueError(f"The saved continuation adds {recorded_updates} updates. "
                             "Use a new destination to change its target.")
        additional_updates = recorded_updates
    elif additional_updates is None:
        additional_updates = DEFAULT_ADDITIONAL_UPDATES
    execution = execution_descriptor() if execution is None else execution
    with (source / ".train.lock").open("r") as source_lock:
        try:
            fcntl.flock(source_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Stop the source trainer before preparing a continuation.") from error
        export = source / "generator/export"
        provenance = verify_local_export(export)["checkpoint"]
        old_context = read_json(source / "run.json")
        if old_context != read_json(export / "run-config.json"):
            raise ValueError("The source export and run configuration disagree.")
        tracked = ["run.json", "generator/state.pt", "generator/tokenizer-snapshot.pt",
                   "generator/tokenizer-snapshot.json", "generator/export/provenance.json",
                   *[f"generator/export/{name}" for name in provenance["files"]]]
        source_hashes = {name: sha256_file(source / name) for name in tracked}
        state = torch.load(source / "generator/state.pt", map_location="cpu", weights_only=True)
        snapshot = read_json(source / "generator/tokenizer-snapshot.json")
        if (snapshot != provenance["training"]["tokenizer_snapshot"]
                or snapshot["sha256"] != source_hashes["generator/tokenizer-snapshot.pt"]
                or snapshot["sha256"] != provenance["files"]["tokenizer.pt"]["sha256"]):
            raise ValueError("The source frozen tokenizer and export disagree.")
        exported_model = torch.load(export / "generator.pt", map_location="cpu", weights_only=True)["model"]
        if not tensors_equal(state["model"], exported_model):
            raise ValueError("The source generator state and exported weights disagree.")
        del exported_model
        old_recipe = deepcopy(old_context["recipe"])
        recipe = deepcopy(old_recipe)
        completed = state["completed_updates"]
        if completed != provenance["training"]["generator_updates"]:
            raise ValueError("The source state and exported update counts disagree.")
        last_lr = state["optimizer"]["param_groups"][0]["lr"]
        if not math.isfinite(last_lr) or last_lr <= 0:
            raise ValueError("Continuation requires a finite positive saved learning rate.")
        target = completed + additional_updates
        global_batch = int(recipe["generator"]["batch_size"])
        microbatch_size = (int(recipe["generator"]["microbatch_size"])
                           if microbatch_size is None else microbatch_size)
        if (type(microbatch_size) is not int or microbatch_size < 1
                or microbatch_size > global_batch or global_batch % microbatch_size):
            raise ValueError("The physical microbatch must divide the preserved global batch.")
        lineage = {"format": "ncm-latin-continuation-v1", "source_run": str(source),
                   "source_files_sha256": source_hashes,
                   "source_run_config_sha256": canonical_hash(old_context),
                   "source_stage_spec": deepcopy(state["immutable"]["stage"]),
                   "start_update": completed, "target_update": target,
                   "learning_rate": last_lr, "schedule": "constant_saved_learning_rate",
                   "global_batch_size": global_batch, "microbatch_size": microbatch_size,
                   "execution": execution,
                   "helper_sha256": sha256_file(Path(__file__))}
        recipe["generator"].update(updates=target, peak_lr=last_lr, floor_lr=last_lr, warmup_updates=0,
                                   microbatch_size=microbatch_size)
        recipe["validation_every"] = target + 1
        recipe["local_choices"].update(
            initialization="generator_optimizer_rng_sampler_from_source_checkpoint",
            training_budget=f"at_most_{additional_updates}_additional_updates_with_600_second_wrapper_deadline",
            schedule="constant_saved_learning_rate_for_continuation", warmup="no_additional_warmup")
        recipe["local_choices"]["generator_batch"] = (
            f"global_{global_batch}_preserved_with_physical_microbatch_{microbatch_size}_and_gradient_accumulation")
        recipe["local_choices"]["platform"] = (
            "mps_float32_parameters_generator_bfloat16_autocast" if execution["generator_precision"] == "bf16"
            else "float32_eager_on_all_backends_bf16_compile_not_enabled")
        recipe["local_choices"]["continuation"] = lineage
        recipe["local_choices"]["execution"] = execution
        if run.exists() != resume:
            raise ValueError("Choose a new destination, or use --resume for an existing continuation.")
        with tempfile.TemporaryDirectory(prefix=f".{run.name}-", dir=run.parent) as directory:
            staging = Path(directory)
            OmegaConf.save(OmegaConf.create(recipe), staging / "continuation.yaml")
            corpus, _, _, context = run_context(DATA, staging, "mps", staging / "continuation.yaml")
            if {key: value for key, value in context.items() if key != "recipe"} != {
                    key: value for key, value in old_context.items() if key != "recipe"}:
                raise ValueError("The current trainer, data, architecture, dependencies, or environment changed.")
            validate_state(state, old_context, snapshot, corpus)
            if resume:
                if (read_json(run / "continuation.json") != lineage
                        or read_json(run / "parent-run.json") != old_context
                        or read_json(run / "run.json") != context
                        or OmegaConf.to_container(OmegaConf.load(run / "continuation.yaml"), resolve=True) != recipe
                        or read_json(run / "generator/tokenizer-snapshot.json") != snapshot
                        or sha256_file(run / "generator/tokenizer-snapshot.pt") != snapshot["sha256"]):
                    raise ValueError("The continuation lineage, recipe, or frozen tokenizer changed.")
                current = torch.load(run / "generator/state.pt", map_location="cpu", weights_only=True)
                current_completed = validate_state(current, context, snapshot, corpus)
                if not completed <= current_completed <= target:
                    raise ValueError("The continuation checkpoint is outside its recorded update range.")
            else:
                migrated = {**state, "immutable": {**state["immutable"],
                            "run_config_sha256": canonical_hash(context),
                            "stage": asdict(stage_spec(OmegaConf.create(recipe), "generator"))}}
                atomic_torch_save(staging / "generator/state.pt", migrated)
                for suffix in ("pt", "json"):
                    shutil.copyfile(source / f"generator/tokenizer-snapshot.{suffix}",
                                    staging / f"generator/tokenizer-snapshot.{suffix}")
                atomic_json(staging / "parent-run.json", old_context)
                atomic_json(staging / "continuation.json", lineage)
                current_completed = completed
            if any(sha256_file(source / name) != digest for name, digest in source_hashes.items()):
                raise ValueError("The source changed during continuation preparation.")
            if not resume:
                staging.rename(run)
        return lineage, current_completed


def verify_ready(run: Path, lineage: dict, initial: int, launched_at: datetime) -> int:
    status = read_json(run / "generator/status.json")
    checkpoint = verify_local_export(run / "generator/export")["checkpoint"]
    context = read_json(run / "run.json")
    state = torch.load(run / "generator/state.pt", map_location="cpu", weights_only=True)
    completed = state["completed_updates"]
    snapshot = read_json(run / "generator/tokenizer-snapshot.json")
    if (status["status"] not in ("stopped", "complete")
            or datetime.fromisoformat(status["finished_at"]) < launched_at
            or not initial < completed <= lineage["target_update"]
            or completed != status["last_saved_updates"]
            or completed != status["completed_updates_in_process"]
            or completed != checkpoint["training"]["generator_updates"]
            or checkpoint["training"]["tokenizer_snapshot"] != snapshot
            or snapshot["sha256"] != sha256_file(run / "generator/tokenizer-snapshot.pt")
            or checkpoint["run_config_sha256"] != canonical_hash(context)
            or state["immutable"] != {"run_config_sha256": canonical_hash(context),
                                      "stage": checkpoint["training"]["stage_spec"],
                                      "tokenizer_snapshot": snapshot}
            or context != read_json(run / "generator/export/run-config.json")
            or context["recipe"]["local_choices"]["continuation"] != lineage):
        raise ValueError("This invocation did not save matching generator state, status, and export.")
    exported = torch.load(run / "generator/export/generator.pt", map_location="cpu", weights_only=True)["model"]
    if not tensors_equal(state["model"], exported):
        raise ValueError("The saved generator state and exported weights disagree.")
    return completed


def evaluate(run: Path, count: int) -> Path:
    output = run / datetime.now(timezone.utc).strftime("evaluation-validation-%Y%m%dT%H%M%S.%fZ")
    execution = read_json(run / "run.json")["recipe"]["local_choices"]["execution"]
    policy = execution["policy"]
    command = [sys.executable, str(ROOT / "resource_runner.py"), "evaluate",
               "--gpu-duty-cycle", str(policy["gpu_duty_cycle"]),
               "--memory-gb", str(policy["memory_limit_bytes"] / 1024 ** 3),
               "--pacing-method", policy.get("pacing_method", "full-sleep"),
               "--local-checkpoint", str(run / "generator/export"),
               "--data", str(DATA), "--split", "validation", "--count", str(count), "--clue-count", "30",
               "--seed", "42", "--batch-size", "8", "--device", "mlx", "--output", str(output)]
    subprocess.run(command, cwd=ROOT, check=True)
    summary = read_json(output / "summary.json")
    if summary["status"] != "complete" or summary["completed"] != count:
        raise ValueError("Evaluation did not finish all requested puzzles.")
    metrics = summary["metrics"]
    print(f"Validation solved {metrics['solved']}/{count}. "
          f"Given digits preserved {metrics['given_cell_agreement']:.2%}. Summary {output / 'summary.json'}", flush=True)
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "runs/latin9-1000")
    parser.add_argument("--run", type=Path, default=ROOT / "runs/latin9-1600")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--additional-updates", type=int,
                        help="Fixed additional-update target. Defaults to 600 for a new run, or the saved target on resume.")
    parser.add_argument("--stop-after", type=int, default=MAX_UPDATES_PER_INVOCATION,
                        help="Maximum new updates, capped at 600 and the recorded target.")
    parser.add_argument("--seconds", type=float, default=600, help="Training budget, greater than zero and at most 600.")
    parser.add_argument("--evaluate", action="store_true", help="Run MLX validation after successful training, outside its deadline.")
    parser.add_argument("--eval-count", type=int, default=2000)
    parser.add_argument("--gpu-duty-cycle", type=float, default=0.8,
                        help="Average GPU duty target, greater than zero and at most 0.8.")
    parser.add_argument("--memory-gb", type=float, default=8,
                        help="Allocator and process resident-memory cap in GiB, at most 8.")
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
    if (args.stop_after < 1 or (args.additional_updates is not None and args.additional_updates < 1)
            or not math.isfinite(args.seconds) or not 0 < args.seconds <= 600
            or not 1 <= args.eval_count <= 2000 or not math.isfinite(args.memory_gb)
            or not 0 < args.memory_gb <= 8):
        parser.error("Use positive --additional-updates and --stop-after, 0 < --seconds <= 600, 1 <= --eval-count <= 2000, "
                     "and 0 < --memory-gb <= 8.")
    started, launched_at = time.monotonic(), datetime.now(timezone.utc)
    source, run = args.source.resolve(), args.run.resolve()
    result = {"ready": False, "source_run": str(source)}
    try:
        from ncm_resource import ResourcePolicy
        policy = ResourcePolicy(gpu_duty_cycle=args.gpu_duty_cycle,
                                memory_limit_bytes=int(args.memory_gb * 1024 ** 3),
                                rss_limit_bytes=int(args.memory_gb * 1024 ** 3),
                                pacing_method=args.pacing_method)
        execution = execution_descriptor(policy, generator_precision=args.generator_precision,
                                         attention_kernel=args.attention_kernel, frozen_pooling=args.frozen_pooling,
                                         checkpoint_copies=args.checkpoint_copies)
        with continuation_lock(source, run):
            lineage, initial = prepare_continuation(source, run, args.resume, execution, microbatch_size=16,
                                                   additional_updates=args.additional_updates)
            result.update(source_completed_updates=lineage["start_update"], initial_updates=initial,
                          target_updates=lineage["target_update"])
            try:
                if initial == lineage["target_update"]:
                    raise ValueError("This continuation already reached its target. Evaluate its export directly.")
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
                           "--resource-report", str(report_path), "generator", "--resume",
                           "--config", str(run / "continuation.yaml"), "--data", str(DATA), "--run", str(run),
                           "--device", "mps", "--allow-partial-tokenizer", "--stop-after",
                           str(min(args.stop_after, MAX_UPDATES_PER_INVOCATION, lineage["target_update"] - initial))]
                print(f"Continuing generator from update {initial} toward {lineage['target_update']}. "
                      f"Learning rate {lineage['learning_rate']:g}. "
                      f"Graceful stop at {args.seconds * 0.9:g} seconds; hard stop before {args.seconds:g} seconds.", flush=True)
                code, reason = run_child(command, started, args.seconds)
                result.update(returncode=code, stop_reason=reason)
                result["resource_report"] = str(report_path)
                resource_reason = None
                if report_path.is_file():
                    resource_reason = read_json(report_path).get("resources", {}).get("stop_reason")
                    if resource_reason:
                        result.update(stop_reason="resource_guard", resource_stop_reason=resource_reason)
                if reason == "killed" or (code != 0 and not resource_reason):
                    raise RuntimeError("The trainer did not exit cleanly. Resume its saved state before evaluating.")
                completed = verify_ready(run, lineage, initial, launched_at)
                result.update(ready=True, completed_updates=completed, new_updates=completed - initial,
                              continuation_updates=completed - lineage["start_update"], export=str(run / "generator/export"))
            except (Exception, KeyboardInterrupt) as error:
                result["error"] = str(error) or "Interrupted before training completed."
            result["training_elapsed_seconds"] = time.monotonic() - started
            atomic_json(run / "continuation-result.json", result)
            if result["ready"] and args.evaluate and result["stop_reason"] not in ("interrupted", "resource_guard"):
                try:
                    result["evaluation_output"] = str(evaluate(run, args.eval_count))
                except (Exception, KeyboardInterrupt) as error:
                    result["evaluation_error"] = str(error) or "Evaluation interrupted."
                atomic_json(run / "continuation-result.json", result)
            print(json.dumps(result), flush=True)
            return 0 if (result["ready"] and "evaluation_error" not in result
                         and result.get("stop_reason") != "resource_guard") else 1
    except (Exception, KeyboardInterrupt) as error:
        print(f"Continuation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
