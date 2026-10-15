#!/usr/bin/env python3
"""Run a task adapter with shared macOS resource controls and a bounded invocation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from ncm_resource import GIB, PACING_METHODS, ResourcePolicy
from puzzle_runtime import NumericalPolicy, run_task


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--task", required=True, help="Python module exporting build_task(argv).")
    result.add_argument("--device", required=True, choices=("cpu", "mps", "mlx"))
    result.add_argument("--run", required=True, type=Path)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--seconds", type=float, default=600)
    result.add_argument("--gpu-duty-cycle", type=float, default=0.8)
    result.add_argument("--memory-limit-gib", type=float, default=8)
    result.add_argument("--rss-limit-gib", type=float, default=8)
    result.add_argument("--min-available-gib", type=float, default=6)
    result.add_argument("--pacing-method", choices=PACING_METHODS, default="idle-credit")
    result.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    result.add_argument("--attention", choices=("reference", "bf16-math"), default="reference")
    result.add_argument("--checkpoint-copies", choices=("blocking", "batched"), default=None)
    result.add_argument("--frozen-pooling", choices=("reference", "metal"), default=None)
    result.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("arguments", nargs=argparse.REMAINDER, help="Task arguments after --.")
    return result


def _configuration(args):
    import math

    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 600:
        raise ValueError("--seconds must be positive and no more than 600.")
    sizes = (args.memory_limit_gib, args.rss_limit_gib, args.min_available_gib)
    if any(not math.isfinite(value) for value in sizes):
        raise ValueError("Memory sizes must be finite.")
    policy = ResourcePolicy(gpu_duty_cycle=args.gpu_duty_cycle,
                            memory_limit_bytes=int(args.memory_limit_gib * GIB),
                            rss_limit_bytes=int(args.rss_limit_gib * GIB),
                            min_available_bytes=int(args.min_available_gib * GIB),
                            pacing_method=args.pacing_method)
    if args.device == "mlx" and args.checkpoint_copies is not None:
        raise ValueError("--checkpoint-copies applies only to Torch tasks.")
    numerical = NumericalPolicy(precision=args.precision, attention=args.attention,
                                checkpoint_copies=args.checkpoint_copies or "batched",
                                frozen_pooling=args.frozen_pooling or ("reference" if args.device == "mlx" else "metal"))
    return policy, numerical


def _supervise(command, seconds):
    started = time.monotonic()
    soft_deadline = started + seconds * 0.9
    hard_deadline = started + seconds * (590 / 600)
    process = None
    requested = None
    previous = {}
    reason = "completed"

    def request_stop(signum, frame):
        nonlocal requested
        requested = requested or signum

    def send(signum):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, request_stop)
        process = subprocess.Popen(command, cwd=Path(__file__).resolve().parent,
                                   start_new_session=True)
        while process.poll() is None and not requested and time.monotonic() < soft_deadline:
            try:
                process.wait(timeout=min(0.2, max(0.001, soft_deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                pass
        if process.poll() is None:
            reason = "signal" if requested else "deadline"
            send(signal.SIGTERM)
            try:
                process.wait(timeout=max(0, hard_deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                reason = "killed"
    finally:
        if process is not None and process.poll() is None:
            send(signal.SIGKILL)
            process.wait()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return process.returncode, reason


def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(raw)
    try:
        policy, numerical = _configuration(args)
        if args.worker:
            arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
            report = run_task(args.task, arguments, run_directory=args.run, device=args.device,
                              resume=args.resume, policy=policy, numerical=numerical,
                              seconds=args.seconds * 0.9)
            print(json.dumps(report, sort_keys=True, allow_nan=False))
            return report["returncode"]
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", *raw]
        code, reason = _supervise(command, args.seconds)
        if reason == "killed":
            print("The worker exceeded its stop deadline and was killed. Only its last completed atomic checkpoint is resumable.", file=sys.stderr)
        return code if code >= 0 else 128 - code
    except Exception as error:
        print(f"{type(error).__name__}. {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
