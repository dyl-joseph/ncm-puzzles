#!/usr/bin/env python3
"""Run the unchanged NCM trainer or evaluator with bounded resource use."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from dataclasses import asdict
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
from unittest.mock import patch

from ncm_attention import ATTENTION_KERNELS, select_generator_attention
from ncm_area_pool import POOLING_KERNELS
from ncm_resource import PACING_METHODS
from ncm_checkpoint import CHECKPOINT_COPY_METHODS


ROOT = Path(__file__).resolve().parent
RUNTIME_FILES = ("resource_runner.py", "ncm_resource.py", "ncm_fast_targets.py", "ncm_attention.py", "ncm_area_pool.py",
                 "ncm_checkpoint.py")


def execution_descriptor(policy=None, *, fast_targets: bool = True,
                         generator_precision: str = "bf16", attention_kernel: str | None = None,
                         frozen_pooling: str = "metal", checkpoint_copies: str = "batched") -> dict:
    from ncm_resource import ResourcePolicy

    if generator_precision not in ("bf16", "fp32", "not_applicable"):
        raise ValueError("Generator precision must be bf16, fp32, or not_applicable.")
    if attention_kernel is not None and attention_kernel not in ATTENTION_KERNELS:
        raise ValueError("Attention kernel must be reference or bf16-math.")
    if frozen_pooling not in POOLING_KERNELS:
        raise ValueError("Frozen pooling kernel must be reference or metal.")
    if checkpoint_copies not in CHECKPOINT_COPY_METHODS:
        raise ValueError("Checkpoint copies must be blocking or batched.")
    if generator_precision == "not_applicable":
        attention_kernel = "not_applicable"
    elif attention_kernel is None:
        attention_kernel = {"bf16": "bf16-math", "fp32": "reference"}[generator_precision]
    if attention_kernel == "bf16-math" and generator_precision != "bf16":
        raise ValueError("The bf16-math attention kernel requires bf16 generator precision.")
    policy = ResourcePolicy() if policy is None else policy
    return {
        "format": "ncm-resource-execution-v1",
        "policy": asdict(policy),
        "fast_frozen_targets": fast_targets,
        "generator_precision": generator_precision,
        "attention_kernel": attention_kernel,
        "frozen_pooling": (frozen_pooling if fast_targets and generator_precision != "not_applicable"
                           else "not_applicable"),
        "checkpoint_copies": (checkpoint_copies if generator_precision != "not_applicable" else "not_applicable"),
        "runtime_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in RUNTIME_FILES
        },
    }


def argument_value(arguments: list[str], name: str, default=None):
    found = default
    for index, value in enumerate(arguments):
        if value == "--":
            break
        if value == name:
            if index + 1 == len(arguments):
                raise ValueError(f"Missing value for {name}.")
            found = arguments[index + 1]
        elif value.startswith(name + "="):
            found = value.partition("=")[2]
    return found


def validate_forwarded_arguments(arguments: list[str]):
    protected = ("--device", "--config", "--run", "--output")
    for value in arguments:
        if value == "--":
            break
        option = value.partition("=")[0]
        if (option.startswith("--") and option not in protected
                and any(name.startswith(option) for name in protected)):
            raise ValueError("Use full names for forwarded --device, --config, --run, and --output options.")


def verify_training_execution(arguments: list[str], execution: dict):
    from omegaconf import OmegaConf

    config = Path(argument_value(arguments, "--config", ROOT / "configs/latin9.yaml"))
    recipe = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    if recipe.get("local_choices", {}).get("execution") != execution:
        raise ValueError("Resource policy or runtime sources differ from the recorded training recipe. "
                         "Start a new continuation with continue_latin.py.")


def write_report(path: Path, report: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


@contextmanager
def guarded_training(core, controller, *, fast_targets: bool, generator_precision: str = "fp32",
                     attention_kernel: str = "reference", frozen_pooling: str = "reference",
                     checkpoint_copies: str = "blocking"):
    if generator_precision not in ("bf16", "fp32"):
        raise ValueError("Training generator precision must be bf16 or fp32.")
    if attention_kernel not in ATTENTION_KERNELS:
        raise ValueError("Attention kernel must be reference or bf16-math.")
    if frozen_pooling not in POOLING_KERNELS:
        raise ValueError("Frozen pooling kernel must be reference or metal.")
    if checkpoint_copies not in CHECKPOINT_COPY_METHODS:
        raise ValueError("Checkpoint copies must be blocking or batched.")
    if attention_kernel == "bf16-math" and generator_precision != "bf16":
        raise ValueError("The bf16-math attention kernel requires bf16 generator precision.")
    original_optimizer = core.make_optimizer
    original_generator_loss = core.generator_loss
    original_tokenizer_loss = core.tokenizer_loss
    original_generate_batch = core.generate_batch
    if fast_targets:
        from ncm_fast_targets import fast_generator_loss
        generator_loss = (partial(fast_generator_loss, pooling_kernel="metal")
                          if frozen_pooling == "metal" else fast_generator_loss)
    else:
        generator_loss = original_generator_loss

    def make_optimizer(*args, **kwargs):
        optimizer = original_optimizer(*args, **kwargs)
        original_step = optimizer.step

        def step(*step_args, **step_kwargs):
            result = original_step(*step_args, **step_kwargs)
            reason = controller.finish_unit()
            if reason:
                os.kill(os.getpid(), signal.SIGTERM)
            return result

        optimizer.step = step
        return optimizer

    def guarded_generator_loss(*args, **kwargs):
        controller.begin_unit()
        return generator_loss(*args, **kwargs)

    def guarded_tokenizer_loss(*args, **kwargs):
        controller.begin_unit()
        return original_tokenizer_loss(*args, **kwargs)

    def guarded_generate_batch(*args, **kwargs):
        controller.begin_unit()
        predictions = original_generate_batch(*args, **kwargs)
        reason = controller.finish_unit()
        if reason:
            raise RuntimeError(f"Resource guard stopped validation. {reason}")
        return predictions

    with ExitStack() as stack:
        if checkpoint_copies == "batched":
            from ncm_checkpoint import install_batched_cpu_tree
            stack.enter_context(install_batched_cpu_tree(core))
        stack.enter_context(patch.object(core, "make_optimizer", make_optimizer))
        stack.enter_context(patch.object(core, "generator_loss", guarded_generator_loss))
        stack.enter_context(patch.object(core, "tokenizer_loss", guarded_tokenizer_loss))
        stack.enter_context(patch.object(core, "generate_batch", guarded_generate_batch))
        if generator_precision == "bf16":
            import torch
            original_generator = core.make_generator

            def make_generator(*args, **kwargs):
                model = original_generator(*args, **kwargs)
                original_forward = model.forward

                def forward(*forward_args, **forward_kwargs):
                    with select_generator_attention(attention_kernel):
                        with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                            logits = original_forward(*forward_args, **forward_kwargs)
                    return logits.float()

                stack.enter_context(patch.object(model, "forward", forward))
                return model

            stack.enter_context(patch.object(core, "make_generator", make_generator))
        yield


@contextmanager
def guarded_evaluation(core, controller):
    original_generate_batch = core.generate_batch

    def generate_batch(*args, **kwargs):
        controller.begin_unit()
        predictions = original_generate_batch(*args, **kwargs)
        reason = controller.finish_unit()
        if reason:
            raise RuntimeError(f"Resource guard stopped evaluation. {reason}")
        return predictions

    with patch.object(core, "generate_batch", generate_batch):
        yield


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate"))
    parser.add_argument("--gpu-duty-cycle", type=float, default=0.8)
    parser.add_argument("--memory-gb", type=float, default=8)
    parser.add_argument("--pacing-method", choices=PACING_METHODS, default="idle-credit",
                        help="Count CPU telemetry after GPU synchronization toward required idle time.")
    parser.add_argument("--checkpoint-copies", choices=CHECKPOINT_COPY_METHODS, default="batched",
                        help="Complete checkpoint transfers together before saving their exact CPU values.")
    parser.add_argument("--no-fast-targets", action="store_true", help="Reference path for verification only.")
    parser.add_argument("--generator-precision", choices=("bf16", "fp32"), default="bf16",
                        help="Training generator forward precision. Frozen preprocessing and evaluation stay FP32.")
    parser.add_argument("--attention-kernel", choices=ATTENTION_KERNELS,
                        help="Training attention kernel. Defaults to bf16-math for BF16, reference for FP32.")
    parser.add_argument("--frozen-pooling", choices=POOLING_KERNELS, default="metal",
                        help="Frozen tokenizer area pooling kernel. Generator operations are unchanged.")
    parser.add_argument("--resource-report", type=Path)
    args, arguments = parser.parse_known_args(argv)
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    from ncm_resource import ResourcePolicy, make_mlx_controller, make_torch_controller

    try:
        validate_forwarded_arguments(arguments)
        if not math.isfinite(args.memory_gb) or not 0 < args.memory_gb <= 8:
            raise ValueError("Memory budget must be finite, above zero, and at most 8 GiB.")
        policy = ResourcePolicy(gpu_duty_cycle=args.gpu_duty_cycle,
                                memory_limit_bytes=int(args.memory_gb * 1024 ** 3),
                                rss_limit_bytes=int(args.memory_gb * 1024 ** 3),
                                pacing_method=args.pacing_method)
        execution = execution_descriptor(policy, fast_targets=not args.no_fast_targets,
                                         generator_precision=(args.generator_precision if args.mode == "train"
                                                              else "not_applicable"),
                                         attention_kernel=args.attention_kernel, frozen_pooling=args.frozen_pooling,
                                         checkpoint_copies=args.checkpoint_copies)
        if args.mode == "train":
            if arguments[:1] != ["generator"]:
                raise ValueError("Guarded training supports only the generator stage. "
                                 "Put generator before the trainer options.")
            verify_training_execution(arguments, execution)
            destination = argument_value(arguments, "--run")
        else:
            destination = argument_value(arguments, "--output")
        if destination is None and args.resource_report is None:
            raise ValueError("Supply --run or --output so the resource report has a destination.")
        report_path = args.resource_report or Path(destination) / f"resources-{args.mode}.json"
    except (ValueError, OverflowError) as error:
        parser.error(str(error))

    controller = None
    report = {"execution": execution, "mode": args.mode, "returncode": 1}
    try:
        device = argument_value(arguments, "--device", "mps" if args.mode == "train" else "mlx")
        if args.mode == "train" and device != "mps":
            raise ValueError("Guarded training requires --device mps.")
        if args.mode == "evaluate" and device not in ("mps", "mlx"):
            raise ValueError("This guarded runner supports Apple MPS training and MLX/MPS evaluation.")
        os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.1")
        if device == "mlx":
            import mlx.core as mx
            controller = make_mlx_controller(mx, policy)
        else:
            import torch
            controller = make_torch_controller(torch, policy)
        allocation_label = "MLX allocator guideline" if device == "mlx" else "MPS allocator limit"
        print(f"Resource policy targets {policy.gpu_duty_cycle:.0%} average GPU duty with brief bursts. "
              f"{allocation_label} is {args.memory_gb:g} GiB. "
              f"Process resident memory guard is {args.memory_gb:g} GiB. "
              "New swap writes or memory pressure stop the run.", flush=True)
        if args.mode == "train":
            import train_latin as core
            print(f"Generator forward precision is {args.generator_precision} with "
                  f"the {execution['attention_kernel']} attention kernel. "
                  "Frozen tokenizer, logits, loss, parameters, gradients, and optimizer state stay FP32.", flush=True)
            with guarded_training(core, controller, fast_targets=not args.no_fast_targets,
                                  generator_precision=args.generator_precision,
                                  attention_kernel=execution["attention_kernel"], frozen_pooling=args.frozen_pooling,
                                  checkpoint_copies=args.checkpoint_copies):
                controller.begin_unit()
                report["returncode"] = core.main(arguments)
        else:
            import evaluate_latin as core
            with guarded_evaluation(core, controller):
                controller.begin_unit()
                report["returncode"] = core.main(arguments)
    except (Exception, KeyboardInterrupt) as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        report["returncode"] = 130 if isinstance(error, KeyboardInterrupt) else 1
        print(f"Resource runner failed. {error}", file=sys.stderr)
    finally:
        if controller is not None:
            report["resources"] = controller.close()
            if report["resources"].get("stop_reason") and report["returncode"] == 0:
                report["returncode"] = 1
        write_report(report_path, report)
    return report["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
