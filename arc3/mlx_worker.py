#!/usr/bin/env python3
"""Frozen GPU-only NCM inference over a JSON-lines pipe."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import mlx.core as mx
if __package__:
    from .mlx_ncm import Generator, Tokenizer, validate_device
else:
    from mlx_ncm import Generator, Tokenizer, validate_device


def emit(value):
    print(json.dumps(value, separators=(",", ":")), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    if metadata["vocab_size"] != 160 or metadata["prefix_len"] != 1142:
        raise ValueError("Checkpoint metadata differs from the frozen LS20 adapter")
    mx.set_default_device(mx.gpu)
    validate_device()
    tokenizer = Tokenizer(metadata["vocab_size"])
    tokenizer.load_weights(str(checkpoint / "tokenizer.safetensors"))
    tokenizer.eval()
    generator = Generator(metadata["vocab_size"], metadata["prefix_len"], tokenizer)
    generator.load_weights(str(checkpoint / "generator.safetensors"))
    generator.eval()
    mx.eval(tokenizer.parameters(), generator.parameters())
    mx.synchronize(mx.gpu)
    stats = {"device": str(mx.default_device()), "mlx_version": importlib.metadata.version("mlx"),
             "duty_target": 0.8, "duty_is_soft_pacing": True,
             "active_seconds": 0.0, "idle_seconds": 0.0, "calls": 0,
             "torch_imported": "torch" in sys.modules, "cpu_neural_inference": False,
             "weight_sha256": {name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
                               for name in ("tokenizer.safetensors", "generator.safetensors")}}
    emit({"type": "ready", **stats})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("type") == "close":
                break
            prefix = request["prefix"]
            if (len(prefix) != metadata["prefix_len"] or
                    any(type(token) is not int or not 0 <= token < metadata["vocab_size"] for token in prefix)):
                raise ValueError("Expected one padded integer prefix with the frozen input shape")
            start = time.monotonic()
            inputs = mx.array([prefix], dtype=mx.int32)
            predictions = mx.argmax(tokenizer.decode(generator.generate(inputs)), axis=-1)
            mx.eval(predictions)
            mx.synchronize(mx.gpu)
            tokens = predictions.tolist()[0]
            active = time.monotonic() - start
            idle_start = time.monotonic()
            time.sleep(active * 0.25)
            idle = time.monotonic() - idle_start
            stats["active_seconds"] += active
            stats["idle_seconds"] += idle
            stats["calls"] += 1
            emit({"type": "prediction", "tokens": tokens,
                  "active_seconds": active, "idle_seconds": idle,
                  "peak_mlx_memory_bytes": mx.get_peak_memory()})
        except Exception as error:
            emit({"type": "policy_failure", "reason": "worker_exception",
                  "error": f"{type(error).__name__}: {error}"})
    stats["peak_mlx_memory_bytes"] = mx.get_peak_memory()
    stats["torch_imported"] = "torch" in sys.modules
    elapsed = stats["active_seconds"] + stats["idle_seconds"]
    stats["observed_process_duty"] = stats["active_seconds"] / elapsed if elapsed else 0.0
    emit({"type": "closed", **stats})


if __name__ == "__main__":
    main()
