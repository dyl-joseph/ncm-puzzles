"""Synthetic full-model backend check on disposable weights, never a game score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

import mlx.core as mx

from .gameplay_worker import GameplayTrainer
from .mlx_ncm import Generator, Tokenizer, gpu_backend, validate_device
from .train_gameplay import digest, write_json


HERE = Path(__file__).resolve().parent


def check(checkpoint, require_cuda):
    mx.set_default_device(mx.gpu)
    validate_device()
    backend = gpu_backend()
    if require_cuda and backend != "cuda":
        raise RuntimeError("Colab requires MLX CUDA. Choose a compatible GPU runtime and rerun setup.")
    original = {path.name: digest(path) for path in checkpoint.iterdir() if path.is_file()}
    fixture = json.loads((HERE / "fixtures/replay_controls.json").read_text())
    with tempfile.TemporaryDirectory(prefix="arc3-gpu-selfcheck-") as temporary:
        trainer = GameplayTrainer(SimpleNamespace(checkpoint=checkpoint, output=Path(temporary),
                                                 seed=9173, deadline=time.monotonic() + 150))
        trainer.begin({"policy_version": 0, "episode_index": 0})
        sample = trainer.act({"policy_version": 0, "prefix": fixture["controls"][0]["prefix"]})
        update = trainer.update({"policy_version": 0, "rewards": [-1.0]})
        if sample["type"] != "prediction" or not update["applied"] or update["gradient_norm"] <= 0:
            raise AssertionError("Synthetic backend check did not apply a finite nonzero gradient")
        saved = trainer.save()
        if saved["generator_parameter_delta_l2"] <= 0:
            raise AssertionError("Synthetic backend check did not change generator weights")
        controls = json.loads((trainer.checkpoint / "inference_controls.json").read_text())
        tokenizer = Tokenizer(trainer.metadata["vocab_size"])
        tokenizer.load_weights(str(trainer.checkpoint / "tokenizer.safetensors"))
        tokenizer.eval()
        generator = Generator(trainer.metadata["vocab_size"], trainer.metadata["prefix_len"], tokenizer)
        generator.load_weights(str(trainer.checkpoint / "generator.safetensors"))
        generator.eval()
        for control, expected in zip(fixture["controls"], controls["controls"]):
            prefix = mx.array([control["prefix"]], dtype=mx.int32)
            predicted = mx.argmax(tokenizer.decode(generator.generate(prefix)), axis=-1)
            mx.eval(predicted)
            mx.synchronize(mx.gpu)
            if predicted.tolist()[0] != expected["tokens"]:
                raise AssertionError("Saved checkpoint changed its same-runtime inference controls on reload")
        stats = trainer.close()
    if original != {path.name: digest(path) for path in checkpoint.iterdir() if path.is_file()}:
        raise AssertionError("Backend check changed the starting checkpoint")
    if stats["torch_imported"] or "torch" in sys.modules:
        raise AssertionError("Torch was imported into the isolated MLX process")
    return {"status": "passed", "purpose": "synthetic backend selfcheck, not game performance",
            "backend": backend, "device": stats["device"], "gradient_norm": update["gradient_norm"],
            "generator_delta_l2": saved["generator_parameter_delta_l2"],
            "sampling_parity": stats["sampling_parity"], "reload_controls_verified": len(controls["controls"]),
            "starting_checkpoint_unchanged": True, "tokenizer_unchanged": True,
            "torch_imported": False, "cpu_neural_computation": False,
            "peak_mlx_memory_bytes": stats["peak_mlx_memory_bytes"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=HERE / "results/gameplay-training/checkpoint")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check(args.checkpoint.resolve(), args.require_cuda)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
