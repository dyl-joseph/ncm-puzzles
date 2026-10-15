#!/usr/bin/env python3
"""Train original NCM models from scratch on the prepared 9x9 Latin corpus."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import signal
import sys
import tempfile
import time
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from evaluate_latin import (ROOT, LoadedNCM, TorchPredictor, board_text, generate_batch,
                            score_prediction, score_totals, sha256_file, verify_environment,
                            training_stage_deviations, verify_local_export)


@dataclass(frozen=True)
class StageSpec:
    name: str
    updates: int
    batch_size: int
    microbatch_size: int
    peak_lr: float
    floor_lr: float
    warmup_updates: int

    def __post_init__(self):
        if self.name not in ("tokenizer", "generator"):
            raise ValueError("Unknown training stage.")
        if self.batch_size < 1 or self.microbatch_size < 1 or self.batch_size % self.microbatch_size:
            raise ValueError("The positive global batch must be divisible by the microbatch size.")
        if self.name == "tokenizer" and self.microbatch_size != self.batch_size:
            raise ValueError("Tokenizer accumulation changes codebook EMA. Use one physical global batch.")
        if (self.updates < 1 or not 0 <= self.warmup_updates < self.updates
                or not math.isfinite(self.peak_lr) or not math.isfinite(self.floor_lr)
                or not 0 < self.floor_lr <= self.peak_lr):
            raise ValueError("Training requires a positive schedule, valid warmup, and finite learning rates.")


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@contextmanager
def exclusive_run(run: Path):
    import fcntl

    run = run.resolve()
    if ROOT / "runs" not in run.parents:
        raise ValueError("Training run directories must live under this project's runs directory.")
    run.mkdir(parents=True, exist_ok=True)
    with (run / ".train.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another trainer owns {run}. Wait for it to stop before resuming.") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def learning_rate(spec: StageSpec, update: int) -> float:
    """Return the learning rate for a one-based optimizer update."""
    if not 1 <= update <= spec.updates:
        raise ValueError("The update must belong to the intended complete schedule.")
    if update <= spec.warmup_updates:
        return spec.peak_lr * update / spec.warmup_updates
    progress = ((update - spec.warmup_updates) / (spec.updates - spec.warmup_updates)
                if spec.warmup_updates else (update - 1) / max(spec.updates - 1, 1))
    return spec.floor_lr + (spec.peak_lr - spec.floor_lr) * (1 + math.cos(math.pi * progress)) / 2


def original_classes():
    upstream = str(ROOT / "vendor/ncm")
    if upstream not in sys.path:
        sys.path.insert(0, upstream)
    from models.next_concept import NextConceptTransformer
    from models.vqvae import VQVAE

    return VQVAE, NextConceptTransformer


def make_tokenizer(architecture, device: str):
    VQVAE, _ = original_classes()
    return VQVAE(**architecture.tokenizer.vqvae).to(device=device, dtype=torch.float32)


def make_generator(architecture, tokenizer, device: str):
    _, NextConceptTransformer = original_classes()
    tokenizer.eval().requires_grad_(False)
    return NextConceptTransformer(vqvae_local=tokenizer, config=architecture.model).to(
        device=device, dtype=torch.float32)


def tokenizer_loss(model, solutions):
    logits, vq_loss, _, _, _ = model(solutions)
    reconstruction = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), solutions.reshape(-1))
    return reconstruction + vq_loss, {"reconstruction_ce": reconstruction.detach(),
                                      "vq_loss": vq_loss.detach()}


def generator_loss(model, tokenizer, solutions, prefixes):
    with torch.no_grad():
        targets = tokenizer.encode(solutions)
        contexts = tokenizer.vq.idx_to_ncm_input(targets)
        prefix_embeddings = tokenizer.encoder.wte(prefixes)
    query = model.build_l0_query_input(solutions.shape[0])
    inputs = torch.cat((prefix_embeddings, query, contexts), dim=1)
    if inputs.shape[1:] != (413, 256) or targets.shape != (solutions.shape[0], 322):
        raise ValueError("The original conditional hierarchy must contain all 322 target codes.")
    logits = model(inputs)
    if logits.shape != (solutions.shape[0], 322, 16):
        raise ValueError("The generator must produce 322 sixteen-way predictions.")
    loss = F.cross_entropy(logits.reshape(-1, 16), targets.reshape(-1))
    return loss, {"code_ce": loss.detach()}


def validate_stage(stage, model, tokenizer, corpus, recipe, device: str, batch_size: int):
    original_classes()
    from tokenization.sudoku_tokenizer import SudokuTokenizer

    saved_rng = rng_state(device)
    model_mode, tokenizer_mode = model.training, tokenizer.training
    try:
        model.eval()
        tokenizer.eval()
        count = min(int(recipe.validation_count), len(corpus.splits["validation"]))
        cases = corpus.evaluation_cases("validation", count, seed=int(recipe.seed), clue_count=int(recipe.clue_count))
        raw_tokenizer = SudokuTokenizer()
        loaded = LoadedNCM(TorchPredictor(model, tokenizer, device), raw_tokenizer, {})
        scores, ce_sum, tokens_correct, tokens_total = [], 0.0, 0, 0
        with torch.inference_mode():
            for offset in range(0, count, min(batch_size, 32)):
                batch = cases[offset:offset + min(batch_size, 32)]
                if stage == "generator":
                    predictions = generate_batch(loaded, batch)
                    if len(predictions) != len(batch):
                        raise RuntimeError("Validation returned a different number of predictions than puzzles.")
                    scores.extend(score_prediction(case, ids) for case, ids in zip(batch, predictions))
                else:
                    targets = torch.tensor([raw_tokenizer.encode(board_text(case.witness)) for case in batch],
                                           dtype=torch.long, device=device)
                    if device == "mps":
                        from ncm_torch_ops import NCMMPSSafeOps
                        operators = NCMMPSSafeOps()
                    else:
                        operators = nullcontext()
                    with operators:
                        logits = tokenizer.decode(tokenizer.encode(targets))
                        ce_sum += float(F.cross_entropy(logits.reshape(-1, 12), targets.reshape(-1), reduction="sum").cpu())
                        tokens_correct += int((logits.argmax(-1) == targets).sum().cpu())
                        tokens_total += targets.numel()
        return ({"count": count, **score_totals(scores)} if stage == "generator" else
                {"count": count, "full_hierarchy_reconstruction_ce": ce_sum / tokens_total,
                 "token_accuracy": tokens_correct / tokens_total})
    finally:
        model.train(model_mode)
        tokenizer.train(tokenizer_mode)
        restore_rng(saved_rng, device)


def make_optimizer(model, recipe):
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    groups = [
        {"params": [p for p in trainable if p.ndim >= 2], "weight_decay": recipe.optimizer.weight_decay},
        {"params": [p for p in trainable if p.ndim < 2], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=0.0, betas=tuple(recipe.optimizer.betas),
                             eps=recipe.optimizer.epsilon, foreach=False, fused=False)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_state(device: str) -> dict:
    np_state = np.random.get_state()
    state = {"python": random.getstate(), "numpy": [np_state[0], np_state[1].tolist(),
             int(np_state[2]), int(np_state[3]), float(np_state[4])], "torch_cpu": torch.get_rng_state()}
    if device == "mps":
        state["torch_mps"] = torch.mps.get_rng_state()
    elif device == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict, device: str):
    random.setstate(state["python"])
    np_state = state["numpy"]
    np.random.set_state((np_state[0], np.asarray(np_state[1], dtype=np.uint32),
                         np_state[2], np_state[3], np_state[4]))
    torch.set_rng_state(state["torch_cpu"])
    if device == "mps":
        torch.mps.set_rng_state(state["torch_mps"])
    elif device == "cuda":
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def atomic_torch_save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".checkpoint-", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("wb") as stream:
            torch.save(cpu_tree(value), stream)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".state-", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_boundary(path, model, optimizer, stream, completed, immutable, device):
    quantizer = model.vq if immutable["stage"]["name"] == "tokenizer" else None
    atomic_torch_save(path, {
        "format": "ncm-latin-training-state-v1", "immutable": immutable,
        "completed_updates": completed, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "sampler": stream.state_dict(),
        "rng": rng_state(device),
        "fine_drop_step": int(quantizer._fd_step.item()) if quantizer is not None else None,
    })


def load_boundary(path, model, optimizer, stream, immutable, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != "ncm-latin-training-state-v1" or state.get("immutable") != immutable:
        raise ValueError("Resume configuration, dataset, source, dependencies, or tokenizer snapshot changed.")
    completed = state["completed_updates"]
    if type(completed) is not int or not 0 <= completed <= immutable["stage"]["updates"]:
        raise ValueError("Invalid completed-update count in checkpoint.")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    stream.load_state_dict(state["sampler"])
    if immutable["stage"]["name"] == "tokenizer":
        model.vq._fd_step.fill_(state["fine_drop_step"])
    restore_rng(state["rng"], device)
    return completed


def stage_spec(recipe, stage: str, batch_size=None, microbatch_size=None):
    settings = recipe[stage]
    batch = batch_size if batch_size is not None else int(settings.batch_size)
    micro = (batch if stage == "tokenizer" else
             min(int(settings.microbatch_size), batch) if microbatch_size is None else microbatch_size)
    if stage == "tokenizer" and microbatch_size not in (None, batch):
        raise ValueError("Tokenizer training requires one physical batch per optimizer update.")
    return StageSpec(stage, int(settings.updates), batch, micro,
                     float(settings.peak_lr), float(settings.floor_lr), int(settings.warmup_updates))


def validate_device(device: str):
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS is unavailable on this machine.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable on this machine.")


def run_context(data: Path, run: Path, device: str, recipe_path: Path):
    from latin_data import LatinCorpus

    environment = verify_environment()
    validate_device(device)
    recipe = OmegaConf.load(recipe_path)
    if (recipe.architecture != "vendor/ncm/config/eval/sudoku-hard.yaml"
            or recipe.schedule != "linear_warmup_cosine" or recipe.optimizer.name != "AdamW"
            or recipe.optimizer.decay_grouping != "matrix_parameters_only"
            or recipe.generator.level_weight_alpha != 0
            or list(recipe.generator.corruption_rates) != [0.0] * 12
            or int(recipe.checkpoint_every) < 1 or int(recipe.validation_every) < 1
            or int(recipe.validation_count) < 1):
        raise ValueError("Unsupported architecture, objective, optimizer, or schedule in training config.")
    architecture_path = ROOT / recipe.architecture
    architecture = OmegaConf.load(architecture_path)
    corpus = LatinCorpus.open(data)
    config = {
        "format": "ncm-latin-run-v1", "upstream": environment["upstream"],
        "packages": environment["packages"], "recipe": OmegaConf.to_container(recipe, resolve=True),
        "architecture_sha256": sha256_file(architecture_path),
        "data_manifest_sha256": corpus.manifest_sha256,
        "device": device, "precision": "float32", "compiled": False,
        "trainer_sha256": sha256_file(Path(__file__)),
        "data_code_sha256": sha256_file(ROOT / "latin_data.py"),
        "mps_adapter_sha256": sha256_file(ROOT / "ncm_torch_ops.py") if device == "mps" else None,
    }
    run = run.resolve()
    if ROOT / "runs" not in run.parents:
        raise ValueError("Training run directories must live under this project's runs directory.")
    run.mkdir(parents=True, exist_ok=True)
    path = run / "run.json"
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError("Run configuration changed. Choose a new --run directory.")
    else:
        atomic_json(path, config)
    return corpus, recipe, architecture, config


def snapshot_tokenizer(run: Path, allow_partial: bool, stop_after: int | None):
    destination = run / "generator" / "tokenizer-snapshot.pt"
    metadata_path = destination.with_suffix(".json")
    if destination.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        if metadata["sha256"] != sha256_file(destination):
            raise ValueError("Tokenizer snapshot checksum changed.")
    else:
        if (run / "generator/state.pt").exists():
            raise ValueError("Incomplete immutable tokenizer snapshot with an existing generator checkpoint.")
        source = run / "tokenizer/export"
        provenance_path = source / "provenance.json"
        if not provenance_path.is_file():
            raise ValueError("Train the tokenizer stage before starting the generator.")
        provenance = verify_local_export(source, stage="tokenizer")["checkpoint"]
        completed = provenance["training"]["tokenizer_updates"]
        expected = provenance["training"]["tokenizer_intended_updates"]
        metadata = {"completed_updates": completed, "intended_updates": expected,
                    "sha256": provenance["files"]["tokenizer.pt"]["sha256"],
                    "stage_spec": provenance["training"]["stage_spec"],
                    "deviations": provenance["training"]["deviations"]}
        if sha256_file(source / "tokenizer.pt") != metadata["sha256"]:
            raise ValueError("Tokenizer export checksum mismatch.")
        if completed != expected and not (allow_partial and stop_after is not None):
            raise ValueError("A partial tokenizer requires --allow-partial-tokenizer and --stop-after.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".snapshot-", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            shutil.copyfile(source / "tokenizer.pt", temporary)
            if sha256_file(temporary) != metadata["sha256"]:
                raise ValueError("Tokenizer export changed while creating its immutable snapshot.")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        atomic_json(metadata_path, metadata)
    if metadata["completed_updates"] != metadata["intended_updates"] and not (allow_partial and stop_after is not None):
        raise ValueError("A partial tokenizer requires --allow-partial-tokenizer and --stop-after on every invocation.")
    return destination, metadata


def export_stage(run, stage, model, architecture, context, corpus, completed, spec, snapshot):
    destination = run / stage / "export"
    destination.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(destination / f"{stage}.pt", {"model": model.state_dict()})
    tokenizer_updates = completed if stage == "tokenizer" else snapshot[1]["completed_updates"]
    if stage == "generator":
        shutil.copyfile(snapshot[0], destination / "tokenizer.pt")
    shutil.copyfile(ROOT / context["recipe"]["architecture"], destination / "config.yaml")
    shutil.copyfile(corpus.directory / "manifest.json", destination / "data-manifest.json")
    atomic_json(destination / "run-config.json", context)
    names = ["tokenizer.pt", "config.yaml", "data-manifest.json", "run-config.json"]
    if stage == "generator":
        names.append("generator.pt")
    provenance = {
        "format": "ncm-latin-export-v1", "stage": stage,
        "upstream": context["upstream"],
        "architecture_sha256": context["architecture_sha256"],
        "files": {name: {"sha256": sha256_file(destination / name), "bytes": (destination / name).stat().st_size}
                  for name in names},
        "run_config_sha256": canonical_hash(context),
        "data": {"manifest_sha256": corpus.manifest_sha256, "manifest": corpus.manifest},
        "training": {"tokenizer_updates": tokenizer_updates,
                     "tokenizer_intended_updates": context["recipe"]["tokenizer"]["updates"],
                     "generator_updates": completed if stage == "generator" else 0,
                     "generator_intended_updates": context["recipe"]["generator"]["updates"],
                     "stage_spec": asdict(spec), "dtype": "float32", "compiled": False,
                     "deviations": training_stage_deviations(asdict(spec)),
                     "local_choices": context["recipe"]["local_choices"],
                     "tokenizer_snapshot": snapshot[1] if snapshot else None},
    }
    atomic_json(destination / "provenance.json", provenance)
    return destination


def train_stage(stage, corpus, recipe, architecture, context, args):
    stage_dir = args.run.resolve() / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = stage_dir / "state.pt"
    if checkpoint.exists() != args.resume:
        raise ValueError("Use --resume for an existing stage checkpoint. Omit it for a new stage.")
    batch_size, microbatch_size = args.batch_size, args.microbatch_size
    if args.resume:
        stored = torch.load(checkpoint, map_location="cpu", weights_only=True)["immutable"]["stage"]
        batch_size = stored["batch_size"] if batch_size is None else batch_size
        microbatch_size = stored["microbatch_size"] if microbatch_size is None else microbatch_size
    spec = stage_spec(recipe, stage, batch_size, microbatch_size)
    seed_everything(int(recipe.seed))
    snapshot = None
    tokenizer = make_tokenizer(architecture, args.device)
    if stage == "generator":
        snapshot = snapshot_tokenizer(args.run.resolve(), args.allow_partial_tokenizer, args.stop_after)
        tokenizer.load_state_dict(torch.load(snapshot[0], map_location="cpu", weights_only=True)["model"], strict=True)
        model = make_generator(architecture, tokenizer, args.device)
    else:
        model = tokenizer
    model.train()
    optimizer = make_optimizer(model, recipe)
    stream = corpus.training_stream(seed=int(recipe.seed), clue_count=int(recipe.clue_count), augment=bool(recipe.augment))
    immutable = {"run_config_sha256": canonical_hash(context), "stage": asdict(spec),
                 "tokenizer_snapshot": snapshot[1] if snapshot else None}
    completed = load_boundary(checkpoint, model, optimizer, stream, immutable, args.device) if args.resume else 0
    if not args.resume:
        save_boundary(checkpoint, model, optimizer, stream, completed, immutable, args.device)
    last_saved = completed
    target = min(spec.updates, completed + args.stop_after) if args.stop_after is not None else spec.updates
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    previous_signal = signal.signal(signal.SIGINT, request_stop)
    previous_term = signal.signal(signal.SIGTERM, request_stop)
    started = time.perf_counter()
    initial = completed
    status = "running"
    try:
        while completed < target and not stop_requested:
            batch = stream.next_batch(spec.batch_size)
            optimizer.zero_grad(set_to_none=True)
            lr = learning_rate(spec, completed + 1)
            for group in optimizer.param_groups:
                group["lr"] = lr
            diagnostics = {}
            total_loss = 0.0
            for offset in range(0, spec.batch_size, spec.microbatch_size):
                solutions = torch.as_tensor(np.array(batch.solutions[offset:offset + spec.microbatch_size], copy=True),
                                            dtype=torch.long, device=args.device)
                prefixes = torch.as_tensor(np.array(batch.prefixes[offset:offset + spec.microbatch_size], copy=True),
                                           dtype=torch.long, device=args.device)
                if args.device == "mps":
                    from ncm_torch_ops import NCMMPSSafeOps
                    operators = NCMMPSSafeOps()
                else:
                    operators = nullcontext()
                with operators:
                    loss, pieces = (tokenizer_loss(model, solutions) if stage == "tokenizer" else
                                    generator_loss(model, tokenizer, solutions, prefixes))
                    if not torch.isfinite(loss).item():
                        raise FloatingPointError("Nonfinite training loss. The previous saved boundary is intact.")
                    (loss * (spec.microbatch_size / spec.batch_size)).backward()
                total_loss += float(loss.detach().cpu()) * (spec.microbatch_size / spec.batch_size)
                for name, value in pieces.items():
                    diagnostics[name] = diagnostics.get(name, 0.0) + float(value.cpu()) * (spec.microbatch_size / spec.batch_size)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe.clip_norm), error_if_nonfinite=True)
            optimizer.step()
            completed += 1
            if completed % int(recipe.checkpoint_every) == 0 or completed == target or stop_requested:
                save_boundary(checkpoint, model, optimizer, stream, completed, immutable, args.device)
                last_saved = completed
            if completed % int(recipe.validation_every) == 0:
                validation = {"stage": stage, "completed_updates": completed,
                              "split": "validation", "seed": int(recipe.seed),
                              **validate_stage(stage, model, tokenizer, corpus, recipe, args.device, spec.microbatch_size)}
                with (stage_dir / "validation.jsonl").open("a") as stream_file:
                    stream_file.write(json.dumps(validation) + "\n")
                print(json.dumps({"validation": validation}), flush=True)
            if completed == initial + 1 or completed % 10 == 0 or completed == target or stop_requested:
                elapsed = time.perf_counter() - started
                record = {"stage": stage, "completed_updates": completed, "intended_updates": spec.updates,
                          "loss": total_loss, "lr": lr, "gradient_norm": float(grad_norm.cpu()),
                          "new_updates": completed - initial, "elapsed_seconds": elapsed,
                          "seconds_per_update": elapsed / (completed - initial), **diagnostics}
                with (stage_dir / "metrics.jsonl").open("a") as stream_file:
                    stream_file.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
        if stop_requested and completed > initial:
            save_boundary(checkpoint, model, optimizer, stream, completed, immutable, args.device)
            last_saved = completed
        status = "complete" if completed == spec.updates else "stopped"
        export = export_stage(args.run.resolve(), stage, model, architecture, context, corpus, completed, spec, snapshot)
        print(f"Saved {stage} export to {export}", flush=True)
    except BaseException:
        status = "failed"
        raise
    finally:
        signal.signal(signal.SIGINT, previous_signal)
        signal.signal(signal.SIGTERM, previous_term)
        atomic_json(stage_dir / "status.json", {"status": status, "completed_updates_in_process": completed,
                    "checkpoint": str(checkpoint), "last_saved_updates": last_saved,
                    "intended_updates": spec.updates, "finished_at": datetime.now(timezone.utc).isoformat()})
    return completed == spec.updates


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("tokenizer", "generator", "all"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/latin9.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, help="Limit new complete updates without changing the full learning-rate schedule.")
    parser.add_argument("--batch-size", type=int, help="Override the global optimizer batch. Tokenizer uses this as one physical batch.")
    parser.add_argument("--microbatch-size", type=int, help="Generator physical microbatch. Its gradients accumulate to --batch-size.")
    parser.add_argument("--allow-partial-tokenizer", action="store_true", help="Permit a generator smoke run only together with --stop-after.")
    args = parser.parse_args(argv)
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("--stop-after must be positive.")
    if args.allow_partial_tokenizer and args.stop_after is None:
        parser.error("--allow-partial-tokenizer requires --stop-after.")
    torch.set_num_threads(8)
    try:
        if args.stage == "all" and args.resume and not any((args.run / stage / "state.pt").is_file()
                                                         for stage in ("tokenizer", "generator")):
            raise ValueError("No stage checkpoint exists to resume.")
        with exclusive_run(args.run):
            corpus, recipe, architecture, context = run_context(args.data, args.run, args.device, args.config)
            stages = ("tokenizer", "generator") if args.stage == "all" else (args.stage,)
            for stage in stages:
                stage_args = argparse.Namespace(**vars(args))
                if args.stage == "all":
                    stage_args.resume = args.resume and (args.run / stage / "state.pt").exists()
                    if stage == "tokenizer":
                        stage_args.microbatch_size = None
                complete = train_stage(stage, corpus, recipe, architecture, context, stage_args)
                if stage == "tokenizer" and args.stage == "all" and not complete:
                    print("Tokenizer schedule is incomplete. The generator stage has not started.", flush=True)
                    break
    except (Exception, KeyboardInterrupt) as error:
        print(f"Training failed: {error}", file=sys.stderr)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
