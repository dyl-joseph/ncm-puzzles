"""Train tiny upstream NCM models on actual 4x4 Sudoku completions."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
LEVELS = (1, 2, 4, 8, 16)
CHECKPOINT_SCHEMA = 1


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@lru_cache(maxsize=1)
def solutions():
    """Enumerate the 288 ordered Sudoku grids, including their symbol labels."""
    rows = tuple(itertools.permutations((1, 2, 3, 4)))
    boards = []
    for grid in itertools.product(rows, repeat=4):
        if any(len({grid[row][column] for row in range(4)}) != 4 for column in range(4)):
            continue
        if any(len({grid[row][column] for row in range(top, top + 2)
                    for column in range(left, left + 2)}) != 4
               for top in (0, 2) for left in (0, 2)):
            continue
        boards.append(tuple(cell for row in grid for cell in row))
    return tuple(boards)


def score_board(puzzle, prediction):
    """Judge a completion independently of its training witness."""
    puzzle, prediction = tuple(puzzle), tuple(prediction)
    if len(puzzle) != 16 or any(type(value) is not int or not 0 <= value <= 4 for value in puzzle):
        raise ValueError("A Sudoku puzzle must contain sixteen integer tokens in 0..4.")
    valid = len(prediction) == 16 and all(type(value) is int and 1 <= value <= 4 for value in prediction)
    clues = sum(value != 0 for value in puzzle)
    clue_matches = sum(value == prediction[index] for index, value in enumerate(puzzle)
                       if value and index < len(prediction))
    alphabet = {1, 2, 3, 4}
    rows = sum(set(prediction[row * 4:row * 4 + 4]) == alphabet for row in range(4)) if valid else 0
    columns = sum(set(prediction[column::4]) == alphabet for column in range(4)) if valid else 0
    boxes = sum({prediction[row * 4 + column] for row in range(top, top + 2)
                 for column in range(left, left + 2)} == alphabet
                for top in (0, 2) for left in (0, 2)) if valid else 0
    return {"clue_matches": clue_matches, "clues": clues, "valid_rows": rows,
            "valid_columns": columns, "valid_boxes": boxes,
            "valid_alphabet": valid,
            "solved": valid and clue_matches == clues and rows == columns == boxes == 4}


def architecture():
    return {
        "tokenizer": {"in_vocab_size": 5, "context_length": 16, "embed_dim": 32,
                      "vqvae_vocab_size": [8] * len(LEVELS), "n_layers": 4,
                      "compression_factor": 1, "dropout": 0.1, "dec_layers": 1,
                      "pre_quant_groupnorm": None,
                      "vector_quantizer_config": {
                          "clss": "multiscale_residual_vector_quantizer", "decay": 0.9,
                          "epsilon": 1e-5, "commitment_cost": 0.25,
                          "quant_resi": {"enabled": False},
                          "levels": {"use_manual_levels": True, "manual_levels": list(LEVELS)}}},
        "generator": {"n_layer": 2, "n_head": 4, "n_embd": 64, "bias": True,
                      "dropout": 0.1, "context_length": 47, "vocab_size": [8] * len(LEVELS),
                      "prefix_len": 16, "attn_pattern": "block_diagonal",
                      "use_positional_encoding": True, "use_level_encoding": True,
                      "use_rope": True, "use_qk_norm": True,
                      "query_level0": True, "raw_token_prefix": True}}


@dataclass
class Sampler:
    boards: tuple
    seed: int
    clues: int
    cursor: int = 0

    def __post_init__(self):
        self.rng = random.Random(self.seed)

    def batch(self, count):
        answers, puzzles = [], []
        for _ in range(count):
            answer = self.boards[self.rng.randrange(len(self.boards))]
            keep = set(self.rng.sample(range(16), self.clues))
            answers.append(answer)
            puzzles.append(tuple(value if index in keep else 0 for index, value in enumerate(answer)))
        self.cursor += count
        return answers, puzzles

    def state_dict(self):
        return {"cursor": self.cursor, "rng": self.rng.getstate()}

    def load_state_dict(self, state):
        self.cursor = int(state["cursor"])
        self.rng.setstate(state["rng"])


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_state(device):
    numpy = np.random.get_state()
    state = {"python": random.getstate(), "numpy": [numpy[0], numpy[1].tolist(),
             int(numpy[2]), int(numpy[3]), float(numpy[4])], "torch_cpu": torch.get_rng_state()}
    if device == "mps":
        state["torch_mps"] = torch.mps.get_rng_state()
    elif device.startswith("cuda"):
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state, device):
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32),
                         numpy[2], numpy[3], numpy[4]))
    torch.set_rng_state(state["torch_cpu"])
    if device == "mps":
        torch.mps.set_rng_state(state["torch_mps"])
    elif device.startswith("cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def make_models(device):
    upstream = str(ROOT / "vendor/ncm")
    if upstream not in sys.path:
        sys.path.insert(0, upstream)
    from models.next_concept import NextConceptTransformer
    from models.vqvae import VQVAE

    config = OmegaConf.create(architecture())
    tokenizer = VQVAE(**config.tokenizer).to(device=device, dtype=torch.float32)
    generator = NextConceptTransformer(tokenizer, config.generator).to(device=device, dtype=torch.float32)
    return tokenizer, generator


def optimizer_for(model, learning_rate):
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    groups = [{"params": [parameter for parameter in trainable if parameter.ndim >= 2], "weight_decay": 0.01},
              {"params": [parameter for parameter in trainable if parameter.ndim < 2], "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.99), eps=1e-8,
                             foreach=False, fused=False)


def operators(device):
    if device == "mps":
        from ncm_torch_ops import NCMMPSSafeOps

        return NCMMPSSafeOps()
    return nullcontext()


def atomic_save(path, payload, runtime):
    payload = runtime.cpu_tree(payload)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".sudoku4-", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def reference_generator_loss(model, tokenizer, answers, puzzles, forward_context):
    with torch.no_grad():
        codes = tokenizer.encode(answers)
        contexts = tokenizer.vq.idx_to_ncm_input(codes)
        prefix = tokenizer.encoder.wte(puzzles)
    query = model.build_l0_query_input(answers.shape[0])
    inputs = torch.cat((prefix, query, contexts), dim=1)
    with forward_context():
        logits = model(inputs)
    expected = (answers.shape[0], sum(tokenizer.vq.levels), 8)
    if tuple(logits.shape) != expected or tuple(codes.shape) != expected[:2]:
        raise RuntimeError("The task's concept hierarchy and prediction shapes disagree.")
    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), codes.reshape(-1))
    return loss, {"code_ce": loss.detach()}


@dataclass(frozen=True)
class Sudoku4Task:
    updates: int
    tokenizer_updates: int
    batch_size: int
    seed: int
    clues: int
    checkpoint_every: int
    stop_after: int | None
    evaluate: bool

    name = "sudoku4-upstream-ncm"

    @property
    def source_paths(self):
        return (Path(__file__).resolve(), ROOT / "ncm_torch_ops.py",
                *(ROOT / "vendor/ncm" / path for path in (
                    "models/vqvae.py", "models/next_concept.py", "models/quant.py",
                    "models/common.py", "models/vae.py", "utils/registry.py", "utils/misc.py")))

    @property
    def metadata(self):
        boards = solutions()
        return {"schema": CHECKPOINT_SCHEMA, "task": self.name,
                "data": {"kind": "all_ordered_4x4_sudoku_with_2x2_boxes", "solutions": len(boards),
                         "sha256": canonical_hash(boards), "train_solutions": 256,
                         "heldout_solutions": 32, "clues": self.clues,
                         "masking": "uniform_random_fixed_clue_count", "blank_token": 0,
                         "symbol_tokens": [1, 2, 3, 4], "split_seed": self.seed},
                "architecture": architecture(), "seed": self.seed, "batch_size": self.batch_size,
                "targets": {"tokenizer": self.tokenizer_updates, "generator": self.updates},
                "objectives": {"tokenizer": "mean_reconstruction_ce_plus_upstream_vq_loss",
                               "generator": "mean_ce_all_31_clean_codes_frozen_tokenizer"},
                "optimizer": {"name": "AdamW", "tokenizer_lr": 3e-4, "generator_lr": 3e-4,
                              "betas": [0.9, 0.99], "epsilon": 1e-8,
                              "weight_decay": "0.01_on_matrix_parameters", "clip_norm": 1.0},
                "evaluation": {"enabled": self.evaluate, "count": 32, "temperature": 0,
                               "criterion": "clues_and_all_rows_columns_2x2_boxes_valid",
                               "training_witness_required": False},
                "recipe": "tiny_demonstration_from_scratch_not_published_training_hyperparameters"}

    def run(self, runtime):
        if runtime.device == "mlx":
            raise ValueError("This upstream NCM example requires a PyTorch backend.")
        from puzzle_ncm import select_ncm_loss

        checkpoint = runtime.run_directory / "checkpoint.pt"
        seed_all(self.seed)
        boards = list(solutions())
        random.Random(self.seed).shuffle(boards)
        sampler = Sampler(tuple(boards[:256]), self.seed + 1, self.clues)
        tokenizer, generator = make_models(runtime.device)
        tokenizer_optimizer = optimizer_for(tokenizer, 3e-4)
        generator_optimizer = optimizer_for(generator, 3e-4)
        completed = {"tokenizer": 0, "generator": 0}
        last_loss = None

        if runtime.resuming:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if state["schema"] != CHECKPOINT_SCHEMA or state["task"] != self.metadata:
                raise ValueError("The checkpoint does not match this task and fixed training target.")
            tokenizer.load_state_dict(state["tokenizer"])
            tokenizer.vq._fd_step.copy_(state["tokenizer_fine_drop_step"].to(runtime.device))
            generator.load_state_dict(state["generator"])
            tokenizer_optimizer.load_state_dict(state["tokenizer_optimizer"])
            generator_optimizer.load_state_dict(state["generator_optimizer"])
            sampler.load_state_dict(state["sampler"])
            completed = dict(state["completed"])
            last_loss = state["last_loss"]
            restore_rng(state["rng"], runtime.device)
        elif checkpoint.exists():
            raise FileExistsError("A task checkpoint already exists. Resume this run explicitly.")

        def save():
            atomic_save(checkpoint, {"schema": CHECKPOINT_SCHEMA, "task": self.metadata,
                        "tokenizer": tokenizer.state_dict(), "generator": generator.state_dict(),
                        "tokenizer_fine_drop_step": tokenizer.vq._fd_step,
                        "tokenizer_optimizer": tokenizer_optimizer.state_dict(),
                        "generator_optimizer": generator_optimizer.state_dict(),
                        "sampler": sampler.state_dict(), "completed": dict(completed),
                        "last_loss": last_loss, "rng": rng_state(runtime.device)}, runtime)

        if not runtime.resuming:
            save()

        applied = 0
        selection = None
        for stage, target in (("tokenizer", self.tokenizer_updates), ("generator", self.updates)):
            if stage == "tokenizer":
                tokenizer.train()
                model, optimizer = tokenizer, tokenizer_optimizer
            else:
                tokenizer.eval().requires_grad_(False)
                generator.train()
                model, optimizer = generator, generator_optimizer

                def reference(model, tokenizer, answers, puzzles):
                    return reference_generator_loss(model, tokenizer, answers, puzzles, runtime.forward_context)

                selection = select_ncm_loss(generator, tokenizer, reference, solution_length=16,
                                            prefix_length=16, pooling=runtime.numerical.frozen_pooling)
            while completed[stage] < target:
                if runtime.stop_requested or self.stop_after is not None and applied >= self.stop_after:
                    break
                with runtime.work_unit():
                    answers, puzzles = sampler.batch(self.batch_size)
                    answers = torch.tensor(answers, dtype=torch.long, device=runtime.device)
                    puzzles = torch.tensor(puzzles, dtype=torch.long, device=runtime.device)
                    optimizer.zero_grad(set_to_none=True)
                    with operators(runtime.device):
                        if stage == "tokenizer":
                            with runtime.forward_context():
                                logits, vq_loss, _, _, _ = tokenizer(answers)
                            loss = (F.cross_entropy(logits.float().reshape(-1, 5), answers.reshape(-1))
                                    + vq_loss.float())
                        else:
                            loss, _ = selection(answers, puzzles)
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError("Sudoku4 training produced a nonfinite loss.")
                        loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    completed[stage] += 1
                    applied += 1
                    last_loss = float(loss.detach().cpu())
                print(json.dumps({"stage": stage, "completed": completed[stage],
                                  "target": target, "loss": last_loss}), flush=True)
                if applied % self.checkpoint_every == 0 or runtime.stop_requested:
                    save()
            if completed[stage] < target:
                break
        save()

        evaluation = None
        if self.evaluate and completed["generator"] == self.updates and not runtime.stop_requested:
            saved_rng = rng_state(runtime.device)
            try:
                generator.eval()
                tokenizer.eval()
                evaluation_rng = random.Random(self.seed + 2)
                witnesses, puzzles = tuple(boards[256:]), []
                for witness in witnesses:
                    keep = set(evaluation_rng.sample(range(16), self.clues))
                    puzzles.append(tuple(value if index in keep else 0
                                         for index, value in enumerate(witness)))
                predictions = []
                with torch.inference_mode():
                    for offset in range(0, 32, self.batch_size):
                        if runtime.stop_requested:
                            break
                        with runtime.work_unit():
                            prefixes = torch.tensor(puzzles[offset:offset + self.batch_size],
                                                    dtype=torch.long, device=runtime.device)
                            with operators(runtime.device):
                                with runtime.forward_context():
                                    codes = generator.generate(None, prefixes, temperature=0, top_k=0,
                                                               top_p=1.0, use_kv_cache=False)
                                logits = tokenizer.decode(codes)
                                predictions.extend(logits.argmax(-1).cpu().tolist())
                scores = [score_board(puzzle, prediction) for puzzle, prediction in zip(puzzles, predictions)]
                count = len(scores)
                evaluation = {"count": count, "requested": 32,
                              "solved": sum(score["solved"] for score in scores),
                              "solved_fraction": sum(score["solved"] for score in scores) / count if count else 0,
                              "clue_accuracy": sum(score["clue_matches"] for score in scores)
                                               / sum(score["clues"] for score in scores) if count else 0,
                              "row_validity": sum(score["valid_rows"] for score in scores) / (4 * count) if count else 0,
                              "column_validity": sum(score["valid_columns"] for score in scores) / (4 * count) if count else 0,
                              "box_validity": sum(score["valid_boxes"] for score in scores) / (4 * count) if count else 0,
                              "witness_matches": sum(tuple(prediction) == witness
                                                     for prediction, witness in zip(predictions, witnesses))}
            finally:
                restore_rng(saved_rng, runtime.device)

        return {"completed": completed, "applied_this_invocation": applied,
                "targets": {"tokenizer": self.tokenizer_updates, "generator": self.updates},
                "checkpoint": str(checkpoint), "last_loss": last_loss,
                "ncm_loss": {"implementation": selection.implementation, "reason": selection.reason}
                            if selection is not None else None,
                "evaluation": evaluation}


def build_task(arguments):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=40, help="Fixed total generator update target.")
    parser.add_argument("--tokenizer-updates", type=int, default=20, help="Initial tokenizer update target.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clues", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--stop-after", type=int, help="Stop this invocation after this many applied updates.")
    parser.add_argument("--evaluate", action="store_true", help="Score 32 independent heldout puzzles at completion.")
    options = parser.parse_args(arguments)
    if min(options.updates, options.tokenizer_updates, options.batch_size, options.checkpoint_every) < 1:
        parser.error("Update targets, batch size, and checkpoint interval must be positive.")
    if not 1 <= options.clues <= 16:
        parser.error("The clue count must be between one and sixteen.")
    if not 0 <= options.seed < 2**32:
        parser.error("The seed must be an unsigned 32-bit integer.")
    if options.stop_after is not None and options.stop_after < 1:
        parser.error("The per-invocation update limit must be positive.")
    return Sudoku4Task(**vars(options))
