#!/usr/bin/env python3
"""Evaluate frozen NCM Sudoku weights on clue-preserving Latin completion."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
import time
from typing import Protocol, Sequence
import urllib.request


ROOT = Path(__file__).resolve().parent
SOURCE_URL = "https://users.cecs.anu.edu.au/~bdm/data/latin_is9nt.1.txt.gz"
SOURCE_LINES = 4096
CLUE_COUNTS = (25, 37, 49, 61)
DIGITS = frozenset(range(1, 10))
SEPARATORS = frozenset(range(9, 80, 10))
UPSTREAM_REVISION = "c70274273f489abc73832b2b98584a8483b6b203"
CHECKPOINT_REVISION = "7100573f83178b6e9f45ebffe0e9aa6df384e712"
CHECKPOINT_HASHES = {
    "tokenizer.pt": "3bc6aecad8e15820f9d9bb09750e3c59bdf91ca953f1f11dbd109aaed1d80c0b",
    "generator.pt": "9e0bfe44abc1ca3930b7a63cd56cd0b9564b95e29edfe12dd0903207b89df8b8",
    "config.yaml": "8d58c49e6fe52e60191e06df4061937d5bb84f97c5ffbcd93ed9830c32f8df2f",
}
DEPENDENCIES = {
    "torch": "2.14.1",
    "numpy": "2.5.4",
    "hydra-core": "1.3.7",
    "omegaconf": "2.3.1",
}
MLX_VERSION = "0.32.4"
GENERATION = {"temperature": 0.0, "top_k": 0, "top_p": 1.0, "cfg_scale": 0.0}
SOURCE_BIAS = (
    "Seeded sampling from the first 4096 entries of McKay's order-9 isotopy "
    "representatives with nontrivial symmetries, not a uniform sample of all Latin squares."
)


@dataclass(frozen=True)
class LatinCase:
    id: str
    source_line: int
    puzzle: tuple[int, ...]
    witness: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.id or self.source_line < 1:
            raise ValueError("A case needs an ID and a positive source line.")
        if not isinstance(self.puzzle, tuple) or not isinstance(self.witness, tuple):
            raise ValueError("Puzzle and witness cells must be immutable tuples.")
        if len(self.puzzle) != 81 or len(self.witness) != 81:
            raise ValueError("Puzzle and witness must each contain exactly 81 cells.")
        if not all(type(cell) is int and 0 <= cell <= 9 for cell in self.puzzle):
            raise ValueError("Puzzle cells must be integers from 0 through 9.")
        if not all(type(cell) is int and cell in DIGITS for cell in self.witness):
            raise ValueError("Witness cells must be integers from 1 through 9.")
        if not latin_rows_valid(self.witness) or not latin_columns_valid(self.witness):
            raise ValueError("The witness must be a completed Latin square.")
        if any(given and given != answer for given, answer in zip(self.puzzle, self.witness)):
            raise ValueError("Every given must agree with the witness.")

    @property
    def clue_count(self) -> int:
        return sum(cell != 0 for cell in self.puzzle)


@dataclass(frozen=True)
class LatinScore:
    shape_valid: bool
    digits_valid: bool
    rows_valid: bool
    columns_valid: bool
    clues_preserved: bool
    sudoku_boxes_valid: bool
    given_cells_correct: int
    given_cells_total: int

    @property
    def solved(self) -> bool:
        return all((self.shape_valid, self.digits_valid, self.rows_valid,
                    self.columns_valid, self.clues_preserved))


class Predictor(Protocol):
    def predict(self, prefix_rows: Sequence[Sequence[int]]) -> list[list[int]]: ...


@dataclass(frozen=True)
class LoadedNCM:
    predictor: Predictor
    tokenizer: object
    metadata: dict


@dataclass(frozen=True)
class TorchPredictor:
    net: object
    vqvae: object
    device: str = "cpu"

    def predict(self, prefix_rows: Sequence[Sequence[int]]) -> list[list[int]]:
        import torch
        from contextlib import nullcontext

        with torch.inference_mode():
            if self.device == "mps":
                from ncm_torch_ops import NCMMPSSafeOps
                operators = NCMMPSSafeOps()
            else:
                operators = nullcontext()
            with operators:
                codes = self.net.generate(None, torch.tensor(prefix_rows, dtype=torch.long, device=self.device), **GENERATION)
                _, levels = self.vqvae.vq.idx_to_zhats(codes, return_zlevels=True)
                return self.vqvae.decoder(levels[-1]).argmax(-1).cpu().tolist()


def latin_rows_valid(cells: Sequence[int]) -> bool:
    return all(set(cells[start:start + 9]) == DIGITS for start in range(0, 81, 9))


def latin_columns_valid(cells: Sequence[int]) -> bool:
    return all(set(cells[column::9]) == DIGITS for column in range(9))


def box_conflicts(cells: Sequence[int]) -> tuple[tuple[int, int], ...]:
    pairs = []
    for row in range(0, 9, 3):
        for column in range(0, 9, 3):
            indices = [(row + dr) * 9 + column + dc for dr in range(3) for dc in range(3)]
            for offset, first in enumerate(indices):
                for second in indices[offset + 1:]:
                    if cells[first] != 0 and cells[first] == cells[second]:
                        pairs.append((first, second))
    return tuple(pairs)


def board_text(cells: Sequence[int]) -> str:
    return "|".join("".join(str(cell) if cell else "_" for cell in cells[start:start + 9])
                    for start in range(0, 81, 9))


def score_prediction(case: LatinCase, raw_ids: Sequence[int]) -> LatinScore:
    shape_valid = len(raw_ids) == 89 and all(
        type(raw_ids[index]) is int and raw_ids[index] == 10 for index in SEPARATORS
    )
    if not shape_valid:
        return LatinScore(False, False, False, False, False, False, 0, case.clue_count)
    cells = tuple(token for index, token in enumerate(raw_ids) if index not in SEPARATORS)
    digits_valid = all(type(cell) is int and cell in DIGITS for cell in cells)
    given_correct = sum(type(cell) is int and cell == given
                        for given, cell in zip(case.puzzle, cells) if given)
    return LatinScore(
        shape_valid=True,
        digits_valid=digits_valid,
        rows_valid=digits_valid and latin_rows_valid(cells),
        columns_valid=digits_valid and latin_columns_valid(cells),
        clues_preserved=given_correct == case.clue_count,
        sudoku_boxes_valid=digits_valid and not box_conflicts(cells),
        given_cells_correct=given_correct,
        given_cells_total=case.clue_count,
    )


def parse_source_line(line: str, source_line: int) -> tuple[int, ...]:
    rows = line.split()
    if len(rows) != 9 or any(len(row) != 9 or set(row) - set("012345678") for row in rows):
        raise ValueError(f"Source line {source_line} is not nine rows of 0-based digits.")
    cells = tuple(int(symbol) + 1 for row in rows for symbol in row)
    if not latin_rows_valid(cells) or not latin_columns_valid(cells):
        raise ValueError(f"Source line {source_line} is not a Latin square.")
    return cells


def build_cases(source_path: Path, *, count: int, seed: int) -> tuple[LatinCase, ...]:
    candidates = []
    seen = set()
    for source_line, line in enumerate(source_path.read_text(encoding="ascii").splitlines(), 1):
        witness = parse_source_line(line, source_line)
        conflicts = box_conflicts(witness)
        if witness not in seen and conflicts:
            candidates.append((source_line, witness, conflicts))
            seen.add(witness)
    if count < 1 or count > len(candidates):
        raise ValueError(f"Requested {count} cases, but the source has {len(candidates)} distinct eligible squares.")
    rng = random.Random(seed)
    selected = rng.sample(candidates, count)
    cases = []
    for index, (source_line, witness, conflicts) in enumerate(selected):
        clue_count = CLUE_COUNTS[index % len(CLUE_COUNTS)]
        pair = rng.choice(conflicts)
        givens = set(pair)
        givens.update(rng.sample([cell for cell in range(81) if cell not in givens], clue_count - 2))
        puzzle = tuple(cell if position in givens else 0 for position, cell in enumerate(witness))
        cases.append(LatinCase(f"latin9-line{source_line:04d}-clues{clue_count}",
                               source_line, puzzle, witness))
    return tuple(cases)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_cache(cache_dir: Path) -> tuple[Path, dict]:
    source_path = cache_dir / "latin_is9nt.1.first4096.txt"
    manifest_path = cache_dir / "latin_is9nt.1.first4096.json"
    if source_path.exists() or manifest_path.exists():
        if not source_path.is_file() or not manifest_path.is_file():
            raise ValueError(f"Incomplete source cache in {cache_dir}. Use a fresh --cache-dir.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("url") != SOURCE_URL or manifest.get("line_count") != SOURCE_LINES
                or manifest.get("bytes") != source_path.stat().st_size or not manifest.get("fetched_at")
                or manifest.get("sha256") != sha256_file(source_path)):
            raise ValueError(f"Source cache provenance or checksum mismatch in {cache_dir}. Use a fresh --cache-dir.")
        lines = source_path.read_text(encoding="ascii").splitlines()
        if len(lines) != SOURCE_LINES:
            raise ValueError(f"Expected {SOURCE_LINES} cached source lines. Use a fresh --cache-dir.")
        for number, line in enumerate(lines, 1):
            parse_source_line(line, number)
        return source_path, manifest

    cache_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    print(f"Fetching the first {SOURCE_LINES} public source lines.", flush=True)
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "ncm-latin-evaluation/1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        with gzip.GzipFile(fileobj=response) as archive:
            for number in range(1, SOURCE_LINES + 1):
                line = archive.readline(128)
                if not line.endswith(b"\n"):
                    raise ValueError(f"Incomplete or oversized source line {number}.")
                text = line.decode("ascii")
                parse_source_line(text, number)
                lines.append(text)
    payload = "".join(lines).encode("ascii")
    manifest = {
        "url": SOURCE_URL,
        "line_count": SOURCE_LINES,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "compression": "decompressed bounded prefix; full archive hash unavailable",
        "sampling_bias": SOURCE_BIAS,
    }
    with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as temporary:
        temporary.write(payload)
        temporary_path = Path(temporary.name)
    temporary_path.replace(source_path)
    write_json(manifest_path, manifest)
    return source_path, manifest


def verify_environment(device: str = "cpu") -> dict:
    versions = {}
    dependencies = {**DEPENDENCIES, **({"mlx": MLX_VERSION} if device == "mlx" else {})}
    for package, expected in dependencies.items():
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError("Missing dependencies. Run .venv/bin/python -m pip install -r requirements.txt. See README.md.") from error
        if versions[package] != expected:
            raise RuntimeError(f"Expected {package} {expected}, found {versions[package]}. Install requirements.txt in .venv.")
    upstream = ROOT / "vendor/ncm"
    if not (upstream / ".git").exists():
        raise RuntimeError("Missing vendor/ncm checkout. Follow the pinned setup commands in README.md.")
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if revision != UPSTREAM_REVISION or dirty:
        raise RuntimeError("vendor/ncm must be clean at the pinned revision. See README.md. No source patches are needed.")
    source_files = ("models/next_concept.py", "models/vqvae.py", "models/quant.py", "models/common.py",
                    "models/vae.py", "tokenization/sudoku_tokenizer.py", "tokenization/vqvae_tokenizer.py",
                    "utils/registry.py", "utils/misc.py")
    return {
        "packages": versions,
        "upstream": {"repository": "https://github.com/HazyResearch/ncm", "revision": revision,
                     "files_sha256": {name: sha256_file(upstream / name) for name in source_files}},
    }


def verify_setup(checkpoint_dir: Path, device: str = "cpu") -> dict:
    environment = verify_environment(device)
    hashes = {}
    for filename, expected in CHECKPOINT_HASHES.items():
        path = checkpoint_dir / filename
        if not path.is_file():
            raise RuntimeError(f"Missing {path}. Follow the checkpoint retrieval commands in README.md.")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch for {path}. Retrieve the pinned checkpoint described in README.md.")
        hashes[filename] = {"sha256": actual, "bytes": path.stat().st_size}
    provenance_path = checkpoint_dir / "provenance.json"
    if not provenance_path.is_file():
        raise RuntimeError(f"Missing {provenance_path}. Follow the provenance setup command in README.md.")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (provenance.get("revision") != CHECKPOINT_REVISION
            or provenance.get("repository") != "hazyresearch/ncm-checkpoints"
            or provenance.get("task") != "sudoku/hard"
            or provenance.get("files") != hashes):
        raise RuntimeError("Checkpoint provenance does not match the pinned files. See README.md.")
    return {
        **environment,
        "checkpoint": provenance,
        "checkpoint_directory": str(checkpoint_dir.resolve()),
        "provenance_sha256": sha256_file(provenance_path),
    }


def training_stage_deviations(spec: dict) -> list[str]:
    published_updates, published_warmup = {"tokenizer": (10000, 500), "generator": (50000, 1000)}[spec["name"]]
    deviations = ["float32 eager execution replaces the published BF16 compiled execution"]
    if spec["batch_size"] != 1024:
        deviations.append(f"global batch {spec['batch_size']} replaces published batch 1024")
    if spec["updates"] != published_updates:
        deviations.append(f"{spec['updates']} optimizer updates replace published {published_updates}")
    if spec["warmup_updates"] != published_warmup:
        deviations.append(f"{spec['warmup_updates']} warmup updates replace published {published_warmup}")
    return deviations


def verify_training_stage_spec(spec: dict, deviations: list[str], recipe: dict, stage: str):
    schedule = {"updates", "peak_lr", "floor_lr", "warmup_updates"}
    if (not isinstance(spec, dict) or set(spec) != schedule | {"name", "batch_size", "microbatch_size"}
            or spec["name"] != stage
            or any(type(spec[name]) is not int for name in ("updates", "warmup_updates", "batch_size", "microbatch_size"))
            or spec["updates"] < 1 or not 0 <= spec["warmup_updates"] < spec["updates"]
            or spec["batch_size"] < 1 or spec["microbatch_size"] < 1
            or spec["batch_size"] % spec["microbatch_size"]
            or (stage == "tokenizer" and spec["batch_size"] != spec["microbatch_size"])
            or any(spec[name] != recipe[stage].get(name) for name in schedule)
            or deviations != training_stage_deviations(spec)):
        raise ValueError(f"Local export {stage} stage specification or deviation records are invalid.")


def verify_local_export(checkpoint_dir: Path, device: str = "cpu", *, stage: str = "generator") -> dict:
    environment = verify_environment(device)
    provenance_path = checkpoint_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    names = {"tokenizer.pt", "config.yaml", "data-manifest.json", "run-config.json"}
    if stage == "generator":
        names.add("generator.pt")
    if (provenance.get("format") != "ncm-latin-export-v1" or provenance.get("stage") != stage
            or provenance.get("upstream") != environment["upstream"]
            or set(provenance.get("files", {})) != names):
        raise ValueError("Local export format, source revision, or file inventory is invalid.")
    for name, record in provenance["files"].items():
        path = checkpoint_dir / name
        if (not path.is_file() or record != {"sha256": sha256_file(path), "bytes": path.stat().st_size}):
            raise ValueError(f"Local export checksum mismatch for {name}.")
    architecture_path = ROOT / "vendor/ncm/config/eval/sudoku-hard.yaml"
    if (provenance.get("architecture_sha256") != sha256_file(architecture_path)
            or sha256_file(checkpoint_dir / "config.yaml") != sha256_file(architecture_path)):
        raise ValueError("Local export must retain the pinned original model architecture.")
    manifest_path = checkpoint_dir / "data-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (provenance.get("data") != {"manifest_sha256": sha256_file(manifest_path), "manifest": manifest}
            or manifest.get("format_version") != 1
            or manifest.get("board_dtype") != "uint8"
            or manifest.get("board_shape") != [manifest.get("board_count"), 81]
            or set(manifest.get("split_counts", {})) != {"train", "validation", "test", "benchmark"}
            or sum(manifest["split_counts"].values()) != manifest["board_count"]
            or manifest.get("validation") != {"every_row_and_column": True, "disjoint_split_coverage": True}):
        raise ValueError("Local export corpus provenance or split coverage is invalid.")
    run_config = json.loads((checkpoint_dir / "run-config.json").read_text())
    run_hash = hashlib.sha256(json.dumps(run_config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (run_hash != provenance.get("run_config_sha256")
            or run_config.get("data_manifest_sha256") != sha256_file(manifest_path)
            or run_config.get("architecture_sha256") != sha256_file(architecture_path)
            or run_config.get("upstream") != environment["upstream"]):
        raise ValueError("Local export training configuration does not match its data or architecture.")
    training = provenance.get("training", {})
    recipe = run_config["recipe"]
    for name in ("tokenizer_updates", "generator_updates"):
        intended = training.get(name.replace("_updates", "_intended_updates"))
        if (type(intended) is not int or intended < 1 or intended != recipe[name.replace("_updates", "")]["updates"]
                or type(training.get(name)) is not int or not 0 <= training[name] <= intended):
            raise ValueError("Local export contains invalid training update counts.")
    verify_training_stage_spec(training.get("stage_spec"), training.get("deviations"), recipe, stage)
    if stage == "generator":
        snapshot = training.get("tokenizer_snapshot") or {}
        if (type(snapshot.get("completed_updates")) is not int or type(snapshot.get("intended_updates")) is not int
                or snapshot.get("sha256") != provenance["files"]["tokenizer.pt"]["sha256"]
                or snapshot.get("completed_updates") != training["tokenizer_updates"]
                or snapshot.get("intended_updates") != training["tokenizer_intended_updates"]):
            raise ValueError("Local export does not match its frozen tokenizer snapshot.")
        verify_training_stage_spec(snapshot.get("stage_spec"), snapshot.get("deviations"), recipe, "tokenizer")
    return {**environment, "checkpoint": provenance, "checkpoint_directory": str(checkpoint_dir.resolve()),
            "provenance_sha256": sha256_file(provenance_path), "local_checkpoint": True}


def load_ncm(config_path: Path, checkpoint_dir: Path, device: str) -> LoadedNCM:
    sys.path.insert(0, str(ROOT / "vendor/ncm"))
    from tokenization.sudoku_tokenizer import SudokuTokenizer

    tokenizer = SudokuTokenizer()
    if device == "mlx":
        from ncm_mlx import MLXPredictor, mx

        return LoadedNCM(MLXPredictor(checkpoint_dir), tokenizer, {
            "backend": "mlx-metal",
            "device_info": mx.device_info(),
            "mlx_enable_tf32": "0",
            "backend_sha256": sha256_file(ROOT / "ncm_mlx.py"),
        })
    if device not in ("cpu", "mps", "cuda"):
        raise ValueError(f"Unsupported inference device {device!r}.")

    import torch
    from omegaconf import OmegaConf
    from models.next_concept import NextConceptTransformer
    from models.vqvae import VQVAE

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable on this machine.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS is unavailable on this machine.")
    torch.set_num_threads(8)
    config = OmegaConf.load(config_path)
    if dict(config.generation) != GENERATION:
        raise ValueError("The checkpoint must use the published greedy generation settings.")
    vqvae = VQVAE(**config.tokenizer.vqvae)
    vqvae.load_state_dict(torch.load(checkpoint_dir / "tokenizer.pt", map_location="cpu",
                                    weights_only=True)["model"], strict=True)
    vqvae = vqvae.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    net = NextConceptTransformer(vqvae_local=vqvae, config=config.model)
    net.load_state_dict(torch.load(checkpoint_dir / "generator.pt", map_location="cpu",
                                  weights_only=True)["model"], strict=True)
    net = net.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    if int(config.model.prefix_len) != 91 or int(config.tokenizer.vqvae.context_length) != 89:
        raise ValueError("The checkpoint does not use the expected 91-token input and 89-token output.")
    return LoadedNCM(TorchPredictor(net, vqvae, device), tokenizer,
                     {"backend": f"pytorch-{device}", "torch_threads": torch.get_num_threads(),
                      "mps_adapter_sha256": sha256_file(ROOT / "ncm_torch_ops.py") if device == "mps" else None})


def generate_batch(model: LoadedNCM, cases: Sequence[LatinCase]) -> list[tuple[int, ...]]:
    prefixes = [model.tokenizer.encode("<BOS>" + board_text(case.puzzle) + "<BOS>")
                for case in cases]
    if any(len(prefix) != 91 for prefix in prefixes):
        raise ValueError("The unchanged Sudoku tokenizer must produce exactly 91 prefix tokens.")
    predictions = model.predictor.predict(prefixes)
    return [tuple(prediction) for prediction in predictions]


def score_totals(scores: Sequence[LatinScore]) -> dict:
    count = len(scores)
    validity = ("shape_valid", "digits_valid", "rows_valid", "columns_valid",
                "clues_preserved", "sudoku_boxes_valid")
    solved = sum(score.solved for score in scores)
    givens = sum(score.given_cells_total for score in scores)
    correct = sum(score.given_cells_correct for score in scores)
    return {
        "scored": count,
        "solved": solved,
        "solve_rate": solved / count if count else None,
        "validity_rates": {key: sum(getattr(score, key) for score in scores) / count if count else None
                           for key in validity},
        "given_cells_correct": correct,
        "given_cells_total": givens,
        "given_cell_agreement": correct / givens if givens else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100, help="Number of distinct base squares, default 100.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("mlx", "cpu", "mps", "cuda"), default="mlx",
                        help="Inference backend. MLX uses the Apple Silicon GPU; CPU uses upstream PyTorch.")
    parser.add_argument("--output", type=Path, help="New directory; defaults to a timestamped directory under runs.")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/latin")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "artifacts/sudoku/hard")
    parser.add_argument("--local-checkpoint", type=Path, help="Validated local generator export directory.")
    parser.add_argument("--data", type=Path, help="Prepared Latin corpus for deterministic held-out evaluation.")
    parser.add_argument("--split", choices=("validation", "test", "benchmark"), default="test")
    parser.add_argument("--clue-count", type=int, default=30, help="Given cells for validation or test, default 30.")
    args = parser.parse_args(argv)
    if not 1 <= args.count <= SOURCE_LINES or args.batch_size < 1 or not 1 <= args.clue_count <= 81:
        parser.error(f"--count must be 1 through {SOURCE_LINES}, and --batch-size must be positive.")
    if args.local_checkpoint is not None and args.checkpoint_dir != ROOT / "artifacts/sudoku/hard":
        parser.error("Use either --checkpoint-dir or --local-checkpoint.")
    output = args.output or ROOT / "runs" / datetime.now(timezone.utc).strftime("latin-%Y%m%dT%H%M%S.%fZ")
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Output directory already exists: {output}. Choose a new --output directory.")
    started = time.perf_counter()
    summary = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "requested": args.count,
        "attempted": 0,
        "completed": 0,
        "seed": args.seed,
        "clue_counts": list(CLUE_COUNTS),
        "device": args.device,
        "dtype": "float32",
        "batch_size": args.batch_size,
        "frozen": True,
        "generation": GENERATION,
        "rules_sent_to_model": False,
        "output_repair": False,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "runner_sha256": sha256_file(Path(__file__)),
        "output_directory": str(output.resolve()),
        "rates_denominator": "completed predictions; inspect status and attempted for partial runs",
    }
    cases = ()
    scores = []
    inference_seconds = 0.0
    exit_code = 0
    try:
        write_json(output / "summary.json", summary)
        if args.device == "mps" and args.local_checkpoint is None:
            raise RuntimeError("NCM's quantizer uses nondivisible area pooling unsupported by MPS. Use --device mlx on Apple Silicon, or --device cpu.")
        checkpoint_dir = args.local_checkpoint or args.checkpoint_dir
        if args.local_checkpoint is None:
            summary.update(verify_setup(checkpoint_dir, args.device))
        else:
            summary.update(verify_local_export(checkpoint_dir, args.device))
        if args.data is None:
            source_path, manifest = source_cache(args.cache_dir)
            summary["source"] = manifest
            cases = build_cases(source_path, count=args.count, seed=args.seed)
        else:
            from latin_data import LatinCorpus
            corpus = LatinCorpus.open(args.data)
            if (args.local_checkpoint is not None
                    and summary["checkpoint"]["data"]["manifest_sha256"] != corpus.manifest_sha256):
                raise ValueError("Evaluation corpus does not match the local checkpoint's split provenance.")
            cases = corpus.evaluation_cases(args.split, args.count, seed=args.seed, clue_count=args.clue_count)
            summary["source"] = corpus.manifest
            summary["data_manifest_sha256"] = corpus.manifest_sha256
            summary["split"] = args.split
            summary["clue_counts"] = sorted({case.clue_count for case in cases})
        summary["clue_bin_counts"] = dict(sorted(Counter(case.clue_count for case in cases).items()))
        with (output / "cases.jsonl").open("w", encoding="utf-8") as stream:
            for case in cases:
                stream.write(json.dumps({**asdict(case), "clue_count": case.clue_count}) + "\n")
        summary["cases_sha256"] = sha256_file(output / "cases.jsonl")
        model = load_ncm(checkpoint_dir / "config.yaml", checkpoint_dir, args.device)
        if args.local_checkpoint is not None:
            rechecked = verify_local_export(checkpoint_dir, args.device)
            if rechecked["provenance_sha256"] != summary["provenance_sha256"]:
                raise ValueError("Local checkpoint changed while loading its weights. Retry after the trainer stops.")
        summary.update(model.metadata)
        summary["startup_seconds"] = time.perf_counter() - started
        print(f"Evaluating {len(cases)} frozen-checkpoint Latin puzzles. Results in {output}", flush=True)
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for offset in range(0, len(cases), args.batch_size):
                batch = cases[offset:offset + args.batch_size]
                summary["attempted"] += len(batch)
                batch_started = time.perf_counter()
                predictions = generate_batch(model, batch)
                if len(predictions) != len(batch):
                    raise RuntimeError("The model returned a different number of predictions than inputs.")
                batch_seconds = time.perf_counter() - batch_started
                inference_seconds += batch_seconds
                for case, raw_ids in zip(batch, predictions):
                    score = score_prediction(case, raw_ids)
                    record = {
                        "id": case.id,
                        "source_line": case.source_line,
                        "clue_count": case.clue_count,
                        "raw_ids": raw_ids,
                        "raw_text": model.tokenizer.decode(raw_ids, skip_special_tokens=False),
                        "score": {**asdict(score), "solved": score.solved},
                    }
                    stream.write(json.dumps(record) + "\n")
                    scores.append(score)
                stream.flush()
                summary["completed"] = len(scores)
                summary["metrics"] = score_totals(scores)
                summary["inference_seconds"] = inference_seconds
                write_json(output / "summary.json", summary)
                eta = (len(cases) - len(scores)) * inference_seconds / len(scores)
                print(f"Completed {len(scores)}/{len(cases)}. Solved {sum(score.solved for score in scores)}. "
                      f"Estimated remaining {eta:.1f}s from completed batches.", flush=True)
        summary["status"] = "complete"
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
        exit_code = 130
        print("Interrupted. Completed predictions are saved.", file=sys.stderr)
    except Exception as error:
        summary["status"] = "failed"
        summary["error"] = {"type": type(error).__name__, "message": str(error)}
        exit_code = 1
        print(f"Evaluation failed: {error}", file=sys.stderr)
    finally:
        summary["completed"] = len(scores)
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        summary["wall_seconds"] = time.perf_counter() - started
        summary["inference_seconds"] = inference_seconds
        summary["metrics"] = score_totals(scores)
        summary["per_clue_count"] = {
            str(clues): score_totals([score for case, score in zip(cases, scores) if case.clue_count == clues])
            for clues in summary["clue_counts"]
        }
        write_json(output / "summary.json", summary)
    print(f"Status {summary['status']}. Summary {output / 'summary.json'}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
