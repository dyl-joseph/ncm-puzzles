"""Compact Latin-square corpus and replayable NCM training batches."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gzip
import json
from pathlib import Path
import tempfile
import urllib.request

import numpy as np

from evaluate_latin import LatinCase, build_cases, sha256_file


FORMAT_VERSION = 1
CELL_POSITIONS = np.array([index for index in range(89) if index % 10 != 9])
SPLIT_NAMES = ("train", "validation", "test", "benchmark")
SOURCE_BIAS = (
    "McKay's order-9 isotopy representatives with nontrivial symmetry. "
    "This collection does not contain all order-9 Latin squares."
)


@dataclass(frozen=True)
class SourceArchive:
    name: str
    url: str
    bytes: int
    sha256: str
    count: int


SOURCES = (
    SourceArchive("latin_is9nt.1.txt.gz", "https://users.cecs.anu.edu.au/~bdm/data/latin_is9nt.1.txt.gz",
                  28326031, "d6ce047eb0db2648bd89c0d202e985de164c274310dea0a6996659c1bc87cec3", 800000),
    SourceArchive("latin_is9nt.2.txt.gz", "https://users.cecs.anu.edu.au/~bdm/data/latin_is9nt.2.txt.gz",
                  28310242, "c11f0768375129fd26a4d02090292f74097512c7172faa6e25d5f8e4b72e9575", 800000),
    SourceArchive("latin_is9nt.3.txt.gz", "https://users.cecs.anu.edu.au/~bdm/data/latin_is9nt.3.txt.gz",
                  27892533, "5ef66f4f39222b1e7e938a723ab84c7d4014a35a5534f086f0b2c83e8c51a61d", 793407),
)


@dataclass(frozen=True)
class LatinBatch:
    solutions: np.ndarray
    prefixes: np.ndarray
    base_ids: np.ndarray

    def __post_init__(self) -> None:
        size = len(self.base_ids)
        if (self.solutions.dtype != np.uint8 or self.solutions.shape != (size, 89)
                or self.prefixes.dtype != np.uint8 or self.prefixes.shape != (size, 91)
                or self.base_ids.dtype != np.uint32 or self.base_ids.shape != (size,)):
            raise ValueError("LatinBatch needs uint8[B,89] solutions, uint8[B,91] prefixes and uint32[B] base IDs.")


def _positive_integer(value: int, name: str, *, maximum: int | None = None) -> None:
    if type(value) is not int or value < 1 or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be an integer from 1 through {maximum or 'the available count'}.")


def _json_write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cached_archive(cache_dir: Path, source: SourceArchive) -> Path:
    path = cache_dir / source.name
    if path.exists():
        if not path.is_file() or path.stat().st_size != source.bytes or sha256_file(path) != source.sha256:
            raise ValueError(f"Source archive checksum mismatch at {path}. Use a fresh cache directory.")
        return path
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            request = urllib.request.Request(source.url, headers={"User-Agent": "ncm-latin-training/1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    temporary.write(chunk)
            temporary.flush()
            if temporary_path.stat().st_size != source.bytes or sha256_file(temporary_path) != source.sha256:
                raise ValueError(f"Downloaded source archive checksum mismatch for {source.name}.")
            if path.exists():
                raise ValueError(f"Source cache appeared during download at {path}. Retry preparation.")
            temporary_path.rename(path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return path


def _copy_archive(path: Path, boards: np.ndarray, offset: int, count: int) -> None:
    with gzip.open(path, "rb") as stream:
        for start in range(0, count, 16384):
            size = min(16384, count - start)
            payload = stream.read(size * 91)
            if len(payload) != size * 91:
                raise ValueError(f"Source archive {path.name} has fewer than {count} complete records.")
            records = np.frombuffer(payload, dtype=np.uint8).reshape(size, 91)
            rows = records[:, :90].reshape(size, 9, 10)
            digits = rows[:, :, :9]
            if (np.any(records[:, 90] != 10) or np.any(rows[:, :, 9] != 32)
                    or np.any(digits < 48) or np.any(digits > 56)):
                raise ValueError(f"Malformed source record in {path.name} near line {start + 1}.")
            cells = digits - np.uint8(47)
            target = np.arange(1, 10, dtype=np.uint8)
            if (np.any(np.sort(cells, axis=2) != target)
                    or np.any(np.sort(cells, axis=1) != target[None, :, None])):
                raise ValueError(f"Invalid Latin square in {path.name} near line {start + 1}.")
            boards[offset + start:offset + start + size] = cells.reshape(size, 81)
        if stream.read(1):
            raise ValueError(f"Source archive {path.name} has more than {count} records.")


def _historical_cases(boards: np.ndarray, count: int, source_lines: int, directory: Path) -> tuple[LatinCase, ...]:
    path = directory / "historical-prefix.txt"
    with path.open("w", encoding="ascii") as stream:
        for board in boards[:source_lines]:
            rows = board.reshape(9, 9)
            stream.write(" ".join("".join(str(int(cell) - 1) for cell in row) for row in rows) + " \n")
    try:
        return build_cases(path, count=count, seed=42)
    finally:
        path.unlink()


def _validate_splits(splits: dict[str, np.ndarray], count: int) -> None:
    if set(splits) != set(SPLIT_NAMES):
        raise ValueError("Corpus must contain train, validation, test and benchmark splits.")
    if any(ids.dtype != np.uint32 or ids.ndim != 1 for ids in splits.values()):
        raise ValueError("Every split must be a uint32 vector.")
    combined = np.concatenate([splits[name] for name in SPLIT_NAMES])
    if len(combined) != count or not np.array_equal(np.sort(combined), np.arange(count, dtype=np.uint32)):
        raise ValueError("Split indices must cover every source board exactly once.")


def prepare_corpus(directory: Path, cache_dir: Path, *, sources: tuple[SourceArchive, ...] = SOURCES,
                   validation_count: int = 2000, test_count: int = 2000,
                   benchmark_count: int = 100, benchmark_source_lines: int = 4096,
                   seed: int = 42) -> "LatinCorpus":
    directory, cache_dir = Path(directory), Path(cache_dir)
    total = sum(source.count for source in sources)
    for value, name in ((validation_count, "validation_count"), (test_count, "test_count"),
                        (benchmark_count, "benchmark_count"), (benchmark_source_lines, "benchmark_source_lines")):
        _positive_integer(value, name, maximum=total)
    if validation_count + test_count + benchmark_count >= total:
        raise ValueError("Holdouts must leave at least one training board.")
    if type(seed) is not int or seed < 0:
        raise ValueError("Split seed must be a nonnegative integer.")
    config = {"format_version": FORMAT_VERSION, "sources": [asdict(source) for source in sources],
              "validation_count": validation_count, "test_count": test_count,
              "benchmark_count": benchmark_count, "benchmark_source_lines": benchmark_source_lines,
              "split_seed": seed, "historical_seed": 42}
    if directory.exists():
        corpus = LatinCorpus.open(directory)
        if corpus.manifest["preparation"] != config:
            raise ValueError(f"Existing corpus at {directory} has an incompatible preparation configuration.")
        return corpus
    archives = [_cached_archive(cache_dir, source) for source in sources]
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{directory.name}.prepare-", dir=directory.parent) as temporary:
        staging = Path(temporary) / "corpus"
        staging.mkdir()
        boards = np.lib.format.open_memmap(staging / "boards.npy", mode="w+", dtype=np.uint8, shape=(total, 81))
        offset = 0
        for source, path in zip(sources, archives):
            _copy_archive(path, boards, offset, source.count)
            offset += source.count
        unique_count = len(np.unique(boards.view("V81").reshape(-1)))
        if unique_count != total:
            raise ValueError(f"Source corpus contains {total - unique_count} duplicate boards.")
        benchmark = _historical_cases(boards, benchmark_count, benchmark_source_lines, staging)
        benchmark_ids = np.array([case.source_line - 1 for case in benchmark], dtype=np.uint32)
        remaining = np.setdiff1d(np.arange(total, dtype=np.uint32), benchmark_ids, assume_unique=True)
        shuffled = np.random.default_rng(seed).permutation(remaining)
        splits = {"validation": shuffled[:validation_count],
                  "test": shuffled[validation_count:validation_count + test_count],
                  "train": shuffled[validation_count + test_count:], "benchmark": benchmark_ids}
        _validate_splits(splits, total)
        boards.flush()
        del boards
        for name, ids in splits.items():
            np.save(staging / f"{name}.npy", ids, allow_pickle=False)
        _json_write(staging / "benchmark.json", [asdict(case) for case in benchmark])
        files = {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                 for path in sorted(staging.iterdir())}
        manifest = {"format_version": FORMAT_VERSION, "preparation": config,
                    "source_bias": SOURCE_BIAS, "board_count": total, "unique_board_count": unique_count,
                    "board_dtype": "uint8", "board_shape": [total, 81],
                    "split_counts": {name: len(splits[name]) for name in SPLIT_NAMES},
                    "validation": {"every_row_and_column": True, "disjoint_split_coverage": True},
                    "files": files, "prepared_file_bytes": sum(item["bytes"] for item in files.values()),
                    "compressed_source_bytes": sum(source.bytes for source in sources),
                    "expanded_source_bytes": total * 91}
        _json_write(staging / "manifest.json", manifest)
        if directory.exists():
            raise ValueError(f"Output directory appeared during preparation at {directory}. Retry to validate it.")
        staging.rename(directory)
    return LatinCorpus.open(directory)


@dataclass(frozen=True)
class LatinCorpus:
    directory: Path
    manifest: dict
    manifest_sha256: str
    boards: np.ndarray
    splits: dict[str, np.ndarray]
    benchmark: tuple[LatinCase, ...]

    @classmethod
    def open(cls, directory: Path) -> "LatinCorpus":
        directory = Path(directory)
        try:
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            expected_files = {"boards.npy", "benchmark.json", *(f"{name}.npy" for name in SPLIT_NAMES)}
            if manifest["format_version"] != FORMAT_VERSION or set(manifest["files"]) != expected_files:
                raise ValueError("Unsupported corpus format or file inventory.")
            for name, record in manifest["files"].items():
                path = directory / name
                if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
                    raise ValueError(f"Corpus checksum mismatch for {name}.")
            count = manifest["board_count"]
            boards = np.load(directory / "boards.npy", mmap_mode="r", allow_pickle=False)
            if boards.dtype != np.uint8 or boards.shape != (count, 81):
                raise ValueError("Corpus boards must be uint8[N,81].")
            splits = {name: np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                      for name in SPLIT_NAMES}
            _validate_splits(splits, count)
            if manifest["split_counts"] != {name: len(ids) for name, ids in splits.items()}:
                raise ValueError("Corpus split counts do not match the manifest.")
            records = json.loads((directory / "benchmark.json").read_text(encoding="utf-8"))
            benchmark = tuple(LatinCase(record["id"], record["source_line"], tuple(record["puzzle"]),
                                        tuple(record["witness"])) for record in records)
            if (len(benchmark) != len(splits["benchmark"])
                    or any(case.source_line - 1 != int(base_id)
                           or tuple(int(value) for value in boards[base_id]) != case.witness
                           for case, base_id in zip(benchmark, splits["benchmark"]))):
                raise ValueError("Historical benchmark does not match its reserved source identities.")
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Incomplete or malformed Latin corpus at {directory}. {error}") from error
        return cls(directory, manifest, sha256_file(directory / "manifest.json"), boards, splits, benchmark)

    def training_stream(self, seed: int = 42, clue_count: int = 30, augment: bool = True) -> "LatinStream":
        return LatinStream(self, seed=seed, clue_count=clue_count, augment=augment)

    def evaluation_cases(self, split: str, count: int, seed: int = 42,
                         clue_count: int = 30) -> tuple[LatinCase, ...]:
        if split not in ("validation", "test", "benchmark"):
            raise ValueError("Evaluation split must be validation, test or benchmark.")
        _positive_integer(count, "count", maximum=len(self.splits[split]))
        if split == "benchmark":
            return self.benchmark[:count]
        _positive_integer(clue_count, "clue_count", maximum=81)
        if type(seed) is not int or seed < 0:
            raise ValueError("Evaluation seed must be a nonnegative integer.")
        rng = np.random.default_rng(np.random.SeedSequence([seed, 1 if split == "validation" else 2]))
        ids = rng.permutation(self.splits[split])[:count]
        cases = []
        for base_id in ids:
            solution, puzzle = _masked_board(self.boards[base_id], rng, clue_count, augment=True)
            cases.append(LatinCase(f"latin9-{split}-base{int(base_id)}-seed{seed}-clues{clue_count}",
                                   int(base_id) + 1, tuple(int(value) for value in puzzle),
                                   tuple(int(value) for value in solution)))
        return tuple(cases)


def _masked_board(board: np.ndarray, rng: np.random.Generator, clue_count: int,
                  augment: bool) -> tuple[np.ndarray, np.ndarray]:
    if augment:
        board = board.reshape(9, 9)[rng.permutation(9)][:, rng.permutation(9)]
        board = rng.permutation(np.arange(1, 10, dtype=np.uint8))[board.reshape(81) - 1]
    else:
        board = board.copy()
    puzzle = np.zeros(81, dtype=np.uint8)
    clues = rng.choice(81, clue_count, replace=False)
    puzzle[clues] = board[clues]
    return board, puzzle


class LatinStream:
    def __init__(self, corpus: LatinCorpus, *, seed: int = 42, clue_count: int = 30, augment: bool = True):
        _positive_integer(clue_count, "clue_count", maximum=81)
        if type(seed) is not int or seed < 0 or type(augment) is not bool:
            raise ValueError("Training seed must be nonnegative and augment must be a boolean.")
        if len(corpus.splits["train"]) == 0:
            raise ValueError("Training split is empty.")
        self.corpus, self.seed, self.clue_count, self.augment = corpus, seed, clue_count, augment
        self.epoch, self.cursor = 0, 0
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))
        self.order = self._epoch_order()

    def _epoch_order(self) -> np.ndarray:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, 4]))
        return rng.permutation(self.corpus.splits["train"])

    def next_batch(self, count: int) -> LatinBatch:
        _positive_integer(count, "count")
        selections = []
        remaining = count
        while remaining:
            available = min(remaining, len(self.order) - self.cursor)
            selections.append(self.order[self.cursor:self.cursor + available])
            self.cursor += available
            remaining -= available
            if self.cursor == len(self.order):
                self.epoch += 1
                self.cursor = 0
                self.order = self._epoch_order()
        base_ids = np.concatenate(selections).astype(np.uint32, copy=False)
        solutions = np.full((count, 89), 10, dtype=np.uint8)
        prefixes = np.full((count, 91), 11, dtype=np.uint8)
        prefixes[:, 1:90] = 10
        for index, base_id in enumerate(base_ids):
            board, puzzle = _masked_board(self.corpus.boards[base_id], self.rng, self.clue_count, self.augment)
            solutions[index, CELL_POSITIONS] = board
            prefixes[index, CELL_POSITIONS + 1] = puzzle
        return LatinBatch(solutions, prefixes, base_ids)

    def state_dict(self) -> dict:
        return {"version": 1, "manifest_sha256": self.corpus.manifest_sha256,
                "seed": self.seed, "clue_count": self.clue_count, "augment": self.augment,
                "epoch": self.epoch, "cursor": self.cursor,
                "rng": json.loads(json.dumps(self.rng.bit_generator.state))}

    def load_state_dict(self, state: dict) -> None:
        expected = {"version": 1, "manifest_sha256": self.corpus.manifest_sha256,
                    "seed": self.seed, "clue_count": self.clue_count, "augment": self.augment}
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError("Sampler state does not match this corpus and training configuration.")
        epoch, cursor = state.get("epoch"), state.get("cursor")
        if (type(epoch) is not int or epoch < 0 or type(cursor) is not int
                or not 0 <= cursor < len(self.corpus.splits["train"])):
            raise ValueError("Sampler epoch or cursor is invalid.")
        restored_rng = np.random.default_rng()
        try:
            restored_rng.bit_generator.state = state["rng"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Sampler RNG state is invalid.") from error
        self.epoch, self.cursor, self.rng = epoch, cursor, restored_rng
        self.order = self._epoch_order()
