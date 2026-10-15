import dataclasses
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from evaluate_latin import build_cases, latin_columns_valid, latin_rows_valid, score_prediction
from latin_data import CELL_POSITIONS, LatinCorpus, SourceArchive, prepare_corpus


class LatinDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.output = self.root / "corpus"
        rng = np.random.default_rng(915)
        base = (np.arange(9)[:, None] + np.arange(9)[None, :]) % 9
        self.boards = np.array([
            rng.permutation(np.arange(1, 10, dtype=np.uint8))[base[rng.permutation(9)][:, rng.permutation(9)]]
            for _ in range(30)
        ], dtype=np.uint8).reshape(30, 81)
        self.sources = self._write_sources(self.boards)

    def _write_sources(self, boards):
        sources = []
        for index, chunk in enumerate((boards[:14], boards[14:]), 1):
            payload = "".join(" ".join("".join(str(int(cell) - 1) for cell in row)
                                       for row in board.reshape(9, 9)) + " \n" for board in chunk).encode("ascii")
            compressed = gzip.compress(payload, mtime=0)
            name = f"fixture{index}.txt.gz"
            (self.cache / name).write_bytes(compressed)
            sources.append(SourceArchive(name, f"https://example.invalid/{name}", len(compressed),
                                          hashlib.sha256(compressed).hexdigest(), len(chunk)))
        return tuple(sources)

    def prepare(self, **overrides):
        options = {"sources": self.sources, "validation_count": 3, "test_count": 4,
                   "benchmark_count": 4, "benchmark_source_lines": 12}
        options.update(overrides)
        with patch("urllib.request.urlopen", side_effect=AssertionError("Fixture preparation must stay offline.")):
            return prepare_corpus(self.output, self.cache, **options)

    def test_preparation_reserves_all_splits_before_augmentation(self):
        corpus = self.prepare()
        self.assertEqual(corpus.manifest["split_counts"],
                         {"train": 19, "validation": 3, "test": 4, "benchmark": 4})
        np.testing.assert_array_equal(corpus.boards, self.boards)
        self.assertIsInstance(corpus.boards, np.memmap)
        self.assertFalse(corpus.boards.flags.writeable)
        combined = np.concatenate(list(corpus.splits.values()))
        self.assertEqual(sorted(combined.tolist()), list(range(30)))
        self.assertEqual(len(set(combined.tolist())), 30)
        self.assertEqual(corpus.manifest["unique_board_count"], 30)

    def test_historical_masks_equal_original_evaluator(self):
        corpus = self.prepare()
        prefix_path = self.root / "prefix.txt"
        payload = gzip.decompress((self.cache / self.sources[0].name).read_bytes())
        prefix_path.write_bytes(b"".join(payload.splitlines(keepends=True)[:12]))
        expected = build_cases(prefix_path, count=4, seed=42)
        self.assertEqual(corpus.evaluation_cases("benchmark", 4, seed=999, clue_count=1), expected)
        self.assertEqual([case.clue_count for case in expected], [25, 37, 49, 61])

    def test_repeat_preparation_validates_and_reuses(self):
        first = self.prepare()
        before = {path.name: (path.stat().st_mtime_ns, path.read_bytes()) for path in self.output.iterdir()}
        second = self.prepare()
        self.assertEqual(first.manifest_sha256, second.manifest_sha256)
        after = {path.name: (path.stat().st_mtime_ns, path.read_bytes()) for path in self.output.iterdir()}
        self.assertEqual(before, after)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.prepare(seed=73)

    def test_corrupted_existing_corpus_is_rejected_without_replacement(self):
        self.prepare()
        path = self.output / "boards.npy"
        damaged = bytearray(path.read_bytes())
        damaged[-1] ^= 1
        path.write_bytes(damaged)
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.prepare()
        self.assertEqual(path.read_bytes(), damaged)

    def test_incomplete_output_is_rejected(self):
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.prepare()
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_bad_cached_source_does_not_create_output(self):
        path = self.cache / self.sources[0].name
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_invalid_latin_record_does_not_publish_partial_output(self):
        boards = self.boards.copy()
        boards[8, 0] = boards[8, 1]
        self.sources = self._write_sources(boards)
        with self.assertRaisesRegex(ValueError, "Invalid Latin square"):
            self.prepare()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".corpus.prepare-*")), [])

    def test_duplicate_boards_are_rejected(self):
        boards = self.boards.copy()
        boards[-1] = boards[0]
        self.sources = self._write_sources(boards)
        with self.assertRaisesRegex(ValueError, "1 duplicate"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_archive_record_count_must_match_pinned_count(self):
        self.sources = (dataclasses.replace(self.sources[0], count=15), self.sources[1])
        with self.assertRaisesRegex(ValueError, "fewer than 15"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_stream_uses_expected_tokens_and_exactly_thirty_clues(self):
        corpus = self.prepare()
        batch = corpus.training_stream().next_batch(7)
        self.assertEqual(batch.solutions.shape, (7, 89))
        self.assertEqual(batch.prefixes.shape, (7, 91))
        self.assertEqual(batch.base_ids.dtype, np.uint32)
        self.assertEqual(batch.solutions.dtype, np.uint8)
        self.assertEqual(batch.prefixes.dtype, np.uint8)
        np.testing.assert_array_equal(batch.prefixes[:, [0, 90]], np.full((7, 2), 11, dtype=np.uint8))
        np.testing.assert_array_equal(batch.solutions[:, list(range(9, 80, 10))], np.full((7, 8), 10))
        np.testing.assert_array_equal(batch.prefixes[:, list(range(10, 81, 10))], np.full((7, 8), 10))
        for solution, prefix in zip(batch.solutions, batch.prefixes):
            cells = tuple(int(value) for value in solution[CELL_POSITIONS])
            givens = prefix[CELL_POSITIONS + 1]
            self.assertTrue(latin_rows_valid(cells))
            self.assertTrue(latin_columns_valid(cells))
            self.assertEqual(np.count_nonzero(givens), 30)
            np.testing.assert_array_equal(givens[givens != 0], np.array(cells)[givens != 0])
        self.assertEqual(len(set(batch.base_ids.tolist())), 7)
        self.assertTrue(set(batch.base_ids.tolist()).issubset(set(corpus.splits["train"].tolist())))

    def test_training_samples_every_base_once_per_epoch(self):
        corpus = self.prepare()
        stream = corpus.training_stream(augment=False)
        first = stream.next_batch(19)
        second = stream.next_batch(19)
        self.assertEqual(sorted(first.base_ids.tolist()), sorted(corpus.splits["train"].tolist()))
        self.assertEqual(sorted(second.base_ids.tolist()), sorted(corpus.splits["train"].tolist()))
        self.assertEqual(stream.state_dict()["epoch"], 2)
        self.assertEqual(stream.state_dict()["cursor"], 0)
        for board, base_id in zip(first.solutions, first.base_ids):
            np.testing.assert_array_equal(board[CELL_POSITIONS], self.boards[base_id])

    def test_resume_replays_augmentation_and_masks_across_epoch_boundary(self):
        corpus = self.prepare()
        original = corpus.training_stream(seed=19)
        original.next_batch(17)
        saved = json.loads(json.dumps(original.state_dict()))
        expected = original.next_batch(25)
        resumed = corpus.training_stream(seed=19)
        resumed.load_state_dict(saved)
        actual = resumed.next_batch(25)
        np.testing.assert_array_equal(actual.base_ids, expected.base_ids)
        np.testing.assert_array_equal(actual.solutions, expected.solutions)
        np.testing.assert_array_equal(actual.prefixes, expected.prefixes)
        self.assertEqual(resumed.state_dict(), original.state_dict())
        self.assertEqual(resumed.state_dict()["epoch"], 2)
        self.assertEqual(resumed.state_dict()["cursor"], 4)

    def test_batch_partition_does_not_change_the_training_sequence(self):
        corpus = self.prepare()
        whole = corpus.training_stream(seed=3).next_batch(27)
        stream = corpus.training_stream(seed=3)
        chunks = [stream.next_batch(size) for size in (4, 13, 10)]
        for field in ("base_ids", "solutions", "prefixes"):
            np.testing.assert_array_equal(np.concatenate([getattr(chunk, field) for chunk in chunks]),
                                          getattr(whole, field))

    def test_evaluation_is_repeatable_and_does_not_advance_training(self):
        corpus = self.prepare()
        stream = corpus.training_stream()
        before = stream.state_dict()
        validation = corpus.evaluation_cases("validation", 3)
        test = corpus.evaluation_cases("test", 4)
        self.assertEqual(validation, corpus.evaluation_cases("validation", 3))
        self.assertEqual(test, corpus.evaluation_cases("test", 4))
        self.assertEqual(stream.state_dict(), before)
        self.assertEqual({case.source_line - 1 for case in validation}, set(corpus.splits["validation"].tolist()))
        self.assertEqual({case.source_line - 1 for case in test}, set(corpus.splits["test"].tolist()))
        for case in validation + test:
            self.assertEqual(case.clue_count, 30)
            raw_ids = np.full(89, 10, dtype=np.uint8)
            raw_ids[CELL_POSITIONS] = case.witness
            self.assertTrue(score_prediction(case, raw_ids.tolist()).solved)

    def test_sampler_rejects_incompatible_state_without_mutating(self):
        corpus = self.prepare()
        stream = corpus.training_stream()
        before = stream.state_dict()
        for key, value in (("manifest_sha256", "bad"), ("clue_count", 31), ("epoch", -1), ("cursor", 19)):
            state = dict(before)
            state[key] = value
            with self.assertRaises(ValueError):
                stream.load_state_dict(state)
            self.assertEqual(stream.state_dict(), before)

    def test_evaluation_and_stream_validate_boundaries(self):
        corpus = self.prepare()
        with self.assertRaises(ValueError):
            corpus.training_stream(clue_count=82)
        with self.assertRaises(ValueError):
            corpus.training_stream().next_batch(0)
        with self.assertRaises(ValueError):
            corpus.evaluation_cases("train", 1)
        with self.assertRaises(ValueError):
            corpus.evaluation_cases("validation", 4)


if __name__ == "__main__":
    unittest.main()
