from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import FrozenInstanceError
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import evaluate_latin as benchmark


WITNESS = tuple((row + column) % 9 + 1 for row in range(9) for column in range(9))


def raw_tokens(cells):
    rows = [list(cells[start:start + 9]) for start in range(0, 81, 9)]
    return tuple(token for index, row in enumerate(rows)
                 for token in ([10] if index else []) + row)


def case_with_clues():
    return benchmark.LatinCase("test", 1,
                               tuple(value if value <= 7 else 0 for value in WITNESS), WITNESS)


class ScoringTests(unittest.TestCase):
    def test_accepts_valid_non_sudoku_completion(self):
        score = benchmark.score_prediction(case_with_clues(), raw_tokens(WITNESS))
        self.assertTrue(score.solved)
        self.assertTrue(score.rows_valid)
        self.assertTrue(score.columns_valid)
        self.assertTrue(score.clues_preserved)
        self.assertFalse(score.sudoku_boxes_valid)

    def test_accepts_an_alternative_valid_completion(self):
        alternative = tuple(17 - value if value in (8, 9) else value for value in WITNESS)
        self.assertNotEqual(alternative, WITNESS)
        score = benchmark.score_prediction(case_with_clues(), raw_tokens(alternative))
        self.assertTrue(score.solved)
        self.assertEqual(score.given_cells_correct, 63)

    def test_rejects_changed_clues_even_when_latin_is_valid(self):
        changed = tuple(3 - value if value in (1, 2) else value for value in WITNESS)
        score = benchmark.score_prediction(case_with_clues(), raw_tokens(changed))
        self.assertTrue(score.rows_valid)
        self.assertTrue(score.columns_valid)
        self.assertFalse(score.clues_preserved)
        self.assertFalse(score.solved)

    def test_detects_row_errors_with_valid_columns(self):
        changed = list(WITNESS)
        for row in range(9):
            changed[row * 9 + 1] = changed[row * 9]
        score = benchmark.score_prediction(case_with_clues(), raw_tokens(changed))
        self.assertFalse(score.rows_valid)
        self.assertTrue(score.columns_valid)
        self.assertFalse(score.solved)

    def test_detects_column_errors_with_valid_rows(self):
        changed = list(WITNESS)
        changed[9:18] = changed[:9]
        score = benchmark.score_prediction(case_with_clues(), raw_tokens(changed))
        self.assertTrue(score.rows_valid)
        self.assertFalse(score.columns_valid)
        self.assertFalse(score.solved)

    def test_rejects_blank_bos_separator_and_noninteger_cell_tokens(self):
        for bad_token in (0, 11, 10, -1, 12, 1.0, True):
            with self.subTest(token=bad_token):
                changed = list(raw_tokens(WITNESS))
                changed[0] = bad_token
                score = benchmark.score_prediction(case_with_clues(), changed)
                self.assertFalse(score.digits_valid)
                self.assertFalse(score.solved)

    def test_rejects_bad_separators_and_missing_or_extra_tokens(self):
        original = raw_tokens(WITNESS)
        changed = list(original)
        changed[9] = 11
        float_separator = list(original)
        float_separator[9] = 10.0
        for invalid in (changed, float_separator, original[:-1], original + (11,), (11,) + original):
            with self.subTest(length=len(invalid)):
                score = benchmark.score_prediction(case_with_clues(), invalid)
                self.assertFalse(score.shape_valid)
                self.assertFalse(score.solved)
                self.assertEqual(score.given_cells_correct, 0)

    def test_case_is_immutable_and_validates_witness_and_clues(self):
        case = case_with_clues()
        self.assertEqual(case.clue_count, 63)
        with self.assertRaises(FrozenInstanceError):
            case.source_line = 2
        with self.assertRaises(ValueError):
            benchmark.LatinCase("bad", 1, (0,) * 80, WITNESS)
        with self.assertRaises(ValueError):
            benchmark.LatinCase("bad", 1, (0,) * 81, (1,) * 81)
        with self.assertRaises(ValueError):
            benchmark.LatinCase("bad", 1, (9,) + (0,) * 80, WITNESS)
        with self.assertRaises(ValueError):
            benchmark.LatinCase("bad", 1, list(case.puzzle), WITNESS)

    def test_main_metric_counts_failed_predictions(self):
        case = case_with_clues()
        scores = [benchmark.score_prediction(case, raw_tokens(WITNESS)),
                  benchmark.score_prediction(case, (11,))]
        totals = benchmark.score_totals(scores)
        self.assertEqual(totals["scored"], 2)
        self.assertEqual(totals["solved"], 1)
        self.assertEqual(totals["solve_rate"], 0.5)
        self.assertEqual(totals["given_cell_agreement"], 0.5)


class DatasetTests(unittest.TestCase):
    def test_source_mapping_and_validation(self):
        source = " ".join("".join(str(value - 1) for value in WITNESS[start:start + 9])
                          for start in range(0, 81, 9))
        self.assertEqual(benchmark.parse_source_line(source, 1), WITNESS)
        for invalid in (source.replace("0", "9"), "000000000 " * 9, source[:20]):
            with self.assertRaises(ValueError):
                benchmark.parse_source_line(invalid, 1)

    def test_deterministic_distinct_cases_bins_and_retained_conflict(self):
        rng = random.Random(8675309)
        witnesses = set()
        while len(witnesses) < 128:
            rows = rng.sample(range(9), 9)
            witnesses.add(tuple((row + column) % 9 for row in rows for column in range(9)))
        lines = [" ".join("".join(str(value) for value in cells[start:start + 9])
                          for start in range(0, 81, 9)) for cells in sorted(witnesses)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.txt"
            path.write_text("\n".join(lines) + "\n", encoding="ascii")
            first = benchmark.build_cases(path, count=100, seed=42)
            second = benchmark.build_cases(path, count=100, seed=42)
            different = benchmark.build_cases(path, count=100, seed=43)
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)
        self.assertEqual(len({case.witness for case in first}), 100)
        self.assertEqual(len({case.source_line for case in first}), 100)
        self.assertEqual(Counter(case.clue_count for case in first), {25: 25, 37: 25, 49: 25, 61: 25})
        for case in first:
            self.assertTrue(benchmark.box_conflicts(case.puzzle))
            self.assertTrue(benchmark.score_prediction(case, raw_tokens(case.witness)).solved)
            for first_cell, second_cell in benchmark.box_conflicts(case.puzzle):
                self.assertNotEqual(first_cell // 9, second_cell // 9)
                self.assertNotEqual(first_cell % 9, second_cell % 9)

    def test_rejects_tampered_source_cache_without_a_network_request(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            path = cache_dir / "latin_is9nt.1.first4096.txt"
            path.write_text("tampered\n", encoding="ascii")
            manifest = {"url": benchmark.SOURCE_URL, "line_count": benchmark.SOURCE_LINES,
                        "sha256": "0" * 64}
            (cache_dir / "latin_is9nt.1.first4096.json").write_text(json.dumps(manifest))
            with patch("urllib.request.urlopen") as network:
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    benchmark.source_cache(cache_dir)
                network.assert_not_called()

    def test_rejects_incomplete_cache_without_a_network_request(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            (cache_dir / "latin_is9nt.1.first4096.txt").write_text("partial\n")
            with patch("urllib.request.urlopen") as network:
                with self.assertRaisesRegex(ValueError, "Incomplete source cache"):
                    benchmark.source_cache(cache_dir)
                network.assert_not_called()


class CommandTests(unittest.TestCase):
    def test_import_does_not_load_inference_libraries(self):
        result = subprocess.run([sys.executable, "-c",
                                 "import evaluate_latin, sys; assert not {'torch', 'mlx.core'} & sys.modules.keys()"],
                                cwd=benchmark.ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_setup_failure_writes_explicit_partial_status(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failed-run"
            with patch.object(benchmark, "verify_setup", side_effect=RuntimeError("missing test checkpoint")), \
                    patch.object(benchmark, "source_cache") as cache, \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = benchmark.main(["--output", str(output), "--device", "cpu"])
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(result, 1)
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["attempted"], 0)
            self.assertEqual(summary["completed"], 0)
            self.assertEqual(summary["error"]["message"], "missing test checkpoint")
            cache.assert_not_called()

    def test_short_backend_output_fails_instead_of_completing(self):
        case = case_with_clues()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "short-run"
            with patch.object(benchmark, "verify_setup", return_value={}), \
                    patch.object(benchmark, "source_cache", return_value=(Path("unused"), {})), \
                    patch.object(benchmark, "build_cases", return_value=(case,)), \
                    patch.object(benchmark, "load_ncm", return_value=benchmark.LoadedNCM(None, None, {})), \
                    patch.object(benchmark, "generate_batch", return_value=[]), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = benchmark.main(["--count", "1", "--output", str(output)])
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(result, 1)
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["attempted"], 1)
            self.assertEqual(summary["completed"], 0)
            self.assertIn("different number of predictions", summary["error"]["message"])

    def test_interruption_preserves_completed_predictions(self):
        first = case_with_clues()
        second = benchmark.LatinCase("second", 2, first.puzzle, first.witness)
        tokenizer = SimpleNamespace(decode=lambda *_args, **_kwargs: "raw text")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "interrupted-run"
            with patch.object(benchmark, "verify_setup", return_value={}), \
                    patch.object(benchmark, "source_cache", return_value=(Path("unused"), {})), \
                    patch.object(benchmark, "build_cases", return_value=(first, second)), \
                    patch.object(benchmark, "load_ncm", return_value=benchmark.LoadedNCM(None, tokenizer, {})), \
                    patch.object(benchmark, "generate_batch", side_effect=[[raw_tokens(WITNESS)], KeyboardInterrupt]), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = benchmark.main(["--count", "2", "--batch-size", "1", "--output", str(output)])
            summary = json.loads((output / "summary.json").read_text())
            predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(result, 130)
            self.assertEqual(summary["status"], "interrupted")
            self.assertEqual(summary["attempted"], 2)
            self.assertEqual(summary["completed"], 1)
            self.assertEqual(summary["metrics"]["solved"], 1)
            self.assertEqual(len(predictions), 1)
            self.assertEqual(predictions[0]["raw_ids"], list(raw_tokens(WITNESS)))

    def test_decode_interruption_counts_only_persisted_predictions(self):
        first = case_with_clues()
        second = benchmark.LatinCase("second", 2, first.puzzle, first.witness)
        tokenizer = SimpleNamespace(decode=Mock(side_effect=["first raw text", KeyboardInterrupt]))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "decode-interrupted-run"
            with patch.object(benchmark, "verify_setup", return_value={}), \
                    patch.object(benchmark, "source_cache", return_value=(Path("unused"), {})), \
                    patch.object(benchmark, "build_cases", return_value=(first, second)), \
                    patch.object(benchmark, "load_ncm", return_value=benchmark.LoadedNCM(None, tokenizer, {})), \
                    patch.object(benchmark, "generate_batch", return_value=[raw_tokens(WITNESS), raw_tokens(WITNESS)]), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = benchmark.main(["--count", "2", "--batch-size", "2", "--output", str(output)])
            summary = json.loads((output / "summary.json").read_text())
            predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(result, 130)
            self.assertEqual(summary["status"], "interrupted")
            self.assertEqual(summary["attempted"], 2)
            self.assertEqual(summary["completed"], 1)
            self.assertEqual(summary["metrics"]["scored"], 1)
            self.assertEqual(summary["metrics"]["solved"], 1)
            self.assertEqual(len(predictions), 1)
            self.assertEqual(predictions[0]["id"], first.id)
            self.assertEqual(predictions[0]["raw_ids"], list(raw_tokens(WITNESS)))
            self.assertEqual(predictions[0]["raw_text"], "first raw text")

    def test_default_mlx_backend_records_metadata_and_predictions(self):
        case = case_with_clues()
        tokenizer = SimpleNamespace(decode=lambda *_args, **_kwargs: "raw text")
        model = benchmark.LoadedNCM(None, tokenizer, {"backend": "mlx-metal", "device_info": {"device_name": "test GPU"}})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "mlx-run"
            with patch.object(benchmark, "verify_setup", return_value={}) as setup, \
                    patch.object(benchmark, "source_cache", return_value=(Path("unused"), {})), \
                    patch.object(benchmark, "build_cases", return_value=(case,)), \
                    patch.object(benchmark, "load_ncm", return_value=model) as load, \
                    patch.object(benchmark, "generate_batch", return_value=[raw_tokens(WITNESS)]), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = benchmark.main(["--count", "1", "--output", str(output)])
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(result, 0)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["device"], "mlx")
            self.assertEqual(summary["backend"], "mlx-metal")
            self.assertEqual(summary["device_info"], {"device_name": "test GPU"})
            self.assertEqual(summary["metrics"]["solved"], 1)
            self.assertEqual(setup.call_args.args[1], "mlx")
            self.assertEqual(load.call_args.args[2], "mlx")

    def test_mps_failure_suggests_the_mlx_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "mps-run"
            with patch.object(benchmark, "verify_setup") as setup, \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = benchmark.main(["--device", "mps", "--output", str(output)])
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(result, 1)
            self.assertEqual(summary["status"], "failed")
            self.assertIn("Use --device mlx", summary["error"]["message"])
            self.assertEqual(summary["completed"], 0)
            setup.assert_not_called()


class BackendTests(unittest.TestCase):
    def test_batch_uses_the_original_prompt_and_returns_raw_tokens(self):
        prefix = [11] + [1] * 89 + [11]
        tokenizer = SimpleNamespace(encode=Mock(return_value=prefix))
        predictor = SimpleNamespace(predict=Mock(return_value=[list(raw_tokens(WITNESS))]))
        model = benchmark.LoadedNCM(predictor, tokenizer, {})
        predictions = benchmark.generate_batch(model, [case_with_clues()])
        self.assertEqual(predictions, [raw_tokens(WITNESS)])
        tokenizer.encode.assert_called_once_with(
            "<BOS>1234567__|234567__1|34567__12|4567__123|567__1234|67__12345|7__123456|__1234567|_1234567_<BOS>")
        predictor.predict.assert_called_once_with([prefix])

    def test_bad_prefix_length_fails_before_inference(self):
        tokenizer = SimpleNamespace(encode=Mock(return_value=[11, 1, 11]))
        predictor = SimpleNamespace(predict=Mock())
        with self.assertRaisesRegex(ValueError, "exactly 91 prefix tokens"):
            benchmark.generate_batch(benchmark.LoadedNCM(predictor, tokenizer, {}), [case_with_clues()])
        predictor.predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
