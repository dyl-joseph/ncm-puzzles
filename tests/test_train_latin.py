from contextlib import redirect_stderr, redirect_stdout
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

import evaluate_latin as evaluation
from latin_data import SourceArchive, prepare_corpus
import train_latin as training


def assert_tree_equal(test, first, second, path="state"):
    test.assertEqual(type(first), type(second), path)
    if isinstance(first, torch.Tensor):
        test.assertTrue(torch.equal(first, second), path)
    elif isinstance(first, dict):
        test.assertEqual(first.keys(), second.keys(), path)
        for key in first:
            assert_tree_equal(test, first[key], second[key], f"{path}.{key}")
    elif isinstance(first, (list, tuple)):
        test.assertEqual(len(first), len(second), path)
        for index, (a, b) in enumerate(zip(first, second)):
            assert_tree_equal(test, a, b, f"{path}[{index}]")
    else:
        test.assertEqual(first, second, path)


class ScheduleTests(unittest.TestCase):
    def test_warmup_peak_floor_and_intended_schedule(self):
        spec = training.StageSpec("generator", 10, 4, 2, 0.001, 0.0001, 2)
        self.assertEqual(training.learning_rate(spec, 1), 0.0005)
        self.assertEqual(training.learning_rate(spec, 2), 0.001)
        self.assertAlmostEqual(training.learning_rate(spec, 6), 0.00055)
        self.assertEqual(training.learning_rate(spec, 10), 0.0001)
        with self.assertRaises(ValueError):
            training.learning_rate(spec, 0)

    def test_tokenizer_rejects_accumulation(self):
        with self.assertRaisesRegex(ValueError, "physical global batch"):
            training.StageSpec("tokenizer", 10000, 4, 2, 5e-4, 5e-5, 500)
        with self.assertRaisesRegex(ValueError, "divisible"):
            training.StageSpec("generator", 50000, 5, 2, 3e-4, 3e-5, 1000)

    def test_dependency_mismatch_is_checked_before_training(self):
        with patch("importlib.metadata.version", return_value="0.0"):
            with self.assertRaisesRegex(RuntimeError, "Expected torch"):
                training.verify_environment()

    def test_second_trainer_cannot_write_the_same_run(self):
        (training.ROOT / "runs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="test-training-lock-", dir=training.ROOT / "runs") as directory:
            output = io.StringIO()
            with training.exclusive_run(Path(directory)), redirect_stderr(output):
                result = training.main(["tokenizer", "--data", "unused", "--run", directory,
                                        "--device", "cpu", "--stop-after", "1"])
            self.assertEqual(result, 1)
            self.assertIn("Another trainer owns", output.getvalue())
            self.assertFalse((Path(directory) / "run.json").exists())


class TrainingBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (training.ROOT / "runs").mkdir(exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(prefix="test-training-", dir=training.ROOT / "runs")
        cls.root = Path(cls.temporary.name)
        cache = cls.root / "source"
        cache.mkdir()
        rng = np.random.default_rng(173)
        base = (np.arange(9)[:, None] + np.arange(9)[None, :]) % 9
        boards = [rng.permutation(np.arange(1, 10, dtype=np.uint8))[base[rng.permutation(9)][:, rng.permutation(9)]]
                  for _ in range(12)]
        payload = "".join(" ".join("".join(str(int(cell) - 1) for cell in row) for row in board) + " \n"
                          for board in boards).encode("ascii")
        compressed = gzip.compress(payload, mtime=0)
        (cache / "fixture.gz").write_bytes(compressed)
        source = SourceArchive("fixture.gz", "https://example.invalid/fixture.gz", len(compressed),
                               hashlib.sha256(compressed).hexdigest(), len(boards))
        cls.corpus = prepare_corpus(cls.root / "data", cache, sources=(source,), validation_count=2,
                                    test_count=2, benchmark_count=2, benchmark_source_lines=10)
        cls.runs = [cls.root / "continuous", cls.root / "resumed"]
        cls.commands = []
        cls.invoke("tokenizer", cls.runs[0], "--batch-size", "2", "--stop-after", "2")
        cls.invoke("tokenizer", cls.runs[1], "--batch-size", "2", "--stop-after", "1")
        cls.invoke("tokenizer", cls.runs[1], "--resume", "--stop-after", "1")
        cls.invoke("generator", cls.runs[0], "--batch-size", "2", "--microbatch-size", "1",
                   "--stop-after", "2", "--allow-partial-tokenizer")
        cls.invoke("generator", cls.runs[1], "--batch-size", "2", "--microbatch-size", "1",
                   "--stop-after", "1", "--allow-partial-tokenizer")
        cls.invoke("generator", cls.runs[1], "--resume", "--stop-after", "1", "--allow-partial-tokenizer")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def invoke(cls, stage, run, *options):
        output = io.StringIO()
        args = [stage, "--data", str(cls.corpus.directory), "--run", str(run), "--device", "cpu", *options]
        with redirect_stdout(output), redirect_stderr(output):
            result = training.main(args)
        if result != 0:
            raise AssertionError(f"Training command failed {args}\n{output.getvalue()}")
        cls.commands.append(args)

    def test_complete_resume_replays_models_optimizer_rng_and_sampling(self):
        for stage in ("tokenizer", "generator"):
            states = [torch.load(run / stage / "state.pt", map_location="cpu", weights_only=True)
                      for run in self.runs]
            self.assertEqual(states[0]["completed_updates"], 2)
            self.assertEqual(states[1]["completed_updates"], 2)
            for key in ("model", "optimizer", "rng", "sampler", "fine_drop_step"):
                assert_tree_equal(self, states[0][key], states[1][key], f"{stage}.{key}")
            self.assertEqual(states[1]["immutable"]["stage"]["batch_size"], 2)
            self.assertEqual(states[1]["immutable"]["stage"]["microbatch_size"], 2 if stage == "tokenizer" else 1)
            first_lr = states[1]["optimizer"]["param_groups"][0]["lr"]
            self.assertAlmostEqual(first_lr, 2e-6 if stage == "tokenizer" else 6e-7)
            del states

    def test_tokenizer_ema_and_complete_portable_exports(self):
        state = torch.load(self.runs[0] / "tokenizer/state.pt", map_location="cpu", weights_only=True)
        self.assertEqual(state["fine_drop_step"], 2)
        self.assertEqual(sum(bool(state["model"][f"vq.ema_cluster_size_{level}"].sum() > 0)
                             for level in range(12)), 12)
        for stage in ("tokenizer", "generator"):
            export = self.runs[0] / stage / "export"
            metadata = evaluation.verify_local_export(export, stage=stage)
            self.assertEqual(metadata["checkpoint"]["training"]["tokenizer_updates"], 2)
            self.assertEqual(metadata["checkpoint"]["data"]["manifest_sha256"], self.corpus.manifest_sha256)
            exported = torch.load(export / f"{stage}.pt", map_location="cpu", weights_only=True)
            saved = torch.load(self.runs[0] / stage / "state.pt", map_location="cpu", weights_only=True)
            assert_tree_equal(self, exported["model"], saved["model"])

    def test_partial_tokenizer_requires_explicit_bounded_generator_run(self):
        with self.assertRaisesRegex(ValueError, "partial tokenizer requires"):
            training.snapshot_tokenizer(self.runs[0], False, 1)
        with self.assertRaisesRegex(ValueError, "partial tokenizer requires"):
            training.snapshot_tokenizer(self.runs[0], True, None)
        snapshot, _ = training.snapshot_tokenizer(self.runs[0], True, 1)
        self.assertEqual(evaluation.sha256_file(snapshot),
                         evaluation.sha256_file(self.runs[0] / "generator/export/tokenizer.pt"))

    def test_saved_model_reload_uses_existing_latin_evaluation_runner(self):
        destination = self.root / "saved-evaluation"
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = evaluation.main(["--local-checkpoint", str(self.runs[0] / "generator/export"),
                                    "--data", str(self.corpus.directory), "--split", "test", "--count", "2",
                                    "--device", "cpu", "--output", str(destination)])
        summary = json.loads((destination / "summary.json").read_text())
        self.assertEqual(code, 0, summary)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["completed"], 2)
        self.assertEqual(summary["split"], "test")
        self.assertEqual(summary["cases_sha256"], evaluation.sha256_file(destination / "cases.jsonl"))

    def test_changed_export_weights_are_rejected_before_loading(self):
        original = self.runs[0] / "generator/export"
        destination = self.root / "tampered-export"
        import shutil
        shutil.copytree(original, destination)
        with (destination / "generator.pt").open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"x")
        with self.assertRaisesRegex(ValueError, "checksum mismatch for generator.pt"):
            evaluation.verify_local_export(destination)

    def test_failed_forward_does_not_persist_partial_ema_or_sampler_changes(self):
        checkpoint = self.runs[1] / "tokenizer/state.pt"
        before = evaluation.sha256_file(checkpoint)
        actual_loss = training.tokenizer_loss

        def fail_after_ema(*args):
            actual_loss(*args)
            raise RuntimeError("Injected failure after the quantizer updated its EMA.")

        with patch.object(training, "tokenizer_loss", side_effect=fail_after_ema), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = training.main(["tokenizer", "--data", str(self.corpus.directory), "--run", str(self.runs[1]),
                                    "--device", "cpu", "--resume", "--stop-after", "1"])
        self.assertEqual(result, 1)
        self.assertEqual(evaluation.sha256_file(checkpoint), before)
        status = json.loads((self.runs[1] / "tokenizer/status.json").read_text())
        self.assertEqual(status["last_saved_updates"], 2)
        self.assertEqual(status["status"], "failed")

    def test_generator_snapshot_survives_a_replaced_tokenizer_export(self):
        source = self.runs[1] / "tokenizer/export/tokenizer.pt"
        saved_source = source.with_suffix(".saved")
        snapshot = self.runs[1] / "generator/tokenizer-snapshot.pt"
        expected = evaluation.sha256_file(snapshot)
        original_metadata = snapshot.with_suffix(".json").read_bytes()
        source.rename(saved_source)
        try:
            source.write_bytes(b"A later tokenizer export has different weights.")
            actual, metadata = training.snapshot_tokenizer(self.runs[1], True, 1)
            self.assertEqual(actual, snapshot)
            self.assertEqual(evaluation.sha256_file(actual), expected)
            self.assertEqual(metadata["completed_updates"], 2)
            self.assertEqual(snapshot.with_suffix(".json").read_bytes(), original_metadata)
        finally:
            source.unlink()
            saved_source.rename(source)

    def test_incomplete_snapshot_publication_can_be_retried_before_generator_training(self):
        run = self.root / "snapshot-publication-retry"
        shutil.copytree(self.runs[0] / "tokenizer/export", run / "tokenizer/export")
        destination = run / "generator/tokenizer-snapshot.pt"
        with patch.object(training, "atomic_json", side_effect=OSError("Injected metadata publication failure")):
            with self.assertRaisesRegex(OSError, "Injected metadata publication failure"):
                training.snapshot_tokenizer(run, True, 1)
        self.assertTrue(destination.is_file())
        self.assertFalse(destination.with_suffix(".json").exists())
        previous = evaluation.sha256_file(destination)
        source = run / "tokenizer/export/tokenizer.pt"
        weights = torch.load(source, map_location="cpu", weights_only=True)
        torch.save(weights, source)
        self.assertNotEqual(evaluation.sha256_file(source), previous)
        provenance_path = source.parent / "provenance.json"
        provenance = json.loads(provenance_path.read_text())
        provenance["files"]["tokenizer.pt"] = {"sha256": evaluation.sha256_file(source),
                                               "bytes": source.stat().st_size}
        provenance_path.write_text(json.dumps(provenance))
        actual, metadata = training.snapshot_tokenizer(run, True, 1)
        self.assertEqual(actual, destination)
        self.assertEqual(metadata["completed_updates"], 2)
        self.assertEqual(metadata["sha256"], evaluation.sha256_file(run / "tokenizer/export/tokenizer.pt"))
        self.assertEqual(json.loads(destination.with_suffix(".json").read_text()), metadata)
        self.assertEqual(training.snapshot_tokenizer(run, True, 1), (actual, metadata))

    def test_incomplete_snapshot_with_generator_checkpoint_cannot_be_replaced(self):
        run = self.root / "snapshot-in-use"
        shutil.copytree(self.runs[0] / "tokenizer/export", run / "tokenizer/export")
        destination = run / "generator/tokenizer-snapshot.pt"
        with patch.object(training, "atomic_json", side_effect=OSError("Injected metadata publication failure")):
            with self.assertRaises(OSError):
                training.snapshot_tokenizer(run, True, 1)
        expected = evaluation.sha256_file(destination)
        (run / "generator/state.pt").write_bytes(b"An existing generator depends on the original snapshot.")
        with self.assertRaisesRegex(ValueError, "Incomplete immutable tokenizer snapshot"):
            training.snapshot_tokenizer(run, True, 1)
        self.assertEqual(evaluation.sha256_file(destination), expected)
        self.assertFalse(destination.with_suffix(".json").exists())

    def test_generator_export_retains_its_tokenizers_actual_training_recipe(self):
        run = self.root / "mixed-stage-batches"
        self.invoke("tokenizer", run, "--batch-size", "2", "--stop-after", "1")
        tokenizer_training = json.loads((run / "tokenizer/export/provenance.json").read_text())["training"]
        snapshot_path, snapshot_metadata = training.snapshot_tokenizer(run, True, 1)
        before = snapshot_path.with_suffix(".json").read_bytes()
        self.invoke("generator", run, "--batch-size", "4", "--microbatch-size", "2",
                    "--stop-after", "1", "--allow-partial-tokenizer")
        export = run / "generator/export"
        metadata = evaluation.verify_local_export(export)["checkpoint"]
        generator_training = metadata["training"]
        snapshot = generator_training["tokenizer_snapshot"]
        self.assertEqual(generator_training["stage_spec"]["batch_size"], 4)
        self.assertEqual(snapshot["stage_spec"], tokenizer_training["stage_spec"])
        self.assertEqual(snapshot["stage_spec"]["batch_size"], 2)
        self.assertEqual(snapshot["stage_spec"]["microbatch_size"], 2)
        self.assertEqual(snapshot["stage_spec"]["updates"], 10000)
        self.assertEqual(snapshot["deviations"], tokenizer_training["deviations"])
        self.assertIn("global batch 2 replaces published batch 1024", snapshot["deviations"])
        self.assertEqual(snapshot, snapshot_metadata)
        self.assertEqual(snapshot_path.with_suffix(".json").read_bytes(), before)
        provenance_path = export / "provenance.json"
        original = provenance_path.read_bytes()
        mutations = (
            ("stage_spec", "microbatch_size", 1),
            ("stage_spec", "updates", 2),
            ("stage_spec", "batch_size", True),
            (None, "intended_updates", 2),
            (None, "completed_updates", 2),
            (None, "completed_updates", True),
            (None, "deviations", []),
        )
        try:
            for section, key, value in mutations:
                with self.subTest(section=section, key=key):
                    altered = json.loads(original)
                    record = altered["training"]["tokenizer_snapshot"]
                    (record[section] if section else record)[key] = value
                    provenance_path.write_text(json.dumps(altered))
                    with self.assertRaises(ValueError):
                        evaluation.verify_local_export(export)
        finally:
            provenance_path.write_bytes(original)

    def test_all_command_waits_for_the_complete_tokenizer_schedule(self):
        run = self.root / "all-command"
        self.invoke("all", run, "--batch-size", "2", "--stop-after", "1")
        state = torch.load(run / "tokenizer/state.pt", map_location="cpu", weights_only=True)
        self.assertEqual(state["completed_updates"], 1)
        self.assertEqual(state["immutable"]["stage"]["updates"], 10000)
        self.assertFalse((run / "generator/state.pt").exists())

    def test_validation_preserves_models_ema_modes_and_all_rng(self):
        architecture = OmegaConf.load(training.ROOT / "vendor/ncm/config/eval/sudoku-hard.yaml")
        recipe = OmegaConf.load(training.ROOT / "configs/latin9.yaml")
        tokenizer = training.make_tokenizer(architecture, "cpu")
        tokenizer.load_state_dict(torch.load(self.runs[0] / "tokenizer/export/tokenizer.pt", weights_only=True)["model"])
        tokenizer.vq._fd_step.fill_(2)
        for stage in ("tokenizer", "generator"):
            model = tokenizer if stage == "tokenizer" else training.make_generator(architecture, tokenizer, "cpu")
            if stage == "generator":
                model.load_state_dict(torch.load(self.runs[0] / "generator/export/generator.pt", weights_only=True)["model"])
            model.train()
            modes = (model.training, tokenizer.training)
            before = {key: value.clone() for key, value in model.state_dict().items()}
            tokenizer_before = {key: value.clone() for key, value in tokenizer.state_dict().items()}
            saved_rng = training.rng_state("cpu")
            result = training.validate_stage(stage, model, tokenizer, self.corpus, recipe, "cpu", 2)
            self.assertEqual(result["count"], 2)
            if stage == "generator":
                self.assertEqual(result["scored"], 2)
                self.assertEqual(result["given_cells_total"], 60)
            else:
                self.assertTrue(np.isfinite(result["full_hierarchy_reconstruction_ce"]))
                self.assertTrue(0 <= result["token_accuracy"] <= 1)
            assert_tree_equal(self, before, dict(model.state_dict()))
            assert_tree_equal(self, tokenizer_before, dict(tokenizer.state_dict()))
            assert_tree_equal(self, saved_rng, training.rng_state("cpu"))
            self.assertEqual(int(tokenizer.vq._fd_step), 2)
            self.assertEqual((model.training, tokenizer.training), modes)


class ObjectiveTests(unittest.TestCase):
    def test_all_code_targets_and_trainable_query_receive_gradients(self):
        torch.set_num_threads(8)
        training.seed_everything(42)
        architecture = OmegaConf.load(training.ROOT / "vendor/ncm/config/eval/sudoku-hard.yaml")
        recipe = OmegaConf.load(training.ROOT / "configs/latin9.yaml")
        tokenizer = training.make_tokenizer(architecture, "cpu")
        model = training.make_generator(architecture, tokenizer, "cpu").train()
        cells = [[(row + column) % 9 + 1 for row in range(9) for column in range(9)]]
        solution = torch.tensor([[cell for index, cell in enumerate(cells[0])
                                  for cell in ([10, cell] if index and index % 9 == 0 else [cell])]])
        prefix = torch.cat((torch.tensor([[11]]), solution, torch.tensor([[11]])), dim=1)
        tokenizer_before = {key: value.clone() for key, value in tokenizer.state_dict().items()}
        query_before = model.l0_queries.detach().clone()
        head_before = model.lm_heads[0].weight.detach().clone()
        optimizer = training.make_optimizer(model, recipe)
        for group in optimizer.param_groups:
            group["lr"] = 0.001
        loss, _ = training.generator_loss(model, tokenizer, solution, prefix)
        loss.backward()
        self.assertTrue(torch.isfinite(model.l0_queries.grad).all())
        self.assertNotEqual(float(model.l0_queries.grad.abs().sum()), 0.0)
        self.assertEqual(sum(head.weight.grad is not None and bool(head.weight.grad.abs().sum() > 0)
                             for head in model.lm_heads), 12)
        self.assertTrue(all(parameter.grad is None for parameter in tokenizer.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(query_before, model.l0_queries))
        self.assertFalse(torch.equal(head_before, model.lm_heads[0].weight))
        assert_tree_equal(self, tokenizer_before, dict(tokenizer.state_dict()))


if __name__ == "__main__":
    unittest.main()
