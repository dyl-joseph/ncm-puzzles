from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import random
import shutil
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import torch

import continue_latin as continuation
from latin_data import LatinStream


def assert_tree_equal(test, first, second):
    test.assertEqual(type(first), type(second))
    if isinstance(first, torch.Tensor):
        test.assertTrue(torch.equal(first, second))
    elif isinstance(first, dict):
        test.assertEqual(first.keys(), second.keys())
        for key in first:
            assert_tree_equal(test, first[key], second[key])
    elif isinstance(first, (tuple, list)):
        test.assertEqual(len(first), len(second))
        for left, right in zip(first, second):
            assert_tree_equal(test, left, right)
    else:
        test.assertEqual(first, second)


class ContinuationBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="test-continuation-", dir=continuation.ROOT / "runs")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source, self.run = self.root / "source", self.root / "continued"
        (self.source / "generator/export").mkdir(parents=True)
        (self.source / ".train.lock").touch()
        self.recipe = OmegaConf.load(continuation.ROOT / "configs/latin9-10min.yaml")
        self.context = {"recipe": OmegaConf.to_container(self.recipe, resolve=True),
                        "trainer_sha256": "pinned-trainer", "device": "mps"}
        self.corpus = SimpleNamespace(manifest_sha256="fixture-manifest", splits={"train": np.arange(10)})
        self.corpus.training_stream = lambda **kwargs: LatinStream(self.corpus, **kwargs)
        stream = self.corpus.training_stream(seed=42, clue_count=30, augment=True)
        stream.cursor, stream.epoch = 7, 12
        np_state = np.random.RandomState(42).get_state()
        tokenizer = self.source / "generator/tokenizer-snapshot.pt"
        tokenizer.write_bytes(b"frozen tokenizer fixture")
        self.snapshot = {"completed_updates": 213, "intended_updates": 10000,
                         "stage_spec": continuation.asdict(continuation.stage_spec(self.recipe, "tokenizer")),
                         "deviations": [], "sha256": continuation.sha256_file(tokenizer)}
        self.state = {"format": "ncm-latin-training-state-v1", "completed_updates": 400,
                      "immutable": {"run_config_sha256": continuation.canonical_hash(self.context),
                                    "stage": continuation.asdict(continuation.stage_spec(self.recipe, "generator")),
                                    "tokenizer_snapshot": self.snapshot},
                      "model": {"weight": torch.tensor([[1.0, 2.0]]), "bias": torch.tensor([3.0])},
                      "optimizer": {"state": {0: {"step": torch.tensor(400.0),
                                                  "exp_avg": torch.tensor([[0.1, 0.2]]),
                                                  "exp_avg_sq": torch.tensor([[0.01, 0.02]])}},
                                    "param_groups": [{"lr": 3e-5, "params": [0]}]},
                      "sampler": stream.state_dict(),
                      "rng": {"python": random.Random(42).getstate(),
                              "numpy": [np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4]],
                              "torch_cpu": torch.Generator().manual_seed(42).get_state(),
                              "torch_mps": torch.tensor([1, 2, 3], dtype=torch.uint8)},
                      "fine_drop_step": None}
        continuation.atomic_json(self.source / "run.json", self.context)
        continuation.atomic_json(self.source / "generator/tokenizer-snapshot.json", self.snapshot)
        self.save_source()
        for name, value in (("verify_local_export", self.verify_export), ("run_context", self.run_context)):
            active = patch.object(continuation, name, side_effect=value)
            active.start()
            self.addCleanup(active.stop)

    def save_export(self, run, state, context):
        export = run / "generator/export"
        export.mkdir(exist_ok=True)
        continuation.atomic_torch_save(export / "generator.pt", {"model": state["model"]})
        shutil.copyfile(run / "generator/tokenizer-snapshot.pt", export / "tokenizer.pt")
        continuation.atomic_json(export / "run-config.json", context)
        provenance = {"files": {name: {"sha256": continuation.sha256_file(export / name)}
                                for name in ("generator.pt", "tokenizer.pt", "run-config.json")},
                      "run_config_sha256": continuation.canonical_hash(context),
                      "training": {"generator_updates": state["completed_updates"],
                                   "tokenizer_snapshot": self.snapshot,
                                   "stage_spec": state["immutable"]["stage"]}}
        continuation.atomic_json(export / "provenance.json", provenance)

    def save_source(self):
        continuation.atomic_torch_save(self.source / "generator/state.pt", self.state)
        self.save_export(self.source, self.state, self.context)

    def verify_export(self, export):
        return {"checkpoint": continuation.read_json(export / "provenance.json")}

    def run_context(self, data, run, device, config):
        recipe = OmegaConf.load(config)
        context = {**self.context, "recipe": OmegaConf.to_container(recipe, resolve=True)}
        continuation.atomic_json(run / "run.json", context)
        return self.corpus, recipe, None, context

    def prepare(self, resume=False):
        return continuation.prepare_continuation(self.source, self.run, resume)

    def checkpoint(self):
        return torch.load(self.run / "generator/state.pt", map_location="cpu", weights_only=True)

    def save_success(self, completed=402):
        state = self.checkpoint()
        state["completed_updates"] = completed
        state["model"]["weight"] += 1
        continuation.atomic_torch_save(self.run / "generator/state.pt", state)
        self.save_export(self.run, state, continuation.read_json(self.run / "run.json"))
        continuation.atomic_json(self.run / "generator/status.json", {
            "status": "stopped", "finished_at": datetime.now(timezone.utc).isoformat(),
            "last_saved_updates": completed, "completed_updates_in_process": completed})

    def test_migration_preserves_all_training_state_and_source_bytes(self):
        before = {str(path.relative_to(self.source)): (path.read_bytes(), path.stat().st_mtime_ns)
                  for path in self.source.rglob("*") if path.is_file()}
        lineage, completed = self.prepare()
        migrated = self.checkpoint()
        self.assertEqual(completed, 400)
        for key in self.state:
            if key != "immutable":
                assert_tree_equal(self, self.state[key], migrated[key])
        self.assertEqual(set(migrated["immutable"]), set(self.state["immutable"]))
        self.assertEqual(migrated["immutable"]["tokenizer_snapshot"], self.state["immutable"]["tokenizer_snapshot"])
        self.assertNotEqual(migrated["immutable"]["run_config_sha256"], self.state["immutable"]["run_config_sha256"])
        self.assertEqual(migrated["immutable"]["stage"], {
            "name": "generator", "updates": 1000, "batch_size": 32, "microbatch_size": 32,
            "peak_lr": 3e-5, "floor_lr": 3e-5, "warmup_updates": 0})
        spec = continuation.stage_spec(OmegaConf.load(self.run / "continuation.yaml"), "generator")
        self.assertEqual(continuation.learning_rate(spec, 401), 3e-5)
        self.assertEqual(continuation.learning_rate(spec, 1000), 3e-5)
        self.assertEqual(continuation.read_json(self.run / "parent-run.json"), self.context)
        self.assertEqual(lineage["source_stage_spec"]["updates"], 400)
        self.assertEqual(lineage["source_stage_spec"]["peak_lr"], 3e-4)
        self.assertEqual({str(path.relative_to(self.source)): (path.read_bytes(), path.stat().st_mtime_ns)
                          for path in self.source.rglob("*") if path.is_file()}, before)

    def test_changed_source_state_or_missing_rng_rejected_without_publication(self):
        for change in ("model", "immutable", "rng", "sampler", "optimizer"):
            with self.subTest(change=change):
                state = deepcopy(self.state)
                if change == "model":
                    state["model"]["weight"] += 1
                elif change == "immutable":
                    state["immutable"]["run_config_sha256"] = "different-run"
                elif change == "rng":
                    del state["rng"]["torch_mps"]
                elif change == "sampler":
                    state["sampler"]["cursor"] = 10
                else:
                    state["optimizer"]["param_groups"][0]["lr"] = 1e-4
                continuation.atomic_torch_save(self.source / "generator/state.pt", state)
                with self.assertRaises((ValueError, KeyError)):
                    self.prepare()
                self.assertFalse(self.run.exists())
        self.save_source()

    def test_each_completed_continuation_adds_600_updates_without_resetting_state(self):
        for completed in (1000, 1600, 2200):
            with self.subTest(completed=completed):
                self.run = self.root / f"continued-{completed}"
                self.recipe.generator.updates = completed
                self.context["recipe"] = OmegaConf.to_container(self.recipe, resolve=True)
                self.state["completed_updates"] = completed
                self.state["optimizer"]["state"][0]["step"] = torch.tensor(float(completed))
                self.state["immutable"].update(
                    run_config_sha256=continuation.canonical_hash(self.context),
                    stage=continuation.asdict(continuation.stage_spec(self.recipe, "generator")))
                continuation.atomic_json(self.source / "run.json", self.context)
                self.save_source()
                source_bytes = {str(path.relative_to(self.source)): path.read_bytes()
                                for path in self.source.rglob("*") if path.is_file()}
                lineage, initial = self.prepare()
                self.assertEqual(initial, completed)
                self.assertEqual(lineage["target_update"], completed + 600)
                migrated = self.checkpoint()
                self.assertEqual(migrated["immutable"]["stage"]["updates"], completed + 600)
                for key in self.state:
                    if key != "immutable":
                        assert_tree_equal(self, self.state[key], migrated[key])
                spec = continuation.stage_spec(OmegaConf.load(self.run / "continuation.yaml"), "generator")
                self.assertEqual(continuation.learning_rate(spec, completed + 1), 3e-5)
                self.assertEqual(continuation.learning_rate(spec, completed + 600), 3e-5)
                self.assertEqual(self.prepare(True)[1], completed)
                self.assertEqual({str(path.relative_to(self.source)): path.read_bytes()
                                  for path in self.source.rglob("*") if path.is_file()}, source_bytes)

    def test_source_run_and_export_counts_must_agree(self):
        provenance = continuation.read_json(self.source / "generator/export/provenance.json")
        provenance["training"]["generator_updates"] = 399
        continuation.atomic_json(self.source / "generator/export/provenance.json", provenance)
        with self.assertRaisesRegex(ValueError, "update counts"):
            self.prepare()
        self.assertFalse(self.run.exists())

    def test_resume_preserves_advanced_state_and_rejects_changed_lineage_or_sampler(self):
        self.prepare()
        self.save_success()
        before = (self.run / "generator/state.pt").read_bytes()
        self.assertEqual(self.prepare(True)[1], 402)
        self.assertEqual((self.run / "generator/state.pt").read_bytes(), before)
        lineage = continuation.read_json(self.run / "continuation.json")
        lineage["start_update"] = 399
        continuation.atomic_json(self.run / "continuation.json", lineage)
        with self.assertRaisesRegex(ValueError, "lineage"):
            self.prepare(True)
        lineage["start_update"] = 400
        continuation.atomic_json(self.run / "continuation.json", lineage)
        state = self.checkpoint()
        state["sampler"]["manifest_sha256"] = "other-data"
        continuation.atomic_torch_save(self.run / "generator/state.pt", state)
        with self.assertRaisesRegex(ValueError, "Sampler state"):
            self.prepare(True)

    def test_long_phase_resume_keeps_its_saved_target_and_all_advanced_state(self):
        lineage, initial = continuation.prepare_continuation(self.source, self.run, False,
                                                             additional_updates=2000)
        self.assertEqual(initial, 400)
        self.assertEqual(lineage["target_update"], 2400)
        self.assertEqual(self.checkpoint()["immutable"]["stage"]["updates"], 2400)
        self.save_success(completed=1000)
        before = {str(path.relative_to(self.run)): path.read_bytes()
                  for path in self.run.rglob("*") if path.is_file()}
        for requested in (None, 2000):
            resumed, current = continuation.prepare_continuation(self.source, self.run, True,
                                                                  additional_updates=requested)
            self.assertEqual(current, 1000)
            self.assertEqual(resumed["target_update"], 2400)
            self.assertEqual({str(path.relative_to(self.run)): path.read_bytes()
                              for path in self.run.rglob("*") if path.is_file()}, before)
        for requested in (600, 2001):
            with self.subTest(requested=requested), self.assertRaisesRegex(ValueError, "saved continuation adds 2000"):
                continuation.prepare_continuation(self.source, self.run, True, additional_updates=requested)
            self.assertEqual({str(path.relative_to(self.run)): path.read_bytes()
                              for path in self.run.rglob("*") if path.is_file()}, before)

    def test_invalid_phase_budget_rejects_before_creating_a_destination(self):
        for value in (0, -1, 1.5, True, "2000"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "positive integer"):
                continuation.prepare_continuation(self.source, self.run, False, additional_updates=value)
            self.assertFalse(self.run.exists())

    def test_invalid_saved_phase_range_rejects_without_changing_state(self):
        self.prepare()
        original = continuation.read_json(self.run / "continuation.json")
        before = (self.run / "generator/state.pt").read_bytes()
        for changes in ({"start_update": True}, {"target_update": True}, {"start_update": -1},
                        {"target_update": 400}, {"target_update": "2400"}):
            continuation.atomic_json(self.run / "continuation.json", {**original, **changes})
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "update range is invalid"):
                self.prepare(True)
            self.assertEqual((self.run / "generator/state.pt").read_bytes(), before)

    def test_long_phase_cli_caps_each_invocation_and_finishes_only_remaining_updates(self):
        def first_child(command, *args):
            self.assertEqual(command[command.index("--stop-after") + 1], "600")
            self.save_success(completed=1000)
            return 0, "completed"
        with patch.object(continuation, "run_child", side_effect=first_child), redirect_stdout(io.StringIO()):
            self.assertEqual(continuation.main(["--source", str(self.source), "--run", str(self.run),
                                              "--additional-updates", "2000", "--stop-after", "2000"]), 0)
        self.assertEqual(continuation.read_json(self.run / "continuation-result.json")["target_updates"], 2400)
        self.save_success(completed=2200)
        def last_child(command, *args):
            self.assertEqual(command[command.index("--stop-after") + 1], "200")
            self.save_success(completed=2400)
            return 0, "completed"
        with patch.object(continuation, "run_child", side_effect=last_child), redirect_stdout(io.StringIO()):
            self.assertEqual(continuation.main(["--source", str(self.source), "--run", str(self.run), "--resume"]), 0)
        result = continuation.read_json(self.run / "continuation-result.json")
        self.assertEqual(result["completed_updates"], 2400)
        self.assertEqual(result["new_updates"], 200)

    def test_reject_current_source_environment_mismatch(self):
        actual = self.run_context
        def changed(*args):
            corpus, recipe, architecture, context = actual(*args)
            context["trainer_sha256"] = "changed-trainer"
            return corpus, recipe, architecture, context
        with patch.object(continuation, "run_context", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "current trainer"):
                self.prepare()
        self.assertFalse(self.run.exists())

    def test_an_active_source_trainer_blocks_migration(self):
        with (self.source / ".train.lock").open("r") as lock:
            continuation.fcntl.flock(lock, continuation.fcntl.LOCK_EX | continuation.fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "Stop the source"):
                self.prepare()
        self.assertFalse(self.run.exists())

    def test_ready_requires_a_fresh_matching_export_and_new_updates(self):
        lineage, initial = self.prepare()
        launched_at = datetime.now(timezone.utc)
        self.save_success()
        self.assertEqual(continuation.verify_ready(self.run, lineage, initial, launched_at), 402)
        with self.assertRaisesRegex(ValueError, "matching generator"):
            continuation.verify_ready(self.run, lineage, 402, launched_at)
        with self.assertRaisesRegex(ValueError, "matching generator"):
            continuation.verify_ready(self.run, lineage, initial, launched_at + timedelta(minutes=1))
        state = self.checkpoint()
        state["model"]["weight"] += 1
        continuation.atomic_torch_save(self.run / "generator/state.pt", state)
        with self.assertRaisesRegex(ValueError, "weights disagree"):
            continuation.verify_ready(self.run, lineage, initial, launched_at)

    def test_failed_child_never_evaluates_an_older_export(self):
        with patch.object(continuation, "run_child", return_value=(-9, "killed")), \
                patch.object(continuation, "verify_ready") as ready, \
                patch.object(continuation, "evaluate") as evaluate, redirect_stdout(io.StringIO()):
            result = continuation.main(["--source", str(self.source), "--run", str(self.run), "--evaluate"])
        self.assertEqual(result, 1)
        ready.assert_not_called()
        evaluate.assert_not_called()
        record = continuation.read_json(self.run / "continuation-result.json")
        self.assertFalse(record["ready"])
        self.assertEqual(record["stop_reason"], "killed")

    def test_successful_continuation_evaluates_after_training_clock_is_recorded(self):
        def child(command, started, seconds):
            self.assertIn("--resume", command)
            self.assertEqual(command[command.index("--stop-after") + 1], "2")
            self.save_success()
            return 0, "completed"
        def evaluate(run, count):
            record = continuation.read_json(run / "continuation-result.json")
            self.assertTrue(record["ready"])
            self.assertEqual(record["new_updates"], 2)
            self.assertEqual(count, 8)
            time.sleep(0.05)
            return run / "evaluation-fixture"
        with patch.object(continuation, "run_child", side_effect=child), \
                patch.object(continuation, "evaluate", side_effect=evaluate), redirect_stdout(io.StringIO()):
            self.assertEqual(continuation.main(["--source", str(self.source), "--run", str(self.run),
                                              "--stop-after", "2", "--evaluate", "--eval-count", "8"]), 0)
        record = continuation.read_json(self.run / "continuation-result.json")
        self.assertTrue(record["ready"])
        self.assertEqual(record["completed_updates"], 402)
        self.assertEqual(record["evaluation_output"], str(self.run / "evaluation-fixture"))

    def test_evaluation_failure_keeps_training_ready_and_returns_failure(self):
        def child(*args):
            self.save_success()
            return 0, "completed"
        with patch.object(continuation, "run_child", side_effect=child), \
                patch.object(continuation, "evaluate", side_effect=RuntimeError("MLX evaluation failed")), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(continuation.main(["--source", str(self.source), "--run", str(self.run), "--evaluate"]), 1)
        record = continuation.read_json(self.run / "continuation-result.json")
        self.assertTrue(record["ready"])
        self.assertEqual(record["evaluation_error"], "MLX evaluation failed")

    def test_evaluation_command_uses_validation_and_a_new_output_directory(self):
        self.prepare()
        def process(command, **kwargs):
            self.assertTrue(kwargs["check"])
            self.assertEqual(command[command.index("--split") + 1], "validation")
            self.assertEqual(command[command.index("--device") + 1], "mlx")
            self.assertEqual(command[command.index("--count") + 1], "8")
            output = Path(command[command.index("--output") + 1])
            self.assertFalse(output.exists())
            continuation.atomic_json(output / "summary.json", {"status": "complete", "completed": 8,
                "metrics": {"solved": 1, "given_cell_agreement": 0.9}})
        with patch.object(continuation.subprocess, "run", side_effect=process), redirect_stdout(io.StringIO()):
            output = continuation.evaluate(self.run, 8)
        self.assertTrue((output / "summary.json").is_file())

    def test_destination_cannot_overwrite_source(self):
        with redirect_stderr(io.StringIO()):
            self.assertEqual(continuation.main(["--source", str(self.source), "--run", str(self.source)]), 1)
        self.assertEqual(continuation.read_json(self.source / "run.json"), self.context)


if __name__ == "__main__":
    unittest.main()
