"""Exercise restart, archive and notebook contracts without training or a GPU."""

from dataclasses import replace
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .build_colab_notebook import notebook
from .colab_training import (CHECKPOINT_FILES, RunSpec, TrainingRun, checkpoint_hashes,
                             frozen_evaluation, run_process)


class FakeBlock:
    def __init__(self):
        self.calls = []
        self.reject_verification = False

    def __call__(self, command, **_):
        self.calls.append(command)
        output = Path(command[command.index("--output") + 1])
        initial = Path(command[command.index("--checkpoint") + 1])
        output.mkdir(parents=True)
        checkpoint = output / "checkpoint"
        checkpoint.mkdir()
        for name in CHECKPOINT_FILES:
            (checkpoint / name).write_bytes((initial / name).read_bytes())
        with (checkpoint / "generator.safetensors").open("ab") as stream:
            stream.write(str(len(self.calls)).encode())
        for name, value in {"config.json": {}, "deadline-result.json": {},
                            "summary.json": {"episodes": [{"episode_index": 0}]}}.items():
            (output / name).write_text(json.dumps(value))
        for name in ("game-trace.jsonl", "training.jsonl"):
            (output / name).write_text('{"fake_process_boundary":true}\n')

    def verify(self, directory, initial):
        if self.reject_verification:
            raise AssertionError("No optimizer work happened")
        if not (directory / "checkpoint/generator.safetensors").read_bytes().startswith(
                (initial / "generator.safetensors").read_bytes()):
            raise AssertionError("Weight warm-start chain differs")
        return {"status": "passed", "optimizer_updates": 1, "counts": {"level_completion_rewards": 0}}


class ColabRunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.initial = self.root / "initial"
        self.initial.mkdir()
        for name in CHECKPOINT_FILES:
            (self.initial / name).write_text(name)
        self.spec = RunSpec("a" * 40, 90, 30, 42, checkpoint_hashes(self.initial), {}, {}, "baseline")
        self.fake = FakeBlock()

    def run_object(self, spec=None):
        return TrainingRun(self.root / "store", self.root / "local", spec or self.spec,
                           self.initial, runner=self.fake, verifier=self.fake.verify)

    def test_warm_start_chain_unique_seeds_and_budget(self):
        run = self.run_object()
        final = run.train("fake-python", "fake-environments")
        self.assertEqual([a.state for a in run.attempts], ["completed"] * 3)
        self.assertEqual([a.seed for a in run.attempts], [42, 100042, 200042])
        self.assertEqual(run.charged_seconds, 90)
        self.assertEqual(final, self.root / "store/blocks/0002/checkpoint")
        starts = [Path(command[command.index("--checkpoint") + 1]) for command in self.fake.calls]
        self.assertEqual(starts, [self.initial, self.root / "store/blocks/0000/checkpoint",
                                  self.root / "store/blocks/0001/checkpoint"])
        self.assertTrue((self.root / "store/blocks/0000/game-trace.jsonl.xz").exists())
        self.assertFalse((self.root / "local/attempt-0000/game-trace.jsonl").exists())
        resumed = self.run_object()
        self.assertEqual(resumed.checkpoint, final)
        resumed.train("fake-python", "fake-environments")
        self.assertEqual(len(self.fake.calls), 3)

    def test_abandoned_attempt_is_charged_and_partial_weights_are_ignored(self):
        run = self.run_object()
        attempt = run.reserve()
        partial = run.bundle_path(attempt) / "checkpoint"
        partial.mkdir(parents=True)
        (partial / "generator.safetensors").write_text("interrupted")
        resumed = self.run_object()
        self.assertEqual(resumed.attempts[0].state, "failed")
        self.assertEqual(resumed.charged_seconds, 30)
        self.assertEqual(resumed.checkpoint, self.initial)
        resumed.train("fake-python", "fake-environments")
        self.assertEqual(len(self.fake.calls), 2)
        self.assertEqual(resumed.charged_seconds, 90)
        self.assertEqual(self.fake.calls[0][self.fake.calls[0].index("--seed") + 1], 100042)

    def test_unallocatable_remainder_does_not_block_frozen_evaluation(self):
        run = self.run_object(replace(self.spec, total_seconds=72))
        self.assertFalse(run.can_evaluate)
        run.train("fake-python", "fake-environments")
        self.assertEqual(len(self.fake.calls), 2)
        self.assertEqual(run.charged_seconds, 60)
        self.assertIsNone(run.reserve())
        self.assertTrue(run.can_evaluate)

    def test_completed_manifest_recovers_after_ledger_write_gap(self):
        run = self.run_object()
        run.train("fake-python", "fake-environments")
        ledger = json.loads(run.ledger_path.read_text())
        ledger["attempts"][-1]["state"] = "running"
        ledger["attempts"][-1]["completion_sha256"] = None
        run.ledger_path.write_text(json.dumps(ledger))
        resumed = self.run_object()
        self.assertEqual(resumed.attempts[-1].state, "completed")
        self.assertEqual(resumed.checkpoint, self.root / "store/blocks/0002/checkpoint")

    def test_published_block_recovers_after_soft_interrupt_marked_it_failed(self):
        run = self.run_object()

        def interrupt_after_publication(*_):
            raise KeyboardInterrupt

        run.validate_bundle = interrupt_after_publication
        with self.assertRaises(KeyboardInterrupt):
            run.train("fake-python", "fake-environments")
        self.assertEqual(run.attempts[0].state, "failed")
        self.assertTrue((self.root / "store/blocks/0000/completion.json").exists())
        resumed = self.run_object()
        self.assertEqual(resumed.attempts[0].state, "completed")
        self.assertEqual(resumed.charged_seconds, 30)
        self.assertEqual(resumed.checkpoint, self.root / "store/blocks/0000/checkpoint")

    def test_corrupt_checkpoint_fails_resume(self):
        run = self.run_object()
        run.train("fake-python", "fake-environments")
        (run.checkpoint / "generator.safetensors").write_text("corrupted")
        with self.assertRaisesRegex(ValueError, "checksum differs"):
            self.run_object()

    def test_missing_completion_evidence_fails_resume(self):
        run = self.run_object()
        run.train("fake-python", "fake-environments")
        (self.root / "store/blocks/0000/training.jsonl.xz").unlink()
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            self.run_object()

    def test_cross_config_and_changed_initial_checkpoint_fail(self):
        self.run_object()
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            self.run_object(replace(self.spec, total_seconds=120))
        (self.initial / "generator.safetensors").write_text("new initial weights")
        with self.assertRaisesRegex(ValueError, "Starting checkpoint changed"):
            self.run_object()

    def test_failed_verifier_stops_without_an_automatic_retry(self):
        self.fake.reject_verification = True
        run = self.run_object()
        with self.assertRaisesRegex(AssertionError, "No optimizer work"):
            run.train("fake-python", "fake-environments")
        self.assertEqual(len(self.fake.calls), 1)
        self.assertEqual(run.attempts[0].state, "failed")
        self.assertEqual(run.charged_seconds, 30)
        self.assertFalse((self.root / "store/blocks/0000/completion.json").exists())

    def test_frozen_evaluation_uses_exact_final_weights_and_reuses_verified_result(self):
        run = self.run_object()
        run.train("fake-python", "fake-environments")
        calls, checked = [], []

        def evaluation(command, **_):
            calls.append(command)
            output = Path(command[command.index("--output") + 1])
            output.mkdir()
            (output / "summary.json").write_text(json.dumps({
                "episodes": [{"game_id": "ls20-9607627b", "completed_levels": 0, "score": 0, "sdk_actions": 10}],
                "all_games": {"games": 25, "completed_levels": 0},
                "excluding_training_game": {"games": 24, "completed_levels": 0}}))
            for name in ("scorecard.json", "adapter_checks.json"):
                (output / name).write_text("{}")
            (output / "trace.jsonl").write_text("{}\n")

        def verify(output, checkpoint):
            checked.append(checkpoint)
            self.assertEqual(checkpoint, run.checkpoint)
            return {"status": "passed", "checkpoint": checkpoint_hashes(checkpoint)}

        result = frozen_evaluation(run, "fake-python", "games", runner=evaluation, verifier=verify)
        self.assertEqual(result["all_25_public_games"]["games"], 25)
        self.assertEqual(calls[0][calls[0].index("--checkpoint") + 1], run.checkpoint)
        frozen_evaluation(run, "fake-python", "games", runner=evaluation, verifier=verify)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(checked), 4)
        (run.store / "frozen-evaluation/summary.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            frozen_evaluation(run, "fake-python", "games", runner=evaluation, verifier=verify)

    def test_incomplete_evaluation_is_preserved(self):
        run = self.run_object()
        run.train("fake-python", "fake-environments")
        (run.store / "frozen-evaluation").mkdir()
        (run.store / "frozen-evaluation/partial.json").write_text("partial evidence")
        calls = []

        def fail_evaluation(command, **_):
            calls.append(command)
            raise RuntimeError("synthetic evaluation interruption")

        with self.assertRaisesRegex(RuntimeError, "synthetic evaluation interruption"):
            frozen_evaluation(run, "fake-python", "games", runner=fail_evaluation)
        self.assertEqual(calls[0][calls[0].index("--checkpoint") + 1], run.checkpoint)
        self.assertEqual((run.store / "incomplete-frozen-evaluation-0000/partial.json").read_text(), "partial evidence")
        self.assertEqual(json.loads((run.store / "evaluation-attempt-0000.json").read_text())["state"], "failed")
        self.assertEqual(len(self.fake.calls), 3)

    def test_timeout_stops_the_child_process_group(self):
        for log in (None, self.root / "child.log"):
            with self.subTest(log=log):
                pid_file = self.root / "child.pid"
                code = ("import os, pathlib, time; "
                        f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid())); "
                        "os.write(2, b'timeout diagnostic\\n'); time.sleep(30)")
                output = io.StringIO()
                with redirect_stdout(output):
                    with self.assertRaises(subprocess.TimeoutExpired):
                        run_process([sys.executable, "-c", code], timeout=0.3, log=log)
                if log:
                    self.assertIn("timeout diagnostic\n", log.read_text())
                else:
                    self.assertIn("timeout diagnostic\n", output.getvalue())
                pid = int(pid_file.read_text())
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)

    def test_child_failure_reaches_notebook_stdout_and_exception(self):
        output = io.StringIO()
        code = "import os; os.write(1, b'child stdout\\n'); os.write(2, b'child stderr'); raise SystemExit(3)"
        with redirect_stdout(output):
            with self.assertRaises(RuntimeError) as failure:
                run_process([sys.executable, "-c", code], timeout=5)
        self.assertIn("child stdout\nchild stderr", output.getvalue())
        self.assertIn("child stdout\nchild stderr", str(failure.exception))

    def test_successful_child_forwards_both_streams_and_uses_headless_environment(self):
        output = io.StringIO()
        code = ("import os; os.write(1, b'success stdout\\n'); "
                "os.write(2, b'success stderr\\n'); print('tf32=' + os.environ['MLX_ENABLE_TF32']); "
                "print('backend=' + os.environ['MPLBACKEND']); "
                "print('inherited=' + os.environ['ARC3_TEST_INHERITED'])")
        with patch.dict(os.environ, {"MPLBACKEND": "module://matplotlib_inline.backend_inline",
                                    "ARC3_TEST_INHERITED": "preserved"}):
            with redirect_stdout(output):
                run_process([sys.executable, "-c", code], timeout=5)
            self.assertEqual(os.environ["MPLBACKEND"], "module://matplotlib_inline.backend_inline")
        self.assertIn("success stdout\nsuccess stderr\ntf32=0\nbackend=Agg\ninherited=preserved\n",
                      output.getvalue())

    def test_partial_utf8_is_forwarded_before_child_exit(self):
        acknowledgement = self.root / "output-observed"

        class Output(io.StringIO):
            def write(self, value):
                if "\u2603" in value:
                    acknowledgement.touch()
                return super().write(value)

        code = ("import os, pathlib, time; os.write(1, b'partial \\xe2'); time.sleep(0.15); "
                "os.write(2, b'\\x98\\x83'); "
                f"ack = pathlib.Path({str(acknowledgement)!r}); deadline = time.monotonic() + 3\n"
                "while not ack.exists() and time.monotonic() < deadline: time.sleep(0.01)\n"
                "assert ack.exists(), 'Output was not forwarded while child was alive'\n"
                "os.write(1, b' complete')")
        output = Output()
        with redirect_stdout(output):
            run_process([sys.executable, "-c", code], timeout=5)
        self.assertIn("partial \u2603 complete", output.getvalue())
        self.assertNotIn("\ufffd", output.getvalue())

    def test_large_output_has_bounded_error_tail_and_complete_explicit_log(self):
        class Output:
            def __init__(self):
                self.count = 0
                self.tail = ""

            def write(self, value):
                self.count += len(value)
                self.tail = (self.tail + value)[-128:]

            def flush(self):
                pass

        code = ("import os; os.write(1, b'X' * 200000); os.write(2, b'Y' * 200000); "
                "os.write(2, b'last diagnostic'); raise SystemExit(7)")
        expected_tail = "Y" * (8000 - len("last diagnostic")) + "last diagnostic"
        for log in (None, self.root / "large-child.log"):
            with self.subTest(log=log):
                output = Output()
                with redirect_stdout(output):
                    with self.assertRaises(RuntimeError) as failure:
                        run_process([sys.executable, "-c", code], timeout=5, log=log)
                self.assertTrue(str(failure.exception).endswith(expected_tail))
                self.assertLess(len(str(failure.exception)), 8300)
                if log:
                    self.assertEqual(log.read_bytes(), b"X" * 200000 + b"Y" * 200000 + b"last diagnostic")
                else:
                    self.assertGreater(output.count, 400000)
                    self.assertTrue(output.tail.endswith("last diagnostic"))

    def test_descendant_is_stopped_after_leader_exits(self):
        for exit_code, logged in ((0, True), (2, True), (0, False), (2, False)):
            with self.subTest(exit_code=exit_code, logged=logged):
                pid_file = self.root / f"descendant-{exit_code}-{logged}.pid"
                child = "import time; time.sleep(30)"
                leader = ("import pathlib, subprocess, sys; "
                          f"p=subprocess.Popen([sys.executable, '-c', {child!r}]); "
                          f"pathlib.Path({str(pid_file)!r}).write_text(str(p.pid)); sys.exit({exit_code})")
                command = [sys.executable, "-c", leader]
                log = self.root / f"leader-{exit_code}.log" if logged else None
                if exit_code:
                    with self.assertRaisesRegex(RuntimeError, "Process failed"):
                        run_process(command, timeout=5, log=log)
                else:
                    run_process(command, timeout=5, log=log)
                pid = int(pid_file.read_text())
                for _ in range(100):
                    state = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], text=True,
                                           capture_output=True).stdout.strip()
                    if not state or state.startswith("Z"):
                        break
                    time.sleep(0.01)
                self.assertTrue(not state or state.startswith("Z"), f"Descendant {pid} still runs")


class NotebookTests(unittest.TestCase):
    def test_generated_cells_compile_and_pin_safe_commands(self):
        revision = "0123456789abcdef0123456789abcdef01234567"
        artifact = notebook(revision)
        self.assertEqual(artifact["nbformat"], 4)
        self.assertEqual(artifact["metadata"]["source_revision"], revision)
        source = "\n".join(item["source"] for item in artifact["cells"])
        for index, item in enumerate(artifact["cells"]):
            if item["cell_type"] == "code":
                compile(item["source"], f"notebook-cell-{index}", "exec")
                self.assertEqual(item["outputs"], [])
        self.assertIn(f'REVISION = "{revision}"', source)
        for expected in ("TRAIN_HOURS = 4", "bootstrap_environment(REPO, VENV,", "--require-cuda", "arc3.download_games",
                         "arc3.colab_training", "--evaluate", 'os.environ["MLX_ENABLE_TF32"] = "0"',
                         'log=LOCAL / "gpu-selfcheck.log"'):
            self.assertIn(expected, source)
        for forbidden in ("render_mode", "renderer=", "KAGGLE_KEY", "KAGGLE_USERNAME", "getpass", "torch.",
                          'checkout", "main', "api_key ="):
            self.assertNotIn(forbidden, source)

    def test_mutable_revision_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "immutable"):
            notebook("main")

    def test_generated_setup_uses_the_bootstrap_on_a_host_without_venv_or_pip(self):
        revision = "0123456789abcdef0123456789abcdef01234567"
        setup, = [item["source"] for item in notebook(revision)["cells"]
                  if item["cell_type"] == "code" and "REPO_URL =" in item["source"]]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def path(value):
                return root / Path(value).name

            def git_run(command, **_):
                self.assertEqual(command[0], "git", "The host cannot install Python environments or packages.")
                if command[1] == "clone":
                    Path(command[-1]).mkdir()

            def git_output(command, **_):
                if command[-1] == "origin":
                    return "https://github.com/dyl-joseph/ncm-puzzles.git\n"
                if command[-1] == "HEAD":
                    return revision + "\n"
                return ""

            python = root / "managed-python"
            namespace = {"Path": path, "subprocess": SimpleNamespace(run=git_run, check_output=git_output),
                         "os": SimpleNamespace(chdir=lambda _: None),
                         "sys": SimpleNamespace(path=[], executable="/usr/bin/python3")}
            with patch("arc3.colab_bootstrap.bootstrap_environment", return_value=(root / "uv", python)) as bootstrap:
                exec(setup, namespace)
            bootstrap.assert_called_once_with(root / "ncm-puzzles-colab", root / "arc3-isolated-venv",
                                              root / "arc3-bootstrap-tools")
            self.assertEqual(namespace["PYTHON"], python)


if __name__ == "__main__":
    unittest.main()
