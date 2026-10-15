from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest.mock import patch

import torch

from ncm_resource import ResourceLimitError, ResourcePolicy
from puzzle_runtime import NumericalPolicy, PuzzleRuntime, run_task


class FakeController:
    def __init__(self, *, finish_reason=None, startup_error=None,
                 finish_error=None, close_error=None, close_reason=None):
        self.events = []
        self.reason = None
        self.finish_reason = finish_reason
        self.startup_error = startup_error
        self.finish_error = finish_error
        self.close_error = close_error
        self.close_reason = close_reason

    def check_startup(self):
        self.events.append("startup")
        if self.startup_error is not None:
            raise self.startup_error

    def begin_unit(self):
        self.events.append("begin")

    def finish_unit(self):
        self.events.append("finish")
        self.reason = self.finish_reason or self.reason
        if self.finish_error is not None:
            raise self.finish_error
        return self.reason

    def report(self):
        return {"backend": "test-cpu", "stop_reason": self.reason,
                "events": list(self.events)}

    def close(self):
        self.events.append("close")
        self.reason = self.close_reason or self.reason
        if self.close_error is not None:
            raise self.close_error
        return self.report()


TASK_SOURCE = """
import json
from pathlib import Path
from types import SimpleNamespace

events = []
error = None

def build_task(arguments):
    target = int(arguments[0]) if arguments else 4
    return SimpleNamespace(name="counter-puzzle", metadata={"target": target, "seed": 17},
                           source_paths=[Path(__file__).with_name("dataset.txt")], run=run)

def run(runtime):
    events.append("run")
    checkpoint = runtime.run_directory / "checkpoint.json"
    completed = json.loads(checkpoint.read_text())["completed"] if runtime.resuming else 0
    if error is not None:
        with runtime.work_unit():
            raise error
    if completed < runtime.descriptor["task"]["target"]:
        with runtime.work_unit():
            events.append("applied")
            completed += 1
    checkpoint.write_text(json.dumps({"completed": completed}))
    events.append("saved")
    return {"completed": completed, "resuming": runtime.resuming,
            "stop_requested": runtime.stop_requested, "stop_reason": runtime.stop_reason}
"""


class TaskFixture:
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.module_name = "puzzle_runtime_fixture_" + str(id(self))
        self.source = self.root / (self.module_name + ".py")
        self.source.write_text(textwrap.dedent(TASK_SOURCE))
        self.dataset = self.root / "dataset.txt"
        self.dataset.write_text("deterministic puzzle fixture\n")
        self.run = self.root / "run"
        sys.path.insert(0, str(self.root))
        importlib.invalidate_caches()
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(lambda: sys.path.remove(str(self.root)))
        self.addCleanup(lambda: sys.modules.pop(self.module_name, None))

    def execute(self, controller=None, **overrides):
        controller = controller or FakeController()
        options = {"run_directory": self.run, "device": "cpu",
                   "_controller_factory": lambda device, policy: controller}
        options.update(overrides)
        return run_task(self.module_name, ["4"], **options)

    def read_result(self):
        return json.loads((self.run / "result.json").read_text())


class TaskExecutionTests(TaskFixture, unittest.TestCase):
    def test_backend_is_configured_and_checked_before_task_import(self):
        marker = self.root / "backend-configured"
        self.source.write_text(
            "from pathlib import Path\n"
            f"assert Path({str(marker)!r}).read_text() == 'configured'\n" + TASK_SOURCE)
        controller = FakeController()
        observed = []

        def factory(device, policy):
            observed.append((device, policy))
            marker.write_text("configured")
            return controller

        report = self.execute(_controller_factory=factory)
        self.assertEqual(observed, [("cpu", ResourcePolicy())])
        self.assertEqual(report["task_result"], {"completed": 1, "resuming": False,
                                               "stop_requested": False, "stop_reason": None})
        self.assertEqual(controller.events, ["startup", "begin", "finish", "close"])
        self.assertEqual(json.loads((self.run / "checkpoint.json").read_text()), {"completed": 1})

    def test_resource_stop_preserves_applied_counter_and_checkpoint_and_fails_run(self):
        report = self.execute(FakeController(finish_reason="New swap writes detected."))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["returncode"], 1)
        self.assertEqual(report["error"], None)
        self.assertEqual(report["task_result"], {
            "completed": 1, "resuming": False, "stop_requested": True,
            "stop_reason": "New swap writes detected."})
        self.assertEqual(json.loads((self.run / "checkpoint.json").read_text()), {"completed": 1})
        self.assertEqual(importlib.import_module(self.module_name).events, ["run", "applied", "saved"])
        self.assertEqual(self.read_result(), report)

    def test_final_memory_failure_overrides_task_success(self):
        report = self.execute(FakeController(close_reason="Available RAM fell below reserve."))
        self.assertEqual((report["status"], report["returncode"]), ("failed", 1))
        self.assertEqual(report["task_result"]["completed"], 1)
        self.assertEqual(report["stop_reason"], "Available RAM fell below reserve.")
        resource_report = json.loads((self.run / "resource-report.json").read_text())
        self.assertEqual(resource_report["runtime_status"], "failed")
        self.assertEqual(resource_report["stop_reason"], "Available RAM fell below reserve.")

    def test_original_task_error_survives_both_unit_and_controller_cleanup_errors(self):
        original = LookupError("task failed at a known boundary")
        module = importlib.import_module(self.module_name)
        module.error = original
        controller = FakeController(finish_error=RuntimeError("unit cleanup failed"),
                                    close_error=OSError("controller cleanup failed"))
        with self.assertRaises(LookupError) as raised:
            self.execute(controller)
        self.assertIs(raised.exception, original)
        report = self.read_result()
        self.assertEqual((report["status"], report["returncode"]), ("failed", 1))
        self.assertEqual(report["error"], {"type": "LookupError", "message": str(original)})
        self.assertEqual(controller.events, ["startup", "begin", "finish", "close"])
        self.assertEqual(module.events, ["run"])

    def test_startup_error_is_reported_without_importing_task(self):
        original = ResourceLimitError("Memory pressure prevents startup.")
        controller = FakeController(startup_error=original)
        with self.assertRaises(ResourceLimitError) as raised:
            self.execute(controller)
        self.assertIs(raised.exception, original)
        self.assertNotIn(self.module_name, sys.modules)
        report = json.loads((self.run.parent / f".{self.run.name}.admission-error-report.json").read_text())
        self.assertEqual(report["error"], {
            "type": "ResourceLimitError", "message": "Memory pressure prevents startup."})
        self.assertEqual(controller.events, ["startup", "close"])
        self.assertEqual(list(self.run.iterdir()), [])
        self.assertEqual(self.execute()["task_result"]["completed"], 1)

    def test_fresh_and_resumed_invocations_advance_saved_boundary(self):
        first = self.execute()
        manifest = (self.run / "run-manifest.json").read_bytes()
        second = self.execute(resume=True)
        self.assertEqual((first["status"], first["returncode"]), ("completed", 0))
        self.assertEqual(second["task_result"], {"completed": 2, "resuming": True,
                                                "stop_requested": False, "stop_reason": None})
        self.assertEqual(json.loads((self.run / "checkpoint.json").read_text()), {"completed": 2})
        self.assertEqual((self.run / "run-manifest.json").read_bytes(), manifest)

    def test_resume_rejects_changed_configuration_before_state_changes(self):
        self.execute()
        checkpoint = (self.run / "checkpoint.json").read_bytes()
        previous_report = (self.run / "result.json").read_bytes()
        previous_resources = (self.run / "resource-report.json").read_bytes()
        module = importlib.import_module(self.module_name)
        events = list(module.events)
        with self.assertRaisesRegex(ValueError, "same task"):
            run_task(self.module_name, ["5"], run_directory=self.run, device="cpu", resume=True,
                     _controller_factory=lambda device, policy: FakeController())
        self.assertEqual((self.run / "checkpoint.json").read_bytes(), checkpoint)
        self.assertEqual(module.events, events)
        self.assertEqual((self.run / "result.json").read_bytes(), previous_report)
        self.assertEqual((self.run / "resource-report.json").read_bytes(), previous_resources)
        admission_error = json.loads((self.run.parent / f".{self.run.name}.admission-error-report.json").read_text())
        self.assertEqual(admission_error["returncode"], 1)
        self.assertEqual(admission_error["error"]["type"], "ValueError")

    def test_resume_rejects_changed_dataset_bytes_before_state_changes(self):
        self.execute()
        checkpoint = (self.run / "checkpoint.json").read_bytes()
        self.dataset.write_text("changed puzzle corpus\n")
        with self.assertRaisesRegex(ValueError, "source hashes"):
            self.execute(resume=True)
        self.assertEqual((self.run / "checkpoint.json").read_bytes(), checkpoint)
        self.assertEqual(importlib.import_module(self.module_name).events, ["run", "applied", "saved"])

    def test_resume_rejects_changed_task_source_before_state_changes(self):
        self.execute()
        checkpoint = (self.run / "checkpoint.json").read_bytes()
        self.source.write_text(self.source.read_text() + "\nnew_task_version = 2\n")
        with self.assertRaisesRegex(ValueError, "source hashes"):
            self.execute(resume=True)
        self.assertEqual((self.run / "checkpoint.json").read_bytes(), checkpoint)

    def test_resume_rejects_changed_numerical_and_resource_policies(self):
        self.execute()
        checkpoint = (self.run / "checkpoint.json").read_bytes()
        for options in ({"numerical": NumericalPolicy(checkpoint_copies="blocking")},
                        {"policy": replace(ResourcePolicy(), gpu_duty_cycle=0.7)}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "numerical policy"):
                self.execute(resume=True, **options)
            self.assertEqual((self.run / "checkpoint.json").read_bytes(), checkpoint)
        manifest = json.loads((self.run / "run-manifest.json").read_text())
        self.assertEqual(manifest["numerical"]["checkpoint_copies"], "batched")
        self.assertEqual(manifest["resources"]["gpu_duty_cycle"], 0.8)

    def test_fresh_start_refuses_existing_checkpoint_and_resume_needs_manifest(self):
        self.run.mkdir()
        checkpoint = self.run / "checkpoint.json"
        checkpoint.write_text('{"completed": 9}')
        for options in ({}, {"resume": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.execute(**options)
            self.assertEqual(checkpoint.read_text(), '{"completed": 9}')
        self.assertEqual(importlib.import_module(self.module_name).events, [])

    def test_cpu_reference_execution_does_not_call_mps_or_autocast(self):
        original = importlib.import_module(self.module_name).run

        def run(runtime):
            with runtime.forward_context():
                value = torch.tensor([2.0, -3.0], requires_grad=True)
                loss = value.square().sum()
            loss.backward()
            result = original(runtime)
            result["gradient"] = value.grad.tolist()
            return result

        module = sys.modules[self.module_name]
        module.run = run
        with patch.object(torch.mps, "synchronize", side_effect=AssertionError("MPS execution")), \
                patch.object(torch.mps, "driver_allocated_memory", side_effect=AssertionError("MPS allocation")), \
                patch.object(torch.mps, "set_per_process_memory_fraction", side_effect=AssertionError("MPS setup")), \
                patch.object(torch, "autocast", side_effect=AssertionError("Unexpected autocast")):
            report = self.execute()
        self.assertEqual(report["task_result"]["gradient"], [4.0, -6.0])

    def test_invalid_backend_limits_and_mlx_numerics_fail_before_task_import(self):
        cases = ({"device": "cuda"}, {"seconds": 601}, {"seconds": float("nan")},
                 {"device": "cpu", "numerical": NumericalPolicy(precision="bf16")},
                 {"device": "mlx", "numerical": NumericalPolicy(precision="bf16", frozen_pooling="reference")},
                 {"device": "mlx", "numerical": NumericalPolicy(frozen_pooling="metal")})
        for index, options in enumerate(cases):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.execute(run_directory=self.root / f"invalid-{index}", **options)
        self.assertNotIn(self.module_name, sys.modules)
        report = json.loads((self.root / ".invalid-0.admission-error-report.json").read_text())
        self.assertEqual(report["returncode"], 1)
        self.assertEqual(report["error"]["type"], "ValueError")

    def test_signal_requests_stop_after_applied_boundary_and_restores_handlers(self):
        module = importlib.import_module(self.module_name)
        previous = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}

        def run(runtime):
            with runtime.work_unit():
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
                completed = 1
            (runtime.run_directory / "checkpoint.json").write_text(json.dumps({"completed": completed}))
            return {"completed": completed, "stop_requested": runtime.stop_requested}

        module.run = run
        report = self.execute()
        self.assertEqual(report["task_result"], {"completed": 1, "stop_requested": True})
        self.assertEqual((report["status"], report["returncode"]), ("stopped", 0))
        self.assertEqual(report["stop_reason"], "Received SIGTERM.")
        self.assertEqual({number: signal.getsignal(number) for number in previous}, previous)
        self.assertEqual(json.loads((self.run / "checkpoint.json").read_text()), {"completed": 1})


class RuntimeBoundaryTests(unittest.TestCase):
    def runtime(self, controller=None, **overrides):
        options = {"device": "cpu", "run_directory": Path("unused-test-run"), "resuming": False,
                   "descriptor": {}, "controller": controller or FakeController(),
                   "numerical": NumericalPolicy(), "seconds": 600}
        options.update(overrides)
        return PuzzleRuntime(**options)

    def test_nested_work_units_fail_and_finish_only_the_outer_boundary(self):
        controller = FakeController()
        runtime = self.runtime(controller)
        with self.assertRaisesRegex(RuntimeError, "cannot be nested"):
            with runtime.work_unit():
                with runtime.work_unit():
                    self.fail("The nested body must not run.")
        self.assertEqual(controller.events, ["begin", "finish"])
        with runtime.work_unit():
            value = 7
        self.assertEqual((value, controller.events), (7, ["begin", "finish", "begin", "finish"]))

    def test_checkpoint_conversion_requires_completed_boundary(self):
        runtime = self.runtime()
        tensor = torch.tensor([-0.0, 2.5], requires_grad=True)
        with runtime.work_unit():
            with self.assertRaisesRegex(RuntimeError, "completed work-unit boundary"):
                runtime.cpu_tree({"weights": tensor})
        result = runtime.cpu_tree({"weights": tensor, "counter": 1})
        self.assertEqual(result["weights"].view(torch.uint8).tolist(), tensor.detach().view(torch.uint8).tolist())
        self.assertFalse(result["weights"].requires_grad)
        self.assertEqual(result["counter"], 1)

    def test_attention_selection_restores_process_setting_after_exception(self):
        option = torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp
        original = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
        option(True)
        try:
            with self.assertRaisesRegex(RuntimeError, "forward failed"):
                with self.runtime().forward_context():
                    self.assertFalse(torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed())
                    raise RuntimeError("forward failed")
            self.assertTrue(torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed())
        finally:
            option(original)

    def test_resource_stop_blocks_next_unit_after_first_unit_finishes(self):
        controller = FakeController(finish_reason="RAM reserve reached.")
        runtime = self.runtime(controller)
        with runtime.work_unit():
            completed = 1
        self.assertEqual((completed, runtime.stop_requested, runtime.stop_reason),
                         (1, True, "RAM reserve reached."))
        with self.assertRaisesRegex(ResourceLimitError, "RAM reserve"):
            with runtime.work_unit():
                self.fail("A stopped runtime must not start another update.")
        self.assertEqual(controller.events, ["begin", "finish"])

    def test_deadline_is_cooperative_and_prevents_next_unit(self):
        now = [10.0]
        controller = FakeController()
        runtime = self.runtime(controller, seconds=3, clock=lambda: now[0])
        with runtime.work_unit():
            now[0] = 14.0
            completed = 1
        self.assertEqual(completed, 1)
        self.assertTrue(runtime.stop_requested)
        self.assertEqual(runtime.stop_reason, "The invocation reached its cooperative time limit.")
        with self.assertRaisesRegex(ResourceLimitError, "cooperative"):
            with runtime.work_unit():
                self.fail("The deadline must prevent a subsequent update.")
        self.assertEqual(controller.events, ["begin", "finish"])

    def test_mlx_keeps_forward_native_and_rejects_torch_checkpoint_conversion(self):
        runtime = self.runtime(device="mlx", numerical=NumericalPolicy(frozen_pooling="reference"))
        with patch.object(torch, "autocast", side_effect=AssertionError("Torch autocast in MLX")), \
                patch.object(torch.mps, "synchronize", side_effect=AssertionError("MPS call in MLX")):
            with runtime.forward_context():
                value = 17
            with self.assertRaisesRegex(TypeError, "MLX serialization"):
                runtime.cpu_tree({"weights": "an MLX-owned value"})
        self.assertEqual(value, 17)

    def test_numerical_policy_rejects_unsupported_and_implicit_reduced_math(self):
        for options in ({"precision": "fp16"}, {"attention": "flash"},
                        {"checkpoint_copies": "async"}, {"frozen_pooling": "any"},
                        {"attention": "bf16-math"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                NumericalPolicy(**options)
        numerical = NumericalPolicy(precision="bf16", attention="bf16-math")
        self.assertEqual((numerical.precision, numerical.attention), ("bf16", "bf16-math"))


class SupervisorProcessTests(unittest.TestCase):
    def probe(self, *, ignore_termination, signal_parent):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ready, stopped = root / "ready", root / "stopped"
            child = root / "child.py"
            child.write_text(textwrap.dedent(f"""
                import os
                from pathlib import Path
                import signal
                import time

                def stop(signum, frame):
                    Path({str(stopped)!r}).write_text("SIGTERM")
                    if not {ignore_termination!r}:
                        raise SystemExit(0)

                signal.signal(signal.SIGTERM, stop)
                Path({str(ready)!r}).write_text(str(os.getpid()))
                while True:
                    time.sleep(0.01)
            """))
            supervisor = root / "supervisor.py"
            project = str(Path(__file__).resolve().parents[1])
            supervisor.write_text(textwrap.dedent(f"""
                import json
                import signal
                import sys
                sys.path.insert(0, {project!r})
                from puzzle_runner import _supervise

                previous = {{number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}}
                code, reason = _supervise([sys.executable, {str(child)!r}], 0.8)
                restored = all(signal.getsignal(number) == handler for number, handler in previous.items())
                print(json.dumps({{"code": code, "reason": reason, "restored": restored}}))
            """))
            process = subprocess.Popen([sys.executable, str(supervisor)], stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, start_new_session=True)
            child_pid = None
            try:
                deadline = time.monotonic() + 5
                while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(ready.exists(), "The child must start before testing supervisor signals.")
                child_pid = int(ready.read_text())
                if signal_parent:
                    os.kill(process.pid, signal.SIGTERM)
                output, error = process.communicate(timeout=5)
                self.assertEqual((process.returncode, error), (0, ""))
                self.assertEqual(stopped.read_text(), "SIGTERM")
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
                return json.loads(output)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
                if child_pid is not None:
                    try:
                        os.killpg(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_parent_sigterm_is_forwarded_child_is_reaped_and_handlers_are_restored(self):
        report = self.probe(ignore_termination=False, signal_parent=True)
        self.assertEqual(report, {"code": 0, "reason": "signal", "restored": True})

    def test_hard_deadline_kills_and_reaps_worker_that_ignores_sigterm(self):
        report = self.probe(ignore_termination=True, signal_parent=False)
        self.assertEqual(report, {"code": -signal.SIGKILL, "reason": "killed", "restored": True})


if __name__ == "__main__":
    unittest.main()
