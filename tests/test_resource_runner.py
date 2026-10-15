from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch
import torch.nn.functional as F

import resource_runner as runner
from ncm_resource import ResourcePolicy
import tests.test_continue_latin as continuation_tests
import continue_latin as continuation
import run_latin_10min as fresh_training
import train_latin as sealed_trainer


class FakeController:
    def __init__(self, reason=None):
        self.reason = reason
        self.events = []
        self.closed = False

    def begin_unit(self):
        self.events.append("begin")

    def finish_unit(self):
        self.events.append("finish")
        return self.reason

    def close(self):
        self.closed = True
        return {"stop_reason": self.reason, "events": self.events}


class GuardedExecutionTests(unittest.TestCase):
    def core(self):
        optimizer = SimpleNamespace(step=lambda: "applied")
        return SimpleNamespace(make_optimizer=lambda *args: optimizer,
                               generator_loss=lambda *args: ("reference", {}),
                               tokenizer_loss=lambda *args: ("tokenizer", {}),
                               generate_batch=lambda *args: [1, 2, 3])

    def test_optimizer_step_is_local_and_sends_stop_after_applied_update(self):
        core, controller = self.core(), FakeController("new swap writes")
        original_factory, original_loss = core.make_optimizer, core.generator_loss
        with patch.object(runner.os, "kill") as send_signal:
            with runner.guarded_training(core, controller, fast_targets=False):
                optimizer = core.make_optimizer(None, None)
                self.assertEqual(core.generator_loss(None), ("reference", {}))
                self.assertEqual(core.generator_loss(None), ("reference", {}))
                self.assertEqual(optimizer.step(), "applied")
                send_signal.assert_called_once_with(runner.os.getpid(), signal.SIGTERM)
                self.assertEqual(controller.events, ["begin", "begin", "finish"])
        self.assertIs(core.make_optimizer, original_factory)
        self.assertIs(core.generator_loss, original_loss)

    def test_fast_loss_only_replaces_generator_and_context_restores_on_failure(self):
        core, controller = self.core(), FakeController()
        original = dict(vars(core))
        fake_fast = SimpleNamespace(fast_generator_loss=lambda *args: ("fast", {}))
        with patch.dict(sys.modules, {"ncm_fast_targets": fake_fast}):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                with runner.guarded_training(core, controller, fast_targets=True):
                    self.assertEqual(core.generator_loss(None), ("fast", {}))
                    self.assertEqual(core.tokenizer_loss(None), ("tokenizer", {}))
                    raise RuntimeError("test failure")
        for name in original:
            self.assertIs(getattr(core, name), original[name])

    def test_metal_pooling_binds_only_fast_frozen_targets_and_restores_after_failure(self):
        core, controller = self.core(), FakeController()
        model = torch.nn.Linear(2, 1)
        core.make_generator = lambda *args: model
        original = dict(vars(core))
        original_forward = model.forward
        calls = []

        def fast_loss(*args, pooling_kernel="reference"):
            calls.append((args, pooling_kernel))
            return "fast", {}

        with patch.dict(sys.modules, {"ncm_fast_targets": SimpleNamespace(fast_generator_loss=fast_loss)}):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                with runner.guarded_training(core, controller, fast_targets=True, frozen_pooling="metal"):
                    self.assertEqual(core.generator_loss(model, "tokenizer", "solutions", "prefixes"), ("fast", {}))
                    self.assertEqual(core.tokenizer_loss(None), ("tokenizer", {}))
                    self.assertIs(core.make_generator, original["make_generator"])
                    self.assertEqual(model.forward, original_forward)
                    raise RuntimeError("test failure")
        self.assertEqual(calls, [((model, "tokenizer", "solutions", "prefixes"), "metal")])
        for name in original:
            self.assertIs(getattr(core, name), original[name])
        with patch.dict(sys.modules, {"ncm_fast_targets": SimpleNamespace(fast_generator_loss=fast_loss)}):
            with runner.guarded_training(core, controller, fast_targets=False, frozen_pooling="metal"):
                self.assertEqual(core.generator_loss(None), ("reference", {}))
        self.assertEqual(len(calls), 1)

    def test_evaluation_paces_each_batch_and_stops_on_unhealthy_memory(self):
        core, controller = self.core(), FakeController()
        original = core.generate_batch
        with runner.guarded_evaluation(core, controller):
            self.assertEqual(core.generate_batch(None), [1, 2, 3])
            controller.reason = "memory pressure"
            with self.assertRaisesRegex(RuntimeError, "memory pressure"):
                core.generate_batch(None)
        self.assertIs(core.generate_batch, original)
        self.assertEqual(controller.events, ["begin", "finish", "begin", "finish"])

    def test_recipe_binds_policy_and_runtime_source_hashes(self):
        execution = runner.execution_descriptor()
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.yaml"
            OmegaConf.save(OmegaConf.create({"local_choices": {"execution": execution}}), config)
            runner.verify_training_execution(["generator", "--config", str(config)], execution)
            changed = runner.execution_descriptor(replace(ResourcePolicy(), gpu_duty_cycle=0.7))
            with self.assertRaisesRegex(ValueError, "recorded training recipe"):
                runner.verify_training_execution(["generator", "--config", str(config)], changed)
            tampered = json.loads(json.dumps(execution))
            tampered["runtime_sha256"]["resource_runner.py"] = "changed"
            with self.assertRaisesRegex(ValueError, "recorded training recipe"):
                runner.verify_training_execution(["--config=" + str(config)], tampered)

    def test_limits_configure_before_core_runs_and_report_is_saved(self):
        controller = FakeController()
        events = []
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evaluation"

            def factory(*args):
                events.append("configure")
                return controller

            def main(arguments):
                events.append("core")
                self.assertEqual(core.generate_batch(None), [1, 2, 3])
                return 0

            core = self.core()
            core.main = main
            stdout = io.StringIO()
            with patch("ncm_resource.make_mlx_controller", side_effect=factory), \
                    patch.dict(sys.modules, {"mlx.core": SimpleNamespace(), "evaluate_latin": core}), \
                    patch.object(torch.backends.cuda, "allow_fp16_bf16_reduction_math_sdp") as select_attention, \
                    patch.object(torch, "autocast") as autocast, \
                    redirect_stdout(stdout):
                self.assertEqual(runner.main(["evaluate", "--output", str(output), "--device", "mlx"]), 0)
            select_attention.assert_not_called()
            autocast.assert_not_called()
            self.assertIn("MLX allocator guideline", stdout.getvalue())
            self.assertEqual(events, ["configure", "core"])
            self.assertTrue(controller.closed)
            report = json.loads((output / "resources-evaluate.json").read_text())
            self.assertEqual(report["returncode"], 0)
            self.assertEqual(report["execution"]["generator_precision"], "not_applicable")
            self.assertEqual(report["execution"]["attention_kernel"], "not_applicable")
            self.assertEqual(report["execution"]["frozen_pooling"], "not_applicable")
            self.assertEqual(report["execution"]["checkpoint_copies"], "not_applicable")
            self.assertEqual(report["resources"]["events"], ["begin", "begin", "finish"])

    def test_startup_failure_does_not_import_model_or_run_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evaluation"
            core = self.core()
            core.main = lambda arguments: self.fail("Model execution must not start.")
            with patch("ncm_resource.make_mlx_controller", side_effect=RuntimeError("memory pressure")), \
                    patch.dict(sys.modules, {"mlx.core": SimpleNamespace(), "evaluate_latin": core}), \
                    redirect_stderr(io.StringIO()):
                self.assertEqual(runner.main(["evaluate", "--output", str(output)]), 1)
            report = json.loads((output / "resources-evaluate.json").read_text())
            self.assertEqual(report["error"]["message"], "memory pressure")

    def test_guard_stop_returns_failure_even_after_core_saves_and_returns_success(self):
        controller = FakeController("New swap writes detected.")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evaluation"
            core = self.core()
            core.main = lambda arguments: 0
            with patch("ncm_resource.make_mlx_controller", return_value=controller), \
                    patch.dict(sys.modules, {"mlx.core": SimpleNamespace(), "evaluate_latin": core}), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(["evaluate", "--output", str(output)]), 1)
            report = json.loads((output / "resources-evaluate.json").read_text())
            self.assertEqual(report["returncode"], 1)
            self.assertEqual(report["resources"]["stop_reason"], "New swap writes detected.")


class ForwardedArgumentTests(unittest.TestCase):
    def test_lookup_matches_last_occurrence_for_each_protected_option(self):
        for option in ("--device", "--config", "--output", "--run"):
            for arguments in ([option, "first", option + "=last"],
                              [option + "=first", option, "last"],
                              [option, "first", option, "last"],
                              [option + "=first", option + "=last"]):
                with self.subTest(option=option, arguments=arguments):
                    self.assertEqual(runner.argument_value(arguments, option), "last")
            self.assertEqual(runner.argument_value([option, "first", "--", option, "last"], option), "first")

    def test_protected_option_abbreviations_are_rejected(self):
        for option in ("--dev", "--conf=path", "--out", "--ru"):
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, "full names"):
                runner.validate_forwarded_arguments([option])
        runner.validate_forwarded_arguments(["--device=mps", "--config", "path", "--resume"])

    def test_recipe_validation_uses_the_last_config(self):
        execution = runner.execution_descriptor()
        with tempfile.TemporaryDirectory() as directory:
            good, bad = Path(directory) / "good.yaml", Path(directory) / "bad.yaml"
            OmegaConf.save(OmegaConf.create({"local_choices": {"execution": execution}}), good)
            OmegaConf.save(OmegaConf.create({"local_choices": {}}), bad)
            runner.verify_training_execution(["--config", str(bad), "--config=" + str(good)], execution)
            with self.assertRaisesRegex(ValueError, "recorded training recipe"):
                runner.verify_training_execution(["--config=" + str(good), "--config", str(bad)], execution)

    def test_final_device_selects_the_backend_before_core_execution(self):
        for device_arguments, expected_backend in ((["--device", "mlx", "--device=mps"], "mps"),
                                                   (["--device=mps", "--device", "mlx"], "mlx"),
                                                   ([], "mlx")):
            with self.subTest(arguments=device_arguments), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "evaluation"
                controller = FakeController()
                events = []
                core = SimpleNamespace(generate_batch=lambda *args: [])

                def main(arguments):
                    events.append("core")
                    self.assertEqual(controller.events, ["begin"])
                    return 0

                core.main = main
                with patch("ncm_resource.make_mlx_controller", side_effect=lambda *args: events.append("mlx") or controller), \
                        patch("ncm_resource.make_torch_controller", side_effect=lambda *args: events.append("mps") or controller), \
                        patch.dict(sys.modules, {"mlx.core": SimpleNamespace(), "evaluate_latin": core}), \
                        redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.main(["evaluate", "--output", str(output), *device_arguments]), 0)
                self.assertEqual(events, [expected_backend, "core"])

    def test_training_rejects_mlx_and_unsupported_stages_before_backend_setup(self):
        execution = runner.execution_descriptor()
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "recipe.yaml"
            OmegaConf.save(OmegaConf.create({"local_choices": {"execution": execution}}), config)
            with patch("ncm_resource.make_mlx_controller") as mlx_factory, \
                    patch("ncm_resource.make_torch_controller") as torch_factory, \
                    redirect_stderr(io.StringIO()):
                self.assertEqual(runner.main(["train", "generator", "--config", str(config),
                                              "--run", directory, "--device=mps", "--device", "mlx"]), 1)
                for stage in ("tokenizer", "all"):
                    with self.subTest(stage=stage), self.assertRaises(SystemExit) as error:
                        runner.main(["train", stage, "--config", str(config), "--run", directory])
                    self.assertEqual(error.exception.code, 2)
            mlx_factory.assert_not_called()
            torch_factory.assert_not_called()

    def test_reports_follow_the_last_run_and_output_destination(self):
        execution = runner.execution_descriptor()
        for mode, option in (("evaluate", "--output"), ("train", "--run")):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                first, last = Path(directory) / "first", Path(directory) / "last"
                config = Path(directory) / "recipe.yaml"
                OmegaConf.save(OmegaConf.create({"local_choices": {"execution": execution}}), config)
                core = SimpleNamespace(make_optimizer=lambda *args: None, make_generator=lambda *args: None,
                                       generator_loss=lambda *args: None,
                                       tokenizer_loss=lambda *args: None, generate_batch=lambda *args: [],
                                       cpu_tree=sealed_trainer.cpu_tree,
                                       main=lambda arguments: 0)
                arguments = [mode]
                if mode == "train":
                    arguments.extend(["generator", "--config", str(config)])
                arguments.extend([option, str(first), option + "=" + str(last)])
                with patch("ncm_resource.make_mlx_controller", return_value=FakeController()), \
                        patch("ncm_resource.make_torch_controller", return_value=FakeController()), \
                        patch.dict(sys.modules, {"mlx.core": SimpleNamespace(), "evaluate_latin": core,
                                                 "train_latin": core}), redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.main(arguments), 0)
                self.assertFalse(first.exists())
                self.assertTrue((last / f"resources-{mode}.json").is_file())


class GeneratorPrecisionTests(unittest.TestCase):
    def setUp(self):
        previous = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
        self.addCleanup(torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp, previous)
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)

    def test_bf16_scope_keeps_frozen_preprocessing_loss_and_saved_state_fp32(self):
        observed = []

        def observe(label):
            observed.append((label, torch.is_autocast_enabled("cpu"),
                             torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()))

        class Generator(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(2, 2)

            def forward(self, inputs):
                observe("generator")
                return self.projection(inputs)

        model = Generator()
        original_forward = model.forward
        original_keys = set(model.state_dict())
        frozen = torch.nn.Linear(2, 2).eval().requires_grad_(False)

        def generator_loss(model, tokenizer, solutions, prefixes):
            with torch.no_grad():
                observe("frozen")
                contexts = tokenizer(solutions)
                self.assertEqual(contexts.dtype, torch.float32)
            logits = model(contexts)
            self.assertEqual(logits.dtype, torch.float32)
            observe("loss")
            logits.register_hook(lambda gradient: observe("backward"))
            loss = F.cross_entropy(logits, torch.tensor([0, 1]))
            self.assertEqual(loss.dtype, torch.float32)
            return loss, {}

        core = SimpleNamespace(make_generator=lambda *args: model,
                               make_optimizer=lambda model, *args: torch.optim.AdamW(model.parameters(), lr=1e-3),
                               generator_loss=generator_loss, tokenizer_loss=lambda *args: None,
                               generate_batch=lambda *args: [])
        original_factory = core.make_generator
        autocast = torch.autocast

        def cpu_autocast(*, device_type, dtype):
            self.assertEqual(device_type, "mps")
            self.assertEqual(dtype, torch.bfloat16)
            return autocast(device_type="cpu", dtype=dtype)

        with patch.object(torch, "autocast", side_effect=cpu_autocast):
            with runner.guarded_training(core, FakeController(), fast_targets=False,
                                         generator_precision="bf16", attention_kernel="bf16-math"):
                generated = core.make_generator(None, frozen, "cpu")
                optimizer = core.make_optimizer(generated, None)
                loss, _ = core.generator_loss(generated, frozen, torch.ones(2, 2), None)
                loss.backward()
                self.assertTrue(all(parameter.dtype == torch.float32 and parameter.grad.dtype == torch.float32
                                    for parameter in generated.parameters()))
                self.assertTrue(all(parameter.grad is None for parameter in frozen.parameters()))
                optimizer.step()
                self.assertEqual(set(generated.state_dict()), original_keys)
                with tempfile.TemporaryDirectory() as directory:
                    checkpoint = Path(directory) / "state.pt"
                    sealed_trainer.save_boundary(checkpoint, generated, optimizer,
                                                 SimpleNamespace(state_dict=lambda: {"cursor": 32}), 1,
                                                 {"stage": {"name": "generator"}}, "cpu")
                    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                    self.assertEqual(saved["format"], "ncm-latin-training-state-v1")
                    self.assertTrue(all(value.dtype == torch.float32 for value in saved["model"].values()))
                    self.assertTrue(all(value.dtype == torch.float32
                                        for state in saved["optimizer"]["state"].values()
                                        for value in state.values() if isinstance(value, torch.Tensor)))
        self.assertEqual(observed, [("frozen", False, False), ("generator", True, True),
                                    ("loss", False, False), ("backward", False, False)])
        self.assertEqual(model.forward, original_forward)
        self.assertIs(core.make_generator, original_factory)
        self.assertFalse(torch.is_autocast_enabled("cpu"))
        self.assertFalse(torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed())

    def test_reference_bf16_forward_restores_an_existing_reduced_math_choice_on_failure(self):
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)

        class Generator(torch.nn.Module):
            def forward(model, inputs):
                self.assertFalse(torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed())
                self.assertTrue(torch.is_autocast_enabled("cpu"))
                raise RuntimeError("generator forward failed")

        model = Generator()
        original_forward = model.forward
        core = SimpleNamespace(make_generator=lambda *args: model, make_optimizer=lambda *args: None,
                               generator_loss=lambda *args: None, tokenizer_loss=lambda *args: None,
                               generate_batch=lambda *args: [])
        autocast = torch.autocast
        with patch.object(torch, "autocast", side_effect=lambda **kwargs: autocast("cpu", dtype=kwargs["dtype"])):
            with runner.guarded_training(core, FakeController(), fast_targets=False,
                                         generator_precision="bf16", attention_kernel="reference"):
                generated = core.make_generator(None)
                with self.assertRaisesRegex(RuntimeError, "generator forward failed"):
                    generated(torch.ones(2, 2))
                self.assertTrue(torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed())
                self.assertFalse(torch.is_autocast_enabled("cpu"))
        self.assertEqual(model.forward, original_forward)

    def test_fp32_path_does_not_install_autocast(self):
        core = SimpleNamespace(make_optimizer=lambda *args: None, generator_loss=lambda *args: ("fp32", {}),
                               tokenizer_loss=lambda *args: None, generate_batch=lambda *args: [])
        with patch.object(torch, "autocast") as autocast, \
                patch.object(torch.backends.cuda, "allow_fp16_bf16_reduction_math_sdp") as select_attention:
            with runner.guarded_training(core, FakeController(), fast_targets=False, generator_precision="fp32"):
                self.assertEqual(core.generator_loss(None), ("fp32", {}))
        autocast.assert_not_called()
        select_attention.assert_not_called()

    def test_precision_is_bound_to_training_recipe_and_invalid_values_are_rejected(self):
        self.assertEqual(runner.execution_descriptor()["generator_precision"], "bf16")
        self.assertEqual(runner.execution_descriptor()["attention_kernel"], "bf16-math")
        fp32 = runner.execution_descriptor(generator_precision="fp32")
        self.assertEqual(fp32["attention_kernel"], "reference")
        self.assertEqual(runner.execution_descriptor(attention_kernel="reference")["attention_kernel"], "reference")
        self.assertEqual(runner.execution_descriptor(generator_precision="not_applicable",
                                                     attention_kernel="bf16-math")["attention_kernel"],
                         "not_applicable")
        self.assertNotEqual(fp32, runner.execution_descriptor())
        with self.assertRaisesRegex(ValueError, "precision"):
            runner.execution_descriptor(generator_precision="float16")
        with self.assertRaisesRegex(ValueError, "precision"):
            with runner.guarded_training(None, FakeController(), fast_targets=False, generator_precision="float16"):
                self.fail("Invalid precision must not enter the training context.")
        for main, arguments in ((runner.main, ["train", "generator"]),
                                (continuation.main, []), (fresh_training.main, [])):
            with self.subTest(main=main), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main([*arguments, "--generator-precision", "float16"])

    def test_frozen_pooling_is_bound_to_execution_and_invalid_choices_are_rejected(self):
        self.assertEqual(runner.execution_descriptor()["frozen_pooling"], "metal")
        self.assertEqual(runner.execution_descriptor(frozen_pooling="reference")["frozen_pooling"], "reference")
        self.assertEqual(runner.execution_descriptor(fast_targets=False)["frozen_pooling"], "not_applicable")
        self.assertEqual(runner.execution_descriptor(generator_precision="not_applicable")["frozen_pooling"],
                         "not_applicable")
        self.assertIn("ncm_area_pool.py", runner.execution_descriptor()["runtime_sha256"])
        with self.assertRaisesRegex(ValueError, "pooling"):
            runner.execution_descriptor(frozen_pooling="approximate")
        with self.assertRaisesRegex(ValueError, "pooling"):
            with runner.guarded_training(None, FakeController(), fast_targets=True, frozen_pooling="approximate"):
                self.fail("Invalid frozen pooling must not enter the training context.")
        for main, arguments in ((runner.main, ["train", "generator"]),
                                (continuation.main, []), (fresh_training.main, [])):
            with self.subTest(main=main), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main([*arguments, "--frozen-pooling", "approximate"])

    def test_incompatible_or_unsupported_kernel_fails_before_backend_or_state_writes(self):
        for kernel in ("flash", "bf16-math"):
            with self.subTest(kernel=kernel):
                with self.assertRaisesRegex(ValueError, "kernel"):
                    runner.execution_descriptor(generator_precision="fp32", attention_kernel=kernel)
                with self.assertRaisesRegex(ValueError, "kernel"):
                    with runner.guarded_training(None, FakeController(), fast_targets=False,
                                                 generator_precision="fp32", attention_kernel=kernel):
                        self.fail("An invalid kernel must not enter the training context.")
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "untouched"
            with patch("ncm_resource.make_mlx_controller") as mlx_factory, \
                    patch("ncm_resource.make_torch_controller") as torch_factory, \
                    patch.object(continuation, "continuation_lock") as continuation_lock, \
                    patch.object(fresh_training, "deadline_lock") as deadline_lock, \
                    redirect_stderr(io.StringIO()):
                flags = ["--run", str(run), "--generator-precision", "fp32", "--attention-kernel", "bf16-math"]
                with self.assertRaises(SystemExit) as error:
                    runner.main(["train", "generator", *flags])
                self.assertEqual(error.exception.code, 2)
                self.assertEqual(continuation.main(flags), 1)
                self.assertEqual(fresh_training.main(flags), 1)
            mlx_factory.assert_not_called()
            torch_factory.assert_not_called()
            continuation_lock.assert_not_called()
            deadline_lock.assert_not_called()
            self.assertFalse(run.exists())

    def test_training_cli_passes_the_resolved_kernel_to_the_generator_context(self):
        choices = (([], "bf16", "bf16-math", "metal"),
                   (["--attention-kernel", "reference"], "bf16", "reference", "metal"),
                   (["--generator-precision", "fp32"], "fp32", "reference", "metal"),
                   (["--frozen-pooling", "reference"], "bf16", "bf16-math", "reference"))
        for flags, precision, kernel, pooling in choices:
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as directory:
                config = Path(directory) / "recipe.yaml"
                execution = runner.execution_descriptor(generator_precision=precision, attention_kernel=kernel,
                                                        frozen_pooling=pooling)
                OmegaConf.save(OmegaConf.create({"local_choices": {"execution": execution}}), config)
                received = []

                @contextmanager
                def training(core, controller, **kwargs):
                    received.append(kwargs)
                    yield

                core = SimpleNamespace(main=lambda arguments: 0)
                with patch("ncm_resource.make_torch_controller", return_value=FakeController()), \
                        patch.dict(sys.modules, {"train_latin": core}), \
                        patch.object(runner, "guarded_training", training), redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.main(["train", *flags, "generator", "--config", str(config),
                                                  "--run", directory]), 0)
                self.assertEqual(received, [{"fast_targets": True, "generator_precision": precision,
                                             "attention_kernel": kernel, "frozen_pooling": pooling,
                                             "checkpoint_copies": "batched"}])
                report = json.loads((Path(directory) / "resources-train.json").read_text())
                self.assertEqual(report["execution"]["attention_kernel"], kernel)
                self.assertEqual(report["execution"]["frozen_pooling"], pooling)


class ContinuationResourceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = continuation_tests.ContinuationBehaviorTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_changed_policy_rejects_resume_without_changing_saved_state(self):
        fixture = self.fixture
        lineage, _ = fixture.prepare()
        before = (fixture.run / "generator/state.pt").read_bytes()
        self.assertEqual(lineage["execution"]["policy"]["gpu_duty_cycle"], 0.8)
        execution = runner.execution_descriptor(replace(ResourcePolicy(), gpu_duty_cycle=0.7))
        with self.assertRaisesRegex(ValueError, "lineage"):
            continuation.prepare_continuation(fixture.source, fixture.run, True, execution)
        self.assertEqual((fixture.run / "generator/state.pt").read_bytes(), before)

    def test_changed_generator_precision_rejects_resume_without_changing_state(self):
        fixture = self.fixture
        fixture.prepare()
        before = (fixture.run / "generator/state.pt").read_bytes()
        execution = runner.execution_descriptor(generator_precision="fp32")
        with self.assertRaisesRegex(ValueError, "lineage"):
            continuation.prepare_continuation(fixture.source, fixture.run, True, execution)
        self.assertEqual((fixture.run / "generator/state.pt").read_bytes(), before)

    def test_changed_attention_kernel_rejects_resume_without_changing_saved_files(self):
        fixture = self.fixture
        lineage, _ = fixture.prepare()
        self.assertEqual(lineage["execution"]["attention_kernel"], "bf16-math")
        before = {str(path.relative_to(fixture.run)): path.read_bytes()
                  for path in fixture.run.rglob("*") if path.is_file()}
        execution = runner.execution_descriptor(attention_kernel="reference")
        with patch.object(continuation, "atomic_torch_save") as save_state, \
                self.assertRaisesRegex(ValueError, "lineage"):
            continuation.prepare_continuation(fixture.source, fixture.run, True, execution)
        save_state.assert_not_called()
        self.assertEqual({str(path.relative_to(fixture.run)): path.read_bytes()
                          for path in fixture.run.rglob("*") if path.is_file()}, before)

    def test_changed_frozen_pooling_rejects_resume_without_changing_saved_files(self):
        fixture = self.fixture
        lineage, _ = fixture.prepare()
        self.assertEqual(lineage["execution"]["frozen_pooling"], "metal")
        before = {str(path.relative_to(fixture.run)): path.read_bytes()
                  for path in fixture.run.rglob("*") if path.is_file()}
        execution = runner.execution_descriptor(frozen_pooling="reference")
        with patch.object(continuation, "atomic_torch_save") as save_state, \
                self.assertRaisesRegex(ValueError, "lineage"):
            continuation.prepare_continuation(fixture.source, fixture.run, True, execution)
        save_state.assert_not_called()
        self.assertEqual({str(path.relative_to(fixture.run)): path.read_bytes()
                          for path in fixture.run.rglob("*") if path.is_file()}, before)

    def test_smaller_microbatch_preserves_global_batch_and_all_training_state(self):
        fixture = self.fixture
        source_before = {str(path.relative_to(fixture.source)): path.read_bytes()
                         for path in fixture.source.rglob("*") if path.is_file()}
        lineage, completed = continuation.prepare_continuation(fixture.source, fixture.run, False,
                                                               microbatch_size=16)
        migrated = fixture.checkpoint()
        self.assertEqual(completed, 400)
        self.assertEqual(lineage["global_batch_size"], 32)
        self.assertEqual(lineage["microbatch_size"], 16)
        self.assertEqual(migrated["immutable"]["stage"]["batch_size"], 32)
        self.assertEqual(migrated["immutable"]["stage"]["microbatch_size"], 16)
        self.assertEqual(fixture.state["immutable"]["stage"]["microbatch_size"], 32)
        for name in fixture.state:
            if name != "immutable":
                continuation_tests.assert_tree_equal(self, fixture.state[name], migrated[name])
        recipe = OmegaConf.load(fixture.run / "continuation.yaml")
        self.assertEqual(recipe.generator.microbatch_size, 16)
        self.assertIn("gradient_accumulation", recipe.local_choices.generator_batch)
        self.assertEqual({str(path.relative_to(fixture.source)): path.read_bytes()
                          for path in fixture.source.rglob("*") if path.is_file()}, source_before)
        self.assertEqual(continuation.prepare_continuation(fixture.source, fixture.run, True,
                                                          microbatch_size=16)[1], 400)

    def test_changed_overhead_choices_reject_resume_without_mutating_saved_files(self):
        fixture = self.fixture
        fixture.prepare()
        before = {str(path.relative_to(fixture.run)): path.read_bytes()
                  for path in fixture.run.rglob("*") if path.is_file()}
        for execution in (runner.execution_descriptor(replace(ResourcePolicy(), pacing_method="full-sleep")),
                          runner.execution_descriptor(checkpoint_copies="blocking")):
            with self.subTest(execution=execution), self.assertRaisesRegex(ValueError, "lineage"):
                continuation.prepare_continuation(fixture.source, fixture.run, True, execution)
            self.assertEqual({str(path.relative_to(fixture.run)): path.read_bytes()
                              for path in fixture.run.rglob("*") if path.is_file()}, before)

    def test_continuation_forwards_reference_overhead_choices(self):
        fixture = self.fixture
        def child(command, *args):
            self.assertEqual(command[command.index("--pacing-method") + 1], "full-sleep")
            self.assertEqual(command[command.index("--checkpoint-copies") + 1], "blocking")
            recipe = OmegaConf.load(command[command.index("--config") + 1])
            self.assertEqual(recipe.local_choices.execution.policy.pacing_method, "full-sleep")
            self.assertEqual(recipe.local_choices.execution.checkpoint_copies, "blocking")
            fixture.save_success()
            return 0, "completed"
        with patch.object(continuation, "run_child", side_effect=child), redirect_stdout(io.StringIO()):
            self.assertEqual(continuation.main(["--source", str(fixture.source), "--run", str(fixture.run),
                                              "--pacing-method", "full-sleep", "--checkpoint-copies", "blocking"]), 0)

    def test_fresh_training_forwards_reference_overhead_choices(self):
        fixture = self.fixture
        def child(command, *args):
            self.assertEqual(command[command.index("--pacing-method") + 1], "full-sleep")
            self.assertEqual(command[command.index("--checkpoint-copies") + 1], "blocking")
            recipe = OmegaConf.load(command[command.index("--config") + 1])
            self.assertEqual(recipe.local_choices.execution.policy.pacing_method, "full-sleep")
            self.assertEqual(recipe.local_choices.execution.checkpoint_copies, "blocking")
            return 0, "completed"
        with patch.object(fresh_training, "prepare_run"), \
                patch.object(fresh_training, "run_child", side_effect=child), \
                patch.object(fresh_training, "verify_ready", return_value=1), redirect_stdout(io.StringIO()):
            self.assertEqual(fresh_training.main(["--run", str(fixture.run), "--pacing-method", "full-sleep",
                                                "--checkpoint-copies", "blocking"]), 0)

    def test_microbatch_must_divide_the_preserved_global_batch(self):
        fixture = self.fixture
        for size in (0, 3, 64, True):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "divide"):
                continuation.prepare_continuation(fixture.source, fixture.run, False,
                                                  microbatch_size=size)
            self.assertFalse(fixture.run.exists())

    def test_fresh_guarded_recipe_accumulates_two_microbatches(self):
        fixture = self.fixture
        fixture.run.mkdir()
        config = fresh_training.prepare_resource_recipe(fixture.run, False, runner.execution_descriptor())
        recipe = OmegaConf.load(config)
        self.assertEqual(recipe.generator.batch_size, 32)
        self.assertEqual(recipe.generator.microbatch_size, 16)
        self.assertEqual(OmegaConf.load(fresh_training.CONFIG).generator.microbatch_size, 32)
        self.assertIn("two_accumulation_steps", recipe.local_choices.generator_batch)
        self.assertEqual(recipe.local_choices.execution.attention_kernel, "bf16-math")
        self.assertEqual(recipe.local_choices.execution.frozen_pooling, "metal")
        self.assertEqual(fresh_training.prepare_resource_recipe(fixture.run, True,
                                                                runner.execution_descriptor()), config)
        before = config.read_bytes()
        for execution in (runner.execution_descriptor(attention_kernel="reference"),
                          runner.execution_descriptor(frozen_pooling="reference"),
                          runner.execution_descriptor(generator_precision="fp32")):
            with self.subTest(execution=execution), \
                    self.assertRaisesRegex(ValueError, "resource recipe"):
                fresh_training.prepare_resource_recipe(fixture.run, True, execution)
            self.assertEqual(config.read_bytes(), before)

    def test_user_interruption_saves_but_does_not_automatically_evaluate(self):
        fixture = self.fixture

        def child(command, *args):
            self.assertEqual(Path(command[1]).name, "resource_runner.py")
            self.assertEqual(command[2], "train")
            self.assertEqual(command[command.index("--generator-precision") + 1], "bf16")
            self.assertEqual(command[command.index("--attention-kernel") + 1], "bf16-math")
            self.assertEqual(command[command.index("--frozen-pooling") + 1], "metal")
            fixture.save_success()
            return 0, "interrupted"

        with patch.object(continuation, "run_child", side_effect=child), \
                patch.object(continuation, "evaluate") as evaluate, redirect_stdout(io.StringIO()):
            self.assertEqual(continuation.main(["--source", str(fixture.source), "--run", str(fixture.run),
                                              "--evaluate"]), 0)
        evaluate.assert_not_called()
        result = continuation.read_json(fixture.run / "continuation-result.json")
        self.assertTrue(result["ready"])
        self.assertEqual(result["stop_reason"], "interrupted")

    def test_continuation_forwards_reference_pooling_and_records_the_same_recipe(self):
        fixture = self.fixture

        def child(command, *args):
            self.assertEqual(command[command.index("--frozen-pooling") + 1], "reference")
            recipe = OmegaConf.load(command[command.index("--config") + 1])
            self.assertEqual(recipe.local_choices.execution.frozen_pooling, "reference")
            fixture.save_success()
            return 0, "completed"

        with patch.object(continuation, "run_child", side_effect=child), redirect_stdout(io.StringIO()):
            self.assertEqual(continuation.main(["--source", str(fixture.source), "--run", str(fixture.run),
                                              "--frozen-pooling", "reference"]), 0)
        self.assertEqual(continuation.read_json(fixture.run / "continuation.json")["execution"]["frozen_pooling"],
                         "reference")

    def test_fresh_training_forwards_and_records_each_frozen_pooling_choice(self):
        fixture = self.fixture
        for pooling in ("metal", "reference"):
            with self.subTest(pooling=pooling):
                run = fixture.root / f"fresh-{pooling}"

                def child(command, *args):
                    self.assertEqual(command[command.index("--frozen-pooling") + 1], pooling)
                    recipe = OmegaConf.load(command[command.index("--config") + 1])
                    self.assertEqual(recipe.local_choices.execution.frozen_pooling, pooling)
                    return 0, "completed"

                flags = [] if pooling == "metal" else ["--frozen-pooling", pooling]
                with patch.object(fresh_training, "prepare_run"), \
                        patch.object(fresh_training, "run_child", side_effect=child), \
                        patch.object(fresh_training, "verify_ready", return_value=1), redirect_stdout(io.StringIO()):
                    self.assertEqual(fresh_training.main(["--run", str(run), *flags]), 0)

    def test_resource_stop_saves_but_does_not_automatically_evaluate(self):
        fixture = self.fixture

        def child(command, *args):
            fixture.save_success()
            self.assertEqual(fixture.checkpoint()["immutable"]["stage"]["microbatch_size"], 16)
            continuation.atomic_json(fixture.run / "resources-train.json", {
                "resources": {"stop_reason": "New swap writes detected."}})
            return 1, "completed"

        with patch.object(continuation, "run_child", side_effect=child), \
                patch.object(continuation, "evaluate") as evaluate, redirect_stdout(io.StringIO()):
            self.assertEqual(continuation.main(["--source", str(fixture.source), "--run", str(fixture.run),
                                              "--evaluate"]), 1)
        evaluate.assert_not_called()
        result = continuation.read_json(fixture.run / "continuation-result.json")
        self.assertTrue(result["ready"])
        self.assertEqual(result["stop_reason"], "resource_guard")
        self.assertEqual(result["resource_stop_reason"], "New swap writes detected.")

    def test_evaluation_keeps_the_recorded_memory_and_duty_budget(self):
        fixture = self.fixture
        execution = runner.execution_descriptor(replace(ResourcePolicy(), gpu_duty_cycle=0.6,
                                                        memory_limit_bytes=6 * 1024 ** 3,
                                                        rss_limit_bytes=6 * 1024 ** 3))
        continuation.prepare_continuation(fixture.source, fixture.run, False, execution)

        def process(command, **kwargs):
            self.assertEqual(Path(command[1]).name, "resource_runner.py")
            self.assertEqual(command[2], "evaluate")
            self.assertEqual(command[command.index("--gpu-duty-cycle") + 1], "0.6")
            self.assertEqual(command[command.index("--memory-gb") + 1], "6.0")
            self.assertEqual(command[command.index("--pacing-method") + 1], "idle-credit")
            output = Path(command[command.index("--output") + 1])
            continuation.atomic_json(output / "summary.json", {"status": "complete", "completed": 8,
                "metrics": {"solved": 0, "given_cell_agreement": 0.9}})

        with patch.object(continuation.subprocess, "run", side_effect=process), redirect_stdout(io.StringIO()):
            continuation.evaluate(fixture.run, 8)


if __name__ == "__main__":
    unittest.main()
