from contextlib import contextmanager, redirect_stdout
import copy
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch
import torch.nn.functional as F

import puzzle_ncm
from puzzle_runtime import NumericalPolicy, PuzzleRuntime
from tests.test_puzzle_runtime import FakeController
from train_latin import make_generator, make_tokenizer


def tensor_record(tensor):
    value = tensor.detach().cpu().contiguous()
    return {"shape": tuple(value.shape), "dtype": str(value.dtype),
            "bytes": value.reshape(-1).view(torch.uint8).numpy().tobytes()}


def state_record(value):
    if isinstance(value, torch.Tensor):
        return tensor_record(value)
    if isinstance(value, dict):
        return {key: state_record(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(state_record(item) for item in value)
    if isinstance(value, list):
        return [state_record(item) for item in value]
    return value


class NoExecution:
    def __getattr__(self, name):
        raise AssertionError(f"Unsupported task inspected a model attribute. {name}")


class TinyPuzzleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(5, 8)
        self.hidden = torch.nn.Linear(8, 12)
        self.dropout = torch.nn.Dropout(0.2)
        self.output = torch.nn.Linear(12, 5)
        self.zero_term = torch.nn.Parameter(torch.tensor([-0.0, 0.0]))

    def forward(self, prefixes):
        hidden = self.dropout(torch.tanh(self.hidden(self.embedding(prefixes))))
        return self.output(hidden) + self.zero_term.sum() * 0


class LossSelectionTests(unittest.TestCase):
    def test_unsupported_16_token_task_calls_its_own_reference_once_without_speculative_work(self):
        model, tokenizer = NoExecution(), NoExecution()
        solutions = torch.tensor([[1, 2, 3, 4] * 4])
        prefixes = torch.tensor([[0, 2, 0, 4] * 4])
        calls = []

        def reference(actual_model, actual_tokenizer, actual_solutions, actual_prefixes):
            calls.append((actual_model, actual_tokenizer, actual_solutions, actual_prefixes))
            return torch.tensor(3.25), {"fixture": "reference result"}

        rng = tensor_record(torch.get_rng_state())
        selection = puzzle_ncm.select_ncm_loss(model, tokenizer, reference,
                                                solution_length=16, prefix_length=16, pooling="metal",
                                                forward_context=lambda: self.fail("Fallback entered a fast forward context."))
        self.assertEqual(selection.implementation, "reference")
        self.assertEqual(selection.pooling, "reference")
        self.assertIn("89 solution tokens", selection.reason)
        self.assertEqual(tensor_record(torch.get_rng_state()), rng)
        loss, metrics = selection(solutions, prefixes)
        self.assertEqual((loss.item(), metrics), (3.25, {"fixture": "reference result"}))
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], model)
        self.assertIs(calls[0][1], tokenizer)
        self.assertIs(calls[0][2], solutions)
        self.assertIs(calls[0][3], prefixes)
        self.assertEqual(tensor_record(torch.get_rng_state()), rng)

    def test_lookalike_models_with_89_tokens_still_use_task_reference(self):
        model, tokenizer = TinyPuzzleModel(), TinyPuzzleModel()
        calls = []

        def reference(*arguments):
            calls.append(arguments)
            return "task-specific result"

        rng = tensor_record(torch.get_rng_state())
        selection = puzzle_ncm.select_ncm_loss(model, tokenizer, reference,
                                                solution_length=89, prefix_length=91)
        self.assertEqual(selection.implementation, "reference")
        self.assertIn("original VQVAE", selection.reason)
        self.assertEqual(selection("solutions", "prefixes"), "task-specific result")
        self.assertEqual(calls, [(model, tokenizer, "solutions", "prefixes")])
        self.assertEqual(tensor_record(torch.get_rng_state()), rng)

    def test_invalid_static_contracts_are_rejected_without_model_access(self):
        for options in ({"solution_length": True, "prefix_length": 16},
                        {"solution_length": 0, "prefix_length": 16},
                        {"solution_length": 16, "prefix_length": -1},
                        {"solution_length": 16, "prefix_length": 16, "pooling": "any"},
                        {"solution_length": 16, "prefix_length": 16, "objective": "weighted"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                puzzle_ncm.select_ncm_loss(NoExecution(), NoExecution(), lambda *args: 1, **options)
        with self.assertRaises(TypeError):
            puzzle_ncm.select_ncm_loss(NoExecution(), NoExecution(), None,
                                       solution_length=16, prefix_length=16)

    def test_reference_runtime_preserves_two_updates_byte_for_byte(self):
        torch.manual_seed(813)
        reference = TinyPuzzleModel().train()
        candidate = copy.deepcopy(reference)
        optimizers = [torch.optim.AdamW(model.parameters(), lr=0.01, betas=(0.8, 0.95))
                      for model in (reference, candidate)]
        solutions = torch.tensor([[1, 2, 3, 4] * 4, [4, 3, 2, 1] * 4])
        prefixes = torch.tensor([[0, 2, 0, 4] * 4, [4, 0, 2, 0] * 4])

        def reference_loss(model, tokenizer, targets, inputs):
            return F.cross_entropy(model(inputs).reshape(-1, 5), targets.reshape(-1)), {}

        selection = puzzle_ncm.select_ncm_loss(candidate, None, reference_loss,
                                                solution_length=16, prefix_length=16, pooling="metal")
        runtime = PuzzleRuntime(device="cpu", run_directory=Path("unused-fixture-run"), resuming=False,
                                descriptor={}, controller=FakeController(), seconds=600,
                                numerical=NumericalPolicy(precision="fp32", attention="reference",
                                                          checkpoint_copies="batched", frozen_pooling="metal"))

        def update(model, optimizer, *, managed):
            optimizer.zero_grad(set_to_none=True)
            if managed:
                with runtime.work_unit():
                    with runtime.forward_context():
                        loss, _ = selection(solutions, prefixes)
                    loss.backward()
                    raw = {name: tensor_record(parameter.grad) for name, parameter in model.named_parameters()}
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.05, error_if_nonfinite=True)
                    clipped = {name: tensor_record(parameter.grad) for name, parameter in model.named_parameters()}
                    optimizer.step()
                saved = runtime.cpu_tree({"model": model.state_dict(), "optimizer": optimizer.state_dict()})
            else:
                loss, _ = reference_loss(model, None, solutions, prefixes)
                loss.backward()
                raw = {name: tensor_record(parameter.grad) for name, parameter in model.named_parameters()}
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.05, error_if_nonfinite=True)
                clipped = {name: tensor_record(parameter.grad) for name, parameter in model.named_parameters()}
                optimizer.step()
                saved = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
            return {"loss": tensor_record(loss), "norm": tensor_record(norm), "raw": raw, "clipped": clipped,
                    "saved": state_record(saved), "rng": tensor_record(torch.get_rng_state())}

        for completed in (1, 2):
            with self.subTest(completed=completed):
                rng = torch.get_rng_state().clone()
                expected = update(reference, optimizers[0], managed=False)
                torch.set_rng_state(rng)
                actual = update(candidate, optimizers[1], managed=True)
                self.assertEqual(actual, expected)
                self.assertEqual(candidate.zero_term.grad.tolist(), [0.0, 0.0])
                self.assertEqual(optimizers[1].state[candidate.zero_term]["step"].item(), completed)
                self.assertNotEqual(expected["raw"]["output.weight"], expected["clipped"]["output.weight"])
        self.assertEqual(selection.implementation, "reference")


class OriginalNCMCapabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        architecture = OmegaConf.load(
            Path(__file__).resolve().parents[1] / "vendor/ncm/config/eval/sudoku-hard.yaml")
        rng = torch.get_rng_state().clone()
        try:
            torch.manual_seed(452)
            with redirect_stdout(io.StringIO()):
                cls.tokenizer = make_tokenizer(architecture, "cpu")
                cls.model = make_generator(architecture, cls.tokenizer, "cpu")
            cls.tokenizer.eval().requires_grad_(False)
        finally:
            torch.set_rng_state(rng)

    @classmethod
    def tearDownClass(cls):
        del cls.model, cls.tokenizer
        torch.set_num_threads(cls.previous_threads)

    @contextmanager
    def mps_parameter_metadata_only(self):
        def parameter_metadata(module):
            return [SimpleNamespace(device=torch.device("mps"), dtype=parameter.dtype,
                                    requires_grad=parameter.requires_grad) for parameter in module.parameters()]

        tokenizer_parameters = parameter_metadata(self.tokenizer)
        model_parameters = parameter_metadata(self.model)
        tokenizer_buffers = [SimpleNamespace(device=torch.device("mps"), dtype=buffer.dtype,
                                              is_floating_point=buffer.is_floating_point)
                             for buffer in self.tokenizer.buffers()]
        with patch.object(self.tokenizer, "parameters", side_effect=lambda: iter(tokenizer_parameters)), \
                patch.object(self.model, "parameters", side_effect=lambda: iter(model_parameters)), \
                patch.object(self.tokenizer, "buffers", side_effect=lambda: iter(tokenizer_buffers)), \
                patch.object(torch.mps, "synchronize", side_effect=AssertionError("MPS execution in metadata test")), \
                patch.object(torch.mps, "driver_allocated_memory", side_effect=AssertionError("MPS allocation query")):
            yield

    def select(self, reference, **options):
        options.setdefault("objective", "mean-ce-all-codes")
        return puzzle_ncm.select_ncm_loss(self.model, self.tokenizer, reference,
                                          solution_length=89, prefix_length=91, **options)

    def test_supported_architecture_keeps_task_objective_without_explicit_opt_in(self):
        source = torch.tensor(2.0, requires_grad=True)

        def weighted_reference(*arguments):
            return 3.0 * source.square(), {"objective": "weighted"}

        with self.mps_parameter_metadata_only(), \
                patch("ncm_fast_targets.fast_generator_loss", side_effect=AssertionError("Objective substitution")):
            selection = puzzle_ncm.select_ncm_loss(self.model, self.tokenizer, weighted_reference,
                                                   solution_length=89, prefix_length=91)
            loss, metrics = selection("solutions", "prefixes")
            loss.backward()
        self.assertEqual(selection.implementation, "reference")
        self.assertIn("explicit objective", selection.reason)
        self.assertEqual((loss.item(), source.grad.item(), metrics), (12.0, 12.0, {"objective": "weighted"}))

    def test_real_original_cpu_models_fall_back_with_no_forward_or_rng_use(self):
        calls = []

        def reference(*arguments):
            calls.append(arguments)
            return "CPU reference result"

        rng = tensor_record(torch.get_rng_state())
        with patch.object(self.tokenizer, "encode", side_effect=AssertionError("Speculative encoding")), \
                patch.object(self.model, "forward", side_effect=AssertionError("Speculative generator")):
            selection = self.select(reference, pooling="metal")
        self.assertEqual(selection.implementation, "reference")
        self.assertIn("FP32 MPS", selection.reason)
        self.assertEqual(selection("solutions", "prefixes"), "CPU reference result")
        self.assertEqual(calls, [(self.model, self.tokenizer, "solutions", "prefixes")])
        self.assertEqual(tensor_record(torch.get_rng_state()), rng)

    def test_supported_static_metadata_binds_fast_callable_and_pooling(self):
        calls = []

        def fast(model, tokenizer, solutions, prefixes, *, pooling_kernel):
            calls.append((model, tokenizer, solutions, prefixes, pooling_kernel))
            return "selected fast result"

        rng = tensor_record(torch.get_rng_state())
        with self.mps_parameter_metadata_only(), \
                patch("ncm_fast_targets.fast_generator_loss", side_effect=fast), \
                patch.object(self.tokenizer, "encode", side_effect=AssertionError("Speculative encoding")), \
                patch.object(self.model, "forward", side_effect=AssertionError("Speculative generator")):
            selection = self.select(lambda *args: self.fail("Supported selection called reference."), pooling="metal")
            self.assertEqual(selection.implementation, "frozen-targets")
            self.assertEqual(selection.pooling, "metal")
            self.assertEqual(selection("solutions", "prefixes"), "selected fast result")
        self.assertEqual(calls, [(self.model, self.tokenizer, "solutions", "prefixes", "metal")])
        self.assertEqual(tensor_record(torch.get_rng_state()), rng)

    def test_selected_fast_failure_propagates_same_exception_without_reference_retry(self):
        original = RuntimeError("Fast path failed after state consumption.")
        reference_calls = []
        before = torch.get_rng_state().clone()

        def fast(*args, **kwargs):
            torch.rand(3)
            raise original

        def reference(*args):
            reference_calls.append(args)
            return "incorrect retry"

        try:
            with self.mps_parameter_metadata_only(), \
                    patch("ncm_fast_targets.fast_generator_loss", side_effect=fast):
                selection = self.select(reference)
                self.assertEqual(selection.implementation, "frozen-targets")
                with self.assertRaises(RuntimeError) as raised:
                    selection("solutions", "prefixes")
            self.assertIs(raised.exception, original)
            self.assertEqual(reference_calls, [])
            self.assertNotEqual(tensor_record(torch.get_rng_state()), tensor_record(before))
        finally:
            torch.set_rng_state(before)

    def test_fast_numerical_context_wraps_only_generator_forward_and_returns_fp32_logits(self):
        active = [False]
        events = []
        source = torch.tensor([1.5, -2.0], requires_grad=True)

        @contextmanager
        def forward_context():
            active[0] = True
            events.append("enter")
            try:
                yield
            finally:
                active[0] = False
                events.append("exit")

        def query(batch_size):
            events.append(("query", batch_size, active[0]))
            return "query value"

        def forward(inputs):
            events.append(("generator", inputs, active[0]))
            return (source * 2).to(torch.bfloat16)

        def fast(model, tokenizer, solutions, prefixes, *, pooling_kernel):
            events.append(("frozen targets", active[0]))
            self.assertEqual(model.build_l0_query_input(2), "query value")
            logits = model("generator inputs")
            events.append(("loss", logits.dtype, active[0]))
            return logits.sum(), {}

        with self.mps_parameter_metadata_only(), \
                patch("ncm_fast_targets.fast_generator_loss", side_effect=fast), \
                patch.object(self.model, "build_l0_query_input", side_effect=query), \
                patch.object(self.model, "forward", side_effect=forward):
            selection = self.select(lambda *args: self.fail("Incorrect reference retry."),
                                    forward_context=forward_context)
            loss, _ = selection("solutions", "prefixes")
            loss.backward()
        self.assertEqual(events, [("frozen targets", False), ("query", 2, False), "enter",
                                  ("generator", "generator inputs", True), "exit",
                                  ("loss", torch.float32, False)])
        self.assertEqual((loss.dtype, loss.item()), (torch.float32, -1.0))
        self.assertEqual(source.grad.tolist(), [2.0, 2.0])

    def test_unfrozen_stochastic_query_and_hierarchy_mismatches_use_reference(self):
        modifications = ((self.tokenizer, "training", True),
                         (self.tokenizer.vq, "_stoch_encode", True),
                         (self.tokenizer.vq, "levels", [1, 89]),
                         (self.model, "query_level0", False),
                         (self.model, "prefix_len", 90),
                         (self.model.config, "dropout", 0.2))
        for owner, name, value in modifications:
            with self.subTest(attribute=name), self.mps_parameter_metadata_only(), patch.object(owner, name, value, create=True):
                rng = tensor_record(torch.get_rng_state())
                selection = self.select(lambda *args: "fallback used")
                self.assertEqual(selection.implementation, "reference")
                self.assertEqual(selection("solutions", "prefixes"), "fallback used")
                self.assertEqual(tensor_record(torch.get_rng_state()), rng)
        parameter = self.tokenizer.encoder.wte.weight
        parameter.requires_grad_(True)
        try:
            with self.mps_parameter_metadata_only():
                selection = self.select(lambda *args: "unfrozen fallback")
            self.assertEqual(selection("solutions", "prefixes"), "unfrozen fallback")
            self.assertEqual(selection.implementation, "reference")
        finally:
            parameter.requires_grad_(False)


if __name__ == "__main__":
    unittest.main()
