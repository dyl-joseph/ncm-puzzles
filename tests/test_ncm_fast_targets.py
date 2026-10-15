from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from ncm_fast_targets import (NCMFastMPSSafeOps, fast_generator_loss,
                              frozen_targets_and_context)
from ncm_torch_ops import NCMMPSSafeOps
import train_latin as training


class FrozenNCMTargetsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(4)
        cls.architecture = OmegaConf.load(
            Path(__file__).resolve().parents[1] / "vendor/ncm/config/eval/sudoku-hard.yaml")
        torch.manual_seed(42)
        with redirect_stdout(io.StringIO()):
            cls.tokenizer = training.make_tokenizer(cls.architecture, "cpu")
        cls.tokenizer.eval().requires_grad_(False)

    @classmethod
    def tearDownClass(cls):
        del cls.tokenizer
        torch.set_num_threads(cls.previous_threads)

    def inputs(self, batch, noncontiguous=False):
        generator = torch.Generator().manual_seed(173)
        if noncontiguous:
            result = torch.randint(0, 12, (89, batch), generator=generator).t()
            self.assertFalse(result.is_contiguous())
            return result
        return torch.randint(0, 12, (batch, 89), generator=generator)

    def test_codes_and_contexts_are_exact_for_original_models(self):
        for batch, noncontiguous in ((1, False), (2, False), (2, True)):
            with self.subTest(batch=batch, noncontiguous=noncontiguous), torch.no_grad():
                solutions = self.inputs(batch, noncontiguous)
                expected_codes = self.tokenizer.encode(solutions)
                expected_contexts = self.tokenizer.vq.idx_to_ncm_input(expected_codes)
                actual_codes, actual_contexts = frozen_targets_and_context(self.tokenizer, solutions)
                self.assertEqual(actual_codes.shape, (batch, 322))
                self.assertEqual(actual_contexts.shape, (batch, 321, 256))
                self.assertTrue(torch.equal(actual_codes, expected_codes))
                self.assertTrue(torch.equal(actual_contexts, expected_contexts))
                self.assertFalse(actual_contexts.requires_grad)

    def test_safe_ops_identity_preserves_codes_contexts_and_input_gradients(self):
        solutions = self.inputs(2, True)
        with torch.no_grad(), NCMMPSSafeOps(devices=("cpu",)):
            expected_codes = self.tokenizer.encode(solutions)
            expected_contexts = self.tokenizer.vq.idx_to_ncm_input(expected_codes)
        with NCMFastMPSSafeOps(devices=("cpu",)):
            actual_codes, actual_contexts = frozen_targets_and_context(self.tokenizer, solutions)
        self.assertTrue(torch.equal(actual_codes, expected_codes))
        self.assertTrue(torch.equal(actual_contexts, expected_contexts))
        x = torch.randn(2, 256, 89, requires_grad=True)
        with NCMFastMPSSafeOps(devices=("cpu",)):
            result = F.interpolate(input=x, size=(89,), mode="area")
        self.assertIs(result, x)
        result.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.ones_like(x)))

    def test_fusion_skips_duplicate_reconstruction_and_unused_last_correction(self):
        vq = self.tokenizer.vq
        with patch.object(vq, "_up_to_full", wraps=vq._up_to_full) as upsample, \
                patch.object(vq, "_apply_quant_resi", wraps=vq._apply_quant_resi) as correction:
            frozen_targets_and_context(self.tokenizer, self.inputs(1))
        self.assertEqual(upsample.call_count, 11)
        self.assertEqual(correction.call_count, 11)
        self.assertEqual([call.args[1] for call in correction.call_args_list], list(range(11)))

    def test_frozen_preprocessing_preserves_rng_and_all_tokenizer_state(self):
        parameters = {name: value.detach().clone() for name, value in self.tokenizer.named_parameters()}
        buffers = {name: value.detach().clone() for name, value in self.tokenizer.named_buffers()}
        rng = torch.get_rng_state().clone()
        frozen_targets_and_context(self.tokenizer, self.inputs(2))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for name, actual in self.tokenizer.named_parameters():
            self.assertTrue(torch.equal(actual, parameters[name]), name)
        for name, actual in self.tokenizer.named_buffers():
            self.assertTrue(torch.equal(actual, buffers[name]), name)

    def test_loss_gradients_dropout_rng_and_adamw_update_match(self):
        with redirect_stdout(io.StringIO()):
            reference = training.make_generator(self.architecture, self.tokenizer, "cpu")
        candidate = copy.deepcopy(reference)
        reference.train()
        candidate.train()
        solutions = self.inputs(2)
        prefixes = torch.randint(0, 12, (2, 91), generator=torch.Generator().manual_seed(19))
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=3e-4, betas=(0.9, 0.99))
        candidate_optimizer = torch.optim.AdamW(candidate.parameters(), lr=3e-4, betas=(0.9, 0.99))
        rng = torch.get_rng_state().clone()
        expected_loss, expected_metrics = training.generator_loss(
            reference, self.tokenizer, solutions, prefixes)
        expected_loss.backward()
        expected_rng = torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        actual_loss, actual_metrics = fast_generator_loss(candidate, self.tokenizer, solutions, prefixes)
        actual_loss.backward()
        self.assertTrue(torch.equal(actual_loss, expected_loss))
        self.assertTrue(torch.equal(actual_metrics["code_ce"], expected_metrics["code_ce"]))
        self.assertFalse(actual_metrics["code_ce"].requires_grad)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        reference_parameters = dict(reference.named_parameters())
        candidate_parameters = dict(candidate.named_parameters())
        self.assertEqual(reference_parameters.keys(), candidate_parameters.keys())
        for name, expected in reference_parameters.items():
            actual = candidate_parameters[name]
            self.assertEqual(actual.grad is None, expected.grad is None, name)
            if expected.grad is not None:
                self.assertTrue(torch.equal(actual.grad, expected.grad), name)
        self.assertIsNotNone(candidate.l0_queries.grad)
        self.assertGreater(candidate.l0_queries.grad.abs().sum().item(), 0)
        reference_optimizer.step()
        candidate_optimizer.step()
        for name, expected in reference_parameters.items():
            self.assertTrue(torch.equal(candidate_parameters[name], expected), name)
        self.assertEqual(reference_optimizer.state_dict()["param_groups"],
                         candidate_optimizer.state_dict()["param_groups"])
        expected_states = reference_optimizer.state_dict()["state"]
        actual_states = candidate_optimizer.state_dict()["state"]
        for index, state in expected_states.items():
            for name, value in state.items():
                self.assertTrue(torch.equal(actual_states[index][name], value), f"{index}.{name}")

    def test_training_unfrozen_stochastic_and_unsupported_inputs_are_rejected(self):
        solutions = self.inputs(1)
        self.tokenizer.train()
        try:
            with self.assertRaisesRegex(ValueError, "eval-mode"):
                frozen_targets_and_context(self.tokenizer, solutions)
        finally:
            self.tokenizer.eval()
        parameter = self.tokenizer.encoder.wte.weight
        parameter.requires_grad_(True)
        try:
            with self.assertRaisesRegex(ValueError, "frozen"):
                frozen_targets_and_context(self.tokenizer, solutions)
        finally:
            parameter.requires_grad_(False)
        self.tokenizer.vq.set_stochastic_encode(True)
        rng = torch.get_rng_state().clone()
        try:
            with self.assertRaisesRegex(ValueError, "deterministic"):
                frozen_targets_and_context(self.tokenizer, solutions)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        finally:
            self.tokenizer.vq.set_stochastic_encode(False)
        with patch.object(self.tokenizer.vq, "levels", [1, 89]):
            with self.assertRaisesRegex(ValueError, "original 322-code"):
                frozen_targets_and_context(self.tokenizer, solutions)
        for malformed in (solutions[:, :-1], solutions.float(), solutions[:0]):
            with self.subTest(shape=malformed.shape, dtype=malformed.dtype):
                with self.assertRaisesRegex(ValueError, "int64"):
                    frozen_targets_and_context(self.tokenizer, malformed)


if __name__ == "__main__":
    unittest.main()
