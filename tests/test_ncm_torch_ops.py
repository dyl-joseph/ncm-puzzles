import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from ncm_torch_ops import NCMMPSSafeOps


class NCMMPSSafeOpsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(4)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def compare_area(self, device, length, keyword_input=False):
        generator = torch.Generator().manual_seed(42)
        reference_input = torch.randn(2, 3, 89, generator=generator, requires_grad=True)
        candidate_input = reference_input.detach().to(device).requires_grad_(True)
        cotangent = torch.randn(2, 3, length, generator=generator)
        expected = F.interpolate(reference_input, size=length, mode="area")
        (expected * cotangent).sum().backward()
        with NCMMPSSafeOps(devices=(device,)):
            if keyword_input:
                actual = F.interpolate(input=candidate_input, size=(length,), mode="area")
            else:
                actual = F.interpolate(candidate_input, (length,), None, "area")
        (actual * cotangent.to(device)).sum().backward()
        torch.testing.assert_close(actual.detach().cpu(), expected.detach(), atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(candidate_input.grad.cpu(), reference_input.grad,
                                   atol=2e-6, rtol=2e-6)

    def compare_convolution(self, device, transpose, keyword_input=False, bias=True):
        torch.manual_seed(42)
        constructor = torch.nn.ConvTranspose1d if transpose else torch.nn.Conv1d
        reference = constructor(256, 256, 89, stride=89, bias=bias)
        candidate = constructor(256, 256, 89, stride=89, bias=bias).to(device)
        candidate.load_state_dict(reference.state_dict())
        parameter_ids = tuple(id(parameter) for parameter in candidate.parameters())
        reference_input = torch.randn(2, 256, 1 if transpose else 89, requires_grad=True)
        candidate_input = reference_input.detach().to(device).requires_grad_(True)
        expected = reference(reference_input)
        cotangent = torch.randn_like(expected)
        (expected * cotangent).sum().backward()
        with NCMMPSSafeOps(devices=(device,)):
            if keyword_input:
                function = F.conv_transpose1d if transpose else F.conv1d
                actual = function(input=candidate_input, weight=candidate.weight,
                                  bias=candidate.bias, stride=(89,), padding=(0,),
                                  dilation=(1,), groups=1,
                                  **({"output_padding": (0,)} if transpose else {}))
            else:
                actual = candidate(candidate_input)
        (actual * cotangent.to(device)).sum().backward()
        comparisons = ((actual.detach().cpu(), expected.detach()),
                       (candidate_input.grad.cpu(), reference_input.grad),
                       (candidate.weight.grad.cpu(), reference.weight.grad))
        if bias:
            comparisons += ((candidate.bias.grad.cpu(), reference.bias.grad),)
        for actual_value, expected_value in comparisons:
            torch.testing.assert_close(actual_value, expected_value, atol=2e-5, rtol=2e-5)
        self.assertEqual(tuple(id(parameter) for parameter in candidate.parameters()), parameter_ids)

    def test_area_outputs_and_gradients_for_all_ncm_levels(self):
        for length in (1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 89, 97):
            with self.subTest(length=length):
                self.compare_area("cpu", length, keyword_input=length % 2 == 0)

    def test_area_has_literal_overlapping_bin_means(self):
        x = torch.arange(89, dtype=torch.float32).reshape(1, 1, 89).requires_grad_(True)
        with NCMMPSSafeOps(devices=("cpu",)):
            actual = F.interpolate(x, size=2, mode="area")
        self.assertEqual(actual.tolist(), [[[22.0, 66.0]]])
        actual.sum().backward()
        expected_gradient = torch.full_like(x, 1 / 45)
        expected_gradient[..., 44] = 2 / 45
        torch.testing.assert_close(x.grad, expected_gradient, atol=0, rtol=0)

    def test_convolution_outputs_and_parameter_gradients(self):
        for transpose in (False, True):
            for keyword_input in (False, True):
                with self.subTest(transpose=transpose, keyword_input=keyword_input):
                    self.compare_convolution("cpu", transpose, keyword_input=keyword_input)

    def test_convolutions_without_bias(self):
        for transpose in (False, True):
            with self.subTest(transpose=transpose):
                self.compare_convolution("cpu", transpose, bias=False)

    def test_other_convolution_configurations_fall_through(self):
        torch.manual_seed(42)
        cases = (
            (F.conv1d, (2, 256, 89), (256, 256, 89), {"stride": 1}),
            (F.conv1d, (2, 256, 89), (256, 256, 89), {"stride": 89, "padding": 1}),
            (F.conv1d, (2, 256, 89), (256, 128, 89), {"stride": 89, "groups": 2}),
            (F.conv1d, (2, 256, 90), (256, 256, 89), {"stride": 89}),
            (F.conv1d, (2, 2, 89), (3, 2, 89), {"stride": 89}),
            (F.conv_transpose1d, (2, 256, 1), (256, 256, 89), {"stride": 1}),
            (F.conv_transpose1d, (2, 256, 1), (256, 256, 89), {"stride": 89, "padding": 1}),
            (F.conv_transpose1d, (2, 256, 1), (256, 128, 89), {"stride": 89, "groups": 2}),
            (F.conv_transpose1d, (2, 256, 1), (256, 256, 89), {"stride": 89, "dilation": 2}),
            (F.conv_transpose1d, (2, 256, 1), (256, 256, 89), {"stride": 89, "output_padding": 1}),
            (F.conv_transpose1d, (2, 256, 2), (256, 256, 89), {"stride": 89}),
        )
        for function, input_shape, weight_shape, kwargs in cases:
            with self.subTest(function=function.__name__, input_shape=input_shape, kwargs=kwargs):
                x = torch.randn(input_shape)
                weight = torch.randn(weight_shape) * 0.01
                expected = function(x, weight, **kwargs)
                with NCMMPSSafeOps(devices=("cpu",)):
                    with patch.object(F, "linear", side_effect=AssertionError("unexpected substitution")):
                        actual = function(x, weight, **kwargs)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_positional_dilation_and_groups_are_not_confused(self):
        x = torch.randn(1, 256, 89)
        weight = torch.randn(256, 256, 89)
        with NCMMPSSafeOps(devices=("cpu",)):
            with self.assertRaisesRegex(RuntimeError, "Kernel size|kernel size"):
                F.conv1d(x, weight, None, 89, 0, 2, 1)
        x = x[..., :1]
        expected = F.conv_transpose1d(x, weight, None, 89, 0, 0, 1, 2)
        with NCMMPSSafeOps(devices=("cpu",)):
            actual = F.conv_transpose1d(x, weight, None, 89, 0, 0, 1, 2)
        self.assertEqual(actual.shape[-1], 177)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_invalid_convolution_arguments_still_raise(self):
        x = torch.randn(1, 256, 89)
        weight = torch.randn(256, 256, 89)
        for kwargs in ({"groups": 2}, {"groups": 1.0}, {"dilation": True}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((RuntimeError, TypeError)):
                    with NCMMPSSafeOps(devices=("cpu",)):
                        F.conv1d(x, weight, stride=89, **kwargs)

    def test_default_mode_leaves_cpu_operators_untouched(self):
        x = torch.randn(1, 256, 89)
        weight = torch.randn(256, 256, 89)
        expected = F.conv1d(x, weight, stride=89)
        with NCMMPSSafeOps():
            with patch.object(F, "linear", side_effect=AssertionError("unexpected substitution")):
                actual = F.conv1d(x, weight, stride=89)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_mode_deactivates_after_normal_and_exceptional_exit(self):
        x = torch.randn(1, 256, 89)
        weight = torch.randn(256, 256, 89)
        expected = F.conv1d(x, weight, stride=89)
        with NCMMPSSafeOps(devices=("cpu",)):
            actual = F.conv1d(x, weight, stride=89)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=2e-5)
        with patch.object(F, "linear", side_effect=RuntimeError("substitution reached")):
            with self.assertRaisesRegex(RuntimeError, "substitution reached"):
                with NCMMPSSafeOps(devices=("cpu",)):
                    F.conv1d(x, weight, stride=89)
            actual = F.conv1d(x, weight, stride=89)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_unrelated_interpolation_modes_and_dimensions_fall_through(self):
        cases = ((torch.randn(2, 3, 89), {"size": 13, "mode": "linear"}),
                 (torch.randn(2, 3, 9, 9), {"size": (4, 4), "mode": "area"}),
                 (torch.randn(2, 3, 89), {"scale_factor": 0.5, "mode": "area"}))
        for x, kwargs in cases:
            with self.subTest(shape=x.shape, kwargs=kwargs):
                expected = F.interpolate(x, **kwargs)
                with NCMMPSSafeOps(devices=("cpu",)):
                    actual = F.interpolate(x, **kwargs)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_invalid_interpolation_arguments_still_raise(self):
        x = torch.randn(2, 3, 89)
        for kwargs in ({"size": 13, "scale_factor": 0.5},
                       {"size": 13, "align_corners": False},
                       {"size": 13, "antialias": True},
                       {"size": 13, "recompute_scale_factor": True}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    with NCMMPSSafeOps(devices=("cpu",)):
                        F.interpolate(x, mode="area", **kwargs)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_mps_area_outputs_and_gradients(self):
        for length in (1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 89, 97):
            with self.subTest(length=length):
                self.compare_area("mps", length)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_mps_convolution_outputs_and_gradients(self):
        for transpose in (False, True):
            with self.subTest(transpose=transpose):
                self.compare_convolution("mps", transpose, keyword_input=True)


if __name__ == "__main__":
    unittest.main()
