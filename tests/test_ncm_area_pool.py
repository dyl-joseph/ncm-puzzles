import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from ncm_area_pool import POOLING_KERNELS, area89, supported_area89
from ncm_torch_ops import NCMMPSSafeOps
from ncm_fast_targets import NCMFastMPSSafeOps


SIZES = (2, 4, 6, 9, 13, 18, 26, 37, 50, 67)


class AreaPoolGuardTests(unittest.TestCase):
    def tensor_metadata(self, **overrides):
        tensor = Mock(spec=torch.Tensor)
        tensor.device = torch.device("mps")
        tensor.dtype = torch.float32
        tensor.ndim = 3
        tensor.shape = (16, 256, 89)
        tensor.stride.return_value = (22784, 1, 256)
        for name, value in overrides.items():
            if name == "stride":
                tensor.stride.return_value = value
            else:
                setattr(tensor, name, value)
        return tensor

    def test_supported_choices_and_frozen_layout(self):
        self.assertEqual(POOLING_KERNELS, ("reference", "metal"))
        with torch.no_grad():
            for batch in (1, 8, 16):
                tensor = self.tensor_metadata(shape=(batch, 256, 89))
                for strides in ((22784, 1, 256), (22784, 89, 1)):
                    tensor.stride.return_value = strides
                    for size in SIZES:
                        self.assertTrue(supported_area89(tensor, size))

    def test_autograd_context_rejects_even_frozen_inputs(self):
        tensor = self.tensor_metadata()
        with torch.enable_grad():
            self.assertFalse(supported_area89(tensor, 13))
        with torch.no_grad():
            self.assertTrue(supported_area89(tensor, 13))

    def test_unverified_shapes_layouts_and_dtypes_use_fallback(self):
        cases = (
            {"device": torch.device("cpu")},
            {"dtype": torch.bfloat16},
            {"dtype": torch.float64},
            {"ndim": 2},
            {"shape": (0, 256, 89)},
            {"shape": (17, 256, 89)},
            {"shape": (1, 255, 89)},
            {"shape": (1, 256, 88)},
            {"stride": (22784, 90, 1)},
            {"stride": (22784, 0, 256)},
            {"stride": (2**31, 1, 256)},
            {"stride": (-22784, 1, 256)},
        )
        with torch.no_grad():
            for changes in cases:
                with self.subTest(changes=changes):
                    self.assertFalse(supported_area89(self.tensor_metadata(**changes), 13))
            for size in (True, 1, 89, 90, 0, -1, 13.0, (13,), None):
                with self.subTest(size=size):
                    self.assertFalse(supported_area89(self.tensor_metadata(), size))
            self.assertFalse(supported_area89(None, 13))

    def test_rejected_cpu_call_does_not_compile_or_allocate_output(self):
        x = torch.zeros((1, 89, 256)).transpose(1, 2)
        with torch.no_grad(), patch("ncm_area_pool._library") as library, \
                patch("ncm_area_pool.torch.empty") as allocate:
            self.assertFalse(supported_area89(x, 13))
            with self.assertRaisesRegex(ValueError, "no-grad FP32 MPS"):
                area89(x, 13)
        library.assert_not_called()
        allocate.assert_not_called()


@unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
class AreaPoolMPSTests(unittest.TestCase):
    def strided_input(self, values, batch=1):
        return values.reshape(1, 89, 1).expand(batch, 89, 256).clone().to("mps").transpose(1, 2)

    def test_signed_ramp_has_literal_bin_means(self):
        x = self.strided_input(torch.arange(-44, 45, dtype=torch.float32), batch=2)
        with torch.no_grad():
            for size in SIZES:
                with self.subTest(size=size):
                    actual = area89(x, size)
                    expected = torch.tensor([
                        (index * 89 // size + ((index + 1) * 89 + size - 1) // size - 1) / 2 - 44
                        for index in range(size)
                    ]).expand(2, 256, size)
                    self.assertTrue(torch.equal(actual.cpu(), expected))
                    self.assertEqual(actual.dtype, torch.float32)
                    self.assertTrue(actual.is_contiguous())
                    self.assertFalse(actual.requires_grad)

    def test_zero_negative_and_endpoint_values_match_original_reductions_exactly(self):
        patterns = (
            torch.zeros(89),
            torch.full((89,), -3.25),
            torch.tensor([1024.0] + [0.0] * 87 + [-1024.0]),
            torch.randn(89, generator=torch.Generator().manual_seed(147)),
        )
        with torch.no_grad():
            for pattern_index, values in enumerate(patterns):
                x = self.strided_input(values, batch=16)
                for values in (x, x.contiguous()):
                    for size in SIZES:
                        with self.subTest(pattern=pattern_index, size=size, strides=values.stride()):
                            with NCMMPSSafeOps():
                                expected = F.interpolate(values, size=(size,), mode="area")
                            actual = area89(values, size)
                            self.assertTrue(torch.equal(actual.reshape(-1).view(torch.uint8),
                                                        expected.reshape(-1).view(torch.uint8)))

    def test_grad_enabled_inputs_remain_on_the_original_autograd_path(self):
        x = self.strided_input(torch.arange(89, dtype=torch.float32)).requires_grad_(True)
        with torch.enable_grad():
            self.assertFalse(supported_area89(x, 13))
            with self.assertRaisesRegex(ValueError, "no-grad FP32 MPS"):
                area89(x, 13)
            reference = x.detach().clone().requires_grad_(True)
            with NCMMPSSafeOps():
                expected = F.interpolate(reference, size=(13,), mode="area")
                expected.sum().backward()
            with NCMFastMPSSafeOps(pooling_kernel="metal"):
                actual = F.interpolate(x, size=(13,), mode="area")
                actual.sum().backward()
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(x.grad.reshape(-1).view(torch.uint8),
                                    reference.grad.reshape(-1).view(torch.uint8)))
        self.assertGreater(x.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
