import unittest
from unittest.mock import patch

import torch

from ncm_attention import select_generator_attention


class GeneratorAttentionTests(unittest.TestCase):
    def setUp(self):
        self.previous = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
        self.addCleanup(torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp, self.previous)

    def test_nested_kernel_scopes_restore_the_enclosing_choice(self):
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
        with select_generator_attention("bf16-math"):
            self.assertTrue(torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed())
            with select_generator_attention("reference"):
                self.assertFalse(torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed())
            self.assertTrue(torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed())
        self.assertFalse(torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed())

    def test_exception_restores_the_prior_flag(self):
        for kernel, previous in (("reference", True), ("bf16-math", False)):
            with self.subTest(kernel=kernel):
                torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(previous)
                with self.assertRaisesRegex(RuntimeError, "forward failed"):
                    with select_generator_attention(kernel):
                        self.assertEqual(torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
                                         not previous)
                        raise RuntimeError("forward failed")
                self.assertEqual(torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(), previous)

    def test_unsupported_kernel_fails_before_changing_the_flag(self):
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
        with patch.object(torch.backends.cuda, "allow_fp16_bf16_reduction_math_sdp") as select:
            with self.assertRaisesRegex(ValueError, "Attention kernel"):
                with select_generator_attention("flash"):
                    self.fail("An unsupported kernel must not enter the context.")
        select.assert_not_called()
        self.assertTrue(torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed())


if __name__ == "__main__":
    unittest.main()
