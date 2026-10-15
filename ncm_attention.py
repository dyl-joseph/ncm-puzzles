"""Scope the math attention precision policy to a generator forward pass."""

from contextlib import contextmanager


_REDUCED_MATH = {"reference": False, "bf16-math": True}
ATTENTION_KERNELS = tuple(_REDUCED_MATH)


@contextmanager
def select_generator_attention(kernel: str):
    if kernel not in _REDUCED_MATH:
        raise ValueError("Attention kernel must be reference or bf16-math.")
    import torch

    previous = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
    try:
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(_REDUCED_MATH[kernel])
        yield
    finally:
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(previous)
