"""Exact frozen-tokenizer area pooling for the original MPS tensor layout."""

from functools import lru_cache

import torch


POOLING_KERNELS = ("reference", "metal")
_AREA_SIZES = frozenset((2, 4, 6, 9, 13, 18, 26, 37, 50, 67))
_VECTOR_SIZES = frozenset((13, 18, 26, 37, 50, 67))
_INPUT_STRIDES = frozenset(((89 * 256, 1, 256), (89 * 256, 89, 1)))
_INT32_LIMIT = 2**31

_SOURCE = r"""
#include <c10/metal/reduction_utils.h>
#include <metal_array>
#include <metal_stdlib>

using namespace metal;

kernel void area89_reduce(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& output_size [[buffer(2)]],
    constant int& channels [[buffer(3)]],
    constant int& stride_b [[buffer(4)]],
    constant int& stride_c [[buffer(5)]],
    constant int& stride_l [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tptg [[threads_per_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint simdgroup_size [[threads_per_simdgroup]]) {
  const uint index = tgid.y;
  const uint size = uint(output_size);
  const uint bin = index % size;
  const uint bc = index / size;
  const uint start = bin * 89u / size;
  const uint end = ((bin + 1u) * 89u + size - 1u) / size;
  const uint rsize = end - start;
  const uint input_base = (bc / uint(channels)) * uint(stride_b)
      + (bc % uint(channels)) * uint(stride_c) + start * uint(stride_l);

  metal::array<float, 8> acc;
  for (uint j = 0; j < 8; j++) {
    acc[j] = 0;
  }
  const uint stride = tptg.x * 8u;
  uint base = tid * 8u;
  for (; base + 8u <= rsize; base += stride) {
    for (uint j = 0; j < 8; j++) {
      acc[j] += input[input_base + (base + j) * uint(stride_l)];
    }
  }
  for (uint idx = base; idx < rsize; idx++) {
    acc[idx % 8u] += input[input_base + idx * uint(stride_l)];
  }

  float output_val = acc[0];
  for (uint j = 1; j < 8; j++) {
    output_val += acc[j];
  }

  // Match the native generic reduction's active lanes and summation order.
  uint threads_remaining = tptg.x;
  threadgroup float shared_outputs[2];
  while (threads_remaining > 1) {
    output_val = c10::metal::simd_sum(output_val);
    threads_remaining = (threads_remaining + simdgroup_size - 1u) / simdgroup_size;
    if (threads_remaining > 1) {
      if (simd_lane_id == 0) {
        shared_outputs[simdgroup_id] = output_val;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid < threads_remaining) {
        output_val = shared_outputs[tid];
      } else {
        return;
      }
    }
  }
  if (tid == 0) {
    output[index] = output_val / float(rsize);
  }
}

kernel void area89_vector(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& output_size [[buffer(2)]],
    constant int& channels [[buffer(3)]],
    constant int& stride_b [[buffer(4)]],
    constant int& stride_c [[buffer(5)]],
    constant int& stride_l [[buffer(6)]],
    constant int& output_count [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= uint(output_count)) {
    return;
  }
  const uint size = uint(output_size);
  const uint bin = index % size;
  const uint bc = index / size;
  const uint start = bin * 89u / size;
  const uint end = ((bin + 1u) * 89u + size - 1u) / size;
  const uint rsize = end - start;
  const uint input_base = (bc / uint(channels)) * uint(stride_b)
      + (bc % uint(channels)) * uint(stride_c) + start * uint(stride_l);

  metal::array<float, 8> acc;
  for (uint j = 0; j < 8; j++) {
    acc[j] = 0.0f;
  }
  for (uint j = 0; j < rsize; j++) {
    acc[j] += input[input_base + j * uint(stride_l)];
  }

  float output_val = acc[0];
  for (uint j = 1; j < 8; j++) {
    output_val += acc[j];
  }
  output[index] = output_val / float(rsize);
}
"""


@lru_cache(maxsize=1)
def _library():
    return torch.mps.compile_shader(_SOURCE)


def supported_area89(x, size):
    """Accept only the frozen tokenizer's no-grad FP32 [B, 256, 89] layout."""
    if not (
        isinstance(x, torch.Tensor)
        and x.device.type == "mps"
        and x.dtype == torch.float32
        and x.ndim == 3
        and 1 <= x.shape[0] <= 16
        and x.shape[1:] == (256, 89)
        and type(size) is int
        and size in _AREA_SIZES
        and not torch.is_grad_enabled()
        and tuple(x.stride()) in _INPUT_STRIDES
    ):
        return False
    return (
        all(0 <= stride < _INT32_LIMIT for stride in x.stride())
        and sum((dimension - 1) * stride
                for dimension, stride in zip(x.shape, x.stride())) < _INT32_LIMIT
        and x.shape[0] * x.shape[1] * size < _INT32_LIMIT
    )


def area89(x, size):
    """Return contiguous FP32 [B, 256, size] without an autograd operation."""
    if not supported_area89(x, size):
        raise ValueError(
            "area89 requires no-grad FP32 MPS [B, 256, 89], B in 1..16, "
            "a verified channel-first or transposed layout, and a supported pooling size."
        )
    output = torch.empty((x.shape[0], x.shape[1], size), dtype=x.dtype, device=x.device)
    if size in _VECTOR_SIZES:
        _library().area89_vector(
            x, output, size, x.shape[1], *x.stride(), output.numel(),
            threads=output.numel(),
            group_size=256,
            arg_casts="int32",
        )
    else:
        threads_per_group = 64 if size == 2 else 32
        _library().area89_reduce(
            x, output, size, x.shape[1], *x.stride(),
            threads=(threads_per_group, output.numel()),
            group_size=(threads_per_group, 1),
            arg_casts="int32",
        )
    return output
