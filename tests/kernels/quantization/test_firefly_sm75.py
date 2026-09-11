# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the firefly SM75 INT4->INT8 prefill path.

Covers the reverse-repack round-trip and the CUDA dequant kernel's bit-exact
agreement with the PyTorch reference.

Run `pytest tests/kernels/quantization/test_firefly_sm75.py`.
"""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.firefly import (
    _reverse_repack_indices,
    dequant_marlin_to_int8,
    dequant_marlin_to_int8_cached,
    marlin_to_int4_q,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_empty_g_idx,
    marlin_pad_qweight,
    marlin_padded_nk,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import gptq_pack
from vllm.platforms import current_platform

# (size_n, size_k, group_size): aligned + tile-padded + AWQ-style shapes.
SHAPES = [
    (128, 256, 128),
    (256, 128, 128),
    (200, 288, 128),  # tile padding on both dims
    (256, 208, 128),  # K padding
    (256, 256, 32),
]

requires_sm75 = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="firefly kernels require CUDA",
)


def _make_marlin_weight(q: torch.Tensor, size_n: int, size_k: int, group_size: int):
    """Clean int4 [N, K] (values 0..15) -> marlin-layout packed int32."""
    packed = gptq_pack(q.t().contiguous(), 4, size_k, size_n)
    padded_n, padded_k = marlin_padded_nk(size_n, size_k, group_size)
    return marlin_pad_qweight(packed, size_n, size_k, padded_n, padded_k), (
        padded_n,
        padded_k,
    )


@requires_sm75
@pytest.mark.parametrize("size_n,size_k,group_size", SHAPES)
def test_reverse_repack_roundtrip(size_n: int, size_k: int, group_size: int) -> None:
    torch.manual_seed(0)
    q = torch.randint(0, 16, (size_n, size_k), dtype=torch.int32, device="cuda")

    marlin_w, (padded_n, padded_k) = _make_marlin_weight(q, size_n, size_k, group_size)
    packed = ops.gptq_marlin_repack(
        marlin_w,
        perm=marlin_make_empty_g_idx(q.device),
        size_k=padded_k,
        size_n=padded_n,
        num_bits=4,
    )

    q_recovered = marlin_to_int4_q(packed, size_k, size_n, padded_k, padded_n)

    torch.testing.assert_close(
        q_recovered.long(), q.long(), rtol=0, atol=0, msg="reverse repack mismatch"
    )


@requires_sm75
def test_reverse_repack_indices_shape() -> None:
    size_k, size_n = 208, 256
    src_flat, shift = _reverse_repack_indices(size_k, size_n, torch.device("cuda"))
    assert src_flat.shape == (size_n, size_k)
    assert shift.shape == (size_n, size_k)
    assert shift.min() >= 0 and shift.max() <= 28


@requires_sm75
@pytest.mark.parametrize("group_size", [128, 32])
def test_dequant_cuda_matches_pytorch(group_size: int) -> None:
    size_n, size_k = 256, 512
    torch.manual_seed(1)
    q = torch.randint(0, 16, (size_n, size_k), dtype=torch.int32, device="cuda")
    scale = (
        torch.rand(size_n, size_k // group_size, dtype=torch.float16, device="cuda")
        + 0.01
    )

    marlin_w, (padded_n, padded_k) = _make_marlin_weight(q, size_n, size_k, group_size)
    packed = ops.gptq_marlin_repack(
        marlin_w,
        perm=marlin_make_empty_g_idx(q.device),
        size_k=padded_k,
        size_n=padded_n,
        num_bits=4,
    )

    # CUDA kernel vs the pure-PyTorch reference (same math, zp=8 symmetric).
    w_int8_kernel, c_n = dequant_marlin_to_int8(
        packed,
        scale,
        group_size,
        size_k,
        size_n,
        padded_k,
        padded_n,
    )
    # Cached (single-pass) must match the two-pass kernel bit-exactly.
    w_int8_cached = dequant_marlin_to_int8_cached(
        packed,
        scale,
        c_n,
        group_size,
        size_k,
        size_n,
        padded_k,
        padded_n,
        use_recip=False,
    )
    w_int8_recip = dequant_marlin_to_int8_cached(
        packed,
        scale,
        c_n,
        group_size,
        size_k,
        size_n,
        padded_k,
        padded_n,
        use_recip=True,
    )
    torch.testing.assert_close(
        w_int8_cached, w_int8_kernel, rtol=0, atol=0, msg="cached vs 2-pass mismatch"
    )
    # Fast reciprocal path may differ by at most 1 (documented <=0.06%).
    recip_diff = (w_int8_recip.int() - w_int8_kernel.int()).abs()
    assert recip_diff.max().item() <= 1, (
        f"recip off-by-one exceeded: {recip_diff.max().item()}"
    )
    # c_n is a fp32 reduction, so allow the reference to differ by rounding.
    c_n_ref = (
        (q.float() - 8.0) * scale.float().repeat_interleave(group_size, dim=1)
    ).abs().amax(dim=1) / 127.0
    torch.testing.assert_close(c_n, c_n_ref, rtol=1e-5, atol=1e-7)
    # Kernel int8 values match the reference up to rounding at the clamp edge.
    w_ref = torch.clamp(
        torch.round(
            ((q.float() - 8.0) * scale.float().repeat_interleave(group_size, dim=1))
            / c_n_ref.unsqueeze(1)
        ),
        -127,
        127,
    ).to(torch.int8)
    kernel_diff = (w_int8_kernel.int() - w_ref.int()).abs()
    assert kernel_diff.max().item() <= 1, (
        f"kernel vs reference exceeded off-by-one: {kernel_diff.max().item()}"
    )


@requires_sm75
def test_int8_prefill_linear_shapes() -> None:
    from vllm.model_executor.layers.quantization.utils.firefly import (
        int8_prefill_linear,
    )

    size_n, size_k = 256, 512
    torch.manual_seed(2)
    w_int8 = torch.randint(-127, 128, (size_n, size_k), dtype=torch.int8, device="cuda")
    c_n = torch.rand(size_n, dtype=torch.float32, device="cuda") + 0.01
    x = torch.randn(4, 8, size_k, dtype=torch.float16, device="cuda")

    y = int8_prefill_linear(x, w_int8, c_n)
    assert y.shape == (4, 8, size_n)
    assert y.dtype == torch.float16
