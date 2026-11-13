"""Int64 offset-width selection and equivalence tests for the fused KDA kernels."""

from __future__ import annotations

import pytest
import torch

from attn_gym._backends.cute.utils import requires_int64_abi
from attn_gym._backends.triton.utils import requires_int64_offsets

FUSED_CAPABLE = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0)

# These tests cover the fused kernels, which SM100-family GPUs run only when asked for.
pytestmark = pytest.mark.usefixtures("fused_chunk_default")


def test_requires_int64_abi_covers_oversized_singleton_strides():
    """Reject the i32 signature for every inventoried ABI boundary case.

    ``requires_int64_offsets`` bounds reachable offsets only, so a size-1 batch
    mode with an oversized declared stride must be caught by the ABI predicate.
    """
    cases = {
        # [1, T, H, D] packed-projection slice at the first ABI failure.
        "packed_q": ((1, 174763, 32, 128), (174763 * 12288, 12288, 128, 1)),
        # Compact [1, T, H, D] at the compact-pitch boundary.
        "compact_q": ((1, 524288, 32, 128), (524288 * 4096, 4096, 128, 1)),
        # Compact chunk state [1, NT, H, K, V] at 4096 chunks.
        "state": ((1, 4096, 32, 128, 128), (4096 * 524288, 524288, 16384, 128, 1)),
    }
    for name, (shape, stride) in cases.items():
        tensor = torch.empty(shape, device="meta").as_strided(shape, stride)
        assert requires_int64_abi(tensor), name

    small = torch.empty_strided((1, 128, 32, 128), (128 * 4096, 4096, 128, 1))
    assert not requires_int64_abi(small, None)


@pytest.mark.skipif(not FUSED_CAPABLE, reason="the fused KDA kernels require capability 8.0")
def test_bound_gate_oversized_singleton_stride_matches_compact():
    """An oversized stride on a size-1 batch mode must not change the gate results."""
    from attn_gym.linear.kda import bound_gate

    def run(oversized_batch_stride: bool):
        torch.manual_seed(23)
        raw_gate = torch.randn(1, 65, 2, 128, device="cuda", dtype=torch.bfloat16)
        if oversized_batch_stride:
            raw_gate = raw_gate.as_strided(raw_gate.shape, (2**31, *raw_gate.stride()[1:]))
        raw_gate = raw_gate.requires_grad_()
        A_log = torch.randn(2, device="cuda", requires_grad=True)
        dt_bias = torch.randn(2, 128, device="cuda", requires_grad=True)
        output = bound_gate(raw_gate, A_log, dt_bias, impl="fused")
        gradients = torch.autograd.grad(output.square().sum(), (raw_gate, A_log, dt_bias))
        return (output, *gradients)

    compact = run(False)
    oversized = run(True)
    for expected, actual in zip(compact, oversized, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not FUSED_CAPABLE, reason="the fused KDA kernels require capability 8.0")
def test_chunk_kda_oversized_singleton_stride_matches_compact():
    """An oversized stride on a size-1 batch mode must not change chunk results."""
    from attn_gym.linear import chunk_kda

    def run(oversized_batch_stride: bool):
        torch.manual_seed(11)
        tokens, heads = 192, 2
        qkv = torch.randn(1, tokens, 3, heads, 128, device="cuda", dtype=torch.bfloat16) / 8
        q, k, v = (qkv[:, :, index] for index in range(3))
        if oversized_batch_stride:
            q, k, v = (
                tensor.as_strided(tensor.shape, (2**31, *tensor.stride()[1:]))
                for tensor in (q, k, v)
            )
        q, k, v = (tensor.requires_grad_() for tensor in (q, k, v))
        gate = (-torch.rand(1, tokens, heads, 128, device="cuda")).requires_grad_()
        beta = torch.rand(1, tokens, heads, device="cuda").requires_grad_()
        offsets = torch.tensor([0, 65, tokens], device="cuda", dtype=torch.int32)
        output, _ = chunk_kda(q, k, v, gate, beta, cu_seqlens=offsets)
        grads = torch.autograd.grad(output.float().square().sum(), (q, k, v, gate, beta))
        return (output, *grads)

    compact = run(False)
    oversized = run(True)
    for name, expected, actual in zip(
        ("output", "dq", "dk", "dv", "dgate", "dbeta"), compact, oversized, strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=lambda m, n=name: n)


@pytest.mark.skipif(
    not FUSED_CAPABLE or torch.cuda.get_device_properties(0).total_memory < 16 * 1024**3,
    reason="the executed-overflow regression peaks near 5.5 GiB",
)
def test_chunk_kda_executes_offsets_past_int32():
    """Actually execute element offsets beyond INT32_MAX, not just select the wide kernels.

    A fully active run whose Q/K/V share a padded projection buffer reaches token offsets near
    2.4e9; the same tokens through compact clones stay below INT32_MAX and take the i32
    specializations. Both must agree. A projection pitch of 72 makes offsets exceed INT32_MAX
    with only 8192 tokens. The unused padding stays unwritten; the buffer is 4.5 GiB and peak
    memory about 5.5 GiB.
    """
    from attn_gym.linear import chunk_kda

    heads, dim, tokens, pitch = 32, 128, 8192, 72
    torch.manual_seed(17)
    qkv = torch.empty(1, tokens, pitch, heads, dim, device="cuda", dtype=torch.bfloat16)
    packed = tuple(qkv[:, :, i] for i in range(3))
    for view in packed:
        view.copy_(torch.randn(view.shape, device="cuda", dtype=torch.bfloat16) / 8)
    compact = tuple(view.contiguous() for view in packed)
    assert (tokens - 1) * packed[0].stride(1) > torch.iinfo(torch.int32).max
    assert requires_int64_offsets(*packed) and not requires_int64_offsets(*compact)

    gate = -torch.rand(1, tokens, heads, dim, device="cuda")
    beta = torch.rand(1, tokens, heads, device="cuda")
    offsets = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)

    with torch.no_grad():
        wide, _ = chunk_kda(*packed, gate, beta, cu_seqlens=offsets)
        narrow, _ = chunk_kda(*compact, gate, beta, cu_seqlens=offsets)
    torch.testing.assert_close(wide, narrow, rtol=0, atol=0)
