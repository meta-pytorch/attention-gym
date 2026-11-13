"""Int64 offset-width selection and equivalence tests for the CuTe KDA kernels."""

from __future__ import annotations

import pytest
import torch

from attn_gym._backends.arch import cuda_runs_sm100_kernels
from attn_gym._backends.cute.utils import requires_int64_abi

CUTE_CAPABLE = cuda_runs_sm100_kernels()

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


@pytest.mark.skipif(not CUTE_CAPABLE, reason="the CuTe KDA kernels require capability 10.x")
def test_forced_int64_chunk_kda_matches_default_path(monkeypatch):
    """Force the i64 specializations on small inputs and match the i32 pipeline."""
    import attn_gym.linear.kda.bwd.cute.chunk_delta_h_bwd as delta_module
    import attn_gym.linear.kda.bwd.cute.chunk_kda_bwd_intra as intra_module
    import attn_gym.linear.kda.bwd.cute.chunk_kda_bwd_wy_dqkg_fused as wy_module
    import attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_inter_solve as inter_module
    import attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_intra_engine as fwd_intra_module
    from attn_gym.linear import chunk_kda

    def run():
        torch.manual_seed(13)
        tokens, heads = 192, 2
        q, k, v = (
            (
                torch.randn(1, tokens, heads, 128, device="cuda", dtype=torch.bfloat16) / 8
            ).requires_grad_()
            for _ in range(3)
        )
        gate = (-torch.rand(1, tokens, heads, 128, device="cuda")).requires_grad_()
        beta = torch.rand(1, tokens, heads, device="cuda").requires_grad_()
        offsets = torch.tensor([0, 65, 192], device="cuda", dtype=torch.int32)
        output, _ = chunk_kda(q, k, v, gate, beta, cu_seqlens=offsets)
        grads = torch.autograd.grad(output.float().square().sum(), (q, k, v, gate, beta))
        return (output, *grads)

    baseline = run()
    for module in (delta_module, intra_module, wy_module, inter_module, fwd_intra_module):
        monkeypatch.setattr(module, "requires_int64_abi", lambda *tensors: True)
    forced = run()
    for name, expected, actual in zip(
        ("output", "dq", "dk", "dv", "dgate", "dbeta"), baseline, forced, strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=lambda m, n=name: n)


@pytest.mark.skipif(not CUTE_CAPABLE, reason="the CuTe KDA kernels require capability 10.x")
def test_forced_int64_dense_delta_h_fusion_matches_default_path(monkeypatch):
    """Force the dense fused delta-H kernel's wide ABI on a small workload."""
    import attn_gym.linear.kda.bwd.cute.chunk_delta_h_bwd as delta_module
    from attn_gym.linear import chunk_kda

    def run():
        torch.manual_seed(17)
        shape = (1, 128, 2, 128)
        q, k, v = (
            (torch.randn(shape, device="cuda", dtype=torch.bfloat16) / 8).requires_grad_()
            for _ in range(3)
        )
        gate = (-torch.rand(shape, device="cuda") / 16).requires_grad_()
        beta = torch.rand(shape[:3], device="cuda").requires_grad_()
        output, _ = chunk_kda(q, k, v, gate, beta, kernel_options={"autotune": False})
        gradients = torch.autograd.grad(output.float().square().sum(), (q, k, v, gate, beta))
        return (output, *gradients)

    baseline = run()
    calls = 0

    def force_int64(*_tensors):
        nonlocal calls
        calls += 1
        return True

    monkeypatch.setattr(delta_module, "requires_int64_abi", force_int64)
    forced = run()
    assert calls > 0
    for expected, actual in zip(baseline, forced, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not CUTE_CAPABLE, reason="the CuTe KDA kernels require capability 10.x")
@pytest.mark.parametrize(
    ("batch", "tokens", "heads"),
    [
        pytest.param(1, 65, 2, id="tile-grid"),
        # The batch folded into gridDim.x: two backward tiles per batch, the second partial.
        pytest.param(65537, 33, 1, id="folded-batch"),
    ],
)
def test_forced_int64_bound_gate_matches_default_path(monkeypatch, batch, tokens, heads):
    """Force the private gate forward's and backward's wide ABI on an ordinary input."""
    import attn_gym.linear.kda.bwd.cute.gate_bwd as gate_module
    import attn_gym.linear.kda.fwd.cute.gate_fwd as gate_forward_module
    from attn_gym.linear.kda import bound_gate

    def run():
        torch.manual_seed(19)
        inputs = (
            torch.randn(
                batch, tokens, heads, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
            ),
            torch.randn(heads, device="cuda", requires_grad=True),
            torch.randn(heads, 128, device="cuda", requires_grad=True),
        )
        output = bound_gate(*inputs, impl="fused")
        return (output, *torch.autograd.grad(output.square().sum(), inputs))

    baseline = run()
    monkeypatch.setattr(gate_module, "requires_int64_abi", lambda *tensors: True)
    monkeypatch.setattr(gate_forward_module, "requires_int64_abi", lambda *tensors: True)
    forced = run()
    for expected, actual in zip(baseline, forced, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not CUTE_CAPABLE, reason="the CuTe KDA kernels require capability 10.x")
def test_bound_gate_oversized_singleton_stride_matches_compact():
    """Exercise automatic wide-ABI routing on the gate forward and backward."""
    from attn_gym.linear.kda import bound_gate

    def run(oversized_batch_stride: bool):
        torch.manual_seed(23)
        raw_gate = torch.randn(1, 65, 2, 128, device="cuda", dtype=torch.bfloat16)
        if oversized_batch_stride:
            raw_gate = raw_gate.as_strided(raw_gate.shape, (2**31, *raw_gate.stride()[1:]))
            assert requires_int64_abi(raw_gate)
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


@pytest.mark.skipif(not CUTE_CAPABLE, reason="the CuTe KDA kernels require capability 10.x")
def test_chunk_kda_oversized_singleton_stride_matches_compact():
    """Exercise automatic int64 ABI selection without allocating unreachable storage."""
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
            assert requires_int64_abi(q, k, v)
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
    not CUTE_CAPABLE or torch.cuda.get_device_properties(0).total_memory < 16 * 1024**3,
    reason="the executed-overflow regression peaks near 5.5 GiB",
)
def test_chunk_kda_executes_offsets_past_int32():
    """Actually execute element offsets beyond INT32_MAX, not just accept the ABI.

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
    assert requires_int64_abi(*packed) and not requires_int64_abi(*compact)

    gate = -torch.rand(1, tokens, heads, dim, device="cuda")
    beta = torch.rand(1, tokens, heads, device="cuda")
    offsets = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)

    with torch.no_grad():
        wide, _ = chunk_kda(*packed, gate, beta, cu_seqlens=offsets)
        narrow, _ = chunk_kda(*compact, gate, beta, cu_seqlens=offsets)
    torch.testing.assert_close(wide, narrow, rtol=0, atol=0)
