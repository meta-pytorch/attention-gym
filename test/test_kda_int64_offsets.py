"""Int64 offset-width selection and equivalence tests for the CuTe KDA kernels."""

from __future__ import annotations

import pytest
import torch

from attn_gym._backends.cute.utils import requires_int64_abi

CUTE_CAPABLE = torch.cuda.is_available() and torch.cuda.get_device_capability() in (
    (10, 0),
    (10, 3),
)


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
    import attn_gym.linear.kda.bwd.cute.chunk_delta_h_bwd_v1 as delta_module
    import attn_gym.linear.kda.bwd.cute.chunk_kda_bwd_intra as intra_module
    import attn_gym.linear.kda.bwd.cute.chunk_kda_bwd_wy_dqkg_fused as wy_module
    import attn_gym.linear.kda.bwd.cute.gate_bwd_fused as gate_module
    import attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_inter_solve as inter_module
    from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd import chunk_kda

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
    for module in (delta_module, intra_module, wy_module, gate_module, inter_module):
        monkeypatch.setattr(module, "requires_int64_abi", lambda *tensors: True)
    forced = run()
    for name, expected, actual in zip(
        ("output", "dq", "dk", "dv", "dgate", "dbeta"), baseline, forced, strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=lambda m, n=name: n)


@pytest.mark.skipif(
    not CUTE_CAPABLE or torch.cuda.get_device_properties(0).total_memory < 100 * 1024**3,
    reason="the over-capture regression allocates ~60GB of capacity buffers",
)
def test_chunk_kda_262144_token_packed_capacity_matches_exact():
    """Run the full core at the first int64-requiring capacity with packed pitch."""
    from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd import chunk_kda

    heads, dim, active = 32, 128, 8192

    def run(capacity):
        torch.manual_seed(11)
        qkv = torch.full(
            (1, capacity, 3, heads, dim), float("nan"), device="cuda", dtype=torch.bfloat16
        )
        qkv[:, :active] = (
            torch.randn(1, active, 3, heads, dim, device="cuda", dtype=torch.bfloat16) / 8
        )
        q, k, v = (qkv[:, :, i].requires_grad_() for i in range(3))
        gate = torch.full((1, capacity, heads, dim), float("nan"), device="cuda")
        gate[:, :active] = -torch.rand(1, active, heads, dim, device="cuda")
        gate = gate.requires_grad_()
        beta = torch.full((1, capacity, heads), float("nan"), device="cuda")
        beta[:, :active] = torch.rand(1, active, heads, device="cuda")
        beta = beta.requires_grad_()
        grad = torch.zeros(1, capacity, heads, dim, device="cuda")
        grad[:, :active] = torch.randn(1, active, heads, dim, device="cuda") / 8

        offsets = torch.tensor([0, active], device="cuda", dtype=torch.int32)
        out, _ = chunk_kda(q, k, v, gate, beta, cu_seqlens=offsets, persistent=True)
        loss = (out[:, :active].float() * grad[:, :active]).sum()
        grads = torch.autograd.grad(loss, (q, k, v, gate, beta))
        return (out[:, :active].clone(), *(g[:, :active].clone() for g in grads))

    small = run(active)
    torch.cuda.empty_cache()
    large = run(262144)
    for name, expected, actual in zip(
        ("output", "dq", "dk", "dv", "dgate", "dbeta"), small, large, strict=True
    ):
        assert not actual.isnan().any(), name
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2, msg=lambda m, n=name: n)
