"""Layout, alignment, and address-width coverage for private GDN Mega launchers."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip(
    "cutlass.experimental",
    reason="the CuTeDSL 4.7 GDN path requires nvidia-cutlass-dsl>=4.7",
)

from attn_gym._backends.cute.utils import requires_int64_abi
from attn_gym.linear._delta_rule.mega.gdn_backward import chunk_gdn_bwd_mega_packed
from attn_gym.linear._delta_rule.mega.gdn_forward import run_forward
from attn_gym.testing import make_gdn_test_inputs

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the CuTeDSL 4.7 GDN path requires SM100 or SM103",
)


def pad_last_mode(tensor: torch.Tensor, padding: int = 8) -> torch.Tensor:
    """Copy a tensor into aligned rows with padding after the contiguous mode."""
    storage = torch.empty(
        *tensor.shape[:-1], tensor.shape[-1] + padding, dtype=tensor.dtype, device=tensor.device
    )
    return storage[..., : tensor.shape[-1]].copy_(tensor)


def pad_scalar_rows(tensor: torch.Tensor, padding: int = 2) -> torch.Tensor:
    """Copy `[B,T,H]` scalars into padded token rows while keeping H contiguous."""
    storage = torch.empty(
        tensor.shape[0],
        tensor.shape[1],
        tensor.shape[2] + padding,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return storage[..., : tensor.shape[2]].copy_(tensor)


def pad_state_sequences(tensor: torch.Tensor, padding: int = 16) -> torch.Tensor:
    """Copy `[N,H,V,K]` state into padded sequence slots with dense inner slabs."""
    sequences, heads, value_dim, key_dim = tensor.shape
    sequence_stride = heads * value_dim * key_dim + padding
    storage = torch.empty(sequences * sequence_stride, dtype=tensor.dtype, device=tensor.device)
    return torch.as_strided(
        storage,
        tensor.shape,
        (sequence_stride, value_dim * key_dim, key_dim, 1),
    ).copy_(tensor)


def test_gdn_mega_outer_strides_match_compact_forward_and_backward() -> None:
    """Dynamic outer strides must preserve all forward and backward values."""
    inputs = make_gdn_test_inputs(
        (65, 63), key_heads=2, value_heads=2, dtype=torch.bfloat16, seed=211
    )
    q, k, value, gate, beta, state, cu_seqlens = inputs
    torch.manual_seed(223)
    d_output = torch.randn_like(value)
    d_final_state = torch.randn_like(state)
    expected_forward = run_forward(
        q, k, value, gate, beta, cu_seqlens, state, scale=None, output_final_state=True
    )
    expected_backward = chunk_gdn_bwd_mega_packed(
        q,
        k,
        value,
        gate,
        beta,
        d_output,
        cu_seqlens,
        state,
        d_final_state,
    )

    strided = (
        pad_last_mode(q),
        pad_last_mode(k),
        pad_last_mode(value),
        pad_scalar_rows(gate),
        pad_scalar_rows(beta),
        pad_state_sequences(state),
        cu_seqlens,
    )
    strided_d_output = pad_last_mode(d_output)
    strided_d_final_state = pad_state_sequences(d_final_state)
    actual_forward = run_forward(
        *strided[:5],
        strided[6],
        strided[5],
        scale=None,
        output_final_state=True,
    )
    actual_backward = chunk_gdn_bwd_mega_packed(
        *strided[:5],
        strided_d_output,
        strided[6],
        strided[5],
        strided_d_final_state,
    )

    for actual, expected in zip(actual_forward, expected_forward, strict=True):
        assert actual is not None and expected is not None
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(actual_backward, expected_backward, strict=True):
        assert actual is not None and expected is not None
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_gdn_mega_forward_rejects_misaligned_and_noncontiguous_inner_modes() -> None:
    """The launcher must reject layouts that violate its advertised TMA signature."""
    q, k, value, gate, beta, state, cu_seqlens = make_gdn_test_inputs(
        (64,), key_heads=2, value_heads=2, dtype=torch.bfloat16, seed=227
    )
    storage = torch.empty(q.numel() + 1, dtype=q.dtype, device=q.device)
    misaligned_q = storage[1:].view_as(q).copy_(q)
    with pytest.raises(TypeError, match="TMA-compatible"):
        run_forward(
            misaligned_q,
            k,
            value,
            gate,
            beta,
            cu_seqlens,
            state,
            scale=None,
            output_final_state=True,
        )

    inner_storage = torch.empty(q.numel() * 2, dtype=q.dtype, device=q.device)
    noncontiguous_inner_q = torch.as_strided(
        inner_storage,
        q.shape,
        (q.stride(0) * 2, q.stride(1) * 2, q.stride(2) * 2, 2),
    ).copy_(q)
    with pytest.raises(TypeError, match="TMA-compatible"):
        run_forward(
            noncontiguous_inner_q,
            k,
            value,
            gate,
            beta,
            cu_seqlens,
            state,
            scale=None,
            output_final_state=True,
        )


def test_gdn_mega_forced_int64_forward_backward_matches_int32(monkeypatch) -> None:
    """Every private kernel must preserve results under its int64 ABI specialization."""
    from attn_gym.linear._delta_rule.mega.kernels import (
        gdn_bprop_f16,
        gdn_prefill_f16,
        gdn_recompute_f16,
    )

    inputs = make_gdn_test_inputs(
        (65, 63), key_heads=2, value_heads=2, dtype=torch.float16, seed=229
    )
    q, k, value, gate, beta, state, cu_seqlens = inputs
    torch.manual_seed(233)
    d_output = torch.randn_like(value)
    d_final_state = torch.randn_like(state)
    expected_forward = run_forward(
        q, k, value, gate, beta, cu_seqlens, state, scale=None, output_final_state=True
    )
    expected_backward = chunk_gdn_bwd_mega_packed(
        q,
        k,
        value,
        gate,
        beta,
        d_output,
        cu_seqlens,
        state,
        d_final_state,
    )

    for module in (gdn_prefill_f16, gdn_recompute_f16, gdn_bprop_f16):
        monkeypatch.setattr(module, "requires_int64_abi", lambda *_: True)
    actual_forward = run_forward(
        q, k, value, gate, beta, cu_seqlens, state, scale=None, output_final_state=True
    )
    actual_backward = chunk_gdn_bwd_mega_packed(
        q,
        k,
        value,
        gate,
        beta,
        d_output,
        cu_seqlens,
        state,
        d_final_state,
    )

    for actual, expected in zip(actual_forward, expected_forward, strict=True):
        assert actual is not None and expected is not None
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(actual_backward, expected_backward, strict=True):
        assert actual is not None and expected is not None
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_gdn_mega_active_offsets_past_int32_match_compact() -> None:
    """Execute forward and backward with active Q/K addresses beyond signed int32."""
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < 24 * 1024**3:
        pytest.skip("active int64 GDN coverage requires at least 24 GiB free")

    q, k, value, gate, beta, _state, cu_seqlens = make_gdn_test_inputs(
        (2,), key_heads=1, value_heads=1, dtype=torch.bfloat16, seed=237
    )
    d_output = torch.randn_like(value)
    expected_forward = run_forward(
        q, k, value, gate, beta, cu_seqlens, None, scale=None, output_final_state=False
    )
    expected_backward = chunk_gdn_bwd_mega_packed(q, k, value, gate, beta, d_output, cu_seqlens)

    active_stride = 2**31 + 128

    def wide_tokens(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        storage = torch.empty(
            active_stride + tensor.shape[-1], dtype=tensor.dtype, device=tensor.device
        )
        view = torch.as_strided(
            storage,
            tensor.shape,
            (active_stride + tensor.shape[-1], active_stride, tensor.shape[-1], 1),
        ).copy_(tensor)
        return storage, view

    q_storage, wide_q = wide_tokens(q)
    k_storage, wide_k = wide_tokens(k)
    assert requires_int64_abi(wide_q, wide_k)
    actual_forward = run_forward(
        wide_q,
        wide_k,
        value,
        gate,
        beta,
        cu_seqlens,
        None,
        scale=None,
        output_final_state=False,
    )
    actual_backward = chunk_gdn_bwd_mega_packed(
        wide_q, wide_k, value, gate, beta, d_output, cu_seqlens
    )
    torch.testing.assert_close(actual_forward[0], expected_forward[0], rtol=0, atol=0)
    for actual, expected in zip(actual_backward, expected_backward, strict=True):
        if actual is None:
            assert expected is None
        else:
            assert expected is not None
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    del q_storage, k_storage


def test_gdn_mega_oversized_singleton_stride_executes_int64_path() -> None:
    """An ABI-visible oversized singleton stride must route wide without huge storage."""
    q, k, value, gate, beta, state, cu_seqlens = make_gdn_test_inputs(
        (1,), key_heads=2, value_heads=2, dtype=torch.bfloat16, seed=239
    )
    wide_gate = torch.empty_strided(
        gate.shape,
        (2**31 + 32, 2**31 + 16, 1),
        dtype=gate.dtype,
        device=gate.device,
    ).copy_(gate)
    assert requires_int64_abi(wide_gate)
    expected = run_forward(
        q, k, value, gate, beta, cu_seqlens, state, scale=None, output_final_state=True
    )
    actual = run_forward(
        q,
        k,
        value,
        wide_gate,
        beta,
        cu_seqlens,
        state,
        scale=None,
        output_final_state=True,
    )
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert actual_tensor is not None and expected_tensor is not None
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


def test_gdn_mega_terminal_capacity_slack_is_not_read() -> None:
    """Poisoned physical rows beyond the terminal packed offset must not affect active work."""
    q, k, value, gate, beta, state, cu_seqlens = make_gdn_test_inputs(
        (65, 0, 63), key_heads=1, value_heads=2, dtype=torch.bfloat16, seed=241
    )
    torch.manual_seed(251)
    d_output = torch.randn_like(value)
    d_final_state = torch.randn_like(state)
    expected_forward = run_forward(
        q, k, value, gate, beta, cu_seqlens, state, scale=None, output_final_state=True
    )
    expected_backward = chunk_gdn_bwd_mega_packed(
        q,
        k,
        value,
        gate,
        beta,
        d_output,
        cu_seqlens,
        state,
        d_final_state,
    )

    slack = 7

    def poison_rows(tensor: torch.Tensor) -> torch.Tensor:
        shape = (tensor.shape[0], slack, *tensor.shape[2:])
        poison = torch.full(shape, torch.inf, dtype=tensor.dtype, device=tensor.device)
        return torch.cat((tensor, poison), dim=1)

    q_slack, k_slack, value_slack = (poison_rows(tensor) for tensor in (q, k, value))
    gate_slack = torch.cat(
        (gate, torch.full((1, slack, gate.shape[2]), 20.0, device=gate.device)), dim=1
    )
    beta_slack = torch.cat(
        (beta, torch.full((1, slack, beta.shape[2]), torch.inf, device=beta.device)), dim=1
    )
    d_output_slack = poison_rows(d_output)
    actual_forward = run_forward(
        q_slack,
        k_slack,
        value_slack,
        gate_slack,
        beta_slack,
        cu_seqlens,
        state,
        scale=None,
        output_final_state=True,
    )
    actual_backward = chunk_gdn_bwd_mega_packed(
        q_slack,
        k_slack,
        value_slack,
        gate_slack,
        beta_slack,
        d_output_slack,
        cu_seqlens,
        state,
        d_final_state,
    )
    active_tokens = q.shape[1]
    torch.testing.assert_close(
        actual_forward[0][:, :active_tokens], expected_forward[0], rtol=0, atol=0
    )
    assert actual_forward[1] is not None and expected_forward[1] is not None
    torch.testing.assert_close(actual_forward[1], expected_forward[1], rtol=0, atol=0)
    for index, (actual, expected) in enumerate(
        zip(actual_backward, expected_backward, strict=True)
    ):
        assert actual is not None and expected is not None
        if index < 5:
            torch.testing.assert_close(actual[:, :active_tokens], expected, rtol=0, atol=0)
        else:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
