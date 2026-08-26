"""Correctness and determinism for dense Aqk-to-delta-H dv fusion."""

from __future__ import annotations

import pytest
import torch

from attn_gym.testing.kda import bwd_daqk_reference

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the fused delta-H backward requires SM100 or SM103",
)

D = 128


def make_inputs(tokens: int, heads: int, saturated: bool) -> tuple[torch.Tensor, ...]:
    """Build dense core tensors with a causal Aqk factor."""
    torch.manual_seed(tokens + heads + saturated)
    q = torch.randn(1, tokens, heads, D, device="cuda", dtype=torch.bfloat16) / 8
    k = torch.randn_like(q) / 8
    w = torch.randn_like(q) / 8
    do = torch.randn_like(q)
    rows = torch.arange(64, device="cuda")
    causal = rows[:, None] >= rows[None, :]
    aqk = torch.randn(1, tokens, heads, 64, device="cuda", dtype=torch.bfloat16) / 16
    aqk = (
        aqk.view(1, tokens // 64, 64, heads, 64)
        .masked_fill(~causal.view(1, 1, 64, 1, 64), 0)
        .view(1, tokens, heads, 64)
        .contiguous()
    )
    if saturated:
        positions = torch.arange(tokens, device="cuda") % 64 + 1
        gate = (-5.0 * torch.log2(torch.tensor(torch.e, device="cuda")) * positions).view(
            1, tokens, 1, 1
        )
        gate = gate.expand(1, tokens, heads, D).contiguous()
    else:
        increments = -torch.rand(1, tokens, heads, D, device="cuda") / 16
        gate = increments.view(1, tokens // 64, 64, heads, D).cumsum(2).view_as(increments)
    return q, k, w, do, aqk, gate


def normalized_rms(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Return normalized FP32 RMS error."""
    difference = actual.float() - expected.float()
    denominator = expected.float().square().mean().sqrt().clamp_min(1e-12)
    return (difference.square().mean().sqrt() / denominator).item()


def delta_h_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    d_output: torch.Tensor,
    aqk: torch.Tensor,
    lengths: list[int],
    *,
    gate: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    d_final_state: torch.Tensor | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Evaluate the packed reverse state recurrence in FP32."""
    _, _, heads, key_dim = q.shape
    value_dim = d_output.shape[-1]
    qf, kf, wf, dof, aqkf = (tensor.float() for tensor in (q, k, w, d_output, aqk))
    gatef = gate.float() if gate is not None else None
    d_value = torch.zeros_like(dof)
    chunk_states = []
    d_initial_state = [] if initial_state is not None else None
    begin = 0
    for sequence, length in enumerate(lengths):
        state = (
            d_final_state[sequence].float().clone()
            if d_final_state is not None
            else torch.zeros(heads, key_dim, value_dim, device=q.device)
        )
        sequence_states = []
        for offset in reversed(range(0, length, 64)):
            end = min(offset + 64, length)
            token = slice(begin + offset, begin + end)
            size = end - offset
            sequence_states.append((offset // 64, state.clone()))
            dv_intra = torch.einsum("blhm,blhv->bmhv", aqkf[:, token, :, :size], dof[:, token])
            dv = torch.einsum("blhk,bhkv->blhv", kf[:, token], state[None]) + dv_intra
            d_value[:, token] = dv
            decay = gatef[0, begin + end - 1].exp2()[..., None] if gatef is not None else 1
            state = (
                decay * state
                + scale * torch.einsum("blhk,blhv->bhkv", qf[:, token], dof[:, token])[0]
                - torch.einsum("blhk,blhv->bhkv", wf[:, token], dv)[0]
            )
        chunk_states.extend(state for _, state in sorted(sequence_states))
        if d_initial_state is not None:
            d_initial_state.append(state)
        begin += length
    chunk_state = (
        torch.stack(chunk_states)[None]
        if chunk_states
        else torch.empty(1, 0, heads, key_dim, value_dim, device=q.device)
    )
    dh0 = torch.stack(d_initial_state) if d_initial_state is not None else None
    return chunk_state, dh0, d_value


@pytest.mark.parametrize("bv", [16, 32])
@pytest.mark.parametrize("saturated", [False, True], ids=["mild", "gate-bound"])
def test_delta_h_dv_fusion_matches_reference_and_is_deterministic(bv: int, saturated: bool):
    from attn_gym.linear.kda.bwd.cute.chunk_delta_h_bwd import (
        blackwell_delta_h_bwd_dhu_dv_fused,
    )

    q, k, w, do, aqk, gate = make_inputs(128, 2, saturated)
    expected = delta_h_reference(
        q,
        k,
        w,
        do,
        aqk,
        [128],
        gate=gate,
        initial_state=None,
        d_final_state=None,
        scale=D**-0.5,
    )
    actual = blackwell_delta_h_bwd_dhu_dv_fused(
        q,
        k,
        w,
        do,
        aqk,
        gk=gate,
        scale=D**-0.5,
        chunk_size=64,
        bv=bv,
    )
    torch.cuda.synchronize()
    for got, reference in zip(actual, expected, strict=True):
        if got is None:
            assert reference is None
            continue
        assert torch.isfinite(got).all()
        assert normalized_rms(got, reference) <= 2e-2

    for _ in range(20):
        rerun = blackwell_delta_h_bwd_dhu_dv_fused(
            q,
            k,
            w,
            do,
            aqk,
            gk=gate,
            scale=D**-0.5,
            chunk_size=64,
            bv=bv,
        )
        torch.cuda.synchronize()
        for got, reference in zip(rerun, actual, strict=True):
            if got is None:
                assert reference is None
            else:
                torch.testing.assert_close(got, reference, rtol=0, atol=0)


@pytest.mark.parametrize("use_gate", [False, True], ids=["no-gate", "gate"])
@pytest.mark.parametrize("state_mode", ["none", "initial", "final", "both"])
def test_delta_h_dv_fusion_runtime_flags(use_gate: bool, state_mode: str):
    from attn_gym.linear.kda.bwd.cute.chunk_delta_h_bwd import (
        blackwell_delta_h_bwd_dhu_dv_fused,
    )

    q, k, w, d_output, aqk, gate = make_inputs(128, 2, False)
    state_shape = (1, 2, D, D)
    initial_state = (
        torch.randn(state_shape, device="cuda") / 100
        if state_mode in ("initial", "both")
        else None
    )
    d_final_state = (
        torch.randn(state_shape, device="cuda") / 100 if state_mode in ("final", "both") else None
    )
    kwargs = {
        "gk": gate if use_gate else None,
        "h0": initial_state,
        "dht": d_final_state,
        "scale": D**-0.5,
        "chunk_size": 64,
        "bv": 16,
    }
    expected = delta_h_reference(
        q,
        k,
        w,
        d_output,
        aqk,
        [128],
        gate=kwargs["gk"],
        initial_state=initial_state,
        d_final_state=d_final_state,
        scale=D**-0.5,
    )
    actual = blackwell_delta_h_bwd_dhu_dv_fused(q, k, w, d_output, aqk, **kwargs)
    for got, reference in zip(actual, expected, strict=True):
        if got is None:
            assert reference is None
        else:
            assert torch.isfinite(got).all()
            assert normalized_rms(got, reference) <= 2e-2


@pytest.mark.parametrize("bv", [16, 32])
@pytest.mark.parametrize("force_int64", [False, True], ids=["i32", "i64"])
@pytest.mark.parametrize("heads", [1, 2])
def test_packed_delta_h_dv_fusion_matches_reference(
    bv: int, force_int64: bool, heads: int, monkeypatch
):
    from attn_gym.linear.kda.bwd.cute.chunk_delta_h_bwd import (
        _blackwell_delta_h_bwd_dhu_dv_fused_packed,
    )
    from attn_gym.linear.kda.bwd.triton.chunk_kda_bwd_daqk import chunk_kda_bwd_daqk
    from attn_gym.linear.kda.chunk_scheduler import prepare_ragged_chunk_metadata
    from attn_gym.testing.kda import cumulative_sequence_offsets

    torch.manual_seed(23)
    lengths = [65, 0, 63]
    tokens = sum(lengths)
    shape = (1, tokens, heads, D)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16) / 8
    k = torch.randn_like(q) / 8
    w = torch.randn_like(q) / 8
    value_new = torch.randn_like(q) / 8
    d_output = torch.randn_like(q) / 8
    gate = -torch.rand(shape, device="cuda") / 16
    aqk = torch.zeros(1, tokens, heads, 64, device="cuda", dtype=torch.bfloat16)
    begin = 0
    for length in lengths:
        for local_token in range(length):
            row = local_token % 64
            aqk[:, begin + local_token, :, : row + 1] = (
                torch.randn(1, heads, row + 1, device="cuda") / 16
            )
        begin += length
    initial_state = torch.randn(len(lengths), heads, D, D, device="cuda") / 32
    d_final_state = torch.randn_like(initial_state) / 32
    metadata = prepare_ragged_chunk_metadata(cumulative_sequence_offsets(lengths), tokens, 64)
    expected_daqk = bwd_daqk_reference(value_new, d_output, lengths, D**-0.5)
    expected = delta_h_reference(
        q,
        k,
        w,
        d_output,
        aqk,
        lengths,
        gate=gate,
        initial_state=initial_state,
        d_final_state=d_final_state,
        scale=D**-0.5,
    )
    if force_int64:
        import attn_gym.linear.kda.bwd.cute.chunk_delta_h_bwd as fused_module

        monkeypatch.setattr(fused_module, "requires_int64_abi", lambda *_tensors: True)
    actual = _blackwell_delta_h_bwd_dhu_dv_fused_packed(
        q,
        k,
        w,
        d_output,
        aqk,
        metadata,
        gk=gate,
        h0=initial_state,
        dht=d_final_state,
        scale=D**-0.5,
        bv=bv,
    )
    actual_daqk = chunk_kda_bwd_daqk(
        value_new,
        d_output,
        D**-0.5,
        metadata=metadata,
    )
    active_chunks = metadata.chunk_offsets[-1].item()
    for got, reference in (
        (actual[0][:, :active_chunks], expected[0][:, :active_chunks]),
        (actual[1], expected[1]),
        (actual[2][:, :tokens], expected[2][:, :tokens]),
    ):
        assert got is not None and reference is not None
        assert torch.isfinite(got).all()
        assert normalized_rms(got, reference) <= 2e-2
    torch.testing.assert_close(actual_daqk, expected_daqk, rtol=5e-4, atol=2e-8)
    torch.testing.assert_close(actual[1][1], d_final_state[1], rtol=0, atol=0)
    if not force_int64 and bv == 16:
        for _ in range(3):
            rerun = _blackwell_delta_h_bwd_dhu_dv_fused_packed(
                q,
                k,
                w,
                d_output,
                aqk,
                metadata,
                gk=gate,
                h0=initial_state,
                dht=d_final_state,
                scale=D**-0.5,
                bv=bv,
            )
            for index, (result, reference) in enumerate(zip(rerun, actual, strict=True)):
                assert result is not None and reference is not None
                if index == 0:
                    result = result[:, :active_chunks]
                    reference = reference[:, :active_chunks]
                torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_packed_delta_h_dv_fusion_cuda_graph_replays_smaller_endpoint():
    from attn_gym.linear.kda.bwd.cute.chunk_delta_h_bwd import (
        _blackwell_delta_h_bwd_dhu_dv_fused_packed,
    )
    from attn_gym.linear.kda.chunk_scheduler import prepare_ragged_chunk_metadata
    from attn_gym.testing.kda import cumulative_sequence_offsets

    torch.manual_seed(41)
    tokens, heads = 1024, 1
    shape = (1, tokens, heads, D)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16) / 8
    k = torch.randn_like(q) / 8
    w = torch.randn_like(q) / 8
    d_output = torch.randn_like(q) / 8
    gate = -torch.rand(shape, device="cuda") / 16
    aqk = torch.zeros(1, tokens, heads, 64, device="cuda", dtype=torch.bfloat16)
    cu_seqlens = cumulative_sequence_offsets([512, 512])
    warm_metadata = prepare_ragged_chunk_metadata(cu_seqlens, tokens, 64)
    _blackwell_delta_h_bwd_dhu_dv_fused_packed(
        q, k, w, d_output, aqk, warm_metadata, gk=gate, scale=D**-0.5, bv=16
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        metadata = prepare_ragged_chunk_metadata(cu_seqlens, tokens, 64)
        actual = _blackwell_delta_h_bwd_dhu_dv_fused_packed(
            q, k, w, d_output, aqk, metadata, gk=gate, scale=D**-0.5, bv=16
        )

    active_lengths = [257, 255]
    active_tokens = sum(active_lengths)
    cu_seqlens.copy_(cumulative_sequence_offsets(active_lengths))
    with torch.no_grad():
        for tensor in (q, k, w, d_output, gate, aqk):
            tensor[:, active_tokens:] = float("nan")
    graph.replay()
    torch.cuda.synchronize()

    expected_metadata = prepare_ragged_chunk_metadata(
        cumulative_sequence_offsets(active_lengths), active_tokens, 64
    )
    expected = delta_h_reference(
        q[:, :active_tokens],
        k[:, :active_tokens],
        w[:, :active_tokens],
        d_output[:, :active_tokens],
        aqk[:, :active_tokens],
        active_lengths,
        gate=gate[:, :active_tokens],
        initial_state=None,
        d_final_state=None,
        scale=D**-0.5,
    )
    active_chunks = expected_metadata.chunk_offsets[-1].item()
    for result, reference in (
        (actual[0][:, :active_chunks], expected[0][:, :active_chunks]),
        (actual[2][:, :active_tokens], expected[2]),
    ):
        assert torch.isfinite(result).all()
        assert normalized_rms(result, reference) <= 2e-2


def test_packed_delta_h_dv_fusion_preserves_empty_state_slots():
    from attn_gym.linear.kda.bwd.cute.chunk_delta_h_bwd import (
        _blackwell_delta_h_bwd_dhu_dv_fused_packed,
    )
    from attn_gym.linear.kda.chunk_scheduler import prepare_ragged_chunk_metadata
    from attn_gym.testing.kda import cumulative_sequence_offsets

    heads = 1
    shape = (1, 0, heads, D)
    q = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    aqk = torch.empty(1, 0, heads, 64, device="cuda", dtype=torch.bfloat16)
    initial_state = torch.randn(2, heads, D, D, device="cuda") / 32
    d_final_state = torch.randn_like(initial_state) / 32
    metadata = prepare_ragged_chunk_metadata(cumulative_sequence_offsets([0, 0]), 0, 64)
    dh, d_initial_state, d_value = _blackwell_delta_h_bwd_dhu_dv_fused_packed(
        q,
        q,
        q,
        q,
        aqk,
        metadata,
        h0=initial_state,
        dht=d_final_state,
        bv=16,
    )

    assert dh.shape == (1, 0, heads, D, D)
    assert d_value.shape == q.shape
    torch.testing.assert_close(d_initial_state, d_final_state, rtol=0, atol=0)


def test_delta_h_dv_fusion_rejects_invalid_dense_contracts():
    from attn_gym.linear.kda.bwd.cute.chunk_delta_h_bwd import (
        blackwell_delta_h_bwd_dhu_dv_fused,
    )

    inputs = make_inputs(128, 2, False)
    with pytest.raises(ValueError, match="complete chunks"):
        blackwell_delta_h_bwd_dhu_dv_fused(
            *(tensor[:, :65].contiguous() for tensor in inputs[:5]),
            gk=inputs[5][:, :65].contiguous(),
        )
    with pytest.raises(ValueError, match="chunk_size=64"):
        blackwell_delta_h_bwd_dhu_dv_fused(*inputs[:5], gk=inputs[5], chunk_size=32)
    bad_state = torch.empty(2, 2, D, D, device="cuda")
    for name in ("h0", "dht"):
        with pytest.raises(ValueError, match=rf"{name} must have shape"):
            blackwell_delta_h_bwd_dhu_dv_fused(
                *inputs[:5],
                gk=inputs[5],
                **{name: bad_state},
            )
