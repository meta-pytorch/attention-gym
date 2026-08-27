# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Correctness and integration tests for the fused KDA recurrence."""

from itertools import pairwise

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

from attn_gym.linear import recurrent_kda
from attn_gym.linear.kda.constants import LOG2_E
from attn_gym.linear.kda.fwd.triton.recurrent import (
    _recurrent_fwd_no_state_op,
    _recurrent_fwd_op,
    _recurrent_fwd_paged_op,
)
from attn_gym.linear.kda.naive import naive_recurrent_kda
from attn_gym.testing import cumulative_sequence_offsets, strided_state_pool

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="recurrent_kda requires CUDA"
)

BLACKWELL = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0)


def _inputs(
    batch: int = 2,
    tokens: int = 37,
    heads: int = 2,
    key_dim: int = 64,
    value_dim: int = 64,
    dtype: torch.dtype = torch.float32,
    initial_state: bool = False,
    seed: int = 0,
):
    torch.manual_seed(seed)
    # KDA L2-normalizes q and k before the core; unnormalized keys make the
    # delta-rule recurrence exponentially unstable and useless for comparison.
    q = F.normalize(torch.randn(batch, tokens, heads, key_dim, device="cuda"), dim=-1).to(dtype)
    k = F.normalize(torch.randn(batch, tokens, heads, key_dim, device="cuda"), dim=-1).to(dtype)
    v = torch.randn(batch, tokens, heads, value_dim, device="cuda", dtype=dtype)
    # Realistic bounded natural-log decays keep the recurrence stable over the scan.
    gate = -torch.rand(batch, tokens, heads, key_dim, device="cuda") * 3.0
    beta = torch.rand(batch, tokens, heads, device="cuda")
    state = torch.randn(batch, heads, key_dim, value_dim, device="cuda") if initial_state else None
    return q, k, v, gate, beta, state


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_initial_state", [False, True])
@pytest.mark.parametrize("tokens", [1, 37])
def test_recurrent_matches_naive_dense(dtype: torch.dtype, use_initial_state: bool, tokens: int):
    """Match the reference scan on dense batches, including single-token decode."""
    q, k, v, gate, beta, state = _inputs(
        tokens=tokens, dtype=dtype, initial_state=use_initial_state
    )

    output, final_state = recurrent_kda(q, k, v, gate, beta, state, output_final_state=True)
    expected, expected_state = naive_recurrent_kda(
        q.float(),
        k.float(),
        v.float(),
        gate * LOG2_E,
        beta,
        initial_state=state,
        output_final_state=True,
    )

    tolerance = 1e-5 if dtype == torch.float32 else 2e-2
    assert output.dtype == q.dtype
    assert final_state is not None and final_state.dtype == torch.float32
    torch.testing.assert_close(output.float(), expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(final_state, expected_state, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize(("key_dim", "value_dim"), [(80, 48), (128, 128)])
def test_recurrent_matches_naive_non_power_of_two(key_dim: int, value_dim: int):
    """Mask partial key and value blocks correctly."""
    q, k, v, gate, beta, _ = _inputs(key_dim=key_dim, value_dim=value_dim, seed=1)

    output, final_state = recurrent_kda(q, k, v, gate, beta, output_final_state=True)
    expected, expected_state = naive_recurrent_kda(
        q, k, v, gate * LOG2_E, beta, output_final_state=True
    )
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(final_state, expected_state, rtol=1e-5, atol=1e-5)


def test_recurrent_autotunes_value_tile():
    """Select the only valid candidate for a value dimension below the minimum tile."""
    q, k, v, gate, beta, _ = _inputs(key_dim=7, value_dim=3, seed=2)

    output, _ = recurrent_kda(q, k, v, gate, beta, autotune=True)
    expected, _ = recurrent_kda(q, k, v, gate, beta, autotune=False)

    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


def test_recurrent_paged_skips_autotune(monkeypatch):
    q, k, v, gate, beta, _ = _inputs(batch=2, tokens=1, seed=3)
    _, pool = strided_state_pool(3, q.shape[2], q.shape[3], v.shape[-1])
    slots = torch.tensor([1, 2], device="cuda", dtype=torch.int32)

    class UnexpectedAutotune:
        def __getitem__(self, _grid):
            raise AssertionError("paged execution must not invoke the autotuner")

    from attn_gym.linear._delta_rule import recurrent as delta_rule_recurrent

    monkeypatch.setattr(
        delta_rule_recurrent, "recurrent_delta_rule_fwd_kernel", UnexpectedAutotune()
    )
    recurrent_kda(q, k, v, gate, beta, pool, state_indices=slots, autotune=True)


def test_recurrent_packed_capacity_and_empty_slots():
    """Pass empty-slot state through and ignore rows past the terminal offset."""
    q, k, v, gate, beta, _ = _inputs(batch=1, tokens=32, seed=3)
    cu_seqlens = cumulative_sequence_offsets([0, 11, 16, 0])
    initial_state = torch.randn(4, q.shape[2], q.shape[3], v.shape[-1], device="cuda")

    output, final_state = recurrent_kda(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
    )
    assert final_state is not None
    # Empty padding slots preserve their incoming state bitwise.
    torch.testing.assert_close(final_state[0], initial_state[0], rtol=0, atol=0)
    torch.testing.assert_close(final_state[3], initial_state[3], rtol=0, atol=0)
    for sequence, (start, end) in enumerate(pairwise(cu_seqlens.cpu().tolist())):
        if start == end:
            continue
        expected, expected_state = naive_recurrent_kda(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            gate[:, start:end] * LOG2_E,
            beta[:, start:end],
            initial_state=initial_state[sequence : sequence + 1],
            output_final_state=True,
        )
        torch.testing.assert_close(output[:, start:end], expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            final_state[sequence : sequence + 1], expected_state, rtol=1e-5, atol=1e-5
        )


@pytest.mark.skipif(not BLACKWELL, reason="chunk_kda requires CUDA capability 10.0")
def test_recurrent_agrees_with_chunked_core():
    """Cross-check the decode scan against the training core on shared inputs."""
    from attn_gym.linear import chunk_kda

    batch, tokens, heads, head_dim = 2, 128, 2, 128
    q, k, v, _, beta = _inputs(
        batch=batch,
        tokens=tokens,
        heads=heads,
        key_dim=head_dim,
        value_dim=head_dim,
        dtype=torch.bfloat16,
        seed=4,
    )[:5]
    gate = -5.0 * torch.rand(batch, tokens, heads, head_dim, device="cuda")

    recurrent_output, recurrent_state = recurrent_kda(q, k, v, gate, beta, output_final_state=True)
    chunked_output, chunked_state = chunk_kda(q, k, v, gate, beta, output_final_state=True)
    torch.testing.assert_close(
        recurrent_output.float(), chunked_output.float(), rtol=5e-2, atol=5e-2
    )
    torch.testing.assert_close(recurrent_state, chunked_state, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("packed", [False, True])
def test_recurrent_paged_matches_gather_scatter(packed: bool):
    """Slot indexing equals a native gather, dense scan, and scatter round trip."""
    q, k, v, gate, beta, _ = _inputs(batch=1 if packed else 3, tokens=32, seed=8)
    cu_seqlens = cumulative_sequence_offsets([11, 16, 5]) if packed else None
    storage, pool = strided_state_pool(7, q.shape[2], q.shape[3], v.shape[-1])
    storage_before = storage.clone()
    slots = torch.tensor([5, 1, 3], device="cuda", dtype=torch.int32)
    before = pool.clone()

    output, final_state = recurrent_kda(
        q, k, v, gate, beta, pool, cu_seqlens=cu_seqlens, state_indices=slots
    )

    assert final_state is None
    expected_pool = before.clone()
    expected, expected_state = naive_recurrent_kda(
        q,
        k,
        v,
        gate * LOG2_E,
        beta,
        initial_state=before[slots.long()].transpose(-1, -2).contiguous(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    expected_pool[slots.long()] = expected_state.transpose(-1, -2)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(pool, expected_pool, rtol=1e-5, atol=1e-5)
    # Rows the indices never name stay bitwise untouched.
    untouched = [row for row in range(pool.shape[0]) if row not in slots.tolist()]
    torch.testing.assert_close(pool[untouched], before[untouched], rtol=0, atol=0)
    state_elements = pool[0].numel()
    torch.testing.assert_close(storage[:, :11], storage_before[:, :11], rtol=0, atol=0)
    torch.testing.assert_close(
        storage[:, 11 + state_elements :],
        storage_before[:, 11 + state_elements :],
        rtol=0,
        atol=0,
    )


def test_recurrent_paged_fresh_slot_ignores_existing_state():
    q, k, v, gate, beta, _ = _inputs(batch=2, tokens=3, seed=9)
    _, pool = strided_state_pool(5, q.shape[2], q.shape[3], v.shape[-1])
    slots = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    has_initial_state = torch.tensor([False, False], device="cuda")

    output, _ = recurrent_kda(
        q,
        k,
        v,
        gate,
        beta,
        pool,
        state_indices=slots,
        has_initial_state=has_initial_state,
    )
    expected, expected_state = naive_recurrent_kda(
        q, k, v, gate * LOG2_E, beta, output_final_state=True
    )

    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        pool[slots.long()], expected_state.transpose(-1, -2), rtol=1e-5, atol=1e-5
    )


def test_recurrent_paged_decode_accumulates_in_place():
    """Successive single-token steps advance each slot without a round trip."""
    _, pool = strided_state_pool(5, 2, 64, 64)
    slots = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    state = pool[slots.long()].transpose(-1, -2).contiguous()

    for step in range(3):
        q, k, v, gate, beta, _ = _inputs(batch=2, tokens=1, seed=20 + step)
        output, _ = recurrent_kda(q, k, v, gate, beta, pool, state_indices=slots)
        expected, state = naive_recurrent_kda(
            q, k, v, gate * LOG2_E, beta, initial_state=state, output_final_state=True
        )
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            pool[slots.long()], state.transpose(-1, -2), rtol=1e-5, atol=1e-5
        )


@pytest.mark.parametrize("padding_slot", [0, -1])
def test_recurrent_paged_padding_slot_is_ignored(padding_slot: int):
    """vLLM padding produces zero output without advancing the reserved cache row."""
    q, k, v, gate, beta, _ = _inputs(batch=3, tokens=1, seed=30)
    _, pool = strided_state_pool(6, q.shape[2], q.shape[3], v.shape[-1])
    slots = torch.tensor([5, padding_slot, 2], device="cuda", dtype=torch.int32)
    before = pool.clone()

    output, _ = recurrent_kda(q, k, v, gate, beta, pool, state_indices=slots)

    active = torch.tensor([0, 2], device="cuda")
    expected, expected_state = naive_recurrent_kda(
        q[active],
        k[active],
        v[active],
        gate[active] * LOG2_E,
        beta[active],
        initial_state=before[slots[active].long()].transpose(-1, -2).contiguous(),
        output_final_state=True,
    )
    torch.testing.assert_close(output[active], expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(output[1], torch.zeros_like(output[1]), rtol=0, atol=0)
    torch.testing.assert_close(
        pool[slots[active].long()], expected_state.transpose(-1, -2), rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(pool[0], before[0], rtol=0, atol=0)


def test_recurrent_paged_validates_contract():
    """Reject a malformed or unsafely aliased state pool before launching."""
    q, k, v, gate, beta, _ = _inputs(batch=2, tokens=4)
    pool = torch.randn(5, q.shape[2], v.shape[-1], q.shape[3], device="cuda")
    slots = torch.tensor([4, 2], device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="requires initial_state"):
        recurrent_kda(q, k, v, gate, beta, state_indices=slots)
    with pytest.raises(ValueError, match="paged state pool must have shape"):
        recurrent_kda(q, k, v, gate, beta, pool[:, :, :-1], state_indices=slots)
    with pytest.raises(TypeError, match="use float32"):
        recurrent_kda(q, k, v, gate, beta, pool.bfloat16(), state_indices=slots)
    with pytest.raises(TypeError, match="contiguous within each"):
        recurrent_kda(q, k, v, gate, beta, pool.transpose(-1, -2), state_indices=slots)
    overlapping = torch.as_strided(
        pool,
        size=pool.shape,
        stride=(pool[0].numel() - 1, *pool.stride()[1:]),
    )
    with pytest.raises(ValueError, match="must not overlap"):
        recurrent_kda(q, k, v, gate, beta, overlapping, state_indices=slots)
    with pytest.raises(ValueError, match="state_indices must be"):
        recurrent_kda(q, k, v, gate, beta, pool, state_indices=slots.long())
    with pytest.raises(ValueError, match="state_indices must be"):
        recurrent_kda(q, k, v, gate, beta, pool, state_indices=slots[:1])
    with pytest.raises(ValueError, match="drop output_final_state"):
        recurrent_kda(q, k, v, gate, beta, pool, state_indices=slots, output_final_state=True)
    with pytest.raises(ValueError, match="requires state_indices"):
        recurrent_kda(
            q, k, v, gate, beta, has_initial_state=torch.ones_like(slots, dtype=torch.bool)
        )


def test_recurrent_validates_public_contract():
    """Reject malformed inputs at the public boundary before a kernel launch."""
    q, k, v, gate, beta, _ = _inputs(tokens=4)
    with pytest.raises(ValueError, match="k must have shape"):
        recurrent_kda(q, k[:, :-1], v, gate, beta)
    with pytest.raises(ValueError, match="gate must have shape"):
        recurrent_kda(q, k, v, gate[..., :-1], beta)
    with pytest.raises(ValueError, match="beta must have shape"):
        recurrent_kda(q, k, v, gate, beta[:, :, :-1])
    with pytest.raises(ValueError, match="initial_state must have shape"):
        recurrent_kda(q, k, v, gate, beta, q.new_zeros(1, 1, 1, 1))
    with pytest.raises(ValueError, match="batch size one"):
        recurrent_kda(
            q,
            k,
            v,
            gate,
            beta,
            cu_seqlens=cumulative_sequence_offsets([4]),
        )
    with pytest.raises(ValueError, match="num_sequences"):
        recurrent_kda(
            q[:1],
            k[:1],
            v[:1],
            gate[:1],
            beta[:1],
            cu_seqlens=torch.tensor([0], device="cuda", dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="contiguous int32"):
        recurrent_kda(
            q[:1],
            k[:1],
            v[:1],
            gate[:1],
            beta[:1],
            cu_seqlens=torch.tensor([0, 4], device="cuda", dtype=torch.int64),
        )
    with pytest.raises(ValueError, match="requires K in"):
        big = q.new_zeros(1, 4, 2, 512)
        recurrent_kda(big, big, big, big, q.new_zeros(1, 4, 2))
    with pytest.raises(TypeError, match="inputs must use one of"):
        recurrent_kda(q.double(), k.double(), v.double(), gate.double(), beta.double())


@pytest.mark.parametrize("operand", range(6))
def test_recurrent_rejects_gradient_tracking(operand: int):
    """State a clear inference-only contract for every gradient-tracking operand."""
    tensors = list(_inputs(tokens=4, initial_state=True))
    tensors[operand] = tensors[operand].float().requires_grad_()
    with pytest.raises(RuntimeError, match="inference-only"):
        recurrent_kda(*tensors)
    with torch.no_grad():
        output, _ = recurrent_kda(*tensors)
    assert not output.requires_grad


@pytest.mark.parametrize("packed", [False, True])
def test_recurrent_custom_op_registration(packed: bool):
    """Exercise the schema and fake implementation for both modes."""
    batch = 1 if packed else 2
    q, k, v, gate, beta, _ = _inputs(batch=batch, tokens=17)
    cu_seqlens = cumulative_sequence_offsets([2, 5, 10]) if packed else None
    num_sequences = 3 if packed else batch
    state = torch.randn(num_sequences, q.shape[2], q.shape[3], v.shape[-1], device="cuda")
    torch.library.opcheck(
        _recurrent_fwd_op,
        (q, k, v, gate, beta, state, cu_seqlens, True),
    )
    torch.library.opcheck(
        _recurrent_fwd_no_state_op,
        (q, k, v, gate, beta, state, cu_seqlens, True),
    )
    _, state_pool = strided_state_pool(num_sequences + 1, q.shape[2], q.shape[3], v.shape[-1])
    slots = torch.arange(1, num_sequences + 1, device="cuda", dtype=torch.int32)
    torch.library.opcheck(
        _recurrent_fwd_paged_op,
        (q, k, v, gate, beta, state_pool, slots, None, cu_seqlens),
    )


def test_recurrent_custom_op_registration_mixed_dtype():
    """Fake outputs follow Q dtype even when V uses the model activation dtype."""
    q, k, v, gate, beta, _ = _inputs(batch=2, tokens=1, dtype=torch.bfloat16)
    q, k = q.float(), k.float()
    state = torch.randn(2, q.shape[2], q.shape[3], v.shape[-1], device="cuda")
    _, state_pool = strided_state_pool(3, q.shape[2], q.shape[3], v.shape[-1])
    slots = torch.tensor([1, 2], device="cuda", dtype=torch.int32)

    torch.library.opcheck(_recurrent_fwd_op, (q, k, v, gate, beta, state, None, True))
    torch.library.opcheck(
        _recurrent_fwd_no_state_op,
        (q, k, v, gate, beta, state, None, True),
    )
    torch.library.opcheck(
        _recurrent_fwd_paged_op,
        (q, k, v, gate, beta, state_pool, slots, None, None),
    )


@pytest.mark.parametrize("output_final_state", [False, True])
@pytest.mark.parametrize("autotune", [False, True])
def test_recurrent_fullgraph_compile(output_final_state: bool, autotune: bool):
    """Compile both optional-state branches of the public operation."""
    q, k, v, gate, beta, state = _inputs(initial_state=True)

    expected, expected_state = recurrent_kda(
        q,
        k,
        v,
        gate,
        beta,
        state,
        output_final_state=output_final_state,
        autotune=autotune,
    )
    compiled = torch.compile(recurrent_kda, fullgraph=True)
    output, final_state = compiled(
        q,
        k,
        v,
        gate,
        beta,
        state,
        output_final_state=output_final_state,
        autotune=autotune,
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    if output_final_state:
        torch.testing.assert_close(final_state, expected_state, rtol=0, atol=0)
    else:
        assert final_state is None and expected_state is None


def test_recurrent_paged_mixed_dtype_fullgraph_compile():
    """Compile the public paged path used by vLLM's FP32-normalized Q/K decode."""
    q, k, v, gate, beta, _ = _inputs(batch=2, tokens=1, dtype=torch.bfloat16)
    q, k = q.float(), k.float()
    _, eager_pool = strided_state_pool(4, q.shape[2], q.shape[3], v.shape[-1])
    compiled_storage, compiled_pool = strided_state_pool(4, q.shape[2], q.shape[3], v.shape[-1])
    compiled_pool.copy_(eager_pool)
    slots = torch.tensor([3, 1], device="cuda", dtype=torch.int32)

    expected, _ = recurrent_kda(q, k, v, gate, beta, eager_pool, state_indices=slots)
    compiled = torch.compile(recurrent_kda, fullgraph=True)
    output, final_state = compiled(q, k, v, gate, beta, compiled_pool, state_indices=slots)

    assert output.dtype == q.dtype and final_state is None
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(compiled_pool, eager_pool, rtol=0, atol=0)
    assert (
        compiled_storage.untyped_storage().data_ptr() == compiled_pool.untyped_storage().data_ptr()
    )


def test_recurrent_cuda_graph_replay():
    """Replay fixed shapes with mutated boundaries, values, and history."""
    q, k, v, gate, beta, _ = _inputs(batch=1, tokens=32, seed=5)
    cu_seqlens = cumulative_sequence_offsets([11, 16, 5])
    initial_state = torch.randn(3, q.shape[2], q.shape[3], v.shape[-1], device="cuda")
    _recurrent_fwd_op(q, k, v, gate, beta, initial_state, cu_seqlens, True)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output, captured_state = _recurrent_fwd_op(
            q, k, v, gate, beta, initial_state, cu_seqlens, True
        )

    active_tokens = 23
    with torch.no_grad():
        initial_state.add_(0.25)
        cu_seqlens.copy_(cumulative_sequence_offsets([8, active_tokens - 8, 0]))
        q[:, active_tokens:].fill_(float("nan"))
        v[:, active_tokens:].fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    expected_output, expected_state = _recurrent_fwd_op(
        q, k, v, gate, beta, initial_state, cu_seqlens, True
    )
    torch.testing.assert_close(
        captured_output[:, :active_tokens],
        expected_output[:, :active_tokens],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(captured_state, expected_state, rtol=0, atol=0)


def test_recurrent_paged_cuda_graph_replay():
    """Replay paged decode with changed strided-cache routing, values, and history."""
    q, k, v, gate, beta, _ = _inputs(batch=3, tokens=1, dtype=torch.bfloat16, seed=31)
    storage, pool = strided_state_pool(7, q.shape[2], q.shape[3], v.shape[-1])
    slots = torch.tensor([5, 1, 3], device="cuda", dtype=torch.int32)
    _recurrent_fwd_paged_op(q, k, v, gate, beta, pool, slots, None, None)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = _recurrent_fwd_paged_op(q, k, v, gate, beta, pool, slots, None, None)

    with torch.no_grad():
        storage.add_(0.25)
        slots.copy_(torch.tensor([6, 0, 2], device="cuda", dtype=torch.int32))
        q.add_(0.1)
        v.mul_(0.9)
    expected_storage = storage.clone()
    state_elements = pool[0].numel()
    expected_pool = expected_storage[:, 11 : 11 + state_elements].view_as(pool)
    expected, _ = recurrent_kda(q, k, v, gate, beta, expected_pool, state_indices=slots)

    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(captured_output, expected, rtol=0, atol=0)
    torch.testing.assert_close(storage, expected_storage, rtol=0, atol=0)


def test_recurrent_launches_beyond_grid_y_limit():
    """Flat 1-D launches must survive sequence-head counts above 65,535."""
    batch, heads, head_dim = 2200, 32, 16
    assert batch * heads > 65_535
    q, k, v, gate, beta, _ = _inputs(
        batch=batch, tokens=1, heads=heads, key_dim=head_dim, value_dim=head_dim, seed=6
    )
    output, _ = recurrent_kda(q, k, v, gate, beta)
    expected, _ = naive_recurrent_kda(q, k, v, gate * LOG2_E, beta)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


def test_recurrent_paged_empty_sequences_initialize_fresh_slots():
    """Empty packed sequences zero freshly assigned slots and preserve resumed ones."""
    q, k, v, gate, beta, _ = _inputs(batch=1, tokens=4, seed=9)
    cu_seqlens = cumulative_sequence_offsets([0, 4, 0])
    _, pool = strided_state_pool(6, q.shape[2], q.shape[3], v.shape[-1])
    original_pool = pool.clone()
    slots = torch.tensor([2, 3, 5], device="cuda", dtype=torch.int32)
    has_initial_state = torch.tensor([False, False, True], device="cuda")

    recurrent_kda(
        q,
        k,
        v,
        gate,
        beta,
        pool,
        cu_seqlens=cu_seqlens,
        state_indices=slots,
        has_initial_state=has_initial_state,
    )

    # The empty fresh sequence must initialize its slot to the zero state.
    torch.testing.assert_close(pool[2], torch.zeros_like(pool[2]), rtol=0, atol=0)
    # The empty resumed sequence and unselected slots keep their contents.
    preserved = [0, 1, 4, 5]
    torch.testing.assert_close(pool[preserved], original_pool[preserved], rtol=0, atol=0)


def test_naive_recurrent_packed_matches_public_contract():
    """The pure reference honors packed semantics: empty slots and capacity tails."""
    q, k, v, gate, beta, _ = _inputs(batch=1, tokens=32, seed=7)
    cu_seqlens = cumulative_sequence_offsets([0, 11, 16, 0])
    initial_state = torch.randn(4, q.shape[2], q.shape[3], v.shape[-1], device="cuda")

    output, final_state = naive_recurrent_kda(
        q,
        k,
        v,
        gate * LOG2_E,
        beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    assert output.shape == v.shape and final_state is not None
    torch.testing.assert_close(final_state[0], initial_state[0], rtol=0, atol=0)
    torch.testing.assert_close(final_state[3], initial_state[3], rtol=0, atol=0)
    torch.testing.assert_close(output[:, 27:], torch.zeros_like(output[:, 27:]), rtol=0, atol=0)
    fused, fused_state = recurrent_kda(
        q, k, v, gate, beta, initial_state, cu_seqlens=cu_seqlens, output_final_state=True
    )
    torch.testing.assert_close(output[:, :27], fused[:, :27], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(final_state, fused_state, rtol=1e-5, atol=1e-5)
