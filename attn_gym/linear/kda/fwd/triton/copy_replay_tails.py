# SPDX-License-Identifier: BSD-3-Clause

"""Copy packed KDA inputs into checkpoint slots before prefill advances the source counts."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from attn_gym.linear._delta_rule.constants import DEFAULT_CHUNK_SIZE
from attn_gym.linear._delta_rule.paged_state import validate_has_initial_state
from attn_gym.linear._delta_rule.triton.chunk_scheduler import assert_packed_offsets
from attn_gym.linear._delta_rule.validation import require_inference_only
from attn_gym.linear.kda.fwd.triton.paged_replay import _validate_replay_aliases
from attn_gym.linear.kda.validation import validate_kda_inputs, validate_replay_state
from attn_gym.linear.types import ReplayState


# Triton specializes integers inside tuples even under do_not_specialize. Any input stride
# can scale with packed T (e.g. a token-innermost view), so pass them as top-level scalars.
@triton.jit(
    debug=True,  # The first reader must enforce the packed-offset contract.
    do_not_specialize=[
        "tokens",
        "num_slots",
        "q_t",
        "q_h",
        "q_d",
        "k_t",
        "k_h",
        "k_d",
        "v_t",
        "v_h",
        "v_d",
        "gate_t",
        "gate_h",
        "gate_d",
        "beta_t",
        "beta_h",
    ],
)
def _copy_replay_tails_kernel(
    inputs,
    caches,
    counts,
    offsets,
    sources,
    has_initial_state,
    positions,
    destinations,
    tokens,
    num_slots,
    q_t,
    q_h,
    q_d,
    k_t,
    k_h,
    k_d,
    v_t,
    v_h,
    v_d,
    gate_t,
    gate_h,
    gate_d,
    beta_t,
    beta_h,
    SLOT_STRIDES: tl.constexpr,
    COUNT_STRIDE: tl.constexpr,
    HEADS: tl.constexpr,
    DIMS: tl.constexpr,
    FEATURE_TILES: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    sequence = (tl.program_id(0) // FEATURE_TILES).to(tl.int64)
    begin = tl.load(offsets + sequence).to(tl.int64)
    end = tl.load(offsets + sequence + 1).to(tl.int64)
    assert_packed_offsets(begin, end, sequence, tokens, None)
    source = tl.load(sources + sequence).to(tl.int64)
    slot = tl.load(destinations + sequence).to(tl.int64)
    if (source <= 0) | (source >= num_slots) | (slot <= 0) | (slot >= num_slots):
        return
    position = tl.load(positions + sequence).to(tl.int64)
    # A neighboring offset pair can fail its assert concurrently with this tile.
    if (begin < 0) | (position <= 0) | (position > end - begin):
        return

    resume = True
    if has_initial_state is not None:
        resume = tl.load(has_initial_state + sequence)
    count = tl.load(counts + source * COUNT_STRIDE, mask=resume, other=0)
    logical = count + position
    if (count < 0) | (count >= CHUNK) | (logical < CHUNK):
        return
    tail = logical % CHUNK
    feature_tile = tl.program_id(0) % FEATURE_TILES
    row_tile = tl.program_id(1)
    if (feature_tile == 0) & (row_tile == 0):
        tl.store(counts + slot * COUNT_STRIDE, tail)
    if row_tile * BLOCK_T >= tail:
        return

    # Once this call completed a boundary, all rows after it come from the new inputs.
    row = row_tile * BLOCK_T + tl.arange(0, BLOCK_T)
    token = begin + position - tail + row
    feature = feature_tile.to(tl.int64) * BLOCK_F + tl.arange(0, BLOCK_F)
    input_strides = (
        (q_t, q_h, q_d),
        (k_t, k_h, k_d),
        (v_t, v_h, v_d),
        (gate_t, gate_h, gate_d),
        (beta_t, beta_h, 0),
    )
    for i in tl.static_range(5):
        width = HEADS * DIMS[i]
        head = feature // DIMS[i]
        dim = feature % DIMS[i]
        inner = head * input_strides[i][1] + dim * input_strides[i][2]
        src = token[:, None] * input_strides[i][0] + inner[None, :]
        dst = slot * SLOT_STRIDES[i] + row[:, None].to(tl.int64) * width + feature[None, :]
        mask = (row[:, None] < tail) & (feature[None, :] < width)
        tl.store(caches[i] + dst, tl.load(inputs[i] + src, mask=mask, other=0), mask=mask)


def copy_replay_tails_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    replay_q: torch.Tensor,
    replay_k: torch.Tensor,
    replay_v: torch.Tensor,
    replay_gate: torch.Tensor,
    replay_beta: torch.Tensor,
    replay_count: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
    checkpoint_positions: torch.Tensor,
    checkpoint_slots: torch.Tensor,
) -> None:
    """Copy tails in one launch. Written slots must be distinct and not routed as sources."""
    if not q.is_cuda:
        raise ValueError("copy_replay_tails requires CUDA inputs")
    inputs = (q, k, v, gate, beta)
    replay = ReplayState(replay_q, replay_k, replay_v, replay_gate, replay_beta, replay_count)
    validate_kda_inputs(*inputs, None, cu_seqlens, op_name="copy_replay_tails", gate_name="gate")
    if not replay_count.ndim:
        raise ValueError("replay_state count tensor requires a slot dimension")
    num_slots = replay_count.shape[0]
    validate_replay_state(replay, q, v, num_slots)
    num_sequences = cu_seqlens.shape[0] - 1
    routes = (
        ("state_indices", state_indices),
        ("checkpoint_positions", checkpoint_positions),
        ("checkpoint_slots", checkpoint_slots),
    )
    for name, tensor in routes:
        if (
            tensor.shape != (num_sequences,)
            or tensor.dtype != torch.int32
            or tensor.device != q.device
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous int32 [N] on the input device")
    validate_has_initial_state(has_initial_state, num_sequences, q.device)
    require_inference_only((*inputs, *replay), "copy_replay_tails is inference-only")
    _validate_replay_aliases(
        tuple(replay._asdict().items()),
        None,
        (
            *zip(ReplayState._fields[:-1], inputs, strict=True),
            ("cu_seqlens", cu_seqlens),
            ("has_initial_state", has_initial_state),
            *routes,
        ),
    )

    _, tokens, heads, key_dim = q.shape
    dims = (key_dim, key_dim, v.shape[3], key_dim, 1)
    block_t, block_f = 8, 64
    feature_tiles = triton.cdiv(heads * max(dims), block_f)
    grid = (num_sequences * feature_tiles, triton.cdiv(DEFAULT_CHUNK_SIZE, block_t))
    _copy_replay_tails_kernel[grid](
        inputs,
        replay[:-1],
        replay_count,
        cu_seqlens,
        state_indices,
        has_initial_state,
        checkpoint_positions,
        checkpoint_slots,
        tokens,
        num_slots,
        *(stride for t in inputs for stride in t.stride()[1:]),
        tuple(t.stride(0) for t in replay[:-1]),
        replay_count.stride(0),
        heads,
        dims,
        feature_tiles,
        DEFAULT_CHUNK_SIZE,
        block_t,
        block_f,
    )
