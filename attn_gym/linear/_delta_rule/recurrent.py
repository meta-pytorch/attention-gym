# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused recurrent scan shared by scalar- and vector-gated delta rules.

Per token, the ``[K, V]`` state decays, a beta-scaled delta rule writes the new value, and the
query reads the updated state:

    state  = decay_t * state                    # GateKind.SCALAR: exp(g_t) * state
                                                # GateKind.VECTOR: diag(exp(g_t)) @ state
    delta  = beta_t * (v_t - k_t @ state)
    state += outer(k_t, delta)
    out_t  = scale * q_t @ state

Gates are natural-log decays end to end; the kernel folds the log2 conversion into the ``exp2``
argument in registers, so no caller pays a separate conversion pass. The FP32 state stays in
registers per (sequence, head, value tile). Forward-only; training uses the chunked forms.
"""

from __future__ import annotations

from enum import Enum

import torch
import triton
import triton.language as tl

from attn_gym._backends.triton.utils import (
    LOG2_E,
    autotune_cache_kwargs,
    configure_triton_allocator,
    ptr_offset,
)
from attn_gym.linear._delta_rule.triton.paged_state import resolve_paged_state

configure_triton_allocator()


# Autotune winners at a fixed sequence count were stable from 64 tokens per sequence upward
# (GB200, H=16, K=V=128) and differed only among shorter scans.
_TOKENS_BUCKET_LIMIT = 64


class GateKind(Enum):
    """Gate layout consumed by the scan."""

    SCALAR = "scalar"  # one natural-log decay per token, shaped [B, T, H]
    VECTOR = "vector"  # one natural-log decay per key channel, shaped [B, T, H, K]


def _prune_recurrent_configs(configs, _named_args, V, **_):
    """Drop value tiles wider than the padded value dimension."""
    max_block_v = max(8, triton.next_power_of_2(V))
    return [config for config in configs if config.kwargs["BV"] <= max_block_v]


@triton.jit
def _load_key_tile(base, o_k_tile, K: tl.constexpr, BK: tl.constexpr, valid):
    """Load an FP32 key tile, masking padding and invalid tokens."""
    if K == BK:
        tile = tl.load(base + o_k_tile, mask=valid, other=0.0)
    else:
        tile = tl.load(base + o_k_tile, mask=(o_k_tile < K) & valid, other=0.0)
    return tile.to(tl.float32)


@triton.jit
def _load_token(
    q,
    k,
    v,
    gate,
    beta,
    row,
    row_k,
    o_k_tile,
    o_v,
    m_v,
    valid,
    K: tl.constexpr,
    BK: tl.constexpr,
    V: tl.constexpr,
    SCALAR_GATE: tl.constexpr,
):
    """Load a token's inputs; `valid=False` returns zeros without reading memory."""
    b_q = _load_key_tile(q + row_k * K, o_k_tile, K, BK, valid)
    b_k = _load_key_tile(k + row_k * K, o_k_tile, K, BK, valid)
    if SCALAR_GATE:
        b_g = tl.load(gate + row, mask=valid, other=0.0).to(tl.float32)
    else:
        b_g = _load_key_tile(gate + row * K, o_k_tile, K, BK, valid)
    b_beta = tl.load(beta + row, mask=valid, other=0.0).to(tl.float32)
    b_v = tl.load(v + row * V + o_v, mask=m_v & valid, other=0.0).to(tl.float32)
    return b_q, b_k, b_g, b_beta, b_v


@triton.jit(do_not_specialize=["T", "N", "tokens_bucket"])
def _recurrent_delta_rule_fwd_kernel(
    q,
    k,
    v,
    gate,
    beta,
    output,
    h0,
    ht,
    cu_seqlens,
    state_indices,
    has_initial_state,
    scale,
    T,
    N,
    tokens_bucket,
    state_batch_stride: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_STATE_INDICES: tl.constexpr,
    USE_HAS_INITIAL_STATE: tl.constexpr,
    SCALAR_GATE: tl.constexpr,
    PREFETCH: tl.constexpr,
):
    """Scan one sequence/head/value tile while retaining the FP32 state in registers.

    Programs are indexed by value head. With multi-value attention (MVA, ``HK < H``) each
    block of ``H // HK`` consecutive value heads reads one shared q/k head, while gates,
    beta, and state stay per value head: grouping shrinks the q/k projections, and each
    value head's state remains independent only if its decay is.
    """
    pid = tl.program_id(0).to(tl.int64)
    NV = tl.cdiv(V, BV)
    i_v = pid % NV
    i_nh = pid // NV
    i_n, i_h = i_nh // H, i_nh % H
    i_hk = i_h // (H // HK)

    if IS_VARLEN:
        # Assumption: seqlens have been validated prior to call
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    else:
        bos = i_n * T
        eos = bos + T

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_kv = m_k[:, None] & m_v[None, :]

    # Empty sequences still take the load/store path so a fresh slot is initialized.
    state_row, state_active, load_initial = resolve_paged_state(
        i_n,
        state_indices,
        has_initial_state,
        USE_STATE_INDICES,
        USE_HAS_INITIAL_STATE,
    )
    if USE_STATE_INDICES and not state_active:
        for t in range(bos, eos):
            row = t * H + i_h
            tl.store(output + ptr_offset((row, o_v), (V, 1)), 0.0, mask=m_v)
        return
    p_state = ptr_offset(
        (state_row, i_h, o_v[None, :], o_k[:, None]),
        (state_batch_stride, V * K, K, 1),
    )
    if USE_INITIAL_STATE:
        b_state = tl.load(h0 + p_state, mask=m_kv & load_initial, other=0.0).to(tl.float32)
    else:
        b_state = tl.zeros([BK, BV], dtype=tl.float32)

    # Load key-indexed operands as [BK, BV] tiles in the state's layout; broadcasting [BK]
    # vectors costs a layout conversion per operand per token.
    o_k_tile = tl.broadcast_to(o_k[:, None], (BK, BV))
    # A 32-bit trip count and scalar row offsets keep 64-bit math out of per-element addresses.
    num_tokens = (eos - bos).to(tl.int32)
    if PREFETCH:
        b_next = _load_token(
            q, k, v, gate, beta, bos * H + i_h, bos * HK + i_hk, o_k_tile, o_v, m_v,
            num_tokens > 0, K, BK, V, SCALAR_GATE,
        )  # fmt: skip
    for i_t in range(num_tokens):
        t = bos + i_t
        row = t * H + i_h
        row_k = t * HK + i_hk
        if PREFETCH:
            # Load the next token before computing this one.
            b_q, b_k, b_g, b_beta, b_v = b_next
            b_next = _load_token(
                q, k, v, gate, beta, row + H, row_k + HK, o_k_tile, o_v, m_v,
                i_t + 1 < num_tokens, K, BK, V, SCALAR_GATE,
            )  # fmt: skip
        else:
            b_q, b_k, b_g, b_beta, b_v = _load_token(
                q, k, v, gate, beta, row, row_k, o_k_tile, o_v, m_v, True, K, BK, V,
                SCALAR_GATE,
            )  # fmt: skip
        b_q *= scale

        # exp(g) computed as exp2(g * log2(e)); the FMA folds into the exp2 argument.
        b_state *= tl.exp2(b_g * LOG2_E)

        if PREFETCH:
            # q @ (S + k delta) = q @ S + (q . k) delta lets both reductions read decayed S
            # without waiting for the state update.
            b_o = tl.sum(b_q * b_state, 0)
            b_delta = (b_v - tl.sum(b_k * b_state, 0)) * b_beta
            b_o += tl.sum(b_q * b_k, 0) * b_delta
            b_state += b_k * b_delta[None, :]
        else:
            b_delta = (b_v - tl.sum(b_k * b_state, 0)) * b_beta
            b_state += b_k * b_delta[None, :]
            b_o = tl.sum(b_q * b_state, 0)
        tl.store(output + row * V + o_v, b_o.to(output.dtype.element_ty), mask=m_v)

    if STORE_FINAL_STATE:
        tl.store(ht + p_state, b_state, mask=m_kv)


recurrent_delta_rule_fwd_kernel = triton.autotune(
    configs=[
        triton.Config({"BV": block_v}, num_warps=num_warps, num_stages=3)
        for block_v in (32, 16, 8, 4, 2)
        for num_warps in (1, 2, 4)
    ],
    # The best tile follows the grid fill (``N * H`` programs) and, for scans shorter than
    # ``_TOKENS_BUCKET_LIMIT`` tokens, the scan length; ``tokens_bucket`` collapses the exact
    # token count to that information so new lengths do not re-benchmark.
    key=[
        "tokens_bucket",
        "N",
        "H",
        "HK",
        "K",
        "V",
        "USE_INITIAL_STATE",
        "STORE_FINAL_STATE",
        "IS_VARLEN",
        "SCALAR_GATE",
        "PREFETCH",
    ],
    prune_configs_by={"early_config_prune": _prune_recurrent_configs},
    **autotune_cache_kwargs,
)(_recurrent_delta_rule_fwd_kernel)


def launch_recurrent_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    *,
    scale: float,
    gate_kind: GateKind,
    store_final_state: bool,
    state_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    autotune: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Launch a scalar- or vector-gated recurrent delta-rule specialization.

    Args:
        q: Queries shaped ``[B, T, HK, K]``; packed callers pass ``B == 1``. ``HK`` may be
            a divisor of the value head count ``H`` (grouped heads): each block of
            ``H // HK`` consecutive value heads shares one query/key head, so callers
            never materialize a ``repeat_interleave``. Gates stay per value head.
        k: Keys shaped like ``q``.
        v: Values shaped ``[B, T, H, V]``.
        gate: Natural-log decay per value head, shaped ``[B, T, H]`` for
            ``GateKind.SCALAR`` or ``[B, T, H, K]`` for ``GateKind.VECTOR``.
        beta: Per-token write gate shaped ``[B, T, H]``.
        initial_state: Optional FP32 starting state shaped ``[N, H, V, K]``, or the mutable
            FP32/BF16 ``[slots, H, V, K]`` pool that ``state_indices`` addresses in paged mode.
            Recurrence math remains FP32.
        cu_seqlens: Optional packed boundaries shaped ``[N + 1]``; offsets are trusted and
            must be validated by the caller before launch.
        scale: Multiplier applied to ``q`` before each state read.
        gate_kind: Gate layout consumed by the scan.
        store_final_state: Allocate and return the post-scan state; implied and mandatory in
            paged mode, where the pool itself is advanced in place.
        state_indices: Optional per-sequence slot indices enabling paged mode. Active
            sequences must select distinct, positive, in-bounds slots; nonpositive entries
            are padding that produce zero output and leave the pool untouched.
        has_initial_state: Optional per-sequence booleans qualifying paged slots. Pool
            allocators hand out slots without zeroing them, so a False entry marks garbage
            contents: the scan starts from zero and overwrites the slot, including for empty
            sequences, so the slot is initialized before a later decode reads it. Without
            this mask every selected slot is treated as real history.
        autotune: Benchmark tile configurations when true; paged launches always use fixed
            heuristics because rerunning candidates would advance the pool repeatedly.

    Returns:
        The output in ``q.dtype`` plus the final state (``None`` unless requested). In paged
        mode the returned state aliases ``initial_state``.
    """
    assert k.shape == q.shape
    assert v.shape[:2] == q.shape[:2]
    assert gate.shape == (
        v.shape[:-1] if gate_kind is GateKind.SCALAR else (*v.shape[:-1], q.shape[-1])
    )
    assert beta.shape == v.shape[:-1]
    assert all(tensor.is_contiguous() for tensor in (q, k, v, gate, beta))
    assert has_initial_state is None or state_indices is not None

    batch, tokens, key_heads, key_dim = q.shape
    heads, value_dim = v.shape[2], v.shape[-1]
    assert heads % key_heads == 0
    if initial_state is not None:
        assert initial_state.stride()[1:] == (value_dim * key_dim, key_dim, 1)
    num_sequences = batch if cu_seqlens is None else cu_seqlens.shape[0] - 1
    output = torch.empty_like(v, dtype=q.dtype)
    if state_indices is not None:
        # Paged mode reads and writes one slot, so `h0` and `ht` alias the pool and
        # the scan needs no gather/scatter around it.
        assert store_final_state and initial_state is not None
        final_state = initial_state
    elif store_final_state:
        final_state = q.new_empty(num_sequences, heads, value_dim, key_dim, dtype=torch.float32)
    else:
        final_state = None
    state_batch_stride = (
        initial_state.stride(0) if initial_state is not None else heads * key_dim * value_dim
    )
    # One flat launch dimension avoids the 65,535 grid-Y limit for large batches.
    grid = lambda meta: (triton.cdiv(value_dim, meta["BV"]) * num_sequences * heads,)

    sequence_heads = num_sequences * heads
    average_tokens = tokens if cu_seqlens is None else triton.cdiv(tokens, num_sequences)
    tokens_bucket = min(triton.next_power_of_2(average_tokens), _TOKENS_BUCKET_LIMIT)
    # Up to 16 sequence-heads, prefetch hides load latency (GB200: 0.6-0.8x at K=64 and K=128).
    # Prefetch was slower for K > 128.
    prefetch = average_tokens > 1 and sequence_heads <= 16 and key_dim <= 128
    use_autotune = autotune and state_indices is None
    kernel = recurrent_delta_rule_fwd_kernel if use_autotune else _recurrent_delta_rule_fwd_kernel
    launch_options = {}
    if not use_autotune:
        # B200 sweeps set the small-value one-token tiles; GB200 sweeps set the rest.
        # Multi-token scans use one warp so per-token reductions stay in registers. Each program
        # carries its tile through the whole sequence, so the grid should fit one wave: tiny
        # grids split V finely, and BV=16 under the 128-register cap fits 256 sequence-heads at
        # K=128. Skewed varlen lengths are bound by the longest sequence, so varlen keeps BV=4.
        if average_tokens > 1 and key_dim >= 128:
            num_warps = 1
            if sequence_heads <= 16:
                block_v_limit = 2
            elif cu_seqlens is not None or sequence_heads <= 64:
                block_v_limit = 4
            else:
                # BV * K <= 2048 fits the 128-register cap.
                block_v_limit = 16 if key_dim <= 128 else 8
        elif average_tokens > 1:
            # Narrow keys leave little per-token work, so wide tiles win once the grid is full.
            if sequence_heads >= 2048:
                block_v_limit, num_warps = 32, 2
            else:
                block_v_limit, num_warps = (4 if sequence_heads <= 256 else 8), 1
        elif value_dim >= 128:
            block_v_limit = 4 if key_dim > 128 or sequence_heads > 512 else 8
            num_warps = 1
        elif sequence_heads <= 8:
            block_v_limit, num_warps = 8, 4
        elif sequence_heads < 1024:
            block_v_limit, num_warps = 16, 2
        else:
            block_v_limit, num_warps = 32, 2
        launch_options = {
            "BV": min(triton.next_power_of_2(value_dim), block_v_limit),
            "num_warps": num_warps,
            "num_stages": 3,
        }
        if num_warps == 1:
            launch_options["maxnreg"] = 128
    kernel[grid](
        q,
        k,
        v,
        gate,
        beta,
        output,
        initial_state,
        final_state,
        cu_seqlens,
        state_indices,
        has_initial_state,
        scale=scale,
        T=tokens,
        N=num_sequences,
        tokens_bucket=tokens_bucket,
        state_batch_stride=state_batch_stride,
        H=heads,
        HK=key_heads,
        K=key_dim,
        V=value_dim,
        BK=triton.next_power_of_2(key_dim),
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=final_state is not None,
        IS_VARLEN=cu_seqlens is not None,
        USE_STATE_INDICES=state_indices is not None,
        USE_HAS_INITIAL_STATE=has_initial_state is not None,
        SCALAR_GATE=gate_kind is GateKind.SCALAR,
        PREFETCH=prefetch,
        **launch_options,
    )
    return output, final_state


__all__ = ["GateKind", "launch_recurrent_delta_rule_fwd"]
