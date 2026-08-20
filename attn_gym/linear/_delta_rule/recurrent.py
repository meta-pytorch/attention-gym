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

configure_triton_allocator()


class GateKind(Enum):
    """Gate layout consumed by the scan."""

    SCALAR = "scalar"  # one natural-log decay per token, shaped [B, T, H]
    VECTOR = "vector"  # one natural-log decay per key channel, shaped [B, T, H, K]


def _prune_recurrent_configs(configs, _named_args, V, **_):
    """Drop value tiles wider than the padded value dimension."""
    max_block_v = max(8, triton.next_power_of_2(V))
    return [config for config in configs if config.kwargs["BV"] <= max_block_v]


@triton.jit(do_not_specialize=["N"])
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
    scale,
    T,
    N,
    state_batch_stride: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_STATE_INDICES: tl.constexpr,
    SCALAR_GATE: tl.constexpr,
):
    """Scan one sequence/head/value tile while retaining the FP32 state in registers."""
    pid = tl.program_id(0).to(tl.int64)
    NV = tl.cdiv(V, BV)
    i_v = pid % NV
    i_nh = pid // NV
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        # Assumption: seqlens have been validated prior to call
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    else:
        bos = i_n * T
        eos = bos + T

    if USE_STATE_INDICES and bos == eos:
        return

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_kv = m_k[:, None] & m_v[None, :]

    # Paged mode addresses the state pool by slot instead of by sequence position. Active
    # sequences must select distinct, positive, in-bounds slots; nonpositive slots are padding.
    if USE_STATE_INDICES:
        i_state = tl.load(state_indices + i_n).to(tl.int64)
        if i_state <= 0:
            for t in range(bos, eos):
                row = t * H + i_h
                tl.store(output + ptr_offset((row, o_v), (V, 1)), 0.0, mask=m_v)
            return
        p_state = ptr_offset(
            (i_state, i_h, o_v[None, :], o_k[:, None]),
            (state_batch_stride, V * K, K, 1),
        )
    else:
        p_state = ptr_offset(
            (i_n, i_h, o_k[:, None], o_v[None, :]),
            (state_batch_stride, K * V, V, 1),
        )
    if USE_INITIAL_STATE:
        b_state = tl.load(h0 + p_state, mask=m_kv, other=0.0).to(tl.float32)
    else:
        b_state = tl.zeros([BK, BV], dtype=tl.float32)

    for t in range(bos, eos):
        row = t * H + i_h
        b_q = tl.load(q + row * K + o_k, mask=m_k, other=0.0).to(tl.float32) * scale
        b_k = tl.load(k + row * K + o_k, mask=m_k, other=0.0).to(tl.float32)
        if SCALAR_GATE:
            b_g = tl.load(gate + row).to(tl.float32)
        else:
            b_g = tl.load(gate + row * K + o_k, mask=m_k, other=0.0).to(tl.float32)
        b_beta = tl.load(beta + row).to(tl.float32)
        b_v = tl.load(v + row * V + o_v, mask=m_v, other=0.0).to(tl.float32)

        # exp(g) computed as exp2(g * log2(e)); the FMA folds into the exp2 argument.
        if SCALAR_GATE:
            b_state *= tl.exp2(b_g * LOG2_E)
        else:
            b_state *= tl.exp2(b_g * LOG2_E)[:, None]

        b_delta = (b_v - tl.sum(b_k[:, None] * b_state, 0)) * b_beta
        b_state += b_k[:, None] * b_delta[None, :]
        b_o = tl.sum(b_q[:, None] * b_state, 0)
        tl.store(output + row * V + o_v, b_o.to(output.dtype.element_ty), mask=m_v)

    if STORE_FINAL_STATE:
        tl.store(ht + p_state, b_state, mask=m_kv)


recurrent_delta_rule_fwd_kernel = triton.autotune(
    configs=[
        triton.Config({"BV": block_v}, num_warps=num_warps, num_stages=3)
        for block_v in (32, 16, 8)
        for num_warps in (2, 4)
    ],
    key=[
        "T",
        "N",
        "H",
        "K",
        "V",
        "USE_INITIAL_STATE",
        "STORE_FINAL_STATE",
        "IS_VARLEN",
        "SCALAR_GATE",
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
    autotune: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Launch a scalar- or vector-gated recurrent delta-rule specialization."""
    assert k.shape == q.shape
    assert v.shape[:3] == q.shape[:3]
    assert gate.shape == (q.shape[:-1] if gate_kind is GateKind.SCALAR else q.shape)
    assert beta.shape == q.shape[:-1]
    assert all(tensor.is_contiguous() for tensor in (q, k, v, gate, beta))

    batch, tokens, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if initial_state is not None:
        if state_indices is None:
            assert initial_state.is_contiguous()
        else:
            assert initial_state.stride()[1:] == (value_dim * key_dim, key_dim, 1)
    num_sequences = batch if cu_seqlens is None else cu_seqlens.shape[0] - 1
    output = torch.empty_like(v, dtype=q.dtype)
    if state_indices is not None:
        # Paged mode reads and writes one slot, so `h0` and `ht` alias the pool and
        # the scan needs no gather/scatter around it.
        assert store_final_state and initial_state is not None
        final_state = initial_state
    elif store_final_state:
        final_state = q.new_empty(num_sequences, heads, key_dim, value_dim, dtype=torch.float32)
    else:
        final_state = None
    state_batch_stride = (
        initial_state.stride(0) if initial_state is not None else heads * key_dim * value_dim
    )
    # One flat launch dimension avoids the 65,535 grid-Y limit for large batches.
    grid = lambda meta: (triton.cdiv(value_dim, meta["BV"]) * num_sequences * heads,)

    use_autotune = autotune and state_indices is None
    kernel = recurrent_delta_rule_fwd_kernel if use_autotune else _recurrent_delta_rule_fwd_kernel
    launch_options = {}
    if not use_autotune:
        # Offline B200 sweeps favor small tiles for long scans and low-occupancy decode.
        average_tokens = tokens if cu_seqlens is None else triton.cdiv(tokens, num_sequences)
        sequence_heads = num_sequences * heads
        if average_tokens >= 32:
            block_v_limit, num_warps = 8, 4
        elif average_tokens > 1:
            block_v_limit, num_warps = 32, 2
        elif value_dim >= 128:
            block_v_limit, num_warps = 16, 4
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
        scale=scale,
        T=tokens,
        N=num_sequences,
        state_batch_stride=state_batch_stride,
        H=heads,
        K=key_dim,
        V=value_dim,
        BK=triton.next_power_of_2(key_dim),
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=final_state is not None,
        IS_VARLEN=cu_seqlens is not None,
        USE_STATE_INDICES=state_indices is not None,
        SCALAR_GATE=gate_kind is GateKind.SCALAR,
        **launch_options,
    )
    return output, final_state


__all__ = ["GateKind", "launch_recurrent_delta_rule_fwd"]
