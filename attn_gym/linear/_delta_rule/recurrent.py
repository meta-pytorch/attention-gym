# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused recurrent scan shared by scalar- and vector-gated delta rules.

Vector gates are per-key-channel log2 decays; scalar gates are per-token natural-log decays.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from attn_gym._backends.triton.utils import ptr_offset


@triton.jit
def recurrent_delta_rule_fwd_kernel(
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
    SCALAR_LN_GATE: tl.constexpr,
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
            (i_state, i_h, o_k[:, None], o_v[None, :]),
            (state_batch_stride, K * V, V, 1),
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
        if SCALAR_LN_GATE:
            b_g = tl.load(gate + row).to(tl.float32)
        else:
            b_g = tl.load(gate + row * K + o_k, mask=m_k, other=0.0).to(tl.float32)
        b_beta = tl.load(beta + row).to(tl.float32)
        b_v = tl.load(v + row * V + o_v, mask=m_v, other=0.0).to(tl.float32)

        if SCALAR_LN_GATE:
            b_state *= tl.exp(b_g)
        else:
            b_state *= tl.exp2(b_g)[:, None]

        b_delta = (b_v - tl.sum(b_k[:, None] * b_state, 0)) * b_beta
        b_state += b_k[:, None] * b_delta[None, :]
        b_o = tl.sum(b_q[:, None] * b_state, 0)
        tl.store(output + row * V + o_v, b_o.to(output.dtype.element_ty), mask=m_v)

    if STORE_FINAL_STATE:
        tl.store(ht + p_state, b_state, mask=m_kv)


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
    scalar_ln_gate: bool,
    store_final_state: bool,
    state_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Launch a scalar-ln or vector-log2 recurrent delta-rule specialization."""
    assert k.shape == q.shape
    assert v.shape[:3] == q.shape[:3]
    assert gate.shape == (q.shape[:-1] if scalar_ln_gate else q.shape)
    assert beta.shape == q.shape[:-1]
    assert all(tensor.is_contiguous() for tensor in (q, k, v, gate, beta))

    batch, tokens, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if initial_state is not None:
        assert initial_state.stride()[1:] == (key_dim * value_dim, value_dim, 1)
        if state_indices is None:
            assert initial_state.is_contiguous()
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
    # BV=32 measured faster than 64 on B200 for both decode and prefill
    # (smaller state tiles schedule better; state traffic is identical).
    block_v = min(triton.next_power_of_2(value_dim), 32)
    # One flat launch dimension: sequence-head counts can exceed the 65,535
    # grid-Y limit, while grid-X is effectively unbounded.
    grid = (triton.cdiv(value_dim, block_v) * num_sequences * heads,)
    recurrent_delta_rule_fwd_kernel[grid](
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
        state_batch_stride=state_batch_stride,
        H=heads,
        K=key_dim,
        V=value_dim,
        BK=triton.next_power_of_2(key_dim),
        BV=block_v,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=final_state is not None,
        IS_VARLEN=cu_seqlens is not None,
        USE_STATE_INDICES=state_indices is not None,
        SCALAR_LN_GATE=scalar_ln_gate,
        num_warps=4,
    )
    return output, final_state


__all__ = ["launch_recurrent_delta_rule_fwd"]
