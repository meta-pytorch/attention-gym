# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused O(T) KDA recurrence for decode and inference prefill.

The kernel scans tokens sequentially per (sequence, head, value block), holding the
FP32 recurrent state in registers. It mirrors :func:`naive_recurrent_kda` exactly:
per step the state decays by ``exp2(gate)`` per key channel, a beta-scaled delta
writes the new value, and the query reads the updated state. The operation is
inference-only; use ``chunk_kda`` for training.
"""

from __future__ import annotations

import threading

import torch
import triton
import triton.language as tl

from attn_gym.linear.kda.ops import recurrent_forward as forward
from attn_gym.linear.kda.ops import (
    recurrent_fwd_no_state_op as _recurrent_fwd_no_state_op,
)
from attn_gym.linear.kda.ops import recurrent_fwd_op as _recurrent_fwd_op
from attn_gym.linear.kda.ops import (
    recurrent_fwd_paged_op as _recurrent_fwd_paged_op,
)
from attn_gym.linear.kda.utils import autotune_cache_kwargs


def _prune_recurrent_configs(configs, _named_args, V, **_):
    """drop value tiles wider than the padded value dimension"""
    max_block_v = max(8, triton.next_power_of_2(V))
    return [config for config in configs if config.kwargs["BV"] <= max_block_v]


@triton.autotune(
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
        "USE_STATE_INDICES",
    ],
    prune_configs_by={"early_config_prune": _prune_recurrent_configs},
    **autotune_cache_kwargs,
)
@triton.jit
def kda_recurrent_fwd_kernel(
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
    N: tl.constexpr,
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
    CONTIGUOUS_FINAL_STATE: tl.constexpr,
):
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

    if USE_STATE_INDICES:
        i_state = tl.load(state_indices + i_n).to(tl.int64)
        # ignore vllm padded entries
        if i_state <= 0:
            for t in range(bos, eos):
                row = t * H + i_h
                tl.store(output + row * V + o_v, 0.0, mask=m_v)
            return
    else:
        i_state = i_n
    p_state = i_state * state_batch_stride + i_h * K * V + o_k[:, None] * V + o_v[None, :]
    if USE_INITIAL_STATE:
        b_state = tl.load(h0 + p_state, mask=m_kv, other=0.0).to(tl.float32)
    else:
        b_state = tl.zeros([BK, BV], dtype=tl.float32)

    for t in range(bos, eos):
        row = t * H + i_h
        b_q = tl.load(q + row * K + o_k, mask=m_k, other=0.0).to(tl.float32) * scale
        b_k = tl.load(k + row * K + o_k, mask=m_k, other=0.0).to(tl.float32)
        b_g = tl.load(gate + row * K + o_k, mask=m_k, other=0.0).to(tl.float32)
        b_beta = tl.load(beta + row).to(tl.float32)
        b_v = tl.load(v + row * V + o_v, mask=m_v, other=0.0).to(tl.float32)

        b_state *= tl.exp2(b_g)[:, None]
        b_delta = (b_v - tl.sum(b_k[:, None] * b_state, 0)) * b_beta
        b_state += b_k[:, None] * b_delta[None, :]
        b_o = tl.sum(b_q[:, None] * b_state, 0)
        tl.store(output + row * V + o_v, b_o.to(output.dtype.element_ty), mask=m_v)

    if STORE_FINAL_STATE:
        if CONTIGUOUS_FINAL_STATE:
            p_final_state = i_n * H * K * V + i_h * K * V + o_k[:, None] * V + o_v[None, :]
        else:
            p_final_state = p_state
        tl.store(ht + p_final_state, b_state, mask=m_kv)


_fixed_recurrent_fwd_kernel = kda_recurrent_fwd_kernel.fn
_recurrent_autotune_configs: dict[tuple[object, ...], dict[str, object]] = {}
_recurrent_autotune_lock = threading.Lock()


def _launch_recurrent_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    store_final_state: bool,
    state_indices: torch.Tensor | None = None,
    autotune: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Allocate outputs and launch the sequential scan over token spans."""
    batch, tokens, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    num_sequences = batch if cu_seqlens is None else cu_seqlens.shape[0] - 1
    output = torch.empty_like(v, dtype=q.dtype)
    if state_indices is not None:
        # Paged mode reads and writes one slot, so `h0` and `ht` alias the pool and
        # the scan needs no gather/scatter around it.
        final_state = initial_state
    elif store_final_state:
        final_state = q.new_empty(num_sequences, heads, key_dim, value_dim, dtype=torch.float32)
    else:
        final_state = None
    if initial_state is not None:
        state_batch_stride = initial_state.stride(0)
    elif final_state is not None:
        state_batch_stride = final_state.stride(0)
    else:
        state_batch_stride = heads * key_dim * value_dim
    # BV=32 measured faster than 64 on B200 for both decode and prefill
    # (smaller state tiles schedule better; state traffic is identical).
    # One flat launch dimension: sequence-head counts can exceed the 65,535
    # grid-Y limit, while grid-X is effectively unbounded.
    grid = lambda meta: (triton.cdiv(value_dim, meta["BV"]) * num_sequences * heads,)

    def launch(kernel, target_final_state, contiguous_final_state, launch_options):
        kernel[grid](
            q,
            k,
            v,
            gate,
            beta,
            output,
            initial_state,
            target_final_state,
            cu_seqlens,
            state_indices,
            scale=key_dim**-0.5,
            T=tokens,
            N=num_sequences,
            state_batch_stride=state_batch_stride,
            H=heads,
            K=key_dim,
            V=value_dim,
            BK=triton.next_power_of_2(key_dim),
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=target_final_state is not None,
            IS_VARLEN=cu_seqlens is not None,
            USE_STATE_INDICES=state_indices is not None,
            CONTIGUOUS_FINAL_STATE=contiguous_final_state,
            **launch_options,
        )

    if not autotune:
        average_tokens = tokens if cu_seqlens is None else triton.cdiv(tokens, num_sequences)
        # smaller value tiles schedule scans of 32 or more tokens better on b200
        block_v_limit = 16 if average_tokens >= 32 else 32
        launch(
            _fixed_recurrent_fwd_kernel,
            final_state,
            False,
            {
                "BV": min(triton.next_power_of_2(value_dim), block_v_limit),
                "num_warps": 4,
                "num_stages": 3,
            },
        )
        return output, final_state

    tune_key = (
        q.device.index,
        tokens,
        num_sequences,
        heads,
        key_dim,
        value_dim,
        initial_state is not None,
        final_state is not None,
        cu_seqlens is not None,
        state_indices is not None,
        q.dtype,
        k.dtype,
        v.dtype,
        gate.dtype,
        beta.dtype,
        None if initial_state is None else initial_state.dtype,
        None if cu_seqlens is None else cu_seqlens.dtype,
        None if state_indices is None else state_indices.dtype,
    )
    launch_options = _recurrent_autotune_configs.get(tune_key)
    result_is_ready = False
    if launch_options is None:
        with _recurrent_autotune_lock:
            launch_options = _recurrent_autotune_configs.get(tune_key)
            if launch_options is None:
                tuning_final_state = final_state
                contiguous_final_state = state_indices is not None
                if contiguous_final_state:
                    tuning_final_state = q.new_empty(
                        num_sequences, heads, key_dim, value_dim, dtype=torch.float32
                    )
                launch(kda_recurrent_fwd_kernel, tuning_final_state, contiguous_final_state, {})
                launch_options = dict(kda_recurrent_fwd_kernel.best_config.all_kwargs())
                _recurrent_autotune_configs[tune_key] = launch_options
                result_is_ready = not contiguous_final_state

    if not result_is_ready:
        launch(_fixed_recurrent_fwd_kernel, final_state, False, launch_options)
    return output, final_state


def _kda_recurrent_fwd_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    autotune: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    output, final_state = _launch_recurrent_fwd(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        cu_seqlens,
        store_final_state=True,
        autotune=autotune,
    )
    assert final_state is not None
    return output, final_state


def _kda_recurrent_fwd_no_state_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    autotune: bool,
) -> torch.Tensor:
    return _launch_recurrent_fwd(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        cu_seqlens,
        store_final_state=False,
        autotune=autotune,
    )[0]


def _kda_recurrent_fwd_paged_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    autotune: bool,
) -> torch.Tensor:
    return _launch_recurrent_fwd(
        q,
        k,
        v,
        gate,
        beta,
        state_cache,
        cu_seqlens,
        store_final_state=True,
        state_indices=state_indices,
        autotune=autotune,
    )[0]


__all__ = [
    "_recurrent_fwd_no_state_op",
    "_recurrent_fwd_op",
    "_recurrent_fwd_paged_op",
    "forward",
]
