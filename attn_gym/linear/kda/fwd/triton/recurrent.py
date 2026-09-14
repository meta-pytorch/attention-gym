# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inference-only KDA wrappers for recurrent prefill and fused decode.

The vector-gate prefill specialization uses the shared recurrent delta-rule scan;
use ``chunk_kda`` for training.
"""

from __future__ import annotations

import torch

from attn_gym.linear._delta_rule.decode import GateTransform, launch_recurrent_delta_rule_decode
from attn_gym.linear._delta_rule.recurrent import GateKind, launch_recurrent_delta_rule_fwd
from attn_gym.linear.kda.ops import recurrent_decode_forward as decode_forward
from attn_gym.linear.kda.ops import recurrent_decode_op as _recurrent_decode_op
from attn_gym.linear.kda.ops import recurrent_forward as forward
from attn_gym.linear.kda.ops import (
    recurrent_fwd_no_state_op as _recurrent_fwd_no_state_op,
)
from attn_gym.linear.kda.ops import recurrent_fwd_op as _recurrent_fwd_op
from attn_gym.linear.kda.ops import (
    recurrent_fwd_paged_op as _recurrent_fwd_paged_op,
)
from attn_gym.linear.types import KernelOptions


def _launch_kda_recurrent_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    *,
    scale: float,
    store_final_state: bool,
    state_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    autotune: bool = True,
    kernel_options: KernelOptions | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Launch the vector-gate specialization used by recurrent KDA."""
    return launch_recurrent_delta_rule_fwd(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        cu_seqlens,
        scale=scale,
        gate_kind=GateKind.VECTOR,
        store_final_state=store_final_state,
        state_indices=state_indices,
        has_initial_state=has_initial_state,
        autotune=autotune,
        kernel_options=kernel_options,
    )


def _kda_recurrent_fwd_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    scale: float,
    autotune: bool,
    batch_invariant: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    output, final_state = _launch_kda_recurrent_fwd(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        cu_seqlens,
        scale=scale,
        store_final_state=True,
        autotune=autotune,
        kernel_options={"batch_invariant": batch_invariant},
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
    scale: float,
    autotune: bool,
    batch_invariant: bool,
) -> torch.Tensor:
    return _launch_kda_recurrent_fwd(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        cu_seqlens,
        scale=scale,
        store_final_state=False,
        autotune=autotune,
        kernel_options={"batch_invariant": batch_invariant},
    )[0]


def _kda_recurrent_fwd_paged_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    scale: float,
    batch_invariant: bool,
) -> torch.Tensor:
    return _launch_kda_recurrent_fwd(
        q,
        k,
        v,
        gate,
        beta,
        state_cache,
        cu_seqlens,
        scale=scale,
        store_final_state=True,
        state_indices=state_indices,
        has_initial_state=has_initial_state,
        autotune=False,
        kernel_options={"batch_invariant": batch_invariant},
    )[0]


def _kda_recurrent_fwd_batch_invariant_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run packed sequences as single-token launches matching decode."""
    num_sequences = int(cu_seqlens.numel()) - 1
    state_cache = q.new_zeros(
        num_sequences + 1,
        v.shape[2],
        v.shape[3],
        q.shape[3],
        dtype=torch.float32,
    )
    if initial_state is not None:
        state_cache[1:].copy_(initial_state)
    state_indices = torch.arange(
        1,
        num_sequences + 1,
        dtype=torch.int32,
        device=q.device,
    )
    has_initial_state = torch.full(
        (num_sequences,),
        initial_state is not None,
        dtype=torch.bool,
        device=q.device,
    )
    output = torch.zeros_like(v, dtype=q.dtype)
    step_cu_seqlens = torch.tensor([0, 1], dtype=torch.int32, device=q.device)
    for sequence_index in range(num_sequences):
        start = int(cu_seqlens[sequence_index].item())
        end = int(cu_seqlens[sequence_index + 1].item())
        state_index = state_indices[sequence_index : sequence_index + 1]
        for token_index in range(start, end):
            token_slice = slice(token_index, token_index + 1)
            step_output, _ = _launch_kda_recurrent_fwd(
                q[:, token_slice],
                k[:, token_slice],
                v[:, token_slice],
                gate[:, token_slice],
                beta[:, token_slice],
                state_cache,
                step_cu_seqlens,
                scale=scale,
                store_final_state=True,
                state_indices=state_index,
                has_initial_state=has_initial_state[sequence_index : sequence_index + 1],
                autotune=False,
                kernel_options={"batch_invariant": True},
            )
            output[:, token_slice] = step_output
            has_initial_state[sequence_index] = True
    return output, state_cache[1:]


def _kda_recurrent_decode_cuda(
    packed_qkv: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
    output: torch.Tensor,
    lower_bound: float,
    use_lower_bound: bool,
    scale: float,
) -> None:
    launch_recurrent_delta_rule_decode(
        packed_qkv,
        # The shared boundary takes token-major gates; drop KDA's leading unit token dim.
        raw_gate[0],
        raw_beta[0],
        A_log,
        dt_bias,
        state_cache,
        state_indices,
        output,
        gate_kind=GateKind.VECTOR,
        gate_transform=GateTransform.BOUNDED if use_lower_bound else GateTransform.SOFTPLUS,
        key_heads=state_cache.shape[1],
        lower_bound=lower_bound,
        scale=scale,
        has_initial_state=has_initial_state,
        op_name="recurrent_kda_decode",
    )


__all__ = [
    "_kda_recurrent_fwd_batch_invariant_cuda",
    "_recurrent_decode_op",
    "_recurrent_fwd_no_state_op",
    "_recurrent_fwd_op",
    "_recurrent_fwd_paged_op",
    "decode_forward",
    "forward",
]
