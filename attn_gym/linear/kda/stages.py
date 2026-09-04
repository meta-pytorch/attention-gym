# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Staged fused KDA around a public affine state boundary.

``chunk_kda`` runs the fused core as one autograd function. Schemes that move recurrent state
between devices (context parallelism, pipelined state handoff, ...) need the same core split around
a communication point in both directions::

    prepared = chunk_kda_prepare(q, k, v, gate, beta, cu_seqlens=cu_seqlens)  # WY factors, once
    summaries = prepared.state_summaries(bounds)     # [bias; transition] per range, one launch
    ...exchange summaries, compose each subsequence's entry state...
    output, final_state = prepared.run(initial_state, output_final_state=True)

    grads = chunk_kda_prepare_backward(saved, d_output, initial_state, scale=prepared.scale)
    grad_summaries = grads.state_grad_summaries(bounds)
    ...exchange, compose each subsequence's exit cotangent...
    dq, dk, dv, dgate, dbeta, d_initial_state = grads.run(d_final_state)

The handles keep the factor tensors private, so the contract is the affine summary described in
``attn_gym.linear.state_summary``: an FP32 ``[HV, V + K, K]`` map packed as ``[bias; transition]``.
Which tokens a device owns and how summaries travel are the caller's decisions; a context-parallel
recipe is one composition built only on these handles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

from attn_gym._backends.cute import normalize_compact_tensor
from attn_gym.linear._delta_rule.cute import build_state_grad_summaries, build_state_summaries
from attn_gym.linear._delta_rule.span import CHUNK_SIZE, prepare_span
from attn_gym.linear.kda.bwd.cute.chunk_kda_bwd import (
    ChunkKDABwdPrepared,
    _finish_chunk_kda_bwd,
    _prepare_chunk_kda_bwd,
)
from attn_gym.linear.kda.chunk_schedule import (
    RaggedChunkMetadata,
    chunk_capacity,
)
from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd import (
    ChunkKDAFactors,
    _finish_chunk_kda_fwd,
    _prepare_chunk_kda_fwd,
)
from attn_gym.linear.kda.impl.fused import _validate_fused_constraints
from attn_gym.linear.kda.ops import _plain_gate_scan_op
from attn_gym.linear.kda.validation import validate_kda_inputs

"""Token block of the fused kernels; see NOTE [Summary ranges are whole chunks of one subsequence]
below."""


class ChunkKDASaved(NamedTuple):
    """Forward tensors an autograd function stores for ``chunk_kda_prepare_backward``.

    Every field is a tensor or ``None`` so the tuple can be splatted into
    ``ctx.save_for_backward`` and rebuilt from ``ctx.saved_tensors``.
    """

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    cumulative_gate: torch.Tensor
    beta: torch.Tensor
    aqk: torch.Tensor
    akk: torch.Tensor
    cu_seqlens: torch.Tensor | None
    chunk_offsets: torch.Tensor | None


def _metadata_from_saved(saved: ChunkKDASaved) -> RaggedChunkMetadata | None:
    if saved.cu_seqlens is None:
        return None
    assert saved.chunk_offsets is not None
    tokens = saved.q.shape[1]
    return RaggedChunkMetadata(
        saved.cu_seqlens,
        saved.chunk_offsets,
        chunk_capacity(tokens, saved.cu_seqlens.shape[0] - 1, CHUNK_SIZE),
        CHUNK_SIZE,
    )


# NOTE [Summary ranges are whole chunks of one subsequence]
# ``state_summaries(bounds)`` and ``state_grad_summaries(bounds)`` take an ``int32 [R, 2]``
# device tensor of LOCAL span offsets (see NOTE [Terminology] in ``attn_gym.linear.state_summary``).
# The factor kernels lay 64-token chunks from each subsequence's first token, and every chunk's
# factors are scaled toward that chunk's last token, so a row is exact only over whole chunks of
# one subsequence:
#
#     start = sub_start + CHUNK_SIZE * i
#     stop  = sub_start + CHUNK_SIZE * j   or   stop = sub_end   (the partial tail chunk is fine)
#     [start, stop) must not cross a ``cu_seqlens`` boundary;  start == stop is the identity
#
# The context-parallel routing always passes whole subsequences, ``(cu_seqlens[i],
# cu_seqlens[i + 1])``, which satisfies this trivially. A row that starts or stops mid-chunk, or
# spans two subsequences, returns a plausible but wrong map. The values are not checked: they live
# on the device so the launch replays under CUDA Graph capture, and reading them back would sync.


def _normalize_state(state: torch.Tensor | None) -> torch.Tensor | None:
    """FP32 entry state with a unit-stride key mode, as the recurrence kernels read it.

    Only the key mode must be contiguous, so a batch or head slice of a larger state buffer is
    passed through without a copy.
    """
    if state is None:
        return None
    state = state.float()
    return state if state.stride(-1) == 1 else state.contiguous()


@dataclass
class ChunkKDAPrepared:
    """Local forward factors shared by ``state_summaries`` and ``run``.

    The factors are large; release the handle once ``run`` has produced the output.
    """

    saved: ChunkKDASaved
    factors: ChunkKDAFactors
    metadata: RaggedChunkMetadata | None
    scale: float
    autotune: bool

    def state_summaries(self, bounds: torch.Tensor) -> torch.Tensor:
        """Return one FP32 ``[HV, V + K, K]`` map per row of ``bounds`` in a single launch.

        ``bounds`` is an ``int32 [R, 2]`` device tensor of ``[start, stop)`` span offsets, each
        obeying NOTE [Summary ranges are whole chunks of one subsequence]; ``start == stop``
        yields the identity. The ranges are read on the device, so a CUDA Graph captured around
        this call replays for any layout of the same shape.
        """
        return build_state_summaries(
            self.factors.kg, self.factors.w, self.factors.u, self.saved.cumulative_gate, bounds
        )

    def run(
        self,
        initial_state: torch.Tensor | None = None,
        *,
        output_final_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Finish the local forward from one FP32 ``[N, HV, V, K]`` entry state per sequence."""
        return _finish_chunk_kda_fwd(
            self.saved.q,
            self.saved.cumulative_gate,
            self.factors,
            _normalize_state(initial_state),
            None,
            None,
            self.metadata,
            scale=self.scale,
            output_final_state=output_final_state,
            autotune=self.autotune,
        )


def chunk_kda_prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None = None,
    scale: float | None = None,
    autotune: bool = True,
) -> ChunkKDAPrepared:
    """Run the factor half of fused ``chunk_kda`` and return a handle for summaries and output.

    Arguments follow ``chunk_kda`` with two restrictions: ``q``/``k``/``v`` must already share a
    float16 or bfloat16 dtype (no silent cast, because the caller owns autograd), and the batch
    dimension must be one so token offsets index one packed span (NOTE [Terminology] in
    ``attn_gym.linear.state_summary``).
    """
    validate_kda_inputs(
        q, k, v, gate, beta, None, cu_seqlens, op_name="chunk_kda_prepare", gate_name="gate"
    )
    _validate_fused_constraints(q, v)
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError(
            "chunk_kda_prepare requires q, k, and v to share dtype float16 or bfloat16"
        )
    q, k, v, beta, metadata, cu_seqlens, chunk_offsets, scale = prepare_span(
        q, k, v, beta, cu_seqlens=cu_seqlens, scale=scale
    )
    cumulative_gate = _plain_gate_scan_op(gate.float(), cu_seqlens, chunk_offsets, False)
    factors = _prepare_chunk_kda_fwd(
        q, k, v, cumulative_gate, beta, metadata, scale=scale, autotune=autotune
    )
    saved = ChunkKDASaved(
        q, k, v, cumulative_gate, beta, factors.aqk, factors.akk, cu_seqlens, chunk_offsets
    )
    return ChunkKDAPrepared(saved, factors, metadata, scale, autotune)


@dataclass
class ChunkKDABackward:
    """Recomputed local backward tensors shared by ``state_grad_summaries`` and ``run``.

    ``run`` consumes the recomputed tensors at their last use, so call it once and last.
    """

    saved: ChunkKDASaved
    d_output: torch.Tensor
    initial_state: torch.Tensor | None
    metadata: RaggedChunkMetadata | None
    prepared: ChunkKDABwdPrepared
    scale: float
    autotune: bool
    fastmath: bool

    def state_grad_summaries(self, bounds: torch.Tensor) -> torch.Tensor:
        """Return one FP32 ``[HV, V + K, K]`` reverse map per row of ``bounds`` in one launch.

        Packed as ``[C; R]`` with ``d_entry_state = d_exit_state @ R + C``, where ``C`` is the
        cotangent the range's own ``d_output`` sends to its entry state. Rows follow the contract
        of ``ChunkKDAPrepared.state_summaries``.
        """
        assert self.prepared.qg is not None and self.prepared.kg is not None
        assert self.prepared.w is not None
        return build_state_grad_summaries(
            self.prepared.qg,
            self.prepared.kg,
            self.prepared.w,
            self.d_output,
            self.saved.aqk,
            self.saved.cumulative_gate,
            self.scale,
            bounds,
        )

    def run(
        self, d_final_state: torch.Tensor | None = None
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        """Finish the local backward from one FP32 ``[N, HV, V, K]`` exit cotangent per sequence.

        Returns ``(dq, dk, dv, dgate, dbeta, d_initial_state)``; the last is ``None`` when the
        forward ran without an entry state.
        """
        if d_final_state is not None:
            d_final_state = normalize_compact_tensor(d_final_state.float())
        saved = self.saved
        dq, dk, dv, d_cumulative, dbeta, d_initial_state = _finish_chunk_kda_bwd(
            saved.q,
            saved.k,
            saved.v,
            saved.cumulative_gate,
            saved.beta,
            saved.aqk,
            saved.akk,
            self.d_output,
            d_final_state,
            self.initial_state,
            self.metadata,
            self.prepared,
            scale=self.scale,
            chunk_size=CHUNK_SIZE,
            fastmath=self.fastmath,
            autotune=self.autotune,
        )
        dgate = _plain_gate_scan_op(d_cumulative, saved.cu_seqlens, saved.chunk_offsets, True)
        return dq, dk, dv, dgate, dbeta, d_initial_state


def chunk_kda_prepare_backward(
    saved: ChunkKDASaved,
    d_output: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    *,
    scale: float,
    autotune: bool = True,
    fastmath: bool = False,
) -> ChunkKDABackward:
    """Recompute the local backward tensors before any reverse-summary exchange.

    ``initial_state`` is the entry state the forward ``run`` consumed and ``scale`` is the
    forward handle's resolved ``prepared.scale``; there is no default because a silently
    re-derived scale would corrupt every gradient. Both live outside ``saved`` because the
    caller's autograd function owns them. ``fastmath`` applies to the gradient kernels as in
    ``chunk_kda``.
    """
    metadata = _metadata_from_saved(saved)
    if d_output is None:
        d_output = torch.zeros_like(saved.v)
    else:
        d_output = normalize_compact_tensor(d_output.to(saved.v.dtype))
    initial_state = _normalize_state(initial_state)  # The backward recomputes the chain from it.
    scale = float(scale)
    prepared = _prepare_chunk_kda_bwd(
        saved.q,
        saved.k,
        saved.v,
        saved.cumulative_gate,
        saved.beta,
        saved.akk,
        d_output,
        initial_state,
        metadata,
        scale=scale,
        chunk_size=CHUNK_SIZE,
        autotune=autotune,
    )
    return ChunkKDABackward(
        saved, d_output, initial_state, metadata, prepared, scale, autotune, fastmath
    )


__all__ = [
    "ChunkKDABackward",
    "ChunkKDAPrepared",
    "ChunkKDASaved",
    "chunk_kda_prepare",
    "chunk_kda_prepare_backward",
]
