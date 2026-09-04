# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Staged fused GDN around the affine state boundary; see ``attn_gym.linear.kda.stages``.

The gated delta rule is KDA with one scalar decay per head instead of a per-channel vector, so
its summaries reuse the per-channel summary kernels with the gate broadcast across channels. The
handles follow the same protocol as the KDA stages and plug into
``attn_gym.linear.context_parallel`` unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple

import torch

from attn_gym._backends.cute import normalize_compact_tensor, normalize_tma_rows
from attn_gym.linear._delta_rule.cute import (
    build_state_grad_summaries,
    build_state_grad_summary,
    build_state_summaries,
    build_state_summary,
)
from attn_gym.linear._delta_rule.validation import check_summary_range, resolve_scale
from attn_gym.linear.gdn.bwd.triton.chunk_gdn_bwd_recompute import (
    chunk_gdn_recompute_aqk_dense,
    chunk_gdn_recompute_aqk_packed,
)
from attn_gym.linear.gdn.impl.chunk import (
    ChunkGDNBwdPrepared,
    ChunkGDNFactors,
    _finish_chunk_gdn_bwd,
    _finish_chunk_gdn_fwd,
    _prepare_chunk_gdn_bwd,
    _prepare_chunk_gdn_fwd,
    reject_int64_offsets,
    resolve_backward_metadata,
    zero_state,
)
from attn_gym.linear.gdn.ops import _validate_fused_chunk_qkv
from attn_gym.linear.gdn.validation import validate_gdn_inputs
from attn_gym.linear.kda.chunk_schedule import RaggedChunkMetadata, prepare_ragged_chunk_metadata
from attn_gym.linear.kda.ops import _plain_gate_scan_op
from attn_gym.linear.kda.stages import CHUNK_SIZE


def _vector_gate(cumulative_gate: torch.Tensor, key_dim: int) -> torch.Tensor:
    """Broadcast the per-head scalar gate to the per-channel layout the summary kernels read."""
    return cumulative_gate.unsqueeze(-1).expand(*cumulative_gate.shape, key_dim).contiguous()


class ChunkGDNSaved(NamedTuple):
    """Forward tensors an autograd function stores for ``chunk_gdn_prepare_backward``."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    cumulative_gate: torch.Tensor
    beta: torch.Tensor
    inverse: torch.Tensor
    cu_seqlens: torch.Tensor | None
    chunk_offsets: torch.Tensor | None


@dataclass
class ChunkGDNPrepared:
    """Local forward factors shared by ``state_summary`` and ``run``."""

    saved: ChunkGDNSaved
    factors: ChunkGDNFactors
    metadata: RaggedChunkMetadata | None
    scale: float

    def state_summary(self, start: int, stop: int) -> torch.Tensor:
        """Return the FP32 ``[HV, V + K, K]`` map of tokens ``[start, stop)`` from the zero state.

        See NOTE [Summary ranges are whole chunks of one subsequence] in
        ``attn_gym.linear.kda.stages``.
        """
        saved = self.saved
        check_summary_range(saved.q.shape[1], start, stop)
        return build_state_summary(
            self.factors.kg[:, start:stop],
            self.factors.w[:, start:stop],
            self.factors.u[:, start:stop],
            _vector_gate(saved.cumulative_gate[:, start:stop], saved.q.shape[-1]),
        )

    def state_summaries(self, bounds: torch.Tensor) -> torch.Tensor:
        """Return one FP32 ``[HV, V + K, K]`` map per row of ``bounds`` in a single launch.

        See ``attn_gym.linear.kda.stages.ChunkKDAPrepared.state_summaries`` for the contract.
        """
        saved = self.saved
        return build_state_summaries(
            self.factors.kg,
            self.factors.w,
            self.factors.u,
            _vector_gate(saved.cumulative_gate, saved.q.shape[-1]),
            bounds,
        )

    def run(
        self,
        initial_state: torch.Tensor | None = None,
        *,
        output_final_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Finish the local forward from one FP32 ``[N, HV, V, K]`` entry state per sequence."""
        saved = self.saved
        if initial_state is None:
            initial_state = zero_state(saved.q, saved.v, self.metadata)
        output, final_state = _finish_chunk_gdn_fwd(
            saved.q,
            saved.k,
            self.factors,
            saved.cumulative_gate,
            normalize_compact_tensor(initial_state.float()),
            self.metadata,
            self.scale,
        )
        return output, final_state if output_final_state else None


def chunk_gdn_prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None = None,
    scale: float | None = None,
) -> ChunkGDNPrepared:
    """Run the factor half of fused ``chunk_gdn`` and return a handle for summaries and output.

    Arguments follow ``chunk_gdn`` with the batch dimension fixed to one so token offsets index
    one packed stream.
    """
    validate_gdn_inputs(q, k, v, gate, beta, None, cu_seqlens)
    _validate_fused_chunk_qkv(q, k, v)
    if q.shape[0] != 1:
        raise ValueError("chunk_gdn_prepare requires B=1; pack sequences with cu_seqlens")
    tokens = q.shape[1]
    if cu_seqlens is None and tokens % CHUNK_SIZE:
        # A partial tail runs as one packed sequence; arange keeps the launch capture-safe.
        cu_seqlens = torch.arange(2, dtype=torch.int32, device=q.device) * tokens
    metadata = (
        None
        if cu_seqlens is None
        else prepare_ragged_chunk_metadata(cu_seqlens, tokens, CHUNK_SIZE)
    )
    cu_seqlens = None if metadata is None else metadata.cu_seqlens
    chunk_offsets = None if metadata is None else metadata.chunk_offsets
    scale = resolve_scale(scale, q.shape[-1])
    q, k, v = (normalize_tma_rows(tensor) for tensor in (q, k, v))
    beta = normalize_compact_tensor(beta.float())
    cumulative_gate = normalize_compact_tensor(
        _plain_gate_scan_op(gate.float().unsqueeze(-1), cu_seqlens, chunk_offsets, False).squeeze(
            -1
        )
    )
    reject_int64_offsets(q, k, v, cumulative_gate, beta)
    factors = _prepare_chunk_gdn_fwd(k, v, cumulative_gate, beta, metadata)
    saved = ChunkGDNSaved(
        q, k, v, cumulative_gate, beta, factors.inverse, cu_seqlens, chunk_offsets
    )
    return ChunkGDNPrepared(saved, factors, metadata, scale)


@dataclass
class ChunkGDNBackward:
    """Recomputed local backward tensors shared by ``state_grad_summary`` and ``run``."""

    saved: ChunkGDNSaved
    d_output: torch.Tensor
    initial_state: torch.Tensor | None
    metadata: RaggedChunkMetadata | None
    prepared: ChunkGDNBwdPrepared
    scale: float

    @cached_property
    def aqk(self) -> torch.Tensor:
        """Intra-chunk Q/K factor for the reverse summary; the scalar backward does not otherwise
        materialize it on pre-Blackwell targets, so it is computed once on first use."""
        saved = self.saved
        groups = saved.v.shape[2] // saved.q.shape[2]
        q, k = (
            tensor.repeat_interleave(groups, dim=2) if groups > 1 else tensor
            for tensor in (saved.q, saved.k)
        )
        if self.metadata is None:
            return chunk_gdn_recompute_aqk_dense(q, k, saved.cumulative_gate, self.scale)
        return chunk_gdn_recompute_aqk_packed(
            q, k, saved.cumulative_gate, self.scale, self.metadata
        )

    def state_grad_summary(self, start: int, stop: int) -> torch.Tensor:
        """Return the FP32 ``[HV, V + K, K]`` reverse map of tokens ``[start, stop)``.

        See ``ChunkKDABackward.state_grad_summary`` for the bias convention.
        """
        saved = self.saved
        check_summary_range(saved.q.shape[1], start, stop)
        return build_state_grad_summary(
            self.prepared.qg[:, start:stop],
            self.prepared.kg[:, start:stop],
            self.prepared.w[:, start:stop],
            self.d_output[:, start:stop],
            self.aqk[:, start:stop],
            _vector_gate(saved.cumulative_gate[:, start:stop], saved.q.shape[-1]),
            self.scale,
        )

    def state_grad_summaries(self, bounds: torch.Tensor) -> torch.Tensor:
        """Return one FP32 ``[HV, V + K, K]`` reverse map per row of ``bounds`` in one launch.

        See ``attn_gym.linear.kda.stages.ChunkKDABackward.state_grad_summaries``.
        """
        saved = self.saved
        return build_state_grad_summaries(
            self.prepared.qg,
            self.prepared.kg,
            self.prepared.w,
            self.d_output,
            self.aqk,
            _vector_gate(saved.cumulative_gate, saved.q.shape[-1]),
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
        return _finish_chunk_gdn_bwd(
            saved.q,
            saved.k,
            saved.v,
            saved.cumulative_gate,
            saved.beta,
            saved.inverse,
            self.d_output,
            d_final_state,
            self.initial_state,
            self.metadata,
            self.prepared,
            self.scale,
        )


def chunk_gdn_prepare_backward(
    saved: ChunkGDNSaved,
    d_output: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    *,
    scale: float,
) -> ChunkGDNBackward:
    """Recompute the local backward tensors before any reverse-summary exchange.

    ``initial_state`` is the entry state the forward ``run`` consumed and ``scale`` is the forward
    handle's resolved ``prepared.scale``; see ``chunk_kda_prepare_backward``.
    """
    metadata = resolve_backward_metadata(saved.q, saved.cu_seqlens, saved.chunk_offsets)
    scale = float(scale)
    if d_output is None:
        d_output = torch.zeros_like(saved.v)
    else:
        d_output = normalize_compact_tensor(d_output.to(saved.v.dtype))
    if initial_state is not None:
        initial_state = normalize_compact_tensor(initial_state.float())
    reject_int64_offsets(d_output, initial_state)
    prepared = _prepare_chunk_gdn_bwd(
        saved.q,
        saved.k,
        saved.v,
        saved.cumulative_gate,
        saved.beta,
        saved.inverse,
        initial_state,
        metadata,
    )
    return ChunkGDNBackward(saved, d_output, initial_state, metadata, prepared, scale)


__all__ = [
    "ChunkGDNBackward",
    "ChunkGDNPrepared",
    "ChunkGDNSaved",
    "chunk_gdn_prepare",
    "chunk_gdn_prepare_backward",
]
