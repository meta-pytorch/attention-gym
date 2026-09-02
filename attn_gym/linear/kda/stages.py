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
    summary = prepared.state_summary(start, stop)             # [bias; transition] of one sequence
    ...exchange summaries, compose each sequence's entry state...
    output, final_state = prepared.run(initial_state, output_final_state=True)

    grads = chunk_kda_prepare_backward(saved, d_output, initial_state, scale=scale)
    grad_summary = grads.state_grad_summary(start, stop)
    ...exchange, compose each sequence's exit cotangent...
    dq, dk, dv, dgate, dbeta, d_initial_state = grads.run(d_final_state)

The handles keep the factor tensors private, so the contract is the affine summary described in
``attn_gym.linear.state_summary``: an FP32 ``[HV, V + K, K]`` map packed as ``[bias; transition]``.
Which tokens a device owns and how summaries travel are the caller's decisions;
``attn_gym.linear.context_parallel`` is one all-gather composition built only on these handles.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple

import torch

from attn_gym._backends.cute import normalize_compact_tensor, normalize_tma_rows
from attn_gym.linear._delta_rule.cute import build_state_grad_summary, build_state_summary
from attn_gym.linear._delta_rule.validation import resolve_scale
from attn_gym.linear.kda.bwd.cute.chunk_kda_bwd import (
    ChunkKDABwdPrepared,
    _finish_chunk_kda_bwd,
    _prepare_chunk_kda_bwd,
)
from attn_gym.linear.kda.chunk_schedule import (
    RaggedChunkMetadata,
    chunk_capacity,
    prepare_ragged_chunk_metadata,
)
from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd import (
    ChunkKDAFactors,
    _finish_chunk_kda_fwd,
    _prepare_chunk_kda_fwd,
)
from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_intra import chunk_kda_fwd_factors
from attn_gym.linear.kda.impl.fused import _validate_fused_constraints
from attn_gym.linear.kda.impl.mega import validate_mega_constraints
from attn_gym.linear.kda.impl.mega_ops import (
    chunk_mega_packed_fwd_with_initial_state_op,
    chunk_mega_packed_fwd_with_state_op,
    validate_mega_available,
)
from attn_gym.linear.kda.ops import _plain_gate_scan_op
from attn_gym.linear.kda.validation import resolve_kernel_options, validate_kda_inputs

_CHUNK_SIZE = 64


class ChunkKDASaved(NamedTuple):
    """Forward tensors an autograd function stores for ``chunk_kda_prepare_backward``.

    Every field is a tensor or ``None`` so the tuple can be splatted into
    ``ctx.save_for_backward`` and rebuilt from ``ctx.saved_tensors``. ``aqk``/``akk`` are
    ``None`` when the forward did not materialize them (Mega); backward recomputes them.
    """

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    cumulative_gate: torch.Tensor
    beta: torch.Tensor
    aqk: torch.Tensor | None
    akk: torch.Tensor | None
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
        chunk_capacity(tokens, saved.cu_seqlens.shape[0] - 1, _CHUNK_SIZE),
        _CHUNK_SIZE,
    )


def _check_token_range(tokens: int, start: int, stop: int) -> None:
    """Reject bounds outside the stream; see NOTE [Summary ranges are whole sequences]."""
    if not 0 <= start < stop <= tokens:
        raise ValueError(f"summary range [{start}, {stop}) must lie inside [0, {tokens})")


# NOTE [Summary ranges are whole sequences]
# The WY factors are chunked relative to each packed sequence's first token, so a summary is only
# meaningful over a range that starts at a sequence start and ends at that sequence's end. The
# bounds are host integers rather than reads from ``cu_seqlens`` so CUDA Graph capture never needs
# a device sync; callers own the packed layout and already know these offsets.


@dataclass
class ChunkKDAPrepared:
    """Local forward factors shared by ``state_summary`` and ``run``.

    The factors are large; release the handle once ``run`` has produced the output.
    """

    saved: ChunkKDASaved
    factors: ChunkKDAFactors
    metadata: RaggedChunkMetadata | None
    scale: float
    autotune: bool

    def state_summary(self, start: int, stop: int) -> torch.Tensor:
        """Return the FP32 ``[HV, V + K, K]`` map of tokens ``[start, stop)`` from the zero state.

        See NOTE [Summary ranges are whole sequences].
        """
        _check_token_range(self.saved.q.shape[1], start, stop)
        return build_state_summary(
            self.factors.kg[:, start:stop],
            self.factors.w[:, start:stop],
            self.factors.u[:, start:stop],
            self.saved.cumulative_gate[:, start:stop],
        )

    def run(
        self,
        initial_state: torch.Tensor | None = None,
        *,
        output_final_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Finish the local forward from one FP32 ``[N, HV, V, K]`` entry state per sequence."""
        if initial_state is not None:
            initial_state = initial_state.float().contiguous()
        return _finish_chunk_kda_fwd(
            self.saved.q,
            self.saved.cumulative_gate,
            self.factors,
            initial_state,
            None,
            None,
            self.metadata,
            scale=self.scale,
            output_final_state=output_final_state,
            autotune=self.autotune,
        )


@dataclass
class ChunkKDAMegaPrepared:
    """Mega forward handle: on-chip factors for ``run``, fused factors only for summaries.

    Mega never materializes the WY factors, so ``run`` executes the Mega with-state kernel over the
    whole local stream and ``state_summary`` computes the fused factors for just the requested
    sequence. Ranks whose fragments all end their sequence never pay for them, and packed streams
    pay only for the fragments that continue elsewhere. Backward recomputes the intra factors, as
    Mega does today.
    """

    saved: ChunkKDASaved  # aqk/akk are None: Mega keeps them on chip
    gate: torch.Tensor
    metadata: RaggedChunkMetadata
    scale: float
    autotune: bool

    def state_summary(self, start: int, stop: int) -> torch.Tensor:
        """Return the FP32 ``[HV, V + K, K]`` map of tokens ``[start, stop)`` from the zero state.

        See NOTE [Summary ranges are whole sequences].
        """
        saved = self.saved
        _check_token_range(saved.q.shape[1], start, stop)
        # One whole sequence starts at ``start``, so its chunk grid and cumulative gate are
        # unchanged when it is factored as a stream of its own.
        tokens = stop - start
        metadata = None
        if tokens % _CHUNK_SIZE:
            cu_seqlens = torch.arange(2, dtype=torch.int32, device=saved.q.device) * tokens
            metadata = prepare_ragged_chunk_metadata(cu_seqlens, tokens, _CHUNK_SIZE)
        q, k, v = (normalize_tma_rows(t[:, start:stop]) for t in (saved.q, saved.k, saved.v))
        cumulative_gate, beta = (
            normalize_compact_tensor(t[:, start:stop]) for t in (saved.cumulative_gate, saved.beta)
        )
        factors = _prepare_chunk_kda_fwd(
            q, k, v, cumulative_gate, beta, metadata, scale=self.scale, autotune=self.autotune
        )
        return build_state_summary(factors.kg, factors.w, factors.u, cumulative_gate)

    def run(
        self,
        initial_state: torch.Tensor | None = None,
        *,
        output_final_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the Mega with-state kernel from one FP32 ``[N, HV, V, K]`` entry state per sequence."""
        saved = self.saved
        if initial_state is None:
            sequences = self.metadata.cu_seqlens.shape[0] - 1
            initial_state = saved.q.new_zeros(
                sequences,
                saved.v.shape[2],
                saved.v.shape[-1],
                saved.q.shape[-1],
                dtype=torch.float32,
            )
        args = (
            saved.q,
            saved.k,
            saved.v,
            self.gate,
            saved.beta,
            initial_state.float().contiguous(),
            self.metadata.cu_seqlens,
            self.scale,
        )
        if output_final_state:
            return chunk_mega_packed_fwd_with_state_op(*args)
        return chunk_mega_packed_fwd_with_initial_state_op(*args), None


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
    kernel_options: Mapping[str, object] | None = None,
) -> ChunkKDAPrepared | ChunkKDAMegaPrepared:
    """Run the factor half of ``chunk_kda`` and return a handle for summaries and output.

    Arguments follow ``chunk_kda`` with two restrictions: ``q``/``k``/``v`` must already share a
    float16 or bfloat16 dtype (no silent cast, because the caller owns autograd), and the batch
    dimension must be one so token offsets index one packed stream.
    ``kernel_options={"backend": "mega"}`` runs the local pass with Mega and computes fused factors
    only when a summary is requested; the split schedules are not available with entry states.
    """
    backend, split_backward, split_forward = resolve_kernel_options(kernel_options)
    if split_backward or split_forward:
        raise ValueError("split schedules are not supported by chunk_kda_prepare")
    validate_kda_inputs(
        q, k, v, gate, beta, None, cu_seqlens, op_name="chunk_kda_prepare", gate_name="gate"
    )
    _validate_fused_constraints(q, v)
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError(
            "chunk_kda_prepare requires q, k, and v to share dtype float16 or bfloat16"
        )
    if q.shape[0] != 1:
        raise ValueError("chunk_kda_prepare requires B=1; pack sequences with cu_seqlens")
    q, k, v = (normalize_tma_rows(tensor) for tensor in (q, k, v))
    tokens = q.shape[1]
    # Mega's with-state kernels are packed-only, so it always gets explicit boundaries; arange
    # keeps the launch capture-safe.
    if cu_seqlens is None and (tokens % _CHUNK_SIZE or backend == "mega"):
        cu_seqlens = torch.arange(2, dtype=torch.int32, device=q.device) * tokens
    metadata = (
        None
        if cu_seqlens is None
        else prepare_ragged_chunk_metadata(cu_seqlens, tokens, _CHUNK_SIZE)
    )
    cu_seqlens = None if metadata is None else metadata.cu_seqlens
    chunk_offsets = None if metadata is None else metadata.chunk_offsets
    scale = resolve_scale(scale, q.shape[-1])
    beta = normalize_compact_tensor(beta.float())
    gate = gate.float()
    cumulative_gate = _plain_gate_scan_op(gate, cu_seqlens, chunk_offsets, False)
    if backend == "mega":
        assert metadata is not None
        validate_mega_available(q)
        gate = normalize_compact_tensor(gate)
        validate_mega_constraints(q, k, v, gate, beta, None, metadata.cu_seqlens)
        saved = ChunkKDASaved(
            q, k, v, cumulative_gate, beta, None, None, cu_seqlens, chunk_offsets
        )
        return ChunkKDAMegaPrepared(saved, gate, metadata, scale, autotune)
    factors = _prepare_chunk_kda_fwd(
        q, k, v, cumulative_gate, beta, metadata, scale=scale, autotune=autotune
    )
    saved = ChunkKDASaved(
        q, k, v, cumulative_gate, beta, factors.aqk, factors.akk, cu_seqlens, chunk_offsets
    )
    return ChunkKDAPrepared(saved, factors, metadata, scale, autotune)


@dataclass
class ChunkKDABackward:
    """Recomputed local backward tensors shared by ``state_grad_summary`` and ``run``.

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

    def state_grad_summary(self, start: int, stop: int) -> torch.Tensor:
        """Return the FP32 ``[HV, V + K, K]`` reverse map of tokens ``[start, stop)``.

        Its bias is the range's own contribution to the entry-state cotangent given a zero exit
        cotangent; callers with a nonzero exit cotangent fold it in with ``merge_state`` before
        sending the summary upstream. See NOTE [Summary ranges are whole sequences].
        """
        _check_token_range(self.saved.q.shape[1], start, stop)
        assert self.prepared.qg is not None and self.prepared.kg is not None
        assert self.prepared.w is not None
        return build_state_grad_summary(
            self.prepared.qg[:, start:stop],
            self.prepared.kg[:, start:stop],
            self.prepared.w[:, start:stop],
            self.d_output[:, start:stop],
            self.saved.aqk[:, start:stop],
            self.saved.cumulative_gate[:, start:stop],
            self.scale,
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
            chunk_size=_CHUNK_SIZE,
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
    scale: float | None = None,
    autotune: bool = True,
    fastmath: bool = False,
) -> ChunkKDABackward:
    """Recompute the local backward tensors before any reverse-summary exchange.

    ``initial_state`` is the entry state the forward ``run`` consumed and ``scale`` must match
    the forward call; both live outside ``saved`` because the caller's autograd function owns
    them. ``fastmath`` applies to the gradient kernels as in ``chunk_kda``.
    """
    metadata = _metadata_from_saved(saved)
    if d_output is None:
        d_output = torch.zeros_like(saved.v)
    else:
        d_output = normalize_compact_tensor(d_output.to(saved.v.dtype))
    if initial_state is not None:
        initial_state = initial_state.float().contiguous()
    scale = resolve_scale(scale, saved.q.shape[-1])
    if saved.aqk is None:
        # Mega forwards keep the intra factors on chip; rebuild them as Mega's backward does.
        aqk, akk = chunk_kda_fwd_factors(
            saved.q, saved.k, saved.cumulative_gate, saved.beta, scale, metadata
        )
        saved = saved._replace(aqk=aqk, akk=akk)
    assert saved.akk is not None
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
        chunk_size=_CHUNK_SIZE,
        autotune=autotune,
    )
    return ChunkKDABackward(
        saved, d_output, initial_state, metadata, prepared, scale, autotune, fastmath
    )


__all__ = [
    "ChunkKDABackward",
    "ChunkKDAMegaPrepared",
    "ChunkKDAPrepared",
    "ChunkKDASaved",
    "chunk_kda_prepare",
    "chunk_kda_prepare_backward",
]
