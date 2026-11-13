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
``attn_gym.linear.context_parallel``: an FP32 ``[HV, V + K, K]`` map packed as ``[bias; transition]``.
The staged tier is eager-only: it is meant to be wrapped by the caller's own autograd function
around a collective, and the summary kernels reject fake tensors rather than trace under
``torch.compile``.
Which tokens a device owns and how summaries travel are the caller's decisions;
``attn_gym.linear.context_parallel`` is one all-gather composition built only on these handles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

from attn_gym._backends.autotune import should_autotune
from attn_gym._backends.cute import normalize_compact_tensor, normalize_tma_tensor
from attn_gym.linear._delta_rule.chunk_ops import _plain_gate_scan_op
from attn_gym.linear._delta_rule.chunk_schedule import RaggedChunkMetadata, ScheduleRequest
from attn_gym.linear._delta_rule.cute import build_state_grad_summaries, build_state_summaries
from attn_gym.linear._delta_rule.span import CHUNK_SIZE, prepare_span, zero_state
from attn_gym.linear.kda.bwd.cute.chunk_kda_bwd import (
    ChunkKDABwdPrepared,
    _finish_chunk_kda_bwd,
    _prepare_chunk_kda_bwd,
)
from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd import (
    ChunkKDAFactors,
    _finish_chunk_kda_fwd,
    _prepare_chunk_kda_fwd,
)
from attn_gym.linear.kda.impl.fused import _validate_fused_constraints
from attn_gym.linear.kda.impl.warpdrive import validate_warpdrive_constraints
from attn_gym.linear.kda.impl.warpdrive_ops import (
    chunk_warpdrive_packed_bwd_with_exit_cotangent_op,
    chunk_warpdrive_packed_bwd_with_state_op,
    chunk_warpdrive_packed_fwd_with_initial_state_op,
    chunk_warpdrive_packed_fwd_with_state_op,
    validate_warpdrive_available,
)
from attn_gym.linear.kda.validation import (
    resolve_kda_backend,
    resolve_kernel_options,
    validate_kda_inputs,
)
from attn_gym.linear.types import KernelOptions


class ChunkKDASaved(NamedTuple):
    """Forward tensors an autograd function stores for ``chunk_kda_prepare_backward``.

    Every field is a tensor or ``None`` so the tuple can be splatted into
    ``ctx.save_for_backward`` and rebuilt from ``ctx.saved_tensors``. ``aqk``/``akk`` are
    ``None`` when the forward did not materialize them; backward recomputes them.
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


class ChunkKDAWarpdriveSaved(NamedTuple):
    """Forward tensors a warpdrive ``chunk_kda_prepare`` stores for ``chunk_kda_prepare_backward``.

    warpdrive's backward reads the raw ``gate`` and keeps its factors on chip, so the tape carries
    the gate instead of WY factors; ``cumulative_gate`` only feeds the fused factor pass behind
    arbitrary-range summaries. warpdrive always runs packed, so the offsets are never ``None``.
    """

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    gate: torch.Tensor
    cumulative_gate: torch.Tensor
    beta: torch.Tensor
    cu_seqlens: torch.Tensor
    chunk_offsets: torch.Tensor


# NOTE [Summary ranges are subsequences]
# ``state_summaries(bounds)`` and ``state_grad_summaries(bounds)`` take an ``int32 [R, 2]``
# device tensor of LOCAL span offsets (see NOTE [Terminology] in ``attn_gym.linear.context_parallel``).
# Every nonempty row must be one subsequence, ``(cu_seqlens[i], cu_seqlens[i + 1])``, in any
# order and selection; ``start == stop`` is the identity map. That is what the context-parallel
# routing produces: a rank summarizes its whole piece of a document, never part of it. The fused
# kernels would also sum an interior run of 64-token chunks of one subsequence, but that is not
# part of the contract and warpdrive's kernels cannot do it. The values are not checked: they live
# on the device so the launch replays under CUDA Graph capture, and reading them back would sync; a
# row that matches no subsequence is filled with NaN on warpdrive and returns a plausible but wrong
# map on the fused kernels.


def _normalize_state(state: torch.Tensor | None) -> torch.Tensor | None:
    """FP32 entry state with a unit-stride key mode, as the recurrence kernels read it.

    Only the key mode must be contiguous, so a batch or head slice of a larger state buffer is
    passed through without a copy.
    """
    if state is None:
        return None
    state = state.float()
    return state if state.stride(-1) == 1 else state.contiguous()


def _normalize_warpdrive_state(state: torch.Tensor | None) -> torch.Tensor | None:
    """FP32 state as warpdrive's launchers read it through TMA.

    Unit-stride keys and 16-byte-aligned outer strides suffice, so only other layouts are copied.
    """
    return None if state is None else normalize_tma_tensor(state.float())


@dataclass
class ChunkKDAPrepared:
    """Local forward factors shared by ``state_summaries`` and ``run``.

    The factors are large; release the handle once ``run`` has produced the output.
    """

    saved: ChunkKDASaved
    factors: ChunkKDAFactors
    metadata: RaggedChunkMetadata | None
    scale: float
    # The request; ``run`` applies ``should_autotune`` again because the mode can change.
    autotune: bool
    schedule: ScheduleRequest
    fastmath: bool = True

    def state_summaries(self, bounds: torch.Tensor) -> torch.Tensor:
        """Return one FP32 ``[HV, V + K, K]`` map per row of ``bounds`` in a single launch.

        ``bounds`` is an ``int32 [R, 2]`` device tensor of ``[start, stop)`` span offsets, each
        obeying NOTE [Summary ranges are subsequences]; ``start == stop`` yields the identity. The
        ranges are read on the device, so a CUDA Graph captured around this call replays for any
        layout of the same shape.
        """
        return build_state_summaries(
            self.factors.kg,
            self.factors.w,
            self.factors.u,
            self.saved.cumulative_gate,
            bounds,
            fastmath=self.fastmath,
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
            autotune=should_autotune(self.autotune),
            schedule=self.schedule,
            fastmath=self.fastmath,
        )


@dataclass
class ChunkKDAWarpdrivePrepared:
    """Forward handle: ``run`` is the warpdrive with-state kernel and the summaries are warpdrive's
    too.

    warpdrive keeps its WY factors on chip, so each subsequence's ``[B; A]`` map comes from two
    runs of warpdrive's state-only pass over it (from a zero entry state for ``B``, from the
    identity with the value term disabled for ``A``). The backward handle is
    :class:`ChunkKDAWarpdriveBackward`.
    """

    saved: ChunkKDAWarpdriveSaved
    metadata: RaggedChunkMetadata
    scale: float
    fastmath: bool

    def state_summaries(self, bounds: torch.Tensor) -> torch.Tensor:
        """Return one FP32 ``[HV, V + K, K]`` map per row of ``bounds`` in a single launch.

        Rows follow NOTE [Summary ranges are subsequences]. The maps carry warpdrive's
        16-token-chunk rounding, which differs from the fused maps and from an unsharded warpdrive
        pass.
        """
        # Lazy import keeps the optional warpdrive backend out of the fused import path.
        from attn_gym.linear._delta_rule.warpdrive.launchers.state_summary import (
            build_warpdrive_state_summaries,
        )

        saved = self.saved
        return build_warpdrive_state_summaries(
            saved.k,
            saved.v,
            saved.gate,
            saved.beta,
            saved.cu_seqlens,
            bounds=bounds,
            fastmath=self.fastmath,
        )

    def run(
        self,
        initial_state: torch.Tensor | None = None,
        *,
        output_final_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the with-state kernel from an FP32 ``[N, HV, V, K]`` entry state per sequence."""
        saved = self.saved
        if initial_state is None:
            initial_state = zero_state(saved.q, saved.v, self.metadata)
        else:
            initial_state = _normalize_warpdrive_state(initial_state)
        args = (
            saved.q,
            saved.k,
            saved.v,
            saved.gate,
            saved.beta,
            initial_state,
            saved.cu_seqlens,
            self.scale,
            self.fastmath,
        )
        if output_final_state:
            return chunk_warpdrive_packed_fwd_with_state_op(*args)
        return chunk_warpdrive_packed_fwd_with_initial_state_op(*args), None


def chunk_kda_prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None = None,
    scale: float | None = None,
    fastmath: bool = True,
    kernel_options: KernelOptions | None = None,
) -> ChunkKDAPrepared | ChunkKDAWarpdrivePrepared:
    """Run the factor half of ``chunk_kda`` and return a handle for summaries and output.

    Arguments follow ``chunk_kda`` with two restrictions: ``q``/``k``/``v`` must already share a
    float16 or bfloat16 dtype (no silent cast, because the caller owns autograd), and the batch
    dimension must be one so token offsets index one packed span (NOTE [Terminology] in
    ``attn_gym.linear.context_parallel``).
    ``fastmath`` follows ``chunk_kda`` and applies to preparation, summaries, and ``run``.
    Pass the same value to ``chunk_kda_prepare_backward``. ``"backend"`` selects the kernels
    of the local pass and both summaries as in ``chunk_kda``, so the default runs warpdrive on
    SM100-family GPUs except for FP16. Split schedules are not available with entry states. ``"autotune"``
    follows ``chunk_kda``.
    """
    backend, split_backward, split_forward, autotune, schedule = resolve_kernel_options(
        kernel_options, op_name="chunk_kda_prepare"
    )
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
    backend = resolve_kda_backend(backend, q, k, v)
    if backend == "warpdrive":
        validate_warpdrive_available(q)
        if cu_seqlens is None:
            # warpdrive's with-state kernels are packed-only, so it always gets explicit
            # boundaries; arange keeps the launch capture-safe.
            cu_seqlens = torch.arange(2, dtype=torch.int32, device=q.device) * q.shape[1]
    q, k, v, beta, metadata, cu_seqlens, chunk_offsets, scale = prepare_span(
        q, k, v, beta, cu_seqlens=cu_seqlens, scale=scale
    )
    gate = gate.float()
    cumulative_gate = _plain_gate_scan_op(gate, cu_seqlens, chunk_offsets, False)
    if backend == "warpdrive":
        assert metadata is not None and cu_seqlens is not None and chunk_offsets is not None
        gate = normalize_compact_tensor(gate)
        validate_warpdrive_constraints(q, v)
        saved = ChunkKDAWarpdriveSaved(
            q, k, v, gate, cumulative_gate, beta, cu_seqlens, chunk_offsets
        )
        return ChunkKDAWarpdrivePrepared(saved, metadata, scale, fastmath)
    factors = _prepare_chunk_kda_fwd(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        metadata,
        scale=scale,
        autotune=should_autotune(autotune),
        schedule=schedule,
        fastmath=fastmath,
    )
    saved = ChunkKDASaved(
        q, k, v, cumulative_gate, beta, factors.aqk, factors.akk, cu_seqlens, chunk_offsets
    )
    return ChunkKDAPrepared(saved, factors, metadata, scale, autotune, schedule, fastmath)


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
    autotune: bool  # The request, as in ``ChunkKDAPrepared``.
    fastmath: bool

    def state_grad_summaries(self, bounds: torch.Tensor) -> torch.Tensor:
        """Return one FP32 ``[HV, V + K, K]`` reverse map per row of ``bounds`` in one launch.

        Packed as ``[C; R]`` with ``d_entry_state = d_exit_state @ R + C``, where ``C`` is the
        cotangent the range's own ``d_output`` sends to its entry state. Rows follow the contract
        of ``ChunkKDAPrepared.state_summaries``.
        """
        assert self.prepared.qg is not None and self.prepared.kg is not None
        assert self.prepared.w is not None and self.prepared.aqk is not None
        return build_state_grad_summaries(
            self.prepared.qg,
            self.prepared.kg,
            self.prepared.w,
            self.d_output,
            self.prepared.aqk,
            self.saved.cumulative_gate,
            self.scale,
            bounds,
            fastmath=self.fastmath,
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
            self.d_output,
            d_final_state,
            self.initial_state,
            self.metadata,
            self.prepared,
            scale=self.scale,
            chunk_size=CHUNK_SIZE,
            fastmath=self.fastmath,
            autotune=should_autotune(self.autotune),
        )
        dgate = _plain_gate_scan_op(d_cumulative, saved.cu_seqlens, saved.chunk_offsets, True)
        return dq, dk, dv, dgate, dbeta, d_initial_state


@dataclass
class ChunkKDAWarpdriveBackward:
    """Backward handle: ``run`` is the warpdrive stateful backward; the reverse maps are
    warpdrive's.

    Each subsequence's ``[C; R]`` map is the entry cotangent of warpdrive's backward run with a
    zero exit cotangent (``C``) and the forward transition transposed (``R``, the exact adjoint; a
    natively probed adjoint would round differently). Nothing is consumed, so ``run`` and
    ``state_grad_summaries`` may be called in either order.
    """

    saved: ChunkKDAWarpdriveSaved
    d_output: torch.Tensor
    initial_state: torch.Tensor | None
    scale: float
    fastmath: bool

    def state_grad_summaries(self, bounds: torch.Tensor) -> torch.Tensor:
        """Return one FP32 ``[HV, V + K, K]`` reverse map per row of ``bounds`` in one launch.

        Rows follow NOTE [Summary ranges are subsequences].
        """
        from attn_gym.linear._delta_rule.warpdrive.launchers.state_summary import (
            build_warpdrive_state_grad_summaries,
        )

        saved = self.saved
        return build_warpdrive_state_grad_summaries(
            saved.q,
            saved.k,
            saved.v,
            saved.gate,
            saved.beta,
            self.d_output,
            saved.cu_seqlens,
            self.scale,
            transpose_forward_transition=True,
            bounds=bounds,
            fastmath=self.fastmath,
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
        """Run warpdrive's backward from one FP32 ``[N, HV, V, K]`` exit cotangent per sequence.

        Returns ``(dq, dk, dv, dgate, dbeta, d_initial_state)``; the last is ``None`` when the
        forward ran without an entry state. An absent ``d_final_state`` skips the cotangent load
        and gives the token gradients of an explicit zero.
        """
        saved = self.saved
        d_final_state = _normalize_warpdrive_state(d_final_state)
        operands = (saved.q, saved.k, saved.v, saved.gate, saved.beta, self.d_output)
        if self.initial_state is None:
            grads = chunk_warpdrive_packed_bwd_with_exit_cotangent_op(
                *operands, saved.cu_seqlens, d_final_state, self.scale, self.fastmath
            )
            return (*grads, None)
        return chunk_warpdrive_packed_bwd_with_state_op(
            *operands,
            saved.cu_seqlens,
            self.initial_state,
            d_final_state,
            self.scale,
            self.fastmath,
        )


def chunk_kda_prepare_backward(
    saved: ChunkKDASaved | ChunkKDAWarpdriveSaved,
    d_output: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    *,
    scale: float,
    fastmath: bool = True,
    kernel_options: KernelOptions | None = None,
) -> ChunkKDABackward | ChunkKDAWarpdriveBackward:
    """Recompute the local backward tensors before any reverse-summary exchange.

    ``initial_state`` is the entry state the forward ``run`` consumed and ``scale`` is the
    forward handle's resolved ``prepared.scale``; there is no default because a silently
    re-derived scale would corrupt every gradient. Both live outside ``saved`` because the
    caller's autograd function owns them. Pass the forward's ``fastmath`` and
    ``kernel_options`` for matching factors, launch geometry (``"schedule"``), and tuning
    (``"autotune"``). The tape selects the backend: a warpdrive tape
    (:class:`ChunkKDAWarpdriveSaved`) returns :class:`ChunkKDAWarpdriveBackward`.
    """
    if d_output is None:
        d_output = torch.zeros_like(saved.v)
    else:
        d_output = normalize_compact_tensor(d_output.to(saved.v.dtype))
    scale = float(scale)
    options = resolve_kernel_options(kernel_options, op_name="chunk_kda_prepare_backward")
    if isinstance(saved, ChunkKDAWarpdriveSaved):
        return ChunkKDAWarpdriveBackward(
            saved, d_output, _normalize_warpdrive_state(initial_state), scale, fastmath
        )
    metadata = None
    if saved.cu_seqlens is not None:
        assert saved.chunk_offsets is not None
        metadata = RaggedChunkMetadata.from_offsets(
            saved.cu_seqlens, saved.chunk_offsets, saved.q.shape[1], CHUNK_SIZE
        )
    initial_state = _normalize_state(initial_state)  # The backward recomputes the chain from it.
    prepared = _prepare_chunk_kda_bwd(
        saved.q,
        saved.k,
        saved.v,
        saved.cumulative_gate,
        saved.beta,
        saved.aqk,
        saved.akk,
        d_output,
        initial_state,
        metadata,
        scale=scale,
        chunk_size=CHUNK_SIZE,
        autotune=should_autotune(options.autotune),
        schedule=options.schedule,
        fastmath=fastmath,
    )
    return ChunkKDABackward(
        saved, d_output, initial_state, metadata, prepared, scale, options.autotune, fastmath
    )


# Deprecated aliases from the ``cudnn`` backend name.
ChunkKDACudnnSaved = ChunkKDAWarpdriveSaved
ChunkKDACudnnPrepared = ChunkKDAWarpdrivePrepared
ChunkKDACudnnBackward = ChunkKDAWarpdriveBackward

__all__ = [
    "ChunkKDABackward",
    "ChunkKDACudnnBackward",
    "ChunkKDACudnnPrepared",
    "ChunkKDACudnnSaved",
    "ChunkKDAPrepared",
    "ChunkKDASaved",
    "ChunkKDAWarpdriveBackward",
    "ChunkKDAWarpdrivePrepared",
    "ChunkKDAWarpdriveSaved",
    "chunk_kda_prepare",
    "chunk_kda_prepare_backward",
]
