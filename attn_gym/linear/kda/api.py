# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Public KDA operations.

``chunk_kda`` supports training and prefill; ``recurrent_kda`` supports decode and
inference prefill. Use ``impl`` to select fused kernels or the eager reference.
"""

from __future__ import annotations

from enum import Enum
from functools import partial

import torch

from attn_gym.linear.kda.impl.fused import chunk_forward as _fused_chunk_forward
from attn_gym.linear.kda.impl.reference import reference_kda
from attn_gym.linear.kda.naive import naive_chunk_kda_from_cumulative, naive_recurrent_kda
from attn_gym.linear.kda.ops import recurrent_forward as _fused_recurrent_forward
from attn_gym.linear.kda.validation import validate_kda_inputs

_CHUNK_SIZE = 64


class Impl(str, Enum):
    """Select a fused or reference KDA implementation.

    The public operations validate shared inputs; fused backends validate their
    extra hardware and shape requirements. There is no automatic fallback.
    """

    FUSED = "fused"
    REFERENCE = "reference"


def _resolve_impl(impl: Impl | str) -> Impl:
    try:
        return Impl(impl)
    except ValueError:
        valid = ", ".join(repr(member.value) for member in Impl)
        raise ValueError(f"unknown impl {impl!r}; expected one of {valid}") from None


def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    cu_seqlens: torch.Tensor | None = None,
    output_final_state: bool = False,
    fastmath: bool = False,
    autotune: bool = True,
    impl: Impl | str = Impl.FUSED,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply chunk-parallel KDA for training and prefill.

    Args:
        q: Queries shaped ``[B, T, H, K]``, scaled by ``1/sqrt(K)`` internally.
        k: Keys shaped like ``q``.
        v: Values shaped ``[B, T, H, V]``.
        cumulative_gate: Inclusive cumulative log2 decay within each 64-token
            chunk, shaped like ``q`` and produced by
            ``bounded_gate_cumsum(chunk_size=64)``.
        beta: Per-token write gate shaped ``[B, T, H]``.
        initial_state: Starting recurrent state, with one ``[H, K, V]`` entry per
            logical sequence.
        cu_seqlens: Packed offsets shaped ``[N + 1]`` for batch-one inputs, as
            contiguous ``int32`` on ``q.device``; they start at zero, never
            decrease, may repeat for empty sequences whose states pass through,
            and may end before ``T``.
        output_final_state: Return the final recurrent state with the output. Rejected
            together with ``state_indices``, which advances the pool in place instead.
        state_indices: Contiguous ``int32`` slot indices, one per logical sequence,
            selecting rows of a paged ``initial_state`` pool shaped
            ``[num_slots, H, K, V]``. Each sequence reads and advances
            ``initial_state[state_indices[i]]`` **in place**, so a caller serving a paged
            cache needs no gather before the scan or scatter after it. The routing is an
            unchecked precondition, since verifying it would cost a device sync: every
            nonempty sequence's slot must lie in ``[0, num_slots)`` and differ from every
            other nonempty sequence's slot. Empty packed sequences touch no state, so
            padding entries may repeat any index. ``"fused"`` only.
        fastmath: Allow less precise fused math for speed; rejected with
            ``"reference"``.
        autotune: Benchmark candidate kernel configurations when true (winners
            are cached and reused); use fixed heuristics when false for
            repeatable selection across machines and cache states.
        impl: ``"fused"`` uses the Blackwell kernels with first-order autograd;
            ``"reference"`` uses differentiable eager PyTorch in FP32, with no
            automatic fallback.

    Returns:
        The output in ``q.dtype`` and either an FP32 final state with one entry
        per logical sequence or ``None``.
    """
    selected_impl = _resolve_impl(impl)
    if selected_impl is Impl.REFERENCE and fastmath:
        raise ValueError("fastmath applies only to impl='fused'")
    validate_kda_inputs(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        initial_state,
        cu_seqlens,
        op_name="chunk_kda",
        gate_name="cumulative_gate",
    )
    if selected_impl is Impl.FUSED:
        return _fused_chunk_forward(
            q,
            k,
            v,
            cumulative_gate,
            beta,
            initial_state,
            cu_seqlens=cu_seqlens,
            output_final_state=output_final_state,
            fastmath=fastmath,
            autotune=autotune,
        )
    return reference_kda(
        partial(naive_chunk_kda_from_cumulative, chunk_size=_CHUNK_SIZE),
        q,
        k,
        v,
        cumulative_gate,
        beta,
        initial_state,
        cu_seqlens,
        output_final_state,
    )


def _validate_paged_state(
    q: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    state_indices: torch.Tensor,
    output_final_state: bool,
) -> None:
    """Validate the paged state pool and its slot indices."""
    if initial_state is None:
        raise ValueError("state_indices requires initial_state as the paged state pool")
    if output_final_state:
        raise ValueError("state_indices advances the pool in place; drop output_final_state")
    if initial_state.ndim != 4 or initial_state.shape[1:] != (*q.shape[2:], v.shape[-1]):
        raise ValueError(
            "the paged state pool must have shape "
            f"[num_slots, {q.shape[2]}, {q.shape[3]}, {v.shape[-1]}], "
            f"got {tuple(initial_state.shape)}"
        )
    # The pool is advanced in place, so a copy would silently drop the update.
    if initial_state.dtype != torch.float32 or not initial_state.is_contiguous():
        raise TypeError("the paged state pool must be contiguous float32")
    num_sequences = q.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1
    if (
        tuple(state_indices.shape) != (num_sequences,)
        or state_indices.dtype != torch.int32
        or not state_indices.is_contiguous()
        or state_indices.device != q.device
    ):
        raise ValueError(
            f"state_indices must be a contiguous int32 tensor of shape ({num_sequences},) "
            f"on q.device, got {tuple(state_indices.shape)} of {state_indices.dtype}"
        )


def recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    cu_seqlens: torch.Tensor | None = None,
    output_final_state: bool = False,
    state_indices: torch.Tensor | None = None,
    autotune: bool = True,
    impl: Impl | str = Impl.FUSED,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply recurrent KDA for decoding and inference prefill.

    Args:
        q: Queries shaped ``[B, T, H, K]``, scaled by ``1/sqrt(K)`` internally.
        k: Keys shaped like ``q``.
        v: Values shaped ``[B, T, H, V]``.
        gate: Per-token log2 decay shaped like ``q``, as produced by
            ``bounded_gate_cumsum(chunk_size=1)``; do not pass chunk-cumulative
            gates.
        beta: Per-token write gate shaped ``[B, T, H]``.
        initial_state: Starting recurrent state, with one ``[H, K, V]`` entry per
            logical sequence.
        cu_seqlens: Packed offsets shaped ``[N + 1]`` for batch-one inputs, as
            contiguous ``int32`` on ``q.device``; they start at zero, never
            decrease, may repeat for empty sequences whose states pass through,
            and may end before ``T``.
        output_final_state: Return the final recurrent state with the output. Rejected
            together with ``state_indices``, which advances the pool in place instead.
        state_indices: Contiguous ``int32`` slot indices, one per logical sequence,
            selecting rows of a paged ``initial_state`` pool shaped
            ``[num_slots, H, K, V]``. Each sequence reads and advances
            ``initial_state[state_indices[i]]`` **in place**, so a caller serving a paged
            cache needs no gather before the scan or scatter after it. The routing is an
            unchecked precondition, since verifying it would cost a device sync: every
            nonempty sequence's slot must lie in ``[0, num_slots)`` and differ from every
            other nonempty sequence's slot. Empty packed sequences touch no state, so
            padding entries may repeat any index. ``"fused"`` only.
        autotune: Reserved for implementation parity with ``chunk_kda``. The
            current recurrent kernel uses the same fixed launch policy for both
            values.
        impl: ``"fused"`` uses the inference-only optimized scan; ``"reference"``
            uses differentiable eager PyTorch in FP32, with no automatic
            fallback.

    Returns:
        The output in ``q.dtype`` and either an FP32 final state with one entry
        per logical sequence or ``None``.

    Serving limitations: without ``state_indices`` state rows map directly to logical
    sequences and final states are written out of place, decode preprocessing and scan are
    separate launches, and speculative-decoding rollback is unsupported.
    """
    del autotune
    selected_impl = _resolve_impl(impl)
    # A paged pool's leading dimension is the slot count, not the sequence count, so the
    # shared per-sequence state check does not apply; the pool is checked below instead.
    validate_kda_inputs(
        q,
        k,
        v,
        gate,
        beta,
        None if state_indices is not None else initial_state,
        cu_seqlens,
        op_name="recurrent_kda",
        gate_name="gate",
    )
    if state_indices is not None:
        _validate_paged_state(q, v, initial_state, cu_seqlens, state_indices, output_final_state)
        if selected_impl is not Impl.FUSED:
            raise ValueError("state_indices requires impl='fused'")
    if selected_impl is Impl.FUSED:
        return _fused_recurrent_forward(
            q,
            k,
            v,
            gate,
            beta,
            initial_state,
            cu_seqlens=cu_seqlens,
            output_final_state=output_final_state,
            state_indices=state_indices,
        )
    return reference_kda(
        naive_recurrent_kda, q, k, v, gate, beta, initial_state, cu_seqlens, output_final_state
    )


__all__ = ["Impl", "chunk_kda", "recurrent_kda"]
