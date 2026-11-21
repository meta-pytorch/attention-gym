# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Public KDA operations.

``chunk_kda`` supports training and prefill; ``paged_chunk_kda`` advances a mutable
state cache during inference; ``recurrent_kda`` supports decode and inference prefill.
Use ``impl`` to select fused kernels or the eager reference where available.
"""

from __future__ import annotations

import sys
from functools import partial
from numbers import Real
from typing import Literal

import torch

from attn_gym._backends.autotune import should_autotune
from attn_gym.linear._delta_rule.constants import DEFAULT_CHUNK_SIZE as _CHUNK_SIZE
from attn_gym.linear._delta_rule.reference import reference_delta_rule
from attn_gym.linear._delta_rule.state_checkpoints import validate_state_checkpoints
from attn_gym.linear._delta_rule.validation import (
    require_inference_only,
    resolve_chunk_backend,
    resolve_decode_out,
    resolve_scale,
    validate_decode_inputs,
    validate_paged_state,
)
from attn_gym.linear.kda.constants import LOG2_E
from attn_gym.linear.kda.impl.fused import chunk_forward as _fused_chunk_forward
from attn_gym.linear.kda.impl.fused import paged_chunk_forward as _fused_paged_chunk_forward
from attn_gym.linear.kda.impl.warpdrive import chunk_forward as _warpdrive_chunk_forward
from attn_gym.linear.kda.impl.warpdrive import (
    paged_chunk_forward as _warpdrive_paged_chunk_forward,
)
from attn_gym.linear.kda.naive import naive_chunk_kda, naive_recurrent_kda
from attn_gym.linear.kda.ops import copy_replay_tails_op
from attn_gym.linear.kda.ops import recurrent_decode_forward as _fused_recurrent_decode_forward
from attn_gym.linear.kda.ops import recurrent_forward as _fused_recurrent_forward
from attn_gym.linear.kda.validation import (
    kernel_dtype,
    resolve_intra_chunk_dtypes,
    resolve_kernel_options,
    validate_kda_inputs,
    validate_replay_state,
)
from attn_gym.linear.types import (
    RECURRENT_OPTION_SPEC,
    Impl,
    KernelOptions,
    RecurrentKernelOptions,
    ReplayState,
    StateCheckpoints,
    resolve_impl,
)
from attn_gym.types import validate_kernel_options

_DECODE_GATE_TRANSFORMS = {
    "bounded": True,
    "softplus": False,
}


def _resolve_decode_gate_transform(gate_transform: str) -> bool:
    try:
        return _DECODE_GATE_TRANSFORMS[gate_transform]
    except KeyError:
        supported = ", ".join(sorted(_DECODE_GATE_TRANSFORMS))
        raise ValueError(
            f"gate_transform must be one of {{{supported}}}, got {gate_transform!r}"
        ) from None


def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    cu_seqlens: torch.Tensor | None = None,
    scale: float | None = None,
    output_final_state: bool = False,
    fastmath: bool = True,
    impl: Impl | str = Impl.FUSED,
    kernel_options: KernelOptions | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply chunk-parallel KDA for training and prefill.

    Args:
        q: Queries shaped ``[B, T, H, K]``, scaled by ``scale`` internally. Use
            L2-normalized Q/K with FP16: the products accumulate in FP32, but unscaled
            attention scores can exceed the largest finite FP16 value (65504) when staged for
            the next multiply. BF16 kernels do not have this FP16 range limit.
        k: Keys shaped like ``q`` and subject to the same FP16 range limitation.
        v: Values shaped ``[B, T, H, V]``.
        gate: Finite, nonpositive per-token natural-log decay shaped like ``q``. At
            each token the previous state is multiplied channelwise by ``exp(gate)``.
            Pass per-token values, not cumulative gates; chunking and log-base conversion
            are internal. For training with warpdrive, keep gates in about ``[-5.25, 0]``
            (``WARPDRIVE_MAX_GATE_LOWER_BOUND_MAGNITUDE``). More negative uniform gates can
            rapidly degrade gradients; large output gradients or key components reduce the margin.
            Triton supports about ``[-5.914, 0]`` (``MAX_GATE_LOWER_BOUND_MAGNITUDE``).
            The common floor of ``-5`` fits both. These limits are not checked at runtime and do
            not apply to reference or recurrent execution. See the KDA gate contract in
            ``docs/linear.md`` for the 16-token decay limit and FP16 state-gradient range.
        beta: Per-token write gate shaped ``[B, T, H]``.
        initial_state: Starting recurrent state, with one ``[H, V, K]`` entry per
            logical sequence.
        cu_seqlens: Packed offsets shaped ``[N + 1]`` for batch-one inputs, as
            contiguous ``int32`` on ``q.device``; they start at zero, never
            decrease, may repeat for empty sequences whose states pass through,
            and may end before ``T``. When the endpoint is below ``T``, output
            values and token-input gradients beyond it are undefined. Fixed-capacity
            callers must sanitize outputs with ``mask_inactive_tokens`` and protect
            parameterized input producers with ``mask_inactive_token_gradients``.
        scale: Query scale, applied inside the kernels in FP32. Defaults to
            ``1 / sqrt(K)``.
        output_final_state: Return the final recurrent state with the output.
        fastmath: ``True`` is the default because it’s faster, and turning it off usually
            didn’t improve accuracy in our tests. If you’re seeing unexpected outputs or
            gradients, especially with strong negative gates, try ``False`` and compare
            against the reference. It can preserve small contributions that fast math
            drops when intermediate values fall into the subnormal range. On warpdrive both
            values give the same bits inside its gate range; ``False`` only keeps results
            finite slightly past it, where accuracy is already poor ("Fast math and
            precision" in ``docs/linear.md``). The reference backend uses normal PyTorch math
            for either value. Configure gate production separately with ``bound_gate`` or
            ``gate_transform``.
        impl: ``"fused"`` uses optimized kernels and ``"reference"`` uses
            differentiable eager PyTorch in FP32. There is no automatic fallback between
            the two.
        kernel_options: Backend-specific options. This is a sneaky BC trick.
            It is annoying to have a bunch of kwargs that its hard to know when they apply
            but, this trick allows us to add new options without breaking the API. I will
            do my best to add new options by providing a nice typed dict and hopefully
            your agent will now how to use this and be able to find the right option to wiggle.
            See :class:`attn_gym.linear.types.KernelOptions`. With ``"backend": "auto"`` (default),
            both FP16 and BF16 use warpdrive on supported SM100-family GPUs with CuTeDSL
            installed; other devices use Triton (see "Chunk backend selection" in
            ``docs/linear.md``). ``"autotune"`` (default ``True``) tunes Triton
            configurations. Warpdrive chooses a plan without benchmarking.
            ``"intra_chunk_dtypes"`` sets warpdrive operand precision in forward and backward.
            BF16 kernels accept ``"tf32+bf16"`` (TF32 decay, BF16 solve; default), ``"tf32"``
            or ``"bf16"``. FP16 kernels default to ``"tf32+fp16"`` and accept no other value.
            Replay-backed ``paged_chunk_kda`` uses BF16 even for FP16 inputs. Incompatible
            choices raise on every implementation. See "Operand precision" in ``docs/linear.md``.
            With explicit warpdrive, ``"batch_invariant": True`` keeps each sequence's outputs,
            final state and gradients bitwise independent of its neighbors.

    Returns:
        The output in ``q.dtype`` and either an FP32 final state with one entry
        per logical sequence or ``None``.
    """
    selected_impl = resolve_impl(impl)
    options = resolve_kernel_options(kernel_options)
    autotune = should_autotune(options.autotune)
    validate_kda_inputs(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        cu_seqlens,
        op_name="chunk_kda",
        gate_name="gate",
    )
    intra_chunk_dtypes = resolve_intra_chunk_dtypes(
        options.intra_chunk_dtypes, kernel_dtype(q, k, v), op_name="chunk_kda"
    )
    scale = resolve_scale(scale, q.shape[-1])
    if selected_impl is Impl.FUSED:
        if resolve_chunk_backend(options.backend, q) == "warpdrive":
            return _warpdrive_chunk_forward(
                q,
                k,
                v,
                gate,
                beta,
                initial_state,
                cu_seqlens=cu_seqlens,
                scale=scale,
                output_final_state=output_final_state,
                fastmath=fastmath,
                split_backward=options.split_backward,
                split_forward=options.split_forward,
                intra_chunk_dtypes=intra_chunk_dtypes,
                batch_invariant=options.batch_invariant,
            )
        return _fused_chunk_forward(
            q,
            k,
            v,
            gate,
            beta,
            initial_state,
            cu_seqlens=cu_seqlens,
            scale=scale,
            output_final_state=output_final_state,
            fastmath=fastmath,
            autotune=autotune,
            schedule=options.schedule,
        )
    return reference_delta_rule(
        partial(naive_chunk_kda, chunk_size=_CHUNK_SIZE),
        q,
        k,
        v,
        gate.float() * LOG2_E,
        beta,
        compute_dtype=torch.float32,
        scale=scale,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        output_final_state=output_final_state,
    )


def paged_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    kernel_options: KernelOptions | None = None,
    replay_state: ReplayState | None = None,
    state_checkpoints: StateCheckpoints | None = None,
) -> torch.Tensor:
    """Apply inference-only chunk KDA while advancing a paged state cache in place.

    Paged and replay-backed calls use ``fastmath=True``.

    Args:
        q: Queries shaped ``[B, T, H, K]``, scaled by ``1/sqrt(K)`` internally.
        k: Keys shaped like ``q``.
        v: Values shaped ``[B, T, H, V]``.
        gate: Finite, nonpositive per-token natural-log decay shaped like ``q``.
            Pass per-token values, not cumulative gates; chunking and log-base
            conversion are internal, matching :func:`chunk_kda`.
        beta: Per-token write gate shaped ``[B, T, H]``.
        state_cache: Mutable FP32 or BF16 state pool shaped ``[num_slots, H, V, K]``.
            Recurrence math remains FP32; BF16 affects persistent storage only. The pool base
            and slot stride need only element alignment.
        state_indices: Contiguous ``int32`` slot indices, one per logical sequence.
            Positive, unique indices select cache slots to read and advance;
            non-positive indices produce zero output and leave the cache untouched.
        cu_seqlens: Packed offsets shaped ``[N + 1]`` for batch-one inputs, as
            contiguous ``int32`` on ``q.device``. They start at zero, never
            decrease, may repeat for empty sequences, and may end before ``T``; output rows
            past ``cu_seqlens[-1]`` are not written and hold unspecified values.
        has_initial_state: Optional contiguous boolean mask, one per logical sequence.
            False entries ignore the selected cache contents and start from zero before
            advancing that slot. This is useful when a slot has just been assigned.
        kernel_options: Backend, schedule, precision and batch invariance follow ``chunk_kda``.
            Replay casts Q/K/V to BF16 before choosing operand precision. Paged execution does
            not support split schedules. ``"autotune"`` only tunes Triton kernels.
        replay_state: Optional :class:`ReplayState` persistent cache.
            Q, K, and V use BF16, gate and beta use FP32, and count uses int32. The five
            token caches have shape ``[num_slots, 64, ...]``; every count must remain in
            ``[0, 63]``, and no replay-state tensors may alias. Together, ``state_cache`` and
            ``replay_state`` own the logical recurrent state: ``state_cache`` is the FP32 state
            after the last complete 64-token chunk, while ``replay_state`` retains the unfinished
            suffix. For example, after 150 tokens they hold the state at token 128 and the final
            22 tokens, respectively. Callers must preserve and pass both across invocations;
            switching to state-only decode would discard that suffix. When supplied, calls with
            one token per dense batch row use replay-backed decode; other calls use streaming
            prefill. Both paths support ``torch.compile`` and CUDA graph capture.

            Persistent storage consists of the five per-slot token caches plus their counts.
            Replay prefill additionally allocates five prefix buffers with token capacity
            ``total_input_capacity + 63 * num_requests`` rounded up to a multiple of 64 and five
            tail buffers shaped ``[num_requests, 64, ...]``, plus routing and compute
            intermediates. CUDA graph users such as vLLM must budget this workspace for every
            captured bucket. Graph replay may change packed offsets, slot routing, initialization
            flags, and replay counts while tensor shapes remain fixed to that bucket.

        state_checkpoints: Optional per-sequence checkpoints written into caller-chosen slots of a
            state pool. See :class:`attn_gym.linear.StateCheckpoints` for layout and boundary
            semantics. Runs on the fused backend: ``"auto"`` selects it, and ``"warpdrive"``
            raises. With replay, only the completed boundary state is saved. The caller must
            separately preserve any replay tail needed to resume.

    Returns:
        The output in ``q.dtype``. ``state_cache`` is advanced in place.
    """
    validate_kda_inputs(
        q,
        k,
        v,
        gate,
        beta,
        None,
        cu_seqlens,
        op_name="paged_chunk_kda",
        gate_name="gate",
    )
    validate_paged_state(
        q,
        v,
        state_cache,
        cu_seqlens,
        state_indices,
        has_initial_state=has_initial_state,
    )
    options = resolve_kernel_options(kernel_options, op_name="paged_chunk_kda")
    # Replay runs Q/K/V in BF16, so FP16 replay calls resolve the option as their BF16 cast.
    compute_dtype = kernel_dtype(q, k, v) if replay_state is None else torch.bfloat16
    intra_chunk_dtypes = resolve_intra_chunk_dtypes(
        options.intra_chunk_dtypes, compute_dtype, op_name="paged_chunk_kda"
    )
    if options.split_backward or options.split_forward:
        raise ValueError("split schedules are not supported by paged_chunk_kda")
    validate_state_checkpoints(state_checkpoints, q, v, cu_seqlens)
    backend = resolve_chunk_backend(options.backend, q, fused_only=state_checkpoints is not None)
    if state_checkpoints is not None and backend != "fused":
        raise ValueError("state_checkpoints requires the fused backend")
    if replay_state is not None:
        replay_state = ReplayState(*replay_state)
        validate_replay_state(replay_state, q, v, state_cache.shape[0])
    if backend == "warpdrive":
        return _warpdrive_paged_chunk_forward(
            q,
            k,
            v,
            gate,
            beta,
            state_cache,
            state_indices,
            cu_seqlens=cu_seqlens,
            has_initial_state=has_initial_state,
            scale=resolve_scale(None, q.shape[-1]),
            replay_state=replay_state,
            intra_chunk_dtypes=intra_chunk_dtypes,
            batch_invariant=options.batch_invariant,
        )
    return _fused_paged_chunk_forward(
        q,
        k,
        v,
        gate,
        beta,
        state_cache,
        state_indices,
        cu_seqlens=cu_seqlens,
        has_initial_state=has_initial_state,
        autotune=should_autotune(options.autotune),
        schedule=options.schedule,
        replay_state=replay_state,
        state_checkpoints=state_checkpoints,
    )


def copy_replay_tails(
    replay_state: ReplayState,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    checkpoint_positions: torch.Tensor,
    checkpoint_slots: torch.Tensor,
    has_initial_state: torch.Tensor | None = None,
) -> None:
    """Copy the unfinished KDA chunk for each selected prefix into its destination replay slot.

    Call immediately **before** packed ``paged_chunk_kda`` on the same stream, with the same
    Q, K, V, gate, beta inputs, replay state, packed offsets and source routing. The helper
    reads the pre-call counts (zero for a fresh route). Pass the positions and slots used by
    ``StateCheckpoints``; publish the destinations only after both calls complete. This op
    copies replay data, not the recurrent boundary state.

    Positions count new tokens in this call. With 19 buffered tokens and position 50, the
    last completed boundary is at new-token position 45: copy input rows [45:50] and set the
    count to five. At position 45, just reset the count. Inactive routes, non-positive
    destinations and prefixes that reach no boundary in this call do not write.

    The contiguous int32 position and slot vectors have one entry per sequence. Written slots
    must be in bounds, distinct, and disjoint from **all** active source slots. The replay
    buffers may share packed pages but their rows must not overlap one another or the inputs.
    Mixed-dtype views of one allocation work in eager mode and CUDA graphs. For
    ``torch.compile(fullgraph=True)``, use separate backing allocations: AOTAutograd cannot
    functionalize mutations of differently typed views on the same storage. This inference-only
    CUDA operation updates the destination buffers and counts in place.
    """
    require_inference_only((*inputs, *replay_state), "copy_replay_tails is inference-only")
    q, k, v, gate, beta = inputs
    copy_replay_tails_op(
        q,
        k,
        v,
        gate,
        beta,
        *replay_state,
        cu_seqlens,
        state_indices,
        has_initial_state,
        checkpoint_positions,
        checkpoint_slots,
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
    scale: float | None = None,
    output_final_state: bool = False,
    state_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    impl: Impl | str = Impl.FUSED,
    kernel_options: RecurrentKernelOptions | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply recurrent KDA for decoding and inference prefill.

    Args:
        q: Queries shaped ``[B, T, HK, K]``, scaled by ``scale`` internally. ``HK``
            may divide the value head count ``H`` for multi-value attention (MVA): each
            block of ``H // HK`` consecutive value heads shares one query/key head, while
            the gate, beta, and state stay per value head.
        k: Keys shaped like ``q``.
        v: Values shaped ``[B, T, H, V]``.
        gate: Finite, nonpositive per-token natural-log decay shaped ``[B, T, H, K]``,
            one decay vector per value head. At each token the previous state is
            multiplied channelwise by ``exp(gate)``. Use the same non-cumulative
            representation for chunked and recurrent execution; recurrent execution has
            no chunk-rebase lower limit.
        beta: Per-token write gate shaped ``[B, T, H]``.
        initial_state: Starting recurrent state, with one ``[H, V, K]`` entry per
            logical sequence.
        cu_seqlens: Packed offsets shaped ``[N + 1]`` for batch-one inputs, as
            contiguous ``int32`` on ``q.device``; they start at zero, never
            decrease, may repeat for empty sequences whose states pass through,
            and may end before ``T``.
        scale: Query scale. Defaults to ``1 / sqrt(K)``.
        output_final_state: Return the final recurrent state with the output. Rejected
            together with ``state_indices``, which advances the pool in place instead.
        state_indices: Contiguous ``int32`` slot indices, one per logical sequence,
            selecting rows of a paged ``initial_state`` pool shaped
            ``[num_slots, H, V, K]``. Each sequence reads and advances
            ``initial_state[state_indices[i]]`` **in place**. An index not in
            [1, num_slots) implies padding and are ignored by the kernel. The active
            indices also have to be unique to prevent two sequences from writing
            the same slot concurrently.
            Example:
                num_slots = 6
                state_indices = [3, -1, 5, 0]

                seq 0 reads and updates initial_state[3]
                seq 1 is padding (index is -1) so this is ignored
                seq 2 reads and updates initial_state[5]
                seq 3 is padding bc 0
        has_initial_state: Optional contiguous boolean mask, one per logical sequence.
            False entries ignore stale contents in the selected slot and start from zero.
        impl: ``"fused"`` uses the inference-only optimized scan; ``"reference"``
            uses differentiable eager PyTorch in FP32, with no automatic
            fallback.
        kernel_options: Fused-scan options, see
            :class:`attn_gym.linear.types.RecurrentKernelOptions`. ``"autotune"`` (default
            ``True``) benchmarks candidate value-tile sizes for non-paged execution;
            winners are cached and reused. Paged execution and ``False`` launch a heuristic
            config, so results are reproducible across processes and autotune cache states.
            That config still follows the batch shape: ``{"batch_invariant": True}`` makes
            a sequence's bits independent of the rest of the batch.

    Returns:
        The output in ``q.dtype`` and either an FP32 final state with one entry
        per logical sequence or ``None``.

    Serving limitations: without ``state_indices`` state rows map directly to logical
    sequences and final states are written out of place, decode preprocessing and scan are
    separate launches, and speculative-decoding rollback is unsupported.
    """
    selected_impl = resolve_impl(impl)
    options = validate_kernel_options(
        kernel_options, RECURRENT_OPTION_SPEC, op_name="recurrent_kda"
    )
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
        allow_grouped_heads=True,
    )
    if state_indices is not None:
        if initial_state is None:
            raise ValueError("state_indices requires initial_state as the paged state pool")
        if output_final_state:
            raise ValueError("state_indices advances the pool in place; drop output_final_state")
        validate_paged_state(q, v, initial_state, cu_seqlens, state_indices, has_initial_state)
        if selected_impl is not Impl.FUSED:
            raise ValueError("state_indices requires impl='fused'")
    elif has_initial_state is not None:
        raise ValueError("has_initial_state requires state_indices")
    scale = resolve_scale(scale, q.shape[-1])
    if selected_impl is Impl.FUSED:
        return _fused_recurrent_forward(
            q,
            k,
            v,
            gate,
            beta,
            initial_state,
            cu_seqlens=cu_seqlens,
            scale=scale,
            output_final_state=output_final_state,
            state_indices=state_indices,
            has_initial_state=has_initial_state,
            autotune=should_autotune(options["autotune"]),
            batch_invariant=options["batch_invariant"],
        )
    return reference_delta_rule(
        naive_recurrent_kda,
        q,
        k,
        v,
        gate.float() * LOG2_E,
        beta,
        compute_dtype=torch.float32,
        scale=scale,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        output_final_state=output_final_state,
    )


def recurrent_kda_decode(
    packed_qkv: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    has_initial_state: torch.Tensor | None = None,
    gate_transform: Literal["bounded", "softplus"] = "bounded",
    lower_bound: float = -5.0,
    scale: float | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one-token paged KDA decode with preprocessing fused into the recurrence.

    Args:
        packed_qkv: Post-convolution QKV shaped ``[B, H * (2 * K + V)]``. Each
            token stores ``[Q for all heads | K for all heads | V for all heads]``;
            within each section, head rows are contiguous.
        raw_gate: Unactivated gate shaped ``[1, B, H, K]``.
        raw_beta: Unactivated write gate shaped ``[1, B, H]``.
        A_log: FP32 per-head log decay parameter shaped ``[H]``.
        dt_bias: FP32 per-head/channel gate bias shaped ``[H, K]``.
        state_cache: FP32 or BF16 paged state pool shaped ``[num_slots, H, V, K]``.
            Slots may have padding between them but each ``[H, V, K]`` row must be dense.
            Recurrence math remains FP32 and the updated state is cast to the pool dtype.
            ``K`` must be at most 256.
            Paged chunk and recurrent prefill use the same ``[H, V, K]`` slot layout, so the
            cache can transition directly from prefill to decode without a layout conversion.
        state_indices: Contiguous int32 slot indices shaped ``[B]``. Non-positive
            indices are padding/null entries: they produce zero output and leave the
            cache untouched. Each positive index must be in ``[1, num_slots)`` and
            unique among active rows because duplicate in-place updates race. These
            value constraints are caller responsibilities and are not host-validated.
        has_initial_state: Optional contiguous boolean mask, one per sequence. False
            entries mark freshly assigned slots whose contents are garbage: the step
            starts from the zero state and overwrites the slot.
        gate_transform: Pointwise gate transform. ``"bounded"`` computes
            ``lower_bound * sigmoid(exp(A_log) * (raw_gate + dt_bias))``;
            ``"softplus"`` computes
            ``-exp(A_log) * softplus(raw_gate + dt_bias)``.
        lower_bound: Finite nonpositive bound used only by the ``"bounded"`` transform.
        scale: Query scale. Defaults to ``1 / sqrt(K)``.
        out: Optional caller-owned contiguous output buffer shaped ``[1, B, H, V]``
            in ``packed_qkv.dtype`` on the same device. When supplied, the kernel
            writes into and returns this exact tensor. It must not alias any input.

    Returns:
        Decode output shaped ``[1, B, H, V]`` in ``packed_qkv.dtype``. This is
        ``out`` itself when a buffer is supplied.

    Q/K L2 normalization, gate activation, beta sigmoid, the recurrent update, and
    the output projection from recurrent state are performed in one Triton kernel.
    The operation is inference-only and advances ``state_cache`` in place.
    """
    batch, heads, value_dim, key_dim = validate_decode_inputs(
        packed_qkv,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        state_cache,
        state_indices,
        has_initial_state,
        op_name="recurrent_kda_decode",
    )
    expected_channels = heads * (2 * key_dim + value_dim)
    if packed_qkv.shape[1] != expected_channels:
        raise ValueError(
            f"packed_qkv must have shape ({batch}, {expected_channels}), "
            f"got {tuple(packed_qkv.shape)}"
        )
    if raw_gate.shape != (1, batch, heads, key_dim) or raw_gate.stride()[2:] != (key_dim, 1):
        raise ValueError(
            f"raw_gate must have shape {(1, batch, heads, key_dim)} and dense [H, K] rows"
        )
    if raw_beta.shape != (1, batch, heads) or raw_beta.stride(2) != 1:
        raise ValueError(f"raw_beta must have shape {(1, batch, heads)} with contiguous heads")
    if (
        dt_bias.shape != (heads, key_dim)
        or dt_bias.dtype != torch.float32
        or not dt_bias.is_contiguous()
    ):
        raise ValueError(f"dt_bias must be contiguous float32 with shape ({heads}, {key_dim})")

    use_lower_bound = _resolve_decode_gate_transform(gate_transform)
    if use_lower_bound:
        if not isinstance(lower_bound, Real) or isinstance(lower_bound, bool):
            raise TypeError("lower_bound must be a real scalar")
        lower_bound = float(lower_bound)
        if (
            lower_bound != lower_bound  # noqa: PLR0124 - compile-safe NaN check
            or lower_bound < -sys.float_info.max
            or lower_bound > 0
        ):
            raise ValueError(f"lower_bound must be finite and nonpositive, got {lower_bound}")
    else:
        lower_bound = 0.0
    scale = resolve_scale(scale, key_dim)

    out = resolve_decode_out(packed_qkv, out, (1, batch, heads, value_dim))

    return _fused_recurrent_decode_forward(
        packed_qkv,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        state_cache,
        state_indices,
        has_initial_state,
        out,
        lower_bound,
        use_lower_bound,
        scale,
    )


__all__ = [
    "Impl",
    "KernelOptions",
    "ReplayState",
    "chunk_kda",
    "copy_replay_tails",
    "paged_chunk_kda",
    "recurrent_kda",
    "recurrent_kda_decode",
]
