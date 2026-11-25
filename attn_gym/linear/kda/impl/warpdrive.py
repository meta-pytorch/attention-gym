# SPDX-License-Identifier: BSD-3-Clause

"""Autograd integration for the warpdrive KDA kernels."""

from __future__ import annotations

from functools import partial

import torch

from attn_gym.linear._delta_rule.span import pack_dense_batch
from attn_gym.linear._delta_rule.validation import (
    require_inference_only,
    resolve_scale,
    tracks_grad,
)
from attn_gym.linear.kda.impl.replay import replay_paged_chunk_forward
from attn_gym.linear.kda.impl.warpdrive_ops import (
    chunk_warpdrive_packed_bwd_with_exit_cotangent_op,
    chunk_warpdrive_packed_bwd_with_state_op,
    chunk_warpdrive_packed_fwd_entry_states_op,
    chunk_warpdrive_packed_fwd_op,
    chunk_warpdrive_packed_fwd_paged_op,
    chunk_warpdrive_packed_fwd_with_initial_state_op,
    chunk_warpdrive_packed_fwd_with_state_op,
    chunk_warpdrive_packed_local_bwd_op,
    validate_warpdrive_available,
)
from attn_gym.linear.kda.validation import cast_to_kernel_dtypes
from attn_gym.linear.types import ReplayState


def normalize_output_grad(d_output: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
    """Materialize the contiguous output cotangent required by the warpdrive operator ABI."""
    return torch.zeros_like(value) if d_output is None else d_output.contiguous()


class ChunkKdaWarpdrive(torch.autograd.Function):
    """Attach warpdrive's BT16 backward to the warpdrive forward.

    Optional entry and exit states determine which operator is called. Forward and backward
    normally choose their own plans from the packed shape. When the forward pins its plan with
    ``batch_invariant``, the backward also runs unchained. ``exact_recompute`` keeps the forward
    off the shared prep, so the backward recomputes the forward's states bit for bit: an uncut
    forward hands a chain backward the states entering its pieces (NOTE [Piece entry states] in
    the warpdrive KDA driver), when autograd records the call (``differentiated``): no backward
    reads them otherwise. With an entry state the backward carries the state cotangent in
    FP32 across chunks, pieces and work items.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
        scale: float,
        split_backward: bool,
        split_forward: bool,
        output_final_state: bool,
        fastmath: bool,
        intra_chunk_dtypes: str,
        nonempty: bool,
        batch_invariant: bool,
        exact_recompute: bool,
        differentiated: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Rebuild the forward's operands at the same precision during backward.
        ctx.split_backward = split_backward
        ctx.intra_chunk_dtypes = intra_chunk_dtypes
        ctx.scale, ctx.fastmath = scale, fastmath
        ctx.batch_invariant = batch_invariant
        entry_states = entry_rows = None
        if (
            exact_recompute
            and differentiated
            and not (split_forward or split_backward or batch_invariant)
        ):
            entry_rows = entry_state_rows(gate, value, cu_seqlens)
        if entry_rows is not None:
            output, final_state, entry_states = chunk_warpdrive_packed_fwd_entry_states_op(
                q,
                k,
                value,
                gate,
                beta,
                initial_state,
                cu_seqlens,
                scale,
                fastmath,
                nonempty,
                output_final_state,
                intra_chunk_dtypes,
                entry_rows,
            )
        elif output_final_state:
            # A zero entry state has no cotangent: the backward then skips every state access.
            entry_state = initial_state
            if entry_state is None:
                entry_state = torch.zeros(
                    cu_seqlens.shape[0] - 1,
                    q.shape[2],
                    value.shape[-1],
                    q.shape[3],
                    dtype=torch.float32,
                    device=q.device,
                )
            output, final_state = chunk_warpdrive_packed_fwd_with_state_op(
                q,
                k,
                value,
                gate,
                beta,
                entry_state,
                cu_seqlens,
                scale,
                fastmath,
                nonempty,
                intra_chunk_dtypes,
                batch_invariant,
                exact_recompute,
            )
        elif initial_state is None:
            output = chunk_warpdrive_packed_fwd_op(
                q,
                k,
                value,
                gate,
                beta,
                cu_seqlens,
                split_forward,
                scale,
                fastmath,
                intra_chunk_dtypes,
                batch_invariant,
                exact_recompute,
            )
        else:
            output = chunk_warpdrive_packed_fwd_with_initial_state_op(
                q,
                k,
                value,
                gate,
                beta,
                initial_state,
                cu_seqlens,
                scale,
                fastmath,
                intra_chunk_dtypes,
                batch_invariant,
                exact_recompute,
            )
        ctx.save_for_backward(q, k, value, gate, beta, initial_state, cu_seqlens, entry_states)
        ctx.set_materialize_grads(False)
        if output_final_state:
            return output, final_state
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, d_output: torch.Tensor | None, d_final_state: torch.Tensor | None = None):
        q, k, value, gate, beta, initial_state, cu_seqlens, entry_states = ctx.saved_tensors
        d_output = normalize_output_grad(d_output, value)
        if initial_state is None and d_final_state is None:
            gradients = chunk_warpdrive_packed_local_bwd_op(
                q,
                k,
                value,
                gate,
                beta,
                d_output,
                cu_seqlens,
                ctx.split_backward,
                ctx.scale,
                ctx.fastmath,
                ctx.intra_chunk_dtypes,
                ctx.batch_invariant,
                entry_states,
            )
            gradients = (*gradients, None)
        elif initial_state is None:
            gradients = chunk_warpdrive_packed_bwd_with_exit_cotangent_op(
                q,
                k,
                value,
                gate,
                beta,
                d_output,
                cu_seqlens,
                d_final_state,
                ctx.scale,
                ctx.fastmath,
                ctx.intra_chunk_dtypes,
                ctx.batch_invariant,
                entry_states,
            )
            gradients = (*gradients, None)
        else:
            gradients = chunk_warpdrive_packed_bwd_with_state_op(
                q,
                k,
                value,
                gate,
                beta,
                d_output,
                cu_seqlens,
                initial_state,
                d_final_state,
                ctx.scale,
                ctx.fastmath,
                ctx.intra_chunk_dtypes,
                ctx.batch_invariant,
                entry_states,
            )
        return *gradients, *(None,) * 11


def entry_state_rows(gate: torch.Tensor, value: torch.Tensor, cu_seqlens: torch.Tensor):
    """Rows of the FP32 piece entry states an uncut forward hands a chain backward, for this
    packed ``[1, T, H, D]`` shape; None when no such pair of plans runs. Under ``torch.compile``
    the rows hold any token count's (``entry_state_rows_bound``), so the graph does not guard on
    it."""
    from attn_gym.linear._delta_rule.warpdrive.drivers import kda as driver

    num_seqs, heads = cu_seqlens.shape[0] - 1, gate.shape[2]
    if torch.compiler.is_compiling():
        return driver.entry_state_rows_bound(num_seqs, heads, gate.device)
    pieces = driver.entry_state_pieces(
        gate.shape[1], num_seqs, heads, value.shape[-1], gate.device
    )
    return num_seqs * pieces if pieces else None


def validate_warpdrive_constraints(q: torch.Tensor, value: torch.Tensor) -> None:
    """Validate the device and head dimensions the warpdrive KDA kernels support.

    Dtypes and layouts are normalized instead: ``cast_to_kernel_dtypes`` applies the fused
    dtype policy and the registered operators copy operands their launchers cannot address.
    """
    if not q.is_cuda:
        raise ValueError("the warpdrive KDA backend requires CUDA tensors")
    if q.shape[-1] != 128 or value.shape[-1] != 128:
        raise ValueError("the warpdrive KDA backend requires K=V=128")


def chunk_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    cu_seqlens: torch.Tensor | None = None,
    scale: float | None = None,
    output_final_state: bool = False,
    fastmath: bool = True,
    split_backward: bool = False,
    split_forward: bool = False,
    intra_chunk_dtypes: str,
    batch_invariant: bool,
    exact_recompute: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the optional warpdrive implementation behind the shared ``chunk_kda`` contract.

    ``exact_recompute=None`` (the option omitted) infers it from grad mode (``KernelOptions``).
    """
    scale = resolve_scale(scale, q.shape[-1])
    if split_backward and (initial_state is not None or output_final_state):
        raise ValueError("split_backward currently requires a no-state call")
    if split_forward and (initial_state is not None or output_final_state):
        raise ValueError("split_forward currently requires a no-state call")
    if not torch.compiler.is_compiling():
        validate_warpdrive_available(q)
    validate_warpdrive_constraints(q, value)

    # Inferred while tracing under torch.compile; inference and no_grad keep the shared prep.
    differentiated = tracks_grad((q, k, value, gate, beta, initial_state))
    if exact_recompute is None:
        exact_recompute = differentiated
    output_dtype, output_shape = q.dtype, value.shape
    q, k, value, gate, beta = cast_to_kernel_dtypes(q, k, value, gate, beta)
    # Dense batches with tokens have no empty sequences, so the final state needs no seeding.
    nonempty = cu_seqlens is None and q.shape[1] > 0
    if cu_seqlens is None:
        cu_seqlens, q, k, value, gate, beta = pack_dense_batch(q, k, value, gate, beta)
    if initial_state is not None:
        initial_state = initial_state.float()
        if initial_state.stride(-1) != 1:
            initial_state = initial_state.contiguous()
    result = ChunkKdaWarpdrive.apply(
        q,
        k,
        value,
        gate,
        beta,
        initial_state,
        cu_seqlens,
        scale,
        split_backward,
        split_forward,
        output_final_state,
        fastmath,
        intra_chunk_dtypes,
        nonempty,
        batch_invariant,
        exact_recompute,
        differentiated,
    )
    output, final_state = result if output_final_state else (result, None)
    return output.reshape(output_shape).to(output_dtype), final_state


def paged_chunk_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
    scale: float,
    replay_state: ReplayState | None = None,
    intra_chunk_dtypes: str,
    batch_invariant: bool,
) -> torch.Tensor:
    """Advance selected state-cache slots with the warpdrive KDA kernels, optionally through
    replay-backed decode and prefill; ``batch_invariant`` pins the plans of both."""
    require_inference_only(
        (q, k, value, gate, beta, state_cache, *(replay_state or ())),
        "paged_chunk_kda is inference-only; call under torch.no_grad() or torch.inference_mode()",
    )
    if not torch.compiler.is_compiling():
        validate_warpdrive_available(q)
    validate_warpdrive_constraints(q, value)
    options = {"intra_chunk_dtypes": intra_chunk_dtypes, "batch_invariant": batch_invariant}
    paged_forward = partial(_paged_chunk_state_forward, scale=scale, **options)
    if replay_state is None:
        return paged_forward(
            q,
            k,
            value,
            gate,
            beta,
            state_cache,
            state_indices,
            cu_seqlens=cu_seqlens,
            has_initial_state=has_initial_state,
        )
    return replay_paged_chunk_forward(
        q,
        k,
        value,
        gate,
        beta,
        state_cache,
        state_indices,
        replay_state,
        cu_seqlens=cu_seqlens,
        has_initial_state=has_initial_state,
        chunk_forward=partial(chunk_forward, scale=scale, **options),
        paged_forward=paged_forward,
    )


def _paged_chunk_state_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
    scale: float,
    intra_chunk_dtypes: str,
    batch_invariant: bool,
    state_checkpoints: None = None,
) -> torch.Tensor:
    """Advance selected state-cache slots with the paged warpdrive KDA forward.

    The kernel writes no ``state_checkpoints``; ``paged_chunk_kda`` keeps checkpointed calls on
    the fused backend, so replay passes only ``None``.
    """
    del state_checkpoints
    output_dtype, output_shape = q.dtype, value.shape
    q, k, value, gate, beta = cast_to_kernel_dtypes(q, k, value, gate, beta)
    if cu_seqlens is None:
        cu_seqlens, q, k, value, gate, beta = pack_dense_batch(q, k, value, gate, beta)
    output = chunk_warpdrive_packed_fwd_paged_op(
        q,
        k,
        value,
        gate,
        beta,
        state_cache,
        state_indices,
        has_initial_state,
        cu_seqlens,
        scale,
        intra_chunk_dtypes,
        batch_invariant,
    )
    return output.reshape(output_shape).to(output_dtype)


__all__ = ["chunk_forward", "paged_chunk_forward"]
