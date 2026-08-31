"""Public gated delta rule operations."""

from __future__ import annotations

import torch

from attn_gym.linear._delta_rule.validation import (
    resolve_decode_out,
    resolve_scale,
    validate_decode_inputs,
    validate_paged_state,
)
from attn_gym.linear.gdn.impl.mega import chunk_forward as mega_chunk_forward
from attn_gym.linear.gdn.impl.reference import chunk_forward, recurrent_forward, reference_gdn
from attn_gym.linear.gdn.ops import recurrent_decode_forward
from attn_gym.linear.gdn.ops import recurrent_forward as fused_recurrent_forward
from attn_gym.linear.gdn.validation import validate_gdn_inputs
from attn_gym.linear.types import Impl, resolve_impl


def chunk_gdn(
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
    impl: Impl | str = Impl.REFERENCE,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply chunk-parallel gated delta rule attention for training and prefill.

    Inputs use the token-major layout ``[batch, sequence, heads, dimension]``. The scalar
    natural-log gate decays the previous state before each beta-scaled delta update, and the query
    reads the updated state. Chunking changes only the decomposition and floating-point order of
    that recurrence. FP16 and BF16 inputs use FP32 recurrence math and state.

    Args:
        q: Queries shaped ``[B, T, HK, K]``. ``HK`` may divide the value head count ``H``
            for grouped-head attention: each block of ``H // HK`` consecutive value heads
            shares one query/key head.
        k: Keys shaped like ``q`` and using the same dtype.
        v: Values shaped ``[B, T, H, V]`` and using the same dtype as ``q``.
        gate: Floating per-token scalar natural-log decay shaped ``[B, T, H]``.
        beta: Floating per-token write gate shaped ``[B, T, H]``.
        initial_state: Initial recurrent state shaped ``[N, H, V, K]`` in the recurrence compute
            dtype, where ``N`` is the number of logical sequences.
        cu_seqlens: Optional packed offsets shaped ``[N + 1]`` for batch-one inputs. They start at
            zero, never decrease, and may end before ``T``; output beyond the terminal offset is
            unspecified.
        scale: Query scale. Defaults to ``1 / sqrt(K)``.
        output_final_state: Return the final recurrent state with the output.
        impl: ``"reference"`` uses eager PyTorch. ``"fused"`` uses the optional CuTeDSL 4.7
            Mega backend on SM100/SM103 with FP16/BF16 QKV and ``K = V = 128``.

    Returns:
        The output in ``q.dtype`` and either the final recurrent state or ``None``.
    """
    selected_impl = resolve_impl(impl)
    validate_gdn_inputs(q, k, v, gate, beta, initial_state, cu_seqlens)
    scale = resolve_scale(scale, q.shape[-1])
    if selected_impl is Impl.FUSED:
        return mega_chunk_forward(
            q,
            k,
            v,
            gate,
            beta,
            initial_state,
            cu_seqlens=cu_seqlens,
            scale=scale,
            output_final_state=output_final_state,
        )

    return reference_gdn(
        chunk_forward,
        q,
        k,
        v,
        gate,
        beta,
        scale=scale,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        output_final_state=output_final_state,
    )


def recurrent_gdn(
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
    autotune: bool = True,
    impl: Impl | str = Impl.FUSED,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply recurrent gated delta rule attention for decoding and inference prefill.

    The recurrence consumes tokens in order, carrying an explicit ``[N, H, V, K]`` state. Inputs
    and outputs use the token-major layout ``[batch, sequence, heads, dimension]``. FP16 and BF16
    inputs use FP32 recurrence math and state.

    Args:
        q: Queries shaped ``[B, T, HK, K]``. ``HK`` may divide the value head count ``H``
            for grouped-head attention: each block of ``H // HK`` consecutive value heads
            shares one query/key head.
        k: Keys shaped like ``q`` and using the same dtype.
        v: Values shaped ``[B, T, H, V]`` and using the same dtype as ``q``.
        gate: Floating per-token scalar natural-log decay shaped ``[B, T, H]``.
        beta: Floating per-token write gate shaped ``[B, T, H]``.
        initial_state: Initial recurrent state shaped ``[N, H, V, K]`` in the recurrence compute
            dtype, where ``N`` is the number of logical sequences.
        cu_seqlens: Optional packed offsets shaped ``[N + 1]`` for batch-one inputs. They start at
            zero, never decrease, and may end before ``T``; output beyond the terminal offset is
            unspecified.
        scale: Query scale. Defaults to ``1 / sqrt(K)``.
        output_final_state: Return the final recurrent state with the output. Rejected with
            ``state_indices``, which advances the state pool in place instead.
        state_indices: Optional contiguous ``int32`` slot indices selecting rows of a mutable
            ``initial_state`` pool shaped ``[num_slots, H, V, K]``. Positive indices must be unique
            and in ``[1, num_slots)``; nonpositive indices produce zero output and leave the pool
            untouched. These value constraints are caller responsibilities.
        has_initial_state: Optional contiguous boolean mask indicating whether each selected slot
            should be loaded. False entries mark freshly assigned slots whose contents are
            garbage: they start from zero and overwrite the slot, even for empty sequences.
        autotune: Benchmark candidate value-tile sizes for non-paged execution when true; paged
            execution always uses deterministic heuristics because it mutates the state cache.
        impl: ``"fused"`` uses the inference-only Triton scan; ``"reference"`` uses eager
            PyTorch with autograd support.

    Returns:
        The output in ``q.dtype`` and either the final recurrent state or ``None``.
    """
    selected_impl = resolve_impl(impl)
    validate_gdn_inputs(
        q,
        k,
        v,
        gate,
        beta,
        None if state_indices is not None else initial_state,
        cu_seqlens,
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

    scale = q.shape[-1] ** -0.5 if scale is None else scale
    if selected_impl is Impl.FUSED:
        return fused_recurrent_forward(
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
            autotune=autotune,
        )

    return reference_gdn(
        recurrent_forward,
        q,
        k,
        v,
        gate,
        beta,
        scale=scale,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        output_final_state=output_final_state,
    )


def recurrent_gdn_decode(
    packed_qkv: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    has_initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one-token paged GDN decode with preprocessing fused into the recurrence.

    One Triton kernel slices the post-convolution QKV buffer, computes the gate as
    ``-exp(A_log) * softplus(raw_gate + dt_bias)`` and the write gate as
    ``sigmoid(raw_beta)``, L2-normalizes q and k, and advances the selected cache
    slots in place, so serving callers launch no separate elementwise kernels.

    Args:
        packed_qkv: Post-convolution QKV shaped ``[B, HK*K + HK*K + H*V]``. Each token
            stores ``[Q for HK heads | K for HK heads | V for H heads]``; within each
            section head rows are contiguous. ``HK`` may divide ``H`` for grouped-head
            attention: each block of ``H // HK`` consecutive value heads shares one
            query/key head.
        raw_gate: Unactivated per-head gate projection shaped ``[1, B, H]``, matching the
            vLLM-style single-token decode convention used by ``recurrent_kda_decode``.
        raw_beta: Unactivated write gate shaped ``[1, B, H]``.
        A_log: FP32 per-head log decay parameter shaped ``[H]``.
        dt_bias: FP32 per-head gate bias shaped ``[H]``.
        state_cache: FP32 paged state pool shaped ``[num_slots, H, V, K]``. Slots may
            have padding between them but each ``[H, V, K]`` row must be dense. ``K``
            must be at most 256.
        state_indices: Contiguous int32 slot indices shaped ``[B]``. Non-positive
            indices are padding/null entries: they produce zero output and leave the
            cache untouched. Each positive index must be in ``[1, num_slots)`` and
            unique among active rows because duplicate in-place updates race. These
            value constraints are caller responsibilities and are not host-validated.
        has_initial_state: Optional contiguous boolean mask, one per sequence. False
            entries mark freshly assigned slots whose contents are garbage: the step
            starts from the zero state and overwrites the slot.
        scale: Query scale. Defaults to ``1 / sqrt(K)``.
        out: Optional caller-owned contiguous output buffer shaped ``[1, B, H, V]`` in
            ``packed_qkv.dtype`` on the same device. When supplied, the kernel writes
            into and returns this exact tensor. It must not alias any input.

    Returns:
        Decode output shaped ``[1, B, H, V]`` in ``packed_qkv.dtype``. This is ``out``
        itself when a buffer is supplied. The operation is inference-only and advances
        ``state_cache`` in place.
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
        op_name="recurrent_gdn_decode",
    )
    qk_channels = packed_qkv.shape[1] - heads * value_dim
    key_heads = qk_channels // (2 * key_dim)
    if (
        qk_channels <= 0
        or key_heads < 1
        or qk_channels != 2 * key_heads * key_dim
        or heads % key_heads != 0
    ):
        raise ValueError(
            f"packed_qkv must have shape [B, 2*HK*{key_dim} + {heads}*{value_dim}] with HK a "
            f"positive divisor of {heads} value heads, got {tuple(packed_qkv.shape)}"
        )
    for name, tensor in (("raw_gate", raw_gate), ("raw_beta", raw_beta)):
        if tensor.shape != (1, batch, heads) or tensor.stride(2) != 1:
            raise ValueError(f"{name} must have shape {(1, batch, heads)} with contiguous heads")
    if dt_bias.shape != (heads,) or dt_bias.dtype != torch.float32 or not dt_bias.is_contiguous():
        raise ValueError(f"dt_bias must be contiguous float32 with shape ({heads},)")

    scale = resolve_scale(scale, key_dim)

    out = resolve_decode_out(packed_qkv, out, (1, batch, heads, value_dim))
    return recurrent_decode_forward(
        packed_qkv,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        state_cache,
        state_indices,
        has_initial_state,
        out,
        scale,
    )


__all__ = ["chunk_gdn", "recurrent_gdn", "recurrent_gdn_decode"]
