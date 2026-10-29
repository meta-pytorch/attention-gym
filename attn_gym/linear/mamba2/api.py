"""Public Mamba2 state-space-dual (SSD) operations."""

from __future__ import annotations

import math
from collections.abc import Callable
from itertools import pairwise

import torch
import torch.nn.functional as F

from attn_gym.linear._delta_rule.validation import (
    require_inference_only,
    resolve_decode_out,
    validate_paged_state,
)
from attn_gym.linear.mamba2.ops import chunk_forward as fused_chunk_forward
from attn_gym.linear.mamba2.ops import (
    chunk_fwd_paged_op,
    decode_op,
    recurrent_fwd_no_state_op,
    recurrent_fwd_op,
    recurrent_fwd_paged_op,
)
from attn_gym.linear.mamba2.reference import chunk_forward, recurrent_forward
from attn_gym.linear.mamba2.validation import (
    validate_decode_inputs,
    validate_fused_inputs,
    validate_mamba2_inputs,
    validate_recurrent_fused_inputs,
)
from attn_gym.linear.types import Impl, resolve_impl

Core = Callable[..., tuple[torch.Tensor, torch.Tensor | None]]

NO_DT_LIMIT = (0.0, math.inf)


def resolve_dt(
    dt: torch.Tensor,
    dt_bias: torch.Tensor | None,
    dt_softplus: bool,
    dt_limit: tuple[float, float],
) -> torch.Tensor:
    """Apply the Mamba2 step-size transform: bias, softplus, then clamp to ``dt_limit``."""
    if dt_bias is not None:
        dt = dt + dt_bias
    if dt_softplus:
        dt = F.softplus(dt, threshold=20.0)
    if dt_limit != NO_DT_LIMIT:
        dt = dt.clamp(*dt_limit)
    return dt


def packed_forward(
    core: Core,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Evaluate each packed logical sequence independently through a dense core."""
    offsets = cu_seqlens.tolist()
    if (
        offsets[0] != 0
        or any(begin > end for begin, end in pairwise(offsets))
        or offsets[-1] > x.shape[1]
    ):
        raise ValueError(
            "cu_seqlens offsets must start at zero, be nondecreasing, and end within "
            "the physical token capacity"
        )
    output = torch.zeros_like(x)
    final_state = None
    if output_final_state:
        heads, head_dim, state_dim = x.shape[2], x.shape[3], B.shape[3]
        final_state = (
            x.new_zeros(len(offsets) - 1, heads, head_dim, state_dim)
            if initial_state is None
            else initial_state.clone()
        )
    for sequence, (begin, end) in enumerate(pairwise(offsets)):
        if begin == end:
            continue
        span = slice(begin, end)
        output[:, span], span_state = core(
            x[:, span],
            dt[:, span],
            A,
            B[:, span],
            C[:, span],
            initial_state=None
            if initial_state is None
            else initial_state[sequence : sequence + 1],
            output_final_state=output_final_state,
        )
        if final_state is not None:
            final_state[sequence] = span_state[0]
    return output, final_state


def reference_mamba2(
    core: Core,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None,
    z: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    *,
    dt_bias: torch.Tensor | None,
    dt_softplus: bool,
    dt_limit: tuple[float, float],
    cu_seqlens: torch.Tensor | None,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run an eager core in the recurrence compute dtype under the public contract.

    Packed execution is eager-only because it reads device offsets on the host.
    """
    output_dtype = x.dtype
    compute_dtype = torch.promote_types(x.dtype, torch.float32)
    x, dt, A, B, C = (tensor.to(compute_dtype) for tensor in (x, dt, A, B, C))
    dt = resolve_dt(
        dt, None if dt_bias is None else dt_bias.to(compute_dtype), dt_softplus, dt_limit
    )
    if initial_state is not None:
        initial_state = initial_state.to(compute_dtype)
    with torch.autocast(device_type=x.device.type, enabled=False):
        if cu_seqlens is None:
            y, state = core(
                x,
                dt,
                A,
                B,
                C,
                initial_state=initial_state,
                output_final_state=output_final_state,
            )
        else:
            y, state = packed_forward(
                core, x, dt, A, B, C, initial_state, cu_seqlens, output_final_state
            )
        if D is not None:
            y = y + x * (D[:, None] if D.ndim == 1 else D).to(compute_dtype)
        if z is not None:
            y = y * F.silu(z.to(compute_dtype))
    return y.to(output_dtype), state


def chunk_mamba2(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    *,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = False,
    dt_limit: tuple[float, float] = NO_DT_LIMIT,
    cu_seqlens: torch.Tensor | None = None,
    output_final_state: bool = False,
    impl: Impl | str = Impl.FUSED,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply the chunk-parallel Mamba2 SSD scan for training and prefill.

    For each head ``h`` with B/C group ``g = h // (H // G)``, token ``t`` updates the ``[P, N]``
    state and reads it:

    .. code-block:: text

        state = exp(dt[t, h] * A[h]) * state + dt[t, h] * outer(x[t, h], B[t, g])
        y[t, h] = (state @ C[t, g] + D[h] * x[t, h]) * silu(z[t, h])

    ``dt`` is first transformed as ``clamp(softplus(dt + dt_bias), *dt_limit)``, with the bias,
    softplus, and clamp each optional. The layouts and step-size transform match
    ``mamba_ssm``'s ``mamba_chunk_scan_combined``. FP16 and BF16 inputs use FP32 recurrence math
    and state.

    Args:
        x: Inputs shaped ``[batch, T, H, P]``.
        dt: Per-token, per-head step sizes shaped ``[batch, T, H]``, before the optional
            ``dt_bias``/softplus/``dt_limit`` transform.
        A: Per-head decay rates shaped ``[H]``; Mamba2 uses ``A = -exp(A_log) < 0``.
        B: Input projections shaped ``[batch, T, G, N]`` with ``H % G == 0``. Each block of
            ``H // G`` consecutive heads shares one group.
        C: Output projections shaped like ``B``.
        D: Optional skip weights shaped ``[H]`` or ``[H, P]``.
        z: Optional gate shaped like ``x``; the output is multiplied by ``silu(z)``.
        initial_state: Optional initial state shaped ``[num_states, H, P, N]``, where
            ``num_states`` is ``batch`` or the number of packed sequences.
        dt_bias: Optional per-head bias shaped ``[H]`` added to ``dt``.
        dt_softplus: Apply ``softplus`` to the biased ``dt``.
        dt_limit: Clamp range applied to the transformed ``dt``.
        cu_seqlens: Optional packed offsets shaped ``[num_sequences + 1]`` for batch-one inputs.
            They start at zero, never decrease, and may end before ``T``; output beyond the
            terminal offset is unspecified.
        output_final_state: Return the final state with the output.
        impl: ``"fused"`` (default) uses Triton kernels on CUDA with FP16/BF16 activations and
            a power-of-two ``N`` in [16, 256]: a state scan that splits long sequences into
            parallel segments, chunk-parallel output, and a chunk-parallel backward; packed
            offsets stay on the device. ``"reference"`` uses eager PyTorch.

    Returns:
        The output in ``x.dtype`` and either the final state or ``None``.
    """
    selected_impl = resolve_impl(impl)
    validate_mamba2_inputs(
        x,
        dt,
        A,
        B,
        C,
        D,
        z,
        initial_state,
        dt_bias=dt_bias,
        dt_limit=dt_limit,
        cu_seqlens=cu_seqlens,
    )
    if selected_impl is Impl.FUSED:
        validate_fused_inputs(x, B)
        return fused_chunk_forward(
            x,
            dt,
            A,
            B,
            C,
            D,
            z,
            initial_state,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            dt_limit=dt_limit,
            cu_seqlens=cu_seqlens,
            output_final_state=output_final_state,
        )
    return reference_mamba2(
        chunk_forward,
        x,
        dt,
        A,
        B,
        C,
        D,
        z,
        initial_state,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
        cu_seqlens=cu_seqlens,
        output_final_state=output_final_state,
    )


def paged_chunk_mamba2(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    *,
    cu_seqlens: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = False,
    dt_limit: tuple[float, float] = NO_DT_LIMIT,
) -> torch.Tensor:
    """Apply inference-only chunk Mamba2 while advancing a paged state cache in place.

    Uses :func:`chunk_mamba2`'s math and fused-path requirements (a power-of-two ``N`` in
    [16, 256]), reading and writing each sequence's state in a shared pool instead of returning
    it. Other state widths can prefill through ``recurrent_mamba2(state_indices=...)``.

    Args:
        x, dt, A, B, C, D, z, dt_bias, dt_softplus, dt_limit: As in :func:`chunk_mamba2`.
        state_cache: Mutable FP32 or BF16 state pool shaped ``[num_slots, H, P, N]``. Recurrence
            math stays FP32; BF16 affects persistent storage only. Slots may have padding
            between them but each ``[H, P, N]`` row must be dense.
        state_indices: Contiguous int32 slot indices, one per logical sequence. Positive,
            unique indices select slots to advance; non-positive indices produce zero output
            and leave the cache untouched.
        cu_seqlens: Optional packed offsets shaped ``[num_sequences + 1]`` for batch-one
            inputs. Output rows past ``cu_seqlens[-1]`` hold unspecified values.
        has_initial_state: Optional contiguous boolean mask, one per logical sequence. False
            entries start from zero and overwrite the selected slot.

    Returns:
        The output in ``x.dtype``. ``state_cache`` is advanced in place.
    """
    validate_mamba2_inputs(
        x, dt, A, B, C, D, z, None, dt_bias=dt_bias, dt_limit=dt_limit, cu_seqlens=cu_seqlens
    )
    validate_fused_inputs(x, B)
    # The pool holds one [P, N] state per head, as a delta-rule pool with q = C and v = x does.
    validate_paged_state(C, x, state_cache, cu_seqlens, state_indices, has_initial_state)
    require_inference_only(
        (x, dt, A, B, C, D, z, dt_bias),
        "paged_chunk_mamba2 is inference-only; call it under torch.no_grad()",
    )
    dt_min, dt_max = dt_limit
    return chunk_fwd_paged_op(
        x,
        dt,
        A,
        B,
        C,
        D,
        z,
        state_cache,
        state_indices,
        has_initial_state,
        dt_bias,
        cu_seqlens,
        dt_softplus,
        float(dt_min),
        float(dt_max),
    )


def recurrent_mamba2(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    *,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = False,
    dt_limit: tuple[float, float] = NO_DT_LIMIT,
    cu_seqlens: torch.Tensor | None = None,
    output_final_state: bool = False,
    state_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    impl: Impl | str = Impl.FUSED,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply the Mamba2 SSD recurrence one token at a time, for decoding and inference prefill.

    Uses the same math as :func:`chunk_mamba2`; only the evaluation order differs.

    Args:
        x, dt, A, B, C, D, z, dt_bias, dt_softplus, dt_limit, cu_seqlens: As in
            :func:`chunk_mamba2`.
        initial_state: Initial state shaped ``[num_states, H, P, N]``, or with
            ``state_indices`` the mutable pool shaped ``[num_slots, H, P, N]`` (FP32 or BF16).
        output_final_state: Return the final state with the output. Rejected with
            ``state_indices``, which advances the pool in place instead.
        state_indices: Optional contiguous ``int32`` slot indices, one per logical sequence,
            selecting rows of the ``initial_state`` pool. Positive indices must be unique and in
            ``[1, num_slots)``; non-positive indices produce zero output and leave the pool
            untouched.
        has_initial_state: Optional contiguous boolean mask, one per sequence. False entries
            mark freshly assigned slots: they start from zero and overwrite the slot.
        impl: ``"fused"`` (default) uses the inference-only Triton scan on CUDA with FP16, BF16,
            or FP32 inputs and ``N <= 256``; ``"reference"`` uses eager PyTorch with autograd.

    Returns:
        The output in ``x.dtype`` and either the final state or ``None``.
    """
    selected_impl = resolve_impl(impl)
    validate_mamba2_inputs(
        x,
        dt,
        A,
        B,
        C,
        D,
        z,
        None if state_indices is not None else initial_state,
        dt_bias=dt_bias,
        dt_limit=dt_limit,
        cu_seqlens=cu_seqlens,
    )
    if state_indices is not None:
        if initial_state is None:
            raise ValueError("state_indices requires initial_state as the paged state pool")
        if output_final_state:
            raise ValueError("state_indices advances the pool in place; drop output_final_state")
        if selected_impl is not Impl.FUSED:
            raise ValueError("state_indices requires impl='fused'")
        validate_paged_state(C, x, initial_state, cu_seqlens, state_indices, has_initial_state)
    elif has_initial_state is not None:
        raise ValueError("has_initial_state requires state_indices")

    if selected_impl is Impl.FUSED:
        validate_recurrent_fused_inputs(x, B)
        require_inference_only(
            (x, dt, A, B, C, D, z, initial_state, dt_bias),
            "recurrent_mamba2(impl='fused') is inference-only and has no backward; "
            "call under torch.no_grad() or use chunk_mamba2",
        )
        dt_min, dt_max = (float(limit) for limit in dt_limit)
        transform = (dt_softplus, dt_min, dt_max)
        if state_indices is not None:
            output = recurrent_fwd_paged_op(
                x,
                dt,
                A,
                B,
                C,
                D,
                z,
                initial_state,
                state_indices,
                has_initial_state,
                dt_bias,
                cu_seqlens,
                *transform,
            )
            return output, None
        args = (x, dt, A, B, C, D, z, initial_state, dt_bias, cu_seqlens, *transform)
        if output_final_state:
            return recurrent_fwd_op(*args)
        return recurrent_fwd_no_state_op(*args), None
    return reference_mamba2(
        recurrent_forward,
        x,
        dt,
        A,
        B,
        C,
        D,
        z,
        initial_state,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
        cu_seqlens=cu_seqlens,
        output_final_state=output_final_state,
    )


def recurrent_mamba2_decode(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    *,
    has_initial_state: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = False,
    dt_limit: tuple[float, float] = NO_DT_LIMIT,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Advance a paged Mamba2 state cache by one token per sequence.

    Mirrors ``mamba_ssm``'s ``selective_state_update``: one Triton kernel applies the step-size
    transform, decays and writes the selected slots in place, and reads them with ``C``. Inputs
    may be strided views (for example ``x``, ``B``, and ``C`` sliced from one post-convolution
    projection buffer) as long as their last axis is contiguous.

    Args:
        x: Inputs shaped ``[batch, H, P]`` in FP16, BF16, or FP32.
        dt: Raw step sizes shaped ``[batch, H]``.
        A, D, dt_bias, dt_softplus, dt_limit: As in :func:`chunk_mamba2`.
        B, C: Projections shaped ``[batch, G, N]`` with ``N <= 256``.
        state_cache: FP32 or BF16 state pool shaped ``[num_slots, H, P, N]``; slots may have
            padding between them but each ``[H, P, N]`` row must be dense.
        state_indices: Contiguous int32 slot indices shaped ``[batch]``. Non-positive indices
            produce zero output and leave the cache untouched; positive indices must be unique
            and in ``[1, num_slots)``.
        z: Optional gate shaped like ``x``.
        has_initial_state: Optional contiguous boolean mask shaped ``[batch]``. False entries
            start from the zero state and overwrite the slot.
        out: Optional caller-owned contiguous output buffer shaped ``[batch, H, P]`` in
            ``x.dtype``; when supplied, the kernel writes into and returns it.

    Returns:
        The output shaped ``[batch, H, P]`` in ``x.dtype``. ``state_cache`` is advanced in
        place. The operation is inference-only.
    """
    validate_decode_inputs(x, dt, A, B, C, D, z, dt_bias=dt_bias, dt_limit=dt_limit)
    validate_paged_state(
        C.unsqueeze(1), x.unsqueeze(1), state_cache, None, state_indices, has_initial_state
    )
    require_inference_only(
        (x, dt, A, B, C, D, z, dt_bias),
        "recurrent_mamba2_decode is inference-only; call it under torch.no_grad()",
    )
    out = resolve_decode_out(x, out, tuple(x.shape))
    dt_min, dt_max = dt_limit
    decode_op(
        x,
        dt,
        A,
        B,
        C,
        D,
        z,
        state_cache,
        state_indices,
        has_initial_state,
        dt_bias,
        dt_softplus,
        float(dt_min),
        float(dt_max),
        out,
    )
    return out


__all__ = [
    "chunk_mamba2",
    "paged_chunk_mamba2",
    "recurrent_mamba2",
    "recurrent_mamba2_decode",
    "resolve_dt",
]
