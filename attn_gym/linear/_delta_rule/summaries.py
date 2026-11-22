"""Host contracts and work planning for the fused delta-rule affine summaries.

The staged KDA/GDN and context-parallel entry points share these wrappers, which launch Triton
kernels on every CUDA architecture of capability 8.0 or newer.
"""

import torch
from torch._subclasses.fake_tensor import FakeTensor

from attn_gym._backends.cute import get_device_properties
from attn_gym.linear._delta_rule.triton.work_items import compose_work_items, work_table
from attn_gym.utils import cdiv

BT = 64
KEY_DIM = 128
VAL_DIM = 128
SUMMARY_DIM = VAL_DIM + KEY_DIM
_IO_DTYPES = (torch.bfloat16, torch.float16)


# Splitting adds the planner and the fold (two small launches), so a work item must carry
# enough chunks for the shortened chain to pay for them. This floor puts the smallest split at
# the measured crossover, so every split is a win.
MIN_CHUNKS_PER_ITEM = 16
# Both kernels put ranges or work items on grid.z (limit 65535); leave room for the work
# budget, which is at most the SM count.
MAX_RANGES = 65535 - 1024


def plan_work_budget(max_chunks: int, work_tiles: int, sm_count: int) -> int:
    """Return how many work items to scan concurrently for a span of at most ``max_chunks``.

    The chunk recurrence is a serial chain per CTA, so when ``work_tiles`` (column tiles times
    heads) leaves SMs idle the chunks are split into items that each scan from ``[0 | I]`` and
    ``compose_work_items`` folds the range maps. ``max_chunks`` is a static upper bound (the span
    length); the items themselves are cut on the device by ``plan_work_items``.
    """
    return max(1, min(max_chunks // MIN_CHUNKS_PER_ITEM, sm_count // work_tiles))


def _configure_allocator(device: torch.device, capability: tuple[int, int]) -> None:
    """Install the Triton scratch allocator that SM100/SM120 kernels may need (triton#10002)."""
    if capability[0] in (10, 12):
        from attn_gym._backends.triton.utils import configure_triton_allocator

        with torch.cuda.device(device):
            configure_triton_allocator()


@torch.compiler.disable
def build_state_summaries(
    kg: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    cumulative_gate: torch.Tensor,
    bounds: torch.Tensor,
    *,
    fastmath: bool = True,
) -> torch.Tensor:
    """Compute one packed affine state summary per token range of a stream's WY chunk factors.

    Args:
        kg: Gated keys ``k * exp2(gk_last - gk)``, shape ``[1, T, H, 128]``, bf16 or fp16.
        w: WY factor ``w``, same shape and dtype as ``kg``.
        u: WY pseudo-values ``u``, shape ``[1, T, H, 128]``, same dtype as ``kg``.
        cumulative_gate: Cumulative log2 gates, shape ``[1, T, H, 128]``, fp32.
        bounds: ``int32 [R, 2]`` device tensor of half-open token ranges ``[start, stop)`` with
            ``0 <= start <= stop <= T``; ``start == stop`` yields the identity map. A range is
            exact only over whole chunks of one packed sequence: ``start`` on that sequence's
            64-token chunk grid, ``stop`` on the grid or at the sequence's end (a partial last
            chunk is neutralized in-kernel). The values are not checked (that would sync);
            other ranges return a plausible but wrong map.
        fastmath: Use the flushing approximate ``exp2`` for the per-chunk state decay. Pass
            the same value the local chunk pass used so the summary decays match its own.

    Returns:
        FP32 tensor of shape ``[R, H, 256, 128]``, V-first packed as state bias then state
        transition for ``H_out = H_in @ transition + bias``, one per row of ``bounds``.

    The ranges are read on the device, so the launch is CUDA Graph capturable and replayable
    across different ``bounds`` values of the same shape. The tokens right after a partial last
    chunk are read but dropped by selection, so they may hold anything. Supported scope: B=1,
    fixed K=V=128 and BT=64, contiguous CUDA inputs of capability 8.0+. Long ranges split into
    work items when the static budget permits, and the partial maps compose on the device.
    """
    assert kg.ndim == 4, f"kg must be 4D [1, T, H, K], got shape {tuple(kg.shape)}"
    batch, tokens, heads, key_dim = kg.shape
    assert batch == 1, f"build_state_summaries requires B=1, got B={batch}"
    assert tokens > 0, "build_state_summaries requires at least one token"
    assert heads > 0, "build_state_summaries requires at least one head"
    assert key_dim == KEY_DIM, f"build_state_summaries requires K={KEY_DIM}, got {key_dim}"
    assert w.shape == kg.shape, f"w must match kg shape {tuple(kg.shape)}, got {tuple(w.shape)}"
    assert u.shape == (1, tokens, heads, VAL_DIM), (
        f"u must have shape {(1, tokens, heads, VAL_DIM)}, got {tuple(u.shape)}"
    )
    assert cumulative_gate.shape == kg.shape, (
        f"cumulative_gate must match kg shape {tuple(kg.shape)}, "
        f"got {tuple(cumulative_gate.shape)}"
    )
    assert kg.dtype in _IO_DTYPES, f"kg dtype must be bf16 or fp16, got {kg.dtype}"
    assert w.dtype == kg.dtype and u.dtype == kg.dtype, (
        f"kg/w/u dtypes must match, got {kg.dtype}/{w.dtype}/{u.dtype}"
    )
    assert cumulative_gate.dtype == torch.float32, (
        f"cumulative_gate must be fp32, got {cumulative_gate.dtype}"
    )
    assert bounds.ndim == 2 and bounds.shape[1] == 2 and bounds.shape[0] > 0, (
        f"bounds must have shape [R, 2] with R > 0, got {tuple(bounds.shape)}"
    )
    assert bounds.dtype == torch.int32, f"bounds must be int32, got {bounds.dtype}"

    if isinstance(kg, FakeTensor):
        raise TypeError(
            "build_state_summaries does not support torch.export; run the context-parallel "
            "summary eagerly or under CUDA Graph capture"
        )

    assert kg.is_cuda, "build_state_summaries requires CUDA tensors"
    tensors = (("kg", kg), ("w", w), ("u", u), ("cumulative_gate", cumulative_gate))
    for name, tensor in (*tensors, ("bounds", bounds)):
        assert tensor.device == kg.device, f"{name} must be on {kg.device}, got {tensor.device}"
        assert tensor.is_contiguous(), f"{name} must be contiguous, got strides {tensor.stride()}"
    ranges = bounds.shape[0]
    if ranges > MAX_RANGES:
        raise ValueError(f"at most {MAX_RANGES} ranges per launch, got {ranges}")
    properties = get_device_properties(kg.device)
    capability = (properties.major, properties.minor)
    if capability < (8, 0):
        raise ValueError(f"affine_summary_fwd requires CUDA capability 8.0+, got {capability}")
    _configure_allocator(kg.device, capability)
    from attn_gym.linear._delta_rule.triton.affine_summary_fwd import (
        _select_block_columns,
        launch_affine_summary_fwd,
    )

    state_bn = _select_block_columns(heads, capability)
    # The bounds live on the device, so the budget comes from the span's chunk count (a static
    # upper bound) and the device cuts the ranges' actual chunks into that many items.
    budget = plan_work_budget(
        cdiv(tokens, BT), heads * (SUMMARY_DIM // state_bn), properties.multi_processor_count
    )
    work, range_ids = work_table(bounds, budget)
    partials = torch.empty(
        (work.shape[0], heads, SUMMARY_DIM, KEY_DIM), dtype=torch.float32, device=kg.device
    )
    launch_affine_summary_fwd(
        kg,
        w,
        u,
        cumulative_gate,
        work,
        partials,
        capability,
        whole_ranges=range_ids is None,
        fastmath=fastmath,
    )
    return compose_work_items(partials, range_ids, ranges, reverse=False)


@torch.compiler.disable
def build_state_grad_summaries(
    qg: torch.Tensor,
    kg: torch.Tensor,
    w: torch.Tensor,
    dout: torch.Tensor,
    Aqk: torch.Tensor,
    cumulative_gate: torch.Tensor,
    scale: float,
    bounds: torch.Tensor,
    *,
    fastmath: bool = True,
) -> torch.Tensor:
    """Compute one packed reverse affine summary per token range of a stream.

    Args:
        qg: Gated queries, shape ``[1, T, H, 128]``, bf16 or fp16.
        kg: Gated keys, with the same shape and dtype as ``qg``.
        w: WY factor, with the same shape and dtype as ``qg``.
        dout: Output gradient, shape ``[1, T, H, 128]``, same dtype as ``qg``.
        Aqk: Intra-chunk query/key factor, shape ``[1, T, H, 64]``, same dtype as ``qg``.
        cumulative_gate: Cumulative log2 gates, shape ``[1, T, H, 128]``, fp32.
        scale: Query scaling factor used by the local backward recurrence.
        bounds: ``int32 [R, 2]`` device tensor of half-open token ranges ``[start, stop)``; the
            same contract as ``build_state_summaries``.
        fastmath: Select fast or non-fast per-chunk decay, matching the local backward.

    Returns:
        FP32 tensor of shape ``[R, H, 256, 128]``, V-first packed as local bias then reverse
        transition for ``dH_in = dH_out @ transition + bias``, one per row of ``bounds``.

    The ranges are read on the device, so the launch is CUDA Graph capturable and replayable
    across different ``bounds`` values of the same shape. The tokens right after a partial last
    chunk are read but dropped by selection, so they may hold anything. Supported scope: B=1,
    fixed K=V=128 and BT=64, contiguous CUDA inputs of capability 8.0+. Long ranges split into
    work items when the static budget permits, and the partial maps compose on the device.
    """
    assert qg.ndim == 4, f"qg must be 4D [1, T, H, K], got shape {tuple(qg.shape)}"
    batch, tokens, heads, key_dim = qg.shape
    assert batch == 1, f"build_state_grad_summaries requires B=1, got B={batch}"
    assert tokens > 0, "build_state_grad_summaries requires at least one token"
    assert heads > 0, "build_state_grad_summaries requires at least one head"
    assert key_dim == KEY_DIM, f"build_state_grad_summaries requires K={KEY_DIM}, got {key_dim}"
    for name, tensor in (("kg", kg), ("w", w), ("dout", dout)):
        assert tensor.shape == qg.shape, (
            f"{name} must match qg shape {tuple(qg.shape)}, got {tuple(tensor.shape)}"
        )
    assert Aqk.shape == (1, tokens, heads, BT), (
        f"Aqk must have shape {(1, tokens, heads, BT)}, got {tuple(Aqk.shape)}"
    )
    assert cumulative_gate.shape == qg.shape, (
        f"cumulative_gate must match qg shape {tuple(qg.shape)}, "
        f"got {tuple(cumulative_gate.shape)}"
    )
    assert qg.dtype in _IO_DTYPES, f"qg dtype must be bf16 or fp16, got {qg.dtype}"
    for name, tensor in (("kg", kg), ("w", w), ("dout", dout), ("Aqk", Aqk)):
        assert tensor.dtype == qg.dtype, (
            f"{name} dtype must match qg dtype {qg.dtype}, got {tensor.dtype}"
        )
    assert cumulative_gate.dtype == torch.float32, (
        f"cumulative_gate must be fp32, got {cumulative_gate.dtype}"
    )
    assert bounds.ndim == 2 and bounds.shape[1] == 2 and bounds.shape[0] > 0, (
        f"bounds must have shape [R, 2] with R > 0, got {tuple(bounds.shape)}"
    )
    assert bounds.dtype == torch.int32, f"bounds must be int32, got {bounds.dtype}"

    if isinstance(qg, FakeTensor):
        raise TypeError(
            "build_state_grad_summaries does not support torch.export; run the context-parallel "
            "summary eagerly or under CUDA Graph capture"
        )

    assert qg.is_cuda, "build_state_grad_summaries requires CUDA tensors"
    inputs = (
        ("qg", qg),
        ("kg", kg),
        ("w", w),
        ("dout", dout),
        ("Aqk", Aqk),
        ("cumulative_gate", cumulative_gate),
        ("bounds", bounds),
    )
    for name, tensor in inputs:
        assert tensor.device == qg.device, f"{name} must be on {qg.device}, got {tensor.device}"
        assert tensor.is_contiguous(), f"{name} must be contiguous, got strides {tensor.stride()}"
    ranges = bounds.shape[0]
    if ranges > MAX_RANGES:
        raise ValueError(f"at most {MAX_RANGES} ranges per launch, got {ranges}")
    properties = get_device_properties(qg.device)
    capability = (properties.major, properties.minor)
    if capability < (8, 0):
        raise ValueError(f"affine_summary_rev requires CUDA capability 8.0+, got {capability}")
    _configure_allocator(qg.device, capability)
    from attn_gym.linear._delta_rule.triton.affine_summary_rev import (
        _select_block_columns,
        launch_affine_summary_rev,
    )

    state_bn = _select_block_columns(heads, capability)
    # As in the forward: a static budget from the span's chunk count, ranges cut on the device.
    budget = plan_work_budget(
        cdiv(tokens, BT), heads * (SUMMARY_DIM // state_bn), properties.multi_processor_count
    )
    work, range_ids = work_table(bounds, budget)
    partials = torch.empty(
        (work.shape[0], heads, SUMMARY_DIM, KEY_DIM), dtype=torch.float32, device=qg.device
    )
    launch_affine_summary_rev(
        qg,
        kg,
        w,
        dout,
        Aqk,
        cumulative_gate,
        scale,
        work,
        partials,
        capability,
        whole_ranges=range_ids is None,
        fastmath=fastmath,
    )
    # The cotangent flows through the later item first.
    return compose_work_items(partials, range_ids, ranges, reverse=True)
