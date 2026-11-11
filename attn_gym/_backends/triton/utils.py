"""Small utilities shared by hand-written Triton kernels."""

import math
from collections.abc import Sequence

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# Kernels fold natural-log inputs into the faster exp2 with ``exp(x) == exp2(x * LOG2_E)``.
# Wrapped as a constexpr so ``@triton.jit`` functions can reference it as a module global.
LOG2_E = tl.constexpr(math.log2(math.e))
LN_2 = tl.constexpr(math.log(2.0))


def configure_triton_allocator() -> None:
    """Install a scratch allocator when Triton kernels may require one.

    On Blackwell (SM100 datacenter / SM120 consumer) the Triton compiler may emit
    global_scratch even without TMA descriptors; a default allocator prevents
    NullAllocator crashes. See triton-lang/triton#10002.
    """
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] in (10, 12):
        triton.set_allocator(
            lambda size, alignment, stream: torch.empty(
                size, device=torch.device("cuda", torch.cuda.current_device()), dtype=torch.int8
            )
        )


@triton.jit
def exp2(x, FASTMATH: tl.constexpr = True):
    """Select approximate or libdevice exp2.

    Launch with ``enable_reflect_ftz=FASTMATH`` to give libdevice the same math policy.
    """
    if FASTMATH:
        return tl.math.exp2(x)
    return libdevice.exp2(x)


@triton.jit
def ptr_offset(indices, strides):
    """Compute a broadcasted linear offset from index and stride tuples.

    Only the tuple arity must be static; stride values may be runtime scalars
    (e.g. a strided token dimension), and compile-time values still fold.

    Pass strides fixed by the model (head and feature strides) as ``tl.constexpr`` so the
    address math folds. A stride that scales with sequence length, such as a contiguous tensor's
    batch stride, compiles one JIT per new length: pass it at runtime with ``do_not_specialize``,
    or zero it when that axis has size one. Runtime inner strides measured 8-20% slower on the
    instruction-bound softplus gate kernel (GB200, 128M elements), so measure before making a hot
    inner stride dynamic.
    """
    tl.static_assert(len(indices) == len(strides), "indices and strides must have equal length")
    offset = 0
    for axis in tl.static_range(len(strides)):
        offset += indices[axis] * strides[axis]
    return offset


@triton.constexpr_function
def _tree_levels(width: int) -> int:
    return width.bit_length() - 1


@triton.jit
def fixed_tree_sum(x):
    """Sum FP32 values along the last axis in an order fixed by that axis's length alone.

    ``tl.sum``'s association order follows the thread layout, which changes with the tile shape
    and ``num_warps``. Each level here adds adjacent pairs; ``a + b == b + a`` makes every pair
    sum exact regardless of which thread holds which lane, so every launch config yields the same
    bits per slice.

    The last axis must have power-of-two length; callers zero-pad masked lanes. Leading axes are
    independent, and the result drops the last axis. Keep each level a ``tl.sum`` over a pair
    rather than ``tl.split`` plus ``+``: the latter let the compiler fuse a caller's ``x * x`` into
    the first add as an FMA under only some layouts, changing FP32 bits across launch configs.
    The pair ``tl.sum`` does not rule that out for every tile layout: on a transposed tile (the
    recurrent scan's sums over K) products still fused under some layouts, so the scan's
    ``batch_invariant`` mode also launches with ``enable_fp_fusion=False``.
    """
    tl.static_assert(len(x.shape) > 0)
    tl.static_assert(x.dtype == tl.float32)
    width: tl.constexpr = x.shape[-1]
    tl.static_assert(width > 0 and (width & (width - 1)) == 0)
    for _ in tl.static_range(_tree_levels(width)):
        x = tl.sum(tl.reshape(x, x.shape[:-1] + [x.shape[-1] // 2, 2]), axis=-1)
    return tl.sum(x, axis=-1)


@triton.jit
def _document_ids(CuQ, query, num_documents, USE_INT64_OFFSETS: tl.constexpr):
    low = tl.full(query.shape, 0, tl.int32)
    high = tl.full(query.shape, num_documents, tl.int32)
    remaining = num_documents
    # A runtime logarithmic loop works for both scalar queries and vectorized bounds.
    # Right-sided search skips empty documents and returns N for inactive capacity.
    while remaining > 0:
        middle = low + (high - low) // 2
        offset = middle.to(tl.int64) if USE_INT64_OFFSETS else middle
        boundary = tl.load(CuQ + offset + 1, middle < num_documents, 0)
        right = query >= boundary
        active = low < high
        low = tl.where(active & right, middle + 1, low)
        high = tl.where(active & ~right, middle, high)
        remaining = remaining // 2
    return low


def storage_cosize(shape: Sequence[int], strides: Sequence[int]) -> int:
    """Return the storage extent of a nonnegative-strided logical layout.

    Analogous to CuTe's ``cosize``: a layout with a zero-sized dimension has
    extent zero; otherwise the result is one plus the largest offset from the
    logical origin.
    """
    if len(shape) != len(strides):
        raise ValueError("shape and strides must have equal length")

    cosize = 1
    is_empty = False
    for size, stride in zip(shape, strides):
        if size < 0:
            raise ValueError("shape dimensions must be nonnegative")
        if stride < 0:
            raise ValueError("strides must be nonnegative")
        is_empty |= size == 0
        if size > 0:
            cosize += (size - 1) * stride
    return 0 if is_empty else cosize


def requires_int64_offsets(*tensors: torch.Tensor | None) -> bool:
    """Return whether any tensor's relative storage extent exceeds signed int32."""
    for tensor in tensors:
        if tensor is not None and storage_cosize(tensor.shape, tensor.stride()) > 2**31:
            return True
    return False


def can_use_tensor_descriptor(tensor: torch.Tensor) -> bool:
    """Return whether a tensor satisfies Triton tensor-descriptor layout constraints."""
    if not tensor.is_cuda:
        return False

    element_size = tensor.element_size()
    storage_is_aligned = (
        torch.compiler.is_compiling() or tensor.storage_offset() * element_size % 16 == 0
    )
    return (
        tensor.shape[-1] <= 256
        and tensor.stride(-1) == 1
        and storage_is_aligned
        and all(stride * element_size % 16 == 0 for stride in tensor.stride()[:-1])
    )


def can_use_tma(tensor: torch.Tensor) -> bool:
    """Return whether a compatible descriptor can use hardware TMA."""
    return (
        tensor.is_cuda
        and torch.cuda.get_device_capability(tensor.device)[0] >= 9
        and can_use_tensor_descriptor(tensor)
    )


def can_use_tensor_descriptors(*tensors: torch.Tensor) -> bool:
    """Return whether every tensor satisfies the host TMA requirements."""
    return all(can_use_tma(tensor) for tensor in tensors)
