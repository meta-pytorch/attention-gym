"""Small utilities shared by hand-written Triton kernels."""

import inspect
import math
from collections.abc import Sequence

import torch
import triton
import triton.language as tl

# Kernels fold natural-log inputs into the faster exp2 with ``exp(x) == exp2(x * LOG2_E)``.
# Wrapped as a constexpr so ``@triton.jit`` functions can reference it as a module global.
LOG2_E = tl.constexpr(math.log2(math.e))

SUPPORTS_AUTOTUNE_CACHE = "cache_results" in inspect.signature(triton.autotune).parameters
autotune_cache_kwargs = {"cache_results": True} if SUPPORTS_AUTOTUNE_CACHE else {}


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
def ptr_offset(indices, strides):
    """Compute a broadcasted linear offset from index and stride tuples.

    Only the tuple arity must be static; stride values may be runtime scalars
    (e.g. a strided token dimension), and compile-time values still fold.
    """
    tl.static_assert(len(indices) == len(strides), "indices and strides must have equal length")
    offset = 0
    for axis in tl.static_range(len(strides)):
        offset += indices[axis] * strides[axis]
    return offset


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
    return any(
        tensor is not None and storage_cosize(tensor.shape, tensor.stride()) > 2**31
        for tensor in tensors
    )


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


class PinnedConfigKernel:
    """An autotuned kernel pinned to its first config, the list's heuristic default.

    Bypasses timing-based selection entirely so launches are reproducible
    across processes, machines, and cache states. Heuristic constexprs
    (``@triton.heuristics``) still apply.
    """

    def __init__(self, kernel):
        from triton.runtime.autotuner import Autotuner, Heuristics

        heuristic_values = None
        if isinstance(kernel, Heuristics):
            heuristic_values = kernel.values
            kernel = kernel.fn
        if not isinstance(kernel, Autotuner):
            raise TypeError(f"expected an autotuned kernel, got {type(kernel).__name__}")
        self._config_kwargs = kernel.configs[0].all_kwargs()
        self._kernel = (
            triton.heuristics(heuristic_values)(kernel.fn) if heuristic_values else kernel.fn
        )

    def __getitem__(self, grid):
        launch = self._kernel[grid]

        def pinned_launch(*args, **kwargs):
            return launch(*args, **kwargs, **self._config_kwargs)

        return pinned_launch
