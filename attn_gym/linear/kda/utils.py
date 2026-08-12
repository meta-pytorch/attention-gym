# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
#
# Portions of this file are derived from flash-linear-attention
# (https://github.com/fla-org/flash-linear-attention) and are licensed under
# the MIT license; for the full list of FLA contributors, visit
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
# The remaining portions are licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Consolidated utilities for the KDA kernels (forward + backward). Merges the
# former ``common/utils.py`` (Meta) and ``fla_ops/utils.py`` (FLA) into a single
# module. Targets NVIDIA GPUs only.

import contextlib
import functools
import inspect
import itertools
import logging
import os
import sys
import warnings
from collections import deque
from collections.abc import Callable
from enum import Enum
from functools import cache, lru_cache
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice
from packaging import version
from torch._subclasses.fake_tensor import FakeTensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RCP_LN2 = 1.4426950216
DEFAULT_CHUNK_SIZE = 64

# ---------------------------------------------------------------------------
# Env flags / autotune caching
# ---------------------------------------------------------------------------
FLA_DISABLE_TENSOR_CACHE = os.getenv("FLA_DISABLE_TENSOR_CACHE", "0") == "1"
FLA_CACHE_RESULTS = os.getenv("FLA_CACHE_RESULTS", "1") == "1"

SUPPORTS_AUTOTUNE_CACHE = "cache_results" in inspect.signature(triton.autotune).parameters
autotune_cache_kwargs = {"cache_results": FLA_CACHE_RESULTS} if SUPPORTS_AUTOTUNE_CACHE else {}


# ---------------------------------------------------------------------------
# Environment checks (OS / Triton / Python version)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def check_environments():
    """Warn if OS, Triton, or Python versions are below recommendations.

    Body only runs once due to lru_cache.
    """
    if sys.platform == "win32":
        try:
            from importlib.metadata import PackageNotFoundError, metadata

            metadata("triton-windows")
        except PackageNotFoundError:
            logger.warning(
                "Detected Windows operating system. Consider installing triton-windows "
                "(https://github.com/triton-lang/triton-windows) for better compatibility. "
                "Without it, some features may not work correctly.",
            )

    triton_version = version.parse(triton.__version__)
    required_triton_version = version.parse("3.3.0")
    if triton_version < required_triton_version:
        logger.warning(
            f"Current Triton version {triton_version} is below the recommended 3.3.0 version. "
            "Errors may occur and these issues will not be fixed. "
            "Please consider upgrading Triton.",
        )

    py_version = version.parse(f"{sys.version_info.major}.{sys.version_info.minor}")
    required_py_version = version.parse("3.11")
    if py_version < required_py_version:
        logger.warning(
            f"Current Python version {py_version} is below the recommended 3.11 version. "
            "It is recommended to upgrade to Python 3.11 or higher for the best experience.",
        )


check_environments()


# ---------------------------------------------------------------------------
# Device detection (NVIDIA only)
# ---------------------------------------------------------------------------
def _cpu_device_warning():
    warnings.warn(("Triton is not supported on current platform, roll back to CPU."), stacklevel=2)


@cache
def get_available_device() -> str:
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except Exception:
        _cpu_device_warning()
        return "cpu"


device = "cuda"
device_torch_lib = torch.cuda
device_name = "cuda"
device_platform = get_available_device()

IS_AMD = device_platform == "hip"
IS_NVIDIA = device_platform == "cuda"
IS_NVIDIA_HOPPER = IS_NVIDIA and (
    "NVIDIA H" in torch.cuda.get_device_name(0) or torch.cuda.get_device_capability()[0] >= 9
)
IS_NVIDIA_BLACKWELL = IS_NVIDIA and torch.cuda.get_device_capability()[0] in (10, 12)
IS_TF32_SUPPORTED = IS_NVIDIA and torch.cuda.get_device_capability(0)[0] >= 8
IS_TMA_SUPPORTED = (
    IS_NVIDIA
    and torch.cuda.get_device_capability(0)[0] >= 9
    and os.environ.get("FLA_USE_TMA", "0") == "1"
    and (
        hasattr(triton.language, "_experimental_make_tensor_descriptor")
        or hasattr(triton.language, "make_tensor_descriptor")
    )
)
IS_GATHER_SUPPORTED = hasattr(triton.language, "gather")
USE_CUDA_GRAPH = IS_NVIDIA and os.environ.get("FLA_USE_CUDA_GRAPH", "0") == "1"

# Lowercase aliases
is_nvidia_hopper = IS_NVIDIA_HOPPER
is_gather_supported = IS_GATHER_SUPPORTED
use_cuda_graph = USE_CUDA_GRAPH

if IS_NVIDIA and not IS_TF32_SUPPORTED:
    # Make old cards happy, since triton will use tf32 by default.
    os.environ["TRITON_F32_DEFAULT"] = "ieee"


def _default_alloc_fn(size: int, alignment: int, stream: int | None):
    return torch.empty(
        size, device=torch.device(device_name, device_torch_lib.current_device()), dtype=torch.int8
    )


if IS_TMA_SUPPORTED:
    logger.info("TMA is supported, using TMA by default.")
    triton.set_allocator(_default_alloc_fn)
elif IS_NVIDIA_BLACKWELL:
    # Blackwell (SM100 datacenter / SM120 consumer): Triton compiler may emit global_scratch for
    # autotuned kernels even without TMA. Register a default allocator to prevent NullAllocator
    # crashes. See triton-lang/triton#10002.
    logger.info("Blackwell detected: registering default global_scratch allocator.")
    triton.set_allocator(_default_alloc_fn)


# ---------------------------------------------------------------------------
# Triton math ops
# ---------------------------------------------------------------------------
if os.environ.get("FLA_USE_FAST_OPS", "0") == "1":
    exp2 = tldevice.exp2
    log = tldevice.fast_logf
    log2 = tldevice.fast_log2f
else:
    exp2 = tl.math.exp2
    log = tl.log
    log2 = tl.log2


@triton.jit
def exp(x):
    return tl.exp(x.to(tl.float32))


@triton.jit
def safe_exp(x):
    return exp(tl.where(x <= 0, x, float("-inf")))


if not IS_GATHER_SUPPORTED:

    @triton.jit
    def gather(src, index, axis, _builder=None):
        return None
else:
    gather = tl.gather  # type: ignore


def _generate_softplus(num_pack):
    template = """
        .reg .pred p;
        setp.gt.f32  p, ${in_reg}, 20.;
        @p  mov.f32  ${out_reg}, ${in_reg};
        @!p mul.f32            ${out_reg}, ${in_reg}, 1.4426950408889634;
        @!p ex2.approx.ftz.f32 ${out_reg}, ${out_reg};
        @!p add.f32            ${out_reg}, ${out_reg}, 1.0;
        @!p lg2.approx.ftz.f32 ${out_reg}, ${out_reg};
        @!p mul.f32            ${out_reg}, ${out_reg}, 0.6931471805599453;
    """
    out_str = ""
    for i in range(num_pack):
        inner_str = template.format(out_reg=i, in_reg=i + num_pack)
        out_str += "{" + inner_str + "}\n"
    # flatten out because torch.compile doesn't like newlines
    out_str = " ".join(out_str.split("\n"))
    return out_str


def _generate_constraints(num_pack):
    return ",".join("=r" for i in range(num_pack)) + "," + ",".join("r" for i in range(num_pack))


_NUM_REG = 1
s_softplus: tl.constexpr = tl.constexpr(_generate_softplus(_NUM_REG))
s_constraints: tl.constexpr = tl.constexpr(_generate_constraints(_NUM_REG))
NUM_REG: tl.constexpr = tl.constexpr(_NUM_REG)


@triton.jit
def softplus(x):
    # equivalent to:
    # return tl.where(x < 20.0, tl.math.log(1 + tl.math.exp(x)), x)
    return tl.inline_asm_elementwise(
        asm=s_softplus,
        constraints=s_constraints,
        pack=NUM_REG,
        args=[
            x,
        ],
        dtype=tl.float32,
        is_pure=True,
    )


# ---------------------------------------------------------------------------
# Shared memory check / device context
# ---------------------------------------------------------------------------
class Backend(Enum):
    ADA = 101376  # RTX 4090
    AMPERE = 166912  # A100
    HOPPER = 232448  # H100
    DEFAULT = 102400  # Default

    @classmethod
    def get_shared_memory(cls, arch: str) -> int:
        try:
            return cls[arch.upper()].value
        except KeyError:
            return cls.DEFAULT.value


def get_all_max_shared_mem():
    try:
        return [
            triton.runtime.driver.active.utils.get_device_properties(i)["max_shared_mem"]
            for i in range(device_torch_lib.device_count())
        ]
    except Exception:
        _cpu_device_warning()
        return [-1]


@cache
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    try:
        device_shared_mem_list = get_all_max_shared_mem()
        max_shared_memory = device_shared_mem_list[tensor_idx]
        return max_shared_memory >= Backend.get_shared_memory(arch)
    except Exception:
        return False


def custom_device_ctx(index: int):
    if index is None:
        return contextlib.nullcontext()
    try:
        return device_torch_lib.device(index)
    except (AttributeError, AssertionError, RuntimeError):
        return contextlib.nullcontext()


# ---------------------------------------------------------------------------
# input_guard decorator
# ---------------------------------------------------------------------------
def _skip_contiguous(
    no_guard_contiguous: bool | list[str] | tuple[str, ...] | set[str],
    param_name: str,
    skip_params: set[str],
) -> bool:
    return no_guard_contiguous is True or param_name in skip_params


def _contiguous_if_needed(arg: Any, skip: bool) -> Any:
    if isinstance(arg, torch.Tensor) and not skip:
        return arg.contiguous()
    return arg


def input_guard(
    fn: Callable[..., torch.Tensor] | None = None,
    *,
    no_guard_contiguous: bool | list[str] | tuple[str, ...] | set[str] = False,
) -> (
    Callable[[Callable[..., torch.Tensor]], Callable[..., torch.Tensor]]
    | Callable[..., torch.Tensor]
):
    """Ensure all input tensors are contiguous and set the device from input tensors.

    Args:
        no_guard_contiguous: If True, skip all contiguous checks. If a
            list/tuple/set of parameter names, skip contiguous checks for those.
    """

    def decorator(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())
        skip_params = (
            set(no_guard_contiguous)
            if isinstance(no_guard_contiguous, (list, tuple, set))
            else set()
        )

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            processed_args = []
            for i, arg in enumerate(args):
                param_name = param_names[i] if i < len(param_names) else f"__arg_{i}"
                processed_args.append(
                    _contiguous_if_needed(
                        arg, _skip_contiguous(no_guard_contiguous, param_name, skip_params)
                    )
                )

            processed_kwargs = {}
            for k, v in kwargs.items():
                processed_kwargs[k] = _contiguous_if_needed(
                    v, _skip_contiguous(no_guard_contiguous, k, skip_params)
                )

            tensor = None
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    tensor = arg
                    break
            if tensor is None:
                for value in kwargs.values():
                    if isinstance(value, torch.Tensor):
                        tensor = value
                        break

            if tensor is not None:
                ctx = custom_device_ctx(tensor.device.index)
            else:
                ctx = contextlib.nullcontext()

            with ctx:
                return fn(*processed_args, **processed_kwargs)

        return wrapper

    # Handle direct usage without parentheses: @input_guard
    if fn is not None:
        return decorator(fn)
    return decorator


# ---------------------------------------------------------------------------
# tensor_cache decorator
# ---------------------------------------------------------------------------
_TENSOR_CACHE_MAXSIZE: int = 2


def tensor_cache(
    fn: Callable[..., torch.Tensor] | None = None,
    *,
    maxsize: int | None = None,
) -> Callable[..., torch.Tensor]:
    """FIFO cache for functions with tensor inputs."""

    def decorator(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        effective_maxsize = maxsize if maxsize is not None else _TENSOR_CACHE_MAXSIZE
        cache: deque[tuple[tuple, dict, Any]] = deque(maxlen=effective_maxsize)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if FLA_DISABLE_TENSOR_CACHE:
                return fn(*args, **kwargs)
            for cached_args, cached_kwargs, cached_result in reversed(cache):
                if (
                    len(args) == len(cached_args)
                    and len(kwargs) == len(cached_kwargs)
                    and all(a is b for a, b in zip(args, cached_args, strict=False))
                    and all(
                        k in cached_kwargs and v is cached_kwargs[k] for k, v in kwargs.items()
                    )
                ):
                    return cached_result
            result = fn(*args, **kwargs)
            cache.append((args, kwargs, result))
            return result

        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


def tensor_tree_nbytes(value: Any) -> int:
    """Count unique tensor storage bytes in a nested structure."""
    seen_storages: set[tuple[str, int | None, int]] = set()

    def _tensor_nbytes(tensor: torch.Tensor) -> int:
        if isinstance(tensor, FakeTensor):
            return tensor.numel() * tensor.element_size()
        storage = tensor.untyped_storage()
        storage_key = (tensor.device.type, tensor.device.index, storage.data_ptr())
        if storage_key in seen_storages:
            return 0
        seen_storages.add(storage_key)
        return storage.nbytes()

    def _walk(obj: Any) -> int:
        if isinstance(obj, torch.Tensor):
            return _tensor_nbytes(obj)
        if isinstance(obj, dict):
            return sum(_walk(item) for item in obj.values())
        if isinstance(obj, (list, tuple, set)):
            return sum(_walk(item) for item in obj)
        return 0

    return _walk(value)


def chunk_local_cumsum_reference(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure PyTorch chunk-local cumsum used to model the stripped gate wrapper."""
    out = torch.empty_like(g)
    g_f = g.float()

    def _fill_range(batch_idx: int, bos: int, eos: int) -> None:
        for chunk_start in range(bos, eos, chunk_size):
            chunk_end = min(chunk_start + chunk_size, eos)
            block = g_f[batch_idx, chunk_start:chunk_end]
            if reverse:
                block = block.flip(0).cumsum(0).flip(0)
            else:
                block = block.cumsum(0)
            if scale is not None:
                block = block * scale
            out[batch_idx, chunk_start:chunk_end] = block.to(out.dtype)

    if cu_seqlens is None:
        for batch_idx in range(g.shape[0]):
            _fill_range(batch_idx, 0, g.shape[1])
    else:
        offsets = cu_seqlens.tolist()
        for bos, eos in itertools.pairwise(offsets):
            _fill_range(0, bos, eos)
    return out


# ---------------------------------------------------------------------------
# Varlen chunk indexing
# ---------------------------------------------------------------------------
class ChunkMetadata(NamedTuple):
    """Internal launch metadata; public callers provide only ``cu_seqlens``.

    ``chunk_indices`` maps global work IDs to sequence-local chunks, while the device
    ``num_chunks`` scalar lets persistent kernels replay without a host transfer.
    """

    cu_seqlens: torch.Tensor
    chunk_indices: torch.Tensor
    num_chunks: torch.Tensor

    @property
    def has_multiple_sequences(self) -> bool:
        """Return whether kernels must honor an internal sequence boundary."""
        return self.cu_seqlens.shape[0] > 2


@triton.jit(debug=True)
def _prepare_complete_chunk_metadata_kernel(
    cu_seqlens,
    chunk_indices,
    num_chunks,
    num_sequences,
    tokens: tl.constexpr,
    chunk_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lanes = tl.arange(0, BLOCK)
    sequence_base = 0
    while sequence_base < num_sequences:
        sequence = sequence_base + lanes
        sequence_mask = sequence < num_sequences
        begin = tl.load(cu_seqlens + sequence, mask=sequence_mask, other=0)
        end = tl.load(cu_seqlens + sequence + 1, mask=sequence_mask, other=0)
        valid = (
            (begin >= 0)
            & (begin <= end)
            & (end <= tokens)
            & (begin % chunk_size == 0)
            & (end % chunk_size == 0)
            & ((sequence != 0) | (begin == 0))
            & ((sequence != num_sequences - 1) | (end == tokens))
        )
        tl.device_assert(valid, "invalid packed cu_seqlens", mask=sequence_mask)

        first_chunk = begin // chunk_size
        chunks = (end - begin) // chunk_size
        local_chunk = 0
        while local_chunk < tl.max(tl.where(sequence_mask & valid, chunks, 0)):
            store_mask = sequence_mask & valid & (local_chunk < chunks)
            output = (first_chunk + local_chunk) * 2
            tl.store(chunk_indices + output, sequence, mask=store_mask)
            tl.store(chunk_indices + output + 1, local_chunk, mask=store_mask)
            local_chunk += 1
        sequence_base += BLOCK

    tl.store(num_chunks + lanes, tokens // chunk_size, mask=lanes == 0)


def prepare_complete_chunk_metadata(
    cu_seqlens: torch.Tensor,
    tokens: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate packed boundaries and construct their complete-chunk work map."""
    chunk_indices = torch.empty(
        (tokens // chunk_size, 2),
        dtype=torch.int32,
        device=cu_seqlens.device,
    )
    num_chunks = torch.empty((), dtype=torch.int32, device=cu_seqlens.device)
    _prepare_complete_chunk_metadata_kernel[(1,)](
        cu_seqlens,
        chunk_indices,
        num_chunks,
        num_sequences=cu_seqlens.shape[0] - 1,
        tokens=tokens,
        chunk_size=chunk_size,
        BLOCK=256,
        num_warps=8,
    )
    return chunk_indices, num_chunks


@tensor_cache(maxsize=10)
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.diff(cu_seqlens)


@tensor_cache(maxsize=10)
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
    cu_seqlens_cpu: torch.LongTensor | None = None,
) -> torch.LongTensor:
    if isinstance(cu_seqlens, FakeTensor):
        num_seqs = cu_seqlens.shape[0] - 1
        chunk_counts = [1] * num_seqs
    else:
        seqlens_for_lens = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
        lens = prepare_lens(seqlens_for_lens)
        chunk_counts = ((lens + chunk_size - 1) // chunk_size).tolist()
    indices = torch.cat([torch.arange(n) for n in chunk_counts])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@tensor_cache(maxsize=10)
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> torch.LongTensor:
    return F.pad(triton.cdiv(prepare_lens(cu_seqlens), chunk_size), (1, 0), value=0).cumsum(-1)
