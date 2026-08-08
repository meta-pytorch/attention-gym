"""Runnable input-aware CuTeDSL compile/cache/tune example.

Run this from an environment containing Attention Gym and CuTeDSL. Reuse the
same cache directory to compare cold and warm behavior::

    ATTN_GYM_CUTE_CACHE_DIR=/tmp/cute-playground \
        python .agents/skills/cutedsl-tunable-kernel-template/copy_reads_example.py
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from typing import Any, NamedTuple

import cutlass
import torch
from cutlass import cute

from attn_gym._backends.cute import benchmark_gpu, compile_tvm_ffi, jit_cache, run_tunable
from attn_gym._backends.cute.cache import get_cache_path

NUM_ELEMENTS = 1 << 24


class ReadConfig(NamedTuple):
    threads: int
    reads: int


class _CopyReadsOp:
    """Internal tunable CuTeDSL op; users call :func:`copy_reads`."""

    default_config = ReadConfig(128, 2)

    @staticmethod
    def configs(
        source: torch.Tensor,
        *_runtime_args: Any,
    ) -> tuple[ReadConfig, ...]:
        """Generate candidates that are sensible for this input shape."""
        max_threads = max(64, min(256, source.numel()))
        reads_per_thread = (1,) if source.numel() < 4 else (1, 2, 4)
        return tuple(
            ReadConfig(threads, reads)
            for threads in (64, 128, 256)
            if threads <= max_threads
            for reads in reads_per_thread
        )

    @staticmethod
    def _name(num_elements: int, threads: int, reads: int) -> str:
        return f"cute_playground_n{num_elements}_t{threads}_r{reads}"

    @cute.kernel
    def _kernel(
        self,
        source: cute.Tensor,
        destination: cute.Tensor,
        threads: cutlass.Constexpr,
        reads: cutlass.Constexpr,
    ):
        """Have each thread copy several values; every iteration is coalesced."""
        tid = cute.arch.thread_idx()[0]
        block = cute.arch.block_idx()[0]
        block_start = block * threads * reads

        # ``reads`` is static, so CuTeDSL unrolls this loop for each config.
        for read in cutlass.range_constexpr(reads):
            index = block_start + read * threads + tid
            if index < cute.size(source):
                destination[index] = source[index]

    @cute.jit
    def _launch(
        self,
        source: cute.Tensor,
        destination: cute.Tensor,
        num_elements: cutlass.Constexpr,
        threads: cutlass.Constexpr,
        reads: cutlass.Constexpr,
        stream,
    ):
        blocks = (num_elements + threads * reads - 1) // (threads * reads)
        self._kernel(
            source,
            destination,
            threads,
            reads,
            _name_prefix=self._name(num_elements, threads, reads),
        ).launch(
            grid=(blocks, 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    @staticmethod
    @jit_cache
    def compile(num_elements: int, config: ReadConfig):
        """Build the fake ABI internally when this specialization misses cache."""
        source = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32,
            (num_elements,),
            stride_order=(0,),
            assumed_align=16,
        )
        destination = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32,
            (num_elements,),
            stride_order=(0,),
            assumed_align=16,
        )
        op = _CopyReadsOp()
        return compile_tvm_ffi(
            op._launch,
            source,
            destination,
            num_elements,
            config.threads,
            config.reads,
            name=op._name(num_elements, config.threads, config.reads),
        )

    @staticmethod
    def compile_call(
        config: ReadConfig,
        source: torch.Tensor,
        _destination: torch.Tensor,
    ) -> tuple[int, ReadConfig]:
        return source.numel(), config

    @staticmethod
    def launch(
        compiled,
        _config: ReadConfig,
        source: torch.Tensor,
        destination: torch.Tensor,
    ) -> torch.Tensor:
        compiled(source, destination)
        return destination


def copy_reads(
    source: torch.Tensor,
    *,
    config: ReadConfig | None = None,
    tune: bool = False,
    configs: Iterable[ReadConfig] | None = None,
    workers: int = 4,
    benchmark: Callable[[Callable[[], Any]], float] | None = None,
) -> torch.Tensor:
    """Copy a real PyTorch tensor using a selected or freshly tuned kernel.

    ``copy_reads(source)`` uses the op's ``default_config``. Pass ``config=...`` to
    force one specialization. ``tune=True`` asks the op to generate candidates
    from the runtime inputs; passing ``configs=...`` overrides those candidates.
    """
    if source.ndim != 1 or source.dtype != torch.float32 or not source.is_cuda:
        raise ValueError("copy_reads expects a 1D CUDA float32 tensor")
    if source.numel() == 0 or not source.is_contiguous() or source.data_ptr() % 16:
        raise ValueError("copy_reads expects nonempty, contiguous, 16-byte-aligned storage")

    destination = torch.empty_like(source)
    output, _selected = run_tunable(
        _CopyReadsOp,
        source,
        destination,
        config=config,
        autotune=tune,
        configs=configs,
        workers=workers,
        benchmark=benchmark,
    )
    return output


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this playground requires a CUDA GPU")

    source = torch.randn(NUM_ELEMENTS, device="cuda", dtype=torch.float32)

    # This wrapper is optional: without benchmark=report, tune() uses the same
    # Inductor benchmark_gpu helper but only returns the winning configuration.
    candidates = _CopyReadsOp.configs(source)
    remaining = iter(candidates)
    timings = {}

    def report(fn):
        config = next(remaining)  # tune benchmarks sequentially in config order
        timings[config] = benchmark_gpu(fn)
        print(f"{config}: {timings[config]:.4f} ms")
        return timings[config]

    started = time.perf_counter()
    destination = copy_reads(
        source,
        tune=True,
        configs=candidates,
        benchmark=report,
        workers=4,
    )
    elapsed = time.perf_counter() - started
    torch.testing.assert_close(destination, source)
    best = min(timings, key=timings.__getitem__)

    print(f"\nbest: {best} ({timings[best]:.4f} ms)")
    print(f"compile + benchmark wall time: {elapsed:.3f} s")
    print(f"cache: {get_cache_path()}")
    print("correctness: output matches input")


if __name__ == "__main__":
    # Give the compiler process an importable module instead of ``__main__``.
    from copy_reads_example import main as importable_main

    importable_main()
