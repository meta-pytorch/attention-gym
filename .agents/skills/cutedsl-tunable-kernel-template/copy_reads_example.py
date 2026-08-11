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

# A 256 MiB source keeps the fixed-buffer benchmark larger than modern GPU L2 caches.
NUM_ELEMENTS = 1 << 26


class ReadConfig(NamedTuple):
    threads: int
    reads: int


class CopyReadsTunable:
    """Compile, tune, and launch the copy kernel."""

    default_config = ReadConfig(128, 4)

    @staticmethod
    def configs(source: torch.Tensor, destination: torch.Tensor) -> tuple[ReadConfig, ...]:
        """Generate candidates that are sensible for this input shape."""
        del destination
        max_threads = min(256, max(64, source.numel()))
        reads_per_thread = (4,) if source.numel() < 16 else (4, 8, 16)
        return tuple(
            ReadConfig(threads, reads)
            for threads in (64, 128, 256)
            if threads <= max_threads
            for reads in reads_per_thread
        )

    @staticmethod
    def kernel_name(threads: int, reads: int) -> str:
        return f"cute_playground_t{threads}_r{reads}"

    @cute.kernel
    def kernel(
        self,
        source: cute.Tensor,
        destination: cute.Tensor,
        threads: cutlass.Constexpr,
        reads: cutlass.Constexpr,
    ):
        """Have each thread copy one contiguous vector."""
        tid = cute.arch.thread_idx()[0]
        block = cute.arch.block_idx()[0]
        tile_size = threads * reads

        # Divide the global tensor into CTA tiles whose inner modes describe
        # the thread and its contiguous values, then select this thread's slice.
        thread_value_layout = cute.make_layout((threads, reads), stride=(reads, 1))
        source_tiles = cute.zipped_divide(source, thread_value_layout)
        destination_tiles = cute.zipped_divide(destination, thread_value_layout)
        thread_source = source_tiles[((tid, None), block)]
        thread_destination = destination_tiles[((tid, None), block)]

        block_start = block * tile_size
        if block_start + tile_size <= cute.size(source):
            cute.autovec_copy(thread_source, thread_destination)
        else:
            # Only the final CTA needs elementwise predication.
            for read in cutlass.range_constexpr(reads):
                index = block_start + cute.crd2idx((tid, read), thread_value_layout)
                if index < cute.size(source):
                    destination[index] = source[index]

    @cute.jit
    def execute(
        self,
        source: cute.Tensor,
        destination: cute.Tensor,
        threads: cutlass.Constexpr,
        reads: cutlass.Constexpr,
        stream,
    ):
        blocks = cute.ceil_div(cute.size(source), threads * reads)
        self.kernel(
            source,
            destination,
            threads,
            reads,
            _name_prefix=self.kernel_name(threads, reads),
        ).launch(
            grid=(blocks, 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    @staticmethod
    @jit_cache
    def compile(config: ReadConfig):
        """Build one symbolic-length fake ABI for this codegen config."""
        num_elements = cute.sym_int()
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
        op = CopyReadsTunable()
        return compile_tvm_ffi(
            op.execute,
            source,
            destination,
            config.threads,
            config.reads,
            name=op.kernel_name(config.threads, config.reads),
        )

    @staticmethod
    def compile_call(
        config: ReadConfig,
        source: torch.Tensor,
        destination: torch.Tensor,
    ) -> tuple[ReadConfig]:
        del source, destination
        return (config,)

    @staticmethod
    def launch(
        compiled,
        config: ReadConfig,
        source: torch.Tensor,
        destination: torch.Tensor,
    ) -> torch.Tensor:
        del config
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
    return run_tunable(
        CopyReadsTunable,
        source,
        destination,
        config=config,
        autotune=tune,
        configs=configs,
        workers=workers,
        benchmark=benchmark,
    )[0]


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this playground requires a CUDA GPU")

    source = torch.randn(NUM_ELEMENTS, device="cuda", dtype=torch.float32)

    # This wrapper is optional: without benchmark=report, tune() uses the same
    # Inductor benchmark_gpu helper but only returns the winning configuration.
    candidates = CopyReadsTunable.configs(source, torch.empty_like(source))
    remaining = iter(candidates)
    timings = {}

    def report(fn):
        config = next(remaining)  # tune benchmarks sequentially in config order
        timings[config] = benchmark_gpu(fn)
        effective_gbps = 2 * source.numel() * source.element_size() / timings[config] / 1e6
        print(f"{config}: {timings[config]:.4f} ms ({effective_gbps:.1f} GB/s read+write)")
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

    effective_gbps = 2 * source.numel() * source.element_size() / timings[best] / 1e6
    print(f"\nbest: {best} ({timings[best]:.4f} ms, {effective_gbps:.1f} GB/s read+write)")
    print(f"compile + benchmark wall time: {elapsed:.3f} s")
    print(f"cache: {get_cache_path()}")
    print("correctness: output matches input")


if __name__ == "__main__":
    # Give the compiler process an importable module instead of ``__main__``.
    from copy_reads_example import main as importable_main

    importable_main()
