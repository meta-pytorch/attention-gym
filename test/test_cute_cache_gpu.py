import os
from pathlib import Path

import pytest
import torch

from attn_gym._backends.cute.cache import CacheInfo, jit_cache
from attn_gym._backends.cute.target import detect_compile_target
from attn_gym._backends.cute.tune import tune
from attn_gym._backends.cute.utils import compile_tvm_ffi

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")


_THREADS = 64
_TOY_VARIANTS = (16, 32)


@cute.kernel
def copy_kernel(source: cute.Tensor, destination: cute.Tensor):
    """Copy one tiny vector for cache export/load validation."""
    thread_idx, _, _ = cute.arch.thread_idx()
    if thread_idx < cute.size(source):
        destination[thread_idx] = source[thread_idx]


@cute.jit
def launch_copy(source: cute.Tensor, destination: cute.Tensor, stream):
    """Launch the toy cache validation kernel."""
    copy_kernel.set_name_prefix("attention_gym_cache_copy")
    copy_kernel(source, destination).launch(
        grid=(1, 1, 1),
        block=(_THREADS, 1, 1),
        stream=stream,
    )


@jit_cache
def compile_copy(vector_size: int):
    """Compile one static vector-size variant with a TVM-FFI signature."""
    compile_log = os.getenv("ATTN_GYM_TEST_CUTE_GPU_COMPILE_LOG")
    if compile_log:
        is_bad_fork = getattr(torch.cuda, "_is_in_bad_fork", lambda: False)()
        with Path(compile_log).open("a") as log:
            log.write(f"{os.getpid()},{vector_size},{int(is_bad_fork)}\n")
    source = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (vector_size,),
        stride_order=(0,),
        assumed_align=16,
    )
    destination = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (vector_size,),
        stride_order=(0,),
        assumed_align=16,
    )
    return compile_tvm_ffi(
        launch_copy,
        source,
        destination,
        name=f"attention_gym_cache_copy_n{vector_size}",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_parallel_compile_then_inductor_benchmark(tmp_path, monkeypatch):
    """A CUDA-owning parent benchmarks variants compiled by offline workers."""
    cache_directory = tmp_path / "cache"
    compile_log = tmp_path / "compiles.log"
    monkeypatch.setenv("ATTN_GYM_CUTE_CACHE_DIR", str(cache_directory))
    monkeypatch.setenv("ATTN_GYM_TEST_CUTE_GPU_COMPILE_LOG", str(compile_log))
    monkeypatch.delenv("CUTE_DSL_NO_CACHE", raising=False)
    compile_copy.cache_clear()

    # Initialize CUDA before creating the fresh compiler process. A direct fork would be poisoned.
    tensors = {
        vector_size: (
            torch.randn(vector_size, device="cuda"),
            torch.empty(vector_size, device="cuda"),
        )
        for vector_size in _TOY_VARIANTS
    }
    target = detect_compile_target()

    def launch(compiled, vector_size):
        source, destination = tensors[vector_size]
        compiled(source, destination)

    best = tune(
        _TOY_VARIANTS,
        compile_copy,
        launch,
        workers=len(_TOY_VARIANTS),
        target=target,
    )

    assert best in _TOY_VARIANTS

    compile_records = [line.split(",") for line in compile_log.read_text().splitlines()]
    assert len(compile_records) == len(_TOY_VARIANTS)
    assert all(record[2] == "1" for record in compile_records)
    assert len(list(cache_directory.rglob("*.o"))) == len(_TOY_VARIANTS)
    assert compile_copy.cache_info() == CacheInfo(
        hits=len(_TOY_VARIANTS),
        misses=0,
        currsize=len(_TOY_VARIANTS),
    )

    for vector_size in _TOY_VARIANTS:
        source, destination = tensors[vector_size]
        compile_copy(vector_size)(source, destination)
        torch.testing.assert_close(destination, source)
