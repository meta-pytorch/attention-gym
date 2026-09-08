"""Compile the Blackwell WY backward on any host with CuTeDSL installed."""

from pathlib import Path

import pytest
import torch

from attn_gym._backends.cute import target as cute_target
from attn_gym._backends.cute.compile import precompile_many
from attn_gym._backends.cute.target import CompileTarget


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("ragged", [False, True])
def test_wy_backward_nvvm_compatibility(
    dtype: torch.dtype, ragged: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Catch NVVM binding changes without requiring a Blackwell GPU or a kernel launch."""
    pytest.importorskip("cutlass")
    from attn_gym.linear.kda.bwd.cute.chunk_kda_bwd_wy_dqkg_fused import (
        _compile_chunk_kda_bwd_wy_dqkg,
    )

    target = CompileTarget("cuda", configured_arch="sm_100a", sm_count=132)
    monkeypatch.setattr(cute_target, "_target", target)
    monkeypatch.setenv("ATTN_GYM_CUTE_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("CUTE_DSL_NO_CACHE", raising=False)
    args = (2, 128, 64, dtype, 128**-0.5, True, 1, ragged)
    assert not _compile_chunk_kda_bwd_wy_dqkg.is_cached(*args)
    precompile_many(
        _compile_chunk_kda_bwd_wy_dqkg,
        [args],
        workers=1,
        target=target,
        timeout=45,
    )
    assert _compile_chunk_kda_bwd_wy_dqkg.is_cached(*args)
