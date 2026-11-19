"""CuTe indexer builds must run on the query GPU, not the DSL's default architecture."""

import pytest
import torch

from attn_gym.sparse import paged_lightning_indexer
from attn_gym.testing.indexer import assert_paged_indexer_selection, make_paged_indexer_test_inputs


@pytest.mark.parametrize("persist", [True, False])
def test_paged_cute_cross_compiles_for_query_device(
    requires_blackwell, monkeypatch, tmp_path, persist
):
    pytest.importorskip("cutlass")
    from cutlass.cutlass_dsl import CuTeDSL

    from attn_gym.sparse.indexer.impl.cute import _compile_topk_kernel

    # Model an SM80 ordinal 0 on an otherwise single-GPU test worker.
    envar = CuTeDSL._get_dsl().envar
    monkeypatch.setattr(envar, "arch", "sm_80")
    monkeypatch.setattr(envar, "keep_ptx", True)
    monkeypatch.setattr(envar, "dump_dir", str(tmp_path))
    monkeypatch.delenv("CUTE_DSL_ARCH", raising=False)
    monkeypatch.setenv("WARPDRIVE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("CUTE_DSL_NO_CACHE", "0" if persist else "1")
    _compile_topk_kernel.cache_clear()
    try:
        args = make_paged_indexer_test_inputs()
        selected = paged_lightning_indexer(*args, topk=7, kernel_options={"backend": "cute"})
        assert_paged_indexer_selection(args, selected, 7)
        major, minor = torch.cuda.get_device_capability(args[0].device)
        (ptx,) = tmp_path.glob("*.ptx")
        assert f".target sm_{major}{minor}a" in ptx.read_text()
        if persist:
            _compile_topk_kernel.cache_clear()
            selected = paged_lightning_indexer(*args, topk=7, kernel_options={"backend": "cute"})
            assert_paged_indexer_selection(args, selected, 7)
            assert _compile_topk_kernel.cache_info().misses == 0
    finally:
        _compile_topk_kernel.cache_clear()
