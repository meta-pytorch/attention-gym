import importlib  # the cute package re-exports tune(), shadowing the tune module

import pytest
import torch
import triton
import triton.language as tl

from attn_gym._backends.triton.tune import TritonTuner

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@triton.jit
def _accumulate_kernel(x, out, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(out + offsets, tl.load(out + offsets, mask) + tl.load(x + offsets, mask), mask)


def _snapshot_out(_grid, args):
    snapshot = args["out"].clone()
    return lambda: args["out"].copy_(snapshot)


@pytest.fixture
def isolated_winners(tmp_path, monkeypatch):
    monkeypatch.setenv("ATTN_GYM_CUTE_CACHE_DIR", str(tmp_path))
    tune_module = importlib.import_module("attn_gym._backends.cute.tune")
    monkeypatch.setattr(tune_module, "_WINNERS", {})
    monkeypatch.setattr(tune_module, "_WINNERS_FAST", {})


def test_tuned_triton_kernel_mutates_its_output_exactly_once(isolated_winners):
    accumulate = TritonTuner(
        _accumulate_kernel,
        [triton.Config({"BLOCK": block}) for block in (128, 256, 512)],
        key=lambda args: (args["n"], args["x"].dtype),
        reset=_snapshot_out,
    )
    x = torch.randn(1000, device="cuda")
    initial = torch.randn(1000, device="cuda")
    out = initial.clone()

    def launch():
        accumulate[lambda meta: (triton.cdiv(meta["n"], meta["BLOCK"]),)](x=x, out=out, n=1000)

    launch()  # benchmarks all three candidates, then launches the winner
    torch.testing.assert_close(out, initial + x)
    launch()  # cached winner
    torch.testing.assert_close(out, initial + 2 * x)


@triton.jit
def _row_sums_kernel(x, out, n, BLOCK: tl.constexpr):
    """Stage a BLOCK x BLOCK FP32 dot, whose shared memory grows with BLOCK squared."""
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tile = tl.load(x + rows[:, None] * 0 + tl.arange(0, BLOCK)[None, :], rows[:, None] < n)
    tl.store(out + rows, tl.sum(tl.dot(tile, tl.trans(tile)), axis=1), rows < n)


def test_tuning_skips_candidates_exceeding_device_resources(isolated_winners):
    """A candidate over the shared-memory limit is skipped; if all are, its error surfaces."""
    oversized = triton.Config({"BLOCK": 512}, num_stages=4)  # about 1 MiB of shared memory
    x = torch.ones(1024, device="cuda")
    out = torch.empty(1024, device="cuda")

    def launch(configs):
        tuner = TritonTuner(_row_sums_kernel, configs, key=lambda args: (args["n"],))
        tuner[lambda meta: (triton.cdiv(meta["n"], meta["BLOCK"]),)](x=x, out=out, n=1024)

    launch([oversized, triton.Config({"BLOCK": 64})])
    torch.testing.assert_close(out, torch.full_like(out, 64 * 64))
    with pytest.raises(triton.OutOfResources):
        launch([oversized])
