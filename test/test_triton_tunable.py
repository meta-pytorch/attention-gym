import importlib  # the cute package re-exports tune(), shadowing the tune module

import pytest
import torch
import triton
import triton.language as tl

from attn_gym._backends.autotune import should_autotune
from attn_gym._backends.triton.tune import PinnedConfigKernel, TritonTuner

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@triton.jit
def _accumulate_kernel(x, out, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(out + offsets, tl.load(out + offsets, mask) + tl.load(x + offsets, mask), mask)


def _snapshot_out(_grid, args):
    snapshot = args["out"].clone()
    return lambda: args["out"].copy_(snapshot)


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


@triton.jit
def _write_block_kernel(out, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out + offsets, tl.full((BLOCK,), BLOCK, tl.float32), offsets < n)


def _block_grid(meta):
    return (triton.cdiv(meta["n"], meta["BLOCK"]),)


def _launched_block(kernel, n=1000) -> int:
    out = torch.zeros(n, device="cuda")
    kernel[_block_grid](out=out, n=n)
    return int(out[0])


def test_pinned_launch_uses_the_first_candidate_that_survives_pruning(no_benchmarking):
    blocks = (128, 256, 512)
    tuner = TritonTuner(
        _write_block_kernel,
        [triton.Config({"BLOCK": block}) for block in blocks],
        key=lambda args: (args["n"],),
        prune=lambda configs, args: [c for c in configs if c.kwargs["BLOCK"] >= args["n"] // 8],
    )
    assert _launched_block(tuner.pinned, n=1000) == 128
    # The default follows the arguments: this length prunes the first candidate.
    assert _launched_block(tuner.pinned, n=2048) == 256


def test_pinned_launch_ignores_a_cached_winner(isolated_winners, monkeypatch):
    tune_module = importlib.import_module("attn_gym._backends.cute.tune")
    tuner = TritonTuner(
        _write_block_kernel,
        [triton.Config({"BLOCK": block}) for block in (128, 256)],
        key=lambda args: (args["n"],),
    )
    monkeypatch.setattr(tune_module, "tune", lambda candidates, *_args, **_kwargs: candidates[1])
    assert _launched_block(tuner) == 256
    assert _launched_block(tuner.pinned) == 128


def test_pinned_launch_skips_candidates_exceeding_device_resources(no_benchmarking):
    oversized = triton.Config({"BLOCK": 512}, num_stages=4)  # about 1 MiB of shared memory
    x = torch.ones(1024, device="cuda")
    out = torch.empty(1024, device="cuda")

    def launch(configs):
        tuner = TritonTuner(_row_sums_kernel, configs, key=lambda args: (args["n"],))
        for _ in range(2):  # the second launch reuses the remembered candidate
            out.zero_()
            tuner.pinned[_block_grid](x=x, out=out, n=1024)

    launch([oversized, triton.Config({"BLOCK": 64})])
    torch.testing.assert_close(out, torch.full_like(out, 64 * 64))
    with pytest.raises(triton.OutOfResources):
        launch([oversized])


def test_pinned_autotune_kernel_launches_its_first_config():
    autotuned = triton.autotune(
        [triton.Config({"BLOCK": block}) for block in (128, 256)], key=["n"]
    )(_write_block_kernel)
    assert _launched_block(PinnedConfigKernel(autotuned)) == 128
    with pytest.raises(TypeError, match="expected a tuned kernel"):
        PinnedConfigKernel(_write_block_kernel)


def test_autotuning_follows_the_request_without_deterministic_algorithms():
    assert should_autotune(True)
    assert not should_autotune(False)


def test_deterministic_algorithms_disable_autotuning(deterministic_algorithms):
    assert not should_autotune(True)
