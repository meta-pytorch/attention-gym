"""Tests for the cached CuTeDSL KDA inter-solve kernels."""

from __future__ import annotations

import itertools

import pytest
import torch

from attn_gym._backends.arch import cuda_runs_sm100_kernels

pytest.importorskip(
    "attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_inter_solve",
    reason="legacy CuTe kernel is not installed",
)


@pytest.mark.parametrize(
    ("head_dim", "chunk_size", "subchunk_size", "message"),
    (
        (64, 64, 16, "head_dim=128"),
        (128, 32, 16, "chunk_size=64"),
        (128, 64, 8, "subchunk_size=16"),
    ),
)
def test_inter_solve_rejects_unsupported_specializations(
    head_dim,
    chunk_size,
    subchunk_size,
    message,
):
    pytest.importorskip("cutlass")
    from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_inter_solve import (
        _validate_specialization,
    )

    with pytest.raises(AssertionError, match=message):
        _validate_specialization(head_dim, chunk_size, subchunk_size)


def test_ragged_fake_tensors_reject_legacy_boolean_schedule():
    pytest.importorskip("cutlass")
    from torch._subclasses.fake_tensor import FakeTensorMode

    from attn_gym.linear._delta_rule.chunk_schedule import RaggedChunkMetadata
    from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_inter_solve import (
        chunk_kda_fwd_k3b_ragged_cute,
        chunk_kda_fwd_k4b_ragged_cute,
    )

    with FakeTensorMode():
        q = torch.empty(1, 64, 1, 128, device="cuda", dtype=torch.bfloat16)
        g = torch.empty(1, 64, 1, 128, device="cuda")
        beta = torch.empty(1, 64, 1, device="cuda")
        aqk = torch.empty(1, 64, 1, 64, device="cuda", dtype=torch.bfloat16)
        offsets = torch.empty(2, device="cuda", dtype=torch.int32)
        metadata = RaggedChunkMetadata(offsets, offsets, capacity=1, chunk_size=64)

        with pytest.raises(TypeError, match="ScheduleRequest"):
            chunk_kda_fwd_k3b_ragged_cute(
                q,
                q,
                g,
                beta,
                aqk,
                128**-0.5,
                metadata,
                schedule=True,
            )

        offdiagonal = torch.empty(6, 256, device="cuda")
        diagonal = torch.empty(1, 64, 1, 16, device="cuda")
        with pytest.raises(TypeError, match="ScheduleRequest"):
            chunk_kda_fwd_k4b_ragged_cute(
                offdiagonal,
                diagonal,
                metadata,
                schedule=False,
            )


@pytest.mark.skipif(
    not cuda_runs_sm100_kernels(),
    reason="the CuTeDSL KDA inter-solve requires an SM100-family GPU",
)
def test_inter_solve_reuses_compiled_specializations(tmp_path, monkeypatch):
    pytest.importorskip("cutlass")
    from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_inter_solve import (
        _compile_k3b,
        _compile_k4b,
        chunk_kda_fwd_inter_solve_cute,
    )

    monkeypatch.setenv("WARPDRIVE_CACHE_DIR", str(tmp_path / "cache"))
    _compile_k3b.cache_clear()
    _compile_k4b.cache_clear()

    torch.manual_seed(0)
    shape = (1, 128, 1, 128)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    cumulative_gate = (torch.randn(shape, device="cuda") * 0.01).cumsum(1)
    beta = torch.rand(1, 128, 1, device="cuda")
    diagonal_inverse = torch.randn(1, 128, 1, 16, device="cuda") * 0.01

    def run_inter_solve():
        return chunk_kda_fwd_inter_solve_cute(
            q, k, cumulative_gate, beta, diagonal_inverse, 128**-0.5
        )

    run_inter_solve()
    torch.cuda.synchronize()
    first_info = (_compile_k3b.cache_info(), _compile_k4b.cache_info())
    outputs = run_inter_solve()
    torch.cuda.synchronize()
    second_info = (_compile_k3b.cache_info(), _compile_k4b.cache_info())

    assert tuple(output.shape for output in outputs) == ((1, 128, 1, 64),) * 2
    Akk = outputs[1].view(1, 2, 64, 1, 64).permute(0, 1, 3, 2, 4)
    upper = torch.ones(64, 64, dtype=torch.bool, device="cuda").triu(1)
    assert torch.count_nonzero(Akk[..., upper]) == 0
    assert all(
        second.hits == first.hits + 1 and second.currsize == first.currsize
        for first, second in zip(first_info, second_info, strict=True)
    )


@pytest.mark.skipif(
    not cuda_runs_sm100_kernels(),
    reason="the CuTeDSL KDA inter-solve requires an SM100-family GPU",
)
def test_persistent_inter_solve_compile_counts(tmp_path, monkeypatch):
    """K3b's persistent chunk-worker cap is a launch argument: varying it with the capacity, the
    token count and the sequence count reuses one compile. K4b keeps the worker count as a
    compile-time stride, so it compiles once per worker count and reuses that across shapes.
    Every persistent result (several heads) matches the exact grid's on the active rows."""
    from attn_gym.linear._delta_rule.chunk_schedule import ScheduleRequest
    from attn_gym.linear._delta_rule.triton import chunk_scheduler
    from attn_gym.linear._delta_rule.triton.chunk_scheduler import prepare_ragged_chunk_metadata
    from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_inter_solve import (
        _compile_k3b,
        _compile_k4b,
        chunk_kda_fwd_k3b_ragged_cute,
        chunk_kda_fwd_k4b_ragged_cute,
    )
    from attn_gym.testing import cumulative_sequence_offsets

    monkeypatch.setenv("WARPDRIVE_CACHE_DIR", str(tmp_path / "cache"))
    _compile_k3b.cache_clear()
    _compile_k4b.cache_clear()
    heads = 3
    sizes = []
    runs = [(1, [65, 63]), (2, [64] * 5), (2, [1, 200, 0, 7]), (148, [300]), (148, [9] * 40)]
    for workers, lengths in runs:
        monkeypatch.setattr(
            chunk_scheduler.GridScheduler, "num_chunk_workers", lambda self, device, w=workers: w
        )
        tokens = sum(lengths)
        offsets = cumulative_sequence_offsets(lengths)
        # Capacity slack: a graph captured for two more chunks than this replay holds.
        metadata = prepare_ragged_chunk_metadata(offsets, tokens + 128, 64)
        shape = (1, tokens, heads, 128)
        q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        gk = -torch.rand(shape, device="cuda").cumsum(1) / 64
        beta = torch.rand(1, tokens, heads, device="cuda")
        aqk = torch.zeros(1, tokens, heads, 64, device="cuda", dtype=torch.bfloat16)
        diagonal = torch.randn(1, tokens, heads, 16, device="cuda")
        results = {}
        for request in (ScheduleRequest.STATIC, ScheduleRequest.PERSISTENT):
            out_aqk, akk_od = chunk_kda_fwd_k3b_ragged_cute(
                q, k, gk, beta, aqk, 1.0, metadata, request
            )
            akk = chunk_kda_fwd_k4b_ragged_cute(akk_od, diagonal, metadata, request)
            results[request] = out_aqk, akk_od, akk
        rows = metadata.chunk_offsets[-1].item() * 6
        (s_aqk, s_od, s_akk), (p_aqk, p_od, p_akk) = results.values()
        assert torch.equal(s_aqk, p_aqk) and torch.equal(s_akk, p_akk), workers
        assert torch.equal(s_od[:rows], p_od[:rows]), workers
        sizes.append((_compile_k3b.cache_info().currsize, _compile_k4b.cache_info().currsize))
    k3b_sizes, k4b_sizes = zip(*sizes, strict=True)
    assert len(set(k3b_sizes)) == 1, sizes
    # A new K4b compile only for a new worker count: runs 1 and 3.
    assert [b - a for a, b in itertools.pairwise(k4b_sizes)] == [1, 0, 1, 0], sizes


@pytest.mark.skipif(
    not cuda_runs_sm100_kernels(),
    reason="the CuTeDSL KDA inter-solve requires an SM100-family GPU",
)
@pytest.mark.parametrize("request_name", ["STATIC", "PERSISTENT"])
def test_ragged_inter_solve_forced_int64_matches_default(monkeypatch, request_name):
    """The wide-offset ABI of K3b and K4b (forced; real inputs need more than 2**31 elements)
    gives the same outputs as the Int32 ABI on both grids."""
    from attn_gym.linear._delta_rule.chunk_schedule import ScheduleRequest
    from attn_gym.linear._delta_rule.triton import chunk_scheduler
    from attn_gym.linear._delta_rule.triton.chunk_scheduler import prepare_ragged_chunk_metadata
    from attn_gym.linear.kda.fwd.cute import chunk_kda_fwd_inter_solve as inter_solve
    from attn_gym.testing import cumulative_sequence_offsets

    monkeypatch.setattr(chunk_scheduler.GridScheduler, "num_chunk_workers", lambda self, d: 2)
    lengths = [65, 63, 130, 0, 7]
    tokens = sum(lengths)
    metadata = prepare_ragged_chunk_metadata(cumulative_sequence_offsets(lengths), tokens, 64)
    shape = (1, tokens, 2, 128)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    gk = -torch.rand(shape, device="cuda").cumsum(1) / 64
    beta = torch.rand(1, tokens, 2, device="cuda")
    aqk = torch.zeros(1, tokens, 2, 64, device="cuda", dtype=torch.bfloat16)
    diagonal = torch.randn(1, tokens, 2, 16, device="cuda")
    request = ScheduleRequest[request_name]

    def run():
        out_aqk, akk_od = inter_solve.chunk_kda_fwd_k3b_ragged_cute(
            q, k, gk, beta, aqk, 1.0, metadata, request
        )
        return (
            out_aqk,
            akk_od,
            inter_solve.chunk_kda_fwd_k4b_ragged_cute(akk_od, diagonal, metadata, request),
        )

    narrow = run()
    calls = []

    def wide(*tensors):
        calls.append(len(tensors))
        return True

    monkeypatch.setattr(inter_solve, "requires_int64_abi", wide)
    widened = run()
    assert len(calls) == 2  # both kernels took the wide ABI
    rows = metadata.chunk_offsets[-1].item() * 6
    assert torch.equal(narrow[0], widened[0]) and torch.equal(narrow[2], widened[2])
    assert torch.equal(narrow[1][:rows], widened[1][:rows])


@pytest.mark.skipif(
    not cuda_runs_sm100_kernels(),
    reason="the CuTeDSL KDA inter-solve requires an SM100-family GPU",
)
def test_dense_inter_solve_without_chunks_launches_nothing():
    """Zero dense chunks return empty outputs without compiling or launching: a one-item grid of
    an empty space would run one CTA."""
    from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_inter_solve import (
        _compile_k3b,
        _compile_k4b,
        chunk_kda_fwd_inter_solve_cute,
    )

    before = (_compile_k3b.cache_info(), _compile_k4b.cache_info())
    q = torch.empty(1, 0, 2, 128, device="cuda", dtype=torch.bfloat16)
    aqk, akk = chunk_kda_fwd_inter_solve_cute(
        q, q, torch.empty(1, 0, 2, 128, device="cuda"), torch.empty(1, 0, 2, device="cuda"),
        torch.empty(1, 0, 2, 16, device="cuda"), 1.0,
    )  # fmt: skip
    assert aqk.shape == akk.shape == (1, 0, 2, 64)
    after = (_compile_k3b.cache_info(), _compile_k4b.cache_info())
    assert all(a.hits + a.misses == b.hits + b.misses for a, b in zip(before, after, strict=True))
