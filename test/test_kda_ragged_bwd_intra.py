"""Ragged scheduler tests for the CuTe KDA intra-chunk backward stage."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from attn_gym.linear._delta_rule.triton.chunk_scheduler import prepare_ragged_chunk_metadata
from attn_gym.linear.kda.bwd.cute.chunk_kda_bwd_intra import (
    ChunkKdaBwdIntraConfig,
    _compile_chunk_kda_bwd_intra,
    chunk_kda_bwd_intra,
)
from attn_gym.linear.kda.constants import LOG2_E, MAX_GATE_LOWER_BOUND_MAGNITUDE
from attn_gym.linear.kda.naive import chunk_cumsum_ref
from attn_gym.testing.kda import (
    assert_matches_low_precision_reference,
    assert_relative_rms_within,
    bwd_intra_reference,
    cumulative_sequence_offsets,
    packed_chunk_routes,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="the CuTe KDA backward stage requires CUDA capability 8.0 or newer",
)


@pytest.mark.parametrize("architecture", ["sm_80", "sm_120"])
def test_bwd_intra_compiles_for_portable_targets(architecture: str):
    """Compile the portable specialization without SM100-only instructions."""
    source = """
import cutlass
from attn_gym.linear.kda.bwd.cute import chunk_kda_bwd_intra as module

module._compile_chunk_kda_bwd_intra(
    heads=1,
    ragged=False,
    io_type=cutlass.BFloat16,
    use_int64_offsets=False,
)
"""
    environment = os.environ.copy()
    environment["CUTE_DSL_ARCH"] = architecture
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    subprocess.run([sys.executable, "-c", source], env=environment, check=True, timeout=180)


def _metadata(lengths: list[int]):
    """Build fixed-capacity routing for one packed length distribution."""
    offsets = cumulative_sequence_offsets(lengths)
    return prepare_ragged_chunk_metadata(offsets, sum(lengths), 64)


def _inputs(tokens: int, dtype: torch.dtype = torch.bfloat16) -> tuple[torch.Tensor, ...]:
    """Create deterministic standalone inputs for the intra backward stage."""
    torch.manual_seed(29)
    shape = (1, tokens, 1, 128)
    q = torch.randn(shape, device="cuda", dtype=dtype) / 8
    k = torch.randn_like(q) / 8
    g = -torch.rand(shape, device="cuda")
    beta = torch.rand(1, tokens, 1, device="cuda")
    dAqk = torch.randn(1, tokens, 1, 64, device="cuda") / 16
    dAkk = torch.randn_like(dAqk) / 16
    dq = torch.randn(shape, device="cuda") / 16
    dk = torch.randn_like(dq) / 16
    db = torch.randn_like(beta) / 16
    dg = torch.randn_like(g) / 16
    return q, k, g, beta, dAqk, dAkk, dq, dk, db, dg


def _run(inputs: tuple[torch.Tensor, ...], lengths: list[int]):
    """Run a ragged launch with the full fixed-capacity grid exposed."""
    metadata = _metadata(lengths)
    return chunk_kda_bwd_intra(
        *inputs, metadata, config=ChunkKdaBwdIntraConfig(metadata.capacity), autotune=False
    )


def _sequence_local_reference(
    inputs: tuple[torch.Tensor, ...],
    lengths: list[int],
) -> tuple[torch.Tensor, ...]:
    """Concatenate independent per-sequence launches as the routing oracle."""
    output_parts: list[list[torch.Tensor]] = [[], [], [], []]
    begin = 0
    for length in lengths:
        if length == 0:
            continue
        end = begin + length
        local_inputs = tuple(tensor[:, begin:end].clone() for tensor in inputs)
        for parts, output in zip(output_parts, _run(local_inputs, [length]), strict=True):
            parts.append(output)
        begin = end
    return tuple(torch.cat(parts, dim=1) for parts in output_parts)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_ragged_bwd_intra_matches_independent_numerical_reference(dtype: torch.dtype):
    lengths = [65, 0, 63]
    inputs = list(_inputs(sum(lengths), dtype))
    cu_seqlens = cumulative_sequence_offsets(lengths)
    chunk_routes = packed_chunk_routes(lengths)

    # Invalid columns in each partial chunk must never affect valid gradients.
    token_start = 0
    for length in lengths:
        for chunk in range((length + 63) // 64):
            valid = min(64, length - chunk * 64)
            row_start = token_start + chunk * 64
            inputs[4][:, row_start : row_start + valid, :, valid:] = torch.nan
            inputs[5][:, row_start : row_start + valid, :, valid:] = torch.nan
        token_start += length
    inputs = tuple(inputs)

    actual = _run(inputs, lengths)
    golden_contribution = bwd_intra_reference(
        *(tensor.double() for tensor in inputs[:6]),
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_routes,
    )
    reference_contribution = bwd_intra_reference(
        *inputs[:6],
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_routes,
    )

    def combine(contribution, *, reference: bool):
        dq, dk, db, dg = contribution
        outputs = (
            inputs[6].to(dq.dtype) + dq,
            inputs[7].to(dk.dtype) + dk,
            inputs[9].to(dg.dtype) * math.log(2.0) + dg,
            inputs[8].to(db.dtype) + db,
        )
        if reference:
            return (outputs[0].to(dtype), outputs[1].to(dtype), *outputs[2:])
        return outputs

    golden = combine(golden_contribution, reference=False)
    reference = combine(reference_contribution, reference=True)
    for name, output, golden_output, reference_output in zip(
        ("dq", "dk", "dg", "db"), actual, golden, reference, strict=True
    ):
        assert torch.isfinite(output).all()
        assert_matches_low_precision_reference(
            output,
            golden_output,
            reference_output,
            name,
            source_dtype=dtype,
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_bwd_intra_diagonal_is_gate_independent(dtype: torch.dtype) -> None:
    """Diagonal pairwise terms affect Q/K/beta gradients, but never the gate."""
    inputs = list(_inputs(64, dtype))
    dAqk = torch.zeros_like(inputs[4])
    dAkk = torch.zeros_like(inputs[5])
    diagonal = torch.arange(64, device="cuda")
    dAqk[0, diagonal, 0, diagonal] = torch.linspace(-2, 2, 64, device="cuda")
    dAkk[0, diagonal, 0, diagonal] = torch.linspace(1, -1, 64, device="cuda")
    inputs[4:6] = (dAqk, dAkk)
    inputs[6:] = tuple(torch.zeros_like(tensor) for tensor in inputs[6:])

    dq, dk, dg, db = _run(tuple(inputs), [64])
    reference = bwd_intra_reference(*inputs[:6])
    high_precision = bwd_intra_reference(*(tensor.double() for tensor in inputs[:6]))

    assert torch.count_nonzero(dg) == 0
    for name, actual, expected, low_precision in zip(
        ("dq", "dk", "dbeta"),
        (dq, dk, db),
        high_precision[:3],
        reference[:3],
        strict=True,
    ):
        assert_matches_low_precision_reference(
            actual,
            expected,
            low_precision,
            name,
            source_dtype=dtype,
        )


def _gate_bound_log2_gates(like: torch.Tensor, cu_seqlens: torch.Tensor | None) -> torch.Tensor:
    """Chunk-local cumulative log2 gates that fall at the documented bound on every token."""
    log2_gate = torch.full_like(like, -MAX_GATE_LOWER_BOUND_MAGNITUDE * LOG2_E)
    return chunk_cumsum_ref(log2_gate, 64, cu_seqlens=cu_seqlens)


@pytest.mark.parametrize("fastmath", [True, False], ids=["fastmath", "exact"])
@pytest.mark.parametrize("lengths", [None, [65, 0, 21, 70]], ids=["dense", "ragged"])
@pytest.mark.parametrize(
    ("dtype", "qk_scale"),
    [(torch.bfloat16, 1.0), (torch.float16, 1.0), (torch.bfloat16, 2.0**59)],
    ids=["bf16", "fp16", "bf16-large-qk"],
)
def test_bwd_intra_keeps_subchunk_decays_at_gate_bound(
    dtype: torch.dtype, qk_scale: float, lengths: list[int] | None, fastmath: bool
) -> None:
    """Every gate falls at the documented bound, so a 16-row subchunk spans 128 log2 units.

    Deriving the in-subchunk row factor from the earlier-key one, which falls below 2^-126 on
    the last row, let the flush-to-zero exp2 drop that row's in-subchunk terms; they dominate
    its gate gradient and shift dbeta. In-subchunk factors reach 2^68.3, so Q/K entries near 2^58 must stay finite
    and accurate.
    """
    tokens = 128 if lengths is None else sum(lengths)
    inputs = list(_inputs(tokens, dtype))
    inputs[0] = inputs[0] * qk_scale
    inputs[1] = inputs[1] * qk_scale
    cu_seqlens = None if lengths is None else cumulative_sequence_offsets(lengths)
    inputs[2] = _gate_bound_log2_gates(inputs[2], cu_seqlens)
    inputs[6:] = tuple(torch.zeros_like(tensor) for tensor in inputs[6:])
    metadata = None if lengths is None else _metadata(lengths)

    dq, dk, dg, db = chunk_kda_bwd_intra(*inputs, metadata, fastmath=fastmath, autotune=False)
    # The FP32 reference overflows exp2 at this gate, so bound the error by the FP64 scale.
    expected = bwd_intra_reference(
        *(tensor.double() for tensor in inputs[:6]),
        cu_seqlens=cu_seqlens,
        chunk_indices=None if lengths is None else packed_chunk_routes(lengths),
    )
    # dq/dk round to the source dtype; dbeta and dg are FP32 sums of TF32 products.
    max_error_bounds = {"dq": 2**-7, "dk": 2**-7, "dbeta": 2**-16, "dg": 2**-8}
    for (name, bound), actual, golden in zip(
        max_error_bounds.items(), (dq, dk, db, dg), expected, strict=True
    ):
        error = (actual.double() - golden).abs().max().item()
        assert error <= bound * golden.abs().max().item(), f"{name}: max error {error:.3e}"
        assert_relative_rms_within(actual, golden, name, max_eps=1.0, source_dtype=dtype)


@pytest.mark.parametrize("fastmath", [True, False], ids=["fastmath", "exact"])
@pytest.mark.parametrize(
    ("k_value", "dAqk_value"),
    [(1.0, 1.0), (2.0**58, 1.0), (1.0, 2.0**-60)],
    ids=["unit", "large-k", "small-dA"],
)
def test_bwd_intra_keeps_earlier_subchunk_pair_at_gate_bound(
    k_value: float, dAqk_value: float, fastmath: bool
) -> None:
    """One query/key pair 7 tokens apart across a subchunk boundary, at the gate bound.

    Its decay is 2^-59.7. Rebasing earlier keys below the query subchunk's first row (at its
    middle row, or 64 log2 units lower) flushes this key factor, or the accumulator built from
    it, whatever the K or dA scale; the term must survive at every operand scale.
    """
    shape = (1, 64, 1, 128)
    q = torch.zeros(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    q[0, 16, 0, 0] = 1
    k[0, 9, 0, 0] = k_value
    g = _gate_bound_log2_gates(torch.zeros(shape, device="cuda"), None)
    beta = torch.zeros(1, 64, 1, device="cuda")
    dAqk = torch.zeros(1, 64, 1, 64, device="cuda")
    dAqk[0, 16, 0, 9] = dAqk_value
    dAkk = torch.zeros_like(dAqk)
    running = [torch.zeros(shape, device="cuda") for _ in range(3)]
    dq, dk, dg, db = chunk_kda_bwd_intra(
        q, k, g, beta, dAqk, dAkk, *running[:2], torch.zeros_like(beta), running[2], None,
        fastmath=fastmath, autotune=False
    )  # fmt: skip
    expected = bwd_intra_reference(*(tensor.double() for tensor in (q, k, g, beta, dAqk, dAkk)))
    for name, actual, golden in zip(("dq", "dk", "dbeta", "dg"), (dq, dk, db, dg), expected):
        error = (actual.double() - golden).abs()
        assert (error <= 2**-7 * golden.abs()).all(), f"{name}: max error {error.max():.3e}"


def test_ragged_bwd_intra_rejects_mismatched_chunk_size():
    metadata = prepare_ragged_chunk_metadata(cumulative_sequence_offsets([64]), 64, 32)
    with pytest.raises(ValueError, match="metadata chunk size"):
        chunk_kda_bwd_intra(*_inputs(64), metadata, autotune=False)


@pytest.mark.parametrize("lengths", [[65, 63], [0, 1, 64, 0, 65]])
def test_ragged_bwd_intra_matches_sequence_local_launches(lengths):
    inputs = _inputs(sum(lengths))
    metadata = _metadata(lengths)
    active_chunks = metadata.chunk_offsets[-1].item()
    assert active_chunks <= metadata.capacity
    if 0 in lengths:
        assert active_chunks < metadata.capacity

    config = ChunkKdaBwdIntraConfig(metadata.capacity) if 0 in lengths else None
    actual = chunk_kda_bwd_intra(*inputs, metadata, config=config, autotune=False)
    expected = _sequence_local_reference(inputs, lengths)

    for packed, sequence_local in zip(actual, expected, strict=True):
        torch.testing.assert_close(packed, sequence_local, rtol=0, atol=0)


def test_ragged_bwd_intra_clamps_grid_wider_than_capacity():
    """An explicit grid past the chunk capacity runs exactly like the capacity-wide grid."""
    lengths = [65, 63]
    inputs = _inputs(sum(lengths))
    metadata = _metadata(lengths)
    wide = chunk_kda_bwd_intra(
        *inputs, metadata, config=ChunkKdaBwdIntraConfig(4 * metadata.capacity), autotune=False
    )
    for actual, expected in zip(wide, _run(inputs, lengths), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("metadata", [None, "ragged"], ids=["dense", "ragged"])
def test_bwd_intra_accepts_zero_tokens(metadata):
    inputs = _inputs(0)
    outputs = (
        chunk_kda_bwd_intra(*inputs, None, autotune=False)
        if metadata is None
        else _run(inputs, [0, 0])
    )
    dq, dk, dg, db = outputs

    assert dq.shape == dk.shape == dg.shape == (1, 0, 1, 128)
    assert dq.dtype == dk.dtype == torch.bfloat16
    assert dg.dtype == db.dtype == torch.float32
    torch.testing.assert_close(db, inputs[8], rtol=0, atol=0)


@pytest.mark.parametrize("ragged", [False, True])
def test_bwd_intra_reuses_compile_across_grid_caps_and_capacities(ragged):
    _compile_chunk_kda_bwd_intra.cache_clear()
    launches = 0
    for chunks in (1, 16, 17, 33):
        tokens = chunks * 64
        lengths = [0, 1, tokens - 1, 0] if ragged else None
        metadata = _metadata(lengths) if ragged else None
        inputs = _inputs(tokens)
        expected = chunk_kda_bwd_intra(
            *inputs, metadata, config=ChunkKdaBwdIntraConfig(1), autotune=False
        )
        launches += 1
        for cap in (1, 16, 17, 64):
            actual = chunk_kda_bwd_intra(
                *inputs, metadata, config=ChunkKdaBwdIntraConfig(cap), autotune=False
            )
            launches += 1
            for output, reference in zip(actual, expected, strict=True):
                torch.testing.assert_close(output, reference, rtol=0, atol=0)
            info = _compile_chunk_kda_bwd_intra.cache_info()
            assert info.currsize == 1
            assert info.hits + info.misses == launches


def test_ragged_bwd_intra_replays_aligned_to_partial():
    inputs = _inputs(128)
    cu_seqlens = torch.tensor([0, 64, 128], device="cuda", dtype=torch.int32)
    warm_metadata = prepare_ragged_chunk_metadata(cu_seqlens, 128, 64)
    config = ChunkKdaBwdIntraConfig(warm_metadata.capacity)
    chunk_kda_bwd_intra(*inputs, warm_metadata, config=config, autotune=False)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        metadata = prepare_ragged_chunk_metadata(cu_seqlens, 128, 64)
        actual = chunk_kda_bwd_intra(*inputs, metadata, config=config, autotune=False)

    cu_seqlens.copy_(torch.tensor([0, 65, 128], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()

    expected = _run(inputs, [65, 63])
    assert metadata.chunk_offsets.tolist() == [0, 2, 3]
    for captured, eager in zip(actual, expected, strict=True):
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)
