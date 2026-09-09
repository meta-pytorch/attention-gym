"""Bitwise ownership invariance of the fused canonical-tile staged primitives.

The reference uses independent per-tile calls, including the dense dispatch for complete
single tiles and the production summary defaults used by the prototype. The candidate
packs those same tiles as separate subsequences in one call with pinned summary work.
Random nonzero entry states and exit cotangents exercise both affine boundary terms.
"""

from __future__ import annotations

import math
from itertools import accumulate, pairwise

import pytest
import torch

pytest.importorskip("cutlass")

from attn_gym.linear.kda.stages import (
    ChunkKDAPrepared,
    chunk_kda_prepare,
    chunk_kda_prepare_backward,
)
from attn_gym.testing.kda import make_kda_test_inputs

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8),
    reason="fused canonical tiles require CUDA capability 8.0+",
)


def assert_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    """Check numerical equality and storage bits, including the sign of zero."""
    assert torch.isfinite(actual).all(), name
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=name)
    integer_dtype = torch.int16 if actual.element_size() == 2 else torch.int32
    torch.testing.assert_close(
        actual.contiguous().view(integer_dtype),
        expected.contiguous().view(integer_dtype),
        rtol=0,
        atol=0,
        msg=f"{name}: integer view",
    )


@torch.no_grad()
def run_tiles(
    inputs: tuple[torch.Tensor, ...],
    d_output: torch.Tensor,
    lengths: list[int],
    entry: torch.Tensor,
    exit_grad: torch.Tensor,
    *,
    packed: bool,
) -> dict[str, torch.Tensor]:
    """Snapshot each stage before backward consumes its prepared intermediates."""
    offsets = [0, *accumulate(lengths)]
    cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device="cuda") if packed else None
    prepared = chunk_kda_prepare(
        *inputs,
        cu_seqlens=cu_seqlens,
        autotune=False,
        kernel_options={"backend": "fused"},
    )
    assert isinstance(prepared, ChunkKDAPrepared)
    bounds = torch.tensor(list(pairwise(offsets)), dtype=torch.int32, device="cuda")
    result = {f"factor/{name}": value for name, value in prepared.factors._asdict().items()}
    result["cumulative_gate"] = prepared.saved.cumulative_gate
    result["summary"] = prepared.state_summaries(bounds, deterministic_work=packed)
    output, final_state = prepared.run(entry, output_final_state=True)
    assert final_state is not None
    result["output"] = output
    result["final_state"] = final_state
    backward = chunk_kda_prepare_backward(
        prepared.saved,
        d_output,
        entry,
        scale=prepared.scale,
        autotune=False,
        schedule=prepared.schedule,
    )
    for name in ("w", "qg", "kg", "aqk", "akk", "v_new", "d_aqk"):
        value = vars(backward.prepared)[name]
        assert value is not None
        result[f"backward/{name}"] = value
    assert backward.prepared.h is not None
    # Packed allocation capacity can exceed the active chunks; never inspect padding.
    chunks = sum(math.ceil(length / 64) for length in lengths)
    result["backward/h"] = backward.prepared.h[:, :chunks]
    result["grad_summary"] = backward.state_grad_summaries(bounds, deterministic_work=packed)
    gradients = backward.run(exit_grad)
    for name, value in zip(
        ("dq", "dk", "dv", "dgate", "dbeta", "d_entry"), gradients, strict=True
    ):
        assert value is not None
        result[name] = value
    return result


@pytest.mark.parametrize("tile_size", [64, 128])
@pytest.mark.parametrize("heads", [4, 64])
@pytest.mark.parametrize("gate", ["mild", "strong"])
def test_canonical_tile_batching(tile_size: int, heads: int, gate: str) -> None:
    """Packing rank-owned ragged tiles must preserve every forward/backward stage bit."""
    torch.manual_seed(41)
    doc_offsets = [0, *accumulate([17, 65, 129, 257, 513])]
    tiles = [
        (start, min(start + tile_size, end))
        for begin, end in pairwise(doc_offsets)
        for start in range(begin, end, tile_size)
    ]
    inputs = make_kda_test_inputs(
        doc_offsets[-1],
        heads=heads,
        seed=41,
        normalize_qk=True,
        sigmoid_beta=True,
        gate_scale=math.log(2) if gate == "strong" else 0.02,
        log_uniform_gate=gate == "strong",
    )
    d_output = torch.randn_like(inputs[2])
    entry = torch.randn(len(tiles), heads, 128, 128, device="cuda") * 0.1
    exit_grad = torch.randn_like(entry) * 0.1
    reference = [
        run_tiles(
            tuple(value[:, start:stop].clone() for value in inputs),
            d_output[:, start:stop].clone(),
            [stop - start],
            entry[i : i + 1],
            exit_grad[i : i + 1],
            packed=False,
        )
        for i, (start, stop) in enumerate(tiles)
    ]
    # Three independent copies cross the summary-budget and short-sequence recurrence
    # thresholds. Rotation makes the default budget split inside a two-chunk tile.
    owners = [
        list(range(len(tiles))),
        list(range(0, len(tiles), 2)),
        list(range(1, len(tiles), 2))[::-1],
        (list(range(4, len(tiles))) + list(range(4))) * 3,
    ]
    owners.append([next(i for i, (start, stop) in enumerate(tiles) if stop - start == tile_size)])
    state_names = {"summary", "grad_summary", "final_state", "d_entry"}
    for owned in owners:
        batched = run_tiles(
            tuple(
                torch.cat([value[:, tiles[i][0] : tiles[i][1]] for i in owned], dim=1)
                for value in inputs
            ),
            torch.cat([d_output[:, tiles[i][0] : tiles[i][1]] for i in owned], dim=1),
            [tiles[i][1] - tiles[i][0] for i in owned],
            entry[owned],
            exit_grad[owned],
            packed=True,
        )
        for name, value in batched.items():
            expected = torch.cat(
                [reference[i][name] for i in owned], dim=0 if name in state_names else 1
            )
            assert_bitwise_equal(value, expected, f"{name}, owned={owned}")
