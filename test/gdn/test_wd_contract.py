"""Launch contracts of the warpdrive backend: exact dtype names, split-table geometry, op
geometry and the GDN launch bundles' plan buffers, all rejected before anything compiles.
Runs on the ``[linear]`` CuTeDSL floor."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("cutlass.cute", reason="the warpdrive backend requires CuTeDSL")

import cutlass

from attn_gym.linear._delta_rule.warpdrive import (
    contract,
    gdn_bprop,
    gdn_bprop_summary,
    gdn_recompute,
    gdn_summary,
    gdn_tinv,
    work_tables,
)
from attn_gym.linear._delta_rule.warpdrive.bundles import (
    gdn_chain_backward,
    gdn_warmup_backward,
)
from attn_gym.linear._delta_rule.warpdrive.drivers import gdn

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA tensors")
TOKENS, HEADS, DIM = 128, 2, 128


def reached_compiler(*args, **kwargs):
    raise RuntimeError("reached the compiler")


# The builders select warm launches through these jit_cache lookups before compiling.
reached_compiler.by_args = reached_compiler
reached_compiler.by_static_key = reached_compiler


def test_dtype_names_are_exact():
    """Dtype names match exactly; a supported name inside another is rejected."""
    assert contract.get_dtype(torch.float16) is cutlass.Float16
    assert contract.get_dtype(torch.bfloat16) is cutlass.BFloat16
    for name in ("torch.bfloat16_suffix", "float", "float32x"):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            contract.get_dtype(name)
    assert contract.dtype_name(torch.int64) == "int64"
    assert contract.dtype_name(cutlass.Float32) == "float32"
    assert contract.op_dtype_name(cutlass.BFloat16) == "bf16"


@needs_cuda
@pytest.mark.parametrize("invalid", ["tiles", "ideal", "chunks", "staging", "gate"])
def test_split_table_rejects_invalid_launch_geometry(monkeypatch, invalid):
    """Each malformed split launch fails with a ValueError before compiling."""
    monkeypatch.setattr(work_tables, "compile_split_table", reached_compiler)
    heads = 2
    gate = torch.zeros(32, heads, device="cuda")
    if invalid == "gate":
        gate = torch.zeros(32, heads, device="cuda", dtype=torch.float64)
    cu = torch.tensor([0, 32], dtype=torch.int32, device="cuda")
    rows = 1 if invalid == "chunks" else work_tables.chunk_scratch_rows(32, 1, 16)
    staging_rows = 1 if invalid == "staging" else 64
    with pytest.raises(ValueError):
        work_tables.build_split_table(
            gate,
            cu,
            torch.empty(rows, heads, device="cuda"),
            torch.empty(staging_rows, work_tables.WORK_ITEM_FIELDS, dtype=torch.int32).cuda(),
            torch.empty(1, dtype=torch.int32, device="cuda"),
            n_tiles=3 if invalid == "tiles" else heads,
            ideal_chunks=0 if invalid == "ideal" else 32,
            num_sms=148,
            b_t=16,
        )


@pytest.mark.parametrize(
    "build",
    [
        lambda: gdn_recompute.bundle_hook(cutlass.BFloat16, 96, 96, False, 148),
        lambda: gdn_summary.bundle_hook(cutlass.BFloat16, cutlass.Float32, 96, 128, True),
        lambda: gdn_summary.bundle_hook(cutlass.BFloat16, cutlass.Float32, 128, 128, False),
        lambda: gdn_bprop_summary.bundle_hook(cutlass.BFloat16, 128, 128, False),
        lambda: gdn_tinv.bundle_hook(cutlass.BFloat16, 96, 148, True),
        lambda: gdn_tinv.bundle_hook(cutlass.BFloat16, 128, 148, False),
        lambda: gdn_bprop.bundle_hook(cutlass.BFloat16, 64, 128, True, True, True, True, 148),
        lambda: gdn_bprop.bundle_hook(cutlass.BFloat16, 96, 96, True, True, True, True, 148),
        lambda: gdn_bprop.bundle_hook(cutlass.BFloat16, 128, 128, True, True, True, False, 148),
    ],
    ids=[
        "recompute-d_k",
        "summary-d_k",
        "summary-linear-gate",
        "bprop_summary-linear-gate",
        "tinv-d_k",
        "tinv-linear-gate",
        "bprop-unequal-dims",
        "bprop-d_k",
        "bprop-linear-gate",
    ],
)
def test_gdn_ops_reject_unsupported_geometry(build):
    """The GDN ops serve 64/128 head dims (d_k == d_v for the bprop) of natural-log gates."""
    with pytest.raises(ValueError):
        build()


def packed(*shape, dtype=torch.bfloat16):
    return torch.zeros(*shape, dtype=dtype, device="cuda")


def misaligned(tensor):
    storage = torch.empty(tensor.numel() + 8, dtype=tensor.dtype, device=tensor.device)
    return storage[1 : 1 + tensor.numel()].view_as(tensor)


def run_gdn_backward(tokens):
    q, k, v, do = (packed(tokens, HEADS, DIM) for _ in range(4))
    gate = packed(tokens, HEADS, dtype=torch.float32)
    beta = packed(tokens, HEADS, dtype=torch.float32)
    cu = torch.tensor([0, tokens], dtype=torch.int32, device="cuda")
    gdn.gdn_backward(q, k, v, gate, beta, do, cu, scale=DIM**-0.5)


@needs_cuda
@pytest.mark.parametrize("invalid", [None, "n_tiles", "checkpoint_rows", "work_fields", "dq"])
def test_gdn_warmup_backward_rejects_invalid_plan_buffers(monkeypatch, invalid):
    """The real uncut plan reaches the compiler; corrupting one of its buffers fails before
    compiling."""
    monkeypatch.setattr(gdn_warmup_backward, "_compile_warmup_backward", reached_compiler)
    monkeypatch.setattr(gdn.BackwardPlan, "chain", property(lambda self: False))
    build = gdn.build_warmup_backward

    def corrupt(**kwargs):
        if invalid == "n_tiles":
            kwargs["n_tiles"] += 1
        elif invalid == "checkpoint_rows":
            kwargs["checkpoints"] = kwargs["checkpoints"][:1]
        elif invalid == "work_fields":
            kwargs["work_items"] = kwargs["work_items"][:, :8]
        elif invalid == "dq":
            kwargs["dq"] = misaligned(kwargs["dq"])
        return build(**kwargs)

    monkeypatch.setattr(gdn, "build_warmup_backward", corrupt)
    with pytest.raises(RuntimeError if invalid is None else ValueError):
        run_gdn_backward(TOKENS)


@needs_cuda
@pytest.mark.parametrize("invalid", [None, "chain_rows", "summary_items"])
def test_gdn_chain_backward_rejects_invalid_plan_buffers(monkeypatch, invalid):
    """The real chain plan reaches the compiler; corrupting one of its buffers fails before
    compiling."""
    monkeypatch.setattr(gdn_chain_backward, "_compile_chain_backward_head", reached_compiler)
    monkeypatch.setitem(gdn.MIN_CHAIN_TOKENS_PER_PIECE_BWD, 128, 0)
    build = gdn.build_chain_backward

    def corrupt(**kwargs):
        assert kwargs["pieces"] > 1
        if invalid == "chain_rows":
            kwargs["chain_rows"] = 3
        elif invalid == "summary_items":
            kwargs["work_items_summary"] = kwargs["work_items_summary"][:1]
        return build(**kwargs)

    monkeypatch.setattr(gdn, "build_chain_backward", corrupt)
    with pytest.raises(RuntimeError if invalid is None else ValueError):
        run_gdn_backward(16384)
