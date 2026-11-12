"""Launch contracts of the frost KDA cuDNN backend: op geometry and the forward launch bundles'
plan buffers, rejected before anything compiles. Runs on the ``[linear]`` CuTeDSL floor."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("cutlass.cute", reason="the cuDNN backend requires CuTeDSL")

from attn_gym.linear._delta_rule.frost import (
    kda_bprop,
    kda_bprop_summary,
    kda_prefill,
    kda_prep,
    kda_prep_prefill,
    kda_recompute,
    kda_summary,
)
from attn_gym.linear._delta_rule.frost.bundles import (
    kda_chain_forward,
    kda_warmup_forward,
)
from attn_gym.linear._delta_rule.frost.drivers import kda

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA tensors")
TOKENS, HEADS, DIM = 128, 2, 128


def reached_compiler(*args, **kwargs):
    raise RuntimeError("reached the compiler")


# The builders select warm launches through these jit_cache lookups before compiling.
reached_compiler.by_args = reached_compiler
reached_compiler.by_static_key = reached_compiler

OPS = {
    "prefill": lambda io, gate, d: kda_prefill.KdaPrefill(io, gate, d, d, 1, True, True, 0, 148),
    "prep": lambda io, gate, d: kda_prep.KdaPrep(io, gate, d, 148),
    "prep_prefill": lambda io, gate, d: kda_prep_prefill.KdaPrepPrefill(
        io, d, 2, True, True, 0, 1
    ),
    "summary": lambda io, gate, d: kda_summary.KdaSummary(io, gate, d, d, 148),
    "recompute": lambda io, gate, d: kda_recompute.KdaRecompute(io, gate, d, d, True, 148),
    "bprop": lambda io, gate, d: kda_bprop.KdaBprop(io, gate, d, d, True, True, True, 148),
}


@pytest.mark.parametrize("op", sorted(OPS))
@pytest.mark.parametrize("invalid", [None, "d_k", "gate"])
def test_kda_ops_reject_unsupported_geometry(op, invalid):
    """The KDA ops serve 64/128 head dims and an fp32 or io-dtype gate."""
    dim = 96 if invalid == "d_k" else DIM
    gate = "fp16" if invalid == "gate" else "fp32"
    if invalid is None:
        OPS[op]("bf16", gate, dim)
        return
    if invalid == "gate" and op == "prep_prefill":
        pytest.skip("the prep prefill reads the prep's decay records, not the gate")
    with pytest.raises(ValueError):
        OPS[op]("bf16", gate, dim)


def test_kda_bprop_summary_rejects_unsupported_geometry():
    import cutlass

    with pytest.raises(ValueError):
        kda_bprop_summary.bundle_hook(cutlass.BFloat16, cutlass.Float32, 96, 96, True)


def packed(*shape, dtype=torch.bfloat16):
    return torch.zeros(*shape, dtype=dtype, device="cuda")


def run_kda_forward(tokens):
    q, k, v = (packed(tokens, HEADS, DIM) for _ in range(3))
    gate = packed(tokens, HEADS, DIM, dtype=torch.float32)
    beta = packed(tokens, HEADS, dtype=torch.float32)
    cu = torch.tensor([0, tokens], dtype=torch.int32, device="cuda")
    kda.kda_forward(q, k, v, gate, beta, cu, scale=DIM**-0.5)


@needs_cuda
@pytest.mark.parametrize("invalid", [None, "n_tiles", "interval", "work_rows"])
def test_kda_warmup_forward_rejects_invalid_plan_buffers(monkeypatch, invalid):
    """The real uncut plan reaches the compiler; corrupting one of its buffers fails before
    compiling."""
    monkeypatch.setattr(kda_warmup_forward, "_compile_warmup_forward", reached_compiler)
    monkeypatch.setattr(kda.ForwardPlan, "chain", property(lambda self: False))
    build = kda.build_warmup_forward

    def corrupt(**kwargs):
        if invalid == "n_tiles":
            kwargs["n_tiles"] += 1
        elif invalid == "interval":
            kwargs["checkpoint_every_n_tokens"] = 8
        elif invalid == "work_rows":
            kwargs["work_items"] = kwargs["work_items"][:1]
        return build(**kwargs)

    monkeypatch.setattr(kda, "build_warmup_forward", corrupt)
    with pytest.raises(RuntimeError if invalid is None else ValueError):
        run_kda_forward(TOKENS)


@needs_cuda
@pytest.mark.parametrize("invalid", [None, "chain_rows", "state_rows"])
def test_kda_chain_forward_rejects_invalid_plan_buffers(monkeypatch, invalid):
    """The real chain plan reaches the compiler; corrupting one of its buffers fails before
    compiling."""
    monkeypatch.setattr(kda_chain_forward, "_compile_chain_forward", reached_compiler)
    monkeypatch.setattr(kda, "MIN_CHAIN_TOKENS_PER_PIECE_FWD", 0)
    build = kda.build_chain_forward

    def corrupt(**kwargs):
        assert kwargs["pieces"] > 1
        if invalid == "chain_rows":
            kwargs["chain_rows"] = 3
        elif invalid == "state_rows":
            kwargs["state_m"] = kwargs["state_m"][:1]
        return build(**kwargs)

    monkeypatch.setattr(kda, "build_chain_forward", corrupt)
    with pytest.raises(RuntimeError if invalid is None else ValueError):
        run_kda_forward(16384)
