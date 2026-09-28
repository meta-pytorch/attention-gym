"""Driver cases that compile every vendored GDN and KDA kernel variant on small shapes.

Each case calls a public driver (``gdn_forward``, ``kda_backward``, ``build_state_summaries``...)
with the plan pinned through the driver's own plan builders: the chain floors
(``MIN_CHAIN_TOKENS_PER_PIECE_*``) select chain versus table plans, ``ForwardPlan.build`` is
wrapped to fix ``tiles_per_head`` (d_v split), and ``PREP_TILE_FRACTION`` selects the KDA prep
path. The int64 variants force ``requires_int64_abi`` in every ``cudnn_fe`` module that imports
it, as the layout tests do. Shapes stay tiny (two sequences, a few chunks, two heads), so a full
run is dominated by compilation.
"""

from __future__ import annotations

import contextlib
import importlib
import pkgutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace

import torch

from attn_gym.linear._delta_rule import cudnn_fe
from attn_gym.linear._delta_rule.cudnn_fe import gdn, kda, summary

from .._common import D, Workload, op_inputs

DEVICE = torch.device("cuda")
SEQS = 2
HEADS = 2
DRIVERS = {"gdn": gdn, "kda": kda}
# Tokens per sequence: enough chunks (B_T=64 GDN, 16 KDA) that ``choose_pieces`` chains with
# four pieces when the chain floor is lowered, one to two chunks otherwise.
CHAIN_TOKENS = {"gdn": 1024, "kda": 256}
TABLE_TOKENS = {"gdn": 128, "kda": 64}


@dataclass(frozen=True)
class Case:
    name: str
    run: Callable[[], None]


def inputs(kind: str, tokens_per_seq: int, dtype: torch.dtype):
    """Driver-level ``(T, H, D)`` operands (no batch dimension) plus an initial state."""
    work = Workload(kind, (tokens_per_seq,) * SEQS, HEADS, HEADS)
    ops = op_inputs(work, seed=0, dtype=dtype, state=True, device=DEVICE)
    return ops.q[0], ops.k[0], ops.v[0], ops.g[0], ops.beta[0], ops.cu_seqlens, ops.state


@contextlib.contextmanager
def pinned(kind: str, plan: str) -> Iterator[None]:
    """Pin the driver's plan: ``uncut``, ``dv`` (two value tiles), ``prep`` (KDA d_v split with
    shared prep) or ``chain`` (exact piece chain)."""
    driver = DRIVERS[kind]
    saved = {
        name: getattr(driver, name)
        for name in ("MIN_CHAIN_TOKENS_PER_PIECE_FWD", "MIN_CHAIN_TOKENS_PER_PIECE_BWD")
    }
    build = driver.ForwardPlan.__dict__["build"]
    floor = 0 if plan == "chain" else 1 << 40
    tiles = 2 if plan in ("dv", "prep") else 1
    driver.MIN_CHAIN_TOKENS_PER_PIECE_FWD = driver.MIN_CHAIN_TOKENS_PER_PIECE_BWD = floor
    natural = build.__func__
    driver.ForwardPlan.build = classmethod(
        lambda cls, *args: replace(natural(cls, *args), tiles_per_head=tiles)
    )
    if kind == "kda":
        saved["PREP_TILE_FRACTION"] = driver.PREP_TILE_FRACTION
        driver.PREP_TILE_FRACTION = 1.0 if plan == "prep" else 0.0
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(driver, name, value)
        driver.ForwardPlan.build = build


@contextlib.contextmanager
def forced_int64(enabled: bool) -> Iterator[None]:
    """Force the int64 launch ABI in every ``cudnn_fe`` module that selects it."""
    if not enabled:
        yield
        return
    modules = [
        importlib.import_module(info.name)
        for info in pkgutil.walk_packages(cudnn_fe.__path__, cudnn_fe.__name__ + ".")
    ]
    patched = [module for module in modules if hasattr(module, "requires_int64_abi")]
    saved = {module: module.requires_int64_abi for module in patched}
    for module in patched:
        module.requires_int64_abi = lambda *tensors: True
    try:
        yield
    finally:
        for module, original in saved.items():
            module.requires_int64_abi = original


def forward(
    kind: str,
    plan: str,
    *,
    state: bool = True,
    split: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    int64: bool = False,
    paged: str | None = None,
) -> Callable[[], None]:
    """``paged`` is ``None``, ``"routes"`` (all resumed) or ``"mask"`` (one fresh slot)."""

    def run() -> None:
        tokens = (CHAIN_TOKENS if plan == "chain" else TABLE_TOKENS)[kind]
        q, k, v, gate, beta, cu_seqlens, initial = inputs(kind, tokens, dtype)
        kwargs = {"scale": D**-0.5}
        if paged:
            pool = torch.cat([torch.zeros_like(initial[:1]), initial])
            kwargs.update(
                initial_state=pool,
                state_indices=torch.arange(1, SEQS + 1, dtype=torch.int32, device=DEVICE),
            )
            if paged == "mask":
                kwargs["has_initial_state"] = torch.tensor(
                    [1, 0], dtype=torch.uint8, device=DEVICE
                )
        else:
            kwargs.update(
                initial_state=initial if state else None, output_final_state=state, split=split
            )
        with pinned(kind, plan), forced_int64(int64):
            out, _ = getattr(DRIVERS[kind], f"{kind}_forward")(
                q, k, v, gate, beta, cu_seqlens, **kwargs
            )
        torch.cuda.synchronize()
        assert torch.isfinite(out.float()).all()

    return run


def backward(
    kind: str,
    plan: str,
    *,
    state: bool = True,
    split: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    int64: bool = False,
) -> Callable[[], None]:
    def run() -> None:
        tokens = (CHAIN_TOKENS if plan == "chain" else TABLE_TOKENS)[kind]
        q, k, v, gate, beta, cu_seqlens, initial = inputs(kind, tokens, dtype)
        d_output = torch.randn_like(v) * 0.1
        with pinned(kind, plan), forced_int64(int64):
            grads = getattr(DRIVERS[kind], f"{kind}_backward")(
                q,
                k,
                v,
                gate,
                beta,
                d_output,
                cu_seqlens,
                scale=D**-0.5,
                initial_state=initial if state else None,
                d_final_state=torch.randn_like(initial) if state else None,
                split=split,
            )
        torch.cuda.synchronize()
        assert all(torch.isfinite(g.float()).all() for g in grads if g is not None)

    return run


def summaries(
    mode: str, *, dtype: torch.dtype = torch.bfloat16, int64: bool = False
) -> Callable[[], None]:
    """``fwd`` (kda_summary), ``bwd`` (kda_summary + kda_bprop_summary with a zero exit
    cotangent) or ``probe`` (kda_bprop_summary seeded with the identity as well)."""

    def run() -> None:
        q, k, v, gate, beta, cu_seqlens, _ = inputs("kda", TABLE_TOKENS["kda"], dtype)
        staged = tuple(t[None] for t in (q, k, v, gate, beta))
        with forced_int64(int64):
            if mode == "fwd":
                maps = summary.build_state_summaries(*staged[1:], cu_seqlens)
            else:
                maps = summary.build_state_grad_summaries(
                    *staged,
                    torch.randn_like(v)[None] * 0.1,
                    cu_seqlens,
                    D**-0.5,
                    transpose_forward_transition=mode == "bwd",
                )
        torch.cuda.synchronize()
        assert torch.isfinite(maps).all()

    return run


def cases() -> list[Case]:
    f16 = torch.float16
    out = []
    for kind in ("gdn", "kda"):
        out += [
            Case(f"{kind}_fwd_uncut", forward(kind, "uncut")),
            Case(f"{kind}_fwd_uncut_nostate", forward(kind, "uncut", state=False)),
            Case(f"{kind}_fwd_uncut_f16", forward(kind, "uncut", dtype=f16)),
            Case(f"{kind}_fwd_uncut_i64", forward(kind, "uncut", int64=True)),
            Case(f"{kind}_fwd_dv", forward(kind, "dv")),
            Case(f"{kind}_fwd_chain", forward(kind, "chain")),
            Case(f"{kind}_fwd_chain_i64", forward(kind, "chain", int64=True)),
            Case(f"{kind}_fwd_warmup", forward(kind, "uncut", state=False, split=True)),
            Case(f"{kind}_fwd_paged", forward(kind, "uncut", paged="routes")),
            Case(f"{kind}_fwd_paged_mask", forward(kind, "uncut", paged="mask")),
            Case(f"{kind}_fwd_paged_dv", forward(kind, "dv", paged="routes")),
            Case(f"{kind}_bwd_uncut", backward(kind, "uncut")),
            Case(f"{kind}_bwd_uncut_nostate", backward(kind, "uncut", state=False)),
            Case(f"{kind}_bwd_uncut_f16", backward(kind, "uncut", dtype=f16)),
            Case(f"{kind}_bwd_uncut_i64", backward(kind, "uncut", int64=True)),
            Case(f"{kind}_bwd_chain", backward(kind, "chain")),
            Case(f"{kind}_bwd_chain_i64", backward(kind, "chain", int64=True)),
            Case(f"{kind}_bwd_warmup", backward(kind, "uncut", state=False, split=True)),
        ]
    dv = [case.name for case in out].index("kda_fwd_dv") + 1
    out[dv:dv] = [
        Case("kda_fwd_prep", forward("kda", "prep")),
        Case("kda_fwd_paged_prep", forward("kda", "prep", paged="routes")),
    ]
    out += [
        Case("kda_summary_fwd", summaries("fwd")),
        Case("kda_summary_fwd_f16", summaries("fwd", dtype=f16)),
        Case("kda_summary_fwd_i64", summaries("fwd", int64=True)),
        Case("kda_summary_bwd", summaries("bwd")),
        Case("kda_summary_bwd_i64", summaries("bwd", int64=True)),
        Case("kda_summary_probe", summaries("probe")),
    ]
    return out
