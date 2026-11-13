# SPDX-License-Identifier: BSD-3-Clause

"""Torch drivers for the warpdrive scalar-GDN kernels (derived from cudnn-frontend v1.30).

Ported from cudnn-frontend's GDN engine plans: the forward runs ``uncut`` (one work item
per sequence and value head), the d_v split, or the exact piece ``chain`` when few tiles leave
SMs idle. Workspace regions are ordinary Torch allocations.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from attn_gym.linear._delta_rule.triton.group_sum import group_sum

from .. import gdn_bprop, gdn_prefill, gdn_tinv
from ..bundles.gdn_chain_backward import build_chain_backward, run_chain_backward
from ..bundles.gdn_chain_forward import build_chain_forward, run_chain_forward
from ..bundles.gdn_warmup_backward import build_warmup_backward, run_warmup_backward
from ..bundles.gdn_warmup_forward import build_warmup_forward, run_warmup_forward
from ..contract import STATE_DTYPES, validate_state_series
from ..piece_chain import chain_rows_per_cta
from ..work_tables import WORK_ITEM_FIELDS
from .plan import BackwardPlan as _BackwardPlan
from .plan import ChainCost, aligned, chain_buffers, int32, split_scratch, tma_slots
from .plan import ForwardPlan as _ForwardPlan

B_T = gdn_prefill.B_T
# See NOTE [Chain plan cost] in plan.py; keyed by head dimension. With 48 heads (three pieces on a
# full grid) the D=128 forward chain beats the d_v split only past ~5-6k tokens per piece; on GB200
# it lost at 1x16384 (393 vs 370 us) and won at 1x40960 (848 vs 901 us). D=64 has no d_v split
# and a quarter of the summary state, so its chain pays off on shorter pieces: 1x4096 H2 forward
# 45 vs 100 us uncut, backward 1x1024 H16 70 vs 96 us.
CHAIN_COST_FWD = {
    64: ChainCost(piece=1.125, overhead_tokens=768),
    128: ChainCost(piece=1.25, overhead_tokens=1664),
}
CHAIN_COST_BWD = {
    64: ChainCost(piece=0.125, overhead_tokens=576),
    128: ChainCost(piece=1.0, overhead_tokens=512),
}
# Natural-log gate with no in-kernel gate or beta activation.
_GATE_FLAGS = {"log_gate": True}


def _work_count(device) -> torch.Tensor:
    """The uncut/split work-item count cell the prologue fills.

    Test hook: tests monkeypatch this to keep the cell and read the compacted item count."""
    return int32(1, device)


class ForwardPlan(_ForwardPlan):
    @classmethod
    def build(cls, tokens: int, num_seqs: int, heads_out: int, dim_v: int, device) -> ForwardPlan:
        return cls.for_shape(
            tokens,
            num_seqs,
            heads_out,
            dim_v,
            device,
            b_t=B_T,
            chain_cost=CHAIN_COST_FWD[dim_v],
        )


def gdn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    split: bool = False,
    state_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Packed ``[T, H, D]`` scalar-GDN forward. ``gate`` is the natural-log decay ``[T, HO]``.

    ``state_indices`` switches to paged state: ``initial_state`` is a ``[slots, HO, V, K]`` pool
    that sequence ``b`` reads from and writes back to slot ``state_indices[b]`` in place (see
    ``gdn_prefill.resolve_paged_state`` for the null, fresh and resumed route rules and
    the optional uint8 ``has_initial_state`` mask); the call returns no separate final state.
    Paged calls run the uncut or d_v-split table, never the chain or the approximate split.
    """
    # The kernels never differentiate.
    q, k, v, gate, beta, cu_seqlens = (
        aligned(t.detach()) for t in (q, k, v, gate, beta, cu_seqlens)
    )
    initial_state = None if initial_state is None else initial_state.detach()
    paged = state_indices is not None
    if paged:
        if initial_state is None or output_final_state or split:
            raise ValueError(
                "paged state needs the pool as initial_state and no final state/split"
            )
    elif has_initial_state is not None:
        raise ValueError("has_initial_state requires state_indices")
    tokens, _, dim_k = q.shape
    dim_v = v.shape[-1]
    heads_out = gate.shape[1]
    num_seqs = cu_seqlens.shape[0] - 1
    device = q.device
    plan = ForwardPlan.build(tokens, num_seqs, heads_out, dim_v, device)
    if split:
        # The approximate forgetting-horizon split replaces the exact schemes.
        plan = replace(plan, pieces=0, tiles_per_head=1)
    if paged and plan.chain:
        # The chain seeds pieces through the state chain, which has no route predicates: take the
        # unchained plan with its own d_v split instead of a chain plan without its pieces.
        plan = ForwardPlan.uncut(num_seqs, heads_out, dim_v, plan.num_sm)
    o = torch.empty(tokens, heads_out, dim_v, dtype=q.dtype, device=device)
    final_state = None
    if output_final_state:
        # Empty sequences emit no work item (compacted table), so seed their exit state here.
        final_state = (
            initial_state.clone(memory_format=torch.contiguous_format)
            if initial_state is not None
            else torch.zeros(num_seqs, heads_out, dim_v, dim_k, dtype=torch.float32, device=device)
        )
    common = {
        "q": q,
        "k": k,
        "v": v,
        "gate": gate,
        "beta": beta,
        "o": o,
        "cu_seqlens": cu_seqlens,
        "seed_indices": None,
        "final_indices": None,
        "checkpoints": None,
        "tma_slots": tma_slots(
            tensors=4,
            b_t=B_T,
            keys=heads_out * plan.tiles_per_head,
            device=device,
        ),
    }
    if plan.chain:
        num_pieces = num_seqs * plan.pieces
        buffers = dict(
            common,
            **chain_buffers(
                num_seqs, plan.pieces, heads_out, dim_v, dim_k, device, backward=False
            ),
            **_tinv_buffers(tokens, num_pieces, heads_out, q.dtype, device),
            seed=initial_state,
            final_state=final_state,
        )
        # The builders cache compiled launches by their static configuration; everything
        # shape-dependent is recomputed per call.
        launch = build_chain_forward(
            **buffers,
            pieces=plan.pieces,
            heads_out=heads_out,
            num_seqs=num_seqs,
            b_t=B_T,
            **_GATE_FLAGS,
            checkpoint_every_n_tokens=0,
            scale=scale,
            chain_rows=chain_rows_per_cta(dim_v, dim_k, num_seqs, heads_out, plan.num_sm),
            num_sm=plan.num_sm,
        )
        run_chain_forward(
            launch,
            **buffers,
            pieces=plan.pieces,
            heads_out=heads_out,
            num_seqs=num_seqs,
            scale=scale,
        )
        return o, final_state

    n_tiles = num_seqs * heads_out * plan.tiles_per_head
    ideal, rows, item_scratch, chunk_scratch = split_scratch(
        split, tokens, num_seqs, heads_out, plan.num_sm, n_tiles, B_T, device
    )
    if paged:
        common.update(state_in=initial_state, state_out=initial_state, seed_indices=state_indices)
    else:
        common.update(state_in=initial_state, state_out=final_state)
    buffers = dict(
        common,
        work_items=torch.empty(rows, WORK_ITEM_FIELDS, dtype=torch.int32, device=device),
        work_count=_work_count(device),
        item_scratch=item_scratch,
        chunk_scratch=chunk_scratch,
        scheduler=int32(2, device),
    )
    launch = build_warmup_forward(
        **buffers,
        split=split,
        tiles_per_head=plan.tiles_per_head,
        n_tiles=n_tiles,
        ideal_chunks=ideal,
        num_sm=plan.num_sm,
        b_t=B_T,
        **_GATE_FLAGS,
        checkpoint_every_n_tokens=0,
        scale=scale,
        has_initial_state=has_initial_state,
        paged_state=paged,
    )
    run_warmup_forward(
        *launch,
        **buffers,
        checkpoint_every_n_tokens=0,
        scale=scale,
        has_initial_state=has_initial_state,
    )
    return o, final_state


def _tinv_buffers(tokens: int, num_pieces: int, heads_out: int, dtype, device) -> dict:
    """Chunk-inverse (T^-1) scratch shared by the chain and the backward stages."""
    rows = gdn_tinv.tinv_rows(tokens, num_pieces)
    return {
        "tinv": torch.empty(rows, heads_out, B_T, B_T, dtype=dtype, device=device),
        "tinv_rows": torch.empty(rows, 4, dtype=torch.int32, device=device),
        "tinv_row_count": int32(1, device),
    }


class BackwardPlan(_BackwardPlan):
    @classmethod
    def build(cls, tokens: int, num_seqs: int, heads_out: int, dim_v: int, device) -> BackwardPlan:
        return cls.for_shape(
            tokens,
            num_seqs,
            heads_out,
            device,
            b_t=B_T,
            chain_cost=CHAIN_COST_BWD[dim_v],
        )


def gdn_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    d_output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor | None = None,
    d_final_state: torch.Tensor | None = None,
    split: bool = False,
):
    """Packed scalar-GDN backward: recompute the checkpoint series, then bprop.

    Returns ``(dq, dk, dv, dgate, dbeta, d_initial_state)`` with dq/dk at the q/k head count
    (grouped heads are reduced in a fixed order) and fp32 dgate/dbeta.
    """
    q, k, v, gate, beta, d_output, cu_seqlens = (
        aligned(t.detach()) for t in (q, k, v, gate, beta, d_output, cu_seqlens)
    )
    initial_state = None if initial_state is None else initial_state.detach()
    d_final_state = None if d_final_state is None else d_final_state.detach()
    tokens, key_heads, dim_k = q.shape
    dim_v = v.shape[-1]
    heads_out = gate.shape[1]
    num_seqs = cu_seqlens.shape[0] - 1
    device = q.device
    plan = BackwardPlan.build(tokens, num_seqs, heads_out, dim_v, device)
    if split:
        plan = replace(plan, pieces=0)
    num_pieces = num_seqs * plan.pieces if plan.chain else num_seqs

    def empty(*shape, dtype=torch.int32):
        return torch.empty(shape, dtype=dtype, device=device)

    dq_ho = empty(tokens, heads_out, dim_k, dtype=q.dtype)
    dk_ho = empty(tokens, heads_out, dim_k, dtype=q.dtype)
    dv = torch.empty_like(v)
    dgate = torch.empty(tokens, heads_out, dtype=torch.float32, device=device)
    dbeta = torch.empty(tokens, heads_out, dtype=torch.float32, device=device)
    d_initial_state = None
    if initial_state is not None:
        # Empty sequences emit no work item; their state cotangent passes through unchanged.
        d_initial_state = (
            torch.zeros_like(initial_state)
            if d_final_state is None
            else d_final_state.clone(memory_format=torch.contiguous_format)
        )
    checkpoints = empty(max(tokens // B_T + num_pieces, 1), heads_out, dim_v, dim_k, dtype=q.dtype)
    common = dict(
        q=q,
        k=k,
        v=v,
        do=d_output,
        gate=gate,
        beta=beta,
        cu_seqlens=cu_seqlens,
        checkpoints=checkpoints,
        tma_slots=tma_slots(tensors=7, b_t=B_T, keys=heads_out, device=device),
        dq=dq_ho,
        dk=dk_ho,
        dv=dv,
        **_tinv_buffers(tokens, num_pieces, heads_out, q.dtype, device),
    )

    if plan.chain:
        buffers = dict(
            common,
            **chain_buffers(num_seqs, plan.pieces, heads_out, dim_v, dim_k, device, backward=True),
            dgate=dgate,
            dbeta=dbeta,
            summary_q=q,
            summary_do=d_output,
            seed=initial_state,
            dseed=d_final_state,
            dstate0=d_initial_state,
        )
        schedule = {
            "pieces": plan.pieces,
            "heads_out": heads_out,
            "num_seqs": num_seqs,
            "scale": scale,
        }
        launch = build_chain_backward(
            **buffers,
            **schedule,
            b_t=B_T,
            **_GATE_FLAGS,
            chain_rows=chain_rows_per_cta(dim_v, dim_k, num_seqs, heads_out, plan.num_sm),
            num_sm=plan.num_sm,
        )
        run_chain_backward(launch, **buffers, **schedule)
    else:
        schedulers = empty(4)
        ideal, rows, item_scratch, chunk_scratch = split_scratch(
            split, tokens, num_seqs, heads_out, plan.num_sm, num_seqs * heads_out, B_T, device
        )
        work_items = empty(rows, WORK_ITEM_FIELDS)
        work_count = _work_count(device)
        buffers = dict(
            common,
            state_in=initial_state,
            dgate=dgate,
            work_items=work_items,
            work_count=work_count,
            item_scratch=item_scratch,
            chunk_scratch=chunk_scratch,
            scheduler_all=schedulers,
            scheduler_recompute=schedulers[0:2],
        )
        warmup = build_warmup_backward(
            **buffers,
            b_t=B_T,
            split=split,
            n_tiles=num_seqs * heads_out,
            ideal_chunks=ideal,
            num_sm=plan.num_sm,
            **_GATE_FLAGS,
        )
        run_warmup_backward(*warmup, **buffers)
        # build_warmup_backward checked the shared operands, checkpoints and work table.
        for name, state in (
            ("d_initial_state", d_initial_state),
            ("d_final_state", d_final_state),
        ):
            if state is not None:
                validate_state_series(name, state, num_seqs, heads_out, dim_v, dim_k, STATE_DTYPES)
        gdn_bprop.gdn_bprop(
            q,
            k,
            v,
            gate,
            beta,
            d_output,
            checkpoints,
            work_items,
            work_count,
            dq_ho,
            dk_ho,
            dv,
            dgate,
            dbeta,
            scale=scale,
            use_initial_state=initial_state is not None,
            dstate_in=d_final_state,
            dstate0=d_initial_state,
            scheduler=schedulers[2:4],
            num_sm=plan.num_sm,
            tma_slots=buffers["tma_slots"],
        )

    groups = heads_out // key_heads
    dq = group_sum(dq_ho, groups, out_dtype=q.dtype) if groups > 1 else dq_ho
    dk = group_sum(dk_ho, groups, out_dtype=k.dtype) if groups > 1 else dk_ho
    return dq, dk, dv, dgate, dbeta, d_initial_state
