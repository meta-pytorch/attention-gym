# SPDX-License-Identifier: BSD-3-Clause

"""Torch drivers for the warpdrive KDA kernels (derived from cudnn-frontend v1.30).

The BT16 plans mirror cudnn-frontend's KDA engine: uncut, value-dimension split (with shared
prep), exact piece chain, or opt-in approximate forgetting-horizon split. Builders
cache compiled launches by static config; shape-dependent scratch is rebuilt per call.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from attn_gym.linear._delta_rule.triton.group_sum import group_sum

from ..bundles.kda_chain_backward import build_chain_backward, run_chain_backward
from ..bundles.kda_chain_forward import build_chain_forward, run_chain_forward
from ..bundles.kda_warmup_backward import build_warmup_backward, run_warmup_backward
from ..bundles.kda_warmup_forward import build_warmup_forward, run_warmup_forward
from ..kda_common import B_T
from ..piece_chain import chain_rows_per_cta
from ..work_tables import WORK_ITEM_FIELDS
from .plan import BackwardPlan as _BackwardPlan
from .plan import ChainCost, aligned, chain_buffers, int32, split_scratch, tma_slots
from .plan import ForwardPlan as _ForwardPlan

# See NOTE [Chain plan cost] in plan.py; keyed by head dimension as in GDN. The prep d_v split is
# strong enough that three forward pieces on a full grid never beat it.
CHAIN_COST_FWD = {128: ChainCost(piece=1.5, overhead_tokens=2560)}
CHAIN_COST_BWD = {128: ChainCost(piece=1.0, overhead_tokens=256)}
PREP_TILE_FRACTION = 0.475
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


def kda_forward(
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
    """Packed [T,H,D] KDA; gate is a natural-log channel decay, state is [N,H,V,K].

    With ``state_indices`` the call is paged: ``initial_state`` is a ``[P,H,V,K]`` pool updated in
    place through the routes (non-positive routes are null; the optional uint8
    ``has_initial_state`` marks resumed slots, zero marks fresh ones) and no final state is
    returned. Paged calls use the unchunked plans (uncut or value-dimension split).
    """
    paged = state_indices is not None
    q, k, v, gate, beta, cu_seqlens = (
        aligned(t.detach()) for t in (q, k, v, gate, beta, cu_seqlens)
    )
    if paged:
        if initial_state is None or output_final_state or split:
            raise ValueError("paged KDA needs a state pool and no final state or split schedule")
        if initial_state.data_ptr() % 16:
            raise ValueError("the paged state pool must be 16-byte aligned")
        initial_state = initial_state.detach()
    else:
        initial_state = None if initial_state is None else aligned(initial_state.detach())
    tokens, heads_out, dim_k = q.shape
    dim_v = v.shape[-1]
    num_seqs = cu_seqlens.shape[0] - 1
    device = q.device
    plan = ForwardPlan.build(tokens, num_seqs, heads_out, dim_v, device)
    if split:
        plan = replace(plan, pieces=0, tiles_per_head=1)
    if paged and plan.chain:
        # The piece chain has no paged routing; seed and store through the prefill instead, with
        # the unchained plan's own d_v split (and shared prep) rather than a chain plan's tiles.
        plan = ForwardPlan.uncut(num_seqs, heads_out, dim_v, plan.num_sm)
    o = torch.empty_like(v)
    final_state = None
    if output_final_state:
        # Compacted empty sequences emit no work: seed their final state here.
        final_state = (
            initial_state.clone()
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
            tensors=5,
            b_t=B_T,
            keys=gate.shape[1] * plan.tiles_per_head,
            device=device,
        ),
    }
    if plan.chain:
        buffers = dict(
            common,
            **chain_buffers(
                num_seqs, plan.pieces, heads_out, dim_v, dim_k, device, backward=False
            ),
            seed=initial_state,
            final_state=final_state,
        )
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

    prep = plan.tiles_per_head > 1 and num_seqs * heads_out <= int(
        PREP_TILE_FRACTION * plan.num_sm
    )
    n_tiles = num_seqs * heads_out * plan.tiles_per_head
    ideal, rows, item_scratch, chunk_scratch = split_scratch(
        split, tokens, num_seqs, heads_out, plan.num_sm, n_tiles, B_T, device
    )
    buffers = dict(
        common,
        state_in=initial_state,
        state_out=initial_state if paged else final_state,
        state_indices=state_indices,
        has_initial_state=has_initial_state,
        work_items=torch.empty(rows, WORK_ITEM_FIELDS, dtype=torch.int32, device=device),
        work_count=_work_count(device),
        item_scratch=item_scratch,
        chunk_scratch=chunk_scratch,
        scheduler=int32(2, device),
        **_prep_buffers(tokens, num_seqs, heads_out, dim_k, q.dtype, device, prep),
    )
    launch = build_warmup_forward(
        **buffers,
        split=split,
        tiles_per_head=plan.tiles_per_head,
        prep=prep,
        n_tiles=n_tiles,
        ideal_chunks=ideal,
        num_sm=plan.num_sm,
        b_t=B_T,
        **_GATE_FLAGS,
        checkpoint_every_n_tokens=0,
        scale=scale,
    )
    run_warmup_forward(*launch, **buffers, scale=scale)
    return o, final_state


def _prep_buffers(
    tokens: int, num_seqs: int, heads_out: int, dim_k: int, dtype, device, prep: bool
) -> dict:
    """The d_v split shares its decay/factor prep across the two value tiles."""
    names = (
        "prep_k_decay",
        "prep_q_decay",
        "prep_t",
        "prep_a",
        "prep_diag",
        "prep_rows",
        "prep_row_count",
    )
    if not prep:
        return dict.fromkeys(names)
    rows = tokens // B_T + num_seqs  # one row per chunk plus one padding row per sequence
    buffers = {
        name: torch.empty(rows, heads_out, B_T, dim_k, dtype=dtype, device=device)
        for name in ("prep_k_decay", "prep_q_decay", "prep_t")
    }
    return dict(
        buffers,
        prep_a=torch.empty(rows, heads_out, B_T * B_T // 2, dtype=torch.int32, device=device),
        prep_diag=torch.empty(rows, heads_out, dim_k, dtype=torch.float32, device=device),
        prep_rows=torch.empty(rows, 4, dtype=torch.int32, device=device),
        prep_row_count=int32(1, device),
    )


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


def kda_backward(
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
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None
]:
    """Recompute checkpoint series and differentiate KDA, with optional state/cotangent.

    Native bprop returns natural-log dgate directly; neither the safe-gate sigmoid
    derivative nor the composed BT64 cumulative-gate reverse scan is needed here. The bprop
    stores dq/dk/dv per output (gate) head; grouped q/k/v heads are summed over their groups.
    """
    q, k, v, gate, beta, d_output, cu_seqlens = (
        aligned(t.detach()) for t in (q, k, v, gate, beta, d_output, cu_seqlens)
    )
    initial_state = None if initial_state is None else aligned(initial_state.detach())
    d_final_state = None if d_final_state is None else aligned(d_final_state.detach())
    tokens, _, dim_k = q.shape
    heads_out = gate.shape[1]
    dim_v = v.shape[-1]
    num_seqs = cu_seqlens.shape[0] - 1
    device = q.device
    plan = BackwardPlan.build(tokens, num_seqs, heads_out, dim_v, device)
    if split:
        plan = replace(plan, pieces=0)
    num_pieces = num_seqs * plan.pieces if plan.chain else num_seqs

    def empty(*shape, dtype=torch.int32):
        return torch.empty(shape, dtype=dtype, device=device)

    # Per output head; grouped inputs are summed into their own head count at the end.
    dq, dk, dv = (
        torch.empty_like(t)
        if t.shape[1] == heads_out
        else empty(tokens, heads_out, t.shape[-1], dtype=t.dtype)
        for t in (q, k, v)
    )
    dgate, dbeta = torch.empty_like(gate), torch.empty_like(beta)
    d_initial_state = None
    if initial_state is not None:
        # Empty sequences have an identity state map.
        d_initial_state = (
            torch.zeros(initial_state.shape, dtype=torch.float32, device=device)
            if d_final_state is None
            else d_final_state.clone(memory_format=torch.contiguous_format)
        )
    # Chunk c of sequence (or chain piece) b checkpoints at row cu[b] // B_T + b + c.
    checkpoints = empty(max(tokens // B_T + num_pieces, 1), heads_out, dim_v, dim_k, dtype=q.dtype)
    common = {
        "q": q,
        "k": k,
        "v": v,
        "do": d_output,
        "gate": gate,
        "beta": beta,
        "cu_seqlens": cu_seqlens,
        "checkpoints": checkpoints,
        "dq": dq,
        "dk": dk,
        "dv": dv,
        "dgate": dgate,
        "dbeta": dbeta,
        "dstate0": d_initial_state,
        "tma_slots": tma_slots(
            tensors=9,
            b_t=B_T,
            keys=heads_out,
            device=device,
        ),
    }
    if plan.chain:
        buffers = dict(
            common,
            **chain_buffers(num_seqs, plan.pieces, heads_out, dim_v, dim_k, device, backward=True),
            seed=initial_state,
            dseed=d_final_state,
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
            chain_rows=chain_rows_per_cta(
                dim_v, dim_k, num_seqs, heads_out, plan.num_sm, chains=2
            ),
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
            dstate_in=d_final_state,
            work_items=work_items,
            work_count=work_count,
            item_scratch=item_scratch,
            chunk_scratch=chunk_scratch,
            scheduler_all=schedulers,
            scheduler_recompute=schedulers[0:2],
            scheduler_bwd=schedulers[2:4],
        )
        launch = build_warmup_backward(
            **buffers,
            b_t=B_T,
            scale=scale,
            use_initial_state=initial_state is not None,
            split=split,
            n_tiles=num_seqs * heads_out,
            ideal_chunks=ideal,
            num_sm=plan.num_sm,
            **_GATE_FLAGS,
        )
        run_warmup_backward(*launch, **buffers, scale=scale)
    dq, dk, dv = (
        group_sum(grad, heads_out // t.shape[1], out_dtype=t.dtype)
        if t.shape[1] < heads_out
        else grad
        for grad, t in ((dq, q), (dk, k), (dv, v))
    )
    return dq, dk, dv, dgate, dbeta, d_initial_state
