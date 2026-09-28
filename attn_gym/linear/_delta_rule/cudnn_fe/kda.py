# SPDX-License-Identifier: BSD-3-Clause

"""Torch drivers for the vendored cudnn-frontend v1.30 KDA kernels.

The BT16 plans mirror KdaFrostEngine: uncut, value-dimension split (with shared
prep), exact piece chain, or opt-in approximate forgetting-horizon split. Builders
cache compiled launches by static config; shape-dependent scratch is rebuilt per call.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from .common.piece_chain import chain_rows_per_cta
from .common.split_k import WORK_ITEM_FIELDS
from .kernel import (
    gdn_tinv_f16,
    kda_bprop_f16,
    kda_bprop_summary_f16,
    kda_prefill_f16,
    kda_prep_f16,
    kda_prep_prefill_f16,
    kda_recompute_f16,
    kda_summary_f16,
)
from .kernel.kda_chain_backward_f16 import build_chain_backward, run_chain_backward
from .kernel.kda_chain_forward_f16 import build_chain_forward, run_chain_forward
from .kernel.kda_warmup_backward_f16 import build_warmup_backward, run_warmup_backward
from .kernel.kda_warmup_forward_f16 import build_warmup_forward, run_warmup_forward
from .plan import BackwardPlan as _BackwardPlan
from .plan import ForwardPlan as _ForwardPlan
from .plan import aligned, chain_buffers, int32, split_scratch, workspace

B_T = kda_prefill_f16.CFG.B_T
# Start with the GDN driver's conservative chain floors; KDA has its own BT16 pieces.
# These guard against summary overhead, rather than claiming KDA-specific tuning.
MIN_CHAIN_TOKENS_PER_PIECE_FWD = 8192
MIN_CHAIN_TOKENS_PER_PIECE_BWD = 2048
PREP_TILE_FRACTION = 0.475
_GATE_FLAGS = {
    "log_gate": True,
    "gate_lower_bound": kda_prefill_f16.DEFAULT_GATE_LOWER_BOUND,
}


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
            min_chain_tokens_per_piece=MIN_CHAIN_TOKENS_PER_PIECE_FWD,
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
    if paged:
        # The piece chain has no paged routing; seed and store through the prefill instead.
        plan = replace(plan, pieces=0)
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
    }
    if plan.chain:
        num_pieces = num_seqs * plan.pieces
        buffers = dict(
            common,
            **chain_buffers(
                num_seqs, plan.pieces, heads_out, dim_v, dim_k, device, backward=False
            ),
            summary_words=workspace(kda_summary_f16, num_pieces, device),
            prefill_words=workspace(kda_prefill_f16, num_pieces, device),
            seed=initial_state,
            final_state=final_state,
        )
        launch = build_chain_forward(
            **buffers,
            pieces=plan.pieces,
            heads_out=heads_out,
            num_seqs=num_seqs,
            unit_chunks=plan.unit_chunks,
            b_t=B_T,
            length_rule=False,
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
            checkpoint_every_n_tokens=0,
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
        workspace=workspace(kda_prep_prefill_f16 if prep else kda_prefill_f16, num_seqs, device),
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
    run_warmup_forward(*launch, **buffers, checkpoint_every_n_tokens=0, scale=scale)
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
        "prep_words",
        "prep_rows",
        "prep_row_count",
    )
    if not prep:
        return dict.fromkeys(names)
    rows = gdn_tinv_f16.tinv_rows(tokens, num_seqs, B_T)
    buffers = {
        name: torch.empty(rows, heads_out, B_T, dim_k, dtype=dtype, device=device)
        for name in ("prep_k_decay", "prep_q_decay", "prep_t")
    }
    return dict(
        buffers,
        prep_a=torch.empty(rows, heads_out, B_T * B_T // 2, dtype=torch.int32, device=device),
        prep_diag=torch.empty(rows, heads_out, dim_k, dtype=torch.float32, device=device),
        prep_words=workspace(kda_prep_f16, num_seqs, device),
        prep_rows=torch.empty(rows, 4, dtype=torch.int32, device=device),
        prep_row_count=int32(1, device),
    )


class BackwardPlan(_BackwardPlan):
    @classmethod
    def build(cls, tokens: int, num_seqs: int, heads_out: int, device) -> BackwardPlan:
        return cls.for_shape(
            tokens,
            num_seqs,
            heads_out,
            device,
            b_t=B_T,
            min_chain_tokens_per_piece=MIN_CHAIN_TOKENS_PER_PIECE_BWD,
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
    derivative nor the composed BT64 cumulative-gate reverse scan is needed here.
    """
    q, k, v, gate, beta, d_output, cu_seqlens = (
        aligned(t.detach()) for t in (q, k, v, gate, beta, d_output, cu_seqlens)
    )
    initial_state = None if initial_state is None else aligned(initial_state.detach())
    d_final_state = None if d_final_state is None else aligned(d_final_state.detach())
    tokens, heads_out, dim_k = q.shape
    dim_v = v.shape[-1]
    num_seqs = cu_seqlens.shape[0] - 1
    device = q.device
    plan = BackwardPlan.build(tokens, num_seqs, heads_out, device)
    if split:
        plan = replace(plan, pieces=0)
    num_pieces = num_seqs * plan.pieces if plan.chain else num_seqs

    def empty(*shape, dtype=torch.int32):
        return torch.empty(shape, dtype=dtype, device=device)

    dq, dk, dv, dgate, dbeta = (torch.empty_like(t) for t in (q, k, v, gate, beta))
    d_initial_state = None
    if initial_state is not None:
        # Empty sequences have an identity state map.
        d_initial_state = (
            torch.zeros(initial_state.shape, dtype=torch.float32, device=device)
            if d_final_state is None
            else d_final_state.clone(memory_format=torch.contiguous_format)
        )
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
        "seed_checkpoints": None,
        "dq": dq,
        "dk": dk,
        "dv": dv,
        "dgate": dgate,
        "dbeta": dbeta,
        "dstate0": d_initial_state,
        "bprop_words": workspace(kda_bprop_f16, num_pieces, device),
    }
    if plan.chain:
        buffers = dict(
            common,
            **chain_buffers(num_seqs, plan.pieces, heads_out, dim_v, dim_k, device, backward=True),
            summary_words=workspace(kda_summary_f16, num_pieces, device),
            recompute_m_words=workspace(kda_recompute_f16, num_pieces, device),
            series_words=workspace(kda_recompute_f16, num_pieces, device),
            bprop_summary_words=workspace(kda_bprop_summary_f16, num_pieces, device),
            seed=initial_state,
            dseed=d_final_state,
        )
        schedule = {
            "pieces": plan.pieces,
            "heads_out": heads_out,
            "num_seqs": num_seqs,
            "b_t": B_T,
            "fused_h_m": True,
            "series": True,
            "coarse": False,
            "series_span_tokens": 0,
            "seed_every_n_tokens": 0,
            "scale": scale,
        }
        launch = build_chain_backward(
            **buffers,
            **schedule,
            **_GATE_FLAGS,
            unit_chunks=plan.unit_chunks,
            length_rule=False,
            chain_rows=chain_rows_per_cta(dim_v, dim_k, num_seqs, heads_out, plan.num_sm),
            num_sm=plan.num_sm,
        )
        run_chain_backward(launch, **buffers, **schedule, log_gate=_GATE_FLAGS["log_gate"])
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
            series_items=work_items,
            series_count=work_count,
            item_scratch=item_scratch,
            chunk_scratch=chunk_scratch,
            scheduler_all=schedulers,
            scheduler_recompute=schedulers[0:2],
            scheduler_bwd=schedulers[2:4],
            recompute_words=workspace(kda_recompute_f16, num_pieces, device),
        )
        stages = {
            "b_t": B_T,
            "recompute": True,
            "recompute_orders": True,
            "coarse": False,
            "bwd_orders": False,
            "seed_span_tokens": 0,
            "seed_every_n_tokens": 0,
            "scale": scale,
        }
        launch = build_warmup_backward(
            **buffers,
            **stages,
            use_initial_state=initial_state is not None,
            split=split,
            n_tiles=num_seqs * heads_out,
            ideal_chunks=ideal,
            num_sm=plan.num_sm,
            **_GATE_FLAGS,
        )
        run_warmup_backward(*launch, **buffers, **stages)
    return dq, dk, dv, dgate, dbeta, d_initial_state
