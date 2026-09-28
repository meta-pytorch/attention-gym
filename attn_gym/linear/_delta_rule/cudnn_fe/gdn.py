# SPDX-License-Identifier: BSD-3-Clause

"""Torch drivers for the vendored cudnn-frontend v1.30 scalar-GDN kernels.

Ported from cudnn-frontend's ``GdnFrostEngine`` plans: the forward runs ``uncut`` (one work item
per sequence and value head), the d_v split, or the exact piece ``chain`` for long sequences on
few tiles. Workspace regions are ordinary Torch allocations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from attn_gym.linear._delta_rule.triton.group_sum import group_sum

from .common.host import tensormap_workspace_bytes
from .common.piece_chain import (
    DV_SPLIT_TILES,
    chain_rows_per_cta,
    choose_pieces,
    piece_table_layout,
)
from .common.split_k import (
    WORK_ITEM_FIELDS,
    chunk_scratch_rows,
    compute_ideal_chunks,
    max_work_items,
)
from .kernel import (
    gdn_bprop_f16,
    gdn_bprop_summary_f16,
    gdn_prefill_f16,
    gdn_recompute_f16,
    gdn_summary_f16,
    gdn_tinv_f16,
)
from .kernel.gdn_chain_backward_f16 import build_chain_backward, run_chain_backward
from .kernel.gdn_chain_forward_f16 import build_chain_forward, run_chain_forward
from .kernel.gdn_warmup_backward_f16 import build_warmup_backward, run_warmup_backward
from .kernel.gdn_warmup_forward_f16 import build_warmup_forward, run_warmup_forward

B_T = gdn_prefill_f16.CFG.B_T
# Shorter chain pieces cost more in summaries than they save. cudnn-frontend v1.30 chains any
# sequence whose heads fill at most a third of the SMs; on GB200 at 48 heads, D=128 the forward
# d_v split beats the chain up to ~5.5k tokens per piece (1x16384: 370 vs 393 us) and the chain
# wins at ~13.6k (1x40960: 848 vs 901 us). The backward has no d_v split, and its chain already
# wins at ~2.7k tokens per piece (1x8192: 705 vs 928 us) but loses at 1k (1x2048: 279 vs 258).
MIN_CHAIN_TOKENS_PER_PIECE_FWD = 8192
MIN_CHAIN_TOKENS_PER_PIECE_BWD = 2048
# Natural-log gate with no in-kernel gate or beta activation.
_GATE_FLAGS = {
    "log_gate": True,
    "safe_gate": False,
    "use_beta_sigmoid": False,
    "allow_neg_eigval": False,
}


def _int32(n: int, device) -> torch.Tensor:
    return torch.empty(n, dtype=torch.int32, device=device)


def _workspace(module, count: int, device) -> torch.Tensor:
    """TMA-descriptor workspace for ``count`` batch entries of a kernel module."""
    return torch.empty(
        tensormap_workspace_bytes(module, count) // 8, dtype=torch.int64, device=device
    )


def _aligned(tensor: torch.Tensor) -> torch.Tensor:
    """Copy a tensor whose base is not 16-byte aligned; the kernels assume 16-byte bases."""
    return tensor if tensor.data_ptr() % 16 == 0 else tensor.clone()


def _work_count(device) -> torch.Tensor:
    """The uncut/split work-item count cell the prologue fills (tests record it here)."""
    return _int32(1, device)


@dataclass
class ForwardPlan:
    """Scheme and scratch sizes for one (shape, device) forward; a pure function of the shapes."""

    pieces: int
    unit_chunks: int
    tiles_per_head: int
    num_sm: int

    @property
    def chain(self) -> bool:
        return self.pieces > 0

    @classmethod
    def build(cls, tokens: int, num_seqs: int, heads_out: int, dim_v: int, device) -> ForwardPlan:
        num_sm = torch.cuda.get_device_properties(device).multi_processor_count
        kw = {
            "num_seqs": num_seqs,
            "heads_out": heads_out,
            "num_sm": num_sm,
            "total_tokens": tokens,
            "b_t": B_T,
            "expand_num": 1,
        }
        pieces, unit = choose_pieces(cadence_tokens=0, batch_invariant=False, **kw)
        if pieces and tokens < MIN_CHAIN_TOKENS_PER_PIECE_FWD * pieces * num_seqs:
            pieces = 0
        # Upstream splits d_v only at a slot budget of exactly two; any unchained plan whose two
        # CTAs per tile fit in one wave gains (1x2048: 67 -> 58 us, 1x8192: 220 -> 193 us).
        dv = not pieces and dim_v == 128 and num_sm // (num_seqs * heads_out) >= DV_SPLIT_TILES
        return cls(pieces, unit, DV_SPLIT_TILES if dv else 1, num_sm)


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
    ``common/paged_state.py`` for the null, fresh and resumed route rules and the optional uint8
    ``has_initial_state`` mask); the call returns no separate final state. Paged calls run the
    uncut or d_v-split table, never the chain or the approximate split.
    """
    # The kernels never differentiate.
    q, k, v, gate, beta, cu_seqlens = (
        _aligned(t.detach()) for t in (q, k, v, gate, beta, cu_seqlens)
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
    if paged:
        # The chain seeds pieces through the state chain, which has no route predicates.
        plan = replace(plan, pieces=0)
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
        "a_log": None,
        "dt_bias": None,
        "o": o,
        "cu_seqlens": cu_seqlens,
        "seed_indices": None,
        "final_indices": None,
        "checkpoints": None,
    }
    if plan.chain:
        num_pieces = num_seqs * plan.pieces
        schedulers = _int32(6, device)
        state = (num_pieces, heads_out, dim_v, dim_k)
        buffers = dict(
            common,
            **_piece_buffers(num_seqs, plan.pieces, heads_out, device),
            **_tinv_buffers(tokens, num_pieces, heads_out, q.dtype, device),
            scheduler_all=schedulers,
            scheduler_summary=schedulers[2:4],
            scheduler_prefill=schedulers[0:2],
            summary_words=_workspace(gdn_summary_f16, num_pieces, device),
            prefill_words=_workspace(gdn_prefill_f16, num_pieces, device),
            state_h=torch.empty(state, dtype=torch.float32, device=device),
            state_m=torch.empty(
                num_pieces, heads_out, dim_k, dim_k, dtype=torch.float32, device=device
            ),
            state_x=torch.empty(state, dtype=torch.float32, device=device),
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
            unit_chunks=plan.unit_chunks,
            b_t=B_T,
            expand_num=1,
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

    n_tiles = num_seqs * heads_out * plan.tiles_per_head
    ideal, rows, item_scratch, chunk_scratch = _split_scratch(
        split, tokens, num_seqs, heads_out, plan.num_sm, n_tiles, device
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
        scheduler=_int32(2, device),
        workspace=_workspace(gdn_prefill_f16, num_seqs, device),
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
        expand_num=1,
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


def _split_scratch(
    split: bool, tokens: int, num_seqs: int, heads_out: int, num_sm: int, n_tiles: int, device
):
    """``(ideal_chunks, work_item_rows, item_scratch, chunk_scratch)`` of the warmup split table."""
    if not split:
        return None, n_tiles, None, None
    ideal = compute_ideal_chunks(tokens, heads_out, num_sm, B_T)
    rows = max_work_items(tokens, num_seqs, heads_out, ideal, B_T, num_sm)
    item_scratch = torch.empty(rows, WORK_ITEM_FIELDS, dtype=torch.int32, device=device)
    chunk_scratch = torch.empty(
        chunk_scratch_rows(tokens, num_seqs, B_T), heads_out, dtype=torch.float32, device=device
    )
    return ideal, rows, item_scratch, chunk_scratch


def _piece_buffers(num_seqs: int, pieces: int, heads_out: int, device) -> dict:
    """Piece-table views and work-item tables of an exact piece chain."""
    num_pieces = num_seqs * pieces
    table = piece_table_layout(num_seqs, pieces, heads_out)
    piece_table = torch.zeros(table.nbytes // 4, dtype=torch.int32, device=device)

    def words(offset: int, count: int) -> torch.Tensor:
        return piece_table[offset // 4 : offset // 4 + count]

    return {
        "cu_pieces": words(table.cu_pieces, num_pieces + 1),
        "main_rows": words(table.main_rows, num_seqs + 1),
        "summary_rows": words(table.summary_rows, num_seqs + 1),
        "main_count": words(table.main_count, 1),
        "summary_count": words(table.summary_count, 1),
        "work_items": torch.empty(
            num_pieces * heads_out, WORK_ITEM_FIELDS, dtype=torch.int32, device=device
        ),
        "work_items_summary": torch.empty(
            table.item_rows, WORK_ITEM_FIELDS, dtype=torch.int32, device=device
        ),
    }


def _tinv_buffers(tokens: int, num_pieces: int, heads_out: int, dtype, device) -> dict:
    """Chunk-inverse (T^-1) scratch shared by the chain and the backward stages."""
    rows = gdn_tinv_f16.tinv_rows(tokens, num_pieces, 1, B_T)
    return {
        "tinv": torch.empty(rows, heads_out, B_T, B_T, dtype=dtype, device=device),
        "tinv_words": _workspace(gdn_tinv_f16, num_pieces, device),
        "tinv_rows": torch.empty(rows, 4, dtype=torch.int32, device=device),
        "tinv_row_count": _int32(1, device),
    }


@dataclass
class BackwardPlan:
    """Uncut or exact-chain backward scheme for one shape; mirrors ``ForwardPlan``."""

    pieces: int
    unit_chunks: int
    num_sm: int

    @property
    def chain(self) -> bool:
        return self.pieces > 0

    @classmethod
    def build(cls, tokens: int, num_seqs: int, heads_out: int, device) -> BackwardPlan:
        num_sm = torch.cuda.get_device_properties(device).multi_processor_count
        pieces, unit = choose_pieces(
            num_seqs=num_seqs,
            heads_out=heads_out,
            num_sm=num_sm,
            total_tokens=tokens,
            b_t=B_T,
            cadence_tokens=0,
            batch_invariant=False,
            expand_num=1,
            reverse=True,
        )
        if pieces and tokens < MIN_CHAIN_TOKENS_PER_PIECE_BWD * pieces * num_seqs:
            pieces = 0
        return cls(pieces, unit, num_sm)


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
        _aligned(t.detach()) for t in (q, k, v, gate, beta, d_output, cu_seqlens)
    )
    initial_state = None if initial_state is None else initial_state.detach()
    d_final_state = None if d_final_state is None else d_final_state.detach()
    tokens, key_heads, dim_k = q.shape
    dim_v = v.shape[-1]
    heads_out = gate.shape[1]
    num_seqs = cu_seqlens.shape[0] - 1
    device = q.device
    plan = BackwardPlan.build(tokens, num_seqs, heads_out, device)
    if split:
        plan = replace(plan, pieces=0)
    stream = torch.cuda.current_stream(device).cuda_stream
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
    bprop_words = _workspace(gdn_bprop_f16, num_pieces, device)
    common = dict(
        q=q,
        k=k,
        v=v,
        do=d_output,
        gate=gate,
        beta=beta,
        a_log=None,
        dt_bias=None,
        cu_seqlens=cu_seqlens,
        checkpoints=checkpoints,
        seed_checkpoints=None,
        dq=dq_ho,
        dk=dk_ho,
        dv=dv,
        bprop_words=bprop_words,
        **_tinv_buffers(tokens, num_pieces, heads_out, q.dtype, device),
    )

    if plan.chain:
        schedulers = empty(10)
        state = (num_pieces, heads_out, dim_v, dim_k)
        buffers = dict(
            common,
            **_piece_buffers(num_seqs, plan.pieces, heads_out, device),
            series_items=None,
            series_count=None,
            scheduler_all=schedulers,
            scheduler_recompute=schedulers[0:2],
            scheduler_bwd=schedulers[2:4],
            scheduler_summary=schedulers[4:6],
            scheduler_m=schedulers[6:8],
            scheduler_series=schedulers[8:10],
            summary_words=_workspace(gdn_summary_f16, num_pieces, device),
            recompute_m_words=_workspace(gdn_recompute_f16, num_pieces, device),
            series_words=_workspace(gdn_recompute_f16, num_pieces, device),
            bprop_summary_words=_workspace(gdn_bprop_summary_f16, num_pieces, device),
            dgate=dgate,
            dbeta=dbeta,
            summary_q=q,
            summary_do=d_output,
            state_h=empty(*state, dtype=torch.float32),
            state_m=empty(num_pieces, heads_out, dim_k, dim_k, dtype=torch.float32),
            state_x=empty(*state, dtype=torch.float32),
            state_g=empty(*state, dtype=torch.float32),
            state_dx_end=empty(*state, dtype=torch.float32),
            seed=initial_state,
            dseed=d_final_state,
            dstate0=d_initial_state,
            inv_q=None,
            inv_k=None,
        )
        schedule = {
            "pieces": plan.pieces,
            "heads_out": heads_out,
            "num_seqs": num_seqs,
            "b_t": B_T,
            "compact_qdo": False,
            "fused_h_m": True,
            "series": True,
            "coarse": False,
            "series_span_tokens": 0,
            "seed_every_n_tokens": 0,
            "scale": scale,
        }
        launch = build_chain_backward(
            bprop_module=gdn_bprop_f16,
            **buffers,
            **schedule,
            unit_chunks=plan.unit_chunks,
            expand_num=1,
            length_rule=False,
            summary_q_step=1,
            **_GATE_FLAGS,
            chain_rows=chain_rows_per_cta(dim_v, dim_k, num_seqs, heads_out, plan.num_sm),
            device=device.index,
            num_sm=plan.num_sm,
            stream=stream,
        )
        run_chain_backward(launch, **buffers, **schedule, stream=stream)
    else:
        schedulers = empty(4)
        ideal, rows, item_scratch, chunk_scratch = _split_scratch(
            split, tokens, num_seqs, heads_out, plan.num_sm, num_seqs * heads_out, device
        )
        work_items = empty(rows, WORK_ITEM_FIELDS)
        work_count = _work_count(device)
        buffers = dict(
            common,
            state_in=initial_state,
            work_items=work_items,
            work_count=work_count,
            series_items=work_items,
            series_count=work_count,
            item_scratch=item_scratch,
            chunk_scratch=chunk_scratch,
            scheduler_all=schedulers,
            scheduler_recompute=schedulers[0:2],
            recompute_words=_workspace(gdn_recompute_f16, num_pieces, device),
        )
        stages = {
            "b_t": B_T,
            "tinv_pass": True,
            "recompute": True,
            "recompute_orders": True,
            "coarse": False,
            "bwd_orders": False,
            "compact_qdo": False,
            "seed_span_tokens": 0,
            "seed_every_n_tokens": 0,
        }
        warmup = build_warmup_backward(
            bprop_module=gdn_bprop_f16,
            **buffers,
            **stages,
            split=split,
            n_tiles=num_seqs * heads_out,
            ideal_chunks=ideal,
            num_sm=plan.num_sm,
            expand_num=1,
            **_GATE_FLAGS,
            device=device.index,
            stream=stream,
        )
        run_warmup_backward(*warmup, **buffers, **stages, stream=stream)
        gdn_bprop_f16.chunk_gdn_bwd(
            q,
            k,
            v,
            gate,
            beta,
            d_output,
            checkpoints,
            dq_ho,
            dk_ho,
            dv,
            dgate,
            dbeta,
            cu_seqlens,
            scale,
            use_initial_state=initial_state is not None,
            d_initial_state=d_initial_state,
            d_final_state=d_final_state,
            **_GATE_FLAGS,
            work_items=work_items,
            work_count=work_count,
            scheduler_counter=schedulers[2:4],
            workspace=bprop_words,
            device=device.index,
            num_sm=plan.num_sm,
            stream=stream,
            own_prologue=False,
            tinv=common["tinv"],
        )

    groups = heads_out // key_heads
    dq = group_sum(dq_ho, groups, out_dtype=q.dtype) if groups > 1 else dq_ho
    dk = group_sum(dk_ho, groups, out_dtype=k.dtype) if groups > 1 else dk_ho
    return dq, dk, dv, dgate, dbeta, d_initial_state
