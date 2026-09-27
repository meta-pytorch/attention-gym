# SPDX-License-Identifier: BSD-3-Clause

"""Torch drivers for the vendored cudnn-frontend v1.30 KDA kernels.

The BT16 plans mirror KdaFrostEngine: uncut, value-dimension split (with shared
prep), exact piece chain, or opt-in approximate forgetting-horizon split. Builders
cache compiled launches by static config; shape-dependent scratch is rebuilt per call.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from attn_gym._backends.cute.utils import get_device_properties

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

B_T = kda_prefill_f16.CFG.B_T
# Start with the GDN driver's conservative chain floors; KDA has its own BT16 pieces.
# These guard against summary overhead, rather than claiming KDA-specific tuning.
MIN_CHAIN_TOKENS_PER_PIECE_FWD = 8192
MIN_CHAIN_TOKENS_PER_PIECE_BWD = 2048
PREP_TILE_FRACTION = 0.475
_GATE_FLAGS = {
    "log_gate": True,
    "safe_gate": False,
    "use_beta_sigmoid": False,
    "allow_neg_eigval": False,
    "use_qk_l2norm": False,
    "gate_lower_bound": kda_prefill_f16.DEFAULT_GATE_LOWER_BOUND,
}


def _int32(n: int, device) -> torch.Tensor:
    return torch.empty(n, dtype=torch.int32, device=device)


def _workspace(module, count: int, device) -> torch.Tensor:
    return torch.empty(
        tensormap_workspace_bytes(module, count) // 8, dtype=torch.int64, device=device
    )


def _aligned(tensor: torch.Tensor) -> torch.Tensor:
    """The upstream DLPack signatures assume 16-byte bases, including beta."""
    return tensor if tensor.data_ptr() % 16 == 0 else tensor.clone()


def _work_count(device) -> torch.Tensor:
    return _int32(1, device)


@dataclass
class ForwardPlan:
    """Scheme and scratch sizes for one shape/device; never cached across shapes."""

    pieces: int
    unit_chunks: int
    tiles_per_head: int
    num_sm: int

    @property
    def chain(self) -> bool:
        return self.pieces > 0

    # Plans intentionally depend on shape; bitwise ownership invariance holds within a plan.
    @classmethod
    def build(cls, tokens: int, num_seqs: int, heads_out: int, dim_v: int, device) -> ForwardPlan:
        num_sm = get_device_properties(device).multi_processor_count
        pieces, unit = choose_pieces(
            num_seqs=num_seqs,
            heads_out=heads_out,
            num_sm=num_sm,
            total_tokens=tokens,
            b_t=B_T,
            cadence_tokens=0,
            batch_invariant=False,
            expand_num=1,
        )
        if pieces and tokens < MIN_CHAIN_TOKENS_PER_PIECE_FWD * pieces * num_seqs:
            pieces = 0
        dv = not pieces and dim_v == 128 and num_sm // (num_seqs * heads_out) >= DV_SPLIT_TILES
        return cls(pieces, unit, DV_SPLIT_TILES if dv else 1, num_sm)


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
        _aligned(t.detach()) for t in (q, k, v, gate, beta, cu_seqlens)
    )
    if paged:
        if initial_state is None or output_final_state or split:
            raise ValueError("paged KDA needs a state pool and no final state or split schedule")
        if initial_state.data_ptr() % 16:
            raise ValueError("the paged state pool must be 16-byte aligned")
        initial_state = initial_state.detach()
    else:
        initial_state = None if initial_state is None else _aligned(initial_state.detach())
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
            scheduler_all=schedulers,
            scheduler_summary=schedulers[2:4],
            scheduler_prefill=schedulers[0:2],
            summary_words=_workspace(kda_summary_f16, num_pieces, device),
            prefill_words=_workspace(kda_prefill_f16, num_pieces, device),
            state_h=torch.empty(state, dtype=torch.float32, device=device),
            state_m=torch.empty(
                num_pieces, heads_out, dim_k, dim_k, dtype=torch.float32, device=device
            ),
            state_x=torch.empty(state, dtype=torch.float32, device=device),
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
    ideal, rows, item_scratch, chunk_scratch = _split_scratch(
        split, tokens, num_seqs, heads_out, plan.num_sm, n_tiles, device
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
        scheduler=_int32(2, device),
        workspace=_workspace(kda_prep_prefill_f16 if prep else kda_prefill_f16, num_seqs, device),
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
    rows = gdn_tinv_f16.tinv_rows(tokens, num_seqs, 1, B_T)
    buffers = {
        name: torch.empty(rows, heads_out, B_T, dim_k, dtype=dtype, device=device)
        for name in ("prep_k_decay", "prep_q_decay", "prep_t")
    }
    return dict(
        buffers,
        prep_a=torch.empty(rows, heads_out, B_T * B_T // 2, dtype=torch.int32, device=device),
        prep_diag=torch.empty(rows, heads_out, dim_k, dtype=torch.float32, device=device),
        prep_words=_workspace(kda_prep_f16, num_seqs, device),
        prep_rows=torch.empty(rows, 4, dtype=torch.int32, device=device),
        prep_row_count=_int32(1, device),
    )


def _split_scratch(
    split: bool, tokens: int, num_seqs: int, heads_out: int, num_sm: int, n_tiles: int, device
):
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


@dataclass
class BackwardPlan:
    pieces: int
    unit_chunks: int
    num_sm: int

    @property
    def chain(self) -> bool:
        return self.pieces > 0

    @classmethod
    def build(cls, tokens: int, num_seqs: int, heads_out: int, device) -> BackwardPlan:
        num_sm = get_device_properties(device).multi_processor_count
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
        _aligned(t.detach()) for t in (q, k, v, gate, beta, d_output, cu_seqlens)
    )
    initial_state = None if initial_state is None else _aligned(initial_state.detach())
    d_final_state = None if d_final_state is None else _aligned(d_final_state.detach())
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
        "a_log": None,
        "dt_bias": None,
        "cu_seqlens": cu_seqlens,
        "checkpoints": checkpoints,
        "seed_checkpoints": None,
        "dq": dq,
        "dk": dk,
        "dv": dv,
        "dgate": dgate,
        "dbeta": dbeta,
        "dstate0": d_initial_state,
        "bprop_words": _workspace(kda_bprop_f16, num_pieces, device),
    }
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
            summary_words=_workspace(kda_summary_f16, num_pieces, device),
            recompute_m_words=_workspace(kda_recompute_f16, num_pieces, device),
            series_words=_workspace(kda_recompute_f16, num_pieces, device),
            bprop_summary_words=_workspace(kda_bprop_summary_f16, num_pieces, device),
            state_h=empty(*state, dtype=torch.float32),
            state_m=empty(num_pieces, heads_out, dim_k, dim_k, dtype=torch.float32),
            state_x=empty(*state, dtype=torch.float32),
            state_g=empty(*state, dtype=torch.float32),
            state_dx_end=empty(*state, dtype=torch.float32),
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
            "log_gate": True,
            "safe_gate": False,
        }
        launch = build_chain_backward(
            **buffers,
            **schedule,
            unit_chunks=plan.unit_chunks,
            length_rule=False,
            gate_lower_bound=_GATE_FLAGS["gate_lower_bound"],
            use_qk_l2norm=False,
            use_beta_sigmoid=False,
            allow_neg_eigval=False,
            chain_rows=chain_rows_per_cta(dim_v, dim_k, num_seqs, heads_out, plan.num_sm),
            num_sm=plan.num_sm,
        )
        run_chain_backward(launch, **buffers, **schedule)
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
            recompute_words=_workspace(kda_recompute_f16, num_pieces, device),
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
