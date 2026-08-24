# SPDX-License-Identifier: BSD-3-Clause

"""Shared work-table and scratch allocation for Mega forward and backward."""

from __future__ import annotations

from typing import NamedTuple

import torch

from .kernels.common.split_k import (
    WORK_ITEM_FIELDS,
    build_split_table,
    chunk_scratch_rows,
    compute_ideal_chunks,
    max_work_items,
)
from .kernels.compat import multiprocessor_count, tensor_device_index


class MegaSchedule(NamedTuple):
    """Allocated work table, counters, and optional split scratch for one launch."""

    work_items: torch.Tensor
    work_count: torch.Tensor
    counters: torch.Tensor
    item_scratch: torch.Tensor | None
    chunk_scratch: torch.Tensor | None
    num_sms: int


def prepare_mega_schedule(
    gate: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    tile_tokens: int,
    counter_count: int,
    split: bool,
    stream: int,
) -> MegaSchedule:
    """Select geometry and allocate every host-owned Mega scheduling buffer."""
    if counter_count < 1:
        raise ValueError("counter_count must be positive")
    tokens, heads = gate.shape[1:3]
    num_sequences = cu_seqlens.shape[0] - 1
    num_sms = multiprocessor_count(tensor_device_index(gate))
    work_count = torch.empty(1, dtype=torch.int32, device=gate.device)
    counters = torch.empty(counter_count, dtype=torch.int32, device=gate.device)

    if not split:
        return MegaSchedule(
            torch.empty(
                num_sequences * heads,
                WORK_ITEM_FIELDS,
                dtype=torch.int32,
                device=gate.device,
            ),
            work_count,
            counters,
            None,
            None,
            num_sms,
        )

    ideal_chunks = compute_ideal_chunks(tokens, heads, num_sms, tile_tokens)
    work_item_rows = max_work_items(
        tokens,
        num_sequences,
        heads,
        ideal_chunks,
        tile_tokens,
        num_sms,
    )
    item_scratch = torch.empty(
        work_item_rows,
        WORK_ITEM_FIELDS,
        dtype=torch.int32,
        device=gate.device,
    )
    chunk_scratch = torch.empty(
        chunk_scratch_rows(tokens, num_sequences, tile_tokens),
        heads,
        dtype=torch.float32,
        device=gate.device,
    )
    work_items = torch.empty_like(item_scratch)
    build_split_table(
        gate[0],
        cu_seqlens,
        work_items,
        work_count,
        ideal_chunks=ideal_chunks,
        n_tiles=num_sequences * heads,
        num_sms=num_sms,
        b_t=tile_tokens,
        chunk_scratch=chunk_scratch,
        item_scratch=item_scratch,
        log_gate=True,
        sched_ctr=counters,
        split=True,
        stream=stream,
    )
    return MegaSchedule(
        work_items,
        work_count,
        counters,
        item_scratch,
        chunk_scratch,
        num_sms,
    )


__all__ = ["MegaSchedule", "prepare_mega_schedule"]
