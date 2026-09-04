# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Affine state-summary algebra shared by every delta-rule variant.

Each delta-rule token step is affine in the V-major recurrent state ``H: [V, K]`` (one per value
head ``HV``; GQA key heads are already expanded by the factor kernels)::

    H_t = H_{t-1} @ A_t + B_t      A_t = diag(exp g_t) (I - beta_t k_t k_t^T),  B_t = beta_t v_t k_t^T

so any token range collapses to one FP32 map ``H_out = H_in @ A + B``, packed here as
``[HV, V + K, K] = [bias; transition]``. Reverse summaries pack the cotangent map
``dH_in = dH_out @ R + C`` the same way. These helpers are pure PyTorch; the per-op ``stages``
modules produce the summaries and the caller moves them between devices.

NOTE [Terminology]
The staged primitives, ownership plans, and the context-parallel recipe share one vocabulary and
two index spaces. GLOBAL means the whole stream one context-parallel group processes: one
data-parallel replica's tokens, so "global" here is DP-local. ``cp_rank`` is the rank within that
CP group. Anything a plan is built from is GLOBAL; anything that touches a tensor on a rank is
LOCAL.

    global stream   the whole packed token stream, ``cu_seqlens_global``            GLOBAL
    sequence        one document in the global stream                               GLOBAL
    fragment        one contiguous global token range a rank owns; may cover        GLOBAL
                    several sequences; a rank owns a list of them (one for a
                    contiguous shard, two for zig-zag, more for finer balancing).
                    The loader or sharder chooses the fragment table; the plan
                    only derives routing from it
    subsequence     the tokens of one fragment that belong to one sequence; a       GLOBAL
                    fragment covering several sequences has several subsequences,
                    and each becomes one ``cu_seqlens`` segment of the owner's span
    span            the concatenation of a rank's fragments, in the order it        LOCAL
                    listed them: the packed ``q``/``k``/``v``/``output`` tensors on
                    that rank, with ``cu_seqlens`` marking its subsequence boundaries
    chunk           the ``kda.stages.CHUNK_SIZE`` (64) token block the fused         LOCAL
                    kernels work in (WY factors and one state step per block);
                    every subsequence is chunked on its own, starting at its first
                    token, so its last chunk may be partial and no fragment cut
                    needs to be chunk-aligned
    summary         the ``[bias; transition]`` map of a token range of the span     LOCAL range
    slot            ``gathered[rank][i]``: the summary of that rank's i-th          (rank, index)
                    subsequence, or the identity ``[0; I]`` when nobody needs it
    predecessors /  subsequences of the same sequence earlier / later in global     (rank, slot)
    successors      order; the only cross-rank references a plan holds
    terminal        subsequences that end their sequence; true final states         LOCAL index
                    live there

How they nest, for ``cu_seqlens_global = (0, 40, 232, 384)`` and the zig-zag fragment table
``fragments_global = [[(0, 96), (288, 384)], [(96, 192), (192, 288)]]``::

    GLOBAL
    tokens      0     40             96       192   232      288      384
    sequences   |- s0 -|------------ s1 --------------|-------- s2 --------|
    rank 0       [====== frag A ======]                        [= frag B =]
    rank 1                             [ frag C ][= frag D ==]

    LOCAL (one span per rank; each cell is one subsequence; cu_seqlens values are that rank's own)
    rank 0 span   A∩s0 | A∩s1 | B∩s2        cu_seqlens (0, 40, 96, 192)
    rank 1 span   C∩s1 | D∩s1 | D∩s2        cu_seqlens (0, 96, 136, 192)

    chains        s1: A∩s1 -> C∩s1 -> D∩s1        s2: D∩s2 -> B∩s2        s0: A∩s0 alone

Who passes which indices, for rank 1 above. Names ending in ``_global`` live in the GLOBAL index
space; everything else is LOCAL to this rank's span. The plan's inputs and every plan field are
host Python integers; the only offsets tensor the kernels ever see is the span's own::

    # Host: routing; ``cp_rank`` picks this rank's row of the fragment table.
    cu_seqlens_global = (0, 40, 232, 384)
    fragments_global = [[(0, 96), (288, 384)], [(96, 192), (192, 288)]]
    plan = ContextParallelPlan.from_fragments(cu_seqlens_global, fragments_global, cp_rank=1)
    plan.subsequences -> (C∩s1 [96,192), D∩s1 [192,232), D∩s2 [232,288))
    plan.cu_seqlens -> (0, 96, 136, 192)

    # Device: the loader lays out fragments_global[1] back to back, in that order, so the span's
    # tokens match plan.subsequences.
    input_ids = <fragments_global[1] flattened by the loader>                 # [1, 192]
    q, k, v, gate, beta = embed / project / short-conv the span               # [1, 192, HV, 128]
    cu_seqlens = torch.tensor(plan.cu_seqlens, dtype=torch.int32, device=device)

    # Device: all offsets LOCAL.
    prepared = chunk_kda_prepare(q, k, v, gate, beta, cu_seqlens=cu_seqlens)
    slots = summary_slots(prepared, plan, neutral_summary(HV, 128, 128, device=device))
    #   one entry per local subsequence, in span order, padded to plan.slots:
    #   slot 0: state_summaries(bounds)[0] over [0, 96)     C∩s1 has a successor (D∩s1)
    #   slot 1: identity [0; I]                             D∩s1 ends s1, nobody needs it
    #   slot 2: state_summaries(bounds)[2] over [136, 192)  D∩s2 has a successor (rank 0's B∩s2)
    gathered = all_gather(slots)                              # [world, plan.slots, HV, V + K, K]
    initial_state = compose_entry_states(gathered, plan)      # one per subsequence
    #   C∩s1: merge(0, gathered[0][1])
    #   D∩s1: merge(merge(0, gathered[0][1]), gathered[1][0])
    #   D∩s2: 0
    output, exit_states = prepared.run(initial_state, output_final_state=True)
    #   exit_states holds one state per subsequence; only plan.terminal entries are a sequence's
    #   true final state (here D∩s1's, for s1), the rest are intermediate. Callers that never use
    #   final states pass output_final_state=False; the backward needs neither.
    exit_states[list(plan.terminal)]

The key contract: ``state_summary`` is always called with consecutive entries of the span's own
``plan.cu_seqlens`` (LOCAL), so every summary covers exactly one whole subsequence.

What the table does not constrain:

- Ranks need not own equal token counts. Nothing is exchanged per token; the gather is over
  ``plan.slots`` fixed-shape summaries, so spans of different lengths are fine.
- Zero-length subsequences are legal: the kernels treat a repeated ``cu_seqlens`` entry as an
  ordinary empty sequence with an unused state row, and the plan never asks one for a summary.
- Summary work is bounded by fragments, not documents. Only a fragment's last subsequence can have
  a successor and only its first can have predecessors; interior subsequences are whole documents
  that start from zero. So a rank computes at most one summary per fragment it owns, and no chain
  is longer than the number of fragments in the table.

CUDA Graph capture is valid for one *plan*, not merely one set of shapes. Fixing shapes is the
easy part: the loader pads the span with a padding *document* (its own sequence, loss-masked) and
the subsequence count could be padded with empty ones. ``state_summaries(bounds)`` already reads
its ranges on the device, but ``compose_entry_states`` bakes the ``(rank, slot)`` chains into
indexing ops, so a replay also needs the same ``cu_seqlens_global`` and therefore the same
subsequence layout. Replaying across sampled document layouts needs device-tensor routing as
well; that does not exist yet.
"""

from __future__ import annotations

import torch


def neutral_summary(
    heads: int, value_dim: int, key_dim: int, *, device: torch.device | str
) -> torch.Tensor:
    """Return the identity map ``[0; I]``: ``merge_state(state, neutral) == state``.

    Collectives need every participant to contribute a same-shaped tensor even when nobody
    consumes its slot; the identity is the harmless filler.
    """
    summary = torch.zeros(heads, value_dim + key_dim, key_dim, dtype=torch.float32, device=device)
    summary[:, value_dim:, :] = torch.eye(key_dim, dtype=torch.float32, device=device)
    return summary


def merge_state(state: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
    """Apply a packed ``[bias; transition]`` summary to a V-major state: ``state @ A + B``.

    The product runs in FP64 so the result does not depend on the caller's
    ``torch.get_float32_matmul_precision()``: under ``"high"`` an FP32 matmul silently rounds
    the operands to TF32, and these maps are small enough that the cast is negligible.
    """
    value_dim = state.shape[-2]
    merged = state.double() @ summary[..., value_dim:, :].double() + summary[..., :value_dim, :]
    return merged.to(state.dtype)


def compose_summaries(first: torch.Tensor, then: torch.Tensor) -> torch.Tensor:
    """Compose two summaries into the map that applies ``first`` and then ``then``.

    ``(A0, B0) ∘ (A1, B1) = (A0 @ A1, B0 @ A1 + B1)``. Folding predecessors from the zero state
    with ``merge_state`` gives the same entry state; composition is for scans that combine maps
    before any state is known.
    """
    value_dim = first.shape[-2] - first.shape[-1]
    bias, transition = first[..., :value_dim, :], first[..., value_dim:, :]
    composed_transition = (transition.double() @ then[..., value_dim:, :].double()).to(first.dtype)
    return torch.cat((merge_state(bias, then), composed_transition), dim=-2)


__all__ = ["compose_summaries", "merge_state", "neutral_summary"]
