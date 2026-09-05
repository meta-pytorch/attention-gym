# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Affine state-summary algebra shared by every delta-rule variant.

Each delta-rule token step is affine in the V-major recurrent state ``H: [V, K]`` (one per value
head ``HV``; GQA key heads are already expanded by the factor kernels)::

    H_t = H_{t-1} @ A_t + B_t
    A_t = diag(exp g_t) (I - beta_t k_t k_t^T)
    B_t = beta_t v_t k_t^T

so any token range collapses to one FP32 map ``H_out = H_in @ A + B``, packed here as
``[HV, V + K, K] = [bias; transition]``. Reverse summaries pack the cotangent map
``dH_in = dH_out @ R + C`` the same way. These helpers are pure PyTorch; the per-op ``stages``
modules produce the summaries and ``attn_gym.linear.context_parallel`` moves them between ranks.

NOTE [Terminology]
The staged primitives, ownership plans, and any context-parallel recipe over them share one
vocabulary and two index spaces. GLOBAL means the whole stream one context-parallel group
processes: one data-parallel replica's tokens, so "global" here is DP-local. ``cp_rank`` is the
rank within that CP group. Anything a plan is built from is GLOBAL; anything that touches a tensor
on a rank is LOCAL.

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
    slot            ``gathered[cp_rank][f]``: one per fragment. Forward, the   (cp_rank, fragment)
                    summary of fragment f's last subsequence; backward, the
                    reverse map of its first; the identity ``[0; I]`` when
                    nobody needs it
    predecessors /  a fragment's predecessors are the slots holding its       (cp_rank, fragment)
    successors      sequence's earlier pieces, in token order; its successors
                    the later ones. The only cross-rank references a plan holds
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

    slot order    s1: A∩s1 -> C∩s1 -> D∩s1        s2: D∩s2 -> B∩s2        s0: A∩s0 alone

Who passes which indices, for rank 1 above. Names ending in ``_global`` live in the GLOBAL index
space; everything else is LOCAL to this rank's span. The plan is host metadata built from Python
integers; ``plan.routing(device)`` turns it into the span-local tensors the kernels read::

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

    # Device: all offsets LOCAL, all of them read from the routing tensors.
    routing = plan.routing(device)
    prepared = chunk_kda_prepare(q, k, v, gate, beta, cu_seqlens=routing.cu_seqlens)
    slots = summary_slots(prepared, routing)                  # state_summaries(forward_bounds)
    #   one entry per local fragment, in span order; routing.forward_bounds is
    #   slot 0 (frag C): [0, 96)      C∩s1 is C's last subsequence, continued by D∩s1
    #   slot 1 (frag D): [136, 192)   D∩s2 is D's last subsequence, continued by B∩s2
    gathered = all_gather(slots)                              # [world, plan.slots, HV, V + K, K]
    initial_state = compose_entry_states(gathered, routing)   # one per subsequence
    #   C∩s1: merge(0, gathered[0][0])                        A∩s1 is rank 0's fragment A
    #   D∩s1: merge(merge(0, gathered[0][0]), gathered[1][0])
    #   D∩s2: 0
    output, exit_states = prepared.run(initial_state, output_final_state=True)
    #   exit_states holds one state per subsequence; only routing.terminal rows are a sequence's
    #   true final state (here D∩s1's, for s1), the rest are intermediate. Callers that never use
    #   final states pass output_final_state=False; the backward needs neither.
    exit_states[routing.terminal]

The key contract: the routing hands ``state_summaries`` / ``state_grad_summaries`` exactly one
whole subsequence per active fragment slot, as consecutive entries of the span's own
``cu_seqlens`` (LOCAL). Direct callers must obey NOTE [Summary ranges are whole chunks of one
subsequence] in ``kda.stages`` themselves; the bounds' shape and dtype are checked, their values
are not.

What the table does not constrain:

- Ranks need not own equal token counts. Nothing is exchanged per token; the gather is over
  ``plan.slots`` fixed-shape summaries, so spans of different lengths are fine.
- Zero-length sequences are legal. An empty global sequence owns no tokens, so the plan has no
  subsequence for it; the empty span segments that ``routing(max_subsequences=N)`` pads with are
  repeated ``cu_seqlens`` entries, which the kernels treat as empty sequences with unused state
  rows, and no summary is ever requested for them.
- Summary work is bounded by fragments, not documents. Only a fragment's last subsequence can have
  a successor and only its first can have predecessors; interior subsequences are whole documents
  that start from zero. So a rank computes at most one summary per fragment it owns, the gather
  is one slot per fragment, and no fragment has more predecessors or successors than there are
  other fragments in the table.

CUDA Graph replay follows from that bound. ``plan.routing(device)`` turns a layout into device
tensors: the span's ``cu_seqlens``, one ``[start, stop)`` per fragment for the summary launches
(empty when the fragment sends nothing), per-fragment predecessor and successor slots as flat
indices padded with an identity slot, and per-segment source maps. Eager callers take the default
sizes. For replay, pass the caps the loader promises never to exceed, ``routing(device, slots=S,
max_subsequences=N)``: every layout within them, whatever its documents or fragment table, yields
identically shaped tensors, so a recipe that reads every index from them is captured once and
replayed after copying the next layout's tensors in place. The span length is the one shape the
loader must hold fixed (pad a short batch with a loss-masked padding document).
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
