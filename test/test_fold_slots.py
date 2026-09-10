# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The fused slot fold behind ``fold_slots`` on CUDA."""

import pytest
import torch

from attn_gym.linear._delta_rule.triton.fold_slots import fold_slots_fused
from attn_gym.linear.context_parallel import fold_slots, merge_state, neutral_summary

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
HEADS, KEY, VALUE = 4, 128, 128


def _gathered(world: int, slots: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(0)
    bias = torch.randn(world, slots, HEADS, VALUE, KEY, device="cuda", generator=generator)
    transition = torch.eye(KEY, device="cuda") + 0.05 * torch.randn(
        world, slots, HEADS, KEY, KEY, device="cuda", generator=generator
    )
    return torch.cat((bias, transition), dim=3)


def _loop(gathered: torch.Tensor, sources: torch.Tensor) -> torch.Tensor:
    """The IEEE FP32 one-merge-per-step reference (the CPU path of ``fold_slots``)."""
    flat = gathered.flatten(0, 1)
    flat = torch.cat((flat, neutral_summary(HEADS, VALUE, KEY, device="cuda").unsqueeze(0)))
    state = flat.new_zeros(sources.shape[0], HEADS, VALUE, KEY)
    for step in range(sources.shape[1]):
        state = merge_state(state, flat[sources[:, step]])
    return state


def test_fused_fold_matches_ieee_merges_and_ignores_padding():
    gathered = _gathered(world=4, slots=3)
    pad = 12
    sources = torch.tensor(
        [
            [0, 4, 8, pad, pad],
            [pad] * 5,
            [5, pad, pad, pad, pad],
            [1, 2, 3, 4, 5],
            [11, 10, pad, pad, pad],
        ],
        device="cuda",
    )
    fused = fold_slots(gathered, sources)
    torch.testing.assert_close(fused, _loop(gathered, sources))
    assert torch.equal(fused[1], torch.zeros_like(fused[1]))
    # A longer padded chain table is the same arithmetic: padding is never applied.
    longer = torch.cat((sources, sources.new_full((5, 3), pad)), dim=1)
    assert torch.equal(fold_slots(gathered, longer), fused)
    assert fold_slots(gathered, sources[:0]).shape == (0, HEADS, VALUE, KEY)
    assert torch.equal(fold_slots(gathered, sources[:, :0]), torch.zeros_like(fused))


def test_fused_fold_replays_in_a_graph_across_index_changes():
    """The launch is fixed by the table shape; the indices may change between replays."""
    gathered = _gathered(world=3, slots=2)
    pad = 6
    sources = torch.tensor([[0, 2, 4, pad], [pad] * 4], device="cuda")
    fold_slots_fused(gathered.flatten(0, 1), sources)  # warm up outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = fold_slots(gathered, sources)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(replayed, fold_slots(gathered, sources))
    sources.copy_(torch.tensor([[5, 3, 1, pad], [4, 2, pad, pad]], device="cuda"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(replayed, fold_slots(gathered, sources))
