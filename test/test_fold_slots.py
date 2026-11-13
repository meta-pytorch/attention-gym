# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The fused slot fold behind ``fold_slots`` on CUDA."""

import pytest
import torch
import triton

from attn_gym.linear._delta_rule.triton.fold_slots import fold_slots_fused, fold_slots_linked
from attn_gym.linear.context_parallel import (
    ContextParallelPlan,
    fold_slots,
    merge_state,
    neutral_summary,
)

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


@pytest.mark.parametrize(("key", "value"), [(16, 32), (128, 128)])
@pytest.mark.parametrize("reverse", [False, True])
def test_linked_fold_matches_independent_bits_and_replays(key, value, reverse):
    generator = torch.Generator(device="cuda").manual_seed(869)
    maps = torch.randn(32, 2, value + key, key, device="cuda", generator=generator)
    maps[:, :, value:] *= 0.02
    fragments = [[(i * 8, (i + 1) * 8) for i in range(rank, 16, 2)] for rank in range(2)]
    direction = "successors" if reverse else "predecessors"
    link_name = "successor_links" if reverse else "predecessor_links"

    def routing(offsets, table):
        return ContextParallelPlan.from_fragments(offsets, table, 1).routing("cuda", slots=16)

    initial = routing((0, 128), fragments)
    sources = getattr(initial, direction)
    links = getattr(initial, link_name)
    assert (links[:, 2] > 1).any()
    actual = fold_slots_linked(maps, sources, links)
    assert torch.equal(actual, fold_slots_fused(maps, sources))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = fold_slots_linked(maps, sources, links)
    for offsets, table in [
        ((0, 13, 13, 57, 128), [list(reversed(row)) for row in fragments]),
        ((0, 128), [[(0, 64)], [(64, 128)]]),
        ((0, 128), fragments),
    ]:
        changed = routing(offsets, table)
        sources.copy_(getattr(changed, direction))
        links.copy_(getattr(changed, link_name))
        maps[:, :, :value].add_(0.1)
        actual.fill_(float("nan"))
        graph.replay()
        assert torch.equal(actual, fold_slots_fused(maps, sources))


@pytest.mark.parametrize(("shape", "dtype"), [((2, 2), torch.int32), ((2, 3), torch.int64)])
def test_linked_fold_rejects_malformed_links(shape, dtype):
    maps = torch.empty(4, 1, 48, 16)
    sources = torch.empty(2, 3, dtype=torch.int64)
    links = torch.empty(shape, dtype=dtype)
    with pytest.raises(ValueError, match="links must be int32"):
        fold_slots_linked(maps, sources, links)


def test_linked_fold_reuses_compilation_across_capacities(monkeypatch):
    compiled = set()

    def record_compile(**kwargs):
        if kwargs["fn"].name == "fold_slots_linked_kernel":
            compiled.add(kwargs["key"])

    monkeypatch.setattr(triton.knobs.runtime, "jit_post_compile_hook", record_compile)
    generator = torch.Generator(device="cuda").manual_seed(869)
    for slots in (1, 16, 17, 32):
        fragments = [[(i, i + 1) for i in range(rank, slots * 2, 2)] for rank in range(2)]
        routing = ContextParallelPlan.from_fragments((0, slots * 2), fragments, 1).routing("cuda")
        maps = torch.randn(slots * 2, 3, 48, 16, device="cuda", generator=generator) * 0.02
        actual = fold_slots_linked(maps, routing.predecessors, routing.predecessor_links)
        assert torch.equal(actual, fold_slots_fused(maps, routing.predecessors))
    assert len(compiled) == 1


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
