"""warpdrive GDN/KDA plan choice: the exact piece chain runs when idle SMs make it pay.

One persistent work item per (sequence, head) leaves most SMs idle on few tiles; the planner
must cut such sequences into chain pieces even when the pieces are short, and keep the
unchained plans where the chain's summaries and state chain cost more than they save. The
expectations follow NOTE [Chain plan cost] on a 152-SM GB300.
"""

from __future__ import annotations

import pytest

pytest.importorskip("cutlass.cute", reason="the warpdrive backend requires CuTeDSL")

from attn_gym.linear._delta_rule.warpdrive.drivers import gdn, kda, plan

NUM_SM = 152
DRIVERS = {"gdn": gdn, "kda": kda}


@pytest.fixture(autouse=True)
def gb300(monkeypatch):
    monkeypatch.setattr(plan, "_num_sm", lambda device: NUM_SM)


@pytest.mark.parametrize("family", DRIVERS)
@pytest.mark.parametrize(
    ("lengths", "heads", "pieces"),
    [
        pytest.param((2048,), 8, {"gdn": 8, "kda": 16}, id="1x2048-h8"),
        pytest.param((1024,), 4, {"gdn": 4, "kda": 16}, id="1x1024-h4"),
        pytest.param((4096,) * 2, 16, {"gdn": 4, "kda": 4}, id="2x4096-h16"),
        # Two pieces of 1024 tokens on a full grid cost more than they save.
        pytest.param((2048,) * 4, 16, {"gdn": 0, "kda": 0}, id="4x2048-h16"),
        # The chain needs at least two pieces; 128 tiles leave no room for them.
        pytest.param((8192,) * 2, 64, {"gdn": 0, "kda": 0}, id="2x8192-h64"),
    ],
)
def test_backward_chains_low_occupancy_shapes(family, lengths, heads, pieces):
    built = DRIVERS[family].BackwardPlan.build(sum(lengths), len(lengths), heads, 128, "cuda")
    assert built.pieces == pieces[family]


@pytest.mark.parametrize("family", DRIVERS)
@pytest.mark.parametrize(
    ("lengths", "heads", "pieces", "tiles_per_head"),
    [
        # Short pieces: the d_v split is as fast and has no chain overhead.
        pytest.param((2048,), 8, 0, 2, id="1x2048-h8"),
        pytest.param((8192,), 2, 16, 1, id="1x8192-h2"),
        pytest.param((131072,), 2, 16, 1, id="1x131072-h2"),
        pytest.param((4096,) * 8, 32, 0, 1, id="8x4096-h32"),
    ],
)
def test_forward_chains_long_pieces_on_few_tiles(family, lengths, heads, pieces, tiles_per_head):
    built = DRIVERS[family].ForwardPlan.build(sum(lengths), len(lengths), heads, 128, "cuda")
    assert (built.pieces, built.tiles_per_head) == (pieces, tiles_per_head)


@pytest.mark.parametrize(
    ("lengths", "heads", "forward", "backward"),
    [
        # Short pieces chain at D=64 (no d_v split, a quarter of the D=128 summary state).
        pytest.param((4096,), 2, 16, 16, id="1x4096-h2"),
        pytest.param((2048,), 16, 8, 8, id="1x2048-h16"),
        pytest.param((1024,), 16, 0, 4, id="1x1024-h16"),
        # Three forward pieces on a nearly full grid lose to the uncut table; the backward wins.
        pytest.param((2048,), 48, 0, 3, id="1x2048-h48"),
        # Two backward pieces pay off only on long pieces.
        pytest.param((512,) * 4, 16, 0, 0, id="4x512-h16"),
        pytest.param((16384,), 64, 0, 2, id="1x16384-h64"),
    ],
)
def test_gdn_d64_plans(lengths, heads, forward, backward):
    tokens, num_seqs = sum(lengths), len(lengths)
    built = gdn.ForwardPlan.build(tokens, num_seqs, heads, 64, "cuda")
    assert (built.pieces, built.tiles_per_head) == (forward, 1)
    assert gdn.BackwardPlan.build(tokens, num_seqs, heads, 64, "cuda").pieces == backward
