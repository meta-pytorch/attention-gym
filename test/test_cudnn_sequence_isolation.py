"""cuDNN GDN/KDA sequence isolation: no packed sequence reads another sequence's values.

Upstream's per-sequence TMA descriptors clip every load at the sequence's exact length, so a
sequence's outputs, gradients and states depend on its own tokens and state only. Poisoning one
sequence (or the capacity slack past the last one) with NaN or Inf must leave every other
sequence's results finite and bit-identical: a kernel that loads another sequence's rows and
masks them by multiplying with zero turns ``NaN * 0`` into NaN in the victim's neighbours.
"""

from __future__ import annotations

import itertools

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cutlass.cute", reason="the cuDNN backend requires CuTeDSL")

from attn_gym.linear import chunk_gdn, chunk_kda
from attn_gym.testing.kda import cumulative_sequence_offsets

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the cuDNN GDN/KDA kernels require SM100 or SM103",
)

# A sequence shorter than one chunk box at the tensor start (5), an empty sequence, sequences
# ending mid-box and on a box edge, then capacity slack past the last sequence.
LENGTHS = (5, 0, 300, 70, 64, 129)
# The split scheme adds a long decaying sequence, which the planner cuts into warmup items.
SPLIT_LENGTHS = (*LENGTHS, 4096)
SLACK = 7
HEADS = 2
DIM = 128
TOKEN_INPUTS = ("q", "k", "v", "gate", "beta", "do")
POISONS = (float("nan"), float("inf"))
# Plans per scheme: (forward pieces, forward value tiles, KDA prep, backward pieces, split).
SCHEMES = {
    "uncut": (0, 1, False, 0, False),
    "dv": (0, 2, False, 0, False),
    "prep": (0, 2, True, 0, False),
    "chain": (2, 1, False, 2, False),
    "split": (0, 1, False, 0, True),
}
CASES = [
    pytest.param(family, scheme, stateful, id=f"{family}-{scheme}-{state_id}")
    for family in ("gdn", "kda")
    for scheme in SCHEMES
    for stateful, state_id in ((False, "stateless"), (True, "state"))
    if not (scheme == "prep" and family == "gdn") and not (scheme == "split" and stateful)
]


def _pin_plans(monkeypatch, family: str, scheme: str) -> bool:
    """Force the scheme's forward/backward plans; returns whether it is the split schedule."""
    from attn_gym.linear._delta_rule.frost.drivers import gdn, kda

    driver = gdn if family == "gdn" else kda
    fwd_pieces, tiles, prep, bwd_pieces, split = SCHEMES[scheme]
    num_sm = torch.cuda.get_device_properties().multi_processor_count
    monkeypatch.setattr(
        driver.ForwardPlan, "build", lambda *args: driver.ForwardPlan(fwd_pieces, tiles, num_sm)
    )
    monkeypatch.setattr(
        driver.BackwardPlan, "build", lambda *args: driver.BackwardPlan(bwd_pieces, num_sm)
    )
    if family == "kda":
        monkeypatch.setattr(driver, "PREP_TILE_FRACTION", 1.0 if prep else 0.0)
    return split


def _assert_split_cuts(family: str, lengths, gate: torch.Tensor) -> None:
    """The split planner cuts these inputs: some row starts writing after chunk 0, so the split
    cases cover warmup items and piece boundaries, not only uncut items."""
    from attn_gym.linear._delta_rule.frost import work_tables as wt

    b_t = 64 if family == "gdn" else 16
    gate, cu = gate[0], cumulative_sequence_offsets(lengths)
    tokens, heads = gate.shape[:2]
    sms = torch.cuda.get_device_properties().multi_processor_count
    ideal = wt.compute_ideal_chunks(tokens, heads, sms, b_t)
    rows = wt.max_work_items(tokens, len(lengths), heads, ideal, b_t, sms)
    staging = torch.zeros(rows, wt.WORK_ITEM_FIELDS, dtype=torch.int32, device="cuda")
    scratch = torch.zeros(wt.chunk_scratch_rows(tokens, len(lengths), b_t), heads, device="cuda")
    count = torch.zeros(1, dtype=torch.int32, device="cuda")
    wt.build_split_table(
        gate, cu, scratch, staging, count,
        n_tiles=len(lengths) * heads, ideal_chunks=ideal, num_sms=sms, b_t=b_t,
    )  # fmt: skip
    write_starts = staging[: count.item(), 2]
    assert (write_starts > 0).any(), "the split case produced no cut"


def _inputs(family: str, stateful: bool, lengths) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(2026)
    tokens = sum(lengths) + SLACK

    def randn(*shape):
        return torch.randn(*shape, device="cuda", generator=generator)

    shape = (1, tokens, HEADS, DIM)
    gate_shape = shape[:3] if family == "gdn" else shape
    inputs = {
        "q": F.normalize(randn(*shape), dim=-1).bfloat16(),
        "k": F.normalize(randn(*shape), dim=-1).bfloat16(),
        "v": (randn(*shape) * 0.5).bfloat16(),
        "gate": -torch.rand(gate_shape, device="cuda", generator=generator) * 0.2,
        "beta": torch.rand(shape[:3], device="cuda", generator=generator),
        "do": randn(*shape).bfloat16(),
    }
    if stateful:
        state_shape = (len(lengths), HEADS, DIM, DIM)
        inputs["state"] = randn(*state_shape) * 0.02
        inputs["dfinal"] = randn(*state_shape) * 0.02
    return inputs


def _run(family: str, split: bool, lengths, inputs: dict[str, torch.Tensor]):
    """Forward and backward through the public API; returns outputs and gradients."""
    chunk = chunk_gdn if family == "gdn" else chunk_kda
    names = ("q", "k", "v", "gate", "beta", "state")
    leaves = {name: inputs[name].clone().requires_grad_() for name in names if name in inputs}
    stateful = "state" in leaves
    out, final = chunk(
        *(leaves[name] for name in names[:5]),
        leaves.get("state"),
        cu_seqlens=cumulative_sequence_offsets(lengths),
        output_final_state=stateful,
        kernel_options={"backend": "cudnn", "split_forward": split, "split_backward": split},
    )
    outputs, cotangents = [out], [inputs["do"]]
    if stateful:
        outputs.append(final)
        cotangents.append(inputs["dfinal"])
    grads = torch.autograd.grad(outputs, list(leaves.values()), cotangents)
    results = {"out": out, **{f"d{name}": grad for name, grad in zip(leaves, grads)}}
    if stateful:
        results["final_state"] = final
    return {name: tensor.detach() for name, tensor in results.items()}


def _sequence_rows(name: str, results, lengths, victim: int | None = None):
    """The result rows of every sequence but ``victim``; slack rows are unspecified."""
    tensor = results[name]
    if name in ("final_state", "dstate"):
        keep = [b for b in range(len(lengths)) if b != victim]
        return tensor[keep]
    bounds = [0, *itertools.accumulate(lengths)]
    keep = [
        torch.arange(bounds[b], bounds[b + 1], device=tensor.device)
        for b in range(len(lengths))
        if b != victim
    ]
    return tensor[:, torch.cat(keep)]


@pytest.mark.parametrize(("family", "scheme", "stateful"), CASES)
def test_poisoned_sequence_does_not_reach_other_sequences(monkeypatch, family, scheme, stateful):
    split = _pin_plans(monkeypatch, family, scheme)
    lengths = SPLIT_LENGTHS if split else LENGTHS
    inputs = _inputs(family, stateful, lengths)
    if split:
        _assert_split_cuts(family, lengths, inputs["gate"])
    clean = _run(family, split, lengths, inputs)
    assert all(_sequence_rows(name, clean, lengths).isfinite().all() for name in clean)
    bounds = [0, *itertools.accumulate(lengths), sum(lengths) + SLACK]
    leaks = []
    for victim in range(len(lengths) + 1):
        targets = [name for name in TOKEN_INPUTS if bounds[victim + 1] > bounds[victim]]
        if stateful and victim < len(lengths):
            targets += ["state", "dfinal"]
        for name, poison in itertools.product(targets, POISONS):
            poisoned = dict(inputs)
            poisoned[name] = inputs[name].clone()
            if name in ("state", "dfinal"):
                poisoned[name][victim] = poison
            else:
                poisoned[name][:, bounds[victim] : bounds[victim + 1]] = poison
            results = _run(family, split, lengths, poisoned)
            for result in clean:
                expected = _sequence_rows(result, clean, lengths, victim)
                if not torch.equal(_sequence_rows(result, results, lengths, victim), expected):
                    leaks.append(f"{name}={poison} in sequence {victim} -> {result}")
    assert not leaks, "\n".join(leaks)
