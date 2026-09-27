"""v1.30 plan execution and changing-shape reuse through the KDA adapter."""

import pytest
import torch

from attn_gym.testing.kda import (
    assert_matches_low_precision_reference,
    cumulative_sequence_offsets,
    kda_reference,
    make_kda_test_inputs,
)

pytest.importorskip("cutlass.experimental")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="cuDNN KDA requires SM100 or SM103",
)


@pytest.mark.parametrize("scheme", ["uncut", "prep", "chain", "warmup"])
def test_v130_plans_match_reference(monkeypatch, scheme):
    """Exercise every launch host with real state/gradients, not just the default selector."""
    from attn_gym.linear._delta_rule.cudnn_fe import kda as driver

    inputs = make_kda_test_inputs(128, heads=2, seed=827, normalize_qk=True, dtype=torch.bfloat16)
    packed = tuple(t[0] for t in inputs)
    offsets = cumulative_sequence_offsets([64, 64])
    state = torch.randn(2, 2, 128, 128, device="cuda") * 0.01
    dstate = torch.randn_like(state) * 0.01
    do = torch.randn_like(inputs[2])
    num_sm = torch.cuda.get_device_properties().multi_processor_count
    pieces = 2 if scheme == "chain" else 0
    tiles = 2 if scheme == "prep" else 1
    monkeypatch.setattr(
        driver.ForwardPlan, "build", lambda *args: driver.ForwardPlan(pieces, 2, tiles, num_sm)
    )
    monkeypatch.setattr(
        driver.BackwardPlan, "build", lambda *args: driver.BackwardPlan(pieces, 2, num_sm)
    )
    split = scheme == "warmup"
    state = None if split else state
    dstate = None if split else dstate
    out, final = driver.kda_forward(
        *packed,
        offsets,
        scale=128**-0.5,
        initial_state=state,
        output_final_state=not split,
        split=split,
    )
    grads = driver.kda_backward(
        *packed,
        do[0],
        offsets,
        scale=128**-0.5,
        initial_state=state,
        d_final_state=dstate,
        split=split,
    )
    references = []
    for precision in (torch.float64, torch.float32):
        leaves = tuple(t.detach().to(precision).requires_grad_() for t in inputs)
        initial = None if state is None else state.to(precision).requires_grad_()
        result = kda_reference(
            *leaves,
            initial,
            cu_seqlens=offsets,
            output_final_state=not split,
        )
        targets = leaves if initial is None else (*leaves, initial)
        outputs = (result[0],) if split else result
        cotangents = (do.to(precision),) if split else (do.to(precision), dstate.to(precision))
        gradients = torch.autograd.grad(outputs, targets, cotangents)
        references.append((*outputs, *gradients))
    actual = (out.unsqueeze(0),) if split else (out.unsqueeze(0), final)
    actual += tuple(g.unsqueeze(0) for g in grads[:5])
    if not split:
        actual += (grads[5],)
    for index, (got, high, low) in enumerate(zip(actual, *references, strict=True)):
        assert_matches_low_precision_reference(got, high, low, f"{scheme} result {index}")


def test_v130_changing_shapes_reuses_only_static_configuration():
    """A prior launch must not freeze the next call's sequence/head geometry or scratch size."""
    from attn_gym.linear import chunk_kda

    # A head count/sequence-count change shares some kernel configs but changes work-table sizes.
    for lengths, heads in (([65, 0, 63], 2), ([17, 31, 48, 0], 3), ([65, 0, 63], 2)):
        inputs = make_kda_test_inputs(
            sum(lengths),
            heads=heads,
            seed=831,
            gate_scale=0.69,
            normalize_qk=True,
            dtype=torch.float16,
            requires_grad=True,
        )
        offsets = cumulative_sequence_offsets(lengths)
        actual, _ = chunk_kda(*inputs, cu_seqlens=offsets, kernel_options={"backend": "cudnn"})
        high, _ = kda_reference(
            *(t.double() for t in inputs), cu_seqlens=offsets, output_final_state=False
        )
        low, _ = kda_reference(
            *(t.float() for t in inputs), cu_seqlens=offsets, output_final_state=False
        )
        assert_matches_low_precision_reference(
            actual, high, low, "changing-shape forward", source_dtype=torch.float16
        )
        grads = torch.autograd.grad(actual, inputs, torch.randn_like(actual))
        assert all(torch.isfinite(g).all() for g in grads)
