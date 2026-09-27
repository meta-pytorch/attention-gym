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
    """A prior launch must not freeze the next call's sequence/head geometry or scratch size,
    and repeating a shape must relaunch the compiled hosts instead of compiling again."""
    from attn_gym.linear import chunk_kda
    from attn_gym.linear._delta_rule.cudnn_fe.kernel import (
        kda_warmup_backward_f16,
        kda_warmup_forward_f16,
    )

    compile_fns = (
        kda_warmup_forward_f16._compile_warmup_forward,
        kda_warmup_backward_f16._compile_warmup_backward,
    )
    sizes = []
    # A head count/sequence-count change shares some kernel configs but changes work-table sizes.
    for lengths, heads in (([65, 0, 63], 2), ([17, 31, 48, 0], 3), ([65, 0, 63], 2)):
        before = [fn.cache_info() for fn in compile_fns]
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
        after = [fn.cache_info() for fn in compile_fns]
        assert [b.hits + b.misses + 1 for b in before] == [a.hits + a.misses for a in after]
        sizes.append([info.currsize for info in after])
    # The third shape repeats the first, so it adds no forward or backward specialization.
    assert sizes[2] == sizes[1]


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16), ids=("bf16", "fp16"))
@pytest.mark.parametrize("with_state", (False, True), ids=("no-state", "state"))
@pytest.mark.parametrize("scheme", ["uncut", "dv", "prep", "chain"])
def test_v130_forward_plans_keep_delta_residual_in_fp32(monkeypatch, scheme, with_state, dtype):
    """Every forward host forms ``beta * (v - state @ k)`` in FP32 before the b16 MMA pack.

    Tokens 0 and 1 write ``state[v0, k0] = 4096`` and ``state[v0, k1] = 2``; token 16 reads
    ``state @ k = 4098`` against ``v = 4096`` with ``beta = 0.25``. Rounding the contraction
    (or the residual) to the I/O dtype first loses the ``-2`` and gives 2.0 at tokens 16 and
    32 instead of 1.5. The chain's second piece re-enters through the summary kernel.
    """
    from attn_gym.linear._delta_rule.cudnn_fe import kda as driver

    tokens = 64
    q = torch.zeros(tokens, 1, 128, device="cuda", dtype=dtype)
    k, v = torch.zeros_like(q), torch.zeros_like(q)
    gate = torch.zeros(tokens, 1, 128, device="cuda")
    beta = torch.zeros(tokens, 1, device="cuda")
    k[0, 0, 0] = k[1, 0, 1] = 1
    v[0, 0, 0], v[1, 0, 0] = 4096, 2
    beta[0, 0] = beta[1, 0] = 1
    q[16, 0, 1] = q[32, 0, 1] = 1
    k[16, 0, :2] = 1
    v[16, 0, 0] = 4096
    beta[16, 0] = 0.25
    num_sm = torch.cuda.get_device_properties().multi_processor_count
    pieces = 2 if scheme == "chain" else 0
    tiles = 2 if scheme in ("dv", "prep") else 1
    monkeypatch.setattr(
        driver.ForwardPlan, "build", lambda *args: driver.ForwardPlan(pieces, 1, tiles, num_sm)
    )
    monkeypatch.setattr(driver, "PREP_TILE_FRACTION", 1.0 if scheme == "prep" else 0.0)
    state = torch.zeros(1, 1, 128, 128, device="cuda") if with_state else None

    out, _ = driver.kda_forward(
        q, k, v, gate, beta, cumulative_sequence_offsets([tokens]), scale=1.0, initial_state=state
    )

    assert (out[16, 0, 0].item(), out[32, 0, 0].item()) == (1.5, 1.5)


@pytest.mark.parametrize("with_exit", [False, True], ids=["no-exit", "exit"])
def test_cudnn_compaction_keeps_empty_state_cotangents(monkeypatch, with_exit):
    """Compaction drops empty intervals from the work table (the count stays at 4 items) while
    their states pass through the forward and their exit cotangents through the backward."""
    from attn_gym.linear._delta_rule.cudnn_fe import kda as driver

    inputs = tuple(t[0] for t in make_kda_test_inputs(32, heads=2, seed=433, normalize_qk=True))
    state = torch.randn(4, 2, 128, 128, device="cuda") / 100
    dstate = torch.randn_like(state) if with_exit else None
    do = torch.randn_like(inputs[2])
    keep = torch.tensor([True, False, True, False], device="cuda")
    counts = []
    sms = torch.cuda.get_device_properties().multi_processor_count
    monkeypatch.setattr(
        driver.ForwardPlan, "build", lambda *args: driver.ForwardPlan(0, 1, 1, sms)
    )
    monkeypatch.setattr(driver.BackwardPlan, "build", lambda *args: driver.BackwardPlan(0, 1, sms))

    def work_count(device):
        count = torch.empty(1, device=device, dtype=torch.int32)
        counts.append(count)
        return count

    monkeypatch.setattr(driver, "_work_count", work_count)
    results = []
    for padded in (False, True):
        cu = cumulative_sequence_offsets([16, 0, 16, 0] if padded else [16, 16])
        seed = state if padded else state[keep].contiguous()
        exit_grad = dstate if padded or dstate is None else dstate[keep].contiguous()
        out, final = driver.kda_forward(
            *inputs, cu, scale=1.0, initial_state=seed, output_final_state=True
        )
        grads = driver.kda_backward(
            *inputs, do, cu, scale=1.0, initial_state=seed, d_final_state=exit_grad
        )
        results.append((out, final, grads))
    compact, padded = results
    torch.testing.assert_close(padded[0], compact[0], rtol=0, atol=0)
    torch.testing.assert_close(padded[1][keep], compact[1], rtol=0, atol=0)
    torch.testing.assert_close(padded[1][~keep], state[~keep], rtol=0, atol=0)
    for got, want in zip(padded[2][:5], compact[2][:5], strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    torch.testing.assert_close(padded[2][5][keep], compact[2][5], rtol=0, atol=0)
    expected = dstate[~keep] if with_exit else torch.zeros_like(state[~keep])
    torch.testing.assert_close(padded[2][5][~keep], expected, rtol=0, atol=0)
    assert [count.item() for count in counts] == [4, 4, 4, 4]
