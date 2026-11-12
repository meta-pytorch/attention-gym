"""Qwen3-MoE decode megakernel: routing, expert-gathered MLP and folded routing weights."""

import dataclasses
import gc

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cutlass.cute")

from attn_gym.warpdrive.examples import qwen_megakernel as mk
from attn_gym.warpdrive.examples import qwen_megakernel_ref as ref_module
from attn_gym.warpdrive.examples.qwen_megakernel_ref import (
    QwenConfig,
    compare_logits,
    decode_step_ref,
    forward_tokens,
    load_qwen3_weights,
    make_decode_state,
    random_weights,
    rms_norm,
)

MAX_LEN = 256


@pytest.fixture(autouse=True)
def release_cuda_memory():
    """Return each test's runners (packed weight copies, up to 64 GB for Qwen3-32B) to the
    device before the next test, which may load another checkpoint on this worker."""
    yield
    gc.collect()
    torch.cuda.empty_cache()


def require_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the megakernel requires an SM100-family GPU")


# A small Qwen3-MoE with Qwen3-30B-A3B's heads, expert count and top-k.
SMALL_MOE = QwenConfig(
    hidden_size=1024, num_layers=2, num_q_heads=32, num_kv_heads=4, intermediate_size=1024,
    vocab_size=4096, num_experts=128, experts_per_token=8, moe_intermediate_size=256,
)  # fmt: skip


def check_moe_step(monkeypatch, runner, weights, caches, token, pos):
    """One megakernel step against the reference forced onto the kernel's experts: logits within
    the bf16 error budget. Unforced, a layer may route differently only at a near-tie: the
    reference's margin between ranks k and k + 1 is within the two router logits' difference."""
    logits = runner(weights, token, pos, *[c.clone() for c in caches])
    experts = runner.routed_experts()
    ref = decode_step_ref(weights, token, pos, *[c.clone() for c in caches], experts=experts)
    high = decode_step_ref(
        weights, token, pos, *[c.clone() for c in caches], dtype=torch.float32, experts=experts
    )
    metrics = compare_logits(logits, ref, high)
    assert metrics["passed"], metrics
    ours = runner.values("router_logits")
    recorded = []
    original = ref_module.mlp

    def spy(config, w, x, forced=None):
        recorded.append(F.linear(x, w["router"]).float()[0])
        return original(config, w, x, forced)

    with monkeypatch.context() as patch:
        patch.setattr(ref_module, "mlp", spy)
        decode_step_ref(weights, token, pos, *[c.clone() for c in caches])
    k = runner.op.top_k
    for layer, theirs in enumerate(recorded):
        top = theirs.sort(descending=True, stable=True)
        if set(top.indices[:k].tolist()) != set(experts[layer, 0].tolist()):
            noise = (ours[layer] - theirs).abs().max()
            assert top.values[k - 1] - top.values[k] <= noise, layer


@pytest.mark.parametrize("engine", ["cuda_core", "tcgen05"])
def test_moe_random_weights(monkeypatch, engine):
    """Routing, expert-gathered gate/up and down, and the folded routing weights on a small MoE.
    No checkpoint needed."""
    require_sm100()
    weights = random_weights(SMALL_MOE)
    runner = mk.QwenMegakernel(weights, MAX_LEN, engine=engine)
    for ctx in (100, 200):
        state = make_decode_state(weights, ctx, MAX_LEN, seed=ctx)
        check_moe_step(monkeypatch, runner, weights, *state)


def test_moe_router_ties_go_to_the_lower_expert():
    """With every router row equal, all logits tie: the kernel and the reference both pick
    experts 0..k-1 with equal weights, so their logits agree to the bf16 budget unforced."""
    require_sm100()
    weights = random_weights(SMALL_MOE)
    weights.router.copy_(weights.router[:, :1].expand_as(weights.router))
    runner = mk.QwenMegakernel(weights, MAX_LEN)
    caches, token, pos = make_decode_state(weights, 120, MAX_LEN, seed=5)
    ours, theirs, exact = ([c.clone() for c in caches] for _ in range(3))
    logits = runner(weights, token, pos, *ours)
    expected = torch.arange(SMALL_MOE.experts_per_token, device="cuda")
    assert (runner.routed_experts() == expected).all()
    ref = decode_step_ref(weights, token, pos, *theirs)
    high = forward_tokens(weights, token.reshape(1), pos, *exact, dtype=torch.float32)
    assert compare_logits(logits, ref, high)["passed"]


@pytest.mark.xdist_group("qwen-large-checkpoints")
@pytest.mark.parametrize("engine", ["cuda_core", "tcgen05"])
def test_moe_checkpoint_decode_steps(monkeypatch, engine):
    """Qwen3-30B-A3B at short and long contexts (see check_moe_step). Skips unless the
    checkpoint is in the local HF cache; never downloads."""
    require_sm100()
    hub = pytest.importorskip("huggingface_hub")
    path = hub.try_to_load_from_cache("Qwen/Qwen3-30B-A3B", "config.json")
    if not isinstance(path, str):
        pytest.skip("Qwen3-30B-A3B is not in the local HF cache")
    # This worker may still cache the dense checkpoints' freed blocks (fragmented): release them.
    gc.collect()
    torch.cuda.empty_cache()
    weights = load_qwen3_weights(path)
    for ctx in (127, 1500):
        max_len = (ctx // mk.BLOCK + 1) * mk.BLOCK
        runner = mk.QwenMegakernel(weights, max_len, engine=engine)
        state = make_decode_state(weights, ctx, max_len, seed=ctx)
        check_moe_step(monkeypatch, runner, weights, *state)
        del runner


@pytest.mark.parametrize("engine", ["cuda_core", "tcgen05"])
@pytest.mark.parametrize("normalized", [True, False])
def test_moe_routing_in_the_kernel(normalized, engine):
    """Expert rows alone, from crafted router logits: the published activation of slot r is
    expert e_r's SiLU(gate) * up times its routing weight, so it shows both which rows the load
    warp fetched and the weight the compute warp applied. Layer 0 mixes signs and ties two
    negative logits at rank k (the lower expert wins); layer 1's logits are all equal. Without
    norm_topk_prob the weights are the plain softmax probabilities (summing well below one)."""
    require_sm100()
    config = dataclasses.replace(SMALL_MOE, norm_topk_prob=normalized)
    weights = random_weights(config)
    runner = mk.QwenMegakernel(weights, MAX_LEN, ops=(mk.GATEUP,), engine=engine)
    k, experts = config.experts_per_token, config.num_experts
    torch.manual_seed(0)
    logits = torch.zeros(config.num_layers, experts, device="cuda")
    # Layer 0: ranks 0..k-2 span both signs, experts 90 and 40 tie at -1 for rank k-1, and the
    # rest sit below -1 in shuffled order.
    logits[0] = torch.linspace(-3, -1.25, experts, device="cuda")[torch.randperm(experts)]
    top = [5, 12, 33, 47, 64, 81, 100]
    logits[0, top] = torch.tensor([4.0, 3.0, 1.0, 0.5, -0.25, -0.5, -0.75], device="cuda")
    logits[0, [90, 40]] = -1.0
    logits = logits.bfloat16().float()  # exact in bf16
    h_mid = torch.randn(config.num_layers, config.hidden_size, device="cuda").bfloat16().float()
    runner.set_values("router_logits", logits)
    runner.set_values("h_mid", h_mid)
    runner.launch(*[
        torch.zeros(config.num_layers, config.num_kv_heads, MAX_LEN, config.head_dim,
                    device="cuda", dtype=torch.bfloat16)
        for _ in "kv"
    ])  # fmt: skip
    torch.cuda.synchronize()
    act = runner.values("act").view(config.num_layers, k, config.moe_intermediate_size)
    for layer in range(config.num_layers):
        probs = F.softmax(logits[layer], dim=-1)
        ranked = probs.sort(descending=True, stable=True)
        chosen, weight = ranked.indices[:k], ranked.values[:k]
        if normalized:
            weight = weight / weight.sum()
        x = rms_norm(h_mid[layer].bfloat16(), weights.post_ln[layer], config.rms_eps)
        for r, e in enumerate(chosen.tolist()):
            gate = F.linear(x, weights.gate[layer, e]).float()
            up = F.linear(x, weights.up[layer, e]).float()
            expected = (F.silu(gate.bfloat16()) * up.bfloat16()).float() * weight[r].bfloat16()
            torch.testing.assert_close(act[layer, r], expected, atol=1e-2, rtol=2e-2)
    reference = logits.sort(dim=-1, descending=True, stable=True).indices[:, :k]
    assert reference[0].tolist() == [*top, 40]  # the lower expert of the tie
    assert reference[1].tolist() == list(range(k))  # all equal: the k lowest experts


def test_moe_phase_spans_count_expert_rows_from_the_router():
    """A MoE table's GATEUP phase runs from the router's last row, not o_proj's."""
    from attn_gym.warpdrive.examples import qwen_megakernel_trace as mk_trace
    from attn_gym.warpdrive.profiler import Range

    rows = [[mk.OPROJ, 0, 0, 0], [mk.ROUTER, 0, 0, 0], [mk.GATEUP, 0, 0, 0]]
    table = torch.tensor([words + [i] for i, words in enumerate(rows)])
    events = [Range("compute", d - 10, d, (0,), payload=i) for i, d in enumerate([100, 150, 230])]
    assert mk_trace.phase_spans(events, table) == {"ROUTER": 0.05, "GATEUP": 0.08}
