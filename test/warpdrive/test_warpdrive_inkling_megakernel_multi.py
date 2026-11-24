"""The multi-token Inkling 2 Mini visit launch (``inkling_megakernel.multi_kernel``, a runner
with ``tokens`` > 1) on quantized experts against the oracle over a block of consecutive tokens:
its attention half against ``ref.attention_block`` (``start=None``; causal inside the block, the
block's new keys crossing a 64-key block and the sliding window, convolutions reaching back into
the block and the rings), its MLP half against ``ref.mlp_block`` on the launch's own residual
(each token's routing, the union of their experts), with message gates, and stale cache rows it
must ignore; and the multi-token message and head launches. The chat's default block (4) and 8
are tested."""

import dataclasses
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass.cute")

from test_warpdrive_inkling_megakernel_decoder import HEAD_ATOL
from test_warpdrive_inkling_megakernel_visit import (
    CACHE_ULPS,
    CONFIG,
    MAX_MEDIAN_ULPS,
    MAX_ULPS,
    MLP_MAX_CHANGED,
    MLP_ULPS,
    RING_ULPS,
    activation_format,
    check_activations,
    clone,
    oracle_messages,
    random_cache,
    random_core_layer,
    random_layer,
    random_pass,
    scale_ulps,
    ulps,
    with_quantized_experts,
)

from attn_gym.warpdrive.examples.inkling_megakernel import ref
from attn_gym.warpdrive.examples.inkling_megakernel.fast import FAST
from attn_gym.warpdrive.examples.inkling_megakernel.megakernel import (
    InklingHeadMegakernel,
    InklingMsgMegakernel,
    InklingVisitMegakernel,
    MessageGates,
)
from attn_gym.warpdrive.examples.megakernel.formats import NVFP4

pytestmark = pytest.mark.usefixtures(
    "requires_sm100_or_sm121", "fp32_bf16_matmul_reductions", "release_cuda_memory"
)


def block_inputs(config, tokens: int, blocks: int, seed: int):
    """``(h, blocks)``: bf16 ``(tokens, hidden)`` visit input and ``blocks`` completed blocks."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    h = config.hidden_size
    states = [
        torch.randn(tokens, h, device="cuda", generator=gen).bfloat16() for _ in range(blocks + 1)
    ]
    return states[-1], states[:-1]


def check_block_visit(
    runner, config, w, layer, visit, h, blocks, cache, pos, messages=None, gates=None, *,
    launch_w, cache_ulps=CACHE_ULPS,
):  # fmt: skip
    """``test_warpdrive_inkling_megakernel_visit.check_visit`` for a block of tokens at
    positions ``pos ..``: the residual, every block row of the caches and rings within their
    budgets and every other row untouched; each token's experts, the union's size, and on the
    launch's own residual the activations of every routed row, then the MLP output and ring
    rows on the launch's activations. The runner launches ``launch_w`` (quantized experts),
    the oracle ``w`` (their block values). Returns the residual's and output's errors."""
    tokens = h.shape[0]
    positions = torch.arange(pos, pos + tokens, device="cuda")
    theirs = clone(cache)
    expected, mlp_blocks = ref.attention_block(
        config, w, layer, visit, h, blocks, messages, theirs, positions, None, FAST
    )
    out = runner(launch_w, layer, visit, h, blocks, cache, torch.tensor([pos], device="cuda"),
                 gates=gates)  # fmt: skip
    residual = runner.residual()
    boundary = layer % config.block_layers == 0
    attn_out = expected if boundary else expected - h.float()
    errors = ulps(residual - expected + attn_out, attn_out)
    rows = torch.zeros(cache.max_len, dtype=torch.bool, device="cuda")
    rows[pos : pos + tokens] = True
    for name in ("k", "v"):
        mine, ref_rows = getattr(cache, name)[visit], getattr(theirs, name)[visit]
        assert ulps(mine[:, rows], ref_rows[:, rows]).max() <= cache_ulps, name
        assert torch.equal(mine[:, ~rows], ref_rows[:, ~rows]), name
    slots = torch.zeros(ref.CONV_RING, dtype=torch.bool, device="cuda")
    slots[positions % ref.CONV_RING] = True
    for name in ("conv_k", "conv_v", "conv_attn"):
        mine, ref_ring = getattr(cache, name)[visit], getattr(theirs, name)[visit]
        assert torch.equal(mine[~slots], ref_ring[~slots]), name
        budget = MAX_ULPS if name == "conv_attn" else RING_ULPS
        assert ulps(mine[slots], ref_ring[slots]).max() <= budget, name
    assert errors.max() <= MAX_ULPS and errors.median() <= MAX_MEDIAN_ULPS, (
        errors.max(),
        errors.median(),
    )

    message = None if messages is None else messages[:, 3]
    x = ref.block_residual(w.mlp_query[None], mlp_blocks, residual)[:, 0]
    if message is not None:
        x = ref.message_gate(w.gates[3], x, message, config.rms_eps)
    x = ref.rms_norm(x, w.post_norm, config.rms_eps).to(torch.bfloat16)
    experts, weights = ref.route(config, w, x)
    assert torch.equal(runner.routed_experts(launch_w), experts.sort(dim=-1).values)
    assert int(runner.union_size().item()) == experts.unique().numel()
    routed, mine = check_activations(
        runner, config, w, layer, x, experts, weights, message, launch_w
    )

    def experts_on_launch_activations(config, w, x, experts, weights):
        return ref.declared_down(w, x, routed, mine)

    kernels = dataclasses.replace(ref.REFERENCE, experts=experts_on_launch_activations)
    expected = ref.mlp_block(
        config, w, visit, residual, mlp_blocks, message, theirs, positions, kernels
    )
    out_errors = scale_ulps(out, expected)
    changed = (out_errors > 0).float().mean()
    assert out_errors.max() <= MLP_ULPS and changed <= MLP_MAX_CHANGED, (out_errors.max(), changed)
    ring, ref_ring = cache.conv_mlp[visit], theirs.conv_mlp[visit]
    assert torch.equal(ring[~slots], ref_ring[~slots])
    assert scale_ulps(ring[slots], ref_ring[slots]).max() <= MLP_ULPS
    return errors, out_errors


# (tokens, layer, blocks, positions): a sliding NVFP4 layer at the chat's default block of 4
# tokens from position 0 (convolutions and keys only inside the block), across a 64-key block
# edge and past the sliding window; a full MXFP8 block-boundary layer at 8 tokens (two 16-row MMA
# tiles of query rows, o_proj split in quarters) reading 4 blocks, also past
# ``log_scaling_floor`` (each token's own log scale).
CASES = [
    (4, 1, 1, (0, 60, 1100)),
    (4, 23, 4, (130, 8990)),
    (8, 23, 4, (130, 8990)),
]


@pytest.mark.parametrize("tokens,layer,blocks,positions", CASES)
@torch.no_grad()
def test_block_visit_random_weights(tokens, layer, blocks, positions):
    if tokens == 8 and torch.cuda.get_device_capability() == (12, 1):
        pytest.skip("8 tokens' input rows exceed GB10 shared memory on full layers")
    w = random_layer(CONFIG, layer, seed=layer)
    quantized, dense = with_quantized_experts(w, activation_format(layer), seed=10)
    runner = InklingVisitMegakernel(CONFIG, (quantized,), tokens=tokens)
    cache = random_cache(CONFIG, visits=2, max_len=9024, seed=layer)
    for case, pos in enumerate(positions):
        h, states = block_inputs(CONFIG, tokens, blocks, seed=100 * layer + case)
        check_block_visit(
            runner, CONFIG, dense, layer, 1, h, states, cache, pos, launch_w=quantized
        )


# A gate's h rounding flip moves its whole gated stream by ~1e-3 relative, which reaches the k/v
# cache rows: 3 ulps measured on a block's rows.
MESSAGE_CACHE_ULPS = 4.0


@torch.no_grad()
def test_block_message_visit_random_weights():
    """A full core visit after the first pass over a block of 4: the multi-token message launch
    against the oracle's messages, then the visit with its q, k, v and MLP gates."""
    config, tokens, layer, pos = CONFIG, 4, 11, 60
    w = random_core_layer(config, layer, seed=layer)
    quantized, dense = with_quantized_experts(w, NVFP4, seed=20)
    queries, _, _ = random_pass(config, 1, seed=layer)
    gen = torch.Generator(device="cuda").manual_seed(layer)
    block, partial = ((torch.randn(tokens, config.hidden_size, device="cuda", generator=gen) * 3)
                      .bfloat16() for _ in range(2))  # fmt: skip
    gates = MessageGates((quantized,), queries, tokens=tokens)
    InklingMsgMegakernel(config, gates)(block, partial)
    messages = oracle_messages(queries, block, partial)[0]  # (tokens, 4, hidden)
    torch.testing.assert_close(
        gates.messages.view(4, tokens, -1).transpose(0, 1), messages, atol=1e-5, rtol=1e-5
    )
    runner = InklingVisitMegakernel(config, (quantized,), tokens=tokens)
    cache = random_cache(config, visits=2, max_len=1216, seed=layer)
    h, states = block_inputs(config, tokens, 2, seed=layer)
    check_block_visit(
        runner, config, dense, layer, 1, h, states, cache, pos, messages, gates.layer(0),
        launch_w=quantized, cache_ulps=MESSAGE_CACHE_ULPS,
    )  # fmt: skip


@torch.no_grad()
def test_block_visit_ignores_stale_cache_rows():
    """Cache rows from the block's position on hold no keys yet, and in sliding layers rows
    before the first token's window are out of every token's (zero probabilities): NaNs there
    give the outputs, residual and stored rows of zeros there, bit for bit. At 1100 the window's
    first key falls inside a KV block."""
    tokens, layer, pos = 4, 1, 1100
    w, _ = with_quantized_experts(random_layer(CONFIG, layer, seed=layer), NVFP4, seed=10)
    runner = InklingVisitMegakernel(CONFIG, (w,), tokens=tokens)
    cache = random_cache(CONFIG, visits=1, max_len=1216, seed=layer)
    h, states = block_inputs(CONFIG, tokens, 1, seed=layer)
    lo = pos - (CONFIG.sliding_window - 1)
    results = []
    for fill in (0.0, float("nan")):
        mine = clone(cache)
        for rows in (slice(None, lo), slice(pos, None)):
            mine.k[:, :, rows] = fill
            mine.v[:, :, rows] = fill
        out = runner(w, layer, 0, h, states, mine, torch.tensor([pos], device="cuda")).clone()
        results.append((out, runner.residual().clone(), mine.k[:, :, lo : pos + tokens],
                        mine.v[:, :, lo : pos + tokens]))  # fmt: skip
    for zero, nan in zip(*results):
        assert torch.equal(zero, nan)


@torch.no_grad()
def test_block_head_random_weights():
    """The multi-token head launch at 4 tokens: each token's logits within a bf16 rounding of
    the one-token head's on that token's states and of the oracle's."""
    config, h, tokens = CONFIG, CONFIG.hidden_size, 4
    gen = torch.Generator(device="cuda").manual_seed(0)

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, device="cuda", generator=gen) * scale

    weights = SimpleNamespace(
        config=config,
        lm_head=randn(config.vocab_size, h, scale=20 * h**-0.5).bfloat16(),
        final_query=randn(h),
        final_norm=(1 + randn(h, scale=0.1)).bfloat16(),
    )
    runner = InklingHeadMegakernel(weights, tokens=tokens)
    single = InklingHeadMegakernel(weights)
    states = [(randn(tokens, h) * 2).bfloat16() for _ in range(4)]
    logits = runner(states[:-1], states[-1]).clone()
    for t in range(tokens):  # fp32 summation order (tensor cores) moves a bf16 rounding
        expected = single([s[t] for s in states[:-1]], states[-1][t])
        assert (logits[t] - expected).abs().max() <= HEAD_ATOL, t
    x = ref.block_residual(weights.final_query[None], states[:-1], states[-1])[:, 0]
    x = ref.rms_norm(x, weights.final_norm, config.rms_eps).to(torch.bfloat16)
    assert (logits - ref.logits_of(weights, x)).abs().max() <= HEAD_ATOL
