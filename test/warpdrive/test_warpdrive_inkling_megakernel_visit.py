"""The Inkling 2 Mini visit launch (``inkling_megakernel.megakernel.InklingVisitMegakernel``) on
random weights at the real shapes, streaming quantized experts as the checkpoint stores them
(``quant.QuantExperts``): its attention half against ``ref.attention_block`` with ``start=None``
(for sliding and full layers; the cache and ring rows it writes), its MLP half against
``ref.mlp_block`` on the launch's own post-attention residual and the experts' bf16 block values,
a core visit after the first pass through its message gates, and a CUDA graph replay with compile
reuse across positions. The helpers here are shared with the decoder and multi-token tests."""

import dataclasses

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cutlass.cute")

from attn_gym.warpdrive.examples.inkling_megakernel import quant, ref
from attn_gym.warpdrive.examples.inkling_megakernel.act_quant import ActivationQuant
from attn_gym.warpdrive.examples.inkling_megakernel.common import GATE_SPLITS, GATES, MLP_GATE
from attn_gym.warpdrive.examples.inkling_megakernel.fast import FAST
from attn_gym.warpdrive.examples.inkling_megakernel.megakernel import (
    InklingMsgMegakernel,
    InklingVisitMegakernel,
    MessageGates,
)
from attn_gym.warpdrive.examples.inkling_megakernel.visit_kernel import compile_visit_kernel
from attn_gym.warpdrive.examples.megakernel.formats import MXFP8, NVFP4

pytestmark = pytest.mark.usefixtures(
    "requires_sm100_or_sm121", "fp32_bf16_matmul_reductions", "release_cuda_memory"
)

CONFIG = ref.InklingConfig()
# The residual differs from the oracle mainly through the flash merge: the oracle rounds the
# softmax probabilities to bf16 before the value sum, the kernel sums unrounded ones. The
# attention output then differs by an ulp in many elements, which o_proj spreads into bf16
# rounding flips of about half the outputs on the checkpoint (max 7 ulps, median <= 0.25, in
# ``ulps`` of the post-attention output; against an oracle with fp32 probabilities: max 2).
MAX_ULPS, MAX_MEDIAN_ULPS = 8.0, 0.5
# The k/v ring rows hold raw projections, whose fp32 order can flip a bf16 rounding; the
# convolution, clip and norm can carry such a flip of a large element into a few ulps of a
# smaller cache element (3 at the checkpoint's visit 8: a raw v element 0.54 moved its
# normalized cache element 0.9 by 3 ulps).
RING_ULPS, CACHE_ULPS = 1.0, 3.0
# The MLP half is checked in two parts on the launch's own residual. Its activations (the
# gate/up rows' bf16 SwiGLU times the routing weight) differ from the oracle's only through fp32
# summation order and the kernel's transcendental functions, which can flip a bf16 rounding of
# a gate/up output or an activation: in ``ulps`` of the expected activations, at most 2 in at
# most 0.6% of the elements on random weights and the checkpoint's visits. The rest (each
# activation row quantized with its own amax, the down rows, their fp32 sum, the post-MLP
# convolution and residual) is checked against the oracle on the launch's own activations: a
# flipped activation that is its row's amax rescales every down output of its slot (all of a
# visit's outputs moved, up to 8 ulps, in a message visit checked end to end). Measured at most
# 0.5 ulps of a bf16 value at the output's RMS, in at most 0.05% of the outputs.
MLP_ULPS, MLP_MAX_CHANGED = 2.0, 0.01
ACT_ULPS, ACT_MAX_CHANGED = 3.0, 0.02


def random_attention(config: ref.InklingConfig, layer: int, gen) -> dict:
    """Attention weights of ``layer`` (full or sliding, with or without a block-residual query)
    at ``config``'s shapes, scaled so q clips sometimes, scores and biases spread over a few
    units and outputs have unit scale."""
    h, heads, kv, d = config.hidden_size, config.num_q_heads, config.kv_dim, config.head_dim
    kernel, rel = config.conv_kernel, config.d_rel

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, device="cuda", generator=gen) * scale

    def weight(rows, cols, scale=1.0):
        return randn(rows, cols, scale=scale * cols**-0.5).bfloat16()

    def norm():
        return (1 + randn(h, scale=0.1)).bfloat16()

    fields = {
        "wq": weight(heads * d, h),
        "wk": weight(kv, h, 0.5),
        "wv": weight(kv, h),
        "wo": weight(h, heads * d),
        "wgate": weight(heads * d, h, 2.0),
        "wrel": weight(heads * rel, h),
        "wsink": weight(heads, h, 2.0),
        "q_scale": (1 + 2 * torch.rand(heads, d, device="cuda", generator=gen)).bfloat16(),
        "rel_scale": (1 + randn(heads, rel, scale=0.2)).bfloat16(),
        "rel_proj": randn(rel, config.rel_extent, scale=0.5).bfloat16(),
        "attn_conv": randn(h, kernel, scale=0.3),
    }
    if layer > 0:
        queries = 3 if config.is_full(layer) else 1
        fields["attn_query"] = randn(queries, h)
    if config.is_full(layer):
        fields.update(
            q_norm=norm(),
            k_norm=norm(),
            v_norm=norm(),
            k_static=randn(kernel, kv, scale=0.3).bfloat16(),
            v_static=randn(kernel, kv, scale=0.3).bfloat16(),
            k_dynamic=weight(config.num_kv_heads * kernel, h, 0.3),
            v_dynamic=weight(config.num_kv_heads * kernel, h, 0.3),
        )
    else:
        fields.update(
            input_norm=norm(),
            k_conv=randn(kv, kernel, scale=0.3),
            v_conv=randn(kv, kernel, scale=0.3),
        )
    return fields


def activation_format(layer: int) -> str:
    """The activation quantization Inkling 2 Mini declares for ``layer``'s experts."""
    return NVFP4 if layer < 20 else MXFP8


def random_mlp(config: ref.InklingConfig, gen, activation: str = NVFP4) -> dict:
    """MLP weights at ``config``'s shapes, scaled like a trained layer's (unit gate/up and down
    outputs), with declared numerics quantizing the experts' inputs to ``activation``
    (``ref.ExpertQuant``; bf16 weights without global scales)."""
    h, width, experts = config.hidden_size, config.moe_intermediate_size, config.num_experts
    numerics = ref.ExpertQuant(ActivationQuant(activation))

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, device="cuda", generator=gen) * scale

    return {
        "mlp_conv": randn(h, config.conv_kernel, scale=0.3),
        "post_norm": (1 + randn(h, scale=0.1)).bfloat16(),
        "mlp_query": randn(h),
        "router": randn(experts, h, scale=h**-0.5).bfloat16().float(),  # bf16-exact, as Inkling's
        "router_bias": randn(experts, scale=0.05),
        "w13": randn(experts, 2 * width, h, scale=h**-0.5).bfloat16(),
        "w2": randn(experts, h, width, scale=width**-0.5).bfloat16(),
        "w13_quant": numerics,
        "w2_quant": numerics,
    }


def random_layer(config: ref.InklingConfig, layer: int, seed: int) -> ref.LayerWeights:
    """Both halves' random weights of ``layer`` (no message gates)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    attention = random_attention(config, layer, gen)
    return ref.LayerWeights(**attention, **random_mlp(config, gen, activation_format(layer)))


def random_cache(config: ref.InklingConfig, visits: int, max_len: int, seed: int):
    """A cache of ``visits`` visits with random keys, values and rings everywhere: rows at and
    past the position must be masked, ring rows before position 0 read as zeros."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    kv, d = config.num_kv_heads, config.head_dim

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, device="cuda", generator=gen) * scale).bfloat16()

    ring = ref.CONV_RING
    return ref.InklingCache(
        k=randn(visits, kv, max_len, d, scale=0.7),
        v=randn(visits, kv, max_len, d),
        conv_k=randn(visits, ring, config.kv_dim, scale=0.5),
        conv_v=randn(visits, ring, config.kv_dim),
        conv_attn=randn(visits, ring, config.hidden_size),
        conv_mlp=randn(visits, ring, config.hidden_size),
    )


def random_inputs(config: ref.InklingConfig, blocks: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    h = config.hidden_size
    states = [
        torch.randn(1, h, device="cuda", generator=gen).bfloat16() for _ in range(blocks + 1)
    ]
    return states[-1], states[:-1]


def ulps(out: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    """|out - expected| in bf16 ulps of each expected element, at least the ulp of a value at
    the expected tensor's RMS (an element that cancelled to near zero keeps its inputs' scale)."""
    expected = expected.float()
    rms = expected.square().mean().sqrt()
    exponent = torch.floor(torch.log2(expected.abs().clamp(min=rms.item())))
    return (out.float() - expected).abs() / 2.0 ** (exponent - 7)


def scale_ulps(out: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    """|out - expected| in ulps of a bf16 value at the expected tensor's RMS."""
    rms = expected.float().square().mean().sqrt()
    return (out.float() - expected.float()).abs() / 2.0 ** (torch.floor(torch.log2(rms)) - 7)


def clone(cache: ref.InklingCache) -> ref.InklingCache:
    return ref.InklingCache(**{k: v.clone() for k, v in vars(cache).items()})


def restore(cache: ref.InklingCache, initial: ref.InklingCache):
    """Copy ``initial``'s contents into ``cache``'s buffers."""
    for name, tensor in vars(cache).items():
        tensor.copy_(getattr(initial, name))


def assert_caches_equal(cache: ref.InklingCache, expected: ref.InklingCache, *context):
    for name, tensor in vars(cache).items():
        assert torch.equal(tensor, getattr(expected, name)), (*context, name)


def capture(fn):
    """``(graph, result)``: a CUDA graph of one call of ``fn`` and that call's result, after a
    warm-up call on the capture stream."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = fn()
    torch.cuda.current_stream().wait_stream(stream)
    return graph, result


def check_activations(runner, config, w, layer, x, experts, weights, message, launch_w=None):
    """``(routed, activations)``: the last launch's activations (the gate/up rows'
    ``bf16(swiglu(bf16(g), bf16(u)) * weight)``, ``runner.activations``) within ``ACT_ULPS`` of
    the oracle's (``ref.declared_activations``) on the launch's MLP input: the oracle's ``x`` (bf16
    ``(tokens, hidden)``), or after a ``message`` gate the launch's own gated stream normalized
    (an ``h`` rounding flip moves the whole gated stream, ``check_gates``, and the quantized
    input can amplify that into every activation of the token). The rest of the MLP is then
    checked against the oracle on the launch's activations (``ref.declared_down``), so that a
    flipped activation, which its own quantization can amplify, does not hide in the output's
    budget. The runner launches ``launch_w`` (default ``w``)."""
    if message is not None:
        gated = runner.launch_of(w if launch_w is None else launch_w, layer, True).values("gated")
        gated = gated.view(-1, GATES, config.hidden_size)[:, MLP_GATE]
        x = ref.rms_norm(gated, w.post_norm, config.rms_eps).to(torch.bfloat16)
        # The routing weights multiplying the activations must use this same gated input.
        experts, weights = ref.route(config, w, x)
    routed, acts = ref.declared_activations(config, w, x, experts, weights)
    mine = runner.activations()
    errors = ulps(mine, acts)
    changed = (errors > 0).float().mean()
    assert errors.max() <= ACT_ULPS and changed <= ACT_MAX_CHANGED, (errors.max(), changed)
    return routed, mine


def check_visit(
    runner,
    config,
    w,
    layer,
    visit,
    h,
    blocks,
    cache,
    positions,
    messages=None,
    gates=None,
    launch_w=None,
):
    """One launch against the oracle on a copy of ``cache``. Attention: ``ref.attention_block``
    (``start=None``) on the launch's inputs: the residual and the attention ring row within
    ``MAX_ULPS``, the k/v cache and ring rows within theirs, every other row untouched. MLP, on
    the launch's own residual: the oracle's experts, the activations within ``ACT_ULPS``
    (``ref.declared_activations``), and the output and the MLP ring's new row within
    ``MLP_ULPS`` of ``ref.mlp_block`` on the launch's activations, every other ring row
    untouched. A visit with ``messages`` (the oracle's ``(1, 4, hidden)``) launches with the
    matching ``gates`` (``megakernel.GateArgs``). The runner launches ``launch_w`` (default
    ``w``): quantized experts whose bf16 block values ``w`` holds. Returns the launch's
    residual."""
    theirs = clone(cache)
    expected, mlp_blocks = ref.attention_block(
        config, w, layer, visit, h, blocks, messages, theirs, positions, None, FAST
    )
    launch_w = w if launch_w is None else launch_w
    out = runner(launch_w, layer, visit, h, blocks, cache, positions, gates=gates)
    residual = runner.residual()[None]
    boundary = layer % config.block_layers == 0
    attn_out = expected[0] if boundary else expected[0] - h[0].float()
    errors = ulps(residual[0] - expected[0] + attn_out, attn_out)
    pos = int(positions)
    for name in ("k", "v"):
        mine, ref_rows = getattr(cache, name)[visit], getattr(theirs, name)[visit]
        assert ulps(mine[:, pos], ref_rows[:, pos]).max() <= CACHE_ULPS, name
        others = torch.arange(cache.max_len, device="cuda") != pos
        assert torch.equal(mine[:, others], ref_rows[:, others]), name
    slot = pos % ref.CONV_RING
    others = torch.arange(ref.CONV_RING, device="cuda") != slot
    for name in ("conv_k", "conv_v", "conv_attn"):
        mine, ref_ring = getattr(cache, name)[visit], getattr(theirs, name)[visit]
        assert torch.equal(mine[others], ref_ring[others]), name
        budget = MAX_ULPS if name == "conv_attn" else RING_ULPS
        assert ulps(mine[slot], ref_ring[slot]).max() <= budget, name
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
    assert torch.equal(runner.routed_experts(launch_w), experts[0].sort().values)
    routed, mine = check_activations(
        runner, config, w, layer, x, experts, weights, message, launch_w
    )

    def experts_on_launch_activations(config, w, x, experts, weights):
        return ref.declared_down(w, x, routed, mine)

    kernels = dataclasses.replace(ref.REFERENCE, experts=experts_on_launch_activations)
    expected = ref.mlp_block(
        config, w, visit, residual, mlp_blocks, message, theirs, positions, kernels
    )
    out_errors = scale_ulps(out, expected[0])
    changed = (out_errors > 0).float().mean()
    assert out_errors.max() <= MLP_ULPS and changed <= MLP_MAX_CHANGED, out_errors.max()
    ring, ref_ring = cache.conv_mlp[visit], theirs.conv_mlp[visit]
    assert torch.equal(ring[others], ref_ring[others])
    assert scale_ulps(ring[slot], ref_ring[slot]).max() <= MLP_ULPS
    return residual


# MXFP8 scale bytes giving weights of a trained layer's scale (the checkpoint's are 113 .. 118).
MX_SCALES = (110, 116)


def with_quantized_experts(w: ref.LayerWeights, fmt: str, seed: int):
    """``(quantized, dense)``: ``w`` with random checkpoint-format experts of ``fmt`` at
    ``CONFIG``'s shapes (every code; NVFP4 also every scale byte) and their declared numerics
    (``fmt`` activations and, NVFP4, the weights' global scales, as the loader declares them),
    and the same layer with the experts' bf16 block values."""
    c = CONFIG
    shapes = (
        (2 * c.moe_intermediate_size, c.hidden_size),
        (c.hidden_size, c.moe_intermediate_size),
    )
    w13, w2 = (
        quant.QuantExperts.from_checkpoint(
            *quant.random_checkpoint_experts(
                fmt, c.num_experts, rows, columns, seed + i, MX_SCALES
            )
        )
        for i, (rows, columns) in enumerate(shapes)
    )
    numerics = [ref.ExpertQuant(ActivationQuant(fmt), q.amax, q.denominator) for q in (w13, w2)]
    quantized = dataclasses.replace(w, w13=w13, w2=w2, w13_quant=numerics[0], w2_quant=numerics[1])
    return quantized, quant.dense_layer(quantized)


def random_core_layer(config: ref.InklingConfig, layer: int, seed: int) -> ref.LayerWeights:
    """A core layer at ``config``'s shapes: the visit tests' random weights and four message
    gates whose down projections give unit-scale ``h`` and whose forget and gain projections
    spread ``tanh`` over its range."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    h, rank = config.hidden_size, config.gate_rank

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, device="cuda", generator=gen) * scale

    gates = tuple(
        ref.MessageGate(
            down=randn(rank, 2 * h, scale=(2 * h) ** -0.5).bfloat16(),
            forget=randn(h, rank, scale=rank**-0.5),
            gain=randn(h, rank, scale=rank**-0.5),
        )
        for _ in range(4)
    )
    attention = random_attention(config, layer, torch.Generator(device="cuda").manual_seed(seed))
    gen = torch.Generator(device="cuda").manual_seed(seed)
    mlp = random_mlp(config, gen, activation_format(layer))
    return ref.LayerWeights(**attention, **mlp, gates=gates)


def random_pass(config: ref.InklingConfig, layers: int, seed: int):
    """``(queries, in12, out17)``: message queries of ``layers`` core layers and the previous
    pass's two states."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    h = config.hidden_size
    queries = torch.randn(layers, 4, h, device="cuda", generator=gen)
    states = [(torch.randn(1, h, device="cuda", generator=gen) * 3).bfloat16() for _ in range(2)]
    return queries, *states


def oracle_messages(queries, block, partial):
    """``ref.hidden_states``' messages: (layers, 1, 4, hidden) fp32."""
    return torch.stack([ref.block_residual(q, [block], partial) for q in queries])


def check_gates(config, launch, w, streams, messages, gates):
    """The last launch's message gates on their fp32 input ``streams`` (``(GATES, hidden)``):
    ``h`` within one bf16 ulp of ``ref.message_gate``'s bf16 GEMM, and the gated streams equal
    to the oracle's gate evaluated on the kernel's own ``h`` up to fp32 rounding. (An ``h``
    rounding flip moves every gated element by about 1e-3 relative: the launch outputs differ
    from the oracle's through those flips, not through the gate arithmetic.)"""
    h, rank = config.hidden_size, config.gate_rank
    h_part = launch.values("h_part").view(GATE_SPLITS, GATES, rank)
    gated = launch.values("gated").view(GATES, h)
    mhalf = gates.mhalf.view(-1, rank)
    for s in range(GATES):
        gate, x, m = w.gates[s], streams[s], messages[0, s]
        inputs = torch.cat([ref.rms_norm(v, None, config.rms_eps) for v in (x, m)])
        expected = F.linear(inputs.bfloat16(), gate.down).float()
        mine = h_part[0, s]
        for k in range(1, GATE_SPLITS):  # the kernel's summation order
            mine = mine + h_part[k, s]
        mine = (mine + mhalf[s]).bfloat16().float()
        ulp = 2.0 ** (torch.floor(torch.log2(expected.abs())) - 7)
        assert ((mine - expected).abs() <= ulp).all(), s
        out = (1 - torch.tanh(gate.forget @ mine)) * x + torch.tanh(gate.gain @ mine) * m
        torch.testing.assert_close(gated[s], out, rtol=1e-5, atol=1e-5)


# (layer, blocks, positions): a sliding NVFP4 layer inside a block at position 0 (only the new
# token and the sink), across a KV block edge and past the sliding window (1100 keys: only the
# last 1024 are read); a full MXFP8 block-boundary layer reading 4 blocks, also past
# ``log_scaling_floor``, where queries and relative states scale up.
CASES = [(1, 1, (0, 64, 1100)), (23, 4, (130, 9000))]


@pytest.mark.parametrize("layer,blocks,positions", CASES)
@torch.no_grad()
def test_visit_random_weights(layer, blocks, positions):
    """Several launches through one runner (the epoch advances), each within ``check_visit``'s
    budgets of the oracle on the experts' block values."""
    w = random_layer(CONFIG, layer, seed=layer)
    quantized, dense = with_quantized_experts(w, activation_format(layer), seed=10)
    runner = InklingVisitMegakernel(CONFIG, (quantized,))
    cache = random_cache(CONFIG, visits=2, max_len=9024, seed=layer)
    for case, pos in enumerate(positions):
        h, states = random_inputs(CONFIG, blocks, seed=100 * layer + case)
        positions_ = torch.tensor([pos], device="cuda")
        check_visit(
            runner, CONFIG, dense, layer, 1, h, states, cache, positions_, launch_w=quantized
        )


@torch.no_grad()
def test_quantized_launch_equals_block_value_launch():
    """Streaming NVFP4 experts as stored, every launch output (visit output, post-attention
    residual, routing, cache and ring rows) equals the same launch on the experts' bf16 block
    values bit for bit: ``quant_gemv`` decodes to the block values and accumulates in the bf16
    GEMV's order."""
    quantized, dense = with_quantized_experts(random_layer(CONFIG, 1, seed=1), NVFP4, seed=10)
    cache = random_cache(CONFIG, visits=1, max_len=1152, seed=1)
    h, states = random_inputs(CONFIG, 1, seed=2)
    results = []
    for w in (quantized, dense):
        runner, mine = InklingVisitMegakernel(CONFIG, (w,)), clone(cache)
        out = runner(w, 1, 0, h, states, mine, torch.tensor([1100], device="cuda")).clone()
        results.append((out, runner.residual().clone(), runner.routed_experts(w), mine))
    (out, residual, experts, mine), (out_d, residual_d, experts_d, theirs) = results
    assert torch.equal(out.view(torch.int16), out_d.view(torch.int16))
    assert torch.equal(residual.view(torch.int32), residual_d.view(torch.int32))
    assert torch.equal(experts, experts_d)
    assert_caches_equal(mine, theirs)


@torch.no_grad()
def test_message_visit_random_weights():
    """A full core visit after the first pass: the message launch (messages against
    ``ref.block_residual``), then the visit launch (q, k, v gates on the attention's inputs, the
    MLP gate on the MLP's) within the plain visits' budgets of the oracle with its messages, and
    the gates themselves against ``ref.message_gate``."""
    config, layer, blocks = CONFIG, 11, 2
    w = random_core_layer(config, layer, seed=layer)
    quantized, dense = with_quantized_experts(w, NVFP4, seed=20)
    queries, block, partial = random_pass(config, 1, seed=layer)
    gates = MessageGates((quantized,), queries)
    InklingMsgMegakernel(config, gates)(block, partial)
    messages = oracle_messages(queries, block, partial)[0]  # (1, 4, hidden)
    torch.testing.assert_close(gates.messages.view_as(messages), messages, atol=1e-5, rtol=1e-5)
    runner = InklingVisitMegakernel(config, (quantized,))
    cache = random_cache(config, visits=2, max_len=1152, seed=layer)
    for case, pos in enumerate((3, 1100)):
        positions = torch.tensor([pos], device="cuda")
        h, states = random_inputs(config, blocks, seed=100 * layer + case)
        args = gates.layer(0)
        residual = check_visit(
            runner, config, dense, layer, 1, h, states, cache, positions, messages, args, quantized
        )
        streams = ref.block_residual(w.attn_query, states, h)[0]
        mlp_stream = ref.block_residual(w.mlp_query[None], states, residual)[0]
        streams = torch.cat([streams, mlp_stream])
        check_gates(config, runner.launch_of(quantized, layer, True), w, streams, messages, args)


@torch.no_grad()
def test_graph_replay_and_compile_reuse():
    """The launch replays bit-identically to eager launches inside a CUDA graph, at device
    positions that change the KV splits, and positions, block counts and cache capacities are
    runtime values: varying them compiles nothing."""
    w, _ = with_quantized_experts(random_layer(CONFIG, 1, seed=1), NVFP4, seed=10)
    runner = InklingVisitMegakernel(CONFIG, (w,))
    cache = random_cache(CONFIG, visits=1, max_len=1152, seed=1)
    initial = clone(cache)
    h, states = random_inputs(CONFIG, 1, seed=2)
    positions = torch.tensor([1030], device="cuda")
    out = torch.empty(CONFIG.hidden_size, device="cuda", dtype=torch.bfloat16)
    runner(w, 1, 0, h, states, cache, positions, out)
    entries = compile_visit_kernel.cache_info().currsize

    graph, _ = capture(lambda: runner(w, 1, 0, h, states, cache, positions, out))
    for pos in (1030, 70):
        positions.fill_(pos)
        expected_cache = clone(initial)
        expected = runner(w, 1, 0, h, states, expected_cache, positions).clone()
        expected_residual = runner.residual().clone()
        restore(cache, initial)
        out.zero_()
        graph.replay()
        assert torch.equal(out, expected), pos
        assert torch.equal(runner.residual(), expected_residual), pos
        assert_caches_equal(cache, expected_cache, pos)

    before = compile_visit_kernel.cache_info()
    other = random_cache(CONFIG, visits=1, max_len=2048, seed=3)
    for pos, blocks, target in [(0, 0, cache), (64, 4, cache), (2000, 2, other)]:
        positions.fill_(pos)
        h, states = random_inputs(CONFIG, blocks, seed=pos)
        runner(w, 1, 0, h, states, target, positions)
    info = compile_visit_kernel.cache_info()
    assert info.currsize == entries
    assert info.misses == before.misses
    assert info.hits + info.misses == before.hits + before.misses + 3  # one per eager launch
