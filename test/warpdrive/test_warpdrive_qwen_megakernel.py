"""Qwen3 decode megakernel: each instruction alone, then whole decode steps."""

import dataclasses

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cutlass.cute")

from attn_gym._backends.arch import cuda_runs_sm100_kernels
from attn_gym.testing.perfetto_trace import read_slices
from attn_gym.warpdrive import ProtocolError, RoleConfig
from attn_gym.warpdrive.examples import qwen_megakernel as mk
from attn_gym.warpdrive.examples.megakernel.constants import COMPUTE_WARPS, HEAD_DIM, ROWS
from attn_gym.warpdrive.examples.qwen_megakernel import trace as mk_trace
from attn_gym.warpdrive.examples.qwen_megakernel.ref import (
    QwenConfig,
    YarnScaling,
    compare_logits,
    decode_step_ref,
    forward_tokens,
    load_qwen3_weights,
    make_decode_state,
    random_weights,
    rms_norm,
    rope,
)
from attn_gym.warpdrive.profiler import Range
from attn_gym.warpdrive.trace_session import _roles, work_item_ranges

pytestmark = pytest.mark.usefixtures("release_cuda_memory")

MAX_LEN = 256


def require_decode_engine(engine: str) -> None:
    if not cuda_runs_sm100_kernels() and not (
        engine == "cuda_core"
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (12, 1)
    ):
        pytest.skip("requires an SM100-family GPU or the CUDA-core engine on SM121")


def cached_weights(repo: str):
    """Load the checkpoint from the local HF cache; skip if it is missing, never download."""
    hub = pytest.importorskip("huggingface_hub")
    pytest.importorskip("safetensors")
    path = hub.try_to_load_from_cache(repo, "config.json")
    if not isinstance(path, str):
        pytest.skip(f"{repo} is not in the local HF cache")
    return load_qwen3_weights(path)


@pytest.fixture(scope="module")
def sm100_checkpoint_matrix():
    if not cuda_runs_sm100_kernels():
        pytest.skip("the checkpoint matrix requires an SM100-family GPU")


@pytest.fixture(scope="module")
def weights_0_6b(sm100_checkpoint_matrix):
    return cached_weights("Qwen/Qwen3-0.6B")


# Checkpoints of 16 GB and up share one xdist worker, so -n never loads two at once.
LARGE = pytest.mark.xdist_group("qwen-large-checkpoints")


@pytest.fixture(
    scope="module",
    params=[
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        pytest.param("Qwen/Qwen3-8B", marks=LARGE),
        pytest.param("Qwen/Qwen3-14B", marks=LARGE),
        pytest.param("Qwen/Qwen3-32B", marks=LARGE),
    ],
)
def weights(request, sm100_checkpoint_matrix):
    """Family members that each change a tiling: 1.7B doubles the hidden size; 8B runs two
    attention rows per kv head, 8-head o_proj splits and an untied LM head; 14B serves one q
    head per attention row (GQA group 5), splits the o_proj combine into partial passes and
    the residual polls into chunks; 32B drops to 11 ring stages."""
    return cached_weights(request.param)


def bf16_values(*shape):
    return torch.randn(*shape, device="cuda").bfloat16().float()


def cache_shape(config: QwenConfig) -> tuple[int, int, int, int]:
    return config.num_layers, config.num_kv_heads, MAX_LEN, config.head_dim


def rope_at(pos: int):
    table = mk.rope_table(QwenConfig(), pos + 1, "cuda")[pos]
    return table[0][None], table[1][None]


@pytest.mark.parametrize("engine", ["cuda_core", "tcgen05"])
@pytest.mark.parametrize("op", ["qkv", "attn", "gateup", "down", "lmhead"])
def test_single_instruction(weights, op, engine):
    """One instruction type over all layers, from random inputs in its scratch buffers."""
    torch.manual_seed(0)
    runner = mk.QwenMegakernel(weights, MAX_LEN, ops=(getattr(mk, op.upper()),), engine=engine)
    config = weights.config
    q_heads, kv_heads = config.num_q_heads, config.num_kv_heads
    view = runner.values
    token, pos = 1234, 150  # three KV blocks, the last partial
    caches = [bf16_values(*cache_shape(config)).bfloat16() for _ in "kv"]
    inputs = {}
    # Not the down partials: the last split would take the others' as this launch's.
    for name in ("h_in", "h_mid", "qkv", "act", "o_part"):
        shape = view(name).shape
        values = torch.randn(shape, device="cuda") / 4 if "part" in name else bf16_values(*shape)
        runner.set_values(name, values)  # tagged with this launch's epoch
        inputs[name] = values
    runner.step_info.copy_(torch.tensor([token, pos], dtype=torch.int32))
    runner.launch(*caches)
    torch.cuda.synchronize()
    assert runner.state[0].item() == 1 and not runner.state[1:].any(), "one epoch per step"

    w, eps = weights, 1e-6
    for layer in range(config.num_layers):
        if op == "qkv":
            x = w.embed[token] if layer == 0 else inputs["h_in"][layer].bfloat16()
            norm = rms_norm(x, w.input_ln[layer], eps)
            expected = F.linear(
                norm, torch.cat((w.q_proj[layer], w.k_proj[layer], w.v_proj[layer]))
            )
            torch.testing.assert_close(view("h_in")[layer], x.float())
            torch.testing.assert_close(view("qkv")[layer], expected.float(), atol=1e-2, rtol=1e-2)
        elif op == "attn":
            qkv = inputs["qkv"][layer].bfloat16()
            cos, sin = rope_at(pos)
            q, k, v = qkv.view(-1, HEAD_DIM).split((q_heads, kv_heads, kv_heads))
            q = rope(rms_norm(q, w.q_norm[layer], eps), cos, sin)
            k = rope(rms_norm(k, w.k_norm[layer], eps), cos, sin)
            torch.testing.assert_close(caches[0][layer, :, pos], k)
            torch.testing.assert_close(caches[1][layer, :, pos], v)
            group = q_heads // kv_heads
            keys = caches[0][layer, :, : pos + 1].float().repeat_interleave(group, 0)
            values = caches[1][layer, :, : pos + 1].float().repeat_interleave(group, 0)
            scores = (q[:, None].float() @ keys.transpose(-1, -2)).bfloat16() * mk.ATTN_SCALE
            expected = (F.softmax(scores.float(), -1) @ values)[:, 0]
            o = view("attn_o")[layer]  # (row group, split, head, dim)
            m, den = view("attn_ml")[layer].transpose(1, 2).unbind(-1)  # (row group, split, head)
            scale = torch.exp2(m - m.amax(1, keepdim=True))
            out = (o * scale[..., None]).sum(1) / (den * scale).sum(1)[..., None]
            torch.testing.assert_close(out.reshape(q_heads, -1), expected, atol=2e-3, rtol=1e-2)
        elif op == "gateup":
            h = (inputs["h_in"][layer] + inputs["o_part"][layer].sum(0).bfloat16()).bfloat16()
            norm = rms_norm(h, w.post_ln[layer], eps)
            expected = F.silu(F.linear(norm, w.gate[layer])) * F.linear(norm, w.up[layer])
            torch.testing.assert_close(view("h_mid")[layer], h.float())
            # SiLU(g) * u rounds g, SiLU(g), u and the product: a few bf16 ulps apart.
            torch.testing.assert_close(view("act")[layer], expected.float(), atol=1e-2, rtol=2e-2)
        elif op == "down":
            # The last split sums the partials and publishes the next layer's residual.
            out = F.linear(inputs["act"][layer].bfloat16(), w.down[layer])
            torch.testing.assert_close(
                view("d_part")[layer].sum(0), out.float(), atol=1e-2, rtol=1e-2
            )
            h = (inputs["h_mid"][layer] + out).bfloat16()
            torch.testing.assert_close(view("h_in")[layer + 1], h.float(), atol=2e-2, rtol=1e-2)
    if op == "lmhead":
        x = inputs["h_in"][-1].bfloat16()
        expected = F.linear(rms_norm(x, w.final_norm, eps), w.lm_head).float()
        torch.testing.assert_close(runner.logits, expected, atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize("engine", ["cuda_core", "tcgen05"])
def test_gateup_sums_a_residual_wider_than_registers(engine):
    """Batch-one gate/up sums h_in and o_proj's partials itself. Past 8192 hidden columns a
    thread's share outgrows its registers and is staged through SMEM: h_mid and the activation
    must match the reference either way."""
    torch.manual_seed(2)
    config = QwenConfig(
        hidden_size=9216, num_layers=1, num_q_heads=8, num_kv_heads=8, intermediate_size=1024,
        vocab_size=1024,
    )  # fmt: skip
    weights = random_weights(config)
    runner = mk.QwenMegakernel(weights, MAX_LEN, ops=(mk.GATEUP,), engine=engine)
    inputs = {}
    for name in ("h_in", "o_part"):
        shape = runner.values(name).shape
        values = torch.randn(shape, device="cuda") / 4 if name == "o_part" else bf16_values(*shape)
        runner.set_values(name, values)
        inputs[name] = values
    runner.step_info.copy_(torch.tensor([0, 10], dtype=torch.int32))
    runner.launch(*[bf16_values(*cache_shape(config)).bfloat16() for _ in "kv"])
    torch.cuda.synchronize()
    h = (inputs["h_in"][0] + inputs["o_part"][0].sum(0).bfloat16()).bfloat16()
    norm = rms_norm(h, weights.post_ln[0], 1e-6)
    expected = F.silu(F.linear(norm, weights.gate[0])) * F.linear(norm, weights.up[0])
    torch.testing.assert_close(runner.values("h_mid")[0], h.float())
    torch.testing.assert_close(runner.values("act")[0], expected.float(), atol=1e-2, rtol=2e-2)


@pytest.mark.parametrize("engine", ["cuda_core", "tcgen05"])
def test_oproj_instruction(weights, engine):
    """o_proj combines KV-split partials: split s carries weight 2^m_s and sum l_s."""
    torch.manual_seed(1)
    runner = mk.QwenMegakernel(weights, MAX_LEN, ops=(mk.OPROJ,), engine=engine)
    o = torch.randn(runner.values("attn_o").shape, device="cuda")  # (L, group, split, head, d)
    m = torch.randn(runner.values("attn_ml").shape[:-1], device="cuda") * 3  # (L, grp, head, sp)
    den = torch.rand(m.shape, device="cuda") + 0.5
    runner.set_values("attn_o", o)
    runner.set_values("attn_ml", torch.stack((m, den), -1))
    m, den = m.transpose(2, 3), den.transpose(2, 3)  # (L, group, split, head) like o
    runner.step_info.copy_(torch.tensor([0, 100], dtype=torch.int32))
    shape = cache_shape(weights.config)
    runner.launch(*[torch.zeros(shape, device="cuda", dtype=torch.bfloat16) for _ in "kv"])
    torch.cuda.synchronize()
    scale = torch.exp2(m - m.amax(2, keepdim=True))
    attn = (o * scale[..., None]).sum(2) / (den * scale).sum(2)[..., None]
    attn = attn.reshape(weights.config.num_layers, -1).bfloat16()
    expected = torch.einsum("lk,lnk->ln", attn.float(), weights.o_proj.float())
    torch.testing.assert_close(runner.values("o_part").sum(1), expected, atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "ctx, max_len, schedule, splits",
    [
        (127, 256, "static", 16),
        (127, 256, "atomic", 16),
        (1500, 1536, "static", 19),
    ],
)
@pytest.mark.parametrize("engine", ["cuda_core", "tcgen05"])
def test_decode_steps(weights, ctx, max_len, schedule, splits, engine):
    check_decode_steps(weights, ctx, max_len, schedule, splits, engine)


def test_sm121_checkpoint_decode_steps():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
        pytest.skip("requires SM121")
    weights = cached_weights("Qwen/Qwen3-0.6B")
    check_decode_steps(
        weights, ctx=127, max_len=256, schedule="static", splits=16, engine="cuda_core"
    )


def check_decode_steps(weights, ctx, max_len, schedule, splits, engine):
    """Test teacher-forced decode steps under both schedules against an fp32 oracle.

    Context 127 crosses a KV-block boundary at position 128. Context 1500 gives each split
    two blocks with a non-power-of-two split count. Logits and new K/V rows stay within the
    bf16 reference's error budget. Pick the oracle's token when its lead exceeds 0.5, write
    only cache row pos, mask NaN-poisoned unused rows, and repeat launches bit for bit.
    """
    runner = mk.QwenMegakernel(weights, max_len, schedule=schedule, splits=splits, engine=engine)
    caches, token, pos = make_decode_state(weights, ctx, max_len, seed=ctx)
    for cache in caches:
        cache[:, :, pos:] = float("nan")
    ours, theirs, oracle = ([c.clone() for c in caches] for _ in range(3))
    errors, ref_errors = [], []
    for step in range(3):
        before = [c.clone() for c in ours]
        logits = runner(weights, token, pos + step, *ours)
        again = runner(weights, token, pos + step, *[c.clone() for c in before])
        assert torch.equal(logits, again)
        for cache, old in zip(ours, before):
            keep = torch.ones(max_len, dtype=torch.bool, device="cuda")
            keep[pos + step] = False
            assert torch.equal(cache[:, :, keep].isnan(), old[:, :, keep].isnan())
            torch.testing.assert_close(cache[:, :, keep], old[:, :, keep], equal_nan=True)
        ref = decode_step_ref(weights, token, pos + step, *theirs)
        high = forward_tokens(weights, token.reshape(1), pos + step, *oracle, dtype=torch.float32)
        metrics = compare_logits(logits, ref, high)
        assert torch.isfinite(logits).all()
        # Single steps differ by bf16 noise accumulated over all layers; bound the max error
        # per step loosely and the RMS error over the steps.
        assert metrics["test_fp32_max_err"] <= 3 * metrics["bf16_fp32_max_err"], metrics
        top2 = high.topk(2).values
        if top2[0] - top2[1] > 0.5:
            assert logits.argmax() == high.argmax(), metrics
        errors.append(metrics["test_fp32_rms_err"])
        ref_errors.append(metrics["bf16_fp32_rms_err"])
        for mine, other, exact in zip(ours, theirs, oracle):  # the new K/V rows
            row, ref_row, exact_row = (c[:, :, pos + step].float() for c in (mine, other, exact))
            assert (row - exact_row).norm() <= 2 * (ref_row - exact_row).norm()
        token = high.argmax()
        for cache, source in zip((*ours, *theirs), (*oracle, *oracle)):
            cache[:, :, pos + step] = source[:, :, pos + step].bfloat16()  # same inputs next step
    assert sum(errors) <= 2 * sum(ref_errors), (errors, ref_errors)
    assert runner.state[0].item() == 6 and not runner.state[1:].any()  # one epoch per step


@pytest.mark.parametrize("engine", ["cuda_core", "tcgen05"])
def test_atomic_schedule_matches_the_static_schedule(engine):
    """The atomic schedule's per-layer launches match the default static schedule bit for bit,
    logits and new K/V rows, across repeated steps. The epoch advances once per step."""
    require_decode_engine(engine)
    config = QwenConfig(
        hidden_size=1024, num_layers=3, num_q_heads=16, num_kv_heads=8,
        intermediate_size=3072, vocab_size=4096,
    )  # fmt: skip
    weights = random_weights(config)
    caches, token, pos = make_decode_state(weights, 130, MAX_LEN, seed=5)
    ours, theirs = [c.clone() for c in caches], [c.clone() for c in caches]
    runner = mk.QwenMegakernel(weights, MAX_LEN, engine=engine, schedule="atomic")
    reference = mk.QwenMegakernel(weights, MAX_LEN, engine=engine)
    assert len(runner.tables) == config.num_layers + 1  # a launch per layer, then the LM head
    for step in range(3):
        logits = runner(weights, token, pos + step, *ours)
        expected = reference(weights, token, pos + step, *theirs)
        assert torch.equal(logits, expected)
        token = expected.argmax()
    for mine, other in zip(ours, theirs):
        assert torch.equal(mine[:, :, : pos + 3], other[:, :, : pos + 3])
    assert runner.state[0].item() == 3 and not runner.state[1:].any()


@pytest.mark.parametrize("engine", ["cuda_core", "tcgen05"])
def test_cuda_graph_of_steps_matches_eager(engine):
    """A CUDA graph of several steps (as ``chat.py`` captures: position and token advanced on the
    device) replays to the same tokens, logits, K/V rows and step state as eager steps."""
    require_decode_engine(engine)
    config = QwenConfig(
        hidden_size=1024, num_layers=3, num_q_heads=16, num_kv_heads=8,
        intermediate_size=3072, vocab_size=4096,
    )  # fmt: skip
    weights = random_weights(config)
    caches, token, pos = make_decode_state(weights, 120, MAX_LEN, seed=7)
    steps, replays = 4, 2

    def decode(runner, caches, out):
        for i in range(steps):
            runner.step_info[1].add_(1)
            runner.launch(*caches)
            runner.step_info[0].copy_(runner.logits.argmax())
            out[i].copy_(runner.step_info[0])

    def start(runner):
        runner.step_info[0] = token
        runner.step_info[1] = pos - 1

    eager = mk.QwenMegakernel(weights, MAX_LEN, engine=engine)
    graphed = mk.QwenMegakernel(weights, MAX_LEN, engine=engine)
    ours, theirs = [c.clone() for c in caches], [c.clone() for c in caches]
    graphed(weights, token, pos, *[c.clone() for c in caches])  # compile and warm
    out = torch.zeros(steps, device="cuda", dtype=torch.long)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode(graphed, ours, out)
    start(graphed)
    start(eager)
    expected = torch.zeros_like(out)
    for _ in range(replays):
        graph.replay()
        decode(eager, theirs, expected)
        assert torch.equal(out, expected)
        assert torch.equal(graphed.logits, eager.logits)
    end = pos + steps * replays
    for mine, other in zip(ours, theirs):
        assert torch.equal(mine[:, :, :end], other[:, :, :end])
    assert graphed.state[0].item() == 1 + steps * replays and not graphed.state[1:].any()


@pytest.mark.parametrize("engine", ["cuda_core", "tcgen05"])
def test_odd_gqa_group_random_weights(engine):
    """A GQA group of 3 (like Qwen3-14B's 5) serves one q head per attention row; its o_proj
    splits hold 6 heads, so the combine runs a full and a partial pass. No checkpoint needed."""
    require_decode_engine(engine)
    config = QwenConfig(
        hidden_size=1024, num_layers=2, num_q_heads=24, num_kv_heads=8,
        intermediate_size=2048, vocab_size=4096,
    )  # fmt: skip
    weights = random_weights(config)
    packed = torch.stack((weights.gate, weights.up), dim=-2)
    weights = dataclasses.replace(
        weights, packed_gate_up=packed, gate=packed.select(-2, 0), up=packed.select(-2, 1)
    )
    with pytest.raises(ValueError, match="smem_bytes is determined by the weights' device"):
        mk.QwenMegakernel(weights, MAX_LEN, smem_bytes=65536)
    runner = mk.QwenMegakernel(weights, MAX_LEN, engine=engine)
    assert runner.wgu.is_set_to(packed.view(-1, config.hidden_size))
    # Replacing a logical weight must also change what decode reads.
    weights = dataclasses.replace(weights, gate=-weights.gate)
    runner = mk.QwenMegakernel(weights, MAX_LEN, engine=engine)
    assert not runner.wgu.is_set_to(packed.view(-1, config.hidden_size))
    assert (runner.op.per_row, runner.op.heads_per_split) == (1, 6)
    caches, token, pos = make_decode_state(weights, 150, MAX_LEN, seed=3)
    ours, theirs, exact = ([c.clone() for c in caches] for _ in range(3))
    logits = runner(weights, token, pos, *ours)
    ref = decode_step_ref(weights, token, pos, *theirs)
    high = forward_tokens(weights, token.reshape(1), pos, *exact, dtype=torch.float32)
    metrics = compare_logits(logits, ref, high)
    assert metrics["passed"], metrics
    for mine, oracle in zip(ours, exact):
        torch.testing.assert_close(mine[:, :, pos], oracle[:, :, pos], atol=0.05, rtol=0.02)


def test_cache_capacity_reuses_the_kernel():
    """The cache capacity is a runtime size (``compile_megakernel``): runners of another
    capacity reuse the compiled kernel."""
    if not cuda_runs_sm100_kernels():
        pytest.skip("the megakernel requires an SM100-family GPU")
    config = QwenConfig(
        hidden_size=1024, num_layers=2, num_q_heads=16, num_kv_heads=8,
        intermediate_size=2048, vocab_size=4096,
    )  # fmt: skip
    weights = random_weights(config)
    mk.compile_megakernel(mk.QwenMegakernel(weights, MAX_LEN).op)
    compiled = mk.compile_megakernel.cache_info().currsize
    mk.compile_megakernel(mk.QwenMegakernel(weights, 6 * MAX_LEN).op)
    assert mk.compile_megakernel.cache_info().currsize == compiled


def test_yarn_rope_random_weights():
    """A YaRN config past its original context: the runner's RoPE table follows the config's
    scaling, so the step matches the reference, and it reuses the plain config's kernel."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the megakernel requires an SM100-family GPU")
    yarn = YarnScaling.from_hf({"factor": 8.0, "original_max_position_embeddings": 32}, 32)
    config = QwenConfig(
        hidden_size=1024, num_layers=2, num_q_heads=16, num_kv_heads=8,
        intermediate_size=2048, vocab_size=4096, rope_scaling=yarn,
    )  # fmt: skip
    weights = random_weights(config)
    plain = dataclasses.replace(weights, config=dataclasses.replace(config, rope_scaling=None))
    mk.compile_megakernel(mk.QwenMegakernel(plain, MAX_LEN).op)
    compiled = mk.compile_megakernel.cache_info().currsize
    runner = mk.QwenMegakernel(weights, MAX_LEN)
    mk.compile_megakernel(runner.op)
    assert mk.compile_megakernel.cache_info().currsize == compiled  # the plain config's kernel
    caches, token, pos = make_decode_state(weights, 200, MAX_LEN, seed=5)
    ours, theirs, exact = ([c.clone() for c in caches] for _ in range(3))
    logits = runner(weights, token, pos, *ours)
    ref = decode_step_ref(weights, token, pos, *theirs)
    high = forward_tokens(weights, token.reshape(1), pos, *exact, dtype=torch.float32)
    metrics = compare_logits(logits, ref, high)
    assert metrics["passed"], metrics
    for mine, oracle in zip(ours, exact):
        torch.testing.assert_close(mine[:, :, pos], oracle[:, :, pos], atol=0.05, rtol=0.02)


@pytest.mark.parametrize("engine", ["cuda_core", "tcgen05"])
def test_moved_roles_match_defaults(weights_0_6b, engine):
    """The roles index their own threads: moving compute off warp 0 changes no bit. With the
    tcgen05 engine the TMEM loads then run on other hardware warps, whose subpartitions (warp
    index % 4) are rotated."""
    moved = {"load": RoleConfig(0), "compute": RoleConfig(range(1, 1 + COMPUTE_WARPS))}
    caches, token, pos = make_decode_state(weights_0_6b, 100, MAX_LEN, seed=0)
    ours, theirs = [c.clone() for c in caches], [c.clone() for c in caches]
    runner = mk.QwenMegakernel(weights_0_6b, MAX_LEN, roles=moved, engine=engine)
    logits = runner(weights_0_6b, token, pos, *ours)
    reference = mk.QwenMegakernel(weights_0_6b, MAX_LEN, engine=engine)
    expected = reference(weights_0_6b, token, pos, *theirs)
    assert torch.equal(logits, expected)
    for mine, other in zip(ours, theirs):  # rows past pos are uninitialized
        assert torch.equal(mine[:, :, : pos + 1], other[:, :, : pos + 1])


@pytest.mark.parametrize("engine", ["cuda_core", "tcgen05"])
def test_globaltimer_trace(weights_0_6b, tmp_path, engine):
    """Tracing preserves logits and records every table row's load and compute ranges, across
    all of a step's launches.

    Every role's work items carry distinct table rows, each row's CTA is the one the schedule
    gave it, each phase has a span, and the trace links rows across CTAs.
    """
    caches, token, pos = make_decode_state(weights_0_6b, 100, MAX_LEN, seed=0)
    ours, theirs = [c.clone() for c in caches], [c.clone() for c in caches]
    runner = mk.QwenMegakernel(weights_0_6b, MAX_LEN, trace="globaltimer", engine=engine)
    with mk_trace.trace_session(runner, trace_path=str(tmp_path / "step.pftrace")) as session:
        logits = runner(weights_0_6b, token, pos, *ours)
    reference = mk.QwenMegakernel(weights_0_6b, MAX_LEN, engine=engine)
    assert torch.equal(logits, reference(weights_0_6b, token, pos, *theirs))
    table = runner.table.tolist()
    rows = [r for r, words in enumerate(table) if words[0] >= 0]
    for role in ("compute", "load"):
        ranges = [e for e in session.ranges if e.name == role]
        assert sorted(e.payload for e in ranges) == rows
    for role in _roles(runner.op):  # e.g. tcgen05's mma and xload: one item per row at most
        payloads = [
            e.payload for e in session.ranges if e.name == role.name and e.payload is not None
        ]
        assert len(payloads) == len(set(payloads)) and set(payloads) <= set(rows), role.name
    op, ctas = runner.op, runner.max_ctas
    planned = {
        tuple(words): span.cta
        for (words, _, _), span in zip(
            mk.instruction_rows(op),
            mk.list_schedule(op, mk.instruction_rows(op), ctas),
        )
    }
    computed = work_item_ranges(session.ranges, "compute")
    assert all(computed[row].cta[0] == planned[tuple(table[row][:4])] for row in rows)
    spans = mk_trace.phase_spans(session.ranges, runner.table)
    ops = {mk.OP_NAMES[op] for op in runner.table[:, 0].tolist() if op >= 0}
    assert set(spans) == ops and all(v > 0 for v in spans.values())
    assert set(mk_trace.input_waits(session.ranges, runner.table)) == ops
    slices = read_slices(tmp_path / "step.pftrace")
    flows = [flow for item in slices for flow in item.flows_out]
    # Layer 0's QKV rows wait on nothing.
    assert len(flows) >= len(rows) - runner.op.qkv_rows // ROWS
    assert {item.name for item in slices if item.track == "compute"} >= ops


def test_trace_analysis_on_synthetic_rows():
    """Phase spans run from the producer phase's last row (QKV: the previous layer's down; the
    LM head: the last layer's) to the phase's last row; input waits from a row's start to its
    first poll's end; arrows leave each input group's last-finishing producer row."""
    kernel = mk.QwenDecodeMegakernel()
    layers = kernel.layers
    rows = [
        [mk.DOWN, 0, 0, 0],
        [mk.QKV, 1, 0, 0],  # q head 0: kv group 0
        [mk.QKV, 1, 1, 0],  # also q head 0
        [mk.ATTN, 1, 0, 0],
        [mk.DOWN, layers - 1, 0, 0],
        [mk.LMHEAD, layers, 0, 8],
    ]
    table = torch.tensor([words + [i] for i, words in enumerate(rows)])
    done = [100, 150, 140, 200, 250, 300]
    events = [Range("compute", d - 10, d, (0,), payload=row) for row, d in enumerate(done)]
    events.append(Range("poll", 140, 144, (0,), payload=1))  # QKV row 1: ready at 144
    events.append(Range("poll", 144, 148, (0,), payload=1))  # a later poll does not count
    spans = mk_trace.phase_spans(events, table)
    assert spans == {"QKV": 0.05, "ATTN": 0.05, "LMHEAD": 0.05}
    assert mk_trace.input_waits(events, table) == {"QKV": 0.004}
    edges = mk_trace.input_edges(kernel, table.tolist(), dict(enumerate(done)))
    assert edges == {
        ("DOWN", "QKV"): [(0, 1), (0, 2)],
        ("QKV", "ATTN"): [(1, 3)],  # row 1 finished after row 2
        ("DOWN", "LMHEAD"): [(4, 5)],
    }


@pytest.mark.parametrize("warps", [range(4), [range(8), range(9, 17)]], ids=["4", "2x8"])
def test_compute_size_is_fixed(warps):
    with pytest.raises(ProtocolError, match="one instance of 8 warps"):
        mk.QwenDecodeMegakernel(roles={"compute": RoleConfig(warps)})


def test_tcgen05_engine_rejects_shapes_its_tiles_do_not_cover():
    """128-row tiles: a vocabulary the CUDA-core kernel covers (a multiple of 32) but not of 128
    is rejected by the tcgen05 kernel's constructor."""
    config = QwenConfig(
        hidden_size=1024, num_layers=1, num_q_heads=16, num_kv_heads=8, intermediate_size=3072,
        vocab_size=4128,
    )  # fmt: skip
    mk.QwenDecodeMegakernel(model=config)
    with pytest.raises(ValueError, match="vocabulary size 4128 must be a multiple of 128"):
        mk.QwenDecodeMegakernelTc(model=config)
