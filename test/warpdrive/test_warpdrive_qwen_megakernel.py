"""Qwen3-0.6B decode megakernel: each instruction alone, then whole decode steps."""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cutlass.cute")

from attn_gym.testing.perfetto_trace import read_slices
from attn_gym.warpdrive import ProtocolError, RoleConfig
from attn_gym.warpdrive.examples import qwen_megakernel as mk
from attn_gym.warpdrive.examples import qwen_megakernel_trace as mk_trace
from attn_gym.warpdrive.examples.qwen_megakernel_ref import (
    compare_logits,
    decode_step_ref,
    forward_tokens,
    load_qwen3_weights,
    make_decode_state,
    rms_norm,
    rope,
)

MAX_LEN = 256


@pytest.fixture(scope="module")
def weights():
    """Load the checkpoint from the local HF cache; skip if it is missing, never download."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the megakernel requires an SM100-family GPU")
    hub = pytest.importorskip("huggingface_hub")
    pytest.importorskip("safetensors")
    path = hub.try_to_load_from_cache("Qwen/Qwen3-0.6B", "model.safetensors")
    if not isinstance(path, str):
        pytest.skip("Qwen3-0.6B is not in the local HF cache")
    return load_qwen3_weights(path)


def bf16_values(*shape):
    return torch.randn(*shape, device="cuda").bfloat16().float()


def rope_at(pos: int):
    table = mk.rope_table(pos + 1, "cuda")[pos]
    return table[0][None], table[1][None]


@pytest.mark.parametrize("op", ["qkv", "attn", "gateup", "down", "lmhead"])
def test_single_instruction(weights, op):
    """One instruction type over all layers, from random inputs in its scratch buffers."""
    torch.manual_seed(0)
    runner = mk.QwenMegakernel(weights, MAX_LEN, ops=(getattr(mk, op.upper()),))
    view = runner.values
    token, pos = 1234, 150  # three KV blocks, the last partial
    caches = [bf16_values(mk.LAYERS, mk.KV_HEADS, MAX_LEN, mk.HEAD_DIM).bfloat16() for _ in "kv"]
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
    assert runner.state.tolist() == [1, 0], "the last CTA advances the epoch"

    w, eps = weights, 1e-6
    for layer in range(mk.LAYERS):
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
            q = rope(rms_norm(qkv[:2048].view(16, 128), w.q_norm[layer], eps), cos, sin)
            k = rope(rms_norm(qkv[2048:3072].view(8, 128), w.k_norm[layer], eps), cos, sin)
            v = qkv[3072:].view(8, 128)
            torch.testing.assert_close(caches[0][layer, :, pos], k)
            torch.testing.assert_close(caches[1][layer, :, pos], v)
            keys = caches[0][layer, :, : pos + 1].float().repeat_interleave(2, 0)
            values = caches[1][layer, :, : pos + 1].float().repeat_interleave(2, 0)
            scores = (q[:, None].float() @ keys.transpose(-1, -2)).bfloat16() * 128**-0.5
            expected = (F.softmax(scores.float(), -1) @ values)[:, 0]
            o = view("attn_o")[layer]  # (kv head, split, head, dim)
            m, den = view("attn_ml")[layer].transpose(1, 2).unbind(-1)  # (kv head, split, head)
            scale = torch.exp2(m - m.amax(1, keepdim=True))
            out = (o * scale[..., None]).sum(1) / (den * scale).sum(1)[..., None]
            torch.testing.assert_close(out.reshape(16, 128), expected, atol=2e-3, rtol=1e-2)
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
        expected = F.linear(rms_norm(x, w.final_norm, eps), w.embed).float()
        torch.testing.assert_close(runner.logits, expected, atol=2e-2, rtol=1e-2)


def test_oproj_instruction(weights):
    """o_proj combines KV-split partials: split s carries weight 2^m_s and sum l_s."""
    torch.manual_seed(1)
    runner = mk.QwenMegakernel(weights, MAX_LEN, ops=(mk.OPROJ,))
    o = torch.randn(runner.values("attn_o").shape, device="cuda")  # (L, kv head, split, head, d)
    m = torch.randn(runner.values("attn_ml").shape[:-1], device="cuda") * 3  # (L, kvh, head, sp)
    den = torch.rand(m.shape, device="cuda") + 0.5
    runner.set_values("attn_o", o)
    runner.set_values("attn_ml", torch.stack((m, den), -1))
    m, den = m.transpose(2, 3), den.transpose(2, 3)  # (L, kv head, split, head) like o
    runner.step_info.copy_(torch.tensor([0, 100], dtype=torch.int32))
    cache_shape = (mk.LAYERS, mk.KV_HEADS, MAX_LEN, mk.HEAD_DIM)
    runner.launch(*[torch.zeros(cache_shape, device="cuda", dtype=torch.bfloat16) for _ in "kv"])
    torch.cuda.synchronize()
    scale = torch.exp2(m - m.amax(2, keepdim=True))
    attn = (o * scale[..., None]).sum(2) / (den * scale).sum(2)[..., None]
    attn = attn.reshape(mk.LAYERS, -1).bfloat16()
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
def test_decode_steps(weights, ctx, max_len, schedule, splits):
    """Test teacher-forced decode steps under both schedules against an fp32 oracle.

    Context 127 crosses a KV-block boundary at position 128. Context 1500 gives each split
    two blocks with a non-power-of-two split count. Logits and new K/V rows stay within the
    bf16 reference's error budget. Pick the oracle's token when its lead exceeds 0.5, write
    only cache row pos, mask NaN-poisoned unused rows, and repeat launches bit for bit.
    """
    runner = mk.QwenMegakernel(weights, max_len, schedule=schedule, splits=splits)
    weights32 = weights.to(torch.float32)
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
        high = forward_tokens(
            weights32, token.reshape(1), pos + step, *oracle, dtype=torch.float32
        )
        metrics = compare_logits(logits, ref, high)
        assert torch.isfinite(logits).all()
        # Single steps differ by bf16 noise accumulated over 28 layers; bound the max error
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
    assert runner.state.tolist() == [6, 0]  # one epoch per launch


def test_moved_roles_match_defaults(weights):
    """The roles index their own threads: moving compute off warp 0 changes no bit."""
    moved = {"load": RoleConfig(0), "compute": RoleConfig(range(1, 1 + mk.COMPUTE_WARPS))}
    caches, token, pos = make_decode_state(weights, 100, MAX_LEN, seed=0)
    ours, theirs = [c.clone() for c in caches], [c.clone() for c in caches]
    logits = mk.QwenMegakernel(weights, MAX_LEN, roles=moved)(weights, token, pos, *ours)
    expected = mk.QwenMegakernel(weights, MAX_LEN)(weights, token, pos, *theirs)
    assert torch.equal(logits, expected)
    for mine, other in zip(ours, theirs):  # rows past pos are uninitialized
        assert torch.equal(mine[:, :, : pos + 1], other[:, :, : pos + 1])


def test_globaltimer_trace(weights, tmp_path):
    """Tracing preserves logits and records every table row's load and compute ranges.

    Ranges carry row IDs, each phase has a span, and the trace links rows across CTAs.
    """
    caches, token, pos = make_decode_state(weights, 100, MAX_LEN, seed=0)
    ours, theirs = [c.clone() for c in caches], [c.clone() for c in caches]
    runner = mk.QwenMegakernel(weights, MAX_LEN, trace="globaltimer")
    with mk_trace.trace_session(runner, trace_path=str(tmp_path / "step.pftrace")) as session:
        logits = runner(weights, token, pos, *ours)
    expected = mk.QwenMegakernel(weights, MAX_LEN)(weights, token, pos, *theirs)
    assert torch.equal(logits, expected)
    rows = [r for r, words in enumerate(runner.table.tolist()) if words[0] >= 0]
    for role in ("compute", "load"):
        ranges = [e for e in session.ranges if e.name == role]
        assert sorted(e.payload for e in ranges) == rows
    computed = mk_trace.item_ranges(session.ranges)
    cta = {row: e.cta[0] for row, e in computed.items()}
    ctas = runner.max_ctas
    assert all(cta[row] == row % ctas for row in rows)  # the static [slot][cta] layout
    spans = mk_trace.phase_spans(session.ranges, runner.table)
    assert set(spans) == set(mk.OP_NAMES) and all(v > 0 for v in spans.values())
    assert set(mk_trace.input_waits(session.ranges, runner.table)) == set(mk.OP_NAMES)
    slices = read_slices(tmp_path / "step.pftrace")
    flows = [flow for item in slices for flow in item.flows_out]
    assert len(flows) >= len(rows) - mk.QKV_ROWS // mk.ROWS  # layer 0's QKV rows wait on nothing
    assert {item.name for item in slices if item.track == "compute"} >= set(mk.OP_NAMES)


def test_trace_analysis_on_synthetic_rows():
    """Phase spans run from the producer phase's last row (QKV: the previous layer's down; the
    LM head: the last layer's) to the phase's last row; input waits from a row's start to its
    first poll's end; arrows leave each input group's last-finishing producer row."""
    from attn_gym.warpdrive.profiler import Range

    rows = [
        [mk.DOWN, 0, 0, 0],
        [mk.QKV, 1, 0, 0],  # q head 0: kv group 0
        [mk.QKV, 1, 1, 0],  # also q head 0
        [mk.ATTN, 1, 0, 0],
        [mk.DOWN, mk.LAYERS - 1, 0, 0],
        [mk.LMHEAD, 0, 0, 8],
    ]
    table = torch.tensor([words + [i] for i, words in enumerate(rows)])
    done = [100, 150, 140, 200, 250, 300]
    events = [Range("compute", d - 10, d, (0,), payload=row) for row, d in enumerate(done)]
    events.append(Range("poll", 140, 144, (0,), payload=1))  # QKV row 1: ready at 144
    events.append(Range("poll", 144, 148, (0,), payload=1))  # a later poll does not count
    spans = mk_trace.phase_spans(events, table)
    assert spans == {"QKV": 0.05, "ATTN": 0.05, "LMHEAD": 0.05}
    assert mk_trace.input_waits(events, table) == {"QKV": 0.004}
    edges = mk_trace.input_edges(table.tolist(), dict(enumerate(done)))
    assert edges == {
        ("DOWN", "QKV"): [(0, 1), (0, 2)],
        ("QKV", "ATTN"): [(1, 3)],  # row 1 finished after row 2
        ("DOWN", "LMHEAD"): [(4, 5)],
    }


@pytest.mark.parametrize("warps", [range(4), [range(8), range(9, 17)]], ids=["4", "2x8"])
def test_compute_size_is_fixed(warps):
    with pytest.raises(ProtocolError, match="one instance of 8 warps"):
        mk.QwenDecodeMegakernel(roles={"compute": RoleConfig(warps)})
