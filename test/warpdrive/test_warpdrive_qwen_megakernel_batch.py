"""Qwen3 decode megakernel, batched: several sequences per launch on the tcgen05 engine."""

import gc
import math

import pytest
import torch

pytest.importorskip("cutlass.cute")

from attn_gym.warpdrive.examples import qwen_megakernel as mk
from attn_gym.warpdrive.examples.qwen_megakernel_ref import (
    QwenConfig,
    compare_logits,
    decode_step_ref,
    forward_tokens,
    load_qwen3_weights,
    make_decode_state,
    random_weights,
)

MAX_LEN = 512
SMALL = QwenConfig(
    hidden_size=1024, num_layers=2, num_q_heads=16, num_kv_heads=8, intermediate_size=3072,
    vocab_size=4096,
)  # fmt: skip


@pytest.fixture(autouse=True)
def release_cuda_memory():
    yield
    gc.collect()
    torch.cuda.empty_cache()


def require_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the megakernel requires an SM100-family GPU")


def batch_state(weights, batch):
    """Sequences of different lengths (60 .. 459), each with its own cache."""
    states = [
        make_decode_state(weights, 60 + (37 * b) % 400, MAX_LEN, seed=b) for b in range(batch)
    ]
    k = torch.stack([s[0][0] for s in states])
    v = torch.stack([s[0][1] for s in states])
    tokens = torch.stack([s[1].reshape(()) for s in states])
    return k, v, tokens, [s[2] for s in states]


def check_batch(weights, batch, **config):
    """Each sequence's logits within the bf16 error budget of its own reference step, and its new
    K/V rows within twice eager's distance from the fp32 oracle's; earlier rows untouched (rows
    past the position are uninitialized)."""
    k, v, tokens, positions = batch_state(weights, batch)
    runner = mk.QwenMegakernel(weights, MAX_LEN, engine="tcgen05", batch=batch, **config)
    ours_k, ours_v = k.clone(), v.clone()
    logits = runner.decode(tokens, positions, ours_k, ours_v)
    assert logits.shape == (batch, weights.config.vocab_size)
    for b, pos in enumerate(positions):
        theirs = (k[b].clone(), v[b].clone())
        exact = (k[b].clone(), v[b].clone())
        ref = decode_step_ref(weights, tokens[b], pos, *theirs)
        high = forward_tokens(weights, tokens[b].reshape(1), pos, *exact, dtype=torch.float32)
        metrics = compare_logits(logits[b], ref, high)
        assert metrics["passed"], (b, metrics)
        for mine, other, oracle in zip((ours_k[b], ours_v[b]), theirs, exact):
            row, ref_row, exact_row = (c[:, :, pos].float() for c in (mine, other, oracle))
            assert (row - exact_row).norm() <= 2 * (ref_row - exact_row).norm() + 1e-3
        assert torch.equal(ours_k[b][:, :, :pos], k[b][:, :, :pos])
        assert torch.equal(ours_v[b][:, :, :pos], v[b][:, :, :pos])


@pytest.mark.parametrize("batch", [5, 37])
def test_batch_random_weights(batch):
    """5: fewer stored input rows than an MMA's 16 columns; 37: not a power of two, so the MMA
    runs 64 columns and the KV split count (16 // 37 -> 1) bottoms out."""
    require_sm100()
    check_batch(random_weights(SMALL), batch)


def test_batch_steps_atomic_schedule():
    """17 sequences (MMA width 32) under the atomic schedule, three steps each across the KV block
    boundary at position 128, with unused cache rows NaN-poisoned: each step's logits stay within
    the budget, only row pos changes, and the next step reads the reference's K/V rows."""
    require_sm100()
    weights = random_weights(SMALL)
    batch = 17
    states = [make_decode_state(weights, 125 + b % 3, MAX_LEN, seed=b) for b in range(batch)]
    ours_k = torch.stack([s[0][0] for s in states])
    ours_v = torch.stack([s[0][1] for s in states])
    tokens = torch.stack([s[1].reshape(()) for s in states])
    positions = [s[2] for s in states]
    for b, pos in enumerate(positions):
        ours_k[b][:, :, pos:] = float("nan")
        ours_v[b][:, :, pos:] = float("nan")
    theirs_k, theirs_v = ours_k.clone(), ours_v.clone()
    runner = mk.QwenMegakernel(weights, MAX_LEN, engine="tcgen05", batch=batch, schedule="atomic")
    for step in range(3):
        steps = [pos + step for pos in positions]
        before_k = ours_k.clone()
        logits = runner.decode(tokens, steps, ours_k, ours_v)
        for b, pos in enumerate(steps):
            exact = (theirs_k[b].clone(), theirs_v[b].clone())
            ref = decode_step_ref(weights, tokens[b], pos, theirs_k[b], theirs_v[b])
            high = forward_tokens(weights, tokens[b].reshape(1), pos, *exact, dtype=torch.float32)
            metrics = compare_logits(logits[b], ref, high)
            assert metrics["passed"], (step, b, metrics)
            keep = torch.ones(MAX_LEN, dtype=torch.bool, device="cuda")
            keep[pos] = False
            assert torch.equal(ours_k[b][:, :, keep].isnan(), before_k[b][:, :, keep].isnan())
            # The next step starts from the same K/V rows on both sides.
            ours_k[b][:, :, pos] = theirs_k[b][:, :, pos]
            ours_v[b][:, :, pos] = theirs_v[b][:, :, pos]
        tokens = logits.argmax(-1)
    assert runner.state.tolist() == [3, 0]


@pytest.mark.xdist_group("qwen-large-checkpoints")
def test_batch_caches_past_2_to_the_31_elements():
    """64 sequences of 16448 positions: the caches hold 2.2e9 elements each, so a 32-bit element
    offset of the new K/V row overflows. The last sequence's logits and new rows must equal a
    batch-one run of it (same split count, so bit for bit)."""
    require_sm100()
    weights = random_weights(SMALL)
    batch, max_len, pos = 64, 16448, 16400
    shape = (batch, SMALL.num_layers, SMALL.num_kv_heads, max_len, SMALL.head_dim)
    assert math.prod(shape) > 2**31
    torch.manual_seed(0)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    tokens = torch.arange(batch, device="cuda") * 17 % SMALL.vocab_size
    one = mk.QwenMegakernel(weights, max_len, engine="tcgen05", splits=1)
    last_k, last_v = k[-1].clone(), v[-1].clone()
    expected = one(weights, tokens[-1], pos, last_k, last_v)
    runner = mk.QwenMegakernel(weights, max_len, engine="tcgen05", batch=batch, splits=1)
    logits = runner.decode(tokens, [pos] * batch, k, v)
    assert torch.equal(logits[-1], expected)
    assert torch.equal(k[-1][:, :, pos], last_k[:, :, pos])
    assert torch.equal(v[-1][:, :, pos], last_v[:, :, pos])


def test_batch_rows_do_not_depend_on_the_batch():
    """A sequence's logits and K/V rows are bit-identical in a batch of 8 and of 3 (same splits):
    tokens share only the weight stream."""
    require_sm100()
    weights = random_weights(SMALL)
    k, v, tokens, positions = batch_state(weights, 8)
    outputs = []
    for batch in (8, 3):
        runner = mk.QwenMegakernel(weights, MAX_LEN, engine="tcgen05", batch=batch, splits=2)
        caches = (k[:batch].clone(), v[:batch].clone())
        outputs.append((runner.decode(tokens[:batch], positions[:batch], *caches), *caches))
    (big, big_k, big_v), (small, small_k, small_v) = outputs
    assert torch.equal(big[:3], small)
    for b, pos in enumerate(positions[:3]):  # rows past pos are uninitialized
        assert torch.equal(big_k[b][:, :, : pos + 1], small_k[b][:, :, : pos + 1])
        assert torch.equal(big_v[b][:, :, : pos + 1], small_v[b][:, :, : pos + 1])


def test_batch_checkpoint():
    """Qwen3-0.6B, 16 sequences. Skips unless the checkpoint is in the local HF cache."""
    require_sm100()
    hub = pytest.importorskip("huggingface_hub")
    path = hub.try_to_load_from_cache("Qwen/Qwen3-0.6B", "config.json")
    if not isinstance(path, str):
        pytest.skip("Qwen3-0.6B is not in the local HF cache")
    check_batch(load_qwen3_weights(path), 16)


def test_batch_needs_the_tcgen05_engine_and_a_dense_model():
    with pytest.raises(ValueError, match="batch must be in"):
        mk.QwenDecodeMegakernel(batch=2)
    moe = QwenConfig(
        hidden_size=1024, num_layers=2, num_q_heads=32, num_kv_heads=4, intermediate_size=1024,
        vocab_size=4096, num_experts=128, experts_per_token=8, moe_intermediate_size=256,
    )  # fmt: skip
    with pytest.raises(ValueError, match="dense models only"):
        mk.QwenDecodeMegakernelTc(model=moe, batch=2)
