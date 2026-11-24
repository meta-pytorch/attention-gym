"""The DSpark drafter's megakernel launches (``draft``) against the torch drafter
(``dspark.DSparkDrafter``) on a reduced random drafter (hidden 768, 6 q / 2 kv heads of 128,
window 128, vocabulary 1000 + a ragged tail): the context launch's committed rows, each backbone
stage against its torch counterpart within bf16 rounding (teacher-forced on its own previous
stage), and the head's drafts. The checkpoint's drafter runs in
``test_warpdrive_inkling_spec``'s ``megakernel-spec`` chat test.
"""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cutlass.cute")

from attn_gym.warpdrive.examples.inkling_megakernel.draft import DSparkMegakernel
from attn_gym.warpdrive.examples.inkling_megakernel.dspark import (
    DSparkConfig,
    DSparkDrafter,
    block_attention,
    head_norm_rope,
    random_weights,
)
from attn_gym.warpdrive.examples.inkling_megakernel.ref import rms_norm

pytestmark = pytest.mark.usefixtures("requires_blackwell", "release_cuda_memory")

CONFIG = DSparkConfig(
    hidden_size=768,
    num_q_heads=6,
    num_kv_heads=2,
    intermediate_size=2048,
    sliding_window=128,
)
VOCAB, PADDED_VOCAB = 1000 + 7, 1024
MAX_LEN = 320


@pytest.fixture(scope="module")
def weights():
    return random_weights(CONFIG, VOCAB, PADDED_VOCAB)


def features(count: int, seed: int, config: DSparkConfig = CONFIG) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    taps = len(config.target_visits)
    return torch.randn(count, taps, config.hidden_size, generator=generator).cuda().bfloat16()


def device_int(value: int) -> torch.Tensor:
    return torch.tensor([value], device="cuda")


def ulps(actual: torch.Tensor, expected: torch.Tensor, floor: float) -> torch.Tensor:
    """|actual - expected| in bf16 ulps of ``max(|expected|, floor)``."""
    scale = expected.float().abs().clamp(min=floor)
    return (actual.float() - expected.float()).abs() / (scale.log2().floor().exp2() / 128)


def row_ulps(actual: torch.Tensor, expected: torch.Tensor, width: int) -> torch.Tensor:
    """|actual - expected| in bf16 ulps of the largest |expected| of each ``width`` elements
    (a head's dims or a token's row): values mixed from several bf16 inputs (RoPE, SiLU * up)
    inherit their inputs' rounding at that scale."""
    diff = (actual.float() - expected.float()).reshape(-1, width).abs()
    scale = expected.float().reshape(-1, width).abs().amax(-1, keepdim=True)
    return diff / (scale.log2().floor().exp2() / 128)


def assert_sum_close(actual, gemv, residual, layer):
    """A residual stream ``bf16 GEMV output + residual``: within a bf16 ulp of the GEMV output
    (its rounding) plus 2 fp32-sum ulps."""
    tol = gemv.abs().clamp(min=2**-4).log2().floor().exp2() / 128
    assert (
        (actual - (gemv + residual)).abs() <= tol + 2 * (gemv + residual).abs() * 2**-23
    ).all(), layer


def committed_pair(weights, count: int, seed: int = 0):
    """The torch drafter and the launches after committing the same ``count`` rows."""
    rows = features(count, seed)
    positions = torch.arange(count, device="cuda")
    reference = DSparkDrafter(weights, MAX_LEN)
    reference.commit(rows, positions)
    kernels = DSparkMegakernel(weights, MAX_LEN)
    kernels.commit_rows(rows, 0)
    return reference, kernels, rows


@torch.no_grad()
def test_commit_matches_torch(weights):
    """Context K/V of committed rows at their positions (a partial block after earlier commits)
    and the Markov head's f_rnn within bf16 rounding of ``DSparkDrafter.commit`` (a key's
    normalized, roped dims within 2 ulps of the head's largest), other positions untouched."""
    start, count = 61, 5
    reference, kernels, _ = committed_pair(weights, start)
    rows = features(count, seed=start + 1)
    positions = torch.arange(start, start + count, device="cuda")
    reference.commit(rows, positions)
    kernels.commit(rows, positions)
    for name, tol in (("k", 2), ("v", 1)):
        expected, actual = getattr(reference.cache, name), getattr(kernels.cache, name)
        span = slice(0, start + count)
        errors = row_ulps(actual[:, :, span], expected[:, :, span], CONFIG.head_dim)
        assert errors.max() <= tol, name
        assert (actual[:, :, span] == expected[:, :, span]).float().mean() > 0.99, name
        assert not actual[:, :, start + count :].any(), name
    assert ulps(kernels.f_rnn, reference.f_rnn, 0.25).max() <= 1


@torch.no_grad()
def test_backbone_stages_match_torch(weights):
    """Each backbone stage against its torch counterpart on the launch's own previous stage:
    the GEMV stages and norms within 2 bf16 ulps (residual sums within one of their GEMV
    output, SiLU * up within one of its row's largest), attention (flash partials, probabilities
    rounded to bf16 against another running max) within a bf16 ulp of its output scale, at a
    position past the local layers' window."""
    pos = 200
    reference, kernels, _ = committed_pair(weights, pos)
    c, w = CONFIG, weights
    anchor = 17
    hidden = kernels.block_states(device_int(anchor), device_int(pos))
    launch, tokens = kernels.backbone, c.block_size

    def stage(name, rows):
        return launch.values(name).view(rows, tokens, -1)

    qkv = launch.values("qkv").view(c.num_layers, launch.op.qkv_splits, tokens, -1).sum(1)
    attn_out = stage("attn_out", c.num_layers)
    h_mid, act = stage("h_mid", c.num_layers), stage("act", c.num_layers)
    h_pre = stage("h_pre", c.num_layers + 1)
    positions = torch.arange(pos, pos + tokens, device="cuda")
    embed = rms_norm(w.embed[anchor : anchor + 1], w.embed_norm, c.rms_eps)
    residual = torch.cat([embed, w.mask_embed.expand(tokens - 1, -1)])
    q_dim, kv_dim = c.num_q_heads * c.head_dim, c.num_kv_heads * c.head_dim
    for i, layer in enumerate(w.layers):
        if i > 0:
            residual = h_pre[i].bfloat16()
        x = rms_norm(residual if i == 0 else h_pre[i], layer.input_norm, c.rms_eps).bfloat16()
        expected = F.linear(x, torch.cat([layer.wq, layer.wk, layer.wv]))
        assert ulps(qkv[i], expected, 2**-4).max() <= 2, i
        proj = qkv[i].bfloat16()
        q = head_norm_rope(c, proj[:, :q_dim], layer.q_norm, positions)
        k = head_norm_rope(c, proj[:, q_dim : q_dim + kv_dim], layer.k_norm, positions)
        v = proj[:, q_dim + kv_dim :].reshape(tokens, c.num_kv_heads, c.head_dim)
        reference.cache.k[i][:, positions] = k.transpose(0, 1)
        reference.cache.v[i][:, positions] = v.transpose(0, 1)
        expected = block_attention(c, q, reference.cache, i, pos)
        scale = expected.float().abs().amax()
        assert ((attn_out[i] - expected.float()).abs() <= scale / 128).all(), i
        o = F.linear(attn_out[i].bfloat16(), layer.wo).float()
        assert_sum_close(h_mid[i], o, residual.float(), i)
        x = rms_norm(h_mid[i], layer.post_norm, c.rms_eps).bfloat16()
        gate, up = F.linear(x, layer.w_gate).float(), F.linear(x, layer.w_up).float()
        expected = (gate * torch.sigmoid(gate) * up).bfloat16()
        assert row_ulps(act[i], expected, c.intermediate_size).max() <= 1, i
        down = F.linear(act[i].bfloat16(), layer.w_down).float()
        assert_sum_close(h_pre[i + 1], down, h_mid[i].bfloat16().float(), i)
    expected = rms_norm(h_pre[-1], w.norm, c.rms_eps).bfloat16()
    assert ulps(hidden[: tokens - 1], expected[1:], 2**-4).max() <= 1


@torch.no_grad()
def test_head_matches_torch(weights):
    """The base logits and the sequential Markov head draft the torch drafter's first
    ``num_drafts`` tokens (3: the chat's default block of 4) from the same mask states (the
    vocabulary's last tile is ragged)."""
    num_drafts = 3
    reference, _, rows = committed_pair(weights, 24)
    kernels = DSparkMegakernel(weights, MAX_LEN, num_drafts=num_drafts)
    kernels.commit_rows(rows, 0)
    generator = torch.Generator().manual_seed(5)
    for trial in range(8):
        hidden = (2 * torch.randn(7, CONFIG.hidden_size, generator=generator)).cuda().bfloat16()
        anchor = (97 * trial) % VOCAB
        expected = reference.markov_drafts(hidden, anchor)[:num_drafts]
        kernels.hidden[:7] = hidden
        actual = kernels.markov(device_int(anchor))
        assert torch.equal(actual, expected), (trial, actual.tolist(), expected.tolist())
