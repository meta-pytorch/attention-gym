"""Inkling 2 Mini's DSpark drafter and greedy speculative decoding over the oracle.

The tests run a random tiny drafter, loaded from its checkpoint naming, and the speculative loop
over the random tiny target of the oracle's HF parity tests (``inkling_tiny``; only those tests
need transformers' Inkling2Loop). ``test_warpdrive_inkling_spec`` runs the checkpoint.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from inkling_tiny import tiny_models

from attn_gym.warpdrive.examples.inkling_megakernel.dspark import (
    DSparkCache,
    DSparkConfig,
    DSparkDrafter,
    block_attention,
    load_dspark_weights,
    spec_max_len,
    speculative_generate,
)
from attn_gym.warpdrive.examples.inkling_megakernel.ref import InklingCache, forward_tokens

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.usefixtures("release_cuda_memory"),
]

HIDDEN, PADDED_VOCAB, VOCAB = 256, 512, 500  # the tiny target's
TINY_VISITS = (1, 5, 9, 16)  # of the tiny target's 16 visits; 16 is its final state
TINY_MTP = {
    "hidden_size": HIDDEN, "num_hidden_layers": 3, "num_attention_heads": 4,
    "num_key_value_heads": 2, "head_dim": 64, "intermediate_size": 512, "rms_norm_eps": 1e-6,
    "rope_theta": 1e6, "local_layer_ids": [0, 2], "sliding_window_size": 4, "block_size": 8,
    "markov_rank": 32, "markov_head_type": "rnn_hidden", "target_layer_ids": list(TINY_VISITS),
    "q_bias": False, "o_bias": False,
}  # fmt: skip
PROMPT = 24


def write_tiny_drafter(path: Path, mtp: dict, padded_vocab: int):
    """A random drafter in ``mtp.safetensors``' naming, and its ``config.json``; skips the
    test without safetensors."""
    save_file = pytest.importorskip("safetensors.torch").save_file

    hidden, rank = mtp["hidden_size"], mtp["markov_rank"]
    q_dim = mtp["num_attention_heads"] * mtp["head_dim"]
    kv_dim = mtp["num_key_value_heads"] * mtp["head_dim"]
    ffn, layers = mtp["intermediate_size"], mtp["num_hidden_layers"]
    generator = torch.Generator().manual_seed(0)

    def matrix(rows, cols):
        return torch.randn(rows, cols, generator=generator) * cols**-0.5

    def norm(dim):
        return 1 + 0.2 * torch.randn(dim, generator=generator)

    tensors = {
        "fuse_weight": torch.randn(layers + 1, len(mtp["target_layer_ids"]), generator=generator),
        "mask_embed": torch.randn(hidden, generator=generator),
        "norm.weight": norm(hidden),
        "markov_head.dflare_norm.weight": norm(hidden),
        "markov_head.hidden_out.weight": matrix(hidden, rank),
        "markov_head.hidden_out.bias": 0.1 * torch.randn(hidden, generator=generator),
        "markov_head.markov_w1.weight": torch.randn(padded_vocab, rank, generator=generator),
        "markov_head.rnn_proj.weight": matrix(3 * rank, 2 * rank + 2 * hidden),
        "markov_head.rnn_proj.bias": 0.1 * torch.randn(3 * rank, generator=generator),
    }
    for i in range(layers):
        p, a = f"layers.{i}.", f"layers.{i}.self_attn."
        tensors |= {
            p + "input_layernorm.weight": norm(hidden),
            p + "post_attention_layernorm.weight": norm(hidden),
            p + "dflare_norm.weight": norm(hidden),
            p + "mlp.gate_proj.weight": matrix(ffn, hidden),
            p + "mlp.up_proj.weight": matrix(ffn, hidden),
            p + "mlp.down_proj.weight": matrix(hidden, ffn),
            a + "q_proj.weight": matrix(q_dim, hidden),
            a + "k_proj.weight": matrix(kv_dim, hidden),
            a + "v_proj.weight": matrix(kv_dim, hidden),
            a + "k_proj_ctx.weight": matrix(kv_dim, hidden),
            a + "v_proj_ctx.weight": matrix(kv_dim, hidden),
            a + "o_proj.weight": matrix(hidden, q_dim),
            a + "q_norm.weight": norm(mtp["head_dim"]),
            a + "k_norm.weight": norm(mtp["head_dim"]),
            a + "k_norm_ctx.weight": norm(mtp["head_dim"]),
        }
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in tensors.items()},
              str(path / "mtp.safetensors"))  # fmt: skip
    config = {"mtp_config": mtp, "text_config": {"final_logit_softcapping": 30.0}}
    (path / "config.json").write_text(json.dumps(config))


@pytest.fixture(scope="module")
def tiny_drafter_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny_dspark")
    write_tiny_drafter(path, TINY_MTP, PADDED_VOCAB)
    return path


@pytest.fixture(scope="module")
def tiny_draft(tiny_drafter_dir):
    """The tiny drafter over a random stand-in for the target's shared tensors."""
    generator = torch.Generator().manual_seed(2)
    shared = SimpleNamespace(
        embed=torch.randn(PADDED_VOCAB, HIDDEN, generator=generator).to("cuda", torch.bfloat16),
        embed_norm=torch.ones(HIDDEN, device="cuda", dtype=torch.bfloat16),
        lm_head=(torch.randn(VOCAB, HIDDEN, generator=generator) * HIDDEN**-0.5).to(
            "cuda", torch.bfloat16
        ),
    )
    return load_dspark_weights(shared, tiny_drafter_dir)


def random_features(count: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(count, len(TINY_VISITS), HIDDEN, generator=generator)
    return features.to("cuda", torch.bfloat16)


def positions(start: int, stop: int) -> torch.Tensor:
    return torch.arange(start, stop, device="cuda")


@torch.no_grad()
def test_drafts_are_deterministic_tokens(tiny_draft):
    """The loader reads the checkpoint naming and config; a fresh drafter given the same
    context proposes the same ``block_size - 1`` in-vocabulary tokens, and the proposal reads
    the committed features (``f_rnn`` and context K/V)."""
    assert tiny_draft.config == DSparkConfig.from_hf(
        {"mtp_config": TINY_MTP, "text_config": {"final_logit_softcapping": 30.0}}
    )
    drafters = [DSparkDrafter(tiny_draft, max_len=64) for _ in range(3)]
    for drafter, seed in zip(drafters, (0, 0, 1)):
        drafter.commit(random_features(PROMPT, seed), positions(0, PROMPT))
    first, again = (drafter.propose(anchor=7, pos=PROMPT) for drafter in drafters[:2])
    assert first.shape == (7,) and first.dtype == torch.int64
    assert ((first >= 0) & (first < VOCAB)).all()
    assert torch.equal(first, again)
    seed0, seed1 = (drafter.block_states(7, PROMPT) for drafter in drafters[1:])
    assert not torch.equal(seed0, seed1)


@torch.no_grad()
def test_rejected_block_rows_do_not_leak(tiny_draft):
    """A rejected proposal's K/V rows are rewritten before they are read: proposing, then
    committing 3 of its positions, drafts as if the rejected block never ran."""
    features = random_features(PROMPT + 3, seed=0)
    rejected, fresh = (DSparkDrafter(tiny_draft, max_len=64) for _ in range(2))
    for drafter in (rejected, fresh):
        drafter.commit(features[:PROMPT], positions(0, PROMPT))
    rejected.propose(anchor=3, pos=PROMPT)
    for drafter in (rejected, fresh):
        drafter.commit(features[PROMPT:], positions(PROMPT, PROMPT + 3))
    torch.testing.assert_close(
        rejected.block_states(9, PROMPT + 3), fresh.block_states(9, PROMPT + 3), atol=0, rtol=0
    )


@pytest.mark.parametrize(
    ("local", "key", "visible"),
    [
        (True, 20 - 4 - 1, False),  # before the window: sliding_window (4) keys before the block
        (True, 20 - 4, True),
        (True, 28, False),  # past the block
        (False, 0, True),
    ],
)
@pytest.mark.parametrize("device_pos", [False, True], ids=["host_pos", "device_pos"])
@torch.no_grad()
def test_block_attention_span(tiny_draft, local, key, visible, device_pos):
    """Every query of a block at 20 .. 27 sees the whole block and the context before it: all
    of it in a full layer, ``sliding_window`` positions in a local one (vLLM's window of
    ``sliding_window + block`` keys up to the block's last position), whether it reads those
    keys (host position) or the whole cache behind a mask (device position)."""
    config = tiny_draft.config
    layer = 0 if local else 1
    assert (layer in config.local_layers) == local
    cache = DSparkCache.empty(config, max_len=32)
    generator = torch.Generator().manual_seed(3)
    cache.k.copy_(torch.randn(cache.k.shape, generator=generator))
    cache.v.copy_(torch.randn(cache.v.shape, generator=generator))
    q = torch.randn(8, config.num_q_heads, config.head_dim, generator=generator)
    q = q.to("cuda", torch.bfloat16)
    pos = torch.tensor([20], device="cuda") if device_pos else 20
    before = block_attention(config, q, cache, layer, pos)
    cache.k[layer, :, key] += 1
    cache.v[layer, :, key] += 1
    changed = (block_attention(config, q, cache, layer, pos) != before).any(-1)
    assert changed.all() if visible else not changed.any()


# The tiny target's logits move by up to ~0.3 between a one-token decode and an 8-token verify
# (HF's do as well: different GEMM shapes and attention spans round differently, and the random
# model amplifies it), enough to flip its narrow top-2 margins: its speculative tokens are
# checked against the one-token decode within this tolerance instead of bit for bit (they fall
# up to 0.07 below its top logit).
LOGIT_TOL = 0.25


class ScriptedDrafter:
    """Drafts that the target's verify accepts up to a scheduled count (7: all). It mirrors the
    loop's target cache by running each verify forward itself, so it knows the verify's argmaxes:
    it iterates the block to the drafts equal to them, then perturbs the scheduled one. Records
    the accepted counts that verify implies and the committed positions."""

    num_drafts = 7

    def __init__(self, weights, prompt: torch.Tensor, max_len: int, accept: list[int]):
        self.weights, self.accept = weights, iter(accept)
        self.cache = InklingCache.empty(weights.config, max_len)
        forward_tokens(weights, prompt, 0, self.cache)
        self.expected, self.committed = [], []

    def commit(self, features, positions):
        assert features.shape[0] == positions.numel()
        self.committed += positions.tolist()

    def verify(self, block: torch.Tensor, pos: int, cache: InklingCache) -> torch.Tensor:
        return forward_tokens(self.weights, block, pos, cache, all_logits=True).argmax(-1)

    def propose(self, anchor: int, pos: int) -> torch.Tensor:
        accept = next(self.accept)
        block = torch.full((8,), anchor, device="cuda")
        # Row k's argmax depends on the drafts before it only: after k rounds, the first k
        # drafts equal the verify's.
        for _ in range(min(accept + 1, 7)):
            scratch = InklingCache(*(t.clone() for t in vars(self.cache).values()))
            block[1:] = self.verify(block, pos, scratch)[:-1]
        if accept < 7:
            block[1 + accept] = (block[1 + accept] + 1) % VOCAB
        matches = (block[1:] == self.verify(block, pos, self.cache)[:-1]).tolist()
        self.expected.append(next((i for i, m in enumerate(matches) if not m), 7))
        return block[1:]


@torch.no_grad()
def assert_near_greedy(weights, prompt: torch.Tensor, tokens: list[int], tol: float):
    """Each of ``tokens`` is within ``tol`` of the top logit of the oracle's one-token decode of
    the tokens before it."""
    cache = InklingCache.empty(weights.config, prompt.numel() + len(tokens))
    logits = forward_tokens(weights, prompt, 0, cache)
    for i, token in enumerate(tokens):
        gap = (logits.max() - logits[token]).item()
        assert gap <= tol, f"token {i}: {gap} below the top logit"
        logits = forward_tokens(weights, prompt.new_tensor([token]), prompt.numel() + i, cache)


@pytest.fixture(scope="module")
def tiny_target(tmp_path_factory):
    """The tiny target's weights and a random prompt."""
    _, weights = tiny_models(tmp_path_factory.mktemp("tiny_inkling"))
    prompt = torch.randint(VOCAB, (PROMPT,), generator=torch.Generator().manual_seed(1))
    return weights, prompt.cuda()


@torch.no_grad()
def test_spec_loop_commits_verified_greedy_tokens(tiny_target):
    """The loop accepts the drafts the verify agrees with and commits them and the verify's next
    token, accepting k = 0 .. 7 drafts per step (rewinding the target's convolution rings past up
    to 7 rejected positions). The tokens are the target's greedy decode (up to its rounding at 8
    tokens per forward); committed features cover each position once, in order."""
    weights, prompt = tiny_target
    max_new = 40
    max_len = spec_max_len(PROMPT, max_new, 8)  # 8-token verify blocks and drafter blocks
    drafter = ScriptedDrafter(weights, prompt, max_len, [i % 8 for i in range(max_new)])
    result = speculative_generate(weights, drafter, prompt, max_new, visits=TINY_VISITS)
    assert len(result.tokens) == max_new
    assert_near_greedy(weights, prompt, result.tokens, LOGIT_TOL)
    assert result.accepted == drafter.expected
    assert set(result.accepted) == set(range(8))
    assert drafter.committed == list(range(PROMPT + sum(result.accepted) + len(result.accepted)))
