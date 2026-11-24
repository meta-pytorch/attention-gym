"""Inkling 2 Mini's shipped DSpark drafter (``mtp.safetensors``) and greedy speculative decoding
over the oracle (``ref.forward_tokens``).

Torch only; no CuTeDSL. There is no HF transformers implementation: this follows vLLM (main
bf151b7e), ``vllm/models/inkling2/nvidia/dspark.py`` (model), ``.../nvidia/ops/dspark.py``
(rounding of the fused ops), ``vllm/models/inkling2/speculator.py`` and
``vllm/v1/worker/gpu/spec_decode/{dspark,dflash}/speculator.py`` (block layout, positions,
context K/V). vLLM's file:line references below are to that commit.

Drafting a block at position ``p`` (the anchor, the target's last sampled token, not yet run
through the target; positions ``< p`` are committed):

- Features: per committed position, the target's bf16 input of visits 5, 23 and 41 (the running
  partial block) and its final normalized state ("visit 48"), ``DSparkConfig.target_visits``.
  vLLM's ``target_layer_ids`` [5, 23, 41, 48] are layer ids of its non-looped Inkling2 model
  (``nvidia/model.py:239-241,304-306,434``: id ``num_hidden_layers`` is the final-norm
  output), whose check ``ids[-1] == num_hidden_layers`` (``configs.py:173``) the looped Mini
  (24 layers) would fail: the ids index the 48 visits (``ref.InklingConfig.visits``).
- Context K/V (``DSparkDrafter.commit``): draft layer ``i`` mixes a position's features with
  ``softmax(fuse_weight[i])`` and ``dflare_norm_i`` (``mix_rmsnorm``), projects K/V with
  ``k/v_proj_ctx``, normalizes K (``k_norm_ctx``) and ropes it at the position, into the draft
  cache at that position (``dspark.py:249-266``, ``dspark.py:517-530``).
- Block (``DSparkDrafter.block_states``): ``[embed_norm(embed(anchor)), mask_embed x 7]`` at
  ``p .. p + 7`` (``dspark.py:424-439``, ``dflash/speculator.py:903-938``) through 3 dense
  layers; each attends bidirectionally to the whole block and to the context, all of it (full
  layer 1) or the ``sliding_window`` positions before the block (local layers 0 and 2): vLLM's
  window is ``sliding_window + block`` keys up to the block's last position
  (``dspark.py:173``, ``ops/attention.py:913-926``). The block's own K/V go to the cache at
  ``p .. p + 7``; committed positions overwrite them later.
- Head (``DSparkDrafter.markov_drafts``): mask ``k`` (position ``p + k``) predicts the token at
  its own position (``configs.py:147-154``: ``sample_from_anchor`` False; ``sample_off`` 1 in
  ``dflash/speculator.py:946-949``). Base logits are a bf16 GEMM with the target's LM head; a
  rank-256 recurrent head adds a bias sequentially over the 7 drafts from the previous token
  (the anchor first) and ``f_rnn``, the mix (``fuse_weight[-1]``) of the last committed
  position's features (``inkling2/speculator.py:49-58,60-77``).

Verification (``speculative_generate``): the target runs ``[anchor, d1 .. d7]`` at
``p .. p + 7`` in one forward; the longest prefix of drafts equal to the target's argmax is
accepted, plus the target's next token (the bonus, the next anchor). Caches stay indexed by
position: rejected rows are rewritten by the next block, and the target's convolution rings
(``ref.CONV_RING``) rewind past at most 7 rejected positions, within their bound of
``CONV_RING - kernel + 1`` (13). The output is the target's greedy decode (in the rounding of an
8-token forward) whatever the draft proposes; draft numerics only move the acceptance, so they
follow vLLM's rounding points but not its kernels' summation orders.
"""

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import torch
import torch.nn.functional as F

from attn_gym.warpdrive.examples.inkling_megakernel.ref import (
    InklingCache,
    InklingWeights,
    Kernels,
    checkpoint_dir,
    experts_out,
    forward_tokens,
    load_inkling_weights,
    rms_norm,
)


@dataclass(frozen=True)
class DSparkConfig:
    """The drafter's shape (``config.json``'s ``mtp_config``); the defaults are Inkling 2
    Mini's. ``block_size`` tokens per draft: the anchor and ``block_size - 1`` masks."""

    hidden_size: int = 2304
    num_layers: int = 3
    num_q_heads: int = 24
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 9216
    rms_eps: float = 1e-6
    rope_theta: float = 1e6
    local_layers: tuple[int, ...] = (0, 2)
    sliding_window: int = 1024
    block_size: int = 8
    markov_rank: int = 256
    target_visits: tuple[int, ...] = (5, 23, 41, 48)  # 48: the final normalized state
    logit_softcap: float = 30.0

    @property
    def num_drafts(self) -> int:
        return self.block_size - 1

    def checked_drafts(self, num_drafts: int | None) -> int:
        """Drafts per block: ``num_drafts``, of at most the block's (None: all of them)."""
        drafts = self.num_drafts if num_drafts is None else num_drafts
        if not 1 <= drafts <= self.num_drafts:
            raise ValueError(f"num_drafts must be in 1 .. {self.num_drafts}, got {drafts}")
        return drafts

    @classmethod
    def from_hf(cls, config: dict) -> "DSparkConfig":
        """The shape of a checkpoint's ``config.json`` (``mtp_config``; the logit softcap of
        its ``text_config``)."""
        mtp = config["mtp_config"]
        text = config.get("text_config", config)
        if mtp.get("markov_head_type") != "rnn_hidden" or mtp.get("q_bias") or mtp.get("o_bias"):
            raise ValueError(f"unsupported mtp_config: {mtp}")
        return cls(
            hidden_size=mtp["hidden_size"],
            num_layers=mtp["num_hidden_layers"],
            num_q_heads=mtp["num_attention_heads"],
            num_kv_heads=mtp["num_key_value_heads"],
            head_dim=mtp["head_dim"],
            intermediate_size=mtp["intermediate_size"],
            rms_eps=mtp["rms_norm_eps"],
            rope_theta=mtp["rope_theta"],
            local_layers=tuple(mtp["local_layer_ids"]),
            sliding_window=mtp["sliding_window_size"],
            block_size=mtp["block_size"],
            markov_rank=mtp["markov_rank"],
            target_visits=tuple(mtp["target_layer_ids"]),
            logit_softcap=text["final_logit_softcapping"],
        )

    @classmethod
    def from_checkpoint(cls, checkpoint: Path) -> "DSparkConfig":
        """The shape in the checkpoint directory ``checkpoint``'s ``config.json``."""
        return cls.from_hf(json.loads((checkpoint / "config.json").read_text()))


@dataclass(frozen=True)
class DSparkLayerWeights:
    """One dense draft layer; projections (out, in) bf16, norms (dim,) bf16."""

    input_norm: torch.Tensor
    post_norm: torch.Tensor
    dflare_norm: torch.Tensor  # of the layer's feature mix
    wq: torch.Tensor
    wk: torch.Tensor
    wv: torch.Tensor
    wk_ctx: torch.Tensor  # context K/V from the mixed features
    wv_ctx: torch.Tensor
    wo: torch.Tensor
    q_norm: torch.Tensor  # (head_dim,)
    k_norm: torch.Tensor
    k_norm_ctx: torch.Tensor
    w_gate: torch.Tensor
    w_up: torch.Tensor
    w_down: torch.Tensor


@dataclass(frozen=True)
class DSparkWeights:
    """The drafter, sharing the target's ``embed``, ``embed_norm`` and ``lm_head`` tensors.

    ``fuse_alpha`` (layers + 1, taps) fp32 is ``softmax(fuse_weight)`` rounded to bf16: row ``i``
    mixes the features of draft layer ``i``, the last row those of the Markov head. ``fold_weight``
    (vocab, rank) and ``fold_bias`` (vocab,) bf16 fold ``hidden_out`` into the LM head in fp64
    (``dspark.py:347-361``). ``rnn_proj`` (3 rank, 2 rank + 2 hidden) takes
    ``[state, markov_w1[prev], hidden, f_rnn]``; ``rnn_bias`` is its bias in fp32."""

    layers: tuple[DSparkLayerWeights, ...]
    norm: torch.Tensor
    mask_embed: torch.Tensor  # (hidden,)
    fuse_alpha: torch.Tensor
    markov_norm: torch.Tensor  # the head's dflare_norm, of f_rnn
    markov_w1: torch.Tensor  # (padded vocab, rank)
    rnn_proj: torch.Tensor
    rnn_bias: torch.Tensor
    fold_weight: torch.Tensor
    fold_bias: torch.Tensor
    embed: torch.Tensor
    embed_norm: torch.Tensor
    lm_head: torch.Tensor  # (vocab, hidden), unpadded
    config: DSparkConfig = field(default_factory=DSparkConfig)


@dataclass
class DSparkCache:
    """Draft K (normalized, roped) and V at ``[layer, kv_head, pos]``, indexed by absolute
    position like ``ref.InklingCache``."""

    k: torch.Tensor  # (layers, kv_heads, max_len, head_dim) bf16
    v: torch.Tensor

    @property
    def max_len(self) -> int:
        return self.k.shape[2]

    @classmethod
    def empty(cls, config: DSparkConfig, max_len: int, device="cuda") -> "DSparkCache":
        shape = (config.num_layers, config.num_kv_heads, max_len, config.head_dim)
        return cls(
            k=torch.zeros(shape, device=device, dtype=torch.bfloat16),
            v=torch.zeros(shape, device=device, dtype=torch.bfloat16),
        )


# Checkpoint loading -------------------------------------------------------------------------


@torch.no_grad()
def fold_markov_head(
    lm_head: torch.Tensor, hidden_out: torch.Tensor, hidden_out_bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(lm_head @ hidden_out, lm_head @ hidden_out_bias)`` in fp64, rounded to the LM head's
    dtype, 4096 rows at a time (``dspark.py:347-361``)."""
    weight = lm_head.new_empty(lm_head.shape[0], hidden_out.shape[1])
    bias = lm_head.new_empty(lm_head.shape[0])
    projection, offset = hidden_out.double(), hidden_out_bias.double()
    for start in range(0, lm_head.shape[0], 4096):
        head = lm_head[start : start + 4096].double()
        weight[start : start + 4096] = head @ projection
        bias[start : start + 4096] = head @ offset
    return weight, bias


def random_weights(
    config: DSparkConfig, vocab: int, padded_vocab: int, seed: int = 0
) -> DSparkWeights:
    """Random drafter weights on the GPU (seeded on the CPU, so the same on every device), the
    target's shared embedding (``padded_vocab`` rows) and LM head (``vocab`` rows) included."""
    generator = torch.Generator().manual_seed(seed)
    h, d = config.hidden_size, config.head_dim

    def matrix(rows, cols):
        return (torch.randn(rows, cols, generator=generator) * cols**-0.5).cuda().bfloat16()

    def norm(dim):
        return (1 + 0.2 * torch.randn(dim, generator=generator)).cuda().bfloat16()

    def vector(dim, scale=1.0):
        return (scale * torch.randn(dim, generator=generator)).cuda()

    q_dim, kv_dim = config.num_q_heads * d, config.num_kv_heads * d
    layers = tuple(
        DSparkLayerWeights(
            input_norm=norm(h),
            post_norm=norm(h),
            dflare_norm=norm(h),
            wq=matrix(q_dim, h),
            wk=matrix(kv_dim, h),
            wv=matrix(kv_dim, h),
            wk_ctx=matrix(kv_dim, h),
            wv_ctx=matrix(kv_dim, h),
            wo=matrix(h, q_dim),
            q_norm=norm(d),
            k_norm=norm(d),
            k_norm_ctx=norm(d),
            w_gate=matrix(config.intermediate_size, h),
            w_up=matrix(config.intermediate_size, h),
            w_down=matrix(h, config.intermediate_size),
        )
        for _ in range(config.num_layers)
    )
    rank = config.markov_rank
    lm_head = matrix(vocab, h)
    fold_weight, fold_bias = fold_markov_head(lm_head, matrix(h, rank), vector(h, 0.1).bfloat16())
    taps = len(config.target_visits)
    fuse = torch.randn(config.num_layers + 1, taps, generator=generator).softmax(-1)
    return DSparkWeights(
        layers=layers,
        norm=norm(h),
        mask_embed=vector(h).bfloat16(),
        fuse_alpha=fuse.bfloat16().float().cuda(),
        markov_norm=norm(h),
        markov_w1=torch.randn(padded_vocab, rank, generator=generator).cuda().bfloat16(),
        rnn_proj=matrix(3 * rank, 2 * rank + 2 * h),
        rnn_bias=vector(3 * rank, 0.1),
        fold_weight=fold_weight,
        fold_bias=fold_bias,
        embed=vector(padded_vocab * h).bfloat16().view(padded_vocab, h),
        embed_norm=norm(h),
        lm_head=lm_head,
        config=config,
    )


def load_dspark_weights(
    target: InklingWeights,
    path_or_repo: str | Path = "thinkingmachines/Inkling-2-Mini",
    device="cuda",
) -> DSparkWeights:
    """The drafter of a checkpoint (``mtp.safetensors``, shape from ``config.json``) over the
    ``target`` it drafts for, whose embedding, embedding norm and LM head it shares."""
    from safetensors import safe_open  # optional: the drafter needs only torch

    checkpoint = checkpoint_dir(path_or_repo, drafter=True)
    config = DSparkConfig.from_checkpoint(checkpoint)
    handle = safe_open(str(checkpoint / "mtp.safetensors"), "pt", device=str(device))

    def get(name: str) -> torch.Tensor:
        return handle.get_tensor(name).to(torch.bfloat16).contiguous()

    layers = []
    for i in range(config.num_layers):
        p, a = f"layers.{i}.", f"layers.{i}.self_attn."
        layers.append(
            DSparkLayerWeights(
                input_norm=get(p + "input_layernorm.weight"),
                post_norm=get(p + "post_attention_layernorm.weight"),
                dflare_norm=get(p + "dflare_norm.weight"),
                wq=get(a + "q_proj.weight"),
                wk=get(a + "k_proj.weight"),
                wv=get(a + "v_proj.weight"),
                wk_ctx=get(a + "k_proj_ctx.weight"),
                wv_ctx=get(a + "v_proj_ctx.weight"),
                wo=get(a + "o_proj.weight"),
                q_norm=get(a + "q_norm.weight"),
                k_norm=get(a + "k_norm.weight"),
                k_norm_ctx=get(a + "k_norm_ctx.weight"),
                w_gate=get(p + "mlp.gate_proj.weight"),
                w_up=get(p + "mlp.up_proj.weight"),
                w_down=get(p + "mlp.down_proj.weight"),
            )
        )
    m = "markov_head."
    fold_weight, fold_bias = fold_markov_head(
        target.lm_head, get(m + "hidden_out.weight"), get(m + "hidden_out.bias")
    )
    return DSparkWeights(
        layers=tuple(layers),
        norm=get("norm.weight"),
        mask_embed=get("mask_embed"),
        # Computed in fp32 and stored in the model's bf16 (dspark.py:403,505-507).
        fuse_alpha=get("fuse_weight").float().softmax(-1).to(torch.bfloat16).float(),
        markov_norm=get(m + "dflare_norm.weight"),
        markov_w1=get(m + "markov_w1.weight"),
        rnn_proj=get(m + "rnn_proj.weight"),
        rnn_bias=get(m + "rnn_proj.bias").float(),
        fold_weight=fold_weight,
        fold_bias=fold_bias,
        embed=target.embed,
        embed_norm=target.embed_norm,
        lm_head=target.lm_head,
        config=config,
    )


# Draft forward ------------------------------------------------------------------------------


def mix_features(features: torch.Tensor, alpha: torch.Tensor, norm: torch.Tensor, eps: float):
    """RMSNorm (weight ``norm``) of the fp32 ``alpha``-weighted sum, tap by tap, of bf16
    (tokens, taps, hidden) ``features``; bf16 (``ops/dspark.py:10-60``)."""
    mixed = features[:, 0].float() * alpha[0]
    for tap in range(1, features.shape[1]):
        mixed = mixed + features[:, tap].float() * alpha[tap]
    return rms_norm(mixed, norm, eps).to(torch.bfloat16)


def add_rms_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float):
    """FlashInfer's ``fused_add_rmsnorm``: returns ``(norm(x + residual), bf16(x + residual))``,
    the norm taken of the unrounded fp32 sum (``dspark.py:59-68``)."""
    total = x.float() + residual.float()
    return rms_norm(total, weight, eps).to(torch.bfloat16), total.to(torch.bfloat16)


def rope_angles(config: DSparkConfig, positions: torch.Tensor) -> torch.Tensor:
    """fp32 (tokens, head_dim / 2) RoPE angles of device ``positions``."""
    d = config.head_dim
    exponents = torch.arange(0, d, 2, device=positions.device).float() / d
    return positions.float()[:, None] * (1.0 / config.rope_theta**exponents)


def rope(config: DSparkConfig, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Neox RoPE of bf16 (tokens, heads, head_dim) ``x`` in fp32, rounded to bf16
    (``ops/dspark.py:96-113``)."""
    angles = rope_angles(config, positions)
    cos, sin = angles.cos()[:, None], angles.sin()[:, None]
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(torch.bfloat16)


def head_norm_rope(config: DSparkConfig, x: torch.Tensor, norm: torch.Tensor, positions):
    """Per-head RMSNorm rounded to bf16, then RoPE: ``qk_norm_rope_`` (``ops/dspark.py:84-142``)
    on (tokens, heads * head_dim)."""
    x = rms_norm(x.view(x.shape[0], -1, config.head_dim), norm, config.rms_eps)
    return rope(config, x, positions)


def block_attention(
    config: DSparkConfig,
    q: torch.Tensor,
    cache: DSparkCache,
    layer: int,
    pos: int | torch.Tensor,
) -> torch.Tensor:
    """bf16 (block, heads * head_dim) attention of the block's queries (block, heads, head_dim)
    at ``pos ..`` over the cache keys up to its last position ``pos + block - 1``: all of them,
    or in a local layer the last ``sliding_window + block`` (``dspark.py:173``,
    ``ops/attention.py:913-926``: every query sits at the block's last position, so the block is
    bidirectional). fp32 scores and softmax; bf16 probabilities times V accumulated in fp32,
    divided by the fp32 sum (the FlashAttention rounding).

    A host ``pos`` reads only those keys; a one-element device ``pos`` reads the whole cache
    behind a mask (static shapes for CUDA graphs; sums of other lengths, so other roundings)."""
    count, heads, d = q.shape
    local = layer in config.local_layers
    if isinstance(pos, torch.Tensor):
        keys, values = cache.k[layer], cache.v[layer]
        end = pos.reshape(1).long() + count
        key_positions = torch.arange(keys.shape[1], device=q.device)
        hidden = key_positions >= end
        if local:
            hidden |= key_positions < end - config.sliding_window - count
    else:
        end = pos + count
        first = max(0, end - config.sliding_window - count) if local else 0
        keys, values = cache.k[layer, :, first:end], cache.v[layer, :, first:end]
        hidden = None
    group = heads // config.num_kv_heads
    q = q.transpose(0, 1).reshape(config.num_kv_heads, group * count, d)
    scores = (q.float() @ keys.float().transpose(1, 2)) * d**-0.5  # (kv, group * count, keys)
    if hidden is not None:
        scores = scores.masked_fill(hidden, float("-inf"))
    probs = (scores - scores.amax(-1, keepdim=True)).exp()
    out = torch.bmm(probs.to(torch.bfloat16), values, out_dtype=torch.float32)
    out = out / probs.sum(-1, keepdim=True)
    return out.view(heads, count, d).transpose(0, 1).reshape(count, heads * d).to(torch.bfloat16)


class Drafter(Protocol):
    """What ``speculative_generate`` drafts with: ``commit`` records the target's features of
    committed ``positions`` (consecutive, in order, the last one most recent); ``propose``
    returns ``num_drafts`` tokens for the positions after ``anchor`` at ``pos``. ``DSparkDrafter``
    also takes device tensors (``examples.megakernel.spec.DraftModel``)."""

    num_drafts: int

    def commit(self, features: torch.Tensor, positions: torch.Tensor) -> None: ...

    def propose(self, anchor: int, pos: int) -> torch.Tensor: ...


class DSparkDrafter:
    """The DSpark drafter with its position-indexed cache (``max_len`` positions) and the
    Markov head's context ``f_rnn`` of the last committed position. It proposes the first
    ``num_drafts`` (default: all ``block_size - 1``) drafts of a block; fewer cost fewer LM-head
    rows and Markov steps, and the drafts it does propose are the same.

    A ``static`` drafter attends over the whole cache behind a mask (``block_attention``) and
    takes anchors and positions as host ints or one-element device tensors without a host sync
    (CUDA-graph capturable); others take host ints and attend over only the visible keys. The
    two round the attention's sums differently. The cache holds ``block_size`` positions from
    each anchor, whatever ``num_drafts``."""

    def __init__(
        self,
        weights: DSparkWeights,
        max_len: int,
        num_drafts: int | None = None,
        static: bool = False,
    ):
        self.weights, self.config = weights, weights.config
        self.cache = DSparkCache.empty(self.config, max_len, device=weights.norm.device)
        self.num_drafts = self.config.checked_drafts(num_drafts)
        self.static = static
        # Updated in place (CUDA graphs read it); zeros and the zero cache: no context yet.
        self.f_rnn = weights.norm.new_zeros(self.config.hidden_size)

    def commit(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        last: torch.Tensor | None = None,
    ) -> None:
        """Stores context K/V of bf16 (tokens, taps, hidden) target ``features`` at
        ``positions`` and takes ``f_rnn`` from row ``last`` (a one-element device index;
        default the last row), the most recent committed position
        (``dspark.py:249-266,517-534``). Rows past ``last`` are stored too: the next block
        rewrites their positions before attending to them."""
        w, c = self.weights, self.config
        count = features.shape[0]
        for i, layer in enumerate(w.layers):
            context = mix_features(features, w.fuse_alpha[i], layer.dflare_norm, c.rms_eps)
            k = head_norm_rope(c, F.linear(context, layer.wk_ctx), layer.k_norm_ctx, positions)
            v = F.linear(context, layer.wv_ctx).view(count, c.num_kv_heads, c.head_dim)
            self.cache.k[i][:, positions] = k.transpose(0, 1)
            self.cache.v[i][:, positions] = v.transpose(0, 1)
        row = features[-1:] if last is None else features.index_select(0, last.reshape(1))
        self.f_rnn.copy_(mix_features(row, w.fuse_alpha[-1], w.markov_norm, c.rms_eps)[0])

    def block_states(self, anchor: int | torch.Tensor, pos: int | torch.Tensor) -> torch.Tensor:
        """The final normalized bf16 (block, hidden) states of ``[anchor, mask x (block_size -
        1)]`` at ``pos ..``; writes the block's K/V at those positions
        (``dspark.py:194-247,303-317,424-439``)."""
        w, c = self.weights, self.config
        device = w.norm.device
        positions = torch.as_tensor(pos, device=device).reshape(1).long()
        positions = positions + torch.arange(c.block_size, device=device)
        if self.static:
            pos = positions[:1]
        elif isinstance(pos, torch.Tensor):
            raise TypeError("device positions need a static drafter")
        anchor = torch.as_tensor(anchor, device=device).reshape(1)
        residual = torch.cat(
            [
                rms_norm(F.embedding(anchor, w.embed), w.embed_norm, c.rms_eps),
                w.mask_embed.expand(c.num_drafts, -1),
            ]
        )
        x = rms_norm(residual, w.layers[0].input_norm, c.rms_eps)
        for i, layer in enumerate(w.layers):
            if i > 0:
                x, residual = add_rms_norm(x, residual, layer.input_norm, c.rms_eps)
            q = head_norm_rope(c, F.linear(x, layer.wq), layer.q_norm, positions)
            k = head_norm_rope(c, F.linear(x, layer.wk), layer.k_norm, positions)
            v = F.linear(x, layer.wv).view(c.block_size, c.num_kv_heads, c.head_dim)
            self.cache.k[i][:, positions] = k.transpose(0, 1)
            self.cache.v[i][:, positions] = v.transpose(0, 1)
            attn = F.linear(block_attention(c, q, self.cache, i, pos), layer.wo)
            x, residual = add_rms_norm(attn, residual, layer.post_norm, c.rms_eps)
            gate, up = F.linear(x, layer.w_gate).float(), F.linear(x, layer.w_up).float()
            x = F.linear((gate * torch.sigmoid(gate) * up).to(torch.bfloat16), layer.w_down)
        return add_rms_norm(x, residual, w.norm, c.rms_eps)[0]

    def markov_drafts(self, hidden: torch.Tensor, anchor: int | torch.Tensor) -> torch.Tensor:
        """Greedy drafts (masks,) int64 from the first masks' bf16 (masks, hidden) states,
        sequentially through the recurrent head (``dspark.py:363-385,555-567``,
        ``inkling2/speculator.py:60-77``; rounding of ``ops/dspark.py:188-254``)."""
        w, c = self.weights, self.config
        rank = c.markov_rank
        base = F.linear(hidden, w.lm_head)  # bf16 GEMM (dspark.py:544-547)
        inputs = torch.cat([hidden, self.f_rnn.expand_as(hidden)], dim=-1)
        context = torch.addmm(
            w.rnn_bias, inputs, w.rnn_proj[:, 2 * rank :].T, out_dtype=torch.float32
        )
        state = hidden.new_zeros(rank)
        prev = torch.as_tensor(anchor, device=hidden.device).reshape(())
        drafts = []
        for i in range(hidden.shape[0]):
            step_in = torch.cat([state, w.markov_w1.index_select(0, prev.reshape(1))[0]])[None]
            raw = torch.addmm(
                context[i : i + 1], step_in, w.rnn_proj[:, : 2 * rank].T, out_dtype=torch.float32
            )[0]
            gate, cand, out = raw.to(torch.bfloat16).chunk(3)  # each bf16 op rounds
            gate = torch.sigmoid(gate)
            state = gate * state + (1 - gate) * torch.tanh(cand)
            bias = torch.addmm(w.fold_bias, torch.tanh(out)[None], w.fold_weight.T)[0]
            logits = (base[i].float() + bias.float()).to(torch.bfloat16).float()
            logits = torch.tanh(logits / c.logit_softcap) * c.logit_softcap  # add_softcap
            prev = logits.to(torch.bfloat16).nan_to_num(nan=-float("inf")).argmax()
            drafts.append(prev)
        return torch.stack(drafts)

    def propose(self, anchor: int | torch.Tensor, pos: int | torch.Tensor) -> torch.Tensor:
        """``num_drafts`` drafts after ``anchor`` at ``pos``; the block spans ``pos .. pos +
        block_size - 1``, inside the cache (unchecked for a device ``pos``)."""
        if not isinstance(pos, torch.Tensor) and pos + self.config.block_size > self.cache.max_len:
            raise ValueError(f"a block at {pos} exceeds max_len {self.cache.max_len}")
        states = self.block_states(anchor, pos)
        return self.markov_drafts(states[1 : 1 + self.num_drafts], anchor)


# Speculative decoding -----------------------------------------------------------------------


def target_forward(
    target: InklingWeights,
    tokens: torch.Tensor,
    pos: int,
    cache: InklingCache,
    visits: Iterable[int],
    all_logits: bool,
    kernels: Kernels | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``ref.forward_tokens``' logits and the bf16 (tokens, len(visits), hidden) inputs of
    ``visits`` (``target.config.num_visits``: the final normalized states)."""
    visits = tuple(visits)
    captured: dict[int, torch.Tensor] = {}

    def observe(visit: int, h: torch.Tensor):
        if visit in visits:
            captured[visit] = h

    logits = forward_tokens(
        target, tokens, pos, cache, all_logits=all_logits, kernels=kernels, observe=observe
    )
    return logits, torch.stack([captured[v] for v in visits], dim=1)


# The target's forward of tokens at host position pos into a cache: fp32 (tokens, vocab) logits
# and bf16 (tokens, taps, hidden) drafter features (``target_forward``'s).
Verify = Callable[[torch.Tensor, int, InklingCache], tuple[torch.Tensor, torch.Tensor]]
# The prompt's prefill into a fresh target cache of at least max_len positions, committing the
# drafter's context: (cache, the last token's fp32 logits).
Prefill = Callable[[torch.Tensor, int], tuple[InklingCache, torch.Tensor]]


@dataclass
class SpecResult:
    """``tokens``: the generated tokens. Per verify step: ``accepted`` drafts and ``matches``,
    whether each draft equals the target's argmax given the drafts before it."""

    tokens: list[int]
    accepted: list[int] = field(default_factory=list)
    matches: list[list[bool]] = field(default_factory=list)

    @property
    def tokens_per_step(self) -> float:
        """Mean tokens a verify step commits (accepted drafts plus the bonus token), before
        truncation at a stop token or ``max_new``."""
        return 1 + sum(self.accepted) / max(len(self.accepted), 1)


def spec_max_len(prompt_len: int, max_new: int, block_tokens: int) -> int:
    """The positions a cache of ``speculative_generate`` needs when a step writes
    ``block_tokens`` positions from its anchor, the last anchor being at most the ``max_new -
    1``-th generated token: the target's verify block, ``num_drafts + 1``; a ``DSparkDrafter``'s
    block, ``block_size`` whatever its ``num_drafts``."""
    return prompt_len + max_new + block_tokens - 1


@torch.no_grad()
def speculative_generate(
    target: InklingWeights,
    drafter: Drafter,
    prompt: torch.Tensor,
    max_new: int,
    stops: Iterable[int] = (),
    visits: Iterable[int] = DSparkConfig.target_visits,
    verify_chunk: int | None = None,
    kernels: Kernels | None = None,
    verify: Verify | None = None,
    prefill: Prefill | None = None,
) -> SpecResult:
    """Greedy decode of up to ``max_new`` tokens after ``prompt`` (ending at a token of
    ``stops``), verifying each of ``drafter``'s proposals with target forwards of
    ``verify_chunk`` tokens (None: the whole block in one forward). The drafter is fresh; the
    target's cache holds ``spec_max_len(prompt.numel(), max_new, drafter.num_drafts + 1)``
    positions and a ``DSparkDrafter``'s needs ``spec_max_len(..., block_size)``.

    The tokens are the target's greedy decode in the verify's rounding: ``verify_chunk=1``
    rounds as a one-token decode (``ref.forward_tokens``), so its tokens equal that greedy
    decode bit for bit; a block forward rounds differently (as HF's does) and may resolve a
    near tie the other way, after which the two decodes diverge. ``kernels``: the target's, as
    ``ref.forward_tokens``'. ``verify``: the verify forward (default: ``target_forward`` of
    ``visits`` with ``kernels``); ``prefill``: the prompt's (default: ``target_forward``'s
    into a cache of ``spec_max_len`` positions, its features committed to ``drafter``)."""
    if max_new < 1 or (verify_chunk is not None and verify_chunk < 1):
        raise ValueError(f"need max_new >= 1 and verify_chunk >= 1, got {max_new}, {verify_chunk}")
    stops, visits = set(stops), tuple(visits)
    if verify is None:

        def verify(tokens: torch.Tensor, pos: int, cache: InklingCache):
            return target_forward(target, tokens, pos, cache, visits, True, kernels)

    if prefill is None:

        def prefill(prompt: torch.Tensor, max_len: int):
            cache = InklingCache.empty(target.config, max_len, device=prompt.device)
            logits, features = target_forward(target, prompt, 0, cache, visits, False, kernels)
            drafter.commit(features, torch.arange(prompt.numel(), device=prompt.device))
            return cache, logits

    prompt_len = prompt.numel()
    cache, logits = prefill(prompt, spec_max_len(prompt_len, max_new, drafter.num_drafts + 1))
    anchor = int(logits.argmax())
    result = SpecResult(tokens=[anchor])
    pos = prompt_len  # the anchor's position
    while len(result.tokens) < max_new and anchor not in stops:
        drafts = drafter.propose(anchor, pos)
        block = torch.cat([prompt.new_tensor([anchor]), drafts.to(prompt.dtype)])
        chunk = verify_chunk or block.numel()
        parts = [
            verify(block[i : i + chunk], pos + i, cache) for i in range(0, block.numel(), chunk)
        ]
        logits, features = map(torch.cat, zip(*parts))
        verified = logits.argmax(-1)
        matches = (drafts == verified[:-1]).tolist()
        accepted = next((i for i, match in enumerate(matches) if not match), len(matches))
        result.accepted.append(accepted)
        result.matches.append(matches)
        drafter.commit(
            features[: accepted + 1], torch.arange(pos, pos + accepted + 1, device=prompt.device)
        )
        for token in verified[: accepted + 1].tolist():  # accepted drafts, then the bonus
            if len(result.tokens) == max_new or result.tokens[-1] in stops:
                break
            result.tokens.append(token)
        pos += accepted + 1
        anchor = result.tokens[-1]
    return result


# Acceptance measurement ---------------------------------------------------------------------

PROMPTS = {
    "code": "Write a Python function that merges overlapping intervals. Include a docstring, "
    "type hints and a few pytest tests.",
    "reasoning": "A train leaves city A at 9:00 at 80 km/h and another leaves city B, 300 km "
    "away, at 10:00 at 120 km/h toward A. When and where do they meet? Explain step by step.",
    "explain": "Explain how a transformer language model generates text, for a curious "
    "high-school student.",
    "story": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "travel": "Plan a five-day first trip to Japan, with a day-by-day itinerary and food to try.",
    "email": "Write a polite email to my landlord asking to fix a leaking kitchen faucet, and "
    "list what I should check myself first.",
}


def acceptance_summary(results: list[SpecResult], num_drafts: int) -> dict:
    """Pooled over the verify steps of ``results``: tokens per step, the distribution of
    accepted drafts, each draft position's agreement with the target's argmax (given the drafts
    before it) and its acceptance given the drafts before it were accepted."""
    accepted = [a for r in results for a in r.accepted]
    if not accepted:
        return {"steps": 0, "tokens": sum(len(r.tokens) for r in results)}
    matches = torch.tensor([m for r in results for m in r.matches], dtype=torch.float64)
    histogram = torch.bincount(torch.tensor(accepted), minlength=num_drafts + 1)
    reached = [sum(a >= k for a in accepted) for k in range(num_drafts)]
    return {
        "steps": len(accepted),
        "tokens": sum(len(r.tokens) for r in results),
        "tokens_per_step": 1 + sum(accepted) / len(accepted),
        "accepted_fraction": (histogram / len(accepted)).tolist(),
        "agreement": matches.mean(0).tolist(),
        "conditional_acceptance": [
            sum(a > k for a in accepted) / reached[k] if reached[k] else float("nan")
            for k in range(num_drafts)
        ],
    }


def main(
    max_new: int = 256,
    efforts: str = "low,high",
    json_out: Path | None = None,
    checkpoint: str = "thinkingmachines/Inkling-2-Mini",
):
    """Speculative greedy decode of ``PROMPTS`` (the chat template at each reasoning effort)
    over the oracle with the shipped drafter; prints acceptance and the routed-expert union of
    the target's 8-token verify forwards (unique experts per MLP visit) and of their first
    ``k`` tokens (what a ``k``-token verify of the same drafts would route).

        python -m attn_gym.warpdrive.examples.inkling_megakernel.dspark --json-out spec.json
    """
    from transformers import AutoTokenizer

    path = checkpoint_dir(checkpoint, drafter=True)  # one snapshot for target and drafter
    target = load_inkling_weights(path)
    draft = load_dspark_weights(target, path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    stops = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>")}
    stops = {s for s in stops if isinstance(s, int) and s >= 0}
    unions: list[list[int]] = []  # per verify MLP visit, of its first 1 .. block tokens
    block = draft.config.block_size
    no_union = [[float("nan")] * block]  # no verify step (max_new 1, or a stop first): NaNs

    def recording_experts(config, w, x, experts, weights):
        if x.shape[0] == block:
            unions.append([experts[:k].unique().numel() for k in range(1, x.shape[0] + 1)])
        return experts_out(config, w, x, experts, weights)

    kernels = Kernels(experts=recording_experts)
    runs, results = [], {}
    for effort in efforts.split(","):
        for name, content in PROMPTS.items():
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort=effort,
            )
            prompt = torch.tensor(tokenizer(text, add_special_tokens=False).input_ids).cuda()
            max_len = spec_max_len(prompt.numel(), max_new, draft.config.block_size)
            start = len(unions)
            result = speculative_generate(
                target, DSparkDrafter(draft, max_len), prompt, max_new, stops, kernels=kernels
            )
            results.setdefault(effort, []).append(result)
            union = torch.tensor(unions[start:] or no_union, dtype=torch.float64)
            runs.append(
                {
                    "prompt": name,
                    "effort": effort,
                    "prompt_tokens": prompt.numel(),
                    "accepted": result.accepted,
                    "matches": result.matches,
                    "text": tokenizer.decode(result.tokens),
                    "expert_union_mean": union[:, -1].mean().item(),
                    **acceptance_summary([result], draft.config.num_drafts),
                }
            )
            print(
                f"{name:9s} {effort:4s} prompt {prompt.numel():3d} tokens {len(result.tokens):3d}"
                f" steps {len(result.accepted):3d} tokens/step {result.tokens_per_step:.2f}"
                f" experts/visit {union[:, -1].mean():.1f}",
                flush=True,
            )
    results["all"] = [result for group in results.values() for result in group]
    summary = {}
    for group, members in results.items():
        summary[group] = acceptance_summary(members, draft.config.num_drafts)
        print(group, json.dumps(summary[group]))
    union = torch.tensor(unions or no_union, dtype=torch.float64)
    summary["expert_union"] = {
        "mean_by_tokens": union.mean(0).tolist(),
        "p10": union[:, -1].quantile(0.1).item(),
        "p90": union[:, -1].quantile(0.9).item(),
        "max": union[:, -1].max().item(),
    }
    print("experts per verify MLP visit", summary["expert_union"])
    if json_out is not None:
        json_out.write_text(json.dumps({"summary": summary, "runs": runs}, indent=1))


if __name__ == "__main__":
    import typer

    typer.run(main)
