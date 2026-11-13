"""Batch-one Qwen3 (dense and MoE) decode oracle with HF's bf16 rounding boundaries.

Torch only; no CuTeDSL. Caches are (layer, kv_head, position, head_dim), keys stored after
head RMSNorm and rotate-half RoPE. Positions are Python ints (torch.compile specializes on
them); tokens are scalar tensors.
"""

import json
from dataclasses import dataclass, fields
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class QwenConfig:
    """A Qwen3 shape; the defaults are Qwen3-0.6B's. ``num_experts > 0`` makes every layer's MLP a
    mixture of experts (Qwen3-MoE): ``experts_per_token`` of them, each ``moe_intermediate_size``
    wide, chosen per token by a softmax router, with top-k weights renormalized when
    ``norm_topk_prob``. ``intermediate_size`` is then unused."""

    hidden_size: int = 1024
    num_layers: int = 28
    num_q_heads: int = 16
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 3072
    vocab_size: int = 151936
    rms_eps: float = 1e-6
    rope_theta: float = 1e6
    num_experts: int = 0
    experts_per_token: int = 0
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = True

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    @classmethod
    def from_hf(cls, config: dict) -> "QwenConfig":
        """The shape of an HF ``config.json`` (``qwen3`` or ``qwen3_moe``)."""
        moe = {}
        # transformers 5 renames num_experts to num_local_experts.
        experts = config.get("num_experts") or config.get("num_local_experts")
        if experts:
            if config.get("decoder_sparse_step", 1) != 1 or config.get("mlp_only_layers"):
                raise ValueError(
                    "only Qwen3-MoE checkpoints with every layer sparse are supported"
                )
            moe = {
                "num_experts": experts,
                "experts_per_token": config["num_experts_per_tok"],
                "moe_intermediate_size": config["moe_intermediate_size"],
                "norm_topk_prob": config.get("norm_topk_prob", True),
            }
        return cls(
            hidden_size=config["hidden_size"],
            num_layers=config["num_hidden_layers"],
            num_q_heads=config["num_attention_heads"],
            num_kv_heads=config["num_key_value_heads"],
            head_dim=config["head_dim"],
            intermediate_size=config["intermediate_size"],
            vocab_size=config["vocab_size"],
            rms_eps=config["rms_norm_eps"],
            # transformers 5 saves it under rope_parameters.
            rope_theta=config.get("rope_theta") or config["rope_parameters"]["rope_theta"],
            **moe,
        )


# QwenWeights' per-layer fields and their checkpoint names, model.layers.<i>.<name>.weight. A MoE
# checkpoint names the MLP's per expert (EXPERT_TENSORS, expert e) and adds the router (mlp.gate).
LAYER_TENSORS = {
    "input_ln": "input_layernorm",
    "q_proj": "self_attn.q_proj",
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "q_norm": "self_attn.q_norm",
    "k_norm": "self_attn.k_norm",
    "o_proj": "self_attn.o_proj",
    "post_ln": "post_attention_layernorm",
    "gate": "mlp.gate_proj",
    "up": "mlp.up_proj",
    "down": "mlp.down_proj",
}
EXPERT_TENSORS = {
    "gate": "mlp.experts.{e}.gate_proj",
    "up": "mlp.experts.{e}.up_proj",
    "down": "mlp.experts.{e}.down_proj",
}


@dataclass(frozen=True)
class QwenWeights:
    """Contiguous bf16 checkpoint tensors; layer tensors have a leading layer axis. For a MoE
    model ``gate``/``up``/``down`` have an expert axis after it and ``router`` is (layers, experts,
    hidden)."""

    embed: torch.Tensor
    input_ln: torch.Tensor
    q_proj: torch.Tensor
    k_proj: torch.Tensor
    v_proj: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    o_proj: torch.Tensor
    post_ln: torch.Tensor
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    final_norm: torch.Tensor
    untied_lm_head: torch.Tensor | None = None  # None: the LM head is the embedding
    router: torch.Tensor | None = None  # MoE only
    config: QwenConfig = QwenConfig()

    @property
    def lm_head(self) -> torch.Tensor:
        return self.embed if self.untied_lm_head is None else self.untied_lm_head

    def tensors(self) -> dict[str, torch.Tensor]:
        """The checkpoint tensors by field name."""
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if isinstance(getattr(self, field.name), torch.Tensor)
        }


def load_qwen3_weights(
    path_or_repo: str = "Qwen/Qwen3-0.6B", device: str | torch.device = "cuda"
) -> QwenWeights:
    """Load a Qwen3 safetensors checkpoint, single-file or sharded, with its ``config.json``:
    a Hub repo, a checkpoint directory, or a file inside one."""
    # Optional dependencies: importing the oracle needs only torch.
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    checkpoint = Path(path_or_repo)
    if checkpoint.is_file():
        checkpoint = checkpoint.parent
    elif not checkpoint.is_dir():
        checkpoint = Path(
            snapshot_download(path_or_repo, allow_patterns=["*.json", "*.safetensors"])
        )
    tensors = {}
    for shard in sorted(checkpoint.glob("*.safetensors")):
        tensors.update(load_file(str(shard), device=str(device)))
    hf_config = json.loads((checkpoint / "config.json").read_text())
    config = QwenConfig.from_hf(hf_config)

    # Pop each layer's tensor as it is stacked: loading peaks near one model, not two.
    def stack(suffix: str) -> torch.Tensor:
        return torch.stack(
            [
                tensors.pop(f"model.layers.{layer}.{suffix}.weight")
                for layer in range(config.num_layers)
            ]
        ).to(torch.bfloat16)

    def stack_experts(suffix: str) -> torch.Tensor:
        experts = range(config.num_experts)
        return torch.stack(
            [
                torch.stack(
                    [
                        tensors.pop(f"model.layers.{layer}.{suffix.format(e=e)}.weight")
                        for e in experts
                    ]
                )
                for layer in range(config.num_layers)
            ]
        ).to(torch.bfloat16)

    stacked = {
        name: stack_experts(EXPERT_TENSORS[name])
        if config.is_moe and name in EXPERT_TENSORS
        else stack(suffix)
        for name, suffix in LAYER_TENSORS.items()
    }
    if config.is_moe:
        stacked["router"] = stack("mlp.gate")
    tied = hf_config.get("tie_word_embeddings", False)
    return QwenWeights(
        embed=tensors["model.embed_tokens.weight"].to(torch.bfloat16),
        final_norm=tensors["model.norm.weight"].to(torch.bfloat16),
        untied_lm_head=None if tied else tensors["lm_head.weight"].to(torch.bfloat16),
        config=config,
        **stacked,
    )


def random_weights(config: QwenConfig, seed: int = 0, device="cuda") -> QwenWeights:
    """bf16 weights of ``config``'s shapes: N(0, 1/fan_in) projections, norms near 1. MoE
    configs get per-expert MLPs, a router and an untied LM head."""
    generator = torch.Generator(device=device).manual_seed(seed)
    q, kv, d = config.num_q_heads, config.num_kv_heads, config.head_dim
    h, inter = config.hidden_size, config.intermediate_size
    experts = (config.num_experts,) if config.is_moe else ()
    if config.is_moe:
        inter = config.moe_intermediate_size
    shapes = {
        "input_ln": (h,), "post_ln": (h,), "q_norm": (d,), "k_norm": (d,),
        "q_proj": (q * d, h), "k_proj": (kv * d, h), "v_proj": (kv * d, h), "o_proj": (h, q * d),
        "gate": (*experts, inter, h), "up": (*experts, inter, h), "down": (*experts, h, inter),
    }  # fmt: skip
    assert set(shapes) == set(LAYER_TENSORS)
    if config.is_moe:
        shapes["router"] = (config.num_experts, h)

    def tensor(*shape, norm=False):
        x = torch.randn(shape, device=device, generator=generator)
        return (1 + 0.1 * x if norm else x * shape[-1] ** -0.5).bfloat16()

    layers = config.num_layers
    return QwenWeights(
        embed=torch.randn(config.vocab_size, h, device=device, generator=generator).bfloat16(),
        final_norm=tensor(h, norm=True),
        untied_lm_head=tensor(config.vocab_size, h) if config.is_moe else None,
        config=config,
        **{name: tensor(layers, *shape, norm=len(shape) == 1) for name, shape in shapes.items()},
    )


CachePair = tuple[torch.Tensor, torch.Tensor]
DecodeState = tuple[CachePair, torch.Tensor, int]


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # HF rounds the normalized activation to x.dtype before the weight multiply.
    acc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    acc = x.to(acc_dtype)
    normalized = acc * torch.rsqrt(acc.square().mean(-1, keepdim=True) + eps)
    return weight * normalized.to(x.dtype)


def swiglu(x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor):
    return F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)


def mlp(
    config: QwenConfig,
    w: dict[str, torch.Tensor],
    x: torch.Tensor,
    experts: torch.Tensor | None = None,
) -> torch.Tensor:
    """The layer's MLP on (tokens, hidden) ``x`` with one layer's weights ``w``. A MoE layer
    routes as HF's Qwen3MoeSparseMoeBlock: softmax in fp32, top-k (renormalized), weights cast to
    ``x.dtype``, and expert outputs added in ascending expert order in ``x.dtype``. ``experts``
    (tokens, k) replaces the top-k choice."""
    if not config.is_moe:
        return swiglu(x, w["gate"], w["up"], w["down"])
    softmax_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    probs = F.softmax(F.linear(x, w["router"]), dim=-1, dtype=softmax_dtype)
    # A stable descending sort breaks ties toward the lower expert index, as the megakernel
    # does; HF's topk leaves their order unspecified.
    if experts is None:
        weight, expert = probs.sort(dim=-1, descending=True, stable=True)
        weight, expert = (
            weight[:, : config.experts_per_token],
            expert[:, : config.experts_per_token],
        )
    else:
        expert = experts
        weight = probs.gather(-1, experts)
    if config.norm_topk_prob:
        weight = weight / weight.sum(dim=-1, keepdim=True)
    weight = weight.to(x.dtype)
    out = torch.zeros_like(x)
    for e in expert.unique().tolist():  # ascending
        token, slot = (expert == e).nonzero(as_tuple=True)
        y = swiglu(x[token], w["gate"][e], w["up"][e], w["down"][e])
        out.index_add_(0, token, y * weight[token, slot, None])
    return out


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin


def rope_angles(config: QwenConfig, positions: torch.Tensor) -> torch.Tensor:
    """fp32 rotate-half RoPE angles, (len(positions), head_dim), for fp32 ``positions``."""
    exponents = torch.arange(0, config.head_dim, 2, device=positions.device).float()
    inv_freq = 1.0 / (config.rope_theta ** (exponents / config.head_dim))
    freqs = positions[:, None] * inv_freq[None, :]
    return torch.cat((freqs, freqs), dim=-1)


def forward_tokens(
    weights: QwenWeights,
    tokens: torch.Tensor,
    pos: int,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    dtype: torch.dtype,
    query_chunk: int = 256,
    experts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Prefill or decode; returns the last token's fp32 logits.

    ``experts``: MoE only, (layers, tokens, experts_per_token) expert indices to use instead of
    the router's top-k (their weights still come from this run's router). Near-tied router
    logits make the top-k flip between implementations at bf16 noise; forcing one choice
    compares the rest of the numerics.

    ``dtype`` is the activation dtype; weights are upcast to it a layer at a time (exactly,
    from bf16). Caches of that dtype give a fully upcast oracle; bf16 caches score one step on
    exactly the megakernel's quantized cache inputs.
    """
    config = weights.config
    count = tokens.numel()
    x = F.embedding(tokens.reshape(-1), weights.embed).to(dtype)
    positions = torch.arange(pos, pos + count, device=x.device, dtype=torch.float32)
    angles = rope_angles(config, positions)
    cos, sin = angles.cos().to(dtype)[:, None, :], angles.sin().to(dtype)[:, None, :]
    group = config.num_q_heads // config.num_kv_heads  # q heads per kv head
    for layer in range(config.num_layers):
        # Upcast one layer at a time: an fp32 oracle of a 32B model fits beside its bf16 weights.
        names = [*LAYER_TENSORS, "router"] if config.is_moe else LAYER_TENSORS
        w = {name: getattr(weights, name)[layer].to(dtype) for name in names}
        norm = rms_norm(x, w["input_ln"], config.rms_eps)
        q = F.linear(norm, w["q_proj"]).view(count, config.num_q_heads, config.head_dim)
        k = F.linear(norm, w["k_proj"]).view(count, config.num_kv_heads, config.head_dim)
        v = F.linear(norm, w["v_proj"]).view(count, config.num_kv_heads, config.head_dim)
        q = rope(rms_norm(q, w["q_norm"], config.rms_eps), cos, sin).transpose(0, 1)
        k = rope(rms_norm(k, w["k_norm"], config.rms_eps), cos, sin).transpose(0, 1)
        k_cache[layer, :, pos : pos + count].copy_(k)
        v_cache[layer, :, pos : pos + count].copy_(v.transpose(0, 1))
        keys = k_cache[layer, :, : pos + count].to(dtype).repeat_interleave(group, dim=0)
        values = v_cache[layer, :, : pos + count].to(dtype).repeat_interleave(group, dim=0)
        outputs = []
        for start in range(0, count, query_chunk):
            end = min(start + query_chunk, count)
            scores = (q[:, start:end] @ keys.transpose(-1, -2)) * config.head_dim**-0.5
            if count > 1:
                key_positions = torch.arange(pos + count, device=x.device)
                mask = key_positions[None, :] > positions[start:end, None]
                scores = scores.masked_fill(mask[None, :, :], torch.finfo(dtype).min)
            softmax_dtype = torch.float64 if dtype == torch.float64 else torch.float32
            probs = F.softmax(scores, dim=-1, dtype=softmax_dtype).to(dtype)
            outputs.append(probs @ values)
        attention = torch.cat(outputs, dim=1).transpose(0, 1).reshape(count, -1)
        x = x + F.linear(attention, w["o_proj"])
        norm = rms_norm(x, w["post_ln"], config.rms_eps)
        x = x + mlp(config, w, norm, None if experts is None else experts[layer])
    x = rms_norm(x[-1:], weights.final_norm.to(dtype), config.rms_eps)
    return F.linear(x, weights.lm_head.to(dtype)).reshape(-1).float()


@torch.no_grad()
def decode_step_ref(
    weights: QwenWeights,
    token: torch.Tensor,
    pos: int,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    experts: torch.Tensor | None = None,
) -> torch.Tensor:
    """One decode step: writes K/V at ``pos`` in place (the caches must have room for it) and
    returns fp32 logits. ``experts``: as for ``forward_tokens``."""
    return forward_tokens(weights, token, pos, k_cache, v_cache, dtype=dtype, experts=experts)


def prompt_tokens(weights: QwenWeights, ctx_len: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(weights.config.vocab_size, (ctx_len,), generator=generator).to(
        weights.embed.device
    )


@torch.no_grad()
def make_decode_state(
    weights: QwenWeights, ctx_len: int, max_len: int, seed: int = 0
) -> DecodeState:
    if not 0 < ctx_len < max_len:
        raise ValueError("Require 0 < ctx_len < max_len (space for at least one decode token)")
    config = weights.config
    shape = (config.num_layers, config.num_kv_heads, max_len, config.head_dim)
    caches = tuple(
        torch.empty(shape, device=weights.embed.device, dtype=weights.embed.dtype)
        for _ in range(2)
    )
    logits = forward_tokens(
        weights, prompt_tokens(weights, ctx_len, seed), 0, *caches, dtype=weights.embed.dtype
    )
    return caches, logits.argmax(), ctx_len


def compare_logits(
    test_logits: torch.Tensor,
    ref_bf16_logits: torch.Tensor,
    ref_fp32_logits: torch.Tensor,
    *,
    error_factor: float = 2.0,
) -> dict[str, float | bool]:
    """Error-budget gate against the fp32 oracle: finite, max and RMS error both within
    ``error_factor`` x eager bf16's plus an fp32 roundoff allowance, and a token the oracle
    scores within that max-error budget of eager's (``argmax_ok``). Eager's argmax itself
    (``argmax_match``) can differ at a near-tie: two bf16 implementations then pick either token.

    ``rel_err`` is max error over max reference magnitude, not a pointwise ratio. The fp32
    oracle must have run on the same token and cache inputs. An empirical gate, not a proof.
    """
    test, bf16, high = (x.double() for x in (test_logits, ref_bf16_logits, ref_fp32_logits))
    allowance = torch.finfo(torch.float32).eps * max(1.0, high.abs().max().item())
    delta = (test - bf16).abs().max().item()
    test_error, eager_error = test - high, bf16 - high
    test_max, eager_max = test_error.abs().max().item(), eager_error.abs().max().item()
    test_rms, eager_rms = (
        test_error.square().mean().sqrt().item(),
        eager_error.square().mean().sqrt().item(),
    )
    argmax_match = bool(test.argmax() == bf16.argmax())
    budget = error_factor * eager_max
    argmax_ok = bool(high[test.argmax()] >= high[bf16.argmax()] - budget)
    finite = bool(torch.isfinite(test).all())
    return {
        "max_abs_err": delta,
        "rel_err": delta / max(bf16.abs().max().item(), 1e-30),
        "argmax_match": argmax_match,
        "argmax_ok": argmax_ok,
        "test_fp32_max_err": test_max,
        "bf16_fp32_max_err": eager_max,
        "test_fp32_rms_err": test_rms,
        "bf16_fp32_rms_err": eager_rms,
        "max_error_ratio": test_max / max(eager_max, allowance),
        "rms_error_ratio": test_rms / max(eager_rms, allowance),
        "passed": finite
        and argmax_ok
        and test_max <= error_factor * eager_max + allowance
        and test_rms <= error_factor * eager_rms + allowance,
    }


@torch.no_grad()
def validate_against_hf(
    weights: QwenWeights,
    path_or_repo: str = "Qwen/Qwen3-0.6B",
    *,
    ctx_len: int = 64,
    steps: int = 16,
    seed: int = 0,
) -> dict[str, float | bool]:
    from transformers import Qwen3ForCausalLM

    model = (
        Qwen3ForCausalLM.from_pretrained(
            path_or_repo, dtype=torch.bfloat16, local_files_only=True, attn_implementation="eager"
        )
        .to(weights.embed.device)
        .eval()
    )
    caches, token, pos = make_decode_state(weights, ctx_len, ctx_len + steps + 1, seed)
    output = model(prompt_tokens(weights, ctx_len, seed)[None], use_cache=True, logits_to_keep=1)
    hf_token = output.logits[0, -1].argmax()
    matches = bool(token == hf_token)
    max_error = 0.0
    for offset in range(steps):
        logits = decode_step_ref(weights, token, pos + offset, *caches)
        output = model(
            hf_token.reshape(1, 1), past_key_values=output.past_key_values, use_cache=True
        )
        expected = output.logits[0, -1].float()
        max_error = max(max_error, (logits - expected).abs().max().item())
        # One bf16 ULP at logit magnitude, plus headroom for GEMM summation order.
        torch.testing.assert_close(logits, expected, atol=0.125, rtol=0.02)
        token, hf_token = logits.argmax(), expected.argmax()
        matches = matches and bool(token == hf_token)
    if not matches:
        raise AssertionError("HF and reference greedy tokens differ")
    return {"hf_max_abs_err": max_error, "greedy_match": matches}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--ctx-len", type=int, default=64)
    parser.add_argument("--steps", type=int, default=16)
    args = parser.parse_args()
    print(
        validate_against_hf(
            load_qwen3_weights(args.model), args.model, ctx_len=args.ctx_len, steps=args.steps
        )
    )
