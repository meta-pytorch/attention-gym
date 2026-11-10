"""Batch-one Qwen3-0.6B decode oracle with HF's bf16 rounding boundaries.

Torch only; no CuTeDSL. Caches are (layer, kv_head, position, head_dim), keys stored after
head RMSNorm and rotate-half RoPE. Positions are Python ints (torch.compile specializes on
them); tokens are scalar tensors.
"""

from dataclasses import dataclass, fields, replace
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class QwenConfig:
    hidden_size: int = 1024
    num_layers: int = 28
    num_q_heads: int = 16
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 3072
    vocab_size: int = 151936
    rms_eps: float = 1e-6
    rope_theta: float = 1e6


@dataclass(frozen=True)
class QwenWeights:
    """Contiguous bf16 checkpoint tensors; layer tensors have a leading layer axis."""

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
    config: QwenConfig = QwenConfig()

    @property
    def lm_head(self) -> torch.Tensor:
        return self.embed

    def to(self, dtype: torch.dtype) -> "QwenWeights":
        return replace(
            self,
            **{
                field.name: getattr(self, field.name).to(dtype)
                for field in fields(self)
                if field.name != "config"
            },
        )


def load_qwen3_weights(
    path_or_repo: str = "Qwen/Qwen3-0.6B", device: str | torch.device = "cuda"
) -> QwenWeights:
    """Load the single-file Qwen3-0.6B safetensors checkpoint (a path or a Hub repo)."""
    # Optional dependencies: importing the oracle needs only torch.
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    checkpoint = Path(path_or_repo)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "model.safetensors"
    elif not checkpoint.is_file():
        checkpoint = Path(hf_hub_download(path_or_repo, "model.safetensors"))
    tensors = load_file(str(checkpoint), device=str(device))
    config = QwenConfig()
    layer_names = {
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
    stacked = {
        name: torch.stack(
            [
                tensors[f"model.layers.{layer}.{suffix}.weight"]
                for layer in range(config.num_layers)
            ]
        ).to(torch.bfloat16)
        for name, suffix in layer_names.items()
    }
    return QwenWeights(
        embed=tensors["model.embed_tokens.weight"].to(torch.bfloat16),
        final_norm=tensors["model.norm.weight"].to(torch.bfloat16),
        **stacked,
    )


CachePair = tuple[torch.Tensor, torch.Tensor]
DecodeState = tuple[CachePair, torch.Tensor, int]


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # HF rounds the normalized activation to x.dtype before the weight multiply.
    acc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    acc = x.to(acc_dtype)
    normalized = acc * torch.rsqrt(acc.square().mean(-1, keepdim=True) + eps)
    return weight * normalized.to(x.dtype)


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin


def forward_tokens(
    weights: QwenWeights,
    tokens: torch.Tensor,
    pos: int,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    dtype: torch.dtype,
    query_chunk: int = 256,
) -> torch.Tensor:
    """Prefill or decode; returns the last token's fp32 logits.

    ``dtype`` is the activation dtype; pass ``weights.to(dtype)``. Caches of that dtype give a
    fully upcast oracle; bf16 caches with upcast weights score one step on exactly the
    megakernel's quantized cache inputs.
    """
    config = weights.config
    count = tokens.numel()
    x = F.embedding(tokens.reshape(-1), weights.embed).to(dtype)
    positions = torch.arange(pos, pos + count, device=x.device, dtype=torch.float32)
    inv_freq = 1.0 / (
        config.rope_theta
        ** (torch.arange(0, config.head_dim, 2, device=x.device).float() / config.head_dim)
    )
    freqs = positions[:, None] * inv_freq[None, :]
    angles = torch.cat((freqs, freqs), dim=-1)
    cos, sin = angles.cos().to(dtype)[:, None, :], angles.sin().to(dtype)[:, None, :]
    for layer in range(config.num_layers):
        norm = rms_norm(x, weights.input_ln[layer], config.rms_eps)
        q = F.linear(norm, weights.q_proj[layer]).view(count, config.num_q_heads, config.head_dim)
        k = F.linear(norm, weights.k_proj[layer]).view(count, config.num_kv_heads, config.head_dim)
        v = F.linear(norm, weights.v_proj[layer]).view(count, config.num_kv_heads, config.head_dim)
        q = rope(rms_norm(q, weights.q_norm[layer], config.rms_eps), cos, sin).transpose(0, 1)
        k = rope(rms_norm(k, weights.k_norm[layer], config.rms_eps), cos, sin).transpose(0, 1)
        k_cache[layer, :, pos : pos + count].copy_(k)
        v_cache[layer, :, pos : pos + count].copy_(v.transpose(0, 1))
        keys = k_cache[layer, :, : pos + count].to(dtype).repeat_interleave(2, dim=0)
        values = v_cache[layer, :, : pos + count].to(dtype).repeat_interleave(2, dim=0)
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
        x = x + F.linear(attention, weights.o_proj[layer])
        norm = rms_norm(x, weights.post_ln[layer], config.rms_eps)
        mlp = F.silu(F.linear(norm, weights.gate[layer])) * F.linear(norm, weights.up[layer])
        x = x + F.linear(mlp, weights.down[layer])
    x = rms_norm(x[-1:], weights.final_norm, config.rms_eps)
    return F.linear(x, weights.lm_head).reshape(-1).float()


@torch.no_grad()
def decode_step_ref(
    weights: QwenWeights,
    token: torch.Tensor,
    pos: int,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """One decode step: writes K/V at ``pos`` in place (the caches must have room for it) and
    returns fp32 logits."""
    return forward_tokens(weights, token, pos, k_cache, v_cache, dtype=dtype)


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
    """Error-budget gate against the fp32 oracle: finite, same argmax as eager bf16, and max
    and RMS error both within ``error_factor`` x eager's plus an fp32 roundoff allowance.

    ``rel_err`` is max error over max reference magnitude, not a pointwise ratio. The fp32
    oracle must have run on the same token and cache inputs. An empirical gate, not a proof.
    """
    test, bf16, high = (x.double() for x in (test_logits, ref_bf16_logits, ref_fp32_logits))
    delta = (test - bf16).abs().max().item()
    test_error, eager_error = test - high, bf16 - high
    test_max, eager_max = test_error.abs().max().item(), eager_error.abs().max().item()
    test_rms, eager_rms = (
        test_error.square().mean().sqrt().item(),
        eager_error.square().mean().sqrt().item(),
    )
    argmax_match = bool(test.argmax() == bf16.argmax())
    finite = bool(torch.isfinite(test).all())
    allowance = torch.finfo(torch.float32).eps * max(1.0, high.abs().max().item())
    return {
        "max_abs_err": delta,
        "rel_err": delta / max(bf16.abs().max().item(), 1e-30),
        "argmax_match": argmax_match,
        "test_fp32_max_err": test_max,
        "bf16_fp32_max_err": eager_max,
        "test_fp32_rms_err": test_rms,
        "bf16_fp32_rms_err": eager_rms,
        "max_error_ratio": test_max / max(eager_max, allowance),
        "rms_error_ratio": test_rms / max(eager_rms, allowance),
        "passed": finite
        and argmax_match
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
