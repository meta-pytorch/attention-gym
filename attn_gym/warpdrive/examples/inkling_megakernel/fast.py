"""A CUDA-graph capturable Inkling 2 Mini decode step: the oracle's dataflow (``ref``) with
device-side positions, static shapes and capturable kernels.

- Attention reads the whole cache (``max_len`` keys) behind its mask, so shapes never change.
- Experts run as two ``F.grouped_mm`` calls over the routed (token, expert) rows sorted by
  expert, reading only the routed experts' weights; no host sync (HF's ``grouped_mm`` path).
  Layers with declared numerics (``ref.ExpertQuant``) quantize the rows as
  ``ref.declared_experts_out`` does and need fp32 GEMM outputs, which ``grouped_mm`` does not
  return: each routed row multiplies a gathered copy of its expert's weights (``torch.bmm``,
  about three times the weight traffic of ``grouped_mm``). The quantizers' many small
  elementwise kernels make such a step launch-bound: 60 ms at context 1100 on GB300, against
  32 ms with HF's numerics. ``torch.compile`` (FMA contraction off) fused them 16x faster but
  changed NVFP4 bits, so they run eagerly under it (``quantize``).
- Short convolutions are the oracle's (they read the rings by device position).

Numerics differ from the oracle only in summation order (grouped GEMMs for attention and
experts, the top-k sum); the decode-step tests bound the difference.
"""

import torch
import torch.nn.functional as F

from attn_gym.warpdrive.examples.inkling_megakernel.act_quant import quantize_dequantize
from attn_gym.warpdrive.examples.inkling_megakernel.ref import (
    ExpertQuant,
    InklingCache,
    InklingConfig,
    InklingWeights,
    Kernels,
    LayerWeights,
    hidden_states,
    logits_of,
    require_dense_experts,
    swiglu,
)

# The declared quantizer, kept out of torch.compile's graphs: Inductor's fusion of its
# elementwise ops changes NVFP4 codes, which the ``compile`` chat backend must not.
quantize = torch.compiler.disable(quantize_dequantize)


def declared_mm(values, amax, weights, row_experts, quant: ExpertQuant) -> torch.Tensor:
    """``ref.scaled_mm`` of each row of quantized activations (block ``values`` and NVFP4
    ``amax``, ``act_quant.quantize_dequantize``) through its expert ``row_experts``' block
    values ``weights``: fp32 dot products times their NVFP4 row scales, rounded to bf16."""
    w = weights[row_experts].transpose(1, 2)  # a copy of each row's expert
    out = torch.bmm(values[:, None], w, out_dtype=torch.float32)[:, 0]
    if amax is not None:
        factors = quant.row_factors(weights.shape[0], values.device)
        out = out * ((amax * factors[row_experts]) * factors[-1])[:, None]
    return out.to(torch.bfloat16)


def grouped_experts(
    config: InklingConfig, w: LayerWeights, x: torch.Tensor, experts, weights
) -> torch.Tensor:
    """``ref.experts_out`` as grouped GEMMs over the routed rows (declared numerics: per-row
    GEMMs, ``declared_mm``)."""
    require_dense_experts(w)
    tokens, k = experts.shape
    flat = experts.reshape(-1)
    if w.declared_experts:  # x quantized once per token; the routing weight before the down
        values, amax = quantize(w.w13_quant.activation, x)
        token = torch.arange(tokens, device=x.device).repeat_interleave(k)
        amax = None if amax is None else amax[token]
        gate_up = declared_mm(values[token], amax, w.w13, flat, w.w13_quant)
        act = swiglu(config, gate_up, weights.reshape(-1, 1)).to(torch.bfloat16)
        values, amax = quantize(w.w2_quant.activation, act)  # per routed row
        y = declared_mm(values, amax, w.w2, flat, w.w2_quant).float()
        return y.view(tokens, k, -1).sum(1).to(torch.bfloat16)
    order = flat.argsort(stable=True)
    counts = torch.histc(flat.int(), bins=config.num_experts, min=0, max=config.num_experts - 1)
    offsets = counts.cumsum(0).int()
    gate_up = F.grouped_mm(x[order // k], w.w13.transpose(1, 2), offs=offsets)
    act = swiglu(config, gate_up).to(torch.bfloat16)
    y = F.grouped_mm(act, w.w2.transpose(1, 2), offs=offsets).float()
    y = y * weights.reshape(-1)[order, None]
    unsorted = torch.empty_like(y).index_copy_(0, order, y)
    return unsorted.view(tokens, k, -1).sum(1).to(torch.bfloat16)


FAST = Kernels(experts=grouped_experts)


def decode_step(
    weights: InklingWeights,
    token: torch.Tensor,
    pos: torch.Tensor,
    cache: InklingCache,
    kernels: Kernels = FAST,
) -> torch.Tensor:
    """One decode step of the one-element ``token`` at the one-element device position ``pos``;
    writes the caches there and returns fp32 logits ``(vocab,)``. No host sync as long as
    ``kernels`` sync nothing (``FAST``'s do not)."""
    positions = pos.reshape(1).long()
    h = hidden_states(weights, token.reshape(1), positions, cache, None, kernels)
    return logits_of(weights, h)[0]
