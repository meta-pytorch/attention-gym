"""Inkling 2 Mini (Thinking Machines, ``Inkling2LoopForConditionalGeneration``) text decoder
oracle with HF's dtype boundaries, for prefill and batch-one decode.

Torch only; no CuTeDSL. The model runs 24 decoder layers as 48 *visits*: layers 0-5 once
(prelude), layers 6-17 ``num_passes`` times with shared weights (core), layers 18-23 once
(coda). Every core pass restarts from the prelude's output; passes after the first read four
fp32 *messages* per core layer from the previous pass and merge them into the attention and MLP
inputs through rank-``gate_rank`` controllers.

Each visit owns its caches (``InklingCache``): keys and values at
``(visit, kv_head, position, head_dim)``, and the inputs of its four short convolutions in rings
of ``CONV_RING`` positions. Both are indexed by absolute position. Rewinding to an earlier
length L needs no cache surgery while the rings still hold positions L - kernel + 1 .. L - 1,
i.e. when at most ``CONV_RING - kernel + 1`` positions past L were written: the chat's decode
over-run past a stop token stays within that, and it rebuilds from position 0 otherwise.

The numerics follow ``transformers``' ``modeling_inkling2_loop`` (branch ``add-inkling2-mini``):
bf16 projections, fp32 block attention residuals, message gates, routing and short
convolutions, and bf16 attention scores with an fp32 softmax that includes a per-head sink
logit, except for the experts of a quantized checkpoint (below).

Experts. The checkpoint stores them as NVFP4 (layers 0-19) and MXFP8 (layers 20-23) and its
``hf_quant_config.json`` declares the quantization of their inputs. ``load_inkling_weights``
follows the declaration by default (``activation_quant``): such a layer carries ``ExpertQuant``
numerics per matrix and ``experts_out`` computes as vLLM's SM100 kernels (main bf151b7e,
``vllm/models/inkling2/nvidia/``; file:line references are to that commit):

- The bf16 expert input is quantized once per token (``act_quant``: NVFP4 per-token amax and
  16-column blocks, or MXFP8 32-column blocks; ``ops/moe.py:307-343``). Each gate/up row is
  ``bf16(dot * row_scale)``: an fp32 dot product of the activations' and weights' block values
  (exact bf16 numbers: ``e2m1 x e4m3`` and ``e4m3 x 2^k``, the NVFP4 weights *without* their
  ``amax / denominator``), times, for NVFP4, ``row_scale = (x_amax * w_amax[e]) * rcp(1536 *
  w_denominator)`` (``ops/moe.py:346-353``, ``ops/trtllm_bmm.cu:77``); MXFP8 GEMMs have no
  output scale (``ones``, ``nvidia/moe.py:141-143``).
- The routing weight multiplies the softcapped SwiGLU before its bf16 rounding,
  ``bf16(((capg(g) * sigmoid(g)) * capu(u)) * weight)`` (``ops/activation.py:78``); that row is
  quantized (NVFP4 with its own amax, ``ops/activation.py:166-173``) as the down projection's
  input, whose output is ``bf16(dot * row_scale)`` with ``w2``'s scale parts.
- A token's down rows (bf16) add up in fp32, rounded once to bf16 (``ops/moe.py:1058-1092``).

HF (and this oracle with ``activation_quant=False``) instead dequantizes the expert weights to
bf16 with their NVFP4 factor folded in, keeps activations bf16, and applies the routing weight
to the fp32 down output. The oracle keeps HF's transcendental functions and softcap division
where vLLM's kernel uses ``tanh.approx``, ``ex2.approx`` and ``rcp.approx``
(``ops/activation.py:19-58``), sums experts in ascending order rather than vLLM's slot order,
and its fp32 dot products' order is cuBLAS's.

Parity: with ``activation_quant=False`` and the same weights, ``forward_tokens`` equals HF's
eager model (eager attention, ``experts_implementation="eager"``) bit for bit on the tiny test
model and on short real prompts, prefill and decode; long prefills match to rounding noise (GEMM
heuristics change with the row count). The weights differ in one place: HF's loader rounds the
NVFP4 ``amax`` and ``outer_scale_denominator`` to bf16 before dequantizing; this loader keeps
the checkpoint's fp32 values (a ~0.4% scale change on some experts).
"""

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from attn_gym.warpdrive.examples.inkling_megakernel.act_quant import (
    NVFP4_GLOBAL,
    ActivationQuant,
    declared_activation_quant,
    quantize_dequantize,
    ue8m0_scale,
)
from attn_gym.warpdrive.examples.megakernel.formats import NVFP4

# Ring length of the convolution input history. Larger than the 3 earlier positions a kernel-4
# convolution reads plus the decode steps a CUDA graph may run past a stop token (chat's
# SYNC_EVERY), so those over-run steps never overwrite a position the next prefill reads.
CONV_RING = 16

E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


@dataclass(frozen=True)
class InklingConfig:
    """The text decoder's shape; the defaults are Inkling 2 Mini's. Every attention layer has
    ``num_q_heads`` query and ``num_kv_heads`` key/value heads of ``head_dim``; sliding layers see
    ``sliding_window`` keys and full layers all keys, both with a learned relative bias over the
    last ``rel_extent`` (sliding: ``sliding_window``) positions."""

    hidden_size: int = 2304
    num_layers: int = 24
    num_q_heads: int = 24
    num_kv_heads: int = 8
    head_dim: int = 128
    d_rel: int = 16
    rel_extent: int = 1024
    sliding_window: int = 1024
    full_layers: tuple[int, ...] = (5, 11, 17, 23)
    num_experts: int = 128
    experts_per_token: int = 8
    moe_intermediate_size: int = 1152
    vocab_size: int = 200058  # the unpadded LM head
    rms_eps: float = 1e-6
    qk_clip: float = 22.627416997969522
    rel_clip: float = 8.0
    log_scaling_alpha: float = 0.1
    log_scaling_floor: int = 8192
    conv_kernel: int = 4
    gate_softcap: float = 4.0
    up_softcap: float = 25.0
    route_scale: float = 8.0
    logit_softcap: float = 30.0
    block_layers: int = 6  # layers per block attention residual block
    loop_start: int = 6
    loop_end: int = 18
    num_passes: int = 3
    gate_rank: int = 64

    @property
    def loop_layers(self) -> int:
        return self.loop_end - self.loop_start

    @property
    def num_visits(self) -> int:
        return self.num_layers + (self.num_passes - 1) * self.loop_layers

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    def is_full(self, layer: int) -> bool:
        return layer in self.full_layers

    def is_loop(self, layer: int) -> bool:
        return self.loop_start <= layer < self.loop_end

    def visits(self) -> list[tuple[int, int, int]]:
        """``(visit, layer, pass)`` in execution order; prelude and coda layers have pass 0."""
        order = [(layer, 0) for layer in range(self.loop_start)]
        for p in range(self.num_passes):
            order += [(layer, p) for layer in range(self.loop_start, self.loop_end)]
        order += [(layer, 0) for layer in range(self.loop_end, self.num_layers)]
        return [(visit, layer, p) for visit, (layer, p) in enumerate(order)]

    @classmethod
    def from_hf(cls, config: dict) -> "InklingConfig":
        """The shape of an HF ``config.json``: the multimodal checkpoint's ``text_config`` or a
        text-only ``inkling2_loop_text`` config."""
        text = config.get("text_config", config)
        loop = text.get("loop_config", {})
        layers = text["num_hidden_layers"]
        if "layer_types" in text:
            full = tuple(i for i, kind in enumerate(text["layer_types"]) if kind == "hybrid")
        else:
            full = tuple(i for i in range(layers) if i not in set(text["local_layer_ids"]))
        if text.get("n_shared_experts", 0):
            raise ValueError("shared experts are not implemented")
        for swa, base in (("swa_num_attention_heads", "num_attention_heads"),
                          ("swa_num_key_value_heads", "num_key_value_heads"),
                          ("swa_head_dim", "head_dim")):  # fmt: skip
            if text.get(swa, text[base]) != text[base]:
                raise ValueError(f"{swa} != {base} is not implemented")
        return cls(
            hidden_size=text["hidden_size"],
            num_layers=layers,
            num_q_heads=text["num_attention_heads"],
            num_kv_heads=text["num_key_value_heads"],
            head_dim=text["head_dim"],
            d_rel=text["d_rel"],
            rel_extent=text["rel_extent"],
            sliding_window=text["sliding_window_size"],
            full_layers=full,
            num_experts=text["n_routed_experts"],
            experts_per_token=text["num_experts_per_tok"],
            moe_intermediate_size=text.get("moe_intermediate_size", text.get("intermediate_size")),
            vocab_size=text["unpadded_vocab_size"],
            rms_eps=text["rms_norm_eps"],
            qk_clip=text["qk_l2_clip_threshold"],
            rel_clip=text["rel_l2_clip_threshold"],
            log_scaling_alpha=text["log_scaling_alpha"],
            log_scaling_floor=text["log_scaling_n_floor"],
            conv_kernel=text.get("sconv_kernel_size", text.get("conv_kernel_size", 4)),
            gate_softcap=text["mlp_gate_softcap"],
            up_softcap=text["mlp_up_softcap"],
            route_scale=text["route_scale"],
            logit_softcap=text["final_logit_softcapping"],
            block_layers=text["n_layers_per_attn_res_block"],
            loop_start=loop.get("core_start", text.get("loop_layer_index_start")),
            loop_end=loop.get("core_end", text.get("loop_layer_index_end")),
            num_passes=loop.get("num_passes", text.get("loop_num_passes")),
            gate_rank=loop.get("gate_rank", text.get("loop_gate_rank")),
        )


@dataclass(frozen=True)
class MessageGate:
    """A core layer's controller for one input (query, key, value or MLP)."""

    down: torch.Tensor  # (gate_rank, 2 * hidden) bf16
    forget: torch.Tensor  # (hidden, gate_rank) fp32
    gain: torch.Tensor  # (hidden, gate_rank) fp32


@dataclass(frozen=True)
class ExpertQuant:
    """The declared numerics of one expert matrix (``w13`` or ``w2``), whose weights then hold
    their block values (NVFP4: ``e2m1 x e4m3`` without the per-expert ``amax / denominator``):
    the quantization of its input and, for NVFP4 weights, those global scale parts, which
    ``experts_out`` applies to the fp32 GEMM output (see the module)."""

    activation: ActivationQuant
    weight_amax: torch.Tensor | None = None  # (experts,) fp32: NVFP4 weights
    weight_denominator: torch.Tensor | None = None  # (1,) fp32: NVFP4 weights

    def __post_init__(self):
        if self.weight_amax is not None and self.activation.format != NVFP4:
            raise ValueError("NVFP4 weight scales need NVFP4 activations (as vLLM's GEMMs)")

    def select(self, experts: torch.Tensor) -> "ExpertQuant":
        """The numerics of the experts ``experts`` (indices), in that order."""
        if self.weight_amax is None:
            return self
        return ExpertQuant(self.activation, self.weight_amax[experts], self.weight_denominator)

    def row_factors(self, experts: int, device=None) -> torch.Tensor:
        """fp32 ``(experts + 1,)`` factors of ``row_scale``: each expert's ``w_amax``, then
        ``rcp(1536 * w_denominator)``; for unscaled (bf16) weights ones and ``fp32(1 / 1536)``.
        The megakernels read them (``megakernel.expert_numerics``)."""
        if self.weight_amax is None:
            factors = torch.ones(experts + 1, device=device)
            factors[-1] = 1.0 / NVFP4_GLOBAL
            return factors
        rcp = torch.reciprocal(NVFP4_GLOBAL * self.weight_denominator)
        return torch.cat((self.weight_amax[:experts], rcp))

    def row_scale(self, amax: torch.Tensor, expert: int) -> torch.Tensor:
        """fp32 output scale of rows with activation ``amax`` (NVFP4) through ``expert``:
        ``(amax * w_amax) * rcp(1536 * w_denominator)``, as ``ops/moe.py:346-353``, from
        ``row_factors`` (``amax * 1`` is exact for unscaled weights)."""
        factors = self.row_factors(expert + 1, amax.device)
        return (amax * factors[expert]) * factors[-1]


@dataclass(frozen=True)
class LayerWeights:
    """One decoder layer. Sliding layers have ``input_norm`` and a static key/value
    convolution (``k_conv``, ``v_conv``: (kv_dim, kernel) fp32, the last tap weighting the
    current token); full layers have ``q_norm``/``k_norm``/``v_norm`` and a dynamic one
    (``k_static``, ``v_static``: (kernel, kv_dim) bf16, tap 0 the current token, plus
    ``k_dynamic``/``v_dynamic`` (kv_heads * kernel, hidden) bf16 projections)."""

    wq: torch.Tensor  # (q_heads * head_dim, hidden)
    wk: torch.Tensor  # (kv_dim, hidden)
    wv: torch.Tensor
    wo: torch.Tensor  # (hidden, q_heads * head_dim)
    wgate: torch.Tensor  # (q_heads * head_dim, hidden): sigmoid gate of the attention output
    wrel: torch.Tensor  # (q_heads * d_rel, hidden)
    wsink: torch.Tensor  # (q_heads, hidden)
    q_scale: torch.Tensor  # (q_heads, head_dim)
    rel_scale: torch.Tensor  # (q_heads, d_rel)
    rel_proj: torch.Tensor  # (d_rel, rel_extent)
    attn_conv: torch.Tensor  # (hidden, kernel) fp32
    mlp_conv: torch.Tensor  # (hidden, kernel) fp32
    post_norm: torch.Tensor  # (hidden,)
    mlp_query: torch.Tensor  # (hidden,) fp32
    router: torch.Tensor  # (experts, hidden) fp32
    router_bias: torch.Tensor  # (experts,) fp32
    # bf16 (experts, 2 * moe_intermediate, hidden): gate rows, then up rows; or the result of
    # load_inkling_weights' ``experts`` hook (e.g. quant.QuantExperts, checkpoint row order).
    w13: torch.Tensor
    w2: torch.Tensor  # bf16 (experts, hidden, moe_intermediate), or the hook's result
    attn_query: torch.Tensor | None = None  # (1 or 3, hidden) fp32; None for layer 0
    input_norm: torch.Tensor | None = None
    q_norm: torch.Tensor | None = None
    k_norm: torch.Tensor | None = None
    v_norm: torch.Tensor | None = None
    k_conv: torch.Tensor | None = None
    v_conv: torch.Tensor | None = None
    k_static: torch.Tensor | None = None
    v_static: torch.Tensor | None = None
    k_dynamic: torch.Tensor | None = None
    v_dynamic: torch.Tensor | None = None
    gates: tuple[MessageGate, ...] = ()  # core layers: query, key, value, MLP
    # The experts' declared numerics (both or neither): set when loaded with activation_quant.
    w13_quant: ExpertQuant | None = None
    w2_quant: ExpertQuant | None = None

    @property
    def declared_experts(self) -> bool:
        """Whether the experts follow the checkpoint's declared numerics (``ExpertQuant``)."""
        return self.w13_quant is not None


@dataclass(frozen=True)
class InklingWeights:
    embed: torch.Tensor  # (padded vocab, hidden) bf16
    embed_norm: torch.Tensor
    final_norm: torch.Tensor
    final_query: torch.Tensor  # (hidden,) fp32
    lm_head: torch.Tensor  # (vocab, hidden) bf16, unpadded
    message_queries: torch.Tensor  # (loop_layers, 4, hidden) fp32: query, key, value, MLP
    layers: tuple[LayerWeights, ...]
    config: InklingConfig = field(default_factory=InklingConfig)


@dataclass
class InklingCache:
    """Per-visit decode state, indexed by absolute position: keys (after the L2 clip) and
    values (after the RMS norm) at ``[visit, kv_head, pos]``, and convolution inputs at
    ``[visit, pos % CONV_RING]``. Zero-initialized: masked cache rows are multiplied by zero
    probabilities, which must not meet NaN."""

    k: torch.Tensor  # (visits, kv_heads, max_len, head_dim) bf16
    v: torch.Tensor
    conv_k: torch.Tensor  # (visits, CONV_RING, kv_dim) bf16
    conv_v: torch.Tensor
    conv_attn: torch.Tensor  # (visits, CONV_RING, hidden) bf16
    conv_mlp: torch.Tensor

    @classmethod
    def empty(cls, config: InklingConfig, max_len: int, device="cuda") -> "InklingCache":
        visits, kv, d = config.num_visits, config.num_kv_heads, config.head_dim

        def zeros(*shape):
            return torch.zeros(shape, device=device, dtype=torch.bfloat16)

        return cls(
            k=zeros(visits, kv, max_len, d),
            v=zeros(visits, kv, max_len, d),
            conv_k=zeros(visits, CONV_RING, config.kv_dim),
            conv_v=zeros(visits, CONV_RING, config.kv_dim),
            conv_attn=zeros(visits, CONV_RING, config.hidden_size),
            conv_mlp=zeros(visits, CONV_RING, config.hidden_size),
        )

    @property
    def max_len(self) -> int:
        return self.k.shape[2]


# Checkpoint loading -------------------------------------------------------------------------


def checkpoint_dir(
    path_or_repo: str | Path, tqdm_class: type | None = None, drafter: bool = False
) -> Path:
    """The local directory of a checkpoint: a Hub repo (downloaded if needed), a checkpoint
    directory, or a file inside one (used as they are). A download skips the DSpark drafter
    (``mtp.safetensors``) unless ``drafter``."""
    checkpoint = Path(path_or_repo)
    if checkpoint.is_file():
        return checkpoint.parent
    if checkpoint.is_dir():
        return checkpoint
    from huggingface_hub import snapshot_download  # optional: the oracle needs only torch

    return Path(
        snapshot_download(
            str(path_or_repo),
            allow_patterns=["*.json", "model*.safetensors", "*.jinja", "tiktoken/*"]
            + (["mtp.safetensors"] if drafter else []),
            tqdm_class=tqdm_class,
        )
    )


def dequantize_experts(
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    amax: torch.Tensor | None,
    denominator: torch.Tensor | None,
    *,
    global_scale: bool = True,
) -> torch.Tensor:
    """bf16 (experts, rows, columns) expert weights from the checkpoint's formats, as HF's
    ``Inkling2LoopDequantizeExperts``: NVFP4 (``amax`` given) packs two E2M1 codes per byte
    (even column in the low nibble) with an fp8 scale per 16 columns and a per-expert factor
    ``amax / denominator``; MXFP8 has fp8 values with a power-of-two scale ``2 ** (scale - 127)``
    per 32 columns. Unquantized weights (no ``scale``) are only cast. Without ``global_scale``,
    NVFP4 weights are their exact block values ``e2m1 x scale`` (``ExpertQuant``)."""
    if scale is None:
        return weight.to(torch.bfloat16)
    experts, rows = weight.shape[:2]
    columns = weight.shape[2] * 2 if amax is not None else weight.shape[2]
    out = torch.empty(experts, rows, columns, dtype=torch.bfloat16, device=weight.device)
    if amax is not None:
        e2m1 = torch.tensor(E2M1_VALUES, device=weight.device)
        byte = torch.arange(256, device=weight.device)
        byte_values = torch.stack((e2m1[byte & 0xF], e2m1[byte >> 4]), dim=-1)
        factor = amax.float() / denominator.float()
    for start in range(0, experts, 8):  # chunks bound the fp32 temporaries
        chunk = slice(start, start + 8)
        block_scale = scale[chunk].float()
        if amax is None:
            values = weight[chunk].float().view(*block_scale.shape, 32)
            values = values * ue8m0_scale(scale[chunk])[..., None]
        else:
            codes = weight[chunk].view(torch.uint8).long()
            values = byte_values[codes].view(*block_scale.shape, 16) * block_scale[..., None]
            if global_scale:
                values = values * factor[chunk, None, None, None]
        out[chunk] = values.reshape(-1, rows, columns).to(torch.bfloat16)
    return out


def deinterleave_gate_up(w13: torch.Tensor) -> torch.Tensor:
    """``(experts, 2 * width, columns)`` gate/up rows in the checkpoint's order (interleaved:
    row 2j gate row j, 2j + 1 up row j) as HF's: gate rows, then up rows."""
    return w13.unflatten(1, (-1, 2)).transpose(1, 2).flatten(1, 2).contiguous()


def load_inkling_weights(
    path_or_repo: str | Path = "thinkingmachines/Inkling-2-Mini",
    device="cuda",
    experts=None,
    activation_quant: bool | None = None,
) -> InklingWeights:
    """The text decoder of an Inkling 2 Mini checkpoint in its original naming
    (``model.llm.*``): a Hub repo, a checkpoint directory, or a file inside one.

    ``activation_quant``: None follows the expert activation quantization the checkpoint's
    ``hf_quant_config.json`` declares (none without the file), True requires a declaration
    for every layer's experts, False keeps HF's numerics (bf16 activations, the NVFP4 factor
    folded into bf16 weights). Layers that follow a declaration carry ``ExpertQuant`` numerics
    and NVFP4 block values without the factor (see the module), which the experts code paths
    (``experts_out``, ``fast``, the megakernels) compute.

    ``experts(weight, scale, amax, denominator)``, when given, replaces the bf16
    dequantization (and the gate/up row deinterleave) of each layer's expert tensors; its
    result is stored in ``w13``/``w2`` as is. It takes ``dequantize_experts``' arguments:
    ``scale``, ``amax`` and ``denominator`` are None when the checkpoint has none."""
    from safetensors import safe_open  # optional: the oracle needs only torch

    checkpoint = checkpoint_dir(path_or_repo)
    config = InklingConfig.from_hf(json.loads((checkpoint / "config.json").read_text()))
    declared = {} if activation_quant is False else declared_activation_quant(checkpoint)
    files = {}
    for shard in sorted(checkpoint.glob("model*.safetensors")):
        handle = safe_open(str(shard), "pt", device=str(device))
        files.update({name: handle for name in handle.keys() if name.startswith("model.llm.")})  # noqa: SIM118 (safe_open is not a dict)

    def get(name: str, dtype: torch.dtype | None = torch.bfloat16) -> torch.Tensor:
        tensor = files[f"model.llm.{name}"].get_tensor(f"model.llm.{name}")
        return tensor if dtype is None else tensor.to(dtype).contiguous()

    def has(name: str) -> bool:
        return f"model.llm.{name}" in files

    def load_experts(prefix: str, gate_up: bool) -> tuple[torch.Tensor, ExpertQuant | None]:
        parts = [f"{prefix}_scale", f"{prefix}_amax", f"{prefix}_outer_scale_denominator"]
        args = (get(prefix, None), *(get(p, None) if has(p) else None for p in parts))
        activation = declared.get(f"model.llm.{prefix}")
        if activation is None and activation_quant:
            raise ValueError(f"the checkpoint declares no activation quantization of {prefix}")
        numerics = None
        if activation is not None:
            amax, denominator = args[2:]
            numerics = ExpertQuant(
                activation,
                None if amax is None else amax.float(),
                None if denominator is None else denominator.float().reshape(1),
            )
        if experts is not None:
            return experts(*args), numerics
        w = dequantize_experts(*args, global_scale=numerics is None)
        return (deinterleave_gate_up(w) if gate_up else w), numerics

    fp32 = torch.float32
    # The checkpoint stores query and MLP message queries of each core layer in one tensor and
    # key and value queries in another; merged as HF's Inkling2LoopMergeMessageQueries.
    qm = get("middle_cycle_cross_pass_queries", fp32).unflatten(0, (-1, 2))
    kv = get("middle_cycle_cross_pass_kv_queries", fp32).unflatten(0, (-1, 2))
    message_queries = torch.stack((qm[:, 0], kv[:, 0], kv[:, 1], qm[:, 1]), dim=1).contiguous()

    layers = []
    for layer in range(config.num_layers):
        p = f"layers.{layer}."
        a = p + "attn."
        w13, w13_quant = load_experts(p + "mlp.experts.w13_weight", True)
        w2, w2_quant = load_experts(p + "mlp.experts.w2_weight", False)
        if (w13_quant is None) != (w2_quant is None):
            raise ValueError(f"layer {layer} declares the activation quantization of one matrix")
        fields = {
            "wq": get(a + "wq_du.weight"),
            "wk": get(a + "wk_dv.weight"),
            "wv": get(a + "wv_dv.weight"),
            "wo": get(a + "wo_ud.weight"),
            "wgate": get(a + "gate_attn.weight"),
            "wrel": get(a + "wr_du.weight"),
            "wsink": get(a + "attn_sink_proj.weight"),
            "q_scale": get(a + "q_l2_clip.scale"),
            "rel_scale": get(a + "rel_l2_clip.scale"),
            "rel_proj": get(a + "rel_logits_proj.proj"),
            "attn_conv": get(p + "attn_sconv.weight", fp32).squeeze(1),
            "mlp_conv": get(p + "mlp_sconv.weight", fp32).squeeze(1),
            "post_norm": get(p + "mlp_norm.weight"),
            "mlp_query": get(f"attn_res.mlp_queries.{layer}", fp32),
            "router": get(p + "mlp.gate.weight", fp32),
            "router_bias": get(p + "mlp.gate.bias", fp32),
            "w13": w13,
            "w2": w2,
            "w13_quant": w13_quant,
            "w2_quant": w2_quant,
        }
        if layer > 0:  # attn_queries.i belongs to layer i + 1
            fields["attn_query"] = get(f"attn_res.attn_queries.{layer - 1}", fp32).reshape(
                -1, config.hidden_size
            )
        if config.is_full(layer):
            fields.update(
                q_norm=get(p + "q_input_norm.weight"),
                k_norm=get(p + "k_input_norm.weight"),
                v_norm=get(p + "v_input_norm.weight"),
                k_static=get(a + "k_sconv.static_w"),
                v_static=get(a + "v_sconv.static_w"),
                k_dynamic=get(a + "k_sconv.weight_proj"),
                v_dynamic=get(a + "v_sconv.weight_proj"),
            )
        else:
            fields.update(
                input_norm=get(p + "attn_norm.weight"),
                k_conv=get(a + "k_sconv.weight", fp32).squeeze(1),
                v_conv=get(a + "v_sconv.weight", fp32).squeeze(1),
            )
        if config.is_loop(layer):
            fields["gates"] = tuple(
                MessageGate(
                    down=get(f"{p}controllers.{name}.down_weight"),
                    forget=get(f"{p}controllers.{name}.forget_weight", fp32),
                    gain=get(f"{p}controllers.{name}.gain_weight", fp32),
                )
                for name in ("attn", "k", "v", "mlp")  # "attn" gates the query input
            )
        layers.append(LayerWeights(**fields))
    return InklingWeights(
        embed=get("embed.weight"),
        embed_norm=get("embed_norm.weight"),
        final_norm=get("norm.weight"),
        final_query=get("attn_res.output_query", fp32),
        lm_head=get("unembed.weight")[: config.vocab_size].contiguous(),
        message_queries=message_queries,
        layers=tuple(layers),
        config=config,
    )


# Reference forward --------------------------------------------------------------------------


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """HF's ``Inkling2LoopRMSNorm`` (weighted: ``weight * fp32`` then back to ``x.dtype``) and
    ``Inkling2LoopUnweightedRMSNorm`` (``weight=None``)."""
    acc = x.float()
    normalized = acc * torch.rsqrt(acc.square().mean(-1, keepdim=True) + eps)
    return (normalized if weight is None else weight * normalized).to(x.dtype)


def l2_clip(x: torch.Tensor, threshold: float) -> torch.Tensor:
    """Scales fp32 vectors longer than ``threshold`` down to it."""
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return x * (threshold / (norm + 1e-6)).clamp(max=1.0)


def log_scale(config: InklingConfig, positions: torch.Tensor) -> torch.Tensor:
    """(tokens, 1, 1) fp32 scale ``1 + alpha * log(max((pos + 1) / floor, 1))`` of full-attention
    queries and relative states."""
    ratio = (positions + 1).float() / config.log_scaling_floor
    return (1.0 + config.log_scaling_alpha * torch.log(ratio.clamp(min=1.0)))[:, None, None]


def clip_heads(
    config: InklingConfig,
    x: torch.Tensor,
    scale: torch.Tensor,
    threshold: float,
    full: bool,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Per-head ``scale``, L2 clip and (full layers) log scaling of bf16 (tokens, heads, dim)."""
    x = l2_clip(x.float() * scale.float(), threshold).to(torch.bfloat16)
    if full:
        x = (x.float() * log_scale(config, positions)).to(torch.bfloat16)
    return x


def block_residual(queries: torch.Tensor, blocks: list[torch.Tensor], partial: torch.Tensor):
    """Block attention residual: each fp32 query of (queries, hidden) attends over the
    RMS-normalized completed blocks and the partial block, keys and values alike. Returns
    (tokens, queries, hidden) fp32."""
    states = torch.stack([b.float() for b in blocks] + [partial.float()], dim=1)
    states = rms_norm(states, None, 1e-6)
    queries = queries.expand(states.shape[0], -1, -1)
    # Batched matmuls in HF's eager-attention shapes: the fp32 summation order matches.
    scores = torch.matmul(queries, states.transpose(1, 2)) * states.shape[-1] ** -0.5
    return torch.matmul(scores.softmax(-1), states)


def message_gate(gate: MessageGate, x: torch.Tensor, message: torch.Tensor, eps: float):
    """``(1 - tanh(forget h)) x + tanh(gain h) message`` with ``h = down [rms(x), rms(message)]``,
    all fp32 except the bf16 down projection."""
    x, message = x.float(), message.float()
    h = torch.cat([rms_norm(x, None, eps), rms_norm(message, None, eps)], dim=-1)
    h = F.linear(h.to(torch.bfloat16), gate.down).float()
    forget, gain = torch.tanh(F.linear(h, gate.forget)), torch.tanh(F.linear(h, gate.gain))
    return (1 - forget) * x + gain * message


def conv_history(ring: torch.Tensor, x: torch.Tensor, positions: torch.Tensor, kernel: int):
    """(tokens, kernel, channels) bf16 inputs of a causal convolution at each of ``positions``
    (consecutive), tap 0 the current token, tap j the token j back (zero before position 0).
    Writes ``x`` into ``ring`` (``(CONV_RING, channels)``) at its positions."""
    count = x.shape[0]
    start = positions[0]
    earlier = start - torch.arange(kernel - 1, 0, -1, device=x.device)
    past = ring[earlier % CONV_RING] * (earlier >= 0)[:, None].to(ring.dtype)
    history = torch.cat([past, x], dim=0)  # positions start - kernel + 1 .. end
    ring[positions[-CONV_RING:] % CONV_RING] = x[-CONV_RING:]
    taps = torch.arange(kernel, device=x.device)
    return history[torch.arange(count, device=x.device)[:, None] + kernel - 1 - taps]


def short_conv(ring, x: torch.Tensor, weight: torch.Tensor, positions, kernel: int):
    """HF's ``Inkling2LoopShortConvolution``: fp32 depthwise causal convolution (``weight``
    (channels, kernel), the last tap on the current token) plus the input, over
    ``sqrt(kernel + 1)``, rounded to bf16. Runs as HF's ``F.conv1d`` (its summation order)."""
    channels = x.shape[1]
    history = conv_history(ring, x, positions, kernel)  # (tokens, kernel, channels)
    # The distinct inputs in position order: kernel - 1 earlier tokens, then the new ones.
    sequence = torch.cat([history[0, 1:].flip(0), history[:, 0]]).float().T[None]
    out = F.conv1d(sequence, weight[:, None, :], groups=channels)[0].T  # (tokens, channels)
    return ((out + x.float()) / math.sqrt(kernel + 1)).to(torch.bfloat16)


def dynamic_conv(ring, x, dynamic_input, static, dynamic, heads: int, positions, kernel: int):
    """HF's ``Inkling2LoopDynamicShortConvolution``: taps ``static`` (kernel, channels) plus
    per-token, per-head taps projected from ``dynamic_input``."""
    count, channels = x.shape
    taps = F.linear(dynamic_input, dynamic).float().view(count, heads, 1, kernel)
    taps = static.float().T.view(heads, -1, kernel) + taps  # (tokens, heads, dim, kernel)
    history = conv_history(ring, x, positions, kernel).float()  # (tokens, kernel, channels)
    windows = history.transpose(1, 2).reshape(count, heads, -1, kernel)
    out = (windows * taps).sum(-1).reshape(count, channels)
    return ((out + x.float()) / math.sqrt(2 * kernel + 1)).to(torch.bfloat16)


def attention(
    config: InklingConfig,
    w: LayerWeights,
    cache: InklingCache,
    visit: int,
    full: bool,
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    positions: torch.Tensor,
    start: int | None,
    kernels: "Kernels",
) -> torch.Tensor:
    """Gated relative-bias attention with a sink; writes this visit's K/V and conv inputs at
    ``positions`` and attends with a mask (causal, plus the window in sliding layers). Bias, sink
    and output gate read the query input ``q_in``.

    ``start`` (the first position, known on the host) reads HF's key span, ``[0, start + T)`` or
    for sliding layers from ``start - window + 1`` (HF's sliding cache keeps window - 1 keys),
    with HF's per-query-head GEMMs: bitwise HF. ``None`` reads the whole cache with grouped
    GEMMs: static shapes for CUDA graphs, rounding that differs from HF's."""
    count = q_in.shape[0]
    heads, kv, d, kernel = (
        config.num_q_heads,
        config.num_kv_heads,
        config.head_dim,
        config.conv_kernel,
    )
    q = F.linear(q_in, w.wq).view(count, heads, d)
    q = clip_heads(config, q, w.q_scale, config.qk_clip, full, positions)
    k, v = F.linear(k_in, w.wk), F.linear(v_in, w.wv)
    if full:
        k = dynamic_conv(
            cache.conv_k[visit], k, k_in, w.k_static, w.k_dynamic, kv, positions, kernel
        )
        v = dynamic_conv(
            cache.conv_v[visit], v, v_in, w.v_static, w.v_dynamic, kv, positions, kernel
        )
    else:
        k = kernels.short_conv(cache.conv_k[visit], k, w.k_conv, positions, kernel)
        v = kernels.short_conv(cache.conv_v[visit], v, w.v_conv, positions, kernel)
    k = l2_clip(k.view(count, kv, d).float(), config.qk_clip).to(torch.bfloat16)
    v = rms_norm(v.view(count, kv, d), None, config.rms_eps)
    cache.k[visit][:, positions] = k.transpose(0, 1)
    cache.v[visit][:, positions] = v.transpose(0, 1)

    group = heads // kv
    if start is None:
        span = slice(0, cache.max_len)
    else:
        first = 0 if full else max(0, start - config.sliding_window + 1)
        span = slice(first, start + count)
    keys, values = cache.k[visit, :, span], cache.v[visit, :, span]  # (kv, keys, d)
    if start is None:
        scores = q.transpose(0, 1).reshape(kv, group * count, d) @ keys.transpose(1, 2)
    else:  # HF's repeat_kv: one GEMM per query head
        keys, values = keys.repeat_interleave(group, 0), values.repeat_interleave(group, 0)
        scores = q.transpose(0, 1) @ keys.transpose(1, 2)
    scores = scores.view(heads, count, -1) * d**-0.5

    rel_extent = config.rel_extent if full else config.sliding_window
    rel = F.linear(q_in, w.wrel).view(count, heads, config.d_rel)
    rel = clip_heads(config, rel, w.rel_scale, config.rel_clip, full, positions)
    rel_logits = (rel @ w.rel_proj[:, :rel_extent]).transpose(0, 1)  # (heads, tokens, extent)
    key_positions = torch.arange(span.start, span.stop, device=q.device)
    distance = positions[:, None] - key_positions[None, :]
    bias = rel_logits.gather(-1, distance.clamp(0, rel_extent - 1).expand(heads, -1, -1))
    in_extent = (distance >= 0) & (distance < rel_extent)
    scores = scores + bias.masked_fill(~in_extent, 0.0)
    visible = distance >= 0 if full else (distance >= 0) & (distance < config.sliding_window)
    scores = scores.masked_fill(~visible, float("-inf"))

    sinks = F.linear(q_in, w.wsink).transpose(0, 1)[..., None]  # (heads, tokens, 1)
    probs = torch.cat([scores, sinks], dim=-1).float().softmax(-1)[..., :-1].to(torch.bfloat16)
    if start is None:
        out = probs.view(kv, group * count, -1) @ values
    else:
        out = probs @ values
    out = out.view(heads, count, d).transpose(0, 1).reshape(count, heads * d)
    gate = torch.sigmoid(F.linear(q_in, w.wgate).float())
    return F.linear((out.float() * gate).to(torch.bfloat16), w.wo)


def route(config: InklingConfig, w: LayerWeights, x: torch.Tensor):
    """Sigmoid router in fp32: top-k of scores plus the correction bias, weights renormalized
    and scaled. Returns ((tokens, k) experts, (tokens, k) fp32 weights)."""
    scores = torch.sigmoid(F.linear(x.float(), w.router))
    # sorted=False as HF: the order of the k weights sets their fp32 sum's rounding.
    experts = torch.topk(scores + w.router_bias, config.experts_per_token, dim=-1, sorted=False)
    experts = experts.indices
    weights = scores.gather(1, experts)
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    return experts, weights * config.route_scale


def require_dense_experts(w: LayerWeights):
    """Raise unless the layer's experts are dense tensors: weights loaded through
    ``load_inkling_weights``' ``experts`` hook (e.g. quantized) need their own kernels."""
    for name in ("w13", "w2"):
        if not isinstance(getattr(w, name), torch.Tensor):
            raise TypeError(
                f"dense bf16 experts required, got {type(getattr(w, name)).__name__} {name}; "
                "dequantize them first"
            )


def swiglu(config: InklingConfig, gate_up: torch.Tensor, scale=None) -> torch.Tensor:
    """fp32 softcapped SwiGLU ``capg(g) sigmoid(g) capu(u)`` of bf16 (rows, 2 * width) gate
    then up columns, in HF's operations, times fp32 ``scale`` (rows, 1) if given."""
    gate, up = gate_up.float().chunk(2, dim=-1)
    act = torch.tanh(gate / config.gate_softcap) * config.gate_softcap * torch.sigmoid(gate)
    act = act * (torch.tanh(up / config.up_softcap) * config.up_softcap)
    return act if scale is None else act * scale


def scaled_mm(values, amax, weight: torch.Tensor, quant: ExpertQuant, expert: int):
    """bf16 ``bf16(dot * row_scale)`` of activations quantized per ``quant.activation``, as
    their exact bf16 block values ``values`` (rows, K) and NVFP4 row ``amax`` (else None),
    through one expert's block values ``weight`` (N, K): the fp32 dot products times the NVFP4
    row scale (``ExpertQuant.row_scale``)."""
    out = torch.mm(values, weight.T, out_dtype=torch.float32)  # exact products, fp32 sums
    if amax is not None:
        out = out * quant.row_scale(amax, expert)[:, None]
    return out.to(torch.bfloat16)


def declared_activations(config: InklingConfig, w: LayerWeights, x, experts, weights):
    """``(routed, acts)``: the first half of ``declared_experts_out``. ``routed`` lists ``(expert,
    tokens, slots)`` in ascending expert order and ``acts`` the bf16 down inputs ``bf16(swiglu
    * weight)`` of its rows in that order (for one token: slot order of the experts sorted)."""
    x_values, x_amax = quantize_dequantize(w.w13_quant.activation, x)
    routed = [(e, *(experts == e).nonzero(as_tuple=True)) for e in experts.unique().tolist()]
    acts = []
    for e, token, slot in routed:  # ascending
        amax = None if x_amax is None else x_amax[token]
        gate_up = scaled_mm(x_values[token], amax, w.w13[e], w.w13_quant, e)
        acts.append(swiglu(config, gate_up, weights[token, slot, None]).to(torch.bfloat16))
    return routed, torch.cat(acts)


def declared_down(w: LayerWeights, x: torch.Tensor, routed: list, acts: torch.Tensor):
    """The second half of ``declared_experts_out``: ``declared_activations``' rows quantized
    (each with its own amax), through ``w2``, summed per token of ``x`` in fp32 in ascending
    expert order; returns bf16."""
    act_values, act_amax = quantize_dequantize(w.w2_quant.activation, acts)
    out = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
    end = 0
    for e, token, _ in routed:
        rows = slice(end, end + token.numel())
        end = rows.stop
        amax = None if act_amax is None else act_amax[rows]
        out.index_add_(0, token, scaled_mm(act_values[rows], amax, w.w2[e], w.w2_quant, e).float())
    return out.to(torch.bfloat16)


def declared_experts_out(config: InklingConfig, w: LayerWeights, x, experts, weights):
    """``experts_out`` of a layer with declared numerics (see the module). Quantization is
    row-local: the gate/up input is quantized once per token and the down input once per
    routed row, as vLLM's kernels do."""
    return declared_down(w, x, *declared_activations(config, w, x, experts, weights))


def experts_out(config: InklingConfig, w: LayerWeights, x: torch.Tensor, experts, weights):
    """The routed experts' fp32 sum on bf16 (tokens, hidden) ``x``, added in ascending expert
    order as HF does; returns bf16. Softcapped SwiGLU: ``capg(g) sigmoid(g) capu(u)``. Layers
    with declared numerics (``ExpertQuant``) compute as vLLM (``declared_experts_out``)."""
    require_dense_experts(w)
    if w.declared_experts:
        return declared_experts_out(config, w, x, experts, weights)
    out = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
    for e in experts.unique().tolist():  # ascending
        token, slot = (experts == e).nonzero(as_tuple=True)
        act = swiglu(config, F.linear(x[token], w.w13[e]))
        y = F.linear(act.to(torch.bfloat16), w.w2[e])
        out.index_add_(0, token, y * weights[token, slot, None])
    return out.to(torch.bfloat16)


def decoder_layer(
    config: InklingConfig,
    w: LayerWeights,
    layer: int,
    visit: int,
    h: torch.Tensor,
    blocks: list[torch.Tensor],
    messages: torch.Tensor | None,
    cache: InklingCache,
    positions: torch.Tensor,
    start: int | None,
    kernels: "Kernels",
) -> torch.Tensor:
    """One visit of ``layer`` on bf16 (tokens, hidden) ``h``; ``messages`` (tokens, 4, hidden)
    fp32 in core passes after the first."""
    residual, blocks = kernels.attention_block(
        config, w, layer, visit, h, blocks, messages, cache, positions, start, kernels
    )
    mlp_message = None if messages is None else messages[:, 3]
    return kernels.mlp_block(
        config, w, visit, residual, blocks, mlp_message, cache, positions, kernels
    )


def attention_block(
    config: InklingConfig,
    w: LayerWeights,
    layer: int,
    visit: int,
    h: torch.Tensor,
    blocks: list[torch.Tensor],
    messages: torch.Tensor | None,
    cache: InklingCache,
    positions: torch.Tensor,
    start: int | None,
    kernels: "Kernels",
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """A visit's first half on its bf16 input ``h``: the attention inputs (block residual,
    message gates, RMSNorm), attention, post-attention convolution and the residual. Returns the
    fp32 post-attention residual and the blocks the MLP half reads (``blocks`` plus ``h`` at a
    block boundary, where the residual restarts from the attention output)."""
    eps, full = config.rms_eps, config.is_full(layer)
    if full:
        streams = list(block_residual(w.attn_query, blocks, h).unbind(1))
        if messages is not None:
            streams = [message_gate(g, s, messages[:, i], eps) for i, (g, s) in
                       enumerate(zip(w.gates[:3], streams))]  # fmt: skip
        norms = (w.q_norm, w.k_norm, w.v_norm)
        q_in, k_in, v_in = (rms_norm(s, n, eps).to(torch.bfloat16) for s, n in zip(streams, norms))
    else:
        x = h if w.attn_query is None else block_residual(w.attn_query, blocks, h)[:, 0]
        if messages is None:
            q_in = k_in = v_in = rms_norm(x, w.input_norm, eps).to(torch.bfloat16)
        else:
            q_in, k_in, v_in = (
                rms_norm(message_gate(g, x, messages[:, i], eps), w.input_norm, eps).to(
                    torch.bfloat16
                )
                for i, g in enumerate(w.gates[:3])
            )
    attn = attention(config, w, cache, visit, full, q_in, k_in, v_in, positions, start, kernels)
    conv = kernels.short_conv
    residual = conv(cache.conv_attn[visit], attn, w.attn_conv, positions, config.conv_kernel)
    residual = residual.float()
    if layer % config.block_layers == 0:  # the attention output starts a new block
        return residual, blocks + [h]
    return h.float() + residual, blocks


def mlp_block(
    config: InklingConfig,
    w: LayerWeights,
    visit: int,
    residual: torch.Tensor,
    blocks: list[torch.Tensor],
    message: torch.Tensor | None,
    cache: InklingCache,
    positions: torch.Tensor,
    kernels: "Kernels",
) -> torch.Tensor:
    """A visit's second half on the fp32 (tokens, hidden) post-attention ``residual``: block
    residual over ``blocks`` (this layer's, including its input at a block boundary), the MLP
    message gate (core passes after the first), RMSNorm, routed experts, post-MLP convolution and
    the residual add. Returns the visit's bf16 output."""
    x = block_residual(w.mlp_query[None], blocks, residual)[:, 0]
    if message is not None:
        x = message_gate(w.gates[3], x, message, config.rms_eps)
    x = rms_norm(x, w.post_norm, config.rms_eps).to(torch.bfloat16)
    y = kernels.experts(config, w, x, *route(config, w, x))
    y = kernels.short_conv(cache.conv_mlp[visit], y, w.mlp_conv, positions, config.conv_kernel)
    return (residual + y.float()).to(torch.bfloat16)


def hidden_states(
    weights: InklingWeights,
    tokens: torch.Tensor,
    positions: torch.Tensor,
    cache: InklingCache,
    start: int | None,
    kernels: "Kernels | None" = None,
    observe: Callable[[int, torch.Tensor], None] | None = None,
) -> torch.Tensor:
    """The final normalized bf16 (tokens, hidden) states of ``tokens`` at consecutive
    ``positions``, updating ``cache``. ``start``: the first position on the host for HF's
    attention spans and rounding, or None for static shapes (see ``attention``). ``kernels``:
    implementations of the short convolution, experts and a visit's halves (``Kernels``;
    default: this module's, HF's numerics). ``observe(visit, h)``, when given, receives each
    visit's bf16 (tokens, hidden) input (the running partial block) before the visit runs,
    then the final normalized states as visit ``num_visits`` (the DSpark drafter's features)."""
    config = weights.config
    kernels = kernels or REFERENCE
    h = rms_norm(F.embedding(tokens, weights.embed), weights.embed_norm, config.rms_eps)
    blocks: list[torch.Tensor] = []
    pre_loop: tuple[torch.Tensor, list[torch.Tensor]] | None = None
    previous: tuple[list[torch.Tensor], torch.Tensor] | None = None  # last pass's blocks, output
    for visit, layer, p in config.visits():
        messages = None
        if config.is_loop(layer):
            core = layer - config.loop_start
            if core == 0 and p == 0:
                pre_loop = (h, blocks)
            elif core == 0:  # restart from the prelude's output, remembering this pass
                # Skip the prelude's blocks and the core's first block (the prelude's output).
                previous = (blocks[len(pre_loop[1]) + 1 :], h)
                h, blocks = pre_loop
            if p > 0:
                prev_blocks, prev_h = previous
                messages = block_residual(weights.message_queries[core], prev_blocks, prev_h)
        if observe is not None:
            observe(visit, h)
        w = weights.layers[layer]
        out = decoder_layer(
            config, w, layer, visit, h, blocks, messages, cache, positions, start, kernels
        )
        if layer % config.block_layers == 0:
            blocks = blocks + [h]
        h = out
    h = block_residual(weights.final_query[None], blocks, h)[:, 0]
    h = rms_norm(h, weights.final_norm, config.rms_eps).to(torch.bfloat16)
    if observe is not None:
        observe(config.num_visits, h)
    return h


@dataclass(frozen=True)
class Kernels:
    """Swappable implementations with the signatures of this module's functions of the same
    names: ``short_conv``, ``experts`` (``experts_out``), and a visit's halves ``attention_block``
    and ``mlp_block`` (which by default call the others)."""

    short_conv: Callable = short_conv
    experts: Callable = experts_out
    attention_block: Callable = attention_block
    mlp_block: Callable = mlp_block


REFERENCE = Kernels()


def logits_of(weights: InklingWeights, h: torch.Tensor) -> torch.Tensor:
    """fp32 softcapped logits of bf16 final states."""
    cap = weights.config.logit_softcap
    return torch.tanh(F.linear(h, weights.lm_head).float() / cap) * cap


@torch.no_grad()
def forward_tokens(
    weights: InklingWeights,
    tokens: torch.Tensor,
    pos: int,
    cache: InklingCache,
    *,
    all_logits: bool = False,
    kernels: "Kernels | None" = None,
    observe: Callable[[int, torch.Tensor], None] | None = None,
) -> torch.Tensor:
    """Prefill or decode ``tokens`` at positions ``pos ..``; returns the last token's fp32
    logits (``all_logits``: every token's, (tokens, vocab)). ``kernels`` and ``observe``: as
    ``hidden_states``'."""
    tokens = tokens.reshape(-1)
    if pos + tokens.numel() > cache.max_len:
        raise ValueError(f"positions up to {pos + tokens.numel()} exceed max_len {cache.max_len}")
    positions = torch.arange(pos, pos + tokens.numel(), device=tokens.device)
    h = hidden_states(
        weights, tokens, positions, cache, start=pos, kernels=kernels, observe=observe
    )
    logits = logits_of(weights, h if all_logits else h[-1:])
    return logits if all_logits else logits[0]
