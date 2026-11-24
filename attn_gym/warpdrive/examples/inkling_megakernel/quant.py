"""The checkpoint's quantized experts in the layout of the expert GEMV
(``examples.megakernel.quant_gemv``), and their exact bf16 decodings.

Inkling 2 Mini stores the experts of layers 0-19 as NVFP4 and of layers 20-23 as MXFP8
(``ref.dequantize_experts`` defines the values). ``QuantExperts.from_checkpoint`` keeps them
quantized, in the checkpoint's row order: ``w13`` rows interleave gate and up (row 2j is gate
row j, 2j + 1 up row j), so a 32-row tile holds 16 (gate, up) pairs, as the MLP kernel's
strided view of the bf16 weights does. ``load_quantized_inkling_weights`` loads the model with
such experts: 19.0 GB of device memory, 15.7 GB of it experts (bf16 experts: 48.9 GB). The
oracle's dense code paths reject them; ``forward_tokens`` runs the oracle on such a model by
decoding only each MLP's routed experts, and ``dequantized_weights`` returns the bf16 model.
By default the loader follows the checkpoint's declared activation quantization
(``ref.load_inkling_weights``' ``activation_quant``): the experts' bf16 form is then their
block values (``QuantExperts.block_values``, the weights of ``ref.ExpertQuant`` numerics, which
the kernels compute), else HF's dequantized weights (``QuantExperts.dequantize``).

The layout follows the GEMV: 32-row x 256-column subtiles, warp w of 8 owning rows
4w .. 4w + 3 and lane l columns 8l .. 8l + 7 of every subtile.

- NVFP4 weights are the checkpoint's bytes: two E2M1 codes each, the even column in the low
  nibble, so byte j of a lane's 32-bit word holds its columns 2j, 2j + 1. The kernel decodes
  their block values ``e2m1 x scale`` exactly; the per-expert ``amax / denominator`` is the
  NVFP4 GEMM's output scale (``ref.ExpertQuant``), kept here (fp32, as the checkpoint stores
  it; HF's loader rounds amax to bf16) for HF's dequantization. ``from_checkpoint`` rejects NaN
  scales and non-finite factors.
- MXFP8 weights are the checkpoint's e4m3 bytes; the kernel converts them to bf16 and
  multiplies by the bf16 power of two ``2 ** (s - 127)`` of the scale byte s. That is the
  oracle's value whenever the product is a normal number or overflows, which
  ``from_checkpoint`` checks (every scale byte in [10, 254]: 255 is E8M0's NaN; the
  checkpoint's are 113 .. 118).
- Scales of both formats are tile-major bytes ``(experts, rows / 32, subtiles, 8, blocks, 4)``:
  per subtile and warp, block by block, the warp's 4 rows; a lane reads its 4 rows' scales as
  one word. ``blocks`` is 16 (NVFP4: 16 columns each) or 8 (MXFP8: 32 columns). A partial last
  subtile (the down projection's 1152 columns are 4.5 subtiles) has zero scales past the end.
"""

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import torch

from attn_gym.warpdrive.examples.inkling_megakernel import ref
from attn_gym.warpdrive.examples.megakernel.constants import (
    COMPUTE_WARPS,
    ROWS,
    ROWS_PER_WARP,
    TILE_K,
)
from attn_gym.warpdrive.examples.megakernel.formats import (
    BLOCK_COLUMNS,
    MIN_MX_SCALE,
    MXFP8,
    NVFP4,
)


def tile_scales(scale: torch.Tensor, block_columns: int) -> torch.Tensor:
    """Row-major ``(experts, rows, blocks)`` scale bytes in the tile-major kernel layout."""
    experts, rows, blocks = scale.shape
    per_subtile = TILE_K // block_columns
    subtiles = -(-blocks // per_subtile)
    padded = torch.zeros(
        experts, rows, subtiles * per_subtile, dtype=torch.uint8, device=scale.device
    )
    padded[..., :blocks] = scale.view(torch.uint8)
    tiles = padded.view(experts, rows // ROWS, COMPUTE_WARPS, ROWS_PER_WARP, subtiles, per_subtile)
    return tiles.permute(0, 1, 4, 2, 5, 3).contiguous()


def untile_scales(tiled: torch.Tensor, blocks: int) -> torch.Tensor:
    """The inverse of ``tile_scales``: row-major ``(experts, rows, blocks)`` bytes."""
    experts, row_tiles = tiled.shape[:2]
    rows = tiled.permute(0, 1, 3, 5, 2, 4).reshape(experts, row_tiles * ROWS, -1)
    return rows[..., :blocks]


@dataclass(frozen=True)
class QuantExperts:
    """One layer's expert matrix ``(experts, rows, columns)`` in the kernel layout (see the
    module). ``weight``: uint8 ``(experts, rows, columns / 2)`` NVFP4 codes or
    ``(experts, rows, columns)`` e4m3 bytes; ``scale``: tile-major uint8 scales; NVFP4's fp32
    per-expert ``amax`` ``(experts,)`` and ``denominator`` ``(1,)`` (None for MXFP8). Not a
    tensor: dense code paths (``ref.experts_out``) reject it."""

    format: str
    columns: int
    weight: torch.Tensor
    scale: torch.Tensor
    amax: torch.Tensor | None = None
    denominator: torch.Tensor | None = None

    @property
    def experts(self) -> int:
        return self.weight.shape[0]

    @property
    def rows(self) -> int:
        return self.weight.shape[1]

    @property
    def subtiles(self) -> int:
        return self.scale.shape[2]

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.weight, self.scale))

    @classmethod
    def from_checkpoint(
        cls,
        weight: torch.Tensor,
        scale: torch.Tensor,
        amax: torch.Tensor | None,
        denominator: torch.Tensor | None,
    ) -> "QuantExperts":
        """The kernel layout of checkpoint tensors as ``ref.dequantize_experts`` takes them:
        NVFP4 when ``amax`` is given, else MXFP8."""
        rows = weight.shape[1]
        if scale is None:
            raise ValueError("unquantized experts: dequantize them with ref.dequantize_experts")
        if rows % ROWS:
            raise ValueError(f"rows must be a multiple of {ROWS}, got {rows}")
        if amax is None:
            if int(scale.min()) < MIN_MX_SCALE:
                raise ValueError(
                    f"MXFP8 scale bytes below {MIN_MX_SCALE} give subnormal products, which the "
                    "kernel's bf16 multiply does not reproduce"
                )
            if int(scale.max()) == 255:
                raise ValueError("MXFP8 scale byte 255 is NaN (E8M0)")
            scales = tile_scales(scale, BLOCK_COLUMNS[MXFP8])
            return cls(MXFP8, weight.shape[2], weight.view(torch.uint8), scales)
        amax, denominator = amax.float(), denominator.float().reshape(1)
        if not bool(torch.isfinite(amax / denominator).all()):
            raise ValueError("NVFP4 amax / denominator must be finite")
        if bool(scale.float().isnan().any()):
            raise ValueError("NVFP4 scales must not be NaN")
        scales = tile_scales(scale, BLOCK_COLUMNS[NVFP4])
        columns = 2 * weight.shape[2]
        return cls(NVFP4, columns, weight.view(torch.uint8), scales, amax, denominator)

    def select(self, experts: torch.Tensor) -> "QuantExperts":
        """The experts ``experts`` (indices), in that order."""
        amax = None if self.amax is None else self.amax[experts]
        return dataclasses.replace(
            self, weight=self.weight[experts], scale=self.scale[experts], amax=amax
        )

    def checkpoint_tensors(self) -> tuple:
        """``(weight, scale, amax, denominator)`` as the checkpoint stores them
        (``ref.dequantize_experts``' arguments)."""
        blocks = self.columns // BLOCK_COLUMNS[self.format]
        scale = untile_scales(self.scale, blocks)
        if self.format == MXFP8:
            return self.weight.view(torch.float8_e4m3fn), scale, None, None
        return self.weight, scale.view(torch.float8_e4m3fn), self.amax, self.denominator

    def dequantize(self, deinterleave: bool = False) -> torch.Tensor:
        """bf16 ``(experts, rows, columns)`` weights with HF's numerics, ``ref.dequantize_experts``
        of the checkpoint tensors. Checkpoint row order, or with ``deinterleave`` gate rows then
        up rows (``ref.load_inkling_weights``' ``w13``)."""
        out = ref.dequantize_experts(*self.checkpoint_tensors())
        return ref.deinterleave_gate_up(out) if deinterleave else out

    def block_values(self, deinterleave: bool = False) -> torch.Tensor:
        """bf16 ``(experts, rows, columns)`` block values, the weights of ``ref.ExpertQuant``
        numerics and of the kernels' decode: NVFP4 ``e2m1 x scale`` (exact, without the
        per-expert factor), MXFP8 the weights (``dequantize``). Row order as ``dequantize``'s."""
        out = ref.dequantize_experts(*self.checkpoint_tensors(), global_scale=False)
        return ref.deinterleave_gate_up(out) if deinterleave else out


def random_checkpoint_experts(
    fmt: str,
    experts: int,
    rows: int,
    columns: int,
    seed: int,
    mx_scales: tuple[int, int] = (MIN_MX_SCALE, 254),
    device="cuda",
) -> tuple:
    """Random checkpoint-format expert tensors ``(weight, scale, amax, denominator)`` as
    ``ref.dequantize_experts`` takes them, over every non-NaN code: NVFP4 with every non-NaN
    fp8 scale byte (negative and subnormal ones included) and amax in [0.1, 0.6); MXFP8 with
    scale bytes in ``mx_scales`` (inclusive; large ones overflow to inf; the checkpoint's are
    113 .. 118)."""
    gen = torch.Generator(device=device).manual_seed(seed)

    def random_bytes(*shape, low=0, high=256):
        return torch.randint(low, high, shape, generator=gen, device=device, dtype=torch.uint8)

    if fmt == NVFP4:
        scale = random_bytes(experts, rows, columns // 16)
        scale[(scale & 0x7F) == 0x7F] = 0x38  # e4m3 NaN -> 1.0
        amax = torch.rand(experts, generator=gen, device=device) * 0.5 + 0.1
        weight = random_bytes(experts, rows, columns // 2)
        denominator = torch.tensor([2688.0], device=device)
        return weight, scale.view(torch.float8_e4m3fn), amax, denominator
    weight = random_bytes(experts, rows, columns)
    weight[(weight & 0x7F) == 0x7F] = 0x3C  # e4m3 NaN codes -> 1.5
    scale = random_bytes(experts, rows, columns // 32, low=mx_scales[0], high=mx_scales[1] + 1)
    return weight.view(torch.float8_e4m3fn), scale, None, None


def load_quantized_inkling_weights(
    path_or_repo: str | Path = "thinkingmachines/Inkling-2-Mini",
    device="cuda",
    activation_quant: bool | None = None,
) -> ref.InklingWeights:
    """``ref.load_inkling_weights`` with every layer's ``w13``/``w2`` a ``QuantExperts`` in
    checkpoint row order (bf16 checkpoint experts, if any, are rejected); ``activation_quant``
    as ``ref.load_inkling_weights``'."""
    return ref.load_inkling_weights(
        path_or_repo, device, QuantExperts.from_checkpoint, activation_quant
    )


def dense_layer(w: ref.LayerWeights, experts: torch.Tensor | None = None) -> ref.LayerWeights:
    """``w`` with bf16 experts in ``ref.load_inkling_weights``' layout: its ``QuantExperts``
    dequantized, or their block values for declared numerics (only ``experts``, in that order,
    if given, with their ``ExpertQuant``), dense ones as they are."""
    if isinstance(w.w13, torch.Tensor):
        return w

    def chosen(per_expert):  # QuantExperts or ExpertQuant
        return per_expert if experts is None else per_expert.select(experts)

    w13, w2 = chosen(w.w13), chosen(w.w2)
    if not w.declared_experts:
        return dataclasses.replace(w, w13=w13.dequantize(deinterleave=True), w2=w2.dequantize())
    return dataclasses.replace(
        w,
        w13=w13.block_values(deinterleave=True),
        w2=w2.block_values(),
        w13_quant=chosen(w.w13_quant),
        w2_quant=chosen(w.w2_quant),
    )


def dequantized_weights(weights: ref.InklingWeights) -> ref.InklingWeights:
    """``weights`` with every layer's experts in bf16 (``dense_layer``)."""
    return dataclasses.replace(weights, layers=tuple(dense_layer(w) for w in weights.layers))


def experts_out(config: ref.InklingConfig, w: ref.LayerWeights, x, experts, weights):
    """``ref.experts_out`` on a layer whose experts may be ``QuantExperts``: the routed experts
    decoded (``dense_layer``; ascending, so the sum keeps the oracle's order) give the same
    result."""
    if isinstance(w.w13, torch.Tensor):
        return ref.experts_out(config, w, x, experts, weights)
    used = experts.unique()
    local = torch.searchsorted(used, experts)
    return ref.experts_out(config, dense_layer(w, used), x, local, weights)


QUANTIZED = ref.Kernels(experts=experts_out)


def forward_tokens(
    weights: ref.InklingWeights, tokens: torch.Tensor, pos: int, cache: ref.InklingCache
) -> torch.Tensor:
    """``ref.forward_tokens`` (the last token's fp32 logits) on a model whose experts may be
    ``QuantExperts`` (``experts_out``), with the oracle's results."""
    return ref.forward_tokens(weights, tokens, pos, cache, kernels=QUANTIZED)
