"""The expert activation quantization an Inkling 2 Mini checkpoint declares, as vLLM's SM100
path implements it, bit for bit: quantizers, block values, and the declaration's parser.

Torch only. ``hf_quant_config.json`` declares per expert tensor a
``required_activation_quantization`` of its input: for the NVFP4 experts (layers 0-19) per-token
amax, 16-column blocks of ``nvfp4_e2m1`` with ``float8_e4m3fn`` scales chosen by the
``fouroversix_mae`` rule; for the MXFP8 experts (layers 20-23) 32-column blocks of
``mxfp8_e4m3`` with ``ue8m0`` scales; both round to nearest. HF transformers
(``add-inkling2-mini``) does not quantize activations, and vLLM's Hopper path (``moe_sm90.py``)
runs bf16 activations; this follows vLLM main bf151b7e's Blackwell kernels,
``vllm/models/inkling2/nvidia/ops/{nvfp4,mxfp8}.py``, whose file:line references below are to
that commit.

NVFP4 (``nvfp4_quantize``) of a row ``x`` (fp32 values of bf16 activations):

- ``amax = max |x|`` over the row (``nvfp4.py:228-233``), ``encoder = min(1536 / amax,
  FP32_MAX)``, ``decoder = 1 / encoder`` and ``outer = amax * fp32(1 / 1536)``
  (``nvfp4.py:235-237``): the row maximum's block gets the 6.0-based scale 256 (1536 = 6 x 256).
- Per 16-column block with ``m = max |x|`` two candidates, ``s6 = e4m3(min(m * (encoder *
  fp32(1 / 6)), 448))`` and ``s4 = e4m3(min(m * (encoder * 0.25), 448))``, each with ``inverse =
  min(1 / (s * decoder), FP32_MAX)`` (``nvfp4.py:95-104``). The values ``x * inverse`` round to
  E2M1 to nearest even, saturating at 6 (``cvt.rn.satfinite.e2m1x2.f32``, ``nvfp4.py:63-81``).
  Each candidate's error is ``sum |(e2m1 * s) * outer - x|`` in fp32, summed as a pairwise tree
  over adjacent columns (``nvfp4.py:84-91``); the 4.0-based candidate is kept only when its
  error is strictly smaller (``nvfp4.py:107-126``).
- Every division is IEEE (``tl.div_rn``) and nothing fuses into FMAs (``enable_fp_fusion=False``,
  ``nvfp4.py:281``), as in this module's separate torch operations.

MXFP8 (``mxfp8_quantize``), per 32-column block: ``raw = max |x| * fp32(1 / 448)``, the scale
the power of two at or above ``raw`` as a biased exponent byte (0 when ``raw <= 2^-127``, at most
254, 255 for NaN), and the values ``x * rcp(scale)`` clamped to +-448 and rounded to e4m3 to
nearest even (``mxfp8.py:13-30``). Triton's ``rcp_rn`` flushes the subnormal ``rcp(2^127)`` to
zero, so an infinity's block (exponent 254) quantizes to zeros and, for the infinity, -448.

Both formats' block values (``e2m1 x e4m3`` and ``e4m3 x 2^(e - 127)``) are bf16 numbers
exactly (``nvfp4_block_values``, ``mxfp8_block_values``); NVFP4's per-token ``amax`` scales the
GEMM's fp32 output instead (``ref.experts_out``). Maxima ignore NaNs, as Triton's
``maximum`` does; a NaN rounds to E2M1 6 of its sign bit and, in MXFP8, to -448.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from attn_gym.warpdrive.examples.megakernel.formats import MXFP8, NVFP4

FP32_MAX = torch.finfo(torch.float32).max
E4M3_MAX = 448.0
NVFP4_GLOBAL = 1536.0  # the largest E2M1 value 6 times the e4m3 scale 256
# E2M1's 8 magnitudes and the upper bounds of the inputs rounding to each but the last, with
# whether a tie at the bound rounds down (to an even code).
E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_BOUNDS = ((0.25, True), (0.75, False), (1.25, True), (1.75, False), (2.5, True),
               (3.5, False), (5.0, True))  # fmt: skip
QUANT_CONFIG = "hf_quant_config.json"
# The ``required_activation_quantization`` of Inkling 2 Mini's NVFP4 and MXFP8 experts.
DECLARATIONS = {
    NVFP4: {
        "amax_type": "per_token",
        "block_size": 16,
        "format": "nvfp4_e2m1",
        "rounding": "nearest",
        "scale_dtype": "float8_e4m3fn",
        "scale_selection_rule": "fouroversix_mae",
    },
    MXFP8: {
        "block_size": 32,
        "format": "mxfp8_e4m3",
        "rounding": "nearest",
        "scale_dtype": "ue8m0",
    },
}


@dataclass(frozen=True)
class ActivationQuant:
    """The quantization of an expert matrix's input: ``format`` (``formats.NVFP4`` or
    ``formats.MXFP8``) with the declared block size, rounding and scale rule."""

    format: str

    @property
    def block_size(self) -> int:
        return {NVFP4: 16, MXFP8: 32}[self.format]

    @classmethod
    def from_declaration(cls, declared: dict) -> "ActivationQuant":
        """The ``required_activation_quantization`` entry of ``hf_quant_config.json``; raises
        for anything but the two schemes this module implements (``DECLARATIONS``)."""
        for fmt, scheme in DECLARATIONS.items():
            if declared == scheme:
                return cls(fmt)
        raise ValueError(f"unsupported activation quantization {declared}")


def declared_activation_quant(checkpoint: Path) -> dict[str, ActivationQuant]:
    """Tensor name -> the quantization its input requires, from the checkpoint directory's
    ``hf_quant_config.json`` (the main model's parameter groups); empty without the file."""
    path = Path(checkpoint) / QUANT_CONFIG
    if not path.exists():
        return {}
    declared = {}
    for group in json.loads(path.read_text())["model"]["parameter_groups"]:
        spec = group.get("required_activation_quantization")
        if spec is not None:
            declared |= dict.fromkeys(group["tensors"], ActivationQuant.from_declaration(spec))
    return declared


def nan_max(x: torch.Tensor, dim: int) -> torch.Tensor:
    """``max`` over ``dim`` ignoring NaNs (NaN only where all are), as Triton's reductions."""
    m = torch.where(x.isnan(), -torch.inf, x).amax(dim)
    return torch.where(m == -torch.inf, torch.nan, m)


def round_e2m1(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(codes, values)`` of fp32 ``q`` rounded to E2M1 to nearest even with saturation:
    uint8 codes with the sign in bit 3, and their fp32 values. NaNs round to 6 of their sign,
    as vLLM's software rounding (``nvfp4.py:24-60``) and ``cvt.rn.satfinite`` do."""
    a = q.abs()
    code = torch.zeros(q.shape, dtype=torch.uint8, device=q.device)
    for bound, tie_down in E2M1_BOUNDS:
        code += ~((a <= bound) if tie_down else (a < bound))  # NaN passes every bound
    negative = q.signbit()
    values = e2m1_magnitude(code)
    return code | (negative.to(torch.uint8) << 3), torch.where(negative, -values, values)


def e2m1_magnitude(code: torch.Tensor) -> torch.Tensor:
    """fp32 magnitudes ``E2M1_MAGNITUDES`` of codes 0-7, computed on their device (no host
    table: CUDA-graph capturable): 0.5 c below 2, else ``2^(c // 2 - 1) (1 + (c % 2) / 2)``."""
    c = code.float()
    normal = torch.exp2(torch.floor(c / 2) - 1) * (1 + (c - 2 * torch.floor(c / 2)) / 2)
    return torch.where(c < 2, c * 0.5, normal)


def pairwise_sum16(x: torch.Tensor) -> torch.Tensor:
    """fp32 sum of the last axis (16) as a pairwise tree over adjacent elements."""
    while x.shape[-1] > 1:
        x = x[..., 0::2] + x[..., 1::2]
    return x[..., 0]


@dataclass(frozen=True)
class NVFP4Activation:
    """Rows quantized to NVFP4 in vLLM's layout: ``values`` uint8 ``(rows, K / 2)`` E2M1 code
    pairs (the even column in the low nibble), ``scales`` e4m3 ``(rows, K / 16)`` and ``amax``
    fp32 ``(rows,)``."""

    values: torch.Tensor
    scales: torch.Tensor
    amax: torch.Tensor


def nvfp4_quantize(x: torch.Tensor) -> NVFP4Activation:
    """vLLM's ``nvfp4_quantize`` (``nvfp4.py:256``) of bf16 ``(rows, K)`` ``x``, bit for bit."""
    x = x.float()
    rows, columns = x.shape
    amax = nan_max(x.abs(), -1)
    amax = torch.where(amax.isnan(), 0.0, amax)  # nvfp4.py:228: a zero-initialized max
    # torch.div: ``float / tensor`` would multiply by the reciprocal, not divide.
    encoder = torch.fmin(torch.div(NVFP4_GLOBAL, amax), torch.tensor(FP32_MAX))
    decoder = 1.0 / encoder
    outer = amax * (1.0 / NVFP4_GLOBAL)
    blocks = x.view(rows, columns // 16, 16)
    block_max = nan_max(blocks.abs(), -1)
    candidates = []
    for factor in (1.0 / 6.0, 0.25):  # nvfp4.py:98-103
        scale = torch.fmin(block_max * (encoder * factor)[:, None], torch.tensor(E4M3_MAX))
        scale = scale.to(torch.float8_e4m3fn).float()
        inverse = torch.fmin(1.0 / (scale * decoder[:, None]), torch.tensor(FP32_MAX))
        codes, rounded = round_e2m1(blocks * inverse[..., None])
        reconstructed = (rounded * scale[..., None]) * outer[:, None, None]
        error = pairwise_sum16((reconstructed - blocks).abs())
        candidates.append((codes, scale, error))
    (codes6, scale6, error6), (codes4, scale4, error4) = candidates
    choose4 = error4 < error6
    codes = torch.where(choose4[..., None], codes4, codes6).view(rows, columns)
    scales = torch.where(choose4, scale4, scale6).to(torch.float8_e4m3fn)
    return NVFP4Activation(codes[:, 0::2] | (codes[:, 1::2] << 4), scales, amax)


def nvfp4_block_values(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """bf16 ``(rows, K)`` ``e2m1 x e4m3`` of NVFP4 code pairs and block scales: exact (at most
    6 significant bits, magnitudes in [2^-10, 2688])."""
    codes = torch.stack((values & 0xF, values >> 4), dim=-1).flatten(-2)
    e2m1 = e2m1_magnitude(codes & 7)
    e2m1 = torch.where(codes >= 8, -e2m1, e2m1)
    blocks = e2m1.unflatten(-1, (-1, 16)) * scales.float()[..., None]
    return blocks.flatten(-2).to(torch.bfloat16)


def mxfp8_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM's ``mxfp8_quantize`` (``mxfp8.py:62``) of bf16 ``(rows, K)`` ``x``, bit for bit:
    e4m3 ``(rows, K)`` values and uint8 ``(rows, K / 32)`` biased scale exponents."""
    rows, columns = x.shape
    groups = x.float().view(rows, columns // 32, 32)
    raw = nan_max(groups.abs(), -1) * (1.0 / E4M3_MAX)
    bits = raw.view(torch.int32)
    exponent = ((bits + 0x007FFFFF) & 0x7F800000) >> 23  # round up to a power of two
    exponent = torch.where(raw <= 2.0**-127, 0, exponent.clamp(max=254))
    exponent = torch.where(raw.isnan(), 255, exponent)
    scale = ue8m0_scale(exponent)
    # rcp_rn flushes the subnormal 1 / 2^127 (exponent 254, only reached by infinities) to 0.
    inverse = torch.where(exponent == 254, 0.0, 1.0 / scale)
    inverse = torch.where(exponent == 0, 2.0**127, inverse)
    inverse = torch.where(exponent == 255, 0.0, inverse)
    values = groups * inverse[..., None]
    # mxfp8.py:24-26: a NaN-dropping maximum, then a NaN-propagating minimum.
    values = torch.fmax(values, torch.tensor(-E4M3_MAX)).clamp(max=E4M3_MAX)
    return values.view(rows, columns).to(torch.float8_e4m3fn), exponent.to(torch.uint8)


def ue8m0_scale(exponent: torch.Tensor) -> torch.Tensor:
    """fp32 ``2^(e - 127)`` of ue8m0 bytes ``exponent``, from its bits (``e`` 0: the subnormal
    ``2^-127``; 255: infinity), exact on every device (``torch.exp2`` need not be)."""
    return torch.fmax((exponent.int() << 23).view(torch.float32), torch.tensor(2.0**-127))


def mxfp8_block_values(values: torch.Tensor, exponents: torch.Tensor) -> torch.Tensor:
    """bf16 ``(rows, K)`` ``e4m3 x 2^(e - 127)``: exact (e4m3 values are multiples of 2^-9 with
    at most 4 significant bits) unless the scale is below 2^-124 (``e < 3``), which needs a
    block maximum below 448 x 2^-125."""
    blocks = values.float().unflatten(-1, (-1, 32)) * ue8m0_scale(exponents)[..., None]
    return blocks.flatten(-2).to(torch.bfloat16)


def quantize_dequantize(
    quant: ActivationQuant, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """bf16 ``(rows, K)`` ``x`` quantized per ``quant`` and decoded: the bf16 block values and,
    for NVFP4, each row's fp32 ``amax`` (the value of block value 1536 is ``amax``, so
    ``x ~ block values * amax / 1536``). MXFP8 has no global scale (None)."""
    if quant.format == NVFP4:
        q = nvfp4_quantize(x)
        return nvfp4_block_values(q.values, q.scales), q.amax
    return mxfp8_block_values(*mxfp8_quantize(x)), None
