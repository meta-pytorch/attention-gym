"""The Inkling 2 Mini quantized experts and declared expert numerics on the host side: the
loader's kernel layout (``quant.QuantExperts``) decodes to ``ref.dequantize_experts``, the
oracle's activation quantizers (``act_quant``) on hand-derived cases, the oracle's MoE with
declared numerics against an fp64 formulation, and ``quant.experts_out`` on a quantized layer
against the dense layer. The device quantizers are tested in
``test_warpdrive_megakernel_act_quant``, the visit launch on quantized experts in
``test_warpdrive_inkling_megakernel_visit``."""

import pytest
import torch

from attn_gym.warpdrive.examples.inkling_megakernel import act_quant, quant, ref
from attn_gym.warpdrive.examples.inkling_megakernel.act_quant import (
    ActivationQuant,
    mxfp8_block_values,
    mxfp8_quantize,
    nvfp4_block_values,
    nvfp4_quantize,
)
from attn_gym.warpdrive.examples.megakernel.formats import MXFP8, NVFP4

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.usefixtures("release_cuda_memory"),
]


@pytest.mark.parametrize("fmt", [NVFP4, MXFP8])
def test_layout_round_trip(fmt):
    """Every non-NaN code and scale byte (NVFP4 negative and subnormal scales, MXFP8 scales up
    to overflow) and the expert width's 4.5-subtile row: the kernel layout decodes to the
    oracle's bf16 weights and block values bit for bit (signed zeros and MXFP8 infinities
    included)."""
    tensors = quant.random_checkpoint_experts(fmt, 4, 64, 1152, seed=0)
    q = quant.QuantExperts.from_checkpoint(*tensors)
    assert q.format == fmt and q.columns == 1152
    for got, global_scale in ((q.dequantize(), True), (q.block_values(), False)):
        expected = ref.dequantize_experts(*tensors, global_scale=global_scale)
        assert torch.equal(got.view(torch.int16), expected.view(torch.int16))


def test_quantizers_on_hand_derived_cases():
    """``cvt.rn.satfinite.e2m1x2`` rounding: ties to the even code, saturation beyond 6, the sign
    bit kept. NVFP4 with amax 1536 (global scale 1): a block both candidates represent exactly
    keeps the 6.0-based scale, one only the 4.0-based scale represents exactly takes it, and the
    block values reconstruct the row. MXFP8: the block scale is the power of two at or above
    amax / 448; zero blocks get exponent 0."""
    q = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0, 1e30, -0.0, -0.3, float("nan")])
    codes, values = act_quant.round_e2m1(q.cuda())
    assert codes.tolist() == [0, 2, 2, 4, 4, 6, 6, 7, 7, 8, 9, 7]
    assert values.tolist()[:-1] == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0, -0.0, -0.5]

    x = torch.zeros(1, 48)
    x[0, 0] = 1536.0
    x[0, 16] = 6.0  # both exact: 6 x 1.0 and 4 x 1.5
    x[0, 32:34] = torch.tensor([6.0, 4.5])  # 4.5 = 3 x 1.5; with scale 1.0 it rounds to 4
    nvfp4 = nvfp4_quantize(x.cuda().bfloat16())
    assert nvfp4.amax.item() == 1536.0
    assert nvfp4.scales.float().tolist() == [[256.0, 1.0, 1.5]]
    torch.testing.assert_close(nvfp4_block_values(nvfp4.values, nvfp4.scales).float().cpu(), x)

    x = torch.zeros(1, 96)
    x[0, 0], x[0, 32], x[0, 33] = 448.0, 450.0, -3.0
    values, exponents = mxfp8_quantize(x.cuda().bfloat16())
    assert exponents.tolist() == [[127, 128, 0]]
    decoded = mxfp8_block_values(values, exponents).float().cpu()
    assert decoded[0, :34:32].tolist() == [448.0, 448.0]  # 450 / 2 rounds to e4m3 224
    assert decoded[0, 33] == -3.0


def random_layer(fmt: str, experts=16, hidden=256, width=128, seed=0):
    """A layer of random checkpoint-format experts with declared numerics (bf16 block values)
    and ``ExpertQuant``, as ``ref.load_inkling_weights`` loads it, and its checkpoint tensors."""
    tensors = [
        quant.random_checkpoint_experts(fmt, experts, rows, columns, seed + i, (110, 116))
        for i, (rows, columns) in enumerate(((2 * width, hidden), (hidden, width)))
    ]
    numerics = [
        ref.ExpertQuant(
            ActivationQuant(fmt),
            None if amax is None else amax.float(),
            None if den is None else den.float(),
        )
        for _, _, amax, den in tensors
    ]
    w13 = ref.deinterleave_gate_up(ref.dequantize_experts(*tensors[0], global_scale=False))
    w2 = ref.dequantize_experts(*tensors[1], global_scale=False)
    fields = dict.fromkeys(ref.LayerWeights.__dataclass_fields__, None)
    w = ref.LayerWeights(
        **fields | {"w13": w13, "w2": w2, "w13_quant": numerics[0], "w2_quant": numerics[1]}
    )
    return w, tensors


def fp64_experts_out(config, w: ref.LayerWeights, x, experts, weights):
    """The declared numerics in float64 from fully scaled values: the activation quantized and
    dequantized (``x ~ values * amax / 1536``), weights times their per-expert factor, the
    routing weight on the SwiGLU before the down input's quantization; rounded to bf16 only
    where vLLM's GEMMs and SwiGLU write bf16."""

    def dequantized(quant_, rows):
        values, amax = act_quant.quantize_dequantize(quant_.activation, rows)
        values = values.double()
        return values if amax is None else values * (amax.double() / 1536.0)[:, None]

    def factor(quant_, e):
        if quant_.weight_amax is None:
            return 1.0
        return quant_.weight_amax[e].double() / quant_.weight_denominator.double()

    out = torch.zeros(x.shape, dtype=torch.float64, device=x.device)
    for e in experts.unique().tolist():
        token, slot = (experts == e).nonzero(as_tuple=True)
        w13 = w.w13[e].double() * factor(w.w13_quant, e)
        gate_up = (dequantized(w.w13_quant, x[token]) @ w13.T).to(torch.bfloat16)
        act = ref.swiglu(config, gate_up, weights[token, slot, None]).to(torch.bfloat16)
        w2 = w.w2[e].double() * factor(w.w2_quant, e)
        out.index_add_(0, token, (dequantized(w.w2_quant, act) @ w2.T).to(torch.bfloat16).double())
    return out


@pytest.mark.parametrize("fmt", [NVFP4, MXFP8])
def test_declared_experts_match_fp64_formulation(fmt):
    """``ref.experts_out`` with declared numerics against ``fp64_experts_out``: only the fp32
    dot products and fp32 row-scale products differ, so the results agree to bf16 rounding;
    a lost or misplaced global scale, the routing weight applied after the down projection or
    the activations left unquantized miss by far more."""
    w, _ = random_layer(fmt)
    config = ref.InklingConfig(hidden_size=256, num_experts=16, experts_per_token=4)
    gen = torch.Generator(device="cuda").manual_seed(1)
    x = (torch.randn(24, 256, device="cuda", generator=gen) * 3).bfloat16()
    experts = torch.stack(
        [torch.randperm(16, device="cuda", generator=gen)[:4] for _ in range(24)]
    )
    weights = torch.rand(24, 4, device="cuda", generator=gen) * 2
    out = ref.experts_out(config, w, x, experts, weights).double()
    expected = fp64_experts_out(config, w, x, experts, weights)
    error = (out - expected).norm() / expected.norm()
    assert error < 2e-3, error
    # The same experts with the routing weight applied after the down projection (HF's place).
    moved = torch.zeros_like(expected)
    for slot in range(4):
        ones = torch.ones_like(weights)
        y = ref.experts_out(config, w, x, experts[:, slot : slot + 1], ones[:, :1]).double()
        moved += y * weights[:, slot, None]
    assert (moved - expected).norm() / expected.norm() > 5 * error


def test_quantized_layer_matches_dense_declared_layer():
    """``quant.experts_out`` on ``QuantExperts`` with declared numerics (each MLP's routed
    experts decoded to block values, their weight scales selected) equals ``ref.experts_out``
    on the dense layer bit for bit."""
    w, tensors = random_layer(NVFP4, seed=3)
    q = ref.LayerWeights(
        **vars(w) | {"w13": quant.QuantExperts.from_checkpoint(*tensors[0]),
                     "w2": quant.QuantExperts.from_checkpoint(*tensors[1])}
    )  # fmt: skip
    config = ref.InklingConfig(hidden_size=256, num_experts=16, experts_per_token=4)
    gen = torch.Generator(device="cuda").manual_seed(4)
    x = torch.randn(5, 256, device="cuda", generator=gen).bfloat16()
    experts = torch.stack([torch.randperm(16, device="cuda", generator=gen)[:4] for _ in range(5)])
    weights = torch.rand(5, 4, device="cuda", generator=gen)
    expected = ref.experts_out(config, w, x, experts, weights)
    assert torch.equal(quant.experts_out(config, q, x, experts, weights), expected)
    assert torch.equal(quant.dense_layer(q).w13, w.w13)
