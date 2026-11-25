"""The random tiny Inkling 2 Mini of the oracle's HF parity tests, saved in the checkpoint
naming. Building it needs transformers' Inkling2Loop: ``tiny_models`` skips the calling test
otherwise, so modules that only sometimes need it still collect without it.

The tiny shape keeps every structural feature: sliding and full layers, block boundaries inside
and outside the loop, three passes with messages, a sliding window and relative extent a
24-token prompt outgrows, and enough tokens to wrap the convolution rings.
"""

import json
import shutil
from pathlib import Path

import pytest
import torch

from attn_gym.warpdrive.examples.inkling_megakernel.act_quant import DECLARATIONS
from attn_gym.warpdrive.examples.inkling_megakernel.quant import random_checkpoint_experts
from attn_gym.warpdrive.examples.inkling_megakernel.ref import load_inkling_weights
from attn_gym.warpdrive.examples.megakernel.formats import MXFP8, NVFP4


def transformers_loop():
    """transformers' Inkling2Loop modeling module; skips the calling test without it."""
    return pytest.importorskip(
        "transformers.models.inkling2_loop.modeling_inkling2_loop",
        reason="needs transformers with Inkling2Loop (huggingface/transformers add-inkling2-mini)",
    )


def tiny_config():
    """Two layers per block, so the four-layer core spans two blocks as in the real model (12
    layers of 6): passes after the first then read messages from a completed block."""
    from transformers import Inkling2LoopTextConfig

    return Inkling2LoopTextConfig(
        hidden_size=256, num_hidden_layers=8, layer_types=["hybrid_sliding", "hybrid"] * 4,
        num_attention_heads=6, num_key_value_heads=2, head_dim=64,
        swa_num_attention_heads=6, swa_num_key_value_heads=2, swa_head_dim=64,
        sliding_window_size=8, rel_extent=16, d_rel=16, log_scaling_n_floor=16,
        vocab_size=512, unpadded_vocab_size=500, eos_token_id=1,
        moe_intermediate_size=128, n_routed_experts=16, num_experts_per_tok=4,
        n_layers_per_attn_res_block=2, loop_layer_index_start=2, loop_layer_index_end=6,
        loop_num_passes=3, loop_gate_rank=16,
    )  # fmt: skip


def tiny_models(path: Path):
    """``(hf, weights)``: the random tiny model saved under ``path``, as HF's eager
    ``Inkling2LoopForCausalLM`` and as the oracle's weights."""
    model_class = transformers_loop().Inkling2LoopForCausalLM
    torch.manual_seed(0)
    model = model_class(tiny_config())
    with torch.no_grad():  # zero-initialized or tiny parameters would hide their wiring
        for name, param in model.named_parameters():
            if "message_gate" in name:  # gates near 0 would barely mix in the messages
                param.normal_(0, 2 * param.shape[-1] ** -0.5)
            elif param.abs().max() == 0:
                param.normal_(0, 0.5)
            elif "norm" in name or name.endswith("scale"):
                param.add_(torch.randn_like(param) * 0.2)
    model.to(torch.bfloat16).save_pretrained(path)
    hf = model_class.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="eager", experts_implementation="eager"
    )
    return hf.cuda().eval(), load_inkling_weights(path)


def quantized_tiny_checkpoint(source: Path, path: Path, nvfp4_layers: int) -> Path:
    """A copy of the tiny checkpoint ``source`` in ``path`` with random quantized experts as
    the real checkpoint stores them, NVFP4 in the first ``nvfp4_layers`` layers and MXFP8 in
    the others, and an ``hf_quant_config.json`` declaring their activation quantization."""
    from safetensors.torch import load_file, save_file

    shutil.copytree(source, path)
    tensors = load_file(str(path / "model.safetensors"))
    groups = {NVFP4: [], MXFP8: []}
    layers = json.loads((path / "config.json").read_text())["num_hidden_layers"]
    for layer in range(layers):
        fmt = NVFP4 if layer < nvfp4_layers else MXFP8
        for seed, name in enumerate(("w13_weight", "w2_weight")):
            prefix = f"model.llm.layers.{layer}.mlp.experts.{name}"
            experts, rows, columns = tensors[prefix].shape
            weight, scale, amax, denominator = random_checkpoint_experts(
                fmt, experts, rows, columns, seed=2 * layer + seed, mx_scales=(110, 116),
                device="cpu",
            )  # fmt: skip
            parts = {"": weight, "_scale": scale}
            if fmt == NVFP4:
                parts |= {"_amax": amax, "_outer_scale_denominator": denominator}
            tensors |= {prefix + suffix: t.contiguous() for suffix, t in parts.items()}
            groups[fmt].append(prefix)
    save_file(tensors, str(path / "model.safetensors"), metadata={"format": "pt"})
    config = {
        "model": {
            "parameter_groups": [
                {"tensors": names, "required_activation_quantization": DECLARATIONS[fmt]}
                for fmt, names in groups.items()
            ]
        }
    }
    (path / "hf_quant_config.json").write_text(json.dumps(config))
    return path
