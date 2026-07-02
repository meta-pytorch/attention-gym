"""End-to-end FastWan2.1 (DMD, VSA-trained) inference with FlexAttention VSA.

Runs FastVideo/FastWan2.1-T2V-1.3B-Diffusers (3-step DMD, trained with VSA at 0.8
sparsity) and swaps the self-attention implementation per run:

- ``dense``: plain SDPA (quality/timing reference, no sparsity).
- ``upstream``: FastVideo's `vsa` package kernel at (4,4,4) 64-token cubes
  (Triton fallback on non-SM90 GPUs; requires ``pip install vsa``).
- ``flex``: attention-gym VSA on the FA4 Flex FLASH backend at (16,4,4)
  256-token cubes, compiled as one region (coarse attn + topk +
  ``create_vsa_flash_block_mask`` + ``flex_attention`` + gated combine).

All modes use the checkpoint's learned per-token compress gate
(``blocks.N.to_gate_compress``), which diffusers drops when loading the
transformer, so it is re-attached from the safetensors file here.

Example:
    python examples/fastwan_vsa.py --mode flex --output_dir outputs/
"""

import math
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention.flex_attention import flex_attention

from attn_gym.masks import (
    compute_vsa_coarse_attention,
    create_vsa_flash_block_mask,
    create_vsa_tile_metadata,
    vsa_additive_combine,
    vsa_topk_from_sparsity,
)

MODEL_ID = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"
DMD_TIMESTEPS = (1000, 757, 522)
FLOW_SHIFT = 3.0
VSA_SPARSITY = 0.8
UPSTREAM_TILE = (4, 4, 4)
FLEX_TILE = (16, 4, 4)


def flow_sigma(timestep: float) -> float:
    """Sigma for FastVideo's DMD timesteps.

    FastVideo's FlowMatchEuler scheduler defines ``timesteps = sigmas * 1000``
    after applying the flow shift, so the DMD timesteps (1000, 757, 522) already
    live in shifted-sigma space and map back linearly.
    """
    return timestep / 1000.0


def load_gate_weights(model_dir: Path, device, dtype) -> dict[int, tuple[Tensor, Tensor]]:
    """Load the learned VSA compress-gate projections that diffusers ignores."""
    from safetensors import safe_open

    gates: dict[int, tuple[Tensor, Tensor]] = {}
    with safe_open(model_dir / "transformer" / "diffusion_pytorch_model.safetensors", "pt") as f:
        keys = [k for k in f.keys() if "to_gate_compress.weight" in k]
        for key in keys:
            layer = int(key.split(".")[1])
            weight = f.get_tensor(key).to(device=device, dtype=dtype)
            bias = f.get_tensor(key.replace("weight", "bias")).to(device=device, dtype=dtype)
            gates[layer] = (weight, bias)
    if not gates:
        raise RuntimeError("checkpoint has no to_gate_compress weights; not a VSA model")
    return gates


@dataclass
class VSALayout:
    """Tile-major permutation for an exactly-divisible latent grid."""

    tile_numel: int
    num_tiles: int
    perm: Tensor
    inverse_perm: Tensor

    @classmethod
    def build(cls, grid: tuple[int, int, int], tile: tuple[int, int, int], device) -> "VSALayout":
        metadata = create_vsa_tile_metadata(grid, tile, device=device)
        if metadata.padded_seq_length != metadata.total_seq_length:
            raise ValueError(f"grid {grid} does not divide into tiles {tile}")
        return cls(
            tile_numel=metadata.tile_numel,
            num_tiles=math.prod(metadata.num_tiles),
            perm=metadata.tile_partition_indices,
            inverse_perm=metadata.reverse_tile_partition_indices,
        )


def import_upstream_vsa():
    """Import the installed `vsa` package, dodging the examples/vsa.py shadow.

    Running ``python examples/fastwan_vsa.py`` puts ``examples/`` at
    ``sys.path[0]``, where ``vsa.py`` (the FlexAttention example) shadows the
    upstream FastVideo ``vsa`` package.
    """
    import importlib
    import sys

    module = sys.modules.get("vsa")
    if module is not None and hasattr(module, "video_sparse_attn"):
        return module
    examples_dir = str(Path(__file__).parent.resolve())
    saved_path = sys.path.copy()
    sys.modules.pop("vsa", None)
    sys.path = [p for p in sys.path if str(Path(p or ".").resolve()) != examples_dir]
    try:
        return importlib.import_module("vsa")
    finally:
        sys.path = saved_path


@lru_cache(maxsize=2)
def compiled_flex_vsa(tile_numel: int, num_tiles: int, top_k: int):
    """One compiled region: coarse attn + topk + block mask build + FLASH fine pass."""

    def flex_vsa(q: Tensor, k: Tensor, v: Tensor, gate: Tensor) -> Tensor:
        coarse = compute_vsa_coarse_attention(q, k, v, tile_numel=tile_numel, top_k=top_k)
        block_mask = create_vsa_flash_block_mask(
            coarse.topk_indices, tile_numel=tile_numel, num_kv_tiles=num_tiles
        )
        fine = flex_attention(q, k, v, block_mask=block_mask, kernel_options={"BACKEND": "FLASH"})
        return vsa_additive_combine(fine, coarse.output, gate, tile_numel=tile_numel)

    return torch.compile(flex_vsa, dynamic=False)


class WanVSASelfAttnProcessor:
    """Drop-in replacement for Wan self-attention with swappable VSA backends.

    Mirrors diffusers' WanAttnProcessor projection/rope path, then runs the
    fine+coarse VSA flow in tile-major order with the checkpoint's learned
    compress gate. ``mode`` selects dense SDPA, the upstream `vsa` kernel, or
    the attention-gym Flex FLASH implementation.
    """

    def __init__(self, mode: str, layer: int, gates, layouts, grid):
        self.mode = mode
        self.layer = layer
        self.gates = gates
        self.layouts = layouts
        self.seq_len = math.prod(grid)
        self.attn_us: list[float] = []

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None
    ):
        query = attn.norm_q(attn.to_q(hidden_states)).unflatten(2, (attn.heads, -1))
        key = attn.norm_k(attn.to_k(hidden_states)).unflatten(2, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            freqs_cos, freqs_sin = rotary_emb
            cos, sin = freqs_cos[..., 0::2], freqs_sin[..., 1::2]

            def apply_rotary(x):
                x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
                out = torch.empty_like(x)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(x)

            query, key = apply_rotary(query), apply_rotary(key)

        q, k, v = (t.transpose(1, 2) for t in (query, key, value))

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if self.mode == "dense":
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            gate_w, gate_b = self.gates[self.layer]
            gate = F.linear(hidden_states, gate_w, gate_b).unflatten(2, (attn.heads, -1))
            layout = self.layouts[self.mode]
            q_t, k_t, v_t, gate_t = (t[:, :, layout.perm] for t in (q, k, v, gate.transpose(1, 2)))
            if self.mode == "upstream":
                upstream_vsa = import_upstream_vsa()

                out = upstream_vsa.video_sparse_attn(
                    q_t.contiguous(),
                    k_t.contiguous(),
                    v_t.contiguous(),
                    torch.full(
                        (layout.num_tiles,), layout.tile_numel, dtype=torch.long, device=q.device
                    ),
                    topk=vsa_topk_from_sparsity(
                        self.seq_len, layout.tile_numel, layout.num_tiles, VSA_SPARSITY
                    ),
                    block_size=UPSTREAM_TILE,
                    compress_attn_weight=gate_t.contiguous(),
                )
            else:
                flex_fn = compiled_flex_vsa(
                    layout.tile_numel,
                    layout.num_tiles,
                    vsa_topk_from_sparsity(
                        self.seq_len, layout.tile_numel, layout.num_tiles, VSA_SPARSITY
                    ),
                )
                out = flex_fn(q_t, k_t, v_t, gate_t)
            out = out[:, :, layout.inverse_perm]
        end.record()
        end.synchronize()
        self.attn_us.append(start.elapsed_time(end) * 1e3)

        hidden_states = out.transpose(1, 2).flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)


@torch.no_grad()
def dmd_denoise(transformer, latents: Tensor, prompt_embeds: Tensor, generator) -> Tensor:
    """FastVideo's 3-step DMD loop: predict flow, step to x0, renoise."""
    for i, t in enumerate(DMD_TIMESTEPS):
        timestep = torch.tensor([float(t)], device=latents.device)
        pred_noise = transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]
        pred_video = latents.double() - flow_sigma(t) * pred_noise.double()
        if i < len(DMD_TIMESTEPS) - 1:
            sigma_next = flow_sigma(DMD_TIMESTEPS[i + 1])
            noise = torch.randn(
                latents.shape, generator=generator, dtype=torch.float64, device=latents.device
            )
            latents = ((1 - sigma_next) * pred_video + sigma_next * noise).to(latents.dtype)
        else:
            latents = pred_video.to(latents.dtype)
    return latents


@torch.no_grad()
def main(
    prompt: str = (
        "A majestic lion strides across the golden savanna at sunset, "
        "its mane flowing in the warm wind, cinematic lighting, photorealistic."
    ),
    mode: str = "flex",
    num_frames: int = 61,
    height: int = 448,
    width: int = 832,
    seed: int = 42,
    output_dir: str = "outputs",
):
    """Generate a FastWan video with the selected self-attention backend."""
    from diffusers import AutoencoderKLWan, WanPipeline, WanTransformer3DModel
    from diffusers.utils import export_to_video
    from huggingface_hub import snapshot_download

    assert mode in ("dense", "upstream", "flex")
    device, dtype = "cuda", torch.bfloat16
    model_dir = Path(snapshot_download(MODEL_ID, ignore_patterns=["assets/*", "examples/*"]))

    transformer = WanTransformer3DModel.from_pretrained(
        model_dir / "transformer", torch_dtype=dtype
    )
    vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(
        model_dir, transformer=transformer, vae=vae, torch_dtype=dtype
    ).to(device)

    latent_t = (num_frames - 1) // pipe.vae_scale_factor_temporal + 1
    grid = (
        latent_t,
        height // pipe.vae_scale_factor_spatial // transformer.config.patch_size[1],
        width // pipe.vae_scale_factor_spatial // transformer.config.patch_size[2],
    )
    processors = {}
    if mode != "dense":
        gates = load_gate_weights(model_dir, device, dtype)
        layouts = {
            "upstream": VSALayout.build(grid, UPSTREAM_TILE, device),
            "flex": VSALayout.build(grid, FLEX_TILE, device),
        }
        for name in transformer.attn_processors:
            if "attn1" in name:
                layer = int(name.split(".")[1])
                processors[name] = WanVSASelfAttnProcessor(mode, layer, gates, layouts, grid)
    else:
        for name in transformer.attn_processors:
            if "attn1" in name:
                processors[name] = WanVSASelfAttnProcessor(mode, -1, None, None, grid)
    transformer.set_attn_processor({**transformer.attn_processors, **processors})

    prompt_embeds, _ = pipe.encode_prompt(
        prompt=prompt, do_classifier_free_guidance=False, device=device
    )
    prompt_embeds = prompt_embeds.to(dtype)

    generator = torch.Generator(device).manual_seed(seed)
    latents = torch.randn(
        (1, transformer.config.in_channels, latent_t, height // 8, width // 8),
        generator=generator,
        device=device,
        dtype=torch.float32,
    ).to(dtype)

    dmd_denoise(transformer, latents.clone(), prompt_embeds, generator)  # warmup/compile
    for proc in processors.values():
        proc.attn_us.clear()
    torch.cuda.synchronize()
    start = time.perf_counter()
    latents = dmd_denoise(transformer, latents, prompt_embeds, generator)
    torch.cuda.synchronize()
    denoise_s = time.perf_counter() - start

    attn_us = [t for proc in processors.values() for t in proc.attn_us]
    print(
        f"mode={mode}: 3-step denoise {denoise_s * 1e3:.0f} ms, "
        f"self-attn total {sum(attn_us) / 1e3:.1f} ms "
        f"({sum(attn_us) / len(attn_us):.0f} us/call, {len(attn_us)} calls)"
    )

    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(device)
    torch.cuda.empty_cache()
    vae.enable_tiling()
    video = vae.decode(latents.float() / latents_std + latents_mean, return_dict=False)[0]
    frames = pipe.video_processor.postprocess_video(video, output_type="pil")[0]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(out / f"fastwan_{mode}.mp4"), fps=16)
    for idx in (0, len(frames) // 2, len(frames) - 1):
        frames[idx].save(out / f"fastwan_{mode}_frame{idx:02d}.png")
    print(f"saved video + frames to {out}/fastwan_{mode}*")


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    CLI(main)
