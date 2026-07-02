"""End-to-end FastWan2.1 (DMD, VSA-trained) inference with FlexAttention VSA.

Runs FastVideo's FastWan2.1 diffusers checkpoints (3-step DMD, sparse-distilled
with VSA at 0.8 sparsity; 14B by default, use ``--model_id`` for 1.3B) and swaps
the self-attention implementation per run:

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

MODEL_ID_1_3B = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"
MODEL_ID_14B = "FastVideo/FastWan2.1-T2V-14B-Diffusers"
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

    shards = sorted((model_dir / "transformer").glob("diffusion_pytorch_model*.safetensors"))
    gates: dict[int, tuple[Tensor, Tensor]] = {}
    pending: dict[str, Tensor] = {}
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                if "to_gate_compress" in key:
                    pending[key] = f.get_tensor(key).to(device=device, dtype=dtype)
    for key, weight in pending.items():
        if key.endswith(".weight"):
            layer = int(key.split(".")[1])
            gates[layer] = (weight, pending[key.replace(".weight", ".bias")])
    if not gates:
        raise RuntimeError("checkpoint has no to_gate_compress weights; not a VSA model")
    return gates


FASTVIDEO_KEY_RENAMES = (
    ("ffn.fc_in.", "ffn.net.0.proj."),
    ("ffn.fc_out.", "ffn.net.2."),
    ("attn2.to_out.", "attn2.to_out.0."),
    ("to_out.", "attn1.to_out.0."),
    ("to_q.", "attn1.to_q."),
    ("to_k.", "attn1.to_k."),
    ("to_v.", "attn1.to_v."),
)


def maybe_fix_14b_transformer(model_dir: Path) -> None:
    """Repair the FastWan 14B diffusers export in the local snapshot, once.

    The published 14B repo ships sharded transformer weights without the
    safetensors index and with half of the keys still in FastVideo naming
    (``blocks.N.to_q``, ``ffn.fc_in``, ...), which diffusers loads as meta
    tensors. Consolidate to one bf16 file with diffusers key names, keeping the
    VSA ``to_gate_compress`` weights for :func:`load_gate_weights`.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    tdir = model_dir / "transformer"
    shards = sorted(tdir.glob("diffusion_pytorch_model-*.safetensors"))
    if not shards or (tdir / "diffusion_pytorch_model.safetensors.index.json").exists():
        return
    state: dict[str, Tensor] = {}
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                state[key] = f.get_tensor(key).to(torch.bfloat16)

    def diffusers_key(key: str) -> str:
        if not key.startswith("blocks.") or ".attn1." in key or ".attn2.to_out.0." in key:
            return key
        prefix, _, rest = key.partition(".")
        block, _, rest = rest.partition(".")
        for src, dst in FASTVIDEO_KEY_RENAMES:
            if rest.startswith(src):
                rest = dst + rest[len(src) :]
                break
        return f"{prefix}.{block}.{rest}"

    fixed = {diffusers_key(k): v for k, v in state.items()}
    fixed.update({k: v for k, v in state.items() if ".attn1." in k or ".ffn.net." in k})
    save_file(fixed, tdir / "diffusion_pytorch_model.safetensors")
    for shard in shards:
        shard.unlink()
    print(f"repaired 14B transformer export in {tdir}")


def exact_tile_metadata(grid: tuple[int, int, int], tile: tuple[int, int, int], device):
    """Tile metadata for a grid that divides exactly, so no padding predicates are needed."""
    metadata = create_vsa_tile_metadata(grid, tile, device=device)
    if metadata.padded_seq_length != metadata.total_seq_length:
        raise ValueError(f"grid {grid} does not divide into tiles {tile}")
    return metadata


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


class SparseVSAAttention:
    """Tile-major VSA self-attention over BHSD tensors for one fixed latent grid.

    Built once per run for the requested backend only, so grids that only
    divide one backend's tiling still work for that backend. The flex path
    compiles coarse attn + topk + block-mask build + FLASH fine pass as one
    region.
    """

    def __init__(self, mode: str, grid: tuple[int, int, int], device):
        tile = UPSTREAM_TILE if mode == "upstream" else FLEX_TILE
        metadata = exact_tile_metadata(grid, tile, device)
        self.mode = mode
        self.tile_numel = metadata.tile_numel
        self.num_tiles = math.prod(metadata.num_tiles)
        self.top_k = vsa_topk_from_sparsity(
            math.prod(grid), self.tile_numel, self.num_tiles, VSA_SPARSITY
        )
        self.perm = metadata.tile_partition_indices
        self.inverse_perm = metadata.reverse_tile_partition_indices
        self.variable_block_sizes = metadata.variable_block_sizes.to(device)
        self.flex_vsa = torch.compile(self._flex_vsa, dynamic=False) if mode == "flex" else None

    def _flex_vsa(self, q: Tensor, k: Tensor, v: Tensor, gate: Tensor) -> Tensor:
        coarse = compute_vsa_coarse_attention(
            q, k, v, tile_numel=self.tile_numel, top_k=self.top_k
        )
        block_mask = create_vsa_flash_block_mask(
            coarse.topk_indices, tile_numel=self.tile_numel, num_kv_tiles=self.num_tiles
        )
        fine = flex_attention(q, k, v, block_mask=block_mask, kernel_options={"BACKEND": "FLASH"})
        return vsa_additive_combine(fine, coarse.output, gate, tile_numel=self.tile_numel)

    def __call__(self, q: Tensor, k: Tensor, v: Tensor, gate: Tensor) -> Tensor:
        q, k, v, gate = (t[:, :, self.perm] for t in (q, k, v, gate))
        match self.mode:
            case "upstream":
                out = import_upstream_vsa().video_sparse_attn(
                    q.contiguous(),
                    k.contiguous(),
                    v.contiguous(),
                    self.variable_block_sizes,
                    topk=self.top_k,
                    block_size=UPSTREAM_TILE,
                    compress_attn_weight=gate.contiguous(),
                )
            case "flex":
                out = self.flex_vsa(q, k, v, gate)
            case _:
                raise ValueError(f"unknown sparse mode {self.mode}")
        return out[:, :, self.inverse_perm]


class WanVSASelfAttnProcessor:
    """Drop-in replacement for Wan self-attention with swappable VSA backends.

    Mirrors diffusers' WanAttnProcessor projection/rope path, then runs the
    fine+coarse VSA flow with the checkpoint's learned compress gate.
    ``sparse_attn=None`` selects dense SDPA.
    """

    def __init__(self, layer: int, gates, sparse_attn: SparseVSAAttention | None):
        self.layer = layer
        self.gates = gates
        self.sparse_attn = sparse_attn
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
        if self.sparse_attn is None:
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            gate_w, gate_b = self.gates[self.layer]
            gate = F.linear(hidden_states, gate_w, gate_b).unflatten(2, (attn.heads, -1))
            out = self.sparse_attn(q, k, v, gate.transpose(1, 2))
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
    model_id: str = MODEL_ID_14B,
    num_frames: int = 61,
    height: int = 448,
    width: int = 832,
    seed: int = 42,
    output_dir: str = "outputs",
):
    """Generate a FastWan video with the selected self-attention backend.

    The latent grid (num_frames, height, width dependent) must divide exactly
    into both VSA tilings; 61x448x832 (1.3B native) and 61x768x1280 (14B
    high-res) both satisfy this.
    """
    from diffusers import AutoencoderKLWan, WanPipeline, WanTransformer3DModel
    from diffusers.utils import export_to_video
    from huggingface_hub import snapshot_download

    device, dtype = "cuda", torch.bfloat16
    model_dir = Path(snapshot_download(model_id, ignore_patterns=["assets/*", "examples/*"]))
    maybe_fix_14b_transformer(model_dir)

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
    match mode:
        case "dense":
            gates, sparse_attn = None, None
        case "upstream" | "flex":
            gates = load_gate_weights(model_dir, device, dtype)
            sparse_attn = SparseVSAAttention(mode, grid, device)
        case _:
            raise ValueError(f"mode must be dense, upstream, or flex, got {mode}")
    processors = {
        name: WanVSASelfAttnProcessor(int(name.split(".")[1]), gates, sparse_attn)
        for name in transformer.attn_processors
        if "attn1" in name
    }
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
    tag = f"{mode}_14b" if model_id == MODEL_ID_14B else mode
    export_to_video(frames, str(out / f"fastwan_{tag}.mp4"), fps=16)
    for idx in (0, len(frames) // 2, len(frames) - 1):
        frames[idx].save(out / f"fastwan_{tag}_frame{idx:02d}.png")
    print(f"saved video + frames to {out}/fastwan_{tag}*")


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    CLI(main)
