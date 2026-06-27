"""Demonstrates Video Sparse Attention top-k tiles with FlexAttention."""

from functools import partial
import math
from typing import Literal

import torch
from torch.nn.attention.flex_attention import flex_attention

from attn_gym.masks import (
    compute_vsa_coarse_attention,
    create_vsa_block_mask,
    create_vsa_flash_block_mask,
    create_vsa_tile_metadata,
    tile_vsa_sequence,
    untile_vsa_sequence,
    vsa_additive_combine,
    vsa_topk_from_sparsity,
)


def masked_reference_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
):
    """Compute dense attention with a boolean mask for small correctness checks."""
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v)


def dense_vsa_token_mask(
    topk_indices: torch.Tensor,
    tile_numel: int,
    num_kv_tiles: int,
    variable_block_sizes: torch.Tensor,
):
    """Expand tile-level VSA top-k indices into a dense token-level mask."""
    block_mask = torch.zeros(
        *topk_indices.shape[:-1], num_kv_tiles, dtype=torch.bool, device=topk_indices.device
    )
    block_mask.scatter_(-1, topk_indices.long(), True)
    token_mask = block_mask.repeat_interleave(tile_numel, dim=-2).repeat_interleave(
        tile_numel, dim=-1
    )
    offsets = torch.arange(num_kv_tiles * tile_numel, device=topk_indices.device) % tile_numel
    kv_tile = torch.arange(num_kv_tiles * tile_numel, device=topk_indices.device) // tile_numel
    valid_kv = offsets < variable_block_sizes.to(device=topk_indices.device)[kv_tile]
    return token_mask & valid_kv.view(*(1 for _ in token_mask.shape[:-1]), -1)


def make_attention(
    backend: Literal["FLASH", "TRITON"] | None,
    use_compile: bool,
):
    """Build a FlexAttention callable, compiling when a backend is forced."""
    if backend is not None:
        return torch.compile(
            partial(flex_attention, kernel_options={"BACKEND": backend}), dynamic=False
        )
    if use_compile:
        return torch.compile(flex_attention, dynamic=False)
    return flex_attention


def main(
    device: str = "cpu",
    backend: Literal["FLASH", "TRITON"] | None = None,
    use_compile: bool = False,
    sparsity: float = 0.5,
):
    """Run a VSA fine-pass example and optionally force the FA4/FlexFlash backend."""
    if backend == "FLASH" and not device.startswith("cuda"):
        raise ValueError("FLASH backend requires a CUDA device")
    torch.manual_seed(0)
    dtype = torch.bfloat16 if backend == "FLASH" else torch.float32
    metadata = create_vsa_tile_metadata((4, 16, 16), (4, 8, 8), device=device)
    batch, heads, head_dim = 1, 2, 128
    num_kv_tiles = math.prod(metadata.num_tiles)
    top_k = vsa_topk_from_sparsity(
        metadata.padded_seq_length, metadata.tile_numel, num_kv_tiles, sparsity=sparsity
    )
    q = tile_vsa_sequence(
        torch.randn(batch, heads, metadata.total_seq_length, head_dim, device=device, dtype=dtype),
        metadata,
    )
    k = tile_vsa_sequence(
        torch.randn(batch, heads, metadata.total_seq_length, head_dim, device=device, dtype=dtype),
        metadata,
    )
    v = tile_vsa_sequence(
        torch.randn(batch, heads, metadata.total_seq_length, head_dim, device=device, dtype=dtype),
        metadata,
    )

    coarse = compute_vsa_coarse_attention(
        q,
        k,
        v,
        tile_numel=metadata.tile_numel,
        top_k=top_k,
        include_self=True,
        q_variable_block_sizes=metadata.variable_block_sizes,
        kv_variable_block_sizes=metadata.variable_block_sizes,
    )
    if backend == "FLASH" or (backend is None and not use_compile):
        block_mask = create_vsa_flash_block_mask(
            coarse.topk_indices,
            tile_numel=metadata.tile_numel,
            num_kv_tiles=num_kv_tiles,
        )
    else:
        block_mask = create_vsa_block_mask(
            coarse.topk_indices,
            tile_numel=metadata.tile_numel,
            num_kv_tiles=num_kv_tiles,
            variable_block_sizes=metadata.variable_block_sizes,
        )
    fine_out = make_attention(backend, use_compile)(q, k, v, block_mask=block_mask)
    out = vsa_additive_combine(
        fine_out,
        coarse.output,
        compress_attn_weight=0.75,
        tile_numel=metadata.tile_numel,
    )

    mask = dense_vsa_token_mask(
        coarse.topk_indices,
        tile_numel=metadata.tile_numel,
        num_kv_tiles=num_kv_tiles,
        variable_block_sizes=metadata.variable_block_sizes,
    )
    reference = masked_reference_attention(q, k, v, mask)
    atol = rtol = 5e-2 if backend == "FLASH" else 1e-5
    torch.testing.assert_close(fine_out, reference, atol=atol, rtol=rtol)

    print(f"backend: {backend or 'default'}")
    print(f"tiled fine sparse VSA output shape: {tuple(fine_out.shape)}")
    print(f"untiled output shape: {tuple(untile_vsa_sequence(out, metadata).shape)}")
    print(f"selected {top_k} of {num_kv_tiles} KV tiles per query tile")


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    CLI(main)
