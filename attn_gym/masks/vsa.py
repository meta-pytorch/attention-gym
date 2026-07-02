"""Video Sparse Attention helpers for FlexAttention.

FlexAttention can represent VSA's fine sparse pass once each query tile's top-k KV
tiles are known. The coarse top-k selection, coarse output, and gated combination
remain ordinary PyTorch work around the FlexAttention call rather than fused into a
single kernel.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, _mask_mod_signature


@dataclass(frozen=True)
class VSACoarseResult:
    """Tile-level VSA coarse attention result.

    Args:
        topk_indices: Selected KV tile indices with shape ``(B, H, Q_TILES, TOP_K)``.
        output: Coarse attention output with shape ``(B, H, Q_TILES, D)``.
        lse: Coarse attention log-sum-exp with shape ``(B, H, Q_TILES)``.
    """

    topk_indices: Tensor
    output: Tensor
    lse: Tensor


@dataclass(frozen=True)
class VSATileMetadata:
    """FastVideo-compatible 3D tile-major layout metadata.

    Args:
        dit_seq_shape: Logical latent token grid as ``(time, height, width)``.
        tile_twh: VSA cube size as ``(time, height, width)``.
        num_tiles: Number of tiles along each 3D axis.
        tile_numel: Padded token capacity of each tile.
        total_seq_length: Number of real tokens before tiling.
        padded_seq_length: Number of tokens after tile-local padding.
        tile_partition_indices: Original row-major token ids in tile-major order.
        reverse_tile_partition_indices: Inverse permutation for real tile-major tokens.
        variable_block_sizes: Number of real tokens in each padded tile.
        non_pad_index: Padded tile-major positions that correspond to real tokens.
        untile_combined_index: Padded tile-major positions ordered back to row-major.
    """

    dit_seq_shape: tuple[int, int, int]
    tile_twh: tuple[int, int, int]
    num_tiles: tuple[int, int, int]
    tile_numel: int
    total_seq_length: int
    padded_seq_length: int
    tile_partition_indices: Tensor
    reverse_tile_partition_indices: Tensor
    variable_block_sizes: Tensor
    non_pad_index: Tensor
    untile_combined_index: Tensor


def create_vsa_tile_metadata(
    dit_seq_shape: tuple[int, int, int],
    tile_twh: tuple[int, int, int] = (4, 4, 4),
    device: torch.device | str | None = None,
) -> VSATileMetadata:
    """Create the tile permutation and padding metadata used by FastVideo VSA.

    Args:
        dit_seq_shape: Logical latent token grid as ``(time, height, width)``.
        tile_twh: VSA cube size as ``(time, height, width)``.
        device: Device for returned index tensors.

    Returns:
        Metadata for converting row-major video tokens to padded tile-major tokens
        and back.

    Raises:
        ValueError: If any shape or tile dimension is non-positive.
    """
    if any(dim <= 0 for dim in dit_seq_shape):
        raise ValueError(f"dit_seq_shape dimensions must be positive, got {dit_seq_shape}")
    if any(dim <= 0 for dim in tile_twh):
        raise ValueError(f"tile_twh dimensions must be positive, got {tile_twh}")

    time, height, width = dit_seq_shape
    tile_time, tile_height, tile_width = tile_twh
    num_tiles = (
        math.ceil(time / tile_time),
        math.ceil(height / tile_height),
        math.ceil(width / tile_width),
    )
    tile_numel = math.prod(tile_twh)
    total_seq_length = math.prod(dit_seq_shape)
    padded_seq_length = math.prod(num_tiles) * tile_numel
    arange_device = torch.device("cpu") if device is None else torch.device(device)
    padded_indices = torch.full(
        (num_tiles[0] * tile_time, num_tiles[1] * tile_height, num_tiles[2] * tile_width),
        -1,
        device=arange_device,
        dtype=torch.long,
    )
    padded_indices[:time, :height, :width] = torch.arange(
        total_seq_length, device=arange_device, dtype=torch.long
    ).reshape(time, height, width)
    padded_tiles = padded_indices.reshape(
        num_tiles[0], tile_time, num_tiles[1], tile_height, num_tiles[2], tile_width
    ).permute(0, 2, 4, 1, 3, 5)
    tile_partition_indices = padded_tiles[padded_tiles >= 0]
    reverse_tile_partition_indices = torch.argsort(tile_partition_indices)

    def sizes_for_dim(dim_len: int, tile: int, n_tiles: int) -> Tensor:
        sizes = torch.full((n_tiles,), tile, dtype=torch.long, device=arange_device)
        remainder = dim_len - (n_tiles - 1) * tile
        sizes[-1] = remainder if remainder > 0 else tile
        return sizes

    time_sizes = sizes_for_dim(time, tile_time, num_tiles[0])
    height_sizes = sizes_for_dim(height, tile_height, num_tiles[1])
    width_sizes = sizes_for_dim(width, tile_width, num_tiles[2])
    variable_block_sizes = (
        time_sizes[:, None, None] * height_sizes[None, :, None] * width_sizes[None, None, :]
    ).reshape(-1)
    tile_starts = torch.arange(variable_block_sizes.shape[0], device=arange_device) * tile_numel
    padded_tile_indices = tile_starts[:, None] + torch.arange(tile_numel, device=arange_device)
    valid_token_mask = (
        torch.arange(tile_numel, device=arange_device) < variable_block_sizes[:, None]
    )
    non_pad_index = padded_tile_indices[valid_token_mask]
    untile_combined_index = non_pad_index[reverse_tile_partition_indices]
    return VSATileMetadata(
        dit_seq_shape=dit_seq_shape,
        tile_twh=tile_twh,
        num_tiles=num_tiles,
        tile_numel=tile_numel,
        total_seq_length=total_seq_length,
        padded_seq_length=padded_seq_length,
        tile_partition_indices=tile_partition_indices,
        reverse_tile_partition_indices=reverse_tile_partition_indices,
        variable_block_sizes=variable_block_sizes,
        non_pad_index=non_pad_index,
        untile_combined_index=untile_combined_index,
    )


def tile_vsa_sequence(x: Tensor, metadata: VSATileMetadata) -> Tensor:
    """Convert a row-major video sequence into padded tile-major order.

    Args:
        x: Tensor with shape ``(..., total_seq_length, head_dim)``.
        metadata: VSA tile layout metadata.

    Returns:
        Tensor with shape ``(..., padded_seq_length, head_dim)``.

    Raises:
        ValueError: If the input sequence length does not match the metadata.
    """
    if x.shape[-2] != metadata.total_seq_length:
        raise ValueError(
            f"input sequence length {x.shape[-2]} must match metadata total_seq_length "
            f"{metadata.total_seq_length}"
        )
    output = x.new_zeros(*x.shape[:-2], metadata.padded_seq_length, x.shape[-1])
    output[..., metadata.non_pad_index, :] = x[..., metadata.tile_partition_indices, :]
    return output


def untile_vsa_sequence(x: Tensor, metadata: VSATileMetadata) -> Tensor:
    """Convert a padded tile-major video sequence back to row-major order.

    Args:
        x: Tensor with shape ``(..., padded_seq_length, head_dim)``.
        metadata: VSA tile layout metadata.

    Returns:
        Tensor with shape ``(..., total_seq_length, head_dim)``.

    Raises:
        ValueError: If the input sequence length does not match the metadata.
    """
    if x.shape[-2] != metadata.padded_seq_length:
        raise ValueError(
            f"input sequence length {x.shape[-2]} must match metadata padded_seq_length "
            f"{metadata.padded_seq_length}"
        )
    return x[..., metadata.untile_combined_index, :]


def vsa_topk_from_sparsity(
    sequence_length: int,
    tile_numel: int,
    num_kv_tiles: int,
    sparsity: float,
) -> int:
    """Compute a VSA top-k tile count from a sparsity fraction.

    Args:
        sequence_length: Token count used for the sparsity convention. Use padded
            length to target a fraction of physical tile slots, or real length to
            match FastVideo's current Python wrapper.
        tile_numel: Number of tokens per VSA tile.
        num_kv_tiles: Total number of KV tiles.
        sparsity: Fraction of KV tile work to skip.

    Returns:
        Clamped top-k tile count.

    Raises:
        ValueError: If lengths are non-positive or sparsity lies outside ``[0, 1]``.
    """
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")
    if tile_numel <= 0:
        raise ValueError(f"tile_numel must be positive, got {tile_numel}")
    if num_kv_tiles <= 0:
        raise ValueError(f"num_kv_tiles must be positive, got {num_kv_tiles}")
    if not 0 <= sparsity <= 1:
        raise ValueError(f"sparsity must be in [0, 1], got {sparsity}")
    return min(num_kv_tiles, max(1, math.ceil((1 - sparsity) * sequence_length / tile_numel)))


def pool_to_vsa_tiles(
    x: Tensor,
    tile_numel: int,
    variable_block_sizes: Tensor | None = None,
) -> Tensor:
    """Mean-pool a tile-major sequence into VSA coarse tokens.

    Args:
        x: Tensor with shape ``(..., seq_len, head_dim)``. The sequence is assumed
            to be arranged tile-major, with every contiguous ``tile_numel`` tokens
            belonging to the same video cube.
        tile_numel: Number of fine tokens in one VSA tile.
        variable_block_sizes: Optional real-token counts for each padded tile.

    Returns:
        Tensor with shape ``(..., num_tiles, head_dim)``.

    Raises:
        ValueError: If ``tile_numel`` is non-positive, if it does not divide
            ``seq_len``, or if ``variable_block_sizes`` is incompatible.
    """
    if tile_numel <= 0:
        raise ValueError(f"tile_numel must be positive, got {tile_numel}")
    if x.shape[-2] % tile_numel != 0:
        raise ValueError(
            f"sequence length {x.shape[-2]} must be divisible by tile_numel {tile_numel}"
        )
    tiled = x.unflatten(-2, (-1, tile_numel))
    if variable_block_sizes is None:
        return tiled.mean(dim=-2)
    if variable_block_sizes.shape != (tiled.shape[-3],):
        raise ValueError(
            f"variable_block_sizes must have shape ({tiled.shape[-3]},), got "
            f"{tuple(variable_block_sizes.shape)}"
        )
    counts = variable_block_sizes.to(device=x.device, dtype=torch.float32).clamp_min(1)
    return (tiled.float().sum(dim=-2) / counts.view(*(1 for _ in x.shape[:-2]), -1, 1)).to(x.dtype)


def compute_vsa_tile_scores(
    query: Tensor,
    key: Tensor,
    tile_numel: int,
    scale: float | None = None,
    q_variable_block_sizes: Tensor | None = None,
    kv_variable_block_sizes: Tensor | None = None,
) -> Tensor:
    """Compute coarse query-tile to KV-tile scores for VSA selection.

    Args:
        query: Query tensor with shape ``(B, H, S_Q, D)`` in tile-major order.
        key: Key tensor with shape ``(B, H, S_KV, D)`` in tile-major order.
        tile_numel: Number of fine tokens in one VSA tile.
        scale: Optional scale applied to the coarse dot products. Defaults to
            ``1 / sqrt(D)``.
        q_variable_block_sizes: Optional real-token counts for padded query tiles.
        kv_variable_block_sizes: Optional real-token counts for padded KV tiles.

    Returns:
        Coarse score tensor with shape ``(B, H, Q_TILES, KV_TILES)``.

    Raises:
        ValueError: If query and key leading dimensions or head dimensions differ.
    """
    if query.shape[:-2] != key.shape[:-2] or query.shape[-1] != key.shape[-1]:
        raise ValueError(
            "query and key must have matching batch/head dimensions and head_dim, "
            f"got {tuple(query.shape)} and {tuple(key.shape)}"
        )
    scale_factor = 1 / math.sqrt(query.shape[-1]) if scale is None else scale
    pooled_query = pool_to_vsa_tiles(query, tile_numel, q_variable_block_sizes)
    pooled_key = pool_to_vsa_tiles(key, tile_numel, kv_variable_block_sizes)
    return torch.matmul(pooled_query, pooled_key.transpose(-2, -1)) * scale_factor


def force_self_into_topk(topk_indices: Tensor, num_kv_tiles: int) -> Tensor:
    """Ensure each query tile includes the same-id KV tile when self-attending."""
    if topk_indices.shape[-2] > num_kv_tiles:
        raise ValueError("include_self=True requires Q_TILES <= KV_TILES")
    self_indices = torch.arange(
        topk_indices.shape[-2], device=topk_indices.device, dtype=torch.int32
    )
    self_indices = self_indices.view(
        *(1 for _ in topk_indices.shape[:-2]), topk_indices.shape[-2], 1
    )
    has_self = (topk_indices == self_indices).any(dim=-1, keepdim=True)
    replacement = topk_indices.clone()
    replacement[..., -1:] = self_indices
    return torch.where(has_self, topk_indices, replacement)


def compute_vsa_topk_indices(
    query: Tensor,
    key: Tensor,
    tile_numel: int,
    top_k: int,
    scale: float | None = None,
    sort_indices: bool = True,
    include_self: bool = False,
    q_variable_block_sizes: Tensor | None = None,
    kv_variable_block_sizes: Tensor | None = None,
) -> Tensor:
    """Select each query tile's top-k KV tiles from VSA coarse scores.

    Args:
        query: Query tensor with shape ``(B, H, S_Q, D)`` in tile-major order.
        key: Key tensor with shape ``(B, H, S_KV, D)`` in tile-major order.
        tile_numel: Number of fine tokens in one VSA tile.
        top_k: Number of KV tiles selected for each query tile.
        scale: Optional scale applied to the coarse dot products. Defaults to
            ``1 / sqrt(D)``.
        sort_indices: Whether to sort selected KV tile ids ascending before returning.
        include_self: Whether to force each query tile to include the same-id KV tile.
        q_variable_block_sizes: Optional real-token counts for padded query tiles.
        kv_variable_block_sizes: Optional real-token counts for padded KV tiles.

    Returns:
        Tensor of selected KV tile ids with shape ``(B, H, Q_TILES, TOP_K)`` and
        dtype ``torch.int32``.

    Raises:
        ValueError: If ``top_k`` is outside ``[1, KV_TILES]``.
    """
    scores = compute_vsa_tile_scores(
        query, key, tile_numel, scale, q_variable_block_sizes, kv_variable_block_sizes
    )
    if top_k <= 0 or top_k > scores.shape[-1]:
        raise ValueError(f"top_k must be in [1, {scores.shape[-1]}], got {top_k}")
    topk_indices = torch.topk(scores, k=top_k, dim=-1).indices.to(torch.int32)
    if include_self:
        topk_indices = force_self_into_topk(topk_indices, scores.shape[-1])
    if sort_indices:
        topk_indices = topk_indices.sort(dim=-1).values
    return topk_indices


def compute_vsa_coarse_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    tile_numel: int,
    top_k: int,
    scale: float | None = None,
    sort_indices: bool = True,
    include_self: bool = False,
    q_variable_block_sizes: Tensor | None = None,
    kv_variable_block_sizes: Tensor | None = None,
) -> VSACoarseResult:
    """Run VSA's coarse attention pass and extract fine-pass top-k tiles.

    Args:
        query: Query tensor with shape ``(B, H, S_Q, D)`` in tile-major order.
        key: Key tensor with shape ``(B, H, S_KV, D)`` in tile-major order.
        value: Value tensor with shape ``(B, H, S_KV, D)`` in tile-major order.
        tile_numel: Number of fine tokens in one VSA tile.
        top_k: Number of KV tiles selected for each query tile.
        scale: Optional scale applied to the coarse dot products. Defaults to
            ``1 / sqrt(D)``.
        sort_indices: Whether to sort selected KV tile ids ascending before returning.
        include_self: Whether to force each query tile to include the same-id KV tile.
        q_variable_block_sizes: Optional real-token counts for padded query tiles.
        kv_variable_block_sizes: Optional real-token counts for padded KV tiles.

    Returns:
        ``VSACoarseResult`` containing top-k indices, tile-level coarse output, and
        coarse log-sum-exp.

    Raises:
        ValueError: If value shape is incompatible or ``top_k`` is invalid.
    """
    if key.shape != value.shape:
        raise ValueError(
            f"key and value must have the same shape, got {key.shape} and {value.shape}"
        )
    scores = compute_vsa_tile_scores(
        query, key, tile_numel, scale, q_variable_block_sizes, kv_variable_block_sizes
    )
    if top_k <= 0 or top_k > scores.shape[-1]:
        raise ValueError(f"top_k must be in [1, {scores.shape[-1]}], got {top_k}")
    topk_indices = torch.topk(scores, k=top_k, dim=-1).indices.to(torch.int32)
    if include_self:
        topk_indices = force_self_into_topk(topk_indices, scores.shape[-1])
    if sort_indices:
        topk_indices = topk_indices.sort(dim=-1).values
    pooled_value = pool_to_vsa_tiles(value, tile_numel, kv_variable_block_sizes)
    probabilities = torch.softmax(scores, dim=-1)
    return VSACoarseResult(
        topk_indices=topk_indices,
        output=torch.matmul(probabilities, pooled_value),
        lse=torch.logsumexp(scores, dim=-1),
    )


def validate_vsa_block_mask_inputs(
    topk_indices: Tensor,
    tile_numel: int,
    num_kv_tiles: int,
    variable_block_sizes: Tensor | None = None,
) -> None:
    """Validate VSA block-mask inputs for tests and debugging.

    This helper may synchronize CUDA tensors. Keep it out of hot inference paths.

    Args:
        topk_indices: Selected KV tile ids with shape ``(B, H, Q_TILES, TOP_K)``.
        tile_numel: Number of fine tokens in one VSA tile.
        num_kv_tiles: Total number of KV tiles before top-k pruning.
        variable_block_sizes: Optional real-token counts for each padded KV tile.

    Raises:
        ValueError: If shapes, ranges, or block sizes are invalid.
    """
    if topk_indices.dim() != 4:
        raise ValueError(
            f"topk_indices must have shape (B, H, Q_TILES, TOP_K), got {topk_indices.shape}"
        )
    if tile_numel <= 0:
        raise ValueError(f"tile_numel must be positive, got {tile_numel}")
    if num_kv_tiles <= 0:
        raise ValueError(f"num_kv_tiles must be positive, got {num_kv_tiles}")
    if topk_indices.shape[-1] > num_kv_tiles:
        raise ValueError(
            f"TOP_K must be <= num_kv_tiles, got {topk_indices.shape[-1]} and {num_kv_tiles}"
        )
    if topk_indices.numel() > 0:
        min_index = topk_indices.min().item()
        max_index = topk_indices.max().item()
        if min_index < 0 or max_index >= num_kv_tiles:
            raise ValueError(
                f"topk_indices must be in [0, {num_kv_tiles}), got min={min_index}, "
                f"max={max_index}"
            )
    if variable_block_sizes is None:
        return
    if variable_block_sizes.shape != (num_kv_tiles,):
        raise ValueError(
            f"variable_block_sizes must have shape ({num_kv_tiles},), got "
            f"{tuple(variable_block_sizes.shape)}"
        )
    min_block_size = variable_block_sizes.min().item()
    max_block_size = variable_block_sizes.max().item()
    if min_block_size <= 0 or max_block_size > tile_numel:
        raise ValueError("variable_block_sizes entries must be in [1, tile_numel]")


def create_vsa_block_mask(
    topk_indices: Tensor,
    tile_numel: int,
    num_kv_tiles: int,
    variable_block_sizes: Tensor | None = None,
) -> BlockMask:
    """Build the runtime FlexAttention ``BlockMask`` for VSA's fine sparse pass.

    Args:
        topk_indices: Selected KV tile ids with shape ``(B, H, Q_TILES, TOP_K)``.
        tile_numel: Number of fine tokens in one VSA tile.
        num_kv_tiles: Total number of KV tiles before top-k pruning.
        variable_block_sizes: Optional real-token counts for each padded KV tile.

    Returns:
        A directly constructed block-sparse mask. Full selected tiles use
        ``full_kv_*`` metadata; padded edge tiles use partial metadata plus a padding
        predicate. The top-k membership itself is encoded by block metadata.

    Raises:
        ValueError: If rank or static shape arguments are invalid.
    """
    if topk_indices.dim() != 4:
        raise ValueError(
            f"topk_indices must have shape (B, H, Q_TILES, TOP_K), got {topk_indices.shape}"
        )
    if tile_numel <= 0:
        raise ValueError(f"tile_numel must be positive, got {tile_numel}")
    if num_kv_tiles <= 0:
        raise ValueError(f"num_kv_tiles must be positive, got {num_kv_tiles}")

    selected_indices = topk_indices.to(dtype=torch.int32).sort(dim=-1).values.contiguous()
    top_k = selected_indices.shape[-1]
    if top_k > num_kv_tiles:
        raise ValueError(f"TOP_K must be <= num_kv_tiles, got {top_k} and {num_kv_tiles}")

    kv_indices = torch.zeros(
        *selected_indices.shape[:-1],
        num_kv_tiles,
        dtype=torch.int32,
        device=selected_indices.device,
    )
    full_kv_indices = torch.zeros_like(kv_indices)
    kv_num_blocks = torch.full(
        selected_indices.shape[:-1], top_k, dtype=torch.int32, device=selected_indices.device
    )
    full_kv_num_blocks = torch.zeros_like(kv_num_blocks)
    mask_mod = None

    if variable_block_sizes is None:
        full_kv_num_blocks = kv_num_blocks
        full_kv_indices[..., :top_k] = selected_indices
        kv_num_blocks = torch.zeros_like(full_kv_num_blocks)
    else:
        variable_block_sizes = variable_block_sizes.to(device=selected_indices.device)
        if variable_block_sizes.shape != (num_kv_tiles,):
            raise ValueError(
                f"variable_block_sizes must have shape ({num_kv_tiles},), got "
                f"{tuple(variable_block_sizes.shape)}"
            )
        selected_is_full = variable_block_sizes[selected_indices.long()] == tile_numel
        full_order = torch.argsort((~selected_is_full).to(torch.int32), dim=-1, stable=True)
        partial_order = torch.argsort(selected_is_full.to(torch.int32), dim=-1, stable=True)
        full_kv_indices[..., :top_k] = torch.gather(selected_indices, -1, full_order)
        kv_indices[..., :top_k] = torch.gather(selected_indices, -1, partial_order)
        full_kv_num_blocks = selected_is_full.sum(dim=-1).to(torch.int32)
        kv_num_blocks = (~selected_is_full).sum(dim=-1).to(torch.int32)
        mask_mod = generate_vsa_padding_mask_mod(variable_block_sizes, tile_numel)

    return BlockMask.from_kv_blocks(
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        full_kv_num_blocks=full_kv_num_blocks,
        full_kv_indices=full_kv_indices,
        BLOCK_SIZE=tile_numel,
        mask_mod=mask_mod,
        seq_lengths=(selected_indices.shape[-2] * tile_numel, num_kv_tiles * tile_numel),
    )


def create_vsa_flash_block_mask(
    topk_indices: Tensor,
    tile_numel: int,
    num_kv_tiles: int,
    kv_block_size: int = 128,
) -> BlockMask:
    """Build the all-full-block VSA mask representation required by Flex FLASH.

    Args:
        topk_indices: Selected KV tile ids with shape ``(B, H, Q_TILES, TOP_K)``.
        tile_numel: Number of fine tokens in one VSA tile.
        num_kv_tiles: Total number of KV tiles before top-k pruning.
        kv_block_size: KV block size of the returned mask. FA4 requires this to
            equal its KV tile size (``tile_n``, 128 on SM100 for head_dim 128), so
            each selected VSA tile is expanded into ``tile_numel // kv_block_size``
            consecutive KV blocks. The Q block size stays ``tile_numel`` because FA4
            only requires it to be a multiple of its effective Q tile.

    Returns:
        A full-block ``BlockMask`` plus a tile-level mask_mod. Current Flex FLASH
        needs both to represent sparse full blocks correctly.

    Raises:
        ValueError: If rank or static shape arguments are invalid.
    """
    if topk_indices.dim() != 4:
        raise ValueError(
            f"topk_indices must have shape (B, H, Q_TILES, TOP_K), got {topk_indices.shape}"
        )
    if tile_numel <= 0:
        raise ValueError(f"tile_numel must be positive, got {tile_numel}")
    if num_kv_tiles <= 0:
        raise ValueError(f"num_kv_tiles must be positive, got {num_kv_tiles}")
    if kv_block_size <= 0 or tile_numel % kv_block_size != 0:
        raise ValueError(
            f"kv_block_size must be positive and divide tile_numel {tile_numel}, "
            f"got {kv_block_size}"
        )

    selected_indices = topk_indices.to(dtype=torch.int32).sort(dim=-1).values.contiguous()
    top_k = selected_indices.shape[-1]
    if top_k > num_kv_tiles:
        raise ValueError(f"TOP_K must be <= num_kv_tiles, got {top_k} and {num_kv_tiles}")

    kv_factor = tile_numel // kv_block_size
    expanded_indices = (
        selected_indices[..., None] * kv_factor
        + torch.arange(kv_factor, device=selected_indices.device, dtype=torch.int32)
    ).flatten(-2)
    kv_indices = torch.zeros(
        *selected_indices.shape[:-1],
        num_kv_tiles * kv_factor,
        dtype=torch.int32,
        device=selected_indices.device,
    )
    full_kv_indices = torch.zeros_like(kv_indices)
    full_kv_indices[..., : top_k * kv_factor] = expanded_indices
    full_kv_num_blocks = torch.full(
        selected_indices.shape[:-1],
        top_k * kv_factor,
        dtype=torch.int32,
        device=selected_indices.device,
    )
    block_map = torch.zeros(
        *selected_indices.shape[:-1],
        num_kv_tiles,
        dtype=torch.bool,
        device=selected_indices.device,
    )
    block_map.scatter_(-1, selected_indices.long(), True)
    return BlockMask.from_kv_blocks(
        kv_num_blocks=torch.zeros_like(full_kv_num_blocks),
        kv_indices=kv_indices,
        full_kv_num_blocks=full_kv_num_blocks,
        full_kv_indices=full_kv_indices,
        BLOCK_SIZE=(tile_numel, kv_block_size),
        mask_mod=generate_vsa_block_map_mask_mod(block_map, tile_numel),
        seq_lengths=(selected_indices.shape[-2] * tile_numel, num_kv_tiles * tile_numel),
    )


def generate_vsa_mask_mod(
    topk_indices: Tensor,
    tile_numel: int,
    variable_block_sizes: Tensor | None = None,
) -> _mask_mod_signature:
    """Create a VSA mask_mod from precomputed top-k tile indices.

    Args:
        topk_indices: Selected KV tile ids with shape ``(B, H, Q_TILES, TOP_K)``.
        tile_numel: Number of fine tokens in one VSA tile.
        variable_block_sizes: Optional real-token counts for each padded KV tile.

    Returns:
        A ``mask_mod`` that allows a query token to attend to KV tokens whose tile
        appears in that query tile's top-k list and, if provided, excludes padded KV
        positions in edge tiles.

    Raises:
        ValueError: If ``topk_indices`` has the wrong rank or ``tile_numel`` is invalid.
    """
    if topk_indices.dim() != 4:
        raise ValueError(
            f"topk_indices must have shape (B, H, Q_TILES, TOP_K), got {topk_indices.shape}"
        )
    if tile_numel <= 0:
        raise ValueError(f"tile_numel must be positive, got {tile_numel}")
    if variable_block_sizes is not None:
        variable_block_sizes = variable_block_sizes.to(device=topk_indices.device)

    def vsa_mask_mod(b, h, q_idx, kv_idx):
        q_tile = q_idx // tile_numel
        kv_tile = kv_idx // tile_numel
        selected_tiles = topk_indices[b, h, q_tile]
        selected = (selected_tiles == kv_tile.unsqueeze(-1)).any(dim=-1)
        if variable_block_sizes is None:
            return selected
        return selected & (kv_idx % tile_numel < variable_block_sizes[kv_tile])

    vsa_mask_mod.__name__ = f"vsa_topk_t{tile_numel}_k{topk_indices.shape[-1]}"
    return vsa_mask_mod


def generate_vsa_block_map_mask_mod(
    block_map: Tensor,
    tile_numel: int,
    variable_block_sizes: Tensor | None = None,
) -> _mask_mod_signature:
    """Create a pointwise mask_mod from a dense tile-level block map.

    Args:
        block_map: Boolean tensor with shape ``(B, H, Q_TILES, KV_TILES)``.
        tile_numel: Number of fine tokens in one VSA tile.
        variable_block_sizes: Optional real-token counts for each padded KV tile.

    Returns:
        A ``mask_mod`` with O(1) tile membership lookup.

    Raises:
        ValueError: If ``block_map`` has the wrong rank or ``tile_numel`` is invalid.
    """
    if block_map.dim() != 4:
        raise ValueError(
            f"block_map must have shape (B, H, Q_TILES, KV_TILES), got {block_map.shape}"
        )
    if tile_numel <= 0:
        raise ValueError(f"tile_numel must be positive, got {tile_numel}")
    block_map = block_map.to(dtype=torch.bool)
    if variable_block_sizes is not None:
        variable_block_sizes = variable_block_sizes.to(device=block_map.device)

    def vsa_block_map_mask_mod(b, h, q_idx, kv_idx):
        q_tile = q_idx // tile_numel
        kv_tile = kv_idx // tile_numel
        selected = block_map[b, h, q_tile, kv_tile]
        if variable_block_sizes is None:
            return selected
        return selected & (kv_idx % tile_numel < variable_block_sizes[kv_tile])

    vsa_block_map_mask_mod.__name__ = f"vsa_block_map_t{tile_numel}"
    return vsa_block_map_mask_mod


def generate_vsa_padding_mask_mod(
    variable_block_sizes: Tensor, tile_numel: int
) -> _mask_mod_signature:
    """Create the lightweight partial-tile padding predicate used by direct BlockMasks.

    Args:
        variable_block_sizes: Real-token counts for each padded KV tile.
        tile_numel: Number of fine tokens in one VSA tile.

    Returns:
        A ``mask_mod`` that only filters padded KV positions inside selected partial
        blocks. Top-k membership is represented by the surrounding ``BlockMask``.

    Raises:
        ValueError: If ``tile_numel`` is invalid.
    """
    if tile_numel <= 0:
        raise ValueError(f"tile_numel must be positive, got {tile_numel}")

    def vsa_padding_mask_mod(b, h, q_idx, kv_idx):
        kv_tile = kv_idx // tile_numel
        return kv_idx % tile_numel < variable_block_sizes[kv_tile]

    vsa_padding_mask_mod.__name__ = f"vsa_padding_t{tile_numel}"
    return vsa_padding_mask_mod


def lift_vsa_tile_output(
    tile_output: Tensor, tile_numel: int, seq_len: int | None = None
) -> Tensor:
    """Expand tile-level coarse output back to token-level output.

    Args:
        tile_output: Tensor with shape ``(..., num_tiles, head_dim)``.
        tile_numel: Number of fine tokens represented by one tile output.
        seq_len: Optional output sequence length. Use this to trim padded tile-major
            sequences back to a logical length.

    Returns:
        Tensor with shape ``(..., num_tiles * tile_numel, head_dim)`` or the same
        tensor truncated to ``seq_len``.

    Raises:
        ValueError: If ``tile_numel`` is invalid or ``seq_len`` exceeds the expanded
            sequence length.
    """
    if tile_numel <= 0:
        raise ValueError(f"tile_numel must be positive, got {tile_numel}")
    lifted = tile_output.repeat_interleave(tile_numel, dim=-2)
    if seq_len is None:
        return lifted
    if seq_len < 0 or seq_len > lifted.shape[-2]:
        raise ValueError(f"seq_len must be in [0, {lifted.shape[-2]}], got {seq_len}")
    return lifted[..., :seq_len, :]


def vsa_additive_combine(
    fine_output: Tensor,
    coarse_tile_output: Tensor,
    compress_attn_weight: float | Tensor,
    tile_numel: int,
) -> Tensor:
    """Match FastVideo's public coarse-plus-sparse VSA wrapper combine.

    Args:
        fine_output: Fine sparse attention output with shape ``(B, H, S_Q, D)``.
        coarse_tile_output: Coarse attention output with shape ``(B, H, Q_TILES, D)``.
        compress_attn_weight: Scalar or tensor broadcastable to ``fine_output``.
        tile_numel: Number of fine tokens represented by one tile output.

    Returns:
        The additive combination ``fine_output + compress_attn_weight * coarse_output``.

    Raises:
        ValueError: If the lifted coarse output shape does not match ``fine_output``.
    """
    coarse_output = lift_vsa_tile_output(coarse_tile_output, tile_numel, fine_output.shape[-2])
    if coarse_output.shape != fine_output.shape:
        raise ValueError(
            f"lifted coarse output shape {coarse_output.shape} must match fine output "
            f"shape {fine_output.shape}"
        )
    return fine_output + coarse_output * compress_attn_weight


def vsa_gated_mix(
    fine_output: Tensor,
    coarse_tile_output: Tensor,
    gate: float | Tensor,
    tile_numel: int,
) -> Tensor:
    """Blend sparse fine output with lifted coarse output outside the attention kernel.

    This helper is a simple post-attention mixture, not FastVideo's fused
    LSE-correct coarse/fine merge. It is useful for experiments with learned gates
    but should not be treated as a numerically faithful VSA kernel replacement.

    Args:
        fine_output: Fine sparse attention output with shape ``(B, H, S_Q, D)``.
        coarse_tile_output: Coarse attention output with shape ``(B, H, Q_TILES, D)``.
        gate: Scalar or tensor broadcastable to ``fine_output``. A value of one keeps
            only fine attention; zero keeps only lifted coarse attention.
        tile_numel: Number of fine tokens represented by one tile output.

    Returns:
        The gated mixture ``gate * fine_output + (1 - gate) * coarse_output``.

    Raises:
        ValueError: If the lifted coarse output shape does not match ``fine_output``.
    """
    coarse_output = lift_vsa_tile_output(coarse_tile_output, tile_numel, fine_output.shape[-2])
    if coarse_output.shape != fine_output.shape:
        raise ValueError(
            f"lifted coarse output shape {coarse_output.shape} must match fine output "
            f"shape {fine_output.shape}"
        )
    return fine_output * gate + coarse_output * (1 - gate)


def main(device: str = "cpu"):
    """Visualize a tiny data-dependent VSA top-k mask."""
    from attn_gym import visualize_attention_scores

    torch.manual_seed(0)
    batch, heads, q_tiles, kv_tiles, tile_numel, head_dim, top_k = 1, 1, 6, 6, 4, 8, 2
    query = torch.randn(batch, heads, q_tiles * tile_numel, head_dim, device=device)
    key = torch.randn(batch, heads, kv_tiles * tile_numel, head_dim, device=device)
    topk_indices = compute_vsa_topk_indices(query, key, tile_numel=tile_numel, top_k=top_k)
    mask_mod = generate_vsa_mask_mod(topk_indices, tile_numel=tile_numel)
    visualize_attention_scores(
        query, key, mask_mod=mask_mod, device=device, name=mask_mod.__name__
    )


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    CLI(main)
