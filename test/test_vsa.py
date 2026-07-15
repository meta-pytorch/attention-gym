from functools import partial
import math

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from attn_gym.masks import (
    compute_vsa_coarse_attention,
    compute_vsa_tile_scores,
    compute_vsa_topk_indices,
    create_vsa_block_mask,
    create_vsa_flash_block_mask,
    create_vsa_tile_metadata,
    generate_vsa_mask_mod,
    lift_vsa_tile_output,
    pool_to_vsa_tiles,
    tile_vsa_sequence,
    untile_vsa_sequence,
    validate_vsa_block_mask_inputs,
    vsa_additive_combine,
    vsa_gated_mix,
    vsa_topk_from_sparsity,
)


def dense_vsa_token_mask(
    topk_indices: torch.Tensor,
    tile_numel: int,
    num_kv_tiles: int,
    variable_block_sizes: torch.Tensor | None = None,
):
    """Expand tile-level VSA top-k indices into a dense token-level mask."""
    block_mask = torch.zeros(
        *topk_indices.shape[:-1], num_kv_tiles, dtype=torch.bool, device=topk_indices.device
    )
    block_mask.scatter_(-1, topk_indices.long(), True)
    token_mask = block_mask.repeat_interleave(tile_numel, dim=-2).repeat_interleave(
        tile_numel, dim=-1
    )
    if variable_block_sizes is None:
        return token_mask
    offsets = torch.arange(num_kv_tiles * tile_numel, device=topk_indices.device) % tile_numel
    kv_tile = torch.arange(num_kv_tiles * tile_numel, device=topk_indices.device) // tile_numel
    valid_kv = offsets < variable_block_sizes.to(device=topk_indices.device)[kv_tile]
    return token_mask & valid_kv.view(*(1 for _ in token_mask.shape[:-1]), -1)


def masked_reference_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
):
    """Compute dense scaled dot-product attention with a pre-expanded mask."""
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v)


def test_pool_to_vsa_tiles_matches_manual_mean():
    x = torch.arange(1 * 2 * 8 * 3, dtype=torch.float32).reshape(1, 2, 8, 3)

    actual = pool_to_vsa_tiles(x, tile_numel=4)
    expected = x.reshape(1, 2, 2, 4, 3).mean(dim=-2)

    torch.testing.assert_close(actual, expected)


def test_vsa_tile_metadata_matches_fastvideo_non_divisible_layout():
    metadata = create_vsa_tile_metadata((5, 9, 10), (4, 4, 4))
    x = torch.arange(metadata.total_seq_length, dtype=torch.float32).reshape(1, 1, -1, 1)

    tiled = tile_vsa_sequence(x, metadata)
    untiled = untile_vsa_sequence(tiled, metadata)

    assert metadata.num_tiles == (2, 3, 3)
    assert metadata.tile_numel == 64
    assert metadata.padded_seq_length == 18 * 64
    assert metadata.variable_block_sizes.tolist() == [
        64,
        64,
        32,
        64,
        64,
        32,
        16,
        16,
        8,
        16,
        16,
        8,
        16,
        16,
        8,
        4,
        4,
        2,
    ]
    torch.testing.assert_close(untiled, x)
    torch.testing.assert_close(
        tiled[..., metadata.non_pad_index, :], x[..., metadata.tile_partition_indices, :]
    )
    assert torch.count_nonzero(tiled).item() == metadata.total_seq_length - 1


def test_pool_to_vsa_tiles_uses_variable_block_sizes_for_padded_tiles():
    metadata = create_vsa_tile_metadata((1, 1, 5), (1, 1, 4))
    x = tile_vsa_sequence(torch.arange(5, dtype=torch.float32).reshape(1, 1, 5, 1), metadata)

    actual = pool_to_vsa_tiles(x, tile_numel=4, variable_block_sizes=metadata.variable_block_sizes)
    expected = torch.tensor([[[[1.5], [4.0]]]])

    torch.testing.assert_close(actual, expected)


def test_vsa_topk_from_sparsity_matches_fastvideo_formula():
    metadata = create_vsa_tile_metadata((21, 30, 52), (4, 4, 4))

    assert metadata.num_tiles == (6, 8, 13)
    assert (
        vsa_topk_from_sparsity(
            metadata.total_seq_length,
            metadata.tile_numel,
            math.prod(metadata.num_tiles),
            sparsity=0.875,
        )
        == 64
    )


def test_compute_vsa_topk_indices_matches_coarse_score_reference():
    torch.manual_seed(0)
    q = torch.randn(2, 3, 12, 5)
    k = torch.randn(2, 3, 16, 5)

    scores = compute_vsa_tile_scores(q, k, tile_numel=4)
    actual = compute_vsa_topk_indices(q, k, tile_numel=4, top_k=2)
    expected = torch.topk(scores, k=2, dim=-1).indices.to(torch.int32).sort(dim=-1).values

    assert actual.dtype == torch.int32
    torch.testing.assert_close(actual, expected)


def test_compute_vsa_coarse_attention_returns_attention_output_and_lse():
    torch.manual_seed(0)
    q = torch.randn(1, 2, 12, 4)
    k = torch.randn(1, 2, 16, 4)
    v = torch.randn(1, 2, 16, 4)

    result = compute_vsa_coarse_attention(q, k, v, tile_numel=4, top_k=2)
    scores = compute_vsa_tile_scores(q, k, tile_numel=4)
    expected_output = torch.matmul(torch.softmax(scores, dim=-1), pool_to_vsa_tiles(v, 4))
    expected_lse = torch.logsumexp(scores, dim=-1)

    torch.testing.assert_close(result.output, expected_output)
    torch.testing.assert_close(result.lse, expected_lse)


def test_create_vsa_block_mask_matches_mask_mod_blocks():
    topk_indices = torch.tensor([[[[0, 2], [1, 3], [0, 3]]]], dtype=torch.int32)
    tile_numel = 2
    num_kv_tiles = 4

    direct_block_mask = create_vsa_block_mask(
        topk_indices, tile_numel=tile_numel, num_kv_tiles=num_kv_tiles
    )
    mask_mod_block_mask = create_block_mask(
        generate_vsa_mask_mod(topk_indices, tile_numel=tile_numel),
        1,
        1,
        topk_indices.shape[-2] * tile_numel,
        num_kv_tiles * tile_numel,
        device="cpu",
        BLOCK_SIZE=tile_numel,
    )

    expected = torch.zeros(1, 1, topk_indices.shape[-2], num_kv_tiles, dtype=torch.int32)
    expected.scatter_(-1, topk_indices.long(), 1)
    torch.testing.assert_close(direct_block_mask.to_dense(), expected)
    torch.testing.assert_close(mask_mod_block_mask.to_dense(), expected)


def test_validate_vsa_block_mask_inputs_rejects_invalid_ranges():
    topk_indices = torch.tensor([[[[0, 4]]]], dtype=torch.int32)

    with pytest.raises(ValueError, match="topk_indices"):
        validate_vsa_block_mask_inputs(topk_indices, tile_numel=2, num_kv_tiles=4)

    with pytest.raises(ValueError, match="variable_block_sizes"):
        validate_vsa_block_mask_inputs(
            torch.tensor([[[[0, 1]]]], dtype=torch.int32),
            tile_numel=2,
            num_kv_tiles=2,
            variable_block_sizes=torch.tensor([2, 3]),
        )


@pytest.mark.parametrize("top_k", [1, 3])
def test_vsa_flex_attention_matches_dense_masked_reference(top_k):
    torch.manual_seed(0)
    batch, heads, q_tiles, kv_tiles, tile_numel, head_dim = 1, 2, 4, 5, 4, 8
    q = torch.randn(batch, heads, q_tiles * tile_numel, head_dim)
    k = torch.randn(batch, heads, kv_tiles * tile_numel, head_dim)
    v = torch.randn(batch, heads, kv_tiles * tile_numel, head_dim)
    topk_indices = compute_vsa_topk_indices(q, k, tile_numel=tile_numel, top_k=top_k)
    block_mask = create_block_mask(
        generate_vsa_mask_mod(topk_indices, tile_numel=tile_numel),
        batch,
        heads,
        q_tiles * tile_numel,
        kv_tiles * tile_numel,
        device="cpu",
        BLOCK_SIZE=tile_numel,
    )

    actual = flex_attention(q, k, v, block_mask=block_mask)
    mask = dense_vsa_token_mask(topk_indices, tile_numel=tile_numel, num_kv_tiles=kv_tiles)
    expected = masked_reference_attention(q, k, v, mask)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_vsa_flex_attention_masks_padded_edge_tiles():
    torch.manual_seed(0)
    metadata = create_vsa_tile_metadata((2, 3, 5), (1, 2, 4))
    batch, heads, head_dim, top_k = 1, 1, 8, 2
    q = torch.randn(batch, heads, metadata.total_seq_length, head_dim)
    k = torch.randn(batch, heads, metadata.total_seq_length, head_dim)
    v = torch.randn(batch, heads, metadata.total_seq_length, head_dim)
    tiled_q = tile_vsa_sequence(q, metadata)
    tiled_k = tile_vsa_sequence(k, metadata)
    tiled_v = tile_vsa_sequence(v, metadata)
    topk_indices = compute_vsa_topk_indices(
        tiled_q,
        tiled_k,
        tile_numel=metadata.tile_numel,
        top_k=top_k,
        include_self=True,
        q_variable_block_sizes=metadata.variable_block_sizes,
        kv_variable_block_sizes=metadata.variable_block_sizes,
    )
    block_mask = create_block_mask(
        generate_vsa_mask_mod(
            topk_indices,
            tile_numel=metadata.tile_numel,
            variable_block_sizes=metadata.variable_block_sizes,
        ),
        batch,
        heads,
        metadata.padded_seq_length,
        metadata.padded_seq_length,
        device="cpu",
        BLOCK_SIZE=metadata.tile_numel,
    )

    actual = untile_vsa_sequence(
        flex_attention(tiled_q, tiled_k, tiled_v, block_mask=block_mask), metadata
    )
    mask = dense_vsa_token_mask(
        topk_indices,
        tile_numel=metadata.tile_numel,
        num_kv_tiles=math.prod(metadata.num_tiles),
        variable_block_sizes=metadata.variable_block_sizes,
    )
    expected = untile_vsa_sequence(
        masked_reference_attention(tiled_q, tiled_k, tiled_v, mask), metadata
    )

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_vsa_flex_attention_backward_cuda_matches_dense_reference():
    torch.manual_seed(0)
    batch, heads, q_tiles, kv_tiles, tile_numel, head_dim, top_k = 1, 1, 3, 4, 64, 32, 2
    q = torch.randn(
        batch, heads, q_tiles * tile_numel, head_dim, device="cuda", requires_grad=True
    )
    k = torch.randn(
        batch, heads, kv_tiles * tile_numel, head_dim, device="cuda", requires_grad=True
    )
    v = torch.randn(
        batch, heads, kv_tiles * tile_numel, head_dim, device="cuda", requires_grad=True
    )
    topk_indices = compute_vsa_topk_indices(q.detach(), k.detach(), tile_numel, top_k)
    block_mask = create_vsa_block_mask(topk_indices, tile_numel=tile_numel, num_kv_tiles=kv_tiles)
    attention = torch.compile(
        partial(flex_attention, kernel_options={"BACKEND": "TRITON"}), dynamic=False
    )

    actual = attention(q, k, v, block_mask=block_mask)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    actual_grads = (q.grad.clone(), k.grad.clone(), v.grad.clone())

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    mask = dense_vsa_token_mask(topk_indices, tile_numel=tile_numel, num_kv_tiles=kv_tiles)
    expected = masked_reference_attention(q_ref, k_ref, v_ref, mask)
    expected.backward(grad)

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    for actual_grad, expected_grad in zip(actual_grads, (q_ref.grad, k_ref.grad, v_ref.grad)):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_vsa_flex_flash_backend_sm100_matches_reference():
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("Flex FLASH block-sparse path requires SM100+")
    flex_flash = pytest.importorskip("torch._inductor.kernel.flex.flex_flash_attention")
    if not flex_flash.ensure_flash_available():
        pytest.skip("Flex FLASH backend is not available")

    torch.manual_seed(0)
    batch, heads, q_tiles, kv_tiles, tile_numel, head_dim, top_k = 1, 2, 4, 4, 256, 128, 2
    q = torch.randn(
        batch, heads, q_tiles * tile_numel, head_dim, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(
        batch, heads, kv_tiles * tile_numel, head_dim, device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn(
        batch, heads, kv_tiles * tile_numel, head_dim, device="cuda", dtype=torch.bfloat16
    )
    topk_indices = compute_vsa_topk_indices(q, k, tile_numel=tile_numel, top_k=top_k)
    block_mask = create_vsa_flash_block_mask(
        topk_indices,
        tile_numel=tile_numel,
        num_kv_tiles=kv_tiles,
    )
    attention = torch.compile(
        partial(flex_attention, kernel_options={"BACKEND": "FLASH"}), dynamic=False
    )

    actual = attention(q, k, v, block_mask=block_mask)
    mask = dense_vsa_token_mask(topk_indices, tile_numel=tile_numel, num_kv_tiles=kv_tiles)
    expected = masked_reference_attention(q, k, v, mask)

    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_vsa_flex_flash_backend_sm100_masks_partial_kv_subblocks():
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("Flex FLASH block-sparse path requires SM100+")
    flex_flash = pytest.importorskip("torch._inductor.kernel.flex.flex_flash_attention")
    if not flex_flash.ensure_flash_available():
        pytest.skip("Flex FLASH backend is not available")

    torch.manual_seed(0)
    batch, heads, q_tiles, kv_tiles, tile_numel, head_dim = 1, 2, 2, 2, 256, 128
    q = torch.randn(
        batch, heads, q_tiles * tile_numel, head_dim, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(
        batch, heads, kv_tiles * tile_numel, head_dim, device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn(
        batch, heads, kv_tiles * tile_numel, head_dim, device="cuda", dtype=torch.bfloat16
    )
    variable_block_sizes = torch.tensor([tile_numel, 64], device="cuda", dtype=torch.long)
    topk_indices = torch.tensor(
        [[[[0, 1], [0, 1]], [[0, 1], [0, 1]]]], device="cuda", dtype=torch.int32
    )
    block_mask = create_vsa_flash_block_mask(
        topk_indices,
        tile_numel=tile_numel,
        num_kv_tiles=kv_tiles,
        variable_block_sizes=variable_block_sizes,
    )
    attention = torch.compile(
        partial(flex_attention, kernel_options={"BACKEND": "FLASH"}), dynamic=False
    )

    actual = attention(q, k, v, block_mask=block_mask)
    mask = dense_vsa_token_mask(
        topk_indices,
        tile_numel=tile_numel,
        num_kv_tiles=kv_tiles,
        variable_block_sizes=variable_block_sizes,
    )
    expected = masked_reference_attention(q, k, v, mask)

    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)


def test_vsa_additive_combine_matches_manual_lift():
    torch.manual_seed(0)
    fine_output = torch.randn(1, 2, 12, 4)
    coarse_tile_output = torch.randn(1, 2, 3, 4)
    weight = torch.randn_like(fine_output)

    torch.testing.assert_close(
        vsa_additive_combine(fine_output, coarse_tile_output, weight, tile_numel=4),
        fine_output + lift_vsa_tile_output(coarse_tile_output, tile_numel=4) * weight,
    )


def test_vsa_gated_mix_extremes():
    torch.manual_seed(0)
    fine_output = torch.randn(1, 2, 12, 4)
    coarse_tile_output = torch.randn(1, 2, 3, 4)

    torch.testing.assert_close(
        vsa_gated_mix(fine_output, coarse_tile_output, gate=1.0, tile_numel=4),
        fine_output,
    )
    torch.testing.assert_close(
        vsa_gated_mix(fine_output, coarse_tile_output, gate=0.0, tile_numel=4),
        lift_vsa_tile_output(coarse_tile_output, tile_numel=4),
    )
