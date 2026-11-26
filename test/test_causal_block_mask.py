"""Tests for efficient causal block mask generation."""

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from attn_gym.masks.causal import causal_mask, create_causal_block_mask_fast


class TestCausalBlockMask:
    """Test the efficient causal block mask implementation."""

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize(
        "batch,heads,q_len,kv_len,block_size",
        [
            (1, 1, 32, 32, 64),  # smaller than one block
            (2, 1, 128, 128, 128),  # exactly one block
            (1, 4, 512, 512, 64),  # multiple full blocks
            (2, 4, 232, 232, 128),  # partial last block
            (1, 1, 45, 67, 32),  # more KV blocks than Q blocks
            (2, 4, 250, 180, 96),  # more Q blocks; non-power-of-two block size
            (1, 1, 1, 1000, 128),  # singleton query
            (1, 1, 1000, 1, 128),  # singleton key: even the diagonal block is full
        ],
    )
    def test_causal_block_mask_correctness(self, device, batch, heads, q_len, kv_len, block_size):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        actual = create_causal_block_mask_fast(batch, heads, q_len, kv_len, device, block_size)
        expected = create_block_mask(
            causal_mask, batch, heads, q_len, kv_len, device=device, BLOCK_SIZE=block_size
        )
        assert actual.shape == (batch, heads, q_len, kv_len)
        assert torch.equal(actual.to_dense(), expected.to_dense())
        assert actual.sparsity() == expected.sparsity()
        if kv_len == 1:
            assert (actual.full_kv_num_blocks == 1).all()
            assert (actual.kv_num_blocks == 0).all()

    @pytest.mark.parametrize("device", ["cuda"])
    @pytest.mark.parametrize("seq_len", [256, 232])
    def test_causal_block_mask_with_flex_attention(self, device, seq_len):
        """Test that the efficient block mask works correctly with flex_attention."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        B, H, HEAD_DIM = 2, 4, 64
        BLOCK_SIZE = 128

        query = torch.randn(
            B, H, seq_len, HEAD_DIM, device=device, dtype=torch.float32, requires_grad=True
        )
        key = torch.randn(
            B, H, seq_len, HEAD_DIM, device=device, dtype=torch.float32, requires_grad=True
        )
        value = torch.randn(
            B, H, seq_len, HEAD_DIM, device=device, dtype=torch.float32, requires_grad=True
        )

        block_mask_fast = create_causal_block_mask_fast(
            B, H, seq_len, seq_len, device, block_size=BLOCK_SIZE
        )

        block_mask_standard = create_block_mask(
            causal_mask, B, H, seq_len, seq_len, device=device, BLOCK_SIZE=BLOCK_SIZE
        )

        # Forward pass
        out_fast = flex_attention(query, key, value, block_mask=block_mask_fast)
        out_standard = flex_attention(query, key, value, block_mask=block_mask_standard)

        # Check forward pass results match
        assert torch.allclose(out_fast, out_standard, atol=1e-5, rtol=1e-4)

        # Backward pass - compute gradients using torch.autograd.grad
        grad_output = torch.randn_like(out_fast)

        grads_fast = torch.autograd.grad(
            out_fast, (query, key, value), grad_output, retain_graph=True
        )
        grads_standard = torch.autograd.grad(
            out_standard, (query, key, value), grad_output, retain_graph=True
        )

        # Check that gradients match
        assert torch.allclose(grads_fast[0], grads_standard[0], atol=1e-5, rtol=1e-4), (
            "Query gradients don't match"
        )
        assert torch.allclose(grads_fast[1], grads_standard[1], atol=1e-5, rtol=1e-4), (
            "Key gradients don't match"
        )
        assert torch.allclose(grads_fast[2], grads_standard[2], atol=1e-5, rtol=1e-4), (
            "Value gradients don't match"
        )

    @pytest.mark.parametrize("separate_full_blocks", [True, False])
    def test_separate_full_blocks_option(self, separate_full_blocks):
        """Test the separate_full_blocks option."""
        device = torch.device("cpu")

        block_mask = create_causal_block_mask_fast(
            batch_size=1,
            num_heads=1,
            q_seq_len=256,
            kv_seq_len=256,
            device=device,
            block_size=64,
            separate_full_blocks=separate_full_blocks,
        )

        if separate_full_blocks:
            assert block_mask.full_kv_num_blocks is not None
            assert block_mask.full_kv_indices is not None
        else:
            assert block_mask.full_kv_num_blocks is None
            assert block_mask.full_kv_indices is None

        # Should still produce valid causal mask
        dense = block_mask.to_dense()
        num_blocks_q = 256 // 64
        num_blocks_kv = 256 // 64

        # Check causal property at block level
        for i in range(num_blocks_q):
            for j in range(num_blocks_kv):
                if i < j:
                    assert dense[0, 0, i, j] == 0


if __name__ == "__main__":
    pytest.main([__file__])
