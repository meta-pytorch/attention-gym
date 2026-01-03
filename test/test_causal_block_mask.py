"""Tests for efficient causal block mask generation."""

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from attn_gym.masks.causal import causal_mask, create_causal_block_mask_fast


class TestCausalBlockMask:
    """Test the efficient causal block mask implementation."""

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize("seq_len", [128, 256, 512, 232])
    @pytest.mark.parametrize("block_size", [64, 128])
    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("num_heads", [1, 4])
    def test_causal_block_mask_correctness(
        self, device, seq_len, block_size, batch_size, num_heads
    ):
        """Test that the efficient implementation produces the same results as the standard one."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Create block masks using both methods
        block_mask_fast = create_causal_block_mask_fast(
            batch_size,
            num_heads,
            seq_len,
            seq_len,
            device,
            block_size=block_size,
        )

        block_mask_standard = create_block_mask(
            causal_mask,
            batch_size,
            num_heads,
            seq_len,
            seq_len,
            device=device,
            BLOCK_SIZE=block_size,
        )

        dense_fast = block_mask_fast.to_dense()
        dense_standard = block_mask_standard.to_dense()

        assert torch.equal(dense_fast, dense_standard), "Block masks should be identical"
        assert torch.equal(
            torch.tensor(block_mask_fast.sparsity()), torch.tensor(block_mask_standard.sparsity())
        )
        assert block_mask_fast.shape == block_mask_standard.shape

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
        assert torch.allclose(
            grads_fast[0], grads_standard[0], atol=1e-5, rtol=1e-4
        ), "Query gradients don't match"
        assert torch.allclose(
            grads_fast[1], grads_standard[1], atol=1e-5, rtol=1e-4
        ), "Key gradients don't match"
        assert torch.allclose(
            grads_fast[2], grads_standard[2], atol=1e-5, rtol=1e-4
        ), "Value gradients don't match"

    def test_causal_block_mask_edge_cases(self):
        """Test edge cases for the efficient causal block mask."""
        device = torch.device("cpu")

        # Test with non-square sequences
        block_mask = create_causal_block_mask_fast(
            batch_size=1,
            num_heads=1,
            q_seq_len=100,
            kv_seq_len=150,
            device=device,
            block_size=64,
        )
        assert block_mask.shape == (1, 1, 100, 150)

        # Test with sequence length smaller than block size
        block_mask = create_causal_block_mask_fast(
            batch_size=1,
            num_heads=1,
            q_seq_len=32,
            kv_seq_len=32,
            device=device,
            block_size=64,
        )
        assert block_mask.shape == (1, 1, 32, 32)

        # Verify it's still causal - when seq_len < block_size, there's only 1 block
        # So we verify the mask works correctly with flex_attention
        dense = block_mask.to_dense()
        # Since we have 32 seq len and 64 block size, we only have 1 block in each dimension
        # The block should be present since it's causal
        assert dense[0, 0, 0, 0] == 1

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

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize("q_len,kv_len", [(100, 150), (200, 100), (300, 300)])
    @pytest.mark.parametrize("block_size", [64, 128])
    def test_non_square_sequences(self, device, q_len, kv_len, block_size):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        block_mask_fast = create_causal_block_mask_fast(
            1, 1, q_len, kv_len, device, block_size=block_size
        )

        block_mask_standard = create_block_mask(
            causal_mask, 1, 1, q_len, kv_len, device=device, BLOCK_SIZE=block_size
        )

        assert torch.equal(block_mask_fast.to_dense(), block_mask_standard.to_dense())
        assert block_mask_fast.shape == (1, 1, q_len, kv_len)

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize(
        "q_len,kv_len,block_size",
        [
            (97, 97, 64),
            (130, 130, 64),
            (191, 191, 128),
            (45, 67, 32),
            (250, 180, 96),
        ],
    )
    def test_non_aligned_sequences(self, device, q_len, kv_len, block_size):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        block_mask_fast = create_causal_block_mask_fast(
            1, 1, q_len, kv_len, device, block_size=block_size
        )

        block_mask_standard = create_block_mask(
            causal_mask, 1, 1, q_len, kv_len, device=device, BLOCK_SIZE=block_size
        )

        dense_fast = block_mask_fast.to_dense()
        dense_standard = block_mask_standard.to_dense()

        assert torch.equal(dense_fast, dense_standard)
        assert block_mask_fast.shape == (1, 1, q_len, kv_len)

        num_q_blocks = (q_len + block_size - 1) // block_size
        num_kv_blocks = (kv_len + block_size - 1) // block_size

        for i in range(num_q_blocks):
            for j in range(num_kv_blocks):
                if i < j:
                    assert dense_fast[0, 0, i, j] == 0, f"Non-causal block at ({i}, {j})"

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_cross_attention_sequences(self, device):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        test_cases = [
            (50, 500),
            (500, 50),
            (1, 1000),
            (1000, 1),
        ]

        for q_len, kv_len in test_cases:
            block_mask_fast = create_causal_block_mask_fast(
                1, 1, q_len, kv_len, device, block_size=128
            )

            block_mask_standard = create_block_mask(
                causal_mask, 1, 1, q_len, kv_len, device=device, BLOCK_SIZE=128
            )

            assert torch.equal(block_mask_fast.to_dense(), block_mask_standard.to_dense())


if __name__ == "__main__":
    pytest.main([__file__])
