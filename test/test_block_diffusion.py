import pytest
import torch
import torch.nn.functional as F

from attn_gym.masks import generate_block_diffusion_mask


def reference_mask(seq_len: int, block_size: int, device: str = "cpu") -> torch.Tensor:
    """Dense reference transcribed from the official BD3-LM repo.

    Source: `block_diff_mask` in kuleshov-group/bd3lms models/dit.py.
    """
    idx = torch.arange(2 * seq_len, device=device)
    q_idx, kv_idx = idx.view(-1, 1), idx.view(1, -1)

    x0_flag_q = q_idx >= seq_len
    x0_flag_kv = kv_idx >= seq_len
    block_q = torch.where(x0_flag_q, (q_idx - seq_len) // block_size, q_idx // block_size)
    block_kv = torch.where(x0_flag_kv, (kv_idx - seq_len) // block_size, kv_idx // block_size)

    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
    offset_block_causal = (block_q > block_kv) & x0_flag_kv & ~x0_flag_q
    block_causal = (block_q >= block_kv) & x0_flag_kv & x0_flag_q
    return block_diagonal | offset_block_causal | block_causal


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("seq_len,block_size", [(8, 1), (8, 4), (12, 4), (16, 16), (100, 7)])
def test_block_diffusion_matches_reference(device, seq_len, block_size):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    mask_mod = generate_block_diffusion_mask(seq_len, block_size)
    q_idx = torch.arange(2 * seq_len, device=device).view(-1, 1)
    kv_idx = torch.arange(2 * seq_len, device=device).view(1, -1)
    b = h = torch.tensor(0, device=device)

    assert torch.equal(mask_mod(b, h, q_idx, kv_idx), reference_mask(seq_len, block_size, device))


def test_block_diffusion_structure():
    seq_len, block_size = 8, 4
    mask = reference_mask(seq_len, block_size)
    noised, clean = mask[:seq_len], mask[seq_len:]

    assert not clean[:, :seq_len].any(), "clean tokens must not attend to noised tokens"
    assert torch.equal(
        noised[:, :seq_len],
        torch.block_diag(*[torch.ones(block_size, block_size, dtype=torch.bool)] * 2),
    )
    assert noised[block_size, seq_len : seq_len + block_size].all()
    assert not noised[block_size - 1, seq_len:].any(), (
        "first-block noised tokens have no clean context"
    )
    assert clean[block_size, seq_len : seq_len + block_size + 1].all()
    assert not clean[block_size - 1, seq_len + block_size :].any()


def test_block_diffusion_flex_matches_sdpa():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    device, seq_len, block_size = "cuda", 256, 32
    mask_mod = generate_block_diffusion_mask(seq_len, block_size)
    block_mask = create_block_mask(mask_mod, None, None, 2 * seq_len, 2 * seq_len, device=device)

    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 4, 2 * seq_len, 64, device=device) for _ in range(3))
    out = torch.compile(flex_attention)(q, k, v, block_mask=block_mask)
    ref = F.scaled_dot_product_attention(
        q, k, v, attn_mask=reference_mask(seq_len, block_size, device)
    )
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)
