import pytest
import torch
import torch.nn.functional as F

from attn_gym.masks import generate_global_sliding_window


def reference_mask(window_size: int, is_global: torch.Tensor) -> torch.Tensor:
    seq_len = is_global.shape[0]
    idx = torch.arange(seq_len, device=is_global.device)
    window = (idx.view(-1, 1) - idx.view(1, -1)).abs() <= window_size
    return window | is_global.view(-1, 1) | is_global.view(1, -1)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("window_size", [1, 3, 8])
@pytest.mark.parametrize("global_positions", [[], [0], [0, 7, 15]])
def test_global_sliding_window_matches_reference(device, window_size, global_positions):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    seq_len = 16
    is_global = torch.zeros(seq_len, dtype=torch.bool, device=device)
    is_global[global_positions] = True

    mask_mod = generate_global_sliding_window(window_size, is_global)
    q_idx = torch.arange(seq_len, device=device).view(-1, 1)
    kv_idx = torch.arange(seq_len, device=device).view(1, -1)
    b = h = torch.tensor(0, device=device)

    assert torch.equal(mask_mod(b, h, q_idx, kv_idx), reference_mask(window_size, is_global))


def test_global_sliding_window_flex_matches_sdpa():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    device, seq_len, window_size = "cuda", 256, 16
    is_global = torch.zeros(seq_len, dtype=torch.bool, device=device)
    is_global[[0, 100]] = True

    mask_mod = generate_global_sliding_window(window_size, is_global)
    block_mask = create_block_mask(mask_mod, None, None, seq_len, seq_len, device=device)

    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 4, seq_len, 64, device=device) for _ in range(3))
    out = torch.compile(flex_attention)(q, k, v, block_mask=block_mask)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=reference_mask(window_size, is_global))
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)
