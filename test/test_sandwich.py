import pytest
import torch
import torch.nn.functional as F
from torch.autograd import grad
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from attn_gym.mods import generate_sandwich_bias


def reference_sandwich_bias(H: int, seq_len: int, d_bar: int = 128) -> torch.Tensor:
    """Paper's reference construction (appendix E): inner products of sinusoidal embeddings."""
    positions = torch.arange(seq_len, dtype=torch.float64)[:, None]
    i = torch.arange(d_bar // 2, dtype=torch.float64)
    angles = positions / 10000 ** (2 * i / d_bar)
    pos_embs = torch.cat([angles.sin(), angles.cos()], dim=-1)
    sandwich = pos_embs @ pos_embs.T
    compression_ratio = torch.arange(1, H + 1, dtype=torch.float64) * 8 / H
    return sandwich[None] / compression_ratio[:, None, None]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sandwich_matches_paper_reference(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    H, seq_len = 8, 64
    score_mod = generate_sandwich_bias(H, seq_len, device=device)
    ref = reference_sandwich_bias(H, seq_len).to(device)

    q_idx = torch.arange(seq_len, device=device).view(-1, 1)
    kv_idx = torch.arange(seq_len, device=device).view(1, -1)
    zero = torch.zeros(seq_len, seq_len, device=device)
    b = torch.tensor(0, device=device)
    for h in range(H):
        bias = score_mod(zero, b, torch.tensor(h, device=device), q_idx, kv_idx)
        torch.testing.assert_close(bias, ref[h].float(), rtol=1e-4, atol=1e-4)


def test_sandwich_flex_matches_sdpa_with_grads():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)
    B, H, S, D = 2, 8, 256, 64
    device = "cuda"
    score_mod = generate_sandwich_bias(H, S, device=device)
    block_mask = create_block_mask(lambda b, h, q, kv: q >= kv, None, None, S, S, device=device)

    q, k, v = (torch.randn(B, H, S, D, device=device, requires_grad=True) for _ in range(3))
    out = torch.compile(flex_attention)(q, k, v, score_mod=score_mod, block_mask=block_mask)

    causal = torch.tril(torch.ones(S, S, device=device, dtype=torch.bool))
    bias = reference_sandwich_bias(H, S).to(device).float()
    ref = F.scaled_dot_product_attention(
        q, k, v, attn_mask=bias.masked_fill(~causal, float("-inf"))
    )
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)

    grads = grad(out.sum(), (q, k, v), retain_graph=True)
    ref_grads = grad(ref.sum(), (q, k, v))
    for g, rg in zip(grads, ref_grads):
        torch.testing.assert_close(g, rg, rtol=2e-3, atol=2e-3)


def test_sandwich_flex_flash_backend():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    pytest.importorskip("flash_attn.cute", reason="flex FLASH backend needs flash-attn CuTeDSL")

    torch.manual_seed(0)
    B, H, S, D = 2, 8, 512, 64
    device, dtype = "cuda", torch.bfloat16
    score_mod = generate_sandwich_bias(H, S, device=device)
    block_mask = create_block_mask(
        lambda b, h, q, kv: q >= kv, None, None, S, S, device=device, BLOCK_SIZE=(256, 128)
    )

    q, k, v = (torch.randn(B, H, S, D, device=device, dtype=dtype) for _ in range(3))
    out = torch.compile(flex_attention)(
        q, k, v, score_mod=score_mod, block_mask=block_mask, kernel_options={"BACKEND": "FLASH"}
    )

    causal = torch.tril(torch.ones(S, S, device=device, dtype=torch.bool))
    bias = reference_sandwich_bias(H, S).to(device).float()
    ref = F.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), attn_mask=bias.masked_fill(~causal, float("-inf"))
    )
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
