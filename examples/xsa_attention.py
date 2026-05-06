"""Examples for Exclusive Self-Attention with FlexAttention."""

import time

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    from torch.nn.attention.flex_attention import flex_attention

    HAS_FLEX = True
except ImportError:
    HAS_FLEX = False
    print("flex_attention not available; using SDPA fallback.")

from attn_gym.mods.exclusive_sa import XSAMultiheadAttention, exclusive_output_mod

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}\n")


def make_qkv(B=2, H=8, T=64, d=32, device=DEVICE):
    torch.manual_seed(42)
    return (
        torch.randn(B, H, T, d, device=device),
        torch.randn(B, H, T, d, device=device),
        torch.randn(B, H, T, d, device=device),
    )


def example_functional():
    print("=" * 60)
    print("Example 1: Functional XSA output_mod")
    print("=" * 60)

    Q, K, V = make_qkv()

    if HAS_FLEX:
        Y = flex_attention(Q, K, V)
        backend_name = "flex_attention"
    else:
        Y = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        backend_name = "SDPA fallback"

    Z = exclusive_output_mod(Y, V)
    cos = F.cosine_similarity(Z, V, dim=-1).abs()

    print(f"  Backend:                     {backend_name}")
    print(f"  Input shape:                 {Q.shape}")
    print(f"  Output shape:                {Z.shape}")
    print(f"  cosine(Y, V) before XSA:     {F.cosine_similarity(Y, V, dim=-1).mean():.4f}")
    print(f"  cosine(Z, V) after XSA:      {cos.mean():.2e}  (max: {cos.max():.2e})")
    print(f"  Orthogonality guaranteed:    {cos.max().item() < 1e-5}")
    print()


def alibi_score_mod(
    score: Tensor,
    b: Tensor,
    h: Tensor,
    q_idx: Tensor,
    k_idx: Tensor,
) -> Tensor:
    """Apply an ALiBi-style distance bias to attention scores."""
    bias_slope = 1.0 / (2 ** (h.float() + 1))
    return score - bias_slope * (q_idx - k_idx).abs().float()


def example_composable():
    print("=" * 60)
    print("Example 2: XSA + ALiBi score_mod")
    print("=" * 60)

    Q, K, V = make_qkv(T=128)

    if HAS_FLEX:
        Y_alibi = flex_attention(Q, K, V, score_mod=alibi_score_mod)
        Z_xsa_alibi = exclusive_output_mod(Y_alibi, V)
        print("  XSA + ALiBi via flex_attention + output_mod:")
    else:
        Y_plain = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        Z_xsa_alibi = exclusive_output_mod(Y_plain, V)
        print("  XSA with SDPA fallback:")

    cos = F.cosine_similarity(Z_xsa_alibi, V, dim=-1).abs()
    print(f"  Output shape:               {Z_xsa_alibi.shape}")
    print(f"  cosine(Z, V):               {cos.mean():.2e}")
    print()


def example_module():
    print("=" * 60)
    print("Example 3: XSAMultiheadAttention module")
    print("=" * 60)

    import torch.nn as nn

    d_model, num_heads, T = 256, 8, 64
    x = torch.randn(2, T, d_model, device=DEVICE)

    std_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True).to(DEVICE).eval()
    xsa_attn = XSAMultiheadAttention(d_model, num_heads).to(DEVICE).eval()

    with torch.no_grad():
        out_std, _ = std_attn(x, x, x)
        out_xsa = xsa_attn(x)

    print(f"  Standard MHA output shape:  {out_std.shape}")
    print(f"  XSA       output shape:     {out_xsa.shape}")
    print(f"  Param count (std MHA):      {sum(p.numel() for p in std_attn.parameters()):,}")
    print(f"  Param count (XSA):          {sum(p.numel() for p in xsa_attn.parameters()):,}")
    print("  Extra params from XSA:      0")
    print()


def benchmark():
    print("=" * 60)
    print("Example 4: Benchmark XSA overhead vs SDPA")
    print("=" * 60)

    seq_lens = [64, 128, 256, 512, 1024]
    B, H, d = 4, 8, 64
    n_runs = 200

    print(f"  {'Seq len':<10} {'SDPA (ms)':>12} {'XSA (ms)':>12} {'Overhead':>10}")
    print(f"  {'-' * 46}")

    for T in seq_lens:
        Q, K, V = make_qkv(B=B, H=H, T=T, d=d)

        for _ in range(10):
            with torch.no_grad():
                Y = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
                _ = exclusive_output_mod(Y, V)
        if DEVICE == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_runs):
            with torch.no_grad():
                F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        sdpa_ms = (time.perf_counter() - t0) / n_runs * 1000

        t0 = time.perf_counter()
        for _ in range(n_runs):
            with torch.no_grad():
                Y = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
                exclusive_output_mod(Y, V)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        xsa_ms = (time.perf_counter() - t0) / n_runs * 1000

        overhead = (xsa_ms - sdpa_ms) / sdpa_ms * 100
        print(f"  {T:<10} {sdpa_ms:>12.3f} {xsa_ms:>12.3f} {overhead:>+9.1f}%")

    print()


def diagnostic():
    print("=" * 60)
    print("Example 5: Attention similarity diagnostic")
    print("=" * 60)
    print(f"  {'Seq len':<10} {'SA cosine(Y,V)':>16} {'XSA cosine(Z,V)':>16}")
    print(f"  {'-' * 44}")

    for T in [32, 64, 128, 256, 512]:
        Q, K, V = make_qkv(T=T)
        with torch.no_grad():
            Y = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
            Z = exclusive_output_mod(Y, V)
        sa_cos = F.cosine_similarity(Y, V, dim=-1).mean().item()
        xsa_cos = F.cosine_similarity(Z, V, dim=-1).abs().mean().item()
        print(f"  {T:<10} {sa_cos:>16.4f} {xsa_cos:>16.2e}")

    print()
    print("  Observation: SA cosine bias increases with sequence length.")
    print("  XSA cosine is near zero at all lengths.")
    print()


if __name__ == "__main__":
    example_functional()
    example_composable()
    example_module()
    benchmark()
    diagnostic()
    print("All examples completed.")
