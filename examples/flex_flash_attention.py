"""
FlexAttention with Flash Backend

Shows how to use flex_attention with the Flash backend (CUTE-based flash-attn).

Requirements:
    - PyTorch >= 2.10
    - flash-attn with CuTeDSL support
    - attn_gym

Usage:
    python flex_flash_attention.py                    # Run all
    python flex_flash_attention.py --mode benchmark   # Just benchmark
    python flex_flash_attention.py --mode compare     # Just numerical comparison
"""

import torch
import warnings
from functools import partial
from typing import Callable, Optional, Literal

from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask
from torch._inductor.utils import do_bench_using_profiling
from tabulate import tabulate
from attn_gym.masks import causal_mask
from attn_gym.mods import generate_alibi_bias, generate_tanh_softcap
from attn_gym.utils import get_flash_block_size, calculate_tflops, cuda_kernel_profiler

# Suppress noisy profiler warning
warnings.filterwarnings("ignore", message=".*Profiler clears events.*")
torch.nn.attention.flex_attention._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True
torch._dynamo.config.recompile_limit = 1000

# IMPORTANT: Select backend via kernel_options={"BACKEND": "FLASH"} or "TRITON"
# Using partial to pre-set the kernel_options for convenience
flex_flash_compiled = torch.compile(
    partial(flex_attention, kernel_options={"BACKEND": "FLASH"}), dynamic=False
)
flex_triton_compiled = torch.compile(
    partial(flex_attention, kernel_options={"BACKEND": "TRITON"}), dynamic=False
)


# =============================================================================
# Check Flash Availability
# =============================================================================


def check_flash_available() -> tuple[bool, str]:
    """Check if Flash backend is available."""
    try:
        from torch._inductor.kernel.flex.flex_flash_attention import ensure_flash_available

        if ensure_flash_available():
            return True, "Flash backend available"
        return False, "flash-attn not found. Install with: pip install flash-attn"
    except ImportError as e:
        return False, f"Import error: {e}"


# =============================================================================
# Numerical Comparison
# =============================================================================


def compare_backends(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    score_mod: Optional[Callable] = None,
    block_mask: Optional[BlockMask] = None,
) -> tuple[float, float]:
    """Compare Flash vs Triton error against FP32 reference."""
    ref = flex_attention(
        q.float(), k.float(), v.float(), score_mod=score_mod, block_mask=block_mask
    ).to(q.dtype)

    flash_out = flex_flash_compiled(q, k, v, score_mod=score_mod, block_mask=block_mask)
    triton_out = flex_triton_compiled(q, k, v, score_mod=score_mod, block_mask=block_mask)

    flash_err = (flash_out - ref).abs().max().item()
    triton_err = (triton_out - ref).abs().max().item()
    return flash_err, triton_err


def run_comparison(B=2, H=8, S=2048, D=64, dtype=torch.bfloat16):
    """Compare Flash vs Triton numerical accuracy."""
    print("\n=== NUMERICAL COMPARISON ===")
    print("Max absolute error vs FP32 reference:\n")

    device = "cuda"
    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)

    block_size = get_flash_block_size(device)
    causal_bm = create_block_mask(causal_mask, B, H, S, S, device=device, BLOCK_SIZE=block_size)

    tests = [
        ("No mod", None, None),
        ("Causal", None, causal_bm),
        ("ALiBi", generate_alibi_bias(H), None),
        ("Softcap", generate_tanh_softcap(30), None),
    ]

    results = []
    for name, score_mod, bm in tests:
        flash_err, triton_err = compare_backends(q, k, v, score_mod=score_mod, block_mask=bm)
        results.append([name, f"{flash_err:.2e}", f"{triton_err:.2e}"])
    print(tabulate(results, headers=["Test", "Flash err", "Triton err"], tablefmt="grid"))


# =============================================================================
# Benchmark
# =============================================================================


def run_benchmark(B=4, H=32, S=8192, D=128, dtype=torch.bfloat16, use_mask=True):
    """Benchmark Flash vs Triton performance."""
    print("\n=== PERFORMANCE BENCHMARK ===\n")

    device = "cuda"
    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)

    block_mask = None
    sparsity = 0.0

    if use_mask:
        block_size = get_flash_block_size(device)
        block_mask = create_block_mask(
            causal_mask, B, H, S, S, device=device, BLOCK_SIZE=block_size
        )
        sparsity = block_mask.sparsity() / 100
        print(f"Block size: Q={block_size[0]}, KV={block_size[1]}")

    print(f"Config: B={B}, H={H}, S={S}, D={D}, dtype={dtype}")
    if use_mask:
        print(f"Mask: causal, sparsity={sparsity*100:.1f}%")

    def run_flash():
        return flex_flash_compiled(q, k, v, block_mask=block_mask)

    def run_triton():
        return flex_triton_compiled(q, k, v, block_mask=block_mask)

    # Warmup
    for _ in range(3):
        run_flash()
        run_triton()
    torch.cuda.synchronize()

    # Profile to verify correct kernels are called
    with cuda_kernel_profiler("flash_attncute") as flash_prof:
        run_flash()
    with cuda_kernel_profiler("flex_attention") as triton_prof:
        run_triton()

    # Show which CUDA kernels are invoked under the hood
    flash_kernel = "flash_attncute" if flash_prof["found"] else "not found"
    triton_kernel = (
        "triton_tem_fused_flex_attention" if triton_prof["kernel_names"] else "not found"
    )
    print(f"\nKernels: FLASH -> {flash_kernel}, TRITON -> {triton_kernel}")

    flash_ms = do_bench_using_profiling(run_flash)
    triton_ms = do_bench_using_profiling(run_triton)

    flash_tflops = calculate_tflops(B, H, S, S, D, flash_ms, sparsity)
    triton_tflops = calculate_tflops(B, H, S, S, D, triton_ms, sparsity)

    results = [
        ["Flash", f"{flash_ms:.3f}", f"{flash_tflops:.2f}", "✓" if flash_prof["found"] else "✗"],
        [
            "Triton",
            f"{triton_ms:.3f}",
            f"{triton_tflops:.2f}",
            "✓" if not flash_prof["found"] else "-",
        ],
    ]
    print(tabulate(results, headers=["Backend", "Time (ms)", "TFLOPs", "Kernel"], tablefmt="grid"))

    speedup = triton_ms / flash_ms
    print(f"\nFlash is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")


# =============================================================================
# Limitations
# =============================================================================


def print_limitations():
    print("""
=== CURRENT LIMITATIONS ===

FUNCTIONAL:
  - Backward pass is NYI (work in progress) - forward only for now

PERFORMANCE:
  - Indexing by kv_idx is a large perf hit, we have ideas on how to improve but not yet implemented.
      SLOW: score + bias[kv_idx]
      FAST: score + bias[h] or score + bias[q_idx]

  - We are further away in performance from the peak causal implementation than need be right now. We are at like 800-900 Tflops,
       we plan to add better work scheduling soon and this should really help w/ flash perf.

  - Block size must match Flash kernel tiles, its easier to up then down right now.
      SM100+ (Blackwell): BLOCK_SIZE=(256, 128)
      SM80/90: BLOCK_SIZE=(128, 128)
      you can use get_flash_block_size() to auto-detect, or you'll get:
      ValueError: mask_block_cnt shape mismatch with the default Q_BLOCK_SIZE of 128 in the forward.

  - Dynamic shapes is not setup correctly in inductor. Will be soon.
""")


# =============================================================================
# Main
# =============================================================================


def main(
    mode: Literal["all", "compare", "benchmark"] = "all",
    B: int = 2,
    H: int = 8,
    S: int = 2048,
    D: int = 64,
):
    """
    FlexAttention with Flash Backend demo.

    Args:
        mode: What to run (all, compare, benchmark)
        B: Batch size
        H: Number of heads
        S: Sequence length
        D: Head dimension
    """
    torch.set_default_device("cuda")
    torch.manual_seed(42)

    # Always check availability
    available, msg = check_flash_available()
    print(f"\n{'✓' if available else '✗'} {msg}")
    if not available:
        return

    if mode in ("all", "compare"):
        run_comparison(B, H, S, D)

    if mode in ("all",):
        print_limitations()

    if mode in ("all", "benchmark"):
        run_benchmark(B=4, H=32, S=8192, D=128)


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("pip install jsonargparse")
    CLI(main)
