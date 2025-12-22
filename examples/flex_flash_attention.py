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
    backward: bool = False,
) -> tuple[float, float, float, float]:
    """Compare Flash vs Triton error against FP32 reference (forward and optionally backward)."""
    if backward:
        q = q.clone().requires_grad_(True)
        k = k.clone().requires_grad_(True)
        v = v.clone().requires_grad_(True)
        q_ref = q.float().detach().requires_grad_(True)
        k_ref = k.float().detach().requires_grad_(True)
        v_ref = v.float().detach().requires_grad_(True)

        ref = flex_attention(q_ref, k_ref, v_ref, score_mod=score_mod, block_mask=block_mask)
        grad_out = torch.randn_like(ref)
        ref.backward(grad_out)
        ref_dq, ref_dk, ref_dv = (
            q_ref.grad.to(q.dtype),
            k_ref.grad.to(q.dtype),
            v_ref.grad.to(q.dtype),
        )
        ref = ref.to(q.dtype)

        q_flash, k_flash, v_flash = (
            q.detach().requires_grad_(True),
            k.detach().requires_grad_(True),
            v.detach().requires_grad_(True),
        )
        flash_out = flex_flash_compiled(
            q_flash, k_flash, v_flash, score_mod=score_mod, block_mask=block_mask
        )
        flash_out.backward(grad_out.to(flash_out.dtype))

        q_triton, k_triton, v_triton = (
            q.detach().requires_grad_(True),
            k.detach().requires_grad_(True),
            v.detach().requires_grad_(True),
        )
        triton_out = flex_triton_compiled(
            q_triton, k_triton, v_triton, score_mod=score_mod, block_mask=block_mask
        )
        triton_out.backward(grad_out.to(triton_out.dtype))

        flash_fwd_err = (flash_out - ref).abs().max().item()
        triton_fwd_err = (triton_out - ref).abs().max().item()

        flash_bwd_err = max(
            (q_flash.grad - ref_dq).abs().max().item(),
            (k_flash.grad - ref_dk).abs().max().item(),
            (v_flash.grad - ref_dv).abs().max().item(),
        )
        triton_bwd_err = max(
            (q_triton.grad - ref_dq).abs().max().item(),
            (k_triton.grad - ref_dk).abs().max().item(),
            (v_triton.grad - ref_dv).abs().max().item(),
        )
        return flash_fwd_err, triton_fwd_err, flash_bwd_err, triton_bwd_err

    ref = flex_attention(
        q.float(), k.float(), v.float(), score_mod=score_mod, block_mask=block_mask
    ).to(q.dtype)
    flash_out = flex_flash_compiled(q, k, v, score_mod=score_mod, block_mask=block_mask)
    triton_out = flex_triton_compiled(q, k, v, score_mod=score_mod, block_mask=block_mask)

    flash_err = (flash_out - ref).abs().max().item()
    triton_err = (triton_out - ref).abs().max().item()
    return flash_err, triton_err, 0.0, 0.0


def run_comparison(B=2, H=8, S=2048, D=64, dtype=torch.bfloat16):
    """Compare Flash vs Triton numerical accuracy (forward and backward)."""
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
        flash_fwd, triton_fwd, flash_bwd, triton_bwd = compare_backends(
            q, k, v, score_mod=score_mod, block_mask=bm, backward=True
        )
        results.append(
            [
                name,
                f"{flash_fwd:.2e}",
                f"{triton_fwd:.2e}",
                f"{flash_bwd:.2e}",
                f"{triton_bwd:.2e}",
            ]
        )
    print(
        tabulate(
            results,
            headers=["Test", "Flash fwd", "Triton fwd", "Flash bwd", "Triton bwd"],
            tablefmt="grid",
        )
    )


# =============================================================================
# Benchmark
# =============================================================================


def run_benchmark(B=4, H=32, S=8192, D=128, dtype=torch.bfloat16, use_mask=True):
    """Benchmark Flash vs Triton performance (forward and backward)."""
    print("\n=== PERFORMANCE BENCHMARK ===\n")

    device = "cuda"

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

    def make_qkv():
        q = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
        return q, k, v

    q, k, v = make_qkv()
    grad_out = torch.randn(B, H, S, D, device=device, dtype=dtype)

    def run_flash_fwd():
        return flex_flash_compiled(q, k, v, block_mask=block_mask)

    def run_triton_fwd():
        return flex_triton_compiled(q, k, v, block_mask=block_mask)

    prev_donated_buffer = torch._functorch.config.donated_buffer
    torch._functorch.config.donated_buffer = False

    q_flash, k_flash, v_flash = make_qkv()
    q_triton, k_triton, v_triton = make_qkv()
    flash_out = flex_flash_compiled(q_flash, k_flash, v_flash, block_mask=block_mask)
    triton_out = flex_triton_compiled(q_triton, k_triton, v_triton, block_mask=block_mask)

    def run_flash_bwd():
        return torch.autograd.grad(
            flash_out, (q_flash, k_flash, v_flash), grad_out, retain_graph=True
        )

    def run_triton_bwd():
        return torch.autograd.grad(
            triton_out, (q_triton, k_triton, v_triton), grad_out, retain_graph=True
        )

    for _ in range(3):
        run_flash_fwd()
        run_triton_fwd()
        run_flash_bwd()
        run_triton_bwd()
    torch.cuda.synchronize()

    with cuda_kernel_profiler("flash_attncute") as flash_prof:
        run_flash_fwd()
    with cuda_kernel_profiler("flex_attention") as triton_prof:
        run_triton_fwd()

    flash_kernel = "flash_attncute" if flash_prof["found"] else "not found"
    triton_kernel = (
        "triton_tem_fused_flex_attention" if triton_prof["kernel_names"] else "not found"
    )
    print(f"\nKernels: FLASH -> {flash_kernel}, TRITON -> {triton_kernel}")

    flash_fwd_ms = do_bench_using_profiling(run_flash_fwd)
    triton_fwd_ms = do_bench_using_profiling(run_triton_fwd)
    flash_bwd_ms = do_bench_using_profiling(run_flash_bwd)
    triton_bwd_ms = do_bench_using_profiling(run_triton_bwd)

    flash_fwd_tflops = calculate_tflops(B, H, S, S, D, flash_fwd_ms, sparsity)
    triton_fwd_tflops = calculate_tflops(B, H, S, S, D, triton_fwd_ms, sparsity)
    flash_bwd_tflops = calculate_tflops(B, H, S, S, D, flash_bwd_ms / 3, sparsity)
    triton_bwd_tflops = calculate_tflops(B, H, S, S, D, triton_bwd_ms / 3, sparsity)

    results = [
        [
            "Flash",
            f"{flash_fwd_ms:.3f}",
            f"{flash_fwd_tflops:.2f}",
            f"{flash_bwd_ms:.3f}",
            f"{flash_bwd_tflops:.2f}",
        ],
        [
            "Triton",
            f"{triton_fwd_ms:.3f}",
            f"{triton_fwd_tflops:.2f}",
            f"{triton_bwd_ms:.3f}",
            f"{triton_bwd_tflops:.2f}",
        ],
    ]
    print(
        tabulate(
            results,
            headers=["Backend", "Fwd (ms)", "Fwd TFLOPs", "Bwd (ms)", "Bwd TFLOPs"],
            tablefmt="grid",
        )
    )

    fwd_speedup = triton_fwd_ms / flash_fwd_ms
    bwd_speedup = triton_bwd_ms / flash_bwd_ms
    print(f"\nFlash fwd is {fwd_speedup:.2f}x {'faster' if fwd_speedup > 1 else 'slower'}")
    print(f"Flash bwd is {bwd_speedup:.2f}x {'faster' if bwd_speedup > 1 else 'slower'}")

    torch._functorch.config.donated_buffer = prev_donated_buffer


# =============================================================================
# Limitations
# =============================================================================


def print_limitations():
    print("""
=== CURRENT LIMITATIONS ===

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
