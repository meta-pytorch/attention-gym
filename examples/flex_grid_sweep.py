"""
Manual exhaustive kernel config sweep for flex attention.

To use this script:
1. Edit the MASK_MOD and SCORE_MOD variables below to specify your attention pattern
2. Run the script with your desired problem size parameters

Example mask_mod configurations:
  from attn_gym.masks import causal_mask
  MASK_MOD = causal_mask

  from attn_gym.masks import generate_sliding_window
  MASK_MOD = generate_sliding_window(window_size=512)

Example score_mod configurations:
  from attn_gym.mods import generate_alibi_bias
  SCORE_MOD = generate_alibi_bias(num_heads=32)

  from attn_gym.mods import generate_tanh_softcap
  SCORE_MOD = generate_tanh_softcap(softcap=50.0)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.nn.attention.flex_attention import FlexKernelOptions, create_block_mask, flex_attention
from attn_gym.utils import benchmark_cuda_function_in_microseconds
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore", message=".*dynamo_pgo force disabled.*")
warnings.filterwarnings("ignore", message=".*Please use the new API settings to control TF32.*")
warnings.filterwarnings("ignore", message=".*isinstance.*LeafSpec.*is deprecated.*")

torch._dynamo.config.recompile_limit = 1000000

# ============================================================================
# EDIT THESE TO CONFIGURE YOUR ATTENTION PATTERN
# ============================================================================
# Example: Uncomment the lines below to use causal masking
# from attn_gym.masks import causal_mask
# MASK_MOD = causal_mask

MASK_MOD = (
    None  # Set to your mask_mod callable (e.g., causal_mask or generate_sliding_window(...))
)
SCORE_MOD = None  # Set to your score_mod callable (e.g., generate_alibi_bias(...))


# ============================================================================
def _sanitize_descriptor_for_path(descriptor: str) -> str:
    """Sanitize descriptors so they are safe to embed in filenames."""
    if not descriptor:
        return "none"
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in descriptor)


@dataclass
class FlexConfig:
    """Forward kernel configuration."""

    BLOCK_M: int
    BLOCK_N: int
    num_stages: int
    num_warps: int


@dataclass
class FlexBwdConfig:
    """Backward kernel configuration."""

    BLOCK_M1: int
    BLOCK_N1: int
    BLOCK_M2: int
    BLOCK_N2: int
    num_stages: int
    num_warps: int


def generate_exhaustive_fwd_configs() -> list[FlexConfig]:
    """
    Generate all forward configs matching PyTorch's exhaustive search space.
    Total: 4 * 3 * 4 * 3 = 144 configs
    """
    configs = [
        FlexConfig(BLOCK_M, BLOCK_N, num_stages, num_warps)
        for BLOCK_M in [16, 32, 64, 128]
        for BLOCK_N in [32, 64, 128]
        for num_stages in [1, 3, 4, 5]
        for num_warps in [2, 4, 8]
    ]
    print(f"Generated {len(configs)} forward configs")
    return configs


def generate_exhaustive_bwd_configs() -> list[FlexBwdConfig]:
    """
    Generate all backward configs matching PyTorch's exhaustive search space.
    Includes kernel static assertion filters.
    """
    configs = [
        FlexBwdConfig(BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2, num_stages, num_warps)
        for BLOCK_M1 in [16, 32, 64, 128]
        for BLOCK_N1 in [16, 32, 64, 128]
        for BLOCK_M2 in [16, 32, 64, 128]
        for BLOCK_N2 in [16, 32, 64, 128]
        for num_stages in [1, 3, 4]
        for num_warps in [2, 4, 8]
        if BLOCK_N1 % BLOCK_M1 == 0 and BLOCK_M2 % BLOCK_N2 == 0  # kernel assertions
    ]
    print(f"Generated {len(configs)} backward configs")
    return configs


def generate_reduced_bwd_configs() -> list[FlexBwdConfig]:
    """
    Generate a smaller but still comprehensive set of backward configs.
    Useful for faster iteration.
    """
    configs = [
        FlexBwdConfig(BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2, num_stages, num_warps)
        for BLOCK_M1 in [32, 64]
        for BLOCK_N1 in [32, 64, 128]
        for BLOCK_M2 in [32, 64, 128]
        for BLOCK_N2 in [32, 64]
        for num_stages in [1, 3, 4]
        for num_warps in [4, 8]
        if BLOCK_N1 % BLOCK_M1 == 0 and BLOCK_M2 % BLOCK_N2 == 0
    ]
    print(f"Generated {len(configs)} reduced backward configs")
    return configs


def config_to_kernel_options(
    fwd_config: Optional[FlexConfig] = None, bwd_config: Optional[FlexBwdConfig] = None
) -> FlexKernelOptions:
    """Convert config objects to FlexKernelOptions dict."""
    options: FlexKernelOptions = {}

    if fwd_config is not None:
        options["fwd_BLOCK_M"] = fwd_config.BLOCK_M
        options["fwd_BLOCK_N"] = fwd_config.BLOCK_N
        options["fwd_num_stages"] = fwd_config.num_stages
        options["fwd_num_warps"] = fwd_config.num_warps

    if bwd_config is not None:
        options["bwd_BLOCK_M1"] = bwd_config.BLOCK_M1
        options["bwd_BLOCK_N1"] = bwd_config.BLOCK_N1
        options["bwd_BLOCK_M2"] = bwd_config.BLOCK_M2
        options["bwd_BLOCK_N2"] = bwd_config.BLOCK_N2
        options["bwd_num_stages"] = bwd_config.num_stages
        options["bwd_num_warps"] = bwd_config.num_warps

    return options


def benchmark_fwd_config(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask,
    fwd_config: FlexConfig,
    score_mod=None,
    warmup_iters: int = 3,
) -> tuple[float, bool]:
    """
    Benchmark a single forward configuration.
    Returns: (time_in_us, success)
    """
    kernel_options = config_to_kernel_options(fwd_config=fwd_config)

    # Compile with this specific config
    compiled_flex = torch.compile(flex_attention)

    def run_fwd():
        return compiled_flex(
            query=query,
            key=key,
            value=value,
            block_mask=block_mask,
            score_mod=score_mod,
            kernel_options=kernel_options,
            enable_gqa=True,
        )

    try:
        # Warmup
        for _ in range(warmup_iters):
            _ = run_fwd()
            torch.cuda.synchronize()

        # Benchmark
        time_us = benchmark_cuda_function_in_microseconds(run_fwd)

        # Clean up
        compiled_flex = None
        torch.cuda.empty_cache()

        return time_us, True

    except Exception as e:
        print(f"  Failed: {e}")
        # Clean up on failure
        torch.cuda.empty_cache()
        return float("inf"), False


def benchmark_bwd_config(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask,
    bwd_config: FlexBwdConfig,
    fwd_config: Optional[FlexConfig] = None,
    score_mod=None,
    warmup_iters: int = 3,
) -> tuple[float, bool]:
    """
    Benchmark a single backward configuration.
    Returns: (time_in_us, success)
    """
    kernel_options = config_to_kernel_options(fwd_config=fwd_config, bwd_config=bwd_config)

    # Compile with this specific config
    compiled_flex = torch.compile(flex_attention)

    def run_fwd():
        return compiled_flex(
            query=query,
            key=key,
            value=value,
            block_mask=block_mask,
            score_mod=score_mod,
            kernel_options=kernel_options,
            enable_gqa=True,
        )

    try:
        # Warmup forward + backward
        for _ in range(warmup_iters):
            out = run_fwd()
            loss = out.sum()
            loss.backward()
            query.grad = None
            key.grad = None
            value.grad = None
            out = None
            loss = None
            torch.cuda.synchronize()

        # Benchmark just backward
        out = run_fwd()
        loss = out.sum()

        def run_bwd():
            loss.backward(retain_graph=True)

        time_us = benchmark_cuda_function_in_microseconds(run_bwd)

        # Clean up
        query.grad = None
        key.grad = None
        value.grad = None
        out = None
        loss = None
        compiled_flex = None
        torch.cuda.empty_cache()

        return time_us, True

    except Exception as e:
        print(f"  Failed: {e}")
        # Clean up on failure
        query.grad = None
        key.grad = None
        value.grad = None
        torch.cuda.empty_cache()
        return float("inf"), False


def sweep_forward_configs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask,
    score_mod=None,
    configs: Optional[list[FlexConfig]] = None,
) -> tuple[FlexConfig, float, list[dict]]:
    """
    Sweep all forward configs and find the best one.
    Returns: (best_config, best_time_us, all_results)
    """
    if configs is None:
        configs = generate_exhaustive_fwd_configs()

    print(f"\nSweeping {len(configs)} forward configurations...")

    results = []
    best_time = float("inf")
    best_config = None

    for config in tqdm(configs, desc="Forward configs", unit="config"):
        time_us, success = benchmark_fwd_config(
            query, key, value, block_mask, config, score_mod=score_mod
        )

        result = {
            "config": {
                "BLOCK_M": config.BLOCK_M,
                "BLOCK_N": config.BLOCK_N,
                "num_stages": config.num_stages,
                "num_warps": config.num_warps,
            },
            "time_us": time_us if success else None,
            "time_ms": time_us / 1000 if success else None,
            "success": success,
        }
        results.append(result)

        if success and time_us < best_time:
            best_time = time_us
            best_config = config
            tqdm.write(f"  New best! {config} -> {time_us/1000:.3f} ms")

    # Sort results by time
    results.sort(key=lambda x: x["time_us"] if x["success"] else float("inf"))

    print(f"\n✓ Best forward config: {best_config}")
    print(f"✓ Best time: {best_time/1000:.3f} ms")

    return best_config, best_time, results


def sweep_backward_configs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask,
    best_fwd_config: Optional[FlexConfig] = None,
    configs: Optional[list[FlexBwdConfig]] = None,
    use_reduced: bool = True,
    score_mod=None,
) -> tuple[FlexBwdConfig, float, list[dict]]:
    """
    Sweep all backward configs and find the best one.
    Returns: (best_config, best_time_us, all_results)
    """
    if configs is None:
        if use_reduced:
            configs = generate_reduced_bwd_configs()
        else:
            configs = generate_exhaustive_bwd_configs()

    print(f"\nSweeping {len(configs)} backward configurations...")

    results = []
    best_time = float("inf")
    best_config = None

    for config in tqdm(configs, desc="Backward configs", unit="config"):
        time_us, success = benchmark_bwd_config(
            query,
            key,
            value,
            block_mask,
            config,
            fwd_config=best_fwd_config,
            score_mod=score_mod,
        )

        result = {
            "config": {
                "BLOCK_M1": config.BLOCK_M1,
                "BLOCK_N1": config.BLOCK_N1,
                "BLOCK_M2": config.BLOCK_M2,
                "BLOCK_N2": config.BLOCK_N2,
                "num_stages": config.num_stages,
                "num_warps": config.num_warps,
            },
            "time_us": time_us if success else None,
            "time_ms": time_us / 1000 if success else None,
            "success": success,
        }
        results.append(result)

        if success and time_us < best_time:
            best_time = time_us
            best_config = config
            tqdm.write(f"  New best! {config} -> {time_us/1000:.3f} ms")

    # Sort results by time
    results.sort(key=lambda x: x["time_us"] if x["success"] else float("inf"))

    print(f"\n✓ Best backward config: {best_config}")
    print(f"✓ Best time: {best_time/1000:.3f} ms")

    return best_config, best_time, results


def main(
    B: int = 4,
    H: int = 16,
    KV_H: int = 16,
    S: int = 1024,
    D: int = 128,
    dtype_str: str = "bfloat16",
    use_reduced_bwd: bool = True,
):
    # Setup
    print("=" * 80)
    print("Manual Flex Attention Config Sweep")
    print("=" * 80)

    # Warn if both are None (dense attention)
    if MASK_MOD is None and SCORE_MOD is None:
        print("\n⚠️  WARNING: Both MASK_MOD and SCORE_MOD are None.")
        print("   This will benchmark dense attention with no masking or score modifications.")
        print("   To use a specific attention pattern, edit MASK_MOD and/or SCORE_MOD")
        print("   at the top of this file.\n")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[dtype_str]

    # Get descriptive names for printing
    mask_descriptor = (
        f"{MASK_MOD.__module__}.{MASK_MOD.__name__}" if MASK_MOD is not None else "none"
    )
    score_descriptor = (
        f"{SCORE_MOD.__module__}.{SCORE_MOD.__name__}" if SCORE_MOD is not None else "none"
    )

    print("\nConfiguration:")
    print(f"  mask_mod: {mask_descriptor}")
    print(f"  score_mod: {score_descriptor}")
    print(f"  use_reduced_bwd: {use_reduced_bwd}")

    # Create test data with GQA
    print(f"\nProblem size: B={B}, H={H}, KV_H={KV_H}, S={S}, D={D}, dtype={dtype_str}")

    query = torch.randn(B, H, S, D, device="cuda", dtype=dtype, requires_grad=True)
    key = torch.randn(B, KV_H, S, D, device="cuda", dtype=dtype, requires_grad=True)
    value = torch.randn(B, KV_H, S, D, device="cuda", dtype=dtype, requires_grad=True)

    block_mask = None
    if MASK_MOD is not None:
        try:
            block_mask = torch.compile(create_block_mask)(
                MASK_MOD, None, None, S, S, device="cuda"
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create block mask using mask_mod '{mask_descriptor}'."
            ) from exc
        print(f"Block mask sparsity: {block_mask.sparsity():.2f}%")
        print(f"Block mask density: {(100 - block_mask.sparsity()):.2f}%")
    else:
        print("Block mask: none (dense attention)")

    # Sweep forward configs
    best_fwd_config, best_fwd_time, fwd_results = sweep_forward_configs(
        query, key, value, block_mask, score_mod=SCORE_MOD
    )

    # Sweep backward configs (using best forward config)
    best_bwd_config, best_bwd_time, bwd_results = sweep_backward_configs(
        query,
        key,
        value,
        block_mask,
        best_fwd_config=best_fwd_config,
        use_reduced=use_reduced_bwd,
        score_mod=SCORE_MOD,
    )

    # Combine best configs into FlexKernelOptions
    best_kernel_options = config_to_kernel_options(
        fwd_config=best_fwd_config,
        bwd_config=best_bwd_config,
    )

    descriptor_parts = []
    if mask_descriptor:
        descriptor_parts.append(mask_descriptor)
    if score_descriptor:
        descriptor_parts.append(score_descriptor)
    attention_descriptor = " + ".join(descriptor_parts) if descriptor_parts else "dense"

    print("\n" + "=" * 80)
    print(f"FINAL RESULTS - {attention_descriptor}")
    print("=" * 80)
    print("\nBest FlexKernelOptions:")
    print(json.dumps(best_kernel_options, indent=2))
    print(f"\nForward time: {best_fwd_time/1000:.3f} ms")
    print(f"Backward time: {best_bwd_time/1000:.3f} ms")
    print(f"Total time: {(best_fwd_time + best_bwd_time)/1000:.3f} ms")

    # Save results
    output = {
        "attention_type": attention_descriptor,
        "mask_mod": mask_descriptor,
        "score_mod": score_descriptor,
        "problem_size": {"B": B, "H": H, "KV_H": KV_H, "S": S, "D": D, "dtype": dtype_str},
        "best_kernel_options": best_kernel_options,
        "best_fwd_time_ms": best_fwd_time / 1000,
        "best_bwd_time_ms": best_bwd_time / 1000,
        "top_10_fwd_configs": fwd_results[:10],
        "top_10_bwd_configs": bwd_results[:10],
        "use_reduced_bwd": use_reduced_bwd,
    }

    results_dir = Path("data")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = (
        results_dir
        / f"flex_grid_sweep_{_sanitize_descriptor_for_path(mask_descriptor)}_{_sanitize_descriptor_for_path(score_descriptor)}.json"
    )
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Results saved to {results_path}")


if __name__ == "__main__":
    # Disable caching to ensure fresh compilations
    torch.compiler.config.force_disable_caches = True
    torch._functorch.config.donated_buffer = False

    try:
        from jsonargparse import CLI
    except ImportError as exc:  # pragma: no cover - CLI import guard
        raise ImportError("Please install jsonargparse: pip install jsonargparse") from exc

    # Usage:
    # python flex_grid_sweep.py --B 4 --H 16 --KV_H 16 --S 1024 --D 128 --dtype_str bfloat16
    # Or with defaults: python flex_grid_sweep.py
    CLI(main)
