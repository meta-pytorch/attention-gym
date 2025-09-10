"""
Test for batch size invariance in FlexAttention.

This module tests whether FlexAttention implementations produce identical results
when processing entries individually vs. in batch. For any given (b, h) position,
the attention output should be the same whether computed in isolation or as part
of a larger batch.
"""

from typing import Optional, Dict, Any, List, Tuple
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    _score_mod_signature,
    _mask_mod_signature,
)

from attn_gym.masks import (
    causal_mask,
    generate_sliding_window,
    generate_prefix_lm_mask,
)
from attn_gym.mods import generate_alibi_bias, generate_tanh_softcap


def test_batch_invariance(
    score_mod: Optional[_score_mod_signature] = None,
    mask_mod: Optional[_mask_mod_signature] = None,
    B: int = 4,
    H: int = 8,
    S: int = 128,
    D: int = 64,
    tolerance: float = 1e-5,
    device: str = "cuda",
    data_type: torch.dtype = torch.float16,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Test batch invariance for FlexAttention with given configurations.
    
    Args:
        score_mod: Optional score modification function
        mask_mod: Optional mask modification function  
        B: Batch size for testing
        H: Number of attention heads
        S: Sequence length
        D: Head dimension
        tolerance: Numerical tolerance for comparison
        device: Device to run on
        data_type: Data type for tensors
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with test results including pass/fail status and metrics
    """
    torch.manual_seed(seed)
    
    # Generate random input tensors
    qkv_batched = [
        torch.randn(B, H, S, D, device=device, dtype=data_type)
        for _ in range(3)
    ]
    
    # Create block mask if mask_mod is provided
    block_mask = None
    if mask_mod is not None:
        block_mask = create_block_mask(mask_mod, B, H, S, S, device=device)
    
    # Compute batched attention
    flex_attention_fn = torch.compile(flex_attention, dynamic=False)
    batched_output = flex_attention_fn(
        *qkv_batched,
        score_mod=score_mod,
        block_mask=block_mask
    )
    
    # Compute individual attention for each batch element
    individual_outputs = []
    for b in range(B):
        qkv_individual = [tensor[b:b+1] for tensor in qkv_batched]
        
        # Create block mask for single batch element if needed
        individual_block_mask = None
        if mask_mod is not None:
            individual_block_mask = create_block_mask(mask_mod, 1, H, S, S, device=device)
        
        individual_output = flex_attention_fn(
            *qkv_individual,
            score_mod=score_mod,
            block_mask=individual_block_mask
        )
        individual_outputs.append(individual_output)
    
    # Concatenate individual outputs
    individual_concat = torch.cat(individual_outputs, dim=0)
    
    # Compare outputs
    max_diff = torch.max(torch.abs(batched_output - individual_concat)).item()
    mean_diff = torch.mean(torch.abs(batched_output - individual_concat)).item()
    
    # Check if test passes
    test_passed = max_diff <= tolerance
    
    # Find positions with largest differences for debugging
    diff_tensor = torch.abs(batched_output - individual_concat)
    max_diff_idx = torch.unravel_index(torch.argmax(diff_tensor), diff_tensor.shape)
    
    return {
        "passed": test_passed,
        "max_difference": max_diff,
        "mean_difference": mean_diff,
        "tolerance": tolerance,
        "max_diff_position": {
            "batch": max_diff_idx[0].item(),
            "head": max_diff_idx[1].item(), 
            "seq_q": max_diff_idx[2].item(),
            "dim": max_diff_idx[3].item(),
        },
        "config": {
            "B": B, "H": H, "S": S, "D": D,
            "has_score_mod": score_mod is not None,
            "has_mask_mod": mask_mod is not None,
        }
    }


def run_test_suite(
    test_configs: Dict[str, Dict[str, Any]],
    B: int = 4,
    H: int = 8, 
    S: int = 128,
    D: int = 64,
    device: str = "cuda",
    tolerance: float = 1e-5,
) -> Dict[str, Dict[str, Any]]:
    """
    Run batch invariance tests for multiple configurations.
    
    Args:
        test_configs: Dictionary of test configurations
        B, H, S, D: Tensor dimensions
        device: Device to run on
        tolerance: Numerical tolerance
        
    Returns:
        Dictionary with results for each test configuration
    """
    results = {}
    
    print(f"Running batch invariance test suite with B={B}, H={H}, S={S}, D={D}")
    print(f"Device: {device}, Tolerance: {tolerance}")
    print("=" * 70)
    
    for test_name, config in test_configs.items():
        print(f"Testing {test_name}...")
        
        try:
            result = test_batch_invariance(
                score_mod=config.get("score_mod"),
                mask_mod=config.get("mask_mod"),
                B=B, H=H, S=S, D=D,
                tolerance=tolerance,
                device=device,
            )
            
            status = "PASS" if result["passed"] else "FAIL"
            print(f"  {status}: max_diff={result['max_difference']:.2e}, "
                  f"mean_diff={result['mean_difference']:.2e}")
            
            if not result["passed"]:
                pos = result["max_diff_position"]
                print(f"    Max diff at batch={pos['batch']}, head={pos['head']}, "
                      f"seq={pos['seq_q']}, dim={pos['dim']}")
            
            results[test_name] = result
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            results[test_name] = {
                "passed": False,
                "error": str(e),
                "config": config,
            }
    
    print("=" * 70)
    
    # Summary
    passed_tests = sum(1 for r in results.values() if r.get("passed", False))
    total_tests = len(results)
    print(f"Summary: {passed_tests}/{total_tests} tests passed")
    
    return results


# Test configurations
TEST_CONFIGS = {
    "no_modifications": {
        # Pure attention without any modifications
    },
    "causal_mask": {
        "mask_mod": causal_mask,
    },
    "alibi_bias": {
        "score_mod": generate_alibi_bias(8),
    },
    "sliding_window": {
        "mask_mod": generate_sliding_window(window_size=32),
    },
    "prefix_lm": {
        "mask_mod": generate_prefix_lm_mask(prefix_length=64),
    },
    "softcap": {
        "score_mod": generate_tanh_softcap(30, approx=False),
    },
    "softcap_approx": {
        "score_mod": generate_tanh_softcap(30, approx=True),
    },
    "causal_plus_alibi": {
        "mask_mod": causal_mask,
        "score_mod": generate_alibi_bias(8),
    },
    "sliding_window_plus_softcap": {
        "mask_mod": generate_sliding_window(window_size=32),
        "score_mod": generate_tanh_softcap(30, approx=True),
    },
}


def main(
    tests: List[str] = ["all"],
    batch_size: int = 4,
    num_heads: int = 8,
    seq_len: int = 128,
    head_dim: int = 64,
    device: str = "cuda",
    tolerance: float = 1e-5,
    list_tests: bool = False,
):
    """
    Main function to run batch invariance tests.
    
    Args:
        tests: List of test names to run, or ["all"] for all tests
        batch_size: Batch size for testing
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Head dimension
        device: Device to run tests on
        tolerance: Numerical tolerance for comparison
        list_tests: If True, just list available tests and exit
    """
    if list_tests:
        print("Available tests:")
        for test_name in TEST_CONFIGS.keys():
            config = TEST_CONFIGS[test_name]
            desc_parts = []
            if config.get("mask_mod"):
                desc_parts.append(f"mask: {config['mask_mod'].__name__}")
            if config.get("score_mod"):
                desc_parts.append(f"score: {config['score_mod'].__name__}")
            if not desc_parts:
                desc_parts.append("no modifications")
            print(f"  {test_name}: {', '.join(desc_parts)}")
        return
    
    # Select tests to run
    if "all" in tests:
        configs_to_run = TEST_CONFIGS
    else:
        configs_to_run = {name: TEST_CONFIGS[name] for name in tests if name in TEST_CONFIGS}
        
        # Check for unknown test names
        unknown_tests = [name for name in tests if name not in TEST_CONFIGS and name != "all"]
        if unknown_tests:
            print(f"Warning: Unknown test names: {unknown_tests}")
            print(f"Available tests: {list(TEST_CONFIGS.keys())}")
    
    if not configs_to_run:
        print("No valid tests selected. Use --list-tests to see available options.")
        return
    
    # Set default device based on availability
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    
    # Run the test suite
    results = run_test_suite(
        test_configs=configs_to_run,
        B=batch_size,
        H=num_heads,
        S=seq_len,
        D=head_dim,
        device=device,
        tolerance=tolerance,
    )
    
    # Check if any tests failed
    failed_tests = [name for name, result in results.items() if not result.get("passed", False)]
    if failed_tests:
        print(f"\nFailed tests: {failed_tests}")
        exit(1)
    else:
        print("\nAll tests passed! ✅")


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    CLI(main)