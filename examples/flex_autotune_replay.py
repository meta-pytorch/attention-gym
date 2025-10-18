import json
import os
import tempfile
from unittest.mock import patch

import torch
from torch.nn.attention.flex_attention import flex_attention, FlexKernelOptions, create_block_mask
from attn_gym.utils import benchmark_cuda_function_in_microseconds
from attn_gym.masks import causal_mask

torch.compiler.config.force_disable_caches = True
torch._functorch.config.donated_buffer = False


def run_autotune(log_file: str):
    """
    Runs flex_attention with max-autotune to generate kernel tuning logs.
    """
    print("Running autotuning phase...")
    query = torch.randn(2, 2, 8192, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    key = torch.randn(2, 2, 8192, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    value = torch.randn(2, 2, 8192, 64, device="cuda", dtype=torch.float16, requires_grad=True)

    block_mask = torch.compile(create_block_mask)(
        causal_mask, None, None, 8192, 8192, device="cuda"
    )

    compiled_flex = torch.compile(flex_attention, mode="max-autotune-no-cudagraphs")

    with patch.dict(os.environ, {"TORCHINDUCTOR_FLEX_ATTENTION_LOGGING_FILE": log_file}):
        out = compiled_flex(
            query=query,
            key=key,
            value=value,
            block_mask=block_mask,
        )
        out.sum().backward()
    print(f"Autotuning logs saved to {log_file}.json")


def parse_log_and_get_best_options(log_file: str) -> FlexKernelOptions:
    """
    Parses the autotuning log and returns the best kernel options.
    """
    print("\nParsing autotuning logs...")
    json_file = log_file + ".json"
    with open(json_file) as f:
        log_data = json.load(f)

    best_options: FlexKernelOptions = {}
    for entry in log_data:
        dims_key, choices = next(iter(entry.items()))
        kernel_type = eval(dims_key)[0]  # 'forward' or 'backward'
        best_choice = choices[0]  # The list is sorted by time

        prefix = "fwd_" if kernel_type == "forward" else "bwd_"

        for key, value in best_choice.items():
            if key not in ["type", "time"]:
                # Ensure the key is valid for FlexKernelOptions
                if key in FlexKernelOptions.__annotations__:
                    best_options[f"{prefix}{key}"] = value
    print("Best kernel options extracted from logs:")
    print(json.dumps(best_options, indent=2))
    return best_options


def run_with_best_options(kernel_options: FlexKernelOptions):
    """
    Runs flex_attention with the provided kernel options.
    """
    print("\nRunning with pre-compiled best options...")
    query = torch.randn(2, 2, 8192, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    key = torch.randn(2, 2, 8192, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    value = torch.randn(2, 2, 8192, 64, device="cuda", dtype=torch.float16, requires_grad=True)

    block_mask = torch.compile(create_block_mask)(
        causal_mask, None, None, 8192, 8192, device="cuda"
    )

    # Note: We are not using max-autotune here
    compiled_flex = torch.compile(flex_attention)

    # Make sure we are not logging this run
    if "TORCHINDUCTOR_FLEX_ATTENTION_LOGGING_FILE" in os.environ:
        del os.environ["TORCHINDUCTOR_FLEX_ATTENTION_LOGGING_FILE"]

    def run_fwd():
        return compiled_flex(
            query=query,
            key=key,
            value=value,
            block_mask=block_mask,
            kernel_options=kernel_options,
        )

    # Warmup
    for _ in range(3):
        run_fwd()
    fwd_time = benchmark_cuda_function_in_microseconds(run_fwd)
    print(f"Execution time with best options: {fwd_time / 1000:.3f} ms")

    out = run_fwd()
    loss = out.sum()

    def run_bwd():
        loss.backward(retain_graph=True)

    # Warmup
    for _ in range(3):
        run_bwd()

    bwd_time = benchmark_cuda_function_in_microseconds(run_bwd)
    print(f"Backward execution time with best options: {bwd_time / 1000:.3f} ms")


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file_path = os.path.join(tmpdir, "flex_attention_configs")

        # 1. Run autotuning
        run_autotune(log_file_path)

        # 2. Parse the log file to get the best options
        best_kernel_options = parse_log_and_get_best_options(log_file_path)

        # 3. Run with the best options
        run_with_best_options(best_kernel_options)


if __name__ == "__main__":
    main()
