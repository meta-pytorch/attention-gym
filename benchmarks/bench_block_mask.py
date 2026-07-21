import itertools
from dataclasses import dataclass
from typing import List
import importlib
import sys

import torch
from tabulate import tabulate
from tqdm import tqdm
from attn_gym.masks import (
    causal_mask,
    generate_sliding_window,
    generate_prefix_lm_mask,
    generate_doc_mask_mod,
    generate_dilated_sliding_window,
)
from attn_gym.masks.causal import create_causal_block_mask_fast
from attn_gym.masks.document_mask import generate_random_lengths, length_to_offsets

from torch.nn.attention.flex_attention import create_block_mask, _mask_mod_signature, noop_mask

has_nuggies = importlib.util.find_spec("transformer_nuggets")
if not has_nuggies:
    print(
        "Need to install transformer_nuggets for this benchmark. "
        "Run `pip install git+https://github.com/drisspg/transformer_nuggets`"
    )
    # Exit if the dependency is missing
    sys.exit(1)

from transformer_nuggets.utils import (  # noqa: E402
    max_memory_usage,
    cuda_memory_usage,
    benchmark_cuda_function_in_microseconds_triton,
)

device = torch.device("cuda")

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


FUNCTION_BASED_CREATORS = {
    "causal": causal_mask,
    "sliding_window": generate_sliding_window,
    "prefix_lm": generate_prefix_lm_mask,
    "doc_mask_mod": generate_doc_mask_mod,
    "dilated_sliding_window": generate_dilated_sliding_window,
}

CUSTOM_CREATORS = {
    "causal_fast": lambda B, H, M, N, device: create_causal_block_mask_fast(B, H, M, N, device),
}


@dataclass(frozen=True)
class ExperimentConfig:
    B: int
    H: int
    M: int
    N: int
    mask_mod_name: str


@dataclass(frozen=True)
class ExperimentResult:
    creation_time_ms: float
    memory_bytes: int
    max_memory_usage: int


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_mask_mod(c: ExperimentConfig) -> _mask_mod_signature:
    match c.mask_mod_name:
        case "sliding_window":
            return generate_sliding_window(window_size=128)
        case "prefix_lm":
            return generate_prefix_lm_mask(prefix_length=128)
        case "doc_mask_mod":
            lengths = generate_random_lengths(total_length=c.M, num_documents=4)
            offsets = length_to_offsets(lengths, device)
            return generate_doc_mask_mod(mask_mod=noop_mask, offsets=offsets)
        case "dilated_sliding_window":
            return generate_dilated_sliding_window(window_size=128, dilation=2)
        case _:
            return FUNCTION_BASED_CREATORS[c.mask_mod_name]


def get_configs(
    mask_types: List[str] | None,
    batch_sizes: List[int],
    num_heads: List[int],
    seq_lens: List[int],
) -> List[ExperimentConfig]:
    # Map string names to mask functions
    all_available_masks = list(FUNCTION_BASED_CREATORS.keys()) + list(CUSTOM_CREATORS.keys())

    # Filter mask types if provided
    if mask_types is not None:
        # Check if all provided mask types are valid
        invalid_masks = [mask for mask in mask_types if mask not in all_available_masks]
        if invalid_masks:
            print(f"Invalid mask types: {invalid_masks}")
            print(f"Available mask types: {all_available_masks}")
            sys.exit(1)
        mask_mods_to_run = mask_types
    else:
        mask_mods_to_run = all_available_masks

    configs = []
    for B, H, S, mask_mod in itertools.product(batch_sizes, num_heads, seq_lens, mask_mods_to_run):
        configs.append(
            ExperimentConfig(
                B=B,
                H=H,
                M=S,
                N=S,
                mask_mod_name=mask_mod,
            )
        )
    return configs


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    # Determine if this is a function-based or custom creator
    assert (
        config.mask_mod_name in FUNCTION_BASED_CREATORS or config.mask_mod_name in CUSTOM_CREATORS
    ), f"Mask mod '{config.mask_mod_name}' not found."

    if config.mask_mod_name in FUNCTION_BASED_CREATORS:
        # Function-based approach using create_block_mask
        mask_mod_fn = get_mask_mod(config)
        cbm = torch.compile(create_block_mask)

        # Warmup
        for _ in range(10):
            cbm(mask_mod_fn, config.B, config.H, config.M, config.N, device=device)
        torch.cuda.synchronize(device)

        creation_time_us = benchmark_cuda_function_in_microseconds_triton(
            lambda: cbm(mask_mod_fn, config.B, config.H, config.M, config.N, device=device),
        )

        torch.cuda.synchronize(device)

        with cuda_memory_usage() as mem:
            bm = cbm(mask_mod_fn, config.B, config.H, config.M, config.N, device=device)
        del bm

        with max_memory_usage() as max_mem:
            bm = cbm(mask_mod_fn, config.B, config.H, config.M, config.N, device=device)
        del bm

    else:
        # Custom creator approach
        custom_creator = CUSTOM_CREATORS[config.mask_mod_name]
        compiled_creator = torch.compile(custom_creator)

        # Warmup
        for _ in range(10):
            compiled_creator(config.B, config.H, config.M, config.N, device)
        torch.cuda.synchronize(device)

        creation_time_us = benchmark_cuda_function_in_microseconds_triton(
            lambda: compiled_creator(config.B, config.H, config.M, config.N, device),
        )

        torch.cuda.synchronize(device)

        with cuda_memory_usage() as mem:
            bm = compiled_creator(config.B, config.H, config.M, config.N, device)
        del bm

        with max_memory_usage() as max_mem:
            bm = compiled_creator(config.B, config.H, config.M, config.N, device)
        del bm

    return ExperimentResult(
        creation_time_ms=creation_time_us / 1000,
        memory_bytes=mem.memory_usage,
        max_memory_usage=max_mem.max_memory,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "B",
        "H",
        "M",
        "N",
        "Mask Mod",
        "Creation Time (ms)",
        "Mask Size Memory (GiB)",
        "Max Construction Memory (GiB)",
    ]
    rows = []
    for experiment in experiments:
        rows.append(
            [
                experiment.config.B,
                experiment.config.H,
                experiment.config.M,
                experiment.config.N,
                experiment.config.mask_mod_name,
                f"{experiment.result.creation_time_ms:.4f}",
                f"{experiment.result.memory_bytes/(1024**3):.4f}",
                f"{experiment.result.max_memory_usage/(1024**3):.4f}",
            ]
        )
    # Sort rows for better readability (e.g., by B, H, M, N)
    rows.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    print(tabulate(rows, headers=headers, tablefmt="github"))


def main(
    mask_types: List[str] | None = None,
    batch_sizes: List[int] | None = None,
    num_heads: List[int] | None = None,
    seq_lens: List[int] | None = None,
):
    """
    Run block mask benchmarks.

    Args:
        mask_types: Optional list of mask types to benchmark. If not provided, all available mask types will be used.
                   Example usage: --mask_types '[causal, causal_fast]'
        batch_sizes: Optional list of batch sizes to benchmark. Default: [1, 4, 8]
                    Example usage: --batch_sizes '[1, 2, 4]'
        num_heads: Optional list of number of heads to benchmark. Default: [8, 16]
                  Example usage: --num_heads '[8, 12, 16]'
        seq_lens: Optional list of sequence lengths to benchmark. Default: [1024, 2048, 4096, 8192]
                 Example usage: --seq_lens '[1024, 2048]'
    """
    # Handle defaults
    if batch_sizes is None:
        batch_sizes = [1, 4, 8]
    if num_heads is None:
        num_heads = [8, 16]
    if seq_lens is None:
        seq_lens = [1024, 2048, 4096, 8192]

    torch.random.manual_seed(123)
    configs = get_configs(mask_types, batch_sizes, num_heads, seq_lens)
    results = []
    print(f"Running {len(configs)} benchmark configurations...")
    for config in tqdm(configs):
        try:
            result = run_experiment(config)
            results.append(Experiment(config=config, result=result))
        except Exception as e:
            print(f"Failed to run config {config}: {e}")

    print_results(results)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
