import torch
from typing import Optional
from pathlib import Path
from contextlib import contextmanager, nullcontext
import numpy as np
import math
from torch.nn.attention.flex_attention import (
    _score_mod_signature,
    _mask_mod_signature,
    _vmap_for_bhqkv,
    _ModificationType,
    _DEFAULT_SPARSE_BLOCK_SIZE,
)
from torch.profiler import profile, ProfilerActivity

# TODO This was moved on nightly, this enables 2.5 and 2.6 | we should remove this once 2.5 is no longer supported
try:
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex
except ImportError:
    from torch._higher_order_ops.flex_attention import TransformGetItemToIndex
from torch._inductor.utils import do_bench_using_profiling
from collections.abc import Callable

Tensor = torch.Tensor


def benchmark_cuda_function_in_microseconds(func: Callable, *args, **kwargs) -> float:
    """Thin wrapper around do_bench_using_profiling"""
    no_args = lambda: func(*args, **kwargs)
    time = do_bench_using_profiling(no_args)
    return time * 1e3


def create_score_mod(
    query: torch.Tensor,
    key: torch.Tensor,
    score_mod: Optional[_score_mod_signature],
    mask_mod: Optional[_mask_mod_signature],
    device: str = "cuda",
    _compile: bool = False,
    scale: Optional[float] = None,
    batch_idx: int = 0,
    head_idx: int = 0,
) -> torch.Tensor:
    B = 1
    H = 1
    M = query.shape[0]
    N = key.shape[0]

    b = torch.arange(0, B, device=device) + batch_idx
    h = torch.arange(0, H, device=device) + head_idx
    m = torch.arange(0, M, device=device)
    n = torch.arange(0, N, device=device)

    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    type = _ModificationType.SCORE_MOD if score_mod is not None else _ModificationType.MASK_MOD
    if _compile:
        ctx = nullcontext()
    else:
        ctx = TransformGetItemToIndex()

    with ctx:
        mod_fn = score_mod if type == _ModificationType.SCORE_MOD else mask_mod
        prefix = (0,) if type == _ModificationType.SCORE_MOD else ()
        mod = _vmap_for_bhqkv(mod_fn, prefix=prefix)
        scores = query @ key.transpose(-2, -1)
        scores *= scale_factor
        scores = scores.view(1, 1, M, N)
        if type == _ModificationType.SCORE_MOD:
            out = mod(scores, b, h, m, n)
        else:
            out = mod(b, h, m, n)

    return out


def _name_to_title(name: str) -> str:
    title = name.replace("_", " ")
    title = " ".join(word.capitalize() for word in title.split())
    return title


def visualize_attention_scores(
    query: Tensor,
    key: Tensor,
    score_mod: Optional[_score_mod_signature] = None,
    mask_mod: Optional[_mask_mod_signature] = None,
    device: str = "cuda",
    name: str = "attention_scores",
    path: Optional[Path] = None,
    batch_idx: int = 0,
    head_idx: int = 0,
    scale: Optional[float] = None,
):
    """
    Generate and save a visualization of attention scores.

    Args:
        query (Tensor): Query tensor of shape (batch_size, num_heads, seq_len_q, head_dim).
        key (Tensor): Key tensor of shape (batch_size, num_heads, seq_len_k, head_dim).
        score_mod (Optional[Callable]): If this is set this will take precedence over the mask_mod.
        mask_mod (Optional[Callable]): The mask_mod function used to create block_mask
        device (str): Device to run computations on (default: "cuda").
        name (str): Base name for the file and title (default: 'attention_scores').
        path (Path): Path to save the visualization. If None, will be saved to the current working directory.
        batch_idx (int): Index of the batch to visualize (default: 0).
        head_idx (int): Index of the head to visualize (default: 0).
        scale (float): Scale factor to apply to the attention scores. If None, will be set to 1 / sqrt(head_dim).

    Returns:
        None
    """
    import matplotlib.pyplot as plt

    assert (
        score_mod is not None or mask_mod is not None
    ), "Must provide either score_mod or mask_mod"
    query = query[batch_idx, head_idx, :, :]
    key = key[batch_idx, head_idx, :, :]
    scores_viz = create_score_mod(
        query,
        key,
        score_mod=score_mod,
        mask_mod=mask_mod,
        scale=scale,
        device=device,
        batch_idx=batch_idx,
        head_idx=head_idx,
    )
    # If both score_mod and mask_mod are provided, apply both
    if score_mod is not None and mask_mod is not None:
        mask_viz = create_score_mod(
            query,
            key,
            score_mod=None,
            mask_mod=mask_mod,
            scale=scale,
            device=device,
            batch_idx=batch_idx,
            head_idx=head_idx,
        )
        # Apply mask by setting masked positions to -inf
        scores_viz = torch.where(mask_viz == 0, float("-inf"), scores_viz)

    suffix_title = f"Batch {batch_idx}, Head {head_idx}" if batch_idx != 0 or head_idx != 0 else ""

    fig, ax = plt.subplots(figsize=(12, 10))
    color = "viridis" if score_mod is not None else "cividis"
    if score_mod is not None and mask_mod is not None:
        color = "plasma"
    im = ax.imshow(scores_viz.cpu().detach()[0, 0, :, :], aspect="auto", cmap=color)
    fig.colorbar(im)

    title = _name_to_title(name)
    file_path = Path(name).with_suffix(".png") if path is None else path.with_suffix(".png")
    ax.set_title(f"{title}\n{suffix_title}", fontsize=20)

    ax.set_xlabel("Key Tokens", fontsize=18)
    ax.set_ylabel("Query Tokens", fontsize=18)

    # Move y-axis ticks and labels to the top
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)

    # Add tick labels if the number of tokens is manageable
    num_query_tokens, num_kv_tokens = scores_viz.shape[-2:]
    if num_query_tokens <= 32 and num_kv_tokens <= 32:
        ax.set_xticks(range(num_kv_tokens))
        rotation = 45 if num_kv_tokens > 12 else 0
        ax.set_xticklabels(
            [f"KV{i}" for i in range(num_kv_tokens)], fontsize=16, rotation=rotation
        )
        ax.set_yticks(range(num_query_tokens))
        ax.set_yticklabels([f"Q{i}" for i in range(num_query_tokens)], fontsize=16)
        # Align grid with pixel boundaries
        ax.set_xticks(np.arange(-0.5, num_kv_tokens, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, num_query_tokens, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=2)

    plt.tight_layout()
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)  # Close the figure to free up memory

    print(f"Visualization saved as {file_path}")


def plot_attention_scores(
    query: Tensor,
    key: Tensor,
    score_mod: Optional[_score_mod_signature] = None,
    mask_mod: Optional[_mask_mod_signature] = None,
    device: str = "cuda",
    name: str = "attention_scores",
    batch_idx: int = 0,
    head_idx: int = 0,
    scale: Optional[float] = None,
    figsize: tuple = (12, 10),
):
    """
    Generate a matplotlib figure of attention scores that can be displayed interactively.

    Args:
        query (Tensor): Query tensor of shape (batch_size, num_heads, seq_len_q, head_dim).
        key (Tensor): Key tensor of shape (batch_size, num_heads, seq_len_k, head_dim).
        score_mod (Optional[Callable]): If this is set this will take precedence over the mask_mod.
        mask_mod (Optional[Callable]): The mask_mod function used to create block_mask
        device (str): Device to run computations on (default: "cuda").
        name (str): Base name for the title (default: 'attention_scores').
        batch_idx (int): Index of the batch to visualize (default: 0).
        head_idx (int): Index of the head to visualize (default: 0).
        scale (float): Scale factor to apply to the attention scores. If None, will be set to 1 / sqrt(head_dim).
        figsize (tuple): Figure size as (width, height) (default: (12, 10)).

    Returns:
        matplotlib.figure.Figure: The figure object that can be displayed with plt.show() or in notebooks
    """
    import matplotlib.pyplot as plt

    assert (
        score_mod is not None or mask_mod is not None
    ), "Must provide either score_mod or mask_mod"
    query = query[batch_idx, head_idx, :, :]
    key = key[batch_idx, head_idx, :, :]
    scores_viz = create_score_mod(
        query,
        key,
        score_mod=score_mod,
        mask_mod=mask_mod,
        scale=scale,
        device=device,
        batch_idx=batch_idx,
        head_idx=head_idx,
    )
    # If both score_mod and mask_mod are provided, apply both
    if score_mod is not None and mask_mod is not None:
        mask_viz = create_score_mod(
            query,
            key,
            score_mod=None,
            mask_mod=mask_mod,
            scale=scale,
            device=device,
            batch_idx=batch_idx,
            head_idx=head_idx,
        )
        # Apply mask by setting masked positions to -inf
        scores_viz = torch.where(mask_viz == 0, float("-inf"), scores_viz)

    suffix_title = f"Batch {batch_idx}, Head {head_idx}" if batch_idx != 0 or head_idx != 0 else ""

    fig, ax = plt.subplots(figsize=figsize)
    color = "viridis" if score_mod is not None else "cividis"
    if score_mod is not None and mask_mod is not None:
        color = "plasma"
    im = ax.imshow(scores_viz.cpu().detach()[0, 0, :, :], aspect="auto", cmap=color)
    fig.colorbar(im)

    title = _name_to_title(name)
    ax.set_title(f"{title}\n{suffix_title}", fontsize=20)

    ax.set_xlabel("Key Tokens", fontsize=18)
    ax.set_ylabel("Query Tokens", fontsize=18)

    # Move y-axis ticks and labels to the top
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)

    # Add tick labels if the number of tokens is manageable
    num_query_tokens, num_kv_tokens = scores_viz.shape[-2:]
    if num_query_tokens <= 32 and num_kv_tokens <= 32:
        ax.set_xticks(range(num_kv_tokens))
        rotation = 45 if num_kv_tokens > 12 else 0
        ax.set_xticklabels(
            [f"KV{i}" for i in range(num_kv_tokens)], fontsize=16, rotation=rotation
        )
        ax.set_yticks(range(num_query_tokens))
        ax.set_yticklabels([f"Q{i}" for i in range(num_query_tokens)], fontsize=16)
        # Align grid with pixel boundaries
        ax.set_xticks(np.arange(-0.5, num_kv_tokens, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, num_query_tokens, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=2)

    plt.tight_layout()
    return fig


def cdiv(a: int, b: int) -> int:
    """Ceiling integer division"""
    return (a + b - 1) // b


def round_up(x: int, y: int) -> int:
    """Round up x to the nearest multiple of y"""
    return cdiv(x, y) * y


# =============================================================================
# Flash Backend Utilities
# =============================================================================


@contextmanager
def cuda_kernel_profiler(kernel_pattern: str = "flash_attncute"):
    """
    Context manager that profiles CUDA kernels and checks for a pattern.

    Usage:
        with cuda_kernel_profiler("flash_attncute") as result:
            flex_attention(...)
        print(result["found"])  # True if flash kernel was called
        print(result["kernel_names"])  # List of all CUDA kernel names
    """
    result = {"found": False, "kernel_names": []}

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        yield result

    kernel_names = [
        evt.name
        for evt in prof.events()
        if evt.device_type == torch.autograd.DeviceType.CUDA and evt.name
    ]
    result["kernel_names"] = kernel_names
    result["found"] = any(kernel_pattern in name for name in kernel_names)


def get_flash_block_size(device: str = "cuda") -> tuple[int, int]:
    """
    Get block size for Flash backend based on GPU compute capability.

    On SM100+ (Blackwell): Q block must be 256, KV block is 128.
    On SM80/SM90: Both use default 128.
    """
    q_block = _DEFAULT_SPARSE_BLOCK_SIZE
    kv_block = _DEFAULT_SPARSE_BLOCK_SIZE

    dev = torch.device(device)
    if dev.type == "cuda":
        major, _ = torch.cuda.get_device_capability(dev)
        if major >= 10:
            q_block *= 2

    return (q_block, kv_block)


def calculate_tflops(
    batch: int,
    heads: int,
    seq_q: int,
    seq_kv: int,
    head_dim: int,
    time_ms: float,
    sparsity: float = 0.0,
) -> float:
    """Calculate TFLOPs for attention forward pass."""
    qk_flops = 2 * seq_q * seq_kv * head_dim
    sv_flops = 2 * seq_q * head_dim * seq_kv
    total_flops = batch * heads * (qk_flops + sv_flops) * (1 - sparsity)
    return total_flops / (time_ms * 1e9)  # ms to TFLOPs
