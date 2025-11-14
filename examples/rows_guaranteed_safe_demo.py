#!/usr/bin/env python3
"""
Repro for the subtle ROWS_GUARANTEED_SAFE contract: we build block-sparse metadata
that declares block 0 "safe" for every row, but at runtime the first query block's
row 0 only attends once the window reaches block 1. Later rows still use block 0,
so the metadata looks perfectly valid even though row 0 has no surviving entries
in its first block.

Under torch.compile the kernel trusts the metadata, skips the masked-row guard,
and evaluates exp2(-inf - -inf) → NaN. Eager materializes the full row, sees the
empty block, and stays finite.
"""

from __future__ import annotations

import warnings

from tabulate import tabulate

import torch
import torch.nn.attention.flex_attention as fa

Q_LEN = 128
KV_LEN = 256
BLOCK_SIZE = 128


fa._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True

warnings.filterwarnings(
    "ignore",
    message="`isinstance\\(treespec, LeafSpec\\)` is deprecated",
    category=FutureWarning,
)


def describe_mask(mask_like, label: str | None = None) -> None:
    if label:
        print(label)
    if isinstance(mask_like, fa.BlockMask):
        dense_blocks = mask_like.to_dense()
        q_block, kv_block = mask_like.BLOCK_SIZE
        dense = dense_blocks.repeat_interleave(q_block, dim=-2).repeat_interleave(kv_block, dim=-1)
        q_len, kv_len = mask_like.seq_lengths
        mask = dense[..., :q_len, :kv_len][0, 0]
    else:
        mask = mask_like[0, 0]
    print("Row summaries (first four queries):")
    for q in range(4):
        allowed = torch.nonzero(mask[q], as_tuple=False).squeeze(-1)
        if allowed.numel() == 0:
            desc = "EMPTY"
        else:
            desc = f"{int(allowed[0])}..{int(allowed[-1])}"
        print(f"q={q}: {desc}")
    print(f"(Total queries={mask.size(0)}, keys={mask.size(1)})\n")


def _rand_inputs(device, dtype):
    q = torch.randn(1, 1, Q_LEN, 16, dtype=dtype, device=device)
    k = torch.randn(1, 1, KV_LEN, 16, dtype=dtype, device=device)
    v = torch.randn_like(k)
    return q, k, v


def run_compiled_flag(device: torch.device, dtype: torch.dtype, block_mask, rows_safe: bool):
    q, k, v = _rand_inputs(device, dtype)
    compiled_flex_attention = torch.compile(fa.flex_attention, fullgraph=True)
    out = compiled_flex_attention(
        q,
        k,
        v,
        block_mask=block_mask,
        kernel_options={
            "ROWS_GUARANTEED_SAFE": rows_safe,
            "FORCE_USE_FLEX_ATTENTION": True,
        },
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return out


def run_eager_flag(device: torch.device, dtype: torch.dtype, block_mask, rows_safe: bool):
    q, k, v = _rand_inputs(device, dtype)
    out = fa.flex_attention(
        q,
        k,
        v,
        block_mask=block_mask,
        kernel_options={
            "ROWS_GUARANTEED_SAFE": rows_safe,
            "FORCE_USE_FLEX_ATTENTION": True,
        },
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    torch.manual_seed(0)

    def subtle_mask_mod(b, h, q_idx, kv_idx):
        first_row = q_idx == 0
        not_first_kv_block = kv_idx >= BLOCK_SIZE
        return torch.where(first_row & ~not_first_kv_block, False, True)

    dense_mask = fa.create_mask(
        subtle_mask_mod, B=1, H=1, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device
    )
    describe_mask(dense_mask, label="Runtime mask")

    block_mask = fa.create_block_mask(
        subtle_mask_mod,
        B=1,
        H=1,
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        BLOCK_SIZE=BLOCK_SIZE,
        device=device,
    )
    describe_mask(block_mask, label="BlockMask metadata")

    print("WARNING: metadata marks block 0 as safe, but active mask zeroes it out for q=0.")
    print(
        "         Compile path assumes safety and will NaN; eager path materializes full rows.\n"
    )

    out_safe = run_compiled_flag(device, dtype, block_mask, rows_safe=False)
    out_fast_compiled = run_compiled_flag(device, dtype, block_mask, rows_safe=True)
    out_fast_eager = run_eager_flag(device, dtype, block_mask, rows_safe=True)

    results = [
        ("eager", True, torch.isnan(out_fast_eager).any().item()),
        ("compiled", False, torch.isnan(out_safe).any().item()),
        ("compiled", True, torch.isnan(out_fast_compiled).any().item()),
    ]

    headers = ["Mode", "ROWS_GUARANTEED_SAFE", "Has NaN"]
    print(tabulate(results, headers=headers, tablefmt="github"))


if __name__ == "__main__":
    main()
