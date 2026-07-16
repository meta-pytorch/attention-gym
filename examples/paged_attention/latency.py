"""
Benchmarking the latency of a paged attention layer against a non-paged attention layer.

Command:
    python3 latency.py --setting change_max_seq_len
"""

import torch
from torch.nn.attention.flex_attention import (
    create_block_mask,
    noop_mask,
)
from torch._inductor.runtime.benchmarking import benchmarker

from utils import random_init_paged_attention, gen_offset, generate_score_mod

dtype = torch.bfloat16


def benchmark_layer(
    bsz,
    n_heads,
    max_seq_len,
    head_dim,
    paged_attention,
    batch_idx,
    input_pos,
    block_mask,
    score_mod,
    converted_block_mask,
    converted_score_mod,
    dtype=torch.bfloat16,
):
    from model import NonPagedAttentionLayer, PagedAttentionLayer

    # compile model
    non_paged_foo = torch.compile(
        NonPagedAttentionLayer(bsz, n_heads, max_seq_len, head_dim, dtype), fullgraph=True
    )
    paged_foo = torch.compile(
        PagedAttentionLayer(n_heads, head_dim, dtype, paged_attention), fullgraph=True
    )

    with torch.no_grad():
        # randomize a token embedding
        x = torch.randn(bsz, 1, n_heads * head_dim, device="cuda", dtype=dtype)

        # warmup
        for _ in range(10):
            non_paged_foo(batch_idx, input_pos, x, block_mask, score_mod)
            paged_foo(batch_idx, input_pos, x, converted_block_mask, converted_score_mod)

        # benchmark
        non_paged_latency = benchmarker.benchmark_gpu(
            lambda: non_paged_foo(batch_idx, input_pos, x, block_mask, score_mod)
        )
        paged_latency = benchmarker.benchmark_gpu(
            lambda: paged_foo(batch_idx, input_pos, x, converted_block_mask, converted_score_mod)
        )
        print(
            f"non_paged_latency: {non_paged_latency} ms, paged_latency: {paged_latency} ms, overhead: {round((paged_latency / non_paged_latency - 1.0) * 100, 2)}%"
        )


def benchmark(
    attn_type: str, page_size: int, bsz: int, max_seq_len: int, n_heads: int, head_dim: int
):
    # For decoding benchmark, we set input_pos to be half of max_seq_len
    input_pos = torch.tensor([max_seq_len // 2] * bsz, device="cuda", dtype=torch.int32).view(
        bsz, 1
    )  # [bsz, 1]
    batch_idx = torch.arange(bsz, device="cuda", dtype=torch.int32)  # [bsz]

    # init paged attention
    n_pages = (max_seq_len + page_size - 1) // page_size * bsz
    paged_attention = random_init_paged_attention(n_pages, page_size, bsz, max_seq_len)

    # Block mask
    if attn_type == "causal":
        mask_mod = gen_offset(
            torch.tensor([max_seq_len // 2] * bsz, device="cuda", dtype=torch.int32)
        )
    else:
        mask_mod = noop_mask
    block_mask = create_block_mask(mask_mod, bsz, 1, 1, max_seq_len, BLOCK_SIZE=page_size)
    converted_block_mask = paged_attention.convert_logical_block_mask(block_mask)

    # Score mod
    score_mod = generate_score_mod(attn_type)
    converted_score_mod = paged_attention.get_score_mod(score_mod, batch_idx=batch_idx)

    benchmark_layer(
        bsz,
        n_heads,
        max_seq_len,
        head_dim,
        paged_attention,
        batch_idx,
        input_pos,
        block_mask,
        score_mod,
        converted_block_mask,
        converted_score_mod,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", type=str, default="change_max_seq_len")
    args = parser.parse_args()

    if args.setting == "change_max_seq_len":
        max_seq_len_candidates = [2048, 4096, 8192, 16384, 32768]
        bsz_candidates = [32]
        page_size_candidates = [128]
    elif args.setting == "change_bsz":
        max_seq_len_candidates = [8192]
        bsz_candidates = [32, 64, 128]
        page_size_candidates = [128]
    elif args.setting == "change_page_size":
        max_seq_len_candidates = [8192]
        bsz_candidates = [32]
        page_size_candidates = [64, 128, 256]
    else:
        raise NotImplementedError

    n_heads, head_dim = 16, 64

    for attn_type in ["noop", "causal", "rel", "head_bias"]:
        print(f"\nattn_type:{attn_type}")
        for page_size in page_size_candidates:
            print(f"page_size:{page_size}")
            for bsz in bsz_candidates:
                for max_seq_len in max_seq_len_candidates:
                    torch._dynamo.reset()

                    print(
                        f"\nbsz: {bsz}, max_seq_len: {max_seq_len}, head_dim: {head_dim}, n_heads: {n_heads}"
                    )
                    benchmark(attn_type, page_size, bsz, max_seq_len, n_heads, head_dim)
