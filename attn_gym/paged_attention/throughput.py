"""
Benchmmarking the throughput of paged attention layer in terms of the
maximum batch size that can be served.

The benchmark is based on the prompt and response length distribution
collected from OpenOrca dataset (https://huggingface.co/datasets/Open-Orca/OpenOrca),
including ~1M GPT-4 completions and ~3.2M GPT-3.5 completions.

For a fair comparison, we assume 4GB KV cache memory budget w/ and w/o paged attention.
We assume bfloat16 as the data type, 4 heads, and 64 embedding dim.

[No Paged Attention] kv cache requires
    2 * (2 * b * h * kv_len * d)
bytes, where the first 2 is for kv cache for query and key, the second 2 is for bfloat16,
b is batch size, h is number of heads, kv_len is kv cache length, and d is embedding dim.
Taking the context length of 131072 from llama-3.1, the max batch size to serve is 32.

[Paged Attention] kv cache requires
    2 * (2 * h * n_pages * page_size * d)
bytes. Assuming a page size of 128, there could be at most 32768 pages.
We empirically observe that the max batch size to serve is 2448, which is 76x of the
max batch size without paged attention.
"""

import torch
from torch.nn.attention.flex_attention import (
    _identity,
    BlockMask,
    create_block_mask,
)
from datasets import load_dataset
import random
from collections import deque
from typing import Tuple
from utils import gen_offset, slice_block_mask
from model import PagedAttentionLayer
from paged_attention import PagedAttention

create_block_mask = torch.compile(create_block_mask)


class Requests:
    def __init__(self):
        self.data = load_dataset("Open-Orca/OpenOrca")["train"]

    def sample_request(self):
        # sample a prompt len and response len from openorca dataset
        # to simulate a real world use case
        idx = random.randint(0, len(self.data) - 1)
        prompt_len = len(self.data[idx]["system_prompt"]) + len(self.data[idx]["question"])
        response_len = len(self.data[idx]["response"])
        return prompt_len, response_len


class Server:
    def __init__(self, batch_size: int, n_pages: int, page_size: int, n_heads: int, head_dim: int):
        self.paged_attention = PagedAttention(n_pages, page_size, batch_size)

        self.model = torch.compile(
            PagedAttentionLayer(n_heads, head_dim, torch.bfloat16, self.paged_attention)
        )

        self.batch_size = batch_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.bsz_watermark = 0  # max batch size served during benchmark

        self.available_batch_idx = list(range(batch_size))[::-1]
        self.request_queue = deque([])
        self.batch_idx = []
        self.input_pos = torch.zeros(batch_size, device="cuda", dtype=torch.int64)
        self.request_length = torch.tensor(
            [float("inf")] * batch_size, device="cuda"
        )  # decide whether a request is completed

        self.token_embedding = torch.randn(
            (batch_size, 1, n_heads * head_dim), device="cuda", dtype=torch.bfloat16
        )  # [B, 1, n_heads*head_dim]

        self.block_mask = create_block_mask(
            lambda b, h, q, kv: q >= kv, batch_size, 1, 64 * 1024, 64 * 1024, BLOCK_SIZE=page_size
        )

    def receive_request(self, prompt_len: int, response_len: int):
        # assume we know prompt length and response length in advance.
        self.request_queue.append((prompt_len, response_len))

    def can_schedule(self, request: Tuple[int, int]) -> bool:
        return len(self.paged_attention.empty_pages) * self.paged_attention.page_size >= sum(
            request
        )

    def prefill_one_token(self, batch_idx: int, prompt_len: int, response_len: int):
        # allocate page table
        # in practice we don't know response length in advance. A good way is to use a heuristic to estimate response length
        # and allocate page table accordingly. We may also allocate pages on the fly. For simplicity, we assume we know
        # response length in advance.
        self.paged_attention.reserve(
            torch.tensor(batch_idx, device="cuda"),
            torch.tensor(prompt_len + response_len, device="cuda"),
        )

        # simulate input token embedding
        token_embedding = torch.randn(
            1, prompt_len, self.head_dim * self.n_heads, device="cuda", dtype=torch.bfloat16
        )

        # generate block mask. The same block mask is used for all layers.
        new_block_mask = slice_block_mask(self.block_mask, batch_idx, prompt_len, prompt_len)
        converted_block_mask = self.paged_attention.convert_logical_block_mask(
            new_block_mask, torch.tensor([batch_idx], device="cuda")
        )
        converted_score_mod = self.paged_attention.get_score_mod(
            _identity, 
            torch.tensor([batch_idx], device="cuda")
        )

        prefill_input_pos = torch.arange(prompt_len, device="cuda").view(1, -1)
        token_embedding = self.model(
            torch.tensor([batch_idx], device="cuda"),
            prefill_input_pos,
            token_embedding,
            converted_block_mask,
            converted_score_mod,
        )
        return token_embedding

    def prefill(self):
        while (
            self.request_queue
            and self.can_schedule(self.request_queue[0])
            and self.available_batch_idx
        ):
            prompt_len, response_len = self.request_queue.popleft()
            print(
                f"serving a new request with prompt_len: {prompt_len}, response_len: {response_len}"
            )
            new_batch_idx = self.available_batch_idx.pop()
            token_embedding = self.prefill_one_token(new_batch_idx, prompt_len, response_len)
            self.token_embedding[new_batch_idx] = token_embedding[:, -1].view(1, -1)

            self.batch_idx.append(new_batch_idx)
            self.input_pos[new_batch_idx] = prompt_len
            self.request_length[new_batch_idx] = prompt_len + response_len

        self.bsz_watermark = max(self.bsz_watermark, len(self.batch_idx))

    def get_decode_mask(self, batch_idx: torch.Tensor, input_pos: torch.Tensor):
        # batch_idx: [B], input_pos: [B]
        (B,) = batch_idx.shape
        input_block_idx = input_pos // self.block_mask.BLOCK_SIZE[0]  # [B]
        kv_num_blocks = self.block_mask.kv_num_blocks[batch_idx, :, input_block_idx].view(B, 1, 1)
        kv_indices = self.block_mask.kv_indices[batch_idx, :, input_block_idx].view(B, 1, 1, -1)
        full_kv_num_blocks, full_kv_indices = None, None
        if self.block_mask.full_kv_num_blocks is not None:
            full_kv_num_blocks = self.block_mask.full_kv_num_blocks[
                batch_idx, :, input_block_idx
            ].view(B, 1, 1)
            full_kv_indices = self.block_mask.full_kv_indices[batch_idx, :, input_block_idx].view(
                B, 1, 1, -1
            )
        seq_length = (1, self.block_mask.seq_lengths[1])
        return BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            BLOCK_SIZE=self.block_mask.BLOCK_SIZE,
            mask_mod=gen_offset(input_pos),
            seq_lengths=seq_length,
        )

    def decode(self):
        B = len(self.batch_idx)
        batch_idx = torch.tensor(self.batch_idx, device="cuda").view(-1)  # [B].
        input_pos = self.input_pos[batch_idx]  # [B]
        mask = self.get_decode_mask(batch_idx, input_pos)
        converted_block_mask = self.paged_attention.convert_logical_block_mask(mask, batch_idx)
        converted_score_mod = self.paged_attention.get_score_mod(_identity, batch_idx=batch_idx)
        self.token_embedding[batch_idx] = self.model(
            batch_idx,
            input_pos.view(B, 1),
            self.token_embedding[batch_idx],
            converted_block_mask,
            converted_score_mod,
        )
        self.input_pos[batch_idx] += 1

    def clean(self):
        completed_batch_indices = torch.where(self.input_pos >= self.request_length)[0]
        self.available_batch_idx += completed_batch_indices.tolist()
        self.batch_idx = [
            idx for idx in self.batch_idx if idx not in completed_batch_indices.tolist()
        ]

        for b in completed_batch_indices:
            self.paged_attention.erase(torch.tensor([b]))

        self.request_length[completed_batch_indices] = float("inf")


if __name__ == "__main__":
    # serving loop
    num_requests = 10  # total number of requests during benchmark
    gap = 3  # get a new request after `gap` number of decoding tokens

    batch_size, n_pages, page_size, n_heads, head_dim = 4096, 32768, 128, 4, 64

    requests = Requests()
    server = Server(batch_size, n_pages, page_size, n_heads, head_dim)

    with torch.no_grad():
        for i in range(num_requests):
            for _ in range(1024):
                server.receive_request(*requests.sample_request())

            server.prefill()
            for _ in range(gap):
                server.decode()

            server.clean()

    print("max batch size served: ", server.bsz_watermark)
