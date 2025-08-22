import torch
import math
from torch.nn.attention.flex_attention import BlockMask, flex_attention, _score_mod_signature
from torch import Tensor
from typing import Dict, Optional


class NonPagedAttentionLayer(torch.nn.Module):
    """An attention layer without paged attention, ported from GPT-Fast:
    https://github.com/meta-pytorch/gpt-fast/blob/main/model.py#L180-L227
    """

    def __init__(self, bsz, n_heads, max_seq_len, head_dim, dtype, block_size: int = 32768):
        super().__init__()
        self.n_head = n_heads
        self.head_dim = head_dim

        # key, query, value projections for all heads, but in a batch
        total_head_dim = n_heads * head_dim
        self.wqkv = torch.nn.Linear(
            total_head_dim, 3 * total_head_dim, bias=False, device="cuda", dtype=dtype
        )
        self.wo = torch.nn.Linear(
            total_head_dim, total_head_dim, bias=False, device="cuda", dtype=dtype
        )
        self.k_cache = torch.randn(
            (bsz, n_heads, max_seq_len, head_dim), device="cuda", dtype=dtype
        )
        self.v_cache = torch.randn(
            (bsz, n_heads, max_seq_len, head_dim), device="cuda", dtype=dtype
        )
        self.freqs_cis = precompute_freqs_cis(block_size, self.head_dim, dtype=dtype)

    def forward(
        self,
        batch_idx: Tensor,
        input_pos: Tensor,
        x: Tensor,
        block_mask: BlockMask,
        score_mod: _score_mod_signature,
    ) -> Tensor:
        # input_pos: [B, S], batch_idx: [B], x: [B, S, D]
        B, S, _ = x.shape

        kv_size = self.n_head * self.head_dim
        q, k, v = self.wqkv(x).split([kv_size, kv_size, kv_size], dim=-1)

        q = q.view(B, S, self.n_head, self.head_dim)
        k = k.view(B, S, self.n_head, self.head_dim)
        v = v.view(B, S, self.n_head, self.head_dim)

        freqs_cis = self.freqs_cis.unsqueeze(0)[
            torch.zeros((B, 1), dtype=torch.int), input_pos
        ]  # [B, S, D//2, 2]

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q = q.transpose(1, 2)
        self.k_cache[batch_idx.view(B, 1), :, input_pos] = k
        self.v_cache[batch_idx.view(B, 1), :, input_pos] = v

        y = flex_attention(
            q, self.k_cache, self.v_cache, block_mask=block_mask, score_mod=score_mod
        )

        y = y.transpose(1, 2).contiguous().view(B, S, -1)

        y = self.wo(y)
        return y


class PagedAttentionLayer(torch.nn.Module):
    """An attention layer with paged attention"""

    def __init__(self, n_heads, head_dim, dtype, paged_attention, block_size: int = 65536):
        super().__init__()
        self.n_head = n_heads
        self.head_dim = head_dim

        # key, query, value projections for all heads, but in a batch
        total_head_dim = n_heads * head_dim
        self.wqkv = torch.nn.Linear(
            total_head_dim, 3 * total_head_dim, bias=False, device="cuda", dtype=dtype
        )
        self.wo = torch.nn.Linear(
            total_head_dim, total_head_dim, bias=False, device="cuda", dtype=dtype
        )

        # allocate kv cache with batch size=1 for paged attention
        max_cached_seq_len = paged_attention.n_pages * paged_attention.page_size
        self.k_cache_paged = torch.randn(
            1,
            n_heads,
            max_cached_seq_len,
            head_dim,
            device="cuda",
            dtype=dtype,
        )
        self.v_cache_paged = torch.randn(
            1,
            n_heads,
            max_cached_seq_len,
            head_dim,
            device="cuda",
            dtype=dtype,
        )
        self.paged_attention = paged_attention

        self.freqs_cis = precompute_freqs_cis(
            block_size, self.head_dim, dtype=dtype
        )  # [block_size, D//2, 2]

    def forward(
        self,
        batch_idx: Tensor,
        input_pos: Tensor,
        x: Tensor,
        converted_block_mask: BlockMask,
        converted_score_mod: _score_mod_signature,
    ) -> Tensor:
        # input_pos: [B, S], batch_idx: [B], x: [B, S, D]
        B, S, _ = x.shape
        kv_size = self.n_head * self.head_dim
        q, k, v = self.wqkv(x).split([kv_size, kv_size, kv_size], dim=-1)

        q = q.view(B, S, self.n_head, self.head_dim)
        k = k.view(B, S, self.n_head, self.head_dim)
        v = v.view(B, S, self.n_head, self.head_dim)

        freqs_cis = self.freqs_cis.unsqueeze(0)[
            torch.zeros((B, 1), dtype=torch.int), input_pos
        ]  # [B, S, D//2, 2]

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        # Comparing with NonPagedAttention, here is the only change for updating kv cache
        self.paged_attention.assign(
            batch_idx, input_pos, k, v, self.k_cache_paged, self.v_cache_paged
        )

        y = flex_attention(
            q,
            self.k_cache_paged,
            self.v_cache_paged,
            block_mask=converted_block_mask,
            score_mod=converted_score_mod,
        )

        y = y.transpose(1, 2).contiguous().view(B, S, -1)

        y = self.wo(y)
        return y


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    # x: [B, S, H, D], freqs_cis: [B, S, D//2, 2]
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)  # [B, S, H, D//2, 2]
    freqs_cis = freqs_cis.view(
        xshaped.size(0), xshaped.size(1), 1, xshaped.size(3), 2
    )  # [B, S, 1, D//2, 2]
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


def apply_rope_scaling(freqs: torch.Tensor, rope_scaling: Dict):
    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    seq_len: int,
    n_elem: int,
    base: int = 10000,
    dtype: torch.dtype = torch.bfloat16,
    rope_scaling: Optional[dict] = None,
) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    if rope_scaling is not None:
        freqs = apply_rope_scaling(freqs, rope_scaling)
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype, device="cuda")
