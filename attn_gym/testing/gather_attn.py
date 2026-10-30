"""Shared operands and an independent per-query oracle for gather attention."""

import math

import torch
from torch import Tensor


def make_gather_attn_test_inputs(
    *,
    heads: int,
    kv_heads: int,
    head_dim: int,
    tokens: int,
    sparse_tokens: int,
    topk: int,
    separate_values: bool,
    dtype: torch.dtype,
    batch: int = 1,
    device: str | torch.device = "cuda",
    seed: int = 42,
) -> dict[str, Tensor]:
    """Generate public gather_attn operands with -1 padding and repeated selections.

    Indices are drawn from [-1, sparse_tokens), so with packed offsets some of them fall
    outside the query's document and exercise the ignored-index contract.
    """
    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape: int) -> Tensor:
        return torch.randn(*shape, device=device, dtype=dtype, generator=generator)

    inputs = {
        "query": randn(batch, heads, tokens, head_dim),
        "local_kv": randn(batch, kv_heads, tokens, head_dim),
        "sparse_kv": randn(batch, kv_heads, sparse_tokens, head_dim),
        "kv_indices": torch.randint(
            -1, sparse_tokens, (batch, tokens, topk), device=device, generator=generator
        ),
        "attention_sink": randn(heads),
    }
    if topk > 1:
        inputs["kv_indices"][:, ::3, 1] = inputs["kv_indices"][:, ::3, 0]
    if separate_values:
        inputs["local_v"] = randn(batch, kv_heads, tokens, head_dim)
        inputs["sparse_v"] = randn(batch, kv_heads, sparse_tokens, head_dim)
    return inputs


def make_qsa_block_indices(
    document_lengths: list[int],
    *,
    block_size: int,
    budget_blocks: int,
    generator: torch.Generator,
    device: str | torch.device = "cuda",
) -> Tensor:
    """Build document-local token indices the way QSA selects them.

    Each query keeps up to ``budget_blocks`` of its complete ``block_size``-token blocks,
    expanded to token indices, plus the partial tail block ending at the query; -1 pads
    the rest. Use with a raw-token sparse pool and ``sliding_window_size=0``.
    """
    width = budget_blocks * block_size + block_size - 1
    indices = torch.full((1, sum(document_lengths), width), -1, dtype=torch.int64)
    position = 0
    for length in document_lengths:
        for local in range(length):
            visible = local + 1
            complete = visible // block_size
            chosen = torch.randperm(complete, generator=generator)[:budget_blocks].tolist()
            selected = [
                block * block_size + offset for block in chosen for offset in range(block_size)
            ]
            selected += list(range(complete * block_size, visible))
            indices[0, position, : len(selected)] = torch.tensor(selected)
            position += 1
    return indices.to(device)


def gather_attn_oracle(
    query: Tensor,
    local_kv: Tensor,
    sparse_kv: Tensor,
    kv_indices: Tensor,
    attention_sink: Tensor | None,
    *,
    sliding_window_size: int,
    scale: float,
    cu_seqlens: Tensor | None = None,
    cu_seqlens_k: Tensor | None = None,
    local_v: Tensor | None = None,
    sparse_v: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Differentiable gather attention built one query position at a time.

    Each query's key list is spelled out explicitly: its causal window clipped to the
    document start, then every valid selected sparse index (repeats included). Query head
    h reads KV head h // (heads // kv_heads), and values default to the keys. Computes in
    the input dtype; pass FP64 operands for a tight oracle. Rows past the final packed
    offset are outside the gather_attn contract and are not supported here.
    """
    batch, heads, tokens, head_dim = query.shape
    kv_heads = local_kv.shape[1]
    group = heads // kv_heads
    local_v = local_kv if local_v is None else local_v
    sparse_v = sparse_kv if sparse_v is None else sparse_v
    sink = None if attention_sink is None else attention_sink.to(query.dtype)

    outputs, lses = [], []
    for b in range(batch):
        batch_outputs, batch_lses = [], []
        for t in range(tokens):
            start, sparse_start, sparse_end = 0, 0, sparse_kv.shape[2]
            if cu_seqlens is not None:
                document = int((cu_seqlens[1:] <= t).sum())
                start = int(cu_seqlens[document])
                sparse_start = int(cu_seqlens_k[document])
                sparse_end = int(cu_seqlens_k[document + 1])
            window_start = max(t - sliding_window_size + 1, start)
            local_positions = list(range(window_start, t + 1)) if sliding_window_size else []
            sparse_positions = [
                sparse_start + index
                for index in kv_indices[b, t].tolist()
                if 0 <= index < sparse_end - sparse_start
            ]
            keys = torch.cat(
                (local_kv[b][:, local_positions], sparse_kv[b][:, sparse_positions]), dim=1
            )
            values = torch.cat(
                (local_v[b][:, local_positions], sparse_v[b][:, sparse_positions]), dim=1
            )
            grouped_query = query[b, :, t].reshape(kv_heads, group, head_dim)
            logits = torch.einsum("kgd,knd->kgn", grouped_query, keys) * scale
            logits = logits.reshape(heads, -1)
            normalized = logits if sink is None else torch.cat((logits, sink[:, None]), dim=1)
            lse = torch.logsumexp(normalized, dim=1)
            probabilities = torch.exp(logits - lse[:, None]).reshape(kv_heads, group, -1)
            output = torch.einsum("kgn,knd->kgd", probabilities, values)
            batch_outputs.append(output.reshape(heads, head_dim))
            batch_lses.append(lse)
        outputs.append(torch.stack(batch_outputs, dim=1))
        lses.append(torch.stack(batch_lses, dim=1))
    return torch.stack(outputs), torch.stack(lses)


def assert_matches_low_precision_eager(
    actual,
    low_precision_expected,
    high_precision_expected,
    reduction_sizes,
):
    """Bound kernel error by low-precision eager error against an FP64 measuring stick."""
    assert torch.isfinite(actual).all()
    actual_difference = (actual.double() - high_precision_expected).abs()
    eager_difference = (low_precision_expected.double() - high_precision_expected).abs()
    accumulation_eps = (
        sum(math.sqrt(size) for size in reduction_sizes) * torch.finfo(torch.float32).eps
    )
    output_rounding_eps = torch.finfo(actual.dtype).eps
    rounding_eps = accumulation_eps + output_rounding_eps
    mean_atol = rounding_eps * high_precision_expected.abs().mean().item()
    max_atol = rounding_eps * high_precision_expected.abs().max().item()
    assert actual_difference.mean().item() <= eager_difference.mean().item() + mean_atol
    assert actual_difference.max().item() <= eager_difference.max().item() + max_atol
