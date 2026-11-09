"""Shared operands, an independent per-query oracle, and numerical checks for gather attention."""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils._pytree import tree_flatten, tree_map

from attn_gym.sparse import RopePair
from attn_gym.sparse.gather_attn.api import AttentionInput
from attn_gym.testing.kda import assert_relative_rms_within


@dataclass(frozen=True)
class AttentionInputs:
    query: AttentionInput
    local_kv: AttentionInput | None
    sparse_kv: AttentionInput
    kv_indices: torch.Tensor
    attention_sink: torch.Tensor | None
    cu_seqlens: torch.Tensor | None
    cu_seqlens_k: torch.Tensor | None

    def kwargs(self):
        return vars(self).copy()

    def differentiable(self) -> tuple[torch.Tensor, ...]:
        return tensor_leaves((self.query, self.local_kv, self.sparse_kv, self.attention_sink))


def tensor_leaves(value) -> tuple[torch.Tensor, ...]:
    leaves, _ = tree_flatten(value)
    return tuple(x for x in leaves if isinstance(x, torch.Tensor) and x.requires_grad)


def latent(value: AttentionInput) -> torch.Tensor:
    return value if isinstance(value, torch.Tensor) else value[0]


def as_fp64(inputs: AttentionInputs) -> AttentionInputs:
    def convert(x):
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            return x
        return x.detach().double().requires_grad_(x.requires_grad)

    return AttentionInputs(**tree_map(convert, inputs.kwargs()))


def skip_unsupported(backend: str) -> None:
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if backend == "cute":
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            pytest.skip("SM100 or SM103 required")
        pytest.importorskip("flash_attn.cute.interface")
    elif torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("Blackwell required for paired Triton")


def make_inputs(
    *,
    heads: int = 64,
    latent_dim: int = 512,
    rope_dim: int = 64,
    seq_len: int = 33,
    topk: int = 16,
    with_docs: bool = True,
    with_sink: bool = True,
    batch: int = 1,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    local: bool = True,
    plain_tuple: bool = False,
) -> AttentionInputs:
    g = torch.Generator(device=device).manual_seed(2027)

    def rand(*shape):
        return torch.randn(*shape, device=device, dtype=dtype, generator=g).requires_grad_()

    def parts(h, tokens):
        x = rand(batch, h, tokens, latent_dim)
        if not rope_dim:
            return x
        r = rand(batch, h, tokens, rope_dim)
        return (x, r) if plain_tuple else RopePair(x, r)

    query = parts(heads, seq_len)
    local_kv = parts(1, seq_len) if local else None
    sparse_kv = parts(1, seq_len)
    starts = list(range(0, seq_len, max(seq_len // 3, 1))) if with_docs else [0]
    positions = torch.arange(seq_len, device=device)
    local_positions = positions.clone()
    if with_docs:
        for start, end in zip(starts, [*starts[1:], seq_len]):
            local_positions[start:end] -= start
    indices = torch.randint(seq_len, (batch, seq_len, topk), device=device, generator=g)
    indices %= local_positions[None, :, None] + 1
    if topk:
        if topk > 1:
            indices[..., 1] = indices[..., 0]
        indices[:, ::3, -1] = -1
        indices[:, 0] = -1
    cu_q = cu_k = None
    if with_docs:
        assert batch == 1
        cu_q = torch.tensor([*starts, seq_len], device=device, dtype=torch.int32)
        cu_k = cu_q.clone()
    sink = torch.randn(heads, device=device, generator=g).requires_grad_() if with_sink else None
    return AttentionInputs(query, local_kv, sparse_kv, indices.to(torch.int32), sink, cu_q, cu_k)


@torch.no_grad()
def assert_numerics(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    assert torch.isfinite(actual).all(), name
    if actual.numel() == 0:
        return
    assert_relative_rms_within(actual, expected, name, max_eps=2)
    # Pair the aggregate bound with an outlier check in source-dtype epsilon units.
    budget = 8 * torch.finfo(torch.bfloat16).eps * expected.abs().max()
    assert (actual.double() - expected).abs().max() <= budget, name


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
) -> dict[str, Tensor]:
    """Generate public gather_attn operands with -1 padding and repeated selections.

    Indices are drawn from [-1, sparse_tokens), so with packed offsets some of them fall
    outside the query's document and exercise the ignored-index contract.
    """
    generator = torch.Generator(device=device).manual_seed(42)

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

    indices = kv_indices.tolist()
    if cu_seqlens is not None:
        offsets, offsets_k = cu_seqlens.tolist(), cu_seqlens_k.tolist()
    outputs, lses = [], []
    for b in range(batch):
        batch_outputs, batch_lses = [], []
        document = 0
        for t in range(tokens):
            start, sparse_start, sparse_end = 0, 0, sparse_kv.shape[2]
            if cu_seqlens is not None:
                while offsets[document + 1] <= t:
                    document += 1
                start = offsets[document]
                sparse_start, sparse_end = offsets_k[document], offsets_k[document + 1]
            window_start = max(t - sliding_window_size + 1, start)
            sparse_positions = [
                sparse_start + index
                for index in indices[b][t]
                if 0 <= index < sparse_end - sparse_start
            ]
            keys = torch.cat(
                (local_kv[b, :, window_start : t + 1], sparse_kv[b, :, sparse_positions]), dim=1
            )
            values = (
                keys
                if local_v is local_kv and sparse_v is sparse_kv
                else torch.cat(
                    (local_v[b, :, window_start : t + 1], sparse_v[b, :, sparse_positions]), dim=1
                )
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
