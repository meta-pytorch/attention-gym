"""Shared operands and numerical checks for bare and paired gather attention."""

from dataclasses import dataclass

import torch
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
