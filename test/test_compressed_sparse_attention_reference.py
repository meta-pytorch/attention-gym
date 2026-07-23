import pytest
import torch
import torch.nn.functional as F

from attn_gym.sparse import compressed_sparse_attention


_BATCHED_ARGUMENTS = frozenset({0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13})


def _make_inputs(share_kv: bool, dtype: torch.dtype):
    batch = 3
    heads = 3
    sequence = 17
    head_dim = 32
    index_heads = 2
    index_dim = 16
    compression_rate = 4
    generator = torch.Generator(device="cuda").manual_seed(123)

    def randn(*shape: int, scale: float = 0.2) -> torch.Tensor:
        return torch.randn(*shape, device="cuda", dtype=dtype, generator=generator) * scale

    def query(*shape: int) -> torch.Tensor:
        return F.normalize(randn(*shape), dim=-1)

    kv_heads = 1 if share_kv else heads
    index_kv_heads = 1 if share_kv else index_heads
    return (
        query(batch, heads, sequence, head_dim),
        query(batch, index_heads, sequence, index_dim),
        randn(batch, kv_heads, sequence, head_dim),
        randn(batch, kv_heads, sequence, head_dim),
        randn(batch, kv_heads, sequence, head_dim),
        randn(batch, kv_heads, sequence, head_dim),
        randn(batch, kv_heads, sequence, head_dim),
        randn(compression_rate, head_dim),
        randn(compression_rate, head_dim),
        randn(batch, sequence, index_heads),
        randn(batch, index_kv_heads, sequence, index_dim),
        randn(batch, index_kv_heads, sequence, index_dim),
        randn(batch, index_kv_heads, sequence, index_dim),
        randn(batch, index_kv_heads, sequence, index_dim),
        randn(compression_rate, index_dim),
        randn(compression_rate, index_dim),
        1.0 + randn(head_dim, scale=0.05),
        1.0 + randn(index_dim, scale=0.05),
        1.0 + randn(head_dim, scale=0.05),
        randn(heads),
        compression_rate,
        3,
        5,
        8,
        share_kv,
    )


def _select_batch(inputs, batch_index: int):
    return tuple(
        value[batch_index : batch_index + 1] if index in _BATCHED_ARGUMENTS else value
        for index, value in enumerate(inputs)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("share_kv", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_reference_is_batch_invariant(share_kv, dtype):
    inputs = _make_inputs(share_kv, dtype)

    with torch.inference_mode():
        batched = compressed_sparse_attention(*inputs, backend="eager")
        independent = torch.cat(
            [
                compressed_sparse_attention(*_select_batch(inputs, batch_index), backend="eager")
                for batch_index in range(inputs[0].shape[0])
            ]
        )

    batched_bytes = batched.contiguous().view(torch.uint8)
    independent_bytes = independent.contiguous().view(torch.uint8)
    differing_bytes = torch.count_nonzero(batched_bytes != independent_bytes).item()
    assert differing_bytes == 0, f"outputs differ in {differing_bytes} bytes"
