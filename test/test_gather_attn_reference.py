import pytest
import torch

from attn_gym.sparse.gather_attn import Impl, gather_attn


@pytest.mark.parametrize("share_kv", [False, True])
@pytest.mark.parametrize("num_topk_blocks", [0, 1, 3])
def test_batch_invariance(share_kv, num_topk_blocks):
    """Outputs should be identical regardless of batch position."""
    b, h, s, head_dim = 3, 5, 14, 17
    kv_heads = 1 if share_kv else h
    sparse_seq_len = s // 2

    generator = torch.Generator().manual_seed(42)

    query = torch.randn(b, h, s, head_dim, generator=generator)
    local_kv = torch.randn(b, kv_heads, s, head_dim, generator=generator)
    sparse_kv = torch.randn(b, kv_heads, sparse_seq_len, head_dim, generator=generator)

    if num_topk_blocks > 0:
        _, kv_indices = torch.topk(
            torch.randn(b, s, sparse_seq_len, generator=generator),
            k=min(num_topk_blocks, sparse_seq_len),
            dim=-1,
        )
    else:
        kv_indices = torch.zeros(b, s, 0, dtype=torch.long)

    attention_sink = torch.randn(h, generator=generator)
    sliding_window_size = 3

    # Run full batch
    full_out = gather_attn(
        query,
        local_kv,
        sparse_kv,
        kv_indices,
        attention_sink,
        sliding_window_size=sliding_window_size,
        impl=Impl.REFERENCE,
    )

    # Run each batch element independently and compare
    for i in range(b):
        single_out = gather_attn(
            query[i : i + 1],
            local_kv[i : i + 1],
            sparse_kv[i : i + 1],
            kv_indices[i : i + 1],
            attention_sink,
            sliding_window_size=sliding_window_size,
            impl=Impl.REFERENCE,
        )
        torch.testing.assert_close(
            full_out[i : i + 1],
            single_out,
            atol=1e-5,
            rtol=1e-5,
            msg=f"Batch element {i} differs when run independently vs. in a batch",
        )
