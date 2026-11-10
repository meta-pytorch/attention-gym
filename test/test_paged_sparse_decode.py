"""Reference semantics and metadata validation for paged sparse decode."""

import pytest
import torch

from attn_gym.sparse import AuxRequest, PagedKV, paged_sparse_decode
from attn_gym.testing.paged_decode import make_paged_sparse_decode_inputs


def test_reference_matches_explicit_key_sets():
    """Shared source: blocks before the aligned window, the window itself, and the sink."""
    torch.manual_seed(0)
    page_size, block, heads, dim = 2, 2, 2, 4
    keys = torch.randn(6, page_size, 1, dim, dtype=torch.float64)
    values = torch.randn(6, page_size, 1, dim, dtype=torch.float64)
    block_table = torch.tensor([[4, 1, 5, 0, 3]], dtype=torch.int32)
    query = torch.randn(1, heads, dim, dtype=torch.float64)
    sink = torch.tensor([0.3, -1.0], dtype=torch.float64)
    # Position 8 with window 2: the window starts at 7, rounded down to 6, so it holds 6..8.
    # Block 3 (entries 6, 7) overlaps the window and is skipped; block 0 is selected twice.
    indices = torch.tensor([[1, 3, 0, -1, 0]], dtype=torch.int32)
    out, aux = paged_sparse_decode(
        query,
        PagedKV(keys, values, block_table),
        indices,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([9], dtype=torch.int32),
        max_seqlen_q=1,
        index_block_size=block,
        window_size=2,
        attention_sink=sink,
        scale=0.5,
        impl="reference",
        return_aux=AuxRequest(lse=True),
    )
    positions = torch.tensor([2, 3, 0, 1, 0, 1, 6, 7, 8])
    pages = block_table[0, positions // page_size].long()
    key_rows = keys[pages, positions % page_size, 0]
    value_rows = values[pages, positions % page_size, 0]
    logits = torch.cat((query[0] @ key_rows.T * 0.5, sink[:, None]), dim=1)
    expected_lse = logits.logsumexp(dim=1)
    expected = torch.softmax(logits, dim=1)[:, :-1] @ value_rows
    torch.testing.assert_close(out[0], expected)
    torch.testing.assert_close(aux.lse[0], expected_lse.float())


def test_reference_separate_window_source():
    """A separate window source keeps every selected entry and reads its own pages."""
    torch.manual_seed(1)
    sparse = PagedKV(
        *(torch.randn(3, 4, 1, 8, dtype=torch.float64) for _ in range(2)),
        torch.tensor([[2, 0]], dtype=torch.int32),
    )
    window = PagedKV(
        *(torch.randn(5, 2, 1, 8, dtype=torch.float64) for _ in range(2)),
        torch.tensor([[4, 3, 1, 0]], dtype=torch.int32),
    )
    query = torch.randn(1, 2, 8, dtype=torch.float64)
    out = paged_sparse_decode(
        query,
        sparse,
        torch.tensor([[5, 1]], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([7], dtype=torch.int32),
        max_seqlen_q=1,
        window_size=3,
        window_kv=window,
        impl="reference",
    )
    sparse_rows = [(sparse.block_table[0, e // 4], e % 4) for e in (5, 1)]
    window_rows = [(window.block_table[0, p // 2], p % 2) for p in (4, 5, 6)]
    keys = torch.stack(
        [sparse.key_cache[p, s, 0] for p, s in sparse_rows]
        + [window.key_cache[p, s, 0] for p, s in window_rows]
    )
    values = torch.stack(
        [sparse.value_cache[p, s, 0] for p, s in sparse_rows]
        + [window.value_cache[p, s, 0] for p, s in window_rows]
    )
    expected = torch.softmax(query[0] @ keys.T / 8**0.5, dim=1) @ values
    torch.testing.assert_close(out[0], expected)


def _valid_inputs():
    return make_paged_sparse_decode_inputs([40, 9], [1, 2], device="cpu", dtype=torch.float32)


@pytest.mark.parametrize(
    ("update", "error", "match"),
    [
        (lambda i: {"indices": i["indices"].long()}, TypeError, "indices must be int32"),
        (lambda i: {"query": i["query"][..., :-1]}, ValueError, "head_dim must match"),
        (lambda i: {"cu_seqlens_q": i["cu_seqlens_q"][:-1]}, ValueError, "cu_seqlens_q"),
        (lambda i: {"index_block_size": 0}, ValueError, "index_block_size"),
        (lambda i: {"window_size": -1}, ValueError, "window_size"),
        (lambda i: {"attention_sink": torch.zeros(3)}, ValueError, "attention_sink"),
        (lambda i: {"topk_length": i["topk_length"][:1]}, ValueError, "topk_length"),
        (lambda i: {"query": i["query"].requires_grad_()}, ValueError, "forward-only"),
        (
            lambda i: {"attention_sink": torch.zeros(i["query"].shape[1], requires_grad=True)},
            ValueError,
            "forward-only",
        ),
        (
            lambda i: {
                "kv": PagedKV(i["kv"].key_cache, i["kv"].value_cache, i["kv"].block_table[:1])
            },
            ValueError,
            "block_table",
        ),
    ],
)
def test_rejects_invalid_inputs(update, error, match):
    inputs = _valid_inputs()
    with pytest.raises(error, match=match):
        paged_sparse_decode(**{**inputs, **update(inputs)}, impl="reference")
