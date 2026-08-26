import pytest
import torch

from attn_gym.sparse.selected_attention import selected_attention


def _run_selected_attention(
    query, local_kv, sparse_kv, kv_indices, attention_sink, doc_ids, sliding_window_size
):
    """Helper that calls selected_attention and returns the output."""
    output, _ = selected_attention(
        query,
        local_kv,
        sparse_kv,
        kv_indices,
        attention_sink,
        doc_ids,
        sliding_window_size,
        backend="eager",
    )
    return output


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
    full_out = _run_selected_attention(
        query, local_kv, sparse_kv, kv_indices, attention_sink, None, sliding_window_size
    )

    # Run each batch element independently and compare
    for i in range(b):
        single_out = _run_selected_attention(
            query[i : i + 1],
            local_kv[i : i + 1],
            sparse_kv[i : i + 1],
            kv_indices[i : i + 1],
            attention_sink,
            None,
            sliding_window_size,
        )
        torch.testing.assert_close(
            full_out[i : i + 1],
            single_out,
            atol=1e-5,
            rtol=1e-5,
            msg=f"Batch element {i} differs when run independently vs. in a batch",
        )


@pytest.mark.parametrize("share_kv", [False, True])
@pytest.mark.parametrize("num_topk_blocks", [0, 1, 3])
def test_doc_id_isolation(share_kv, num_topk_blocks):
    """Tokens in different documents must not attend to each other.

    Strategy: pack two documents into one sequence, run selected_attention with
    doc_ids, then compare against running each document independently. The
    outputs for each document's tokens should match regardless of what occupies
    the other document's positions.
    """
    h, head_dim = 4, 16
    kv_heads = 1 if share_kv else h
    sliding_window_size = 3

    # Document lengths
    doc1_len = 6
    doc2_len = 8
    packed_len = doc1_len + doc2_len
    sparse_seq_len = packed_len // 2

    generator = torch.Generator().manual_seed(123)

    # ---------- Packed inputs (both docs in one sequence) ----------
    query_packed = torch.randn(1, h, packed_len, head_dim, generator=generator)
    local_kv_packed = torch.randn(1, kv_heads, packed_len, head_dim, generator=generator)
    sparse_kv_packed = torch.randn(1, kv_heads, sparse_seq_len, head_dim, generator=generator)

    if num_topk_blocks > 0:
        _, kv_indices_packed = torch.topk(
            torch.randn(1, packed_len, sparse_seq_len, generator=generator),
            k=min(num_topk_blocks, sparse_seq_len),
            dim=-1,
        )
    else:
        kv_indices_packed = torch.zeros(1, packed_len, 0, dtype=torch.long)

    attention_sink = torch.randn(h, generator=generator)

    # doc_ids: [0,0,0,0,0,0, 1,1,1,1,1,1,1,1]
    doc_ids = torch.cat(
        [
            torch.zeros(doc1_len, dtype=torch.long),
            torch.ones(doc2_len, dtype=torch.long),
        ]
    ).unsqueeze(0)  # shape (1, packed_len)

    out_packed = _run_selected_attention(
        query_packed,
        local_kv_packed,
        sparse_kv_packed,
        kv_indices_packed,
        attention_sink,
        doc_ids,
        sliding_window_size,
    )

    # ---------- Perturb the other document's data ----------
    # If isolation is correct, changing doc2's tokens shouldn't affect doc1's output
    query_perturbed = query_packed.clone()
    local_kv_perturbed = local_kv_packed.clone()
    # Overwrite doc2's positions with different random data
    query_perturbed[:, :, doc1_len:, :] = torch.randn(
        1, h, doc2_len, head_dim, generator=generator
    )
    local_kv_perturbed[:, :, doc1_len:, :] = torch.randn(
        1, kv_heads, doc2_len, head_dim, generator=generator
    )

    out_perturbed = _run_selected_attention(
        query_perturbed,
        local_kv_perturbed,
        sparse_kv_packed,
        kv_indices_packed,
        attention_sink,
        doc_ids,
        sliding_window_size,
    )

    # Doc1's output must be unchanged
    torch.testing.assert_close(
        out_packed[:, :, :doc1_len, :],
        out_perturbed[:, :, :doc1_len, :],
        atol=1e-5,
        rtol=1e-5,
        msg="Doc1 output changed when doc2 content was perturbed — doc isolation broken",
    )

    # Symmetrically, perturb doc1 and check doc2 is unchanged
    query_perturbed2 = query_packed.clone()
    local_kv_perturbed2 = local_kv_packed.clone()
    query_perturbed2[:, :, :doc1_len, :] = torch.randn(
        1, h, doc1_len, head_dim, generator=generator
    )
    local_kv_perturbed2[:, :, :doc1_len, :] = torch.randn(
        1, kv_heads, doc1_len, head_dim, generator=generator
    )

    out_perturbed2 = _run_selected_attention(
        query_perturbed2,
        local_kv_perturbed2,
        sparse_kv_packed,
        kv_indices_packed,
        attention_sink,
        doc_ids,
        sliding_window_size,
    )

    torch.testing.assert_close(
        out_packed[:, :, doc1_len:, :],
        out_perturbed2[:, :, doc1_len:, :],
        atol=1e-5,
        rtol=1e-5,
        msg="Doc2 output changed when doc1 content was perturbed — doc isolation broken",
    )


@pytest.mark.parametrize("share_kv", [False, True])
def test_doc_id_matches_separate_execution(share_kv):
    """Packed execution with doc_ids should match running each doc alone.

    This is a stronger test: we run each document as its own independent
    sequence (no packing), then pack them together with doc_ids, and verify
    the outputs match token-for-token.
    """
    h, head_dim = 4, 16
    kv_heads = 1 if share_kv else h
    sliding_window_size = 3
    num_topk_blocks = 2

    doc1_len = 5
    doc2_len = 7
    packed_len = doc1_len + doc2_len

    generator = torch.Generator().manual_seed(99)

    # Generate data for each document independently
    query1 = torch.randn(1, h, doc1_len, head_dim, generator=generator)
    local_kv1 = torch.randn(1, kv_heads, doc1_len, head_dim, generator=generator)
    sparse_kv1_len = doc1_len // 2
    sparse_kv1 = torch.randn(1, kv_heads, sparse_kv1_len, head_dim, generator=generator)
    _, kv_indices1 = torch.topk(
        torch.randn(1, doc1_len, sparse_kv1_len, generator=generator),
        k=min(num_topk_blocks, sparse_kv1_len),
        dim=-1,
    )

    query2 = torch.randn(1, h, doc2_len, head_dim, generator=generator)
    local_kv2 = torch.randn(1, kv_heads, doc2_len, head_dim, generator=generator)
    sparse_kv2_len = doc2_len // 2
    sparse_kv2 = torch.randn(1, kv_heads, sparse_kv2_len, head_dim, generator=generator)
    _, kv_indices2 = torch.topk(
        torch.randn(1, doc2_len, sparse_kv2_len, generator=generator),
        k=min(num_topk_blocks, sparse_kv2_len),
        dim=-1,
    )

    attention_sink = torch.randn(h, generator=generator)

    # Run each document independently (no doc_ids needed — single doc)
    out1 = _run_selected_attention(
        query1,
        local_kv1,
        sparse_kv1,
        kv_indices1,
        attention_sink,
        None,
        sliding_window_size,
    )
    out2 = _run_selected_attention(
        query2,
        local_kv2,
        sparse_kv2,
        kv_indices2,
        attention_sink,
        None,
        sliding_window_size,
    )

    # ---------- Pack into one sequence ----------
    query_packed = torch.cat([query1, query2], dim=2)
    local_kv_packed = torch.cat([local_kv1, local_kv2], dim=2)

    # For the sparse branch, pack both sparse KVs and adjust indices for doc2
    sparse_kv_packed = torch.cat([sparse_kv1, sparse_kv2], dim=2)

    # Build packed indices: doc1 indices stay the same, doc2 indices offset
    # by sparse_kv1_len.
    indices2_shifted = kv_indices2 + sparse_kv1_len

    # We need a common topk width — use the max
    topk_width = max(
        min(num_topk_blocks, sparse_kv1_len),
        min(num_topk_blocks, sparse_kv2_len),
    )
    kv_indices_packed = torch.zeros(1, packed_len, topk_width, dtype=torch.long)
    kv_indices_packed[:, :doc1_len, : min(num_topk_blocks, sparse_kv1_len)] = kv_indices1
    kv_indices_packed[:, doc1_len:, : min(num_topk_blocks, sparse_kv2_len)] = indices2_shifted

    doc_ids = torch.cat(
        [
            torch.zeros(doc1_len, dtype=torch.long),
            torch.ones(doc2_len, dtype=torch.long),
        ]
    ).unsqueeze(0)

    out_packed = _run_selected_attention(
        query_packed,
        local_kv_packed,
        sparse_kv_packed,
        kv_indices_packed,
        attention_sink,
        doc_ids,
        sliding_window_size,
    )

    # Verify doc1 tokens match
    torch.testing.assert_close(
        out_packed[:, :, :doc1_len, :],
        out1,
        atol=1e-5,
        rtol=1e-5,
        msg="Packed doc1 output doesn't match standalone doc1 output",
    )

    # Verify doc2 tokens match
    torch.testing.assert_close(
        out_packed[:, :, doc1_len:, :],
        out2,
        atol=1e-5,
        rtol=1e-5,
        msg="Packed doc2 output doesn't match standalone doc2 output",
    )
