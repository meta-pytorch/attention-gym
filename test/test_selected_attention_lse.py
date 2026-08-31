import math

import pytest
import torch

from attn_gym.sparse.selected_attention import AuxRequest, selected_attention

BACKENDS = ["eager"]
if torch.cuda.is_available():
    BACKENDS.append("triton")

pytestmark = pytest.mark.usefixtures("selected_attention_single_config")


def _dense_qk_softmax_oracle(
    query,
    local_kv,
    sparse_kv,
    kv_indices,
    attention_sink,
    doc_ids,
    sliding_window_size,
):
    batch, heads, sequence_length, head_dim = query.shape
    accumulation_dtype = torch.promote_types(query.dtype, torch.float32)
    query = query.to(accumulation_dtype)
    local_kv = local_kv.to(accumulation_dtype).expand(batch, heads, -1, -1)
    sparse_kv = sparse_kv.to(accumulation_dtype).expand(batch, heads, -1, -1)

    scale = 1.0 / math.sqrt(head_dim)
    sparse_logits = torch.matmul(query, sparse_kv.transpose(-2, -1)) * scale
    local_logits = torch.matmul(query, local_kv.transpose(-2, -1)) * scale

    query_positions = torch.arange(sequence_length, device=query.device)[:, None]
    key_positions = torch.arange(sequence_length, device=query.device)[None, :]
    local_is_valid = (key_positions <= query_positions) & (
        key_positions >= query_positions - sliding_window_size + 1
    )
    local_is_valid = local_is_valid[None, None]
    if doc_ids is not None:
        same_document = doc_ids[:, None, :, None] == doc_ids[:, None, None, :]
        local_is_valid = local_is_valid & same_document
    local_logits = local_logits.masked_fill(~local_is_valid, float("-inf"))

    selected_is_valid = kv_indices >= 0
    selected_indices = kv_indices.clamp_min(0)[:, None].expand(-1, heads, -1, -1)
    selected_logits = sparse_logits.gather(dim=-1, index=selected_indices)
    selected_logits = selected_logits.masked_fill(~selected_is_valid[:, None], float("-inf"))

    if attention_sink is None:
        attention_sink = torch.full(
            (heads,), float("-inf"), dtype=accumulation_dtype, device=query.device
        )
    sink_logits = attention_sink.to(accumulation_dtype)[None, :, None, None].expand(
        batch, -1, sequence_length, -1
    )
    logits = torch.cat((selected_logits, local_logits, sink_logits), dim=-1)
    probabilities = torch.softmax(logits, dim=-1)
    return torch.logsumexp(logits, dim=-1), probabilities[..., : kv_indices.shape[-1]]


def _make_inputs(backend, share_kv, with_doc_ids, with_attention_sink):
    device = torch.device("cuda" if backend == "triton" else "cpu")
    dtype = torch.float32 if backend == "triton" else torch.float64
    generator = torch.Generator(device=device).manual_seed(9384)
    batch, heads, sequence_length, head_dim = 2, 3, 9, 32
    sparse_sequence_length, num_topk_blocks = 7, 4
    kv_heads = 1 if share_kv else heads

    query = torch.randn(
        batch, heads, sequence_length, head_dim, dtype=dtype, device=device, generator=generator
    )
    local_kv = torch.randn(
        batch,
        kv_heads,
        sequence_length,
        head_dim,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    sparse_kv = torch.randn(
        batch,
        kv_heads,
        sparse_sequence_length,
        head_dim,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    selection_scores = torch.randn(
        batch, sequence_length, sparse_sequence_length, device=device, generator=generator
    )
    kv_indices = selection_scores.topk(num_topk_blocks, dim=-1).indices
    kv_indices[:, ::3, -1] = -1
    kv_indices[:, 1::4, 1] = kv_indices[:, 1::4, 0]

    attention_sink = None
    if with_attention_sink:
        attention_sink = torch.randn(heads, dtype=dtype, device=device, generator=generator)

    doc_ids = None
    if with_doc_ids:
        doc_ids = torch.tensor(
            [[0, 0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 2, 2, 2]],
            dtype=torch.int32,
            device=device,
        )

    return {
        "query": query,
        "local_kv": local_kv,
        "sparse_kv": sparse_kv,
        "kv_indices": kv_indices,
        "attention_sink": attention_sink,
        "doc_ids": doc_ids,
        "sliding_window_size": 4,
    }


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("share_kv", "with_doc_ids", "with_attention_sink"),
    [(False, False, True), (True, True, True), (True, False, False)],
)
def test_lse_and_selected_key_probabilities_match_dense_qk_softmax_oracle(
    backend, share_kv, with_doc_ids, with_attention_sink
):
    inputs = _make_inputs(backend, share_kv, with_doc_ids, with_attention_sink)

    with torch.inference_mode():
        expected_lse, expected_selected_probabilities = _dense_qk_softmax_oracle(**inputs)
        _, aux = selected_attention(
            **inputs,
            backend=backend,
            return_aux=AuxRequest(lse=True),
        )

    assert aux.lse is not None
    assert aux.lse.shape == expected_lse.shape
    tolerance = 1e-3 if backend == "triton" else 1e-12
    torch.testing.assert_close(aux.lse, expected_lse, atol=tolerance, rtol=tolerance)

    query = inputs["query"].to(expected_lse.dtype)
    sparse_kv = (
        inputs["sparse_kv"].to(expected_lse.dtype).expand(query.shape[0], query.shape[1], -1, -1)
    )
    sparse_logits = torch.matmul(query, sparse_kv.transpose(-2, -1)) / math.sqrt(query.shape[-1])
    selected_indices = (
        inputs["kv_indices"].clamp_min(0)[:, None].expand(-1, query.shape[1], -1, -1)
    )
    selected_logits = sparse_logits.gather(dim=-1, index=selected_indices)
    actual_selected_probabilities = torch.exp(selected_logits - aux.lse[..., None])
    actual_selected_probabilities = actual_selected_probabilities.masked_fill(
        ~(inputs["kv_indices"] >= 0)[:, None], 0.0
    )
    torch.testing.assert_close(
        actual_selected_probabilities,
        expected_selected_probabilities,
        atol=tolerance,
        rtol=tolerance,
    )
