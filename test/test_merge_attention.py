import math

import pytest
import torch
from torch.nn.attention.flex_attention import AuxRequest, flex_attention

from attn_gym.utils import merge_attention


def _attention_with_lse(q, k, v):
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    lse = torch.logsumexp(scores, dim=-1)
    out = torch.softmax(scores, dim=-1) @ v
    return out, lse


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_merge_attention_matches_concatenated_kv_attention_cuda():
    torch.manual_seed(0)
    device = "cuda"
    q = torch.randn(2, 3, 5, 8, device=device)
    k_prefix = torch.randn(2, 3, 4, 8, device=device)
    v_prefix = torch.randn(2, 3, 4, 8, device=device)
    k_response = torch.randn(2, 3, 6, 8, device=device)
    v_response = torch.randn(2, 3, 6, 8, device=device)

    prefix_out, prefix_lse = _attention_with_lse(q, k_prefix, v_prefix)
    response_out, response_lse = _attention_with_lse(q, k_response, v_response)
    merged_out, merged_lse = merge_attention(prefix_out, prefix_lse, response_out, response_lse)

    expected_out, expected_lse = _attention_with_lse(
        q,
        torch.cat([k_prefix, k_response], dim=-2),
        torch.cat([v_prefix, v_response], dim=-2),
    )

    torch.testing.assert_close(merged_out, expected_out)
    torch.testing.assert_close(merged_lse, expected_lse)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_flex_attention_aux_lse_matches_natural_logsumexp_cuda():
    torch.manual_seed(0)
    device = "cuda"
    q = torch.randn(1, 2, 128, 64, device=device, dtype=torch.float16)
    k = torch.randn(1, 2, 128, 64, device=device, dtype=torch.float16)
    v = torch.randn(1, 2, 128, 64, device=device, dtype=torch.float16)
    out, aux = torch.compile(flex_attention)(
        q, k, v, return_aux=AuxRequest(lse=True), kernel_options={"BACKEND": "TRITON"}
    )
    expected_lse = torch.logsumexp(
        (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(64), dim=-1
    )

    assert out.is_cuda
    torch.testing.assert_close(aux.lse.float(), expected_lse, atol=1e-5, rtol=1e-5)
