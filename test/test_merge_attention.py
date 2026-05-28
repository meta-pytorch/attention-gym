import math

import pytest
import torch

from attn_gym.utils import LN2, flex_attention_lse_to_merge_lse, merge_attention


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
def test_flex_attention_lse_to_merge_lse_converts_flash_domain_cuda():
    lse = torch.randn(2, 3, 5, device="cuda")
    torch.testing.assert_close(
        flex_attention_lse_to_merge_lse(lse, {"BACKEND": "FLASH"}),
        lse / LN2,
    )
    assert flex_attention_lse_to_merge_lse(lse, {"BACKEND": "TRITON"}) is lse
    assert flex_attention_lse_to_merge_lse(lse) is lse
