"""Tests for paged MLA decode over a compressed latent cache."""

import math

import pytest
import torch

from attn_gym.mla import AuxRequest, Impl, paged_mla_decode
from attn_gym.mla.impl.cute import SUPPORTED_CAPABILITIES
from attn_gym.testing import cumulative_sequence_offsets
from attn_gym.testing.kda import (
    assert_matches_low_precision_reference,
    assert_rms_matches_low_precision_reference,
)

CUTE_CAPABLE = (
    torch.cuda.is_available() and torch.cuda.get_device_capability() in SUPPORTED_CAPABILITIES
)


def _make_inputs(
    query_lengths,
    key_lengths,
    *,
    num_heads,
    latent_dim,
    rope_dim,
    block_size,
    dtype,
    device,
    seed=0,
):
    """Build packed queries and a shuffled paged cache stored as one [latent | rope] tensor."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    blocks_per_sequence = [-(-length // block_size) for length in key_lengths]
    max_blocks = max(blocks_per_sequence)
    num_blocks = sum(blocks_per_sequence) + 3
    cache = torch.randn(
        num_blocks, block_size, latent_dim + rope_dim, generator=generator, dtype=torch.float32
    ).to(device=device, dtype=dtype)
    order = torch.randperm(num_blocks, generator=generator).tolist()
    block_table = torch.zeros(len(key_lengths), max_blocks, dtype=torch.int32)
    for sequence, count in enumerate(blocks_per_sequence):
        block_table[sequence, :count] = torch.tensor(order[:count])
        order = order[count:]
    num_tokens = sum(query_lengths)
    q = torch.randn(
        num_tokens, num_heads, latent_dim + rope_dim, generator=generator, dtype=torch.float32
    ).to(device=device, dtype=dtype)
    return {
        "q_latent": q[..., :latent_dim],
        "q_rope": q[..., latent_dim:],
        "latent_cache": cache[..., :latent_dim],
        "rope_cache": cache[..., latent_dim:],
        "block_table": block_table.to(device),
        "cu_seqlens_q": cumulative_sequence_offsets(query_lengths, device=device),
        "seqused_k": torch.tensor(key_lengths, dtype=torch.int32, device=device),
    }


def test_reference_matches_bottom_right_causal_dense_attention():
    """The paged reference equals 576-style attention with the latent rows as values."""
    query_lengths, key_lengths = [1, 3, 2], [5, 9, 2]
    inputs = _make_inputs(
        query_lengths,
        key_lengths,
        num_heads=3,
        latent_dim=16,
        rope_dim=8,
        block_size=4,
        dtype=torch.float64,
        device="cpu",
    )
    scale = 0.3
    out, aux = paged_mla_decode(
        **inputs,
        max_seqlen_q=3,
        scale=scale,
        impl=Impl.REFERENCE,
        return_aux=AuxRequest(lse=True),
    )
    lse = aux.lse

    cache = torch.cat((inputs["latent_cache"], inputs["rope_cache"]), dim=-1)
    q = torch.cat((inputs["q_latent"], inputs["q_rope"]), dim=-1)
    start = 0
    for sequence, (num_queries, num_keys) in enumerate(zip(query_lengths, key_lengths)):
        keys = cache[inputs["block_table"][sequence].long()].flatten(0, 1)[:num_keys]
        query = q[start : start + num_queries].transpose(0, 1)
        mask = torch.ones(num_queries, num_keys, dtype=torch.bool).tril(num_keys - num_queries)
        logits = scale * query @ keys.T
        expected = torch.nn.functional.scaled_dot_product_attention(
            query, keys[None], keys[None, :, :16], attn_mask=mask, scale=scale
        )
        torch.testing.assert_close(out[start : start + num_queries], expected.transpose(0, 1))
        torch.testing.assert_close(
            lse[start : start + num_queries],
            logits.masked_fill(~mask, float("-inf")).logsumexp(-1).T.float(),
        )
        start += num_queries


def test_rejects_mismatched_metadata():
    inputs = _make_inputs(
        [1, 1],
        [3, 4],
        num_heads=2,
        latent_dim=16,
        rope_dim=8,
        block_size=4,
        dtype=torch.float32,
        device="cpu",
    )
    with pytest.raises(ValueError, match="cu_seqlens_q"):
        paged_mla_decode(
            **{**inputs, "cu_seqlens_q": inputs["cu_seqlens_q"][:-1]},
            max_seqlen_q=1,
            scale=1.0,
            impl="reference",
        )
    with pytest.raises(TypeError, match="block_table must be int32"):
        paged_mla_decode(
            **{**inputs, "block_table": inputs["block_table"].long()},
            max_seqlen_q=1,
            scale=1.0,
            impl="reference",
        )
    with pytest.raises(ValueError, match="latent_cache must have a contiguous last dimension"):
        paged_mla_decode(
            **{**inputs, "latent_cache": inputs["latent_cache"].transpose(1, 2).contiguous().mT},
            max_seqlen_q=1,
            scale=1.0,
            impl="reference",
        )
    with pytest.raises(ValueError, match="seqused_k must be contiguous"):
        paged_mla_decode(
            **{**inputs, "seqused_k": inputs["seqused_k"].repeat_interleave(2)[::2]},
            max_seqlen_q=1,
            scale=1.0,
            impl="reference",
        )
    with pytest.raises(ValueError, match="scale"):
        paged_mla_decode(**inputs, max_seqlen_q=1, scale=0.0, impl="reference")


def test_output_is_forward_only():
    inputs = _make_inputs(
        [2],
        [5],
        num_heads=2,
        latent_dim=16,
        rope_dim=8,
        block_size=4,
        dtype=torch.float32,
        device="cpu",
    )
    out = paged_mla_decode(**inputs, max_seqlen_q=2, scale=1.0, impl="reference")
    assert not out.requires_grad
    q_latent = inputs["q_latent"].clone().requires_grad_()
    with pytest.raises(ValueError, match="q_latent requires grad"):
        paged_mla_decode(
            **{**inputs, "q_latent": q_latent}, max_seqlen_q=2, scale=1.0, impl="reference"
        )


@pytest.mark.skipif(not CUTE_CAPABLE, reason="FA4 absorbed MLA requires SM100 or SM103")
@pytest.mark.parametrize("block_size", [16, 64])
@pytest.mark.parametrize("num_heads", [1, 16, 128])
def test_fused_matches_reference(num_heads, block_size):
    """Decode rows, multi-token rows, and one-key rows over shuffled 576-wide cache views."""
    pytest.importorskip("flash_attn.cute")
    query_lengths = [1, 1, 3, 1, 2, 1]
    key_lengths = [17, 1000, 333, 4096, 64, 1]
    inputs = _make_inputs(
        query_lengths,
        key_lengths,
        num_heads=num_heads,
        latent_dim=512,
        rope_dim=64,
        block_size=block_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    scale = 1 / math.sqrt(192)
    call = {"max_seqlen_q": max(query_lengths), "scale": scale, "return_aux": AuxRequest(lse=True)}
    out, aux = paged_mla_decode(**inputs, **call)
    lse = aux.lse
    low_out, _ = paged_mla_decode(**inputs, impl=Impl.REFERENCE, **call)
    high_inputs = {
        name: tensor.double() if tensor.is_floating_point() else tensor
        for name, tensor in inputs.items()
    }
    high_out, high_aux = paged_mla_decode(**high_inputs, impl=Impl.REFERENCE, **call)

    assert out.shape == inputs["q_latent"].shape and out.dtype == torch.bfloat16
    assert lse.shape == (sum(query_lengths), num_heads) and lse.dtype == torch.float32
    assert_matches_low_precision_reference(out, high_out, low_out, "out")
    assert_rms_matches_low_precision_reference(out, high_out, low_out, "out")
    torch.testing.assert_close(lse, high_aux.lse, atol=1e-4, rtol=0)


OPCHECK_UTILITIES = (
    "test_schema",
    "test_autograd_registration",
    "test_faketensor",
    "test_aot_dispatch_dynamic",
)


def _decode_inputs(num_heads=16, block_size=64):
    query_lengths, key_lengths = [1, 3, 1, 2], [40, 300, 1, 129]
    inputs = _make_inputs(
        query_lengths,
        key_lengths,
        num_heads=num_heads,
        latent_dim=512,
        rope_dim=64,
        block_size=block_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    return inputs, max(query_lengths)


@pytest.mark.skipif(not CUTE_CAPABLE, reason="FA4 absorbed MLA requires SM100 or SM103")
@pytest.mark.parametrize("layout", ["cache_views", "compact", "head_major"])
def test_fused_operator_registration(layout):
    """The registered operator's schema, fake, and dynamic AOT dispatch match the kernel."""
    pytest.importorskip("flash_attn.cute")
    inputs, max_seqlen_q = _decode_inputs()
    if layout != "cache_views":
        for name in ("q_latent", "q_rope", "latent_cache", "rope_cache"):
            compact = inputs[name].contiguous()
            inputs[name] = (
                compact.transpose(0, 1).contiguous().transpose(0, 1)
                if layout == "head_major"
                else compact
            )
    torch.library.opcheck(
        torch.ops.attn_gym.paged_mla_decode_fwd.default,
        (*inputs.values(), max_seqlen_q, 1 / math.sqrt(192)),
        test_utils=OPCHECK_UTILITIES,
    )


@pytest.mark.skipif(not CUTE_CAPABLE, reason="FA4 absorbed MLA requires SM100 or SM103")
def test_fused_compiles_fullgraph():
    pytest.importorskip("flash_attn.cute")
    inputs, max_seqlen_q = _decode_inputs()
    call = {
        "max_seqlen_q": max_seqlen_q,
        "scale": 1 / math.sqrt(192),
        "return_aux": AuxRequest(lse=True),
    }
    torch._dynamo.reset()
    compiled = torch.compile(paged_mla_decode, fullgraph=True)
    out, aux = compiled(**inputs, **call)
    expected_out, expected_aux = paged_mla_decode(**inputs, **call)
    assert torch.equal(out, expected_out)
    assert torch.equal(aux.lse, expected_aux.lse)
