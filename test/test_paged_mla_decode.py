"""Tests for paged MLA decode over a compressed latent cache."""

import functools
import math

import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend

from attn_gym._backends import tune as tune_module
from attn_gym._backends.tune import run_tunable
from attn_gym.mla import AuxRequest, Impl, paged_mla_decode
from attn_gym.mla.impl import cute, fa3
from attn_gym.testing import cumulative_sequence_offsets
from attn_gym.testing.kda import (
    assert_matches_low_precision_reference,
    assert_rms_matches_low_precision_reference,
)

CAPABILITY = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
FUSED_CAPABLE = CAPABILITY in cute.SUPPORTED_CAPABILITIES + fa3.SUPPORTED_CAPABILITIES
FUSED_MODULE = (
    "flash_attn.cute" if CAPABILITY in cute.SUPPORTED_CAPABILITIES else "flash_attn_interface"
)
requires_fused = pytest.mark.skipif(
    not FUSED_CAPABLE, reason="Fused absorbed MLA requires SM90 (FA3) or SM100/SM103 (FA4)"
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
    with pytest.raises(ValueError, match="requires CUDA tensors"):
        paged_mla_decode(**inputs, max_seqlen_q=1, scale=1.0)
    for max_seqlen_k in (0, -1, 1.5, True, torch.tensor(3)):
        with pytest.raises(ValueError, match="max_seqlen_k must be None or a positive"):
            paged_mla_decode(
                **inputs, max_seqlen_q=1, max_seqlen_k=max_seqlen_k, scale=1.0, impl="reference"
            )


def test_autotune_buckets_workloads_at_power_of_two_edges():
    """Rows, batch, and keys share winners within a power of two and split at its edges."""
    empty = functools.partial(torch.empty, device="meta")

    def key(*, heads=16, max_seqlen_q=1, num_sequences=4, max_seqlen_k=None):
        args = cute.PagedMLADecodeFA4.Args(
            empty(num_sequences, heads, 512),
            empty(num_sequences, heads, 64),
            empty(100, 64, 512),
            empty(100, 64, 64),
            empty(num_sequences, 16, dtype=torch.int32),
            empty(num_sequences + 1, dtype=torch.int32),
            empty(num_sequences, dtype=torch.int32),
            max_seqlen_q,
            max_seqlen_k,
            1.0,
            True,
        )
        key_heads, rows, two_q_stages, block_size, batch, keys = cute.PagedMLADecodeFA4.tuning_key(
            args, target=None
        )
        assert (key_heads, block_size) == (heads, 64)
        return rows, two_q_stages, batch, keys

    assert [key(max_seqlen_q=rows)[0] for rows in (1, 2, 3, 4, 5)] == [1, 2, 4, 4, 8]
    # At 40 heads, 3 and 4 rows share a bucket but only 4 rows need a second Q stage.
    assert [key(heads=40, max_seqlen_q=rows)[:2] for rows in (3, 4)] == [(4, False), (4, True)]
    assert [key(num_sequences=count)[2] for count in (1, 2, 3, 4, 5, 8, 9)] == [
        1,
        2,
        4,
        4,
        8,
        8,
        16,
    ]
    # 16 pages of 64 keys hold 1024 keys: larger hints and no hint both key as 1024.
    assert [key(max_seqlen_k=hint)[3] for hint in (1, 63, 64, 65, 512, 513, 1025, None)] == [
        1,
        64,
        64,
        128,
        512,
        1024,
        1024,
        1024,
    ]


def test_autotune_skips_split_counts_over_workspace_budget():
    """Split-KV partial outputs over 1 GiB never become candidates."""
    pytest.importorskip("flash_attn.cute.config")
    empty = functools.partial(torch.empty, device="meta")

    def max_splits(num_sequences, rows):
        args = cute.PagedMLADecodeFA4.Args(
            empty(num_sequences * rows, 128, 512),
            empty(num_sequences * rows, 128, 64),
            empty(100, 64, 512),
            empty(100, 64, 64),
            empty(num_sequences, 16, dtype=torch.int32),
            empty(num_sequences + 1, dtype=torch.int32),
            empty(num_sequences, dtype=torch.int32),
            rows,
            None,
            1.0,
            False,
        )
        configs = cute.PagedMLADecodeFA4.configs(args)
        return max(config.num_splits for config in configs if config != "auto")

    # 512 tokens x 128 heads keep 128 MiB of partials per split, so 8 splits fill 1 GiB.
    assert max_splits(256, 2) == 8
    assert max_splits(8, 1) == 128


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


def _make_fused_inputs(query_lengths, key_lengths, *, num_heads, block_size=64):
    return _make_inputs(
        query_lengths,
        key_lengths,
        num_heads=num_heads,
        latent_dim=512,
        rope_dim=64,
        block_size=block_size,
        dtype=torch.bfloat16,
        device="cuda",
    )


def _fused_reference_check(inputs, call):
    """Compute the references once; the returned check compares one fused output and LSE."""
    low_out, _ = paged_mla_decode(**inputs, impl=Impl.REFERENCE, **call)
    high_inputs = {
        name: tensor.double() if tensor.is_floating_point() else tensor
        for name, tensor in inputs.items()
    }
    high_out, high_aux = paged_mla_decode(**high_inputs, impl=Impl.REFERENCE, **call)

    def check(out, lse):
        assert out.shape == inputs["q_latent"].shape and out.dtype == torch.bfloat16
        assert lse.shape == inputs["q_latent"].shape[:2] and lse.dtype == torch.float32
        assert_matches_low_precision_reference(out, high_out, low_out, "out")
        assert_rms_matches_low_precision_reference(out, high_out, low_out, "out")
        torch.testing.assert_close(lse, high_aux.lse, atol=1e-4, rtol=0)

    return check


@requires_fused
@pytest.mark.parametrize("block_size", [16, 64])
@pytest.mark.parametrize("num_heads", [1, 16, 96, 128])
def test_fused_matches_reference(num_heads, block_size):
    """Decode rows, multi-token rows, and one-key rows over shuffled 576-wide cache views."""
    pytest.importorskip(FUSED_MODULE)
    query_lengths = [1, 1, 3, 1, 2, 1]
    key_lengths = [17, 1000, 333, 4096, 64, 1]
    inputs = _make_fused_inputs(
        query_lengths, key_lengths, num_heads=num_heads, block_size=block_size
    )
    call = {
        "max_seqlen_q": max(query_lengths),
        "scale": 1 / math.sqrt(192),
        "return_aux": AuxRequest(lse=True),
    }
    out, aux = paged_mla_decode(**inputs, **call)
    _fused_reference_check(inputs, call)(out, aux.lse)


@requires_fused
@pytest.mark.parametrize("num_heads", [16, 96])
@pytest.mark.parametrize("return_lse", [False, True])
def test_fused_operator_registration(return_lse, num_heads):
    """The fakes match FlashAttention for strided and heads-major queries, including head
    counts that do not divide 128."""
    pytest.importorskip(FUSED_MODULE)
    inputs = _make_fused_inputs([1, 2, 1], [17, 300, 64], num_heads=num_heads)
    for layout in (
        inputs,
        {
            **inputs,
            **{
                name: inputs[name].transpose(0, 1).contiguous().transpose(0, 1)
                for name in ("q_latent", "q_rope")
            },
        },
    ):
        for max_seqlen_k in (None, 300):
            torch.library.opcheck(
                torch.ops.attn_gym.paged_mla_decode_with_lse.default
                if return_lse
                else torch.ops.attn_gym.paged_mla_decode.default,
                (*layout.values(), 2, max_seqlen_k, 1 / math.sqrt(192), False),
            )


@requires_fused
def test_fused_fullgraph_compile_reuses_one_graph():
    """One dynamic graph serves changing batches and bounds and matches eager exactly."""
    pytest.importorskip(FUSED_MODULE)
    counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(paged_mla_decode, fullgraph=True, dynamic=True, backend=counter)
    call = {"scale": 1 / math.sqrt(192), "return_aux": AuxRequest(lse=True)}
    frame_counts = []
    # No two sizes are equal (tokens, num_sequences + 1, block-table width, heads, head dims,
    # max_seqlen_q), so duck sizing adds no guards.
    for step, (query_lengths, key_lengths) in enumerate(
        (
            ([1, 2, 2], [17, 1000, 64]),
            ([2, 1, 3, 1, 2], [1000, 1, 333, 900, 70]),
            ([1, 1, 4, 1, 1, 2, 1], [5, 3000, 333, 900, 70, 2, 1]),
        )
    ):
        inputs = _make_fused_inputs(query_lengths, key_lengths, num_heads=32)
        # New bounds every step, as in a decode loop.
        bounds = {"max_seqlen_q": max(query_lengths), "max_seqlen_k": max(key_lengths) + step}
        out, aux = compiled(**inputs, **call, **bounds)
        expected, expected_aux = paged_mla_decode(**inputs, **call, **bounds)
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
        torch.testing.assert_close(aux.lse, expected_aux.lse, atol=0, rtol=0)
        frame_counts.append(counter.frame_count)
    # The first call compiles twice (the second graph specializes scale); later steps reuse it.
    assert frame_counts == [frame_counts[0]] * 3


@pytest.mark.skipif(
    CAPABILITY not in cute.SUPPORTED_CAPABILITIES, reason="Autotuning requires SM100/SM103 (FA4)"
)
@pytest.mark.parametrize(
    "num_heads, query_lengths",
    [
        (16, [1, 1, 1, 1, 1]),
        (96, [1, 3, 2, 1, 1]),
        (128, [1, 3, 2, 1, 1]),
    ],
)
def test_autotune_matches_reference_and_reuses_winner(
    num_heads, query_lengths, tmp_path, monkeypatch
):
    """The winner is correct and serves other batches in its bucket, with or without LSE,
    while a shorter ``max_seqlen_k`` bucket re-tunes."""
    pytest.importorskip("flash_attn.cute")
    monkeypatch.setenv("WARPDRIVE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(tune_module, "_WINNERS", {})
    monkeypatch.setattr(tune_module, "_WINNERS_FAST", {})
    winners = []
    real_tune = tune_module.tune

    def recording_tune(configs, *args, **kwargs):
        configs = list(configs)
        winners.append(real_tune(configs, *args, **kwargs))
        assert winners[-1] in configs
        return winners[-1]

    monkeypatch.setattr(tune_module, "tune", recording_tune)
    # Five and seven sequences share a batch bucket and the same cache width.
    for extra_queries in ([], [1, 1]):
        lengths = query_lengths + extra_queries
        inputs = _make_fused_inputs(
            lengths, [1000, 17, 333, 64, 900, 1, 512][: len(lengths)], num_heads=num_heads
        )
        call = {
            "max_seqlen_q": max(lengths),
            "max_seqlen_k": 1000,
            "scale": 1 / math.sqrt(192),
            "return_aux": AuxRequest(lse=True),
        }
        out, aux = paged_mla_decode(**inputs, **call, kernel_options={"autotune": True})
        _fused_reference_check(inputs, call)(out, aux.lse)
    assert len(winners) == 1
    torch.testing.assert_close(
        paged_mla_decode(
            **inputs, **{**call, "return_aux": None}, kernel_options={"autotune": True}
        ),
        out,
        atol=0,
        rtol=0,
    )
    assert len(winners) == 1

    # An empty batch does not tune.
    assert paged_mla_decode(
        **{name: inputs[name][:0] for name in ("q_latent", "q_rope", "block_table", "seqused_k")},
        **{name: inputs[name] for name in ("latent_cache", "rope_cache")},
        cu_seqlens_q=inputs["cu_seqlens_q"][:1],
        **{**call, "return_aux": None},
        kernel_options={"autotune": True},
    ).shape == (0, num_heads, 512)
    assert len(winners) == 1

    # A tuned bucket replays under CUDA graph capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured, captured_aux = paged_mla_decode(
            **inputs, **call, kernel_options={"autotune": True}
        )
    graph.replay()
    torch.testing.assert_close(captured, out, atol=0, rtol=0)
    torch.testing.assert_close(captured_aux.lse, aux.lse, atol=0, rtol=0)
    assert len(winners) == 1

    # Shorter contexts in the same block table get their own winner.
    short_inputs = {**inputs, "seqused_k": inputs["seqused_k"].clamp(max=64)}
    short_call = {**call, "max_seqlen_k": 64}
    out, aux = paged_mla_decode(**short_inputs, **short_call, kernel_options={"autotune": True})
    _fused_reference_check(short_inputs, short_call)(out, aux.lse)
    assert len(winners) == 2

    # An untuned bucket raises a clear error under capture.
    with (
        pytest.raises(RuntimeError, match="during CUDA graph capture"),
        torch.cuda.graph(torch.cuda.CUDAGraph()),
    ):
        paged_mla_decode(
            **short_inputs,
            **{**short_call, "max_seqlen_k": 128},
            kernel_options={"autotune": True},
        )
    assert len(winners) == 2


@pytest.mark.skipif(
    CAPABILITY not in cute.SUPPORTED_CAPABILITIES, reason="Autotuning requires SM100/SM103 (FA4)"
)
@pytest.mark.parametrize(
    "num_heads, query_lengths",
    [(16, [1, 1, 1]), (64, [3, 1, 2]), (96, [1, 1, 1]), (128, [1, 3, 2])],
)
def test_every_autotune_candidate_matches_reference(num_heads, query_lengths):
    """Every generated candidate is correct, not only the winner tuning picks on this GPU."""
    pytest.importorskip("flash_attn.cute")
    inputs = _make_fused_inputs(query_lengths, [300, 17, 1000], num_heads=num_heads)
    max_seqlen_q, scale = max(query_lengths), 1 / math.sqrt(192)
    check = _fused_reference_check(
        inputs, {"max_seqlen_q": max_seqlen_q, "scale": scale, "return_aux": AuxRequest(lse=True)}
    )
    args = cute.PagedMLADecodeFA4.Args(*inputs.values(), max_seqlen_q, 1000, scale, True)
    for config in cute.PagedMLADecodeFA4.configs(args):
        (out, lse), _ = run_tunable(cute.PagedMLADecodeFA4, args, config=config)
        check(out, lse)
