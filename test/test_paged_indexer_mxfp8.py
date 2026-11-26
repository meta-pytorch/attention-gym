"""Explicit-scale MXFP8 selection from physically paged completed-block keys."""

import pytest
import torch

from attn_gym.sparse import paged_lightning_indexer
from attn_gym.testing.indexer import assert_paged_indexer_selection, make_paged_indexer_test_inputs


def require_native_mxfp8() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("native paged MXFP8 requires SM100 or SM103")
    pytest.importorskip("cutlass.cute")


@pytest.fixture(params=["reference", "fused"])
def route(request):
    if request.param == "reference":
        return "cpu", "reference"
    require_native_mxfp8()
    return "cuda", "fused"


def paged_inputs(weight_dtype=torch.bfloat16, device="cpu") -> dict[str, torch.Tensor]:
    q = torch.zeros(4, 4, 128, device=device)
    q[:, 0, 0] = q[:, 0, 32] = q[:, 1, 64] = 1
    weights = torch.zeros(4, 4, dtype=weight_dtype, device=device)
    weights[:, 0], weights[:, 1] = 1, -1
    keys = torch.zeros(3, 64, 128, device=device)
    # Sequence 1's first logical page is physical page 2. Candidate 4 would win
    # without its negative head weight. Its last page has only one completed key.
    keys[2, 3, 0] = 2
    keys[2, 4, 0], keys[2, 4, 64] = 4, 32
    keys[0, 0, 32] = 1
    keys[0, 1:] = torch.nan
    qs = torch.ones(4, 4, 4, device=device)
    ks = torch.ones(3, 64, 4, device=device)
    qs[2, 0, 0] = 4
    ks[0, 0, 1] = 4
    ks[0, 1:] = torch.nan
    table = torch.tensor([[1, -1], [2, 0]], dtype=torch.int32, device=device)
    return {
        "query": q.to(torch.float8_e4m3fn),
        "key_cache": keys.to(torch.float8_e4m3fn),
        "weights": weights,
        "block_table": table,
        "cu_seqlens_q": table.new_tensor([0, 1, 3]),
        "seqused_k": table.new_tensor([2, 131]),
        "q_scale": qs.to(torch.float8_e8m0fnu),
        "k_scale": ks.to(torch.float8_e8m0fnu),
    }


@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_scales_follow_physical_pages_and_packed_queries(route, weight_dtype):
    device, impl = route
    operands = paged_inputs(weight_dtype, device)
    actual = paged_lightning_indexer(**operands, topk=1, compress_ratio=2, impl=impl)
    # For row 1 key 64 beats key 3 by 4 to 2. Row 2's Q scale reverses this, 4 to 8.
    assert actual.tolist() == [[0], [64], [3], [-1]]


def test_fused_graph_replay_observes_scale_pages_and_query_offsets():
    require_native_mxfp8()
    operands = paged_inputs(device="cuda")

    def select():
        return paged_lightning_indexer(**operands, topk=1, compress_ratio=2)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        warm = select()
    torch.cuda.current_stream().wait_stream(stream)
    assert warm.flatten().tolist() == [0, 64, 3, -1]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = select()

    qs, ks = operands["q_scale"], operands["k_scale"]
    qs[1, 0, 0], qs[2, 0, 0] = 4, 1
    graph.replay()
    assert selected.flatten().tolist() == [0, 3, 64, -1]
    ks[0, 0, 1] = 1
    graph.replay()
    assert selected.flatten().tolist() == [0, 3, 3, -1]
    # Repack: sequence 0 empties, and sequence 1 acquires a different first page.
    operands["cu_seqlens_q"].copy_(operands["cu_seqlens_q"].new_tensor([0, 0, 2]))
    operands["block_table"][1, 0] = 1
    graph.replay()
    assert selected.flatten().tolist() == [64, 64, -1, -1]


def test_public_fullgraph_with_explicit_scales():
    require_native_mxfp8()
    x = paged_inputs(device="cuda")
    compiled = torch.compile(paged_lightning_indexer, fullgraph=True, dynamic=True)
    out = compiled(**x, topk=1, compress_ratio=2)
    assert out.tolist() == [[0], [64], [3], [-1]]


def test_fused_rejects_explicit_triton():
    require_native_mxfp8()
    with pytest.raises(ValueError, match="CuTe backend"):
        paged_lightning_indexer(
            **paged_inputs(device="cuda"), topk=1, kernel_options={"backend": "triton"}
        )


@pytest.mark.parametrize("name", ["q_scale", "k_scale"])
def test_strided_scales_are_reference_only(route, name):
    device, impl = route
    operands = paged_inputs(device=device)
    scale = operands[name]
    operands[name] = scale.transpose(1, 2).contiguous().transpose(1, 2)
    if impl == "fused":
        with pytest.raises(ValueError, match="compact.*scales"):
            paged_lightning_indexer(**operands, topk=1, compress_ratio=2)
    else:
        actual = paged_lightning_indexer(**operands, topk=1, compress_ratio=2, impl=impl)
        assert actual.flatten().tolist() == [0, 64, 3, -1]


def test_unscaled_bf16_still_selects_raw_winner():
    operands = paged_inputs()
    del operands["q_scale"], operands["k_scale"]
    for name in ("query", "key_cache"):
        operands[name] = operands[name].bfloat16()
    actual = paged_lightning_indexer(**operands, topk=1, compress_ratio=2, impl="reference")
    assert actual.tolist() == [[0], [3], [3], [-1]]


def test_opposite_scale_exponents_and_cancellation(route):
    device, impl = route
    q = torch.zeros(1, 4, 128, device=device)
    q[0, 0, 0] = q[0, 0, 32] = 448
    keys = torch.zeros(1, 64, 128, device=device)
    keys[0, 0, 0], keys[0, 0, 32] = 2, -2
    keys[0, 1, 0] = 1
    qs = torch.ones(1, 4, 4, device=device)
    ks = torch.ones(1, 64, 4, device=device)
    qs[..., 0], ks[..., 0] = 2.0**120, 2.0**-120
    table = torch.zeros(1, 1, dtype=torch.int32, device=device)
    actual = paged_lightning_indexer(
        q.to(torch.float8_e4m3fn),
        keys.to(torch.float8_e4m3fn),
        torch.ones(1, 4, device=device),
        table,
        table.new_tensor([0, 1]),
        table.new_tensor([2]),
        topk=1,
        compress_ratio=1,
        q_scale=qs.to(torch.float8_e8m0fnu),
        k_scale=ks.to(torch.float8_e8m0fnu),
        impl=impl,
    )
    # Dequantizing Q first overflows FP32. With combined factors the unnormalized
    # scores are 896 - 896 and 448; applying ReLU per 32-channel group also fails.
    assert actual.item() == 1


@pytest.mark.parametrize("topk", [0, 1, 512])
def test_cuda_op_rejects_scales_for_bf16(topk, requires_blackwell):
    from attn_gym.sparse.indexer.ops import _paged_indexer_op

    x = paged_inputs(device="cuda")
    with pytest.raises(ValueError, match="only supported for MXFP8"):
        _paged_indexer_op(
            x["query"].bfloat16(),
            x["key_cache"].bfloat16(),
            x["weights"],
            x["block_table"],
            x["cu_seqlens_q"],
            x["seqused_k"],
            topk,
            2,
            None,
            "cute",
            x["q_scale"],
            x["k_scale"],
        )


def test_empty_candidate_table_returns_padding(route):
    device, impl = route
    x = paged_inputs(device=device)
    x["block_table"] = x["block_table"].new_empty((2, 0))
    x["seqused_k"].zero_()
    result = paged_lightning_indexer(**x, topk=1, impl=impl)
    assert result.tolist() == [[-1], [-1], [-1], [-1]]


def test_reference_multiquery_suffix_uses_per_query_scales():
    # Cross the reference's query-tile boundary. Scaling group 0 breaks the tie
    # between the two keys; swapping the group scales must swap the selected ID.
    n = 131
    q = torch.ones(n, 1, 64).to(torch.float8_e4m3fn)
    k = torch.zeros(1, 2, 64)
    k[0, 0, 0] = k[0, 1, 32] = 1
    qs = torch.ones(n, 1, 2)
    qs[::2, :, 0] = qs[1::2, :, 1] = 2
    cu = torch.tensor([0, n], dtype=torch.int32)
    ids = paged_lightning_indexer(
        q,
        k.to(q.dtype),
        torch.ones(n, 1),
        cu.new_tensor([[0]]),
        cu,
        cu.new_tensor([n * 2]),
        topk=1,
        compress_ratio=n,
        query_positions=cu.new_full((n,), n * 2 - 1),
        q_scale=qs.to(torch.float8_e8m0fnu),
        k_scale=torch.ones(1, 2, 2).to(torch.float8_e8m0fnu),
        impl="reference",
    )
    assert ids[:, 0].tolist() == [i % 2 for i in range(n)]


def test_reference_scale_product_can_exceed_fp32():
    # Two E8M0 scales of 2**64 produce a finite scaled dot with subnormal E4M3 payloads.
    q = torch.full((1, 1, 32), 2.0**-9).to(torch.float8_e4m3fn)
    keys = torch.tensor([2.0**-9, 2.0**-8]).view(1, 2, 1).expand(1, 2, 32).to(q.dtype)
    table = torch.zeros(1, 1, dtype=torch.int32)
    actual = paged_lightning_indexer(
        q,
        keys,
        torch.ones(1, 1),
        table,
        table.new_tensor([0, 1]),
        table.new_tensor([2]),
        topk=1,
        compress_ratio=1,
        q_scale=torch.full((1, 1, 1), 2.0**64).to(torch.float8_e8m0fnu),
        k_scale=torch.full((1, 2, 1), 2.0**64).to(torch.float8_e8m0fnu),
        impl="reference",
    )
    assert actual.item() == 1


def test_fp64_dequantized_oracle(route):
    device, impl = route
    args = make_paged_indexer_test_inputs((2, 1), (280, 771), tail=1, device=device)
    q, keys, weights, table, cu, lengths = args
    q, keys = q.to(torch.float8_e4m3fn), keys.to(torch.float8_e4m3fn)
    rng = torch.Generator(device=device).manual_seed(91)

    def scales(payload: torch.Tensor) -> torch.Tensor:
        shape = (*payload.shape[:-1], payload.size(-1) // 32)
        exponents = torch.randint(-4, 5, shape, generator=rng, device=device)
        return exponents.float().exp2().to(torch.float8_e8m0fnu)

    qs, ks = scales(q), scales(keys)
    positions = cu.new_tensor([267, -1, 767, -1])
    actual = paged_lightning_indexer(
        q,
        keys,
        weights,
        table,
        cu,
        lengths,
        topk=7,
        query_positions=positions,
        impl=impl,
        q_scale=qs,
        k_scale=ks,
    )
    # Independently dequantize operands in FP64 before the oracle gathers logical keys.
    q64 = q.double() * qs.double().repeat_interleave(32, dim=-1)
    k64 = keys.double() * ks.double().repeat_interleave(32, dim=-1)
    assert_paged_indexer_selection(
        (q64, k64, weights, table, cu, lengths), actual, 7, positions=positions
    )


@pytest.mark.parametrize(
    "fault,error,message",
    [
        ("no_q_scale", ValueError, "q_scale"),
        ("no_k_scale", ValueError, "k_scale"),
        ("wrong_q_groups", ValueError, "q_scale"),
        ("wrong_page_count", ValueError, "k_scale"),
        ("payload_as_scale", TypeError, "k_scale"),
        ("scale_device", ValueError, "q_scale"),
        ("fp16_weights", TypeError, "weights"),
        ("non_mx_with_scales", ValueError, "scale"),
        ("partial_group", ValueError, "32"),
    ],
)
def test_invalid_scale_metadata(fault, error, message):
    operands = paged_inputs()
    match fault:
        case "no_q_scale" | "no_k_scale":
            del operands[fault.removeprefix("no_")]
        case "wrong_q_groups":
            operands["q_scale"] = operands["q_scale"][..., :-1]
        case "wrong_page_count":
            operands["k_scale"] = operands["k_scale"][:1]
        case "payload_as_scale":
            operands["k_scale"] = operands["k_scale"].view(torch.float8_e4m3fn)
        case "scale_device":
            operands["q_scale"] = operands["q_scale"].to("meta")
        case "fp16_weights":
            operands["weights"] = operands["weights"].half()
        case "non_mx_with_scales":
            for name in ("query", "key_cache"):
                operands[name] = operands[name].bfloat16()
        case "partial_group":
            for name in ("query", "key_cache"):
                operands[name] = operands[name][..., :-1]
    with pytest.raises(error, match=message):
        paged_lightning_indexer(**operands, topk=1, impl="reference")
