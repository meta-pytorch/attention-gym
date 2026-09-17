"""Explicit-scale MXFP8 public semantics and compiler/registration contracts."""

import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend

from attn_gym.sparse.indexer import lightning_indexer
from attn_gym.sparse.indexer.ops import _indexer_op
from attn_gym.testing.indexer import (
    ScaledIndexerInputs,
    assert_indexer_selection,
    make_indexer_mxfp8_test_inputs,
)


def require_mxfp8_cuda(backend: str = "cute") -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("MXFP8 indexer requires SM100 or SM103")
    if backend == "cute":
        pytest.importorskip("cutlass.cute")


@pytest.fixture(params=["cute", "triton"])
def backend(request: pytest.FixtureRequest) -> str:
    require_mxfp8_cuda(request.param)
    return request.param


def check_result(
    actual: torch.Tensor, inputs: ScaledIndexerInputs, topk: int, causal: bool, ratio: int
) -> None:
    assert_indexer_selection(
        actual, **inputs._asdict(), topk=topk, causal=causal, compress_ratio=ratio
    )


@pytest.mark.parametrize("weights_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("causal,ratio", [(False, 1), (True, 1), (True, 4)])
def test_mxfp8_reference_scales(weights_dtype, causal, ratio):
    inputs = make_indexer_mxfp8_test_inputs(
        17, 3, 64, weights_dtype, compress_ratio=ratio, device="cpu"
    )
    actual = lightning_indexer(
        **inputs._asdict(), topk=3, causal=causal, compress_ratio=ratio, impl="reference"
    )
    check_result(actual, inputs, 3, causal, ratio)


@pytest.mark.parametrize(
    "fault,error,message",
    [
        ("missing_q", ValueError, "both q_scale and k_scale"),
        ("missing_k", ValueError, "both q_scale and k_scale"),
        ("q_shape", ValueError, "q_scale must have shape"),
        ("k_shape", ValueError, "k_scale must have shape"),
        ("q_dtype", TypeError, "q_scale must have dtype float8_e8m0fnu"),
        ("k_dtype", TypeError, "k_scale must have dtype float8_e8m0fnu"),
        ("q_device", ValueError, "q_scale must be on the same device"),
        ("k_device", ValueError, "k_scale must be on the same device"),
        ("not_tensor", TypeError, "q_scale must be a torch.Tensor"),
        ("weights_dtype", TypeError, "BF16 or FP32 weights"),
        ("bf16_with_scales", ValueError, "only supported for MXFP8"),
    ],
)
def test_mxfp8_scale_metadata_rejected(fault, error, message):
    values = make_indexer_mxfp8_test_inputs(8, 2, 64, torch.bfloat16, device="cpu")._asdict()
    match fault:
        case "missing_q":
            values["q_scale"] = None
        case "missing_k":
            values["k_scale"] = None
        case "q_shape":
            values["q_scale"] = values["q_scale"][..., :1]
        case "k_shape":
            values["k_scale"] = values["k_scale"][..., None]
        case "q_dtype":
            values["q_scale"] = values["q_scale"].float()
        case "k_dtype":
            values["k_scale"] = values["k_scale"].float()
        case "q_device":
            values["q_scale"] = values["q_scale"].to("meta")
        case "k_device":
            values["k_scale"] = values["k_scale"].to("meta")
        case "not_tensor":
            values["q_scale"] = 1.0
        case "weights_dtype":
            values["weights"] = values["weights"].half()
        case "bf16_with_scales":
            values["q"] = values["q"].bfloat16()
            values["k"] = values["k"].bfloat16()
    with pytest.raises(error, match=message):
        lightning_indexer(**values, topk=2, impl="reference")


@pytest.mark.parametrize(
    "weights_dtype,heads,ratio", [(torch.bfloat16, 32, 1), (torch.float32, 64, 4)]
)
def test_mxfp8_public_matches_reference(weights_dtype, heads, ratio, backend):
    inputs = make_indexer_mxfp8_test_inputs(129, heads, 128, weights_dtype, compress_ratio=ratio)
    actual = lightning_indexer(
        **inputs._asdict(),
        topk=7,
        causal=True,
        compress_ratio=ratio,
        kernel_options={"backend": backend},
    )
    check_result(actual, inputs, 7, True, ratio)


@pytest.mark.parametrize("weights_dtype", [torch.bfloat16, torch.float32])
def test_mxfp8_registration(weights_dtype, backend):
    inputs = make_indexer_mxfp8_test_inputs(65, 32, 128, weights_dtype, compress_ratio=4)
    inputs.weights.requires_grad_()
    # Selecting all candidates has stable order for AOT opcheck's exact comparison.
    # SchemaCheckMode uses allclose, whose FP8 mul_cuda path is unimplemented.
    # Check input mutation bytewise and retain the other registration utilities.
    before = [tensor.detach().view(torch.uint8).clone() for tensor in inputs]
    torch.library.opcheck(
        _indexer_op,
        (inputs.q, inputs.k, inputs.weights, 16, True, 4, backend),
        {"q_scale": inputs.q_scale, "k_scale": inputs.k_scale},
        test_utils=("test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic"),
    )
    actual = lightning_indexer(
        **inputs._asdict(),
        topk=16,
        causal=True,
        compress_ratio=4,
        kernel_options={"backend": backend},
    )
    for tensor, saved in zip(inputs, before):
        torch.testing.assert_close(tensor.detach().view(torch.uint8), saved, rtol=0, atol=0)
        assert actual.untyped_storage().data_ptr() != tensor.untyped_storage().data_ptr()
    assert not actual.requires_grad and actual.grad_fn is None
    check_result(actual, inputs, 16, True, 4)


def test_mxfp8_dynamic_fullgraph_and_scale_replay(backend):
    counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(lightning_indexer, fullgraph=True, dynamic=True, backend=counter)
    for tokens in (65, 129):
        inputs = make_indexer_mxfp8_test_inputs(
            tokens, 32, 128, torch.float32, batch=3, compress_ratio=4
        )
        actual = compiled(
            **inputs._asdict(),
            topk=7,
            causal=True,
            compress_ratio=4,
            kernel_options={"backend": backend},
        )
        check_result(actual, inputs, 7, True, 4)
    assert counter.frame_count == 1

    def run():
        return lightning_indexer(
            **inputs._asdict(),
            topk=7,
            causal=True,
            compress_ratio=4,
            kernel_options={"backend": backend},
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for seed in (89, 90):
        updated = make_indexer_mxfp8_test_inputs(
            129, 32, 128, torch.float32, batch=3, compress_ratio=4, seed=seed
        )
        inputs.q_scale.copy_(updated.q_scale)
        inputs.k_scale.copy_(updated.k_scale)
        graph.replay()
        check_result(actual, inputs, 7, True, 4)


@pytest.mark.parametrize("operand", ["q_scale", "k_scale"])
def test_mxfp8_scale_wide_singleton_stride(operand, backend):
    inputs = make_indexer_mxfp8_test_inputs(65, 32, 128, torch.bfloat16, batch=1, compress_ratio=4)
    scale = getattr(inputs, operand)
    scale = scale.as_strided(scale.shape, (2**31 + 1, *scale.stride()[1:]))
    inputs = inputs._replace(**{operand: scale})
    actual = lightning_indexer(
        **inputs._asdict(),
        topk=7,
        causal=True,
        compress_ratio=4,
        kernel_options={"backend": backend},
    )
    check_result(actual, inputs, 7, True, 4)


@pytest.mark.parametrize(
    "q_packed,k_packed,fallback_layout",
    [
        (True, True, "misaligned"),
        (True, False, "misaligned"),
        (False, True, "misaligned"),
        (False, False, "misaligned"),
        (True, False, "group_stride"),
        (False, True, "group_stride"),
        (False, False, "group_stride"),
    ],
)
def test_mxfp8_cute_scale_specialization(monkeypatch, q_packed, k_packed, fallback_layout):
    require_mxfp8_cuda()
    from attn_gym.sparse.indexer.impl import cute

    inputs = make_indexer_mxfp8_test_inputs(65, 32, 128, torch.float32, compress_ratio=4)
    for name, packed in (("q_scale", q_packed), ("k_scale", k_packed)):
        if packed:
            continue
        scale = getattr(inputs, name)
        if fallback_layout == "misaligned":
            storage = torch.empty(scale.numel() + 1, dtype=scale.dtype, device=scale.device)
            replacement = storage[1:].view(scale.shape)
        else:
            storage = torch.empty(
                (*scale.shape[:-1], scale.shape[-1] * 2), dtype=scale.dtype, device=scale.device
            )
            replacement = storage[..., ::2]
        replacement.copy_(scale)
        inputs = inputs._replace(**{name: replacement})

    compile_scores = cute._compile_mxfp8_scores
    flags = []

    def record_compile(*args):
        flags.append(args[-2:])
        return compile_scores(*args)

    monkeypatch.setattr(cute, "_compile_mxfp8_scores", record_compile)
    actual = lightning_indexer(
        **inputs._asdict(),
        topk=7,
        causal=True,
        compress_ratio=4,
        kernel_options={"backend": "cute"},
    )
    assert flags == [(q_packed, k_packed)]
    check_result(actual, inputs, 7, True, 4)


@pytest.mark.parametrize("heads,dim", [(16, 128), (32, 64)])
def test_mxfp8_fused_domain_rejected(backend, heads, dim):
    inputs = make_indexer_mxfp8_test_inputs(65, heads, dim, torch.bfloat16, compress_ratio=4)
    with pytest.raises(ValueError, match="MXFP8.*H.*D"):
        lightning_indexer(
            **inputs._asdict(),
            topk=7,
            causal=True,
            compress_ratio=4,
            kernel_options={"backend": backend},
        )


def test_mxfp8_group_size_rejected():
    inputs = make_indexer_mxfp8_test_inputs(8, 2, 64, torch.bfloat16, device="cpu")
    inputs = inputs._replace(q=inputs.q[..., :48], k=inputs.k[..., :48])
    with pytest.raises(ValueError, match="divisible by 32"):
        lightning_indexer(**inputs._asdict(), topk=3, impl="reference")


def test_mxfp8_default_route():
    require_mxfp8_cuda()
    inputs = make_indexer_mxfp8_test_inputs(65, 32, 128, torch.bfloat16, compress_ratio=4)
    actual = lightning_indexer(**inputs._asdict(), topk=7, causal=True, compress_ratio=4)
    check_result(actual, inputs, 7, True, 4)


@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "packed_q,packed_k", [(False, False), (False, True), (True, False), (True, True)]
)
def test_production_mxfp8_fake_signature(monkeypatch, wide, weight_dtype, packed_q, packed_k):
    """Inspect production fake promises, rather than a mirrored test compiler."""
    pytest.importorskip("cutlass.cute")
    from attn_gym.sparse.indexer.impl import cute as impl

    monkeypatch.setattr(impl, "compile_tvm_ffi", lambda operation, *args: args)
    q, k, weights, qs, ks, scores, *_ = impl._compile_mxfp8_scores.__wrapped__(
        weight_dtype, 32, 128, True, 4, wide, False, packed_q, packed_k
    )
    width = 64 if wide else 32
    for tensor in (q, k):
        assert tensor._assumed_align == 16
        assert tensor.stride[-1] == 1
        assert all(s.width == width and s.divisibility == 16 for s in tensor.stride[:-1])
    assert weights._assumed_align == torch.empty((), dtype=weight_dtype).element_size()
    assert all(s.width == width and s.divisibility == 1 for s in weights.stride)
    for tensor, packed in ((qs, packed_q), (ks, packed_k)):
        assert tensor.shape[-1] == 4
        assert tensor._assumed_align == (4 if packed else 1)
        if packed:
            assert tensor.stride[-1] == 1
        dynamic = tensor.stride[:-1] if packed else tensor.stride
        assert all(s.width == width and s.divisibility == (4 if packed else 1) for s in dynamic)
    assert scores.stride[-1] == 1
    assert scores.stride[0].width == width
