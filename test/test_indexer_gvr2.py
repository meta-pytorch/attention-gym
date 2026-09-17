"""Public selector options, registration, compilation and changed-input graph replay."""

import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend

from attn_gym.sparse.indexer import lightning_indexer, ops
from attn_gym.testing.indexer import (
    assert_indexer_selection,
    make_indexer_mxfp8_test_inputs,
    make_indexer_test_inputs,
)


@pytest.fixture(params=["cute", "triton"])
def backend(request):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    capability = torch.cuda.get_device_capability()
    if request.param == "cute":
        if capability not in ((10, 0), (10, 3)):
            pytest.skip("CuTe requires SM100/SM103")
        pytest.importorskip("cutlass.cute")
    elif capability[0] < 9:
        pytest.skip("Triton requires SM90 or newer")
    return request.param


def inputs_for(precision, tokens=65, seed=37):
    if precision == "mxfp8":
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            pytest.skip("MXFP8 requires SM100/SM103")
        return make_indexer_mxfp8_test_inputs(
            tokens, 32, 128, torch.float32, batch=3, compress_ratio=4, seed=seed
        )._asdict()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    q, k, weights = make_indexer_test_inputs(
        tokens, 32, 128, dtype, batch=3, compress_ratio=4, seed=seed
    )
    return {"q": q, "k": k, "weights": weights}


@pytest.mark.parametrize("selector", ["radix", "approximate", None, 1])
def test_invalid_selector(selector):
    inputs = make_indexer_test_inputs(8, 2, 16, torch.bfloat16, device="cpu")
    with pytest.raises(ValueError, match="unsupported lightning_indexer kernel options"):
        lightning_indexer(*inputs, 3, kernel_options={"selector": selector})
    with pytest.raises(ValueError, match="kernel_options are not supported"):
        lightning_indexer(*inputs, 3, impl="reference", kernel_options={"selector": selector})


@pytest.mark.parametrize(
    "selector,topk,expected",
    [
        ("auto", 1, "gvr2"),
        ("auto", 2047, "gvr2"),
        ("auto", 2048, "default"),
        ("auto", 4096, "default"),
        ("default", 1, "default"),
        ("gvr2", 4096, "gvr2"),
    ],
)
def test_cute_auto_selector_rule(selector, topk, expected):
    """CuTe ``auto`` uses GVR2 below its native K limit and radix at or above it."""
    pytest.importorskip("cutlass.cute")
    from attn_gym.sparse.indexer.impl import cute

    assert cute.resolve_selector(selector, topk) == expected
    with pytest.raises(ValueError, match="unknown indexer selector"):
        cute.resolve_selector("radix", topk)


@pytest.mark.parametrize("topk", [7, 2048])
def test_cute_launch_resolves_auto(monkeypatch, topk):
    """The public omitted selector reaches the CuTe compile step already resolved."""
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("CuTe requires SM100/SM103")
    from attn_gym.sparse.indexer.impl import cute

    seen = []
    compile_topk = cute._compile_topk

    def spy(*args):
        seen.append(args[-1])
        return compile_topk(*args)

    monkeypatch.setattr(cute, "_compile_topk", spy)
    q, k, weights = make_indexer_test_inputs(4096, 32, 128, torch.bfloat16, batch=1, seed=5)
    actual = lightning_indexer(q, k, weights, topk, kernel_options={"backend": "cute"})
    assert seen == ["gvr2" if topk < 2048 else "default"]
    assert_indexer_selection(actual, q, k, weights, topk, False)


@pytest.mark.parametrize("precision", ["bf16", "fp16", "mxfp8"])
@pytest.mark.parametrize("selector", [None, "auto", "default", "gvr2"])
def test_public_selector(backend, precision, selector):
    inputs = inputs_for(precision)
    options = {"backend": backend}
    if selector is not None:
        options["selector"] = selector
    actual = lightning_indexer(
        **inputs, topk=7, causal=True, compress_ratio=4, kernel_options=options
    )
    assert_indexer_selection(actual, **inputs, topk=7, causal=True, compress_ratio=4)


@pytest.mark.parametrize("precision", ["bf16", "fp16", "mxfp8"])
def test_selector_fullgraph_and_replay(backend, precision):
    counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(lightning_indexer, fullgraph=True, dynamic=True, backend=counter)
    for selector in ("default", "gvr2"):
        options = {"backend": backend, "selector": selector}
        for tokens in (65, 129):
            inputs = inputs_for(precision, tokens)
            actual = compiled(
                **inputs, topk=7, causal=True, compress_ratio=4, kernel_options=options
            )
            assert_indexer_selection(actual, **inputs, topk=7, causal=True, compress_ratio=4)
    assert counter.frame_count == 2  # Each static selector reuses its graph across lengths.

    for selector in ("default", "gvr2"):
        options = {"backend": backend, "selector": selector}
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = compiled(
                **inputs, topk=7, causal=True, compress_ratio=4, kernel_options=options
            )
        for seed in (91, 92):
            updated = inputs_for(precision, 129, seed)
            for name, tensor in inputs.items():
                tensor.copy_(updated[name])
            graph.replay()
            assert_indexer_selection(actual, **inputs, topk=7, causal=True, compress_ratio=4)


@pytest.mark.parametrize("precision", ["bf16", "fp16", "mxfp8"])
def test_gvr2_registration(backend, precision):
    inputs = inputs_for(precision)
    inputs["weights"].requires_grad_()
    args = (
        inputs["q"],
        inputs["k"],
        inputs["weights"],
        16,
        True,
        4,
        backend,
    )
    kwargs = {"selector": "gvr2"}
    if precision == "mxfp8":
        kwargs.update(q_scale=inputs["q_scale"], k_scale=inputs["k_scale"])
    utilities = [
        "test_schema",
        "test_autograd_registration",
        "test_faketensor",
        "test_aot_dispatch_dynamic",
    ]
    if precision == "mxfp8":
        utilities.remove("test_schema")  # SchemaCheckMode's FP8 allclose is unsupported.
    before = {name: tensor.detach().view(torch.uint8).clone() for name, tensor in inputs.items()}
    torch.library.opcheck(ops._indexer_op, args, kwargs, test_utils=tuple(utilities))
    actual = ops._indexer_op(*args, **kwargs)
    for name, tensor in inputs.items():
        torch.testing.assert_close(tensor.detach().view(torch.uint8), before[name], rtol=0, atol=0)
        assert tensor.untyped_storage().data_ptr() != actual.untyped_storage().data_ptr()
    assert not actual.requires_grad
