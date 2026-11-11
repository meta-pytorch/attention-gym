"""QSA's grouped/separate-value contract on native warpdrive kernels."""

import pytest
import torch

from attn_gym.sparse.gather_attn import AuxRequest, gather_attn
from attn_gym.testing.gather_attn import (
    assert_numerics,
    assert_sink_gradient_fp32,
    make_gather_attn_test_inputs,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="datacenter Blackwell required",
)


def _inputs(
    *,
    heads=6,
    kv_heads=2,
    dim=32,
    tokens=17,
    sparse_tokens=23,
    topk=37,
    separate=True,
    batch=1,
):
    return make_gather_attn_test_inputs(
        heads=heads,
        kv_heads=kv_heads,
        head_dim=dim,
        tokens=tokens,
        sparse_tokens=sparse_tokens,
        topk=topk,
        separate_values=separate,
        dtype=torch.bfloat16,
        batch=batch,
    )


def _leaves(inputs):
    return list(
        {
            id(tensor): tensor.requires_grad_()
            for tensor in inputs.values()
            if isinstance(tensor, torch.Tensor) and tensor.is_floating_point()
        }.values()
    )


def _run(inputs, window, grad, operation=gather_attn, **options):
    leaves = _leaves(inputs)
    output, aux = operation(
        **inputs, sliding_window_size=window, return_aux=AuxRequest(lse=True), **options
    )
    grads = torch.autograd.grad(output, leaves, grad)
    return output, aux.lse, grads


def _check(inputs, window):
    # Preserve QSA's local/sparse aliases when making the FP64 reference leaves.
    converted = {}
    reference = {}
    for name, tensor in inputs.items():
        if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
            reference[name] = converted.setdefault(id(tensor), tensor.detach().double())
        else:
            reference[name] = tensor
    grad = torch.randn_like(inputs["query"])
    expected = _run(reference, window, grad.double(), impl="reference")
    actual = _run(inputs, window, grad, kernel_options={"backend": "cute"})
    assert_numerics(actual[0], expected[0], "output")
    finite = expected[1].isfinite()
    assert torch.equal(actual[1].isfinite(), finite)
    torch.testing.assert_close(
        actual[1][finite].double(), expected[1][finite], atol=3e-5, rtol=3e-5
    )
    for tensor, got, want in zip(_leaves(inputs), actual[2], expected[2], strict=True):
        if tensor is inputs.get("attention_sink") and tensor.dtype == torch.float32:
            assert_sink_gradient_fp32(tensor, actual[1], actual[0], grad, got)
        else:
            assert_numerics(got, want, "gradient")
    return actual


@pytest.mark.parametrize(
    "heads,kv_heads,dim,separate,window,topk",
    [
        (6, 2, 32, True, 3, 1),
        (24, 2, 256, True, 0, 2051),
        (8, 2, 96, False, 19, 67),
        (5, 1, 64, True, 7, 37),
        (4, 4, 16, True, 0, 17),
        (6, 2, 32, True, 0, 0),
        (34, 2, 48, True, 5, 65),
        (33, 1, 80, True, 0, 129),
    ],
)
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.usefixtures("nan_filled_empty")
def test_forward_and_gradients(heads, kv_heads, dim, separate, window, topk, packed):
    inputs = _inputs(
        heads=heads,
        kv_heads=kv_heads,
        dim=dim,
        separate=separate,
        tokens=33,
        sparse_tokens=41,
        topk=topk,
        batch=1 if packed else 2,
    )
    if packed:
        inputs["cu_seqlens"] = torch.tensor([0, 0, 5, 5, 17, 33], device="cuda", dtype=torch.int32)
        inputs["cu_seqlens_k"] = torch.tensor(
            [0, 0, 7, 7, 7, 41], device="cuda", dtype=torch.int32
        )
    if dim == 96:
        inputs["attention_sink"] = inputs["attention_sink"].float()
    _check(inputs, window)


@pytest.mark.parametrize(
    "variant",
    ["outer-strided", "misaligned", "permuted", "wide", "singleton-wide", "no-local", "index32"],
)
def test_layouts_and_wide_offsets(monkeypatch, variant):
    from attn_gym.sparse.gather_attn.impl import cute_grouped

    # Observe the actual compile ABI, including when the disk cache is warm.
    compile_grouped = cute_grouped.compile_grouped
    vectorized = []

    def compile_op(op):
        vectorized.append(op.vectorized)
        return compile_grouped(op)

    compile_op.by_static_key = lambda key, factory: factory()
    monkeypatch.setattr(cute_grouped, "compile_grouped", compile_op)
    inputs = _inputs()
    for name, tensor in list(inputs.items()):
        if variant == "misaligned":
            storage = tensor.new_empty(tensor.numel() + 1)
            inputs[name] = storage[1:].view(tensor.shape).copy_(tensor)
        elif variant == "outer-strided" and tensor.ndim > 1:
            storage = tensor.new_empty(
                (*tensor.shape[:-2], tensor.shape[-2] * 2, tensor.shape[-1])
            )
            inputs[name] = storage[..., ::2, :].copy_(tensor)
        elif variant == "permuted" and name in ("query", "kv_indices"):
            inputs[name] = tensor.transpose(-1, -2).contiguous().transpose(-1, -2)
    if variant == "wide":
        monkeypatch.setattr(cute_grouped, "requires_int64_abi", lambda *args: True)
    if variant == "singleton-wide":
        query = inputs["query"]
        inputs["query"] = query.as_strided(query.shape, (2**33, *query.stride()[1:]))
    if variant == "index32":
        inputs["kv_indices"] = inputs["kv_indices"].int()
    if variant == "no-local":
        inputs["local_kv"] = inputs["local_v"] = None
        inputs["attention_sink"] = None
    _check(inputs, 0 if variant == "no-local" else 7)
    assert vectorized == [variant != "misaligned"] * 2


@pytest.mark.parametrize("sink", [False, True])
def test_empty_and_invalid_selections(sink):
    inputs = _inputs(topk=67, sparse_tokens=0)
    if not sink:
        inputs["attention_sink"] = None
    _check(inputs, 0)
    inputs = _inputs(topk=67)
    inputs["kv_indices"].fill_(2**40)
    inputs["kv_indices"][:, 0] = -1
    if not sink:
        inputs["attention_sink"] = None
    _check(inputs, 0)


def test_active_offset_beyond_int32():
    free, _ = torch.cuda.mem_get_info()
    if free < 6 * 1024**3:
        pytest.skip("needs 4 GiB of sparse-stride storage plus execution headroom")
    inputs = _inputs(tokens=3, topk=5, batch=2)
    compact = inputs["query"]
    batch_stride = 2**31 + 32
    storage = compact.new_empty(batch_stride + compact[0].numel())
    inputs["query"] = storage.as_strided(
        compact.shape, (batch_stride, *compact.stride()[1:])
    ).copy_(compact)
    _check(inputs, 3)


@pytest.mark.parametrize("window", [0, 7])
def test_qsa_one_pool_blocks(window):
    inputs = _inputs(heads=24, kv_heads=2, dim=256, tokens=65, sparse_tokens=65, topk=19)
    inputs["local_kv"] = inputs["sparse_kv"]
    inputs["local_v"] = inputs["sparse_v"]
    offsets = torch.tensor([0, 33, 65], device="cuda", dtype=torch.int32)
    inputs["cu_seqlens"] = inputs["cu_seqlens_k"] = offsets
    # Four complete causal blocks and up to three trailing tokens.
    for begin, end in ((0, 33), (33, 65)):
        for token in range(end - begin):
            full = token // 4 * 4
            slots = torch.arange(full - 16, full + 3, device="cuda")
            inputs["kv_indices"][0, begin + token] = torch.where(
                (slots >= 0) & (slots <= token), slots, -1
            )
    _check(inputs, window)


def test_dynamic_kernel_reuse():
    from attn_gym.sparse.gather_attn.impl.cute_grouped import compile_grouped

    compile_grouped.cache_clear()
    for tokens, topk, documents, window in (
        (1, 1, 1, 1),
        (16, 16, 2, 7),
        (19, 31, 3, 9),
        (31, 32, 4, 3),
        (32, 33, 5, 11),
        (33, 65, 6, 19),
    ):
        inputs = _inputs(tokens=tokens, sparse_tokens=tokens + 7, topk=topk)
        inputs["cu_seqlens"] = torch.linspace(
            0, tokens, documents + 1, device="cuda", dtype=torch.int32
        )
        inputs["cu_seqlens_k"] = torch.linspace(
            0, tokens + 7, documents + 1, device="cuda", dtype=torch.int32
        )
        _check(inputs, window)
        assert compile_grouped.cache_info().currsize == 2  # forward + backward
    assert compile_grouped.cache_info().hits + compile_grouped.cache_info().misses == 12


@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.usefixtures("fresh_compile_cache")
def test_public_fullgraph_dynamic(separate):
    from torch._dynamo.testing import CompileCounterWithBackend

    counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(gather_attn, fullgraph=True, dynamic=True, backend=counter)
    for tokens in (17, 32):
        inputs = _inputs(tokens=tokens, separate=separate)
        grad = torch.randn_like(inputs["query"])
        actual = _run(inputs, 7, grad, compiled, kernel_options={"backend": "cute"})
        expected = _run(inputs, 7, grad, kernel_options={"backend": "cute"})
        torch.testing.assert_close(actual, expected)
        if tokens == 17:
            frames = counter.frame_count
        else:
            assert counter.frame_count == frames


@pytest.mark.parametrize("layout", ["compact", "permuted"])
def test_opcheck(layout):
    from attn_gym.sparse.gather_attn.impl.ops import _gather_attn_bwd_op, _gather_attn_fwd_op

    inputs = _inputs(tokens=5, topk=5)
    if layout == "permuted":
        for name, tensor in list(inputs.items()):
            if tensor.ndim > 1:
                # Dense permuted tensors, with feature-contiguous KV and a noncontiguous Q.
                axes = (-2, -1) if name in ("query", "kv_indices") else (1, 2)
                inputs[name] = tensor.transpose(*axes).contiguous().transpose(*axes)
    q, sk, lk = (inputs[name] for name in ("query", "sparse_kv", "local_kv"))
    idx, sink = inputs["kv_indices"], inputs["attention_sink"]
    tail = ("cute", True, None, None, None, inputs["sparse_v"], inputs["local_v"])
    forward_args = (q, sk, lk, idx, sink, None, None, 3, False, 0.25, True, *tail)
    torch.library.opcheck(_gather_attn_fwd_op, forward_args)
    output, lse, state = _gather_attn_fwd_op(*forward_args)
    assert state == []
    torch.library.opcheck(
        _gather_attn_bwd_op,
        (
            q,
            sk,
            lk,
            idx,
            state,
            sink,
            None,
            None,
            output,
            lse,
            torch.randn_like(output),
            3,
            False,
            0.25,
            *tail,
        ),
    )


def test_cuda_graph_replays_changed_inputs():
    inputs = _inputs(tokens=17, topk=67)
    grad = torch.randn_like(inputs["query"])

    def run():
        return _run(inputs, 7, grad, kernel_options={"backend": "cute"})

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = run()
        with torch.no_grad():
            inputs["query"].mul_(0.5)
            inputs["sparse_v"].add_(0.25)
            inputs["kv_indices"][..., ::3] = -1
        graph.replay()
        expected = run()
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(captured, expected)


def test_deterministic_backward_rejected():
    inputs = _inputs()
    leaves = _leaves(inputs)
    output = gather_attn(**inputs, sliding_window_size=7, kernel_options={"backend": "cute"})
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with pytest.raises(RuntimeError, match="does not support deterministic backward"):
            torch.autograd.grad(output.sum(), leaves)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def test_zero_width_rope_and_probability_option():
    inputs = _inputs(separate=False)
    ropes = []
    for name in ("query", "local_kv", "sparse_kv"):
        tensor = inputs[name].requires_grad_()
        rope = tensor.new_empty((*tensor.shape[:-1], 0), requires_grad=True)
        inputs[name] = (tensor, rope)
        ropes.append(rope)
    output = gather_attn(
        **inputs,
        sliding_window_size=7,
        scale=0.25,
        kernel_options={"backend": "cute", "bwd_recompute_p": False},
    )
    expected = gather_attn(**inputs, sliding_window_size=7, scale=0.25, impl="reference")
    torch.testing.assert_close(output, expected, atol=0.02, rtol=0.02)
    grads = torch.autograd.grad(output.sum(), ropes)
    for grad, rope in zip(grads, ropes, strict=True):
        assert grad.shape == rope.shape and grad.numel() == 0


def test_inactive_capacity_and_unselected_nan_rows():
    inputs = _inputs(tokens=19, sparse_tokens=23, topk=67)
    # Only rows 0..4 are selected, and only queries 0..16 belong to a document.
    inputs["kv_indices"].remainder_(5)
    inputs["cu_seqlens"] = torch.tensor([0, 0, 17], device="cuda", dtype=torch.int32)
    inputs["cu_seqlens_k"] = torch.tensor([0, 0, 23], device="cuda", dtype=torch.int32)
    for name in ("sparse_kv", "sparse_v"):
        inputs[name][:, :, 5:] = float("nan")
    for name in ("query", "local_kv", "local_v"):
        inputs[name][:, :, 17:] = float("nan")
    leaves = _leaves(inputs)
    output = gather_attn(**inputs, sliding_window_size=0, kernel_options={"backend": "cute"})
    assert torch.isfinite(output[:, :, :17]).all()
    grad = torch.zeros_like(output)
    grad[:, :, :17] = torch.randn_like(grad[:, :, :17])
    gradients = torch.autograd.grad(output, leaves, grad)
    for got in gradients:
        assert torch.isfinite(got).all()
    # Compare the defined prefix and all gradients to a finite-capacity reference.
    finite = {
        name: tensor.detach().nan_to_num().double().requires_grad_()
        if tensor.is_floating_point()
        else tensor
        for name, tensor in inputs.items()
    }
    expected = gather_attn(**finite, sliding_window_size=0, impl="reference")
    expected_grads = torch.autograd.grad(expected, _leaves(finite), grad.double())
    assert_numerics(output[:, :, :17], expected[:, :, :17], "active output")
    for got, want in zip(gradients, expected_grads, strict=True):
        assert_numerics(got, want, "capacity gradient")


def test_no_legacy_launches(monkeypatch):
    from attn_gym.sparse.gather_attn.impl import cute
    from attn_gym.sparse.gather_attn.impl.triton import backward, forward

    def forbidden(*args, **kwargs):
        raise AssertionError("native CuTe grouped call entered another backend")

    monkeypatch.setattr(cute, "_launch_forward", forbidden)
    monkeypatch.setattr(cute, "_launch_backward", forbidden)
    monkeypatch.setattr(forward, "_launch_forward", forbidden)
    monkeypatch.setattr(backward, "_launch_backward", forbidden)
    _check(_inputs(), 7)
