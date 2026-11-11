"""FA4 must remain the selected training backend inside a compiled gather_attn call."""

from collections import Counter
from functools import wraps

import pytest
import torch
from torch.utils._pytree import tree_map

from attn_gym.sparse.gather_attn import AuxRequest, gather_attn
from attn_gym.testing.gather_attn import (
    as_fp64,
    assert_numerics,
    latent,
    make_inputs,
    skip_unsupported,
    tensor_leaves,
)

pytestmark = pytest.mark.usefixtures("fresh_compile_cache")


@pytest.fixture
def fa4_launches(monkeypatch):
    """Observe real FA4 launches, not just metadata that would qualify for that backend."""
    skip_unsupported("cute")
    import flash_attn.cute.interface as fa4

    counts = Counter()
    for name in ("_flash_attn_fwd", "_flash_attn_bwd_sparse_mla"):
        original = getattr(fa4, name)

        def observe(fn, label):
            @wraps(fn)
            def call(*args, **kwargs):
                counts[label] += 1
                counts[(label, "capturing")] += int(torch.cuda.is_current_stream_capturing())
                return fn(*args, **kwargs)

            return call

        monkeypatch.setattr(fa4, name, observe(original, name))
    return counts


@pytest.mark.parametrize(
    "rope_dim,packed,with_sink,window,recompute,backend",
    [
        pytest.param(64, True, False, 0, True, None, id="glm-auto"),
        pytest.param(64, False, True, 0, False, "cute", id="glm-batched-saved-p"),
        pytest.param(0, True, True, 4, True, None, id="dsv4-auto"),
        pytest.param(0, False, False, 4, False, "cute", id="dsv4-batched-saved-p"),
    ],
)
def test_compiled_training_uses_fa4(
    fa4_launches, rope_dim, packed, with_sink, window, recompute, backend
):
    from torch._dynamo.testing import CompileCounterWithBackend

    counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(gather_attn, fullgraph=True, dynamic=True, backend=counter)
    for tokens in (33, 49):
        inputs = make_inputs(
            heads=64,
            latent_dim=512,
            rope_dim=rope_dim,
            seq_len=tokens,
            topk=16,
            with_docs=packed,
            with_sink=with_sink,
            batch=1 if packed else 2,
            local=window != 0,
        )
        options = {"bwd_recompute_p": recompute}
        if backend is not None:
            options["backend"] = backend
        kwargs = {"sliding_window_size": window, "scale": 0.0625}
        ref_inputs = as_fp64(inputs)
        expected, ref_aux = gather_attn(
            **ref_inputs.kwargs(), **kwargs, impl="reference", return_aux=AuxRequest(lse=True)
        )
        before = fa4_launches.copy()
        actual, aux = compiled(
            **inputs.kwargs(), **kwargs, kernel_options=options, return_aux=AuxRequest(lse=True)
        )
        grad = torch.randn_like(actual)
        leaves = inputs.differentiable()
        ref_leaves = ref_inputs.differentiable()
        actual_grads = torch.autograd.grad(actual, leaves, grad)
        expected_grads = torch.autograd.grad(expected, ref_leaves, grad.double())
        assert fa4_launches["_flash_attn_fwd"] > before["_flash_attn_fwd"]
        assert fa4_launches["_flash_attn_bwd_sparse_mla"] > before["_flash_attn_bwd_sparse_mla"]
        assert_numerics(actual, expected, "output")
        for i, (got, ref) in enumerate(zip(actual_grads, expected_grads, strict=True)):
            assert_numerics(got, ref, f"gradient {i}")
        assert not aux.lse.requires_grad
        torch.testing.assert_close(aux.lse.double(), ref_aux.lse, rtol=1e-5, atol=1e-5)
        if tokens == 33:
            # Some torch versions first emit an empty initialization graph. Neither it nor
            # the attention graph may recompile when token/document counts change.
            initial_frames = counter.frame_count
            assert initial_frames > 0
        else:
            assert counter.frame_count == initial_frames


@pytest.mark.parametrize("window", [0, 7])
def test_compiled_aliases_use_fa4(fa4_launches, window):
    """One KV pool and one offsets tensor may fill both roles, as model callers do."""
    inputs = make_inputs(
        heads=64, latent_dim=512, rope_dim=64, seq_len=33, topk=16, with_docs=True, with_sink=False
    )
    q, kv, indices, cu = inputs.query, inputs.sparse_kv, inputs.kv_indices, inputs.cu_seqlens

    def operation(q, kv, indices, cu):
        return gather_attn(
            q,
            kv,
            kv,
            indices,
            cu_seqlens=cu,
            cu_seqlens_k=cu,
            sliding_window_size=window,
            scale=0.0625,
        )

    ref_inputs = as_fp64(inputs)
    # Both candidate pools refer to the same values, but remain independent oracle leaves.
    with torch.no_grad():
        for local, sparse in zip(ref_inputs.local_kv, ref_inputs.sparse_kv, strict=True):
            local.copy_(sparse)
    expected = gather_attn(
        **ref_inputs.kwargs(), sliding_window_size=window, scale=0.0625, impl="reference"
    )
    grad = torch.randn_like(latent(q))
    ref_grads = torch.autograd.grad(expected, ref_inputs.differentiable(), grad.double())
    expected_grads = (*ref_grads[:2], ref_grads[2] + ref_grads[4], ref_grads[3] + ref_grads[5])
    before = fa4_launches.copy()
    actual = torch.compile(operation, fullgraph=True)(q, kv, indices, cu)
    actual_grads = torch.autograd.grad(actual, tensor_leaves((q, kv)), grad)
    assert fa4_launches["_flash_attn_fwd"] > before["_flash_attn_fwd"]
    assert fa4_launches["_flash_attn_bwd_sparse_mla"] > before["_flash_attn_bwd_sparse_mla"]
    assert_numerics(actual, expected, "aliased output")
    for i, (got, ref) in enumerate(zip(actual_grads, expected_grads, strict=True)):
        assert_numerics(got, ref, f"aliased gradient {i}")


@pytest.mark.parametrize(
    "rope_dim,packed,recompute,with_local",
    [(64, True, True, True), (64, True, True, False), (0, False, False, True)],
)
@pytest.mark.parametrize("permuted", [False, True], ids=["compact", "dense-permuted"])
def test_fa4_operator_contracts(fa4_launches, rope_dim, packed, recompute, with_local, permuted):
    """Real allocations and fake metadata agree for both training and inference tapes."""
    from attn_gym.sparse.gather_attn.impl.ops import _gather_attn_bwd_op, _gather_attn_fwd_op

    inputs = make_inputs(
        heads=64,
        latent_dim=512,
        rope_dim=rope_dim,
        seq_len=33,
        topk=16,
        with_docs=packed,
        with_sink=True,
        batch=1 if packed else 2,
        local=with_local,
    )

    def prepare(tensor):
        if tensor is None:
            return None
        tensor = tensor.detach()
        if permuted and tensor.ndim >= 2:
            tensor = tensor.transpose(-1, -2).contiguous().transpose(-1, -2)
        return tensor

    tensors = tree_map(prepare, inputs.kwargs())
    q, local, sparse = (tensors[name] for name in ("query", "local_kv", "sparse_kv"))
    indices, sink = tensors["kv_indices"], tensors["attention_sink"]
    cu, cuk = tensors["cu_seqlens"], tensors["cu_seqlens_k"]
    ropes = tuple(
        None if x is None or isinstance(x, torch.Tensor) else x.rope for x in (q, sparse, local)
    )
    q, local, sparse = (None if x is None else latent(x) for x in (q, local, sparse))
    window = 0 if rope_dim else 4
    args = (q, sparse, local, indices, sink, cu, cuk, window, True, 0.0625)
    for needs_backward in (False, True):
        torch.library.opcheck(
            _gather_attn_fwd_op,
            (*args, needs_backward, "cute", recompute, *ropes),
            {"autotune": True},
        )
    out, lse, tape = _gather_attn_fwd_op(*args, True, "cute", recompute, *ropes, autotune=True)
    grad = torch.randn_like(out).transpose(-1, -2).contiguous().transpose(-1, -2)
    torch.library.opcheck(
        _gather_attn_bwd_op,
        (
            q,
            sparse,
            local,
            indices,
            tape,
            sink,
            cu,
            cuk,
            out,
            lse,
            grad,
            window,
            True,
            0.0625,
            "cute",
            recompute,
            *ropes,
        ),
        {"autotune": True},
    )


@pytest.mark.parametrize("recompute", [True, False], ids=["recompute-p", "save-p"])
def test_fa4_cuda_graph_training_replay(fa4_launches, recompute):
    """Replay must refresh saved tensor buffers from changed selections."""
    inputs = make_inputs(
        heads=64,
        latent_dim=512,
        rope_dim=64,
        seq_len=33,
        topk=16,
        with_docs=True,
        with_sink=True,
    )
    leaves = inputs.differentiable()
    grad = torch.randn_like(latent(inputs.query))

    def step():
        out = gather_attn(
            **inputs.kwargs(),
            sliding_window_size=0,
            scale=0.0625,
            kernel_options={"backend": "cute", "bwd_recompute_p": recompute},
        )
        return out, *torch.autograd.grad(out, leaves, grad)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = step()
    torch.cuda.synchronize()
    with torch.no_grad():
        for tensor in tensor_leaves(inputs.query):
            tensor.mul_(0.7)
        for tensor in tensor_leaves(inputs.sparse_kv):
            tensor.add_(0.1)
        # Change selection membership while retaining valid document-local positions.
        inputs.kv_indices.fill_(0)
    expected = step()
    for tensor in captured:
        tensor.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    for got, ref in zip(captured, expected, strict=True):
        torch.testing.assert_close(got, ref)


def test_fa4_failure_does_not_retry_triton(fa4_launches, monkeypatch):
    import flash_attn.cute.interface as fa4

    from attn_gym.sparse.gather_attn.impl.triton import forward

    def failure(*args, **kwargs):
        raise RuntimeError("intentional FA4 launch failure")

    def unexpected(*args, **kwargs):
        raise AssertionError("FA4 failure retried Triton")

    monkeypatch.setattr(fa4, "_flash_attn_fwd", failure)
    monkeypatch.setattr(forward, "_launch_forward", unexpected)
    inputs = make_inputs(
        heads=64,
        latent_dim=512,
        rope_dim=64,
        seq_len=33,
        topk=16,
        with_docs=True,
        with_sink=False,
    )
    with pytest.raises(RuntimeError, match="intentional FA4 launch failure"):
        torch.compile(gather_attn, fullgraph=True)(
            **inputs.kwargs(),
            sliding_window_size=0,
            scale=0.0625,
        )


@pytest.mark.parametrize("warn_only", [False, True], ids=["strict", "warn-only"])
def test_compiled_explicit_fa4_determinism(fa4_launches, warn_only):
    inputs = make_inputs(
        heads=64,
        latent_dim=512,
        rope_dim=64,
        seq_len=33,
        topk=16,
        with_docs=True,
        with_sink=False,
    )
    compiled = torch.compile(gather_attn, fullgraph=True)
    enabled = torch.are_deterministic_algorithms_enabled()
    old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        # PyTorch itself disallows toggling deterministic mode between compiled forward and
        # backward. Set it before capture; the opaque backward must enforce FA4's contract.
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        out = compiled(
            **inputs.kwargs(),
            sliding_window_size=0,
            scale=0.0625,
            kernel_options={"backend": "cute"},
        )
        before = fa4_launches.copy()
        if warn_only:
            with pytest.warns(UserWarning, match="does not support deterministic backward"):
                out.sum().backward()
            assert (
                fa4_launches["_flash_attn_bwd_sparse_mla"] > before["_flash_attn_bwd_sparse_mla"]
            )
        else:
            with pytest.raises(RuntimeError, match="does not support deterministic backward"):
                out.sum().backward()
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=old_warn)


def test_reduce_overhead_captures_fa4_forward_and_backward(fa4_launches):
    """Successful compilation is insufficient: Inductor must capture both opaque launches."""
    inputs = make_inputs(
        heads=64,
        latent_dim=512,
        rope_dim=64,
        seq_len=33,
        topk=16,
        with_docs=True,
        with_sink=False,
    )
    q, kv, idx, cu = inputs.query, inputs.sparse_kv, inputs.kv_indices, inputs.cu_seqlens
    grad = torch.randn_like(latent(q))

    def operation(q, kv, idx, cu):
        return gather_attn(
            q,
            kv,
            kv,
            idx,
            cu_seqlens=cu,
            cu_seqlens_k=cu,
            sliding_window_size=0,
            scale=0.0625,
        )

    compiled = torch.compile(operation, fullgraph=True, mode="reduce-overhead")
    for _ in range(4):
        torch.compiler.cudagraph_mark_step_begin()
        out = compiled(q, kv, idx, cu)
        grads = torch.autograd.grad(out, tensor_leaves((q, kv)), grad)
        del out, grads
    for name in ("_flash_attn_fwd", "_flash_attn_bwd_sparse_mla"):
        assert fa4_launches[(name, "capturing")] > 0, f"{name} was never CUDA-graph captured"
    launches_before = fa4_launches.copy()
    with torch.no_grad():
        for tensor in tensor_leaves(q):
            tensor.mul_(0.7)
        idx.fill_(0)
    torch.compiler.cudagraph_mark_step_begin()
    out = compiled(q, kv, idx, cu)
    grads = torch.autograd.grad(out, tensor_leaves((q, kv)), grad)
    # Replay launches GPU kernels without reentering the Python FA4 functions.
    assert fa4_launches == launches_before
    expected = operation(q, kv, idx, cu)
    expected_grads = torch.autograd.grad(expected, tensor_leaves((q, kv)), grad)
    torch.testing.assert_close(out, expected)
    for got, ref in zip(grads, expected_grads, strict=True):
        torch.testing.assert_close(got, ref)
