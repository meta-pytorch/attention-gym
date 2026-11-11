"""Sparse MLA with explicit latent/RoPE inputs and a shared latent key/value."""

import pytest
import torch
from torch.utils._pytree import tree_map

from attn_gym.sparse import AuxRequest, RopePair, gather_attn
from attn_gym.testing.gather_attn import (
    AttentionInputs,
    as_fp64,
    assert_numerics,
    latent,
    make_inputs,
    skip_unsupported,
)

pytestmark = pytest.mark.usefixtures("gather_attn_single_config")


@pytest.mark.parametrize(
    "backend,heads,dim,rope,local,window,packed,batch,saved",
    [
        ("triton", 16, 128, 32, False, 0, True, 1, False),
        ("triton", 17, 96, 16, True, 7, True, 1, False),
        ("triton", 64, 512, 64, False, 0, False, 2, False),
        ("triton", 16, 512, 128, True, 7, True, 1, False),
        ("triton", 32, 128, 64, True, 7, False, 2, False),
        ("triton", 128, 128, 64, True, 7, True, 1, False),
        ("cute", 64, 512, 64, False, 0, True, 1, False),
        ("cute", 64, 512, 64, True, 7, True, 1, False),
        ("cute", 128, 512, 64, True, 7, False, 2, True),
        ("cute", 16, 512, 64, False, 0, True, 1, True),
    ],
)
def test_paired_training(backend, heads, dim, rope, local, window, packed, batch, saved):
    skip_unsupported(backend)
    inputs = make_inputs(
        heads=heads, latent_dim=dim, rope_dim=rope, local=local, with_docs=packed, batch=batch
    )
    ref_inputs = as_fp64(inputs)
    kwargs = {"sliding_window_size": window, "scale": 0.0625, "return_aux": AuxRequest(lse=True)}
    ref, ref_aux = gather_attn(**ref_inputs.kwargs(), **kwargs, impl="reference")
    out, aux = gather_attn(
        **inputs.kwargs(),
        **kwargs,
        kernel_options={"backend": backend, "bwd_recompute_p": not saved},
    )
    assert out.shape == latent(inputs.query).shape
    grad = torch.randn(
        *out.shape[:-2], out.shape[-1], out.shape[-2], device=out.device, dtype=out.dtype
    ).transpose(-1, -2)
    ref_grads = torch.autograd.grad(ref, ref_inputs.differentiable(), grad.double())
    grads = torch.autograd.grad(out, inputs.differentiable(), grad)
    assert_numerics(out, ref, "output")
    for i, (got, expected) in enumerate(zip(grads, ref_grads, strict=True)):
        assert_numerics(got, expected, f"gradient {i}")
    torch.testing.assert_close(aux.lse.double(), ref_aux.lse, atol=1e-4, rtol=0)
    assert not aux.lse.requires_grad


def test_plain_tuple_and_named_tuple_are_equivalent():
    inputs = make_inputs(
        heads=2, latent_dim=8, rope_dim=4, seq_len=9, topk=3, device="cpu", dtype=torch.float64
    )
    plain = AttentionInputs(
        **{name: tuple(x) if isinstance(x, RopePair) else x for name, x in inputs.kwargs().items()}
    )
    kwargs = {"sliding_window_size": 3, "scale": 0.2, "impl": "reference"}
    a = gather_attn(**inputs.kwargs(), **kwargs)
    b = gather_attn(**plain.kwargs(), **kwargs)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    grad = torch.randn_like(a)
    ga = torch.autograd.grad(a, inputs.differentiable(), grad)
    gb = torch.autograd.grad(b, plain.differentiable(), grad)
    for x, y in zip(ga, gb, strict=True):
        torch.testing.assert_close(x, y, atol=0, rtol=0)


@pytest.mark.parametrize("backend", ["triton", "cute"])
def test_components_can_be_views_of_shared_storage(backend):
    skip_unsupported(backend)
    torch.manual_seed(912)
    q = torch.randn(1, 64, 33, 576, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    kv = torch.randn(1, 1, 33, 576, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    indices = torch.arange(16, device="cuda", dtype=torch.int32)[None, None].expand(1, 33, -1)

    def parts(x):
        return RopePair(x[..., :512], x[..., 512:])

    qr, kr = q.detach().double().requires_grad_(), kv.detach().double().requires_grad_()
    opts = {"sliding_window_size": 0, "scale": 0.0625}
    qparts, kvparts = parts(q), parts(kv)
    out = gather_attn(qparts, None, kvparts, indices, **opts, kernel_options={"backend": backend})
    ref_q, ref_kv = parts(qr), parts(kr)
    expected = gather_attn(ref_q, None, ref_kv, indices, **opts, impl="reference")
    go = torch.randn_like(out)
    gs = torch.autograd.grad(out, (q, kv, *qparts, *kvparts), go)
    rs = torch.autograd.grad(expected, (qr, kr, *ref_q, *ref_kv), go.double())
    for i, (got, ref) in enumerate(zip(gs, rs, strict=True)):
        assert_numerics(got, ref, f"shared-storage gradient {i}")


def test_value_dim_is_not_a_public_argument():
    x = torch.randn(1, 1, 3, 8)
    idx = torch.zeros(1, 3, 1, dtype=torch.int32)
    with pytest.raises(TypeError, match="value_dim"):
        gather_attn(x, x, x, idx, impl="reference", value_dim=4)


@pytest.mark.parametrize("pair", [(torch.empty(1),), (torch.empty(1), None), (1, 2)])
def test_invalid_pair_is_rejected(pair):
    x = torch.randn(1, 1, 3, 8)
    idx = torch.zeros(1, 3, 1, dtype=torch.int32)
    with pytest.raises((TypeError, ValueError)):
        gather_attn(pair, x, x, idx, impl="reference", scale=0.2)


def test_paired_query_requires_model_scale():
    inputs = make_inputs(
        heads=2, latent_dim=8, rope_dim=4, seq_len=9, topk=3, device="cpu", dtype=torch.float64
    )
    with pytest.raises(ValueError, match="scale"):
        gather_attn(**inputs.kwargs(), sliding_window_size=0, impl="reference")


def test_local_none_requires_zero_window():
    x = torch.randn(1, 1, 3, 8)
    idx = torch.zeros(1, 3, 1, dtype=torch.int32)
    with pytest.raises(ValueError, match="local"):
        gather_attn(x, None, x, idx, sliding_window_size=1, impl="reference")


def test_reference_matches_joint_score_and_latent_value_oracle():
    """A small direct oracle scores joint Q/K but sums only the latent values."""
    inputs = make_inputs(
        heads=2, latent_dim=8, rope_dim=4, seq_len=9, topk=3, device="cpu", dtype=torch.float64
    )
    q, local, sparse = inputs.query, inputs.local_kv, inputs.sparse_kv
    boundaries = inputs.cu_seqlens.tolist()
    heads = []
    for h in range(2):
        rows = []
        for t in range(9):
            doc = next(
                i for i in range(len(boundaries) - 1) if boundaries[i] <= t < boundaries[i + 1]
            )
            begin, end = boundaries[doc : doc + 2]
            selected = [(local, j) for j in range(max(begin, t - 2), t + 1)]
            selected += [
                (sparse, begin + j)
                for j in inputs.kv_indices[0, t].tolist()
                if 0 <= j < end - begin
            ]
            keys = torch.stack(
                [torch.cat((pool.latent[0, 0, j], pool.rope[0, 0, j])) for pool, j in selected]
            )
            values = torch.stack([pool.latent[0, 0, j] for pool, j in selected])
            joint_q = torch.cat((q.latent[0, h, t], q.rope[0, h, t]))
            scores = torch.cat((keys @ joint_q * 0.2, inputs.attention_sink[h : h + 1].double()))
            rows.append(scores.softmax(0)[:-1] @ values)
        heads.append(torch.stack(rows))
    expected = torch.stack(heads)[None]
    actual = gather_attn(**inputs.kwargs(), sliding_window_size=3, scale=0.2, impl="reference")
    torch.testing.assert_close(actual, expected)
    go = torch.randn_like(actual)
    got = torch.autograd.grad(actual, inputs.differentiable(), go)
    refs = torch.autograd.grad(expected, inputs.differentiable(), go)
    for a, b in zip(got, refs, strict=True):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("backend", ["triton", "cute"])
@pytest.mark.parametrize("window", [0, 7])
def test_zero_width_rope_has_empty_gradients(backend, window):
    skip_unsupported(backend)
    inputs = make_inputs(
        heads=2 if backend == "triton" else 64,
        rope_dim=0,
        local=window != 0,
        seq_len=17,
        topk=8,
        with_docs=True,
        with_sink=False,
    )
    q, kv = inputs.query, inputs.sparse_kv
    qr = q.new_empty((*q.shape[:-1], 0)).requires_grad_()
    kr = kv.new_empty((*kv.shape[:-1], 0)).requires_grad_()
    kwargs = {
        "sliding_window_size": window,
        "scale": 0.0625,
        "cu_seqlens": inputs.cu_seqlens,
        "cu_seqlens_k": inputs.cu_seqlens_k,
        "kernel_options": {"backend": backend},
    }
    local = inputs.local_kv
    local_pair = None
    latent_inputs = (q, kv)
    rope_inputs = (qr, kr)
    if local is not None:
        lr = local.new_empty((*local.shape[:-1], 0)).requires_grad_()
        local_pair = (local, lr)
        latent_inputs += (local,)
        rope_inputs += (lr,)
    out = gather_attn((q, qr), local_pair, (kv, kr), inputs.kv_indices, **kwargs)
    expected = gather_attn(q, local, kv, inputs.kv_indices, **kwargs)
    go = torch.randn_like(out)
    grads = torch.autograd.grad(out, latent_inputs + rope_inputs, go)
    expected_grads = torch.autograd.grad(expected, latent_inputs, go)
    torch.testing.assert_close(out, expected)
    for got, ref in zip(grads[: len(latent_inputs)], expected_grads, strict=True):
        torch.testing.assert_close(got, ref)
    for grad, operand in zip(grads[len(latent_inputs) :], rope_inputs, strict=True):
        assert grad.shape == operand.shape


def test_triton_sparse_only_avoids_unused_map_and_local_staging(monkeypatch):
    skip_unsupported("triton")
    from attn_gym.sparse.gather_attn.impl.triton import backward

    def reject_map(*args):
        raise AssertionError("paired atomic backward does not use a reverse map")

    monkeypatch.setattr(backward, "_build_index_query_map", reject_map)
    inputs = make_inputs(
        heads=64, latent_dim=128, rope_dim=64, seq_len=1024, topk=16, with_docs=True, local=True
    )
    with torch.no_grad():
        for t in inputs.local_kv:
            t.fill_(float("nan"))
    out = gather_attn(
        **inputs.kwargs(),
        sliding_window_size=0,
        scale=0.0625,
        kernel_options={"backend": "triton"},
    )
    go = torch.randn_like(out)
    for _ in range(2):
        torch.autograd.grad(out, inputs.differentiable(), go, retain_graph=True)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    grads = torch.autograd.grad(out, inputs.differentiable(), go)
    torch.cuda.synchronize()
    peak_extra = torch.cuda.max_memory_allocated() - before
    budget = (
        sum(t.nbytes for t in inputs.query) + 16 * sum(t.nbytes for t in inputs.local_kv) + 1024**2
    )
    assert peak_extra < budget, (peak_extra, budget)
    assert all(torch.isfinite(g).all() for g in grads)
    # Query components are first, followed by the two unused local components.
    assert grads[2].count_nonzero() == 0 and grads[3].count_nonzero() == 0


@pytest.mark.parametrize("backend", ["triton", "cute"])
def test_paired_compressed_pool_and_empty_document(backend):
    from dataclasses import replace

    skip_unsupported(backend)
    inputs = make_inputs(heads=64, seq_len=35, topk=16, local=False, with_docs=True)
    sp = RopePair(*(t[:, :, :11].detach().requires_grad_() for t in inputs.sparse_kv))
    inputs = replace(
        inputs,
        sparse_kv=sp,
        cu_seqlens_k=torch.tensor([0, 0, 3, 7, 11], device="cuda", dtype=torch.int32),
    )
    reference = as_fp64(inputs)
    options = {"sliding_window_size": 0, "scale": 0.0625}
    expected = gather_attn(**reference.kwargs(), **options, impl="reference")
    actual = gather_attn(**inputs.kwargs(), **options, kernel_options={"backend": backend})
    go = torch.randn_like(actual)
    gs = torch.autograd.grad(actual, inputs.differentiable(), go)
    rs = torch.autograd.grad(expected, reference.differentiable(), go.double())
    assert_numerics(actual, expected, "compressed output")
    for i, (got, ref) in enumerate(zip(gs, rs, strict=True)):
        assert_numerics(got, ref, f"compressed gradient {i}")


@pytest.mark.parametrize("kind", ["mixed-form", "rope-width", "rope-heads", "rope-dtype"])
def test_paired_pool_validation(kind):
    from dataclasses import replace

    inputs = make_inputs(heads=2, latent_dim=8, rope_dim=4, seq_len=9, topk=3, device="cpu")
    k, r = inputs.sparse_kv
    if kind == "mixed-form":
        sparse = k
    elif kind == "rope-width":
        sparse = (k, r[..., :2])
    elif kind == "rope-heads":
        sparse = (k, r.expand(-1, 2, -1, -1))
    else:
        sparse = (k, r.double())
    inputs = replace(inputs, sparse_kv=sparse)
    with pytest.raises(ValueError):
        gather_attn(**inputs.kwargs(), sliding_window_size=0, scale=0.2, impl="reference")


def test_same_pair_can_fill_query_and_sparse_roles():
    skip_unsupported("cute")
    x = torch.randn(1, 1, 17, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    r = torch.randn(1, 1, 17, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    idx = torch.arange(8, device="cuda", dtype=torch.int32)[None, None].expand(1, 17, -1)
    pair = RopePair(x, r)
    xr, rr = x.detach().double().requires_grad_(), r.detach().double().requires_grad_()
    ref_pair = RopePair(xr, rr)
    opts = {"sliding_window_size": 0, "scale": 0.0625}
    out = torch.compile(gather_attn, fullgraph=True)(pair, None, pair, idx, **opts)
    expected = gather_attn(ref_pair, None, ref_pair, idx, **opts, impl="reference")
    go = torch.randn_like(out)
    gs = torch.autograd.grad(out, (x, r), go)
    rs = torch.autograd.grad(expected, (xr, rr), go.double())
    assert_numerics(out, expected, "aliased output")
    for i, (a, b) in enumerate(zip(gs, rs, strict=True)):
        assert_numerics(a, b, f"aliased gradient {i}")


def test_paired_expanded_heads_are_rejected_before_launch():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        q = torch.empty(1, 16, 17, 128, device="cuda", dtype=torch.bfloat16)
        qr = torch.empty(1, 16, 17, 64, device="cuda", dtype=q.dtype)
        kv = torch.empty(1, 1, 17, 128, device="cuda", dtype=q.dtype).expand(-1, 16, -1, -1)
        kr = torch.empty(1, 1, 17, 64, device="cuda", dtype=q.dtype).expand(-1, 16, -1, -1)
        idx = torch.empty(1, 17, 8, device="cuda", dtype=torch.int32)
        with pytest.raises(ValueError, match="single shared KV head"):
            gather_attn((q, qr), None, (kv, kr), idx, sliding_window_size=0, scale=0.2)


def triton_step(inputs, window=0, operation=gather_attn):
    out = operation(
        **inputs.kwargs(),
        sliding_window_size=window,
        scale=0.0625,
        kernel_options={"backend": "triton"},
    )
    go = torch.randn(
        out.shape,
        device=out.device,
        dtype=out.dtype,
        generator=torch.Generator(device="cuda").manual_seed(431),
    )
    return out, *torch.autograd.grad(out, inputs.differentiable(), go)


@pytest.mark.parametrize("window", [0, 7])
@pytest.mark.usefixtures("fresh_compile_cache")
def test_triton_paired_dynamic_training(window):
    skip_unsupported("triton")
    from torch._dynamo.testing import CompileCounterWithBackend

    counter = CompileCounterWithBackend("inductor")
    operation = torch.compile(gather_attn, fullgraph=True, dynamic=True, backend=counter)
    for tokens in (33, 49):
        inputs = make_inputs(
            heads=16, latent_dim=128, rope_dim=64, seq_len=tokens, local=bool(window)
        )
        expected = triton_step(inputs, window)
        actual = triton_step(inputs, window, operation)
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b)
        if tokens == 33:
            initial = counter.frame_count
        else:
            assert counter.frame_count == initial


@pytest.mark.parametrize("window", [0, 7])
def test_triton_paired_forced_int64(window, monkeypatch):
    skip_unsupported("triton")
    from attn_gym.sparse.gather_attn.impl.triton import backward, forward

    inputs = make_inputs(heads=16, latent_dim=128, rope_dim=64, local=bool(window))
    expected = triton_step(inputs, window)
    monkeypatch.setattr(forward, "requires_int64_offsets", lambda *args: True)
    monkeypatch.setattr(backward, "requires_int64_offsets", lambda *args: True)
    for a, b in zip(triton_step(inputs, window), expected, strict=True):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("window", [0, 7])
def test_triton_paired_operator_contracts(window):
    skip_unsupported("triton")
    from attn_gym.sparse.gather_attn.impl.ops import _gather_attn_bwd_op, _gather_attn_fwd_op

    inputs = make_inputs(heads=16, latent_dim=128, rope_dim=64, local=bool(window))
    args = tree_map(lambda t: t.detach() if isinstance(t, torch.Tensor) else t, inputs.kwargs())
    q, qr = args["query"]
    sp, sr = args["sparse_kv"]
    local, lr = (None, None) if args["local_kv"] is None else args["local_kv"]
    indices, sink, cu, cuk = (
        args[n] for n in ("kv_indices", "attention_sink", "cu_seqlens", "cu_seqlens_k")
    )
    fwd = (q, sp, local, indices, sink, cu, cuk, window, True, 0.0625)
    ropes = {"query_rope": qr, "sparse_rope": sr, "local_rope": lr}
    for training in (False, True):
        torch.library.opcheck(
            _gather_attn_fwd_op, (*fwd, training, "triton"), {**ropes, "autotune": True}
        )
    out, lse, state = _gather_attn_fwd_op(*fwd, True, "triton", **ropes, autotune=True)
    assert state == [None, None]
    torch.library.opcheck(
        _gather_attn_bwd_op,
        (
            q,
            sp,
            local,
            indices,
            state,
            sink,
            cu,
            cuk,
            out,
            lse,
            torch.randn_like(out),
            window,
            True,
            0.0625,
            "triton",
        ),
        {**ropes, "autotune": True},
    )


@pytest.mark.parametrize("warn_only", [False, True])
@pytest.mark.parametrize("after_forward", [False, True])
def test_triton_paired_determinism(warn_only, after_forward):
    skip_unsupported("triton")
    inputs = make_inputs(heads=16, latent_dim=128, rope_dim=64, local=False)
    enabled = torch.are_deterministic_algorithms_enabled()
    old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        if not after_forward:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
        out = gather_attn(
            **inputs.kwargs(),
            sliding_window_size=0,
            scale=0.0625,
            kernel_options={"backend": "triton"},
        )
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        context = (
            pytest.warns(UserWarning, match="no deterministic backward")
            if warn_only
            else pytest.raises(RuntimeError, match="no deterministic backward")
        )
        with context:
            gs = torch.autograd.grad(out, inputs.differentiable(), torch.randn_like(out))
        if warn_only:
            assert all(torch.isfinite(g).all() for g in gs)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=old_warn)


def test_triton_paired_reuses_shape_specializations(triton_compiles):
    skip_unsupported("triton")

    def counts(rope_dim):
        triton_compiles.clear()
        for tokens in (1, 16, 17, 32, 45):
            inputs = make_inputs(
                heads=16,
                latent_dim=128,
                rope_dim=rope_dim,
                seq_len=tokens,
                with_docs=False,
                local=False,
            )
            triton_step(inputs)
        return {name: n for name, n in triton_compiles.items() if "shared" in name}

    plain = counts(0)
    assert plain
    assert counts(64) == plain


def test_fa4_sparse_only_inactive_capacity_replay():
    from dataclasses import replace

    skip_unsupported("cute")
    inputs = make_inputs(heads=64, seq_len=35, topk=16, local=False, with_docs=True)
    cu = torch.tensor([0, 0, 7, 21, 21], device="cuda", dtype=torch.int32)
    cuk = torch.tensor([0, 0, 3, 9, 9], device="cuda", dtype=torch.int32)
    inputs = replace(inputs, cu_seqlens=cu, cu_seqlens_k=cuk)
    with torch.no_grad():
        for tensor in inputs.sparse_kv:
            tensor[:, :, 9:].fill_(float("nan"))
    go = torch.randn(1, 64, 21, 512, device="cuda", dtype=torch.bfloat16)

    def step():
        out = gather_attn(
            **inputs.kwargs(),
            sliding_window_size=0,
            scale=0.0625,
            kernel_options={"backend": "cute"},
        )
        return out, *torch.autograd.grad(out[:, :, :21], inputs.differentiable(), go)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = step()
    for q_bounds, k_bounds in (
        ([0, 0, 7, 21, 21], [0, 0, 3, 9, 9]),
        ([0, 0, 5, 13, 21], [0, 0, 1, 6, 9]),
    ):
        cu.copy_(torch.tensor(q_bounds, device="cuda", dtype=torch.int32))
        cuk.copy_(torch.tensor(k_bounds, device="cuda", dtype=torch.int32))
        graph.replay()
        torch.cuda.synchronize()
        ref = as_fp64(inputs)
        ref = replace(
            ref,
            query=RopePair(*(t[:, :, :21].detach().requires_grad_() for t in ref.query)),
            sparse_kv=RopePair(*(t[:, :, :9].detach().requires_grad_() for t in ref.sparse_kv)),
            kv_indices=inputs.kv_indices[:, :21],
        )
        expected = gather_attn(
            **ref.kwargs(), sliding_window_size=0, scale=0.0625, impl="reference"
        )
        grads = torch.autograd.grad(expected, ref.differentiable(), go.double())
        assert_numerics(captured[0][:, :, :21], expected, "active output")
        for i, (got, want) in enumerate(zip(captured[1:], grads, strict=True)):
            if i < 2:
                got = got[:, :, :21]
            elif i < 4:
                got = got[:, :, :9]
            assert_numerics(got, want, f"active gradient {i}")


@pytest.mark.parametrize("recompute", [True, False])
@pytest.mark.parametrize("paired,with_sink", [(False, False), (True, True)])
def test_fa4_zero_capacity_pool_training(paired, with_sink, recompute):
    skip_unsupported("cute")
    from attn_gym.sparse.gather_attn.impl.ops import _gather_attn_bwd_op, _gather_attn_fwd_op

    inputs = make_inputs(
        rope_dim=64 if paired else 0, local=False, with_docs=False, with_sink=with_sink
    )
    inputs = AttentionInputs(
        **{
            **inputs.kwargs(),
            "sparse_kv": tree_map(
                lambda t: t[:, :, :0].detach().requires_grad_(), inputs.sparse_kv
            ),
        }
    )
    q, qr = inputs.query if paired else (inputs.query, None)
    sp, sr = inputs.sparse_kv if paired else (inputs.sparse_kv, None)
    fwd = tree_map(
        lambda t: t.detach() if isinstance(t, torch.Tensor) else t,
        (q, sp, None, inputs.kv_indices, inputs.attention_sink, None, None, 0, True, 0.0625),
    )
    ropes = {
        "query_rope": None if qr is None else qr.detach(),
        "sparse_rope": None if sr is None else sr.detach(),
    }
    torch.library.opcheck(
        _gather_attn_fwd_op, (*fwd, True, "cute", recompute), {**ropes, "autotune": True}
    )
    out, lse, state = _gather_attn_fwd_op(*fwd, True, "cute", recompute, **ropes, autotune=True)
    torch.library.opcheck(
        _gather_attn_bwd_op,
        (*fwd[:4], state, *fwd[4:7], out, lse, torch.randn_like(out), *fwd[7:], "cute", recompute),
        {**ropes, "autotune": True},
    )
    compiled = torch.compile(gather_attn, fullgraph=True)
    out, aux = compiled(
        **inputs.kwargs(),
        sliding_window_size=0,
        scale=0.0625,
        return_aux=AuxRequest(lse=True),
        kernel_options={"backend": "cute", "bwd_recompute_p": recompute},
    )
    assert out.count_nonzero() == 0
    grads = torch.autograd.grad(out, inputs.differentiable(), torch.randn_like(out))
    assert all(g.count_nonzero() == 0 for g in grads)
    if with_sink:
        torch.testing.assert_close(
            aux.lse, inputs.attention_sink[None, :, None].expand_as(aux.lse)
        )
    else:
        assert torch.isneginf(aux.lse).all()
