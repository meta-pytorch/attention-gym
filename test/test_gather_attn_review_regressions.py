"""Empty selections, packed-document isolation, and native FA4 query layouts."""

from dataclasses import replace
from functools import wraps

import pytest
import torch

from attn_gym.sparse import AuxRequest, RopePair, gather_attn
from attn_gym.testing.gather_attn import (
    as_fp64,
    assert_numerics,
    latent,
    make_inputs,
    skip_unsupported,
    tensor_leaves,
)

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.version.hip is not None,
        reason="NVIDIA CUDA required",
    ),
    pytest.mark.usefixtures("gather_attn_single_config", "fresh_compile_cache"),
]


@pytest.mark.parametrize("heads", [1, 8])
@pytest.mark.parametrize("with_sink", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
def test_empty_paired_selection_has_zero_gradients(heads, with_sink, deterministic):
    inputs = make_inputs(
        heads=heads, seq_len=9, topk=0, local=False, with_docs=False, with_sink=with_sink
    )
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(deterministic)
        for operation in (gather_attn, torch.compile(gather_attn, fullgraph=True)):
            out, aux = operation(
                **inputs.kwargs(),
                sliding_window_size=0,
                scale=0.0625,
                return_aux=AuxRequest(lse=True),
            )
            torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)
            grads = torch.autograd.grad(out, inputs.differentiable(), torch.randn_like(out))
            for grad in grads:
                torch.testing.assert_close(grad, torch.zeros_like(grad), atol=0, rtol=0)
            if with_sink:
                expected_lse = inputs.attention_sink[None, :, None].expand_as(aux.lse)
                torch.testing.assert_close(aux.lse, expected_lse, atol=0, rtol=0)
            else:
                assert torch.isneginf(aux.lse).all()
            assert not aux.lse.requires_grad
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


@pytest.mark.parametrize("backend", ["triton", "cute"])
def test_empty_paired_operator_contract(backend):
    from attn_gym.sparse.gather_attn.impl.ops import _gather_attn_bwd_op, _gather_attn_fwd_op

    inputs = make_inputs(heads=1, seq_len=3, topk=0, local=False, with_docs=False)
    q, qr = (x.detach() for x in inputs.query)
    kv, kr = (x.detach() for x in inputs.sparse_kv)
    sink = inputs.attention_sink.detach()
    args = (q, kv, None, inputs.kv_indices, sink, None, None, 0, True, 0.0625)
    ropes = {"query_rope": qr, "sparse_rope": kr}
    for training in (False, True):
        torch.library.opcheck(_gather_attn_fwd_op, (*args, training, backend), ropes)
    out, lse, state = _gather_attn_fwd_op(*args, True, backend, **ropes)
    torch.library.opcheck(
        _gather_attn_bwd_op,
        (*args[:4], state, *args[4:7], out, lse, torch.randn_like(out), *args[7:], backend),
        ropes,
    )


@pytest.mark.parametrize("rope_dim,component", [(0, "latent"), (64, "latent"), (64, "rope")])
def test_local_nan_cannot_cross_document_boundary(rope_dim, component):
    skip_unsupported("triton")
    inputs = make_inputs(
        heads=16, latent_dim=128, rope_dim=rope_dim, seq_len=4, topk=0, with_docs=False
    )
    offsets = torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32)
    inputs = replace(inputs, cu_seqlens=offsets, cu_seqlens_k=offsets.clone())
    ref_inputs = as_fp64(inputs)
    with torch.no_grad():
        poisoned = latent(inputs.local_kv) if component == "latent" else inputs.local_kv.rope
        poisoned[:, :, :2].fill_(torch.nan)
    kwargs = {"sliding_window_size": 3, "scale": 0.0625}
    expected = gather_attn(**ref_inputs.kwargs(), **kwargs, impl="reference")
    actual = gather_attn(**inputs.kwargs(), **kwargs, kernel_options={"backend": "triton"})
    grad = torch.randn_like(actual[:, :, 2:])
    actual_grads = torch.autograd.grad(
        actual[:, :, 2:], tensor_leaves((inputs.query, inputs.local_kv)), grad
    )
    expected_grads = torch.autograd.grad(
        expected[:, :, 2:], tensor_leaves((ref_inputs.query, ref_inputs.local_kv)), grad.double()
    )
    assert_numerics(actual[:, :, 2:], expected[:, :, 2:], "unaffected document output")
    names = (
        ("query latent", "query rope", "local latent", "local rope")
        if rope_dim
        else ("query", "local")
    )
    for name, got, ref in zip(names, actual_grads, expected_grads, strict=True):
        assert_numerics(got[:, :, 2:], ref[:, :, 2:], f"unaffected document {name} gradient")


@pytest.mark.parametrize("rope_dim", [0, 64], ids=["sequence-major", "paired-prefix-view"])
def test_cute_preserves_aligned_query_storage(monkeypatch, rope_dim):
    skip_unsupported("cute")
    import flash_attn.cute.interface as fa4

    inputs = make_inputs(heads=64, seq_len=17, topk=8, rope_dim=rope_dim, local=False)
    storage = torch.randn(
        1, 17, 64, 512 + rope_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    q = storage[..., :512].permute(0, 2, 1, 3)
    query = q if rope_dim == 0 else RopePair(q, storage[..., 512:].permute(0, 2, 1, 3))
    inputs = replace(inputs, query=query)
    ref_inputs = as_fp64(inputs)
    expected_pointer = q.data_ptr()
    launches = []
    original = fa4._flash_attn_fwd

    @wraps(original)
    def observe(*args, **kwargs):
        assert kwargs["qv"].data_ptr() == expected_pointer, "FA4 query was unnecessarily repacked"
        launches.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(fa4, "_flash_attn_fwd", observe)
    options = {"sliding_window_size": 0, "scale": 0.0625}
    expected = gather_attn(**ref_inputs.kwargs(), **options, impl="reference")
    grad = torch.randn_like(latent(inputs.query))
    expected_grads = torch.autograd.grad(expected, ref_inputs.differentiable(), grad.double())
    expected_storage_grad = (
        torch.cat(expected_grads[:2], dim=-1) if rope_dim else expected_grads[0]
    ).permute(0, 2, 1, 3)
    for operation in (gather_attn, torch.compile(gather_attn, fullgraph=True)):
        before = len(launches)
        actual = operation(**inputs.kwargs(), **options, kernel_options={"backend": "cute"})
        grads = torch.autograd.grad(actual, (storage, *inputs.differentiable()), grad)
        assert len(launches) > before
        assert_numerics(actual, expected, "output")
        assert_numerics(grads[0], expected_storage_grad, "shared-storage query gradient")
        for i, (got, ref) in enumerate(zip(grads[1:], expected_grads, strict=True)):
            assert_numerics(got, ref, f"gradient {i}")
