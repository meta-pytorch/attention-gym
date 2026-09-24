"""Dynamic graph and kernel-reuse contracts for Triton gather attention."""

import pytest
import torch

from attn_gym.sparse.gather_attn import gather_attn


def _cuda():
    if not torch.cuda.is_available() or torch.version.hip:
        pytest.skip("NVIDIA CUDA required")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("Ampere or newer required")


def _inputs(
    tokens, candidates, *, heads=2, dim=16, share_kv=True, dtype=torch.float32, inner_stride=1
):
    torch.manual_seed(817)
    kv_heads = 1 if share_kv else heads
    q = torch.randn(1, tokens, heads, dim, device="cuda", dtype=dtype).transpose(1, 2) * 0.1
    local = torch.randn(
        1, kv_heads, tokens * 2, dim * inner_stride, device="cuda", dtype=dtype
    ).mul_(0.1)[:, :, ::2, ::inner_stride]
    sparse = torch.randn(
        1, kv_heads, candidates * 2, dim * inner_stride, device="cuda", dtype=dtype
    ).mul_(0.1)[:, :, ::2, ::inner_stride]
    indices = torch.tensor([0, 1, -1], device="cuda", dtype=torch.int32).repeat(1, tokens, 1)
    sink = torch.linspace(-0.4, 0.6, heads * 2, device="cuda", dtype=torch.float32)[::2]
    cu_q = torch.tensor([0, 5, tokens], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, 4, candidates], device="cuda", dtype=torch.int32)
    return q, local, sparse, indices, sink, cu_q, cu_k


def _forward_and_grads(tensors, indices, cu_q, cu_k, **kwargs):
    q, local, sparse, sink = tensors
    out = gather_attn(
        q,
        local,
        sparse,
        indices,
        sink,
        cu_seqlens=cu_q,
        cu_seqlens_k=cu_k,
        sliding_window_size=7,
        **kwargs,
    )
    return out, *torch.autograd.grad(out.sum(), tensors)


@pytest.mark.parametrize("share_kv", [False, True])
def test_triton_operator_contracts(share_kv, gather_attn_single_config):
    """Both opaque operators preserve fake metadata, aliasing and compiled dispatch."""
    _cuda()
    from attn_gym.sparse.gather_attn.impl.triton.ops import (
        _gather_attn_bwd_op,
        _gather_attn_fwd_op,
    )

    q, local, sparse, indices, sink, cu_q, cu_k = _inputs(20, 22, share_kv=share_kv)
    indices = indices.repeat_interleave(2, dim=-1)[..., ::2]
    args = (q, sparse, local, indices, sink, cu_q, cu_k, 7, share_kv, 0.25)
    for needs_backward in (False, True):
        torch.library.opcheck(_gather_attn_fwd_op, (*args, needs_backward))
    out, lse, queries, offsets = _gather_attn_fwd_op(*args, True)
    torch.library.opcheck(
        _gather_attn_bwd_op,
        (
            q,
            sparse,
            local,
            indices,
            queries,
            offsets,
            sink,
            cu_q,
            cu_k,
            out,
            lse,
            torch.randn_like(out),
            7,
            share_kv,
            0.25,
        ),
    )


def test_wide_addresses_keep_tma_coordinates_int32(monkeypatch, gather_attn_single_config):
    """Wide byte addressing must not widen the TMA descriptor's logical coordinates."""
    _cuda()
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("TMA requires Hopper or newer")
    from attn_gym.sparse.gather_attn.impl.triton import backward, forward

    q, local, sparse, indices, sink, cu_q, cu_k = _inputs(20, 22, dim=32, share_kv=False)
    tensors = tuple(t.requires_grad_() for t in (q, local, sparse, sink))
    inputs = (tensors, indices, cu_q, cu_k)

    expected = _forward_and_grads(*inputs, kernel_options={"backend": "triton"})
    monkeypatch.setattr(forward, "requires_int64_offsets", lambda *args: True)
    monkeypatch.setattr(backward, "requires_int64_offsets", lambda *args: True)
    actual = _forward_and_grads(*inputs, kernel_options={"backend": "triton"})
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference)


def test_runtime_kv_addresses_beyond_int32(gather_attn_single_config):
    """Real wide KV batch strides must address distinct batches in forward and backward."""
    _cuda()
    if torch.cuda.mem_get_info()[0] < 24 * 2**30:
        pytest.skip("wide-address test needs two 8 GiB allocations plus headroom")
    q = torch.randn(2, 2, 19, 32, device="cuda") * 0.1
    local = torch.randn_like(q) * 0.1
    sparse = torch.randn(2, 2, 13, 32, device="cuda") * 0.1
    sink = torch.randn(2, device="cuda") * 0.1
    indices = torch.tensor([0, 1, -1], device="cuda", dtype=torch.int32).repeat(2, 19, 1)

    def widen(tensor):
        strides = (2**31 + 32, tensor.shape[2] * 32, 32, 1)
        wide = torch.empty_strided(tensor.shape, strides, device="cuda", dtype=tensor.dtype)
        return wide.copy_(tensor)

    compact = (q, local, sparse, sink)
    wide = (q.clone(), widen(local), widen(sparse), sink.clone())

    def run(tensors):
        q, local, sparse, sink = (tensor.requires_grad_() for tensor in tensors)
        out = gather_attn(
            q,
            local,
            sparse,
            indices,
            sink,
            sliding_window_size=7,
            kernel_options={"backend": "triton"},
        )
        return out, *torch.autograd.grad(out.sum(), tensors)

    expected = run(compact)
    actual = run(wide)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference)


def _kernel_keys():
    """Observe the explicit no-sequence-recompile contract, not launch counts."""
    from triton.runtime.autotuner import Autotuner, Heuristics
    from triton.runtime.jit import JITFunction

    from attn_gym.sparse.gather_attn.impl.triton import backward, forward, shared_backward

    device = torch.cuda.current_device()
    result = {}
    for module in (forward, backward, shared_backward):
        for name, value in vars(module).items():
            while isinstance(value, (Autotuner, Heuristics)):
                value = value.fn
            if isinstance(value, JITFunction):
                cache = value.device_caches.get(device)
                if cache is not None and cache[0]:
                    result[(module.__name__, name)] = frozenset(cache[0])
    return result


@pytest.mark.parametrize(
    "heads,dim,dtype,share_kv,deterministic",
    [
        (2, 17, torch.float32, False, False),
        (2, 32, torch.float32, False, False),
        (16, 32, torch.bfloat16, True, False),
        (16, 32, torch.bfloat16, True, True),
    ],
    ids=["generic", "tma", "shared-atomic", "shared-deterministic"],
)
def test_sequence_lengths_reuse_triton_kernels(heads, dim, dtype, share_kv, deterministic):
    """Changing T/S within one alignment class must reuse compiled kernels."""
    _cuda()
    from test_gather_attn_triton import assert_matches_low_precision_eager

    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(deterministic)
    try:
        first_keys = None
        # Runtime tuples retain Triton's normal alignment specialization, not exact lengths.
        for tokens, candidates in ((32, 16), (48, 32)):
            q, local, sparse, indices, sink, cu_q, cu_k = _inputs(
                tokens,
                candidates,
                heads=heads,
                dim=dim,
                dtype=dtype,
                share_kv=share_kv,
                inner_stride=2 if dim == 17 else 1,
            )
            if tokens == 48:
                # Document-count changes also stay runtime data; add a leading empty document.
                cu_q = torch.cat((cu_q[:1], cu_q))
                cu_k = torch.cat((cu_k[:1], cu_k))
            tensors = tuple(t.requires_grad_() for t in (q, local, sparse, sink))
            actual = _forward_and_grads(
                tensors, indices, cu_q, cu_k, kernel_options={"backend": "triton"}
            )
            low, high = (
                _forward_and_grads(
                    tuple(cast(t.detach()).requires_grad_() for t in tensors),
                    indices,
                    cu_q,
                    cu_k,
                    impl="reference",
                )
                for cast in (torch.Tensor.clone, torch.Tensor.double)
            )
            forward_reductions = (dim, 10, 10)
            head_reduction = heads if share_kv else 1
            reductions = (
                forward_reductions,
                forward_reductions + (dim, 10),
                forward_reductions + (dim, 7, head_reduction),
                forward_reductions + (dim, tokens, head_reduction),
                forward_reductions + (dim, tokens),
            )
            for i, (value, low_ref, high_ref) in enumerate(zip(actual, low, high)):
                assert_matches_low_precision_eager(
                    value,
                    low_ref,
                    high_ref,
                    reductions[i],
                    quantized_intermediates=2 if i == 4 else 0,
                    intermediate_dtype=dtype,
                )
            keys = _kernel_keys()
            assert keys, "No Triton consumer kernels were compiled"
            if first_keys is None:
                first_keys = keys
            else:
                assert keys == first_keys, "Sequence length changed the Triton compilation keys"
    finally:
        torch.use_deterministic_algorithms(previous)
