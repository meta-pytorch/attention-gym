"""Backend compilation reuse and numerical gates for the KDA intra stages."""

from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("triton")
pytest.importorskip("cutlass.cute", reason="requires nvidia-cutlass-dsl with CuTe TVM-FFI")

from attn_gym.linear import chunk_kda
from attn_gym.linear._delta_rule.chunk_schedule import prepare_ragged_chunk_metadata
from attn_gym.linear.kda.bwd.cute.chunk_kda_bwd_intra import (
    ChunkKdaBwdIntraConfig,
    _compile_chunk_kda_bwd_intra,
    chunk_kda_bwd_intra,
)
from attn_gym.linear.kda.fwd.triton.chunk_kda_fwd_intra_sub_chunk_forloop import (
    chunk_kda_fwd_intra_diagonal,
    chunk_kda_fwd_kernel_intra_sub_chunk_forloop,
)
from attn_gym.linear.kda.utils import IS_GATHER_SUPPORTED
from attn_gym.testing.kda import (
    assert_matches_low_precision_reference,
    bwd_intra_reference,
    clone_kda_inputs,
    cumulative_sequence_offsets,
    kda_reference,
    make_kda_test_inputs,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="KDA intra requires CUDA capability >= 8.0",
)


def intra_inputs(tokens: int, dtype: torch.dtype, *, strided: bool = False):
    torch.manual_seed(41)
    q, k, _, g, beta = make_kda_test_inputs(tokens, dtype=dtype, gate_scale=0.5)
    if strided:
        # The normal unbound-QKV layout, not a compact surrogate.
        qkv = torch.stack((q, k, q), dim=2)
        q, k, _ = qkv.unbind(2)
        assert not q.is_contiguous()
    da = [torch.randn(1, tokens, 1, 64, device="cuda") * 0.1 for _ in range(2)]
    running = [torch.randn_like(g) * 0.01 for _ in range(3)]
    db = torch.randn_like(beta) * 0.01
    return q, k, g, beta, *da, running[0], running[1], db, running[2]


def metadata_for(tokens: int, lengths: list[int] | None):
    if lengths is None:
        return None
    return prepare_ragged_chunk_metadata(cumulative_sequence_offsets(lengths), tokens, 64)


def diagonal(inputs, metadata, *, fixed: bool):
    q, k, g, beta = inputs[:4]
    if not fixed:
        return chunk_kda_fwd_intra_diagonal(q, k, g, beta, 128**-0.5, metadata, fastmath=True)
    tokens = q.shape[1]
    capacity = tokens // 64 if metadata is None else metadata.capacity
    aq = torch.empty(1, tokens, 1, 64, device="cuda", dtype=q.dtype)
    ak = torch.empty(1, tokens, 1, 16, device="cuda")
    chunk_kda_fwd_kernel_intra_sub_chunk_forloop[(1, 4, 1)](
        q,
        k,
        g,
        beta,
        aq,
        ak,
        128**-0.5,
        None if metadata is None else metadata.cu_seqlens,
        None if metadata is None else metadata.chunk_offsets,
        tokens,
        q.stride(1),
        q.stride(2),
        k.stride(1),
        k.stride(2),
        H=1,
        K=128,
        BT=64,
        BC=16,
        BK=128,
        num_sequences=0 if metadata is None else metadata.cu_seqlens.numel() - 1,
        USE_GATHER=IS_GATHER_SUPPORTED,
        GRID_NT=1,
        MAX_NT=capacity,
        FASTMATH=True,
    )
    return aq, ak


def check_diagonal(inputs, outputs, lengths):
    q, k, g, beta = inputs[:4]
    begin = 0
    for length in lengths:
        for offset in range(0, length, 16):
            size = min(16, length - offset)
            rows = slice(begin + offset, begin + offset + size)
            actual_q = outputs[0][0, rows, 0, offset % 64 : offset % 64 + size]
            actual_k = outputs[1][0, rows, 0, :size]
            references = []
            for acc in (torch.float64, torch.float32):
                qr, kr, gr = (x[0, rows, 0].to(acc) for x in (q, k, g))
                br = beta[0, rows, 0].to(acc)
                decay = (gr[:, None] - gr[None, :]).exp2()
                aq = ((qr[:, None] * kr[None, :] * decay).sum(-1) * 128**-0.5).tril()
                lower = ((kr[:, None] * kr[None, :] * decay).sum(-1) * br[:, None]).tril(-1)
                eye = torch.eye(size, dtype=acc, device=q.device)
                ak = torch.linalg.solve_triangular(eye + lower, eye, upper=False)
                references.append((aq, ak))
            for name, actual, high, low in zip(
                ("Aqk", "Akk"), (actual_q, actual_k), references[0], references[1]
            ):
                assert_matches_low_precision_reference(
                    actual, high, low, name, source_dtype=q.dtype
                )
        begin += length


def check_backward(inputs, outputs, lengths):
    q, k, g, beta, daq, dak, dq, dk, db, dg = inputs
    cu = cumulative_sequence_offsets(lengths)
    indices = torch.tensor(
        [
            (seq, chunk)
            for seq, length in enumerate(lengths)
            for chunk in range(math.ceil(length / 64))
        ],
        dtype=torch.int64,
        device="cuda",
    ).reshape(-1, 2)
    references = []
    for acc in (torch.float64, torch.float32):
        rq, rk, rb, rg = bwd_intra_reference(
            *(x.to(acc) for x in (q, k, g, beta, daq, dak)),
            cu_seqlens=cu,
            chunk_indices=indices,
        )
        references.append(
            (rq + dq.to(acc), rk + dk.to(acc), rg + dg.to(acc) * math.log(2), rb + db.to(acc))
        )
    active = sum(lengths)
    # Intra stages and the public core leave suffix outputs undefined. Fixed-capacity
    # callers own suffix masking; check only the documented active domain here.
    for name, actual, high, low in zip(("dq", "dk", "dg", "db"), outputs, *references):
        if active:
            assert_matches_low_precision_reference(
                actual[:, :active], high[:, :active], low[:, :active], name, source_dtype=q.dtype
            )


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("fixed", [False, True], ids=["default-grid", "fixed-grid"])
@pytest.mark.parametrize("stage", ["diagonal", "backward"])
def test_intra_backend_cache_reuse(stage, fixed, packed):
    """T and N vary independently, including equal capacity and multiple grid waves."""
    jit = chunk_kda_fwd_kernel_intra_sub_chunk_forloop.fn.fn
    cache = jit.device_caches[torch.cuda.current_device()][0]
    cache.clear()
    _compile_chunk_kda_bwd_intra.cache_clear()
    cases = (
        [
            (128, [0, 63, 0, 65]),
            (129, [0, 63, 0, 65]),
            (193, [0, 63, 0, 130]),
            (193, [0, 31, 32, 0, 65, 65]),
            (2, [0, 1, 1]),
            (2, [0, 1, 0, 1]),
        ]
        if packed
        else [(128, None), (192, None), (320, None)]
    )
    for tokens, lengths in cases:
        inputs = intra_inputs(tokens, torch.bfloat16)
        metadata = metadata_for(tokens, lengths)
        if stage == "diagonal":
            outputs = diagonal(inputs, metadata, fixed=fixed)
            check_diagonal(inputs, outputs, lengths or [tokens])
            assert len(cache) == 1, "diagonal compiled again for runtime T/N/capacity/grid"
        else:
            outputs = chunk_kda_bwd_intra(
                *inputs,
                metadata,
                config=ChunkKdaBwdIntraConfig(1) if fixed else None,
                fastmath=True,
            )
            check_backward(inputs, outputs, lengths or [tokens])
            info = _compile_chunk_kda_bwd_intra.cache_info()
            # Disk-cache loads count as hits, including another xdist worker's compile.
            assert info.currsize == 1, "backward compiled again for runtime T/N/capacity/grid"


def test_diagonal_runtime_grid_int64_offsets(monkeypatch):
    inputs = intra_inputs(193, torch.float16)
    lengths = [65, 0, 128]
    metadata = metadata_for(193, lengths)
    expected = diagonal(inputs, metadata, fixed=True)
    monkeypatch.setattr(
        "attn_gym.linear.kda.fwd.triton."
        "chunk_kda_fwd_intra_sub_chunk_forloop.requires_int64_offsets",
        lambda *tensors: True,
    )
    actual = diagonal(inputs, metadata, fixed=True)
    check_diagonal(inputs, actual, lengths)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("poison", [0.0, float("nan")], ids=["finite", "nan"])
def test_intra_graph_replay_changed_metadata(dtype, poison):
    inputs = intra_inputs(193, dtype, strided=True)
    cu = cumulative_sequence_offsets([0, 63, 0, 130])

    def operation():
        metadata = prepare_ragged_chunk_metadata(cu, 193, 64)
        return diagonal(inputs, metadata, fixed=True), chunk_kda_bwd_intra(
            *inputs, metadata, config=ChunkKdaBwdIntraConfig(1), fastmath=True
        )

    for _ in range(3):
        operation()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward, backward = operation()
    # Shrink to an empty batch, then grow again; no host metadata reads in replay.
    for lengths in ([65, 0, 64, 1], [0, 0, 0, 0], [1, 127, 0, 65]):
        fresh = intra_inputs(193, dtype, strided=True)
        for dst, src in zip(inputs, fresh):
            dst.copy_(src)
            dst[:, sum(lengths) :] = poison
        cu.copy_(cumulative_sequence_offsets(lengths))
        graph.replay()
        check_diagonal(inputs, forward, lengths)
        check_backward(inputs, backward, lengths)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "fullgraph"])
def test_public_kda_dynamic_training(compiled):
    """The documented op, not only private launchers, supports dynamic backward."""
    operation = torch.compile(chunk_kda, fullgraph=True, dynamic=True) if compiled else chunk_kda
    # FP16 selects the diagonal stage on both Hopper and Blackwell. BF16 Blackwell
    # uses a different forward engine; backward intra is CuTe on both architectures.
    for tokens, lengths in ((128, [0, 63, 65]), (193, [65, 0, 64, 64])):
        inputs = make_kda_test_inputs(
            tokens, dtype=torch.float16, normalize_qk=True, requires_grad=True
        )
        state = torch.randn(len(lengths), 1, 128, 128, device="cuda", requires_grad=True) * 0.01
        state = state.detach().requires_grad_()
        cu = cumulative_sequence_offsets(lengths)
        expected_inputs = clone_kda_inputs((*inputs, state), dtype=torch.float64)
        low_inputs = clone_kda_inputs((*inputs, state), dtype=torch.float32)
        actual = operation(*inputs, initial_state=state, cu_seqlens=cu, output_final_state=True)
        high = kda_reference(*expected_inputs, cu_seqlens=cu)
        low = kda_reference(*low_inputs, cu_seqlens=cu)
        gradients = []
        for output, leaves in (
            (actual, (*inputs, state)),
            (high, expected_inputs),
            (low, low_inputs),
        ):
            gradients.append(torch.autograd.grad(sum(x.float().sum() for x in output), leaves))
        for idx, (a, h, l) in enumerate(
            zip((*actual, *gradients[0]), (*high, *gradients[1]), (*low, *gradients[2]))
        ):
            assert_matches_low_precision_reference(
                a, h, l, f"public-{idx}", source_dtype=torch.float16
            )


def test_public_kda_graph_replay_training():
    inputs = make_kda_test_inputs(193, dtype=torch.float16, normalize_qk=True, requires_grad=True)
    state = torch.zeros(4, 1, 128, 128, device="cuda", requires_grad=True)
    cu = cumulative_sequence_offsets([0, 63, 0, 130])
    torch.autograd.graph.set_override_stale_capture_stream(True)

    def operation():
        output = chunk_kda(*inputs, initial_state=state, cu_seqlens=cu, output_final_state=True)
        gradients = torch.autograd.grad(sum(x.float().sum() for x in output), (*inputs, state))
        return (*output, *gradients)

    for _ in range(3):
        operation()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = operation()
    for seed, lengths in enumerate(([65, 0, 64, 1], [1, 127, 0, 65])):
        fresh = make_kda_test_inputs(193, dtype=torch.float16, normalize_qk=True, seed=seed)
        with torch.no_grad():
            for dst, src in zip(inputs, fresh):
                dst.copy_(src)
                dst[:, sum(lengths) :] = float("nan")
            state.uniform_(-0.01, 0.01)
        cu.copy_(cumulative_sequence_offsets(lengths))
        graph.replay()
        references = []
        for dtype in (torch.float64, torch.float32):
            leaves = clone_kda_inputs((*inputs, state), dtype=dtype)
            outputs = kda_reference(*leaves, cu_seqlens=cu)
            gradients = torch.autograd.grad(sum(x.float().sum() for x in outputs), leaves)
            references.append((*outputs, *gradients))
        for idx, (a, high, low) in enumerate(zip(actual, *references)):
            if idx not in (1, 7):  # Final state and initial-state gradient are not token tensors.
                a, high, low = (x[:, : sum(lengths)] for x in (a, high, low))
            assert_matches_low_precision_reference(
                a, high, low, f"replay-{idx}", source_dtype=torch.float16
            )
