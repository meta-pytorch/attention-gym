import math

import pytest
import torch

pytest.importorskip(
    "cutlass.experimental",
    reason="the CuTeDSL 4.7 KDA path requires nvidia-cutlass-dsl>=4.7",
)

from attn_gym.linear._delta_rule.mega import forward as adapter
from attn_gym.linear._delta_rule.mega.forward import (
    chunk_delta_rule_fwd_mega,
    chunk_delta_rule_fwd_mega_unsplit_with_state,
)
from attn_gym.linear._delta_rule.mega.kernels.common.split_k import (
    WORK_ITEM_FIELDS,
    build_split_table,
    chunk_scratch_rows,
)
from attn_gym.testing.kda import (
    assert_matches_low_precision_reference,
    cumulative_sequence_offsets,
    kda_reference,
    make_kda_test_inputs,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the CuTeDSL 4.7 KDA path requires SM100 or SM103",
)


def _reference_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    output, final_state = kda_reference(
        q.to(dtype),
        k.to(dtype),
        value.to(dtype),
        gate.to(dtype),
        beta.to(dtype),
        initial_state.to(dtype),
        cu_seqlens=cu_seqlens,
    )
    assert final_state is not None
    return output, final_state


def _inputs(dtype: torch.dtype = torch.bfloat16) -> tuple[torch.Tensor, ...]:
    lengths = (65, 0, 127)
    q, k, value, gate, beta = make_kda_test_inputs(
        sum(lengths),
        seed=0,
        gate_scale=math.log(2.0),
        log_uniform_gate=True,
        sigmoid_beta=True,
        dtype=dtype,
        normalize_qk=True,
        value_scale=1.0,
    )
    initial_state = torch.randn(len(lengths), 1, 128, 128, device="cuda").div_(100)
    return (
        q,
        k,
        value,
        gate,
        beta,
        cumulative_sequence_offsets(lengths),
        initial_state,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mega_forward_packed_tail_empty_and_deterministic(dtype: torch.dtype) -> None:
    q, k, value, gate, beta, cu_seqlens, initial_state = _inputs(dtype)
    zero_state = torch.zeros_like(initial_state)
    low_precision, _ = _reference_forward(
        q, k, value, gate, beta, zero_state, cu_seqlens, torch.float32
    )
    high_precision, _ = _reference_forward(
        q, k, value, gate, beta, zero_state, cu_seqlens, torch.float64
    )
    actual = chunk_delta_rule_fwd_mega(q, k, value, gate, beta, cu_seqlens)
    assert_matches_low_precision_reference(
        actual, high_precision, low_precision, "output", source_dtype=dtype
    )
    for _ in range(20):
        torch.testing.assert_close(
            chunk_delta_rule_fwd_mega(q, k, value, gate, beta, cu_seqlens),
            actual,
            rtol=0,
            atol=0,
        )

    low_output, low_state = _reference_forward(
        q, k, value, gate, beta, initial_state, cu_seqlens, torch.float32
    )
    high_output, high_state = _reference_forward(
        q, k, value, gate, beta, initial_state, cu_seqlens, torch.float64
    )
    actual_output, actual_state = chunk_delta_rule_fwd_mega_unsplit_with_state(
        q,
        k,
        value,
        gate,
        beta,
        initial_state,
        cu_seqlens,
    )
    assert_matches_low_precision_reference(
        actual_output, high_output, low_output, "stateful output", source_dtype=dtype
    )
    assert_matches_low_precision_reference(
        actual_state, high_state, low_state, "final state", source_dtype=dtype
    )


def test_mega_split_table_rejects_misaligned_vector_gate_rows() -> None:
    tokens, heads, dim = 16, 2, 128
    gate = torch.empty(tokens, heads, dim + 1, device="cuda")[..., :dim]
    assert gate.data_ptr() % 16 == 0
    assert gate.stride(-1) == 1

    cu_seqlens = torch.tensor([0, tokens], dtype=torch.int32, device="cuda")
    work_items = torch.empty(16, WORK_ITEM_FIELDS, dtype=torch.int32, device="cuda")
    item_scratch = torch.empty_like(work_items)
    work_count = torch.empty(1, dtype=torch.int32, device="cuda")
    chunk_scratch = torch.empty(
        chunk_scratch_rows(tokens, 1, 16),
        heads,
        dtype=torch.float32,
        device="cuda",
    )

    with pytest.raises(ValueError, match="rows and head slices must be 16-byte aligned"):
        build_split_table(
            gate,
            cu_seqlens,
            work_items,
            work_count,
            ideal_chunks=1,
            n_tiles=heads,
            num_sms=1,
            b_t=16,
            chunk_scratch=chunk_scratch,
            item_scratch=item_scratch,
            log_gate=True,
            split=True,
            stream=torch.cuda.current_stream().cuda_stream,
        )


def test_mega_split_table_rejects_invalid_geometry_and_capacity() -> None:
    tokens, heads, dim = 16, 2, 128
    gate = torch.zeros(tokens, heads, dim, device="cuda")
    cu_seqlens = torch.tensor([0, tokens], dtype=torch.int32, device="cuda")
    work_count = torch.empty(1, dtype=torch.int32, device="cuda")
    stream = torch.cuda.current_stream().cuda_stream

    with pytest.raises(ValueError, match="n_tiles must equal"):
        build_split_table(
            gate,
            cu_seqlens,
            torch.empty(heads, WORK_ITEM_FIELDS, dtype=torch.int32, device="cuda"),
            work_count,
            n_tiles=heads - 1,
            num_sms=1,
            b_t=16,
            split=False,
            stream=stream,
        )

    work_items = torch.empty(1, WORK_ITEM_FIELDS, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="requires at least"):
        build_split_table(
            gate,
            cu_seqlens,
            work_items,
            work_count,
            ideal_chunks=1,
            n_tiles=heads,
            num_sms=1,
            b_t=16,
            chunk_scratch=torch.empty(
                chunk_scratch_rows(tokens, 1, 16), heads, dtype=torch.float32, device="cuda"
            ),
            item_scratch=torch.empty_like(work_items),
            log_gate=True,
            split=True,
            stream=stream,
        )


def test_mega_forward_rejects_misaligned_tma() -> None:
    q, k, value, gate, beta, cu_seqlens, _ = _inputs()
    storage = torch.empty(q.numel() + 1, dtype=q.dtype, device="cuda")
    misaligned_q = storage[1:].view_as(q)
    with pytest.raises(TypeError, match="TMA-compatible inner mode"):
        adapter.chunk_delta_rule_fwd_mega(misaligned_q, k, value, gate, beta, cu_seqlens)


def test_mega_public_forward_preserves_large_finite_state() -> None:
    torch.manual_seed(19)
    tokens, heads, dim = 1024, 1, 128
    shape = (1, tokens, heads, dim)
    q = torch.nn.functional.normalize(torch.randn(shape, device="cuda"), dim=-1).bfloat16()
    k = torch.nn.functional.normalize(torch.randn(shape, device="cuda"), dim=-1).bfloat16()
    value = torch.zeros(shape, device="cuda", dtype=torch.bfloat16)
    gate = torch.zeros(shape, device="cuda")
    gate[:, ::4] = -2.5
    beta = torch.zeros(*shape[:3], device="cuda")
    cu_seqlens = torch.tensor([0, tokens], dtype=torch.int32, device="cuda")
    initial_state = torch.full((1, heads, dim, dim), 1e30, device="cuda")

    low_precision, _ = _reference_forward(
        q,
        k,
        value,
        gate,
        beta,
        initial_state,
        cu_seqlens,
        torch.float32,
    )
    high_precision, _ = _reference_forward(
        q,
        k,
        value,
        gate,
        beta,
        initial_state,
        cu_seqlens,
        torch.float64,
    )
    actual, _ = adapter.chunk_delta_rule_fwd_mega_unsplit_with_state(
        q,
        k,
        value,
        gate,
        beta,
        initial_state,
        cu_seqlens,
    )
    assert_matches_low_precision_reference(
        actual,
        high_precision,
        low_precision,
        "large-state output",
    )
