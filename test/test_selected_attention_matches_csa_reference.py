"""Verify that composing CSA from selected_attention matches the standalone reference.

The composition path (CSA built from selected_attention) is imported directly
from examples/compressed_sparse_attention.py. The standalone reference CSA is
defined inline here for comparison.
"""

import math
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from attn_gym.sparse.selected_attention import AuxRequest

ATOL = 1e-8
RTOL = 1e-5

pytestmark = pytest.mark.usefixtures("selected_attention_single_config")


# ---------------------------------------------------------------------------
# Load the CSA example module (composition path via selected_attention).
# ---------------------------------------------------------------------------


def _load_csa_example():
    example_path = (
        Path(__file__).resolve().parents[1] / "examples" / "compressed_sparse_attention.py"
    )
    spec = spec_from_file_location("_compressed_sparse_attention_example", example_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load example from {example_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


csa_example = _load_csa_example()


# ---------------------------------------------------------------------------
# Helper functions for the standalone reference.
# ---------------------------------------------------------------------------


def _pad_to_block_size(x: torch.Tensor, m: int, value: float) -> torch.Tensor:
    pad_length = (-x.shape[-2]) % m
    if pad_length == 0:
        return x
    return F.pad(x, (0, 0, 0, pad_length), mode="constant", value=value)


def _split_blocks(x: torch.Tensor, compression_rate: int) -> torch.Tensor:
    return x.reshape(
        *x.shape[:-2],
        x.shape[-2] // compression_rate,
        compression_rate,
        x.shape[-1],
    )


def compress(C_a, C_b, Z_a, Z_b, B_a, B_b, compression_rate):
    C_a = _pad_to_block_size(C_a, compression_rate, 0.0)
    C_b = _pad_to_block_size(C_b, compression_rate, 0.0)
    Z_a = _pad_to_block_size(Z_a, compression_rate, float("-inf"))
    Z_b = _pad_to_block_size(Z_b, compression_rate, float("-inf"))

    C_b = F.pad(C_b, (0, 0, compression_rate, 0), "constant", 0.0)[:, :, :-compression_rate, :]
    Z_b = F.pad(Z_b, (0, 0, compression_rate, 0), "constant", float("-inf"))[
        :, :, :-compression_rate, :
    ]

    Z_a = _split_blocks(Z_a, compression_rate)
    Z_b = _split_blocks(Z_b, compression_rate)
    C_a = _split_blocks(C_a, compression_rate)
    C_b = _split_blocks(C_b, compression_rate)

    logits = torch.cat([Z_a + B_a, Z_b + B_b], dim=-2)
    logits_normalized = F.softmax(logits, dim=-2)
    S_a = logits_normalized[:, :, :, :compression_rate, :]
    S_b = logits_normalized[:, :, :, compression_rate:, :]

    weighted = C_a * S_a + C_b * S_b
    return torch.sum(weighted, dim=-2)


def apply_rope(
    x: torch.Tensor,
    positions=None,
    base: float = 160_000.0,
    original_seq_len: int = 65_536,
    factor: float = 16.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    position_offset: int = 0,
    inverse: bool = False,
) -> torch.Tensor:
    sequence_length = x.shape[-2]
    rotary_dim = x.shape[-1]
    dtype = x.dtype
    if positions is None:
        positions = torch.arange(
            position_offset,
            position_offset + sequence_length,
            device=x.device,
            dtype=dtype,
        )
    else:
        positions = positions.to(device=x.device, dtype=dtype)

    frequencies = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, device=x.device, dtype=dtype) / rotary_dim)
    )

    if original_seq_len > 0:

        def correction_dimension(num_rotations):
            return (
                rotary_dim
                * math.log(original_seq_len / (num_rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(math.floor(correction_dimension(beta_fast)), 0)
        high = min(math.ceil(correction_dimension(beta_slow)), rotary_dim - 1)
        if low == high:
            high += 0.001

        ramp = (torch.arange(rotary_dim // 2, device=x.device, dtype=dtype) - low) / (high - low)
        smooth = 1 - ramp.clamp(0, 1)
        frequencies = frequencies / factor * (1 - smooth) + frequencies * smooth

    angles = torch.outer(positions, frequencies)
    frequencies_complex = torch.polar(torch.ones_like(angles), angles)
    if inverse:
        frequencies_complex = frequencies_complex.conj()

    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], rotary_dim // 2, 2))
    frequencies_complex = frequencies_complex.view(
        *([1] * (x.ndim - 2)),
        sequence_length,
        rotary_dim // 2,
    )
    rotated = torch.view_as_real(x_complex * frequencies_complex).flatten(-2)
    return rotated.to(x.dtype)


def make_block_mask(query_length, num_blocks, compression_rate, device, dtype):
    query_positions = torch.arange(query_length, device=device)
    block_positions = torch.arange(num_blocks, device=device)
    completed_blocks = (query_positions + 1) // compression_rate
    bool_mask = block_positions[None, :] < completed_blocks[:, None]
    mask = torch.zeros(bool_mask.shape, device=bool_mask.device, dtype=dtype)
    return mask.masked_fill(~bool_mask, float("-inf"))


# ---------------------------------------------------------------------------
# Standalone CSA reference (no dependency on selected_attention).
# ---------------------------------------------------------------------------


def _make_sliding_window_mask(query_length, window_size, device, dtype):
    query_positions = torch.arange(query_length, device=device)[:, None]
    key_positions = torch.arange(query_length, device=device)[None, :]
    valid = (key_positions <= query_positions) & (
        key_positions >= query_positions - window_size + 1
    )
    return torch.zeros((query_length, query_length), device=device, dtype=dtype).masked_fill(
        ~valid, float("-inf")
    )


def _sink_softmax(x, sink, dim):
    sink = sink[None, :, None, None]
    maximums = torch.max(x, dim=dim, keepdim=True).values
    maximums = torch.maximum(maximums, sink)
    x = x - maximums
    sink = sink - maximums
    x = torch.exp(x)
    return x / (torch.sum(x, dim, keepdim=True) + torch.exp(sink))


def reference_CSA(
    Q,
    Q_I,
    KV,
    C_a,
    C_b,
    Z_a,
    Z_b,
    B_a,
    B_b,
    W_I,
    K_Ia,
    K_Ib,
    Z_Ia,
    Z_Ib,
    B_Ia,
    B_Ib,
    KV_norm_weight,
    compressed_indices_norm_weight,
    compressed_kv_norm_weight,
    attention_sink,
    compression_rate,
    num_topk_blocks,
    sliding_window_size,
    rope_dims: int,
    share_kv: bool,
):
    device = Q.device
    dtype = Q.dtype
    b, h, s, head_dim = Q.shape
    _, h_I, _, head_dim_I = Q_I.shape
    if share_kv:
        KV = KV.expand(-1, h, -1, -1)
        C_a = C_a.expand(-1, h, -1, -1)
        C_b = C_b.expand(-1, h, -1, -1)
        Z_a = Z_a.expand(-1, h, -1, -1)
        Z_b = Z_b.expand(-1, h, -1, -1)

        K_Ia = K_Ia.expand(-1, h_I, -1, -1)
        K_Ib = K_Ib.expand(-1, h_I, -1, -1)
        Z_Ia = Z_Ia.expand(-1, h_I, -1, -1)
        Z_Ib = Z_Ib.expand(-1, h_I, -1, -1)

    compressed_kv = compress(C_a, C_b, Z_a, Z_b, B_a, B_b, compression_rate)
    compressed_indices = compress(K_Ia, K_Ib, Z_Ia, Z_Ib, B_Ia, B_Ib, compression_rate)
    num_total_blocks = compressed_kv.shape[-2]

    Q = torch.cat([Q[:, :, :, :-rope_dims], apply_rope(Q[:, :, :, -rope_dims:])], dim=-1)
    Q_I = torch.cat([Q_I[:, :, :, :-rope_dims], apply_rope(Q_I[:, :, :, -rope_dims:])], dim=-1)
    KV = F.rms_norm(KV, (KV.shape[-1],), weight=KV_norm_weight)
    KV = torch.cat([KV[:, :, :, :-rope_dims], apply_rope(KV[:, :, :, -rope_dims:])], dim=-1)

    compressed_positions = torch.arange(num_total_blocks, device=device) * compression_rate
    compressed_indices = F.rms_norm(
        compressed_indices,
        (compressed_indices.shape[-1],),
        weight=compressed_indices_norm_weight,
    )
    compressed_indices = torch.cat(
        [
            compressed_indices[:, :, :, :-rope_dims],
            apply_rope(compressed_indices[:, :, :, -rope_dims:], positions=compressed_positions),
        ],
        dim=-1,
    )
    compressed_kv = F.rms_norm(
        compressed_kv, (compressed_kv.shape[-1],), weight=compressed_kv_norm_weight
    )
    compressed_kv = torch.cat(
        [
            compressed_kv[:, :, :, :-rope_dims],
            apply_rope(compressed_kv[:, :, :, -rope_dims:], positions=compressed_positions),
        ],
        dim=-1,
    )

    indexer_mask = make_block_mask(s, num_total_blocks, compression_rate, device, dtype)
    indexer_scale = (head_dim_I * h_I) ** 0.5
    scores = F.relu(Q_I @ torch.permute(compressed_indices, (0, 1, 3, 2))) / indexer_scale
    W_I_perm = torch.permute(W_I, (0, 2, 1)).unsqueeze(-1)
    scores = torch.sum(torch.multiply(W_I_perm, scores), dim=1) + indexer_mask

    _, topk_blocks = torch.topk(scores, k=min(num_topk_blocks, num_total_blocks), dim=-1)
    topk_mask = torch.full(scores.shape, float("-inf"), device=device, dtype=dtype)
    topk_mask.scatter_(dim=-1, index=topk_blocks, value=0.0)
    topk_mask += indexer_mask

    SWA_mask = _make_sliding_window_mask(s, sliding_window_size, device, dtype).unsqueeze(0)
    SWA_mask = SWA_mask.expand(b, -1, -1)

    attention_kv = torch.cat([compressed_kv, KV], dim=-2)
    attention_mask = torch.cat([topk_mask, SWA_mask], dim=-1).unsqueeze(1)
    scale = head_dim**0.5

    P = _sink_softmax(
        torch.matmul(Q, torch.permute(attention_kv, (0, 1, 3, 2))) / scale + attention_mask,
        attention_sink,
        dim=-1,
    )
    attn_output = P @ attention_kv
    return torch.cat(
        [
            attn_output[..., :-rope_dims],
            apply_rope(attn_output[..., -rope_dims:], inverse=True),
        ],
        dim=-1,
    )


# ---------------------------------------------------------------------------
# Test input generation.
# ---------------------------------------------------------------------------


def _make_inputs(
    share_kv: bool,
    num_topk_blocks: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
):
    generator = torch.Generator(device=device).manual_seed(0)
    batch_size = 1
    num_heads = 2
    num_index_heads = 1
    sequence_length = 5
    head_dim = 4
    index_head_dim = 4
    compression_rate = 2
    sliding_window_size = 2
    rope_dims = 4
    num_blocks = (sequence_length + compression_rate - 1) // compression_rate

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=dtype, device=device)

    kv_heads = 1 if share_kv else num_heads
    index_kv_heads = 1 if share_kv else num_index_heads
    return (
        randn(batch_size, num_heads, sequence_length, head_dim),
        randn(batch_size, num_index_heads, sequence_length, index_head_dim),
        randn(batch_size, kv_heads, sequence_length, head_dim),
        randn(batch_size, kv_heads, sequence_length, head_dim),
        randn(batch_size, kv_heads, sequence_length, head_dim),
        randn(batch_size, kv_heads, sequence_length, 1),
        randn(batch_size, kv_heads, sequence_length, 1),
        randn(1, num_heads, num_blocks, 1, 1),
        randn(1, num_heads, num_blocks, 1, 1),
        randn(batch_size, sequence_length, num_index_heads),
        randn(batch_size, index_kv_heads, sequence_length, index_head_dim),
        randn(batch_size, index_kv_heads, sequence_length, index_head_dim),
        randn(batch_size, index_kv_heads, sequence_length, 1),
        randn(batch_size, index_kv_heads, sequence_length, 1),
        randn(1, num_index_heads, num_blocks, 1, 1),
        randn(1, num_index_heads, num_blocks, 1, 1),
        torch.ones(head_dim, dtype=dtype, device=device),
        torch.ones(index_head_dim, dtype=dtype, device=device),
        torch.ones(head_dim, dtype=dtype, device=device),
        torch.zeros(num_heads, dtype=dtype, device=device),
        compression_rate,
        num_topk_blocks,
        sliding_window_size,
        rope_dims,
        share_kv,
    )


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("share_kv", [False, True])
@pytest.mark.parametrize("num_topk_blocks", [0, 1])
def test_selected_attention_matches_csa_reference_fp64(share_kv, num_topk_blocks):
    inputs = _make_inputs(
        share_kv,
        num_topk_blocks,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    floating_inputs = inputs[:20]
    assert all(tensor.dtype == torch.float64 for tensor in floating_inputs)

    with torch.inference_mode():
        expected = reference_CSA(*inputs)
        actual = csa_example.CSA(*inputs)

    assert actual.dtype == expected.dtype == torch.float64
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_selected_attention_matches_csa_reference_cuda_fp64():
    inputs = _make_inputs(
        True,
        1,
        dtype=torch.float64,
        device="cuda",
    )

    with torch.inference_mode():
        expected = reference_CSA(*inputs)
        actual = csa_example.CSA(*inputs)

    assert actual.device.type == expected.device.type == "cuda"
    assert actual.dtype == expected.dtype == torch.float64
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# End-to-end indexer_loss tests.
# ---------------------------------------------------------------------------


def _indexer_loss_oracle(
    main_query: torch.Tensor,
    selected_compressed_kv: torch.Tensor,
    attention_lse: torch.Tensor,
    selected_indexer_logits: torch.Tensor,
    selected_is_valid: torch.Tensor,
) -> torch.Tensor:
    query = main_query.detach().to(torch.float32)
    keys = selected_compressed_kv.detach().to(torch.float32)
    selected_attention_logits = torch.matmul(query.unsqueeze(-2), keys.transpose(-2, -1)).squeeze(
        -2
    ) / math.sqrt(query.shape[-1])
    teacher_mass = torch.exp(
        selected_attention_logits - attention_lse.detach().to(torch.float32).unsqueeze(-1)
    ).sum(dim=1)
    teacher_mass = torch.where(selected_is_valid, teacher_mass, 0.0)
    teacher_probs = teacher_mass / teacher_mass.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(torch.float32).tiny
    )

    valid_rows = selected_is_valid.any(dim=-1)
    student_logits = selected_indexer_logits.to(torch.float32).masked_fill(
        ~selected_is_valid, float("-inf")
    )
    student_logits = torch.where(valid_rows.unsqueeze(-1), student_logits, 0.0)
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_log_probs = torch.where(selected_is_valid, student_log_probs, 0.0)
    row_kl = (
        torch.special.xlogy(teacher_probs, teacher_probs) - teacher_probs * student_log_probs
    ).sum(dim=-1)
    return row_kl.masked_select(valid_rows).mean()


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
        ),
    ],
)
def test_end_to_end_indexer_loss(device):
    """Run compress → index → selected_attention → indexer_loss end-to-end.

    Uses share_kv=True with the standard _make_inputs config (num_heads=2,
    per-head compression biases).  compression_rate=2 with sequence_length=5
    means position 0 has no completed blocks, exercising the early-position
    validity mask.
    """
    dtype = torch.float64
    inputs = _make_inputs(share_kv=True, num_topk_blocks=2, dtype=dtype, device=device)
    (
        Q,
        Q_I,
        KV,
        C_a,
        C_b,
        Z_a,
        Z_b,
        B_a,
        B_b,
        W_I,
        K_Ia,
        K_Ib,
        Z_Ia,
        Z_Ib,
        B_Ia,
        B_Ib,
        KV_norm_weight,
        compressed_indices_norm_weight,
        compressed_kv_norm_weight,
        attention_sink,
        compression_rate,
        num_topk_blocks,
        sliding_window_size,
        rope_dims,
        _share_kv,
    ) = inputs

    b, num_heads, sequence_length, head_dim = Q.shape
    _, num_index_heads, _, index_head_dim = Q_I.shape

    # Expand shared KV heads (mirrors CSA internals)
    KV = KV.expand(-1, num_heads, -1, -1)
    C_a = C_a.expand(-1, num_heads, -1, -1)
    C_b = C_b.expand(-1, num_heads, -1, -1)
    Z_a = Z_a.expand(-1, num_heads, -1, -1)
    Z_b = Z_b.expand(-1, num_heads, -1, -1)
    K_Ia = K_Ia.expand(-1, num_index_heads, -1, -1)
    K_Ib = K_Ib.expand(-1, num_index_heads, -1, -1)
    Z_Ia = Z_Ia.expand(-1, num_index_heads, -1, -1)
    Z_Ib = Z_Ib.expand(-1, num_index_heads, -1, -1)

    compressed_kv = csa_example.compress(C_a, C_b, Z_a, Z_b, B_a, B_b, compression_rate)
    compressed_indices = csa_example.compress(K_Ia, K_Ib, Z_Ia, Z_Ib, B_Ia, B_Ib, compression_rate)
    num_total_blocks = compressed_kv.shape[-2]

    Q_roped = torch.cat([Q[..., :-rope_dims], csa_example.apply_rope(Q[..., -rope_dims:])], dim=-1)
    Q_I_roped = torch.cat(
        [Q_I[..., :-rope_dims], csa_example.apply_rope(Q_I[..., -rope_dims:])], dim=-1
    )
    KV = F.rms_norm(KV, (head_dim,), weight=KV_norm_weight)
    KV = torch.cat([KV[..., :-rope_dims], csa_example.apply_rope(KV[..., -rope_dims:])], dim=-1)

    compressed_positions = torch.arange(num_total_blocks, device=device) * compression_rate
    compressed_indices = F.rms_norm(
        compressed_indices, (index_head_dim,), weight=compressed_indices_norm_weight
    )
    compressed_indices = torch.cat(
        [
            compressed_indices[..., :-rope_dims],
            csa_example.apply_rope(
                compressed_indices[..., -rope_dims:], positions=compressed_positions
            ),
        ],
        dim=-1,
    )
    compressed_kv = F.rms_norm(compressed_kv, (head_dim,), weight=compressed_kv_norm_weight)
    compressed_kv = torch.cat(
        [
            compressed_kv[..., :-rope_dims],
            csa_example.apply_rope(
                compressed_kv[..., -rope_dims:], positions=compressed_positions
            ),
        ],
        dim=-1,
    )

    # Indexer scoring
    indexer_mask = csa_example.make_block_mask(
        sequence_length, num_total_blocks, compression_rate, device, dtype
    )
    indexer_scale = math.sqrt(index_head_dim * num_index_heads)
    scores = F.relu(Q_I_roped @ compressed_indices.transpose(-2, -1)) / indexer_scale
    index_head_weights = W_I.transpose(1, 2).unsqueeze(-1)
    scores = torch.sum(index_head_weights * scores, dim=1) + indexer_mask

    topk_blocks = torch.topk(scores, k=min(num_topk_blocks, num_total_blocks), dim=-1).indices

    # selected_attention with causal blocks → (output, aux, selected_is_valid)
    _attn_output, aux, selected_is_valid = csa_example._selected_attention_with_causal_blocks(
        Q_roped,
        KV,
        compressed_kv,
        topk_blocks,
        indexer_mask,
        attention_sink,
        sliding_window_size,
        return_aux=AuxRequest(lse=True),
    )
    lse = aux.lse
    assert lse is not None

    # Gather selected compressed keys: (B, H, num_blocks, D) → (B, H, S, K, D)
    idx = topk_blocks[:, None, :, :].expand(b, num_heads, sequence_length, num_topk_blocks)
    idx_flat = idx.reshape(b, num_heads, -1)
    gathered = compressed_kv.gather(
        dim=2, index=idx_flat.unsqueeze(-1).expand(*idx_flat.shape, head_dim)
    )
    selected_compressed_kv = gathered.reshape(
        b, num_heads, sequence_length, num_topk_blocks, head_dim
    )

    selected_indexer_logits = scores.gather(dim=-1, index=topk_blocks).detach()
    actual_logits = selected_indexer_logits.clone().requires_grad_()
    expected_logits = selected_indexer_logits.clone().requires_grad_()

    loss = csa_example.indexer_loss(
        Q_roped, selected_compressed_kv, lse, actual_logits, selected_is_valid
    )
    expected_loss = _indexer_loss_oracle(
        Q_roped,
        selected_compressed_kv,
        lse,
        expected_logits,
        selected_is_valid,
    )
    (actual_grad,) = torch.autograd.grad(loss, actual_logits)
    (expected_grad,) = torch.autograd.grad(expected_loss, expected_logits)

    assert loss.shape == ()
    torch.testing.assert_close(loss, expected_loss, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(actual_grad[selected_is_valid]) > 0
    assert torch.count_nonzero(actual_grad[~selected_is_valid]) == 0

    # Position 0 has no completed blocks (compression_rate=2, seq_len=5)
    assert not selected_is_valid[:, 0, :].any(), "Position 0 should have no valid blocks"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_indexer_loss_teacher_logits_use_fp32(dtype):
    generator = torch.Generator().manual_seed(1234)
    b, h, s, k, d = 2, 3, 4, 3, 64
    query = torch.randn(b, h, s, d, generator=generator, dtype=dtype)
    selected_kv = torch.randn(b, h, s, k, d, generator=generator, dtype=dtype)
    selected_is_valid = torch.tensor(
        [
            [
                [False, False, False],
                [True, False, False],
                [True, True, False],
                [True, True, True],
            ],
            [
                [False, False, False],
                [True, False, False],
                [True, True, False],
                [True, True, True],
            ],
        ]
    )
    attention_logits = torch.matmul(
        query.float().unsqueeze(-2), selected_kv.float().transpose(-2, -1)
    ).squeeze(-2) / math.sqrt(d)
    other_logits = torch.randn(b, h, s, 5, generator=generator)
    attention_lse = torch.logsumexp(torch.cat([attention_logits, other_logits], dim=-1), dim=-1)
    logits = torch.randn(b, s, k, generator=generator)
    actual_logits = logits.clone().requires_grad_()
    expected_logits = logits.clone().requires_grad_()

    actual_loss = csa_example.indexer_loss(
        query, selected_kv, attention_lse, actual_logits, selected_is_valid
    )
    expected_loss = _indexer_loss_oracle(
        query, selected_kv, attention_lse, expected_logits, selected_is_valid
    )
    (actual_grad,) = torch.autograd.grad(actual_loss, actual_logits)
    (expected_grad,) = torch.autograd.grad(expected_loss, expected_logits)

    torch.testing.assert_close(actual_loss, expected_loss, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=1e-6)
