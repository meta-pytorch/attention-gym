"""Triton forward backend for compressed sparse attention."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def _pad_to_block_size(x: torch.Tensor, block_size: int, value: float) -> torch.Tensor:
    pad_length = (-x.shape[-2]) % block_size
    if pad_length == 0:
        return x
    return F.pad(x, (0, 0, 0, pad_length), mode="constant", value=value)


def _compress(
    c_a: torch.Tensor,
    c_b: torch.Tensor,
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    b_a: torch.Tensor,
    b_b: torch.Tensor,
    compression_rate: int,
) -> torch.Tensor:
    """Apply the reference compression equations using native PyTorch views."""
    c_a = _pad_to_block_size(c_a, compression_rate, 0.0)
    c_b = _pad_to_block_size(c_b, compression_rate, 0.0)
    z_a = _pad_to_block_size(z_a, compression_rate, float("-inf"))
    z_b = _pad_to_block_size(z_b, compression_rate, float("-inf"))

    c_b = F.pad(c_b, (0, 0, compression_rate, 0), value=0.0)[
        :, :, :-compression_rate, :
    ]
    z_b = F.pad(z_b, (0, 0, compression_rate, 0), value=float("-inf"))[
        :, :, :-compression_rate, :
    ]

    batch, heads, padded_sequence, dimension = c_a.shape
    num_blocks = padded_sequence // compression_rate
    blocked_shape = (batch, heads, num_blocks, compression_rate, dimension)
    c_a = c_a.reshape(blocked_shape)
    c_b = c_b.reshape(blocked_shape)
    z_a = z_a.reshape(blocked_shape)
    z_b = z_b.reshape(blocked_shape)

    logits = torch.cat((z_a + b_a, z_b + b_b), dim=-2)
    probabilities = logits.softmax(dim=-2)
    return (
        c_a * probabilities[..., :compression_rate, :]
        + c_b * probabilities[..., compression_rate:, :]
    ).sum(dim=-2)


def _rope_frequencies(rotary_dim: int, device: torch.device) -> torch.Tensor:
    base = 160_000.0
    original_seq_len = 65_536
    factor = 16.0
    beta_fast = 32.0
    beta_slow = 1.0
    frequencies = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32)
            / rotary_dim
        )
    )

    def correction_dimension(num_rotations: float) -> float:
        return rotary_dim * math.log(
            original_seq_len / (num_rotations * 2 * math.pi)
        ) / (2 * math.log(base))

    low = max(math.floor(correction_dimension(beta_fast)), 0)
    high = min(math.ceil(correction_dimension(beta_slow)), rotary_dim - 1)
    if low == high:
        high += 0.001
    ramp = (
        torch.arange(rotary_dim // 2, device=device, dtype=torch.float32) - low
    ) / (high - low)
    smooth = 1 - ramp.clamp(0, 1)
    return frequencies / factor * (1 - smooth) + frequencies * smooth


def _apply_rope(
    x: torch.Tensor,
    rotary_dim: int,
    *,
    positions: torch.Tensor | None = None,
    inverse: bool = False,
) -> torch.Tensor:
    if positions is None:
        positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
    else:
        positions = positions.to(device=x.device, dtype=torch.float32)
    angles = torch.outer(positions, _rope_frequencies(rotary_dim, x.device))
    frequencies_complex = torch.polar(torch.ones_like(angles), angles)
    if inverse:
        frequencies_complex = frequencies_complex.conj()
    tail = x[..., -rotary_dim:].float().reshape(*x.shape[:-1], rotary_dim // 2, 2)
    tail = torch.view_as_complex(tail)
    frequency_shape = [1] * (x.ndim - 2) + [x.shape[-2], rotary_dim // 2]
    rotated = torch.view_as_real(
        tail * frequencies_complex.view(*frequency_shape)
    ).flatten(-2)
    return torch.cat((x[..., :-rotary_dim], rotated.to(x.dtype)), dim=-1)


@triton.jit
def _tiled_selected_attention_fwd(
    q_ptr,
    compressed_kv_ptr,
    local_kv_ptr,
    topk_ptr,
    sink_ptr,
    out_ptr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qs: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_cb: tl.constexpr,
    stride_ch: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_cd: tl.constexpr,
    stride_lb: tl.constexpr,
    stride_lh: tl.constexpr,
    stride_ls: tl.constexpr,
    stride_ld: tl.constexpr,
    stride_tb: tl.constexpr,
    stride_ts: tl.constexpr,
    stride_tk: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_os: tl.constexpr,
    stride_od: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    COMPRESSION_RATE: tl.constexpr,
    TOPK: tl.constexpr,
    WINDOW: tl.constexpr,
    SCALE: tl.constexpr,
    NUM_LOCAL_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute selected attention without materializing dense logits or probabilities."""
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    head = batch_head % H
    batch = batch_head // H

    offsets_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_d = tl.arange(0, BLOCK_D)
    query_mask = offsets_m < S
    dimension_mask = offsets_d < D
    query = tl.load(
        q_ptr
        + batch * stride_qb
        + head * stride_qh
        + offsets_m[:, None] * stride_qs
        + offsets_d[None, :] * stride_qd,
        mask=query_mask[:, None] & dimension_mask[None, :],
        other=0.0,
    )

    sink = tl.load(sink_ptr + head).to(tl.float32)
    running_max = tl.full((BLOCK_M,), sink, tl.float32)
    running_sum = tl.full((BLOCK_M,), 1.0, tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    completed_blocks = (offsets_m + 1) // COMPRESSION_RATE

    # Selected compressed keys differ per query, so fold them into the online softmax one at a
    # time while retaining the query tile and accumulator in registers.
    for selected_slot in tl.static_range(0, TOPK):
        selected_block = tl.load(
            topk_ptr
            + batch * stride_tb
            + offsets_m * stride_ts
            + selected_slot * stride_tk,
            mask=query_mask,
            other=0,
        )
        valid = (
            query_mask
            & (selected_block >= 0)
            & (selected_block < completed_blocks)
            & (selected_block < N_BLOCKS)
        )
        compressed_value = tl.load(
            compressed_kv_ptr
            + batch * stride_cb
            + head * stride_ch
            + selected_block[:, None] * stride_cn
            + offsets_d[None, :] * stride_cd,
            mask=valid[:, None] & dimension_mask[None, :],
            other=0.0,
        )
        logit = tl.sum(query * compressed_value, axis=1) * SCALE
        logit = tl.where(valid, logit, -float("inf"))
        new_max = tl.maximum(running_max, logit)
        alpha = tl.exp(running_max - new_max)
        probability = tl.exp(logit - new_max)
        accumulator = accumulator * alpha[:, None] + probability[:, None] * compressed_value
        running_sum = running_sum * alpha + probability
        running_max = new_max

    # Adjacent query windows overlap heavily. Process their union in tensor-core-friendly tiles.
    first_local_position = query_block * BLOCK_M - WINDOW + 1
    offsets_n_base = tl.arange(0, BLOCK_N)
    for local_tile in tl.static_range(0, NUM_LOCAL_TILES):
        offsets_n = first_local_position + local_tile * BLOCK_N + offsets_n_base
        local_mask = (offsets_n >= 0) & (offsets_n < S)
        local_values = tl.load(
            local_kv_ptr
            + batch * stride_lb
            + head * stride_lh
            + offsets_n[:, None] * stride_ls
            + offsets_d[None, :] * stride_ld,
            mask=local_mask[:, None] & dimension_mask[None, :],
            other=0.0,
        )
        logits = tl.dot(query, tl.trans(local_values), input_precision="tf32x3") * SCALE
        causal_window_mask = (
            query_mask[:, None]
            & local_mask[None, :]
            & (offsets_n[None, :] <= offsets_m[:, None])
            & (offsets_n[None, :] >= offsets_m[:, None] - WINDOW + 1)
        )
        logits = tl.where(causal_window_mask, logits, -float("inf"))

        tile_max = tl.max(logits, axis=1)
        new_max = tl.maximum(running_max, tile_max)
        alpha = tl.exp(running_max - new_max)
        probabilities = tl.exp(logits - new_max[:, None])
        accumulator *= alpha[:, None]
        accumulator += tl.dot(
            probabilities.to(local_values.dtype), local_values, input_precision="tf32x3"
        )
        running_sum = running_sum * alpha + tl.sum(probabilities, axis=1)
        running_max = new_max

    output = accumulator / running_sum[:, None]
    tl.store(
        out_ptr
        + batch * stride_ob
        + head * stride_oh
        + offsets_m[:, None] * stride_os
        + offsets_d[None, :] * stride_od,
        output,
        mask=query_mask[:, None] & dimension_mask[None, :],
    )


def _launch_selected_attention(
    query: torch.Tensor,
    compressed_kv: torch.Tensor,
    local_kv: torch.Tensor,
    topk_blocks: torch.Tensor,
    attention_sink: torch.Tensor,
    compression_rate: int,
    sliding_window_size: int,
) -> torch.Tensor:
    batch, heads, sequence_length, head_dim = query.shape
    topk = topk_blocks.shape[-1]
    block_m = 64
    block_n = 128
    block_d = max(16, triton.next_power_of_2(head_dim))
    num_local_tiles = triton.cdiv(sliding_window_size + block_m - 1, block_n)
    output = torch.empty_like(query)
    _tiled_selected_attention_fwd[(triton.cdiv(sequence_length, block_m), batch * heads)](
        query,
        compressed_kv,
        local_kv,
        topk_blocks,
        attention_sink,
        output,
        *query.stride(),
        *compressed_kv.stride(),
        *local_kv.stride(),
        *topk_blocks.stride(),
        *output.stride(),
        H=heads,
        S=sequence_length,
        D=head_dim,
        N_BLOCKS=compressed_kv.shape[-2],
        COMPRESSION_RATE=compression_rate,
        TOPK=topk,
        WINDOW=sliding_window_size,
        SCALE=1.0 / math.sqrt(head_dim),
        NUM_LOCAL_TILES=num_local_tiles,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=8,
    )
    return output


def _validate_inputs(
    tensors: tuple[tuple[str, torch.Tensor], ...],
    compression_rate: int,
    num_topk_blocks: int,
    sliding_window_size: int,
    rope_dims: int,
    share_kv: bool,
    *,
    backend_name: str = "Triton",
    max_head_dim: int = 256,
    supported_dtypes: tuple[torch.dtype, ...] = (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ),
) -> None:
    query = tensors[0][1]
    integer_arguments = {
        "compression_rate": compression_rate,
        "num_topk_blocks": num_topk_blocks,
        "sliding_window_size": sliding_window_size,
        "rope_dims": rope_dims,
    }
    for name, value in integer_arguments.items():
        if type(value) is not int:
            raise TypeError(f"{name} must be a Python int, got {type(value).__name__}.")
    if type(share_kv) is not bool:
        raise TypeError(f"share_kv must be a Python bool, got {type(share_kv).__name__}.")
    if query.device.type != "cuda":
        raise ValueError(
            f"The {backend_name} compressed sparse attention backend requires CUDA tensors."
        )
    if query.dtype not in supported_dtypes:
        dtype_names = ", ".join(str(dtype).removeprefix("torch.") for dtype in supported_dtypes)
        raise TypeError(f"The {backend_name} backend supports {dtype_names} tensors.")
    for name, tensor in tensors:
        if tensor.device != query.device:
            raise ValueError(f"{name} must be on {query.device}, got {tensor.device}.")
        if tensor.dtype != query.dtype:
            raise TypeError(f"{name} must have dtype {query.dtype}, got {tensor.dtype}.")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous for the {backend_name} backend.")
    if torch.is_grad_enabled() and any(tensor.requires_grad for _, tensor in tensors):
        raise NotImplementedError(
            f"The {backend_name} compressed sparse attention backend is forward-only."
        )

    if query.ndim != 4:
        raise ValueError("Q must have shape [batch, heads, sequence, head_dim].")
    batch, heads, sequence_length, head_dim = query.shape
    if min(batch, heads, sequence_length, head_dim) <= 0:
        raise ValueError("Q dimensions must all be positive.")
    q_i = tensors[1][1]
    if q_i.ndim != 4 or q_i.shape[0] != batch or q_i.shape[2] != sequence_length:
        raise ValueError("Q_I must have shape [batch, index_heads, sequence, index_dim].")
    index_heads, index_dim = q_i.shape[1], q_i.shape[3]
    if index_heads <= 0 or index_dim <= 0:
        raise ValueError("Q_I index_heads and index_dim must be positive.")
    if head_dim > max_head_dim:
        raise ValueError(
            f"The {backend_name} backend currently supports head_dim <= {max_head_dim}."
        )
    if head_dim % 2 or index_dim % 2:
        raise ValueError("head_dim and index_dim must be even so the RoPE tail is complex-aligned.")
    if rope_dims <= 0 or rope_dims % 2 or rope_dims > min(head_dim, index_dim):
        raise ValueError("rope_dims must be positive, even, and no larger than either head dimension.")
    if compression_rate <= 0:
        raise ValueError("compression_rate must be positive.")
    if num_topk_blocks < 0 or sliding_window_size < 0:
        raise ValueError("num_topk_blocks and sliding_window_size must be non-negative.")

    by_name = dict(tensors)
    expected_kv_heads = (1, heads) if share_kv else (heads,)
    for name in ("KV", "C_a", "C_b", "Z_a", "Z_b"):
        tensor = by_name[name]
        if (
            tensor.ndim != 4
            or tensor.shape[0] != batch
            or tensor.shape[1] not in expected_kv_heads
            or tensor.shape[2:] != (sequence_length, head_dim)
        ):
            raise ValueError(
                f"{name} must have shape [batch, {'1 or heads' if share_kv else 'heads'}, "
                "sequence, head_dim]."
            )

    expected_index_heads = (1, index_heads) if share_kv else (index_heads,)
    for name in ("K_Ia", "K_Ib", "Z_Ia", "Z_Ib"):
        tensor = by_name[name]
        if (
            tensor.ndim != 4
            or tensor.shape[0] != batch
            or tensor.shape[1] not in expected_index_heads
            or tensor.shape[2:] != (sequence_length, index_dim)
        ):
            raise ValueError(
                f"{name} must have shape [batch, "
                f"{'1 or index_heads' if share_kv else 'index_heads'}, sequence, index_dim]."
            )

    expected_shapes = {
        "B_a": (compression_rate, head_dim),
        "B_b": (compression_rate, head_dim),
        "W_I": (batch, sequence_length, index_heads),
        "B_Ia": (compression_rate, index_dim),
        "B_Ib": (compression_rate, index_dim),
        "KV_norm_weight": (head_dim,),
        "compressed_indices_norm_weight": (index_dim,),
        "compressed_kv_norm_weight": (head_dim,),
        "attention_sink": (heads,),
    }
    for name, expected_shape in expected_shapes.items():
        if by_name[name].shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {by_name[name].shape}.")


def _prepare_attention_inputs(
    Q: torch.Tensor,
    Q_I: torch.Tensor,
    KV: torch.Tensor,
    C_a: torch.Tensor,
    C_b: torch.Tensor,
    Z_a: torch.Tensor,
    Z_b: torch.Tensor,
    B_a: torch.Tensor,
    B_b: torch.Tensor,
    W_I: torch.Tensor,
    K_Ia: torch.Tensor,
    K_Ib: torch.Tensor,
    Z_Ia: torch.Tensor,
    Z_Ib: torch.Tensor,
    B_Ia: torch.Tensor,
    B_Ib: torch.Tensor,
    KV_norm_weight: torch.Tensor,
    compressed_indices_norm_weight: torch.Tensor,
    compressed_kv_norm_weight: torch.Tensor,
    attention_sink: torch.Tensor,
    compression_rate: int,
    num_topk_blocks: int,
    sliding_window_size: int,
    rope_dims: int,
    share_kv: bool,
    *,
    backend_name: str = "Triton",
    max_head_dim: int = 256,
    supported_dtypes: tuple[torch.dtype, ...] = (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ),
    expand_attention_kv: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate and prepare the four inputs consumed by the selected-attention kernel."""
    tensors = (
        ("Q", Q),
        ("Q_I", Q_I),
        ("KV", KV),
        ("C_a", C_a),
        ("C_b", C_b),
        ("Z_a", Z_a),
        ("Z_b", Z_b),
        ("B_a", B_a),
        ("B_b", B_b),
        ("W_I", W_I),
        ("K_Ia", K_Ia),
        ("K_Ib", K_Ib),
        ("Z_Ia", Z_Ia),
        ("Z_Ib", Z_Ib),
        ("B_Ia", B_Ia),
        ("B_Ib", B_Ib),
        ("KV_norm_weight", KV_norm_weight),
        ("compressed_indices_norm_weight", compressed_indices_norm_weight),
        ("compressed_kv_norm_weight", compressed_kv_norm_weight),
        ("attention_sink", attention_sink),
    )
    _validate_inputs(
        tensors,
        compression_rate,
        num_topk_blocks,
        sliding_window_size,
        rope_dims,
        share_kv,
        backend_name=backend_name,
        max_head_dim=max_head_dim,
        supported_dtypes=supported_dtypes,
    )

    batch, heads, sequence_length, head_dim = Q.shape
    index_heads, index_dim = Q_I.shape[1], Q_I.shape[3]
    kv_inputs = (KV, C_a, C_b, Z_a, Z_b)
    if share_kv and len({tensor.shape[1] for tensor in kv_inputs}) > 1:
        KV, C_a, C_b, Z_a, Z_b = (
            tensor.expand(-1, heads, -1, -1) for tensor in kv_inputs
        )
    index_inputs = (K_Ia, K_Ib, Z_Ia, Z_Ib)
    if share_kv and len({tensor.shape[1] for tensor in index_inputs}) > 1:
        K_Ia, K_Ib, Z_Ia, Z_Ib = (
            tensor.expand(-1, index_heads, -1, -1) for tensor in index_inputs
        )
    compressed_kv = _compress(C_a, C_b, Z_a, Z_b, B_a, B_b, compression_rate)
    compressed_indices = _compress(
        K_Ia, K_Ib, Z_Ia, Z_Ib, B_Ia, B_Ib, compression_rate
    )
    num_blocks = compressed_kv.shape[-2]

    query = _apply_rope(Q, rope_dims)
    index_query = _apply_rope(Q_I, rope_dims)
    local_kv = F.rms_norm(KV, (head_dim,), weight=KV_norm_weight)
    local_kv = _apply_rope(local_kv, rope_dims)

    compressed_positions = torch.arange(num_blocks, device=Q.device) * compression_rate
    compressed_indices = F.rms_norm(
        compressed_indices, (index_dim,), weight=compressed_indices_norm_weight
    )
    compressed_indices = _apply_rope(
        compressed_indices, rope_dims, positions=compressed_positions
    )
    compressed_kv = F.rms_norm(
        compressed_kv, (head_dim,), weight=compressed_kv_norm_weight
    )
    compressed_kv = _apply_rope(compressed_kv, rope_dims, positions=compressed_positions)

    if compressed_indices.shape[1] == 1 and index_heads != 1:
        compressed_indices = compressed_indices.expand(-1, index_heads, -1, -1)
    scores = F.relu(index_query @ compressed_indices.transpose(-2, -1)) / math.sqrt(
        index_dim * index_heads
    )
    scores = (scores * W_I.permute(0, 2, 1).unsqueeze(-1)).sum(dim=1)
    query_positions = torch.arange(sequence_length, device=Q.device)
    block_positions = torch.arange(num_blocks, device=Q.device)
    completed_blocks = (query_positions + 1) // compression_rate
    scores.masked_fill_(
        block_positions[None, None, :] >= completed_blocks[None, :, None],
        float("-inf"),
    )
    topk = min(num_topk_blocks, num_blocks)
    topk_blocks = torch.topk(scores, k=topk, dim=-1).indices

    if expand_attention_kv and local_kv.shape[1] == 1 and heads != 1:
        local_kv = local_kv.expand(-1, heads, -1, -1)
    if expand_attention_kv and compressed_kv.shape[1] == 1 and heads != 1:
        compressed_kv = compressed_kv.expand(-1, heads, -1, -1)
    return query, compressed_kv, local_kv, topk_blocks


def compressed_sparse_attention(
    Q: torch.Tensor,
    Q_I: torch.Tensor,
    KV: torch.Tensor,
    C_a: torch.Tensor,
    C_b: torch.Tensor,
    Z_a: torch.Tensor,
    Z_b: torch.Tensor,
    B_a: torch.Tensor,
    B_b: torch.Tensor,
    W_I: torch.Tensor,
    K_Ia: torch.Tensor,
    K_Ib: torch.Tensor,
    Z_Ia: torch.Tensor,
    Z_Ib: torch.Tensor,
    B_Ia: torch.Tensor,
    B_Ib: torch.Tensor,
    KV_norm_weight: torch.Tensor,
    compressed_indices_norm_weight: torch.Tensor,
    compressed_kv_norm_weight: torch.Tensor,
    attention_sink: torch.Tensor,
    compression_rate: int,
    num_topk_blocks: int,
    sliding_window_size: int,
    rope_dims: int,
    share_kv: bool,
) -> torch.Tensor:
    """Run the forward-only hybrid Triton CSA implementation."""
    query, compressed_kv, local_kv, topk_blocks = _prepare_attention_inputs(
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
        share_kv,
    )
    output = _launch_selected_attention(
        query,
        compressed_kv,
        local_kv,
        topk_blocks,
        attention_sink,
        compression_rate,
        sliding_window_size,
    )
    return _apply_rope(output, rope_dims, inverse=True)


__all__ = ["compressed_sparse_attention"]
