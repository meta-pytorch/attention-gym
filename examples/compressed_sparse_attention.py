"""Compose compressed sparse attention from the selected-attention primitive."""

import math

import torch
import torch.nn.functional as F

from attn_gym.sparse.selected_attention import selected_attention


def pad_to_block_size(x: torch.Tensor, m: int, value: float) -> torch.Tensor:
    n = x.shape[-2]
    pad_length = (-n) % m
    if pad_length == 0:
        return x
    return F.pad(x, (0, 0, 0, pad_length), mode="constant", value=value)


def _split_blocks(x: torch.Tensor, compression_rate: int) -> torch.Tensor:
    """Split the sequence dimension into block and within-block dimensions."""
    return x.reshape(
        *x.shape[:-2],
        x.shape[-2] // compression_rate,
        compression_rate,
        x.shape[-1],
    )


def compress(C_a, C_b, Z_a, Z_b, B_a, B_b, compression_rate):
    C_a = pad_to_block_size(C_a, compression_rate, 0.0)
    C_b = pad_to_block_size(C_b, compression_rate, 0.0)
    Z_a = pad_to_block_size(Z_a, compression_rate, float("-inf"))
    Z_b = pad_to_block_size(Z_b, compression_rate, float("-inf"))

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


def make_block_mask(query_length, num_blocks, compression_rate, device, dtype):
    query_positions = torch.arange(query_length, device=device)
    block_positions = torch.arange(num_blocks, device=device)
    completed_blocks = (query_positions + 1) // compression_rate
    bool_mask = block_positions[None, :] < completed_blocks[:, None]
    mask = torch.zeros(bool_mask.shape, device=bool_mask.device, dtype=dtype)
    return mask.masked_fill(~bool_mask, float("-inf"))


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

    if positions is None:
        positions = torch.arange(
            position_offset,
            position_offset + sequence_length,
            device=x.device,
            dtype=x.dtype,
        )
    else:
        positions = positions.to(device=x.device, dtype=x.dtype)

    frequencies = 1.0 / (
        base
        ** (
            torch.arange(
                0,
                rotary_dim,
                2,
                device=x.device,
                dtype=x.dtype,
            )
            / rotary_dim
        )
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

        ramp = (
            torch.arange(
                rotary_dim // 2,
                device=x.device,
                dtype=x.dtype,
            )
            - low
        ) / (high - low)
        smooth = 1 - ramp.clamp(0, 1)
        frequencies = frequencies / factor * (1 - smooth) + frequencies * smooth

    angles = torch.outer(positions, frequencies)
    frequencies_complex = torch.polar(torch.ones_like(angles), angles)
    if inverse:
        frequencies_complex = frequencies_complex.conj()

    x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], rotary_dim // 2, 2))
    frequencies_complex = frequencies_complex.view(
        *([1] * (x.ndim - 2)),
        sequence_length,
        rotary_dim // 2,
    )
    return torch.view_as_real(x_complex * frequencies_complex).flatten(-2)


def _selected_attention_with_causal_blocks(
    Q,
    KV,
    compressed_kv,
    topk_blocks,
    indexer_mask,
    attention_sink,
    sliding_window_size,
):
    """Call selected_attention while preserving the completed-block constraint."""
    if topk_blocks.shape[-1] == 0:
        return selected_attention(
            Q,
            KV,
            compressed_kv,
            topk_blocks,
            attention_sink,
            None,
            sliding_window_size,
            False,
        )

    valid_blocks = torch.isfinite(indexer_mask).unsqueeze(0).expand(Q.shape[0], -1, -1)
    selected_is_valid = valid_blocks.gather(dim=-1, index=topk_blocks)

    # scatter() treats duplicate indices as one selection. Replacing an invalid
    # top-k entry with any valid entry therefore reproduces the original additive mask.
    first_valid_slot = selected_is_valid.to(torch.int64).argmax(dim=-1, keepdim=True)
    first_valid_block = topk_blocks.gather(dim=-1, index=first_valid_slot)
    causal_topk_blocks = torch.where(selected_is_valid, topk_blocks, first_valid_block)

    selected_output = selected_attention(
        Q,
        KV,
        compressed_kv,
        causal_topk_blocks,
        attention_sink,
        None,
        sliding_window_size,
        False,
    )

    # Before the first compressed block is complete there is no valid replacement.
    # An empty index tensor keeps the compressed branch present but fully masked.
    local_output = selected_attention(
        Q,
        KV,
        compressed_kv,
        topk_blocks[..., :0],
        attention_sink,
        None,
        sliding_window_size,
        False,
    )
    has_selected_block = selected_is_valid.any(dim=-1)[:, None, :, None]
    return torch.where(has_selected_block, selected_output, local_output)


def CSA(
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
    _, num_heads, sequence_length, _ = Q.shape
    _, num_index_heads, _, index_head_dim = Q_I.shape
    if share_kv:
        KV = KV.expand(-1, num_heads, -1, -1)
        C_a = C_a.expand(-1, num_heads, -1, -1)
        C_b = C_b.expand(-1, num_heads, -1, -1)
        Z_a = Z_a.expand(-1, num_heads, -1, -1)
        Z_b = Z_b.expand(-1, num_heads, -1, -1)

        K_Ia = K_Ia.expand(-1, num_index_heads, -1, -1)
        K_Ib = K_Ib.expand(-1, num_index_heads, -1, -1)
        Z_Ia = Z_Ia.expand(-1, num_index_heads, -1, -1)
        Z_Ib = Z_Ib.expand(-1, num_index_heads, -1, -1)

    compressed_kv = compress(C_a, C_b, Z_a, Z_b, B_a, B_b, compression_rate)
    compressed_indices = compress(K_Ia, K_Ib, Z_Ia, Z_Ib, B_Ia, B_Ib, compression_rate)
    num_total_blocks = compressed_kv.shape[-2]

    Q = torch.cat([Q[..., :-rope_dims], apply_rope(Q[..., -rope_dims:])], dim=-1)
    Q_I = torch.cat([Q_I[..., :-rope_dims], apply_rope(Q_I[..., -rope_dims:])], dim=-1)
    KV = torch.cat([KV[..., :-rope_dims], apply_rope(KV[..., -rope_dims:])], dim=-1)

    compressed_positions = torch.arange(num_total_blocks, device=device) * compression_rate
    compressed_indices = torch.cat(
        [
            compressed_indices[..., :-rope_dims],
            apply_rope(
                compressed_indices[..., -rope_dims:],
                positions=compressed_positions,
            ),
        ],
        dim=-1,
    )
    compressed_kv = torch.cat(
        [
            compressed_kv[..., :-rope_dims],
            apply_rope(
                compressed_kv[..., -rope_dims:],
                positions=compressed_positions,
            ),
        ],
        dim=-1,
    )

    indexer_mask = make_block_mask(
        sequence_length,
        num_total_blocks,
        compression_rate,
        device,
        dtype,
    )
    indexer_scale = math.sqrt(index_head_dim * num_index_heads)
    scores = F.relu(Q_I @ compressed_indices.transpose(-2, -1)) / indexer_scale
    index_head_weights = W_I.transpose(1, 2).unsqueeze(-1)
    scores = torch.sum(index_head_weights * scores, dim=1) + indexer_mask

    topk_blocks = torch.topk(
        scores,
        k=min(num_topk_blocks, num_total_blocks),
        dim=-1,
    ).indices

    attention_output = _selected_attention_with_causal_blocks(
        Q,
        KV,
        compressed_kv,
        topk_blocks,
        indexer_mask,
        attention_sink,
        sliding_window_size,
    )

    return torch.cat(
        [
            attention_output[..., :-rope_dims],
            apply_rope(attention_output[..., -rope_dims:], inverse=True),
        ],
        dim=-1,
    )


def main() -> None:
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    batch_size = 1
    num_heads = 2
    num_index_heads = 1
    sequence_length = 5
    head_dim = 4
    index_head_dim = 4

    compression_rate = 2
    num_topk_blocks = 1
    sliding_window_size = 2
    rope_dims = 4
    share_kv = True
    num_blocks = (sequence_length + compression_rate - 1) // compression_rate

    def randn(*shape):
        return torch.randn(*shape, device=device, dtype=dtype)

    Q = randn(batch_size, num_heads, sequence_length, head_dim)
    Q_I = randn(batch_size, num_index_heads, sequence_length, index_head_dim)

    # Shared KV inputs start with one head and are expanded by CSA.
    KV = randn(batch_size, 1, sequence_length, head_dim)
    C_a = randn(batch_size, 1, sequence_length, head_dim)
    C_b = randn(batch_size, 1, sequence_length, head_dim)
    Z_a = randn(batch_size, 1, sequence_length, 1)
    Z_b = randn(batch_size, 1, sequence_length, 1)
    B_a = randn(1, num_heads, num_blocks, 1, 1)
    B_b = randn(1, num_heads, num_blocks, 1, 1)

    W_I = randn(batch_size, sequence_length, num_index_heads)
    K_Ia = randn(batch_size, 1, sequence_length, index_head_dim)
    K_Ib = randn(batch_size, 1, sequence_length, index_head_dim)
    Z_Ia = randn(batch_size, 1, sequence_length, 1)
    Z_Ib = randn(batch_size, 1, sequence_length, 1)
    B_Ia = randn(1, num_index_heads, num_blocks, 1, 1)
    B_Ib = randn(1, num_index_heads, num_blocks, 1, 1)

    KV_norm_weight = torch.ones(head_dim, device=device, dtype=dtype)
    compressed_indices_norm_weight = torch.ones(
        index_head_dim,
        device=device,
        dtype=dtype,
    )
    compressed_kv_norm_weight = torch.ones(head_dim, device=device, dtype=dtype)
    attention_sink = torch.zeros(num_heads, device=device, dtype=dtype)

    output = CSA(
        Q=Q,
        Q_I=Q_I,
        KV=KV,
        C_a=C_a,
        C_b=C_b,
        Z_a=Z_a,
        Z_b=Z_b,
        B_a=B_a,
        B_b=B_b,
        W_I=W_I,
        K_Ia=K_Ia,
        K_Ib=K_Ib,
        Z_Ia=Z_Ia,
        Z_Ib=Z_Ib,
        B_Ia=B_Ia,
        B_Ib=B_Ib,
        KV_norm_weight=KV_norm_weight,
        compressed_indices_norm_weight=compressed_indices_norm_weight,
        compressed_kv_norm_weight=compressed_kv_norm_weight,
        attention_sink=attention_sink,
        compression_rate=compression_rate,
        num_topk_blocks=num_topk_blocks,
        sliding_window_size=sliding_window_size,
        rope_dims=rope_dims,
        share_kv=share_kv,
    )

    print("Output shape:", output.shape)
    print(output)


if __name__ == "__main__":
    main()
