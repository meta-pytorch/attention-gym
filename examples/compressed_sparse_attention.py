"""
Compose compressed sparse attention from the selected-attention primitive.
Implementation of compressed sparse attention from here: https://arxiv.org/html/2606.19348v1
"""

import math

import torch
import torch.nn.functional as F

from attn_gym.sparse.selected_attention import selected_attention


def pad_to_block_size(x: torch.Tensor, m: int, value: float) -> torch.Tensor:
    """
    Pads a tensor such that the tensor's sequence_length % m == 0
    """
    n = x.shape[-2]
    pad_length = (-n) % m
    if pad_length == 0:
        return x
    return F.pad(x, (0, 0, 0, pad_length), mode="constant", value=value)


def _split_blocks(x: torch.Tensor, compression_rate: int) -> torch.Tensor:
    """
    Chunk the sequence into blocks as mentioned in the paper.
    Args:
        x: tensor in shape of (batch, heads, sequence_length, dim)
        compression_rate: integer, size of each block
    Returns:
        tensor in shape of (batch, heads, sequence_length/compression_rate, compression_rate, dim)
    """
    return x.reshape(
        *x.shape[:-2],
        x.shape[-2] // compression_rate,
        compression_rate,
        x.shape[-1],
    )


def compress(C_a, C_b, Z_a, Z_b, B_a, B_b, compression_rate):
    """
    Function to compress the non-sliding window KV into blocks
    Formula is from the paper.
    Args:
        C_a, C_b, Z_a, Z_b, B_a, B_b: Tensors from CSA inputs
        compression_rate: int, size of each block (and also how much they're compressed by)
    Returns:
        Weighted, tensor in shape of (batch, num_heads, sequence_length / compression_rate, head_dim)
    """
    # Pad everything to be evenly divisible by block size
    C_a = pad_to_block_size(C_a, compression_rate, 0.0)
    C_b = pad_to_block_size(C_b, compression_rate, 0.0)
    Z_a = pad_to_block_size(Z_a, compression_rate, float("-inf"))
    Z_b = pad_to_block_size(Z_b, compression_rate, float("-inf"))
    # Shift C_b, Z_b one block to the right and pad / index accordingly
    C_b = F.pad(C_b, (0, 0, compression_rate, 0), "constant", 0.0)[:, :, :-compression_rate, :]
    Z_b = F.pad(Z_b, (0, 0, compression_rate, 0), "constant", float("-inf"))[
        :, :, :-compression_rate, :
    ]
    # Reshape tensors into blocks
    Z_a = _split_blocks(Z_a, compression_rate)
    Z_b = _split_blocks(Z_b, compression_rate)
    C_a = _split_blocks(C_a, compression_rate)
    C_b = _split_blocks(C_b, compression_rate)
    # Add biases, perform a softmax, and split out the blocks
    logits = torch.cat([Z_a + B_a, Z_b + B_b], dim=-2)
    logits_normalized = F.softmax(logits, dim=-2)
    S_a = logits_normalized[:, :, :, :compression_rate, :]
    S_b = logits_normalized[:, :, :, compression_rate:, :]
    # Perform a weighted multiplication and reduce to compress information
    weighted = C_a * S_a + C_b * S_b
    return torch.sum(weighted, dim=-2)


def make_block_mask(query_length, num_blocks, compression_rate, device, dtype):
    """
    Masks out causally invalid blocks (queries can only attend to blocks created by KV values before them)
    Args:
        query_length: int, length of query
        num_blocks: int, total number of blocks
        compression_rate: int, size of each block
        device: device to create mask on
        dtype: dtype of mask
    Returns:
        Mask in shape of (query_length, num_blocks), with entries of -inf or 0
    """
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
    """
    Applies YaRN as mentioned in the inference code on huggingface here:
    https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/model.py
    """
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
    query,
    local_kv,
    sparse_kv,
    topk_blocks,
    indexer_mask,
    attention_sink,
    sliding_window_size,
):
    """Call selected_attention while preserving the completed-block constraint.

    Invalid (causally unavailable) selections are replaced with -1 sentinels.
    """

    valid_blocks = torch.isfinite(indexer_mask).unsqueeze(0).expand(query.shape[0], -1, -1)
    selected_is_valid = valid_blocks.gather(dim=-1, index=topk_blocks)

    # Replace causally invalid selections with -1 sentinel
    causal_topk_blocks = torch.where(selected_is_valid, topk_blocks, -1)

    backend = "triton" if query.device.type == "cuda" else "eager"
    return selected_attention(
        query,
        local_kv,
        sparse_kv,
        causal_topk_blocks,
        attention_sink,
        None,
        sliding_window_size,
        backend=backend,
    )


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
    """
    Naming of args uses convention from DeepSeek V4 paper

    Shape notation: B=batch size, H=attention heads, H_I=indexer heads, S=sequence length,
    D=attention dimension, D_I=index dimension, R=compression_rate,
    H_KV=H and H_IKV=H_I when share_kv=False. When share_kv=True, those dimensions
    may instead be 1.

    Normalization is applied within the function for KV vectors, but not for Q or Q_I
    All rope is applied within the function
    This is because Q and Q_I are expected to be projected from the same normalized latent
    Q: Query vector for attention; expected to be normalized beforehand; expected shape: (B, H, S, D)
    Q_I: Query vector for indexing; expected to be normalized beforehand; expected shape: (B, H_I, S, D_I)

    KV: Projection from residual stream; expected shape: (B, H_KV, S, D)
    C_a, C_b: Projections from the residual stream that will be attended to; each expected shape: (B, H_KV, S, D)
    Z_a, Z_b: Projections from the residual stream that weight C_a and C_b; each expected shape: (B, H_KV, S, D)
    B_a, B_b: Learnable per-position-within-block biases (Eq 11 in the paper). Added to Z_a, Z_b
        after reshaping into blocks. Expected shape: any broadcastable to (B, H_KV, num_blocks, R, D)
        — typically (R, D) or (1, 1, 1, R, D).

    W_I: Projection from the residual stream, per-head weight on indexer scores (Batch, sequence, num_heads); expected shape: (B, S, H_I)
    K_Ia, K_Ib: Projections from the residual stream for computing indexing; each expected shape: (B, H_IKV, S, D_I)
    Z_Ia, Z_Ib: Projections from the residual stream, perform similar role to Z_a and Z_b for indexing; each expected shape: (B, H_IKV, S, D_I)
    B_Ia, B_Ib: Same role as B_a/B_b but for the indexing branch. Expected shape: any broadcastable
        to (B, H_IKV, num_blocks, R, D_I) — typically (R, D_I) or (1, 1, 1, R, D_I).

    KV_norm_weight: RMS norm weights for KV; expected shape: (D,)
    compressed_indices_norm_weight: RMS weights for compressed indices; expected shape: (D_I,)
    compressed_kv_norm_weight: RMS norm weights for compressed blocks; expected shape: (D,)

    attention_sink: Learned weight in shape of (num_heads, ), functions as attention sink; expected shape: (H,)

    compression_rate: size of each compressed block (the paper's m). Block interleaving means
        each compressed entry draws from 2m tokens, but the compression ratio is m:1.
    num_topk_blocks: number of blocks to attend to per query
    sliding_window_size: size of sliding window for SWA
    rope_dims: number of dimensions to apply rope to
    share_kv: True if all query heads attend to one kv head
    """

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
    KV = F.rms_norm(KV, (KV.shape[-1],), weight=KV_norm_weight)
    KV = torch.cat([KV[..., :-rope_dims], apply_rope(KV[..., -rope_dims:])], dim=-1)

    compressed_positions = torch.arange(num_total_blocks, device=device) * compression_rate
    compressed_indices = F.rms_norm(
        compressed_indices, (compressed_indices.shape[-1],), weight=compressed_indices_norm_weight
    )
    # Apply rope to the last rope_dim dimensions
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
    compressed_kv = F.rms_norm(
        compressed_kv, (compressed_kv.shape[-1],), weight=compressed_kv_norm_weight
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

    attention_output, _ = _selected_attention_with_causal_blocks(
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


def indexer_loss(
    main_query: torch.Tensor,
    selected_compressed_kv: torch.Tensor,
    attention_lse: torch.Tensor,
    selected_indexer_logits: torch.Tensor,
    attention_sink: torch.Tensor | None,
    softmax_scale: float,
) -> torch.Tensor:
    """Computes the auxilary indexer loss Deepseek used in their paper. Takes the KL divergence between the attention 
    
    Args:
        main_query: (B, H, S, D) — main attention queries (detached).
        selected_compressed_kv: (B, S, K, D) — the K compressed keys selected by
            the indexer for each query position (detached).
        attention_lse: (B, H, S) — log-sum-exp returned by selected_attention.
            This includes the sliding window, sparse, AND sink contributions.
        selected_indexer_logits: (B, S, K) — raw logits the indexer produced for
            the K selected keys.
        attention_sink: (H,) or None
        softmax_scale: float — the 1/sqrt(d) scale used in the attention computation.

    Returns:
        Scalar KL-divergence loss (mean over batch and sequence).
    """
    # Recompute main-attention logits for selected compressed keys.
    # Selected attention doesn't store an attention matrix, so we have to do this
    compressed_attention_logits = (
        torch.einsum(
            "bhsd,bskd->bhsk",
            main_query.detach(),
            selected_compressed_kv.detach(),
        )
        * softmax_scale
    )

    full_attention_lse = attention_lse  # (B, H, S)
    if attention_sink is not None:
        full_attention_lse = torch.logaddexp(
            full_attention_lse,
            attention_sink[None, :, None],  # (1, H, 1)
        )

    per_head_compressed_probs = torch.exp(
        compressed_attention_logits - full_attention_lse[..., None]
    )
    compressed_teacher_mass = per_head_compressed_probs.sum(dim=1)
    # Normalize so we have a valid kl divergence
    eps = torch.finfo(torch.float32).tiny
    compressed_teacher_probs = (
        compressed_teacher_mass / compressed_teacher_mass.sum(dim=-1, keepdim=True).clamp_min(eps)
    ).detach()

    indexer_probs = F.softmax(selected_indexer_logits.float(), dim=-1)
    # Return KL
    return (
        (
            compressed_teacher_probs
            * (compressed_teacher_probs.clamp_min(eps).log() - indexer_probs.clamp_min(eps).log())
        )
        .sum(dim=-1)
        .mean()
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
