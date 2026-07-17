import torch
import torch.nn.functional as F
from einops import rearrange, repeat

def pad_to_block_size(x: torch.Tensor, m: int, value: float):
    n = x.shape[-2] # sequence dimension
    pad_length = (-n)%m
    if pad_length == 0:
        return x
    return F.pad(x, (0, 0, 0, pad_length), mode="constant", value = value)



def compress(C_a, C_b, Z_a, Z_b, B_a, B_b, compression_rate):
    C_a = pad_to_block_size(C_a, compression_rate, 0.0)
    C_b = pad_to_block_size(C_b, compression_rate, 0.0)
    Z_a = pad_to_block_size(Z_a, compression_rate, float("-inf"))
    Z_b = pad_to_block_size(Z_b, compression_rate, float("-inf"))

    # Pad for interleaving (add 0s to the front and cut off back)
    C_b = F.pad(C_b, (0, 0, compression_rate, 0), "constant", 0.0)[:, :, :-compression_rate, :]
    Z_b = F.pad(Z_b, (0, 0, compression_rate, 0), "constant", float("-inf"))[:, :, :-compression_rate, :]

    # Rearrange everything to batch, heads, num_blocks, block_size, head_dim format
    Z_a = rearrange(Z_a, "b h (num_blocks c) d -> b h num_blocks c d", c = compression_rate)
    Z_b = rearrange(Z_b, "b h (num_blocks c) d -> b h num_blocks c d", c = compression_rate)
    C_a = rearrange(C_a, "b h (num_blocks c) d -> b h num_blocks c d", c = compression_rate)
    C_b = rearrange(C_b, "b h (num_blocks c) d -> b h num_blocks c d", c = compression_rate)

    logits = torch.cat([Z_a + B_a, Z_b + B_b], dim = -2)
    
    logits_normalized = F.softmax(logits, dim=-2)
    S_a = logits_normalized[:, :, :, :compression_rate, :]
    S_b = logits_normalized[:, :, :, compression_rate:, :]
    
    
    weighted = torch.multiply(C_a, S_a) + torch.multiply(C_b, S_b)
    compressed = torch.sum(weighted, dim = -2)

    return compressed

def make_block_mask(query_length, num_blocks, compression_rate, device, dtype):
    query_positions = torch.arange(query_length, device=device)
    block_positions = torch.arange(num_blocks, device = device)
    completed_blocks = (query_positions + 1) // compression_rate
    bool_mask = (block_positions[None, :] < completed_blocks[:, None])
    mask = torch.zeros(bool_mask.shape, device = bool_mask.device, dtype=dtype)
    mask = mask.masked_fill(~bool_mask, float("-inf"))
    return mask

def make_sliding_window_mask(query_length, window_size, device, dtype): 
    query_positions = torch.arange(query_length, device=device)[:, None]
    key_positions = torch.arange(query_length, device=device)[None, :]

    valid = (
        (key_positions <= query_positions)
        & (key_positions >= query_positions - window_size + 1)
    )

    return torch.zeros(
        (query_length, query_length),
        device=device,
        dtype=dtype,
    ).masked_fill(~valid, float("-inf"))
    

def sink_softmax(x, sink, dim):
    # (b, h, s)
    sink = sink[None, :, None, None] 
    maximums = torch.max(x, dim = dim, keepdim=True).values
    maximums = torch.maximum(maximums, sink)
    x = x - maximums
    # Also (b, h, s)
    sink = sink - maximums
    x = torch.exp(x)
    return x / (torch.sum(x, dim, keepdim=True) + torch.exp(sink))

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
            dtype=torch.float32,
        )
    else:
        positions = positions.to(
            device=x.device,
            dtype=torch.float32,
        )

    frequencies = 1.0 / (
        base ** (
            torch.arange(
                0,
                rotary_dim,
                2,
                device=x.device,
                dtype=torch.float32,
            )
            / rotary_dim
        )
    )

    if original_seq_len > 0:
        def correction_dimension(num_rotations):
            return (
                rotary_dim
                * math.log(
                    original_seq_len
                    / (num_rotations * 2 * math.pi)
                )
                / (2 * math.log(base))
            )

        low = math.floor(correction_dimension(beta_fast))
        high = math.ceil(correction_dimension(beta_slow))

        low = max(low, 0)
        high = min(high, rotary_dim - 1)

        if low == high:
            high += 0.001

        ramp = (
            torch.arange(
                rotary_dim // 2,
                device=x.device,
                dtype=torch.float32,
            )
            - low
        ) / (high - low)

        smooth = 1 - ramp.clamp(0, 1)

        frequencies = (
            frequencies / factor * (1 - smooth)
            + frequencies * smooth
        )

    angles = torch.outer(positions, frequencies)

    frequencies_complex = torch.polar(
        torch.ones_like(angles),
        angles,
    )

    if inverse:
        frequencies_complex = frequencies_complex.conj()

    x_complex = torch.view_as_complex(
        x.float().reshape(
            *x.shape[:-1],
            rotary_dim // 2,
            2,
        )
    )

    frequencies_complex = frequencies_complex.view(
        *([1] * (x.ndim - 2)),
        sequence_length,
        rotary_dim // 2,
    )

    rotated = torch.view_as_real(
        x_complex * frequencies_complex
    ).flatten(-2)

    return rotated.to(x.dtype)



def CSA(Q, Q_I, KV, C_a, C_b, Z_a, Z_b, B_a, B_b, W_I, K_Ia, K_Ib, Z_Ia, Z_Ib, B_Ia, B_Ib, KV_norm_weight, compressed_indices_norm_weight, compressed_kv_norm_weight, attention_sink, compression_rate, num_topk_blocks, sliding_window_size, rope_dims: int, share_kv: bool):
    '''
    Naming of args uses convention from Deepseek v4 paper

    Rope is applied within the function 
    Q: Query vector for attention; expected to be normalized beforehand
    Q_I: Query vector for indexing; expected to be normalized beforehand

    KV: Projection from residual stream, represents K and V vectors for SWA
    C_a, C_b: Projections from the residual stream that will be attended to
    Z_a, Z_b: Projections from the residual stream

    W_I: Projection from the residual stream, weight on indexer scores (Batch, sequence, num_heads)
    K_Ia, K_Ib: Projections from the residual stream for computing indexing
    Z_Ia, Z_Ib: Projections from the residual stream 

    num_topk_blocks: number of blocks to attend to per query
    sink: Learned weight in shape of (num_heads, )
    '''
    device = Q.device
    dtype = Q.dtype
    b, h, s, head_dim = Q.shape
    _, h_I, s, head_dim_I = Q_I.shape
    # Pad to fix block size issues
    if share_kv:
        KV = KV.expand(-1, h, -1, -1)
        C_a = C_a.expand(-1, h, -1, -1)
        C_b =C_b.expand(-1, h, -1, -1)
        Z_a = Z_a.expand(-1, h, -1, -1)
        Z_b = Z_b.expand(-1, h, -1, -1)

        K_Ia = K_Ia.expand(-1, h_I, -1, -1)
        K_Ib = K_Ib.expand(-1, h_I, -1, -1)
        Z_Ia = Z_Ia.expand(-1, h_I, -1, -1)
        Z_Ib = Z_Ib.expand(-1, h_I, -1, -1)


    compressed_kv = compress(C_a, C_b, Z_a, Z_b, B_a, B_b, compression_rate)
    compressed_indices = compress(K_Ia, K_Ib, Z_Ia, Z_Ib, B_Ia, B_Ib, compression_rate)
    num_total_blocks = compressed_kv.shape[-2]

    
    Q = torch.cat([Q[:, :, :, :-rope_dims], apply_rope(Q[:, :, :, -rope_dims:])], dim = -1)
    Q_I = torch.cat([Q_I[:, :, :, :-rope_dims], apply_rope(Q_I[:, :, :, -rope_dims:])], dim = -1)
    KV = F.rms_norm(KV, (KV.shape[-1], ), weight = KV_norm_weight)
    KV = torch.cat([KV[:, :, :, :-rope_dims], apply_rope(KV[:, :, :, -rope_dims:])], dim = -1)

    compressed_indices = F.rms_norm(compressed_indices, (compressed_indices.shape[-1], ), weight = compressed_indices_norm_weight)
    compressed_positions = torch.arange(num_total_blocks, device = device) * compression_rate
    compressed_indices = torch.cat([compressed_indices[:, :, :, :-rope_dims], apply_rope(compressed_indices[:, :, :, -rope_dims:], positions = compressed_positions)], dim = -1)

    compressed_kv = F.rms_norm(compressed_kv, (compressed_kv.shape[-1], ), weight = compressed_kv_norm_weight)
    compressed_kv = torch.cat([compressed_kv[:, :, :, :-rope_dims], apply_rope(compressed_kv[:, :, :, -rope_dims:], positions = compressed_positions)], dim = -1)


    
    indexer_mask = make_block_mask(s, num_total_blocks, compression_rate, device, dtype)
    indexer_scale = (head_dim_I * h_I) ** 0.5
    scores = F.relu(Q_I @ torch.permute(compressed_indices, (0, 1, 3, 2))) / indexer_scale
    # Reshape W_I into (batch, num_heads, sequence, 1)
    W_I = torch.permute(W_I, (0, 2, 1)).unsqueeze(-1) 
    # Reduce over head dim, shape is (b, s, num_topk_blocks)
    scores = torch.sum(torch.multiply(W_I, scores), dim = 1) + indexer_mask

    # Now we have (b, s, num_topk_blocks)
    topk_scores, topk_blocks = torch.topk(scores, k=min(num_topk_blocks, num_total_blocks), dim=-1)
    topk_mask = torch.full(scores.shape, float("-inf"), device =device, dtype = dtype)
    topk_mask.scatter_(dim=-1, index = topk_blocks, value = 0.0)
    topk_mask += indexer_mask # Don't violate causality
    SWA_mask = make_sliding_window_mask(s, sliding_window_size, device, dtype).unsqueeze(0)
    SWA_mask = repeat(SWA_mask, "1 s1 s2 -> b s1 s2", b=b)
    
    attention_kv = torch.cat([compressed_kv, KV], dim = -2)
    # Add in the num_heads dim
    attention_mask = torch.cat([topk_mask, SWA_mask], dim = -1).unsqueeze(1)
    scale = head_dim ** 0.5
    
    P = sink_softmax((torch.matmul(Q, torch.permute(attention_kv, (0, 1, 3, 2))))/ scale + attention_mask, attention_sink, dim = -1)
    attn_output = P @ attention_kv
    attn_output = torch.cat([attn_output[..., :-rope_dims], apply_rope(
        attn_output[..., -rope_dims:],
        inverse=True,
    )], dim = -1)
    return attn_output





    







if __name__ == "__main__":
    #print(pad_to_block_size(torch.zeros(1, 1, 17, 1), 8, 1)[0])
    b, h, s, head_dim = 17, 5, 19, 64
    num_indexer_heads = 2
    compression_rate = 4
    indexer_dim = 8
    num_topk_blocks = 2
    sliding_window_size = 3
    rope_dims = 2

    Q = torch.randn(b, h, s, head_dim)
    Q_I = torch.randn(b, num_indexer_heads, s, indexer_dim)
    KV = torch.randn(b, h, s, head_dim)
    
    C_a = torch.randn(b, h, s, head_dim)
    C_b = torch.randn(b, h, s, head_dim)
    Z_a = torch.randn(b, h, s, head_dim)
    Z_b = torch.randn(b, h, s, head_dim)
    B_a = torch.randn(compression_rate, head_dim)
    B_b = torch.randn(compression_rate, head_dim)

    W_I = torch.randn(b, s, num_indexer_heads) # Computed from H @ W, where W has shape (residual_dim, num_heads)
    K_Ia = torch.randn(b, num_indexer_heads, s, indexer_dim)
    K_Ib = torch.randn(b, num_indexer_heads, s, indexer_dim)
    Z_Ia = torch.randn(b, num_indexer_heads, s, indexer_dim)
    Z_Ib = torch.randn(b, num_indexer_heads, s, indexer_dim)
    B_Ia = torch.randn(compression_rate, indexer_dim)
    B_Ib = torch.randn(compression_rate, indexer_dim)

    KV_norm_weight = torch.randn(head_dim)
    compressed_indices_norm_weight = torch.randn(indexer_dim)
    compressed_kv_norm_weight = torch.randn(head_dim)


    attention_sink = torch.randn(h)

    out = CSA(Q, Q_I, KV, C_a, C_b, Z_a, Z_b, B_a, B_b, W_I, K_Ia, K_Ib, Z_Ia, Z_Ib, B_Ia, B_Ib, KV_norm_weight, compressed_indices_norm_weight, compressed_kv_norm_weight, attention_sink, compression_rate, num_topk_blocks, sliding_window_size, rope_dims, share_kv = False)
    
    Q = torch.randn(b, h, s, head_dim)
    Q.requires_grad = True
    Q_I = torch.randn(b, num_indexer_heads, s, indexer_dim)
    KV = torch.randn(b, 1, s, head_dim)
    C_a = torch.randn(b, 1, s, head_dim)
    C_b = torch.randn(b, 1, s, head_dim)
    Z_a = torch.randn(b, 1, s, head_dim)
    Z_b = torch.randn(b, 1, s, head_dim)
    B_a = torch.randn(compression_rate, head_dim)
    B_b = torch.randn(compression_rate, head_dim)

    W_I = torch.randn(b, s, num_indexer_heads) # Computed from H @ W, where W has shape (residual_dim, num_heads)
    K_Ia = torch.randn(b, 1, s, indexer_dim)
    K_Ib = torch.randn(b, 1, s, indexer_dim)
    Z_Ia = torch.randn(b, 1, s, indexer_dim)
    Z_Ib = torch.randn(b, 1, s, indexer_dim)
    B_Ia = torch.randn(compression_rate, indexer_dim)
    B_Ib = torch.randn(compression_rate, indexer_dim)
    KV_norm_weight = torch.randn(head_dim)
    compressed_indices_norm_weight = torch.randn(indexer_dim)
    compressed_kv_norm_weight = torch.randn(head_dim)

    
    attention_sink = torch.randn(h)
    out = CSA(Q, Q_I, KV, C_a, C_b, Z_a, Z_b, B_a, B_b, W_I, K_Ia, K_Ib, Z_Ia, Z_Ib, B_Ia, B_Ib, KV_norm_weight, compressed_indices_norm_weight, compressed_kv_norm_weight, attention_sink, compression_rate, num_topk_blocks, sliding_window_size, rope_dims, share_kv = True)
    loss = out.sum()
    loss.backward()