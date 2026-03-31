"""Build a custom FlexAttention-like op from the forward and backward primitives.

We recently added the ability to invoke the flex-attention backward HOP in
isolation. This is a lower level API with more edge cases but useful when you
need full control over the backward pass.

This example implements pseudo-ring attention: chunked forward with online
softmax merging, then an explicit backward using the merged LSE. For local
attention like this you could simply use ``return_aux=AuxRequest(lse=True)``
and let autograd handle the merging through the differentiable LSE. Here we
do it explicitly to show how to build up a custom op from the primitives —
the same pattern you would use in real ring attention where K/V chunks arrive
via communication and autograd cannot orchestrate the backward.
"""

import math

import torch
from torch._dynamo import config as dynamo_config
from torch.nn.attention.flex_attention import (
    AuxRequest,
    create_block_mask,
    flex_attention,
)

# Workaround: dynamo by default skips guards on constant function defaults (like the
# _offset capture in mask_mod closures), which causes stale compiled masks when the
# same function is recompiled with different default values across loop iterations.
dynamo_config.skip_guards_on_constant_func_defaults = False


LN2 = math.log(2)


def _merge_attention(
    out: torch.Tensor,
    lse: torch.Tensor,
    new_out: torch.Tensor,
    new_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    lse = lse.unsqueeze(-1)
    new_lse = new_lse.unsqueeze(-1)
    max_lse = torch.maximum(lse, new_lse)
    exp_lse = torch.exp(lse - max_lse)
    exp_new_lse = torch.exp(new_lse - max_lse)
    denom = exp_lse + exp_new_lse
    return (
        (out * exp_lse + new_out * exp_new_lse) / denom,
        (max_lse + torch.log(denom)).squeeze(-1),
    )


merge_attention = torch.compile(_merge_attention, fullgraph=True)


def _build_chunks(q, k, v, num_chunks):
    B, H, S, _ = q.shape
    kv_len = k.shape[2]
    chunk_size = kv_len // num_chunks
    chunks = []
    for step in range(num_chunks):
        kv_offset = step * chunk_size
        kv_end = kv_len if step == num_chunks - 1 else kv_offset + chunk_size
        cur_chunk_len = kv_end - kv_offset

        def mask_mod(b, h, q_idx, kv_idx, _offset=kv_offset):
            return q_idx >= kv_idx + _offset

        block_mask = torch.compile(create_block_mask, fullgraph=True)(
            mask_mod, B=B, H=H, Q_LEN=S, KV_LEN=cur_chunk_len, device=q.device
        )
        chunks.append(
            (
                k[:, :, kv_offset:kv_end],
                v[:, :, kv_offset:kv_end],
                block_mask,
            )
        )
    return chunks


def _flex_chunk(q, k, v, block_mask, scale):
    out, aux = flex_attention(
        q,
        k,
        v,
        block_mask=block_mask,
        scale=scale,
        return_aux=AuxRequest(lse=True),
    )
    return out, aux.lse


def _identity_score_mod(score, b, h, q_idx, kv_idx):
    return score


def _flex_bw_chunk(q, k, v, fwd_out, lse, grad_out, block_mask, scale):
    return torch.ops.higher_order.flex_attention_backward(
        q,
        k,
        v,
        fwd_out,
        lse,
        grad_out,
        None,
        _identity_score_mod,
        None,
        block_mask,
        scale,
        {},
        (),
        (),
    )


# ---------------------------------------------------------------------------
# Forward: flex_attention per chunk under no_grad, merge (out, lse) online.
# Backward: feed the single merged LSE into flex_attention_backward per chunk.
# ---------------------------------------------------------------------------


class PseudoRingAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, num_chunks):
        scale = q.shape[-1] ** -0.5
        chunks = _build_chunks(q, k, v, num_chunks)

        merged_out = merged_lse = None
        for k_chunk, v_chunk, block_mask in chunks:
            chunk_out, chunk_lse = torch.compile(_flex_chunk, fullgraph=True)(
                q, k_chunk, v_chunk, block_mask, scale
            )
            if merged_out is None:
                merged_out, merged_lse = chunk_out, chunk_lse
            else:
                merged_out, merged_lse = merge_attention(
                    merged_out, merged_lse, chunk_out, chunk_lse
                )

        ctx.save_for_backward(q, k, v, merged_out, merged_lse)
        ctx.num_chunks = num_chunks
        ctx.scale = scale
        return merged_out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, merged_out, merged_lse = ctx.saved_tensors
        scale = ctx.scale
        chunks = _build_chunks(q, k, v, ctx.num_chunks)

        merged_lse_log2 = merged_lse / LN2

        dq = torch.zeros_like(q)
        dk_parts = []
        dv_parts = []

        for k_chunk, v_chunk, block_mask in chunks:
            dq_i, dk_i, dv_i, _ = torch.compile(_flex_bw_chunk, fullgraph=True)(
                q,
                k_chunk,
                v_chunk,
                merged_out,
                merged_lse_log2,
                grad_out,
                block_mask.as_tuple(),
                scale,
            )
            dq += dq_i
            dk_parts.append(dk_i)
            dv_parts.append(dv_i)

        dk = torch.cat(dk_parts, dim=2)
        dv = torch.cat(dv_parts, dim=2)
        return dq, dk, dv, None


def pseudo_ring_attention(q, k, v, num_chunks=2):
    return PseudoRingAttention.apply(q, k, v, num_chunks)


# ---------------------------------------------------------------------------
# Reference: standard flex_attention with a causal mask
# ---------------------------------------------------------------------------


def reference_attention(q, k, v):
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    block_mask = torch.compile(create_block_mask, fullgraph=True)(
        causal_mask,
        B=q.shape[0],
        H=q.shape[1],
        Q_LEN=q.shape[2],
        KV_LEN=k.shape[2],
        device=q.device,
    )

    @torch.compile(fullgraph=True)
    def run(q, k, v, bm, scale):
        return flex_attention(q, k, v, block_mask=bm, scale=scale)

    return run(q, k, v, block_mask, q.shape[-1] ** -0.5)


# ---------------------------------------------------------------------------
# Test: forward + backward through the autograd function vs. reference
# ---------------------------------------------------------------------------


def main():
    torch.manual_seed(0)
    B, H, S, D = 1, 4, 262_144, 128
    NUM_CHUNKS = 4
    device = "cuda"
    dtype = torch.float32

    q = torch.testing.make_tensor((B, H, S, D), device=device, dtype=dtype, requires_grad=True)
    k = torch.testing.make_tensor((B, H, S, D), device=device, dtype=dtype, requires_grad=True)
    v = torch.testing.make_tensor((B, H, S, D), device=device, dtype=dtype, requires_grad=True)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    # --- Custom autograd op (chunked ring attention) ---
    out = pseudo_ring_attention(q, k, v, NUM_CHUNKS)

    # --- Reference (standard flex_attention) ---
    ref_out = reference_attention(q_ref, k_ref, v_ref)

    torch.testing.assert_close(out, ref_out, atol=1e-4, rtol=1e-4)
    print(f"Forward PASS  (num_chunks={NUM_CHUNKS})")

    # --- Backward: autograd drives our custom backward automatically ---
    grad = torch.randn_like(out)
    out.backward(grad)
    ref_out.backward(grad)

    torch.testing.assert_close(q.grad, q_ref.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=1e-3, rtol=1e-3)
    print("Backward PASS (dQ, dK, dV all match reference via autograd)")


if __name__ == "__main__":
    main()
