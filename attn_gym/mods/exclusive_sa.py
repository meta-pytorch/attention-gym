from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.attention.flex_attention import flex_attention
def exclusive_output_mod(Y: Tensor, V: Tensor) -> Tensor:
    """Remove each attention output vector's projection onto its value vector."""
    v_sq_norm = (V * V).sum(dim=-1, keepdim=True)
    y_dot_v = (Y * V).sum(dim=-1, keepdim=True)
    scale = torch.where(
        v_sq_norm > 0,
        y_dot_v / v_sq_norm.clamp_min(torch.finfo(V.dtype).tiny),
        torch.zeros_like(y_dot_v),
    )
    Z = Y - scale * V

    if Z.is_floating_point():
        residual_norm = Z.norm(dim=-1, keepdim=True)
        reference_norm = torch.maximum(
            Y.norm(dim=-1, keepdim=True),
            V.norm(dim=-1, keepdim=True),
        )
        cleanup_tol = torch.finfo(Z.dtype).eps * Z.shape[-1] * reference_norm
        Z = torch.where(
            (v_sq_norm > 0) & (residual_norm <= cleanup_tol),
            torch.zeros_like(Z),
            Z,
        )
    return Z


class XSAMultiheadAttention(nn.Module):
    """Multi-head self-attention with exclusive output projection."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.dropout = dropout

        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(m.weight)

    def _split_heads(self, x: Tensor) -> Tensor:
        """(B, T, D) -> (B, H, T, d_head)."""
        B, T, _ = x.shape
        return x.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2).contiguous()

    def _merge_heads(self, x: Tensor) -> Tensor:
        """(B, H, T, d_head) -> (B, T, D)."""
        B, H, T, d = x.shape
        return x.transpose(1, 2).reshape(B, T, H * d).contiguous()

    def forward(
        self,
        x: Tensor,
        score_mod=None,
        block_mask=None,
        is_causal: bool = True,
    ) -> Tensor:
        Q = self._split_heads(self.W_q(x))
        K = self._split_heads(self.W_k(x))
        V = self._split_heads(self.W_v(x))
        Y = flex_attention(Q, K, V, score_mod=score_mod, block_mask=block_mask)
        Z = exclusive_output_mod(Y, V)
        return self.W_o(self._merge_heads(Z))
