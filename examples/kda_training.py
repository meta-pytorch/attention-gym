"""Train a small KDA attention module on one device.

This example shows the module boundary expected by a transformer block:
``[B, T, hidden_size]`` hidden states enter and tensors with the same shape
leave. ``gate_backward="torch"`` uses ordinary PyTorch autograd.
``gate_backward="cute"`` keeps the reference KDA forward but uses the fused
CuTeDSL reverse-cumsum plus bounded-gate kernel during backward.

The module is an integration example, not a checkpoint-compatible Kimi K3
implementation. A checkpoint adapter would still need K3's short convolution,
factorized projections, exact gated RMSNorm, parameter names/layouts, and
parallelism. Those details are deliberately separate from the KDA recurrence
and its autograd boundary shown here.

Run a reference training step with::

    python examples/kda_training.py --gate-backward=torch

On an SM90-or-newer CUDA GPU with CuTeDSL installed, exercise the fused leaf
with::

    python examples/kda_training.py --gate-backward=cute
"""

from __future__ import annotations

import math
from typing import Literal, NamedTuple

import torch
import torch.nn.functional as F
from jsonargparse import ArgumentParser
from torch import nn

from attn_gym.linear import naive_chunk_kda_from_cumulative
from attn_gym.linear.kda.naive import gate_fwd_ref

GateBackward = Literal["torch", "cute"]


class KDAAttentionOutput(NamedTuple):
    """Hidden states and an optional recurrent KDA state."""

    hidden_states: torch.Tensor
    final_state: torch.Tensor | None


class _CuteBoundedGateCumsum(torch.autograd.Function):
    """Reference gate forward with the fused CuTeDSL first-order backward."""

    @staticmethod
    def forward(
        ctx,
        raw_gate: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        chunk_size: int,
        lower_bound: float,
        fastmath: bool,
    ) -> torch.Tensor:
        if not raw_gate.is_cuda or raw_gate.dtype != torch.bfloat16:
            raise TypeError("the CuTe gate backward requires a CUDA bfloat16 raw gate")
        if A_log.dtype != torch.float32 or dt_bias.dtype != torch.float32:
            raise TypeError("the CuTe gate backward requires float32 A_log and dt_bias")
        if torch.cuda.get_device_capability(raw_gate.device) < (9, 0):
            raise ValueError("the CuTe gate backward requires CUDA capability 9.0 or newer")
        ctx.save_for_backward(raw_gate, A_log, dt_bias)
        ctx.chunk_size = chunk_size
        ctx.lower_bound = lower_bound
        ctx.fastmath = fastmath
        return gate_fwd_ref(
            raw_gate,
            A_log,
            dt_bias,
            lower_bound,
            math.log2(math.e),
            False,
            chunk_size,
            None,
        )

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, d_cumulative: torch.Tensor):
        raw_gate, A_log, dt_bias = ctx.saved_tensors
        from attn_gym.linear.kda.bwd.cute.gate_bwd_fused import fused_gate_bwd

        result = fused_gate_bwd(
            raw_gate,
            A_log,
            dt_bias,
            d_cumulative.float().contiguous(),
            chunk_size=ctx.chunk_size,
            lower_bound=ctx.lower_bound,
            fastmath=ctx.fastmath,
        )
        d_raw_gate = result.dg.to(raw_gate.dtype)
        dA_log = result.dA_partial.sum((0, 1))
        d_dt_bias = result.dg.sum((0, 1))
        return d_raw_gate, dA_log, d_dt_bias, None, None, None


class KDAAttention(nn.Module):
    """Minimal trainable KDA attention with a transformer-style module ABI."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        *,
        chunk_size: int = 64,
        lower_bound: float = -5.0,
        gate_backward: GateBackward = "torch",
        fastmath: bool = False,
        rms_norm_eps: float = 1e-5,
        compute_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or num_heads < 1 or head_dim < 1:
            raise ValueError("hidden_size, num_heads, and head_dim must be positive")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if gate_backward not in ("torch", "cute"):
            raise ValueError(f"gate_backward must be 'torch' or 'cute', got {gate_backward!r}")
        if gate_backward == "cute" and (head_dim < 32 or head_dim > 1024 or head_dim % 32):
            raise ValueError(
                "the CuTe gate backward requires head_dim divisible by 32 in [32, 1024]"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.lower_bound = lower_bound
        self.gate_backward = gate_backward
        self.fastmath = fastmath
        self.rms_norm_eps = rms_norm_eps
        self.compute_dtype = (
            (torch.bfloat16 if gate_backward == "cute" else torch.float32)
            if compute_dtype is None
            else compute_dtype
        )
        if gate_backward == "cute" and self.compute_dtype != torch.bfloat16:
            raise ValueError("the CuTe gate backward requires compute_dtype=torch.bfloat16")

        projection_size = num_heads * head_dim
        factory_kwargs = {"device": device}
        self.qkv_proj = nn.Linear(hidden_size, 3 * projection_size, bias=False, **factory_kwargs)
        self.gate_proj = nn.Linear(hidden_size, projection_size, bias=False, **factory_kwargs)
        self.beta_proj = nn.Linear(hidden_size, num_heads, bias=False, **factory_kwargs)
        self.output_gate_proj = nn.Linear(
            hidden_size,
            projection_size,
            bias=False,
            **factory_kwargs,
        )
        self.out_proj = nn.Linear(projection_size, hidden_size, bias=False, **factory_kwargs)
        self.A_log = nn.Parameter(torch.zeros(num_heads, device=device, dtype=torch.float32))
        self.dt_bias = nn.Parameter(
            torch.zeros(num_heads, head_dim, device=device, dtype=torch.float32)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        *,
        return_final_state: bool = False,
    ) -> KDAAttentionOutput:
        """Apply KDA and optionally return state ``[B, H, D, D]``."""
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden_states must have shape [B, T, {self.hidden_size}], "
                f"got {tuple(hidden_states.shape)}"
            )
        batch, tokens, _ = hidden_states.shape
        if batch == 0 or tokens == 0:
            raise ValueError("batch size and sequence length must be greater than zero")

        hidden_states_compute = hidden_states.to(self.compute_dtype)
        qkv = F.linear(hidden_states_compute, self.qkv_proj.weight.to(self.compute_dtype))
        q, k, v = qkv.view(batch, tokens, 3, self.num_heads, self.head_dim).unbind(2)
        q = F.normalize(q.float(), dim=-1)
        k = F.normalize(k.float(), dim=-1)
        raw_gate = F.linear(
            hidden_states_compute,
            self.gate_proj.weight.to(self.compute_dtype),
        ).view(
            batch,
            tokens,
            self.num_heads,
            self.head_dim,
        )
        beta = (
            F.linear(
                hidden_states_compute,
                self.beta_proj.weight.to(self.compute_dtype),
            )
            .view(batch, tokens, self.num_heads)
            .sigmoid()
        )
        raw_gate = raw_gate.contiguous()
        if self.gate_backward == "torch":
            cumulative_gate = gate_fwd_ref(
                raw_gate,
                self.A_log,
                self.dt_bias,
                self.lower_bound,
                math.log2(math.e),
                False,
                self.chunk_size,
                None,
            )
        else:
            cumulative_gate = _CuteBoundedGateCumsum.apply(
                raw_gate,
                self.A_log,
                self.dt_bias,
                self.chunk_size,
                self.lower_bound,
                self.fastmath,
            )
        # The optimized ``chunk_kda_fwd_intra`` consumes this same cumulative
        # gate, but currently exposes forward intermediates rather than a
        # complete autograd operator. Keep the consumer reference and swap it
        # once the optimized forward and backward pipelines are composed.
        output, final_state = naive_chunk_kda_from_cumulative(
            q,
            k,
            v.float(),
            cumulative_gate,
            beta,
            initial_state=initial_state,
            output_final_state=return_final_state,
            chunk_size=self.chunk_size,
        )
        output = F.rms_norm(
            output,
            (self.head_dim,),
            eps=self.rms_norm_eps,
        )
        output_gate = (
            F.linear(
                hidden_states_compute,
                self.output_gate_proj.weight.to(self.compute_dtype),
            )
            .view_as(output)
            .sigmoid()
        )
        output = F.linear(
            (output * output_gate).flatten(-2).to(self.compute_dtype),
            self.out_proj.weight.to(self.compute_dtype),
        )
        return KDAAttentionOutput(output.to(hidden_states.dtype), final_state)


def train(
    gate_backward: GateBackward,
    steps: int,
    batch_size: int,
    tokens: int,
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    chunk_size: int,
    device: str,
) -> None:
    """Run a few optimizer steps as an executable usage example."""
    torch.manual_seed(0)
    model = KDAAttention(
        hidden_size,
        num_heads,
        head_dim,
        chunk_size=chunk_size,
        gate_backward=gate_backward,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    hidden_states = torch.randn(batch_size, tokens, hidden_size, device=device)
    target = torch.randn_like(hidden_states)

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(hidden_states).hidden_states
        loss = F.mse_loss(output.float(), target)
        loss.backward()
        optimizer.step()
        print(f"step={step} loss={loss.item():.6f}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gate-backward", choices=("torch", "cute"), default="torch")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train(
        args.gate_backward,
        args.steps,
        args.batch_size,
        args.tokens,
        args.hidden_size,
        args.num_heads,
        args.head_dim,
        args.chunk_size,
        args.device,
    )
