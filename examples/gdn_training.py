"""Train a small GDN (gated delta net) attention module on one device.

This is the GDN sibling of ``examples/kda_training.py``: a minimal Qwen3-next-style
attention block built from the attn_gym primitives, trained to overfit a fixed target.
Grouped heads are first class, with fewer q/k heads than value heads accepted by both
backends. Like all good things, the only thing this training run proves is that it can
overfit a fixed target - much success!

Run a reference training step with::

    python examples/gdn_training.py --backend=reference

On a Blackwell GPU, exercise the fused backend (the CuTe KDA chunk pipeline driven through
``chunk_gdn``) with::

    python examples/gdn_training.py --backend=fused

Add ``--compile`` to compile the module with ``torch.compile(fullgraph=True)``.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Annotated, Literal, NamedTuple

import torch
import torch.nn.functional as F
import typer
from torch import nn

from attn_gym.linear import causal_conv1d, chunk_gdn, l2norm

Backend = Literal["reference", "fused"]


class BackendOption(str, Enum):
    """Implementations exposed by the command line."""

    REFERENCE = "reference"
    FUSED = "fused"


class GDNAttentionOutput(NamedTuple):
    hidden_states: torch.Tensor
    final_state: torch.Tensor | None


class GDNAttention(nn.Module):
    """Minimal trainable GDN attention with grouped q/k heads.

    The block follows the Qwen3-next shape: one projection produces the packed
    grouped-head QKV activations, a depthwise causal convolution smooths them, the gate is
    ``-exp(A_log) * softplus(a + dt_bias)`` from a per-head projection, the write gate is
    ``sigmoid(beta)``, and q/k are L2-normalized before the delta-rule core.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_heads: int,
        head_dim: int,
        *,
        short_conv_kernel_size: int = 4,
        backend: Backend = "reference",
        rms_norm_eps: float = 1e-5,
        compute_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or num_heads < 1 or num_key_heads < 1 or head_dim < 1:
            raise ValueError("hidden_size, head counts, and head_dim must be positive")
        if num_heads % num_key_heads != 0:
            raise ValueError(
                f"num_heads must be a multiple of num_key_heads, "
                f"got {num_heads} and {num_key_heads}"
            )
        if backend not in ("reference", "fused"):
            raise ValueError(f"backend must be 'reference' or 'fused', got {backend!r}")
        if backend == "fused" and head_dim != 128:
            raise ValueError("the fused backend requires head_dim=128")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_heads = num_key_heads
        self.head_dim = head_dim
        self.backend = backend
        self.rms_norm_eps = rms_norm_eps
        self.compute_dtype = (
            (torch.bfloat16 if backend == "fused" else torch.float32)
            if compute_dtype is None
            else compute_dtype
        )

        value_size = num_heads * head_dim
        key_size = num_key_heads * head_dim
        conv_channels = 2 * key_size + value_size
        factory_kwargs = {"device": device}
        self.qkv_proj = nn.Linear(hidden_size, conv_channels, bias=False, **factory_kwargs)
        self.qkv_conv1d = nn.Conv1d(
            conv_channels,
            conv_channels,
            short_conv_kernel_size,
            groups=conv_channels,
            bias=False,
            **factory_kwargs,
        )
        self.gate_proj = nn.Linear(hidden_size, num_heads, bias=False, **factory_kwargs)
        self.beta_proj = nn.Linear(hidden_size, num_heads, bias=False, **factory_kwargs)
        self.A_log = nn.Parameter(torch.zeros(num_heads, device=device, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.zeros(num_heads, device=device, dtype=torch.float32))
        self.output_norm_weight = nn.Parameter(
            torch.ones(head_dim, device=device, dtype=torch.float32)
        )
        self.out_proj = nn.Linear(value_size, hidden_size, bias=False, **factory_kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        *,
        return_final_state: bool = False,
    ) -> GDNAttentionOutput:
        """Apply GDN attention and optionally return the final recurrent state."""
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden_states must have shape [B, T, {self.hidden_size}], "
                f"got {tuple(hidden_states.shape)}"
            )
        batch, tokens, _ = hidden_states.shape
        key_size = self.num_key_heads * self.head_dim

        compute_states = hidden_states.to(self.compute_dtype)
        packed_qkv = F.linear(compute_states, self.qkv_proj.weight.to(self.compute_dtype))
        packed_qkv = causal_conv1d(
            packed_qkv,
            self.qkv_conv1d.weight[:, 0].to(packed_qkv.dtype),
            activation="silu",
        )
        q, k, v = packed_qkv.split((key_size, key_size, self.num_heads * self.head_dim), dim=-1)
        q = l2norm(q.view(batch, tokens, self.num_key_heads, self.head_dim))
        k = l2norm(k.view(batch, tokens, self.num_key_heads, self.head_dim))
        v = v.view(batch, tokens, self.num_heads, self.head_dim)

        gate = -self.A_log.float().exp() * F.softplus(
            F.linear(compute_states, self.gate_proj.weight.to(self.compute_dtype)).float()
            + self.dt_bias
        )
        beta = torch.sigmoid(
            F.linear(compute_states, self.beta_proj.weight.to(self.compute_dtype)).float()
        )

        core_output, final_state = chunk_gdn(
            q,
            k,
            v,
            gate,
            beta,
            initial_state,
            output_final_state=return_final_state,
            impl=self.backend,
        )
        normalized = F.rms_norm(
            core_output.float(),
            (self.head_dim,),
            weight=self.output_norm_weight,
            eps=self.rms_norm_eps,
        ).to(self.compute_dtype)
        output = F.linear(normalized.flatten(-2), self.out_proj.weight.to(self.compute_dtype))
        return GDNAttentionOutput(output.to(hidden_states.dtype), final_state)


def training_loop(
    *,
    backend: Backend,
    batch_size: int,
    tokens: int,
    hidden_size: int,
    num_heads: int,
    num_key_heads: int,
    head_dim: int,
    steps: int,
    learning_rate: float,
    compile_module: bool,
) -> list[float]:
    """Overfit a fixed random target and return the per-step losses."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    module = GDNAttention(
        hidden_size,
        num_heads,
        num_key_heads,
        head_dim,
        backend=backend,
        device=device,
    )
    forward = torch.compile(module, fullgraph=True) if compile_module else module
    optimizer = torch.optim.AdamW(module.parameters(), lr=learning_rate)
    hidden_states = torch.randn(batch_size, tokens, hidden_size, device=device)
    target = torch.randn_like(hidden_states)

    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = forward(hidden_states).hidden_states
        loss = F.mse_loss(output.float(), target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def main(
    backend: Annotated[BackendOption, typer.Option(help="Core implementation.")] = (
        BackendOption.REFERENCE
    ),
    batch_size: Annotated[int, typer.Option(min=1)] = 2,
    tokens: Annotated[int, typer.Option(min=1)] = 256,
    hidden_size: Annotated[int, typer.Option(min=1)] = 512,
    num_heads: Annotated[int, typer.Option(min=1)] = 12,
    num_key_heads: Annotated[int, typer.Option(min=1)] = 4,
    head_dim: Annotated[int, typer.Option(min=1)] = 128,
    steps: Annotated[int, typer.Option(min=1)] = 25,
    learning_rate: Annotated[float, typer.Option(min=0.0)] = 3e-3,
    compile_module: Annotated[bool, typer.Option("--compile")] = False,
) -> None:
    """Overfit a fixed target with a single GDN attention block."""
    losses = training_loop(
        backend=backend.value,
        batch_size=batch_size,
        tokens=tokens,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_key_heads=num_key_heads,
        head_dim=head_dim,
        steps=steps,
        learning_rate=learning_rate,
        compile_module=compile_module,
    )
    for step, loss in enumerate(losses):
        if step % 5 == 0 or step == len(losses) - 1:
            print(f"step {step:>3}  loss {loss:.6f}")
    if not math.isfinite(losses[-1]):
        raise SystemExit("training diverged")
    print(f"loss {losses[0]:.4f} -> {losses[-1]:.4f} over {len(losses)} steps")


if __name__ == "__main__":
    typer.run(main)
