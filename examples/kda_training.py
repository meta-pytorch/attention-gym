"""Train a small KDA attention module on one device.

This is a small `example` KDA attention module. The goal is to show how one might use the primitives(kernels)
we have in attn_gym to build a performant KDA implementation. It roughly follows Kimi's architecture.
There is a reason though that this is in examples/ and not the core package. We want to encourage people
to own their own implementation and surrounding ops. Don't wont short convs then dont add em!

Below is a reference where you can run an unfused vs fused implementation an get profiles
as well as full graph compile. The kernels primarily focus blackwell for the fused implementation.
Like all good things the only thing this training run proves is that it can overfit a fixed target - much success!

Run a reference training step with::

    python examples/kda_training.py --backend=reference

On a Blackwell GPU, exercise the fused backend with::

    python examples/kda_training.py --backend=fused

Add ``--profile`` to export a backend- and shape-named Chrome trace. The trace
contains explicit ``forward`` and ``backward`` record-function ranges. Add
``--compile`` to compile the complete module with
``torch.compile(fullgraph=True)`` and fuse the PyTorch work around the custom
KDA operator.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import nullcontext
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Annotated, Any, Literal, NamedTuple

import torch
import torch.nn.functional as F
import typer
from torch import nn

from attn_gym.linear import naive_chunk_kda_from_cumulative
from attn_gym.linear.kda.fwd.cute import chunk_kda
from attn_gym.linear.kda.fwd.triton.gate_fwd import bounded_gate_cumsum
from attn_gym.linear.kda.fwd.triton.l2norm_fwd import l2norm
from attn_gym.linear.kda.naive import gate_fwd_ref, l2norm_fwd_ref

Backend = Literal["reference", "fused"]


class BackendOption(str, Enum):
    """Implementations exposed by the command line."""

    REFERENCE = "reference"
    FUSED = "fused"


def _record_function(enabled: bool, name: str):
    """Create a profiler range only when profiling is requested."""
    return torch.profiler.record_function(name) if enabled else nullcontext()


def record_function(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Label a KDA stage when the module has profiling ranges enabled."""

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(function)
        def profiled(self: Any, *args: Any, **kwargs: Any) -> Any:
            if not self.profile_ranges:
                return function(self, *args, **kwargs)
            with torch.profiler.record_function(name.format(backend=self.backend)):
                return function(self, *args, **kwargs)

        return profiled

    return decorate


class KDAAttentionOutput(NamedTuple):
    """Hidden states and optional recurrent and convolution states."""

    hidden_states: torch.Tensor
    final_state: torch.Tensor | None
    final_conv_state: torch.Tensor | None


class KDAAttention(nn.Module):
    """Minimal trainable KDA attention with a transformer-style module ABI."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        *,
        chunk_size: int = 64,
        short_conv_kernel_size: int = 4,
        lower_bound: float = -5.0,
        backend: Backend = "reference",
        fastmath: bool = False,
        rms_norm_eps: float = 1e-5,
        compute_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        profile_ranges: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or num_heads < 1 or head_dim < 1:
            raise ValueError("hidden_size, num_heads, and head_dim must be positive")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if short_conv_kernel_size < 1:
            raise ValueError(
                f"short_conv_kernel_size must be positive, got {short_conv_kernel_size}"
            )
        if not math.isfinite(lower_bound) or lower_bound > 0:
            raise ValueError(f"lower_bound must be finite and non-positive, got {lower_bound}")
        if backend not in ("reference", "fused"):
            raise ValueError(f"backend must be 'reference' or 'fused', got {backend!r}")
        if backend == "fused" and (head_dim != 128 or chunk_size != 64):
            raise ValueError("the fused backend requires head_dim=128 and chunk_size=64")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.lower_bound = lower_bound
        self.backend = backend
        self.fastmath = fastmath
        self.rms_norm_eps = rms_norm_eps
        self.profile_ranges = profile_ranges
        self.compute_dtype = (
            (torch.bfloat16 if backend == "fused" else torch.float32)
            if compute_dtype is None
            else compute_dtype
        )
        if backend == "fused" and self.compute_dtype != torch.bfloat16:
            raise ValueError("the fused backend requires compute_dtype=torch.bfloat16")

        projection_size = num_heads * head_dim
        factory_kwargs = {"device": device}
        self.qkv_proj = nn.Linear(hidden_size, 3 * projection_size, bias=False, **factory_kwargs)
        self.qkv_conv1d = nn.Conv1d(
            3 * projection_size,
            3 * projection_size,
            short_conv_kernel_size,
            groups=3 * projection_size,
            bias=False,
            **factory_kwargs,
        )
        self.beta_proj = nn.Linear(hidden_size, num_heads, bias=False, **factory_kwargs)
        self.f_a_proj = nn.Linear(hidden_size, head_dim, bias=False, **factory_kwargs)
        self.f_b_proj = nn.Linear(head_dim, projection_size, bias=False, **factory_kwargs)
        self.g_a_proj = nn.Linear(hidden_size, head_dim, bias=False, **factory_kwargs)
        self.g_b_proj = nn.Linear(head_dim, projection_size, bias=False, **factory_kwargs)
        self.output_norm_weight = nn.Parameter(
            torch.ones(head_dim, device=device, dtype=torch.float32)
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
        initial_conv_state: torch.Tensor | None = None,
        return_final_state: bool = False,
    ) -> KDAAttentionOutput:
        """Apply KDA and optionally return recurrent and short-convolution states."""
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden_states must have shape [B, T, {self.hidden_size}], "
                f"got {tuple(hidden_states.shape)}"
            )
        batch, tokens, _ = hidden_states.shape
        if batch == 0 or tokens == 0:
            raise ValueError("batch size and sequence length must be greater than zero")
        expected_state = (batch, self.num_heads, self.head_dim, self.head_dim)
        if initial_state is not None:
            if initial_state.shape != expected_state:
                raise ValueError(
                    f"initial_state must have shape {expected_state}, got {tuple(initial_state.shape)}"
                )
            initial_state = initial_state.to(device=hidden_states.device, dtype=torch.float32)

        hidden_states_compute, qkv = self.qkv_projection(hidden_states)
        qkv, final_conv_state = self.short_convolution(
            qkv,
            initial_conv_state,
            return_final_state=return_final_state,
        )
        q, k, v = qkv.view(batch, tokens, 3, self.num_heads, self.head_dim).unbind(2)
        q, k = self.qk_normalization(q, k)
        raw_gate, beta = self.gate_projections(hidden_states_compute)
        cumulative_gate = self.gate_prefix_sum(raw_gate)
        output, final_state = self.kda_core(
            q,
            k,
            v,
            cumulative_gate,
            beta,
            initial_state,
            return_final_state=return_final_state,
        )
        output = self.output_normalization(output)
        output_gate = self.output_gate(hidden_states_compute, output)
        output = self.output_projection(output, output_gate)

        return KDAAttentionOutput(
            output.to(hidden_states.dtype),
            final_state,
            final_conv_state,
        )

    @record_function("kda/qkv_projection")
    def qkv_projection(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.to(self.compute_dtype)
        qkv = F.linear(hidden_states, self.qkv_proj.weight.to(self.compute_dtype))
        return hidden_states, qkv

    @record_function("kda/short_convolution")
    def short_convolution(
        self,
        qkv: torch.Tensor,
        initial_state: torch.Tensor | None,
        *,
        return_final_state: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, tokens, channels = qkv.shape
        expected_state = (batch, self.qkv_conv1d.kernel_size[0] - 1, channels)
        if initial_state is None:
            initial_state = qkv.new_zeros(expected_state)
        elif initial_state.shape != expected_state:
            raise ValueError(
                f"initial_conv_state must have shape {expected_state}, "
                f"got {tuple(initial_state.shape)}"
            )
        else:
            initial_state = initial_state.to(device=qkv.device, dtype=qkv.dtype)

        conv_input = torch.cat((initial_state, qkv), dim=1)
        qkv = F.conv1d(
            conv_input.transpose(1, 2),
            self.qkv_conv1d.weight.to(self.compute_dtype),
            groups=channels,
        ).transpose(1, 2)
        qkv = F.silu(qkv)
        final_state = conv_input[:, tokens:].clone() if return_final_state else None
        return qkv, final_state

    @record_function("kda/qk_normalization")
    def qk_normalization(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.backend == "reference":
            return l2norm_fwd_ref(q.float()), l2norm_fwd_ref(k.float())

        return l2norm(q), l2norm(k)

    @record_function("kda/gate_projections")
    def gate_projections(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, tokens, _ = hidden_states.shape
        gate_features = F.linear(
            hidden_states,
            self.f_a_proj.weight.to(self.compute_dtype),
        )
        raw_gate = F.linear(
            gate_features,
            self.f_b_proj.weight.to(self.compute_dtype),
        ).view(batch, tokens, self.num_heads, self.head_dim)
        beta = (
            F.linear(
                hidden_states,
                self.beta_proj.weight.to(self.compute_dtype),
            )
            .view(batch, tokens, self.num_heads)
            .float()
            .sigmoid()
        )
        return raw_gate.contiguous(), beta

    @record_function("kda/gate_prefix_sum/{backend}")
    def gate_prefix_sum(self, raw_gate: torch.Tensor) -> torch.Tensor:
        if self.backend == "reference":
            return gate_fwd_ref(
                raw_gate,
                self.A_log,
                self.dt_bias,
                self.lower_bound,
                math.log2(math.e),
                False,
                self.chunk_size,
                None,
            )

        return bounded_gate_cumsum(
            raw_gate,
            self.A_log,
            self.dt_bias,
            chunk_size=self.chunk_size,
            lower_bound=self.lower_bound,
            fastmath=self.fastmath,
            profile_ranges=self.profile_ranges,
        )

    @record_function("kda/core/{backend}")
    def kda_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cumulative_gate: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None,
        *,
        return_final_state: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.backend == "reference":
            return naive_chunk_kda_from_cumulative(
                q,
                k,
                v.float(),
                cumulative_gate,
                beta,
                initial_state=initial_state,
                output_final_state=return_final_state,
                chunk_size=self.chunk_size,
            )

        return chunk_kda(
            q,
            k,
            v,
            cumulative_gate,
            beta,
            initial_state,
            output_final_state=return_final_state,
            fastmath=self.fastmath,
            profile_ranges=self.profile_ranges,
        )

    @record_function("kda/output_normalization")
    def output_normalization(self, output: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(
            output,
            (self.head_dim,),
            self.output_norm_weight.to(output.dtype),
            eps=self.rms_norm_eps,
        )

    @record_function("kda/output_gate")
    def output_gate(self, hidden_states: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        gate_features = F.linear(
            hidden_states,
            self.g_a_proj.weight.to(self.compute_dtype),
        )
        return (
            F.linear(
                gate_features,
                self.g_b_proj.weight.to(self.compute_dtype),
            )
            .view_as(output)
            .sigmoid()
        )

    @record_function("kda/output_projection")
    def output_projection(self, output: torch.Tensor, output_gate: torch.Tensor) -> torch.Tensor:
        return F.linear(
            (output * output_gate).flatten(-2).to(self.compute_dtype),
            self.out_proj.weight.to(self.compute_dtype),
        )


def main(
    backend: Annotated[
        BackendOption,
        typer.Option(help="Use the reference or the best integrated fused kernels."),
    ] = BackendOption.REFERENCE,
    steps: Annotated[int, typer.Option(min=1, help="Number of optimizer steps.")] = 2,
    batch_size: Annotated[int, typer.Option(min=1, help="Training batch size.")] = 1,
    tokens: Annotated[int, typer.Option(min=1, help="Tokens per sequence.")] = 16384,
    hidden_size: Annotated[int, typer.Option(min=1, help="Transformer hidden size.")] = 2304,
    num_heads: Annotated[int, typer.Option(min=1, help="Number of KDA heads.")] = 32,
    head_dim: Annotated[int, typer.Option(min=1, help="Channels per KDA head.")] = 128,
    chunk_size: Annotated[int, typer.Option(min=1, help="KDA recurrence chunk size.")] = 64,
    short_conv_kernel_size: Annotated[
        int,
        typer.Option(min=1, help="Causal Q/K/V convolution width."),
    ] = 4,
    device: Annotated[str, typer.Option(help="Torch device used for training.")] = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
    profile: Annotated[
        bool,
        typer.Option(help="Export a named Chrome trace with forward/backward ranges."),
    ] = False,
    compile_model: Annotated[
        bool,
        typer.Option("--compile", help="Compile the complete module as one full graph."),
    ] = False,
) -> None:
    """Train the single-device KDA example."""
    torch.manual_seed(0)
    model = KDAAttention(
        hidden_size,
        num_heads,
        head_dim,
        chunk_size=chunk_size,
        short_conv_kernel_size=short_conv_kernel_size,
        backend=backend.value,
        device=device,
        profile_ranges=profile and not compile_model,
    )
    if compile_model:
        model = torch.compile(model, fullgraph=True, mode="reduce-overhead")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    hidden_states = torch.randn(batch_size, tokens, hidden_size, device=device)
    target = torch.randn_like(hidden_states)
    execution_name = "_compiled" if compile_model else ""
    profile_name = (
        f"kda_training_backend-{backend.value}{execution_name}"
        f"_b{batch_size}_t{tokens}_c{hidden_size}_h{num_heads}_d{head_dim}"
    )

    def train_step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        with _record_function(profile, f"{profile_name}/forward"):
            output = model(hidden_states).hidden_states
        with _record_function(profile, f"{profile_name}/loss"):
            loss = F.mse_loss(output.float(), target)
        with _record_function(profile, f"{profile_name}/backward"):
            loss.backward()
        with _record_function(profile, f"{profile_name}/optimizer"):
            optimizer.step()
        return loss

    if profile or compile_model:
        for _ in range(3):
            train_step()
        if hidden_states.is_cuda:
            torch.cuda.synchronize(hidden_states.device)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if hidden_states.is_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    profile_context = torch.profiler.profile(activities=activities) if profile else nullcontext()
    with profile_context as active_profiler:
        for step in range(steps):
            loss = train_step()
            if active_profiler is not None:
                active_profiler.step()
            print(f"step={step} loss={loss.item():.6f}")

    if active_profiler is not None:
        profile_path = Path(f"{profile_name}.json")
        active_profiler.export_chrome_trace(str(profile_path))
        print(f"profile={profile_path.resolve()}")


if __name__ == "__main__":
    typer.run(main)
