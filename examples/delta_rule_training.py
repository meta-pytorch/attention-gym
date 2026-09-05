"""Train a small delta-rule attention module (KDA or GDN) on one device.

This is a small `example` delta-rule attention module. The goal is to show how one might use the
primitives(kernels) we have in attn_gym to build a performant implementation. The ``kda`` variant
roughly follows Kimi's architecture; the ``gdn`` variant follows the Gated DeltaNet / Qwen3-Next
recipe. They share every stage except the gate: KDA learns a bounded per-channel decay
(``bound_gate`` on ``[B, T, H, D]``) while GDN learns one softplus decay per head (``[B, T, H]``).
There is a reason though that this is in examples/ and not the core package. We want to encourage people
to own their own implementation and surrounding ops. Don't wont short convs then dont add em!

Below is a reference where you can run an unfused vs fused implementation and get profiles
as well as full graph compile. The fused training path supports Hopper and Blackwell.
Like all good things the only thing this training run proves is that it can overfit a fixed target - much success!

Run a reference training step with::

    python examples/delta_rule_training.py --backend=reference

On a Hopper or Blackwell GPU, exercise the fused backend with::

    python examples/delta_rule_training.py --backend=fused

Train the GDN variant instead of KDA with::

    python examples/delta_rule_training.py --variant=gdn --backend=fused

Run the fused training loop in FP16 with::

    python examples/delta_rule_training.py --backend=fused --compute-dtype=float16

Pack Zipf-distributed sequence lengths into one physical batch with::

    python examples/delta_rule_training.py --backend=fused --packed --batch-size=4 --tokens=256

In packed mode, ``batch-size`` is the number of logical sequences and ``tokens``
is the longest sequence. Add ``--profile`` to export a backend- and shape-named
Chrome trace. The trace
contains explicit ``forward`` and ``backward`` record-function ranges. Add
``--compile`` to compile the complete module with
``torch.compile(fullgraph=True)`` and fuse the PyTorch work around the custom
delta-rule operator.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from enum import Enum
from functools import wraps
from importlib import import_module
from importlib.util import find_spec
from itertools import accumulate
from pathlib import Path
from typing import Annotated, Any, Literal, NamedTuple

import torch
import torch.nn.functional as F
import typer
from torch import nn

from attn_gym.linear.gdn import chunk_gdn
from attn_gym.linear.kda import (
    MAX_GATE_LOWER_BOUND_MAGNITUDE,
    active_token_mask,
    bound_gate,
    causal_conv1d,
    chunk_kda,
    l2norm,
    mask_inactive_token_gradients,
    mask_inactive_tokens,
)

Backend = Literal["reference", "fused"]
Variant = Literal["kda", "gdn"]


class VariantOption(str, Enum):
    """Delta-rule recipes exposed by the command line."""

    KDA = "kda"
    GDN = "gdn"


class BackendOption(str, Enum):
    """Implementations exposed by the command line."""

    REFERENCE = "reference"
    FUSED = "fused"


class ComputeDTypeOption(str, Enum):
    """Low-precision compute dtypes exposed by the command line."""

    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


def _record_function(enabled: bool, name: str):
    """Create a profiler range only when profiling is requested."""
    return torch.profiler.record_function(name) if enabled else nullcontext()


@contextmanager
def _profile_trace(
    enabled: bool,
    path: Path,
    activities: list[torch.profiler.ProfilerActivity],
) -> Iterator[torch.profiler.profile | None]:
    """Export an enhanced trace when transformer-nuggets is installed."""
    if not enabled:
        yield None
        return

    if find_spec("transformer_nuggets") is not None:
        profiler = import_module("transformer_nuggets.utils.benchmark").profiler
        with profiler(path, record_shapes=True, trace_format="chrome_json") as active_profiler:
            yield active_profiler
        return

    with torch.profiler.profile(activities=activities) as active_profiler:
        yield active_profiler
    active_profiler.export_chrome_trace(str(path))


def mark_kernels(*args: Any, **kwargs: Any):
    """Load optional CUDA Graph annotations only when they are requested."""
    from torch.cuda.graph_annotations import mark_kernels as annotate

    return annotate(*args, **kwargs)


def annotate_kernels(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Label one stage during annotation-enabled CUDA Graph capture."""

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(function)
        def annotated(self: Any, *args: Any, **kwargs: Any) -> Any:
            if not self.enable_graph_annotations:
                return function(self, *args, **kwargs)
            with mark_kernels(name.format(variant=self.variant, backend=self.backend)):
                return function(self, *args, **kwargs)

        return annotated

    return decorate


class DeltaRuleAttentionOutput(NamedTuple):
    """Hidden states and optional recurrent and convolution states."""

    hidden_states: torch.Tensor
    final_state: torch.Tensor | None
    final_conv_state: torch.Tensor | None


def packed_sequence_metadata(
    num_sequences: int,
    max_tokens: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Sample token-level sequence lengths from a truncated Zipf distribution."""
    if max_tokens < 1:
        raise ValueError("packed tokens must include at least one token")
    weights = torch.arange(1, max_tokens + 1, dtype=torch.float64).reciprocal()
    lengths = tuple(torch.multinomial(weights, num_sequences, replacement=True).add(1).tolist())
    return lengths, (0, *accumulate(lengths))


class DeltaRuleAttention(nn.Module):
    """Minimal trainable delta-rule attention with a transformer-style module ABI.

    ``variant="kda"`` learns a bounded per-channel log decay through ``f_a_proj``/``f_b_proj``
    and ``bound_gate``; ``variant="gdn"`` learns one softplus log decay per head through
    ``a_proj``. ``lower_bound`` and ``fastmath`` apply to the KDA gate and core only.

    Set ``mask_inactive_capacity=True`` when a packed input reserves physical rows
    beyond ``cu_seqlens[-1]``. The endpoint may then change across CUDA Graph replay
    without changing the physical shape. Leave the option disabled for dense or
    exact-packed inputs to avoid masking work.

    Set ``enable_graph_annotations=True`` only when calling the module inside
    ``torch.cuda.graph(..., enable_annotations=True)``. Leave it disabled for
    ``torch.compile`` because ``mark_kernels`` is intentionally not traceable.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        *,
        variant: Variant = "kda",
        short_conv_kernel_size: int = 4,
        lower_bound: float = -5.0,
        backend: Backend = "reference",
        fastmath: bool = False,
        rms_norm_eps: float = 1e-5,
        compute_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        enable_graph_annotations: bool = False,
        mask_inactive_capacity: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or num_heads < 1 or head_dim < 1:
            raise ValueError("hidden_size, num_heads, and head_dim must be positive")
        if short_conv_kernel_size < 1:
            raise ValueError(
                f"short_conv_kernel_size must be positive, got {short_conv_kernel_size}"
            )
        if backend not in ("reference", "fused"):
            raise ValueError(f"backend must be 'reference' or 'fused', got {backend!r}")
        if variant not in ("kda", "gdn"):
            raise ValueError(f"variant must be 'kda' or 'gdn', got {variant!r}")
        if not math.isfinite(lower_bound) or lower_bound > 0.0:
            raise ValueError(f"lower_bound must be finite and nonpositive, got {lower_bound}")
        if backend == "fused" and lower_bound < -MAX_GATE_LOWER_BOUND_MAGNITUDE:
            raise ValueError(
                f"the fused backend requires lower_bound >= "
                f"{-MAX_GATE_LOWER_BOUND_MAGNITUDE:.3f}, got {lower_bound}"
            )
        if backend == "fused" and head_dim != 128:
            raise ValueError("the fused backend requires head_dim=128")
        if fastmath and (backend == "reference" or variant == "gdn"):
            raise ValueError("fastmath applies only to variant='kda' with backend='fused'")

        self.variant = variant
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.lower_bound = lower_bound
        self.backend = backend
        self.fastmath = fastmath
        self.rms_norm_eps = rms_norm_eps
        self.enable_graph_annotations = enable_graph_annotations
        self.mask_inactive_capacity = mask_inactive_capacity
        self.compute_dtype = (
            (torch.bfloat16 if backend == "fused" else torch.float32)
            if compute_dtype is None
            else compute_dtype
        )
        if backend == "fused" and self.compute_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("the fused backend requires compute_dtype float16 or bfloat16")

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
        if variant == "kda":
            self.f_a_proj = nn.Linear(hidden_size, head_dim, bias=False, **factory_kwargs)
            self.f_b_proj = nn.Linear(head_dim, projection_size, bias=False, **factory_kwargs)
            dt_bias_shape = (num_heads, head_dim)
        else:
            self.a_proj = nn.Linear(hidden_size, num_heads, bias=False, **factory_kwargs)
            dt_bias_shape = (num_heads,)
        self.g_a_proj = nn.Linear(hidden_size, head_dim, bias=False, **factory_kwargs)
        self.g_b_proj = nn.Linear(head_dim, projection_size, bias=False, **factory_kwargs)
        self.output_norm_weight = nn.Parameter(
            torch.ones(head_dim, device=device, dtype=torch.float32)
        )
        self.out_proj = nn.Linear(projection_size, hidden_size, bias=False, **factory_kwargs)
        self.A_log = nn.Parameter(torch.zeros(num_heads, device=device, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.zeros(dt_bias_shape, device=device, dtype=torch.float32))

    def forward(
        self,
        hidden_states: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        *,
        initial_conv_state: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        return_final_state: bool = False,
    ) -> DeltaRuleAttentionOutput:
        """Apply the delta rule and optionally return recurrent and short-convolution states."""
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden_states must have shape [B, T, {self.hidden_size}], "
                f"got {tuple(hidden_states.shape)}"
            )
        batch, tokens, _ = hidden_states.shape
        if batch == 0 or tokens == 0:
            raise ValueError("batch size and sequence length must be greater than zero")
        if cu_seqlens is not None and self.backend != "fused":
            raise ValueError("packed cu_seqlens currently require backend='fused'")
        state_batch = batch if cu_seqlens is None else cu_seqlens.shape[0] - 1
        expected_state = (state_batch, self.num_heads, self.head_dim, self.head_dim)
        if initial_state is not None:
            if initial_state.shape != expected_state:
                raise ValueError(
                    f"initial_state must have shape {expected_state}, got {tuple(initial_state.shape)}"
                )
            initial_state = initial_state.to(device=hidden_states.device, dtype=torch.float32)
        return self.run_stages(
            hidden_states,
            initial_state,
            initial_conv_state,
            cu_seqlens=cu_seqlens,
            return_final_state=return_final_state,
        )

    def run_stages(
        self,
        hidden_states: torch.Tensor,
        initial_state: torch.Tensor | None,
        initial_conv_state: torch.Tensor | None,
        *,
        cu_seqlens: torch.Tensor | None,
        return_final_state: bool,
        **stage_kwargs: Any,
    ) -> DeltaRuleAttentionOutput:
        """Run the validated stage pipeline shared by every ``forward`` variant.

        ``stage_kwargs`` are forwarded to ``short_convolution`` and ``delta_rule_core`` so a subclass
        can thread per-call context (for example context-parallel routing) through those two
        stateful stages without storing it on the module or re-spelling the pipeline.
        """
        batch, tokens, _ = hidden_states.shape
        # --8<-- [start:kda-fixed-capacity-masking]
        # Keep the endpoint on-device: a captured graph rebuilds this one mask on
        # replay, then every value mask and gradient barrier below reuses it.
        active_mask = (  # (1)!
            active_token_mask(hidden_states, cu_seqlens)
            if self.mask_inactive_capacity and cu_seqlens is not None
            else None
        )
        # A zero `grad` cannot neutralize a NaN activation in a weight reduction.
        hidden_states = mask_inactive_tokens(hidden_states, active_mask)  # (2)!
        hidden_states_compute, qkv = self.qkv_projection(hidden_states)
        # The short-convolution dInput suffix is undefined; keep it out of qkv_proj dW.
        qkv = mask_inactive_token_gradients(qkv, active_mask)  # (3)!
        # --8<-- [end:kda-fixed-capacity-masking]
        qkv, final_conv_state = self.short_convolution(
            qkv,
            initial_conv_state,
            cu_seqlens=cu_seqlens,
            return_final_state=return_final_state,
            **stage_kwargs,
        )
        # Ordinary Q/K normalization reads and saves every physical row.
        qkv = mask_inactive_tokens(qkv, active_mask)
        q, k, v = qkv.view(batch, tokens, 3, self.num_heads, self.head_dim).unbind(2)
        norm_cu_seqlens = cu_seqlens if active_mask is not None else None
        q, k = self.qk_normalization(q, k, norm_cu_seqlens)
        raw_gate, beta = self.gate_projections(hidden_states_compute)
        # These barriers exclude undefined primitive dInputs from projection dW.
        raw_gate = mask_inactive_token_gradients(raw_gate, active_mask)
        beta = mask_inactive_token_gradients(beta, active_mask)
        gate = self.gate_activation(raw_gate)
        output, final_state = self.delta_rule_core(
            q,
            k,
            v,
            gate,
            beta,
            initial_state,
            cu_seqlens=cu_seqlens,
            return_final_state=return_final_state,
            **stage_kwargs,
        )
        # The core leaves its output suffix undefined; sanitize it before RMSNorm saves it.
        output = self.output_normalization(mask_inactive_tokens(output, active_mask))
        output_gate = self.output_gate(hidden_states_compute, output)
        output = self.output_projection(output, output_gate)
        # Keep arbitrary suffix cotangents out of every model parameter reduction.
        output = mask_inactive_token_gradients(output, active_mask)

        return DeltaRuleAttentionOutput(
            output.to(hidden_states.dtype),
            final_state,
            final_conv_state,
        )

    @annotate_kernels("{variant}/qkv_projection")
    def qkv_projection(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.to(self.compute_dtype)
        qkv = F.linear(hidden_states, self.qkv_proj.weight.to(self.compute_dtype))
        return hidden_states, qkv

    @annotate_kernels("{variant}/short_convolution")
    def short_convolution(
        self,
        qkv: torch.Tensor,
        initial_state: torch.Tensor | None,
        *,
        cu_seqlens: torch.Tensor | None = None,
        return_final_state: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, tokens, channels = qkv.shape
        state_length = self.qkv_conv1d.kernel_size[0] - 1
        state_batch = batch if cu_seqlens is None else cu_seqlens.shape[0] - 1
        expected_state = (state_batch, state_length, channels)
        if initial_state is not None:
            if initial_state.shape != expected_state:
                raise ValueError(
                    f"initial_conv_state must have shape {expected_state}, "
                    f"got {tuple(initial_state.shape)}"
                )
            initial_state = initial_state.to(device=qkv.device, dtype=qkv.dtype).contiguous()

        if self.backend == "fused":
            result = causal_conv1d(
                qkv,
                self.qkv_conv1d.weight[:, 0].to(self.compute_dtype),
                activation="silu",
                initial_state=initial_state,
                cu_seqlens=cu_seqlens,
                return_final_state=return_final_state,
            )
            return result if return_final_state else (result, None)

        if initial_state is None:
            initial_state = qkv.new_zeros(expected_state)
        conv_input = torch.cat((initial_state, qkv), dim=1)
        qkv = F.conv1d(
            conv_input.transpose(1, 2),
            self.qkv_conv1d.weight.to(self.compute_dtype),
            groups=channels,
        ).transpose(1, 2)
        qkv = F.silu(qkv)
        final_state = conv_input[:, tokens:].clone() if return_final_state else None
        return qkv, final_state

    @annotate_kernels("{variant}/qk_normalization")
    def qk_normalization(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.backend == "reference":
            q, k = q.float(), k.float()
            q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
            k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
            return q, k

        return l2norm(q, cu_seqlens=cu_seqlens), l2norm(k, cu_seqlens=cu_seqlens)

    @annotate_kernels("{variant}/gate_projections")
    def gate_projections(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, tokens, _ = hidden_states.shape
        if self.variant == "kda":
            gate_features = F.linear(
                hidden_states,
                self.f_a_proj.weight.to(self.compute_dtype),
            )
            raw_gate = F.linear(
                gate_features,
                self.f_b_proj.weight.to(self.compute_dtype),
            ).view(batch, tokens, self.num_heads, self.head_dim)
        else:
            raw_gate = F.linear(
                hidden_states,
                self.a_proj.weight.to(self.compute_dtype),
            ).view(batch, tokens, self.num_heads)
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

    @annotate_kernels("{variant}/gate_activation")
    def gate_activation(self, raw_gate: torch.Tensor) -> torch.Tensor:
        """Map projection outputs to per-token natural-log decay."""
        if self.variant == "kda":
            return bound_gate(
                raw_gate,
                self.A_log,
                self.dt_bias,
                lower_bound=self.lower_bound,
                fastmath=self.fastmath,
                impl=self.backend,
            )
        # Mamba2/GDN convention, matching GateTransform.SOFTPLUS in the fused decode kernel;
        # the training path has no fused counterpart yet, so both backends run it in eager FP32.
        return -torch.exp(self.A_log) * F.softplus(raw_gate.float() + self.dt_bias)

    @annotate_kernels("{variant}/core/{backend}")
    def delta_rule_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None,
        *,
        cu_seqlens: torch.Tensor | None = None,
        return_final_state: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.variant == "kda":
            return chunk_kda(
                q,
                k,
                v,
                gate,
                beta,
                initial_state,
                cu_seqlens=cu_seqlens,
                output_final_state=return_final_state,
                fastmath=self.fastmath,
                impl=self.backend,
            )
        return chunk_gdn(
            q,
            k,
            v,
            gate,
            beta,
            initial_state,
            cu_seqlens=cu_seqlens,
            output_final_state=return_final_state,
            impl=self.backend,
        )

    @annotate_kernels("{variant}/output_normalization")
    def output_normalization(self, output: torch.Tensor) -> torch.Tensor:
        # TODO: Consider a cu_seqlens-aware RMSNorm for fixed-capacity CUDA Graph
        # replay. Masking makes the inactive suffix numerically inert, but native
        # RMSNorm's gamma backward still scans all physical tokens.
        return F.rms_norm(
            output,
            (self.head_dim,),
            self.output_norm_weight.to(output.dtype),
            eps=self.rms_norm_eps,
        )

    @annotate_kernels("{variant}/output_gate")
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

    @annotate_kernels("{variant}/output_projection")
    def output_projection(self, output: torch.Tensor, output_gate: torch.Tensor) -> torch.Tensor:
        return F.linear(
            (output * output_gate).flatten(-2).to(self.compute_dtype),
            self.out_proj.weight.to(self.compute_dtype),
        )


def main(
    variant: Annotated[
        VariantOption,
        typer.Option(help="Train the KDA (bounded per-channel gate) or GDN (softplus) recipe."),
    ] = VariantOption.KDA,
    backend: Annotated[
        BackendOption,
        typer.Option(help="Use the reference or the best integrated fused kernels."),
    ] = BackendOption.REFERENCE,
    compute_dtype: Annotated[
        ComputeDTypeOption | None,
        typer.Option(help="Use float16 or bfloat16 projection and fused-kernel inputs."),
    ] = None,
    steps: Annotated[int, typer.Option(min=1, help="Number of optimizer steps.")] = 2,
    batch_size: Annotated[
        int,
        typer.Option(min=1, help="Training batch size, or packed logical sequence count."),
    ] = 1,
    tokens: Annotated[
        int,
        typer.Option(min=1, help="Tokens per sequence, or longest packed sequence."),
    ] = 16384,
    hidden_size: Annotated[int, typer.Option(min=1, help="Transformer hidden size.")] = 2304,
    num_heads: Annotated[int, typer.Option(min=1, help="Number of attention heads.")] = 32,
    head_dim: Annotated[int, typer.Option(min=1, help="Channels per head.")] = 128,
    short_conv_kernel_size: Annotated[
        int,
        typer.Option(min=1, help="Causal Q/K/V convolution width."),
    ] = 4,
    device: Annotated[str, typer.Option(help="Torch device used for training.")] = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
    profile: Annotated[
        bool,
        typer.Option(
            help=(
                "Export a named Chrome trace with forward/backward ranges and optional "
                "transformer-nuggets postprocessing."
            )
        ),
    ] = False,
    compile_model: Annotated[
        bool,
        typer.Option("--compile", help="Compile the complete module as one full graph."),
    ] = False,
    packed: Annotated[
        bool,
        typer.Option(help="Pack batch-size Zipf-distributed sequences bounded by tokens."),
    ] = False,
) -> None:
    """Train the single-device delta-rule example."""
    torch.manual_seed(0)
    if packed and backend != BackendOption.FUSED:
        raise ValueError("--packed requires --backend=fused")
    model = DeltaRuleAttention(
        hidden_size,
        num_heads,
        head_dim,
        variant=variant.value,
        short_conv_kernel_size=short_conv_kernel_size,
        backend=backend.value,
        compute_dtype=(None if compute_dtype is None else getattr(torch, compute_dtype.value)),
        device=device,
    )
    use_grad_scaler = model.compute_dtype == torch.float16 and torch.device(device).type == "cuda"
    if compile_model:
        model = torch.compile(model, fullgraph=True, mode="reduce-overhead")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    cu_seqlens = None
    input_shape = (batch_size, tokens, hidden_size)
    layout_name = ""
    if packed:
        sequence_lengths, offsets = packed_sequence_metadata(batch_size, tokens)
        cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device=device)
        input_shape = (1, offsets[-1], hidden_size)
        layout_name = "_packed"
        print(f"packed_sequence_lengths={sequence_lengths} cu_seqlens={offsets}")
    hidden_states = torch.randn(input_shape, device=device)
    target = torch.randn_like(hidden_states)
    execution_name = "_compiled" if compile_model else ""
    dtype_name = "" if compute_dtype is None else f"_{compute_dtype.value}"
    profile_name = (
        f"{variant.value}_training_backend-{backend.value}{dtype_name}{layout_name}{execution_name}"
        f"_b{batch_size}_t{tokens}_c{hidden_size}_h{num_heads}_d{head_dim}"
    )

    def train_step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        with _record_function(profile, f"{profile_name}/forward"):
            output = model(hidden_states, cu_seqlens=cu_seqlens).hidden_states
        with _record_function(profile, f"{profile_name}/loss"):
            loss = F.mse_loss(output.float(), target)
        with _record_function(profile, f"{profile_name}/backward"):
            grad_scaler.scale(loss).backward()
        with _record_function(profile, f"{profile_name}/optimizer"):
            grad_scaler.step(optimizer)
            grad_scaler.update()
        return loss

    if profile or compile_model:
        for _ in range(3):
            train_step()
        if hidden_states.is_cuda:
            torch.cuda.synchronize(hidden_states.device)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if hidden_states.is_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    profile_path = Path(f"{profile_name}.json")
    with _profile_trace(profile, profile_path, activities) as active_profiler:
        for step in range(steps):
            loss = train_step()
            if active_profiler is not None:
                active_profiler.step()
            print(f"step={step} loss={loss.item():.6f}")

    if profile:
        print(f"profile={profile_path.resolve()}")


if __name__ == "__main__":
    typer.run(main)
