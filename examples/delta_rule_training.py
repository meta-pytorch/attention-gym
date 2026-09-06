"""Train a small delta-rule attention module (KDA or GDN) on one device.

This is a small `example` delta-rule attention module. The goal is to show how one might use the
primitives(kernels) we have in attn_gym to build a performant implementation. The ``kda`` variant
roughly follows Kimi's architecture; the ``gdn`` variant follows the Gated DeltaNet / Qwen3-Next
recipe. They share the surrounding module stages and differ in the gate and the core: KDA learns a
bounded per-channel decay (``gate_transform(kind="bounded")`` on ``[B, T, H, D]``) and runs
``chunk_kda``; GDN learns one softplus decay per head (``gate_transform(kind="softplus")`` on
``[B, T, H]``) and runs ``chunk_gdn``.
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
native Perfetto trace (``.pftrace``, requires transformer-nuggets). The trace
contains explicit ``forward`` and ``backward`` record-function ranges. Add
``--compile`` to compile the complete module with
``torch.compile(fullgraph=True)`` and fuse the PyTorch work around the custom
delta-rule operator.

The batch, loss/backward, and capture recipes below are also imported by the context-parallel
example. Numerical assertions, trace export, and benchmark reporting stay in ``attn_gym.testing``.
"""

from __future__ import annotations

import gc
import math
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import Enum
from functools import partial
from itertools import accumulate
from pathlib import Path
from typing import Annotated, Any, Literal, NamedTuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import typer
from torch import nn

from attn_gym.linear import gate_transform
from attn_gym.linear.context_parallel import ContextParallelPlan, ContextParallelRouting
from attn_gym.linear.gdn import chunk_gdn
from attn_gym.linear.kda import (
    MAX_GATE_LOWER_BOUND_MAGNITUDE,
    active_token_mask,
    causal_conv1d,
    chunk_kda,
    l2norm,
    mask_inactive_token_gradients,
    mask_inactive_tokens,
)
from attn_gym.linear.types import KernelOptions
from attn_gym.testing import annotate_kernels, kernel_stage, profile_trace, record_function
from attn_gym.testing.delta_rule import (
    assert_context_parallel_matches_reference,
    measure_training_step,
    profile_training_step,
    write_benchmark_report,
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


class CoreBackendOption(str, Enum):
    """Chunk-core kernel families selectable through ``kernel_options``."""

    FUSED = "fused"
    MEGA = "mega"


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


class DeltaRuleAttentionOutput(NamedTuple):
    """Hidden states and optional recurrent and convolution states."""

    hidden_states: torch.Tensor
    final_state: torch.Tensor | None
    final_conv_state: torch.Tensor | None


@dataclass(frozen=True)
class PackedTrainingBatch:
    """Global reference tensors and this rank's span and per-call routing.

    ``token_ids`` maps span positions to global token ids; ``terminal_index`` lists the local
    subsequences that end their sequence and ``terminal_sequences`` the matching global sequence
    ids, so endpoint states can be compared with the unsharded run.
    """

    global_hidden: torch.Tensor | None
    global_target: torch.Tensor | None
    global_offsets: torch.Tensor
    local_hidden: torch.Tensor
    local_target: torch.Tensor
    routing: ContextParallelRouting
    token_ids: torch.Tensor
    terminal_index: torch.Tensor
    terminal_sequences: torch.Tensor
    loss_scale: float


class DeltaRuleAttention(nn.Module):
    """Minimal trainable delta-rule attention with a transformer-style module ABI.

    ``variant="kda"`` learns a bounded per-channel log decay through ``f_a_proj``/``f_b_proj``
    and the bounded ``gate_transform``; ``variant="gdn"`` learns one softplus log decay per head
    through ``a_proj``. ``lower_bound`` applies to the KDA gate only; ``fastmath`` applies to
    both fused gates and to the KDA core. ``kernel_options`` are passed through to the chunk
    core (``{"backend": "mega"}`` selects the SM100 Mega kernels for either variant).

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
        kernel_options: KernelOptions | None = None,
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
        if variant == "kda" and (not math.isfinite(lower_bound) or lower_bound > 0.0):
            raise ValueError(f"lower_bound must be finite and nonpositive, got {lower_bound}")
        if (
            variant == "kda"
            and backend == "fused"
            and lower_bound < -MAX_GATE_LOWER_BOUND_MAGNITUDE
        ):
            raise ValueError(
                f"the fused backend requires lower_bound >= "
                f"{-MAX_GATE_LOWER_BOUND_MAGNITUDE:.3f}, got {lower_bound}"
            )
        if backend == "fused" and head_dim != 128:
            raise ValueError("the fused backend requires head_dim=128")
        if backend == "reference" and fastmath:
            raise ValueError("fastmath applies only to backend='fused'")
        if backend == "reference" and kernel_options:
            raise ValueError("kernel_options apply only to backend='fused'")

        self.variant = variant
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.lower_bound = lower_bound
        self.backend = backend
        self.fastmath = fastmath
        self.kernel_options = kernel_options
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

    @annotate_kernels("{module.variant}/qkv_projection")
    def qkv_projection(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.to(self.compute_dtype)
        qkv = F.linear(hidden_states, self.qkv_proj.weight.to(self.compute_dtype))
        return hidden_states, qkv

    @annotate_kernels("{module.variant}/short_convolution")
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

    @annotate_kernels("{module.variant}/qk_normalization")
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

    @annotate_kernels("{module.variant}/gate_projections")
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

    @annotate_kernels("{module.variant}/gate_activation")
    def gate_activation(self, raw_gate: torch.Tensor) -> torch.Tensor:
        """Map projection outputs to per-token natural-log decay."""
        if self.variant == "kda":
            kind, lower_bound = "bounded", self.lower_bound
        else:
            kind, lower_bound = "softplus", None
        return gate_transform(
            raw_gate,
            self.A_log,
            self.dt_bias,
            kind=kind,
            lower_bound=lower_bound,
            fastmath=self.fastmath,
            impl=self.backend,
        )

    @annotate_kernels("{module.variant}/core/{module.backend}")
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
                kernel_options=self.kernel_options,
            )
        # The reference path normalizes Q/K in FP32; chunk_gdn requires one dtype across Q/K/V.
        return chunk_gdn(
            q,
            k,
            v.to(q.dtype),
            gate,
            beta,
            initial_state,
            cu_seqlens=cu_seqlens,
            output_final_state=return_final_state,
            impl=self.backend,
            kernel_options=self.kernel_options,
        )

    @annotate_kernels("{module.variant}/output_normalization")
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

    @annotate_kernels("{module.variant}/output_gate")
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

    @annotate_kernels("{module.variant}/output_projection")
    def output_projection(self, output: torch.Tensor, output_gate: torch.Tensor) -> torch.Tensor:
        return F.linear(
            (output * output_gate).flatten(-2).to(self.compute_dtype),
            self.out_proj.weight.to(self.compute_dtype),
        )


@contextmanager
def distributed_device() -> Iterator[torch.device]:
    """Own the single-node NCCL world group launched by torchrun."""
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("launch with torchrun --standalone --nproc-per-node=<world_size>")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    try:
        yield device
    finally:
        torch.cuda.synchronize(device)
        dist.destroy_process_group()


def make_context_parallel_batch(
    plan: ContextParallelPlan,
    offsets: tuple[int, ...],
    hidden_size: int,
    device: torch.device,
    *,
    conv_history: int,
    validate: bool,
) -> PackedTrainingBatch:
    """Build a deterministic global stream, retaining only this rank's shard unless validating.

    All ranks use the same data seeds. Without reference validation, generate on CPU and move
    only the shard: allocating the whole stream on CUDA can leave large cached segments behind
    and distort the captured benchmark's memory footprint.
    """
    token_ids = plan.global_token_ids(device)
    stream_device = device if validate else torch.device("cpu")
    global_hidden = torch.randn(
        1,
        offsets[-1],
        hidden_size,
        device=stream_device,
        generator=torch.Generator(stream_device).manual_seed(123),
    )
    global_target = torch.randn(
        1,
        offsets[-1],
        hidden_size,
        device=stream_device,
        generator=torch.Generator(stream_device).manual_seed(124),
    )
    shard_ids = token_ids.to(stream_device)
    return PackedTrainingBatch(
        global_hidden=global_hidden if validate else None,
        global_target=global_target if validate else None,
        global_offsets=torch.tensor(offsets, dtype=torch.int32, device=device),
        local_hidden=global_hidden[:, shard_ids].to(device).requires_grad_(),
        local_target=global_target[:, shard_ids].to(device),
        routing=plan.routing(device, conv_history=conv_history),
        token_ids=token_ids,
        terminal_index=torch.tensor(plan.terminal, dtype=torch.long, device=device),
        terminal_sequences=torch.tensor(
            [plan.subsequences[index].sequence for index in plan.terminal],
            dtype=torch.long,
            device=device,
        ),
        loss_scale=1.0 / global_target.numel(),
    )


def run_training_step(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    target: torch.Tensor,
    routing: ContextParallelRouting,
    state_index: torch.Tensor,
    loss_scale: float,
    *,
    annotate: bool = False,
) -> tuple[DeltaRuleAttentionOutput, tuple[torch.Tensor, ...]]:
    """Run the complete module and differentiate its token and endpoint losses."""
    module.enable_graph_annotations = annotate
    result = module(
        hidden_states,
        routing=routing,
        return_final_state=True,
    )
    return result, training_gradients(
        module, result, hidden_states, target, state_index, loss_scale, annotate=annotate
    )


def training_gradients(
    module: torch.nn.Module,
    result: DeltaRuleAttentionOutput,
    hidden_states: torch.Tensor,
    target: torch.Tensor,
    state_index: torch.Tensor,
    loss_scale: float,
    *,
    annotate: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Differentiate the same token and true-endpoint losses in CP and unsharded runs."""
    assert result.final_state is not None
    assert result.final_conv_state is not None
    loss = F.mse_loss(result.hidden_states.float(), target, reduction="sum") * loss_scale
    # Only true sequence ends carry a loss; intermediate subsequence states are handed downstream.
    loss = loss + 1e-4 * result.final_state[state_index].square().sum()
    loss = loss + 1e-4 * result.final_conv_state[state_index].float().square().sum()
    inputs = (hidden_states, *tuple(module.parameters()))
    with kernel_stage("cp/bwd", annotate, backward=False):
        return torch.autograd.grad(loss, inputs)


def validate_against_reference(
    model: torch.nn.Module,
    reference_model: torch.nn.Module,
    batch: PackedTrainingBatch,
) -> None:
    """Compare one sharded full-module backward with the unsharded module."""
    result, gradients = run_training_step(
        model,
        batch.local_hidden,
        batch.local_target,
        batch.routing,
        batch.terminal_index,
        batch.loss_scale,
    )
    assert batch.global_hidden is not None and batch.global_target is not None
    reference_hidden = batch.global_hidden.detach().clone().requires_grad_()
    # Unsharded, every sequence in the stream ends here, so every exit state is a true final state.
    every_sequence = torch.arange(
        batch.global_offsets.shape[0] - 1, device=reference_hidden.device
    )
    reference_result = reference_model(
        reference_hidden, cu_seqlens=batch.global_offsets, return_final_state=True
    )
    reference_gradients = training_gradients(
        reference_model,
        reference_result,
        reference_hidden,
        batch.global_target,
        every_sequence,
        batch.loss_scale,
    )

    assert_context_parallel_matches_reference(
        [
            (result.hidden_states, reference_result.hidden_states[:, batch.token_ids]),
            (gradients[0], reference_gradients[0][:, batch.token_ids]),
            (
                result.final_state[batch.terminal_index],
                reference_result.final_state[batch.terminal_sequences],
            ),
            (
                result.final_conv_state[batch.terminal_index],
                reference_result.final_conv_state[batch.terminal_sequences],
            ),
        ],
        zip(gradients[1:], reference_gradients[1:], strict=True),
        model.compute_dtype,
    )


@contextmanager
def capture_training_graph(
    make_model: Callable[[], torch.nn.Module],
    eager_model: torch.nn.Module,
    batch: PackedTrainingBatch,
    *,
    validate: bool = False,
    annotations: bool = False,
) -> Iterator[torch.cuda.CUDAGraph]:
    """Own a captured training step and its backing model/input until the caller finishes.

    Validation retains an eager oracle and captured outputs to check a changed-input replay.
    Benchmarking discards step outputs and clears the cache before capture instead, so those
    allocations do not inflate its graph-pool footprint. Both paths use the same stream lifecycle.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # Parameter AccumulateGrad nodes retain their first stream: use a fresh model and keep
        # its warmup, eager oracle (if requested), and capture on this stream.
        model = make_model()
        model.load_state_dict(eager_model.state_dict())
        static_hidden = batch.local_hidden.detach().clone().requires_grad_()
        step = partial(
            run_training_step,
            model,
            target=batch.local_target,
            routing=batch.routing,
            state_index=batch.terminal_index,
            loss_scale=batch.loss_scale,
        )
        if validate:
            # Warm up on the changed input that replay will see, before the graph pool exists.
            eager_hidden = (static_hidden.detach() * 0.75).requires_grad_()
            eager_result, eager_gradients = step(eager_hidden)
            del eager_hidden
        else:
            step(static_hidden)
    stream.synchronize()
    gc.collect()
    if not validate:
        # The warmup activations are dead; do not layer the graph pool over their cached segments.
        torch.cuda.empty_cache()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream, enable_annotations=annotations):
            if validate:
                graph_result, graph_gradients = step(static_hidden, annotate=annotations)
            else:
                step(static_hidden, annotate=annotations)
        torch.cuda.current_stream().wait_stream(stream)
        if validate:
            captured_output = graph_result.hidden_states.clone()
            torch.cuda.synchronize(static_hidden.device)
            with torch.no_grad():
                static_hidden.mul_(0.75)
            graph.replay()
            torch.cuda.synchronize(static_hidden.device)
            if torch.equal(graph_result.hidden_states, captured_output):
                raise AssertionError("CUDA Graph replay did not observe changed inputs")
            torch.testing.assert_close(graph_result, eager_result)
            torch.testing.assert_close(graph_gradients, eager_gradients)
        yield graph
    finally:
        torch.cuda.synchronize(static_hidden.device)
        graph.reset()
        gc.collect()


def profile_eager_step(
    model: torch.nn.Module,
    batch: PackedTrainingBatch,
    profile_path: Path,
    device: torch.device,
    warmup_steps: int,
) -> None:
    """Profile a complete eager forward/backward on a fresh input leaf."""
    hidden_states = batch.local_hidden.detach().clone().requires_grad_()
    profile_training_step(
        partial(
            run_training_step,
            model,
            hidden_states,
            batch.local_target,
            batch.routing,
            batch.terminal_index,
            batch.loss_scale,
        ),
        profile_path,
        device,
        warmup_steps,
    )


def run_benchmark(
    make_model: Callable[[], torch.nn.Module],
    model: torch.nn.Module,
    batch: PackedTrainingBatch,
    device: torch.device,
    *,
    steps: int,
    warmup_steps: int,
    cuda_graph: bool,
    sequence_lengths: tuple[int, ...],
    partition: str,
) -> None:
    """Time steady-state training steps and record per-rank memory; rank 0 writes JSON.

    Step time is measured per rank with CUDA events; the reported step time is
    the slowest rank per iteration, which is what gates training. Memory is the
    peak above what is resident before the timed steps (parameters, inputs, and
    the graph's static buffers when captured).
    """
    hidden_states = batch.local_hidden.detach().clone().requires_grad_()
    # The capture context owns its model and static input through timing and memory sampling.
    with (
        capture_training_graph(make_model, model, batch) if cuda_graph else nullcontext()
    ) as graph:
        step: Callable[[], object] = (
            graph.replay
            if graph is not None
            else partial(
                run_training_step,
                model,
                hidden_states,
                batch.local_target,
                batch.routing,
                batch.terminal_index,
                batch.loss_scale,
            )
        )
        measurement = measure_training_step(
            step,
            device,
            steps=steps,
            warmup_steps=warmup_steps,
            local_tokens=int(batch.token_ids.numel()),
        )
    # Release the graph pool before rank-report collectives allocate staging buffers.
    write_benchmark_report(
        measurement,
        model,
        device,
        steps=steps,
        warmup_steps=warmup_steps,
        cuda_graph=cuda_graph,
        sequence_lengths=sequence_lengths,
        partition=partition,
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
    core_backend: Annotated[
        CoreBackendOption,
        typer.Option(help="Chunk kernels for the fused backend: repo-local or Mega (SM100)."),
    ] = CoreBackendOption.FUSED,
    fastmath: Annotated[
        bool, typer.Option(help="Use approximate exponentials in the fused gate and KDA core.")
    ] = False,
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
                "Export a native Perfetto .pftrace with forward/backward ranges "
                "(requires transformer-nuggets)."
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
        fastmath=fastmath,
        kernel_options=(
            {"backend": core_backend.value} if core_backend is CoreBackendOption.MEGA else None
        ),
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
        with record_function(profile, f"{profile_name}/forward"):
            output = model(hidden_states, cu_seqlens=cu_seqlens).hidden_states
        with record_function(profile, f"{profile_name}/loss"):
            loss = F.mse_loss(output.float(), target)
        with record_function(profile, f"{profile_name}/backward"):
            grad_scaler.scale(loss).backward()
        with record_function(profile, f"{profile_name}/optimizer"):
            grad_scaler.step(optimizer)
            grad_scaler.update()
        return loss

    if profile or compile_model:
        for _ in range(3):
            train_step()
        if hidden_states.is_cuda:
            torch.cuda.synchronize(hidden_states.device)

    profile_path = Path(f"{profile_name}.pftrace")
    with profile_trace(profile_path) if profile else nullcontext() as active_profiler:
        for step in range(steps):
            loss = train_step()
            if active_profiler is not None:
                active_profiler.step()
            print(f"step={step} loss={loss.item():.6f}")

    if profile:
        print(f"profile={profile_path.resolve()}")


if __name__ == "__main__":
    typer.run(main)
