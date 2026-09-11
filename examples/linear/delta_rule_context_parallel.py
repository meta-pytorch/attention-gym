"""Run the delta-rule training module (KDA or GDN) with packed context parallelism.

Each rank owns fragments of one packed stream. The subclass below changes only the two stateful
stages: a halo exchange supplies the short convolution's history, and affine summaries supply the
recurrence's incoming state. Projections, gates, normalization, and output projection are inherited
from ``delta_rule_training.py``.

Like that example's ``--packed`` mode, ``--batch-size`` is the number of logical sequences and
``--tokens`` bounds their Zipf-distributed lengths. CP is always packed and fused. Defaults use
TorchTitan's K3 attention dimensions: hidden size 7168 and 96 heads of dimension 128, not a full
K3 model. Launch with:

    torchrun --standalone --nproc-per-node=2 examples/linear/delta_rule_context_parallel.py --batch-size 4 --tokens 1024

Add ``--variant gdn``, ``--partition zigzag``, or ``--compute-dtype float16`` to vary the recipe.
``--partition documents`` assigns whole documents to ranks (``document_aligned_fragments``): no
document crosses ranks, so the delta-rule op exchanges no state and its result is bitwise
independent of the CP degree (``test/test_context_parallel_distributed.py``, table ``documents``).
``--core-backend cudnn`` selects the KDA cuDNN backend (SM100/SM103). ``--cuda-graph`` checks a
changed-input replay; ``--profile`` writes a merged native Perfetto trace. ``--no-validate`` skips
the unsharded reference at scales where it does not fit. Batch construction, loss/backward, and
capture are shown in ``delta_rule_training.py`` and imported here. Numerical assertions, trace
export, and benchmark reporting live in ``attn_gym.testing``.
"""

from __future__ import annotations

import gc
from enum import Enum
from functools import partial
from itertools import accumulate, pairwise
from pathlib import Path
from typing import Annotated, Literal

import torch
import torch.distributed as dist
import typer

from attn_gym.linear.context_parallel import (
    ContextParallelPlan,
    ContextParallelRouting,
    context_parallel_conv_history,
)
from attn_gym.linear.gdn.context_parallel import context_parallel_gdn
from attn_gym.linear.kda.context_parallel import context_parallel_kda
from attn_gym.testing import TraceFormat, kernel_stage, record_distributed_profile
from attn_gym.testing.profiling import graph_annotations_available
from examples.linear.delta_rule_training import (
    ComputeDTypeOption,
    CoreBackendOption,
    DeltaRuleAttention,
    DeltaRuleAttentionOutput,
    VariantOption,
    capture_training_graph,
    distributed_device,
    make_context_parallel_batch,
    packed_sequence_metadata,
    profile_eager_step,
    run_benchmark,
    run_training_step,
    validate_against_reference,
)


class PartitionOption(str, Enum):
    """Rank ownership of the packed stream."""

    CONTIGUOUS = "contiguous"
    ZIGZAG = "zigzag"
    DOCUMENTS = "documents"


class TraceFormatOption(str, Enum):
    """Per-rank trace format for ``--profile``."""

    PERFETTO = "perfetto"
    KINETO = "kineto"

    @property
    def trace_format(self) -> TraceFormat:
        """Return the transformer-nuggets trace format."""
        return "track_event" if self is TraceFormatOption.PERFETTO else "chrome_json"


def partition_fragments(
    tokens: int,
    world_size: int,
    partition: Literal["contiguous", "zigzag"] = "contiguous",
) -> list[list[tuple[int, int]]]:
    """Example rank mappings; edit this function to assign your own global token ranges.

    Contiguous assigns rank r block r of W near-balanced blocks. Zigzag assigns
    blocks r and 2W - 1 - r of 2W blocks. Floor-based boundaries preserve equal
    blocks for divisible totals without dropping or padding tokens otherwise.
    """
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if partition not in ("contiguous", "zigzag"):
        raise ValueError(f"partition must be 'contiguous' or 'zigzag', got {partition!r}")
    blocks = world_size if partition == "contiguous" else 2 * world_size
    if tokens < blocks:
        raise ValueError(f"{tokens} tokens cannot fill {blocks} nonempty blocks")
    owned = [
        [rank] if partition == "contiguous" else [rank, blocks - 1 - rank]
        for rank in range(world_size)
    ]
    return [
        [(block * tokens // blocks, (block + 1) * tokens // blocks) for block in rank_blocks]
        for rank_blocks in owned
    ]


def document_aligned_fragments(
    offsets: tuple[int, ...], world_size: int
) -> list[list[tuple[int, int]]]:
    """Assign whole documents to ranks so no fragment ever splits one.

    Every document then runs from the zero state to its end on the rank that owns it, exactly as
    the unsharded op runs it: the recipe exchanges nothing across ranks and its result is bitwise
    the same for any CP degree, including 1. The price is balance: longest documents first onto
    the least-loaded rank keeps ranks within one document's length of each other, and a document
    longer than a rank can hold has no document-aligned cut at all. Consecutive documents on one
    rank merge into one fragment.
    """
    if world_size < 1:
        raise ValueError("world_size must be positive")
    documents = list(pairwise(offsets))
    if len(documents) < world_size:
        raise ValueError(f"{len(documents)} documents cannot give {world_size} ranks work")
    loads = [0] * world_size
    owner = [0] * len(documents)
    for index in sorted(range(len(documents)), key=lambda i: documents[i][0] - documents[i][1]):
        rank = loads.index(min(loads))
        owner[index] = rank
        loads[rank] += documents[index][1] - documents[index][0]
    fragments: list[list[tuple[int, int]]] = [[] for _ in range(world_size)]
    for (start, stop), rank in zip(documents, owner, strict=True):
        if fragments[rank] and fragments[rank][-1][1] == start:
            fragments[rank][-1] = (fragments[rank][-1][0], stop)
        else:
            fragments[rank].append((start, stop))
    return fragments


class ContextParallelDeltaRuleAttention(DeltaRuleAttention):
    """The training example's complete delta-rule module with distributed state plumbing."""

    def __init__(self, *args, group: dist.ProcessGroup, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.variant == "gdn" and self.kernel_options:
            raise ValueError("context_parallel_gdn does not take kernel_options")
        self.cp_group = group

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        routing: ContextParallelRouting,
        return_final_state: bool = False,
    ) -> DeltaRuleAttentionOutput:
        """Apply the packed delta rule using this batch's routing for both distributed state exchanges.

        Routing covers every local token; recurrent and convolution entry states are
        constructed from the other ranks, not supplied by the caller.
        """
        if (
            hidden_states.ndim != 3
            or hidden_states.shape[0] != 1
            or hidden_states.shape[-1] != self.hidden_size
        ):
            raise ValueError(
                f"hidden_states must have shape [1, T, {self.hidden_size}], "
                f"got {tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[1] == 0:
            raise ValueError("sequence length must be greater than zero")
        if self.backend != "fused":
            raise ValueError("packed context parallelism requires backend='fused'")
        if routing.tail_sources.shape[1] != self.qkv_conv1d.kernel_size[0] - 1:
            raise ValueError(
                "routing conv_history must match the model's convolution width minus one"
            )
        # Pass CP routing info straight to internal stages in delta rule training examples
        return self.run_stages(
            hidden_states,
            None,
            None,
            cu_seqlens=routing.cu_seqlens,
            return_final_state=return_final_state,
            routing=routing,
        )

    def short_convolution(
        self,
        qkv: torch.Tensor,
        initial_state: torch.Tensor | None,
        *,
        cu_seqlens: torch.Tensor | None = None,
        return_final_state: bool,
        routing: ContextParallelRouting,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert initial_state is None, "context parallelism constructs the convolution history"
        with kernel_stage("cp/conv/halo", self.enable_graph_annotations):
            initial_state = context_parallel_conv_history(qkv, routing, self.cp_group)
        return super().short_convolution(
            qkv,
            initial_state,
            cu_seqlens=cu_seqlens,
            return_final_state=return_final_state,
        )

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
        routing: ContextParallelRouting,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert initial_state is None
        if self.variant == "kda":
            output, final_state = context_parallel_kda(
                q,
                k,
                v,
                gate,
                beta,
                routing=routing,
                group=self.cp_group,
                fastmath=self.fastmath,
                kernel_options=self.kernel_options,
            )
        else:
            output, final_state = context_parallel_gdn(
                q, k, v, gate, beta, routing=routing, group=self.cp_group
            )
        return output, final_state if return_final_state else None


def main(
    variant: Annotated[
        VariantOption, typer.Option(help="Run the KDA or GDN recipe.")
    ] = VariantOption.KDA,
    compute_dtype: Annotated[
        ComputeDTypeOption,
        typer.Option(help="Use float16 or bfloat16 projection and kernel inputs."),
    ] = ComputeDTypeOption.BFLOAT16,
    core_backend: Annotated[
        CoreBackendOption,
        typer.Option(
            "--core-backend",
            "--kda-backend",
            help="Local KDA chunk kernels: fused or cuDNN (SM100).",
        ),
    ] = CoreBackendOption.FUSED,
    fastmath: Annotated[
        bool, typer.Option(help="Use approximate exponentials in the fused gate and KDA core.")
    ] = False,
    batch_size: Annotated[
        int, typer.Option(min=1, help="Number of packed logical sequences.")
    ] = 4,
    tokens: Annotated[
        int,
        typer.Option(min=1, help="Longest packed sequence; lengths follow a Zipf distribution."),
    ] = 1024,
    hidden_size: Annotated[int, typer.Option(min=1, help="Transformer hidden size.")] = 7168,
    num_heads: Annotated[
        int, typer.Option("--num-heads", "--heads", min=1, help="Number of attention heads.")
    ] = 96,
    short_conv_kernel_size: Annotated[
        int, typer.Option(min=1, help="Causal Q/K/V convolution width.")
    ] = 4,
    partition: Annotated[
        PartitionOption, typer.Option(help="How ranks own fragments of the global stream.")
    ] = PartitionOption.CONTIGUOUS,
    sequence_lengths: Annotated[
        str | None,
        typer.Option(help="Explicit comma-separated lengths, overriding batch-size/tokens."),
    ] = None,
    cuda_graph: Annotated[
        bool, typer.Option(help="Capture forward/backward and validate a changed-input replay.")
    ] = False,
    validate: Annotated[
        bool, typer.Option(help="Compare against the unsharded module on the whole stream.")
    ] = True,
    profile: Annotated[
        bool, typer.Option(help="Export a merged trace of one steady-state step.")
    ] = False,
    trace_format: Annotated[
        TraceFormatOption,
        typer.Option(help="Per-rank format; kineto writes gzipped JSON for annotate-roofline."),
    ] = TraceFormatOption.PERFETTO,
    warmup_steps: Annotated[
        int, typer.Option(min=0, help="Warmup steps before profiling or timing.")
    ] = 5,
    benchmark_steps: Annotated[
        int,
        typer.Option(
            min=0, help="Timed forward/backward steps; write a scaling report (0 disables)."
        ),
    ] = 0,
) -> None:
    """Run the same packed module recipe across one Hopper-or-newer GPU per rank."""
    with distributed_device() as device:
        torch.manual_seed(0)
        if sequence_lengths is None:
            lengths, offsets = packed_sequence_metadata(batch_size, tokens)
        else:
            lengths = tuple(int(length) for length in sequence_lengths.split(","))
            offsets = (0, *accumulate(lengths))
        if dist.get_rank() == 0:
            print(f"packed_sequence_lengths={lengths} cu_seqlens={offsets}", flush=True)
        if partition is PartitionOption.DOCUMENTS:
            fragments = document_aligned_fragments(offsets, dist.get_world_size())
        else:
            fragments = partition_fragments(offsets[-1], dist.get_world_size(), partition.value)
        if dist.get_rank() == 0:
            print(f"fragments={fragments}", flush=True)
        plan = ContextParallelPlan.from_fragments(offsets, fragments, dist.get_rank())
        batch = make_context_parallel_batch(
            plan,
            offsets,
            hidden_size,
            device,
            conv_history=short_conv_kernel_size - 1,
            validate=validate,
        )
        model_options = {
            "hidden_size": hidden_size,
            "num_heads": num_heads,
            "head_dim": 128,
            "variant": variant.value,
            "short_conv_kernel_size": short_conv_kernel_size,
            "backend": "fused",
            "fastmath": fastmath,
            "compute_dtype": getattr(torch, compute_dtype.value),
            "device": device,
        }
        # The reference keeps the repo-local core; only each rank's local KDA pass may use cuDNN.
        make_model = partial(
            ContextParallelDeltaRuleAttention,
            **model_options,
            group=dist.group.WORLD,
            kernel_options={"backend": core_backend.value}
            if core_backend is CoreBackendOption.CUDNN
            else None,
        )
        model = make_model()

        # --validate checks outputs and gradients against the complete unsharded model.
        if validate:
            reference = DeltaRuleAttention(**model_options)
            reference.load_state_dict(model.state_dict())
            validate_against_reference(model, reference, batch)
            del reference  # Free the full-model oracle before capturing a graph pool.
        else:
            run_training_step(
                model,
                batch.local_hidden,
                batch.local_target,
                batch.routing,
                batch.terminal_index,
                batch.loss_scale,
            )
        gc.collect()

        mode = "cuda_graph" if cuda_graph else "eager"
        profile_path = Path(
            "data",
            f"{variant.value}_context_parallel_{mode}_{partition.value}_{core_backend.value}"
            f"_{compute_dtype.value}_w{dist.get_world_size()}_t{offsets[-1]}"
            f"_h{num_heads}_c{hidden_size}_conv{short_conv_kernel_size}",
        ).resolve()
        if benchmark_steps:
            run_benchmark(
                make_model,
                model,
                batch,
                device,
                steps=benchmark_steps,
                warmup_steps=warmup_steps,
                cuda_graph=cuda_graph,
                sequence_lengths=lengths,
                partition=partition.value,
            )
        elif cuda_graph:
            # Capture once; profiling below records replay, not graph construction.
            annotations = graph_annotations_available()
            if not annotations and dist.get_rank() == 0:
                print(
                    "Capturing without kernel labels: graph annotations need a supported PyTorch/driver."
                )
            with capture_training_graph(
                make_model, model, batch, validate=True, annotations=annotations
            ) as graph:
                if profile:
                    merged_path = record_distributed_profile(
                        graph.replay,
                        profile_path,
                        "cuda_graph_replay",
                        device,
                        warmup_steps=warmup_steps,
                        trace_format=trace_format.trace_format,
                    )
                    if merged_path is not None:
                        print(f"profile={merged_path}", flush=True)
        elif profile:
            profile_eager_step(
                model, batch, profile_path, device, warmup_steps, trace_format.trace_format
            )

        status = "passed" if validate else "ran"
        print(
            f"rank {dist.get_rank()}: full packed {variant.value.upper()} CP "
            f"({partition.value}, {core_backend.value}) {status} [{mode}]",
            flush=True,
        )


if __name__ == "__main__":
    typer.run(main)
