"""Train a complete packed KDA attention module with context parallelism.

This reuses the transformer-style module from ``kda_training.py`` and distributes both stateful
operations through the reference recipe in ``attn_gym.linear.context_parallel``: a halo
exchange supplies the short convolution's finite history, and the KDA recurrence's initial state
is composed from the affine summaries of the ranks that hold the preceding tokens. Each rank owns a list of fragments (global token ranges) chosen
here in plain Python (``fragments``), so the same code validates contiguous shards and zig-zag load
balancing; see NOTE [Terminology] in ``attn_gym.linear.state_summary``. Launch with:

    torchrun --standalone --nproc-per-node=2 examples/kda_context_parallel.py

Add ``--partition zigzag`` to give each rank two mirrored blocks, ``--compute-dtype=float16``
to validate the FP16 route, ``--kda-backend mega`` to run each local pass with the Mega backend
(SM100/SM103), ``--cuda-graph`` to capture forward and backward together and validate a replay
with changed inputs, ``--profile`` to export a merged native Perfetto trace using
transformer-nuggets, and ``--no-validate`` to skip the unsharded reference at scales where it does
not fit. The example requires one Hopper or datacenter Blackwell GPU per rank.
"""

from __future__ import annotations

import gc
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import partial
from itertools import accumulate
from pathlib import Path
from typing import Annotated

import torch
import torch.distributed as dist
import torch.nn.functional as F
import typer

from attn_gym.linear.context_parallel import ContextParallelPlan, context_parallel_conv_history
from attn_gym.linear.kda.context_parallel import context_parallel_kda
from attn_gym.linear.types import KernelOptions
from attn_gym.testing import kernel_stage, record_distributed_profile
from examples.kda_training import ComputeDTypeOption, KDAAttention, KDAAttentionOutput


class PartitionOption(str, Enum):
    CONTIGUOUS = "contiguous"
    ZIGZAG = "zigzag"


class KDABackendOption(str, Enum):
    FUSED = "fused"
    MEGA = "mega"


def fragments(
    tokens: int, world_size: int, partition: PartitionOption
) -> list[list[tuple[int, int]]]:
    """Choose each rank's fragments (global token ranges), in span order.

    Contiguous gives rank ``r`` block ``r`` of ``W`` equal blocks. Zig-zag gives it blocks ``r``
    and ``2W - 1 - r`` of ``2W``: the load-balanced layout ring softmax attention uses for causal
    masks, which a hybrid model's KDA layers inherit. Any other assignment is just a different
    list; the plan cuts fragments at sequence boundaries itself.
    """
    blocks = world_size if partition is PartitionOption.CONTIGUOUS else 2 * world_size
    if tokens % blocks:
        raise ValueError(f"{tokens} tokens do not split into {blocks} equal blocks")
    size = tokens // blocks
    owned = [
        [cp_rank] if partition is PartitionOption.CONTIGUOUS else [cp_rank, blocks - 1 - cp_rank]
        for cp_rank in range(world_size)
    ]
    return [[(block * size, (block + 1) * size) for block in blocks_of] for blocks_of in owned]


@dataclass(frozen=True)
class PackedTrainingBatch:
    """Global reference tensors and this rank's span of them.

    ``token_ids`` maps span positions to global token ids; ``terminal_index`` lists the local
    subsequences that end their sequence and ``terminal_sequences`` the matching global sequence
    ids, so endpoint states can be compared with the unsharded run.
    """

    global_hidden: torch.Tensor | None
    global_target: torch.Tensor | None
    global_offsets: torch.Tensor
    local_hidden: torch.Tensor
    local_target: torch.Tensor
    local_offsets: torch.Tensor
    token_ids: torch.Tensor
    terminal_index: torch.Tensor
    terminal_sequences: torch.Tensor
    loss_scale: float


class ContextParallelKDAAttention(KDAAttention):
    """The training example's complete KDA module with distributed state plumbing."""

    def __init__(
        self,
        *args,
        group: dist.ProcessGroup,
        plan: ContextParallelPlan,
        kernel_options: KernelOptions | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cp_group = group
        self.kernel_options = kernel_options
        self.routing = plan.routing(
            torch.device("cuda", torch.cuda.current_device()),
            conv_history=self.qkv_conv1d.kernel_size[0] - 1,
        )

    def short_convolution(
        self,
        qkv: torch.Tensor,
        initial_state: torch.Tensor | None,
        *,
        cu_seqlens: torch.Tensor | None = None,
        return_final_state: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if initial_state is not None:
            raise ValueError("context parallelism constructs the short-convolution state")
        with kernel_stage("cp/conv/halo", self.enable_graph_annotations):
            initial_state = context_parallel_conv_history(qkv, self.routing, self.cp_group)
        return super().short_convolution(
            qkv,
            initial_state,
            cu_seqlens=cu_seqlens,
            return_final_state=return_final_state,
        )

    def kda_core(
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
        if initial_state is not None:
            raise ValueError("context parallelism constructs the KDA recurrent state")
        output, final_state = context_parallel_kda(
            q,
            k,
            v,
            gate,
            beta,
            routing=self.routing,
            group=self.cp_group,
            fastmath=self.fastmath,
            kernel_options=self.kernel_options,
        )
        return output, final_state if return_final_state else None


def run_training_step(
    module: KDAAttention,
    hidden_states: torch.Tensor,
    target: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_index: torch.Tensor,
    loss_scale: float,
    *,
    annotate: bool = False,
) -> tuple[KDAAttentionOutput, tuple[torch.Tensor, ...]]:
    """Run the complete module and differentiate its token and endpoint losses."""
    module.enable_graph_annotations = annotate
    result = module(
        hidden_states,
        cu_seqlens=cu_seqlens,
        return_final_state=True,
    )
    assert result.final_state is not None
    assert result.final_conv_state is not None
    loss = F.mse_loss(result.hidden_states.float(), target, reduction="sum") * loss_scale
    # Only true sequence ends carry a loss; intermediate subsequence states are handed downstream.
    loss = loss + 1e-4 * result.final_state[state_index].square().sum()
    loss = loss + 1e-4 * result.final_conv_state[state_index].float().square().sum()
    inputs = (hidden_states, *tuple(module.parameters()))
    with kernel_stage("cp/bwd", annotate, backward=False):
        gradients = torch.autograd.grad(loss, inputs)
    return result, gradients


def validate_against_reference(
    model: ContextParallelKDAAttention,
    reference_model: KDAAttention,
    batch: PackedTrainingBatch,
) -> None:
    """Compare one sharded full-module backward with the unsharded module."""
    result, gradients = run_training_step(
        model,
        batch.local_hidden,
        batch.local_target,
        batch.local_offsets,
        batch.terminal_index,
        batch.loss_scale,
    )
    assert batch.global_hidden is not None and batch.global_target is not None
    reference_hidden = batch.global_hidden.detach().clone().requires_grad_()
    # Unsharded, every sequence in the stream ends here, so every exit state is a true final state.
    every_sequence = torch.arange(
        batch.global_offsets.shape[0] - 1, device=reference_hidden.device
    )
    reference_result, reference_gradients = run_training_step(
        reference_model,
        reference_hidden,
        batch.global_target,
        batch.global_offsets,
        every_sequence,
        batch.loss_scale,
    )

    # Rank partitioning changes accumulation order, so compare at input-precision accuracy.
    assert_close = partial(
        torch.testing.assert_close,
        atol=torch.finfo(model.compute_dtype).eps,
        rtol=torch.finfo(model.compute_dtype).eps,
    )
    assert_close(result.hidden_states, reference_result.hidden_states[:, batch.token_ids])
    assert_close(gradients[0], reference_gradients[0][:, batch.token_ids])
    assert_close(
        result.final_state[batch.terminal_index],
        reference_result.final_state[batch.terminal_sequences],
    )
    assert_close(
        result.final_conv_state[batch.terminal_index],
        reference_result.final_conv_state[batch.terminal_sequences],
    )
    for actual, expected in zip(gradients[1:], reference_gradients[1:], strict=True):
        reduced = actual.detach().clone()
        dist.all_reduce(reduced)
        assert_close(reduced, expected)


def validate_cuda_graph(
    make_model: Callable[[], ContextParallelKDAAttention],
    eager_model: ContextParallelKDAAttention,
    batch: PackedTrainingBatch,
    annotations: bool,
    profile_path: Path | None,
    device: torch.device,
    warmup_steps: int,
) -> None:
    """Capture the full distributed backward and compare a changed-input replay."""
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        # Parameter AccumulateGrad nodes retain their first stream, so the captured path needs
        # a fresh model whose warmup and eager replay oracle both run on this stream.
        model = make_model()
        model.load_state_dict(eager_model.state_dict())
        capture_hidden = batch.local_hidden.detach().clone().requires_grad_()
        # The warmup step doubles as the replay oracle: run it on the changed input the replay
        # will see, so its activations are gone before the graph pool holds the captured ones.
        eager_hidden = (capture_hidden.detach() * 0.75).requires_grad_()
        eager_result, eager_gradients = run_training_step(
            model,
            eager_hidden,
            batch.local_target,
            batch.local_offsets,
            batch.terminal_index,
            batch.loss_scale,
        )
        del eager_hidden
    warmup_stream.synchronize()
    gc.collect()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=warmup_stream, enable_annotations=annotations):
            graph_result, graph_gradients = run_training_step(
                model,
                capture_hidden,
                batch.local_target,
                batch.local_offsets,
                batch.terminal_index,
                batch.loss_scale,
                annotate=annotations,
            )
        torch.cuda.current_stream().wait_stream(warmup_stream)
        captured_output = graph_result.hidden_states.clone()
        torch.cuda.synchronize(device)

        with torch.no_grad():
            capture_hidden.mul_(0.75)
        graph.replay()
        torch.cuda.synchronize(device)
        if torch.equal(graph_result.hidden_states, captured_output):
            raise AssertionError("CUDA Graph replay did not observe changed inputs")
        torch.testing.assert_close(graph_result, eager_result)
        torch.testing.assert_close(graph_gradients, eager_gradients)
        if profile_path is not None:
            merged_path = record_distributed_profile(
                graph.replay,
                profile_path,
                "cuda_graph_replay",
                device,
                warmup_steps=warmup_steps,
            )
            if merged_path is not None:
                print(f"profile={merged_path}", flush=True)
    finally:
        torch.cuda.synchronize(device)
        graph.reset()
        gc.collect()


def profile_eager_step(
    model: ContextParallelKDAAttention,
    batch: PackedTrainingBatch,
    profile_path: Path,
    device: torch.device,
    warmup_steps: int,
) -> None:
    """Profile one complete eager context-parallel forward and backward."""
    hidden_states = batch.local_hidden.detach().clone().requires_grad_()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    resident = torch.cuda.memory_allocated(device)
    merged_path = record_distributed_profile(
        lambda: run_training_step(
            model,
            hidden_states,
            batch.local_target,
            batch.local_offsets,
            batch.terminal_index,
            batch.loss_scale,
        ),
        profile_path,
        "iteration",
        device,
        warmup_steps=warmup_steps,
    )
    step_peak = torch.cuda.max_memory_allocated(device) - resident
    print(
        f"rank {dist.get_rank()}: step peak {step_peak / 2**30:.2f} GiB above "
        f"{resident / 2**30:.2f} GiB resident",
        flush=True,
    )
    if merged_path is not None:
        print(f"profile={merged_path}", flush=True)


def capture_training_graph(
    make_model: Callable[[], ContextParallelKDAAttention],
    eager_model: ContextParallelKDAAttention,
    batch: PackedTrainingBatch,
) -> tuple[torch.cuda.CUDAGraph, ContextParallelKDAAttention, torch.Tensor]:
    """Capture one full training step.

    Returns the graph with the model and static input it reads from; callers
    must keep both alive for as long as they replay.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # See validate_cuda_graph: AccumulateGrad nodes pin their first stream.
        model = make_model()
        model.load_state_dict(eager_model.state_dict())
        static_hidden = batch.local_hidden.detach().clone().requires_grad_()
        run_training_step(
            model,
            static_hidden,
            batch.local_target,
            batch.local_offsets,
            batch.terminal_index,
            batch.loss_scale,
        )
    stream.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run_training_step(
            model,
            static_hidden,
            batch.local_target,
            batch.local_offsets,
            batch.terminal_index,
            batch.loss_scale,
        )
    torch.cuda.current_stream().wait_stream(stream)
    return graph, model, static_hidden


def run_benchmark(
    make_model: Callable[[], ContextParallelKDAAttention],
    model: ContextParallelKDAAttention,
    batch: PackedTrainingBatch,
    device: torch.device,
    *,
    steps: int,
    warmup_steps: int,
    cuda_graph: bool,
    output_path: Path,
    config: dict[str, object],
) -> None:
    """Time steady-state training steps and record per-rank memory; rank 0 writes JSON.

    Step time is measured per rank with CUDA events; the reported step time is
    the slowest rank per iteration, which is what gates training. Memory is the
    peak above what is resident before the timed steps (parameters, inputs, and
    the graph's static buffers when captured).
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    hidden_states = batch.local_hidden.detach().clone().requires_grad_()
    if cuda_graph:
        graph, graph_model, static_hidden = capture_training_graph(make_model, model, batch)
        step: Callable[[], object] = graph.replay
    else:

        def step() -> object:
            return run_training_step(
                model,
                hidden_states,
                batch.local_target,
                batch.local_offsets,
                batch.terminal_index,
                batch.loss_scale,
            )

    for _ in range(warmup_steps):
        step()
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    torch.cuda.reset_peak_memory_stats(device)
    resident = torch.cuda.memory_allocated(device)

    step_ms: list[float] = []
    for _ in range(steps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        step()
        end.record()
        end.synchronize()
        step_ms.append(start.elapsed_time(end))
    peak = torch.cuda.max_memory_allocated(device)
    if cuda_graph:
        del graph, graph_model, static_hidden
    local = {
        "rank": rank,
        "host": os.uname().nodename,
        "local_tokens": int(batch.token_ids.numel()),
        "step_ms": step_ms,
        "resident_bytes": resident,
        "peak_bytes": peak,
        "step_peak_bytes": peak - resident,
        "reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    gathered: list[dict | None] = [None] * world_size
    dist.all_gather_object(gathered, local)
    if rank != 0:
        return
    ranks = [entry for entry in gathered if entry is not None]
    slowest = [max(entry["step_ms"][index] for entry in ranks) for index in range(steps)]
    mean = sum(slowest) / steps
    std = (sum((value - mean) ** 2 for value in slowest) / max(steps - 1, 1)) ** 0.5
    tokens = int(config["tokens"])
    report = {
        **config,
        "world_size": world_size,
        "mode": "cuda_graph" if cuda_graph else "eager",
        "steps": steps,
        "warmup_steps": warmup_steps,
        "step_ms_mean": mean,
        "step_ms_std": std,
        "step_ms_min": min(slowest),
        "tokens_per_s": tokens / (mean / 1e3),
        "step_peak_gib_max": max(entry["step_peak_bytes"] for entry in ranks) / 2**30,
        "resident_gib_max": max(entry["resident_bytes"] for entry in ranks) / 2**30,
        "peak_gib_max": max(entry["peak_bytes"] for entry in ranks) / 2**30,
        # Graph replays allocate from the graph's private pool, which
        # memory_allocated does not see; reserved is the footprint that matters there.
        "reserved_gib_max": max(entry["reserved_bytes"] for entry in ranks) / 2**30,
        "ranks": ranks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=1))
    print(
        f"benchmark: W={world_size} tokens={tokens} ({tokens // world_size}/rank) "
        f"{report['mode']} step {mean:.2f}±{std:.2f} ms  {report['tokens_per_s'] / 1e6:.2f} Mtok/s  "
        f"peak {report['peak_gib_max']:.2f} GiB allocated (step +{report['step_peak_gib_max']:.2f}), "
        f"{report['reserved_gib_max']:.2f} GiB reserved -> {output_path}",
        flush=True,
    )


def main(
    sequence_lengths: Annotated[
        str,
        typer.Option(help="Comma-separated nonempty packed sequence lengths."),
    ] = "256,1280,512",
    partition: Annotated[
        PartitionOption,
        typer.Option(help="How ranks own fragments of the global stream."),
    ] = PartitionOption.CONTIGUOUS,
    hidden_size: Annotated[int, typer.Option(min=1, help="Transformer hidden size.")] = 256,
    heads: Annotated[int, typer.Option(min=1, help="Number of KDA heads.")] = 2,
    compute_dtype: Annotated[
        ComputeDTypeOption,
        typer.Option(help="Use float16 or bfloat16 projection and kernel inputs."),
    ] = ComputeDTypeOption.BFLOAT16,
    short_conv_kernel_size: Annotated[
        int,
        typer.Option(min=1, help="Causal Q/K/V convolution width."),
    ] = 4,
    kda_backend: Annotated[
        KDABackendOption,
        typer.Option(help="Chunk backend for each rank's local KDA pass."),
    ] = KDABackendOption.FUSED,
    cuda_graph: Annotated[
        bool,
        typer.Option(help="Capture forward and backward and validate a changed-input replay."),
    ] = False,
    validate: Annotated[
        bool,
        typer.Option(help="Compare against the unsharded module on the whole stream."),
    ] = True,
    profile: Annotated[
        bool,
        typer.Option(help="Export a merged native Perfetto trace with transformer-nuggets."),
    ] = False,
    warmup_steps: Annotated[
        int,
        typer.Option(
            min=0,
            help="Steps run before the profiled one and dropped from the trace, so it shows "
            "steady state (as if deep inside a model) rather than launch skew.",
        ),
    ] = 5,
    benchmark_steps: Annotated[
        int,
        typer.Option(
            min=0,
            help="Timed steady-state steps after warmup; rank 0 writes data/kda_cp_scaling_*.json "
            "with step time, throughput, and per-rank memory (0 disables).",
        ),
    ] = 0,
) -> None:
    """Run and validate the complete packed context-parallel KDA module."""
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    cp_rank = dist.get_rank()
    world_size = dist.get_world_size()
    graph_annotations_available = False
    if cuda_graph:
        try:
            from torch.cuda.graph_annotations import is_available
        except ImportError:
            pass
        else:
            graph_annotations_available = is_available()
        if cp_rank == 0 and not graph_annotations_available:
            print(
                "CUDA Graph capture is available, but kernel labels require PyTorch graph "
                "annotations and CUDA >=13.1 tools-ID driver support.",
                flush=True,
            )

    lengths = tuple(int(length) for length in sequence_lengths.split(","))
    cu_seqlens_global = tuple(accumulate(lengths, initial=0))
    tokens = cu_seqlens_global[-1]
    plan = ContextParallelPlan.from_fragments(
        cu_seqlens_global, fragments(tokens, world_size, partition), cp_rank
    )
    token_ids = plan.global_token_ids(device)

    torch.manual_seed(0)
    model_options = {
        "hidden_size": hidden_size,
        "num_heads": heads,
        "head_dim": 128,
        "short_conv_kernel_size": short_conv_kernel_size,
        "backend": "fused",
        "compute_dtype": getattr(torch, compute_dtype.value),
        "device": device,
    }
    make_model = partial(
        ContextParallelKDAAttention,
        **model_options,
        group=dist.group.WORLD,
        plan=plan,
        kernel_options={"backend": kda_backend.value},
    )
    model = make_model()

    torch.manual_seed(123)
    global_hidden = torch.randn(1, tokens, hidden_size, device=device)
    global_target = torch.randn_like(global_hidden)
    batch = PackedTrainingBatch(
        # Only the reference needs the whole stream; drop it when not validating so large runs fit.
        global_hidden=global_hidden if validate else None,
        global_target=global_target if validate else None,
        global_offsets=torch.tensor(cu_seqlens_global, dtype=torch.int32, device=device),
        local_hidden=global_hidden[:, token_ids].clone().requires_grad_(),
        local_target=global_target[:, token_ids].clone(),
        local_offsets=model.routing.cu_seqlens,
        token_ids=token_ids,
        terminal_index=torch.tensor(plan.terminal, dtype=torch.long, device=device),
        terminal_sequences=torch.tensor(
            [plan.subsequences[index].sequence for index in plan.terminal],
            dtype=torch.long,
            device=device,
        ),
        loss_scale=1.0 / global_target.numel(),
    )
    del global_hidden, global_target
    if validate:
        reference_model = KDAAttention(**model_options)
        reference_model.load_state_dict(model.state_dict())
        validate_against_reference(model, reference_model, batch)
        del reference_model
    else:
        # Warm up compilation and autotuning so the profiled step measures only the model.
        run_training_step(
            model,
            batch.local_hidden,
            batch.local_target,
            batch.local_offsets,
            batch.terminal_index,
            batch.loss_scale,
        )
    gc.collect()

    profile_mode = "cuda_graph" if cuda_graph else "eager"
    profile_path = Path(
        "data",
        f"kda_context_parallel_{profile_mode}_{partition.value}_{kda_backend.value}"
        f"_{compute_dtype.value}_w{world_size}_t{tokens}_h{heads}_c{hidden_size}"
        f"_conv{short_conv_kernel_size}",
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
            output_path=Path(
                "data",
                f"kda_cp_scaling_{profile_mode}_{partition.value}_{kda_backend.value}"
                f"_{compute_dtype.value}_w{world_size}_t{tokens}_h{heads}_c{hidden_size}.json",
            ).resolve(),
            config={
                "tokens": tokens,
                "sequence_lengths": list(lengths),
                "partition": partition.value,
                "hidden_size": hidden_size,
                "heads": heads,
                "head_dim": 128,
                "compute_dtype": compute_dtype.value,
                "short_conv_kernel_size": short_conv_kernel_size,
                "kda_backend": kda_backend.value,
                "device_name": torch.cuda.get_device_name(device),
                "torch": torch.__version__,
            },
        )
    elif cuda_graph:
        validate_cuda_graph(
            make_model,
            model,
            batch,
            graph_annotations_available,
            profile_path if profile else None,
            device,
            warmup_steps,
        )
    elif profile:
        profile_eager_step(model, batch, profile_path, device, warmup_steps)

    mode = " with CUDA Graph replay" if cuda_graph else ""
    status = "passed" if validate else "ran"
    print(
        f"rank {cp_rank}: full packed KDA CP ({partition.value}, {kda_backend.value}) {status}{mode}",
        flush=True,
    )
    torch.cuda.synchronize(device)
    dist.destroy_process_group()


if __name__ == "__main__":
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("launch with torchrun --standalone --nproc-per-node=<world_size>")
    typer.run(main)
