# --8<-- [start:hello-world]
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise
from pathlib import Path
from typing import Literal, TypeVar

import torch
import typer
from torch import nn
from torch.cuda.graph_annotations import mark_kernels
from torch.nn.attention.varlen import varlen_attn

Tensor = torch.Tensor
GraphOutput = TypeVar("GraphOutput")
TraceFormat = Literal["chrome_json", "track_event"]
TRACE_PATH = Path(__file__).resolve().parents[1]


class VarLenAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: int,
        *,
        mask_inactive_capacity: bool = False,
    ) -> Tensor:
        tokens = x.shape[0]
        qkv = self.qkv(x).view(tokens, 3, self.num_heads, self.head_dim)
        active = None
        if mask_inactive_capacity:
            active = torch.arange(tokens, device=x.device, dtype=torch.int32) < cu_seqlens[-1]
            qkv = torch.where(active[:, None, None, None], qkv, qkv.detach())
        q, k, v = qkv.unbind(dim=1)
        attn = varlen_attn(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
        )
        if active is not None:
            attn = torch.where(active[:, None, None], attn, 0)
        return self.out_proj(attn.reshape(tokens, self.dim))


def get_zipf_tokens(
    total_tokens: int, n_seqs: int, device: torch.device | str = "cuda"
) -> tuple[Tensor, int]:
    """Distribute tokens with randomized Zipf-ranked sequence lengths."""
    if total_tokens < n_seqs:
        raise ValueError("total_tokens must provide at least one token per sequence")

    ranks = torch.randperm(n_seqs).add(1)
    weights = ranks.to(torch.float64).pow(-1.2)
    scaled_lengths = weights * ((total_tokens - n_seqs) / weights.sum())
    extra_lengths = scaled_lengths.floor().to(torch.int32)
    remainder = total_tokens - n_seqs - int(extra_lengths.sum())
    if remainder:
        fractional_order = (scaled_lengths - extra_lengths).argsort(descending=True)
        extra_lengths[fractional_order[:remainder]] += 1

    lengths = extra_lengths.add(1)
    cu_seqlens = torch.cat(
        (torch.zeros(1, dtype=torch.int32), lengths.cumsum(0, dtype=torch.int32))
    ).to(device)
    return cu_seqlens, int(lengths.max())


def hello_world() -> Tensor:
    from transformer_nuggets.utils.benchmark import profiler

    total_tokens = 4096
    dim = 512
    num_heads = 8
    n_seqs = 32

    cu_seqlens, max_seqlen = get_zipf_tokens(total_tokens, n_seqs)
    model = VarLenAttention(dim, num_heads).cuda().to(torch.bfloat16)
    inputs = torch.randn(total_tokens, dim, device="cuda", dtype=torch.bfloat16)
    # Warmup then run
    for _ in range(5):
        out = model(inputs, cu_seqlens, max_seqlen)
        torch.autograd.grad(out, model.parameters(), torch.ones_like(out), retain_graph=True)
    torch.cuda.synchronize()
    with profiler(TRACE_PATH / "docs/assets/traces/hello_world_no_cuda_graphs"):  # (1)!
        out = model(inputs, cu_seqlens, max_seqlen)
        torch.autograd.grad(out, model.parameters(), torch.ones_like(out), retain_graph=True)


# --8<-- [end:hello-world]


# --8<-- [start:capture-graph]
def capture_graph(
    function: Callable[[], GraphOutput],
    warmup: int = 3,
    enable_annotations: bool = False,
) -> tuple[torch.cuda.CUDAGraph, GraphOutput]:
    stream = torch.cuda.Stream()  # (1)!
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            function()
    torch.cuda.current_stream().wait_stream(stream)  # (2)!

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(
        graph,
        stream=stream,
        enable_annotations=enable_annotations,
    ):
        output = function()
    torch.cuda.current_stream().wait_stream(stream)
    return graph, output


# --8<-- [end:capture-graph]


# --8<-- [start:hello-world-graph]
def hello_world_graph() -> Tensor:
    from transformer_nuggets.utils.benchmark import profiler

    total_tokens = 4096
    dim = 512
    num_heads = 8
    n_seqs = 32

    cu_seqlens, max_seqlen = get_zipf_tokens(total_tokens, n_seqs)
    model = VarLenAttention(dim, num_heads).cuda().to(torch.bfloat16)
    inputs = torch.randn(total_tokens, dim, device="cuda", dtype=torch.bfloat16)
    parameters = tuple(model.parameters())
    grad_output = torch.ones_like(inputs)

    def forward_backward() -> Tensor:
        output = model(inputs, cu_seqlens, max_seqlen)
        torch.autograd.grad(output, parameters, grad_output, retain_graph=True)
        return output

    graph, output = capture_graph(forward_backward)
    torch.cuda.synchronize()
    with profiler(
        TRACE_PATH / "docs/assets/traces/hello_world_with_cuda_graphs",
        warmup=1,
    ) as active_profiler:
        graph.replay()
        active_profiler.step()
        graph.replay()  # (1)!
        active_profiler.step()
    return output


# --8<-- [end:hello-world-graph]


# --8<-- [start:training-batches]
@dataclass(frozen=True, slots=True)
class PackedBatch:
    """A fixed-capacity CPU batch with variable active tokens and sequences."""

    input_ids: Tensor
    labels: Tensor
    loss_mask: Tensor
    cu_seqlens: Tensor
    active_tokens: int
    active_sequences: int
    max_seqlen: int


def packed_batch_loader(
    num_batches: int,
    token_capacity: int,
    vocab_size: int,
    sequence_capacity: int,
) -> Iterator[PackedBatch]:
    active_token_counts = torch.linspace(
        token_capacity,
        max(sequence_capacity, token_capacity // 8),
        num_batches,
        dtype=torch.int64,
    )
    for active_tokens_tensor in active_token_counts:
        active_tokens = int(active_tokens_tensor)
        active_sequences = max(
            1,
            round(sequence_capacity * active_tokens / token_capacity),
        )
        active_cu_seqlens, actual_max_seqlen = get_zipf_tokens(
            active_tokens,
            active_sequences,
            device="cpu",
        )

        cu_seqlens = torch.full(
            (sequence_capacity + 1,),
            active_tokens,
            dtype=torch.int32,
        )
        cu_seqlens[: active_sequences + 1].copy_(active_cu_seqlens)
        loss_mask = torch.zeros(token_capacity, dtype=torch.bool)
        for start, end in pairwise(active_cu_seqlens):
            response_start = int(start) + (int(end) - int(start)) // 2
            loss_mask[response_start : int(end)] = True

        yield PackedBatch(
            input_ids=torch.randint(vocab_size, (token_capacity,), pin_memory=True),
            labels=torch.randint(vocab_size, (token_capacity,), pin_memory=True),
            loss_mask=loss_mask.pin_memory(),
            cu_seqlens=cu_seqlens.pin_memory(),
            active_tokens=active_tokens,
            active_sequences=active_sequences,
            max_seqlen=actual_max_seqlen,
        )


# --8<-- [end:training-batches]


# --8<-- [start:realistic-training-loop]
def hello_world_training_loop(
    *,
    enable_graph_annotations: bool = False,
    trace_path: Path | None = None,
    trace_format: TraceFormat = "track_event",
    fix_overlapping_events: bool = True,
) -> Tensor:
    token_capacity = 4096  # (1)!
    dim = 4096
    num_heads = 32
    sequence_capacity = 32  # (2)!
    max_seqlen = token_capacity  # (3)!
    num_batches = 4

    batches = packed_batch_loader(
        num_batches,
        token_capacity,
        dim,
        sequence_capacity,
    )
    first_batch = next(batches)  # (4)!
    static_input_ids = torch.empty_like(first_batch.input_ids, device="cuda")
    static_labels = torch.empty_like(first_batch.labels, device="cuda")
    static_loss_mask = torch.empty_like(first_batch.loss_mask, device="cuda")
    static_cu_seqlens = torch.empty_like(first_batch.cu_seqlens, device="cuda")
    static_input_ids.copy_(first_batch.input_ids, non_blocking=True)
    static_labels.copy_(first_batch.labels, non_blocking=True)
    static_loss_mask.copy_(first_batch.loss_mask, non_blocking=True)
    static_cu_seqlens.copy_(first_batch.cu_seqlens, non_blocking=True)
    torch.cuda.synchronize()

    embedding = nn.Embedding(dim, dim, device="cuda", dtype=torch.bfloat16)
    model = VarLenAttention(dim, num_heads).cuda().to(torch.bfloat16)
    parameters = (*embedding.parameters(), *model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=1e-3, fused=True)

    def forward_backward() -> tuple[Tensor, tuple[Tensor, ...]]:
        with mark_kernels("embedding"):
            inputs = embedding(static_input_ids)
        with mark_kernels("attention"):  # (5)!
            output = model(
                inputs,
                static_cu_seqlens,
                max_seqlen,
                mask_inactive_capacity=True,
            )
        with mark_kernels("loss"):
            token_losses = torch.nn.functional.cross_entropy(
                output.float(), static_labels, reduction="none"
            )
            loss = torch.where(static_loss_mask, token_losses, 0).sum() / static_loss_mask.sum()
        with mark_kernels("backward", backward=False):
            grads = torch.autograd.grad(loss, parameters)  # (6)!
        return loss, grads

    def eager_optimizer_step(graph_grads: tuple[Tensor, ...]) -> None:
        for parameter, grad in zip(parameters, graph_grads, strict=True):
            parameter.grad = grad  # (8)!
        optimizer.step()  # (9)!

    trace_profiler = training_loop_profiler(
        trace_path=trace_path,
        trace_format=trace_format,
        fix_overlapping_events=fix_overlapping_events,
    )
    graph, (loss, graph_grads) = capture_graph(
        forward_backward,
        enable_annotations=enable_graph_annotations,
    )
    graph.replay()  # (7)!
    eager_optimizer_step(graph_grads)
    torch.cuda.synchronize()
    with trace_profiler:
        for _ in range(num_batches - 1):
            with torch.profiler.record_function("data_loading"):
                batch = next(batches)
            batch_shape = (
                f"L={batch.active_tokens}, M={batch.active_sequences}, "
                f"max_seqlen={batch.max_seqlen}"
            )
            with torch.profiler.record_function(f"copy_to_static[{batch_shape}]"):
                static_input_ids.copy_(batch.input_ids, non_blocking=True)
                static_labels.copy_(batch.labels, non_blocking=True)
                static_loss_mask.copy_(batch.loss_mask, non_blocking=True)
                static_cu_seqlens.copy_(batch.cu_seqlens, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)  # (10)!
            with torch.profiler.record_function("fwd_bwd_replay"):
                graph.replay()
            with torch.profiler.record_function("optimizer_step"):
                eager_optimizer_step(graph_grads)
        torch.cuda.synchronize()
    return loss


# --8<-- [end:realistic-training-loop]


def training_loop_profiler(
    *,
    trace_path: Path | None = None,
    trace_format: TraceFormat = "track_event",
    fix_overlapping_events: bool = True,
):
    from transformer_nuggets.utils.benchmark import profiler

    if trace_path is None:
        trace_path = TRACE_PATH / "docs/assets/traces/hello_world_training_loop"
    return profiler(
        trace_path,
        record_shapes=False,
        trace_format=trace_format,
        gzip_trace=trace_format == "chrome_json",
        fix_overlapping_events=fix_overlapping_events,
    )


class Example(str, Enum):
    HELLO_WORLD = "hw"
    HELLO_WORLD_GRAPH = "hwg"
    HELLO_WORLD_TRAINING = "hwt"


def main(example: Example) -> None:
    match example:
        case Example.HELLO_WORLD:
            hello_world()
        case Example.HELLO_WORLD_GRAPH:
            hello_world_graph()
        case Example.HELLO_WORLD_TRAINING:
            hello_world_training_loop()
    print("Jobs Done!")


if __name__ == "__main__":
    typer.run(main)
