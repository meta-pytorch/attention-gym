"""CuTe DSL (SM100/SM103) backend for paged MLA decode.

Delegates to FlashAttention-4's absorbed MLA path (``qv``): FA4 sums
``q_rope @ rope^T`` and ``q_latent @ latent^T`` into one score accumulator and uses the
latent rows as values, which is the 576-wide latent dot product split into its rope and
latent parts. FA4 owns compilation caching, split-KV, and the paged gather. The backend calls
FA4's inner forward, ``_flash_attn_fwd``: it takes the same config as
``flash_attn_varlen_func`` without the autograd wrapper, which costs about 25 us of host time
per call to save inputs for a backward that decode never runs.

By default FA4 picks the kernel and split count. ``autotune=True`` benchmarks FA4's pick
against explicit ``MlaFwdConfig`` candidates (2CTA, 1CTA, and 1CTA with 64-key blocks, over
split counts and schedulers) and launches the fastest. Winners are cached per workload bucket
(``PagedMLADecodeFA4.tuning_key``), so changing decode batches reuse a winner while longer
contexts re-tune.

Constraints
-----------
- latent_dim = 512, rope_dim = 64, at most 128 heads
- dtype = bfloat16, SM100 or SM103 (compute capability 10.0 or 10.3)
"""

from __future__ import annotations

import inspect
import math
import statistics
from collections.abc import Callable
from itertools import product
from pathlib import Path
from typing import Any, NamedTuple

import torch

from attn_gym._backends.cute import run_tunable
from attn_gym._backends.cute._key import source_fingerprint

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3))


def _validate_inputs(q_latent: torch.Tensor, q_rope: torch.Tensor) -> None:
    """Raise for metadata this backend cannot run."""
    if q_latent.dtype != torch.bfloat16:
        raise TypeError(f"CuTe paged MLA decode requires bfloat16, got {q_latent.dtype}.")
    if q_latent.shape[-1] != 512 or q_rope.shape[-1] != 64:
        raise ValueError(
            "CuTe paged MLA decode requires latent_dim=512 and rope_dim=64, "
            f"got {q_latent.shape[-1]} and {q_rope.shape[-1]}."
        )
    if q_latent.shape[1] > 128:
        raise ValueError(
            f"CuTe paged MLA decode supports at most 128 query heads, got {q_latent.shape[1]}."
        )


def _fa4_forward(*_static_args: Any) -> Any:
    from flash_attn.cute.interface import _flash_attn_fwd

    return _flash_attn_fwd  # FA4 compiles on first launch and caches the kernel itself.


# Saved winners are keyed on this namespace, so they re-tune when FA4's sources (the directory
# of its interface module; local rebuilds keep the version string), this module, or the CuTe
# runtime change.
_fa4_forward.cache_namespace = lambda: source_fingerprint(  # type: ignore[attr-defined]
    _fa4_forward, (str(Path(inspect.getfile(_fa4_forward())).parent),)
)


class PagedMLADecodeFA4:
    """``run_tunable`` adapter over FA4's absorbed MLA forward.

    A config is an FA4 ``MlaFwdConfig`` or ``"auto"`` for FA4's per-call pick (``None``
    would read as a missing winner).
    """

    class Args(NamedTuple):
        # max_seqlen_k only keys winners: passing it to FA4 slowed short contexts (2-token rows
        # over 512 keys: 22.5 us to 36.8 us).
        q_latent: torch.Tensor
        q_rope: torch.Tensor
        latent_cache: torch.Tensor
        rope_cache: torch.Tensor
        block_table: torch.Tensor
        cu_seqlens_q: torch.Tensor
        seqused_k: torch.Tensor
        max_seqlen_q: int
        max_seqlen_k: int | None
        scale: float
        return_lse: bool

    compile = staticmethod(_fa4_forward)

    @staticmethod
    def benchmark_all(launches: list[Callable[[], Any]]) -> list[float]:
        """Time CUDA graph replays of every candidate round-robin, halving the field per round.

        Graphs, because decode kernels run shorter than FA4's host launch and serving replays
        graphs. Round-robin, because power capping swings clocks by tens of percent within a
        second, which ranks sequentially timed candidates by when they ran. Each round keeps
        the faster half by median; eliminated candidates time as infinity.

        The first launch is FA4's pick (``tune`` drops no candidate: the adapter defines no
        ``benchmark_errors``). It stays in every round and wins unless another candidate is
        more than 3% faster: it re-selects per call within the bucket, and smaller leads were
        noise that did not carry over to other lengths.
        """
        # The graphs never run concurrently and nobody reads their outputs: share one pool.
        pool = torch.cuda.graph_pool_handle()
        # torch.cuda.graph's default capture stream is on whichever device captured first.
        stream = torch.cuda.Stream()
        graphs = []
        for launch in launches:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                launch()
            graphs.append(graph)
        # Flush L2 between replays: in serving, other layers evict the cache from L2.
        flush = torch.empty(
            2 * torch.cuda.get_device_properties(torch.cuda.current_device()).L2_cache_size,
            dtype=torch.int8,
            device="cuda",
        )
        samples: list[list[float]] = [[] for _ in graphs]
        alive = list(range(len(graphs)))
        repeats = 2
        while True:
            events = []
            for repeat in range(repeats):
                # Rotate the order so no candidate always runs first after the host sync.
                for index in alive[repeat % len(alive) :] + alive[: repeat % len(alive)]:
                    flush.zero_()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    graphs[index].replay()
                    end.record()
                    events.append((index, start, end))
            torch.cuda.synchronize()
            for index, start, end in events:
                samples[index].append(start.elapsed_time(end))
            medians = {index: statistics.median(samples[index]) for index in alive}
            if len(alive) <= 3 and repeats >= 16:
                break
            if len(alive) > 3:
                alive = sorted(
                    set(sorted(alive, key=medians.__getitem__)[: (len(alive) + 1) // 2]) | {0}
                )
            # Closer races need more samples; discard the previous round's.
            repeats *= 2
            samples = [[] for _ in graphs]
        if medians[0] <= 1.03 * min(medians.values()):
            return [medians[0]] + [math.inf] * (len(graphs) - 1)
        return [medians.get(index, math.inf) for index in range(len(graphs))]

    @staticmethod
    def default_config(args: Args, *, target: Any) -> str:
        return "auto"

    @staticmethod
    def tuning_key(args: Args, *, target: Any) -> tuple[int, ...]:
        # Batch and context length change every step, so bucket them to powers of two.
        # Longest sequence: the block table's capacity, lowered to the max_seqlen_k hint if given.
        keys = args.block_table.shape[1] * args.latent_cache.shape[1]
        if args.max_seqlen_k is not None:
            keys = min(args.max_seqlen_k, keys)
        return (
            args.q_latent.shape[1],
            1 << (args.max_seqlen_q - 1).bit_length(),
            # configs()' q_stage cutoff, which the row bucket misses for non-power-of-two heads.
            args.max_seqlen_q * args.q_latent.shape[1] > 128,
            args.latent_cache.shape[1],
            1 << max(args.cu_seqlens_q.shape[0] - 2, 0).bit_length(),
            1 << (keys - 1).bit_length(),
        )

    @staticmethod
    def configs(args: Args) -> list[Any]:
        """FA4's pick plus every MLA kernel FA4 accepts here, over power-of-two split counts.

        Splits go to 128 at any context length, within a 1 GiB workspace: more splits than key
        blocks measured faster on short contexts (512 keys: 16 splits 14.3 us, 4 splits 16.4 us).
        """
        from flash_attn.cute.config import MLA_KB64_TILE_MN, MlaFwdConfig

        heads = args.q_latent.shape[1]
        q_stage = 2 if args.max_seqlen_q * heads > 128 else 1
        variants = [("1cta", 128, 128, q_stage)]
        # FA4 accepts 64-key blocks at 64 heads, or fewer heads with one-token rows.
        if heads == 64 or (heads < 64 and args.max_seqlen_q == 1):
            variants.append(("1cta_kb64", *MLA_KB64_TILE_MN, 1))
        # Split-KV allocates FP32 partial outputs of splits * tokens * heads * 512 * 4 bytes.
        # Skip split counts over 1 GiB for the bucket's largest token count, so one candidate
        # cannot OOM the sweep; every benchmarked winner used under 40 MiB.
        max_tokens = (1 << (args.cu_seqlens_q.shape[0] - 2).bit_length()) * (
            1 << (args.max_seqlen_q - 1).bit_length()
        )
        candidates: list[Any] = [
            "auto",
            MlaFwdConfig("2cta", 128, 128, q_stage, 1, True, True, False),
        ]
        candidates.extend(
            MlaFwdConfig(
                kernel=kernel,
                tile_m=tile_m,
                tile_n=tile_n,
                q_stage=stages,
                num_splits=num_splits,
                pack_gqa=True,
                use_clc_scheduler=use_clc_scheduler,
                s_ahead=False,
            )
            for (kernel, tile_m, tile_n, stages), num_splits, use_clc_scheduler in product(
                variants,
                [
                    1 << log_splits
                    for log_splits in range(8)
                    if log_splits == 0 or (max_tokens * heads * 2048) << log_splits <= 1 << 30
                ],
                (False, True),
            )
        )
        return candidates

    @staticmethod
    def compile_call(config: Any, args: Args) -> tuple[Any, ...]:
        return (config,)

    @staticmethod
    def launch(
        flash_attn_fwd: Any, config: Any, args: Args
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        out, lse, *_ = flash_attn_fwd(
            args.q_rope,
            args.rope_cache.unsqueeze(2),
            args.latent_cache.unsqueeze(2),
            qv=args.q_latent,
            cu_seqlens_q=args.cu_seqlens_q,
            max_seqlen_q=args.max_seqlen_q,
            seqused_k=args.seqused_k,
            page_table=args.block_table,
            softmax_scale=args.scale,
            causal=True,
            pack_gqa=True,
            num_splits=0,
            return_lse=args.return_lse,
            config=None if config == "auto" else config,
        )
        return out, lse


def paged_mla_decode(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    latent_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    *,
    max_seqlen_q: int,
    scale: float,
    return_lse: bool,
    autotune: bool,
    max_seqlen_k: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run FA4's absorbed MLA forward over the paged latent cache."""
    _validate_inputs(q_latent, q_rope)
    args = PagedMLADecodeFA4.Args(
        q_latent,
        q_rope,
        latent_cache,
        rope_cache,
        block_table,
        cu_seqlens_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        scale,
        return_lse,
    )
    return run_tunable(
        PagedMLADecodeFA4,
        args,
        # Empty inputs have nothing to time, and their winner would claim a real bucket.
        autotune=autotune and q_latent.shape[0] > 0 and block_table.shape[1] > 0,
    )[0]


__all__ = ["SUPPORTED_CAPABILITIES", "PagedMLADecodeFA4", "paged_mla_decode"]
