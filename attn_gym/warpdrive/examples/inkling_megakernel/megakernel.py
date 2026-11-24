"""Host side of the Inkling 2 Mini decode megakernel: the whole batch-one decode step as
warpdrive launches (``InklingDecoder``), and runners of each launch kind:

- ``InklingVisitMegakernel``: a visit's attention and MLP halves in one launch
  (``visit_kernel``), with the token's embedding at layer 0 and the message gates in core
  passes after the first;
- ``InklingMsgMegakernel``: a core pass's messages and message halves (``msg_kernel``);
- ``InklingHeadMegakernel``: the final block residual, norm, LM head and softcap
  (``head_kernel``).

A step is one launch per visit (48), one per core pass after the first (2) and the head; each
visit's output stays in a per-visit buffer that later block residuals read. Launches read
every input from device memory (token and position too) and sync nothing, so a step is
CUDA-graph capturable, and every launch kind compiles once for all positions.

A visit launch's CTAs wait on each other's tagged words, so it needs its whole grid (one CTA
per SM) resident: it launches cooperatively, which the driver admits only when every CTA can
be resident (it fails with CUDA error 720 under a smaller SM limit, such as an MPS or green
context partition). The message and head launches' CTAs wait on nothing outside themselves.

The experts compute the checkpoint's declared numerics (``ref.ExpertQuant``: quantized
inputs, vLLM's SM100 semantics; ``expert_numerics``), and the launches reject layers without
them. They stream in each layer's stored format: bf16 tensors of the weights' block values
(``quant.dense_layer``), or ``quant.QuantExperts`` as ``quant.load_quantized_inkling_weights``
keeps them (NVFP4 layers 0-19, MXFP8 20-23 for Inkling 2 Mini); the quantized launches decode
them in registers to the same dot products, bit for bit, as the bf16 launches on
``QuantExperts.block_values()`` weights.

Numerics follow the oracle with ``start=None``; outputs differ by fp32 summation order (block
residuals, GEMV accumulation, convolutions, routing weights, message gates) and, in attention,
by the flash merge (probabilities are not rounded to bf16 before the value sum), which can flip
a bf16 rounding; a flipped rounding of an expert input can move its quantization (a code,
rarely a block scale or the amax).

Kernels compile from the kernel modules and their imports, so host-side edits here recompile
nothing.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import NamedTuple

import torch

from attn_gym.warpdrive.examples.inkling_megakernel import ref
from attn_gym.warpdrive.examples.inkling_megakernel.common import (
    GATE_RANK,
    GATE_TILES,
    GATES,
    MAX_BLOCKS,
    MLP_GATE,
    RING,
    WORDS,
)
from attn_gym.warpdrive.examples.inkling_megakernel.head_kernel import (
    LMHEAD,
    InklingHeadKernel,
    compile_head_kernel,
)
from attn_gym.warpdrive.examples.inkling_megakernel.mlp_half import (
    DOWN,
    GATEUP,
    ROUTER,
)
from attn_gym.warpdrive.examples.inkling_megakernel.msg_kernel import (
    MSG,
    InklingMsgKernel,
    compile_msg_kernel,
)
from attn_gym.warpdrive.examples.inkling_megakernel.ref import (
    InklingCache,
    InklingConfig,
    InklingWeights,
    LayerWeights,
)
from attn_gym.warpdrive.examples.inkling_megakernel.visit_kernel import (
    InklingVisitKernel,
    compile_visit_kernel,
    visit_kernel_class,
)
from attn_gym.warpdrive.examples.megakernel.constants import BLOCK, HEAD_DIM, ROWS
from attn_gym.warpdrive.examples.megakernel.formats import BF16, NVFP4
from attn_gym.warpdrive.examples.megakernel.launch import Launch, deal, table_of
from attn_gym.warpdrive.profiler.host import ProfileSession
from attn_gym.warpdrive.trace_session import trace_session

__all__ = [
    "GateArgs",
    "InklingDecoder",
    "InklingHeadMegakernel",
    "InklingMsgMegakernel",
    "InklingVisitMegakernel",
    "MessageGates",
    "build_visit_table",
    "expert_numerics",
    "experts_format",
    "visit_kernel_for",
]

# Default KV splits per q head. A sliding layer reads at most window / 64 + 1 = 17 blocks, so 4
# splits keep an item at 5 blocks or fewer; full layers read every cached block.
SLIDING_SPLITS, FULL_SPLITS = 4, 8


def visit_kernel_for(
    config: InklingConfig,
    full: bool,
    messages: bool = False,
    experts_format: str = BF16,
    activation_format: str = NVFP4,
) -> InklingVisitKernel:
    """The visit kernel for ``config``'s full (``full``) or sliding layers, with message gates
    (``messages``) or without, streaming experts in ``experts_format`` and quantizing their
    inputs to ``activation_format``."""
    if config.conv_kernel > RING or ref.CONV_RING != RING:
        raise ValueError(f"the kernels' convolution rings hold {RING} positions")
    if config.head_dim != HEAD_DIM:
        raise ValueError(f"head dim must be {HEAD_DIM}, got {config.head_dim}")
    return visit_kernel_class(experts_format).get_or_build(
        hidden=config.hidden_size,
        q_heads=config.num_q_heads,
        kv_heads=config.num_kv_heads,
        d_rel=config.d_rel,
        rel_width=config.rel_extent,
        extent=config.rel_extent if full else config.sliding_window,
        conv_kernel=config.conv_kernel,
        rms_eps=config.rms_eps,
        qk_clip=config.qk_clip,
        rel_clip=config.rel_clip,
        splits=FULL_SPLITS if full else SLIDING_SPLITS,
        full=full,
        width=config.moe_intermediate_size,
        experts=config.num_experts,
        top_k=config.experts_per_token,
        gate_softcap=config.gate_softcap,
        up_softcap=config.up_softcap,
        route_scale=config.route_scale,
        window=0 if full else config.sliding_window,
        log_alpha=config.log_scaling_alpha if full else 0.0,
        log_floor=config.log_scaling_floor if full else 0,
        messages=messages,
        experts_format=experts_format,
        activation_format=activation_format,
    )


def experts_format(w: LayerWeights) -> str:
    """The format of ``w``'s experts: ``"bf16"`` tensors or ``quant.QuantExperts``' format."""
    dense = isinstance(w.w13, torch.Tensor)
    if dense != isinstance(w.w2, torch.Tensor) or (not dense and w.w13.format != w.w2.format):
        raise ValueError("w13 and w2 must have the same format")
    return BF16 if dense else w.w13.format


def expert_numerics(config: InklingConfig, w: LayerWeights) -> tuple[str, torch.Tensor]:
    """``(activation_format, expert_scales)`` of ``w``'s declared expert numerics
    (``ref.ExpertQuant``), which the kernels compute: the format the experts' inputs quantize
    to, and fp32 ``(2, experts + 1)`` per matrix (w13, w2) the factors of the NVFP4 row scale
    (``ref.ExpertQuant.row_factors``). Raises for layers with HF's numerics."""
    if not w.declared_experts:
        raise ValueError(
            "the megakernels compute the checkpoint's declared expert numerics: the layer needs "
            "ref.ExpertQuant (load a checkpoint that declares them, activation_quant=None)"
        )
    activation = w.w13_quant.activation.format
    if w.w2_quant.activation.format != activation:
        raise ValueError("w13 and w2 must quantize their inputs to the same format")
    factors = [q.row_factors(config.num_experts, w.wq.device) for q in (w.w13_quant, w.w2_quant)]
    return activation, torch.stack(factors).contiguous()


def expert_args(w: LayerWeights) -> tuple[torch.Tensor, ...]:
    """``w``'s experts as the visit kernel's arguments (``mlp_half.fake_expert_args``): bf16
    tensors as they are, ``quant.QuantExperts`` as Int32 views of their weights and scales;
    placeholders for what a format has not."""
    if experts_format(w) == BF16:
        stub = torch.zeros(1, 1, 4, device=w.wq.device, dtype=torch.int32)
        return (w.w13, w.w2, stub, stub)
    pair = (w.w13, w.w2)
    weights = tuple(q.weight.view(torch.int32).view(q.experts * q.rows, -1) for q in pair)
    scales = tuple(
        q.scale.view(torch.int32).view(q.experts * q.rows // ROWS, q.subtiles, -1) for q in pair
    )
    return weights + scales


def mlp_lists(op: InklingVisitKernel, ctas: int) -> list[list[list[int]]]:
    """Each CTA's MLP rows in dependency order: ROUTER tiles, GATEUP and DOWN rows, both
    slot-major, dealt round-robin over the CTAs; a row tile's last-slot DOWN row, which
    reduces the tile, follows the rows it polls.

    Round-robin gate/up rows complete the slots in waves (slots 0 and 1 with the first row of
    every CTA), so a CTA's first down rows, of early slots, wait for nothing; only its last one
    (slot 6 or 7) waits for the last wave. Down rows start where gate/up rows end, so the
    CTAs with one row fewer of each kind are mostly different ones; the router's single-row
    items go to the least loaded CTAs. This measured about 1.5 us per launch faster than
    contiguous runs of rows per CTA, which reuse a down slot's activation fetch but leave every
    slot's down rows waiting for its last gate/up rows (CUDA-graph replay of an MLP-only
    launch, GB300, Inkling 2 Mini; 2026-11-21)."""
    gate_up = [[GATEUP, t, s] for s in range(op.top_k) for t in range(op.gate_tiles)]
    down_rows = [[DOWN, t, s] for s in range(op.top_k) for t in range(op.row_tiles)]
    lists = [[] for _ in range(ctas)]
    for j, row in enumerate(gate_up):
        lists[j % ctas].append(row)
    shift = len(gate_up) % ctas
    for j, row in enumerate(down_rows):
        lists[(j + shift) % ctas].append(row)
    cost = {GATEUP: op.hidden_k_tiles, DOWN: op.down_k_tiles}
    load = [sum(cost[row[0]] for row in rows) for rows in lists]
    for tile in range(op.experts // ROWS):
        cta = min(range(ctas), key=lambda c: (load[c], c))
        lists[cta].insert(0, [ROUTER, tile, 0])
        load[cta] += op.hidden_k_tiles
    return lists


def build_visit_table(op: InklingVisitKernel, ctas: int) -> torch.Tensor:
    """The visit launch's table for ``StaticPersistent``: every CTA's attention rows, then its
    MLP rows, each half's message-gate rows first. The attention's rows (``attention_rows``)
    and its gates' are dealt round-robin, the MLP's as ``mlp_lists``. Every CTA's rows stay in
    the global dependency order (attention before any MLP row), so with all CTAs resident the
    earliest unfinished row has its producers finished."""
    attention = deal(op.gate_rows(range(MLP_GATE)), ctas)
    attention = [g + rows for g, rows in zip(attention, deal(op.attention_rows(), ctas))]
    mlp = deal(op.gate_rows(range(MLP_GATE, GATES)), ctas)
    mlp = [g + rows for g, rows in zip(mlp, mlp_lists(op, ctas))]
    return table_of([a + m for a, m in zip(attention, mlp)], WORDS)


def bf16_router(router: torch.Tensor) -> torch.Tensor:
    """The router as bf16; the kernel streams it so, which is exact only when every fp32 weight
    is a bf16 value (Inkling 2 Mini's are)."""
    packed = router.to(torch.bfloat16)
    if not torch.equal(packed.float(), router):
        raise ValueError("the router's fp32 weights are not exactly representable in bf16")
    return packed.contiguous()


class GateArgs(NamedTuple):
    """One core layer's message gates in the kernels' form (``common.fake_gate_args``) and the
    current pass's message halves and messages of that layer."""

    down: torch.Tensor  # (GATES * GATE_RANK, 2 * hidden) bf16
    fg: torch.Tensor  # (GATES * hidden, 4 * GATE_RANK) bf16 view of the fp32 [F | G] rows
    mhalf: torch.Tensor  # (GATES * GATE_RANK,) fp32
    messages: torch.Tensor  # (GATES, hidden) fp32

    @classmethod
    def placeholder(cls, hidden: int, device) -> "GateArgs":
        """Operands of the right shapes for launches without message gates (never read)."""
        rows = GATES * GATE_RANK
        return cls(
            torch.zeros(rows, 2 * hidden, device=device, dtype=torch.bfloat16),
            torch.zeros(GATES * hidden, 4 * GATE_RANK, device=device, dtype=torch.bfloat16),
            torch.zeros(rows, device=device),
            torch.zeros(GATES, hidden, device=device),
        )


class MessageGates:
    """The message gates of the core layers ``core`` (in order) in the kernels' form and one
    pass's message buffers, which ``InklingMsgMegakernel`` fills from ``message_queries``
    (``(layers, GATES, hidden)`` fp32) and the pass's launches read (``layer``)."""

    def __init__(self, core: tuple[LayerWeights, ...], message_queries: torch.Tensor):
        down = torch.stack([torch.stack([g.down for g in w.gates]) for w in core])
        if down.shape[2] != GATE_RANK:
            raise ValueError(f"the kernels' message gates have rank {GATE_RANK}")
        layers, hidden = len(core), down.shape[-1] // 2
        self.down = down.reshape(layers * GATES * GATE_RANK, 2 * hidden).contiguous()
        fg = torch.stack(
            [torch.stack([torch.cat((g.forget, g.gain), 1) for g in w.gates]) for w in core]
        )  # (layers, GATES, hidden, 2 * GATE_RANK) fp32
        self.fg = fg.contiguous().view(torch.bfloat16).reshape(layers, GATES * hidden, -1)
        self.queries = message_queries.reshape(layers * GATES, hidden).contiguous()
        self.messages = torch.zeros_like(self.queries)
        self.mhalf = torch.zeros(layers * GATES * GATE_RANK, device=self.queries.device)

    def layer(self, core: int) -> GateArgs:
        """Core layer ``core``'s gates and the current pass's message state."""
        rows = slice(core * GATES * GATE_RANK, (core + 1) * GATES * GATE_RANK)
        messages = self.messages[core * GATES : (core + 1) * GATES]
        return GateArgs(self.down[rows], self.fg[core], self.mhalf[rows], messages)


def device_position(positions: torch.Tensor) -> torch.Tensor:
    """The one-element int32 device position the kernels read (a view when already int32)."""
    return positions.reshape(1).to(torch.int32)


@dataclass(frozen=True)
class VisitLayer:
    """A layer's arguments in the visit kernel's form: per-query block-residual queries,
    per-stream norm weights (one query in sliding layers, three in full ones; three streams in
    full layers and in launches with message gates, where sliding layers repeat their input
    norm), the dynamic tap projections (placeholders in sliding layers, which read none), the
    k/v taps, the relative-bias projection transposed (a row per distance), the router as
    bf16 and the experts' formats and scales."""

    query: torch.Tensor  # (queries, hidden) fp32; zeros (unread) at layer 0
    norms: torch.Tensor  # (queries, hidden) bf16
    gated_norms: torch.Tensor  # (3, hidden) bf16
    k_dynamic: torch.Tensor  # (kv_heads * kernel, hidden) bf16
    v_dynamic: torch.Tensor
    k_taps: torch.Tensor  # sliding: (kv_dim, kernel) fp32; full: (kernel, kv_dim) bf16
    v_taps: torch.Tensor
    rel_proj: torch.Tensor  # (rel_extent, d_rel) bf16
    router: torch.Tensor  # (experts, hidden) bf16 (``bf16_router``)
    experts_format: str  # ``experts_format``
    activation_format: str  # ``expert_numerics``
    expert_scales: torch.Tensor  # (2, experts + 1) fp32 (``expert_numerics``)
    experts: tuple[torch.Tensor, ...]  # ``expert_args``
    full: bool  # a full-attention layer (it has q/k/v norms), else a sliding one

    @classmethod
    def of(cls, config: InklingConfig, w: LayerWeights) -> "VisitLayer":
        query = w.attn_query
        full = w.q_norm is not None
        if full:
            norms = gated_norms = torch.stack((w.q_norm, w.k_norm, w.v_norm))
            k_dynamic, v_dynamic = w.k_dynamic, w.v_dynamic
            k_taps, v_taps = w.k_static, w.v_static
        else:
            if query is None:
                query = torch.zeros(1, config.hidden_size, device=w.wq.device)
            norms, gated_norms = w.input_norm[None], w.input_norm.expand(3, -1)
            k_dynamic = v_dynamic = w.wk[: config.num_kv_heads * config.conv_kernel]
            k_taps, v_taps = w.k_conv, w.v_conv
        activation_format, expert_scales = expert_numerics(config, w)
        return cls(
            query=query.contiguous(),
            norms=norms.contiguous(),
            gated_norms=gated_norms.contiguous(),
            k_dynamic=k_dynamic,
            v_dynamic=v_dynamic,
            k_taps=k_taps.contiguous(),
            v_taps=v_taps.contiguous(),
            rel_proj=w.rel_proj.T.contiguous(),
            router=bf16_router(w.router),
            experts_format=experts_format(w),
            activation_format=activation_format,
            expert_scales=expert_scales,
            experts=expert_args(w),
            full=full,
        )


class EmbedArgs(NamedTuple):
    """The token's embedding, read by layer 0's visit launch."""

    table: torch.Tensor  # (vocab rows, hidden) bf16
    norm: torch.Tensor  # (hidden,) bf16
    token: torch.Tensor  # (1,) int64


class InklingVisitMegakernel:
    """Launch state of the visit kernels for ``config`` (per layer kind: sliding or full,
    without or with message gates, and each pair of expert and activation formats of
    ``layers``) and the kernel-form
    arguments of ``layers``. Launches of one runner must run one at a time in stream order
    (each kind's launches share a scratch and an epoch). Each launch is cooperative: it
    occupies every SM until it finishes and fails (CUDA error 720) where its one-CTA-per-SM
    grid cannot be resident at once (see the module docstring). ``trace``: compile with ranges
    for the work items and ``trace_region``s, recorded by ``"globaltimer"`` (``traced``) or
    ``"iket"``; ``trace_events``: per-role capacity (see ``Launch``; raise it for long contexts)."""

    def __init__(
        self,
        config: InklingConfig,
        layers: tuple[LayerWeights, ...],
        device="cuda",
        trace: str | None = None,
        trace_events: int | None = None,
    ):
        self.config = config
        self.args = {id(w): VisitLayer.of(config, w) for w in layers}
        self.layers = layers  # keeps the ids of ``args`` valid
        formats = sorted({(a.experts_format, a.activation_format) for a in self.args.values()})
        self.launches = {
            (full, messages, *fmt): Launch(
                visit_kernel_for(config, full, messages, *fmt),
                build_visit_table,
                device,
                trace,
                trace_events,
            )
            for full in (False, True)
            for messages in (False, True)
            for fmt in formats
        }
        self.last: Launch | None = None  # the launch state of the last call
        hidden = config.hidden_size
        self.no_gates = GateArgs.placeholder(hidden, device)
        self.no_embed = EmbedArgs(
            torch.zeros(1, hidden, device=device, dtype=torch.bfloat16),
            torch.zeros(hidden, device=device, dtype=torch.bfloat16),
            torch.zeros(1, device=device, dtype=torch.int64),
        )

    def launch_of(self, w: LayerWeights, layer: int, messages: bool) -> Launch:
        """The launch state of layer ``layer`` (weights ``w``, one of the runner's layers), with
        message gates or without."""
        args = self.args.get(id(w))
        if args is None:
            raise ValueError("w must be one of the layers this runner was built with")
        if args.full != self.config.is_full(layer):
            kind = "full" if args.full else "sliding"
            raise ValueError(f"layer {layer}'s weights are a {kind} layer's, unlike config's")
        return self.launches[args.full, messages, args.experts_format, args.activation_format]

    def __call__(
        self,
        w: LayerWeights,
        layer: int,
        visit: int,
        h: torch.Tensor,
        blocks: list[torch.Tensor],
        cache: InklingCache,
        positions: torch.Tensor,
        out: torch.Tensor | None = None,
        gates: GateArgs | None = None,
        embed: EmbedArgs | None = None,
    ) -> torch.Tensor:
        """``ref.attention_block`` (``start=None``) then ``ref.mlp_block`` of one token: bf16
        ``h`` and ``blocks`` (0 to MAX_BLOCKS, the attention's; the MLP's add ``h`` when the
        layer starts a block) of ``hidden`` elements, ``visit``'s caches (written at the
        position) and the one-element device ``positions``. In core passes after the first,
        ``gates`` holds the layer's message gates (messages 0-2 gate q, k, v, message 3 the MLP
        input). With ``embed`` (layer 0 only), the launch computes ``h`` itself,
        ``ref.hidden_states``' normalized embedding of the token, and stores it into ``h``.
        Returns the bf16 ``(hidden,)`` visit output (into ``out`` if given); ``residual()``
        holds the post-attention residual.

        The position must satisfy ``0 <= pos < cache.max_len``: the caller owns it and the
        launch reads it on the device, so it is not checked here. Only ``pos >= max_len`` is
        guarded (the cache-row store skips it rather than write past the visit's rows); a
        negative position writes before the visit's cache rows and convolution rings."""
        config, hidden = self.config, self.config.hidden_size
        if cache.max_len % BLOCK:
            raise ValueError(f"the cache capacity must be a multiple of {BLOCK}")
        launch = self.launch_of(w, layer, gates is not None)
        args = self.args[id(w)]
        if embed is not None and (w.attn_query is not None or blocks):
            raise ValueError("only layer 0 (no query, no blocks) reads the embedding")
        if gates is not None and w.attn_query is None:
            raise ValueError("message gates need a layer with a block-residual query")
        boundary = layer % config.block_layers == 0
        h = h.reshape(hidden)
        states = [b.reshape(hidden) for b in blocks] + ([h] if boundary else [])
        if len(states) > MAX_BLOCKS:
            raise ValueError(f"the MLP half reads at most {MAX_BLOCKS} blocks")
        states += [h] * (MAX_BLOCKS - len(states))  # unread
        if out is None:
            out = torch.empty(hidden, device=h.device, dtype=torch.bfloat16)
        self.last = launch
        table, norm, token = embed or self.no_embed
        compile_visit_kernel(launch.op)(
            w.wq,
            w.wk,
            w.wv,
            w.wrel,
            args.k_dynamic,
            args.v_dynamic,
            w.wgate,
            w.wsink,
            w.wo,
            cache.k[visit].view(-1, HEAD_DIM),
            cache.v[visit].view(-1, HEAD_DIM),
            args.query,
            args.norms if gates is None else args.gated_norms,
            w.q_scale,
            w.rel_scale,
            args.rel_proj,
            args.k_taps,
            args.v_taps,
            w.attn_conv,
            args.router,
            *args.experts,
            w.router_bias,
            args.expert_scales,
            w.mlp_query[None],
            w.post_norm[None],
            w.mlp_conv,
            *(gates or self.no_gates),
            table,
            norm,
            h,
            *states,
            len(blocks),
            int(w.attn_query is not None),
            int(boundary),
            int(embed is not None),
            cache.conv_k[visit],
            cache.conv_v[visit],
            cache.conv_attn[visit],
            cache.conv_mlp[visit],
            token.reshape(1),
            device_position(positions),
            out,
            launch.trace_buffer,
            launch.scratch,
            launch.state,
            launch.table,
            launch.workspace,
            launch.max_ctas,
        )
        return out

    @contextmanager
    def traced(
        self, w: LayerWeights, layer: int, messages: bool = False
    ) -> Iterator[ProfileSession]:
        """A ``trace_session`` of a ``trace="globaltimer"`` runner's launch of layer ``layer``
        (weights ``w``, with message gates or without): launch it once inside the block; after
        exit, ``session.ranges`` hold its ranges (work-item payloads index ``launch_of(...)
        .table``)."""
        launch = self.launch_of(w, layer, messages)
        placeholder = launch.trace_buffer
        with torch.cuda.device(launch.device):  # the session allocates on the current device
            compile_visit_kernel(launch.op)  # the session reads range names from the compile
            with trace_session(launch.op, launch.max_ctas) as session:
                launch.trace_buffer = session.prof.tensor
                try:
                    yield session
                finally:
                    launch.trace_buffer = placeholder

    def residual(self) -> torch.Tensor:
        """The fp32 ``(hidden,)`` post-attention residual of the last launch, from its tagged
        words (valid until the next launch)."""
        return self.last.values("resid")

    def activations(self) -> torch.Tensor:
        """The bf16 ``(experts_per_token, width)`` down inputs of the last launch (``bf16(swiglu
        * routing weight)`` per slot, the experts ascending), from its tagged words (valid
        until the next launch)."""
        return self.last.values("act").view(self.last.op.top_k, -1).to(torch.bfloat16)

    def routed_experts(self, w: LayerWeights) -> torch.Tensor:
        """The ``(experts_per_token,)`` experts the last launch (which must be of layer weights
        ``w``) routed to, ascending: the top-k of its published router logits' scores plus ``w``'s
        correction bias, recomputed here with the kernel's tie rule (equal keys go to the lower
        expert index; ``torch.topk`` does not promise an order among ties). A diagnostic of the
        published logits, not the IDs the kernel dispatched: each CTA selects those in shared
        memory and publishes none."""
        keys = torch.sigmoid(self.last.values("logits")) + w.router_bias
        order = torch.sort(keys, descending=True, stable=True).indices
        return order[: self.last.op.top_k].sort().values


class InklingMsgMegakernel:
    """Launch state of the message kernel over ``gates``' core layers: each call fills
    ``gates.messages`` and ``gates.mhalf`` for the next pass."""

    def __init__(self, config: InklingConfig, gates: MessageGates, device="cuda"):
        op = InklingMsgKernel.get_or_build(hidden=config.hidden_size, rms_eps=config.rms_eps)
        tiles = gates.queries.shape[0] * GATE_TILES

        def build(op, ctas):
            return table_of(deal([[MSG, t, 0] for t in range(tiles)], ctas), WORDS)

        self.launch = Launch(op, build, device)
        self.gates = gates

    def __call__(self, block: torch.Tensor, partial: torch.Tensor):
        """The messages of the previous pass's bf16 ``block`` (its in12) and ``partial``
        (its out17)."""
        g, launch = self.gates, self.launch
        compile_msg_kernel(launch.op)(
            g.down,
            g.queries,
            block.reshape(-1),
            partial.reshape(-1),
            g.messages,
            g.mhalf,
            launch.table,
            launch.workspace,
            launch.max_ctas,
        )


class InklingHeadMegakernel:
    """Launch state of the head kernel (final block residual, norm, LM head, softcap) for
    ``weights``."""

    def __init__(self, weights: InklingWeights, device="cuda"):
        config = weights.config
        op = InklingHeadKernel.get_or_build(
            hidden=config.hidden_size,
            vocab=config.vocab_size,
            rms_eps=config.rms_eps,
            softcap=config.logit_softcap,
        )

        def build(op, ctas):
            return table_of(deal([[LMHEAD, t, 0] for t in range(op.row_tiles)], ctas), WORDS)

        self.launch = Launch(op, build, device)
        self.weights = weights
        self.query = weights.final_query[None].contiguous()
        self.norm = weights.final_norm[None].contiguous()
        self.logits = torch.empty(config.vocab_size, device=device)

    def __call__(self, blocks: list[torch.Tensor], h: torch.Tensor) -> torch.Tensor:
        """``ref.logits_of`` of ``ref.hidden_states``' last block residual and norm over bf16
        ``blocks`` (1 to MAX_BLOCKS) and the last visit's output ``h``: fp32 ``(vocab,)``
        logits (a buffer the next call overwrites)."""
        hidden = self.weights.config.hidden_size
        if not 1 <= len(blocks) <= MAX_BLOCKS:
            raise ValueError(f"need 1 to {MAX_BLOCKS} blocks, got {len(blocks)}")
        states = [b.reshape(hidden) for b in blocks]
        states += [states[0]] * (MAX_BLOCKS - len(states))  # unread
        launch = self.launch
        compile_head_kernel(launch.op)(
            self.weights.lm_head,
            self.query,
            self.norm,
            h.reshape(hidden),
            *states,
            len(blocks),
            self.logits,
            launch.table,
            launch.workspace,
            launch.max_ctas,
        )
        return self.logits


class InklingDecoder:
    """The batch-one decode step of ``weights`` as megakernel launches only: ``ref.hidden_states``
    and ``ref.logits_of`` of one token at a device position, ``start=None``. Calls must run
    one at a time in stream order (they share the runners' state and the step buffers)."""

    def __init__(self, weights: InklingWeights, device="cuda"):
        config = weights.config
        blocks = config.block_layers
        # A pass's messages read two states (the core's second block input and its output), and
        # the final block residual reads every block: the kernels' layout.
        if config.loop_start % blocks or config.loop_layers != 2 * blocks:
            raise ValueError("the core must start a block and span exactly two blocks")
        if config.num_layers > blocks * MAX_BLOCKS:
            raise ValueError(f"the kernels' block residuals read at most {MAX_BLOCKS} blocks")
        self.weights, self.config = weights, config
        self.gates = MessageGates(
            weights.layers[config.loop_start : config.loop_end], weights.message_queries
        )
        self.visit = InklingVisitMegakernel(config, weights.layers, device=device)
        self.msg = InklingMsgMegakernel(config, self.gates, device)
        self.head = InklingHeadMegakernel(weights, device=device)
        hidden = config.hidden_size
        # [0]: the token's normalized embedding; [v + 1]: visit v's output.
        self.states = torch.zeros(
            config.num_visits + 1, hidden, device=device, dtype=torch.bfloat16
        )

    def __call__(
        self, token: torch.Tensor, pos: torch.Tensor, cache: InklingCache
    ) -> torch.Tensor:
        """One decode step of the one-element int64 ``token`` at the one-element device
        position ``pos`` (int32 avoids a cast); writes the caches there and returns fp32 logits
        ``(vocab,)`` (a buffer the next step overwrites). Syncs nothing, so ``pos`` is not
        checked: the caller keeps ``0 <= pos < cache.max_len``."""
        config, weights = self.config, self.weights
        positions = device_position(pos)
        embed = EmbedArgs(weights.embed, weights.embed_norm, token.reshape(1))
        h = self.states[0]
        blocks: list[torch.Tensor] = []
        pre_loop = None  # the prelude's output and blocks, where every core pass restarts
        for visit, layer, p in config.visits():
            gates = None
            if config.is_loop(layer):
                core = layer - config.loop_start
                if core == 0 and p == 0:
                    pre_loop = (h, blocks)
                elif core == 0:
                    # The finished pass's messages: its core's second block input and output.
                    (block,) = blocks[len(pre_loop[1]) + 1 :]
                    self.msg(block, h)
                    h, blocks = pre_loop
                if p > 0:
                    gates = self.gates.layer(core)
            out = self.states[visit + 1]
            self.visit(
                weights.layers[layer], layer, visit, h, blocks, cache, positions, out, gates,
                embed if visit == 0 else None,
            )  # fmt: skip
            if layer % config.block_layers == 0:  # h completes a block
                blocks = blocks + [h]
            h = out
        return self.head(blocks, h)
