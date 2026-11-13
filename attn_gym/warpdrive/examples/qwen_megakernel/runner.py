"""Qwen3 decode step as one persistent warpdrive kernel (a "megakernel").

The code spans the modules of this package (see its ``__init__``). Kernels compile from ``core``'s
imports plus their own module, so host-side edits recompile nothing.

One kernel class serves the dense Qwen3 family: ``QwenDecodeMegakernel(model=QwenConfig)`` derives
every tiling from the shape and ``check_model`` rejects what they do not cover (0.6B, 1.7B,
8B, 14B and 32B fit; 4B's hidden size does not). The host lowers the step into a table of
instructions, one row per work item (0.6B counts)::

    for layer in 28 layers:
        QKV     128 items   32 rows of [Wq; Wk; Wv] x RMSNorm(h)            -> qkv
        ATTN    8 x S       kv head x KV split: q/k norm, RoPE, flash decode  -> attention partials
        OPROJ   32 x 4      32 rows x 512-column slice of Wo (split K)        -> o partials
        GATEUP  192         16 gate + 16 up rows x RMSNorm(h + o)             -> SiLU(g) * u
        DOWN    32 x 4      32 rows x 768-column slice of Wd (split K)        -> down partials
    LMHEAD  594         8 x 32 rows of the LM head x final RMSNorm            -> logits

An attention row serves two q heads of one kv head (one when the GQA group is odd), so larger
models run several rows per kv head and split; o_proj splits a quarter of the q heads each.

By default the host also assigns rows to CTAs (``list_schedule``, over a latency model) and
lays the table out ``[slot][cta]`` for ``StaticPersistent`` (``assign_static``); the atomic
schedule claims rows in dependency order instead. Each CTA's rows stay in dependency order, so
with every CTA resident (one per SM, guaranteed by the cooperative launch) the earliest
unfinished row can always run.

Cross-CTA dependencies use no counters. Each activation is published as one 64-bit word
``(fp32 value, launch epoch)`` with a relaxed store; a consumer spins on relaxed loads until
every word it needs carries the current epoch (NCCL's LL protocol). The word is single-copy
atomic, so a matching tag means the value is this launch's. One write and one read
propagation per dependency, no fences, no separate flag (streamed inputs add per-token and
per-chunk flags for their bf16 buffers). The last CTA to finish advances the epoch for the next
launch.

Roles of one CTA (9 warps; 10 with the atomic schedule's scheduler warp)::

    warp 8    load     TMA weight / KV-cache tiles into the ``w`` ring (16 KB stages). It never
                       waits on activations, so it streams the next rows' weights while the
                       compute warps wait for inputs.
    warps 0-7 compute  poll the row's tagged inputs (norms etc. into SMEM) -> consume the row's
                       stages -> publish the outputs

Numerics follow HF's bf16 boundaries: activations hold bf16-rounded values; GEMV accumulation
is fp32, rounded once per output (split-K partials are summed in fp32 and rounded by their
consumer, down's by its last split); attention scores round like the eager matmul; the
softmax merges flash-decoding partials in fp32.

Every per-step intermediate has its own per-layer buffer, so a launch has no write-after-read
hazards. The new K/V row at ``pos`` is written by the one attention row whose TMA-loaded block
holds it (masked), after those loads completed.

The step is latency-bound: about 140 dependent phases, each paying input propagation plus its
row's latency (docs/warpdrive/examples.md). ``QwenMegakernel(trace="globaltimer")``
records every CTA's work items and ``trace_region``s with the table row as payload
(``trace`` decodes them into phase spans and a Perfetto trace);
``trace="iket"`` emits the same ranges for ``run-iket``.
"""

import math

import torch

from attn_gym.warpdrive.examples.qwen_megakernel.core import (
    ALL_OPS,
    ATTN,
    ATTN_SCALE,
    BLOCK,
    COMPUTE_WARPS,
    DOWN,
    GATEUP,
    HEAD_DIM,
    LMHEAD,
    NORM,
    OP_NAMES,
    OPROJ,
    QKV,
    QWEN3_0_6B,
    REDUCE,
    ROUTER,
    ROWS,
    TRACE_EVENTS,
    TRACE_EVENTS_PER_ROW,
    QwenDecodeMegakernel,
    check_model,
    compile_megakernel,
    packed_mlp_shapes,
)
from attn_gym.warpdrive.examples.qwen_megakernel.schedule import (
    CONTEXT,
    DEPENDENCY_LATENCY,
    ROW_COST_MODELS,
    RowCostModel,
    RowSpan,
    assign_static,
    build_table,
    instruction_rows,
    list_schedule,
    row_cost,
    row_groups,
    weight_stages,
)
from attn_gym.warpdrive.examples.qwen_megakernel.tc import (
    QwenDecodeMegakernelTc,
    QwenDecodeMegakernelTcStream,
    check_tc_model,
)

__all__ = [
    "ALL_OPS",
    "ATTN",
    "ATTN_SCALE",
    "BLOCK",
    "COMPUTE_WARPS",
    "CONTEXT",
    "DEPENDENCY_LATENCY",
    "DOWN",
    "GATEUP",
    "HEAD_DIM",
    "LMHEAD",
    "NORM",
    "OPROJ",
    "OP_NAMES",
    "QKV",
    "QWEN3_0_6B",
    "REDUCE",
    "ROUTER",
    "ROWS",
    "ROW_COST_MODELS",
    "TRACE_EVENTS",
    "TRACE_EVENTS_PER_ROW",
    "QwenDecodeMegakernel",
    "QwenDecodeMegakernelTc",
    "QwenDecodeMegakernelTcStream",
    "QwenMegakernel",
    "RowCostModel",
    "RowSpan",
    "assign_static",
    "build_table",
    "check_model",
    "check_tc_model",
    "compile_megakernel",
    "instruction_rows",
    "kernel_class",
    "list_schedule",
    "packed_mlp_shapes",
    "rope_table",
    "row_cost",
    "row_groups",
    "weight_stages",
]


def rope_table(max_len: int, device, theta: float = 1e6) -> torch.Tensor:
    """(max_len, 2, HEAD_DIM) bf16 cos and sin, computed as HF does."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, HEAD_DIM, 2, device=device).float() / HEAD_DIM))
    freqs = torch.arange(max_len, device=device, dtype=torch.float32)[:, None] * inv_freq
    angles = torch.cat((freqs, freqs), dim=-1)
    return torch.stack((angles.cos(), angles.sin()), dim=1).to(torch.bfloat16).contiguous()


def kernel_class(engine: str, stream_inputs: bool = False) -> type[QwenDecodeMegakernel]:
    """Return the kernel class selected by ``QwenMegakernel``'s ``engine`` and ``stream_inputs``."""
    kernels = {"cuda_core": QwenDecodeMegakernel, "tcgen05": QwenDecodeMegakernelTc}
    if engine not in kernels:
        raise ValueError(f"engine must be one of {sorted(kernels)}, got {engine!r}")
    if stream_inputs:
        if engine != "tcgen05":
            raise ValueError("stream_inputs needs the tcgen05 engine")
        return QwenDecodeMegakernelTcStream
    return kernels[engine]


class QwenMegakernel:
    """Packed weights, scratch and the compiled kernel for one cache capacity ``max_len``."""

    def __init__(
        self,
        weights,
        max_len: int,
        ops: tuple[int, ...] = ALL_OPS,
        trace: str | None = None,
        waits: bool = False,
        engine: str = "cuda_core",
        stream_inputs: bool = False,
        **config,
    ):
        """``ops``: keep only these instruction kinds (tests). ``trace``: compile with ranges for
        the work items and ``trace_region``s, recorded by ``"globaltimer"``
        (``trace.trace_session`` decodes a launch) or ``"iket"`` (``run-iket``);
        ``waits`` adds the layer's wait ranges. ``engine``: the GEMV engine, ``"cuda_core"`` or
        ``"tcgen05"`` (``QwenDecodeMegakernelTc``). ``stream_inputs`` (tcgen05): stream GEMV
        inputs from global memory through an activation ring (``QwenDecodeMegakernelTcStream``).
        QKV and LM-head inputs, plus dense MLP inputs, are shared across CTAs; other inputs
        use per-CTA scratch. See that class for the dataflow. ``config``: ``QwenDecodeMegakernel``
        constructor arguments; the model shape comes from ``weights.config``."""
        kernel = kernel_class(engine, stream_inputs)
        if max_len % BLOCK:
            raise ValueError(f"max_len must be a multiple of {BLOCK}, got {max_len}")
        device = weights.embed.device
        self.max_len = max_len
        self.op = op = kernel(model=weights.config, **config)
        self.workspace, self.max_ctas = op.work.launch_args(device)
        self.table = build_table(op, self.max_ctas, ops).to(device)
        rows_per_cta = -(-self.table.shape[0] // self.max_ctas)
        events = max(TRACE_EVENTS, TRACE_EVENTS_PER_ROW * rows_per_cta)
        op.enable_trace(trace, waits=waits, events_per_role=events)
        self.wqkv = torch.cat((weights.q_proj, weights.k_proj, weights.v_proj), dim=1).reshape(
            -1, op.hidden
        )
        self.wo = weights.o_proj.reshape(-1, op.q_heads * HEAD_DIM).contiguous()
        # Gate and up rows interleaved, so a 32-row tile holds 16 (gate, up) pairs; MoE weights
        # carry an expert axis after the layer axis.
        self.wgu = torch.stack((weights.gate, weights.up), dim=-2).reshape(-1, op.hidden)
        self.wd = weights.down.reshape(-1, weights.down.shape[-1]).contiguous()
        if op.moe:
            self.router = weights.router.reshape(-1, op.hidden).contiguous()
        else:  # a stub tile for the unused router atom
            shape = packed_mlp_shapes(op)[2]
            self.router = torch.zeros(shape, device=device, dtype=torch.bfloat16)
        self.weights = weights
        self.rope = rope_table(max_len, device, weights.config.rope_theta)
        self.scratch = torch.zeros(op.scratch_layout.size, device=device, dtype=torch.int64)
        self.batch = op.batch
        # (batch, vocab); ``logits`` is token 0's row at batch one, the whole tensor otherwise.
        self.all_logits = torch.zeros(op.batch, op.vocab, device=device)
        self.logits = self.all_logits[0] if op.batch == 1 else self.all_logits
        self.state = torch.zeros(2, device=device, dtype=torch.int32)
        # (batch, 2) [token, pos]; ``step_info`` is token 0's ``(2,)`` row at batch one, as before.
        self.all_steps = torch.zeros(op.batch, 2, device=device, dtype=torch.int32)
        self.step_info = self.all_steps[0] if op.batch == 1 else self.all_steps
        # Per-CTA GEMV inputs of the streaming engine (one stub row otherwise).
        rows = self.max_ctas if op.streams_inputs else 1
        self.xscratch = torch.zeros(rows, op.x_slot, device=device, dtype=torch.bfloat16)
        self.xshared = torch.zeros(op.shared_size, device=device, dtype=torch.bfloat16)
        # Placeholder; trace_session swaps in the real globaltimer buffer for a traced launch.
        self.trace_buffer = torch.zeros(1, device=device, dtype=torch.int64)
        self.fn = compile_megakernel(self.op)

    def tagged(self, name: str) -> torch.Tensor:
        """The int64 words of a tagged buffer, shaped as ``ScratchLayout.shapes``."""
        layout = self.op.scratch_layout
        shape = layout.shapes[name]
        offset = layout.offsets[name]
        words = self.scratch[offset : offset + math.prod(shape)].view(shape)
        if name == "h_mid":
            words = words[:, : self.op.layers]
        return words[0] if self.batch == 1 else words

    def values(self, name: str) -> torch.Tensor:
        """A tagged buffer's fp32 values, ignoring the tags."""
        return self.tagged(name).view(torch.int32)[..., 0::2].view(torch.float32)

    def routed_experts(self) -> torch.Tensor:
        """MoE: the (layers, batch, experts_per_token) experts the last launch routed to, by rank:
        the top-k of its router logits, ties to the lower index as in the kernel."""
        logits = self.values("router_logits")  # (layers, experts), batched (batch, layers, ...)
        order = logits.sort(dim=-1, descending=True, stable=True).indices[..., : self.op.top_k]
        return order[:, None] if self.batch == 1 else order.transpose(0, 1)

    def set_values(self, name: str, values: torch.Tensor):
        """Publish ``values`` to a tagged buffer for the next launch; tests feed single
        instructions this way."""
        words = self.tagged(name).view(torch.int32)
        words[..., 0::2] = values.float().view(torch.int32)
        words[..., 1::2] = self.state[0] + 1

    def launch(self, k_cache: torch.Tensor, v_cache: torch.Tensor):
        """One decode step for the tokens and positions in ``all_steps`` (``(batch, 2)``). Both are
        read on the device, so the caller keeps ``0 < pos < max_len`` and the tokens in the
        vocabulary, and runs one launch at a time: the scratch and epoch state belong to this
        runner. Caches: ``(layers, kv_heads, max_len, head_dim)`` at batch one, else with a
        leading batch axis."""
        shape = (self.op.layers, self.op.kv_heads, self.max_len, HEAD_DIM)
        if self.batch > 1:
            shape = (self.batch, *shape)
        for cache in (k_cache, v_cache):
            if cache.shape != shape or cache.dtype != torch.bfloat16 or not cache.is_contiguous():
                raise ValueError(f"caches must be contiguous bf16 {shape}, got {cache.shape}")
            if cache.device != self.logits.device or cache.data_ptr() % 16:
                raise ValueError("caches must be 16-byte aligned on the kernel's device (TMA)")
        w = self.weights
        self.fn(
            self.wqkv,
            self.wo,
            self.wgu,
            self.wd,
            w.embed,
            w.lm_head,
            self.router,
            k_cache.view(-1, HEAD_DIM),
            v_cache.view(-1, HEAD_DIM),
            self.scratch,
            self.all_logits,
            self.state,
            self.all_steps,
            self.trace_buffer,
            w.input_ln,
            w.post_ln,
            w.q_norm,
            w.k_norm,
            w.final_norm,
            self.rope,
            self.xscratch,
            self.xshared,
            self.table,
            self.workspace,
            self.max_ctas,
        )

    def __call__(self, weights, token, pos: int, k_cache, v_cache) -> torch.Tensor:
        """``decode_step_ref``'s signature: writes K/V at ``pos``, returns fp32 logits.
        ``weights`` must be the runner's own; they were packed at construction."""
        if weights is not self.weights:
            raise ValueError("weights must be the ones this QwenMegakernel was built from")
        if not 0 < pos < self.max_len:
            raise ValueError(f"need 0 < pos < max_len = {self.max_len}, got {pos}")
        if self.batch != 1:
            raise ValueError("a batched runner decodes with decode(tokens, positions, ...)")
        self.step_info[0] = token
        self.step_info[1] = pos
        self.launch(k_cache, v_cache)
        return self.logits.clone()

    def decode(self, tokens, positions, k_cache, v_cache) -> torch.Tensor:
        """One step of ``batch`` sequences: sequence b decodes ``tokens[b]`` at ``positions[b]``
        (each in ``(0, max_len)``) and writes its K/V there; returns ``(batch, vocab)`` fp32
        logits."""
        positions = [int(pos) for pos in positions]
        if len(positions) != self.batch or len(tokens) != self.batch:
            raise ValueError(f"need {self.batch} tokens and positions")
        if not all(0 < pos < self.max_len for pos in positions):
            raise ValueError(f"need 0 < pos < max_len = {self.max_len}, got {positions}")
        # Tokens may be device tensors (a previous step's argmax): copy, never read back.
        self.all_steps[:, 0].copy_(torch.as_tensor(tokens).reshape(-1), non_blocking=True)
        self.all_steps[:, 1].copy_(torch.tensor(positions, dtype=torch.int32), non_blocking=True)
        self.launch(k_cache, v_cache)
        return self.all_logits.clone()
