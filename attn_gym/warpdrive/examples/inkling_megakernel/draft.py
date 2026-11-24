"""Host side of the DSpark drafter's megakernel launches: ``DSparkMegakernel`` drafts with Inkling
2 Mini's shipped drafter (``dspark``) through three launches that read every input from device
memory and sync nothing, so a draft step is CUDA-graph capturable and each launch compiles once for
all positions. It is an ``examples.megakernel.spec.DraftModel`` (the device speculative step's
drafter protocol):

- ``commit``: context K/V of up to ``TOKENS`` committed rows and the Markov head's context
  (``draft_ctx_kernel``);
- ``propose``: the block's backbone (``draft_backbone_kernel``), then the base logits and the
  sequential Markov head's ``num_drafts`` steps (``draft_head_kernel``).

Numerics follow ``dspark.DSparkDrafter`` (vLLM's rounding points) up to fp32 summation order:
the GEMVs accumulate on tensor cores (the Markov fold on CUDA cores) and attention merges flash
partials. Drafts only move the verify's acceptance, so they need not match the torch drafter
bit for bit (``test_warpdrive_inkling_draft_kernel`` checks each stage within rounding).
"""

import torch

from attn_gym.warpdrive.examples.inkling_megakernel.draft_backbone_kernel import (
    QKV_SPLITS,
    DraftBackboneKernel,
    compile_draft_backbone_kernel,
)
from attn_gym.warpdrive.examples.inkling_megakernel.draft_common import TOKENS, WORDS
from attn_gym.warpdrive.examples.inkling_megakernel.draft_ctx_kernel import (
    DraftCtxKernel,
    compile_draft_ctx_kernel,
)
from attn_gym.warpdrive.examples.inkling_megakernel.draft_head_kernel import (
    DraftHeadKernel,
    compile_draft_head_kernel,
)
from attn_gym.warpdrive.examples.inkling_megakernel.dspark import (
    DSparkCache,
    DSparkConfig,
    DSparkWeights,
    rope_angles,
)
from attn_gym.warpdrive.examples.megakernel.constants import BLOCK, HEAD_DIM, ROWS, TILE_K
from attn_gym.warpdrive.examples.megakernel.launch import Launch, table_of

__all__ = ["DSparkMegakernel", "MegakernelDrafter", "rope_table"]


def draft_table(op, ctas: int) -> torch.Tensor:
    """A draft launch's ``StaticPersistent`` table of ``op.lists(ctas)``."""
    return table_of(op.lists(ctas), WORDS)


def qkv_splits_for(hidden: int) -> int:
    """The backbone's q/k/v split-K for a ``hidden`` size: the most of ``QKV_SPLITS``, 2 and 1
    that cut it into whole ``TILE_K`` tiles."""
    for splits in (QKV_SPLITS, 2, 1):
        if hidden % (splits * TILE_K) == 0:
            return splits
    raise ValueError(f"the draft backbone needs a multiple of {TILE_K} hidden size, got {hidden}")


def rope_table(config: DSparkConfig, max_len: int, device) -> torch.Tensor:
    """fp32 ``(max_len, head_dim)``: ``[cos | sin]`` of each position's RoPE angles
    (``dspark.rope_angles``, as ``dspark.rope`` takes them)."""
    angles = rope_angles(config, torch.arange(max_len, device=device))
    return torch.cat([angles.cos(), angles.sin()], dim=-1).contiguous()


class DSparkMegakernel:
    """The DSpark drafter of ``weights`` as megakernel launches over a position-indexed cache
    of ``max_len`` positions, drafting ``num_drafts`` tokens (default the block's 7; ``k - 1``
    for verify blocks of ``k``: the Markov steps past them are skipped). Calls must run one at
    a time in stream order. ``trace``: compile the launches with that trace backend
    (``"globaltimer"``)."""

    def __init__(
        self,
        weights: DSparkWeights,
        max_len: int,
        num_drafts: int | None = None,
        device="cuda",
        trace: str | None = None,
    ):
        c = weights.config
        if c.block_size != TOKENS or c.head_dim != HEAD_DIM:
            raise ValueError(f"the draft launches need blocks of {TOKENS} and head dim {HEAD_DIM}")
        self.num_drafts = c.checked_drafts(num_drafts)
        max_len = -(-max_len // BLOCK) * BLOCK  # the backbone reads 64-position KV blocks
        self.weights, self.config, self.max_len = weights, c, max_len
        self.taps = len(c.target_visits)
        layers = weights.layers
        self.cache = DSparkCache.empty(c, max_len, device=device)
        self.wkv = torch.cat([torch.cat([w.wk_ctx, w.wv_ctx]) for w in layers]).contiguous()
        self.alpha = weights.fuse_alpha.contiguous()
        self.dflare = torch.stack([w.dflare_norm for w in layers]).contiguous()
        self.k_norm_ctx = torch.stack([w.k_norm_ctx for w in layers]).contiguous()
        self.rope = rope_table(c, max_len, device)
        self.f_rnn = torch.zeros(c.hidden_size, device=device, dtype=torch.bfloat16)
        ctx = DraftCtxKernel.get_or_build(
            hidden=c.hidden_size,
            layers=c.num_layers,
            kv_heads=c.num_kv_heads,
            taps=self.taps,
            rms_eps=c.rms_eps,
        )
        self.ctx = Launch(ctx, draft_table, device, trace)
        # The block's layers: [wq; wk; wv], wo, gate/up interleaved by 16 rows, w_down.
        self.wqkv = torch.cat([torch.cat([w.wq, w.wk, w.wv]) for w in layers]).contiguous()
        self.wo = torch.cat([w.wo for w in layers]).contiguous()
        self.wgu = torch.cat([
            torch.stack([w.w_gate.view(-1, 16, c.hidden_size), w.w_up.view(-1, 16, c.hidden_size)],
                        dim=1).reshape(-1, c.hidden_size)
            for w in layers
        ]).contiguous()  # fmt: skip
        self.wd = torch.cat([w.w_down for w in layers]).contiguous()
        self.input_norm = torch.stack([w.input_norm for w in layers]).contiguous()
        self.post_norm = torch.stack([w.post_norm for w in layers]).contiguous()
        self.q_norm = torch.stack([w.q_norm for w in layers]).contiguous()
        self.k_norm = torch.stack([w.k_norm for w in layers]).contiguous()
        backbone = DraftBackboneKernel.get_or_build(
            hidden=c.hidden_size,
            layers=c.num_layers,
            q_heads=c.num_q_heads,
            kv_heads=c.num_kv_heads,
            intermediate=c.intermediate_size,
            rms_eps=c.rms_eps,
            local=c.local_layers,
            window=c.sliding_window,
            qkv_splits=qkv_splits_for(c.hidden_size),
        )
        self.backbone = Launch(backbone, draft_table, device, trace)
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        vocab = weights.lm_head.shape[0]
        head = DraftHeadKernel.get_or_build(
            hidden=c.hidden_size,
            vocab=vocab,
            rank=c.markov_rank,
            drafts=self.num_drafts,
            softcap=c.logit_softcap,
            slots=-(-vocab // ROWS // sms),
        )
        self.head = Launch(head, draft_table, device, trace)
        # The mask rows' final states (row TOKENS - 1 stays zero) and the drafts.
        self.hidden = torch.zeros(TOKENS, c.hidden_size, device=device, dtype=torch.bfloat16)
        self.drafts = torch.zeros(self.num_drafts, device=device, dtype=torch.int64)
        # commit's default ``last`` for each row count, made before any capture.
        self.last_rows = torch.arange(TOKENS, device=device).view(TOKENS, 1)

    def commit(
        self, features: torch.Tensor, positions: torch.Tensor, last: torch.Tensor | None = None
    ) -> None:
        """Context K/V of bf16 ``(rows, taps, hidden)`` target ``features`` (1 to ``TOKENS``
        rows) at the consecutive int64 device ``positions`` (element 0 is read), and the Markov
        head's context from row ``last`` (a one-element int64 device index; None: the last
        row), the most recent committed position. Rows at positions outside ``[0, max_len)``
        are dropped; nothing else is checked (nothing syncs)."""
        rows = features.shape[0]
        if not 1 <= rows <= TOKENS:
            raise ValueError(f"commit takes 1 to {TOKENS} rows, got {rows}")
        if positions.numel() == 0:  # the kernel reads element 0
            raise ValueError("commit needs the rows' positions, got none")
        launch = self.ctx
        compile_draft_ctx_kernel(launch.op)(
            self.wkv,
            features.reshape(rows * self.taps, -1),
            self.alpha,
            self.dflare,
            self.weights.markov_norm,
            self.k_norm_ctx,
            self.rope,
            self.cache.k.view(-1, self.config.head_dim),
            self.cache.v.view(-1, self.config.head_dim),
            self.f_rnn,
            positions,
            self.last_rows[rows - 1] if last is None else last.reshape(1),
            launch.trace_buffer,
            launch.scratch,
            launch.state,
            launch.table,
            launch.workspace,
            launch.max_ctas,
        )

    def commit_rows(self, features: torch.Tensor, start: int):
        """Host-driven commit of bf16 ``(rows, taps, hidden)`` features at positions ``start
        ..``, ``TOKENS`` rows per launch (prompt prefill)."""
        positions = torch.arange(start, start + features.shape[0], device=features.device)
        for first in range(0, features.shape[0], TOKENS):
            self.commit(features[first : first + TOKENS], positions[first : first + TOKENS])

    def block_states(self, anchor: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """The mask rows' final normalized states of the block of the one-element int64 device
        ``anchor`` at the one-element int64 device ``pos``, in ``hidden`` (rows 0 .. 6)."""
        launch = self.backbone
        compile_draft_backbone_kernel(launch.op)(
            self.wqkv,
            self.wo,
            self.wgu,
            self.wd,
            self.cache.k.view(-1, self.config.head_dim),
            self.cache.v.view(-1, self.config.head_dim),
            self.weights.embed,
            self.weights.embed_norm,
            self.weights.mask_embed,
            self.input_norm,
            self.post_norm,
            self.q_norm,
            self.k_norm,
            self.weights.norm,
            self.rope,
            anchor.reshape(1),
            pos.reshape(1),
            self.hidden,
            launch.trace_buffer,
            launch.scratch,
            launch.state,
            launch.table,
            launch.workspace,
            launch.max_ctas,
        )
        return self.hidden

    def markov(self, anchor: torch.Tensor) -> torch.Tensor:
        """Greedy drafts (num_drafts,) int64 (a buffer the next call overwrites) from the mask
        rows' states in ``hidden`` and the one-element int64 device ``anchor``."""
        w, launch = self.weights, self.head
        compile_draft_head_kernel(launch.op)(
            w.rnn_proj,
            w.lm_head,
            w.fold_weight,
            self.hidden,
            self.f_rnn,
            w.rnn_bias,
            w.markov_w1,
            w.fold_bias,
            anchor.reshape(1),
            self.drafts,
            launch.trace_buffer,
            launch.scratch,
            launch.state,
            launch.table,
            launch.workspace,
            launch.max_ctas,
        )
        return self.drafts

    def propose(self, anchor: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Greedy drafts (num_drafts,) int64 for the positions after the one-element int64
        device ``anchor`` at the one-element int64 device ``pos`` (``pos + TOKENS <=
        max_len``, not checked), over the committed context; a buffer the next call
        overwrites."""
        self.block_states(anchor, pos)
        return self.markov(anchor)

    def draft(
        self,
        anchor: torch.Tensor,
        pos: torch.Tensor,
        features: torch.Tensor,
        positions: torch.Tensor,
        last: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One speculative step's drafting: ``commit(features, positions, last)`` (the last
        verify block's rows: rows past ``last`` were rejected, and this proposal rewrites their
        positions before reading them), then ``propose(anchor, pos)``. Every argument is a
        device tensor, so the step is CUDA-graph capturable."""
        self.commit(features, positions, last)
        return self.propose(anchor, pos)


class MegakernelDrafter:
    """``dspark.Drafter`` over ``DSparkMegakernel`` for host-driven loops
    (``dspark.speculative_generate``): host positions and tokens become device tensors."""

    def __init__(
        self, weights: DSparkWeights, max_len: int, num_drafts: int | None = None, device="cuda"
    ):
        self.kernels = DSparkMegakernel(weights, max_len, num_drafts, device=device)
        self.num_drafts = self.kernels.num_drafts
        self.device = device

    def commit(self, features: torch.Tensor, positions: torch.Tensor) -> None:
        self.kernels.commit_rows(features, int(positions[0]))

    def propose(self, anchor: int, pos: int) -> torch.Tensor:
        if pos + TOKENS > self.kernels.max_len:
            raise ValueError(f"a block at {pos} exceeds max_len {self.kernels.max_len}")
        anchor_t = torch.tensor([anchor], device=self.device)
        pos_t = torch.tensor([pos], device=self.device)
        return self.kernels.propose(anchor_t, pos_t).clone()
