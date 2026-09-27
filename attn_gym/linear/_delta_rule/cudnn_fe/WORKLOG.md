# Worklog: what we found porting cudnn-frontend v1.30.0, and why we changed it

This is the narrative companion to [MAINTENANCE.md](MAINTENANCE.md). MAINTENANCE.md holds the
upgrade procedure and a generated index of the authoritative `fixes.toml` ledger; this file
records, per ledger ID, **what was wrong, how we found it, the evidence,
the fix, and what to watch when replaying it**, followed by decisions, known limitations, lessons
and the v1.30 verification baseline. It does not repeat the ledger table.

- Ledger source: [`tools/cudnn_fe/fixes.toml`](../../../../tools/cudnn_fe/fixes.toml)
  (ID, kind, commit subjects, guarding pytest node ids, upstream patch/repro, upstream status).
- Automated check: `python tools/cudnn_fe/verify_fixes.py --tree . --upstream-python <env>` runs
  every guarding test and every upstream repro (see [Verification baseline](#verification-baseline-v130)).
- Draft upstream patches: [`tools/cudnn_fe/upstream/`](../../../../tools/cudnn_fe/upstream/README.md)
  (**DRAFT, not filed**).

Vendored tag **v1.30.0** (`42286a6f`): a 49-file closure (GDN + KDA scalar linear-attention kernels,
`common/`, `tile_dsl/`) replacing the 085d50b copy vendored by #368 (KDA) / #427 (GDN). Stack:
#606 vendor → #607 restyle → #608 integration → #609 cleanup → #610/#611 tooling.

**Reading an entry.** Entries are keyed on ledger IDs; old catalog IDs (C01–C13, E1–E12) are given
where a change descends from an earlier AG fix. **Where** names files (relative to this package
unless prefixed) and stack **commit subjects**; hashes change on every rebase, so find commits with
`git log --format='%h %s' origin/main..<tip> | grep -F '<subject>'`. **Found by** says how the
problem surfaced. Test aliases used below:
GT/GB/GL/GP = `test/gdn/cudnn/test_gdn_cudnn_{training,backward,layouts,paged}.py`;
KT/KN/KV/KF/KP/KS = `test/kda/cudnn/test_{kda_cudnn_training,delta_rule_numerics,kda_cudnn_v130,`
`kda_cudnn_forward,kda_cudnn_paged,kda_cudnn_native_summary}.py`; CM/LC/PO =
`test/test_cudnn_fe_{common,launch_contract,paged_order}.py`.
"Fails without" evidence comes from the bug-reversal (mutation) audit on the pre-reorder branch
and has not been re-run on the final stack; re-run it on the next upgrade (MAINTENANCE step 6).

Evidence environment: GB200 (SM100), torch 2.15.0.dev20260926+cu132, CuTeDSL 4.8.0, TVM-FFI
0.1.14.post1. Timings are µs, **lower is better**, CUDA-graph replay, median of 3 rounds.

## 0. Layer A: verbatim vendor (not a modification)

`vendor.py` output for v1.30.0: closure, relocated `cudnn.*` imports, license texts, per-file notice,
plus the mandatory prune cut of the chain prologue's `gdp_bprop_v64` import under `compact_qdo`
(R11). Subjects: "Vendor the cudnn-frontend v1.30 GDN and KDA kernels verbatim", "Keep the vendored
cuDNN license texts verbatim under pre-commit". Replay: prove the tool against this drop first
(`vendor.py --rev v1.30.0 --verify <A commit>`), then vendor the new tag.

## 1. Bugs in upstream v1.30 (upstream draft exists or is possible)

### B7 — GDN dBeta divided by beta (C07) · numerics
- **Problem:** bprop recovers an already beta-scaled intermediate by `rowsum / (beta + 1e-10)`:
  0% of the gradient at beta = 0, 50% at 1e-10, FP16 underflow for larger small betas. The API
  accepts any post-activation beta.
- **Found by:** originally #475; on v1.30 it resurfaced as 9 small-beta test failures when the
  existing AG suite first ran against the new kernels.
- **Fix:** keep beta-free inverse factors (T = I − T_b L) and stage Z separately from beta·Z, so
  dBeta is computed directly; includes the coupled GEMM issue order and scratch lifetimes.
  `kernel/gdn_bprop_f16.py`; "Compute the GDN bprop dBeta without dividing by beta".
- **Evidence:** fails without = YES (mutation back to the divide: all 8 exact cases fail, dBeta
  0 / 0.0099 / 0.5 / 0.9901 instead of 1). SASS: gdn_bprop +296 instructions, STACK 16→24, REG 128.
  Fwd+bwd +0.4…+3.6% over four workloads (the cost #604 accepted). KDA bprop has no beta divide
  (KN::test_kda_cudnn_backward_preserves_small_beta_gradient passes).
- **Tests:** GB::test_gdn_cudnn_backward_preserves_small_beta_gradient,
  GB::test_gdn_cudnn_backward_mixed_small_beta_matches_reference.
- **Upstream:** draft `01-gdn-beta-free-dbeta.patch`; stock v1.30 returns dBeta = 0 instead of 1.
- **Replay:** the largest hunk. Overlaps B2 in the bprop barrier-init block: the two new barriers
  must sit inside the thread-0 guard; keep the init fence and CTA sync outside it. Upstream `build_cfg`
  rejects fused l2norm at d_v = 128 after this change; AG never passes `inv_q`.

### B6 — KDA delta residual rounded before subtraction (C06) · numerics
- **Problem:** `pack(beta) * (V − pack(state@K))` loses small residuals next to large
  contractions: 4098 rounds to 4096 in BF16/FP16, so −2 becomes 0. Exact 64-token fixture: output
  2.0 vs 1.5, dBeta 0 vs −256, later dQ 128 vs 96.
- **Found by:** originally #465 (exact cancellation fixture); re-checked on v1.30 with the same
  fixture across every forward plan (uncut, d_v split, prep, chain).
- **Fix:** helper `beta_residual_f16x2` (unpack V, subtract FP32 state·K, apply FP32 beta, pack once
  for MMA, keep the FP32 residual for dBeta) in kda prefill / prep_prefill / summary / recompute /
  bprop; prep keeps residual-only staging because its factors already contain beta. Subjects: "Add
  an FP32 KDA delta-residual staging helper", "Keep the KDA forward delta residual in FP32 before the
  MMA pack", "Keep the KDA backward delta residual in FP32 through subtraction and beta scaling".
- **Evidence:** fails without = YES (forward hunks reverted 16/16 fail, backward 2/2). SASS +16…56
  instructions/kernel, REG/STACK unchanged. Chain T32768 H48 fwd 1360→1378 µs (+1.3%), bwd ≈+1% at
  T2048/packed, noise elsewhere.
- **Tests:** KV::test_v130_forward_plans_keep_delta_residual_in_fp32,
  KN::test_kda_cudnn_delta_residual_keeps_fp32_precision.
- **Upstream:** draft `02-kda-fp32-delta-residual.patch`.
- **Replay:** touches five kernels in steady/seeded/first stages; grep every KDA kernel, including
  new ones, for the `pack(state@K)` subtraction.

### B8 — Empty `cu_seqlens` intervals occupy the unsplit work table (C08) · bugfix
- **Problem:** serving pads `cu_seqlens` with repeated boundaries; each empty interval consumed sort
  slots, descriptors and persistent tiles. Naively dropping empties would leave empty-sequence
  state gradients uninitialized or skip clearing a fresh paged slot.
- **Found by:** benchmark (#603): 71 padded empties inflated GDN forward 29.9 → 96.3 µs (32.3 µs
  after compaction).
- **Fix:** deterministic ballot/prefix compaction of the unsplit table keeping original sequence IDs;
  THD builders skip empty descriptors; empty stateful items keep pass-through writes (final = seed,
  d_initial = d_final) and fresh-empty paged slots are still cleared (F3). `common/split_k.py`,
  `common/thd.py`; "Route unpaged GDN and KDA and native CP summaries to the v1.30 kernels", KDA test
  in "Close regression-test gaps: KDA empty-state compaction, pinned split cuts, exact dtype names,
  grouped value heads".
- **Evidence:** fails without = YES. Compaction disabled: GDN work counts `[8,4,2134,2134]` vs
  `[8,4,4,4]`; KDA `[4,4,8,8]` vs 4; a clone→zeros mutation breaks all 65,536 empty-state cotangents.
- **Tests:** GT::test_gdn_cudnn_padding_is_bitwise_and_emits_no_empty_work,
  KV::test_cudnn_compaction_keeps_empty_state_cotangents.
- **Upstream:** draft `03-compact-empty-unsplit-work.patch`, narrower (no-state uncut forward only,
  since upstream has no per-sequence has-initial-state mask); stock fails 8/20 work-count checks.
- **Replay:** most entangled with paging. Reapply compaction first, then F3's "Preserve fresh-empty
  paged routes during work compaction", then run PO + GP + KP. Watch the >4096-sequence
  (ORDER_CAPACITY) uncompacted fallback.

### B10 — Split walk emits zero-chunk items (C10, E2) · bugfix
- **Problem:** the no-cut branch emitted a whole-sequence item unconditionally. The old vendor notice
  tied empty items to invalid TMEM lifecycle transitions; no v1.30 hang was reproduced, the verified
  effect is wasted work items.
- **Found by:** carried from the old vendor catalog; confirmed on v1.30 by counting work items.
- **Fix:** `num_chunks_b > 0` guard. `common/split_k.py`; "Omit zero-chunk split work items".
- **Evidence:** fails without = YES: interior/trailing empties 8 items vs 4; all-empty 6 vs 0.
- **Tests:** CM::test_split_table_omits_zero_chunk_sequences.
- **Upstream:** draft `04-omit-zero-chunk-split-work.patch` adds an opt-in `skip_empty`, so the
  stock repro stays "bug" even against the patched tree unless it opts in.
- **Replay:** if upstream adopts an opt-in flag, AG must still opt in on every no-state split path.

### B13 — Scalar-gate split scan ignores the head stride (C13, E3) · bugfix
- **Problem:** the scalar gate load used bare `h` instead of `h * stride[1]`; a non-unit-head-stride
  view read the wrong head and corrupted decay sums and split decisions.
- **Found by:** code audit of the address expression. Public GDN requires contiguous heads, so this
  is raw-helper robustness, not a reproduced public failure.
- **Fix:** Int64-widened `h * stride[1]`; symbolic head stride in the scalar-gate fake signature.
  `common/split_k.py`; "Respect scalar-gate head strides in split scans".
- **Evidence:** fails without = YES: 256/516 scan entries wrong, max abs error 4.6166.
- **Tests:** CM::test_scalar_split_scan_respects_head_stride.
- **Upstream:** draft `05-scalar-scan-head-stride.patch`.
- **Replay:** keep the dynamic-stride fake ABI; `mark_layout_dynamic(leading_dim=None)` still infers
  stride 1 from a compact first call, and the test then only exercises the host rejection.

### B2 — All-thread mbarrier init in GDN kernels (C02) · hardening
- **Problem:** upstream initializes every barrier from every thread; one sync object should have one
  initializer.
- **Found by:** hardened during #444's deadlock hunt (the proven cause there was C01); re-applied
  on v1.30 by inspection.
- **Fix:** init under `if tidx == 0`, fence and CTA sync outside, in GDN prefill, bprop, recompute,
  bprop_summary, summary. "Initialize the GDN prefill mbarriers from one thread", "Initialize the GDN
  backward mbarriers from one thread", "Initialize GDN summary mbarriers from one thread and tidy
  vendored kernel names".
- **Evidence:** fails without = **NO** (1,000 fwd / 500 bwd iterations pass with all-thread init).
  SASS ±8, gdn_recompute ckpt64 STACK 120→72, perf neutral. CUTracer `random_delay` 9/9 bitwise on
  every GDN/KDA filter at the final SASS.
- **Tests:** GT::test_public_gdn_cudnn_repeated_stateful_launches_cross_wave_boundary,
  GT::test_public_gdn_cudnn_repeated_backward_crosses_wave_boundary
  (`ATTN_GYM_RUN_STRESS_TESTS=1`).
- **Upstream:** draft `07-single-thread-gdn-mbarrier-init.patch`, hardening only.

### S12 — Launch contracts unchecked; literal scheduler arrival counts (A8, E6, E10) · validation
- **Problem:** invalid metadata (TMA shape/alignment, device, work-table shapes, `n_tiles != B*HO`,
  checkpoint interval/capacity, role/barrier-ID conflicts) reached the compiler or kernel; literal
  15/11 arrival counts break silently if a role map changes.
- **Found by:** E-series diff audit of host code; the capacity case by constructing short-checkpoint
  inputs that compiled.
- **Fix:** host checks before compile; `mb_sched_done` arrivals = CTA warps − TMA publisher (KDA
  bprop_summary: 11, warps 8–11 idle). Five "Validate … launch contracts …" / "Preserve strided gate
  parameter layouts in host validation" commits.
- **Evidence:** capacity guard YES (3 short-checkpoint cases reach the compiler without it);
  arrival count NO (default literals are correct; SASS 8/8 identical).
- **Tests:** the nine LC tests in `fixes.toml`.
- **Upstream:** draft `08-derive-scheduler-arrival-counts.patch` (count derivation only, hardening).

### B14 — Split-table replay passes a tensor into a compiled `None` slot (E4) · bugfix, AG-only
- **Problem:** replaying a table compiled without a scheduler passed `sched_ctr` anyway → TVM-FFI
  TypeError at argument 14.
- **Found by:** E-series diff audit.
- **Fix:** `r.has_sched` guard. "Preserve the scheduler ABI when replaying split tables".
- **Tests:** CM::test_split_table_replay_preserves_absent_scheduler_abi. Fails without = YES.
- **Upstream:** excluded. Upstream `build_split_table` always converts `scheduler_counter` through
  dlpack, so `None` fails at build time (`repro_09_replay_abi_scope.py`); proposing it would be a
  feature request. Reapply unconditionally while AG keeps the optional-scheduler recipe.

## 2. Bugs in our own port (found in review, bench or audit)

### B15 — Launch caches keyed on shape-unaware state
- **Found by:** PR #604 review. A shape-unaware cache reused the first shape's launch.
- **Fix:** builders cache compiled launches by static configuration only; scratch is rebuilt from
  the batch shape per call (`plan.py`). "Route unpaged GDN and KDA …".
- **Evidence:** fails without = YES: split GDN 4,909/33,280 values wrong, incl. NaNs, on the second
  shape. **Tests:** GT::test_gdn_cudnn_changing_batch_shape_in_one_process_matches_default,
  KV::test_v130_changing_shapes_reuses_only_static_configuration (asserts `cache_info()`).

### B16 — 4-byte-aligned gate/beta views rejected
- **Found by:** PR #604 review: legal 4-byte-offset views failed TVM-FFI's 16-byte base ABI.
- **Fix:** `aligned()` copies such gate/beta into a 16-byte base (`plan.py`).
- **Tests:** GL::test_gdn_cudnn_accepts_four_byte_aligned_gate_and_beta. Fails without = YES.

### R3 — Warmup forward state fakes shared one symbolic extent
- A shared symbol forced equal extents on the distinct `state_in`/`state_out` fakes. Found while
  writing the fake signatures; "Give the warmup forward state signatures independent extents". No
  dedicated test.

### Other tests encoding decisions
- **B3** — KT::test_cudnn_backward_past_sort_capacity_runs_empty_work_items: native KDA stateful
  backward over ORDER_CAPACITY+3 sequences (5,464/8,198 zero-chunk items). The fix (an empty item
  must not consume a dstate handshake phase, C03) is upstream in v1.30; reverting it historically hung.
- **R8** — test/test_delta_rule_stages.py::test_simulated_context_parallel_matches_unsharded_op:
  sharded KDA dbeta bounded by an operand-pack budget (see Decisions).
- **Superseded by v1.30, tests kept** (verify_fixes.py runs them as `superseded`): B1 seeded-state
  wait, B5 KDA FP32 factors, B11 terminal TMA overfetch (upstream bit-21 descriptor fix, NVIDIA
  #1013/#1015), B12 checkpoint `[V,K]` descriptors, S11 V-major state. Inherited from main: B4 dO
  dtype check.

## 3. Features and integration (AG-specific)

- **F1/F2/F4/F6/F9 — drivers on v1.30.** `gdn.py` (from #604) and `kda.py` pick uncut, d_v split,
  KDA prep, exact piece chain or approximate warmup split per call; staged stateful backward; AG plan
  floors (d_v split below one wave, chain floors 8192/2048). KDA vs main: T8192 H48 fwd 434→344,
  fwd+bwd 3207→1670; T32768 H48 fwd+bwd 12748→6297; T2048 H8 fwd 114.5→68.4 (bwd 490.5→496.4,
  +1.2%, inside threshold). GDN bitwise identical to #604. Plan thresholds are GB200-tuned; re-bench
  on new drops.
- **F7/F8 — native CP summaries** `[B;A]` and reverse maps `[C;R]` for arbitrary bounds via a
  device-selected uncut work table (graph-safe Triton selector). 0.43–0.55× of main (T16384 B4 H16
  fwd 536→289 µs, rev 1439→647). KS::test_native_selected_bounds.
- **F5 — grouped q/k GDN backward** through AG's deterministic `group_sum` (upstream `head_reduce`
  unused, later deleted by R13).
- **F3 — paged recurrent state** in the v1.30 prefill: null route → zero output by select, no pool
  write; fresh → zero seed + write; empty fresh → clear; resumed/null empty → untouched. Upstream
  has seed/final indices but no has-initial-state mask; the hardest port. 6/8 KDA paged tests fail
  on the pre-commit tree; dense-prefill SASS unchanged; paged fwd vs main 0.88–0.90×. Reapply after
  B8; if upstream adds a has-initial-state mask, evaluate mapping AG routes onto it.
- **R2 — `compile_tvm_ffi(opt_level=...)`**, hosts keep O2 (see Lessons).

## 4. Restyle, pruning and infra (gated by SASS, not behavior)

S1–S4, S8/S6/S3, S15 (restyle), S5/S13 (frozen cfgs, `@jit_cache` over fake TVM-FFI signatures,
compile target in the key), R5 (driver simplification, −95 lines), S10 (ruff), R16 (renames), R10,
R11, R13, R14 (pruning) and R12 (notices) are listed with their gates in the ledger. Replay notes:

- One commit per kernel family; SASS gate each; then the bench gate (R1).
- Upstream-only knobs and helpers come back verbatim with every drop; re-prune after the behavior
  rows. `python -m tools.cudnn_fe.audit --root <candidate> --strict` lists what is back.
- Run S10 (ruff) last so upstream diffs stay readable during replay.
- Append to, never replace, per-file notices (R12).


## Decisions

| Decision | Rationale / evidence | Revisit when |
|---|---|---|
| Shape-dependent KDA plans + pinned-plan tests | Auto plans keep v1.30 prep (fwd) and exact chain (bwd): T8192 H48 fwd 443.7→341.4 µs, fwd+bwd 2612.7→1836.8; T32768 H48 fwd+bwd 10362→6250. Prep/chain change arithmetic order, so bitwise tests pin the uncut plan and auto-plan companions use rel-L2 < 1e-2 (GDN policy from #604). | Planner heuristics change or new GPU. |
| Opt level 2 | Upstream and #604 compile at O2; O3 was mixed (+1.3% / −4.0%). An earlier claim that #604 used O3 was wrong. | New CuTeDSL release. |
| Keep upstream untimed waits (S7) | `try_wait=True` / `spin=True`; `cute.arch.mbarrier_wait` changes the wait loop. | Upstream changes wait primitives. |
| `fmul2`/`ffma2` stay inline PTX | `cute.arch` versions changed 35 cubins (STACK 24→0 kda_summary, 96→144 gdn_recompute); the `fadd2` wrapper is identical and used. | New CuTeDSL; re-check SASS. |
| CP dbeta pack-budget criterion (R8) | The partial-chunk KDA-cuDNN CP case exceeded the old dbeta tolerance (2.9×) with both new and legacy summaries, i.e. rounding of the unsharded realization, not a bug. New bound: magnitude-weighted BF16 operand-pack budget; fail-closed (no-dstate penultimate chunk and a 1.01× terminal chunk both fail). | CP numerics change. |
| Drop the replay-ABI upstream draft (B14) | Upstream has no optional-scheduler ABI; it would be a feature request. | Upstream makes the scheduler optional. |

## Known limitations and pre-existing issues (not fixed)

- **FP16 tiny beta:** with all beta ≤ 1e-6, FP16 beta-scaled operands underflow (subnormal range);
  inherent to FP16, not the dBeta formula. BF16 is fine.
- **FP16 KDA CP:** R8's six FP16 KDA-cuDNN CP cases are xfailed (`cuDNN FP16 gate overflow`).
  KDA FP16 with `gate_scale=5` gives all-NaN output on main and v1.30 alike (FP16 range).
- **`get_compile_target()` latch:** `attn_gym/_backends/cute/target.py` caches the first detected
  target process-wide, so a process that switches to a GPU of different compute capability keeps a
  stale target in `jit_cache` keys (mocked 10.0→10.3 repro). Pre-existing; mixed-GPU processes only.
- **Shapes:** the cuDNN GDN/KDA adapters require `q.shape == [1, T, H, 128]` (V = 128); summaries
  derive V, K ∈ {64, 128}.
- Single-token KDA dgate leaves a 6e-10 residual vs an exact-zero reference.
- Fails-without evidence predates the final commit order (see "Reading an entry").

## Lessons

2. **Fake signatures are an ABI (S5/S13, B13).** Upstream compiled from live tensors marked
   `mark_layout_dynamic(leading_dim=rank-1)`: int32 shapes, int64 outer strides, divisibility 1.
   int64 shapes changed register allocation (REG 52→46); promising aligned strides grew a
   scalar-gate kernel from 200 to 512 instructions. A stride that must vary needs an explicit
   symbolic stride; `leading_dim=None` still infers 1 from a compact first call.
3. **Match upstream's opt level (R2).** The compiler default is not upstream's `--opt-level 2`;
   without an explicit level SASS differs and perf comparisons are meaningless.
6. **Repro before patching upstream.** Two drafted fixes (B2, S12) reproduce nothing on stock v1.30,
   and one (B14) targets an ABI upstream lacks; the repros and `verify_fixes.py` classification keep
   hardening drafts from being filed as bug fixes.
