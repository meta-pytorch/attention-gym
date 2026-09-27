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

## 2. Bugs in our own port (found in review, bench or audit)

### Other tests encoding decisions
- **Superseded by v1.30, tests kept** (verify_fixes.py runs them as `superseded`): B1 seeded-state
  wait, B5 KDA FP32 factors, B11 terminal TMA overfetch (upstream bit-21 descriptor fix, NVIDIA
  #1013/#1015), B12 checkpoint `[V,K]` descriptors, S11 V-major state. Inherited from main: B4 dO
  dtype check.

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
| Keep upstream untimed waits (S7) | `try_wait=True` / `spin=True`; `cute.arch.mbarrier_wait` changes the wait loop. | Upstream changes wait primitives. |
| `fmul2`/`ffma2` stay inline PTX | `cute.arch` versions changed 35 cubins (STACK 24→0 kda_summary, 96→144 gdn_recompute); the `fadd2` wrapper is identical and used. | New CuTeDSL; re-check SASS. |

## Known limitations and pre-existing issues (not fixed)

- **`get_compile_target()` latch:** `attn_gym/_backends/cute/target.py` caches the first detected
  target process-wide, so a process that switches to a GPU of different compute capability keeps a
  stale target in `jit_cache` keys (mocked 10.0→10.3 repro). Pre-existing; mixed-GPU processes only.
- Fails-without evidence predates the final commit order (see "Reading an entry").
