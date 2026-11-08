# warpdrive fuzz findings

Keep failure mechanisms that teach an authoring constraint or protect a future layer change,
with a minimized case and regression. Campaign totals and clean-run logs belong in run
artifacts, not here. See [the harness guide](README.md) for execution and coverage boundaries
and [case labels](CASES.md) for the historical identifiers. Old repros below retain the API
names of their fixing commits; use the named current regressions to exercise today's code.

## Declaration and compile-time fuzzing

### Declaration campaign (seed 0)

Each row is a minimized host repro; these are declaration validation bugs/hardening gaps, not evidence of a valid-program GPU hang. Regressions live in `test/warpdrive_fuzz/test_static.py`.

| Seed | Contract | Minimized repro | Before | Class/status | Commit |
|---|---|---|---|---|---|
| 0 | G3 / E29,E35,E36,E43 | `Tma(-1)`; `Ring(1,Threads,Tma(16))`; `Ring(1,Threads())`; `Barriers(1,counts=[1.5])` | Accepted (bad scalar counts can instead TypeError) | Validation bug; fixed; 21 regression cases failed before / passed after | e1872d6 |
| 0 | G4 / R8 | `TmemPlan().region('a',32).alias('a',of='a',offset=0,cols=16)`; alias width 0/-1; duplicate region; unknown parent | Self-cycle accepted (lookup recurses), widths accepted, unknown parent KeyError | Validation bug; fixed; 6 regressions before / after (nonpositive region cases already rejected, alias cases did not) | bd6d5f6 |
| 0 | G2 / R9 | `role(warps=-1)`; `role(warps=32)`; `role(warps=[])`; `role(warps=0,regs=25)` | Accepted; role-free kernel reports unrelated max() error | Validation bug; fixed; 16 regressions failed before / passed after | 2c1f892 |
| 0 | G2,G6 / E83 | `TileStream(scheduler_warp=-1)`; `TileStream(record=-1)`; `Raster(swizzle=True)` | Accepted | Validation bug; fixed; 8 regressions failed before / passed after | 8962cdf |
| 0 | E44–E47 | `Ring(1,override={'full_counts':1.5})` collected with Producer/Consumer; boolean counts/mask/rank | TypeError or accepted bool-as-int | Diagnostic/domain bug; fixed; 4 regressions failed before / passed after | db8baa0 |
| 0 | G7 / E68 | One role `tiles: Tiles`, no stream; declare `tiles_work` beside a dynamic/CLC stream; same Channel under two attributes | Accepted; later lookup failure/overwritten channel/aliased bindings | Validation bug; fixed; 4 regressions failed before / passed after | db37604 |

### Empty work and Int32 overflow (G6, fixed)

Raw `Raster().decode(0, (0, 1, 1))` divides by zero: the decoder expects a bound space,
not unchecked extents. `WorkStream.bind` now normalizes an empty/negative Raster space,
and `bound.grid()` checks the schedule's terminal-index bound before a wrapped launch
can run. Hand-built grids and custom mappings still need their own valid domain.
This distinction matters when extending a mapping: making decode safe is not enough if
the count or terminal index can overflow first.

Regressions in `test/warpdrive/test_warpdrive_scheduling.py`:
`test_bound_empty_space_has_no_items_and_decodes`, `test_swizzled_decode_does_not_overflow_int32`,
`test_empty_space_runs_no_items` and `test_space_overflowing_int32_is_rejected`.

### Cursor mutation campaign (seed 0)

| Seed | Contract | Minimized source | Before | Class/status | Commit |
|---|---|---|---|---|---|
| 0 | G1 / E73 | `while tiles.valid & (count < ctx.n): count += 1; tiles.advance()` | Compile accepted; compound condition can stop before terminal | Validation hardening; fixed with exact-condition check; regression failed before / passed after across all four policies | c08178b |
| 0 | G1 / R12 | `while tiles.valid: tiles.advance(); m,n,l = tiles.work; ctx.out[0] = m` | Compile accepted; reads next item as current | Validation hardening; fixed with same-iteration/descendant-block work-read check; regression failed before / passed after across all four policies | c08178b |
| 0 | U1 | Producer n+1 vs consumer n; complete acquire/commit pair gated by lane 0 | Compile accepted | Accepted (documented obligation); never launched | n/a |
| 0 | U10 | `self.step(ch)` where step acquires/commits but does not return/reassign end | Compile accepted | Accepted (documented obligation); never launched | n/a |
| 0 | U6 | TMA copy barrier rebuilt through an opaque pointer, in single/pair-local/pair-scoped channels | Compile accepted with unverified warning/status | Accepted (documented obligation); never launched | n/a |

Mutation sources are written to `$WARPDRIVE_FUZZ_MUTATIONS_DIR` (default: a temp dir) as `mutation_<sha256-prefix>.py`. Reproduce deterministically from the normal harness: `.venv/bin/python -c "from warpdrive_fuzz.mutate import compile_mutation,mutations; compile_mutation(next(m for m in mutations() if m.name=='compound_condition'),0)"` (now raises ProtocolError). `advance_before_work` uses the same command with that mutation name.

## Runtime fuzzing

Findings distinguish layer defects from generator and toolchain failures. Each retained case
explains the violated invariant and how to protect it when evolving the code.

### F-A1: the inserted scheduler warp breaks a valid `regs=None` warpgroup (R9)

A role on warp 0 used `regs=None` while warps 4–7 used `regs=160`. Inserting a
scheduler on warp 1 incorrectly treated `None` as an absent budget and chose 160,
creating a mixed-budget warpgroup and rejecting a valid declaration.

The scheduler must inherit its occupied warpgroup's choice, including `None`; only an
otherwise idle group uses the minimum declared numeric budget. Fixed in 727c3b9.
Regression:
`test/warpdrive_fuzz/test_runtime.py::test_scheduler_takes_its_warpgroups_default_budget`.
This is a declaration-time rejection bug, not a runtime synchronization failure.

### Multicast releases must cover every reader

Generator seeds 300150 and 300175 returned stale TMA words: manual per-warp releases
arrived only in the local CTA, while a multicast stage was still read by other CTAs.
This was a generator bug, not evidence against the managed release derivation. Manual
signals, overrides and unclosed handles now skip multicast in the generator.
When authoring a raw replacement, derive release targets from every CTA whose reads depend
on the published bytes, not just the producer's own pair. The cluster example tests exercise
the managed protocol; a new raw protocol needs a discriminating test of its remote releases.

### An oracle must observe the property being changed

The first reverse-schedule oracle checked coverage/agreement but not per-unit order. It now
checks the exact sequence. Similarly, replacing an UMMA consumer's release with a plain
arrival escaped the data oracle because MMA finished before the next TMA write arrived.
A correct output in that run did not validate the missing completion ordering. See the
[harness coverage boundaries](README.md#coverage-boundaries) before relying on a clean run.
Keep the checkout fixed during compilation: rebasing while children imported it produced
failures that were harness artifacts rather than properties of either revision.

### F-C1: orderings only a raw signal provides were reported as potential hazards

- **Minimized repro:** `Slots(2, Threads, Threads, buffer={"P": p, "Q": q})` with `q`
  declared `overlaps=p`; the producer writes Q, the consumer reads P and closes with
  `release(signal=False)` after one elected `mbarrier_arrive` per warp (override
  `empty_counts=1`). The diagnostic reported `potential hazard ... no channel path orders them`.
  Variants: a managed TMA commit beside the raw producer commit (reported its complete_tx
  failure instead), and a consumer that opens under `unchecked("unclosed")` and arrives itself.
- **Classification:** diagnostic bug, not a synchronization bug. The program is ordered (the
  per-warp arrival follows `__syncwarp`, the producer's acquire waits on it), and the contract
  documents raw signals as `unverified`. `_Graph._ops` dropped `signal=False` closes, so no
  path existed and no failure was recorded.
- **Fix:** `alias.py` keeps those closes as separate raw edges; a pair reachable through one (even
  beside a failed managed edge) is `unverified`, and so is a missing path on a channel stage with a
  never-closed handle (again even beside a failed managed edge). Retired stages still publish
  nothing (the close event now records a retire), and a potential hazard on any instance pair wins
  over an unverified pair found earlier in the scan, so the new `unverified` answers cannot hide
  one. Regressions:
  `test/warpdrive/test_warpdrive_alias_models.py::test_raw_release_is_unverified_not_a_hazard`,
  `::test_raw_commit_beside_a_tma_commit_is_unverified`,
  `::test_unclosed_handle_is_unverified_not_a_hazard` (each failed before, passes after), and
  `::test_an_unverified_pair_does_not_hide_a_later_hazard`.

### F-C2: register budgets the layer accepted could hang in `setmaxnreg.inc` (R9)

- **Minimized repro:** 12 warps, budgets 232/232/48 per warpgroup: 65536 registers, which the
  R9 check (`<= 65536`) accepted. Each warp starts with `default_regs = 65536 / 384` rounded
  down to 168, so the launch holds 64512, and `setmaxnreg.inc` only takes registers other
  warps released: the increase waits forever.
- **Classification:** layer bug (valid-looking declaration accepted, then a device hang).
- **Fix:** `kernel.py` rejects a total above `default_regs x padded threads` at declaration
  (E05), with the pool in the message.
  Regression: `test/warpdrive/test_warpdrive.py::test_register_budgets_fit_the_registers_the_launch_starts_with`
  (accepted before, raises after; 232/232/40 runs).

### F-C3: ptxas register allocation fails beside MMA code at 24/32-register budgets

- **Repro:** the generated MMA subgraph (budgets 64-224) next to generic roles whose
  warpgroups have 24 or 32 registers: `ptxas fatal: (C7600) Register allocation failed with
  register count of '224'`. 40 for those warpgroups compiles; without the MMA roles 24 compiles.
- **Classification:** toolchain resource limit, outside the contract (G2/G3: the layer does not
  certify CUDA resource feasibility). The generator now keeps budgets >= 40 in kernels with
  the MMA subgraph.
