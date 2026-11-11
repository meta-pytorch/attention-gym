# Safety contract

[Write a kernel](writing-kernels.md) · [API guides](index.md#more-examples-and-details)

warpdrive manages synchronization for your declared roles and channels. You supply the math,
layouts and matching work on both sides of each handoff. **A kernel that compiles can still
race or deadlock.**

## What the layer handles

These guarantees require matching epochs and participants, enough storage for live values,
a dependency graph that can make progress, and terminating user work.

| When you use… | warpdrive handles… | You still supply… |
|---|---|---|
| Managed roles and the declared launch shape | Warp placement, register-budget redistribution, barrier initialization and dispatch | Converged entry/exit and correct per-role math |
| A managed `Ring` or `Slots` handoff | Stage/phase bookkeeping, declared arrival counts, publication and reuse after the required releases | Matching epochs, participating threads and storage lifetimes |
| `Tma` / `Umma` mechanisms | Transaction arming / elected MMA completion signals and supported cross-CTA targets | Correct atoms, descriptors, copies and declared byte counts |
| `SmemPlan` and handle-bound views | Allocation and projection onto the handle's stage | Layouts, partitions and views of the named buffer |
| `TmemPlan` and annotated users | Column allocation and coordinated teardown | Every actual TMEM user declared, and correct alias ordering |
| A drained `WorkStream` | Once-only assignment of kept indices to work units, with the same sequence for their participating roles | A valid mapping/schedule and correct work for each item |

A work unit is one CTA or one cluster, not each CTA in a cluster. Custom schedules and
mappings must meet the [scheduling protocol](scheduling.md#custom-mappings).

This contract covers CuTeDSL 4.6.2 through 4.8 on Blackwell SM100/SM103, with one CTA, a CTA
pair or a cluster of at most eight CTAs. Example wrappers have their own tensor and numerical
contracts.

## The authoring rules that matter

**Balance participants and epochs.** A `Threads` side counts every thread of its bound roles.
Predicate the data access, not the handoff:

```python
with acc.consume() as h:
    if output_in_bounds:
        ...  # store valid output; every participating thread still releases
```

A lane-dependent condition around the whole `with` changes arrivals without changing the
count. The [liveness check](#what-a-successful-check-means) leaves runtime branches
unverified; predicate data accesses rather than synchronization.

On an [exclusive-consumer ring](channels.md#exclusive-consumers), every epoch needs exactly
one owner. This is checked only for static waits and skips. A wait after more than one ring
lap of skips needs a rendezvous with that epoch's previous owner.

**Carry current state.** Return every managed endpoint, work cursor and turnstile seat from
its role. Helpers return updated state; callers reassign it. Close each handle once, in its
opening dynamic region, and stop accessing its storage. The checker catches missing returns,
local lifetime errors and discarded advancing-helper results for managed ends and cursors,
including in loops. It does not prove freshness of reconstructed snapshots, raw state or
turnstile seats, or catch every post-close access.

**Drain the work loop.** Use `while work.valid:`, read `work.item`, then call `work.advance()`
once, directly at the end of the body. Alternatively, use the
[counted form](scheduling.md#a-counted-loop) with `StaticPersistent` or `TilePersistent`.
Return the cursor. The checker recognizes loop structure; it does not compare runtime trip
counts across roles.

**Own the memory ordering.** Declared channel fences run before managed signals. The default
`smem_async` fence applies to:

- Release by thread consumers of a TMA-filled channel. On a
  [per-role consumer side](channels.md#per-role-consumer-mechanisms), only thread roles run it.
- Commit by thread producers writing SMEM that an MMA consumes.

TMA stores, raw async copies, TMEM aliases and storage shared across channels may need more
waits and proxy fences. Complete those accesses before reuse or exit. For ordinary writes
read by another CTA, supply the correct peer address and `remote_scope="cluster"`; a channel
cannot infer either.

**Launch what you declared.** Call `begin_roles` and `run_roles` in converged CTA/cluster code.
Use `launch_config(bound)` for the declared block/cluster shape and checked work grid. Keep
raw named barriers clear of IDs 1 and 2, `op.named_barrier_ids` and each `NamedBarrier.id`.
Initialize any `Barriers(counts=None)` before dispatch. Cluster exit synchronization and
automatic producer tails cover the managed protocol, not arbitrary async work.

**Keep configuration and workspaces valid.** Put static choices in the constructor and use
the [cache integration](roles.md#compile-keys). Construct a new op when declarations change;
do not mutate a collected one. Reassigning its cluster, work stream or `trace_protocol` after
collection raises. Trace/alias settings may still change; other public attributes assigned
after construction are rejected at cache lookup. In-place declaration mutations are not
checked or keyed.

Do not share scheduler workspaces between concurrent launches. Use `launch_args()` for the
current device and stream. Launch checks do not validate user indices, pointer arithmetic or
resource use.

## What a successful check means

- **`describe()`:** checks declarations and reports derived roles, counts, masks, storage and
  budgets. Body checks appear only after compilation. `jit_cache` restores saved TMA/alias/fence
  reports and warnings; the examples compile protocol-traced builds uncached.
- **Compilation:** checks local handle lifetimes, returned state and recognizable work loops.
  It checks attributable TMA operations and binds try tokens/previews to their end, epoch and
  dynamic region, including loop carries whose entry and back-edge still name that epoch.
  It rejects a wait for a stage the same role still holds
  ([overlapping handles](channels.md#overlapping-handles)).
  Exclusive/round-robin epoch ownership is checked when all acquires, waits and skips are
  static.
- **Liveness:** `verified` means balanced, acyclic straight-line managed channel ops in one
  CTA, including already-expanded compile-time loops, ring reuse, acquired-slot tails and
  named-channel rendezvous on both sides. Runtime regions, raw roles/state/barriers/signals,
  external sides, overrides, TMA/async completion, cross-CTA placement, spans, retire, skips,
  turnstiles, standalone named barriers and unknown epochs are `unverified`. Node/edge budgets,
  opt-outs and analysis failures are unverified too; no runtime inputs are sampled.
  Supported cycles, starved waits and imbalances are `error`: `liveness_check="strict"` raises
  with sites and an override, `"warn"` warns and `"off"` skips this independent pass.
- **TMA `verified`:** recognized group and byte accounting, not descriptor legality, disjoint
  writes or async safety. Inspect the reasons for any `unverified` use.
- **Alias `no hazard found`:** no conflicting annotated pair under the diagnostic's rules,
  not proof of safe aliasing. Clusters, raw accesses, round-robin channels and orderings
  through exclusive-ring epochs have [limited or no coverage](storage.md#read-the-result).
- **Fence `covered`:** no missing or unverified fence requirement was found for the accesses
  examined by the [fence check](channels.md#checking-explicit-fences). Unannotated accesses
  are not checked.
- **Passing tests:** evidence for the shapes, schedules and toolchain tested. Profiling plans
  and protocol logs prove neither liveness nor performance.

Invalid configurations generally raise `ValueError`; invalid bindings and traced protocols
raise `ProtocolError`. Read the message and any nested compiler cause. **There is no public
numbered error-code API.** E/R/U/G labels in fuzz reports name test cases and contract clauses,
not exception attributes or message prefixes.

## Hardware assumptions

The layer relies on three ordering assumptions:

- **`tcgen05` completion orders the MMA's SMEM reads before later generic writes.** PTX states
  this implicit proxy fence for `cp.async.bulk` and `wgmma`, not for `tcgen05`. A `Threads`
  producer refilling a stage that an `Umma` consumer released relies on it.
- **CTA-scope releases order reads before a remote refill.** When a thread releases another
  CTA's barrier with `remote_scope="cta"`, the layer assumes its reads precede that CTA's
  next fill ([Memory scope](clusters.md#memory-scope)).
- **Ordering is preserved without the `tcgen05` thread-sync fences.** PTX's cross-thread
  `tcgen05` pattern puts `tcgen05.fence::after_thread_sync` after a wait and
  `::before_thread_sync` before an arrive. These are also
  [compiler code-motion fences](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-special-sync-operations-fence).
  Channels emit neither: omitting them assumes both execution and compiler ordering hold.
  Emit them by hand in the role, as the KDA WY backward kernel does:

```python
from cutlass._mlir.dialects import nvvm

h = acc.wait()
nvvm.tcgen05_fence(kind=nvvm.Tcgen05FenceKind.AFTER_THREAD_SYNC)
...  # TMEM work and its required completion waits
nvvm.tcgen05_fence(kind=nvvm.Tcgen05FenceKind.BEFORE_THREAD_SYNC)
h.release()
```

## Dropping down

Raw roles keep placement, initialization and annotated TMEM teardown, but lose managed
endpoint/cursor checks. `signal=False` omits both the signal and its fence. Overrides and
`unchecked(...)` leave the corresponding reasoning to you. Suppress only the specific check
and document why. See [Dropping down](dropping-down.md) for the APIs and boundaries.

## Further reading

Exact semantics live with their APIs: [roles](roles.md), [channels](channels.md),
[storage](storage.md), [scheduling](scheduling.md), [clusters](clusters.md)
and [tracing](tracing.md). For layer changes, see the
[fuzz harness and its coverage boundaries](https://github.com/meta-pytorch/attention-gym/blob/main/test/warpdrive_fuzz/README.md):
declaration and mutation fuzzing, and the synthetic layer bugs the tests must catch.
