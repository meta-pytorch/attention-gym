# Safety contract

[Write a kernel](writing-kernels.md) · [API guides](index.md#more-examples-and-details)

warpdrive derives synchronization from your roles and channels. You supply the algorithm,
layouts and matching work on both sides of each handoff. **A kernel that compiles can still
race or deadlock.**

This contract covers CuTeDSL 4.6.2 through 4.8 on Blackwell SM100/SM103, with a single CTA, a
CTA pair or a cluster of at most eight CTAs. Example wrappers have their own tensor and numerical contracts.

## What the layer handles

These guarantees assume matching channel epochs and participants, enough storage for live
values, a dependency graph that can make progress, and terminating user work.

| When you use… | warpdrive handles… | You still supply… |
|---|---|---|
| Managed roles and the declared launch shape | Warp placement, register-budget redistribution, barrier initialization and dispatch | Converged entry/exit and correct per-role math |
| A managed `Ring` or `Slots` handoff | Stage/phase bookkeeping, declared arrival counts, publication and reuse after the required releases | Matching epochs, participating threads and storage lifetimes |
| `Tma` / `Umma` mechanisms | Transaction arming / elected MMA completion signals and supported cross-CTA targets | Correct atoms, descriptors, copies and declared byte counts |
| `SmemPlan` and handle-bound views | Allocation and projection onto the handle's stage | Layouts, partitions and views of the named buffer |
| `TmemPlan` and annotated users | Column allocation and coordinated teardown | Every actual TMEM user declared, and correct alias ordering |
| A drained `WorkStream` | Once-only assignment of kept indices to work units, with the same sequence for their participating roles | A valid mapping/schedule and correct work for each item |

A work unit is one CTA or one cluster, not each CTA of a cluster independently. Custom
schedules and mappings must meet the [scheduling protocol](scheduling.md#custom-mappings).

## The authoring rules that matter

**Keep participants and epochs balanced.** A `Threads` side counts every thread of its bound
roles. Predicate the data access, not the handoff:

```python
with acc.consume() as h:
    if output_in_bounds:
        ...  # store valid output; every participating thread still releases
```

Putting the whole `with` inside a lane-dependent condition changes the arrivals without
changing the count. Likewise, producing n+1 stages while consuming n can compile. Counts
alone do not establish liveness: two roles can each wait for a value the other hasn't produced.
On a ring with [exclusive consumers](channels.md#exclusive-consumers), every epoch needs
exactly one owner (checked only when the waits and skips are static), and a wait that follows
more than a ring lap of skips needs a rendezvous with the epoch's previous owner.

**Carry current state.** Return every managed endpoint, work cursor and turnstile seat from a
role. Helpers return updated state and callers reassign it. Close each handle once, in its
opening dynamic region, and do not access its storage after closing. The checker detects
missing returns, local lifetime errors and discarded advancing-helper results for managed
channel ends/work cursors, including inside loops; it does not prove freshness of reconstructed
snapshots, manually supplied raw state or turnstile seats, or all post-close accesses.

**Drain the work loop.** Use `while work.valid:`, read `work.item` before advancing, and call
`work.advance()` once directly at the end of the body. Return the cursor. The layer checks
the traced loop structure, not matching runtime trip counts inside different roles.

**Own the memory ordering.** A channel's fences run before its managed signal: the declared ones,
which default to `smem_async` on the release of a TMA-filled channel with thread consumers.
Buffer binding does not add fences. TMA stores, raw async copies, TMEM aliases and storage
shared across channels may need additional waits and proxy fences. Complete that work before
reusing its storage or exiting. For ordinary writes read by another CTA, use the correct peer
address and `remote_scope="cluster"`; a channel cannot infer either for you.

**Launch what you declared.** Call `begin_roles` and `run_roles` in converged CTA/cluster code.
Use `launch_config(bound)` for the declared block/cluster shape and checked work grid. Keep
raw named barriers clear of IDs 1 and 2, `op.named_barrier_ids` and each `NamedBarrier.id`. Initialize any
`Barriers(counts=None)` yourself before dispatch. Cluster exit synchronization and automatic
producer tails cover the managed protocol, not arbitrary user async operations.

**Keep configuration and workspaces valid.** Construct a new op when static declarations
change; do not mutate a collected one (reassigning its cluster, a work stream or
`trace_protocol` after collection raises; other declarations are not checked). Include all
static algorithm choices in its compile
key, using the [cache integration](roles.md#compile-keys). Do not share a scheduler
workspace between concurrent launches; use `launch_args()` for the current device and stream.
The launch checks do not validate arbitrary indices, pointer arithmetic or resource use in
user code.

## What a successful check means

- **`describe()`:** declaration checks and derived roles, counts, masks, storage and budgets.
  Before a body has been traced, it cannot report that body's checks.
  A build loaded through `jit_cache` restores its saved TMA/alias reports and warnings;
  protocol-traced builds compile uncached.
- **Compilation:** also checks local handle lifetimes, required returned state, recognizable
  work-loop structure, attributable TMA operations, and that try tokens and previews are used
  by their own end, epoch and dynamic region (or a loop carry whose entry and back-edge
  still name that epoch), and the epoch ownership of an exclusive or round-robin channel
  when every acquire, wait and skip on it is static. It does not compare arbitrary role
  bodies to prove balanced epochs or a live dependency graph.
- **TMA `verified`:** recognized group and byte accounting. Not proof of descriptor legality,
  disjoint writes or async safety. Opaque uses may remain `unverified`; inspect the reasons.
- **Alias `no hazard found`:** no conflicting annotated pair found under the diagnostic's
  rules. Not proof of safe aliasing. Clusters, raw accesses, round-robin channels and
  orderings through an exclusive ring's epochs have
  [limited or no ordering coverage](storage.md#read-the-result).
- **Passing tests:** evidence for the executed shapes, schedules and toolchain, not all
  accepted programs. Neither profiling plans nor protocol logs prove liveness or performance.

Invalid configurations generally raise `ValueError`; invalid bindings and traced protocols
raise `ProtocolError`. Read the message and any nested compiler cause. **There is no public
numbered error-code API.** The E/R/U/G labels in fuzz reports identify test cases
and contract clauses, not exception attributes or message prefixes.

## Dropping down

Raw roles still get placement, initialization and annotated TMEM teardown, but not managed
endpoint/cursor checks. `signal=False` omits both the signal and its fence. Overrides and
`unchecked(...)` leave the corresponding reasoning to you; suppress only the specific check
and document the reason. See [Dropping down](dropping-down.md) for the APIs and boundaries.

## Further reading

Exact semantics live with their APIs: [roles](roles.md), [channels](channels.md),
[storage](storage.md), [scheduling](scheduling.md), [clusters](clusters.md)
and [tracing](tracing.md). For layer changes, see the
[fuzz harness and its coverage boundaries](https://github.com/meta-pytorch/attention-gym/blob/main/test/warpdrive_fuzz/README.md) and
[findings that protect future changes](https://github.com/meta-pytorch/attention-gym/blob/main/test/warpdrive_fuzz/FINDINGS.md).

<details>
<summary>Older section links</summary>

<a id="the-supported-program"></a>
- [The supported program](#the-authoring-rules-that-matter)
<a id="runtime-guarantees"></a>
- [Runtime guarantees](#what-the-layer-handles)
<a id="round-robin-and-named-channels-r22-r23"></a>
- [Round-robin and named channels (R22, R23)](channels.md#round-robin-instances)
<a id="constructor-selected-role-configurations"></a>
- [Constructor-selected role configurations](roles.md#choose-placement-in-the-constructor)
<a id="alias-diagnostic-a-bug-finder"></a>
- [Alias diagnostic: a bug finder](storage.md#check-declared-aliases)
<a id="cluster-placement-rules-r11"></a>
- [Cluster placement rules (R11)](clusters.md#placement-rules)
<a id="work-assignment-r12"></a>
- [Work assignment (R12)](scheduling.md#choose-a-schedule)
<a id="custom-schedules-r19"></a>
- [Custom schedules (R19)](scheduling.md#custom-index-schedules)
<a id="rejection-catalogue"></a>
- [Rejection catalogue](https://github.com/meta-pytorch/attention-gym/blob/main/test/warpdrive_fuzz/CASES.md#required-rejections)
<a id="roles-and-injection--kernelpy"></a>
- [Roles and injection — `kernel.py`](roles.md)
<a id="channels-raw-barriers-pair-scope-and-tmem--channelspy-clusterpy-tmempy"></a>
- [Channels, raw barriers, pair scope and TMEM — `channels.py`, `cluster.py`, `tmem.py`](channels.md)
<a id="bodies-and-calls--channelspy-protocolpy-kernelpy-schedulingpy"></a>
- [Bodies and calls — `channels.py`, `protocol.py`, `kernel.py`, `scheduling.py`](roles.md#return-the-updated-state)
<a id="tma-and-scheduling--clusterpy-tmapy-schedulingpy-kernelpy"></a>
- [TMA and scheduling — `cluster.py`, `tma.py`, `scheduling.py`, `kernel.py`](scheduling.md)
<a id="domain-hardening-added-by-fuzzing"></a>
- [Domain hardening added by fuzzing](https://github.com/meta-pytorch/attention-gym/blob/main/test/warpdrive_fuzz/CASES.md#required-rejections)
<a id="what-remains-your-responsibility"></a>
- [What remains your responsibility](#the-authoring-rules-that-matter)
<a id="known-gaps"></a>
- [Known gaps](https://github.com/meta-pytorch/attention-gym/blob/main/test/warpdrive_fuzz/README.md#coverage-boundaries)
<a id="fuzzing-and-evidence"></a>
- [Fuzzing and evidence](https://github.com/meta-pytorch/attention-gym/blob/main/test/warpdrive_fuzz/README.md)

</details>
