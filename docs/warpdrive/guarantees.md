# Safety contract

[Write a kernel](writing-kernels.md) · [API guides](index.md#more-examples-and-details)

warpdrive manages synchronization for declared roles and channels. You supply the math and
memory accesses. **A kernel that compiles can still race or deadlock.** See
[channels](channels.md#stages-and-phases) for the vocabulary used here.

The supported target is Blackwell SM100/SM103 with CuTeDSL 4.6.2 through 4.8, using one CTA,
a CTA pair or a cluster of at most eight CTAs. Example wrappers have their own input contracts.

## What a successful check means

What the layer supplies, checks and leaves to you:

1. **Declarations and launch.** The layer assigns warps, redistributes declared register
    budgets, allocates barriers and storage, initializes barriers and dispatches roles.
    Collection rejects invalid bindings, overlapping roles, incompatible mechanisms and
    register budgets that exceed the launch's starting pool. Mechanisms are `Tma`, `Umma`,
    `Threads` and `Warps`; non-bulk `cp.async` needs raw synchronization. Use `launch_config(bound)` for
    the declared block/cluster shape and work-index bounds. You must enter `begin_roles()`
    and `run_roles()` convergently, initialize `Barriers(counts=None)` before dispatch, and
    respect CUDA resource limits. User indices, pointer arithmetic, layouts, numerical results
    and arbitrary async work at exit are not validated.

2. <a id="lifetimes"></a>**Ends and lifetimes.** Return every managed end, work cursor and
    turnstile seat. Helpers return updated state; callers reassign it. The lineage check
    rejects stale ends/cursors and an end advanced in a loop or branch that did not carry it
    in, including an in-place helper call whose result was discarded. It does not establish
    freshness of reconstructed snapshots, raw state or turnstile seats.

    Close each handle once in its opening region; a [span](channels.md#spans) may close in
    its counted loop's last iteration. The next wait in trace order adopts a preview, even
    across regions. No handle, span or preview may remain live at role exit. Tokens are bound
    to their end, slot, epoch and region; loop carries are checked at entry and back-edge.
    `token=` accepts only a `Token`, not a Boolean.
    These checks do not prove runtime preview matching or catch every access before a span's
    wait or after a handle closes. Stop using storage when you hand it back.

3. <a id="participation"></a>**Participation and reuse.** Managed channels track stages and
    phases, derive arrival counts, and publish/reuse stages through the declared mechanisms.
    `Threads` counts all bound threads; `Warps` synchronizes every participating lane before
    one elected arrival per warp. Predicate data accesses, not handoffs. Counts do not elect
    lanes or establish that the intended threads arrived.

    The local held-stage check rejects a wait at least one ring lap beyond a handle the same
    end still holds (one epoch for a `Slots` slot), using statically known advances in enclosing
    regions. It does not count later loop trips and permits TMA commits, raw signals and
    external sides that could complete earlier. Static exclusive/round-robin ownership checks
    require one owner per published epoch and compatible per-item advancement. Dynamic skips,
    raw state and unmatched sequences remain unverified. After skipping past a stage's previous
    use, an exclusive consumer may need a [rendezvous](channels.md#static-ownership-check):
    ownership alone does not make its phase wait exact.

4. <a id="liveness"></a>**Liveness.** `verified` means the modeled channel/barrier graph
    balances and completes with static trip counts and one work item per CTA. With runtime
    bounds or persistent work, `no deadlock found` names the sampled counts and assumptions,
    including any assumption that values unbalancing a channel never occur. Sampling is not a
    proof for all inputs. A constant difference between recognizable fill/take counts can
    establish an imbalance even with runtime bounds.

    Definite cycles, missing signals, established imbalances and recognized lane-divergent
    thread arrivals raise under `liveness_check="strict"` (default); `"warn"` warns and `"off"`
    skips analysis. Raw roles/state, external sides, changed count/mask overrides, dynamic
    skips, unreadable loops/branches and `unchecked("liveness")` leave it `unverified`.
    Related runtime expressions, uncertain signal timing and analysis limits can also leave
    liveness `unverified`. User work and readiness waits must terminate independently.

    `min stages for liveness` gives each eligible multi-stage ring's minimum for the checked
    runs, holding other depths fixed. It omits `Slots`, one-stage rings and rotating named
    rings; it is not a tuning recommendation.

5. <a id="named-barriers"></a>**Named barriers.** Declarations check counts and the shared ID
    budget. Recorded calls check available participating threads per CTA;
    `alias.named_barrier` also checks static IDs for collisions. Native calls are invisible:
    reading `.id` can make missing participation unverified, but cannot justify a count above
    the available threads. Raw IDs must avoid 1, 2 and all allocated IDs.

    Liveness links each user's successive calls as successive barrier epochs when recorded
    users account for the count. It does not prove that arrive-only users cannot arrive twice
    before others finish the current epoch. Supply an acknowledgement for a reusable
    arrive/sync protocol. A sync covering exactly the calling role orders its warps in the
    alias model; an arrive alone does not.

6. <a id="tma"></a>**TMA.** The layer arms declared transaction bytes. The check follows
    channel barrier pointers to recognized
    tensor/bulk copies. Wrong CTA groups, undeclared multicast and complete byte-count
    mismatches raise. Tensor copies count in the handle's region or its `elect_one()` block;
    bulk copies require that block. Unknown atoms, masks, pointers, nested/conditional copies
    or incomplete totals leave uses unverified. `verified` covers recognized uses only; some
    barriers may be opted out. `no copies` is not validation.

    You supply valid descriptors, partitions, disjoint writes and legal bulk transfer sizes.
    A separate context check inspects direct fields for MMA CTA span and unsupported
    group-TWO/multicast G2S atoms; it neither walks nested objects nor obeys `unchecked("tma")`.
    See [TMA checks](tracing.md#tma-checks) for pointer/atom spellings.

7. <a id="fences"></a>**Fences.** Managed signals emit the declared fences. Defaults order
    generic SMEM reads before a TMA refill and generic SMEM writes before an UMMA read;
    [explicit values replace defaults](channels.md#storage-and-fences). The fence check compares
    explicit overrides against annotated accesses while handles were open, including a ring
    refilling itself. TMA fills and UMMA reads imply accesses to bound SMEM.

    `covered` means no missing or uncertain requirement for the accesses examined, not all
    accesses. `payload=` accesses are unrecorded; missing annotations, raw roles/signals,
    external peers and uncertain conditional or cross-channel ordering can be `unverified`.
    A required `tmem_store` or `tmem_load` fence absent from the trace is `unverified`, not
    `missing fence`: manual `tcgen05.wait` calls are invisible. Recorded fences between access
    and signal count only when they run on that path.
    Missing fences warn, or raise with `alias_check="strict"`; `"off"` skips analysis.
    The check inserts no fences. Complete TMA stores, raw async copies and TMEM accesses before
    reuse. Ordinary writes read by another CTA need the correct peer address and
    `remote_scope="cluster"`; neither is inferred.

8. <a id="aliases"></a>**Aliases.** The alias diagnostic examines annotated conflicting pairs
    of declared SMEM overlaps and TMEM aliases, not arbitrary kernel accesses. A
    `potential hazard` has a missing modeled ordering edge; `unverified` lacks enough
    information; `no hazard found` is not proof of safe aliasing. `alias_check="warn"` reports
    findings, `"strict"` raises on hazards, and `"off"` skips analysis.

    The model covers single-CTA handoffs and cross-channel paths, unrolling dynamic loops
    twice and persistent work for two items. Ring reuse orders only its own statically disjoint
    stage spans. Ring-stage SMEM across channels, round-robin matching, cross-channel paths
    through exclusive rings and unequal traced counts can remain unverified. Ring reuse inside
    an inner work-item loop adds no modeled edge. Fan-out readers and fan-in writers are not
    ordered with each other. SMEM accesses of a role are one thread set, so there is no
    cross-warp SMEM check. Clusters are unverified.

    Annotations must cover actual accesses; raw pointers and `h.data` do not annotate later
    uses. The model assumes TMEM loads finish before the next signal and the required
    `tcgen05` thread-sync fences surround synchronization. Declare every TMEM user with
    `tmem: Tmem` for allocation and teardown. Layouts, partitions, storage lifetimes and alias
    ordering remain yours.

9. <a id="work"></a>**Work and configuration.** A correctly launched, drained built-in stream
    assigns each kept index to exactly one work unit and gives that unit's roles the same
    sequence. A unit is one CTA or cluster. Follow the [work loop](scheduling.md#the-role-loop); checks recognize
    its structure and reject item reads after advance, but do not prove useful work happened,
    advance is lexically last, or custom mappings/schedules agree, cover all items and terminate.
    Keep mapping tables stable after readiness and scheduler workspaces private to concurrent
    launches. Use `launch_args()` for the current device and stream.

    Put static choices in the constructor and follow the [compile-key contract](roles.md#compile-keys).
    Collection freezes declarations; unsupported public attribute assignments raise at cache
    lookup. In-place mutations of declarations are neither checked nor keyed.

10. <a id="opt-outs"></a>**Raw code and opt-outs.** `signal=False` closes a handle without its
    signal or fences; it does not relax the lifetime rule. `external=` leaves one side to raw
    code; `Barriers` leaves the whole protocol to you. Raw roles retain placement,
    initialization and declared TMEM teardown, but have no managed end/cursor checks.
    Raw accessors do not validate pointer bounds or ownership; restoring old `end.state`
    can silently break the protocol.

    `unchecked(check, reason=...)` accepts exactly `"tma"`, `"fence"` and `"liveness"`.
    Respectively, it excludes copies through barrier pointers obtained inside the block,
    annotated accesses from the fence check, or channel/barrier ops from liveness (making the
    kernel unverified). It records the reason, site and suppressed findings in `describe()`.
    Other checks stay active. Leave an access unannotated to exclude it from alias analysis;
    use a span, carried token or raw protocol where the managed form does not fit. None of
    these adds ordering. Trace bookkeeping is process-global: compile in isolated processes,
    not concurrent threads.

11. <a id="hardware-assumptions"></a>**Hardware assumptions.** Channels assume `tcgen05`
    completion orders MMA SMEM reads before later generic writes; PTX states this proxy
    ordering for `cp.async.bulk` and `wgmma`, not `tcgen05`. Channels also assume CTA-scope
    releases order reads before a remote refill, and omit
    `tcgen05.fence::after_thread_sync` and `::before_thread_sync`, relying on execution and
    compiler ordering. These are
    [compiler code-motion fences](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-special-sync-operations-fence)
    too. Emit them explicitly around the handoff when needed:

    ```python
    from cutlass._mlir.dialects import nvvm

    h = acc.wait()
    nvvm.tcgen05_fence(kind=nvvm.Tcgen05FenceKind.AFTER_THREAD_SYNC)
    ...  # TMEM work and its completion waits
    nvvm.tcgen05_fence(kind=nvvm.Tcgen05FenceKind.BEFORE_THREAD_SYNC)
    h.release()
    ```

12. **Reports and tests.** `describe()` checks declarations before compilation; body reports
    appear after tracing or [cache restoration](cache.md#cached-builds-were-never-traced).
    Kernels built inside compile functions restore no reports on a hit. Tests cover only the
    inputs and toolchain exercised.
    Protocol logs record trace sites; profiling plans and timings prove neither safety nor
    liveness. Inspect every unverified reason before relying on a result.

Invalid configurations generally raise `ValueError`; invalid bindings and traced protocols
raise `ProtocolError`. Read any nested compiler cause. There is no numbered error-code API:
E/R/U/G labels in the [fuzz harness](https://github.com/meta-pytorch/attention-gym/blob/main/test/warpdrive_fuzz/README.md)
name cases, not exception attributes.
