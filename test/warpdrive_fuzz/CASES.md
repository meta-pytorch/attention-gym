# Fuzz case labels

These stable labels identify cases in `static.py`, `mutate.py` and historical findings.
They are not emitted exception codes. Kernel authors should read the
[authoring guide](../../docs/warpdrive/writing-kernels.md) instead.
Keep a label's identity when extending its case; describe current behavior in the linked
topic guide, not in a second numbered contract.

## Required rejections

The rows identify each negative case and expected exception class. Compiler wrappers may
nest the cause; an unrelated compiler failure is not a successful rejection check.
Configuration domains reject booleans as integer quantities except the older Python-int
Slots accessor checks (E60/E64). E102 is a runtime launch rejection, not a Python exception
inside a compiled entry point.

| ID | Rejected input | Error / owner |
|---|---|---|
| E01 | cta other than all/leader/rank0 | ValueError |
| E02 | Empty outer warp list; empty/non-unit-step/non-range normalized ranges | ValueError |
| E03 | Overlapping role warp ownership | ValueError |
| E04 | Mixed regs (including None) within an owned warpgroup | ValueError |
| E05 | Accounted registers including padding above default budget x padded threads (R9) | ValueError |
| E06 | More than one WorkStream | ProtocolError |
| E07 | Unresolvable string annotation | ProtocolError |
| E08 | Layer declaration/implementation class instead of a marker annotation | ProtocolError |
| E09 | Annotation neither a marker, bound marker, Python type, nor absent | ProtocolError |
| E10 | Raw role takes a managed end, WorkCursor or Turn | ProtocolError |
| E11 | Barriers parameter has no matching declaration name | ProtocolError |
| E12 | More than one parameter would receive context | ProtocolError |
| E13 | Annotated context type is not a frozen dataclass | ProtocolError |
| E14 | Context annotations lack a most-derived type compatible with the others | ProtocolError |
| E15 | Actual context is not an instance of the annotated type | ProtocolError |
| E16 | Actual context has neither attributes nor dataclass status | ProtocolError |
| E17 | Actual context is a mutable dataclass, even if unannotated | ProtocolError |
| E18 | Producer/Consumer parameter has no matching channel name | ProtocolError |
| E19 | Instance-bound end names a Ring | ProtocolError |
| E20 | Instance index >=Slots.n | ProtocolError |
| E21 | Producer method disagrees with declared producer_role | ProtocolError |
| E22 | Consumer method absent from declared consumer_roles | ProtocolError |
| E23 | Turn parameter has no matching turnstile | ProtocolError |
| E24 | Channel slot has no bound producer or consumer contributions | ProtocolError |
| E25 | Tma/Umma side belongs to several role instances | ProtocolError |
| E26 | Tma/Umma side belongs to a multi-warp role | ProtocolError |
| E27 | Turnstile has <2 parties or parties are not instances of one method (disabled too) | ProtocolError |
| E28 | A role takes Tmem but no tmem_plan exists | ProtocolError |
| E29 | Channel stage count not a positive Python int (bool excluded) | ValueError |
| E30 | Fence not None, tmem_store, smem_async or an ordered tuple of distinct kinds | ValueError |
| E31 | TMA producer has a commit fence | ValueError |
| E32 | remote_scope not None/cta/cluster | ValueError |
| E33 | Unknown scope-override key | ValueError |
| E34 | Slots.n >31 | ValueError |
| E35 | Barriers.n not a positive Python int | ValueError |
| E36 | Raw counts not a positive Python int or n positive integer entries | ValueError |
| E37 | UMMA signaler not leader-only in a cluster with pairs, or missing the leader of a pair the other side runs in (without pairs: a UMMA producer missing a consumer CTA) | ProtocolError |
| E38 | Pair turnstile seats have different placements | ProtocolError |
| E39 | Cluster has TMEM users but none runs in every CTA | ProtocolError |
| E40 | User pair-scoped channel has a layer-owned payload | ProtocolError |
| E41 | TMA consumer CTA whose stage no copy fills (a leader or rank0 producer feeding other CTAs without `fills`) | ProtocolError |
| E42 | Other cross-CTA TMA shapes: a copy lands in a CTA with no waiting consumer (with pairs: whose leader has none), a multicast copy lands in a CTA without a producer and without `fills`, or barriers would complete uneven byte counts | ProtocolError |
| E43 | Tma used as a consumer, including on a local channel | ValueError (constructor) |
| E44 | tx/mask override not a nonnegative Python int | ProtocolError |
| E45 | tx override on a channel without derived TMA bytes | ProtocolError |
| E46 | Count-override rank mapping does not name exactly the integer cluster ranks | ProtocolError |
| E47 | Override counts have wrong length/type or nonpositive entries | ProtocolError |
| E48 | TMEM cols/slots not positive Python ints | ValueError |
| E49 | Alias offset negative or offset+cols exceeds parent width | ValueError |
| E50 | TMEM col lookup names an unknown region | ProtocolError |
| E51 | TMEM slot not a Python int in the region's slot range | ProtocolError |
| E52 | Rounded TMEM allocation exceeds 512 columns | ValueError |
| E53 | h.data on a channel without payload or buffer | ProtocolError |
| E54 | retire on an automatically tailed pair channel | ProtocolError |
| E55 | Wrong-side handle close (producer release, consumer commit/retire) | ProtocolError |
| E56 | A handle closes twice, including manual close inside a with scope | ProtocolError |
| E57 | Close outside its opening MLIR block without cross_region opt-out | ProtocolError |
| E58 | Unclosed handle without unclosed opt-out at open | ProtocolError |
| E59 | Slot argument on a Ring acquire/wait | ProtocolError |
| E60 | Non-Python-int Slots acquire/wait slot | ProtocolError |
| E61 | Slots acquire/wait slot outside [0,n) | ProtocolError |
| E62 | Instance-bound acquire/wait uses another slot | ProtocolError |
| E63 | .index on a Slots end | ProtocolError |
| E64 | Non-Python-int Slots phase argument (range/binding are not checked here) | ProtocolError |
| E65 | Unknown fence reaches device-side fence selection | ValueError |
| E66 | Unknown barrier_ptr name after begin_roles | ProtocolError |
| E67 | begin_roles problem names differ from declared streams | ProtocolError |
| E68 | WorkCursor annotation names no declared stream, including when none exists | ProtocolError |
| E69 | Persistent schedule has a nonraw role without WorkCursor | ProtocolError |
| E70 | Explicit broadcast scheduler warp collides with a role | ValueError |
| E71 | Cursor has no traced advance | ProtocolError |
| E72 | Cursor has more than one traced advance site | ProtocolError |
| E73 | Advance is not directly in the body of the while whose condition reads valid | ProtocolError |
| E74 | Managed role does not return its final work cursor (IKET on or off) | ProtocolError |
| E75 | Managed role does not return a producer/consumer endpoint or turnstile seat | ProtocolError |
| E76 | Unknown unchecked check (allowed: unclosed/cross_region/tma/alias) | ValueError |
| E77 | unchecked reason not a nonblank string | ValueError |
| E78 | Direct context TiledMma has known CTA span inconsistent with the cluster's MMA pair size | ProtocolError |
| E79 | Direct context group-TWO G2S CopyAtom in a kernel without pairs; multicast G2S CopyAtom in a single-CTA kernel | ProtocolError |
| E80 | Recognized TMA copy through a tracked barrier has a known wrong required group | ProtocolError |
| E81 | Complete attributable stage's known copy-byte sum differs from effective expected per-CTA bytes | ProtocolError |
| E82 | Raster.along not m/n | ValueError |
| E83 | Raster.swizzle not a positive Python int (bool excluded) | ValueError |
| E84 | Raster.reverse contains an axis outside m/n/l | ValueError |
| E85 | Raster space does not have three entries | ValueError |
| E86 | WorkMap.words not a positive Python int | ValueError |
| E87 | Atomic or recorded stream bound without a workspace | ValueError |
| E88 | Recorded stream calls launch_args before roles are bound | ValueError |
| E89 | Static/atomic grid requested without max_ctas | ValueError |
| E90 | Producer/Consumer subscript other than Instance | TypeError |
| E91 | Role-marked function lacks the JIT wrapper | TypeError |
| E92 | Warp input cannot normalize, or any warp lies outside [0,32); kernel has no roles | ValueError / kernel.py |
| E93 | regs not None or an integer multiple of 8 in [24,256] | ValueError / kernel.py |
| E94 | Mechanism other than the Threads/Umma class or a Tma instance | ValueError / channels.py |
| E95 | Tma.tx_bytes not a positive Python int | ValueError / channels.py |
| E96 | Duplicate TMEM region/alias name (including a self-alias); alias cols not positive int; offset not int | ValueError / tmem.py |
| E97 | Alias names an unknown parent | ProtocolError / tmem.py |
| E98 | Two attributes reference the same channel/turnstile/barrier/stream object, or a generated scheduler name overwrites an attribute | ProtocolError / kernel.py |
| E99 | scheduler_warp not None or int in [0,32); record not a nonnegative int; broadcast stages not positive int | ValueError / scheduling.py |
| E100 | Work-loop yielded condition is not exactly the cursor's valid SSA value | ProtocolError / scheduling.py |
| E101 | work.item read after advance in the same iteration's block or descendant block (this also rejects lookahead reads of the next item; read it on the next iteration instead) | ProtocolError / scheduling.py |
| E102 | Space whose `schedule.index_limit` exceeds Int32 (R18), launched through `launch_config(bound)`/`bound.grid()` | CUDA launch error (`CUDA_LAUNCH_INVALID_CONFIG`; zero-wide grid, raised at the call) / scheduling.py |
| E103 | Managed role returns an unsupported value, state from another role, or duplicate endpoint/cursor state | ProtocolError |
| E104 | Role signature has variadic positional or keyword parameters | ProtocolError |
| E105 | WorkStream `mapping`/`schedule`/`ready` is not a `WorkMap`/`IndexSchedule`/`Readiness` (or None) subclass instance | TypeError |
| E106 | Local schedule without `next_index`, or broadcast schedule without `claim_next` | TypeError |
| E107 | `single_item()` under a persistent or broadcast schedule, or outside a role (before `begin_roles`) | ProtocolError |
| E108 | Unsupported adapter configuration: `Fa4Mapping` of a scheduler that is not index-addressable, non-positive-int head dims/element size, non-bool `lpt` flags, or a space of the wrong rank; `CutlassRaster` along not m/n or swizzle not a positive int | ValueError / adapters.py |
| E109 | `decode` returns a non-constant `keep` from a mapping without `skips`, or a word count other than `words` | ProtocolError (trace time) |
| E110 | SmemPlan buffer with an empty/duplicate name, a non-power-of-two alignment, a non-int stage axis, or an overlap that needs a stricter alignment than its root | ValueError / storage.py |
| E111 | SmemPlan overlap naming an unknown buffer or a buffer of another plan; a channel buffer that is not in the kernel's `smem_plan` | ProtocolError / storage.py, kernel.py |
| E112 | Channel with both `payload=` and `buffer=`, or a `buffer=` that is not a `SmemBuffer` or a non-empty `{str: SmemBuffer}` dict | ValueError / channels.py |
| E113 | `begin_roles(layouts=...)` not naming exactly the plan's buffers; a stage axis outside the layout's rank; an overlap larger than its root (static layouts); a bound buffer or `StageView` whose static stage mode differs from the channel's stage count, or is dynamic | ProtocolError / kernel.py, storage.py |
| E114 | `end.views(...)` naming no buffer of the channel or passing a non-StageView; `h.view`/`h.access` naming an unknown buffer, or omitting the name on a multi-buffer channel; `h.data` on a channel without payload or buffer (E53); `self.buffer(name)` before allocation | ProtocolError / channels.py, kernel.py |
| E115 | Annotation without read/write, an unknown agent, or a TMEM agent other than threads/umma; TMEM `lanes`/`cols` that are not a range or callable, or a range leaving the region; `warps` that are not ints (at annotation); warps outside the role or a callable footprint leaving the region (when the diagnostic runs) | ValueError / channels.py, tmem.py |
| E116 | `alias_check` other than warn/strict/off; `"strict"` with a potential hazard | ValueError; ProtocolError / kernel.py, alias.py |
| E117 | `producer_sharing`/`consumer_sharing` that is not a `Sharing`; `Sharing.ROUND_ROBIN` on `Slots`; `sync` other than mbarrier/named; a named channel with a non-Threads side or an `override`; `turnstile=` without a named channel with a round-robin side | ValueError / channels.py |
| E118 | A round-robin side with fewer than two instances; round-robin instances of different roles or widths; `k` round-robin instances on an mbarrier ring whose stage count is not a multiple of `k` | ProtocolError / kernel.py |
| E119 | A named, round-robin or half-external channel that crosses CTAs (mixed placement or multicast) | ProtocolError / kernel.py |
| E120 | Named channels needing more barrier IDs than 3-15 provide | ProtocolError / kernel.py |
| E121 | Named round-robin instances sharing stage barriers without an enabled `turnstile` over exactly them; an acquire/wait on that side outside its turn | ProtocolError / kernel.py, channels.py (trace) |
| E122 | `alias.named_barrier(id, ...)` using a layer or named-channel ID | ProtocolError / kernel.py (trace) |
| E123 | `commit(wait=True)` on an mbarrier channel or with `signal=False`; mbarrier pointer accessors on a named channel | ProtocolError / channels.py |
| E124 | `Cluster` with a non-positive-int extent, non-bool `pair`, odd `m` with pairs, or more than 8 CTAs; `Tma.multicast` naming anything but "m"/"n"; `Tma.fills` other than None/"pair"/"cluster" | ValueError / cluster.py, channels.py |
| E125 | `Tma(fills="pair")` without MMA pairs or with a producer not exactly on the pair leaders; `fills="cluster"` with a producer not exactly on rank 0 | ProtocolError / cluster.py |
| E126 | Recognized multicast TMA copy into a channel whose `Tma` declares no multicast, or whose mask (from `Cluster.multicast`/`Cluster.fill`) has an undeclared mode; a `Cluster.fill` mask into a channel without `fills` | ProtocolError (trace time) / tma.py |
| E127 | `token=` that is not this end's own `try_wait`/`try_acquire` result, names another slot, is stale (an acquire/wait or `end.state` assignment on the end since the probe), was probed in another dynamic region without a loop carry, or is a loop-carried token whose entering or back-edge value is stale or that a branch merged; a raw Boolean outside `unchecked("token")`; a probe or `token=` on a named channel | ProtocolError (trace time) / channels.py |
| E128 | A preview written, committed, released or retired; used after its owning wait's handle closed or in another dynamic region; left with no later wait in its role; `peek()` on a named channel | ProtocolError (trace time) / channels.py |
| E129 | `producer_sharing=Sharing.EXCLUSIVE`; exclusive consumers on `Slots`, with `sync="named"`, with round-robin producers, or without `consumer_roles` | ValueError / channels.py |
| E130 | Exclusive consumers with fewer than two consumer instances, unequal contributions (unless `override` sets `empty_counts`), more than one producer instance, a declared consumer role without a Consumer end, or crossing CTAs | ProtocolError / kernel.py |
| E131 | `skip()` on a join or round-robin end, with a count that is neither a positive Python int nor a dynamic `Int32`, or past an unconsumed preview; a token used across a skip; static exclusive or round-robin ownership with an epoch filled or waited by no or several instances, a consumer past the producers' epochs, or (persistent) a round-robin instance not taking its share of each item | ProtocolError (trace time) / channels.py, kernel.py |
| E132 | `external` other than producer/consumer, without an `override` of that side's counts, on a named channel, with a non-join side, with that side's role names or fences, or with TMA multicast/fills; a role bound to an external side; managed roles of a half-external channel in different CTAs | ValueError / channels.py; ProtocolError / kernel.py |

## Runtime labels

These name the property a runtime oracle checks. Its assumptions come from the
[safety contract](../../docs/warpdrive/guarantees.md); they are not
established merely by accepting the source.

| Label | Topic |
|---|---|
| R1 | [Role placement and initialization](../../docs/warpdrive/roles.md) |
| R2–R5 | [Channel stages, phases, counts and signals](../../docs/warpdrive/channels.md) |
| R6 | [Payload storage](../../docs/warpdrive/channels.md#storage-and-fences) |
| R7 | [Cross-CTA tails and exit](../../docs/warpdrive/clusters.md#work-and-exit) |
| R8 | [TMEM allocation and teardown](../../docs/warpdrive/channels.md#tmem) |
| R9 | [Warpgroup register budgets](../../docs/warpdrive/roles.md) |
| R10 | [Turnstiles](../../docs/warpdrive/channels.md#turnstiles) |
| R11 | [Cluster placement and release targets](../../docs/warpdrive/clusters.md) |
| R12–R14 | [Assignment, raster order and guarded decoding](../../docs/warpdrive/scheduling.md) |
| R15 | [Conditional progress, not proof of arbitrary program liveness](../../docs/warpdrive/guarantees.md#what-the-layer-handles) |
| R16 | [Inspection and tracing](../../docs/warpdrive/tracing.md) |
| R17–R18 | [Empty spaces and checked index domain](../../docs/warpdrive/scheduling.md#the-grid-and-empty-work) |
| R19 | [Custom scheduling protocol](../../docs/warpdrive/scheduling.md#custom-index-schedules) |
| R20 | [Raw-role work items](../../docs/warpdrive/scheduling.md#one-item-in-a-raw-role) |
| R21 | [Buffer allocation and stage projection](../../docs/warpdrive/storage.md) |
| R22–R23 | [Round-robin and named channels](../../docs/warpdrive/channels.md#round-robin-instances) |
| R24 | [Exclusive consumers](../../docs/warpdrive/channels.md#exclusive-consumers) |

## Caller-obligation labels

Accepted negative controls use these labels; acceptance does not make them safe to launch.

| Label | Obligation / authoritative topic |
|---|---|
| U1–U2 | [Epoch/participant balance and algorithm liveness](../../docs/warpdrive/guarantees.md#the-authoring-rules-that-matter) |
| U3–U4 | [Correct data/layouts, access lifetimes and async completion](../../docs/warpdrive/storage.md) |
| U5 | [Peer memory scope and addressing](../../docs/warpdrive/clusters.md#memory-scope) |
| U6 | [Independently validate unverified TMA](../../docs/warpdrive/guarantees.md#what-a-successful-check-means) |
| U7–U8 | [Raw signals, overrides and scoped opt-outs](../../docs/warpdrive/dropping-down.md) |
| U9 | [Raw-role limits](../../docs/warpdrive/dropping-down.md#escape-hatches-by-feature) |
| U10 | [Carry state](../../docs/warpdrive/roles.md#return-the-updated-state); [mapping, workspace and index obligations](../../docs/warpdrive/scheduling.md) |
| U11 | [Frozen params are shallow](../../docs/warpdrive/roles.md#arguments) |
| U12 | [Declaration snapshots and compile keys](../../docs/warpdrive/roles.md#compile-keys) |
| U13 | [Post-close access and raw accessor limits](../../docs/warpdrive/dropping-down.md#rules-that-still-apply) |
| U14 | [No performance, race-freedom or liveness proof](../../docs/warpdrive/guarantees.md#what-a-successful-check-means) |
| U15 | [Custom schedule/map correctness](../../docs/warpdrive/scheduling.md#custom-index-schedules) |
| U16 | [Storage identity, partitions and overlapping lifetimes](../../docs/warpdrive/storage.md) |
| U17 | [Named-barrier IDs and per-epoch participation](../../docs/warpdrive/channels.md#named-barrier-channels) |
| U18 | [Exclusive ownership completeness, bootstrap and skew](../../docs/warpdrive/channels.md#exclusive-consumers) |

## Historical gap labels

These are names used in old findings, not a roadmap of pending fixes.

| Label | Meaning / current disposition |
|---|---|
| G1 | Work-loop structural checks exist; stale state and useful work remain caller obligations (U10). |
| G2–G4 | Warp/register, mechanism/count and TMEM declaration domains hardened. Arbitrary CUDA resource feasibility is not certified. |
| G5 | Cross-role balance and dependency-cycle proof are not provided (U1/U2). |
| G6 | Raster empty/negative spaces and Int32 bounds handled through bind/grid; raw decoder inputs and custom mappings remain caller-owned. |
| G7 | Missing streams, duplicate declarations and scheduler-name collisions reject; post-collection mutation is unsupported. |
| G8 | Post-close access and raw accessor limits (U13). |
| G9 | TMA recognition is partial; verified means local group/byte evidence. |
| G10 | [Example input checks do not prove the entire advertised tensor contract](../../docs/warpdrive/examples.md#attention-forward). |
| G11 | [Swap-AB decode is OneShot-only](../../docs/warpdrive/examples.md#single-token-decode). |
