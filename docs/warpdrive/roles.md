# Roles

[Back to warpdrive](index.md)

A role is a CuTeDSL method assigned to specific warps. In the built-in
[GEMM](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/gemm.py),
load, MMA and epilogue are separate roles. Each receives its declared channel endpoints and
work cursor, does its work, then returns the updated state.

This load-role excerpt runs on **warp 4**, not four warps:

```python
@role(warps=4, regs=48)
@cute.jit
def load(self, params: GemmParams, *, ab: Producer):
    for k in cutlass.range(params.num_k_tiles, unroll=1):
        with ab.produce() as h:
            ...  # load A/B for this stage
    return ab
```

Put `@role` directly above `@cute.jit`. Placement means:

```text
warps=4                       one instance: warp 4
warps=range(4)                 one instance: warps 0–3
warps=[range(4), range(4, 8)]   instance 0: warps 0–3
                              instance 1: warps 4–7
```

Each instance is a separate execution of the method on a nonempty, contiguous warp range.
Ranges cannot overlap other roles. Unassigned warps run no user role.

## Arguments

Parameter annotations tell the dispatcher what to pass. Names connect channels and other
named dependencies to kernel attributes:

```text
self.ab = Ring(...)  <---  ab: Producer in load
                    <---  ab: Consumer in mma
run_roles(params)    --->  params: GemmParams in both
```

Declare parameters explicitly: no `*args` or `**kwargs`. `*` makes dependencies keyword-only;
positional signatures also work.

| Parameter | Receives |
|---|---|
| `ab: Producer` / `ab: Consumer` | An end of `self.ab` |
| `s: Producer[Instance]` / `s: Consumer[Instance]` | An end of `self.s`, bound to this instance's slot |
| `stage: Instance` | Instance index as a Python int |
| `me: Local` | This instance's [own indices](#index-the-role-not-the-hardware) and size |
| `tmem: Tmem` | Kernel's TMEM view |
| `work: WorkCursor` | Cursor for `self.work` |
| `mufu: Turn` | A seat in the `self.mufu` turnstile |
| `bars: Barriers` | Base pointer of `self.bars` |
| `params: GemmParams` | Object passed to `run_roles(params)` |

Names matter for channels, streams, turnstiles and raw barriers. An instance-bound channel
must be `Slots`, not `Ring`; you still pass the slot explicitly, as in `s.wait(stage)`.

### Index the role, not the hardware

`cute.arch.thread_idx()` and `warp_idx()` depend on where the role runs, and
[`roles=` overrides](#choose-placement-in-the-constructor) move and resize roles. Role code
that counts its own threads takes a `Local`:

```python
@role(warps=range(1, 3))
@cute.jit
def consumer(self, ctx, *, ring: Consumer, me: Local):
    t = me.tid  # 0..63 wherever the two warps are placed
    if me.warp == 0:
        ...
```

| Attribute | Value |
|---|---|
| `me.tid` | Thread within the instance: hardware thread minus the instance's first thread |
| `me.warp` | Warp within the instance, warp-uniform |
| `me.lane` | Lane within the warp |
| `me.tmem_row` | The TMEM row this thread may access ([below](#tmem-rows-follow-the-hardware-warp)) |
| `me.warps`, `me.threads` | The instance's resolved size, as Python ints |

Each instance counts from its own first warp. At warp 0, these constant offsets give the
hardware IDs. Raw roles accept `Local` too. Use `me.warps` to size per-warp work or reject
an override when the math requires a fixed size.

#### TMEM rows follow the hardware warp

Use `me.tmem_row` for TMEM accumulators and per-row data, as FA4 and GEMM do. Unlike
role-local indices, TMEM access follows the hardware warp: `tcgen05.ld`/`st` from warp `w`
reach lanes `32 * (w % 4)` through `32 * (w % 4) + 31`.

`me.tmem_row` is the hardware thread index modulo `me.threads`. It gives a row in
`0 .. me.threads - 1`, at TMEM lane `tmem_row % 128`, for any contiguous placement of 4k
warps. Using `me.tid` for this works only on a warpgroup boundary. Accessing `tmem_row`
with a warp count not divisible by 4 raises `ProtocolError` during tracing.

### Pass shared setup as params

Build a frozen dataclass in the kernel prologue from the atoms, partitions, layouts and loop
counts the roles need:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class GemmParams:
    tma_a: cute.CopyAtom
    tAgA: cute.Tensor
    num_k_tiles: cutlass.Int32
```

Frozen fields cannot be rebound. Tensor contents and referenced lists can still change;
this is not read-only memory.

An annotation matching the params type or a base class receives that object.
`run_roles(params)` checks the type. Alternatively, pass a `SimpleNamespace` and leave the
role's single context parameter unannotated; the layer does not make it immutable.

`ProtocolError` covers missing dependencies, unresolved annotations, multiple context
parameters and incompatible context types. Mutable dataclasses are rejected even when
unannotated. String annotations—including all annotations under
`from __future__ import annotations`—resolve in the role's module globals, so define params
types at module level.

## Return the updated state

Return every endpoint, work cursor and turnstile seat in its current state:

```python
@role(warps=5, regs=48)
@cute.jit
def mma(self, params: GemmParams, *, ab: Consumer, acc: Producer, work: WorkCursor, tmem: Tmem):
    while work.valid:
        ...  # use the channel ends and the current item
        work.advance()
    return ab, acc, work
```

Return one object or a tuple; tuple order does not matter. Missing, duplicate or foreign
state is an error with tracing on or off, for one or two CTAs.

- Return **all** endpoints, cursors and seats—not `self`, params, instance indices, TMEM
  views or raw barrier pointers.
- Close stage handles inside the role. They are temporary lifetimes, not state to return.
- Helpers must return updated state too; reassign it at the call site:

  ```python
  ab = load_next(ab, params)
  ```

Do not return a saved pre-loop snapshot. Checks catch ownership/return errors and discarded
advancing-helper results for endpoints and cursors, including inside loops. They do **not**
prove freshness of reconstructed snapshots, manually supplied raw state or turnstile seats.

The returned endpoints supply final phases for producer tails. Returned cursors also let
tracing close any remaining work range.

## Registers belong to warpgroups

`regs=48` requests a per-thread `setmaxnreg` budget for the role's warpgroup (four warps).
`None`, the default, leaves the budget unchanged—it does not request automatic tuning.

- All roles in one warpgroup must agree, including `None` versus a number.
- Numeric budgets are multiples of 8 in `[24, 256]`.
- Any numeric budget pads the launch to complete warpgroups. Padding warps take their
  group's budget; a wholly idle group takes 24.
- Launch with `launch_config()`: with any budget it sets `min_blocks_per_mp=1`. CuTeDSL's
  launch emits no other entry bound, and without one ptxas silently drops `setmaxnreg` and
  compiles every warp at one register count.
- Budget adjustment runs in a warpgroup-uniform branch before dispatch, including in a
  pair's peer CTA. Adjacent warpgroups with one budget share a branch; non-adjacent ones
  never do (see [configuration](configuration.md#placement-and-registers)).

Redistribution cannot create registers. Every launched thread starts with
`min(256, floor((65536 / threads) / 8) * 8)` registers; `None` counts as that budget.
For a 12-warp launch, this is 168 registers per thread:

```text
available:       168 × 384 = 64512 registers
232 / 232 / 40:  (232 + 232 + 40) × 128 = 64512    fits
232 / 232 / 48:  (232 + 232 + 48) × 128 = 65536    rejected
```

The three numbers are warpgroup budgets. The second split raises `ValueError`: although it
fits 65536, it exceeds the launch's starting pool and its increase would wait forever.

## CTA placement and common entry

| `cta=` | Where this role runs |
|---|---|
| `"all"` (default) | Every CTA |
| `"leader"` | Rank 0 of each MMA pair in the [cluster](clusters.md) |

Both are equivalent for one CTA. Without MMA pairs, every CTA is a leader.

Call `begin_roles()` and `run_roles()` in converged CTA/cluster code with the declared launch
shape. One lane of warp 0 initializes each managed mbarrier; the common initialization
fence/sync precedes dispatch. **Every participating thread reaches initialization and exit,
including warp holes.** See [raw barriers and roles](dropping-down.md#escape-hatches-by-feature)
for manually initialized barriers.

## Choose placement in the constructor

Use the decorator for defaults. To override placement by method name, set `self.roles` in
`__init__` with frozen `RoleConfig` values:

```python
from attn_gym.warpdrive import RoleConfig


class Attention(WarpSpecialized):
    def __init__(self, tile_hq: int, roles: dict[str, RoleConfig] | None = None):
        defaults = {}
        if tile_hq > 16:
            defaults["softmax"] = RoleConfig([range(4), range(4, 8)], regs=120)
        self.roles = {**defaults, **(roles or {})}
        ...  # channels and storage

    @role(warps=[range(4), range(4, 8)], regs=192)
    @cute.jit
    def softmax(self, params, *, stage: Instance): ...
```

This is a placement sketch, not a complete kernel. The decode example computes its 32-head
FA4 register split this way; `fa4_fwd` accepts overrides too:

```python
from attn_gym.warpdrive import RoleConfig
from attn_gym.warpdrive.examples.fa4_fwd import fa4_fwd

roles = {
    "softmax": RoleConfig([range(4), range(4, 8)], regs=184),
    "correction": RoleConfig(range(8, 12), regs=96),
}
out, lse = fa4_fwd(q, k, v, roles=roles)
```

`RoleConfig` has the decorator's `warps`, `regs`, `cta` and `raw` fields. It copies lists of
ranges into tuples; `dataclasses.replace` can derive a variant.

The resolved placement gets the same checks as decorators: nonoverlapping contiguous warps,
one budget per warpgroup, and a fitting register pool. Unknown names or non-`RoleConfig`
values raise `ProtocolError` at collection. `describe()` shows the resolved budgets.

**Placement does not repartition tensors.** Update partitions when moving work.
[`Local`](#index-the-role-not-the-hardware) indices follow the role; hardware IDs do not.

**Collection freezes declarations.** The first `describe()`, trace or compile-cache lookup
collects the kernel. Construct a new instance to change it; mutation or recollection is
unsupported. Assigning `self.roles` after construction also makes the compile key raise:
the constructor no longer describes the kernel.

## Compile keys

Use a module-level [`jit_cache`](cache.md) helper. It reuses compiled kernels in memory and on
disk across processes. Schematic helper (define `MyKernel` and `fake_args`
for your ABI):

```python
@jit_cache
def compile_kernel(op: MyKernel):
    return compile_tvm_ffi(op, *fake_args, name=op.get_name())
```

`op.__cute_cache_key__()` includes:

- The constructor call, with defaults filled in and its arguments copied/frozen for the key.
- Cluster and each work stream's schedule, mapping, readiness and scheduler warp.
- The trace settings that change the code (`enable_trace`'s backend, and its `waits` and
  `events_per_role` where they apply), `trace_protocol`, `alias_check`, and resolved roles.
- Source of the modules defining the kernel, cluster and stream-component classes, including
  their base classes.

Thus `Gemm()` and `Gemm(ab_stages=3)` share an entry. A different role budget does not.
Editing a kernel subclass in your own source file recompiles it, even when the compile helper
does not import that file.

Put static algorithm choices in constructor arguments. Mutation rules are deliberately
narrow:

| Change | Contract |
|---|---|
| `op.tile_n = 128` after construction | Key raises `TypeError`; pass it to the constructor |
| Reassign cluster, work stream or `trace_protocol` before collection | Allowed and keyed |
| Reassign those after collection | Raises `ProtocolError`; channels already depend on them |
| Change the trace settings (`enable_trace`) or `alias_check` | Allowed and keyed |
| Mutate a constructed object, e.g. `op.ab.n = 4` | Not tracked; do not do it |

A cache hit skips tracing. `jit_cache` [restores](cache.md#cached-builds-were-never-traced)
TMA, SMEM and alias reports, opt-outs and warnings for `describe()`. Protocol logs need a
fresh trace; the example helpers use the optional [uncached hook](cache.md#force-a-new-trace)
for those builds. Kernels pickle as their constructor call, so `compile_many` can build
several in parallel.

Keep kernel definitions in source files: CuTeDSL re-reads them for AST preprocessing.
