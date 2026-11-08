# Roles

[Back to warpdrive](index.md)

A role is a CuTeDSL method assigned to specific warps. The dispatcher gives it channel ends,
a work cursor and other declared dependencies; the method does the math and returns its
updated protocol state.

This partial load role runs on **warp 4**:

```python
@role(warps=4, regs=40)
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

Each instance occupies a nonempty contiguous range. Ranges cannot overlap other roles;
warp holes run no user role.

### Registers belong to warpgroups

`regs=40` requests a per-thread `setmaxnreg` budget for the role's warpgroup (four warps).
`None`, the default, leaves the budget unchanged—it does not request automatic tuning.

- All roles in one warpgroup must agree, including `None` versus a number.
- Numeric budgets are multiples of 8 in `[24, 256]`.
- Any numeric budget pads the launch to complete warpgroups. Padding warps take their
  group's budget; a wholly idle group takes 24.
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

### CTA placement and common entry

| `cta=` | Where this role runs |
|---|---|
| `"all"` (default) | Every CTA |
| `"leader"` | Rank 0 of each MMA pair in the [cluster](clusters.md) |
| `"rank0"` | Once per cluster, on cluster rank 0 |

All three are equivalent for one CTA. Without MMA pairs, every CTA is a leader.

Call `begin_roles()` and `run_roles()` in converged CTA/cluster code with the declared launch
shape. One lane of warp 0 initializes each managed mbarrier; the common initialization
fence/sync precedes dispatch. **Every participating thread reaches initialization and exit,
including warp holes.** See [raw barriers and roles](dropping-down.md#escape-hatches-by-feature)
for manually initialized barriers.

## Arguments

The dispatcher binds by annotation and, for named dependencies, parameter name:

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

Each instance of a multi-instance role counts from its own first warp. The offsets are
constants, so an instance at warp 0 computes exactly the hardware ids. Raw roles accept
`Local` too. Use `me.warps` to size per-warp work, or to raise `ProtocolError` when the
math fixes the size an override could change.

#### TMEM rows follow the hardware warp

`me.tid` and `me.warp` are role-local; TMEM lane access is not. `tcgen05.ld`/`st` from
hardware warp `w` reach only lanes `32 * (w % 4)` onward, wherever the role is placed.
`me.tmem_row` is the TMEM row this thread may access, `0 .. me.threads - 1` at lane
`tmem_row % 128`: the hardware thread index modulo the instance's size. It is right for any
contiguous placement of a role of 4k warps, while `me.tid` is right only for an instance that
starts on a warpgroup boundary. Tracing `tmem_row` in an instance whose warp count is not a
multiple of 4 raises `ProtocolError`. The FA4 and GEMM examples index their TMEM accumulators
and per-row data with it.

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

Frozen means fields cannot be rebound; tensor contents and referenced lists can still change.
It is shallow immutability, not read-only memory.

An annotation matching the params type or a base class receives that object.
`run_roles(params)` checks the type. Alternatively, pass a `SimpleNamespace` and leave the
role's single context parameter unannotated; the layer does not make it immutable.

`ProtocolError` covers missing dependencies, unresolved annotations, multiple context
parameters and incompatible context types. Mutable dataclasses are rejected even when
unannotated. String annotations—including all annotations under
`from __future__ import annotations`—resolve in the role's module globals, so define params
types at module level.

## Return the updated state

A role returns the current endpoints, work cursor and turnstile seats it received:

```python
@role(warps=5, regs=40)
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

Returning an old pre-loop snapshot is wrong. Checks catch ownership/return errors and
channel-end or cursor state superseded by an advancing helper, including discarded helper
results inside loops. They do **not** prove freshness of reconstructed/laundered snapshots,
manually supplied raw state or turnstile seats.

The returned endpoints supply final phases for producer tails. Returned cursors also let
tracing close any remaining work range.

## Choose placement in the constructor

The decorator is the readable default. Set `self.roles` in `__init__` to replace defaults by
method name, using frozen `RoleConfig` values:

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

Two boundaries matter:

1. **Placement does not repartition tensors.** Update your partitions when moving work.
   Indices from [`Local`](#index-the-role-not-the-hardware) follow the role; raw hardware
   ids do not.
2. **Collection freezes declarations.** The first `describe()`, trace or compile-cache lookup
   collects the kernel. Construct a new instance to change it; mutation or recollection is
   unsupported. Assigning `self.roles` after construction also makes the compile key raise,
   because the constructor no longer describes the kernel.

## Compile keys

Use a module-level [`jit_cache`](cache.md) helper. It reuses compiled kernels in memory and on
disk across processes. Schematic helper (define `MyKernel` and `fake_args`
for your ABI):

```python
@jit_cache(uncached=lambda op: op.trace_protocol)  # protocol logs require a real trace
def compile_kernel(op: MyKernel):
    return compile_tvm_ffi(op, *fake_args, name=op.get_name())
```

`op.__cute_cache_key__()` includes:

- The constructor call, with defaults filled in and its arguments copied/frozen for the key.
- Cluster and each work stream's schedule, mapping, readiness, recording and scheduler warp.
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

A cache hit performs no new trace; `jit_cache` [restores](cache.md#cached-builds-were-never-traced)
saved TMA, SMEM and alias reports, opt-outs and associated warnings so `describe()` still reports
the build's findings. Protocol-traced builds are uncached. Kernels pickle as their constructor
call, so `compile_many` can build several in parallel.

Keep kernel definitions in source files: CuTeDSL re-reads them for AST preprocessing.
