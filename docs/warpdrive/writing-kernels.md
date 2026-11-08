# Write a warpdrive kernel

[Back to warpdrive](index.md) · [Safety contract](guarantees.md)

Start with a complete kernel. warpdrive connects the roles and manages declared handoffs;
you write the CuTeDSL math, layouts and data movement inside them.

```text
work item       what output does this CTA/cluster own?
    |
roles + edges   who produces each value, who reads it, when can storage be reused?
    |
role bodies     exchange handles, do the math, return current state
    |
launch + tests  enforce the input contract, then check the actual lifetimes
```

You do not need the numbered fuzz rejection catalog to author a kernel. You do need the
[safety contract](guarantees.md): successful compilation is not a proof of correctness.

## Run a complete example

[gemm.py](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/gemm.py) is the smallest complete GEMM: three roles, SMEM partitions,
TMEM views, launch setup, cached compilation and a PyTorch wrapper. It computes BF16
`D = A @ B.T` with FP32 accumulation on Blackwell SM100/SM103. Use CuTeDSL 4.6.2 or newer and
the repository's [environment setup](index.md#tests).

From the checkout root, in that environment:

```bash
python -m attn_gym.warpdrive.examples.gemm
pytest -q test/warpdrive/test_warpdrive.py -k test_gemm
```

On a shared GPU host, run each command through its GPU reservation tool with a process
timeout. The module prints numerical errors; the tests assert correctness and wrapper
boundaries. Keep kernel definitions in a source file: the CuTeDSL compiler re-reads the
Python source, so a REPL or notebook cell is not enough.

Read the example in this order:

| Part | What to learn or adapt |
|---|---|
| `Gemm.__init__` | Channels, stages, storage and work assignment |
| `load / mma / epilogue` | The math and handoffs for one item |
| `Gemm.kernel` and `GemmParams` | Initialization, layouts and role-specific partitions |
| `Gemm.__call__` | Atoms, TMA descriptors, bound work and launch |
| `compile_gemm / check_gemm_args / gemm` | Static choices, runtime ABI and input checks |

For an operation without MMA, keep the role/channel structure and omit TMEM. Do not copy
MMA-specific layouts or the GEMM's register split merely to get a skeleton.

## Define one work item

In the default GEMM, item `(m_tile, n_tile, 0)` owns one 128 × 256 output tile. All three
roles visit that item. Load and MMA exchange `ceil(K / 64)` input tiles; epilogue stores one
result.

Write the equivalent sentence for your operation before choosing a schedule:

- Which output does one CTA or cluster own?
- Which inputs and reduction steps produce it?
- Do all roles agree on those steps?

Start with `OneShot()` unless the algorithm needs persistence. The
[persistent GEMM](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/persistent_gemm.py) shows roles reused across items, CTA pairs
and alternate schedules, with an ABI that supports changing the schedule.

For split reductions, give each partial a distinct item such as `(tile, split)` and reduce
partials separately. Claiming the last item does **not** mean earlier items have finished.

## Draw the handoffs

The default GEMM uses two channels:

```text
warp 4: load A/B -- ab: 3 SMEM stages --> warp 5: MMA
                                               |
                                      acc: 1 TMEM stage
                                               |
                                               v
                                       warps 0–3: store D
```

| Channel | Producer → consumer | Epochs per item | Storage can be reused after… |
|---|---|---|---|
| `ab` | TMA load → MMA | `ceil(K / 64)` | MMA completion releases the input stage |
| `acc` | MMA → epilogue threads | 1 | Every epilogue thread releases the accumulator stage |

An **epoch** is one exchanged value. A **stage** is a reusable storage slot:

```text
epoch:  0  1  2  3  4  5 ...
stage:  0  1  2  0  1  2 ...    (three-stage ring)
```

Adding stages adds capacity, not values: producer and consumer epoch counts must still
match. With joined consumers, all readers release before reuse. Other ownership policies
have their own [channel contracts](channels.md).

Draw feedback edges too. Identify the initial value or free capacity that lets something
run first; balanced counts do not help if every role starts by waiting for another.

## Declare storage and connect the roles

The GEMM constructor contains these declarations (an excerpt, not a standalone kernel):

```python
self.ab = Ring(ab_stages, Tma(stage_bytes), Umma, buffer={"A": a, "B": b})
self.acc = Ring(acc_stages, Umma, Threads)
self.tmem_plan = TmemPlan().region("acc", tile_n, acc_stages)
self.work = WorkStream(schedule=OneShot())
```

- `a` and `b` are buffers declared in `SmemPlan`. They hold the A/B payload; the channel
  separately owns barrier state that controls publication and reuse.
- `stage_bytes` counts TMA's A and B writes for **one stage**. At the default tile size it is
  `(128 + 256) * 64 * 2` bytes for BF16. Do not multiply by ring depth.
- `self.acc` controls the accumulator lifetime; `TmemPlan` allocates its TMEM columns.
- Parameter names connect roles: `ab: Producer` and `ab: Consumer` both bind to `self.ab`.

`@role(warps=4)` names warp **index 4**, not four warps. Use `warps=range(4)` for warps 0–3.
TMA and UMMA ends require a single-warp role; `Threads` counts every bound thread. See
[roles](roles.md) for bindings and register-budget rules.

Start with separate buffers for overlapping lifetimes. Allocate with `SmemPlan`, build
TMA/MMA partitions with CuTe, then wrap them in `StageView`. `h.view(...)` selects the handle's stage.
You still own the layouts and must ensure each view refers to its named buffer. Read
[storage](storage.md) before adding aliases or accessing buffers outside handles.

## Implement the lifecycle, then the math

The load role follows this structure:

```python
@role(warps=4, regs=40)
@cute.jit
def load(self, params: GemmParams, *, ab: Producer, work: WorkCursor):
    ab = ab.views(A=params.tAsA, B=params.tBsB)
    while work.valid:
        m, n, _ = work.item
        for k in cutlass.range(params.num_k_tiles, unroll=1):
            with ab.produce() as h:
                cute.copy(
                    params.tma_a, params.tAgA[None, m, k], h.view("A"), tma_bar_ptr=h.barrier
                )
                cute.copy(
                    params.tma_b, params.tBgB[None, n, k], h.view("B"), tma_bar_ptr=h.barrier
                )
        work.advance()
    return ab, work
```

Follow one input stage through the consumer:

1. `with ab.consume() as h:` waits for the TMA writes to complete.
2. MMA reads that stage through `h.view(...)`.
3. Exiting the handle requests its declared UMMA completion signal. Reuse waits for that
   completion, not merely for the Python body to finish issuing asynchronous MMA.

The MMA role holds `acc.produce()` around the **whole K reduction**, publishing once.
Epilogue holds `acc.consume()` until it has finished reading that accumulator stage.

Index threads through the role, not the hardware: take `me: Local` and use `me.tid` /
`me.warp`, so a `roles=` override that moves the role keeps its indices; TMEM rows use
`me.tmem_row` ([role-local indices](roles.md#index-the-role-not-the-hardware)).

Keep participation outside lane-dependent predicates. For a ragged output, every epilogue
thread enters and releases its handle; only in-bounds lanes store. Guarding the whole
exchange can strand a barrier whose count still includes the absent lanes.

Carry state and respect handle lifetimes:

- Return every endpoint, cursor and turnstile seat from each role. An advancing helper must
  also return its updated state, and its caller must reassign it.
- Do not return handles, params or TMEM views.
- Close a handle exactly once, within its opening dynamic region. Do not keep using its
  pointers after release.
- For non-nested lifetimes, use explicit `acquire/wait/commit/release`; see
  [overlapping handles](channels.md#overlapping-handles).

## Wire the prologue, launch and cache

Keep four boundaries visible:

| Boundary | Your responsibilities |
|---|---|
| **Entry point** | Build atoms/descriptors and bind runtime work. Name the inner kernel with `set_name_prefix(...)`; launch with `self.launch_config(bound)`. Persistent schedules pass `launch_args()` through the entry point ([Bind and launch](scheduling.md#bind-and-launch)). |
| **Kernel prologue** | Call `begin_roles(work=bound, layouts=...)` in converged code. Get buffers, build partitions and frozen params, then call `run_roles(params)`. Annotate every TMEM-using role with `tmem: Tmem` for teardown. |
| **Compile helper** | Follow the example's [compile-key contract](roles.md#compile-keys). Use fake compact tensors for a compact ABI. Key every static code choice; runtime dimensions may stay dynamic within their constraints. |
| **Python wrapper** | Validate dtype, device, rank, related shapes, compactness, alignment and output overlap before relying on that ABI. Define empty-output and empty-reduction behavior. |

In particular, zero K work must not publish an uninitialized sum. A successful compile does
not replace these checks: the wrapper enforces the input contract and the role bodies
implement it.

## Check the kernel before tuning

Compare nonuniform inputs against a reference with an explicit tolerance. GEMM's tests
compare FP32-reference error against the BF16 PyTorch baseline; another reduction or
approximation needs its own criterion.

Exercise the lifetimes your kernel claims to support:

- One item, then enough epochs to wrap every ring's stages **and phases**.
- Ragged tails and supported empty work/reductions.
- Persistent multi-item units and uneven work between units.
- Repeated launches and changed runtime shapes through one compiled entry point.
- Every advertised cluster/schedule path; separate workspaces for concurrent launches.

Use `op.describe()` before compilation for declaration checks, then after tracing for
TMA/alias findings. Explain and test every `unverified` site. `no hazard found` is not a
proof. For a new raw async handoff or overlapping storage, independently validate completion
and ordering, including appropriate sanitizer checks. Investigate a timeout as a failure;
do not extend the run until it passes.

Run the [repository testing tiers](https://github.com/meta-pytorch/attention-gym/blob/main/AGENTS.md#warpdrive-testing-tiers) before
tuning stages, warps, registers, layouts or schedules. Hold workload and numerical checks
fixed across measurements. More stages or fewer instructions alone do not establish speed.

## Where to go next

- [Persistent GEMM](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/persistent_gemm.py): multiple items, pairs and multicast.
- [FA4 forward](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/fa4_fwd.py): multiple consumers, slots and overlapping storage.
- [Swap-AB decode](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/fa4_decode_swapab.py): round-robin instances and named channels.
  `SharedKVDecode` (`shared_kv=True` in wrappers) adds exclusive consumers and FA4's
  rendezvous. It is currently OneShot-only, not a persistent template.
- [Packed work](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/packed_work.py): device-built lists and readiness before decoding.
- [Dropping down](dropping-down.md): raw barriers and targeted opt-outs. Explain the
  replacement ordering; do not suppress a check simply to make compilation pass.

For agent-assisted authoring, ask for the work item, role/channel diagram and epoch counts
before implementation. The local [authoring skill](https://github.com/meta-pytorch/attention-gym/blob/main/.agents/skills/authoring-warpdrive/SKILL.md) follows this guide and the
safety contract. Finish with a runnable kernel, tests and remaining raw/unverified sites—not
just declarations or a profiler screenshot.
