# warpdrive

![warpdrive](../../docs/warpdrive/images/warpdrive-hero-light.svg#gh-light-mode-only)
![warpdrive](../../docs/warpdrive/images/warpdrive-hero-dark.svg#gh-dark-mode-only)

warpdrive helps write warp-specialized CuTeDSL kernels, and is also the CuTeDSL authoring base
of attention-gym: its [compile cache](../../docs/warpdrive/cache.md), CuTeDSL version shims, the intra-kernel
[profiler](../../docs/warpdrive/profiler.md) and work scheduling live here.

Write the load, MMA and epilogue of a warp-specialized CuTeDSL kernel as separate methods.
Connect them with channels:

```python
self.ab = Ring(3, Tma((128 + 256) * 64 * 2), Umma)
self.acc = Ring(1, Umma, Threads)
```

You still choose the tiles, layouts, warps and register budgets. warpdrive handles the barrier
counts and phases, role dispatch and TMEM allocation/free. The math is ordinary CuTeDSL.

Experimental. Requires `nvidia-cutlass-dsl` 4.6.2 or newer (tested on 4.6, 4.7 and 4.8) and a
Blackwell SM100/SM103 GPU.

**Start with [Write a kernel](../../docs/warpdrive/writing-kernels.md):** run the complete GEMM, define a work
item, connect the roles, and validate the handoffs. The [safety contract](../../docs/warpdrive/guarantees.md) explains
what warpdrive handles and what you must establish. Compilation alone does not prove liveness.
For agent-assisted implementation, use the repository's [authoring skill](../../.agents/skills/authoring-warpdrive/SKILL.md).

**To relearn the vocabulary,** read the [tour notebook](../../docs/warpdrive/tour.ipynb): it builds from a
two-warp producer/consumer kernel to the pipelined GEMM, then schedules, clusters, storage
aliases and the checkers, with saved `describe()` output, PTX excerpts and checks against
PyTorch.

## A GEMM

[gemm.py](examples/gemm.py) computes BF16 `D = A @ B.T`. Warp 4 loads A/B, warp 5 issues MMA,
and warps 0–3 store D. Here is the declaration and load role; the linked file has the MMA,
epilogue, tensor partitions and launch:

```python
import cutlass
from cutlass import cute
from attn_gym.warpdrive import (
    OneShot,
    Producer,
    Ring,
    SmemPlan,
    Threads,
    Tma,
    TmemPlan,
    Umma,
    WarpSpecialized,
    WorkCursor,
    WorkStream,
    role,
)
from attn_gym.warpdrive.examples.gemm import GemmParams


class Gemm(WarpSpecialized):  # excerpt; other roles and the kernel prologue are in gemm.py
    def __init__(self):
        self.smem_plan = SmemPlan()
        a = self.smem_plan.buffer("A", cutlass.BFloat16, align=1024)
        b = self.smem_plan.buffer("B", cutlass.BFloat16, align=1024)
        self.ab = Ring(3, Tma((128 + 256) * 64 * 2), Umma, buffer={"A": a, "B": b})
        self.acc = Ring(1, Umma, Threads)
        self.tmem_plan = TmemPlan().region("acc", 256)
        self.work = WorkStream(schedule=OneShot())

    @role(warps=4, regs=40)
    @cute.jit
    def load(self, params: GemmParams, *, ab: Producer, work: WorkCursor):
        ab = ab.views(A=params.tAsA, B=params.tBsB)
        while work.valid:
            m, n, _ = work.item
            for k in cutlass.range(params.num_k_tiles, unroll=1):
                with ab.produce() as h:
                    cute.copy(
                        params.tma_a,
                        params.tAgA[None, m, k],
                        h.view("A"),
                        tma_bar_ptr=h.barrier,
                    )
                    cute.copy(
                        params.tma_b,
                        params.tBgB[None, n, k],
                        h.view("B"),
                        tma_bar_ptr=h.barrier,
                    )
            work.advance()
        return ab, work
```

`ab.produce()` waits for a free stage; TMA fills it and completes its full barrier. The MMA
role waits with `ab.consume()`, reads that stage and releases it for the next load.
`warps=4` means warp index 4, not four warps.

Run the complete example through its wrapper:

```python
import torch
from attn_gym.warpdrive.examples.gemm import gemm

a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
b = torch.randn_like(a)
out = gemm(a, b)
```

## How it fits together

A role gets its input values from `params` and its channel ends from annotated arguments.
The argument `ab: Producer` binds to `self.ab`. Use a frozen dataclass for `params`; return
every endpoint, work cursor and turnstile seat you take as an argument. Helpers must return
updated state too. See [roles](../../docs/warpdrive/roles.md) for placement and binding.

Channels describe handoffs, not the math. A `Ring` cycles through stages; `Slots` lets you name
the slot. `Tma`, `Umma`, `Threads` and `Warps` say how each side signals. For `Threads`, all
threads of the bound role participate, even when only some lanes have data to write; `Warps`
counts one elected lane per warp.

`WorkStream` decides which work each CTA runs. `mapping=` turns a linear index into an item;
`schedule=` chooses the indices. Change `OneShot()` to `StaticPersistent()`,
`AtomicPersistent()` or `ClusterLaunchControl()` without changing the role loop. The host
setup must also supply the [schedule's launch arguments](../../docs/warpdrive/scheduling.md#bind-and-launch).

In the kernel prologue, `begin_roles()` initializes the barriers and returns a `SmemAllocator`.
Allocate your tensors there, or declare a `SmemPlan` and bind its buffers to channels with
`buffer=`. `run_roles(params)` synchronizes, runs the roles and frees TMEM after its users
finish. [Storage](../../docs/warpdrive/storage.md) shows how a handle selects its own buffer stage.

`print(op.describe())` shows what warpdrive worked out for a kernel: roles, warps, register
budgets, every channel's per-CTA arrival counts, masks and TMA bytes, the scheduler and TMEM.
[Tracing](../../docs/warpdrive/tracing.md#inspect-the-kernel) has the full output for the 2-CTA persistent GEMM.
`op.describe(format="mermaid")` draws the same pipeline as a
[diagram](../../docs/warpdrive/tracing.md#draw-the-pipeline).

## More examples and details

- [Tour](../../docs/warpdrive/tour.ipynb): every noun and verb on small runnable kernels, executed on a GB300.
- [Roles](../../docs/warpdrive/roles.md): placement, params, returned state and register budgets.
- [Scheduling](../../docs/warpdrive/scheduling.md): launch setup, custom work lists and CUTLASS/FA4 mappings.
- [Channels](../../docs/warpdrive/channels.md): rings, slots, round-robin instances, exclusive consumers,
  named barriers, overlapping handles, try tokens, previews, TMEM and turnstiles.
- [Storage](../../docs/warpdrive/storage.md): stage views and checking declared aliases.
- [Clusters](../../docs/warpdrive/clusters.md): two-CTA MMA, larger clusters, TMA multicast and cross-CTA
  handoffs.
- [Dropping down](../../docs/warpdrive/dropping-down.md): raw barriers, count overrides and unchecked sites.
- [Tracing](../../docs/warpdrive/tracing.md): inspect the kernel, capture IKET ranges and use the profiler.
- [Profiler](../../docs/warpdrive/profiler.md): compact intra-kernel records, summaries and Perfetto traces.
- [Compile cache](../../docs/warpdrive/cache.md): `jit_cache`, what keys include, cache location and CuTeDSL
  version compatibility.
- [Examples](../../docs/warpdrive/examples.md): persistent GEMM, MoE grouped GEMM, FA4 forward and backward,
  swap-AB decode, a Qwen3-0.6B decode megakernel, their input limits and benchmarks.
- [Configuration](../../docs/warpdrive/configuration.md): which choices of the layer and the examples are
  constructor arguments, which are fixed, and why.

## Tests

From the repository root:

```bash
pip install -e ".[linear,dev]"
pytest -n 12 test/warpdrive test/profiler
pytest -n 12 -m fuzz test/warpdrive test/warpdrive_fuzz
```

The first pytest command runs the normal suite; fuzz tests are skipped by default. The second
opts into quick fuzzing (declaration and mutation fuzzing, and the check that synthetic layer
bugs fail the tests). Compiled kernels stay in the [compile cache](../../docs/warpdrive/cache.md), so a second
run mostly loads them. FA4 tests skip without `flash_attn.cute`. Long campaigns follow the
[fuzz harness guide](../../test/warpdrive_fuzz/README.md).

The suite covers numerical correctness and determinism, shape reuse, schedules, CTA pairs and
larger clusters, protocol errors, raw/managed synchronization, TMA and alias diagnostics, and
both trace backends' paths into the profiler.

To test another supported CuTeDSL version, install it into a separate environment along with
this checkout, and give it its own caches:

```bash
uv venv /tmp/dsl-4.7 && source /tmp/dsl-4.7/bin/activate
uv pip install -e ".[linear,dev]" "nvidia-cutlass-dsl[cu13]==4.7.1"
export CUTE_DSL_CACHE_DIR=/tmp/dsl-4.7/cute-dsl-cache
export WARPDRIVE_CACHE_DIR=/tmp/dsl-4.7/warpdrive-cache
pytest -n 12 test/warpdrive
```

The FA4 examples also need `flash_attn.cute` and a `quack-kernels` release that supports that
CuTeDSL version (quack 0.6.5 needs 4.7; 0.6.3 works with 4.6). Version-dependent CuTeDSL APIs
are covered in [CuTeDSL versions](../../docs/warpdrive/cache.md#cutedsl-versions).
