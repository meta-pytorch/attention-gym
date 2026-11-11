# warpdrive

![warpdrive](images/warpdrive-hero-light.svg#gh-light-mode-only)
![warpdrive](images/warpdrive-hero-dark.svg#gh-dark-mode-only)

warpdrive helps write warp-specialized CuTeDSL kernels, and is also the CuTeDSL authoring base
of attention-gym: its [compile cache](cache.md), CuTeDSL version shims, the intra-kernel
[profiler](profiler.md) and [work scheduling](scheduling.md) live here.

At a high level this provides nice utilities for the common mechanisms (channels) to communicate between warps. While also
providing some syntactic sugar that helps guarantee correctness when you author your kernels in a certain way.
Connect them with channels:

```python
self.ab = Ring(3, Tma((128 + 256) * 64 * 2), Umma)
self.acc = Ring(1, Umma, Threads)
```

You (your agent) still choose the tiles, layouts, warps and register budgets. warpdrive handles the barrier
counts and phases, role dispatch and TMEM allocation/free. The individual warps bodies are ordinary CuTeDSL.

Requires `nvidia-cutlass-dsl` 4.6.2 or newer (tested on 4.6, 4.7 and 4.8).

**Start with [Write a kernel](writing-kernels.md):** run the complete GEMM, define a work
item, connect the roles, and validate the handoffs. The [safety contract](guarantees.md) explains
what warpdrive handles and what you must establish. You can still compile busted kernels, but its harder.
We also got a skill boss: [authoring skill](https://github.com/meta-pytorch/attention-gym/blob/main/.agents/skills/authoring-warpdrive/SKILL.md).

**For the vocabulary,** see [channels](channels.md#stages-and-phases). The
[tour notebook](tour.ipynb) builds from a
two-warp producer/consumer kernel to the pipelined GEMM, then schedules, clusters, storage
aliases and the checkers, with saved `describe()` output, PTX excerpts and checks against
PyTorch.

## A GEMM

GEMM is one example of what you can author with warpdrive. The two channels above connect
a load role, an MMA role and an epilogue:

```text
warp 4: load A/B --> warp 5: MMA --> warps 0–3: store D
```

Here is the shape of those methods inside a `WarpSpecialized` kernel. This is pseudocode for
one output tile; layouts, MMA setup, work cursors and launch code are omitted.

```python
@role(warps=4, regs=48)
@cute.jit
def load(self, params: GemmParams, *, ab: Producer):
    for k in cutlass.range(params.num_k_tiles):
        with ab.produce() as tile:
            ...  # issue TMA loads of A/B into tile, completing tile.barrier
    return ab


@role(warps=5, regs=48)
@cute.jit
def mma(self, params: GemmParams, *, ab: Consumer, acc: Producer, tmem: Tmem):
    with acc.produce() as result:
        for k in cutlass.range(params.num_k_tiles):
            with ab.consume() as tile:
                ...  # issue MMA using tile, accumulating into result's TMEM stage
    return ab, acc


@role(warps=range(4), regs=160)
@cute.jit
def epilogue(self, params: GemmParams, *, acc: Consumer, tmem: Tmem):
    with acc.consume() as result:
        ...  # finish reading result from TMEM and store D
    return acc
```

`@role` places the warps; `Producer` and `Consumer` bind each role to its channel ends.
The handles delimit when a stage is safe to fill, read and reuse. You supply the CuTeDSL
operations inside them. `warps=4` means warp index 4; `warps=range(4)` means warps 0–3.

[gemm.py](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/gemm.py)
has the complete implementation. See [Write a kernel](writing-kernels.md) to build your own,
or [Examples](examples.md#gemm) to run this one.

## How it fits together

The two rings control when input and accumulator stages are ready to read or safe to reuse.
`ab.produce()` waits for a free input stage; TMA fills it, and the MMA role consumes it.
The stage can be reused after MMA completes.

- [Roles](roles.md) receive data through `params` and channel ends through annotated
  arguments: `ab: Producer` binds to `self.ab`. Each role returns its updated state.
- [Channels](channels.md) describe publication and reuse. `Ring` cycles through stages;
  `Slots` lets you choose a slot. `Threads` counts all bound threads, even if only some write
  data; `Warps` counts one elected lane per warp.
- [Scheduling](scheduling.md) decides which items each CTA runs. Start with `OneShot()`;
  persistent schedules reuse the role loop, with their own
  [launch setup](scheduling.md#bind-and-launch).
- [Storage](storage.md) connects buffers to channel stages. `begin_roles()` initializes
  barriers and returns a `SmemAllocator`; `run_roles(params)` dispatches the roles and frees
  TMEM after its users finish.

Use `print(op.describe())` to inspect roles, warps, registers, channel counts, TMA bytes and
scheduling. `op.describe(format="mermaid")` draws the pipeline. See
[Tracing](tracing.md#inspect-the-kernel) for examples and IKET capture.

## More examples and details

- [Tour](tour.ipynb): learn the API through small kernels, with results from a GB300.
- [Roles](roles.md): place warps, pass parameters and carry state.
- [Scheduling](scheduling.md): choose a schedule or supply a custom work list.
- [Channels](channels.md): exchange data, coordinate readers and manage overlapping work.
- [Storage](storage.md): bind stage views and check storage aliases.
- [Clusters](clusters.md): use CTA pairs, multicast and cross-CTA handoffs.
- [Dropping down](dropping-down.md): mix managed channels with raw synchronization.
- [Tracing](tracing.md): inspect declarations, diagnose handoffs and capture execution.
- [Profiler](profiler.md): record with IKET, read timings and explore Perfetto traces.
- [Compile cache](cache.md): reuse builds and understand what causes a rebuild.
- [Examples](examples.md): GEMM, grouped GEMM, attention and Qwen3 decode, with input limits
  and benchmark commands.
- [Configuration](configuration.md): find the available knobs and fixed choices.

## Tests

From the repository root:

```bash
pip install -e ".[tests,linear,dev]"  # uv: add --prerelease allow
pytest -n 12 test/warpdrive test/profiler test/warpdrive_fuzz
pytest -n 12 -m fuzz test/warpdrive test/warpdrive_fuzz
```

The `tests` extra supplies `perfetto`, which the tracing and profiler tests import. The first
pytest command runs the normal suite, including the fuzz regressions in
`test/warpdrive_fuzz/test_static.py`; tests marked `fuzz` are skipped by default. The second
opts into those (seeded declaration and mutation fuzzing, and the check that synthetic layer
bugs fail the tests). Compiled kernels stay in the [compile cache](cache.md), so a second
run mostly loads them. FA4 tests skip without `flash_attn.cute`. Long campaigns follow the
[fuzz harness guide](https://github.com/meta-pytorch/attention-gym/blob/main/test/warpdrive_fuzz/README.md).

The suite covers numerical correctness and determinism, shape reuse, schedules, CTA pairs and
larger clusters, protocol errors, raw/managed synchronization, TMA and alias diagnostics, and
both trace backends' paths into the profiler.

<details markdown="1">
<summary>Testing another CuTeDSL version</summary>

Use a separate environment with this checkout and its own caches:

```bash
uv venv /tmp/dsl-4.7 && source /tmp/dsl-4.7/bin/activate
uv pip install -e ".[linear,dev]" perfetto "nvidia-cutlass-dsl[cu13]==4.7.1"
export CUTE_DSL_CACHE_DIR=/tmp/dsl-4.7/cute-dsl-cache
export WARPDRIVE_CACHE_DIR=/tmp/dsl-4.7/warpdrive-cache
pytest -n 12 test/warpdrive
```

`perfetto` is installed by name because the `tests` extra also installs FlashAttention on
x86_64, which pins its own CuTeDSL version. The FA4 examples need `flash_attn.cute` and a
`quack-kernels` release that supports the CuTeDSL version under test (quack 0.6.5 needs 4.7;
0.6.3 works with 4.6). Version-dependent CuTeDSL APIs are covered in
[CuTeDSL versions](cache.md#cutedsl-versions).

</details>
