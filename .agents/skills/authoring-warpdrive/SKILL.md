---
name: authoring-warpdrive
description: Authors and ports warp-specialized CuTeDSL kernels using warpdrive. Use when building a kernel from roles, channels and work streams, or adapting an existing kernel to that API.
---

# Authoring warpdrive kernels

This skill ships in the repository that holds the library (`attn_gym/warpdrive/`). Resolve its
relative paths from this directory, using the same checkout for docs, examples and imported code. The repository docs and implementation
are the API authority; do not substitute another checkout's newer API.

Use this for a new warpdrive kernel or a port to managed roles. A standalone CuTeDSL kernel
that doesn't use warpdrive needs the usual CuTeDSL guidance, not this workflow. Layer-internal
changes also need the relevant topic contracts and the repository's layer validation tiers.

## Read the authoring path

Read [Write a kernel](../../../docs/warpdrive/writing-kernels.md) and the [safety contract](../../../docs/warpdrive/guarantees.md).
Then read the closest complete example, including its host wrapper, launch and compile helper:

- [GEMM](../../../attn_gym/warpdrive/examples/gemm.py): single-CTA TMA → MMA → epilogue.
- [Persistent GEMM](../../../attn_gym/warpdrive/examples/persistent_gemm.py): repeated work, schedules and clusters.
- [Grouped GEMM](../../../attn_gym/warpdrive/examples/grouped_gemm.py): a custom `WorkMap` decoding device-side group
  sizes, per-item TMA addressing and a ragged TMA store.
- [FA4 forward](../../../attn_gym/warpdrive/examples/fa4_fwd.py): slots, fan-out and overlapping lifetimes.
- [Swap-AB decode](../../../attn_gym/warpdrive/examples/fa4_decode_swapab.py): round-robin and named channels, and in
  `SharedKVDecode` (`shared_kv=True` in the wrappers) exclusive consumers plus a named-barrier
  rendezvous; not a persistent template.
- [Packed work](../../../attn_gym/warpdrive/examples/packed_work.py): mapping a device-built list, not a complete kernel.
- [Qwen3 decode megakernel](../../../attn_gym/warpdrive/examples/qwen_megakernel.py): a whole decode step as one
  persistent launch; host instruction table, static schedule and raw cross-CTA dependencies.

Read the linked topic guide when using that feature. If investigating a fuzz report, resolve
its labels in `test/warpdrive_fuzz/static.py` and `mutate.py`, where each case states the
rejection it expects. They are not library error codes or an authoring checklist.

## Establish the protocol before implementing

State the input/output contract, numerical reference and supported shapes. For each work
item, record which CTA/cluster owns the result, the roles and their warps, and each channel's
producer, consumers, epoch count and storage lifetime. A short diagram or table is enough.
Include feedback edges: identify the initial available value or capacity that allows progress.

Keep role math and tensor layouts explicit. Choose managed channels when they express the
handoff, not an additional abstraction around the CuTeDSL math. Begin without storage aliases
unless the algorithm requires them. Preserve the ability to use raw code where necessary.

When adapting an existing kernel, compare both protocols before replacing synchronization:
which operation completes the access, which threads signal, which CTA's barrier is targeted,
and when the stage may be overwritten. Similar-looking acquire/commit calls are not evidence
of equivalent semantics.

## Implement against the contract

Follow the guide's lifecycle, prologue and launch sections. In particular:

- Return current endpoints, cursors and turnstile seats; helpers return and callers reassign
  state. Do not restore a saved pre-loop snapshot.
- Take role-local thread/warp indices from `me: Local`, not raw `thread_idx()`/`warp_idx()`,
  so `roles=` overrides keep them right; TMEM rows use `me.tmem_row`.
- Keep the declared participants in each handoff; predicate data accesses, not lane-dependent
  participation. Account for epochs across work items, including odd trip counts.
- Bind views to the buffers they actually partition. Handle stage projection is not a proof
  of storage identity, layout validity or safe aliasing.
- Use the current example's compile helper and fake runtime ABI (the
  [compile cache](../../../docs/warpdrive/cache.md) explains keys). Put static choices in the
  constructor so they reach the cache key, validate runtime alignment/compactness before relying on them, and keep scheduler
  workspaces disjoint across concurrent launches.
- For raw synchronization, overrides or opt-outs, explain the exact ordering supplied instead.
  Never suppress a check just to obtain a successful compile. Inspect unverified TMA/alias
  sites rather than interpreting silence as proof.
- Raw waits and signals in a TMEM or MMA role need the tcgen05 thread-sync fences that managed
  ones emit ([dropping down](../../../docs/warpdrive/dropping-down.md#tcgen05-thread-sync-fences-on-raw-synchronization));
  run `ptx_check.check_tcgen05_sync` on the kernel's PTX to find a missing one. Drop a
  redundant managed fence only with `skip_fence=..., reason=...`
  ([skip a fence](../../../docs/warpdrive/dropping-down.md#skip-a-redundant-thread-sync-fence)),
  and rerun that check over every configuration the skip depends on.

## Validate and hand back

Use the repository's [testing tiers](../../../AGENTS.md#warpdrive-testing-tiers) and its GPU
reservation rules. Verify imports point at this checkout before running tests. Keep launches
bounded by a process timeout; a timeout is a failure requiring diagnosis, not a performance
measurement or permission to extend the run.

Test the operation's numerical and lifecycle contract: one item, stage/phase wraparound,
supported empty/ragged work, multiple persistent items, repeated launches, and the advertised
schedule/cluster variants. For newly introduced async or alias lifetimes, add targeted
ordering tests and appropriate sanitizer checks. Don't run long campaigns merely because the
maintainer catalog lists them; follow the repository's scope and authorization rules.

Establish correctness before benchmarking. The example `bench_*.py` scripts next to each
kernel show the interleaved CUDA-graph timing pattern; [tracing](../../../docs/warpdrive/tracing.md)
and the [profiler](../../../docs/warpdrive/profiler.md) show where role time goes. Compare a fixed
workload and reference, keep
buffers alive throughout timed/replayed work, and report codegen/resource changes when they
are relevant. A compile-only check or a profiler plan is not a correctness or speed result.

Deliver the runnable kernel and wrapper, tests and reproduction commands, the role/channel
sketch, and any remaining limitations or raw/unverified sites. Report tests that couldn't run
as validation gaps. Finish the authorized implementation and checks; don't stop at a skeleton
or silently broaden the supported input domain.
