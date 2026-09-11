# cuDNN delta-rule CuTeDSL kernels

This package contains the shared SM100/SM103 cuDNN kernels for delta-rule variants. The KDA and
scalar-GDN prefill, checkpoint-recompute, and bprop cores are adapted from NVIDIA's Frost
implementations; `kda_plain_gate_bwd.py` is an Attention Gym-authored dense gate-gradient helper.
Variant-owned adapters provide Torch validation and orchestration over the shared runtime.

## Raw specialization

- CuTeDSL 4.7 or newer on NVIDIA SM100 or SM103.
- Native FP16 or BF16 Q/K/V with K=V=128.
- FP32 per-token natural-log gate increments: `[T,H,K]` for KDA and `[T,H]` for GDN.
- FP32 post-sigmoid beta.
- Packed THD execution with contiguous int32 `cu_seqlens`, including tails and empty sequences.
- Internal recurrent state and checkpoints use `[sequence, head, V, K]`.

The `[V, K]` state layout is schedule-native rather than an inherited naming choice. State GEMMs use
the value dimension as the MMA M mode and contiguous key vectors as the K mode; forward, recompute,
and bprop share that orientation. It now matches the public and paged delta-rule state contract, so
stateful calls require no layout conversion at the cuDNN boundary.

## Checkpoint contract

For a nonempty sequence of length `L`, checkpoint recompute stores `ceil(L / N)` entering states at
interval `N`. Row zero is the provided initial state or zeros. Empty sequences allocate no row and
emit no token work. Bprop consumes the same entering-state convention.

## Scheduling

The persistent kernels use one work item per `(sequence, head)` for exact execution. The optional
forgetting-horizon split table is an approximate scheduling primitive for contracting-update
experiments; it is not part of the raw kernel's exactness guarantee.

Pipeline indices and phases are CTA-lifetime state and advance across persistent work items. The
prologue constructs runtime tensor maps and the work table before the role-partitioned main kernel
runs.

## Source and licensing

The Frost-derived kernels and required `common/` and `tile_dsl/` helpers are adapted from NVIDIA
`cudnn-frontend` commit `085d50b33691f06e2309f8e6724741a021985649`. Runtime imports were moved
into the Attention Gym namespace, cuDNN host utilities became plain Torch calls plus
`attn_gym._backends.cute.utils` helpers, and `tile_dsl/` is
trimmed to the single-CTA, non-block-scaled paths these kernels use. Helpers with a `cute.arch`
equivalent use it; the mbarrier try-wait spin, TMA tensor copies over runtime descriptors, and
tcgen05 descriptor stepping stay on `cutlass.experimental.primitives` (see the comments in
`tile_dsl/barrier.py` for the SASS evidence). There is no
`cudnn.frost` runtime dependency. `kda_plain_gate_bwd.py` remains BSD-3-Clause Attention Gym code.

See `NOTICE.md`, `LICENSE.Apache-2.0`, and `LICENSE.MIT`.
