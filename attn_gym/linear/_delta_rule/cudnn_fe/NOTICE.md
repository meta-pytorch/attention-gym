# Third-party notice

The upstream `NOTICE`, `LICENSE.txt` (Apache-2.0), `LICENSE-MIT.txt`, `LICENSING.md`, and
`THIRD_PARTY_LICENSES.txt` in this directory are copied unchanged from NVIDIA's `cudnn-frontend`
repository at tag `v1.30.0`.

## Attention Gym adaptation

The scalar-GDN and KDA linear-attention kernels (`kernel/`), their shared helpers (`common/`), and
the low-level tile helpers (`tile_dsl/`) are vendored from
`python/cudnn/linear_attention/frost/{kernel,common}` and `python/cudnn/frost/tile_dsl` at that
tag.
Only the modules the GDN and KDA launch paths import are kept; GDN2, GDP (including the
`gdp_bprop_v64` backward fork; `compact_qdo` is rejected), SDPA, and the upstream engines are
omitted. The Torch drivers `gdn.py`, `kda.py`, and `summary.py` are Attention Gym code.

Changes made for Attention Gym:

- Imports were moved from `cudnn.frost.*` into this package; host device queries, dtype checks, and
  TMA validation use PyTorch instead of cuDNN host shims.
- Restyle without SASS changes: `tile_dsl/` and `common/` were pruned to the helpers the kept
  kernels reach and call the `cute.arch` wrappers where the generated SASS is unchanged (packed
  FP32 multiply and FMA stay inline PTX); kernels use register tensors, a `SharedStorage` struct
  with `SmemAllocator` for tile buffers, and tensor-aware `SmemTile` bases, and call the swizzle
  helpers where they match the inline offset math exactly; the whole package is formatted and
  linted with Ruff (code at 99 columns; wide docstring tables are kept).
- Compilation: frozen config dataclasses and module-level `jit_cache` compile functions over fake
  TVM-FFI tensor signatures (`common/tvm_ffi.py`) that reproduce the upstream placeholder ABI, with
  int64-extent variants for oversized tensors, so no live placeholder tensors are allocated and
  compiled kernels persist across processes.
- Bug fixes: mbarriers initialized by one thread (GDN prefill and backward); the KDA delta residual
  stays FP32 through subtraction and beta scaling in every kernel that forms it; the GDN bprop
  computes dBeta without dividing by beta; split tables omit empty and zero-chunk work items;
  scalar-gate split scans respect the head stride; split-table replay preserves the
  scheduler-counter ABI; per-channel gates are checked for the vectorized scan's row alignment.
- Features: paged recurrent state (null, fresh, and empty-sequence semantics) in the GDN and KDA
  prefill kernels, and context-parallel forward and reverse state summaries over arbitrary bounds.
- Validation: host launch contracts are checked before compiling (work-table and scratch shapes,
  checkpoint intervals and capacity, warp-role and named-barrier maps, and scheduler arrival counts
  derived from the warp count).

Attention Gym previously vendored older copies of the KDA and paged scalar-GDN kernels from
cudnn-frontend commit `085d50b33691f06e2309f8e6724741a021985649` under
`_delta_rule/cudnn/kernels/`; this package replaces them, and every cuDNN route now launches the
v1.30 kernels.

Modified upstream files carry an explicit modification notice and keep their original SPDX
identifiers.
