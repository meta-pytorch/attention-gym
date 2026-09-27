# Third-party notice

The upstream `NOTICE`, `LICENSE.txt` (Apache-2.0), `LICENSE-MIT.txt`, `LICENSING.md`, and
`THIRD_PARTY_LICENSES.txt` in this directory are copied unchanged from NVIDIA's `cudnn-frontend` repository at tag `v1.30.0`.

## Attention Gym adaptation

The scalar-GDN forward and backward kernels, their chain/warmup launch hosts, and the supporting
`common/` and `tile_dsl/` helpers are vendored from `python/cudnn/linear_attention/frost/` at that
tag. Only the modules the GDN launch paths import are kept. `kernel/gdp_bprop_v64_f16.py` is
imported by the chain prologue for the upstream compact-Q/dO GDP mode, which the Attention Gym
driver never selects (`compact_qdo=False`); it stays unmodified so the chain hosts match upstream.

Changes made for Attention Gym:

- imports were moved from `cudnn.frost.*` into this package, and the cuDNN host buffer/device
  utilities were replaced by the Torch-backed shims in `_compat.py`;
- `gdn.py` is a new driver that mirrors the upstream engine's plan choice (uncut, d_v split, exact
  piece chain, or approximate warmup split) and adds a minimum piece length for the exact chain;
- `common/split_k.py` compacts empty `cu_seqlens` intervals out of the unsplit work table, and
  `common/thd.py` skips TMA descriptor builds for empty sequences;
- `kernel/gdn_bprop_f16.py` computes dBeta without dividing by beta: it rebuilds a beta-free
  T = I - T_b L from the beta-scaled chunk inverse and stages Z = T^T dU for the dM, dK, and dBeta
  paths, so the gradient is exact at beta == 0; `build_cfg` rejects layouts this staging does not
  support;
- `gdn.py` uses the d_v split for any unchained forward whose two CTAs per tile fit in one wave;
- the launch compiles are persisted through Attention Gym's `jit_cache` (`_persist.py`).

Modified upstream files carry an explicit modification notice and keep their original SPDX
identifiers.
