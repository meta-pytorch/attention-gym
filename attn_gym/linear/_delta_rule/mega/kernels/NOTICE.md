# Third-party notice

## Upstream notice

cudnn-frontend

Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This product includes software developed at NVIDIA CORPORATION
(https://www.nvidia.com/).

The upstream project is distributed primarily under the Apache License, Version 2.0, with a subset
of files under the MIT License as declared by each file's SPDX identifier.

## Attention Gym adaptation

The low-level KDA and scalar-GDN prefill, checkpoint-recompute, bprop kernels, and supporting
helpers are adapted from NVIDIA's `cudnn-frontend` repository at commit
`085d50b33691f06e2309f8e6724741a021985649`.

Source mapping:

- `linear_attention/frost/kernel/gdn_prefill_f16.py` ->
  `_delta_rule/mega/kernels/gdn_prefill_f16.py`;
- `linear_attention/frost/kernel/gdn_recompute_f16.py` ->
  `_delta_rule/mega/kernels/gdn_recompute_f16.py`;
- `linear_attention/frost/kernel/gdn_bprop_f16.py` ->
  `_delta_rule/mega/kernels/gdn_bprop_f16.py`;
- `linear_attention/frost/kernel/kda_prefill_f16.py` ->
  `_delta_rule/mega/kernels/kda_prefill_f16.py`;
- `linear_attention/frost/kernel/kda_recompute_f16.py` ->
  `_delta_rule/mega/kernels/kda_recompute_f16.py`;
- `linear_attention/frost/kernel/kda_bprop_f16.py` ->
  `_delta_rule/mega/kernels/kda_bprop_f16.py`;
- required `linear_attention/frost/common/` and `frost/tile_dsl/` files retain their package names.

Changes made for Attention Gym:

- imports were moved from `cudnn.frost.*` into this package;
- cuDNN host/device utilities were replaced by the Torch shim in `compat.py`;
- host dtype validation, device-aware compilation caching, role invariants, and stale documentation
  were tightened for the Attention Gym integration;
- unused upstream forward, recompute, and bprop replay wrappers were removed; the public adapters
  call the cached launch paths directly;
- FP8 conversion and MMA helpers used only by upstream SDPA kernels were omitted from this
  linear-attention subset;
- split backward scheduling omits empty-sequence work items, preventing invalid persistent-kernel
  TMEM lifecycle transitions;
- package-marker descriptions identify the vendored subset;
- Attention Gym adapters map tensors, scheduling metadata, gradients, and state layouts to the
  kernel ABIs.

Original copyright and license notices are retained where present. Modified upstream files carry an
explicit modification notice. Kernel and linear-attention common helpers are Apache-2.0 licensed;
low-level tile helpers are MIT licensed. Copies are provided in `LICENSE.Apache-2.0` and
`LICENSE.MIT`.
