# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""KDA over flash-linear-attention's triton kernels.

The fla kernels are triton, so this backend covers CUDA architectures the
cute kernels do not. ``fla`` is an optional dependency; importing this module
without it raises with an install pointer.

The adapter feeds fla the same contract ``chunk_kda`` exposes: q/k already
l2-normalized, ``gate`` the bound log-decay, ``beta`` already sigmoided --
so every ``use_*_in_kernel`` switch stays off.
"""

import torch


def _import_fla_chunk_kda():
    try:
        from fla.ops.kda import chunk_kda as fla_chunk_kda
    except ImportError as err:
        raise ImportError(
            "kernel_options={'backend': 'fla'} needs flash-linear-attention: "
            "pip install fla-core. It is an optional dependency, like the "
            "cute kernels' cutlass."
        ) from err
    return fla_chunk_kda


def chunk_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    *,
    cu_seqlens: torch.Tensor | None,
    scale: float,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    fla_chunk_kda = _import_fla_chunk_kda()
    # The two libraries transpose the recurrent state's last two axes
    # relative to each other; the adapter owns the flip in both directions.
    if initial_state is not None:
        initial_state = initial_state.transpose(-1, -2).contiguous()
    output, final_state = fla_chunk_kda(
        q,
        k,
        v,
        gate,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
        cu_seqlens=cu_seqlens,
    )
    if output_final_state and final_state is not None:
        final_state = final_state.transpose(-1, -2).contiguous()
    return output, final_state if output_final_state else None
