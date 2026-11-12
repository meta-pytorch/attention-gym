# Copyright (c) 2026 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Optional-dependency isolation tests for the warpdrive delta-rule implementation."""

import subprocess
import sys

import pytest
import torch


def test_linear_import_does_not_load_warpdrive_kernel_dependencies() -> None:
    """Keep the warpdrive driver modules behind explicit warpdrive dispatch."""
    code = """
import sys
import attn_gym.linear

assert "attn_gym.linear._delta_rule.warpdrive.launchers.forward" not in sys.modules
assert "attn_gym.linear._delta_rule.warpdrive.launchers.backward" not in sys.modules
assert "attn_gym.linear._delta_rule.warpdrive.launchers.gdn_forward" not in sys.modules
assert "attn_gym.linear._delta_rule.warpdrive.launchers.gdn_backward" not in sys.modules
assert not any(name.startswith("attn_gym.linear._delta_rule.warpdrive") for name in sys.modules)
assert not any(name.startswith("cutlass.experimental") for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


_DRIVER_PATHS = """
import sys
import torch
import attn_gym.linear
from attn_gym.linear import chunk_gdn, chunk_kda
from attn_gym.linear._delta_rule.warpdrive.drivers import gdn, kda, summary
from attn_gym.testing import make_gdn_test_inputs
from attn_gym.testing.kda import cumulative_sequence_offsets, make_kda_test_inputs

built = set()


def recording(driver, name):
    build = getattr(driver, name)

    def record(**kwargs):
        tiles = kwargs.get("tiles_per_head", 1)
        built.add((driver.__name__.rsplit(".", 1)[1], name, kwargs.get("split"), tiles))
        return build(**kwargs)

    return record


for driver in (gdn, kda):
    driver.MIN_CHAIN_TOKENS_PER_PIECE_FWD = driver.MIN_CHAIN_TOKENS_PER_PIECE_BWD = 0
    for name in (
        "build_warmup_forward",
        "build_chain_forward",
        "build_warmup_backward",
        "build_chain_backward",
    ):
        setattr(driver, name, recording(driver, name))
warpdrive = {"backend": "warpdrive"}
lengths = (3000, 0, 2000)
cu = cumulative_sequence_offsets(lengths)
tokens = sum(lengths)
for heads in (1, 17, 32):  # chained; d_v split forward; uncut
    q, k, v, gate, beta, state, _ = make_gdn_test_inputs(
        lengths, key_heads=heads, value_heads=heads, seed=5
    )
    leaves = [t.requires_grad_() for t in (q, k, v, gate, beta, state)]
    out, final = chunk_gdn(
        *leaves[:5], initial_state=leaves[5], output_final_state=True, cu_seqlens=cu,
        kernel_options=warpdrive,
    )
    torch.autograd.grad((out.sum(), final.sum()), leaves)
    q, k, v, gate, beta = make_kda_test_inputs(
        tokens, heads=heads, seed=7, normalize_qk=True, requires_grad=True
    )
    out, _ = chunk_kda(q, k, v, gate, beta, cu_seqlens=cu, kernel_options=warpdrive)
    torch.autograd.grad(out.sum(), (q, k, v, gate, beta))
    split = {**warpdrive, "split_forward": True, "split_backward": True}
    out, _ = chunk_kda(q, k, v, gate, beta, cu_seqlens=cu, kernel_options=split)
    torch.autograd.grad(out.sum(), (q, k, v, gate, beta))
staged = [t.detach()[:, :, :1].contiguous() for t in (k, v, gate)]
summary.build_state_summaries(*staged, beta.detach()[:, :, :1].contiguous(), cu)
torch.cuda.synchronize()

for family in ("gdn", "kda"):
    for scheme in ("chain_forward", "chain_backward"):
        assert (family, f"build_{scheme}", None, 1) in built, (family, scheme, built)
    assert (family, "build_warmup_forward", False, 2) in built, (family, built)
    assert (family, "build_warmup_backward", False, 1) in built, (family, built)
assert ("kda", "build_warmup_forward", True, 1) in built, built
assert ("kda", "build_warmup_backward", True, 1) in built, built
assert not [name for name in sys.modules if name.startswith("cutlass.experimental")]
"""


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the warpdrive backend requires SM100 or SM103",
)
def test_warpdrive_driver_paths_build_every_bundle_without_cutlass_experimental() -> None:
    """Every warpdrive driver path (uncut, d_v split, chain, split tables, summaries) builds its
    launch bundle, and none imports ``cutlass.experimental``."""
    pytest.importorskip("cutlass.cute", reason="the warpdrive backend requires CuTeDSL")
    subprocess.run([sys.executable, "-c", _DRIVER_PATHS], check=True)
