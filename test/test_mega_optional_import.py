# Copyright (c) 2026 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Optional-dependency isolation tests for the Mega delta-rule implementation."""

import subprocess
import sys


def test_linear_import_does_not_load_mega_kernel_dependencies() -> None:
    """Keep CuTeDSL 4.7 modules behind explicit Mega dispatch."""
    code = """
import sys
import attn_gym.linear

assert "attn_gym.linear._delta_rule.mega.forward" not in sys.modules
assert "attn_gym.linear._delta_rule.mega.backward" not in sys.modules
assert "attn_gym.linear._delta_rule.mega.gdn_forward" not in sys.modules
assert "attn_gym.linear._delta_rule.mega.gdn_backward" not in sys.modules
assert not any(name.startswith("attn_gym.linear._delta_rule.mega.kernels") for name in sys.modules)
assert not any(name.startswith("cutlass.experimental") for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)
