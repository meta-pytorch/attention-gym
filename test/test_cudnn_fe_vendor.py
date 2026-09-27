"""The vendored cudnn-frontend v1.30 linear-attention package imports without the cuDNN wheel."""

from __future__ import annotations

import importlib
import pkgutil
import sys

import pytest

pytest.importorskip("cutlass")


def test_every_vendored_cudnn_fe_module_imports() -> None:
    package = importlib.import_module("attn_gym.linear._delta_rule.cudnn_fe")
    names = [info.name for info in pkgutil.walk_packages(package.__path__, package.__name__ + ".")]
    assert any(name.endswith(".kernel.kda_bprop_f16") for name in names)
    assert any(name.endswith(".kernel.gdn_bprop_f16") for name in names)
    for name in names:
        importlib.import_module(name)
    # The drop must not depend on the upstream cuDNN frontend wheel.
    assert not any(name == "cudnn" or name.startswith("cudnn.") for name in sys.modules)
