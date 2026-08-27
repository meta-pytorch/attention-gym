import subprocess
import sys

import attn_gym
import attn_gym.linear
import attn_gym.sparse
from attn_gym.linear import Impl, chunk_gdn, recurrent_gdn
from attn_gym.linear.kda.api import Impl as KDAImpl
from attn_gym.linear.types import Impl as SharedImpl
from attn_gym.masks import (
    batchify_mask_mod,
    generate_spatial_head_mask_mod,
    generate_temporal_head_mask_mod,
    generate_vision_cross_attention_mask_mod,
)
from attn_gym.mods import generate_mla_rope_score_mod


def test_attention_namespaces_are_exported():
    assert attn_gym.linear.__name__ == "attn_gym.linear"
    assert attn_gym.sparse.__name__ == "attn_gym.sparse"
    assert "linear" in attn_gym.__all__
    assert "sparse" in attn_gym.__all__
    assert "paged_attention" not in attn_gym.__all__


def test_linear_impl_uses_shared_owner():
    assert Impl is SharedImpl
    assert KDAImpl is SharedImpl


def test_gdn_operations_are_exported():
    assert callable(chunk_gdn)
    assert callable(recurrent_gdn)
    assert "gated_delta_rule" not in attn_gym.linear.__all__


def test_documented_mask_and_score_mods_are_exported():
    assert all(
        callable(function)
        for function in (
            batchify_mask_mod,
            generate_spatial_head_mask_mod,
            generate_temporal_head_mask_mod,
            generate_vision_cross_attention_mask_mod,
            generate_mla_rope_score_mod,
        )
    )


def test_linear_base_import_keeps_cutedsl_lazy():
    """The optional CuTeDSL backend must never become an eager import."""
    script = "import sys, attn_gym.linear; assert 'cutlass' not in sys.modules"
    subprocess.run([sys.executable, "-c", script], check=True)


def test_base_import_does_not_require_numpy():
    """The base package declares only torch and must not import NumPy directly."""
    script = (
        "import builtins\n"
        "original_import = builtins.__import__\n"
        "def without_numpy(name, globals=None, locals=None, fromlist=(), level=0):\n"
        "    if level == 0 and (name == 'numpy' or name.startswith('numpy.')):\n"
        "        raise ModuleNotFoundError('blocked NumPy import')\n"
        "    return original_import(name, globals, locals, fromlist, level)\n"
        "builtins.__import__ = without_numpy\n"
        "import attn_gym\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_short_conv_decode_uses_decode_name():
    assert "causal_conv1d_decode" in attn_gym.linear.__all__
    assert "causal_conv1d_update" not in attn_gym.linear.__all__
