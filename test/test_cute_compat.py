"""Version selection and canonical aliases for optional CuTeDSL backends."""

import importlib.metadata
import importlib.util
import sys
import types
import warnings

import pytest
from packaging.specifiers import InvalidSpecifier
from packaging.version import Version

pytest.importorskip("cutlass")

from attn_gym._backends.cute import compat


@pytest.mark.parametrize(
    "installed,specifier,expected",
    [
        ("4.6.2", ">=4.6.2,<4.8", True),
        ("4.6.2", "==4.6.2", True),
        ("4.6.2", "==4.6.1", False),
        ("4.7.0", ">=4.6,<4.8,!=4.7.0", False),
        ("4.7.1", ">=4.6,<4.8,!=4.7.0", True),
        ("4.7.1", ">=4.8", False),
        ("4.8.0", ">=4.8", True),
        ("4.8.0", ">=4.6,<4.8", False),
        ("4.10.0", ">=4.8", True),
        ("4.8.0rc1", ">=4.8", False),
        ("4.8.0rc1", ">=4.8.0rc1", True),
        ("4.9.0.dev1", ">=4.8", True),
        ("4.8.0.post1", ">=4.8", True),
        ("4.8.0+local", "==4.8.0", True),
    ],
)
def test_version_matches(monkeypatch, installed: str, specifier: str, expected: bool):
    monkeypatch.setattr(compat, "CUTEDSL_VERSION", Version(installed))
    assert compat.cutedsl_version_matches(specifier) is expected


def test_invalid_version_specifier_is_not_silently_ignored():
    with pytest.raises(InvalidSpecifier):
        compat.cutedsl_version_matches("not-a-version-range")


@pytest.mark.parametrize("installed", ["4.6.2", "4.7.1", "4.8.0", "4.9.0.dev1"])
def test_import_selects_only_available_api(monkeypatch, installed: str):
    """The unused branch may not exist at all; aliases preserve class identity."""
    calls = []

    def package_version(name):
        calls.append(name)
        assert name == "nvidia-cutlass-dsl"
        return installed

    monkeypatch.setattr(importlib.metadata, "version", package_version)
    for name in ("cutlass.memory", "cutlass.tensor_utils", "cutlass.utils"):
        monkeypatch.setitem(sys.modules, name, None)

    smem = type("SmemAllocator", (), {})
    tmem = type("TmemAllocator", (), {})
    layout = type("LayoutEnum", (), {})

    def alloc_cols(tensor):
        return tensor

    modern = Version(installed) >= Version("4.8")
    memory_name = "cutlass.memory" if modern else "cutlass.utils"
    memory = types.ModuleType(memory_name)
    memory.SmemAllocator, memory.TmemAllocator = smem, tmem
    memory.get_num_tmem_alloc_cols = alloc_cols
    monkeypatch.setitem(sys.modules, memory_name, memory)
    if modern:
        tensors = types.ModuleType("cutlass.tensor_utils")
        tensors.LayoutEnum = layout
        monkeypatch.setitem(sys.modules, "cutlass.tensor_utils", tensors)
    else:
        memory.LayoutEnum = layout

    spec = importlib.util.spec_from_file_location("compat_under_test", compat.__file__)
    selected = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selected)
    assert selected.SmemAllocator is smem
    assert selected.TmemAllocator is tmem
    assert selected.LayoutEnum is layout
    assert selected.get_num_tmem_alloc_cols is alloc_cols
    assert selected.CUTEDSL_VERSION == Version(installed)
    for _ in range(3):
        assert selected.cutedsl_version_matches(f"=={installed}")
    assert calls == ["nvidia-cutlass-dsl"]


def test_real_aliases_are_canonical_without_deprecation_warnings():
    """Exercise the installed CuTeDSL, not just simulated API namespaces."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        spec = importlib.util.spec_from_file_location("compat_under_test", compat.__file__)
        selected = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(selected)
        if selected.cutedsl_version_matches(">=4.8"):
            from cutlass.memory import SmemAllocator, TmemAllocator, get_num_tmem_alloc_cols
            from cutlass.tensor_utils import LayoutEnum
        else:
            from cutlass.utils import (
                LayoutEnum,
                SmemAllocator,
                TmemAllocator,
                get_num_tmem_alloc_cols,
            )

        assert selected.SmemAllocator is SmemAllocator
        assert selected.TmemAllocator is TmemAllocator
        assert selected.LayoutEnum is LayoutEnum
        assert selected.get_num_tmem_alloc_cols is get_num_tmem_alloc_cols
