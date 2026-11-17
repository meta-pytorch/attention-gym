"""Version selection and canonical aliases for optional CuTeDSL backends."""

import importlib.metadata
import importlib.util
import math
import sys
import types
import warnings
from types import SimpleNamespace

import pytest
import torch
from packaging.specifiers import InvalidSpecifier
from packaging.version import Version

pytest.importorskip("cutlass")

from cuda.bindings import driver as cuda
from cutlass import Int32, cute

from attn_gym.warpdrive import Local, WarpSpecialized, compat, role
from attn_gym.warpdrive.cache import compile_tvm_ffi, make_fake_compact_tensor


@pytest.mark.parametrize(
    "installed,specifier,expected",
    [
        ("4.6.2", ">=4.6.2,<4.8", True),
        ("4.6.2", "==4.6.1", False),
        ("4.7.0", ">=4.6,<4.8,!=4.7.0", False),
        ("4.7.1", ">=4.6,<4.8,!=4.7.0", True),
        ("4.8.0", ">=4.8", True),
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

    # The shims compat selects by version must come from the simulated API, too.
    import cutlass.base_dsl
    from cutlass import cute

    arch, divisor, arrive, store = (object() for _ in range(4))
    if Version(installed) >= Version("4.7"):
        monkeypatch.setattr(cutlass.base_dsl, "Arch", arch, raising=False)
        monkeypatch.setattr(cute, "FastDivmodDivisorV2", divisor, raising=False)
        monkeypatch.setattr(cute.arch, "mbarrier_arrive", arrive, raising=False)
    else:
        arch_module = types.ModuleType("cutlass.base_dsl.arch")
        arch_module.Arch = arch
        monkeypatch.setitem(sys.modules, "cutlass.base_dsl.arch", arch_module)
        monkeypatch.setattr(cute, "FastDivmodDivisor", divisor, raising=False)
    if modern:
        monkeypatch.setattr(cute.arch, "store_async_dsmem", store, raising=False)
    divmod_flip = Version(installed) >= Version("4.9.0.dev1")
    if divmod_flip:
        monkeypatch.setattr(cute, "DivmodSemantics", lambda mode: ("divmod", mode), raising=False)

    spec = importlib.util.spec_from_file_location("compat_under_test", compat.__file__)
    selected = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selected)
    assert selected.Arch is arch
    assert selected.FastDivmodDivisor is divisor
    assert (selected.mbarrier_arrive is arrive) == (Version(installed) >= Version("4.7"))
    assert (selected.store_async_dsmem is store) == modern
    assert selected.NATIVE_MULTI_ADVANCE == modern
    assert selected.COMPILE_OPTIONS == ((("divmod", "legacy"),) if divmod_flip else ())
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


def test_import_rejects_cutedsl_below_the_floor(monkeypatch):
    """Below 4.6.2 the backends fail at import with the reason, not later with a GPU hang."""
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "4.6.1")
    spec = importlib.util.spec_from_file_location("compat_floor_probe", compat.__file__)
    with pytest.raises(ImportError, match=r"nvidia-cutlass-dsl>=4\.6\.2, found 4\.6\.1.*elect"):
        spec.loader.exec_module(importlib.util.module_from_spec(spec))


class SignedDivmod(WarpSpecialized):
    """One thread writes ``a % b`` and ``a // b`` of runtime operands."""

    @role(warps=0, raw=True)
    @cute.jit
    def compute(self, ctx, me: Local):
        if me.tid == 0:
            ctx.out[0] = ctx.a % ctx.b
            ctx.out[1] = ctx.a // ctx.b

    @cute.kernel
    def kernel(self, out: cute.Tensor, a: Int32, b: Int32):
        self.begin_roles()
        self.run_roles(SimpleNamespace(out=out, a=a, b=b))

    @cute.jit
    def __call__(self, out: cute.Tensor, a: Int32, b: Int32, stream: cuda.CUstream):
        self.kernel(out, a, b).launch(**self.launch_config(grid=(1, 1, 1)), stream=stream)


def test_signed_remainder_keeps_c_semantics_without_migration_warning(requires_blackwell, capfd):
    """CuTeDSL 4.9 warns that signed ``%`` will take the divisor's sign; compiles pin the C
    remainder (the dividend's sign) on every version, beside the floored ``//``."""
    capfd.readouterr()
    fake = make_fake_compact_tensor(Int32, (2,), assumed_align=4)
    compiled = compile_tvm_ffi(SignedDivmod(), fake, Int32(0), Int32(1), name="signed_divmod")
    assert "DivmodMigration" not in capfd.readouterr().err
    out = torch.empty(2, dtype=torch.int32, device="cuda")
    for a, b in [(-7, 3), (7, -3), (-7, -3), (7, 3), (-6, 3)]:
        compiled(out, a, b)
        assert out.tolist() == [int(math.fmod(a, b)), a // b], (a, b)
