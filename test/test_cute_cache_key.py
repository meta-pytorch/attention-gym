"""The CuTeDSL cache namespace covers exactly a compile function's import closure."""

from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path

import pytest

from attn_gym._backends.cute import _key

MODULES = {
    "__init__.py": "",
    "entry.py": """
import pkg.absolute
import pkg.aliased as aliased
import pkg.sub.leaf
from . import relative
from pkg.lazymap import thing
from pkg.nested import VALUE
from pkg.reexport import NAME
from pkg.starred import *

def compile_fn():
    from pkg import local
    return "pkg.dynamic"
""",
    "absolute.py": "from pkg import cycle\n",
    "cycle.py": "from pkg import absolute\n",
    "relative.py": "VALUE = 1\n",
    "local.py": "VALUE = 1\n",
    "dynamic.py": "VALUE = 1\n",
    "unrelated.py": "VALUE = 1\n",
    # ``import pkg.sub.leaf`` runs this implicitly; its imports are not dependencies.
    "sub/__init__.py": "from pkg import unrelated\n",
    "sub/leaf.py": "from ..relative import VALUE\n",
    "reexport/__init__.py": "from .impl import NAME\n",
    "reexport/impl.py": "NAME = 1\n",
    "aliased.py": "VALUE = 1\n",
    "starred.py": "VALUE = 1\n",
    "nested/__init__.py": "from . import child\nfrom .. import upper\nVALUE = 1\n",
    "nested/child.py": "VALUE = 1\n",
    "upper.py": "VALUE = 1\n",
    # A lazy-export map names its owners only as strings (see attn_gym/linear/_lazy.py).
    "lazymap.py": """
EXPORTS = {"thing": "pkg.lazy_target"}

def __getattr__(name):
    import importlib
    if name not in EXPORTS:
        raise AttributeError(name)
    return getattr(importlib.import_module(EXPORTS[name]), name)
""",
    "lazy_target.py": "thing = 1\n",
}
CLOSURE = {name for name in MODULES if name not in {"sub/__init__.py", "unrelated.py"}}


@pytest.fixture
def package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "pkg"
    for name, source in MODULES.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(_key, "_PACKAGE_ROOT", root)
    _key._direct_imports.cache_clear()
    yield root
    _key._direct_imports.cache_clear()
    for name in [name for name in sys.modules if name == "pkg" or name.startswith("pkg.")]:
        del sys.modules[name]


def test_module_closure_follows_every_import_form(package: Path):
    """Every import form, including aliases, stars, relative imports inside ``__init__``, and
    string-named lazy exports; cycles terminate."""
    closure = _key.module_closure(package / "entry.py", package)
    assert {path.relative_to(package).as_posix() for path in closure} == CLOSURE


def test_source_fingerprint_tracks_only_the_import_closure(package: Path):
    """Editing a transitive dependency recompiles; editing an unrelated module does not."""
    pytest.importorskip("cutlass")
    compile_fn = importlib.import_module("pkg.entry").compile_fn

    def fingerprint() -> str:
        _key._direct_imports.cache_clear()
        return _key.source_fingerprint.__wrapped__(compile_fn)

    baseline = fingerprint()
    (package / "unrelated.py").write_text("VALUE = 2\n")
    assert fingerprint() == baseline
    (package / "reexport/impl.py").write_text("NAME = 2\n")
    assert fingerprint() != baseline


def test_kernel_constants_are_in_the_closure():
    """Regression: module constants inlined by a kernel once escaped the ``cute``-path hash."""
    pytest.importorskip("cutlass")
    from attn_gym.linear.kda.fwd.cute import gate_fwd

    closure = _key.module_closure(Path(gate_fwd.__file__))
    package = Path(_key.__file__).resolve().parents[2]
    assert package / "linear/kda/constants.py" in closure
    assert package / "linear/short_conv/cute.py" not in closure


def test_callers_outside_the_package_hash_their_package_imports(
    package: Path, tmp_path_factory: pytest.TempPathFactory
):
    """A compile function defined outside the package still keys on the package code it uses."""
    caller = tmp_path_factory.mktemp("outside") / "kernels.py"
    caller.write_text("from . import sibling\nimport pkg.sub.leaf\n")
    closure = set(_key.module_closure(caller))
    assert closure == {caller, package / "sub/leaf.py", package / "relative.py"}


def test_sources_modified_after_import_are_reported(
    package: Path, monkeypatch: pytest.MonkeyPatch
):
    """Only a file in the closure edited after import can disagree with the traced code."""
    compile_fn = importlib.import_module("pkg.entry").compile_fn
    snapshot = time.time()
    monkeypatch.setattr(_key, "_SOURCE_SNAPSHOT_TIME", snapshot)
    assert _key.sources_modified_since_import(compile_fn) == []
    for name in ("unrelated.py", "reexport/impl.py"):
        os.utime(package / name, (snapshot + 10, snapshot + 10))
    assert _key.sources_modified_since_import(compile_fn) == [package / "reexport/impl.py"]
