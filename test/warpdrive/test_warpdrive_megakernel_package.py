"""The import boundary of the shared megakernel package (``examples/megakernel``). A kernel's
compile key hashes every module it imports (``module_closure``), so a model's kernels import the
package and their own model's modules only: editing one model recompiles no other model's
kernels, and the package's device code imports no model."""

from pathlib import Path

import pytest

from attn_gym.warpdrive.cache._key import _direct_imports, module_closure

EXAMPLES = Path(__file__).resolve().parents[2] / "attn_gym" / "warpdrive" / "examples"
PACKAGE = EXAMPLES / "megakernel"
MODELS = sorted(EXAMPLES.glob("*_megakernel"))
MODULES = [path for model in MODELS for path in sorted(model.glob("*.py"))]
# Modules that define compiled kernels.
KERNELS = [path for path in MODULES if "@jit_cache" in path.read_text()]
DRIVER = PACKAGE / "chat.py"  # names every family's chat module (``FAMILIES``)
HOST = {DRIVER, PACKAGE / "launch.py"}  # the package's host-only modules
# Modules each kernel's key must hash: a computed import would drop them silently.
REQUIRED = {
    "qwen_megakernel/core.py": {
        "megakernel/attention.py",
        "megakernel/gemv.py",
        "megakernel/ring.py",
        "megakernel/table.py",
        "megakernel/tagged.py",
        "qwen_megakernel/ref.py",
    },
    "qwen_megakernel/tc.py": {"megakernel/attention.py", "qwen_megakernel/core.py"},
}


def relative(paths) -> set[str]:
    return {str(path.relative_to(EXAMPLES)) for path in paths if path.is_relative_to(EXAMPLES)}


def closure_short_of_driver(module: Path) -> set[Path]:
    """``module_closure`` without the chat driver's imports (every family's chat module)."""
    seen, pending = set(), [module.resolve()]
    while pending:
        current = pending.pop()
        if current not in seen:
            seen.add(current)
            if current != DRIVER:
                pending.extend(_direct_imports(current) - seen)
    return seen


@pytest.mark.parametrize("module", MODULES, ids=lambda path: str(path.relative_to(EXAMPLES)))
def test_models_import_no_other_model(module):
    """A model's modules reach other models only through the chat driver."""
    others = {path for path in closure_short_of_driver(module) for model in MODELS
              if model != module.parent and path.is_relative_to(model)}  # fmt: skip
    assert not others, f"{module.name} hashes {sorted(relative(others))}"


@pytest.mark.parametrize("module", KERNELS, ids=lambda path: str(path.relative_to(EXAMPLES)))
def test_kernels_hash_no_host_module(module):
    assert not set(module_closure(module)) & HOST


@pytest.mark.parametrize("module", sorted(REQUIRED))
def test_kernels_hash_what_they_trace(module):
    closure = relative(module_closure(EXAMPLES / module))
    assert REQUIRED[module] <= closure, sorted(REQUIRED[module] - closure)


# The chat driver names every family module (``FAMILIES``).
@pytest.mark.parametrize(
    "module", sorted(p.stem for p in PACKAGE.glob("*.py") if p.stem != "chat")
)
def test_package_imports_no_model(module):
    closure = module_closure(PACKAGE / f"{module}.py")
    assert all(p.is_relative_to(PACKAGE) or not p.is_relative_to(EXAMPLES) for p in closure)
