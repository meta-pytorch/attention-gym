"""warpdrive is self-contained: the rest of attn_gym may import it, never the reverse."""

import ast
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import attn_gym.warpdrive

WARPDRIVE = Path(attn_gym.warpdrive.__file__).parent


def _outside_references(source: Path) -> list[str]:
    """Imports (and import-shaped strings) of attn_gym modules outside warpdrive.

    A relative import resolves against the source's package, so ``from .. import linear`` counts.
    """
    package = ".".join(("attn_gym", *source.parent.relative_to(WARPDRIVE.parent).parts))
    names = []
    for node in ast.walk(ast.parse(source.read_text(), filename=str(source))):
        match node:
            case ast.Import(names=aliases):
                names += [alias.name for alias in aliases]
            case ast.ImportFrom(module=module, level=level, names=aliases):
                base = importlib.util.resolve_name("." * level + (module or ""), package)
                names += [f"{base}.{alias.name}" for alias in aliases]
            case ast.Constant(value=str(value)) if re.fullmatch(r"attn_gym(\.\w+)+", value):
                names.append(value)
    return [
        name
        for name in names
        if name.split(".")[0] == "attn_gym" and name.split(".")[:2] != ["attn_gym", "warpdrive"]
    ]


def test_warpdrive_imports_nothing_else_from_attn_gym():
    offenders = {
        str(path.relative_to(WARPDRIVE)): refs
        for path in sorted(WARPDRIVE.rglob("*.py"))
        if (refs := _outside_references(path))
    }
    assert offenders == {}


def test_importing_attn_gym_loads_neither_the_layer_nor_cutlass():
    probe = (
        "import sys, attn_gym, attn_gym.warpdrive.cache\n"
        "print(sorted(m for m in sys.modules if m == 'cutlass' or (\n"
        "    m.startswith('attn_gym.warpdrive.') and not m.startswith('attn_gym.warpdrive.cache'))))"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", probe], check=True, capture_output=True, text=True
    ).stdout
    assert loaded.strip() == "[]"
