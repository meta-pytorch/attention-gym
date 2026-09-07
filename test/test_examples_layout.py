"""Portable checks for the shallow example layout and its relocated entrypoints."""

import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from examples.flex_attention import paged_attention, paged_attention_model, paged_attention_utils

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = REPO_ROOT / "examples"


def test_examples_have_one_concept_directory() -> None:
    """Keep Python examples and notebooks directly inside their high-level group."""
    sources = sorted(EXAMPLES_ROOT.rglob("*.py")) + sorted(EXAMPLES_ROOT.rglob("*.ipynb"))
    for source in sources:
        relative = source.relative_to(EXAMPLES_ROOT)
        if relative == Path("__init__.py"):
            continue
        assert len(relative.parts) == 2, f"example must be one directory deep: {relative}"


def test_relocated_paged_recipe_reserves_and_releases_cpu_pages() -> None:
    """The renamed helpers import the same cache implementation and keep its behavior."""
    assert paged_attention_utils.PagedAttention is paged_attention.PagedAttention
    assert issubclass(paged_attention_model.PagedAttentionLayer, torch.nn.Module)
    cache = paged_attention.PagedAttention(4, 2, 2, device="cpu")
    paged_attention_utils.batch_reserve(cache, torch.tensor([3, 0]))
    torch.testing.assert_close(cache.capacity, torch.tensor([4, 0]))
    torch.testing.assert_close(cache.page_table[0, :2], torch.tensor([1, 0]))
    assert len(cache.empty_pages) == 2
    cache.erase(torch.tensor([0]))
    assert len(cache.empty_pages) == 4
    assert cache.capacity[0].item() == 0


def test_paged_throughput_imports_as_package() -> None:
    """The throughput benchmark resolves its siblings under the package import path."""
    pytest.importorskip("datasets")
    from examples.flex_attention import paged_attention_throughput

    assert paged_attention_throughput.PagedAttention is paged_attention.PagedAttention


def test_paged_latency_script_runs_without_checkout_on_import_path(tmp_path: Path) -> None:
    """Direct ``python <script>`` execution must not depend on an editable-install path hook."""
    script = EXAMPLES_ROOT / "flex_attention" / "paged_attention_latency.py"
    # Emulate ``python script.py``: only the script directory leads sys.path, and the editable
    # install's checkout-root entry is removed so ``import examples`` cannot resolve.
    bootstrap = (
        "import runpy, sys\n"
        f"sys.path = [{str(script.parent)!r}] + "
        f"[p for p in sys.path[1:] if p != {str(REPO_ROOT)!r}]\n"
        f"sys.argv = [{str(script)!r}, '--help']\n"
        f"runpy.run_path({str(script)!r}, run_name='__main__')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", bootstrap],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize(
    "document",
    [
        REPO_ROOT / "README.md",
        EXAMPLES_ROOT / "README.md",
        *sorted((REPO_ROOT / "docs").rglob("*.md")),
    ],
    ids=lambda document: str(document.relative_to(REPO_ROOT)),
)
def test_documented_example_paths_exist(document: Path) -> None:
    """Keep commands, links, and snippet source paths aligned with file moves."""
    text = document.read_text()
    for example in re.findall(r"examples/[\w./-]+\.(?:py|ipynb)\b", text):
        assert (REPO_ROOT / example).is_file(), f"{document}: missing {example}"
    for example, section in re.findall(r'--8<-- "(examples/[^":]+):([^"\n]+)"', text):
        source = (REPO_ROOT / example).read_text()
        assert f"[start:{section}]" in source, f"{document}: missing {example}:{section}"
        assert f"[end:{section}]" in source, f"{document}: missing {example}:{section}"
