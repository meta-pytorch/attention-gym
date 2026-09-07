"""Portable checks for the shallow example layout and its relocated entrypoints."""

import re
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
        assert relative.parts[0] in {"flex_attention", "linear", "sparse"}


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
