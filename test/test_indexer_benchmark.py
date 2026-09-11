"""The optional benchmark dependency must not block CLI help or hide installation guidance."""

import sys
from types import ModuleType

import pytest

from benchmarks.sparse import indexer_benchmark


@pytest.mark.parametrize("dependency", [None, ModuleType("transformer_nuggets.utils.benchmark")])
def test_benchmark_help_without_compatible_nuggets(monkeypatch, capsys, dependency):
    """Help remains available when Nuggets is absent or lacks graph sample statistics."""
    monkeypatch.setitem(sys.modules, "transformer_nuggets.utils.benchmark", dependency)
    monkeypatch.setattr(sys, "argv", ["indexer_benchmark.py", "--help"])
    with pytest.raises(SystemExit) as exc:
        indexer_benchmark.main()
    assert exc.value.code == 0
    assert "uv pip install" in capsys.readouterr().out


@pytest.mark.parametrize("dependency", [None, ModuleType("transformer_nuggets.utils.benchmark")])
def test_benchmark_reports_missing_or_old_nuggets(monkeypatch, dependency):
    """An unavailable statistics helper reports a compatible installation before GPU setup."""
    monkeypatch.setitem(sys.modules, "transformer_nuggets.utils.benchmark", dependency)
    monkeypatch.setattr(sys, "argv", ["indexer_benchmark.py"])
    with pytest.raises(SystemExit, match="Install the compatible revision"):
        indexer_benchmark.main()
