"""Which CUDA capabilities run the SM100-family (tcgen05) kernels."""

import pytest

from attn_gym._backends import arch


@pytest.fixture
def fresh_cutedsl_probe():
    arch.cutedsl_targets_rubin.cache_clear()
    yield
    arch.cutedsl_targets_rubin.cache_clear()


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        (None, False),
        ((9, 0), False),
        ((10, 0), True),
        ((10, 3), True),
        ((11, 0), False),
        ((12, 0), False),
    ],
)
def test_blackwell_runs_sm100_kernels(capability, expected):
    assert arch.runs_sm100_kernels(capability) is expected


@pytest.mark.usefixtures("fresh_cutedsl_probe")
@pytest.mark.parametrize(
    ("installed", "expected"),
    [
        ("4.6.2", False),
        ("4.7.0rc1", False),
        ("4.8rc1", True),
        ("4.8.0", True),
        ("4.10.0", True),
    ],
)
def test_rubin_runs_sm100_kernels_only_with_a_cutedsl_that_targets_it(
    monkeypatch, installed, expected
):
    """CuTeDSL before 4.8 cannot compile sm_107a, so Rubin keeps the portable paths."""
    monkeypatch.setattr(arch, "version", lambda _package: installed)
    assert arch.runs_sm100_kernels((10, 7)) is expected


@pytest.mark.usefixtures("fresh_cutedsl_probe")
def test_rubin_keeps_portable_paths_without_cutedsl(monkeypatch):
    def missing(package):
        raise arch.PackageNotFoundError(package)

    monkeypatch.setattr(arch, "version", missing)
    assert not arch.runs_sm100_kernels((10, 7))
