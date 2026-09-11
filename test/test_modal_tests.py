"""Exercise the Modal CI failure budget and reporting without allocating a GPU."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("modal", reason="Modal runner tests require the optional Modal client")


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the CI runner without building images or depending on wheel-mode environment state."""
    import modal

    monkeypatch.delenv("ATTN_GYM_WHEEL", raising=False)
    monkeypatch.setattr(modal, "is_local", lambda: False)
    spec = spec_from_file_location(
        "_modal_test_runner", Path(__file__).resolve().parents[1] / "modal_tests.py"
    )
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("extras", [["tests"], ["cudnn", "dev"]])
def test_image_ships_runner_and_only_overrides_fa4_for_tests(
    runner: ModuleType, extras: list[str]
):
    """Keep runner regression tests mounted and the two dependency environments separate."""
    image = Mock()
    image.pip_install_from_pyproject.return_value = image
    image.apt_install.return_value = image
    image.pip_install_from_requirements.return_value = image
    image.add_local_python_source.return_value = image
    image.add_local_dir.return_value = image
    image.add_local_file.return_value = image

    assert runner.configure_local_image(image, extras) is image

    image.add_local_file.assert_any_call(
        runner.ROOT_PATH / "modal_tests.py", remote_path="/root/modal_tests.py"
    )
    if "tests" in extras:
        image.apt_install.assert_called_once_with("git")
        image.pip_install_from_requirements.assert_called_once_with(
            str(runner.ROOT_PATH / "requirements-test.txt"), pre=True
        )
    else:
        image.pip_install_from_requirements.assert_not_called()


def test_failure_report_includes_assertions_and_errors(runner: ModuleType, tmp_path: Path):
    """Preserve both assertion failures and setup errors in the GitHub summary."""
    report = tmp_path / "report.xml"
    report.write_text(
        '<testsuites><testsuite tests="4" failures="1" errors="1" skipped="1" time="2.5">'
        '<testcase name="passes"/>'
        '<testcase name="asserts"><failure>unsupported sink</failure></testcase>'
        '<testcase name="errors"><error message="setup failed"/></testcase>'
        '<testcase name="skips"><skipped/></testcase>'
        "</testsuite></testsuites>"
    )
    summary = runner.format_pytest_summary(report)
    assert "1 passed, 1 failed, 1 errors, 1 skipped" in summary
    assert "unsupported sink" in summary
    assert "setup failed" in summary


def test_pytest_failure_budget_and_missing_report(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Remove stale reports and expose signal termination without hiding the pytest command."""
    report = tmp_path / "report.xml"
    report.write_text("stale report")
    monkeypatch.setattr(runner, "Path", lambda name: report)
    run = Mock(return_value=SimpleNamespace(returncode=-9))
    monkeypatch.setattr(runner.subprocess, "run", run)

    return_code, summary = runner.execute_pytest(["test"], "report", "Suite")

    assert return_code == -9
    assert "exited with code -9 before writing a report" in summary
    assert not report.exists()
    command = run.call_args.args[0]
    assert "--maxfail=5" in command
    assert "--instafail" in command
    assert "-vra" in command
    assert command[command.index("-n") + 1] == "4"


@pytest.mark.parametrize("preflight_code", [0, 1, 5])
def test_dependency_preflight_gates_full_suite(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch, preflight_code: int
):
    """A failed or uncollected dependency smoke never launches the expensive matrix."""
    execute = Mock(side_effect=[(preflight_code, "preflight"), (0, "suite")])
    monkeypatch.setattr(runner, "execute_pytest", execute)

    return_code, summary = runner.run_pytest.local()

    assert return_code == preflight_code
    assert "preflight" in summary
    first_call = execute.call_args_list[0]
    assert first_call.args[0] == [
        "test/test_selected_attention_cute.py::test_cute_sink_dependency_smoke"
    ]
    assert first_call.kwargs == {"workers": 0}
    assert execute.call_count == (1 if preflight_code else 2)
    if not preflight_code:
        assert "suite" in summary


@pytest.mark.parametrize("first_code", [0, 1, -9])
def test_main_publishes_each_suite_before_allocating_the_next(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, first_code: int
):
    """Publish completed evidence even if the next remote call times out; stop after failure."""
    summary_file = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))
    first = Mock(return_value=(first_code, "ordinary summary\n"))
    second = Mock(side_effect=TimeoutError("remote deadline"))
    monkeypatch.setattr(runner, "run_pytest", SimpleNamespace(remote=first))
    monkeypatch.setattr(runner, "run_cudnn_pytest", SimpleNamespace(remote=second))

    with pytest.raises(SystemExit if first_code else TimeoutError):
        runner.main()

    assert summary_file.read_text() == "ordinary summary\n"
    assert second.call_count == (0 if first_code else 1)
