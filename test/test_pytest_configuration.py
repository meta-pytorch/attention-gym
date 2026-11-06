"""Guard the worker isolation used by tests that share fixed GPU resources."""

import os
import subprocess
import sys
from pathlib import Path


def test_xdist_group_keeps_shared_resource_cases_on_one_worker(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "pytest.ini").write_text("[pytest]\n")
    (suite / "test_grouped.py").write_text(
        "from pathlib import Path\n"
        "import pytest\n\n"
        "@pytest.mark.xdist_group('shared-resource')\n"
        "@pytest.mark.parametrize('case', range(4))\n"
        "def test_case(case, worker_id):\n"
        "    Path(f'worker-{case}.txt').write_text(worker_id)\n"
    )
    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "xdist.plugin",
            "-n",
            "2",
            "--dist=loadgroup",
            "-q",
        ],
        cwd=suite,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    workers = [(suite / f"worker-{case}.txt").read_text() for case in range(4)]
    assert len(set(workers)) == 1, f"xdist_group was ignored: {workers}"
