"""Every synthetic layer bug in ``runtime_mutants.py`` fails (or hangs) its listed test."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("cutlass.cute")

from warpdrive_fuzz import runtime_mutants

pytestmark = pytest.mark.fuzz

TEST_ROOT = Path(__file__).resolve().parents[1]
# Allow cold compilation before classifying a timeout as a caught hang.
# On GB300, failing children exit within ~15 s with 15 compiling concurrently.
HANG_TIMEOUT_S = 90.0


@pytest.mark.parametrize("mutant", runtime_mutants.MUTANTS)
def test_mutant_is_caught(requires_blackwell, tmp_path, mutant):
    _, file, test = runtime_mutants.MUTANTS[mutant]
    env = {
        **os.environ,
        "WDFUZZ_MUTANT": mutant,
        # Cached kernels were compiled against the real layer.
        "WARPDRIVE_CACHE_DIR": str(tmp_path),
        "CUTE_DSL_DISABLE_FILE_CACHING": "1",
        "PYTHONPATH": os.pathsep.join(
            filter(None, [str(TEST_ROOT), os.environ.get("PYTHONPATH")])
        ),
    }
    cmd = [
        *(sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"),
        *("-p", "warpdrive_fuzz.runtime_mutants", f"{TEST_ROOT}/warpdrive/{file}::{test}"),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=TEST_ROOT.parent,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=HANG_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        return  # hung: caught
    text = out.decode(errors="replace")
    assert proc.returncode == pytest.ExitCode.TESTS_FAILED, f"{mutant} survived {test}:\n{text}"
