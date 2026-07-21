import subprocess
import sys
from pathlib import Path

import pytest


BENCHMARK_DIRECTORY = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "sparse"
    / "benchmark_compressed_sparse_attention"
)
BENCHMARK_SCRIPTS = (
    "benchmark_compressed_sparse_attention_cute.py",
    "benchmark_compressed_sparse_attention_cute_forward_backward.py",
    "benchmark_compressed_sparse_attention_cute_tflops.py",
    "benchmark_compressed_sparse_attention_triton.py",
    "benchmark_compressed_sparse_attention_triton_tflops.py",
)


@pytest.mark.parametrize("script_name", BENCHMARK_SCRIPTS)
def test_benchmark_supports_direct_execution(script_name):
    result = subprocess.run(
        [sys.executable, str(BENCHMARK_DIRECTORY / script_name), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
