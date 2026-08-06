import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import modal

ROOT_PATH = Path(__file__).parent
PYTORCH_NIGHTLY_INDEX = "https://download.pytorch.org/whl/nightly/cu132"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", pre=True, index_url=PYTORCH_NIGHTLY_INDEX, force_build=True)
    .pip_install_from_pyproject(
        str(ROOT_PATH / "pyproject.toml"), optional_dependencies=["tests"], pre=True
    )
    .add_local_python_source("attn_gym")
    .add_local_dir(ROOT_PATH / "test", remote_path="/root/test")
)

app = modal.App("attention-gym-modal-tests", image=image)


def format_pytest_summary(report_path: Path) -> str:
    """Format a pytest JUnit report as a concise Markdown summary."""
    root = ET.parse(report_path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        return "## B200 pytest summary\n\nPytest did not produce a readable test suite."

    total = int(suite.attrib.get("tests", 0))
    failures = int(suite.attrib.get("failures", 0))
    errors = int(suite.attrib.get("errors", 0))
    skipped = int(suite.attrib.get("skipped", 0))
    passed = total - failures - errors - skipped
    duration = float(suite.attrib.get("time", 0))
    lines = [
        "## B200 pytest summary",
        "",
        (
            f"**{passed} passed, {failures} failed, {errors} errors, {skipped} skipped** "
            f"in {duration:.2f}s."
        ),
    ]

    failed_tests = []
    for test_case in suite.iter("testcase"):
        failure = test_case.find("failure")
        error = test_case.find("error")
        problem = failure if failure is not None else error
        if problem is None:
            continue
        test_name = test_case.attrib.get("name", "unknown test")
        details = (problem.text or problem.attrib.get("message", "No failure details."))[-4000:]
        failed_tests.append((test_name, details.strip()))

    if failed_tests:
        lines.extend(["", "### Failures"])
        for test_name, details in failed_tests:
            lines.extend(["", f"#### `{test_name}`", "", "```text", details, "```"])

    return "\n".join(lines) + "\n"


@app.function(gpu="B200", timeout=30 * 60)
def run_pytest() -> tuple[int, str]:
    """Run the repository test suite and return its exit code and summary."""
    report_path = Path("/tmp/pytest-report.xml")
    result = subprocess.run(
        ["python", "-m", "pytest", "test", "-ra", "--tb=short", f"--junitxml={report_path}"],
        cwd="/root",
        check=False,
    )
    return result.returncode, format_pytest_summary(report_path)


@app.local_entrypoint()
def main() -> None:
    """Run pytest remotely and publish its summary to GitHub Actions."""
    return_code, summary = run_pytest.remote()
    print(f"\n{summary}")
    if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary_path).open("a") as summary_file:
            summary_file.write(summary)
    raise SystemExit(return_code)
