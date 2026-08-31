import importlib
import os
import subprocess
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import modal

ROOT_PATH = Path(__file__).parent
PYTORCH_NIGHTLY_INDEX = "https://download.pytorch.org/whl/nightly/cu132"
WHEEL_PATH = Path(os.environ["ATTN_GYM_WHEEL"]).resolve() if os.getenv("ATTN_GYM_WHEEL") else None
REMOTE_WHEEL_PATH = f"/tmp/{WHEEL_PATH.name}" if WHEEL_PATH else None
# Rebuild once per UTC day without disabling Modal's cache for every commit.
NIGHTLY_CACHE_DATE = datetime.now(UTC).date().isoformat()

base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"PYTORCH_NIGHTLY_CACHE_DATE": NIGHTLY_CACHE_DATE})
    .pip_install("torch", pre=True, index_url=PYTORCH_NIGHTLY_INDEX)
)
image = base_image
mega_image = base_image


def configure_local_image(
    source_image: modal.Image, optional_dependencies: list[str]
) -> modal.Image:
    """Install one compatible dependency set and attach the local test sources."""
    configured = source_image.pip_install_from_pyproject(
        str(ROOT_PATH / "pyproject.toml"), optional_dependencies=optional_dependencies, pre=True
    )
    if WHEEL_PATH:
        if not WHEEL_PATH.is_file() or WHEEL_PATH.suffix != ".whl":
            raise ValueError(f"ATTN_GYM_WHEEL must name an existing wheel: {WHEEL_PATH}")
        configured = (
            configured.env({"ATTN_GYM_WHEEL": REMOTE_WHEEL_PATH})
            .add_local_file(WHEEL_PATH, REMOTE_WHEEL_PATH, copy=True)
            .run_commands(
                f"python -m pip install --no-deps {REMOTE_WHEEL_PATH}",
                "python -m pip check",
            )
        )
    else:
        configured = configured.add_local_python_source("attn_gym")
    return configured.add_local_dir(ROOT_PATH / "test", remote_path="/root/test").add_local_dir(
        ROOT_PATH / "examples", remote_path="/root/examples"
    )


if modal.is_local():
    image = configure_local_image(image, ["tests"])
    mega_image = configure_local_image(mega_image.pip_install("pytest-xdist"), ["mega", "dev"])

app = modal.App("attention-gym-modal-tests", image=image)


def format_pytest_summary(report_path: Path, title: str = "B200 pytest summary") -> str:
    """Format a pytest JUnit report as a concise Markdown summary."""
    root = ET.parse(report_path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        return f"## {title}\n\nPytest did not produce a readable test suite."

    total = int(suite.attrib.get("tests", 0))
    failures = int(suite.attrib.get("failures", 0))
    errors = int(suite.attrib.get("errors", 0))
    skipped = int(suite.attrib.get("skipped", 0))
    passed = total - failures - errors - skipped
    duration = float(suite.attrib.get("time", 0))
    lines = [
        f"## {title}",
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


def verify_wheel_install() -> None:
    """Require wheel-mode imports to resolve outside the mounted test checkout."""
    if WHEEL_PATH is None:
        return

    package_path = Path(importlib.import_module("attn_gym").__file__).resolve()
    if "site-packages" not in package_path.parts:
        raise RuntimeError(f"attn_gym did not import from site-packages: {package_path}")
    if Path("/root/attn_gym").exists():
        raise RuntimeError("checkout source unexpectedly exists at /root/attn_gym")
    print(f"attn_gym imported from {package_path}", flush=True)


def execute_pytest(test_paths: list[str], report_name: str, title: str) -> tuple[int, str]:
    """Run one isolated dependency-compatible pytest suite."""
    verify_wheel_install()
    report_path = Path(f"/tmp/{report_name}.xml")
    report_path.unlink(missing_ok=True)
    result = subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            *test_paths,
            "-n",
            "4",
            "--dist=worksteal",
            "-ra",
            "--tb=short",
            "--durations=50",
            f"--junitxml={report_path}",
        ],
        cwd="/root",
        check=False,
    )
    summary = (
        format_pytest_summary(report_path, title)
        if report_path.is_file()
        else f"## {title}\n\nPytest exited with code {result.returncode} before writing a report.\n"
    )
    return result.returncode, summary


@app.function(gpu="B200", timeout=30 * 60)
def run_pytest() -> tuple[int, str]:
    """Run the ordinary repository suite with the FlashAttention-compatible test extra."""
    return execute_pytest(["test"], "pytest-report", "B200 pytest summary")


@app.function(image=mega_image, gpu="B200", timeout=30 * 60)
def run_mega_pytest() -> tuple[int, str]:
    """Run the CuTeDSL 4.7+ GDN/KDA Mega suites in their compatible environment."""
    return execute_pytest(
        ["test/gdn/mega", "test/kda/mega"],
        "mega-pytest-report",
        "B200 Mega pytest summary",
    )


@app.local_entrypoint()
def main() -> None:
    """Run both B200 suites and publish their summaries to GitHub Actions."""
    results = (run_pytest.remote(), run_mega_pytest.remote())
    combined_summary = "\n".join(summary for _return_code, summary in results)
    print(f"\n{combined_summary}")
    if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary_path).open("a") as summary_file:
            summary_file.write(combined_summary)
    if any(return_code != 0 for return_code, _summary in results):
        raise SystemExit(1)
