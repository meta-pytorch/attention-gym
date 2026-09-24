import contextlib
import importlib
import os
import subprocess
import tarfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import modal

ROOT_PATH = Path(__file__).parent
PYTORCH_NIGHTLY_INDEX = "https://download.pytorch.org/whl/nightly/cu132"
WHEEL_PATH = Path(os.environ["ATTN_GYM_WHEEL"]).resolve() if os.getenv("ATTN_GYM_WHEEL") else None
REMOTE_WHEEL_PATH = f"/tmp/{WHEEL_PATH.name}" if WHEEL_PATH else None
# Rebuild once per UTC day without disabling Modal's cache for every commit.
NIGHTLY_CACHE_DATE = datetime.now(UTC).date().isoformat()

# Compile artifacts persist across runs as one tarball per suite on a Volume. Tests read and write
# a local copy: pytest workers must not share cache files over the Volume's network filesystem.
CACHE_VOLUME = modal.Volume.from_name("attention-gym-compile-cache", create_if_missing=True)
CACHE_VOLUME_PATH = Path("/compile-cache-volume")
COMPILE_CACHE_PATH = Path("/tmp/compile-cache")
# Every cache is content-keyed, so pruning costs only recompiles. CuTe and Inductor keys include
# the torch version, and the nightly changes daily, so older entries are rarely reusable.
CACHE_MAX_AGE_SECONDS = 3 * 24 * 60 * 60
CACHE_ENV = {
    "ATTN_GYM_CUTE_CACHE_DIR": f"{COMPILE_CACHE_PATH}/attn_gym_cute",
    "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": f"{COMPILE_CACHE_PATH}/flash_attn_cute",
    "TRITON_CACHE_DIR": f"{COMPILE_CACHE_PATH}/triton",
    "TORCHINDUCTOR_CACHE_DIR": f"{COMPILE_CACHE_PATH}/inductor",
}

base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"PYTORCH_NIGHTLY_CACHE_DATE": NIGHTLY_CACHE_DATE, **CACHE_ENV})
    .pip_install("torch", pre=True, index_url=PYTORCH_NIGHTLY_INDEX)
    .pip_install("pytest-instafail")
)
image = base_image
cudnn_image = base_image


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
    # test_examples_layout checks README.md and docs/ against examples/, so ship them too.
    return (
        configured.add_local_dir(ROOT_PATH / "test", remote_path="/root/test")
        .add_local_dir(ROOT_PATH / "examples", remote_path="/root/examples")
        .add_local_dir(ROOT_PATH / "benchmarks", remote_path="/root/benchmarks")
        .add_local_dir(ROOT_PATH / "docs", remote_path="/root/docs")
        .add_local_file(ROOT_PATH / "README.md", remote_path="/root/README.md")
        .add_local_file(ROOT_PATH / "modal_tests.py", remote_path="/root/modal_tests.py")
    )


if modal.is_local():
    image = configure_local_image(image, ["tests"])
    cudnn_image = configure_local_image(cudnn_image.pip_install("pytest-xdist"), ["cudnn", "dev"])

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


def execute_pytest(
    test_paths: list[str], report_path: Path, title: str, *, workers: int = 4
) -> tuple[int, str]:
    """Run one isolated dependency-compatible pytest suite."""
    verify_wheel_install()
    report_path.unlink(missing_ok=True)
    result = subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            *test_paths,
            "-n",
            str(workers),
            "--dist=worksteal",
            "-vra",
            "--tb=short",
            "--instafail",
            "--maxfail=5",
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


def restore_compile_cache(archive: Path, cache_dir: Path) -> None:
    """Unpack a previous run's compile artifacts; a missing archive is a cold start."""
    if not archive.is_file():
        print(f"no compile cache at {archive}; compiling from scratch", flush=True)
        return
    start = time.perf_counter()
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(cache_dir, filter="data")
    size_mb = archive.stat().st_size / 2**20
    print(f"restored {size_mb:.0f} MiB compile cache in {time.perf_counter() - start:.1f}s")


def save_compile_cache(cache_dir: Path, archive: Path, max_age_seconds: float) -> None:
    """Archive artifacts written or restored within ``max_age_seconds``, replacing ``archive``.

    Restoring preserves each file's original mtime, so an entry ages from when it was compiled.
    """
    if not cache_dir.is_dir():
        return
    cutoff = time.time() - max_age_seconds
    staging = archive.with_name(f"{archive.name}.partial")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(staging, "w") as tar:
        for path in sorted(cache_dir.rglob("*")):
            if path.is_file() and not path.is_symlink() and path.stat().st_mtime >= cutoff:
                tar.add(path, arcname=path.relative_to(cache_dir), recursive=False)
    staging.replace(archive)
    print(f"saved {archive.stat().st_size / 2**20:.0f} MiB compile cache to {archive}")


@contextlib.contextmanager
def persistent_compile_cache(suite: str) -> Iterator[None]:
    """Restore ``suite``'s compile cache from the Volume and save it back, even after failures."""
    archive = CACHE_VOLUME_PATH / f"{suite}.tar"
    restore_compile_cache(archive, COMPILE_CACHE_PATH)
    try:
        yield
    finally:
        save_compile_cache(COMPILE_CACHE_PATH, archive, CACHE_MAX_AGE_SECONDS)
        CACHE_VOLUME.commit()


# Four compile-heavy pytest workers should not depend on spare host CPU capacity.
@app.function(gpu="B200", cpu=4.0, timeout=30 * 60, volumes={CACHE_VOLUME_PATH: CACHE_VOLUME})
def run_pytest() -> tuple[int, str]:
    """Check FA4 sink support before running the ordinary repository suite."""
    with persistent_compile_cache("main"):
        return_code, preflight_summary = execute_pytest(
            ["test/test_gather_attn_cute.py::test_cute_dependency_smoke[forced-sink]"],
            Path("/tmp/pytest-preflight.xml"),
            "B200 FA4 dependency preflight",
            workers=0,
        )
        if return_code:
            return return_code, preflight_summary
        return_code, summary = execute_pytest(
            ["test"], Path("/tmp/pytest-report.xml"), "B200 pytest summary"
        )
    return return_code, f"{preflight_summary}\n{summary}"


@app.function(
    image=cudnn_image,
    gpu="B200",
    cpu=4.0,
    timeout=30 * 60,
    volumes={CACHE_VOLUME_PATH: CACHE_VOLUME},
)
def run_cudnn_pytest() -> tuple[int, str]:
    """Run the CuTeDSL 4.7+ GDN/KDA cuDNN suites in their compatible environment."""
    # The cuDNN image pins a different CuTeDSL, so it keeps a separate archive.
    with persistent_compile_cache("cudnn"):
        return execute_pytest(
            ["test/gdn/cudnn", "test/kda/cudnn"],
            Path("/tmp/cudnn-pytest-report.xml"),
            "B200 cuDNN pytest summary",
        )


@app.local_entrypoint()
def main() -> None:
    """Publish each suite immediately and stop allocating GPUs after a failed suite."""
    for suite in (run_pytest, run_cudnn_pytest):
        return_code, summary = suite.remote()
        print(f"\n{summary}", flush=True)
        if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
            with Path(summary_path).open("a") as summary_file:
                summary_file.write(summary)
        if return_code != 0:
            raise SystemExit(1)
