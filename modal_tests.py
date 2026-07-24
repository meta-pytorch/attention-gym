import subprocess
from pathlib import Path

import modal

ROOT_PATH = Path(__file__).parent
PYTORCH_NIGHTLY_INDEX = "https://download.pytorch.org/whl/nightly/cu130"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "filelock",
        "typing-extensions",
        "setuptools<82",
        "sympy",
        "networkx",
        "jinja2",
        "fsspec",
        "numpy",
        "pillow",
        "spmd-types",
    )
    .pip_install("torch", pre=True, index_url=PYTORCH_NIGHTLY_INDEX, force_build=True)
    .pip_install("pytest", "jsonargparse", "docstring-parser")
    .add_local_python_source("attn_gym")
    .add_local_dir(ROOT_PATH / "test", remote_path="/root/test")
)

app = modal.App("attention-gym-modal-tests", image=image)


@app.function(gpu="B200", timeout=30 * 60)
def run_pytest() -> None:
    """Run the repository test suite on a single B200."""
    subprocess.run(["python", "-m", "pytest", "test"], check=True, cwd="/root")
