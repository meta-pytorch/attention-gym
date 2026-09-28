"""Shared plumbing of the cuDNN maintenance tools: logged subprocesses, tree selection and the
GDN/KDA workloads.

Every tool runs as ``python -m tools.cudnn_fe.<tool>`` from the repository root (the checkout
that holds the tools). A *tree* is another checkout whose ``attn_gym`` package is under test:
``--tree DIR`` (or ``[LABEL=]DIR`` where labels are reported) names the directory that contains
``attn_gym/``. A process that imports from a tree calls :func:`activate_tree` before importing
``attn_gym`` (or torch) and :func:`check_tree` afterwards; without ``--tree`` the importable
installation is used. Trees are never activated through ``PYTHONPATH``: ``python -m`` puts the
working directory ahead of it, so the live checkout would win silently.

GPU reservations belong to the caller (``gpu-run --timeout 900 auto -- ...``). Tools that run
many children may claim a GPU per child with :data:`GPU_RUN_NO_WAIT`, which fails fast with
exit :data:`GPU_BUSY` instead of queueing inside the harness.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
GPU_RUN_NO_WAIT = ("gpu-run", "--no-wait", "auto", "--")
GPU_BUSY = 75  # gpu-run: every GPU is reserved
TIMEOUT_EXIT = 124  # coreutils timeout: the command was terminated


def run_logged(
    cmd: list[str],
    log: Path,
    *,
    timeout: int,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    gpu_run: bool = False,
) -> tuple[int, str]:
    """Run ``cmd`` under ``timeout -k 10 <timeout>`` (optionally behind a no-wait ``gpu-run``
    claim), append the command, combined output and exit status to ``log`` and return
    ``(exit status, output)``."""
    cmd = ["timeout", "-k", "10", str(timeout), *cmd]
    if gpu_run:
        cmd = [*GPU_RUN_NO_WAIT, *cmd]
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log.write_text(f"$ {shlex.join(cmd)}\n{proc.stdout}\nEXIT_STATUS={proc.returncode}\n")
    return proc.returncode, proc.stdout


def describe_exit(rc: int) -> str:
    if rc == GPU_BUSY:
        return "GPU busy (gpu-run exit 75)"
    if rc in (TIMEOUT_EXIT, 137):
        return f"timeout (rc={rc})"
    return f"rc={rc}"


def parse_tree(spec: str, default_label: str) -> tuple[str, Path]:
    """``[LABEL=]PATH`` -> ``(label, resolved path)``."""
    label, sep, path = spec.partition("=")
    if not sep:
        label, path = default_label, spec
    return label, Path(path).resolve()


def activate_tree(tree: Path | None) -> None:
    """Put ``tree`` ahead of the installed ``attn_gym``; call before importing it."""
    if tree is not None:
        sys.path.insert(0, str(tree.resolve()))


def check_tree(tree: Path | None) -> Path:
    """Return the imported ``attn_gym`` package directory, asserting it lives under ``tree``."""
    import attn_gym

    package = Path(attn_gym.__file__).resolve().parent
    if tree is not None and not package.is_relative_to(tree.resolve()):
        raise SystemExit(f"attn_gym resolved to {package}, not to --tree {tree}")
    return package


def git_head(tree: Path) -> str:
    out = subprocess.run(
        ["git", "-C", str(tree), "rev-parse", "--short", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    return out or "n/a"


def timestamp() -> str:
    return time.strftime("%F %T")


# -------------------------------------------------------------------------------- workloads

D = 128
KINDS = ("gdn", "kda")


@dataclass(frozen=True)
class Workload:
    """One packed GDN/KDA problem: sequence lengths (zeros are empty intervals), query/key heads
    and value heads, ``d_k = d_v = D``."""

    kind: str
    lengths: tuple[int, ...]
    key_heads: int
    value_heads: int

    @property
    def tokens(self) -> int:
        return sum(self.lengths)

    @property
    def sequences(self) -> int:
        return len(self.lengths)


@dataclass
class Operands:
    """Batch-first ``(1, T, H, D)`` operands of ``chunk_gdn``/``chunk_kda``."""

    q: Any
    k: Any
    v: Any
    g: Any
    beta: Any
    cu_seqlens: Any
    state: Any = None

    @property
    def leaves(self) -> list[Any]:
        return [
            t for t in (self.q, self.k, self.v, self.g, self.beta, self.state) if t is not None
        ]


def cu_seqlens(lengths: tuple[int, ...], device: Any = "cuda") -> Any:
    import torch

    offsets = [0]
    for n in lengths:
        offsets.append(offsets[-1] + n)
    return torch.tensor(offsets, dtype=torch.int32, device=device)


def op_inputs(
    work: Workload,
    *,
    seed: int,
    dtype: Any = None,
    state: bool = False,
    device: Any = "cuda",
) -> Operands:
    """Deterministic operands: unit-norm ``q``/``k``, ``v`` in ``dtype`` (bf16 by default), the
    log gate (GDN per value head, KDA per channel, both negative), ``beta`` in ``(0, 1)`` and,
    with ``state``, an initial state per sequence."""
    import torch
    import torch.nn.functional as F

    dtype = dtype or torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, device=device, generator=generator)

    t, hk, hv = work.tokens, work.key_heads, work.value_heads
    q = F.normalize(randn(1, t, hk, D), dim=-1).to(dtype)
    k = F.normalize(randn(1, t, hk, D), dim=-1).to(dtype)
    v = randn(1, t, hv, D).to(dtype)
    if work.kind == "gdn":
        g = -F.softplus(randn(1, t, hv))
    else:
        g = torch.rand(1, t, hv, D, device=device, generator=generator) * -0.049 - 0.001
    beta = torch.rand(1, t, hv, device=device, generator=generator)
    initial = randn(work.sequences, hv, D, D) * 0.01 if state else None
    return Operands(q, k, v, g, beta, cu_seqlens(work.lengths, device), initial)


def chunk_op(kind: str):
    from attn_gym.linear import chunk_gdn, chunk_kda

    return {"gdn": chunk_gdn, "kda": chunk_kda}[kind]


def paged_chunk_op(kind: str):
    from attn_gym.linear import paged_chunk_gdn, paged_chunk_kda

    return {"gdn": paged_chunk_gdn, "kda": paged_chunk_kda}[kind]
