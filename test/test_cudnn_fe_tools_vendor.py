"""The cudnn-frontend vendoring tool relocates imports, marks files, and reproduces PR A."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tools.cudnn_fe import closure, vendor

KERNEL = "cudnn.linear_attention.frost.kernel."
LICENSE_HEADER = (
    "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.\n"
    "# SPDX-License-Identifier: Apache-2.0\n"
)


def test_relocate_moves_frost_imports_into_the_package() -> None:
    source = (
        "from cudnn.frost.tile_dsl.barrier import MBarrier\n"
        "from cudnn.frost.tile_dsl import mma\n"
        "from cudnn.frost.buffers import DeviceView, probe\n"
        "    from cudnn.frost.device import current_device\n"
        "from ..common.host import get_dtype\n"
        "x = 'cudnn.frost.tile_dsl.barrier'\n"
    )
    assert vendor.relocate(source, "k") == (
        "from ..tile_dsl.barrier import MBarrier\n"
        "from ..tile_dsl import mma\n"
        "from .._compat import DeviceView, probe\n"
        "    from .._compat import current_device\n"
        "from ..common.host import get_dtype\n"
        "x = 'cudnn.frost.tile_dsl.barrier'\n"
    )


@pytest.mark.parametrize(
    "line",
    [
        "import cudnn",
        "from cudnn.frost.workspace import Workspace",
        "    import cudnn.frost.occupancy",
    ],
)
def test_relocate_rejects_imports_it_cannot_relocate(line: str) -> None:
    with pytest.raises(ValueError, match="unhandled cudnn import"):
        vendor.relocate(f"import math\n{line}\n", "kernel/x.py")


def test_notice_follows_the_license_block_and_keeps_the_source() -> None:
    body = '"""Docstring."""\n\n# a later comment\nimport math\n'
    marked = vendor.add_notice(LICENSE_HEADER + body, "v9.9.9")
    notice = (
        "#\n# Modified by Attention Gym in 2026: vendored from cudnn-frontend v9.9.9; imports "
        "relocated into\n# attn_gym.linear._delta_rule.cudnn_fe.\n"
    )
    assert marked == LICENSE_HEADER + notice + body
    assert vendor.add_notice(body, "v9.9.9") == notice + body


def test_prune_cuts_only_the_gdp_fork_branch() -> None:
    source = (
        "from . import gdn_bprop_f16, gdp_bprop_v64_f16\n"
        "def f():\n"
        "    if cutlass.const_expr(compact_qdo):\n"
        "        gdp_bprop_v64_f16.build_descs_body(\n"
        "            a,\n"
        "        )\n"
        "    else:\n"
        "        gdn_bprop_f16.build_descs_body(a)\n"
    )
    pruned = vendor.prune_gdp_fork(source)
    assert "gdp_bprop_v64_f16" not in pruned
    assert pruned.startswith("from . import gdn_bprop_f16\n")
    assert "raise NotImplementedError" in pruned
    assert pruned.endswith("    else:\n        gdn_bprop_f16.build_descs_body(a)\n")


class FakeUpstream:
    """Upstream modules as a dict of module name -> source."""

    def __init__(self, sources: dict[str, str]):
        self.sources = sources

    def module_file(self, module: str) -> str | None:
        return module if module in self.sources else None

    def show(self, module: str) -> str:
        return self.sources[module]


@pytest.mark.parametrize(
    "root",
    [
        f'import importlib\nhelper = importlib.import_module("{KERNEL}helper")\n',
        'import importlib\nhelper = importlib.import_module(".helper", __package__)\n',
        f'helper = __import__("{KERNEL}helper", fromlist=["run"])\n',
    ],
)
def test_closure_follows_literal_dynamic_imports(root: str) -> None:
    up = FakeUpstream({KERNEL + "root": root, KERNEL + "helper": "def run(): pass\n"})
    assert sorted(closure.compute(up, [KERNEL + "root"]).files) == [
        "kernel/helper.py",
        "kernel/root.py",
    ]


def test_closure_rejects_non_literal_dynamic_imports() -> None:
    up = FakeUpstream({KERNEL + "root": "import importlib\nm = importlib.import_module(name)\n"})
    with pytest.raises(ValueError, match="non-literal"):
        closure.compute(up, [KERNEL + "root"])


GDP_TREE = {
    KERNEL + "root": "from . import gdn_chain_prologue_f16\n",
    KERNEL + "gdn_chain_prologue_f16": "from . import gdn_bprop_f16, gdp_bprop_v64_f16\n",
    KERNEL + "gdn_bprop_f16": "X = 1\n",
    KERNEL + "gdp_bprop_v64_f16": "from . import gdp_bprop_v64_config\n",
    KERNEL + "gdp_bprop_v64_config": "VALUE = 1\n",
}


def test_prune_cuts_only_the_documented_edge() -> None:
    cl = closure.compute(FakeUpstream(GDP_TREE), [KERNEL + "root"], prune=True)
    assert sorted(cl.files) == [
        "kernel/gdn_bprop_f16.py",
        "kernel/gdn_chain_prologue_f16.py",
        "kernel/root.py",
    ]
    assert [e.target for e in cl.cut] == [KERNEL + "gdp_bprop_v64_f16"]


def test_prune_fails_when_gdp_stays_reachable() -> None:
    tree = dict(GDP_TREE)
    tree[KERNEL + "root"] = "from . import gdn_chain_prologue_f16, scalar_path\n"
    tree[KERNEL + "scalar_path"] = "from . import gdp_bprop_v64_f16\n"
    with pytest.raises(ValueError, match="scalar_path"):
        closure.compute(FakeUpstream(tree), [KERNEL + "root"], prune=True)


@pytest.fixture
def fake_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A checkout whose package holds a sentinel kernel; the tool resolves paths inside it."""
    package = tmp_path / closure.PACKAGE_RELPATH
    (package / "kernel").mkdir(parents=True)
    (package / "kernel/gdn_prefill_f16.py").write_text("SENTINEL\n")
    monkeypatch.setattr(closure, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(closure, "package_dir", lambda: package)
    return tmp_path


def test_default_destination_refuses_a_symlink_into_the_package(fake_checkout: Path) -> None:
    link = fake_checkout / "agent_space/cudnn_fe_vendor/v1.30.0"
    link.parent.mkdir(parents=True)
    link.symlink_to(fake_checkout / closure.PACKAGE_RELPATH, target_is_directory=True)
    with pytest.raises(SystemExit, match="symlink"):
        vendor.default_dest("v1.30.0")
    package = fake_checkout / closure.PACKAGE_RELPATH
    assert (package / "kernel/gdn_prefill_f16.py").read_text() == "SENTINEL\n"


def test_default_destination_refuses_a_symlinked_parent(fake_checkout: Path) -> None:
    (fake_checkout / "agent_space").mkdir()
    (fake_checkout / "agent_space/cudnn_fe_vendor").symlink_to(
        fake_checkout / closure.PACKAGE_RELPATH, target_is_directory=True
    )
    with pytest.raises(SystemExit, match="symlink"):
        vendor.default_dest("kernel")


def test_explicit_destination_refuses_symlinks(fake_checkout: Path, tmp_path: Path) -> None:
    package = fake_checkout / closure.PACKAGE_RELPATH
    link = tmp_path / "out"
    link.symlink_to(package, target_is_directory=True)
    with pytest.raises(SystemExit, match="symlink"):
        vendor.check_dest(link)
    real = tmp_path / "real"
    (real / "kernel").mkdir(parents=True)
    (real / "kernel/gdn_prefill_f16.py").symlink_to(package / "kernel/gdn_prefill_f16.py")
    with pytest.raises(SystemExit, match="symlink"):
        vendor.check_dest(real)


def _upstream_clone() -> Path | None:
    candidates = [
        os.environ.get("CUDNN_FE_UPSTREAM"),
        closure.repo_root() / "agent_space/upstream/cudnn-frontend",
        closure.repo_root().parent / "attention-gym/agent_space/upstream/cudnn-frontend",
    ]
    for candidate in candidates:
        if candidate and (Path(candidate) / ".git").exists():
            return Path(candidate)
    return None


@pytest.fixture
def clone() -> Path:
    clone = _upstream_clone()
    if clone is None:
        pytest.skip("set CUDNN_FE_UPSTREAM to a cudnn-frontend clone")
    if vendor.find_drop_commit() is None:
        pytest.skip(f"no {vendor.DROP_SUBJECT!r} commit in this history")
    return clone


def test_verify_reproduces_the_verbatim_v130_drop(
    clone: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = vendor.main(["--rev", "v1.30.0", "--upstream", str(clone), "--verify", "auto"])
    assert rc == 0, capsys.readouterr().out


@pytest.mark.parametrize("rel", ["_compat.py", "LICENSE.txt", "kernel/gdn_prefill_f16.py"])
def test_verify_fails_on_any_corrupted_generated_file(
    rel: str, clone: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    package = repo / closure.PACKAGE_RELPATH
    vendor.vendor(closure.Upstream(clone, "v1.30.0"), package, "engines")
    (package / rel).write_bytes((package / rel).read_bytes() + b"# corrupted\n")
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "drop"], check=True)
    monkeypatch.setattr(closure, "repo_root", lambda: repo)
    assert vendor.verify(closure.Upstream(clone, "v1.30.0"), "HEAD") == 1
