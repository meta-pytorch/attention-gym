"""Layer-owned globaltimer buffers and host-side role reports."""

import json
import os
import pickle
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from attn_gym.warpdrive import StaticPersistent
from attn_gym.warpdrive.cache import compile_many, precompile_many
from attn_gym.warpdrive.examples.persistent_gemm import PersistentGemm, compile_persistent_gemm
from attn_gym.warpdrive.profiler.ranges import Range
from attn_gym.warpdrive.trace_session import owned_buffer, role_summary, trace_session


def test_role_summary_containment():
    ranges = [
        Range("load", 0, 100, (0,), 4),
        Range("load", 100, 300, (0,), 4),
        Range("ab.acquire", 10, 30, (0,), 4),
        Range("turn.turn", 110, 150, (0,), 4),
        Range("inner.sync", 120, 130, (0,), 4),
        Range("user", 50, 70, (0,), 4),
        Range("ab.wait", 10, 30, (1,), 4),  # unrelated CTA
        Range("ab.peek", 10, 30, (0,), 5),  # unrelated warp
        Range("ab.tail", 290, 310, (0,), 4),  # not contained
    ]
    report = role_summary(ranges)
    (row,) = report.roles
    assert (row.name, row.warp, row.items) == ("load", 4, 2)
    # inner.sync sits inside turn.turn: counted once, through the turn.
    assert (row.p50_ns, row.p90_ns, row.item_ns, row.wait_ns) == (150, 190, 300, 60)
    assert row.wait_share == 60 / 300
    assert row.work_share == 1 - 60 / 300
    assert "p90 ns" in str(report) and "20.0%" in str(report)
    op = SimpleNamespace(
        _collect=lambda: None,
        _ws_roles=[
            SimpleNamespace(name="load", first_warp=4),
            SimpleNamespace(name="idle", first_warp=6),
        ],
    )
    empty = role_summary([], op).roles
    assert all(r.items == r.wait_ns == r.item_ns == r.wait_share == 0 for r in empty)
    assert role_summary([]).roles == ()


def test_role_summary_multiple_ctas():
    ranges = [
        r
        for cta in range(64)
        for r in (
            Range("load", 0, 100, (cta,), 4),
            Range("ab.acquire", 10, 30, (cta,), 4),
            Range("load", 100, 300, (cta,), 4),
            Range("ab.wait", 110, 150, (cta,), 4),
        )
    ]
    (row,) = role_summary(ranges).roles
    assert (row.items, row.item_ns, row.wait_ns) == (128, 64 * 300, 64 * 60)
    assert row.wait_share == 0.2


def test_uncached_explicit_buffer_refreshes(requires_blackwell):
    from test_warpdrive_tracing import RingRoles, compile_traced_ring

    op = RingRoles().enable_trace("globaltimer", ctas=2)
    compile_traced_ring.__wrapped__(op)
    first = op._ws_trace_address
    op.enable_trace("globaltimer", ctas=3)
    compile_traced_ring.__wrapped__(op)
    assert not op._ws_trace_owned
    assert op._ws_trace_address != first
    assert op._ws_trace_address[-1] * 2 == first[-1] * 3


def test_untraced_key_and_ir_ignore_owned_buffer(monkeypatch, traced_compiles):
    op = PersistentGemm(schedule=StaticPersistent())
    key = op.__cute_cache_key__()
    before = str(compile_persistent_gemm.__wrapped__(op).ir_module)
    op.enable_trace(None, ctas=7, events_per_role=17)
    monkeypatch.setattr(
        sys.modules[owned_buffer.__module__],
        "owned_buffer",
        lambda op: pytest.fail("untraced build allocated a buffer"),
    )
    assert op.__cute_cache_key__() == key
    after = str(compile_persistent_gemm.__wrapped__(op).ir_module)
    assert before == after


def test_owned_gemm_capture(requires_blackwell):
    op = PersistentGemm(schedule=StaticPersistent()).enable_trace("globaltimer", ctas=2)
    fn = compile_persistent_gemm(op)
    prof = owned_buffer(op)
    prof.tensor.fill_(42)
    a = torch.randn(512, 64, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(512, 64, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(512, 512, dtype=torch.bfloat16, device="cuda")
    workspace, _ = op.work.launch_args(a.device)
    with trace_session(op, device="cuda") as session:
        assert torch.count_nonzero(session.prof.tensor).item() == 0
        fn(a, b, out, workspace, 2)
    torch.testing.assert_close(out, a @ b.T)
    report = role_summary(session.ranges, op)
    for row in report.roles:
        assert row.items == 8  # four M tiles x two N tiles, distributed over two CTAs
        items = [r for r in session.ranges if r.name == row.name and r.warp == row.warp]
        expected = sum(
            r.duration_ns
            for r in session.ranges
            if r.name.rpartition(".")[2] in {"acquire", "wait", "peek", "tail", "turn", "sync"}
            and any(
                i.cta == r.cta
                and i.warp == r.warp
                and i.start_ns <= r.start_ns <= r.end_ns <= i.end_ns
                for i in items
            )
        )
        assert row.wait_ns == expected
        assert row.wait_share == expected / row.item_ns
    # A fresh session reuses and clears the same storage.
    with trace_session(op) as again:
        assert again.prof.tensor.data_ptr() == prof.tensor.data_ptr()
    assert again.ranges == []
    with pytest.raises(ValueError, match="device must match"), trace_session(op, device="cpu"):
        pass


def test_cached_traced_build_after_untraced_compile(requires_blackwell):
    """An op compiled untraced, then switched to globaltimer, gets a cached traced build: its
    trace tags and buffer ownership come from that build, so trace_session works."""
    compile_persistent_gemm(PersistentGemm().enable_trace("globaltimer", ctas=1))  # warm
    op = PersistentGemm()
    compile_persistent_gemm(op)
    fn = compile_persistent_gemm(op.enable_trace("globaltimer", ctas=1))
    a = torch.randn(128, 64, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(128, 64, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(128, 128, dtype=torch.bfloat16, device="cuda")
    workspace, _ = op.work.launch_args(a.device)
    with trace_session(op) as session:
        fn(a, b, out, workspace, 1)
    torch.testing.assert_close(out, a @ b.T)
    assert {r.name for r in session.ranges} >= {"load", "mma", "epilogue"}


def test_owned_overflow(requires_blackwell):
    op = PersistentGemm().enable_trace("globaltimer", ctas=1, events_per_role=1)
    fn = compile_persistent_gemm(op)
    a = torch.ones(128, 64, dtype=torch.bfloat16, device="cuda")
    b = torch.ones(128, 64, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(128, 128, dtype=torch.bfloat16, device="cuda")
    workspace, _ = op.work.launch_args(a.device)
    with pytest.warns(UserWarning, match="filled their 1 records"), trace_session(op) as session:
        fn(a, b, out, workspace, 1)
    assert session.ranges


def test_uncached_recompile_refreshes_buffer(requires_blackwell):
    op = PersistentGemm().enable_trace("globaltimer", ctas=2)
    op.trace_protocol = True  # jit_cache bypasses key hashing
    first = str(compile_persistent_gemm(op).ir_module)
    address = op._ws_trace_address
    op.enable_trace("globaltimer", ctas=3)
    second = str(compile_persistent_gemm(op).ir_module)
    assert op._ws_trace_address != address
    assert op._ws_trace_address[-1] * 2 == address[-1] * 3
    assert first != second


def test_trace_pickle_and_parallel_rejection(requires_blackwell):
    op = PersistentGemm().enable_trace("globaltimer", ctas=2, events_per_role=17)
    restored = pickle.loads(pickle.dumps(op))
    assert restored._trace_settings() == op._trace_settings()
    with pytest.raises(RuntimeError, match="cannot use compile_many/precompile_many"):
        precompile_many(compile_persistent_gemm, [(restored,)], workers=1, timeout=60)
    # Warm parent-local entries need no worker and retain the launching process's address.
    fn = compile_persistent_gemm(restored)
    address = restored._ws_trace_address
    precompile_many(compile_persistent_gemm, [(restored,)], workers=1, timeout=60)
    assert compile_many(compile_persistent_gemm, [(restored,)], workers=1, timeout=60) == [fn]
    assert restored._ws_trace_address == address


def test_buffer_registry_devices(requires_blackwell):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two visible GPUs")
    keys = []
    for device in (0, 1):
        with torch.cuda.device(device):
            op = PersistentGemm().enable_trace("globaltimer", ctas=1)
            fn = compile_persistent_gemm(op)
            keys.append(op._trace_settings())
            a = torch.ones(128, 64, dtype=torch.bfloat16, device=f"cuda:{device}")
            b = torch.ones_like(a)
            out = torch.empty(128, 128, dtype=torch.bfloat16, device=a.device)
            workspace, _ = op.work.launch_args(a.device)
            with trace_session(op) as session:
                fn(a, b, out, workspace, 1)
            assert session.prof.tensor.device.index == device
            assert session.ranges
            torch.testing.assert_close(out, a @ b.T)
    assert keys[0][-1][0] != keys[1][-1][0]


def test_fresh_process_address_recompiles(requires_blackwell, tmp_path):
    env = dict(os.environ, WARPDRIVE_CACHE_DIR=str(tmp_path / "cache"))
    code = """
import json, sys, torch
from attn_gym.warpdrive.examples.persistent_gemm import PersistentGemm, compile_persistent_gemm
padding = torch.empty(int(sys.argv[1]), dtype=torch.int64, device="cuda")
op = PersistentGemm().enable_trace("globaltimer", ctas=1)
key = op._trace_settings()
compile_persistent_gemm(op)
print("RESULT " + json.dumps(key[-1]))
"""
    addresses = []
    for size in (1, 8_000_000):
        result = subprocess.run(
            [sys.executable, "-c", code, str(size)],
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        addresses.append(
            json.loads(
                next(
                    line.removeprefix("RESULT ")
                    for line in result.stdout.splitlines()
                    if line.startswith("RESULT ")
                )
            )
        )
    assert addresses[0][1] != addresses[1][1]
    # Distinct baked addresses require distinct persistent artifacts, not a disk hit.
    assert len(list((tmp_path / "cache").rglob("*.o"))) == 2
