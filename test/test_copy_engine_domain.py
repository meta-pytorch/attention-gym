"""CPU-only checks of NVLink-domain detection against a fake NVML."""

import socket
import sys
import types

import pytest
import torch
import torch.distributed as dist

from attn_gym.distributed import nvlink_domain, same_nvlink_domain

COMPLETED, IN_PROGRESS = 3, 2


class _NVMLError(Exception):
    pass


def _fake_nvml(state: int, status: int = 0, uuid: bytes = bytes(range(16)), clique: int = 7):
    """An ``nvidia-ml-py``-shaped module whose fabric query fills the caller's v2 struct."""
    module = types.ModuleType("pynvml")
    real = pytest.importorskip("pynvml")
    module.NVMLError = _NVMLError
    module.NVML_GPU_FABRIC_STATE_COMPLETED = COMPLETED
    module.c_nvmlGpuFabricInfo_v2_t = real.c_nvmlGpuFabricInfo_v2_t
    module.nvmlGpuFabricInfo_v2 = real.nvmlGpuFabricInfo_v2
    module.nvmlInit = lambda: None
    module.nvmlDeviceGetHandleByUUID = lambda uuid_str: uuid_str

    def query(handle, info_ref):
        info = info_ref._obj
        info.state, info.status, info.cliqueId = state, status, clique
        info.clusterUuid[:] = list(uuid)

    module.nvmlDeviceGetGpuFabricInfoV = query
    return module


@pytest.fixture(autouse=True)
def _fake_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda index: types.SimpleNamespace(uuid="abc")
    )


def test_completed_fabric_is_identified_by_cluster_and_clique(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", _fake_nvml(COMPLETED))
    assert nvlink_domain() == f"fabric:{bytes(range(16)).hex()}:7"


@pytest.mark.parametrize(("state", "status"), [(IN_PROGRESS, 0), (COMPLETED, 1)])
def test_unready_fabric_falls_back_to_host(monkeypatch, state, status) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", _fake_nvml(state, status))
    assert nvlink_domain() == f"host:{socket.gethostname()}"


def test_failed_query_or_missing_pynvml_falls_back_to_host(monkeypatch) -> None:
    module = _fake_nvml(COMPLETED)

    def unsupported(handle, info_ref):
        raise _NVMLError("not supported")

    module.nvmlDeviceGetGpuFabricInfoV = unsupported
    monkeypatch.setitem(sys.modules, "pynvml", module)
    assert nvlink_domain() == f"host:{socket.gethostname()}"
    monkeypatch.setitem(sys.modules, "pynvml", None)  # `import pynvml` raises ImportError
    assert nvlink_domain() == f"host:{socket.gethostname()}"


@pytest.mark.parametrize(
    ("domains", "expected"),
    [
        (["fabric:aa:32766", "fabric:aa:32766"], True),
        # The MAST failure: same default clique, different fabrics (one per node).
        (["fabric:aa:32766", "fabric:bb:32766"], False),
        (["host:a", "host:b"], False),
    ],
)
def test_same_nvlink_domain_compares_every_rank(monkeypatch, domains, expected) -> None:
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: len(domains))

    def all_gather_object(out, obj, group=None):
        out[:] = domains

    monkeypatch.setattr(dist, "all_gather_object", all_gather_object)
    assert same_nvlink_domain(group=object()) is expected
