"""Which CUDA devices run the SM100-family (tcgen05) CuTeDSL and warpdrive kernels."""

import functools
import re
from importlib.metadata import PackageNotFoundError, version

import torch

# Datacenter Blackwell (SM100, SM103) and Rubin (SM107) share the tcgen05 MMA, tensor memory,
# and TMA programming model that the CuTeDSL and warpdrive kernels are written against.
_SM100_FAMILY = frozenset(((10, 0), (10, 3), (10, 7)))
_RUBIN = (10, 7)
# CuTeDSL 4.8 added the sm_107a target; older versions cannot compile for Rubin.
_RUBIN_MIN_CUTEDSL = (4, 8)

SM100_KERNELS_REQUIREMENT = "an SM100-family GPU (SM100, SM103, or SM107 with CuTeDSL 4.8+)"


@functools.cache
def cutedsl_targets_rubin() -> bool:
    """Return whether the installed CuTeDSL can compile for SM107 (without importing it)."""
    try:
        installed = version("nvidia-cutlass-dsl")
    except PackageNotFoundError:
        return False
    release = re.match(r"(\d+)\.(\d+)", installed)
    return release is not None and tuple(map(int, release.groups())) >= _RUBIN_MIN_CUTEDSL


def runs_sm100_kernels(capability: tuple[int, int] | None) -> bool:
    """Return whether a CUDA capability runs the SM100-family kernels.

    Rubin qualifies only when the installed CuTeDSL can target it; otherwise it keeps the
    portable paths.
    """
    if capability == _RUBIN:
        return cutedsl_targets_rubin()
    return capability in _SM100_FAMILY


def cuda_runs_sm100_kernels(device: torch.device | int | None = None) -> bool:
    """Return whether CUDA is available and ``device`` runs the SM100-family kernels."""
    return torch.cuda.is_available() and runs_sm100_kernels(
        torch.cuda.get_device_capability(device)
    )
