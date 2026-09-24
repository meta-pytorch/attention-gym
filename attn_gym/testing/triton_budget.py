"""Check which Triton launches another GPU would select, and their shared-memory needs.

``compile_only_for_target`` compiles launches for the emulated target without running them, so
a GB300 host can check which configurations an A10G (SM86) would select and whether they fit
its per-block limit.
"""

from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from unittest import mock

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.runtime.autotuner import Autotuner
from triton.runtime.jit import JITFunction


@dataclass
class Launch:
    """One host launch: a fixed config, or every config its autotuner may benchmark."""

    kernel: str
    shared: list[int] = field(default_factory=list)


class EmulatedDeviceProperties:
    """Real device properties with an emulated capability and shared-memory limit."""

    def __init__(self, properties, capability: tuple[int, int], max_shared_mem: int):
        self._properties = properties
        self.major, self.minor = capability
        self.shared_memory_per_block_optin = max_shared_mem

    def __getattr__(self, name: str):
        return getattr(self._properties, name)


@contextmanager
def emulate_device_dispatch(capability: tuple[int, int], max_shared_mem: int) -> Iterator[None]:
    """Make host dispatch see ``capability`` and ``max_shared_mem`` while compiling locally.

    ``torch.cuda.get_device_capability`` and ``get_device_properties`` report the emulated
    device. Triton derives its compile target from the capability, so it is pinned to the
    local GPU; kernels still run here.
    """
    local_target = triton.runtime.driver.active.get_current_target()
    real_get_device_properties = torch.cuda.get_device_properties

    def device_properties(device=None):
        return EmulatedDeviceProperties(
            real_get_device_properties(device), capability, max_shared_mem
        )

    with (
        mock.patch.object(
            triton.runtime.driver.active, "get_current_target", return_value=local_target
        ),
        mock.patch.object(torch.cuda, "get_device_capability", return_value=capability),
        mock.patch.object(torch.cuda, "get_device_properties", device_properties),
    ):
        yield


@contextmanager
def compile_only_for_target(
    capability: tuple[int, int], max_shared_mem: int
) -> Iterator[list[Launch]]:
    """Record Triton launches as compile-only builds for ``capability``.

    Host dispatch is emulated as in ``emulate_device_dispatch``, but kernels compile for the
    emulated target and never run: outputs stay unwritten and only the recorded shared memory
    is meaningful. Autotuned launches record every config left after pruning, since the
    autotuner may pick any of them that fits.
    """
    target = GPUTarget("cuda", capability[0] * 10 + capability[1], 32)
    launches: list[Launch] = []
    active_autotune: list[Launch] = []
    replaced_caches: dict[JITFunction, dict] = {}
    original_jit_run = JITFunction.run

    def jit_run(self, *args, grid, warmup, **kwargs):
        # Compiled kernels and the target are cached per device; keep emulated builds apart.
        if self not in replaced_caches:
            replaced_caches[self] = self.device_caches
            self.device_caches = defaultdict(self.create_binder)
        kernel = original_jit_run(self, *args, grid=grid, warmup=True, **kwargs)
        if active_autotune:
            active_autotune[0].shared.append(kernel.metadata.shared)
        else:
            launches.append(Launch(self.__name__, [kernel.metadata.shared]))
        return kernel

    def autotune_run(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        launch = Launch(self.base_fn.__name__)
        active_autotune.append(launch)
        try:
            for config in self.prune_configs(kwargs):
                self.fn.run(*args, **kwargs, **config.all_kwargs())
        finally:
            active_autotune.clear()
            self.nargs = None
        launches.append(launch)

    try:
        with (
            emulate_device_dispatch(capability, max_shared_mem),
            mock.patch.object(JITFunction, "run", jit_run),
            mock.patch.object(Autotuner, "run", autotune_run),
            mock.patch.object(
                triton.runtime.driver.active, "get_current_target", return_value=target
            ),
        ):
            yield launches
    finally:
        for function, caches in replaced_caches.items():
            function.device_caches = caches
