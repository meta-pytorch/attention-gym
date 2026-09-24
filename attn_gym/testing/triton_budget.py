"""Compile Triton launches for another GPU target and report their shared-memory needs.

Kernels are compiled for the requested target but never launched, so a GB300 host can check
which configurations an A10G (SM86) would select and whether they fit its per-block limit.
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


@dataclass(frozen=True)
class CompiledConfig:
    """Resources of one compiled kernel specialization."""

    num_warps: int
    num_stages: int
    shared: int


@dataclass
class Launch:
    """One host launch: a fixed config, or every config its autotuner may benchmark."""

    kernel: str
    configs: list[CompiledConfig] = field(default_factory=list)

    @property
    def min_shared(self) -> int:
        return min(config.shared for config in self.configs)


class EmulatedDeviceProperties:
    """Real device properties with an emulated capability and shared-memory limit."""

    def __init__(self, properties, capability: tuple[int, int], max_shared_mem: int):
        self._properties = properties
        self.major, self.minor = capability
        self.shared_memory_per_block_optin = max_shared_mem

    def __getattr__(self, name: str):
        return getattr(self._properties, name)


@contextmanager
def compile_only_for_target(
    capability: tuple[int, int], max_shared_mem: int
) -> Iterator[list[Launch]]:
    """Record Triton launches as compile-only builds for ``capability``.

    Within the context, ``torch.cuda.get_device_capability`` and ``get_device_properties``
    report the emulated capability and opt-in shared memory per block, so host dispatch takes
    that device's path. Patched launches never run: their outputs stay unwritten and only the
    recorded compile metadata is meaningful. Autotuned launches record every config left after
    pruning, since the autotuner may pick any of them that fits.
    """
    target = GPUTarget("cuda", capability[0] * 10 + capability[1], 32)
    launches: list[Launch] = []
    active_autotune: list[Launch] = []
    replaced_caches: dict[JITFunction, dict] = {}
    original_jit_run = JITFunction.run
    real_get_device_properties = torch.cuda.get_device_properties

    def device_properties(device=None):
        return EmulatedDeviceProperties(
            real_get_device_properties(device), capability, max_shared_mem
        )

    def jit_run(self, *args, grid, warmup, **kwargs):
        # Compiled kernels and the target are cached per device; keep emulated builds apart.
        if self not in replaced_caches:
            replaced_caches[self] = self.device_caches
            self.device_caches = defaultdict(self.create_binder)
        kernel = original_jit_run(self, *args, grid=grid, warmup=True, **kwargs)
        compiled = CompiledConfig(
            kernel.metadata.num_warps, kernel.metadata.num_stages, kernel.metadata.shared
        )
        if active_autotune:
            active_autotune[0].configs.append(compiled)
        else:
            launches.append(Launch(self.__name__, [compiled]))
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
            mock.patch.object(JITFunction, "run", jit_run),
            mock.patch.object(Autotuner, "run", autotune_run),
            mock.patch.object(
                triton.runtime.driver.active, "get_current_target", return_value=target
            ),
            mock.patch.object(torch.cuda, "get_device_capability", return_value=capability),
            mock.patch.object(torch.cuda, "get_device_properties", device_properties),
        ):
            yield launches
    finally:
        for function, caches in replaced_caches.items():
            function.device_caches = caches
