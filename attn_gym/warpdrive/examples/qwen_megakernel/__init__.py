"""The Qwen3 decode megakernel (see ``runner``). Modules:

- ``core``: scratch layout, tagged words, the CUDA-core GEMV engine, ``QwenDecodeMegakernel`` and its
  compile function;
- ``tc``: the tcgen05 engine and streamed inputs;
- ``schedule``: the host's instruction table, row-cost model and CTA assignment;
- ``runner``: the ``QwenMegakernel`` runner;
- ``ref``: the PyTorch reference and checkpoint loading;
- ``trace``, ``viz``: globaltimer decoding and the schedule page;
- ``bench``, ``bench_batch``, ``chat``: benchmarks and a streaming chat demo.

The runner's public names are re-exported here, loaded on first use: importing one module (say
``ref``) does not load the runner and every kernel.
"""

import importlib

_RUNNER = "attn_gym.warpdrive.examples.qwen_megakernel.runner"


def __getattr__(name: str):
    runner = importlib.import_module(_RUNNER)
    if name in runner.__all__:
        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(importlib.import_module(_RUNNER).__all__))
