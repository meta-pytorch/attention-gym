"""The Inkling 2 Mini checkpoint of the Inkling tests: a local snapshot, never downloaded."""

import json
from pathlib import Path

import pytest

MODEL = "thinkingmachines/Inkling-2-Mini"
# hf_quant_config.json declares the expert numerics: without it the loader falls back to HF's.
WEIGHTS = ("*.json", "hf_quant_config.json", "model*.safetensors")
TOKENIZER = ("*.jinja", "tiktoken/*")


def checkpoint_path(patterns: tuple[str, ...] = WEIGHTS) -> str:
    """The local snapshot directory of ``MODEL``'s files matching ``patterns``; skips the test
    unless the snapshot holds them in the local HF cache and the optional readers of the weights
    (safetensors) and the tokenizer (transformers) asked for are installed. Checked before any
    weight is loaded."""
    hub = pytest.importorskip("huggingface_hub")
    from huggingface_hub.errors import LocalEntryNotFoundError

    if "model*.safetensors" in patterns:
        pytest.importorskip("safetensors")
    if set(TOKENIZER) & set(patterns):
        pytest.importorskip("transformers")
    try:
        path = Path(
            hub.snapshot_download(MODEL, allow_patterns=list(patterns), local_files_only=True)
        )
    except LocalEntryNotFoundError as error:
        pytest.skip(f"{MODEL} is not in the local HF cache: {error}")
    # A snapshot can exist with only some files (say config.json): check that every pattern
    # matches a file and that the weight shards the index names are there, so a partial cache
    # skips instead of failing in the loader.
    missing = [p for p in patterns if not any(f.exists() for f in path.glob(p))]
    index = path / "model.safetensors.index.json"
    if "model*.safetensors" in patterns and index.exists():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        missing += [s for s in shards if not (path / s).exists()]
    if missing:
        pytest.skip(f"{MODEL}'s local snapshot lacks {missing}")
    return str(path)
