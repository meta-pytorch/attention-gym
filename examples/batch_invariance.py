"""Example script to probe batch invariance properties of FlexAttention.

This script exercises a collection of representative FlexAttention shapes and checks
that running them as one big batch produces equivalent outputs and gradients to
running the same examples in smaller sub-batches. It is useful when debugging cases
where heuristics or compilation caches introduce batch-size–dependent behavior.

Usage:
    python batch_invariance.py --device cuda --backend inductor

The script will print a table of max deviations per split and flag any failures.
Use --no-grads to skip gradient checks when only forwards are needed.
Use --dynamic false to disable dynamic shapes when debugging recompilation issues.
Use --mask-mod/--score-mod to exercise different mask or score mods (e.g. --mask-mod sliding_window:1024).
Use --permutations to control how many random batch permutations are checked for invariance.
"""

from __future__ import annotations

import contextlib
import random
import warnings
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Literal

import torch
from tabulate import tabulate
from torch.nn.attention import flex_attention as flex_mod
from jsonargparse import CLI

from attn_gym.masks import causal_mask, generate_prefix_lm_mask, generate_sliding_window
from attn_gym.masks.document_mask import generate_doc_mask_mod, length_to_offsets
from attn_gym.mods import generate_alibi_bias, generate_tanh_softcap

# Suppress noisy warnings
warnings.filterwarnings("ignore", message=".*TF32.*")
warnings.filterwarnings("ignore", message=".*_maybe_guard_rel.*")

# this is required for redcuctions in bwd to be deterministic
torch.use_deterministic_algorithms(True)


@dataclass(frozen=True)
class BatchInvarianceConfig:
    name: str
    B: int
    Hq: int
    Hkv: int
    Q: int
    KV: int
    Dqk: int
    Dv: int
    mask_mod: flex_mod._mask_mod_signature = causal_mask
    score_mod: flex_mod._score_mod_signature = None


DEFAULT_CONFIGS: Sequence[BatchInvarianceConfig] = (
    BatchInvarianceConfig("standard", B=4, Hq=16, Hkv=16, Q=2048, KV=2048, Dqk=128, Dv=128),
    BatchInvarianceConfig("decode", B=4, Hq=8, Hkv=8, Q=1, KV=2048, Dqk=128, Dv=128),
    BatchInvarianceConfig("gqa", B=4, Hq=32, Hkv=8, Q=1, KV=4096, Dqk=128, Dv=128),
    BatchInvarianceConfig("long_context", B=2, Hq=8, Hkv=8, Q=4096, KV=4096, Dqk=128, Dv=128),
)


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_COMPILED_BLOCK_MASK_FNS: Dict[str, Callable] = {}


def _get_compiled_block_mask_fn(backend: str) -> Callable:
    key = backend or "inductor"
    fn = _COMPILED_BLOCK_MASK_FNS.get(key)
    if fn is None:
        fn = torch.compile(
            flex_mod.create_block_mask,
            backend=key,
            fullgraph=True,
            dynamic=True,
        )
        _COMPILED_BLOCK_MASK_FNS[key] = fn
    return fn


def random_document_lengths(total_length: int, num_documents: int, seed: int) -> List[int]:
    if num_documents <= 0:
        raise ValueError("num_documents must be positive")
    if num_documents > total_length:
        raise ValueError("num_documents cannot exceed total sequence length")

    lengths = [1] * num_documents
    rng = random.Random(seed)
    remaining = total_length - num_documents
    for _ in range(remaining):
        idx = rng.randrange(num_documents)
        lengths[idx] += 1
    return lengths


def sieve_primes_upto(limit: int) -> List[int]:
    if limit < 2:
        return []
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[:2] = b"\x00\x00"
    for p in range(2, int(limit**0.5) + 1):
        if sieve[p]:
            step = p
            start = p * p
            sieve[start : limit + 1 : step] = b"\x00" * (((limit - start) // step) + 1)
    return [i for i in range(2, limit + 1) if sieve[i]]


def prime_document_lengths(total_length: int, num_documents: int, seed: int) -> List[int]:
    if num_documents <= 0:
        raise ValueError("num_documents must be positive")
    if total_length < 2 * num_documents:
        raise ValueError("Total length too small to distribute prime lengths")

    primes = sieve_primes_upto(total_length)
    if not primes:
        raise ValueError("No primes available for the requested total length")
    prime_set = set(primes)
    offset = seed % len(primes)

    def iter_candidates(max_val: int):
        for p in primes[offset:]:
            if p > max_val:
                break
            yield p
        for p in primes[:offset]:
            if p > max_val:
                break
            yield p

    @lru_cache(None)
    def dfs(index: int, remaining: int) -> Optional[Tuple[int, ...]]:
        if index == num_documents - 1:
            return (remaining,) if remaining in prime_set else None
        min_remaining = 2 * (num_documents - index - 1)
        max_val = remaining - min_remaining
        if max_val < 2:
            return None
        for prime_value in iter_candidates(max_val):
            result = dfs(index + 1, remaining - prime_value)
            if result is not None:
                return (prime_value,) + result
        return None

    combination = dfs(0, total_length)
    if combination is None:
        raise ValueError(
            "Unable to decompose total length into the requested number of prime segments"
        )
    return list(combination)


def parse_spec(spec: str) -> Tuple[str, List[str]]:
    name, rest = (spec.split(":", 1) + [""])[:2]
    params = [item.strip() for item in rest.split(",") if item.strip()] if rest else []
    return name.strip(), params


def convert_arg(value: str):
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        return int(value, 0)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lower = value.lower()
        if lower in {"true", "1", "yes", "y"}:
            return True
        if lower in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"Cannot interpret {value!r} as boolean")


def make_mask_builder(
    name: str, params: Sequence[str]
) -> Callable[[BatchInvarianceConfig, torch.device], flex_mod._mask_mod_signature]:
    key = name.lower()
    args = [convert_arg(p) for p in params]

    if key in {"config", "default"}:
        return lambda cfg, device: cfg.mask_mod
    if key in {"none", "null"}:
        return lambda cfg, device: None
    if key == "causal":
        return lambda cfg, device: causal_mask
    if key == "noop":
        return lambda cfg, device: flex_mod.noop_mask
    if key == "sliding_window":
        if not args:
            raise ValueError("sliding_window requires window size, e.g. sliding_window:1024")
        window = int(args[0])
        return lambda cfg, device, window=window: generate_sliding_window(window)
    if key == "prefix_lm":
        if not args:
            raise ValueError("prefix_lm requires prefix length, e.g. prefix_lm:512")
        prefix_length = int(args[0])
        return lambda cfg, device, prefix_length=prefix_length: generate_prefix_lm_mask(
            prefix_length
        )
    if key == "document":
        num_docs: Optional[int] = None
        seed = 0
        length_mode = "random"

        for arg in args:
            if isinstance(arg, int) and num_docs is None:
                num_docs = arg
            elif isinstance(arg, str):
                lower = arg.lower()
                if lower == "prime":
                    length_mode = "prime"
                elif lower.startswith("seed="):
                    seed = int(lower.split("=", 1)[1], 0)
                else:
                    raise ValueError("Unrecognized document mask parameter: '{}'".format(arg))
            else:
                raise ValueError(f"Cannot interpret document mask parameter: {arg!r}")

        if num_docs is None:
            num_docs = 4

        def build(
            cfg: BatchInvarianceConfig, device: torch.device
        ) -> flex_mod._mask_mod_signature:
            total_length = max(cfg.Q, cfg.KV)
            doc_count = min(num_docs, total_length)
            if doc_count <= 0:
                raise ValueError("Document count must be positive after adjustment")
            if length_mode == "prime":
                try:
                    lengths = prime_document_lengths(total_length, doc_count, seed)
                except ValueError as exc:
                    raise ValueError(
                        f"Failed to generate prime document lengths for total={total_length}, docs={doc_count}: {exc}"
                    ) from exc
            else:
                lengths = random_document_lengths(total_length, doc_count, seed)

            offsets = length_to_offsets(lengths, device)
            base_mask = cfg.mask_mod if cfg.mask_mod is not None else causal_mask
            mask = generate_doc_mask_mod(base_mask, offsets)
            setattr(mask, "_doc_lengths", tuple(int(length) for length in lengths))
            return mask

        return build

    raise ValueError(f"Unknown mask_mod spec '{name}'")


def make_score_builder(
    name: str, params: Sequence[str]
) -> Callable[[BatchInvarianceConfig], flex_mod._score_mod_signature]:
    key = name.lower()
    args = [convert_arg(p) for p in params]

    if key in {"config", "default"}:
        return lambda cfg: cfg.score_mod
    if key in {"none", "null"}:
        return lambda cfg: None
    if key == "alibi":
        heads = int(args[0]) if args else None

        def build(cfg, heads=heads):
            return generate_alibi_bias(heads if heads is not None else cfg.Hq)

        return build
    if key == "tanh_softcap":
        scale = float(args[0]) if args else 30.0
        approx = as_bool(args[1]) if len(args) > 1 else False
        return lambda cfg, scale=scale, approx=approx: generate_tanh_softcap(scale, approx=approx)
    if key in {"tanh_softcap_approx", "tanh_softcapapprox"}:
        scale = float(args[0]) if args else 30.0
        return lambda cfg, scale=scale: generate_tanh_softcap(scale, approx=True)

    raise ValueError(f"Unknown score_mod spec '{name}'")


def build_mask_variants(
    specs: Optional[Sequence[str]],
) -> List[
    Tuple[str, Callable[[BatchInvarianceConfig, torch.device], flex_mod._mask_mod_signature]]
]:
    if not specs:
        return [("config", lambda cfg, device: cfg.mask_mod)]
    variants = []
    for spec in specs:
        name, params = parse_spec(spec)
        builder = make_mask_builder(name, params)
        variants.append((spec, builder))
    return variants


def build_score_variants(
    specs: Optional[Sequence[str]],
) -> List[Tuple[str, Callable[[BatchInvarianceConfig], flex_mod._score_mod_signature]]]:
    if not specs:
        return [("config", lambda cfg: cfg.score_mod)]
    variants = []
    for spec in specs:
        name, params = parse_spec(spec)
        builder = make_score_builder(name, params)
        variants.append((spec, builder))
    return variants


def mask_mod_to_score_mod(
    mask_mod: flex_mod._mask_mod_signature,
) -> flex_mod._score_mod_signature:
    def score_mod(
        score: torch.Tensor,
        batch: torch.Tensor,
        head: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        allowed = mask_mod(batch, head, q_idx, kv_idx).to(dtype=torch.bool)
        return score.masked_fill(~allowed, float("-inf"))

    score_mod.__name__ = f"score_from_{getattr(mask_mod, '__name__', 'mask_mod')}"
    return score_mod


def build_runner(
    *,
    score_mod: flex_mod._score_mod_signature,
    kernel_options: Optional[dict],
    compile_backend: str,
    compile_mode: str,
    dynamic: bool,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
    def forward(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return flex_mod.flex_attention(
            query=query,
            key=key,
            value=value,
            block_mask=block_mask,
            score_mod=score_mod,
            enable_gqa=query.size(1) != key.size(1),
            kernel_options=kernel_options,
        )

    return torch.compile(
        forward,
        backend=compile_backend,
        mode=compile_mode,
        dynamic=dynamic,
        fullgraph=True,
    )


def generate_splits(total: int) -> List[Tuple[int, ...]]:
    splits = {(total,)}
    if total <= 1:
        return [(total,)]

    splits.add(tuple(1 for _ in range(total)))
    splits.add((1, total - 1))
    mid = total // 2
    splits.add((mid, total - mid))
    if total % 2 == 0:
        splits.add(tuple(2 for _ in range(total // 2)))

    ordered: List[Tuple[int, ...]] = sorted(splits, key=lambda s: (len(s), s))
    return ordered


def make_block_mask(
    mask_mod: flex_mod._mask_mod_signature,
    B: int,
    H: int,
    Q: int,
    KV: int,
    *,
    device: torch.device,
    compile_backend: str,
) -> Optional[torch.Tensor]:
    if mask_mod is None:
        return None
    compiled_fn = _get_compiled_block_mask_fn(compile_backend)
    return compiled_fn(mask_mod, B, H, Q, KV, device=device)


def run_split(
    split: Sequence[int],
    config: BatchInvarianceConfig,
    runner: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor
    ],
    *,
    base_q: torch.Tensor,
    base_k: torch.Tensor,
    base_v: torch.Tensor,
    grad_out: Optional[torch.Tensor],
    device: torch.device,
    with_grads: bool,
    compile_backend: str,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    outputs: List[torch.Tensor] = []
    grads_q: List[torch.Tensor] = []
    grads_k: List[torch.Tensor] = []
    grads_v: List[torch.Tensor] = []

    start = 0
    for chunk in split:
        end = start + chunk

        q = base_q[start:end].clone().detach().requires_grad_(with_grads)
        k = base_k[start:end].clone().detach().requires_grad_(with_grads)
        v = base_v[start:end].clone().detach().requires_grad_(with_grads)
        grad_chunk = grad_out[start:end] if with_grads and grad_out is not None else None

        block_mask = make_block_mask(
            config.mask_mod,
            chunk,
            config.Hq,
            config.Q,
            config.KV,
            device=device,
            compile_backend=compile_backend,
        )

        out = runner(q, k, v, block_mask)
        outputs.append(out.detach())

        if with_grads:
            assert grad_chunk is not None
            grads = torch.autograd.grad(
                out,
                (q, k, v),
                grad_chunk,
                retain_graph=False,
                allow_unused=False,
            )
            grads_q.append(grads[0].detach())
            grads_k.append(grads[1].detach())
            grads_v.append(grads[2].detach())

        start = end

    out_tensor = torch.cat(outputs, dim=0)
    if not with_grads:
        return out_tensor, None, None, None

    return (
        out_tensor,
        torch.cat(grads_q, dim=0),
        torch.cat(grads_k, dim=0),
        torch.cat(grads_v, dim=0),
    )


def run_permutation(
    perm: torch.Tensor,
    config: BatchInvarianceConfig,
    runner: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor
    ],
    *,
    base_q: torch.Tensor,
    base_k: torch.Tensor,
    base_v: torch.Tensor,
    grad_out: Optional[torch.Tensor],
    device: torch.device,
    with_grads: bool,
    compile_backend: str,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    q = base_q[perm].clone().detach().requires_grad_(with_grads)
    k = base_k[perm].clone().detach().requires_grad_(with_grads)
    v = base_v[perm].clone().detach().requires_grad_(with_grads)
    grad_chunk = grad_out[perm] if with_grads and grad_out is not None else None

    block_mask = make_block_mask(
        config.mask_mod,
        config.B,
        config.Hq,
        config.Q,
        config.KV,
        device=device,
        compile_backend=compile_backend,
    )

    out = runner(q, k, v, block_mask)
    unperm_out = torch.empty_like(out)
    unperm_out[perm] = out

    if not with_grads:
        return unperm_out.detach(), None, None, None

    assert grad_chunk is not None
    grads = torch.autograd.grad(
        out,
        (q, k, v),
        grad_chunk,
        retain_graph=False,
        allow_unused=False,
    )
    unperm_gq = torch.empty_like(grads[0])
    unperm_gq[perm] = grads[0]
    unperm_gk = torch.empty_like(grads[1])
    unperm_gk[perm] = grads[1]
    unperm_gv = torch.empty_like(grads[2])
    unperm_gv[perm] = grads[2]

    return (
        unperm_out.detach(),
        unperm_gq.detach(),
        unperm_gk.detach(),
        unperm_gv.detach(),
    )


def check_close(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> Tuple[bool, float]:
    if torch.equal(reference, candidate):
        return True, 0.0

    diff = (reference - candidate).abs().max()
    return False, diff.float().item()


def format_split(split: Sequence[int]) -> str:
    return "|".join(str(part) for part in split)


def format_status(passed: bool) -> str:
    if passed:
        return "\033[92mPASS\033[0m"  # Green
    return "\033[91mFAIL\033[0m"  # Red


def run_config(
    config: BatchInvarianceConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
    backend: str,
    mode: str,
    dynamic: bool,
    kernel_options: Optional[dict],
    check_grads: bool,
    num_permutations: int,
) -> Tuple[bool, List[List[str]]]:
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed(42)

    score_mod = config.score_mod
    if score_mod is None and config.mask_mod is not None:
        score_mod = mask_mod_to_score_mod(config.mask_mod)

    q = torch.randn(
        config.B, config.Hq, config.Q, config.Dqk, device=device, dtype=dtype, requires_grad=False
    )
    k = torch.randn(
        config.B,
        config.Hkv,
        config.KV,
        config.Dqk,
        device=device,
        dtype=dtype,
        requires_grad=False,
    )
    v = torch.randn(
        config.B, config.Hkv, config.KV, config.Dv, device=device, dtype=dtype, requires_grad=False
    )
    grad_out = (
        torch.randn(
            config.B,
            config.Hq,
            config.Q,
            config.Dv,
            device=device,
            dtype=dtype,
            requires_grad=False,
        )
        if check_grads
        else None
    )

    runner = build_runner(
        score_mod=score_mod,
        kernel_options=kernel_options,
        compile_backend=backend,
        compile_mode=mode,
        dynamic=dynamic,
    )

    splits = generate_splits(config.B)
    reference_split = (config.B,)

    ref_out, ref_gq, ref_gk, ref_gv = run_split(
        reference_split,
        config,
        runner,
        base_q=q,
        base_k=k,
        base_v=v,
        grad_out=grad_out,
        device=device,
        with_grads=check_grads,
        compile_backend=backend,
    )

    rows: List[List[str]] = []
    all_pass = True

    for split in splits:
        out, grad_q, grad_k, grad_v = run_split(
            split,
            config,
            runner,
            base_q=q,
            base_k=k,
            base_v=v,
            grad_out=grad_out,
            device=device,
            with_grads=check_grads,
            compile_backend=backend,
        )

        f_pass, f_diff = check_close(ref_out, out)
        if check_grads:
            assert ref_gq is not None and grad_q is not None
            assert ref_gk is not None and grad_k is not None
            assert ref_gv is not None and grad_v is not None
            gq_pass, gq_diff = check_close(ref_gq, grad_q)
            gk_pass, gk_diff = check_close(ref_gk, grad_k)
            gv_pass, gv_diff = check_close(ref_gv, grad_v)

            row = [
                format_split(split),
                format_status(f_pass),
                f"{f_diff:.3e}",
                format_status(gq_pass),
                f"{gq_diff:.3e}",
                format_status(gk_pass),
                f"{gk_diff:.3e}",
                format_status(gv_pass),
                f"{gv_diff:.3e}",
            ]
            rows.append(row)

            all_pass &= f_pass and gq_pass and gk_pass and gv_pass
        else:
            row = [format_split(split), format_status(f_pass), f"{f_diff:.3e}"]
            rows.append(row)
            all_pass &= f_pass

    if device.type == "cuda":
        torch.cuda.synchronize()

    # check permutations
    for i in range(num_permutations):
        perm = torch.randperm(config.B, device=device)
        out, grad_q, grad_k, grad_v = run_permutation(
            perm,
            config,
            runner,
            base_q=q,
            base_k=k,
            base_v=v,
            grad_out=grad_out,
            device=device,
            with_grads=check_grads,
            compile_backend=backend,
        )

        f_pass, f_diff = check_close(ref_out, out)
        if check_grads:
            assert ref_gq is not None and grad_q is not None
            assert ref_gk is not None and grad_k is not None
            assert ref_gv is not None and grad_v is not None
            gq_pass, gq_diff = check_close(ref_gq, grad_q)
            gk_pass, gk_diff = check_close(ref_gk, grad_k)
            gv_pass, gv_diff = check_close(ref_gv, grad_v)

            row = [
                f"permute_{i}",
                format_status(f_pass),
                f"{f_diff:.3e}",
                format_status(gq_pass),
                f"{gq_diff:.3e}",
                format_status(gk_pass),
                f"{gk_diff:.3e}",
                format_status(gv_pass),
                f"{gv_diff:.3e}",
            ]
            rows.append(row)

            all_pass &= f_pass and gq_pass and gk_pass and gv_pass
        else:
            row = [f"permute_{i}", format_status(f_pass), f"{f_diff:.3e}"]
            rows.append(row)
            all_pass &= f_pass

    return all_pass, rows


def parse_kernel_options(options: Optional[Iterable[str]]) -> Optional[dict]:
    if not options:
        return None

    parsed = {}
    for item in options:
        if "=" not in item:
            raise ValueError(f"kernel option '{item}' must be KEY=VALUE")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.lower() in {"true", "false"}:
            parsed[key] = value.lower() == "true"
        else:
            with contextlib.suppress(ValueError):
                parsed[key] = int(value)
                continue
            with contextlib.suppress(ValueError):
                parsed[key] = float(value)
                continue
            parsed[key] = value

    return parsed


def main(
    device: str = DEFAULT_DEVICE,
    dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16",
    backend: str = "inductor",
    mode: str = "default",
    dynamic: bool = True,
    no_grads: bool = False,
    config_name: Optional[List[str]] = None,
    kernel_option: Optional[List[str]] = None,
    mask_mod: Optional[List[str]] = None,
    score_mod: Optional[List[str]] = None,
    permutations: int = 4,
) -> int:
    """Test FlexAttention batch invariance across configurations."""

    device_obj = torch.device(device)
    dtype_obj = DTYPE_MAP[dtype]
    kernel_options = parse_kernel_options(kernel_option)
    if kernel_options is None:
        kernel_options = {}
    kernel_options.setdefault("FORCE_FLASH_ATTENTION", True)

    config_filter = set(config_name) if config_name else None
    selected_configs = (
        tuple(cfg for cfg in DEFAULT_CONFIGS if not config_filter or cfg.name in config_filter)
        or DEFAULT_CONFIGS
    )

    mask_variants = build_mask_variants(mask_mod)
    score_variants = build_score_variants(score_mod)

    check_grads = not no_grads
    headers = (
        [
            "Split",
            "Forward",
            "Δ Forward",
            "Grad q",
            "Δ Grad q",
            "Grad k",
            "Δ Grad k",
            "Grad v",
            "Δ Grad v",
        ]
        if check_grads
        else ["Split", "Forward", "Δ Forward"]
    )

    overall_pass = True

    print(" FLEXATTENTION BATCH INVARIANCE CHECK ".center(100, "="))
    print(
        f"Device: {device_obj}, dtype: {dtype_obj}, backend: {backend}, mode: {mode}, dynamic={dynamic}"
    )
    if kernel_options:
        print(f"kernel_options: {kernel_options}")
    if permutations > 0:
        print(f"Permutations: {permutations} random permutations per config")
    print("=" * 100)

    for config_entry in selected_configs:
        for mask_label, mask_builder in mask_variants:
            mask_mod_fn = mask_builder(config_entry, device_obj)
            for score_label, score_builder in score_variants:
                score_mod_fn = score_builder(config_entry)
                combined_cfg = replace(
                    config_entry,
                    mask_mod=mask_mod_fn,
                    score_mod=score_mod_fn,
                )

                label_parts: List[str] = []
                if mask_label != "config":
                    label_parts.append(f"mask={mask_label}")
                if score_label != "config":
                    label_parts.append(f"score={score_label}")
                label_suffix = f" [{' | '.join(label_parts)}]" if label_parts else ""

                print(f"\n>> Configuration: {config_entry.name}{label_suffix}")
                print(
                    f"B={combined_cfg.B}, Hq={combined_cfg.Hq}, Hkv={combined_cfg.Hkv}, Q={combined_cfg.Q}, "
                    f"KV={combined_cfg.KV}, Dqk={combined_cfg.Dqk}, Dv={combined_cfg.Dv}"
                )

                doc_lengths = getattr(mask_mod_fn, "_doc_lengths", None)
                if doc_lengths is not None:
                    print(f"Document lengths: {doc_lengths}")

                success, rows = run_config(
                    combined_cfg,
                    device=device_obj,
                    dtype=dtype_obj,
                    backend=backend,
                    mode=mode,
                    dynamic=dynamic,
                    kernel_options=kernel_options,
                    check_grads=check_grads,
                    num_permutations=permutations,
                )

                overall_pass &= success

                print(tabulate(rows, headers=headers, tablefmt="grid"))

    summary_status = format_status(overall_pass)
    print("\n" + summary_status.center(100, "="))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    CLI(main, as_positional=False)
