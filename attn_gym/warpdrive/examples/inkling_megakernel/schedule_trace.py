"""Overlay one Inkling speculative graph replay's model schedule on a Kineto Chrome trace.

GPU timestamps locate launches; the kernel signatures and config's execution order identify
visits. No weights, GPU execution, or CPU-to-GPU timing alignment are needed.
"""

import argparse
import json
import re
from pathlib import Path

from attn_gym.warpdrive.examples.inkling_megakernel.ref import InklingConfig

PREFIX = "warpdrive_inkling_"
CATEGORY = "Inkling schedule"
DEFAULT_CONFIG = InklingConfig()


def annotate_round(
    trace: dict, config: InklingConfig = DEFAULT_CONFIG, *, emitted_tokens: int | None = None
) -> dict:
    """Return the trace with schedule tracks added; validate the named model launches.

    Supply a complete replay of one round on one stream, including its final position update.
    ``emitted_tokens`` is the observed output count; timestamps do not reveal it.
    """
    events = trace["traceEvents"]
    if any(e.get("cat") == CATEGORY for e in events):
        raise ValueError("trace already contains an Inkling schedule")
    gpu = [
        e
        for e in events
        if e.get("ph") == "X" and e.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
    ]

    def replay(e: dict) -> tuple:
        args = e.get("args", {})
        return e["pid"], args.get("graph id"), args.get("correlation")

    models = [e for e in gpu if e["name"].startswith(PREFIX)]
    launches = {replay(e) for e in models}
    if len(launches) != 1:
        raise ValueError("expected exactly one Inkling graph replay")
    launch = launches.pop()
    pid, graph_id, correlation = launch
    if not graph_id or correlation is None:
        raise ValueError("missing graph replay identifiers")
    ops = sorted((e for e in gpu if replay(e) == launch), key=lambda e: e["ts"])
    if len({e["tid"] for e in ops}) != 1:
        raise ValueError("expected a single-stream Inkling graph replay")
    named = iter(
        (i, e["name"].removeprefix(PREFIX))
        for i, e in enumerate(ops)
        if e["name"].startswith(PREFIX)
    )

    def take(pattern: str) -> tuple[int, re.Match]:
        index, name = next(named, (-1, "<end>"))
        match = re.match(pattern, name)
        if not match:
            raise ValueError(f"expected Inkling kernel {pattern!r}, found {name!r}")
        return index, match

    used = {e.get("tid") for e in events if e.get("pid") == pid}
    # Perfetto imports negative Chrome thread ids as tid 0, collapsing distinct tracks.
    tid = max((t for t in used if isinstance(t, int)), default=0) + 1
    phase, block, layer_track = tid, tid + 1, tid + 2
    added = []
    for track, label in zip(
        (phase, block, layer_track),
        ("Decode phase", "Loop passes and heads", "Target layer visits"),
        strict=True,
    ):
        for name, args in (
            ("thread_name", {"name": label}),
            ("thread_sort_index", {"sort_index": track - layer_track - 1}),
        ):
            added.append({"ph": "M", "pid": pid, "tid": track, "name": name, "args": args})

    def span(track: int, label: str, first: int, last: int, **args: int) -> None:
        start = ops[first]["ts"]
        end = ops[last]["ts"] + ops[last]["dur"]
        if track == phase and last + 1 < len(ops):
            # Kineto can report slightly overlapping kernels even on one stream. Adjacent
            # phases share that boundary; keep the individual GPU slices unchanged.
            end = min(end, ops[last + 1]["ts"])
        added.append(
            {
                "ph": "X",
                "cat": CATEGORY,
                "pid": pid,
                "tid": track,
                "name": label,
                "ts": start,
                "dur": end - start,
                "args": args,
            }
        )

    backbone, _ = take("draft_backbone_")
    draft_head, head_signature = take(r"draft_head_.*?_d(\d+)_")
    drafts = int(head_signature[1])
    k = drafts + 1
    if emitted_tokens is not None and not 1 <= emitted_tokens <= k:
        raise ValueError(f"emitted_tokens must be between 1 and {k}")
    span(phase, f"Draft {drafts} token{'s' if drafts != 1 else ''}", 0, draft_head)
    span(block, "Draft backbone", backbone, backbone)
    span(block, "Draft vocabulary head", draft_head, draft_head)

    groups: dict[str, list[int]] = {}
    for visit, layer, p in config.visits():
        if config.is_loop(layer):
            group = f"Core pass {p + 1}/{config.num_passes}"
        else:
            group = "Entrance" if layer < config.loop_start else "Exit"
        indices = groups.setdefault(group, [])
        if p and layer == config.loop_start:
            message, _ = take(rf"msg_h{config.hidden_size}_t{k}_")
            indices.append(message)
            span(layer_track, "Core pass messages", message, message, core_pass=p + 1)
        attn = "full" if config.is_full(layer) else f"swa{config.sliding_window}"
        gates = "_msg" if p else ""
        index, _ = take(rf"t{k}_{attn}{gates}_h{config.hidden_size}_")
        indices.append(index)
        span(
            layer_track,
            f"Visit {visit}: L{layer} {attn}",
            index,
            index,
            visit=visit,
            layer=layer,
            pass_index=p,
        )
    for group, indices in groups.items():
        span(block, group, indices[0], indices[-1])
    head, _ = take(rf"head_h{config.hidden_size}_v{config.vocab_size}_t{k}_")
    ctx, _ = take("draft_ctx_")
    if next(named, None) is not None:
        raise ValueError("unexpected Inkling launches after context commit")
    if ctx == len(ops) - 1:
        raise ValueError("missing output and position updates after context commit")
    # decode_block copies the target feature taps *after* the vocabulary head. Acceptance
    # starts at the verified logits' argmax, not at the first post-head copy.
    argmax = [i for i in range(head + 1, ctx) if "ArgMaxOps" in ops[i]["name"]]
    if len(argmax) != 1:
        raise ValueError("expected one target argmax before context commit")
    accept = argmax[0]
    span(phase, f"Verify {k} positions", draft_head + 1, accept - 1)
    span(block, "Target vocabulary head", head, head)
    span(block, "Commit drafter context", ctx, ctx)
    count = {} if emitted_tokens is None else {"emitted_tokens": emitted_tokens}
    label = "Accept prefix and commit"
    if emitted_tokens is not None:
        label += f" | {emitted_tokens} emitted"
    span(phase, label, accept, len(ops) - 1, **count)
    return {**trace, "traceEvents": events + added}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--config", type=Path, help="checkpoint config.json (default: Inkling 2 Mini)"
    )
    parser.add_argument(
        "--emitted-tokens", type=int, help="observed output count from this replay"
    )
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("choose an output distinct from the original trace")
    config = (
        InklingConfig.from_hf(json.loads(args.config.read_text()))
        if args.config
        else InklingConfig()
    )
    try:
        result = annotate_round(
            json.loads(args.input.read_text()), config, emitted_tokens=args.emitted_tokens
        )
    except ValueError as exc:
        parser.error(str(exc))
    args.output.write_text(json.dumps(result))


if __name__ == "__main__":
    main()
