"""Format collected kernel declarations and trace results as plain text.

``describe_kernel`` reads host state from ``WarpSpecialized._collect``. After compilation,
it also shows TMA reports, opt-outs, alias and fence findings and the traced protocol.
``kernel_checks`` returns the channel, alias and fence verdicts as data.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Literal

from attn_gym.warpdrive.channels import Threads, Tma, Umma, Warps
from attn_gym.warpdrive.fences import FenceStatus
from attn_gym.warpdrive.scheduling import _tile_grid, scheduler_role

WIDTH = 100  # wrap channel flows past this many columns
_LABEL = 8  # width of the channel detail labels ("override")
NOT_CHECKED = "not checked yet"  # no traced or restored result
TmaVerdict = Literal["verified", "unverified", "opted out", "no copies", "not checked yet"]
EpochsVerdict = Literal["verified", "unverified", "not checked yet"]
AliasStatus = Literal["no hazard found", "unverified", "potential hazard"]


@dataclass(frozen=True)
class ChannelChecks:
    """One channel's verdicts, with None for checks that do not apply.

    ``tma`` checks copies from a managed ``Tma`` producer (``TmaReport.verdict``).
    ``epochs`` checks round-robin or exclusive epoch ownership. Applicable checks are
    ``NOT_CHECKED`` until compilation or cache restoration.
    """

    tma: TmaVerdict | None = None
    epochs: EpochsVerdict | None = None


@dataclass(frozen=True)
class KernelChecks:
    """Channel, alias and fence verdicts returned by ``WarpSpecialized.checks()``.

    ``channels`` includes every channel. ``aliases`` maps each examined overlap to its status,
    keyed by ``(name, overlapped name)`` (``name ~ overlapped name`` in ``describe()``).
    ``fences`` maps each explicit fence to its status, keyed by ``(channel, "commit_fence"``
    or ``"release_fence")``. Both are empty before compilation or with ``alias_check="off"``.
    """

    channels: dict[str, ChannelChecks]
    aliases: dict[tuple[str, str], AliasStatus]
    fences: dict[tuple[str, str], FenceStatus]


def kernel_checks(kernel) -> KernelChecks:
    channels = {}
    for name, ch in kernel._ws_chans.items():
        report = kernel._ws_tma.get(name)
        tma = report.verdict if report is not None else NOT_CHECKED if ch.checks_tma else None
        epochs = None
        if ch.owned_epochs:
            ownership = kernel._ws_ownership.get(name)
            epochs = NOT_CHECKED if ownership is None else ownership_verdict(ownership)
        channels[name] = ChannelChecks(tma, epochs)
    rep = kernel._ws_alias
    aliases = {} if rep is None else {r.names: r.status for r in rep.results}
    fence_rep = kernel._ws_fences
    fences = {} if fence_rep is None else {(r.channel, r.key): r.status for r in fence_rep.results}
    return KernelChecks(channels, aliases, fences)


def ownership_verdict(ownership: str) -> str:
    return "verified" if ownership.startswith("verified") else "unverified"


def describe_kernel(kernel) -> str:
    sections = [
        _header(kernel),
        _roles(kernel),
        _channels(kernel),
        _turnstiles(kernel),
        _raw_barriers(kernel),
        _named_barriers(kernel),
        _streams(kernel),
        _tmem(kernel),
        _smem(kernel),
        _opt_outs(kernel),
        _alias(kernel),
        _fences(kernel),
        _protocol(kernel),
    ]
    return "\n\n".join(s for s in sections if s)


# Value formatting


def fmt_counts(counts) -> str:
    """Per-slot arrival counts: ``1 x6`` when uniform, ``[1, 4]`` otherwise."""
    counts = list(counts)
    if len(set(counts)) == 1:
        return str(counts[0]) if len(counts) == 1 else f"{counts[0]} x{len(counts)}"
    return f"[{', '.join(str(c) for c in counts)}]"


def is_pair(cluster) -> bool:
    """A lone CTA pair, whose ranks print as leader and peer."""
    return cluster.pair and cluster.size == 2


def rank_label(rank: int, pair: bool = True) -> str:
    """``leader``/``peer`` in a CTA pair, ``rank r`` in any other cluster."""
    if pair:
        return "leader" if rank == 0 else "peer"
    return f"rank {rank}"


def fmt_per_cta(per_rank: dict, ranks, by_rank: bool = False, pair: bool = True) -> str:
    """Format ``{rank: per-slot counts}`` for ``ranks``.

    Label by rank when counts differ or ``by_rank`` is True.
    """
    ranks = list(ranks)
    values = [fmt_counts(per_rank[r]) for r in ranks]
    if not by_rank and len(set(values)) == 1:
        return values[0]
    return ", ".join(f"{rank_label(r, pair)} {v}" for r, v in zip(ranks, values))


def fmt_mask(mask, cluster) -> str:
    """A ``tcgen05.commit`` mask (one bit per CTA rank), or each signalling rank's mask."""
    width = max(cluster.size, 2) + 2
    if isinstance(mask, int):
        return f"{mask:#0{width}b}"
    pair = is_pair(cluster)
    return ", ".join(f"{rank_label(r, pair)} {m:#0{width}b}" for r, m in mask.items())


def _fmt_value(key: str, value, cluster) -> str:
    if key in ("full_counts", "empty_counts"):
        return fmt_per_cta(value, range(cluster.size), pair=is_pair(cluster))
    if key in ("full_mask", "empty_mask"):
        return fmt_mask(value, cluster)
    return str(value)


def _sites(sites) -> str:
    """``a.py:1, a.py:7, b.py:3`` -> ``a.py:1,7, b.py:3``."""
    by_file = {}
    for site in sorted(sites, key=_site_key):
        path, _, line = site.rpartition(":")
        by_file.setdefault(path, []).append(line)
    return ", ".join(f"{path}:{','.join(lines)}" for path, lines in by_file.items())


def _site_key(site: str):
    path, _, line = site.rpartition(":")
    return (path, int(line) if line.isdigit() else 0)


def _table(rows: list[list[str]], right: set[int] = frozenset()) -> list[str]:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = []
    for row in rows:
        cells = [
            c.rjust(w) if i in right else c.ljust(w) for i, (c, w) in enumerate(zip(row, widths))
        ]
        lines.append(("  " + "  ".join(cells)).rstrip())
    return lines


# Sections


def _header(kernel) -> str:
    lines = [f"{type(kernel).__name__}: {kernel.num_warps} warps, {kernel.num_threads} threads"]
    ctx = kernel._ws_ctx_type
    rows = [
        ["cluster", kernel.cluster.describe()],
        ["params", ctx.__name__ if ctx is not None else "unannotated (any object)"],
    ]
    return "\n".join(lines + _table(rows))


def _roles(kernel) -> str:
    pair = kernel.cluster.size > 1
    rows = [["warps", "role", "regs", "cta", "raw"]]
    entries = [(r.first_warp, r.warps, r.name, r.regs, r.cta, r.raw) for r in kernel._ws_roles]
    entries += [(w.start, w, "(padding)", regs, "", False) for w, regs in kernel._ws_padding]
    for _, warps, name, regs, cta, raw in sorted(entries, key=lambda e: e[0]):
        span = str(warps.start) if len(warps) == 1 else f"{warps.start}-{warps.stop - 1}"
        rows.append(
            [
                span,
                name,
                "-" if regs is None else str(regs),
                cta if pair else ("" if name == "(padding)" else "all"),
                "raw" if raw else "",
            ]
        )
    return "\n".join(["Roles", *_table(rows, right={2})])


def _mech(mech) -> str:
    if isinstance(mech, dict):  # a per-role consumer side
        return "{" + ", ".join(f"{role}: {_mech(m)}" for role, m in mech.items()) + "}"
    return f"Tma({mech.tx_bytes} B)" if isinstance(mech, Tma) else mech.__name__


def _bound(bound) -> str:
    return ", ".join(r.name + ("" if s is None else f"@{s}") for r, s in bound)


def channel_kind(ch) -> str:
    return f"Slots({ch.n})" if ch.fixed else f"Ring({ch.n})"


def channel_signal(ch) -> str:
    return f"{_mech(ch.producer)} -> {_mech(ch.consumer)}"


def channel_barriers(ch, cluster) -> list[tuple[str, str]]:
    """Format named-barrier IDs/counts or mbarrier full/empty counts and TMA arming.

    Returns labelled pairs: ``("named", ...)`` or ``("full", ...)``, ``("empty", ...)``.
    """
    if ch.named:
        # One hardware barrier per slot and side; producers and consumers both arrive.
        full_ids, empty_ids = (
            [nb._allocated_id() for nb in bars] for bars in (ch.full_barriers, ch.empty_barriers)
        )
        count = fmt_counts(ch.named_counts())
        named = f"full ids {full_ids}, empty ids {empty_ids}, {count} threads"
        if ch.turnstile is not None:
            named += f", taken in turn ({ch.turnstile})"
        return [("named", named)]
    cs = cluster.size
    pair = is_pair(cluster)
    sc = ch.scope
    full_ranks = sc.full_ctas if sc.pair else range(cs)
    empty_ranks = sc.empty_ctas if sc.pair else range(cs)
    full = fmt_per_cta(sc.full_counts, full_ranks, by_rank=sc.pair, pair=pair)
    if sc.tx is not None:
        # Leaders arm for both CTAs of their pair, or rank 0's scheduler for every CTA.
        armers = [r for r, targets in sc.full_targets.items() if targets]
        armed = sc.pair and (sc.tx_ctas > 1 or armers == [0])
        full += f", tx {sc.tx} B" + (" (leader-armed)" if armed else "")
    empty = fmt_per_cta(sc.empty_counts, empty_ranks, by_rank=sc.pair, pair=pair)
    return [("full", full), ("empty", empty)]


def channel_scope(ch, cluster) -> list[str]:
    """Describe cross-CTA scope, multicast and masks.

    Single-CTA kernels return no scope labels.
    """
    if cluster.size == 1:
        return []
    sc = ch.scope
    if not sc.pair:
        return ["CTA-local"]
    scope = ["pair-scoped" if is_pair(cluster) else "cross-CTA"]
    if isinstance(ch.producer, Tma) and ch.producer.multicast:
        scope.append(f"multicast {'+'.join(ch.producer.multicast)}")
    if ch.producer is Umma:
        scope.append(f"commit_mask {fmt_mask(sc.full_mask, cluster)}")
    if ch.consumer is Umma:
        scope.append(f"release_mask {fmt_mask(sc.empty_mask, cluster)}")
    if {Threads, Warps} & {ch.producer, *ch.mechanisms("consumer")}:
        scope.append(f"remote_scope {ch.remote_scope}")
    if sc.tail:
        scope.append("tail")
    return scope


def channel_storage(ch) -> list[str]:
    """The channel's payload or plan buffers, then its fences."""
    extra = []
    if ch.payload is not None:
        dtype, shape = ch.payload
        extra.append(f"payload {getattr(dtype, '__name__', dtype)}{tuple(shape)}")
    if ch.buffers:
        names = [k if k == b.name else f"{k}={b.name}" for k, b in ch.buffers.items()]
        extra.append(f"buffer {', '.join(names)}")
    if ch.commit_fence:
        extra.append(f"commit_fence {'+'.join(ch.commit_fence)}")
    if ch.release_fence:
        fence = f"release_fence {'+'.join(ch.release_fence)}"
        if ch.per_role:  # the thread roles' fence; Umma roles emit none
            fence += f" ({', '.join(r for r, m in ch.consumer.items() if m is not Umma)})"
        extra.append(fence)
    return extra


def _channels(kernel) -> str:
    chans = kernel._ws_chans
    if not chans:
        return ""
    width = max(len(n) for n in chans)
    pad = " " * (2 + width + 2)
    checks = kernel_checks(kernel).channels
    lines = ["Channels"]
    for name, ch in chans.items():
        head = f"  {name:<{width}}  {channel_kind(ch):<9}"
        sides = [
            "raw code" if side == ch.external else _bound(bound)
            for side, bound in (("producer", ch.producers), ("consumer", ch.consumers))
        ]
        flow = " -> ".join(sides)
        lines += textwrap.wrap(
            flow,
            WIDTH,
            initial_indent=head,
            subsequent_indent=" " * (len(head) + 4),
            break_on_hyphens=False,
        )
        details = [("signal", channel_signal(ch)), *channel_barriers(ch, kernel.cluster)]
        if ch.exclusive:
            owners = ", ".join(r.name for r, _ in ch.consumers)
            counts = (
                "empty counts overridden: each owner's release must supply them (unverified)"
                if ch.empty_counts_overridden
                else "counts are one consumer's"
            )
            details.append(
                (
                    "owners",
                    (
                        f"exclusive: one of {owners} owns each epoch, the others skip it; {counts}; "
                        "bootstrap and skew are the author's"
                    ),
                )
            )
        for side in ("producer", "consumer"):
            if ch.turns(side) > 1:
                details.append(
                    (
                        "turns",
                        (
                            f"{side}s round-robin: instance j of {ch.turns(side)} takes epochs "
                            f"j, j+{ch.turns(side)}, ...; counts are one instance's"
                        ),
                    )
                )
        if checks[name].epochs == NOT_CHECKED:
            details.append(("epochs", "ownership not traced"))
        elif checks[name].epochs is not None:
            details.append(("epochs", kernel._ws_ownership[name]))
        if ch.external is not None:
            counts = ch.external_counts.replace("_", " ")
            details.append(
                (
                    "external",
                    (
                        f"{ch.external} side driven by raw code; {counts} are the override's "
                        "(unverified)"
                    ),
                )
            )
        scope = channel_scope(ch, kernel.cluster)
        if scope:
            details.append(("scope", ", ".join(scope)))
        extra = channel_storage(ch)
        if extra:
            storage = ch.payload is not None or ch.buffers
            details.append(("storage" if storage else "fences", ", ".join(extra)))
        details += [("override", o) for o in _overrides(ch, kernel.cluster)]
        tma = _tma_status(kernel, name, checks[name].tma)
        if tma:
            details.append(("tma", tma))
        lines += [f"{pad}{k:<{_LABEL}}  {v}" for k, v in details]
    return "\n".join(lines)


def _overrides(ch, cluster) -> list[str]:
    """``key derived -> effective`` for every overridden value."""
    out = []
    derived = ch.derived_scope
    for key in ch.override or {}:
        if key == ch.external_counts:  # no role derives them (the "external" line)
            continue
        ours, theirs = getattr(derived, key), getattr(ch.scope, key)
        note = "  (differs from derived)" if ours != theirs else ""
        out.append(
            f"{key} {_fmt_value(key, ours, cluster)} -> {_fmt_value(key, theirs, cluster)}{note}"
        )
    if ch.remote_scope_overridden:
        ours, theirs = ch.derived_remote_scope, ch.remote_scope
        note = "  (differs from derived)" if ours != theirs else ""
        out.append(f"remote_scope {ours} -> {theirs}{note}")
    # Internal channels own their fences; the st.async work ring needs none.
    fences = (
        ("commit_fence", ch.derived_commit_fence, ch.commit_fence_overridden, ch.commit_fence),
        ("release_fence", ch.derived_release_fence, ch.release_fence_overridden, ch.release_fence),
    )
    for key, ours, overridden, theirs in fences:
        if ours and overridden and not ch.internal:
            note = "  (differs from derived)" if ours != theirs else ""
            out.append(f"{key} {'+'.join(ours)} -> {'+'.join(theirs) or 'none'}{note}")
    return out


def _tma_status(kernel, name: str, verdict: str | None) -> str:
    if verdict is None:
        return ""
    if verdict == NOT_CHECKED:
        return f"{NOT_CHECKED} (compile to check the copies)"
    return kernel._ws_tma[name].status(_sites)


def _turnstiles(kernel) -> str:
    if not kernel._ws_turns:
        return ""
    rows = [
        [name, " -> ".join(r.name for r in ts.parties), "" if ts.enabled else "(disabled)"]
        for name, ts in kernel._ws_turns.items()
    ]
    return "\n".join(["Turnstiles", *_table(rows)])


def _raw_barriers(kernel) -> str:
    if not kernel._ws_raw_bars:
        return ""
    rows = []
    for name, b in kernel._ws_raw_bars.items():
        init = "user-initialized" if b.counts is None else f"layer-initialized {b.counts}"
        rows.append([name, f"{b.n} mbarriers", init, "user-driven (unchecked)"])
    return "\n".join(["Raw barriers", *_table(rows)])


def _named_barriers(kernel) -> str:
    if not kernel._ws_named_bars:
        return ""
    traced = getattr(kernel, "_ws_traced", False)
    rows = []
    for name, nb in kernel._ws_named_bars.items():
        users = kernel._ws_named_users.get(name, {})
        labels = {"arrive": "arrive", "sync": "sync", "native": "native id"}
        ops = [f"{labels[op]} by {', '.join(users[op])}" for op in labels if users.get(op)]
        ops += [f"unverified: {note}" for note in users.get("unverified", ())]
        if not traced:
            ops = ["users known after a compile"]
        rows.append(
            [name, f"id {nb._allocated_id()}", f"{nb.threads} threads", "; ".join(ops) or "unused"]
        )
    return "\n".join(["Named barriers", *_table(rows)])


def _streams(kernel) -> str:
    if not kernel._ws_streams:
        return ""
    lines = ["Streams"]
    for name, st in kernel._ws_streams.items():
        pad = " " * (2 + len(name) + 2)
        lines.append(f"  {name}  {st.describe()}")
        sched = next((r for r in kernel._ws_roles if r.fn is scheduler_role), None)
        if sched is not None:
            lines.append(f"{pad}scheduler warp {sched.first_warp} broadcasts over {name}_ring")
        else:
            grid = ""
            if _tile_grid(st):
                grid = (
                    "; launch grid = "
                    + {
                        "auto": "tile space (1-D past 65535 n_tiles or l)",
                        "3d": "tile space",
                        "2d": "(m_tiles * l, n_tiles)",
                    }[st.schedule.grid]
                )
            lines.append(f"{pad}decoded locally in every role{grid}")
        users = [r for r in st.roles if sched is None or r != sched.name]
        lines.append(f"{pad}roles: {', '.join(users)}")
    return "\n".join(lines)


def _tmem(kernel) -> str:
    plan = kernel.tmem_plan
    if plan is None:
        return ""
    head = f"TMEM: {plan.used_cols} columns used, {plan.alloc_cols} allocated"
    return f"{head}\n{plan.describe()}"


def _smem(kernel) -> str:
    plan = kernel.smem_plan
    if plan is None:
        return ""
    resolved = kernel._ws_smem
    per_cta = " per CTA" if kernel.cluster.size > 1 else ""  # each CTA holds its own copy
    if resolved:
        owned = [r.nbytes for r in resolved.values() if r.buffer.overlaps is None]
        total = "?" if None in owned else str(sum(owned))
        head = f"SMEM plan: {total} B{per_cta} in {len(owned)} buffer(s), plus alignment padding"
    else:
        head = f"SMEM plan: layouts bind at begin_roles (compile to see bytes){per_cta}"
    rows = []
    for name, buf in plan.buffers.items():
        res = resolved.get(name)
        size = "" if res is None or res.nbytes is None else f"{res.nbytes} B"
        chans = [
            n for n, ch in kernel._ws_chans.items() if any(b is buf for b in ch.buffers.values())
        ]
        stages = ""  # the stage mode matters only to the channels binding the buffer
        if chans and res is not None and res.stages is not None:
            stages = f"{res.stages} stages (mode {buf.stage_axis})"
        where = (
            f"overlaps {buf.overlaps.name}" if buf.overlaps is not None else f"align {buf.align}"
        )
        rows.append(
            [
                name,
                getattr(buf.dtype, "__name__", str(buf.dtype)),
                size,
                stages,
                where,
                f"-> {', '.join(chans)}" if chans else "",
            ]
        )
    return "\n".join([head, *_table(rows, right={2})])


def _alias(kernel) -> str:
    rep = kernel._ws_alias
    return "" if rep is None else rep.describe()


def _fences(kernel) -> str:
    rep = kernel._ws_fences
    return "" if rep is None else rep.describe()


def _opt_outs(kernel) -> str:
    if not kernel._ws_opt_outs:
        return ""
    return "\n".join(["Unchecked opt-outs", *(f"  {o.describe()}" for o in kernel._ws_opt_outs)])


def _protocol(kernel) -> str:
    log = kernel._ws_log
    if log is None or not log.ops:
        return ""
    body = "\n".join(f"  {line}" for line in log.describe().splitlines())
    return f"Traced protocol\n{body}"
