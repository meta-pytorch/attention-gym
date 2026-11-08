"""The warp-role layer's structure as a profiler ``PipelinePlan`` (names match ``tracing.py``)."""

from __future__ import annotations

from attn_gym.profiler.pipeline.plan import (
    Dependency,
    PipelinePlan,
    Region,
    Resource,
    Role,
)
from attn_gym.warpdrive.describe import is_pair, rank_label


def plan_from_warpdrive(kernel, *, cta: int = 0, name: str | None = None) -> PipelinePlan:
    """The kernel's roles, channels, and handoffs as a ``PipelinePlan`` for CTA rank ``cta``.

    Names match the ranges the layer emits (either trace backend): one ``Role`` and one ``Region`` per role
    instance running in this CTA, one ``Resource`` per channel between two of them. Per channel:

    * a ``data`` dependency producer -> consumer at distance 0 for every pair of roles. Handoffs
      in both directions within one item (FA4's S and P) would form a same-iteration cycle; the
      edge that closes it gets distance 1, kind ``state``, and the label ``(feedback)``;
    * a ``reuse`` dependency back to each producer from every consumer that no other consumer of
      the channel follows in the data order (the join's last arrivals), at distance
      ``slots / uses per item`` when the trace saw a static number of acquires/waits per item,
      else 1 (a lower bound for channels used in an inner loop). ``slots`` is the stage count,
      or 1 for a role bound to one slot of ``Slots``.
      A ``WorkStream`` work ring gets one more: every role releases a work slot as it starts
      the item (on receipt), one item before the end of the region that holds it.

    Reuse distances need a compiled kernel; before a compile every channel counts as used once
    per item.
    """
    kernel._collect()
    cluster = kernel.cluster
    roles = [r for r in kernel._ws_roles if cta in cluster.ctas(r.cta)]
    in_cta = {r.name for r in roles}
    plan = PipelinePlan(
        name or _kernel_name(kernel),
        iteration_name="work item" if kernel._ws_streams else "launch",
    )
    where = f" ({rank_label(cta, is_pair(cluster))})" if cluster.size > 1 else ""
    for r in roles:
        plan.add_role(Role(r.name, f"{r.name}{where}", r.first_warp, r.first_warp + r.num_warps))
        plan.add_region(Region(r.name, r.name, r.name, 0, 1))
    channels = {}
    for cname, ch in kernel._ws_chans.items():
        producers = [p.name for p, _ in ch.producers if p.name in in_cta]
        consumers = [c.name for c, _ in ch.consumers if c.name in in_cta]
        if any(p != c for p in producers for c in consumers):
            channels[cname] = (ch, producers, consumers)
            plan.add_resource(Resource(cname, cname, ch.n))
    downstream: dict[str, set[str]] = {r: set() for r in in_cta}
    for cname, (_, producers, consumers) in channels.items():
        for prod in producers:
            for cons in consumers:
                if prod == cons:
                    continue
                if _reaches(downstream, cons, prod):
                    plan.add_dependency(
                        Dependency(prod, cons, cname, 1, "state", f"{cname} (feedback)")
                    )
                else:
                    downstream[prod].add(cons)
                    plan.add_dependency(Dependency(prod, cons, cname, 0, "data", cname))
    uses = getattr(kernel, "_ws_uses", {})
    work_rings = {f"{stream}_ring" for stream in kernel._ws_streams}
    for cname, (ch, producers, consumers) in channels.items():
        bound = {ri.name for ri, slot in [*ch.producers, *ch.consumers] if slot is not None}
        for prod in producers:
            others = [c for c in consumers if c != prod]
            # Consumers no other consumer finishes after (data order): the join's last arrivals.
            for cons in [
                c for c in others if not any(_reaches(downstream, c, o) for o in others if o != c)
            ]:
                per_item = (uses.get((prod, cname), 1), uses.get((cons, cname), 1))
                # A role bound to one slot of Slots reuses that slot only.
                capacity = 1 if bound & {prod, cons} else ch.n
                distance = 1 if 0 in per_item else max(1, capacity // max(per_item))
                distance += cname in work_rings
                plan.add_dependency(
                    Dependency(cons, prod, cname, distance, "reuse", f"reuse {cname}")
                )
    return plan


def _kernel_name(kernel) -> str:
    get_name = getattr(kernel, "get_name", None)
    return get_name() if callable(get_name) else type(kernel).__name__


def _reaches(edges: dict, src: str, dst: str) -> bool:
    seen, stack = set(), [src]
    while stack:
        node = stack.pop()
        if node == dst:
            return True
        if node not in seen:
            seen.add(node)
            stack.extend(edges.get(node, ()))
    return False
