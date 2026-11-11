"""Export kernel roles and channels as a profiler ``PipelinePlan`` with trace-matching names."""

from __future__ import annotations

from attn_gym.warpdrive.describe import is_pair, rank_label
from attn_gym.warpdrive.profiler.pipeline.plan import (
    Dependency,
    PipelinePlan,
    Region,
    Resource,
    Role,
)


def plan_from_warpdrive(kernel, *, cta: int = 0, name: str | None = None) -> PipelinePlan:
    """Build a ``PipelinePlan`` for the roles and handoffs on CTA rank ``cta``.

    Names match both trace backends. Each role instance has one ``Role`` and ``Region``;
    each channel between distinct roles has one ``Resource``.

    Data edges run producer -> consumer at distance 0. An edge closing a same-item cycle
    (such as FA4's S/P handoff) instead has distance 1, kind ``state`` and label ``(feedback)``.
    Reuse edges run from the last consumers in data order back to each producer. Their
    distance is at least 1, using ``slots // uses_per_item`` for static traced counts and
    1 for inner-loop uses. Capacity is the stage count, or 1 for a role bound to a Slots slot.
    Work rings add one to the distance: roles release a slot on receipt, one item before
    the holding region ends.

    Before compilation, each channel is assumed to be used once per item. Compile the
    kernel to obtain traced reuse distances.
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
    uses = kernel._ws_uses
    work_rings = {f"{stream}_ring" for stream in kernel._ws_streams}
    for cname, (ch, producers, consumers) in channels.items():
        bound = {ri.name for ri, slot in [*ch.producers, *ch.consumers] if slot is not None}
        for prod in producers:
            others = [c for c in consumers if c != prod]
            # Reuse waits on the join's last arrivals in data order.
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
