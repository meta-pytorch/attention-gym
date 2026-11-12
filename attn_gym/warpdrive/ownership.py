"""Verify epoch ownership on channels whose epochs have owners.

Those are the channels with a round-robin side or exclusive consumers. Verification requires
acquires, waits and skips to be static per work item: every epoch must be filled once and waited
by exactly one consumer; the others skip it. Round-robin instance ``j`` of ``k`` takes epochs
``j, j + k, ...``: an owner that skips the other instances' epochs after each of its own. A
static violation raises; dynamic sequences stay unverified.

A verdict also names the first wait whose parity test is exact only behind a rendezvous. A
wait of epoch ``e`` is exact once epoch ``e - n`` was filled, which the waiter saw itself only
if its previous wait was of epoch ``e - n`` or later (``docs/warpdrive/channels.md``).
"""

from __future__ import annotations

from attn_gym.warpdrive.channels import SIDES, Sharing
from attn_gym.warpdrive.protocol import ProtocolError


def check(kernel, epoch_seqs: dict) -> dict[str, str]:
    """Each owned-epoch channel's verdict, by channel name. ``epoch_seqs`` maps
    ``(channel, role instance)`` to that role's ``RoleTrace.epoch_sequence``."""
    persistent = any(st.schedule.persistent for st in kernel._ws_streams.values())
    verdicts = {}
    for name, ch in kernel._ws_chans.items():
        if ch.owned_epochs:
            verdicts[name] = _epoch_ownership(name, ch, epoch_seqs, persistent)
    return verdicts


def _epoch_ownership(name: str, ch, epoch_seqs: dict, persistent: bool) -> str:
    """``check``'s verdict for one channel."""
    bound = {"producer": ch.producers, "consumer": ch.consumers}
    joined = [side for side in SIDES if ch.sharing(side) is Sharing.JOIN]
    if not joined:
        return "unverified (both sides take turns)"
    if len(bound[joined[0]]) != 1:
        return f"unverified (several joining {joined[0]}s)"
    names = {side: [ri.name for ri, _ in bound[side]] for side in SIDES}
    traced = {who: epoch_seqs[(name, who)] for who in names["producer"] + names["consumer"]}
    dynamic = [seq for seq in traced.values() if isinstance(seq, str)]
    if dynamic:
        return f"unverified ({dynamic[0]})"
    if len({per_item for per_item, _ in traced.values()} - {None}) > 1:
        return "unverified (some roles use it once per work item, others once per launch)"
    per_item = any(scope for scope, _ in traced.values())
    unit = "work item" if per_item else "launch"
    seqs = {who: seq for who, (_, seq) in traced.items()}
    continues = persistent and per_item  # each end carries on into the next work item

    def walk(side):
        """(who, first epoch, epochs per acquire/wait) of each instance on ``side``."""
        k = ch.turns(side)
        return [(who, j if k > 1 else 0, k) for j, who in enumerate(names[side])]

    # The producers' epochs: a round-robin instance's m-th acquire takes start + m * k.
    filled, retired, acquired = [], set(), {}
    for who, start, step in walk("producer"):
        acquires = [op for op, _ in seqs[who] if op == "acquire"]
        filled += [(start + m * step, who) for m in range(len(acquires))]
        # A retired acquire publishes nothing: its epoch has no owner, only skips.
        retired |= {start + order * step for op, order in seqs[who] if op == "retire"}
        acquired[who] = (start, step, len(acquires))
    total = len(filled)
    fillers = [[] for _ in range(total)]
    for e, who in filled:
        if e < total:
            fillers[e].append(who)
    producer = " and ".join(names["producer"])

    def verb(word):  # agrees with ``producer``
        return word if len(names["producer"]) > 1 else f"{word}s"

    gap = next(((e, f) for e, f in enumerate(fillers) if len(f) != 1), None)
    if gap is not None:  # round-robin producers whose trip counts do not add up
        e, f = gap
        who = "no producer" if not f else " and ".join(f)
        raise ProtocolError(
            f"channel {name}: epoch {e} of {total} per {unit} is acquired by {who}; "
            "round-robin producers take epochs j, j + k, ... in turn, so their acquires "
            "must add up to every epoch once"
        )
    if continues:
        for who, (start, step, count) in acquired.items():
            if step > 1 and count * step != total:
                raise ProtocolError(
                    f"channel {name}: {who} acquires {count} of the {total} epochs per "
                    f"{unit}; a round-robin producer continues into the next item, so each "
                    f"of the {step} instances must take {total} / {step}"
                )
    owners: list[list[str]] = [[] for _ in range(total)]
    waited = {who: [] for who in names["consumer"]}
    blind = None  # (consumer, epoch) of the first wait that needs a rendezvous
    for who, start, step in walk("consumer"):
        e, seen = start, -1  # the next epoch; the last one this consumer waited for
        for op, count in seqs[who]:
            if op == "wait":
                if e >= total or e in retired:
                    raise ProtocolError(
                        f"channel {name}: {who} waits on epoch {e}, but {producer} "
                        + (
                            f"{verb('retire')} it unpublished"
                            if e in retired
                            else f"{verb('fill')} only {total} per {unit}"
                        )
                    )
                owners[e].append(who)
                waited[who].append(e)
                if e - ch.n > seen and blind is None and not ch.named:
                    blind = (who, e)
                seen = e
                e += step
            else:  # a skip
                e += count
        # The next item's first wait follows this item's last one.
        mine = waited[who]
        if continues and mine and mine[0] + total - mine[-1] > ch.n and not ch.named:
            blind = blind or (who, mine[0] + total)
        # A persistent role continues into the next item: it must cover every epoch.
        if (step == 1 and e > total) or (continues and e - start != total):
            raise ProtocolError(
                f"channel {name}: {who} advances {e - start} epochs per {unit} ("
                + ("waits and skips" if step == 1 else f"{step} per wait")
                + f"), but {producer} {verb('acquire')} {total}"
            )
    bad = next(((e, o) for e, o in enumerate(owners) if len(o) != 1 and e not in retired), None)
    if bad is not None:
        e, o = bad
        who = "no consumer" if not o else " and ".join(o)
        rule = (
            "each epoch of an exclusive ring has exactly one owner, which the other "
            "consumers skip()"
            if ch.exclusive
            else "round-robin consumers take epochs j, j + k, ... in turn, so their waits "
            "must add up to every epoch once"
            if ch.turns("consumer") > 1
            else "the consumer must wait for every epoch the round-robin producers fill"
        )
        raise ProtocolError(
            f"channel {name}: epoch {e} of {total} per {unit} is waited by {who}; {rule}"
        )
    owned = "producer" if ch.turns("producer") > 1 else "consumer"
    counts = (
        {who: count for who, (_, _, count) in acquired.items()}
        if owned == "producer"
        else {who: len(mine) for who, mine in waited.items()}
    )
    share = ", ".join(f"{who} {count}" for who, count in counts.items())
    verdict = f"verified: {total - len(retired)} epochs per {unit} ({share})"
    if blind is not None:
        who, e = blind
        verdict += (
            f"; {who}'s wait of epoch {e} is exact only after epoch {e - ch.n} was "
            "filled, which it did not observe: order it with a rendezvous"
        )
    return verdict
