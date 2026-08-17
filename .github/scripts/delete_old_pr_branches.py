"""Delete old branches that still point at the head of a closed pull request."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

RETENTION = timedelta(days=30)
NO_DELETE_LABEL = "no-delete-branch"
GHSTACK_BRANCH = re.compile(r"^(gh/.+)/(?:head|base|orig)$")
BranchGroupKey = tuple[str, str]


@dataclass(frozen=True)
class Branch:
    """A remote branch at the object observed during discovery."""

    name: str
    oid: str
    last_commit_at: datetime


@dataclass(frozen=True)
class BranchGroup:
    """Branches that must be retained or deleted together."""

    key: BranchGroupKey
    branches: tuple[Branch, ...]

    @property
    def name(self) -> str:
        return self.key[1]

    @property
    def last_commit_at(self) -> datetime:
        return max(branch.last_commit_at for branch in self.branches)


@dataclass(frozen=True)
class PullRequest:
    """Pull request state needed to decide whether its branch can be deleted."""

    number: int
    head_ref: str
    head_sha: str
    head_repo: str | None
    base_ref: str
    is_open: bool
    updated_at: datetime
    labels: frozenset[str]


@dataclass(frozen=True)
class DeletionCandidate:
    """A branch group and the closed pull request that authorizes its deletion."""

    group: BranchGroup
    pull_request: PullRequest


def branch_group_key(branch: str) -> BranchGroupKey:
    """Return a collision-free key for a ghstack family or ordinary branch."""
    match = GHSTACK_BRANCH.match(branch)
    return ("ghstack", match.group(1)) if match else ("branch", branch)


def group_branches(branches: Iterable[Branch]) -> list[BranchGroup]:
    """Group ghstack refs while leaving ordinary branches independent."""
    grouped: dict[BranchGroupKey, list[Branch]] = {}
    for branch in branches:
        grouped.setdefault(branch_group_key(branch.name), []).append(branch)
    return [
        BranchGroup(key, tuple(sorted(members, key=lambda branch: branch.name)))
        for key, members in sorted(grouped.items())
    ]


def select_deletions(
    groups: Iterable[BranchGroup],
    pull_requests: Iterable[PullRequest],
    repository: str,
    now: datetime,
) -> list[DeletionCandidate]:
    """Select old closed-PR groups without touching active or unrelated refs."""
    latest_closed: dict[BranchGroupKey, PullRequest] = {}
    retained: set[BranchGroupKey] = set()

    for pull_request in pull_requests:
        if pull_request.is_open:
            # Base refs always belong to this repository; heads may belong to forks.
            retained.add(branch_group_key(pull_request.base_ref))
        if pull_request.head_repo != repository:
            continue
        key = branch_group_key(pull_request.head_ref)
        if pull_request.is_open or NO_DELETE_LABEL in pull_request.labels:
            retained.add(key)
        elif key not in latest_closed or pull_request.number > latest_closed[key].number:
            latest_closed[key] = pull_request

    deletions = []
    for group in groups:
        pull_request = latest_closed.get(group.key)
        if pull_request is None or group.key in retained:
            continue
        if now - max(group.last_commit_at, pull_request.updated_at) < RETENTION:
            continue
        observed = {(branch.name, branch.oid) for branch in group.branches}
        if (pull_request.head_ref, pull_request.head_sha) not in observed:
            continue
        deletions.append(DeletionCandidate(group, pull_request))
    return deletions


def run(*args: str) -> str:
    """Run a command and return its standard output."""
    return subprocess.run(args, check=True, text=True, capture_output=True).stdout


def load_branches() -> list[Branch]:
    """Load remote branches, object IDs, and commit timestamps from the checkout."""
    output = run(
        "git",
        "for-each-ref",
        "--format=%(refname:strip=3)\t%(objectname)\t%(committerdate:unix)",
        "refs/remotes/origin",
    )
    branches = []
    for line in output.splitlines():
        name, oid, timestamp = line.split("\t")
        if name != "HEAD":
            branches.append(
                Branch(name, oid, datetime.fromtimestamp(int(timestamp), timezone.utc))
            )
    return branches


def parse_pull_request(payload: dict) -> PullRequest:
    """Convert a GitHub pull request response into cleanup policy state."""
    head, base = payload["head"], payload["base"]
    return PullRequest(
        number=payload["number"],
        head_ref=head["ref"],
        head_sha=head["sha"],
        head_repo=head["repo"]["full_name"] if head["repo"] else None,
        base_ref=base["ref"],
        is_open=payload["state"] == "open",
        updated_at=datetime.fromisoformat(payload["updated_at"].replace("Z", "+00:00")),
        labels=frozenset(label["name"] for label in payload["labels"]),
    )


def load_pull_requests(repository: str) -> list[PullRequest]:
    """Load every pull request, following each REST API page."""
    endpoint = f"repos/{repository}/pulls?state=all&per_page=100"
    pages = json.loads(run("gh", "api", "--paginate", "--slurp", endpoint))
    return [parse_pull_request(payload) for page in pages for payload in page]


def is_protected(repository: str, branch: str) -> bool:
    """Return whether a branch is protected, failing closed on API errors."""
    try:
        response = run("gh", "api", f"repos/{repository}/branches/{quote(branch, safe='')}")
    except subprocess.CalledProcessError as error:
        print(f"[{branch}] Could not check branch protection: {error.stderr}")
        return True
    return bool(json.loads(response)["protected"])


def group_is_protected(repository: str, group: BranchGroup) -> bool:
    """Return whether any branch in a group is protected."""
    return any(is_protected(repository, branch.name) for branch in group.branches)


def refresh_candidate(
    candidate: DeletionCandidate, repository: str, now: datetime
) -> DeletionCandidate | None:
    """Recheck protection and freshly fetched pull request state before deleting."""
    if group_is_protected(repository, candidate.group):
        return None
    refreshed = select_deletions(
        [candidate.group], load_pull_requests(repository), repository, now
    )
    return refreshed[0] if refreshed else None


def delete_group(group: BranchGroup) -> None:
    """Atomically delete a group only if every remote ref still has its observed OID."""
    leases = [
        f"--force-with-lease=refs/heads/{branch.name}:{branch.oid}" for branch in group.branches
    ]
    deletions = [f":refs/heads/{branch.name}" for branch in group.branches]
    subprocess.run(["git", "push", "--atomic", *leases, "origin", *deletions], check=True)


def main() -> None:
    """Delete eligible branch groups or print them when running in dry-run mode."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repository = os.environ["GITHUB_REPOSITORY"]
    now = datetime.now(timezone.utc)
    groups = group_branches(load_branches())
    candidates = select_deletions(groups, load_pull_requests(repository), repository, now)

    found_deletion = False
    for candidate in candidates:
        if args.dry_run:
            refreshed = None if group_is_protected(repository, candidate.group) else candidate
        else:
            refreshed = refresh_candidate(candidate, repository, now)
        if refreshed is None:
            print(f"[{candidate.group.name}] State changed; skipping")
            continue

        found_deletion = True
        action = "Would delete" if args.dry_run else "Deleting"
        for branch in refreshed.group.branches:
            print(
                f"[{refreshed.group.name}] {action} {branch.name} "
                f"for closed PR #{refreshed.pull_request.number}"
            )
        if not args.dry_run:
            delete_group(refreshed.group)

    if not found_deletion:
        print("No old pull request branches to delete")


if __name__ == "__main__":
    main()
