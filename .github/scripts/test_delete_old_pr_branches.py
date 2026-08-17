"""Test the cleanup policy with only the Python standard library."""

import os
import subprocess
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from delete_old_pr_branches import (
    NO_DELETE_LABEL,
    Branch,
    BranchGroup,
    DeletionCandidate,
    PullRequest,
    branch_group_key,
    delete_group,
    group_branches,
    is_protected,
    parse_pull_request,
    refresh_candidate,
    select_deletions,
)

NOW = datetime(2026, 8, 17, tzinfo=timezone.utc)
OLD = NOW - timedelta(days=60)
RECENT = NOW - timedelta(days=10)
REPOSITORY = "meta-pytorch/attention-gym"


def branch(name: str = "feature", oid: str = "feature-oid", at: datetime = OLD) -> Branch:
    return Branch(name, oid, at)


def group(*branches: Branch) -> BranchGroup:
    """Build the group that ``group_branches`` produces for these branches."""
    return BranchGroup(branch_group_key(branches[0].name), branches)


def select(groups: list[BranchGroup], pull_requests: list[PullRequest]) -> list[DeletionCandidate]:
    """Apply the cleanup policy to this repository at NOW."""
    return select_deletions(groups, pull_requests, REPOSITORY, NOW)


CLOSED_PR = PullRequest(
    number=1,
    head_ref="feature",
    head_sha="feature-oid",
    head_repo=REPOSITORY,
    base_ref="main",
    is_open=False,
    updated_at=OLD,
    labels=frozenset(),
)
FEATURE = group(branch())
GHSTACK = group(
    branch("gh/user/1/base", "base-oid"),
    branch("gh/user/1/head", "head-oid"),
    branch("gh/user/1/orig", "orig-oid"),
)
GHSTACK_HEAD_PR = replace(CLOSED_PR, head_ref="gh/user/1/head", head_sha="head-oid")


class TestBranchGrouping(unittest.TestCase):
    def test_branch_group_keys(self) -> None:
        """ghstack refs share one key; every other branch keeps its own."""
        expected = {
            "gh/user/1/base": ("ghstack", "gh/user/1"),
            "gh/user/1/head": ("ghstack", "gh/user/1"),
            "gh/user/1/orig": ("ghstack", "gh/user/1"),
            # A branch may share a ghstack family prefix without joining the family.
            "gh/user/1": ("branch", "gh/user/1"),
            "user/stack/1": ("branch", "user/stack/1"),
        }

        self.assertEqual({name: branch_group_key(name) for name in expected}, expected)

    def test_groups_ghstack_family_only(self) -> None:
        groups = group_branches(
            [
                branch("gh/user/1/head", "head-oid", RECENT),
                branch("gh/user/1/base", "base-oid"),
                branch("gh/user/1"),
                branch(),
            ]
        )

        self.assertEqual(
            [(item.key, [member.name for member in item.branches]) for item in groups],
            [
                (("branch", "feature"), ["feature"]),
                (("branch", "gh/user/1"), ["gh/user/1"]),
                (("ghstack", "gh/user/1"), ["gh/user/1/base", "gh/user/1/head"]),
            ],
        )


class TestSelectDeletions(unittest.TestCase):
    def test_selects_old_closed_pr_branch_at_its_reviewed_sha(self) -> None:
        candidate = DeletionCandidate(FEATURE, CLOSED_PR)
        self.assertEqual(select([FEATURE], [CLOSED_PR]), [candidate])

    def test_selects_closed_ghstack_family_atomically(self) -> None:
        candidate = DeletionCandidate(GHSTACK, GHSTACK_HEAD_PR)
        self.assertEqual(select([GHSTACK], [GHSTACK_HEAD_PR]), [candidate])

    def test_pull_request_retention_rules(self) -> None:
        """Every pull request state that keeps an otherwise deletable branch alive."""
        cases: dict[str, list[PullRequest]] = {
            "branch moved off the reviewed sha": [replace(CLOSED_PR, head_sha="other-oid")],
            "newer pr moved the head despite an older update time": [
                CLOSED_PR,
                replace(
                    CLOSED_PR,
                    number=2,
                    head_sha="new-oid",
                    updated_at=OLD - timedelta(days=10),
                ),
            ],
            "open pr head": [replace(CLOSED_PR, is_open=True)],
            "open pr base": [
                CLOSED_PR,
                replace(CLOSED_PR, head_ref="child", base_ref="feature", is_open=True),
            ],
            "recently updated pr": [replace(CLOSED_PR, updated_at=RECENT)],
            "no-delete label": [replace(CLOSED_PR, labels=frozenset({NO_DELETE_LABEL}))],
            "no pull request": [],
            "fork head": [replace(CLOSED_PR, head_repo="contributor/attention-gym")],
            "deleted fork head": [replace(CLOSED_PR, head_repo=None)],
        }

        for reason, pull_requests in cases.items():
            with self.subTest(reason=reason):
                self.assertEqual(select([FEATURE], pull_requests), [])

    def test_retains_recently_pushed_branch(self) -> None:
        self.assertEqual(select([group(branch(at=RECENT))], [CLOSED_PR]), [])

    def test_retains_ghstack_family_with_recently_pushed_sibling(self) -> None:
        family = group(
            branch("gh/user/1/base", "base-oid"), branch("gh/user/1/head", "head-oid", RECENT)
        )
        self.assertEqual(select([family], [GHSTACK_HEAD_PR]), [])


class TestPullRequestParsing(unittest.TestCase):
    def test_parses_api_payload(self) -> None:
        payload = {
            "number": 7,
            "state": "closed",
            "updated_at": "2026-06-18T02:30:00Z",
            "labels": [{"name": NO_DELETE_LABEL}],
            "head": {"ref": "feature", "sha": "feature-oid", "repo": {"full_name": REPOSITORY}},
            "base": {"ref": "main"},
        }

        self.assertEqual(
            parse_pull_request(payload),
            replace(
                CLOSED_PR,
                number=7,
                updated_at=datetime(2026, 6, 18, 2, 30, tzinfo=timezone.utc),
                labels=frozenset({NO_DELETE_LABEL}),
            ),
        )

        payload["head"]["repo"] = None
        self.assertIsNone(parse_pull_request(payload).head_repo)


class TestDeletionSafety(unittest.TestCase):
    @patch("delete_old_pr_branches.load_pull_requests")
    @patch("delete_old_pr_branches.is_protected", return_value=False)
    def test_refresh_rejects_new_open_base_dependency(
        self, _is_protected_mock, load_pull_requests_mock
    ) -> None:
        candidate = DeletionCandidate(FEATURE, CLOSED_PR)
        child = replace(
            CLOSED_PR,
            number=2,
            head_ref="child",
            head_sha="child-oid",
            base_ref="feature",
            is_open=True,
        )
        load_pull_requests_mock.return_value = [CLOSED_PR, child]

        self.assertIsNone(refresh_candidate(candidate, REPOSITORY, NOW))

    def test_changed_ref_rejects_entire_atomic_deletion(self) -> None:
        """git must reject the whole push when one lease is stale."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        work = directory / "work"
        self.git("init", "--bare", "remote.git", cwd=directory)
        self.git("init", "work", cwd=directory)
        self.git("remote", "add", "origin", str(directory / "remote.git"), cwd=work)
        self.git("commit", "--allow-empty", "-m", "reviewed", cwd=work)
        reviewed_oid = self.git("rev-parse", "HEAD", cwd=work).stdout.strip()
        head = "HEAD:refs/heads/gh/user/1/head"
        self.git("push", "origin", "HEAD:refs/heads/gh/user/1/base", head, cwd=work)
        self.git("commit", "--allow-empty", "-m", "pushed after discovery", cwd=work)
        self.git("push", "origin", head, cwd=work)
        stale = group(
            branch("gh/user/1/base", reviewed_oid), branch("gh/user/1/head", reviewed_oid)
        )

        self.addCleanup(os.chdir, Path.cwd())
        os.chdir(work)
        with self.assertRaises(subprocess.CalledProcessError):
            delete_group(stale)

        refs = self.git("ls-remote", "--heads", "origin", cwd=work).stdout
        self.assertIn("refs/heads/gh/user/1/base", refs)
        self.assertIn("refs/heads/gh/user/1/head", refs)

    @staticmethod
    def git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
        identity = ("-c", "user.name=cleanup", "-c", "user.email=cleanup@example.com")
        return subprocess.run(
            ["git", *identity, *args], cwd=cwd, check=True, capture_output=True, text=True
        )

    @patch("delete_old_pr_branches.run")
    def test_protection_lookup_fails_closed(self, run_mock) -> None:
        run_mock.side_effect = subprocess.CalledProcessError(
            1, ["gh", "api"], stderr="API failure"
        )

        self.assertTrue(is_protected(REPOSITORY, "feature"))


if __name__ == "__main__":
    unittest.main()
