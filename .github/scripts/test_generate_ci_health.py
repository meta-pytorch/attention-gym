"""Test CI health aggregation with only the Python standard library."""

import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from generate_ci_health import (
    build_dashboard,
    duration_seconds,
    extract_pytest_failures,
    percentile,
    summarize_jobs,
    summarize_runs,
)

NOW = datetime(2026, 8, 17, tzinfo=timezone.utc)
REPOSITORY = "meta-pytorch/attention-gym"


def workflow_run(
    run_id: int,
    conclusion: str,
    started_at: str = "2026-08-17T10:00:00Z",
    updated_at: str = "2026-08-17T10:02:00Z",
) -> dict:
    """Build a representative completed workflow run payload."""
    return {
        "id": run_id,
        "name": "Run Tests",
        "display_title": f"Run {run_id}",
        "event": "pull_request",
        "status": "completed",
        "conclusion": conclusion,
        "created_at": started_at,
        "run_started_at": started_at,
        "updated_at": updated_at,
        "head_branch": "feature",
        "head_sha": "1234567890",
        "html_url": f"https://example.com/runs/{run_id}",
    }


def job(job_id: int, name: str, conclusion: str, duration_minutes: int) -> dict:
    """Build a representative completed job payload."""
    return {
        "id": job_id,
        "name": name,
        "status": "completed",
        "conclusion": conclusion,
        "started_at": "2026-08-17T10:00:00Z",
        "completed_at": f"2026-08-17T10:{duration_minutes:02d}:00Z",
        "html_url": f"https://example.com/jobs/{job_id}",
        "steps": [{"name": "pytest", "conclusion": conclusion}],
    }


class TestMetricHelpers(unittest.TestCase):
    def test_duration_and_percentile(self) -> None:
        self.assertEqual(duration_seconds("2026-08-17T10:00:00Z", "2026-08-17T10:02:30Z"), 150)
        self.assertEqual(percentile([1, 2, 3, 4, 5], 0.95), 4.8)
        self.assertIsNone(percentile([], 0.95))

    def test_extracts_unique_pytest_node_ids(self) -> None:
        log = """
FAILED test/test_masks.py::test_causal - AssertionError
ERROR test/test_kda.py::test_backward - RuntimeError
FAILED test/test_masks.py::test_causal - AssertionError
"""
        self.assertEqual(
            extract_pytest_failures(log),
            ["test/test_masks.py::test_causal", "test/test_kda.py::test_backward"],
        )


class TestAggregation(unittest.TestCase):
    def test_summarizes_runs_without_counting_cancelled_runs(self) -> None:
        runs = [
            workflow_run(3, "success", updated_at="2026-08-17T10:01:00Z"),
            workflow_run(2, "failure", updated_at="2026-08-17T10:03:00Z"),
            workflow_run(1, "cancelled", updated_at="2026-08-17T10:09:00Z"),
        ]

        summary = summarize_runs(runs)

        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["successes"], 1)
        self.assertEqual(summary["failures"], 1)
        self.assertEqual(summary["ignored"], 1)
        self.assertEqual(summary["success_rate"], 50)
        self.assertEqual(summary["median_duration_seconds"], 120)

    def test_summarizes_each_job_name(self) -> None:
        runs = [workflow_run(2, "success"), workflow_run(1, "failure")]
        jobs_by_run = {
            2: [job(20, "pytest", "success", 2)],
            1: [job(10, "pytest", "failure", 4)],
        }

        summaries = summarize_jobs("Run Tests", runs, jobs_by_run)

        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["success_rate"], 50)
        self.assertEqual(summaries[0]["median_duration_seconds"], 180)
        self.assertEqual(summaries[0]["p95_duration_seconds"], 234)

    @patch("generate_ci_health.run_command_for_log", return_value="FAILED test/test_ci.py::test_x")
    def test_builds_recent_failure_details(self, _log_mock) -> None:
        workflow = {
            "id": 7,
            "name": "Run Tests",
            "path": ".github/workflows/test.yml",
            "state": "active",
        }
        failed_run = workflow_run(10, "failure")
        failed_job = job(100, "pytest", "failure", 2)

        dashboard = build_dashboard(
            REPOSITORY,
            [workflow],
            {7: [failed_run]},
            {10: [failed_job]},
            NOW,
        )

        self.assertEqual(dashboard["overall"]["failures"], 1)
        self.assertEqual(dashboard["failures"][0]["jobs"][0]["failed_steps"], ["pytest"])
        self.assertEqual(dashboard["failures"][0]["jobs"][0]["tests"], ["test/test_ci.py::test_x"])


if __name__ == "__main__":
    unittest.main()
