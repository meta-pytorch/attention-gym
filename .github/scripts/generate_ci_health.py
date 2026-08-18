"""Build the static data consumed by the documentation site's CI health dashboard."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

EXCLUDED_WORKFLOW_PATHS = {".github/workflows/claude-code.yml"}
FAILURE_CONCLUSIONS = {"failure", "startup_failure", "stale", "timed_out"}
IGNORED_CONCLUSIONS = {"action_required", "cancelled", "neutral", "skipped"}
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PYTEST_FAILURE = re.compile(r"(?:FAILED|ERROR)\s+([^\s]+::[^\s]+)")


def run(*args: str) -> str:
    """Run a command and return its standard output."""
    return subprocess.run(args, check=True, text=True, capture_output=True).stdout


def github_api(endpoint: str) -> object:
    """Load one GitHub REST API response through the authenticated gh CLI."""
    return json.loads(run("gh", "api", endpoint))


def parse_time(value: str | None) -> datetime | None:
    """Parse a GitHub timestamp, preserving absence for unstarted jobs."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def duration_seconds(started_at: str | None, completed_at: str | None) -> float | None:
    """Return elapsed seconds when both GitHub timestamps are present."""
    start, end = parse_time(started_at), parse_time(completed_at)
    return (end - start).total_seconds() if start and end else None


def percentile(values: list[float], fraction: float) -> float | None:
    """Return a linearly interpolated percentile for a small metric sample."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def extract_pytest_failures(log: str) -> list[str]:
    """Extract unique pytest node IDs from a GitHub Actions job log."""
    clean_log = ANSI_ESCAPE.sub("", log)
    return list(dict.fromkeys(PYTEST_FAILURE.findall(clean_log)))[:12]


def conclusion_kind(conclusion: str | None) -> str:
    """Map GitHub conclusions to the dashboard's success/failure/ignored groups."""
    if conclusion == "success":
        return "success"
    if conclusion in FAILURE_CONCLUSIONS:
        return "failure"
    return "ignored"


def summarize_runs(runs: list[dict]) -> dict:
    """Summarize completed workflow runs and preserve a compact recent timeline."""
    completed = [run for run in runs if run.get("status") == "completed"]
    counted = [run for run in completed if conclusion_kind(run.get("conclusion")) != "ignored"]
    successes = sum(run.get("conclusion") == "success" for run in counted)
    failures = len(counted) - successes
    durations = [
        duration
        for item in counted
        if (duration := duration_seconds(item.get("run_started_at"), item.get("updated_at")))
        is not None
    ]
    latest = counted[0] if counted else None
    success_rate = successes / len(counted) * 100 if counted else None

    return {
        "total": len(counted),
        "successes": successes,
        "failures": failures,
        "ignored": len(completed) - len(counted),
        "success_rate": success_rate,
        "median_duration_seconds": median(durations) if durations else None,
        "p95_duration_seconds": percentile(durations, 0.95),
        "latest": compact_run(latest) if latest else None,
        "timeline": [compact_run(run) for run in completed[:20]],
    }


def compact_run(run: dict) -> dict:
    """Keep only fields required by the browser dashboard."""
    return {
        "id": run["id"],
        "title": run.get("display_title") or run.get("name"),
        "event": run.get("event"),
        "branch": run.get("head_branch"),
        "sha": (run.get("head_sha") or "")[:7],
        "created_at": run.get("created_at"),
        "conclusion": run.get("conclusion"),
        "kind": conclusion_kind(run.get("conclusion")),
        "duration_seconds": duration_seconds(run.get("run_started_at"), run.get("updated_at")),
        "url": run.get("html_url"),
    }


def summarize_jobs(
    workflow_name: str, runs: list[dict], jobs_by_run: dict[int, list[dict]]
) -> list[dict]:
    """Aggregate duration and success statistics for each job name in a workflow."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for workflow_run in runs:
        for job in jobs_by_run.get(workflow_run["id"], []):
            grouped[job["name"]].append(job)

    summaries = []
    for job_name, jobs in grouped.items():
        completed = [job for job in jobs if job.get("status") == "completed"]
        counted = [job for job in completed if conclusion_kind(job.get("conclusion")) != "ignored"]
        durations = [
            duration
            for job in counted
            if (duration := duration_seconds(job.get("started_at"), job.get("completed_at")))
            is not None
        ]
        successes = sum(job.get("conclusion") == "success" for job in counted)
        latest = completed[0] if completed else None
        summaries.append(
            {
                "workflow": workflow_name,
                "name": job_name,
                "total": len(counted),
                "success_rate": successes / len(counted) * 100 if counted else None,
                "median_duration_seconds": median(durations) if durations else None,
                "p95_duration_seconds": percentile(durations, 0.95),
                "latest_conclusion": latest.get("conclusion") if latest else None,
                "latest_url": latest.get("html_url") if latest else None,
            }
        )
    return sorted(summaries, key=lambda item: (item["workflow"], item["name"]))


def failed_run_details(repository: str, run: dict, workflow_name: str, jobs: list[dict]) -> dict:
    """Describe failed jobs, steps, and pytest node IDs for one workflow run."""
    failed_jobs = []
    for job in jobs:
        if conclusion_kind(job.get("conclusion")) != "failure":
            continue
        failed_steps = [
            step["name"]
            for step in job.get("steps", [])
            if conclusion_kind(step.get("conclusion")) == "failure"
        ]
        try:
            log = run_command_for_log(repository, job["id"])
        except subprocess.CalledProcessError:
            log = ""
        failed_jobs.append(
            {
                "name": job["name"],
                "url": job.get("html_url"),
                "duration_seconds": duration_seconds(
                    job.get("started_at"), job.get("completed_at")
                ),
                "failed_steps": failed_steps,
                "tests": extract_pytest_failures(log),
            }
        )

    return {
        "workflow": workflow_name,
        "title": run.get("display_title") or run.get("name"),
        "branch": run.get("head_branch"),
        "sha": (run.get("head_sha") or "")[:7],
        "created_at": run.get("created_at"),
        "url": run.get("html_url"),
        "jobs": failed_jobs,
    }


def run_command_for_log(repository: str, job_id: int) -> str:
    """Download a failed job's text log without treating it as JSON."""
    return run("gh", "api", f"repos/{repository}/actions/jobs/{job_id}/logs")


def build_dashboard(
    repository: str,
    workflows: list[dict],
    runs_by_workflow: dict[int, list[dict]],
    jobs_by_run: dict[int, list[dict]],
    now: datetime,
) -> dict:
    """Build the complete JSON payload from fetched GitHub workflow data."""
    workflow_summaries = []
    job_summaries = []
    failures = []
    named_runs = []

    for workflow in workflows:
        workflow_runs = runs_by_workflow.get(workflow["id"], [])
        named_runs.extend((run, workflow["name"]) for run in workflow_runs)
        workflow_summaries.append(
            {
                "name": workflow["name"],
                "path": workflow["path"],
                "state": workflow["state"],
                **summarize_runs(workflow_runs),
            }
        )
        job_summaries.extend(summarize_jobs(workflow["name"], workflow_runs, jobs_by_run))

    failed_runs = sorted(
        (
            (run, workflow_name)
            for run, workflow_name in named_runs
            if conclusion_kind(run.get("conclusion")) == "failure"
        ),
        key=lambda item: item[0].get("created_at") or "",
        reverse=True,
    )[:10]
    for workflow_run, workflow_name in failed_runs:
        failures.append(
            failed_run_details(
                repository,
                workflow_run,
                workflow_name,
                jobs_by_run.get(workflow_run["id"], []),
            )
        )

    overall = summarize_runs(
        sorted(
            (run for run, _ in named_runs),
            key=lambda run: run.get("created_at") or "",
            reverse=True,
        )
    )
    return {
        "repository": repository,
        "generated_at": now.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "runs_per_workflow": max((len(runs) for runs in runs_by_workflow.values()), default=0),
        "overall": overall,
        "workflows": workflow_summaries,
        "jobs": job_summaries,
        "failures": failures,
    }


def load_dashboard(repository: str, runs_per_workflow: int, now: datetime) -> dict:
    """Fetch active and disabled workflows plus their recent runs and jobs."""
    response = github_api(f"repos/{repository}/actions/workflows?per_page=100")
    workflows = [
        workflow
        for workflow in response["workflows"]
        if workflow["path"] not in EXCLUDED_WORKFLOW_PATHS
    ]
    runs_by_workflow = {}
    jobs_by_run = {}

    for workflow in workflows:
        response = github_api(
            f"repos/{repository}/actions/workflows/{workflow['id']}/runs?per_page={runs_per_workflow}"
        )
        workflow_runs = response["workflow_runs"]
        runs_by_workflow[workflow["id"]] = workflow_runs
        for workflow_run in workflow_runs:
            if workflow_run.get("status") != "completed":
                continue
            response = github_api(
                f"repos/{repository}/actions/runs/{workflow_run['id']}/jobs?per_page=100"
            )
            jobs_by_run[workflow_run["id"]] = response["jobs"]

    return build_dashboard(repository, workflows, runs_by_workflow, jobs_by_run, now)


def main() -> None:
    """Generate the CI health JSON file used by the documentation site."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--output", type=Path, default=Path("docs/assets/ci-health.json"))
    parser.add_argument("--runs-per-workflow", type=int, default=12)
    args = parser.parse_args()
    if not args.repository:
        parser.error("--repository or GITHUB_REPOSITORY is required")

    dashboard = load_dashboard(args.repository, args.runs_per_workflow, datetime.now(timezone.utc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dashboard, indent=2) + "\n")
    print(f"Wrote CI health data to {args.output}")


if __name__ == "__main__":
    main()
