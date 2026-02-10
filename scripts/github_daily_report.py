#!/usr/bin/env python3
"""Generate a daily GitHub stars/issues report for a repository."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

GITHUB_API = "https://api.github.com"


def github_get(path: str, token: str | None = None, query: dict[str, Any] | None = None) -> tuple[Any, dict[str, str]]:
    params = f"?{urlencode(query)}" if query else ""
    url = f"{GITHUB_API}{path}{params}"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "vital-series-daily-reporter",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = Request(url, headers=headers)
    try:
        with urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            return payload, dict(resp.headers.items())
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"GitHub API request failed: {exc.code} {exc.reason} {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"GitHub API request failed: {exc.reason}") from exc


def collect_recent_issues(owner: str, repo: str, since_iso: str, token: str | None = None) -> list[dict[str, Any]]:
    page = 1
    per_page = 100
    results: list[dict[str, Any]] = []

    while True:
        data, _ = github_get(
            f"/repos/{owner}/{repo}/issues",
            token=token,
            query={
                "state": "all",
                "since": since_iso,
                "per_page": per_page,
                "page": page,
                "sort": "updated",
                "direction": "desc",
            },
        )

        if not data:
            break

        for item in data:
            if "pull_request" in item:
                continue
            results.append(item)

        if len(data) < per_page:
            break
        page += 1

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a 24-hour GitHub stars/issues report")
    parser.add_argument("--repo", default="jzhws1/VITAL-Series", help="GitHub repo in owner/name format")
    parser.add_argument("--state-file", default="reports/.github_daily_state.json", help="Path to persistent state json")
    parser.add_argument("--report-file", default="reports/github_daily_report.md", help="Path to markdown report output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if "/" not in args.repo:
        raise SystemExit("--repo must be in owner/name format")

    owner, repo = args.repo.split("/", 1)
    token = os.getenv("GITHUB_TOKEN")
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=24)
    since_iso = since.isoformat().replace("+00:00", "Z")

    repo_info, headers = github_get(f"/repos/{owner}/{repo}", token=token)
    stars_now = int(repo_info["stargazers_count"])
    open_issues_now = int(repo_info["open_issues_count"])

    recent = collect_recent_issues(owner, repo, since_iso, token=token)
    created = []
    closed = []
    for issue in recent:
        created_at = datetime.fromisoformat(issue["created_at"].replace("Z", "+00:00"))
        closed_at = issue.get("closed_at")
        if created_at >= since:
            created.append(issue)
        if closed_at:
            closed_at_dt = datetime.fromisoformat(closed_at.replace("Z", "+00:00"))
            if closed_at_dt >= since:
                closed.append(issue)

    state_path = Path(args.state_file)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    previous = {}
    if state_path.exists():
        previous = json.loads(state_path.read_text(encoding="utf-8"))

    stars_prev = previous.get("stars")
    open_issues_prev = previous.get("open_issues")

    stars_delta = stars_now - stars_prev if isinstance(stars_prev, int) else None
    open_issues_delta = open_issues_now - open_issues_prev if isinstance(open_issues_prev, int) else None

    lines = [
        "# GitHub Daily Report",
        "",
        f"- Repository: `{owner}/{repo}`",
        f"- Generated at (UTC): `{now.isoformat().replace('+00:00', 'Z')}`",
        f"- Window: last 24 hours since `{since_iso}`",
        "",
        "## Snapshot",
        f"- Stars: **{stars_now}**" + (f" ({stars_delta:+d} vs previous run)" if stars_delta is not None else " (first run)") ,
        f"- Open issues: **{open_issues_now}**" + (f" ({open_issues_delta:+d} vs previous run)" if open_issues_delta is not None else " (first run)"),
        "",
        "## Issue activity in last 24 hours",
        f"- Newly created issues: **{len(created)}**",
        f"- Newly closed issues: **{len(closed)}**",
        "",
    ]

    if created:
        lines.append("### Created")
        for issue in created:
            lines.append(f"- #{issue['number']} [{issue['title']}]({issue['html_url']}) · created `{issue['created_at']}`")
        lines.append("")

    if closed:
        lines.append("### Closed")
        for issue in closed:
            lines.append(f"- #{issue['number']} [{issue['title']}]({issue['html_url']}) · closed `{issue['closed_at']}`")
        lines.append("")

    rate_remain = headers.get("X-RateLimit-Remaining")
    rate_reset = headers.get("X-RateLimit-Reset")
    if rate_remain or rate_reset:
        lines += [
            "## API quota",
            f"- X-RateLimit-Remaining: `{rate_remain}`",
            f"- X-RateLimit-Reset: `{rate_reset}`",
            "",
        ]

    report_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    state = {
        "repo": f"{owner}/{repo}",
        "generated_at": now.isoformat().replace("+00:00", "Z"),
        "stars": stars_now,
        "open_issues": open_issues_now,
    }
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote report: {report_path}")
    print(f"Wrote state:  {state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
