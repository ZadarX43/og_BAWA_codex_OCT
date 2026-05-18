#!/usr/bin/env python3
"""Run the public proof settlement publish command.

This command productizes the Results proof loop:

1. Read published website prediction JSON and available final result snapshots.
2. Settle picks into weekly and archive proof JSON.
3. Smoke-check the Results page data/rendering contract.

It does not generate predictions or alter deploy routing.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "latest" / "RESULTS_PUBLISH_RUN_REPORT.md"

COMMANDS = [
    {
        "name": "settle_published_results",
        "cmd": ["python3", "scripts/settle_published_results.py"],
        "why": "Grade published website picks against final provider/live result snapshots.",
    },
    {
        "name": "results_page_smoke",
        "cmd": ["python3", "scripts/smoke_results_page.py"],
        "why": "Confirm Results page data contract, market splits, and visual state classes.",
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_command(entry: dict[str, object]) -> dict[str, object]:
    completed = subprocess.run(
        entry["cmd"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "name": entry["name"],
        "cmd": " ".join(entry["cmd"]),
        "why": entry["why"],
        "returncode": completed.returncode,
        "ok": completed.returncode == 0,
        "output": (completed.stdout or "").strip(),
    }


def read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def write_report(results: list[dict[str, object]]) -> None:
    weekly = read_json(ROOT / "frontend" / "public" / "data" / "weekly_results.json")
    archive = read_json(ROOT / "frontend" / "public" / "data" / "results_archive.json")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Results Publish Run Report",
        "",
        f"Generated: `{utc_now()}`",
        "",
        "## Summary",
        "",
        f"- Checks: `{len(results)}`",
        f"- Passed: `{sum(1 for item in results if item['ok'])}`",
        f"- Failed: `{sum(1 for item in results if not item['ok'])}`",
        f"- Weekly window: `{weekly.get('period_start', '')}` to `{weekly.get('period_end', '')}`",
        f"- Weekly picks: `{weekly.get('total_picks', 0)}`",
        f"- Weekly settled/pending: `{weekly.get('settled_picks', 0)}` / `{weekly.get('pending_picks', 0)}`",
        f"- Weekly wins/losses/voids: `{weekly.get('wins', 0)}` / `{weekly.get('losses', 0)}` / `{weekly.get('voids', 0)}`",
        f"- Archive picks: `{archive.get('total_picks', 0)}`",
        "",
        "## Outputs",
        "",
        "- `frontend/public/data/weekly_results.json`",
        "- `frontend/public/data/results_archive.json`",
        "- `reports/latest/RESULTS_SETTLEMENT_REPORT.md`",
        "- `reports/latest/RESULTS_PAGE_SMOKE_REPORT.md`",
        "- `reports/latest/RESULTS_PUBLISH_RUN_REPORT.md`",
        "",
        "## Checks",
        "",
    ]
    for item in results:
        status = "PASS" if item["ok"] else "FAIL"
        lines.extend(
            [
                f"### {item['name']} - {status}",
                "",
                f"- Command: `{item['cmd']}`",
                f"- Purpose: {item['why']}",
                "",
                "```text",
                str(item["output"])[:6000] or "(no output)",
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Guardrails",
            "",
            "- This command settles already-published website prediction JSON only.",
            "- This command does not run model generation, deploy routing, or slip formatting.",
            "- Pending rows remain visible until final provider scores are available.",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    results = [run_command(entry) for entry in COMMANDS]
    write_report(results)
    payload = {
        "ok": all(item["ok"] for item in results),
        "checks": len(results),
        "passed": sum(1 for item in results if item["ok"]),
        "failed": [item["name"] for item in results if not item["ok"]],
        "report": str(REPORT_PATH.relative_to(ROOT)),
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
