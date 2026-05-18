#!/usr/bin/env python3
"""Run the website/product launch smoke suite."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "latest" / "LAUNCH_SMOKE_REPORT.md"

COMMANDS = [
    {
        "name": "frontend_js_syntax",
        "cmd": ["node", "--check", "frontend/assets/app.js"],
        "why": "The frontend bundle parses before browser smoke.",
    },
    {
        "name": "publish_results_proof",
        "cmd": ["python3", "scripts/publish_results_proof.py"],
        "why": "Published picks are settled into weekly/archive proof JSON and the Results page proof contract is smoked.",
    },
    {
        "name": "site_publish_smoke",
        "cmd": ["python3", "scripts/site_publish_smoke.py"],
        "why": "Website-safe JSON has required fields, proof states, and no blocked leaks.",
    },
    {
        "name": "frontend_static_smoke",
        "cmd": ["python3", "scripts/smoke_frontend_static.py"],
        "why": "Static HTML, referenced assets, and top-level JSON parse cleanly.",
    },
    {
        "name": "worker_account_checkout_smoke",
        "cmd": ["node", "worker/test_worker_local.js"],
        "why": "Worker auth, premium gating, Stripe Checkout, portal, and payment states pass.",
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
    output = (completed.stdout or "").strip()
    return {
        "name": entry["name"],
        "cmd": " ".join(entry["cmd"]),
        "why": entry["why"],
        "returncode": completed.returncode,
        "ok": completed.returncode == 0,
        "output": output,
    }


def write_report(results: list[dict[str, object]]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Launch Smoke Report",
        "",
        f"Generated: `{utc_now()}`",
        "",
        "## Summary",
        "",
        f"- Checks: `{len(results)}`",
        f"- Passed: `{sum(1 for item in results if item['ok'])}`",
        f"- Failed: `{sum(1 for item in results if not item['ok'])}`",
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
