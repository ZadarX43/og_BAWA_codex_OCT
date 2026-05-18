#!/usr/bin/env python3
"""Smoke-check the public Results proof surface.

This is intentionally a static/data smoke. It verifies that the public Results
page can render the settlement contract before a preview publish.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
DATA_ROOT = FRONTEND / "public" / "data"
REPORT_PATH = ROOT / "reports" / "latest" / "RESULTS_PAGE_SMOKE_REPORT.md"

REQUIRED_MARKETS = ["FTR", "BTTS", "OU25", "TG1.5"]
VALID_STATES = {"won", "lost", "void", "pending"}
STATE_CLASSES = {
    "won": "public-result-row-won",
    "lost": "public-result-row-lost",
    "void": "public-result-row-void",
    "pending": "public-result-row-pending",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def rows_from(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return [row for row in payload["items"] if isinstance(row, dict)]
    return []


def main() -> int:
    blockers: list[str] = []
    warnings: list[str] = []
    facts: list[str] = []

    results_html = FRONTEND / "results.html"
    app_js = FRONTEND / "assets" / "app.js"
    styles_css = FRONTEND / "assets" / "styles.css"
    weekly_path = DATA_ROOT / "weekly_results.json"
    archive_path = DATA_ROOT / "results_archive.json"

    for path in [results_html, app_js, styles_css, weekly_path, archive_path]:
        if not path.exists():
            blockers.append(f"Missing required Results surface file: `{rel(path)}`")

    weekly: dict[str, Any] = {}
    archive: dict[str, Any] = {}
    if not blockers:
        weekly = read_json(weekly_path)
        archive = read_json(archive_path)
        html = results_html.read_text(encoding="utf-8")
        app = app_js.read_text(encoding="utf-8")
        css = styles_css.read_text(encoding="utf-8")

        if "assets/app.js" not in html:
            blockers.append("`results.html` does not load the frontend app bundle.")
        if "weekly_results.json" not in app:
            blockers.append("Frontend app does not load `weekly_results.json`.")
        if "results_archive.json" not in app:
            blockers.append("Frontend app does not load `results_archive.json`.")
        for state, class_name in STATE_CLASSES.items():
            if class_name not in css:
                blockers.append(f"Missing visual treatment for `{state}` rows: `.{class_name}`.")
        if "resultsView" not in app or "renderResultFeedItems" not in app:
            blockers.append("Frontend app is missing Results page renderer helpers.")

        rows = rows_from(weekly)
        archive_rows = rows_from(archive)
        facts.append(f"Weekly generated at `{weekly.get('generated_at', 'missing')}`.")
        facts.append(f"Weekly rows: `{len(rows)}`.")
        facts.append(f"Archive rows: `{len(archive_rows)}`.")
        facts.append(
            f"Weekly settled/pending: `{weekly.get('settled_picks', 0)}` / `{weekly.get('pending_picks', 0)}`."
        )

        if not weekly.get("generated_at"):
            blockers.append("`weekly_results.json` is missing `generated_at`.")
        if not isinstance(weekly.get("by_market"), list):
            blockers.append("`weekly_results.json` is missing `by_market` rollups.")
        if not rows:
            blockers.append("`weekly_results.json` has no result rows for the Results page.")
        if archive.get("total_picks") is None:
            blockers.append("`results_archive.json` is missing `total_picks`.")

        markets = {str(item.get("market") or "") for item in weekly.get("by_market", []) if isinstance(item, dict)}
        for market in REQUIRED_MARKETS:
            if market not in markets:
                warnings.append(f"Market `{market}` has no current weekly rollup. It should still render as pending/empty.")

        bad_states = sorted(
            {
                str(row.get("result_status") or "").lower()
                for row in rows
                if str(row.get("result_status") or "").lower() not in VALID_STATES
            }
        )
        if bad_states:
            blockers.append(f"Weekly results contain unsupported states: `{', '.join(bad_states)}`.")

        missing_keys = [
            row.get("fixture_key") or row.get("settlement_key") or f"row_{idx}"
            for idx, row in enumerate(rows)
            if not row.get("settlement_key") or not row.get("market") or not row.get("pick")
        ]
        if missing_keys:
            blockers.append(f"Weekly result rows missing settlement identity/market/pick: `{len(missing_keys)}`.")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = [
        "# Results Page Smoke Report",
        "",
        f"Generated: `{utc_now()}`",
        "",
        "## Status",
        "",
        f"- Blockers: `{len(blockers)}`",
        f"- Warnings: `{len(warnings)}`",
        "",
        "## Facts",
        "",
        *([f"- {fact}" for fact in facts] or ["- None"]),
        "",
        "## Blockers",
        "",
        *([f"- {item}" for item in blockers] or ["- None"]),
        "",
        "## Warnings",
        "",
        *([f"- {item}" for item in warnings] or ["- None"]),
        "",
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    print(
        json.dumps(
            {
                "ok": not blockers,
                "blockers": len(blockers),
                "warnings": len(warnings),
                "report": rel(REPORT_PATH),
            },
            indent=2,
        )
    )
    return 1 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
