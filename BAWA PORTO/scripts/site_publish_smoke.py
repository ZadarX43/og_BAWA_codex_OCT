#!/usr/bin/env python3
"""Smoke-check website-safe JSON before a launch publish."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "frontend" / "public" / "data"
REPORT_PATH = ROOT / "reports" / "latest" / "SITE_PUBLISH_SMOKE_REPORT.md"

REQUIRED_FILES = [
    "publish_summary.json",
    "public_predictions.json",
    "weekly_results.json",
    "results_archive.json",
    "fixture_intelligence_public.json",
]

ALLOWED_RESULT_STATES = {"won", "lost", "void", "pending", "cashed", "missed"}
BLOCKED_PREMIUM_FIELDS = {"model_path", "gate_detail", "raw_features", "feature_vector", "secret", "api_key"}
PUBLIC_BLOCKED_FIELDS = {
    "bookie_implied_prob",
    "correct_score_shortlist",
    "human_reason",
    "reason_tokens",
    "safe_for_large_acca_flag",
    "safe_for_small_acca_flag",
    "slip_role_hint",
}


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rows_from_payload(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("predictions", "data", "items", "fixtures"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def data_size_bytes() -> int:
    total = 0
    for path in DATA_ROOT.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def main() -> int:
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    blockers = []
    warnings = []
    facts = []

    for filename in REQUIRED_FILES:
        path = DATA_ROOT / filename
        if not path.exists():
            blockers.append(f"Missing required website data file: `{rel(path)}`")

    if blockers:
        payloads = {}
    else:
        payloads = {filename: read_json(DATA_ROOT / filename) for filename in REQUIRED_FILES}

    if payloads:
        summary = payloads["publish_summary.json"]
        if not summary.get("generated_at"):
            blockers.append("`publish_summary.json` is missing `generated_at`.")
        facts.append(f"Publish summary generated at `{summary.get('generated_at', 'missing')}`.")

        public_rows = rows_from_payload(payloads["public_predictions.json"])
        if not public_rows:
            warnings.append("Public predictions payload has no rows.")
        for index, row in enumerate(public_rows):
            leaked = sorted(PUBLIC_BLOCKED_FIELDS.intersection(row.keys()))
            if leaked:
                blockers.append(f"Public row {index} leaks premium fields: {', '.join(leaked)}")
        facts.append(f"Public prediction rows: `{len(public_rows)}`.")

        weekly = payloads["weekly_results.json"]
        result_rows = rows_from_payload(weekly)
        if not weekly.get("generated_at"):
            blockers.append("`weekly_results.json` is missing `generated_at`.")
        if weekly.get("total_picks") is None:
            blockers.append("`weekly_results.json` is missing `total_picks`.")
        for index, row in enumerate(result_rows):
            state = str(row.get("result_status", "")).strip().lower()
            if state and state not in ALLOWED_RESULT_STATES:
                blockers.append(f"Weekly result row {index} has unsupported status `{state}`.")
        facts.append(
            "Weekly results: "
            f"`{weekly.get('settled_picks', 0)}` settled, "
            f"`{weekly.get('pending_picks', 0)}` pending."
        )

        archive = payloads["results_archive.json"]
        if archive.get("total_picks") is None:
            blockers.append("`results_archive.json` is missing `total_picks`.")
        facts.append(f"Results archive rows: `{archive.get('total_picks', 'missing')}`.")

        fixtures_payload = payloads["fixture_intelligence_public.json"]
        fixtures = rows_from_payload(fixtures_payload)
        if not fixtures_payload.get("generated_at"):
            blockers.append("`fixture_intelligence_public.json` is missing `generated_at`.")
        if not fixtures:
            warnings.append("Fixture intelligence payload has no fixtures.")
        missing_coverage = [row.get("fixture_key") or row.get("fixture_id") for row in fixtures if not row.get("coverage_status")]
        missing_freshness = [
            row.get("fixture_key") or row.get("fixture_id")
            for row in fixtures
            if not (row.get("updated_at") or row.get("capture_generated_at") or row.get("source_data_cutoff_at"))
        ]
        if missing_coverage:
            warnings.append(f"Fixtures missing coverage status: `{len(missing_coverage)}`.")
        if missing_freshness:
            warnings.append(f"Fixtures missing freshness metadata: `{len(missing_freshness)}`.")
        facts.append(f"Fixture intelligence rows: `{len(fixtures)}`.")

    premium_path = DATA_ROOT / "premium_predictions.json"
    if premium_path.exists():
        premium_rows = rows_from_payload(read_json(premium_path))
        for index, row in enumerate(premium_rows):
            leaked = sorted(BLOCKED_PREMIUM_FIELDS.intersection(row.keys()))
            if leaked:
                blockers.append(f"Premium row {index} leaks blocked fields: {', '.join(leaked)}")
        facts.append(f"Premium prediction rows: `{len(premium_rows)}`.")
    else:
        warnings.append("Premium predictions payload is not present in the static export.")

    total_mb = data_size_bytes() / (1024 * 1024)
    facts.append(f"Website data footprint: `{total_mb:.1f} MB`.")
    if total_mb > 200:
        warnings.append("Website data footprint is above 200 MB; review Cloudflare hosting and bandwidth pressure.")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = [
        "# Site Publish Smoke Report",
        "",
        f"Generated: `{generated_at}`",
        "",
        "## Status",
        "",
        f"- Blockers: `{len(blockers)}`",
        f"- Warnings: `{len(warnings)}`",
        "",
        "## Facts",
        "",
        *[f"- {fact}" for fact in facts],
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
                "data_footprint_mb": round(total_mb, 1),
            },
            indent=2,
        )
    )
    return 1 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
