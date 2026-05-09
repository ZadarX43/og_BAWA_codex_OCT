#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "frontend" / "public" / "data" / "league_coverage_audit.json"

REQUIRED_TOP_LEVEL_FIELDS = {"generated_at", "source_run_id", "source_window", "coverage_summary", "leagues"}
REQUIRED_LEAGUE_FIELDS = {
    "league",
    "window_from",
    "window_to",
    "fixture_counts",
    "market_output",
    "routing_presence",
    "overlay_support",
    "source_counts",
    "classification",
    "notes",
}
ALLOWED_CLASSIFICATIONS = {"full_coverage", "partial_coverage", "context_only", "blind_spot"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_league(record: dict[str, Any], idx: int, errors: list[str]) -> None:
    missing = sorted(REQUIRED_LEAGUE_FIELDS - set(record.keys()))
    if missing:
        errors.append(f"leagues[{idx}]: missing fields {missing}")
    if str(record.get("league", "") or "").strip() == "":
        errors.append(f"leagues[{idx}]: blank league")
    if record.get("classification") not in ALLOWED_CLASSIFICATIONS:
        errors.append(f"leagues[{idx}]: invalid classification `{record.get('classification')}`")
    for field in ("fixture_counts", "market_output", "routing_presence", "overlay_support", "source_counts"):
        if not isinstance(record.get(field), dict):
            errors.append(f"leagues[{idx}]: `{field}` must be an object")
    if not isinstance(record.get("notes"), list):
        errors.append(f"leagues[{idx}]: `notes` must be a list")


def main() -> int:
    if not DATA_PATH.exists():
        print(f"ERROR: missing required file: {DATA_PATH.relative_to(ROOT)}")
        return 1

    payload = load_json(DATA_PATH)
    errors: list[str] = []
    if not isinstance(payload, dict):
        print("Validation failed.")
        print("- top-level payload must be an object")
        return 1

    missing_top = sorted(REQUIRED_TOP_LEVEL_FIELDS - set(payload.keys()))
    if missing_top:
        errors.append(f"top-level payload missing fields {missing_top}")

    leagues = payload.get("leagues")
    if not isinstance(leagues, list):
        errors.append("top-level `leagues` must be a list")
        leagues = []

    seen: set[str] = set()
    for idx, record in enumerate(leagues):
        if not isinstance(record, dict):
            errors.append(f"leagues[{idx}]: record must be an object")
            continue
        validate_league(record, idx, errors)
        league = str(record.get("league", "") or "").strip()
        if league:
            if league in seen:
                errors.append(f"leagues[{idx}]: duplicate league `{league}`")
            seen.add(league)

    summary = payload.get("coverage_summary")
    if isinstance(summary, dict):
        expected = {
            "total_leagues": len(leagues),
            "full_coverage_count": sum(1 for record in leagues if isinstance(record, dict) and record.get("classification") == "full_coverage"),
            "partial_coverage_count": sum(1 for record in leagues if isinstance(record, dict) and record.get("classification") == "partial_coverage"),
            "context_only_count": sum(1 for record in leagues if isinstance(record, dict) and record.get("classification") == "context_only"),
            "blind_spot_count": sum(1 for record in leagues if isinstance(record, dict) and record.get("classification") == "blind_spot"),
        }
        for key, value in expected.items():
            if summary.get(key) != value:
                errors.append(f"coverage_summary.{key}: expected `{value}` but found `{summary.get(key)}`")
    else:
        errors.append("top-level `coverage_summary` must be an object")

    if errors:
        print("Validation failed.")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Validation passed.")
    print(f"- leagues: {len(leagues)}")
    print(f"- file: {DATA_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
