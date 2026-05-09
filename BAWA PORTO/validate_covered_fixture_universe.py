#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
FRONTEND_DATA_DIR = ROOT / "frontend" / "public" / "data"
UNIVERSE_PATH = FRONTEND_DATA_DIR / "covered_fixture_universe.json"

REQUIRED_TOP_LEVEL_FIELDS = {"generated_at", "source_run_id", "source_window", "coverage_summary", "fixtures"}
REQUIRED_FIXTURE_FIELDS = {
    "fixture_id",
    "fixture_key",
    "kickoff_time",
    "league",
    "home_team",
    "away_team",
    "league_logo_url",
    "league_flag_url",
    "home_team_logo_url",
    "away_team_logo_url",
    "logo_join_status",
    "coverage_status",
    "routing_status",
    "identity_source",
    "source_availability",
    "follow_candidates",
    "updated_at",
}
ALLOWED_ROUTING_STATUS = {"routed", "non_routed", "unsupported", "hidden"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_fixture(record: dict[str, Any], idx: int, errors: list[str]) -> None:
    keys = set(record.keys())
    missing = sorted(REQUIRED_FIXTURE_FIELDS - keys)
    if missing:
        errors.append(f"fixtures[{idx}]: missing fields {missing}")
    for field in ("fixture_id", "fixture_key", "kickoff_time", "league", "home_team", "away_team", "updated_at"):
        if str(record.get(field, "") or "").strip() == "":
            errors.append(f"fixtures[{idx}]: blank critical field `{field}`")
    if record.get("routing_status") not in ALLOWED_ROUTING_STATUS:
        errors.append(f"fixtures[{idx}]: invalid routing_status `{record.get('routing_status')}`")
    if not isinstance(record.get("source_availability"), dict):
        errors.append(f"fixtures[{idx}]: `source_availability` must be an object")
    if not isinstance(record.get("follow_candidates"), dict):
        errors.append(f"fixtures[{idx}]: `follow_candidates` must be an object")


def main() -> int:
    errors: list[str] = []
    if not UNIVERSE_PATH.exists():
        print(f"ERROR: missing required file: {UNIVERSE_PATH.relative_to(ROOT)}")
        return 1

    payload = load_json(UNIVERSE_PATH)
    if not isinstance(payload, dict):
        print("Validation failed.")
        print("- top-level payload must be an object")
        return 1

    missing_top = sorted(REQUIRED_TOP_LEVEL_FIELDS - set(payload.keys()))
    if missing_top:
        errors.append(f"top-level payload missing fields {missing_top}")

    fixtures = payload.get("fixtures")
    if not isinstance(fixtures, list):
        errors.append("top-level `fixtures` must be a list")
        fixtures = []

    seen: set[str] = set()
    for idx, record in enumerate(fixtures):
        if not isinstance(record, dict):
            errors.append(f"fixtures[{idx}]: record must be an object")
            continue
        validate_fixture(record, idx, errors)
        fixture_key = str(record.get("fixture_key", "") or "").strip()
        if fixture_key:
            if fixture_key in seen:
                errors.append(f"fixtures[{idx}]: duplicate fixture_key `{fixture_key}`")
            seen.add(fixture_key)

    summary = payload.get("coverage_summary")
    if isinstance(summary, dict):
        expected = {
            "total_fixtures": len(fixtures),
            "routed_count": sum(1 for record in fixtures if isinstance(record, dict) and record.get("routing_status") == "routed"),
            "non_routed_count": sum(1 for record in fixtures if isinstance(record, dict) and record.get("routing_status") == "non_routed"),
            "hidden_count": sum(1 for record in fixtures if isinstance(record, dict) and record.get("routing_status") == "hidden"),
            "covered_leagues_count": len({record.get("league") for record in fixtures if isinstance(record, dict) and record.get("league")}),
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
    print(f"- fixtures: {len(fixtures)}")
    print(f"- file: {UNIVERSE_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
