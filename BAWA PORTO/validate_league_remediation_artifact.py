#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "frontend" / "public" / "data" / "league_remediation_plan.json"

REQUIRED_TOP_LEVEL_FIELDS = {"generated_at", "source_run_id", "source_window", "coverage_summary", "leagues"}
REQUIRED_LEAGUE_FIELDS = {
    "league",
    "current_classification",
    "target_classification",
    "primary_issue",
    "secondary_issue",
    "market_gaps",
    "fixture_counts",
    "overlay_support",
    "recommended_actions",
    "priority_rank",
}
ALLOWED_CLASSIFICATIONS = {"partial_coverage", "blind_spot", "context_only", "hidden_for_now", "full_coverage"}
ALLOWED_ISSUES = {"model_expansion", "routing_expansion", "overlay_context_enhancement", "ingestion_repair", "none"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_record(record: dict[str, Any], idx: int, errors: list[str]) -> None:
    missing = sorted(REQUIRED_LEAGUE_FIELDS - set(record.keys()))
    if missing:
        errors.append(f"leagues[{idx}]: missing fields {missing}")
    if str(record.get("league", "") or "").strip() == "":
        errors.append(f"leagues[{idx}]: blank league")
    if record.get("current_classification") not in {"partial_coverage", "blind_spot", "context_only"}:
        errors.append(f"leagues[{idx}]: invalid current_classification `{record.get('current_classification')}`")
    if record.get("target_classification") not in ALLOWED_CLASSIFICATIONS:
        errors.append(f"leagues[{idx}]: invalid target_classification `{record.get('target_classification')}`")
    if record.get("primary_issue") not in ALLOWED_ISSUES:
        errors.append(f"leagues[{idx}]: invalid primary_issue `{record.get('primary_issue')}`")
    if record.get("secondary_issue") is not None and record.get("secondary_issue") not in ALLOWED_ISSUES:
        errors.append(f"leagues[{idx}]: invalid secondary_issue `{record.get('secondary_issue')}`")
    if not isinstance(record.get("market_gaps"), dict):
        errors.append(f"leagues[{idx}]: `market_gaps` must be an object")
    if not isinstance(record.get("fixture_counts"), dict):
        errors.append(f"leagues[{idx}]: `fixture_counts` must be an object")
    if not isinstance(record.get("overlay_support"), dict):
        errors.append(f"leagues[{idx}]: `overlay_support` must be an object")
    if not isinstance(record.get("recommended_actions"), list) or not record.get("recommended_actions"):
        errors.append(f"leagues[{idx}]: `recommended_actions` must be a non-empty list")
    if not isinstance(record.get("priority_rank"), int) or record.get("priority_rank", 0) <= 0:
        errors.append(f"leagues[{idx}]: invalid priority_rank `{record.get('priority_rank')}`")


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

    seen_ranks: set[int] = set()
    seen_leagues: set[str] = set()
    for idx, record in enumerate(leagues):
        if not isinstance(record, dict):
            errors.append(f"leagues[{idx}]: record must be an object")
            continue
        validate_record(record, idx, errors)
        league = str(record.get("league", "") or "").strip()
        if league:
            if league in seen_leagues:
                errors.append(f"leagues[{idx}]: duplicate league `{league}`")
            seen_leagues.add(league)
        rank = record.get("priority_rank")
        if isinstance(rank, int):
            if rank in seen_ranks:
                errors.append(f"leagues[{idx}]: duplicate priority_rank `{rank}`")
            seen_ranks.add(rank)

    summary = payload.get("coverage_summary")
    if isinstance(summary, dict):
        expected = {
            "total_weak_leagues": len(leagues),
            "blind_spot_count": sum(1 for record in leagues if isinstance(record, dict) and record.get("current_classification") == "blind_spot"),
            "partial_coverage_count": sum(1 for record in leagues if isinstance(record, dict) and record.get("current_classification") == "partial_coverage"),
            "model_expansion_count": sum(1 for record in leagues if isinstance(record, dict) and record.get("primary_issue") == "model_expansion"),
            "routing_expansion_count": sum(1 for record in leagues if isinstance(record, dict) and record.get("primary_issue") == "routing_expansion"),
            "overlay_context_enhancement_count": sum(1 for record in leagues if isinstance(record, dict) and record.get("primary_issue") == "overlay_context_enhancement"),
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
    print(f"- weak leagues: {len(leagues)}")
    print(f"- file: {DATA_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
