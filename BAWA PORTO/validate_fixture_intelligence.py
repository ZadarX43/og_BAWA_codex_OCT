#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
FRONTEND_DATA_DIR = ROOT / "frontend" / "public" / "data"
FIXTURE_INTELLIGENCE_PATH = FRONTEND_DATA_DIR / "fixture_intelligence_public.json"

REQUIRED_TOP_LEVEL_FIELDS = {
    "generated_at",
    "source_run_id",
    "source_window",
    "coverage_summary",
    "fixtures",
}
REQUIRED_FIXTURE_FIELDS = {
    "fixture_id",
    "fixture_key",
    "publish_class",
    "coverage_status",
    "kickoff_time",
    "league",
    "league_logo_url",
    "league_flag_url",
    "home_team",
    "home_team_logo_url",
    "away_team",
    "away_team_logo_url",
    "logo_join_status",
    "odds_summary",
    "signal_summary",
    "context_summary",
    "follow_relevance",
    "updated_at",
}
ALLOWED_PUBLISH_CLASSES = {"DEPLOY", "OBSERVE", "CONTEXT", "MONITOR"}
ALLOWED_SIGNAL_STATES = {"deploy", "observe", "context_only", "monitor_only"}
ALLOWED_NOTIFICATION_PRIORITIES = {"critical", "high", "medium", "low", "none"}
OBSERVE_BANNED_WORDS = {"pick", "prediction", "bet", "banker", "guaranteed", "lock"}
SUSPICIOUS_VALUE_SNIPPETS = {
    "/Users/",
    "\\Users\\",
    ".pkl",
    ".cbm",
    ".env",
    "ModelStore/",
    "ModelStore\\",
    "context_reason_codes",
    "deploy_reason_codes",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def is_bad_number(value: Any) -> bool:
    return isinstance(value, float) and (math.isnan(value) or math.isinf(value))


def walk_values(value: Any, path: str, errors: list[str]) -> None:
    if is_bad_number(value):
        errors.append(f"{path}: contains NaN or Infinity")
        return
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"nan", "inf", "infinity", "-inf", "-infinity"}:
            errors.append(f"{path}: contains string NaN/Infinity marker")
        for snippet in SUSPICIOUS_VALUE_SNIPPETS:
            if snippet in value:
                errors.append(f"{path}: contains suspicious private snippet `{snippet}`")
        return
    if isinstance(value, list):
        for idx, item in enumerate(value):
            walk_values(item, f"{path}[{idx}]", errors)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            walk_values(item, f"{path}.{key}", errors)


def validate_fixture(record: dict[str, Any], idx: int, errors: list[str]) -> None:
    keys = set(record.keys())
    missing = sorted(REQUIRED_FIXTURE_FIELDS - keys)
    if missing:
        errors.append(f"fixtures[{idx}]: missing fields {missing}")

    publish_class = record.get("publish_class")
    if publish_class not in ALLOWED_PUBLISH_CLASSES:
        errors.append(f"fixtures[{idx}]: invalid publish_class `{publish_class}`")

    for field in ("fixture_id", "fixture_key", "kickoff_time", "league", "home_team", "away_team", "updated_at"):
        if str(record.get(field, "") or "").strip() == "":
            errors.append(f"fixtures[{idx}]: critical field `{field}` is blank")

    signal_summary = record.get("signal_summary")
    if not isinstance(signal_summary, dict):
        errors.append(f"fixtures[{idx}]: `signal_summary` must be an object")
        return

    signal_state = signal_summary.get("signal_state")
    if signal_state not in ALLOWED_SIGNAL_STATES:
        errors.append(f"fixtures[{idx}]: invalid signal_state `{signal_state}`")

    if publish_class == "DEPLOY":
        if signal_state != "deploy":
            errors.append(f"fixtures[{idx}]: DEPLOY fixture must use `deploy` signal_state")
        if not signal_summary.get("market_family"):
            errors.append(f"fixtures[{idx}]: DEPLOY fixture missing market_family")
        if not signal_summary.get("confidence_tier"):
            errors.append(f"fixtures[{idx}]: DEPLOY fixture missing confidence_tier")
        if not signal_summary.get("deploy_pick"):
            errors.append(f"fixtures[{idx}]: DEPLOY fixture missing deploy_pick")
    else:
        label = str(signal_summary.get("signal_label", "") or "")
        summary_text = str(signal_summary.get("summary_text", "") or "")
        lowered = f"{label} {summary_text}".lower()
        for word in OBSERVE_BANNED_WORDS:
            if re.search(rf"\b{re.escape(word)}\b", lowered):
                errors.append(f"fixtures[{idx}]: non-deploy wording contains banned word `{word}`")
        if publish_class == "OBSERVE" and signal_state != "observe":
            errors.append(f"fixtures[{idx}]: OBSERVE fixture must use `observe` signal_state")
        if publish_class == "OBSERVE" and not summary_text:
            errors.append(f"fixtures[{idx}]: OBSERVE fixture missing summary_text")
        if publish_class == "CONTEXT" and signal_state != "context_only":
            errors.append(f"fixtures[{idx}]: CONTEXT fixture must use `context_only` signal_state")
        if publish_class == "MONITOR" and signal_state != "monitor_only":
            errors.append(f"fixtures[{idx}]: MONITOR fixture must use `monitor_only` signal_state")

    follow_relevance = record.get("follow_relevance")
    if not isinstance(follow_relevance, dict):
        errors.append(f"fixtures[{idx}]: `follow_relevance` must be an object")
    else:
        priority = follow_relevance.get("notification_priority")
        if priority not in ALLOWED_NOTIFICATION_PRIORITIES:
            errors.append(f"fixtures[{idx}]: invalid notification_priority `{priority}`")

    walk_values(record, f"fixtures[{idx}]", errors)


def main() -> int:
    errors: list[str] = []

    if not FIXTURE_INTELLIGENCE_PATH.exists():
        errors.append(f"missing required file: {FIXTURE_INTELLIGENCE_PATH.relative_to(ROOT)}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    payload = load_json(FIXTURE_INTELLIGENCE_PATH)
    if not isinstance(payload, dict):
        print("Validation failed.")
        print("- top-level payload must be an object")
        return 1

    top_level_keys = set(payload.keys())
    missing_top = sorted(REQUIRED_TOP_LEVEL_FIELDS - top_level_keys)
    if missing_top:
        errors.append(f"top-level payload missing fields {missing_top}")

    fixtures = payload.get("fixtures")
    if not isinstance(fixtures, list):
        errors.append("top-level `fixtures` must be a list")
        fixtures = []

    for idx, record in enumerate(fixtures):
        if not isinstance(record, dict):
            errors.append(f"fixtures[{idx}]: record must be an object")
            continue
        validate_fixture(record, idx, errors)

    coverage_summary = payload.get("coverage_summary")
    if isinstance(coverage_summary, dict):
        actual = {
            "total_fixtures": len(fixtures),
            "deploy_count": sum(1 for record in fixtures if isinstance(record, dict) and record.get("publish_class") == "DEPLOY"),
            "observe_count": sum(1 for record in fixtures if isinstance(record, dict) and record.get("publish_class") == "OBSERVE"),
            "context_count": sum(1 for record in fixtures if isinstance(record, dict) and record.get("publish_class") == "CONTEXT"),
            "monitor_count": sum(1 for record in fixtures if isinstance(record, dict) and record.get("publish_class") == "MONITOR"),
            "covered_leagues_count": len({record.get("league") for record in fixtures if isinstance(record, dict) and record.get("league")}),
        }
        for key, expected in actual.items():
            if coverage_summary.get(key) != expected:
                errors.append(
                    f"coverage_summary.{key}: expected `{expected}` from fixtures but found `{coverage_summary.get(key)}`"
                )
    else:
        errors.append("top-level `coverage_summary` must be an object")

    walk_values(payload, "payload", errors)

    if errors:
        print("Validation failed.")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Validation passed.")
    print(f"- fixtures: {len(fixtures)}")
    print(f"- file: {FIXTURE_INTELLIGENCE_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
