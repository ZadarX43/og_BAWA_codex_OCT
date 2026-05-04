#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
WEEKLY_RESULTS_PATH = ROOT / "frontend" / "public" / "data" / "weekly_results.json"

REQUIRED_TOP_LEVEL_FIELDS = {
    "period_start",
    "period_end",
    "generated_at",
    "total_picks",
    "settled_picks",
    "pending_picks",
    "overall_hit_rate",
    "by_market",
    "by_tier",
    "featured_wins",
    "featured_misses",
    "notes",
}

ALLOWED_KEY_NAMES = REQUIRED_TOP_LEVEL_FIELDS | {
    "market",
    "tier",
    "hit_rate",
    "fixture_key",
    "kickoff_time",
    "league",
    "home_team",
    "away_team",
    "pick",
    "confidence_tier",
    "result",
}

CRITICAL_FIELDS = {"period_start", "period_end", "generated_at"}

FORBIDDEN_TERMS = {
    "threshold",
    "thr",
    "gate",
    "veto",
    "lambda",
    "p00",
    "meta",
    "support",
    "raw",
    "model_path",
    "bundle",
    "feature",
    "xg",
    "h2h",
    "streak",
    "power_diff",
    "draw_risk",
    "draw_chaos",
    "policy",
    "branch",
    "state",
    "source_path",
    "api",
    "secret",
}

SUSPICIOUS_VALUE_SNIPPETS = {
    "/Users/",
    "\\Users\\",
    ".pkl",
    ".cbm",
    ".env",
    "ModelStore/",
    "ModelStore\\",
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
                errors.append(f"{path}: contains suspicious path/model snippet `{snippet}`")
        lowered = text.lower()
        for term in FORBIDDEN_TERMS:
            if term in lowered:
                errors.append(f"{path}: contains forbidden/private term `{term}`")
        return
    if isinstance(value, list):
        for idx, item in enumerate(value):
            walk_values(item, f"{path}[{idx}]", errors)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            lowered_key = key.lower()
            if key not in ALLOWED_KEY_NAMES:
                for term in FORBIDDEN_TERMS:
                    if term in lowered_key:
                        errors.append(f"{path}: key `{key}` contains forbidden/private term `{term}`")
            walk_values(item, f"{path}.{key}", errors)


def validate_summary_blocks(items: Any, label_key: str, errors: list[str], path: str) -> None:
    if not isinstance(items, list):
        errors.append(f"{path}: must be an array")
        return

    for idx, item in enumerate(items):
        item_path = f"{path}[{idx}]"
        if not isinstance(item, dict):
            errors.append(f"{item_path}: must be an object")
            continue
        if label_key not in item or not str(item.get(label_key, "")).strip():
            errors.append(f"{item_path}: missing `{label_key}`")
        total = item.get("total_picks")
        settled = item.get("settled_picks")
        pending = item.get("pending_picks")
        hit_rate = item.get("hit_rate")

        if not isinstance(total, int) or total < 0:
            errors.append(f"{item_path}: `total_picks` must be a non-negative integer")
        if not isinstance(settled, int) or settled < 0:
            errors.append(f"{item_path}: `settled_picks` must be a non-negative integer")
        if not isinstance(pending, int) or pending < 0:
            errors.append(f"{item_path}: `pending_picks` must be a non-negative integer")
        if isinstance(total, int) and isinstance(settled, int) and settled > total:
            errors.append(f"{item_path}: `settled_picks` cannot exceed `total_picks`")
        if hit_rate is not None:
            if not isinstance(hit_rate, (int, float)) or not (0 <= float(hit_rate) <= 1):
                errors.append(f"{item_path}: `hit_rate` must be between 0 and 1 when present")


def validate_featured_rows(items: Any, path: str, errors: list[str]) -> None:
    if not isinstance(items, list):
        errors.append(f"{path}: must be an array")
        return
    for idx, item in enumerate(items):
        item_path = f"{path}[{idx}]"
        if not isinstance(item, dict):
            errors.append(f"{item_path}: must be an object")
            continue
        for key in ("fixture_key", "kickoff_time", "league", "home_team", "away_team", "market", "pick", "confidence_tier", "result"):
            value = item.get(key)
            if value is None or str(value).strip() == "":
                errors.append(f"{item_path}: missing `{key}`")


def main() -> int:
    errors: list[str] = []

    if not WEEKLY_RESULTS_PATH.exists():
        print(f"ERROR: missing required file: {WEEKLY_RESULTS_PATH.relative_to(ROOT)}")
        return 1

    payload = load_json(WEEKLY_RESULTS_PATH)
    if not isinstance(payload, dict):
        print("ERROR: weekly_results.json must contain a top-level object")
        return 1

    keys = set(payload.keys())
    missing = sorted(REQUIRED_TOP_LEVEL_FIELDS - keys)
    if missing:
        errors.append(f"missing top-level fields: {missing}")

    for field in sorted(CRITICAL_FIELDS):
        value = payload.get(field)
        if value is None or str(value).strip() == "":
            errors.append(f"critical field `{field}` is null/blank")

    total = payload.get("total_picks")
    settled = payload.get("settled_picks")
    pending = payload.get("pending_picks")
    overall_hit_rate = payload.get("overall_hit_rate")

    if not isinstance(total, int) or total < 0:
        errors.append("`total_picks` must be a non-negative integer")
    if not isinstance(settled, int) or settled < 0:
        errors.append("`settled_picks` must be a non-negative integer")
    if not isinstance(pending, int) or pending < 0:
        errors.append("`pending_picks` must be a non-negative integer")
    if isinstance(total, int) and isinstance(settled, int) and settled > total:
        errors.append("`total_picks` must be greater than or equal to `settled_picks`")
    if overall_hit_rate is not None:
        if not isinstance(overall_hit_rate, (int, float)) or not (0 <= float(overall_hit_rate) <= 1):
            errors.append("`overall_hit_rate` must be between 0 and 1 when present")

    validate_summary_blocks(payload.get("by_market"), "market", errors, "by_market")
    validate_summary_blocks(payload.get("by_tier"), "tier", errors, "by_tier")
    validate_featured_rows(payload.get("featured_wins"), "featured_wins", errors)
    validate_featured_rows(payload.get("featured_misses"), "featured_misses", errors)

    notes = payload.get("notes")
    if not isinstance(notes, list):
        errors.append("`notes` must be an array")
    else:
        for idx, note in enumerate(notes):
            if not isinstance(note, str):
                errors.append(f"notes[{idx}] must be a string")

    walk_values(payload, "weekly_results", errors)

    if errors:
        print("Weekly results validation failed.")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Weekly results validation passed.")
    print(f"- file: {WEEKLY_RESULTS_PATH.relative_to(ROOT)}")
    print(f"- total picks: {payload['total_picks']}")
    print(f"- settled picks: {payload['settled_picks']}")
    print(f"- pending picks: {payload['pending_picks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
