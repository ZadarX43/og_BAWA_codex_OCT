#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
WEEKLY_RESULTS_PATH = ROOT / "frontend" / "public" / "data" / "weekly_results.json"
RESULTS_ARCHIVE_PATH = ROOT / "frontend" / "public" / "data" / "results_archive.json"

REQUIRED_TOP_LEVEL_FIELDS = {
    "period_start",
    "period_end",
    "generated_at",
    "total_picks",
    "settled_picks",
    "pending_picks",
    "wins",
    "losses",
    "voids",
    "overall_hit_rate",
    "overall_roi",
    "overall_profit_units",
    "by_market",
    "by_tier",
    "by_league",
    "chart_points",
    "featured_wins",
    "featured_misses",
    "notes",
}

ALLOWED_KEY_NAMES = REQUIRED_TOP_LEVEL_FIELDS | {
    "market",
    "tier",
    "date",
    "wins",
    "losses",
    "voids",
    "hit_rate",
    "roi",
    "profit_units",
    "settled_picks",
    "total_picks",
    "pending_picks",
    "cumulative_profit_units",
    "cumulative_roi",
    "rolling_hit_rate",
    "fixture_key",
    "kickoff_time",
    "league",
    "home_team",
    "away_team",
    "pick",
    "confidence_tier",
    "premium_tier",
    "bookie_od",
    "model_prob",
    "bookie_implied_prob",
    "value_edge",
    "result",
    "result_status",
    "final_home_score",
    "final_away_score",
    "settled_at",
    "published_run_id",
    "items",
    "source_file",
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
        wins = item.get("wins")
        losses = item.get("losses")
        voids = item.get("voids")
        hit_rate = item.get("hit_rate")
        roi = item.get("roi")
        profit_units = item.get("profit_units")

        if not isinstance(total, int) or total < 0:
            errors.append(f"{item_path}: `total_picks` must be a non-negative integer")
        if not isinstance(settled, int) or settled < 0:
            errors.append(f"{item_path}: `settled_picks` must be a non-negative integer")
        if not isinstance(pending, int) or pending < 0:
            errors.append(f"{item_path}: `pending_picks` must be a non-negative integer")
        if not isinstance(wins, int) or wins < 0:
            errors.append(f"{item_path}: `wins` must be a non-negative integer")
        if not isinstance(losses, int) or losses < 0:
            errors.append(f"{item_path}: `losses` must be a non-negative integer")
        if not isinstance(voids, int) or voids < 0:
            errors.append(f"{item_path}: `voids` must be a non-negative integer")
        if isinstance(total, int) and isinstance(settled, int) and settled > total:
            errors.append(f"{item_path}: `settled_picks` cannot exceed `total_picks`")
        if isinstance(settled, int) and isinstance(wins, int) and isinstance(losses, int) and isinstance(voids, int):
            if wins + losses + voids != settled:
                errors.append(f"{item_path}: `wins + losses + voids` must equal `settled_picks`")
        if hit_rate is not None:
            if not isinstance(hit_rate, (int, float)) or not (0 <= float(hit_rate) <= 1):
                errors.append(f"{item_path}: `hit_rate` must be between 0 and 1 when present")
        if roi is not None:
            if not isinstance(roi, (int, float)):
                errors.append(f"{item_path}: `roi` must be numeric when present")
        if profit_units is None or not isinstance(profit_units, (int, float)):
            errors.append(f"{item_path}: `profit_units` must be numeric")


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


def validate_chart_points(items: Any, path: str, errors: list[str]) -> None:
    if not isinstance(items, list):
        errors.append(f"{path}: must be an array")
        return
    for idx, item in enumerate(items):
        item_path = f"{path}[{idx}]"
        if not isinstance(item, dict):
            errors.append(f"{item_path}: must be an object")
            continue
        for key in ("date", "settled_picks", "wins", "losses", "voids", "profit_units", "cumulative_profit_units"):
            if key not in item:
                errors.append(f"{item_path}: missing `{key}`")
        for int_key in ("settled_picks", "wins", "losses", "voids"):
            value = item.get(int_key)
            if not isinstance(value, int) or value < 0:
                errors.append(f"{item_path}: `{int_key}` must be a non-negative integer")
        for num_key in ("profit_units", "cumulative_profit_units"):
            value = item.get(num_key)
            if not isinstance(value, (int, float)):
                errors.append(f"{item_path}: `{num_key}` must be numeric")
        rolling_hit_rate = item.get("rolling_hit_rate")
        if rolling_hit_rate is not None and (
            not isinstance(rolling_hit_rate, (int, float)) or not (0 <= float(rolling_hit_rate) <= 1)
        ):
            errors.append(f"{item_path}: `rolling_hit_rate` must be between 0 and 1 when present")
        cumulative_roi = item.get("cumulative_roi")
        if cumulative_roi is not None and not isinstance(cumulative_roi, (int, float)):
            errors.append(f"{item_path}: `cumulative_roi` must be numeric when present")


def validate_archive(payload: Any, errors: list[str]) -> None:
    required_fields = {
        "period_start",
        "period_end",
        "generated_at",
        "source_file",
        "published_run_id",
        "total_picks",
        "settled_picks",
        "pending_picks",
        "items",
    }
    if not isinstance(payload, dict):
        errors.append("results_archive.json must contain a top-level object")
        return
    missing = sorted(required_fields - set(payload.keys()))
    if missing:
        errors.append(f"results_archive.json missing top-level fields: {missing}")
        return
    items = payload.get("items")
    if not isinstance(items, list):
        errors.append("results_archive.items must be an array")
        return
    for idx, item in enumerate(items):
        item_path = f"results_archive.items[{idx}]"
        if not isinstance(item, dict):
            errors.append(f"{item_path}: must be an object")
            continue
        for key in (
            "fixture_key",
            "kickoff_time",
            "league",
            "home_team",
            "away_team",
            "market",
            "pick",
            "confidence_tier",
            "premium_tier",
            "result_status",
            "profit_units",
            "published_run_id",
        ):
            value = item.get(key)
            if value is None or str(value).strip() == "":
                errors.append(f"{item_path}: missing `{key}`")
        if item.get("result_status") not in {"pending", "won", "lost", "void"}:
            errors.append(f"{item_path}: invalid `result_status`")
    walk_values(payload, "results_archive", errors)


def main() -> int:
    errors: list[str] = []

    if not WEEKLY_RESULTS_PATH.exists():
        print(f"ERROR: missing required file: {WEEKLY_RESULTS_PATH.relative_to(ROOT)}")
        return 1
    if not RESULTS_ARCHIVE_PATH.exists():
        print(f"ERROR: missing required file: {RESULTS_ARCHIVE_PATH.relative_to(ROOT)}")
        return 1

    payload = load_json(WEEKLY_RESULTS_PATH)
    if not isinstance(payload, dict):
        print("ERROR: weekly_results.json must contain a top-level object")
        return 1
    archive_payload = load_json(RESULTS_ARCHIVE_PATH)

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
    wins = payload.get("wins")
    losses = payload.get("losses")
    voids = payload.get("voids")
    overall_hit_rate = payload.get("overall_hit_rate")
    overall_roi = payload.get("overall_roi")
    overall_profit_units = payload.get("overall_profit_units")

    if not isinstance(total, int) or total < 0:
        errors.append("`total_picks` must be a non-negative integer")
    if not isinstance(settled, int) or settled < 0:
        errors.append("`settled_picks` must be a non-negative integer")
    if not isinstance(pending, int) or pending < 0:
        errors.append("`pending_picks` must be a non-negative integer")
    if not isinstance(wins, int) or wins < 0:
        errors.append("`wins` must be a non-negative integer")
    if not isinstance(losses, int) or losses < 0:
        errors.append("`losses` must be a non-negative integer")
    if not isinstance(voids, int) or voids < 0:
        errors.append("`voids` must be a non-negative integer")
    if isinstance(total, int) and isinstance(settled, int) and settled > total:
        errors.append("`total_picks` must be greater than or equal to `settled_picks`")
    if isinstance(settled, int) and isinstance(wins, int) and isinstance(losses, int) and isinstance(voids, int):
        if wins + losses + voids != settled:
            errors.append("`wins + losses + voids` must equal `settled_picks`")
    if overall_hit_rate is not None:
        if not isinstance(overall_hit_rate, (int, float)) or not (0 <= float(overall_hit_rate) <= 1):
            errors.append("`overall_hit_rate` must be between 0 and 1 when present")
    if overall_roi is not None and not isinstance(overall_roi, (int, float)):
        errors.append("`overall_roi` must be numeric when present")
    if overall_profit_units is None or not isinstance(overall_profit_units, (int, float)):
        errors.append("`overall_profit_units` must be numeric")

    validate_summary_blocks(payload.get("by_market"), "market", errors, "by_market")
    validate_summary_blocks(payload.get("by_tier"), "tier", errors, "by_tier")
    validate_summary_blocks(payload.get("by_league"), "league", errors, "by_league")
    validate_chart_points(payload.get("chart_points"), "chart_points", errors)
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
    validate_archive(archive_payload, errors)

    if errors:
        print("Weekly results validation failed.")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Weekly results validation passed.")
    print(f"- file: {WEEKLY_RESULTS_PATH.relative_to(ROOT)}")
    print(f"- archive: {RESULTS_ARCHIVE_PATH.relative_to(ROOT)}")
    print(f"- total picks: {payload['total_picks']}")
    print(f"- settled picks: {payload['settled_picks']}")
    print(f"- pending picks: {payload['pending_picks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
