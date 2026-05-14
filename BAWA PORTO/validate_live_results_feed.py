#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
FEED_PATH = ROOT / "frontend" / "public" / "data" / "live_results_feed.json"

REQUIRED_SUMMARY_FIELDS = {
    "windows",
    "deploy_rows",
    "deploy_settled",
    "deploy_wins",
    "deploy_hit_rate",
    "observe_rows",
    "observe_settled",
    "observe_wins",
    "observe_hit_rate",
}

REQUIRED_WINDOW_FIELDS = {
    "window_id",
    "title",
    "period_start",
    "period_end",
    "summary",
    "by_market",
    "by_tier",
    "featured_results",
    "items",
}

REQUIRED_ITEM_FIELDS = {
    "fixture_key",
    "home_team",
    "away_team",
    "tier",
    "publish_class",
    "market",
    "pick",
    "actual",
    "result_status",
}

ALLOWED_RESULT_STATUSES = {"won", "lost", "void", "cashed", "pending"}
ALLOWED_PUBLISH_CLASSES = {"DEPLOY", "OBSERVE", "CONTEXT"}


def is_bad_number(value: Any) -> bool:
    return isinstance(value, float) and (math.isnan(value) or math.isinf(value))


def walk_numbers(value: Any, path: str, errors: list[str]) -> None:
    if is_bad_number(value):
        errors.append(f"{path}: contains NaN or Infinity")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            walk_numbers(item, f"{path}[{index}]", errors)
    elif isinstance(value, dict):
        for key, item in value.items():
            walk_numbers(item, f"{path}.{key}", errors)


def validate_rate(value: Any, path: str, errors: list[str]) -> None:
    if value is None:
        return
    if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
        errors.append(f"{path}: must be a hit rate between 0 and 1")


def validate_summary(summary: Any, path: str, errors: list[str]) -> None:
    if not isinstance(summary, dict):
        errors.append(f"{path}: must be an object")
        return
    for key in ("rows", "settled", "wins", "losses", "voids"):
        if key in summary and (not isinstance(summary[key], int) or summary[key] < 0):
            errors.append(f"{path}.{key}: must be a non-negative integer")
    if {"settled", "wins", "losses", "voids"} <= set(summary):
        if summary["wins"] + summary["losses"] + summary["voids"] != summary["settled"]:
            errors.append(f"{path}: wins + losses + voids must equal settled")
    validate_rate(summary.get("hit_rate"), f"{path}.hit_rate", errors)


def main() -> int:
    errors: list[str] = []
    if not FEED_PATH.exists():
        errors.append(f"missing file: {FEED_PATH.relative_to(ROOT)}")
    else:
        try:
            payload = json.loads(FEED_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"invalid JSON: {exc}")
            payload = None

        if isinstance(payload, dict):
            summary = payload.get("summary")
            windows = payload.get("windows")
            if not isinstance(summary, dict):
                errors.append("summary: must be an object")
            else:
                missing = REQUIRED_SUMMARY_FIELDS - set(summary)
                for key in sorted(missing):
                    errors.append(f"summary: missing `{key}`")
                validate_rate(summary.get("deploy_hit_rate"), "summary.deploy_hit_rate", errors)
                validate_rate(summary.get("observe_hit_rate"), "summary.observe_hit_rate", errors)
            if not isinstance(windows, list) or not windows:
                errors.append("windows: must be a non-empty array")
            else:
                for window_index, window in enumerate(windows):
                    path = f"windows[{window_index}]"
                    if not isinstance(window, dict):
                        errors.append(f"{path}: must be an object")
                        continue
                    missing = REQUIRED_WINDOW_FIELDS - set(window)
                    for key in sorted(missing):
                        errors.append(f"{path}: missing `{key}`")
                    for block in ("all", "deploy", "observe"):
                        if isinstance(window.get("summary"), dict) and block in window["summary"]:
                            validate_summary(window["summary"][block], f"{path}.summary.{block}", errors)
                    items = window.get("items")
                    if not isinstance(items, list):
                        errors.append(f"{path}.items: must be an array")
                    else:
                        for item_index, item in enumerate(items):
                            item_path = f"{path}.items[{item_index}]"
                            if not isinstance(item, dict):
                                errors.append(f"{item_path}: must be an object")
                                continue
                            missing_item_keys = REQUIRED_ITEM_FIELDS - set(item)
                            for key in sorted(missing_item_keys):
                                errors.append(f"{item_path}: missing `{key}`")
                            if item.get("result_status") not in ALLOWED_RESULT_STATUSES:
                                errors.append(f"{item_path}.result_status: unsupported status `{item.get('result_status')}`")
                            if item.get("publish_class") not in ALLOWED_PUBLISH_CLASSES:
                                errors.append(f"{item_path}.publish_class: unsupported class `{item.get('publish_class')}`")
                            if item.get("publish_class") == "OBSERVE" and str(item.get("tier", "")).upper() != "OBSERVE":
                                errors.append(f"{item_path}: OBSERVE publish_class must keep tier OBSERVE")
            walk_numbers(payload, "live_results_feed", errors)
        else:
            errors.append("live_results_feed.json must contain a top-level object")

    if errors:
        print("ERROR: live_results_feed.json validation failed")
        for error in errors:
            print(f"- {error}")
        return 1
    print("live_results_feed.json validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
