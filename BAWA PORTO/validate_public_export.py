#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
FRONTEND_DATA_DIR = ROOT / "frontend" / "public" / "data"

PUBLIC_PATH = FRONTEND_DATA_DIR / "public_predictions.json"
PREMIUM_PATH = FRONTEND_DATA_DIR / "premium_predictions.json"

PUBLIC_FIELDS = {
    "fixture_id",
    "fixture_key",
    "kickoff_time",
    "league",
    "home_team",
    "away_team",
    "market",
    "pick",
    "confidence_tier",
    "display_confidence",
    "bookie_od",
    "model_prob_display",
    "value_edge_display",
    "short_reason",
    "is_free",
}

PREMIUM_FIELDS = {
    "fixture_id",
    "fixture_key",
    "kickoff_time",
    "league",
    "home_team",
    "away_team",
    "market",
    "pick",
    "confidence_tier",
    "model_prob",
    "bookie_implied_prob",
    "value_edge",
    "bookie_od",
    "reason_tokens",
    "human_reason",
    "slip_role_hint",
    "safe_for_small_acca_flag",
    "safe_for_large_acca_flag",
    "correct_score_shortlist",
    "premium_tier",
}

FORBIDDEN_PUBLIC_SUBSTRINGS = {
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

CRITICAL_FIELDS = {"fixture_id", "fixture_key", "league", "market", "pick", "kickoff_time"}


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
        return
    if isinstance(value, list):
        for idx, item in enumerate(value):
            walk_values(item, f"{path}[{idx}]", errors)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            walk_values(item, f"{path}.{key}", errors)


def validate_records(records: Any, allowed_fields: set[str], label: str, errors: list[str]) -> None:
    if not isinstance(records, list):
        errors.append(f"{label}: top-level payload must be a list")
        return

    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"{label}[{idx}]: record must be an object")
            continue

        keys = set(record.keys())
        missing = sorted(allowed_fields - keys)
        extra = sorted(keys - allowed_fields)
        if missing:
            errors.append(f"{label}[{idx}]: missing fields {missing}")
        if extra:
            errors.append(f"{label}[{idx}]: unexpected fields {extra}")

        for key in sorted(keys):
            lowered = key.lower()
            if label == "public":
                for snippet in FORBIDDEN_PUBLIC_SUBSTRINGS:
                    if snippet in lowered and key not in allowed_fields:
                        errors.append(f"{label}[{idx}]: forbidden public field name `{key}`")

        for field in sorted(CRITICAL_FIELDS):
            value = record.get(field)
            if value is None or str(value).strip() == "":
                errors.append(f"{label}[{idx}]: critical field `{field}` is null/blank")

        walk_values(record, f"{label}[{idx}]", errors)

        if label == "premium":
            reason_tokens = record.get("reason_tokens")
            if not isinstance(reason_tokens, list):
                errors.append(f"{label}[{idx}]: `reason_tokens` must be a list")
            shortlist = record.get("correct_score_shortlist")
            if not isinstance(shortlist, list):
                errors.append(f"{label}[{idx}]: `correct_score_shortlist` must be a list")


def main() -> int:
    errors: list[str] = []

    for path in (PUBLIC_PATH, PREMIUM_PATH):
        if not path.exists():
            errors.append(f"missing required file: {path.relative_to(ROOT)}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    public_records = load_json(PUBLIC_PATH)
    premium_records = load_json(PREMIUM_PATH)

    validate_records(public_records, PUBLIC_FIELDS, "public", errors)
    validate_records(premium_records, PREMIUM_FIELDS, "premium", errors)

    if errors:
        print("Validation failed.")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Validation passed.")
    print(f"- public records: {len(public_records)}")
    print(f"- premium records: {len(premium_records)}")
    print(f"- public file: {PUBLIC_PATH.relative_to(ROOT)}")
    print(f"- premium file: {PREMIUM_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
