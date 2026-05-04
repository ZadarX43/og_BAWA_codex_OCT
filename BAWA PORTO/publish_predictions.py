#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PREDICTIONS_ROOT = ROOT / "predictions_output"
FRONTEND_DATA_DIR = ROOT / "frontend" / "public" / "data"
REPORTS_DIR = ROOT / "reports" / "latest"

PUBLIC_FIELDS = [
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
]

PREMIUM_FIELDS = [
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
]

SAFE_REASON_BLOCKLIST = {
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
    "source",
    "api",
    "key",
    "secret",
    "token",
}

TIER_PRIORITY = {"ELITE": 0, "STANDARD": 1}
MARKET_PRIORITY = {"FTR": 0, "BTTS": 1, "OU25": 2, "CS": 3}
ALLOWED_MARKETS = {"FTR", "BTTS", "OU25", "CS"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dirs() -> None:
    FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def find_latest_source() -> Path:
    files = [
        path
        for path in PREDICTIONS_ROOT.rglob("DEPLOY_COMBINED_*.csv")
        if path.is_file() and "SCORED" not in path.name.upper()
    ]
    if not files:
        raise FileNotFoundError("No true DEPLOY_COMBINED_*.csv source files found.")
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return files[0]


def resolve_source_path(src: str | None) -> Path:
    if src:
        candidate = Path(src)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        candidate = candidate.resolve()
        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError(f"Explicit --src file not found: {candidate}")
        if "SCORED" in candidate.name.upper():
            raise ValueError(f"Explicit --src must not be a scored export: {candidate}")
        return candidate
    return find_latest_source()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def pick_first(row: dict[str, str], keys: list[str], fallback_name: str, counters: Counter[str]) -> str:
    for key in keys:
        value = str(row.get(key, "") or "").strip()
        if value:
            if key != keys[0]:
                counters[f"{fallback_name}:{key}"] += 1
            return value
    counters[f"{fallback_name}:missing"] += 1
    return ""


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def parse_bool_flag(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y"}


def normalize_market(value: str) -> str:
    text = str(value or "").strip().upper()
    if text in {"FTR", "BTTS", "CS"}:
        return text
    if text in {"OU25", "OVER25", "UNDER25"}:
        return "OU25"
    return text


def normalize_pick(value: str) -> str:
    return str(value or "").strip().upper()


def normalize_tier(value: str) -> str:
    text = str(value or "").strip().upper()
    if text in {"ELITE", "STANDARD", "OBSERVE"}:
        return text
    return text


def build_fixture_id(fixture_key: str, market: str, pick: str) -> str:
    base = "__".join(part for part in [fixture_key, market, pick] if part)
    return re.sub(r"[^A-Za-z0-9_:-]+", "_", base)


def confidence_bucket(prob: float | None, tier: str) -> str:
    if tier == "ELITE":
        return "High"
    if prob is None:
        return "Qualified"
    if prob >= 0.65:
        return "High"
    if prob >= 0.56:
        return "Medium"
    return "Lean"


def format_pct_display(prob: float | None) -> str:
    if prob is None:
        return "N/A"
    return f"{round(prob * 100):.0f}%"


def format_edge_display(edge: float | None) -> str:
    if edge is None:
        return "N/A"
    pts = edge * 100.0
    return f"{pts:+.1f} pts"


def round_prob(prob: float | None) -> float | None:
    if prob is None:
        return None
    return round(prob, 4)


def parse_reason_tokens(row: dict[str, str]) -> list[str]:
    raw = str(row.get("context_reason_codes", "") or "").strip()
    if not raw:
        raw = str(row.get("deploy_reason_codes", "") or "").strip()
    tokens: list[str] = []
    if raw:
        for token in re.split(r"[|,;]+", raw):
            cleaned = re.sub(r"[^A-Za-z0-9_:-]+", "_", token.strip().upper()).strip("_")
            if cleaned:
                lowered = cleaned.lower()
                if not any(block in lowered for block in SAFE_REASON_BLOCKLIST):
                    tokens.append(cleaned)
    return tokens


def derive_safe_reason_tokens(row: dict[str, str], market: str, tier: str) -> list[str]:
    tokens = parse_reason_tokens(row)
    safe_tokens: list[str] = []
    for token in tokens:
        if token not in safe_tokens:
            safe_tokens.append(token)
        if len(safe_tokens) >= 5:
            break
    generic = ["DEPLOYABLE", f"MARKET_{market}", f"TIER_{tier}"]
    for token in generic:
        if token not in safe_tokens:
            safe_tokens.append(token)
    return safe_tokens[:6]


def short_reason_for(market: str, tier: str, pick: str) -> str:
    market_label = {
        "FTR": "result",
        "BTTS": "both-teams-to-score",
        "OU25": "totals",
        "CS": "correct-score",
    }.get(market, "market")
    pick_text = pick.replace("_", " ").title() if pick else "selection"
    if tier == "ELITE":
        return f"Top-rated {pick_text.lower()} {market_label} play cleared live routing checks."
    return f"Qualified {pick_text.lower()} {market_label} play stayed inside the deployable board."


def premium_reason_for(market: str, tier: str, pick: str) -> str:
    pick_text = pick.replace("_", " ").title() if pick else "selection"
    if market == "FTR":
        return f"{pick_text} carried enough support to remain live in the {tier.lower()} board."
    if market == "BTTS":
        return f"{pick_text} met the live BTTS posture for the {tier.lower()} board."
    if market == "OU25":
        return f"{pick_text} passed the live totals posture for the {tier.lower()} board."
    if market == "CS":
        return f"{pick_text} remained viable as a correct-score support view."
    return f"{pick_text} remained deployable in the {tier.lower()} board."


def slip_role_hint(tier: str, market: str) -> str:
    if tier == "ELITE":
        return "anchor" if market in {"FTR", "OU25", "BTTS"} else "featured"
    return "support"


def build_correct_score_shortlist(row: dict[str, str]) -> list[dict[str, Any]]:
    shortlist: list[dict[str, Any]] = []
    for idx in (1, 2, 3):
        scoreline = str(row.get(f"cs{idx}", "") or "").strip()
        probability = parse_float(row.get(f"cs{idx}_p", ""))
        if not scoreline or probability is None:
            continue
        shortlist.append(
            {
                "scoreline": scoreline,
                "probability": round(probability, 3),
            }
        )
    return shortlist


def build_record(row: dict[str, str], counters: Counter[str]) -> dict[str, Any] | None:
    fixture_key = pick_first(row, ["fixture_key"], "fixture_key", counters)
    kickoff_time = pick_first(row, ["match_date", "kickoff_time"], "kickoff_time", counters)
    league = pick_first(row, ["league"], "league", counters)
    home_team = pick_first(row, ["home_team_name", "home_team"], "home_team", counters)
    away_team = pick_first(row, ["away_team_name", "away_team"], "away_team", counters)
    market = normalize_market(pick_first(row, ["market"], "market", counters))
    pick = normalize_pick(pick_first(row, ["selection", "bookie_pick", "pick"], "pick", counters))
    tier = normalize_tier(pick_first(row, ["deploy_tier", "tier"], "tier", counters))

    required = [fixture_key, kickoff_time, league, home_team, away_team, market, pick, tier]
    if any(not value for value in required):
        counters["dropped:critical_missing"] += 1
        return None

    if market not in ALLOWED_MARKETS:
        counters[f"dropped:unsupported_market:{market}"] += 1
        return None

    model_prob = parse_float(row.get("model_p_for_bookie", ""))
    bookie_implied = parse_float(row.get("bookie_implied", ""))
    if bookie_implied is None:
        bookie_implied = parse_float(row.get("bookie_implied_novig", ""))
        if bookie_implied is not None:
            counters["bookie_implied_prob:bookie_implied_novig"] += 1
    value_edge = parse_float(row.get("gap", ""))
    if value_edge is None:
        value_edge = parse_float(row.get("gap_novig", ""))
        if value_edge is not None:
            counters["value_edge:gap_novig"] += 1
    bookie_od = parse_float(row.get("bookie_od", ""))

    if tier not in TIER_PRIORITY:
        counters[f"dropped:non_deployable_tier:{tier}"] += 1
        return None

    fixture_id = build_fixture_id(fixture_key, market, pick)
    safe_reason_tokens = derive_safe_reason_tokens(row, market, tier)
    correct_score_shortlist = build_correct_score_shortlist(row)

    return {
        "fixture_id": fixture_id,
        "fixture_key": fixture_key,
        "kickoff_time": kickoff_time,
        "league": league,
        "home_team": home_team,
        "away_team": away_team,
        "market": market,
        "pick": pick,
        "confidence_tier": tier,
        "display_confidence": confidence_bucket(model_prob, tier),
        "bookie_od": round(bookie_od, 4) if bookie_od is not None else None,
        "model_prob_display": format_pct_display(model_prob),
        "value_edge_display": format_edge_display(value_edge),
        "short_reason": short_reason_for(market, tier, pick),
        "is_free": tier == "ELITE",
        "model_prob": round_prob(model_prob),
        "bookie_implied_prob": round_prob(bookie_implied),
        "value_edge": round_prob(value_edge),
        "reason_tokens": safe_reason_tokens,
        "human_reason": premium_reason_for(market, tier, pick),
        "slip_role_hint": slip_role_hint(tier, market),
        "safe_for_small_acca_flag": tier == "ELITE",
        "safe_for_large_acca_flag": tier == "ELITE",
        "correct_score_shortlist": correct_score_shortlist,
        "premium_tier": tier,
        "_sort_tier": TIER_PRIORITY[tier],
        "_sort_market": MARKET_PRIORITY.get(market, 99),
        "_sort_model_prob": model_prob if model_prob is not None else -1.0,
        "_sort_value_edge": value_edge if value_edge is not None else -999.0,
    }


def sort_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda record: (
            record["_sort_tier"],
            record["_sort_market"],
            -record["_sort_model_prob"],
            -record["_sort_value_edge"],
            record["league"],
            record["fixture_key"],
        ),
    )


def select_public_records(records: list[dict[str, Any]], counters: Counter[str]) -> list[dict[str, Any]]:
    elite = [record for record in records if record["confidence_tier"] == "ELITE"]
    if elite:
        return elite
    counters["public_fallback:standard_subset"] += 1
    standard = [record for record in records if record["confidence_tier"] == "STANDARD"]
    return standard[:5]


def project_fields(record: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: record[key] for key in keys}


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def build_report(
    source_path: Path,
    source_rows: int,
    premium_rows: list[dict[str, Any]],
    public_rows: list[dict[str, Any]],
    counters: Counter[str],
) -> str:
    tier_counts = Counter(record["confidence_tier"] for record in premium_rows)
    market_counts = Counter(record["market"] for record in premium_rows)
    fallback_lines = "\n".join(f"- `{key}`: {value}" for key, value in sorted(counters.items()) if value)
    if not fallback_lines:
        fallback_lines = "- none"

    return "\n".join(
        [
            "# PUBLISH_REPORT",
            "",
            f"Generated: `{utc_now_iso()}`",
            f"Selected source CSV: `{source_path}`",
            f"Source modified time (UTC): `{datetime.fromtimestamp(source_path.stat().st_mtime, tz=timezone.utc).isoformat()}`",
            "",
            "## Counts",
            f"- Source rows read: `{source_rows}`",
            f"- Premium predictions written: `{len(premium_rows)}`",
            f"- Public predictions written: `{len(public_rows)}`",
            "",
            "## Premium Tier Counts",
            *(f"- `{tier}`: `{count}`" for tier, count in sorted(tier_counts.items())),
            "",
            "## Premium Market Counts",
            *(f"- `{market}`: `{count}`" for market, count in sorted(market_counts.items())),
            "",
            "## Fallbacks And Drops",
            fallback_lines,
            "",
            "## Notes",
            "- Exporter uses a strict allowlist and excludes `OBSERVE` rows from premium output.",
            "- Public board prefers `ELITE` rows only.",
            "- If no `ELITE` rows exist, a small `STANDARD` fallback is used and recorded in publish summary.",
            "- Run `python3 validate_public_export.py` after publishing.",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish strict website-safe public and premium prediction JSON from a routed deploy CSV."
    )
    parser.add_argument(
        "--src",
        default="",
        help="Optional explicit DEPLOY_COMBINED CSV path. If omitted, the newest true DEPLOY_COMBINED_*.csv is used.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()
    counters: Counter[str] = Counter()

    source_path = resolve_source_path(str(args.src or "").strip() or None)
    source_rows = load_rows(source_path)
    built_records = [record for row in source_rows if (record := build_record(row, counters)) is not None]
    premium_records = sort_records(built_records)
    public_records = sort_records(select_public_records(premium_records, counters))

    public_payload = [project_fields(record, PUBLIC_FIELDS) for record in public_records]
    premium_payload = [project_fields(record, PREMIUM_FIELDS) for record in premium_records]

    public_path = FRONTEND_DATA_DIR / "public_predictions.json"
    premium_path = FRONTEND_DATA_DIR / "premium_predictions.json"
    summary_path = FRONTEND_DATA_DIR / "publish_summary.json"
    report_path = REPORTS_DIR / "PUBLISH_REPORT.md"

    write_json(public_path, public_payload)
    write_json(premium_path, premium_payload)

    summary = {
        "generated_at": utc_now_iso(),
        "selected_source_csv": str(source_path.relative_to(ROOT)),
        "selected_by": "explicit_src" if args.src else "latest_mtime",
        "selected_source_mtime_utc": datetime.fromtimestamp(source_path.stat().st_mtime, tz=timezone.utc).isoformat(),
        "source_rows_read": len(source_rows),
        "public_predictions_count": len(public_payload),
        "premium_predictions_count": len(premium_payload),
        "public_output": str(public_path.relative_to(ROOT)),
        "premium_output": str(premium_path.relative_to(ROOT)),
        "report_output": str(report_path.relative_to(ROOT)),
        "public_fields": PUBLIC_FIELDS,
        "premium_fields": PREMIUM_FIELDS,
        "fallbacks_and_drops": dict(sorted((key, value) for key, value in counters.items() if value)),
    }
    write_json(summary_path, summary)

    report_path.write_text(
        build_report(source_path, len(source_rows), premium_records, public_records, counters) + "\n",
        encoding="utf-8",
    )

    print(f"Selected source CSV: {source_path.relative_to(ROOT)}")
    print(f"Public predictions written: {len(public_payload)} -> {public_path.relative_to(ROOT)}")
    print(f"Premium predictions written: {len(premium_payload)} -> {premium_path.relative_to(ROOT)}")
    print(f"Publish summary written: {summary_path.relative_to(ROOT)}")
    print(f"Publish report written: {report_path.relative_to(ROOT)}")
    print("Next step: run `python3 validate_public_export.py`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
