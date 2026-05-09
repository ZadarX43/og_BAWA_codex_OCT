#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from publish_predictions import (
    DEFAULT_LOGO_MANIFEST,
    FRONTEND_DATA_DIR,
    REPORTS_DIR,
    ROOT,
    ensure_dirs,
    load_logo_manifest,
    load_rows,
    logo_assets_for,
    normalize_market,
    normalize_pick,
    normalize_tier,
    parse_float,
    pick_first,
    resolve_source_path,
    utc_now_iso,
    write_json,
)

OUTPUT_PATH = FRONTEND_DATA_DIR / "fixture_intelligence_public.json"
REPORT_PATH = REPORTS_DIR / "FIXTURE_INTELLIGENCE_REPORT.md"

PUBLISH_CLASS_ORDER = {"DEPLOY": 0, "OBSERVE": 1, "CONTEXT": 2, "MONITOR": 3, "HIDDEN": 4}
MARKET_PRIORITY = {"FTR": 0, "BTTS": 1, "OU25": 2, "CS": 3}
DEPLOY_TIER_PRIORITY = {"ELITE": 0, "STANDARD": 1}
OBSERVE_STRENGTH_PRIORITY = {"medium": 0, "low": 1, "none": 2}
ALLOWED_MARKETS = {"FTR", "BTTS", "OU25"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish website-safe fixture intelligence JSON from routed deploy and observe CSVs."
    )
    parser.add_argument(
        "--src",
        default="",
        help=(
            "Optional explicit routed CSV path. DEPLOY_PRESET files automatically expand to ELITE/STANDARD/OBSERVE "
            "siblings when they exist."
        ),
    )
    parser.add_argument(
        "--logo-manifest",
        default=str(DEFAULT_LOGO_MANIFEST),
        help="Optional API-Football logo manifest CSV. Use an empty string to disable logo enrichment.",
    )
    return parser.parse_args()


def normalize_kickoff_time(value: str, counters: Counter[str]) -> str:
    text = str(value or "").strip()
    if not text:
        counters["kickoff_time:missing"] += 1
        return ""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        counters["kickoff_time:date_only"] += 1
        return f"{text}T00:00:00Z"
    if text.endswith("Z"):
        return text
    if "T" in text:
        return f"{text}Z"
    return text


def parse_source_window(paths: list[Path]) -> dict[str, str]:
    dates: list[str] = []
    for path in paths:
        dates.extend(re.findall(r"\d{4}-\d{2}-\d{2}", path.name))
    if len(dates) >= 2:
        return {"date_from": dates[0], "date_to": dates[1]}
    return {"date_from": "", "date_to": ""}


def parse_source_run_id(paths: list[Path]) -> str:
    for path in paths:
        for part in path.parts[::-1]:
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", part):
                return part
    return paths[0].parent.name if paths else "unknown"


def expand_source_bundle(source_path: Path, counters: Counter[str]) -> list[Path]:
    name = source_path.name
    if "__DEPLOY_PRESET__" in name:
        bundle: list[Path] = []
        for tier in ("ELITE", "STANDARD", "OBSERVE"):
            candidate = source_path.with_name(name.replace("__DEPLOY_PRESET__", f"__DEPLOY_TIER_{tier}__"))
            if candidate.exists():
                bundle.append(candidate)
        if bundle:
            counters["source_bundle:expanded_from_preset"] += 1
            counters["source_bundle:files"] += len(bundle)
            return bundle

    tier_match = re.search(r"__DEPLOY_TIER_(ELITE|STANDARD|OBSERVE)__", name)
    if tier_match:
        prefix = name[: tier_match.start()]
        suffix = name[tier_match.end() :]
        bundle = []
        for tier in ("ELITE", "STANDARD", "OBSERVE"):
            candidate = source_path.with_name(f"{prefix}__DEPLOY_TIER_{tier}__{suffix}")
            if candidate.exists():
                bundle.append(candidate)
        if bundle:
            counters["source_bundle:expanded_from_tier"] += 1
            counters["source_bundle:files"] += len(bundle)
            return bundle

    counters["source_bundle:single_file"] += 1
    return [source_path]


def load_bundle_rows(paths: list[Path], counters: Counter[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for path in paths:
        file_rows = load_rows(path)
        counters[f"source_rows:{path.name}"] += len(file_rows)
        for row in file_rows:
            key = (
                str(row.get("fixture_key", "") or "").strip(),
                normalize_market(str(row.get("market", "") or "")),
                normalize_pick(str(row.get("selection", "") or row.get("bookie_pick", "") or "")),
                normalize_tier(str(row.get("deploy_tier", "") or row.get("tier", "") or "")),
            )
            if key in seen:
                counters["dedupe:source_row"] += 1
                continue
            seen.add(key)
            rows.append(row)
    return rows


def parse_reason_codes(row: dict[str, str]) -> list[str]:
    raw = str(row.get("context_reason_codes", "") or "").strip()
    if not raw:
        raw = str(row.get("deploy_reason_codes", "") or "").strip()
    tokens: list[str] = []
    if raw:
        for token in re.split(r"[|,;]+", raw):
            cleaned = re.sub(r"[^A-Za-z0-9_:-]+", "_", token.strip().upper()).strip("_")
            if cleaned and cleaned not in tokens:
                tokens.append(cleaned)
    return tokens


def parse_boolish(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def build_row_snapshot(row: dict[str, str], counters: Counter[str], logo_index: dict[str, Any]) -> dict[str, Any] | None:
    fixture_key = pick_first(row, ["fixture_key"], "fixture_key", counters)
    kickoff_time = normalize_kickoff_time(
        pick_first(row, ["kickoff_time", "match_date"], "kickoff_time", counters),
        counters,
    )
    league = pick_first(row, ["league"], "league", counters)
    home_team = pick_first(row, ["home_team_name", "home_team"], "home_team", counters)
    away_team = pick_first(row, ["away_team_name", "away_team"], "away_team", counters)
    market = normalize_market(pick_first(row, ["market"], "market", counters))
    pick = normalize_pick(pick_first(row, ["selection", "bookie_pick", "pick"], "pick", counters))
    tier = normalize_tier(pick_first(row, ["deploy_tier", "tier"], "tier", counters))

    required = [fixture_key, kickoff_time, league, home_team, away_team, market, tier]
    if any(not value for value in required):
        counters["dropped:critical_missing"] += 1
        return None

    if market not in ALLOWED_MARKETS:
        counters[f"dropped:unsupported_market:{market}"] += 1
        return None

    logo_assets = logo_assets_for(logo_index, league, home_team, away_team, counters)
    return {
        "fixture_id": fixture_key,
        "fixture_key": fixture_key,
        "kickoff_time": kickoff_time,
        "league": league,
        "home_team": home_team,
        "away_team": away_team,
        **logo_assets,
        "market": market,
        "pick": pick,
        "tier": tier,
        "model_prob": parse_float(row.get("model_p_for_bookie", "")),
        "bookie_implied": parse_float(row.get("bookie_implied", "")) or parse_float(row.get("bookie_implied_novig", "")),
        "value_edge": parse_float(row.get("value_edge", "")),
        "bookie_od": parse_float(row.get("bookie_od", "")),
        "home_win_odds": parse_float(row.get("od_home", "")),
        "draw_odds": parse_float(row.get("od_draw", "")),
        "away_win_odds": parse_float(row.get("od_away", "")),
        "btts_yes_odds": parse_float(row.get("od_yes", "")) or parse_float(row.get("odds_btts_yes", "")),
        "btts_no_odds": parse_float(row.get("od_no", "")) or parse_float(row.get("odds_btts_no", "")),
        "over25_odds": parse_float(row.get("od_over", "")) or parse_float(row.get("odds_ft_over25", "")),
        "under25_odds": parse_float(row.get("od_under", "")) or parse_float(row.get("odds_ft_under25", "")),
        "signal_btts": str(row.get("signal_btts", "") or "").strip().upper(),
        "signal_over25": str(row.get("signal_over25", "") or "").strip().upper(),
        "model_strength": parse_float(row.get("model_strength", "")),
        "p00_est": parse_float(row.get("p00_est", "")),
        "p_home_fts": parse_float(row.get("p_home_fts", "")),
        "p_away_fts": parse_float(row.get("p_away_fts", "")),
        "ftr_margin": parse_float(row.get("ftr_margin", "")),
        "power_diff": parse_float(row.get("power_diff", "")),
        "pick_side_mass_top3": parse_float(row.get("pick_side_mass_top3", "")),
        "pick_side_margin_top3": parse_float(row.get("pick_side_margin_top3", "")),
        "top3_over_count": parse_float(row.get("top3_over_count", "")),
        "h2h_n": parse_float(row.get("h2h_n", "")),
        "h2h_btts_rate": parse_float(row.get("h2h_btts_rate", "")),
        "h2h_over25_rate": parse_float(row.get("h2h_over25_rate", "")),
        "team_intel_overlay_action": str(row.get("team_intel_overlay_action", "") or "").strip().upper(),
        "team_intel_overlay_policy_bucket": str(row.get("team_intel_overlay_policy_bucket", "") or "").strip().upper(),
        "uefa_rotation_any": parse_boolish(row.get("uefa_rotation_any", "")),
        "uefa_goal_hunt_flag": parse_boolish(row.get("uefa_goal_hunt_flag", "")),
        "uefa_both_must_win": parse_boolish(row.get("uefa_both_must_win", "")),
        "reason_codes": parse_reason_codes(row),
    }


def value_edge_label(edge: float | None) -> str:
    if edge is None:
        return "unknown"
    if edge > 0.02:
        return "positive"
    if edge < -0.02:
        return "negative"
    return "neutral"


def deploy_signal_strength(tier: str, model_prob: float | None) -> str:
    if tier == "ELITE":
        return "high"
    if model_prob is not None and model_prob >= 0.65:
        return "high"
    return "medium"


def deploy_signal_label(row: dict[str, Any]) -> str:
    market = row["market"]
    tier = row["tier"].title()
    if market == "BTTS":
        return f"{tier} BTTS deployment"
    if market == "OU25":
        return f"{tier} scoring deployment"
    return f"{tier} result deployment"


def deploy_signal_summary(row: dict[str, Any]) -> str:
    market = row["market"]
    pick = row["pick"]
    tier = row["tier"]
    pick_text = pick.replace("_", " ").title() if pick else "Selection"
    if market == "FTR":
        return f"{tier.title()} result deployment remained live on {pick_text}."
    if market == "BTTS":
        return f"{tier.title()} BTTS deployment remained live on {pick_text}."
    return f"{tier.title()} scoring deployment remained live on {pick_text}."


def has_token(row: dict[str, Any], token: str) -> bool:
    return token in row["reason_codes"]


def observe_strength(row: dict[str, Any]) -> str:
    signal_text = " ".join([row["signal_btts"], row["signal_over25"]]).upper()
    if any(flag in signal_text for flag in ("VERY_STRONG", "STRONG")) or has_token(row, "BTTS_META_ELITE") or has_token(row, "VALUE_EDGE_PREMIUM"):
        return "medium"
    return "low"


def observe_context_tags(row: dict[str, Any]) -> list[str]:
    market = row["market"]
    pick = row["pick"]
    tags: list[str] = []
    if market == "BTTS":
        if pick == "YES":
            tags.extend(["attacking_shape", "defensive_instability"])
        else:
            tags.append("low_event_risk")
    elif market == "OU25":
        if pick == "OVER25":
            tags.extend(["goal_shape_positive", "high_event_potential"])
        else:
            tags.extend(["goal_shape_muted", "low_event_risk"])
    elif market == "FTR":
        if pick == "HOME":
            tags.append("home_side_lean")
        elif pick == "AWAY":
            tags.append("away_side_lean")
        else:
            tags.append("draw_risk")

    if row["team_intel_overlay_action"] == "CAUTION" or has_token(row, "TEAM_INTEL_CAUTION") or has_token(row, "TEAM_INTEL_AVOID_IN_ACCA"):
        tags.append("volatility")
    if row["uefa_rotation_any"]:
        tags.append("lineup_pending")
    tags.append("not_deployable")

    deduped: list[str] = []
    for tag in tags:
        if tag not in deduped:
            deduped.append(tag)
    return deduped[:4]


def classify_observe_row(row: dict[str, Any]) -> dict[str, Any]:
    market = row["market"]
    pick = row["pick"]
    strength = observe_strength(row)
    tags = observe_context_tags(row)
    caution = row["team_intel_overlay_action"] == "CAUTION" or has_token(row, "TEAM_INTEL_CAUTION")

    if market == "BTTS":
        if pick == "YES":
            return {
                "publish_class": "OBSERVE",
                "signal_state": "observe",
                "signal_label": "Observed BTTS lean",
                "signal_strength": strength,
                "summary_text": "Observed BTTS lean based on attacking shape, but not enough stability for deployment."
                if not caution
                else "Observed BTTS lean remained active, but fragility kept it out of deployment.",
                "context_tags": tags,
            }
        if pick == "NO":
            return {
                "publish_class": "OBSERVE",
                "signal_state": "observe",
                "signal_label": "Observed low-event BTTS profile",
                "signal_strength": strength,
                "summary_text": "Observed lower-event BTTS profile, but the row remained in monitoring rather than deployment."
                if not caution
                else "Observed lower-event BTTS profile, but caution kept it outside deployment.",
                "context_tags": tags,
            }

    if market == "OU25":
        if pick == "OVER25":
            return {
                "publish_class": "OBSERVE",
                "signal_state": "observe",
                "signal_label": "Observed scoring lean",
                "signal_strength": strength,
                "summary_text": "Goal-shape profile suggested elevated scoring potential, but structural support remained too weak for deployment."
                if not caution
                else "Observed scoring lean remained active, but fragility kept it out of deployment.",
                "context_tags": tags,
            }
        if pick == "UNDER25":
            return {
                "publish_class": "OBSERVE",
                "signal_state": "observe",
                "signal_label": "Observed low-event scoring profile",
                "signal_strength": strength,
                "summary_text": "Observed lower-event scoring shape, but not enough support remained for deployment."
                if not caution
                else "Observed lower-event scoring shape, but caution kept it out of deployment.",
                "context_tags": tags,
            }

    if market == "FTR":
        if pick == "HOME":
            return {
                "publish_class": "OBSERVE",
                "signal_state": "observe",
                "signal_label": "Observed home-side lean",
                "signal_strength": strength,
                "summary_text": "Observed home-side lean, but side support was not strong enough for deployment."
                if not caution
                else "Observed home-side lean remained active, but fragility kept it out of deployment.",
                "context_tags": tags,
            }
        if pick == "AWAY":
            return {
                "publish_class": "OBSERVE",
                "signal_state": "observe",
                "signal_label": "Observed away-side lean",
                "signal_strength": strength,
                "summary_text": "Observed away-side lean, with enough shape to monitor but not enough stability to deploy."
                if not caution
                else "Observed away-side lean remained active, but fragility kept it out of deployment.",
                "context_tags": tags,
            }
        return {
            "publish_class": "CONTEXT",
            "signal_state": "context_only",
            "signal_label": "",
            "signal_strength": "none",
            "summary_text": "No deployable edge. Draw and volatility context remained relevant.",
            "context_tags": ["draw_risk", "volatility"],
        }

    if caution:
        return {
            "publish_class": "CONTEXT",
            "signal_state": "context_only",
            "signal_label": "",
            "signal_strength": "none",
            "summary_text": "No deployable edge. Team-intelligence caution remained relevant.",
            "context_tags": ["volatility"],
        }

    return {
        "publish_class": "MONITOR",
        "signal_state": "monitor_only",
        "signal_label": "",
        "signal_strength": "none",
        "summary_text": "Covered fixture. No strong deploy signal at current state.",
        "context_tags": [],
    }


def build_context_summary(rows: list[dict[str, Any]]) -> dict[str, str]:
    summary: dict[str, str] = {}

    if any(row["team_intel_overlay_action"] == "CAUTION" or has_token(row, "TEAM_INTEL_CAUTION") for row in rows):
        summary["volatility_note"] = "Team-intelligence caution remains active around this fixture."

    if any(has_token(row, "TEAM_HOME_GE2_POCKET") for row in rows):
        summary["form_note"] = "Home-side scoring context remains active."
    elif any(has_token(row, "TEAM_AWAY_GE2_POCKET") for row in rows):
        summary["form_note"] = "Away-side scoring context remains active."

    if any(row["uefa_rotation_any"] for row in rows):
        summary["fatigue_note"] = "Rotation risk remains active around this fixture."

    if any(row["uefa_both_must_win"] or row["uefa_goal_hunt_flag"] for row in rows):
        summary["schedule_note"] = "Competitive pressure remains elevated around this fixture."

    h2h_n = max((row["h2h_n"] for row in rows if row["h2h_n"] is not None), default=None)
    h2h_over = max((row["h2h_over25_rate"] for row in rows if row["h2h_over25_rate"] is not None), default=None)
    h2h_btts = max((row["h2h_btts_rate"] for row in rows if row["h2h_btts_rate"] is not None), default=None)
    if h2h_n is not None and h2h_n >= 3:
        if h2h_over is not None and h2h_over >= 0.65:
            summary["h2h_note"] = "Recent meetings have leaned high-event."
        elif h2h_btts is not None and h2h_btts >= 0.65:
            summary["h2h_note"] = "Recent meetings have shown a regular both-teams-to-score pattern."
        elif h2h_over is not None and h2h_over <= 0.35:
            summary["h2h_note"] = "Recent meetings have leaned lower-event."

    return summary


def build_odds_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    odds = {
        "home_win_odds": None,
        "draw_odds": None,
        "away_win_odds": None,
        "btts_yes_odds": None,
        "btts_no_odds": None,
        "over25_odds": None,
        "under25_odds": None,
    }
    for row in rows:
        for key in list(odds.keys()):
            if odds[key] is None and row.get(key) is not None:
                odds[key] = round(row[key], 4)
    available = sum(1 for value in odds.values() if value is not None)
    odds["odds_snapshot_status"] = "available" if available >= 4 else "partial" if available > 0 else "missing"
    return odds


def notification_priority(publish_class: str, signal_strength: str, confidence_tier: str | None) -> str:
    if publish_class == "DEPLOY":
        return "critical" if confidence_tier == "ELITE" else "high"
    if publish_class == "OBSERVE":
        return "high" if signal_strength == "medium" else "medium"
    if publish_class == "CONTEXT":
        return "medium"
    if publish_class == "MONITOR":
        return "low"
    return "none"


def select_primary_deploy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        rows,
        key=lambda row: (
            DEPLOY_TIER_PRIORITY.get(row["tier"], 99),
            MARKET_PRIORITY.get(row["market"], 99),
            -(row["model_prob"] if row["model_prob"] is not None else -1.0),
            -(row["value_edge"] if row["value_edge"] is not None else -999.0),
        ),
    )[0]


def select_primary_nondeploy(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    translated = [(row, classify_observe_row(row)) for row in rows]
    return sorted(
        translated,
        key=lambda item: (
            PUBLISH_CLASS_ORDER[item[1]["publish_class"]],
            OBSERVE_STRENGTH_PRIORITY.get(item[1]["signal_strength"], 99),
            MARKET_PRIORITY.get(item[0]["market"], 99),
            -(item[0]["model_prob"] if item[0]["model_prob"] is not None else -1.0),
            -(item[0]["value_edge"] if item[0]["value_edge"] is not None else -999.0),
        ),
    )[0]


def build_fixture_record(
    fixture_rows: list[dict[str, Any]],
    generated_at: str,
    counters: Counter[str],
) -> dict[str, Any] | None:
    deploy_rows = [row for row in fixture_rows if row["tier"] in {"ELITE", "STANDARD"}]
    observe_rows = [row for row in fixture_rows if row["tier"] == "OBSERVE"]
    context_summary = build_context_summary(fixture_rows)
    odds_summary = build_odds_summary(fixture_rows)

    base = fixture_rows[0]
    record: dict[str, Any] = {
        "fixture_id": base["fixture_id"],
        "fixture_key": base["fixture_key"],
        "publish_class": "",
        "coverage_status": "covered",
        "kickoff_time": base["kickoff_time"],
        "league": base["league"],
        "league_logo_url": base["league_logo_url"],
        "league_flag_url": base["league_flag_url"],
        "home_team": base["home_team"],
        "home_team_logo_url": base["home_team_logo_url"],
        "away_team": base["away_team"],
        "away_team_logo_url": base["away_team_logo_url"],
        "logo_join_status": base["logo_join_status"],
        "odds_summary": odds_summary,
        "signal_summary": {},
        "context_summary": context_summary,
        "follow_relevance": {},
        "updated_at": generated_at,
    }

    if deploy_rows:
        primary = select_primary_deploy(deploy_rows)
        record["publish_class"] = "DEPLOY"
        record["signal_summary"] = {
            "signal_state": "deploy",
            "market_family": primary["market"],
            "signal_label": deploy_signal_label(primary),
            "signal_strength": deploy_signal_strength(primary["tier"], primary["model_prob"]),
            "confidence_tier": primary["tier"],
            "deploy_pick": primary["pick"],
            "premium_tier": primary["tier"],
            "summary_text": deploy_signal_summary(primary),
            "context_tags": [],
        }
        record["deploy_summary"] = {
            "market": primary["market"],
            "pick": primary["pick"],
            "confidence_tier": primary["tier"],
            "bookie_od": round(primary["bookie_od"], 4) if primary["bookie_od"] is not None else None,
            "value_edge_label": value_edge_label(primary["value_edge"]),
        }
    elif observe_rows:
        primary, translated = select_primary_nondeploy(observe_rows)
        publish_class = translated["publish_class"]
        if publish_class == "HIDDEN":
            counters["hidden:observe_only"] += 1
            return None
        record["publish_class"] = publish_class
        record["signal_summary"] = {
            "signal_state": translated["signal_state"],
            "market_family": primary["market"],
            "signal_label": translated["signal_label"],
            "signal_strength": translated["signal_strength"],
            "confidence_tier": None,
            "deploy_pick": None,
            "premium_tier": None,
            "summary_text": translated["summary_text"],
            "context_tags": translated["context_tags"],
        }
    else:
        counters["hidden:no_supported_rows"] += 1
        return None

    priority = notification_priority(
        record["publish_class"],
        str(record["signal_summary"].get("signal_strength", "none")),
        record["signal_summary"].get("confidence_tier"),
    )
    record["follow_relevance"] = {
        "team_follow_candidate": True,
        "fixture_follow_candidate": True,
        "league_follow_candidate": True,
        "market_follow_candidate": record["signal_summary"].get("market_family") in ALLOWED_MARKETS,
        "notification_priority": priority,
    }

    counters[f"publish_class:{record['publish_class']}"] += 1
    return record


def sort_fixture_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda record: (
            record["kickoff_time"],
            PUBLISH_CLASS_ORDER[record["publish_class"]],
            record["league"],
            record["home_team"],
            record["away_team"],
        ),
    )


def build_report(
    source_paths: list[Path],
    source_rows: int,
    fixture_records: list[dict[str, Any]],
    counters: Counter[str],
    source_run_id: str,
    source_window: dict[str, str],
) -> str:
    class_counts = Counter(record["publish_class"] for record in fixture_records)
    market_counts = Counter(record["signal_summary"].get("market_family", "") for record in fixture_records)
    fallback_lines = "\n".join(f"- `{key}`: `{value}`" for key, value in sorted(counters.items()) if value)
    if not fallback_lines:
        fallback_lines = "- none"

    return "\n".join(
        [
            "# FIXTURE_INTELLIGENCE_REPORT",
            "",
            f"Generated: `{utc_now_iso()}`",
            f"Source run id: `{source_run_id}`",
            f"Source window: `{source_window.get('date_from', '')}` to `{source_window.get('date_to', '')}`",
            "",
            "## Source Files",
            *(f"- `{path.relative_to(ROOT)}`" for path in source_paths),
            "",
            "## Counts",
            f"- Source rows read: `{source_rows}`",
            f"- Published fixtures written: `{len(fixture_records)}`",
            "",
            "## Publish Class Counts",
            *(f"- `{publish_class}`: `{count}`" for publish_class, count in sorted(class_counts.items())),
            "",
            "## Primary Market Counts",
            *(f"- `{market}`: `{count}`" for market, count in sorted(market_counts.items()) if market),
            "",
            "## Fallbacks And Drops",
            fallback_lines,
            "",
            "## Notes",
            "- Exporter is additive only and does not alter deploy routing.",
            "- `OBSERVE` rows are translated into public-safe intelligence, never premium picks.",
            "- Current first exporter covers routed fixtures present in the selected bundle.",
            "- Run `python3 validate_fixture_intelligence.py` after publishing.",
        ]
    )


def main() -> int:
    args = parse_args()
    ensure_dirs()
    counters: Counter[str] = Counter()

    source_path = resolve_source_path(str(args.src or "").strip() or None)
    source_paths = expand_source_bundle(source_path, counters)
    source_window = parse_source_window(source_paths)
    source_run_id = parse_source_run_id(source_paths)

    logo_manifest_text = str(args.logo_manifest or "").strip()
    if logo_manifest_text:
        logo_manifest_path = (
            (ROOT / logo_manifest_text).resolve()
            if not Path(logo_manifest_text).is_absolute()
            else Path(logo_manifest_text)
        )
    else:
        logo_manifest_path = None
    logo_index = load_logo_manifest(logo_manifest_path, counters)

    source_rows = load_bundle_rows(source_paths, counters)
    row_snapshots = [
        snapshot
        for row in source_rows
        if (snapshot := build_row_snapshot(row, counters, logo_index)) is not None
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for snapshot in row_snapshots:
        grouped[snapshot["fixture_key"]].append(snapshot)

    generated_at = utc_now_iso()
    fixture_records = [
        record
        for fixture_rows in grouped.values()
        if (record := build_fixture_record(fixture_rows, generated_at, counters)) is not None
    ]
    fixture_records = sort_fixture_records(fixture_records)

    coverage_summary = {
        "total_fixtures": len(fixture_records),
        "deploy_count": sum(1 for record in fixture_records if record["publish_class"] == "DEPLOY"),
        "observe_count": sum(1 for record in fixture_records if record["publish_class"] == "OBSERVE"),
        "context_count": sum(1 for record in fixture_records if record["publish_class"] == "CONTEXT"),
        "monitor_count": sum(1 for record in fixture_records if record["publish_class"] == "MONITOR"),
        "hidden_count": int(counters.get("hidden:observe_only", 0) + counters.get("hidden:no_supported_rows", 0)),
        "covered_leagues_count": len({record["league"] for record in fixture_records}),
    }

    payload = {
        "generated_at": generated_at,
        "source_run_id": source_run_id,
        "source_window": source_window,
        "coverage_summary": coverage_summary,
        "fixtures": fixture_records,
    }
    write_json(OUTPUT_PATH, payload)
    REPORT_PATH.write_text(
        build_report(source_paths, len(source_rows), fixture_records, counters, source_run_id, source_window) + "\n",
        encoding="utf-8",
    )

    print(f"Selected source bundle: {', '.join(str(path.relative_to(ROOT)) for path in source_paths)}")
    print(f"Fixture intelligence written: {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"Fixture intelligence report written: {REPORT_PATH.relative_to(ROOT)}")
    print("Next step: run `python3 validate_fixture_intelligence.py`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
