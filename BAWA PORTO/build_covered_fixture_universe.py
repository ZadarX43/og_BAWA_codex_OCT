#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
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
    normalize_lookup_text,
    utc_now_iso,
    write_json,
)

OUTPUT_PATH = FRONTEND_DATA_DIR / "covered_fixture_universe.json"
REPORT_PATH = REPORTS_DIR / "COVERED_FIXTURE_UNIVERSE_REPORT.md"
FIXTURES_MASTER_PATH = ROOT / "data_sources" / "api_football" / "normalized" / "fixtures_master.csv"
INJURIES_PATH = ROOT / "data_sources" / "api_football" / "normalized" / "injuries.csv"
LINEUPS_PATH = ROOT / "data_sources" / "api_football" / "normalized" / "lineups.csv"
MATCH_TEAM_STATS_PATH = ROOT / "data_sources" / "api_football" / "normalized" / "match_team_stats.csv"
MATCH_PLAYER_STATS_PATH = ROOT / "data_sources" / "api_football" / "normalized" / "match_player_stats.csv"
MATCH_EVENTS_PATH = ROOT / "data_sources" / "api_football" / "normalized" / "match_events.csv"
FEATURES_DIR = ROOT / "data_sources" / "api_football" / "features"
PLAYER_EVENTS_FEATURES_DIR = FEATURES_DIR / "player_events"
CURRENT_CONTEXT_OVERLAY_SUMMARY_PATH = ROOT / "reports" / "latest" / "api_current_context_overlay_window" / "CURRENT_CONTEXT_OVERLAY_SUMMARY.json"

LEAGUE_TAG_ALIASES: dict[str, list[str]] = {
    "belgium pro": ["Belgium_Pro"],
    "brazil serie a": ["Brazil_Serie_A"],
    "england premier league": ["England_Premier_League"],
    "england championship": ["England_Championship"],
    "france ligue 1": ["France_Ligue_1"],
    "germany bundesliga": ["Germany_Bundesliga"],
    "italy serie a": ["Italy_Serie_A"],
    "netherlands eredivisie": ["Netherlands_Eredivisie"],
    "portugal liga": ["Portugal_Liga"],
    "scotland premiership": ["Scotland_Premiership"],
    "spain la liga": ["Spain_La_Liga"],
    "usa mls": ["USA_MLS"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the covered upcoming fixture universe for the active publish window."
    )
    parser.add_argument(
        "--src",
        default="",
        help=(
            "Optional current-window ALLMARKETS or DEPLOY CSV path. DEPLOY paths automatically resolve to the sibling "
            "ALLMARKETS source and routed tier files."
        ),
    )
    parser.add_argument(
        "--logo-manifest",
        default=str(DEFAULT_LOGO_MANIFEST),
        help="Optional API-Football logo manifest CSV. Use an empty string to disable logo enrichment.",
    )
    return parser.parse_args()


def resolve_source_path(src: str | None) -> Path:
    if src:
        candidate = Path(src)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        candidate = candidate.resolve()
        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError(f"Explicit --src file not found: {candidate}")
        return candidate
    runs = sorted(
        (path for path in (ROOT / "predictions_output").iterdir() if path.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", path.name)),
        key=lambda path: path.name,
    )
    if not runs:
        raise FileNotFoundError("No dated predictions_output directories found.")
    latest_dir = runs[-1]
    matches = sorted(latest_dir.glob("BOOKIE_*ALLMARKETS_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No ALLMARKETS CSV found under {latest_dir}")
    return matches[-1]


def resolve_allmarkets_source(path: Path) -> Path:
    name = path.name
    if "__DEPLOY_" in name:
        candidate = path.with_name(name.split("__DEPLOY_")[0] + ".csv")
        if candidate.exists():
            return candidate
    return path


def resolve_related_paths(allmarkets_path: Path) -> dict[str, Path]:
    stem = allmarkets_path.name[:-4] if allmarkets_path.name.endswith(".csv") else allmarkets_path.name
    base_dir = allmarkets_path.parent
    window = extract_window_from_name(allmarkets_path.name)

    def find_tier_file(tier: str) -> Path:
        matches = sorted(base_dir.glob(f"{stem}__DEPLOY_TIER_{tier}__*.csv"))
        return matches[-1] if matches else base_dir / f"{stem}__DEPLOY_TIER_{tier}__.csv"

    return {
        "allmarkets": allmarkets_path,
        "deploy_candidates_raw": base_dir / "DEPLOY_CANDIDATES_RAW.csv",
        "deploy_candidates_after_gates": base_dir / "DEPLOY_CANDIDATES_AFTER_GATES.csv",
        "deploy_elite": find_tier_file("ELITE"),
        "deploy_standard": find_tier_file("STANDARD"),
        "deploy_observe": find_tier_file("OBSERVE"),
        "loss_details": base_dir / f"PRE_ALLMARKETS_FIXTURE_LOSS_DETAILS_{window['date_from']}_to_{window['date_to']}.csv",
        "loss_report": base_dir / f"PRE_ALLMARKETS_FIXTURE_LOSS_REPORT_{window['date_from']}_to_{window['date_to']}.csv",
    }


def extract_window_from_name(name: str) -> dict[str, str]:
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", name)
    if len(dates) >= 2:
        return {"date_from": dates[0], "date_to": dates[1]}
    return {"date_from": "", "date_to": ""}


def parse_source_run_id(path: Path) -> str:
    for part in path.parts[::-1]:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", part):
            return part
    return path.parent.name


def normalize_kickoff(match_date: str) -> str:
    text = str(match_date or "").strip()
    if not text:
        return ""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return f"{text}T00:00:00Z"
    if text.endswith("Z"):
        return text
    return f"{text}Z" if "T" in text else text


def fixture_record_key(league: str, match_date: str, home_team: str, away_team: str) -> tuple[str, str, str, str]:
    return (
        normalize_lookup_text(league),
        str(match_date or "").strip(),
        normalize_lookup_text(home_team),
        normalize_lookup_text(away_team),
    )


def load_fixtures_master_index() -> dict[tuple[str, str, str, str], dict[str, str]]:
    if not FIXTURES_MASTER_PATH.exists():
        return {}
    with FIXTURES_MASTER_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        index: dict[tuple[str, str, str, str], dict[str, str]] = {}
        for row in reader:
            key = fixture_record_key(
                str(row.get("league", "") or ""),
                str(row.get("match_date", "") or ""),
                str(row.get("home_team_name", "") or ""),
                str(row.get("away_team_name", "") or ""),
            )
            index.setdefault(key, row)
        return index


def load_fixture_id_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    rows = load_rows(path)
    values: set[str] = set()
    for row in rows:
        fixture_id = str(row.get("fixture_id", "") or "").strip()
        if fixture_id:
            values.add(fixture_id)
    return values


def load_team_profile_index(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    rows = load_rows(path)
    aggregates: dict[str, dict[str, float]] = {}
    for row in rows:
        team_name = str(row.get("team_name", "") or "").strip()
        if not team_name:
            continue
        key = normalize_lookup_text(team_name)
        agg = aggregates.setdefault(
            key,
            {
                "matches": 0.0,
                "shots_total_sum": 0.0,
                "shots_on_goal_sum": 0.0,
                "goals_for_sum": 0.0,
                "goals_against_sum": 0.0,
                "yellow_cards_sum": 0.0,
            },
        )
        agg["matches"] += 1.0
        for field, target in (
            ("shots_total", "shots_total_sum"),
            ("shots_on_goal", "shots_on_goal_sum"),
            ("goals_for", "goals_for_sum"),
            ("goals_against", "goals_against_sum"),
            ("yellow_cards", "yellow_cards_sum"),
        ):
            try:
                agg[target] += float(str(row.get(field, "") or "0").strip() or 0.0)
            except ValueError:
                continue
    profiles: dict[str, dict[str, float]] = {}
    for key, agg in aggregates.items():
        matches = agg["matches"]
        if matches <= 0:
            continue
        profiles[key] = {
            "matches": matches,
            "shots_total_avg": round(agg["shots_total_sum"] / matches, 2),
            "shots_on_goal_avg": round(agg["shots_on_goal_sum"] / matches, 2),
            "goals_for_avg": round(agg["goals_for_sum"] / matches, 2),
            "goals_against_avg": round(agg["goals_against_sum"] / matches, 2),
            "yellow_cards_avg": round(agg["yellow_cards_sum"] / matches, 2),
        }
    return profiles


def league_tag_candidates(league: str) -> list[str]:
    normalized = normalize_lookup_text(league)
    tags = list(LEAGUE_TAG_ALIASES.get(normalized, []))
    fallback = re.sub(r"[^A-Za-z0-9]+", "_", str(league or "").strip()).strip("_")
    if fallback:
        tags.append(fallback)
    seen: set[str] = set()
    ordered: list[str] = []
    for tag in tags:
        if tag and tag not in seen:
            seen.add(tag)
            ordered.append(tag)
    return ordered


def _accumulate_overlay_side(
    aggregates: dict[str, dict[str, float]],
    team_name: str,
    *,
    injury_count: float = 0.0,
    suspended_count: float = 0.0,
    missing_attackers: float = 0.0,
    missing_defenders: float = 0.0,
    ppg_l5: float = 0.0,
    ppg_season: float = 0.0,
    win_rate_l5: float = 0.0,
    attacking_shape: float = 0.0,
    defensive_shape: float = 0.0,
    shots_l5: float = 0.0,
    shots_on_goal_l5: float = 0.0,
    fouls_l5: float = 0.0,
    tackles_l5: float = 0.0,
    goal_environment_score: float = 0.0,
    battle_on_score: float = 0.0,
) -> None:
    if not team_name:
        return
    key = normalize_lookup_text(team_name)
    agg = aggregates.setdefault(
        key,
        {
            "matches": 0.0,
            "injury_count_sum": 0.0,
            "suspended_count_sum": 0.0,
            "missing_attackers_sum": 0.0,
            "missing_defenders_sum": 0.0,
            "ppg_l5_sum": 0.0,
            "ppg_season_sum": 0.0,
            "win_rate_l5_sum": 0.0,
            "attacking_shape_sum": 0.0,
            "defensive_shape_sum": 0.0,
            "shots_l5_sum": 0.0,
            "shots_on_goal_l5_sum": 0.0,
            "fouls_l5_sum": 0.0,
            "tackles_l5_sum": 0.0,
            "goal_environment_sum": 0.0,
            "battle_on_sum": 0.0,
        },
    )
    agg["matches"] += 1.0
    agg["injury_count_sum"] += injury_count
    agg["suspended_count_sum"] += suspended_count
    agg["missing_attackers_sum"] += missing_attackers
    agg["missing_defenders_sum"] += missing_defenders
    agg["ppg_l5_sum"] += ppg_l5
    agg["ppg_season_sum"] += ppg_season
    agg["win_rate_l5_sum"] += win_rate_l5
    agg["attacking_shape_sum"] += attacking_shape
    agg["defensive_shape_sum"] += defensive_shape
    agg["shots_l5_sum"] += shots_l5
    agg["shots_on_goal_l5_sum"] += shots_on_goal_l5
    agg["fouls_l5_sum"] += fouls_l5
    agg["tackles_l5_sum"] += tackles_l5
    agg["goal_environment_sum"] += goal_environment_score
    agg["battle_on_sum"] += battle_on_score


def _safe_float(row: dict[str, Any], key: str) -> float:
    try:
        return float(str(row.get(key, "") or "0").strip() or 0.0)
    except ValueError:
        return 0.0


def load_historical_overlay_profile_index() -> dict[str, dict[str, dict[str, float]]]:
    profiles_by_tag: dict[str, dict[str, dict[str, float]]] = {}
    for tag in sorted({tag for tags in LEAGUE_TAG_ALIASES.values() for tag in tags}):
        aggregates: dict[str, dict[str, float]] = {}
        injury_path = FEATURES_DIR / f"api_injury_features__{tag}__2024.csv"
        lineup_path = FEATURES_DIR / f"api_lineup_features__{tag}__2024.csv"
        rolling_path = FEATURES_DIR / f"api_team_rolling_features__{tag}__2024.csv"
        style_path = PLAYER_EVENTS_FEATURES_DIR / f"fixture_style_overlay__{tag}__2024.csv"
        goal_env_path = PLAYER_EVENTS_FEATURES_DIR / f"og_goal_environment_overlay__{tag}__2024.csv"

        for path in (injury_path, lineup_path, rolling_path, style_path, goal_env_path):
            if not path.exists():
                continue
            for row in load_rows(path):
                home_team = str(row.get("home_team_name", "") or "").strip()
                away_team = str(row.get("away_team_name", "") or "").strip()
                if path == injury_path:
                    _accumulate_overlay_side(
                        aggregates,
                        home_team,
                        injury_count=_safe_float(row, "home_injured_players_count"),
                        suspended_count=_safe_float(row, "home_suspended_players_count"),
                        missing_attackers=_safe_float(row, "home_missing_attackers_count"),
                        missing_defenders=_safe_float(row, "home_missing_defenders_count"),
                    )
                    _accumulate_overlay_side(
                        aggregates,
                        away_team,
                        injury_count=_safe_float(row, "away_injured_players_count"),
                        suspended_count=_safe_float(row, "away_suspended_players_count"),
                        missing_attackers=_safe_float(row, "away_missing_attackers_count"),
                        missing_defenders=_safe_float(row, "away_missing_defenders_count"),
                    )
                elif path == lineup_path:
                    _accumulate_overlay_side(
                        aggregates,
                        home_team,
                        attacking_shape=_safe_float(row, "home_attacking_shape_score"),
                        defensive_shape=_safe_float(row, "home_defensive_shape_score"),
                    )
                    _accumulate_overlay_side(
                        aggregates,
                        away_team,
                        attacking_shape=_safe_float(row, "away_attacking_shape_score"),
                        defensive_shape=_safe_float(row, "away_defensive_shape_score"),
                    )
                elif path == rolling_path:
                    _accumulate_overlay_side(
                        aggregates,
                        home_team,
                        ppg_l5=_safe_float(row, "home_team_ppg_l5"),
                        ppg_season=_safe_float(row, "home_team_ppg_season"),
                        win_rate_l5=_safe_float(row, "home_team_win_rate_l5"),
                    )
                    _accumulate_overlay_side(
                        aggregates,
                        away_team,
                        ppg_l5=_safe_float(row, "away_team_ppg_l5"),
                        ppg_season=_safe_float(row, "away_team_ppg_season"),
                        win_rate_l5=_safe_float(row, "away_team_win_rate_l5"),
                    )
                elif path == style_path:
                    _accumulate_overlay_side(
                        aggregates,
                        home_team,
                        shots_l5=_safe_float(row, "home_team_shots_l5"),
                        shots_on_goal_l5=_safe_float(row, "home_team_shots_on_goal_l5"),
                        fouls_l5=_safe_float(row, "home_team_fouls_l5"),
                        tackles_l5=_safe_float(row, "home_team_tackles_l5"),
                    )
                    _accumulate_overlay_side(
                        aggregates,
                        away_team,
                        shots_l5=_safe_float(row, "away_team_shots_l5"),
                        shots_on_goal_l5=_safe_float(row, "away_team_shots_on_goal_l5"),
                        fouls_l5=_safe_float(row, "away_team_fouls_l5"),
                        tackles_l5=_safe_float(row, "away_team_tackles_l5"),
                    )
                elif path == goal_env_path:
                    goal_environment_score = _safe_float(row, "og_goal_environment_score")
                    battle_on_score = _safe_float(row, "og_battle_on_score")
                    _accumulate_overlay_side(
                        aggregates,
                        home_team,
                        goal_environment_score=goal_environment_score,
                        battle_on_score=battle_on_score,
                    )
                    _accumulate_overlay_side(
                        aggregates,
                        away_team,
                        goal_environment_score=goal_environment_score,
                        battle_on_score=battle_on_score,
                    )

        profiles: dict[str, dict[str, float]] = {}
        for key, agg in aggregates.items():
            matches = agg["matches"]
            if matches <= 0:
                continue
            profiles[key] = {
                "matches": matches,
                "injury_count_avg": round(agg["injury_count_sum"] / matches, 2),
                "suspended_count_avg": round(agg["suspended_count_sum"] / matches, 2),
                "missing_attackers_avg": round(agg["missing_attackers_sum"] / matches, 2),
                "missing_defenders_avg": round(agg["missing_defenders_sum"] / matches, 2),
                "ppg_l5_avg": round(agg["ppg_l5_sum"] / matches, 2),
                "ppg_season_avg": round(agg["ppg_season_sum"] / matches, 2),
                "win_rate_l5_avg": round(agg["win_rate_l5_sum"] / matches, 2),
                "attacking_shape_avg": round(agg["attacking_shape_sum"] / matches, 2),
                "defensive_shape_avg": round(agg["defensive_shape_sum"] / matches, 2),
                "shots_l5_avg": round(agg["shots_l5_sum"] / matches, 2),
                "shots_on_goal_l5_avg": round(agg["shots_on_goal_l5_sum"] / matches, 2),
                "fouls_l5_avg": round(agg["fouls_l5_sum"] / matches, 2),
                "tackles_l5_avg": round(agg["tackles_l5_sum"] / matches, 2),
                "goal_environment_score_avg": round(agg["goal_environment_sum"] / matches, 2),
                "battle_on_score_avg": round(agg["battle_on_sum"] / matches, 2),
            }
        profiles_by_tag[tag] = profiles
    return profiles_by_tag


def load_current_context_overlay_index(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    rows = payload.get("fixtures", [])
    if not isinstance(rows, list):
        return {}
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        fixture_key = str(row.get("fixture_key", "") or "").strip()
        if fixture_key:
            index[fixture_key] = row
    return index


def read_fixture_keys(path: Path, key_fields: tuple[str, str, str, str] = ("league", "match_date", "home_team_name", "away_team_name")) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    rows = load_rows(path)
    keyed: dict[str, dict[str, str]] = {}
    for row in rows:
        fixture_key = str(row.get("fixture_key", "") or "").strip()
        if not fixture_key:
            continue
        keyed.setdefault(fixture_key, row)
    return keyed


def build_base_record(
    fixture_key: str,
    league: str,
    match_date: str,
    home_team: str,
    away_team: str,
    identity_source: str,
    logo_index: dict[str, Any],
    counters: Counter[str],
    fixtures_master_index: dict[tuple[str, str, str, str], dict[str, str]],
    injuries_fixture_ids: set[str],
    lineups_fixture_ids: set[str],
    player_stats_fixture_ids: set[str],
    match_events_fixture_ids: set[str],
    team_profile_index: dict[str, dict[str, float]],
    historical_overlay_profiles: dict[str, dict[str, dict[str, float]]],
    current_overlay_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    kickoff_time = normalize_kickoff(match_date)
    master_match = fixtures_master_index.get(fixture_record_key(league, match_date, home_team, away_team))
    logo_assets = logo_assets_for(logo_index, league, home_team, away_team, counters)
    fixture_id = str(master_match.get("fixture_id", "") or "").strip() if master_match else ""
    home_profile = team_profile_index.get(normalize_lookup_text(home_team))
    away_profile = team_profile_index.get(normalize_lookup_text(away_team))
    current_overlay = current_overlay_index.get(fixture_key, {})
    current_overlay_availability = current_overlay.get("availability", {}) if isinstance(current_overlay.get("availability"), dict) else {}
    overlay_home_profile: dict[str, float] = {}
    overlay_away_profile: dict[str, float] = {}
    for tag in league_tag_candidates(league):
        profile_index = historical_overlay_profiles.get(tag, {})
        if not overlay_home_profile:
            overlay_home_profile = profile_index.get(normalize_lookup_text(home_team), {})
        if not overlay_away_profile:
            overlay_away_profile = profile_index.get(normalize_lookup_text(away_team), {})
        if overlay_home_profile or overlay_away_profile:
            break
    return {
        "fixture_id": fixture_id or fixture_key,
        "fixture_key": fixture_key,
        "kickoff_time": kickoff_time,
        "league": league,
        "home_team": home_team,
        "away_team": away_team,
        **logo_assets,
        "coverage_status": "covered",
        "routing_status": "non_routed",
        "identity_source": identity_source,
        "source_availability": {
            "fixtures_master": bool(master_match),
            "routed_deploy": False,
            "routed_observe": False,
            "goal_shape_base": False,
            "prematch_odds": bool(current_overlay_availability.get("prematch_odds")),
            "injuries": bool(fixture_id and fixture_id in injuries_fixture_ids) or bool(current_overlay_availability.get("injuries")),
            "lineups": bool(fixture_id and fixture_id in lineups_fixture_ids) or bool(current_overlay_availability.get("lineups")),
            "team_stats": bool(home_profile or away_profile) or bool(current_overlay_availability.get("team_stats")),
            "player_stats": bool(fixture_id and fixture_id in player_stats_fixture_ids) or bool(current_overlay_availability.get("player_stats")),
            "match_events": bool(fixture_id and fixture_id in match_events_fixture_ids) or bool(current_overlay_availability.get("match_events")),
            "historical_overlay": bool(overlay_home_profile or overlay_away_profile),
            "current_overlay": bool(current_overlay_availability.get("current_overlay")),
        },
        "follow_candidates": {
            "team_follow_candidate": True,
            "fixture_follow_candidate": True,
            "league_follow_candidate": True,
        },
        "team_profile": {
            "home": home_profile or {},
            "away": away_profile or {},
        },
        "historical_overlay_profile": {
            "home": overlay_home_profile or {},
            "away": overlay_away_profile or {},
        },
        "current_overlay_summary": current_overlay.get("summary", {}) if isinstance(current_overlay.get("summary"), dict) else {},
        "updated_at": utc_now_iso(),
    }


def mark_allmarkets_availability(record: dict[str, Any], row: dict[str, str]) -> None:
    record["source_availability"]["goal_shape_base"] = True
    odds_fields = [
        "bookie_od",
        "od_home",
        "od_draw",
        "od_away",
        "od_yes",
        "od_no",
        "od_over",
        "od_under",
        "odds_ft_over25",
        "odds_ft_under25",
    ]
    if any(str(row.get(field, "") or "").strip() for field in odds_fields):
        record["source_availability"]["prematch_odds"] = True


def load_loss_detail_fixture_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    rows = load_rows(path)
    fixtures: dict[str, dict[str, str]] = {}
    for row in rows:
        if str(row.get("allmarkets_status", "") or "").strip().upper() != "NOT_EMITTED":
            continue
        fixture_key = str(row.get("fixture_key", "") or "").strip()
        if fixture_key:
            fixtures.setdefault(fixture_key, row)
    return fixtures


def build_report(
    source_paths: dict[str, Path],
    payload: dict[str, Any],
    counters: Counter[str],
) -> str:
    summary = payload["coverage_summary"]
    fallback_lines = "\n".join(f"- `{key}`: `{value}`" for key, value in sorted(counters.items()) if value)
    if not fallback_lines:
        fallback_lines = "- none"
    return "\n".join(
        [
            "# COVERED_FIXTURE_UNIVERSE_REPORT",
            "",
            f"Generated: `{payload['generated_at']}`",
            f"Source run id: `{payload['source_run_id']}`",
            f"Source window: `{payload['source_window']['date_from']}` to `{payload['source_window']['date_to']}`",
            "",
            "## Source Files",
            *(f"- `{label}`: `{path.relative_to(ROOT)}`" for label, path in source_paths.items() if path.exists()),
            "",
            "## Coverage Summary",
            f"- Total fixtures: `{summary['total_fixtures']}`",
            f"- Routed fixtures: `{summary['routed_count']}`",
            f"- Non-routed fixtures: `{summary['non_routed_count']}`",
            f"- Hidden fixtures: `{summary['hidden_count']}`",
            f"- Covered leagues: `{summary['covered_leagues_count']}`",
            "",
            "## Availability Counters",
            fallback_lines,
            "",
            "## Notes",
            "- Builder uses the current-window ALLMARKETS file as the primary live fixture intake base.",
            "- The pre-ALLMARKETS loss detail report fills upstream fixtures that never emitted into ALLMARKETS.",
            "- Historical normalized API-Football fixture master is joined opportunistically when identity matches exist.",
            "- This builder is an intake artifact for later CONTEXT and MONITOR classification.",
        ]
    )


def main() -> int:
    args = parse_args()
    ensure_dirs()
    counters: Counter[str] = Counter()

    source_path = resolve_source_path(str(args.src or "").strip() or None)
    allmarkets_path = resolve_allmarkets_source(source_path)
    if not allmarkets_path.exists():
        raise FileNotFoundError(f"Could not resolve ALLMARKETS source from {source_path}")

    related_paths = resolve_related_paths(allmarkets_path)
    source_window = extract_window_from_name(allmarkets_path.name)
    source_run_id = parse_source_run_id(allmarkets_path)

    logo_manifest_text = str(args.logo_manifest or "").strip()
    logo_manifest_path = (
        (ROOT / logo_manifest_text).resolve()
        if logo_manifest_text and not Path(logo_manifest_text).is_absolute()
        else Path(logo_manifest_text)
        if logo_manifest_text
        else None
    )
    logo_index = load_logo_manifest(logo_manifest_path, counters)
    fixtures_master_index = load_fixtures_master_index()
    injuries_fixture_ids = load_fixture_id_set(INJURIES_PATH)
    lineups_fixture_ids = load_fixture_id_set(LINEUPS_PATH)
    player_stats_fixture_ids = load_fixture_id_set(MATCH_PLAYER_STATS_PATH)
    match_events_fixture_ids = load_fixture_id_set(MATCH_EVENTS_PATH)
    team_profile_index = load_team_profile_index(MATCH_TEAM_STATS_PATH)
    historical_overlay_profiles = load_historical_overlay_profile_index()
    current_overlay_index = load_current_context_overlay_index(CURRENT_CONTEXT_OVERLAY_SUMMARY_PATH)

    allmarkets_rows = load_rows(related_paths["allmarkets"])
    counters["source_rows:allmarkets"] = len(allmarkets_rows)
    universe: dict[str, dict[str, Any]] = {}

    for row in allmarkets_rows:
        fixture_key = str(row.get("fixture_key", "") or "").strip()
        league = str(row.get("league", "") or "").strip()
        match_date = str(row.get("match_date", "") or "").strip()
        home_team = str(row.get("home_team_name", "") or "").strip()
        away_team = str(row.get("away_team_name", "") or "").strip()
        if not all([fixture_key, league, match_date, home_team, away_team]):
            counters["dropped:allmarkets_missing_identity"] += 1
            continue
        record = universe.get(fixture_key)
        if record is None:
            record = build_base_record(
                fixture_key,
                league,
                match_date,
                home_team,
                away_team,
                "allmarkets",
                logo_index,
                counters,
                fixtures_master_index,
                injuries_fixture_ids,
                lineups_fixture_ids,
                player_stats_fixture_ids,
                match_events_fixture_ids,
                team_profile_index,
                historical_overlay_profiles,
                current_overlay_index,
            )
            universe[fixture_key] = record
        mark_allmarkets_availability(record, row)

    raw_candidates = read_fixture_keys(related_paths["deploy_candidates_raw"])
    after_gates = read_fixture_keys(related_paths["deploy_candidates_after_gates"])
    counters["source_rows:deploy_candidates_raw"] = len(raw_candidates)
    counters["source_rows:deploy_candidates_after_gates"] = len(after_gates)
    for fixture_key, row in {**raw_candidates, **after_gates}.items():
        record = universe.get(fixture_key)
        if record is not None:
            record["source_availability"]["goal_shape_base"] = True

    deploy_keys = set()
    observe_keys = set()
    for label in ("deploy_elite", "deploy_standard"):
        path = related_paths[label]
        if not path.exists():
            continue
        rows = read_fixture_keys(path)
        counters[f"source_rows:{label}"] = len(rows)
        deploy_keys.update(rows.keys())
    if related_paths["deploy_observe"].exists():
        observe_rows = read_fixture_keys(related_paths["deploy_observe"])
        counters["source_rows:deploy_observe"] = len(observe_rows)
        observe_keys.update(observe_rows.keys())

    for fixture_key, record in universe.items():
        if fixture_key in deploy_keys:
            record["source_availability"]["routed_deploy"] = True
            record["routing_status"] = "routed"
        if fixture_key in observe_keys:
            record["source_availability"]["routed_observe"] = True
            if record["routing_status"] != "routed":
                record["routing_status"] = "routed"

    loss_detail_rows = load_loss_detail_fixture_rows(related_paths["loss_details"])
    counters["source_rows:loss_details_not_emitted"] = len(loss_detail_rows)
    for fixture_key, row in loss_detail_rows.items():
        if fixture_key in universe:
            counters["loss_details:fixture_already_in_universe"] += 1
            continue
        league = str(row.get("league", "") or "").strip()
        match_date = str(row.get("match_date", "") or "").strip()
        home_team = str(row.get("home_team_name", "") or "").strip()
        away_team = str(row.get("away_team_name", "") or "").strip()
        if not all([fixture_key, league, match_date, home_team, away_team]):
            counters["dropped:loss_details_missing_identity"] += 1
            continue
        record = build_base_record(
            fixture_key,
            league,
            match_date,
            home_team,
            away_team,
            "pre_allmarkets_loss_details",
            logo_index,
            counters,
            fixtures_master_index,
            injuries_fixture_ids,
            lineups_fixture_ids,
            player_stats_fixture_ids,
            match_events_fixture_ids,
            team_profile_index,
            historical_overlay_profiles,
            current_overlay_index,
        )
        universe[fixture_key] = record
        counters["loss_details:fixture_added_to_universe"] += 1

    fixtures = sorted(
        universe.values(),
        key=lambda record: (record["kickoff_time"], record["league"], record["home_team"], record["away_team"]),
    )

    for record in fixtures:
        counters[f"routing_status:{record['routing_status']}"] += 1
        for key, value in record["source_availability"].items():
            if value:
                counters[f"availability:{key}"] += 1

    payload = {
        "generated_at": utc_now_iso(),
        "source_run_id": source_run_id,
        "source_window": source_window,
        "coverage_summary": {
            "total_fixtures": len(fixtures),
            "routed_count": sum(1 for record in fixtures if record["routing_status"] == "routed"),
            "non_routed_count": sum(1 for record in fixtures if record["routing_status"] == "non_routed"),
            "hidden_count": 0,
            "covered_leagues_count": len({record["league"] for record in fixtures}),
        },
        "fixtures": fixtures,
    }

    write_json(OUTPUT_PATH, payload)
    REPORT_PATH.write_text(build_report(related_paths, payload, counters) + "\n", encoding="utf-8")

    print(f"Covered fixture universe written: {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"Covered fixture universe report written: {REPORT_PATH.relative_to(ROOT)}")
    print(f"Total fixtures: {payload['coverage_summary']['total_fixtures']}")
    print(f"Routed fixtures: {payload['coverage_summary']['routed_count']}")
    print(f"Non-routed fixtures: {payload['coverage_summary']['non_routed_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
