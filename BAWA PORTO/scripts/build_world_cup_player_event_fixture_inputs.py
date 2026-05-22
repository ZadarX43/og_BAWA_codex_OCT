#!/usr/bin/env python3
"""Build World Cup player-event fixture inputs in the existing schema shape.

Research-only adapter. Historical 2018/2022 rows use local API-Football match
player stats and are lagged by prior same-tournament player/team appearances.
The 2026 output is a pre-tournament projection board scaffold from roster plus
FootyStats additions context, not confirmed lineups.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NORMALIZED = ROOT / "data_sources" / "api_football" / "normalized"
FEATURE_DIR = ROOT / "data_sources" / "api_football" / "features" / "player_events"
WC_ROOT = ROOT / "data_sources" / "footystats_world_cup"
DEFAULT_OUTDIR = WC_ROOT / "player_event_fixture_inputs"
SCHEMA_COLUMNS = list(pd.read_csv(ROOT / "docs" / "PLAYER_EVENTS_INPUT_SCHEMA.csv", nrows=0).columns)


def num(values: Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_slug(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def clean_name(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def safe_div(num_value: float, den_value: float) -> float:
    return float(num_value) / float(den_value) if den_value else 0.0


def sum_recent(rows: list[dict[str, Any]], col: str, n: int) -> float:
    return sum(safe_float(row.get(col)) for row in rows[:n])


def mean_recent(rows: list[dict[str, Any]], col: str, n: int) -> float:
    sample = rows[:n]
    return sum_recent(sample, col, n) / len(sample) if sample else 0.0


def position_group(position: Any) -> str:
    pos = str(position or "").upper()[:1]
    return {"G": "Goalkeeper", "D": "Defender", "M": "Midfielder", "F": "Forward"}.get(pos, "Unknown")


def tactical_role(position: Any, tackles_per90: float = 0.0, shots_per90: float = 0.0) -> str:
    pos = str(position or "").upper()[:1]
    if pos == "G":
        return "Goalkeeper"
    if pos == "D":
        return "Wide defender / wing-back" if tackles_per90 >= 2.2 else "Centre-back enforcer"
    if pos == "M":
        return "Holding midfielder" if tackles_per90 >= 2.0 else "Central midfielder"
    if pos == "F":
        return "Central striker" if shots_per90 >= 2.0 else "Wide forward"
    return "General role"


def expected_minutes(avg_prior_minutes: float, started_flag: int, position: Any) -> float:
    if started_flag:
        floor = 90.0 if str(position).upper()[:1] == "G" else 62.0
        return round(min(90.0, max(floor, avg_prior_minutes or 0.0)), 1)
    return round(min(45.0, max(18.0, avg_prior_minutes or 0.0)), 1)


def base_record() -> dict[str, Any]:
    rec: dict[str, Any] = {}
    for col in SCHEMA_COLUMNS:
        rec[col] = 0.0
    for col in [
        "fixture_key",
        "match_date",
        "competition",
        "league",
        "home_team_name",
        "away_team_name",
        "venue",
        "referee_name",
        "weather_summary",
        "pitch_condition",
        "team_name",
        "player_name",
        "player_team_side",
        "position_group",
        "tactical_role",
        "likely_marking_assignment",
        "manual_overload_target_side",
        "manual_pitch_side",
        "manual_flank_role",
        "team_formation",
        "opponent_formation",
        "formation_matchup_label",
        "flank_zone",
        "wide_overload_target_side",
        "fixture_style_label",
        "fixture_attacking_style_label",
        "og_goal_environment_label",
        "analyst_notes",
        "opponent_striker_profile",
        "opponent_striker_context_note",
        "opponent_striker_pressure_tag",
        "opponent_striker_subtype_note",
        "contact_role_archetype",
        "contact_role_subtype",
        "pressure_channel_source",
        "game_state_contact_regime",
        "draw_state_contact_regime",
        "keeper_siege_type",
    ]:
        if col in rec:
            rec[col] = ""
    if "manual_pitch_side" in rec:
        rec["manual_pitch_side"] = "UNSET"
    if "manual_flank_role" in rec:
        rec["manual_flank_role"] = "UNSET"
    if "manual_overload_target_side" in rec:
        rec["manual_overload_target_side"] = "UNSET"
    return rec


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    for col in SCHEMA_COLUMNS:
        if col not in df.columns:
            df[col] = base_record()[col]
    extra_cols = [c for c in df.columns if c not in SCHEMA_COLUMNS]
    return df[SCHEMA_COLUMNS + extra_cols].copy()


def stage_bucket(round_value: Any) -> str:
    text = str(round_value or "").lower()
    if "group" in text:
        return "GROUP_STAGE"
    if "final" in text:
        return "FINAL_OR_SEMI" if "semi" in text or "final" in text else "KNOCKOUT"
    if "quarter" in text or "8th" in text or "round" in text:
        return "KNOCKOUT"
    return "UNKNOWN"


def build_historical(season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    fixtures_path = NORMALIZED / f"fixtures_master__World_Cup__{season}.csv"
    stats_path = NORMALIZED / f"match_player_stats__World_Cup__{season}.csv"
    team_path = NORMALIZED / f"match_team_stats__World_Cup__{season}.csv"
    lineups_path = NORMALIZED / f"lineups__World_Cup__{season}.csv"
    if not fixtures_path.exists() or not stats_path.exists() or not team_path.exists():
        raise FileNotFoundError(f"Missing World Cup API normalized files for {season}")

    fixtures = pd.read_csv(fixtures_path, low_memory=False)
    stats = pd.read_csv(stats_path, low_memory=False)
    team_stats = pd.read_csv(team_path, low_memory=False)
    lineups = pd.read_csv(lineups_path, low_memory=False) if lineups_path.exists() else pd.DataFrame()
    fixtures["kickoff_ts_utc"] = pd.to_datetime(fixtures["kickoff_ts_utc"], errors="coerce", utc=True)
    stats = stats.merge(
        fixtures[
            [
                "fixture_id",
                "fixture_key",
                "league",
                "season",
                "match_date",
                "kickoff_ts_utc",
                "home_team_id",
                "away_team_id",
                "home_team_name",
                "away_team_name",
                "venue_name",
                "referee_name",
            ]
        ],
        on="fixture_id",
        how="left",
    ).sort_values(["kickoff_ts_utc", "fixture_id", "team_id", "player_id"])
    team_stats = team_stats.merge(
        fixtures[["fixture_id", "kickoff_ts_utc", "home_team_id", "away_team_id", "home_team_name", "away_team_name"]],
        on="fixture_id",
        how="left",
    ).sort_values(["kickoff_ts_utc", "fixture_id", "team_id"])

    formation_map: dict[tuple[int, int], str] = {}
    if not lineups.empty and "formation" in lineups.columns:
        form = lineups.dropna(subset=["formation"]).drop_duplicates(["fixture_id", "team_id"], keep="last")
        formation_map = {(int(r.fixture_id), int(r.team_id)): str(r.formation) for r in form.itertuples()}

    team_history: dict[int, list[dict[str, Any]]] = defaultdict(list)
    team_roll: dict[tuple[int, int], dict[str, float]] = {}
    for _, row in team_stats.iterrows():
        tid = int(row["team_id"])
        prev = list(reversed(team_history[tid]))
        team_roll[(int(row["fixture_id"]), tid)] = {
            "team_avg_fouls": mean_recent(prev, "fouls_for", 5),
            "team_avg_yellows": mean_recent(prev, "yellow_cards", 5),
            "team_avg_possession": mean_recent(prev, "possession_pct", 5) or 50.0,
            "team_shots_l5": mean_recent(prev, "shots_total", 5),
            "team_sot_l5": mean_recent(prev, "shots_on_goal", 5),
            "team_tackles_l5": mean_recent(prev, "tackles", 5),
            "team_fouls_l5": mean_recent(prev, "fouls_for", 5),
            "team_cards_l5": mean_recent(prev, "yellow_cards", 5),
            "goals_for_l5": mean_recent(prev, "goals_for", 5),
            "goals_against_l5": mean_recent(prev, "goals_against", 5),
        }
        team_history[tid].append(row.to_dict())

    player_history: dict[int, list[dict[str, Any]]] = defaultdict(list)
    records: list[dict[str, Any]] = []
    actual_rows: list[dict[str, Any]] = []
    for _, row in stats.iterrows():
        player_id = int(row["player_id"])
        team_id = int(row["team_id"])
        fixture_id = int(row["fixture_id"])
        prev = list(reversed(player_history[player_id]))
        minutes_l5_sum = sum_recent(prev, "minutes", 5)
        minutes_l10_sum = sum_recent(prev, "minutes", 10)
        minutes_l5_avg = mean_recent(prev, "minutes", 5)
        shots_per90 = safe_div(sum_recent(prev, "shots_total", 5) * 90.0, minutes_l5_sum)
        sot_per90 = safe_div(sum_recent(prev, "shots_on_target", 5) * 90.0, minutes_l5_sum)
        tackles_per90 = safe_div(sum_recent(prev, "tackles", 5) * 90.0, minutes_l5_sum)
        fouls_per90 = safe_div(sum_recent(prev, "fouls_committed", 10) * 90.0, minutes_l10_sum)
        cards_per90 = safe_div(sum_recent(prev, "yellow_cards", 10) * 90.0, minutes_l10_sum)
        fouls_won_per90 = safe_div(sum_recent(prev, "fouls_drawn", 5) * 90.0, minutes_l5_sum)
        duels_total = sum_recent(prev, "duels_total", 5)
        duels_won = sum_recent(prev, "duels_won", 5)
        position = row.get("position", "")
        role = tactical_role(position, tackles_per90=tackles_per90, shots_per90=shots_per90)
        is_home = team_id == int(row["home_team_id"])
        opp_id = int(row["away_team_id"] if is_home else row["home_team_id"])
        team_name = row["home_team_name"] if is_home else row["away_team_name"]
        opp_roll = team_roll.get((fixture_id, opp_id), {})
        home_roll = team_roll.get((fixture_id, int(row["home_team_id"])), {})
        away_roll = team_roll.get((fixture_id, int(row["away_team_id"])), {})
        started = int(safe_float(row.get("started_flag")))
        rec = base_record()
        rec.update(
            {
                "fixture_key": row["fixture_key"],
                "match_date": row["match_date"],
                "competition": "World Cup",
                "league": "World Cup",
                "home_team_name": row["home_team_name"],
                "away_team_name": row["away_team_name"],
                "venue": row.get("venue_name", ""),
                "referee_name": row.get("referee_name", ""),
                "team_name": team_name,
                "player_name": row.get("player_name", ""),
                "player_team_side": "HOME" if is_home else "AWAY",
                "expected_start_flag": started,
                "expected_minutes": expected_minutes(minutes_l5_avg, started, position),
                "position_group": position_group(position),
                "tactical_role": role,
                "team_formation": formation_map.get((fixture_id, team_id), ""),
                "opponent_formation": formation_map.get((fixture_id, opp_id), ""),
                "formation_matchup_label": f"{formation_map.get((fixture_id, team_id), '')} vs {formation_map.get((fixture_id, opp_id), '')}".strip(),
                "flank_zone": "CENTRAL" if role in {"Holding midfielder", "Central midfielder", "Centre-back enforcer", "Central striker"} else "WIDE",
                "central_battle_flag": int(role in {"Holding midfielder", "Central midfielder", "Centre-back enforcer"}),
                "counterattack_defender_flag": int(position_group(position) == "Defender"),
                "fouls_per90": round(fouls_per90, 4),
                "yellow_cards_per90": round(cards_per90, 4),
                "booking_efficiency": round(safe_div(sum_recent(prev, "fouls_committed", 10), max(1.0, sum_recent(prev, "yellow_cards", 10))), 4),
                "tackles_per90": round(tackles_per90, 4),
                "interceptions_per90": round(safe_div(sum_recent(prev, "interceptions", 5) * 90.0, minutes_l5_sum), 4),
                "blocks_per90": round(safe_div(sum_recent(prev, "blocks", 5) * 90.0, minutes_l5_sum), 4),
                "duels_total_per90": round(safe_div(duels_total * 90.0, minutes_l5_sum), 4),
                "duels_won_per90": round(safe_div(duels_won * 90.0, minutes_l5_sum), 4),
                "ground_duel_loss_rate": round(1.0 - safe_div(duels_won, duels_total), 4) if duels_total else 0.0,
                "dribbles_faced_per90": round(safe_div(sum_recent(prev, "dribbled_past", 5) * 90.0, minutes_l5_sum), 4),
                "fouls_won_per90": round(fouls_won_per90, 4),
                "shots_per90": round(shots_per90, 4),
                "shots_on_target_per90": round(sot_per90, 4),
                "goals_per90": round(safe_div(sum_recent(prev, "goals", 5) * 90.0, minutes_l5_sum), 4),
                "assists_per90": round(safe_div(sum_recent(prev, "assists", 5) * 90.0, minutes_l5_sum), 4),
                "key_passes_per90": round(safe_div(sum_recent(prev, "passes_key", 5) * 90.0, minutes_l5_sum), 4),
                "player_form_rating_l5": round(mean_recent(prev, "rating", 5), 4),
                "player_quality_score_l5": round(mean_recent(prev, "rating", 5), 4),
                "pass_accuracy_pct_l5": round(safe_div(sum_recent(prev, "passes_accurate", 5) * 100.0, sum_recent(prev, "passes_total", 5)), 4),
                "minutes_last_3_matches": round(sum_recent(prev, "minutes", 3), 1),
                "days_rest": 4.0,
                "temperament_flag": 1 if cards_per90 >= 0.25 or fouls_per90 >= 2.0 else 0,
                "match_stakes_score": 3.0 if stage_bucket(row.get("league_round", "")) == "GROUP_STAGE" else 4.0,
                "team_avg_fouls": round(team_roll.get((fixture_id, team_id), {}).get("team_avg_fouls", 0.0), 4),
                "team_avg_yellows": round(team_roll.get((fixture_id, team_id), {}).get("team_avg_yellows", 0.0), 4),
                "opponent_possession_projection": round(opp_roll.get("team_avg_possession", 50.0), 4),
                "home_team_fouls_l5": round(home_roll.get("team_fouls_l5", 0.0), 4),
                "away_team_fouls_l5": round(away_roll.get("team_fouls_l5", 0.0), 4),
                "home_team_tackles_l5": round(home_roll.get("team_tackles_l5", 0.0), 4),
                "away_team_tackles_l5": round(away_roll.get("team_tackles_l5", 0.0), 4),
                "home_team_shots_l5": round(home_roll.get("team_shots_l5", 0.0), 4),
                "away_team_shots_l5": round(away_roll.get("team_shots_l5", 0.0), 4),
                "home_team_shots_on_goal_l5": round(home_roll.get("team_sot_l5", 0.0), 4),
                "away_team_shots_on_goal_l5": round(away_roll.get("team_sot_l5", 0.0), 4),
                "fixture_foul_density_score": round((home_roll.get("team_fouls_l5", 0.0) + away_roll.get("team_fouls_l5", 0.0)) / 28.0, 4),
                "fixture_tackle_density_score": round((home_roll.get("team_tackles_l5", 0.0) + away_roll.get("team_tackles_l5", 0.0)) / 35.0, 4),
                "fixture_attack_pressure_score": round((home_roll.get("team_sot_l5", 0.0) + away_roll.get("team_sot_l5", 0.0)) / 11.0, 4),
                "fixture_corner_pressure_score": 0.0,
                "fixture_style_label": "WORLD_CUP_TOURNAMENT_CONTEXT",
                "fixture_attacking_style_label": "SAME_TOURNAMENT_LAGGED_API",
                "og_xg_total": round(home_roll.get("goals_for_l5", 0.0) + away_roll.get("goals_for_l5", 0.0), 4),
                "keeper_siege_type": "SOT_PRESSURE" if position_group(position) == "Goalkeeper" else "",
                "analyst_notes": "World Cup historical player-event adapter; player/team rates lagged by prior same-tournament matches.",
                "world_cup_scope": "HISTORICAL_API_LAGGED_PLAYER_TEAM_EVENTS",
                "world_cup_lineup_scope": "ACTUAL_MATCH_PLAYER_MEMBERSHIP_RESEARCH_ONLY",
                "fixture_id": fixture_id,
                "player_id": player_id,
                "team_id": team_id,
                "season": season,
            }
        )
        records.append(rec)
        actual_rows.append(
            {
                "season": season,
                "fixture_id": fixture_id,
                "fixture_key": row["fixture_key"],
                "match_date": row["match_date"],
                "team_id": team_id,
                "team_name": team_name,
                "player_id": player_id,
                "player_name": row.get("player_name", ""),
                "position": position,
                "minutes": safe_float(row.get("minutes")),
                "started_flag": started,
                "actual_shots_total": safe_float(row.get("shots_total")),
                "actual_shots_on_target": safe_float(row.get("shots_on_target")),
                "actual_tackles": safe_float(row.get("tackles")),
                "actual_fouls_committed": safe_float(row.get("fouls_committed")),
                "actual_yellow_cards": safe_float(row.get("yellow_cards")),
                "actual_saves": safe_float(row.get("saves")),
            }
        )
        player_history[player_id].append(row.to_dict())

    features = finalize(pd.DataFrame(records))
    actuals = pd.DataFrame(actual_rows)
    return features, actuals


def load_additions_player_context() -> pd.DataFrame:
    path = WC_ROOT / "additions_context_2026" / "world_cup_additions_player_source_rows.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return df
    df["_player_key"] = df["player_name"].map(clean_name)
    df["_team_slug"] = df["team_slug"].map(safe_slug)
    grouped = (
        df.groupby(["_team_slug", "_player_key", "player_name", "position"], dropna=False)
        .agg(
            source_rows=("player_name", "size"),
            minutes=("minutes_played_overall", "sum"),
            appearances=("appearances_overall", "sum"),
            goals=("goals_overall", "sum"),
            assists=("assists_overall", "sum"),
            xg=("xg_total_overall", "sum"),
            xa=("xa_total_overall", "sum"),
            starts=("games_started", "sum"),
            yellows=("yellow_cards_overall", "sum"),
            avg_rating=("average_rating_overall", "mean"),
        )
        .reset_index()
    )
    return grouped


def build_2026_projection() -> pd.DataFrame:
    schedule_path = WC_ROOT / "api_bridge" / "world_cup_api_fixture_schedule.csv"
    roster_path = WC_ROOT / "player_intelligence_2026" / "world_cup_2026_api_player_roster_scheduled_teams_only.csv"
    fixture_power_path = WC_ROOT / "player_power_projection_2026" / "world_cup_2026_player_power_fixture_board.csv"
    if not schedule_path.exists() or not roster_path.exists():
        raise FileNotFoundError("Missing 2026 World Cup schedule or roster scaffold.")

    schedule = pd.read_csv(schedule_path, low_memory=False)
    schedule = schedule[schedule["season"].eq(2026)].copy()
    roster = pd.read_csv(roster_path, low_memory=False)
    roster["_team_slug"] = roster["team_slug"].map(safe_slug)
    roster["_player_key"] = roster["player_name"].map(clean_name)
    additions = load_additions_player_context()
    if not additions.empty:
        roster = roster.merge(additions, on=["_team_slug", "_player_key"], how="left", suffixes=("", "_additions"))
    if fixture_power_path.exists():
        fixture_power = pd.read_csv(fixture_power_path, low_memory=False)
    else:
        fixture_power = pd.DataFrame()

    records: list[dict[str, Any]] = []
    for _, fx in schedule.iterrows():
        fixture_id = int(fx["api_fixture_id"])
        fixture_key = f"2026_{fixture_id}_{safe_slug(fx['api_home_team_name'])}_{safe_slug(fx['api_away_team_name'])}"
        teams = [
            ("HOME", int(fx["api_home_team_id"]), fx["api_home_team_name"], fx["api_away_team_name"]),
            ("AWAY", int(fx["api_away_team_id"]), fx["api_away_team_name"], fx["api_home_team_name"]),
        ]
        power = fixture_power[fixture_power["api_fixture_id"].eq(fixture_id)].head(1) if not fixture_power.empty else pd.DataFrame()
        for side, team_id, team_name, opponent_name in teams:
            team_slug = safe_slug(team_name)
            team_players = roster[roster["team_id"].eq(team_id)].copy()
            if len(team_players) < 8:
                by_slug = roster[roster["_team_slug"].eq(team_slug)].copy()
                if len(by_slug) > len(team_players):
                    team_players = by_slug
            if len(team_players) < 8 and not additions.empty:
                additions_players = additions[additions["_team_slug"].eq(team_slug)].copy()
                if not additions_players.empty:
                    additions_players["_sort_rating"] = pd.to_numeric(additions_players["avg_rating"], errors="coerce").fillna(0.0)
                    additions_players["_sort_minutes"] = pd.to_numeric(additions_players["minutes"], errors="coerce").fillna(0.0)
                    additions_players = additions_players.sort_values(
                        ["_sort_rating", "_sort_minutes"],
                        ascending=[False, False],
                    ).head(26)
                    additions_players = additions_players.rename(
                        columns={
                            "_team_slug": "team_slug",
                            "_player_key": "player_slug",
                            "source_rows": "source_rows",
                        }
                    )
                    additions_players["_team_slug"] = team_slug
                    additions_players["_player_key"] = additions_players["player_name"].map(clean_name)
                    additions_players["team_id"] = team_id
                    additions_players["team_name"] = team_name
                    additions_players["api_player_id"] = pd.NA
                    additions_players["source"] = "FOOTYSTATS_ADDITIONS_PLAYER_CONTEXT_FALLBACK"
                    team_players = additions_players.copy()
            for _, player in team_players.iterrows():
                position = player.get("position", "")
                minutes = safe_float(player.get("minutes"), 0.0)
                apps = safe_float(player.get("appearances"), 0.0)
                starts = safe_float(player.get("starts"), 0.0)
                shots_proxy = safe_div((safe_float(player.get("goals")) + safe_float(player.get("xg")) + 0.45 * safe_float(player.get("assists"))) * 90.0, minutes)
                sot_proxy = shots_proxy * 0.42
                tackles_proxy = 1.7 if str(position).lower().startswith(("mid", "def")) else 0.7
                fouls_proxy = 1.5 if str(position).lower().startswith(("mid", "def")) else 0.9
                cards_proxy = safe_div(safe_float(player.get("yellows")) * 90.0, minutes)
                start_rate = safe_div(starts, apps)
                role = tactical_role(position, tackles_per90=tackles_proxy, shots_per90=shots_proxy)
                rec = base_record()
                rec.update(
                    {
                        "fixture_key": fixture_key,
                        "match_date": str(fx["api_date"])[:10],
                        "competition": "World Cup",
                        "league": "World Cup",
                        "home_team_name": fx["api_home_team_name"],
                        "away_team_name": fx["api_away_team_name"],
                        "venue": fx.get("api_venue_name", ""),
                        "team_name": team_name,
                        "player_name": player.get("player_name", ""),
                        "player_team_side": side,
                        "expected_start_flag": int(start_rate >= 0.45 or str(position).lower().startswith("goal")),
                        "expected_minutes": round(90.0 if str(position).lower().startswith("goal") else max(45.0, min(78.0, 90.0 * max(start_rate, 0.50))), 1),
                        "position_group": position_group(position[:1] if position else ""),
                        "tactical_role": role,
                        "flank_zone": "CENTRAL" if role in {"Goalkeeper", "Holding midfielder", "Central midfielder", "Centre-back enforcer", "Central striker"} else "WIDE",
                        "central_battle_flag": int(role in {"Holding midfielder", "Central midfielder", "Centre-back enforcer"}),
                        "counterattack_defender_flag": int(str(position).lower().startswith("def")),
                        "fouls_per90": round(fouls_proxy, 4),
                        "yellow_cards_per90": round(cards_proxy, 4),
                        "tackles_per90": round(tackles_proxy, 4),
                        "shots_per90": round(shots_proxy, 4),
                        "shots_on_target_per90": round(sot_proxy, 4),
                        "goals_per90": round(safe_div(safe_float(player.get("goals")) * 90.0, minutes), 4),
                        "assists_per90": round(safe_div(safe_float(player.get("assists")) * 90.0, minutes), 4),
                        "player_form_rating_l5": round(safe_float(player.get("avg_rating")), 4),
                        "player_quality_score_l5": round(safe_float(player.get("avg_rating")), 4),
                        "minutes_last_3_matches": round(min(270.0, minutes), 1),
                        "match_stakes_score": 3.5 if "Group" in str(fx.get("api_round", "")) else 4.5,
                        "fixture_style_label": "WORLD_CUP_2026_PRE_TOURNAMENT",
                        "fixture_attacking_style_label": "MACRO_PLAYER_POWER_ADDITIONS_CONTEXT",
                        "analyst_notes": "2026 pre-tournament player-event projection; not confirmed squad or lineup.",
                        "world_cup_scope": "2026_PRE_TOURNAMENT_ROSTER_ADDITIONS_CONTEXT",
                        "world_cup_lineup_scope": "OFFICIAL_LINEUP_PENDING",
                        "api_fixture_id": fixture_id,
                        "api_player_id": player.get("api_player_id", pd.NA),
                        "team_id": team_id,
                        "season": 2026,
                        "team_slug": player.get("team_slug", ""),
                        "opponent_team_name": opponent_name,
                        "source_rows": safe_float(player.get("source_rows"), 0.0),
                        "source_minutes": minutes,
                    }
                )
                if not power.empty:
                    prefix = "home" if side == "HOME" else "away"
                    opp_prefix = "away" if side == "HOME" else "home"
                    rec["team_power_rating"] = safe_float(power.iloc[0].get(f"{prefix}_player_power_score"))
                    rec["opponent_power_rating"] = safe_float(power.iloc[0].get(f"{opp_prefix}_player_power_score"))
                    rec["team_power_edge"] = rec["team_power_rating"] - rec["opponent_power_rating"]
                    rec["og_xg_total"] = safe_float(power.iloc[0].get("macro_prob_over25")) * 3.2
                    rec["og_btts_pre"] = safe_float(power.iloc[0].get("macro_prob_btts_yes"))
                    rec["og_over25_pre"] = safe_float(power.iloc[0].get("macro_prob_over25"))
                    rec["fixture_attack_pressure_score"] = max(
                        0.0,
                        min(1.0, safe_float(power.iloc[0].get(f"{prefix}_player_attack_power_score"))),
                    )
                    rec["power_gap_directional_pressure_score"] = max(0.0, min(1.0, -rec["team_power_edge"]))
                    rec["keeper_siege_type"] = "PLAYER_POWER_SOT_PRESSURE" if rec["position_group"] == "Goalkeeper" else ""
                records.append(rec)
    return finalize(pd.DataFrame(records))


def write_summary(outdir: Path, outputs: list[dict[str, Any]]) -> None:
    lines = [
        "# World Cup Player-Event Fixture Inputs",
        "",
        "- Research-only adapter into `docs/PLAYER_EVENTS_INPUT_SCHEMA.csv` shape.",
        "- 2018/2022 player and team rates are lagged by prior same-tournament matches.",
        "- Historical row membership uses API match player/lineup payloads, so it is valid for research backtests but not proof that the player was known pre-kickoff.",
        "- 2026 rows are roster/additions projections only. They are ranked intelligence candidates, not priced props.",
        "",
        "## Outputs",
    ]
    for item in outputs:
        lines.append(f"- {item['label']} | rows={item['rows']} | path={item['path']}")
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build World Cup player-event fixture inputs.")
    parser.add_argument("--seasons", default="2018,2022,2026")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, Any]] = []
    for season in [int(s.strip()) for s in args.seasons.split(",") if s.strip()]:
        if season in {2018, 2022}:
            features, actuals = build_historical(season)
            feature_path = FEATURE_DIR / f"player_events_fixture_input__World_Cup__{season}.csv"
            actuals_path = outdir / f"world_cup_player_event_actuals__{season}.csv"
            features.to_csv(feature_path, index=False)
            features.to_csv(outdir / f"player_events_fixture_input__World_Cup__{season}.csv", index=False)
            actuals.to_csv(actuals_path, index=False)
            outputs.append({"label": f"{season} fixture input", "rows": len(features), "path": feature_path})
            outputs.append({"label": f"{season} actuals", "rows": len(actuals), "path": actuals_path})
        elif season == 2026:
            features = build_2026_projection()
            path = outdir / "player_events_fixture_input__World_Cup__2026_pre_tournament.csv"
            features.to_csv(path, index=False)
            outputs.append({"label": "2026 pre-tournament fixture input", "rows": len(features), "path": path})
        else:
            raise ValueError(f"Unsupported season: {season}")

    write_summary(outdir, outputs)
    print(f"[ok] wrote {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
