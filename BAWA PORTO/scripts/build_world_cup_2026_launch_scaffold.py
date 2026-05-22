#!/usr/bin/env python3
"""Build a 2026 World Cup pre-match launch scaffold.

Research-only. This file is for fixture intelligence and coverage planning, not
model training labels. It attaches pre-2026 World Cup team priors to the current
API-Football 2026 group-stage schedule and highlights where player/squad/injury
sources still need to land.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_MATCH_SPINE = Path("data_sources/footystats_world_cup/research_foundation/footystats_world_cup_match_spine.csv")
DEFAULT_PLAYER_STATIC = Path("data_sources/footystats_world_cup/research_foundation/footystats_world_cup_player_static_profiles.csv")
DEFAULT_API_SCHEDULE = Path("data_sources/footystats_world_cup/api_bridge/world_cup_api_fixture_schedule.csv")
DEFAULT_OUTDIR = Path("data_sources/footystats_world_cup/launch_2026")

HOST_TEAMS = {"canada", "mexico", "usa"}
RECENCY_WEIGHTS = {2006: 0.6, 2010: 0.8, 2014: 1.0, 2018: 1.4, 2022: 2.0}

ALIASES = {
    "usmnt": "usa",
    "united states": "usa",
    "u s a": "usa",
    "cote d ivoire": "ivory coast",
    "côte d ivoire": "ivory coast",
    "côte d’ivoire": "ivory coast",
    "bosnia herzegovina": "bosnia and herzegovina",
    "cape verde islands": "cape verde",
    "cape verde island": "cape verde",
    "congo dr": "dr congo",
    "drc": "dr congo",
    "turkiye": "turkey",
    "czech republic": "czech republic",
    "korea republic": "south korea",
}


def norm_team(value: object) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"\s+national\s+team$", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return ALIASES.get(text, text)


def side_rows(matches: pd.DataFrame) -> pd.DataFrame:
    home = matches[
        [
            "season",
            "fixture_key",
            "kickoff_dt",
            "tournament_stage",
            "home_team_name",
            "away_team_name",
            "home_team_goal_count",
            "away_team_goal_count",
        ]
    ].copy()
    home = home.rename(
        columns={
            "home_team_name": "team_name",
            "away_team_name": "opponent_name",
            "home_team_goal_count": "goals_for",
            "away_team_goal_count": "goals_against",
        }
    )
    home["side"] = "home"

    away = matches[
        [
            "season",
            "fixture_key",
            "kickoff_dt",
            "tournament_stage",
            "away_team_name",
            "home_team_name",
            "away_team_goal_count",
            "home_team_goal_count",
        ]
    ].copy()
    away = away.rename(
        columns={
            "away_team_name": "team_name",
            "home_team_name": "opponent_name",
            "away_team_goal_count": "goals_for",
            "home_team_goal_count": "goals_against",
        }
    )
    away["side"] = "away"

    out = pd.concat([home, away], ignore_index=True, sort=False)
    out["team_slug"] = out["team_name"].map(norm_team)
    out["points"] = np.where(
        out["goals_for"] > out["goals_against"],
        3,
        np.where(out["goals_for"].eq(out["goals_against"]), 1, 0),
    )
    out["goal_diff"] = out["goals_for"] - out["goals_against"]
    out["btts"] = ((out["goals_for"] > 0) & (out["goals_against"] > 0)).astype(int)
    out["over25"] = ((out["goals_for"] + out["goals_against"]) > 2).astype(int)
    out["knockout_match"] = out["tournament_stage"].astype(str).ne("GROUP_STAGE").astype(int)
    out["weight"] = out["season"].map(RECENCY_WEIGHTS).fillna(1.0)
    return out


def weighted_mean(values: pd.Series, weights: pd.Series) -> float | None:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = v.notna() & w.notna() & (w > 0)
    if not mask.any():
        return None
    return float((v[mask] * w[mask]).sum() / w[mask].sum())


def build_team_priors(matches: pd.DataFrame) -> pd.DataFrame:
    sides = side_rows(matches)
    rows: list[dict[str, Any]] = []
    for team, group in sides.groupby("team_slug", dropna=False):
        seasons = sorted(group["season"].dropna().astype(int).unique().tolist())
        rows.append(
            {
                "team_slug": team,
                "team_name_latest": group.sort_values(["season", "kickoff_dt"])["team_name"].iloc[-1],
                "wc_matches_2006_2022": int(len(group)),
                "wc_tournaments_2006_2022": int(len(seasons)),
                "wc_last_seen_year": int(max(seasons)) if seasons else None,
                "wc_points_per_match": float(group["points"].mean()),
                "wc_goal_diff_per_match": float(group["goal_diff"].mean()),
                "wc_goals_for_per_match": float(group["goals_for"].mean()),
                "wc_goals_against_per_match": float(group["goals_against"].mean()),
                "wc_btts_rate": float(group["btts"].mean()),
                "wc_over25_rate": float(group["over25"].mean()),
                "wc_knockout_match_rate": float(group["knockout_match"].mean()),
                "wc_weighted_points_per_match": weighted_mean(group["points"], group["weight"]),
                "wc_weighted_goal_diff_per_match": weighted_mean(group["goal_diff"], group["weight"]),
                "wc_weighted_goals_for_per_match": weighted_mean(group["goals_for"], group["weight"]),
                "wc_weighted_goals_against_per_match": weighted_mean(group["goals_against"], group["weight"]),
            }
        )
    return pd.DataFrame(rows)


def build_player_static_priors(players: pd.DataFrame) -> pd.DataFrame:
    if players.empty:
        return pd.DataFrame()
    players = players.copy()
    players["team_slug"] = players["squad_team_name"].map(norm_team)
    rows: list[dict[str, Any]] = []
    for team, group in players.groupby("team_slug", dropna=False):
        latest_year = int(pd.to_numeric(group["season"], errors="coerce").max())
        latest = group[pd.to_numeric(group["season"], errors="coerce").eq(latest_year)].copy()
        rows.append(
            {
                "team_slug": team,
                "last_wc_squad_year": latest_year,
                "last_wc_squad_players": int(len(latest)),
                "last_wc_squad_avg_age": float(pd.to_numeric(latest["age"], errors="coerce").mean()),
                "last_wc_goalkeepers": int(latest["position"].astype(str).str.lower().eq("goalkeeper").sum()),
                "last_wc_defenders": int(latest["position"].astype(str).str.lower().eq("defender").sum()),
                "last_wc_midfielders": int(latest["position"].astype(str).str.lower().eq("midfielder").sum()),
                "last_wc_forwards": int(latest["position"].astype(str).str.lower().eq("forward").sum()),
            }
        )
    return pd.DataFrame(rows)


def attach_side(prefix: str, schedule: pd.DataFrame, priors: pd.DataFrame, player_priors: pd.DataFrame) -> pd.DataFrame:
    side_col = f"api_{prefix}_team_name"
    out = schedule.copy()
    out[f"{prefix}_team_slug"] = out[side_col].map(norm_team)
    p = priors.add_prefix(f"{prefix}_")
    out = out.merge(p, left_on=f"{prefix}_team_slug", right_on=f"{prefix}_team_slug", how="left")
    if not player_priors.empty:
        pp = player_priors.add_prefix(f"{prefix}_")
        out = out.merge(pp, left_on=f"{prefix}_team_slug", right_on=f"{prefix}_team_slug", how="left")
    return out


def coverage_bucket(row: pd.Series) -> str:
    home_last = row.get("home_wc_last_seen_year")
    away_last = row.get("away_wc_last_seen_year")
    home_recent = pd.notna(home_last) and int(home_last) >= 2022
    away_recent = pd.notna(away_last) and int(away_last) >= 2022
    home_any = pd.notna(home_last)
    away_any = pd.notna(away_last)
    if home_recent and away_recent:
        return "BOTH_2022_PRIORS"
    if home_any and away_any:
        return "BOTH_HISTORICAL_PRIORS"
    if home_any or away_any:
        return "ONE_SIDE_WORLD_CUP_PRIOR"
    return "NO_WORLD_CUP_PRIOR"


def build_schedule(matches: pd.DataFrame, players: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    priors = build_team_priors(matches)
    player_priors = build_player_static_priors(players)
    current = schedule[schedule["season"].eq(2026)].copy()
    current["match_date"] = pd.to_datetime(current["api_date"], errors="coerce").dt.date.astype(str)
    current["api_kickoff_dt"] = pd.to_datetime(current["api_date"], errors="coerce", utc=True)
    current["world_cup_group_matchday"] = current["api_round"].astype(str).str.extract(r"(\d+)$")[0].astype("Int64")

    current = attach_side("home", current, priors, player_priors)
    current = attach_side("away", current, priors, player_priors)

    current["home_is_host"] = current["home_team_slug"].isin(HOST_TEAMS).astype(int)
    current["away_is_host"] = current["away_team_slug"].isin(HOST_TEAMS).astype(int)
    current["coverage_bucket"] = current.apply(coverage_bucket, axis=1)
    current["api_schedule_status"] = "GROUP_STAGE_ONLY_CURRENTLY"
    current["player_intel_status"] = "PENDING_2026_SQUAD_API_OR_KAGGLE"
    current["injury_intel_status"] = "PENDING_PREMATCH_INJURY_SNAPSHOT"
    current["odds_intel_status"] = "PENDING_7_DAY_API_ODDS_WINDOW_OR_MARKET_SOURCE"

    for metric in [
        "wc_weighted_points_per_match",
        "wc_weighted_goal_diff_per_match",
        "wc_weighted_goals_for_per_match",
        "wc_weighted_goals_against_per_match",
        "wc_matches_2006_2022",
        "wc_tournaments_2006_2022",
        "wc_knockout_match_rate",
        "last_wc_squad_avg_age",
    ]:
        h = f"home_{metric}"
        a = f"away_{metric}"
        if h in current.columns and a in current.columns:
            current[f"diff_{metric}"] = pd.to_numeric(current[h], errors="coerce") - pd.to_numeric(current[a], errors="coerce")
    return current


def markdown_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            view[col] = view[col].astype("string").fillna("")
    lines = [
        "| " + " | ".join(view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "/") for col in view.columns) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(df)} rows._")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-spine", default=str(DEFAULT_MATCH_SPINE))
    parser.add_argument("--player-static", default=str(DEFAULT_PLAYER_STATIC))
    parser.add_argument("--api-schedule", default=str(DEFAULT_API_SCHEDULE))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    matches = pd.read_csv(args.match_spine, low_memory=False)
    players = pd.read_csv(args.player_static, low_memory=False)
    schedule = pd.read_csv(args.api_schedule, low_memory=False)

    team_priors = build_team_priors(matches)
    player_priors = build_player_static_priors(players)
    launch = build_schedule(matches, players, schedule)

    team_priors.to_csv(outdir / "world_cup_team_priors_2006_2022.csv", index=False)
    player_priors.to_csv(outdir / "world_cup_last_squad_static_priors.csv", index=False)
    launch.to_csv(outdir / "world_cup_2026_launch_scaffold.csv", index=False)

    coverage = (
        launch.groupby(["api_round", "coverage_bucket"], dropna=False)
        .agg(fixtures=("api_fixture_id", "size"))
        .reset_index()
        .sort_values(["api_round", "coverage_bucket"])
    )
    coverage.to_csv(outdir / "world_cup_2026_launch_coverage_summary.csv", index=False)

    first_round = launch[launch["api_round"].astype(str).eq("Group Stage - 1")].copy()
    first_cols = [
        "match_date",
        "api_home_team_name",
        "api_away_team_name",
        "coverage_bucket",
        "home_wc_last_seen_year",
        "away_wc_last_seen_year",
        "diff_wc_weighted_points_per_match",
        "diff_wc_weighted_goal_diff_per_match",
        "home_is_host",
        "away_is_host",
        "api_venue_name",
    ]
    first_round[[c for c in first_cols if c in first_round.columns]].to_csv(
        outdir / "world_cup_2026_group_stage_1_watchlist.csv",
        index=False,
    )

    summary = [
        "# World Cup 2026 Launch Scaffold",
        "",
        "Research-only pre-match fixture intelligence scaffold for the currently available API-Football 2026 group-stage schedule.",
        "",
        "## Coverage Summary",
        markdown_table(coverage),
        "",
        "## Group Stage 1 Watchlist",
        markdown_table(first_round[[c for c in first_cols if c in first_round.columns]]),
        "",
        "## Source Status",
        "- API-Football currently provides `72` 2026 fixtures, all group-stage rows.",
        "- Historical World Cup team priors use only 2006-2022 FootyStats match outcomes and are pre-2026 safe.",
        "- Last-squad static priors are historical roster profile summaries only; 2026 player intelligence is marked pending until squad/API/Kaggle sources land.",
        "- Odds are intentionally pending because API-Football odds are only available inside its retrieval window and must be timestamped.",
        "",
        "## Outputs",
        f"- `{outdir / 'world_cup_2026_launch_scaffold.csv'}`",
        f"- `{outdir / 'world_cup_2026_group_stage_1_watchlist.csv'}`",
        f"- `{outdir / 'world_cup_team_priors_2006_2022.csv'}`",
        f"- `{outdir / 'world_cup_last_squad_static_priors.csv'}`",
        f"- `{outdir / 'world_cup_2026_launch_coverage_summary.csv'}`",
    ]
    (outdir / "SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Launch fixtures: {len(launch)}")
    print(f"Team priors: {len(team_priors)}")
    print(f"Output: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
