#!/usr/bin/env python3
"""Build a pre-match World Cup research model matrix.

This adapts the existing training-input shape without writing into
Matches/__merged__. It combines:
- FootyStats match spine and market odds
- lagged in-tournament team state, widened to home/away features
- API-Football fixture/team identifiers where available

Post-match actual stat columns are intentionally excluded. Goal columns are
preserved only as labels, matching the existing trainers' target derivation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_FOUNDATION_DIR = Path("data_sources/footystats_world_cup/research_foundation")
DEFAULT_BRIDGE_DIR = Path("data_sources/footystats_world_cup/api_bridge")
DEFAULT_OUTDIR = Path("data_sources/footystats_world_cup/model_matrix")

MATCH_FEATURE_COLUMNS = [
    "fixture_key",
    "season",
    "competition",
    "league_tag",
    "timestamp",
    "date_GMT",
    "kickoff_dt",
    "status",
    "tournament_stage",
    "group_matchday",
    "home_team_name",
    "away_team_name",
    "home_team_goal_count",
    "away_team_goal_count",
    "total_goal_count",
    "odds_ft_home_team_win",
    "odds_ft_draw",
    "odds_ft_away_team_win",
    "odds_ft_over15",
    "odds_ft_over25",
    "odds_btts_yes",
    "odds_btts_no",
    "Pre-Match PPG (Home)",
    "Pre-Match PPG (Away)",
    "Home Team Pre-Match xG",
    "Away Team Pre-Match xG",
    "average_goals_per_match_pre_match",
    "btts_percentage_pre_match",
    "over_15_percentage_pre_match",
    "over_25_percentage_pre_match",
    "stadium_name",
    "attendance",
    "referee",
]

API_COLUMNS = [
    "fixture_key",
    "api_fixture_id",
    "api_home_team_id",
    "api_home_team_name",
    "api_away_team_id",
    "api_away_team_name",
    "api_round",
    "api_venue_id",
    "api_venue_name",
    "api_venue_city",
    "join_status",
]

LAG_COLUMNS = [
    "prior_matches_played",
    "prior_points",
    "prior_points_per_match",
    "prior_goal_diff",
    "prior_goals_for_per_match",
    "prior_goals_against_per_match",
    "prior_btts_rate",
    "prior_over25_rate",
]


def widen_team_state(team_state: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for side in ["home", "away"]:
        sub = team_state[team_state["side"].astype(str).eq(side)].copy()
        keep = ["fixture_key", *[col for col in LAG_COLUMNS if col in sub.columns]]
        sub = sub[keep]
        sub = sub.rename(columns={col: f"{side}_{col}" for col in LAG_COLUMNS if col in sub.columns})
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="fixture_key", how="outer")
    return out


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    view = df.copy()
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
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundation-dir", default=str(DEFAULT_FOUNDATION_DIR))
    parser.add_argument("--bridge-dir", default=str(DEFAULT_BRIDGE_DIR))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    args = parser.parse_args()

    foundation = Path(args.foundation_dir)
    bridge_dir = Path(args.bridge_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    matches = pd.read_csv(foundation / "footystats_world_cup_match_spine.csv", low_memory=False)
    team_state = pd.read_csv(foundation / "footystats_world_cup_lagged_team_state_long.csv", low_memory=False)
    bridge = pd.read_csv(bridge_dir / "world_cup_footystats_api_fixture_bridge.csv", low_memory=False)

    base = matches[[col for col in MATCH_FEATURE_COLUMNS if col in matches.columns]].copy()
    base["league"] = "World Cup"
    base["match_date"] = pd.to_datetime(base["kickoff_dt"], errors="coerce").dt.date.astype(str)
    base["is_world_cup"] = 1
    base["neutral_venue_flag"] = 1
    base["is_knockout_stage"] = base["tournament_stage"].astype(str).ne("GROUP_STAGE").astype(int)
    base["is_first_group_match"] = (
        base["tournament_stage"].astype(str).eq("GROUP_STAGE")
        & pd.to_numeric(base["group_matchday"], errors="coerce").eq(1)
    ).astype(int)

    wide_state = widen_team_state(team_state)
    out = base.merge(wide_state, on="fixture_key", how="left")

    bridge_keep = bridge[[col for col in API_COLUMNS if col in bridge.columns]].copy()
    out = out.merge(bridge_keep, on="fixture_key", how="left")

    out["api_fixture_joined_flag"] = out.get("api_fixture_id", pd.Series(index=out.index)).notna().astype(int)
    out["pre_match_feature_policy"] = "GOAL_LABELS_PLUS_PREMATCH_MARKET_AND_LAGGED_TEAM_STATE"
    out["player_aggregate_policy"] = "EXCLUDED_UNTIL_LAGGED_OR_EXTERNAL_PRE_TOURNAMENT_JOIN"

    out_path = outdir / "world_cup_research_model_matrix.csv"
    out.to_csv(out_path, index=False)

    summary = (
        out.groupby(["season", "tournament_stage"], dropna=False)
        .agg(
            rows=("fixture_key", "size"),
            api_joined=("api_fixture_joined_flag", "sum"),
            first_group_matches=("is_first_group_match", "sum"),
        )
        .reset_index()
        .sort_values(["season", "tournament_stage"])
    )
    season_summary = (
        out.groupby("season", dropna=False)
        .agg(
            rows=("fixture_key", "size"),
            api_join_rate=("api_fixture_joined_flag", "mean"),
            first_group_matches=("is_first_group_match", "sum"),
            knockout_rows=("is_knockout_stage", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(outdir / "world_cup_model_matrix_stage_summary.csv", index=False)
    season_summary.to_csv(outdir / "world_cup_model_matrix_season_summary.csv", index=False)

    lines = [
        "# World Cup Research Model Matrix",
        "",
        "Research-only training candidate. Not written to `Matches/__merged__`.",
        "",
        "## Season Summary",
        markdown_table(season_summary),
        "",
        "## Stage Summary",
        markdown_table(summary),
        "",
        "## Leakage Policy",
        "- Goal columns are retained as labels for existing trainer target derivation.",
        "- Match actual stat columns such as shots, xG actuals, corners, cards, and possession are excluded.",
        "- Team/player tournament aggregate performance files are excluded from this pre-match matrix until transformed into lagged or external pre-tournament features.",
        "- FootyStats pre-match market/context columns are retained but still need source timestamp assumptions documented before promotion beyond research.",
        "",
        "## Outputs",
        f"- `{out_path}`",
        f"- `{outdir / 'world_cup_model_matrix_stage_summary.csv'}`",
        f"- `{outdir / 'world_cup_model_matrix_season_summary.csv'}`",
    ]
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Rows: {len(out)}")
    print(f"Columns: {len(out.columns)}")
    print(f"Output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
