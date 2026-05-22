from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.api_football.team_name_map import base_normalize_team_name, normalize_team_name


def _norm_cap(series: pd.Series, cap: float) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").astype(float)
    return (out.clip(lower=0.0, upper=cap) / cap).fillna(0.0)


def _pick_col(df: pd.DataFrame, candidates: list[str], default: float = 0.0) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def build_og_goal_environment_overlay(
    league_tag: str,
    season: int,
    merged_csv: str,
    fixtures_csv: str,
    output_csv: str,
) -> pd.DataFrame:
    merged = pd.read_csv(merged_csv, low_memory=False)
    fixtures = pd.read_csv(fixtures_csv)

    merged["home_norm"] = merged["home_team_name"].map(base_normalize_team_name)
    merged["away_norm"] = merged["away_team_name"].map(base_normalize_team_name)
    fixtures["home_norm"] = fixtures["home_team_name"].map(lambda x: normalize_team_name(x, league_tag))
    fixtures["away_norm"] = fixtures["away_team_name"].map(lambda x: normalize_team_name(x, league_tag))

    merged["join_key"] = merged["match_date"].astype(str) + "|" + merged["home_norm"] + "|" + merged["away_norm"]
    fixtures["join_key"] = fixtures["match_date"].astype(str) + "|" + fixtures["home_norm"] + "|" + fixtures["away_norm"]

    og = merged.copy()
    og["og_pre_match_xg_home"] = _pick_col(og, ["pre_match_xg_home", "Home Team Pre-Match xG", "team_a_xg"])
    og["og_pre_match_xg_away"] = _pick_col(og, ["pre_match_xg_away", "Away Team Pre-Match xG", "team_b_xg"])
    og["og_btts_pre"] = _pick_col(og, ["btts_percentage_pre_match", "home_snap__btts_percentage", "away_snap__btts_percentage"]) / 100.0
    og["og_over25_pre"] = _pick_col(og, ["over_25_percentage_pre_match", "home_snap__over25_percentage", "away_snap__over25_percentage"]) / 100.0
    og["og_home_over25_snap"] = _pick_col(og, ["home_snap__over25_percentage", "over_25_percentage_pre_match"]) / 100.0
    og["og_away_over25_snap"] = _pick_col(og, ["away_snap__over25_percentage", "over_25_percentage_pre_match"]) / 100.0
    og["og_home_power_rating"] = _pick_col(og, ["home_power_rating"])
    og["og_away_power_rating"] = _pick_col(og, ["away_power_rating"])

    og["og_xg_total"] = og["og_pre_match_xg_home"] + og["og_pre_match_xg_away"]
    og["og_xg_weaker_side"] = og[["og_pre_match_xg_home", "og_pre_match_xg_away"]].min(axis=1)
    og["og_snap_over25_avg"] = (og["og_home_over25_snap"] + og["og_away_over25_snap"]) / 2.0
    og["og_power_gap_abs"] = (og["og_home_power_rating"] - og["og_away_power_rating"]).abs()
    og["og_balance_score"] = 1.0 - _norm_cap(og["og_power_gap_abs"], 35.0)

    og["og_goal_environment_score"] = (
        0.30 * _norm_cap(og["og_xg_total"], 3.6)
        + 0.20 * _norm_cap(og["og_xg_weaker_side"], 1.4)
        + 0.20 * _norm_cap(og["og_btts_pre"], 1.0)
        + 0.15 * _norm_cap(og["og_over25_pre"], 1.0)
        + 0.15 * _norm_cap(og["og_snap_over25_avg"], 1.0)
    )
    og["og_battle_on_score"] = (
        0.50 * og["og_goal_environment_score"]
        + 0.30 * og["og_balance_score"]
        + 0.20 * _norm_cap((og["og_btts_pre"] + og["og_over25_pre"]) / 2.0, 1.0)
    )
    og["og_goal_support_flag"] = og["og_goal_environment_score"].ge(0.60).astype(int)
    og["og_battle_on_flag"] = og["og_battle_on_score"].ge(0.62).astype(int)
    og["og_goal_environment_label"] = pd.cut(
        og["og_goal_environment_score"],
        bins=[-1, 0.42, 0.62, 2.0],
        labels=["LOW", "MEDIUM", "HIGH"],
    ).astype("string")

    export_cols = [
        "join_key",
        "fixture_key",
        "match_date",
        "home_team_name",
        "away_team_name",
        "og_pre_match_xg_home",
        "og_pre_match_xg_away",
        "og_xg_total",
        "og_xg_weaker_side",
        "og_btts_pre",
        "og_over25_pre",
        "og_snap_over25_avg",
        "og_home_power_rating",
        "og_away_power_rating",
        "og_power_gap_abs",
        "og_balance_score",
        "og_goal_environment_score",
        "og_battle_on_score",
        "og_goal_support_flag",
        "og_battle_on_flag",
        "og_goal_environment_label",
    ]
    export = og[export_cols].drop_duplicates(subset=["join_key"]).copy()

    out = fixtures.merge(
        export,
        on="join_key",
        how="left",
        suffixes=("", "_og"),
    )
    out["og_overlay_join_hit"] = out["og_goal_environment_score"].notna().astype(int)
    keep = [
        "fixture_id",
        "fixture_key",
        "league",
        "season",
        "match_date",
        "home_team_name",
        "away_team_name",
        "og_overlay_join_hit",
        "og_pre_match_xg_home",
        "og_pre_match_xg_away",
        "og_xg_total",
        "og_xg_weaker_side",
        "og_btts_pre",
        "og_over25_pre",
        "og_snap_over25_avg",
        "og_home_power_rating",
        "og_away_power_rating",
        "og_power_gap_abs",
        "og_balance_score",
        "og_goal_environment_score",
        "og_battle_on_score",
        "og_goal_support_flag",
        "og_battle_on_flag",
        "og_goal_environment_label",
    ]
    out = out[keep].copy()
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def _default_merged_path(league_tag: str) -> Path:
    return Path("Matches/__merged__") / f"{league_tag}__merged.csv"


def _default_output_path(league_tag: str, season: int) -> Path:
    return Path("data_sources/api_football/features/player_events") / f"og_goal_environment_overlay__{league_tag}__{season}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an Odds Genius goal-environment overlay for player-events research.")
    parser.add_argument("--league-tag", required=True, help="League tag like Italy_Serie_A")
    parser.add_argument("--season", type=int, required=True, help="Season integer, e.g. 2024")
    parser.add_argument("--merged-csv", default="", help="Override OG merged csv path")
    parser.add_argument("--fixtures-csv", default="", help="Override normalized fixtures csv path")
    parser.add_argument("--output-csv", default="", help="Override output csv path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged_csv = args.merged_csv or str(_default_merged_path(args.league_tag))
    fixtures_csv = args.fixtures_csv or str(Path("data_sources/api_football/normalized") / f"fixtures_master__{args.league_tag}__{args.season}.csv")
    output_csv = args.output_csv or str(_default_output_path(args.league_tag, args.season))
    df = build_og_goal_environment_overlay(args.league_tag, args.season, merged_csv, fixtures_csv, output_csv)
    print(f"WROTE: {output_csv}")
    print(f"rows: {len(df)} | join_hit_rate: {round(float(df['og_overlay_join_hit'].mean()), 4) if len(df) else 0.0}")


if __name__ == "__main__":
    main()
