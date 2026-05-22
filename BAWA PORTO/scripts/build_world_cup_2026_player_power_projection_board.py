#!/usr/bin/env python3
"""Build a 2026-only World Cup player-power projection board.

Research only. This projects fixture and team player-power context from the
existing 2026 World Cup player-intelligence scaffold and FootyStats additions
sidecar. It is not a trained model and is not historical validation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_FIXTURE_MATRIX = Path(
    "data_sources/footystats_world_cup/research_feature_matrix_2026/world_cup_2026_research_feature_matrix.csv"
)
DEFAULT_TEAM_PLAYER = Path(
    "data_sources/footystats_world_cup/player_intelligence_2026/world_cup_2026_team_player_intelligence_scaffold.csv"
)
DEFAULT_ADDITIONS = Path("data_sources/footystats_world_cup/additions_context_2026/world_cup_additions_context_sidecar.csv")
DEFAULT_OUTDIR = Path("data_sources/footystats_world_cup/player_power_projection_2026")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            out[col] = out[col].astype("string").fillna("")
    lines = [
        "| " + " | ".join(out.columns) + " |",
        "| " + " | ".join(["---"] * len(out.columns)) + " |",
    ]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in out.columns) + " |")
    return "\n".join(lines)


def pct_rank(series: pd.Series, *, inverse: bool = False) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    ranks = values.rank(pct=True)
    if inverse:
        ranks = 1.0 - ranks
    return ranks


def mean_available(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series(np.nan, index=df.index)
    return df[present].mean(axis=1, skipna=True)


def build_team_board(team_player_path: Path, additions_path: Path) -> pd.DataFrame:
    team = pd.read_csv(team_player_path, low_memory=False)
    additions = pd.read_csv(additions_path, low_memory=False) if additions_path.exists() else pd.DataFrame()
    if not additions.empty:
        additions = additions.drop_duplicates("team_slug", keep="first")
        team = team.merge(additions, on="team_slug", how="left", suffixes=("", "_additions"))

    rank_inputs = {
        "rank_squad_quality": ("squad_quality_proxy", False),
        "rank_macro_prior": ("macro_prior_score", False),
        "rank_depth": ("player_intel_depth_score", False),
        "rank_additions_top11": ("additions_player_top11_avg_rating", False),
        "rank_additions_top5": ("additions_player_top5_avg_rating", False),
        "rank_additions_xg": ("additions_player_xg", False),
        "rank_additions_xa": ("additions_player_xa", False),
        "rank_additions_goals": ("additions_player_goals", False),
        "rank_team_xg_for": ("additions_team_xg_for_avg", False),
        "rank_team_goals_for": ("additions_team_goals_for_per_match", False),
        "rank_team_xg_against_inverse": ("additions_team_xg_against_avg", True),
        "rank_team_goals_against_inverse": ("additions_team_goals_against_per_match", True),
    }
    for out_col, (src_col, inverse) in rank_inputs.items():
        team[out_col] = pct_rank(team[src_col], inverse=inverse) if src_col in team.columns else np.nan

    team["player_power_score"] = mean_available(
        team,
        [
            "rank_squad_quality",
            "rank_macro_prior",
            "rank_depth",
            "rank_additions_top11",
            "rank_additions_top5",
            "rank_additions_xg",
            "rank_additions_xa",
            "rank_additions_goals",
        ],
    )
    team["player_attack_power_score"] = mean_available(
        team,
        [
            "rank_additions_xg",
            "rank_additions_xa",
            "rank_additions_goals",
            "rank_team_xg_for",
            "rank_team_goals_for",
            "rank_macro_prior",
        ],
    )
    team["player_defence_power_score"] = mean_available(
        team,
        ["rank_team_xg_against_inverse", "rank_team_goals_against_inverse", "rank_macro_prior"],
    )

    readiness_flags = []
    for col in [
        "api_2026_roster_joined_flag",
        "historical_player_prior_joined_flag",
        "external_player_prior_joined_flag",
        "additions_player_context_ready_flag",
        "additions_team_context_ready_flag",
        "additions_match_context_ready_flag",
    ]:
        if col in team.columns:
            readiness_flags.append(pd.to_numeric(team[col], errors="coerce").fillna(0))
    if readiness_flags:
        team["player_power_coverage_score"] = sum(readiness_flags) / len(readiness_flags)
    else:
        team["player_power_coverage_score"] = np.nan

    partial = pd.to_numeric(team.get("api_2026_roster_partial_flag"), errors="coerce").fillna(0)
    additions_ready = pd.to_numeric(team.get("additions_player_context_ready_flag"), errors="coerce").fillna(0)
    api_ready = pd.to_numeric(team.get("api_2026_roster_joined_flag"), errors="coerce").fillna(0)
    team["player_power_projection_status"] = np.select(
        [
            additions_ready.eq(1) & api_ready.eq(1) & partial.eq(0),
            additions_ready.eq(1) & (api_ready.eq(1) | pd.to_numeric(team.get("historical_player_prior_joined_flag"), errors="coerce").fillna(0).eq(1)),
            additions_ready.eq(1),
        ],
        [
            "CURRENT_PLAYER_CONTEXT_READY",
            "PROJECTED_MIXED_CURRENT_AND_HISTORICAL",
            "ADDITIONS_PLAYER_CONTEXT_ONLY",
        ],
        default="WEAK_OR_MACRO_ONLY_PLAYER_PRIOR",
    )
    team["player_power_uncertainty_flag"] = (
        partial.eq(1)
        | additions_ready.eq(0)
        | team["player_power_score"].isna()
        | team["player_power_coverage_score"].fillna(0).lt(0.4)
    ).astype(int)

    keep = [
        "team_slug",
        "team_name",
        "player_power_score",
        "player_attack_power_score",
        "player_defence_power_score",
        "player_power_coverage_score",
        "player_power_projection_status",
        "player_power_uncertainty_flag",
        "api_2026_roster_players",
        "api_2026_roster_partial_flag",
        "hist_last_wc_squad_year",
        "hist_last_wc_squad_players",
        "macro_prior_score",
        "macro_prior_percentile",
        "additions_player_rated_players",
        "additions_player_top11_avg_rating",
        "additions_player_top5_avg_rating",
        "additions_player_goals",
        "additions_player_assists",
        "additions_player_xg",
        "additions_player_xa",
        "additions_player_source_competitions",
        "additions_team_weighted_ppg",
        "additions_team_xg_for_avg",
        "additions_team_xg_against_avg",
        "additions_context_source_status",
    ]
    keep = [c for c in keep if c in team.columns]
    return team[keep].sort_values("player_power_score", ascending=False).reset_index(drop=True)


def pick_edge(delta: float) -> str:
    if pd.isna(delta):
        return "UNKNOWN"
    if delta >= 0.15:
        return "HOME_PLAYER_POWER_EDGE"
    if delta <= -0.15:
        return "AWAY_PLAYER_POWER_EDGE"
    return "PLAYER_POWER_BALANCED"


def build_fixture_board(fixture_path: Path, team_board: pd.DataFrame) -> pd.DataFrame:
    fixtures = pd.read_csv(fixture_path, low_memory=False)
    teams = team_board.set_index("team_slug")
    board = fixtures[
        [
            "season",
            "api_fixture_id",
            "api_date",
            "api_round",
            "api_home_team_name",
            "api_away_team_name",
            "home_team_slug",
            "away_team_slug",
            "macro_pick_ftr",
            "macro_ftr_confidence",
            "macro_ftr_risk_band",
            "macro_draw_stalemate_risk",
            "macro_prob_over25",
            "macro_prob_btts_yes",
            "world_cup_overlay_readiness",
            "world_cup_ftr_research_band",
            "world_cup_goal_market_research_band",
            "fixture_player_intel_coverage",
            "player_intel_lineup_uncertainty_proxy",
        ]
    ].copy()

    for side in ["home", "away"]:
        for col in [
            "player_power_score",
            "player_attack_power_score",
            "player_defence_power_score",
            "player_power_coverage_score",
            "player_power_projection_status",
            "player_power_uncertainty_flag",
            "additions_player_top11_avg_rating",
            "additions_player_top5_avg_rating",
            "additions_player_rated_players",
        ]:
            board[f"{side}_{col}"] = board[f"{side}_team_slug"].map(teams[col]) if col in teams.columns else np.nan

    board["player_power_delta_home_minus_away"] = board["home_player_power_score"] - board["away_player_power_score"]
    board["player_attack_delta_home_minus_away"] = board["home_player_attack_power_score"] - board["away_player_attack_power_score"]
    board["player_defence_delta_home_minus_away"] = board["home_player_defence_power_score"] - board["away_player_defence_power_score"]
    board["player_power_edge"] = board["player_power_delta_home_minus_away"].map(pick_edge)
    board["fixture_player_power_source_uncertainty_flag"] = (
        pd.to_numeric(board["home_player_power_uncertainty_flag"], errors="coerce").fillna(1).astype(int).eq(1)
        | pd.to_numeric(board["away_player_power_uncertainty_flag"], errors="coerce").fillna(1).astype(int).eq(1)
    ).astype(int)
    board["official_lineup_truth_pending_flag"] = pd.to_numeric(
        board["player_intel_lineup_uncertainty_proxy"], errors="coerce"
    ).fillna(1).astype(int)
    board["fixture_player_power_uncertainty_flag"] = (
        board["fixture_player_power_source_uncertainty_flag"].eq(1) | board["official_lineup_truth_pending_flag"].eq(1)
    ).astype(int)

    attack_q60 = team_board["player_attack_power_score"].quantile(0.60)
    attack_q75 = team_board["player_attack_power_score"].quantile(0.75)
    board["player_power_goal_market_hint"] = np.select(
        [
            board["fixture_player_power_source_uncertainty_flag"].eq(1),
            board["home_player_attack_power_score"].ge(attack_q75) & board["away_player_attack_power_score"].ge(attack_q75),
            board["home_player_attack_power_score"].ge(attack_q60) & board["away_player_attack_power_score"].ge(attack_q60),
            board["home_player_attack_power_score"].sub(board["away_player_defence_power_score"]).ge(0.20),
            board["away_player_attack_power_score"].sub(board["home_player_defence_power_score"]).ge(0.20),
        ],
        [
            "WEAK_SOURCE_PLAYER_POWER",
            "STRONG_BTTS_OU25_PLAYER_POWER_SUPPORT",
            "BTTS_OU25_PLAYER_POWER_WATCH",
            "HOME_TG15_PLAYER_POWER_SUPPORT",
            "AWAY_TG15_PLAYER_POWER_SUPPORT",
        ],
        default="NO_CLEAR_PLAYER_POWER_GOAL_EDGE",
    )
    board["player_power_ftr_hint"] = np.select(
        [
            board["fixture_player_power_source_uncertainty_flag"].eq(1),
            board["player_power_edge"].eq("HOME_PLAYER_POWER_EDGE"),
            board["player_power_edge"].eq("AWAY_PLAYER_POWER_EDGE"),
        ],
        [
            "WEAK_SOURCE_PLAYER_POWER",
            "HOME_SIDE_PLAYER_POWER_SUPPORT",
            "AWAY_SIDE_PLAYER_POWER_SUPPORT",
        ],
        default="NO_CLEAR_PLAYER_POWER_FTR_EDGE",
    )
    board["player_power_truth_layer_status"] = np.where(
        board["official_lineup_truth_pending_flag"].eq(1),
        "OFFICIAL_2026_SQUAD_INJURY_LINEUP_LAYER_PENDING",
        "OFFICIAL_TRUTH_LAYER_READY_OR_NOT_REQUIRED",
    )
    return board.sort_values(["api_date", "api_fixture_id"]).reset_index(drop=True)


def build_projection(fixture_path: Path, team_player_path: Path, additions_path: Path, outdir: Path) -> None:
    team_board = build_team_board(team_player_path, additions_path)
    fixture_board = build_fixture_board(fixture_path, team_board)

    coverage = (
        team_board.groupby(["player_power_projection_status", "player_power_uncertainty_flag"], dropna=False)
        .agg(teams=("team_slug", "count"), avg_power=("player_power_score", "mean"), avg_coverage=("player_power_coverage_score", "mean"))
        .reset_index()
        .sort_values(["player_power_uncertainty_flag", "teams"], ascending=[True, False])
    )
    fixture_coverage = (
        fixture_board.groupby(
            [
                "player_power_edge",
                "player_power_goal_market_hint",
                "fixture_player_power_source_uncertainty_flag",
                "official_lineup_truth_pending_flag",
            ],
            dropna=False,
        )
        .agg(fixtures=("api_fixture_id", "count"))
        .reset_index()
        .sort_values("fixtures", ascending=False)
    )

    outdir.mkdir(parents=True, exist_ok=True)
    team_path = outdir / "world_cup_2026_player_power_team_board.csv"
    fixture_path_out = outdir / "world_cup_2026_player_power_fixture_board.csv"
    coverage_path = outdir / "world_cup_2026_player_power_coverage.csv"
    fixture_coverage_path = outdir / "world_cup_2026_player_power_fixture_coverage.csv"
    team_board.to_csv(team_path, index=False)
    fixture_board.to_csv(fixture_path_out, index=False)
    coverage.to_csv(coverage_path, index=False)
    fixture_coverage.to_csv(fixture_coverage_path, index=False)

    md = [
        "# World Cup 2026 Player-Power Projection Board",
        "",
        "Research-only projection board built from current 2026 API roster scaffold, historical World Cup squad priors, macro priors, and FootyStats additions player/team context.",
        "",
        "## Team Coverage",
        "",
        markdown_table(coverage),
        "",
        "## Top Projected Player-Power Teams",
        "",
        markdown_table(team_board.head(15)),
        "",
        "## Fixture Hint Coverage",
        "",
        markdown_table(fixture_coverage),
        "",
        "## Outputs",
        "",
        f"- Team board: `{team_path}`",
        f"- Fixture board: `{fixture_path_out}`",
        f"- Team coverage: `{coverage_path}`",
        f"- Fixture coverage: `{fixture_coverage_path}`",
        "",
        "## Guardrail",
        "",
        "This is a 2026 projection surface, not a historical accuracy claim. The fixture hints deliberately separate player-power source quality from the later official squad/injury/lineup truth layer.",
        "",
    ]
    (outdir / "SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[ok] teams={len(team_board)} fixtures={len(fixture_board)}")
    print(f"[ok] wrote {outdir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-matrix", type=Path, default=DEFAULT_FIXTURE_MATRIX)
    parser.add_argument("--team-player", type=Path, default=DEFAULT_TEAM_PLAYER)
    parser.add_argument("--additions", type=Path, default=DEFAULT_ADDITIONS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    build_projection(args.fixture_matrix, args.team_player, args.additions, args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
