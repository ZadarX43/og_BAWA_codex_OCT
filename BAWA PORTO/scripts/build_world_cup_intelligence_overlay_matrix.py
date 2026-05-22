#!/usr/bin/env python3
"""Join World Cup macro priors and player-intelligence sidecars into a model-ready research matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_MACRO = Path("data_sources/footystats_world_cup/macro_prior_engine/world_cup_2026_macro_prior_fixture_matrix.csv")
DEFAULT_PLAYER = Path("data_sources/footystats_world_cup/player_intelligence_2026/world_cup_2026_fixture_player_intelligence_matrix.csv")
DEFAULT_OUTDIR = Path("data_sources/footystats_world_cup/intelligence_overlay_2026")


KEEP_PLAYER_COLS = [
    "api_fixture_id",
    "home_player_intel_status",
    "away_player_intel_status",
    "home_api_2026_roster_joined_flag",
    "away_api_2026_roster_joined_flag",
    "home_api_2026_roster_partial_flag",
    "away_api_2026_roster_partial_flag",
    "home_historical_player_prior_joined_flag",
    "away_historical_player_prior_joined_flag",
    "home_external_player_prior_joined_flag",
    "away_external_player_prior_joined_flag",
    "home_player_intel_depth_score",
    "away_player_intel_depth_score",
    "home_squad_quality_proxy",
    "away_squad_quality_proxy",
    "home_squad_continuity_risk",
    "away_squad_continuity_risk",
    "diff_squad_quality_proxy",
    "diff_player_intel_depth_score",
    "fixture_player_intel_coverage",
    "player_intel_lineup_uncertainty_proxy",
]


MODEL_READY_COLS = [
    "season",
    "api_fixture_id",
    "api_date",
    "api_round",
    "api_home_team_name",
    "api_away_team_name",
    "macro_prior_coverage_bucket",
    "fixture_player_intel_coverage",
    "macro_score_diff",
    "macro_host_bonus",
    "macro_home_xg_prior",
    "macro_away_xg_prior",
    "macro_total_goals_prior",
    "macro_prob_home",
    "macro_prob_draw",
    "macro_prob_away",
    "macro_prob_over25",
    "macro_prob_btts_yes",
    "macro_pick_ftr",
    "macro_ftr_confidence",
    "macro_ftr_risk_band",
    "macro_draw_stalemate_risk",
    "home_macro_prior_score",
    "away_macro_prior_score",
    "home_macro_prior_percentile",
    "away_macro_prior_percentile",
    "diff_squad_quality_proxy",
    "diff_player_intel_depth_score",
    "home_player_intel_depth_score",
    "away_player_intel_depth_score",
    "home_squad_quality_proxy",
    "away_squad_quality_proxy",
    "home_player_intel_status",
    "away_player_intel_status",
    "player_intel_lineup_uncertainty_proxy",
    "home_squad_continuity_risk",
    "away_squad_continuity_risk",
    "world_cup_overlay_readiness",
    "world_cup_ftr_research_band",
    "world_cup_goal_market_research_band",
]


def num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def overlay_readiness(row: pd.Series) -> str:
    if row.get("home_external_player_prior_joined_flag", 0) == 1 and row.get("away_external_player_prior_joined_flag", 0) == 1:
        return "MACRO_PLUS_EXTERNAL_PLAYER_PRIORS"
    if row.get("home_api_2026_roster_joined_flag", 0) == 1 and row.get("away_api_2026_roster_joined_flag", 0) == 1:
        return "MACRO_PLUS_API_ROSTERS"
    if row.get("fixture_player_intel_coverage") == "NO_PLAYER_PRIOR":
        return "MACRO_ONLY_PLAYER_PRIOR_MISSING"
    if row.get("player_intel_lineup_uncertainty_proxy", 1) == 1:
        return "MACRO_PLUS_HISTORICAL_PLAYER_PRIORS_NEEDS_SQUAD_REFRESH"
    return "MACRO_PLUS_PLAYER_PRIORS"


def ftr_research_band(row: pd.Series) -> str:
    if row.get("fixture_player_intel_coverage") == "NO_PLAYER_PRIOR":
        return "FTR_BLOCK_RESEARCH_PLAYER_MISSING"
    if row.get("macro_draw_stalemate_risk", 0) == 1:
        return "FTR_HIGH_RISK_DRAW_STALEMATE"
    if row.get("macro_ftr_risk_band") in {"HIGH_RISK", "CAUTION"}:
        return "FTR_CAUTION_LOW_MACRO_SEPARATION"
    if row.get("player_intel_lineup_uncertainty_proxy", 0) == 1:
        return "FTR_CAUTION_SQUAD_REFRESH_NEEDED"
    return "FTR_SAFE_MACRO_PLAYER_PRIOR"


def goal_market_band(row: pd.Series) -> str:
    total = float(row.get("macro_total_goals_prior") or 0)
    btts = float(row.get("macro_prob_btts_yes") or 0)
    if row.get("fixture_player_intel_coverage") == "NO_PLAYER_PRIOR":
        return "GOALS_CAUTION_PLAYER_PRIOR_MISSING"
    if total >= 2.85 and btts >= 0.52:
        return "GOALS_ATTACK_LEAN"
    if total <= 2.10 and btts <= 0.44:
        return "GOALS_SUPPRESSION_LEAN"
    return "GOALS_NEUTRAL_PRIOR"


def write_summary(outdir: Path, overlay: pd.DataFrame) -> None:
    readiness = overlay["world_cup_overlay_readiness"].value_counts(dropna=False).rename_axis("status").reset_index(name="fixtures")
    ftr = overlay["world_cup_ftr_research_band"].value_counts(dropna=False).rename_axis("band").reset_index(name="fixtures")
    goals = overlay["world_cup_goal_market_research_band"].value_counts(dropna=False).rename_axis("band").reset_index(name="fixtures")
    readiness.to_csv(outdir / "world_cup_2026_overlay_readiness_counts.csv", index=False)
    ftr.to_csv(outdir / "world_cup_2026_ftr_research_band_counts.csv", index=False)
    goals.to_csv(outdir / "world_cup_2026_goal_market_band_counts.csv", index=False)

    def md_table(df: pd.DataFrame, left: str) -> list[str]:
        lines = [f"| {left} | fixtures |", "|---|---:|"]
        for row in df.itertuples(index=False):
            lines.append(f"| {getattr(row, left)} | {int(row.fixtures)} |")
        return lines

    top_cols = [
        "api_home_team_name",
        "api_away_team_name",
        "macro_pick_ftr",
        "macro_ftr_confidence",
        "world_cup_ftr_research_band",
        "world_cup_overlay_readiness",
    ]
    top = overlay.sort_values("macro_ftr_confidence", ascending=False).head(12)
    top_lines = [
        f"- {r.api_home_team_name} vs {r.api_away_team_name}: {r.macro_pick_ftr} "
        f"({r.macro_ftr_confidence:.3f}) {r.world_cup_ftr_research_band}"
        for r in top[top_cols].itertuples()
    ]

    lines = [
        "# World Cup 2026 Intelligence Overlay Matrix",
        "",
        "Research-only joined sidecar for macro priors plus player/squad intelligence.",
        "",
        "## Outputs",
        "",
        f"- `{outdir / 'world_cup_2026_intelligence_overlay_matrix.csv'}`",
        f"- `{outdir / 'world_cup_2026_model_ready_sidecar.csv'}`",
        "",
        "## Overlay Readiness",
        "",
        *md_table(readiness, "status"),
        "",
        "## FTR Research Bands",
        "",
        *md_table(ftr, "band"),
        "",
        "## Goal Market Bands",
        "",
        *md_table(goals, "band"),
        "",
        "## Top Macro FTR Leans",
        "",
        *top_lines,
        "",
        "## Boundary",
        "",
        "- Not a deploy gate.",
        "- Designed for World Cup CatBoost/XGB/goal-mass sidecar experiments.",
        "- Current player-intelligence layer remains incomplete until external/domestic player quality sources are joined.",
    ]
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macro-fixtures", type=Path, default=DEFAULT_MACRO)
    parser.add_argument("--player-fixtures", type=Path, default=DEFAULT_PLAYER)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    macro = pd.read_csv(args.macro_fixtures, low_memory=False)
    players = pd.read_csv(args.player_fixtures, low_memory=False)
    keep = [c for c in KEEP_PLAYER_COLS if c in players.columns]
    overlay = macro.merge(players[keep], on="api_fixture_id", how="left", validate="one_to_one")
    overlay["world_cup_overlay_readiness"] = overlay.apply(overlay_readiness, axis=1)
    overlay["world_cup_ftr_research_band"] = overlay.apply(ftr_research_band, axis=1)
    overlay["world_cup_goal_market_research_band"] = overlay.apply(goal_market_band, axis=1)
    overlay.to_csv(args.outdir / "world_cup_2026_intelligence_overlay_matrix.csv", index=False)
    model_ready = overlay[[c for c in MODEL_READY_COLS if c in overlay.columns]].copy()
    model_ready.to_csv(args.outdir / "world_cup_2026_model_ready_sidecar.csv", index=False)
    write_summary(args.outdir, overlay)
    print(f"[ok] fixtures={len(overlay)}")
    print(f"[ok] wrote {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
