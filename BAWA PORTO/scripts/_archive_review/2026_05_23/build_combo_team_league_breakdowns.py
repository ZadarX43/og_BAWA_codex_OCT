#!/usr/bin/env python3
"""Build team/league breakdowns for combo backtest outputs.

Research-only. This summarizes:
  - Team-goal combo lane: HOME_WIN_AND_HOME_GE2 / AWAY_WIN_AND_AWAY_GE2
  - FTR + BTTS combo lane: HOME/AWAY + BTTS YES/NO

The source rows are already from the 3-year Phase 8H full-estate replay outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TEAM_GOAL_ROWS = Path(
    "reports/2026-05-06/team_goal_combo_c4_full_estate_validation/"
    "team_goal_combo_c4_full_estate_rows.csv"
)
DEFAULT_FTR_BTTS_ROWS = Path(
    "reports/2026-05-06/ftr_btts_combo_discovery_audit/ftr_btts_combo_rows_scored.csv"
)
DEFAULT_FTR_BTTS_CANDIDATE_ROWS = Path(
    "reports/2026-05-06/ftr_btts_combo_window_stability/"
    "ftr_btts_combo_window_threshold_candidate_rows.csv"
)
DEFAULT_OUTDIR = Path("reports/2026-05-06/combo_team_league_breakdowns")


def num(values) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def profit_series(hit: pd.Series, odds: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(hit.eq(1), odds - 1.0, np.where(hit.eq(0), -1.0, np.nan)),
        index=hit.index,
    )


def scorecard(df: pd.DataFrame, group_cols: list[str], odds_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []
    grouped = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        hit = num(group["combo_correct"])
        odds = num(group.get(odds_col, pd.Series(np.nan, index=group.index)))
        profit = profit_series(hit, odds)
        graded = int(hit.notna().sum())
        wins = float(hit.eq(1).sum())
        row = dict(zip(group_cols, keys, strict=False))
        row.update(
            {
                "rows": int(len(group)),
                "graded": graded,
                "wins": wins,
                "losses": int(hit.eq(0).sum()),
                "hit_rate": wins / graded if graded else np.nan,
                "avg_odds": float(odds.mean()) if odds.notna().any() else np.nan,
                "profit": float(profit.sum(skipna=True)) if graded else np.nan,
                "roi": float(profit.sum(skipna=True) / graded) if graded else np.nan,
                "active_windows": int(group["window_id"].nunique()) if "window_id" in group.columns else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def add_team_goal_team(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    product = out["combo_product"].astype("string")
    out["combo_side"] = np.where(product.str.startswith("HOME_"), "HOME", "AWAY")
    out["combo_team_name"] = np.where(
        out["combo_side"].eq("HOME"),
        out.get("home_team_name", pd.Series("", index=out.index)),
        out.get("away_team_name", pd.Series("", index=out.index)),
    )
    out["opponent_team_name"] = np.where(
        out["combo_side"].eq("HOME"),
        out.get("away_team_name", pd.Series("", index=out.index)),
        out.get("home_team_name", pd.Series("", index=out.index)),
    )
    return out


def add_ftr_btts_team(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    side = out.get("ftr_side", pd.Series("", index=out.index)).astype("string").str.upper()
    out["combo_side"] = side
    out["combo_team_name"] = np.where(
        side.eq("HOME"),
        out.get("home_team_name", pd.Series("", index=out.index)),
        out.get("away_team_name", pd.Series("", index=out.index)),
    )
    out["opponent_team_name"] = np.where(
        side.eq("HOME"),
        out.get("away_team_name", pd.Series("", index=out.index)),
        out.get("home_team_name", pd.Series("", index=out.index)),
    )
    return out


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    text = df.copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            text[col] = text[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            text[col] = text[col].astype("string").fillna("")
    lines = [
        "| " + " | ".join(text.columns) + " |",
        "| " + " | ".join(["---"] * len(text.columns)) + " |",
    ]
    for _, row in text.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in text.columns) + " |")
    return "\n".join(lines)


def save_card(df: pd.DataFrame, path: Path, min_graded: int) -> pd.DataFrame:
    filtered = df[df["graded"].ge(min_graded)].copy() if not df.empty else df
    filtered = filtered.sort_values(["hit_rate", "graded", "roi"], ascending=[False, False, False])
    filtered.to_csv(path, index=False)
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-goal-rows", default=str(DEFAULT_TEAM_GOAL_ROWS))
    parser.add_argument("--ftr-btts-rows", default=str(DEFAULT_FTR_BTTS_ROWS))
    parser.add_argument("--ftr-btts-candidate-rows", default=str(DEFAULT_FTR_BTTS_CANDIDATE_ROWS))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--min-team-graded", type=int, default=5)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    team_goal = add_team_goal_team(pd.read_csv(args.team_goal_rows))
    ftr_btts = add_ftr_btts_team(pd.read_csv(args.ftr_btts_rows))
    ftr_btts_candidates = add_ftr_btts_team(pd.read_csv(args.ftr_btts_candidate_rows))

    tg_league_team = scorecard(
        team_goal,
        ["league", "combo_product", "combo_tier", "combo_team_name"],
        "bookie_od",
    )
    tg_league_team = save_card(
        tg_league_team,
        outdir / "team_goal_combo_by_league_team_tier.csv",
        args.min_team_graded,
    )

    tg_proof = team_goal[
        team_goal["league"].isin(["Spain La Liga", "Germany Bundesliga"])
        & team_goal["combo_product"].eq("HOME_WIN_AND_HOME_GE2")
    ].copy()
    tg_proof_team = scorecard(
        tg_proof,
        ["league", "combo_product", "combo_tier", "combo_team_name"],
        "bookie_od",
    )
    tg_proof_team = save_card(
        tg_proof_team,
        outdir / "team_goal_combo_spain_germany_proof_by_team.csv",
        args.min_team_graded,
    )

    fb_league_team = scorecard(
        ftr_btts,
        ["league", "combo_product", "combo_team_name"],
        "synthetic_combo_od",
    )
    fb_league_team = save_card(
        fb_league_team,
        outdir / "ftr_btts_combo_by_league_team.csv",
        args.min_team_graded,
    )

    fb_candidate_team = scorecard(
        ftr_btts_candidates,
        ["league", "combo_product", "candidate_feature", "candidate_threshold", "combo_team_name"],
        "synthetic_combo_od",
    )
    fb_candidate_team = save_card(
        fb_candidate_team,
        outdir / "ftr_btts_combo_threshold_candidates_by_team.csv",
        args.min_team_graded,
    )

    summary = [
        "# Combo Team / League Breakdowns",
        "",
        "Research-only team cuts from the 3-year Phase 8H combo row-level backtests.",
        "",
        "## Team-Goal Spain/Germany Proof Teams",
        md_table(
            tg_proof_team[
                [
                    "league",
                    "combo_product",
                    "combo_tier",
                    "combo_team_name",
                    "active_windows",
                    "graded",
                    "wins",
                    "hit_rate",
                    "avg_odds",
                    "roi",
                ]
            ].head(30)
        ),
        "",
        "## Best Team-Goal Combo Teams",
        md_table(
            tg_league_team[
                [
                    "league",
                    "combo_product",
                    "combo_tier",
                    "combo_team_name",
                    "active_windows",
                    "graded",
                    "wins",
                    "hit_rate",
                    "avg_odds",
                    "roi",
                ]
            ].head(30)
        ),
        "",
        "## Best Broad FTR + BTTS Combo Teams",
        md_table(
            fb_league_team[
                [
                    "league",
                    "combo_product",
                    "combo_team_name",
                    "active_windows",
                    "graded",
                    "wins",
                    "hit_rate",
                    "avg_odds",
                    "roi",
                ]
            ].head(30)
        ),
        "",
        "## Best Strict FTR + BTTS Candidate Teams",
        md_table(
            fb_candidate_team[
                [
                    "league",
                    "combo_product",
                    "candidate_feature",
                    "combo_team_name",
                    "active_windows",
                    "graded",
                    "wins",
                    "hit_rate",
                    "avg_odds",
                    "roi",
                ]
            ].head(30)
        ),
        "",
        "## Read",
        "",
        "- League/competition combo backtests already existed; this adds the missing team-level carrier view.",
        "- Team rows are descriptive proof cuts, not deploy gates by themselves.",
        "- Use team-level carriers as confirmers for future combo shadow policies.",
    ]
    (outdir / "combo_team_league_breakdowns_summary.md").write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    print(f"[ok] wrote {outdir}")


if __name__ == "__main__":
    main()
