#!/usr/bin/env python3
"""Audit window stability for Team-Goal Combo C4 proof lanes.

Research-only. This focuses on the Spain/Germany HOME_WIN_AND_HOME_GE2 proof
lanes and keeps team-goal combos separate from FTR rescue.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "reports/2026-05-06/team_goal_combo_c4_full_estate_validation/"
    "team_goal_combo_c4_full_estate_rows.csv"
)
DEFAULT_OUTDIR = Path(
    "reports/2026-05-06/team_goal_combo_c4_proof_window_stability"
)

PROOF_LANES = {
    ("Spain La Liga", "HOME_WIN_AND_HOME_GE2"),
    ("Germany Bundesliga", "HOME_WIN_AND_HOME_GE2"),
}


def num(values) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def profit_series(hit: pd.Series, odds: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(hit.eq(1), odds - 1.0, np.where(hit.eq(0), -1.0, np.nan)),
        index=hit.index,
    )


def scorecard(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []
    grouped = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        hit = num(group["combo_correct"])
        odds = num(group.get("bookie_od", pd.Series(np.nan, index=group.index)))
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
                "avg_bookie_od": float(odds.mean()) if odds.notna().any() else np.nan,
                "avg_combo_prob": float(num(group.get("combo_prob", np.nan)).mean()),
                "profit": float(profit.sum(skipna=True)) if graded else np.nan,
                "roi": float(profit.sum(skipna=True) / graded) if graded else np.nan,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def stability(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    window = scorecard(df, group_cols + ["window_id"])
    rows = []
    for keys, group in window.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        graded = int(group["graded"].sum())
        wins = float(group["wins"].sum())
        profit = float(group["profit"].sum(skipna=True))
        row = dict(zip(group_cols, keys, strict=False))
        row.update(
            {
                "active_windows": int(group["window_id"].nunique()),
                "rows": int(group["rows"].sum()),
                "graded": graded,
                "wins": wins,
                "losses": int(group["losses"].sum()),
                "hit_rate": wins / graded if graded else np.nan,
                "profit": profit,
                "roi": profit / graded if graded else np.nan,
                "median_rows_per_window": float(group["rows"].median()),
                "p25_rows_per_window": float(group["rows"].quantile(0.25)),
                "windows_below_90": int(group["hit_rate"].lt(0.90).sum()),
                "windows_below_85": int(group["hit_rate"].lt(0.85).sum()),
                "mean_window_hit_rate": float(group["hit_rate"].mean()),
                "median_window_hit_rate": float(group["hit_rate"].median()),
                "p25_window_hit_rate": float(group["hit_rate"].quantile(0.25)),
                "p10_window_hit_rate": float(group["hit_rate"].quantile(0.10)),
                "min_window_hit_rate": float(group["hit_rate"].min()),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    required = {"league", "combo_product", "combo_correct", "window_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    mask = pd.Series(False, index=df.index)
    for league, product in PROOF_LANES:
        mask |= df["league"].astype("string").eq(league) & df["combo_product"].astype("string").eq(product)
    proof = df.loc[mask].copy()
    proof.to_csv(outdir / "team_goal_combo_c4_proof_rows.csv", index=False)

    overall = scorecard(proof, ["league", "combo_product"])
    overall.to_csv(outdir / "team_goal_combo_c4_proof_overall.csv", index=False)

    by_tier = scorecard(proof, ["league", "combo_product", "combo_tier"])
    by_tier.to_csv(outdir / "team_goal_combo_c4_proof_by_tier.csv", index=False)

    window_rows = scorecard(proof, ["league", "combo_product", "combo_tier", "window_id"])
    window_rows.to_csv(outdir / "team_goal_combo_c4_proof_window_rows.csv", index=False)

    lane_stability = stability(proof, ["league", "combo_product"])
    lane_stability.to_csv(outdir / "team_goal_combo_c4_proof_lane_stability.csv", index=False)

    tier_stability = stability(proof, ["league", "combo_product", "combo_tier"])
    tier_stability.to_csv(outdir / "team_goal_combo_c4_proof_tier_stability.csv", index=False)

    summary = [
        "# Team-Goal Combo C4 Proof Window Stability",
        "",
        "Research-only audit for Spain/Germany `HOME_WIN_AND_HOME_GE2` proof lanes.",
        "These remain a separate combo lane, not FTR rescue.",
        "",
        "## Lane Stability",
        md_table(
            lane_stability[
                [
                    "league",
                    "combo_product",
                    "active_windows",
                    "graded",
                    "wins",
                    "hit_rate",
                    "roi",
                    "median_window_hit_rate",
                    "p25_window_hit_rate",
                    "windows_below_90",
                    "median_rows_per_window",
                ]
            ].sort_values(["hit_rate", "graded"], ascending=[False, False])
        ),
        "",
        "## Tier Stability",
        md_table(
            tier_stability[
                [
                    "league",
                    "combo_product",
                    "combo_tier",
                    "active_windows",
                    "graded",
                    "wins",
                    "hit_rate",
                    "roi",
                    "median_window_hit_rate",
                    "p25_window_hit_rate",
                    "windows_below_90",
                    "median_rows_per_window",
                ]
            ].sort_values(["league", "combo_tier"])
        ),
        "",
        "## Read",
        "",
        "- Use this before any combo live sidecar promotion conversation.",
        "- Spain/Germany proof lanes can continue as shadow-only instrumentation.",
        "- Any broader league/team-goal expansion needs its own league/team proof.",
    ]
    (outdir / "team_goal_combo_c4_proof_window_stability_summary.md").write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    print(f"[ok] wrote {outdir}")


if __name__ == "__main__":
    main()
