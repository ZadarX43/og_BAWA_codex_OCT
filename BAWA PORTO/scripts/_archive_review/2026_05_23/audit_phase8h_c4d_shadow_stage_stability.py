#!/usr/bin/env python3
"""Audit window stability for the Phase 8H C4 shadow restore stages."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SELECTED = Path(
    "reports/2026-05-06/phase8h_research_deploy_shadow_simulator/full_estate_c4__PHASE8H_C4_SHADOW_SELECTED.csv"
)
DEFAULT_OUTDIR = Path("reports/2026-05-06/phase8h_c4d_shadow_stage_stability")


def num(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def rate_stats(values: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {
            "mean_window_hit_rate": np.nan,
            "median_window_hit_rate": np.nan,
            "p25_window_hit_rate": np.nan,
            "p10_window_hit_rate": np.nan,
            "min_window_hit_rate": np.nan,
        }
    return {
        "mean_window_hit_rate": float(clean.mean()),
        "median_window_hit_rate": float(clean.median()),
        "p25_window_hit_rate": float(clean.quantile(0.25)),
        "p10_window_hit_rate": float(clean.quantile(0.10)),
        "min_window_hit_rate": float(clean.min()),
    }


def per_window(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols + ["window_id"], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols + ["window_id"], keys, strict=False))
        hit = num(group["correct"])
        graded = int(hit.notna().sum())
        wins = int((hit == 1).sum())
        losses = int((hit == 0).sum())
        odds = num(group["bookie_od"]) if "bookie_od" in group.columns else pd.Series(np.nan, index=group.index)
        profit = np.where(hit == 1, odds - 1.0, np.where(hit == 0, -1.0, np.nan))
        base.update(
            {
                "rows": int(len(group)),
                "graded": graded,
                "wins": wins,
                "losses": losses,
                "hit_rate": wins / graded if graded else np.nan,
                "profit": float(np.nansum(profit)) if graded else np.nan,
                "roi": float(np.nansum(profit) / graded) if graded else np.nan,
            }
        )
        rows.append(base)
    return pd.DataFrame(rows)


def stability_summary(window_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in window_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys, strict=False))
        graded = int(group["graded"].sum())
        wins = int(group["wins"].sum())
        losses = int(group["losses"].sum())
        row.update(
            {
                "active_windows": int(group["window_id"].nunique()),
                "rows": int(group["rows"].sum()),
                "graded": graded,
                "wins": wins,
                "losses": losses,
                "hit_rate": wins / graded if graded else np.nan,
                "profit": float(group["profit"].sum()),
                "roi": float(group["profit"].sum() / graded) if graded else np.nan,
                "median_rows_per_window": float(group["rows"].median()),
                "p25_rows_per_window": float(group["rows"].quantile(0.25)),
                "windows_below_90": int((group["hit_rate"] < 0.90).sum()),
                "windows_below_85": int((group["hit_rate"] < 0.85).sum()),
            }
        )
        row.update(rate_stats(group["hit_rate"]))
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    text = df.copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            text[col] = text[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            text[col] = text[col].astype("string").fillna("")
    lines = [
        "| " + " | ".join(text.columns.astype(str)) + " |",
        "| " + " | ".join(["---"] * len(text.columns)) + " |",
    ]
    for _, row in text.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in text.columns) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", default=str(DEFAULT_SELECTED), help="Shadow-selected rows CSV.")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="Output directory.")
    args = parser.parse_args()

    selected_path = Path(args.selected)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(selected_path)
    if "window_id" not in df.columns:
        raise SystemExit("selected CSV must include window_id")

    stage_col = "phase8h_c4_shadow_stage"
    market_col = "_phase8h_c4_market_norm"
    ring_col = "phase8h_c4_shadow_recovery_ring"

    stage_window = per_window(df, [stage_col, market_col, ring_col])
    stage_summary = stability_summary(stage_window, [stage_col, market_col, ring_col])
    league_window = per_window(df, [stage_col, market_col, "league", ring_col])
    league_summary = stability_summary(league_window, [stage_col, market_col, "league", ring_col])

    stage_rank = {
        "OU25_RESTORE_NOW_SHADOW": 1,
        "OU25_RESTORE_WITH_CONFIRM_SHADOW": 2,
        "BTTS_RESTORE_NOW_SHADOW": 3,
        "BTTS_RESTORE_WITH_CONFIRM_SHADOW": 4,
    }
    for table in (stage_window, stage_summary, league_window, league_summary):
        table["stage_rank"] = table[stage_col].map(stage_rank)

    stage_window = stage_window.sort_values(["stage_rank", "window_id"])
    stage_summary = stage_summary.sort_values(["stage_rank"])
    league_window = league_window.sort_values(["stage_rank", "league", "window_id"])
    league_summary = league_summary.sort_values(["stage_rank", "league"])

    stage_window.to_csv(outdir / "phase8h_c4d_stage_window_scorecard.csv", index=False)
    stage_summary.to_csv(outdir / "phase8h_c4d_stage_stability_summary.csv", index=False)
    league_window.to_csv(outdir / "phase8h_c4d_stage_league_window_scorecard.csv", index=False)
    league_summary.to_csv(outdir / "phase8h_c4d_stage_league_stability_summary.csv", index=False)

    watch = league_summary[
        (league_summary["graded"] >= 40)
        & (
            (league_summary["hit_rate"] < 0.93)
            | (league_summary["p25_window_hit_rate"] < 0.85)
            | (league_summary["windows_below_90"] >= 8)
        )
    ].copy()
    watch = watch.sort_values(["stage_rank", "hit_rate", "p25_window_hit_rate"])
    watch.to_csv(outdir / "phase8h_c4d_watch_cells.csv", index=False)

    summary = [
        "# Phase 8H C4d Shadow Stage Stability",
        "",
        f"Source: `{selected_path}`",
        "",
        "## Stage Stability",
        markdown_table(stage_summary),
        "",
        "## Watch Cells",
        markdown_table(watch.head(40)),
        "",
        "## Read",
        "",
        "- Use this as the final stage-level QA before translating any C4 shadow stage into a research deploy script.",
        "- Watch cells are not automatic blockers, but they should be reviewed before live promotion.",
    ]
    (outdir / "phase8h_c4d_shadow_stage_stability_summary.md").write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    print(f"[ok] wrote {outdir}")


if __name__ == "__main__":
    main()
