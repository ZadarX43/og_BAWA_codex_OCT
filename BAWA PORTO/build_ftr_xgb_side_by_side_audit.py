#!/usr/bin/env python3
"""Side-by-side FTR audit: CatBoost vs XGBoost vs consensus.

Outputs:
  - FTR_XGB_SIDE_BY_SIDE_AUDIT.csv
  - FTR_XGB_SIDE_BY_SIDE_AUDIT__BY_LEAGUE.csv
  - FTR_XGB_SIDE_BY_SIDE_AUDIT__BY_SIDE.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


ACTUAL_COLS = [
    "actual_ftr",
    "actual_result",
    "actual_ft_result",
    "full_time_result",
    "ft_result",
    "result",
]


def _norm_pick(v) -> str | None:
    s = str(v).strip().upper()
    if not s or s == "NAN":
        return None
    if s in {"H", "HOME", "HOME_WIN", "HOMETEAM"}:
        return "HOME"
    if s in {"A", "AWAY", "AWAY_WIN", "AWAYTEAM"}:
        return "AWAY"
    if s in {"D", "DRAW", "X"}:
        return "DRAW"
    # try prefix match
    if s.startswith("H"):
        return "HOME"
    if s.startswith("A"):
        return "AWAY"
    if s.startswith("D"):
        return "DRAW"
    return None


def _resolve_actual_col(df: pd.DataFrame) -> str | None:
    for c in ACTUAL_COLS:
        if c in df.columns:
            return c
    return None


def _pick_odds(df: pd.DataFrame, pick_col: str) -> pd.Series:
    pick = df[pick_col]
    od_home = pd.to_numeric(df.get("od_home"), errors="coerce")
    od_draw = pd.to_numeric(df.get("od_draw"), errors="coerce")
    od_away = pd.to_numeric(df.get("od_away"), errors="coerce")
    return np.where(
        pick.eq("HOME"), od_home,
        np.where(pick.eq("DRAW"), od_draw, np.where(pick.eq("AWAY"), od_away, np.nan))
    )


def _summarize(df: pd.DataFrame, cohort: str, pick_col: str, actual_col: str) -> dict:
    d = df.copy()
    d["__pick"] = d[pick_col].map(_norm_pick)
    d["__actual"] = d[actual_col].map(_norm_pick)
    odds = pd.Series(_pick_odds(d, "__pick"), index=d.index)
    valid = d["__pick"].notna() & d["__actual"].notna() & pd.to_numeric(odds, errors="coerce").notna()
    d = d.loc[valid].copy()
    if d.empty:
        return {
            "cohort": cohort,
            "rows": int(len(df)),
            "graded": 0,
            "wins": 0,
            "losses": 0,
            "hit_rate": np.nan,
            "roi": np.nan,
            "profit": 0.0,
        }
    odds = pd.to_numeric(pd.Series(_pick_odds(d, "__pick"), index=d.index), errors="coerce")
    wins = (d["__pick"] == d["__actual"]).sum()
    graded = int(len(d))
    losses = graded - int(wins)
    profit = float(np.nansum(np.where(d["__pick"] == d["__actual"], odds - 1.0, -1.0)))
    roi = profit / graded if graded else np.nan
    hit = float(wins) / graded if graded else np.nan
    return {
        "cohort": cohort,
        "rows": int(len(df)),
        "graded": graded,
        "wins": float(wins),
        "losses": float(losses),
        "hit_rate": hit,
        "roi": roi,
        "profit": profit,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Side-by-side FTR CatBoost vs XGBoost audit")
    ap.add_argument("--base", default="predictions_output/walk_forward", help="walk_forward base dir")
    ap.add_argument("--outdir", default="predictions_output/walk_forward/_MASTER", help="output folder")
    ap.add_argument("--source", default="after", choices=["after", "raw", "combined"], help="scored source type")
    args = ap.parse_args()

    base = Path(args.base)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pattern = {
        "after": "w*/03_scored/DEPLOY_CANDIDATES_AFTER_GATES_SCORED_*.csv",
        "raw": "w*/03_scored/DEPLOY_CANDIDATES_RAW_SCORED_*.csv",
        "combined": "w*/03_scored/DEPLOY_COMBINED_SCORED_*.csv",
    }[args.source]

    files = sorted(base.glob(pattern))
    if not files:
        raise SystemExit(f"No files found for pattern: {pattern}")

    frames = []
    for p in files:
        try:
            df = pd.read_csv(p, low_memory=False)
            df["__src_file"] = p.name
            frames.append(df)
        except Exception:
            continue
    if not frames:
        raise SystemExit("No readable files found.")

    df = pd.concat(frames, axis=0, ignore_index=True)
    if "market" in df.columns:
        df = df[df["market"].astype(str).str.lower().eq("ftr")].copy()

    if df.empty:
        raise SystemExit("No FTR rows found in scored files.")

    actual_col = _resolve_actual_col(df)
    if not actual_col:
        raise SystemExit("No actual FTR result column found.")

    pick_cat = "model_top_pick" if "model_top_pick" in df.columns else None
    pick_xgb = "ftr_pick_xgb" if "ftr_pick_xgb" in df.columns else None
    if not pick_cat or not pick_xgb:
        raise SystemExit("Missing model pick columns (need model_top_pick and ftr_pick_xgb).")
    if df[pick_xgb].notna().sum() == 0:
        raise SystemExit("ftr_pick_xgb is empty in scored files. Re-run walkforward after XGB inference changes.")

    # Cohorts
    df_cat = df.copy()
    df_xgb = df.copy()
    df_both = df[df[pick_cat].notna() & df[pick_xgb].notna()].copy()
    df_agree = df_both[df_both[pick_cat] == df_both[pick_xgb]].copy()
    df_disagree = df_both[df_both[pick_cat] != df_both[pick_xgb]].copy()

    summary = []
    summary.append(_summarize(df_cat, "cat_all", pick_cat, actual_col))
    summary.append(_summarize(df_xgb, "xgb_all", pick_xgb, actual_col))
    summary.append(_summarize(df_agree, "consensus", pick_cat, actual_col))
    summary.append(_summarize(df_disagree, "cat_disagree", pick_cat, actual_col))
    summary.append(_summarize(df_disagree, "xgb_disagree", pick_xgb, actual_col))

    out_summary = pd.DataFrame(summary)
    out_summary.to_csv(outdir / "FTR_XGB_SIDE_BY_SIDE_AUDIT.csv", index=False)

    # By league
    by_league_rows = []
    for lg, g in df.groupby("league"):
        by_league_rows.append(_summarize(g, "cat_all", pick_cat, actual_col) | {"league": lg})
        by_league_rows.append(_summarize(g, "xgb_all", pick_xgb, actual_col) | {"league": lg})
        gb = g[g[pick_cat].notna() & g[pick_xgb].notna()]
        ga = gb[gb[pick_cat] == gb[pick_xgb]]
        gd = gb[gb[pick_cat] != gb[pick_xgb]]
        by_league_rows.append(_summarize(ga, "consensus", pick_cat, actual_col) | {"league": lg})
        by_league_rows.append(_summarize(gd, "cat_disagree", pick_cat, actual_col) | {"league": lg})
        by_league_rows.append(_summarize(gd, "xgb_disagree", pick_xgb, actual_col) | {"league": lg})

    pd.DataFrame(by_league_rows).to_csv(outdir / "FTR_XGB_SIDE_BY_SIDE_AUDIT__BY_LEAGUE.csv", index=False)

    # By side (pick)
    by_side_rows = []
    for side, g in df.groupby(df[pick_cat].map(_norm_pick)):
        if side:
            by_side_rows.append(_summarize(g, "cat_all", pick_cat, actual_col) | {"side": side})
    for side, g in df.groupby(df[pick_xgb].map(_norm_pick)):
        if side:
            by_side_rows.append(_summarize(g, "xgb_all", pick_xgb, actual_col) | {"side": side})
    pd.DataFrame(by_side_rows).to_csv(outdir / "FTR_XGB_SIDE_BY_SIDE_AUDIT__BY_SIDE.csv", index=False)

    print(f"Wrote: {outdir / 'FTR_XGB_SIDE_BY_SIDE_AUDIT.csv'}")


if __name__ == "__main__":
    main()
