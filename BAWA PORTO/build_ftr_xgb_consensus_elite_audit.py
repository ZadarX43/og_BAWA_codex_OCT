#!/usr/bin/env python3
"""Cat+XGB consensus ELITE lane audit for FTR.

Outputs:
  - FTR_XGB_CONSENSUS_ELITE_AUDIT.csv
  - FTR_XGB_CONSENSUS_ELITE_AUDIT__BY_LEAGUE.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def _norm_pick(v) -> str | None:
    s = str(v).strip().upper()
    if not s or s == "NAN":
        return None
    if s in {"H", "HOME"}:
        return "HOME"
    if s in {"A", "AWAY"}:
        return "AWAY"
    if s in {"D", "DRAW", "X"}:
        return "DRAW"
    return s if s in {"HOME", "AWAY", "DRAW"} else None


def _pick_odds(df: pd.DataFrame, pick_col: str) -> pd.Series:
    pick = df[pick_col]
    od_home = pd.to_numeric(df.get("od_home"), errors="coerce")
    od_draw = pd.to_numeric(df.get("od_draw"), errors="coerce")
    od_away = pd.to_numeric(df.get("od_away"), errors="coerce")
    return pd.Series(
        np.where(
            pick.eq("HOME"), od_home,
            np.where(pick.eq("DRAW"), od_draw, np.where(pick.eq("AWAY"), od_away, np.nan))
        ),
        index=df.index,
    )


def _summarize(df: pd.DataFrame, cohort: str) -> dict:
    d = df.copy()
    d["__pick"] = d["selection"].map(_norm_pick)
    d["__actual"] = d.get("actual_ftr", "").map(_norm_pick)
    odds = _pick_odds(d, "__pick")
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
    odds = pd.to_numeric(_pick_odds(d, "__pick"), errors="coerce")
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
    ap = argparse.ArgumentParser(description="FTR consensus ELITE audit (Cat + XGB)")
    ap.add_argument("--base", default="predictions_output/walk_forward")
    ap.add_argument("--outdir", default="predictions_output/walk_forward/_MASTER")
    ap.add_argument("--source", default="after", choices=["after", "raw", "combined"])
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
            frames.append(df)
        except Exception:
            continue
    if not frames:
        raise SystemExit("No readable files found.")

    df = pd.concat(frames, axis=0, ignore_index=True)
    df = df[df["market"].astype(str).str.lower().eq("ftr")].copy()

    if "ftr_pick_xgb" not in df.columns:
        raise SystemExit("Missing ftr_pick_xgb column; re-run walkforward with XGB.")
    if df["ftr_pick_xgb"].notna().sum() == 0:
        raise SystemExit("ftr_pick_xgb is empty; re-run walkforward after XGB inference fix.")

    df["model_top_pick_xgb"] = df["ftr_pick_xgb"]
    df["__consensus"] = (
        df["model_top_pick"].astype("string").fillna("").str.upper().str.strip()
        == df["model_top_pick_xgb"].astype("string").fillna("").str.upper().str.strip()
    )

    consensus = df[df["__consensus"]].copy()
    consensus_elite = consensus.copy()
    if "deploy_tier" in consensus_elite.columns:
        consensus_elite = consensus_elite[consensus_elite["deploy_tier"].astype("string").str.upper().isin(["ELITE", "STANDARD"])]

    summary = []
    summary.append(_summarize(df, "cat_all"))
    summary.append(_summarize(consensus, "consensus_all"))
    summary.append(_summarize(consensus_elite, "consensus_elite"))

    out_summary = pd.DataFrame(summary)
    out_summary.to_csv(outdir / "FTR_XGB_CONSENSUS_ELITE_AUDIT.csv", index=False)

    by_league = []
    for lg, g in df.groupby("league"):
        by_league.append(_summarize(g, "cat_all") | {"league": lg})
        g_cons = g[g["__consensus"]]
        by_league.append(_summarize(g_cons, "consensus_all") | {"league": lg})
        g_el = g_cons
        if "deploy_tier" in g_el.columns:
            g_el = g_el[g_el["deploy_tier"].astype("string").str.upper().isin(["ELITE", "STANDARD"])]
        by_league.append(_summarize(g_el, "consensus_elite") | {"league": lg})

    pd.DataFrame(by_league).to_csv(outdir / "FTR_XGB_CONSENSUS_ELITE_AUDIT__BY_LEAGUE.csv", index=False)

    print(f"Wrote: {outdir / 'FTR_XGB_CONSENSUS_ELITE_AUDIT.csv'}")


if __name__ == "__main__":
    main()
