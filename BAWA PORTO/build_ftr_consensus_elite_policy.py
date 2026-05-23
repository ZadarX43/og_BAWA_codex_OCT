#!/usr/bin/env python3
"""Consensus ELITE lane audit + allowlist builder (Cat + XGB agree).

Outputs (default to _MASTER/CONSENSUS_ELITE):
  - FTR_CONSENSUS_ELITE_AUDIT.csv
  - FTR_CONSENSUS_ELITE_AUDIT__BY_LEAGUE.csv
  - FTR_CONSENSUS_ELITE_ALLOWLIST.csv
  - FTR_CONSENSUS_ELITE_POLICY.json
"""
from __future__ import annotations

import argparse
import json
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
    ap = argparse.ArgumentParser(description="Consensus ELITE audit + allowlist")
    ap.add_argument("--base", default="predictions_output/walk_forward")
    ap.add_argument("--outdir", default="predictions_output/walk_forward/_MASTER/CONSENSUS_ELITE")
    ap.add_argument("--source", default="after", choices=["after", "raw", "combined"])
    ap.add_argument("--min-rows", type=int, default=50)
    ap.add_argument("--min-roi-lift", type=float, default=0.02)
    ap.add_argument("--min-hit-lift", type=float, default=0.02)
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

    cat_all = df.copy()
    consensus = df[df["__consensus"]].copy()

    # Global summary
    summary = []
    summary.append(_summarize(cat_all, "cat_all"))
    summary.append(_summarize(consensus, "consensus_all"))
    out_summary = pd.DataFrame(summary)
    out_summary.to_csv(outdir / "FTR_CONSENSUS_ELITE_AUDIT.csv", index=False)

    # By league
    by_league = []
    for lg, g in df.groupby("league"):
        by_league.append(_summarize(g, "cat_all") | {"league": lg})
        g_cons = g[g["__consensus"]]
        by_league.append(_summarize(g_cons, "consensus_all") | {"league": lg})

    by_league_df = pd.DataFrame(by_league)
    by_league_df.to_csv(outdir / "FTR_CONSENSUS_ELITE_AUDIT__BY_LEAGUE.csv", index=False)

    # Coverage summary (cat vs consensus)
    cov_rows = []
    for lg, g in df.groupby("league"):
        cat_rows = int(len(g))
        cons_rows = int(g["__consensus"].sum())
        cov = (cons_rows / cat_rows * 100.0) if cat_rows else np.nan
        cov_rows.append({
            "league": lg,
            "cat_rows": cat_rows,
            "consensus_rows": cons_rows,
            "coverage_pct": cov,
        })
    cov_df = pd.DataFrame(cov_rows).sort_values(["coverage_pct"], ascending=[False])
    cov_df.to_csv(outdir / "FTR_CONSENSUS_ELITE_COVERAGE.csv", index=False)

    # Allowlist: consensus uplift vs cat_all
    allow_rows = []
    for lg in df["league"].dropna().unique().tolist():
        g = by_league_df[(by_league_df["league"] == lg) & (by_league_df["cohort"] == "cat_all")]
        c = by_league_df[(by_league_df["league"] == lg) & (by_league_df["cohort"] == "consensus_all")]
        if g.empty or c.empty:
            continue
        g = g.iloc[0]
        c = c.iloc[0]
        if c["graded"] < int(args.min_rows):
            continue
        hit_lift = (c["hit_rate"] - g["hit_rate"]) if (pd.notna(c["hit_rate"]) and pd.notna(g["hit_rate"])) else np.nan
        roi_lift = (c["roi"] - g["roi"]) if (pd.notna(c["roi"]) and pd.notna(g["roi"])) else np.nan
        if pd.isna(hit_lift) or pd.isna(roi_lift):
            continue
        if (hit_lift >= float(args.min_hit_lift)) and (roi_lift >= float(args.min_roi_lift)):
            allow_rows.append({
                "league": lg,
                "consensus_rows": int(c["graded"]),
                "consensus_hit_rate": c["hit_rate"],
                "consensus_roi": c["roi"],
                "hit_rate_lift_vs_cat": hit_lift,
                "roi_lift_vs_cat": roi_lift,
            })

    allow_df = pd.DataFrame(allow_rows).sort_values(["roi_lift_vs_cat","hit_rate_lift_vs_cat"], ascending=[False, False])
    allow_df.to_csv(outdir / "FTR_CONSENSUS_ELITE_ALLOWLIST.csv", index=False)

    policy = {
        "min_rows": int(args.min_rows),
        "min_hit_lift": float(args.min_hit_lift),
        "min_roi_lift": float(args.min_roi_lift),
        "allowlist": allow_df["league"].tolist() if not allow_df.empty else [],
    }
    with open(outdir / "FTR_CONSENSUS_ELITE_POLICY.json", "w", encoding="utf-8") as fh:
        json.dump(policy, fh, indent=2)

    print(f"Wrote: {outdir / 'FTR_CONSENSUS_ELITE_AUDIT.csv'}")


if __name__ == "__main__":
    main()
