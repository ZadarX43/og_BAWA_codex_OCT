#!/usr/bin/env python3
"""Cat+XGB consensus audit for OU25.

Outputs:
  - OU25_XGB_CONSENSUS_AUDIT.csv
  - OU25_XGB_CONSENSUS_AUDIT__BY_LEAGUE.csv
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
    return s if s in {"OVER25", "UNDER25"} else None


def _pick_odds(df: pd.DataFrame, pick_col: str) -> pd.Series:
    pick = df[pick_col].astype("string").fillna("").str.upper().str.strip()
    od_over = pd.to_numeric(df.get("od_over", df.get("odds_ft_over25")), errors="coerce")
    od_under = pd.to_numeric(df.get("od_under", df.get("odds_ft_under25")), errors="coerce")
    return pd.Series(
        np.where(pick.eq("OVER25"), od_over, np.where(pick.eq("UNDER25"), od_under, np.nan)),
        index=df.index,
    )


def _summarize(df: pd.DataFrame, cohort: str) -> dict:
    d = df.copy()
    d["__pick"] = d["selection"].map(_norm_pick)
    if "actual_over25" in d.columns:
        d["__actual"] = np.where(pd.to_numeric(d["actual_over25"], errors="coerce").eq(1), "OVER25", "UNDER25")
    else:
        d["__actual"] = None
    odds = _pick_odds(d, "__pick")
    valid = d["__pick"].notna() & pd.Series(d["__actual"]).notna() & pd.to_numeric(odds, errors="coerce").notna()
    d = d.loc[valid].copy()
    if d.empty:
        return {"cohort": cohort, "rows": int(len(df)), "graded": 0, "wins": 0, "losses": 0, "hit_rate": np.nan, "roi": np.nan, "profit": 0.0}
    odds = pd.to_numeric(_pick_odds(d, "__pick"), errors="coerce")
    wins = (d["__pick"] == d["__actual"]).sum()
    graded = int(len(d))
    profit = float(np.nansum(np.where(d["__pick"] == d["__actual"], odds - 1.0, -1.0)))
    return {
        "cohort": cohort,
        "rows": int(len(df)),
        "graded": graded,
        "wins": float(wins),
        "losses": float(graded - int(wins)),
        "hit_rate": float(wins) / graded if graded else np.nan,
        "roi": profit / graded if graded else np.nan,
        "profit": profit,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="OU25 consensus audit (Cat + XGB)")
    ap.add_argument("--base", default="predictions_output/walk_forward")
    ap.add_argument("--outdir", default="predictions_output/walk_forward/_MASTER")
    ap.add_argument("--source", default="combined", choices=["after", "raw", "combined"])
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
            frames.append(pd.read_csv(p, low_memory=False))
        except Exception:
            continue
    if not frames:
        raise SystemExit("No readable files found.")

    df = pd.concat(frames, axis=0, ignore_index=True)
    df = df[df["market"].astype(str).str.lower().eq("ou25")].copy()
    if "ou25_pick_xgb" not in df.columns:
        raise SystemExit("Missing ou25_pick_xgb column; re-run scoring after OU25 XGB wiring.")
    if df["ou25_pick_xgb"].notna().sum() == 0:
        raise SystemExit("ou25_pick_xgb is empty; re-run scoring after OU25 XGB bundle load.")

    df["__consensus"] = (
        df["model_top_pick"].astype("string").fillna("").str.upper().str.strip()
        == df["ou25_pick_xgb"].astype("string").fillna("").str.upper().str.strip()
    )

    consensus = df[df["__consensus"]].copy()

    summary = [
        _summarize(df, "cat_all"),
        _summarize(consensus, "consensus_all"),
    ]
    out_summary = pd.DataFrame(summary)
    out_summary.to_csv(outdir / "OU25_XGB_CONSENSUS_AUDIT.csv", index=False)

    by_league = []
    for lg, g in df.groupby("league"):
        by_league.append(_summarize(g, "cat_all") | {"league": lg})
        by_league.append(_summarize(g[g["__consensus"]], "consensus_all") | {"league": lg})
    pd.DataFrame(by_league).to_csv(outdir / "OU25_XGB_CONSENSUS_AUDIT__BY_LEAGUE.csv", index=False)
    print(f"Wrote: {outdir / 'OU25_XGB_CONSENSUS_AUDIT.csv'}")


if __name__ == "__main__":
    main()
