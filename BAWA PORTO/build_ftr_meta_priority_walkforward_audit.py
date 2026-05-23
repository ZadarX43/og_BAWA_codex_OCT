#!/usr/bin/env python3
"""
Summarize META_ELITE vs META_STANDARD distribution from walkforward tier outputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Meta tier walkforward audit")
    ap.add_argument("--base", required=True, help="predictions_output/walk_forward")
    ap.add_argument("--outdir", required=True, help="output directory")
    ap.add_argument("--source", default="tiers", choices=["tiers"], help="Currently only tiered outputs")
    return ap.parse_args()


def _find_tier_files(base: Path) -> list[Path]:
    return sorted(base.glob("w*/02_deploy/*__DEPLOY_TIER_*.csv"))


def main() -> None:
    args = parse_args()
    base = Path(args.base)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = _find_tier_files(base)
    if not files:
        raise SystemExit("No tier files found.")

    frames = []
    for p in files:
        df = pd.read_csv(p)
        if df is None or df.empty:
            continue
        frames.append(df)

    if not frames:
        raise SystemExit("No data loaded.")

    df = pd.concat(frames, ignore_index=True, sort=False)

    # normalize meta_tier
    if "meta_tier" not in df.columns:
        df["meta_tier"] = "MISSING"
    mt = df["meta_tier"].astype("string").fillna("")
    mt = mt.replace({"": "MISSING"})
    df["meta_tier"] = mt

    if "deploy_tier" not in df.columns:
        df["deploy_tier"] = "UNKNOWN"

    # summary
    summary = (
        df.groupby(["deploy_tier", "meta_tier"])
          .size()
          .reset_index(name="rows")
    )
    summary["coverage_pct"] = summary.groupby("deploy_tier")["rows"].transform(lambda s: s / s.sum())

    by_league = (
        df.groupby(["league", "deploy_tier", "meta_tier"])
          .size()
          .reset_index(name="rows")
    )
    by_league["coverage_pct"] = by_league.groupby(["league", "deploy_tier"])["rows"].transform(lambda s: s / s.sum())

    summary_path = outdir / "FTR_META_TIER_WALKFORWARD__SUMMARY__TIERS.csv"
    league_path = outdir / "FTR_META_TIER_WALKFORWARD__BY_LEAGUE__TIERS.csv"
    summary.to_csv(summary_path, index=False)
    by_league.to_csv(league_path, index=False)

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {league_path}")


if __name__ == "__main__":
    main()
