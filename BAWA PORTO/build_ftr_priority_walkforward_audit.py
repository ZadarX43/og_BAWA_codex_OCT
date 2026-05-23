#!/usr/bin/env python3
"""Walkforward master audit: FTR priority tier distribution.

Outputs:
  - FTR_PRIORITY_WALKFORWARD__SUMMARY.csv
  - FTR_PRIORITY_WALKFORWARD__BY_LEAGUE.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def _load_scored(base: Path, source: str) -> pd.DataFrame:
    pattern = {
        "after": "w*/03_scored/DEPLOY_CANDIDATES_AFTER_GATES_SCORED_*.csv",
        "raw": "w*/03_scored/DEPLOY_CANDIDATES_RAW_SCORED_*.csv",
        "combined": "w*/03_scored/DEPLOY_COMBINED_SCORED_*.csv",
    }[source]
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
    return pd.concat(frames, axis=0, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="FTR priority walkforward audit")
    ap.add_argument("--base", default="predictions_output/walk_forward")
    ap.add_argument("--outdir", default="predictions_output/walk_forward/_MASTER/CONSENSUS_ELITE")
    ap.add_argument("--source", default="after", choices=["after", "raw", "combined", "tiers", "both", "all"])
    args = ap.parse_args()

    base = Path(args.base)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def _load_tiered() -> pd.DataFrame:
        files = []
        files += sorted(base.glob("w*/02_deploy/*__DEPLOY_TIER_ELITE*.csv"))
        files += sorted(base.glob("w*/02_deploy/*__DEPLOY_TIER_STANDARD*.csv"))
        files += sorted(base.glob("w*/02_deploy/*__DEPLOY_TIER_OBSERVE*.csv"))
        if not files:
            raise SystemExit("No tiered deploy outputs found for --source tiers.")
        frames = []
        for p in files:
            try:
                frames.append(pd.read_csv(p, low_memory=False))
            except Exception:
                continue
        if not frames:
            raise SystemExit("No readable tier files found.")
        return pd.concat(frames, axis=0, ignore_index=True)

    # Support --source both: write candidate + tiered outputs with suffixes
    if args.source in ("both", "all"):
        df = _load_scored(base, "after")
    elif args.source == "tiers":
        df = _load_tiered()
    else:
        df = _load_scored(base, args.source)
    mk = df.get("market", "").astype("string").fillna("").str.lower().str.strip()
    ftr = df.loc[mk.eq("ftr")].copy()
    if ftr.empty:
        raise SystemExit("No FTR rows found in scored files.")

    # If ftr_priority is missing, derive from cat vs xgb picks (if available).
    if "ftr_priority" not in ftr.columns:
        if "ftr_pick_xgb" not in ftr.columns:
            raise SystemExit("Missing ftr_priority and ftr_pick_xgb; re-run walkforward with XGB outputs.")
        # Warn if XGB picks are empty
        if ftr["ftr_pick_xgb"].notna().sum() == 0:
            raise SystemExit("ftr_pick_xgb is empty; re-run walkforward after XGB inference fix.")

        cat = ftr.get("model_top_pick", ftr.get("selection", "")).astype("string").fillna("").str.upper().str.strip()
        xgb = ftr.get("ftr_pick_xgb", ftr.get("model_top_pick_xgb", "")).astype("string").fillna("").str.upper().str.strip()
        consensus = (cat != "") & (xgb != "") & cat.eq(xgb)

        ftr["ftr_priority"] = "CAT_ELITE"
        ftr.loc[consensus, "ftr_priority"] = "CONSENSUS_ELITE"
        ftr["ftr_priority"] = ftr["ftr_priority"].astype("string")

    if "deploy_tier" not in ftr.columns:
        ftr["deploy_tier"] = "UNKNOWN"
    ftr["deploy_tier"] = ftr["deploy_tier"].astype("string").fillna("UNKNOWN").str.upper().str.strip()
    ftr["ftr_priority"] = ftr["ftr_priority"].astype("string").fillna("UNKNOWN").str.upper().str.strip()

    written = []

    def _write_outputs(ftr_df: pd.DataFrame, suffix: str = "") -> None:
        # Summary (overall)
        summary = (
            ftr_df.groupby(["deploy_tier", "ftr_priority"], dropna=False)
            .size()
            .reset_index(name="rows")
            .sort_values(["deploy_tier", "rows"], ascending=[True, False])
        )
        # Add coverage % within each deploy_tier
        tier_totals = summary.groupby("deploy_tier")["rows"].transform("sum")
        summary["coverage_pct"] = (summary["rows"] / tier_totals * 100.0).round(6)
        p_summary = outdir / f"FTR_PRIORITY_WALKFORWARD__SUMMARY{suffix}.csv"
        summary.to_csv(p_summary, index=False)
        written.append({"file": str(p_summary), "rows": int(len(summary))})

        # By league
        by_league = (
            ftr_df.groupby(["league", "deploy_tier", "ftr_priority"], dropna=False)
            .size()
            .reset_index(name="rows")
            .sort_values(["league", "deploy_tier", "rows"], ascending=[True, True, False])
        )
        # Add coverage % within each (league, deploy_tier)
        league_totals = by_league.groupby(["league", "deploy_tier"])["rows"].transform("sum")
        by_league["coverage_pct"] = (by_league["rows"] / league_totals * 100.0).round(6)
        p_league = outdir / f"FTR_PRIORITY_WALKFORWARD__BY_LEAGUE{suffix}.csv"
        by_league.to_csv(p_league, index=False)
        written.append({"file": str(p_league), "rows": int(len(by_league))})

    def _map_unknown_tier(ftr_df: pd.DataFrame, label: str) -> pd.DataFrame:
        if ftr_df is None or ftr_df.empty:
            return ftr_df
        out = ftr_df.copy()
        if "deploy_tier" not in out.columns:
            out["deploy_tier"] = label
        out["deploy_tier"] = out["deploy_tier"].astype("string").fillna("UNKNOWN").str.upper().str.strip()
        out.loc[out["deploy_tier"].eq("UNKNOWN"), "deploy_tier"] = label
        return out

    if args.source in ("both", "all"):
        # Candidate universe outputs
        ftr_c = _map_unknown_tier(ftr, "CANDIDATE")
        _write_outputs(ftr_c, suffix="__CANDIDATES")
        # Tiered deploy outputs
        df_tier = _load_tiered()
        mk_t = df_tier.get("market", "").astype("string").fillna("").str.lower().str.strip()
        ftr_t = df_tier.loc[mk_t.eq("ftr")].copy()
        if ftr_t.empty:
            raise SystemExit("No FTR rows found in tiered deploy outputs.")
        if "ftr_priority" not in ftr_t.columns:
            # Derive if possible
            if "ftr_pick_xgb" not in ftr_t.columns or ftr_t["ftr_pick_xgb"].notna().sum() == 0:
                raise SystemExit("Tiered outputs missing ftr_priority and usable ftr_pick_xgb; re-run deploy with priority stamping.")
            cat_t = ftr_t.get("model_top_pick", ftr_t.get("selection", "")).astype("string").fillna("").str.upper().str.strip()
            xgb_t = ftr_t.get("ftr_pick_xgb", ftr_t.get("model_top_pick_xgb", "")).astype("string").fillna("").str.upper().str.strip()
            consensus_t = (cat_t != "") & (xgb_t != "") & cat_t.eq(xgb_t)
            ftr_t["ftr_priority"] = "CAT_ELITE"
            ftr_t.loc[consensus_t, "ftr_priority"] = "CONSENSUS_ELITE"
        if "deploy_tier" not in ftr_t.columns:
            ftr_t["deploy_tier"] = "UNKNOWN"
        ftr_t["deploy_tier"] = ftr_t["deploy_tier"].astype("string").fillna("UNKNOWN").str.upper().str.strip()
        ftr_t["ftr_priority"] = ftr_t["ftr_priority"].astype("string").fillna("UNKNOWN").str.upper().str.strip()
        _write_outputs(ftr_t, suffix="__TIERS")

        if args.source == "all":
            # raw/after/combined outputs
            for src in ("raw", "after", "combined"):
                df_x = _load_scored(base, src)
                mk_x = df_x.get("market", "").astype("string").fillna("").str.lower().str.strip()
                ftr_x = df_x.loc[mk_x.eq("ftr")].copy()
                if ftr_x.empty:
                    continue
                if "ftr_priority" not in ftr_x.columns:
                    # derive if missing
                    if "ftr_pick_xgb" not in ftr_x.columns or ftr_x["ftr_pick_xgb"].notna().sum() == 0:
                        continue
                    cat_x = ftr_x.get("model_top_pick", ftr_x.get("selection", "")).astype("string").fillna("").str.upper().str.strip()
                    xgb_x = ftr_x.get("ftr_pick_xgb", ftr_x.get("model_top_pick_xgb", "")).astype("string").fillna("").str.upper().str.strip()
                    consensus_x = (cat_x != "") & (xgb_x != "") & cat_x.eq(xgb_x)
                    ftr_x["ftr_priority"] = "CAT_ELITE"
                    ftr_x.loc[consensus_x, "ftr_priority"] = "CONSENSUS_ELITE"
                if "deploy_tier" not in ftr_x.columns:
                    ftr_x["deploy_tier"] = "UNKNOWN"
                ftr_x["deploy_tier"] = ftr_x["deploy_tier"].astype("string").fillna("UNKNOWN").str.upper().str.strip()
                ftr_x["ftr_priority"] = ftr_x["ftr_priority"].astype("string").fillna("UNKNOWN").str.upper().str.strip()
                # map UNKNOWN to source label
                ftr_x = _map_unknown_tier(ftr_x, src.upper())
                _write_outputs(ftr_x, suffix=f"__{src.upper()}")

        # index file
        idx_path = outdir / "FTR_PRIORITY_WALKFORWARD__INDEX.csv"
        pd.DataFrame(written).to_csv(idx_path, index=False)
        print(f"Wrote: {idx_path}")
        print(f"Wrote: {outdir / 'FTR_PRIORITY_WALKFORWARD__SUMMARY__CANDIDATES.csv'}")
        print(f"Wrote: {outdir / 'FTR_PRIORITY_WALKFORWARD__SUMMARY__TIERS.csv'}")
        return

    # Single source outputs
    # Summary (overall)
    summary = (
        ftr.groupby(["deploy_tier", "ftr_priority"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["deploy_tier", "rows"], ascending=[True, False])
    )
    # Add coverage % within each deploy_tier
    tier_totals = summary.groupby("deploy_tier")["rows"].transform("sum")
    summary["coverage_pct"] = (summary["rows"] / tier_totals * 100.0).round(6)
    summary.to_csv(outdir / "FTR_PRIORITY_WALKFORWARD__SUMMARY.csv", index=False)

    # By league
    by_league = (
        ftr.groupby(["league", "deploy_tier", "ftr_priority"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["league", "deploy_tier", "rows"], ascending=[True, True, False])
    )
    # Add coverage % within each (league, deploy_tier)
    league_totals = by_league.groupby(["league", "deploy_tier"])["rows"].transform("sum")
    by_league["coverage_pct"] = (by_league["rows"] / league_totals * 100.0).round(6)
    by_league.to_csv(outdir / "FTR_PRIORITY_WALKFORWARD__BY_LEAGUE.csv", index=False)

    idx_path = outdir / "FTR_PRIORITY_WALKFORWARD__INDEX.csv"
    pd.DataFrame(written).to_csv(idx_path, index=False)
    print(f"Wrote: {idx_path}")
    print(f"Wrote: {outdir / 'FTR_PRIORITY_WALKFORWARD__SUMMARY.csv'}")


if __name__ == "__main__":
    main()
