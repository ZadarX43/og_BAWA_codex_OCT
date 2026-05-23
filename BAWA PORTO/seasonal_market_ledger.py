#!/usr/bin/env python3
import argparse, re
from pathlib import Path
import pandas as pd
import numpy as np

DATE_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})")

def safe_num(s):
    return pd.to_numeric(s, errors="coerce")

def roi_from_correct_odds(correct, odds):
    # profit per bet: correct*(od-1) - (1-correct) == correct*od - 1
    return (correct * odds - 1.0)

def summarize_block(df, label, market=None, pick=None):
    if df is None or len(df) == 0:
        return None
    out = {}
    out["label"] = label
    if market is not None: out["market"] = market
    if pick is not None: out["pick"] = pick

    correct_col = "correct" if "correct" in df.columns else None
    if correct_col is None:
        return None

    odds_col = None
    for c in ["bookie_od", "od", "odds", "price"]:
        if c in df.columns:
            odds_col = c
            break
    if odds_col is None:
        return None

    correct = safe_num(df[correct_col]).fillna(0.0)
    odds = safe_num(df[odds_col]).fillna(np.nan)

    out["n"] = int(len(df))
    out["hit"] = float(correct.mean()) if len(df) else np.nan
    out["avg_od"] = float(odds.mean()) if len(df) else np.nan
    out["roi"] = float(roi_from_correct_odds(correct, odds).mean()) if len(df) else np.nan
    return out

def parse_window_from_filename(p: Path):
    m = DATE_RE.search(p.name)
    if not m:
        return None, None
    return m.group(1), m.group(2)

def month_key(date_from: str):
    # YYYY-MM-.. -> YYYY-MM
    return date_from[:7]

def month_of_year(date_from: str):
    # YYYY-MM-.. -> MM
    return date_from[5:7]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="predictions_output root (or a subfolder)")
    ap.add_argument("--pattern", default="*__GATED__BACKTEST_SCORED.csv", help="glob pattern")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--markets", default="ftr,ou25,btts", help="comma list")
    ap.add_argument("--min_n_print", type=int, default=15, help="guardrail for pick splits (for reporting only)")

    # FTR strict overlay thresholds (optional)
    ap.add_argument("--ftr_ppg_abs", type=float, default=0.70)
    ap.add_argument("--ftr_power_abs", type=float, default=15.08)
    args = ap.parse_args()

    root = Path(args.root)
    paths = sorted(root.rglob(args.pattern))
    markets = [m.strip() for m in args.markets.split(",") if m.strip()]

    rows = []
    errors = 0

    for p in paths:
        date_from, date_to = parse_window_from_filename(p)
        if not date_from:
            continue

        try:
            df = pd.read_csv(p)
        except Exception as e:
            errors += 1
            rows.append({"file": str(p), "error": str(e)})
            continue

        if "market" not in df.columns:
            continue

        df["market"] = df["market"].astype(str).str.lower().str.strip()

        mk = month_key(date_from)
        moy = month_of_year(date_from)

        # baseline per market
        for mkt in markets:
            d = df[df["market"] == mkt].copy()
            block = summarize_block(d, label="BASE", market=mkt)
            if block:
                block.update({
                    "month": mk,
                    "month_of_year": moy,
                    "date_from": date_from,
                    "date_to": date_to,
                    "file": str(p),
                })
                rows.append(block)

            # pick splits where possible
            if "bookie_pick" in d.columns:
                d["bookie_pick"] = d["bookie_pick"].astype(str).str.upper().str.strip()
                for pick in sorted(d["bookie_pick"].dropna().unique()):
                    dd = d[d["bookie_pick"] == pick]
                    blockp = summarize_block(dd, label="PICK", market=mkt, pick=pick)
                    if blockp:
                        blockp.update({
                            "month": mk,
                            "month_of_year": moy,
                            "date_from": date_from,
                            "date_to": date_to,
                            "file": str(p),
                        })
                        rows.append(blockp)

        # FTR strict overlay (parallel view)
        d_ftr = df[df["market"] == "ftr"].copy()
        if len(d_ftr):
            # column names in your file
            ppg_col = "ppg_diff_pre" if "ppg_diff_pre" in d_ftr.columns else None
            pow_col = "power_diff" if "power_diff" in d_ftr.columns else None

            if ppg_col or pow_col:
                ppg_ok = safe_num(d_ftr[ppg_col]).abs() >= args.ftr_ppg_abs if ppg_col else pd.Series(False, index=d_ftr.index)
                pow_ok = safe_num(d_ftr[pow_col]).abs() >= args.ftr_power_abs if pow_col else pd.Series(False, index=d_ftr.index)

                d_or = d_ftr[ppg_ok | pow_ok]
                d_ppg_only = d_ftr[ppg_ok & ~pow_ok]
                d_pow_only = d_ftr[pow_ok & ~ppg_ok]
                d_both = d_ftr[ppg_ok & pow_ok]

                for label, dd in [
                    ("FTR_OR", d_or),
                    ("FTR_PPG_ONLY", d_ppg_only),
                    ("FTR_POWER_ONLY", d_pow_only),
                    ("FTR_BOTH", d_both),
                ]:
                    block = summarize_block(dd, label=label, market="ftr")
                    if block:
                        block.update({
                            "month": mk,
                            "month_of_year": moy,
                            "date_from": date_from,
                            "date_to": date_to,
                            "file": str(p),
                            "ftr_ppg_abs": args.ftr_ppg_abs,
                            "ftr_power_abs": args.ftr_power_abs,
                        })
                        rows.append(block)

    out_df = pd.DataFrame(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Wrote: {out_path}")
    if errors:
        print(f"Warnings: {errors} files had read errors (see rows with 'error').")
    print(f"Rows: {len(out_df)}")
    print(out_df.head(20).to_string(index=False))

    # quick seasonality rollup (month-of-year) for BASE only
    if len(out_df):
        base = out_df[(out_df["label"] == "BASE") & (out_df["market"].isin(markets))].copy()
        if len(base):
            roll = (
                base.groupby(["month_of_year", "market"], dropna=False)
                    .agg(n=("n","sum"), hit=("hit","mean"), roi=("roi","mean"), avg_od=("avg_od","mean"))
                    .reset_index()
                    .sort_values(["month_of_year","market"])
            )
            print("\n=== Seasonality rollup (month_of_year x market) [BASE] ===")
            print(roll.to_string(index=False))

if __name__ == "__main__":
    main()