#!/usr/bin/env python3
"""Generate markdown summary comparing meta super-score vs consensus lane."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Meta vs Consensus summary")
    ap.add_argument("--meta-dir", required=True, help="Directory containing META outputs")
    ap.add_argument("--out", required=True, help="Output markdown path")
    return ap.parse_args()


def pct(x: float) -> str:
    return f"{x*100:.2f}%"


def fmt_int(x: float) -> str:
    return f"{int(x):,}"


def md_table(df, cols):
    if df is None or df.empty:
        return "(missing)"
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join([str(r[c]) for c in cols]) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    meta_dir = Path(args.meta_dir)
    out_path = Path(args.out)

    win_path = meta_dir / "FTR_META_SUPER_SCORE__WINDOW_SUMMARY.csv"
    league_path = meta_dir / "FTR_META_SUPER_SCORE__BY_LEAGUE.csv"

    if not win_path.exists():
        raise SystemExit(f"Missing: {win_path}")

    win = pd.read_csv(win_path)
    league = pd.read_csv(league_path) if league_path.exists() else pd.DataFrame()

    # global aggregates
    agg = win[["consensus_rows", "meta_rows", "consensus_hit_rate", "consensus_roi", "meta_hit_rate", "meta_roi"]].copy()
    global_row = {
        "consensus_rows": int(agg["consensus_rows"].sum()),
        "meta_rows": int(agg["meta_rows"].sum()),
        "consensus_hit_rate": float(agg["consensus_hit_rate"].mean()),
        "consensus_roi": float(agg["consensus_roi"].mean()),
        "meta_hit_rate": float(agg["meta_hit_rate"].mean()),
        "meta_roi": float(agg["meta_roi"].mean()),
    }

    md = []
    md.append("# Meta vs Consensus Summary\n")
    md.append("## Executive summary")
    md.append(f"- Total consensus rows: {fmt_int(global_row['consensus_rows'])}")
    md.append(f"- Total meta rows: {fmt_int(global_row['meta_rows'])}")
    md.append(f"- Avg consensus hit rate: {pct(global_row['consensus_hit_rate'])}")
    md.append(f"- Avg consensus ROI: {global_row['consensus_roi']:.4f}")
    md.append(f"- Avg meta hit rate: {pct(global_row['meta_hit_rate'])}")
    md.append(f"- Avg meta ROI: {global_row['meta_roi']:.4f}")

    md.append("\n## By window")
    win_fmt = win.copy()
    for c in ["consensus_rows", "meta_rows"]:
        win_fmt[c] = win_fmt[c].map(fmt_int)
    for c in ["consensus_hit_rate", "meta_hit_rate"]:
        win_fmt[c] = win_fmt[c].map(pct)
    for c in ["consensus_roi", "meta_roi"]:
        win_fmt[c] = win_fmt[c].map(lambda x: f"{x:.4f}")
    md.append(md_table(win_fmt[["window_id", "consensus_rows", "meta_rows", "consensus_hit_rate", "meta_hit_rate", "consensus_roi", "meta_roi"]],
                       ["window_id", "consensus_rows", "meta_rows", "consensus_hit_rate", "meta_hit_rate", "consensus_roi", "meta_roi"]))

    md.append("\n## By league")
    if not league.empty:
        league_fmt = league.copy()
        for c in ["consensus_rows", "meta_rows"]:
            league_fmt[c] = league_fmt[c].fillna(0).map(fmt_int)
        for c in ["consensus_hit_rate", "meta_hit_rate"]:
            league_fmt[c] = league_fmt[c].fillna(0).map(pct)
        for c in ["consensus_roi", "meta_roi"]:
            league_fmt[c] = league_fmt[c].fillna(0).map(lambda x: f"{x:.4f}")
        md.append(md_table(league_fmt[["league", "consensus_rows", "meta_rows", "consensus_hit_rate", "meta_hit_rate", "consensus_roi", "meta_roi"]],
                           ["league", "consensus_rows", "meta_rows", "consensus_hit_rate", "meta_hit_rate", "consensus_roi", "meta_roi"]))
    else:
        md.append("(missing)")

    out_path.write_text("\n".join(md))
    print(out_path)


if __name__ == "__main__":
    main()
