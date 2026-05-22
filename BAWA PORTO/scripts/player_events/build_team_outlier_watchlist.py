from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def classify_watch(row: pd.Series) -> str:
    if row["threshold_posture"] == "RELAX" and row["outlier_score"] >= 0.30:
        return "LIVE_RELAX_OUTLIER"
    if row["threshold_posture"] == "TIGHTEN" and row["outlier_score"] >= 0.20:
        return "LIVE_TIGHTEN_OUTLIER"
    if row["rows"] >= 2 and row["fixtures"] >= 2:
        return "SMALL_SAMPLE_WATCH"
    return "LOW_SIGNAL"


def build_watchlist(input_csv: str, output_csv: str, output_md: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv, low_memory=False)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        df.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Team Outlier Watchlist\n\nNo rows matched.\n")
        return df
    out = df.copy()
    out["expected_hit_3y"] = pd.to_numeric(out["expected_hit_3y"], errors="coerce")
    out["observed_hit"] = pd.to_numeric(out["observed_hit"], errors="coerce")
    out["near_misses"] = pd.to_numeric(out["near_misses"], errors="coerce").fillna(0)
    out["missed_correct"] = pd.to_numeric(out["missed_correct"], errors="coerce").fillna(0)
    out["rows"] = pd.to_numeric(out["rows"], errors="coerce").fillna(0)
    out["fixtures"] = pd.to_numeric(out["fixtures"], errors="coerce").fillna(0)
    out["outlier_gap"] = (out["observed_hit"] - out["expected_hit_3y"]).abs()
    out["miss_imbalance"] = (out["missed_correct"] - out["near_misses"]).abs() / out["rows"].clip(lower=1)
    out["outlier_score"] = out["outlier_gap"] + out["miss_imbalance"]
    out["watchlist_tier"] = out.apply(classify_watch, axis=1)
    out = out[out["watchlist_tier"] != "LOW_SIGNAL"].sort_values(["watchlist_tier", "outlier_score", "rows"], ascending=[True, False, False])
    out.to_csv(output_csv, index=False)

    lines = [
        "# Team Outlier Watchlist",
        "",
        "- `LIVE_RELAX_OUTLIER`: team-market cohort looks under-selected and keeps producing missed winners.",
        "- `LIVE_TIGHTEN_OUTLIER`: team-market cohort looks over-trusted and keeps producing avoidable misses.",
        "- `SMALL_SAMPLE_WATCH`: not decisive yet, but the cohort is worth tracking.",
        "",
    ]
    if out.empty:
        lines.append("No team outlier rows matched.")
    else:
        for tier, sub in out.groupby("watchlist_tier", sort=False):
            lines.append(f"## {tier}")
            for _, row in sub.head(25).iterrows():
                lines.append(
                    f"- {row['team_name']} | {row['league']} | {row['competition']} | {row['market']} | {row['review_family']} | posture={row['threshold_posture']} | rows={int(row['rows'])} | fixtures={int(row['fixtures'])} | expected={row['expected_hit_3y']:.3f} | observed={row['observed_hit']:.3f} | outlier_score={row['outlier_score']:.3f}"
                )
            lines.append("")
    Path(output_md).write_text("\n".join(lines) + "\n")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a team outlier watchlist from the threshold map.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build_watchlist(args.input_csv, args.output_csv, args.output_md)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")
