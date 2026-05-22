from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_map(input_csv: str, output_csv: str, output_md: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv, low_memory=False)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        df.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Team x League x Competition Threshold Map\n\nNo rows matched.\n")
        return df

    summary = (
        df.groupby(["team_name", "league", "competition", "market", "review_family"], dropna=False)
        .agg(
            rows=("fixture_key", "size"),
            fixtures=("fixture_key", pd.Series.nunique),
            expected_hit_3y=("expected_hit_rate_3y", lambda s: pd.to_numeric(s, errors="coerce").mean()),
            observed_hit=("observed_success_flag", lambda s: pd.to_numeric(s, errors="coerce").mean()),
            avg_score=("score", lambda s: pd.to_numeric(s, errors="coerce").mean()),
            near_misses=("near_miss_flag", "sum"),
            missed_correct=("missed_correct_flag", "sum"),
        )
        .reset_index()
        .sort_values(["expected_hit_3y", "observed_hit", "rows"], ascending=[False, False, False])
    )
    summary["threshold_posture"] = summary.apply(
        lambda row: (
            "RELAX"
            if row["missed_correct"] > row["near_misses"] and row["expected_hit_3y"] >= 0.70
            else "TIGHTEN"
            if row["near_misses"] > row["missed_correct"] and row["expected_hit_3y"] >= 0.70
            else "WATCH"
        ),
        axis=1,
    )
    summary.to_csv(output_csv, index=False)

    lines = [
        "# Team x League x Competition Threshold Map",
        "",
        "- `RELAX`: this team-market cohort may be too constrained by the current gate.",
        "- `TIGHTEN`: this team-market cohort is producing more avoidable misses than missed winners.",
        "- `WATCH`: no strong tuning direction yet.",
        "",
    ]
    for posture, sub in summary.groupby("threshold_posture", sort=False):
        lines.append(f"## {posture}")
        for _, row in sub.head(20).iterrows():
            lines.append(
                f"- {row['team_name']} | {row['league']} | {row['competition']} | {row['market']} | {row['review_family']} | rows={int(row['rows'])} | fixtures={int(row['fixtures'])} | expected_hit_3y={row['expected_hit_3y']:.3f} | observed_hit={row['observed_hit']:.3f} | near_misses={int(row['near_misses'])} | missed_correct={int(row['missed_correct'])}"
            )
        lines.append("")

    Path(output_md).write_text("\n".join(lines) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a team x league x competition threshold map from the rolling player-market runner.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build_map(args.input_csv, args.output_csv, args.output_md)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")
