from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_sheet(input_csv: str, output_csv: str, output_md: str, max_fixtures: int = 8) -> pd.DataFrame:
    df = pd.read_csv(input_csv, low_memory=False)
    if df.empty:
        df.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Bookings Super-Elite Weekend Sheet\n\nNo rows matched.\n")
        return df

    fixture_rank = (
        df.groupby(["fixture_key", "home_team_name", "away_team_name"], as_index=False)
        .agg(
            rows=("player_name", "count"),
            fixture_super_score=("fixture_super_score", "max"),
            avg_quality=("fixture_quality_score", "mean"),
            top_score=("market_score", "max"),
        )
        .sort_values(["fixture_super_score", "rows", "top_score"], ascending=[False, False, False])
        .head(max_fixtures)
    )
    out = df[df["fixture_key"].isin(fixture_rank["fixture_key"])].copy()
    out = out.merge(fixture_rank[["fixture_key", "fixture_super_score"]], on="fixture_key", how="left", suffixes=("", "_rank"))
    out = out.sort_values(["fixture_super_score_rank", "fixture_key", "super_elite_priority"], ascending=[False, True, False]).reset_index(drop=True)
    out.to_csv(output_csv, index=False)

    lines = ["# Bookings Super-Elite Weekend Sheet", "", f"- fixtures: {out['fixture_key'].nunique()} | rows: {len(out)}", ""]
    for fixture_key, sub in out.groupby("fixture_key", sort=False):
        first = sub.iloc[0]
        lines.append(f"## {fixture_key}")
        lines.append(
            f"- {first['home_team_name']} vs {first['away_team_name']} | fixture score={first['fixture_super_score_rank']:.1f}"
        )
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['player_name']} ({row['team_name']}) | family={row['source_family']} | tier={row['portable_tier']} | score={row['market_score']:.1f}"
            )
        lines.append("")
    Path(output_md).write_text("\n".join(lines) + "\n")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a weekend sheet from the portable bookings super-elite board.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--max-fixtures", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build_sheet(args.input_csv, args.output_csv, args.output_md, args.max_fixtures)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)} | fixtures: {out['fixture_key'].nunique() if not out.empty else 0}")
