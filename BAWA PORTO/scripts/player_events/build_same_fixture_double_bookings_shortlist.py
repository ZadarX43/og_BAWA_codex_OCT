from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_shortlist(input_csv: str, output_csv: str, output_md: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv, low_memory=False)
    if df.empty:
        df.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Same-Fixture Double Bookings Shortlist\n\nNo rows matched.\n")
        return df

    fixture_counts = (
        df.groupby("fixture_key", as_index=False)
        .agg(
            booking_names=("player_name", "count"),
            combined_booking_score=("market_score", "sum"),
            avg_quality=("fixture_quality_score", "mean"),
            avg_pressure=("formation_pressure_score", "mean"),
        )
    )
    shortlist_fixtures = fixture_counts[fixture_counts["booking_names"].ge(2)].sort_values(
        ["combined_booking_score", "avg_quality", "avg_pressure"], ascending=[False, False, False]
    )
    out = df[df["fixture_key"].isin(shortlist_fixtures["fixture_key"])].copy()
    out = out.merge(shortlist_fixtures, on="fixture_key", how="left")
    out = out.sort_values(
        ["combined_booking_score", "fixture_key", "market_score"], ascending=[False, True, False]
    ).reset_index(drop=True)
    out.to_csv(output_csv, index=False)

    lines = ["# Same-Fixture Double Bookings Shortlist", "", f"- fixtures: {out['fixture_key'].nunique() if not out.empty else 0} | rows: {len(out)}", ""]
    for fixture_key, sub in out.groupby("fixture_key", sort=False):
        first = sub.iloc[0]
        lines.append(f"## {fixture_key}")
        lines.append(
            f"- {first['home_team_name']} vs {first['away_team_name']} | combined_score={first['combined_booking_score']:.1f} | quality={first['avg_quality']:.3f}"
        )
        for _, row in sub.iterrows():
            lines.append(f"- {row['player_name']} ({row['team_name']}) | family={row['source_family']} | score={row['market_score']:.1f}")
        lines.append("")
    Path(output_md).write_text("\n".join(lines) + "\n")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a same-fixture double bookings shortlist from the super-elite bookings board.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build_shortlist(args.input_csv, args.output_csv, args.output_md)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)} | fixtures: {out['fixture_key'].nunique() if not out.empty else 0}")
