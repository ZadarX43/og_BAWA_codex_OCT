from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CONTACT_MARKETS = {"fouls_committed", "tackles"}


def build_shortlist(input_csv: str, output_csv: str, output_md: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv, low_memory=False)
    if df.empty:
        df.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Same-Fixture Contact + Bookings Doubles\n\nNo rows matched.\n")
        return df

    fixture_meta = (
        df.groupby("fixture_key", as_index=False)
        .agg(
            contact_rows=("market", lambda s: int(pd.Series(s).isin(CONTACT_MARKETS).sum())),
            booking_rows=("market", lambda s: int((pd.Series(s) == "yellow_cards").sum())),
            combined_review_score=("review_score", "sum"),
            avg_quality=("fixture_quality_score", "mean"),
        )
    )
    fixture_meta = fixture_meta[
        fixture_meta["contact_rows"].ge(1) & fixture_meta["booking_rows"].ge(1)
    ].sort_values(["combined_review_score", "avg_quality"], ascending=[False, False])
    out = df[df["fixture_key"].isin(fixture_meta["fixture_key"])].copy()
    out = out.merge(fixture_meta, on="fixture_key", how="left", suffixes=("", "_fixture"))
    score_col = "combined_review_score"
    if score_col not in out.columns and "combined_review_score_fixture" in out.columns:
        score_col = "combined_review_score_fixture"
    out = out.sort_values([score_col, "fixture_key", "review_score"], ascending=[False, True, False]).reset_index(drop=True)
    out.to_csv(output_csv, index=False)

    lines = ["# Same-Fixture Contact + Bookings Doubles", "", f"- fixtures: {out['fixture_key'].nunique() if not out.empty else 0} | rows: {len(out)}", ""]
    for fixture_key, sub in out.groupby("fixture_key", sort=False):
        first = sub.iloc[0]
        lines.append(f"## {fixture_key}")
        lines.append(
            f"- {first['home_team_name']} vs {first['away_team_name']} | contact_rows={int(first['contact_rows'])} | booking_rows={int(first['booking_rows'])} | combined_score={first[score_col]:.1f}"
        )
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['market']}: {row['player_name']} ({row['team_name']}) | family={row['review_family']} | score={row['review_score']:.1f}"
            )
        lines.append("")
    Path(output_md).write_text("\n".join(lines) + "\n")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a same-fixture contact + bookings doubles shortlist.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build_shortlist(args.input_csv, args.output_csv, args.output_md)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)} | fixtures: {out['fixture_key'].nunique() if not out.empty else 0}")
