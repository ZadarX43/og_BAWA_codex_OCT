from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_leaderboard(family_summary_csv: str, family_market_csv: str, master_board_csv: str, output_md: str) -> None:
    family_summary = pd.read_csv(family_summary_csv)
    family_market = pd.read_csv(family_market_csv)
    master_board = pd.read_csv(master_board_csv, low_memory=False)

    lines = ["# Specialist Family Leaderboard", ""]
    if family_summary.empty:
        lines.append("No family summary rows matched.")
        Path(output_md).write_text("\n".join(lines) + "\n")
        return

    ranked = family_summary.sort_values(["row_hit_rate", "rows", "p1_rows"], ascending=[False, False, False]).reset_index(drop=True)
    for idx, row in ranked.iterrows():
        family = row["source_family"]
        market_slice = family_market[family_market["source_family"].eq(family)].sort_values(["hit_rate", "rows"], ascending=[False, False])
        strongest_markets = ", ".join(
            f"{m.market} ({m.hit_rate:.3f})" for m in market_slice.head(2).itertuples(index=False)
        )
        example_slice = master_board[master_board["source_family_tag"].astype(str).str.contains(family, regex=False, na=False)].copy()
        if not example_slice.empty:
            best_fixture = (
                example_slice.groupby(["fixture_key", "home_team_name", "away_team_name"], as_index=False)
                .agg(
                    top_score=("market_score", "max"),
                    rows=("market", "count"),
                    top_bucket=("priority_bucket", "min"),
                )
                .sort_values(["top_score", "rows"], ascending=[False, False])
                .iloc[0]
            )
            best_fixture_text = f"{best_fixture['fixture_key']} ({best_fixture['home_team_name']} vs {best_fixture['away_team_name']})"
        else:
            best_fixture_text = "None"

        lines.extend(
            [
                f"## {idx + 1}. {family}",
                f"- sample size: {int(row['rows'])} rows across {int(row['fixtures'])} fixtures",
                f"- hit rate: {row['row_hit_rate']:.3f}",
                f"- strongest markets: {strongest_markets or 'None'}",
                f"- best example fixture: {best_fixture_text}",
                "",
            ]
        )

    Path(output_md).write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a markdown leaderboard for specialist formation families.")
    parser.add_argument("--family-summary-csv", required=True)
    parser.add_argument("--family-market-csv", required=True)
    parser.add_argument("--master-board-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_leaderboard(args.family_summary_csv, args.family_market_csv, args.master_board_csv, args.output_md)
    print(f"WROTE: {args.output_md}")
