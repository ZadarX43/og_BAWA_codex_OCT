from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def export_shots_name_buckets(board_csv: str, output_csv: str, output_md: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    board = pd.read_csv(board_csv)
    shots = board[board["market"].eq("shots")].copy()
    if shots.empty:
        fixture_df = pd.DataFrame(
            columns=[
                "fixture_key",
                "league",
                "home_team_name",
                "away_team_name",
                "fixture_attacking_style_label",
                "fixture_attack_quality_score",
                "shot_name_count",
                "top_shot_names",
                "bucket_label",
            ]
        )
        summary_df = pd.DataFrame(columns=["bucket_label", "fixtures", "avg_attack_quality_score"])
    else:
        rows = []
        for fixture_key, group in shots.groupby("fixture_key", sort=False):
            ranked = group.sort_values("market_score", ascending=False).reset_index(drop=True)
            names = ranked["player_name"].astype(str).tolist()
            count = len(names)
            if count >= 4:
                bucket = "SHOTS_4PLUS"
            elif count >= 3:
                bucket = "SHOTS_3PLUS"
            elif count >= 2:
                bucket = "SHOTS_2PLUS"
            else:
                bucket = "SHOTS_1"
            rows.append(
                {
                    "fixture_key": fixture_key,
                    "league": ranked["league"].iloc[0],
                    "home_team_name": ranked["home_team_name"].iloc[0],
                    "away_team_name": ranked["away_team_name"].iloc[0],
                    "fixture_attacking_style_label": ranked["fixture_attacking_style_label"].iloc[0],
                    "fixture_attack_quality_score": float(ranked["fixture_attack_quality_score"].iloc[0]),
                    "shot_name_count": count,
                    "top_shot_names": " | ".join(names[:4]),
                    "bucket_label": bucket,
                }
            )
        fixture_df = pd.DataFrame(rows).sort_values(
            ["shot_name_count", "fixture_attack_quality_score", "fixture_key"],
            ascending=[False, False, True],
        )
        summary_df = (
            fixture_df.groupby("bucket_label", as_index=False)
            .agg(
                fixtures=("fixture_key", "count"),
                avg_attack_quality_score=("fixture_attack_quality_score", "mean"),
            )
            .sort_values(["fixtures", "avg_attack_quality_score"], ascending=[False, False])
        )

    out_csv = Path(output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fixture_df.to_csv(out_csv, index=False)

    out_md = Path(output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Shots Name Buckets",
        "",
        "## Summary",
    ]
    if summary_df.empty:
        lines.append("- no qualifying shots rows found")
    else:
        for row in summary_df.itertuples(index=False):
            lines.append(
                f"- `{row.bucket_label}`: fixtures={row.fixtures} | avg_attack_quality={row.avg_attack_quality_score:.3f}"
            )
    lines.extend(["", "## Fixture Buckets"])
    if fixture_df.empty:
        lines.append("- no fixtures")
    else:
        for row in fixture_df.itertuples(index=False):
            lines.append(
                f"- `{row.bucket_label}` | {row.fixture_key} | {row.home_team_name} vs {row.away_team_name} | "
                f"attack={row.fixture_attacking_style_label} | quality={row.fixture_attack_quality_score:.3f} | "
                f"names={row.shot_name_count} | {row.top_shot_names}"
            )
    out_md.write_text("\n".join(lines))
    return fixture_df, summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export per-fixture shots-name buckets from a combined attacking board.")
    parser.add_argument("--board-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixture_df, summary_df = export_shots_name_buckets(
        board_csv=args.board_csv,
        output_csv=args.output_csv,
        output_md=args.output_md,
    )
    print(f"WROTE: {args.output_csv}")
    print(f"fixtures: {len(fixture_df)}")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
