from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def build(input_csv: Path, output_csv: Path, output_md: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv, low_memory=False)
    targets = df[df["diagnosis"] == "ARCHIVE_MISSING"].copy()
    if not targets.empty:
        targets["target_key"] = targets["league_tag"].astype(str) + "__" + targets["season"].astype(int).astype(str)
        targets["recommended_action"] = "API_FOOTBALL_BACKFILL"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    targets.to_csv(output_csv, index=False)

    lines = [
        "# Player Events Historical Backfill Targets",
        "",
        "- Explicit target list for missing local league-season archives blocking true full-coverage Track A.",
        "",
        "## Targets",
    ]
    if targets.empty:
        lines.append("- No archive-missing targets found.")
    else:
        for _, row in targets.iterrows():
            lines.append(
                f"- {row['league_tag']} | season={int(row['season'])} | action={row['recommended_action']}"
            )
    output_md.write_text("\n".join(lines) + "\n")
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an explicit backfill target list from the player-events historical gap diagnosis.")
    parser.add_argument(
        "--input-csv",
        default=str(REPO_ROOT / "reports" / "player_events" / "quality_audits" / "player_events_historical_gap_diagnosis.csv"),
    )
    parser.add_argument(
        "--output-csv",
        default=str(REPO_ROOT / "reports" / "player_events" / "quality_audits" / "player_events_historical_backfill_targets.csv"),
    )
    parser.add_argument(
        "--output-md",
        default=str(REPO_ROOT / "reports" / "player_events" / "quality_audits" / "player_events_historical_backfill_targets.md"),
    )
    args = parser.parse_args()
    out = build(Path(args.input_csv), Path(args.output_csv), Path(args.output_md))
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")


if __name__ == "__main__":
    main()
