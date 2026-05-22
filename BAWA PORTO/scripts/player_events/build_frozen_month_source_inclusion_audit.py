from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path("/Users/hughwade/Documents/Code/OG_master/BAWA PORTO")
REGEN_ROOT = REPO_ROOT / "reports/player_events/quality_audits/frozen_month_regen_archive__2026_05_03"


def build(output_csv: str, output_md: str) -> pd.DataFrame:
    rows = []
    for month in ["2024-06", "2024-07"]:
        path = REGEN_ROOT / month / f"raw_predictions_{month}.csv"
        df = pd.read_csv(path, usecols=["league", "match_date", "home_team_name", "away_team_name", "market"], low_memory=False)
        mls = df[df["league"].astype(str).str.contains("MLS", case=False, na=False)].copy()
        unique_fixtures = (
            mls[["match_date", "home_team_name", "away_team_name"]]
            .drop_duplicates()
            .sort_values(["match_date", "home_team_name", "away_team_name"])
        )
        rows.append(
            {
                "month_tag": month,
                "source_csv": str(path),
                "total_rows": int(len(df)),
                "mls_rows": int(len(mls)),
                "mls_unique_fixtures": int(len(unique_fixtures)),
                "mls_fixture_examples": " | ".join(
                    f"{r.match_date}:{r.home_team_name} vs {r.away_team_name}"
                    for r in unique_fixtures.head(5).itertuples(index=False)
                ),
            }
        )
    out = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    lines = [
        "# Frozen Month Source Inclusion Audit",
        "",
        "- Checks whether June 2024 MLS and July 2024 MLS are represented at all in the regenerated month source universe.",
        "",
    ]
    for _, row in out.iterrows():
        lines.append(
            f"- `{row['month_tag']}` | total_rows=`{int(row['total_rows'])}` | mls_rows=`{int(row['mls_rows'])}` | mls_unique_fixtures=`{int(row['mls_unique_fixtures'])}`"
        )
        lines.append(f"  examples: {row['mls_fixture_examples']}")
    Path(output_md).write_text("\n".join(lines) + "\n")
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a frozen month source inclusion audit for MLS months.")
    ap.add_argument("--output-csv", default=str(REPO_ROOT / "reports/player_events/quality_audits/frozen_month_source_inclusion_audit.csv"))
    ap.add_argument("--output-md", default=str(REPO_ROOT / "reports/player_events/quality_audits/frozen_month_source_inclusion_audit.md"))
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build(args.output_csv, args.output_md)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")
