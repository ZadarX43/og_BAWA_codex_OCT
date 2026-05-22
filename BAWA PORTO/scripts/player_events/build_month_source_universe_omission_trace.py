from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path("/Users/hughwade/Documents/Code/OG_master/BAWA PORTO")
REGEN_ROOT = REPO_ROOT / "reports/player_events/quality_audits/frozen_month_regen_archive__2026_05_03"

FIXTURES = [
    ("2024-06-20_san_jose_earthquakes_portland_timbers", "2024-06", "San Jose Earthquakes", "Portland Timbers"),
    ("2024-07-18_vancouver_whitecaps_sporting_kansas_city", "2024-07", "Vancouver Whitecaps", "Sporting Kansas City"),
]


def find_rows(path: Path, home: str, away: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if {"home_team_name", "away_team_name"}.issubset(df.columns):
        return df[
            df["home_team_name"].astype(str).str.contains(home, case=False, na=False)
            & df["away_team_name"].astype(str).str.contains(away, case=False, na=False)
        ].copy()
    if {"home", "away"}.issubset(df.columns):
        return df[
            df["home"].astype(str).str.contains(home, case=False, na=False)
            & df["away"].astype(str).str.contains(away, case=False, na=False)
        ].copy()
    if "fixture_key" in df.columns:
        return df[
            df["fixture_key"].astype(str).str.contains(home.lower().replace(" ", "_"), na=False)
            & df["fixture_key"].astype(str).str.contains(away.lower().replace(" ", "_"), na=False)
        ].copy()
    return pd.DataFrame()


def build(output_csv: str, output_md: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fixture_key, month_tag, home, away in FIXTURES:
        month_dir = REGEN_ROOT / month_tag
        files = [
            ("raw_predictions", month_dir / f"raw_predictions_{month_tag}.csv"),
            ("backtest", month_dir / f"backtest_{month_tag}.csv"),
            ("backtest_unscored", month_dir / f"backtest_unscored_{month_tag}.csv"),
            ("frozen_gated", month_dir / f"frozen_gated_{month_tag}.csv"),
        ]
        for stage, path in files:
            sub = find_rows(path, home, away)
            rows.append(
                {
                    "fixture_key": fixture_key,
                    "month_tag": month_tag,
                    "stage": stage,
                    "path": str(path),
                    "row_count": int(len(sub)),
                    "sample_match_date": str(sub.iloc[0]["match_date"])[:10] if len(sub) and "match_date" in sub.columns else "",
                }
            )
    out = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    lines = ["# Month Source-Universe Omission Trace", "", "- Checks whether the two remaining month-rebuild omissions ever enter the regenerated month source universe.", ""]
    for fixture_key, sub in out.groupby("fixture_key", sort=False):
        lines.append(f"## {fixture_key}")
        for _, row in sub.iterrows():
            lines.append(f"- `{row['stage']}` | rows=`{int(row['row_count'])}` | date=`{row['sample_match_date'] or 'n/a'}`")
        lines.append("")
    Path(output_md).write_text("\n".join(lines) + "\n")
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Trace source-universe omission for rebuilt month fixtures.")
    ap.add_argument("--output-csv", default=str(REPO_ROOT / "reports/player_events/quality_audits/month_source_universe_omission_trace.csv"))
    ap.add_argument("--output-md", default=str(REPO_ROOT / "reports/player_events/quality_audits/month_source_universe_omission_trace.md"))
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build(args.output_csv, args.output_md)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")
