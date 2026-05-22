from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path("/Users/hughwade/Documents/Code/OG_master/BAWA PORTO")
WEEKLY_ROOT = REPO_ROOT / "predictions_output/walk_forward_team_intelligence_full_validation_3y_weekly_2026_04_22"


def find_rows(path: Path, home: str, away: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if {"home_team_name", "away_team_name"}.issubset(df.columns):
        return df[
            df["home_team_name"].astype(str).str.contains(home, case=False, na=False)
            & df["away_team_name"].astype(str).str.contains(away, case=False, na=False)
        ].copy()
    return pd.DataFrame()


def build(output_csv: str, output_md: str) -> pd.DataFrame:
    fixture_key = "2025-05-04_real_madrid_celta_vigo"
    window_id = "w107_2025_05_02_2025_05_06"
    home = "Real Madrid"
    away = "Celta Vigo"
    parts = window_id.split("_")
    rng = f"{parts[1]}-{parts[2]}-{parts[3]}_to_{parts[4]}-{parts[5]}-{parts[6]}"
    files = [
        ("source_allmarkets", WEEKLY_ROOT / window_id / "01_source" / f"BOOKIE_IMP20_ALLMARKETS_{rng}.csv"),
        ("deploy_raw", WEEKLY_ROOT / window_id / "02_deploy" / "DEPLOY_CANDIDATES_RAW.csv"),
        ("deploy_after_gates", WEEKLY_ROOT / window_id / "02_deploy" / "DEPLOY_CANDIDATES_AFTER_GATES.csv"),
        ("scored_raw", WEEKLY_ROOT / window_id / "03_scored" / f"DEPLOY_CANDIDATES_RAW_SCORED_{rng}.csv"),
        ("scored_combined", WEEKLY_ROOT / window_id / "03_scored" / f"DEPLOY_COMBINED_SCORED_{rng}.csv"),
    ]
    rows = []
    for stage, path in files:
        sub = find_rows(path, home, away)
        rows.append(
            {
                "fixture_key": fixture_key,
                "window_id": window_id,
                "stage": stage,
                "path": str(path),
                "row_count": int(len(sub)),
            }
        )
    out = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    lines = [
        "# Weekly Source-Universe Omission Trace",
        "",
        f"- Fixture: `{fixture_key}`",
        "- Checks whether the fixture ever entered the weekly source and scored chain.",
        "",
    ]
    for _, row in out.iterrows():
        lines.append(f"- `{row['stage']}` | rows=`{int(row['row_count'])}`")
        lines.append(f"  path: `{row['path']}`")
    Path(output_md).write_text("\n".join(lines) + "\n")
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a weekly source-universe omission trace for a hard-gap fixture.")
    ap.add_argument("--output-csv", default=str(REPO_ROOT / "reports/player_events/quality_audits/weekly_source_universe_omission_trace.csv"))
    ap.add_argument("--output-md", default=str(REPO_ROOT / "reports/player_events/quality_audits/weekly_source_universe_omission_trace.md"))
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build(args.output_csv, args.output_md)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")
