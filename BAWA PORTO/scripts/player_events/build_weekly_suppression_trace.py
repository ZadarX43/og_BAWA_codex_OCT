from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path("/Users/hughwade/Documents/Code/OG_master/BAWA PORTO")
WEEKLY_ROOT = REPO_ROOT / "predictions_output/walk_forward_team_intelligence_full_validation_3y_weekly_2026_04_22"

FIXTURES = [
    ("2025-02-08_las_palmas_villarreal", "w095_2025_02_07_2025_02_11", "Las Palmas", "Villarreal"),
    ("2025-02-22_rayo_vallecano_villarreal", "w097_2025_02_21_2025_02_25", "Rayo Vallecano", "Villarreal"),
    ("2025-05-04_real_madrid_celta_vigo", "w107_2025_05_02_2025_05_06", "Real Madrid", "Celta Vigo"),
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


def extract_reason_sample(df: pd.DataFrame) -> str:
    for col in ["deploy_veto_reason", "deterministic_veto_reason", "context_reason_codes", "reason_codes", "standard_reporting_bucket"]:
        if col in df.columns:
            vals = [str(v).strip() for v in df[col].dropna().tolist() if str(v).strip()]
            if vals:
                return " | ".join(vals[:3])
    return ""


def build(output_csv: str, output_md: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fixture_key, window_id, home, away in FIXTURES:
        parts = window_id.split("_")
        rng = f"{parts[1]}-{parts[2]}-{parts[3]}_to_{parts[4]}-{parts[5]}-{parts[6]}"
        files = [
            ("source_allmarkets", WEEKLY_ROOT / window_id / "01_source" / f"BOOKIE_IMP20_ALLMARKETS_{rng}.csv"),
            ("deploy_raw", WEEKLY_ROOT / window_id / "02_deploy" / "DEPLOY_CANDIDATES_RAW.csv"),
            ("deploy_after_gates", WEEKLY_ROOT / window_id / "02_deploy" / "DEPLOY_CANDIDATES_AFTER_GATES.csv"),
            ("tier_observe", WEEKLY_ROOT / window_id / "02_deploy" / f"BOOKIE_IMP20_ALLMARKETS_{rng}__DEPLOY_TIER_OBSERVE__PRESET_V1__FTR_accuracy.csv"),
            ("tier_standard", WEEKLY_ROOT / window_id / "02_deploy" / f"BOOKIE_IMP20_ALLMARKETS_{rng}__DEPLOY_TIER_STANDARD__PRESET_V1__FTR_accuracy.csv"),
            ("tier_elite", WEEKLY_ROOT / window_id / "02_deploy" / f"BOOKIE_IMP20_ALLMARKETS_{rng}__DEPLOY_TIER_ELITE__PRESET_V1__FTR_accuracy.csv"),
            ("scored_raw", WEEKLY_ROOT / window_id / "03_scored" / f"DEPLOY_CANDIDATES_RAW_SCORED_{rng}.csv"),
            ("scored_after_gates", WEEKLY_ROOT / window_id / "03_scored" / f"DEPLOY_CANDIDATES_AFTER_GATES_SCORED_{rng}.csv"),
            ("scored_combined", WEEKLY_ROOT / window_id / "03_scored" / f"DEPLOY_COMBINED_SCORED_{rng}.csv"),
        ]
        for stage, path in files:
            sub = find_rows(path, home, away)
            rows.append(
                {
                    "fixture_key": fixture_key,
                    "window_id": window_id,
                    "stage": stage,
                    "path": str(path),
                    "row_count": int(len(sub)),
                    "reason_sample": extract_reason_sample(sub),
                }
            )

    out = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    lines = ["# Weekly Suppression Trace", "", "- Traces the three remaining weekly-covered La Liga gaps through source, deploy, gated, and scored stages.", ""]
    for fixture_key, sub in out.groupby("fixture_key", sort=False):
        lines.append(f"## {fixture_key}")
        for _, row in sub.iterrows():
            lines.append(f"- `{row['stage']}` | rows=`{int(row['row_count'])}`")
            if row["reason_sample"]:
                lines.append(f"  sample: {row['reason_sample']}")
        lines.append("")
    Path(output_md).write_text("\n".join(lines) + "\n")
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Trace weekly suppression for remaining hard-gap fixtures.")
    ap.add_argument("--output-csv", default=str(REPO_ROOT / "reports/player_events/quality_audits/weekly_suppression_trace.csv"))
    ap.add_argument("--output-md", default=str(REPO_ROOT / "reports/player_events/quality_audits/weekly_suppression_trace.md"))
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build(args.output_csv, args.output_md)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")
