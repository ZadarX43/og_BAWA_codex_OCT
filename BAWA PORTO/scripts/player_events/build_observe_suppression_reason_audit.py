from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path("/Users/hughwade/Documents/Code/OG_master/BAWA PORTO")
WEEKLY_ROOT = REPO_ROOT / "predictions_output/walk_forward_team_intelligence_full_validation_3y_weekly_2026_04_22"
FIXTURES = [
    ("2025-02-08_las_palmas_villarreal", "w095_2025_02_07_2025_02_11", "Las Palmas", "Villarreal"),
    ("2025-02-22_rayo_vallecano_villarreal", "w097_2025_02_21_2025_02_25", "Rayo Vallecano", "Villarreal"),
]


def find_fixture_rows(path: Path, home: str, away: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if {"home_team_name", "away_team_name"}.issubset(df.columns):
        return df[
            df["home_team_name"].astype(str).str.contains(home, case=False, na=False)
            & df["away_team_name"].astype(str).str.contains(away, case=False, na=False)
        ].copy()
    return pd.DataFrame()


def reason_snapshot(row: pd.Series) -> dict[str, object]:
    return {
        "market": row.get("market"),
        "selection": row.get("selection"),
        "deploy_tier": row.get("deploy_tier"),
        "standard_reporting_bucket": row.get("standard_reporting_bucket"),
        "context_reason_codes": row.get("context_reason_codes"),
        "reason_codes": row.get("reason_codes"),
        "deterministic_veto_reason": row.get("deterministic_veto_reason"),
        "learned_veto_reason": row.get("learned_veto_reason"),
        "deploy_veto_reason": row.get("deploy_veto_reason"),
        "deterministic_warn_reason": row.get("deterministic_warn_reason"),
        "btts_yes_policy_reason": row.get("btts_yes_policy_reason"),
    }


def build(output_csv: str, output_md: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fixture_key, window_id, home, away in FIXTURES:
        parts = window_id.split("_")
        rng = f"{parts[1]}-{parts[2]}-{parts[3]}_to_{parts[4]}-{parts[5]}-{parts[6]}"
        observe_csv = WEEKLY_ROOT / window_id / "02_deploy" / f"BOOKIE_IMP20_ALLMARKETS_{rng}__DEPLOY_TIER_OBSERVE__PRESET_V1__FTR_accuracy.csv"
        sub = find_fixture_rows(observe_csv, home, away)
        for _, row in sub.iterrows():
            out = {"fixture_key": fixture_key, "window_id": window_id}
            out.update(reason_snapshot(row))
            rows.append(out)
    out = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    lines = [
        "# Observe Suppression Reason Audit",
        "",
        "- Focused on the two weekly-covered La Liga fixtures that survived only into `OBSERVE`.",
        "",
    ]
    for fixture_key, sub in out.groupby("fixture_key", sort=False):
        lines.append(f"## {fixture_key}")
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['market']} | pick={row['selection']} | bucket={row.get('standard_reporting_bucket','')} | context={row.get('context_reason_codes','') or 'none'}"
            )
            if pd.notna(row.get("deterministic_warn_reason")):
                lines.append(f"  warn: {row.get('deterministic_warn_reason')}")
            if pd.notna(row.get("btts_yes_policy_reason")):
                lines.append(f"  policy: {row.get('btts_yes_policy_reason')}")
        lines.append("")
    Path(output_md).write_text("\n".join(lines) + "\n")
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a tiny OBSERVE suppression reason audit.")
    ap.add_argument("--output-csv", default=str(REPO_ROOT / "reports/player_events/quality_audits/observe_suppression_reason_audit.csv"))
    ap.add_argument("--output-md", default=str(REPO_ROOT / "reports/player_events/quality_audits/observe_suppression_reason_audit.md"))
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build(args.output_csv, args.output_md)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")
