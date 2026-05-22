from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


REPO_ROOT = Path("/Users/hughwade/Documents/Code/OG_master/BAWA PORTO")
FROZEN_ROOTS = {
    "frozen_accuracy": REPO_ROOT / "walkforward_frozen_accuracy",
    "frozen_valueev_balanced": REPO_ROOT / "walkforward_frozen_valueev_balanced",
    "frozen_valueev_aggressive": REPO_ROOT / "walkforward_frozen_valueev_aggressive",
}


def month_tag_from_fixture_key(fixture_key: str) -> str:
    date_token = str(fixture_key).split("_", 1)[0]
    return datetime.strptime(date_token, "%Y-%m-%d").strftime("%Y-%m")


def classify(row: pd.Series) -> dict[str, object]:
    fixture_key = str(row.get("fixture_key", "")).strip()
    month_tag = month_tag_from_fixture_key(fixture_key)
    existing_archives: list[str] = []
    for family_name, root in FROZEN_ROOTS.items():
        month_dir = root / month_tag
        if month_dir.exists():
            existing_archives.append(str(month_dir))

    if existing_archives:
        regeneration_path = "broader_full_estate_frozen_month_archive"
        regeneration_status = "HAS_MONTH_ARCHIVE__HARVEST_OR_TARGETED_MONTH_REFRESH"
        recommendation = (
            "Use run_frozen_walkforward.py as the canonical nonweekly path; a matching frozen month archive exists, "
            "but the fixture did not recover via the weekly window estate."
        )
    else:
        regeneration_path = "broader_full_estate_month_rebuild"
        regeneration_status = "NO_MONTH_ARCHIVE__RUN_FROZEN_WALKFORWARD"
        recommendation = (
            "Use run_frozen_walkforward.py for a month rebuild because no matching frozen month archive exists locally."
        )

    return {
        "fixture_key": fixture_key,
        "month_tag": month_tag,
        "weekly_window_status": row.get("weekly_window_status", ""),
        "daily_goal_market_family_candidate": 0,
        "daily_goal_market_family_note": (
            "No general daily goal-market regeneration family was found in the repo. "
            "The closest daily-window code is backtest_uefa_ab.py, which looks experimental and UEFA-specific."
        ),
        "recommended_regeneration_path": regeneration_path,
        "recommended_runner": str(REPO_ROOT / "run_frozen_walkforward.py"),
        "existing_frozen_month_archives": " | ".join(existing_archives),
        "regeneration_status": regeneration_status,
        "recommended_markets": "ftr,ou25,btts,tg15,tg25",
        "recommendation": recommendation,
        "suggested_command_hint": (
            f"python3 {REPO_ROOT / 'run_frozen_walkforward.py'} --start-month {month_tag} --end-month {month_tag} "
            "--markets ftr,ou25,btts,tg15,tg25 --strict"
        ),
    }


def build(input_csv: str, output_csv: str, output_md: str) -> pd.DataFrame:
    src = pd.read_csv(input_csv, low_memory=False)
    src = src[src["weekly_window_status"].eq("needs_nonweekly_goal_market_regen")].copy()
    if src.empty:
        out = pd.DataFrame(
            columns=[
                "fixture_key",
                "month_tag",
                "weekly_window_status",
                "daily_goal_market_family_candidate",
                "daily_goal_market_family_note",
                "recommended_regeneration_path",
                "recommended_runner",
                "existing_frozen_month_archives",
                "regeneration_status",
                "recommended_markets",
                "recommendation",
                "suggested_command_hint",
            ]
        )
    else:
        out = pd.DataFrame([classify(row) for _, row in src.iterrows()])
        out = out.sort_values(["month_tag", "fixture_key"]).reset_index(drop=True)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    lines = [
        "# Nonweekly Goal-Market Regeneration Board",
        "",
        "- Built for fixtures that sit outside the weekly 5-day window manifest.",
        "- This separates weekly manifest gaps from the broader month-based frozen walkforward path.",
        "",
        "## Summary",
    ]
    if out.empty:
        lines.append("- No nonweekly fixtures were found.")
    else:
        status_summary = out.groupby("regeneration_status", dropna=False).agg(rows=("fixture_key", "size")).reset_index()
        for _, row in status_summary.iterrows():
            lines.append(f"- {row['regeneration_status']} | rows={int(row['rows'])}")
        lines.append("")
        lines.append("## Fixtures")
        for _, row in out.iterrows():
            lines.append(f"- `{row['fixture_key']}` | month=`{row['month_tag']}` | path=`{row['recommended_regeneration_path']}`")
            if row["existing_frozen_month_archives"]:
                lines.append(f"  archive: `{row['existing_frozen_month_archives']}`")
            lines.append(f"  note: {row['recommendation']}")
    Path(output_md).write_text("\n".join(lines) + "\n")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a nonweekly goal-market regeneration board.")
    parser.add_argument(
        "--input-csv",
        default=str(REPO_ROOT / "reports/player_events/quality_audits/goal_market_regeneration_route_board.csv"),
    )
    parser.add_argument(
        "--output-csv",
        default=str(REPO_ROOT / "reports/player_events/quality_audits/nonweekly_goal_market_regeneration_board.csv"),
    )
    parser.add_argument(
        "--output-md",
        default=str(REPO_ROOT / "reports/player_events/quality_audits/nonweekly_goal_market_regeneration_board.md"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args.input_csv, args.output_csv, args.output_md)
