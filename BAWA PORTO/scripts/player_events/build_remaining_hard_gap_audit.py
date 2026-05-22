from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def classify_gap(row: pd.Series) -> str:
    fixture_key = str(row.get("fixture_key", ""))
    if fixture_key.startswith("2024-06") or fixture_key.startswith("2024-07"):
        return "MONTH_REBUILT_BUT_FIXTURE_STILL_ABSENT"
    if "Villarreal" in str(row.get("fixture_label", "")) or "Real Madrid" in str(row.get("fixture_label", "")):
        return "NO_RANKED_HISTORY_AFTER_WEEKLY_AND_FROZEN_RECOVERY"
    return "MANUAL_RECHECK_REQUIRED"


def build(tracker_csv: str, route_csv: str, nonweekly_csv: str, output_csv: str, output_md: str) -> pd.DataFrame:
    tracker = pd.read_csv(tracker_csv, low_memory=False)
    route = pd.read_csv(route_csv, low_memory=False)
    nonweekly = pd.read_csv(nonweekly_csv, low_memory=False)

    pending = tracker[tracker["tracker_status"].eq("BACKFILL_PENDING")].copy()
    route_small = route[
        ["fixture_key", "weekly_window_status", "owning_window_id", "window_date_from", "window_date_to", "rerun_hint"]
    ]
    nonweekly_small = nonweekly[
        ["fixture_key", "recommended_regeneration_path", "regeneration_status", "existing_frozen_month_archives"]
    ]

    out = pending.merge(route_small, on="fixture_key", how="left").merge(nonweekly_small, on="fixture_key", how="left")
    out["hard_gap_reason"] = out.apply(classify_gap, axis=1)
    out["next_recovery_step"] = out.apply(
        lambda row: (
            "Trace the original ranked-board source or archived export for this weekly-covered fixture."
            if str(row.get("weekly_window_status", "")) == "can_regenerate_from_weekly_windows"
            else "Investigate fixture-level omission inside the rebuilt frozen month archive."
        ),
        axis=1,
    )
    out = out.sort_values(["backfill_priority", "fixture_key"])

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    lines = [
        "# Remaining Hard-Gap Audit",
        "",
        "- Focused only on fixtures still `BACKFILL_PENDING` after weekly and frozen-month recovery work.",
        "",
        "## Summary",
    ]
    summary = out.groupby("hard_gap_reason", dropna=False).agg(rows=("fixture_key", "size")).reset_index()
    for _, row in summary.iterrows():
        lines.append(f"- {row['hard_gap_reason']} | rows={int(row['rows'])}")
    lines.append("")
    lines.append("## Fixtures")
    for _, row in out.iterrows():
        lines.append(
            f"- `{row['fixture_key']}` | priority=`{row['backfill_priority']}` | focus=`{row['goal_market_focus']}` | reason=`{row['hard_gap_reason']}`"
        )
        lines.append(f"  next: {row['next_recovery_step']}")
    Path(output_md).write_text("\n".join(lines) + "\n")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an audit for the remaining hard goal-market coverage gaps.")
    parser.add_argument("--tracker-csv", required=True)
    parser.add_argument("--route-csv", required=True)
    parser.add_argument("--nonweekly-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build(args.tracker_csv, args.route_csv, args.nonweekly_csv, args.output_csv, args.output_md)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")
