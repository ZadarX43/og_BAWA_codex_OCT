from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build(
    surprise_join_csv: str,
    overlap_csv: str,
    tracker_csv: str,
    output_csv: str,
    output_md: str,
) -> pd.DataFrame:
    surprise = pd.read_csv(surprise_join_csv, low_memory=False)
    overlap = pd.read_csv(overlap_csv, low_memory=False)
    tracker = pd.read_csv(tracker_csv, low_memory=False)

    join_summary = (
        surprise.groupby("fixture_key", dropna=False)
        .agg(
            joined_rows=("fixture_key", "size"),
            joined_markets=("market", lambda s: "|".join(sorted({str(v) for v in s if str(v).strip()}))),
            joined_hits=("hit_flag", "sum"),
        )
        .reset_index()
    )
    overlap_small = overlap[
        ["fixture_key", "status", "has_ranked_history", "ranked_markets_found", "ranked_rows"]
    ].rename(
        columns={
            "status": "latest_overlap_status",
            "has_ranked_history": "latest_has_ranked_history",
        }
    )
    tracker_small = tracker[
        ["fixture_key", "tracker_status", "backfill_priority", "goal_market_focus", "next_action"]
    ]

    out = tracker_small.merge(overlap_small, on="fixture_key", how="left").merge(join_summary, on="fixture_key", how="left")
    out["joined_rows"] = pd.to_numeric(out["joined_rows"], errors="coerce").fillna(0).astype(int)
    out["joined_hits"] = pd.to_numeric(out["joined_hits"], errors="coerce").fillna(0).astype(int)
    out["join_fix_result"] = out.apply(
        lambda row: (
            "JOIN_FIXED"
            if str(row.get("tracker_status", "")) == "RECOVERED_IN_JOIN"
            else "RECOVERED_NOT_JOINED"
            if int(pd.to_numeric(row.get("latest_has_ranked_history"), errors="coerce") or 0) == 1
            else "HARD_GAP_REMAINS"
        ),
        axis=1,
    )
    out = out.sort_values(
        ["join_fix_result", "backfill_priority", "fixture_key"],
        ascending=[True, True, True],
    )

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    lines = [
        "# Recovered Overlap Join Fixer",
        "",
        "- Tracks which recovered ranked-history fixtures now flow into the surprise join.",
        "- `JOIN_FIXED` means the recovered overlap now joins into the goal-market surprise audit.",
        "",
        "## Summary",
    ]
    summary = out.groupby("join_fix_result", dropna=False).agg(rows=("fixture_key", "size")).reset_index()
    for _, row in summary.iterrows():
        lines.append(f"- {row['join_fix_result']} | rows={int(row['rows'])}")
    lines.append("")
    for status, sub in out.groupby("join_fix_result", sort=False):
        lines.append(f"## {status}")
        for _, row in sub.iterrows():
            lines.append(
                f"- `{row['fixture_key']}` | priority=`{row['backfill_priority']}` | focus=`{row['goal_market_focus']}` | overlap=`{row['latest_overlap_status']}` | joined_markets=`{row.get('joined_markets', '') or 'none'}` | joined_rows=`{int(row['joined_rows'])}`"
            )
        lines.append("")
    Path(output_md).write_text("\n".join(lines) + "\n")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small recovered-overlap join fixer summary.")
    parser.add_argument("--surprise-join-csv", required=True)
    parser.add_argument("--overlap-csv", required=True)
    parser.add_argument("--tracker-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build(args.surprise_join_csv, args.overlap_csv, args.tracker_csv, args.output_csv, args.output_md)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")
