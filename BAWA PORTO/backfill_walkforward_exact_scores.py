#!/usr/bin/env python3
"""Backfill exact goal counts into existing walk-forward 03_scored exports using merged actuals."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from run_walkforward_windows import enrich_with_merged_actuals, load_window_merged_actuals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill exact scores into walk-forward scored deploy exports")
    parser.add_argument("--walkforward-root", required=True, help="Walk-forward root containing window folders")
    parser.add_argument("--merged-dir", required=True, help="Directory holding merged league result CSVs")
    parser.add_argument("--write", action="store_true", help="Actually rewrite the scored CSVs in place")
    return parser.parse_args()


def choose_source_csv(source_dir: Path) -> Path | None:
    candidates = sorted(source_dir.glob("*.csv"))
    if not candidates:
        return None
    for candidate in candidates:
        if "ALLMARKETS" in candidate.name.upper():
            return candidate
    return candidates[0]


def main() -> None:
    args = parse_args()
    walkforward_root = Path(args.walkforward_root)
    merged_dir = Path(args.merged_dir)

    if not walkforward_root.exists():
        raise SystemExit(f"Missing walkforward root: {walkforward_root}")
    if not merged_dir.exists():
        raise SystemExit(f"Missing merged dir: {merged_dir}")

    rows: list[dict] = []
    for window_dir in sorted(p for p in walkforward_root.iterdir() if p.is_dir() and p.name.startswith("w")):
        source_dir = window_dir / "01_source"
        scored_dir = window_dir / "03_scored"
        if not source_dir.exists() or not scored_dir.exists():
            continue

        source_csv = choose_source_csv(source_dir)
        if source_csv is None:
            continue

        source_df = pd.read_csv(source_csv, low_memory=False)
        merged_actuals = load_window_merged_actuals(source_df, merged_dir)
        if merged_actuals.empty:
            rows.append(
                {
                    "window_id": window_dir.name,
                    "scored_files": 0,
                    "goal_rows_before": 0,
                    "goal_rows_after": 0,
                    "rewritten": 0,
                    "status": "no_merged_actuals",
                }
            )
            continue

        scored_files = sorted(scored_dir.glob("*.csv"))
        total_before = 0
        total_after = 0
        rewritten = 0
        for scored_csv in scored_files:
            df = pd.read_csv(scored_csv, low_memory=False)
            before = int(
                pd.to_numeric(df.get("home_team_goal_count", pd.Series([], dtype=float)), errors="coerce").notna().sum()
            )
            enriched = enrich_with_merged_actuals(df, merged_actuals)
            after = int(
                pd.to_numeric(enriched.get("home_team_goal_count", pd.Series([], dtype=float)), errors="coerce").notna().sum()
            )
            total_before += before
            total_after += after
            if args.write and after > before:
                enriched.to_csv(scored_csv, index=False)
                rewritten += 1

        rows.append(
            {
                "window_id": window_dir.name,
                "scored_files": len(scored_files),
                "goal_rows_before": total_before,
                "goal_rows_after": total_after,
                "rewritten": rewritten,
                "status": "written" if args.write else "dry_run",
            }
        )

    summary = pd.DataFrame(rows).sort_values("window_id").reset_index(drop=True)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
