#!/usr/bin/env python3
"""Run repeat live-shadow collection and stability QA.

Research-only orchestration:
  1. Rebuild team-goal threshold stability buckets.
  2. Rebuild the unified live-shadow dashboard across complete real boards.
  3. Emit a compact status note for forward-facing QA.

This script does not run predictions, does not edit deploy tiers, and does not
modify production rulebook behavior.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_OUTDIR = Path("reports/2026-05-06/live_shadow_repeat_collection")
DEFAULT_DASHBOARD_INDEX = Path("reports/2026-05-06/live_shadow_research_dashboard/live_shadow_research_dashboard_index.csv")


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    text = df.copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            text[col] = text[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            text[col] = text[col].astype("string").fillna("")
    lines = [
        "| " + " | ".join(str(c) for c in text.columns) + " |",
        "| " + " | ".join(["---"] * len(text.columns)) + " |",
    ]
    for _, row in text.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in text.columns) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--dashboard-index", default=str(DEFAULT_DASHBOARD_INDEX))
    parser.add_argument("--skip-stability", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    if not args.skip_stability:
        run([python, "scripts/build_team_goal_threshold_stability_report.py"])
    run([python, "scripts/build_live_shadow_research_dashboard.py", "--all"])

    index_path = Path(args.dashboard_index)
    if not index_path.exists():
        raise SystemExit(f"Missing dashboard index: {index_path}")
    index = pd.read_csv(index_path)
    status_counts = index["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="boards")
    totals = index[
        [
            "dashboard_rows",
            "ou25_rows",
            "ftr_c5_rows",
            "team_goal_market_rows",
            "team_goal_combo_rows",
            "ftr_btts_combo_rows",
        ]
    ].sum().reset_index()
    totals.columns = ["metric", "rows"]

    latest = index.head(12)[
        [
            "fixture_range",
            "status",
            "source_rows",
            "dashboard_rows",
            "ou25_rows",
            "ftr_c5_rows",
            "team_goal_market_rows",
            "team_goal_combo_rows",
            "ftr_btts_combo_rows",
        ]
    ]

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Live Shadow Repeat Collection Status",
        "",
        f"- Generated: `{now}`",
        f"- Dashboard boards scanned: `{len(index)}`",
        "",
        "## Status Counts",
        markdown_table(status_counts),
        "",
        "## Shadow Row Totals",
        markdown_table(totals),
        "",
        "## Latest Boards",
        markdown_table(latest),
        "",
        "## Operating Guardrails",
        "",
        "- Keep all outputs shadow-only.",
        "- Prioritise `SHADOW_CORE` and `TEAM_CORE` rows in watch tables.",
        "- Do not promote `TG25_MONSTER` or `MATCH_OVER_3_5_SHADOW` without tighter league/team proof.",
        "- Value remains additive only and cannot override deploy gates.",
        "- `OBSERVE` remains non-deployable.",
        "- World Cup remains OBSERVE-only until fixtures, odds, squads, injuries, venues, and rest/travel context are normalized.",
    ]
    status_path = outdir / "live_shadow_repeat_collection_status.md"
    status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[ok] boards={len(index)}")
    print(f"[ok] wrote {status_path}")


if __name__ == "__main__":
    main()
