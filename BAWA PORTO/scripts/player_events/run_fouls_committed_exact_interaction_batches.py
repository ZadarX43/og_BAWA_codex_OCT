#!/usr/bin/env python3
"""Run fouls-committed exact interaction audit in league/season batches.

Research-only orchestrator. Batching avoids one large full-estate process and
keeps failures isolated by competition/season. No deploy outputs are written.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-07" / "fouls_committed_exact_interaction_batches"
DEFAULT_LEAGUES = (
    "Belgium_Pro",
    "Brazil_Serie_A",
    "England_Championship",
    "England_EFL_League_1",
    "England_Premier_League",
    "France_Ligue_1",
    "Germany_Bundesliga",
    "Italy_Serie_A",
    "Netherlands_Eredivisie",
    "Norway_Eliteserien",
    "Portugal_Liga",
    "Scotland_Premiership",
    "Spain_La_Liga",
    "USA_MLS",
)
DEFAULT_SEASONS = (2022, 2023, 2024)


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def csv_row_count(path: Path, usecols: list[str] | None = None) -> int:
    if not path.exists():
        return 0
    try:
        return int(len(pd.read_csv(path, usecols=usecols, low_memory=False)))
    except EmptyDataError:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--leagues", default=",".join(DEFAULT_LEAGUES))
    parser.add_argument("--seasons", default=",".join(str(season) for season in DEFAULT_SEASONS))
    parser.add_argument("--max-target-rows", type=int, default=0)
    parser.add_argument("--stop-on-fail", action="store_true")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    rows: list[dict[str, object]] = []
    script = ROOT / "scripts" / "player_events" / "build_fouls_committed_exact_interaction_audit.py"

    for league in parse_csv(args.leagues):
        for season in parse_csv(args.seasons):
            batch_name = f"{league}__{season}"
            batch_outdir = args.outdir / batch_name
            cmd = [
                py,
                str(script),
                "--leagues",
                league,
                "--seasons",
                season,
                "--outdir",
                str(batch_outdir),
            ]
            if args.max_target_rows > 0:
                cmd += ["--max-target-rows", str(args.max_target_rows)]
            started = datetime.now(timezone.utc).isoformat(timespec="seconds")
            result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
            finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
            (batch_outdir / "batch_stdout.log").parent.mkdir(parents=True, exist_ok=True)
            (batch_outdir / "batch_stdout.log").write_text(result.stdout)
            (batch_outdir / "batch_stderr.log").write_text(result.stderr)
            candidates_path = batch_outdir / "fouls_committed_exact_interaction_candidate_cells.csv"
            proof_path = batch_outdir / "fouls_committed_exact_interaction_proof_rows.csv"
            rows.append(
                {
                    "league": league,
                    "season": season,
                    "status": "PASS" if result.returncode == 0 else "FAIL",
                    "returncode": result.returncode,
                    "started_at_utc": started,
                    "finished_at_utc": finished,
                    "batch_outdir": str(batch_outdir),
                    "proof_rows": csv_row_count(proof_path, usecols=["fixture_key"]),
                    "candidate_cells": csv_row_count(candidates_path),
                }
            )
            print(f"{rows[-1]['status']} {batch_name} proof_rows={rows[-1]['proof_rows']} candidate_cells={rows[-1]['candidate_cells']}")
            if result.returncode != 0 and args.stop_on_fail:
                break

    index = pd.DataFrame(rows)
    index.to_csv(args.outdir / "fouls_committed_exact_interaction_batch_index.csv", index=False)
    lines = [
        "# Fouls Committed Exact Interaction Batches",
        "",
        "Research-only batch index.",
        "",
        "## Safety",
        "- No priced player-prop odds.",
        "- No deploy routing, tiers, slips, or production rulebook changes.",
        "",
        "## Overall",
        f"- batches: `{len(index)}`",
        f"- pass: `{int(index['status'].eq('PASS').sum()) if not index.empty else 0}`",
        f"- fail: `{int(index['status'].eq('FAIL').sum()) if not index.empty else 0}`",
        f"- candidate cells: `{int(index['candidate_cells'].sum()) if not index.empty else 0}`",
    ]
    (args.outdir / "FOULS_COMMITTED_EXACT_INTERACTION_BATCHES.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
