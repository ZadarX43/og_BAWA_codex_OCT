#!/usr/bin/env python3
"""Run the post-refresh player-event exact interaction shadow chain.

Research-only orchestration:
1. Build player-event hit-rate band board from refreshed fixture inputs.
2. Project live recent-form and opponent-allowance features onto those rows.
3. Join the projected features onto the board.
4. Emit exact PLAYER_EVENT_INTERACTION watch labels.
5. Optionally refresh the unified live shadow dashboard with those labels.

No priced player-prop odds, deploy routing, source tier mutations, or
production rulebook changes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_OUTDIR = ROOT / "reports" / "2026-05-06" / "player_event_exact_interaction_shadow_refresh"


def run_step(label: str, cmd: list[str]) -> None:
    print(f"\n[step] {label}")
    print("[cmd]", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data_sources/api_football/features/player_events")
    parser.add_argument("--base-outdir", type=Path, default=DEFAULT_BASE_OUTDIR)
    parser.add_argument("--leagues", default="England_Premier_League,Spain_La_Liga")
    parser.add_argument("--hitrate-leagues", default="", help="Optional comma-separated display league names for the hit-rate board.")
    parser.add_argument("--target-seasons", default="", help="Optional comma-separated target seasons.")
    parser.add_argument("--history-seasons", default="2022,2023,2024")
    parser.add_argument("--min-minutes", type=float, default=45.0)
    parser.add_argument("--min-prob", type=float, default=0.25)
    parser.add_argument("--per-market-limit", type=int, default=200)
    parser.add_argument("--allow-proof-label-only", action="store_true")
    parser.add_argument("--skip-dashboard", action="store_true")
    parser.add_argument("--dashboard-board-dir", default="")
    args = parser.parse_args()

    base = args.base_outdir
    hitrate_dir = base / "player_event_hitrate_band_board"
    feature_dir = base / "player_event_live_interaction_features"
    join_dir = base / "player_event_live_feature_join"
    interaction_dir = base / "player_event_interaction_live_shadow_board_exact"
    dashboard_dir = base / "live_shadow_research_dashboard"
    base.mkdir(parents=True, exist_ok=True)

    py = sys.executable

    hitrate_cmd = [
        py,
        "scripts/build_player_event_hitrate_band_board.py",
        "--input-dir",
        args.input_dir,
        "--outdir",
        str(hitrate_dir),
        "--min-minutes",
        str(args.min_minutes),
        "--min-prob",
        str(args.min_prob),
    ]
    if args.hitrate_leagues:
        hitrate_cmd += ["--leagues", args.hitrate_leagues]
    run_step("build player-event hit-rate band board", hitrate_cmd)

    feature_cmd = [
        py,
        "scripts/build_player_event_live_interaction_features.py",
        "--input-dir",
        args.input_dir,
        "--leagues",
        args.leagues,
        "--history-seasons",
        args.history_seasons,
        "--outdir",
        str(feature_dir),
    ]
    if args.target_seasons:
        feature_cmd += ["--target-seasons", args.target_seasons]
    run_step("project live recent/opponent interaction features", feature_cmd)

    enriched_board = hitrate_dir / "PLAYER_EVENT_HITRATE_BAND_DASHBOARD.csv"
    recent_features = feature_dir / "player_attacker_recent_form_live_features.csv"
    opponent_features = feature_dir / "player_event_opponent_attack_allowance_live_features.csv"
    run_step(
        "join interaction features onto player-event board",
        [
            py,
            "scripts/build_player_event_live_feature_join.py",
            "--player-event-board",
            str(enriched_board),
            "--recent-form",
            str(recent_features),
            "--opponent-allowance",
            str(opponent_features),
            "--outdir",
            str(join_dir),
        ],
    )

    joined_board = join_dir / "PLAYER_EVENT_HITRATE_BAND_DASHBOARD__WITH_INTERACTION_FEATURES.csv"
    interaction_board = interaction_dir / "PLAYER_EVENT_INTERACTION_LIVE_SHADOW_BOARD.csv"
    interaction_cmd = [
        py,
        "scripts/build_player_event_interaction_live_shadow_board.py",
        "--player-event-board",
        str(joined_board),
        "--outdir",
        str(interaction_dir),
        "--per-market-limit",
        str(args.per_market_limit),
    ]
    if args.allow_proof_label_only:
        interaction_cmd.append("--allow-proof-label-only")
    run_step("emit exact player-event interaction watch labels", interaction_cmd)

    if not args.skip_dashboard:
        dashboard_cmd = [
            py,
            "scripts/build_live_shadow_research_dashboard.py",
            "--outdir",
            str(dashboard_dir),
            "--player-event-interaction-board",
            str(interaction_board),
            "--limit",
            "1",
        ]
        if args.dashboard_board_dir:
            dashboard_cmd += ["--board-dir", args.dashboard_board_dir]
        run_step("refresh unified live shadow dashboard", dashboard_cmd)

    print("\n[ok] player-event exact interaction shadow refresh complete")
    print(f"[ok] interaction board: {interaction_board}")
    print(f"[ok] joined board: {joined_board}")


if __name__ == "__main__":
    main()
