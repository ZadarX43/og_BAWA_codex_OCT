from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def build(backtest_dir: Path, output_md: Path) -> None:
    lines = [
        "# Current Player Events Research Baseline",
        "",
        "- Current best research baseline: `cycle6_recent_relaxed_contact`.",
        f"- Backtest pack: `{backtest_dir}`",
        "",
        "## Why This Is The Baseline",
        "- Keeps the stricter recent-involvement treatment for `shots_on_target`.",
        "- Relaxes `tackles` and `fouls_committed` enough to avoid over-suppressing contact markets.",
        "- Beats the raw baseline by improving selected hits and reducing selected misses at the same total selected count.",
        "",
        "## Working Market Posture",
        "- `shots_on_target`: premium striker / lead-winger lane.",
        "- `shots`: broader attacking pool lane, to be refined separately from premium `shots_on_target`.",
        "- `tackles`: strongest contact benchmark lane.",
        "- `fouls_committed`: still building; keep research-only and more cautious.",
        "- `yellow_cards`: separate, noisier lane.",
        "",
        "## Use",
        "- Treat this as the comparison baseline for the next wave of player-events refinement.",
        "- New research-only gate or threshold trials should be compared against this cycle first, not the old raw baseline.",
    ]
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a small note freezing the current best player-events research baseline.")
    parser.add_argument(
        "--backtest-dir",
        default=str(REPO_ROOT / "reports" / "player_events" / "backtests" / "player_events_3y_backtest__2026-05-04__tuned_cycle_6_recent_relaxed_contact"),
    )
    parser.add_argument(
        "--output-md",
        default=str(REPO_ROOT / "reports" / "player_events" / "quality_audits" / "CURRENT_PLAYER_EVENTS_RESEARCH_BASELINE.md"),
    )
    args = parser.parse_args()
    build(Path(args.backtest_dir), Path(args.output_md))
    print(f"WROTE: {args.output_md}")


if __name__ == "__main__":
    main()
