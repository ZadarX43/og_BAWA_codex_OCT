from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def build(output_csv: Path, output_md: Path) -> pd.DataFrame:
    rows = [
        {
            "market": "shots_on_target",
            "min_prior_hits": 1,
            "min_prior_apps": 1,
            "min_prior_hit_rate": 0.0,
            "reason": "ONLY_PLAYERS_WITH_PRIOR_SOT_HIT",
        },
        {
            "market": "tackles",
            "min_prior_hits": 1,
            "min_prior_apps": 1,
            "min_prior_hit_rate": 0.0,
            "reason": "ONLY_PLAYERS_WITH_PRIOR_TACKLE_HIT",
        },
        {
            "market": "fouls_committed",
            "min_prior_hits": 1,
            "min_prior_apps": 1,
            "min_prior_hit_rate": 0.0,
            "reason": "ONLY_PLAYERS_WITH_PRIOR_FOUL_HIT",
        },
    ]
    out = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    lines = [
        "# Player Events Market History Gate",
        "",
        "- Research-only market-history gate for premium-style filtering.",
        "- Principle: only predict a market for players who have already shown that market historically in the local 3Y actuals layer.",
        "",
        "## Applied",
        "- shots_on_target -> at least 1 prior SOT hit",
        "- tackles -> at least 1 prior tackle hit",
        "- fouls_committed -> at least 1 prior foul hit",
    ]
    output_md.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a research-only market-history gate for player-events backtesting.")
    parser.add_argument(
        "--output-csv",
        default=str(REPO_ROOT / "reports" / "player_events" / "quality_audits" / "player_events_market_history_gate.csv"),
    )
    parser.add_argument(
        "--output-md",
        default=str(REPO_ROOT / "reports" / "player_events" / "quality_audits" / "player_events_market_history_gate.md"),
    )
    args = parser.parse_args()
    out = build(Path(args.output_csv), Path(args.output_md))
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")


if __name__ == "__main__":
    main()
