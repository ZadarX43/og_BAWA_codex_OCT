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
            "min_hits_l3": 0,
            "min_hits_l5": 1,
            "min_hits_l8": 1,
            "min_hits_l10": 2,
            "min_hit_rate_l3": 0.0,
            "min_hit_rate_l5": 0.20,
            "min_hit_rate_l8": 0.15,
            "min_hit_rate_l10": 0.20,
            "reason": "RECENT_SOT_INVOLVEMENT_GATE",
        },
        {
            "market": "tackles",
            "min_prior_hits": 1,
            "min_prior_apps": 1,
            "min_prior_hit_rate": 0.0,
            "min_hits_l3": 0,
            "min_hits_l5": 1,
            "min_hits_l8": 2,
            "min_hits_l10": 3,
            "min_hit_rate_l3": 0.0,
            "min_hit_rate_l5": 0.20,
            "min_hit_rate_l8": 0.20,
            "min_hit_rate_l10": 0.25,
            "reason": "RECENT_TACKLE_INVOLVEMENT_GATE_RELAXED",
        },
        {
            "market": "fouls_committed",
            "min_prior_hits": 1,
            "min_prior_apps": 1,
            "min_prior_hit_rate": 0.0,
            "min_hits_l3": 0,
            "min_hits_l5": 1,
            "min_hits_l8": 2,
            "min_hits_l10": 3,
            "min_hit_rate_l3": 0.0,
            "min_hit_rate_l5": 0.20,
            "min_hit_rate_l8": 0.20,
            "min_hit_rate_l10": 0.25,
            "reason": "RECENT_FOUL_INVOLVEMENT_GATE_RELAXED",
        },
    ]
    out = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    lines = [
        "# Player Events Recent Involvement Gate",
        "",
        "- Research-only gate that requires recent market-specific involvement across the last `3 / 5 / 8 / 10` player appearances.",
        "- This extends the simpler market-history gate so we are not just asking whether the player has ever done the thing, but whether they are still doing it recently.",
        "",
        "## Applied",
        "- shots_on_target: at least 1 hit in l5, 1 in l8, 2 in l10, with mild l5/l8/l10 hit-rate floors",
        "- tackles: relaxed repeat recent tackle hits across l5/l8/l10, with softer hit-rate floors",
        "- fouls_committed: relaxed repeat recent foul hits across l5/l8/l10, with softer hit-rate floors",
    ]
    output_md.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a research-only recent involvement gate for player-events markets.")
    parser.add_argument(
        "--output-csv",
        default=str(REPO_ROOT / "reports" / "player_events" / "quality_audits" / "player_events_recent_involvement_gate.csv"),
    )
    parser.add_argument(
        "--output-md",
        default=str(REPO_ROOT / "reports" / "player_events" / "quality_audits" / "player_events_recent_involvement_gate.md"),
    )
    args = parser.parse_args()
    out = build(Path(args.output_csv), Path(args.output_md))
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")


if __name__ == "__main__":
    main()
