from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def build(output_csv: Path, output_md: Path) -> pd.DataFrame:
    rows = [
        {
            "market": "shots_on_target",
            "review_family": "4231v433",
            "prematch_risk_focus": "no core structural flag",
            "lookback_source": "MARKET_ONLY",
            "applied_hit_threshold": 0.50,
            "override_reason": "RELAX_FALLBACK_HIT_THRESHOLD_RESEARCH_ONLY",
        },
        {
            "market": "shots_on_target",
            "review_family": "4231v433",
            "prematch_risk_focus": "no core structural flag",
            "lookback_source": "FAMILY_ROLE",
            "applied_hit_threshold": 0.50,
            "override_reason": "RELAX_FALLBACK_HIT_THRESHOLD_RESEARCH_ONLY",
        },
    ]
    out = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    lines = [
        "# Player Events Fallback Hit Threshold Overrides",
        "",
        "- Research-only hit-threshold override set for the `shots_on_target | 4231v433 | no core structural flag` cohort.",
        "- Keeps stronger buckets unchanged while relaxing weaker fallback buckets just enough to test whether `0.55` is too strict.",
        "",
        "## Applied",
        "- MARKET_ONLY -> `0.50`",
        "- FAMILY_ROLE -> `0.50`",
        "",
        "## Unchanged",
        "- ROLE_MARKET stays at the default global threshold.",
        "- EXACT stays at the default global threshold.",
    ]
    output_md.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a research-only fallback hit-threshold override set for the SOT 4231v433 cohort.")
    parser.add_argument(
        "--output-csv",
        default=str(REPO_ROOT / "reports" / "player_events" / "quality_audits" / "player_events_fallback_hit_threshold_overrides.csv"),
    )
    parser.add_argument(
        "--output-md",
        default=str(REPO_ROOT / "reports" / "player_events" / "quality_audits" / "player_events_fallback_hit_threshold_overrides.md"),
    )
    args = parser.parse_args()
    out = build(Path(args.output_csv), Path(args.output_md))
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")


if __name__ == "__main__":
    main()
