from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _exists(path: str) -> bool:
    return Path(path).exists()


def build_hunt_note(output_md: str) -> None:
    normalized_examples = [
        "data_sources/api_football/normalized/match_player_stats__England_EFL_League_1__2022.csv",
        "data_sources/api_football/normalized/match_player_stats__England_FA_Cup__2024.csv",
    ]
    rolling_examples = [
        "data_sources/api_football/features/api_player_rolling_features__Europa_League__2024.csv",
        "data_sources/api_football/features/api_player_rolling_features__Japan_J1__2024.csv",
    ]
    schema_examples = [
        "docs/API_FOOTBALL_SCHEMA.md",
        "docs/PLAYER_EVENTS_FRAMEWORK.md",
        "docs/PLAYER_EVENTS_INPUT_SCHEMA.csv",
    ]

    lines = [
        "# CB Duel Context Data Hunt",
        "",
        "## Current Local Evidence",
        "",
        "The current repo already contains raw or derived duel-pressure ingredients that can help unlock the striker-vs-front-foot-CB lane:",
        "",
        "- Normalized player stats include raw defensive duel fields like `interceptions`, `blocks`, `duels_total`, `duels_won`, and `dribbled_past`.",
        "- Rolling player feature files already carry `player_duel_win_rate_l5`.",
        "- The player-events fixture input already derives `interceptions_per90`, `ground_duel_loss_rate`, `aerial_duel_loss_rate`, and `dribbles_faced_per90`.",
        "",
        "## Verified File Anchors",
        "",
    ]
    lines.extend([f"- `{path}`{' (present)' if _exists(path) else ' (missing)'}" for path in normalized_examples + rolling_examples + schema_examples])
    lines.extend(
        [
            "",
            "## What Is Still Missing",
            "",
            "The weak point is not total absence of duel data; it is that the current specialist outputs do not preserve a strong centre-back-specific pressure context end-to-end.",
            "",
            "What we still need for a deployable striker-CB lane:",
            "- explicit CB-facing aerial pressure or header-volume proxies",
            "- front-foot vs passive centre-back classification that survives into the specialist boards",
            "- opponent striker / target-forward pressure tags",
            "- recovery-defending or space-behind pressure proxies carried into the final shortlist layer",
            "",
            "## Best Next Enrichment Targets",
            "",
            "1. Extend `build_player_events_fixture_input.py` so centre-backs carry stronger derived context from:",
            "   - `blocks`",
            "   - `duels_total` / `duels_won`",
            "   - `dribbled_past`",
            "   - existing `aerial_duel_loss_rate` placeholder",
            "2. Promote a CB-specific role slice into the portable contact outputs rather than letting it remain too sparse.",
            "3. Add opponent striker-profile tags for direct leagues and competitions before trusting this lane live.",
            "",
            "## Current Conclusion",
            "",
            "The striker/front-foot-CB lane is still research-only. The repo has useful raw ingredients, but we have not yet converted them into a repeated, surviving centre-back matchup signal in the elite boards.",
            "",
        ]
    )
    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md).write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a small local data-hunt note for richer centre-back duel context.")
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_hunt_note(args.output_md)
    print(f"WROTE: {args.output_md}")
