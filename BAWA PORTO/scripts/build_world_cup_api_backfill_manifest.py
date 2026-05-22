#!/usr/bin/env python3
"""Build a reviewable API-Football World Cup backfill manifest.

Research-only. The generated manifest is intentionally explicit about fixture
status because historical World Cups should use completed matches, while the
2026 tournament schedule needs a no-status fixture pull so upcoming matches are
not accidentally filtered out.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_OUT = Path("reports/latest/world_cup_api_football_backfill_manifest.csv")
WORLD_CUP_LEAGUE_ID = 1
WORLD_CUP_LEAGUE_TAG = "World_Cup"
DEFAULT_SEASONS = "2006,2010,2014,2018,2022,2026"


def expected_fixture_count(season: int) -> int:
    if season == 2026:
        return 104
    if season in {2010, 2014, 2018, 2022}:
        return 64
    if season == 2006:
        return 64
    return 0


def manifest_row(season: int) -> dict[str, object]:
    if season == 2006:
        return {
            "manifest_status": "EXCLUDE_API_FIXTURE_GAP",
            "league": "World Cup",
            "league_tag": WORLD_CUP_LEAGUE_TAG,
            "league_id": WORLD_CUP_LEAGUE_ID,
            "season": season,
            "fixture_status": "FT-AET-PEN",
            "expected_fixtures": expected_fixture_count(season),
            "planned_use": "FootyStats model spine only unless API-Football fixture coverage appears.",
            "notes": "Coverage audit found no local API-Football World Cup fixture source for 2006.",
        }
    if season == 2026:
        return {
            "manifest_status": "READY",
            "league": "World Cup",
            "league_tag": WORLD_CUP_LEAGUE_TAG,
            "league_id": WORLD_CUP_LEAGUE_ID,
            "season": season,
            "fixture_status": "",
            "expected_fixtures": expected_fixture_count(season),
            "planned_use": "Live tournament schedule and pre-match intelligence estate.",
            "notes": "Blank fixture_status intentionally fetches all scheduled/live/finished fixtures for 2026.",
        }
    return {
        "manifest_status": "READY",
        "league": "World Cup",
        "league_tag": WORLD_CUP_LEAGUE_TAG,
        "league_id": WORLD_CUP_LEAGUE_ID,
        "season": season,
        "fixture_status": "FT-AET-PEN",
        "expected_fixtures": expected_fixture_count(season),
        "planned_use": "Historical tournament validation estate for model spine and intelligence joins.",
        "notes": "Completed fixture pull for historical World Cup seasons.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", default=DEFAULT_SEASONS)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    seasons = [int(item.strip()) for item in args.seasons.split(",") if item.strip()]
    rows = [manifest_row(season) for season in seasons]
    out = pd.DataFrame(rows)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)

    ready = int(out["manifest_status"].eq("READY").sum()) if not out.empty else 0
    print(f"Rows: {len(out)}")
    print(f"Ready rows: {ready}")
    print(f"Output: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
