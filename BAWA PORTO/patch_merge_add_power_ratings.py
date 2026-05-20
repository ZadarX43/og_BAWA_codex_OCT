#!/usr/bin/env python3
"""
Patch rolling power-rating columns from ModelStore artifacts back into merged CSVs.

``team_ratings.py --mode rolling`` writes:
    ModelStore/<LeagueTag>_match_power_ratings.csv

This script joins the rolling power columns back into:
    Matches/__merged__/<LeagueTag>__merged.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
MERGED_ROOT = PROJECT_ROOT / "Matches" / "__merged__"
MODEL_STORE = PROJECT_ROOT / "ModelStore"

POWER_COLS = [
    "home_power_rating_raw",
    "away_power_rating_raw",
    "home_power_rating_post_raw",
    "away_power_rating_post_raw",
    "home_power_rating",
    "away_power_rating",
    "power_diff",
]

JOIN_KEY_CANDIDATES = [
    ["fixture_key"],
    ["match_id"],
    ["match_date", "home_team_name", "away_team_name"],
    ["date_GMT", "home_team_name", "away_team_name"],
]


def _discover_leagues(merged_root: Path) -> list[str]:
    return sorted(p.stem.replace("__merged", "") for p in merged_root.glob("*__merged.csv"))


def _pick_join_keys(merged: pd.DataFrame, ratings: pd.DataFrame) -> list[str]:
    for keys in JOIN_KEY_CANDIDATES:
        if all(k in merged.columns for k in keys) and all(k in ratings.columns for k in keys):
            return keys
    return []


def patch_league(
    league_tag: str,
    merged_root: Path,
    model_store: Path,
    dry_run: bool = False,
) -> dict:
    merged_path = merged_root / f"{league_tag}__merged.csv"
    ratings_path = model_store / f"{league_tag}_match_power_ratings.csv"

    if not merged_path.exists():
        return {"league": league_tag, "status": "SKIP", "reason": "merged file not found"}
    if not ratings_path.exists():
        return {
            "league": league_tag,
            "status": "SKIP",
            "reason": f"ratings artifact not found: {ratings_path.name}",
        }

    try:
        merged = pd.read_csv(merged_path, low_memory=False)
        ratings = pd.read_csv(ratings_path, low_memory=False)

        missing_power = [c for c in POWER_COLS if c not in ratings.columns]
        if missing_power:
            return {
                "league": league_tag,
                "status": "ERROR",
                "reason": f"ratings artifact missing power cols: {missing_power}",
            }

        join_keys = _pick_join_keys(merged, ratings)
        if not join_keys:
            return {
                "league": league_tag,
                "status": "ERROR",
                "reason": "no common join key found between merged and ratings artifact",
            }

        existing_power = [c for c in POWER_COLS if c in merged.columns]
        if existing_power:
            merged = merged.drop(columns=existing_power)

        ratings_slim = ratings[join_keys + POWER_COLS].drop_duplicates(subset=join_keys).copy()
        merged_out = merged.merge(ratings_slim, on=join_keys, how="left")

        rows_matched = int(merged_out["home_power_rating"].notna().sum())
        max_nan_rate = float(merged_out[POWER_COLS].isna().mean().max())

        if not dry_run:
            merged_out.to_csv(merged_path, index=False)

        return {
            "league": league_tag,
            "status": "DRY_RUN" if dry_run else "OK",
            "rows_total": int(len(merged_out)),
            "rows_matched": rows_matched,
            "max_nan_rate": round(max_nan_rate, 4),
            "join_keys": ",".join(join_keys),
        }
    except Exception as exc:  # pragma: no cover - defensive utility
        return {"league": league_tag, "status": "ERROR", "reason": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch rolling power ratings into merged league CSVs.")
    parser.add_argument(
        "--leagues",
        default="",
        help="Comma-separated league tags. Default: all discovered merged files.",
    )
    parser.add_argument("--merged-root", default=str(MERGED_ROOT))
    parser.add_argument("--model-store", default=str(MODEL_STORE))
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing.")
    args = parser.parse_args()

    merged_root = Path(args.merged_root)
    model_store = Path(args.model_store)
    if args.leagues.strip():
        leagues = [s.strip().replace(" ", "_") for s in args.leagues.split(",") if s.strip()]
    else:
        leagues = _discover_leagues(merged_root)

    print(f"Patching power ratings for {len(leagues)} leagues ({'DRY RUN' if args.dry_run else 'LIVE'})...")
    results = [
        patch_league(league, merged_root, model_store, dry_run=args.dry_run)
        for league in leagues
    ]

    for result in results:
        status = result["status"]
        league = result["league"]
        if status in {"OK", "DRY_RUN"}:
            icon = "🔍" if status == "DRY_RUN" else "✅"
            print(
                f"  {icon} {league}: {result['rows_matched']}/{result['rows_total']} rows matched "
                f"join={result['join_keys']} max_nan={result['max_nan_rate']:.1%}"
            )
        elif status == "SKIP":
            print(f"  ⏭️  {league}: {result['reason']}")
        else:
            print(f"  ❌ {league}: {result['reason']}")

    errors = [r for r in results if r["status"] == "ERROR"]
    skips = [r for r in results if r["status"] == "SKIP"]
    ok = [r for r in results if r["status"] in {"OK", "DRY_RUN"}]
    print(f"\nDone. {len(ok)} patched, {len(skips)} skipped, {len(errors)} errors.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
