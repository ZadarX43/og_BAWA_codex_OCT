#!/usr/bin/env python3
"""
Patch H2H and streak features into merged league CSVs.

This script wraps ``streaks_module.attach_streaks_and_h2h`` and writes the
result back into ``Matches/__merged__/<LeagueTag>__merged.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
MERGED_ROOT = PROJECT_ROOT / "Matches" / "__merged__"

sys.path.insert(0, str(PROJECT_ROOT))
from streaks_module import attach_streaks_and_h2h  # type: ignore  # noqa: E402


def _discover_leagues(merged_root: Path) -> list[str]:
    return sorted(p.stem.replace("__merged", "") for p in merged_root.glob("*__merged.csv"))


def _iter_new_family_cols(before: Iterable[str], after: Iterable[str], token: str) -> list[str]:
    before_set = set(before)
    return sorted(c for c in after if c not in before_set and token in c.lower())


def patch_league(league_tag: str, merged_root: Path, dry_run: bool = False) -> dict:
    merged_path = merged_root / f"{league_tag}__merged.csv"
    if not merged_path.exists():
        return {"league": league_tag, "status": "SKIP", "reason": "merged file not found"}

    try:
        df = pd.read_csv(merged_path, low_memory=False)
        cols_before = list(df.columns)
        patched = attach_streaks_and_h2h(
            df,
            include_implied_vs_actual=False,
            include_composites=True,
        )
        cols_after = list(patched.columns)

        new_cols = sorted(set(cols_after) - set(cols_before))
        h2h_cols = _iter_new_family_cols(cols_before, cols_after, "h2h")
        streak_cols = _iter_new_family_cols(cols_before, cols_after, "streak")
        max_nan_rate = (
            float(patched[new_cols].isna().mean().max()) if new_cols else 1.0
        )

        if not dry_run:
            patched.to_csv(merged_path, index=False)

        return {
            "league": league_tag,
            "status": "DRY_RUN" if dry_run else "OK",
            "rows": int(len(patched)),
            "new_cols": int(len(new_cols)),
            "h2h_cols": int(len(h2h_cols)),
            "streak_cols": int(len(streak_cols)),
            "max_nan_rate": round(max_nan_rate, 4),
        }
    except Exception as exc:  # pragma: no cover - defensive utility
        return {"league": league_tag, "status": "ERROR", "reason": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch H2H and streak features into merged league CSVs.")
    parser.add_argument(
        "--leagues",
        default="",
        help="Comma-separated league tags or folder names. Default: all merged CSVs.",
    )
    parser.add_argument(
        "--merged-root",
        default=str(MERGED_ROOT),
        help="Path to Matches/__merged__ (default: repo Matches/__merged__).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing files.")
    args = parser.parse_args()

    merged_root = Path(args.merged_root)
    if args.leagues.strip():
        leagues = [s.strip().replace(" ", "_") for s in args.leagues.split(",") if s.strip()]
    else:
        leagues = _discover_leagues(merged_root)

    print(f"Patching streaks/H2H for {len(leagues)} leagues ({'DRY RUN' if args.dry_run else 'LIVE'})...")

    results = [patch_league(league, merged_root, dry_run=args.dry_run) for league in leagues]
    for result in results:
        status = result["status"]
        league = result["league"]
        if status in {"OK", "DRY_RUN"}:
            icon = "🔍" if status == "DRY_RUN" else "✅"
            print(
                f"  {icon} {league}: +{result['new_cols']} cols "
                f"(h2h={result['h2h_cols']}, streak={result['streak_cols']}) "
                f"max_nan={result['max_nan_rate']:.1%}"
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
