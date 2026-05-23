#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd


ODDS_GEOM_COLS = [
    "draw_implied",
    "implied_prob_diff",
    "odds_diff",
    "odds_parity",
    "odds_skew",
]

ROLLING_PRESS_COLS = [
    "rolling5_home_press_intensity",
    "rolling5_away_press_intensity",
    "rolling5_press_intensity_diff",
    "rolling5_home_press_z",
    "rolling5_away_press_z",
    "rolling5_press_z_diff",
]


def _league_tag(name: str) -> str:
    return name.replace(" ", "_")


def _iter_merged_files(merged_dir: Path, leagues: list[str] | None) -> list[Path]:
    if leagues:
        out = []
        for lg in leagues:
            tag = _league_tag(lg)
            p = merged_dir / f"{tag}__merged.csv"
            out.append(p)
        return out
    return sorted([p for p in merged_dir.glob("*.csv") if p.is_file()])


def _check_cols(df: pd.DataFrame, cols: list[str]) -> tuple[list[str], list[str]]:
    missing = [c for c in cols if c not in df.columns]
    empty = []
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() == 0:
            empty.append(c)
    return missing, empty


def main() -> int:
    ap = argparse.ArgumentParser(description="Fail if merged outputs lack odds geometry or rolling press columns.")
    ap.add_argument(
        "--merged-dir",
        default="Matches/__merged__",
        help="Merged CSV directory (default: Matches/__merged__)",
    )
    ap.add_argument(
        "--leagues",
        default="",
        help="Comma-separated league names to check (default: all merged CSVs)",
    )
    args = ap.parse_args()

    merged_dir = Path(args.merged_dir)
    if not merged_dir.exists():
        print(f"ERROR: merged dir not found: {merged_dir}", file=sys.stderr)
        return 2

    leagues = [s.strip() for s in args.leagues.split(",") if s.strip()] if args.leagues else None
    files = _iter_merged_files(merged_dir, leagues)

    failures = []
    for p in files:
        if not p.exists():
            failures.append((p.name, "missing_file", "merged csv not found"))
            continue
        df = pd.read_csv(p, nrows=2000, low_memory=False)
        miss_odds, empty_odds = _check_cols(df, ODDS_GEOM_COLS)
        miss_press, empty_press = _check_cols(df, ROLLING_PRESS_COLS)
        if miss_odds or empty_odds or miss_press or empty_press:
            failures.append(
                (
                    p.name,
                    "schema_gap",
                    f"missing_odds={miss_odds} empty_odds={empty_odds} "
                    f"missing_press={miss_press} empty_press={empty_press}",
                )
            )

    if failures:
        print("SCHEMA GUARD FAILURES:")
        for name, kind, detail in failures:
            print(f"- {name} | {kind} | {detail}")
        return 2

    print("Schema guard OK: odds geometry + rolling press present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
