#!/usr/bin/env python3
"""Audit current API-Football feature coverage by league, family, and year.

This is an offline coverage/readiness pass. It reads normalized/features CSVs
already on disk and does not call the API or spend quota.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_API_ROOT = Path("data_sources/api_football")
DEFAULT_OUTDIR = Path("reports/2026-05-06/api_football_feature_coverage_audit")

YEAR_RE = re.compile(r"__(?P<league>.+)__(?P<year>20\d{2})$")

FOUNDATION_FAMILIES = {
    "fixtures_master",
    "match_team_stats",
    "match_events",
    "match_player_stats",
    "lineups",
    "injuries",
    "odds_prematch_long",
}

FEATURE_FAMILIES = {
    "team_rolling",
    "player_rolling",
    "lineup",
    "injury",
    "event",
    "odds",
    "live",
    "enriched_fixture",
    "h2h",
    "referee",
    "team_identity",
    "matchup",
}


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    text = df.copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            text[col] = text[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            text[col] = text[col].astype("string").fillna("")
    lines = [
        "| " + " | ".join(str(c) for c in text.columns) + " |",
        "| " + " | ".join(["---"] * len(text.columns)) + " |",
    ]
    for _, row in text.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in text.columns) + " |")
    return "\n".join(lines)


def parse_file(path: Path, api_root: Path) -> dict[str, str]:
    stem = path.stem
    match = YEAR_RE.search(stem)
    if match:
        league = match.group("league").replace("_", " ")
        year = match.group("year")
        family = stem[: match.start()]
    else:
        league = "COMBINED_OR_UNKNOWN"
        year = "UNKNOWN"
        family = stem

    source_layer = path.relative_to(api_root).parts[0] if api_root in path.parents else "unknown"
    family_norm = family
    for prefix in ["api_", "feature_"]:
        if family_norm.startswith(prefix):
            family_norm = family_norm[len(prefix) :]

    return {
        "source_layer": source_layer,
        "family": family_norm,
        "league": league,
        "year": year,
        "path": str(path),
    }


def count_csv(path: Path) -> dict[str, int | float | str]:
    try:
        header = pd.read_csv(path, nrows=0)
        usecols = list(header.columns)
        sample = pd.read_csv(path, usecols=usecols, low_memory=False)
    except Exception as exc:
        return {
            "rows": 0,
            "columns": 0,
            "unique_fixtures": 0,
            "unique_teams": 0,
            "unique_players": 0,
            "non_null_ratio": np.nan,
            "read_error": str(exc),
        }

    cols = sample.columns
    fixture_cols = [c for c in ["fixture_id", "fixture_key"] if c in cols]
    team_cols = [c for c in ["team_id", "home_team_id", "away_team_id", "team_name"] if c in cols]
    player_cols = [c for c in ["player_id", "player_name"] if c in cols]
    unique_fixtures = 0
    if fixture_cols:
        unique_fixtures = int(sample[fixture_cols[0]].nunique(dropna=True))
    unique_teams = 0
    if team_cols:
        unique_teams = int(pd.concat([sample[c] for c in team_cols], ignore_index=True).nunique(dropna=True))
    unique_players = 0
    if player_cols:
        unique_players = int(pd.concat([sample[c] for c in player_cols], ignore_index=True).nunique(dropna=True))
    non_null_ratio = float(sample.notna().mean().mean()) if len(sample) and len(cols) else np.nan
    return {
        "rows": int(len(sample)),
        "columns": int(len(cols)),
        "unique_fixtures": unique_fixtures,
        "unique_teams": unique_teams,
        "unique_players": unique_players,
        "non_null_ratio": non_null_ratio,
        "read_error": "",
    }


def readiness_bucket(row: pd.Series) -> str:
    fixtures = int(row.get("fixtures_master_rows", 0))
    team_stats = int(row.get("match_team_stats_rows", 0))
    events = int(row.get("match_events_rows", 0))
    players = int(row.get("match_player_stats_rows", 0))
    lineups = int(row.get("lineups_rows", 0))
    injuries = int(row.get("injuries_rows", 0))
    odds = int(row.get("odds_prematch_long_rows", 0))
    populated_families = int(row.get("populated_families", 0))

    if fixtures >= 100 and team_stats >= fixtures and events >= fixtures and lineups >= fixtures:
        if odds >= fixtures * 0.50 and (players >= fixtures or injuries > 0):
            return "SAFE_ENRICHMENT"
        return "FOUNDATION_SAFE_NOISY_EXTRAS"
    if fixtures >= 50 and team_stats >= fixtures * 0.75 and populated_families >= 4:
        return "OBSERVE_WITH_CONFIRM"
    if fixtures > 0:
        return "SPARSE_OBSERVE"
    return "NO_LOCAL_COVERAGE"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-root", default=str(DEFAULT_API_ROOT))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    args = parser.parse_args()

    api_root = Path(args.api_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(api_root.glob("normalized/*.csv")) + sorted(api_root.glob("features/*.csv"))
    rows = []
    for path in files:
        meta = parse_file(path, api_root)
        counts = count_csv(path)
        rows.append({**meta, **counts})

    coverage = pd.DataFrame(rows)
    if coverage.empty:
        raise SystemExit(f"No API-Football CSVs found under {api_root}")
    coverage.to_csv(outdir / "api_football_feature_coverage_by_family.csv", index=False)

    known = coverage[coverage["year"].ne("UNKNOWN")].copy()
    matrix = (
        known.pivot_table(
            index=["league", "year"],
            columns="family",
            values="rows",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for family in sorted(FOUNDATION_FAMILIES | FEATURE_FAMILIES):
        if family in matrix.columns:
            matrix[f"{family}_rows"] = matrix[family].astype(int)
            matrix = matrix.drop(columns=[family])
        else:
            matrix[f"{family}_rows"] = 0
    row_cols = [c for c in matrix.columns if c.endswith("_rows")]
    matrix["populated_families"] = matrix[row_cols].gt(0).sum(axis=1)
    matrix["readiness_bucket"] = matrix.apply(readiness_bucket, axis=1)
    matrix = matrix.sort_values(["readiness_bucket", "league", "year"])
    matrix.to_csv(outdir / "api_football_feature_coverage_matrix.csv", index=False)

    league_readiness = (
        matrix.groupby("league", dropna=False)
        .agg(
            years=("year", "nunique"),
            latest_year=("year", "max"),
            readiness_best=("readiness_bucket", lambda s: ", ".join(sorted(set(s)))),
            total_fixtures=("fixtures_master_rows", "sum"),
            total_team_stats=("match_team_stats_rows", "sum"),
            total_events=("match_events_rows", "sum"),
            total_lineups=("lineups_rows", "sum"),
            total_injuries=("injuries_rows", "sum"),
            total_odds=("odds_prematch_long_rows", "sum"),
            max_populated_families=("populated_families", "max"),
        )
        .reset_index()
    )
    bucket_rank = {
        "SAFE_ENRICHMENT": 0,
        "FOUNDATION_SAFE_NOISY_EXTRAS": 1,
        "OBSERVE_WITH_CONFIRM": 2,
        "SPARSE_OBSERVE": 3,
        "NO_LOCAL_COVERAGE": 4,
    }
    best_bucket = []
    for _, group in matrix.groupby("league", dropna=False):
        buckets = sorted(set(group["readiness_bucket"]), key=lambda x: bucket_rank.get(x, 99))
        best_bucket.append({"league": group["league"].iloc[0], "best_bucket": buckets[0]})
    best_bucket_df = pd.DataFrame(best_bucket)
    league_readiness = league_readiness.merge(best_bucket_df, on="league", how="left")
    league_readiness = league_readiness.sort_values(["best_bucket", "league"])
    league_readiness.to_csv(outdir / "api_football_enrichment_readiness.csv", index=False)

    bucket_counts = (
        league_readiness.groupby("best_bucket", dropna=False)
        .size()
        .reset_index(name="leagues")
        .sort_values("best_bucket")
    )
    family_totals = (
        coverage.groupby(["source_layer", "family"], dropna=False)
        .agg(files=("path", "count"), rows=("rows", "sum"), unique_fixtures=("unique_fixtures", "sum"))
        .reset_index()
        .sort_values(["source_layer", "rows"], ascending=[True, False])
    )
    family_totals.to_csv(outdir / "api_football_family_totals.csv", index=False)

    summary = [
        "# API-Football Feature Coverage Audit",
        "",
        "Offline audit over local normalized/features CSVs. No API calls were made.",
        "",
        "## Readiness Bucket Counts",
        markdown_table(bucket_counts),
        "",
        "## League Readiness",
        markdown_table(
            league_readiness[
                [
                    "league",
                    "best_bucket",
                    "years",
                    "latest_year",
                    "total_fixtures",
                    "total_team_stats",
                    "total_events",
                    "total_lineups",
                    "total_injuries",
                    "total_odds",
                    "max_populated_families",
                ]
            ].head(60)
        ),
        "",
        "## Family Totals",
        markdown_table(family_totals.head(40)),
        "",
        "## Operating Recommendation",
        (
            "Use SAFE_ENRICHMENT and FOUNDATION_SAFE_NOISY_EXTRAS leagues for API feature ablations. "
            "Keep OBSERVE_WITH_CONFIRM and SPARSE_OBSERVE leagues out of restoration gates until "
            "league-specific backtests prove the enrichment is stable."
        ),
    ]
    (outdir / "api_football_feature_coverage_audit.md").write_text("\n".join(summary), encoding="utf-8")

    print(f"WROTE {outdir}")
    print(f"files={len(coverage)} leagues={league_readiness['league'].nunique()}")


if __name__ == "__main__":
    main()
