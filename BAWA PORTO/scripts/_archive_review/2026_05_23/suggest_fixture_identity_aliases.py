#!/usr/bin/env python3
"""Suggest team aliases from unresolved fixture identity rows.

Research-only helper. It reads scored fixtures plus local API-Football calendar
fixtures and emits a review worklist. It does not edit alias config by itself.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd

from build_fixture_identity_map import base_team, league_tag, load_aliases, canonical_team, markdown_table


DEFAULT_SCORED_ROOT = Path("predictions_output/hybrid_shadow_walkforward_2026_05_01_parity_rebuild")
DEFAULT_API_FIXTURES_DIR = Path("data_sources/api_football/normalized_calendar_year")
DEFAULT_ALIASES = Path("config/team_identity_aliases.csv")
DEFAULT_OUTDIR = Path("reports/latest/fixture_identity_alias_suggestions")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def scored_files(root: Path) -> list[Path]:
    return sorted(root.glob("w*/03_scored/DEPLOY_COMBINED_SCORED_*.csv"))


def load_scored(root: Path, aliases: dict[tuple[str, str], str]) -> pd.DataFrame:
    wanted = {"league", "match_date", "fixture_key", "home_team_name", "away_team_name"}
    frames = []
    for path in scored_files(root):
        frame = pd.read_csv(path, usecols=lambda c: c in wanted, low_memory=False)
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["match_date"] = out["match_date"].astype(str).str[:10]
    out["calendar_year"] = pd.to_datetime(out["match_date"], errors="coerce").dt.year.astype("Int64")
    out["league_tag"] = out["league"].map(league_tag)
    out["scored_home_base"] = out["home_team_name"].map(base_team)
    out["scored_away_base"] = out["away_team_name"].map(base_team)
    out["scored_home_canonical"] = out.apply(lambda r: canonical_team(r.get("home_team_name"), str(r.get("league_tag")), aliases), axis=1)
    out["scored_away_canonical"] = out.apply(lambda r: canonical_team(r.get("away_team_name"), str(r.get("league_tag")), aliases), axis=1)
    return out.dropna(subset=["calendar_year"]).drop_duplicates(["league_tag", "calendar_year", "fixture_key", "match_date", "home_team_name", "away_team_name"])


def load_api(api_dir: Path, league: str, year: int, aliases: dict[tuple[str, str], str]) -> pd.DataFrame:
    path = api_dir / f"fixtures_master__{league}__{year}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return df
    df["match_date"] = df["match_date"].astype(str).str[:10]
    df["api_home_base"] = df["home_team_name"].map(base_team)
    df["api_away_base"] = df["away_team_name"].map(base_team)
    df["api_home_canonical"] = df.apply(lambda r: canonical_team(r.get("home_team_name"), league, aliases), axis=1)
    df["api_away_canonical"] = df.apply(lambda r: canonical_team(r.get("away_team_name"), league, aliases), axis=1)
    return df


def sim(a: Any, b: Any) -> float:
    return SequenceMatcher(None, str(a or ""), str(b or "")).ratio()


def best_team_candidate(scored_name: str, api_names: list[str]) -> tuple[str, float]:
    if not api_names:
        return "", 0.0
    ranked = sorted(((name, sim(scored_name, name)) for name in api_names), key=lambda x: x[1], reverse=True)
    return ranked[0]


def build_suggestions(scored: pd.DataFrame, api_dir: Path, aliases: dict[tuple[str, str], str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    alias_records: list[dict[str, Any]] = []
    scope_records: list[dict[str, Any]] = []
    api_cache: dict[tuple[str, int], pd.DataFrame] = {}
    for (league, year), group in scored.groupby(["league_tag", "calendar_year"], dropna=False):
        league = str(league)
        year = int(year)
        api = api_cache.setdefault((league, year), load_api(api_dir, league, year, aliases))
        if api.empty:
            scope_records.append(
                {
                    "league_tag": league,
                    "calendar_year": year,
                    "scope_issue": "NO_API_FIXTURES_FILE_OR_ROWS",
                    "scored_fixtures": int(group["fixture_key"].nunique()),
                    "api_fixtures": 0,
                }
            )
            continue
        scored_dates = set(group["match_date"].astype(str))
        api_dates = set(api["match_date"].astype(str))
        missing_dates = scored_dates - api_dates
        if missing_dates:
            scope_records.append(
                {
                    "league_tag": league,
                    "calendar_year": year,
                    "scope_issue": "SCORED_DATES_NOT_IN_API",
                    "scored_fixtures": int(group[group["match_date"].isin(missing_dates)]["fixture_key"].nunique()),
                    "api_fixtures": int(api["fixture_key"].nunique()) if "fixture_key" in api.columns else int(len(api)),
                    "sample_dates": ",".join(sorted(missing_dates)[:10]),
                }
            )
        for _, row in group.iterrows():
            same_date = api[api["match_date"].eq(str(row["match_date"]))]
            if same_date.empty:
                continue
            home_match = same_date[same_date["api_home_canonical"].eq(row["scored_home_canonical"])]
            away_match = same_date[same_date["api_away_canonical"].eq(row["scored_away_canonical"])]
            if home_match.empty:
                api_home_names = sorted(set(same_date["api_home_base"].dropna().astype(str)))
                best, score = best_team_candidate(str(row["scored_home_base"]), api_home_names)
                alias_records.append(
                    {
                        "league_tag": league,
                        "provider": "footystats",
                        "provider_team_name": row["home_team_name"],
                        "provider_team_base": row["scored_home_base"],
                        "suggested_canonical_team_name": best,
                        "side": "home",
                        "evidence_type": "same_date_home_slot",
                        "similarity": score,
                        "match_date": row["match_date"],
                    }
                )
            if away_match.empty:
                api_away_names = sorted(set(same_date["api_away_base"].dropna().astype(str)))
                best, score = best_team_candidate(str(row["scored_away_base"]), api_away_names)
                alias_records.append(
                    {
                        "league_tag": league,
                        "provider": "footystats",
                        "provider_team_name": row["away_team_name"],
                        "provider_team_base": row["scored_away_base"],
                        "suggested_canonical_team_name": best,
                        "side": "away",
                        "evidence_type": "same_date_away_slot",
                        "similarity": score,
                        "match_date": row["match_date"],
                    }
                )
    aliases_df = pd.DataFrame(alias_records)
    if not aliases_df.empty:
        summary = (
            aliases_df.groupby(
                ["league_tag", "provider", "provider_team_name", "provider_team_base", "suggested_canonical_team_name", "evidence_type"],
                dropna=False,
            )
            .agg(evidence_rows=("match_date", "size"), avg_similarity=("similarity", "mean"), max_similarity=("similarity", "max"))
            .reset_index()
        )
        summary["review_priority"] = summary.apply(priority, axis=1)
        summary = summary.sort_values(["review_priority", "evidence_rows", "avg_similarity"], ascending=[True, False, False])
    else:
        summary = pd.DataFrame()
    return summary, pd.DataFrame(scope_records)


def priority(row: pd.Series) -> str:
    n = int(row.get("evidence_rows", 0))
    avg = float(row.get("avg_similarity", 0.0))
    if n >= 3 and avg >= 0.78:
        return "P1_PROMOTE_REVIEW"
    if n >= 2 and avg >= 0.68:
        return "P2_REVIEW"
    return "P3_LOW_CONFIDENCE"


def write_report(outdir: Path, aliases: pd.DataFrame, scope: pd.DataFrame) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    aliases.to_csv(outdir / "fixture_identity_alias_suggestions.csv", index=False)
    scope.to_csv(outdir / "fixture_identity_scope_worklist.csv", index=False)
    promoted = aliases[aliases["review_priority"].eq("P1_PROMOTE_REVIEW")] if not aliases.empty else pd.DataFrame()
    lines = [
        "# Fixture Identity Alias Suggestions",
        "",
        f"Generated: `{utc_now()}`",
        "",
        "Research-only. Suggestions are not automatically applied.",
        "",
        "## Suggested Aliases By Priority",
        markdown_table(aliases.head(80) if not aliases.empty else aliases),
        "",
        "## P1 Promote Review",
        markdown_table(promoted.head(80) if not promoted.empty else promoted),
        "",
        "## Scope Worklist",
        markdown_table(scope.head(80) if not scope.empty else scope),
    ]
    meta = {
        "generated_at": utc_now(),
        "suggestions": int(len(aliases)),
        "p1_promote_review": int(len(promoted)),
        "scope_rows": int(len(scope)),
    }
    (outdir / "summary.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-root", default=str(DEFAULT_SCORED_ROOT))
    parser.add_argument("--api-fixtures-dir", default=str(DEFAULT_API_FIXTURES_DIR))
    parser.add_argument("--team-aliases", default=str(DEFAULT_ALIASES))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    args = parser.parse_args()

    aliases = load_aliases(Path(args.team_aliases))
    scored = load_scored(Path(args.scored_root), aliases)
    suggestions, scope = build_suggestions(scored, Path(args.api_fixtures_dir), aliases)
    write_report(Path(args.outdir), suggestions, scope)
    print(f"Suggestions: {len(suggestions)}")
    print(f"P1 promote-review: {int(suggestions['review_priority'].eq('P1_PROMOTE_REVIEW').sum()) if not suggestions.empty else 0}")
    print(f"Scope rows: {len(scope)}")
    print(f"Outputs: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
