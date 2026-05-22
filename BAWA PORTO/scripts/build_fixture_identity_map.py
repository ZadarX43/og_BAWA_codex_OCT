#!/usr/bin/env python3
"""Build a trusted fixture identity map between scored rows and API-Football.

Research-only. This script does not train models, generate predictions, or edit
deploy routing. It maps the existing walk-forward scored fixture spine onto the
local API-Football calendar-year fixture estate with explicit confidence tiers
and scope caveats.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_SCORED_ROOT = Path("predictions_output/hybrid_shadow_walkforward_2026_05_01_parity_rebuild")
DEFAULT_API_FIXTURES_DIR = Path("data_sources/api_football/normalized_calendar_year")
DEFAULT_ALIASES = Path("config/team_identity_aliases.csv")
DEFAULT_SCOPE = Path("config/competition_scope_map.csv")
DEFAULT_OUTDIR = Path("reports/latest/fixture_identity_map")

USABLE_TIERS = {
    "EXACT_DATE_CANONICAL_TEAMS",
    "DATE_PLUS_MINUS_1_CANONICAL_TEAMS",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def markdown_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "_No rows._"
    text = df.head(max_rows).copy()
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
    if len(df) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(df)} rows._")
    return "\n".join(lines)


def league_tag(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def base_team(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    replacements = {
        "man utd": "manchester united",
        "man united": "manchester united",
        "man city": "manchester city",
        "spurs": "tottenham",
        "tottenham hotspur": "tottenham",
        "brighton hove albion": "brighton",
        "wolverhampton wanderers": "wolves",
        "west ham united": "west ham",
        "leeds united": "leeds",
        "newcastle united": "newcastle",
        "nottm forest": "nottingham forest",
        "nottingham forest": "nottingham forest",
        "psg": "paris saint germain",
        "olympique marseille": "marseille",
        "olympique lyonnais": "lyon",
        "borussia monchengladbach": "borussia m gladbach",
        "borussia moenchengladbach": "borussia m gladbach",
        "borussia m gladbach": "borussia m gladbach",
        "borussia mgladbach": "borussia m gladbach",
        "1 fc koln": "koln",
        "fc koln": "koln",
        "athletic club bilbao": "athletic club",
        "athletic bilbao": "athletic club",
    }
    text = replacements.get(text, text)
    parts = [part for part in text.split() if part not in {"fc", "cf", "sc", "afc", "club", "the"}]
    return " ".join(parts)


def load_aliases(path: Path) -> dict[tuple[str, str], str]:
    aliases: dict[tuple[str, str], str] = {}
    if not path.exists():
        return aliases
    df = pd.read_csv(path)
    if "active_flag" in df.columns:
        df = df[pd.to_numeric(df["active_flag"], errors="coerce").fillna(0).astype(int).eq(1)]
    for _, row in df.iterrows():
        league = str(row.get("league_tag") or "").strip()
        provider_name = base_team(row.get("provider_team_name"))
        canonical = base_team(row.get("canonical_team_name"))
        if league and provider_name and canonical:
            aliases[(league, provider_name)] = canonical
            aliases[("ALL", provider_name)] = canonical
    return aliases


def canonical_team(value: Any, league: str, aliases: dict[tuple[str, str], str]) -> str:
    raw = base_team(value)
    return aliases.get((league, raw), aliases.get(("ALL", raw), raw))


def scored_files(root: Path, max_files: int = 0) -> list[Path]:
    files = sorted(root.glob("w*/03_scored/DEPLOY_COMBINED_SCORED_*.csv"))
    if max_files > 0:
        return files[:max_files]
    return files


def load_scored(root: Path, aliases: dict[tuple[str, str], str], max_files: int = 0) -> pd.DataFrame:
    wanted = {"league", "match_date", "fixture_key", "home_team_name", "away_team_name"}
    frames: list[pd.DataFrame] = []
    for path in scored_files(root, max_files=max_files):
        frame = pd.read_csv(path, usecols=lambda c: c in wanted, low_memory=False)
        frame["source_scored_file"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["match_date_dt"] = pd.to_datetime(out["match_date"], errors="coerce")
    out["calendar_year"] = out["match_date_dt"].dt.year.astype("Int64")
    out["league_tag"] = out["league"].map(league_tag)
    out["canonical_home"] = out.apply(lambda r: canonical_team(r.get("home_team_name"), str(r.get("league_tag")), aliases), axis=1)
    out["canonical_away"] = out.apply(lambda r: canonical_team(r.get("away_team_name"), str(r.get("league_tag")), aliases), axis=1)
    out = out.dropna(subset=["calendar_year"]).copy()
    out["calendar_year"] = out["calendar_year"].astype(int)
    return out.drop_duplicates(["league_tag", "calendar_year", "fixture_key", "match_date", "home_team_name", "away_team_name"])


def load_scope(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["league_tag", "calendar_year", "scope_policy", "auto_join_policy", "notes"])
    return pd.read_csv(path).fillna("")


def scope_for(scope: pd.DataFrame, league: str, year: int) -> dict[str, str]:
    if scope.empty:
        return {"scope_policy": "REGULAR_SCOPE", "auto_join_policy": "ALLOW_TRUSTED", "notes": ""}
    year_text = str(year)
    exact = scope[scope["league_tag"].astype(str).eq(league) & scope["calendar_year"].astype(str).eq(year_text)]
    if exact.empty:
        exact = scope[scope["league_tag"].astype(str).eq(league) & scope["calendar_year"].astype(str).str.upper().eq("ALL")]
    if exact.empty:
        return {"scope_policy": "REGULAR_SCOPE", "auto_join_policy": "ALLOW_TRUSTED", "notes": ""}
    row = exact.iloc[0]
    return {
        "scope_policy": str(row.get("scope_policy") or "REGULAR_SCOPE"),
        "auto_join_policy": str(row.get("auto_join_policy") or "ALLOW_TRUSTED"),
        "notes": str(row.get("notes") or ""),
    }


def load_api_fixtures(api_dir: Path, league: str, year: int, aliases: dict[tuple[str, str], str]) -> pd.DataFrame:
    path = api_dir / f"fixtures_master__{league}__{year}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return df
    df["api_match_date_dt"] = pd.to_datetime(df.get("match_date"), errors="coerce")
    df["api_calendar_year"] = df["api_match_date_dt"].dt.year.astype("Int64")
    df["league_tag"] = league
    df["canonical_home"] = df.apply(lambda r: canonical_team(r.get("home_team_name"), league, aliases), axis=1)
    df["canonical_away"] = df.apply(lambda r: canonical_team(r.get("away_team_name"), league, aliases), axis=1)
    return df


def choose_candidate(candidates: pd.DataFrame) -> pd.Series | None:
    if len(candidates) == 1:
        return candidates.iloc[0]
    return None


def resolve_one(row: pd.Series, api: pd.DataFrame, scope_row: dict[str, str]) -> dict[str, Any]:
    base = {
        "league": row.get("league"),
        "league_tag": row.get("league_tag"),
        "calendar_year": int(row.get("calendar_year")),
        "scored_fixture_key": row.get("fixture_key"),
        "scored_match_date": str(row.get("match_date"))[:10],
        "scored_home_team_name": row.get("home_team_name"),
        "scored_away_team_name": row.get("away_team_name"),
        "scored_canonical_home": row.get("canonical_home"),
        "scored_canonical_away": row.get("canonical_away"),
        "api_fixture_id": "",
        "api_fixture_key": "",
        "api_match_date": "",
        "api_home_team_name": "",
        "api_away_team_name": "",
        "api_canonical_home": "",
        "api_canonical_away": "",
        "match_tier": "",
        "auto_usable_flag": False,
        "scope_policy": scope_row["scope_policy"],
        "auto_join_policy": scope_row["auto_join_policy"],
        "coverage_caveat": "",
        "confidence": 0.0,
    }
    if api.empty:
        base.update({"match_tier": "NO_API_SOURCE", "coverage_caveat": "NO_LOCAL_API_FIXTURES", "confidence": 0.0})
        return base
    if str(scope_row["auto_join_policy"]).upper().startswith("EXCLUDE"):
        base.update({"match_tier": "SCOPE_MISMATCH_EXCLUDED", "coverage_caveat": scope_row["scope_policy"], "confidence": 0.0})
        return base

    same_teams = api[
        api["canonical_home"].astype(str).eq(str(row.get("canonical_home")))
        & api["canonical_away"].astype(str).eq(str(row.get("canonical_away")))
    ].copy()
    match_date = row.get("match_date_dt")
    exact = same_teams[same_teams["api_match_date_dt"].dt.date.eq(match_date.date())] if pd.notna(match_date) else pd.DataFrame()
    picked = choose_candidate(exact)
    tier = "EXACT_DATE_CANONICAL_TEAMS"
    confidence = 1.0
    if picked is None:
        near = same_teams[(same_teams["api_match_date_dt"] - match_date).abs().dt.days.le(1)].copy() if pd.notna(match_date) else pd.DataFrame()
        picked = choose_candidate(near)
        tier = "DATE_PLUS_MINUS_1_CANONICAL_TEAMS"
        confidence = 0.94
    if picked is None:
        reversed_teams = api[
            api["canonical_home"].astype(str).eq(str(row.get("canonical_away")))
            & api["canonical_away"].astype(str).eq(str(row.get("canonical_home")))
        ].copy()
        reversed_exact = reversed_teams[reversed_teams["api_match_date_dt"].dt.date.eq(match_date.date())] if pd.notna(match_date) else pd.DataFrame()
        if len(reversed_exact) == 1:
            base.update({"match_tier": "HOME_AWAY_REVERSED_REVIEW", "coverage_caveat": "HOME_AWAY_REVERSED", "confidence": 0.60})
            picked = reversed_exact.iloc[0]
        elif len(exact) > 1 or (pd.notna(match_date) and len(same_teams[(same_teams["api_match_date_dt"] - match_date).abs().dt.days.le(1)]) > 1):
            base.update({"match_tier": "AMBIGUOUS_ALIAS_REVIEW", "coverage_caveat": "MULTIPLE_API_CANDIDATES", "confidence": 0.50})
        else:
            same_date = api[api["api_match_date_dt"].dt.date.eq(match_date.date())] if pd.notna(match_date) else pd.DataFrame()
            caveat = "UNRESOLVED_IDENTITY"
            if not same_date.empty:
                caveat = "TEAM_ALIAS_OR_SCOPE_MISMATCH"
            base.update({"match_tier": "UNRESOLVED", "coverage_caveat": caveat, "confidence": 0.0})

    if picked is not None:
        base.update(
            {
                "api_fixture_id": picked.get("fixture_id", ""),
                "api_fixture_key": picked.get("fixture_key", ""),
                "api_match_date": str(picked.get("match_date", ""))[:10],
                "api_home_team_name": picked.get("home_team_name", ""),
                "api_away_team_name": picked.get("away_team_name", ""),
                "api_canonical_home": picked.get("canonical_home", ""),
                "api_canonical_away": picked.get("canonical_away", ""),
                "match_tier": tier if not base.get("match_tier") else base["match_tier"],
                "confidence": confidence if not base.get("confidence") else base["confidence"],
            }
        )
    usable = base["match_tier"] in USABLE_TIERS
    if scope_row["auto_join_policy"] == "ALLOW_EXACT_ONLY" and base["match_tier"] != "EXACT_DATE_CANONICAL_TEAMS":
        usable = False
    base["auto_usable_flag"] = bool(usable)
    caveats = [str(base.get("coverage_caveat") or "")]
    if scope_row["scope_policy"] != "REGULAR_SCOPE":
        caveats.append(scope_row["scope_policy"])
    base["coverage_caveat"] = "|".join(c for c in caveats if c) or "OK"
    return base


def build_map(scored: pd.DataFrame, api_dir: Path, aliases: dict[tuple[str, str], str], scope: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    api_cache: dict[tuple[str, int], pd.DataFrame] = {}
    for _, row in scored.iterrows():
        league = str(row["league_tag"])
        year = int(row["calendar_year"])
        key = (league, year)
        if key not in api_cache:
            api_cache[key] = load_api_fixtures(api_dir, league, year, aliases)
        records.append(resolve_one(row, api_cache[key], scope_for(scope, league, year)))
    return pd.DataFrame(records)


def write_report(outdir: Path, *, scored_root: Path, api_dir: Path, aliases_path: Path, scope_path: Path, mapping: pd.DataFrame) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(outdir / "fixture_identity_map.csv", index=False)
    usable = mapping[mapping["auto_usable_flag"].astype(bool)] if not mapping.empty else pd.DataFrame()
    tier_summary = (
        mapping.groupby(["match_tier", "auto_usable_flag"], dropna=False)
        .agg(fixtures=("scored_fixture_key", "nunique"), rows=("scored_fixture_key", "size"))
        .reset_index()
        .sort_values(["auto_usable_flag", "fixtures"], ascending=[False, False])
        if not mapping.empty
        else pd.DataFrame()
    )
    league_summary = (
        mapping.groupby(["league", "calendar_year"], dropna=False)
        .agg(
            fixtures=("scored_fixture_key", "nunique"),
            usable_fixtures=("auto_usable_flag", lambda s: int(pd.Series(s).astype(bool).sum())),
            avg_confidence=("confidence", "mean"),
        )
        .reset_index()
        if not mapping.empty
        else pd.DataFrame()
    )
    if not league_summary.empty:
        league_summary["usable_rate"] = league_summary["usable_fixtures"] / league_summary["fixtures"]
        league_summary = league_summary.sort_values(["usable_rate", "fixtures"], ascending=[True, False])
    caveats = (
        mapping.groupby("coverage_caveat", dropna=False)
        .agg(fixtures=("scored_fixture_key", "nunique"))
        .reset_index()
        .sort_values("fixtures", ascending=False)
        if not mapping.empty
        else pd.DataFrame()
    )
    tier_summary.to_csv(outdir / "fixture_identity_tier_summary.csv", index=False)
    league_summary.to_csv(outdir / "fixture_identity_league_summary.csv", index=False)
    caveats.to_csv(outdir / "fixture_identity_caveat_summary.csv", index=False)
    total = int(mapping["scored_fixture_key"].nunique()) if not mapping.empty else 0
    usable_count = int(usable["scored_fixture_key"].nunique()) if not usable.empty else 0
    lines = [
        "# Fixture Identity Map",
        "",
        f"Generated: `{utc_now()}`",
        "",
        "Research-only. No deploy gates, prediction outputs, or provider-season files changed.",
        "",
        "## Inputs",
        f"- scored root: `{scored_root}`",
        f"- API fixtures dir: `{api_dir}`",
        f"- aliases: `{aliases_path}`",
        f"- competition scope: `{scope_path}`",
        "",
        "## Trusted Coverage",
        f"- total unique scored fixtures: `{total}`",
        f"- auto-usable mapped fixtures: `{usable_count}`",
        f"- auto-usable rate: `{usable_count / total:.4f}`" if total else "- auto-usable rate: `n/a`",
        "",
        "## Match Tier Summary",
        markdown_table(tier_summary),
        "",
        "## Lowest League Coverage",
        markdown_table(league_summary.head(40)),
        "",
        "## Caveat Summary",
        markdown_table(caveats),
        "",
        "## Interpretation",
        "- Use only `auto_usable_flag=true` rows for trusted sidecar joins.",
        "- `SPLIT_STAGE_REVIEW`, `PLAYOFF_SCOPE_REVIEW`, and `CUP_SCOPE_REVIEW` are not failures; they are source-scope decisions.",
        "- `AMBIGUOUS_ALIAS_REVIEW`, `HOME_AWAY_REVERSED_REVIEW`, and `UNRESOLVED` should not be auto-joined.",
    ]
    meta = {
        "generated_at": utc_now(),
        "scored_root": str(scored_root),
        "api_dir": str(api_dir),
        "aliases_path": str(aliases_path),
        "scope_path": str(scope_path),
        "total_unique_scored_fixtures": total,
        "auto_usable_mapped_fixtures": usable_count,
        "auto_usable_rate": usable_count / total if total else None,
    }
    (outdir / "summary.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-root", default=str(DEFAULT_SCORED_ROOT))
    parser.add_argument("--api-fixtures-dir", default=str(DEFAULT_API_FIXTURES_DIR))
    parser.add_argument("--team-aliases", default=str(DEFAULT_ALIASES))
    parser.add_argument("--competition-scope", default=str(DEFAULT_SCOPE))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--max-files", type=int, default=0)
    args = parser.parse_args()

    aliases_path = Path(args.team_aliases)
    scope_path = Path(args.competition_scope)
    aliases = load_aliases(aliases_path)
    scope = load_scope(scope_path)
    scored = load_scored(Path(args.scored_root), aliases, max_files=args.max_files)
    if scored.empty:
        raise SystemExit(f"No scored fixtures found under {args.scored_root}")
    mapping = build_map(scored, Path(args.api_fixtures_dir), aliases, scope)
    write_report(
        Path(args.outdir),
        scored_root=Path(args.scored_root),
        api_dir=Path(args.api_fixtures_dir),
        aliases_path=aliases_path,
        scope_path=scope_path,
        mapping=mapping,
    )
    total = mapping["scored_fixture_key"].nunique()
    usable = mapping[mapping["auto_usable_flag"].astype(bool)]["scored_fixture_key"].nunique()
    print(f"Scored fixtures: {total}")
    print(f"Trusted mapped fixtures: {usable}")
    print(f"Trusted coverage: {usable / total:.4f}" if total else "Trusted coverage: n/a")
    print(f"Outputs: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
