#!/usr/bin/env python3
"""Bridge FootyStats World Cup match spine to API-Football fixture identities.

Research-only. This creates fixture/team identity joins for the World Cup
estate without changing the normal FootyStats ingest or production pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_MATCH_SPINE = Path("data_sources/footystats_world_cup/research_foundation/footystats_world_cup_match_spine.csv")
DEFAULT_API_RAW_DIR = Path("data_sources/api_football/raw")
DEFAULT_OUTDIR = Path("data_sources/footystats_world_cup/api_bridge")
DEFAULT_SEASONS = "2006,2010,2014,2018,2022,2026"

ALIASES = {
    "usmnt": "usa",
    "united states": "usa",
    "u s a": "usa",
    "cote d ivoire": "ivory coast",
    "côte d ivoire": "ivory coast",
    "côte d’ivoire": "ivory coast",
    "korea republic": "south korea",
    "bosnia herzegovina": "bosnia and herzegovina",
}


def norm_team(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+national\s+team$", "", text)
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return ALIASES.get(text, text)


def read_api_fixtures(raw_dir: Path, season: int) -> pd.DataFrame:
    path = raw_dir / f"fixtures__league_1__season_{season}__fixtures.jsonl"
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return pd.DataFrame()
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.strip():
                continue
            payload = json.loads(line)
            for item in payload.get("response", []) or []:
                fixture = item.get("fixture") or {}
                league = item.get("league") or {}
                teams = item.get("teams") or {}
                home = teams.get("home") or {}
                away = teams.get("away") or {}
                venue = fixture.get("venue") or {}
                rows.append(
                    {
                        "season": season,
                        "api_fixture_id": fixture.get("id"),
                        "api_timestamp": fixture.get("timestamp"),
                        "api_date": fixture.get("date"),
                        "api_status_short": (fixture.get("status") or {}).get("short"),
                        "api_round": league.get("round"),
                        "api_home_team_id": home.get("id"),
                        "api_home_team_name": home.get("name"),
                        "api_away_team_id": away.get("id"),
                        "api_away_team_name": away.get("name"),
                        "api_venue_id": venue.get("id"),
                        "api_venue_name": venue.get("name"),
                        "api_venue_city": venue.get("city"),
                        "api_home_norm": norm_team(home.get("name")),
                        "api_away_norm": norm_team(away.get("name")),
                    }
                )
    return pd.DataFrame(rows)


def build_bridge(fs: pd.DataFrame, api: pd.DataFrame) -> pd.DataFrame:
    if fs.empty:
        return pd.DataFrame()
    fs = fs.copy()
    fs["fs_home_norm"] = fs["home_team_name"].map(norm_team)
    fs["fs_away_norm"] = fs["away_team_name"].map(norm_team)
    fs["fs_timestamp"] = pd.to_numeric(fs["timestamp"], errors="coerce").astype("Int64")
    if api.empty:
        out = fs.copy()
        out["join_status"] = "NO_LOCAL_API_FIXTURES"
        return out

    api = api.copy()
    api["api_timestamp_int"] = pd.to_numeric(api["api_timestamp"], errors="coerce").astype("Int64")
    exact = fs.merge(
        api,
        left_on=["season", "fs_timestamp", "fs_home_norm", "fs_away_norm"],
        right_on=["season", "api_timestamp_int", "api_home_norm", "api_away_norm"],
        how="left",
        suffixes=("", "_api"),
    )
    exact["join_status"] = exact["api_fixture_id"].notna().map({True: "MATCH_EXACT_TS_TEAMS", False: "NO_MATCH"})

    # Fallback for any source with tiny timestamp differences: same teams, same date.
    need = exact["join_status"].eq("NO_MATCH")
    if need.any():
        fs_missing = exact.loc[need, fs.columns.tolist()].copy()
        fs_missing["fs_date"] = pd.to_datetime(fs_missing["kickoff_dt"], errors="coerce").dt.date.astype(str)
        api["api_date_only"] = pd.to_datetime(api["api_date"], errors="coerce").dt.date.astype(str)
        fallback = fs_missing.merge(
            api,
            left_on=["season", "fs_date", "fs_home_norm", "fs_away_norm"],
            right_on=["season", "api_date_only", "api_home_norm", "api_away_norm"],
            how="left",
            suffixes=("", "_api"),
        )
        fallback["join_status"] = fallback["api_fixture_id"].notna().map({True: "MATCH_DATE_TEAMS", False: "NO_MATCH"})
        if not fallback.empty:
            exact = pd.concat([exact.loc[~need], fallback], ignore_index=True, sort=False)

    return exact


def api_schedule_only(api: pd.DataFrame) -> pd.DataFrame:
    if api.empty:
        return api
    keep = [
        "season",
        "api_fixture_id",
        "api_timestamp",
        "api_date",
        "api_status_short",
        "api_round",
        "api_home_team_id",
        "api_home_team_name",
        "api_away_team_id",
        "api_away_team_name",
        "api_venue_name",
        "api_venue_city",
    ]
    return api[[col for col in keep if col in api.columns]].copy()


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            view[col] = view[col].astype("string").fillna("")
    lines = [
        "| " + " | ".join(view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "/") for col in view.columns) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-spine", default=str(DEFAULT_MATCH_SPINE))
    parser.add_argument("--api-raw-dir", default=str(DEFAULT_API_RAW_DIR))
    parser.add_argument("--seasons", default=DEFAULT_SEASONS)
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    args = parser.parse_args()

    seasons = [int(item.strip()) for item in args.seasons.split(",") if item.strip()]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fs_all = pd.read_csv(args.match_spine, low_memory=False)

    bridge_frames: list[pd.DataFrame] = []
    schedule_frames: list[pd.DataFrame] = []
    for season in seasons:
        api = read_api_fixtures(Path(args.api_raw_dir), season)
        if not api.empty:
            schedule_frames.append(api_schedule_only(api))
        fs = fs_all[fs_all["season"].eq(season)].copy()
        if not fs.empty:
            bridge_frames.append(build_bridge(fs, api))

    bridge = pd.concat(bridge_frames, ignore_index=True, sort=False) if bridge_frames else pd.DataFrame()
    schedule = pd.concat(schedule_frames, ignore_index=True, sort=False) if schedule_frames else pd.DataFrame()

    bridge.to_csv(outdir / "world_cup_footystats_api_fixture_bridge.csv", index=False)
    schedule.to_csv(outdir / "world_cup_api_fixture_schedule.csv", index=False)

    summary = (
        bridge.groupby(["season", "join_status"], dropna=False)
        .agg(rows=("fixture_key", "size"))
        .reset_index()
        .sort_values(["season", "join_status"])
        if not bridge.empty
        else pd.DataFrame(columns=["season", "join_status", "rows"])
    )
    totals = (
        bridge.groupby("season", dropna=False)
        .agg(
            footystats_matches=("fixture_key", "size"),
            joined_api_fixtures=("api_fixture_id", lambda s: int(s.notna().sum())),
            unique_api_home_teams=("api_home_team_id", "nunique"),
            unique_api_away_teams=("api_away_team_id", "nunique"),
        )
        .reset_index()
        if not bridge.empty
        else pd.DataFrame()
    )
    if not totals.empty:
        totals["join_rate"] = totals["joined_api_fixtures"] / totals["footystats_matches"]
    summary.to_csv(outdir / "world_cup_footystats_api_bridge_join_status.csv", index=False)
    totals.to_csv(outdir / "world_cup_footystats_api_bridge_summary.csv", index=False)

    schedule_summary = (
        schedule.groupby(["season", "api_round"], dropna=False)
        .agg(fixtures=("api_fixture_id", "size"))
        .reset_index()
        .sort_values(["season", "api_round"])
        if not schedule.empty
        else pd.DataFrame()
    )
    schedule_summary.to_csv(outdir / "world_cup_api_schedule_round_summary.csv", index=False)

    lines = [
        "# World Cup FootyStats/API-Football Bridge",
        "",
        "Research-only fixture/team identity bridge.",
        "",
        "## Join Totals",
        markdown_table(totals),
        "",
        "## Join Status",
        markdown_table(summary),
        "",
        "## API Schedule Round Summary",
        markdown_table(schedule_summary),
        "",
        "## Outputs",
        f"- `{outdir / 'world_cup_footystats_api_fixture_bridge.csv'}`",
        f"- `{outdir / 'world_cup_api_fixture_schedule.csv'}`",
        f"- `{outdir / 'world_cup_footystats_api_bridge_summary.csv'}`",
        f"- `{outdir / 'world_cup_footystats_api_bridge_join_status.csv'}`",
        f"- `{outdir / 'world_cup_api_schedule_round_summary.csv'}`",
    ]
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Bridge rows: {len(bridge)}")
    print(f"API schedule rows: {len(schedule)}")
    print(f"Output: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
