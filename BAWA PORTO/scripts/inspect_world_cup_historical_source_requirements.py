#!/usr/bin/env python3
"""Inspect World Cup historical source requirements for player-power validation.

Research only. Default mode is offline/dry-run: it audits local files and writes
the exact API/source request plan for 2018/2022. With `--execute-api-probe`, it
performs a small API-Football availability probe and caches the probe payloads.

This does not train models, alter production routing, or write ModelStore files.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.api_football.client import APIFootballClient  # noqa: E402


WORLD_CUP_LEAGUE_ID = 1
DEFAULT_SEASONS = "2018,2022"
DEFAULT_RAW = Path("data_sources/api_football/raw")
DEFAULT_DROP = Path("/Users/hughwade/Desktop/FOOTYSTATS_DROP")
DEFAULT_OUTDIR = Path("data_sources/footystats_world_cup/historical_source_inspection")
QUALIFIER_GAPS = {
    2018: ["WC Qualification CONCACAF", "WC Qualification OFC"],
    2022: ["WC Qualification OFC"],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def markdown_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "_No rows._"
    out = df.head(max_rows).copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            out[col] = out[col].astype("string").fillna("")
    lines = [
        "| " + " | ".join(out.columns) + " |",
        "| " + " | ".join(["---"] * len(out.columns)) + " |",
    ]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in out.columns) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(df)} rows._")
    return "\n".join(lines)


def count_jsonl_results(path: Path, response_key: str = "response") -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    payloads = 0
    rows = 0
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payloads += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            response = data.get(response_key)
            if isinstance(response, list):
                rows += len(response)
    return payloads, rows


def get_path(payload: dict[str, Any], *parts: str, default: Any = None) -> Any:
    node: Any = payload
    for part in parts:
        if not isinstance(node, dict):
            return default
        node = node.get(part)
    return default if node is None else node


def coverage_flat(payload: dict[str, Any]) -> dict[str, Any]:
    response = payload.get("response") or []
    if not response:
        return {}
    coverage = ((response[0] or {}).get("coverage") or {})
    fixtures = coverage.get("fixtures") or {}
    return {
        "coverage_events": bool(fixtures.get("events")),
        "coverage_lineups": bool(fixtures.get("lineups")),
        "coverage_fixture_statistics": bool(fixtures.get("statistics_fixtures")),
        "coverage_player_statistics": bool(fixtures.get("statistics_players")),
        "coverage_players": bool(coverage.get("players")),
        "coverage_injuries": bool(coverage.get("injuries")),
        "coverage_standings": bool(coverage.get("standings")),
        "coverage_odds": bool(coverage.get("odds")),
        "coverage_predictions": bool(coverage.get("predictions")),
    }


def write_raw(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=True) + "\n")


def scan_footystats(drop: Path) -> pd.DataFrame:
    rows = []
    if not drop.exists():
        return pd.DataFrame(columns=["path", "name", "kind", "years"])
    for path in drop.rglob("*.csv"):
        name = path.name.lower()
        years = sorted({int(y) for y in re.findall(r"(?:19|20)\d{2}", name)})
        if "players" in name:
            kind = "players"
        elif "teams" in name:
            kind = "teams"
        elif "matches" in name:
            kind = "matches"
        elif "squads" in name:
            kind = "squads"
        else:
            kind = "other"
        rows.append({"path": str(path), "name": name, "kind": kind, "years": years})
    return pd.DataFrame(rows)


def archive_squad_paths_for_season(drop: Path, season: int) -> list[str]:
    path = drop / "archive" / "squads.csv"
    if not path.exists():
        return []
    try:
        squads = pd.read_csv(path, usecols=["tournament_id"], low_memory=False)
    except Exception:
        return []
    target = f"WC-{season}"
    if squads["tournament_id"].astype(str).eq(target).any():
        return [str(path)]
    return []


def file_matches(files: pd.DataFrame, season: int, tokens: list[str], kind: str | None = None) -> list[str]:
    if files.empty:
        return []
    def year_ok(years: object) -> bool:
        if not isinstance(years, list):
            return False
        if season == 2018:
            return 2018 in years or (2016 in years and 2018 in years)
        return season in years

    mask = files["name"].map(lambda name: all(token in str(name) for token in tokens))
    mask &= files["years"].map(year_ok)
    if kind:
        mask &= files["kind"].eq(kind)
    return sorted(files.loc[mask, "path"].astype(str).tolist())


def local_api_cache_rows(raw_dir: Path, season: int) -> list[dict[str, Any]]:
    specs = [
        ("fixtures", raw_dir / f"fixtures__league_1__season_{season}__fixtures.jsonl", "Fixture schedule"),
        ("players", raw_dir / f"players__league_1__season_{season}__players.jsonl", "World Cup roster/player endpoint"),
        (
            "fixtures_players",
            raw_dir / f"fixtures__league_1__season_{season}__fixtures_players.jsonl",
            "Fixture-player ratings/stats bundle",
        ),
        ("injuries", raw_dir / f"fixtures__league_1__season_{season}__injuries.jsonl", "Injury/suspension endpoint cache"),
    ]
    rows = []
    for layer, path, label in specs:
        payloads, response_rows = count_jsonl_results(path)
        rows.append(
            {
                "season": season,
                "source_family": "api_football_local_cache",
                "layer": layer,
                "label": label,
                "local_path": str(path),
                "exists": int(path.exists()),
                "payloads": payloads,
                "response_rows": response_rows,
                "current_status": "FOUND_LOCAL_CACHE" if path.exists() and response_rows > 0 else "MISSING_LOCAL_CACHE",
            }
        )
    return rows


def build_api_request_plan(seasons: list[int], raw_dir: Path) -> pd.DataFrame:
    rows = []
    for season in seasons:
        rows.extend(
            [
                {
                    "season": season,
                    "endpoint": "/leagues",
                    "params": f"id=1&season={season}",
                    "purpose": "Verify advertised coverage flags for players, injuries, lineups, fixture player stats.",
                    "cache_target": str(raw_dir / f"league__id_1__season_{season}__coverage_probe.jsonl"),
                    "priority": "P0",
                },
                {
                    "season": season,
                    "endpoint": "/fixtures",
                    "params": f"league=1&season={season}",
                    "purpose": "Confirm fixture ids and round/date spine for historical World Cup.",
                    "cache_target": str(raw_dir / f"fixtures__league_1__season_{season}__fixtures.jsonl"),
                    "priority": "P0",
                },
                {
                    "season": season,
                    "endpoint": "/players",
                    "params": f"league=1&season={season}&page=1..N",
                    "purpose": "Fetch roster/player endpoint; use only if source represents tournament squad/player pool, not post-match performance.",
                    "cache_target": str(raw_dir / f"players__league_1__season_{season}__players.jsonl"),
                    "priority": "P0",
                },
                {
                    "season": season,
                    "endpoint": "/injuries",
                    "params": f"league=1&season={season}",
                    "purpose": "Check injury/suspension rows. Historical first-seen timestamp cannot be inferred unless provider supplies it.",
                    "cache_target": str(raw_dir / f"fixtures__league_1__season_{season}__injuries.jsonl"),
                    "priority": "P0",
                },
                {
                    "season": season,
                    "endpoint": "/fixtures/players",
                    "params": "fixture=<each_fixture_id>",
                    "purpose": "Fetch per-match player ratings/stats. For modelling, lag strictly by prior completed matches only.",
                    "cache_target": str(raw_dir / f"fixtures__league_1__season_{season}__fixtures_players.jsonl"),
                    "priority": "P1",
                },
                {
                    "season": season,
                    "endpoint": "/fixtures",
                    "params": "ids=<up_to_20_fixture_ids>",
                    "purpose": "Fetch fixture detail bundles with embedded events, lineups, statistics, players when available.",
                    "cache_target": str(raw_dir / f"fixtures__league_1__season_{season}__fixtures_bundle.jsonl"),
                    "priority": "P1",
                },
            ]
        )
    return pd.DataFrame(rows)


def execute_api_probe(seasons: list[int], raw_dir: Path, outdir: Path, sleep_seconds: float, daily_cap: int) -> pd.DataFrame:
    client = APIFootballClient(sleep_seconds=sleep_seconds, daily_cap=daily_cap)
    rows = []
    for season in seasons:
        league_payload = client.get_json("/leagues", {"id": WORLD_CUP_LEAGUE_ID, "season": season})
        fixtures_payload = client.get_json("/fixtures", {"league": WORLD_CUP_LEAGUE_ID, "season": season})
        players_payload = client.get_json("/players", {"league": WORLD_CUP_LEAGUE_ID, "season": season, "page": 1})
        injuries_payload = client.get_json("/injuries", {"league": WORLD_CUP_LEAGUE_ID, "season": season})

        write_raw(raw_dir / f"league__id_1__season_{season}__coverage_probe.jsonl", league_payload)
        write_raw(raw_dir / f"players__league_1__season_{season}__players_probe_page1.jsonl", players_payload)
        write_raw(raw_dir / f"fixtures__league_1__season_{season}__injuries_probe.jsonl", injuries_payload)

        fixtures = fixtures_payload.get("response") or []
        first_fixture_id = get_path(fixtures[0], "fixture", "id") if fixtures else None
        detail_results = None
        fixture_player_results = None
        fixture_player_rows = None
        if first_fixture_id:
            detail_payload = client.get_json("/fixtures", {"id": first_fixture_id})
            fixture_players_payload = client.get_json("/fixtures/players", {"fixture": first_fixture_id})
            write_raw(raw_dir / f"fixtures__league_1__season_{season}__fixture_detail_probe.jsonl", detail_payload)
            write_raw(raw_dir / f"fixtures__league_1__season_{season}__fixtures_players_probe.jsonl", fixture_players_payload)
            detail_results = int(detail_payload.get("results") or 0)
            fixture_player_results = int(fixture_players_payload.get("results") or 0)
            fixture_player_rows = sum(len(team.get("players") or []) for team in (fixture_players_payload.get("response") or []))

        rows.append(
            {
                "season": season,
                "api_probe_ts_utc": utc_now(),
                "league_results": int(league_payload.get("results") or 0),
                "fixtures_results": int(fixtures_payload.get("results") or 0),
                "players_page1_results": int(players_payload.get("results") or 0),
                "players_pages_total": int(get_path(players_payload, "paging", "total", default=0) or 0),
                "injuries_results": int(injuries_payload.get("results") or 0),
                "sample_fixture_id": first_fixture_id,
                "sample_fixture_detail_results": detail_results,
                "sample_fixture_players_results": fixture_player_results,
                "sample_fixture_players_player_rows": fixture_player_rows,
                **coverage_flat(league_payload),
            }
        )
    probe = pd.DataFrame(rows)
    probe.to_csv(outdir / "api_football_world_cup_2018_2022_probe_results.csv", index=False)
    return probe


def build_source_manifest(seasons: list[int], raw_dir: Path, drop: Path, outdir: Path, api_probe: pd.DataFrame | None) -> None:
    files = scan_footystats(drop)
    local_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for season in seasons:
        local_rows.extend(local_api_cache_rows(raw_dir, season))
        for comp_label, tokens in [
            ("World Cup player stats", ["fifa-world-cup", "players"]),
            ("World Cup team stats", ["fifa-world-cup", "teams"]),
            ("World Cup match stats", ["fifa-world-cup", "matches"]),
            ("Qualifier player stats", ["wc-qualification", "players"]),
            ("Qualifier match stats", ["wc-qualification", "matches"]),
            ("Fjelstul squads archive", ["squads"]),
        ]:
            found = file_matches(files, season, tokens)
            if comp_label == "Fjelstul squads archive":
                found = sorted(set(found + archive_squad_paths_for_season(drop, season)))
            manifest_rows.append(
                {
                    "season": season,
                    "source_family": "footystats_or_archive",
                    "layer": comp_label,
                    "found_files": " | ".join(found),
                    "found_count": len(found),
                    "current_status": "FOUND" if found else "MISSING",
                    "timestamp_safe_use": (
                        "YES_IF_LAGGED_OR_PRE_TOURNAMENT"
                        if "Qualifier" in comp_label or "squads" in comp_label.lower()
                        else "NO_IF_FINAL_TOURNAMENT_AGGREGATE"
                    ),
                }
            )

        for missing_comp in QUALIFIER_GAPS.get(season, []):
            manifest_rows.append(
                {
                    "season": season,
                    "source_family": "documented_exclusion_or_source_gap",
                    "layer": missing_comp,
                    "found_files": "",
                    "found_count": 0,
                    "current_status": "DOCUMENT_GAP_OR_SOURCE_IF_FOUND",
                    "timestamp_safe_use": "EXCLUDE_FROM_CLAIMS_UNTIL_SOURCED",
                }
            )

        for layer, action in [
            (
                "dated_external_player_rating_or_market_value_snapshot",
                "Find pre-tournament player rating/value data keyed by player/team/club with source date before first World Cup kickoff.",
            ),
            (
                "fifa_ranking_or_elo_snapshot",
                "Source dated FIFA ranking or Elo snapshots before each fixture; attach country aliases.",
            ),
            (
                "player_identity_alias_map",
                "Build alias map across API-Football player ids, FootyStats names, Fjelstul squads, and external rating names.",
            ),
            (
                "injury_first_seen_registry",
                "Historical API pulls can prove rows exist, but first-seen safety needs cached timestamp or source timestamp.",
            ),
        ]:
            manifest_rows.append(
                {
                    "season": season,
                    "source_family": "required_external_or_contract_layer",
                    "layer": layer,
                    "found_files": "",
                    "found_count": 0,
                    "current_status": "NEEDS_SOURCE_OR_CONTRACT",
                    "timestamp_safe_use": action,
                }
            )

    local_cache = pd.DataFrame(local_rows)
    manifest = pd.DataFrame(manifest_rows)
    api_plan = build_api_request_plan(seasons, raw_dir)

    if api_probe is not None and not api_probe.empty:
        for _, probe_row in api_probe.iterrows():
            season = int(probe_row["season"])
            for layer, col in [
                ("api_coverage_players", "coverage_players"),
                ("api_coverage_player_statistics", "coverage_player_statistics"),
                ("api_coverage_injuries", "coverage_injuries"),
                ("api_players_page1", "players_page1_results"),
                ("api_injuries", "injuries_results"),
                ("api_sample_fixture_player_rows", "sample_fixture_players_player_rows"),
            ]:
                value = probe_row.get(col)
                manifest.loc[len(manifest)] = {
                    "season": season,
                    "source_family": "api_football_live_probe",
                    "layer": layer,
                    "found_files": str(value),
                    "found_count": int(value) if pd.notna(value) and str(value).replace(".", "", 1).isdigit() else int(bool(value)),
                    "current_status": "PROBE_PRESENT" if bool(value) else "PROBE_EMPTY_OR_FALSE",
                    "timestamp_safe_use": "AVAILABILITY_ONLY_REQUIRES_LAG_OR_FIRST_SEEN_CONTRACT",
                }

    outdir.mkdir(parents=True, exist_ok=True)
    local_cache.to_csv(outdir / "world_cup_historical_api_local_cache_status.csv", index=False)
    api_plan.to_csv(outdir / "world_cup_historical_api_request_plan.csv", index=False)
    manifest.to_csv(outdir / "world_cup_historical_source_requirement_manifest.csv", index=False)

    source_summary = (
        manifest.groupby(["season", "source_family", "current_status"], dropna=False)
        .agg(layers=("layer", "count"), files=("found_count", "sum"))
        .reset_index()
        .sort_values(["season", "source_family", "current_status"])
    )
    source_summary.to_csv(outdir / "world_cup_historical_source_requirement_summary.csv", index=False)

    critical = manifest[
        manifest["current_status"].astype(str).isin(["MISSING", "NEEDS_SOURCE_OR_CONTRACT", "DOCUMENT_GAP_OR_SOURCE_IF_FOUND"])
    ].copy()
    md = [
        "# World Cup Historical Source Inspection",
        "",
        f"Generated: `{utc_now()}`",
        "",
        "Research-only inspection for 2018/2022 World Cup player-power validation sources.",
        "",
        "## Source Summary",
        "",
        markdown_table(source_summary),
        "",
        "## Local API Cache Status",
        "",
        markdown_table(local_cache),
        "",
        "## Critical Gaps / Actions",
        "",
        markdown_table(critical[["season", "source_family", "layer", "current_status", "timestamp_safe_use"]]),
        "",
        "## API Request Plan",
        "",
        markdown_table(api_plan),
        "",
        "## Guardrails",
        "",
        "- API-Football historical `/fixtures/players` rows are post-match actuals; use them only as prior-match player form after lagging by kickoff.",
        "- API-Football historical `/injuries` rows are not automatically timestamp-safe unless the provider payload includes a first-seen/source timestamp or we have a cached pull timestamp before kickoff.",
        "- Final FootyStats World Cup player/team aggregates are not pre-match features for that same tournament unless converted to prior-match rolling values.",
        "- OFC 2018/2022 and CONCACAF 2018 should be explicit exclusions if source files cannot be found.",
        "",
        "## Outputs",
        "",
        f"- Local API cache status: `{outdir / 'world_cup_historical_api_local_cache_status.csv'}`",
        f"- API request plan: `{outdir / 'world_cup_historical_api_request_plan.csv'}`",
        f"- Source manifest: `{outdir / 'world_cup_historical_source_requirement_manifest.csv'}`",
        f"- Source summary: `{outdir / 'world_cup_historical_source_requirement_summary.csv'}`",
    ]
    if api_probe is not None:
        md.extend(["", f"- API probe results: `{outdir / 'api_football_world_cup_2018_2022_probe_results.csv'}`"])
    (outdir / "SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[ok] manifest_rows={len(manifest)} api_plan={len(api_plan)}")
    print(f"[ok] wrote {outdir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", default=DEFAULT_SEASONS)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--footystats-drop", type=Path, default=DEFAULT_DROP)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--execute-api-probe", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--daily-cap", type=int, default=75000)
    args = parser.parse_args()

    seasons = [int(item.strip()) for item in str(args.seasons).split(",") if item.strip()]
    args.outdir.mkdir(parents=True, exist_ok=True)
    api_probe = None
    if args.execute_api_probe:
        api_probe = execute_api_probe(seasons, args.raw_dir, args.outdir, args.sleep_seconds, args.daily_cap)
    build_source_manifest(seasons, args.raw_dir, args.footystats_drop, args.outdir, api_probe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
