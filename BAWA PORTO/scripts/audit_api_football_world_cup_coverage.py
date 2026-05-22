#!/usr/bin/env python3
"""Audit API-Football World Cup coverage across historical tournament seasons.

Research-only. Default mode is dry-run so the request footprint is reviewable
before spending API budget.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.api_football.client import APIFootballClient


DEFAULT_SEASONS = "2006,2010,2014,2018,2022,2026"
DEFAULT_OUTDIR = Path("reports/latest/world_cup_api_football_coverage_audit")
WORLD_CUP_LEAGUE_ID = 1
OFFICIAL_EXPECTED_FIXTURES = {2006: 64, 2010: 64, 2014: 64, 2018: 64, 2022: 64, 2026: 104}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def get_path(payload: dict[str, Any], *parts: str, default: Any = None) -> Any:
    node: Any = payload
    for part in parts:
        if not isinstance(node, dict):
            return default
        node = node.get(part)
    return default if node is None else node


def safe_int(value: Any, default: int = 0) -> int:
    if value is None or pd.isna(value):
        return default
    try:
        return int(value)
    except Exception:
        return default


def coverage_flat(league_payload: dict[str, Any]) -> dict[str, Any]:
    response = league_payload.get("response") or []
    if not response:
        return {}
    coverage = ((response[0] or {}).get("coverage") or {})
    fixtures = coverage.get("fixtures") or {}
    return {
        "coverage_fixtures_events": bool(fixtures.get("events")),
        "coverage_fixtures_lineups": bool(fixtures.get("lineups")),
        "coverage_fixtures_statistics_fixtures": bool(fixtures.get("statistics_fixtures")),
        "coverage_fixtures_statistics_players": bool(fixtures.get("statistics_players")),
        "coverage_standings": bool(coverage.get("standings")),
        "coverage_players": bool(coverage.get("players")),
        "coverage_top_scorers": bool(coverage.get("top_scorers")),
        "coverage_top_assists": bool(coverage.get("top_assists")),
        "coverage_top_cards": bool(coverage.get("top_cards")),
        "coverage_injuries": bool(coverage.get("injuries")),
        "coverage_predictions": bool(coverage.get("predictions")),
        "coverage_odds": bool(coverage.get("odds")),
    }


def markdown_table(df: pd.DataFrame, max_rows: int = 60) -> str:
    if df.empty:
        return "_No rows._"
    text = df.head(max_rows).copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            text[col] = text[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
        else:
            text[col] = text[col].astype("string").fillna("")
    lines = [
        "| " + " | ".join(str(col) for col in text.columns) + " |",
        "| " + " | ".join(["---"] * len(text.columns)) + " |",
    ]
    for _, row in text.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in text.columns) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(df)} rows._")
    return "\n".join(lines)


def finalize_execute_outputs(out: pd.DataFrame, outdir: Path) -> None:
    source_read = []
    for _, row in out.iterrows():
        flags = [
            bool(row.get("coverage_fixtures_events")),
            bool(row.get("coverage_fixtures_lineups")),
            bool(row.get("coverage_fixtures_statistics_players")),
            bool(row.get("coverage_players")),
        ]
        detail_player_rows = safe_int(row.get("sample_fixture_players_player_rows"))
        detail_lineups = safe_int(row.get("sample_detail_lineups"))
        fixture_results = safe_int(row.get("fixtures_results"))
        if fixture_results > 0 and all(flags):
            readiness = "PLAYER_FIXTURE_INTEL_READY_TO_BACKFILL"
        elif fixture_results > 0 and (detail_player_rows > 0 or detail_lineups > 0):
            readiness = "PLAYER_DETAIL_PROBE_PRESENT_COVERAGE_FLAG_FALSE"
        elif fixture_results > 0:
            readiness = "FIXTURE_READY_PLAYER_INTEL_PARTIAL"
        else:
            readiness = "NO_FIXTURE_SOURCE"
        source_read.append(readiness)
    out["readiness_bucket"] = source_read
    out.to_csv(outdir / "world_cup_api_football_coverage.csv", index=False)

    summary = [
        "# API-Football World Cup Coverage Audit",
        "",
        f"Generated: `{utc_now()}`",
        "",
        "Mode: `EXECUTE`",
        "",
        "## Coverage By Season",
        markdown_table(out),
        "",
        "## Interpretation",
        "- `PLAYER_FIXTURE_INTEL_READY_TO_BACKFILL` means fixtures, lineups, fixture player stats, and player roster coverage all appear available.",
        "- `PLAYER_DETAIL_PROBE_PRESENT_COVERAGE_FLAG_FALSE` means the advertised coverage flag is false but a sample fixture returned lineups or fixture-player rows.",
        "- `FIXTURE_READY_PLAYER_INTEL_PARTIAL` means use fixture/team/venue data but do not claim player-intelligence completeness yet.",
        "- `fixture_coverage_ratio_vs_official` compares returned fixture count against the official World Cup format target: `64` matches for 2010-2022 and `104` for 2026.",
        "- The API-Sports 2026 guide advertises schedule, teams, rounds, standings, squads, injuries, fixture-player ratings, predictions, odds, and live data under `league=1&season=2026`; this audit checks availability but does not prove timestamp-safe pre-match use.",
        "- This is an availability audit only; pre-kickoff timestamp safety still needs a separate snapshot contract.",
    ]
    (outdir / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")


def planned_request_rows(
    seasons: list[int],
    *,
    probe_players: bool,
    probe_injuries: bool,
    probe_fixture_details: bool,
    probe_tournament_endpoints: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for season in seasons:
        rows.extend(
            [
                {"season": season, "endpoint": "/leagues", "params": f"id=1&season={season}"},
                {"season": season, "endpoint": "/fixtures", "params": f"league=1&season={season}"},
                {"season": season, "endpoint": "/teams", "params": f"league=1&season={season}"},
                {"season": season, "endpoint": "/fixtures/rounds", "params": f"league=1&season={season}"},
            ]
        )
        if probe_players:
            rows.append({"season": season, "endpoint": "/players", "params": f"league=1&season={season}&page=1"})
        if probe_injuries:
            rows.append({"season": season, "endpoint": "/injuries", "params": f"league=1&season={season}"})
        if probe_tournament_endpoints:
            rows.extend(
                [
                    {"season": season, "endpoint": "/standings", "params": f"league=1&season={season}"},
                    {"season": season, "endpoint": "/players/topscorers", "params": f"league=1&season={season}"},
                    {"season": season, "endpoint": "/players/topassists", "params": f"league=1&season={season}"},
                    {"season": season, "endpoint": "/players/topyellowcards", "params": f"league=1&season={season}"},
                    {"season": season, "endpoint": "/coachs", "params": "team=<first_team_id_from_season>"},
                ]
            )
        if probe_fixture_details:
            rows.extend(
                [
                    {"season": season, "endpoint": "/fixtures", "params": "id=<first_fixture_id_from_season>"},
                    {"season": season, "endpoint": "/fixtures/players", "params": "fixture=<first_fixture_id_from_season>"},
                ]
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", default=DEFAULT_SEASONS)
    parser.add_argument("--execute", action="store_true", help="Actually call API-Football.")
    parser.add_argument("--probe-players", action="store_true", default=True)
    parser.add_argument("--no-probe-players", action="store_false", dest="probe_players")
    parser.add_argument("--probe-injuries", action="store_true", default=True)
    parser.add_argument("--no-probe-injuries", action="store_false", dest="probe_injuries")
    parser.add_argument("--probe-fixture-details", action="store_true", default=True)
    parser.add_argument("--no-probe-fixture-details", action="store_false", dest="probe_fixture_details")
    parser.add_argument("--probe-tournament-endpoints", action="store_true", default=True)
    parser.add_argument("--no-probe-tournament-endpoints", action="store_false", dest="probe_tournament_endpoints")
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--daily-cap", type=int, default=75000)
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--finalize-existing", action="store_true", help="Finalize SUMMARY.md from an existing coverage CSV without making API calls.")
    args = parser.parse_args()

    seasons = [int(item.strip()) for item in args.seasons.split(",") if item.strip()]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    planned = planned_request_rows(
        seasons,
        probe_players=args.probe_players,
        probe_injuries=args.probe_injuries,
        probe_fixture_details=args.probe_fixture_details,
        probe_tournament_endpoints=args.probe_tournament_endpoints,
    )
    planned.to_csv(outdir / "planned_requests.csv", index=False)

    if not args.execute:
        summary = [
            "# API-Football World Cup Coverage Audit",
            "",
            f"Generated: `{utc_now()}`",
            "",
            "Mode: `DRY_RUN`",
            "",
            f"Planned requests: `{len(planned)}`",
            "",
            "Run with `--execute` after reviewing API budget.",
            "",
            "Official API-Sports World Cup guide target: `league=1`, `season=2026`, `104` matches, `48` teams, `12` groups.",
            "",
            "## Planned Requests",
            markdown_table(planned),
        ]
        (outdir / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")
        print(f"DRY_RUN planned_requests={len(planned)} outdir={outdir}")
        return 0

    if args.finalize_existing:
        existing = outdir / "world_cup_api_football_coverage.csv"
        if not existing.exists():
            raise SystemExit(f"Missing existing coverage CSV: {existing}")
        out = pd.read_csv(existing)
        finalize_execute_outputs(out, outdir)
        print(f"FINALIZED {existing} rows={len(out)}")
        return 0

    client = APIFootballClient(sleep_seconds=args.sleep_seconds, daily_cap=args.daily_cap)
    rows: list[dict[str, Any]] = []
    for season in seasons:
        league_payload = client.get_json("/leagues", {"id": WORLD_CUP_LEAGUE_ID, "season": season})
        fixtures_payload = client.get_json("/fixtures", {"league": WORLD_CUP_LEAGUE_ID, "season": season})
        teams_payload = client.get_json("/teams", {"league": WORLD_CUP_LEAGUE_ID, "season": season})
        rounds_payload = client.get_json("/fixtures/rounds", {"league": WORLD_CUP_LEAGUE_ID, "season": season})

        player_page_results = None
        player_pages_total = None
        if args.probe_players:
            players_payload = client.get_json("/players", {"league": WORLD_CUP_LEAGUE_ID, "season": season, "page": 1})
            player_page_results = int(players_payload.get("results") or 0)
            player_pages_total = int(get_path(players_payload, "paging", "total", default=0) or 0)

        injury_results = None
        if args.probe_injuries:
            injuries_payload = client.get_json("/injuries", {"league": WORLD_CUP_LEAGUE_ID, "season": season})
            injury_results = int(injuries_payload.get("results") or 0)

        standings_results = None
        top_scorers_results = None
        top_assists_results = None
        top_yellow_cards_results = None
        sample_team_id = None
        sample_coachs_results = None
        team_response = teams_payload.get("response") or []
        if team_response:
            sample_team_id = get_path(team_response[0], "team", "id")
        if args.probe_tournament_endpoints:
            standings_payload = client.get_json("/standings", {"league": WORLD_CUP_LEAGUE_ID, "season": season})
            top_scorers_payload = client.get_json("/players/topscorers", {"league": WORLD_CUP_LEAGUE_ID, "season": season})
            top_assists_payload = client.get_json("/players/topassists", {"league": WORLD_CUP_LEAGUE_ID, "season": season})
            top_yellow_cards_payload = client.get_json("/players/topyellowcards", {"league": WORLD_CUP_LEAGUE_ID, "season": season})
            standings_results = int(standings_payload.get("results") or 0)
            top_scorers_results = int(top_scorers_payload.get("results") or 0)
            top_assists_results = int(top_assists_payload.get("results") or 0)
            top_yellow_cards_results = int(top_yellow_cards_payload.get("results") or 0)
            if sample_team_id is not None:
                coachs_payload = client.get_json("/coachs", {"team": sample_team_id})
                sample_coachs_results = int(coachs_payload.get("results") or 0)

        fixture_id = None
        sample_fixture_detail_results = None
        sample_detail_events = None
        sample_detail_lineups = None
        sample_detail_statistics = None
        sample_detail_players = None
        sample_fixture_players_results = None
        sample_fixture_players_player_rows = None
        fixture_response = fixtures_payload.get("response") or []
        if fixture_response:
            fixture_id = get_path(fixture_response[0], "fixture", "id")
        if args.probe_fixture_details and fixture_id is not None:
            detail_payload = client.get_json("/fixtures", {"id": fixture_id})
            detail_response = detail_payload.get("response") or []
            sample_fixture_detail_results = int(detail_payload.get("results") or 0)
            if detail_response:
                detail = detail_response[0] or {}
                sample_detail_events = len(detail.get("events") or [])
                sample_detail_lineups = len(detail.get("lineups") or [])
                sample_detail_statistics = len(detail.get("statistics") or [])
                sample_detail_players = len(detail.get("players") or [])
            fixture_players_payload = client.get_json("/fixtures/players", {"fixture": fixture_id})
            sample_fixture_players_results = int(fixture_players_payload.get("results") or 0)
            sample_fixture_players_player_rows = sum(
                len(team_block.get("players", []) or [])
                for team_block in (fixture_players_payload.get("response") or [])
            )

        rows.append(
            {
                "league_id": WORLD_CUP_LEAGUE_ID,
                "season": season,
                "expected_fixtures_official": OFFICIAL_EXPECTED_FIXTURES.get(season),
                "league_results": int(league_payload.get("results") or 0),
                "fixtures_results": int(fixtures_payload.get("results") or 0),
                "teams_results": int(teams_payload.get("results") or 0),
                "rounds_results": int(rounds_payload.get("results") or 0),
                "players_page1_results": player_page_results,
                "players_pages_total": player_pages_total,
                "injuries_results": injury_results,
                "standings_results": standings_results,
                "top_scorers_results": top_scorers_results,
                "top_assists_results": top_assists_results,
                "top_yellow_cards_results": top_yellow_cards_results,
                "sample_team_id": sample_team_id,
                "sample_coachs_results": sample_coachs_results,
                "sample_fixture_id": fixture_id,
                "sample_fixture_detail_results": sample_fixture_detail_results,
                "sample_detail_events": sample_detail_events,
                "sample_detail_lineups": sample_detail_lineups,
                "sample_detail_statistics": sample_detail_statistics,
                "sample_detail_players": sample_detail_players,
                "sample_fixture_players_results": sample_fixture_players_results,
                "sample_fixture_players_player_rows": sample_fixture_players_player_rows,
                **coverage_flat(league_payload),
            }
        )

    out = pd.DataFrame(rows)
    if "expected_fixtures_official" in out.columns:
        out["fixture_coverage_ratio_vs_official"] = out.apply(
            lambda row: (
                float(row["fixtures_results"]) / float(row["expected_fixtures_official"])
                if pd.notna(row.get("expected_fixtures_official")) and float(row.get("expected_fixtures_official") or 0) > 0
                else None
            ),
            axis=1,
        )
    out.to_csv(outdir / "world_cup_api_football_coverage.csv", index=False)

    finalize_execute_outputs(out, outdir)
    print(f"WROTE {outdir / 'world_cup_api_football_coverage.csv'} rows={len(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
