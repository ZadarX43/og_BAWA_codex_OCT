#!/usr/bin/env python3
"""Benchmark page-shaped reads against the local Odds Genius SQLite export."""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import time
from pathlib import Path
from typing import Any, Callable


DEFAULT_DB = Path("build/site_data/odds_genius.sqlite")


def decode_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = dict(row)
    for key in list(payload):
        if key.endswith("_json") and payload[key]:
            payload[key[:-5]] = json.loads(payload.pop(key))
    return payload


def get_fixture_detail(conn: sqlite3.Connection, fixture_key: str) -> dict[str, Any]:
    fixture = decode_row(conn.execute("SELECT payload_json FROM fixtures WHERE fixture_key = ?", (fixture_key,)).fetchone())
    decision = decode_row(conn.execute("SELECT payload_json FROM fixture_decisions WHERE fixture_key = ?", (fixture_key,)).fetchone())
    lineup = decode_row(conn.execute("SELECT payload_json FROM fixture_lineups WHERE fixture_key = ?", (fixture_key,)).fetchone())
    h2h = decode_row(conn.execute("SELECT payload_json FROM fixture_h2h WHERE fixture_key = ?", (fixture_key,)).fetchone())
    return {"fixture": fixture and fixture["payload"], "decision": decision and decision["payload"], "lineup": lineup and lineup["payload"], "h2h": h2h and h2h["payload"]}


def get_team_detail(conn: sqlite3.Connection, competition_key: str, team_slug: str) -> dict[str, Any]:
    team = decode_row(
        conn.execute(
            """
            SELECT payload_json
            FROM team_intelligence
            WHERE competition_key = ? AND team_slug = ?
            ORDER BY season DESC
            LIMIT 1
            """,
            (competition_key, team_slug),
        ).fetchone()
    )
    squad = decode_row(
        conn.execute(
            """
            SELECT payload_json
            FROM club_squads
            WHERE competition_key = ? AND club_slug = ?
            ORDER BY season DESC
            LIMIT 1
            """,
            (competition_key, team_slug),
        ).fetchone()
    )
    snapshot = decode_row(
        conn.execute(
            """
            SELECT payload_json
            FROM team_lineup_snapshots
            WHERE competition_key = ? AND team_key = ?
            LIMIT 1
            """,
            (competition_key, team_slug),
        ).fetchone()
    )
    return {"team": team and team["payload"], "squad": squad and squad["payload"], "lineup_snapshot": snapshot and snapshot["payload"]}


def get_current_fixtures(conn: sqlite3.Connection, limit: int = 80) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT fixture_key, kickoff_time, league, home_team, away_team, fixture_class, publish_class, coverage_status
        FROM fixtures
        ORDER BY kickoff_time, league, home_team
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]


def time_call(fn: Callable[[], Any], iterations: int) -> tuple[list[float], Any]:
    timings: list[float] = []
    last_payload: Any = None
    for _ in range(iterations):
        started = time.perf_counter()
        last_payload = fn()
        timings.append((time.perf_counter() - started) * 1000)
    return timings, last_payload


def summarize_timings(timings: list[float]) -> dict[str, float]:
    ordered = sorted(timings)
    p95_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
    return {
        "min_ms": round(min(ordered), 3),
        "median_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(ordered[p95_index], 3),
        "max_ms": round(max(ordered), 3),
    }


def benchmark(db_path: Path, iterations: int) -> dict[str, Any]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        sample_fixture = conn.execute(
            "SELECT fixture_key FROM fixture_lineups WHERE coverage_status = 'predicted' LIMIT 1"
        ).fetchone()
        if sample_fixture is None:
            sample_fixture = conn.execute("SELECT fixture_key FROM fixtures LIMIT 1").fetchone()
        fixture_key = sample_fixture["fixture_key"]

        sample_team = conn.execute(
            """
            SELECT team_intelligence.competition_key, team_intelligence.team_slug
            FROM team_intelligence
            INNER JOIN club_squads
              ON club_squads.competition_key = team_intelligence.competition_key
             AND club_squads.club_slug = team_intelligence.team_slug
             AND club_squads.season = team_intelligence.season
            ORDER BY team_intelligence.season DESC, team_intelligence.competition_key, team_intelligence.team_slug
            LIMIT 1
            """
        ).fetchone()

        fixture_timings, fixture_payload = time_call(lambda: get_fixture_detail(conn, fixture_key), iterations)
        current_timings, current_payload = time_call(lambda: get_current_fixtures(conn), iterations)
        team_timings: list[float] = []
        team_payload = None
        if sample_team:
            team_timings, team_payload = time_call(
                lambda: get_team_detail(conn, sample_team["competition_key"], sample_team["team_slug"]),
                iterations,
            )

        return {
            "db_path": str(db_path),
            "iterations": iterations,
            "fixture_key": fixture_key,
            "fixture_detail": summarize_timings(fixture_timings),
            "fixture_payload_present": {key: bool(value) for key, value in fixture_payload.items()},
            "current_fixtures": summarize_timings(current_timings),
            "current_fixture_count": len(current_payload),
            "team_lookup": summarize_timings(team_timings) if team_timings else None,
            "team_payload_present": {key: bool(value) for key, value in (team_payload or {}).items()},
        }
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark local Odds Genius SQLite page reads.")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--iterations", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(benchmark(Path(args.db), args.iterations), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
