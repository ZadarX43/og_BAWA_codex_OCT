#!/usr/bin/env python3
"""Export the publish-safe website estate into a local SQLite database.

The output is a reproducible launch-planning artifact, not a source-of-truth
prediction artifact. It lets us benchmark the exact API reads we would later
serve from Cloudflare D1, Turso, or a self-hosted SQLite service.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATA_ROOT = Path("frontend/public/data")
DEFAULT_OUTPUT = Path("build/site_data/odds_genius.sqlite")
SCHEMA_VERSION = 1


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def json_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if not text or not text.lstrip("-").isdigit():
            return None
        return int(text)
    except (TypeError, ValueError):
        return None


def normalize_key(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def season_sort_key(season: Any) -> tuple[int, str]:
    text = str(season or "").strip()
    years = [int(part) for part in text.replace("-", "/").split("/") if part.isdigit()]
    return (max(years) if years else 0, text)


def active_competition_seasons(data_root: Path) -> dict[str, str]:
    lineup_index = read_json(data_root / "fixture_lineup_intelligence" / "index.json", [])
    active_competitions = {
        str(row.get("competition_key") or "").strip()
        for row in lineup_index
        if isinstance(row, dict) and row.get("competition_key")
    }
    team_index = read_json(data_root / "team_intelligence" / "team_ratings_index.json", [])
    seasons: dict[str, str] = {}
    for row in team_index if isinstance(team_index, list) else []:
        competition_key = str(row.get("competition_key") or "").strip()
        season = str(row.get("season") or "").strip()
        if not competition_key or not season or competition_key not in active_competitions:
            continue
        if competition_key not in seasons or season_sort_key(season) > season_sort_key(seasons[competition_key]):
            seasons[competition_key] = season
    return seasons


def execute_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        PRAGMA foreign_keys = ON;

        DROP TABLE IF EXISTS metadata;
        DROP TABLE IF EXISTS fixtures;
        DROP TABLE IF EXISTS fixture_decisions;
        DROP TABLE IF EXISTS fixture_lineups;
        DROP TABLE IF EXISTS fixture_h2h;
        DROP TABLE IF EXISTS team_intelligence;
        DROP TABLE IF EXISTS club_squads;
        DROP TABLE IF EXISTS team_lineup_snapshots;

        CREATE TABLE metadata (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );

        CREATE TABLE fixtures (
          fixture_key TEXT PRIMARY KEY,
          fixture_id INTEGER,
          kickoff_time TEXT,
          league TEXT,
          league_key TEXT,
          api_league_id INTEGER,
          api_season TEXT,
          home_team TEXT,
          away_team TEXT,
          fixture_class TEXT,
          publish_class TEXT,
          coverage_status TEXT,
          payload_json TEXT NOT NULL
        );

        CREATE TABLE fixture_decisions (
          fixture_key TEXT PRIMARY KEY,
          primary_signal TEXT,
          signal_state TEXT,
          agreement_score INTEGER,
          confidence_band TEXT,
          payload_json TEXT NOT NULL
        );

        CREATE TABLE fixture_lineups (
          fixture_key TEXT PRIMARY KEY,
          competition_key TEXT,
          coverage_status TEXT,
          lineup_status TEXT,
          lineup_mode TEXT,
          home_team TEXT,
          away_team TEXT,
          payload_json TEXT NOT NULL
        );

        CREATE TABLE fixture_h2h (
          fixture_key TEXT PRIMARY KEY,
          competition TEXT,
          coverage_status TEXT,
          fallback_mode TEXT,
          sample_size INTEGER,
          payload_json TEXT NOT NULL
        );

        CREATE TABLE team_intelligence (
          competition_key TEXT NOT NULL,
          season TEXT NOT NULL,
          team_slug TEXT NOT NULL,
          team TEXT,
          headline_rating INTEGER,
          payload_json TEXT NOT NULL,
          PRIMARY KEY (competition_key, season, team_slug)
        );

        CREATE TABLE club_squads (
          competition_key TEXT NOT NULL,
          season TEXT NOT NULL,
          club_slug TEXT NOT NULL,
          club TEXT,
          player_count INTEGER,
          payload_json TEXT NOT NULL,
          PRIMARY KEY (competition_key, season, club_slug)
        );

        CREATE TABLE team_lineup_snapshots (
          competition_key TEXT NOT NULL,
          team_key TEXT NOT NULL,
          team TEXT,
          season TEXT,
          snapshot_source_year TEXT,
          source_match_date TEXT,
          starter_count INTEGER,
          bench_count INTEGER,
          payload_json TEXT NOT NULL,
          PRIMARY KEY (competition_key, team_key)
        );

        CREATE INDEX idx_fixtures_league_time ON fixtures(league_key, kickoff_time);
        CREATE INDEX idx_fixtures_home ON fixtures(home_team);
        CREATE INDEX idx_fixtures_away ON fixtures(away_team);
        CREATE INDEX idx_team_intelligence_team ON team_intelligence(team_slug);
        CREATE INDEX idx_club_squads_club ON club_squads(club_slug);
        CREATE INDEX idx_team_lineup_snapshots_team ON team_lineup_snapshots(team_key);
        """
    )


def insert_metadata(conn: sqlite3.Connection, rows: dict[str, Any]) -> None:
    conn.executemany(
        "INSERT INTO metadata(key, value) VALUES (?, ?)",
        [(key, str(value)) for key, value in rows.items()],
    )


def insert_fixtures(conn: sqlite3.Connection, data_root: Path) -> int:
    payload = read_json(data_root / "fixture_intelligence_public.json", {})
    fixtures = payload.get("fixtures") if isinstance(payload, dict) else []
    rows = []
    for fixture in fixtures or []:
        fixture_key = str(fixture.get("fixture_key") or "").strip()
        if not fixture_key:
            continue
        rows.append(
            (
                fixture_key,
                safe_int(fixture.get("api_fixture_id") or fixture.get("fixture_id")),
                fixture.get("kickoff_time"),
                fixture.get("league"),
                normalize_key(fixture.get("league")),
                safe_int(fixture.get("api_league_id")),
                str(fixture.get("api_season") or fixture.get("season") or ""),
                fixture.get("home_team"),
                fixture.get("away_team"),
                fixture.get("fixture_class"),
                fixture.get("publish_class"),
                fixture.get("coverage_status"),
                json_text(fixture),
            )
        )
    conn.executemany(
        """
        INSERT INTO fixtures(
          fixture_key, fixture_id, kickoff_time, league, league_key, api_league_id,
          api_season, home_team, away_team, fixture_class, publish_class,
          coverage_status, payload_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def insert_payload_dir(
    conn: sqlite3.Connection,
    data_root: Path,
    rel_dir: str,
    table: str,
    columns: tuple[str, ...],
    row_builder,
) -> int:
    index_rows = read_json(data_root / rel_dir / "index.json", [])
    if not isinstance(index_rows, list):
        return 0
    rows = []
    for index_row in index_rows:
        fixture_key = str(index_row.get("fixture_key") or "").strip()
        if not fixture_key:
            continue
        payload = read_json(data_root / rel_dir / f"{fixture_key}.json", None)
        if payload is None:
            payload = index_row
        rows.append(row_builder(index_row, payload))
    placeholders = ", ".join("?" for _ in columns)
    conn.executemany(
        f"INSERT INTO {table}({', '.join(columns)}) VALUES ({placeholders})",
        rows,
    )
    return len(rows)


def insert_fixture_decisions(conn: sqlite3.Connection, data_root: Path) -> int:
    return insert_payload_dir(
        conn,
        data_root,
        "fixture_decision_intelligence",
        "fixture_decisions",
        ("fixture_key", "primary_signal", "signal_state", "agreement_score", "confidence_band", "payload_json"),
        lambda index_row, payload: (
            index_row.get("fixture_key"),
            payload.get("primary_signal") or index_row.get("primary_signal"),
            payload.get("signal_state") or index_row.get("signal_state"),
            safe_int(payload.get("agreement_score") or index_row.get("agreement_score")),
            payload.get("confidence_band") or index_row.get("confidence_band"),
            json_text(payload),
        ),
    )


def insert_fixture_lineups(conn: sqlite3.Connection, data_root: Path) -> int:
    return insert_payload_dir(
        conn,
        data_root,
        "fixture_lineup_intelligence",
        "fixture_lineups",
        ("fixture_key", "competition_key", "coverage_status", "lineup_status", "lineup_mode", "home_team", "away_team", "payload_json"),
        lambda index_row, payload: (
            index_row.get("fixture_key"),
            payload.get("competition_key") or index_row.get("competition_key"),
            payload.get("coverage_status") or index_row.get("coverage_status"),
            payload.get("lineup_status") or index_row.get("lineup_status"),
            payload.get("lineup_mode") or index_row.get("lineup_mode"),
            payload.get("home_team") or index_row.get("home_team"),
            payload.get("away_team") or index_row.get("away_team"),
            json_text(payload),
        ),
    )


def insert_fixture_h2h(conn: sqlite3.Connection, data_root: Path) -> int:
    return insert_payload_dir(
        conn,
        data_root,
        "fixture_h2h_support",
        "fixture_h2h",
        ("fixture_key", "competition", "coverage_status", "fallback_mode", "sample_size", "payload_json"),
        lambda index_row, payload: (
            index_row.get("fixture_key"),
            payload.get("competition") or index_row.get("competition"),
            payload.get("coverage_status") or index_row.get("coverage_status"),
            payload.get("fallback_mode") or index_row.get("fallback_mode"),
            safe_int(payload.get("sample_size") or index_row.get("sample_size")),
            json_text(payload),
        ),
    )


def competition_payload_path(data_root: Path, competition_key: str, season: str) -> Path:
    return data_root / "team_intelligence" / "competitions" / f"{competition_key}__{season}.json"


def insert_team_intelligence(
    conn: sqlite3.Connection,
    data_root: Path,
    active_seasons: dict[str, str] | None = None,
) -> int:
    index_rows = read_json(data_root / "team_intelligence" / "team_ratings_index.json", [])
    rows = []
    payload_cache: dict[Path, list[dict[str, Any]]] = {}
    for row in index_rows if isinstance(index_rows, list) else []:
        competition_key = row.get("competition_key")
        season = str(row.get("season") or "")
        team_slug = row.get("team_slug")
        if not competition_key or not season or not team_slug:
            continue
        if active_seasons is not None and active_seasons.get(competition_key) != season:
            continue
        path = competition_payload_path(data_root, competition_key, season)
        if path not in payload_cache:
            payload = read_json(path, [])
            if isinstance(payload, dict):
                payload_cache[path] = payload.get("teams") if isinstance(payload.get("teams"), list) else []
            else:
                payload_cache[path] = payload if isinstance(payload, list) else []
        team_payload = next(
            (item for item in payload_cache[path] if item.get("team_slug") == team_slug or item.get("team") == row.get("team")),
            row,
        )
        rows.append(
            (
                competition_key,
                season,
                team_slug,
                row.get("team"),
                safe_int(row.get("headline_rating")),
                json_text(team_payload),
            )
        )
    conn.executemany(
        """
        INSERT OR REPLACE INTO team_intelligence(
          competition_key, season, team_slug, team, headline_rating, payload_json
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def insert_club_squads(
    conn: sqlite3.Connection,
    data_root: Path,
    active_seasons: dict[str, str] | None = None,
) -> int:
    squads = read_json(data_root / "player_intelligence" / "club_squad_ratings.json", [])
    rows = []
    for squad in squads if isinstance(squads, list) else []:
        competition_key = squad.get("competition_key")
        season = str(squad.get("season") or "")
        club_slug = squad.get("club_slug")
        if not competition_key or not season or not club_slug:
            continue
        if active_seasons is not None and active_seasons.get(competition_key) != season:
            continue
        rows.append(
            (
                competition_key,
                season,
                club_slug,
                squad.get("club"),
                len(squad.get("players") or []),
                json_text(squad),
            )
        )
    conn.executemany(
        """
        INSERT OR REPLACE INTO club_squads(
          competition_key, season, club_slug, club, player_count, payload_json
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def insert_team_lineup_snapshots(conn: sqlite3.Connection, data_root: Path) -> int:
    index_rows = read_json(data_root / "fixture_lineup_intelligence" / "team_snapshots" / "index.json", [])
    rows = []
    for row in index_rows if isinstance(index_rows, list) else []:
        competition_key = row.get("competition_key")
        team_key = row.get("team_key")
        if not competition_key or not team_key:
            continue
        payload = read_json(
            data_root / "fixture_lineup_intelligence" / "team_snapshots" / competition_key / f"{team_key}.json",
            row,
        )
        rows.append(
            (
                competition_key,
                team_key,
                row.get("team"),
                str(row.get("season") or ""),
                str(row.get("snapshot_source_year") or ""),
                row.get("source_match_date"),
                safe_int(row.get("starter_count")),
                safe_int(row.get("bench_count")),
                json_text(payload),
            )
        )
    conn.executemany(
        """
        INSERT OR REPLACE INTO team_lineup_snapshots(
          competition_key, team_key, team, season, snapshot_source_year,
          source_match_date, starter_count, bench_count, payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def export_database(data_root: Path, output_path: Path, include_history: bool = False) -> dict[str, int | str]:
    started = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    active_seasons = None if include_history else active_competition_seasons(data_root)

    conn = sqlite3.connect(output_path)
    try:
        execute_schema(conn)
        counts = {
            "fixtures": insert_fixtures(conn, data_root),
            "fixture_decisions": insert_fixture_decisions(conn, data_root),
            "fixture_lineups": insert_fixture_lineups(conn, data_root),
            "fixture_h2h": insert_fixture_h2h(conn, data_root),
            "team_intelligence": insert_team_intelligence(conn, data_root, active_seasons),
            "club_squads": insert_club_squads(conn, data_root, active_seasons),
            "team_lineup_snapshots": insert_team_lineup_snapshots(conn, data_root),
        }
        insert_metadata(
            conn,
            {
                "schema_version": SCHEMA_VERSION,
                "source_data_root": str(data_root),
                "created_unix": int(time.time()),
                "scope": "full_history" if include_history else "active_site_latest_seasons",
                "active_competition_count": len(active_seasons or {}),
                **{f"count_{key}": value for key, value in counts.items()},
            },
        )
        conn.commit()
        conn.execute("PRAGMA optimize")
        conn.commit()
    finally:
        conn.close()

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    size_bytes = output_path.stat().st_size
    return {
        **counts,
        "output": str(output_path),
        "size_bytes": size_bytes,
        "elapsed_ms": elapsed_ms,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export publish-safe website JSON into SQLite.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--include-history",
        action="store_true",
        help="Export all historical team/player club-season rows instead of the active-site latest-season footprint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_database(Path(args.data_root), Path(args.output), include_history=args.include_history)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
