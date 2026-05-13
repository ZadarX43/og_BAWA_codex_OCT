#!/usr/bin/env python3
"""Export the publish-safe website estate into a local SQLite database.

The output is a reproducible launch-planning artifact, not a source-of-truth
prediction artifact. It lets us benchmark the exact API reads we would later
serve from Cloudflare D1, Turso, or a self-hosted SQLite service.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATA_ROOT = Path("frontend/public/data")
DEFAULT_NORMALIZED_ROOT = Path("data_sources/api_football/normalized")
DEFAULT_OUTPUT = Path("build/site_data/odds_genius.sqlite")
SCHEMA_VERSION = 5
DEFAULT_ROUTE_CACHE_LIMIT = 20
DEFAULT_EVENT_SHORTLIST_LIMIT = 3


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


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def slug_key(value: Any) -> str:
    return normalize_text(value).replace(" ", "_")


def slug_aliases(value: Any) -> set[str]:
    base = slug_key(value)
    aliases = {base} if base else set()
    for prefix in ("fc_", "cf_", "afc_", "sc_"):
        if base.startswith(prefix):
            aliases.add(base[len(prefix) :])
    for suffix in ("_fc", "_cf", "_afc", "_sc"):
        if base.endswith(suffix):
            aliases.add(base[: -len(suffix)])
    return {alias for alias in aliases if alias}


COMPETITION_KEY_ALIASES = {
    "a_league": "australia_a_league",
    "bundesliga": "germany_bundesliga",
    "2_bundesliga": "germany_bundesliga_2",
    "championship": "england_championship",
    "efl_league_1": "england_efl_league_1",
    "premier_league": "england_premier_league",
    "la_liga": "spain_la_liga",
    "serie_a": "italy_serie_a",
    "brazil_serie_a": "brazil_serie_a",
    "ligue_1": "france_ligue_1",
    "eredivisie": "netherlands_eredivisie",
    "eliteserien": "norway_eliteserien",
    "liga": "portugal_liga",
    "pro_league": "saudi_pro_league",
    "premiership": "scotland_premiership",
    "k_league_1": "south_korea_k_league",
    "super_league": "swiss_super_league",
    "super_lig": "turkey_super_lig",
    "mls": "usa_mls",
}


def competition_key_from_source(source_file: Any, league: Any = "") -> str:
    source = str(source_file or "")
    match = re.match(r"^[a-z_]+__(.+)__\d{4}\.csv$", source)
    raw = match.group(1) if match else str(league or "")
    key = slug_key(raw.replace("__", "_"))
    if key.startswith("england_"):
        return key
    if key.startswith("germany_") or key.startswith("spain_") or key.startswith("italy_") or key.startswith("brazil_"):
        return key
    if key.startswith("france_") or key.startswith("netherlands_") or key.startswith("norway_") or key.startswith("portugal_"):
        return key
    if key.startswith("scotland_") or key.startswith("switzerland_") or key.startswith("turkey_") or key.startswith("usa_"):
        return "swiss_super_league" if key == "switzerland_super_league" else key
    return COMPETITION_KEY_ALIASES.get(key, key)


def person_keys(value: Any) -> dict[str, str]:
    parts = [part for part in normalize_text(value).split() if part]
    if not parts:
        return {"full": "", "surname": "", "initial_surname": ""}
    return {
        "full": " ".join(parts),
        "surname": parts[-1],
        "initial_surname": f"{parts[0][0]} {parts[-1]}",
    }


def broad_position(value: Any) -> str:
    key = str(value or "").strip().upper()
    if key in {"G", "GK", "GOALKEEPER"}:
        return "G"
    if key in {"D", "DEFENDER", "DEFENCE"}:
        return "D"
    if key in {"M", "MIDFIELDER", "MIDFIELD"}:
        return "M"
    if key in {"F", "FORWARD", "ATTACKER"}:
        return "F"
    return key[:1] if key[:1] in {"G", "D", "M", "F"} else ""


def position_group_from_broad(value: Any) -> str:
    return {
        "G": "goalkeeper",
        "D": "defender",
        "M": "midfielder",
        "F": "forward",
    }.get(broad_position(value), "utility")


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if text == "":
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    for row in rows:
        row["_source_file"] = path.name
    return rows


def read_normalized_family_rows(root: Path, prefix: str, latest_per_competition: bool = True) -> list[dict[str, str]]:
    if not root.exists():
        return []
    if not latest_per_competition:
        rows: list[dict[str, str]] = []
        for path in sorted(root.glob(f"{prefix}*.csv")):
            rows.extend(read_csv_rows(path))
        return rows
    latest: dict[str, tuple[int, Path]] = {}
    pattern = re.compile(rf"^{re.escape(prefix)}__(.+)__(\d{{4}})$")
    for path in root.glob(f"{prefix}__*.csv"):
        match = pattern.match(path.stem)
        if not match:
            continue
        competition = normalize_text(match.group(1).replace("_", " "))
        year = int(match.group(2))
        if competition not in latest or year > latest[competition][0]:
            latest[competition] = (year, path)
    if not latest:
        return read_csv_rows(root / f"{prefix}.csv")
    rows: list[dict[str, str]] = []
    for _year, path in sorted(latest.values(), key=lambda item: str(item[1])):
        rows.extend(read_csv_rows(path))
    return rows


def season_sort_key(season: Any) -> tuple[int, str]:
    text = str(season or "").strip()
    years = [int(part) for part in text.replace("-", "/").split("/") if part.isdigit()]
    return (max(years) if years else 0, text)


def season_years(season: Any) -> set[int]:
    return {int(part) for part in re.findall(r"\d{4}", str(season or ""))}


def source_year(source_file: Any) -> int | None:
    match = re.search(r"__(\d{4})\.csv$", str(source_file or ""))
    return int(match.group(1)) if match else None


def normalized_row_in_scope(row: dict[str, Any], context: dict[str, Any] | None, active_seasons: dict[str, str] | None) -> bool:
    if active_seasons is None:
        return True
    competition_key = competition_key_from_source(
        (context or row).get("_source_file"),
        (context or row).get("league") or row.get("competition") or row.get("league_name"),
    )
    if competition_key not in active_seasons:
        return False
    allowed_years = season_years(active_seasons[competition_key])
    if not allowed_years:
        return True
    row_years = season_years((context or row).get("season"))
    file_year = source_year((context or row).get("_source_file"))
    if file_year is not None:
        row_years.add(file_year)
    return bool(row_years & allowed_years)


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


def active_site_fixture_aliases(data_root: Path) -> dict[tuple[str, str, str], str]:
    payload = read_json(data_root / "fixture_intelligence_public.json", {})
    fixtures = payload.get("fixtures") if isinstance(payload, dict) else []
    aliases: dict[tuple[str, str, str], str] = {}
    for fixture in fixtures or []:
        fixture_key = str(fixture.get("fixture_key") or "").strip()
        if not fixture_key:
            continue
        date_key = str(fixture.get("kickoff_time") or "")[:10]
        if not date_key:
            match = re.match(r"^(\d{4})_(\d{2})_(\d{2})_", fixture_key)
            if match:
                date_key = "-".join(match.groups())
        if not date_key:
            continue
        home_aliases = slug_aliases(fixture.get("home_team"))
        away_aliases = slug_aliases(fixture.get("away_team"))
        for home_alias in home_aliases:
            for away_alias in away_aliases:
                aliases[(date_key, home_alias, away_alias)] = fixture_key
    return aliases


def resolve_active_site_fixture_key(
    context: dict[str, Any],
    site_fixture_aliases: dict[tuple[str, str, str], str] | None,
) -> str:
    provider_key = str(context.get("fixture_key") or "").strip()
    if not site_fixture_aliases:
        return provider_key
    date_key = str(context.get("match_date") or "")[:10]
    for home_alias in slug_aliases(context.get("home_team_name")):
        for away_alias in slug_aliases(context.get("away_team_name")):
            mapped = site_fixture_aliases.get((date_key, home_alias, away_alias))
            if mapped:
                return mapped
    return provider_key


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
        DROP TABLE IF EXISTS site_player_identity_map;
        DROP TABLE IF EXISTS site_player_match_stats;
        DROP TABLE IF EXISTS site_team_match_stats;
        DROP TABLE IF EXISTS site_match_events;
        DROP TABLE IF EXISTS site_lineup_slots;
        DROP TABLE IF EXISTS site_formation_slots;
        DROP TABLE IF EXISTS site_fixture_market_intelligence;
        DROP TABLE IF EXISTS site_player_event_shortlists;
        DROP TABLE IF EXISTS site_external_sources;
        DROP TABLE IF EXISTS site_fixture_external_content;
        DROP TABLE IF EXISTS site_fixture_context_payloads;
        DROP TABLE IF EXISTS site_fixture_stats_payloads;
        DROP TABLE IF EXISTS site_team_premium_payloads;

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

        CREATE TABLE site_player_identity_map (
          player_key TEXT PRIMARY KEY,
          api_player_id INTEGER,
          rating_player_id TEXT,
          name TEXT,
          canonical_name TEXT,
          club TEXT,
          club_slug TEXT,
          competition_key TEXT,
          season TEXT,
          position TEXT,
          position_group TEXT,
          rating_power INTEGER,
          rank_overall INTEGER,
          rank_position INTEGER,
          rank_club INTEGER,
          payload_json TEXT NOT NULL
        );

        CREATE TABLE site_player_match_stats (
          row_id TEXT PRIMARY KEY,
          fixture_key TEXT NOT NULL,
          fixture_id INTEGER,
          player_key TEXT,
          api_player_id INTEGER,
          team_id INTEGER,
          team_name TEXT,
          team_slug TEXT,
          is_home INTEGER,
          position TEXT,
          position_group TEXT,
          minutes INTEGER,
          started_flag INTEGER,
          rating REAL,
          goals INTEGER,
          assists INTEGER,
          shots_total INTEGER,
          shots_on_target INTEGER,
          passes_key INTEGER,
          tackles INTEGER,
          duels_total INTEGER,
          duels_won INTEGER,
          yellow_cards INTEGER,
          red_cards INTEGER,
          payload_json TEXT NOT NULL
        );

        CREATE TABLE site_team_match_stats (
          row_id TEXT PRIMARY KEY,
          fixture_key TEXT NOT NULL,
          fixture_id INTEGER,
          team_id INTEGER,
          team_name TEXT,
          team_slug TEXT,
          is_home INTEGER,
          possession_pct REAL,
          shots_total INTEGER,
          shots_on_goal INTEGER,
          corners_for INTEGER,
          fouls_for INTEGER,
          yellow_cards INTEGER,
          red_cards INTEGER,
          passes_total INTEGER,
          passes_accurate INTEGER,
          payload_json TEXT NOT NULL
        );

        CREATE TABLE site_match_events (
          row_id TEXT PRIMARY KEY,
          fixture_key TEXT NOT NULL,
          fixture_id INTEGER,
          event_id INTEGER,
          minute INTEGER,
          extra_minute INTEGER,
          team_id INTEGER,
          team_name TEXT,
          is_home INTEGER,
          api_player_id INTEGER,
          player_key TEXT,
          player_name TEXT,
          event_type TEXT,
          event_detail TEXT,
          score_home_after INTEGER,
          score_away_after INTEGER,
          payload_json TEXT NOT NULL
        );

        CREATE TABLE site_lineup_slots (
          row_id TEXT PRIMARY KEY,
          fixture_key TEXT NOT NULL,
          fixture_id INTEGER,
          team_id INTEGER,
          team_name TEXT,
          team_slug TEXT,
          is_home INTEGER,
          player_key TEXT,
          api_player_id INTEGER,
          player_name TEXT,
          formation TEXT,
          is_starting_xi INTEGER,
          broad_position TEXT,
          position_group TEXT,
          slot_code TEXT,
          pitch_x REAL,
          pitch_y REAL,
          provider_rating REAL,
          rating_power INTEGER,
          rank_overall INTEGER,
          rank_position INTEGER,
          rank_club INTEGER,
          payload_json TEXT NOT NULL
        );

        CREATE TABLE site_formation_slots (
          formation TEXT NOT NULL,
          slot_code TEXT NOT NULL,
          broad_position TEXT,
          line_index INTEGER,
          slot_index INTEGER,
          pitch_x REAL,
          pitch_y REAL,
          PRIMARY KEY (formation, slot_code)
        );

        CREATE TABLE site_fixture_market_intelligence (
          row_id TEXT PRIMARY KEY,
          fixture_key TEXT NOT NULL,
          market_key TEXT NOT NULL,
          market_family TEXT,
          market_group TEXT,
          market_label TEXT,
          selection_label TEXT,
          rank_role TEXT,
          state TEXT,
          alignment_score INTEGER,
          rating INTEGER,
          band TEXT,
          model_lean TEXT,
          confidence_band TEXT,
          signal_state TEXT,
          support_count INTEGER,
          caution_count INTEGER,
          source_status TEXT,
          public_summary TEXT,
          payload_json TEXT NOT NULL
        );

        CREATE TABLE site_player_event_shortlists (
          row_id TEXT PRIMARY KEY,
          fixture_key TEXT NOT NULL,
          event_key TEXT NOT NULL,
          event_family TEXT,
          event_label TEXT,
          threshold REAL,
          player_key TEXT,
          api_player_id INTEGER,
          player_name TEXT,
          team_name TEXT,
          team_slug TEXT,
          is_home INTEGER,
          position TEXT,
          position_group TEXT,
          is_starting_xi INTEGER,
          shortlist_rank INTEGER,
          shortlist_score REAL,
          recent_per90 REAL,
          recent_average REAL,
          sample_size INTEGER,
          minutes_sample INTEGER,
          rating_power INTEGER,
          rank_overall INTEGER,
          rank_position INTEGER,
          rank_club INTEGER,
          source_lineup_status TEXT,
          beta_status TEXT,
          confidence_label TEXT,
          reason TEXT,
          payload_json TEXT NOT NULL
        );

        CREATE TABLE site_external_sources (
          source_id TEXT PRIMARY KEY,
          provider TEXT,
          usage_mode TEXT,
          terms_url TEXT,
          payload_json TEXT NOT NULL
        );

        CREATE TABLE site_fixture_external_content (
          row_id TEXT PRIMARY KEY,
          fixture_key TEXT NOT NULL,
          content_type TEXT,
          source_id TEXT,
          provider TEXT,
          priority INTEGER,
          payload_json TEXT NOT NULL
        );

        CREATE TABLE site_fixture_context_payloads (
          fixture_key TEXT PRIMARY KEY,
          payload_json TEXT NOT NULL
        );

        CREATE TABLE site_fixture_stats_payloads (
          fixture_key TEXT PRIMARY KEY,
          payload_json TEXT NOT NULL
        );

        CREATE TABLE site_team_premium_payloads (
          competition_key TEXT NOT NULL,
          team_slug TEXT NOT NULL,
          payload_json TEXT NOT NULL,
          PRIMARY KEY (competition_key, team_slug)
        );

        CREATE INDEX idx_fixtures_league_time ON fixtures(league_key, kickoff_time);
        CREATE INDEX idx_fixtures_home ON fixtures(home_team);
        CREATE INDEX idx_fixtures_away ON fixtures(away_team);
        CREATE INDEX idx_team_intelligence_team ON team_intelligence(team_slug);
        CREATE INDEX idx_club_squads_club ON club_squads(club_slug);
        CREATE INDEX idx_team_lineup_snapshots_team ON team_lineup_snapshots(team_key);
        CREATE INDEX idx_site_player_identity_api ON site_player_identity_map(api_player_id);
        CREATE INDEX idx_site_player_identity_club ON site_player_identity_map(competition_key, club_slug);
        CREATE INDEX idx_site_player_stats_fixture ON site_player_match_stats(fixture_key);
        CREATE INDEX idx_site_player_stats_player ON site_player_match_stats(player_key);
        CREATE INDEX idx_site_team_stats_fixture ON site_team_match_stats(fixture_key);
        CREATE INDEX idx_site_team_stats_team ON site_team_match_stats(team_slug);
        CREATE INDEX idx_site_match_events_fixture ON site_match_events(fixture_key);
        CREATE INDEX idx_site_match_events_player ON site_match_events(player_key);
        CREATE INDEX idx_site_lineup_slots_fixture ON site_lineup_slots(fixture_key);
        CREATE INDEX idx_site_lineup_slots_team ON site_lineup_slots(team_slug);
        CREATE INDEX idx_site_fixture_market_fixture ON site_fixture_market_intelligence(fixture_key);
        CREATE INDEX idx_site_fixture_market_market ON site_fixture_market_intelligence(market_key, rank_role);
        CREATE INDEX idx_site_player_event_fixture ON site_player_event_shortlists(fixture_key);
        CREATE INDEX idx_site_player_event_player ON site_player_event_shortlists(player_key);
        CREATE INDEX idx_site_player_event_team ON site_player_event_shortlists(team_slug, event_key);
        CREATE INDEX idx_site_fixture_external_fixture ON site_fixture_external_content(fixture_key);
        CREATE INDEX idx_site_fixture_external_source ON site_fixture_external_content(source_id, content_type);
        CREATE INDEX idx_site_team_premium_comp_team ON site_team_premium_payloads(competition_key, team_slug);
        """
    )


def insert_metadata(conn: sqlite3.Connection, rows: dict[str, Any]) -> None:
    conn.executemany(
        "INSERT INTO metadata(key, value) VALUES (?, ?)",
        [(key, str(value)) for key, value in rows.items()],
    )


def table_count(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


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


def insert_team_lineup_snapshots_from_normalized(
    conn: sqlite3.Connection,
    normalized_root: Path,
    identities: dict[str, dict[str, Any]],
    active_seasons: dict[str, str] | None = None,
    site_fixture_aliases: dict[tuple[str, str, str], str] | None = None,
) -> int:
    fixtures, team_names = fixture_context(normalized_root, active_seasons, site_fixture_aliases)
    by_api = identity_by_api_id(identities)
    by_team_fixture: dict[tuple[str, str], list[dict[str, Any]]] = {}
    context_by_group: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_normalized_family_rows(normalized_root, "lineups"):
        fixture_id = str(row.get("fixture_id") or "").strip()
        context = fixtures.get(fixture_id)
        if not context or not context.get("fixture_key"):
            continue
        if not normalized_row_in_scope(row, context, active_seasons):
            continue
        team_id = str(row.get("team_id") or "").strip()
        if not team_id:
            continue
        group_key = (fixture_id, team_id)
        by_team_fixture.setdefault(group_key, []).append(row)
        context_by_group[group_key] = context

    latest_by_team: dict[tuple[str, str], tuple[str, str, list[dict[str, Any]], dict[str, Any]]] = {}
    for (fixture_id, team_id), rows_for_team in by_team_fixture.items():
        context = context_by_group[(fixture_id, team_id)]
        team_name = team_names.get((fixture_id, team_id), "")
        if not team_name:
            continue
        team_key = slug_key(team_name)
        competition_key = competition_key_from_source(context.get("_source_file"), context.get("league"))
        match_date = str(context.get("match_date") or context.get("fixture_key") or "")
        latest_key = (competition_key, team_key)
        current = latest_by_team.get(latest_key)
        if current is None or match_date > current[0]:
            latest_by_team[latest_key] = (match_date, team_id, rows_for_team, context)

    insert_rows = []
    for (competition_key, team_key), (match_date, team_id, lineup_rows, context) in latest_by_team.items():
        team_name = team_names.get((str(context.get("fixture_id")), str(team_id)), "") or team_key.replace("_", " ").title()
        formation = str((lineup_rows[0] or {}).get("formation") or "").strip()
        starters_seen: dict[str, int] = {}
        players = []
        for idx, lineup_row in enumerate(lineup_rows):
            api_id = safe_int(lineup_row.get("player_id"))
            identity = by_api.get(api_id or -1)
            broad = broad_position(lineup_row.get("position"))
            is_starting = safe_int(lineup_row.get("is_starting_xi")) or 0
            slot = None
            if is_starting:
                starters_seen[broad] = starters_seen.get(broad, 0) + 1
                slot = slot_for_player(formation, broad, starters_seen[broad])
            players.append(
                {
                    "player_key": (identity or {}).get("player_key") or player_key_for(api_id, None, lineup_row.get("player_name")),
                    "api_player_id": api_id,
                    "name": (identity or {}).get("name") or lineup_row.get("player_name"),
                    "surname": str((identity or {}).get("name") or lineup_row.get("player_name") or "Player").split()[-1],
                    "lineup_position": broad,
                    "position": (identity or {}).get("position") or lineup_row.get("position"),
                    "position_group": (identity or {}).get("position_group") or position_group_from_broad(broad),
                    "is_starting_xi": is_starting,
                    "slot_code": (slot or {}).get("slot_code"),
                    "pitch_x": (slot or {}).get("pitch_x"),
                    "pitch_y": (slot or {}).get("pitch_y"),
                    "power": safe_int((identity or {}).get("rating_power")),
                    "rank_overall": safe_int((identity or {}).get("rank_overall")),
                    "rank_position": safe_int((identity or {}).get("rank_position")),
                    "rank_club": safe_int((identity or {}).get("rank_club")),
                    "source_order": idx,
                }
            )
        starters = [player for player in players if player["is_starting_xi"]]
        bench = [player for player in players if not player["is_starting_xi"]]
        opponent = context.get("away_team_name") if str(team_id) == str(context.get("home_team_id")) else context.get("home_team_name")
        payload = {
            "payload_type": "team_latest_lineup_snapshot",
            "competition_key": competition_key,
            "team_key": team_key,
            "team": team_name,
            "team_id": safe_int(team_id),
            "season": str(context.get("season") or ""),
            "formation": formation,
            "source_fixture_key": context.get("fixture_key"),
            "source_match_date": match_date,
            "source_opponent": opponent,
            "lineup_status": "last_fixture_snapshot",
            "lineup_mode": "team_snapshot_last_fixture",
            "lineup_label": "Last fixture lineup snapshot",
            "starters": starters,
            "bench": bench,
            "starter_count": len(starters),
            "bench_count": len(bench),
            "summary": f"Most recent normalized lineup snapshot for {team_name}.",
        }
        insert_rows.append(
            (
                competition_key,
                team_key,
                team_name,
                str(context.get("season") or ""),
                str(context.get("season") or ""),
                match_date,
                len(starters),
                len(bench),
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
        insert_rows,
    )
    return len(insert_rows)


def fixture_context(
    normalized_root: Path,
    active_seasons: dict[str, str] | None = None,
    site_fixture_aliases: dict[tuple[str, str, str], str] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], str]]:
    fixtures = read_normalized_family_rows(normalized_root, "fixtures_master", latest_per_competition=False)
    by_fixture_id: dict[str, dict[str, Any]] = {}
    team_names: dict[tuple[str, str], str] = {}
    for row in fixtures:
        fixture_id = str(row.get("fixture_id") or "").strip()
        if not fixture_id:
            continue
        if not normalized_row_in_scope(row, row, active_seasons):
            continue
        home_id = str(row.get("home_team_id") or "").strip()
        away_id = str(row.get("away_team_id") or "").strip()
        context = {
            "fixture_id": fixture_id,
            "fixture_key": row.get("fixture_key") or "",
            "provider_fixture_key": row.get("fixture_key") or "",
            "league": row.get("league") or "",
            "_source_file": row.get("_source_file") or "",
            "league_id": row.get("league_id") or "",
            "season": str(row.get("season") or ""),
            "match_date": row.get("match_date") or "",
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_team_name": row.get("home_team_name") or "",
            "away_team_name": row.get("away_team_name") or "",
        }
        context["fixture_key"] = resolve_active_site_fixture_key(context, site_fixture_aliases)
        by_fixture_id[fixture_id] = context
        if home_id:
            team_names[(fixture_id, home_id)] = context["home_team_name"]
        if away_id:
            team_names[(fixture_id, away_id)] = context["away_team_name"]
    return by_fixture_id, team_names


def team_name_for_row(row: dict[str, Any], fixtures: dict[str, dict[str, Any]], team_names: dict[tuple[str, str], str]) -> str:
    fixture_id = str(row.get("fixture_id") or "").strip()
    team_id = str(row.get("team_id") or "").strip()
    if row.get("team_name"):
        return str(row.get("team_name") or "")
    return team_names.get((fixture_id, team_id), "")


def is_home_for_row(row: dict[str, Any], context: dict[str, Any] | None) -> int | None:
    if row.get("is_home") not in {None, ""}:
        return safe_int(row.get("is_home"))
    if not context:
        return None
    team_id = str(row.get("team_id") or "").strip()
    if team_id and team_id == str(context.get("home_team_id") or ""):
        return 1
    if team_id and team_id == str(context.get("away_team_id") or ""):
        return 0
    return None


def build_rating_lookup(data_root: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    ratings = read_json(data_root / "player_intelligence" / "player_ratings.json", [])
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    rows = ratings if isinstance(ratings, list) else []
    for player in rows:
        club_slugs = slug_aliases(player.get("club_slug") or player.get("club"))
        for key in person_keys(player.get("name")).values():
            for club_slug in club_slugs:
                if club_slug and key:
                    lookup[(club_slug, key)] = player
    return lookup, rows


def resolve_rating_player(team_name: str, player_name: str, rating_lookup: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any] | None:
    keys = person_keys(player_name)
    for club_slug in slug_aliases(team_name):
        for key in (keys["full"], keys["initial_surname"], keys["surname"]):
            if club_slug and key and (club_slug, key) in rating_lookup:
                return rating_lookup[(club_slug, key)]
    return None


def player_key_for(api_player_id: Any, rating_player: dict[str, Any] | None, name: Any) -> str:
    if rating_player and rating_player.get("player_id"):
        return str(rating_player["player_id"])
    api_id = safe_int(api_player_id)
    if api_id is not None:
        return f"api_football_{api_id}"
    return f"unresolved_{slug_key(name)}"


def build_player_identity_rows(
    data_root: Path,
    normalized_root: Path,
    active_seasons: dict[str, str] | None = None,
    site_fixture_aliases: dict[tuple[str, str, str], str] | None = None,
) -> dict[str, dict[str, Any]]:
    fixtures, team_names = fixture_context(normalized_root, active_seasons, site_fixture_aliases)
    rating_lookup, rating_rows = build_rating_lookup(data_root)
    identities: dict[str, dict[str, Any]] = {}

    for rating_player in rating_rows:
        player_key = str(rating_player.get("player_id") or "")
        if not player_key:
            continue
        competition_key = rating_player.get("competition_key")
        season = str(rating_player.get("season") or "")
        if active_seasons is not None and active_seasons.get(competition_key) != season:
            continue
        ranks = rating_player.get("ranks") or {}
        payload = {
            "player_key": player_key,
            "api_player_id": None,
            "rating_player_id": player_key,
            "name": rating_player.get("name"),
            "canonical_name": normalize_text(rating_player.get("name")),
            "club": rating_player.get("club"),
            "club_slug": rating_player.get("club_slug") or slug_key(rating_player.get("club")),
            "competition_key": rating_player.get("competition_key"),
            "season": str(rating_player.get("season") or ""),
            "position": rating_player.get("position"),
            "position_group": rating_player.get("position_group"),
            "rating_power": safe_int((rating_player.get("ratings") or {}).get("og_player_power")),
            "rank_overall": safe_int(ranks.get("league_overall_rank")),
            "rank_position": safe_int(ranks.get("position_rank")),
            "rank_club": safe_int(ranks.get("club_rank")),
            "rating": rating_player,
        }
        identities[player_key] = payload

    observation_rows = read_normalized_family_rows(normalized_root, "lineups") + read_normalized_family_rows(normalized_root, "match_player_stats")
    for row in observation_rows:
        api_id = safe_int(row.get("player_id"))
        if api_id is None:
            continue
        context = fixtures.get(str(row.get("fixture_id") or "").strip())
        if not context or not normalized_row_in_scope(row, context, active_seasons):
            continue
        team_name = team_name_for_row(row, fixtures, team_names)
        rating_player = resolve_rating_player(team_name, row.get("player_name"), rating_lookup)
        player_key = player_key_for(api_id, rating_player, row.get("player_name"))
        existing = identities.get(player_key, {})
        ranks = (rating_player or {}).get("ranks") or {}
        identities[player_key] = {
            **existing,
            "player_key": player_key,
            "api_player_id": api_id,
            "rating_player_id": (rating_player or {}).get("player_id") or existing.get("rating_player_id"),
            "name": (rating_player or {}).get("name") or row.get("player_name") or existing.get("name"),
            "canonical_name": normalize_text((rating_player or {}).get("name") or row.get("player_name")),
            "club": (rating_player or {}).get("club") or team_name or existing.get("club"),
            "club_slug": (rating_player or {}).get("club_slug") or slug_key(team_name) or existing.get("club_slug"),
            "competition_key": (rating_player or {}).get("competition_key")
            or competition_key_from_source((context or {}).get("_source_file"), (context or {}).get("league"))
            or existing.get("competition_key"),
            "season": str((rating_player or {}).get("season") or (context or {}).get("season") or existing.get("season") or ""),
            "position": (rating_player or {}).get("position") or row.get("position") or existing.get("position"),
            "position_group": (rating_player or {}).get("position_group") or position_group_from_broad(row.get("position")) or existing.get("position_group"),
            "rating_power": safe_int(((rating_player or {}).get("ratings") or {}).get("og_player_power")) or existing.get("rating_power"),
            "rank_overall": safe_int(ranks.get("league_overall_rank")) or existing.get("rank_overall"),
            "rank_position": safe_int(ranks.get("position_rank")) or existing.get("rank_position"),
            "rank_club": safe_int(ranks.get("club_rank")) or existing.get("rank_club"),
            "rating": rating_player or existing.get("rating"),
        }
    return identities


def insert_site_player_identity_map(
    conn: sqlite3.Connection,
    data_root: Path,
    normalized_root: Path,
    active_seasons: dict[str, str] | None = None,
    site_fixture_aliases: dict[tuple[str, str, str], str] | None = None,
) -> tuple[int, dict[str, dict[str, Any]]]:
    identities = build_player_identity_rows(data_root, normalized_root, active_seasons, site_fixture_aliases)
    rows = []
    for identity in identities.values():
        rows.append(
            (
                identity.get("player_key"),
                identity.get("api_player_id"),
                identity.get("rating_player_id"),
                identity.get("name"),
                identity.get("canonical_name"),
                identity.get("club"),
                identity.get("club_slug"),
                identity.get("competition_key"),
                str(identity.get("season") or ""),
                identity.get("position"),
                identity.get("position_group"),
                safe_int(identity.get("rating_power")),
                safe_int(identity.get("rank_overall")),
                safe_int(identity.get("rank_position")),
                safe_int(identity.get("rank_club")),
                json_text(identity),
            )
        )
    conn.executemany(
        """
        INSERT OR REPLACE INTO site_player_identity_map(
          player_key, api_player_id, rating_player_id, name, canonical_name,
          club, club_slug, competition_key, season, position, position_group,
          rating_power, rank_overall, rank_position, rank_club, payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows), identities


def identity_by_api_id(identities: dict[str, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {
        int(identity["api_player_id"]): identity
        for identity in identities.values()
        if safe_int(identity.get("api_player_id")) is not None
    }


def compact_identity(identity: dict[str, Any] | None) -> dict[str, Any] | None:
    if not identity:
        return None
    return {
        "player_key": identity.get("player_key"),
        "api_player_id": identity.get("api_player_id"),
        "rating_player_id": identity.get("rating_player_id"),
        "name": identity.get("name"),
        "position": identity.get("position"),
        "position_group": identity.get("position_group"),
        "rating_power": identity.get("rating_power"),
        "rank_overall": identity.get("rank_overall"),
        "rank_position": identity.get("rank_position"),
        "rank_club": identity.get("rank_club"),
    }


def insert_site_player_match_stats(
    conn: sqlite3.Connection,
    normalized_root: Path,
    identities: dict[str, dict[str, Any]],
    active_seasons: dict[str, str] | None = None,
    site_fixture_aliases: dict[tuple[str, str, str], str] | None = None,
) -> int:
    fixtures, team_names = fixture_context(normalized_root, active_seasons, site_fixture_aliases)
    by_api = identity_by_api_id(identities)
    rows = []
    for idx, row in enumerate(read_normalized_family_rows(normalized_root, "match_player_stats")):
        fixture_id = str(row.get("fixture_id") or "").strip()
        context = fixtures.get(fixture_id)
        if not context or not context.get("fixture_key"):
            continue
        if not normalized_row_in_scope(row, context, active_seasons):
            continue
        api_id = safe_int(row.get("player_id"))
        identity = by_api.get(api_id or -1)
        team_name = team_name_for_row(row, fixtures, team_names)
        player_key = (identity or {}).get("player_key") or player_key_for(api_id, None, row.get("player_name"))
        payload = {
            **row,
            "fixture_key": context.get("fixture_key"),
            "team_name": team_name,
            "team_slug": slug_key(team_name),
            "player_key": player_key,
            "identity": compact_identity(identity),
        }
        rows.append(
            (
                f"{context['fixture_key']}:{row.get('team_id')}:{api_id or idx}",
                context.get("fixture_key"),
                safe_int(fixture_id),
                player_key,
                api_id,
                safe_int(row.get("team_id")),
                team_name,
                slug_key(team_name),
                is_home_for_row(row, context),
                row.get("position"),
                (identity or {}).get("position_group") or position_group_from_broad(row.get("position")),
                safe_int(row.get("minutes")),
                safe_int(row.get("started_flag")),
                safe_float(row.get("rating")),
                safe_int(row.get("goals")),
                safe_int(row.get("assists")),
                safe_int(row.get("shots_total")),
                safe_int(row.get("shots_on_target")),
                safe_int(row.get("passes_key")),
                safe_int(row.get("tackles")),
                safe_int(row.get("duels_total")),
                safe_int(row.get("duels_won")),
                safe_int(row.get("yellow_cards")),
                safe_int(row.get("red_cards")),
                json_text(payload),
            )
        )
    conn.executemany(
        """
        INSERT OR REPLACE INTO site_player_match_stats(
          row_id, fixture_key, fixture_id, player_key, api_player_id, team_id,
          team_name, team_slug, is_home, position, position_group, minutes,
          started_flag, rating, goals, assists, shots_total, shots_on_target,
          passes_key, tackles, duels_total, duels_won, yellow_cards, red_cards,
          payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def insert_site_team_match_stats(
    conn: sqlite3.Connection,
    normalized_root: Path,
    active_seasons: dict[str, str] | None = None,
    site_fixture_aliases: dict[tuple[str, str, str], str] | None = None,
) -> int:
    fixtures, _team_names = fixture_context(normalized_root, active_seasons, site_fixture_aliases)
    rows = []
    for idx, row in enumerate(read_normalized_family_rows(normalized_root, "match_team_stats")):
        fixture_id = str(row.get("fixture_id") or "").strip()
        context = fixtures.get(fixture_id)
        if not context or not context.get("fixture_key"):
            continue
        if not normalized_row_in_scope(row, context, active_seasons):
            continue
        team_name = row.get("team_name") or ""
        payload = {**row, "fixture_key": context.get("fixture_key"), "team_slug": slug_key(team_name)}
        rows.append(
            (
                f"{context['fixture_key']}:{row.get('team_id') or idx}",
                context.get("fixture_key"),
                safe_int(fixture_id),
                safe_int(row.get("team_id")),
                team_name,
                slug_key(team_name),
                safe_int(row.get("is_home")),
                safe_float(row.get("possession_pct")),
                safe_int(row.get("shots_total")),
                safe_int(row.get("shots_on_goal")),
                safe_int(row.get("corners_for")),
                safe_int(row.get("fouls_for")),
                safe_int(row.get("yellow_cards")),
                safe_int(row.get("red_cards")),
                safe_int(row.get("passes_total")),
                safe_int(row.get("passes_accurate")),
                json_text(payload),
            )
        )
    conn.executemany(
        """
        INSERT OR REPLACE INTO site_team_match_stats(
          row_id, fixture_key, fixture_id, team_id, team_name, team_slug, is_home,
          possession_pct, shots_total, shots_on_goal, corners_for, fouls_for,
          yellow_cards, red_cards, passes_total, passes_accurate, payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def insert_site_match_events(
    conn: sqlite3.Connection,
    normalized_root: Path,
    identities: dict[str, dict[str, Any]],
    active_seasons: dict[str, str] | None = None,
    site_fixture_aliases: dict[tuple[str, str, str], str] | None = None,
) -> int:
    fixtures, _team_names = fixture_context(normalized_root, active_seasons, site_fixture_aliases)
    by_api = identity_by_api_id(identities)
    rows = []
    for idx, row in enumerate(read_normalized_family_rows(normalized_root, "match_events")):
        fixture_id = str(row.get("fixture_id") or "").strip()
        context = fixtures.get(fixture_id)
        if not context or not context.get("fixture_key"):
            continue
        if not normalized_row_in_scope(row, context, active_seasons):
            continue
        api_id = safe_int(row.get("player_id"))
        identity = by_api.get(api_id or -1)
        is_home = safe_int(row.get("is_home"))
        team_name = context.get("home_team_name") if is_home == 1 else context.get("away_team_name") if is_home == 0 else ""
        player_key = (identity or {}).get("player_key") or player_key_for(api_id, None, row.get("player_id"))
        payload = {
            **row,
            "fixture_key": context.get("fixture_key"),
            "provider_fixture_key": context.get("provider_fixture_key"),
            "team_name": team_name,
            "player_key": player_key,
            "player_name": (identity or {}).get("name"),
            "identity": compact_identity(identity),
        }
        rows.append(
            (
                f"{context['fixture_key']}:{row.get('event_id') or idx}",
                context.get("fixture_key"),
                safe_int(fixture_id),
                safe_int(row.get("event_id")),
                safe_int(row.get("minute")),
                safe_int(row.get("extra_minute")),
                safe_int(row.get("team_id")),
                team_name,
                is_home,
                api_id,
                player_key,
                (identity or {}).get("name"),
                row.get("event_type"),
                row.get("event_detail"),
                safe_int(row.get("score_home_after")),
                safe_int(row.get("score_away_after")),
                json_text(payload),
            )
        )
    conn.executemany(
        """
        INSERT OR REPLACE INTO site_match_events(
          row_id, fixture_key, fixture_id, event_id, minute, extra_minute,
          team_id, team_name, is_home, api_player_id, player_key, player_name,
          event_type, event_detail, score_home_after, score_away_after, payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def formation_lines(formation: Any) -> list[int]:
    values = [int(part) for part in re.findall(r"\d+", str(formation or ""))]
    return values if values else [4, 4, 2]


def formation_slots_for(formation: str) -> list[dict[str, Any]]:
    lines = formation_lines(formation)
    slots = [{"slot_code": "G1", "broad_position": "G", "line_index": 0, "slot_index": 1, "pitch_x": 7.0, "pitch_y": 50.0}]
    if not lines:
        return slots
    x_values = [22.0]
    if len(lines) > 1:
        span = 68.0
        x_values = [22.0 + (span * i / max(1, len(lines) - 1)) for i in range(len(lines))]
    for line_index, count in enumerate(lines, start=1):
        broad = "D" if line_index == 1 else "F" if line_index == len(lines) else "M"
        for slot_index in range(1, count + 1):
            y = 50.0 if count == 1 else 14.0 + ((slot_index - 1) * 72.0 / max(1, count - 1))
            prefix = broad if broad in {"D", "F"} else f"M{line_index - 1}"
            slots.append(
                {
                    "slot_code": f"{prefix}{slot_index}",
                    "broad_position": broad,
                    "line_index": line_index,
                    "slot_index": slot_index,
                    "pitch_x": round(x_values[line_index - 1], 2),
                    "pitch_y": round(y, 2),
                }
            )
    return slots


def slot_for_player(formation: str, broad: str, ordinal: int) -> dict[str, Any] | None:
    slots = [slot for slot in formation_slots_for(formation) if slot["broad_position"] == broad]
    if not slots:
        return None
    return slots[min(max(ordinal - 1, 0), len(slots) - 1)]


def insert_site_formation_slots(
    conn: sqlite3.Connection,
    normalized_root: Path,
    active_seasons: dict[str, str] | None = None,
    site_fixture_aliases: dict[tuple[str, str, str], str] | None = None,
) -> int:
    fixtures, _team_names = fixture_context(normalized_root, active_seasons, site_fixture_aliases)
    formations = {
        str(row.get("formation") or "").strip()
        for row in read_normalized_family_rows(normalized_root, "lineups")
        if str(row.get("formation") or "").strip()
        and normalized_row_in_scope(row, fixtures.get(str(row.get("fixture_id") or "").strip()), active_seasons)
    }
    rows = []
    for formation in sorted(formations):
        for slot in formation_slots_for(formation):
            rows.append(
                (
                    formation,
                    slot["slot_code"],
                    slot["broad_position"],
                    slot["line_index"],
                    slot["slot_index"],
                    slot["pitch_x"],
                    slot["pitch_y"],
                )
            )
    conn.executemany(
        """
        INSERT OR REPLACE INTO site_formation_slots(
          formation, slot_code, broad_position, line_index, slot_index, pitch_x, pitch_y
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def insert_site_lineup_slots(
    conn: sqlite3.Connection,
    normalized_root: Path,
    identities: dict[str, dict[str, Any]],
    active_seasons: dict[str, str] | None = None,
    site_fixture_aliases: dict[tuple[str, str, str], str] | None = None,
) -> int:
    fixtures, team_names = fixture_context(normalized_root, active_seasons, site_fixture_aliases)
    by_api = identity_by_api_id(identities)
    lineup_rows = read_normalized_family_rows(normalized_root, "lineups")
    starters_seen: dict[tuple[str, str, str], int] = {}
    rows = []
    for idx, row in enumerate(lineup_rows):
        fixture_id = str(row.get("fixture_id") or "").strip()
        context = fixtures.get(fixture_id)
        if not context or not context.get("fixture_key"):
            continue
        if not normalized_row_in_scope(row, context, active_seasons):
            continue
        api_id = safe_int(row.get("player_id"))
        identity = by_api.get(api_id or -1)
        team_name = team_name_for_row(row, fixtures, team_names)
        broad = broad_position(row.get("position"))
        is_starting = safe_int(row.get("is_starting_xi")) or 0
        slot = None
        if is_starting:
            key = (fixture_id, str(row.get("team_id") or ""), broad)
            starters_seen[key] = starters_seen.get(key, 0) + 1
            slot = slot_for_player(row.get("formation") or "", broad, starters_seen[key])
        player_key = (identity or {}).get("player_key") or player_key_for(api_id, None, row.get("player_name"))
        payload = {
            **row,
            "fixture_key": context.get("fixture_key"),
            "team_name": team_name,
            "team_slug": slug_key(team_name),
            "player_key": player_key,
            "identity": compact_identity(identity),
            "slot": slot,
        }
        rows.append(
            (
                f"{context['fixture_key']}:{row.get('team_id')}:{api_id or idx}:{safe_int(row.get('is_starting_xi')) or 0}",
                context.get("fixture_key"),
                safe_int(fixture_id),
                safe_int(row.get("team_id")),
                team_name,
                slug_key(team_name),
                is_home_for_row(row, context),
                player_key,
                api_id,
                row.get("player_name"),
                row.get("formation"),
                is_starting,
                broad,
                (identity or {}).get("position_group") or position_group_from_broad(broad),
                (slot or {}).get("slot_code"),
                (slot or {}).get("pitch_x"),
                (slot or {}).get("pitch_y"),
                None,
                safe_int((identity or {}).get("rating_power")),
                safe_int((identity or {}).get("rank_overall")),
                safe_int((identity or {}).get("rank_position")),
                safe_int((identity or {}).get("rank_club")),
                json_text(payload),
            )
        )
    conn.executemany(
        """
        INSERT OR REPLACE INTO site_lineup_slots(
          row_id, fixture_key, fixture_id, team_id, team_name, team_slug, is_home,
          player_key, api_player_id, player_name, formation, is_starting_xi,
          broad_position, position_group, slot_code, pitch_x, pitch_y,
          provider_rating, rating_power, rank_overall, rank_position, rank_club,
          payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


MARKET_META = {
    "ftr": ("FTR", "core_goal_market", "Full Time Result"),
    "ou25": ("OU25", "core_goal_market", "Over / Under 2.5 Match Goals"),
    "btts": ("BTTS", "core_goal_market", "Both Teams To Score"),
    "team_goals": ("TEAM_GOALS", "team_goal_market", "Team Goals 1.5+"),
    "correct_score": ("CORRECT_SCORE", "scoreline_market", "Correct Score"),
    "corners": ("CORNERS", "team_event_market", "Corners"),
    "cards": ("CARDS", "team_event_market", "Cards"),
}


def market_rank_role(market_key: str, state: Any, rating: Any, alignment_score: Any) -> str:
    state_key = str(state or "").strip().upper()
    score = safe_int(alignment_score) or safe_int(rating) or 0
    if state_key in {"AVOID", "RED_FLAG"} or score < 50:
        return "avoid"
    if state_key in {"SUPPORTED", "DEPLOY"} or score >= 82:
        return "best"
    if score >= 68:
        return "secondary"
    if market_key in {"ftr", "ou25", "btts", "team_goals"}:
        return "weak"
    return "context"


def insert_site_fixture_market_intelligence(conn: sqlite3.Connection) -> int:
    rows = []
    for fixture_key, signal_state, confidence_band, agreement_score, payload_json in conn.execute(
        """
        SELECT fixture_key, signal_state, confidence_band, agreement_score, payload_json
        FROM fixture_decisions
        ORDER BY fixture_key
        """
    ):
        try:
            payload = json.loads(payload_json)
        except (TypeError, json.JSONDecodeError):
            continue
        intelligence = payload.get("market_intelligence") if isinstance(payload.get("market_intelligence"), dict) else {}
        suitability = payload.get("market_suitability") if isinstance(payload.get("market_suitability"), dict) else {}
        market_keys = sorted(set(intelligence) | set(suitability))
        for market_key in market_keys:
            market_read = intelligence.get(market_key) if isinstance(intelligence.get(market_key), dict) else {}
            suitability_read = suitability.get(market_key) if isinstance(suitability.get(market_key), dict) else {}
            market_family, market_group, market_label = MARKET_META.get(
                market_key,
                (market_key.upper(), "other_market", market_key.replace("_", " ").title()),
            )
            support_tokens = market_read.get("structural_support") if isinstance(market_read.get("structural_support"), list) else []
            caution_tokens = market_read.get("cautions") if isinstance(market_read.get("cautions"), list) else []
            alignment = safe_int(market_read.get("alignment_score"))
            rating = safe_int(market_read.get("rating") or suitability_read.get("rating"))
            state = market_read.get("state") or suitability_read.get("label")
            row_payload = {
                "fixture_key": fixture_key,
                "market_key": market_key,
                "market_family": market_family,
                "market_group": market_group,
                "market_label": market_label,
                "selection_label": market_read.get("model_lean"),
                "rank_role": market_rank_role(market_key, state, rating, alignment),
                "state": state,
                "alignment_score": alignment,
                "rating": rating,
                "band": market_read.get("band") or suitability_read.get("label"),
                "model_lean": market_read.get("model_lean"),
                "confidence_band": confidence_band,
                "signal_state": signal_state,
                "agreement_score": safe_int(agreement_score),
                "structural_support": support_tokens,
                "cautions": caution_tokens,
                "suitability": suitability_read,
                "public_summary": market_read.get("public_summary") or suitability_read.get("read"),
                "source_status": "fixture_decision_reconciler",
                "product_tier_hint": "core_public" if market_key in {"ftr", "ou25", "btts", "team_goals"} else "premium_market_context",
            }
            rows.append(
                (
                    f"{fixture_key}:{market_key}",
                    fixture_key,
                    market_key,
                    market_family,
                    market_group,
                    market_label,
                    market_read.get("model_lean"),
                    row_payload["rank_role"],
                    state,
                    alignment,
                    rating,
                    row_payload["band"],
                    market_read.get("model_lean"),
                    confidence_band,
                    signal_state,
                    len(support_tokens),
                    len(caution_tokens),
                    "fixture_decision_reconciler",
                    row_payload["public_summary"],
                    json_text(row_payload),
                )
            )
    conn.executemany(
        """
        INSERT OR REPLACE INTO site_fixture_market_intelligence(
          row_id, fixture_key, market_key, market_family, market_group,
          market_label, selection_label, rank_role, state, alignment_score,
          rating, band, model_lean, confidence_band, signal_state,
          support_count, caution_count, source_status, public_summary, payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


PLAYER_EVENT_CONFIG = [
    {
        "event_key": "shots_on_target_0_5",
        "event_family": "shots_on_target",
        "event_label": "Shots On Target 0.5+",
        "stat_field": "shots_on_target",
        "threshold": 0.5,
        "benchmark_per90": 0.65,
        "position_groups": {"forward", "midfielder"},
    },
    {
        "event_key": "shots_on_target_1_5",
        "event_family": "shots_on_target",
        "event_label": "Shots On Target 1.5+",
        "stat_field": "shots_on_target",
        "threshold": 1.5,
        "benchmark_per90": 1.20,
        "position_groups": {"forward"},
    },
    {
        "event_key": "shots_1_5",
        "event_family": "shots",
        "event_label": "Shots 1.5+",
        "stat_field": "shots_total",
        "threshold": 1.5,
        "benchmark_per90": 1.80,
        "position_groups": {"forward", "midfielder"},
    },
    {
        "event_key": "shots_2_5",
        "event_family": "shots",
        "event_label": "Shots 2.5+",
        "stat_field": "shots_total",
        "threshold": 2.5,
        "benchmark_per90": 2.60,
        "position_groups": {"forward"},
    },
    {
        "event_key": "tackles_0_5",
        "event_family": "tackles",
        "event_label": "Tackles 0.5+",
        "stat_field": "tackles",
        "threshold": 0.5,
        "benchmark_per90": 1.40,
        "position_groups": {"defender", "midfielder"},
    },
    {
        "event_key": "tackles_1_5",
        "event_family": "tackles",
        "event_label": "Tackles 1.5+",
        "stat_field": "tackles",
        "threshold": 1.5,
        "benchmark_per90": 2.20,
        "position_groups": {"defender", "midfielder"},
    },
    {
        "event_key": "fouls_committed_0_5",
        "event_family": "fouls",
        "event_label": "Fouls 0.5+",
        "stat_field": "fouls_committed",
        "threshold": 0.5,
        "benchmark_per90": 1.20,
        "position_groups": {"defender", "midfielder", "forward"},
    },
    {
        "event_key": "fouls_committed_1_5",
        "event_family": "fouls",
        "event_label": "Fouls 1.5+",
        "stat_field": "fouls_committed",
        "threshold": 1.5,
        "benchmark_per90": 2.00,
        "position_groups": {"defender", "midfielder"},
    },
    {
        "event_key": "player_fouled_0_5",
        "event_family": "player_fouled",
        "event_label": "Player To Be Fouled 0.5+",
        "stat_field": "fouls_drawn",
        "threshold": 0.5,
        "benchmark_per90": 1.20,
        "position_groups": {"forward", "midfielder"},
    },
    {
        "event_key": "bookings",
        "event_family": "bookings",
        "event_label": "Player Booking",
        "stat_field": "yellow_cards",
        "threshold": 0.5,
        "benchmark_per90": 0.30,
        "position_groups": {"defender", "midfielder"},
    },
    {
        "event_key": "key_passes_0_5",
        "event_family": "key_passes",
        "event_label": "Key Passes 0.5+",
        "stat_field": "passes_key",
        "threshold": 0.5,
        "benchmark_per90": 1.25,
        "position_groups": {"midfielder", "forward"},
    },
    {
        "event_key": "goalkeeper_saves_1_5",
        "event_family": "goalkeeper_saves",
        "event_label": "Goalkeeper Saves 1.5+",
        "stat_field": "saves",
        "threshold": 1.5,
        "benchmark_per90": 2.80,
        "position_groups": {"goalkeeper"},
    },
]


def player_event_profiles(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    fields = {config["stat_field"] for config in PLAYER_EVENT_CONFIG}
    for player_key, payload_json in conn.execute(
        """
        SELECT player_key, payload_json
        FROM site_player_match_stats
        WHERE player_key IS NOT NULL
        """
    ):
        try:
            payload = json.loads(payload_json)
        except (TypeError, json.JSONDecodeError):
            continue
        profile = profiles.setdefault(
            player_key,
            {"sample_size": 0, "minutes": 0.0, "totals": {field: 0.0 for field in fields}},
        )
        minutes = safe_float(payload.get("minutes")) or 0.0
        if minutes <= 0:
            continue
        profile["sample_size"] += 1
        profile["minutes"] += minutes
        for field in fields:
            profile["totals"][field] = profile["totals"].get(field, 0.0) + (safe_float(payload.get(field)) or 0.0)
    return profiles


def team_lineup_candidates(
    conn: sqlite3.Connection,
    fixture_key: str,
    competition_key: str,
    team_name: str,
    is_home: int,
) -> list[dict[str, Any]]:
    team_key = slug_key(team_name)
    team_aliases = sorted(slug_aliases(team_name))
    team_alias_placeholders = ", ".join("?" for _ in team_aliases) or "?"
    snapshot_rows = conn.execute(
        f"""
        SELECT payload_json
        FROM team_lineup_snapshots
        WHERE competition_key = ? AND team_key IN ({team_alias_placeholders})
        """,
        (competition_key, *(team_aliases or [team_key])),
    ).fetchall()
    if not snapshot_rows:
        snapshot_rows = conn.execute(
            f"""
            SELECT payload_json
            FROM team_lineup_snapshots
            WHERE team_key IN ({team_alias_placeholders})
            ORDER BY competition_key
            LIMIT 1
            """,
            tuple(team_aliases or [team_key]),
        ).fetchall()
    if snapshot_rows:
        try:
            snapshot = json.loads(snapshot_rows[0][0])
        except (TypeError, json.JSONDecodeError):
            snapshot = {}
        candidates = []
        for player in snapshot.get("starters") or []:
            candidates.append(
                {
                    "fixture_key": fixture_key,
                    "team_name": team_name,
                    "team_slug": slug_key(team_name),
                    "is_home": is_home,
                    "player_key": player.get("player_key"),
                    "api_player_id": safe_int(player.get("api_player_id")),
                    "player_name": player.get("name") or player.get("player_name"),
                    "position": player.get("position") or player.get("lineup_position"),
                    "position_group": player.get("position_group") or position_group_from_broad(player.get("lineup_position")),
                    "is_starting_xi": 1,
                    "rating_power": safe_int(player.get("power") or player.get("rating_power")),
                    "rank_overall": safe_int(player.get("rank_overall")),
                    "rank_position": safe_int(player.get("rank_position")),
                    "rank_club": safe_int(player.get("rank_club")),
                    "source_lineup_status": snapshot.get("lineup_status") or "last_fixture_snapshot",
                }
            )
        return candidates

    rows = conn.execute(
        f"""
        SELECT player_key, api_player_id, player_name, broad_position, position_group,
               is_starting_xi, rating_power, rank_overall, rank_position, rank_club
        FROM site_lineup_slots
        WHERE fixture_key = ? AND team_slug IN ({team_alias_placeholders}) AND is_starting_xi = 1
        ORDER BY broad_position, slot_code, player_name
        """,
        (fixture_key, *(team_aliases or [team_key])),
    ).fetchall()
    return [
        {
            "fixture_key": fixture_key,
            "team_name": team_name,
            "team_slug": team_key,
            "is_home": is_home,
            "player_key": row[0],
            "api_player_id": row[1],
            "player_name": row[2],
            "position": row[3],
            "position_group": row[4],
            "is_starting_xi": row[5],
            "rating_power": row[6],
            "rank_overall": row[7],
            "rank_position": row[8],
            "rank_club": row[9],
            "source_lineup_status": "confirmed_or_fixture_lineup",
        }
        for row in rows
    ]


def player_event_score(candidate: dict[str, Any], profile: dict[str, Any], config: dict[str, Any]) -> tuple[float, float, float]:
    minutes = float(profile.get("minutes") or 0.0)
    sample_size = int(profile.get("sample_size") or 0)
    total = float((profile.get("totals") or {}).get(config["stat_field"]) or 0.0)
    per90 = (total / minutes * 90.0) if minutes > 0 else 0.0
    average = (total / sample_size) if sample_size > 0 else 0.0
    benchmark = float(config["benchmark_per90"] or 1.0)
    base = min(1.0, per90 / benchmark) * 74.0 if benchmark > 0 else 0.0
    rating_boost = min(12.0, max(0.0, ((safe_int(candidate.get("rating_power")) or 50) - 50) / 50 * 12.0))
    starter_boost = 8.0 if safe_int(candidate.get("is_starting_xi")) else 0.0
    sample_boost = min(6.0, sample_size * 1.5)
    return round(min(99.0, base + rating_boost + starter_boost + sample_boost), 2), round(per90, 3), round(average, 3)


def insert_site_player_event_shortlists(conn: sqlite3.Connection, limit: int = DEFAULT_EVENT_SHORTLIST_LIMIT) -> int:
    profiles = player_event_profiles(conn)
    fixture_rows = conn.execute(
        """
        SELECT f.fixture_key,
               COALESCE(fl.competition_key, f.league_key) AS competition_key,
               f.home_team,
               f.away_team
        FROM fixtures f
        LEFT JOIN fixture_lineups fl ON fl.fixture_key = f.fixture_key
        ORDER BY f.kickoff_time, f.fixture_key
        """
    ).fetchall()
    rows = []
    for fixture_key, competition_key, home_team, away_team in fixture_rows:
        candidates = []
        candidates.extend(team_lineup_candidates(conn, fixture_key, competition_key or "", home_team or "", 1))
        candidates.extend(team_lineup_candidates(conn, fixture_key, competition_key or "", away_team or "", 0))
        for config in PLAYER_EVENT_CONFIG:
            scored: list[tuple[float, float, float, dict[str, Any], dict[str, Any]]] = []
            allowed_groups = config.get("position_groups") or set()
            for candidate in candidates:
                player_key = candidate.get("player_key")
                if not player_key or player_key not in profiles:
                    continue
                position_group = str(candidate.get("position_group") or "").lower()
                if allowed_groups and position_group not in allowed_groups:
                    continue
                profile = profiles[player_key]
                if int(profile.get("sample_size") or 0) <= 0:
                    continue
                score, per90, average = player_event_score(candidate, profile, config)
                if score <= 12:
                    continue
                scored.append((score, per90, average, candidate, profile))
            scored.sort(key=lambda item: (item[0], item[1], safe_int(item[3].get("rating_power")) or 0), reverse=True)
            for rank, (score, per90, average, candidate, profile) in enumerate(scored[:limit], start=1):
                sample_size = int(profile.get("sample_size") or 0)
                minutes_sample = int(round(float(profile.get("minutes") or 0.0)))
                reason = (
                    f"{per90:g} {config['event_family'].replace('_', ' ')} per 90 "
                    f"from {sample_size} current-season sample{'s' if sample_size != 1 else ''}; "
                    f"lineup source is {candidate.get('source_lineup_status') or 'unknown'}."
                )
                payload = {
                    **candidate,
                    "event_key": config["event_key"],
                    "event_family": config["event_family"],
                    "event_label": config["event_label"],
                    "threshold": config["threshold"],
                    "shortlist_rank": rank,
                    "shortlist_score": score,
                    "recent_per90": per90,
                    "recent_average": average,
                    "sample_size": sample_size,
                    "minutes_sample": minutes_sample,
                    "beta_status": "beta_shortlist",
                    "confidence_label": "manual_review",
                    "source_status": "current_season_recent_stats_and_latest_lineup",
                    "priced_probability": None,
                    "deployable": False,
                    "reason": reason,
                    "product_tier_hint": "premium_player_events",
                }
                rows.append(
                    (
                        f"{fixture_key}:{config['event_key']}:{candidate.get('player_key')}:{rank}",
                        fixture_key,
                        config["event_key"],
                        config["event_family"],
                        config["event_label"],
                        config["threshold"],
                        candidate.get("player_key"),
                        safe_int(candidate.get("api_player_id")),
                        candidate.get("player_name"),
                        candidate.get("team_name"),
                        candidate.get("team_slug"),
                        safe_int(candidate.get("is_home")),
                        candidate.get("position"),
                        candidate.get("position_group"),
                        safe_int(candidate.get("is_starting_xi")),
                        rank,
                        score,
                        per90,
                        average,
                        sample_size,
                        minutes_sample,
                        safe_int(candidate.get("rating_power")),
                        safe_int(candidate.get("rank_overall")),
                        safe_int(candidate.get("rank_position")),
                        safe_int(candidate.get("rank_club")),
                        candidate.get("source_lineup_status"),
                        "beta_shortlist",
                        "manual_review",
                        reason,
                        json_text(payload),
                    )
                )
    conn.executemany(
        """
        INSERT OR REPLACE INTO site_player_event_shortlists(
          row_id, fixture_key, event_key, event_family, event_label, threshold,
          player_key, api_player_id, player_name, team_name, team_slug, is_home,
          position, position_group, is_starting_xi, shortlist_rank,
          shortlist_score, recent_per90, recent_average, sample_size,
          minutes_sample, rating_power, rank_overall, rank_position, rank_club,
          source_lineup_status, beta_status, confidence_label, reason, payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def json_rows(conn: sqlite3.Connection, query: str, params: Iterable[Any] = ()) -> list[Any]:
    rows = conn.execute(query, tuple(params)).fetchall()
    payloads = []
    for row in rows:
        try:
            payloads.append(json.loads(row[0]))
        except (TypeError, json.JSONDecodeError):
            continue
    return payloads


def insert_site_fixture_stats_payloads(conn: sqlite3.Connection) -> int:
    fixture_keys = [row[0] for row in conn.execute("SELECT fixture_key FROM fixtures ORDER BY kickoff_time, fixture_key")]
    rows = []
    for fixture_key in fixture_keys:
        payload = {
            "team_stats": json_rows(
                conn,
                """
                SELECT payload_json
                FROM site_team_match_stats
                WHERE fixture_key = ?
                ORDER BY is_home DESC, team_name
                """,
                (fixture_key,),
            ),
            "player_stats": json_rows(
                conn,
                """
                SELECT payload_json
                FROM site_player_match_stats
                WHERE fixture_key = ?
                ORDER BY is_home DESC, started_flag DESC, minutes DESC, rating DESC, player_key
                """,
                (fixture_key,),
            ),
            "match_events": json_rows(
                conn,
                """
                SELECT payload_json
                FROM site_match_events
                WHERE fixture_key = ?
                ORDER BY minute, extra_minute, event_id
                """,
                (fixture_key,),
            ),
            "lineup_slots": json_rows(
                conn,
                """
                SELECT payload_json
                FROM site_lineup_slots
                WHERE fixture_key = ?
                ORDER BY is_home DESC, is_starting_xi DESC, broad_position, slot_code, player_name
                """,
                (fixture_key,),
            ),
            "market_intelligence": json_rows(
                conn,
                """
                SELECT payload_json
                FROM site_fixture_market_intelligence
                WHERE fixture_key = ?
                ORDER BY
                  CASE rank_role
                    WHEN 'best' THEN 1
                    WHEN 'secondary' THEN 2
                    WHEN 'weak' THEN 3
                    WHEN 'avoid' THEN 4
                    ELSE 5
                  END,
                  alignment_score DESC,
                  rating DESC,
                  market_key
                """,
                (fixture_key,),
            ),
            "player_event_shortlists": json_rows(
                conn,
                """
                SELECT payload_json
                FROM site_player_event_shortlists
                WHERE fixture_key = ?
                ORDER BY event_family, event_key, shortlist_rank, team_name, player_name
                """,
                (fixture_key,),
            ),
        }
        rows.append((fixture_key, json_text(payload)))
    conn.executemany(
        "INSERT OR REPLACE INTO site_fixture_stats_payloads(fixture_key, payload_json) VALUES (?, ?)",
        rows,
    )
    return len(rows)


def insert_site_team_premium_payloads(conn: sqlite3.Connection, limit: int = DEFAULT_ROUTE_CACHE_LIMIT) -> int:
    teams = conn.execute(
        """
        SELECT competition_key, club_slug
        FROM club_squads
        ORDER BY competition_key, club_slug
        """
    ).fetchall()
    rows = []
    for competition_key, team_slug in teams:
        payload = {
            "players": json_rows(
                conn,
                """
                SELECT payload_json
                FROM site_player_identity_map
                WHERE competition_key = ? AND club_slug = ?
                ORDER BY rating_power DESC, rank_club ASC, name
                LIMIT ?
                """,
                (competition_key, team_slug, limit),
            ),
            "recent_team_stats": json_rows(
                conn,
                """
                SELECT payload_json
                FROM site_team_match_stats
                WHERE team_slug = ?
                ORDER BY fixture_key DESC
                LIMIT ?
                """,
                (team_slug, limit),
            ),
            "recent_lineup_slots": json_rows(
                conn,
                """
                SELECT payload_json
                FROM site_lineup_slots
                WHERE team_slug = ?
                ORDER BY fixture_key DESC, is_starting_xi DESC, broad_position, slot_code, player_name
                LIMIT ?
                """,
                (team_slug, limit * 2),
            ),
            "player_event_shortlists": json_rows(
                conn,
                """
                SELECT payload_json
                FROM site_player_event_shortlists
                WHERE team_slug = ?
                ORDER BY fixture_key DESC, event_family, shortlist_rank
                LIMIT ?
                """,
                (team_slug, limit),
            ),
        }
        rows.append((competition_key, team_slug, json_text(payload)))
    conn.executemany(
        """
        INSERT OR REPLACE INTO site_team_premium_payloads(
          competition_key, team_slug, payload_json
        ) VALUES (?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def insert_site_external_sources(conn: sqlite3.Connection, data_root: Path) -> int:
    registry = read_json(data_root / "external_content" / "source_registry.json", {})
    rows = []
    for source in registry.get("sources", []) if isinstance(registry, dict) else []:
        source_id = str(source.get("source_id") or "").strip()
        if not source_id:
            continue
        rows.append(
            (
                source_id,
                source.get("provider") or "",
                source.get("usage_mode") or "",
                source.get("terms_url") or "",
                json_text(source),
            )
        )
    conn.executemany(
        """
        INSERT OR REPLACE INTO site_external_sources(
          source_id, provider, usage_mode, terms_url, payload_json
        ) VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def insert_site_fixture_external_content(conn: sqlite3.Connection, data_root: Path) -> int:
    root = data_root / "external_content" / "fixture_media"
    rows = []
    if not root.exists():
        return 0
    for path in sorted(root.glob("*.json")):
        if path.name == "index.json":
            continue
        payload = read_json(path, {})
        fixture_key = str(payload.get("fixture_key") or path.stem).strip()
        if not fixture_key:
            continue
        for collection_name in ("media", "news_signals", "weather_signals", "space_weather_signals", "sentiment_signals"):
            items = payload.get(collection_name)
            if not isinstance(items, list):
                continue
            for index, item in enumerate(items, start=1):
                if not isinstance(item, dict):
                    continue
                content_id = str(item.get("content_id") or f"{collection_name}_{index}").strip()
                row_id = f"{fixture_key}:{collection_name}:{content_id}"
                rows.append(
                    (
                        row_id,
                        fixture_key,
                        item.get("type") or collection_name,
                        item.get("source_id") or "",
                        item.get("provider") or "",
                        safe_int(item.get("priority")) or index,
                        json_text(item),
                    )
                )
    conn.executemany(
        """
        INSERT OR REPLACE INTO site_fixture_external_content(
          row_id, fixture_key, content_type, source_id, provider, priority, payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def insert_site_fixture_context_payloads(conn: sqlite3.Connection) -> int:
    fixture_keys = [
        row[0]
        for row in conn.execute(
            """
            SELECT DISTINCT fixture_key
            FROM site_fixture_external_content
            ORDER BY fixture_key
            """
        ).fetchall()
    ]
    rows = []
    for fixture_key in fixture_keys:
        payload = {
            "media": json_rows(
                conn,
                """
                SELECT payload_json
                FROM site_fixture_external_content
                WHERE fixture_key = ? AND content_type = 'youtube_embed'
                ORDER BY priority, row_id
                """,
                (fixture_key,),
            ),
            "news_signals": json_rows(
                conn,
                """
                SELECT payload_json
                FROM site_fixture_external_content
                WHERE fixture_key = ? AND content_type IN ('rss_headline_link', 'news_signal')
                ORDER BY priority, row_id
                """,
                (fixture_key,),
            ),
            "weather_signals": json_rows(
                conn,
                """
                SELECT payload_json
                FROM site_fixture_external_content
                WHERE fixture_key = ? AND content_type IN ('weather_context', 'weather_signal')
                ORDER BY priority, row_id
                """,
                (fixture_key,),
            ),
            "sentiment_signals": json_rows(
                conn,
                """
                SELECT payload_json
                FROM site_fixture_external_content
                WHERE fixture_key = ? AND content_type = 'sentiment_signal'
                ORDER BY priority, row_id
                """,
                (fixture_key,),
            ),
            "space_weather_signals": json_rows(
                conn,
                """
                SELECT payload_json
                FROM site_fixture_external_content
                WHERE fixture_key = ? AND content_type = 'environmental_volatility'
                ORDER BY priority, row_id
                """,
                (fixture_key,),
            ),
        }
        rows.append((fixture_key, json_text(payload)))
    conn.executemany(
        "INSERT OR REPLACE INTO site_fixture_context_payloads(fixture_key, payload_json) VALUES (?, ?)",
        rows,
    )
    return len(rows)


def export_database(
    data_root: Path,
    output_path: Path,
    include_history: bool = False,
    normalized_root: Path = DEFAULT_NORMALIZED_ROOT,
) -> dict[str, int | str]:
    started = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    active_seasons = None if include_history else active_competition_seasons(data_root)
    site_fixture_aliases = active_site_fixture_aliases(data_root)

    conn = sqlite3.connect(output_path)
    try:
        execute_schema(conn)
        identity_count, identities = insert_site_player_identity_map(
            conn,
            data_root,
            normalized_root,
            active_seasons,
            site_fixture_aliases,
        )
        team_lineup_snapshot_count = insert_team_lineup_snapshots(conn, data_root)
        if team_lineup_snapshot_count == 0:
            team_lineup_snapshot_count = insert_team_lineup_snapshots_from_normalized(
                conn,
                normalized_root,
                identities,
                active_seasons,
                site_fixture_aliases,
            )
        insert_counts = {
            "fixtures": insert_fixtures(conn, data_root),
            "fixture_decisions": insert_fixture_decisions(conn, data_root),
            "fixture_lineups": insert_fixture_lineups(conn, data_root),
            "fixture_h2h": insert_fixture_h2h(conn, data_root),
            "team_intelligence": insert_team_intelligence(conn, data_root, active_seasons),
            "club_squads": insert_club_squads(conn, data_root, active_seasons),
            "team_lineup_snapshots": team_lineup_snapshot_count,
            "site_player_identity_map": identity_count,
            "site_player_match_stats": insert_site_player_match_stats(conn, normalized_root, identities, active_seasons, site_fixture_aliases),
            "site_team_match_stats": insert_site_team_match_stats(conn, normalized_root, active_seasons, site_fixture_aliases),
            "site_match_events": insert_site_match_events(conn, normalized_root, identities, active_seasons, site_fixture_aliases),
            "site_formation_slots": insert_site_formation_slots(conn, normalized_root, active_seasons, site_fixture_aliases),
            "site_lineup_slots": insert_site_lineup_slots(conn, normalized_root, identities, active_seasons, site_fixture_aliases),
            "site_fixture_market_intelligence": insert_site_fixture_market_intelligence(conn),
            "site_player_event_shortlists": insert_site_player_event_shortlists(conn),
            "site_external_sources": insert_site_external_sources(conn, data_root),
            "site_fixture_external_content": insert_site_fixture_external_content(conn, data_root),
            "site_fixture_context_payloads": insert_site_fixture_context_payloads(conn),
            "site_fixture_stats_payloads": insert_site_fixture_stats_payloads(conn),
            "site_team_premium_payloads": insert_site_team_premium_payloads(conn),
        }
        counts = {key: table_count(conn, key) for key in insert_counts}
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
    parser.add_argument("--normalized-root", default=str(DEFAULT_NORMALIZED_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--include-history",
        action="store_true",
        help="Export all historical team/player club-season rows instead of the active-site latest-season footprint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_database(
        Path(args.data_root),
        Path(args.output),
        include_history=args.include_history,
        normalized_root=Path(args.normalized_root),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
