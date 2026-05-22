#!/usr/bin/env python3
"""Compile compact per-fixture website intelligence payloads.

This is a website/data-publishing compiler only. It reads the local compact
site SQLite plus local injury-shock reports, then emits one consistent
per-fixture contract for the static frontend/Worker publish layer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sqlite3
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "build" / "site_data" / "odds_genius.sqlite"
DEFAULT_OUTPUT_DIR = ROOT / "build" / "site_brain" / "current"
DEFAULT_REPORT_JSON = ROOT / "reports" / "latest" / "FIXTURE_BRAIN_COMPILER_REPORT.json"
DEFAULT_REPORT_MD = ROOT / "reports" / "latest" / "FIXTURE_BRAIN_COMPILER_REPORT.md"
DEFAULT_INJURY_FIXTURE_CSV = ROOT / "reports" / "latest" / "injury_shock_coverage_scan" / "INJURY_SHOCK_COVERAGE_SCAN.csv"
DEFAULT_INJURY_PLAYER_CSVS = (
    ROOT / "reports" / "latest" / "injury_shock_coverage_scan" / "INJURY_SHOCK_PLAYER_IMPACT.csv",
    ROOT / "reports" / "latest" / "injury_shock_coverage_scan" / "SUNDAY_2026_05_17_INJURY_LINEUP_IMPACT.csv",
)
DEFAULT_INJURY_MARKET_IMPACT_CSV = ROOT / "reports" / "latest" / "injury_shock_market_impact" / "INJURY_SHOCK_MARKET_IMPACT_FIXTURE.csv"
DEFAULT_INJURY_ADMIN_JSON = ROOT / "frontend" / "public" / "data" / "internal" / "injury_shock_admin_dashboard.json"
CORE_MARKETS = ("ftr", "ou25", "btts", "team_goals")
SUMMARY_TIERS = ("standard", "premium", "pro", "pro_plus")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile compact fixture-brain payloads for website publishing.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--from-date", default=date.today().isoformat())
    parser.add_argument("--to-date", default="")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--all-fixtures", action="store_true")
    parser.add_argument("--fixture-key", action="append", default=[], help="Compile one fixture key. May be repeated.")
    parser.add_argument("--injury-fixture-csv", type=Path, default=DEFAULT_INJURY_FIXTURE_CSV)
    parser.add_argument("--injury-player-csv", type=Path, action="append", default=[])
    parser.add_argument("--injury-market-impact-csv", type=Path, default=DEFAULT_INJURY_MARKET_IMPACT_CSV)
    parser.add_argument("--injury-admin-json", type=Path, default=DEFAULT_INJURY_ADMIN_JSON)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--preserve-output", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def pretty_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def parse_json(value: Any, default: Any = None) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return default


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> int:
    text = pretty_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


def parse_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text[:10]).date()
    except ValueError:
        return None


def normalize_key(value: Any) -> str:
    text = str(value or "").lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def split_tokens(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [token.strip() for token in re.split(r"[|,;]", text) if token.strip()]


def to_number(value: Any) -> float | int | None:
    text = str(value or "").strip()
    if text == "":
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if number.is_integer():
        return int(number)
    return round(number, 4)


def compact_row(row: sqlite3.Row | None) -> dict[str, Any]:
    if row is None:
        return {}
    return {key: row[key] for key in row.keys() if row[key] not in (None, "") and key != "payload_json"}


def parse_row_payload(row: sqlite3.Row | None) -> dict[str, Any]:
    payload = parse_json(row["payload_json"], {}) if row else {}
    return payload if isinstance(payload, dict) else {}


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)).fetchone()
    return row is not None


def one_by_fixture(conn: sqlite3.Connection, table: str, fixture_key: str) -> sqlite3.Row | None:
    if not table_exists(conn, table):
        return None
    return conn.execute(f"SELECT * FROM {table} WHERE fixture_key = ? LIMIT 1", (fixture_key,)).fetchone()


def rows_by_fixture(conn: sqlite3.Connection, table: str, fixture_key: str, order_sql: str = "") -> list[sqlite3.Row]:
    if not table_exists(conn, table):
        return []
    suffix = f" ORDER BY {order_sql}" if order_sql else ""
    return list(conn.execute(f"SELECT * FROM {table} WHERE fixture_key = ?{suffix}", (fixture_key,)))


def fixture_rows(conn: sqlite3.Connection, start: date, end: date, include_all: bool, fixture_keys: set[str]) -> list[sqlite3.Row]:
    rows = list(conn.execute("SELECT * FROM fixtures ORDER BY kickoff_time, fixture_key"))
    if fixture_keys:
        return [row for row in rows if row["fixture_key"] in fixture_keys]
    if include_all:
        return rows
    selected: list[sqlite3.Row] = []
    for row in rows:
        kickoff_date = parse_date(row["kickoff_time"])
        if kickoff_date and start <= kickoff_date <= end:
            selected.append(row)
    return selected


def injury_match_keys(row: dict[str, Any]) -> set[str]:
    keys = {
        str(row.get("fixture_key") or "").strip(),
        str(row.get("fixture_join_key") or "").strip(),
    }
    match_date = str(row.get("match_date") or "").strip()
    home = row.get("home_team_name") or row.get("home_team") or ""
    away = row.get("away_team_name") or row.get("away_team") or ""
    if match_date and home and away:
        keys.add(f"{match_date}_{home}_{away}")
    return {normalize_key(key) for key in keys if key}


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_injury_fixture_index(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rows = read_csv_rows(path)
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        for key in injury_match_keys(row):
            index.setdefault(key, row)
    return index, {"path": str(path), "exists": path.exists(), "rows": len(rows), "indexed_keys": len(index)}


def player_impact_score(row: dict[str, Any]) -> float:
    candidates = [
        row.get("importance_weight_used"),
        row.get("importance_weight"),
        row.get("attack_score"),
        row.get("midfield_score"),
        row.get("defence_score"),
        row.get("keeper_score"),
        row.get("mobility_score"),
    ]
    scores = [float(value) for value in (to_number(item) for item in candidates) if isinstance(value, (int, float))]
    return max(scores) if scores else 0.0


def load_injury_player_index(paths: list[Path]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sources: list[dict[str, Any]] = []
    seen_rows: set[tuple[str, str, str, str]] = set()
    for path in paths:
        rows = read_csv_rows(path)
        sources.append({"path": str(path), "exists": path.exists(), "rows": len(rows)})
        for row in rows:
            identity = (
                str(row.get("fixture_id") or ""),
                str(row.get("team_id") or ""),
                str(row.get("player_id") or ""),
                str(row.get("published_ts_utc") or ""),
            )
            if identity in seen_rows:
                continue
            seen_rows.add(identity)
            keys = injury_match_keys(row)
            for key in keys:
                index[key].append(row)
    for key, rows in index.items():
        rows.sort(key=player_impact_score, reverse=True)
        index[key] = rows[:12]
    return dict(index), sources


def load_injury_market_impact_index(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rows = read_csv_rows(path)
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        for key in injury_match_keys(row):
            index.setdefault(key, row)
    return index, {"path": str(path), "exists": path.exists(), "rows": len(rows), "indexed_keys": len(index)}


def fixture_injury_keys(fixture: dict[str, Any], row: sqlite3.Row) -> set[str]:
    fixture_key = fixture.get("fixture_key") or row["fixture_key"]
    kickoff = fixture.get("kickoff_time") or row["kickoff_time"]
    match_date = str(kickoff or "")[:10]
    home = fixture.get("home_team") or row["home_team"]
    away = fixture.get("away_team") or row["away_team"]
    return {
        normalize_key(fixture_key),
        normalize_key(f"{match_date}_{home}_{away}"),
        normalize_key(f"{match_date.replace('-', '_')}_{home}_{away}"),
    }


def pick_injury_fixture(keys: set[str], index: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    for key in keys:
        row = index.get(key)
        if row:
            return row
    return None


def pick_injury_players(keys: set[str], index: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    best: list[dict[str, Any]] = []
    for key in keys:
        rows = index.get(key)
        if rows and len(rows) > len(best):
            best = rows
    return best


def compact_injury_fixture(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {"status": "missing", "summary": "No local injury-shock fixture row matched this site fixture."}
    side_scores = {
        "home": {
            "attack_absence": to_number(row.get("home_attack_absence_score")),
            "midfield_absence": to_number(row.get("home_midfield_absence_score")),
            "defence_absence": to_number(row.get("home_defence_absence_score")),
            "keeper_absence": to_number(row.get("home_keeper_absence_score")),
            "mobility_risk": to_number(row.get("home_mobility_risk_score")),
            "lineup_confidence": to_number(row.get("home_lineup_confidence_score")),
            "news_severity": to_number(row.get("home_injury_news_severity")),
            "player_impact_rows": to_number(row.get("home_player_impact_rows")),
            "absence_reasons": split_tokens(row.get("home_absence_reasons")),
        },
        "away": {
            "attack_absence": to_number(row.get("away_attack_absence_score")),
            "midfield_absence": to_number(row.get("away_midfield_absence_score")),
            "defence_absence": to_number(row.get("away_defence_absence_score")),
            "keeper_absence": to_number(row.get("away_keeper_absence_score")),
            "mobility_risk": to_number(row.get("away_mobility_risk_score")),
            "lineup_confidence": to_number(row.get("away_lineup_confidence_score")),
            "news_severity": to_number(row.get("away_injury_news_severity")),
            "player_impact_rows": to_number(row.get("away_player_impact_rows")),
            "absence_reasons": split_tokens(row.get("away_absence_reasons")),
        },
    }
    warning_flag = str(row.get("deploy_warning_flag") or "").strip() in {"1", "true", "TRUE", "yes", "YES"}
    return {
        "status": "matched",
        "research_only": True,
        "fixture_id": row.get("fixture_id"),
        "fixture_key": row.get("fixture_key"),
        "fixture_join_key": row.get("fixture_join_key"),
        "match_date": row.get("match_date"),
        "warning_flag": warning_flag,
        "warning_tokens": split_tokens(row.get("warning_tokens")),
        "absence_edge_side": row.get("absence_edge_side") or "",
        "side_scores": side_scores,
        "model_adjustments": {
            "goal_model": to_number(row.get("goal_model_adjustment")),
            "btts": to_number(row.get("btts_adjustment")),
            "ou25": to_number(row.get("ou25_adjustment")),
            "ftr_volatility": to_number(row.get("ftr_volatility_adjustment")),
            "motivation_volatility": to_number(row.get("motivation_volatility_score")),
        },
        "context_flags": split_tokens(row.get("context_flags")),
        "source_refs": {
            "league_tag": row.get("league_tag") or "",
            "injury_source_csv": row.get("injury_source_csv") or "",
            "sidelined_source_csv": row.get("sidelined_source_csv") or "",
            "player_stats_source_csv": row.get("player_stats_source_csv") or "",
        },
    }


def compact_injury_market_impact(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {
            "status": "missing",
            "summary": "No local injury-shock market-impact row matched this site fixture.",
        }
    return {
        "status": "matched",
        "research_only": True,
        "fixture_key": row.get("fixture_key") or "",
        "match_date": row.get("match_date") or "",
        "deploy_markets": split_tokens(row.get("deploy_markets")),
        "deploy_picks": split_tokens(row.get("deploy_picks")),
        "overall_adjustment": row.get("fixture_injury_market_adjustment") or "",
        "market_impacts": {
            "ftr": row.get("ftr_injury_impact") or "",
            "btts": row.get("btts_injury_impact") or "",
            "ou25": row.get("ou25_injury_impact") or "",
        },
        "key_absences": to_number(row.get("key_absences")),
        "high_or_medium_absences": to_number(row.get("high_or_medium")),
        "key_players": split_tokens(row.get("key_players")),
        "shock_roles": split_tokens(row.get("shock_roles")),
        "market_impact_notes": split_tokens(row.get("market_impact")),
        "lineup_status": row.get("lineup_status") or "",
        "injury_scan_status": row.get("injury_scan_status") or "",
    }


def compact_player_impacts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows[:10]:
        out.append(
            {
                "team_side": row.get("team_side") or "",
                "team_name": row.get("team_name") or "",
                "player_id": row.get("player_id") or "",
                "player_name": row.get("player_name") or "",
                "absence_type": row.get("absence_type") or "",
                "reason": row.get("reason") or "",
                "absence_state": row.get("absence_state") or "",
                "availability_impact_status": row.get("availability_impact_status") or "",
                "structural_function": row.get("structural_function") or "",
                "role_impact": row.get("role_impact") or "",
                "market_impact_note": row.get("market_impact_note") or "",
                "injury_shock_severity": row.get("injury_shock_severity") or "",
                "position": row.get("position") or "",
                "player_rating_summary": row.get("player_rating_summary") or "",
                "importance_weight": to_number(row.get("importance_weight_used") or row.get("importance_weight")),
                "og_player_power": to_number(row.get("og_player_power")),
                "goal_threat": to_number(row.get("goal_threat")),
                "creative_spark": to_number(row.get("creative_spark")),
                "defensive_lock": to_number(row.get("defensive_lock")),
                "club_rank": to_number(row.get("club_rank")),
                "key_player": str(row.get("key_player_flag") or "").strip() in {"1", "true", "TRUE", "yes", "YES"},
                "known_hours_before_kickoff": to_number(row.get("known_hours_before_kickoff")),
                "expected_xi_absence": str(row.get("expected_xi_absence_flag") or "").strip() in {"1", "true", "TRUE"},
                "confirmed_lineup_absence": str(row.get("current_confirmed_lineup_absence_flag") or "").strip() in {"1", "true", "TRUE"},
                "impact_scores": {
                    "attack": to_number(row.get("attack_score")),
                    "midfield": to_number(row.get("midfield_score")),
                    "defence": to_number(row.get("defence_score")),
                    "keeper": to_number(row.get("keeper_score")),
                    "mobility": to_number(row.get("mobility_score")),
                },
                "join_status": {
                    "rating": row.get("player_rating_join_status") or "",
                    "importance": row.get("player_importance_join_status") or "",
                },
            }
        )
    return out


def load_team_indexes(conn: sqlite3.Connection) -> tuple[dict[tuple[str, str], sqlite3.Row], dict[tuple[str, str], sqlite3.Row]]:
    team_rows: dict[tuple[str, str], sqlite3.Row] = {}
    premium_rows: dict[tuple[str, str], sqlite3.Row] = {}
    if table_exists(conn, "team_intelligence"):
        for row in conn.execute("SELECT * FROM team_intelligence"):
            competition_key = normalize_key(row["competition_key"])
            for team_key in (row["team_slug"], row["team"]):
                slug = normalize_key(team_key)
                if competition_key and slug:
                    team_rows[(competition_key, slug)] = row
    if table_exists(conn, "site_team_premium_payloads"):
        for row in conn.execute("SELECT * FROM site_team_premium_payloads"):
            key = (normalize_key(row["competition_key"]), normalize_key(row["team_slug"]))
            premium_rows[key] = row
    return team_rows, premium_rows


def compact_team_intelligence_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload.get(key)
        for key in (
            "team",
            "team_slug",
            "competition",
            "competition_key",
            "season",
            "summary",
            "matches_played",
            "sample_confidence",
            "profile_tags",
            "ratings",
            "rating_bands",
            "market_tendencies",
            "timing_profile",
        )
        if payload.get(key) not in (None, "", [], {})
    }


def compact_premium_team_payload(payload: dict[str, Any]) -> dict[str, Any]:
    players = payload.get("players") if isinstance(payload.get("players"), list) else []
    shortlists = payload.get("player_event_shortlists") if isinstance(payload.get("player_event_shortlists"), list) else []
    recent_lineup_slots = payload.get("recent_lineup_slots") if isinstance(payload.get("recent_lineup_slots"), list) else []
    recent_team_stats = payload.get("recent_team_stats") if isinstance(payload.get("recent_team_stats"), list) else []
    top_players = sorted(
        (item for item in players if isinstance(item, dict)),
        key=lambda item: float(item.get("rating_power") or item.get("rank_overall") or 0),
        reverse=True,
    )[:8]
    return {
        "counts": {
            "players": len(players),
            "player_event_shortlists": len(shortlists),
            "recent_lineup_slots": len(recent_lineup_slots),
            "recent_team_stats": len(recent_team_stats),
        },
        "top_players": [
            {
                "player_name": item.get("player_name") or item.get("name") or "",
                "position_group": item.get("position_group") or item.get("position") or "",
                "rating_power": item.get("rating_power"),
                "rank_overall": item.get("rank_overall"),
                "rank_club": item.get("rank_club"),
            }
            for item in top_players
        ],
        "top_player_events": [
            {
                "event_key": item.get("event_key") or "",
                "event_label": item.get("event_label") or "",
                "player_name": item.get("player_name") or "",
                "team_name": item.get("team_name") or payload.get("team") or "",
                "shortlist_rank": item.get("shortlist_rank"),
                "shortlist_score": item.get("shortlist_score"),
            }
            for item in shortlists[:10]
            if isinstance(item, dict)
        ],
    }


def team_payload_for(
    index: dict[tuple[str, str], sqlite3.Row],
    competition_key: str,
    team_name: str,
    premium: bool = False,
) -> dict[str, Any]:
    row = index.get((normalize_key(competition_key), normalize_key(team_name)))
    payload = parse_row_payload(row)
    if not row:
        return {"status": "missing", "team": team_name}
    return {
        "status": "available",
        "meta": compact_row(row),
        "payload": compact_premium_team_payload(payload) if premium else compact_team_intelligence_payload(payload),
    }


def compact_market_cards(decision: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
    market_intel = decision.get("market_intelligence") if isinstance(decision.get("market_intelligence"), dict) else {}
    stats_markets = stats.get("market_intelligence") if isinstance(stats.get("market_intelligence"), list) else []
    by_key = {str(item.get("market_key")): item for item in stats_markets if isinstance(item, dict)}
    cards: dict[str, Any] = {}
    for market in CORE_MARKETS:
        block = market_intel.get(market) if isinstance(market_intel, dict) else None
        stats_block = by_key.get(market, {})
        if isinstance(block, dict):
            model_output = block.get("model_output") if isinstance(block.get("model_output"), dict) else None
            cards[market] = {
                "status": "available",
                "state": block.get("state") or stats_block.get("state") or "",
                "model_lean": block.get("model_lean") or stats_block.get("model_lean") or "",
                "team_context_lean": block.get("team_context_lean") or "",
                "model_output": model_output,
                "support": block.get("alignment_score") or stats_block.get("alignment_score"),
                "rating": block.get("rating") or stats_block.get("rating"),
                "band": block.get("band") or stats_block.get("band") or "",
                "public_summary": block.get("public_summary") or stats_block.get("public_summary") or "",
                "source_status": "all_markets_model_output" if model_output else "context_support_only",
            }
        elif stats_block:
            cards[market] = {
                "status": "available",
                "state": stats_block.get("state") or "",
                "model_lean": stats_block.get("model_lean") or "",
                "support": stats_block.get("alignment_score"),
                "rating": stats_block.get("rating"),
                "band": stats_block.get("band") or "",
                "public_summary": stats_block.get("public_summary") or "",
                "source_status": stats_block.get("source_status") or "stats_payload",
            }
        else:
            cards[market] = {"status": "missing", "source_status": "not_in_decision_payload"}
    return cards


def compact_shortlists(rows: list[sqlite3.Row], limit: int = 24) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows[:limit]:
        out.append(
            {
                "event_key": row["event_key"],
                "event_family": row["event_family"],
                "event_label": row["event_label"],
                "player_name": row["player_name"],
                "team_name": row["team_name"],
                "is_home": bool(row["is_home"]),
                "position_group": row["position_group"],
                "source_lineup_status": row["source_lineup_status"],
                "rank": row["shortlist_rank"],
                "score": row["shortlist_score"],
                "recent_per90": row["recent_per90"],
                "recent_average": row["recent_average"],
                "sample_size": row["sample_size"],
                "minutes_sample": row["minutes_sample"],
                "confidence_label": row["confidence_label"],
                "beta_status": row["beta_status"],
                "reason": row["reason"],
            }
        )
    return out


def player_event_phase(shortlists: list[dict[str, Any]], fixture: dict[str, Any], lineup: dict[str, Any]) -> dict[str, Any]:
    statuses = sorted({str(item.get("source_lineup_status") or "").strip() for item in shortlists if item.get("source_lineup_status")})
    lineup_text = canonical_json(lineup).lower() if lineup else ""
    fixture_phase = str(fixture.get("snapshot_phase") or "").strip()
    if any(status in {"confirmed_lineups", "confirmed_lineup", "lineup_confirmed"} for status in statuses) or "confirmed" in lineup_text:
        phase = "lineup_confirmed_refresh"
        lineup_status = "confirmed"
    elif statuses:
        phase = "pre_lineup_preview"
        lineup_status = statuses[0] if len(statuses) == 1 else "mixed_pre_lineup_sources"
    elif fixture_phase:
        phase = fixture_phase
        lineup_status = "unknown"
    else:
        phase = "lineup_pending"
        lineup_status = "missing"
    return {
        "phase": phase,
        "lineup_status": lineup_status,
        "source_lineup_statuses": statuses,
    }


def player_event_families() -> dict[str, str]:
    return {
        "shots": "Player Shots",
        "shots_on_target": "Shots On Target",
        "key_passes": "Key Passes",
        "tackles": "Player Tackles",
        "fouls": "Player Fouls",
        "player_fouled": "Player Fouled",
        "bookings": "Bookings Watch",
        "keeper_saves": "Keeper Saves",
        "team_tackles": "Team / Match Tackles",
    }


def normalize_player_event_family(value: Any) -> str:
    raw = str(value or "").strip()
    aliases = {
        "cards": "bookings",
        "yellow_cards": "bookings",
        "saves": "keeper_saves",
        "goalkeeper_saves": "keeper_saves",
        "passes_key": "key_passes",
    }
    if raw in aliases:
        return aliases[raw]
    key = raw.lower()
    if "card" in key or "booking" in key:
        return "bookings"
    if "save" in key:
        return "keeper_saves"
    if "key_pass" in key or "passes_key" in key:
        return "key_passes"
    return raw


def fallback_player_event_cards_from_team_context(
    team_context: dict[str, Any] | None,
    phase: dict[str, Any],
    limit_per_card: int,
) -> list[dict[str, Any]]:
    if not isinstance(team_context, dict):
        return []
    families = player_event_families()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for side in ("premium_home", "premium_away"):
        block = team_context.get(side, {}) if isinstance(team_context.get(side), dict) else {}
        payload = block.get("payload", {}) if isinstance(block.get("payload"), dict) else {}
        team_name = (payload.get("top_players") or [{}])[0].get("team_name") if isinstance(payload.get("top_players"), list) and payload.get("top_players") else ""
        for item in payload.get("top_player_events") or []:
            if not isinstance(item, dict):
                continue
            family = normalize_player_event_family(item.get("event_key") or item.get("event_label"))
            if family not in families:
                continue
            grouped[family].append(
                {
                    "player_name": item.get("player_name") or "Player pending",
                    "team_name": item.get("team_name") or team_name or ("Home" if side == "premium_home" else "Away"),
                    "is_home": side == "premium_home",
                    "position_group": "",
                    "rank": item.get("shortlist_rank") or "",
                    "score": item.get("shortlist_score") or "",
                    "confidence_label": "squad_profile_fallback",
                    "sample_size": "",
                    "minutes_sample": "",
                    "reason": "Squad-profile fallback until the fixture-specific player-event compiler publishes a shortlist.",
                }
            )
        for item in payload.get("top_players") or []:
            if not isinstance(item, dict):
                continue
            position = str(item.get("position_group") or item.get("position") or "").lower()
            derived_families = ["shots"] if any(token in position for token in ("forward", "wing", "attacking")) else []
            if "midfielder" in position or "midfield" in position:
                derived_families.extend(["key_passes", "tackles"])
            if "defender" in position or "back" in position:
                derived_families.append("tackles")
            if "keeper" in position or "goal" in position:
                derived_families.append("keeper_saves")
            for family in derived_families[:2]:
                grouped[family].append(
                    {
                        "player_name": item.get("player_name") or "Player pending",
                        "team_name": team_name or ("Home" if side == "premium_home" else "Away"),
                        "is_home": side == "premium_home",
                        "position_group": item.get("position_group") or "",
                        "rank": item.get("rank_club") or item.get("rank_overall") or "",
                        "score": item.get("rating_power") or "",
                        "confidence_label": "squad_profile_fallback",
                        "sample_size": "",
                        "minutes_sample": "",
                        "reason": "Fallback from team squad profile; refresh after lineups and event shortlists are available.",
                    }
                )
    cards: list[dict[str, Any]] = []
    for family, label in families.items():
        rows = sorted(grouped.get(family, []), key=lambda item: float(item.get("score") or 0), reverse=True)
        if not rows:
            continue
        cards.append(
            {
                "event_family": family,
                "card_title": label,
                "status": "fallback",
                "beta_status": "squad_profile_fallback",
                "market_availability": "watchlist_pending",
                "lineup_status": phase["lineup_status"] or "last_fixture_snapshot",
                "shortlist": rows[:limit_per_card],
            }
        )
    return cards


def compact_player_event_cards(
    shortlist_rows: list[sqlite3.Row],
    fixture: dict[str, Any],
    lineup: dict[str, Any],
    team_context: dict[str, Any] | None = None,
    limit_per_card: int = 4,
) -> dict[str, Any]:
    shortlists = compact_shortlists(shortlist_rows, limit=48)
    phase = player_event_phase(shortlists, fixture, lineup)
    families = player_event_families()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in shortlists:
        family = normalize_player_event_family(item.get("event_family") or item.get("event_key"))
        if family not in families:
            key = str(item.get("event_key") or "")
            if "card" in key or "booking" in key:
                family = "bookings"
            elif "save" in key:
                family = "keeper_saves"
            elif "key_pass" in key or "passes_key" in key:
                family = "key_passes"
        if family in families:
            grouped[family].append(item)

    cards: list[dict[str, Any]] = []
    for family, label in families.items():
        sorted_items = sorted(
            grouped.get(family, []),
            key=lambda item: (float(item.get("score") or 0), -int(item.get("rank") or 999)),
            reverse=True,
        )
        items: list[dict[str, Any]] = []
        seen_players: set[tuple[str, str]] = set()
        for item in sorted_items:
            player_key = (
                normalize_key(item.get("team_name")),
                normalize_key(item.get("player_name")),
            )
            if player_key in seen_players:
                continue
            seen_players.add(player_key)
            items.append(item)
        if not items:
            continue
        cards.append(
            {
                "event_family": family,
                "card_title": label,
                "status": "available",
                "beta_status": "beta_shortlist",
                "market_availability": "watchlist_only",
                "lineup_status": phase["lineup_status"],
                "shortlist": [
                    {
                        "player_name": item.get("player_name"),
                        "team_name": item.get("team_name"),
                        "is_home": item.get("is_home"),
                        "position_group": item.get("position_group"),
                        "rank": item.get("rank"),
                        "score": item.get("score"),
                        "confidence_label": item.get("confidence_label"),
                        "sample_size": item.get("sample_size"),
                        "minutes_sample": item.get("minutes_sample"),
                        "reason": item.get("reason"),
                    }
                    for item in items[:limit_per_card]
                ],
            }
        )
    if not cards:
        cards = fallback_player_event_cards_from_team_context(team_context, phase, limit_per_card)
    missing = [family for family in families if family not in grouped]
    return {
        "status": "available" if cards else "missing",
        "research_only": True,
        "phase": phase["phase"],
        "lineup_status": phase["lineup_status"],
        "source_lineup_statuses": phase["source_lineup_statuses"],
        "source_scope": "last_starting_xi_and_bench_until_confirmed_lineups_refresh",
        "refresh_policy": "Rebuild after lineup automation at roughly T-60, then republish compact fixture payload.",
        "cards": cards,
        "missing_event_families": missing,
    }


def compact_table_payloads(rows: list[sqlite3.Row], limit: int = 40) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows[:limit]:
        item = compact_row(row)
        out.append(item)
    return out


def compact_fixture_stats_payload(stats: dict[str, Any]) -> dict[str, Any]:
    team_stats = stats.get("team_stats") if isinstance(stats.get("team_stats"), list) else []
    player_stats = stats.get("player_stats") if isinstance(stats.get("player_stats"), list) else []
    match_events = stats.get("match_events") if isinstance(stats.get("match_events"), list) else []
    lineup_slots = stats.get("lineup_slots") if isinstance(stats.get("lineup_slots"), list) else []
    market_intelligence = stats.get("market_intelligence") if isinstance(stats.get("market_intelligence"), list) else []
    player_event_shortlists = stats.get("player_event_shortlists") if isinstance(stats.get("player_event_shortlists"), list) else []
    goal_markets = [
        item
        for item in market_intelligence
        if isinstance(item, dict) and str(item.get("market_key") or "").lower() in {"ftr", "ou25", "btts", "team_goals"}
    ]
    attack_summary = "Post-match team stats are pending; using market and model-context rows until the API-football stats refresh lands."
    if team_stats:
        team_bits = []
        for item in team_stats[:2]:
            if not isinstance(item, dict):
                continue
            shots = item.get("shots_total") if item.get("shots_total") not in (None, "") else "shots pending"
            sot = item.get("shots_on_goal") if item.get("shots_on_goal") not in (None, "") else "SOT pending"
            team_bits.append(f"{item.get('team_name') or 'Team'}: {shots} shots, {sot} SOT")
        if team_bits:
            attack_summary = " | ".join(team_bits)
    pressure_summary = "No event pressure profile yet."
    if team_stats:
        fouls = [to_number(item.get("fouls_for")) for item in team_stats if isinstance(item, dict)]
        corners = [to_number(item.get("corners_for")) for item in team_stats if isinstance(item, dict)]
        pressure_summary = f"Fouls tracked: {sum(value or 0 for value in fouls)}; corners tracked: {sum(value or 0 for value in corners)}."
    elif goal_markets:
        pressure_summary = f"{len(goal_markets)} market-context rows available for pre-match support/contradiction."
    goal_support = [
        {
            "market_key": item.get("market_key"),
            "selection_label": item.get("selection_label"),
            "state": item.get("state"),
            "alignment_score": item.get("alignment_score"),
            "band": item.get("band"),
            "public_summary": item.get("public_summary"),
        }
        for item in goal_markets
    ]
    return {
        "counts": {
            "team_stats": len(team_stats),
            "player_stats": len(player_stats),
            "match_events": len(match_events),
            "lineup_slots": len(lineup_slots),
            "market_intelligence": len(market_intelligence),
            "player_event_shortlists": len(player_event_shortlists),
        },
        "readable_cards": [
            {
                "card_key": "team_attack_defence",
                "title": "Team Attack / Defence",
                "status": "available" if team_stats else "context_only",
                "summary": attack_summary,
            },
            {
                "card_key": "match_tempo",
                "title": "Match Tempo",
                "status": "available" if match_events else "context_only",
                "summary": f"{len(match_events)} match events captured." if match_events else "Pre-match tempo read is taken from model and market context until live event data arrives.",
            },
            {
                "card_key": "pressure_profile",
                "title": "Pressure Profile",
                "status": "available" if team_stats else "context_only",
                "summary": pressure_summary,
            },
            {
                "card_key": "goal_market_support",
                "title": "Goal-Market Support / Contradiction",
                "status": "available" if goal_support else "pending",
                "summary": f"{len(goal_support)} goal-market support rows available." if goal_support else "Goal-market support rows pending.",
                "markets": goal_support,
            },
        ],
        "market_intelligence": [
            {
                "market_key": item.get("market_key"),
                "market_family": item.get("market_family"),
                "market_label": item.get("market_label"),
                "selection_label": item.get("selection_label"),
                "rank_role": item.get("rank_role"),
                "state": item.get("state"),
                "alignment_score": item.get("alignment_score"),
                "rating": item.get("rating"),
                "band": item.get("band"),
                "model_lean": item.get("model_lean"),
                "public_summary": item.get("public_summary"),
            }
            for item in market_intelligence
            if isinstance(item, dict)
        ],
    }


def weather_context(context: dict[str, Any], fixture: dict[str, Any]) -> dict[str, Any]:
    weather = context.get("weather_signals") if isinstance(context.get("weather_signals"), list) else []
    space_weather = context.get("space_weather_signals") if isinstance(context.get("space_weather_signals"), list) else []
    if weather or space_weather:
        return {
            "status": "available" if weather else "experimental_only",
            "weather_signals": weather,
            "space_weather_signals": space_weather,
        }
    if "weather" in canonical_json(fixture).lower():
        return {"status": "available", "source": "fixture_payload"}
    return {"status": "missing", "summary": "No weather context matched this fixture yet."}


def tier_contract() -> dict[str, list[str]]:
    return {
        "free": ["fixture_core", "market_cards", "freshness", "coverage"],
        "founder": ["fixture_core", "market_cards", "h2h", "weather", "team_context_summary", "lineup_summary", "injury_summary"],
        "premium": ["fixture_core", "market_cards", "decision", "h2h", "weather", "team_context", "lineup_context", "fixture_stats"],
        "pro": ["fixture_core", "market_cards", "team_context", "lineup_context", "player_context", "player_event_cards", "injury_context", "player_event_shortlists"],
        "pro_plus": ["all_sections", "audit", "source_refs", "raw_compact_debug"],
    }


def freshness_block(*payloads: dict[str, Any], injury_admin: dict[str, Any]) -> dict[str, Any]:
    timestamps: list[str] = []
    for payload in payloads:
        for key in ("capture_generated_at", "source_data_cutoff_at", "fixture_kickoff_at", "updated_at"):
            value = payload.get(key)
            if value:
                timestamps.append(str(value))
    if injury_admin.get("generated_at"):
        timestamps.append(str(injury_admin["generated_at"]))
    return {
        "last_updated": max(timestamps) if timestamps else "",
        "source_timestamps": sorted(set(timestamps)),
        "next_refresh": "",
        "coverage_status": "compiled_from_local_sources",
    }


def compact_fixture_summary(fixture: dict[str, Any], row: sqlite3.Row) -> dict[str, Any]:
    row_keys = set(row.keys())
    return {
        "fixture_key": fixture.get("fixture_key") or row["fixture_key"],
        "kickoff_time": fixture.get("kickoff_time") or row["kickoff_time"],
        "league": fixture.get("league") or row["league"],
        "home_team": fixture.get("home_team") or row["home_team"],
        "away_team": fixture.get("away_team") or row["away_team"],
        "status": fixture.get("status") or (row["status"] if "status" in row_keys else ""),
    }


def compact_summary_markets(market_cards: dict[str, Any]) -> list[dict[str, Any]]:
    markets: list[dict[str, Any]] = []
    for market in CORE_MARKETS:
        card = market_cards.get(market, {}) if isinstance(market_cards, dict) else {}
        model_output = card.get("model_output") if isinstance(card.get("model_output"), dict) else {}
        probabilities = {}
        for key, value in model_output.items():
            number = to_number(value)
            if number is not None:
                probabilities[str(key)] = number
        markets.append(
            {
                "market_key": market,
                "status": card.get("status") or "missing",
                "model_lean": card.get("model_lean") or "",
                "team_context_lean": card.get("team_context_lean") or "",
                "state": card.get("state") or "",
                "support": to_number(card.get("support")),
                "rating": to_number(card.get("rating")),
                "band": card.get("band") or "",
                "source_status": card.get("source_status") or "",
                "probabilities": probabilities,
                "public_summary": card.get("public_summary") or "",
            }
        )
    return markets


def compact_team_summary(team_context: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for side in ("home", "away"):
        block = team_context.get(side, {}) if isinstance(team_context, dict) else {}
        payload = block.get("payload", {}) if isinstance(block.get("payload"), dict) else {}
        out[side] = {
            "status": block.get("status") or "missing",
            "team": payload.get("team") or block.get("team") or "",
            "summary": payload.get("summary") or "",
            "sample_confidence": payload.get("sample_confidence") or "",
            "profile_tags": payload.get("profile_tags") or [],
            "rating_bands": payload.get("rating_bands") or {},
            "market_tendencies": payload.get("market_tendencies") or {},
        }
    return out


def compact_h2h_summary(h2h: dict[str, Any]) -> dict[str, Any]:
    if not h2h:
        return {"status": "missing"}
    keys = (
        "status",
        "sample_size",
        "summary",
        "goal_heat",
        "over25_heat",
        "btts_pressure",
        "attack_flow",
        "defensive_lock",
        "chaos_rating",
        "first_strike",
    )
    return {key: h2h.get(key) for key in keys if h2h.get(key) not in (None, "", [], {})} or {"status": "available"}


def compact_weather_summary(weather: dict[str, Any]) -> dict[str, Any]:
    if not weather:
        return {"status": "missing"}
    return {
        "status": weather.get("status") or "missing",
        "weather_signals": weather.get("weather_signals") or [],
        "space_weather_signals": weather.get("space_weather_signals") or [],
        "summary": weather.get("summary") or "",
        "source": weather.get("source") or "",
    }


def compact_player_event_summary(player_event_cards: dict[str, Any]) -> dict[str, Any]:
    cards = player_event_cards.get("cards") if isinstance(player_event_cards.get("cards"), list) else []
    return {
        "status": player_event_cards.get("status") or "missing",
        "research_only": True,
        "phase": player_event_cards.get("phase") or "",
        "lineup_status": player_event_cards.get("lineup_status") or "",
        "refresh_policy": player_event_cards.get("refresh_policy") or "",
        "cards": [
            {
                "event_family": card.get("event_family") or "",
                "card_title": card.get("card_title") or "",
                "lineup_status": card.get("lineup_status") or "",
                "shortlist": (card.get("shortlist") or [])[:3],
            }
            for card in cards[:8]
            if isinstance(card, dict)
        ],
    }


def compact_injury_summary(injury_context: dict[str, Any]) -> dict[str, Any]:
    market_impact = injury_context.get("market_impact", {}) if isinstance(injury_context.get("market_impact"), dict) else {}
    player_impacts = injury_context.get("player_impacts") if isinstance(injury_context.get("player_impacts"), list) else []
    summary = injury_context.get("summary", {}) if isinstance(injury_context.get("summary"), dict) else {}
    return {
        "status": summary.get("status") or market_impact.get("status") or "missing",
        "research_only": True,
        "market_adjustment": summary.get("market_adjustment") or market_impact.get("overall_adjustment") or "",
        "market_impacts": summary.get("market_impacts") or market_impact.get("market_impacts") or {},
        "key_absences": market_impact.get("key_absences"),
        "key_players": market_impact.get("key_players") or [],
        "market_impact_notes": market_impact.get("market_impact_notes") or [],
        "player_impacts": [
            {
                "team_side": item.get("team_side") or "",
                "team_name": item.get("team_name") or "",
                "player_name": item.get("player_name") or "",
                "position": item.get("position") or "",
                "absence_type": item.get("absence_type") or "",
                "reason": item.get("reason") or "",
                "availability_impact_status": item.get("availability_impact_status") or "",
                "role_impact": item.get("role_impact") or "",
                "injury_shock_severity": item.get("injury_shock_severity") or "",
                "player_rating_summary": item.get("player_rating_summary") or "",
                "market_impact_note": item.get("market_impact_note") or "",
            }
            for item in player_impacts[:8]
            if isinstance(item, dict)
        ],
    }


def summary_inputs_contract(
    fixture_summary: dict[str, Any],
    market_cards: dict[str, Any],
    decision: dict[str, Any],
    h2h: dict[str, Any],
    weather: dict[str, Any],
    team_context: dict[str, Any],
    player_event_cards: dict[str, Any],
    injury_context: dict[str, Any],
    fixture_stats: dict[str, Any],
    freshness: dict[str, Any],
    coverage: dict[str, Any],
) -> dict[str, Any]:
    standard = {
        "tier": "standard",
        "audience": "novice_public",
        "fixture": fixture_summary,
        "markets": compact_summary_markets(market_cards),
        "freshness": freshness,
        "coverage": {
            "has_decision": coverage.get("has_decision", False),
            "has_fixture_stats": coverage.get("has_fixture_stats", False),
        },
        "copy_rules": [
            "Explain the model output in plain language.",
            "Mention uncertainty without betting advice.",
            "Do not expose premium-only player or audit detail.",
        ],
    }
    premium = {
        **standard,
        "tier": "premium",
        "audience": "paid_context_user",
        "h2h": compact_h2h_summary(h2h),
        "weather": compact_weather_summary(weather),
        "team_context": compact_team_summary(team_context),
        "context_markets": (fixture_stats.get("stats_payload") or {}).get("market_intelligence", []),
        "copy_rules": [
            "Explain support and contradiction from fixture/team/H2H context.",
            "Keep missing data explicit and graceful.",
            "Do not include player-event beta shortlists.",
        ],
    }
    pro = {
        **premium,
        "tier": "pro",
        "audience": "player_event_power_user",
        "player_events": compact_player_event_summary(player_event_cards),
        "injury_context": compact_injury_summary(injury_context),
        "copy_rules": [
            "Connect player-event and injury signals to FTR, BTTS, and OU25 when relevant.",
            "Label beta/player-event outputs as watchlists, not settled recommendations.",
            "Call out pre-lineup versus confirmed-lineup phase.",
        ],
    }
    pro_plus = {
        **pro,
        "tier": "pro_plus",
        "audience": "audit_and_b2b_user",
        "audit": {
            "coverage": coverage,
            "decision_keys": sorted(decision.keys())[:40] if isinstance(decision, dict) else [],
            "fixture_stats_counts": (fixture_stats.get("stats_payload") or {}).get("counts", {}),
            "source_contract": "summary_inputs_only",
        },
        "copy_rules": [
            "Expose audit posture, source coverage, and contradiction reasons.",
            "Keep raw compact debug out of user-facing prose unless explicitly requested.",
            "Prefer structured caveats over confident prose when coverage is partial.",
        ],
    }
    return {
        "schema": "fixture_summary_inputs_v1",
        "generator_status": "ready_for_dry_run",
        "tiers": {tier: block for tier, block in zip(SUMMARY_TIERS, (standard, premium, pro, pro_plus))},
    }


def compile_fixture(
    conn: sqlite3.Connection,
    db_path: Path,
    fixture_row: sqlite3.Row,
    team_index: dict[tuple[str, str], sqlite3.Row],
    premium_team_index: dict[tuple[str, str], sqlite3.Row],
    injury_fixture_index: dict[str, dict[str, Any]],
    injury_player_index: dict[str, list[dict[str, Any]]],
    injury_market_impact_index: dict[str, dict[str, Any]],
    injury_admin: dict[str, Any],
) -> dict[str, Any]:
    fixture_key = fixture_row["fixture_key"]
    fixture = parse_row_payload(fixture_row)
    decision_row = one_by_fixture(conn, "fixture_decisions", fixture_key)
    h2h_row = one_by_fixture(conn, "fixture_h2h", fixture_key)
    lineup_row = one_by_fixture(conn, "fixture_lineups", fixture_key)
    context_row = one_by_fixture(conn, "site_fixture_context_payloads", fixture_key)
    stats_row = one_by_fixture(conn, "site_fixture_stats_payloads", fixture_key)
    market_rows = rows_by_fixture(conn, "site_fixture_market_intelligence", fixture_key, "alignment_score DESC")
    shortlist_rows = rows_by_fixture(conn, "site_player_event_shortlists", fixture_key, "shortlist_score DESC")
    team_stat_rows = rows_by_fixture(conn, "site_team_match_stats", fixture_key, "is_home DESC")
    player_stat_rows = rows_by_fixture(conn, "site_player_match_stats", fixture_key, "rating DESC")

    decision = parse_row_payload(decision_row)
    h2h = parse_row_payload(h2h_row)
    lineup = parse_row_payload(lineup_row)
    context = parse_row_payload(context_row)
    stats = parse_row_payload(stats_row)
    injury_keys = fixture_injury_keys(fixture, fixture_row)
    injury_fixture = compact_injury_fixture(pick_injury_fixture(injury_keys, injury_fixture_index))
    injury_players = compact_player_impacts(pick_injury_players(injury_keys, injury_player_index))
    injury_market_impact = compact_injury_market_impact(pick_injury_fixture(injury_keys, injury_market_impact_index))
    injury_summary = {
        "status": injury_fixture.get("status"),
        "research_only": True,
        "warning_flag": bool(injury_fixture.get("warning_flag")),
        "warning_tokens": injury_fixture.get("warning_tokens", []),
        "absence_edge_side": injury_fixture.get("absence_edge_side", ""),
        "market_adjustment": injury_market_impact.get("overall_adjustment", ""),
        "market_impacts": injury_market_impact.get("market_impacts", {}),
        "player_impacts": len(injury_players),
    }
    market_cards = compact_market_cards(decision, stats)
    weather = weather_context(context, fixture)
    fixture_core = {
        **compact_row(fixture_row),
        "payload": fixture,
    }
    fixture_summary = compact_fixture_summary(fixture, fixture_row)
    competition_key = fixture.get("league_key") or fixture_row["league_key"] or fixture_row["league"]
    home_team = fixture.get("home_team") or fixture_row["home_team"]
    away_team = fixture.get("away_team") or fixture_row["away_team"]
    team_context = {
        "home": team_payload_for(team_index, competition_key, home_team),
        "away": team_payload_for(team_index, competition_key, away_team),
        "premium_home": team_payload_for(premium_team_index, competition_key, home_team, premium=True),
        "premium_away": team_payload_for(premium_team_index, competition_key, away_team, premium=True),
    }
    player_event_cards = compact_player_event_cards(shortlist_rows, fixture, lineup, team_context=team_context)
    injury_context = {
        "fixture_shock": injury_fixture,
        "market_impact": injury_market_impact,
        "player_impacts": injury_players,
        "summary": injury_summary,
        "admin_dashboard": {
            "generated_at": injury_admin.get("generated_at", ""),
            "contract_version": injury_admin.get("contract_version", ""),
            "research_only": injury_admin.get("research_only", True),
        },
    }
    fixture_stats = {
        "stats_payload": compact_fixture_stats_payload(stats),
        "market_intelligence_rows": compact_table_payloads(market_rows, limit=12),
        "team_match_stats": compact_table_payloads(team_stat_rows, limit=8),
    }
    freshness = freshness_block(fixture, decision, h2h, lineup, context, stats, injury_admin=injury_admin)
    coverage = {
        "has_decision": bool(decision_row),
        "has_h2h": bool(h2h_row),
        "has_weather": weather.get("status") != "missing",
        "has_lineup": bool(lineup_row),
        "has_team_context": team_context["home"]["status"] == "available" and team_context["away"]["status"] == "available",
        "has_player_context": bool(player_stat_rows or shortlist_rows or player_event_cards.get("cards")),
        "has_injury_context": injury_fixture.get("status") == "matched",
        "has_injury_market_impact": injury_market_impact.get("status") == "matched",
        "has_fixture_stats": bool(stats_row),
    }
    payload = {
        "schema": "fixture_brain_payload_v2",
        "contract_version": 2,
        "fixture_key": fixture_key,
        "compiled_at": utc_now(),
        "fixture_core": fixture_core,
        "market_cards": market_cards,
        "decision": {"meta": compact_row(decision_row), "payload": decision} if decision_row else {"status": "missing"},
        "h2h": {"meta": compact_row(h2h_row), "payload": h2h} if h2h_row else {"status": "missing"},
        "weather": weather,
        "team_context": team_context,
        "player_context": {
            "player_match_stats": compact_table_payloads(player_stat_rows, limit=18),
            "player_event_shortlists": compact_shortlists(shortlist_rows, limit=18),
        },
        "player_event_cards": player_event_cards,
        "lineup_context": {"meta": compact_row(lineup_row), "payload": lineup} if lineup_row else {"status": "missing"},
        "injury_context": injury_context,
        "fixture_stats": fixture_stats,
        "tier_visibility": tier_contract(),
        "freshness": freshness,
        "coverage": coverage,
        "summary_inputs": summary_inputs_contract(
            fixture_summary=fixture_summary,
            market_cards=market_cards,
            decision=decision,
            h2h=h2h,
            weather=weather,
            team_context=team_context,
            player_event_cards=player_event_cards,
            injury_context=injury_context,
            fixture_stats=fixture_stats,
            freshness=freshness,
            coverage=coverage,
        ),
        "source_refs": {
            "site_db": str(db_path),
            "source_tables": [
                "fixtures",
                "fixture_decisions",
                "fixture_h2h",
                "fixture_lineups",
                "site_fixture_context_payloads",
                "site_fixture_stats_payloads",
                "site_fixture_market_intelligence",
                "site_player_event_shortlists",
                "site_team_match_stats",
                "site_player_match_stats",
                "team_intelligence",
                "site_team_premium_payloads",
            ],
            "injury_match_keys": sorted(injury_keys),
        },
    }
    return payload


def write_manifest(output_dir: Path, objects: list[dict[str, Any]], source_summary: dict[str, Any], window: dict[str, Any]) -> dict[str, Any]:
    coverage_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for item in objects:
        coverage = item.get("coverage", {})
        for key, value in coverage.items():
            coverage_counts[key][str(bool(value)).lower()] += 1
    manifest = {
        "schema": "fixture_brain_manifest_v2",
        "contract_version": 2,
        "generated_at": utc_now(),
        "window": window,
        "source_summary": source_summary,
        "objects": [{key: value for key, value in item.items() if key != "coverage"} for item in objects],
        "summary": {
            "fixtures_compiled": len(objects),
            "total_bytes": sum(int(item.get("bytes") or 0) for item in objects),
            "coverage": {key: dict(counter) for key, counter in sorted(coverage_counts.items())},
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def write_report(report_json: Path, report_md: Path, manifest: dict[str, Any]) -> None:
    write_json(report_json, manifest)
    lines = [
        "# Fixture Brain Compiler Report",
        "",
        f"- Generated: `{manifest['generated_at']}`",
        f"- Window: `{manifest['window']['from']}` to `{manifest['window']['to']}`",
        f"- All fixtures: `{manifest['window']['all_fixtures']}`",
        f"- Fixtures compiled: `{manifest['summary']['fixtures_compiled']}`",
        f"- Total payload bytes: `{manifest['summary']['total_bytes']}`",
        "",
        "## Coverage",
        "",
    ]
    for key, counts in manifest["summary"]["coverage"].items():
        lines.append(f"- {key}: " + ", ".join(f"{label}={count}" for label, count in counts.items()))
    lines.extend(["", "## Injury Sources", ""])
    injury = manifest["source_summary"].get("injury", {})
    fixture_source = injury.get("fixture_source", {})
    lines.append(f"- fixture shock rows: `{fixture_source.get('rows', 0)}` from `{fixture_source.get('path', '')}`")
    for source in injury.get("player_sources", []):
        lines.append(f"- player impact rows: `{source.get('rows', 0)}` from `{source.get('path', '')}`")
    market_source = injury.get("market_impact_source", {})
    lines.append(f"- market impact rows: `{market_source.get('rows', 0)}` from `{market_source.get('path', '')}`")
    if manifest["summary"]["fixtures_compiled"] == 0:
        lines.extend(["", "## Next Step", "", "- Fresh fixture/model/API-football outputs are still needed for this window before site-brain payloads can compile."])
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compile_brain(
    db_path: Path,
    output_dir: Path,
    start: date,
    end: date,
    include_all: bool,
    fixture_keys: set[str],
    injury_fixture_csv: Path,
    injury_player_csvs: list[Path],
    injury_market_impact_csv: Path,
    injury_admin_json: Path,
    preserve_output: bool,
) -> dict[str, Any]:
    if output_dir.exists() and not preserve_output:
        shutil.rmtree(output_dir)
    payload_dir = output_dir / "payloads" / "fixtures"
    payload_dir.mkdir(parents=True, exist_ok=True)

    injury_fixture_index, injury_fixture_source = load_injury_fixture_index(injury_fixture_csv)
    injury_player_index, injury_player_sources = load_injury_player_index(injury_player_csvs)
    injury_market_impact_index, injury_market_impact_source = load_injury_market_impact_index(injury_market_impact_csv)
    injury_admin = read_json(injury_admin_json, {})
    if not isinstance(injury_admin, dict):
        injury_admin = {}

    objects: list[dict[str, Any]] = []
    with connect(db_path) as conn:
        team_index, premium_team_index = load_team_indexes(conn)
        for fixture_row in fixture_rows(conn, start, end, include_all, fixture_keys):
            payload = compile_fixture(
                conn,
                db_path,
                fixture_row,
                team_index,
                premium_team_index,
                injury_fixture_index,
                injury_player_index,
                injury_market_impact_index,
                injury_admin,
            )
            rel_path = Path("payloads") / "fixtures" / f"{fixture_row['fixture_key']}.json"
            bytes_written = write_json(output_dir / rel_path, payload)
            objects.append(
                {
                    "fixture_key": fixture_row["fixture_key"],
                    "relative_path": rel_path.as_posix(),
                    "sha256": sha256_payload(payload),
                    "bytes": bytes_written,
                    "kickoff_time": fixture_row["kickoff_time"],
                    "league": fixture_row["league"],
                    "coverage": payload["coverage"],
                }
            )

    source_summary = {
        "site_db": {"path": str(db_path), "exists": db_path.exists()},
        "injury": {
            "fixture_source": injury_fixture_source,
            "player_sources": injury_player_sources,
            "market_impact_source": injury_market_impact_source,
            "admin_dashboard": {
                "path": str(injury_admin_json),
                "exists": injury_admin_json.exists(),
                "generated_at": injury_admin.get("generated_at", ""),
                "research_only": injury_admin.get("research_only", True),
            },
        },
    }
    window = {"from": start.isoformat(), "to": end.isoformat(), "all_fixtures": include_all, "fixture_keys": sorted(fixture_keys)}
    return write_manifest(output_dir, objects, source_summary, window)


def main() -> int:
    args = parse_args()
    db_path = resolve(args.db)
    output_dir = resolve(args.output_dir)
    start = parse_date(args.from_date) or date.today()
    end = parse_date(args.to_date) if args.to_date else start + timedelta(days=args.days)
    if end is None:
        raise SystemExit(f"Invalid --to-date: {args.to_date}")
    manifest = compile_brain(
        db_path=db_path,
        output_dir=output_dir,
        start=start,
        end=end,
        include_all=args.all_fixtures,
        fixture_keys=set(args.fixture_key or []),
        injury_fixture_csv=resolve(args.injury_fixture_csv),
        injury_player_csvs=[resolve(path) for path in (args.injury_player_csv or list(DEFAULT_INJURY_PLAYER_CSVS))],
        injury_market_impact_csv=resolve(args.injury_market_impact_csv),
        injury_admin_json=resolve(args.injury_admin_json),
        preserve_output=args.preserve_output,
    )
    write_report(resolve(args.report_json), resolve(args.report_md), manifest)
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
