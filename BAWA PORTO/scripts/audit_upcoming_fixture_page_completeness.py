#!/usr/bin/env python3
"""Audit upcoming website fixture page completeness.

This is a website/data-publishing audit only. It reads the local compact site
SQLite and static site payloads, then reports whether each fixture has the
content needed for Standard, Founder, Premium, Pro, and Pro+ surfaces.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "build" / "site_data" / "odds_genius.sqlite"
DEFAULT_DATA_ROOT = ROOT / "frontend" / "public" / "data"
DEFAULT_PUBLISH_DIR = ROOT / "build" / "site_publish" / "current"
DEFAULT_REPORT_JSON = ROOT / "reports" / "latest" / "UPCOMING_FIXTURE_COMPLETENESS_AUDIT.json"
DEFAULT_REPORT_CSV = ROOT / "reports" / "latest" / "UPCOMING_FIXTURE_COMPLETENESS_AUDIT.csv"
DEFAULT_REPORT_MD = ROOT / "reports" / "latest" / "UPCOMING_FIXTURE_COMPLETENESS_AUDIT.md"
CORE_MODEL_MARKETS = ("ftr", "btts", "ou25")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit upcoming fixture page content completeness.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--publish-dir", type=Path, default=DEFAULT_PUBLISH_DIR)
    parser.add_argument("--from-date", default=date.today().isoformat())
    parser.add_argument("--to-date", default="")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--all", action="store_true", help="Audit all fixtures in the site DB.")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_REPORT_CSV)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_REPORT_MD)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def parse_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text[:10]).date()
    except ValueError:
        return None


def parse_json(value: Any, default: Any = None) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def normalize_slug(value: Any) -> str:
    text = str(value or "").lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def one_by_key(conn: sqlite3.Connection, table: str, key_column: str, key_value: str) -> sqlite3.Row | None:
    return conn.execute(f"SELECT * FROM {table} WHERE {key_column} = ? LIMIT 1", (key_value,)).fetchone()


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)).fetchone()
    return row is not None


def load_team_index(conn: sqlite3.Connection) -> dict[tuple[str, str], sqlite3.Row]:
    index: dict[tuple[str, str], sqlite3.Row] = {}
    if not table_exists(conn, "team_intelligence"):
        return index
    for row in conn.execute("SELECT * FROM team_intelligence"):
        competition_key = normalize_slug(row["competition_key"])
        for key in (row["team_slug"], row["team"]):
            slug = normalize_slug(key)
            if competition_key and slug:
                index[(competition_key, slug)] = row
    return index


def load_premium_team_index(conn: sqlite3.Connection) -> set[tuple[str, str]]:
    index: set[tuple[str, str]] = set()
    if not table_exists(conn, "site_team_premium_payloads"):
        return index
    for row in conn.execute("SELECT competition_key, team_slug FROM site_team_premium_payloads"):
        competition_key = normalize_slug(row["competition_key"])
        team_slug = normalize_slug(row["team_slug"])
        if competition_key and team_slug:
            index.add((competition_key, team_slug))
    return index


def status_label(ok: bool, partial: bool = False) -> str:
    if ok:
        return "complete"
    if partial:
        return "partial"
    return "missing"


def has_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple, set)):
        return bool(value)
    return True


def recursive_has_key_fragment(value: Any, fragment: str) -> bool:
    fragment = fragment.lower()
    if isinstance(value, dict):
        for key, child in value.items():
            if fragment in str(key).lower() and has_nonempty(child):
                return True
            if recursive_has_key_fragment(child, fragment):
                return True
    if isinstance(value, list):
        return any(recursive_has_key_fragment(item, fragment) for item in value)
    return False


def model_market_status(decision: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    market_intel = decision.get("market_intelligence")
    if not isinstance(market_intel, dict):
        return "missing", [], list(CORE_MODEL_MARKETS)
    present: list[str] = []
    missing: list[str] = []
    for market in CORE_MODEL_MARKETS:
        block = market_intel.get(market)
        if isinstance(block, dict) and isinstance(block.get("model_output"), dict):
            present.append(market.upper())
        else:
            missing.append(market.upper())
    tg_block = market_intel.get("team_goals")
    if isinstance(tg_block, dict) and (tg_block.get("model_lean") or tg_block.get("team_context_lean") or tg_block.get("state")):
        present.append("TG1.5_SUPPORT")
    status = "complete" if not missing else "partial" if present else "missing"
    return status, present, missing


def odds_status(fixture: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    odds = fixture.get("odds_summary")
    required = {
        "home_win_odds": "FTR_HOME",
        "draw_odds": "FTR_DRAW",
        "away_win_odds": "FTR_AWAY",
        "over25_odds": "OU25_OVER",
        "under25_odds": "OU25_UNDER",
        "btts_yes_odds": "BTTS_YES",
        "btts_no_odds": "BTTS_NO",
    }
    if not isinstance(odds, dict):
        return "missing", [], list(required.values())
    present = [label for key, label in required.items() if odds.get(key) not in (None, "")]
    missing = [label for key, label in required.items() if odds.get(key) in (None, "")]
    status = "complete" if not missing else "partial" if present else "missing"
    return status, present, missing


def h2h_status(h2h: dict[str, Any] | None) -> tuple[str, int, str]:
    if not isinstance(h2h, dict) or not h2h:
        return "missing", 0, "No H2H payload"
    sample_size = int(h2h.get("sample_size") or 0)
    coverage = str(h2h.get("coverage_status") or "").lower()
    fallback = str(h2h.get("fallback_mode") or "").lower()
    if sample_size > 0 and coverage not in {"missing", "unpublished"}:
        return "complete", sample_size, h2h.get("summary") or ""
    if sample_size > 0:
        return "partial", sample_size, h2h.get("summary") or ""
    if coverage or fallback:
        return "partial", sample_size, h2h.get("summary") or ""
    return "missing", sample_size, "No H2H sample"


def is_world_cup_fixture(fixture: dict[str, Any]) -> bool:
    haystack = " ".join(
        str(fixture.get(key) or "")
        for key in ("league", "league_key", "competition", "competition_key", "fixture_key")
    ).lower()
    return "world_cup" in haystack or "world cup" in haystack or "fifa" in haystack


def h2h_status_for_fixture(h2h: dict[str, Any] | None, fixture: dict[str, Any]) -> tuple[str, int, str, bool]:
    state, sample_size, summary = h2h_status(h2h)
    if state == "missing" and is_world_cup_fixture(fixture):
        return "partial", sample_size, "World Cup H2H sparse; graceful fallback allowed", True
    return state, sample_size, summary, False


def lineup_status(lineup: dict[str, Any] | None) -> tuple[str, str]:
    if not isinstance(lineup, dict) or not lineup:
        return "missing", "No lineup payload"
    coverage = str(lineup.get("coverage_status") or "").lower()
    has_units = bool(lineup.get("home_units") or lineup.get("away_units"))
    has_profiles = bool(lineup.get("home_lineup_profiles") or lineup.get("away_lineup_profiles"))
    if coverage not in {"", "missing", "unpublished"} and (has_units or has_profiles):
        return "complete", lineup.get("summary") or ""
    return "partial", lineup.get("summary") or "Lineup fallback available"


def team_status(
    fixture: dict[str, Any],
    team_index: dict[tuple[str, str], sqlite3.Row],
    premium_team_index: set[tuple[str, str]],
) -> tuple[str, list[str], list[str]]:
    competition_key = normalize_slug(fixture.get("league_key") or fixture.get("league"))
    teams = [fixture.get("home_team"), fixture.get("away_team")]
    present: list[str] = []
    missing: list[str] = []
    for team in teams:
        slug = normalize_slug(team)
        if (competition_key, slug) in team_index or (competition_key, slug) in premium_team_index:
            present.append(str(team))
        else:
            missing.append(str(team))
    status = "complete" if not missing else "partial" if present else "missing"
    return status, present, missing


def player_status(stats: dict[str, Any] | None, fixture: dict[str, Any], premium_team_index: set[tuple[str, str]]) -> tuple[str, str]:
    if isinstance(stats, dict):
        player_stats = stats.get("player_stats")
        shortlists = stats.get("player_event_shortlists")
        if isinstance(player_stats, list) and player_stats:
            return "complete", f"{len(player_stats)} player stat rows"
        if isinstance(shortlists, list) and shortlists:
            return "partial", f"{len(shortlists)} player-event shortlist rows"
    competition_key = normalize_slug(fixture.get("league_key") or fixture.get("league"))
    home_key = (competition_key, normalize_slug(fixture.get("home_team")))
    away_key = (competition_key, normalize_slug(fixture.get("away_team")))
    if home_key in premium_team_index and away_key in premium_team_index:
        return "partial", "Team premium player payloads available"
    return "missing", "No player intelligence payload"


def compact_payload_status(path: Path, payload: Any) -> tuple[str, str]:
    if not path.exists():
        return "missing", "No compact R2 payload file"
    if not isinstance(payload, dict):
        return "partial", "Compact payload file is unreadable or malformed"
    if payload.get("schema") == "fixture_page_payload_v2":
        return "complete", "fixture_page_payload_v2"
    return "partial", str(payload.get("schema") or "schema missing")


def fixture_brain_status(payload: Any) -> tuple[str, str]:
    if not isinstance(payload, dict):
        return "missing", "No compact payload"
    brain = payload.get("fixture_brain")
    if not isinstance(brain, dict) or not brain:
        return "missing", "No fixture brain payload"
    coverage = brain.get("coverage")
    tier_visibility = brain.get("tier_visibility")
    has_core = bool(brain.get("market_cards") or brain.get("team_context") or brain.get("fixture_stats"))
    if has_core and isinstance(coverage, dict) and isinstance(tier_visibility, dict):
        return "complete", "Fixture brain contract populated"
    if has_core or coverage or tier_visibility:
        return "partial", "Fixture brain contract partially populated"
    return "missing", "Fixture brain has no usable sections"


def player_event_cards_status(stats: dict[str, Any] | None, publish_payload: Any) -> tuple[str, int, str]:
    brain = publish_payload.get("fixture_brain") if isinstance(publish_payload, dict) else {}
    cards_payload = brain.get("player_event_cards") if isinstance(brain, dict) else {}
    cards = cards_payload.get("cards") if isinstance(cards_payload, dict) else []
    if isinstance(cards, list) and cards:
        return "complete", len(cards), "Fixture-brain player-event cards available"
    protected_shortlists = stats.get("player_event_shortlists") if isinstance(stats, dict) else []
    if isinstance(protected_shortlists, list) and protected_shortlists:
        return "partial", len(protected_shortlists), "Shortlists available; fixture-brain cards not yet compiled"
    return "missing", 0, "No player-event cards or shortlists"


def logo_status(fixture: dict[str, Any]) -> tuple[str, list[str]]:
    missing: list[str] = []
    if not fixture.get("home_team_logo_url"):
        missing.append("home")
    if not fixture.get("away_team_logo_url"):
        missing.append("away")
    return ("complete" if not missing else "partial" if len(missing) == 1 else "missing"), missing


def tier_readiness(checks: dict[str, str]) -> dict[str, str]:
    standard = "ready" if all(checks[name] == "complete" for name in ("logos", "prediction", "odds", "compact_payload")) else "blocked"
    founder = (
        "ready"
        if standard == "ready"
        and checks["fixture_brain"] in {"complete", "partial"}
        and checks["h2h"] in {"complete", "partial"}
        and checks["weather"] != "missing"
        else "partial"
    )
    premium = "ready" if founder == "ready" and checks["team_intelligence"] == "complete" and checks["lineups"] != "missing" else "partial"
    pro = (
        "ready"
        if premium == "ready"
        and checks["player_intelligence"] in {"complete", "partial"}
        and checks["player_event_cards"] in {"complete", "partial"}
        else "partial"
    )
    pro_plus = "ready" if pro == "ready" and checks["fixture_stats"] in {"complete", "partial"} and checks["compact_payload"] == "complete" else "partial"
    return {
        "standard": standard,
        "founder": founder,
        "premium": premium,
        "pro": pro,
        "pro_plus": pro_plus,
    }


def page_status(checks: dict[str, str], tiers: dict[str, str]) -> str:
    if tiers["standard"] == "blocked":
        return "blocked"
    if all(value == "ready" for value in tiers.values()):
        return "launch_ready"
    if checks["weather"] == "missing" or checks["team_intelligence"] == "missing" or checks["fixture_brain"] == "missing":
        return "partial"
    return "tier_partial"


def audit_fixture(
    conn: sqlite3.Connection,
    fixture_row: sqlite3.Row,
    data_root: Path,
    publish_dir: Path,
    team_index: dict[tuple[str, str], sqlite3.Row],
    premium_team_index: set[tuple[str, str]],
) -> dict[str, Any]:
    fixture_key = fixture_row["fixture_key"]
    fixture = parse_json(fixture_row["payload_json"], {}) or {}
    compact_payload_path = publish_dir / "payloads" / "fixtures" / f"{fixture_key}.json"
    compact_payload = read_json(compact_payload_path, None)
    decision_row = one_by_key(conn, "fixture_decisions", "fixture_key", fixture_key)
    h2h_row = one_by_key(conn, "fixture_h2h", "fixture_key", fixture_key)
    lineup_row = one_by_key(conn, "fixture_lineups", "fixture_key", fixture_key)
    stats_row = one_by_key(conn, "site_fixture_stats_payloads", "fixture_key", fixture_key)
    context_row = one_by_key(conn, "site_fixture_context_payloads", "fixture_key", fixture_key) if table_exists(conn, "site_fixture_context_payloads") else None

    decision = parse_json(decision_row["payload_json"], {}) if decision_row else {}
    h2h = parse_json(h2h_row["payload_json"], {}) if h2h_row else {}
    lineup = parse_json(lineup_row["payload_json"], {}) if lineup_row else {}
    stats = parse_json(stats_row["payload_json"], {}) if stats_row else {}
    context = parse_json(context_row["payload_json"], {}) if context_row else {}

    logos, missing_logos = logo_status(fixture)
    prediction, model_markets, missing_model_markets = model_market_status(decision)
    odds, odds_markets, missing_odds_markets = odds_status(fixture)
    h2h_state, h2h_sample_size, h2h_summary, world_cup_h2h_graceful_fallback = h2h_status_for_fixture(h2h, fixture)
    lineup_state, lineup_summary = lineup_status(lineup)
    team_state, teams_present, teams_missing = team_status(fixture, team_index, premium_team_index)
    player_state, player_summary = player_status(stats, fixture, premium_team_index)
    compact_state, compact_summary = compact_payload_status(compact_payload_path, compact_payload)
    brain_state, brain_summary = fixture_brain_status(compact_payload)
    player_event_state, player_event_count, player_event_summary = player_event_cards_status(stats, compact_payload)
    weather = (
        "complete"
        if recursive_has_key_fragment(fixture, "weather")
        or recursive_has_key_fragment(context, "weather")
        or recursive_has_key_fragment(compact_payload, "weather")
        else "missing"
    )
    fixture_stats = "complete" if isinstance(stats, dict) and bool(stats.get("team_stats")) else "partial" if isinstance(stats, dict) and bool(stats.get("market_intelligence")) else "missing"
    static_payload_exists = (data_root / "fixture_decision_intelligence" / f"{fixture_key}.json").exists()
    h2h_static_exists = (data_root / "fixture_h2h_support" / f"{fixture_key}.json").exists()
    lineup_static_exists = (data_root / "fixture_lineup_intelligence" / f"{fixture_key}.json").exists()

    checks = {
        "logos": logos,
        "prediction": prediction,
        "odds": odds,
        "compact_payload": compact_state,
        "fixture_brain": brain_state,
        "weather": weather,
        "h2h": h2h_state,
        "team_intelligence": team_state,
        "player_intelligence": player_state,
        "player_event_cards": player_event_state,
        "lineups": lineup_state,
        "fixture_stats": fixture_stats,
    }
    tiers = tier_readiness(checks)
    kickoff_date = parse_date(fixture.get("kickoff_time") or fixture_row["kickoff_time"])
    row = {
        "fixture_key": fixture_key,
        "kickoff_date": kickoff_date.isoformat() if kickoff_date else "",
        "kickoff_time": fixture.get("kickoff_time") or fixture_row["kickoff_time"],
        "league": fixture.get("league") or fixture_row["league"],
        "home_team": fixture.get("home_team") or fixture_row["home_team"],
        "away_team": fixture.get("away_team") or fixture_row["away_team"],
        "publish_class": fixture.get("publish_class") or fixture_row["publish_class"],
        "fixture_class": fixture.get("fixture_class") or fixture_row["fixture_class"],
        "coverage_status": fixture.get("coverage_status") or fixture_row["coverage_status"],
        "checks": checks,
        "tiers": tiers,
        "page_status": page_status(checks, tiers),
        "details": {
            "missing_logos": missing_logos,
            "model_markets": model_markets,
            "missing_model_markets": missing_model_markets,
            "odds_markets": odds_markets,
            "missing_odds_markets": missing_odds_markets,
            "compact_payload_path": str(compact_payload_path),
            "compact_payload_summary": compact_summary,
            "fixture_brain_summary": brain_summary,
            "h2h_sample_size": h2h_sample_size,
            "h2h_summary": h2h_summary,
            "world_cup_h2h_graceful_fallback": world_cup_h2h_graceful_fallback,
            "lineup_summary": lineup_summary,
            "teams_present": teams_present,
            "teams_missing": teams_missing,
            "player_summary": player_summary,
            "player_event_cards_count": player_event_count,
            "player_event_cards_summary": player_event_summary,
            "static_payload_exists": static_payload_exists,
            "h2h_static_exists": h2h_static_exists,
            "lineup_static_exists": lineup_static_exists,
            "source_data_cutoff_at": fixture.get("source_data_cutoff_at") or fixture_row["source_data_cutoff_at"],
            "capture_generated_at": fixture.get("capture_generated_at") or fixture_row["capture_generated_at"],
        },
    }
    return row


def fixture_rows(conn: sqlite3.Connection, start: date, end: date, include_all: bool) -> list[sqlite3.Row]:
    rows = list(conn.execute("SELECT * FROM fixtures ORDER BY kickoff_time, fixture_key"))
    if include_all:
        return rows
    selected: list[sqlite3.Row] = []
    for row in rows:
        kickoff_date = parse_date(row["kickoff_time"])
        if kickoff_date and start <= kickoff_date <= end:
            selected.append(row)
    return selected


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "fixture_key",
        "kickoff_date",
        "league",
        "home_team",
        "away_team",
        "publish_class",
        "page_status",
        "standard",
        "founder",
        "premium",
        "pro",
        "pro_plus",
        "logos",
        "prediction",
        "odds",
        "compact_payload",
        "fixture_brain",
        "weather",
        "h2h",
        "team_intelligence",
        "player_intelligence",
        "player_event_cards",
        "lineups",
        "fixture_stats",
        "missing_model_markets",
        "missing_odds_markets",
        "teams_missing",
        "player_event_cards_count",
        "world_cup_h2h_graceful_fallback",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{key: row.get(key, "") for key in fields},
                    **row["checks"],
                    **row["tiers"],
                    "missing_model_markets": "|".join(row["details"]["missing_model_markets"]),
                    "missing_odds_markets": "|".join(row["details"]["missing_odds_markets"]),
                    "teams_missing": "|".join(row["details"]["teams_missing"]),
                    "player_event_cards_count": row["details"]["player_event_cards_count"],
                    "world_cup_h2h_graceful_fallback": row["details"]["world_cup_h2h_graceful_fallback"],
                }
            )


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = payload["summary"]
    window_label = f"{payload['window']['from']} to {payload['window']['to']}"
    if payload["window"].get("all"):
        window_label += " (all fixtures)"
    lines = [
        "# Upcoming Fixture Page Completeness Audit",
        "",
        f"- Window: `{window_label}`",
        f"- Fixtures audited: `{summary['fixtures_total']}`",
        f"- Launch-ready: `{summary['page_status'].get('launch_ready', 0)}`",
        f"- Partial: `{summary['page_status'].get('partial', 0) + summary['page_status'].get('tier_partial', 0)}`",
        f"- Blocked: `{summary['page_status'].get('blocked', 0)}`",
        "",
        "## Check Summary",
        "",
    ]
    for check, counts in summary["checks"].items():
        lines.append(f"- {check}: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    lines.extend(["", "## Tier Readiness", ""])
    for tier, counts in summary["tiers"].items():
        lines.append(f"- {tier}: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    blocked = [row for row in payload["fixtures"] if row["page_status"] == "blocked"]
    partial = [row for row in payload["fixtures"] if row["page_status"] in {"partial", "tier_partial"}]
    lines.extend(["", "## Blocked Fixtures", ""])
    if blocked:
        for row in blocked[:60]:
            blockers = [key for key in ("logos", "prediction", "odds", "compact_payload") if row["checks"][key] != "complete"]
            missing = [key for key, value in row["checks"].items() if value == "missing" and key not in blockers]
            suffix = f" | missing: {', '.join(missing)}" if missing else ""
            lines.append(
                f"- `{row['fixture_key']}` {row['home_team']} vs {row['away_team']} | standard blockers: {', '.join(blockers) or 'n/a'}{suffix}"
            )
    else:
        lines.append("- None.")
    lines.extend(["", "## Partial Fixtures", ""])
    if partial:
        for row in partial[:80]:
            partial_bits = [key for key, value in row["checks"].items() if value in {"partial", "missing"}]
            lines.append(f"- `{row['fixture_key']}` {row['home_team']} vs {row['away_team']} | needs: {', '.join(partial_bits) or 'n/a'}")
    else:
        lines.append("- None.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    page_counter = Counter(row["page_status"] for row in rows)
    check_counters: dict[str, dict[str, int]] = {}
    tier_counters: dict[str, dict[str, int]] = {}
    for check in (
        "logos",
        "prediction",
        "odds",
        "compact_payload",
        "fixture_brain",
        "weather",
        "h2h",
        "team_intelligence",
        "player_intelligence",
        "player_event_cards",
        "lineups",
        "fixture_stats",
    ):
        check_counters[check] = dict(Counter(row["checks"][check] for row in rows))
    for tier in ("standard", "founder", "premium", "pro", "pro_plus"):
        tier_counters[tier] = dict(Counter(row["tiers"][tier] for row in rows))
    return {
        "fixtures_total": len(rows),
        "page_status": dict(page_counter),
        "checks": check_counters,
        "tiers": tier_counters,
    }


def main() -> int:
    args = parse_args()
    db_path = resolve_path(args.db)
    data_root = resolve_path(args.data_root)
    publish_dir = resolve_path(args.publish_dir)
    start = parse_date(args.from_date) or date.today()
    end = parse_date(args.to_date) if args.to_date else start + timedelta(days=args.days)
    if end is None:
        raise SystemExit(f"Invalid --to-date: {args.to_date}")
    conn = connect(db_path)
    team_index = load_team_index(conn)
    premium_team_index = load_premium_team_index(conn)
    rows = [
        audit_fixture(conn, row, data_root, publish_dir, team_index, premium_team_index)
        for row in fixture_rows(conn, start, end, args.all)
    ]
    payload = {
        "schema": "upcoming_fixture_page_completeness_audit_v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "db": str(db_path),
        "publish_dir": str(publish_dir),
        "window": {"from": start.isoformat(), "to": end.isoformat(), "all": args.all},
        "summary": summarize(rows),
        "fixtures": rows,
    }
    json_out = resolve_path(args.json_out)
    csv_out = resolve_path(args.csv_out)
    md_out = resolve_path(args.md_out)
    write_json(json_out, payload)
    write_csv(csv_out, rows)
    write_markdown(md_out, payload)
    print(
        json.dumps(
            {
                "fixtures_total": payload["summary"]["fixtures_total"],
                "page_status": payload["summary"]["page_status"],
                "json_out": str(json_out),
                "csv_out": str(csv_out),
                "md_out": str(md_out),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
