#!/usr/bin/env python3
"""Settle published website picks into weekly proof and archive JSON.

This script is a website proof publisher only. It reads already-published,
website-safe prediction JSON and local normalized provider result snapshots,
then writes:

- frontend/public/data/weekly_results.json
- frontend/public/data/results_archive.json

It is intentionally idempotent: rerunning with the same published picks and
same final-score snapshots produces the same archive rows, keyed by
fixture/market/pick/tier/visibility/run lineage.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "frontend" / "public" / "data"
DEFAULT_PUBLIC = DATA_ROOT / "public_predictions.json"
DEFAULT_PREMIUM = DATA_ROOT / "premium_predictions.json"
DEFAULT_SUMMARY = DATA_ROOT / "publish_summary.json"
DEFAULT_PROOF_FEED = DATA_ROOT / "live_results_feed.json"
DEFAULT_WEEKLY = DATA_ROOT / "weekly_results.json"
DEFAULT_ARCHIVE = DATA_ROOT / "results_archive.json"
DEFAULT_REPORT = ROOT / "reports" / "latest" / "RESULTS_SETTLEMENT_REPORT.md"

FINAL_STATUSES = {"FT", "AET", "PEN"}
SETTLED_STATUSES = {"won", "lost", "void"}
MARKET_ORDER = ["FTR", "BTTS", "OU25", "TG1.5"]
ROW_TIMESTAMP_FIELDS = {"settled_at"}
PAYLOAD_TIMESTAMP_FIELDS = {"generated_at"}

LEAGUE_TAG_HINTS = {
    "Australia A-League": "Australia_A_League",
    "Austria Bundesliga": "Austria_Bundesliga",
    "Belgium Pro": "Belgium_Pro",
    "Brazil Serie A": "Brazil_Serie_A",
    "Denmark Superliga": "Denmark_Superliga",
    "England Championship": "England_Championship",
    "England EFL League 1": "England_EFL_League_1",
    "England FA Cup": "England_FA_Cup",
    "England Premier League": "England_Premier_League",
    "France Ligue 1": "France_Ligue_1",
    "Germany Bundesliga": "Germany_Bundesliga",
    "Germany Bundesliga 2": "Germany_Bundesliga_2",
    "Italy Serie A": "Italy_Serie_A",
    "Netherlands Eredivisie": "Netherlands_Eredivisie",
    "Norway Eliteserien": "Norway_Eliteserien",
    "Portugal Liga": "Portugal_Liga",
    "Saudi Pro League": "Saudi_Pro_League",
    "Scotland Premiership": "Scotland_Premiership",
    "South Korea K League": "South_Korea_K_League",
    "Spain La Liga": "Spain_La_Liga",
    "Swiss Super League": "Switzerland_Super_League",
    "Turkey Super Lig": "Turkey_Super_Lig",
    "USA MLS": "USA_MLS",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def comparable(value: Any, excluded_keys: set[str]) -> Any:
    if isinstance(value, dict):
        return {key: comparable(value[key], excluded_keys) for key in sorted(value) if key not in excluded_keys}
    if isinstance(value, list):
        return [comparable(item, excluded_keys) for item in value]
    return value


def materially_equal(left: Any, right: Any, *, excluded_keys: set[str]) -> bool:
    return comparable(left, excluded_keys) == comparable(right, excluded_keys)


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error, UnicodeDecodeError):
        return []


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if text.endswith("%"):
        try:
            return round(float(text[:-1]) / 100.0, 6)
        except ValueError:
            return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    number = as_float(value)
    return None if number is None else int(number)


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def parse_date(value: Any) -> str:
    text = str(value or "").strip()
    return text[:10] if text else ""


def fixture_key_date(value: Any) -> str:
    return parse_date(value).replace("-", "_")


def token_set(value: Any) -> set[str]:
    drop = {
        "1",
        "1899",
        "ac",
        "afc",
        "athletic",
        "borussia",
        "cd",
        "cf",
        "city",
        "club",
        "fc",
        "fk",
        "krc",
        "kv",
        "kvc",
        "rc",
        "royal",
        "sc",
        "stade",
        "sv",
        "the",
        "united",
    }
    aliases = {
        "sj": "san_jose",
        "rsl": "real_salt_lake",
        "gladbach": "monchengladbach",
        "mgladbach": "monchengladbach",
        "munich": "munchen",
        "koln": "koln",
        "köln": "koln",
    }
    tokens: set[str] = set()
    for token in normalize_text(value).split("_"):
        mapped = aliases.get(token, token)
        if mapped and mapped not in drop:
            tokens.update(mapped.split("_"))
    return tokens


def token_score(left: Any, right: Any) -> float:
    lset = token_set(left)
    rset = token_set(right)
    if not lset or not rset:
        return 0.0
    return len(lset & rset) / max(1, min(len(lset), len(rset)))


def market_key(value: Any) -> str:
    text = str(value or "").strip().upper().replace(" ", "_").replace("-", "_")
    text = re.sub(r"[^A-Z0-9_+.]", "", text)
    aliases = {
        "OVER25": "OU25",
        "UNDER25": "OU25",
        "OVER_25": "OU25",
        "UNDER_25": "OU25",
        "OVER_2_5": "OU25",
        "UNDER_2_5": "OU25",
        "BOTH_TEAMS_TO_SCORE": "BTTS",
        "TEAM_GOALS": "TG1.5",
        "TEAM_GOALS_15": "TG1.5",
        "TEAM_GOALS_1_5": "TG1.5",
        "TG15": "TG1.5",
        "TG1_5": "TG1.5",
    }
    return aliases.get(text, text)


def normalize_pick(value: Any) -> str:
    text = str(value or "").strip().upper().replace(" ", "_").replace("-", "_")
    aliases = {
        "HOME_WIN": "HOME",
        "AWAY_WIN": "AWAY",
        "OVER_25": "OVER25",
        "UNDER_25": "UNDER25",
        "OVER_2_5": "OVER25",
        "UNDER_2_5": "UNDER25",
        "BTTSYES": "YES",
        "BTTSNO": "NO",
        "HOME_OVER_15": "HOME_OVER15",
        "AWAY_OVER_15": "AWAY_OVER15",
        "HOME_OVER_1_5": "HOME_OVER15",
        "AWAY_OVER_1_5": "AWAY_OVER15",
        "HOME_TEAM_OVER_15": "HOME_OVER15",
        "AWAY_TEAM_OVER_15": "AWAY_OVER15",
        "HOME_TG15": "HOME_OVER15",
        "AWAY_TG15": "AWAY_OVER15",
    }
    return aliases.get(text, text)


def actual_outcome(market: str, home_goals: int | None, away_goals: int | None) -> str:
    if home_goals is None or away_goals is None:
        return ""
    market = market_key(market)
    if market == "FTR":
        if home_goals > away_goals:
            return "HOME"
        if away_goals > home_goals:
            return "AWAY"
        return "DRAW"
    if market == "BTTS":
        return "YES" if home_goals > 0 and away_goals > 0 else "NO"
    if market == "OU25":
        return "OVER25" if home_goals + away_goals > 2 else "UNDER25"
    return ""


def score_pick(market: str, pick: str, home_goals: int | None, away_goals: int | None) -> str:
    if home_goals is None or away_goals is None:
        return "pending"
    market = market_key(market)
    normalized_pick = normalize_pick(pick)
    if market in {"FTR", "BTTS", "OU25"}:
        actual = actual_outcome(market, home_goals, away_goals)
        return "won" if normalized_pick == actual else "lost"
    if market == "TG1.5":
        if normalized_pick == "HOME_OVER15":
            return "won" if home_goals > 1 else "lost"
        if normalized_pick == "AWAY_OVER15":
            return "won" if away_goals > 1 else "lost"
    return "void"


def profit_units(status: str, odds: float | None) -> float | None:
    if status == "won" and odds is not None:
        return round(odds - 1.0, 4)
    if status == "lost":
        return -1.0
    if status == "void":
        return 0.0
    return None


def prediction_rows(public_path: Path, premium_path: Path) -> list[dict[str, Any]]:
    rows_by_key: dict[str, dict[str, Any]] = {}
    for visibility, path in (("public", public_path), ("premium", premium_path)):
        rows = read_json(path, [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            key = settlement_identity(row)
            existing = rows_by_key.get(key, {})
            merged = {**existing, **row}
            visibilities = set(existing.get("_visibilities", []))
            visibilities.add(visibility)
            merged["_visibilities"] = sorted(visibilities)
            rows_by_key[key] = merged
    return list(rows_by_key.values())


def settlement_identity(row: dict[str, Any], run_id: str = "") -> str:
    fixture_key = row.get("fixture_key") or row.get("fixture_id") or ""
    market = market_key(row.get("market"))
    pick = normalize_pick(row.get("pick") or row.get("selection") or row.get("bookie_pick"))
    tier = str(row.get("confidence_tier") or row.get("premium_tier") or row.get("tier") or "").upper()
    lineage = run_id or str(row.get("published_run_id") or "")
    return "|".join([str(fixture_key), market, pick, tier, lineage])


def candidate_roots(extra_roots: list[Path]) -> list[Path]:
    roots: list[Path] = []
    roots.extend(extra_roots)
    for base in (ROOT / "reports" / "latest", ROOT / "reports", ROOT / "data_sources" / "api_football"):
        if base.exists():
            roots.extend(path for path in base.rglob("normalized") if path.is_dir())
    seen: set[Path] = set()
    ordered: list[Path] = []
    for root in roots:
        resolved = root if root.is_absolute() else ROOT / root
        if resolved.exists() and resolved not in seen:
            ordered.append(resolved)
            seen.add(resolved)
    return sorted(ordered, key=lambda path: (0 if "reports/latest" in str(path) else 1, len(str(path))))


def collect_local_results(extra_roots: list[Path]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for root in candidate_roots(extra_roots):
        for fixture_path in root.glob("fixtures_master__*.csv"):
            league_tag = fixture_path.name.replace("fixtures_master__", "").rsplit("__", 1)[0]
            fixture_rows = read_csv(fixture_path)
            if not fixture_rows:
                continue
            score_by_fixture = scores_for_root(root, fixture_path.name)
            for fixture in fixture_rows:
                fixture_id = as_int(fixture.get("fixture_id"))
                score = score_by_fixture.get(fixture_id or -1, {})
                status = str(fixture.get("status") or score.get("status") or "").upper()
                results.append(
                    {
                        "fixture_id": fixture_id,
                        "league_id": as_int(fixture.get("league_id")),
                        "season": as_int(fixture.get("season")),
                        "league": fixture.get("league") or league_tag.replace("_", " "),
                        "league_tag": league_tag,
                        "match_date": parse_date(fixture.get("match_date") or fixture.get("kickoff_ts_utc") or fixture.get("date")),
                        "home_team": fixture.get("home_team_name") or fixture.get("home_team"),
                        "away_team": fixture.get("away_team_name") or fixture.get("away_team"),
                        "home_goals": score.get("home_goals"),
                        "away_goals": score.get("away_goals"),
                        "status": status,
                        "source_root": str(root),
                    }
                )
    return results


def parse_scoreline(value: Any) -> tuple[int | None, int | None]:
    text = str(value or "").strip()
    match = re.search(r"(\d+)\s*[-:]\s*(\d+)", text)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def collect_proof_feed_results(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path, {})
    windows = payload.get("windows") if isinstance(payload, dict) else []
    if not isinstance(windows, list):
        return []
    by_fixture: dict[str, dict[str, Any]] = {}
    for window in windows:
        if not isinstance(window, dict):
            continue
        rows = []
        for field in ("items", "featured_results"):
            value = window.get(field)
            if isinstance(value, list):
                rows.extend(item for item in value if isinstance(item, dict))
        for row in rows:
            fixture_key = str(row.get("fixture_key") or "")
            if not fixture_key:
                continue
            home_goals, away_goals = parse_scoreline(row.get("score"))
            if home_goals is None or away_goals is None:
                continue
            current = by_fixture.get(fixture_key)
            if current and current.get("source_window") == window.get("window_id"):
                continue
            by_fixture[fixture_key] = {
                "fixture_key": fixture_key,
                "league": row.get("league") or "",
                "match_date": parse_date(row.get("kickoff_time")),
                "home_team": row.get("home_team") or "",
                "away_team": row.get("away_team") or "",
                "home_goals": home_goals,
                "away_goals": away_goals,
                "status": "FT",
                "source_root": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
                "source_window": window.get("window_id") or "",
            }
    return list(by_fixture.values())


def scores_for_root(root: Path, fixture_filename: str) -> dict[int, dict[str, Any]]:
    fixture_rows = read_csv(root / fixture_filename)
    status_by_fixture = {
        as_int(row.get("fixture_id")): str(row.get("status") or "").upper()
        for row in fixture_rows
        if as_int(row.get("fixture_id")) is not None
    }
    stat_path = root / fixture_filename.replace("fixtures_master__", "match_team_stats__")
    stat_rows = read_csv(stat_path)
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in stat_rows:
        fixture_id = as_int(row.get("fixture_id"))
        if fixture_id is not None:
            grouped[fixture_id].append(row)
    out: dict[int, dict[str, Any]] = {}
    for fixture_id, rows in grouped.items():
        home = next((row for row in rows if str(row.get("is_home")).lower() in {"1", "true"}), None)
        away = next((row for row in rows if str(row.get("is_home")).lower() in {"0", "false"}), None)
        if home and away:
            out[fixture_id] = {
                "home_goals": as_int(home.get("goals_for")),
                "away_goals": as_int(away.get("goals_for")),
                "status": status_by_fixture.get(fixture_id, ""),
            }
    return out


def resolve_result(row: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any] | None:
    row_date = parse_date(row.get("kickoff_time") or row.get("match_date"))
    league_hint = LEAGUE_TAG_HINTS.get(str(row.get("league") or ""), "")
    fixture_id = as_int(row.get("api_fixture_id") or row.get("provider_fixture_id"))
    fixture_key = str(row.get("fixture_key") or "")
    if fixture_key:
        exact_key = next((result for result in results if str(result.get("fixture_key") or "") == fixture_key), None)
        if exact_key:
            return exact_key
    if fixture_id is not None:
        exact = next((result for result in results if result.get("fixture_id") == fixture_id), None)
        if exact:
            return exact
    best: dict[str, Any] | None = None
    best_score = 0.0
    best_final = -1
    for result in results:
        if row_date and result.get("match_date") != row_date:
            continue
        if league_hint and result.get("league_tag") != league_hint and normalize_text(row.get("league")) not in normalize_text(result.get("league")):
            continue
        home_score = token_score(row.get("home_team"), result.get("home_team"))
        away_score = token_score(row.get("away_team"), result.get("away_team"))
        score = min(home_score, away_score)
        final_score = 1 if str(result.get("status") or "").upper() in FINAL_STATUSES else 0
        if score > best_score or (score == best_score and final_score > best_final):
            best = result
            best_score = score
            best_final = final_score
    if best and best_score >= 0.66:
        matched = dict(best)
        matched["match_score"] = round(best_score, 4)
        return matched
    return None


def row_visibility(row: dict[str, Any]) -> str:
    visibilities = set(row.get("_visibilities") or [])
    if "public" in visibilities:
        return "public"
    return "premium"


def settle_row(row: dict[str, Any], result: dict[str, Any] | None, *, run_id: str, settled_at: str) -> dict[str, Any]:
    market = market_key(row.get("market"))
    pick = normalize_pick(row.get("pick") or row.get("selection") or row.get("bookie_pick"))
    status = str(result.get("status") or "").upper() if result else ""
    final = status in FINAL_STATUSES
    home_score = result.get("home_goals") if result and final else None
    away_score = result.get("away_goals") if result and final else None
    result_status = score_pick(market, pick, home_score, away_score) if final else "pending"
    odds = as_float(row.get("bookie_od"))
    tier = str(row.get("confidence_tier") or row.get("premium_tier") or row.get("tier") or "").upper()
    item = {
        "settlement_key": settlement_identity(row, run_id),
        "fixture_id": row.get("fixture_id") or row.get("api_fixture_id") or result.get("fixture_id") if result else row.get("fixture_id"),
        "fixture_key": row.get("fixture_key") or "",
        "kickoff_time": row.get("kickoff_time") or "",
        "league": row.get("league") or result.get("league") if result else row.get("league") or "",
        "home_team": row.get("home_team") or "",
        "away_team": row.get("away_team") or "",
        "market": market,
        "pick": pick,
        "confidence_tier": tier,
        "tier": tier,
        "premium_tier": str(row.get("premium_tier") or tier),
        "visibility": row_visibility(row),
        "bookie_od": odds,
        "model_prob": as_float(row.get("model_prob") or row.get("model_prob_display")),
        "bookie_implied_prob": as_float(row.get("bookie_implied_prob")),
        "value_edge": as_float(row.get("value_edge") or row.get("value_edge_display")),
        "actual": actual_outcome(market, home_score, away_score) if final else "",
        "result_status": result_status,
        "profit_units": profit_units(result_status, odds),
        "final_home_score": home_score,
        "final_away_score": away_score,
        "provider_status": status,
        "actual_source": result.get("source_root") if result else "",
        "match_score": result.get("match_score") if result else None,
        "settled_at": settled_at if result_status in SETTLED_STATUSES else "",
        "published_run_id": run_id,
    }
    return item


def existing_rows_by_key(*payloads: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows_by_key: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        items = payload.get("items") if isinstance(payload, dict) else []
        if not isinstance(items, list):
            continue
        for row in items:
            if not isinstance(row, dict):
                continue
            key = row.get("settlement_key") or settlement_identity(row, str(row.get("published_run_id") or ""))
            if key:
                rows_by_key[str(key)] = row
    return rows_by_key


def preserve_settled_at(new_row: dict[str, Any], existing_row: dict[str, Any] | None) -> dict[str, Any]:
    if new_row.get("result_status") not in SETTLED_STATUSES:
        new_row["settled_at"] = ""
        return new_row
    if not existing_row:
        return new_row
    existing_settled_at = str(existing_row.get("settled_at") or "")
    if existing_settled_at and materially_equal(new_row, existing_row, excluded_keys=ROW_TIMESTAMP_FIELDS):
        new_row["settled_at"] = existing_settled_at
    return new_row


def preserve_generated_at(new_payload: dict[str, Any], existing_payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(existing_payload, dict):
        return new_payload
    existing_generated_at = str(existing_payload.get("generated_at") or "")
    if existing_generated_at and materially_equal(new_payload, existing_payload, excluded_keys=PAYLOAD_TIMESTAMP_FIELDS):
        new_payload["generated_at"] = existing_generated_at
    return new_payload


def summarize(rows: list[dict[str, Any]], label_field: str | None = None) -> dict[str, Any] | list[dict[str, Any]]:
    def block(items: list[dict[str, Any]]) -> dict[str, Any]:
        settled = [row for row in items if row.get("result_status") in SETTLED_STATUSES]
        wins = sum(1 for row in settled if row.get("result_status") == "won")
        losses = sum(1 for row in settled if row.get("result_status") == "lost")
        voids = sum(1 for row in settled if row.get("result_status") == "void")
        pending = sum(1 for row in items if row.get("result_status") == "pending")
        profits = [row.get("profit_units") for row in settled if row.get("profit_units") is not None]
        profit = round(float(sum(profits)), 4) if profits else 0.0
        return {
            "total_picks": len(items),
            "settled_picks": len(settled),
            "pending_picks": pending,
            "wins": wins,
            "losses": losses,
            "voids": voids,
            "hit_rate": round(wins / max(1, wins + losses), 4) if wins + losses else None,
            "roi": round(profit / max(1, len([p for p in profits if p is not None])), 4) if profits else None,
            "profit_units": profit,
        }

    if label_field is None:
        return block(rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(label_field) or "UNKNOWN")].append(row)
    output: list[dict[str, Any]] = []
    for label in sorted(grouped):
        item = block(grouped[label])
        item[label_field] = label
        output.append(item)
    return output


def chart_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_date[parse_date(row.get("kickoff_time"))].append(row)
    points: list[dict[str, Any]] = []
    cumulative_profit = 0.0
    cumulative_settled = 0
    cumulative_wins = 0
    cumulative_losses = 0
    for date in sorted(key for key in by_date if key):
        settled = [row for row in by_date[date] if row.get("result_status") in SETTLED_STATUSES]
        wins = sum(1 for row in settled if row.get("result_status") == "won")
        losses = sum(1 for row in settled if row.get("result_status") == "lost")
        profit = round(sum(float(row.get("profit_units") or 0) for row in settled), 4)
        cumulative_profit = round(cumulative_profit + profit, 4)
        cumulative_settled += len(settled)
        cumulative_wins += wins
        cumulative_losses += losses
        points.append(
            {
                "date": date,
                "settled_picks": len(settled),
                "wins": wins,
                "losses": losses,
                "voids": sum(1 for row in settled if row.get("result_status") == "void"),
                "profit_units": profit,
                "cumulative_profit_units": cumulative_profit,
                "rolling_hit_rate": round(wins / max(1, wins + losses), 4) if wins + losses else None,
                "cumulative_roi": round(cumulative_profit / max(1, cumulative_settled), 4),
                "cumulative_hit_rate": round(cumulative_wins / max(1, cumulative_wins + cumulative_losses), 4)
                if cumulative_wins + cumulative_losses
                else None,
            }
        )
    return points


def build_weekly_payload(rows: list[dict[str, Any]], *, generated_at: str, run_id: str) -> dict[str, Any]:
    dates = [parse_date(row.get("kickoff_time")) for row in rows if parse_date(row.get("kickoff_time"))]
    overall = summarize(rows)
    assert isinstance(overall, dict)
    featured_wins = [row for row in rows if row.get("result_status") == "won"][:8]
    featured_misses = [row for row in rows if row.get("result_status") == "lost"][:8]
    payload = {
        "period_start": min(dates) if dates else "",
        "period_end": max(dates) if dates else "",
        "generated_at": generated_at,
        "source_file": "published_website_predictions",
        "published_run_id": run_id,
        **overall,
        "overall_hit_rate": overall.get("hit_rate"),
        "overall_roi": overall.get("roi"),
        "overall_profit_units": overall.get("profit_units"),
        "by_market": summarize(rows, "market"),
        "by_tier": summarize(rows, "tier"),
        "by_league": summarize(rows, "league"),
        "by_visibility": summarize(rows, "visibility"),
        "chart_points": chart_points(rows),
        "featured_wins": featured_wins,
        "featured_misses": featured_misses,
        "items": sorted(rows, key=lambda row: (row.get("kickoff_time") or "", row.get("league") or "", row.get("fixture_key") or "")),
        "notes": [
            "Generated from published website prediction JSON and normalized provider result snapshots.",
            "OBSERVE/research rows are not promoted here unless they were present in the published prediction inputs.",
            "Pending rows are retained until a final provider score is available.",
        ],
    }
    return payload


def merge_archive(existing: dict[str, Any], new_items: list[dict[str, Any]], *, generated_at: str) -> dict[str, Any]:
    existing_items = existing.get("items") if isinstance(existing, dict) else []
    rows_by_key: dict[str, dict[str, Any]] = {}
    if isinstance(existing_items, list):
        for row in existing_items:
            if isinstance(row, dict):
                key = row.get("settlement_key") or settlement_identity(row, str(row.get("published_run_id") or ""))
                rows_by_key[str(key)] = row
    for row in new_items:
        rows_by_key[str(row["settlement_key"])] = row
    items = sorted(rows_by_key.values(), key=lambda row: (row.get("kickoff_time") or "", row.get("league") or "", row.get("fixture_key") or ""))
    dates = [parse_date(row.get("kickoff_time")) for row in items if parse_date(row.get("kickoff_time"))]
    overall = summarize(items)
    assert isinstance(overall, dict)
    return {
        "period_start": min(dates) if dates else "",
        "period_end": max(dates) if dates else "",
        "generated_at": generated_at,
        **overall,
        "overall_hit_rate": overall.get("hit_rate"),
        "overall_roi": overall.get("roi"),
        "overall_profit_units": overall.get("profit_units"),
        "by_market": summarize(items, "market"),
        "by_tier": summarize(items, "tier"),
        "by_league": summarize(items, "league"),
        "by_visibility": summarize(items, "visibility"),
        "chart_points": chart_points(items),
        "featured_wins": [row for row in items if row.get("result_status") == "won"][-12:],
        "featured_misses": [row for row in items if row.get("result_status") == "lost"][-12:],
        "items": items,
        "notes": [
            "Cumulative idempotent archive. Reruns replace rows with the same settlement_key instead of duplicating them.",
        ],
    }


def write_report(path: Path, weekly: dict[str, Any], archive: dict[str, Any], unmatched: list[dict[str, Any]]) -> None:
    lines = [
        "# Results Settlement Report",
        "",
        f"Generated: `{weekly['generated_at']}`",
        f"Published run: `{weekly['published_run_id']}`",
        "",
        "## Current Window",
        "",
        f"- Window: `{weekly['period_start']}` to `{weekly['period_end']}`",
        f"- Picks: `{weekly['total_picks']}`",
        f"- Settled: `{weekly['settled_picks']}`",
        f"- Pending: `{weekly['pending_picks']}`",
        f"- Wins/Losses/Voids: `{weekly['wins']}/{weekly['losses']}/{weekly['voids']}`",
        f"- Hit rate: `{weekly['overall_hit_rate']}`",
        f"- ROI: `{weekly['overall_roi']}`",
        "",
        "## Archive",
        "",
        f"- Archive picks: `{archive['total_picks']}`",
        f"- Archive settled: `{archive['settled_picks']}`",
        f"- Archive hit rate: `{archive['overall_hit_rate']}`",
        "",
        "## Current Market Split",
        "",
    ]
    for row in weekly["by_market"]:
        lines.append(
            f"- {row['market']}: {row['wins']}/{row['settled_picks']} settled, pending={row['pending_picks']}, hit_rate={row['hit_rate']}"
        )
    lines.extend(["", "## Unmatched / Pending Inputs", ""])
    if unmatched:
        for row in unmatched[:40]:
            lines.append(
                f"- {row.get('kickoff_time')} | {row.get('league')} | {row.get('home_team')} vs {row.get('away_team')} | {row.get('market')} {row.get('pick')}"
            )
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- This script does not generate predictions.",
            "- This script does not alter deploy routing, tiers, vetoes, or slip formatting.",
            "- Pending rows are retained rather than dropped.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Settle published website predictions into proof JSON.")
    parser.add_argument("--public", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--premium", type=Path, default=DEFAULT_PREMIUM)
    parser.add_argument("--publish-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--proof-feed", type=Path, default=DEFAULT_PROOF_FEED)
    parser.add_argument("--weekly-out", type=Path, default=DEFAULT_WEEKLY)
    parser.add_argument("--archive-out", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--results-root", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    public_path = args.public if args.public.is_absolute() else ROOT / args.public
    premium_path = args.premium if args.premium.is_absolute() else ROOT / args.premium
    summary_path = args.publish_summary if args.publish_summary.is_absolute() else ROOT / args.publish_summary
    proof_feed_path = args.proof_feed if args.proof_feed.is_absolute() else ROOT / args.proof_feed
    weekly_out = args.weekly_out if args.weekly_out.is_absolute() else ROOT / args.weekly_out
    archive_out = args.archive_out if args.archive_out.is_absolute() else ROOT / args.archive_out
    report_out = args.report_out if args.report_out.is_absolute() else ROOT / args.report_out

    publish_summary = read_json(summary_path, {})
    run_id = (
        str(publish_summary.get("published_run_id") or "")
        or Path(str(publish_summary.get("selected_source_csv") or "published_website_predictions")).stem
    )
    predictions = prediction_rows(public_path, premium_path)
    results = collect_proof_feed_results(proof_feed_path) + collect_local_results(args.results_root)
    existing_weekly = read_json(weekly_out, {})
    existing_archive = read_json(archive_out, {})
    existing_rows = existing_rows_by_key(existing_archive, existing_weekly)
    generated_at = utc_now()

    settled_rows: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    for row in predictions:
        result = resolve_result(row, results)
        settled = settle_row(row, result, run_id=run_id, settled_at=generated_at)
        settled = preserve_settled_at(settled, existing_rows.get(str(settled.get("settlement_key") or "")))
        settled_rows.append(settled)
        if settled["result_status"] == "pending":
            unmatched.append(settled)

    weekly = build_weekly_payload(settled_rows, generated_at=generated_at, run_id=run_id)
    archive = merge_archive(existing_archive, settled_rows, generated_at=generated_at)
    weekly = preserve_generated_at(weekly, existing_weekly)
    archive = preserve_generated_at(archive, existing_archive)

    write_json(weekly_out, weekly)
    write_json(archive_out, archive)
    write_report(report_out, weekly, archive, unmatched)
    print(
        json.dumps(
            {
                "weekly_out": display_path(weekly_out),
                "archive_out": display_path(archive_out),
                "report_out": display_path(report_out),
                "published_run_id": run_id,
                "current_total": weekly["total_picks"],
                "current_settled": weekly["settled_picks"],
                "current_pending": weekly["pending_picks"],
                "archive_total": archive["total_picks"],
                "archive_settled": archive["settled_picks"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
