#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.api_football.client import APIFootballClient

FIXTURE_FEED_PATH = ROOT / "frontend" / "public" / "data" / "fixture_intelligence_public.json"
DECISION_ROOT = ROOT / "frontend" / "public" / "data" / "fixture_decision_intelligence"
LINEUP_ROOT = ROOT / "frontend" / "public" / "data" / "fixture_lineup_intelligence"
H2H_ROOT = ROOT / "frontend" / "public" / "data" / "fixture_h2h_support"
SQLITE_PATH = ROOT / "build" / "site_data" / "odds_genius.sqlite"
PREMIUM_PREDICTIONS_PATH = ROOT / "frontend" / "public" / "data" / "premium_predictions.json"
DEFAULT_OUTDIR = ROOT / "reports" / "latest" / "weekend_prediction_intelligence_scoring"

CORE_MARKETS = ("FTR", "BTTS", "OU25")
PUBLISH_CLASSES = ("DEPLOY", "OBSERVE", "CONTEXT")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path, fallback: Any = None) -> Any:
    if not path.exists():
        return fallback
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def normalize_fixture_key_date(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text[:10].replace("-", "_")


def provider_match_key(match_date: Any, home_team: Any, away_team: Any) -> str:
    return "_".join(
        part
        for part in (
            normalize_fixture_key_date(match_date),
            normalize_text(home_team),
            normalize_text(away_team),
        )
        if part
    )


def club_tokens(value: Any) -> set[str]:
    aliases = {
        "saint": "st",
        "sint": "st",
        "st": "st",
    }
    drop = {
        "a",
        "ac",
        "afc",
        "cf",
        "cd",
        "club",
        "fc",
        "fk",
        "krc",
        "kv",
        "kvc",
        "rc",
        "royal",
        "sc",
        "the",
        "va",
    }
    tokens = set()
    for token in normalize_text(value).split("_"):
        token = aliases.get(token, token)
        if token and token not in drop:
            tokens.add(token)
    return tokens


def token_match_score(left: Any, right: Any) -> float:
    left_tokens = club_tokens(left)
    right_tokens = club_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    return overlap / max(1, min(len(left_tokens), len(right_tokens)))


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def to_int(value: Any) -> int | None:
    number = to_float(value)
    if number is None:
        return None
    return int(number)


def market_key(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"OVER25", "UNDER25", "OVER_25", "UNDER_25"}:
        return "OU25"
    if text in {"BOTH_TEAMS_TO_SCORE"}:
        return "BTTS"
    return text


def normalize_pick(value: Any) -> str:
    text = str(value or "").strip().upper()
    text = text.replace(" ", "_")
    aliases = {
        "HOME_WIN": "HOME",
        "AWAY_WIN": "AWAY",
        "DRAW": "DRAW",
        "YES": "YES",
        "NO": "NO",
        "OVER_2_5": "OVER25",
        "UNDER_2_5": "UNDER25",
        "OVER_25": "OVER25",
        "UNDER_25": "UNDER25",
    }
    return aliases.get(text, text)


def score_pick(market: str, pick: str, home_score: int | None, away_score: int | None) -> str:
    if home_score is None or away_score is None:
        return "missing_actual"
    market = market_key(market)
    pick = normalize_pick(pick)
    total_goals = home_score + away_score
    if market == "FTR":
        actual = "HOME" if home_score > away_score else "AWAY" if away_score > home_score else "DRAW"
    elif market == "BTTS":
        actual = "YES" if home_score > 0 and away_score > 0 else "NO"
    elif market == "OU25":
        actual = "OVER25" if total_goals > 2.5 else "UNDER25"
    else:
        return "unsupported_market"
    return "won" if pick == actual else "lost"


def actual_outcome(market: str, home_score: int | None, away_score: int | None) -> str:
    if home_score is None or away_score is None:
        return ""
    market = market_key(market)
    if market == "FTR":
        return "HOME" if home_score > away_score else "AWAY" if away_score > home_score else "DRAW"
    if market == "BTTS":
        return "YES" if home_score > 0 and away_score > 0 else "NO"
    if market == "OU25":
        return "OVER25" if home_score + away_score > 2.5 else "UNDER25"
    return ""


def profit_units(status: str, odds: float | None) -> float | None:
    if odds is None:
        return None
    if status == "won":
        return round(odds - 1.0, 4)
    if status == "lost":
        return -1.0
    return None


def result_from_provider_item(item: dict[str, Any]) -> dict[str, Any]:
    fixture = item.get("fixture") or {}
    teams = item.get("teams") or {}
    goals = item.get("goals") or {}
    league = item.get("league") or {}
    status = fixture.get("status") or {}
    home_team = (teams.get("home") or {}).get("name")
    away_team = (teams.get("away") or {}).get("name")
    return {
        "api_fixture_id": fixture.get("id"),
        "match_date": fixture.get("date"),
        "league": league.get("name"),
        "league_id": league.get("id"),
        "season": league.get("season"),
        "home_team": home_team,
        "away_team": away_team,
        "home_score": to_int(goals.get("home")),
        "away_score": to_int(goals.get("away")),
        "status_short": status.get("short"),
        "status_long": status.get("long"),
        "provider_key": provider_match_key(fixture.get("date"), home_team, away_team),
    }


def fetch_provider_results(fixtures: list[dict[str, Any]], args: argparse.Namespace, outdir: Path) -> dict[str, dict[str, Any]]:
    pair_to_site_leagues: dict[tuple[int, int], set[str]] = defaultdict(set)
    for fx in fixtures:
        if fx.get("api_league_id") is None or fx.get("api_season") is None:
            continue
        pair_to_site_leagues[(int(fx["api_league_id"]), int(fx["api_season"]))].add(str(fx.get("league") or ""))
    league_pairs = sorted(pair_to_site_leagues)
    if not league_pairs:
        return {}

    client = APIFootballClient(sleep_seconds=args.sleep_seconds, daily_cap=args.daily_cap)
    raw_payloads: list[dict[str, Any]] = []
    by_api_id: dict[str, dict[str, Any]] = {}
    by_provider_key: dict[str, dict[str, Any]] = {}
    by_league_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_league_name_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_site_league_date: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for league_id, season in league_pairs:
        payload = client.get_json(
            "/fixtures",
            {
                "league": league_id,
                "season": season,
                "from": args.from_date,
                "to": args.to_date,
                "timezone": args.timezone,
            },
        )
        raw_payloads.append(
            {
                "league_id": league_id,
                "season": season,
                "payload": payload,
            }
        )
        for item in payload.get("response") or []:
            result = result_from_provider_item(item)
            api_id = result.get("api_fixture_id")
            if api_id is not None:
                by_api_id[str(api_id)] = result
            if result.get("provider_key"):
                by_provider_key[str(result["provider_key"])] = result
            league_date_key = f"{league_id}:{normalize_fixture_key_date(result.get('match_date'))}"
            by_league_date[league_date_key].append(result)
            league_name_date_key = f"{normalize_text(result.get('league'))}:{normalize_fixture_key_date(result.get('match_date'))}"
            by_league_name_date[league_name_date_key].append(result)
            for site_league in pair_to_site_leagues.get((league_id, season), set()):
                site_league_date_key = f"{normalize_text(site_league)}:{normalize_fixture_key_date(result.get('match_date'))}"
                by_site_league_date[site_league_date_key].append(result)

    write_json(outdir / "raw_provider_results.json", raw_payloads)
    return {
        "by_api_id": by_api_id,
        "by_provider_key": by_provider_key,
        "by_league_date": dict(by_league_date),
        "by_league_name_date": dict(by_league_name_date),
        "by_site_league_date": dict(by_site_league_date),
    }


def sqlite_actuals(sqlite_path: Path) -> dict[str, dict[str, Any]]:
    if not sqlite_path.exists():
        return {}
    query = """
        select
          fixture_key,
          max(case when is_home = 1 then json_extract(payload_json, '$.goals_for') end) as home_score,
          max(case when is_home = 0 then json_extract(payload_json, '$.goals_for') end) as away_score
        from site_team_match_stats
        group by fixture_key
    """
    out: dict[str, dict[str, Any]] = {}
    with sqlite3.connect(str(sqlite_path)) as conn:
        conn.row_factory = sqlite3.Row
        for row in conn.execute(query):
            out[str(row["fixture_key"])] = {
                "home_score": to_int(row["home_score"]),
                "away_score": to_int(row["away_score"]),
                "source": "site_team_match_stats",
            }
    return out


def premium_prediction_index(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows = read_json(path, []) or []
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("fixture_key") or ""),
            market_key(row.get("market")),
            normalize_pick(row.get("pick")),
        )
        index[key] = row
    return index


def decision_payload(fixture_key: str) -> dict[str, Any]:
    return read_json(DECISION_ROOT / f"{fixture_key}.json", {}) or {}


def coverage_payload(root: Path, fixture_key: str) -> dict[str, Any]:
    return read_json(root / f"{fixture_key}.json", {}) or {}


def extract_observe_pick(fixture: dict[str, Any], decision: dict[str, Any]) -> tuple[str, str]:
    summary = fixture.get("signal_summary") or {}
    market = market_key(summary.get("market_family"))
    if not market:
        return "", ""

    market_payload = (decision.get("market_intelligence") or {}).get(market.lower()) or {}
    lean = normalize_pick(market_payload.get("model_lean") or market_payload.get("selection_label"))
    if lean and lean not in {"OPEN", "ELEVATED"}:
        return market, lean

    label = " ".join(
        str(value or "").lower()
        for value in (
            summary.get("signal_label"),
            summary.get("headline"),
            summary.get("summary_text"),
        )
    )
    if market == "FTR":
        if "away" in label:
            return market, "AWAY"
        if "home" in label:
            return market, "HOME"
        if "draw" in label:
            return market, "DRAW"
    if market == "BTTS":
        return market, "NO" if "low-event" in label or "lower-event" in label else "YES"
    if market == "OU25":
        return market, "UNDER25" if "low-event" in label or "lower-event" in label else "OVER25"
    return market, lean


def extract_primary_pick(fixture: dict[str, Any], decision: dict[str, Any]) -> tuple[str, str, str, str]:
    publish_class = str(fixture.get("publish_class") or fixture.get("fixture_class") or "").upper()
    summary = fixture.get("signal_summary") or {}
    deploy = fixture.get("deploy_summary") or {}
    if publish_class == "DEPLOY":
        market = market_key(deploy.get("market") or summary.get("market_family"))
        pick = normalize_pick(deploy.get("pick") or summary.get("deploy_pick"))
        tier = str(deploy.get("confidence_tier") or summary.get("confidence_tier") or "STANDARD").upper()
        return market, pick, tier, "deploy_pick"
    if publish_class == "OBSERVE":
        market, pick = extract_observe_pick(fixture, decision)
        return market, pick, "OBSERVE", "observe_signal"
    return "", "", publish_class or "CONTEXT", "context_only"


def actual_for_fixture(
    fixture: dict[str, Any],
    provider_results: dict[str, dict[str, dict[str, Any]]],
    db_actuals: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    fixture_key = str(fixture.get("fixture_key") or "")
    db = db_actuals.get(fixture_key)
    if db and db.get("home_score") is not None and db.get("away_score") is not None:
        return {
            "home_score": db.get("home_score"),
            "away_score": db.get("away_score"),
            "actual_source": str(db.get("source") or "site_team_match_stats"),
            "status_short": "FT",
        }

    by_api_id = provider_results.get("by_api_id") or {}
    by_provider_key = provider_results.get("by_provider_key") or {}
    by_league_date = provider_results.get("by_league_date") or {}
    by_league_name_date = provider_results.get("by_league_name_date") or {}
    by_site_league_date = provider_results.get("by_site_league_date") or {}
    api_fixture_id = fixture.get("api_fixture_id")
    result = by_api_id.get(str(api_fixture_id)) if api_fixture_id is not None else None
    if result is None:
        result = by_provider_key.get(
            provider_match_key(fixture.get("kickoff_time"), fixture.get("home_team"), fixture.get("away_team"))
        )
    if result is None and fixture.get("api_league_id") is not None:
        league_date_key = f"{int(fixture['api_league_id'])}:{normalize_fixture_key_date(fixture.get('kickoff_time'))}"
        candidates = by_league_date.get(league_date_key) or []
        best_result = None
        best_score = 0.0
        for candidate in candidates:
            home_score_match = token_match_score(fixture.get("home_team"), candidate.get("home_team"))
            away_score_match = token_match_score(fixture.get("away_team"), candidate.get("away_team"))
            combined = min(home_score_match, away_score_match)
            if combined > best_score:
                best_score = combined
                best_result = candidate
        if best_score >= 0.66:
            result = dict(best_result or {})
            result["match_method"] = "league_date_team_alias"
    if result is None:
        league_name_date_key = f"{normalize_text(fixture.get('league'))}:{normalize_fixture_key_date(fixture.get('kickoff_time'))}"
        candidates = (by_site_league_date.get(league_name_date_key) or []) + (
            by_league_name_date.get(league_name_date_key) or []
        )
        best_result = None
        best_score = 0.0
        for candidate in candidates:
            home_score_match = token_match_score(fixture.get("home_team"), candidate.get("home_team"))
            away_score_match = token_match_score(fixture.get("away_team"), candidate.get("away_team"))
            combined = min(home_score_match, away_score_match)
            if combined > best_score:
                best_score = combined
                best_result = candidate
        if best_score >= 0.66:
            result = dict(best_result or {})
            result["match_method"] = "league_name_date_team_alias"
    if result is None:
        return {"home_score": None, "away_score": None, "actual_source": "", "status_short": ""}
    return {
        "home_score": result.get("home_score"),
        "away_score": result.get("away_score"),
        "actual_source": result.get("match_method") or "api_football_fixture_result",
        "status_short": result.get("status_short") or "",
    }


def count_sqlite_rows(sqlite_path: Path, table: str, fixture_key: str) -> int:
    if not sqlite_path.exists():
        return 0
    with sqlite3.connect(str(sqlite_path)) as conn:
        return int(conn.execute(f"select count(*) from {table} where fixture_key = ?", (fixture_key,)).fetchone()[0])


def intelligence_context(fixture_key: str, decision: dict[str, Any], sqlite_path: Path) -> dict[str, Any]:
    lineup = coverage_payload(LINEUP_ROOT, fixture_key)
    h2h = coverage_payload(H2H_ROOT, fixture_key)
    drivers = decision.get("key_player_drivers") or []
    if isinstance(drivers, dict):
        drivers = drivers.get("items") or drivers.get("drivers") or []
    driver_names = []
    for item in drivers[:4] if isinstance(drivers, list) else []:
        if isinstance(item, dict):
            driver_names.append(str(item.get("player_name") or item.get("name") or "").strip())
    market_intel = decision.get("market_intelligence") or {}
    return {
        "decision_signal_state": str(decision.get("signal_state") or ""),
        "decision_primary_signal": str(decision.get("primary_signal") or ""),
        "agreement_score": decision.get("agreement_score"),
        "confidence_band": str(decision.get("confidence_band") or ""),
        "lineup_coverage_status": str(lineup.get("coverage_status") or lineup.get("source_status") or ""),
        "h2h_coverage_status": str(h2h.get("coverage_status") or h2h.get("fallback_mode") or ""),
        "player_driver_count": len(driver_names),
        "player_driver_names": "; ".join(name for name in driver_names if name),
        "team_stat_rows": count_sqlite_rows(sqlite_path, "site_team_match_stats", fixture_key),
        "player_stat_rows": count_sqlite_rows(sqlite_path, "site_player_match_stats", fixture_key),
        "player_event_shortlist_rows": count_sqlite_rows(sqlite_path, "site_player_event_shortlists", fixture_key),
        "ftr_state": str((market_intel.get("ftr") or {}).get("state") or ""),
        "btts_state": str((market_intel.get("btts") or {}).get("state") or ""),
        "ou25_state": str((market_intel.get("ou25") or {}).get("state") or ""),
        "ftr_alignment": (market_intel.get("ftr") or {}).get("alignment_score"),
        "btts_alignment": (market_intel.get("btts") or {}).get("alignment_score"),
        "ou25_alignment": (market_intel.get("ou25") or {}).get("alignment_score"),
    }


def summarize(rows: list[dict[str, Any]], group_fields: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field, "") for field in group_fields)].append(row)

    out: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items(), key=lambda pair: tuple(str(v) for v in pair[0])):
        settled = [row for row in items if row.get("result_status") in {"won", "lost"}]
        wins = sum(1 for row in settled if row.get("result_status") == "won")
        losses = sum(1 for row in settled if row.get("result_status") == "lost")
        profit_values = [float(row.get("profit_units")) for row in settled if row.get("profit_units") not in (None, "")]
        block = {field: key[idx] for idx, field in enumerate(group_fields)}
        block.update(
            {
                "rows": len(items),
                "settled": len(settled),
                "missing_actual": sum(1 for row in items if row.get("result_status") == "missing_actual"),
                "wins": wins,
                "losses": losses,
                "hit_rate": round(wins / len(settled), 4) if settled else None,
                "profit_units": round(sum(profit_values), 4) if profit_values else None,
                "roi": round(sum(profit_values) / len(profit_values), 4) if profit_values else None,
            }
        )
        out.append(block)
    return out


def build_rows(
    fixtures: list[dict[str, Any]],
    provider_results: dict[str, dict[str, dict[str, Any]]],
    db_actuals: dict[str, dict[str, Any]],
    premium_index: dict[tuple[str, str, str], dict[str, Any]],
    sqlite_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    primary_rows: list[dict[str, Any]] = []
    market_rows: list[dict[str, Any]] = []

    for fixture in fixtures:
        fixture_key = str(fixture.get("fixture_key") or "")
        publish_class = str(fixture.get("publish_class") or fixture.get("fixture_class") or "").upper()
        if publish_class not in PUBLISH_CLASSES:
            continue
        decision = decision_payload(fixture_key)
        actual = actual_for_fixture(fixture, provider_results, db_actuals)
        home_score = actual.get("home_score")
        away_score = actual.get("away_score")
        context = intelligence_context(fixture_key, decision, sqlite_path)

        market, pick, tier, signal_source = extract_primary_pick(fixture, decision)
        deployed_market_payload = (decision.get("market_intelligence") or {}).get(market.lower()) or {}
        deployed_market_state = str(deployed_market_payload.get("state") or "")
        deployed_market_rank_role = str(deployed_market_payload.get("rank_role") or "")
        deployed_market_support = deployed_market_payload.get("structural_support") or []
        deployed_market_cautions = deployed_market_payload.get("cautions") or []
        deployed_market_summary = str(deployed_market_payload.get("public_summary") or "")
        prediction = premium_index.get((fixture_key, market, pick), {})
        odds = to_float((fixture.get("deploy_summary") or {}).get("bookie_od") or prediction.get("bookie_od"))
        value_edge = to_float(prediction.get("value_edge"))
        ev_label = str((fixture.get("deploy_summary") or {}).get("value_edge_label") or "").lower()
        ev_positive = bool(value_edge is not None and value_edge > 0) or ev_label == "positive"
        result_status = score_pick(market, pick, home_score, away_score) if market and pick else "no_signal"
        primary_rows.append(
            {
                "fixture_key": fixture_key,
                "kickoff_time": fixture.get("kickoff_time"),
                "league": fixture.get("league"),
                "home_team": fixture.get("home_team"),
                "away_team": fixture.get("away_team"),
                "publish_class": publish_class,
                "tier": tier,
                "signal_source": signal_source,
                "market": market,
                "pick": pick,
                "actual": actual_outcome(market, home_score, away_score),
                "home_score": home_score,
                "away_score": away_score,
                "status_short": actual.get("status_short"),
                "actual_source": actual.get("actual_source"),
                "result_status": result_status,
                "bookie_od": odds,
                "value_edge": value_edge,
                "ev_positive": ev_positive,
                "profit_units": profit_units(result_status, odds),
                "deployed_market_state": deployed_market_state,
                "deployed_market_rank_role": deployed_market_rank_role,
                "deployed_market_support": ";".join(str(item) for item in deployed_market_support),
                "deployed_market_cautions": ";".join(str(item) for item in deployed_market_cautions),
                "deployed_market_summary": deployed_market_summary,
                **context,
            }
        )

        market_intel = decision.get("market_intelligence") or {}
        for core_market in CORE_MARKETS:
            payload = market_intel.get(core_market.lower()) or {}
            if not payload:
                continue
            market_pick = normalize_pick(payload.get("model_lean") or payload.get("selection_label"))
            if not market_pick or market_pick in {"OPEN", "ELEVATED"}:
                continue
            market_status = score_pick(core_market, market_pick, home_score, away_score)
            market_rows.append(
                {
                    "fixture_key": fixture_key,
                    "kickoff_time": fixture.get("kickoff_time"),
                    "league": fixture.get("league"),
                    "home_team": fixture.get("home_team"),
                    "away_team": fixture.get("away_team"),
                    "publish_class": publish_class,
                    "primary_tier": tier,
                    "market": core_market,
                    "pick": market_pick,
                    "actual": actual_outcome(core_market, home_score, away_score),
                    "home_score": home_score,
                    "away_score": away_score,
                    "result_status": market_status,
                    "state": payload.get("state"),
                    "rank_role": payload.get("rank_role"),
                    "alignment_score": payload.get("alignment_score"),
                    "rating": payload.get("rating"),
                    "band": payload.get("band"),
                    "signal_state": payload.get("signal_state"),
                    "public_summary": payload.get("public_summary"),
                    **context,
                }
            )

    return primary_rows, market_rows


def summary_markdown(summary_payload: dict[str, Any]) -> str:
    primary = summary_payload["primary_prediction_summary"]
    market = summary_payload["market_intelligence_summary"]
    lines = [
        "# Weekend Prediction Intelligence Scoring",
        "",
        f"Generated: {summary_payload['generated_at']}",
        f"Window: {summary_payload['period_start']} to {summary_payload['period_end']}",
        "",
        "## Coverage",
        "",
        f"- Fixture feed rows: {summary_payload['fixture_count']}",
        f"- Primary scored/settled rows: {primary['settled']} of {primary['rows']}",
        f"- Missing actuals: {primary['missing_actual']}",
        f"- Market-intelligence settled rows: {market['settled']} of {market['rows']}",
        "",
        "## Primary Prediction Score",
        "",
        f"- Wins: {primary['wins']}",
        f"- Losses: {primary['losses']}",
        f"- Hit rate: {primary['hit_rate']}",
        f"- EV+ rows settled: {summary_payload['ev_positive_summary']['settled']}",
        f"- EV+ wins: {summary_payload['ev_positive_summary']['wins']}",
        f"- EV+ hit rate: {summary_payload['ev_positive_summary']['hit_rate']}",
        "",
        "## Observe Signals",
        "",
        f"- Observe settled rows: {summary_payload['observe_summary']['settled']}",
        f"- Observe wins: {summary_payload['observe_summary']['wins']}",
        f"- Observe hit rate: {summary_payload['observe_summary']['hit_rate']}",
        "",
        "## Notes",
        "",
        "- DEPLOY rows use the published deploy pick and ELITE/STANDARD tier.",
        "- OBSERVE rows use the strongest published observed signal and are marked as tier OBSERVE.",
        "- Market-intelligence rows score the FTR, BTTS and OU25 decision-layer leans separately from the primary deploy/observe pick.",
        "- Missing actuals mean the provider result layer is not available locally or from the optional API fetch for that fixture.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score current weekend DEPLOY/OBSERVE fixture predictions alongside published intelligence layers."
    )
    parser.add_argument("--fixture-feed", type=Path, default=FIXTURE_FEED_PATH)
    parser.add_argument("--sqlite", type=Path, default=SQLITE_PATH)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--from-date", default="")
    parser.add_argument("--to-date", default="")
    parser.add_argument("--timezone", default="Europe/London")
    parser.add_argument("--fetch-api-results", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--daily-cap", type=int, default=75000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feed = read_json(args.fixture_feed, {})
    fixtures = list(feed.get("fixtures") or [])
    if not fixtures:
        raise SystemExit(f"No fixtures found in {args.fixture_feed}")

    source_window = feed.get("source_window") or {}
    args.from_date = args.from_date or source_window.get("date_from") or ""
    args.to_date = args.to_date or source_window.get("date_to") or ""
    if not args.from_date or not args.to_date:
        raise SystemExit("Provide --from-date and --to-date, or use a fixture feed with source_window.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    provider_results: dict[str, dict[str, dict[str, Any]]] = {}
    if args.fetch_api_results:
        provider_results = fetch_provider_results(fixtures, args, args.outdir)

    db_actuals = sqlite_actuals(args.sqlite)
    premium_index = premium_prediction_index(PREMIUM_PREDICTIONS_PATH)
    primary_rows, market_rows = build_rows(fixtures, provider_results, db_actuals, premium_index, args.sqlite)

    primary_fieldnames = [
        "fixture_key",
        "kickoff_time",
        "league",
        "home_team",
        "away_team",
        "publish_class",
        "tier",
        "signal_source",
        "market",
        "pick",
        "actual",
        "home_score",
        "away_score",
        "status_short",
        "actual_source",
        "result_status",
        "bookie_od",
        "value_edge",
        "ev_positive",
        "profit_units",
        "deployed_market_state",
        "deployed_market_rank_role",
        "deployed_market_support",
        "deployed_market_cautions",
        "deployed_market_summary",
        "decision_signal_state",
        "decision_primary_signal",
        "agreement_score",
        "confidence_band",
        "lineup_coverage_status",
        "h2h_coverage_status",
        "player_driver_count",
        "player_driver_names",
        "team_stat_rows",
        "player_stat_rows",
        "player_event_shortlist_rows",
        "ftr_state",
        "btts_state",
        "ou25_state",
        "ftr_alignment",
        "btts_alignment",
        "ou25_alignment",
    ]
    market_fieldnames = [
        "fixture_key",
        "kickoff_time",
        "league",
        "home_team",
        "away_team",
        "publish_class",
        "primary_tier",
        "market",
        "pick",
        "actual",
        "home_score",
        "away_score",
        "result_status",
        "state",
        "rank_role",
        "alignment_score",
        "rating",
        "band",
        "signal_state",
        "public_summary",
        "decision_primary_signal",
        "agreement_score",
        "confidence_band",
        "lineup_coverage_status",
        "h2h_coverage_status",
        "player_driver_count",
        "team_stat_rows",
        "player_stat_rows",
        "player_event_shortlist_rows",
    ]

    write_csv(args.outdir / "primary_prediction_score_rows.csv", primary_rows, primary_fieldnames)
    write_csv(args.outdir / "market_intelligence_score_rows.csv", market_rows, market_fieldnames)

    primary_summary = summarize(primary_rows, [])[0]
    market_summary = summarize(market_rows, [])[0] if market_rows else {"rows": 0, "settled": 0, "missing_actual": 0, "wins": 0, "losses": 0, "hit_rate": None}
    observe_rows = [row for row in primary_rows if row.get("tier") == "OBSERVE"]
    ev_rows = [row for row in primary_rows if row.get("ev_positive") is True]

    payload = {
        "generated_at": utc_now(),
        "period_start": args.from_date,
        "period_end": args.to_date,
        "fixture_count": len(fixtures),
        "provider_result_fetch_enabled": bool(args.fetch_api_results),
        "provider_result_count_by_api_id": len(provider_results.get("by_api_id") or {}),
        "sqlite_actual_fixture_count": len(db_actuals),
        "primary_prediction_summary": primary_summary,
        "market_intelligence_summary": market_summary,
        "by_tier": summarize(primary_rows, ["tier"]),
        "by_publish_class": summarize(primary_rows, ["publish_class"]),
        "by_market": summarize(primary_rows, ["market"]),
        "by_tier_market": summarize(primary_rows, ["tier", "market"]),
        "observe_summary": summarize(observe_rows, [])[0] if observe_rows else {"rows": 0, "settled": 0, "wins": 0, "losses": 0, "hit_rate": None},
        "ev_positive_summary": summarize(ev_rows, [])[0] if ev_rows else {"rows": 0, "settled": 0, "wins": 0, "losses": 0, "hit_rate": None},
        "market_intelligence_by_market_state": summarize(market_rows, ["market", "state"]),
        "outputs": {
            "primary_prediction_score_rows": str((args.outdir / "primary_prediction_score_rows.csv").relative_to(ROOT)),
            "market_intelligence_score_rows": str((args.outdir / "market_intelligence_score_rows.csv").relative_to(ROOT)),
            "summary_json": str((args.outdir / "summary.json").relative_to(ROOT)),
            "summary_md": str((args.outdir / "SUMMARY.md").relative_to(ROOT)),
        },
    }
    write_json(args.outdir / "summary.json", payload)
    (args.outdir / "SUMMARY.md").write_text(summary_markdown(payload), encoding="utf-8")

    print(f"Fixture rows: {len(fixtures)}")
    print(f"Provider result fetch enabled: {bool(args.fetch_api_results)}")
    print(f"Provider result rows by API id: {payload['provider_result_count_by_api_id']}")
    print(f"SQLite actual fixtures: {len(db_actuals)}")
    print(f"Primary settled: {primary_summary['settled']} / {primary_summary['rows']}")
    print(f"Primary wins/losses: {primary_summary['wins']} / {primary_summary['losses']}")
    print(f"Primary hit rate: {primary_summary['hit_rate']}")
    print(f"Observe settled: {payload['observe_summary']['settled']}")
    print(f"EV+ settled: {payload['ev_positive_summary']['settled']}")
    print(f"Outputs: {args.outdir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
