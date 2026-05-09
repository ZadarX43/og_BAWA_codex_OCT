#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import re
import unicodedata

ROOT = Path(__file__).resolve().parent
FRONTEND_DATA = ROOT / "frontend" / "public" / "data"
REPORTS_LATEST = ROOT / "reports" / "latest"

UNIVERSE_PATH = FRONTEND_DATA / "covered_fixture_universe.json"
INTELLIGENCE_PATH = FRONTEND_DATA / "fixture_intelligence_public.json"
OVERLAY_SUMMARY_PATH = REPORTS_LATEST / "api_current_context_overlay_window" / "CURRENT_CONTEXT_OVERLAY_SUMMARY.json"

AUDIT_JSON_PATH = FRONTEND_DATA / "league_coverage_audit.json"
AUDIT_REPORT_PATH = REPORTS_LATEST / "LEAGUE_COVERAGE_AUDIT_REPORT.md"

TEAM_STOPWORDS = {
    "fc", "cf", "sc", "ac", "kv", "fk", "rb", "eh", "club",
    "city", "united",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_team_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("&", " and ")
    tokens = re.findall(r"[a-z0-9]+", text)
    filtered = [token for token in tokens if token not in TEAM_STOPWORDS]
    return "_".join(filtered)


def fixture_identity_key(match_date: str, home_team: str, away_team: str) -> str:
    return "|".join(
        [
            str(match_date or "").strip(),
            normalize_team_name(home_team),
            normalize_team_name(away_team),
        ]
    )


def bool_count(records: list[dict[str, Any]], key: str) -> int:
    return sum(1 for record in records if bool(record.get(key)))


def classify_league(record: dict[str, Any]) -> tuple[str, list[str]]:
    fixture_counts = record["fixture_counts"]
    market_output = record["market_output"]
    overlay_support = record["overlay_support"]
    routing_presence = record["routing_presence"]

    notes: list[str] = []
    covered_total = int(fixture_counts["covered_total"])
    routed_total = int(fixture_counts["routed_total"])
    deploy_total = int(fixture_counts["deploy_total"])
    observe_total = int(fixture_counts["observe_total"])
    context_total = int(fixture_counts["context_total"])
    historical_overlay = bool(overlay_support["historical_overlay"])
    current_support_count = sum(
        1
        for key in (
            "current_odds",
            "current_injuries",
            "current_lineups",
            "current_team_stats",
            "current_player_stats",
            "current_match_events",
        )
        if overlay_support.get(key)
    )
    market_count = sum(1 for value in market_output.values() if value)

    if routing_presence["allmarkets_present"]:
        notes.append("League has fixtures entering the ALLMARKETS intake base.")
    else:
        notes.append("League did not reach the ALLMARKETS intake base in this window.")

    if deploy_total > 0:
        notes.append("League routed at least one deployable fixture in the active window.")
    elif observe_total > 0:
        notes.append("League routed observe intelligence but no deployable fixture in the active window.")
    elif context_total > 0:
        notes.append("League is currently represented through non-routed context coverage.")

    if historical_overlay:
        notes.append("Historical overlay support exists for this league.")
    if current_support_count > 0:
        notes.append(f"Current-window overlay refresh produced {current_support_count} live support families.")

    if not routing_presence["allmarkets_present"] and not historical_overlay and current_support_count == 0:
        return "blind_spot", notes

    if deploy_total == 0 and observe_total == 0 and context_total > 0:
        if historical_overlay or current_support_count > 0:
            return "context_only", notes
        return "blind_spot", notes

    if covered_total == 0 and routed_total == 0 and current_support_count == 0 and not historical_overlay:
        return "blind_spot", notes

    non_routed_total = int(fixture_counts["non_routed_total"])
    route_ratio = routed_total / covered_total if covered_total else 0.0
    if deploy_total > 0 and observe_total > 0 and market_count >= 2 and route_ratio >= 0.6:
        return "full_coverage", notes

    if routing_presence["allmarkets_present"] or observe_total > 0 or deploy_total > 0 or historical_overlay or current_support_count > 0:
        return "partial_coverage", notes

    return "blind_spot", notes


def build_audit() -> dict[str, Any]:
    universe = load_json(UNIVERSE_PATH)
    intelligence = load_json(INTELLIGENCE_PATH)
    overlay_summary = load_json(OVERLAY_SUMMARY_PATH) if OVERLAY_SUMMARY_PATH.exists() else {"fixtures": []}

    source_window = universe.get("source_window") or intelligence.get("source_window") or {}
    generated_at = intelligence.get("generated_at") or universe.get("generated_at")
    source_run_id = intelligence.get("source_run_id") or universe.get("source_run_id")

    universe_fixtures = universe.get("fixtures", [])
    intelligence_fixtures = intelligence.get("fixtures", [])
    overlay_fixtures = overlay_summary.get("fixtures", [])

    leagues = sorted(
        {
            str(row.get("league", "") or "").strip()
            for row in [*universe_fixtures, *intelligence_fixtures, *overlay_fixtures]
            if str(row.get("league", "") or "").strip()
        }
    )

    universe_by_league: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in universe_fixtures:
        league = str(row.get("league", "") or "").strip()
        if league:
            universe_by_league[league].append(row)

    intelligence_by_league: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in intelligence_fixtures:
        league = str(row.get("league", "") or "").strip()
        if league:
            intelligence_by_league[league].append(row)

    overlay_by_league: dict[str, list[dict[str, Any]]] = defaultdict(list)
    overlay_by_fixture_key: dict[str, dict[str, Any]] = {}
    overlay_by_identity: dict[str, dict[str, Any]] = {}
    for row in overlay_fixtures:
        league = str(row.get("league", "") or "").strip()
        if league:
            overlay_by_league[league].append(row)
        fixture_key = str(row.get("fixture_key", "") or "").strip()
        if fixture_key:
            overlay_by_fixture_key[fixture_key] = row
        identity = fixture_identity_key(
            str(row.get("match_date", "") or "").strip(),
            str(row.get("home_team_name", "") or "").strip(),
            str(row.get("away_team_name", "") or "").strip(),
        )
        if identity != "||":
            overlay_by_identity[identity] = row

    records: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()

    for league in leagues:
        u_rows = universe_by_league.get(league, [])
        i_rows = intelligence_by_league.get(league, [])
        matched_overlay_rows = []
        for row in u_rows:
            fixture_key = str(row.get("fixture_key", "") or "").strip()
            overlay_row = overlay_by_fixture_key.get(fixture_key)
            if overlay_row is None:
                identity = fixture_identity_key(
                    str(row.get("kickoff_time", "") or "").strip()[:10],
                    str(row.get("home_team", "") or "").strip(),
                    str(row.get("away_team", "") or "").strip(),
                )
                overlay_row = overlay_by_identity.get(identity)
            if overlay_row is not None:
                matched_overlay_rows.append(overlay_row)
        o_rows = matched_overlay_rows or overlay_by_league.get(league, [])

        publish_counts = Counter(str(row.get("publish_class", "") or "").strip() for row in i_rows)
        routed_market_families = {
            str((row.get("signal_summary") or {}).get("market_family", "") or "").strip()
            for row in i_rows
            if str(row.get("publish_class", "") or "").strip() in {"DEPLOY", "OBSERVE"}
        }
        overlay_availability = {
            "current_odds": any(bool((row.get("availability") or {}).get("prematch_odds")) for row in o_rows),
            "current_injuries": any(bool((row.get("availability") or {}).get("injuries")) for row in o_rows),
            "current_lineups": any(bool((row.get("availability") or {}).get("lineups")) for row in o_rows),
            "current_team_stats": any(bool((row.get("availability") or {}).get("team_stats")) for row in o_rows),
            "current_player_stats": any(bool((row.get("availability") or {}).get("player_stats")) for row in o_rows),
            "current_match_events": any(bool((row.get("availability") or {}).get("match_events")) for row in o_rows),
        }
        historical_overlay = any(
            bool((row.get("historical_overlay_profile") or {}).get(side))
            for row in u_rows
            for side in ("home", "away")
        )

        record = {
            "league": league,
            "window_from": source_window.get("date_from"),
            "window_to": source_window.get("date_to"),
            "fixture_counts": {
                "covered_total": len(u_rows),
                "routed_total": sum(1 for row in u_rows if row.get("routing_status") == "routed"),
                "non_routed_total": sum(1 for row in u_rows if row.get("routing_status") == "non_routed"),
                "deploy_total": int(publish_counts.get("DEPLOY", 0)),
                "observe_total": int(publish_counts.get("OBSERVE", 0)),
                "context_total": int(publish_counts.get("CONTEXT", 0)),
                "monitor_total": int(publish_counts.get("MONITOR", 0)),
                "hidden_total": sum(1 for row in u_rows if row.get("routing_status") == "hidden"),
            },
            "market_output": {
                "ftr_present": "FTR" in routed_market_families,
                "btts_present": "BTTS" in routed_market_families,
                "ou25_present": "OU25" in routed_market_families,
            },
            "routing_presence": {
                "allmarkets_present": any(row.get("routing_status") == "routed" for row in u_rows),
                "observe_present": publish_counts.get("OBSERVE", 0) > 0,
                "deploy_present": publish_counts.get("DEPLOY", 0) > 0,
                "loss_report_presence": any(row.get("routing_status") == "non_routed" for row in u_rows),
            },
            "overlay_support": {
                "historical_overlay": historical_overlay,
                **overlay_availability,
            },
            "source_counts": {
                "overlay_fixture_rows": len(o_rows),
                "intelligence_fixture_rows": len(i_rows),
            },
        }
        classification, notes = classify_league(record)
        record["classification"] = classification
        record["notes"] = notes
        class_counts[classification] += 1
        records.append(record)

    payload = {
        "generated_at": generated_at,
        "source_run_id": source_run_id,
        "source_window": source_window,
        "coverage_summary": {
            "total_leagues": len(records),
            "full_coverage_count": int(class_counts.get("full_coverage", 0)),
            "partial_coverage_count": int(class_counts.get("partial_coverage", 0)),
            "context_only_count": int(class_counts.get("context_only", 0)),
            "blind_spot_count": int(class_counts.get("blind_spot", 0)),
        },
        "leagues": records,
    }
    return payload


def write_report(payload: dict[str, Any]) -> None:
    summary = payload["coverage_summary"]
    records = payload["leagues"]
    lines = [
        "# LEAGUE_COVERAGE_AUDIT_REPORT",
        "",
        f"Generated: `{payload.get('generated_at', '')}`",
        f"Source run id: `{payload.get('source_run_id', '')}`",
        f"Source window: `{payload.get('source_window', {}).get('date_from', '')}` to `{payload.get('source_window', {}).get('date_to', '')}`",
        "",
        "## Coverage Summary",
        f"- Total leagues: `{summary['total_leagues']}`",
        f"- `full_coverage`: `{summary['full_coverage_count']}`",
        f"- `partial_coverage`: `{summary['partial_coverage_count']}`",
        f"- `context_only`: `{summary['context_only_count']}`",
        f"- `blind_spot`: `{summary['blind_spot_count']}`",
        "",
        "## League Breakdown",
    ]
    for record in sorted(records, key=lambda row: (row["classification"], row["league"])):
        markets = record["market_output"]
        counts = record["fixture_counts"]
        overlay = record["overlay_support"]
        lines.extend(
            [
                f"### {record['league']}",
                f"- Classification: `{record['classification']}`",
                (
                    f"- Fixtures: covered `{counts['covered_total']}` | routed `{counts['routed_total']}` | "
                    f"deploy `{counts['deploy_total']}` | observe `{counts['observe_total']}` | "
                    f"context `{counts['context_total']}` | monitor `{counts['monitor_total']}`"
                ),
                (
                    f"- Markets: `FTR={markets['ftr_present']}` | `BTTS={markets['btts_present']}` | "
                    f"`OU25={markets['ou25_present']}`"
                ),
                (
                    f"- Overlay: `historical={overlay['historical_overlay']}` | `odds={overlay['current_odds']}` | "
                    f"`injuries={overlay['current_injuries']}` | `lineups={overlay['current_lineups']}` | "
                    f"`team_stats={overlay['current_team_stats']}` | `player_stats={overlay['current_player_stats']}`"
                ),
            ]
        )
        for note in record["notes"]:
            lines.append(f"- Note: {note}")
        lines.append("")

    AUDIT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_REPORT_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    payload = build_audit()
    AUDIT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    write_report(payload)

    print(f"League coverage audit written: {AUDIT_JSON_PATH.relative_to(ROOT)}")
    print(f"League coverage audit report written: {AUDIT_REPORT_PATH.relative_to(ROOT)}")
    print(f"Total leagues: {payload['coverage_summary']['total_leagues']}")
    print(f"Partial coverage: {payload['coverage_summary']['partial_coverage_count']}")
    print(f"Context only: {payload['coverage_summary']['context_only_count']}")
    print(f"Blind spots: {payload['coverage_summary']['blind_spot_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
