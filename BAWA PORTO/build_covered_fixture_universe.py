#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from publish_predictions import (
    DEFAULT_LOGO_MANIFEST,
    FRONTEND_DATA_DIR,
    REPORTS_DIR,
    ROOT,
    ensure_dirs,
    load_logo_manifest,
    load_rows,
    logo_assets_for,
    normalize_lookup_text,
    utc_now_iso,
    write_json,
)

OUTPUT_PATH = FRONTEND_DATA_DIR / "covered_fixture_universe.json"
REPORT_PATH = REPORTS_DIR / "COVERED_FIXTURE_UNIVERSE_REPORT.md"
FIXTURES_MASTER_PATH = ROOT / "data_sources" / "api_football" / "normalized" / "fixtures_master.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the covered upcoming fixture universe for the active publish window."
    )
    parser.add_argument(
        "--src",
        default="",
        help=(
            "Optional current-window ALLMARKETS or DEPLOY CSV path. DEPLOY paths automatically resolve to the sibling "
            "ALLMARKETS source and routed tier files."
        ),
    )
    parser.add_argument(
        "--logo-manifest",
        default=str(DEFAULT_LOGO_MANIFEST),
        help="Optional API-Football logo manifest CSV. Use an empty string to disable logo enrichment.",
    )
    return parser.parse_args()


def resolve_source_path(src: str | None) -> Path:
    if src:
        candidate = Path(src)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        candidate = candidate.resolve()
        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError(f"Explicit --src file not found: {candidate}")
        return candidate
    runs = sorted(
        (path for path in (ROOT / "predictions_output").iterdir() if path.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", path.name)),
        key=lambda path: path.name,
    )
    if not runs:
        raise FileNotFoundError("No dated predictions_output directories found.")
    latest_dir = runs[-1]
    matches = sorted(latest_dir.glob("BOOKIE_*ALLMARKETS_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No ALLMARKETS CSV found under {latest_dir}")
    return matches[-1]


def resolve_allmarkets_source(path: Path) -> Path:
    name = path.name
    if "__DEPLOY_" in name:
        candidate = path.with_name(name.split("__DEPLOY_")[0] + ".csv")
        if candidate.exists():
            return candidate
    return path


def resolve_related_paths(allmarkets_path: Path) -> dict[str, Path]:
    stem = allmarkets_path.name[:-4] if allmarkets_path.name.endswith(".csv") else allmarkets_path.name
    base_dir = allmarkets_path.parent
    window = extract_window_from_name(allmarkets_path.name)

    def find_tier_file(tier: str) -> Path:
        matches = sorted(base_dir.glob(f"{stem}__DEPLOY_TIER_{tier}__*.csv"))
        return matches[-1] if matches else base_dir / f"{stem}__DEPLOY_TIER_{tier}__.csv"

    return {
        "allmarkets": allmarkets_path,
        "deploy_candidates_raw": base_dir / "DEPLOY_CANDIDATES_RAW.csv",
        "deploy_candidates_after_gates": base_dir / "DEPLOY_CANDIDATES_AFTER_GATES.csv",
        "deploy_elite": find_tier_file("ELITE"),
        "deploy_standard": find_tier_file("STANDARD"),
        "deploy_observe": find_tier_file("OBSERVE"),
        "loss_details": base_dir / f"PRE_ALLMARKETS_FIXTURE_LOSS_DETAILS_{window['date_from']}_to_{window['date_to']}.csv",
        "loss_report": base_dir / f"PRE_ALLMARKETS_FIXTURE_LOSS_REPORT_{window['date_from']}_to_{window['date_to']}.csv",
    }


def extract_window_from_name(name: str) -> dict[str, str]:
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", name)
    if len(dates) >= 2:
        return {"date_from": dates[0], "date_to": dates[1]}
    return {"date_from": "", "date_to": ""}


def parse_source_run_id(path: Path) -> str:
    for part in path.parts[::-1]:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", part):
            return part
    return path.parent.name


def normalize_kickoff(match_date: str) -> str:
    text = str(match_date or "").strip()
    if not text:
        return ""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return f"{text}T00:00:00Z"
    if text.endswith("Z"):
        return text
    return f"{text}Z" if "T" in text else text


def fixture_record_key(league: str, match_date: str, home_team: str, away_team: str) -> tuple[str, str, str, str]:
    return (
        normalize_lookup_text(league),
        str(match_date or "").strip(),
        normalize_lookup_text(home_team),
        normalize_lookup_text(away_team),
    )


def load_fixtures_master_index() -> dict[tuple[str, str, str, str], dict[str, str]]:
    if not FIXTURES_MASTER_PATH.exists():
        return {}
    with FIXTURES_MASTER_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        index: dict[tuple[str, str, str, str], dict[str, str]] = {}
        for row in reader:
            key = fixture_record_key(
                str(row.get("league", "") or ""),
                str(row.get("match_date", "") or ""),
                str(row.get("home_team_name", "") or ""),
                str(row.get("away_team_name", "") or ""),
            )
            index.setdefault(key, row)
        return index


def read_fixture_keys(path: Path, key_fields: tuple[str, str, str, str] = ("league", "match_date", "home_team_name", "away_team_name")) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    rows = load_rows(path)
    keyed: dict[str, dict[str, str]] = {}
    for row in rows:
        fixture_key = str(row.get("fixture_key", "") or "").strip()
        if not fixture_key:
            continue
        keyed.setdefault(fixture_key, row)
    return keyed


def build_base_record(
    fixture_key: str,
    league: str,
    match_date: str,
    home_team: str,
    away_team: str,
    identity_source: str,
    logo_index: dict[str, Any],
    counters: Counter[str],
    fixtures_master_index: dict[tuple[str, str, str, str], dict[str, str]],
) -> dict[str, Any]:
    kickoff_time = normalize_kickoff(match_date)
    master_match = fixtures_master_index.get(fixture_record_key(league, match_date, home_team, away_team))
    logo_assets = logo_assets_for(logo_index, league, home_team, away_team, counters)
    fixture_id = str(master_match.get("fixture_id", "") or "").strip() if master_match else ""
    return {
        "fixture_id": fixture_id or fixture_key,
        "fixture_key": fixture_key,
        "kickoff_time": kickoff_time,
        "league": league,
        "home_team": home_team,
        "away_team": away_team,
        **logo_assets,
        "coverage_status": "covered",
        "routing_status": "non_routed",
        "identity_source": identity_source,
        "source_availability": {
            "fixtures_master": bool(master_match),
            "routed_deploy": False,
            "routed_observe": False,
            "goal_shape_base": False,
            "prematch_odds": False,
            "injuries": False,
            "lineups": False,
            "team_stats": False,
            "player_stats": False,
            "match_events": False,
        },
        "follow_candidates": {
            "team_follow_candidate": True,
            "fixture_follow_candidate": True,
            "league_follow_candidate": True,
        },
        "updated_at": utc_now_iso(),
    }


def mark_allmarkets_availability(record: dict[str, Any], row: dict[str, str]) -> None:
    record["source_availability"]["goal_shape_base"] = True
    odds_fields = [
        "bookie_od",
        "od_home",
        "od_draw",
        "od_away",
        "od_yes",
        "od_no",
        "od_over",
        "od_under",
        "odds_ft_over25",
        "odds_ft_under25",
    ]
    if any(str(row.get(field, "") or "").strip() for field in odds_fields):
        record["source_availability"]["prematch_odds"] = True


def load_loss_detail_fixture_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    rows = load_rows(path)
    fixtures: dict[str, dict[str, str]] = {}
    for row in rows:
        if str(row.get("allmarkets_status", "") or "").strip().upper() != "NOT_EMITTED":
            continue
        fixture_key = str(row.get("fixture_key", "") or "").strip()
        if fixture_key:
            fixtures.setdefault(fixture_key, row)
    return fixtures


def build_report(
    source_paths: dict[str, Path],
    payload: dict[str, Any],
    counters: Counter[str],
) -> str:
    summary = payload["coverage_summary"]
    fallback_lines = "\n".join(f"- `{key}`: `{value}`" for key, value in sorted(counters.items()) if value)
    if not fallback_lines:
        fallback_lines = "- none"
    return "\n".join(
        [
            "# COVERED_FIXTURE_UNIVERSE_REPORT",
            "",
            f"Generated: `{payload['generated_at']}`",
            f"Source run id: `{payload['source_run_id']}`",
            f"Source window: `{payload['source_window']['date_from']}` to `{payload['source_window']['date_to']}`",
            "",
            "## Source Files",
            *(f"- `{label}`: `{path.relative_to(ROOT)}`" for label, path in source_paths.items() if path.exists()),
            "",
            "## Coverage Summary",
            f"- Total fixtures: `{summary['total_fixtures']}`",
            f"- Routed fixtures: `{summary['routed_count']}`",
            f"- Non-routed fixtures: `{summary['non_routed_count']}`",
            f"- Hidden fixtures: `{summary['hidden_count']}`",
            f"- Covered leagues: `{summary['covered_leagues_count']}`",
            "",
            "## Availability Counters",
            fallback_lines,
            "",
            "## Notes",
            "- Builder uses the current-window ALLMARKETS file as the primary live fixture intake base.",
            "- The pre-ALLMARKETS loss detail report fills upstream fixtures that never emitted into ALLMARKETS.",
            "- Historical normalized API-Football fixture master is joined opportunistically when identity matches exist.",
            "- This builder is an intake artifact for later CONTEXT and MONITOR classification.",
        ]
    )


def main() -> int:
    args = parse_args()
    ensure_dirs()
    counters: Counter[str] = Counter()

    source_path = resolve_source_path(str(args.src or "").strip() or None)
    allmarkets_path = resolve_allmarkets_source(source_path)
    if not allmarkets_path.exists():
        raise FileNotFoundError(f"Could not resolve ALLMARKETS source from {source_path}")

    related_paths = resolve_related_paths(allmarkets_path)
    source_window = extract_window_from_name(allmarkets_path.name)
    source_run_id = parse_source_run_id(allmarkets_path)

    logo_manifest_text = str(args.logo_manifest or "").strip()
    logo_manifest_path = (
        (ROOT / logo_manifest_text).resolve()
        if logo_manifest_text and not Path(logo_manifest_text).is_absolute()
        else Path(logo_manifest_text)
        if logo_manifest_text
        else None
    )
    logo_index = load_logo_manifest(logo_manifest_path, counters)
    fixtures_master_index = load_fixtures_master_index()

    allmarkets_rows = load_rows(related_paths["allmarkets"])
    counters["source_rows:allmarkets"] = len(allmarkets_rows)
    universe: dict[str, dict[str, Any]] = {}

    for row in allmarkets_rows:
        fixture_key = str(row.get("fixture_key", "") or "").strip()
        league = str(row.get("league", "") or "").strip()
        match_date = str(row.get("match_date", "") or "").strip()
        home_team = str(row.get("home_team_name", "") or "").strip()
        away_team = str(row.get("away_team_name", "") or "").strip()
        if not all([fixture_key, league, match_date, home_team, away_team]):
            counters["dropped:allmarkets_missing_identity"] += 1
            continue
        record = universe.get(fixture_key)
        if record is None:
            record = build_base_record(
                fixture_key,
                league,
                match_date,
                home_team,
                away_team,
                "allmarkets",
                logo_index,
                counters,
                fixtures_master_index,
            )
            universe[fixture_key] = record
        mark_allmarkets_availability(record, row)

    raw_candidates = read_fixture_keys(related_paths["deploy_candidates_raw"])
    after_gates = read_fixture_keys(related_paths["deploy_candidates_after_gates"])
    counters["source_rows:deploy_candidates_raw"] = len(raw_candidates)
    counters["source_rows:deploy_candidates_after_gates"] = len(after_gates)
    for fixture_key, row in {**raw_candidates, **after_gates}.items():
        record = universe.get(fixture_key)
        if record is not None:
            record["source_availability"]["goal_shape_base"] = True

    deploy_keys = set()
    observe_keys = set()
    for label in ("deploy_elite", "deploy_standard"):
        path = related_paths[label]
        if not path.exists():
            continue
        rows = read_fixture_keys(path)
        counters[f"source_rows:{label}"] = len(rows)
        deploy_keys.update(rows.keys())
    if related_paths["deploy_observe"].exists():
        observe_rows = read_fixture_keys(related_paths["deploy_observe"])
        counters["source_rows:deploy_observe"] = len(observe_rows)
        observe_keys.update(observe_rows.keys())

    for fixture_key, record in universe.items():
        if fixture_key in deploy_keys:
            record["source_availability"]["routed_deploy"] = True
            record["routing_status"] = "routed"
        if fixture_key in observe_keys:
            record["source_availability"]["routed_observe"] = True
            if record["routing_status"] != "routed":
                record["routing_status"] = "routed"

    loss_detail_rows = load_loss_detail_fixture_rows(related_paths["loss_details"])
    counters["source_rows:loss_details_not_emitted"] = len(loss_detail_rows)
    for fixture_key, row in loss_detail_rows.items():
        if fixture_key in universe:
            counters["loss_details:fixture_already_in_universe"] += 1
            continue
        league = str(row.get("league", "") or "").strip()
        match_date = str(row.get("match_date", "") or "").strip()
        home_team = str(row.get("home_team_name", "") or "").strip()
        away_team = str(row.get("away_team_name", "") or "").strip()
        if not all([fixture_key, league, match_date, home_team, away_team]):
            counters["dropped:loss_details_missing_identity"] += 1
            continue
        record = build_base_record(
            fixture_key,
            league,
            match_date,
            home_team,
            away_team,
            "pre_allmarkets_loss_details",
            logo_index,
            counters,
            fixtures_master_index,
        )
        universe[fixture_key] = record
        counters["loss_details:fixture_added_to_universe"] += 1

    fixtures = sorted(
        universe.values(),
        key=lambda record: (record["kickoff_time"], record["league"], record["home_team"], record["away_team"]),
    )

    for record in fixtures:
        counters[f"routing_status:{record['routing_status']}"] += 1
        for key, value in record["source_availability"].items():
            if value:
                counters[f"availability:{key}"] += 1

    payload = {
        "generated_at": utc_now_iso(),
        "source_run_id": source_run_id,
        "source_window": source_window,
        "coverage_summary": {
            "total_fixtures": len(fixtures),
            "routed_count": sum(1 for record in fixtures if record["routing_status"] == "routed"),
            "non_routed_count": sum(1 for record in fixtures if record["routing_status"] == "non_routed"),
            "hidden_count": 0,
            "covered_leagues_count": len({record["league"] for record in fixtures}),
        },
        "fixtures": fixtures,
    }

    write_json(OUTPUT_PATH, payload)
    REPORT_PATH.write_text(build_report(related_paths, payload, counters) + "\n", encoding="utf-8")

    print(f"Covered fixture universe written: {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"Covered fixture universe report written: {REPORT_PATH.relative_to(ROOT)}")
    print(f"Total fixtures: {payload['coverage_summary']['total_fixtures']}")
    print(f"Routed fixtures: {payload['coverage_summary']['routed_count']}")
    print(f"Non-routed fixtures: {payload['coverage_summary']['non_routed_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
