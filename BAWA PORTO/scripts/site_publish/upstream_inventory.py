#!/usr/bin/env python3
"""Inventory upstream inputs for the website publish pipeline.

This is a read-only preflight. It does not run models, deploy routing, or data
refresh. It tells the site publish orchestrator whether the expected upstream
artifacts are present and whether the local compact site DB covers the target
fixture window.
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


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "frontend" / "public" / "data"
DEFAULT_DB = ROOT / "build" / "site_data" / "odds_genius.sqlite"
DEFAULT_PREDICTIONS_ROOT = ROOT / "predictions_output"
DEFAULT_NORMALIZED_DIR = ROOT / "data_sources" / "api_football" / "normalized"
DEFAULT_JSON_OUT = ROOT / "reports" / "latest" / "SITE_UPSTREAM_INVENTORY.json"
DEFAULT_MD_OUT = ROOT / "reports" / "latest" / "SITE_UPSTREAM_INVENTORY.md"
API_FAMILIES = (
    "fixtures_master",
    "match_team_stats",
    "match_player_stats",
    "match_events",
    "lineups",
    "injuries",
    "sidelined",
    "odds_prematch_long",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory website publish upstream artifacts.")
    parser.add_argument("--from-date", default=date.today().isoformat())
    parser.add_argument("--to-date", default="")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--predictions-root", type=Path, default=DEFAULT_PREDICTIONS_ROOT)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_day(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text[:10]).date()
    except ValueError:
        return None


def mtime_utc(path: Path) -> str:
    if not path.exists():
        return ""
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def file_status(path: Path) -> dict[str, Any]:
    exists = path.exists()
    stat = path.stat() if exists else None
    return {
        "path": str(path),
        "exists": exists,
        "bytes": stat.st_size if stat else 0,
        "mtime_utc": mtime_utc(path) if exists else "",
    }


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def read_csv_count(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            return sum(1 for _ in reader)
    except (OSError, UnicodeDecodeError, csv.Error):
        return 0


def count_json_items(path: Path) -> int:
    payload = read_json(path, [])
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        items = payload.get("items")
        if isinstance(items, list):
            return len(items)
    return 0


def date_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    return sorted((path for path in root.iterdir() if path.is_dir() and pattern.match(path.name)), key=lambda path: path.name)


def prediction_outputs(root: Path, start: date, end: date) -> dict[str, Any]:
    publish_summary = read_json(DATA_ROOT / "publish_summary.json", {})
    selected_source = publish_summary.get("selected_source_csv") if isinstance(publish_summary, dict) else ""
    selected_path = ROOT / selected_source if selected_source else None
    dirs = date_dirs(root)
    window_dirs = [path for path in dirs if (day := parse_day(path.name)) and start <= day <= end]
    latest_dirs = dirs[-7:]
    csvs: list[Path] = []
    for folder in window_dirs:
        csvs.extend(sorted(folder.glob("*.csv")))
    latest_csvs: list[Path] = []
    for folder in latest_dirs:
        latest_csvs.extend(sorted(folder.glob("*.csv")))
    return {
        "root": str(root),
        "selected_source": file_status(selected_path) if selected_path else {"exists": False, "path": "", "bytes": 0, "mtime_utc": ""},
        "selected_source_rows": read_csv_count(selected_path) if selected_path and selected_path.exists() else 0,
        "window_dirs": [str(path.relative_to(ROOT)) for path in window_dirs],
        "window_csv_count": len(csvs),
        "window_csvs": [
            {**file_status(path), "rows": read_csv_count(path), "relative_path": str(path.relative_to(ROOT))}
            for path in csvs[:40]
        ],
        "latest_dirs": [str(path.relative_to(ROOT)) for path in latest_dirs],
        "latest_csv_count": len(latest_csvs),
        "latest_csvs": [
            {**file_status(path), "rows": read_csv_count(path), "relative_path": str(path.relative_to(ROOT))}
            for path in latest_csvs[-40:]
        ],
    }


def api_family_inventory(normalized_dir: Path, start: date, end: date) -> dict[str, Any]:
    families: dict[str, Any] = {}
    for family in API_FAMILIES:
        files = sorted(normalized_dir.glob(f"{family}__*.csv"))
        mtimes = [path.stat().st_mtime for path in files if path.exists()]
        families[family] = {
            "file_count": len(files),
            "latest_mtime_utc": datetime.fromtimestamp(max(mtimes), timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            if mtimes
            else "",
            "latest_files": [str(path.relative_to(ROOT)) for path in sorted(files, key=lambda p: p.stat().st_mtime if p.exists() else 0)[-8:]],
        }
    fixture_window_rows = 0
    fixture_window_leagues: Counter[str] = Counter()
    for path in normalized_dir.glob("fixtures_master__*.csv"):
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    row_day = parse_day(row.get("match_date") or row.get("kickoff_ts_utc") or row.get("date"))
                    if row_day and start <= row_day <= end:
                        fixture_window_rows += 1
                        league = path.name.replace("fixtures_master__", "").rsplit("__", 1)[0]
                        fixture_window_leagues[league] += 1
        except (OSError, UnicodeDecodeError, csv.Error):
            continue
    return {
        "normalized_dir": str(normalized_dir),
        "families": families,
        "fixture_window_rows": fixture_window_rows,
        "fixture_window_leagues": dict(sorted(fixture_window_leagues.items())),
    }


def site_db_inventory(db_path: Path, start: date, end: date) -> dict[str, Any]:
    if not db_path.exists():
        return {"exists": False, "path": str(db_path)}
    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    metadata = {row["key"]: row["value"] for row in conn.execute("SELECT key, value FROM metadata")} if table_exists(conn, "metadata") else {}
    fixtures = []
    if table_exists(conn, "fixtures"):
        fixtures = list(conn.execute("SELECT fixture_key, kickoff_time, league, home_team, away_team, publish_class, coverage_status FROM fixtures"))
    dates = [parse_day(row["kickoff_time"]) for row in fixtures]
    valid_dates = [day for day in dates if day]
    window_fixtures = [row for row in fixtures if (day := parse_day(row["kickoff_time"])) and start <= day <= end]
    publish_counts = Counter(str(row["publish_class"] or "") for row in window_fixtures)
    league_counts = Counter(str(row["league"] or "") for row in window_fixtures)
    table_counts: dict[str, int] = {}
    for table in ("fixtures", "fixture_decisions", "fixture_h2h", "fixture_lineups", "site_fixture_stats_payloads", "team_intelligence", "site_team_premium_payloads"):
        if table_exists(conn, table):
            table_counts[table] = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    return {
        "exists": True,
        "path": str(db_path),
        "bytes": db_path.stat().st_size,
        "mtime_utc": mtime_utc(db_path),
        "metadata": metadata,
        "table_counts": table_counts,
        "fixture_count": len(fixtures),
        "fixture_date_min": min(valid_dates).isoformat() if valid_dates else "",
        "fixture_date_max": max(valid_dates).isoformat() if valid_dates else "",
        "window_fixture_count": len(window_fixtures),
        "window_publish_counts": dict(publish_counts),
        "window_league_counts": dict(sorted(league_counts.items())),
        "window_sample": [dict(row) for row in window_fixtures[:20]],
    }


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    return row is not None


def website_payload_inventory() -> dict[str, Any]:
    files = {
        "publish_summary": DATA_ROOT / "publish_summary.json",
        "public_predictions": DATA_ROOT / "public_predictions.json",
        "premium_predictions": DATA_ROOT / "premium_predictions.json",
        "weekly_results": DATA_ROOT / "weekly_results.json",
        "results_archive": DATA_ROOT / "results_archive.json",
        "live_results_feed": DATA_ROOT / "live_results_feed.json",
        "logo_manifest": DATA_ROOT / "api_football_logo_asset_manifest.json",
    }
    payloads = {name: {**file_status(path), "items": count_json_items(path)} for name, path in files.items()}
    static_dirs = {
        "fixture_decision_intelligence": DATA_ROOT / "fixture_decision_intelligence",
        "fixture_h2h_support": DATA_ROOT / "fixture_h2h_support",
        "fixture_lineup_intelligence": DATA_ROOT / "fixture_lineup_intelligence",
        "team_intelligence": DATA_ROOT / "team_intelligence",
        "player_intelligence": DATA_ROOT / "player_intelligence",
    }
    return {
        "payloads": payloads,
        "static_dirs": {
            name: {
                "path": str(path),
                "exists": path.exists(),
                "json_file_count": len(list(path.rglob("*.json"))) if path.exists() else 0,
            }
            for name, path in static_dirs.items()
        },
    }


def readiness(predictions: dict[str, Any], api: dict[str, Any], site_db: dict[str, Any], website: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    if not predictions.get("selected_source", {}).get("exists"):
        blockers.append("selected_model_source_missing")
    if site_db.get("window_fixture_count", 0) == 0:
        blockers.append("site_db_window_empty")
    if predictions.get("window_csv_count", 0) == 0:
        warnings.append("no_prediction_csvs_in_target_window")
    if api.get("fixture_window_rows", 0) == 0:
        warnings.append("api_football_fixture_window_empty")
    payloads = website.get("payloads") or {}
    for name in ("public_predictions", "premium_predictions", "publish_summary"):
        if not payloads.get(name, {}).get("exists"):
            blockers.append(f"{name}_missing")
    return {
        "state": "ready" if not blockers else "waiting_on_fresh_window" if blockers == ["site_db_window_empty"] else "blocked",
        "blockers": blockers,
        "warnings": warnings,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ready = payload["readiness"]
    site_db = payload["site_db"]
    predictions = payload["prediction_outputs"]
    api = payload["api_football"]
    lines = [
        "# Site Upstream Inventory",
        "",
        f"- Generated: `{payload['generated_at']}`",
        f"- Window: `{payload['window']['from']}` to `{payload['window']['to']}`",
        f"- Readiness: `{ready['state']}`",
        f"- Blockers: `{', '.join(ready['blockers']) or 'none'}`",
        f"- Warnings: `{', '.join(ready['warnings']) or 'none'}`",
        "",
        "## Model Outputs",
        "",
        f"- Selected source exists: `{predictions['selected_source']['exists']}`",
        f"- Selected source rows: `{predictions.get('selected_source_rows', 0)}`",
        f"- Target-window prediction CSVs: `{predictions.get('window_csv_count', 0)}`",
        "",
        "## API-Football",
        "",
        f"- Fixture rows in target window: `{api.get('fixture_window_rows', 0)}`",
        f"- Window leagues: `{len(api.get('fixture_window_leagues', {}))}`",
        "",
        "## Local Site DB",
        "",
        f"- DB exists: `{site_db.get('exists', False)}`",
        f"- Fixture date range: `{site_db.get('fixture_date_min', '')}` to `{site_db.get('fixture_date_max', '')}`",
        f"- Fixtures in target window: `{site_db.get('window_fixture_count', 0)}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    start = parse_day(args.from_date) or date.today()
    end = parse_day(args.to_date) if args.to_date else start + timedelta(days=args.days)
    if end is None:
        raise SystemExit(f"Invalid --to-date: {args.to_date}")
    db_path = resolve(args.db)
    predictions_root = resolve(args.predictions_root)
    normalized_dir = resolve(args.normalized_dir)
    predictions = prediction_outputs(predictions_root, start, end)
    api = api_family_inventory(normalized_dir, start, end)
    site_db = site_db_inventory(db_path, start, end)
    website = website_payload_inventory()
    payload = {
        "schema": "site_upstream_inventory_v1",
        "generated_at": utc_now(),
        "window": {"from": start.isoformat(), "to": end.isoformat()},
        "prediction_outputs": predictions,
        "api_football": api,
        "site_db": site_db,
        "website_payloads": website,
        "readiness": readiness(predictions, api, site_db, website),
    }
    json_out = resolve(args.json_out)
    md_out = resolve(args.md_out)
    write_json(json_out, payload)
    write_markdown(md_out, payload)
    print(
        json.dumps(
            {
                "readiness": payload["readiness"],
                "site_db_window_fixture_count": site_db.get("window_fixture_count", 0),
                "prediction_window_csv_count": predictions.get("window_csv_count", 0),
                "api_fixture_window_rows": api.get("fixture_window_rows", 0),
                "json_out": str(json_out),
                "md_out": str(md_out),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["readiness"]["state"] != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
