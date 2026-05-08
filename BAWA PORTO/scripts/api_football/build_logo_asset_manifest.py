#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data_sources" / "api_football" / "raw"
NORMALIZED_DIR = ROOT / "data_sources" / "api_football" / "normalized"
FRONTEND_DATA_DIR = ROOT / "frontend" / "public" / "data"

DEFAULT_CSV = NORMALIZED_DIR / "api_football_logo_asset_manifest.csv"
DEFAULT_JSON = FRONTEND_DATA_DIR / "api_football_logo_asset_manifest.json"

CSV_FIELDS = [
    "asset_type",
    "league_id",
    "league_name",
    "league_country",
    "league_logo_url",
    "league_flag_url",
    "season",
    "team_id",
    "team_name",
    "team_logo_url",
    "first_seen_fixture_id",
    "appearances",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def iter_fixture_payloads(raw_dir: Path) -> tuple[list[dict[str, Any]], int]:
    payloads: list[dict[str, Any]] = []
    files = sorted(raw_dir.glob("fixtures__league_*__season_*__fixtures.jsonl"))
    for path in files:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for item in payload.get("response", []) or []:
                    if isinstance(item, dict):
                        payloads.append(item)
    return payloads, len(files)


def safe_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def build_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    league_rows: dict[tuple[int, int], dict[str, Any]] = {}
    team_rows: dict[tuple[int, int, int], dict[str, Any]] = {}
    team_counts: Counter[tuple[int, int, int]] = Counter()
    league_counts: Counter[tuple[int, int]] = Counter()

    for item in items:
        fixture = item.get("fixture") or {}
        league = item.get("league") or {}
        teams = item.get("teams") or {}
        league_id = safe_int(league.get("id"))
        season = safe_int(league.get("season"))
        fixture_id = safe_int(fixture.get("id"))
        league_key = (league_id, season)

        league_counts[league_key] += 1
        if league_key not in league_rows:
            league_rows[league_key] = {
                "asset_type": "league",
                "league_id": league_id,
                "league_name": league.get("name") or "",
                "league_country": league.get("country") or "",
                "league_logo_url": league.get("logo") or "",
                "league_flag_url": league.get("flag") or "",
                "season": season,
                "team_id": "",
                "team_name": "",
                "team_logo_url": "",
                "first_seen_fixture_id": fixture_id,
                "appearances": 0,
            }

        for side in ("home", "away"):
            team = (teams.get(side) or {}) if isinstance(teams, dict) else {}
            team_id = safe_int(team.get("id"))
            if not team_id:
                continue
            team_key = (league_id, season, team_id)
            team_counts[team_key] += 1
            if team_key not in team_rows:
                team_rows[team_key] = {
                    "asset_type": "team",
                    "league_id": league_id,
                    "league_name": league.get("name") or "",
                    "league_country": league.get("country") or "",
                    "league_logo_url": league.get("logo") or "",
                    "league_flag_url": league.get("flag") or "",
                    "season": season,
                    "team_id": team_id,
                    "team_name": team.get("name") or "",
                    "team_logo_url": team.get("logo") or "",
                    "first_seen_fixture_id": fixture_id,
                    "appearances": 0,
                }

    for key, count in league_counts.items():
        league_rows[key]["appearances"] = count
    for key, count in team_counts.items():
        team_rows[key]["appearances"] = count

    rows = list(league_rows.values()) + list(team_rows.values())
    return sorted(
        rows,
        key=lambda row: (
            str(row["asset_type"]),
            str(row["league_name"]),
            int(row["season"] or 0),
            str(row["team_name"]),
        ),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, Any]], source_files: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    leagues = [row for row in rows if row["asset_type"] == "league"]
    teams = [row for row in rows if row["asset_type"] == "team"]
    payload = {
        "generated_at": utc_now_iso(),
        "source": "existing_api_football_raw_fixture_jsonl",
        "source_files_count": source_files,
        "league_asset_count": len(leagues),
        "team_asset_count": len(teams),
        "leagues": leagues,
        "teams": teams,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a no-network API-Football logo/badge asset manifest from existing raw fixture payloads."
    )
    parser.add_argument("--raw-dir", default=str(RAW_DIR), help="Directory containing raw API-Football fixture JSONL files.")
    parser.add_argument("--output-csv", default=str(DEFAULT_CSV), help="CSV manifest output path.")
    parser.add_argument("--output-json", default=str(DEFAULT_JSON), help="Website JSON manifest output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw API-Football directory not found: {raw_dir}")

    items, source_files = iter_fixture_payloads(raw_dir)
    rows = build_rows(items)
    write_csv(Path(args.output_csv), rows)
    write_json(Path(args.output_json), rows, source_files)

    team_rows = [row for row in rows if row["asset_type"] == "team"]
    league_rows = [row for row in rows if row["asset_type"] == "league"]
    print(f"Raw fixture files scanned: {source_files}")
    print(f"Fixture rows read: {len(items)}")
    print(f"League assets written: {len(league_rows)}")
    print(f"Team assets written: {len(team_rows)}")
    print(f"CSV manifest: {Path(args.output_csv).relative_to(ROOT)}")
    print(f"JSON manifest: {Path(args.output_json).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
