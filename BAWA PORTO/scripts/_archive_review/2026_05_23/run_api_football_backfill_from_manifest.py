#!/usr/bin/env python3
"""Run reviewed API-Football backfill rows from a manifest.

Default mode is `--dry-run`; pass `--execute` only after reviewing the manifest
and confirming API budget. The script writes per-league/season raw, normalized,
and feature files using the existing API-Football modules.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


DEFAULT_MANIFEST = Path("reports/latest/api_football_2025_2026_backfill_manifest.csv")
RAW_DIR = Path("data_sources/api_football/raw")
NORMALIZED_DIR = Path("data_sources/api_football/normalized")
FEATURES_DIR = Path("data_sources/api_football/features")


def run(cmd: list[str], *, execute: bool, log: list[str]) -> None:
    text = " ".join(cmd)
    log.append(text)
    if execute:
        subprocess.run(cmd, check=True)
    else:
        print(text)


def module_cmd(module: str, *args: str) -> list[str]:
    return [sys.executable, "-m", f"scripts.api_football.{module}", *args]


def fixture_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            for item in payload.get("response", []) or []:
                fixture = item.get("fixture") or {}
                if fixture.get("id") is not None:
                    count += 1
    return count


def player_stat_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            for team_block in payload.get("response", []) or []:
                count += len(team_block.get("players", []) or [])
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--execute", action="store_true", help="Actually make API calls and write derived files.")
    parser.add_argument("--refresh-existing", action="store_true", help="Refetch raw files even when present.")
    parser.add_argument("--sleep-seconds", default="1.0")
    parser.add_argument("--daily-cap", default="75000")
    parser.add_argument("--limit-fixtures", default="", help="Optional fixture cap for smoke tests.")
    parser.add_argument("--log-file", default="reports/latest/api_football_backfill_commands.log")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    ready = manifest[manifest["manifest_status"].astype("string").eq("READY")].copy()
    if ready.empty:
        raise SystemExit("No READY manifest rows found.")

    log: list[str] = []
    for _, row in ready.iterrows():
        league_tag = str(row["league_tag"])
        league_id = str(int(row["league_id"]))
        season = str(int(row["season"]))
        stem = f"fixtures__league_{league_id}__season_{season}"
        fixtures_raw = RAW_DIR / f"{stem}__fixtures.jsonl"
        bundle_raw = RAW_DIR / f"{stem}__fixtures_bundle.jsonl"
        injuries_raw = RAW_DIR / f"{stem}__injuries.jsonl"
        players_raw = RAW_DIR / f"{stem}__fixtures_players.jsonl"

        fixtures_csv = NORMALIZED_DIR / f"fixtures_master__{league_tag}__{season}.csv"
        team_stats_csv = NORMALIZED_DIR / f"match_team_stats__{league_tag}__{season}.csv"
        events_csv = NORMALIZED_DIR / f"match_events__{league_tag}__{season}.csv"
        player_stats_csv = NORMALIZED_DIR / f"match_player_stats__{league_tag}__{season}.csv"
        lineups_csv = NORMALIZED_DIR / f"lineups__{league_tag}__{season}.csv"
        injuries_csv = NORMALIZED_DIR / f"injuries__{league_tag}__{season}.csv"

        team_features_csv = FEATURES_DIR / f"api_team_rolling_features__{league_tag}__{season}.csv"
        player_features_csv = FEATURES_DIR / f"api_player_rolling_features__{league_tag}__{season}.csv"
        lineup_features_csv = FEATURES_DIR / f"api_lineup_features__{league_tag}__{season}.csv"
        injury_features_csv = FEATURES_DIR / f"api_injury_features__{league_tag}__{season}.csv"
        event_features_csv = FEATURES_DIR / f"api_event_features__{league_tag}__{season}.csv"

        fetch_common = ["--sleep-seconds", args.sleep_seconds, "--daily-cap", args.daily_cap]
        raw_fixture_status = row.get("fixture_status", "FT-AET-PEN")
        fixture_status = "" if pd.isna(raw_fixture_status) else str(raw_fixture_status)
        raw_from_date = row.get("from_date", "")
        raw_to_date = row.get("to_date", "")
        from_date = "" if pd.isna(raw_from_date) else str(raw_from_date)
        to_date = "" if pd.isna(raw_to_date) else str(raw_to_date)
        fixtures_before = fixture_count(fixtures_raw)
        fixtures_refetched = False
        if args.refresh_existing or not fixtures_raw.exists() or fixtures_before == 0:
            fixture_cmd = module_cmd("fetch_fixtures", "--league-ids", league_id, "--season", season, *fetch_common)
            if fixture_status:
                fixture_cmd.extend(["--status", fixture_status])
            else:
                fixture_cmd.append("--all-statuses")
            if from_date:
                fixture_cmd.extend(["--from-date", from_date])
            if to_date:
                fixture_cmd.extend(["--to-date", to_date])
            run(fixture_cmd, execute=args.execute, log=log)
            fixtures_refetched = True
        fixtures_after = fixture_count(fixtures_raw) if args.execute else max(fixtures_before, 1)
        if fixtures_after == 0:
            msg = f"SKIP_NO_FIXTURES {league_tag} season={season} league_id={league_id} raw={fixtures_raw}"
            print(msg)
            log.append(msg)
            continue
        if args.refresh_existing or fixtures_refetched or not bundle_raw.exists():
            cmd = module_cmd("fetch_fixture_bundle", "--fixtures-raw", str(fixtures_raw), *fetch_common)
            if args.limit_fixtures:
                cmd.extend(["--limit", args.limit_fixtures])
            run(cmd, execute=args.execute, log=log)
        if args.refresh_existing or fixtures_refetched or not injuries_raw.exists():
            cmd = module_cmd("fetch_injuries", "--fixtures-raw", str(fixtures_raw), *fetch_common)
            if args.limit_fixtures:
                cmd.extend(["--limit", args.limit_fixtures])
            run(cmd, execute=args.execute, log=log)
        players_before = player_stat_count(players_raw)
        if args.refresh_existing or fixtures_refetched or not players_raw.exists() or players_before == 0:
            cmd = module_cmd("fetch_players", "--fixtures-raw", str(fixtures_raw), *fetch_common)
            if args.limit_fixtures:
                cmd.extend(["--limit", args.limit_fixtures])
            run(cmd, execute=args.execute, log=log)

        run(module_cmd("normalize_fixtures_master", "--fixtures-raw", str(fixtures_raw), "--output-csv", str(fixtures_csv)), execute=args.execute, log=log)
        run(module_cmd("normalize_match_team_stats", "--bundle-raw", str(bundle_raw), "--output-csv", str(team_stats_csv)), execute=args.execute, log=log)
        run(module_cmd("normalize_match_events", "--bundle-raw", str(bundle_raw), "--output-csv", str(events_csv)), execute=args.execute, log=log)
        run(module_cmd("normalize_match_player_stats", "--bundle-raw", str(players_raw), "--output-csv", str(player_stats_csv)), execute=args.execute, log=log)
        run(module_cmd("normalize_lineups", "--bundle-raw", str(bundle_raw), "--output-csv", str(lineups_csv)), execute=args.execute, log=log)
        run(module_cmd("normalize_injuries", "--injuries-raw", str(injuries_raw), "--output-csv", str(injuries_csv)), execute=args.execute, log=log)

        run(module_cmd("build_team_rolling_features", "--fixtures-csv", str(fixtures_csv), "--team-stats-csv", str(team_stats_csv), "--output-csv", str(team_features_csv)), execute=args.execute, log=log)
        run(module_cmd("build_player_rolling_features", "--fixtures-csv", str(fixtures_csv), "--player-stats-csv", str(player_stats_csv), "--output-csv", str(player_features_csv)), execute=args.execute, log=log)
        run(module_cmd("build_lineup_features", "--fixtures-csv", str(fixtures_csv), "--lineups-csv", str(lineups_csv), "--player-stats-csv", str(player_stats_csv), "--output-csv", str(lineup_features_csv)), execute=args.execute, log=log)
        run(module_cmd("build_injury_features", "--fixtures-csv", str(fixtures_csv), "--injuries-csv", str(injuries_csv), "--player-stats-csv", str(player_stats_csv), "--output-csv", str(injury_features_csv)), execute=args.execute, log=log)
        run(module_cmd("build_event_features", "--fixtures-csv", str(fixtures_csv), "--events-csv", str(events_csv), "--team-stats-csv", str(team_stats_csv), "--output-csv", str(event_features_csv)), execute=args.execute, log=log)

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(log) + "\n", encoding="utf-8")
    print(f"Commands: {len(log)}")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY_RUN'}")
    print(f"Log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
