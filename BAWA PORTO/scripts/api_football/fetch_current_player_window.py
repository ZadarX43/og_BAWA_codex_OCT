from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.api_football.client import APIFootballClient
from scripts.api_football.normalize_fixtures_master import build_fixtures_master
from scripts.api_football.normalize_lineups import build_lineups
from scripts.api_football.normalize_match_player_stats import build_match_player_stats
from scripts.api_football.utils import chunk_list


DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-07" / "api_current_player_window"

# Season is the API-Football season label. European 2025/26 competitions use
# 2025; calendar-year leagues use 2026.
DEFAULT_LEAGUES = {
    "Australia_A_League": {"league_id": 188, "season": 2025, "start": "2025-07-01"},
    "Austria_Bundesliga": {"league_id": 218, "season": 2025, "start": "2025-07-01"},
    "Belgium_Pro": {"league_id": 144, "season": 2025, "start": "2025-07-01"},
    "Brazil_Serie_A": {"league_id": 71, "season": 2026, "start": "2026-01-01"},
    "Denmark_Superliga": {"league_id": 119, "season": 2025, "start": "2025-07-01"},
    "England_Championship": {"league_id": 40, "season": 2025, "start": "2025-07-01"},
    "England_EFL_League_1": {"league_id": 41, "season": 2025, "start": "2025-07-01"},
    "England_Premier_League": {"league_id": 39, "season": 2025, "start": "2025-07-01"},
    "France_Ligue_1": {"league_id": 61, "season": 2025, "start": "2025-07-01"},
    "Germany_Bundesliga": {"league_id": 78, "season": 2025, "start": "2025-07-01"},
    "Germany_Bundesliga_2": {"league_id": 79, "season": 2025, "start": "2025-07-01"},
    "Italy_Serie_A": {"league_id": 135, "season": 2025, "start": "2025-07-01"},
    "Netherlands_Eredivisie": {"league_id": 88, "season": 2025, "start": "2025-07-01"},
    "Norway_Eliteserien": {"league_id": 103, "season": 2026, "start": "2026-01-01"},
    "Portugal_Liga": {"league_id": 94, "season": 2025, "start": "2025-07-01"},
    "Saudi_Pro_League": {"league_id": 307, "season": 2025, "start": "2025-07-01"},
    "Scotland_Premiership": {"league_id": 179, "season": 2025, "start": "2025-07-01"},
    "South_Korea_K_League": {"league_id": 292, "season": 2026, "start": "2026-01-01"},
    "Spain_La_Liga": {"league_id": 140, "season": 2025, "start": "2025-07-01"},
    "Switzerland_Super_League": {"league_id": 207, "season": 2025, "start": "2025-07-01"},
    "Turkey_Super_Lig": {"league_id": 203, "season": 2025, "start": "2025-07-01"},
    "USA_MLS": {"league_id": 253, "season": 2026, "start": "2026-01-01"},
}


def parse_csv_set(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")


def fixture_ids(payload: dict[str, Any]) -> list[int]:
    out: list[int] = []
    for item in payload.get("response", []) or []:
        fid = ((item.get("fixture") or {}).get("id"))
        if fid is not None:
            out.append(int(fid))
    return out


def clean_status(value: str) -> str | None:
    text = str(value or "").strip()
    return text or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch a safe API-Football current player-stat/lineup window into reports/ "
            "and normalize it without touching production normalized season files."
        )
    )
    parser.add_argument("--league-tags", default="", help="Comma-separated tags. Defaults to the 14 current-board leagues.")
    parser.add_argument("--from-date", default="", help="Override lower bound YYYY-MM-DD for every league.")
    parser.add_argument("--to-date", default=date.today().isoformat(), help="Upper bound YYYY-MM-DD.")
    parser.add_argument("--status", default="FT-AET-PEN", help="Fixture status filter. Empty string fetches all statuses.")
    parser.add_argument("--timezone", default="Europe/London")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--sleep-seconds", type=float, default=None)
    parser.add_argument("--daily-cap", type=int, default=75000)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--max-fixtures-per-league", type=int, default=0, help="Optional safety cap after fixture fetch.")
    args = parser.parse_args()

    selected = parse_csv_set(args.league_tags) if args.league_tags else set(DEFAULT_LEAGUES)
    unknown = selected - set(DEFAULT_LEAGUES)
    if unknown:
        raise SystemExit(f"Unknown league tags: {sorted(unknown)}")

    raw_dir = args.outdir / "raw"
    normalized_dir = args.outdir / "normalized"
    raw_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)

    client = APIFootballClient(sleep_seconds=args.sleep_seconds, daily_cap=args.daily_cap)
    summary_rows: list[dict[str, Any]] = []
    for tag in sorted(selected):
        cfg = DEFAULT_LEAGUES[tag]
        league_id = int(cfg["league_id"])
        season = int(cfg["season"])
        from_date = args.from_date or str(cfg["start"])
        to_date = args.to_date
        stem = f"{tag}__league_{league_id}__season_{season}__{from_date}_to_{to_date}"

        params = {
            "league": league_id,
            "season": season,
            "from": from_date,
            "to": to_date,
            "status": clean_status(args.status),
            "timezone": args.timezone,
        }
        fixtures_payload = client.get_json("/fixtures", params)
        ids = fixture_ids(fixtures_payload)
        if args.max_fixtures_per_league and len(ids) > args.max_fixtures_per_league:
            ids = ids[: args.max_fixtures_per_league]
        fixtures_raw = raw_dir / f"{stem}__fixtures.jsonl"
        write_jsonl(fixtures_raw, [fixtures_payload])

        bundle_payloads: list[dict[str, Any]] = []
        for fixture_chunk in chunk_list(ids, max(1, min(int(args.chunk_size), 20))):
            ids_param = "-".join(str(fid) for fid in fixture_chunk)
            bundle_payloads.append(client.get_json("/fixtures", {"ids": ids_param}))
        bundle_raw = raw_dir / f"{stem}__fixtures_bundle.jsonl"
        write_jsonl(bundle_raw, bundle_payloads)

        fixtures_out = normalized_dir / f"fixtures_master__{tag}__{season}.csv"
        player_out = normalized_dir / f"match_player_stats__{tag}__{season}.csv"
        lineups_out = normalized_dir / f"lineups__{tag}__{season}.csv"
        fixtures_df = build_fixtures_master(str(fixtures_raw), str(fixtures_out))
        player_df = build_match_player_stats(str(bundle_raw), str(player_out))
        lineups_df = build_lineups(str(bundle_raw), str(lineups_out))

        summary_rows.append(
            {
                "league_tag": tag,
                "league_id": league_id,
                "season": season,
                "from_date": from_date,
                "to_date": to_date,
                "fixture_ids": len(ids),
                "bundle_requests": len(bundle_payloads),
                "fixtures_rows": len(fixtures_df),
                "player_stat_rows": len(player_df),
                "lineup_rows": len(lineups_df),
                "fixtures_raw": str(fixtures_raw),
                "bundle_raw": str(bundle_raw),
                "player_stats_csv": str(player_out),
                "lineups_csv": str(lineups_out),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.outdir / "API_CURRENT_PLAYER_WINDOW_SUMMARY.csv", index=False)
    print(f"WROTE {args.outdir}")
    print(summary[["league_tag", "season", "fixture_ids", "bundle_requests", "player_stat_rows", "lineup_rows"]].to_string(index=False))


if __name__ == "__main__":
    main()
