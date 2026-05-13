from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.api_football.client import APIFootballClient
from scripts.api_football.fetch_current_player_window import write_jsonl
from scripts.api_football.normalize_fixtures_master import build_fixtures_master
from scripts.api_football.normalize_lineups import build_lineups
from scripts.api_football.normalize_match_events import build_match_events
from scripts.api_football.normalize_match_player_stats import build_match_player_stats
from scripts.api_football.normalize_match_team_stats import build_match_team_stats


DEFAULT_OUTDIR = ROOT / "reports" / "latest" / "demo_fixture_full_provider_bundle"


def response_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return list(payload.get("response") or [])


def build_full_fixture_item(
    fixture_item: dict[str, Any],
    lineups_payload: dict[str, Any],
    players_payload: dict[str, Any],
    statistics_payload: dict[str, Any],
    events_payload: dict[str, Any],
) -> dict[str, Any]:
    item = deepcopy(fixture_item)
    item["lineups"] = response_items(lineups_payload)
    item["players"] = response_items(players_payload)
    item["statistics"] = response_items(statistics_payload)
    item["events"] = response_items(events_payload)
    return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch one API-Football fixture with lineups, player stats, team stats, and events, "
            "then normalize it as a report-scoped demo bundle."
        )
    )
    parser.add_argument("--fixture-id", required=True, type=int)
    parser.add_argument("--league-tag", required=True)
    parser.add_argument("--season", required=True, type=int)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--daily-cap", type=int, default=75000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = args.outdir / "raw"
    normalized_dir = args.outdir / "normalized"
    raw_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)

    client = APIFootballClient(sleep_seconds=args.sleep_seconds, daily_cap=args.daily_cap)
    fixture_payload = client.get_json("/fixtures", {"id": args.fixture_id})
    fixture_items = response_items(fixture_payload)
    if not fixture_items:
        raise SystemExit(f"No provider fixture returned for fixture id {args.fixture_id}.")

    lineups_payload = client.get_json("/fixtures/lineups", {"fixture": args.fixture_id})
    players_payload = client.get_json("/fixtures/players", {"fixture": args.fixture_id})
    statistics_payload = client.get_json("/fixtures/statistics", {"fixture": args.fixture_id})
    events_payload = client.get_json("/fixtures/events", {"fixture": args.fixture_id})

    full_item = build_full_fixture_item(
        fixture_items[0],
        lineups_payload,
        players_payload,
        statistics_payload,
        events_payload,
    )
    stem = f"{args.league_tag}__demo_fixture_{args.fixture_id}__season_{args.season}"
    fixtures_raw = raw_dir / f"{stem}__fixtures.jsonl"
    bundle_raw = raw_dir / f"{stem}__fixtures_bundle.jsonl"
    write_jsonl(fixtures_raw, [fixture_payload])
    write_jsonl(bundle_raw, [{"response": [full_item]}])

    fixtures_csv = normalized_dir / f"fixtures_master__{args.league_tag}__{args.season}.csv"
    lineups_csv = normalized_dir / f"lineups__{args.league_tag}__{args.season}.csv"
    player_stats_csv = normalized_dir / f"match_player_stats__{args.league_tag}__{args.season}.csv"
    team_stats_csv = normalized_dir / f"match_team_stats__{args.league_tag}__{args.season}.csv"
    events_csv = normalized_dir / f"match_events__{args.league_tag}__{args.season}.csv"

    fixtures_df = build_fixtures_master(str(fixtures_raw), str(fixtures_csv))
    lineups_df = build_lineups(str(bundle_raw), str(lineups_csv))
    player_stats_df = build_match_player_stats(str(bundle_raw), str(player_stats_csv))
    team_stats_df = build_match_team_stats(str(bundle_raw), str(team_stats_csv))
    events_df = build_match_events(str(bundle_raw), str(events_csv))

    summary = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "fixture_id": args.fixture_id,
        "league_tag": args.league_tag,
        "season": args.season,
        "raw": {
            "fixtures": str(fixtures_raw),
            "bundle": str(bundle_raw),
        },
        "normalized": {
            "fixtures_master": str(fixtures_csv),
            "lineups": str(lineups_csv),
            "match_player_stats": str(player_stats_csv),
            "match_team_stats": str(team_stats_csv),
            "match_events": str(events_csv),
        },
        "counts": {
            "fixture_rows": len(fixtures_df),
            "lineup_rows": len(lineups_df),
            "player_stat_rows": len(player_stats_df),
            "team_stat_rows": len(team_stats_df),
            "event_rows": len(events_df),
        },
    }
    summary_path = args.outdir / "DEMO_FIXTURE_FULL_PROVIDER_BUNDLE_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
