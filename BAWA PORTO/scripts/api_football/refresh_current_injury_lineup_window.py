from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.api_football.client import APIFootballClient
from scripts.api_football.build_current_injury_lineup_window_manifest import build_combined_manifest
from scripts.api_football.fetch_current_player_window import DEFAULT_LEAGUES, clean_status, fixture_ids, parse_csv_set, write_jsonl
from scripts.api_football.normalize_fixtures_master import build_fixtures_master
from scripts.api_football.normalize_injuries import build_injuries
from scripts.api_football.normalize_lineups import build_lineups
from scripts.api_football.normalize_match_player_stats import build_match_player_stats
from scripts.api_football.normalize_sidelined import build_sidelined
from scripts.api_football.utils import chunk_list


DEFAULT_OUTDIR = ROOT / "reports/latest/api_current_injury_lineup_window"
DEFAULT_SEEN_REGISTRY = ROOT / "data_sources/api_football/availability_seen_registry.csv"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def with_fetch_meta(payload: dict[str, Any], *, source_scope: str, params: dict[str, Any], fetched_ts: str) -> dict[str, Any]:
    out = dict(payload)
    out["_og_fetch"] = {"source_scope": source_scope, "params": params, "fetched_ts_utc": fetched_ts}
    return out


def date_range(start: str, end: str) -> list[str]:
    start_dt = date.fromisoformat(start)
    end_dt = date.fromisoformat(end)
    if end_dt < start_dt:
        return []
    days = []
    current = start_dt
    while current <= end_dt:
        days.append(current.isoformat())
        current += timedelta(days=1)
    return days


def teams_from_fixtures_payload(payload: dict[str, Any]) -> list[int]:
    ids: set[int] = set()
    for item in payload.get("response", []) or []:
        teams = item.get("teams") or {}
        for side in ["home", "away"]:
            team_id = ((teams.get(side) or {}).get("id"))
            if team_id is not None:
                ids.add(int(team_id))
    return sorted(ids)


def parse_scope_set(value: str) -> set[str]:
    return {part.strip() for part in str(value or "").split(",") if part.strip()}


def update_earliest_seen_registry(injuries: pd.DataFrame, registry_csv: Path, generated_at: str) -> pd.DataFrame:
    if injuries.empty or "availability_key" not in injuries.columns:
        return injuries
    registry_csv.parent.mkdir(parents=True, exist_ok=True)
    if registry_csv.exists():
        registry = pd.read_csv(registry_csv, low_memory=False)
    else:
        registry = pd.DataFrame(columns=["availability_key", "first_seen_ts_utc", "first_seen_source_scope", "last_seen_ts_utc", "last_seen_source_scope"])

    existing = {
        str(row["availability_key"]): row.to_dict()
        for _, row in registry.iterrows()
        if str(row.get("availability_key") or "").strip()
    }
    source_rank = {"league_season": 0, "league_date": 1, "team_season": 2, "fixture": 3}
    for _, row in injuries.iterrows():
        key = str(row.get("availability_key") or "").strip()
        if not key:
            continue
        seen_ts = str(row.get("fetched_ts_utc") or generated_at)
        source_scope = str(row.get("source_scope") or "")
        rec = existing.get(key)
        if rec is None:
            existing[key] = {
                "availability_key": key,
                "first_seen_ts_utc": seen_ts,
                "first_seen_source_scope": source_scope,
                "last_seen_ts_utc": seen_ts,
                "last_seen_source_scope": source_scope,
            }
        else:
            if (
                str(rec.get("first_seen_ts_utc") or "") == seen_ts
                and source_rank.get(source_scope, 9) < source_rank.get(str(rec.get("first_seen_source_scope") or ""), 9)
            ):
                rec["first_seen_source_scope"] = source_scope
            rec["last_seen_ts_utc"] = seen_ts
            rec["last_seen_source_scope"] = source_scope
    out = pd.DataFrame(existing.values()).sort_values("availability_key").reset_index(drop=True)
    out.to_csv(registry_csv, index=False)

    first_seen = out.set_index("availability_key")["first_seen_ts_utc"].to_dict() if not out.empty else {}
    first_scope = out.set_index("availability_key")["first_seen_source_scope"].to_dict() if not out.empty else {}
    enriched = injuries.copy()
    enriched["availability_first_seen_ts_utc"] = enriched["availability_key"].astype(str).map(first_seen).fillna(enriched.get("fetched_ts_utc", ""))
    enriched["availability_first_seen_source_scope"] = enriched["availability_key"].astype(str).map(first_scope).fillna("")
    enriched["fixture_only_late_confirmation_flag"] = (
        enriched.get("source_scope", pd.Series("", index=enriched.index)).astype(str).eq("fixture")
        & enriched["availability_first_seen_source_scope"].astype(str).eq("fixture")
    ).astype(int)
    return enriched


def fetch_league_bundle(
    tag: str,
    cfg: dict[str, Any],
    args: argparse.Namespace,
    client: APIFootballClient,
    raw_dir: Path,
    normalized_dir: Path,
) -> dict[str, Any]:
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

    fetched_ts = utc_now()
    injury_payloads: list[dict[str, Any]] = []
    injury_scopes = parse_scope_set(args.injury_query_scopes)
    if "fixture" in injury_scopes:
        for fixture_id in ids:
            injury_params = {"fixture": fixture_id}
            injury_payloads.append(
                with_fetch_meta(
                    client.get_json("/injuries", injury_params),
                    source_scope="fixture",
                    params=injury_params,
                    fetched_ts=fetched_ts,
                )
            )
    if "league_season" in injury_scopes:
        injury_params = {"league": league_id, "season": season}
        injury_payloads.append(
            with_fetch_meta(
                client.get_json("/injuries", injury_params),
                source_scope="league_season",
                params=injury_params,
                fetched_ts=fetched_ts,
            )
        )
    if "league_date" in injury_scopes:
        for day in date_range(from_date, to_date):
            injury_params = {"league": league_id, "season": season, "date": day, "timezone": args.timezone}
            injury_payloads.append(
                with_fetch_meta(
                    client.get_json("/injuries", injury_params),
                    source_scope="league_date",
                    params=injury_params,
                    fetched_ts=fetched_ts,
                )
            )
    if "team_season" in injury_scopes:
        for team_id in teams_from_fixtures_payload(fixtures_payload):
            injury_params = {"team": team_id, "season": season}
            injury_payloads.append(
                with_fetch_meta(
                    client.get_json("/injuries", injury_params),
                    source_scope="team_season",
                    params=injury_params,
                    fetched_ts=fetched_ts,
                )
            )
    injuries_raw = raw_dir / f"{stem}__injuries.jsonl"
    write_jsonl(injuries_raw, injury_payloads)

    fixtures_csv = normalized_dir / f"fixtures_master__{tag}__{season}.csv"
    lineups_csv = normalized_dir / f"lineups__{tag}__{season}.csv"
    player_stats_csv = normalized_dir / f"match_player_stats__{tag}__{season}.csv"
    injuries_csv = normalized_dir / f"injuries__{tag}__{season}.csv"
    sidelined_csv = normalized_dir / f"sidelined__{tag}__{season}.csv"

    fixtures_df = build_fixtures_master(str(fixtures_raw), str(fixtures_csv))
    lineups_df = build_lineups(str(bundle_raw), str(lineups_csv))
    player_stats_df = build_match_player_stats(str(bundle_raw), str(player_stats_csv))
    injuries_df = build_injuries(str(injuries_raw), str(injuries_csv))
    injuries_df = update_earliest_seen_registry(injuries_df, args.seen_registry_csv, fetched_ts)
    injuries_df.to_csv(injuries_csv, index=False)

    sidelined_payloads: list[dict[str, Any]] = []
    if args.include_sidelined:
        player_ids = sorted(
            set(pd.to_numeric(player_stats_df.get("player_id"), errors="coerce").dropna().astype(int).tolist())
            | set(pd.to_numeric(injuries_df.get("player_id"), errors="coerce").dropna().astype(int).tolist())
        )
        if args.max_sidelined_players_per_league:
            player_ids = player_ids[: args.max_sidelined_players_per_league]
        for player_id in player_ids:
            sidelined_params = {"player": player_id}
            sidelined_payloads.append(
                with_fetch_meta(
                    client.get_json("/sidelined", sidelined_params),
                    source_scope="sidelined_player",
                    params=sidelined_params,
                    fetched_ts=fetched_ts,
                )
            )
    sidelined_raw = raw_dir / f"{stem}__sidelined.jsonl"
    write_jsonl(sidelined_raw, sidelined_payloads)
    sidelined_df = build_sidelined(str(sidelined_raw), str(sidelined_csv))

    return {
        "tag": tag,
        "league_id": league_id,
        "season": season,
        "from_date": from_date,
        "to_date": to_date,
        "fixture_ids": len(ids),
        "bundle_requests": len(bundle_payloads),
        "injury_requests": len(injury_payloads),
        "sidelined_requests": len(sidelined_payloads),
        "paths": {
            "fixtures_raw": str(fixtures_raw),
            "bundle_raw": str(bundle_raw),
            "injuries_raw": str(injuries_raw),
            "sidelined_raw": str(sidelined_raw),
            "fixtures_csv": str(fixtures_csv),
            "lineups_csv": str(lineups_csv),
            "player_stats_csv": str(player_stats_csv),
            "injuries_csv": str(injuries_csv),
            "sidelined_csv": str(sidelined_csv),
        },
        "counts": {
            "fixtures_rows": len(fixtures_df),
            "lineup_rows": len(lineups_df),
            "player_stat_rows": len(player_stats_df),
            "injury_rows": len(injuries_df),
            "sidelined_rows": len(sidelined_df),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch current-window fixtures, lineups, player stats, and injuries for injury shock coverage."
    )
    parser.add_argument("--league-tags", default="", help="Comma-separated league tags. Defaults to all API-Football board leagues.")
    parser.add_argument("--from-date", default="", help="Override lower bound YYYY-MM-DD for every league.")
    parser.add_argument("--to-date", default=date.today().isoformat(), help="Upper bound YYYY-MM-DD.")
    parser.add_argument("--status", default="", help="Fixture status filter. Empty string fetches all statuses in the date window.")
    parser.add_argument("--timezone", default="Europe/London")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--sleep-seconds", type=float, default=None)
    parser.add_argument("--daily-cap", type=int, default=75000)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--max-fixtures-per-league", type=int, default=0)
    parser.add_argument(
        "--injury-query-scopes",
        default="fixture",
        help=(
            "Comma-separated /injuries scopes. Use fixture for late match confirmation; "
            "add league_season, league_date, team_season for recurring early-warning sweeps."
        ),
    )
    parser.add_argument("--include-sidelined", action="store_true", help="Fetch /sidelined for current-window players.")
    parser.add_argument("--max-sidelined-players-per-league", type=int, default=0, help="Optional quota safety cap for /sidelined.")
    parser.add_argument("--seen-registry-csv", type=Path, default=DEFAULT_SEEN_REGISTRY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = parse_csv_set(args.league_tags) if args.league_tags else set(DEFAULT_LEAGUES)
    unknown = selected - set(DEFAULT_LEAGUES)
    if unknown:
        raise SystemExit(f"Unknown league tags: {sorted(unknown)}")

    raw_dir = args.outdir / "raw"
    normalized_dir = args.outdir / "normalized"
    raw_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)

    client = APIFootballClient(sleep_seconds=args.sleep_seconds, daily_cap=args.daily_cap)
    bundles: list[dict[str, Any]] = []
    for tag in sorted(selected):
        bundles.append(fetch_league_bundle(tag, DEFAULT_LEAGUES[tag], args, client, raw_dir, normalized_dir))

    summary, manifest = build_combined_manifest(
        args.outdir,
        from_date=args.from_date,
        to_date=args.to_date,
        injury_query_scopes=parse_scope_set(args.injury_query_scopes),
        include_sidelined=bool(args.include_sidelined),
        seen_registry_csv=str(args.seen_registry_csv),
    )
    manifest_path = args.outdir / "CURRENT_INJURY_LINEUP_WINDOW_MANIFEST.json"

    print(f"WROTE {args.outdir}")
    print(f"Manifest: {manifest_path}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
