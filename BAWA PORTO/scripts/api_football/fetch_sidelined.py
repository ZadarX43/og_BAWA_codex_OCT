from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .client import APIFootballClient, write_raw_json


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def with_meta(payload: dict[str, Any], *, source_scope: str, params: dict[str, Any], fetched_ts: str) -> dict[str, Any]:
    out = dict(payload)
    out["_og_fetch"] = {"source_scope": source_scope, "params": params, "fetched_ts_utc": fetched_ts}
    return out


def csv_ids(path: Path, column: str) -> list[int]:
    if not path.exists():
        return []
    df = pd.read_csv(path, low_memory=False)
    if column not in df.columns:
        return []
    ids = pd.to_numeric(df[column], errors="coerce").dropna().astype(int).tolist()
    return sorted(set(ids))


def parse_ids(value: str) -> list[int]:
    ids: list[int] = []
    for part in str(value or "").split(","):
        part = part.strip()
        if not part:
            continue
        ids.append(int(float(part)))
    return sorted(set(ids))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch API-Football /sidelined payloads by player or coach id.")
    parser.add_argument("--player-ids", default="", help="Comma-separated API-Football player ids.")
    parser.add_argument("--players-csv", default="", help="CSV containing player ids.")
    parser.add_argument("--player-id-column", default="player_id")
    parser.add_argument("--coach-ids", default="", help="Comma-separated API-Football coach ids.")
    parser.add_argument("--sleep-seconds", type=float, default=None)
    parser.add_argument("--daily-cap", type=int, default=75000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--stem", default="sidelined")
    args = parser.parse_args()

    player_ids = parse_ids(args.player_ids)
    if args.players_csv:
        player_ids.extend(csv_ids(Path(args.players_csv), args.player_id_column))
    player_ids = sorted(set(player_ids))
    coach_ids = parse_ids(args.coach_ids)
    if args.limit:
        player_ids = player_ids[: args.limit]
        coach_ids = coach_ids[: args.limit]

    client = APIFootballClient(sleep_seconds=args.sleep_seconds, daily_cap=args.daily_cap)
    rows: list[dict[str, Any]] = []
    fetched_ts = utc_now()
    for player_id in player_ids:
        params = {"player": player_id}
        rows.append(with_meta(client.get_json("/sidelined", params), source_scope="sidelined_player", params=params, fetched_ts=fetched_ts))
    for coach_id in coach_ids:
        params = {"coach": coach_id}
        rows.append(with_meta(client.get_json("/sidelined", params), source_scope="sidelined_coach", params=params, fetched_ts=fetched_ts))

    path = write_raw_json("sidelined", rows, stem=args.stem)
    print(f"WROTE RAW: {path} players={len(player_ids)} coaches={len(coach_ids)}")


if __name__ == "__main__":
    main()
