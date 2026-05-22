#!/usr/bin/env python3
"""Fetch API-Football World Cup /players pages for a season."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.api_football.client import APIFootballClient
from scripts.api_football.paths import RAW_DIR, ensure_dirs


WORLD_CUP_LEAGUE_ID = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--league-id", type=int, default=WORLD_CUP_LEAGUE_ID)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--daily-cap", type=int, default=75000)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = APIFootballClient(sleep_seconds=args.sleep_seconds, daily_cap=args.daily_cap)
    payloads = client.paged_get(
        "/players",
        {"league": args.league_id, "season": args.season},
        max_pages=args.max_pages,
    )
    ensure_dirs()
    out = args.out or RAW_DIR / f"players__league_{args.league_id}__season_{args.season}__players.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for payload in payloads:
            fh.write(json.dumps(payload, ensure_ascii=True) + "\n")

    rows = sum(len(p.get("response") or []) for p in payloads)
    pages = len(payloads)
    print(f"WROTE RAW: {out} pages={pages} rows={rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
