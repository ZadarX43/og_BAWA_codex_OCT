from __future__ import annotations

import argparse
import json
from pathlib import Path

from .client import APIFootballClient, write_raw_json
from .paths import RAW_DIR


def iter_fixture_ids(source: Path) -> list[int]:
    fixture_ids: list[int] = []
    with source.open('r', encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            for item in payload.get('response', []) or []:
                fixture = item.get('fixture') or {}
                fid = fixture.get('id')
                if fid is not None:
                    fixture_ids.append(int(fid))
    return fixture_ids


def main() -> None:
    parser = argparse.ArgumentParser(description='Fetch API-Football fixture statistics into raw JSONL files.')
    parser.add_argument('--fixtures-raw', default='', help='Path to raw fixtures JSONL from fetch_fixtures.py')
    parser.add_argument('--sleep-seconds', type=float, default=None)
    parser.add_argument('--daily-cap', type=int, default=75000)
    parser.add_argument('--limit', type=int, default=None, help='Optional cap for test runs.')
    args = parser.parse_args()

    if not args.fixtures_raw:
        raise SystemExit('Provide --fixtures-raw pointing to a fixtures raw JSONL file.')

    source = Path(args.fixtures_raw)
    fixture_ids = iter_fixture_ids(source)
    if args.limit is not None:
        fixture_ids = fixture_ids[: args.limit]

    client = APIFootballClient(sleep_seconds=args.sleep_seconds, daily_cap=args.daily_cap)
    rows = []
    for fixture_id in fixture_ids:
        rows.append(client.get_json('/fixtures/statistics', {'fixture': fixture_id}))
    stem = source.stem.replace('__fixtures', '') if '__fixtures' in source.stem else source.stem + '__stats'
    path = write_raw_json('fixtures_statistics', rows, stem=stem)
    print(f'WROTE RAW: {path} fixtures={len(fixture_ids)}')


if __name__ == '__main__':
    main()
