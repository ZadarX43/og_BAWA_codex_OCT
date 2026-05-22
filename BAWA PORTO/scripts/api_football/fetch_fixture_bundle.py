from __future__ import annotations

import argparse
import json
from pathlib import Path

from .client import APIFootballClient, write_raw_json
from .utils import chunk_list


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
    parser = argparse.ArgumentParser(description='Fetch efficient fixture bundles via /fixtures?ids=... using groups of up to 20 fixture ids.')
    parser.add_argument('--fixtures-raw', required=True, help='Path to raw fixtures JSONL from fetch_fixtures.py')
    parser.add_argument('--sleep-seconds', type=float, default=None)
    parser.add_argument('--daily-cap', type=int, default=75000)
    parser.add_argument('--limit', type=int, default=None, help='Optional cap for test runs before chunking.')
    parser.add_argument('--chunk-size', type=int, default=20, help='API docs recommend max 20 fixture ids per ids query.')
    args = parser.parse_args()

    source = Path(args.fixtures_raw)
    fixture_ids = iter_fixture_ids(source)
    if args.limit is not None:
        fixture_ids = fixture_ids[: args.limit]
    if not fixture_ids:
        raise SystemExit('No fixture ids found in the supplied raw fixtures file.')

    client = APIFootballClient(sleep_seconds=args.sleep_seconds, daily_cap=args.daily_cap)
    chunks = chunk_list(fixture_ids, max(1, min(int(args.chunk_size), 20)))
    rows = []
    for fixture_chunk in chunks:
        ids_param = '-'.join(str(x) for x in fixture_chunk)
        rows.append(client.get_json('/fixtures', {'ids': ids_param}))

    stem = source.stem.replace('__fixtures', '') if '__fixtures' in source.stem else source.stem + '__bundle'
    path = write_raw_json('fixtures_bundle', rows, stem=stem)
    print(f'WROTE RAW: {path} chunks={len(chunks)} fixtures={len(fixture_ids)} chunk_size={min(int(args.chunk_size), 20)}')


if __name__ == '__main__':
    main()
