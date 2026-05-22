from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from .paths import NORMALIZED_FILES
from .raw_helpers import fixture_base, iter_fixture_rows, read_jsonl_payloads
from .schema_contracts import NORMALIZED_SCHEMAS
from .scaffold import build_csv_stub

PURPOSE = 'Build normalized fixtures master table from API-Football raw fixture payloads.'
TARGET_PATH = NORMALIZED_FILES['fixtures_master']


def build_stub() -> pd.DataFrame:
    return build_csv_stub(TARGET_PATH, NORMALIZED_SCHEMAS['fixtures_master'], PURPOSE, placeholder_row=False)


def build_fixtures_master(fixtures_raw: str, output_csv: str = str(TARGET_PATH)) -> pd.DataFrame:
    payloads = read_jsonl_payloads(fixtures_raw)
    rows = [fixture_base(item) for item in iter_fixture_rows(payloads)]
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=NORMALIZED_SCHEMAS['fixtures_master'])
    else:
        df = df.drop_duplicates(subset=['fixture_id']).sort_values(['match_date', 'fixture_id']).reset_index(drop=True)
        df = df.reindex(columns=NORMALIZED_SCHEMAS['fixtures_master'])
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=PURPOSE)
    parser.add_argument('--write-stub', action='store_true', help='Write the scaffold output even though live transform logic is not implemented yet.')
    parser.add_argument('--fixtures-raw', default='', help='Path to raw fixtures JSONL from fetch_fixtures.py')
    parser.add_argument('--output-csv', default=str(TARGET_PATH))
    args = parser.parse_args()
    if args.write_stub:
        df = build_stub()
        print(f'WROTE STUB: {TARGET_PATH} rows={len(df)}')
        return
    if not args.fixtures_raw:
        raise SystemExit('Provide --fixtures-raw to build the live normalized table.')
    df = build_fixtures_master(args.fixtures_raw, args.output_csv)
    print(f'WROTE: {args.output_csv} rows={len(df)}')


if __name__ == '__main__':
    main()
