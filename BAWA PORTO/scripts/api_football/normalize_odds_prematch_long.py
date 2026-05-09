from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .paths import NORMALIZED_FILES
from .schema_contracts import NORMALIZED_SCHEMAS
from .scaffold import build_csv_stub
from .raw_helpers import to_float, to_int

PURPOSE = 'Build normalized prematch odds table from API-Football raw odds payloads.'
TARGET_PATH = NORMALIZED_FILES['odds_prematch_long']


MARKET_MAP = {
    'Match Winner': 'FTR',
    'Both Teams Score': 'BTTS',
    'Goals Over/Under': 'OU',
}
SELECTION_ALIASES = {
    'Home': 'HOME', 'Draw': 'DRAW', 'Away': 'AWAY',
    'Yes': 'YES', 'No': 'NO',
}


def build_stub() -> pd.DataFrame:
    return build_csv_stub(TARGET_PATH, NORMALIZED_SCHEMAS['odds_prematch_long'], PURPOSE, placeholder_row=False)


def _read_payloads(path: str | Path) -> list[dict]:
    source = Path(path)
    if not source.exists():
        return []
    payloads = []
    with source.open('r', encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            line = line.strip()
            if line:
                payloads.append(json.loads(line))
    return payloads


def build_odds_prematch_long(odds_raw: str | None = None, output_csv: str = str(TARGET_PATH)) -> pd.DataFrame:
    rows = []
    payloads = _read_payloads(odds_raw) if odds_raw else []
    for payload in payloads:
        for item in payload.get('response', []) or []:
            fixture = item.get('fixture') or {}
            fixture_id = to_int(fixture.get('id'))
            update_ts = fixture.get('update') or item.get('update') or ''
            for bookmaker in item.get('bookmakers', []) or []:
                bookmaker_id = to_int(bookmaker.get('id'))
                bookmaker_name = bookmaker.get('name') or ''
                for bet in bookmaker.get('bets', []) or []:
                    market_name = bet.get('name') or ''
                    market_code = MARKET_MAP.get(market_name, market_name.upper().replace(' ', '_'))
                    for value in bet.get('values', []) or []:
                        selection_name = str(value.get('value') or '').strip()
                        line_value = str(value.get('handicap') or value.get('main') or '').strip()
                        rows.append({
                            'fixture_id': fixture_id,
                            'bookmaker_id': bookmaker_id,
                            'bookmaker_name': bookmaker_name,
                            'market_code': market_code,
                            'market_name': market_name,
                            'selection_code': SELECTION_ALIASES.get(selection_name, selection_name.upper().replace(' ', '_')),
                            'selection_name': selection_name,
                            'line_value': line_value,
                            'odds': to_float(value.get('odd')),
                            'snapshot_ts_utc': update_ts,
                            'is_opening': 0,
                            'is_latest_pre_kickoff': 1,
                        })
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=NORMALIZED_SCHEMAS['odds_prematch_long'])
    else:
        df = df.reindex(columns=NORMALIZED_SCHEMAS['odds_prematch_long'])
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=PURPOSE)
    parser.add_argument('--write-stub', action='store_true', help='Write the scaffold output even though live transform logic is not implemented yet.')
    parser.add_argument('--odds-raw', default='', help='Path to raw odds JSONL when available.')
    parser.add_argument('--output-csv', default=str(TARGET_PATH))
    args = parser.parse_args()
    if args.write_stub:
        df = build_stub()
        print(f'WROTE STUB: {TARGET_PATH} rows={len(df)}')
        return
    df = build_odds_prematch_long(args.odds_raw or None, args.output_csv)
    print(f'WROTE: {args.output_csv} rows={len(df)}')


if __name__ == '__main__':
    main()
