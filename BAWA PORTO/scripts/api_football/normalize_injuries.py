from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from .paths import NORMALIZED_FILES
from .raw_helpers import to_int
from .schema_contracts import NORMALIZED_SCHEMAS
from .scaffold import build_csv_stub

PURPOSE = 'Build normalized injuries table from API-Football raw injury payloads.'
TARGET_PATH = NORMALIZED_FILES['injuries']


def build_stub() -> pd.DataFrame:
    return build_csv_stub(TARGET_PATH, NORMALIZED_SCHEMAS['injuries'], PURPOSE, placeholder_row=False)


def _read_payloads(path: str | Path) -> list[dict]:
    payloads = []
    source = Path(path)
    if not source.exists():
        return payloads
    with source.open('r', encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            line = line.strip()
            if line:
                payloads.append(json.loads(line))
    return payloads


def _as_flag(value: object) -> int:
    text = str(value or '').lower().strip()
    return int(text in {'1', 'true', 'yes', 'y'})


def _norm_key(value: object) -> str:
    text = str(value or '').strip().lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return re.sub(r'_+', '_', text).strip('_')


def availability_key(team_id: int, player_id: int, absence_type: object, reason: object) -> str:
    return f'{team_id}:{player_id}:{_norm_key(absence_type)}:{_norm_key(reason)}'


def build_injuries(injuries_raw: str | None = None, output_csv: str = str(TARGET_PATH)) -> pd.DataFrame:
    rows = []
    payloads = _read_payloads(injuries_raw) if injuries_raw else []
    for payload in payloads:
        fetch_meta = payload.get('_og_fetch') or {}
        source_scope = fetch_meta.get('source_scope') or ''
        source_params = json.dumps(fetch_meta.get('params') or {}, sort_keys=True, ensure_ascii=True)
        fetched_ts = fetch_meta.get('fetched_ts_utc') or ''
        for item in payload.get('response', []) or []:
            player = item.get('player') or {}
            team = item.get('team') or {}
            fixture = item.get('fixture') or {}
            league = item.get('league') or {}
            fixture_id = to_int(fixture.get('id'))
            team_id = to_int(team.get('id'))
            player_id = to_int(player.get('id'))
            absence_type = player.get('type') or item.get('type') or item.get('absence_type') or ''
            reason = player.get('reason') or item.get('reason') or ''
            rows.append({
                'fixture_id': fixture_id,
                'team_id': team_id,
                'player_id': player_id,
                'player_name': player.get('name') or '',
                'absence_type': absence_type,
                'reason': reason,
                'status': item.get('status') or player.get('status') or '',
                'known_pre_kickoff_flag': _as_flag(item.get('known_pre_kickoff_flag')),
                'published_ts_utc': item.get('published_ts_utc') or fetched_ts or fixture.get('date') or '',
                'provider_fixture_ts_utc': fixture.get('date') or '',
                'source_scope': source_scope,
                'source_params': source_params,
                'fetched_ts_utc': fetched_ts,
                'availability_key': availability_key(team_id, player_id, absence_type, reason),
                'availability_first_seen_ts_utc': '',
                'fixture_only_late_confirmation_flag': int(source_scope == 'fixture'),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=NORMALIZED_SCHEMAS['injuries'])
    else:
        df = df.drop_duplicates(subset=['fixture_id', 'team_id', 'player_id', 'absence_type', 'reason', 'source_scope']).reset_index(drop=True)
        df = df.reindex(columns=NORMALIZED_SCHEMAS['injuries'])
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=PURPOSE)
    parser.add_argument('--write-stub', action='store_true', help='Write the scaffold output even though live transform logic is not implemented yet.')
    parser.add_argument('--injuries-raw', default='', help='Path to raw injuries JSONL when available.')
    parser.add_argument('--output-csv', default=str(TARGET_PATH))
    args = parser.parse_args()
    if args.write_stub:
        df = build_stub()
        print(f'WROTE STUB: {TARGET_PATH} rows={len(df)}')
        return
    df = build_injuries(args.injuries_raw or None, args.output_csv)
    print(f'WROTE: {args.output_csv} rows={len(df)}')


if __name__ == '__main__':
    main()
