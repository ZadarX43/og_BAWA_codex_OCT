from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from .paths import NORMALIZED_FILES
from .raw_helpers import fixture_base, iter_fixture_rows, read_jsonl_payloads, to_int
from .schema_contracts import NORMALIZED_SCHEMAS
from .scaffold import build_csv_stub

PURPOSE = 'Build normalized event table from API-Football raw event payloads.'
TARGET_PATH = NORMALIZED_FILES['match_events']
GOAL_DETAILS = {'Normal Goal', 'Own Goal', 'Penalty'}


def build_stub() -> pd.DataFrame:
    return build_csv_stub(TARGET_PATH, NORMALIZED_SCHEMAS['match_events'], PURPOSE, placeholder_row=False)


def build_match_events(bundle_raw: str, output_csv: str = str(TARGET_PATH)) -> pd.DataFrame:
    payloads = read_jsonl_payloads(bundle_raw)
    rows = []
    for item in iter_fixture_rows(payloads):
        base = fixture_base(item)
        score_home = 0
        score_away = 0
        for idx, event in enumerate(item.get('events', []) or [], start=1):
            team = event.get('team') or {}
            team_id = to_int(team.get('id'))
            is_home = int(team_id == base['home_team_id'])
            event_type = event.get('type') or ''
            event_detail = event.get('detail') or ''
            if event_type == 'Goal' and event_detail in GOAL_DETAILS:
                if event_detail == 'Own Goal':
                    if is_home:
                        score_away += 1
                    else:
                        score_home += 1
                else:
                    if is_home:
                        score_home += 1
                    else:
                        score_away += 1
            rows.append({
                'fixture_id': base['fixture_id'],
                'event_id': int(base['fixture_id']) * 1000 + idx,
                'minute': to_int(((event.get('time') or {}).get('elapsed'))),
                'extra_minute': to_int(((event.get('time') or {}).get('extra'))),
                'team_id': team_id,
                'player_id': to_int(((event.get('player') or {}).get('id'))),
                'event_type': event_type,
                'event_detail': event_detail,
                'is_home': is_home,
                'score_home_after': score_home,
                'score_away_after': score_away,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=NORMALIZED_SCHEMAS['match_events'])
    else:
        df = df.sort_values(['fixture_id', 'minute', 'extra_minute', 'event_id']).reset_index(drop=True)
        df = df.reindex(columns=NORMALIZED_SCHEMAS['match_events'])
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=PURPOSE)
    parser.add_argument('--write-stub', action='store_true', help='Write the scaffold output even though live transform logic is not implemented yet.')
    parser.add_argument('--bundle-raw', default='', help='Path to raw fixtures bundle JSONL from fetch_fixture_bundle.py')
    parser.add_argument('--output-csv', default=str(TARGET_PATH))
    args = parser.parse_args()
    if args.write_stub:
        df = build_stub()
        print(f'WROTE STUB: {TARGET_PATH} rows={len(df)}')
        return
    if not args.bundle_raw:
        raise SystemExit('Provide --bundle-raw to build the live normalized table.')
    df = build_match_events(args.bundle_raw, args.output_csv)
    print(f'WROTE: {args.output_csv} rows={len(df)}')


if __name__ == '__main__':
    main()
