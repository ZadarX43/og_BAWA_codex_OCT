from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from .paths import NORMALIZED_FILES
from .raw_helpers import fixture_base, iter_fixture_rows, read_jsonl_payloads, to_int
from .schema_contracts import NORMALIZED_SCHEMAS
from .scaffold import build_csv_stub

PURPOSE = 'Build normalized lineups table from API-Football raw lineup payloads.'
TARGET_PATH = NORMALIZED_FILES['lineups']


def build_stub() -> pd.DataFrame:
    return build_csv_stub(TARGET_PATH, NORMALIZED_SCHEMAS['lineups'], PURPOSE, placeholder_row=False)


def _iter_players(items: list[dict], is_starting_xi: int):
    for slot in items or []:
        player = slot.get('player') or {}
        yield {
            'player_id': to_int(player.get('id')),
            'player_name': player.get('name') or '',
            'position': player.get('pos') or '',
            'is_starting_xi': is_starting_xi,
        }


def build_lineups(bundle_raw: str, output_csv: str = str(TARGET_PATH)) -> pd.DataFrame:
    payloads = read_jsonl_payloads(bundle_raw)
    rows = []
    for item in iter_fixture_rows(payloads):
        base = fixture_base(item)
        kickoff_ts = base.get('kickoff_ts_utc') or ''
        for lineup in item.get('lineups', []) or []:
            team = lineup.get('team') or {}
            team_id = to_int(team.get('id'))
            formation = lineup.get('formation') or ''
            for player_row in _iter_players(lineup.get('startXI', []) or [], 1):
                rows.append({
                    'fixture_id': base['fixture_id'],
                    'team_id': team_id,
                    'player_id': player_row['player_id'],
                    'player_name': player_row['player_name'],
                    'formation': formation,
                    'is_starting_xi': player_row['is_starting_xi'],
                    'position': player_row['position'],
                    # API bundle payload does not expose lineup publication timestamp, so default conservative flags.
                    'lineup_known_pre_kickoff_flag': 0,
                    'lineup_published_ts_utc': '',
                })
            for player_row in _iter_players(lineup.get('substitutes', []) or [], 0):
                rows.append({
                    'fixture_id': base['fixture_id'],
                    'team_id': team_id,
                    'player_id': player_row['player_id'],
                    'player_name': player_row['player_name'],
                    'formation': formation,
                    'is_starting_xi': player_row['is_starting_xi'],
                    'position': player_row['position'],
                    'lineup_known_pre_kickoff_flag': 0,
                    'lineup_published_ts_utc': '',
                })
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=NORMALIZED_SCHEMAS['lineups'])
    else:
        df = df.drop_duplicates(subset=['fixture_id', 'team_id', 'player_id', 'is_starting_xi']).sort_values(
            ['fixture_id', 'team_id', 'is_starting_xi', 'player_id'], ascending=[True, True, False, True]
        ).reset_index(drop=True)
        df = df.reindex(columns=NORMALIZED_SCHEMAS['lineups'])
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
    df = build_lineups(args.bundle_raw, args.output_csv)
    print(f'WROTE: {args.output_csv} rows={len(df)}')


if __name__ == '__main__':
    main()
