from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from .paths import NORMALIZED_FILES
from .raw_helpers import fixture_base, iter_fixture_rows, read_jsonl_payloads, to_float, to_int
from .schema_contracts import NORMALIZED_SCHEMAS
from .scaffold import build_csv_stub

PURPOSE = 'Build normalized match-team statistics table from API-Football raw team stats payloads.'
TARGET_PATH = NORMALIZED_FILES['match_team_stats']

STAT_MAP = {
    'Total Shots': 'shots_total',
    'Shots on Goal': 'shots_on_goal',
    'Shots insidebox': 'shots_inside_box',
    'Shots outsidebox': 'shots_outside_box',
    'Blocked Shots': 'blocked_shots',
    'Ball Possession': 'possession_pct',
    'Total passes': 'passes_total',
    'Passes accurate': 'passes_accurate',
    'Corner Kicks': 'corners_for',
    'Fouls': 'fouls_for',
    'Yellow Cards': 'yellow_cards',
    'Red Cards': 'red_cards',
}


def build_stub() -> pd.DataFrame:
    return build_csv_stub(TARGET_PATH, NORMALIZED_SCHEMAS['match_team_stats'], PURPOSE, placeholder_row=False)


def build_match_team_stats(bundle_raw: str, output_csv: str = str(TARGET_PATH)) -> pd.DataFrame:
    payloads = read_jsonl_payloads(bundle_raw)
    rows = []
    for item in iter_fixture_rows(payloads):
        base = fixture_base(item)
        goals = item.get('goals') or {}
        halftime = ((item.get('score') or {}).get('halftime') or {})
        for team_block in item.get('statistics', []) or []:
            team = team_block.get('team') or {}
            team_id = to_int(team.get('id'))
            is_home = int(team_id == base['home_team_id'])
            stat_values = {col: 0 for col in NORMALIZED_SCHEMAS['match_team_stats']}
            stat_values.update({
                'fixture_id': base['fixture_id'],
                'team_id': team_id,
                'team_name': team.get('name') or '',
                'is_home': is_home,
                'goals_for': to_int(goals.get('home' if is_home else 'away')),
                'goals_against': to_int(goals.get('away' if is_home else 'home')),
                'ht_goals_for': to_int(halftime.get('home' if is_home else 'away')),
                'ht_goals_against': to_int(halftime.get('away' if is_home else 'home')),
            })
            for stat in team_block.get('statistics', []) or []:
                out_col = STAT_MAP.get(stat.get('type'))
                if not out_col:
                    continue
                value = stat.get('value')
                stat_values[out_col] = to_float(value) if out_col == 'possession_pct' else to_int(value)
            rows.append({key: stat_values.get(key, 0) for key in NORMALIZED_SCHEMAS['match_team_stats']})
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=NORMALIZED_SCHEMAS['match_team_stats'])
    else:
        df = df.drop_duplicates(subset=['fixture_id', 'team_id']).sort_values(['fixture_id', 'is_home'], ascending=[True, False]).reset_index(drop=True)
        df = df.reindex(columns=NORMALIZED_SCHEMAS['match_team_stats'])
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
    df = build_match_team_stats(args.bundle_raw, args.output_csv)
    print(f'WROTE: {args.output_csv} rows={len(df)}')


if __name__ == '__main__':
    main()
