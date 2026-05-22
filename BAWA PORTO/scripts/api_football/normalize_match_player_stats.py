from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from .paths import NORMALIZED_FILES
from .raw_helpers import read_jsonl_payloads, to_float, to_int
from .schema_contracts import NORMALIZED_SCHEMAS
from .scaffold import build_csv_stub

PURPOSE = 'Build normalized match-player stats table from API-Football raw player payloads.'
TARGET_PATH = NORMALIZED_FILES['match_player_stats']


def build_stub() -> pd.DataFrame:
    return build_csv_stub(TARGET_PATH, NORMALIZED_SCHEMAS['match_player_stats'], PURPOSE, placeholder_row=False)


def build_match_player_stats(bundle_raw: str, output_csv: str = str(TARGET_PATH)) -> pd.DataFrame:
    payloads = read_jsonl_payloads(bundle_raw)
    rows = []
    for payload in payloads:
        fixture_id = to_int((payload.get('parameters') or {}).get('fixture'))
        response = payload.get('response', []) or []
        # `/fixtures/players?fixture=...` returns team blocks directly, while
        # fixture bundle payloads may return fixture objects containing
        # `players`. Support both shapes to preserve historical player stats.
        if response and any('players' in item and 'team' in item for item in response):
            team_blocks = response
        else:
            team_blocks = []
            for item in response:
                if fixture_id == 0:
                    fixture_id = to_int((item.get('fixture') or {}).get('id'))
                team_blocks.extend(item.get('players', []) or [])
        for team_block in team_blocks:
            team = team_block.get('team') or {}
            team_id = to_int(team.get('id'))
            for player_block in team_block.get('players', []) or []:
                player = player_block.get('player') or {}
                stat = ((player_block.get('statistics') or [{}])[0]) or {}
                games = stat.get('games') or {}
                shots = stat.get('shots') or {}
                goals = stat.get('goals') or {}
                passes = stat.get('passes') or {}
                tackles = stat.get('tackles') or {}
                duels = stat.get('duels') or {}
                dribbles = stat.get('dribbles') or {}
                fouls = stat.get('fouls') or {}
                cards = stat.get('cards') or {}
                row = {
                    'fixture_id': fixture_id,
                    'player_id': to_int(player.get('id')),
                    'team_id': team_id,
                    'player_name': player.get('name') or '',
                    'position': games.get('position') or '',
                    'minutes': to_int(games.get('minutes')),
                    'started_flag': int(not bool(games.get('substitute'))),
                    'subbed_on_flag': int(bool(games.get('substitute')) and to_int(games.get('minutes')) > 0),
                    'subbed_off_flag': int((not bool(games.get('substitute'))) and 0 < to_int(games.get('minutes')) < 90),
                    'rating': to_float(games.get('rating')),
                    'goals': to_int(goals.get('total')),
                    'assists': to_int(goals.get('assists')),
                    'shots_total': to_int(shots.get('total')),
                    'shots_on_target': to_int(shots.get('on')),
                    'passes_total': to_int(passes.get('total')),
                    'passes_key': to_int(passes.get('key')),
                    'passes_accurate': to_int(passes.get('accuracy')),
                    'tackles': to_int(tackles.get('total')),
                    'interceptions': to_int(tackles.get('interceptions')),
                    'blocks': to_int(tackles.get('blocks')),
                    'duels_total': to_int(duels.get('total')),
                    'duels_won': to_int(duels.get('won')),
                    'dribbles_attempted': to_int(dribbles.get('attempts')),
                    'dribbles_successful': to_int(dribbles.get('success')),
                    'dribbled_past': to_int(dribbles.get('past')),
                    'fouls_drawn': to_int(fouls.get('drawn')),
                    'fouls_committed': to_int(fouls.get('committed')),
                    'yellow_cards': to_int(cards.get('yellow')),
                    'red_cards': to_int(cards.get('red')),
                    'saves': to_int(goals.get('saves')),
                    'goals_conceded': to_int(goals.get('conceded')),
                }
                rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=NORMALIZED_SCHEMAS['match_player_stats'])
    else:
        df = df.drop_duplicates(subset=['fixture_id', 'player_id', 'team_id']).sort_values(['fixture_id', 'team_id', 'player_id']).reset_index(drop=True)
        df = df.reindex(columns=NORMALIZED_SCHEMAS['match_player_stats'])
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
    df = build_match_player_stats(args.bundle_raw, args.output_csv)
    print(f'WROTE: {args.output_csv} rows={len(df)}')


if __name__ == '__main__':
    main()
