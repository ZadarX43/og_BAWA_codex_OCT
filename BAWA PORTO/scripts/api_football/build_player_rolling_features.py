from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .paths import FEATURE_FILES, NORMALIZED_FILES
from .schema_contracts import FEATURE_SCHEMAS
from .scaffold import build_csv_stub
from .utils import safe_div

PURPOSE = 'Build additive rolling player feature table using pre-fixture appearances only.'
TARGET_PATH = FEATURE_FILES['api_player_rolling_features']


def build_stub() -> pd.DataFrame:
    return build_csv_stub(TARGET_PATH, FEATURE_SCHEMAS['api_player_rolling_features'], PURPOSE, placeholder_row=False)


def _mean(records, key, n):
    sample = records[:n]
    return sum(float(r.get(key, 0.0) or 0.0) for r in sample) / len(sample) if sample else 0.0


def _sum(records, key, n):
    sample = records[:n]
    return sum(float(r.get(key, 0.0) or 0.0) for r in sample) if sample else 0.0


def build_player_rolling_features(fixtures_csv: str, player_stats_csv: str, output_csv: str = str(TARGET_PATH)) -> pd.DataFrame:
    fixtures = pd.read_csv(fixtures_csv)
    player_stats = pd.read_csv(player_stats_csv)
    fixtures['kickoff_ts_utc'] = pd.to_datetime(fixtures['kickoff_ts_utc'], errors='coerce', utc=True)
    ps = player_stats.merge(fixtures[['fixture_id', 'fixture_key', 'league', 'league_id', 'season', 'match_date', 'home_team_id', 'away_team_id', 'home_team_name', 'away_team_name', 'kickoff_ts_utc']], on='fixture_id', how='left')
    ps = ps.sort_values(['kickoff_ts_utc', 'fixture_id', 'team_id', 'player_id']).reset_index(drop=True)

    history: dict[int, list[dict]] = defaultdict(list)
    out_rows = []
    for _, row in ps.iterrows():
        hist = list(reversed(history.get(int(row['player_id']), [])))
        sample5 = hist[:5]
        sample10 = hist[:10]
        minutes5 = _sum(sample5, 'minutes', 5)
        minutes10 = _sum(sample10, 'minutes', 10)
        shots5 = _sum(sample5, 'shots_total', 5)
        sot5 = _sum(sample5, 'shots_on_target', 5)
        passes_total5 = _sum(sample5, 'passes_total', 5)
        passes_accurate5 = _sum(sample5, 'passes_accurate', 5)
        duels_total5 = _sum(sample5, 'duels_total', 5)
        duels_won5 = _sum(sample5, 'duels_won', 5)
        drib_attempt5 = _sum(sample5, 'dribbles_attempted', 5)
        drib_success5 = _sum(sample5, 'dribbles_successful', 5)
        out_rows.append({
            'fixture_id': int(row['fixture_id']),
            'fixture_key': row['fixture_key'],
            'league': row['league'],
            'league_id': int(row['league_id']),
            'season': int(row['season']),
            'match_date': row['match_date'],
            'home_team_id': int(row['home_team_id']),
            'away_team_id': int(row['away_team_id']),
            'home_team_name': row['home_team_name'],
            'away_team_name': row['away_team_name'],
            'player_id': int(row['player_id']),
            'player_name': row['player_name'],
            'team_id': int(row['team_id']),
            'position': row['position'],
            'player_minutes_l5': _mean(sample5, 'minutes', 5),
            'player_start_rate_l5': _mean(sample5, 'started_flag', 5),
            'player_rating_l5': _mean(sample5, 'rating', 5),
            'player_goals_l5': _mean(sample5, 'goals', 5),
            'player_assists_l5': _mean(sample5, 'assists', 5),
            'player_shots_l5': _mean(sample5, 'shots_total', 5),
            'player_sot_l5': _mean(sample5, 'shots_on_target', 5),
            'player_tackles_l5': _mean(sample5, 'tackles', 5),
            'player_fouls_committed_l5': _mean(sample5, 'fouls_committed', 5),
            'player_fouls_drawn_l5': _mean(sample5, 'fouls_drawn', 5),
            'player_yellow_cards_l10': _mean(sample10, 'yellow_cards', 10),
            'player_red_cards_l10': _mean(sample10, 'red_cards', 10),
            'player_goals_per90_l5': safe_div(_sum(sample5, 'goals', 5) * 90.0, minutes5),
            'player_assists_per90_l5': safe_div(_sum(sample5, 'assists', 5) * 90.0, minutes5),
            'player_shots_per90_l5': safe_div(shots5 * 90.0, minutes5),
            'player_sot_per90_l5': safe_div(sot5 * 90.0, minutes5),
            'player_tackles_per90_l5': safe_div(_sum(sample5, 'tackles', 5) * 90.0, minutes5),
            'player_fouls_committed_per90_l5': safe_div(_sum(sample5, 'fouls_committed', 5) * 90.0, minutes5),
            'player_fouls_drawn_per90_l5': safe_div(_sum(sample5, 'fouls_drawn', 5) * 90.0, minutes5),
            'player_cards_per90_l10': safe_div((_sum(sample10, 'yellow_cards', 10) + _sum(sample10, 'red_cards', 10)) * 90.0, minutes10),
            'player_shot_accuracy_l5': safe_div(sot5, shots5),
            'player_pass_accuracy_l5': safe_div(passes_accurate5, passes_total5),
            'player_duel_win_rate_l5': safe_div(duels_won5, duels_total5),
            'player_dribble_success_rate_l5': safe_div(drib_success5, drib_attempt5),
        })
        history[int(row['player_id'])].append(row.to_dict())

    df = pd.DataFrame(out_rows)
    if df.empty:
        df = pd.DataFrame(columns=FEATURE_SCHEMAS['api_player_rolling_features'])
    else:
        df = df.reindex(columns=FEATURE_SCHEMAS['api_player_rolling_features'])
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=PURPOSE)
    parser.add_argument('--write-stub', action='store_true', help='Write the scaffold output even though live transform logic is not implemented yet.')
    parser.add_argument('--fixtures-csv', default=str(NORMALIZED_FILES['fixtures_master']))
    parser.add_argument('--player-stats-csv', default=str(NORMALIZED_FILES['match_player_stats']))
    parser.add_argument('--output-csv', default=str(TARGET_PATH))
    args = parser.parse_args()
    if args.write_stub:
        df = build_stub()
        print(f'WROTE STUB: {TARGET_PATH} rows={len(df)}')
        return
    df = build_player_rolling_features(args.fixtures_csv, args.player_stats_csv, args.output_csv)
    print(f'WROTE: {args.output_csv} rows={len(df)}')


if __name__ == '__main__':
    main()
