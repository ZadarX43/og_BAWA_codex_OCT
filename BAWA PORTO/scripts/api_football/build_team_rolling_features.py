from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .paths import FEATURE_FILES, NORMALIZED_FILES
from .raw_helpers import to_float
from .schema_contracts import FEATURE_SCHEMAS
from .scaffold import build_csv_stub
from .utils import safe_div

PURPOSE = 'Build additive rolling team feature table using completed historical fixtures only.'
TARGET_PATH = FEATURE_FILES['api_team_rolling_features']
WEIGHTS_L5 = [0.35, 0.25, 0.20, 0.12, 0.08]


def build_stub() -> pd.DataFrame:
    return build_csv_stub(TARGET_PATH, FEATURE_SCHEMAS['api_team_rolling_features'], PURPOSE, placeholder_row=False)


def _mean(records, key, n):
    sample = records[:n]
    return sum(float(r.get(key, 0.0) or 0.0) for r in sample) / len(sample) if sample else 0.0


def _rate(records, pred, n):
    sample = records[:n]
    return sum(1 for r in sample if pred(r)) / len(sample) if sample else 0.0


def _weighted_points(records):
    sample = records[:5]
    return sum((float(r.get('points', 0.0) or 0.0) * WEIGHTS_L5[idx]) for idx, r in enumerate(sample)) if sample else 0.0


def _team_metrics(records):
    records = list(records)
    points_l3 = _mean(records, 'points', 3)
    points_l5 = _mean(records, 'points', 5)
    points_l10 = _mean(records, 'points', 10)
    points_season = _mean(records, 'points', len(records))
    shots_l5 = _mean(records, 'shots_total', 5)
    sot_l5 = _mean(records, 'shots_on_goal', 5)
    passes_total_l5 = _mean(records, 'passes_total', 5)
    passes_accurate_l5 = _mean(records, 'passes_accurate', 5)
    corners_l5 = _mean(records, 'corners_for', 5)
    shots_inside_box_l5 = _mean(records, 'shots_inside_box', 5)
    yellow_l5 = _mean(records, 'yellow_cards', 5)
    red_l10 = _mean(records, 'red_cards', 10)
    cards_total_l5 = _mean(records, 'cards_total', 5)
    return {
        'ppg_l3': points_l3,
        'ppg_l5': points_l5,
        'ppg_l10': points_l10,
        'ppg_season': points_season,
        'win_rate_l5': _rate(records, lambda r: r.get('points', 0) == 3, 5),
        'draw_rate_l5': _rate(records, lambda r: r.get('points', 0) == 1, 5),
        'loss_rate_l5': _rate(records, lambda r: r.get('points', 0) == 0, 5),
        'points_weighted_l5': _weighted_points(records),
        'goals_for_l3': _mean(records, 'goals_for', 3),
        'goals_for_l5': _mean(records, 'goals_for', 5),
        'goals_for_l10': _mean(records, 'goals_for', 10),
        'goals_against_l3': _mean(records, 'goals_against', 3),
        'goals_against_l5': _mean(records, 'goals_against', 5),
        'goals_against_l10': _mean(records, 'goals_against', 10),
        'goal_diff_l5': _mean(records, 'goal_diff', 5),
        'total_goals_l5': _mean(records, 'total_goals', 5),
        'over05_rate_l5': _rate(records, lambda r: r.get('total_goals', 0) >= 1, 5),
        'over15_rate_l5': _rate(records, lambda r: r.get('total_goals', 0) >= 2, 5),
        'over25_rate_l5': _rate(records, lambda r: r.get('total_goals', 0) >= 3, 5),
        'over35_rate_l5': _rate(records, lambda r: r.get('total_goals', 0) >= 4, 5),
        'over45_rate_l5': _rate(records, lambda r: r.get('total_goals', 0) >= 5, 5),
        'btts_rate_l5': _rate(records, lambda r: r.get('btts', 0) == 1, 5),
        'btts_rate_l10': _rate(records, lambda r: r.get('btts', 0) == 1, 10),
        'clean_sheet_rate_l5': _rate(records, lambda r: r.get('goals_against', 0) == 0, 5),
        'fts_rate_l5': _rate(records, lambda r: r.get('goals_for', 0) == 0, 5),
        'scored_rate_l5': _rate(records, lambda r: r.get('goals_for', 0) > 0, 5),
        'conceded_rate_l5': _rate(records, lambda r: r.get('goals_against', 0) > 0, 5),
        'ht_goals_for_l5': _mean(records, 'ht_goals_for', 5),
        'ht_goals_against_l5': _mean(records, 'ht_goals_against', 5),
        'ht_leading_rate_l5': _rate(records, lambda r: r.get('ht_goals_for', 0) > r.get('ht_goals_against', 0), 5),
        'ht_drawing_rate_l5': _rate(records, lambda r: r.get('ht_goals_for', 0) == r.get('ht_goals_against', 0), 5),
        'ht_losing_rate_l5': _rate(records, lambda r: r.get('ht_goals_for', 0) < r.get('ht_goals_against', 0), 5),
        'ht_over05_rate_l5': _rate(records, lambda r: (r.get('ht_goals_for', 0) + r.get('ht_goals_against', 0)) >= 1, 5),
        'ht_over15_rate_l5': _rate(records, lambda r: (r.get('ht_goals_for', 0) + r.get('ht_goals_against', 0)) >= 2, 5),
        'ht_over25_rate_l5': _rate(records, lambda r: (r.get('ht_goals_for', 0) + r.get('ht_goals_against', 0)) >= 3, 5),
        'shots_l5': shots_l5,
        'sot_l5': sot_l5,
        'shot_accuracy_l5': safe_div(sot_l5, shots_l5),
        'shots_inside_box_l5': shots_inside_box_l5,
        'shots_outside_box_l5': _mean(records, 'shots_outside_box', 5),
        'blocked_shots_l5': _mean(records, 'blocked_shots', 5),
        'possession_l5': _mean(records, 'possession_pct', 5),
        'passes_l5': passes_total_l5,
        'pass_accuracy_l5': safe_div(passes_accurate_l5, passes_total_l5),
        'progressive_proxy_l5': passes_accurate_l5 * safe_div(_mean(records, 'possession_pct', 5), 100.0),
        'corners_for_l5': corners_l5,
        'corners_against_l5': _mean(records, 'corners_against', 5),
        'corner_pressure_l5': safe_div(corners_l5, shots_l5),
        'fouls_for_l5': _mean(records, 'fouls_for', 5),
        'fouls_against_l5': _mean(records, 'fouls_against', 5),
        'yellow_cards_l5': yellow_l5,
        'red_cards_l10': red_l10,
        'cards_total_l5': cards_total_l5,
    }


def build_team_rolling_features(fixtures_csv: str, team_stats_csv: str, output_csv: str = str(TARGET_PATH)) -> pd.DataFrame:
    fixtures = pd.read_csv(fixtures_csv)
    team_stats = pd.read_csv(team_stats_csv)
    # Provider-normalized CSVs can occasionally carry duplicate labels after
    # upstream joins/retries; strip them before we rely on scalar row fields.
    fixtures = fixtures.loc[:, ~fixtures.columns.duplicated()].copy()
    team_stats = team_stats.loc[:, ~team_stats.columns.duplicated()].copy()
    fixtures['kickoff_ts_utc'] = pd.to_datetime(fixtures['kickoff_ts_utc'], errors='coerce', utc=True)
    merged = team_stats.merge(
        fixtures[['fixture_id', 'fixture_key', 'league', 'league_id', 'season', 'match_date', 'home_team_id', 'away_team_id', 'home_team_name', 'away_team_name', 'kickoff_ts_utc']],
        on='fixture_id', how='left'
    )
    merged = merged.loc[:, ~merged.columns.duplicated()].copy()
    merged['is_home'] = pd.to_numeric(merged['is_home'], errors='coerce').fillna(0).astype(int)
    merged['home_team_id'] = pd.to_numeric(merged['home_team_id'], errors='coerce')
    merged['away_team_id'] = pd.to_numeric(merged['away_team_id'], errors='coerce')
    home_mask = merged['is_home'] == 1
    merged['opponent_team_id'] = merged['home_team_id'].where(~home_mask, merged['away_team_id']).astype('Int64')
    merged['opponent_team_name'] = merged['home_team_name'].where(~home_mask, merged['away_team_name'])
    merged['points'] = merged.apply(lambda r: 3 if r['goals_for'] > r['goals_against'] else (1 if r['goals_for'] == r['goals_against'] else 0), axis=1)
    merged['goal_diff'] = merged['goals_for'] - merged['goals_against']
    merged['total_goals'] = merged['goals_for'] + merged['goals_against']
    merged['btts'] = ((merged['goals_for'] > 0) & (merged['goals_against'] > 0)).astype(int)
    merged['cards_total'] = merged['yellow_cards'] + (2 * merged['red_cards'])

    opp = merged[['fixture_id', 'team_id', 'corners_for', 'fouls_for']].rename(columns={
        'team_id': 'opponent_team_id', 'corners_for': 'corners_against', 'fouls_for': 'fouls_against'
    })
    merged = merged.merge(opp, on=['fixture_id', 'opponent_team_id'], how='left')
    merged = merged.sort_values(['kickoff_ts_utc', 'fixture_id', 'team_id']).reset_index(drop=True)

    history: dict[int, list[dict]] = defaultdict(list)
    out_rows = []
    for _, fx in fixtures.sort_values(['kickoff_ts_utc', 'fixture_id']).iterrows():
        home_history = list(reversed(history.get(int(fx['home_team_id']), [])))
        away_history = list(reversed(history.get(int(fx['away_team_id']), [])))
        h = _team_metrics(home_history)
        a = _team_metrics(away_history)
        row = {key: fx[key] for key in FEATURE_SCHEMAS['api_team_rolling_features'] if key in fx.index}
        row.update({
            'home_team_ppg_l3': h['ppg_l3'], 'home_team_ppg_l5': h['ppg_l5'], 'home_team_ppg_l10': h['ppg_l10'], 'home_team_ppg_season': h['ppg_season'],
            'away_team_ppg_l3': a['ppg_l3'], 'away_team_ppg_l5': a['ppg_l5'], 'away_team_ppg_l10': a['ppg_l10'], 'away_team_ppg_season': a['ppg_season'],
            'home_team_win_rate_l5': h['win_rate_l5'], 'home_team_draw_rate_l5': h['draw_rate_l5'], 'home_team_loss_rate_l5': h['loss_rate_l5'],
            'away_team_win_rate_l5': a['win_rate_l5'], 'away_team_draw_rate_l5': a['draw_rate_l5'], 'away_team_loss_rate_l5': a['loss_rate_l5'],
            'home_team_points_weighted_l5': h['points_weighted_l5'], 'away_team_points_weighted_l5': a['points_weighted_l5'],
            'ppg_diff_l5': h['ppg_l5'] - a['ppg_l5'], 'ppg_diff_l10': h['ppg_l10'] - a['ppg_l10'], 'ppg_diff_season': h['ppg_season'] - a['ppg_season'],
            'form_points_diff_weighted_l5': h['points_weighted_l5'] - a['points_weighted_l5'],
            'home_goals_for_l3': h['goals_for_l3'], 'home_goals_for_l5': h['goals_for_l5'], 'home_goals_for_l10': h['goals_for_l10'],
            'home_goals_against_l3': h['goals_against_l3'], 'home_goals_against_l5': h['goals_against_l5'], 'home_goals_against_l10': h['goals_against_l10'],
            'away_goals_for_l3': a['goals_for_l3'], 'away_goals_for_l5': a['goals_for_l5'], 'away_goals_for_l10': a['goals_for_l10'],
            'away_goals_against_l3': a['goals_against_l3'], 'away_goals_against_l5': a['goals_against_l5'], 'away_goals_against_l10': a['goals_against_l10'],
            'home_goal_diff_l5': h['goal_diff_l5'], 'away_goal_diff_l5': a['goal_diff_l5'], 'goal_diff_delta_l5': h['goal_diff_l5'] - a['goal_diff_l5'],
            'home_total_goals_l5': h['total_goals_l5'], 'away_total_goals_l5': a['total_goals_l5'], 'combined_total_goals_l5': h['total_goals_l5'] + a['total_goals_l5'],
            'home_over05_rate_l5': h['over05_rate_l5'], 'home_over15_rate_l5': h['over15_rate_l5'], 'home_over25_rate_l5': h['over25_rate_l5'], 'home_over35_rate_l5': h['over35_rate_l5'], 'home_over45_rate_l5': h['over45_rate_l5'],
            'away_over05_rate_l5': a['over05_rate_l5'], 'away_over15_rate_l5': a['over15_rate_l5'], 'away_over25_rate_l5': a['over25_rate_l5'], 'away_over35_rate_l5': a['over35_rate_l5'], 'away_over45_rate_l5': a['over45_rate_l5'],
            'combined_over15_rate_l5': (h['over15_rate_l5'] + a['over15_rate_l5']) / 2.0,
            'combined_over25_rate_l5': (h['over25_rate_l5'] + a['over25_rate_l5']) / 2.0,
            'combined_over35_rate_l5': (h['over35_rate_l5'] + a['over35_rate_l5']) / 2.0,
            'combined_over45_rate_l5': (h['over45_rate_l5'] + a['over45_rate_l5']) / 2.0,
            'home_btts_rate_l5': h['btts_rate_l5'], 'home_btts_rate_l10': h['btts_rate_l10'], 'away_btts_rate_l5': a['btts_rate_l5'], 'away_btts_rate_l10': a['btts_rate_l10'], 'combined_btts_rate_l5': (h['btts_rate_l5'] + a['btts_rate_l5']) / 2.0,
            'home_clean_sheet_rate_l5': h['clean_sheet_rate_l5'], 'away_clean_sheet_rate_l5': a['clean_sheet_rate_l5'], 'home_fts_rate_l5': h['fts_rate_l5'], 'away_fts_rate_l5': a['fts_rate_l5'],
            'home_scored_rate_l5': h['scored_rate_l5'], 'away_scored_rate_l5': a['scored_rate_l5'], 'home_conceded_rate_l5': h['conceded_rate_l5'], 'away_conceded_rate_l5': a['conceded_rate_l5'],
            'home_ht_goals_for_l5': h['ht_goals_for_l5'], 'home_ht_goals_against_l5': h['ht_goals_against_l5'], 'away_ht_goals_for_l5': a['ht_goals_for_l5'], 'away_ht_goals_against_l5': a['ht_goals_against_l5'],
            'home_ht_leading_rate_l5': h['ht_leading_rate_l5'], 'home_ht_drawing_rate_l5': h['ht_drawing_rate_l5'], 'home_ht_losing_rate_l5': h['ht_losing_rate_l5'],
            'away_ht_leading_rate_l5': a['ht_leading_rate_l5'], 'away_ht_drawing_rate_l5': a['ht_drawing_rate_l5'], 'away_ht_losing_rate_l5': a['ht_losing_rate_l5'],
            'combined_ht_over05_rate_l5': (h['ht_over05_rate_l5'] + a['ht_over05_rate_l5']) / 2.0,
            'combined_ht_over15_rate_l5': (h['ht_over15_rate_l5'] + a['ht_over15_rate_l5']) / 2.0,
            'combined_ht_over25_rate_l5': (h['ht_over25_rate_l5'] + a['ht_over25_rate_l5']) / 2.0,
            'home_shots_l5': h['shots_l5'], 'home_sot_l5': h['sot_l5'], 'home_shot_accuracy_l5': h['shot_accuracy_l5'], 'home_shots_inside_box_l5': h['shots_inside_box_l5'], 'home_shots_outside_box_l5': h['shots_outside_box_l5'], 'home_blocked_shots_l5': h['blocked_shots_l5'],
            'away_shots_l5': a['shots_l5'], 'away_sot_l5': a['sot_l5'], 'away_shot_accuracy_l5': a['shot_accuracy_l5'], 'away_shots_inside_box_l5': a['shots_inside_box_l5'], 'away_shots_outside_box_l5': a['shots_outside_box_l5'], 'away_blocked_shots_l5': a['blocked_shots_l5'],
            'shot_delta_l5': h['shots_l5'] - a['shots_l5'], 'sot_delta_l5': h['sot_l5'] - a['sot_l5'], 'shot_accuracy_delta_l5': h['shot_accuracy_l5'] - a['shot_accuracy_l5'], 'box_shot_delta_l5': h['shots_inside_box_l5'] - a['shots_inside_box_l5'],
            'home_possession_l5': h['possession_l5'], 'home_passes_l5': h['passes_l5'], 'home_pass_accuracy_l5': h['pass_accuracy_l5'], 'home_progressive_proxy_l5': h['progressive_proxy_l5'],
            'away_possession_l5': a['possession_l5'], 'away_passes_l5': a['passes_l5'], 'away_pass_accuracy_l5': a['pass_accuracy_l5'], 'away_progressive_proxy_l5': a['progressive_proxy_l5'],
            'possession_delta_l5': h['possession_l5'] - a['possession_l5'], 'pass_accuracy_delta_l5': h['pass_accuracy_l5'] - a['pass_accuracy_l5'],
            'home_corners_for_l5': h['corners_for_l5'], 'home_corners_against_l5': h['corners_against_l5'], 'away_corners_for_l5': a['corners_for_l5'], 'away_corners_against_l5': a['corners_against_l5'],
            'corner_delta_l5': h['corners_for_l5'] - a['corners_for_l5'], 'combined_corners_l5': h['corners_for_l5'] + a['corners_for_l5'], 'home_corner_pressure_l5': h['corner_pressure_l5'], 'away_corner_pressure_l5': a['corner_pressure_l5'],
            'home_fouls_for_l5': h['fouls_for_l5'], 'home_fouls_against_l5': h['fouls_against_l5'], 'home_yellow_cards_l5': h['yellow_cards_l5'], 'home_red_cards_l10': h['red_cards_l10'], 'home_cards_total_l5': h['cards_total_l5'],
            'away_fouls_for_l5': a['fouls_for_l5'], 'away_fouls_against_l5': a['fouls_against_l5'], 'away_yellow_cards_l5': a['yellow_cards_l5'], 'away_red_cards_l10': a['red_cards_l10'], 'away_cards_total_l5': a['cards_total_l5'],
            'foul_delta_l5': h['fouls_for_l5'] - a['fouls_for_l5'], 'card_delta_l5': h['cards_total_l5'] - a['cards_total_l5'],
            'combined_card_pressure_l5': h['cards_total_l5'] + a['cards_total_l5'], 'combined_foul_pressure_l5': h['fouls_for_l5'] + a['fouls_for_l5'],
        })
        out_rows.append(row)
        fx_rows = merged[merged['fixture_id'] == fx['fixture_id']]
        for _, tm in fx_rows.iterrows():
            history[int(tm['team_id'])].append(tm.to_dict())

    df = pd.DataFrame(out_rows)
    df = df.reindex(columns=FEATURE_SCHEMAS['api_team_rolling_features'])
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=PURPOSE)
    parser.add_argument('--write-stub', action='store_true', help='Write the scaffold output even though live transform logic is not implemented yet.')
    parser.add_argument('--fixtures-csv', default=str(NORMALIZED_FILES['fixtures_master']))
    parser.add_argument('--team-stats-csv', default=str(NORMALIZED_FILES['match_team_stats']))
    parser.add_argument('--output-csv', default=str(TARGET_PATH))
    args = parser.parse_args()
    if args.write_stub:
        df = build_stub()
        print(f'WROTE STUB: {TARGET_PATH} rows={len(df)}')
        return
    df = build_team_rolling_features(args.fixtures_csv, args.team_stats_csv, args.output_csv)
    print(f'WROTE: {args.output_csv} rows={len(df)}')


if __name__ == '__main__':
    main()
