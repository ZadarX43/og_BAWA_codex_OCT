from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .paths import FEATURE_FILES, NORMALIZED_FILES
from .schema_contracts import FEATURE_SCHEMAS
from .scaffold import build_csv_stub
from .utils import safe_div

PURPOSE = 'Build lineup and formation-derived fixture features.'
TARGET_PATH = FEATURE_FILES['api_lineup_features']


def build_stub() -> pd.DataFrame:
    return build_csv_stub(TARGET_PATH, FEATURE_SCHEMAS['api_lineup_features'], PURPOSE, placeholder_row=False)


def _parse_formation(text: object) -> tuple[int, int, int]:
    parts = [p for p in str(text or '').split('-') if p.isdigit()]
    nums = [int(p) for p in parts]
    if not nums:
        return 0, 0, 0
    if len(nums) == 1:
        return nums[0], 0, 0
    defenders = nums[0]
    forwards = nums[-1]
    midfielders = sum(nums[1:-1]) if len(nums) > 2 else 0
    return defenders, midfielders, forwards


def _formation_features(formation: object) -> dict[str, float]:
    defenders, midfielders, forwards = _parse_formation(formation)
    attacking_mid_proxy = max(0, midfielders - 2)
    defensive_mid_proxy = 1 if midfielders >= 4 else 0
    attacking_shape_score = float(forwards + (0.5 * attacking_mid_proxy) + (0.5 if forwards >= 3 else 0.0) - (0.5 if forwards <= 1 else 0.0))
    defensive_shape_score = float(defenders + defensive_mid_proxy + (1.0 if defenders >= 5 else 0.0))
    return {
        'backline_count': defenders,
        'midfield_count': midfielders,
        'forward_count': forwards,
        'attacking_shape_score': attacking_shape_score,
        'defensive_shape_score': defensive_shape_score,
    }


def _mean(records, key, n):
    sample = records[:n]
    return sum(float(r.get(key, 0.0) or 0.0) for r in sample) / len(sample) if sample else 0.0


def _sum(records, key, n):
    sample = records[:n]
    return sum(float(r.get(key, 0.0) or 0.0) for r in sample) if sample else 0.0


def _player_history_metrics(records: list[dict]) -> dict[str, float]:
    recs5 = records[:5]
    recs10 = records[:10]
    minutes5 = _sum(recs5, 'minutes', 5)
    return {
        'avg_rating_l5': _mean(recs5, 'rating', 5),
        'minutes_l5': _mean(recs5, 'minutes', 5),
        'goals_per90_l5': safe_div(_sum(recs5, 'goals', 5) * 90.0, minutes5),
        'assists_per90_l5': safe_div(_sum(recs5, 'assists', 5) * 90.0, minutes5),
        'shots_per90_l5': safe_div(_sum(recs5, 'shots_total', 5) * 90.0, minutes5),
        'sot_per90_l5': safe_div(_sum(recs5, 'shots_on_target', 5) * 90.0, minutes5),
        'tackles_per90_l5': safe_div(_sum(recs5, 'tackles', 5) * 90.0, minutes5),
        'fouls_committed_per90_l5': safe_div(_sum(recs5, 'fouls_committed', 5) * 90.0, minutes5),
        'cards_per90_l10': safe_div((_sum(recs10, 'yellow_cards', 10) + _sum(recs10, 'red_cards', 10)) * 90.0, _sum(recs10, 'minutes', 10)),
    }


def _aggregate_xi(player_ids: list[int], history: dict[int, list[dict]]) -> dict[str, float]:
    metrics = [
        _player_history_metrics(list(reversed(history.get(pid, []))))
        for pid in player_ids
    ]
    if not metrics:
        metrics = [{}]
    def avg(key: str) -> float:
        return sum(float(m.get(key, 0.0) or 0.0) for m in metrics) / len(metrics)
    return {
        'starting_xi_avg_rating_l5': avg('avg_rating_l5'),
        'starting_xi_minutes_l5': avg('minutes_l5'),
        'starting_xi_goals_per90_l5': avg('goals_per90_l5'),
        'starting_xi_assists_per90_l5': avg('assists_per90_l5'),
        'starting_xi_shots_per90_l5': avg('shots_per90_l5'),
        'starting_xi_sot_per90_l5': avg('sot_per90_l5'),
        'starting_xi_tackles_per90_l5': avg('tackles_per90_l5'),
        'starting_xi_fouls_committed_per90_l5': avg('fouls_committed_per90_l5'),
        'starting_xi_cards_per90_l10': avg('cards_per90_l10'),
    }


def _mode_or_blank(series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return ''
    mode = non_null.astype(str).mode()
    return str(mode.iloc[0]) if not mode.empty else ''


def build_lineup_features(fixtures_csv: str, lineups_csv: str, player_stats_csv: str, output_csv: str = str(TARGET_PATH)) -> pd.DataFrame:
    fixtures = pd.read_csv(fixtures_csv)
    lineups = pd.read_csv(lineups_csv)
    player_stats = pd.read_csv(player_stats_csv)
    fixtures['kickoff_ts_utc'] = pd.to_datetime(fixtures['kickoff_ts_utc'], errors='coerce', utc=True)
    ps = player_stats.merge(fixtures[['fixture_id', 'kickoff_ts_utc']], on='fixture_id', how='left')
    ps = ps.sort_values(['kickoff_ts_utc', 'fixture_id', 'player_id']).reset_index(drop=True)

    history: dict[int, list[dict]] = defaultdict(list)
    stats_by_fixture: dict[int, list[dict]] = defaultdict(list)
    for _, row in ps.iterrows():
        stats_by_fixture[int(row['fixture_id'])].append(row.to_dict())

    out_rows = []
    for _, fx in fixtures.sort_values(['kickoff_ts_utc', 'fixture_id']).iterrows():
        fx_lineups = lineups[(lineups['fixture_id'] == fx['fixture_id']) & (lineups['is_starting_xi'] == 1)]
        home = fx_lineups[fx_lineups['team_id'] == fx['home_team_id']]
        away = fx_lineups[fx_lineups['team_id'] == fx['away_team_id']]
        home_formation = _mode_or_blank(home['formation']) if not home.empty else ''
        away_formation = _mode_or_blank(away['formation']) if not away.empty else ''
        hf = _formation_features(home_formation)
        af = _formation_features(away_formation)
        hagg = _aggregate_xi([int(x) for x in home['player_id'].tolist()], history)
        aagg = _aggregate_xi([int(x) for x in away['player_id'].tolist()], history)
        row = {
            'fixture_id': int(fx['fixture_id']),
            'fixture_key': fx['fixture_key'],
            'league': fx['league'],
            'league_id': int(fx['league_id']),
            'season': int(fx['season']),
            'match_date': fx['match_date'],
            'home_team_id': int(fx['home_team_id']),
            'away_team_id': int(fx['away_team_id']),
            'home_team_name': fx['home_team_name'],
            'away_team_name': fx['away_team_name'],
            'home_formation': home_formation,
            'away_formation': away_formation,
            'same_formation_flag': int(home_formation != '' and home_formation == away_formation),
            'formation_mismatch_flag': int(home_formation != away_formation),
            'home_backline_count': hf['backline_count'], 'away_backline_count': af['backline_count'],
            'home_midfield_count': hf['midfield_count'], 'away_midfield_count': af['midfield_count'],
            'home_forward_count': hf['forward_count'], 'away_forward_count': af['forward_count'],
            'home_attacking_shape_score': hf['attacking_shape_score'], 'away_attacking_shape_score': af['attacking_shape_score'],
            'home_defensive_shape_score': hf['defensive_shape_score'], 'away_defensive_shape_score': af['defensive_shape_score'],
            'formation_attack_delta': hf['attacking_shape_score'] - af['attacking_shape_score'],
            'formation_defence_delta': hf['defensive_shape_score'] - af['defensive_shape_score'],
            'home_starting_xi_avg_rating_l5': hagg['starting_xi_avg_rating_l5'], 'away_starting_xi_avg_rating_l5': aagg['starting_xi_avg_rating_l5'],
            'home_starting_xi_minutes_l5': hagg['starting_xi_minutes_l5'], 'away_starting_xi_minutes_l5': aagg['starting_xi_minutes_l5'],
            'home_starting_xi_goals_per90_l5': hagg['starting_xi_goals_per90_l5'], 'away_starting_xi_goals_per90_l5': aagg['starting_xi_goals_per90_l5'],
            'home_starting_xi_assists_per90_l5': hagg['starting_xi_assists_per90_l5'], 'away_starting_xi_assists_per90_l5': aagg['starting_xi_assists_per90_l5'],
            'home_starting_xi_shots_per90_l5': hagg['starting_xi_shots_per90_l5'], 'away_starting_xi_shots_per90_l5': aagg['starting_xi_shots_per90_l5'],
            'home_starting_xi_sot_per90_l5': hagg['starting_xi_sot_per90_l5'], 'away_starting_xi_sot_per90_l5': aagg['starting_xi_sot_per90_l5'],
            'home_starting_xi_tackles_per90_l5': hagg['starting_xi_tackles_per90_l5'], 'away_starting_xi_tackles_per90_l5': aagg['starting_xi_tackles_per90_l5'],
            'home_starting_xi_fouls_committed_per90_l5': hagg['starting_xi_fouls_committed_per90_l5'], 'away_starting_xi_fouls_committed_per90_l5': aagg['starting_xi_fouls_committed_per90_l5'],
            'home_starting_xi_cards_per90_l10': hagg['starting_xi_cards_per90_l10'], 'away_starting_xi_cards_per90_l10': aagg['starting_xi_cards_per90_l10'],
            'xi_rating_delta': hagg['starting_xi_avg_rating_l5'] - aagg['starting_xi_avg_rating_l5'],
            'xi_minutes_delta': hagg['starting_xi_minutes_l5'] - aagg['starting_xi_minutes_l5'],
            'xi_goal_power_delta': hagg['starting_xi_goals_per90_l5'] - aagg['starting_xi_goals_per90_l5'],
            'xi_shot_power_delta': hagg['starting_xi_shots_per90_l5'] - aagg['starting_xi_shots_per90_l5'],
            'xi_sot_power_delta': hagg['starting_xi_sot_per90_l5'] - aagg['starting_xi_sot_per90_l5'],
            'xi_tackle_pressure_delta': hagg['starting_xi_tackles_per90_l5'] - aagg['starting_xi_tackles_per90_l5'],
            'xi_card_risk_delta': hagg['starting_xi_cards_per90_l10'] - aagg['starting_xi_cards_per90_l10'],
        }
        out_rows.append(row)
        for rec in stats_by_fixture.get(int(fx['fixture_id']), []):
            history[int(rec['player_id'])].append(rec)

    df = pd.DataFrame(out_rows)
    df = df.reindex(columns=FEATURE_SCHEMAS['api_lineup_features'])
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=PURPOSE)
    parser.add_argument('--write-stub', action='store_true', help='Write the scaffold output even though live transform logic is not implemented yet.')
    parser.add_argument('--fixtures-csv', default=str(NORMALIZED_FILES['fixtures_master']))
    parser.add_argument('--lineups-csv', default=str(NORMALIZED_FILES['lineups']))
    parser.add_argument('--player-stats-csv', default=str(NORMALIZED_FILES['match_player_stats']))
    parser.add_argument('--output-csv', default=str(TARGET_PATH))
    args = parser.parse_args()
    if args.write_stub:
        df = build_stub()
        print(f'WROTE STUB: {TARGET_PATH} rows={len(df)}')
        return
    df = build_lineup_features(args.fixtures_csv, args.lineups_csv, args.player_stats_csv, args.output_csv)
    print(f'WROTE: {args.output_csv} rows={len(df)}')


if __name__ == '__main__':
    main()
