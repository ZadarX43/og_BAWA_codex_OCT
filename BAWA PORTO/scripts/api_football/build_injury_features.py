from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .paths import FEATURE_FILES, NORMALIZED_FILES
from .schema_contracts import FEATURE_SCHEMAS
from .scaffold import build_csv_stub
from .utils import safe_div

PURPOSE = 'Build fixture-level injury and suspension severity features.'
TARGET_PATH = FEATURE_FILES['api_injury_features']


def build_stub() -> pd.DataFrame:
    return build_csv_stub(TARGET_PATH, FEATURE_SCHEMAS['api_injury_features'], PURPOSE, placeholder_row=False)


def _role_from_position(position: object) -> str:
    text = str(position or '').upper().strip()
    if text.startswith('G'):
        return 'goalkeepers'
    if text.startswith('D'):
        return 'defenders'
    if text.startswith('M'):
        return 'midfielders'
    if text.startswith('F'):
        return 'attackers'
    return 'unknown'


def _player_hist_metrics(records: list[dict]) -> dict[str, float]:
    sample5 = records[:5]
    sample10 = records[:10]
    minutes5 = sum(float(r.get('minutes', 0.0) or 0.0) for r in sample5)
    return {
        'minutes_l5_total': minutes5,
        'goals_per90_l5': safe_div(sum(float(r.get('goals', 0.0) or 0.0) for r in sample5) * 90.0, minutes5),
        'assists_per90_l5': safe_div(sum(float(r.get('assists', 0.0) or 0.0) for r in sample5) * 90.0, minutes5),
        'tackles_per90_l5': safe_div(sum(float(r.get('tackles', 0.0) or 0.0) for r in sample5) * 90.0, minutes5),
        'goalkeeper_flag': int(any(_role_from_position(r.get('position')) == 'goalkeepers' for r in sample10)),
        'role': _role_from_position(sample10[0].get('position')) if sample10 else 'unknown',
    }


def _team_absence_features(absences: pd.DataFrame, player_history: dict[int, list[dict]]) -> dict[str, float]:
    out = {
        'injured_players_count': 0, 'suspended_players_count': 0,
        'missing_defenders_count': 0, 'missing_midfielders_count': 0, 'missing_attackers_count': 0, 'missing_goalkeepers_count': 0,
        'missing_minutes_l5_total': 0.0, 'missing_goals_per90_l5': 0.0, 'missing_assists_per90_l5': 0.0, 'missing_tackles_per90_l5': 0.0,
        'absence_severity_score': 0.0,
    }
    if absences.empty:
        return out
    for _, row in absences.iterrows():
        absence_type = str(row.get('absence_type') or '').lower()
        if 'susp' in absence_type:
            out['suspended_players_count'] += 1
        else:
            out['injured_players_count'] += 1
        metrics = _player_hist_metrics(list(reversed(player_history.get(int(row['player_id']), []))))
        role = metrics.get('role')
        if role == 'defenders':
            out['missing_defenders_count'] += 1
        elif role == 'midfielders':
            out['missing_midfielders_count'] += 1
        elif role == 'attackers':
            out['missing_attackers_count'] += 1
        elif role == 'goalkeepers':
            out['missing_goalkeepers_count'] += 1
        out['missing_minutes_l5_total'] += metrics['minutes_l5_total']
        out['missing_goals_per90_l5'] += metrics['goals_per90_l5']
        out['missing_assists_per90_l5'] += metrics['assists_per90_l5']
        out['missing_tackles_per90_l5'] += metrics['tackles_per90_l5']
        out['absence_severity_score'] += (
            metrics['minutes_l5_total'] * 0.25 +
            metrics['goals_per90_l5'] * 2.0 +
            metrics['assists_per90_l5'] * 1.5 +
            metrics['tackles_per90_l5'] * 0.75 +
            metrics['goalkeeper_flag'] * 2.5
        )
    return out


def build_injury_features(fixtures_csv: str, injuries_csv: str, player_stats_csv: str, output_csv: str = str(TARGET_PATH)) -> pd.DataFrame:
    fixtures = pd.read_csv(fixtures_csv)
    injuries = pd.read_csv(injuries_csv)
    player_stats = pd.read_csv(player_stats_csv)
    fixtures['kickoff_ts_utc'] = pd.to_datetime(fixtures['kickoff_ts_utc'], errors='coerce', utc=True)
    player_stats = player_stats.merge(fixtures[['fixture_id', 'kickoff_ts_utc']], on='fixture_id', how='left')
    player_stats = player_stats.sort_values(['kickoff_ts_utc', 'fixture_id', 'player_id']).reset_index(drop=True)
    history: dict[int, list[dict]] = defaultdict(list)
    stats_by_fixture: dict[int, list[dict]] = defaultdict(list)
    for _, row in player_stats.iterrows():
        stats_by_fixture[int(row['fixture_id'])].append(row.to_dict())

    out_rows = []
    injuries_by_fixture = {fid: df.copy() for fid, df in injuries.groupby('fixture_id')} if not injuries.empty else {}
    for _, fx in fixtures.sort_values(['kickoff_ts_utc', 'fixture_id']).iterrows():
        fx_inj = injuries_by_fixture.get(int(fx['fixture_id']), pd.DataFrame(columns=injuries.columns))
        home_abs = _team_absence_features(fx_inj[fx_inj['team_id'] == fx['home_team_id']], history)
        away_abs = _team_absence_features(fx_inj[fx_inj['team_id'] == fx['away_team_id']], history)
        out_rows.append({
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
            'home_injured_players_count': home_abs['injured_players_count'],
            'away_injured_players_count': away_abs['injured_players_count'],
            'home_suspended_players_count': home_abs['suspended_players_count'],
            'away_suspended_players_count': away_abs['suspended_players_count'],
            'home_missing_defenders_count': home_abs['missing_defenders_count'],
            'away_missing_defenders_count': away_abs['missing_defenders_count'],
            'home_missing_midfielders_count': home_abs['missing_midfielders_count'],
            'away_missing_midfielders_count': away_abs['missing_midfielders_count'],
            'home_missing_attackers_count': home_abs['missing_attackers_count'],
            'away_missing_attackers_count': away_abs['missing_attackers_count'],
            'home_missing_goalkeepers_count': home_abs['missing_goalkeepers_count'],
            'away_missing_goalkeepers_count': away_abs['missing_goalkeepers_count'],
            'home_missing_minutes_l5_total': home_abs['missing_minutes_l5_total'],
            'away_missing_minutes_l5_total': away_abs['missing_minutes_l5_total'],
            'home_missing_goals_per90_l5': home_abs['missing_goals_per90_l5'],
            'away_missing_goals_per90_l5': away_abs['missing_goals_per90_l5'],
            'home_missing_assists_per90_l5': home_abs['missing_assists_per90_l5'],
            'away_missing_assists_per90_l5': away_abs['missing_assists_per90_l5'],
            'home_missing_tackles_per90_l5': home_abs['missing_tackles_per90_l5'],
            'away_missing_tackles_per90_l5': away_abs['missing_tackles_per90_l5'],
            'home_absence_severity_score': home_abs['absence_severity_score'],
            'away_absence_severity_score': away_abs['absence_severity_score'],
            'absence_severity_delta': home_abs['absence_severity_score'] - away_abs['absence_severity_score'],
        })
        for rec in stats_by_fixture.get(int(fx['fixture_id']), []):
            history[int(rec['player_id'])].append(rec)

    df = pd.DataFrame(out_rows)
    if df.empty:
        df = pd.DataFrame(columns=FEATURE_SCHEMAS['api_injury_features'])
    else:
        df = df.reindex(columns=FEATURE_SCHEMAS['api_injury_features'])
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=PURPOSE)
    parser.add_argument('--write-stub', action='store_true', help='Write the scaffold output even though live transform logic is not implemented yet.')
    parser.add_argument('--fixtures-csv', default=str(NORMALIZED_FILES['fixtures_master']))
    parser.add_argument('--injuries-csv', default=str(NORMALIZED_FILES['injuries']))
    parser.add_argument('--player-stats-csv', default=str(NORMALIZED_FILES['match_player_stats']))
    parser.add_argument('--output-csv', default=str(TARGET_PATH))
    args = parser.parse_args()
    if args.write_stub:
        df = build_stub()
        print(f'WROTE STUB: {TARGET_PATH} rows={len(df)}')
        return
    df = build_injury_features(args.fixtures_csv, args.injuries_csv, args.player_stats_csv, args.output_csv)
    print(f'WROTE: {args.output_csv} rows={len(df)}')


if __name__ == '__main__':
    main()
