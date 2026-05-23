#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.api_football.hybrid_training_utils import ensure_columns_exist, load_training_frame

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAINING_CSV = PROJECT_ROOT / 'data_sources' / 'hybrid' / 'hybrid_match_training__England_Premier_League.csv'
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / 'data_sources' / 'hybrid' / 'hybrid_goal_mass_inputs__England_Premier_League.csv'

BASE_COLS = [
    'fixture_id', 'fixture_key', 'league', 'season', 'match_date', 'home_team_name', 'away_team_name',
    'Pre-Match PPG (Home)', 'Pre-Match PPG (Away)', 'home_ppg', 'away_ppg',
    'team_a_xg', 'team_b_xg', 'average_goals_per_match_pre_match',
    'btts_percentage_pre_match', 'over_25_percentage_pre_match',
    'odds_ft_home_team_win', 'odds_ft_draw', 'odds_ft_away_team_win', 'odds_ft_over25', 'odds_btts_yes',
]
TARGET_COLS = [
    'home_team_goal_count', 'away_team_goal_count',
    'target_home_goals_over15', 'target_away_goals_over15',
    'target_home_fts', 'target_away_fts',
    'target_btts_yes', 'target_btts_first_half', 'target_ou25_over'
]
LEAKAGE_ONLY_COLS = [
    'status',
    'home_team_goal_count_half_time', 'away_team_goal_count_half_time',
    'target_ftr_home', 'target_ftr_draw', 'target_ftr_away',
]


def build_goal_mass_inputs(training_csv: Path, output_csv: Path) -> pd.DataFrame:
    df = load_training_frame(training_csv)
    keep_targets = ensure_columns_exist(TARGET_COLS, df)
    exclude_for_features = set(TARGET_COLS + LEAKAGE_ONLY_COLS)
    passthrough_cols = [c for c in df.columns if c not in exclude_for_features]

    # Preserve a stable front-of-table order for identity and core pre-match fields,
    # then carry through the rest of the hybrid feature estate so lambda can learn
    # from the same richer families as the core Cat/XGB stack.
    ordered = []
    for col in BASE_COLS:
        if col in passthrough_cols and col not in ordered:
            ordered.append(col)
    for col in passthrough_cols:
        if col not in ordered:
            ordered.append(col)
    for col in keep_targets:
        if col not in ordered:
            ordered.append(col)

    out = df[ordered].copy()

    def _num(col: str) -> pd.Series:
        if col not in out.columns:
            return pd.Series(0.0, index=out.index, dtype=float)
        return pd.to_numeric(out[col], errors='coerce')

    out['home_attack_pressure_index'] = (
        _num('home_shots_l5').fillna(0) * 0.35 +
        _num('home_sot_l5').fillna(0) * 0.35 +
        _num('home_shots_inside_box_l5').fillna(0) * 0.15 +
        _num('home_starting_xi_shots_per90_l5').fillna(0) * 0.15
    )
    out['away_attack_pressure_index'] = (
        _num('away_shots_l5').fillna(0) * 0.35 +
        _num('away_sot_l5').fillna(0) * 0.35 +
        _num('away_shots_inside_box_l5').fillna(0) * 0.15 +
        _num('away_starting_xi_shots_per90_l5').fillna(0) * 0.15
    )
    out['match_pressure_delta'] = out['home_attack_pressure_index'] - out['away_attack_pressure_index']

    out['home_lambda_seed'] = (
        _num('team_a_xg').fillna(0) * 0.30 +
        _num('home_goals_for_l5').fillna(0) * 0.25 +
        _num('away_goals_against_l5').fillna(0) * 0.20 +
        _num('home_starting_xi_goals_per90_l5').fillna(0) * 0.15 +
        _num('bookie_home_prob_norm').fillna(0) * 0.10 * 3.0
    )
    out['away_lambda_seed'] = (
        _num('team_b_xg').fillna(0) * 0.30 +
        _num('away_goals_for_l5').fillna(0) * 0.25 +
        _num('home_goals_against_l5').fillna(0) * 0.20 +
        _num('away_starting_xi_goals_per90_l5').fillna(0) * 0.15 +
        _num('bookie_away_prob_norm').fillna(0) * 0.10 * 3.0
    )
    out['lambda_total_seed'] = out['home_lambda_seed'] + out['away_lambda_seed']

    out['home_absence_attack_penalty'] = _num('home_absence_severity_score').fillna(0) + _num('home_missing_goals_per90_l5').fillna(0)
    out['away_absence_attack_penalty'] = _num('away_absence_severity_score').fillna(0) + _num('away_missing_goals_per90_l5').fillna(0)
    out['home_absence_defensive_penalty'] = _num('home_absence_severity_score').fillna(0) + _num('home_missing_tackles_per90_l5').fillna(0)
    out['away_absence_defensive_penalty'] = _num('away_absence_severity_score').fillna(0) + _num('away_missing_tackles_per90_l5').fillna(0)

    out['target_home_goals'] = _num('home_team_goal_count').fillna(0).astype(int)
    out['target_away_goals'] = _num('away_team_goal_count').fillna(0).astype(int)
    out['target_total_goals'] = out['target_home_goals'] + out['target_away_goals']

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description='Build hybrid goal-mass / lambda training inputs from hybrid match training data.')
    parser.add_argument('--training-csv', default=str(DEFAULT_TRAINING_CSV))
    parser.add_argument('--output-csv', default=str(DEFAULT_OUTPUT_CSV))
    args = parser.parse_args()
    df = build_goal_mass_inputs(Path(args.training_csv), Path(args.output_csv))
    print(f'WROTE: {args.output_csv} rows={len(df)} cols={len(df.columns)}')


if __name__ == '__main__':
    main()
