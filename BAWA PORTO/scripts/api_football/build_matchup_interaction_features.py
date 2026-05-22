from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .paths import FEATURE_FILES

PURPOSE = 'Build matchup interaction features from team identity and enriched pre-match context.'
TARGET_PATH = FEATURE_FILES['api_matchup_interaction_features']
KEYS = [
    'fixture_id', 'fixture_key', 'league', 'league_id', 'season', 'match_date',
    'home_team_id', 'away_team_id', 'home_team_name', 'away_team_name',
]


def _to_num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype='float64')
    return pd.to_numeric(df[col], errors='coerce').fillna(default)


def _scaled(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce').fillna(0.0)
    mn = float(s.min()) if len(s) else 0.0
    mx = float(s.max()) if len(s) else 0.0
    if mx - mn <= 1e-12:
        return pd.Series(0.5, index=s.index, dtype='float64')
    return ((s - mn) / (mx - mn)).clip(0.0, 1.0)


def build_matchup_interaction_features(identity_csv: str, enriched_csv: str, output_csv: str = str(TARGET_PATH)) -> pd.DataFrame:
    identity = pd.read_csv(identity_csv)
    enriched = pd.read_csv(enriched_csv)
    merged = identity.merge(enriched, on=KEYS, how='left')
    out = merged[KEYS].copy()

    numeric_cols = [
        'home_attack_strength', 'away_attack_strength',
        'home_defensive_strength', 'away_defensive_strength',
        'home_midfield_control', 'away_midfield_control',
        'home_wing_strength', 'away_wing_strength',
        'home_defensive_restraint', 'away_defensive_restraint',
        'home_conversion_quality', 'away_conversion_quality',
        'home_possession_l5', 'away_possession_l5',
        'home_pass_accuracy_l5', 'away_pass_accuracy_l5',
        'home_chaos_index_l10', 'away_chaos_index_l10',
        'home_cards_total_l5', 'away_cards_total_l5',
        'home_fouls_for_l5', 'away_fouls_for_l5',
        'home_team_ppg_l5', 'away_team_ppg_l5',
        'home_team_points_weighted_l5', 'away_team_points_weighted_l5',
        'home_btts_rate_l5', 'away_btts_rate_l5',
        'home_over25_rate_l5', 'away_over25_rate_l5',
        'home_scored_rate_l5', 'away_scored_rate_l5',
        'home_conceded_rate_l5', 'away_conceded_rate_l5',
        'home_shot_accuracy_l5', 'away_shot_accuracy_l5',
    ]
    for col in numeric_cols:
        out[col] = _to_num(merged, col)

    out['home_attack_vs_away_defence_gap'] = out['home_attack_strength'] - out['away_defensive_strength']
    out['away_attack_vs_home_defence_gap'] = out['away_attack_strength'] - out['home_defensive_strength']
    out['home_attack_vs_away_restraint_gap'] = out['home_attack_strength'] - out['away_defensive_restraint']
    out['away_attack_vs_home_restraint_gap'] = out['away_attack_strength'] - out['home_defensive_restraint']

    out['home_buildup_resistance'] = (
        0.55 * _scaled(out['home_pass_accuracy_l5']) +
        0.45 * _scaled(out['home_midfield_control'])
    )
    out['away_buildup_resistance'] = (
        0.55 * _scaled(out['away_pass_accuracy_l5']) +
        0.45 * _scaled(out['away_midfield_control'])
    )

    out['home_press_intensity_proxy'] = (
        0.50 * _scaled(out['home_fouls_for_l5']) +
        0.50 * (1.0 - _scaled(out['home_possession_l5']))
    )
    out['away_press_intensity_proxy'] = (
        0.50 * _scaled(out['away_fouls_for_l5']) +
        0.50 * (1.0 - _scaled(out['away_possession_l5']))
    )

    out['home_press_vs_away_buildup_gap'] = out['home_press_intensity_proxy'] - out['away_buildup_resistance']
    out['away_press_vs_home_buildup_gap'] = out['away_press_intensity_proxy'] - out['home_buildup_resistance']
    out['press_mismatch_index'] = (out['home_press_vs_away_buildup_gap'] - out['away_press_vs_home_buildup_gap']).abs()
    out['pressed_vs_pressed'] = out['home_press_intensity_proxy'] * out['away_press_intensity_proxy']

    out['both_teams_chaos_interaction'] = out['home_chaos_index_l10'] * out['away_chaos_index_l10']
    out['style_conflict_index'] = (out['home_possession_l5'] - out['away_possession_l5']).abs()
    out['midfield_control_conflict_index'] = (out['home_midfield_control'] - out['away_midfield_control']).abs()
    out['wing_mismatch_index'] = (out['home_wing_strength'] - out['away_wing_strength']).abs()

    out['both_teams_booking_risk'] = (
        out['home_cards_total_l5'] +
        out['away_cards_total_l5'] +
        out['home_fouls_for_l5'] +
        out['away_fouls_for_l5']
    )
    out['booking_pressure_interaction'] = _scaled(out['both_teams_booking_risk']) * _scaled(out['both_teams_chaos_interaction'])

    out['goal_environment_interaction'] = (
        out['home_over25_rate_l5'] * out['away_over25_rate_l5'] +
        out['home_btts_rate_l5'] * out['away_btts_rate_l5']
    ) / 2.0
    out['mutual_scoring_interaction'] = out['home_scored_rate_l5'] * out['away_scored_rate_l5']
    out['mutual_conceding_interaction'] = out['home_conceded_rate_l5'] * out['away_conceded_rate_l5']
    out['conversion_clash_index'] = out['home_conversion_quality'] - out['away_conversion_quality']
    out['balanced_strength_flag'] = (
        (out['home_team_ppg_l5'] - out['away_team_ppg_l5']).abs() <= 0.35
    ).astype(int)
    out['high_volatility_balanced_flag'] = (
        (out['balanced_strength_flag'] == 1) &
        (_scaled(out['both_teams_chaos_interaction']) > 0.6)
    ).astype(int)

    keep_cols = KEYS + [
        'home_attack_vs_away_defence_gap',
        'away_attack_vs_home_defence_gap',
        'home_attack_vs_away_restraint_gap',
        'away_attack_vs_home_restraint_gap',
        'home_buildup_resistance',
        'away_buildup_resistance',
        'home_press_intensity_proxy',
        'away_press_intensity_proxy',
        'home_press_vs_away_buildup_gap',
        'away_press_vs_home_buildup_gap',
        'press_mismatch_index',
        'pressed_vs_pressed',
        'both_teams_chaos_interaction',
        'style_conflict_index',
        'midfield_control_conflict_index',
        'wing_mismatch_index',
        'both_teams_booking_risk',
        'booking_pressure_interaction',
        'goal_environment_interaction',
        'mutual_scoring_interaction',
        'mutual_conceding_interaction',
        'conversion_clash_index',
        'balanced_strength_flag',
        'high_volatility_balanced_flag',
    ]
    out = out[keep_cols]
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=PURPOSE)
    parser.add_argument('--identity-csv', default=str(FEATURE_FILES['api_team_identity_features']))
    parser.add_argument('--enriched-csv', default=str(FEATURE_FILES['api_enriched_fixture_features']))
    parser.add_argument('--output-csv', default=str(TARGET_PATH))
    args = parser.parse_args()
    df = build_matchup_interaction_features(args.identity_csv, args.enriched_csv, args.output_csv)
    print(f'WROTE: {args.output_csv} rows={len(df)} cols={len(df.columns)}')


if __name__ == '__main__':
    main()
