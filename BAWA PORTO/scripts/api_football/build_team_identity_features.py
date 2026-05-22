from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .paths import FEATURE_FILES
from .utils import safe_div

PURPOSE = 'Build pre-match team identity composites from enriched hybrid inputs.'
TARGET_PATH = FEATURE_FILES['api_team_identity_features']
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


def _league_relative(series: pd.Series, group_keys: pd.DataFrame) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce').fillna(0.0)
    means = s.groupby([group_keys['league_id'], group_keys['season']]).transform('mean')
    return (s - means).fillna(0.0)


def _safe_rate(num: pd.Series, den: pd.Series) -> pd.Series:
    out = []
    for a, b in zip(num.tolist(), den.tolist()):
        out.append(safe_div(a, b))
    return pd.Series(out, index=num.index, dtype='float64')


def _weighted_composite(df: pd.DataFrame, components: list[tuple[str, float]]) -> pd.Series:
    total_weight = sum(weight for _, weight in components) or 1.0
    acc = pd.Series(0.0, index=df.index, dtype='float64')
    for col, weight in components:
        acc = acc + (_scaled(df[col]) * weight)
    return acc / total_weight


def build_team_identity_features(enriched_csv: str, output_csv: str = str(TARGET_PATH)) -> pd.DataFrame:
    df = pd.read_csv(enriched_csv)
    out = df[KEYS].copy()

    for col in [
        'home_goals_for_l5', 'away_goals_for_l5', 'home_goals_against_l5', 'away_goals_against_l5',
        'home_shots_l5', 'away_shots_l5', 'home_sot_l5', 'away_sot_l5',
        'home_shot_accuracy_l5', 'away_shot_accuracy_l5',
        'home_shots_inside_box_l5', 'away_shots_inside_box_l5',
        'home_possession_l5', 'away_possession_l5', 'home_pass_accuracy_l5', 'away_pass_accuracy_l5',
        'home_scored_rate_l5', 'away_scored_rate_l5', 'home_conceded_rate_l5', 'away_conceded_rate_l5',
        'home_clean_sheet_rate_l5', 'away_clean_sheet_rate_l5',
        'home_fouls_for_l5', 'away_fouls_for_l5', 'home_cards_total_l5', 'away_cards_total_l5',
        'home_red_cards_l10', 'away_red_cards_l10',
        'home_starting_xi_avg_rating_l5', 'away_starting_xi_avg_rating_l5',
        'home_starting_xi_goals_per90_l5', 'away_starting_xi_goals_per90_l5',
        'home_starting_xi_assists_per90_l5', 'away_starting_xi_assists_per90_l5',
        'home_starting_xi_shots_per90_l5', 'away_starting_xi_shots_per90_l5',
        'home_starting_xi_sot_per90_l5', 'away_starting_xi_sot_per90_l5',
        'home_starting_xi_tackles_per90_l5', 'away_starting_xi_tackles_per90_l5',
        'home_starting_xi_fouls_committed_per90_l5', 'away_starting_xi_fouls_committed_per90_l5',
        'home_starting_xi_cards_per90_l10', 'away_starting_xi_cards_per90_l10',
        'home_attacking_shape_score', 'away_attacking_shape_score',
        'home_defensive_shape_score', 'away_defensive_shape_score',
        'home_missing_goals_per90_l5', 'away_missing_goals_per90_l5',
        'home_missing_assists_per90_l5', 'away_missing_assists_per90_l5',
        'home_missing_tackles_per90_l5', 'away_missing_tackles_per90_l5',
        'home_absence_severity_score', 'away_absence_severity_score',
        'home_chaos_index_l10', 'away_chaos_index_l10',
    ]:
        out[col] = _to_num(df, col)

    out['home_conversion_quality_raw'] = _safe_rate(out['home_goals_for_l5'], out['home_sot_l5'])
    out['away_conversion_quality_raw'] = _safe_rate(out['away_goals_for_l5'], out['away_sot_l5'])
    out['home_shot_conversion_raw'] = _safe_rate(out['home_goals_for_l5'], out['home_shots_l5'])
    out['away_shot_conversion_raw'] = _safe_rate(out['away_goals_for_l5'], out['away_shots_l5'])
    out['home_sot_conversion_raw'] = _safe_rate(out['home_goals_for_l5'], out['home_sot_l5'])
    out['away_sot_conversion_raw'] = _safe_rate(out['away_goals_for_l5'], out['away_sot_l5'])

    out['home_conversion_quality'] = _league_relative(out['home_conversion_quality_raw'], out)
    out['away_conversion_quality'] = _league_relative(out['away_conversion_quality_raw'], out)
    out['conversion_delta'] = out['home_conversion_quality'] - out['away_conversion_quality']

    out['home_absence_attack_penalty'] = (
        0.60 * out['home_missing_goals_per90_l5'] +
        0.40 * out['home_missing_assists_per90_l5'] +
        0.10 * out['home_absence_severity_score']
    )
    out['away_absence_attack_penalty'] = (
        0.60 * out['away_missing_goals_per90_l5'] +
        0.40 * out['away_missing_assists_per90_l5'] +
        0.10 * out['away_absence_severity_score']
    )
    out['home_absence_defensive_penalty'] = (
        0.70 * out['home_missing_tackles_per90_l5'] +
        0.10 * out['home_absence_severity_score'] +
        0.20 * out['home_conceded_rate_l5']
    )
    out['away_absence_defensive_penalty'] = (
        0.70 * out['away_missing_tackles_per90_l5'] +
        0.10 * out['away_absence_severity_score'] +
        0.20 * out['away_conceded_rate_l5']
    )

    out['home_defensive_foul_rate'] = _safe_rate(out['home_fouls_for_l5'], pd.Series(1.0, index=out.index))
    out['away_defensive_foul_rate'] = _safe_rate(out['away_fouls_for_l5'], pd.Series(1.0, index=out.index))
    out['home_card_per_foul'] = _safe_rate(out['home_cards_total_l5'], out['home_fouls_for_l5'])
    out['away_card_per_foul'] = _safe_rate(out['away_cards_total_l5'], out['away_fouls_for_l5'])
    out['home_red_card_volatility'] = _safe_rate(out['home_red_cards_l10'], pd.Series(10.0, index=out.index))
    out['away_red_card_volatility'] = _safe_rate(out['away_red_cards_l10'], pd.Series(10.0, index=out.index))

    out['home_defensive_restraint'] = 1.0 - _weighted_composite(out, [
        ('home_defensive_foul_rate', 0.45),
        ('home_card_per_foul', 0.35),
        ('home_red_card_volatility', 0.20),
    ])
    out['away_defensive_restraint'] = 1.0 - _weighted_composite(out, [
        ('away_defensive_foul_rate', 0.45),
        ('away_card_per_foul', 0.35),
        ('away_red_card_volatility', 0.20),
    ])
    out['defensive_restraint_delta'] = out['home_defensive_restraint'] - out['away_defensive_restraint']

    out['home_attack_strength'] = _weighted_composite(out, [
        ('home_starting_xi_goals_per90_l5', 0.22),
        ('home_starting_xi_shots_per90_l5', 0.18),
        ('home_starting_xi_sot_per90_l5', 0.18),
        ('home_starting_xi_assists_per90_l5', 0.14),
        ('home_shot_accuracy_l5', 0.08),
        ('home_scored_rate_l5', 0.10),
        ('home_conversion_quality', 0.10),
    ]) - (0.12 * _scaled(out['home_absence_attack_penalty']))
    out['away_attack_strength'] = _weighted_composite(out, [
        ('away_starting_xi_goals_per90_l5', 0.22),
        ('away_starting_xi_shots_per90_l5', 0.18),
        ('away_starting_xi_sot_per90_l5', 0.18),
        ('away_starting_xi_assists_per90_l5', 0.14),
        ('away_shot_accuracy_l5', 0.08),
        ('away_scored_rate_l5', 0.10),
        ('away_conversion_quality', 0.10),
    ]) - (0.12 * _scaled(out['away_absence_attack_penalty']))
    out['attack_strength_delta'] = out['home_attack_strength'] - out['away_attack_strength']

    out['home_conceded_suppression_proxy'] = 1.0 - _scaled(out['home_conceded_rate_l5'])
    out['away_conceded_suppression_proxy'] = 1.0 - _scaled(out['away_conceded_rate_l5'])
    out['home_goals_against_suppression_proxy'] = 1.0 - _scaled(out['home_goals_against_l5'])
    out['away_goals_against_suppression_proxy'] = 1.0 - _scaled(out['away_goals_against_l5'])

    out['home_defensive_strength'] = _weighted_composite(out, [
        ('home_starting_xi_tackles_per90_l5', 0.22),
        ('home_clean_sheet_rate_l5', 0.20),
        ('home_conceded_suppression_proxy', 0.18),
        ('home_goals_against_suppression_proxy', 0.18),
        ('home_defensive_shape_score', 0.10),
        ('home_defensive_restraint', 0.12),
    ]) - (0.10 * _scaled(out['home_absence_defensive_penalty']))
    out['away_defensive_strength'] = _weighted_composite(out, [
        ('away_starting_xi_tackles_per90_l5', 0.22),
        ('away_clean_sheet_rate_l5', 0.20),
        ('away_conceded_suppression_proxy', 0.18),
        ('away_goals_against_suppression_proxy', 0.18),
        ('away_defensive_shape_score', 0.10),
        ('away_defensive_restraint', 0.12),
    ]) - (0.10 * _scaled(out['away_absence_defensive_penalty']))
    out['defensive_strength_delta'] = out['home_defensive_strength'] - out['away_defensive_strength']

    out['home_ball_winning_rate'] = _scaled(out['home_starting_xi_tackles_per90_l5'])
    out['away_ball_winning_rate'] = _scaled(out['away_starting_xi_tackles_per90_l5'])
    out['home_midfield_control'] = _weighted_composite(out, [
        ('home_pass_accuracy_l5', 0.28),
        ('home_possession_l5', 0.22),
        ('home_starting_xi_assists_per90_l5', 0.16),
        ('home_starting_xi_avg_rating_l5', 0.14),
        ('home_ball_winning_rate', 0.20),
    ])
    out['away_midfield_control'] = _weighted_composite(out, [
        ('away_pass_accuracy_l5', 0.28),
        ('away_possession_l5', 0.22),
        ('away_starting_xi_assists_per90_l5', 0.16),
        ('away_starting_xi_avg_rating_l5', 0.14),
        ('away_ball_winning_rate', 0.20),
    ])
    out['midfield_control_delta'] = out['home_midfield_control'] - out['away_midfield_control']

    out['home_wing_strength'] = _weighted_composite(out, [
        ('home_starting_xi_shots_per90_l5', 0.26),
        ('home_starting_xi_assists_per90_l5', 0.22),
        ('home_attacking_shape_score', 0.20),
        ('home_chaos_index_l10', 0.10),
        ('home_shots_inside_box_l5', 0.22),
    ])
    out['away_wing_strength'] = _weighted_composite(out, [
        ('away_starting_xi_shots_per90_l5', 0.26),
        ('away_starting_xi_assists_per90_l5', 0.22),
        ('away_attacking_shape_score', 0.20),
        ('away_chaos_index_l10', 0.10),
        ('away_shots_inside_box_l5', 0.22),
    ])
    out['wing_strength_delta'] = out['home_wing_strength'] - out['away_wing_strength']

    keep_cols = KEYS + [
        'home_conversion_quality_raw', 'away_conversion_quality_raw',
        'home_conversion_quality', 'away_conversion_quality', 'conversion_delta',
        'home_shot_conversion_raw', 'away_shot_conversion_raw',
        'home_sot_conversion_raw', 'away_sot_conversion_raw',
        'home_absence_attack_penalty', 'away_absence_attack_penalty',
        'home_absence_defensive_penalty', 'away_absence_defensive_penalty',
        'home_defensive_foul_rate', 'away_defensive_foul_rate',
        'home_card_per_foul', 'away_card_per_foul',
        'home_red_card_volatility', 'away_red_card_volatility',
        'home_defensive_restraint', 'away_defensive_restraint', 'defensive_restraint_delta',
        'home_attack_strength', 'away_attack_strength', 'attack_strength_delta',
        'home_defensive_strength', 'away_defensive_strength', 'defensive_strength_delta',
        'home_ball_winning_rate', 'away_ball_winning_rate',
        'home_midfield_control', 'away_midfield_control', 'midfield_control_delta',
        'home_wing_strength', 'away_wing_strength', 'wing_strength_delta',
    ]
    out = out[keep_cols]
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=PURPOSE)
    parser.add_argument('--enriched-csv', default=str(FEATURE_FILES['api_enriched_fixture_features']))
    parser.add_argument('--output-csv', default=str(TARGET_PATH))
    args = parser.parse_args()
    df = build_team_identity_features(args.enriched_csv, args.output_csv)
    print(f'WROTE: {args.output_csv} rows={len(df)} cols={len(df.columns)}')


if __name__ == '__main__':
    main()
