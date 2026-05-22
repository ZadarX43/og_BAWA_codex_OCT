from __future__ import annotations

from typing import Dict, List, Sequence

import pandas as pd

from .hybrid_training_utils import BASELINE_CORE_FEATURES, EXCLUDE_ALWAYS

FAMILY_PREFIXES: Dict[str, Sequence[str]] = {
    'team': (
        'home_team_ppg_', 'away_team_ppg_', 'home_team_win_rate_', 'away_team_win_rate_',
        'home_team_draw_rate_', 'away_team_draw_rate_', 'home_team_loss_rate_', 'away_team_loss_rate_',
        'home_team_points_weighted_', 'away_team_points_weighted_',
        'ppg_diff_', 'form_points_',
        'home_goals_', 'away_goals_', 'goal_diff_', 'combined_total_goals_',
        'home_over', 'away_over', 'combined_over',
        'home_btts_', 'away_btts_', 'combined_btts_',
        'home_clean_sheet_', 'away_clean_sheet_',
        'home_fts_', 'away_fts_', 'home_scored_', 'away_scored_', 'home_conceded_', 'away_conceded_',
        'home_ht_', 'away_ht_', 'combined_ht_',
        'home_shots_', 'away_shots_', 'home_sot_', 'away_sot_',
        'home_possession_', 'away_possession_', 'home_pass', 'away_pass',
        'home_corners_', 'away_corners_', 'home_corner_', 'away_corner_',
        'home_fouls_', 'away_fouls_', 'home_yellow_', 'away_yellow_', 'home_red_', 'away_red_',
        'home_cards_', 'away_cards_',
        'shot_delta_', 'sot_delta_', 'shot_accuracy_delta_', 'box_shot_delta_',
        'possession_delta_', 'pass_accuracy_delta_', 'corner_delta_', 'foul_delta_', 'card_delta_',
        'combined_corners_', 'combined_card_pressure_', 'combined_foul_pressure_',
    ),
    'lineup': (
        'home_formation', 'away_formation', 'same_formation_flag', 'formation_mismatch_flag',
        'home_backline_', 'away_backline_', 'home_midfield_', 'away_midfield_', 'home_forward_', 'away_forward_',
        'home_attacking_', 'away_attacking_', 'home_defensive_', 'away_defensive_',
        'formation_', 'home_starting_xi_', 'away_starting_xi_', 'xi_',
    ),
    'injury': (
        'home_injured_', 'away_injured_', 'home_suspended_', 'away_suspended_',
        'home_missing_', 'away_missing_', 'home_absence_', 'away_absence_', 'absence_',
    ),
    'event': (
        'home_first_', 'away_first_', 'home_concede_', 'away_concede_',
        'home_late_', 'away_late_', 'home_goal_after_', 'away_goal_after_',
        'home_chaos_', 'away_chaos_', 'combined_chaos_', 'combined_late_volatility_',
        'card_volatility_', 'goal_timing_volatility_',
    ),
}

EXTRA_EXCLUDE = {
    'home_team_goal_count', 'away_team_goal_count', 'home_team_goal_count_half_time', 'away_team_goal_count_half_time',
    'home_team_name', 'away_team_name', 'api_home_team_name', 'api_away_team_name',
    'league', 'season'
}


def baseline_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in BASELINE_CORE_FEATURES if c in df.columns]


def family_cols(df: pd.DataFrame, family: str) -> List[str]:
    prefixes = FAMILY_PREFIXES[family]
    cols: List[str] = []
    for c in df.columns:
        if c in EXCLUDE_ALWAYS or c in EXTRA_EXCLUDE:
            continue
        if c.startswith('target_') or c.startswith('view_') or c.endswith('_ready_flag'):
            continue
        if any(c.startswith(p) for p in prefixes):
            cols.append(c)
    return cols


def stacked_cols(df: pd.DataFrame, families: Sequence[str]) -> List[str]:
    cols = baseline_cols(df)
    for family in families:
        cols.extend(family_cols(df, family))
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out
