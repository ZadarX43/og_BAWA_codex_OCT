from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

import pandas as pd

BASELINE_CORE_FEATURES = [
    'league', 'season', 'home_team_name', 'away_team_name',
    'Pre-Match PPG (Home)', 'Pre-Match PPG (Away)', 'home_ppg', 'away_ppg',
    'pre_match_ppg_home', 'pre_match_ppg_away',
    'pre_match_xg_home', 'pre_match_xg_away',
    'team_a_xg', 'team_b_xg', 'average_goals_per_match_pre_match',
    'btts_percentage_pre_match', 'over_15_percentage_pre_match', 'over_25_percentage_pre_match',
    'over_35_percentage_pre_match', 'over_45_percentage_pre_match',
    'over_05_HT_FHG_percentage_pre_match', 'over_15_HT_FHG_percentage_pre_match',
    'average_corners_per_match_pre_match', 'average_cards_per_match_pre_match',
    'odds_ft_home_team_win', 'odds_ft_draw', 'odds_ft_away_team_win',
    'odds_ft_over15', 'odds_ft_over25', 'odds_ft_over35', 'odds_ft_over45',
    'odds_btts_yes', 'odds_btts_no',
]

EXCLUDE_ALWAYS = {
    'fixture_id', 'fixture_key', 'league_id', 'home_team_id', 'away_team_id', 'match_date', 'kickoff_ts_utc', 'status', 'venue_id', 'venue_name',
    'api_home_team_name', 'api_away_team_name',
    'home_team_goal_count', 'away_team_goal_count', 'home_team_goal_count_half_time', 'away_team_goal_count_half_time',
    'total_goal_count', 'total_goals_at_half_time',
    'home_team_shots', 'away_team_shots',
    'home_team_shots_on_target', 'away_team_shots_on_target',
    'home_team_shots_off_target', 'away_team_shots_off_target',
    'home_team_corner_count', 'away_team_corner_count',
    'home_team_yellow_cards', 'away_team_yellow_cards',
    'home_team_red_cards', 'away_team_red_cards',
    'home_team_first_half_cards', 'away_team_first_half_cards',
    'home_team_second_half_cards', 'away_team_second_half_cards',
    'home_team_fouls', 'away_team_fouls',
    'home_team_possession', 'away_team_possession',
    'home_team_goal_timings', 'away_team_goal_timings',
    'team_a_xg', 'team_b_xg',
    'target_ftr_home', 'target_ftr_draw', 'target_ftr_away', 'target_btts_yes', 'target_ou25_over',
    'target_home_goals_over15', 'target_away_goals_over15', 'target_home_fts', 'target_away_fts', 'target_btts_first_half',
    'view_baseline_core_only', 'view_hybrid_core_plus_api', 'view_api_only_experimental',
    'api_team_ready_flag', 'api_event_ready_flag', 'api_lineup_ready_flag', 'api_injury_ready_flag', 'api_odds_ready_flag',
}

SAFE_EXACT_API = {
    'pre_match_ppg_home', 'pre_match_ppg_away',
    'pre_match_xg_home', 'pre_match_xg_away',
    'home_power_rating', 'away_power_rating', 'power_diff',
    'h2h_n', 'h2h_btts_rate', 'h2h_over25_rate', 'h2h_goaliness_avg',
    'snap_timing_early_goal_pressure',
    'snap_timing_second_half_acceleration',
    'snap_home_first_to_score_edge',
    'snap_ht_goal_regime_blend',
    'snap_ou25_over_regime_blend',
    'snap_goal_environment_blend',
    'snap_btts_regime_blend',
    'same_formation_flag', 'formation_mismatch_flag',
    'home_formation', 'away_formation',
    'goal_diff_delta_l5', 'goal_timing_volatility_l10', 'card_volatility_l10',
    'ppg_diff_l5', 'ppg_diff_l10', 'ppg_diff_season', 'form_points_diff_weighted_l5',
    'foul_delta_l5', 'card_delta_l5', 'shot_delta_l5', 'sot_delta_l5', 'shot_accuracy_delta_l5',
    'box_shot_delta_l5', 'possession_delta_l5', 'pass_accuracy_delta_l5', 'corner_delta_l5',
    'combined_total_goals_l5', 'combined_corners_l5', 'combined_card_pressure_l5', 'combined_foul_pressure_l5',
    'combined_chaos_index_l10', 'combined_late_volatility_l10',
    'absence_severity_delta',
    'bookie_home_prob_norm', 'bookie_draw_prob_norm', 'bookie_away_prob_norm', 'bookie_over25_prob_norm', 'bookie_btts_yes_prob_norm',
    'home_market_disagreement', 'draw_market_disagreement', 'away_market_disagreement', 'over25_market_disagreement', 'btts_market_disagreement',
}

SAFE_PREFIXES = (
    'snap_', 'h2h_',
    'home_team_', 'away_team_',
    'home_goals_', 'away_goals_',
    'home_over', 'away_over', 'combined_over',
    'home_btts_', 'away_btts_', 'combined_btts_',
    'home_clean_sheet_', 'away_clean_sheet_',
    'home_fts_', 'away_fts_', 'home_scored_', 'away_scored_', 'home_conceded_', 'away_conceded_',
    'home_ht_', 'away_ht_', 'combined_ht_',
    'home_shots_', 'away_shots_',
    'home_sot_', 'away_sot_',
    'home_possession_', 'away_possession_',
    'home_pass', 'away_pass',
    'home_corners_', 'away_corners_',
    'home_corner_', 'away_corner_',
    'home_fouls_', 'away_fouls_',
    'home_yellow_', 'away_yellow_', 'home_red_', 'away_red_',
    'home_cards_', 'away_cards_',
    'home_first_', 'away_first_',
    'home_concede_', 'away_concede_',
    'home_late_', 'away_late_',
    'home_goal_after_', 'away_goal_after_',
    'home_chaos_', 'away_chaos_',
    'home_backline_', 'away_backline_', 'home_midfield_', 'away_midfield_', 'home_forward_', 'away_forward_',
    'home_attacking_', 'away_attacking_', 'home_defensive_', 'away_defensive_',
    'formation_',
    'home_starting_xi_', 'away_starting_xi_', 'xi_',
    'home_injured_', 'away_injured_', 'home_suspended_', 'away_suspended_',
    'home_missing_', 'away_missing_', 'home_absence_', 'away_absence_', 'absence_',
    'odds_', 'bookie_', 'home_odds_', 'draw_odds_', 'away_odds_', 'over25_odds_', 'btts_yes_odds_',
)

DENY_TOKEN_RE = re.compile(
    r'(?:^|_)(?:result|winner|final|scoreline|full[_]?time|half[_]?time|current|elapsed|minute|live|inplay)(?:_|$)'
)


def load_training_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename_map = {
        'Pre-Match PPG (Home)': 'pre_match_ppg_home',
        'Pre-Match PPG (Away)': 'pre_match_ppg_away',
        'Home Team Pre-Match xG': 'pre_match_xg_home',
        'Away Team Pre-Match xG': 'pre_match_xg_away',
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    if 'match_date' in df.columns:
        df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
    sort_cols = [c for c in ['match_date', 'fixture_id'] if c in df.columns]
    if sort_cols:
        return df.sort_values(sort_cols).reset_index(drop=True)
    return df.reset_index(drop=True)


def baseline_feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in BASELINE_CORE_FEATURES if c in df.columns and c not in EXCLUDE_ALWAYS]


def is_safe_api_feature(col: str) -> bool:
    if col in SAFE_EXACT_API:
        return True
    if col in EXCLUDE_ALWAYS:
        return False
    if DENY_TOKEN_RE.search(col.lower()):
        return False
    return col.startswith(SAFE_PREFIXES)


def api_feature_columns(df: pd.DataFrame) -> List[str]:
    baseline = set(baseline_feature_columns(df))
    cols: List[str] = []
    for col in df.columns:
        if col in baseline or col in EXCLUDE_ALWAYS:
            continue
        if col.startswith('target_') or col.startswith('view_') or col.endswith('_ready_flag'):
            continue
        if is_safe_api_feature(col):
            cols.append(col)
    return cols


def feature_columns(df: pd.DataFrame, feature_view: str) -> List[str]:
    baseline = baseline_feature_columns(df)
    api_cols = api_feature_columns(df)
    if feature_view == 'baseline':
        return baseline
    if feature_view == 'api_only':
        return api_cols
    return baseline + api_cols


def ensure_columns_exist(cols: Iterable[str], df: pd.DataFrame) -> List[str]:
    return [c for c in cols if c in df.columns]
