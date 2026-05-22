from __future__ import annotations

FIXTURE_KEYS = [
    'fixture_id','fixture_key','league','league_id','season','match_date',
    'home_team_id','away_team_id','home_team_name','away_team_name',
]

NORMALIZED_SCHEMAS = {
    'fixtures_master': FIXTURE_KEYS + ['kickoff_ts_utc','status','venue_id','venue_name','referee_name'],
    'match_team_stats': ['fixture_id','team_id','team_name','is_home','goals_for','goals_against','ht_goals_for','ht_goals_against','shots_total','shots_on_goal','shots_inside_box','shots_outside_box','blocked_shots','possession_pct','passes_total','passes_accurate','corners_for','fouls_for','yellow_cards','red_cards'],
    'match_events': ['fixture_id','event_id','minute','extra_minute','team_id','player_id','event_type','event_detail','is_home','score_home_after','score_away_after'],
    'match_player_stats': ['fixture_id','player_id','team_id','player_name','position','minutes','started_flag','subbed_on_flag','subbed_off_flag','rating','goals','assists','shots_total','shots_on_target','passes_total','passes_key','passes_accurate','tackles','interceptions','blocks','duels_total','duels_won','dribbles_attempted','dribbles_successful','dribbled_past','fouls_drawn','fouls_committed','yellow_cards','red_cards','saves','goals_conceded'],
    'lineups': ['fixture_id','team_id','player_id','player_name','formation','is_starting_xi','position','lineup_known_pre_kickoff_flag','lineup_published_ts_utc'],
    'injuries': [
        'fixture_id','team_id','player_id','player_name','absence_type','reason','status','known_pre_kickoff_flag','published_ts_utc',
        'provider_fixture_ts_utc','source_scope','source_params','fetched_ts_utc','availability_key','availability_first_seen_ts_utc',
        'fixture_only_late_confirmation_flag',
    ],
    'sidelined': [
        'player_id','player_name','coach_id','coach_name','team_id','team_name','absence_type','reason','start_date','end_date',
        'is_open_absence','source_scope','source_params','fetched_ts_utc',
    ],
    'odds_prematch_long': ['fixture_id','bookmaker_id','bookmaker_name','market_code','market_name','selection_code','selection_name','line_value','odds','snapshot_ts_utc','is_opening','is_latest_pre_kickoff'],
    'odds_live_long': ['fixture_id','live_minute','bookmaker_id','market_code','selection_code','odds','snapshot_ts_utc'],
}

FEATURE_SCHEMAS = {
    'api_team_rolling_features': FIXTURE_KEYS + [
        'home_team_ppg_l3','home_team_ppg_l5','home_team_ppg_l10','home_team_ppg_season',
        'away_team_ppg_l3','away_team_ppg_l5','away_team_ppg_l10','away_team_ppg_season',
        'home_team_win_rate_l5','home_team_draw_rate_l5','home_team_loss_rate_l5',
        'away_team_win_rate_l5','away_team_draw_rate_l5','away_team_loss_rate_l5',
        'home_team_points_weighted_l5','away_team_points_weighted_l5',
        'ppg_diff_l5','ppg_diff_l10','ppg_diff_season','form_points_diff_weighted_l5',
        'home_goals_for_l3','home_goals_for_l5','home_goals_for_l10',
        'home_goals_against_l3','home_goals_against_l5','home_goals_against_l10',
        'away_goals_for_l3','away_goals_for_l5','away_goals_for_l10',
        'away_goals_against_l3','away_goals_against_l5','away_goals_against_l10',
        'home_goal_diff_l5','away_goal_diff_l5','goal_diff_delta_l5',
        'home_total_goals_l5','away_total_goals_l5','combined_total_goals_l5',
        'home_over05_rate_l5','home_over15_rate_l5','home_over25_rate_l5','home_over35_rate_l5','home_over45_rate_l5',
        'away_over05_rate_l5','away_over15_rate_l5','away_over25_rate_l5','away_over35_rate_l5','away_over45_rate_l5',
        'combined_over15_rate_l5','combined_over25_rate_l5','combined_over35_rate_l5','combined_over45_rate_l5',
        'home_btts_rate_l5','home_btts_rate_l10','away_btts_rate_l5','away_btts_rate_l10','combined_btts_rate_l5',
        'home_clean_sheet_rate_l5','away_clean_sheet_rate_l5','home_fts_rate_l5','away_fts_rate_l5',
        'home_scored_rate_l5','away_scored_rate_l5','home_conceded_rate_l5','away_conceded_rate_l5',
        'home_ht_goals_for_l5','home_ht_goals_against_l5','away_ht_goals_for_l5','away_ht_goals_against_l5',
        'home_ht_leading_rate_l5','home_ht_drawing_rate_l5','home_ht_losing_rate_l5',
        'away_ht_leading_rate_l5','away_ht_drawing_rate_l5','away_ht_losing_rate_l5',
        'combined_ht_over05_rate_l5','combined_ht_over15_rate_l5','combined_ht_over25_rate_l5',
        'home_shots_l5','home_sot_l5','home_shot_accuracy_l5','home_shots_inside_box_l5','home_shots_outside_box_l5','home_blocked_shots_l5',
        'away_shots_l5','away_sot_l5','away_shot_accuracy_l5','away_shots_inside_box_l5','away_shots_outside_box_l5','away_blocked_shots_l5',
        'shot_delta_l5','sot_delta_l5','shot_accuracy_delta_l5','box_shot_delta_l5',
        'home_possession_l5','home_passes_l5','home_pass_accuracy_l5','home_progressive_proxy_l5',
        'away_possession_l5','away_passes_l5','away_pass_accuracy_l5','away_progressive_proxy_l5',
        'possession_delta_l5','pass_accuracy_delta_l5',
        'home_corners_for_l5','home_corners_against_l5','away_corners_for_l5','away_corners_against_l5',
        'corner_delta_l5','combined_corners_l5','home_corner_pressure_l5','away_corner_pressure_l5',
        'home_fouls_for_l5','home_fouls_against_l5','home_yellow_cards_l5','home_red_cards_l10','home_cards_total_l5',
        'away_fouls_for_l5','away_fouls_against_l5','away_yellow_cards_l5','away_red_cards_l10','away_cards_total_l5',
        'foul_delta_l5','card_delta_l5','combined_card_pressure_l5','combined_foul_pressure_l5'
    ],
    'api_player_rolling_features': FIXTURE_KEYS + ['player_id','player_name','team_id','position','player_minutes_l5','player_start_rate_l5','player_rating_l5','player_goals_l5','player_assists_l5','player_shots_l5','player_sot_l5','player_tackles_l5','player_fouls_committed_l5','player_fouls_drawn_l5','player_yellow_cards_l10','player_red_cards_l10','player_goals_per90_l5','player_assists_per90_l5','player_shots_per90_l5','player_sot_per90_l5','player_tackles_per90_l5','player_fouls_committed_per90_l5','player_fouls_drawn_per90_l5','player_cards_per90_l10','player_shot_accuracy_l5','player_pass_accuracy_l5','player_duel_win_rate_l5','player_dribble_success_rate_l5'],
    'api_lineup_features': FIXTURE_KEYS + [
        'home_formation','away_formation','same_formation_flag','formation_mismatch_flag',
        'home_backline_count','away_backline_count','home_midfield_count','away_midfield_count','home_forward_count','away_forward_count',
        'home_attacking_shape_score','away_attacking_shape_score','home_defensive_shape_score','away_defensive_shape_score',
        'formation_attack_delta','formation_defence_delta',
        'home_starting_xi_avg_rating_l5','away_starting_xi_avg_rating_l5','home_starting_xi_minutes_l5','away_starting_xi_minutes_l5',
        'home_starting_xi_goals_per90_l5','away_starting_xi_goals_per90_l5','home_starting_xi_assists_per90_l5','away_starting_xi_assists_per90_l5',
        'home_starting_xi_shots_per90_l5','away_starting_xi_shots_per90_l5','home_starting_xi_sot_per90_l5','away_starting_xi_sot_per90_l5',
        'home_starting_xi_tackles_per90_l5','away_starting_xi_tackles_per90_l5','home_starting_xi_fouls_committed_per90_l5','away_starting_xi_fouls_committed_per90_l5',
        'home_starting_xi_cards_per90_l10','away_starting_xi_cards_per90_l10',
        'xi_rating_delta','xi_minutes_delta','xi_goal_power_delta','xi_shot_power_delta','xi_sot_power_delta','xi_tackle_pressure_delta','xi_card_risk_delta'
    ],
    'api_injury_features': FIXTURE_KEYS + [
        'home_injured_players_count','away_injured_players_count','home_suspended_players_count','away_suspended_players_count',
        'home_missing_defenders_count','away_missing_defenders_count','home_missing_midfielders_count','away_missing_midfielders_count',
        'home_missing_attackers_count','away_missing_attackers_count','home_missing_goalkeepers_count','away_missing_goalkeepers_count',
        'home_missing_minutes_l5_total','away_missing_minutes_l5_total','home_missing_goals_per90_l5','away_missing_goals_per90_l5',
        'home_missing_assists_per90_l5','away_missing_assists_per90_l5','home_missing_tackles_per90_l5','away_missing_tackles_per90_l5',
        'home_absence_severity_score','away_absence_severity_score','absence_severity_delta'
    ],
    'api_event_features': FIXTURE_KEYS + [
        'home_first_goal_rate_l10','away_first_goal_rate_l10','home_concede_first_rate_l10','away_concede_first_rate_l10',
        'home_late_goal_scored_rate_l10','away_late_goal_scored_rate_l10','home_late_goal_conceded_rate_l10','away_late_goal_conceded_rate_l10',
        'home_red_card_rate_l20','away_red_card_rate_l20','home_yellow_card_rate_l10','away_yellow_card_rate_l10',
        'home_goal_after_75_rate_l10','away_goal_after_75_rate_l10','home_concede_after_75_rate_l10','away_concede_after_75_rate_l10',
        'home_chaos_index_l10','away_chaos_index_l10','combined_chaos_index_l10',
        'home_late_volatility_l10','away_late_volatility_l10','combined_late_volatility_l10',
        'card_volatility_l10','goal_timing_volatility_l10'
    ],
    'api_odds_features': FIXTURE_KEYS + [
        'odds_home_win_best','odds_draw_best','odds_away_win_best','odds_over25_best','odds_under25_best','odds_btts_yes_best','odds_btts_no_best',
        'odds_home_win_mean','odds_draw_mean','odds_away_win_mean','odds_over25_mean','odds_btts_yes_mean',
        'bookie_home_prob_norm','bookie_draw_prob_norm','bookie_away_prob_norm','bookie_over25_prob_norm','bookie_btts_yes_prob_norm',
        'home_odds_std','draw_odds_std','away_odds_std','over25_odds_std','btts_yes_odds_std',
        'home_market_disagreement','draw_market_disagreement','away_market_disagreement','over25_market_disagreement','btts_market_disagreement',
        'home_odds_open','home_odds_latest','home_odds_drift','draw_odds_open','draw_odds_latest','draw_odds_drift',
        'away_odds_open','away_odds_latest','away_odds_drift','over25_odds_open','over25_odds_latest','over25_odds_drift',
        'btts_yes_odds_open','btts_yes_odds_latest','btts_yes_odds_drift'
    ],
    'api_live_features': FIXTURE_KEYS + ['live_minute'],
    'api_enriched_fixture_features': FIXTURE_KEYS,
}
