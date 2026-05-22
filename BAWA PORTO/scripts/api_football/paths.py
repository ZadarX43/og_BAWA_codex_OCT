from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
API_ROOT = REPO_ROOT / 'data_sources' / 'api_football'
RAW_DIR = API_ROOT / 'raw'
NORMALIZED_DIR = API_ROOT / 'normalized'
FEATURES_DIR = API_ROOT / 'features'
HYBRID_DIR = REPO_ROOT / 'data_sources' / 'hybrid'
REPORTS_DIR = REPO_ROOT / 'reports' / 'api_football'

NORMALIZED_FILES = {
    'fixtures_master': NORMALIZED_DIR / 'fixtures_master.csv',
    'match_team_stats': NORMALIZED_DIR / 'match_team_stats.csv',
    'match_events': NORMALIZED_DIR / 'match_events.csv',
    'match_player_stats': NORMALIZED_DIR / 'match_player_stats.csv',
    'lineups': NORMALIZED_DIR / 'lineups.csv',
    'injuries': NORMALIZED_DIR / 'injuries.csv',
    'sidelined': NORMALIZED_DIR / 'sidelined.csv',
    'odds_prematch_long': NORMALIZED_DIR / 'odds_prematch_long.csv',
    'odds_live_long': NORMALIZED_DIR / 'odds_live_long.csv',
    'teams_master': NORMALIZED_DIR / 'teams_master.csv',
    'players_master': NORMALIZED_DIR / 'players_master.csv',
    'bookmaker_map': NORMALIZED_DIR / 'bookmaker_map.csv',
    'bet_market_map': NORMALIZED_DIR / 'bet_market_map.csv',
}

FEATURE_FILES = {
    'api_team_rolling_features': FEATURES_DIR / 'api_team_rolling_features.csv',
    'api_player_rolling_features': FEATURES_DIR / 'api_player_rolling_features.csv',
    'api_lineup_features': FEATURES_DIR / 'api_lineup_features.csv',
    'api_injury_features': FEATURES_DIR / 'api_injury_features.csv',
    'api_event_features': FEATURES_DIR / 'api_event_features.csv',
    'api_odds_features': FEATURES_DIR / 'api_odds_features.csv',
    'api_live_features': FEATURES_DIR / 'api_live_features.csv',
    'api_enriched_fixture_features': FEATURES_DIR / 'api_enriched_fixture_features.csv',
    'api_team_identity_features': FEATURES_DIR / 'api_team_identity_features.csv',
    'api_matchup_interaction_features': FEATURES_DIR / 'api_matchup_interaction_features.csv',
    'api_h2h_regime_features': FEATURES_DIR / 'api_h2h_regime_features.csv',
    'api_referee_profile_features': FEATURES_DIR / 'api_referee_profile_features.csv',
}

HYBRID_FILES = {
    'hybrid_match_training_epl': HYBRID_DIR / 'hybrid_match_training__England_Premier_League.csv',
}

REPORT_FILES = {
    'api_feature_coverage_report': REPORTS_DIR / 'api_feature_coverage_report.csv',
    'api_missing_values_report': REPORTS_DIR / 'api_missing_values_report.csv',
    'api_feature_uplift_matrix': REPORTS_DIR / 'api_feature_uplift_matrix.csv',
    'api_footystats_join_audit': REPORTS_DIR / 'api_footystats_join_audit.csv',
    'api_odds_market_coverage_report': REPORTS_DIR / 'api_odds_market_coverage_report.csv',
    'api_hybrid_feature_coverage': REPORTS_DIR / 'api_hybrid_feature_coverage.csv',
}

LIVE_DATASET_FILES = {
    15: FEATURES_DIR / 'live_minute_15_dataset.csv',
    30: FEATURES_DIR / 'live_minute_30_dataset.csv',
    45: FEATURES_DIR / 'live_minute_45_dataset.csv',
    60: FEATURES_DIR / 'live_minute_60_dataset.csv',
    75: FEATURES_DIR / 'live_minute_75_dataset.csv',
}


def ensure_dirs() -> None:
    for d in (RAW_DIR, NORMALIZED_DIR, FEATURES_DIR, HYBRID_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
