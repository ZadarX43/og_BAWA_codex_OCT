from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .paths import FEATURE_FILES, HYBRID_FILES, REPORT_FILES, ensure_dirs
from .team_name_map import normalize_team_name

PURPOSE = 'Build retrain-ready hybrid match training table from FootyStats core + API boosters.'
TARGET_PATH = HYBRID_FILES['hybrid_match_training_epl']
CORE_COLS = [
    'match_date','league','home_team_name','away_team_name','Pre-Match PPG (Home)','Pre-Match PPG (Away)','home_ppg','away_ppg',
    'team_a_xg','team_b_xg','average_goals_per_match_pre_match','btts_percentage_pre_match','over_15_percentage_pre_match',
    'over_25_percentage_pre_match','over_35_percentage_pre_match','over_45_percentage_pre_match','over_05_HT_FHG_percentage_pre_match',
    'over_15_HT_FHG_percentage_pre_match','average_corners_per_match_pre_match','average_cards_per_match_pre_match',
    'odds_ft_home_team_win','odds_ft_draw','odds_ft_away_team_win','odds_ft_over15','odds_ft_over25','odds_ft_over35','odds_ft_over45','odds_btts_yes','odds_btts_no',
    'home_team_goal_count','away_team_goal_count','home_team_goal_count_half_time','away_team_goal_count_half_time'
]
CANONICAL_FRONT = [
    'fixture_id', 'fixture_key', 'league', 'league_id', 'season', 'match_date',
    'home_team_id', 'away_team_id', 'home_team_name', 'away_team_name',
    'api_home_team_name', 'api_away_team_name', 'kickoff_ts_utc', 'status', 'venue_id', 'venue_name', 'referee_name'
]
CANONICAL_ALIAS_MAP = {
    'Pre-Match PPG (Home)': 'pre_match_ppg_home',
    'Pre-Match PPG (Away)': 'pre_match_ppg_away',
    'team_a_xg': 'pre_match_xg_home',
    'team_b_xg': 'pre_match_xg_away',
}

def normalize_name(value: object, tag: str | None = None) -> str:
    return normalize_team_name(value, tag=tag)


def _clean_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        'match_date_x': 'api_match_date_bridge',
        'match_date_y': 'match_date_fs',
        'match_date': 'match_date_api',
        'league_x': 'league_bridge',
        'league_y': 'league_fs',
        'league': 'league_api',
        'fixture_key_api': 'fixture_key_source_api',
        'home_team_name_api': 'api_home_team_name',
        'away_team_name_api': 'api_away_team_name',
        'referee_name_x': 'referee_name_api',
        'referee_name_y': 'referee_name_ref_profile',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if 'fixture_key_source_api' in df.columns:
        same_key = df['fixture_key'].astype(str).fillna('') == df['fixture_key_source_api'].astype(str).fillna('')
        if bool(same_key.all()):
            df = df.drop(columns=['fixture_key_source_api'])

    if 'league_bridge' in df.columns:
        df = df.drop(columns=['league_bridge'])

    if 'api_match_date_bridge' in df.columns:
        df = df.drop(columns=['api_match_date_bridge'])

    if 'api_home_team_name' in df.columns and 'home_team_name' in df.columns:
        df['api_home_team_name'] = df['api_home_team_name'].fillna(df['home_team_name'])
    if 'api_away_team_name' in df.columns and 'away_team_name' in df.columns:
        df['api_away_team_name'] = df['api_away_team_name'].fillna(df['away_team_name'])
    if 'referee_name_api' in df.columns or 'referee_name_ref_profile' in df.columns:
        api_ref = df['referee_name_api'] if 'referee_name_api' in df.columns else pd.Series([None] * len(df))
        prof_ref = df['referee_name_ref_profile'] if 'referee_name_ref_profile' in df.columns else pd.Series([None] * len(df))
        df['referee_name'] = api_ref.fillna(prof_ref)
        df = df.drop(columns=[c for c in ['referee_name_api', 'referee_name_ref_profile'] if c in df.columns])

    if 'league_fs' in df.columns:
        df['league'] = df['league_fs']
        if 'league_api' in df.columns:
            df['league'] = df['league'].fillna(df['league_api'])
        df = df.drop(columns=[c for c in ['league_fs', 'league_api'] if c in df.columns])

    if 'match_date_fs' in df.columns:
        df['match_date'] = df['match_date_fs']
        if 'match_date_api' in df.columns:
            df['match_date'] = df['match_date'].fillna(df['match_date_api'])
        df = df.drop(columns=[c for c in ['match_date_fs', 'match_date_api'] if c in df.columns])

    if 'match_date' in df.columns:
        df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce').dt.strftime('%Y-%m-%d')

    if '_join_key' in df.columns:
        df = df.drop(columns=['_join_key'])

    df = df.loc[:, ~df.columns.duplicated()]
    ordered_front = [c for c in CANONICAL_FRONT if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered_front]
    return df[ordered_front + remaining]


def _add_canonical_aliases(df: pd.DataFrame) -> pd.DataFrame:
    for source_col, alias_col in CANONICAL_ALIAS_MAP.items():
        if source_col not in df.columns:
            continue
        if alias_col not in df.columns:
            df[alias_col] = df[source_col]
            continue
        df[alias_col] = df[alias_col].where(df[alias_col].notna(), df[source_col])
    return df


def build_hybrid_match_training(footystats_merged_csv: str, join_audit_csv: str, api_enriched_csv: str, output_csv: str) -> pd.DataFrame:
    ensure_dirs()
    fs = pd.read_csv(footystats_merged_csv)
    ja = pd.read_csv(join_audit_csv)
    api = pd.read_csv(api_enriched_csv)
    tag = Path(footystats_merged_csv).name.replace('__merged__.csv', '').replace('__merged.csv', '')

    fs['_norm_home'] = fs['home_team_name'].map(lambda value: normalize_name(value, tag=tag))
    fs['_norm_away'] = fs['away_team_name'].map(lambda value: normalize_name(value, tag=tag))
    fs['_date'] = pd.to_datetime(fs['match_date'], errors='coerce').dt.date.astype(str)
    fs['_join_key'] = fs['_date'] + '|' + fs['_norm_home'] + '|' + fs['_norm_away']

    matched = ja[ja['join_found_flag'] == 1].copy()
    if matched.empty:
        raise ValueError(
            f'No matched fixture rows found in join audit: {join_audit_csv}. '
            'This usually means the API league/season data does not align to the target __merged__ file '
            '(for example wrong league_id or season).'
        )
    required_join_cols = {'fs_home_team_name', 'fs_away_team_name', 'fs_match_date'}
    missing_join_cols = sorted(required_join_cols - set(matched.columns))
    if missing_join_cols:
        raise ValueError(
            f'Join audit is missing required matched columns {missing_join_cols}: {join_audit_csv}. '
            'This usually indicates the audit contained only unmatched rows.'
        )
    matched['_norm_home'] = matched['fs_home_team_name'].map(lambda value: normalize_name(value, tag=tag))
    matched['_norm_away'] = matched['fs_away_team_name'].map(lambda value: normalize_name(value, tag=tag))
    matched['_date'] = pd.to_datetime(matched['fs_match_date'], errors='coerce').dt.date.astype(str)
    matched['_join_key'] = matched['_date'] + '|' + matched['_norm_home'] + '|' + matched['_norm_away']

    fs_keep = ['_join_key'] + [c for c in CORE_COLS if c in fs.columns]
    bridge_keep = ['fixture_id', 'fixture_key', 'league', 'match_date', 'api_home_team_name', 'api_away_team_name', '_join_key']
    out = matched[bridge_keep].merge(fs[fs_keep], on='_join_key', how='left')
    out = out.merge(api, on='fixture_id', how='left', suffixes=('', '_api'))
    out = _clean_duplicate_columns(out)
    out = _add_canonical_aliases(out)

    out['target_ftr_home'] = (out['home_team_goal_count'] > out['away_team_goal_count']).astype(int)
    out['target_ftr_draw'] = (out['home_team_goal_count'] == out['away_team_goal_count']).astype(int)
    out['target_ftr_away'] = (out['away_team_goal_count'] > out['home_team_goal_count']).astype(int)
    out['target_btts_yes'] = ((out['home_team_goal_count'] > 0) & (out['away_team_goal_count'] > 0)).astype(int)
    out['target_ou25_over'] = ((out['home_team_goal_count'] + out['away_team_goal_count']) >= 3).astype(int)
    out['target_home_goals_over15'] = (out['home_team_goal_count'] >= 2).astype(int)
    out['target_away_goals_over15'] = (out['away_team_goal_count'] >= 2).astype(int)
    out['target_home_fts'] = (out['home_team_goal_count'] == 0).astype(int)
    out['target_away_fts'] = (out['away_team_goal_count'] == 0).astype(int)
    out['target_btts_first_half'] = ((out['home_team_goal_count_half_time'] > 0) & (out['away_team_goal_count_half_time'] > 0)).astype(int)

    out['view_baseline_core_only'] = 1
    out['view_hybrid_core_plus_api'] = 1
    out['view_api_only_experimental'] = 1
    out['api_team_ready_flag'] = ((out['home_team_ppg_l5'].notna()) & (out['combined_btts_rate_l5'].notna())).astype(int)
    out['api_event_ready_flag'] = ((out['home_first_goal_rate_l10'].notna()) & (out['combined_chaos_index_l10'].notna())).astype(int)
    out['api_lineup_ready_flag'] = ((out['home_formation'].notna()) & (out['xi_rating_delta'].notna())).astype(int)
    out['api_injury_ready_flag'] = ((out['home_injured_players_count'].notna()) & (out['absence_severity_delta'].notna())).astype(int)
    out['api_odds_ready_flag'] = ((out['bookie_home_prob_norm'].notna()) & (out['home_market_disagreement'].notna())).astype(int)
    identity_home_attack = out['home_attack_strength'] if 'home_attack_strength' in out.columns else pd.Series([pd.NA] * len(out))
    identity_home_defence = out['home_defensive_strength'] if 'home_defensive_strength' in out.columns else pd.Series([pd.NA] * len(out))
    identity_home_midfield = out['home_midfield_control'] if 'home_midfield_control' in out.columns else pd.Series([pd.NA] * len(out))
    interaction_attack_gap = out['home_attack_vs_away_defence_gap'] if 'home_attack_vs_away_defence_gap' in out.columns else pd.Series([pd.NA] * len(out))
    interaction_chaos = out['both_teams_chaos_interaction'] if 'both_teams_chaos_interaction' in out.columns else pd.Series([pd.NA] * len(out))
    interaction_press = out['press_mismatch_index'] if 'press_mismatch_index' in out.columns else pd.Series([pd.NA] * len(out))
    out['api_identity_ready_flag'] = (identity_home_attack.notna() & identity_home_defence.notna() & identity_home_midfield.notna()).astype(int)
    out['api_interaction_ready_flag'] = (interaction_attack_gap.notna() & interaction_chaos.notna() & interaction_press.notna()).astype(int)
    h2h_goal_env = out['h2h_goal_environment'] if 'h2h_goal_environment' in out.columns else pd.Series([pd.NA] * len(out))
    h2h_btts = out['h2h_btts_regime'] if 'h2h_btts_regime' in out.columns else pd.Series([pd.NA] * len(out))
    ref_strict = out['ref_strictness_score'] if 'ref_strictness_score' in out.columns else pd.Series([pd.NA] * len(out))
    ref_bookings = out['ref_bookings_per_match'] if 'ref_bookings_per_match' in out.columns else pd.Series([pd.NA] * len(out))
    out['api_h2h_ready_flag'] = (h2h_goal_env.notna() & h2h_btts.notna()).astype(int)
    out['api_referee_ready_flag'] = (ref_strict.notna() & ref_bookings.notna()).astype(int)

    out = out.drop_duplicates(subset=['fixture_id']).sort_values(['match_date','fixture_id']).reset_index(drop=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=PURPOSE)
    parser.add_argument('--footystats-merged-csv', default='Matches/__merged__/England_Premier_League__merged.csv')
    parser.add_argument('--join-audit-csv', default=str(REPORT_FILES['api_footystats_join_audit']))
    parser.add_argument('--api-enriched-csv', default=str(FEATURE_FILES['api_enriched_fixture_features']))
    parser.add_argument('--output-csv', default=str(TARGET_PATH))
    args = parser.parse_args()
    df = build_hybrid_match_training(args.footystats_merged_csv, args.join_audit_csv, args.api_enriched_csv, args.output_csv)
    print(f'WROTE: {args.output_csv} rows={len(df)} cols={len(df.columns)}')


if __name__ == '__main__':
    main()
