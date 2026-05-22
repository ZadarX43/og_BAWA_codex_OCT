from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .paths import FEATURE_FILES, NORMALIZED_FILES, REPORT_FILES
from .team_name_map import normalize_team_name

PURPOSE = 'Write fixture join audit between FootyStats and API-Football fixture identities.'
TARGET_PATH = REPORT_FILES['api_footystats_join_audit']


def normalize_name(value: object, tag: str | None = None) -> str:
    return normalize_team_name(value, tag=tag)


def add_team_keys(df: pd.DataFrame, home_col: str, away_col: str, date_col: str, tag: str | None = None) -> pd.DataFrame:
    out = df.copy()
    out['_norm_home'] = out[home_col].map(lambda value: normalize_name(value, tag=tag))
    out['_norm_away'] = out[away_col].map(lambda value: normalize_name(value, tag=tag))
    out['_date'] = pd.to_datetime(out[date_col], errors='coerce').dt.date
    out['_join_key'] = out['_date'].astype(str) + '|' + out['_norm_home'] + '|' + out['_norm_away']
    return out


def load_match_scores(team_stats_csv: str) -> pd.DataFrame:
    ts = pd.read_csv(team_stats_csv)
    home = ts[ts['is_home'] == 1][['fixture_id', 'goals_for', 'ht_goals_for']].rename(columns={
        'goals_for': 'api_home_goals_fulltime', 'ht_goals_for': 'api_home_goals_ht'
    })
    away = ts[ts['is_home'] == 0][['fixture_id', 'goals_for', 'ht_goals_for']].rename(columns={
        'goals_for': 'api_away_goals_fulltime', 'ht_goals_for': 'api_away_goals_ht'
    })
    return home.merge(away, on='fixture_id', how='outer')


def build_join_audit(api_fixtures_csv: str, api_enriched_csv: str, footystats_merged_csv: str, team_stats_csv: str, output_csv: str = str(TARGET_PATH)) -> pd.DataFrame:
    api_fx = pd.read_csv(api_fixtures_csv)
    api_enriched = pd.read_csv(api_enriched_csv)
    fs = pd.read_csv(footystats_merged_csv)
    api_scores = load_match_scores(team_stats_csv)
    tag = Path(footystats_merged_csv).name.replace('__merged__.csv', '').replace('__merged.csv', '')

    date_min = pd.to_datetime(api_fx['match_date'], errors='coerce').min()
    date_max = pd.to_datetime(api_fx['match_date'], errors='coerce').max()
    if 'match_date' in fs.columns:
        fs = fs[(pd.to_datetime(fs['match_date'], errors='coerce') >= date_min) & (pd.to_datetime(fs['match_date'], errors='coerce') <= date_max)].copy()
    if 'league' in fs.columns and 'league' in api_fx.columns:
        target_league = str(api_fx['league'].mode().iloc[0]).strip().lower()
        fs_league = fs['league'].astype(str).str.strip().str.lower()
        exact_mask = fs_league == target_league
        if exact_mask.any():
            fs = fs[exact_mask].copy()
        else:
            normalized_target = target_league.replace(' league', '').strip()
            contains_mask = fs_league.str.contains(normalized_target, regex=False)
            if bool(contains_mask.any()):
                fs = fs[contains_mask].copy()

    api = api_fx.merge(api_enriched, on=['fixture_id','fixture_key','league','league_id','season','match_date','home_team_id','away_team_id','home_team_name','away_team_name'], how='left')
    api = api.merge(api_scores, on='fixture_id', how='left')

    api = add_team_keys(api, 'home_team_name', 'away_team_name', 'match_date', tag=tag)
    fs = add_team_keys(fs, 'home_team_name', 'away_team_name', 'match_date', tag=tag)

    exact_map = {k: i for i, k in fs['_join_key'].items()}
    fs_by_name = {}
    for idx, row in fs.iterrows():
        fs_by_name.setdefault((row['_norm_home'], row['_norm_away']), []).append(idx)

    out_rows = []
    for _, row in api.iterrows():
        join_method = 'UNMATCHED'
        date_delta_days = None
        fs_row = None
        idx = exact_map.get(row['_join_key'])
        if idx is not None:
            fs_row = fs.loc[idx]
            join_method = 'EXACT'
            date_delta_days = 0
        else:
            candidates = fs_by_name.get((row['_norm_home'], row['_norm_away']), [])
            if candidates:
                api_date = pd.to_datetime(row['match_date'], errors='coerce').date()
                best = None
                best_delta = None
                for cand_idx in candidates:
                    cand = fs.loc[cand_idx]
                    cand_date = cand['_date']
                    if pd.isna(pd.Timestamp(cand_date)):
                        continue
                    delta = abs((api_date - cand_date).days)
                    if best_delta is None or delta < best_delta:
                        best, best_delta = cand_idx, delta
                if best is not None and best_delta is not None and best_delta <= 1:
                    fs_row = fs.loc[best]
                    join_method = 'FUZZY_DATE'
                    date_delta_days = best_delta
        if fs_row is None:
            out_rows.append({
                'fixture_id': row['fixture_id'],
                'fixture_key': row['fixture_key'],
                'league': row['league'],
                'match_date': row['match_date'],
                'api_home_team_name': row['home_team_name'],
                'api_away_team_name': row['away_team_name'],
                'join_found_flag': 0,
                'join_method': join_method,
                'date_delta_days': date_delta_days,
            })
            continue

        alias_needed = int(
            (normalize_name(fs_row['home_team_name'], tag=tag) != normalize_team_name(str(fs_row['home_team_name']).lower(), tag=tag))
            or (normalize_name(fs_row['away_team_name'], tag=tag) != normalize_team_name(str(fs_row['away_team_name']).lower(), tag=tag))
        )
        out_rows.append({
            'fixture_id': row['fixture_id'],
            'fixture_key': row['fixture_key'],
            'league': row['league'],
            'match_date': row['match_date'],
            'api_home_team_name': row['home_team_name'],
            'api_away_team_name': row['away_team_name'],
            'fs_match_date': fs_row.get('match_date'),
            'fs_home_team_name': fs_row.get('home_team_name'),
            'fs_away_team_name': fs_row.get('away_team_name'),
            'join_found_flag': 1,
            'join_method': join_method,
            'date_delta_days': date_delta_days,
            'name_alias_applied_flag': alias_needed,
            'api_home_goals_fulltime': row.get('api_home_goals_fulltime'),
            'api_away_goals_fulltime': row.get('api_away_goals_fulltime'),
            'fs_home_goals_fulltime': fs_row.get('home_team_goal_count'),
            'fs_away_goals_fulltime': fs_row.get('away_team_goal_count'),
            'fulltime_score_match_flag': int((row.get('api_home_goals_fulltime') == fs_row.get('home_team_goal_count')) and (row.get('api_away_goals_fulltime') == fs_row.get('away_team_goal_count'))),
            'api_home_goals_ht': row.get('api_home_goals_ht'),
            'api_away_goals_ht': row.get('api_away_goals_ht'),
            'fs_home_goals_ht': fs_row.get('home_team_goal_count_half_time'),
            'fs_away_goals_ht': fs_row.get('away_team_goal_count_half_time'),
            'halftime_score_match_flag': int((row.get('api_home_goals_ht') == fs_row.get('home_team_goal_count_half_time')) and (row.get('api_away_goals_ht') == fs_row.get('away_team_goal_count_half_time'))),
            'fs_pre_match_ppg_home': fs_row.get('Pre-Match PPG (Home)'),
            'fs_pre_match_ppg_away': fs_row.get('Pre-Match PPG (Away)'),
            'api_home_team_ppg_l5': row.get('home_team_ppg_l5'),
            'api_away_team_ppg_l5': row.get('away_team_ppg_l5'),
            'api_home_team_ppg_season': row.get('home_team_ppg_season'),
            'api_away_team_ppg_season': row.get('away_team_ppg_season'),
            'fs_btts_percentage_pre_match': fs_row.get('btts_percentage_pre_match'),
            'api_combined_btts_rate_l5': row.get('combined_btts_rate_l5'),
            'fs_over_25_percentage_pre_match': fs_row.get('over_25_percentage_pre_match'),
            'api_combined_over25_rate_l5': row.get('combined_over25_rate_l5'),
            'fs_over_35_percentage_pre_match': fs_row.get('over_35_percentage_pre_match'),
            'api_combined_over35_rate_l5': row.get('combined_over35_rate_l5'),
            'fs_average_goals_per_match_pre_match': fs_row.get('average_goals_per_match_pre_match'),
            'api_combined_total_goals_l5': row.get('combined_total_goals_l5'),
            'api_team_feature_family_ready_flag': int(pd.notna(row.get('home_team_ppg_l5')) and pd.notna(row.get('combined_btts_rate_l5'))),
            'api_event_feature_family_ready_flag': int(pd.notna(row.get('home_first_goal_rate_l10')) and pd.notna(row.get('combined_chaos_index_l10'))),
            'api_lineup_feature_family_ready_flag': int(pd.notna(row.get('home_formation')) and pd.notna(row.get('xi_rating_delta'))),
        })

    out = pd.DataFrame(out_rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=PURPOSE)
    parser.add_argument('--api-fixtures-csv', default=str(NORMALIZED_FILES['fixtures_master']))
    parser.add_argument('--api-enriched-csv', default=str(FEATURE_FILES['api_enriched_fixture_features']))
    parser.add_argument('--footystats-merged-csv', required=False, default='Matches/__merged__/England_Premier_League__merged.csv')
    parser.add_argument('--team-stats-csv', default=str(NORMALIZED_FILES['match_team_stats']))
    parser.add_argument('--output-csv', default=str(TARGET_PATH))
    args = parser.parse_args()
    df = build_join_audit(args.api_fixtures_csv, args.api_enriched_csv, args.footystats_merged_csv, args.team_stats_csv, args.output_csv)
    print(f'WROTE: {args.output_csv} rows={len(df)} matched={int(df["join_found_flag"].fillna(0).sum())}')
    if 'join_method' in df.columns:
        print('join_method_counts=', df['join_method'].fillna('EMPTY').value_counts().to_dict())


if __name__ == '__main__':
    main()
