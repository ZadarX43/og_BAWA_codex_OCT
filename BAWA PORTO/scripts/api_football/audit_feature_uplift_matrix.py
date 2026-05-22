from __future__ import annotations

import argparse
import math

import pandas as pd

from .audit_footystats_join import normalize_name
from .paths import FEATURE_FILES, REPORT_FILES
from .utils import safe_div

PURPOSE = 'Write baseline-vs-enriched uplift comparison scaffold.'
TARGET_PATH = REPORT_FILES['api_feature_uplift_matrix']
JOIN_AUDIT_DEFAULT = REPORT_FILES['api_footystats_join_audit']
FOOTYSTATS_DEFAULT = 'Matches/__merged__/England_Premier_League__merged.csv'
API_ENRICHED_DEFAULT = FEATURE_FILES['api_enriched_fixture_features']


def _sigmoid(x: float) -> float:
    x = max(min(float(x), 12.0), -12.0)
    return 1.0 / (1.0 + math.exp(-x))


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _auc_binary(y_true: pd.Series, score: pd.Series) -> float:
    df = pd.DataFrame({'y': y_true, 's': score}).dropna()
    if df.empty or df['y'].nunique() < 2:
        return float('nan')
    df = df.sort_values('s')
    ranks = df['s'].rank(method='average')
    pos = df['y'] == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    rank_sum = ranks[pos].sum()
    return float((rank_sum - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg))


def _binary_row(market: str, source: str, feature_family: str, features_used: str, y_true: pd.Series, prob: pd.Series) -> dict:
    df = pd.DataFrame({'y': y_true, 'p': prob}).dropna().copy()
    if df.empty:
        return {'market': market, 'source': source, 'feature_family': feature_family, 'features_used': features_used, 'sample_n': 0}
    df['p'] = df['p'].map(_clip01)
    q1 = df['p'].quantile(0.25)
    q3 = df['p'].quantile(0.75)
    top = df[df['p'] >= q3]
    bottom = df[df['p'] <= q1]
    pred = (df['p'] >= 0.5).astype(int)
    acc = float((pred == df['y']).mean())
    brier = float(((df['p'] - df['y']) ** 2).mean())
    eps = 1e-6
    logloss = float((-(df['y'] * (df['p'] + eps).map(math.log) + (1 - df['y']) * (1 - df['p'] + eps).map(math.log))).mean())
    return {
        'market': market,
        'source': source,
        'feature_family': feature_family,
        'features_used': features_used,
        'sample_n': int(len(df)),
        'coverage_pct': float(len(df) / len(y_true)) if len(y_true) else 0.0,
        'positive_rate': float(df['y'].mean()),
        'auc': _auc_binary(df['y'], df['p']),
        'accuracy_at_0p50': acc,
        'brier_score': brier,
        'log_loss': logloss,
        'top_quartile_hit_rate': float(top['y'].mean()) if not top.empty else float('nan'),
        'bottom_quartile_hit_rate': float(bottom['y'].mean()) if not bottom.empty else float('nan'),
        'lift_q4_vs_q1': float(top['y'].mean() - bottom['y'].mean()) if (not top.empty and not bottom.empty) else float('nan'),
    }


def _ftr_row(source: str, feature_family: str, features_used: str, home_prob: pd.Series, draw_prob: pd.Series, away_prob: pd.Series, y_actual: pd.Series) -> dict:
    df = pd.DataFrame({'home': home_prob, 'draw': draw_prob, 'away': away_prob, 'actual': y_actual}).dropna().copy()
    if df.empty:
        return {'market': 'FTR_3WAY', 'source': source, 'feature_family': feature_family, 'features_used': features_used, 'sample_n': 0}
    probs = df[['home', 'draw', 'away']].clip(lower=0.0)
    prob_sum = probs.sum(axis=1).replace(0, 1.0)
    probs = probs.div(prob_sum, axis=0)
    labels = pd.Series(['HOME', 'DRAW', 'AWAY'])
    pred = probs.idxmax(axis=1).str.upper()
    acc = float((pred == df['actual']).mean())
    multiclass_brier = float((((probs['home'] - (df['actual'] == 'HOME').astype(int)) ** 2) + ((probs['draw'] - (df['actual'] == 'DRAW').astype(int)) ** 2) + ((probs['away'] - (df['actual'] == 'AWAY').astype(int)) ** 2)).mean())
    return {
        'market': 'FTR_3WAY',
        'source': source,
        'feature_family': feature_family,
        'features_used': features_used,
        'sample_n': int(len(df)),
        'coverage_pct': float(len(df) / len(y_actual)) if len(y_actual) else 0.0,
        'accuracy_top_pick': acc,
        'multiclass_brier_score': multiclass_brier,
        'pred_home_rate': float((pred == 'HOME').mean()),
        'pred_draw_rate': float((pred == 'DRAW').mean()),
        'pred_away_rate': float((pred == 'AWAY').mean()),
    }


def build_uplift_matrix(join_audit_csv: str, footystats_merged_csv: str, api_enriched_csv: str) -> pd.DataFrame:
    ja = pd.read_csv(join_audit_csv)
    fs = pd.read_csv(footystats_merged_csv)
    api = pd.read_csv(api_enriched_csv)

    fs = fs.copy()
    fs['_norm_home'] = fs['home_team_name'].map(normalize_name)
    fs['_norm_away'] = fs['away_team_name'].map(normalize_name)
    fs['_date'] = pd.to_datetime(fs['match_date'], errors='coerce').dt.date.astype(str)
    fs['_join_key'] = fs['_date'] + '|' + fs['_norm_home'] + '|' + fs['_norm_away']

    ja = ja[ja['join_found_flag'] == 1].copy()
    ja['_norm_home'] = ja['fs_home_team_name'].map(normalize_name)
    ja['_norm_away'] = ja['fs_away_team_name'].map(normalize_name)
    ja['_date'] = pd.to_datetime(ja['fs_match_date'], errors='coerce').dt.date.astype(str)
    ja['_join_key'] = ja['_date'] + '|' + ja['_norm_home'] + '|' + ja['_norm_away']

    base = ja.merge(fs, on='_join_key', how='left', suffixes=('', '_fsfull'))
    api_keep = ['fixture_id'] + [c for c in api.columns if c != 'fixture_id']
    base = base.merge(api[api_keep], on='fixture_id', how='left', suffixes=('', '_api'))

    base['target_btts'] = ((base['home_team_goal_count'] > 0) & (base['away_team_goal_count'] > 0)).astype(int)
    base['target_ou25'] = ((base['home_team_goal_count'] + base['away_team_goal_count']) >= 3).astype(int)
    base['target_home_win'] = (base['home_team_goal_count'] > base['away_team_goal_count']).astype(int)
    base['target_away_win'] = (base['away_team_goal_count'] > base['home_team_goal_count']).astype(int)
    base['target_ftr'] = 'DRAW'
    base.loc[base['home_team_goal_count'] > base['away_team_goal_count'], 'target_ftr'] = 'HOME'
    base.loc[base['away_team_goal_count'] > base['home_team_goal_count'], 'target_ftr'] = 'AWAY'

    raw_probs = pd.DataFrame({
        'home': 1.0 / base['odds_ft_home_team_win'].replace(0, pd.NA),
        'draw': 1.0 / base['odds_ft_draw'].replace(0, pd.NA),
        'away': 1.0 / base['odds_ft_away_team_win'].replace(0, pd.NA),
    })
    raw_sum = raw_probs.sum(axis=1)
    base['bookie_home_prob_norm'] = raw_probs['home'] / raw_sum
    base['bookie_draw_prob_norm'] = raw_probs['draw'] / raw_sum
    base['bookie_away_prob_norm'] = raw_probs['away'] / raw_sum

    baseline_edge = (
        (base['Pre-Match PPG (Home)'] - base['Pre-Match PPG (Away)']) +
        (base['home_ppg'] - base['away_ppg']) +
        0.75 * (base['team_a_xg'] - base['team_b_xg'])
    )
    base['baseline_home_prob'] = 0.55 * base['bookie_home_prob_norm'] + 0.45 * baseline_edge.map(_sigmoid)
    base['baseline_away_prob'] = 0.55 * base['bookie_away_prob_norm'] + 0.45 * (-baseline_edge).map(_sigmoid)
    base['baseline_draw_prob'] = (0.60 * base['bookie_draw_prob_norm']) + 0.40 * (1.0 - (base['baseline_home_prob'] - base['baseline_away_prob']).abs()).clip(lower=0.0)

    attack_strength_home = (0.6 * base['home_goals_for_l5']) + (0.3 * base['home_sot_l5']) + (0.1 * base['home_shots_inside_box_l5'])
    attack_strength_away = (0.6 * base['away_goals_for_l5']) + (0.3 * base['away_sot_l5']) + (0.1 * base['away_shots_inside_box_l5'])
    defence_strength_home = (0.6 * base['home_goals_against_l5']) + (0.25 * base['away_sot_l5']) + (0.15 * base['away_shots_inside_box_l5'])
    defence_strength_away = (0.6 * base['away_goals_against_l5']) + (0.25 * base['home_sot_l5']) + (0.15 * base['home_shots_inside_box_l5'])
    api_edge = (
        0.40 * base['ppg_diff_l5'] +
        0.30 * base['ppg_diff_season'] +
        0.20 * (attack_strength_home - attack_strength_away) +
        0.10 * (defence_strength_away - defence_strength_home) +
        0.08 * base['shot_delta_l5'] +
        0.05 * base['xi_rating_delta'] +
        0.03 * (base['home_first_goal_rate_l10'] - base['away_first_goal_rate_l10'])
    )
    base['api_home_prob'] = api_edge.map(_sigmoid)
    base['api_away_prob'] = (-api_edge).map(_sigmoid)
    base['api_draw_prob'] = (1.0 - (base['api_home_prob'] - base['api_away_prob']).abs() - 0.15 * base['combined_total_goals_l5'] / 4.0 - 0.10 * base['combined_btts_rate_l5']).clip(lower=0.0)

    base['baseline_btts_prob'] = (base['btts_percentage_pre_match'] / 100.0).clip(lower=0.0, upper=1.0)
    base['api_btts_prob'] = (
        0.35 * base['combined_btts_rate_l5'] +
        0.15 * base['home_scored_rate_l5'] +
        0.15 * base['away_scored_rate_l5'] +
        0.175 * base['home_conceded_rate_l5'] +
        0.175 * base['away_conceded_rate_l5']
    ).clip(lower=0.0, upper=1.0)

    base['baseline_ou25_prob'] = (base['over_25_percentage_pre_match'] / 100.0).clip(lower=0.0, upper=1.0)
    total_goals_proxy = (base['combined_total_goals_l5'] / 4.0).clip(lower=0.0, upper=1.0)
    shot_proxy = ((base['home_sot_l5'] + base['away_sot_l5']) / 10.0).clip(lower=0.0, upper=1.0)
    base['api_ou25_prob'] = (
        0.55 * base['combined_over25_rate_l5'] +
        0.25 * total_goals_proxy +
        0.20 * shot_proxy
    ).clip(lower=0.0, upper=1.0)

    rows = []
    base['hybrid_home_prob'] = (0.70 * base['baseline_home_prob']) + (0.30 * base['api_home_prob'])
    base['hybrid_away_prob'] = (0.70 * base['baseline_away_prob']) + (0.30 * base['api_away_prob'])
    base['hybrid_draw_prob'] = (0.75 * base['baseline_draw_prob']) + (0.25 * base['api_draw_prob'])
    base['hybrid_btts_prob'] = (0.70 * base['baseline_btts_prob']) + (0.30 * base['api_btts_prob'])
    base['hybrid_ou25_prob'] = (0.70 * base['baseline_ou25_prob']) + (0.30 * base['api_ou25_prob'])

    rows.append(_ftr_row('FOOTYSTATS_BASELINE', 'CORE', 'Pre-Match PPG, team_xg, normalized odds', base['baseline_home_prob'], base['baseline_draw_prob'], base['baseline_away_prob'], base['target_ftr']))
    rows.append(_ftr_row('HYBRID_BASELINE_PLUS_API', 'CORE+BOOSTERS', 'FootyStats baseline blended with API ppg/attack/shot/lineup/event signals', base['hybrid_home_prob'], base['hybrid_draw_prob'], base['hybrid_away_prob'], base['target_ftr']))
    rows.append(_ftr_row('API_ONLY', 'CORE+BOOSTERS', 'ppg_diff, attack/defence strength proxy, shots, lineup delta, first-goal rate', base['api_home_prob'], base['api_draw_prob'], base['api_away_prob'], base['target_ftr']))
    rows.append(_binary_row('BTTS', 'FOOTYSTATS_BASELINE', 'CORE', 'btts_percentage_pre_match', base['target_btts'], base['baseline_btts_prob']))
    rows.append(_binary_row('BTTS', 'HYBRID_BASELINE_PLUS_API', 'CORE+BOOSTERS', 'FootyStats BTTS% blended with API BTTS/scored/conceded rates', base['target_btts'], base['hybrid_btts_prob']))
    rows.append(_binary_row('BTTS', 'API_ONLY', 'CORE+BOOSTERS', 'combined_btts_rate_l5 + scored/conceded rates', base['target_btts'], base['api_btts_prob']))
    rows.append(_binary_row('OU25', 'FOOTYSTATS_BASELINE', 'CORE', 'over_25_percentage_pre_match', base['target_ou25'], base['baseline_ou25_prob']))
    rows.append(_binary_row('OU25', 'HYBRID_BASELINE_PLUS_API', 'CORE+BOOSTERS', 'FootyStats O2.5% blended with API O2.5/goal/SOT proxies', base['target_ou25'], base['hybrid_ou25_prob']))
    rows.append(_binary_row('OU25', 'API_ONLY', 'CORE+BOOSTERS', 'combined_over25_rate_l5 + total goals proxy + SOT proxy', base['target_ou25'], base['api_ou25_prob']))
    rows.append(_binary_row('FTR_HOME_WIN', 'FOOTYSTATS_BASELINE', 'CORE', 'Pre-Match PPG, team_xg, normalized odds', base['target_home_win'], base['baseline_home_prob']))
    rows.append(_binary_row('FTR_HOME_WIN', 'HYBRID_BASELINE_PLUS_API', 'CORE+BOOSTERS', 'Baseline home-win probability blended with API ppg/attack/lineup/event boosts', base['target_home_win'], base['hybrid_home_prob']))
    rows.append(_binary_row('FTR_HOME_WIN', 'API_ONLY', 'CORE+BOOSTERS', 'ppg_diff + attack/defence strength + lineup/event boosts', base['target_home_win'], base['api_home_prob']))
    rows.append(_binary_row('FTR_AWAY_WIN', 'FOOTYSTATS_BASELINE', 'CORE', 'Pre-Match PPG, team_xg, normalized odds', base['target_away_win'], base['baseline_away_prob']))
    rows.append(_binary_row('FTR_AWAY_WIN', 'HYBRID_BASELINE_PLUS_API', 'CORE+BOOSTERS', 'Baseline away-win probability blended with API ppg/attack/lineup/event boosts', base['target_away_win'], base['hybrid_away_prob']))
    rows.append(_binary_row('FTR_AWAY_WIN', 'API_ONLY', 'CORE+BOOSTERS', 'ppg_diff + attack/defence strength + lineup/event boosts', base['target_away_win'], base['api_away_prob']))

    out = pd.DataFrame(rows)
    out.to_csv(TARGET_PATH, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=PURPOSE)
    parser.add_argument('--join-audit-csv', default=str(JOIN_AUDIT_DEFAULT))
    parser.add_argument('--footystats-merged-csv', default=FOOTYSTATS_DEFAULT)
    parser.add_argument('--api-enriched-csv', default=str(API_ENRICHED_DEFAULT))
    args = parser.parse_args()
    df = build_uplift_matrix(args.join_audit_csv, args.footystats_merged_csv, args.api_enriched_csv)
    print(f'WROTE: {TARGET_PATH} rows={len(df)}')
    print(df[['market','source']].to_dict(orient='records'))


if __name__ == '__main__':
    main()
