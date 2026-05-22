from __future__ import annotations

import argparse

import pandas as pd

from .paths import HYBRID_FILES, REPORT_FILES

PURPOSE = 'Audit hybrid training feature coverage by column and feature family.'
TARGET_PATH = REPORT_FILES['api_hybrid_feature_coverage']
FAMILIES = {
    'footystats_core': ['Pre-Match PPG (Home)','Pre-Match PPG (Away)','home_ppg','away_ppg','team_a_xg','team_b_xg','btts_percentage_pre_match','over_25_percentage_pre_match'],
    'api_team': ['home_team_ppg_l5','combined_btts_rate_l5','combined_over25_rate_l5','home_shots_l5'],
    'api_event': ['home_first_goal_rate_l10','combined_chaos_index_l10'],
    'api_lineup': ['home_formation','xi_rating_delta'],
    'api_injury': ['home_injured_players_count','absence_severity_delta'],
    'api_odds': ['bookie_home_prob_norm','home_market_disagreement'],
    'targets': ['target_ftr_home','target_btts_yes','target_ou25_over'],
}


def build_coverage(hybrid_csv: str) -> pd.DataFrame:
    df = pd.read_csv(hybrid_csv)
    rows = []
    total = len(df)
    for col in df.columns:
        nn = int(df[col].notna().sum())
        rows.append({'row_type': 'column', 'name': col, 'non_null_rows': nn, 'coverage_pct': (nn / total) if total else 0.0})
    for family, cols in FAMILIES.items():
        present = [c for c in cols if c in df.columns]
        if not present:
            rows.append({'row_type': 'family', 'name': family, 'non_null_rows': 0, 'coverage_pct': 0.0})
            continue
        family_non_null = int(df[present].notna().all(axis=1).sum())
        rows.append({'row_type': 'family', 'name': family, 'non_null_rows': family_non_null, 'coverage_pct': (family_non_null / total) if total else 0.0, 'present_cols': ','.join(present)})
    out = pd.DataFrame(rows)
    out.to_csv(TARGET_PATH, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=PURPOSE)
    parser.add_argument('--hybrid-csv', default=str(HYBRID_FILES['hybrid_match_training_epl']))
    args = parser.parse_args()
    df = build_coverage(args.hybrid_csv)
    print(f'WROTE: {TARGET_PATH} rows={len(df)}')


if __name__ == '__main__':
    main()
