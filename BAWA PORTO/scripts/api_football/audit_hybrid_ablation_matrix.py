from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from .hybrid_training_utils import BASELINE_CORE_FEATURES, EXCLUDE_ALWAYS, load_training_frame
from .paths import HYBRID_FILES, REPORT_FILES, ensure_dirs

PURPOSE = 'Run feature-family ablation audit across baseline and incremental API booster stacks.'
DEFAULT_INPUT = HYBRID_FILES['hybrid_match_training_epl']
DEFAULT_OUTPUT = REPORT_FILES['api_feature_uplift_matrix'].parent / 'api_hybrid_ablation_matrix.csv'

TARGET_MAP = {
    'ftr': ['target_ftr_home', 'target_ftr_draw', 'target_ftr_away'],
    'btts': ['target_btts_yes'],
    'ou25': ['target_ou25_over'],
}

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

STACKS = [
    ('baseline', []),
    ('baseline_plus_team', ['team']),
    ('baseline_plus_team_lineup', ['team', 'lineup']),
    ('baseline_plus_team_lineup_injury', ['team', 'lineup', 'injury']),
    ('baseline_plus_team_lineup_injury_event', ['team', 'lineup', 'injury', 'event']),
]

EXTRA_EXCLUDE = {
    'home_team_goal_count', 'away_team_goal_count', 'home_team_goal_count_half_time', 'away_team_goal_count_half_time',
    'home_team_name', 'away_team_name', 'api_home_team_name', 'api_away_team_name',
    'league', 'season'
}


def _baseline_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in BASELINE_CORE_FEATURES if c in df.columns]


def _family_cols(df: pd.DataFrame, family: str) -> List[str]:
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


def _stack_cols(df: pd.DataFrame, stack_families: Sequence[str]) -> List[str]:
    cols = list(_baseline_cols(df))
    for family in stack_families:
        cols.extend(_family_cols(df, family))
    seen = set()
    out: List[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _build_target(df: pd.DataFrame, market: str):
    if market == 'ftr':
        return (
            df[['target_ftr_home', 'target_ftr_draw', 'target_ftr_away']]
            .idxmax(axis=1)
            .map({'target_ftr_home': 0, 'target_ftr_draw': 1, 'target_ftr_away': 2})
            .astype(int)
            .to_numpy()
        )
    return df[TARGET_MAP[market][0]].astype(int).to_numpy()


def _split(df: pd.DataFrame, y, holdout_frac: float):
    split_idx = max(1, int(len(df) * (1.0 - holdout_frac)))
    split_idx = min(split_idx, len(df) - 1)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy(), y[:split_idx], y[split_idx:]


def _score_stack(df: pd.DataFrame, market: str, stack_name: str, stack_families: Sequence[str], holdout_frac: float, random_seed: int) -> Dict[str, object]:
    feature_cols = _stack_cols(df, stack_families)
    X = df[feature_cols].copy()
    categorical_cols = [c for c in X.columns if X[c].dtype == 'object']
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].fillna('MISSING')
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce')
    y = _build_target(df, market)
    X_train, X_test, y_train, y_test = _split(X, y, holdout_frac)
    cat_idx = [X_train.columns.get_loc(c) for c in categorical_cols]
    model = CatBoostClassifier(
        random_seed=random_seed,
        verbose=False,
        loss_function='MultiClass' if market == 'ftr' else 'Logloss',
        eval_metric='MultiClass' if market == 'ftr' else 'Logloss',
        depth=6,
        learning_rate=0.05,
        iterations=250,
    )
    model.fit(X_train, y_train, cat_features=cat_idx, eval_set=(X_test, y_test), use_best_model=True)
    pred = model.predict(X_test)
    pred = pd.Series(pred.reshape(-1)).astype(int).to_numpy()
    proba = model.predict_proba(X_test)
    row: Dict[str, object] = {
        'market': market,
        'stack': stack_name,
        'families': '|'.join(stack_families),
        'rows_train': len(X_train),
        'rows_test': len(X_test),
        'n_features': len(feature_cols),
        'accuracy': float(accuracy_score(y_test, pred)),
        'log_loss': float(log_loss(y_test, proba)),
        'auc': None,
    }
    if market != 'ftr':
        try:
            row['auc'] = float(roc_auc_score(y_test, proba[:, 1]))
        except Exception:
            row['auc'] = None
    return row


def build_ablation_matrix(input_csv: str, output_csv: str, holdout_frac: float, random_seed: int) -> pd.DataFrame:
    ensure_dirs()
    df = load_training_frame(Path(input_csv))
    rows: List[Dict[str, object]] = []
    for market in ['ftr', 'btts', 'ou25']:
        for stack_name, families in STACKS:
            rows.append(_score_stack(df, market, stack_name, families, holdout_frac, random_seed))
    out = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=PURPOSE)
    parser.add_argument('--input-csv', default=str(DEFAULT_INPUT))
    parser.add_argument('--output-csv', default=str(DEFAULT_OUTPUT))
    parser.add_argument('--holdout-frac', type=float, default=0.2)
    parser.add_argument('--random-seed', type=int, default=42)
    args = parser.parse_args()
    df = build_ablation_matrix(args.input_csv, args.output_csv, args.holdout_frac, args.random_seed)
    print(f'WROTE: {args.output_csv} rows={len(df)}')


if __name__ == '__main__':
    main()
