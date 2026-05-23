#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, log_loss, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = PROJECT_ROOT / 'data_sources' / 'hybrid' / 'hybrid_goal_mass_inputs__England_Premier_League.csv'
DEFAULT_MODEL_DIR = PROJECT_ROOT / 'ModelStore' / 'Hybrid' / 'England_Premier_League'
DEFAULT_REPORT_CSV = PROJECT_ROOT / 'reports' / 'api_football' / 'hybrid_goal_mass_metrics__England_Premier_League.csv'

ID_COLS = {
    'fixture_id', 'fixture_key', 'league', 'season', 'match_date', 'home_team_name', 'away_team_name'
}
TARGET_COLS = {
    'home_team_goal_count', 'away_team_goal_count',
    'home_team_goal_count_half_time', 'away_team_goal_count_half_time',
    'target_ftr_home', 'target_ftr_draw', 'target_ftr_away',
    'target_home_goals_over15', 'target_away_goals_over15',
    'target_home_fts', 'target_away_fts',
    'target_btts_yes', 'target_btts_first_half', 'target_ou25_over',
    'target_home_goals', 'target_away_goals', 'target_total_goals'
}
EXCLUDE_COLS = ID_COLS | TARGET_COLS


def _load_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'match_date' in df.columns:
        df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
    return df.sort_values(['match_date', 'fixture_id']).reset_index(drop=True)


def _feature_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    return cols


def _prep_features(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    X = df[cols].copy()
    cat_cols = [c for c in X.columns if X[c].dtype == 'object']
    for c in X.columns:
        if X[c].dtype == 'object':
            X[c] = X[c].fillna('MISSING')
        else:
            X[c] = pd.to_numeric(X[c], errors='coerce')
    return X, cat_cols


def _split(df: pd.DataFrame, holdout_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = max(1, int(len(df) * (1.0 - holdout_frac)))
    split_idx = min(split_idx, len(df) - 1)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def _pois_prob_ge2(lmbda: np.ndarray) -> np.ndarray:
    l = np.clip(np.asarray(lmbda, dtype=float), 0.0, None)
    return 1.0 - np.exp(-l) * (1.0 + l)


def _pois_prob_eq0(lmbda: np.ndarray) -> np.ndarray:
    l = np.clip(np.asarray(lmbda, dtype=float), 0.0, None)
    return np.exp(-l)


def _fit_regressor(X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray, cat_cols: List[str], seed: int) -> CatBoostRegressor:
    cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]
    model = CatBoostRegressor(
        random_seed=seed,
        verbose=False,
        depth=6,
        learning_rate=0.05,
        iterations=300,
        loss_function='RMSE',
        eval_metric='RMSE',
    )
    model.fit(X_train, y_train, cat_features=cat_idx, eval_set=(X_val, y_val), use_best_model=True)
    return model


def _clip_preds(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), 0.0, 6.0)


def _binary_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float | None]:
    p = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    pred = (p >= 0.5).astype(int)
    out: Dict[str, float | None] = {
        'accuracy': float(accuracy_score(y_true, pred)),
        'log_loss': None,
        'auc': None,
    }
    labels = np.unique(np.asarray(y_true, dtype=int))
    if len(labels) >= 2:
        try:
            out['log_loss'] = float(log_loss(y_true, p, labels=[0, 1]))
        except Exception:
            out['log_loss'] = None
        try:
            out['auc'] = float(roc_auc_score(y_true, p))
        except Exception:
            out['auc'] = None
    return out


def train_goal_mass(input_csv: Path, model_dir: Path, report_csv: Path, holdout_frac: float, random_seed: int) -> pd.DataFrame:
    df = _load_frame(input_csv)
    feat_cols = _feature_cols(df)
    train_df, test_df = _split(df, holdout_frac)
    X_train, cat_cols = _prep_features(train_df, feat_cols)
    X_test, _ = _prep_features(test_df, feat_cols)

    y_home_train = pd.to_numeric(train_df['target_home_goals'], errors='coerce').fillna(0).to_numpy()
    y_away_train = pd.to_numeric(train_df['target_away_goals'], errors='coerce').fillna(0).to_numpy()
    y_home_test = pd.to_numeric(test_df['target_home_goals'], errors='coerce').fillna(0).to_numpy()
    y_away_test = pd.to_numeric(test_df['target_away_goals'], errors='coerce').fillna(0).to_numpy()

    home_model = _fit_regressor(X_train, y_home_train, X_test, y_home_test, cat_cols, random_seed)
    away_model = _fit_regressor(X_train, y_away_train, X_test, y_away_test, cat_cols, random_seed + 1)

    home_lambda = _clip_preds(home_model.predict(X_test))
    away_lambda = _clip_preds(away_model.predict(X_test))

    rows: List[Dict[str, object]] = []
    rows.append({
        'artifact': 'home_lambda_regressor',
        'metric_group': 'regression',
        'rows_train': len(train_df),
        'rows_test': len(test_df),
        'n_features': len(feat_cols),
        'mae': float(mean_absolute_error(y_home_test, home_lambda)),
        'rmse': float(math.sqrt(mean_squared_error(y_home_test, home_lambda))),
    })
    rows.append({
        'artifact': 'away_lambda_regressor',
        'metric_group': 'regression',
        'rows_train': len(train_df),
        'rows_test': len(test_df),
        'n_features': len(feat_cols),
        'mae': float(mean_absolute_error(y_away_test, away_lambda)),
        'rmse': float(math.sqrt(mean_squared_error(y_away_test, away_lambda))),
    })

    side_defs = [
        ('home_goals_over15', pd.to_numeric(test_df['target_home_goals_over15'], errors='coerce').fillna(0).astype(int).to_numpy(), _pois_prob_ge2(home_lambda)),
        ('away_goals_over15', pd.to_numeric(test_df['target_away_goals_over15'], errors='coerce').fillna(0).astype(int).to_numpy(), _pois_prob_ge2(away_lambda)),
        ('home_fts', pd.to_numeric(test_df['target_home_fts'], errors='coerce').fillna(0).astype(int).to_numpy(), _pois_prob_eq0(home_lambda)),
        ('away_fts', pd.to_numeric(test_df['target_away_fts'], errors='coerce').fillna(0).astype(int).to_numpy(), _pois_prob_eq0(away_lambda)),
        ('btts_yes', pd.to_numeric(test_df['target_btts_yes'], errors='coerce').fillna(0).astype(int).to_numpy(), (1.0 - _pois_prob_eq0(home_lambda)) * (1.0 - _pois_prob_eq0(away_lambda))),
        ('ou25_over', pd.to_numeric(test_df['target_ou25_over'], errors='coerce').fillna(0).astype(int).to_numpy(), 1.0 - np.exp(-(home_lambda + away_lambda)) * (1.0 + (home_lambda + away_lambda) + ((home_lambda + away_lambda) ** 2) / 2.0)),
    ]

    for name, y_true, proba in side_defs:
        m = _binary_metrics(y_true, proba)
        rows.append({
            'artifact': name,
            'metric_group': 'side_market',
            'rows_train': len(train_df),
            'rows_test': len(test_df),
            'n_features': len(feat_cols),
            'mae': None,
            'rmse': None,
            'accuracy': m['accuracy'],
            'log_loss': m['log_loss'],
            'auc': m['auc'],
        })

    # BTTS first-half placeholder proxy: use full BTTS probability damped by first-goal/tempo seeds if available.
    fh_anchor = np.clip(
        pd.to_numeric(test_df.get('home_first_goal_rate_l10', 0), errors='coerce').fillna(0).to_numpy() * 0.5 +
        pd.to_numeric(test_df.get('away_first_goal_rate_l10', 0), errors='coerce').fillna(0).to_numpy() * 0.5,
        0.0,
        1.0,
    )
    fh_proba = np.clip(side_defs[4][2] * (0.35 + 0.65 * fh_anchor), 1e-6, 1 - 1e-6)
    fh_true = pd.to_numeric(test_df['target_btts_first_half'], errors='coerce').fillna(0).astype(int).to_numpy()
    fh_metrics = _binary_metrics(fh_true, fh_proba)
    rows.append({
        'artifact': 'btts_first_half_proxy',
        'metric_group': 'side_market',
        'rows_train': len(train_df),
        'rows_test': len(test_df),
        'n_features': len(feat_cols),
        'mae': None,
        'rmse': None,
        'accuracy': fh_metrics['accuracy'],
        'log_loss': fh_metrics['log_loss'],
        'auc': fh_metrics['auc'],
    })

    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / 'home_lambda__hybrid_goal_mass.pkl').open('wb') as fh:
        pickle.dump({'model': home_model, 'features': feat_cols, 'cat_features': cat_cols, 'training_csv': str(input_csv)}, fh)
    with (model_dir / 'away_lambda__hybrid_goal_mass.pkl').open('wb') as fh:
        pickle.dump({'model': away_model, 'features': feat_cols, 'cat_features': cat_cols, 'training_csv': str(input_csv)}, fh)

    report = pd.DataFrame(rows)
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_csv, index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description='Train first hybrid goal-mass / lambda regressors and evaluate derived side markets.')
    parser.add_argument('--input-csv', default=str(DEFAULT_INPUT_CSV))
    parser.add_argument('--model-dir', default=str(DEFAULT_MODEL_DIR))
    parser.add_argument('--report-csv', default=str(DEFAULT_REPORT_CSV))
    parser.add_argument('--holdout-frac', type=float, default=0.2)
    parser.add_argument('--random-seed', type=int, default=42)
    args = parser.parse_args()
    report = train_goal_mass(Path(args.input_csv), Path(args.model_dir), Path(args.report_csv), args.holdout_frac, args.random_seed)
    print(f'WROTE: {args.report_csv} rows={len(report)}')
    print(report.to_string(index=False))


if __name__ == '__main__':
    main()
