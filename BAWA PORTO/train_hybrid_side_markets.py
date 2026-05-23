#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = PROJECT_ROOT / 'data_sources' / 'hybrid' / 'hybrid_goal_mass_inputs__England_Premier_League.csv'
DEFAULT_MODEL_DIR = PROJECT_ROOT / 'ModelStore' / 'Hybrid' / 'England_Premier_League'
DEFAULT_REPORT_CSV = PROJECT_ROOT / 'reports' / 'api_football' / 'hybrid_side_market_metrics__England_Premier_League.csv'
DEFAULT_HOME_LAMBDA = DEFAULT_MODEL_DIR / 'home_lambda__hybrid_goal_mass.pkl'
DEFAULT_AWAY_LAMBDA = DEFAULT_MODEL_DIR / 'away_lambda__hybrid_goal_mass.pkl'

TARGET_MAP = {
    'home_goals_over15': 'target_home_goals_over15',
    'away_goals_over15': 'target_away_goals_over15',
    'home_fts': 'target_home_fts',
    'away_fts': 'target_away_fts',
    'btts_first_half': 'target_btts_first_half',
}
ID_COLS = {
    'fixture_id', 'fixture_key', 'league', 'season', 'match_date', 'home_team_name', 'away_team_name'
}
DROP_TARGETS = set(TARGET_MAP.values()) | {
    'home_team_goal_count', 'away_team_goal_count',
    'target_home_goals', 'target_away_goals', 'target_total_goals',
    'target_btts_yes', 'target_ou25_over'
}


def _load_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'match_date' in df.columns:
        df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
    return df.sort_values(['match_date', 'fixture_id']).reset_index(drop=True)


def _load_lambda_bundle(path: Path):
    with path.open('rb') as fh:
        return pickle.load(fh)


def _score_lambda(df: pd.DataFrame, bundle_path: Path, out_col: str) -> pd.DataFrame:
    bundle = _load_lambda_bundle(bundle_path)
    model = bundle['model']
    feat_cols = bundle['features']
    cat_cols = bundle.get('cat_features', [])
    X = df[feat_cols].copy()
    for c in X.columns:
        if X[c].dtype == 'object':
            X[c] = X[c].fillna('MISSING')
        else:
            X[c] = pd.to_numeric(X[c], errors='coerce')
    pred = np.clip(np.asarray(model.predict(X), dtype=float), 0.0, 6.0)
    df[out_col] = pred
    return df


def _feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in ID_COLS and c not in DROP_TARGETS]


def _prep_xy(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[str]]:
    cols = _feature_cols(df)
    X = df[cols].copy()
    cat_cols = [c for c in X.columns if X[c].dtype == 'object']
    for c in X.columns:
        if X[c].dtype == 'object':
            X[c] = X[c].fillna('MISSING')
        else:
            X[c] = pd.to_numeric(X[c], errors='coerce')
    y = pd.to_numeric(df[target_col], errors='coerce').fillna(0).astype(int).to_numpy()
    return X, y, cols, cat_cols


def _split(X: pd.DataFrame, y: np.ndarray, holdout_frac: float):
    split_idx = max(1, int(len(X) * (1.0 - holdout_frac)))
    split_idx = min(split_idx, len(X) - 1)
    return X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy(), y[:split_idx], y[split_idx:]


def _fit_market(df: pd.DataFrame, market: str, holdout_frac: float, random_seed: int) -> Dict[str, object]:
    target_col = TARGET_MAP[market]
    X, y, feat_cols, cat_cols = _prep_xy(df, target_col)
    X_train, X_test, y_train, y_test = _split(X, y, holdout_frac)
    cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]
    model = CatBoostClassifier(
        random_seed=random_seed,
        verbose=False,
        depth=6,
        learning_rate=0.05,
        iterations=300,
        loss_function='Logloss',
        eval_metric='Logloss',
    )
    model.fit(X_train, y_train, cat_features=cat_idx, eval_set=(X_test, y_test), use_best_model=True)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics: Dict[str, object] = {
        'market': market,
        'rows_train': len(X_train),
        'rows_test': len(X_test),
        'n_features': len(feat_cols),
        'accuracy': float(accuracy_score(y_test, pred)),
        'log_loss': None,
        'auc': None,
        'model': model,
        'features': feat_cols,
        'cat_features': cat_cols,
    }
    if len(np.unique(y_test)) >= 2:
        try:
            metrics['log_loss'] = float(log_loss(y_test, np.clip(proba, 1e-6, 1 - 1e-6), labels=[0, 1]))
        except Exception:
            metrics['log_loss'] = None
        try:
            metrics['auc'] = float(roc_auc_score(y_test, proba))
        except Exception:
            metrics['auc'] = None
    return metrics


def train_side_markets(input_csv: Path, model_dir: Path, report_csv: Path, home_lambda_path: Path, away_lambda_path: Path, holdout_frac: float, random_seed: int) -> pd.DataFrame:
    df = _load_frame(input_csv)
    if home_lambda_path.exists():
        df = _score_lambda(df, home_lambda_path, 'home_lambda_pred')
    if away_lambda_path.exists():
        df = _score_lambda(df, away_lambda_path, 'away_lambda_pred')
    if 'home_lambda_pred' in df.columns and 'away_lambda_pred' in df.columns:
        total = df['home_lambda_pred'] + df['away_lambda_pred']
        df['lambda_total_pred'] = total
        df['home_lambda_share'] = np.where(total > 0, df['home_lambda_pred'] / total, 0.5)
        df['away_lambda_share'] = np.where(total > 0, df['away_lambda_pred'] / total, 0.5)

    rows: List[Dict[str, object]] = []
    model_dir.mkdir(parents=True, exist_ok=True)
    for idx, market in enumerate(TARGET_MAP):
        result = _fit_market(df, market, holdout_frac, random_seed + idx)
        artifact = model_dir / f'{market}__hybrid_side.pkl'
        bundle = {
            'market': market,
            'training_csv': str(input_csv),
            'model': result.pop('model'),
            'features': result.pop('features'),
            'cat_features': result.pop('cat_features'),
            'metrics': {k: result[k] for k in ['accuracy', 'log_loss', 'auc', 'rows_train', 'rows_test', 'n_features']},
        }
        with artifact.open('wb') as fh:
            pickle.dump(bundle, fh)
        row = {k: result.get(k) for k in ['market', 'rows_train', 'rows_test', 'n_features', 'accuracy', 'log_loss', 'auc']}
        row['artifact_path'] = str(artifact)
        rows.append(row)
        print(json.dumps(row, default=str))

    report = pd.DataFrame(rows)
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_csv, index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description='Train dedicated hybrid side-market classifiers from hybrid goal-mass inputs.')
    parser.add_argument('--input-csv', default=str(DEFAULT_INPUT_CSV))
    parser.add_argument('--model-dir', default=str(DEFAULT_MODEL_DIR))
    parser.add_argument('--report-csv', default=str(DEFAULT_REPORT_CSV))
    parser.add_argument('--home-lambda-bundle', default=str(DEFAULT_HOME_LAMBDA))
    parser.add_argument('--away-lambda-bundle', default=str(DEFAULT_AWAY_LAMBDA))
    parser.add_argument('--holdout-frac', type=float, default=0.2)
    parser.add_argument('--random-seed', type=int, default=42)
    args = parser.parse_args()
    report = train_side_markets(
        Path(args.input_csv), Path(args.model_dir), Path(args.report_csv),
        Path(args.home_lambda_bundle), Path(args.away_lambda_bundle),
        args.holdout_frac, args.random_seed,
    )
    print(f'WROTE: {args.report_csv} rows={len(report)}')


if __name__ == '__main__':
    main()
