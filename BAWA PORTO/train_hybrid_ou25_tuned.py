#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from scripts.api_football.feature_family_stacks import stacked_cols
from scripts.api_football.hybrid_training_utils import load_training_frame

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / 'data_sources' / 'hybrid' / 'hybrid_match_training__England_Premier_League.csv'
DEFAULT_MODEL_DIR = PROJECT_ROOT / 'ModelStore' / 'Hybrid' / 'England_Premier_League'
DEFAULT_REPORT = PROJECT_ROOT / 'reports' / 'api_football' / 'hybrid_ou25_tuned_metrics__England_Premier_League.csv'

STACK_FAMILIES = ['team', 'lineup', 'injury']
DEPTH_GRID = [4, 6, 8]
L2_GRID = [3.0, 5.0, 8.0]
ITER_GRID = [200, 350]


def _prep_xy(df: pd.DataFrame, cols: List[str]):
    X = df[cols].copy()
    cat_cols = [c for c in X.columns if X[c].dtype == 'object']
    for c in X.columns:
        if X[c].dtype == 'object':
            X[c] = X[c].fillna('MISSING')
        else:
            X[c] = pd.to_numeric(X[c], errors='coerce')
    y = pd.to_numeric(df['target_ou25_over'], errors='coerce').fillna(0).astype(int).to_numpy()
    return X, y, cat_cols


def _split(X: pd.DataFrame, y: np.ndarray, holdout_frac: float):
    split_idx = max(1, int(len(X) * (1.0 - holdout_frac)))
    split_idx = min(split_idx, len(X) - 1)
    return X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy(), y[:split_idx], y[split_idx:]


def tune_ou25(input_csv: Path, model_dir: Path, report_csv: Path, holdout_frac: float, random_seed: int) -> pd.DataFrame:
    df = load_training_frame(input_csv)
    cols = stacked_cols(df, STACK_FAMILIES)
    X, y, cat_cols = _prep_xy(df, cols)
    X_train, X_test, y_train, y_test = _split(X, y, holdout_frac)
    cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]

    rows: List[Dict[str, object]] = []
    best = None
    best_key = None
    for depth in DEPTH_GRID:
        for l2 in L2_GRID:
            for iters in ITER_GRID:
                model = CatBoostClassifier(
                    random_seed=random_seed,
                    verbose=False,
                    depth=depth,
                    l2_leaf_reg=l2,
                    learning_rate=0.05,
                    iterations=iters,
                    loss_function='Logloss',
                    eval_metric='Logloss',
                )
                model.fit(X_train, y_train, cat_features=cat_idx, eval_set=(X_test, y_test), use_best_model=True)
                proba = model.predict_proba(X_test)[:, 1]
                pred = (proba >= 0.5).astype(int)
                row = {
                    'depth': depth,
                    'l2_leaf_reg': l2,
                    'iterations': iters,
                    'rows_train': len(X_train),
                    'rows_test': len(X_test),
                    'n_features': len(cols),
                    'accuracy': float(accuracy_score(y_test, pred)),
                    'log_loss': None,
                    'auc': None,
                }
                if len(np.unique(y_test)) >= 2:
                    try:
                        row['log_loss'] = float(log_loss(y_test, np.clip(proba, 1e-6, 1 - 1e-6), labels=[0, 1]))
                    except Exception:
                        row['log_loss'] = None
                    try:
                        row['auc'] = float(roc_auc_score(y_test, proba))
                    except Exception:
                        row['auc'] = None
                rows.append(row)
                key = ((row['auc'] if row['auc'] is not None else -1.0), -((row['log_loss'] if row['log_loss'] is not None else 999.0)), row['accuracy'])
                if best is None or key > best_key:
                    best = (model, row)
                    best_key = key

    report = pd.DataFrame(rows).sort_values(['auc', 'log_loss', 'accuracy'], ascending=[False, True, False]).reset_index(drop=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model, best_row = best
    with (model_dir / 'ou25__baseline_team_lineup_injury__catboost.pkl').open('wb') as fh:
        pickle.dump({
            'market': 'ou25',
            'families': STACK_FAMILIES,
            'features': cols,
            'cat_features': cat_cols,
            'training_csv': str(input_csv),
            'metrics': best_row,
            'model': best_model,
        }, fh)
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_csv, index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description='Tune OU25 CatBoost on baseline + team + lineup + injury stack.')
    parser.add_argument('--input-csv', default=str(DEFAULT_INPUT))
    parser.add_argument('--model-dir', default=str(DEFAULT_MODEL_DIR))
    parser.add_argument('--report-csv', default=str(DEFAULT_REPORT))
    parser.add_argument('--holdout-frac', type=float, default=0.2)
    parser.add_argument('--random-seed', type=int, default=42)
    args = parser.parse_args()
    report = tune_ou25(Path(args.input_csv), Path(args.model_dir), Path(args.report_csv), args.holdout_frac, args.random_seed)
    print(f'WROTE: {args.report_csv} rows={len(report)}')
    print(report.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
