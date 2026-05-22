from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

PURPOSE = 'Compare lambda-derived side probabilities against direct side-classifier outputs.'


def _load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'match_date' in df.columns:
        df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
    return df.sort_values(['match_date', 'fixture_id']).reset_index(drop=True)


def _load_bundle(path: Path):
    with path.open('rb') as fh:
        return pickle.load(fh)


def _prep_for_model(df: pd.DataFrame, features: List[str], cat_features: List[str]) -> pd.DataFrame:
    X = df[features].copy()
    for c in X.columns:
        if c in cat_features or X[c].dtype == 'object':
            X[c] = X[c].fillna('MISSING')
        else:
            X[c] = pd.to_numeric(X[c], errors='coerce')
    return X


def _pois_prob_ge2(lmbda: np.ndarray) -> np.ndarray:
    l = np.clip(np.asarray(lmbda, dtype=float), 0.0, None)
    return 1.0 - np.exp(-l) * (1.0 + l)


def _pois_prob_eq0(lmbda: np.ndarray) -> np.ndarray:
    l = np.clip(np.asarray(lmbda, dtype=float), 0.0, None)
    return np.exp(-l)


def _binary_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float | None]:
    p = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    pred = (p >= 0.5).astype(int)
    out: Dict[str, float | None] = {
        'accuracy': float(accuracy_score(y_true, pred)),
        'log_loss': None,
        'auc': None,
    }
    if len(np.unique(np.asarray(y_true, dtype=int))) >= 2:
        try:
            out['log_loss'] = float(log_loss(y_true, p, labels=[0, 1]))
        except Exception:
            out['log_loss'] = None
        try:
            out['auc'] = float(roc_auc_score(y_true, p))
        except Exception:
            out['auc'] = None
    return out


def build_audit(goal_mass_csv: Path, model_dir: Path, output_csv: Path) -> pd.DataFrame:
    df = _load_df(goal_mass_csv)
    split_idx = max(1, int(len(df) * 0.8))
    split_idx = min(split_idx, len(df) - 1)
    test_df = df.iloc[split_idx:].copy()

    home_lambda_bundle = _load_bundle(model_dir / 'home_lambda__hybrid_goal_mass.pkl')
    away_lambda_bundle = _load_bundle(model_dir / 'away_lambda__hybrid_goal_mass.pkl')

    Xh = _prep_for_model(test_df, home_lambda_bundle['features'], home_lambda_bundle.get('cat_features', []))
    Xa = _prep_for_model(test_df, away_lambda_bundle['features'], away_lambda_bundle.get('cat_features', []))
    home_lambda = np.clip(np.asarray(home_lambda_bundle['model'].predict(Xh), dtype=float), 0.0, 6.0)
    away_lambda = np.clip(np.asarray(away_lambda_bundle['model'].predict(Xa), dtype=float), 0.0, 6.0)
    test_df['home_lambda_pred'] = home_lambda
    test_df['away_lambda_pred'] = away_lambda
    total_lambda = home_lambda + away_lambda
    test_df['lambda_total_pred'] = total_lambda
    test_df['home_lambda_share'] = np.where(total_lambda > 0, home_lambda / total_lambda, 0.5)
    test_df['away_lambda_share'] = np.where(total_lambda > 0, away_lambda / total_lambda, 0.5)
    btts_lambda = (1.0 - _pois_prob_eq0(home_lambda)) * (1.0 - _pois_prob_eq0(away_lambda))
    fh_anchor = np.clip(
        pd.to_numeric(test_df.get('home_first_goal_rate_l10', 0), errors='coerce').fillna(0).to_numpy() * 0.5 +
        pd.to_numeric(test_df.get('away_first_goal_rate_l10', 0), errors='coerce').fillna(0).to_numpy() * 0.5,
        0.0,
        1.0,
    )
    btts_fh_lambda = np.clip(btts_lambda * (0.35 + 0.65 * fh_anchor), 1e-6, 1 - 1e-6)

    market_defs = [
        ('home_goals_over15', 'target_home_goals_over15', _pois_prob_ge2(home_lambda)),
        ('away_goals_over15', 'target_away_goals_over15', _pois_prob_ge2(away_lambda)),
        ('home_fts', 'target_home_fts', _pois_prob_eq0(home_lambda)),
        ('away_fts', 'target_away_fts', _pois_prob_eq0(away_lambda)),
        ('btts_first_half', 'target_btts_first_half', btts_fh_lambda),
    ]

    rows: List[Dict[str, object]] = []
    for market, target_col, lambda_proba in market_defs:
        y_true = pd.to_numeric(test_df[target_col], errors='coerce').fillna(0).astype(int).to_numpy()
        lam_metrics = _binary_metrics(y_true, lambda_proba)
        rows.append({'market': market, 'source': 'lambda_derived', **lam_metrics})

        side_bundle = _load_bundle(model_dir / f'{market}__hybrid_side.pkl')
        Xs = _prep_for_model(test_df, side_bundle['features'], side_bundle.get('cat_features', []))
        side_proba = side_bundle['model'].predict_proba(Xs)[:, 1]
        side_metrics = _binary_metrics(y_true, side_proba)
        rows.append({'market': market, 'source': 'direct_classifier', **side_metrics})

    out = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=PURPOSE)
    parser.add_argument('--goal-mass-csv', default='data_sources/hybrid/hybrid_goal_mass_inputs__England_Premier_League.csv')
    parser.add_argument('--model-dir', default='ModelStore/Hybrid/England_Premier_League')
    parser.add_argument('--output-csv', default='reports/api_football/hybrid_lambda_vs_direct_side.csv')
    args = parser.parse_args()
    df = build_audit(Path(args.goal_mass_csv), Path(args.model_dir), Path(args.output_csv))
    print(f'WROTE: {args.output_csv} rows={len(df)}')
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
