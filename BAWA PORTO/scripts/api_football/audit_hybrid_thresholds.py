from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from .hybrid_training_utils import load_training_frame

PURPOSE = 'Audit threshold ladders for tuned OU25 and selected hybrid side-market winners.'

THRESHOLDS = [round(x, 2) for x in np.arange(0.50, 0.92, 0.02)]
MIN_ROWS = 5


def _safe_float(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        return None


def _load_bundle(path: Path):
    with path.open('rb') as fh:
        return pickle.load(fh)


def _prep_features(df: pd.DataFrame, features: Iterable[str], cat_features: Iterable[str]) -> pd.DataFrame:
    X = df[list(features)].copy()
    cats = set(cat_features)
    for c in X.columns:
        if c in cats or X[c].dtype == 'object':
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


def _threshold_rows(market: str, league: str, source: str, y_true: np.ndarray, proba: np.ndarray) -> List[Dict[str, object]]:
    rows = []
    p = np.asarray(proba, dtype=float)
    for thr in THRESHOLDS:
        keep = p >= thr
        kept = int(keep.sum())
        hit_rate = float(np.mean(y_true[keep])) if kept else None
        avg_p = float(np.mean(p[keep])) if kept else None
        coverage = kept / len(p) if len(p) else 0.0
        rows.append({
            'league': league,
            'market': market,
            'source': source,
            'threshold': thr,
            'rows_kept': kept,
            'coverage': coverage,
            'hit_rate': hit_rate,
            'avg_model_p': avg_p,
        })
    return rows


def _winner_row(rows: pd.DataFrame) -> pd.Series:
    # Favor meaningful volume, then hit rate, then threshold severity.
    scored = rows.copy()
    eligible = scored[scored['rows_kept'] >= MIN_ROWS].copy()
    if eligible.empty:
        eligible = scored[scored['rows_kept'] > 0].copy()
    if eligible.empty:
        return scored.iloc[0]
    eligible['hit_rate_score'] = eligible['hit_rate'].fillna(0.0)
    eligible['volume_score'] = eligible['coverage'].fillna(0.0)
    eligible['composite'] = eligible['hit_rate_score'] * 0.75 + eligible['volume_score'] * 0.25
    eligible = eligible.sort_values(['composite', 'rows_kept', 'threshold'], ascending=[False, False, False])
    return eligible.iloc[0]


def build_threshold_audit(hybrid_csv: Path, goal_mass_csv: Path, model_dir: Path, output_csv: Path, winners_json: Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    hybrid = load_training_frame(hybrid_csv)
    goal_mass = load_training_frame(goal_mass_csv)
    split_idx = max(1, int(len(hybrid) * 0.8))
    split_idx = min(split_idx, len(hybrid) - 1)
    hybrid_test = hybrid.iloc[split_idx:].copy().reset_index(drop=True)
    goal_test = goal_mass.iloc[split_idx:].copy().reset_index(drop=True)
    league = str(hybrid_test['league'].iloc[0]) if 'league' in hybrid_test.columns and not hybrid_test.empty else 'Unknown'

    rows: List[Dict[str, object]] = []
    winners: Dict[str, object] = {'league': league, 'winners': {}}

    # OU25 tuned winner
    ou_bundle = _load_bundle(model_dir / 'ou25__baseline_team_lineup_injury__catboost.pkl')
    X_ou = _prep_features(hybrid_test, ou_bundle['features'], ou_bundle.get('cat_features', []))
    p_ou = ou_bundle['model'].predict_proba(X_ou)[:, 1]
    y_ou = pd.to_numeric(hybrid_test['target_ou25_over'], errors='coerce').fillna(0).astype(int).to_numpy()
    rows.extend(_threshold_rows('ou25', league, 'catboost_baseline_team_lineup_injury', y_ou, p_ou))
    base_metrics = _binary_metrics(y_ou, p_ou)
    winners['winners']['ou25'] = {
        'artifact_path': str(model_dir / 'ou25__baseline_team_lineup_injury__catboost.pkl'),
        'source': 'catboost_baseline_team_lineup_injury',
        'base_metrics': base_metrics,
    }

    # Lambda bundles for side markets
    home_lambda_bundle = _load_bundle(model_dir / 'home_lambda__hybrid_goal_mass.pkl')
    away_lambda_bundle = _load_bundle(model_dir / 'away_lambda__hybrid_goal_mass.pkl')
    Xh = _prep_features(goal_test, home_lambda_bundle['features'], home_lambda_bundle.get('cat_features', []))
    Xa = _prep_features(goal_test, away_lambda_bundle['features'], away_lambda_bundle.get('cat_features', []))
    home_lambda = np.clip(np.asarray(home_lambda_bundle['model'].predict(Xh), dtype=float), 0.0, 6.0)
    away_lambda = np.clip(np.asarray(away_lambda_bundle['model'].predict(Xa), dtype=float), 0.0, 6.0)
    goal_test['home_lambda_pred'] = home_lambda
    goal_test['away_lambda_pred'] = away_lambda
    total = home_lambda + away_lambda
    goal_test['lambda_total_pred'] = total
    goal_test['home_lambda_share'] = np.where(total > 0, home_lambda / total, 0.5)
    goal_test['away_lambda_share'] = np.where(total > 0, away_lambda / total, 0.5)

    side_specs = [
        ('home_goals_over15', 'direct_classifier', model_dir / 'home_goals_over15__hybrid_side.pkl', pd.to_numeric(goal_test['target_home_goals_over15'], errors='coerce').fillna(0).astype(int).to_numpy(), None),
        ('away_goals_over15', 'direct_classifier', model_dir / 'away_goals_over15__hybrid_side.pkl', pd.to_numeric(goal_test['target_away_goals_over15'], errors='coerce').fillna(0).astype(int).to_numpy(), None),
        ('home_fts', 'lambda_derived', model_dir / 'home_lambda__hybrid_goal_mass.pkl', pd.to_numeric(goal_test['target_home_fts'], errors='coerce').fillna(0).astype(int).to_numpy(), _pois_prob_eq0(home_lambda)),
        ('away_fts', 'lambda_derived', model_dir / 'away_lambda__hybrid_goal_mass.pkl', pd.to_numeric(goal_test['target_away_fts'], errors='coerce').fillna(0).astype(int).to_numpy(), _pois_prob_eq0(away_lambda)),
    ]

    for market, source, artifact_path, y_true, lambda_proba in side_specs:
        if source == 'direct_classifier':
            bundle = _load_bundle(artifact_path)
            X = _prep_features(goal_test, bundle['features'], bundle.get('cat_features', []))
            proba = bundle['model'].predict_proba(X)[:, 1]
            winners['winners'][market] = {
                'artifact_path': str(artifact_path),
                'source': source,
                'base_metrics': _binary_metrics(y_true, proba),
            }
        else:
            proba = lambda_proba
            winners['winners'][market] = {
                'artifact_path': str(artifact_path),
                'source': source,
                'base_metrics': _binary_metrics(y_true, proba),
            }
        rows.extend(_threshold_rows(market, league, source, y_true, proba))

    out = pd.DataFrame(rows)
    winners_summary = {}
    for market in out['market'].unique():
        best = _winner_row(out[out['market'] == market])
        winners_summary[market] = {
            **winners['winners'][market],
            'threshold_winner': {
                'threshold': _safe_float(best['threshold']),
                'rows_kept': int(_safe_float(best['rows_kept']) or 0),
                'coverage': _safe_float(best['coverage']) or 0.0,
                'hit_rate': _safe_float(best['hit_rate']),
                'avg_model_p': _safe_float(best['avg_model_p']),
            },
        }
    winners['winners'] = winners_summary

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    winners_json.parent.mkdir(parents=True, exist_ok=True)
    winners_json.write_text(json.dumps(winners, indent=2))
    return out, winners


def main() -> None:
    parser = argparse.ArgumentParser(description=PURPOSE)
    parser.add_argument('--hybrid-csv', default='data_sources/hybrid/hybrid_match_training__England_Premier_League.csv')
    parser.add_argument('--goal-mass-csv', default='data_sources/hybrid/hybrid_goal_mass_inputs__England_Premier_League.csv')
    parser.add_argument('--model-dir', default='ModelStore/Hybrid/England_Premier_League')
    parser.add_argument('--output-csv', default='reports/api_football/hybrid_threshold_audit__England_Premier_League.csv')
    parser.add_argument('--winners-json', default='reports/api_football/hybrid_research_winners__England_Premier_League.json')
    args = parser.parse_args()
    df, winners = build_threshold_audit(Path(args.hybrid_csv), Path(args.goal_mass_csv), Path(args.model_dir), Path(args.output_csv), Path(args.winners_json))
    print(f'WROTE: {args.output_csv} rows={len(df)}')
    print(json.dumps(winners, indent=2))


if __name__ == '__main__':
    main()
