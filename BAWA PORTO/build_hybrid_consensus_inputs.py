#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from scripts.api_football.hybrid_training_utils import load_training_frame

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_HYBRID_CSV = PROJECT_ROOT / 'data_sources' / 'hybrid' / 'hybrid_match_training__England_Premier_League.csv'
DEFAULT_GOAL_MASS_CSV = PROJECT_ROOT / 'data_sources' / 'hybrid' / 'hybrid_goal_mass_inputs__England_Premier_League.csv'
DEFAULT_POLICY_JSON = PROJECT_ROOT / 'reports' / 'api_football' / 'hybrid_threshold_policy__England_Premier_League.json'
DEFAULT_MODEL_DIR = PROJECT_ROOT / 'ModelStore' / 'Hybrid' / 'England_Premier_League'
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / 'data_sources' / 'hybrid' / 'hybrid_consensus_inputs__England_Premier_League.csv'


def _load_bundle(path: Path):
    with path.open('rb') as fh:
        return pickle.load(fh)


def _prep_frame(df: pd.DataFrame, features: Iterable[str], cat_features: Iterable[str]) -> pd.DataFrame:
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


def build_consensus_inputs(hybrid_csv: Path, goal_mass_csv: Path, policy_json: Path, model_dir: Path, output_csv: Path) -> pd.DataFrame:
    hybrid = load_training_frame(hybrid_csv)
    goal_mass = load_training_frame(goal_mass_csv)
    policy = json.loads(policy_json.read_text())

    goal_key_cols = ['fixture_id']
    goal_extra_cols = [c for c in ['home_lambda_seed', 'away_lambda_seed', 'lambda_total_seed', 'home_attack_pressure_index', 'away_attack_pressure_index', 'match_pressure_delta'] if c in goal_mass.columns]
    merged = hybrid.merge(goal_mass[goal_key_cols + goal_extra_cols], on='fixture_id', how='left')

    # OU25 tuned hybrid core
    ou25_policy = policy['markets'].get('ou25')
    if ou25_policy:
        bundle = _load_bundle(Path(ou25_policy['artifact_path']))
        X = _prep_frame(merged, bundle['features'], bundle.get('cat_features', []))
        merged['consensus_ou25_over_p'] = bundle['model'].predict_proba(X)[:, 1]
        thr = ou25_policy.get('threshold')
        merged['consensus_ou25_source'] = ou25_policy['source']
        merged['consensus_ou25_style_bucket'] = ou25_policy['style_bucket']
        if thr is None:
            merged['consensus_ou25_threshold'] = np.nan
            merged['consensus_ou25_keep_flag'] = 0
        else:
            thr = float(thr)
            merged['consensus_ou25_threshold'] = thr
            merged['consensus_ou25_keep_flag'] = (merged['consensus_ou25_over_p'] >= thr).astype(int)

    # Lambda bundles shared by FTS/lambda-derived lanes
    home_lambda_bundle = None
    away_lambda_bundle = None
    home_lambda = None
    away_lambda = None
    try:
        home_lambda_bundle = _load_bundle(model_dir / 'home_lambda__hybrid_goal_mass.pkl')
        Xh = _prep_frame(goal_mass, home_lambda_bundle['features'], home_lambda_bundle.get('cat_features', []))
        home_lambda = np.clip(np.asarray(home_lambda_bundle['model'].predict(Xh), dtype=float), 0.0, 6.0)
    except Exception:
        pass
    try:
        away_lambda_bundle = _load_bundle(model_dir / 'away_lambda__hybrid_goal_mass.pkl')
        Xa = _prep_frame(goal_mass, away_lambda_bundle['features'], away_lambda_bundle.get('cat_features', []))
        away_lambda = np.clip(np.asarray(away_lambda_bundle['model'].predict(Xa), dtype=float), 0.0, 6.0)
    except Exception:
        pass
    if home_lambda is not None:
        merged['home_lambda_pred'] = home_lambda
        goal_mass['home_lambda_pred'] = home_lambda
    if away_lambda is not None:
        merged['away_lambda_pred'] = away_lambda
        goal_mass['away_lambda_pred'] = away_lambda
    if home_lambda is not None and away_lambda is not None:
        total_lambda = home_lambda + away_lambda
        merged['lambda_total_pred'] = total_lambda
        goal_mass['lambda_total_pred'] = total_lambda
        merged['home_lambda_share'] = np.where(total_lambda > 0, home_lambda / total_lambda, 0.5)
        merged['away_lambda_share'] = np.where(total_lambda > 0, away_lambda / total_lambda, 0.5)
        goal_mass['home_lambda_share'] = np.where(total_lambda > 0, home_lambda / total_lambda, 0.5)
        goal_mass['away_lambda_share'] = np.where(total_lambda > 0, away_lambda / total_lambda, 0.5)

    # Side markets
    for market in ['home_goals_over15', 'away_goals_over15', 'home_fts', 'away_fts']:
        pol = policy['markets'].get(market)
        if not pol:
            continue
        if pol.get('threshold') is None:
            continue
        thr = float(pol['threshold'])
        src = pol['source']
        if src == 'direct_classifier':
            bundle = _load_bundle(Path(pol['artifact_path']))
            X = _prep_frame(goal_mass, bundle['features'], bundle.get('cat_features', []))
            proba = bundle['model'].predict_proba(X)[:, 1]
        else:
            if market == 'home_fts' and home_lambda is not None:
                proba = _pois_prob_eq0(home_lambda)
            elif market == 'away_fts' and away_lambda is not None:
                proba = _pois_prob_eq0(away_lambda)
            else:
                continue
        merged[f'consensus_{market}_p'] = proba
        merged[f'consensus_{market}_threshold'] = thr
        merged[f'consensus_{market}_keep_flag'] = (merged[f'consensus_{market}_p'] >= thr).astype(int)
        merged[f'consensus_{market}_source'] = src
        merged[f'consensus_{market}_style_bucket'] = pol['style_bucket']

    # Value-aware helpers
    if 'odds_ft_over25' in merged.columns and 'consensus_ou25_over_p' in merged.columns:
        merged['consensus_ou25_bookie_implied'] = np.where(pd.to_numeric(merged['odds_ft_over25'], errors='coerce') > 0, 1.0 / pd.to_numeric(merged['odds_ft_over25'], errors='coerce'), np.nan)
        merged['consensus_ou25_value_edge'] = merged['consensus_ou25_over_p'] - merged['consensus_ou25_bookie_implied']
    if 'odds_ft_home_team_win' in merged.columns and 'consensus_home_goals_over15_p' in merged.columns:
        merged['consensus_home_goals_context_anchor'] = np.where(pd.to_numeric(merged['odds_ft_home_team_win'], errors='coerce') > 0, 1.0 / pd.to_numeric(merged['odds_ft_home_team_win'], errors='coerce'), np.nan)
    if 'odds_ft_away_team_win' in merged.columns and 'consensus_away_goals_over15_p' in merged.columns:
        merged['consensus_away_goals_context_anchor'] = np.where(pd.to_numeric(merged['odds_ft_away_team_win'], errors='coerce') > 0, 1.0 / pd.to_numeric(merged['odds_ft_away_team_win'], errors='coerce'), np.nan)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description='Build first hybrid consensus input table from threshold policy + research winners.')
    parser.add_argument('--hybrid-csv', default=str(DEFAULT_HYBRID_CSV))
    parser.add_argument('--goal-mass-csv', default=str(DEFAULT_GOAL_MASS_CSV))
    parser.add_argument('--policy-json', default=str(DEFAULT_POLICY_JSON))
    parser.add_argument('--model-dir', default=str(DEFAULT_MODEL_DIR))
    parser.add_argument('--output-csv', default=str(DEFAULT_OUTPUT_CSV))
    args = parser.parse_args()
    df = build_consensus_inputs(Path(args.hybrid_csv), Path(args.goal_mass_csv), Path(args.policy_json), Path(args.model_dir), Path(args.output_csv))
    print(f'WROTE: {args.output_csv} rows={len(df)} cols={len(df.columns)}')


if __name__ == '__main__':
    main()
