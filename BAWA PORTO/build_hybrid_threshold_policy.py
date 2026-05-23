#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_AUDIT_CSV = PROJECT_ROOT / 'reports' / 'api_football' / 'hybrid_threshold_audit__England_Premier_League.csv'
DEFAULT_WINNERS_JSON = PROJECT_ROOT / 'reports' / 'api_football' / 'hybrid_research_winners__England_Premier_League.json'
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / 'reports' / 'api_football' / 'hybrid_threshold_policy__England_Premier_League.json'
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / 'reports' / 'api_football' / 'hybrid_threshold_policy__England_Premier_League.csv'

STYLE_BUCKETS = {
    'ou25': ('total-goals', 'hybrid-core'),
    'home_goals_over15': ('team-goals', 'direct-side'),
    'away_goals_over15': ('team-goals', 'direct-side'),
    'home_fts': ('fail-to-score', 'lambda-side'),
    'away_fts': ('fail-to-score', 'lambda-side'),
}


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


def build_policy(audit_csv: Path, winners_json: Path, output_json: Path, output_csv: Path, min_rows: int, min_hit_rate: float) -> Dict[str, Any]:
    audit = pd.read_csv(audit_csv)
    winners = json.loads(winners_json.read_text())
    league = winners.get('league', 'Unknown')

    policy_rows = []
    policy = {
        'league': league,
        'generated_from': {
            'audit_csv': str(audit_csv),
            'winners_json': str(winners_json),
        },
        'markets': {},
    }

    for market, winner in winners.get('winners', {}).items():
        market_rows = audit[audit['market'] == market].copy()
        source = winner.get('source')
        market_rows = market_rows[market_rows['source'] == source].copy()
        if market_rows.empty:
            continue

        eligible = market_rows[(market_rows['rows_kept'] >= min_rows) & (market_rows['hit_rate'].fillna(0) >= min_hit_rate)].copy()
        if eligible.empty:
            eligible = market_rows[market_rows['rows_kept'] > 0].copy()

        family_bucket, source_bucket = STYLE_BUCKETS.get(market, ('research', 'research'))
        if eligible.empty:
            chosen = {
                'threshold': None,
                'rows_kept': 0,
                'coverage': 0.0,
                'hit_rate': None,
                'avg_model_p': None,
            }
            deployable_flag = False
        else:
            eligible = eligible.sort_values(['hit_rate', 'coverage', 'threshold'], ascending=[False, False, False])
            chosen = eligible.iloc[0]
            deployable_flag = bool(int(chosen['rows_kept']) >= min_rows and (_safe_float(chosen['hit_rate']) or 0.0) >= min_hit_rate)

        policy['markets'][market] = {
            'source': source,
            'artifact_path': winner.get('artifact_path'),
            'threshold': None if chosen['threshold'] is None else float(chosen['threshold']),
            'rows_kept': int(chosen['rows_kept']),
            'coverage': float(chosen['coverage']),
            'hit_rate': _safe_float(chosen['hit_rate']),
            'avg_model_p': _safe_float(chosen['avg_model_p']),
            'min_rows_required': int(min_rows),
            'min_hit_rate_required': float(min_hit_rate),
            'style_bucket': family_bucket,
            'source_bucket': source_bucket,
            'deployable_flag': deployable_flag,
            'research_only_flag': not deployable_flag,
        }
        policy_rows.append({
            'league': league,
            'market': market,
            **policy['markets'][market],
        })

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(policy, indent=2))
    pd.DataFrame(policy_rows).to_csv(output_csv, index=False)
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description='Build league threshold policy from hybrid threshold audit + research winners.')
    parser.add_argument('--audit-csv', default=str(DEFAULT_AUDIT_CSV))
    parser.add_argument('--winners-json', default=str(DEFAULT_WINNERS_JSON))
    parser.add_argument('--output-json', default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument('--output-csv', default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument('--min-rows', type=int, default=8)
    parser.add_argument('--min-hit-rate', type=float, default=0.60)
    args = parser.parse_args()
    policy = build_policy(Path(args.audit_csv), Path(args.winners_json), Path(args.output_json), Path(args.output_csv), args.min_rows, args.min_hit_rate)
    print(f'WROTE: {args.output_json}')
    print(json.dumps(policy, indent=2))


if __name__ == '__main__':
    main()
