from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def clamp_score(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 1)


def build(overlap_csv: str, runner_csv: str, priority_csv: str, regime_csv: str, output_csv: str, output_md: str) -> pd.DataFrame:
    overlap = pd.read_csv(overlap_csv, low_memory=False)
    runner = pd.read_csv(runner_csv, low_memory=False)
    priority = pd.read_csv(priority_csv, low_memory=False)
    regime = pd.read_csv(regime_csv, low_memory=False)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    rows = []

    overlap_total = len(overlap)
    covered = int(overlap.get('has_ranked_history', pd.Series(dtype=float)).fillna(0).astype(int).sum()) if overlap_total else 0
    join_ready = int(overlap.get('joined_flag', pd.Series(dtype=float)).fillna(0).astype(int).sum()) if overlap_total else 0
    coverage_ratio = (covered / overlap_total) if overlap_total else 0.0
    coverage_score = clamp_score(coverage_ratio * 100.0)
    rows.append({
        'dimension': 'coverage_completeness',
        'score': coverage_score,
        'raw_ratio': round(coverage_ratio, 3),
        'supporting_count': covered,
        'total_count': overlap_total,
        'status': 'READY' if coverage_score >= 70 else ('PARTIAL' if coverage_score >= 40 else 'EARLY'),
        'note': f'{covered}/{overlap_total} structural-risk fixtures currently find ranked-board history; {join_ready} already survive into the joined audit.',
    })

    settled_total = len(runner)
    settled_rows = int(runner.get('settled_actual_available', pd.Series(dtype=float)).fillna(0).astype(int).sum()) if settled_total else 0
    settled_ratio = (settled_rows / settled_total) if settled_total else 0.0
    settled_score = clamp_score(settled_ratio * 100.0)
    rows.append({
        'dimension': 'settled_result_depth',
        'score': settled_score,
        'raw_ratio': round(settled_ratio, 3),
        'supporting_count': settled_rows,
        'total_count': settled_total,
        'status': 'READY' if settled_score >= 70 else ('PARTIAL' if settled_score >= 40 else 'EARLY'),
        'note': f'{settled_rows}/{settled_total} runner rows carry settled player-stat evidence.',
    })

    priority_total = len(priority)
    high = int((priority.get('priority_bucket', pd.Series(dtype=str)).astype(str) == 'HIGH_PRIORITY').sum()) if priority_total else 0
    trial = int((priority.get('priority_bucket', pd.Series(dtype=str)).astype(str) == 'TRIAL_PRIORITY').sum()) if priority_total else 0
    patch_score = clamp_score((((high * 1.0) + (trial * 0.6)) / max(priority_total, 1)) * 100.0)
    rows.append({
        'dimension': 'patch_trial_maturity',
        'score': patch_score,
        'raw_ratio': round((((high * 1.0) + (trial * 0.6)) / max(priority_total, 1)), 3),
        'supporting_count': high + trial,
        'total_count': priority_total,
        'status': 'READY' if patch_score >= 70 else ('PARTIAL' if patch_score >= 40 else 'EARLY'),
        'note': f'{high} HIGH_PRIORITY and {trial} TRIAL_PRIORITY cohorts are ready for deeper shadow stress-testing.',
    })

    regime_total = len(regime)
    calibrated = int((regime.get('regime_posture', pd.Series(dtype=str)).astype(str) != 'MIXED_WATCH').sum()) if regime_total else 0
    rows_with_density = int((pd.to_numeric(regime.get('rows', pd.Series(dtype=float)), errors='coerce').fillna(0) >= 5).sum()) if regime_total else 0
    league_ratio = ((calibrated + rows_with_density) / max(regime_total * 2, 1)) if regime_total else 0.0
    league_score = clamp_score(league_ratio * 100.0)
    rows.append({
        'dimension': 'league_calibration_maturity',
        'score': league_score,
        'raw_ratio': round(league_ratio, 3),
        'supporting_count': calibrated + rows_with_density,
        'total_count': regime_total * 2,
        'status': 'READY' if league_score >= 70 else ('PARTIAL' if league_score >= 40 else 'EARLY'),
        'note': f'{calibrated}/{regime_total} target leagues have a non-watch regime posture; {rows_with_density}/{regime_total} have at least five mapped rows.',
    })

    out = pd.DataFrame(rows)
    out.to_csv(output_csv, index=False)

    overall_score = round(out['score'].mean(), 1) if not out.empty else 0.0
    lines = [
        '# Backtesting Readiness Board',
        '',
        f'- Overall readiness score: `{overall_score}` / `100`.',
        '- This is a compact audit of whether the backtesting stack is ready for deeper threshold and overlap work.',
        '',
    ]
    for _, row in out.sort_values('score', ascending=False).iterrows():
        lines.append(f"## {row['dimension']}")
        lines.append(f"- score={row['score']:.1f} | status={row['status']} | ratio={row['raw_ratio']}")
        lines.append(f"- {row['note']}")
        lines.append('')
    Path(output_md).write_text('\n'.join(lines) + '\n')
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a tiny backtesting readiness board from overlap, runner, patch, and league calibration audits.')
    parser.add_argument('--overlap-csv', required=True)
    parser.add_argument('--runner-csv', required=True)
    parser.add_argument('--priority-csv', required=True)
    parser.add_argument('--regime-csv', required=True)
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--output-md', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    out = build(args.overlap_csv, args.runner_csv, args.priority_csv, args.regime_csv, args.output_csv, args.output_md)
    print(f'WROTE: {args.output_csv}')
    print(f'rows: {len(out)}')
