from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CONFIDENCE_WEIGHT = {
    'APPLY_SOFT_PATCH': 1.0,
    'TRIAL_PATCH': 0.85,
}


def compute_sample_stability(row: pd.Series) -> float:
    runner_rows = float(pd.to_numeric(row.get('runner_rows'), errors='coerce') or 0.0)
    current_selected = float(pd.to_numeric(row.get('current_selected'), errors='coerce') or 0.0)
    newly_removed = float(pd.to_numeric(row.get('newly_removed'), errors='coerce') or 0.0)
    volume_score = min(runner_rows / 20.0, 1.0)
    removal_penalty = min(newly_removed / max(current_selected, 1.0), 1.0) * 0.25
    confidence_weight = CONFIDENCE_WEIGHT.get(str(row.get('patch_confidence')), 0.75)
    stability = max(0.0, (volume_score - removal_penalty) * confidence_weight)
    return round(stability, 3)


def build(input_csv: str, output_csv: str, output_md: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv, low_memory=False)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        empty = pd.DataFrame()
        empty.to_csv(output_csv, index=False)
        Path(output_md).write_text('# Shadow Patch Priority Board\n\nNo shadow patch cohorts matched.\n')
        return empty

    out = df.copy()
    numeric_cols = [
        'runner_rows',
        'current_selected',
        'shadow_selected',
        'newly_admitted',
        'newly_removed',
        'current_hits',
        'shadow_hits',
        'newly_admitted_hits',
        'newly_removed_hits',
        'proposed_score_cut_shift',
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0)

    out['net_hit_gain'] = (out['shadow_hits'] - out['current_hits']).astype(int)
    out['admit_hit_rate'] = (out['newly_admitted_hits'] / out['newly_admitted'].replace(0, pd.NA)).fillna(0.0)
    out['sample_stability'] = out.apply(compute_sample_stability, axis=1)
    out['priority_score'] = (
        out['newly_admitted_hits'] * 3.0
        + out['net_hit_gain'] * 2.0
        + out['sample_stability'] * 4.0
        + out['admit_hit_rate']
    ).round(3)
    out['priority_bucket'] = pd.cut(
        out['priority_score'],
        bins=[-1, 4.5, 8.0, 999],
        labels=['WATCH', 'TRIAL_PRIORITY', 'HIGH_PRIORITY'],
    ).astype(str)

    out = out.sort_values(
        ['newly_admitted_hits', 'net_hit_gain', 'sample_stability', 'priority_score'],
        ascending=[False, False, False, False],
    )
    out.to_csv(output_csv, index=False)

    lines = [
        '# Shadow Patch Priority Board',
        '',
        '- Ranks research-only patch cohorts by newly admitted hits, net hit gain, and sample stability.',
        '- `sample_stability` is a light confidence heuristic based on cohort size, patch tier, and removal churn.',
        '- This is still a shadow comparator, not a full deploy-policy simulator.',
        '',
    ]
    for bucket, sub in out.groupby('priority_bucket', sort=False):
        lines.append(f'## {bucket}')
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['market']} | {row['review_family']} | risk={row['prematch_risk_focus']} | tier={row['patch_confidence']} | admitted_hits={int(row['newly_admitted_hits'])} | net_hit_gain={int(row['net_hit_gain'])} | sample_stability={row['sample_stability']:.3f} | priority_score={row['priority_score']:.3f}"
            )
        lines.append('')
    Path(output_md).write_text('\n'.join(lines) + '\n')
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Rank shadow patch cohorts by newly admitted hits, net hit gain, and sample stability.')
    parser.add_argument('--input-csv', required=True)
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--output-md', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    out = build(args.input_csv, args.output_csv, args.output_md)
    print(f'WROTE: {args.output_csv}')
    print(f'rows: {len(out)}')
