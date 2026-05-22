from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


INCLUDED_PATCH_TIERS = {'APPLY_SOFT_PATCH', 'TRIAL_PATCH'}


def build(patch_csv: str, runner_csv: str, output_csv: str, output_md: str) -> pd.DataFrame:
    patches = pd.read_csv(patch_csv, low_memory=False)
    runner = pd.read_csv(runner_csv, low_memory=False)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    cohort_patches = patches[patches['patch_confidence'].isin(INCLUDED_PATCH_TIERS)].copy()
    if cohort_patches.empty or runner.empty:
        empty = pd.DataFrame()
        empty.to_csv(output_csv, index=False)
        Path(output_md).write_text('# Shadow Threshold Trial Table\n\nNo APPLY_SOFT_PATCH / TRIAL_PATCH cohorts matched.\n')
        return empty

    runner['score_delta_vs_3y'] = pd.to_numeric(runner['score_delta_vs_3y'], errors='coerce')
    runner['observed_success_flag'] = pd.to_numeric(runner['observed_success_flag'], errors='coerce').fillna(0)
    runner['selection_gate_flag'] = pd.to_numeric(runner['selection_gate_flag'], errors='coerce').fillna(0)

    rows: list[dict[str, object]] = []
    for _, patch in cohort_patches.iterrows():
        cohort = runner[
            (runner['market'].astype(str) == str(patch['market']))
            & (runner['review_family'].astype(str) == str(patch['review_family']))
            & (runner['prematch_risk_focus'].astype(str) == str(patch['prematch_risk_focus']))
        ].copy()
        if cohort.empty:
            continue
        shift = float(pd.to_numeric(patch['proposed_score_cut_shift'], errors='coerce') or 0.0)
        cohort['shadow_gate_flag'] = (cohort['score_delta_vs_3y'] >= shift).astype(int)
        cohort['newly_admitted_flag'] = ((cohort['shadow_gate_flag'] == 1) & (cohort['selection_gate_flag'] == 0)).astype(int)
        cohort['newly_removed_flag'] = ((cohort['shadow_gate_flag'] == 0) & (cohort['selection_gate_flag'] == 1)).astype(int)
        rows.append(
            {
                'market': patch['market'],
                'review_family': patch['review_family'],
                'prematch_risk_focus': patch['prematch_risk_focus'],
                'patch_confidence': patch['patch_confidence'],
                'proposed_gate_action': patch['proposed_gate_action'],
                'proposed_score_cut_shift': shift,
                'runner_rows': int(len(cohort)),
                'current_selected': int(cohort['selection_gate_flag'].sum()),
                'shadow_selected': int(cohort['shadow_gate_flag'].sum()),
                'newly_admitted': int(cohort['newly_admitted_flag'].sum()),
                'newly_removed': int(cohort['newly_removed_flag'].sum()),
                'current_hits': int(((cohort['selection_gate_flag'] == 1) & (cohort['observed_success_flag'] == 1)).sum()),
                'shadow_hits': int(((cohort['shadow_gate_flag'] == 1) & (cohort['observed_success_flag'] == 1)).sum()),
                'newly_admitted_hits': int(((cohort['newly_admitted_flag'] == 1) & (cohort['observed_success_flag'] == 1)).sum()),
                'newly_removed_hits': int(((cohort['newly_removed_flag'] == 1) & (cohort['observed_success_flag'] == 1)).sum()),
                'avg_score_delta': float(cohort['score_delta_vs_3y'].mean()),
            }
        )
    out = pd.DataFrame(rows).sort_values(['patch_confidence', 'market', 'review_family', 'prematch_risk_focus'])
    out.to_csv(output_csv, index=False)

    lines = [
        '# Shadow Threshold Trial Table',
        '',
        '- Research-only table for `APPLY_SOFT_PATCH` and `TRIAL_PATCH` cohorts.',
        '- `shadow_selected` simulates the cohort under the proposed score-cut shift without touching live rules.',
        '',
    ]
    for confidence, sub in out.groupby('patch_confidence', sort=False):
        lines.append(f'## {confidence}')
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['market']} | {row['review_family']} | risk={row['prematch_risk_focus']} | {row['proposed_gate_action']} {row['proposed_score_cut_shift']:+.1f} | current={int(row['current_selected'])} | shadow={int(row['shadow_selected'])} | admitted={int(row['newly_admitted'])} | removed={int(row['newly_removed'])} | shadow_hits={int(row['shadow_hits'])}"
            )
        lines.append('')
    Path(output_md).write_text('\n'.join(lines) + '\n')
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Trial APPLY_SOFT_PATCH and TRIAL_PATCH cohorts inside a research-only shadow threshold table.')
    parser.add_argument('--patch-csv', required=True)
    parser.add_argument('--runner-csv', required=True)
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--output-md', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    out = build(args.patch_csv, args.runner_csv, args.output_csv, args.output_md)
    print(f'WROTE: {args.output_csv}')
    print(f'rows: {len(out)}')
