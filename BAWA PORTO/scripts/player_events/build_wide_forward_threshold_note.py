from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TARGETS = [
    ('3421v4231', 'shots'),
    ('3421v4231', 'shots_on_target'),
    ('4231v433', 'shots'),
    ('4231v433', 'shots_on_target'),
]


def action_for(row: pd.Series) -> str:
    family = str(row['review_family'])
    market = str(row['market'])
    hit_rate = float(row['observed_hit_rate'])
    selected = int(row['selected_rows'])
    near = int(row['near_misses'])
    missed = int(row['missed_correct'])

    if family == '3421v4231' and market == 'shots_on_target':
        return 'KEEP_PRIMARY__RELAX_CAREFULLY'
    if family == '3421v4231' and market == 'shots':
        return 'KEEP_SECONDARY__NEEDS_MORE_FINISHING_PROOF'
    if family == '4231v433' and market == 'shots_on_target' and selected >= 4:
        return 'KEEP_PRIMARY__DO_NOT_OVER-TIGHTEN'
    if family == '4231v433' and market == 'shots':
        return 'SECONDARY_ATTACK_SUPPORT'
    if hit_rate >= 0.4 and near >= 1:
        return 'SOFT_RELAX_RESEARCH_ONLY'
    return 'HOLD'


def note_for(row: pd.Series) -> str:
    family = str(row['review_family'])
    market = str(row['market'])
    if family == '3421v4231' and market == 'shots_on_target':
        return 'This is the clearest attacking refinement lane left in the stack; keep leading with shots_on_target and relax only carefully.'
    if family == '3421v4231' and market == 'shots':
        return 'The attacking lane is real here, but raw shots still trail shots_on_target as the cleaner tactical expression.'
    if family == '4231v433' and market == 'shots_on_target':
        return 'This lane is already live enough that the bigger risk is over-tightening it, not under-filtering it.'
    if family == '4231v433' and market == 'shots':
        return 'Useful support market, but not strong enough to lead the lane on its own.'
    return 'Hold this lane steady until more attack-side sample arrives.'


def build(runner_csv: str, output_csv: str, output_md: str) -> pd.DataFrame:
    runner = pd.read_csv(runner_csv, low_memory=False)
    runner = runner[runner['tactical_role'] == 'Wide forward'].copy()
    for col in ['observed_success_flag', 'selection_gate_flag', 'near_miss_flag', 'missed_correct_flag', 'expected_hit_rate_3y', 'score_delta_vs_3y']:
        runner[col] = pd.to_numeric(runner[col], errors='coerce').fillna(0)

    grouped = (
        runner.groupby(['review_family', 'market'], dropna=False)
        .agg(
            rows=('fixture_key', 'size'),
            fixtures=('fixture_key', 'nunique'),
            observed_hit_rate=('observed_success_flag', 'mean'),
            expected_hit_rate_3y=('expected_hit_rate_3y', 'mean'),
            selected_rows=('selection_gate_flag', 'sum'),
            near_misses=('near_miss_flag', 'sum'),
            missed_correct=('missed_correct_flag', 'sum'),
            avg_score_delta=('score_delta_vs_3y', 'mean'),
        )
        .reset_index()
    )
    grouped = grouped[grouped.apply(lambda r: (r['review_family'], r['market']) in TARGETS, axis=1)].copy()
    grouped['priority_action'] = grouped.apply(action_for, axis=1)
    grouped['note'] = grouped.apply(note_for, axis=1)
    grouped['observed_hit_rate'] = grouped['observed_hit_rate'].round(3)
    grouped['expected_hit_rate_3y'] = grouped['expected_hit_rate_3y'].round(3)
    grouped['avg_score_delta'] = grouped['avg_score_delta'].round(2)
    grouped = grouped.sort_values(['review_family', 'market'])

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_csv, index=False)

    lines = [
        '# Wide-Forward Threshold Note',
        '',
        '- Research-only threshold posture note for the strongest wide-forward attack families.',
        '- Intended use: keep the attack lane honest before any later shadow-threshold or rebuild work.',
        '',
    ]
    for _, row in grouped.iterrows():
        lines.append(
            f"- {row['review_family']} | {row['market']} | rows={int(row['rows'])} | fixtures={int(row['fixtures'])} | hit_rate={row['observed_hit_rate']:.3f} | expected_hit={row['expected_hit_rate_3y']:.3f} | selected={int(row['selected_rows'])} | near={int(row['near_misses'])} | missed_correct={int(row['missed_correct'])} | action={row['priority_action']}"
        )
        lines.append(f"  note: {row['note']}")
    Path(output_md).write_text('\n'.join(lines) + '\n')
    return grouped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a tiny research-only wide-forward threshold note.')
    parser.add_argument('--runner-csv', default='reports/player_events/quality_audits/player_market_walkforward_runner.csv')
    parser.add_argument('--output-csv', default='reports/player_events/quality_audits/wide_forward_threshold_note.csv')
    parser.add_argument('--output-md', default='reports/player_events/quality_audits/wide_forward_threshold_note.md')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    out = build(args.runner_csv, args.output_csv, args.output_md)
    print(f'WROTE: {args.output_csv}')
    print(f'rows: {len(out)}')
