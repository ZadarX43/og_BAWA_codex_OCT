from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TARGETS = [
    ('4231v442', 'fouls_committed'),
    ('4231v442', 'tackles'),
    ('4231v433', 'tackles'),
    ('3421v4231', 'tackles'),
]


def action_for(row: pd.Series) -> str:
    family = str(row['review_family'])
    market = str(row['market'])
    hit_rate = float(row['observed_hit_rate'])
    selected = int(row['selected_rows'])
    near = int(row['near_misses'])
    missed = int(row['missed_correct'])

    if family == '4231v442' and market == 'fouls_committed':
        return 'KEEP_CONSERVATIVE__EDGE_REVIEW_ONLY'
    if family == '4231v442' and market == 'tackles':
        return 'RELAX_RESEARCH_SHADOW'
    if family == '4231v433' and market == 'tackles':
        return 'RELAX_RESEARCH_SHADOW'
    if family == '3421v4231' and market == 'tackles':
        if hit_rate >= 0.35 and (missed >= 2 or near >= 1 or selected >= 3):
            return 'SOFT_RELAX_RESEARCH_ONLY'
    return 'HOLD'


def note_for(row: pd.Series) -> str:
    family = str(row['review_family'])
    market = str(row['market'])
    if family == '4231v442' and market == 'fouls_committed':
        return 'Best fouls version of the DM lane, but survivor count is still too low versus missed-correct pressure to justify a broad relax.'
    if family == '4231v442' and market == 'tackles':
        return 'Strongest live DM lane right now; tackle survivors can be admitted more freely than the paired fouls lane.'
    if family == '4231v433' and market == 'tackles':
        return 'Good hit rate and near-miss pressure suggest we are still slightly under-admitting this version of the DM lane.'
    if family == '3421v4231' and market == 'tackles':
        return 'Useful lane, but still a softer relax than the 4-2-3-1 families until hit-rate depth thickens more.'
    return 'Hold this lane steady for now.'


def build(runner_csv: str, output_csv: str, output_md: str) -> pd.DataFrame:
    runner = pd.read_csv(runner_csv, low_memory=False)
    runner = runner[runner['tactical_role'] == 'Holding midfielder'].copy()
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
        '# DM Threshold Note',
        '',
        '- Research-only threshold posture note for the strongest holding-midfielder families.',
        '- Intended use: keep the DM lane honest before any later shadow-threshold or rebuild work.',
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
    parser = argparse.ArgumentParser(description='Build a tiny research-only DM threshold note by family.')
    parser.add_argument('--runner-csv', default='reports/player_events/quality_audits/player_market_walkforward_runner.csv')
    parser.add_argument('--output-csv', default='reports/player_events/quality_audits/dm_threshold_note.csv')
    parser.add_argument('--output-md', default='reports/player_events/quality_audits/dm_threshold_note.md')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    out = build(args.runner_csv, args.output_csv, args.output_md)
    print(f'WROTE: {args.output_csv}')
    print(f'rows: {len(out)}')
