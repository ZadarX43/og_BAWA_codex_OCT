from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TARGETS = {
    ('Wide forward', '3421v4231'),
    ('Wide forward', '4231v433'),
    ('Wide midfielder / winger', '3421v4231'),
    ('Wide midfielder / winger', '4231v433'),
    ('Wide midfielder / winger', '4231v442'),
}


def lane_label(role: str) -> str:
    if role == 'Wide forward':
        return 'Wide-forward attack'
    if role == 'Wide midfielder / winger':
        return 'Wide-winger attack'
    return role


def action_for(row: pd.Series) -> str:
    role = str(row['tactical_role'])
    family = str(row['review_family'])
    market = str(row['market'])
    hit_rate = float(row['observed_hit_rate'])
    selected = int(row['selected_rows'])
    near = int(row['near_misses'])
    missed = int(row['missed_correct'])

    if role == 'Wide forward' and family == '3421v4231' and market == 'shots_on_target':
        return 'KEEP_PRIMARY__RELAX_CAREFULLY'
    if role == 'Wide forward' and family == '3421v4231' and market == 'shots':
        return 'KEEP_SECONDARY__NEEDS_MORE_FINISHING_PROOF'
    if role == 'Wide forward' and family == '4231v433' and market == 'shots_on_target' and selected >= 4:
        return 'KEEP_PRIMARY__DO_NOT_OVER-TIGHTEN'
    if role == 'Wide forward' and family == '4231v433' and market == 'shots':
        return 'SECONDARY_ATTACK_SUPPORT'
    if role == 'Wide midfielder / winger' and market == 'shots_on_target' and hit_rate >= 0.5:
        return 'BETA_KEEP_TRACKING'
    if role == 'Wide midfielder / winger' and market == 'shots' and missed >= 1:
        return 'WATCHLIST_ONLY'
    return 'HOLD'


def note_for(row: pd.Series) -> str:
    role = str(row['tactical_role'])
    family = str(row['review_family'])
    market = str(row['market'])
    if role == 'Wide forward' and family == '3421v4231' and market == 'shots_on_target':
        return 'This is the strongest attack lane still to refine; the signal is real enough to keep leading with shots_on_target.'
    if role == 'Wide forward' and family == '3421v4231' and market == 'shots':
        return 'The wider attack story is live, but raw shots still look less trustworthy than shots_on_target in this family.'
    if role == 'Wide forward' and family == '4231v433' and market == 'shots_on_target':
        return 'Selection count already says this lane is active; the main risk now is tightening it too much rather than leaving it too loose.'
    if role == 'Wide forward' and family == '4231v433' and market == 'shots':
        return 'Useful support market, but not the lead expression of the lane.'
    if role == 'Wide midfielder / winger' and market == 'shots_on_target':
        return 'There is enough life here to keep tracking the lane, but it still belongs in beta rather than core status.'
    if role == 'Wide midfielder / winger' and market == 'shots':
        return 'Still mostly watchlist material; too thin or too noisy to call stable yet.'
    return 'Hold this lane in the refinement pool.'


def build(runner_csv: str, output_csv: str, output_md: str) -> pd.DataFrame:
    runner = pd.read_csv(runner_csv, low_memory=False)
    runner = runner[runner.apply(lambda r: (r['tactical_role'], r['review_family']) in TARGETS, axis=1)].copy()
    for col in ['observed_success_flag', 'selection_gate_flag', 'near_miss_flag', 'missed_correct_flag', 'expected_hit_rate_3y', 'score_delta_vs_3y']:
        runner[col] = pd.to_numeric(runner[col], errors='coerce').fillna(0)

    grouped = (
        runner.groupby(['tactical_role', 'review_family', 'market'], dropna=False)
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
    grouped['lane'] = grouped['tactical_role'].map(lane_label)
    grouped['priority_action'] = grouped.apply(action_for, axis=1)
    grouped['note'] = grouped.apply(note_for, axis=1)
    grouped['observed_hit_rate'] = grouped['observed_hit_rate'].round(3)
    grouped['expected_hit_rate_3y'] = grouped['expected_hit_rate_3y'].round(3)
    grouped['avg_score_delta'] = grouped['avg_score_delta'].round(2)
    grouped = grouped.sort_values(['tactical_role', 'review_family', 'market'])
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_csv, index=False)

    lines = [
        '# Attack Lane Refinement Board',
        '',
        '- Research-only board for the current attack-side tactical lanes.',
        '- Scope: wide-forward attack in `3421v4231` and `4231v433`, plus wide-winger variants still in beta tracking.',
        '',
    ]
    current_lane = None
    for _, row in grouped.iterrows():
        lane = str(row['lane'])
        if lane != current_lane:
            current_lane = lane
            lines.append(f'## {lane}')
        lines.append(
            f"- {row['review_family']} | {row['market']} | rows={int(row['rows'])} | fixtures={int(row['fixtures'])} | hit_rate={row['observed_hit_rate']:.3f} | expected_hit={row['expected_hit_rate_3y']:.3f} | selected={int(row['selected_rows'])} | near={int(row['near_misses'])} | missed_correct={int(row['missed_correct'])} | action={row['priority_action']}"
        )
        lines.append(f"  note: {row['note']}")
    lines.extend([
        '',
        '## Current Read',
        '- `Wide-forward attack | 3421v4231` remains the main attack refinement frontier, especially through shots_on_target.',
        '- `Wide-forward attack | 4231v433` is active enough to keep live, but shots should stay secondary to shots_on_target.',
        '- Wide-winger variants still belong in the beta bucket: useful enough to watch, not mature enough to promote.',
        '',
    ])
    Path(output_md).write_text('\n'.join(lines) + '\n')
    return grouped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a small attack lane refinement board from the current walkforward evidence.')
    parser.add_argument('--runner-csv', default='reports/player_events/quality_audits/player_market_walkforward_runner.csv')
    parser.add_argument('--output-csv', default='reports/player_events/quality_audits/attack_lane_refinement_board.csv')
    parser.add_argument('--output-md', default='reports/player_events/quality_audits/attack_lane_refinement_board.md')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    out = build(args.runner_csv, args.output_csv, args.output_md)
    print(f'WROTE: {args.output_csv}')
    print(f'rows: {len(out)}')
