from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def lane_bucket(row: pd.Series) -> str:
    role = str(row['tactical_role'])
    family = str(row['review_family'])
    if role == 'Wide forward' and family in {'3421v4231', '4231v433'}:
        return 'WIDE_FORWARD_ATTACK'
    if role == 'Wide midfielder / winger':
        return 'WIDE_WINGER_ATTACK'
    if role == 'Centre-back enforcer':
        return 'CB_SUBTYPE_WATCH'
    return 'OTHER'


def build(runner_csv: str, cb_csv: str, output_csv: str, output_md: str) -> pd.DataFrame:
    runner = pd.read_csv(runner_csv, low_memory=False)
    cb = pd.read_csv(cb_csv, low_memory=False)
    for col in ['observed_success_flag', 'selection_gate_flag', 'near_miss_flag', 'missed_correct_flag']:
        if col in runner.columns:
            runner[col] = pd.to_numeric(runner[col], errors='coerce').fillna(0)

    runner['lane_bucket'] = runner.apply(lane_bucket, axis=1)
    runner = runner[runner['lane_bucket'].isin(['WIDE_FORWARD_ATTACK', 'WIDE_WINGER_ATTACK', 'CB_SUBTYPE_WATCH'])].copy()

    grouped = (
        runner.groupby(['lane_bucket', 'review_family', 'market'], dropna=False)
        .agg(
            rows=('fixture_key', 'size'),
            fixtures=('fixture_key', 'nunique'),
            observed_hit_rate=('observed_success_flag', 'mean'),
            selected_rows=('selection_gate_flag', 'sum'),
            near_misses=('near_miss_flag', 'sum'),
            missed_correct=('missed_correct_flag', 'sum'),
        )
        .reset_index()
    )

    cb_detail = cb.copy()
    cb_detail['lane_bucket'] = 'CB_SUBTYPE_WATCH'
    cb_detail = cb_detail.rename(columns={'avg_market_hit_rate': 'observed_hit_rate'})
    cb_detail['selected_rows'] = 0
    cb_detail['near_misses'] = 0
    cb_detail['missed_correct'] = 0
    cb_detail['subtype'] = cb_detail['opponent_striker_profile']

    grouped['subtype'] = ''
    cb_rows = cb_detail[['lane_bucket', 'review_family', 'market', 'rows', 'fixtures', 'observed_hit_rate', 'selected_rows', 'near_misses', 'missed_correct', 'subtype']]
    grouped = grouped[['lane_bucket', 'review_family', 'market', 'rows', 'fixtures', 'observed_hit_rate', 'selected_rows', 'near_misses', 'missed_correct', 'subtype']]
    out = pd.concat([grouped, cb_rows], ignore_index=True)
    out['observed_hit_rate'] = pd.to_numeric(out['observed_hit_rate'], errors='coerce').round(3)
    out = out.sort_values(['lane_bucket', 'review_family', 'subtype', 'market'])

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    lines = [
        '# Tactical Lane Accumulation Tracker',
        '',
        '- Focus tracker for the lanes we still want to keep feeding with new evidence before the fresh full goal-market rebuild.',
        '- Scope: wide-forward attack, wide-winger attack, and CB subtype watch.',
        '',
    ]
    current = None
    for _, row in out.iterrows():
        bucket = str(row['lane_bucket'])
        if bucket != current:
            current = bucket
            lines.append(f'## {bucket}')
        subtype = f" | subtype={row['subtype']}" if str(row['subtype']) else ''
        lines.append(
            f"- {row['review_family']} | {row['market']}{subtype} | rows={int(row['rows'])} | fixtures={int(row['fixtures'])} | hit_rate={float(row['observed_hit_rate']):.3f} | selected={int(row['selected_rows'])} | near={int(row['near_misses'])} | missed_correct={int(row['missed_correct'])}"
        )
    lines.extend([
        '',
        '## Use',
        '- Keep adding new settled rows into these buckets and rerun this tracker periodically.',
        '- The goal is not just to find more ideas; it is to make sure later full-estate auditing has enough actual-result feature depth to score, readjust, and tune parameters cleanly.',
        '',
    ])
    Path(output_md).write_text('\n'.join(lines) + '\n')
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a small accumulation tracker for the tactical lanes still gathering evidence.')
    parser.add_argument('--runner-csv', default='reports/player_events/quality_audits/player_market_walkforward_runner.csv')
    parser.add_argument('--cb-csv', default='reports/player_events/quality_audits/cb_subtype_walkforward_audit.csv')
    parser.add_argument('--output-csv', default='reports/player_events/quality_audits/tactical_lane_accumulation_tracker.csv')
    parser.add_argument('--output-md', default='reports/player_events/quality_audits/tactical_lane_accumulation_tracker.md')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    out = build(args.runner_csv, args.cb_csv, args.output_csv, args.output_md)
    print(f'WROTE: {args.output_csv}')
    print(f'rows: {len(out)}')
