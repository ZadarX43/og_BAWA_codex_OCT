from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEY_COLS = ['lane_bucket', 'review_family', 'market', 'subtype']
VALUE_COLS = ['rows', 'fixtures', 'selected_rows', 'near_misses', 'missed_correct']


def load_tracker(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=KEY_COLS + VALUE_COLS + ['observed_hit_rate'])
    df = pd.read_csv(p, low_memory=False)
    for col in KEY_COLS:
        if col not in df.columns:
            df[col] = ''
        df[col] = df[col].fillna('').astype(str)
    for col in VALUE_COLS + ['observed_hit_rate']:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df


def build(current_csv: str, baseline_csv: str, output_csv: str, output_md: str) -> pd.DataFrame:
    current = load_tracker(current_csv)
    baseline = load_tracker(baseline_csv)

    merged = current.merge(
        baseline,
        on=KEY_COLS,
        how='outer',
        suffixes=('_current', '_baseline'),
    ).fillna(0)

    for col in VALUE_COLS + ['observed_hit_rate']:
        merged[f'delta_{col}'] = merged[f'{col}_current'] - merged[f'{col}_baseline']

    merged['status'] = 'UNCHANGED'
    moved_mask = False
    for col in VALUE_COLS:
        moved_mask = moved_mask | (merged[f'delta_{col}'] != 0)
    merged.loc[moved_mask, 'status'] = 'MOVED'
    merged.loc[(merged['rows_baseline'] == 0) & (merged['rows_current'] > 0), 'status'] = 'NEW_BUCKET'

    out_cols = KEY_COLS + [
        'status',
        'rows_current', 'rows_baseline', 'delta_rows',
        'fixtures_current', 'fixtures_baseline', 'delta_fixtures',
        'selected_rows_current', 'selected_rows_baseline', 'delta_selected_rows',
        'near_misses_current', 'near_misses_baseline', 'delta_near_misses',
        'missed_correct_current', 'missed_correct_baseline', 'delta_missed_correct',
        'observed_hit_rate_current', 'observed_hit_rate_baseline', 'delta_observed_hit_rate',
    ]
    out = merged[out_cols].sort_values(['status', 'lane_bucket', 'review_family', 'subtype', 'market'], ascending=[True, True, True, True, True])
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    moved = out[out['status'] != 'UNCHANGED']
    new_rows = int(max(0, current['rows'].sum() - baseline['rows'].sum())) if not current.empty else 0
    lines = [
        '# Tactical Accumulation Delta Audit',
        '',
        '- Compares the current tactical accumulation tracker against the saved baseline snapshot.',
        '- Use this after each rerun so we can see whether the tactical layer is genuinely gaining new settled evidence.',
        '',
        f'- total current rows across tracked buckets: `{int(current["rows"].sum()) if not current.empty else 0}`',
        f'- total baseline rows across tracked buckets: `{int(baseline["rows"].sum()) if not baseline.empty else 0}`',
        f'- net tracked row increase: `{new_rows}`',
        f'- moved buckets: `{len(moved)}`',
        '',
    ]
    if moved.empty:
        lines.append('## No Meaningful Change')
        lines.append('- Current tracker matches the saved baseline snapshot, so no new tactical evidence has been added yet.')
        lines.append('')
    else:
        lines.append('## Moved Buckets')
        for _, row in moved.iterrows():
            subtype = f" | subtype={row['subtype']}" if str(row['subtype']) else ''
            lines.append(
                f"- {row['lane_bucket']} | {row['review_family']} | {row['market']}{subtype} | status={row['status']} | delta_rows={int(row['delta_rows'])} | delta_fixtures={int(row['delta_fixtures'])} | delta_selected={int(row['delta_selected_rows'])}"
            )
        lines.append('')

    lines.extend([
        '## Trigger Rule',
        '- Treat this as a meaningful increase when one of the priority buckets adds enough settled evidence to change the tactical completion conversation, not just because a file was regenerated.',
        '- The most important watch items remain: wide-forward attack, wide-winger attack, and CB subtype multi-fixture support.',
        '',
    ])
    Path(output_md).write_text('\n'.join(lines) + '\n')
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a delta audit between the current tactical accumulation tracker and a saved baseline snapshot.')
    parser.add_argument('--current-csv', default='reports/player_events/quality_audits/tactical_lane_accumulation_tracker.csv')
    parser.add_argument('--baseline-csv', default='reports/player_events/quality_audits/tactical_lane_accumulation_tracker__baseline.csv')
    parser.add_argument('--output-csv', default='reports/player_events/quality_audits/tactical_accumulation_delta_audit.csv')
    parser.add_argument('--output-md', default='reports/player_events/quality_audits/tactical_accumulation_delta_audit.md')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    out = build(args.current_csv, args.baseline_csv, args.output_csv, args.output_md)
    print(f'WROTE: {args.output_csv}')
    print(f'rows: {len(out)}')
