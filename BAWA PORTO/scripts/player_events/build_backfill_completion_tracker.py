from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def infer_tracker_status(row: pd.Series) -> str:
    status = str(row.get('latest_overlap_status', ''))
    joined_flag = int(pd.to_numeric(row.get('latest_joined_flag'), errors='coerce') or 0)
    has_ranked = int(pd.to_numeric(row.get('latest_has_ranked_history'), errors='coerce') or 0)
    has_actual = int(pd.to_numeric(row.get('latest_has_actual_result'), errors='coerce') or 0)
    if joined_flag or status == 'MATCHED_IN_JOIN':
        return 'RECOVERED_IN_JOIN'
    if has_ranked:
        return 'RECOVERED_RANKED_HISTORY'
    if has_actual and not has_ranked:
        return 'BACKFILL_PENDING'
    return 'NEEDS_MANUAL_RECHECK'


def infer_next_action(row: pd.Series) -> str:
    tracker_status = str(row.get('tracker_status', ''))
    if tracker_status == 'RECOVERED_IN_JOIN':
        return 'No action needed; fixture already flows into the joined audit.'
    if tracker_status == 'RECOVERED_RANKED_HISTORY':
        return 'Re-run the goal-market surprise join to confirm the recovered ranked-board history now joins cleanly.'
    if tracker_status == 'BACKFILL_PENDING':
        return 'Backfill ranked-board history for the listed goal-market focus, then rerun overlap and join audits.'
    return 'Check fixture key normalization and ranked-board archive paths manually.'


def build(target_csv: str, overlap_csv: str, output_csv: str, output_md: str) -> pd.DataFrame:
    targets = pd.read_csv(target_csv, low_memory=False)
    overlap = pd.read_csv(overlap_csv, low_memory=False)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    if targets.empty:
        empty = pd.DataFrame()
        empty.to_csv(output_csv, index=False)
        Path(output_md).write_text('# Backfill Completion Tracker\n\nNo ranked-board backfill targets matched.\n')
        return empty

    latest = overlap[[
        'fixture_key',
        'status',
        'joined_flag',
        'has_ranked_history',
        'has_actual_result',
        'ranked_markets_found',
        'ranked_rows',
        'reason',
    ]].drop_duplicates('fixture_key').rename(
        columns={
            'status': 'latest_overlap_status',
            'joined_flag': 'latest_joined_flag',
            'has_ranked_history': 'latest_has_ranked_history',
            'has_actual_result': 'latest_has_actual_result',
            'ranked_markets_found': 'latest_ranked_markets_found',
            'ranked_rows': 'latest_ranked_rows',
            'reason': 'latest_overlap_reason',
        }
    )

    out = targets.merge(latest, on='fixture_key', how='left')
    out['tracker_status'] = out.apply(infer_tracker_status, axis=1)
    out['completion_flag'] = out['tracker_status'].isin({'RECOVERED_IN_JOIN', 'RECOVERED_RANKED_HISTORY'}).astype(int)
    out['next_action'] = out.apply(infer_next_action, axis=1)
    out = out.sort_values(['completion_flag', 'backfill_priority', 'league', 'match_date', 'fixture_key'], ascending=[True, True, True, True, True])
    out.to_csv(output_csv, index=False)

    lines = [
        '# Backfill Completion Tracker',
        '',
        '- Snapshot tracker for ranked-board coverage recovery over time.',
        '- `BACKFILL_PENDING` means the fixture still has settled results but no matched ranked-board history.',
        '- `RECOVERED_RANKED_HISTORY` means ranked-board history exists and the next check is whether it joins cleanly.',
        '- `RECOVERED_IN_JOIN` means the fixture is now flowing through the goal-market surprise join audit.',
        '',
    ]
    summary = out.groupby('tracker_status', dropna=False).agg(rows=('fixture_key', 'size')).reset_index()
    lines.append('## Summary')
    for _, row in summary.iterrows():
        lines.append(f"- {row['tracker_status']} | rows={int(row['rows'])}")
    lines.append('')
    for status, sub in out.groupby('tracker_status', sort=False):
        lines.append(f'## {status}')
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['fixture_key']} | priority={row['backfill_priority']} | focus={row['goal_market_focus']} | markets={('none' if pd.isna(row.get('latest_ranked_markets_found')) or not str(row.get('latest_ranked_markets_found')).strip() else row.get('latest_ranked_markets_found'))}"
            )
            lines.append(f"  next: {row['next_action']}")
        lines.append('')
    Path(output_md).write_text('\n'.join(lines) + '\n')
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a snapshot tracker for ranked-board backfill completion.')
    parser.add_argument('--target-csv', required=True)
    parser.add_argument('--overlap-csv', required=True)
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--output-md', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    out = build(args.target_csv, args.overlap_csv, args.output_csv, args.output_md)
    print(f'WROTE: {args.output_csv}')
    print(f'rows: {len(out)}')
