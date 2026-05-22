from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def infer_regen_priority(row: pd.Series) -> str:
    backfill_priority = str(row.get('backfill_priority', ''))
    league = str(row.get('league', ''))
    focus = str(row.get('goal_market_focus', ''))
    if backfill_priority == 'P1_BACKFILL':
        return 'R1_REGENERATE_FIRST'
    if league in {'La Liga', 'Serie A', 'UEFA Europa League'}:
        return 'R2_REGENERATE_NEXT'
    if 'OU25' in focus:
        return 'R3_REGEN_IF_TIME'
    return 'R4_LONG_TAIL'


def infer_regen_scope(row: pd.Series) -> str:
    focus = str(row.get('goal_market_focus', ''))
    markets = []
    if 'FTR' in focus:
        markets.append('FTR')
    if 'BTTS' in focus:
        markets.append('BTTS')
    if 'OU25' in focus:
        markets.append('OU25')
    return '|'.join(markets)


def infer_source_hint(row: pd.Series) -> str:
    league = str(row.get('league', ''))
    match_date = str(row.get('match_date', ''))[:10]
    if league in {'La Liga', 'Serie A', 'UEFA Europa League'}:
        return f'Try most recent full 3-year goal-market estate rerun slice covering {match_date} first.'
    return f'Try archive rerun slice covering {match_date}, then broader weekly regeneration if missing.'


def build(input_csv: str, output_csv: str, output_md: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv, low_memory=False)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    pending = df[df['tracker_status'].astype(str) == 'BACKFILL_PENDING'].copy()
    if pending.empty:
        empty = pd.DataFrame()
        empty.to_csv(output_csv, index=False)
        Path(output_md).write_text('# Historical Board Regeneration Target List\n\nNo pending regeneration targets matched.\n')
        return empty

    pending['regen_priority'] = pending.apply(infer_regen_priority, axis=1)
    pending['regen_market_scope'] = pending.apply(infer_regen_scope, axis=1)
    pending['regen_source_hint'] = pending.apply(infer_source_hint, axis=1)
    pending['regen_reason'] = pending.apply(
        lambda row: f"Recover fixture-level ranked-board history for {row['regen_market_scope']} so the structural caution overlay can join against settled goal-market outcomes.",
        axis=1,
    )
    out = pending.sort_values(['regen_priority', 'backfill_priority', 'league', 'match_date', 'fixture_key'])
    out.to_csv(output_csv, index=False)

    lines = [
        '# Historical Board Regeneration Target List',
        '',
        '- Derived from the current backfill tracker.',
        '- These are the fixtures where we likely need regeneration rather than archive search, because neither weekly ranked boards nor alternate estate families preserved a recoverable row.',
        '',
    ]
    for priority, sub in out.groupby('regen_priority', sort=False):
        lines.append(f'## {priority}')
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['fixture_key']} | {row['league']} | scope={row['regen_market_scope']} | focus={row['goal_market_focus']}"
            )
            lines.append(f"  reason: {row['regen_reason']}")
            lines.append(f"  hint: {row['regen_source_hint']}")
        lines.append('')
    Path(output_md).write_text('\n'.join(lines) + '\n')
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a small historical board regeneration target list from the backfill completion tracker.')
    parser.add_argument('--input-csv', required=True)
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--output-md', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    out = build(args.input_csv, args.output_csv, args.output_md)
    print(f'WROTE: {args.output_csv}')
    print(f'rows: {len(out)}')
