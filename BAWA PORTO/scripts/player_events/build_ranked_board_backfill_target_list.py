from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def infer_priority(row: pd.Series) -> str:
    risk = str(row.get('prematch_risk_focus', ''))
    league = str(row.get('league', ''))
    if 'missing DM' in risk and league in {'La Liga', 'Serie A'}:
        return 'P1_BACKFILL'
    if 'missing full-back' in risk or 'missing CB duel anchor' in risk:
        return 'P2_BACKFILL'
    return 'P3_BACKFILL'


def infer_goal_market_focus(row: pd.Series) -> str:
    risk = str(row.get('prematch_risk_focus', ''))
    markets = []
    if 'missing DM' in risk:
        markets.extend(['FTR', 'BTTS'])
    if 'missing full-back' in risk:
        markets.extend(['BTTS', 'OU25'])
    if 'missing CB duel anchor' in risk:
        markets.extend(['FTR', 'OU25'])
    ordered = []
    for m in markets:
        if m not in ordered:
            ordered.append(m)
    return '|'.join(ordered)


def build(input_csv: str, output_csv: str, output_md: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv, low_memory=False)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out = df[df['status'] == 'HAS_ACTUAL_NO_RANKED_HISTORY'].copy()
    if out.empty:
        out.to_csv(output_csv, index=False)
        Path(output_md).write_text('# Ranked-Board Coverage Backfill Target List\n\nNo backfill targets matched.\n')
        return out
    out['backfill_priority'] = out.apply(infer_priority, axis=1)
    out['goal_market_focus'] = out.apply(infer_goal_market_focus, axis=1)
    out['backfill_reason'] = out.apply(
        lambda row: f"Need historical ranked-board coverage for {row['goal_market_focus']} because this structural-risk fixture has a settled result but no archived ranked-board match.",
        axis=1,
    )
    out = out.sort_values(['backfill_priority', 'league', 'match_date', 'fixture_key'])
    out.to_csv(output_csv, index=False)

    lines = [
        '# Ranked-Board Coverage Backfill Target List',
        '',
        '- These fixtures have canonical actual results but no archived ranked FTR / BTTS / OU25 history match.',
        '- This is the clean queue for widening the goal-market surprise join coverage.',
        '',
    ]
    for priority, sub in out.groupby('backfill_priority', sort=False):
        lines.append(f'## {priority}')
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['fixture_key']} | {row['league']} | {row['competition']} | risk={row['prematch_risk_focus']} | focus={row['goal_market_focus']}"
            )
            lines.append(f"  reason: {row['backfill_reason']}")
        lines.append('')
    Path(output_md).write_text('\n'.join(lines) + '\n')
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a ranked-board coverage backfill target list from overlap expansion audit rows.')
    parser.add_argument('--input-csv', required=True)
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--output-md', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    out = build(args.input_csv, args.output_csv, args.output_md)
    print(f'WROTE: {args.output_csv}')
    print(f'rows: {len(out)}')
