from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build(input_csv: str, output_md: str) -> None:
    df = pd.read_csv(input_csv, low_memory=False)
    lines = [
        '# Player Events Historical Coverage Map',
        '',
        '- Small map of which greenlist leagues currently have 2022/2023/2024 local player-stat actual coverage.',
        '',
    ]
    pivot = (
        df.assign(flag=lambda x: x['coverage_flag'].eq('AVAILABLE').astype(int))
        .pivot_table(index='league_tag', columns='season', values='flag', aggfunc='max', fill_value=0)
        .reset_index()
    )
    for _, row in pivot.iterrows():
        years = []
        for col in pivot.columns[1:]:
            years.append(f"{col}:{'YES' if int(row[col]) == 1 else 'NO'}")
        lines.append(f"- {row['league_tag']} | {' | '.join(years)}")
    lines.append('')
    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md).write_text('\n'.join(lines) + '\n')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a markdown coverage map from the greenlist historical actuals coverage CSV.')
    parser.add_argument('--input-csv', default='reports/player_events/quality_audits/greenlist_historical_actuals_coverage.csv')
    parser.add_argument('--output-md', default='reports/player_events/quality_audits/PLAYER_EVENTS_HISTORICAL_COVERAGE_MAP.md')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    build(args.input_csv, args.output_md)
