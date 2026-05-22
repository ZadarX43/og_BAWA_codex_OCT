from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build(scorecut_csv: str, patch_csv: str, outlier_csv: str, output_md: str) -> None:
    scorecuts = pd.read_csv(scorecut_csv, low_memory=False)
    patches = pd.read_csv(patch_csv, low_memory=False)
    outliers = pd.read_csv(outlier_csv, low_memory=False)
    Path(output_md).parent.mkdir(parents=True, exist_ok=True)

    lines = [
        '# Team x League x Competition Tuning Dashboard',
        '',
        '- Pulls together score-cut pressure, applied patch posture, and team outlier behavior.',
        '- Research only: this is a tuning board, not a live deploy rules change.',
        '',
    ]

    if not patches.empty:
        lines.append('## Patch Priority')
        for _, row in patches[patches['patch_confidence'].isin(['APPLY_SOFT_PATCH', 'TRIAL_PATCH'])].head(20).iterrows():
            lines.append(
                f"- {row['market']} | {row['review_family']} | risk={row['prematch_risk_focus']} | {row['patch_confidence']} | {row['proposed_gate_action']} {row['proposed_score_cut_shift']:+.1f}"
            )
        lines.append('')

    if not outliers.empty:
        lines.append('## Team Outliers')
        for _, row in outliers.head(25).iterrows():
            lines.append(
                f"- {row['team_name']} | {row['league']} | {row['competition']} | {row['market']} | {row['review_family']} | posture={row['threshold_posture']} | tier={row['watchlist_tier']} | outlier_score={row['outlier_score']:.3f}"
            )
        lines.append('')

    if not scorecuts.empty:
        lines.append('## Market x Family Pressure')
        summary = (
            scorecuts.groupby(['market', 'review_family'], dropna=False)
            .agg(rows=('rows', 'sum'), avg_shift=('proposed_score_cut_shift', 'mean'))
            .reset_index()
            .sort_values(['rows', 'avg_shift'], ascending=[False, True])
        )
        for _, row in summary.iterrows():
            direction = 'LOWER' if row['avg_shift'] < 0 else 'RAISE' if row['avg_shift'] > 0 else 'HOLD'
            lines.append(
                f"- {row['market']} | {row['review_family']} | rows={int(row['rows'])} | avg_shift={row['avg_shift']:+.2f} | bias={direction}"
            )
        lines.append('')

    Path(output_md).write_text('\n'.join(lines) + '\n')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a combined team x league x competition tuning dashboard markdown.')
    parser.add_argument('--scorecut-csv', required=True)
    parser.add_argument('--patch-csv', required=True)
    parser.add_argument('--outlier-csv', required=True)
    parser.add_argument('--output-md', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    build(args.scorecut_csv, args.patch_csv, args.outlier_csv, args.output_md)
    print(f'WROTE: {args.output_md}')
