from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def classify(avg_outlier: float, tighten: int, relax: int, patch_pressure: float) -> str:
    if tighten > relax and (avg_outlier >= 1.0 or patch_pressure > 1.0):
        return 'TIGHTEN_REGIME'
    if relax > tighten and (avg_outlier >= 0.7 or patch_pressure < -0.5):
        return 'RELAX_REGIME'
    return 'MIXED_WATCH'


def build(outlier_csv: str, scorecut_csv: str, output_csv: str, output_md: str) -> pd.DataFrame:
    outliers = pd.read_csv(outlier_csv, low_memory=False)
    scorecuts = pd.read_csv(scorecut_csv, low_memory=False)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    if outliers.empty:
        empty = pd.DataFrame()
        empty.to_csv(output_csv, index=False)
        Path(output_md).write_text('# League-Specific Threshold Regime Board\n\nNo rows matched.\n')
        return empty

    outliers['outlier_score'] = pd.to_numeric(outliers['outlier_score'], errors='coerce').fillna(0.0)
    scorecuts['proposed_score_cut_shift'] = pd.to_numeric(scorecuts['proposed_score_cut_shift'], errors='coerce').fillna(0.0)

    league_summary = (
        outliers.groupby(['league', 'competition'], dropna=False)
        .agg(
            teams=('team_name', pd.Series.nunique),
            rows=('team_name', 'size'),
            avg_outlier_score=('outlier_score', 'mean'),
            tighten_outliers=('threshold_posture', lambda s: int((s == 'TIGHTEN').sum())),
            relax_outliers=('threshold_posture', lambda s: int((s == 'RELAX').sum())),
        )
        .reset_index()
    )

    pressure = (
        outliers.merge(scorecuts[['market', 'review_family', 'prematch_risk_focus', 'proposed_score_cut_shift']], on=['market', 'review_family'], how='left')
        .groupby(['league', 'competition'], dropna=False)
        .agg(avg_patch_pressure=('proposed_score_cut_shift', 'mean'))
        .reset_index()
    )
    board = league_summary.merge(pressure, on=['league', 'competition'], how='left')
    board['avg_patch_pressure'] = pd.to_numeric(board['avg_patch_pressure'], errors='coerce').fillna(0.0)
    board['regime_posture'] = board.apply(
        lambda row: classify(row['avg_outlier_score'], int(row['tighten_outliers']), int(row['relax_outliers']), row['avg_patch_pressure']),
        axis=1,
    )
    board = board.sort_values(['regime_posture', 'avg_outlier_score', 'rows'], ascending=[True, False, False])
    board.to_csv(output_csv, index=False)

    lines = [
        '# League-Specific Threshold Regime Board',
        '',
        '- `TIGHTEN_REGIME`: local evidence says the league/competition is over-trusted relative to current gates.',
        '- `RELAX_REGIME`: local evidence says the league/competition is under-selected relative to current gates.',
        '- `MIXED_WATCH`: not stable enough yet; keep learning.',
        '',
    ]
    for posture, sub in board.groupby('regime_posture', sort=False):
        lines.append(f'## {posture}')
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['league']} | {row['competition']} | teams={int(row['teams'])} | rows={int(row['rows'])} | avg_outlier={row['avg_outlier_score']:.3f} | tighten={int(row['tighten_outliers'])} | relax={int(row['relax_outliers'])} | avg_patch_pressure={row['avg_patch_pressure']:+.2f}"
            )
        lines.append('')
    Path(output_md).write_text('\n'.join(lines) + '\n')
    return board


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a league-specific threshold regime board from outliers and score-cut posture.')
    parser.add_argument('--outlier-csv', required=True)
    parser.add_argument('--scorecut-csv', required=True)
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--output-md', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    out = build(args.outlier_csv, args.scorecut_csv, args.output_csv, args.output_md)
    print(f'WROTE: {args.output_csv}')
    print(f'rows: {len(out)}')
