from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def classify(row: pd.Series) -> str:
    rows = int(row.get('rows', 0) or 0)
    fixtures = int(row.get('fixtures', 0) or 0)
    shift = float(pd.to_numeric(row.get('proposed_score_cut_shift'), errors='coerce') or 0.0)
    if rows >= 3 and fixtures >= 2 and abs(shift) >= 2.0:
        return 'APPLY_SOFT_PATCH'
    if rows >= 2 and abs(shift) >= 1.0:
        return 'TRIAL_PATCH'
    return 'WATCH_ONLY'


def build(input_csv: str, output_csv: str, output_md: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv, low_memory=False)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        df.to_csv(output_csv, index=False)
        Path(output_md).write_text('# Applied Threshold Patch Proposal\n\nNo rows matched.\n')
        return df

    out = df.copy()
    out['proposed_score_cut_shift'] = pd.to_numeric(out['proposed_score_cut_shift'], errors='coerce').fillna(0.0)
    out['rows'] = pd.to_numeric(out['rows'], errors='coerce').fillna(0)
    out['fixtures'] = pd.to_numeric(out['fixtures'], errors='coerce').fillna(0)
    out['avg_expected_hit'] = pd.to_numeric(out['avg_expected_hit'], errors='coerce')
    out['avg_score_delta'] = pd.to_numeric(out['avg_score_delta'], errors='coerce')
    out['patch_confidence'] = out.apply(classify, axis=1)
    out['patch_key'] = out['market'].astype(str) + ' | ' + out['review_family'].astype(str) + ' | ' + out['prematch_risk_focus'].astype(str)
    out = out.sort_values(['patch_confidence', 'market', 'review_family', 'prematch_risk_focus', 'rows'], ascending=[True, True, True, True, False])
    out.to_csv(output_csv, index=False)

    lines = [
        '# Applied Threshold Patch Proposal',
        '',
        '- `APPLY_SOFT_PATCH`: enough sample and directional signal to trial a real gate change in the research stack.',
        '- `TRIAL_PATCH`: promising directional signal, but still smaller sample.',
        '- `WATCH_ONLY`: keep learning before changing thresholds.',
        '',
    ]
    for confidence, sub in out.groupby('patch_confidence', sort=False):
        lines.append(f'## {confidence}')
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['market']} | {row['review_family']} | risk={row['prematch_risk_focus']} | action={row['proposed_gate_action']} | shift={row['proposed_score_cut_shift']:+.1f} | rows={int(row['rows'])} | fixtures={int(row['fixtures'])} | avg_expected_hit={row['avg_expected_hit']:.3f} | avg_score_delta={row['avg_score_delta']:.2f}"
            )
        lines.append('')
    Path(output_md).write_text('\n'.join(lines) + '\n')
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build an applied threshold patch proposal from the numeric score-cut table.')
    parser.add_argument('--input-csv', required=True)
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--output-md', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    out = build(args.input_csv, args.output_csv, args.output_md)
    print(f'WROTE: {args.output_csv}')
    print(f'rows: {len(out)}')
