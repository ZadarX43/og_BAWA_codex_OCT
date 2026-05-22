from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_summary(inputs: list[str], output_md: str) -> None:
    frames = []
    for path in inputs:
        df = pd.read_csv(path)
        if not df.empty:
            df['source_file'] = Path(path).name
            frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    lines = ['# Weekend Priority Summary','']
    if out.empty:
        lines.append('No specialist rows matched.')
        Path(output_md).write_text('\n'.join(lines)+'\n')
        return

    fixture_rank = (
        out.groupby(['fixture_key','home_team_name','away_team_name'], as_index=False)
        .agg(
            fixture_quality_score=('fixture_quality_score','max'),
            total_rows=('market','count'),
            top_market_score=('market_score','max'),
            buckets=('priority_bucket', lambda s: '|'.join(sorted(pd.Series(s).astype(str).unique()))),
        )
        .sort_values(['fixture_quality_score','top_market_score','total_rows'], ascending=[False,False,False])
    )
    best_fixture = fixture_rank.iloc[0]

    attack = out[out['market'].isin(['shots','shots_on_target'])].copy()
    best_attack = None
    if not attack.empty:
        attack_rank = (
            attack.groupby(['fixture_key','home_team_name','away_team_name'], as_index=False)
            .agg(total_rows=('market','count'), top_market_score=('market_score','max'))
            .sort_values(['top_market_score','total_rows'], ascending=[False,False])
        )
        best_attack = attack_rank.iloc[0]

    contact = out[out['market'].isin(['fouls_committed','tackles'])].copy()
    best_contact = None
    if not contact.empty:
        contact_rank = (
            contact.groupby(['fixture_key','home_team_name','away_team_name'], as_index=False)
            .agg(total_rows=('market','count'), top_market_score=('market_score','max'))
            .sort_values(['top_market_score','total_rows'], ascending=[False,False])
        )
        best_contact = contact_rank.iloc[0]

    lines.extend([
        '## Best Fixture',
        f"- {best_fixture['fixture_key']}: {best_fixture['home_team_name']} vs {best_fixture['away_team_name']} | fixture_quality={best_fixture['fixture_quality_score']:.3f} | top_score={best_fixture['top_market_score']:.1f} | rows={int(best_fixture['total_rows'])} | buckets={best_fixture['buckets']}",
        '',
    ])
    if best_attack is not None:
        lines.extend([
            '## Best Attack Stack',
            f"- {best_attack['fixture_key']}: {best_attack['home_team_name']} vs {best_attack['away_team_name']} | top_score={best_attack['top_market_score']:.1f} | rows={int(best_attack['total_rows'])}",
            '',
        ])
    if best_contact is not None:
        lines.extend([
            '## Best Contact Stack',
            f"- {best_contact['fixture_key']}: {best_contact['home_team_name']} vs {best_contact['away_team_name']} | top_score={best_contact['top_market_score']:.1f} | rows={int(best_contact['total_rows'])}",
            '',
        ])

    Path(output_md).write_text('\n'.join(lines)+'\n')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a tiny specialist weekend priority summary.')
    parser.add_argument('--inputs', required=True, help='Comma-separated merged board csv paths')
    parser.add_argument('--output-md', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    inputs = [x.strip() for x in args.inputs.split(',') if x.strip()]
    build_summary(inputs, args.output_md)
    print(f'WROTE: {args.output_md}')
