from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _bucket(row: pd.Series) -> str:
    source = str(row.get('source_board', '') or '')
    market = str(row.get('market', '') or '')
    fixture_count = int(row.get('fixture_source_count', 0) or 0)
    if source == 'SUPER_ELITE':
        return 'P1_SUPER_ELITE'
    if fixture_count >= 2 and market in {'fouls_committed', 'tackles'}:
        return 'P2_CONTACT_STACK'
    if fixture_count >= 2 and market in {'shots', 'shots_on_target'}:
        return 'P2_ATTACK_STACK'
    if source == 'CONTACT':
        return 'P3_CONTACT_CORE'
    if source == 'ATTACKING':
        return 'P3_ATTACK_CORE'
    return 'P4_WATCH'


def build_board(super_csv: str, attacking_csv: str, contact_csv: str, output_csv: str, output_md: str, title: str = 'Final 4-2-3-1 vs 4-4-2 Weekend Merge Board') -> pd.DataFrame:
    super_df = pd.read_csv(super_csv)
    att_df = pd.read_csv(attacking_csv)
    con_df = pd.read_csv(contact_csv)

    frames = []
    if not super_df.empty:
        s = super_df.copy()
        s['source_board'] = 'SUPER_ELITE'
        frames.append(s)
    if not att_df.empty:
        a = att_df.copy()
        a['source_board'] = 'ATTACKING'
        frames.append(a)
    if not con_df.empty:
        c = con_df.copy()
        c['source_board'] = 'CONTACT'
        frames.append(c)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False)
        Path(output_md).write_text(f'# {title}\n\nNo rows matched.\n')
        return out

    out['row_key'] = (
        out['fixture_key'].astype(str) + '|' + out['player_name'].astype(str) + '|' + out['team_name'].astype(str) + '|' + out['market'].astype(str)
    )
    row_sources = (
        out.groupby('row_key', as_index=False)
        .agg(
            row_source_count=('source_board', lambda s: int(pd.Series(s).nunique())),
            row_sources=('source_board', lambda s: '|'.join(sorted(pd.Series(s).astype(str).unique()))),
        )
    )
    source_priority = {'SUPER_ELITE': 1, 'CONTACT': 2, 'ATTACKING': 3}
    out['source_priority'] = out['source_board'].map(source_priority).fillna(99).astype(int)
    out = out.sort_values(['source_priority', 'market_score'], ascending=[True, False]).drop_duplicates(subset=['row_key'], keep='first').copy()
    out = out.merge(row_sources, on='row_key', how='left')

    fixture_sources = (
        out.groupby('fixture_key', as_index=False)
        .agg(
            fixture_source_count=('source_board', lambda s: int(pd.Series(s).nunique())),
            fixture_sources=('source_board', lambda s: '|'.join(sorted(pd.Series(s).astype(str).unique()))),
        )
    )
    out = out.merge(fixture_sources, on='fixture_key', how='left')

    player_fixture_counts = (
        out.groupby(['fixture_key', 'player_name', 'team_name'], as_index=False)
        .agg(
            player_market_count=('market', 'nunique'),
            player_markets=('market', lambda s: '|'.join(sorted(pd.Series(s).astype(str).unique()))),
        )
    )
    out = out.merge(player_fixture_counts, on=['fixture_key', 'player_name', 'team_name'], how='left')
    out['priority_bucket'] = out.apply(_bucket, axis=1)

    bucket_order = {
        'P1_SUPER_ELITE': 1,
        'P2_CONTACT_STACK': 2,
        'P2_ATTACK_STACK': 3,
        'P3_CONTACT_CORE': 4,
        'P3_ATTACK_CORE': 5,
        'P4_WATCH': 6,
    }
    out['priority_rank'] = out['priority_bucket'].map(bucket_order).fillna(99).astype(int)
    out = out.sort_values(['priority_rank', 'fixture_quality_score', 'market_score'], ascending=[True, False, False]).reset_index(drop=True)

    cols = [
        'priority_bucket','fixture_key','match_date','competition','league','home_team_name','away_team_name',
        'team_name','player_name','player_team_side','position_group','tactical_role','market','market_score','market_confidence','source_board',
        'fixture_source_count','fixture_sources','player_market_count','player_markets',
        'formation_matchup_label','fixture_style_label','fixture_attacking_style_label','fixture_quality_score',
        'formation_pressure_score','starting_xi_quality_edge','player_quality_score_l5','manual_pitch_side','manual_overload_target_side'
    ]
    out = out[cols]

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    lines = [
        f'# {title}',
        '',
        'Priority buckets:',
        '- `P1_SUPER_ELITE`: strongest surviving ultra-tight rows',
        '- `P2_CONTACT_STACK`: contact rows in fixtures represented across all specialist boards',
        '- `P2_ATTACK_STACK`: attacking rows in fixtures represented across all specialist boards',
        '- `P3_CONTACT_CORE`: contact-only specialist survivors',
        '- `P3_ATTACK_CORE`: attacking-only specialist survivors',
        '- `P4_WATCH`: lower-priority residual rows',
        '',
    ]
    for bucket, sub in out.groupby('priority_bucket', sort=False):
        lines.append(f'## {bucket}')
        for _, row in sub.iterrows():
            side = f" | manual_side={row['manual_pitch_side']}->{row['manual_overload_target_side']}" if str(row.get('manual_pitch_side','')) not in {'', 'nan', 'NaN'} else ''
            lines.append(
                f"- {row['fixture_key']}: {row['player_name']} ({row['team_name']}) | {row['market']} | score={row['market_score']:.1f} | source={row['source_board']} | fixture_sources={row['fixture_sources']} | edge={row['starting_xi_quality_edge']:.1f}{side}"
            )
        lines.append('')
    Path(output_md).write_text('\n'.join(lines) + '\n')
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build final merged 4-2-3-1 vs 4-4-2 weekend board.')
    parser.add_argument('--super-csv', required=True)
    parser.add_argument('--attacking-csv', required=True)
    parser.add_argument('--contact-csv', required=True)
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--output-md', required=True)
    parser.add_argument('--title', default='Final 4-2-3-1 vs 4-4-2 Weekend Merge Board')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    df = build_board(args.super_csv, args.attacking_csv, args.contact_csv, args.output_csv, args.output_md, title=args.title)
    print(f'WROTE: {args.output_csv}')
    print(f'rows: {len(df)} | fixtures: {df["fixture_key"].nunique() if not df.empty else 0}')
