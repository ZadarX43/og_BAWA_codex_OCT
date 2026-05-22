from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_master(inputs: list[str], family_tags: list[str], output_csv: str, output_md: str) -> pd.DataFrame:
    frames = []
    for path, family in zip(inputs, family_tags):
        df = pd.read_csv(path)
        if df.empty:
            continue
        df = df.copy()
        df['source_family'] = family
        frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False)
        Path(output_md).write_text('# Master Specialist Board\n\nNo rows matched.\n')
        return out
    for col in ['manual_pitch_side', 'manual_overload_target_side']:
        if col in out.columns:
            out[col] = out[col].fillna('UNSET').astype(str)

    out['master_row_key'] = (
        out['fixture_key'].astype(str)
        + '|'
        + out['player_name'].astype(str)
        + '|'
        + out['team_name'].astype(str)
        + '|'
        + out['market'].astype(str)
    )
    row_families = (
        out.groupby('master_row_key', as_index=False)
        .agg(
            row_source_family_count=('source_family', lambda s: int(pd.Series(s).nunique())),
            source_family_tag=('source_family', lambda s: '|'.join(sorted(pd.Series(s).astype(str).unique()))),
        )
    )
    bucket_rank = {'P1_SUPER_ELITE': 1, 'P2_CONTACT_STACK': 2, 'P2_ATTACK_STACK': 3, 'P3_CONTACT_CORE': 4, 'P3_ATTACK_CORE': 5, 'P4_WATCH': 6}
    out['priority_rank'] = out['priority_bucket'].map(bucket_rank).fillna(99).astype(int)
    out = (
        out.sort_values(['priority_rank', 'market_score', 'fixture_quality_score'], ascending=[True, False, False])
        .drop_duplicates(subset=['master_row_key'], keep='first')
        .copy()
    )
    out = out.merge(row_families, on='master_row_key', how='left')
    family_counts = (
        out.groupby('fixture_key', as_index=False)
        .agg(
            specialist_family_count=('source_family', lambda s: int(pd.Series(s).nunique())),
            specialist_families=('source_family', lambda s: '|'.join(sorted(pd.Series(s).astype(str).unique()))),
        )
    )
    out = out.merge(family_counts, on='fixture_key', how='left')
    out = out.sort_values(['priority_rank','fixture_quality_score','market_score'], ascending=[True,False,False]).reset_index(drop=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    lines = ['# Master Specialist Board','']
    for family, sub in out.groupby('source_family_tag', sort=False):
        lines.append(f'## {family}')
        for _, row in sub.head(20).iterrows():
            lines.append(f"- {row['priority_bucket']}: {row['fixture_key']} | {row['player_name']} ({row['team_name']}) | {row['market']} | score={row['market_score']:.1f}")
        lines.append('')
    Path(output_md).write_text('\n'.join(lines)+'\n')
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Build master specialist board from family merged boards.')
    p.add_argument('--inputs', required=True, help='Comma-separated merged board csv paths')
    p.add_argument('--family-tags', required=True, help='Comma-separated family tags aligned to inputs')
    p.add_argument('--output-csv', required=True)
    p.add_argument('--output-md', required=True)
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    inputs=[x.strip() for x in args.inputs.split(',') if x.strip()]
    tags=[x.strip() for x in args.family_tags.split(',') if x.strip()]
    if len(inputs)!=len(tags):
        raise SystemExit('inputs and family-tags must align')
    df=build_master(inputs,tags,args.output_csv,args.output_md)
    print(f'WROTE: {args.output_csv}')
    print(f'rows: {len(df)} | fixtures: {df["fixture_key"].nunique() if not df.empty else 0}')
