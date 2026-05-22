from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEEP_BUCKETS = ["P1_SUPER_ELITE", "P2_CONTACT_STACK", "P2_ATTACK_STACK"]


def _confidence_note(row: pd.Series) -> str:
    bucket = str(row.get("priority_bucket", "") or "")
    fixture_quality = float(row.get("fixture_quality_score", 0.0) or 0.0)
    family_count = int(row.get("specialist_family_count", 1) or 1)
    source_count = int(row.get("fixture_source_count", 1) or 1)
    if bucket == "P1_SUPER_ELITE":
        return "Highest-conviction specialist survivor."
    if family_count >= 2 and source_count >= 2:
        return "Multi-family agreement with cross-board support."
    if bucket == "P2_ATTACK_STACK":
        return "Attack stack supported by formation-family concentration."
    if bucket == "P2_CONTACT_STACK":
        return "Contact stack supported by pressure/shape agreement."
    if fixture_quality >= 0.80:
        return "Strong fixture quality, but support stack is thinner."
    return "Useful specialist lean, but lower agreement than top buckets."


def build_export(inputs: list[str], output_csv: str, output_md: str) -> pd.DataFrame:
    frames = []
    for path in inputs:
        df = pd.read_csv(path)
        if not df.empty:
            df['source_file'] = Path(path).name
            frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False)
        Path(output_md).write_text('# Specialist Final Shortlist Export\n\nNo rows matched.\n')
        return out
    out = out[out['priority_bucket'].isin(KEEP_BUCKETS)].copy()
    if 'source_family_tag' not in out.columns and 'source_family' in out.columns:
        out['source_family_tag'] = out['source_family']
    for col in ['manual_pitch_side', 'manual_overload_target_side']:
        if col in out.columns:
            out[col] = out[col].fillna('UNSET').astype(str)
    bucket_order = {b: i for i, b in enumerate(KEEP_BUCKETS, start=1)}
    out['priority_rank'] = out['priority_bucket'].map(bucket_order).fillna(99).astype(int)
    out['fixture_confidence_note'] = out.apply(_confidence_note, axis=1)
    out = out.sort_values(['priority_rank','fixture_quality_score','market_score'], ascending=[True,False,False]).reset_index(drop=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    lines = ['# Specialist Final Shortlist Export','',f"- kept buckets: {', '.join(KEEP_BUCKETS)}",'']
    for bucket, sub in out.groupby('priority_bucket', sort=False):
        lines.append(f'## {bucket}')
        for _, row in sub.iterrows():
            family = row.get('source_family_tag', '')
            family_text = f" | family={family}" if str(family) not in {'', 'nan', 'NaN'} else ''
            lines.append(f"- {row['fixture_key']}: {row['player_name']} ({row['team_name']}) | {row['market']} | score={row['market_score']:.1f} | note={row['fixture_confidence_note']}{family_text} | source_file={row['source_file']}")
        lines.append('')
    Path(output_md).write_text('\n'.join(lines)+'\n')
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build final shortlist export from merged specialist boards.')
    parser.add_argument('--inputs', required=True, help='Comma-separated merged board csv paths')
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--output-md', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    inputs = [x.strip() for x in args.inputs.split(',') if x.strip()]
    df = build_export(inputs, args.output_csv, args.output_md)
    print(f'WROTE: {args.output_csv}')
    print(f'rows: {len(df)} | fixtures: {df["fixture_key"].nunique() if not df.empty else 0}')
