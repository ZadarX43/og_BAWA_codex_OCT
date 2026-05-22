from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DIRECT_LEAGUE_HINTS = {
    'England_Championship',
    'England_EFL_League_1',
    'Scotland_Premiership',
    'Belgium_Pro',
    'Norway_Eliteserien',
    'Brazil_Serie_A',
    'USA_MLS',
    'Japan_J1',
    'Portugal_Liga',
    'France_Ligue_1',
    'Netherlands_Eredivisie',
    'Europa_Conference',
}


def build_hunt(inputs: list[str], output_csv: str, output_md: str) -> pd.DataFrame:
    frames = []
    for path in inputs:
        p = Path(path)
        if p.exists():
            frames.append(pd.read_csv(p, low_memory=False))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        Path(output_md).write_text('# Striker vs Front-Foot CB Hunt\n\nNo rows matched.\n')
        return df

    if 'tactical_role' not in df.columns:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.head(0).to_csv(output_csv, index=False)
        Path(output_md).write_text('# Striker vs Front-Foot CB Hunt\n\nSource rows do not carry tactical_role.\n')
        return df.head(0)

    family_series = df.get('source_family_tag', df.get('review_family', pd.Series('', index=df.index))).astype(str)
    score_series = pd.to_numeric(df.get('market_score', df.get('score', pd.Series(0.0, index=df.index))), errors='coerce').fillna(0.0)
    quality_series = pd.to_numeric(df.get('fixture_quality_score', pd.Series(0.0, index=df.index)), errors='coerce').fillna(0.0)
    pressure_series = pd.to_numeric(df.get('formation_pressure_score', pd.Series(0.0, index=df.index)), errors='coerce').fillna(0.0)
    role_mask = df['tactical_role'].astype(str).eq('Centre-back enforcer')
    family_mask = family_series.isin(['3421v4231', '4231v442', '4231v433'])
    league_mask = df.get('league', pd.Series('', index=df.index)).astype(str).isin(DIRECT_LEAGUE_HINTS)
    market_mask = df.get('market', pd.Series('', index=df.index)).astype(str).isin(['tackles', 'fouls_committed'])
    pressure_hint = pressure_series.ge(0.45) | score_series.ge(95.0) | quality_series.ge(0.78)
    hunt = df[role_mask & family_mask & league_mask & market_mask & pressure_hint].copy()
    if hunt.empty:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        hunt.to_csv(output_csv, index=False)
        lines = [
            '# Striker vs Front-Foot CB Hunt',
            '',
        '- No current centre-back enforcer rows survived even after widening the direct / duel-heavy league group.',
        '- This lane is wired, but the current specialist outputs are not yet surfacing enough front-foot CB sample.',
        '',
    ]
        Path(output_md).write_text('\n'.join(lines))
        return hunt

    hunt['hunt_priority'] = (
        pd.to_numeric(hunt.get('market_score'), errors='coerce').fillna(0.0)
        + 6.0 * pd.to_numeric(hunt.get('formation_pressure_score'), errors='coerce').fillna(0.0)
        + 4.0 * pd.to_numeric(hunt.get('fixture_quality_score'), errors='coerce').fillna(0.0)
    )
    cols = [c for c in [
        'fixture_key','match_date','league','home_team_name','away_team_name','team_name','player_name','market','tactical_role','source_family_tag','market_score','fixture_quality_score','formation_pressure_score','hunt_priority'
    ] if c in hunt.columns]
    hunt = hunt[cols].sort_values(['hunt_priority', 'market_score'], ascending=[False, False]).reset_index(drop=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    hunt.to_csv(output_csv, index=False)

    lines = ['# Striker vs Front-Foot CB Hunt', '', f'- rows: {len(hunt)} | fixtures: {hunt["fixture_key"].nunique() if not hunt.empty else 0}', '']
    for _, row in hunt.head(20).iterrows():
        lines.append(
            f"- {row['fixture_key']} | {row['player_name']} ({row['team_name']}) | {row['market']} | family={row['source_family_tag']} | score={row['market_score']:.1f}"
        )
    Path(output_md).write_text('\n'.join(lines) + '\n')
    return hunt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Hunt for striker-vs-front-foot-CB style rows in direct / duel-heavy leagues.')
    parser.add_argument('--inputs', required=True, help='Comma-separated CSV inputs to scan.')
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--output-md', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    out = build_hunt([x.strip() for x in args.inputs.split(',') if x.strip()], args.output_csv, args.output_md)
    print(f'WROTE: {args.output_csv}')
    print(f'rows: {len(out)} | fixtures: {out["fixture_key"].nunique() if not out.empty else 0}')
