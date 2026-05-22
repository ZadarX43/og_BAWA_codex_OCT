from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MATCHUP_CONFIG = {
    'ADVANCED_8S_VS_HOLDING_MID': {
        'slug': 'dm_screen',
        'title': 'DM Screen Weekend Deploy Sheet',
    },
    'WINGER_VS_FULLBACK_ISOLATION': {
        'slug': 'winger_isolation',
        'title': 'Winger Isolation Weekend Deploy Sheet',
    },
}


def _write_md(df: pd.DataFrame, path: Path, title: str, matchup_tag: str) -> None:
    lines = [f'# {title}', '', f'- matchup_tag: `{matchup_tag}`', f"- fixtures: {df['fixture_key'].nunique() if not df.empty else 0} | rows: {len(df)}", '']
    if df.empty:
        lines.append('No rows matched.')
        path.write_text('\n'.join(lines) + '\n')
        return
    for (fixture_key, team_name), sub in df.groupby(['fixture_key', 'team_name'], sort=False):
        first = sub.iloc[0]
        lines.append(f'## {fixture_key} | {team_name}')
        lines.append(f"- {first['home_team_name']} vs {first['away_team_name']} | cascade_strength={first['cascade_strength']:.1f}")
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['player_name']} | {row['market']} | {row['tactical_role']} | sample={row['sample_bucket']} | market_hit={row['market_hit_rate']:.3f} | role_hit={row['role_hit_rate']:.3f}"
            )
            lines.append(f"  opponent_context={row['opponent_flank_profile']} | {row['opponent_role_context_note']}")
            lines.append(f"  matchup_note={row['player_vs_player_matchup_note']}")
            if str(row.get("opponent_striker_profile", "UNSET")) != "UNSET":
                lines.append(
                    f"  striker_profile={row['opponent_striker_profile']} | pressure_tag={row.get('opponent_striker_pressure_tag','UNSET')} | cb_duel_pressure={float(row.get('cb_duel_pressure_score', 0.0)):.3f}"
                )
            if 'manual_side_override_active' in row.index and int(row['manual_side_override_active']) == 1:
                lines.append(f"  manual_override=YES | pitch_side={row.get('manual_pitch_side','UNSET')} | overload_target={row.get('manual_overload_target_side','UNSET')}")
        lines.append('')
    path.write_text('\n'.join(lines) + '\n')


def build_sheets(input_csv: str, output_dir: str) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(input_csv, low_memory=False)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, pd.DataFrame] = {}

    for matchup_tag, cfg in MATCHUP_CONFIG.items():
        sub = df[df['player_vs_player_matchup_tag'].astype(str).eq(matchup_tag)].copy()
        if not sub.empty:
            sub = sub.sort_values(['cascade_strength', 'fixture_key', 'team_name', 'team_specific_priority'], ascending=[False, True, True, False]).reset_index(drop=True)
        csv_path = out_dir / f"matchup_weekend_deploy__{cfg['slug']}.csv"
        md_path = out_dir / f"matchup_weekend_deploy__{cfg['slug']}.md"
        sub.to_csv(csv_path, index=False)
        _write_md(sub, md_path, cfg['title'], matchup_tag)
        results[matchup_tag] = sub
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Split the team-specific weekend shortlist into matchup-specific deploy sheets.')
    parser.add_argument('--input-csv', required=True)
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    results = build_sheets(args.input_csv, args.output_dir)
    for matchup_tag, df in results.items():
        print(f"WROTE: {matchup_tag} | rows={len(df)} | fixtures={df['fixture_key'].nunique() if not df.empty else 0}")
