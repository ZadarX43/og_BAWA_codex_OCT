from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def render_shortlist(board_csv: str, output_md: str, output_csv: str) -> pd.DataFrame:
    board = pd.read_csv(board_csv)
    rows = []
    for fixture_key, group in board.groupby('fixture_key', sort=False):
        goal = group[group['market'].eq('goal')].sort_values('market_score', ascending=False).head(1)
        sot = group[group['market'].eq('shots_on_target')].sort_values('market_score', ascending=False).head(1)
        first = group.iloc[0]
        rows.append({
            'fixture_key': fixture_key,
            'league': first['league'],
            'home_team_name': first['home_team_name'],
            'away_team_name': first['away_team_name'],
            'fixture_attacking_style_label': first['fixture_attacking_style_label'],
            'fixture_attack_quality_score': float(first['fixture_attack_quality_score']),
            'og_goal_environment_label': first['og_goal_environment_label'],
            'og_battle_on_score': float(first['og_battle_on_score']),
            'goal_player': goal['player_name'].iloc[0] if not goal.empty else '',
            'goal_score': float(goal['market_score'].iloc[0]) if not goal.empty else 0.0,
            'sot_player': sot['player_name'].iloc[0] if not sot.empty else '',
            'sot_score': float(sot['market_score'].iloc[0]) if not sot.empty else 0.0,
            'same_player_dual_trigger_flag': int(first['same_player_dual_trigger_flag']) if 'same_player_dual_trigger_flag' in first else 0,
            'combo_reason_bucket': first.get('combo_reason_bucket', ''),
            'fixture_attack_reason_codes': first['fixture_attack_reason_codes'],
        })
    out = pd.DataFrame(rows).sort_values(['fixture_attack_quality_score','og_battle_on_score','fixture_key'], ascending=[False,False,True]) if rows else pd.DataFrame()
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    lines = ['# Goal + SOT Weekend Shortlist', '']
    if out.empty:
        lines.append('- no fixtures')
    else:
        for row in out.itertuples(index=False):
            dual = 'YES' if int(row.same_player_dual_trigger_flag) == 1 else 'NO'
            lines.extend([
                f"## {row.home_team_name} vs {row.away_team_name}",
                f"- fixture: `{row.fixture_key}`",
                f"- league: {row.league}",
                f"- attack style: {row.fixture_attacking_style_label} | quality={row.fixture_attack_quality_score:.3f} | battle_on={row.og_battle_on_score:.3f} | goal_env={row.og_goal_environment_label}",
                f"- goal pick: {row.goal_player} ({row.goal_score:.1f})",
                f"- SOT pick: {row.sot_player} ({row.sot_score:.1f})",
                f"- same-player dual trigger: {dual}",
                f"- bucket: {row.combo_reason_bucket}",
                f"- reasons: {row.fixture_attack_reason_codes}",
                '',
            ])
    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md).write_text('\n'.join(lines))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Render a weekend-ready shortlist from a goal + SOT combo board.')
    parser.add_argument('--board-csv', required=True)
    parser.add_argument('--output-md', required=True)
    parser.add_argument('--output-csv', required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = render_shortlist(args.board_csv, args.output_md, args.output_csv)
    print(f'WROTE: {args.output_md}')
    print(f'fixtures: {len(out)}')


if __name__ == '__main__':
    main()
