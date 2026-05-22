from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


STATUS_ORDER = [
    'COMPLETE_ENOUGH_CORE',
    'BUILDING_WITH_LIVE_SIGNAL',
    'SPECIALIST_WATCHLIST',
]


def build(completeness_csv: str, dm_note_csv: str, wf_note_csv: str, refinement_csv: str, output_md: str) -> None:
    completeness = pd.read_csv(completeness_csv, low_memory=False)
    dm = pd.read_csv(dm_note_csv, low_memory=False)
    wf = pd.read_csv(wf_note_csv, low_memory=False)
    refinement = pd.read_csv(refinement_csv, low_memory=False)

    lines = [
        '# Tactical Layer Release Candidate Note',
        '',
        '- Compact release-candidate read of the current player-events tactical layer.',
        '- Scope: identify which lanes are now core, which are still building, and which must remain specialist-watch before the fresh full goal-market rebuild.',
        '',
    ]

    for status in STATUS_ORDER:
        block = completeness[completeness['status'] == status].copy()
        if block.empty:
            continue
        title = {
            'COMPLETE_ENOUGH_CORE': 'Core Lanes',
            'BUILDING_WITH_LIVE_SIGNAL': 'Building Lanes',
            'SPECIALIST_WATCHLIST': 'Specialist-Watch Lanes',
        }[status]
        lines.append(f'## {title}')
        for _, row in block.sort_values(['lane_name', 'review_family']).iterrows():
            lines.append(
                f"- {row['lane_name']} | {row['review_family']} | markets={row['markets']} | rows={int(row['rows'])} | fixtures={int(row['fixtures'])} | hit_rate={float(row['observed_hit_rate']):.3f}"
            )
            lines.append(f"  note: {row['note']}")
        lines.append('')

    lines.append('## Threshold Posture Snapshot')
    for _, row in dm.sort_values(['review_family', 'market']).iterrows():
        lines.append(f"- DM screen | {row['review_family']} | {row['market']} | action={row['priority_action']}")
    for _, row in wf.sort_values(['review_family', 'market']).iterrows():
        lines.append(f"- Wide-forward attack | {row['review_family']} | {row['market']} | action={row['priority_action']}")
    lines.append('')

    cb_watch = refinement[refinement['segment'] == 'CB_SUBTYPE_WATCH'].copy()
    lines.append('## CB Watch Position')
    lines.append('- Keep accumulating centre-back subtype evidence in the background without changing its status.')
    for _, row in cb_watch.sort_values(['review_family', 'market']).iterrows():
        lines.append(
            f"- {row['review_family']} | {row['market']} | action={row['priority_action']}"
        )
    lines.append('')

    core_count = int((completeness['status'] == 'COMPLETE_ENOUGH_CORE').sum())
    build_count = int((completeness['status'] == 'BUILDING_WITH_LIVE_SIGNAL').sum())
    cb_count = int((completeness['status'] == 'SPECIALIST_WATCHLIST').sum())
    lines.extend([
        '## Release Candidate Read',
        f'- core lanes: `{core_count}`',
        f'- building lanes: `{build_count}`',
        f'- specialist-watch lanes: `{cb_count}`',
        '- current call: the tactical layer is close enough to be treated as a serious release candidate, but not fully locked until the attack-side refinements and CB subtype accumulation mature a little further.',
        '',
    ])

    lines.extend([
        '## What Still Needs More Time',
        '- Keep the DM split disciplined: fouls tighter, tackles looser where the evidence supports it.',
        '- Keep wide-forward shots_on_target as the main attack refinement frontier.',
        '- Keep wide-winger variants in beta/watch mode rather than promoting them into the core layer too soon.',
        '- Keep the CB duel lane specialist-only until subtype evidence reaches multi-fixture depth.',
        '',
    ])

    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md).write_text('\n'.join(lines) + '\n')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a tactical layer release candidate note.')
    parser.add_argument('--completeness-csv', default='reports/player_events/quality_audits/tactical_signal_completeness_board.csv')
    parser.add_argument('--dm-note-csv', default='reports/player_events/quality_audits/dm_threshold_note.csv')
    parser.add_argument('--wf-note-csv', default='reports/player_events/quality_audits/wide_forward_threshold_note.csv')
    parser.add_argument('--refinement-csv', default='reports/player_events/quality_audits/tactical_refinement_priority_pack.csv')
    parser.add_argument('--output-md', default='reports/player_events/quality_audits/TACTICAL_LAYER_RELEASE_CANDIDATE.md')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    build(args.completeness_csv, args.dm_note_csv, args.wf_note_csv, args.refinement_csv, args.output_md)
    print(f'WROTE: {args.output_md}')
