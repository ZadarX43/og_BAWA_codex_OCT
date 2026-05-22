from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def status_line(done: bool) -> str:
    return 'GREEN' if done else 'AMBER'


def build(
    completeness_csv: str,
    dm_note_csv: str,
    wf_note_csv: str,
    refinement_csv: str,
    output_md: str,
) -> None:
    completeness = pd.read_csv(completeness_csv, low_memory=False)
    dm = pd.read_csv(dm_note_csv, low_memory=False)
    wf = pd.read_csv(wf_note_csv, low_memory=False)
    refinement = pd.read_csv(refinement_csv, low_memory=False)

    core_count = int((completeness['status'] == 'COMPLETE_ENOUGH_CORE').sum())
    building_count = int((completeness['status'] == 'BUILDING_WITH_LIVE_SIGNAL').sum())
    specialist_count = int((completeness['status'] == 'SPECIALIST_WATCHLIST').sum())

    dm_ready = int(dm['priority_action'].isin(['RELAX_RESEARCH_SHADOW', 'SOFT_RELAX_RESEARCH_ONLY', 'KEEP_CONSERVATIVE__EDGE_REVIEW_ONLY']).sum()) >= 4
    wf_ready = int(wf['priority_action'].isin(['KEEP_PRIMARY__RELAX_CAREFULLY', 'KEEP_PRIMARY__DO_NOT_OVER-TIGHTEN', 'KEEP_SECONDARY__NEEDS_MORE_FINISHING_PROOF', 'SECONDARY_ATTACK_SUPPORT']).sum()) >= 4
    cb_watch_only = int((refinement['segment'] == 'CB_SUBTYPE_WATCH').sum()) >= 6
    cb_multi_fixture = False

    building_lanes = completeness[completeness['status'] == 'BUILDING_WITH_LIVE_SIGNAL']
    attack_building = int(building_lanes['lane_name'].isin(['Wide-forward attack', 'Wide-winger attack']).sum())

    lines = [
        '# Tactical Layer Completion Checklist',
        '',
        '- Purpose: define what still has to happen before we call the player-events tactical layer fully complete enough to hand into the fresh full goal-market 3-year rebuild.',
        '- Standard: we want enough durable feature evidence that the full estate audit can score actuals, survive parameter tuning, and support later readjustment without leaning on guesswork.',
        '',
        '## Core Structure',
        f"- [{status_line(core_count >= 3)}] At least three lanes are clearly core. Current: `{core_count}`.",
        f"- [{status_line(building_count >= 4)}] Building lane set is large enough to justify continued refinement rather than broad lane discovery. Current: `{building_count}`.",
        f"- [{status_line(specialist_count >= 3)}] Specialist-watch lanes are isolated rather than being mixed into the core layer. Current: `{specialist_count}`.",
        '',
        '## DM Lane',
        f"- [{status_line(dm_ready)}] DM threshold posture is explicitly split by family and market rather than treated as one generic contact lane.",
        '- Needed state: `4231v442 fouls` stays conservative while `4231v442 tackles`, `4231v433 tackles`, and `3421v4231 tackles` each have an explicit research stance.',
        '',
        '## Attack Lane',
        f"- [{status_line(wf_ready)}] Wide-forward attack now has an explicit primary-vs-secondary market hierarchy in the strongest families.",
        f"- [{status_line(attack_building >= 4)}] Enough attack-side building lanes remain live to justify later tuning from real actuals rather than one-off anecdotes. Current attack-side building lanes: `{attack_building}`.",
        '- Needed state: `shots_on_target` leads the wide-forward lanes; raw `shots` stays secondary/support where the evidence says so.',
        '',
        '## CB Specialist Watch',
        f"- [{status_line(cb_watch_only)}] CB subtype watch is explicitly isolated and not promoted into the core tactical layer.",
        f"- [{status_line(cb_multi_fixture)}] CB subtype evidence has multi-fixture depth before any promotion. Current state: `not yet green`.",
        '- Needed state: keep accumulating subtype rows in the background without changing status until at least one subtype has repeat multi-fixture support.',
        '',
        '## Audit Survivability',
        '- [AMBER] Before the full estate audit, keep adding actual-result rows into the live tactical families so later parameter readjustment is based on feature survival rather than a small protected sample.',
        '- [AMBER] Keep preserving outputs as auditable artifacts so threshold tuning can be traced back to actual historical behaviour.',
        '',
        '## Pivot Rule',
        '- Treat the tactical layer as ready to hand into the fresh full goal-market rebuild once everything above is green except the deliberate CB multi-fixture watch item.',
        '- That means we can pivot with CB still in specialist watch mode, as long as it is clearly ring-fenced and not presented as a finished core lane.',
        '',
        '## Current Call',
        '- We are close, but not fully complete yet.',
        '- The remaining work is mostly accumulation and refinement, not architecture discovery.',
        '- That is a good sign for the later full estate audit, because it means the system is starting to become tuneable rather than just exploratory.',
        '',
    ]

    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md).write_text('\n'.join(lines) + '\n')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a tiny tactical layer completion checklist.')
    parser.add_argument('--completeness-csv', default='reports/player_events/quality_audits/tactical_signal_completeness_board.csv')
    parser.add_argument('--dm-note-csv', default='reports/player_events/quality_audits/dm_threshold_note.csv')
    parser.add_argument('--wf-note-csv', default='reports/player_events/quality_audits/wide_forward_threshold_note.csv')
    parser.add_argument('--refinement-csv', default='reports/player_events/quality_audits/tactical_refinement_priority_pack.csv')
    parser.add_argument('--output-md', default='reports/player_events/quality_audits/TACTICAL_LAYER_COMPLETION_CHECKLIST.md')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    build(args.completeness_csv, args.dm_note_csv, args.wf_note_csv, args.refinement_csv, args.output_md)
    print(f'WROTE: {args.output_md}')
