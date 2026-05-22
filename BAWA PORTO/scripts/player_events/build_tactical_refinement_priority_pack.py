from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def dm_action(row: pd.Series) -> str:
    market = str(row['market'])
    family = str(row['review_family'])
    hit_rate = float(row['observed_hit_rate'])
    selected = int(row['selected_rows'])
    near = int(row['near_misses'])
    missed = int(row['missed_correct'])

    if market == 'fouls_committed':
        if family == '4231v442' and hit_rate >= 0.40:
            return 'KEEP_CONSERVATIVE__REVIEW_EDGE_CASES'
        if family in {'4231v433', '3421v4231'}:
            return 'KEEP_TIGHT__DO_NOT_RELAX_YET'
        return 'HOLD'

    if market == 'tackles':
        if family == '4231v442' and (selected >= 10 or near >= 4 or hit_rate >= 0.55):
            return 'ADMIT_MORE_SURVIVORS'
        if family == '4231v433' and (near >= 2 or missed >= 4):
            return 'ADMIT_MORE_SURVIVORS'
        if family == '3421v4231' and missed >= 2:
            return 'SOFT_RELAX_ONLY'
        return 'HOLD'

    return 'SUPPORT_ONLY'


def dm_note(row: pd.Series) -> str:
    market = str(row['market'])
    family = str(row['review_family'])
    hit_rate = float(row['observed_hit_rate'])
    missed = int(row['missed_correct'])
    near = int(row['near_misses'])

    if market == 'fouls_committed' and family == '4231v442':
        return 'This is the best fouls version of the DM lane, but the mix of 10 missed-correct rows and only 2 selected survivors says we should recheck edge cases without opening the gate too widely.'
    if market == 'fouls_committed':
        return 'The DM fouls lane outside 4-2-3-1 vs 4-4-2 is still too weak to loosen; keep it as a secondary confirmation layer.'
    if market == 'tackles' and family == '4231v442':
        return 'This is the strongest live DM contact lane and can carry a slightly looser survivor posture than the fouls version.'
    if market == 'tackles' and family == '4231v433':
        return 'Good hit rate plus near-miss pressure says more tackle survivors should be admitted before we call the lane mature.'
    if market == 'tackles' and family == '3421v4231':
        return 'Still useful, but this version of the DM lane wants only a soft relax until the hit rate improves.'
    if market == 'yellow_cards':
        return 'Bookings remain support evidence only inside the DM lane.'
    return 'Keep reviewing this lane in context rather than promoting it outright.'


def wide_action(row: pd.Series) -> str:
    market = str(row['market'])
    hit_rate = float(row['observed_hit_rate'])
    rows = int(row['rows'])
    if market == 'tackles' and hit_rate >= 0.70:
        return 'KEEP_LIVE_PRIMARY'
    if market == 'yellow_cards':
        return 'SUPPORTING_EVIDENCE_ONLY'
    if market == 'fouls_committed' and rows >= 4:
        return 'OBSERVE_SECONDARY_ONLY'
    return 'HOLD'


def wide_note(row: pd.Series) -> str:
    market = str(row['market'])
    if market == 'tackles':
        return 'This is the real live read in the 4-2-3-1 vs 4-3-3 wide-defender lane; keep it as the headline market.'
    if market == 'yellow_cards':
        return 'Bookings do show tactical logic here, but the sample is still too noisy to let cards lead the lane.'
    if market == 'fouls_committed':
        return 'Useful as side-context, but still not strong enough to carry the lane on its own.'
    return 'Keep as supporting context only.'


def cb_watch_status(row: pd.Series) -> str:
    rows = int(row['rows'])
    market = str(row['market'])
    hit_rate = float(row['avg_market_hit_rate'])
    if rows >= 2 and hit_rate >= 0.50:
        return 'BUILDING_SUBTYPE'
    if market == 'fouls_committed' and hit_rate >= 1.0:
        return 'EARLY_POSITIVE_SIGNAL'
    return 'WATCH_ONLY'


def cb_note(row: pd.Series) -> str:
    profile = str(row['opponent_striker_profile'])
    market = str(row['market'])
    hit_rate = float(row['avg_market_hit_rate'])
    if profile == 'DIRECT_TARGET_STRIKER':
        return 'This remains the clearest early CB subtype and should stay the main accumulation target.'
    if profile == 'AERIAL_BOX_NINE':
        return 'Aerial-box-nine still looks intuitive, but it needs more than a single fixture before we trust it operationally.'
    if profile == 'MOBILE_PRESSING_9' and market == 'tackles' and hit_rate == 0.0:
        return 'This subtype is live enough to track, but the tackle miss is exactly why the whole CB lane should stay specialist-watch for now.'
    if profile == 'MOBILE_PRESSING_9':
        return 'This subtype has some promise, but it still needs accumulation across both markets.'
    return 'Keep accumulating subtype evidence before promoting this lane.'


def build(
    runner_csv: str,
    cb_subtype_csv: str,
    output_csv: str,
    output_md: str,
) -> pd.DataFrame:
    runner = pd.read_csv(runner_csv, low_memory=False)
    cb = pd.read_csv(cb_subtype_csv, low_memory=False)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    for col in ['observed_success_flag', 'selection_gate_flag', 'near_miss_flag', 'missed_correct_flag', 'expected_hit_rate_3y']:
        runner[col] = pd.to_numeric(runner[col], errors='coerce').fillna(0)

    dm = runner[(runner['tactical_role'] == 'Holding midfielder') & (runner['market'].isin(['fouls_committed', 'tackles', 'yellow_cards']))]
    dm = (
        dm.groupby(['review_family', 'market'], dropna=False)
        .agg(
            rows=('fixture_key', 'size'),
            fixtures=('fixture_key', 'nunique'),
            observed_hit_rate=('observed_success_flag', 'mean'),
            expected_hit_rate_3y=('expected_hit_rate_3y', 'mean'),
            selected_rows=('selection_gate_flag', 'sum'),
            near_misses=('near_miss_flag', 'sum'),
            missed_correct=('missed_correct_flag', 'sum'),
        )
        .reset_index()
    )
    dm['lane'] = 'DM screen'
    dm['priority_action'] = dm.apply(dm_action, axis=1)
    dm['note'] = dm.apply(dm_note, axis=1)
    dm['segment'] = 'DM_LANE_SPLIT'

    wide = runner[(runner['tactical_role'] == 'Wide defender / wing-back') & (runner['review_family'] == '4231v433')]
    wide = (
        wide.groupby(['review_family', 'market'], dropna=False)
        .agg(
            rows=('fixture_key', 'size'),
            fixtures=('fixture_key', 'nunique'),
            observed_hit_rate=('observed_success_flag', 'mean'),
            expected_hit_rate_3y=('expected_hit_rate_3y', 'mean'),
            selected_rows=('selection_gate_flag', 'sum'),
            missed_correct=('missed_correct_flag', 'sum'),
        )
        .reset_index()
    )
    wide['lane'] = 'Wide-defender pressure'
    wide['priority_action'] = wide.apply(wide_action, axis=1)
    wide['note'] = wide.apply(wide_note, axis=1)
    wide['segment'] = 'WIDE_4231V433_SUPPORT_POSTURE'
    wide['near_misses'] = 0

    cb = cb.copy()
    cb['lane'] = 'Centre-back duel'
    cb['priority_action'] = cb.apply(cb_watch_status, axis=1)
    cb['note'] = cb.apply(cb_note, axis=1)
    cb['segment'] = 'CB_SUBTYPE_WATCH'
    cb = cb.rename(columns={'avg_market_hit_rate': 'observed_hit_rate'})
    cb['expected_hit_rate_3y'] = 0.0
    cb['selected_rows'] = 0
    cb['near_misses'] = 0
    cb['missed_correct'] = 0
    cb['rows'] = pd.to_numeric(cb['rows'], errors='coerce').fillna(0).astype(int)
    cb['fixtures'] = pd.to_numeric(cb['fixtures'], errors='coerce').fillna(0).astype(int)
    cb_detail = cb.copy()

    common_cols = ['segment', 'lane', 'review_family', 'market', 'rows', 'fixtures', 'observed_hit_rate', 'expected_hit_rate_3y', 'selected_rows', 'near_misses', 'missed_correct', 'priority_action', 'note']
    dm = dm.reindex(columns=common_cols)
    wide = wide.reindex(columns=common_cols)
    cb = cb.reindex(columns=common_cols)
    out = pd.concat([dm, wide, cb], ignore_index=True)
    out['observed_hit_rate'] = pd.to_numeric(out['observed_hit_rate'], errors='coerce').round(3)
    out['expected_hit_rate_3y'] = pd.to_numeric(out['expected_hit_rate_3y'], errors='coerce').round(3)
    out.to_csv(output_csv, index=False)

    lines = [
        '# Tactical Refinement Priority Pack',
        '',
        '- Research-only refinement board for the next tactical-layer pass.',
        '- Goal: tighten the DM split, keep the 4-2-3-1 vs 4-3-3 wide-defender lane live without over-promoting cards, and keep the CB duel lane specialist-only while subtype evidence accumulates.',
        '',
        '## DM Lane Split',
    ]
    for _, row in dm.sort_values(['review_family', 'market']).iterrows():
        lines.append(
            f"- {row['review_family']} | {row['market']} | rows={int(row['rows'])} | fixtures={int(row['fixtures'])} | hit_rate={row['observed_hit_rate']:.3f} | selected={int(row['selected_rows'])} | near={int(row['near_misses'])} | missed_correct={int(row['missed_correct'])} | action={row['priority_action']}"
        )
        lines.append(f"  note: {row['note']}")
    lines.extend(['', '## 4231v433 Wide-Defender Posture'])
    for _, row in wide.sort_values('market').iterrows():
        lines.append(
            f"- {row['market']} | rows={int(row['rows'])} | fixtures={int(row['fixtures'])} | hit_rate={row['observed_hit_rate']:.3f} | selected={int(row['selected_rows'])} | missed_correct={int(row['missed_correct'])} | action={row['priority_action']}"
        )
        lines.append(f"  note: {row['note']}")
    lines.extend(['', '## CB Subtype Watch'])
    for _, row in cb_detail.sort_values(['review_family', 'opponent_striker_profile', 'market']).iterrows():
        lines.append(
            f"- {row['review_family']} | {row['opponent_striker_profile']} | {row['market']} | rows={int(row['rows'])} | fixtures={int(row['fixtures'])} | hit_rate={float(row['observed_hit_rate']):.3f} | action={row['priority_action']}"
        )
        lines.append(f"  note: {row['note']}")
    lines.extend([
        '',
        '## Current Tactical Read',
        '- DM screen is the main core lane, but it now clearly wants separate posture for fouls versus tackles rather than one shared contact story.',
        '- The 4-2-3-1 vs 4-3-3 wide-defender lane should stay live through tackles first; yellow_cards should stay supporting-only until the sample is thicker.',
        '- The centre-back duel lane is worth keeping, but it is still specialist-sized and should stay in subtype watch mode rather than being promoted to a core system lane.',
        '',
    ])
    Path(output_md).write_text('\n'.join(lines) + '\n')
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a small tactical refinement pack for DM split, wide-defender posture, and CB subtype watch.')
    parser.add_argument('--runner-csv', default='reports/player_events/quality_audits/player_market_walkforward_runner.csv')
    parser.add_argument('--cb-subtype-csv', default='reports/player_events/quality_audits/cb_subtype_walkforward_audit.csv')
    parser.add_argument('--output-csv', default='reports/player_events/quality_audits/tactical_refinement_priority_pack.csv')
    parser.add_argument('--output-md', default='reports/player_events/quality_audits/tactical_refinement_priority_pack.md')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    out = build(args.runner_csv, args.cb_subtype_csv, args.output_csv, args.output_md)
    print(f'WROTE: {args.output_csv}')
    print(f'rows: {len(out)}')
