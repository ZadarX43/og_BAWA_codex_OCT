from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROLE_LANE_LABELS = {
    'Holding midfielder': 'DM screen',
    'Wide defender / wing-back': 'Wide-defender pressure',
    'Centre-back enforcer': 'Centre-back duel',
    'Wide forward': 'Wide-forward attack',
    'Wide midfielder / winger': 'Wide-winger attack',
    'Central striker': 'Central striker attack',
    'Central midfielder': 'Central-mid support',
}


def lane_label(role: str) -> str:
    return ROLE_LANE_LABELS.get(str(role), str(role))


def classify_lane(row: pd.Series) -> str:
    rows = float(row['rows'])
    fixtures = float(row['fixtures'])
    hit_rate = float(row['observed_hit_rate'])
    market_count = float(row['market_count'])
    live_rows = float(row['live_shortlist_rows'])
    patch_score = float(row['patch_priority_score'])
    lane = str(row['lane_name'])

    if rows >= 15 and fixtures >= 8 and hit_rate >= 0.38 and market_count >= 2:
        return 'COMPLETE_ENOUGH_CORE'
    if rows >= 8 and (hit_rate >= 0.30 or live_rows >= 2 or patch_score >= 8.0):
        return 'BUILDING_WITH_LIVE_SIGNAL'
    if lane == 'Centre-back duel' and rows >= 4:
        return 'SPECIALIST_WATCHLIST'
    if rows >= 4:
        return 'EARLY_BETA'
    return 'THIN_SAMPLE'


def lane_note(row: pd.Series) -> str:
    lane = str(row['lane_name'])
    review_family = str(row['review_family'])
    markets = str(row['markets'])
    tuning = str(row['tuning_signals']) if pd.notna(row['tuning_signals']) else ''
    cb_profiles = str(row['cb_profiles']) if pd.notna(row['cb_profiles']) else ''

    if lane == 'DM screen' and review_family == '4231v442':
        return 'Most mature contact lane so far; keep as the benchmark family while we refine fouls-vs-tackles threshold posture.'
    if lane == 'Wide-defender pressure' and review_family == '4231v433':
        return 'Live flank-isolation lane is real; main remaining job is to separate strong tackles pressure from noisier bookings carryover.'
    if lane == 'Wide-forward attack' and review_family == '3421v4231':
        return 'Attack lane is live enough to keep, but shots and shots_on_target still want cleaner gate tuning before we call it stable.'
    if lane == 'Centre-back duel':
        if cb_profiles:
            return f'CB duel lane is real but still specialist-sized; strongest subtype evidence currently sits in {cb_profiles}.'
        return 'CB duel lane has football logic and some live rows, but still needs subtype accumulation before it graduates from specialist watch.'
    if 'yellow_cards' in markets:
        return 'Bookings signal still reads as beta-supporting evidence rather than a fully trusted standalone lane.'
    if 'LOWER_SCORE_GATE' in tuning or 'RAISE_SCORE_GATE' in tuning:
        return 'Lane is tactically coherent, but threshold posture is still doing too much of the work.'
    return 'Lane stays in the pool, but it still needs more sample depth or cleaner threshold behaviour before we mark it complete.'


def build(
    runner_csv: str,
    threshold_csv: str,
    shadow_priority_csv: str,
    team_role_csv: str,
    cb_subtype_csv: str,
    master_sheet_csv: str,
    output_csv: str,
    output_md: str,
) -> pd.DataFrame:
    runner = pd.read_csv(runner_csv, low_memory=False)
    threshold = pd.read_csv(threshold_csv, low_memory=False)
    shadow = pd.read_csv(shadow_priority_csv, low_memory=False)
    team_role = pd.read_csv(team_role_csv, low_memory=False)
    cb_subtype = pd.read_csv(cb_subtype_csv, low_memory=False)
    master = pd.read_csv(master_sheet_csv, low_memory=False)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    runner['observed_success_flag'] = pd.to_numeric(runner['observed_success_flag'], errors='coerce').fillna(0)
    runner['settled_actual_available'] = pd.to_numeric(runner['settled_actual_available'], errors='coerce').fillna(0)
    runner['selection_gate_flag'] = pd.to_numeric(runner['selection_gate_flag'], errors='coerce').fillna(0)
    runner['near_miss_flag'] = pd.to_numeric(runner['near_miss_flag'], errors='coerce').fillna(0)
    runner['missed_correct_flag'] = pd.to_numeric(runner['missed_correct_flag'], errors='coerce').fillna(0)

    grouped = (
        runner.groupby(['tactical_role', 'review_family'], dropna=False)
        .agg(
            rows=('fixture_key', 'size'),
            fixtures=('fixture_key', 'nunique'),
            observed_hit_rate=('observed_success_flag', 'mean'),
            settled_rows=('settled_actual_available', 'sum'),
            selected_rows=('selection_gate_flag', 'sum'),
            near_misses=('near_miss_flag', 'sum'),
            missed_correct=('missed_correct_flag', 'sum'),
        )
        .reset_index()
    )

    market_sets = (
        runner.groupby(['tactical_role', 'review_family'])['market']
        .agg(lambda s: '|'.join(sorted(set(str(v) for v in s if pd.notna(v)))))
        .reset_index(name='markets')
    )
    grouped = grouped.merge(market_sets, on=['tactical_role', 'review_family'], how='left')
    grouped['market_count'] = grouped['markets'].fillna('').map(lambda x: len([v for v in str(x).split('|') if v]))
    grouped['lane_name'] = grouped['tactical_role'].map(lane_label)

    live = master.groupby(['tactical_role', 'source_family'], dropna=False).size().reset_index(name='live_shortlist_rows')
    grouped = grouped.merge(
        live,
        left_on=['tactical_role', 'review_family'],
        right_on=['tactical_role', 'source_family'],
        how='left',
    ).drop(columns=['source_family'])
    grouped['live_shortlist_rows'] = grouped['live_shortlist_rows'].fillna(0).astype(int)

    team_patterns = (
        team_role.groupby(['tactical_role', 'review_family'], dropna=False)
        .agg(
            top_team_patterns=('team_name', 'nunique'),
            best_team_hit_rate=('hit_rate', 'max'),
        )
        .reset_index()
    )
    grouped = grouped.merge(team_patterns, on=['tactical_role', 'review_family'], how='left')
    grouped['top_team_patterns'] = grouped['top_team_patterns'].fillna(0).astype(int)
    grouped['best_team_hit_rate'] = grouped['best_team_hit_rate'].fillna(0.0)

    tuning = (
        threshold.groupby(['review_family'], dropna=False)['tuning_signal']
        .agg(lambda s: '|'.join(sorted(set(str(v) for v in s if pd.notna(v) and str(v)))))
        .reset_index(name='tuning_signals')
    )
    grouped = grouped.merge(tuning, on='review_family', how='left')
    grouped['tuning_signals'] = grouped['tuning_signals'].fillna('')

    patch = (
        shadow.groupby(['review_family'], dropna=False)
        .agg(
            patch_priority_score=('priority_score', 'max'),
            patch_priority_bucket=('priority_bucket', lambda s: '|'.join(sorted(set(str(v) for v in s if pd.notna(v) and str(v))))),
        )
        .reset_index()
    )
    grouped = grouped.merge(patch, on='review_family', how='left')
    grouped['patch_priority_score'] = pd.to_numeric(grouped['patch_priority_score'], errors='coerce').fillna(0.0)
    grouped['patch_priority_bucket'] = grouped['patch_priority_bucket'].fillna('')

    cb_profiles = (
        cb_subtype.groupby('review_family', dropna=False)['opponent_striker_profile']
        .agg(lambda s: '|'.join(sorted(set(str(v) for v in s if pd.notna(v) and str(v)))))
        .reset_index(name='cb_profiles')
    )
    grouped = grouped.merge(cb_profiles, on='review_family', how='left')
    grouped['cb_profiles'] = grouped['cb_profiles'].fillna('')

    grouped['status'] = grouped.apply(classify_lane, axis=1)
    grouped['note'] = grouped.apply(lane_note, axis=1)
    grouped['observed_hit_rate'] = grouped['observed_hit_rate'].round(3)
    grouped['best_team_hit_rate'] = grouped['best_team_hit_rate'].round(3)
    grouped['patch_priority_score'] = grouped['patch_priority_score'].round(3)
    grouped['core_signal_score'] = (
        grouped['observed_hit_rate'] * 40.0
        + grouped['rows'].clip(upper=20) * 1.5
        + grouped['live_shortlist_rows'].clip(upper=6) * 3.0
        + grouped['top_team_patterns'].clip(upper=6) * 2.0
    ).round(2)

    grouped = grouped.sort_values(
        ['status', 'core_signal_score', 'rows', 'observed_hit_rate'],
        ascending=[True, False, False, False],
    )
    grouped.to_csv(output_csv, index=False)

    lines = [
        '# Tactical Signal Completeness Board',
        '',
        '- Compact read of which player-events tactical lanes now look core, which are still building, and which should stay specialist/beta only.',
        '- This is the board I would use to judge whether the tactical layer is close to "complete enough" before the fresh full goal-market rebuild.',
        '',
    ]

    status_order = [
        'COMPLETE_ENOUGH_CORE',
        'BUILDING_WITH_LIVE_SIGNAL',
        'SPECIALIST_WATCHLIST',
        'EARLY_BETA',
        'THIN_SAMPLE',
    ]
    for status in status_order:
        block = grouped[grouped['status'] == status]
        if block.empty:
            continue
        lines.append(f'## {status}')
        for _, row in block.iterrows():
            lines.append(
                f"- {row['lane_name']} | {row['review_family']} | markets={row['markets']} | rows={int(row['rows'])} | fixtures={int(row['fixtures'])} | hit_rate={row['observed_hit_rate']:.3f} | live_rows={int(row['live_shortlist_rows'])}"
            )
            lines.append(f"  note: {row['note']}")
        lines.append('')

    core = grouped[grouped['status'] == 'COMPLETE_ENOUGH_CORE']
    building = grouped[grouped['status'] == 'BUILDING_WITH_LIVE_SIGNAL']
    cb_block = grouped[grouped['lane_name'] == 'Centre-back duel']

    lines.extend([
        '## Completeness Read',
        f"- core lanes now trusted most: `{len(core)}`",
        f"- building lanes with live signal: `{len(building)}`",
        f"- centre-back duel specialist lanes still under accumulation: `{len(cb_block)}`",
        '- current practical read: the DM screen lane is now clearly the strongest core pattern, winger/full-back pressure is real, attacking winger/forward lanes are good enough to keep refining, and the CB duel lane is real but still specialist-sized.',
        '',
    ])

    lines.extend([
        '## Next Frontier',
        '- Keep tightening the DM lane by separating where fouls_committed should stay conservative versus where tackles should admit more survivors.',
        '- Keep the 4-2-3-1 vs 4-3-3 wide-defender pressure lane live, but treat yellow_cards as supporting evidence until the bookings sample thickens more.',
        '- Keep relaxing / retesting the strongest attacking wide-forward lanes, especially shots_on_target in 3-4-2-1 vs 4-2-3-1.',
        '- Keep the centre-back duel lane in specialist-watch mode and accumulate subtype evidence before calling it fully mature.',
        '',
    ])

    Path(output_md).write_text('\n'.join(lines) + '\n')
    return grouped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a compact tactical signal completeness board from the current player-event audits.')
    parser.add_argument('--runner-csv', default='reports/player_events/quality_audits/player_market_walkforward_runner.csv')
    parser.add_argument('--threshold-csv', default='reports/player_events/quality_audits/threshold_tuning_audit.csv')
    parser.add_argument('--shadow-priority-csv', default='reports/player_events/quality_audits/shadow_patch_priority_board.csv')
    parser.add_argument('--team-role-csv', default='reports/player_events/quality_audits/team_family_role_audit.csv')
    parser.add_argument('--cb-subtype-csv', default='reports/player_events/quality_audits/cb_subtype_walkforward_audit.csv')
    parser.add_argument('--master-sheet-csv', default='reports/player_events/combined_boards/master_weekend_specialist_sheet.csv')
    parser.add_argument('--output-csv', default='reports/player_events/quality_audits/tactical_signal_completeness_board.csv')
    parser.add_argument('--output-md', default='reports/player_events/quality_audits/tactical_signal_completeness_board.md')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    out = build(
        runner_csv=args.runner_csv,
        threshold_csv=args.threshold_csv,
        shadow_priority_csv=args.shadow_priority_csv,
        team_role_csv=args.team_role_csv,
        cb_subtype_csv=args.cb_subtype_csv,
        master_sheet_csv=args.master_sheet_csv,
        output_csv=args.output_csv,
        output_md=args.output_md,
    )
    print(f'WROTE: {args.output_csv}')
    print(f'rows: {len(out)}')
