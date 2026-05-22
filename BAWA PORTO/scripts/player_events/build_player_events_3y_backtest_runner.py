from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from build_applied_threshold_patch_proposal import build as build_patch_proposal
from build_greenlist_historical_actuals_layer import build_layer as build_actuals_layer
from build_player_events_historical_coverage_map import build as build_coverage_map
from build_numeric_threshold_scorecut_table import build_table as build_scorecuts
from build_player_market_miss_audit import build_audit as build_miss_audit
from build_player_market_walkforward_runner import build_runner
from build_shadow_patch_priority_board import build as build_shadow_priority
from build_shadow_threshold_trial_table import build as build_shadow_trials
from build_team_league_comp_threshold_map import build_map as build_team_map
from build_threshold_tuning_audit import build_audit as build_threshold_audit

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MASTER = REPO_ROOT / 'reports' / 'player_events' / 'combined_boards' / 'master_weekend_specialist_sheet.csv'
DEFAULT_BOOKINGS = REPO_ROOT / 'reports' / 'player_events' / 'combined_boards' / 'bookings_super_elite_weekend_sheet.csv'
DEFAULT_TEAM = REPO_ROOT / 'reports' / 'player_events' / 'combined_boards' / 'team_specific_weekend_sheet.csv'
DEFAULT_OUTROOT = REPO_ROOT / 'reports' / 'player_events' / 'backtests'
DEFAULT_ACTUALS_STATS = REPO_ROOT / 'data_sources' / 'api_football' / 'normalized' / 'match_player_stats__GREENLIST_FULL_3Y__2022_2024.csv'
DEFAULT_ACTUALS_FIXTURES = REPO_ROOT / 'data_sources' / 'api_football' / 'normalized' / 'fixtures_master__GREENLIST_FULL_3Y__2022_2024.csv'
DEFAULT_COVERAGE_CSV = REPO_ROOT / 'reports' / 'player_events' / 'quality_audits' / 'greenlist_historical_actuals_coverage.csv'
DEFAULT_COVERAGE_MD = REPO_ROOT / 'reports' / 'player_events' / 'quality_audits' / 'greenlist_historical_actuals_coverage.md'
DEFAULT_COVERAGE_MAP_MD = REPO_ROOT / 'reports' / 'player_events' / 'quality_audits' / 'PLAYER_EVENTS_HISTORICAL_COVERAGE_MAP.md'
DEFAULT_OVERRIDES_CSV = REPO_ROOT / 'reports' / 'player_events' / 'quality_audits' / 'player_events_research_threshold_overrides.csv'
DEFAULT_HIT_THRESHOLD_OVERRIDES_CSV = REPO_ROOT / 'reports' / 'player_events' / 'quality_audits' / 'player_events_fallback_hit_threshold_overrides.csv'
DEFAULT_MARKET_HISTORY_GATE_CSV = REPO_ROOT / 'reports' / 'player_events' / 'quality_audits' / 'player_events_market_history_gate.csv'


def summarize(
    runner_csv: Path,
    miss_csv: Path,
    threshold_csv: Path,
    patch_csv: Path,
    shadow_csv: Path,
    priority_csv: Path,
    team_map_csv: Path,
    coverage_csv: Path,
    overrides_csv: Path,
    hit_threshold_overrides_csv: Path,
    market_history_gate_csv: Path,
    output_md: Path,
) -> None:
    runner = pd.read_csv(runner_csv, low_memory=False)
    miss = pd.read_csv(miss_csv, low_memory=False) if miss_csv.exists() else pd.DataFrame()
    threshold = pd.read_csv(threshold_csv, low_memory=False) if threshold_csv.exists() else pd.DataFrame()
    patch = pd.read_csv(patch_csv, low_memory=False) if patch_csv.exists() else pd.DataFrame()
    shadow = pd.read_csv(shadow_csv, low_memory=False) if shadow_csv.exists() else pd.DataFrame()
    priority = pd.read_csv(priority_csv, low_memory=False) if priority_csv.exists() else pd.DataFrame()
    team_map = pd.read_csv(team_map_csv, low_memory=False) if team_map_csv.exists() else pd.DataFrame()
    coverage = pd.read_csv(coverage_csv, low_memory=False) if coverage_csv.exists() else pd.DataFrame()
    overrides = pd.read_csv(overrides_csv, low_memory=False) if overrides_csv.exists() else pd.DataFrame()
    hit_threshold_overrides = pd.read_csv(hit_threshold_overrides_csv, low_memory=False) if hit_threshold_overrides_csv.exists() else pd.DataFrame()
    market_history_gate = pd.read_csv(market_history_gate_csv, low_memory=False) if market_history_gate_csv.exists() else pd.DataFrame()

    lines = [
        '# Player Events 3Y Backtest Pack',
        '',
        '- Dedicated player-events backtest pack over the current research estate.',
        '- Purpose: prove and tune the player-events system itself before or alongside broader goal-market estate work.',
        '- Important: this pack scores the current player-events candidate estate against settled actuals; it does not depend on the old goal-market-only walkforward artifacts.',
        '',
    ]

    if not coverage.empty:
        available = int((coverage['coverage_flag'] == 'AVAILABLE').sum())
        total = int(len(coverage))
        leagues = int(coverage['league_tag'].nunique())
        lines.extend([
            '## Historical Actuals Coverage',
            f"- greenlist leagues tracked: `{leagues}`",
            f"- season coverage cells available: `{available}/{total}`",
            '- Coverage comes from the local normalized player-stat archive combined into a single joinable greenlist layer.',
            '',
        ])

    if not overrides.empty:
        lines.extend([
            '## Research Overrides',
            f"- active research-only score-cut overrides: `{len(overrides)}`",
            '- These are applied only inside the research backtest runner, not live deploy logic.',
            '',
        ])

    if not hit_threshold_overrides.empty:
        lines.extend([
            '## Fallback Hit-Threshold Overrides',
            f"- active research-only hit-threshold overrides: `{len(hit_threshold_overrides)}`",
            '- These only affect fallback-source testing inside the research runner.',
            '',
        ])

    if not market_history_gate.empty:
        lines.extend([
            '## Market History Gate',
            f"- active research-only market-history gate markets: `{len(market_history_gate)}`",
            '- These require prior market-specific player evidence before a research selection can pass.',
            '',
        ])

    if not runner.empty:
        settled_rows = int(pd.to_numeric(runner.get('settled_actual_available', 0), errors='coerce').fillna(0).sum())
        lines.extend([
            '## Snapshot',
            f"- rows: `{len(runner)}`",
            f"- fixtures: `{runner['fixture_key'].nunique()}`",
            f"- markets: `{runner['market'].astype(str).nunique()}`",
            f"- settled rows: `{settled_rows}`",
            '',
        ])

        summary = (
            runner.groupby(['market', 'review_family'], dropna=False)
            .agg(
                rows=('fixture_key', 'size'),
                fixtures=('fixture_key', pd.Series.nunique),
                observed_hit=('observed_success_flag', lambda s: pd.to_numeric(s, errors='coerce').mean()),
                expected_hit=('expected_hit_rate_3y', lambda s: pd.to_numeric(s, errors='coerce').mean()),
                near_misses=('near_miss_flag', 'sum'),
                missed_correct=('missed_correct_flag', 'sum'),
            )
            .reset_index()
            .sort_values(['observed_hit', 'rows'], ascending=[False, False])
        )
        lines.append('## Best Market / Family Reads')
        for _, row in summary.head(10).iterrows():
            lines.append(
                f"- {row['market']} | {row['review_family']} | rows={int(row['rows'])} | fixtures={int(row['fixtures'])} | observed_hit={row['observed_hit']:.3f} | expected_hit={row['expected_hit']:.3f} | near_misses={int(row['near_misses'])} | missed_correct={int(row['missed_correct'])}"
            )
        lines.append('')

    if not miss.empty:
        lines.append('## Miss Pressure')
        miss_summary = (
            miss.groupby('audit_label', dropna=False)
            .size()
            .reset_index(name='rows')
            .sort_values('rows', ascending=False)
        )
        for _, row in miss_summary.iterrows():
            lines.append(f"- {row['audit_label']}: `{int(row['rows'])}` rows")
        lines.append('')

    if not threshold.empty:
        lines.append('## Threshold Tuning Signals')
        thresh_summary = threshold.groupby('tuning_signal', dropna=False).size().reset_index(name='rows')
        for _, row in thresh_summary.iterrows():
            lines.append(f"- {row['tuning_signal']}: `{int(row['rows'])}` cohorts")
        lines.append('')

    if not patch.empty:
        lines.append('## Patch Posture')
        patch_summary = patch.groupby('patch_confidence', dropna=False).size().reset_index(name='rows')
        for _, row in patch_summary.iterrows():
            lines.append(f"- {row['patch_confidence']}: `{int(row['rows'])}` cohorts")
        lines.append('')

    if not priority.empty:
        lines.append('## Shadow Trial Highlights')
        priority = priority.sort_values(['newly_admitted_hits', 'net_hit_gain'], ascending=[False, False])
        for _, row in priority.head(6).iterrows():
            lines.append(
                f"- {row['market']} | {row['review_family']} | risk={row['prematch_risk_focus']} | admitted_hits={int(row['newly_admitted_hits'])} | net_hit_gain={int(row['net_hit_gain'])}"
            )
        lines.append('')

    if not team_map.empty:
        lines.append('## Team / League / Competition Map')
        team_summary = team_map.groupby('threshold_posture', dropna=False).size().reset_index(name='rows')
        for _, row in team_summary.iterrows():
            lines.append(f"- {row['threshold_posture']}: `{int(row['rows'])}` mapped cohorts")
        lines.append('')

    lines.extend([
        '## How To Use This Pack',
        '- Use the runner as the base historical truth table for player-events research.',
        '- Use the miss audit and threshold tuning outputs to identify where the system is too strict or too loose.',
        '- Use the shadow trial and priority boards to test research-only threshold changes before changing any live logic.',
        '- Repeat the run after new settled evidence or after research-side parameter changes, then compare the outputs run-to-run.',
        '',
    ])

    output_md.write_text('\n'.join(lines) + '\n')


def build_backtest_pack(
    master_csv: Path,
    bookings_csv: Path,
    team_csv: Path,
    outdir: Path,
    overrides_csv: Path = DEFAULT_OVERRIDES_CSV,
    hit_threshold_overrides_csv: Path = DEFAULT_HIT_THRESHOLD_OVERRIDES_CSV,
    market_history_gate_csv: Path = DEFAULT_MARKET_HISTORY_GATE_CSV,
) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)

    build_actuals_layer(DEFAULT_ACTUALS_STATS, DEFAULT_ACTUALS_FIXTURES, DEFAULT_COVERAGE_CSV, DEFAULT_COVERAGE_MD)
    build_coverage_map(str(DEFAULT_COVERAGE_CSV), str(DEFAULT_COVERAGE_MAP_MD))

    runner_csv = outdir / 'player_events_3y_backtest_runner.csv'
    runner_md = outdir / 'player_events_3y_backtest_runner.md'
    build_runner(
        str(master_csv),
        str(bookings_csv),
        str(team_csv),
        str(runner_csv),
        str(runner_md),
        str(overrides_csv),
        str(hit_threshold_overrides_csv),
        str(market_history_gate_csv),
    )

    miss_csv = outdir / 'player_events_3y_miss_audit.csv'
    miss_md = outdir / 'player_events_3y_miss_audit.md'
    build_miss_audit(str(runner_csv), str(miss_csv), str(miss_md))

    threshold_csv = outdir / 'player_events_3y_threshold_tuning.csv'
    threshold_md = outdir / 'player_events_3y_threshold_tuning.md'
    build_threshold_audit(str(miss_csv), str(threshold_csv), str(threshold_md))

    scorecut_csv = outdir / 'player_events_3y_numeric_scorecuts.csv'
    scorecut_md = outdir / 'player_events_3y_numeric_scorecuts.md'
    build_scorecuts(str(threshold_csv), str(scorecut_csv), str(scorecut_md))

    patch_csv = outdir / 'player_events_3y_patch_proposal.csv'
    patch_md = outdir / 'player_events_3y_patch_proposal.md'
    build_patch_proposal(str(scorecut_csv), str(patch_csv), str(patch_md))

    shadow_csv = outdir / 'player_events_3y_shadow_trials.csv'
    shadow_md = outdir / 'player_events_3y_shadow_trials.md'
    build_shadow_trials(str(patch_csv), str(runner_csv), str(shadow_csv), str(shadow_md))

    priority_csv = outdir / 'player_events_3y_shadow_priority.csv'
    priority_md = outdir / 'player_events_3y_shadow_priority.md'
    build_shadow_priority(str(shadow_csv), str(priority_csv), str(priority_md))

    team_map_csv = outdir / 'player_events_3y_team_league_comp_map.csv'
    team_map_md = outdir / 'player_events_3y_team_league_comp_map.md'
    build_team_map(str(runner_csv), str(team_map_csv), str(team_map_md))

    summary_md = outdir / 'PLAYER_EVENTS_3Y_BACKTEST_PACK.md'
    summarize(
        runner_csv,
        miss_csv,
        threshold_csv,
        patch_csv,
        shadow_csv,
        priority_csv,
        team_map_csv,
        DEFAULT_COVERAGE_CSV,
        overrides_csv,
        hit_threshold_overrides_csv,
        market_history_gate_csv,
        summary_md,
    )

    return {
        'runner_csv': runner_csv,
        'runner_md': runner_md,
        'miss_csv': miss_csv,
        'miss_md': miss_md,
        'threshold_csv': threshold_csv,
        'threshold_md': threshold_md,
        'scorecut_csv': scorecut_csv,
        'scorecut_md': scorecut_md,
        'patch_csv': patch_csv,
        'patch_md': patch_md,
        'shadow_csv': shadow_csv,
        'shadow_md': shadow_md,
        'priority_csv': priority_csv,
        'priority_md': priority_md,
        'team_map_csv': team_map_csv,
        'team_map_md': team_map_md,
        'coverage_csv': DEFAULT_COVERAGE_CSV,
        'coverage_md': DEFAULT_COVERAGE_MD,
        'coverage_map_md': DEFAULT_COVERAGE_MAP_MD,
        'overrides_csv': overrides_csv,
        'hit_threshold_overrides_csv': hit_threshold_overrides_csv,
        'market_history_gate_csv': market_history_gate_csv,
        'summary_md': summary_md,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a dedicated 3-year player-events backtest pack from the current research estate.')
    parser.add_argument('--master-csv', default=str(DEFAULT_MASTER))
    parser.add_argument('--bookings-csv', default=str(DEFAULT_BOOKINGS))
    parser.add_argument('--team-csv', default=str(DEFAULT_TEAM))
    parser.add_argument('--overrides-csv', default=str(DEFAULT_OVERRIDES_CSV))
    parser.add_argument('--hit-threshold-overrides-csv', default=str(DEFAULT_HIT_THRESHOLD_OVERRIDES_CSV))
    parser.add_argument('--market-history-gate-csv', default=str(DEFAULT_MARKET_HISTORY_GATE_CSV))
    parser.add_argument('--outdir', default=str(DEFAULT_OUTROOT / f"player_events_3y_backtest__{datetime.now().strftime('%Y-%m-%d__%H%M%S')}"))
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    outputs = build_backtest_pack(
        Path(args.master_csv),
        Path(args.bookings_csv),
        Path(args.team_csv),
        Path(args.outdir),
        Path(args.overrides_csv),
        Path(args.hit_threshold_overrides_csv),
        Path(args.market_history_gate_csv),
    )
    print(f"WROTE PACK: {args.outdir}")
    for key, value in outputs.items():
        print(f"{key}: {value}")
