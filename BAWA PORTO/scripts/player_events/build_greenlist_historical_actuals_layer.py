from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from run_greenlist_specialist_family_batch import BATCHES

REPO_ROOT = Path(__file__).resolve().parents[2]
NORMALIZED_DIR = REPO_ROOT / 'data_sources' / 'api_football' / 'normalized'
DEFAULT_STATS_OUT = NORMALIZED_DIR / 'match_player_stats__GREENLIST_FULL_3Y__2022_2024.csv'
DEFAULT_FIXTURES_OUT = NORMALIZED_DIR / 'fixtures_master__GREENLIST_FULL_3Y__2022_2024.csv'
DEFAULT_COVERAGE_CSV = REPO_ROOT / 'reports' / 'player_events' / 'quality_audits' / 'greenlist_historical_actuals_coverage.csv'
DEFAULT_COVERAGE_MD = REPO_ROOT / 'reports' / 'player_events' / 'quality_audits' / 'greenlist_historical_actuals_coverage.md'


def target_leagues() -> list[str]:
    out: list[str] = []
    for leagues in BATCHES.values():
        out.extend(leagues)
    return sorted(set(out))


def _load_pair(league_tag: str, season: int) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    stats_path = NORMALIZED_DIR / f'match_player_stats__{league_tag}__{season}.csv'
    fixtures_path = NORMALIZED_DIR / f'fixtures_master__{league_tag}__{season}.csv'
    if not stats_path.exists() or not fixtures_path.exists():
        return None, None

    stats = pd.read_csv(stats_path, low_memory=False)
    fixtures = pd.read_csv(fixtures_path, low_memory=False)
    return stats, fixtures


def build_layer(stats_out: Path, fixtures_out: Path, coverage_csv: Path, coverage_md: Path) -> dict[str, Path]:
    stats_out.parent.mkdir(parents=True, exist_ok=True)
    coverage_csv.parent.mkdir(parents=True, exist_ok=True)

    coverage_rows: list[dict[str, object]] = []
    all_stats: list[pd.DataFrame] = []
    all_fixtures: list[pd.DataFrame] = []

    for league_tag in target_leagues():
        for season in (2022, 2023, 2024):
            stats, fixtures = _load_pair(league_tag, season)
            if stats is None or fixtures is None:
                coverage_rows.append(
                    {
                        'league_tag': league_tag,
                        'season': season,
                        'coverage_flag': 'MISSING',
                        'player_rows': 0,
                        'fixture_rows': 0,
                        'unique_fixtures': 0,
                        'date_from': pd.NA,
                        'date_to': pd.NA,
                    }
                )
                continue

            fixtures = fixtures.copy()
            fixtures['league_tag'] = league_tag
            fixtures['source_season'] = season

            stats = stats.copy()
            stats['league_tag'] = league_tag
            stats['source_season'] = season

            fixture_cols = [
                'fixture_id',
                'fixture_key',
                'league',
                'league_id',
                'season',
                'match_date',
                'home_team_id',
                'away_team_id',
                'home_team_name',
                'away_team_name',
                'kickoff_ts_utc',
                'status',
                'venue_id',
                'venue_name',
                'referee_name',
                'league_tag',
                'source_season',
            ]
            for col in fixture_cols:
                if col not in fixtures.columns:
                    fixtures[col] = pd.NA
            fixtures = fixtures[fixture_cols].drop_duplicates(subset=['fixture_id'])

            merged = stats.merge(
                fixtures[['fixture_id', 'fixture_key', 'league', 'season', 'match_date', 'home_team_id', 'away_team_id', 'home_team_name', 'away_team_name', 'league_tag', 'source_season']],
                on='fixture_id',
                how='left',
            )
            merged['team_name'] = pd.NA
            merged.loc[merged['team_id'] == merged['home_team_id'], 'team_name'] = merged.loc[merged['team_id'] == merged['home_team_id'], 'home_team_name']
            merged.loc[merged['team_id'] == merged['away_team_id'], 'team_name'] = merged.loc[merged['team_id'] == merged['away_team_id'], 'away_team_name']

            player_cols = [
                'fixture_id', 'fixture_key', 'league_tag', 'league', 'season', 'source_season', 'match_date',
                'home_team_name', 'away_team_name', 'team_id', 'team_name', 'player_id', 'player_name', 'position',
                'minutes', 'started_flag', 'subbed_on_flag', 'subbed_off_flag', 'rating', 'goals', 'assists',
                'shots_total', 'shots_on_target', 'passes_total', 'passes_key', 'passes_accurate',
                'tackles', 'interceptions', 'blocks', 'duels_total', 'duels_won',
                'dribbles_attempted', 'dribbles_successful', 'dribbled_past',
                'fouls_drawn', 'fouls_committed', 'yellow_cards', 'red_cards', 'saves', 'goals_conceded',
            ]
            for col in player_cols:
                if col not in merged.columns:
                    merged[col] = pd.NA
            merged = merged[player_cols]

            all_stats.append(merged)
            all_fixtures.append(fixtures)

            match_dates = pd.to_datetime(fixtures['match_date'], errors='coerce')
            coverage_rows.append(
                {
                    'league_tag': league_tag,
                    'season': season,
                    'coverage_flag': 'AVAILABLE',
                    'player_rows': int(len(merged)),
                    'fixture_rows': int(len(fixtures)),
                    'unique_fixtures': int(fixtures['fixture_key'].nunique()),
                    'date_from': match_dates.min().date().isoformat() if not match_dates.dropna().empty else pd.NA,
                    'date_to': match_dates.max().date().isoformat() if not match_dates.dropna().empty else pd.NA,
                }
            )

    stats_out_df = pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()
    fixtures_out_df = pd.concat(all_fixtures, ignore_index=True).drop_duplicates(subset=['fixture_id']) if all_fixtures else pd.DataFrame()
    coverage_df = pd.DataFrame(coverage_rows).sort_values(['league_tag', 'season'])

    stats_out_df.to_csv(stats_out, index=False)
    fixtures_out_df.to_csv(fixtures_out, index=False)
    coverage_df.to_csv(coverage_csv, index=False)

    lines = [
        '# Greenlist Historical Actuals Coverage',
        '',
        '- Builds a consistent joinable historical player-events actuals layer across the greenlist leagues/batches we currently have locally.',
        '- Coverage here reflects the local normalized API Football player-stat archive, not an internet backfill.',
        '',
    ]
    summary = (
        coverage_df.groupby('league_tag', dropna=False)
        .agg(
            seasons_available=('coverage_flag', lambda s: int((pd.Series(s) == 'AVAILABLE').sum())),
            player_rows=('player_rows', 'sum'),
            unique_fixtures=('unique_fixtures', 'sum'),
        )
        .reset_index()
        .sort_values(['seasons_available', 'player_rows'], ascending=[False, False])
    )
    lines.append('## League Summary')
    for _, row in summary.iterrows():
        lines.append(
            f"- {row['league_tag']} | seasons_available={int(row['seasons_available'])}/3 | player_rows={int(row['player_rows'])} | fixtures={int(row['unique_fixtures'])}"
        )
    lines.append('')

    missing = coverage_df[coverage_df['coverage_flag'] == 'MISSING']
    lines.append('## Missing Coverage')
    if missing.empty:
        lines.append('- None.')
    else:
        for _, row in missing.iterrows():
            lines.append(f"- {row['league_tag']} | season={int(row['season'])}")
    lines.append('')

    coverage_md.write_text('\n'.join(lines) + '\n')
    return {
        'stats_out': stats_out,
        'fixtures_out': fixtures_out,
        'coverage_csv': coverage_csv,
        'coverage_md': coverage_md,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a combined greenlist historical player-events actuals layer in a normalized joinable format.')
    parser.add_argument('--stats-out', default=str(DEFAULT_STATS_OUT))
    parser.add_argument('--fixtures-out', default=str(DEFAULT_FIXTURES_OUT))
    parser.add_argument('--coverage-csv', default=str(DEFAULT_COVERAGE_CSV))
    parser.add_argument('--coverage-md', default=str(DEFAULT_COVERAGE_MD))
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    outputs = build_layer(Path(args.stats_out), Path(args.fixtures_out), Path(args.coverage_csv), Path(args.coverage_md))
    for key, value in outputs.items():
        print(f'{key}: {value}')
