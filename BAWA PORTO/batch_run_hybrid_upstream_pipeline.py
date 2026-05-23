#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from build_hybrid_consensus_inputs import build_consensus_inputs
from build_hybrid_goal_mass_inputs import build_goal_mass_inputs
from build_hybrid_threshold_policy import build_policy
from train_hybrid_goal_mass import train_goal_mass
from train_hybrid_ou25_tuned import tune_ou25
from train_hybrid_side_markets import train_side_markets
from scripts.api_football.audit_footystats_join import build_join_audit
from scripts.api_football.audit_hybrid_thresholds import build_threshold_audit
from scripts.api_football.audit_lambda_vs_direct_side import build_audit as build_lambda_vs_direct_audit
from scripts.api_football.build_enriched_fixture_features import build_enriched_fixture_features
from scripts.api_football.build_event_features import build_event_features
from scripts.api_football.build_hybrid_match_training import build_hybrid_match_training
from scripts.api_football.build_h2h_regime_features import build_h2h_regime_features
from scripts.api_football.build_injury_features import build_injury_features
from scripts.api_football.build_matchup_interaction_features import build_matchup_interaction_features
from scripts.api_football.build_referee_profile_features import build_referee_profile_features
from scripts.api_football.build_lineup_features import build_lineup_features
from scripts.api_football.build_team_identity_features import build_team_identity_features
from scripts.api_football.build_odds_features import build_odds_features
from scripts.api_football.build_player_rolling_features import build_player_rolling_features
from scripts.api_football.build_team_rolling_features import build_team_rolling_features
from scripts.api_football.normalize_fixtures_master import build_fixtures_master
from scripts.api_football.normalize_injuries import build_injuries
from scripts.api_football.normalize_lineups import build_lineups
from scripts.api_football.normalize_match_events import build_match_events
from scripts.api_football.normalize_match_player_stats import build_match_player_stats
from scripts.api_football.normalize_match_team_stats import build_match_team_stats
from scripts.api_football.paths import API_ROOT, FEATURES_DIR, HYBRID_DIR, NORMALIZED_DIR, RAW_DIR, REPORTS_DIR, REPO_ROOT
from scripts.api_football.schema_contracts import NORMALIZED_SCHEMAS

DEFAULT_MANIFEST = REPO_ROOT / 'configs' / 'hybrid_league_manifest.json'
DEFAULT_SUMMARY = REPORTS_DIR / 'hybrid_upstream_pipeline_summary.csv'
DEFAULT_MODELSTORE = REPO_ROOT / 'ModelStore' / 'Hybrid'
DEFAULT_HOLDOUT_FRAC = 0.2
DEFAULT_RANDOM_SEED = 42


def parse_csv_text(text: str) -> List[str]:
    return [part.strip() for part in str(text or '').split(',') if part.strip()]


def parse_csv_ints(text: str) -> List[int]:
    return [int(part.strip()) for part in str(text or '').split(',') if part.strip()]


def tagged_csv(directory: Path, stem: str, tag: str, season: int) -> Path:
    return directory / f'{stem}__{tag}__{season}.csv'


def tagged_json(directory: Path, stem: str, tag: str, season: int) -> Path:
    return directory / f'{stem}__{tag}__{season}.json'


def raw_fixture_stem(league_id: int, season: int) -> str:
    return f'fixtures__league_{league_id}__season_{season}'


def ensure_empty_odds_long(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=NORMALIZED_SCHEMAS['odds_prematch_long']).to_csv(path, index=False)


def load_manifest(path: Path) -> List[Dict[str, object]]:
    with path.open('r', encoding='utf-8') as fh:
        return json.load(fh)


def pick_entries(entries: List[Dict[str, object]], tags: Iterable[str]) -> List[Dict[str, object]]:
    wanted = {tag for tag in tags if tag}
    if not wanted:
        return entries
    return [entry for entry in entries if str(entry.get('tag')) in wanted]


def season_list(entry: Dict[str, object], override: List[int]) -> List[int]:
    if override:
        return override
    raw = entry.get('seasons') or []
    return [int(x) for x in raw]


def build_paths(tag: str, league_id: int, season: int) -> Dict[str, Path]:
    stem = raw_fixture_stem(league_id, season)
    normalized = {
        'fixtures_master': tagged_csv(NORMALIZED_DIR, 'fixtures_master', tag, season),
        'match_team_stats': tagged_csv(NORMALIZED_DIR, 'match_team_stats', tag, season),
        'match_events': tagged_csv(NORMALIZED_DIR, 'match_events', tag, season),
        'match_player_stats': tagged_csv(NORMALIZED_DIR, 'match_player_stats', tag, season),
        'lineups': tagged_csv(NORMALIZED_DIR, 'lineups', tag, season),
        'injuries': tagged_csv(NORMALIZED_DIR, 'injuries', tag, season),
        'odds_prematch_long': tagged_csv(NORMALIZED_DIR, 'odds_prematch_long', tag, season),
    }
    features = {
        'api_team_rolling_features': tagged_csv(FEATURES_DIR, 'api_team_rolling_features', tag, season),
        'api_event_features': tagged_csv(FEATURES_DIR, 'api_event_features', tag, season),
        'api_lineup_features': tagged_csv(FEATURES_DIR, 'api_lineup_features', tag, season),
        'api_injury_features': tagged_csv(FEATURES_DIR, 'api_injury_features', tag, season),
        'api_odds_features': tagged_csv(FEATURES_DIR, 'api_odds_features', tag, season),
        'api_enriched_fixture_features': tagged_csv(FEATURES_DIR, 'api_enriched_fixture_features', tag, season),
        'api_player_rolling_features': tagged_csv(FEATURES_DIR, 'api_player_rolling_features', tag, season),
        'api_team_identity_features': tagged_csv(FEATURES_DIR, 'api_team_identity_features', tag, season),
        'api_matchup_interaction_features': tagged_csv(FEATURES_DIR, 'api_matchup_interaction_features', tag, season),
        'api_h2h_regime_features': tagged_csv(FEATURES_DIR, 'api_h2h_regime_features', tag, season),
        'api_referee_profile_features': tagged_csv(FEATURES_DIR, 'api_referee_profile_features', tag, season),
    }
    hybrid = {
        'hybrid_match_training': tagged_csv(HYBRID_DIR, 'hybrid_match_training', tag, season),
        'hybrid_goal_mass_inputs': tagged_csv(HYBRID_DIR, 'hybrid_goal_mass_inputs', tag, season),
        'hybrid_consensus_inputs': tagged_csv(HYBRID_DIR, 'hybrid_consensus_inputs', tag, season),
    }
    reports = {
        'join_audit': tagged_csv(REPORTS_DIR, 'api_footystats_join_audit', tag, season),
        'goal_mass_metrics': tagged_csv(REPORTS_DIR, 'hybrid_goal_mass_metrics', tag, season),
        'side_market_metrics': tagged_csv(REPORTS_DIR, 'hybrid_side_market_metrics', tag, season),
        'ou25_tuned_metrics': tagged_csv(REPORTS_DIR, 'hybrid_ou25_tuned_metrics', tag, season),
        'lambda_vs_direct_side': tagged_csv(REPORTS_DIR, 'hybrid_lambda_vs_direct_side', tag, season),
        'threshold_audit': tagged_csv(REPORTS_DIR, 'hybrid_threshold_audit', tag, season),
        'research_winners': tagged_json(REPORTS_DIR, 'hybrid_research_winners', tag, season),
        'threshold_policy_json': tagged_json(REPORTS_DIR, 'hybrid_threshold_policy', tag, season),
        'threshold_policy_csv': tagged_csv(REPORTS_DIR, 'hybrid_threshold_policy', tag, season),
    }
    raw = {
        'fixtures': RAW_DIR / f'{stem}__fixtures.jsonl',
        'bundle': RAW_DIR / f'{stem}__fixtures_bundle.jsonl',
        'injuries': RAW_DIR / f'{stem}__injuries.jsonl',
    }
    model_dir = DEFAULT_MODELSTORE / f'{tag}__{season}'
    merged_csv = REPO_ROOT / 'Matches' / '__merged__' / f'{tag}__merged.csv'
    return {
        'raw': raw,
        'normalized': normalized,
        'features': features,
        'hybrid': hybrid,
        'reports': reports,
        'model_dir': model_dir,
        'merged_csv': merged_csv,
    }


def validate_inputs(paths: Dict[str, object]) -> List[str]:
    missing = []
    for key, path in paths['raw'].items():
        if not Path(path).exists():
            missing.append(str(path))
    if not Path(paths['merged_csv']).exists():
        missing.append(str(paths['merged_csv']))
    return missing


def run_one(league: str, tag: str, league_id: int, season: int, holdout_frac: float, random_seed: int) -> Dict[str, object]:
    paths = build_paths(tag, league_id, season)
    missing = validate_inputs(paths)
    if missing:
        return {
            'league': league,
            'tag': tag,
            'league_id': league_id,
            'season': season,
            'status': 'skipped_missing_inputs',
            'missing_count': len(missing),
            'missing_inputs': ' | '.join(missing),
        }

    ensure_empty_odds_long(paths['normalized']['odds_prematch_long'])

    fixtures_df = build_fixtures_master(str(paths['raw']['fixtures']), str(paths['normalized']['fixtures_master']))
    team_stats_df = build_match_team_stats(str(paths['raw']['bundle']), str(paths['normalized']['match_team_stats']))
    events_df = build_match_events(str(paths['raw']['bundle']), str(paths['normalized']['match_events']))
    player_stats_df = build_match_player_stats(str(paths['raw']['bundle']), str(paths['normalized']['match_player_stats']))
    lineups_df = build_lineups(str(paths['raw']['bundle']), str(paths['normalized']['lineups']))
    injuries_df = build_injuries(str(paths['raw']['injuries']), str(paths['normalized']['injuries']))

    team_features_df = build_team_rolling_features(str(paths['normalized']['fixtures_master']), str(paths['normalized']['match_team_stats']), str(paths['features']['api_team_rolling_features']))
    event_features_df = build_event_features(str(paths['normalized']['fixtures_master']), str(paths['normalized']['match_events']), str(paths['normalized']['match_team_stats']), str(paths['features']['api_event_features']))
    lineup_features_df = build_lineup_features(str(paths['normalized']['fixtures_master']), str(paths['normalized']['lineups']), str(paths['normalized']['match_player_stats']), str(paths['features']['api_lineup_features']))
    injury_features_df = build_injury_features(str(paths['normalized']['fixtures_master']), str(paths['normalized']['injuries']), str(paths['normalized']['match_player_stats']), str(paths['features']['api_injury_features']))
    odds_features_df = build_odds_features(str(paths['normalized']['fixtures_master']), str(paths['normalized']['odds_prematch_long']), str(paths['features']['api_odds_features']))
    enriched_base_df = build_enriched_fixture_features(
        str(paths['normalized']['fixtures_master']),
        str(paths['features']['api_team_rolling_features']),
        str(paths['features']['api_event_features']),
        str(paths['features']['api_lineup_features']),
        str(paths['features']['api_injury_features']),
        str(paths['features']['api_odds_features']),
        str(paths['features']['api_enriched_fixture_features']),
    )
    player_rolling_df = build_player_rolling_features(str(paths['normalized']['fixtures_master']), str(paths['normalized']['match_player_stats']), str(paths['features']['api_player_rolling_features']))
    team_identity_df = build_team_identity_features(
        str(paths['features']['api_enriched_fixture_features']),
        str(paths['features']['api_team_identity_features']),
    )
    matchup_df = build_matchup_interaction_features(
        str(paths['features']['api_team_identity_features']),
        str(paths['features']['api_enriched_fixture_features']),
        str(paths['features']['api_matchup_interaction_features']),
    )
    h2h_df = build_h2h_regime_features(
        str(paths['normalized']['fixtures_master']),
        str(paths['normalized']['match_team_stats']),
        str(paths['features']['api_h2h_regime_features']),
    )
    referee_df = build_referee_profile_features(
        str(paths['normalized']['fixtures_master']),
        str(paths['normalized']['match_team_stats']),
        str(paths['normalized']['match_events']),
        str(paths['features']['api_enriched_fixture_features']),
        str(paths['features']['api_referee_profile_features']),
    )
    enriched_df = build_enriched_fixture_features(
        str(paths['normalized']['fixtures_master']),
        str(paths['features']['api_team_rolling_features']),
        str(paths['features']['api_event_features']),
        str(paths['features']['api_lineup_features']),
        str(paths['features']['api_injury_features']),
        str(paths['features']['api_odds_features']),
        str(paths['features']['api_enriched_fixture_features']),
        str(paths['features']['api_team_identity_features']),
        str(paths['features']['api_matchup_interaction_features']),
        str(paths['features']['api_h2h_regime_features']),
        str(paths['features']['api_referee_profile_features']),
    )

    join_audit_df = build_join_audit(
        str(paths['normalized']['fixtures_master']),
        str(paths['features']['api_enriched_fixture_features']),
        str(paths['merged_csv']),
        str(paths['normalized']['match_team_stats']),
        str(paths['reports']['join_audit']),
    )
    hybrid_df = build_hybrid_match_training(
        str(paths['merged_csv']),
        str(paths['reports']['join_audit']),
        str(paths['features']['api_enriched_fixture_features']),
        str(paths['hybrid']['hybrid_match_training']),
    )
    goal_mass_inputs_df = build_goal_mass_inputs(paths['hybrid']['hybrid_match_training'], paths['hybrid']['hybrid_goal_mass_inputs'])
    goal_mass_report = train_goal_mass(paths['hybrid']['hybrid_goal_mass_inputs'], paths['model_dir'], paths['reports']['goal_mass_metrics'], holdout_frac, random_seed)
    side_market_report = train_side_markets(
        paths['hybrid']['hybrid_goal_mass_inputs'],
        paths['model_dir'],
        paths['reports']['side_market_metrics'],
        paths['model_dir'] / 'home_lambda__hybrid_goal_mass.pkl',
        paths['model_dir'] / 'away_lambda__hybrid_goal_mass.pkl',
        holdout_frac,
        random_seed,
    )
    ou25_report = tune_ou25(paths['hybrid']['hybrid_match_training'], paths['model_dir'], paths['reports']['ou25_tuned_metrics'], holdout_frac, random_seed)
    lambda_vs_direct_df = build_lambda_vs_direct_audit(paths['hybrid']['hybrid_goal_mass_inputs'], paths['model_dir'], paths['reports']['lambda_vs_direct_side'])
    threshold_df, winners = build_threshold_audit(
        paths['hybrid']['hybrid_match_training'],
        paths['hybrid']['hybrid_goal_mass_inputs'],
        paths['model_dir'],
        paths['reports']['threshold_audit'],
        paths['reports']['research_winners'],
    )
    policy = build_policy(
        paths['reports']['threshold_audit'],
        paths['reports']['research_winners'],
        paths['reports']['threshold_policy_json'],
        paths['reports']['threshold_policy_csv'],
        min_rows=8,
        min_hit_rate=0.60,
    )
    consensus_df = build_consensus_inputs(
        paths['hybrid']['hybrid_match_training'],
        paths['hybrid']['hybrid_goal_mass_inputs'],
        paths['reports']['threshold_policy_json'],
        paths['model_dir'],
        paths['hybrid']['hybrid_consensus_inputs'],
    )

    return {
        'league': league,
        'tag': tag,
        'league_id': league_id,
        'season': season,
        'status': 'built',
        'fixtures_rows': len(fixtures_df),
        'team_stats_rows': len(team_stats_df),
        'events_rows': len(events_df),
        'player_stats_rows': len(player_stats_df),
        'lineups_rows': len(lineups_df),
        'injuries_rows': len(injuries_df),
        'team_features_rows': len(team_features_df),
        'event_features_rows': len(event_features_df),
        'lineup_features_rows': len(lineup_features_df),
        'injury_features_rows': len(injury_features_df),
        'odds_features_rows': len(odds_features_df),
        'enriched_rows': len(enriched_df),
        'player_rolling_rows': len(player_rolling_df),
        'team_identity_rows': len(team_identity_df),
        'matchup_rows': len(matchup_df),
        'h2h_rows': len(h2h_df),
        'referee_rows': len(referee_df),
        'join_audit_rows': len(join_audit_df),
        'hybrid_rows': len(hybrid_df),
        'goal_mass_rows': len(goal_mass_inputs_df),
        'goal_mass_metrics_rows': len(goal_mass_report),
        'side_market_metrics_rows': len(side_market_report),
        'ou25_tuning_rows': len(ou25_report),
        'lambda_vs_direct_rows': len(lambda_vs_direct_df),
        'threshold_rows': len(threshold_df),
        'policy_market_count': len(policy.get('markets', {})),
        'consensus_rows': len(consensus_df),
        'hybrid_csv': str(paths['hybrid']['hybrid_match_training']),
        'consensus_csv': str(paths['hybrid']['hybrid_consensus_inputs']),
        'model_dir': str(paths['model_dir']),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the upstream multi-league hybrid pipeline for each league-season raw API set that is already fetched.')
    parser.add_argument('--manifest-json', default=str(DEFAULT_MANIFEST))
    parser.add_argument('--tags', default='', help='Optional comma-separated league tags to restrict the run.')
    parser.add_argument('--seasons', default='', help='Optional comma-separated season start years to override manifest seasons.')
    parser.add_argument('--summary-csv', default=str(DEFAULT_SUMMARY))
    parser.add_argument('--holdout-frac', type=float, default=DEFAULT_HOLDOUT_FRAC)
    parser.add_argument('--random-seed', type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument('--stop-on-error', action='store_true')
    args = parser.parse_args()

    manifest = pick_entries(load_manifest(Path(args.manifest_json)), parse_csv_text(args.tags))
    seasons_override = parse_csv_ints(args.seasons)
    rows: List[Dict[str, object]] = []

    for entry in manifest:
        league = str(entry.get('league'))
        tag = str(entry.get('tag'))
        league_id = entry.get('league_id')
        if league_id is None:
            rows.append({'league': league, 'tag': tag, 'season': '', 'status': 'skipped_missing_league_id'})
            continue
        for season in season_list(entry, seasons_override):
            try:
                row = run_one(league, tag, int(league_id), int(season), args.holdout_frac, args.random_seed)
                rows.append(row)
                print(f"{row['status'].upper()}: {league} {season} -> {tag}")
            except Exception as exc:
                row = {
                    'league': league,
                    'tag': tag,
                    'league_id': league_id,
                    'season': season,
                    'status': 'failed',
                    'error': repr(exc),
                }
                rows.append(row)
                print(f"FAILED: {league} {season} -> {exc!r}")
                if args.stop_on_error:
                    pd.DataFrame(rows).to_csv(args.summary_csv, index=False)
                    raise

    summary = pd.DataFrame(rows)
    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False)
    print(f'WROTE: {args.summary_csv} rows={len(summary)}')


if __name__ == '__main__':
    main()
