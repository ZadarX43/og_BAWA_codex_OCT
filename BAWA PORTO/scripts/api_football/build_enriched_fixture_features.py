from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from .paths import FEATURE_FILES, NORMALIZED_FILES
from .schema_contracts import FEATURE_SCHEMAS
from .scaffold import build_csv_stub

PURPOSE = 'Build final pre-match enriched fixture table by joining API feature families.'
TARGET_PATH = FEATURE_FILES['api_enriched_fixture_features']
KEYS = ['fixture_id','fixture_key','league','league_id','season','match_date','home_team_id','away_team_id','home_team_name','away_team_name']


def build_stub() -> pd.DataFrame:
    return build_csv_stub(TARGET_PATH, FEATURE_SCHEMAS['api_enriched_fixture_features'], PURPOSE, placeholder_row=False)


def _merge_one(base: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in other.columns if c not in KEYS] + ['fixture_id']
    return base.merge(other[cols].drop_duplicates(subset=['fixture_id']), on='fixture_id', how='left')


def build_enriched_fixture_features(fixtures_csv: str, team_csv: str, event_csv: str, lineup_csv: str, injury_csv: str, odds_csv: str, output_csv: str = str(TARGET_PATH), team_identity_csv: str | None = None, matchup_csv: str | None = None, h2h_csv: str | None = None, referee_csv: str | None = None) -> pd.DataFrame:
    base = pd.read_csv(fixtures_csv)
    team = pd.read_csv(team_csv)
    event = pd.read_csv(event_csv)
    lineup = pd.read_csv(lineup_csv)
    injury = pd.read_csv(injury_csv)
    odds = pd.read_csv(odds_csv)
    team_identity = pd.read_csv(team_identity_csv) if team_identity_csv else None
    matchup = pd.read_csv(matchup_csv) if matchup_csv else None
    h2h = pd.read_csv(h2h_csv) if h2h_csv else None
    referee = pd.read_csv(referee_csv) if referee_csv else None
    out = base.copy()
    out = _merge_one(out, team)
    out = _merge_one(out, event)
    out = _merge_one(out, lineup)
    out = _merge_one(out, injury)
    out = _merge_one(out, odds)
    if team_identity is not None:
        out = _merge_one(out, team_identity)
    if matchup is not None:
        out = _merge_one(out, matchup)
    if h2h is not None:
        out = _merge_one(out, h2h)
    if referee is not None:
        out = _merge_one(out, referee)
    ordered = KEYS + [c for c in out.columns if c not in KEYS]
    out = out.reindex(columns=ordered)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=PURPOSE)
    parser.add_argument('--write-stub', action='store_true', help='Write the scaffold output even though live transform logic is not implemented yet.')
    parser.add_argument('--fixtures-csv', default=str(NORMALIZED_FILES['fixtures_master']))
    parser.add_argument('--team-features-csv', default=str(FEATURE_FILES['api_team_rolling_features']))
    parser.add_argument('--event-features-csv', default=str(FEATURE_FILES['api_event_features']))
    parser.add_argument('--lineup-features-csv', default=str(FEATURE_FILES['api_lineup_features']))
    parser.add_argument('--injury-features-csv', default=str(FEATURE_FILES['api_injury_features']))
    parser.add_argument('--odds-features-csv', default=str(FEATURE_FILES['api_odds_features']))
    parser.add_argument('--output-csv', default=str(TARGET_PATH))
    parser.add_argument('--team-identity-csv', default='')
    parser.add_argument('--matchup-csv', default='')
    parser.add_argument('--h2h-csv', default='')
    parser.add_argument('--referee-csv', default='')
    args = parser.parse_args()
    if args.write_stub:
        df = build_stub()
        print(f'WROTE STUB: {TARGET_PATH} rows={len(df)}')
        return
    df = build_enriched_fixture_features(args.fixtures_csv, args.team_features_csv, args.event_features_csv, args.lineup_features_csv, args.injury_features_csv, args.odds_features_csv, args.output_csv, args.team_identity_csv or None, args.matchup_csv or None, args.h2h_csv or None, args.referee_csv or None)
    print(f'WROTE: {args.output_csv} rows={len(df)} cols={len(df.columns)}')


if __name__ == '__main__':
    main()
