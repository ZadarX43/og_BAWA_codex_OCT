from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .paths import FEATURE_FILES, NORMALIZED_FILES
from .utils import safe_div

PURPOSE = 'Build rolling pre-match referee profile features and matchup flags.'
TARGET_PATH = FEATURE_FILES['api_referee_profile_features']
KEYS = [
    'fixture_id', 'fixture_key', 'league', 'league_id', 'season', 'match_date',
    'home_team_id', 'away_team_id', 'home_team_name', 'away_team_name',
]


def _mean(records: list[dict], key: str, n: int) -> float:
    sample = records[:n]
    return sum(float(r.get(key, 0.0) or 0.0) for r in sample) / len(sample) if sample else 0.0


def _sum(records: list[dict], key: str, n: int) -> float:
    sample = records[:n]
    return sum(float(r.get(key, 0.0) or 0.0) for r in sample) if sample else 0.0


def _strictness_band(score: float) -> str:
    if score >= 0.75:
        return 'STRICT'
    if score >= 0.55:
        return 'HIGH'
    if score >= 0.35:
        return 'MEDIUM'
    return 'LOW'


def build_referee_profile_features(fixtures_csv: str, team_stats_csv: str, events_csv: str, enriched_csv: str, output_csv: str = str(TARGET_PATH)) -> pd.DataFrame:
    fixtures = pd.read_csv(fixtures_csv)
    team_stats = pd.read_csv(team_stats_csv)
    events = pd.read_csv(events_csv)
    enriched = pd.read_csv(enriched_csv)
    fixtures['kickoff_ts_utc'] = pd.to_datetime(fixtures['kickoff_ts_utc'], errors='coerce', utc=True)

    stats = team_stats.groupby('fixture_id').agg(
        total_fouls=('fouls_for', 'sum'),
        total_yellow=('yellow_cards', 'sum'),
        total_red=('red_cards', 'sum'),
        home_cards=('yellow_cards', lambda s: float(s.iloc[0]) if len(s) else 0.0),
        away_cards=('yellow_cards', lambda s: float(s.iloc[1]) if len(s) > 1 else 0.0),
    ).reset_index()
    stats['total_cards'] = stats['total_yellow'] + stats['total_red']

    ev = events.copy()
    ev['event_type_norm'] = ev['event_type'].astype(str).str.lower()
    ev['event_detail_norm'] = ev['event_detail'].astype(str).str.lower()
    penalties = ev[(ev['event_type_norm'].str.contains('goal|card|var|penalty', regex=True)) & (ev['event_detail_norm'].str.contains('penalty', regex=False))]
    pen_counts = penalties.groupby('fixture_id').size().rename('penalty_events').reset_index()

    merged = fixtures.merge(stats, on='fixture_id', how='left').merge(pen_counts, on='fixture_id', how='left')
    merged = merged.merge(enriched[KEYS + ['home_fouls_for_l5','away_fouls_for_l5','home_cards_total_l5','away_cards_total_l5']], on=KEYS, how='left')
    merged['penalty_events'] = pd.to_numeric(merged['penalty_events'], errors='coerce').fillna(0.0)
    merged = merged.sort_values(['kickoff_ts_utc', 'fixture_id']).reset_index(drop=True)

    history: dict[str, list[dict]] = defaultdict(list)
    out_rows = []
    for _, fx in merged.iterrows():
        ref = str(fx.get('referee_name', '') or '').strip()
        prev = list(reversed(history.get(ref, []))) if ref else []

        cards_per_foul = safe_div(_sum(prev, 'total_cards', 20), _sum(prev, 'total_fouls', 20)) if prev else 0.0
        red_tendency = safe_div(_sum(prev, 'total_red', 20), min(len(prev), 20)) if prev else 0.0
        penalty_tendency = safe_div(_sum(prev, 'penalty_events', 20), min(len(prev), 20)) if prev else 0.0
        strictness_score = min(1.0, (
            (0.40 * min(_mean(prev, 'total_cards', 20) / 6.0, 1.0)) +
            (0.30 * min(_mean(prev, 'total_fouls', 20) / 30.0, 1.0)) +
            (0.20 * min(cards_per_foul / 0.30, 1.0)) +
            (0.10 * min(red_tendency / 0.25, 1.0))
        )) if prev else 0.0

        home_aggression = float(fx.get('home_fouls_for_l5', 0.0) or 0.0) + float(fx.get('home_cards_total_l5', 0.0) or 0.0)
        away_aggression = float(fx.get('away_fouls_for_l5', 0.0) or 0.0) + float(fx.get('away_cards_total_l5', 0.0) or 0.0)

        row = {k: fx[k] for k in KEYS}
        row.update({
            'referee_name': ref,
            'ref_matches_sample_l20': min(len(prev), 20),
            'ref_bookings_per_match': _mean(prev, 'total_cards', 20),
            'ref_fouls_per_match': _mean(prev, 'total_fouls', 20),
            'ref_cards_per_foul': cards_per_foul,
            'ref_red_card_tendency': red_tendency,
            'ref_penalty_tendency': penalty_tendency,
            'ref_home_bias': safe_div(_sum(prev, 'home_cards', 20), max(_sum(prev, 'away_cards', 20), 1e-9)) if prev else 0.0,
            'ref_strictness_score': strictness_score,
            'ref_leniency_band': _strictness_band(strictness_score),
            'home_aggressive_team_strict_ref_flag': int(home_aggression >= 15.0 and strictness_score >= 0.70),
            'away_aggressive_team_strict_ref_flag': int(away_aggression >= 15.0 and strictness_score >= 0.70),
            'combined_aggression_strict_ref_flag': int((home_aggression + away_aggression) >= 28.0 and strictness_score >= 0.70),
            'booking_pressure_with_ref': (home_aggression + away_aggression) * strictness_score,
        })
        out_rows.append(row)

        if ref:
            history[ref].append({
                'total_cards': float(fx.get('total_cards', 0.0) or 0.0),
                'total_fouls': float(fx.get('total_fouls', 0.0) or 0.0),
                'total_red': float(fx.get('total_red', 0.0) or 0.0),
                'penalty_events': float(fx.get('penalty_events', 0.0) or 0.0),
                'home_cards': float(fx.get('home_cards', 0.0) or 0.0),
                'away_cards': float(fx.get('away_cards', 0.0) or 0.0),
            })

    out = pd.DataFrame(out_rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=PURPOSE)
    parser.add_argument('--fixtures-csv', default=str(NORMALIZED_FILES['fixtures_master']))
    parser.add_argument('--team-stats-csv', default=str(NORMALIZED_FILES['match_team_stats']))
    parser.add_argument('--events-csv', default=str(NORMALIZED_FILES['match_events']))
    parser.add_argument('--enriched-csv', default=str(FEATURE_FILES['api_enriched_fixture_features']))
    parser.add_argument('--output-csv', default=str(TARGET_PATH))
    args = parser.parse_args()
    df = build_referee_profile_features(args.fixtures_csv, args.team_stats_csv, args.events_csv, args.enriched_csv, args.output_csv)
    print(f'WROTE: {args.output_csv} rows={len(df)} cols={len(df.columns)}')


if __name__ == '__main__':
    main()
