from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .paths import FEATURE_FILES, NORMALIZED_FILES

PURPOSE = 'Build pre-match H2H tactical regime features from previous meetings only.'
TARGET_PATH = FEATURE_FILES['api_h2h_regime_features']
KEYS = [
    'fixture_id', 'fixture_key', 'league', 'league_id', 'season', 'match_date',
    'home_team_id', 'away_team_id', 'home_team_name', 'away_team_name',
]


def _safe_float(value: object, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    if value is None or pd.isna(value):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _mean(records: list[dict], key: str, n: int) -> float:
    sample = records[:n]
    return sum(_safe_float(r.get(key, 0.0), 0.0) for r in sample) / len(sample) if sample else 0.0


def _rate(records: list[dict], pred, n: int) -> float:
    sample = records[:n]
    return sum(1 for r in sample if pred(r)) / len(sample) if sample else 0.0


def _pair_key(home_team_id: int, away_team_id: int) -> tuple[int, int]:
    return tuple(sorted((int(home_team_id), int(away_team_id))))


def build_h2h_regime_features(fixtures_csv: str, team_stats_csv: str, output_csv: str = str(TARGET_PATH)) -> pd.DataFrame:
    fixtures = pd.read_csv(fixtures_csv)
    team_stats = pd.read_csv(team_stats_csv)
    fixtures['kickoff_ts_utc'] = pd.to_datetime(fixtures['kickoff_ts_utc'], errors='coerce', utc=True)

    agg = team_stats.groupby('fixture_id').agg(
        total_goals=('goals_for', 'sum'),
        total_fouls=('fouls_for', 'sum'),
        total_yellow=('yellow_cards', 'sum'),
        total_red=('red_cards', 'sum'),
        home_possession=('possession_pct', lambda s: float(s.iloc[0]) if len(s) else 0.0),
        away_possession=('possession_pct', lambda s: float(s.iloc[1]) if len(s) > 1 else (100.0 - float(s.iloc[0]) if len(s) else 0.0)),
        home_goals=('goals_for', lambda s: float(s.iloc[0]) if len(s) else 0.0),
        away_goals=('goals_for', lambda s: float(s.iloc[1]) if len(s) > 1 else 0.0),
    ).reset_index()
    agg['btts'] = ((agg['home_goals'] > 0) & (agg['away_goals'] > 0)).astype(int)
    agg['total_cards'] = agg['total_yellow'] + agg['total_red']
    agg['possession_diff_abs'] = (agg['home_possession'] - agg['away_possession']).abs()

    merged = fixtures.merge(agg, on='fixture_id', how='left')
    merged = merged.sort_values(['kickoff_ts_utc', 'fixture_id']).reset_index(drop=True)

    history: dict[tuple[int, int], list[dict]] = defaultdict(list)
    out_rows = []

    for _, fx in merged.iterrows():
        key = _pair_key(int(fx['home_team_id']), int(fx['away_team_id']))
        prev = list(reversed(history.get(key, [])))
        current_ref = str(fx.get('referee_name', '') or '').strip()
        recent5 = prev[:5]

        row = {k: fx[k] for k in KEYS}
        row.update({
            'h2h_n_l5': min(len(prev), 5),
            'h2h_goal_environment': _mean(prev, 'total_goals', 5),
            'h2h_btts_regime': _mean(prev, 'btts', 5),
            'h2h_booking_heat': _mean(prev, 'total_cards', 5),
            'h2h_foul_intensity': _mean(prev, 'total_fouls', 5),
            'h2h_style_conflict_index': _mean(prev, 'possession_diff_abs', 5),
            'h2h_home_win_rate_last5': _rate(prev, lambda r: r.get('home_goals', 0) > r.get('away_goals', 0), 5),
            'h2h_draw_rate_last5': _rate(prev, lambda r: r.get('home_goals', 0) == r.get('away_goals', 0), 5),
            'h2h_over25_rate_last5': _rate(prev, lambda r: r.get('total_goals', 0) >= 3, 5),
            'h2h_high_cards_rate_last5': _rate(prev, lambda r: r.get('total_cards', 0) >= 5, 5),
            'h2h_same_referee_overlap': int(bool(current_ref) and any(str(r.get('referee_name', '') or '').strip() == current_ref for r in recent5)),
            'h2h_same_referee_count_l5': sum(1 for r in recent5 if str(r.get('referee_name', '') or '').strip() == current_ref) if current_ref else 0,
        })
        out_rows.append(row)

        history[key].append({
            'total_goals': _safe_float(fx.get('total_goals', 0.0), 0.0),
            'btts': _safe_int(fx.get('btts', 0), 0),
            'total_cards': _safe_float(fx.get('total_cards', 0.0), 0.0),
            'total_fouls': _safe_float(fx.get('total_fouls', 0.0), 0.0),
            'possession_diff_abs': _safe_float(fx.get('possession_diff_abs', 0.0), 0.0),
            'home_goals': _safe_float(fx.get('home_goals', 0.0), 0.0),
            'away_goals': _safe_float(fx.get('away_goals', 0.0), 0.0),
            'referee_name': current_ref,
        })

    out = pd.DataFrame(out_rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=PURPOSE)
    parser.add_argument('--fixtures-csv', default=str(NORMALIZED_FILES['fixtures_master']))
    parser.add_argument('--team-stats-csv', default=str(NORMALIZED_FILES['match_team_stats']))
    parser.add_argument('--output-csv', default=str(TARGET_PATH))
    args = parser.parse_args()
    df = build_h2h_regime_features(args.fixtures_csv, args.team_stats_csv, args.output_csv)
    print(f'WROTE: {args.output_csv} rows={len(df)} cols={len(df.columns)}')


if __name__ == '__main__':
    main()
