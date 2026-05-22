from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .paths import FEATURE_FILES, NORMALIZED_FILES
from .schema_contracts import FEATURE_SCHEMAS
from .scaffold import build_csv_stub

PURPOSE = 'Build rolling event timing, chaos, and volatility features.'
TARGET_PATH = FEATURE_FILES['api_event_features']


def build_stub() -> pd.DataFrame:
    return build_csv_stub(TARGET_PATH, FEATURE_SCHEMAS['api_event_features'], PURPOSE, placeholder_row=False)


def _mean(records, key, n):
    sample = records[:n]
    return sum(float(r.get(key, 0.0) or 0.0) for r in sample) / len(sample) if sample else 0.0


def _rate(records, pred, n):
    sample = records[:n]
    return sum(1 for r in sample if pred(r)) / len(sample) if sample else 0.0


def _team_event_metrics(records):
    records = list(records)
    return {
        'first_goal_rate_l10': _rate(records, lambda r: r.get('scored_first', 0) == 1, 10),
        'concede_first_rate_l10': _rate(records, lambda r: r.get('conceded_first', 0) == 1, 10),
        'late_goal_scored_rate_l10': _rate(records, lambda r: r.get('late_goal_scored', 0) == 1, 10),
        'late_goal_conceded_rate_l10': _rate(records, lambda r: r.get('late_goal_conceded', 0) == 1, 10),
        'red_card_rate_l20': _mean(records, 'red_cards', 20),
        'yellow_card_rate_l10': _mean(records, 'yellow_cards', 10),
        'goal_after_75_rate_l10': _rate(records, lambda r: r.get('goal_after_75', 0) == 1, 10),
        'concede_after_75_rate_l10': _rate(records, lambda r: r.get('concede_after_75', 0) == 1, 10),
        'chaos_index_l10': _mean(records, 'chaos_index', 10),
        'late_volatility_l10': _mean(records, 'late_volatility', 10),
    }


def build_event_features(fixtures_csv: str, events_csv: str, team_stats_csv: str, output_csv: str = str(TARGET_PATH)) -> pd.DataFrame:
    fixtures = pd.read_csv(fixtures_csv)
    events = pd.read_csv(events_csv)
    team_stats = pd.read_csv(team_stats_csv)
    fixtures['kickoff_ts_utc'] = pd.to_datetime(fixtures['kickoff_ts_utc'], errors='coerce', utc=True)

    team_pair = team_stats[['fixture_id', 'team_id', 'yellow_cards', 'red_cards']].copy()
    event_rows = []
    for _, fx in fixtures.sort_values(['kickoff_ts_utc', 'fixture_id']).iterrows():
        fx_events = events[events['fixture_id'] == fx['fixture_id']].sort_values(['minute', 'extra_minute', 'event_id'])
        goal_events = fx_events[fx_events['event_type'] == 'Goal']
        first_goal_team = int(goal_events.iloc[0]['team_id']) if not goal_events.empty else 0
        event_rows_by_team = {}
        for team_id, opp_team_id in [(int(fx['home_team_id']), int(fx['away_team_id'])), (int(fx['away_team_id']), int(fx['home_team_id']))]:
            team_fx = fx_events[fx_events['team_id'] == team_id]
            opp_fx = fx_events[fx_events['team_id'] == opp_team_id]
            yellow = team_pair[(team_pair['fixture_id'] == fx['fixture_id']) & (team_pair['team_id'] == team_id)]['yellow_cards']
            red = team_pair[(team_pair['fixture_id'] == fx['fixture_id']) & (team_pair['team_id'] == team_id)]['red_cards']
            yellow_val = int(yellow.iloc[0]) if not yellow.empty else 0
            red_val = int(red.iloc[0]) if not red.empty else 0
            goals_total = len(team_fx[team_fx['event_type'] == 'Goal'])
            opp_goals_total = len(opp_fx[opp_fx['event_type'] == 'Goal'])
            goals_after_75 = len(team_fx[(team_fx['event_type'] == 'Goal') & (team_fx['minute'] >= 75)])
            opp_goals_after_75 = len(opp_fx[(opp_fx['event_type'] == 'Goal') & (opp_fx['minute'] >= 75)])
            cards_after_75 = len(team_fx[(team_fx['event_type'] == 'Card') & (team_fx['minute'] >= 75)])
            subs_after_70 = len(team_fx[(team_fx['event_type'] == 'subst') & (team_fx['minute'] >= 70)])
            goal_minutes = list(team_fx[team_fx['event_type'] == 'Goal']['minute'])
            goal_timing_vol = float(pd.Series(goal_minutes).std()) if len(goal_minutes) >= 2 else 0.0
            event_rows_by_team[team_id] = {
                'team_id': team_id,
                'scored_first': int(first_goal_team == team_id),
                'conceded_first': int(first_goal_team == opp_team_id),
                'late_goal_scored': int(goals_after_75 > 0),
                'late_goal_conceded': int(opp_goals_after_75 > 0),
                'red_cards': red_val,
                'yellow_cards': yellow_val,
                'goal_after_75': int(goals_after_75 > 0),
                'concede_after_75': int(opp_goals_after_75 > 0),
                'chaos_index': (goals_total + opp_goals_total) * 0.30 + (yellow_val + red_val * 2) * 0.20 + red_val * 0.30 + (goals_after_75 + opp_goals_after_75) * 0.20,
                'late_volatility': goals_after_75 + cards_after_75 + subs_after_70,
                'card_volatility': yellow_val + (2 * red_val),
                'goal_timing_volatility': goal_timing_vol,
            }
        event_rows.append((fx, event_rows_by_team))

    history: dict[int, list[dict]] = defaultdict(list)
    out_rows = []
    for fx, team_map in event_rows:
        home_history = list(reversed(history.get(int(fx['home_team_id']), [])))
        away_history = list(reversed(history.get(int(fx['away_team_id']), [])))
        h = _team_event_metrics(home_history)
        a = _team_event_metrics(away_history)
        row = {key: fx[key] for key in FEATURE_SCHEMAS['api_event_features'] if key in fx.index}
        row.update({
            'home_first_goal_rate_l10': h['first_goal_rate_l10'], 'away_first_goal_rate_l10': a['first_goal_rate_l10'],
            'home_concede_first_rate_l10': h['concede_first_rate_l10'], 'away_concede_first_rate_l10': a['concede_first_rate_l10'],
            'home_late_goal_scored_rate_l10': h['late_goal_scored_rate_l10'], 'away_late_goal_scored_rate_l10': a['late_goal_scored_rate_l10'],
            'home_late_goal_conceded_rate_l10': h['late_goal_conceded_rate_l10'], 'away_late_goal_conceded_rate_l10': a['late_goal_conceded_rate_l10'],
            'home_red_card_rate_l20': h['red_card_rate_l20'], 'away_red_card_rate_l20': a['red_card_rate_l20'],
            'home_yellow_card_rate_l10': h['yellow_card_rate_l10'], 'away_yellow_card_rate_l10': a['yellow_card_rate_l10'],
            'home_goal_after_75_rate_l10': h['goal_after_75_rate_l10'], 'away_goal_after_75_rate_l10': a['goal_after_75_rate_l10'],
            'home_concede_after_75_rate_l10': h['concede_after_75_rate_l10'], 'away_concede_after_75_rate_l10': a['concede_after_75_rate_l10'],
            'home_chaos_index_l10': h['chaos_index_l10'], 'away_chaos_index_l10': a['chaos_index_l10'], 'combined_chaos_index_l10': (h['chaos_index_l10'] + a['chaos_index_l10']) / 2.0,
            'home_late_volatility_l10': h['late_volatility_l10'], 'away_late_volatility_l10': a['late_volatility_l10'], 'combined_late_volatility_l10': (h['late_volatility_l10'] + a['late_volatility_l10']) / 2.0,
            'card_volatility_l10': (h['yellow_card_rate_l10'] + 2*h['red_card_rate_l20']) + (a['yellow_card_rate_l10'] + 2*a['red_card_rate_l20']),
            'goal_timing_volatility_l10': (_mean(home_history, 'goal_timing_volatility', 10) + _mean(away_history, 'goal_timing_volatility', 10)) / 2.0,
        })
        out_rows.append(row)
        history[int(fx['home_team_id'])].append(team_map.get(int(fx['home_team_id']), {}))
        history[int(fx['away_team_id'])].append(team_map.get(int(fx['away_team_id']), {}))

    df = pd.DataFrame(out_rows)
    df = df.reindex(columns=FEATURE_SCHEMAS['api_event_features'])
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=PURPOSE)
    parser.add_argument('--write-stub', action='store_true', help='Write the scaffold output even though live transform logic is not implemented yet.')
    parser.add_argument('--fixtures-csv', default=str(NORMALIZED_FILES['fixtures_master']))
    parser.add_argument('--events-csv', default=str(NORMALIZED_FILES['match_events']))
    parser.add_argument('--team-stats-csv', default=str(NORMALIZED_FILES['match_team_stats']))
    parser.add_argument('--output-csv', default=str(TARGET_PATH))
    args = parser.parse_args()
    if args.write_stub:
        df = build_stub()
        print(f'WROTE STUB: {TARGET_PATH} rows={len(df)}')
        return
    df = build_event_features(args.fixtures_csv, args.events_csv, args.team_stats_csv, args.output_csv)
    print(f'WROTE: {args.output_csv} rows={len(df)}')


if __name__ == '__main__':
    main()
