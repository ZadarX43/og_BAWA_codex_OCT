from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd


def _safe_div(num: float, den: float) -> float:
    if not den:
        return 0.0
    return float(num) / float(den)


def _sum(records: list[dict], key: str, n: int) -> float:
    sample = records[:n]
    return sum(float(r.get(key, 0.0) or 0.0) for r in sample) if sample else 0.0


def _mean(records: list[dict], key: str, n: int) -> float:
    sample = records[:n]
    return (_sum(records, key, n) / len(sample)) if sample else 0.0


def _position_group(position: str) -> str:
    pos = str(position or '').strip().upper()
    return {'G': 'Goalkeeper', 'D': 'Defender', 'M': 'Midfielder', 'F': 'Forward'}.get(pos, 'Unknown')


def _norm_cap(series: pd.Series, cap: float) -> pd.Series:
    out = pd.to_numeric(series, errors='coerce').astype(float)
    return (out.clip(lower=0.0, upper=cap) / cap).fillna(0.0)


def _build_player_overlay_rows(player_stats: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    merged = player_stats.merge(
        fixtures[[
            'fixture_id', 'fixture_key', 'league', 'season', 'match_date', 'kickoff_ts_utc',
            'home_team_id', 'away_team_id', 'home_team_name', 'away_team_name'
        ]],
        on='fixture_id',
        how='left',
    )
    merged['kickoff_ts_utc'] = pd.to_datetime(merged['kickoff_ts_utc'], errors='coerce', utc=True)
    merged = merged.sort_values(['kickoff_ts_utc', 'fixture_id', 'team_id', 'player_id']).reset_index(drop=True)

    history: dict[int, list[dict]] = defaultdict(list)
    rows: list[dict] = []
    for _, row in merged.iterrows():
        player_id = int(row['player_id'])
        prev = list(reversed(history.get(player_id, [])))
        mins5 = _sum(prev, 'minutes', 5)
        passes_total5 = _sum(prev, 'passes_total', 5)
        passes_accurate5 = _sum(prev, 'passes_accurate', 5)
        duels_total5 = _sum(prev, 'duels_total', 5)
        duels_won5 = _sum(prev, 'duels_won', 5)
        started5 = _sum(prev, 'started_flag', 5)

        rec = {
            'fixture_id': int(row['fixture_id']),
            'fixture_key': row.get('fixture_key', ''),
            'league': row.get('league', ''),
            'season': row.get('season', ''),
            'match_date': row.get('match_date', ''),
            'team_id': int(row['team_id']),
            'player_id': player_id,
            'player_name': row.get('player_name', ''),
            'position': row.get('position', ''),
            'position_group': _position_group(row.get('position', '')),
            'player_form_rating_l5': round(_mean(prev, 'rating', 5), 4),
            'player_minutes_avg_l5': round(_mean(prev, 'minutes', 5), 4),
            'player_started_share_l5': round(_safe_div(started5, min(len(prev), 5) or 1), 4),
            'goals_per90_l5': round(_safe_div(_sum(prev, 'goals', 5) * 90.0, mins5), 4),
            'assists_per90_l5': round(_safe_div(_sum(prev, 'assists', 5) * 90.0, mins5), 4),
            'shots_per90_l5': round(_safe_div(_sum(prev, 'shots_total', 5) * 90.0, mins5), 4),
            'shots_on_target_per90_l5': round(_safe_div(_sum(prev, 'shots_on_target', 5) * 90.0, mins5), 4),
            'key_passes_per90_l5': round(_safe_div(_sum(prev, 'passes_key', 5) * 90.0, mins5), 4),
            'tackles_per90_l5': round(_safe_div(_sum(prev, 'tackles', 5) * 90.0, mins5), 4),
            'interceptions_per90_l5': round(_safe_div(_sum(prev, 'interceptions', 5) * 90.0, mins5), 4),
            'dribbled_past_per90_l5': round(_safe_div(_sum(prev, 'dribbled_past', 5) * 90.0, mins5), 4),
            'pass_accuracy_pct_l5': round(_safe_div(passes_accurate5 * 100.0, passes_total5), 4),
            'duel_win_pct_l5': round(_safe_div(duels_won5 * 100.0, duels_total5), 4),
        }
        rows.append(rec)
        history[player_id].append(row.to_dict())
    return pd.DataFrame(rows)


def _compute_player_quality(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pos = out['position_group'].astype('string').fillna('Unknown')

    common = 0.35 * _norm_cap(out['player_form_rating_l5'], 8.5) + 0.15 * _norm_cap(out['player_minutes_avg_l5'], 90.0) + 0.10 * _norm_cap(out['player_started_share_l5'], 1.0)
    attack = (
        0.25 * _norm_cap(out['goals_per90_l5'], 1.0)
        + 0.20 * _norm_cap(out['shots_per90_l5'], 4.0)
        + 0.20 * _norm_cap(out['shots_on_target_per90_l5'], 2.0)
        + 0.15 * _norm_cap(out['assists_per90_l5'], 0.7)
        + 0.20 * _norm_cap(out['key_passes_per90_l5'], 3.0)
    )
    control = (
        0.45 * _norm_cap(out['pass_accuracy_pct_l5'], 100.0)
        + 0.25 * _norm_cap(out['key_passes_per90_l5'], 3.0)
        + 0.15 * _norm_cap(out['assists_per90_l5'], 0.7)
        + 0.15 * _norm_cap(out['duel_win_pct_l5'], 100.0)
    )
    defend = (
        0.30 * _norm_cap(out['tackles_per90_l5'], 4.0)
        + 0.25 * _norm_cap(out['interceptions_per90_l5'], 3.0)
        + 0.20 * _norm_cap(out['duel_win_pct_l5'], 100.0)
        + 0.15 * _norm_cap(out['pass_accuracy_pct_l5'], 100.0)
        + 0.10 * (1.0 - _norm_cap(out['dribbled_past_per90_l5'], 4.0))
    )

    score = common.copy()
    score += (pos == 'Forward').astype(float) * (0.40 * attack + 0.10 * control)
    score += (pos == 'Midfielder').astype(float) * (0.22 * attack + 0.28 * control + 0.10 * defend)
    score += (pos == 'Defender').astype(float) * (0.35 * defend + 0.10 * control)
    score += (pos == 'Goalkeeper').astype(float) * (0.25 * _norm_cap(out['pass_accuracy_pct_l5'], 100.0) + 0.25 * _norm_cap(out['player_form_rating_l5'], 8.5))
    score += (pos == 'Unknown').astype(float) * (0.15 * attack + 0.15 * defend + 0.10 * control)

    out['player_quality_score_l5'] = (score * 100.0).round(4)
    out['player_form_tier'] = pd.cut(
        out['player_quality_score_l5'],
        bins=[-1, 45, 60, 75, 200],
        labels=['Weak', 'Stable', 'Strong', 'Elite'],
    ).astype('string')

    out['player_quality_rank_in_position'] = (
        out.groupby(['league', 'position_group'])['player_quality_score_l5']
        .rank(method='min', ascending=False)
        .astype(int)
    )
    out['player_quality_percentile_in_position'] = (
        1.0
        - out.groupby(['league', 'position_group'])['player_quality_score_l5']
        .rank(method='average', pct=True, ascending=True)
    ).round(4)
    return out


def build_player_form_quality_overlay(
    fixtures_csv: str,
    player_stats_csv: str,
    lineups_csv: str,
    output_csv: str,
) -> pd.DataFrame:
    fixtures = pd.read_csv(fixtures_csv)
    player_stats = pd.read_csv(player_stats_csv)
    lineups = pd.read_csv(lineups_csv)

    fixtures['kickoff_ts_utc'] = pd.to_datetime(fixtures['kickoff_ts_utc'], errors='coerce', utc=True)
    player_overlay = _build_player_overlay_rows(player_stats, fixtures)
    player_overlay = _compute_player_quality(player_overlay)

    lineup_base = lineups.copy()
    lineup_base['is_starting_xi'] = pd.to_numeric(lineup_base['is_starting_xi'], errors='coerce').fillna(0).astype(int)
    lineup_base = lineup_base[lineup_base['is_starting_xi'].eq(1)].copy()
    if lineup_base.empty:
        lineup_base = player_stats[player_stats['started_flag'].eq(1)][['fixture_id', 'team_id', 'player_id']].copy()
        lineup_base['is_starting_xi'] = 1

    starters = lineup_base.merge(
        player_overlay,
        on=['fixture_id', 'team_id', 'player_id'],
        how='left',
    )
    starters['position_group'] = starters['position_group'].astype('string').fillna('Unknown')
    team_scores = (
        starters.groupby(['fixture_id', 'team_id', 'league'], as_index=False)
        .agg(
            starting_xi_team_quality_score=('player_quality_score_l5', 'mean'),
            starting_xi_attack_quality_score=('player_quality_score_l5', lambda s: s[starters.loc[s.index, 'position_group'].isin(['Forward', 'Midfielder'])].mean() if len(s) else 0.0),
            starting_xi_defensive_quality_score=('player_quality_score_l5', lambda s: s[starters.loc[s.index, 'position_group'].isin(['Defender', 'Goalkeeper'])].mean() if len(s) else 0.0),
            starting_xi_avg_form_rating_l5=('player_form_rating_l5', 'mean'),
        )
    )
    team_scores[['starting_xi_attack_quality_score', 'starting_xi_defensive_quality_score', 'starting_xi_avg_form_rating_l5']] = team_scores[[
        'starting_xi_attack_quality_score', 'starting_xi_defensive_quality_score', 'starting_xi_avg_form_rating_l5'
    ]].fillna(0.0)
    team_scores['starting_xi_team_quality_rank_league'] = team_scores.groupby('league')['starting_xi_team_quality_score'].rank(method='min', ascending=False).astype(int)
    team_scores['starting_xi_team_quality_percentile_league'] = team_scores.groupby('league')['starting_xi_team_quality_score'].rank(method='average', pct=True, ascending=True)
    team_scores['starting_xi_team_quality_percentile_league'] = (1.0 - team_scores['starting_xi_team_quality_percentile_league']).round(4)

    fixture_team = fixtures[['fixture_id', 'home_team_id', 'away_team_id']].drop_duplicates()
    home = team_scores.merge(fixture_team, on='fixture_id', how='left')
    home_side = home[home['team_id'].eq(home['home_team_id'])].copy()
    away_side = home[home['team_id'].eq(home['away_team_id'])].copy()
    matchup = home_side[['fixture_id', 'starting_xi_team_quality_score']].rename(columns={'starting_xi_team_quality_score': 'home_starting_xi_team_quality_score'}).merge(
        away_side[['fixture_id', 'starting_xi_team_quality_score']].rename(columns={'starting_xi_team_quality_score': 'away_starting_xi_team_quality_score'}),
        on='fixture_id',
        how='outer',
    )

    out = player_overlay.merge(team_scores, on=['fixture_id', 'team_id', 'league'], how='left').merge(matchup, on='fixture_id', how='left')
    out = out.merge(fixture_team, on='fixture_id', how='left')
    out['player_team_side'] = out.apply(lambda r: 'HOME' if int(r['team_id']) == int(r['home_team_id']) else 'AWAY', axis=1)
    out['opponent_starting_xi_team_quality_score'] = out.apply(
        lambda r: r['away_starting_xi_team_quality_score'] if r['player_team_side'] == 'HOME' else r['home_starting_xi_team_quality_score'],
        axis=1,
    )
    out['starting_xi_quality_edge'] = (out['starting_xi_team_quality_score'] - out['opponent_starting_xi_team_quality_score']).round(4)

    keep = [
        'fixture_id', 'fixture_key', 'league', 'season', 'match_date', 'team_id', 'player_id', 'player_name', 'position', 'position_group',
        'player_form_rating_l5', 'player_minutes_avg_l5', 'player_started_share_l5', 'goals_per90_l5', 'assists_per90_l5',
        'shots_per90_l5', 'shots_on_target_per90_l5', 'key_passes_per90_l5', 'tackles_per90_l5', 'interceptions_per90_l5',
        'dribbled_past_per90_l5', 'pass_accuracy_pct_l5', 'duel_win_pct_l5', 'player_quality_score_l5', 'player_form_tier',
        'player_quality_rank_in_position', 'player_quality_percentile_in_position', 'starting_xi_team_quality_score',
        'starting_xi_attack_quality_score', 'starting_xi_defensive_quality_score', 'starting_xi_avg_form_rating_l5',
        'starting_xi_team_quality_rank_league', 'starting_xi_team_quality_percentile_league', 'player_team_side',
        'opponent_starting_xi_team_quality_score', 'starting_xi_quality_edge',
    ]
    out = out[keep].copy()
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def _default_path(league_tag: str, season: int) -> Path:
    return Path('data_sources/api_football/features/player_events') / f'player_form_quality_overlay__{league_tag}__{season}.csv'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build player form + positional quality overlay for player-events research.')
    parser.add_argument('--league-tag', required=True)
    parser.add_argument('--season', type=int, required=True)
    parser.add_argument('--fixtures-csv', default='')
    parser.add_argument('--player-stats-csv', default='')
    parser.add_argument('--lineups-csv', default='')
    parser.add_argument('--output-csv', default='')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    normalized = Path('data_sources/api_football/normalized')
    fixtures_csv = args.fixtures_csv or str(normalized / f'fixtures_master__{args.league_tag}__{args.season}.csv')
    player_stats_csv = args.player_stats_csv or str(normalized / f'match_player_stats__{args.league_tag}__{args.season}.csv')
    lineups_csv = args.lineups_csv or str(normalized / f'lineups__{args.league_tag}__{args.season}.csv')
    output_csv = args.output_csv or str(_default_path(args.league_tag, args.season))
    df = build_player_form_quality_overlay(fixtures_csv, player_stats_csv, lineups_csv, output_csv)
    print(f'WROTE: {output_csv}')
    print(f'rows: {len(df)} | fixtures: {df["fixture_id"].nunique() if len(df) else 0}')
    if len(df):
        print('avg_player_quality_score_l5:', round(float(df['player_quality_score_l5'].mean()), 4))
        print('avg_starting_xi_team_quality_score:', round(float(df['starting_xi_team_quality_score'].mean()), 4))


if __name__ == '__main__':
    main()
