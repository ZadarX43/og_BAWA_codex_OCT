from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .paths import FEATURE_FILES, NORMALIZED_FILES
from .schema_contracts import FEATURE_SCHEMAS
from .scaffold import build_csv_stub
from .utils import safe_div

PURPOSE = 'Build prematch bookmaker, implied, drift, and disagreement features.'
TARGET_PATH = FEATURE_FILES['api_odds_features']


def build_stub() -> pd.DataFrame:
    return build_csv_stub(TARGET_PATH, FEATURE_SCHEMAS['api_odds_features'], PURPOSE, placeholder_row=False)


def _best(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors='coerce').dropna()
    return float(s.max()) if not s.empty else 0.0


def _mean(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors='coerce').dropna()
    return float(s.mean()) if not s.empty else 0.0


def _std(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors='coerce').dropna()
    return float(s.std(ddof=0)) if len(s) > 1 else 0.0


def build_odds_features(fixtures_csv: str, odds_csv: str, output_csv: str = str(TARGET_PATH)) -> pd.DataFrame:
    fixtures = pd.read_csv(fixtures_csv)
    odds = pd.read_csv(odds_csv)
    rows = []
    odds_by_fixture = {fid: df.copy() for fid, df in odds.groupby('fixture_id')} if not odds.empty else {}

    for _, fx in fixtures.iterrows():
        fo = odds_by_fixture.get(int(fx['fixture_id']), pd.DataFrame(columns=odds.columns))
        def filt(market: str, selection: str | None = None, line_hint: str | None = None):
            df = fo[fo['market_code'] == market]
            if selection is not None:
                df = df[df['selection_code'] == selection]
            if line_hint is not None:
                df = df[df['selection_name'].astype(str).str.contains(line_hint, regex=False) | df['line_value'].astype(str).str.contains(line_hint, regex=False)]
            return df

        home = filt('FTR', 'HOME')
        draw = filt('FTR', 'DRAW')
        away = filt('FTR', 'AWAY')
        over25 = filt('OU', None, 'Over 2.5')
        under25 = filt('OU', None, 'Under 2.5')
        btts_yes = filt('BTTS', 'YES')
        btts_no = filt('BTTS', 'NO')

        odds_home_best = _best(home['odds'])
        odds_draw_best = _best(draw['odds'])
        odds_away_best = _best(away['odds'])
        odds_over25_best = _best(over25['odds'])
        odds_under25_best = _best(under25['odds'])
        odds_btts_yes_best = _best(btts_yes['odds'])
        odds_btts_no_best = _best(btts_no['odds'])

        raw_home = safe_div(1.0, odds_home_best)
        raw_draw = safe_div(1.0, odds_draw_best)
        raw_away = safe_div(1.0, odds_away_best)
        raw_sum = raw_home + raw_draw + raw_away
        raw_ou25 = safe_div(1.0, odds_over25_best)
        raw_u25 = safe_div(1.0, odds_under25_best)
        raw_ou_sum = raw_ou25 + raw_u25
        raw_btts_yes = safe_div(1.0, odds_btts_yes_best)
        raw_btts_no = safe_div(1.0, odds_btts_no_best)
        raw_btts_sum = raw_btts_yes + raw_btts_no

        rows.append({
            'fixture_id': int(fx['fixture_id']),
            'fixture_key': fx['fixture_key'],
            'league': fx['league'],
            'league_id': int(fx['league_id']),
            'season': int(fx['season']),
            'match_date': fx['match_date'],
            'home_team_id': int(fx['home_team_id']),
            'away_team_id': int(fx['away_team_id']),
            'home_team_name': fx['home_team_name'],
            'away_team_name': fx['away_team_name'],
            'odds_home_win_best': odds_home_best,
            'odds_draw_best': odds_draw_best,
            'odds_away_win_best': odds_away_best,
            'odds_over25_best': odds_over25_best,
            'odds_under25_best': odds_under25_best,
            'odds_btts_yes_best': odds_btts_yes_best,
            'odds_btts_no_best': odds_btts_no_best,
            'odds_home_win_mean': _mean(home['odds']),
            'odds_draw_mean': _mean(draw['odds']),
            'odds_away_win_mean': _mean(away['odds']),
            'odds_over25_mean': _mean(over25['odds']),
            'odds_btts_yes_mean': _mean(btts_yes['odds']),
            'bookie_home_prob_norm': safe_div(raw_home, raw_sum),
            'bookie_draw_prob_norm': safe_div(raw_draw, raw_sum),
            'bookie_away_prob_norm': safe_div(raw_away, raw_sum),
            'bookie_over25_prob_norm': safe_div(raw_ou25, raw_ou_sum),
            'bookie_btts_yes_prob_norm': safe_div(raw_btts_yes, raw_btts_sum),
            'home_odds_std': _std(home['odds']),
            'draw_odds_std': _std(draw['odds']),
            'away_odds_std': _std(away['odds']),
            'over25_odds_std': _std(over25['odds']),
            'btts_yes_odds_std': _std(btts_yes['odds']),
            'home_market_disagreement': safe_div(_std(home['odds']), _mean(home['odds'])),
            'draw_market_disagreement': safe_div(_std(draw['odds']), _mean(draw['odds'])),
            'away_market_disagreement': safe_div(_std(away['odds']), _mean(away['odds'])),
            'over25_market_disagreement': safe_div(_std(over25['odds']), _mean(over25['odds'])),
            'btts_market_disagreement': safe_div(_std(btts_yes['odds']), _mean(btts_yes['odds'])),
            'home_odds_open': 0.0, 'home_odds_latest': odds_home_best, 'home_odds_drift': 0.0,
            'draw_odds_open': 0.0, 'draw_odds_latest': odds_draw_best, 'draw_odds_drift': 0.0,
            'away_odds_open': 0.0, 'away_odds_latest': odds_away_best, 'away_odds_drift': 0.0,
            'over25_odds_open': 0.0, 'over25_odds_latest': odds_over25_best, 'over25_odds_drift': 0.0,
            'btts_yes_odds_open': 0.0, 'btts_yes_odds_latest': odds_btts_yes_best, 'btts_yes_odds_drift': 0.0,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=FEATURE_SCHEMAS['api_odds_features'])
    else:
        df = df.reindex(columns=FEATURE_SCHEMAS['api_odds_features'])
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=PURPOSE)
    parser.add_argument('--write-stub', action='store_true', help='Write the scaffold output even though live transform logic is not implemented yet.')
    parser.add_argument('--fixtures-csv', default=str(NORMALIZED_FILES['fixtures_master']))
    parser.add_argument('--odds-csv', default=str(NORMALIZED_FILES['odds_prematch_long']))
    parser.add_argument('--output-csv', default=str(TARGET_PATH))
    args = parser.parse_args()
    if args.write_stub:
        df = build_stub()
        print(f'WROTE STUB: {TARGET_PATH} rows={len(df)}')
        return
    df = build_odds_features(args.fixtures_csv, args.odds_csv, args.output_csv)
    print(f'WROTE: {args.output_csv} rows={len(df)}')


if __name__ == '__main__':
    main()
