from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd


def slugify(value: str) -> str:
    text = unicodedata.normalize('NFKD', str(value)).encode('ascii', 'ignore').decode('ascii')
    text = text.lower().replace('&', ' and ')
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return re.sub(r'_+', '_', text).strip('_')


def norm_fixture(date_value: str, home: str, away: str) -> str:
    return f"{str(date_value)[:10]}_{slugify(home)}_{slugify(away)}"


def team_slug_match(left: str, right: str) -> bool:
    left_slug = slugify(left)
    right_slug = slugify(right)
    if not left_slug or not right_slug:
        return False
    if left_slug == right_slug or left_slug in right_slug or right_slug in left_slug:
        return True
    left_tokens = set(left_slug.split('_'))
    right_tokens = set(right_slug.split('_'))
    if not left_tokens or not right_tokens:
        return False
    overlap = len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))
    return overlap >= 0.5


def classify_source_family(path: Path) -> str:
    text = str(path)
    if 'BACKTEST_3Y__' in text:
        return 'FULL_ESTATE_BACKTEST'
    if 'WEAKEST_FAILED_LEG_AUDIT__3Y_WEEKLY' in text:
        return 'WEAKEST_FAILED_DETAIL'
    if 'SLIP_WALKFORWARD_AUDIT__3Y_WEEKLY' in text:
        return 'WEEKLY_WALKFORWARD'
    return 'OTHER_ARCHIVE'


def candidate_paths(roots: list[str]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for root_str in roots:
        root = Path(root_str)
        if not root.exists():
            continue
        patterns = ['*.csv']
        for pat in patterns:
            for path in root.rglob(pat):
                if path in seen:
                    continue
                name = path.name.lower()
                if 'snapshot_proxy' in name or '__backup__' in str(path):
                    continue
                if not (
                    name.startswith('ranked_board_')
                    or 'detail' in name
                    or 'profile' in name
                    or 'timing' in name
                    or 'report' in name
                ):
                    continue
                seen.add(path)
                out.append(path)
    return out


def load_matchable(path: Path) -> pd.DataFrame:
    try:
        header = pd.read_csv(path, nrows=0).columns.tolist()
    except Exception:
        return pd.DataFrame()
    cols = set(header)
    fixture_col = 'fixture_key' if 'fixture_key' in cols else None
    date_col = 'match_date' if 'match_date' in cols else ('window_date_from' if 'window_date_from' in cols else None)
    home_col = 'home' if 'home' in cols else ('home_team_name' if 'home_team_name' in cols else None)
    away_col = 'away' if 'away' in cols else ('away_team_name' if 'away_team_name' in cols else None)
    if not fixture_col and not (date_col and home_col and away_col):
        return pd.DataFrame()
    keep = [c for c in [fixture_col, date_col, home_col, away_col, 'market', 'selection', 'league', 'competition'] if c]
    try:
        df = pd.read_csv(path, usecols=lambda c: c in keep, low_memory=False)
    except Exception:
        return pd.DataFrame()
    if fixture_col and fixture_col in df.columns:
        df['fixture_key_norm'] = df[fixture_col].astype(str).str.replace('__', '_', regex=False).str.lower()
    else:
        df['fixture_key_norm'] = ''
    if date_col and home_col and away_col:
        df['date_norm'] = pd.to_datetime(df[date_col], errors='coerce').dt.strftime('%Y-%m-%d')
        df['home_norm'] = df[home_col].astype(str)
        df['away_norm'] = df[away_col].astype(str)
    else:
        df['date_norm'] = ''
        df['home_norm'] = ''
        df['away_norm'] = ''
    return df


def build(target_csv: str, output_csv: str, output_md: str, roots: list[str]) -> pd.DataFrame:
    targets = pd.read_csv(target_csv, low_memory=False)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    if targets.empty:
        empty = pd.DataFrame()
        empty.to_csv(output_csv, index=False)
        Path(output_md).write_text('# Ranked-Board Backfill Source Finder\n\nNo backfill targets matched.\n')
        return empty

    files = candidate_paths(roots)
    loaded = [(path, classify_source_family(path), load_matchable(path)) for path in files]
    rows: list[dict[str, object]] = []

    for _, t in targets.iterrows():
        fixture_key = str(t['fixture_key'])
        date = str(t['match_date'])[:10]
        home = str(t['home_team_name'])
        away = str(t['away_team_name'])
        fixture_norm = fixture_key.replace('__', '_').lower()
        hits: list[dict[str, object]] = []
        for path, family, df in loaded:
            if df.empty:
                continue
            sub = pd.DataFrame()
            if 'fixture_key_norm' in df.columns and df['fixture_key_norm'].astype(str).eq(fixture_norm).any():
                sub = df[df['fixture_key_norm'].astype(str) == fixture_norm].copy()
                match_kind = 'FIXTURE_KEY_EXACT'
            elif 'date_norm' in df.columns:
                date_match = df['date_norm'].astype(str).eq(date)
                team_match = df.apply(
                    lambda r: team_slug_match(home, r.get('home_norm', '')) and team_slug_match(away, r.get('away_norm', '')),
                    axis=1,
                ) if not df.empty else pd.Series(dtype=bool)
                mask = date_match & team_match
                if mask.any():
                    sub = df[mask].copy()
                    match_kind = 'DATE_TEAM_MATCH'
                else:
                    continue
            else:
                continue
            hits.append(
                {
                    'fixture_key': fixture_key,
                    'backfill_priority': t.get('backfill_priority', ''),
                    'goal_market_focus': t.get('goal_market_focus', ''),
                    'source_family': family,
                    'source_path': str(path),
                    'match_kind': match_kind,
                    'matched_rows': int(len(sub)),
                    'matched_markets': '|'.join(sorted(set(sub.get('market', pd.Series(dtype=str)).astype(str)))) if 'market' in sub.columns else '',
                    'source_leagues': '|'.join(sorted(set(sub.get('league', pd.Series(dtype=str)).astype(str)))) if 'league' in sub.columns else '',
                }
            )
        if hits:
            rows.extend(hits)
        else:
            rows.append(
                {
                    'fixture_key': fixture_key,
                    'backfill_priority': t.get('backfill_priority', ''),
                    'goal_market_focus': t.get('goal_market_focus', ''),
                    'source_family': 'NO_ALT_SOURCE_FOUND',
                    'source_path': '',
                    'match_kind': 'NONE',
                    'matched_rows': 0,
                    'matched_markets': '',
                    'source_leagues': str(t.get('league', '')),
                }
            )

    out = pd.DataFrame(rows)
    out = out.sort_values(['fixture_key', 'matched_rows', 'source_family'], ascending=[True, False, True])
    out.to_csv(output_csv, index=False)

    lines = [
        '# Ranked-Board Backfill Source Finder',
        '',
        '- Searches alternate archive families beyond the current weekly ranked-board roots.',
        '- This is the right next audit when the weekly walkforward windows do not preserve a recoverable ranked-board row.',
        '',
    ]
    summary = out.groupby('source_family', dropna=False).agg(rows=('fixture_key', 'size')).reset_index().sort_values('rows', ascending=False)
    lines.append('## Summary')
    for _, row in summary.iterrows():
        lines.append(f"- {row['source_family']} | rows={int(row['rows'])}")
    lines.append('')
    for fixture, sub in out.groupby('fixture_key', sort=False):
        lines.append(f'## {fixture}')
        for _, row in sub.iterrows():
            if row['source_family'] == 'NO_ALT_SOURCE_FOUND':
                lines.append(f"- no alternate archive-family match found for focus={row['goal_market_focus']}")
            else:
                lines.append(
                    f"- {row['source_family']} | kind={row['match_kind']} | rows={int(row['matched_rows'])} | markets={row['matched_markets'] or 'none'}"
                )
                lines.append(f"  path: {row['source_path']}")
        lines.append('')
    Path(output_md).write_text('\n'.join(lines) + '\n')
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Search alternate archive families for backfillable fixture-level history.')
    parser.add_argument('--target-csv', required=True)
    parser.add_argument('--root', action='append', required=True)
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--output-md', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    out = build(args.target_csv, args.output_csv, args.output_md, args.root)
    print(f'WROTE: {args.output_csv}')
    print(f'rows: {len(out)}')
