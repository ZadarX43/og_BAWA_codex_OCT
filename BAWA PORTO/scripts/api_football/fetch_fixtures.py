from __future__ import annotations

import argparse

from .client import APIFootballClient, write_raw_json


DEFAULT_LEAGUES = [39, 140, 135, 78, 61, 94, 88, 144, 179, 203, 40, 42, 71]
DEFAULT_COMPLETED_STATUS = 'FT-AET-PEN'
DEFAULT_COMPLETED_OR_LIVE_STATUS = 'FT-AET-PEN-1H-HT-2H-ET-BT-P'


def parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(',') if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description='Fetch API-Football fixtures into raw JSONL files.')
    parser.add_argument('--league-ids', default='', help='Comma-separated league ids. Defaults to a starter set when omitted.')
    parser.add_argument('--season', required=True, type=int, help='Season year to fetch, e.g. 2024 or 2025.')
    parser.add_argument('--from-date', default='', help='Optional YYYY-MM-DD lower date bound.')
    parser.add_argument('--to-date', default='', help='Optional YYYY-MM-DD upper date bound.')
    parser.add_argument('--timezone', default='Europe/London')
    parser.add_argument('--status', default=DEFAULT_COMPLETED_STATUS, help='Fixture status filter. Use FT-AET-PEN for completed, or FT-AET-PEN-1H-HT-2H-ET-BT-P to include live.')
    parser.add_argument('--all-statuses', action='store_true', help='Omit the status parameter so scheduled, live, and finished fixtures are all returned.')
    parser.add_argument('--include-live', action='store_true', help='Shortcut to use the completed+live status set.')
    parser.add_argument('--max-pages', type=int, default=None)
    parser.add_argument('--sleep-seconds', type=float, default=None)
    parser.add_argument('--daily-cap', type=int, default=75000)
    args = parser.parse_args()

    leagues = parse_csv_ints(args.league_ids) if args.league_ids else DEFAULT_LEAGUES
    client = APIFootballClient(sleep_seconds=args.sleep_seconds, daily_cap=args.daily_cap)
    status = '' if args.all_statuses else (DEFAULT_COMPLETED_OR_LIVE_STATUS if args.include_live else args.status)

    for league_id in leagues:
        params = {
            'league': league_id,
            'season': args.season,
            'status': status,
            'from': args.from_date or None,
            'to': args.to_date or None,
            'timezone': args.timezone,
        }
        payload = client.get_json('/fixtures', params)
        rows = [payload]
        stem = f'fixtures__league_{league_id}__season_{args.season}'
        path = write_raw_json('fixtures', rows, stem=stem)
        total_results = int(payload.get('results') or 0)
        errors = payload.get('errors') or {}
        error_suffix = f" errors={errors}" if errors else ''
        print(f'WROTE RAW: {path} pages=1 league_id={league_id} results={total_results} status={status}{error_suffix}')


if __name__ == '__main__':
    main()
