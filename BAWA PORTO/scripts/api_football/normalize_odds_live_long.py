from __future__ import annotations

import argparse
import pandas as pd

from .paths import NORMALIZED_FILES
from .schema_contracts import NORMALIZED_SCHEMAS
from .scaffold import build_csv_stub

PURPOSE = 'Build normalized live odds table from API-Football raw live-odds payloads.'
TARGET_PATH = NORMALIZED_FILES['odds_live_long']


def build_stub() -> pd.DataFrame:
    return build_csv_stub(TARGET_PATH, NORMALIZED_SCHEMAS['odds_live_long'], PURPOSE, placeholder_row=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=PURPOSE)
    parser.add_argument('--write-stub', action='store_true', help='Write the scaffold output even though live transform logic is not implemented yet.')
    args = parser.parse_args()
    if not args.write_stub:
        raise SystemExit('Use --write-stub during the foundation pass. Live transformation logic is not implemented yet.')
    df = build_stub()
    print(f'WROTE STUB: {TARGET_PATH} rows={len(df)}')


if __name__ == '__main__':
    main()
