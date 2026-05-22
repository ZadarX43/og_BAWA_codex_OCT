from __future__ import annotations

import argparse
import pandas as pd

from .paths import REPORT_FILES
from .scaffold import build_csv_stub

PURPOSE = 'Write feature coverage audit for API-Football feature outputs.'
TARGET_PATH = REPORT_FILES['api_feature_coverage_report']


def build_stub() -> pd.DataFrame:
    return build_csv_stub(TARGET_PATH, [], PURPOSE, placeholder_row=True)


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
