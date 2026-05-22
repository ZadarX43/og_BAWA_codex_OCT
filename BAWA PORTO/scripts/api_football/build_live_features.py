from __future__ import annotations

import argparse
import pandas as pd

from .paths import FEATURE_FILES, LIVE_DATASET_FILES
from .schema_contracts import FEATURE_SCHEMAS
from .scaffold import build_csv_stub

PURPOSE = 'Build live snapshot features and minute-bucket datasets.'
TARGET_PATH = FEATURE_FILES['api_live_features']


def build_stub() -> pd.DataFrame:
    df = build_csv_stub(TARGET_PATH, FEATURE_SCHEMAS['api_live_features'], PURPOSE, placeholder_row=False)
    for minute, path in LIVE_DATASET_FILES.items():
        build_csv_stub(path, FEATURE_SCHEMAS['api_live_features'], f'{PURPOSE} Minute={minute}', placeholder_row=False)
    return df


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
