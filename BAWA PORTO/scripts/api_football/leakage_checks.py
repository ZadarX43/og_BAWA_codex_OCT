from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .paths import FEATURE_FILES, LIVE_DATASET_FILES


@dataclass
class LeakageCheckResult:
    ok: bool
    message: str


def check_no_live_columns_in_prematch(df: pd.DataFrame) -> LeakageCheckResult:
    banned_prefixes = ('live_',)
    offenders = [c for c in df.columns if c.startswith(banned_prefixes)]
    if offenders:
        return LeakageCheckResult(False, f'Prematch table contains live columns: {offenders}')
    return LeakageCheckResult(True, 'No live-prefixed columns found in prematch table.')


def check_known_pre_kickoff_flags(df: pd.DataFrame, flag_columns: list[str]) -> LeakageCheckResult:
    missing = [c for c in flag_columns if c not in df.columns]
    if missing:
        return LeakageCheckResult(False, f'Missing leakage-control flags: {missing}')
    return LeakageCheckResult(True, 'Leakage-control flags present.')


def run_foundation_leakage_checks() -> list[LeakageCheckResult]:
    results: list[LeakageCheckResult] = []
    enriched = FEATURE_FILES['api_enriched_fixture_features']
    if enriched.exists() and enriched.stat().st_size > 0:
        df = pd.read_csv(enriched)
        results.append(check_no_live_columns_in_prematch(df))
    else:
        results.append(LeakageCheckResult(True, 'Enriched fixture features not built yet; no prematch leakage check executed.'))
    for minute, path in LIVE_DATASET_FILES.items():
        if path.exists() and path.stat().st_size > 0:
            df_live = pd.read_csv(path)
            if 'live_minute' not in df_live.columns:
                results.append(LeakageCheckResult(False, f'Live dataset {path.name} missing live_minute column.'))
            else:
                results.append(LeakageCheckResult(True, f'Live dataset {path.name} contains live_minute column.'))
        else:
            results.append(LeakageCheckResult(True, f'Live dataset {path.name} not built yet.'))
    return results


def main() -> None:
    for result in run_foundation_leakage_checks():
        label = 'PASS' if result.ok else 'FAIL'
        print(f'[{label}] {result.message}')


if __name__ == '__main__':
    main()
