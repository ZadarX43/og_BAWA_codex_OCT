from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .paths import ensure_dirs
from .utils import write_stub_csv


def build_csv_stub(path: Path, columns: Iterable[str], purpose: str, *, placeholder_row: bool = False) -> pd.DataFrame:
    ensure_dirs()
    cols = list(columns)
    if cols:
        write_stub_csv(path, cols)
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame([{'status': 'STUB', 'purpose': purpose}]) if placeholder_row else pd.DataFrame()
    df.to_csv(path, index=False)
    return df
