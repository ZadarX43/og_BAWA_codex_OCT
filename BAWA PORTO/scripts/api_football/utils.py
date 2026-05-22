from __future__ import annotations

import re
import unicodedata
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

from .paths import ensure_dirs


def normalize_name(value: object) -> str:
    text = '' if value is None else str(value)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return re.sub(r'_+', '_', text).strip('_')


def build_fixture_key(match_date: object, home_team_name: object, away_team_name: object) -> str:
    return f"{pd.to_datetime(match_date, errors='coerce').date()}_{normalize_name(home_team_name)}_{normalize_name(away_team_name)}"


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b not in (0, 0.0, None) else 0.0


def write_stub_csv(path: Path, columns: Iterable[str]) -> None:
    ensure_dirs()
    pd.DataFrame(columns=list(columns)).to_csv(path, index=False)


def fuzzy_join_window(match_date: object) -> tuple[pd.Timestamp, pd.Timestamp]:
    ts = pd.to_datetime(match_date, errors='coerce')
    return ts - timedelta(days=1), ts + timedelta(days=1)



def chunk_list(items: list[int], chunk_size: int) -> list[list[int]]:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
