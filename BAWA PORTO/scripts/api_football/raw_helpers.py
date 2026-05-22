from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .utils import build_fixture_key


def read_jsonl_payloads(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    payloads: list[dict[str, Any]] = []
    with source.open('r', encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payloads.append(json.loads(line))
    return payloads


def iter_fixture_rows(payloads: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        for item in payload.get('response', []) or []:
            rows.append(item)
    return rows


def fixture_base(item: dict[str, Any]) -> dict[str, Any]:
    fixture = item.get('fixture') or {}
    league = item.get('league') or {}
    teams = item.get('teams') or {}
    home = teams.get('home') or {}
    away = teams.get('away') or {}
    match_ts = pd.to_datetime(fixture.get('date'), errors='coerce')
    match_date = match_ts.date().isoformat() if pd.notna(match_ts) else ''
    return {
        'fixture_id': to_int(fixture.get('id')),
        'fixture_key': build_fixture_key(match_date, home.get('name'), away.get('name')),
        'league': league.get('name') or '',
        'league_id': to_int(league.get('id')),
        'season': to_int(league.get('season')),
        'match_date': match_date,
        'home_team_id': to_int(home.get('id')),
        'away_team_id': to_int(away.get('id')),
        'home_team_name': home.get('name') or '',
        'away_team_name': away.get('name') or '',
        'kickoff_ts_utc': fixture.get('date') or '',
        'status': ((fixture.get('status') or {}).get('short') or ''),
        'venue_id': to_int(((fixture.get('venue') or {}).get('id'))),
        'venue_name': ((fixture.get('venue') or {}).get('name') or ''),
        'referee_name': fixture.get('referee') or '',
    }


def to_int(value: Any, default: int = 0) -> int:
    if value is None or value == '':
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip().replace('%', '')
    if text == '' or text.lower() == 'none':
        return default
    try:
        return int(float(text))
    except Exception:
        return default


def to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == '':
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace('%', '')
    if text == '' or text.lower() == 'none':
        return default
    try:
        return float(text)
    except Exception:
        return default
