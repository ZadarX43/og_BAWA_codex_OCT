#!/usr/bin/env python3
"""deploy_presets.py

Season-aware deploy preset switchboard.

Buckets:
- A = Jul–Sep
- B = Oct–Dec
- C = Jan–May

These presets are conservative caps derived from the 2022/23–2023/24 leave-one-out volume baselines.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, Optional


@dataclass(frozen=True)
class DeployPreset:
    name: str
    bucket: str  # A/B/C

    # Market caps (max rows kept after gating)
    max_ou25: int
    max_btts: int
    max_ftr: int
    max_tg15: int
    max_tg25: int

    # Optional tightenings (used mainly in Season B)
    ftr_margin_min: Optional[float] = None
    btts_yes_labels: Optional[str] = None  # comma-separated labels (if present in data)

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "bucket": self.bucket,
            "max_ou25": int(self.max_ou25),
            "max_btts": int(self.max_btts),
            "max_ftr": int(self.max_ftr),
            "max_tg15": int(self.max_tg15),
            "max_tg25": int(self.max_tg25),
            "ftr_margin_min": None if self.ftr_margin_min is None else float(self.ftr_margin_min),
            "btts_yes_labels": self.btts_yes_labels,
        }


def _to_date(x) -> date:
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, str):
        return datetime.strptime(x.strip(), "%Y-%m-%d").date()
    raise TypeError(f"Unsupported date type: {type(x)}")


def season_bucket(d: date | datetime | str) -> str:
    """Return season bucket A/B/C based on month."""
    dd = _to_date(d)
    m = int(dd.month)
    if m in (7, 8, 9):
        return "A"
    if m in (10, 11, 12):
        return "B"
    # Jan–May (C). We treat June as C by default.
    return "C"


# Conservative caps + minimal tightening (Season B)
_PRESETS: Dict[str, DeployPreset] = {
    "A": DeployPreset(
        name="Season A (Jul–Sep)",
        bucket="A",
        max_ou25=72,
        max_btts=32,
        max_ftr=34,
        max_tg15=20,
        max_tg25=8,
    ),
    "B": DeployPreset(
        name="Season B (Oct–Dec) — TG25 OFF",
        bucket="B",
        max_ou25=63,
        max_btts=20,
        max_ftr=31,
        max_tg15=23,
        max_tg25=0,
        ftr_margin_min=0.06,
        btts_yes_labels="VERY_STRONG_YES",
    ),
    "C": DeployPreset(
        name="Season C (Jan–May)",
        bucket="C",
        max_ou25=67,
        max_btts=18,
        max_ftr=31,
        max_tg15=27,
        max_tg25=10,
    ),
}


def get_preset(bucket: str) -> DeployPreset:
    b = str(bucket or "").strip().upper()
    if b == "AUTO":
        raise ValueError("Use get_preset_auto(date_from, date_to) for AUTO")
    if b not in _PRESETS:
        raise ValueError(f"Unknown season bucket '{bucket}'. Expected A/B/C.")
    return _PRESETS[b]


def get_preset_auto(date_from: date | datetime | str, date_to: Optional[date | datetime | str] = None) -> DeployPreset:
    """Choose preset based on date_from (primary)."""
    b = season_bucket(date_from)
    return get_preset(b)


def list_presets() -> Dict[str, Dict[str, object]]:
    return {k: v.as_dict() for k, v in _PRESETS.items()}
