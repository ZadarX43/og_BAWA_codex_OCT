from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def slugify(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_") or "unknown"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = []
    for col in df.columns:
        text = str(col).strip().lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        renamed.append(re.sub(r"_+", "_", text).strip("_"))
    out = df.copy()
    out.columns = renamed
    return out


def coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
      if col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def series_or_default(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return numerator / denominator


def percentile_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return pd.Series(50.0, index=series.index, dtype="float64")
    ranks = valid.rank(method="average", pct=True) * 100.0
    if not higher_is_better:
        ranks = 100.0 - ranks
    scored = pd.Series(50.0, index=series.index, dtype="float64")
    scored.loc[valid.index] = ranks.clip(0, 100)
    return scored


def weighted_average(frame: pd.DataFrame, components: list[tuple[str, bool, float]]) -> pd.Series:
    pieces = []
    for column, higher_is_better, weight in components:
        scored = percentile_score(frame[column], higher_is_better=higher_is_better)
        pieces.append(scored * float(weight))
    if not pieces:
        return pd.Series(50.0, index=frame.index, dtype="float64")
    total_weight = sum(weight for _, _, weight in components) or 1.0
    return sum(pieces) / total_weight


def blend_with_neutral(score: pd.Series, confidence_multiplier: pd.Series | float) -> pd.Series:
    return 50.0 + ((pd.to_numeric(score, errors="coerce") - 50.0) * confidence_multiplier)


def team_confidence_multiplier(matches_played: pd.Series) -> pd.Series:
    matches = pd.to_numeric(matches_played, errors="coerce").fillna(0)
    return (matches / 18.0).clip(lower=0.35, upper=1.0)


def team_confidence_label(matches_played: float) -> str:
    if matches_played >= 24:
        return "Strong sample"
    if matches_played >= 16:
        return "Trusted sample"
    if matches_played >= 8:
        return "Developing sample"
    return "Low sample"


def player_confidence_multiplier(minutes_played: pd.Series) -> pd.Series:
    minutes = pd.to_numeric(minutes_played, errors="coerce").fillna(0)
    return (minutes / 1500.0).clip(lower=0.25, upper=1.0)


def player_confidence_label(minutes_played: float) -> str:
    if minutes_played >= 1500:
        return "Strong sample"
    if minutes_played >= 900:
        return "Trusted sample"
    if minutes_played >= 600:
        return "Medium sample"
    if minutes_played >= 300:
        return "Low sample"
    return "Very low sample"


def score_to_int(value: float | pd.Series) -> int | pd.Series:
    if isinstance(value, pd.Series):
        return value.round().clip(lower=0, upper=100).astype(int)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 50
    return int(max(0, min(100, round(float(value)))))


def rating_band(score: float) -> str:
    if score >= 90:
        return "Elite"
    if score >= 80:
        return "Strong"
    if score >= 70:
        return "Positive"
    if score >= 55:
        return "Mixed"
    if score >= 40:
        return "Weak"
    return "Red Flag"


def market_lean(score: float) -> str:
    if score >= 85:
        return "Strong"
    if score >= 70:
        return "Positive"
    if score >= 55:
        return "Mixed"
    if score >= 40:
        return "Weak"
    return "Red Flag"
