from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TEAM_NAME_JOIN_MAP_CSV = ROOT / "configs" / "team_name_join_map.csv"
TEAM_NAME_JOIN_MAP_GENERATED_CSV = ROOT / "configs" / "team_name_join_map.generated.csv"


def _base_normalize(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii").lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def base_normalize_team_name(value: object) -> str:
    """Public wrapper for raw team-name normalization without alias lookup."""
    return _base_normalize(value)


def _load_join_map_csv(path: Path) -> dict[tuple[str, str], str]:
    mapping: dict[tuple[str, str], str] = {}
    if not path.exists():
        return mapping
    df = pd.read_csv(path)
    if df.empty:
        return mapping
    approved = df[df.get("approval_status", pd.Series(dtype=object)).astype(str).str.upper().eq("APPROVED")].copy()
    for _, row in approved.iterrows():
        tag = str(row.get("tag", "*") or "*").strip()
        api_name = _base_normalize(row.get("api_team_name"))
        fs_name = _base_normalize(row.get("fs_team_name"))
        if api_name and fs_name:
            mapping[(tag, api_name)] = fs_name
    return mapping


@lru_cache(maxsize=1)
def _load_join_map() -> dict[tuple[str, str], str]:
    mapping: dict[tuple[str, str], str] = {}
    # Auto-generated mappings come first; hand-maintained mappings are loaded
    # last so they remain the explicit override layer.
    for path in (TEAM_NAME_JOIN_MAP_GENERATED_CSV, TEAM_NAME_JOIN_MAP_CSV):
        mapping.update(_load_join_map_csv(path))
    return mapping


def normalize_team_name(value: object, tag: str | None = None) -> str:
    normalized = _base_normalize(value)
    if not normalized:
        return normalized
    mapping = _load_join_map()
    if tag is not None:
        scoped = mapping.get((str(tag), normalized))
        if scoped:
            return scoped
    return mapping.get(("*", normalized), normalized)
