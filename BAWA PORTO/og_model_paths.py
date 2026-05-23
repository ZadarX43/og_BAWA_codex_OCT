from __future__ import annotations

from pathlib import Path


def _market_aliases(market: str) -> list[str]:
    """Canonical aliases for model bundle filenames (without version suffix)."""
    m = str(market).strip().lower()
    alias_map = {
        # Core models
        "btts": ["btts"],
        "ftr": ["ftr"],
        "over25": ["over25", "ou25"],   # prefer over25, fallback alias ou25
        "ou25": ["over25", "ou25"],

        # Side models (legacy v2 allowed only where explicitly enabled)
        "home_fts": ["home_fts"],
        "away_fts": ["away_fts"],
        "home_ge2": ["home_ge2"],
        "away_ge2": ["away_ge2"],
        "home_ge3": ["home_ge3"],
        "away_ge3": ["away_ge3"],
    }
    return alias_map.get(m, [m])


def resolve_market_bundle_path(
    modelstore: Path,
    league_tag: str,
    market: str,
    *,
    prefer: str = "v3",
    allow_v2: bool = False,
    engine: str | None = None,
) -> dict:
    """Resolve a model bundle path with stable alias/version rules.

    Returns a dict with legacy + canonical keys for compatibility:
      - path
      - version / version_loaded
      - alias / alias_used
      - exists
      - tried / candidates

    IMPORTANT:
      - Core markets should call this with allow_v2=False (strict V3-only).
      - Side-model workflows may call allow_v2=True where legacy bundles are valid.
    """
    prefer = str(prefer).lower().strip()
    if prefer not in {"v3", "v2"}:
        prefer = "v3"

    ldir = Path(modelstore) / str(league_tag)
    aliases = _market_aliases(market)

    eng = str(engine or "").strip().lower()
    subdir = None
    if eng in {"xgb", "xgboost"}:
        subdir = "xgb"
    elif eng in {"cat", "catboost"}:
        subdir = "cat"

    versions = [prefer]
    if allow_v2 and prefer != "v2":
        versions.append("v2")
    elif allow_v2 and prefer == "v2":
        versions.append("v3")

    candidates: list[Path] = []
    if subdir:
        for ver in versions:
            for alias in aliases:
                candidates.append(ldir / subdir / f"{alias}_{ver}.pkl")
    for ver in versions:
        for alias in aliases:
            candidates.append(ldir / f"{alias}_{ver}.pkl")

    tried = [str(x) for x in candidates]

    for p in candidates:
        if p.exists():
            stem = p.stem  # e.g. "over25_v3"
            try:
                alias_used, version_loaded = stem.rsplit("_", 1)
            except ValueError:
                alias_used, version_loaded = stem, None

            return {
                "path": p,
                "version": version_loaded,
                "version_loaded": version_loaded,
                "alias": alias_used,
                "alias_used": alias_used,
                "exists": True,
                "tried": tried,
                "candidates": tried,
            }

    return {
        "path": None,
        "version": None,
        "version_loaded": None,
        "alias": None,
        "alias_used": None,
        "exists": False,
        "tried": tried,
        "candidates": tried,
    }
