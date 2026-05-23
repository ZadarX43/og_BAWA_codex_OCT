#!/usr/bin/env python3
"""signal_layers.py

League-aware banding for Over 2.5 and BTTS probabilities.

Produces categorical labels:
  signal_over25 ∈ {VERY_STRONG_OVER, STRONG_OVER, NEUTRAL, STRONG_UNDER, VERY_STRONG_UNDER}
  signal_btts   ∈ {VERY_STRONG_YES, STRONG_YES, NEUTRAL, WEAK_NO, STRONG_NO}

Design goals:
- Prefer per-league mid thresholds from ModelStore/<LeagueTag>_market_thresholds.json (trainer output)
- Use adaptive (std-based) band widths so bands don’t collapse on tight distributions
- If banding collapses (missing YES/NO or OVER/UNDER), fall back to quantile pseudo-bands
  so downstream “bands mode” is never impossible.

This module is intentionally lightweight (pandas + numpy only).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Over25Band = Literal[
    "VERY_STRONG_OVER",
    "STRONG_OVER",
    "NEUTRAL",
    "STRONG_UNDER",
    "VERY_STRONG_UNDER",
]

BttsBand = Literal[
    "VERY_STRONG_YES",
    "STRONG_YES",
    "NEUTRAL",
    "WEAK_NO",
    "STRONG_NO",
]


@dataclass(frozen=True)
class Over25Thresholds:
    mid: float
    over_strong: float = 0.10
    over_very_strong: float = 0.20
    under_strong: float = 0.10
    under_very_strong: float = 0.20


@dataclass(frozen=True)
class BttsThresholds:
    mid: float
    yes_strong: float = 0.15
    yes_very_strong: float = 0.30
    no_strong: float = 0.15
    no_very_strong: float = 0.30


# ---------------------------------------------------------------------------
# ModelStore thresholds (preferred mid source of truth)
# ---------------------------------------------------------------------------


_MODELSTORE_DIR = Path(__file__).resolve().parent / "ModelStore"

# ---------------------------------------------------------------------------
# Optional per-league band config (learned cutoffs)
# ---------------------------------------------------------------------------

_DEF_BANDCFG_CANDIDATES = (
    "{tag}_signal_bands.json",
    "{tag}_signal_band_config.json",
    "{tag}/signal_bands.json",
    "{tag}/signal_band_config.json",
)

# Simple in-process cache to avoid repeated disk reads during per-league loops.
# Keyed by league tag.
_BANDCFG_CACHE: Dict[str, Dict[str, Any]] = {}


def _clip01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return float("nan")
    if not np.isfinite(x):
        return float("nan")
    return float(min(0.99, max(0.01, x)))



def _load_signal_band_config(league_name: str) -> Dict[str, Any]:
    """Load a per-league signal band config from ModelStore.

    This is optional and only used when present.

    Expected shapes (any of these are accepted):
      - {"over25": {...}, "btts": {...}}
      - {"markets": {"over25": {...}, "btts": {...}}}
      - {"leagues": {"<LeagueTag>": {"over25": {...}, "btts": {...}}}}

    Returns {} if missing/unreadable.
    """
    try:
        tag = _league_tag(league_name)

        # Cache hit
        try:
            cached = _BANDCFG_CACHE.get(tag)
            if isinstance(cached, dict):
                return cached
        except Exception:
            pass

        for tpl in _DEF_BANDCFG_CANDIDATES:
            p = _MODELSTORE_DIR / tpl.format(tag=tag)
            if not p.exists():
                continue
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    _BANDCFG_CACHE[tag] = payload
                    return payload
            except Exception:
                continue

        # Cache the miss so we don't repeatedly scan disk.
        _BANDCFG_CACHE[tag] = {}
        return {}
    except Exception:
        return {}


def _pick_league_cfg(band_config: Optional[Dict[str, Any]], league_name: str) -> Optional[Dict[str, Any]]:
    """Pick the best band_config payload for a given league.

    Supports:
      - None
      - a per-league payload (already for this league)
      - {"leagues": {...}} payload
      - {"<LeagueTag>": {...}} payload
    """
    if not isinstance(band_config, dict) or not band_config:
        return None

    tag = _league_tag(league_name)

    # 1) Nested under 'leagues'
    leagues_obj = band_config.get("leagues")
    if isinstance(leagues_obj, dict):
        v = leagues_obj.get(tag)
        if isinstance(v, dict):
            return v
        v2 = leagues_obj.get(str(league_name))
        if isinstance(v2, dict):
            return v2

    # 2) Direct tag key
    v3 = band_config.get(tag)
    if isinstance(v3, dict):
        return v3

    # 3) Direct league name key
    v4 = band_config.get(str(league_name))
    if isinstance(v4, dict):
        return v4

    # 4) Assume the payload itself is a market config dict
    return band_config


def _market_cfg(band_cfg: Optional[Dict[str, Any]], market: str) -> Optional[Dict[str, Any]]:
    if not isinstance(band_cfg, dict) or not band_cfg:
        return None
    m = str(market).strip().lower()

    mk = band_cfg.get("markets")
    if isinstance(mk, dict):
        v = mk.get(m)
        if isinstance(v, dict):
            return v

    v2 = band_cfg.get(m)
    if isinstance(v2, dict):
        return v2

    return None


def _league_tag(league_name: str) -> str:
    return str(league_name).strip().replace(" ", "_")


def _load_market_thresholds_flat(league_name: str) -> Dict[str, float]:
    """Load ModelStore/<LeagueTag>_market_thresholds.json.

    Supports both:
      - flat mapping: {"btts": 0.55, "over25": 0.47, ...}
      - nested mapping: {"markets": {"btts": {"threshold": 0.55}, ...}}

    Returns {} if missing/unreadable.
    """
    try:
        tag = _league_tag(league_name)
        p = _MODELSTORE_DIR / f"{tag}_market_thresholds.json"
        if not p.exists():
            return {}

        payload = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {}

        mk = payload.get("markets")
        if isinstance(mk, dict):
            out: Dict[str, float] = {}
            for mkt, info in mk.items():
                if isinstance(info, dict) and ("threshold" in info):
                    try:
                        out[str(mkt)] = float(info["threshold"])
                    except Exception:
                        continue
            return out

        out2: Dict[str, float] = {}
        for k, v in payload.items():
            try:
                out2[str(k)] = float(v)
            except Exception:
                continue
        return out2
    except Exception:
        return {}


def _resolve_band_mid(
    league_name: str,
    market: str,
    p: pd.Series,
    default_mid: float,
) -> float:
    """Resolve a *band mid* for a market.

    IMPORTANT: `*_market_thresholds.json` primarily stores **decision thresholds**.
    Those must NOT be used as band mids.

    We only use a JSON value if it is explicitly named as a mid, e.g.:
      - mid_over25 / over25_mid / band_mid_over25
      - mid_btts   / btts_mid   / band_mid_btts

    Otherwise we use the median of the current probability distribution.
    """
    # 1) Explicit mid keys (preferred if present)
    th = _load_market_thresholds_flat(league_name)
    m = str(market).strip().lower()
    cand_keys = [
        f"mid_{m}",
        f"{m}_mid",
        f"band_mid_{m}",
        f"band_{m}_mid",
    ]
    for k in cand_keys:
        v = th.get(k)
        try:
            fv = float(v)
            if np.isfinite(fv):
                return float(min(0.99, max(0.01, fv)))
        except Exception:
            continue

    # 2) Distribution median (default)
    try:
        s = pd.to_numeric(p, errors="coerce").astype(float)
        s = s[np.isfinite(s)]
        if not s.empty:
            return float(min(0.99, max(0.01, float(s.median()))))
    except Exception:
        pass

    # 3) Hard fallback
    return float(min(0.99, max(0.01, float(default_mid))))


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def _finite_series(p: pd.Series) -> pd.Series:
    s = pd.to_numeric(p, errors="coerce").astype(float)
    return s[np.isfinite(s)]


def _finite_std(p: pd.Series) -> float:
    try:
        s = _finite_series(p)
        if s.empty:
            return 0.0
        return float(s.std(ddof=0))
    except Exception:
        return 0.0


def _q(p: pd.Series, q: float) -> float:
    try:
        s = _finite_series(p)
        if s.empty:
            return float("nan")
        return float(s.quantile(q))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Quantile fallback banding (pseudo-bands)
# ---------------------------------------------------------------------------


def _band_from_quantiles_over25(p: pd.Series) -> pd.Series:
    p = pd.to_numeric(p, errors="coerce").clip(0, 1)
    q05, q25, q75, q95 = _q(p, 0.05), _q(p, 0.25), _q(p, 0.75), _q(p, 0.95)
    out = pd.Series("NEUTRAL", index=p.index, dtype="string")
    out = out.mask(p <= q25, "STRONG_UNDER")
    out = out.mask(p <= q05, "VERY_STRONG_UNDER")
    out = out.mask(p >= q75, "STRONG_OVER")
    out = out.mask(p >= q95, "VERY_STRONG_OVER")
    return out


def _band_from_quantiles_btts(p: pd.Series) -> pd.Series:
    p = pd.to_numeric(p, errors="coerce").clip(0, 1)
    q05, q25, q75, q95 = _q(p, 0.05), _q(p, 0.25), _q(p, 0.75), _q(p, 0.95)
    out = pd.Series("NEUTRAL", index=p.index, dtype="string")
    out = out.mask(p <= q25, "WEAK_NO")
    out = out.mask(p <= q05, "STRONG_NO")
    out = out.mask(p >= q75, "STRONG_YES")
    out = out.mask(p >= q95, "VERY_STRONG_YES")
    return out


def _score_from_band_over25(band: pd.Series) -> pd.Series:
    m = {
        "VERY_STRONG_OVER": 2,
        "STRONG_OVER": 1,
        "NEUTRAL": 0,
        "STRONG_UNDER": -1,
        "VERY_STRONG_UNDER": -2,
    }
    return band.astype("string").map(m).fillna(0).astype(int)


def _score_from_band_btts(band: pd.Series) -> pd.Series:
    m = {
        "VERY_STRONG_YES": 2,
        "STRONG_YES": 1,
        "NEUTRAL": 0,
        "WEAK_NO": -1,
        "STRONG_NO": -2,
    }
    return band.astype("string").map(m).fillna(0).astype(int)


# ---------------------------------------------------------------------------
# Adaptive banding (mid + sigma)
# ---------------------------------------------------------------------------


def _grade_over25_series(p: pd.Series, league_name: str, band_cfg: Optional[Dict[str, Any]] = None) -> Tuple[pd.Series, pd.Series]:
    """Return (band, score) for over25.

    Uses ModelStore mid threshold and sigma-based widths.
    Falls back to quantiles if the result collapses.
    """
    p = pd.to_numeric(p, errors="coerce").clip(0, 1)

    # Optional per-league learned band config (absolute cutoffs preferred).
    cfg = _market_cfg(band_cfg, "over25")

    # Mid: allow cfg override; else distribution median.
    mid = _resolve_band_mid(league_name, "over25", p, default_mid=0.55)
    if isinstance(cfg, dict):
        for k in ("mid", "band_mid", "mid_over25", "over25_mid"):
            if k in cfg:
                vv = _clip01(cfg.get(k))
                if np.isfinite(vv):
                    mid = float(vv)
                    break

    # If explicit absolute thresholds exist, use them.
    # Expected keys (any accepted):
    #   t_over_strong / t_strong_over
    #   t_over_very   / t_very_over
    #   t_under_strong / t_strong_under
    #   t_under_very   / t_very_under
    t_over = t_very_over = t_under = t_very_under = None
    if isinstance(cfg, dict):
        def _get_any(*keys: str) -> float:
            for kk in keys:
                if kk in cfg:
                    vv = _clip01(cfg.get(kk))
                    if np.isfinite(vv):
                        return float(vv)
            return float("nan")

        t_over = _get_any("t_over_strong", "t_strong_over")
        t_very_over = _get_any("t_over_very", "t_very_over", "t_very_strong_over")
        t_under = _get_any("t_under_strong", "t_strong_under")
        t_very_under = _get_any("t_under_very", "t_very_under", "t_very_strong_under")

        # If not present, allow delta-style configuration (relative to mid)
        if not np.isfinite(t_over) or not np.isfinite(t_under):
            d_str = _get_any("over_strong", "strong_delta", "over_strong_delta")
            d_v = _get_any("over_very_strong", "very_delta", "over_very_delta")
            d_us = _get_any("under_strong", "under_strong_delta")
            d_uv = _get_any("under_very_strong", "under_very_delta")

            if np.isfinite(d_str):
                if not np.isfinite(d_v):
                    d_v = float(min(0.20, max(0.02, 2.0 * float(d_str))))
                if not np.isfinite(d_us):
                    d_us = float(d_str)
                if not np.isfinite(d_uv):
                    d_uv = float(min(0.20, max(0.02, 2.0 * float(d_us))))

                t_over = _clip01(mid + float(d_str))
                t_very_over = _clip01(mid + float(d_v))
                t_under = _clip01(mid - float(d_us))
                t_very_under = _clip01(mid - float(d_uv))

    # Fall back to adaptive sigma-based widths when config is not usable.
    if (t_over is None) or (not np.isfinite(t_over)) or (t_under is None) or (not np.isfinite(t_under)):
        sigma = _finite_std(p)
        # widths adapt to distribution tightness
        # NOTE: use small floors so VERY_STRONG bands can exist even when the model distribution is tight
        w_strong = float(min(0.10, max(0.01, 1.0 * sigma)))
        w_very = float(min(0.20, max(0.02, 2.0 * sigma)))

        t_over = float(min(1.0, mid + w_strong))
        t_very_over = float(min(1.0, mid + w_very))
        t_under = float(max(0.0, mid - w_strong))
        t_very_under = float(max(0.0, mid - w_very))

    # Sanity ordering
    t_over = float(min(1.0, max(0.0, t_over)))
    t_very_over = float(min(1.0, max(t_over, t_very_over)))
    t_under = float(min(1.0, max(0.0, t_under)))
    t_very_under = float(min(t_under, max(0.0, t_very_under)))

    band = pd.Series("NEUTRAL", index=p.index, dtype="string")
    band = band.mask(p >= t_over, "STRONG_OVER")
    band = band.mask(p >= t_very_over, "VERY_STRONG_OVER")
    band = band.mask(p <= t_under, "STRONG_UNDER")
    band = band.mask(p <= t_very_under, "VERY_STRONG_UNDER")

    try:
        counts = band.value_counts(dropna=False)
        has_over = int(counts.get("STRONG_OVER", 0) + counts.get("VERY_STRONG_OVER", 0))
        has_under = int(counts.get("STRONG_UNDER", 0) + counts.get("VERY_STRONG_UNDER", 0))
        neutral_rate = float(counts.get("NEUTRAL", 0) / max(len(band), 1))
        very_over = int(counts.get("VERY_STRONG_OVER", 0))
        very_under = int(counts.get("VERY_STRONG_UNDER", 0))
        # If VERY_STRONG bands are vanishingly small, band-mode becomes unusable (esp. for EL-like leagues).
        # In that case fall back to quantile pseudo-bands which guarantee meaningful tails.
        n_tot = max(int(len(band)), 1)
        min_vs = max(5, int(round(0.005 * n_tot)))  # at least 5 rows or 0.5%

        if (
            (counts.shape[0] < 3)
            or (has_over == 0)
            or (has_under == 0)
            or (very_over < min_vs)
            or (very_under < min_vs)
            or (neutral_rate > 0.95)
        ):
            band = _band_from_quantiles_over25(p)
    except Exception:
        band = _band_from_quantiles_over25(p)

    score = _score_from_band_over25(band)
    return band, score


def _grade_btts_series(p: pd.Series, league_name: str, band_cfg: Optional[Dict[str, Any]] = None) -> Tuple[pd.Series, pd.Series]:
    """Return (band, score) for BTTS.

    Uses ModelStore mid threshold and sigma-based widths.
    Falls back to quantiles if the result collapses (missing yes/no sides).
    """
    p = pd.to_numeric(p, errors="coerce").clip(0, 1)

    cfg = _market_cfg(band_cfg, "btts")

    mid = _resolve_band_mid(league_name, "btts", p, default_mid=0.40)
    if isinstance(cfg, dict):
        for k in ("mid", "band_mid", "mid_btts", "btts_mid"):
            if k in cfg:
                vv = _clip01(cfg.get(k))
                if np.isfinite(vv):
                    mid = float(vv)
                    break

    # Absolute thresholds preferred.
    # Expected keys (any accepted):
    #   t_yes_strong / t_strong_yes
    #   t_yes_very   / t_very_yes
    #   t_no_weak    / t_weak_no
    #   t_no_strong  / t_strong_no
    t_yes_strong = t_yes_very = t_no_weak = t_no_strong = None
    if isinstance(cfg, dict):
        def _get_any(*keys: str) -> float:
            for kk in keys:
                if kk in cfg:
                    vv = _clip01(cfg.get(kk))
                    if np.isfinite(vv):
                        return float(vv)
            return float("nan")

        t_yes_strong = _get_any("t_yes_strong", "t_strong_yes")
        t_yes_very = _get_any("t_yes_very", "t_very_yes", "t_very_strong_yes")
        t_no_weak = _get_any("t_no_weak", "t_weak_no")
        t_no_strong = _get_any("t_no_strong", "t_strong_no")

        # Delta-style fallback
        if (not np.isfinite(t_yes_strong)) or (not np.isfinite(t_no_weak)):
            d_str = _get_any("yes_strong", "strong_delta", "yes_strong_delta")
            d_v = _get_any("yes_very_strong", "very_delta", "yes_very_delta")
            d_nw = _get_any("no_strong", "no_weak_delta", "no_strong_delta")
            d_nv = _get_any("no_very_strong", "no_strong_delta2", "no_very_delta")

            if np.isfinite(d_str):
                if not np.isfinite(d_v):
                    d_v = float(min(0.30, max(0.02, 2.0 * float(d_str))))
                if not np.isfinite(d_nw):
                    d_nw = float(d_str)
                if not np.isfinite(d_nv):
                    d_nv = float(min(0.30, max(0.02, 2.0 * float(d_nw))))

                t_yes_strong = _clip01(mid + float(d_str))
                t_yes_very = _clip01(mid + float(d_v))
                t_no_weak = _clip01(mid - float(d_nw))
                t_no_strong = _clip01(mid - float(d_nv))

    if (t_yes_strong is None) or (not np.isfinite(t_yes_strong)) or (t_no_weak is None) or (not np.isfinite(t_no_weak)):
        sigma = _finite_std(p)
        w_strong = float(min(0.15, max(0.01, 1.0 * sigma)))
        w_very = float(min(0.30, max(0.02, 2.0 * sigma)))

        t_yes_strong = float(min(1.0, mid + w_strong))
        t_yes_very = float(min(1.0, mid + w_very))
        t_no_weak = float(max(0.0, mid - w_strong))
        t_no_strong = float(max(0.0, mid - w_very))

    # Sanity ordering
    t_yes_strong = float(min(1.0, max(0.0, t_yes_strong)))
    t_yes_very = float(min(1.0, max(t_yes_strong, t_yes_very)))
    t_no_weak = float(min(1.0, max(0.0, t_no_weak)))
    t_no_strong = float(min(t_no_weak, max(0.0, t_no_strong)))

    band = pd.Series("NEUTRAL", index=p.index, dtype="string")
    band = band.mask(p >= t_yes_strong, "STRONG_YES")
    band = band.mask(p >= t_yes_very, "VERY_STRONG_YES")
    band = band.mask(p <= t_no_weak, "WEAK_NO")
    band = band.mask(p <= t_no_strong, "STRONG_NO")

    try:
        counts = band.value_counts(dropna=False)
        has_yes = int(counts.get("STRONG_YES", 0) + counts.get("VERY_STRONG_YES", 0))
        has_no = int(counts.get("WEAK_NO", 0) + counts.get("STRONG_NO", 0))
        neutral_rate = float(counts.get("NEUTRAL", 0) / max(len(band), 1))
        very_yes = int(counts.get("VERY_STRONG_YES", 0))
        no_mass = int(counts.get("WEAK_NO", 0) + counts.get("STRONG_NO", 0))
        # Same idea as over25: if tails are too thin, fall back to quantiles.
        n_tot = max(int(len(band)), 1)
        min_vs_yes = max(5, int(round(0.005 * n_tot)))  # at least 5 rows or 0.5%
        min_no_mass = max(5, int(round(0.005 * n_tot)))

        if (
            (counts.shape[0] < 3)
            or (has_yes == 0)
            or (has_no == 0)
            or (very_yes < min_vs_yes)
            or (no_mass < min_no_mass)
            or (neutral_rate > 0.95)
        ):
            band = _band_from_quantiles_btts(p)
    except Exception:
        band = _band_from_quantiles_btts(p)

    score = _score_from_band_btts(band)
    return band, score


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _resolve_prob_col(df: pd.DataFrame, preferred: str, fallbacks: Tuple[str, ...]) -> Optional[str]:
    if preferred and preferred in df.columns:
        return preferred
    for c in fallbacks:
        if c in df.columns:
            return c
    return None


def attach_over25_btts_signals(
    df: pd.DataFrame,
    *,
    league_name: Optional[str] = None,
    over25_col: str = "prob_over25_v2",
    btts_col: str = "prob_btts_v2",
    league_col: str = "league_name",
    band_config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Attach signal_over25/signal_btts (+ *_score) to a dataframe.

    Column resolution:
      Over: prob_over25_v2 → prob_over25 → adjusted_over25_confidence → over25_confidence
      BTTS: prob_btts_v2   → prob_btts   → adjusted_btts_confidence   → btts_confidence

    If league_name is provided, we grade the whole frame using that league.
    Otherwise, we try to infer a league column from {league_col, league, League, league_name}.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df

    out = df.copy()

    # If no band_config is provided, attempt to load a per-league band config from ModelStore.
    # This is optional and safe (returns {} if missing).
    loaded_band_cfg: Optional[Dict[str, Any]] = None
    try:
        if band_config is None and league_name is not None:
            cfg0 = _load_signal_band_config(str(league_name))
            loaded_band_cfg = cfg0 if isinstance(cfg0, dict) and cfg0 else None
    except Exception:
        loaded_band_cfg = None

    # league resolution
    if league_name is not None:
        leagues = pd.Series([str(league_name)] * len(out), index=out.index)
    else:
        cand_cols = [league_col, "league", "League", "league_name"]
        use_col = next((c for c in cand_cols if c in out.columns), None)
        if use_col is None:
            leagues = pd.Series(["generic"] * len(out), index=out.index)
        else:
            leagues = out[use_col].astype("string").fillna("generic").astype(str)

    out["signal_over25"] = "NEUTRAL"
    out["signal_over25_score"] = 0
    out["signal_btts"] = "NEUTRAL"
    out["signal_btts_score"] = 0

    ocol = _resolve_prob_col(out, over25_col, ("prob_over25_v2", "prob_over25", "adjusted_over25_confidence", "over25_confidence"))
    bcol = _resolve_prob_col(out, btts_col, ("prob_btts_v2", "prob_btts", "adjusted_btts_confidence", "btts_confidence"))

    # -------------------------------------------------------------------
    # V3 ALLMARKETS compatibility:
    # If we don't have canonical prob columns (prob_over25_v2 / prob_btts_v2),
    # derive them from `model_p_for_bookie` (or p_model/confidence) plus the
    # side pick encoded in `selection` or `bookie_pick`.
    #
    # Semantics:
    #   - Over25 banding expects P(OVER25). If the row is an UNDER pick, we
    #     convert using complement: P(OVER25) = 1 - P(UNDER25).
    #   - BTTS banding expects P(YES). If the row is a NO pick, we convert:
    #     P(YES) = 1 - P(NO).
    # This lets downstream gates use labels even when upstream only provides
    # "probability for the chosen side".
    # -------------------------------------------------------------------
    try:
        pick_col = "selection" if "selection" in out.columns else ("bookie_pick" if "bookie_pick" in out.columns else None)
        mkt_col = "market" if "market" in out.columns else None

        base_p = out.get("model_p_for_bookie", None)
        if base_p is None:
            base_p = out.get("p_model", None)
        if base_p is None:
            base_p = out.get("confidence", None)

        base_p = pd.to_numeric(base_p, errors="coerce") if base_p is not None else pd.Series(np.nan, index=out.index)

        if mkt_col is not None:
            mkt = out[mkt_col].astype("string").fillna("").str.strip().str.lower()
        else:
            mkt = pd.Series("", index=out.index, dtype="string")

        if pick_col is not None:
            pick = out[pick_col].astype("string").fillna("").str.strip().str.upper()
        else:
            pick = pd.Series("", index=out.index, dtype="string")

        # --- Derive P(OVER25) ---
        if ocol is None:
            is_ou = mkt.isin(["ou25", "over25", "under25"])
            is_over = pick.isin(["OVER25", "OVER 2.5", "OVER", "O2.5", "O25"])
            is_under = pick.isin(["UNDER25", "UNDER 2.5", "UNDER", "U2.5", "U25"])

            p_over = pd.Series(np.nan, index=out.index, dtype=float)
            # If the pick is explicit OVER -> base_p is already P(OVER25)
            p_over = p_over.mask(is_ou & is_over, base_p)
            # If the pick is explicit UNDER -> convert using complement
            p_over = p_over.mask(is_ou & is_under, 1.0 - base_p)

            out["__prob_over25_derived"] = p_over.clip(0.0, 1.0)
            if out["__prob_over25_derived"].notna().any():
                ocol = "__prob_over25_derived"

        # --- Derive P(BTTS YES) ---
        if bcol is None:
            is_bt = mkt.isin(["btts", "btts_no"])
            is_yes = pick.isin(["YES", "Y", "GG", "BTTS_YES", "BTTS YES"])
            is_no = pick.isin(["NO", "N", "NG", "BTTS_NO", "BTTS NO"])

            p_yes = pd.Series(np.nan, index=out.index, dtype=float)
            # If the pick is YES -> base_p is already P(YES)
            p_yes = p_yes.mask(is_bt & is_yes, base_p)
            # If the pick is NO -> convert using complement
            p_yes = p_yes.mask(is_bt & is_no, 1.0 - base_p)

            out["__prob_btts_yes_derived"] = p_yes.clip(0.0, 1.0)
            if out["__prob_btts_yes_derived"].notna().any():
                bcol = "__prob_btts_yes_derived"
    except Exception:
        pass

    if ocol is not None:
        p_o = pd.to_numeric(out[ocol], errors="coerce").clip(0, 1)
        if league_name is not None:
            cfg_for_league = _pick_league_cfg(loaded_band_cfg or band_config, str(league_name)) or loaded_band_cfg or band_config
            band_o, score_o = _grade_over25_series(p_o, str(league_name), band_cfg=cfg_for_league)
            out["signal_over25"] = band_o
            out["signal_over25_score"] = score_o
        else:
            band_all = pd.Series("NEUTRAL", index=out.index, dtype="string")
            score_all = pd.Series(0, index=out.index, dtype=int)
            for lg in pd.Series(leagues).astype(str).fillna("generic").unique().tolist():
                m = pd.Series(leagues).astype(str) == str(lg)
                cfg_for_league = _pick_league_cfg(band_config, str(lg)) or _load_signal_band_config(str(lg))
                band_o, score_o = _grade_over25_series(p_o[m], str(lg), band_cfg=cfg_for_league)
                band_all.loc[m] = band_o
                score_all.loc[m] = score_o
            out["signal_over25"] = band_all
            out["signal_over25_score"] = score_all

    if bcol is not None:
        p_b = pd.to_numeric(out[bcol], errors="coerce").clip(0, 1)
        if league_name is not None:
            cfg_for_league = _pick_league_cfg(loaded_band_cfg or band_config, str(league_name)) or loaded_band_cfg or band_config
            band_b, score_b = _grade_btts_series(p_b, str(league_name), band_cfg=cfg_for_league)
            out["signal_btts"] = band_b
            out["signal_btts_score"] = score_b
        else:
            band_all = pd.Series("NEUTRAL", index=out.index, dtype="string")
            score_all = pd.Series(0, index=out.index, dtype=int)
            for lg in pd.Series(leagues).astype(str).fillna("generic").unique().tolist():
                m = pd.Series(leagues).astype(str) == str(lg)
                cfg_for_league = _pick_league_cfg(band_config, str(lg)) or _load_signal_band_config(str(lg))
                band_b, score_b = _grade_btts_series(p_b[m], str(lg), band_cfg=cfg_for_league)
                band_all.loc[m] = band_b
                score_all.loc[m] = score_b
            out["signal_btts"] = band_all
            out["signal_btts_score"] = score_all

    return out


def attach_signal_layers(
    df: pd.DataFrame,
    *,
    league_name: Optional[str] = None,
    over25_col: str = "prob_over25_v2",
    btts_col: str = "prob_btts_v2",
    league_col: str = "league_name",
    band_config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Back-compat wrapper."""
    return attach_over25_btts_signals(
        df,
        league_name=league_name,
        over25_col=over25_col,
        btts_col=btts_col,
        league_col=league_col,
        band_config=band_config,
    )
