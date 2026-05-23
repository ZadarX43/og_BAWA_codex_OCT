#!/usr/bin/env python3
"""odds_synth.py

Learn a robust mapping from a *known* side of a 2-way market (e.g. Over2.5 / BTTS YES)
into the *missing* side (e.g. Under2.5 / BTTS NO), using historical rows where
both sides exist.

Why this exists
---------------
Your OG pipeline often has forward-facing fixtures with only one side quoted
(e.g. Over2.5 but missing Under2.5). This module learns the conditional
relationship (per league, optionally per source) and synthesizes the missing odds
with hard safety gates.

Supported markets
-----------------
- OU25: Over 2.5 vs Under 2.5
- BTTS: YES vs NO

Primary method
--------------
A bin rulebook on the *known-side implied probability* (p = 1/odds), storing:
- conditional overround R = 1/O_known + 1/O_missing
- missing-odds quantiles (q10/q25/median/q75/q90)

At prediction time:
- find the bin by p_known
- estimate p_missing_hat = R_median - p_known
- odds_missing_hat = 1/p_missing_hat
- clamp odds_missing_hat into [q10,q90]
- return with confidence + reason codes; drop if low confidence

CLI
---
1) Fit tables (JSON)
   python odds_synth.py fit \
     --src "Matches/*/fd_odds_enriched.csv" \
     --out ModelStore/ODDS_SYNTH_TABLES.json \
     --group-by league \
     --markets ou25 btts \
     --known-side over yes \
     --bins 10 \
     --min-n 200

2) Backtest (time split)
   python odds_synth.py backtest \
     --src "Matches/*/fd_odds_enriched.csv" \
     --markets ou25 btts \
     --known-side over yes \
     --bins 10 \
     --min-n 200 \
     --holdout-weeks 10 \
     --out predictions_output/ODDS_SYNTH_BACKTEST.csv

3) Apply to future fixtures
   python odds_synth.py apply \
     --src Matches/England Premier League/fd_odds_enriched.csv \
     --tables ModelStore/ODDS_SYNTH_TABLES.json \
     --out Matches/England Premier League/fd_odds_enriched_synth.csv

Notes
-----
- We use league-level tables by default. If you want per-bookmaker/source tables,
  you can pass --group-by league,od_source and provide a usable source column.
- This code intentionally refuses to hallucinate when the bin support is weak.

"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass

from datetime import datetime, UTC

def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Column resolution
# -----------------------------

OU25_OVER_CANDS = [
    # OG / your schema
    "odds_ft_over25",
    "odds_over25",
    "odds_over_2_5",
    "odds_ft_over_2_5",
    # football-data.co.uk
    "P>2.5",
    "PC>2.5",
    "B365>2.5",
    "B365C>2.5",
    "Avg>2.5",
    "AvgC>2.5",
    "Max>2.5",
    "MaxC>2.5",
]

OU25_UNDER_CANDS = [
    # OG / your schema
    "odds_ft_under25",
    "odds_under25",
    "odds_under_2_5",
    "odds_ft_under_2_5",
    "odds_ft_u25",
    # football-data.co.uk
    "P<2.5",
    "PC<2.5",
    "B365<2.5",
    "B365C<2.5",
    "Avg<2.5",
    "AvgC<2.5",
    "Max<2.5",
    "MaxC<2.5",
]

BTTS_YES_CANDS = [
    # OG / your schema
    "odds_btts_yes",
    "odds_ft_btts_yes",
    # football-data.co.uk (common)
    "B365BTTSY",
    "B365CBTTSY",
    "BTTSY",
    "BTTS_Y",
    "Yes",
]

BTTS_NO_CANDS = [
    # OG / your schema
    "odds_btts_no",
    "odds_ft_btts_no",
    # football-data.co.uk (common)
    "B365BTTSN",
    "B365CBTTSN",
    "BTTSN",
    "BTTS_N",
    "No",
]


def _dedupe_list(xs: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


# Priority: use the same known-side you have forward (OG), and use FD as truth for the missing side.
# OU25 known side (deploy-time): prefer OG over25 first.
OU25_OVER_PRIORITY = _dedupe_list([
    "odds_ft_over25",
    "fd_odds_ft_over25",
] + OU25_OVER_CANDS)

# OU25 missing side truth (training): prefer FD under25 first.
OU25_UNDER_TRUTH_PRIORITY = _dedupe_list([
    "fd_odds_ft_under25",
    "odds_ft_under25",
] + OU25_UNDER_CANDS)

# If you ever fit the reverse direction (known=under), prefer OG under first.
OU25_UNDER_PRIORITY = _dedupe_list([
    "odds_ft_under25",
    "fd_odds_ft_under25",
] + OU25_UNDER_CANDS)

# BTTS known side (deploy-time): prefer OG yes first.
BTTS_YES_PRIORITY = _dedupe_list([
    "odds_btts_yes",
    "odds_ft_btts_yes",
] + BTTS_YES_CANDS)

# BTTS missing side truth (training): prefer OG no first (if present), else FD/no cols.
BTTS_NO_TRUTH_PRIORITY = _dedupe_list([
    "odds_btts_no",
    "odds_ft_btts_no",
] + BTTS_NO_CANDS)


def _pick_first_from(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    """Pick first existing column from a candidate list."""
    return _pick_first(df, cands)


def _get_known_odds_value(row: pd.Series, *, market: str, known_side: str) -> Tuple[float, str]:
    """Return (known_odds_value, used_column_name) using stable priority rules.

    This avoids accidentally learning from FD-over when deploy-time only has OG-over.
    """
    market = str(market).strip().lower()
    known_side = str(known_side).strip().lower()

    def _try_cols(cols: List[str]) -> Tuple[float, str]:
        for c in cols:
            if c not in row.index:
                continue
            try:
                v = float(pd.to_numeric(row.get(c, np.nan), errors="coerce"))
            except Exception:
                v = float("nan")
            if np.isfinite(v) and v > 1.0001:
                return (v, c)
        return (float("nan"), "")

    if market == "ou25":
        if known_side == "over":
            return _try_cols(OU25_OVER_PRIORITY)
        return _try_cols(OU25_UNDER_PRIORITY)

    # btts
    if known_side == "yes":
        return _try_cols(BTTS_YES_PRIORITY)
    return _try_cols(BTTS_NO_TRUTH_PRIORITY)

DATE_CANDS = [
    "match_date",
    "date_GMT",
    "Date",
    "date",
    "timestamp",
]

LEAGUE_CANDS = [
    "league",
    "League",
    "Div",
]

OD_SOURCE_CANDS = [
    "od_source",
    "odds_source",
    "odds_source_under25",
    "source",
]


def _pick_first(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    return None

# Helper: return all candidate columns that exist in df.columns
def _first_existing_candidates(df: pd.DataFrame, cands: List[str]) -> List[str]:
    return [c for c in cands if c in df.columns]


def _to_odds(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    # treat placeholders and nonsense as missing
    x = x.where(x > 1.0001, np.nan)
    return x


def _to_date_series(df: pd.DataFrame) -> pd.Series:
    # best-effort parse to date
    out = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    for c in DATE_CANDS:
        if c not in df.columns:
            continue
        try:
            cand = pd.to_datetime(df[c], errors="coerce", utc=True)
        except Exception:
            cand = pd.to_datetime(df[c].astype(str), errors="coerce", utc=True)
        out = out.fillna(cand)

    # If timestamp looks like unix seconds
    if out.isna().all() and "timestamp" in df.columns:
        try:
            cand = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="s", utc=True)
            out = out.fillna(cand)
        except Exception:
            pass

    return out.dt.date


def _infer_league_from_path(path: str) -> str:
    # Matches/<League>/... pattern
    parts = Path(path).parts
    if "Matches" in parts:
        i = parts.index("Matches")
        if i + 1 < len(parts):
            return str(parts[i + 1])
    return "UNKNOWN_LEAGUE"


# -----------------------------
# Core tables
# -----------------------------

@dataclass
class BinStats:
    n: int
    p_lo: float
    p_hi: float
    R_median: float
    R_q25: float
    R_q75: float
    R_p05: float
    R_p95: float
    miss_med: float
    miss_q10: float
    miss_q25: float
    miss_q75: float
    miss_q90: float


def _quantile(x: pd.Series, q: float) -> float:
    try:
        v = float(x.quantile(q))
        return v if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def _build_bins(p_known: pd.Series, n_bins: int) -> pd.Categorical:
    # Quantile bins are more stable than fixed odds bins across leagues.
    # Add tiny jitter to avoid identical edges when distribution is discrete.
    pk = pd.to_numeric(p_known, errors="coerce")
    pk = pk.dropna()
    if pk.empty:
        return pd.Categorical([])

    jitter = (np.random.default_rng(7).normal(0, 1e-8, size=len(pk)))
    pk2 = pk.to_numpy(dtype=float) + jitter
    # compute quantile edges
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(pk2, qs)
    # de-duplicate edges
    edges = np.unique(edges)
    if len(edges) < 3:
        # not enough spread
        edges = np.array([pk2.min() - 1e-6, pk2.max() + 1e-6])

    # pd.cut requires full series; reindex later
    return pd.cut(p_known, bins=edges, include_lowest=True, duplicates="drop")


def fit_rulebook(
    df: pd.DataFrame,
    *,
    market: str,
    known_side: str,
    group_key: Tuple[str, ...],
    n_bins: int = 10,
    min_n: int = 200,
) -> Dict[str, object]:
    """Fit a rulebook for one market across groups.

    Returns a JSON-serializable dict.

    market:
      - "ou25" or "btts"

    known_side:
      - for ou25: "over" (supported) or "under" (supported)
      - for btts: "yes" (supported) or "no" (supported)
    """

    market = str(market).strip().lower()
    known_side = str(known_side).strip().lower()

    if market not in ("ou25", "btts"):
        raise ValueError(f"unsupported market: {market}")

    # resolve columns (IMPORTANT: learn from the same known-side you have forward, and use FD for missing-side truth)
    known_col_used = ""
    miss_col_used = ""

    if market == "ou25":
        if known_side == "over":
            c_known = _pick_first_from(df, OU25_OVER_PRIORITY)
            c_miss = _pick_first_from(df, OU25_UNDER_TRUTH_PRIORITY)
        elif known_side == "under":
            c_known = _pick_first_from(df, OU25_UNDER_PRIORITY)
            # when known is under, missing is over; prefer OG over first
            c_miss = _pick_first_from(df, OU25_OVER_PRIORITY)
        else:
            raise ValueError("known_side for ou25 must be 'over' or 'under'")

        if not c_known or not c_miss:
            known_existing = _first_existing_candidates(df, OU25_OVER_PRIORITY if known_side == "over" else OU25_UNDER_PRIORITY)
            miss_existing = _first_existing_candidates(df, OU25_UNDER_TRUTH_PRIORITY if known_side == "over" else OU25_OVER_PRIORITY)
            raise ValueError(
                "OU25 requires known+missing columns | "
                f"known_side={known_side} | "
                f"known_existing={known_existing} | "
                f"missing_existing={miss_existing}"
            )

        known_col_used = str(c_known)
        miss_col_used = str(c_miss)

        known_od = _to_odds(df[c_known])
        miss_od = _to_odds(df[c_miss])

    else:  # btts
        if known_side == "yes":
            c_known = _pick_first_from(df, BTTS_YES_PRIORITY)
            c_miss = _pick_first_from(df, BTTS_NO_TRUTH_PRIORITY)
        elif known_side == "no":
            c_known = _pick_first_from(df, BTTS_NO_TRUTH_PRIORITY)
            c_miss = _pick_first_from(df, BTTS_YES_PRIORITY)
        else:
            raise ValueError("known_side for btts must be 'yes' or 'no'")

        if not c_known or not c_miss:
            known_existing = _first_existing_candidates(df, BTTS_YES_PRIORITY if known_side == "yes" else BTTS_NO_TRUTH_PRIORITY)
            miss_existing = _first_existing_candidates(df, BTTS_NO_TRUTH_PRIORITY if known_side == "yes" else BTTS_YES_PRIORITY)
            raise ValueError(
                "BTTS requires yes+no columns | "
                f"known_side={known_side} | "
                f"known_existing={known_existing} | "
                f"missing_existing={miss_existing}"
            )

        known_col_used = str(c_known)
        miss_col_used = str(c_miss)

        known_od = _to_odds(df[c_known])
        miss_od = _to_odds(df[c_miss])

    # keep only rows with both sides
    m = known_od.notna() & miss_od.notna()
    work = df.loc[m].copy()
    known_od = known_od.loc[m]
    miss_od = miss_od.loc[m]

    # implieds and overround
    p_known = 1.0 / known_od
    p_miss = 1.0 / miss_od
    R = p_known + p_miss

    # group columns
    for g in group_key:
        if g not in work.columns:
            # fill with a default (allows group_key including od_source, etc.)
            work[g] = "*"

    # bins within each group
    out: Dict[str, object] = {
        "market": market,
        "known_side": known_side,
        "group_key": list(group_key),
        "n_bins": int(n_bins),
        "min_n": int(min_n),
        "fitted_at_utc": _utc_now_iso(),
        "groups": {},
        "source_cols": {
            "known": str(known_col_used),
            "missing": str(miss_col_used),
        },
    }

    # group id is "val1||val2||..."
    grouped = work.groupby(list(group_key), dropna=False)
    # IMPORTANT: use `.groups` (index labels), not `.indices` (positional), to keep `.loc` safe
    for gvals, idx in grouped.groups.items():
        if not isinstance(gvals, tuple):
            gvals = (gvals,)
        gid = "||".join([str(x) for x in gvals])

        pk = p_known.loc[idx]
        rr = R.loc[idx]
        mo = miss_od.loc[idx]

        if pk.notna().sum() < int(min_n):
            # still store group meta but with empty bins
            out["groups"][gid] = {
                "n": int(len(idx)),
                "bins": [],
                "note": f"below_min_n({min_n})",
            }
            continue

        cats = _build_bins(pk, int(n_bins))
        # cats is aligned to pk index; safe
        bins: List[Dict[str, object]] = []

        # groupby on cats
        by_bin = pd.Series(cats, index=pk.index)
        for b, bidx in by_bin.groupby(by_bin, observed=False).groups.items():
            sel = list(bidx)
            n = int(len(sel))
            if n < max(30, int(min_n * 0.05)):
                # too thin to be stable
                continue

            pk_b = pk.loc[sel]
            rr_b = rr.loc[sel]
            mo_b = mo.loc[sel]

            bs = BinStats(
                n=n,
                p_lo=float(pk_b.min()),
                p_hi=float(pk_b.max()),
                R_median=float(_quantile(rr_b, 0.50)),
                R_q25=float(_quantile(rr_b, 0.25)),
                R_q75=float(_quantile(rr_b, 0.75)),
                R_p05=float(_quantile(rr_b, 0.05)),
                R_p95=float(_quantile(rr_b, 0.95)),
                miss_med=float(_quantile(mo_b, 0.50)),
                miss_q10=float(_quantile(mo_b, 0.10)),
                miss_q25=float(_quantile(mo_b, 0.25)),
                miss_q75=float(_quantile(mo_b, 0.75)),
                miss_q90=float(_quantile(mo_b, 0.90)),
            )

            bins.append({
                "n": bs.n,
                "p_lo": bs.p_lo,
                "p_hi": bs.p_hi,
                "R_median": bs.R_median,
                "R_q25": bs.R_q25,
                "R_q75": bs.R_q75,
                "R_p05": bs.R_p05,
                "R_p95": bs.R_p95,
                "miss_med": bs.miss_med,
                "miss_q10": bs.miss_q10,
                "miss_q25": bs.miss_q25,
                "miss_q75": bs.miss_q75,
                "miss_q90": bs.miss_q90,
            })

        # sort bins by p_lo
        bins = sorted(bins, key=lambda d: float(d.get("p_lo", 0.0)))
        out["groups"][gid] = {
            "n": int(len(idx)),
            "bins": bins,
        }

    # --- Global fallback group ---
    # If a league has no stable group table (or is below min_n), apply-time will
    # fall back to this global group. This is the key mechanism that lets us
    # generalise OU25 / BTTS synthesis to leagues without historical missing-side odds.
    #
    # IMPORTANT: Use an explicit global key rather than "*" to avoid accidental
    # collisions with real league/source values.
    GLOBAL_KEY = "__GLOBAL__"
    wild_gid = GLOBAL_KEY if len(group_key) == 1 else "||".join([GLOBAL_KEY] * len(group_key))

    # Always (re)build the global fallback from *all* rows with both sides.
    pk_all = p_known
    rr_all = R
    mo_all = miss_od

    if pk_all.notna().sum() >= int(min_n):
        cats = _build_bins(pk_all, int(n_bins))
        bins: List[Dict[str, object]] = []

        by_bin = pd.Series(cats, index=pk_all.index)
        for _, bidx in by_bin.groupby(by_bin, observed=False).groups.items():
            sel = list(bidx)
            n = int(len(sel))
            if n < max(30, int(min_n * 0.05)):
                continue

            pk_b = pk_all.loc[sel]
            rr_b = rr_all.loc[sel]
            mo_b = mo_all.loc[sel]

            bs = BinStats(
                n=n,
                p_lo=float(pk_b.min()),
                p_hi=float(pk_b.max()),
                R_median=float(_quantile(rr_b, 0.50)),
                R_q25=float(_quantile(rr_b, 0.25)),
                R_q75=float(_quantile(rr_b, 0.75)),
                R_p05=float(_quantile(rr_b, 0.05)),
                R_p95=float(_quantile(rr_b, 0.95)),
                miss_med=float(_quantile(mo_b, 0.50)),
                miss_q10=float(_quantile(mo_b, 0.10)),
                miss_q25=float(_quantile(mo_b, 0.25)),
                miss_q75=float(_quantile(mo_b, 0.75)),
                miss_q90=float(_quantile(mo_b, 0.90)),
            )

            bins.append({
                "n": bs.n,
                "p_lo": bs.p_lo,
                "p_hi": bs.p_hi,
                "R_median": bs.R_median,
                "R_q25": bs.R_q25,
                "R_q75": bs.R_q75,
                "R_p05": bs.R_p05,
                "R_p95": bs.R_p95,
                "miss_med": bs.miss_med,
                "miss_q10": bs.miss_q10,
                "miss_q25": bs.miss_q25,
                "miss_q75": bs.miss_q75,
                "miss_q90": bs.miss_q90,
            })

        bins = sorted(bins, key=lambda d: float(d.get("p_lo", 0.0)))
        out["groups"][wild_gid] = {
            "n": int(pk_all.notna().sum()),
            "bins": bins,
            "note": "global_fallback",
        }
    else:
        out["groups"][wild_gid] = {
            "n": int(pk_all.notna().sum()),
            "bins": [],
            "note": f"global_below_min_n({min_n})",
        }

    return out


# -----------------------------
# Prediction
# -----------------------------

@dataclass
class SynthResult:
    odds: float
    method: str
    conf: float
    reason: str
    group_id: str
    bin_idx: int


def _pick_group_id(row: pd.Series, group_key: List[str]) -> str:
    vals = []
    for g in group_key:
        vals.append(str(row.get(g, "*") if pd.notna(row.get(g, "*")) else "*") )
    return "||".join(vals)


def _choose_bin(bins: List[dict], p_known: float) -> Tuple[int, Optional[dict]]:
    if not bins:
        return (-1, None)
    # bins are sorted by p_lo
    for i, b in enumerate(bins):
        try:
            lo = float(b.get("p_lo", -1.0))
            hi = float(b.get("p_hi", 2.0))
        except Exception:
            continue
        if (p_known >= lo) and (p_known <= hi):
            return (i, b)
    # out of range: choose nearest by center distance
    best = (0, bins[0], float("inf"))
    for i, b in enumerate(bins):
        lo = float(b.get("p_lo", 0.0)); hi = float(b.get("p_hi", 0.0))
        ctr = 0.5 * (lo + hi)
        d = abs(p_known - ctr)
        if d < best[2]:
            best = (i, b, d)
    return (best[0], best[1])


def _conf_from_bin(b: dict, *, min_n_bin: int = 50) -> float:
    try:
        n = int(b.get("n", 0))
    except Exception:
        n = 0
    if n <= 0:
        return 0.0

    # tighter IQR => higher confidence
    R_q25 = float(b.get("R_q25", np.nan))
    R_q75 = float(b.get("R_q75", np.nan))
    iqr = (R_q75 - R_q25) if (np.isfinite(R_q25) and np.isfinite(R_q75)) else np.nan

    # n factor saturates
    n_factor = min(1.0, math.log1p(n) / math.log1p(max(min_n_bin * 4, 200)))

    if not np.isfinite(iqr):
        return float(0.25 * n_factor)

    # typical OU25 iqr is small; scale conservatively
    iqr_scale = 0.02
    iqr_factor = float(max(0.0, min(1.0, 1.0 - (iqr / iqr_scale))))

    return float(max(0.0, min(1.0, 0.15 + 0.85 * (0.55 * n_factor + 0.45 * iqr_factor))))


def synth_missing_odds_for_row(
    row: pd.Series,
    *,
    tables: dict,
    market: str,
    known_side: str,
    group_key: List[str],
    min_n_group: int = 200,
    min_p_missing: float = 1e-6,
    clamp_quantiles: Tuple[str, str] = ("miss_q10", "miss_q90"),
    conf_min: float = 0.25,
) -> SynthResult:
    market = str(market).lower().strip()
    known_side = str(known_side).lower().strip()

    group_id = _pick_group_id(row, group_key)
    g = tables.get("groups", {}).get(group_id)
    if not isinstance(g, dict) or int(g.get("n", 0)) < int(min_n_group):
        # fallback: try global group if present
        GLOBAL_KEY = "__GLOBAL__"
        global_gid = GLOBAL_KEY if len(group_key) == 1 else "||".join([GLOBAL_KEY] * len(group_key))
        g = tables.get("groups", {}).get(global_gid)
        group_id = global_gid if g else group_id

    if not isinstance(g, dict):
        return SynthResult(odds=float("nan"), method="drop", conf=0.0, reason="no_group", group_id=group_id, bin_idx=-1)

    bins = g.get("bins", [])
    if not isinstance(bins, list) or not bins:
        return SynthResult(odds=float("nan"), method="drop", conf=0.0, reason="no_bins", group_id=group_id, bin_idx=-1)

    # Get known odds (stable priority; matches training intent)
    ok, used_known_col = _get_known_odds_value(row, market=market, known_side=known_side)
    if not np.isfinite(ok) or ok <= 1.0001:
        return SynthResult(
            odds=float("nan"),
            method="drop",
            conf=0.0,
            reason="known_missing",
            group_id=group_id,
            bin_idx=-1,
        )

    p_known = 1.0 / ok

    bi, b = _choose_bin(bins, p_known)
    if b is None:
        return SynthResult(odds=float("nan"), method="drop", conf=0.0, reason="no_bin", group_id=group_id, bin_idx=-1)

    R_hat = float(b.get("R_median", np.nan))
    if not np.isfinite(R_hat) or R_hat <= 0:
        return SynthResult(odds=float("nan"), method="drop", conf=0.0, reason="bad_R", group_id=group_id, bin_idx=int(bi))

    p_miss = R_hat - p_known
    if (not np.isfinite(p_miss)) or (p_miss <= float(min_p_missing)):
        return SynthResult(odds=float("nan"), method="drop", conf=0.0, reason="p_miss_nonpositive", group_id=group_id, bin_idx=int(bi))

    odds_hat = 1.0 / p_miss

    # clamp
    qlo_name, qhi_name = clamp_quantiles
    qlo = float(b.get(qlo_name, np.nan))
    qhi = float(b.get(qhi_name, np.nan))
    if np.isfinite(qlo) and np.isfinite(qhi) and (qhi > qlo > 1.0001):
        odds_hat = float(min(max(odds_hat, qlo), qhi))

    conf = _conf_from_bin(b)
    if conf < float(conf_min):
        return SynthResult(odds=float("nan"), method="drop", conf=float(conf), reason="low_conf", group_id=group_id, bin_idx=int(bi))

    return SynthResult(odds=float(odds_hat), method="bin_R_sub", conf=float(conf), reason="ok", group_id=group_id, bin_idx=int(bi))


# -----------------------------
# Batch apply
# -----------------------------

def apply_synth(
    df: pd.DataFrame,
    *,
    tables_ou25: Optional[dict] = None,
    tables_btts: Optional[dict] = None,
    group_key: List[str],
    conf_min: float = 0.25,
    min_n_group: int = 200,
) -> pd.DataFrame:
    out = df.copy()

    # Ensure group columns exist
    for g in group_key:
        if g not in out.columns:
            out[g] = "*"

    # OU25: synth under
    if tables_ou25 is not None:
        res_od = []
        res_conf = []
        res_reason = []
        res_gid = []
        res_bin = []
        for _, r in out.iterrows():
            sr = synth_missing_odds_for_row(
                r,
                tables=tables_ou25,
                market="ou25",
                known_side=tables_ou25.get("known_side", "over"),
                group_key=group_key,
                conf_min=float(conf_min),
                min_n_group=int(min_n_group),
            )
            res_od.append(sr.odds)
            res_conf.append(sr.conf)
            res_reason.append(sr.reason)
            res_gid.append(sr.group_id)
            res_bin.append(sr.bin_idx)

        out["odds_ft_under25_synth"] = pd.to_numeric(pd.Series(res_od, index=out.index), errors="coerce")
        out["under25_synth_conf"] = pd.to_numeric(pd.Series(res_conf, index=out.index), errors="coerce")
        out["under25_synth_reason"] = pd.Series(res_reason, index=out.index, dtype="string")
        out["under25_synth_group"] = pd.Series(res_gid, index=out.index, dtype="string")
        out["under25_synth_bin"] = pd.to_numeric(pd.Series(res_bin, index=out.index), errors="coerce")

        # --- NEW: promote synth into canonical output column (V3) ---
        # Keep the raw synth column for diagnostics, but ensure downstream always has
        # the canonical odds_ft_under25 (and optionally a source column).
        if "odds_ft_under25" not in out.columns:
            out["odds_ft_under25"] = np.nan

        _u25_base = pd.to_numeric(out["odds_ft_under25"], errors="coerce")
        _u25_syn  = pd.to_numeric(out["odds_ft_under25_synth"], errors="coerce")
        _u25_missing = _u25_base.isna() & _u25_syn.notna()
        out.loc[_u25_missing, "odds_ft_under25"] = _u25_syn.loc[_u25_missing]

        # Optional provenance column for downstream pack-building / QA
        if "odds_source_under25" not in out.columns:
            out["odds_source_under25"] = pd.Series(["" for _ in range(len(out))], index=out.index, dtype="string")
        out.loc[_u25_missing, "odds_source_under25"] = "synth_ou25"

    # BTTS: synth NO
    if tables_btts is not None:
        res_od = []
        res_conf = []
        res_reason = []
        res_gid = []
        res_bin = []
        for _, r in out.iterrows():
            sr = synth_missing_odds_for_row(
                r,
                tables=tables_btts,
                market="btts",
                known_side=tables_btts.get("known_side", "yes"),
                group_key=group_key,
                conf_min=float(conf_min),
                min_n_group=int(min_n_group),
            )
            res_od.append(sr.odds)
            res_conf.append(sr.conf)
            res_reason.append(sr.reason)
            res_gid.append(sr.group_id)
            res_bin.append(sr.bin_idx)

        out["odds_btts_no_synth"] = pd.to_numeric(pd.Series(res_od, index=out.index), errors="coerce")
        out["bttsno_synth_conf"] = pd.to_numeric(pd.Series(res_conf, index=out.index), errors="coerce")
        out["bttsno_synth_reason"] = pd.Series(res_reason, index=out.index, dtype="string")
        out["bttsno_synth_group"] = pd.Series(res_gid, index=out.index, dtype="string")
        out["bttsno_synth_bin"] = pd.to_numeric(pd.Series(res_bin, index=out.index), errors="coerce")

        # --- NEW: promote synth into canonical output column (V3) ---
        if "odds_btts_no" not in out.columns:
            out["odds_btts_no"] = np.nan

        _bn_base = pd.to_numeric(out["odds_btts_no"], errors="coerce")
        _bn_syn  = pd.to_numeric(out["odds_btts_no_synth"], errors="coerce")
        _bn_missing = _bn_base.isna() & _bn_syn.notna()
        out.loc[_bn_missing, "odds_btts_no"] = _bn_syn.loc[_bn_missing]

    # --- Backwards-compat aliases (do not override existing cols / values) ---
    # Canonical -> legacy
    if "odds_ft_over25" in out.columns and "odds_over25" not in out.columns:
        out["odds_over25"] = out["odds_ft_over25"]
    if "odds_ft_under25" in out.columns and "odds_under25" not in out.columns:
        out["odds_under25"] = out["odds_ft_under25"]

    # Legacy -> canonical (safety net for older feeds / downstream code)
    if "odds_over25" in out.columns and "odds_ft_over25" not in out.columns:
        out["odds_ft_over25"] = out["odds_over25"]
    if "odds_under25" in out.columns and "odds_ft_under25" not in out.columns:
        out["odds_ft_under25"] = out["odds_under25"]

    return out


# -----------------------------
# Backtesting
# -----------------------------

@dataclass
class BacktestRow:
    league: str
    market: str
    known_side: str
    n_test: int
    coverage: float
    mae_implied: float
    medae_implied: float
    mae_odds: float
    medae_odds: float


def backtest_rulebook(
    df: pd.DataFrame,
    *,
    market: str,
    known_side: str,
    group_key: Tuple[str, ...],
    n_bins: int,
    min_n: int,
    holdout_weeks: int = 10,
    conf_min: float = 0.25,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    market = str(market).lower().strip()
    known_side = str(known_side).lower().strip()

    # date split
    dts = _to_date_series(df)
    df2 = df.copy()
    df2["__dt"] = pd.to_datetime(dts.astype(str), errors="coerce", utc=True)

    # Keep only rows with dt
    df2 = df2[df2["__dt"].notna()].copy()
    if df2.empty:
        return pd.DataFrame(), {"error": "no_dates"}

    # Restrict to rows where BOTH sides exist BEFORE splitting the holdout.
    # This prevents the "last N weeks" window being dominated by forward fixtures
    # that often have only the known side quoted.
    if market == "ou25":
        if known_side == "over":
            known_col = _pick_first_from(df2, OU25_OVER_PRIORITY)
            miss_col = _pick_first_from(df2, OU25_UNDER_TRUTH_PRIORITY)
        else:
            known_col = _pick_first_from(df2, OU25_UNDER_PRIORITY)
            miss_col = _pick_first_from(df2, OU25_OVER_PRIORITY)
        if not known_col or not miss_col:
            return pd.DataFrame(), {"error": "missing_ou25_cols"}
    else:
        if known_side == "yes":
            known_col = _pick_first_from(df2, BTTS_YES_PRIORITY)
            miss_col = _pick_first_from(df2, BTTS_NO_TRUTH_PRIORITY)
        else:
            known_col = _pick_first_from(df2, BTTS_NO_TRUTH_PRIORITY)
            miss_col = _pick_first_from(df2, BTTS_YES_PRIORITY)
        if not known_col or not miss_col:
            return pd.DataFrame(), {"error": "missing_btts_cols"}

    known_od_all = _to_odds(df2[known_col])
    miss_od_all = _to_odds(df2[miss_col])
    df_both = df2.loc[known_od_all.notna() & miss_od_all.notna()].copy()

    # Drop future-dated rows (common with forward fixtures / placeholders)
    try:
        now_utc = pd.Timestamp.now(tz="UTC")
        df_both = df_both[df_both["__dt"] <= now_utc].copy()
    except Exception:
        pass

    if df_both.empty:
        return pd.DataFrame(), {
            "error": "no_rows_with_both_sides",
            "known_col": str(known_col),
            "miss_col": str(miss_col),
        }

    cutoff = df_both["__dt"].max() - pd.Timedelta(days=int(holdout_weeks) * 7)
    train = df_both[df_both["__dt"] < cutoff].copy()
    test = df_both[df_both["__dt"] >= cutoff].copy()

    if train.empty or test.empty:
        return pd.DataFrame(), {
            "error": "empty_train_or_test",
            "train_n": int(len(train)),
            "test_n": int(len(test)),
            "cutoff": str(cutoff),
        }

    # fit on train
    tables = fit_rulebook(train, market=market, known_side=known_side, group_key=group_key, n_bins=n_bins, min_n=min_n)

    # synth on test (predict missing side)
    # Build a lightweight view for synth
    test_view = test.copy()
    # Ensure group columns exist
    for g in group_key:
        if g not in test_view.columns:
            test_view[g] = "*"

    # per-row predict
    pred = []
    confs = []
    for _, r in test_view.iterrows():
        sr = synth_missing_odds_for_row(
            r,
            tables=tables,
            market=market,
            known_side=known_side,
            group_key=list(group_key),
            conf_min=conf_min,
            min_n_group=min_n,
        )
        pred.append(sr.odds)
        confs.append(sr.conf)

    pred_s = pd.to_numeric(pd.Series(pred, index=test_view.index), errors="coerce")
    true_s = _to_odds(test_view[miss_col])

    # coverage
    ok = pred_s.notna() & true_s.notna()
    cov = float(ok.mean()) if len(ok) else 0.0

    # errors
    if ok.any():
        p_true = 1.0 / true_s.loc[ok]
        p_hat = 1.0 / pred_s.loc[ok]
        ae_p = (p_true - p_hat).abs()
        ae_o = (true_s.loc[ok] - pred_s.loc[ok]).abs()
        mae_p = float(ae_p.mean())
        med_p = float(ae_p.median())
        mae_o = float(ae_o.mean())
        med_o = float(ae_o.median())
    else:
        mae_p = med_p = mae_o = med_o = float("nan")

    # group-level summary
    lg_col = _pick_first(df2, LEAGUE_CANDS)
    if lg_col is None:
        lg_col = "league"
        test_view["league"] = "UNKNOWN_LEAGUE"

    # produce a compact result table by league (best for your use)
    rows = []
    for lg, sub in test_view.groupby(lg_col, dropna=False):
        sub_pred = pred_s.loc[sub.index]
        sub_true = _to_odds(sub[miss_col])
        ok2 = sub_pred.notna() & sub_true.notna()
        if len(ok2) == 0:
            continue
        cov2 = float(ok2.mean())
        if ok2.any():
            p_true2 = 1.0 / sub_true.loc[ok2]
            p_hat2 = 1.0 / sub_pred.loc[ok2]
            ae_p2 = (p_true2 - p_hat2).abs()
            ae_o2 = (sub_true.loc[ok2] - sub_pred.loc[ok2]).abs()
            rows.append({
                "league": str(lg),
                "market": market,
                "known_side": known_side,
                "n_test": int(len(sub)),
                "coverage": float(cov2),
                "mae_implied": float(ae_p2.mean()),
                "medae_implied": float(ae_p2.median()),
                "mae_odds": float(ae_o2.mean()),
                "medae_odds": float(ae_o2.median()),
            })
        else:
            rows.append({
                "league": str(lg),
                "market": market,
                "known_side": known_side,
                "n_test": int(len(sub)),
                "coverage": float(cov2),
                "mae_implied": float("nan"),
                "medae_implied": float("nan"),
                "mae_odds": float("nan"),
                "medae_odds": float("nan"),
            })

    summary = {
        "market": market,
        "known_side": known_side,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "coverage": float(cov),
        "mae_implied": float(mae_p),
        "medae_implied": float(med_p),
        "mae_odds": float(mae_o),
        "medae_odds": float(med_o),
        "holdout_weeks": int(holdout_weeks),
        "n_bins": int(n_bins),
        "min_n": int(min_n),
        "conf_min": float(conf_min),
        "known_col": str(known_col),
        "miss_col": str(miss_col),
    }

    return pd.DataFrame(rows), summary


# -----------------------------
# IO helpers
# -----------------------------

def _read_many_csvs(patterns: List[str]) -> pd.DataFrame:
    files: List[str] = []
    for pat in patterns:
        files.extend([str(p) for p in sorted(Path().glob(pat))])

    # If glob didn't find anything (common with quoted paths), try recursive glob
    if not files:
        for pat in patterns:
            files.extend([str(p) for p in sorted(Path(".").glob(pat))])

    dfs = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue

        # normalize league
        lg_col = _pick_first(df, LEAGUE_CANDS)
        if lg_col is None:
            df["league"] = _infer_league_from_path(fp)
        else:
            if lg_col != "league":
                df = df.rename(columns={lg_col: "league"})
            df["league"] = df["league"].astype("string").fillna("").str.strip().replace({"": _infer_league_from_path(fp)})

        # normalize od_source if present
        src_col = _pick_first(df, OD_SOURCE_CANDS)
        if src_col is not None and src_col != "od_source":
            df = df.rename(columns={src_col: "od_source"})
        if "od_source" not in df.columns:
            df["od_source"] = "*"
        df["od_source"] = df["od_source"].astype("string").fillna("").str.strip().replace({"": "*"})

        df["__src"] = fp
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No readable CSVs for patterns: {patterns}")

    return pd.concat(dfs, ignore_index=True, sort=False)


def _write_json(path: str, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# CLI
# -----------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Odds synthesis: infer missing Under25 / BTTS NO from known side")
    sub = p.add_subparsers(dest="cmd", required=True)

    # fit
    f = sub.add_parser("fit", help="Fit synthesis tables from historical rows where both sides exist")
    f.add_argument("--src", nargs="+", required=True, help="CSV glob(s) to read")
    f.add_argument("--out", required=True, help="Output JSON path")
    f.add_argument("--markets", nargs="+", default=["ou25", "btts"], help="Markets to fit: ou25 btts")
    f.add_argument("--known-side", nargs="+", default=["over", "yes"], help="Known side per market: over|under and yes|no")
    f.add_argument("--bins", type=int, default=10)
    f.add_argument("--min-n", type=int, default=200)
    f.add_argument("--group-by", default="league", help="Comma-separated grouping columns (e.g. league or league,od_source)")

    # backtest
    b = sub.add_parser("backtest", help="Backtest synthesis on holdout window")
    b.add_argument("--src", nargs="+", required=True, help="CSV glob(s) to read")
    b.add_argument("--out", default=None, help="Optional output CSV for league-level rows")
    b.add_argument("--markets", nargs="+", default=["ou25", "btts"], help="Markets to test: ou25 btts")
    b.add_argument("--known-side", nargs="+", default=["over", "yes"], help="Known side per market: over|under and yes|no")
    b.add_argument("--bins", type=int, default=10)
    b.add_argument("--min-n", type=int, default=200)
    b.add_argument("--holdout-weeks", type=int, default=10)
    b.add_argument("--conf-min", type=float, default=0.25)
    b.add_argument("--group-by", default="league", help="Comma-separated grouping columns (e.g. league or league,od_source)")

    # apply
    a = sub.add_parser("apply", help="Apply existing tables to a CSV (forward fixtures) and write enriched output")
    a.add_argument("--src", required=True, help="Input CSV")
    a.add_argument("--tables", required=True, help="Tables JSON from fit")
    a.add_argument("--out", required=True, help="Output CSV")
    a.add_argument("--conf-min", type=float, default=0.25)
    a.add_argument("--min-n", type=int, default=200)
    a.add_argument("--group-by", default="league", help="Comma-separated grouping columns (must match fit)")

    return p


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()

    if args.cmd == "fit":
        df = _read_many_csvs(args.src)
        group_key = tuple([s.strip() for s in str(args.group_by).split(",") if s.strip()])
        markets = [m.lower().strip() for m in args.markets]
        ks = [k.lower().strip() for k in args.known_side]

        # map market->known side
        # defaults: ou25->over, btts->yes
        known_map = {"ou25": "over", "btts": "yes"}
        for k in ks:
            if k in ("over", "under"):
                known_map["ou25"] = k
            if k in ("yes", "no"):
                known_map["btts"] = k

        payload = {"tables": {}, "fit_errors": {}}
        for mk in markets:
            try:
                if mk == "ou25":
                    payload["tables"]["ou25"] = fit_rulebook(
                        df,
                        market="ou25",
                        known_side=known_map["ou25"],
                        group_key=group_key,
                        n_bins=int(args.bins),
                        min_n=int(args.min_n),
                    )
                elif mk == "btts":
                    payload["tables"]["btts"] = fit_rulebook(
                        df,
                        market="btts",
                        known_side=known_map["btts"],
                        group_key=group_key,
                        n_bins=int(args.bins),
                        min_n=int(args.min_n),
                    )
            except Exception as e:
                payload["fit_errors"][mk] = str(e)

        payload["meta"] = {
            "created_utc": _utc_now_iso(),
            "src": args.src,
            "group_by": str(args.group_by),
            "bins": int(args.bins),
            "min_n": int(args.min_n),
        }

        _write_json(args.out, payload)
        print(f"[OK] wrote tables: {args.out}")
        # tiny summary
        for mk, t in payload["tables"].items():
            ng = len(t.get("groups", {}))
            print(f"  {mk}: groups={ng} known_side={t.get('known_side')}")

        if payload.get("fit_errors"):
            for mk, err in payload["fit_errors"].items():
                print(f"  {mk}: SKIPPED | {err}")

    elif args.cmd == "backtest":
        df = _read_many_csvs(args.src)
        group_key = tuple([s.strip() for s in str(args.group_by).split(",") if s.strip()])
        markets = [m.lower().strip() for m in args.markets]
        ks = [k.lower().strip() for k in args.known_side]

        known_map = {"ou25": "over", "btts": "yes"}
        for k in ks:
            if k in ("over", "under"):
                known_map["ou25"] = k
            if k in ("yes", "no"):
                known_map["btts"] = k

        all_rows = []
        summaries = []
        for mk in markets:
            if mk not in ("ou25", "btts"):
                continue
            rows, summ = backtest_rulebook(
                df,
                market=mk,
                known_side=known_map[mk],
                group_key=group_key,
                n_bins=int(args.bins),
                min_n=int(args.min_n),
                holdout_weeks=int(args.holdout_weeks),
                conf_min=float(args.conf_min),
            )
            if not rows.empty:
                all_rows.append(rows)
            summaries.append(summ)

            print(f"\n=== BACKTEST {mk} (known={known_map[mk]}) ===")
            print(json.dumps(summ, indent=2, default=str))

        out_rows = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            out_rows.to_csv(args.out, index=False)
            print(f"\n[OK] wrote: {args.out}")

        # also write a JSON summary next to it (or cwd)
        out_json = None
        if args.out:
            out_json = str(Path(args.out).with_suffix(".json"))
        else:
            out_json = f"ODDS_SYNTH_BACKTEST_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        _write_json(out_json, {"summaries": summaries})
        print(f"[OK] wrote: {out_json}")

    elif args.cmd == "apply":
        payload = _read_json(args.tables)
        tables_ou = payload.get("tables", {}).get("ou25")
        tables_bt = payload.get("tables", {}).get("btts")
        fit_errors = payload.get("fit_errors", {})
        group_key = [s.strip() for s in str(args.group_by).split(",") if s.strip()]

        df = pd.read_csv(args.src)
        # normalize league + od_source
        if "league" not in df.columns:
            df["league"] = _infer_league_from_path(args.src)
        if "od_source" not in df.columns:
            # don't blow up grouping; keep wildcard
            df["od_source"] = "*"

        out = apply_synth(
            df,
            tables_ou25=tables_ou,
            tables_btts=tables_bt,
            group_key=group_key,
            conf_min=float(args.conf_min),
            min_n_group=int(args.min_n),
        )

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False)
        print(f"[OK] wrote: {args.out}")

        # quick coverage prints
        if tables_ou is not None:
            nn = int(pd.to_numeric(out.get("odds_ft_under25_synth"), errors="coerce").notna().sum())
            print(f"  under25_synth nonnull: {nn}/{len(out)}")
        elif "ou25" in fit_errors:
            print(f"  under25_synth skipped: {fit_errors['ou25']}")

        if tables_bt is not None:
            nn = int(pd.to_numeric(out.get("odds_btts_no_synth"), errors="coerce").notna().sum())
            print(f"  bttsno_synth nonnull: {nn}/{len(out)}")
        elif "btts" in fit_errors:
            print(f"  bttsno_synth skipped: {fit_errors['btts']}")


if __name__ == "__main__":
    main()