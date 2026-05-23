#!/usr/bin/env python3
from __future__ import annotations

"""
acca_builder.py

Legacy acca builder / slip backtester.

Current status
--------------
This file now contains both:
1. the legacy pool/slip builder and backtester, and
2. a staged v2 architecture scaffold for modern deploy-driven acca construction.

The v2 scaffold is intentionally additive and non-destructive.
It is designed so the file can be rebuilt in controlled stages rather than by
trying to replace the whole script in one pass.

What is still legacy
--------------------
- The original `build` command still uses the older generic pool scoring flow.
- Glue / anchor / monster logic remains part of the legacy builder path.
- The legacy builder does not yet act as the final production shortlist engine.

What is now added
-----------------
- Direct deploy-tier ingestion helpers.
- Tier-aware canonical pool construction.
- Calendar-regime-aware leg multipliers.
- A modern `compute_leg_priority()` path that respects deploy metadata.
- A new `build_v2` command that writes a canonical acca pool plus summary artefacts.
- A new `build_v2_slips` command that builds ranked FTR-only, BTTS-only, OU25-only, and combined acca outputs from deploy tiers.

Current rebuild plan
--------------------
Phase 1: ingest deploy_rulebook outputs directly.
Phase 2: canonicalize ELITE / STANDARD deploy rows into a production pool.
Phase 3: apply calendar-sensitive multipliers and modern leg priority.
Phase 4: add template-driven slip construction.
Phase 5: wire final shortlist generation and investor-grade summaries.

Important
---------
Treat `build_v2` as the modern architecture entrypoint.
Treat legacy `build` as preserved backward-compatible behaviour until the v2 slip
construction path fully replaces it.
"""
 # -----------------------------
 # v2 deploy-driven constants
 # -----------------------------

DEFAULT_INCLUDE_TIERS_V2 = ("ELITE", "STANDARD")
DEFAULT_EXCLUDE_TIERS_V2 = ("OBSERVE",)

DEPLOY_TIER_TOKEN_MAP = {
    "ELITE": "DEPLOY_TIER_ELITE",
    "STANDARD": "DEPLOY_TIER_STANDARD",
    "OBSERVE": "DEPLOY_TIER_OBSERVE",
}

CALENDAR_REGIME_POLICY = {
    "NORMAL": {
        "btts_leg_multiplier": 1.00,
        "ftr_leg_multiplier": 1.00,
        "stake_multiplier": 1.00,
        "overlay_note": "Normal deployment window; no calendar adjustment applied.",
    },
    "POST_FIFA_BREAK": {
        "btts_leg_multiplier": 0.75,
        "ftr_leg_multiplier": 1.00,
        "stake_multiplier": 0.90,
        "overlay_note": "Reduce BTTS exposure after FIFA/international breaks; FTR unchanged.",
    },
    "PRE_UCL_KNOCKOUT_FIRST_LEG": {
        "btts_leg_multiplier": 1.00,
        "ftr_leg_multiplier": 0.75,
        "stake_multiplier": 0.90,
        "overlay_note": "Reduce FTR exposure before UCL knockout first-leg weeks; BTTS unchanged.",
    },
    "NOV_FIFA_PLUS_PRE_UCL": {
        "btts_leg_multiplier": 0.70,
        "ftr_leg_multiplier": 0.70,
        "stake_multiplier": 0.75,
        "overlay_note": "Reduce both BTTS and FTR exposure in the November FIFA plus pre-UCL congestion window.",
    },
    "POST_UCL_LEAGUE_PHASE": {
        "btts_leg_multiplier": 1.00,
        "ftr_leg_multiplier": 1.00,
        "stake_multiplier": 1.00,
        "overlay_note": "Post-UCL league-phase weekend; deploy normally.",
    },
    "POST_R16_SECOND_LEG_MARCH": {
        "btts_leg_multiplier": 0.75,
        "ftr_leg_multiplier": 1.00,
        "stake_multiplier": 0.90,
        "overlay_note": "Reduce BTTS after UCL R16 second legs in March; FTR unchanged.",
    },
}

V2_PRIORITY_TIER_WEIGHT = {
    "ELITE": 1.00,
    "STANDARD": 0.82,
    "OBSERVE": 0.45,
}

V2_PRIORITY_MARKET_WEIGHT = {
    "ftr": 1.00,
    "over25": 0.90,
    "under25": 0.86,
    "btts": 0.92,
    "btts_no": 0.88,
}


import argparse
import glob
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Column normalisation helpers
# -----------------------------

POOL_COL_ALIASES = {
    # canonical -> candidates
    "fixture_key": ["fixture_key", "match_key", "fixtureId", "fixture_id", "key"],
    "league": ["league", "league_tag", "League", "division", "Div", "country_league"],
    "match_date": ["match_date", "date", "date_GMT", "Date", "kickoff", "timestamp"],
    "home_team": ["home_team", "home_team_name", "HomeTeam", "home"],
    "away_team": ["away_team", "away_team_name", "AwayTeam", "away"],
    "od_source": ["od_source", "odds_source", "price_source"],
    "market": ["market", "Market", "mkt"],
    "selection": ["selection", "Selection", "bookie_pick", "pick", "model_top_pick"],
    "odds": ["od", "odds", "price", "bookie_od", "bookie_odds", "bookie_price", "B365", "avg_odds"],
    "bookie_implied": ["bookie_implied", "implied_prob", "bookie_prob", "bookie_p"],
    "model_p": ["model_p_for_bookie", "p_model", "confidence", "model_prob", "prob", "p"],
    "gap": ["gap_used", "gap", "edge", "implied_gap", "p_gap"],
    "ftr_margin": ["ftr_margin", "margin", "top_margin"],
    "signal_over25": ["signal_over25", "over25_signal", "signal_ou25"],
    "signal_btts": ["signal_btts", "btts_signal"],
    "source_tier": ["source_tier", "source_tier_file", "tier", "deploy_tier"],
    "standard_reporting_bucket": ["standard_reporting_bucket"],
    "product_profile": ["product_profile"],
    "product_lane": ["product_lane"],
    "calendar_regime": ["calendar_regime"],
    "btts_leg_multiplier": ["btts_leg_multiplier"],
    "ftr_leg_multiplier": ["ftr_leg_multiplier"],
    "stake_multiplier": ["stake_multiplier"],
    "overlay_note": ["overlay_note"],
    "context_reason_codes": ["context_reason_codes"],
    "variant_reject_reason": ["variant_reject_reason"],
    # Poisson lambda + correct-score shortlist (from bookie_allmarkets / deploy_rulebook)
    "home_goals_pred": ["home_goals_pred", "lambda_home", "home_goals", "mu_home", "lam_h"],
    "away_goals_pred": ["away_goals_pred", "lambda_away", "away_goals", "mu_away", "lam_a"],
    "exp_goals_sum": ["exp_goals_sum", "lambda_total", "lam_total", "xg_sum"],
    "cs1": ["cs1"],
    "cs1_p": ["cs1_p", "cs1_prob"],
    "cs2": ["cs2"],
    "cs2_p": ["cs2_p", "cs2_prob"],
    "cs3": ["cs3"],
    "cs3_p": ["cs3_p", "cs3_prob"],
    "p_home_pois": ["p_home_pois", "p_home_poisson"],
    "p_draw_pois": ["p_draw_pois", "p_draw_poisson"],
    "p_away_pois": ["p_away_pois", "p_away_poisson"],
    "cs_trunc_mass_0_6": ["cs_trunc_mass_0_6", "cs_trunc_mass"],
}

MATCHES_COL_ALIASES = {
    "league": ["league", "league_tag", "League", "Div", "division"],
    "match_date": ["match_date", "date", "date_GMT", "Date", "timestamp"],
    "home_team": ["home_team_name", "HomeTeam", "home_team", "home"],
    "away_team": ["away_team_name", "AwayTeam", "away_team", "away"],
    "home_goals": [
        # OG / internal
        "home_goals",
        "home_team_goal_count",
        "home_team_goals",
        "home_score",
        "score_home",
        # football-data.co.uk / common
        "FTHG",
        "HG",
        "HomeGoals",
        "homeGoals",
    ],
    "away_goals": [
        # OG / internal
        "away_goals",
        "away_team_goal_count",
        "away_team_goals",
        "away_score",
        "score_away",
        # football-data.co.uk / common
        "FTAG",
        "AG",
        "AwayGoals",
        "awayGoals",
    ],
    "status": ["status", "Status", "match_status"],
    "fixture_key": ["fixture_key", "match_key", "fixtureId", "fixture_id", "key"],
}



def _pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# -----------------------------
# Duplicate-column coalescing (guards against alias renames creating dupes)
# -----------------------------

def _coalesce_duplicate_column(df: pd.DataFrame, col: str, *, kind: str = "string") -> pd.DataFrame:
    """If `df` has duplicate columns named `col`, coalesce them into a single column.

    pandas returns a DataFrame for df[col] when there are duplicate column names.
    We merge duplicates row-wise by taking the first non-null (and for strings, first non-empty).
    """
    if df is None or df.empty:
        return df
    if col not in df.columns:
        return df

    # If not duplicated, nothing to do
    try:
        if int((df.columns == col).sum()) <= 1:
            return df
    except Exception:
        return df

    block = df.loc[:, df.columns == col]
    if not isinstance(block, pd.DataFrame) or block.shape[1] <= 1:
        return df

    if kind == "numeric":
        blk = block.apply(pd.to_numeric, errors="coerce")
        merged = blk.bfill(axis=1).iloc[:, 0]
    else:
        # string: treat empty strings as missing
        blk = block.astype("string").fillna("")
        blk = blk.mask(blk.eq(""), pd.NA)
        merged = blk.bfill(axis=1).iloc[:, 0]
        merged = merged.astype("string").fillna("")

    # Assign merged series back to the first occurrence, then drop other duplicates
    df[col] = merged
    # Keep first occurrence of each column name globally (safe after coalescing key columns)
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")].copy()
    return df


def _coalesce_common_dupes(df: pd.DataFrame) -> pd.DataFrame:
    """Coalesce duplicates for the canonical columns we depend on."""
    if df is None or df.empty:
        return df

    # Strings first
    for c in [
        "fixture_key",
        "league",
        "match_date",
        "home_team",
        "away_team",
        "od_source",
        "market",
        "selection",
        "signal_over25",
        "signal_btts",
        "source_tier",
        "standard_reporting_bucket",
        "product_profile",
        "product_lane",
        "calendar_regime",
        "overlay_note",
        "context_reason_codes",
        "variant_reject_reason",
    ]:
        df = _coalesce_duplicate_column(df, c, kind="string")

    # Numerics
    for c in ["odds", "model_p", "bookie_implied", "gap", "ftr_margin", "btts_leg_multiplier", "ftr_leg_multiplier", "stake_multiplier"]:
        df = _coalesce_duplicate_column(df, c, kind="numeric")

    return df


def normalise_pool_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    ren = {}
    for canon, cands in POOL_COL_ALIASES.items():
        found = _pick_first_existing(df, cands)
        if found and found != canon:
            ren[found] = canon
    if ren:
        df = df.rename(columns=ren)

    # Guard: alias renames can create duplicate canonical columns (e.g. multiple odds columns -> 'odds').
    # Coalesce duplicates before any type coercions.
    df = _coalesce_common_dupes(df)

    # mandatory fields
    if "fixture_key" not in df.columns:
        raise ValueError("Pool CSV is missing fixture_key (required).")
    if "market" not in df.columns:
        raise ValueError("Pool CSV is missing market (required).")
    if "selection" not in df.columns:
        raise ValueError("Pool CSV is missing selection/bookie_pick (required).")

    # optional but strongly recommended
    if "league" not in df.columns:
        df["league"] = "UNKNOWN_LEAGUE"
    if "match_date" not in df.columns:
        df["match_date"] = pd.NaT
    if "home_team" not in df.columns:
        df["home_team"] = ""
    if "away_team" not in df.columns:
        df["away_team"] = ""
    if "od_source" not in df.columns:
        df["od_source"] = "unknown"
    if "source_tier" not in df.columns:
        df["source_tier"] = "UNKNOWN"
    if "standard_reporting_bucket" not in df.columns:
        df["standard_reporting_bucket"] = ""
    if "product_profile" not in df.columns:
        df["product_profile"] = ""
    if "product_lane" not in df.columns:
        df["product_lane"] = ""
    if "calendar_regime" not in df.columns:
        df["calendar_regime"] = "NORMAL"
    if "overlay_note" not in df.columns:
        df["overlay_note"] = ""
    if "context_reason_codes" not in df.columns:
        df["context_reason_codes"] = ""
    if "variant_reject_reason" not in df.columns:
        df["variant_reject_reason"] = ""

    # Poisson / correct-score optional columns (kept for slip display if present)
    for c in [
        "home_goals_pred", "away_goals_pred", "exp_goals_sum",
        "cs1", "cs1_p", "cs2", "cs2_p", "cs3", "cs3_p",
        "p_home_pois", "p_draw_pois", "p_away_pois", "cs_trunc_mass_0_6",
    ]:
        if c not in df.columns:
            df[c] = np.nan if c.endswith("_p") or c.startswith("p_") or c.endswith("_sum") or c.endswith("_mass_0_6") or c.endswith("_pred") else ""

    # odds
    if "odds" not in df.columns:
        df["odds"] = np.nan

    # probabilities
    if "model_p" not in df.columns:
        df["model_p"] = np.nan
    if "bookie_implied" not in df.columns:
        df["bookie_implied"] = np.nan
    if "gap" not in df.columns:
        df["gap"] = df["model_p"] - df["bookie_implied"]
    if "btts_leg_multiplier" not in df.columns:
        df["btts_leg_multiplier"] = 1.0
    if "ftr_leg_multiplier" not in df.columns:
        df["ftr_leg_multiplier"] = 1.0
    if "stake_multiplier" not in df.columns:
        df["stake_multiplier"] = 1.0

    # parse match_date into date (not datetime) where possible
    # parse match_date into date (not datetime) where possible
    if "match_date" in df.columns:
        s_md = df["match_date"]
        if isinstance(s_md, pd.DataFrame):
            # pick first column if somehow still duplicated
            s_md = s_md.bfill(axis=1).iloc[:, 0]
        df["match_date"] = pd.to_datetime(s_md, errors="coerce", utc=True).dt.date
    else:
        df["match_date"] = pd.NaT

    # coerce numeric columns
    for col in ["odds", "model_p", "bookie_implied", "gap", "ftr_margin", "btts_leg_multiplier", "ftr_leg_multiplier", "stake_multiplier"]:
        if col in df.columns:
            x = df[col]
            if isinstance(x, pd.DataFrame):
                # if duplicates remain, coalesce again
                df = _coalesce_duplicate_column(df, col, kind="numeric")
                x = df[col]
            df[col] = pd.to_numeric(x, errors="coerce")

    # canonical market naming normalization
    df["market"] = df["market"].astype(str).str.strip().str.lower()
    df["selection"] = df["selection"].astype(str).str.strip().str.upper()

    # normalize some common market labels
    df["market"] = df["market"].replace(
        {
            "ou25": "ou25",
            "over25": "over25",
            "under25": "under25",
            "btts": "btts",
            "ftr": "ftr",
            "1x2": "ftr",
        }
    )

    # If market is ou25 and selection is OVER25/UNDER25, map to over25/under25 for scoring/backtest
    is_ou = df["market"].eq("ou25")
    df.loc[is_ou & df["selection"].str.contains("OVER"), "market"] = "over25"
    df.loc[is_ou & df["selection"].str.contains("UNDER"), "market"] = "under25"

    # If selection implies BTTS NO, normalize market
    df.loc[df["market"].eq("btts") & df["selection"].isin(["NO", "BTTS_NO", "BTTS N", "N"]), "market"] = "btts_no"
    df.loc[df["market"].eq("btts_no"), "selection"] = "NO"
    df.loc[df["market"].eq("btts") & df["selection"].isin(["YES", "BTTS_YES", "BTTS Y", "Y"]), "selection"] = "YES"

    return df


def normalise_matches_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ren = {}
    for canon, cands in MATCHES_COL_ALIASES.items():
        found = _pick_first_existing(df, cands)
        if found and found != canon:
            ren[found] = canon
    if ren:
        df = df.rename(columns=ren)

    # parse date
    if "match_date" in df.columns:
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce", utc=True).dt.date
    else:
        df["match_date"] = pd.NaT

    # numeric goals
    for col in ["home_goals", "away_goals"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    if "status" not in df.columns:
        df["status"] = ""

    if "league" not in df.columns:
        df["league"] = "UNKNOWN_LEAGUE"

    # ensure team cols exist for fallback key
    for col in ["home_team", "away_team"]:
        if col not in df.columns:
            df[col] = ""

    return df


# -----------------------------
# v2 deploy ingestion + overlay helpers
# -----------------------------


def safe_upper_str_series(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.upper().str.strip()


def _standardize_source_tier_label(x: object) -> object:
    """Normalize source tier labels for either a scalar value or a pandas Series."""
    if isinstance(x, pd.Series):
        s = x.astype("string").fillna("").str.strip().str.upper()
        out = pd.Series("UNKNOWN", index=s.index, dtype="string")
        out = out.mask(s.str.contains("ELITE", na=False), "ELITE")
        out = out.mask(s.str.contains("STANDARD", na=False), "STANDARD")
        out = out.mask(s.str.contains("OBSERVE", na=False), "OBSERVE")
        return out

    s = str("" if x is None else x).strip().upper()
    if not s:
        return "UNKNOWN"
    if "ELITE" in s:
        return "ELITE"
    if "STANDARD" in s:
        return "STANDARD"
    if "OBSERVE" in s:
        return "OBSERVE"
    return s


def infer_source_tier_from_path(path: str) -> str:
    s = str(path or "").upper()
    for tier, token in DEPLOY_TIER_TOKEN_MAP.items():
        if token in s:
            return tier
    return "UNKNOWN"


def get_calendar_policy(regime: object) -> dict:
    key = str(regime or "NORMAL").strip().upper() or "NORMAL"
    return CALENDAR_REGIME_POLICY.get(key, CALENDAR_REGIME_POLICY["NORMAL"])


def stamp_calendar_overlay_fields(df: pd.DataFrame, default_regime: str = "NORMAL") -> pd.DataFrame:
    df = df.copy()
    if "calendar_regime" not in df.columns:
        df["calendar_regime"] = default_regime
    df["calendar_regime"] = safe_upper_str_series(df["calendar_regime"]).replace("", default_regime)

    regimes = df["calendar_regime"].tolist()
    df["btts_leg_multiplier"] = [float(get_calendar_policy(r)["btts_leg_multiplier"]) for r in regimes]
    df["ftr_leg_multiplier"] = [float(get_calendar_policy(r)["ftr_leg_multiplier"]) for r in regimes]
    df["stake_multiplier"] = [float(get_calendar_policy(r)["stake_multiplier"]) for r in regimes]
    df["overlay_note"] = [str(get_calendar_policy(r)["overlay_note"]) for r in regimes]
    return df


def compute_leg_priority(df: pd.DataFrame) -> pd.Series:
    model_p = pd.to_numeric(df.get("model_p", np.nan), errors="coerce").fillna(0.0)
    gap = pd.to_numeric(df.get("gap", np.nan), errors="coerce").fillna(0.0)
    odds = pd.to_numeric(df.get("odds", np.nan), errors="coerce")

    tier_weight = safe_upper_str_series(df.get("source_tier", pd.Series("UNKNOWN", index=df.index))).map(V2_PRIORITY_TIER_WEIGHT).fillna(0.35)
    market_key = df.get("market", pd.Series("", index=df.index)).astype("string").fillna("").str.lower().str.strip()
    market_weight = market_key.map(V2_PRIORITY_MARKET_WEIGHT).fillna(0.70)

    ftr_mult = pd.to_numeric(df.get("ftr_leg_multiplier", 1.0), errors="coerce").fillna(1.0)
    btts_mult = pd.to_numeric(df.get("btts_leg_multiplier", 1.0), errors="coerce").fillna(1.0)
    stake_mult = pd.to_numeric(df.get("stake_multiplier", 1.0), errors="coerce").fillna(1.0)

    calendar_mult = pd.Series(1.0, index=df.index, dtype="float64")
    calendar_mult = calendar_mult.mask(market_key.eq("ftr"), ftr_mult)
    calendar_mult = calendar_mult.mask(market_key.isin(["btts", "btts_no"]), btts_mult)

    short_odds_bonus = pd.Series(0.0, index=df.index, dtype="float64")
    short_odds_bonus = short_odds_bonus.mask(odds.le(1.40), 0.06)
    short_odds_bonus = short_odds_bonus.mask(odds.gt(2.40), -0.03)

    product_bonus = pd.Series(0.0, index=df.index, dtype="float64")
    if "product_profile" in df.columns:
        product_bonus = product_bonus + safe_upper_str_series(df["product_profile"]).map({
            "ELITE": 0.08,
            "STANDARD": 0.04,
        }).fillna(0.0)

    lane_bonus = pd.Series(0.0, index=df.index, dtype="float64")
    if "product_lane" in df.columns:
        lane_bonus = lane_bonus + safe_upper_str_series(df["product_lane"]).map({
            "PROFIT_FIRST": 0.04,
            "STANDARD_FTR_CS_PROMOTED": 0.03,
            "NON_PROFIT_FIRST": 0.00,
        }).fillna(0.0)

    base = (1.20 * model_p) + (0.35 * gap)
    priority = (base * tier_weight * market_weight * calendar_mult * stake_mult) + short_odds_bonus + product_bonus + lane_bonus
    return priority.astype(float)


def load_deploy_tier_csvs(paths: List[str]) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []
    for path in paths:
        p = str(path)
        d = pd.read_csv(p, low_memory=False)
        d = normalise_pool_columns(d)
        d["__source_file"] = p
        if "source_tier" in d.columns:
            d["source_tier"] = _standardize_source_tier_label(d["source_tier"])
        else:
            d["source_tier"] = _standardize_source_tier_label(infer_source_tier_from_path(p))
        dfs.append(d)
    if not dfs:
        raise FileNotFoundError("No deploy tier CSVs could be loaded.")
    out = pd.concat(dfs, ignore_index=True)
    out["source_tier"] = safe_upper_str_series(out["source_tier"]).map(_standardize_source_tier_label)
    return out


def collect_deploy_tier_paths(inputs: List[str]) -> List[str]:
    out: list[str] = []
    for raw in inputs:
        p = Path(str(raw))
        if p.is_file():
            out.append(str(p))
            continue
        if p.is_dir():
            for tier in ("ELITE", "STANDARD", "OBSERVE"):
                token = DEPLOY_TIER_TOKEN_MAP[tier]
                out.extend(sorted(str(x) for x in p.glob(f"*{token}*.csv")))
    deduped = []
    seen = set()
    for x in out:
        if x not in seen:
            deduped.append(x)
            seen.add(x)
    return deduped


def build_canonical_v2_pool(
    deploy_inputs: List[str],
    include_tiers: Tuple[str, ...] = DEFAULT_INCLUDE_TIERS_V2,
    exclude_tiers: Tuple[str, ...] = DEFAULT_EXCLUDE_TIERS_V2,
    allow_observe: bool = False,
) -> pd.DataFrame:
    tier_paths = collect_deploy_tier_paths(deploy_inputs)
    if not tier_paths:
        raise FileNotFoundError("No deploy tier CSVs found from the provided inputs.")

    pool = load_deploy_tier_csvs(tier_paths)
    pool["source_tier"] = safe_upper_str_series(pool["source_tier"]).map(_standardize_source_tier_label)

    keep_tiers = {str(x).upper() for x in include_tiers}
    drop_tiers = {str(x).upper() for x in exclude_tiers}
    if allow_observe:
        drop_tiers.discard("OBSERVE")
        keep_tiers.add("OBSERVE")

    pool = pool[pool["source_tier"].isin(keep_tiers)].copy()
    if drop_tiers:
        pool = pool[~pool["source_tier"].isin(drop_tiers)].copy()

    pool["fixture_key"] = pool["fixture_key"].astype("string").fillna("").str.strip()
    pool = pool[pool["fixture_key"].ne("")].copy()

    pool = stamp_calendar_overlay_fields(pool, default_regime="NORMAL")
    pool["leg_priority"] = compute_leg_priority(pool)
    pool = pool.sort_values(["leg_priority", "model_p", "odds"], ascending=[False, False, True]).reset_index(drop=True)
    return pool


def summarize_v2_pool(pool: pd.DataFrame) -> pd.DataFrame:
    if pool.empty:
        return pd.DataFrame(columns=[
            "source_tier",
            "market",
            "rows",
            "avg_odds",
            "avg_model_p",
            "avg_leg_priority",
            "calendar_regimes",
        ])

    grp = pool.groupby(["source_tier", "market"], dropna=False).agg(
        rows=("fixture_key", "count"),
        avg_odds=("odds", lambda x: float(pd.to_numeric(x, errors="coerce").mean())),
        avg_model_p=("model_p", lambda x: float(pd.to_numeric(x, errors="coerce").mean())),
        avg_leg_priority=("leg_priority", lambda x: float(pd.to_numeric(x, errors="coerce").mean())),
        calendar_regimes=("calendar_regime", lambda x: "|".join(sorted(set(safe_upper_str_series(pd.Series(x)).tolist())))),
    ).reset_index()
    return grp.sort_values(["source_tier", "market"]).reset_index(drop=True)
def _df_to_markdown_safe(df: pd.DataFrame, *, index: bool = False) -> str:
    """Render markdown when tabulate is available, else plain text."""
    if df is None:
        return ""
    try:
        return df.to_markdown(index=index)
    except Exception:
        return df.to_string(index=index)

def cmd_build_v2(args: argparse.Namespace) -> None:
    deploy_inputs = list(args.deploy_input or [])
    if not deploy_inputs:
        raise ValueError("build_v2 requires at least one --deploy-input path (file or directory).")

    include_tiers = tuple(x.strip().upper() for x in str(args.include_tiers).split(",") if x.strip())
    exclude_tiers = tuple(x.strip().upper() for x in str(args.exclude_tiers).split(",") if x.strip())

    pool = build_canonical_v2_pool(
        deploy_inputs=deploy_inputs,
        include_tiers=include_tiers or DEFAULT_INCLUDE_TIERS_V2,
        exclude_tiers=exclude_tiers or DEFAULT_EXCLUDE_TIERS_V2,
        allow_observe=bool(args.allow_observe),
    )

    if args.only_markets:
        keep = {m.strip().lower() for m in str(args.only_markets).split(",") if m.strip()}
        pool = pool[pool["market"].isin(keep)].copy()

    summary = summarize_v2_pool(pool)

    out_dir = args.out_dir or "."
    os.makedirs(out_dir, exist_ok=True)

    pool_path = os.path.join(out_dir, f"acca_pool_{args.tag}.csv")
    summary_path = os.path.join(out_dir, f"acca_summary_{args.tag}.csv")
    investor_md_path = os.path.join(out_dir, f"acca_investor_summary_{args.tag}.md")

    pool.to_csv(pool_path, index=False)
    summary.to_csv(summary_path, index=False)

    with open(investor_md_path, "w", encoding="utf-8") as fh:
        fh.write(f"# Acca Investor Summary — {args.tag}\n\n")
        fh.write("## Build mode\n\n")
        fh.write("deploy-driven v2 canonical pool build\n\n")
        fh.write("## Included tiers\n\n")
        fh.write(f"{', '.join(include_tiers or DEFAULT_INCLUDE_TIERS_V2)}\n\n")
        fh.write("## Excluded tiers\n\n")
        fh.write(f"{', '.join(exclude_tiers or DEFAULT_EXCLUDE_TIERS_V2)}\n\n")
        fh.write("## Pool summary\n\n")
        if summary.empty:
            fh.write("No rows in canonical pool.\n")
        else:
            fh.write(_df_to_markdown_safe(summary, index=False))
            fh.write("\n")

    print(f"[OK] wrote: {pool_path}")
    print(f"[OK] wrote: {summary_path}")
    print(f"[OK] wrote: {investor_md_path}")
    print(f"[INFO] canonical pool rows: {len(pool)}")

    run_json_path = os.path.join(out_dir, f"acca_pool_{args.tag}__RUN.json")
    _write_json(
        run_json_path,
        {
            "cmd": "build_v2",
            "created_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "tag": args.tag,
            "deploy_inputs": [os.path.abspath(x) for x in deploy_inputs],
            "include_tiers": list(include_tiers or DEFAULT_INCLUDE_TIERS_V2),
            "exclude_tiers": list(exclude_tiers or DEFAULT_EXCLUDE_TIERS_V2),
            "allow_observe": bool(args.allow_observe),
            "only_markets": args.only_markets,
            "out_dir": os.path.abspath(out_dir),
            "git_commit": _git_commit_hash(),
            "args": vars(args),
            "pool_rows": int(len(pool)),
            "summary_rows": int(len(summary)),
        },
    )
    print(f"[OK] wrote: {run_json_path}")


# -----------------------------
# Signal label strength mapping
# -----------------------------

LABEL_STRENGTH = {
    # generic
    "VERY_STRONG": 1.00,
    "STRONG": 0.70,
    "MEDIUM": 0.35,
    "WEAK": 0.10,
    "VERY_WEAK": -0.05,
    # btts specifics
    "VERY_STRONG_YES": 1.00,
    "STRONG_YES": 0.70,
    "MEDIUM_YES": 0.35,
    "WEAK_YES": 0.10,
    "VERY_STRONG_NO": 1.00,
    "STRONG_NO": 0.70,
    "MEDIUM_NO": 0.35,
    "WEAK_NO": 0.10,
}


def label_bonus(x: object) -> float:
    if pd.isna(x):
        return 0.0
    s = str(x).strip().upper()
    return LABEL_STRENGTH.get(s, 0.0)


# -----------------------------
# Scoring
# -----------------------------

def compute_pick_score(df: pd.DataFrame) -> pd.Series:
    """
    Score picks so the builder can rank legs.

    Priority:
    - high model_p
    - positive gap (model_p - bookie_implied)
    - ftr_margin (if exists)
    - signal label strength bonuses
    - small preference for 'reasonable' odds (handled later as glue/anchor/monster shape)
    """
    model_p = df["model_p"].fillna(0.0)
    gap = df["gap"].fillna(0.0)
    margin = df["ftr_margin"].fillna(0.0) if "ftr_margin" in df.columns else 0.0

    bonus = 0.0
    if "signal_over25" in df.columns:
        bonus = bonus + df["signal_over25"].apply(label_bonus)
    if "signal_btts" in df.columns:
        bonus = bonus + df["signal_btts"].apply(label_bonus)

    # keep weights modest: you already learned gating elsewhere
    score = (
        1.00 * model_p
        + 0.30 * gap
        + 0.10 * margin
        + 0.05 * bonus
    )
    return score


def format_cs_note(row: pd.Series) -> str:
    """Format a compact correct-score shortlist note for a leg.

    Example:
      CS: 1-1 (0.113), 2-1 (0.094), 1-0 (0.091)

    Returns empty string if the required columns are missing.
    """
    try:
        s1 = str(row.get("cs1", "") or "").strip()
        s2 = str(row.get("cs2", "") or "").strip()
        s3 = str(row.get("cs3", "") or "").strip()
        p1 = pd.to_numeric(row.get("cs1_p", np.nan), errors="coerce")
        p2 = pd.to_numeric(row.get("cs2_p", np.nan), errors="coerce")
        p3 = pd.to_numeric(row.get("cs3_p", np.nan), errors="coerce")

        parts = []
        if s1 and np.isfinite(p1):
            parts.append(f"{s1} ({float(p1):.3f})")
        if s2 and np.isfinite(p2):
            parts.append(f"{s2} ({float(p2):.3f})")
        if s3 and np.isfinite(p3):
            parts.append(f"{s3} ({float(p3):.3f})")

        if not parts:
            return ""
        return "CS: " + ", ".join(parts)
    except Exception:
        return ""



# -----------------------------
# v2 template-driven slip construction
# -----------------------------

V2_TEMPLATE_SPECS = {
    "FTR_STANDARD_INVESTOR": {
        "description": "FTR-heavy investor template built from ELITE + STANDARD deploy truth.",
        "template_class": "monster_acca",
        "base_k": 6,
        "base_slips": 12,
        "included_markets": ["ftr"],
        "included_tiers": ["ELITE", "STANDARD"],
        "market_min": {"ftr": 5},
        "market_max": {"ftr": 6},
        "max_per_league": 3,
        "max_per_date": 5,
        "allow_same_fixture_multi_market": False,
    },
    "BTTS_ELITE_GOALS": {
        "description": "Goals-focused template prioritising BTTS and OVER25 from stronger tiers.",
        "template_class": "coverage_system",
        "base_k": 6,
        "base_slips": 12,
        "included_markets": ["btts", "btts_no", "over25"],
        "included_tiers": ["ELITE", "STANDARD"],
        "market_min": {"btts": 2, "over25": 1},
        "market_max": {"btts": 4, "btts_no": 1, "over25": 3},
        "max_per_league": 4,
        "max_per_date": 6,
        "allow_same_fixture_multi_market": True,
    },
    "OU25_STANDARD_STRUCTURED": {
        "description": "Structured OU25 template using OVER25 / UNDER25 legs from production tiers.",
        "template_class": "coverage_system",
        "base_k": 6,
        "base_slips": 12,
        "included_markets": ["over25", "under25"],
        "included_tiers": ["ELITE", "STANDARD"],
        "market_min": {"over25": 2},
        "market_max": {"over25": 5, "under25": 3},
        "max_per_league": 4,
        "max_per_date": 6,
        "allow_same_fixture_multi_market": True,
    },
    "HYBRID_BALANCED": {
        "description": "Balanced hybrid template across FTR, BTTS and OVER25 using deploy-approved rows.",
        "template_class": "coverage_system",
        "base_k": 6,
        "base_slips": 12,
        "included_markets": ["ftr", "btts", "btts_no", "over25", "under25"],
        "included_tiers": ["ELITE", "STANDARD"],
        "market_min": {"ftr": 2, "over25": 1},
        "market_max": {"ftr": 3, "btts": 2, "btts_no": 1, "over25": 2, "under25": 1},
        "max_per_league": 4,
        "max_per_date": 6,
        "allow_same_fixture_multi_market": True,
    },
    "FTR_ONLY": {
        "description": "FTR-only ranked acca template using deploy-approved ELITE + STANDARD legs.",
        "template_class": "monster_acca",
        "base_k": 8,
        "base_slips": 10,
        "included_markets": ["ftr"],
        "included_tiers": ["ELITE", "STANDARD"],
        "market_min": {"ftr": 8},
        "market_max": {"ftr": 16},
        "max_per_league": 3,
        "max_per_date": 5,
        "allow_same_fixture_multi_market": False,
    },
    "BTTS_ONLY": {
        "description": "BTTS-only ranked acca template using deploy-approved ELITE + STANDARD legs.",
        "template_class": "coverage_system",
        "base_k": 8,
        "base_slips": 10,
        "included_markets": ["btts", "btts_no"],
        "included_tiers": ["ELITE", "STANDARD"],
        "market_min": {"btts": 4},
        "market_max": {"btts": 16, "btts_no": 4},
        "max_per_league": 4,
        "max_per_date": 6,
        "allow_same_fixture_multi_market": True,
    },
    "OU25_ONLY": {
        "description": "OU25-only ranked acca template using OVER25 / UNDER25 deploy-approved legs.",
        "template_class": "coverage_system",
        "base_k": 8,
        "base_slips": 10,
        "included_markets": ["over25", "under25"],
        "included_tiers": ["ELITE", "STANDARD"],
        "market_min": {"over25": 4},
        "market_max": {"over25": 16, "under25": 6},
        "max_per_league": 4,
        "max_per_date": 6,
        "allow_same_fixture_multi_market": True,
    },
    "COMBINED_ALLMARKETS": {
        "description": "Combined FTR + BTTS + OU25 ranked acca template using deploy-approved ELITE + STANDARD legs.",
        "template_class": "monster_acca",
        "base_k": 10,
        "base_slips": 12,
        "included_markets": ["ftr", "btts", "btts_no", "over25", "under25"],
        "included_tiers": ["ELITE", "STANDARD"],
        "market_min": {"ftr": 3, "btts": 2, "over25": 2},
        "market_max": {"ftr": 6, "btts": 4, "btts_no": 2, "over25": 5, "under25": 3},
        "max_per_league": 3,
        "max_per_date": 5,
        "allow_same_fixture_multi_market": False,
    },
}


# --- Helper functions for same-fixture multi-market logic ---

def _market_family_key(market: object, selection: object = None) -> str:
    m = str(market or "").strip().lower()
    s = str(selection or "").strip().upper()

    if m == "ftr":
        return "ftr"
    if m in {"btts", "btts_no"}:
        return "btts"
    if m in {"over25", "under25", "ou25"}:
        return "ou25"
    if m == "ou25" and "UNDER" in s:
        return "ou25"
    if m == "ou25" and "OVER" in s:
        return "ou25"
    return m or "unknown"


def _same_fixture_combo_allowed(existing_row: pd.Series, cand_row: pd.Series, spec: dict) -> bool:
    """Coverage templates may allow compatible same-fixture multi-market coexistence.

    Allowed only when template explicitly opts in, and only across different market families.
    This permits combinations like FTR + BTTS or FTR + OU25 or BTTS + OU25 on the same fixture,
    while still blocking same-family duplicates such as BTTS + BTTS_NO or OVER25 + UNDER25.
    """
    if not bool(spec.get("allow_same_fixture_multi_market", False)):
        return False

    template_class = str(spec.get("template_class", "")).strip().lower()
    if template_class != "coverage_system":
        return False

    existing_family = _market_family_key(existing_row.get("market", ""), existing_row.get("selection", ""))
    cand_family = _market_family_key(cand_row.get("market", ""), cand_row.get("selection", ""))

    if not existing_family or not cand_family:
        return False

    return existing_family != cand_family


def get_run_calendar_regime(pool: pd.DataFrame) -> str:
    if pool is None or pool.empty or "calendar_regime" not in pool.columns:
        return "NORMAL"
    s = safe_upper_str_series(pool["calendar_regime"])
    if s.empty:
        return "NORMAL"
    mode = s.mode(dropna=True)
    if len(mode):
        return str(mode.iloc[0])
    first = str(s.iloc[0]).strip()
    return first or "NORMAL"



def _template_effective_k(base_k: int, regime: str, included_markets: list[str]) -> int:
    regime = str(regime or "NORMAL").strip().upper()
    included = {str(x).lower().strip() for x in included_markets}
    k = int(base_k)

    if regime == "POST_FIFA_BREAK" and ({"btts", "btts_no"} & included):
        k -= 1
    elif regime == "PRE_UCL_KNOCKOUT_FIRST_LEG" and ("ftr" in included):
        k -= 1
    elif regime == "NOV_FIFA_PLUS_PRE_UCL":
        k -= 2
    elif regime == "POST_R16_SECOND_LEG_MARCH" and ({"btts", "btts_no"} & included):
        k -= 1

    return max(4, k)



def apply_calendar_shortlist_shaping(pool: pd.DataFrame) -> pd.DataFrame:
    if pool is None or pool.empty:
        return pool.copy()

    df = pool.copy()
    regime = get_run_calendar_regime(df)
    market_key = df["market"].astype("string").fillna("").str.lower().str.strip()
    tier_key = safe_upper_str_series(df.get("source_tier", pd.Series("UNKNOWN", index=df.index)))
    model_p = pd.to_numeric(df.get("model_p", np.nan), errors="coerce").fillna(0.0)

    suppress = pd.Series(False, index=df.index)

    if regime == "POST_FIFA_BREAK":
        suppress = suppress | (market_key.isin(["btts", "btts_no"]) & (tier_key.ne("ELITE")) & (model_p < 0.58))
    elif regime == "PRE_UCL_KNOCKOUT_FIRST_LEG":
        suppress = suppress | (market_key.eq("ftr") & (tier_key.eq("STANDARD")) & (model_p < 0.52))
    elif regime == "NOV_FIFA_PLUS_PRE_UCL":
        suppress = suppress | (market_key.isin(["btts", "btts_no"]) & (model_p < 0.60))
        suppress = suppress | (market_key.eq("ftr") & (model_p < 0.54))
    elif regime == "POST_R16_SECOND_LEG_MARCH":
        suppress = suppress | (market_key.isin(["btts", "btts_no"]) & (model_p < 0.58))

    df["calendar_shortlist_suppressed"] = suppress.astype(int)
    df = df.loc[~suppress].copy()
    df = df.sort_values(["leg_priority", "model_p", "odds"], ascending=[False, False, True]).reset_index(drop=True)
    return df



def build_v2_template_pool(pool: pd.DataFrame, template_name: str) -> tuple[pd.DataFrame, dict, str]:
    if template_name not in V2_TEMPLATE_SPECS:
        raise ValueError(f"Unknown v2 template: {template_name}")

    spec = V2_TEMPLATE_SPECS[template_name].copy()
    regime = get_run_calendar_regime(pool)

    df = pool.copy()
    df["source_tier"] = safe_upper_str_series(df.get("source_tier", pd.Series("UNKNOWN", index=df.index)))
    df["market"] = df["market"].astype("string").fillna("").str.lower().str.strip()

    df = df[df["source_tier"].isin([str(x).upper() for x in spec["included_tiers"]])].copy()
    df = df[df["market"].isin([str(x).lower() for x in spec["included_markets"]])].copy()
    df = apply_calendar_shortlist_shaping(df)

    spec["effective_k"] = _template_effective_k(spec["base_k"], regime, spec["included_markets"])
    spec["recommended_stake_multiplier"] = float(get_calendar_policy(regime)["stake_multiplier"])
    spec["calendar_regime"] = regime
    return df.reset_index(drop=True), spec, regime



def _v2_market_count(chosen_rows: list[pd.Series], market: str) -> int:
    m = str(market).lower().strip()
    return int(sum(1 for row in chosen_rows if str(row.get("market", "")).lower().strip() == m))



def _v2_can_add(
    cand: pd.Series,
    chosen_rows: list[pd.Series],
    chosen_fks: set[str],
    league_counts: dict[str, int],
    date_counts: dict[object, int],
    spec: dict,
) -> bool:
    fk = str(cand.get("fixture_key", "")).strip()
    if not fk:
        return False

    cand_market = str(cand.get("market", "")).strip().lower()
    cand_selection = str(cand.get("selection", "")).strip().upper()

    same_fixture_rows = [
        row for row in chosen_rows
        if str(row.get("fixture_key", "")).strip() == fk
    ]

    # Hard block exact same leg reuse
    for row in same_fixture_rows:
        row_market = str(row.get("market", "")).strip().lower()
        row_selection = str(row.get("selection", "")).strip().upper()
        if row_market == cand_market and row_selection == cand_selection:
            return False

    if same_fixture_rows:
        # Coverage templates may allow same-fixture coexistence,
        # but only across different market families
        if not bool(spec.get("allow_same_fixture_multi_market", False)):
            return False

        if not all(_same_fixture_combo_allowed(row, cand, spec) for row in same_fixture_rows):
            return False

    league = str(cand.get("league", "UNKNOWN_LEAGUE")).strip()
    match_date = cand.get("match_date", pd.NaT)
    market = str(cand.get("market", "")).lower().strip()

    if int(spec.get("max_per_league", 0) or 0) > 0:
        if int(league_counts.get(league, 0)) >= int(spec["max_per_league"]):
            return False

    if int(spec.get("max_per_date", 0) or 0) > 0 and pd.notna(match_date):
        if int(date_counts.get(match_date, 0)) >= int(spec["max_per_date"]):
            return False

    market_max = spec.get("market_max", {}) or {}
    if market in market_max and int(market_max[market]) > 0:
        if _v2_market_count(chosen_rows, market) >= int(market_max[market]):
            return False

    return True



def _v2_select_candidate(cands: pd.DataFrame, deterministic: bool, rng: np.random.Generator) -> Optional[pd.Series]:
    if cands is None or cands.empty:
        return None
    if deterministic:
        return cands.sort_values(["leg_priority", "model_p", "odds"], ascending=[False, False, True]).iloc[0]

    top = cands.nlargest(min(25, len(cands)), "leg_priority")
    weights = np.exp(top["leg_priority"].to_numpy(dtype=float) - float(top["leg_priority"].max()))
    weights_sum = float(weights.sum())
    if not np.isfinite(weights_sum) or weights_sum <= 0:
        return top.iloc[0]
    weights = weights / weights_sum
    idx = int(rng.choice(np.arange(len(top)), p=weights))
    return top.iloc[idx]



def build_single_v2_slip(pool: pd.DataFrame, spec: dict, deterministic: bool, rng: np.random.Generator) -> Optional[pd.DataFrame]:
    if pool is None or pool.empty:
        return None

    chosen_rows: list[pd.Series] = []
    chosen_fks: set[str] = set()
    league_counts: dict[str, int] = {}
    date_counts: dict[object, int] = {}
    target_k = int(spec.get("effective_k", spec.get("base_k", 6)))

    def eligible(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in df.iterrows():
            if _v2_can_add(row, chosen_rows, chosen_fks, league_counts, date_counts, spec):
                rows.append(row)
        if not rows:
            return pd.DataFrame(columns=df.columns)
        return pd.DataFrame(rows, columns=df.columns)

    def add(row: pd.Series) -> None:
        chosen_rows.append(row)
        fk = str(row.get("fixture_key", "")).strip()
        chosen_fks.add(fk)
        league = str(row.get("league", "UNKNOWN_LEAGUE")).strip()
        league_counts[league] = league_counts.get(league, 0) + 1
        match_date = row.get("match_date", pd.NaT)
        if pd.notna(match_date):
            date_counts[match_date] = date_counts.get(match_date, 0) + 1

    market_min = spec.get("market_min", {}) or {}
    for market, min_count in market_min.items():
        while _v2_market_count(chosen_rows, market) < int(min_count) and len(chosen_rows) < target_k:
            cands = eligible(pool[pool["market"].astype("string").fillna("").str.lower().str.strip().eq(str(market).lower().strip())])
            picked = _v2_select_candidate(cands, deterministic, rng)
            if picked is None:
                break
            add(picked)

    while len(chosen_rows) < target_k:
        cands = eligible(pool)
        picked = _v2_select_candidate(cands, deterministic, rng)
        if picked is None:
            break
        add(picked)

    if len(chosen_rows) != target_k:
        return None

    slip = pd.DataFrame(chosen_rows).reset_index(drop=True)

    if not bool(spec.get("allow_same_fixture_multi_market", False)):
        slip = slip.drop_duplicates(subset=["fixture_key"], keep="first").reset_index(drop=True)

    if len(slip) != target_k:
        return None
    return slip



def build_v2_slips(pool: pd.DataFrame, template_name: str, n_slips: Optional[int] = None, requested_k: Optional[int] = None, deterministic: bool = True, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    template_pool, spec, regime = build_v2_template_pool(pool, template_name)
    if requested_k is not None and int(requested_k) > 0:
        spec["effective_k"] = int(requested_k)
    rng = np.random.default_rng(seed)
    target_slips = int(n_slips or spec.get("base_slips", 12))

    slips: list[pd.DataFrame] = []
    seen_signatures: set[str] = set()
    working = template_pool.copy()
    attempts = 0
    max_attempts = max(1000, target_slips * 150)

    while len(slips) < target_slips and attempts < max_attempts:
        attempts += 1
        slip = build_single_v2_slip(working, spec, deterministic, rng)
        if slip is None or slip.empty:
            break

        sig = "|".join(sorted((slip["fixture_key"].astype(str) + ":" + slip["market"].astype(str)).tolist()))
        if sig in seen_signatures:
            # small decay to encourage different next picks
            working["leg_priority"] = working["leg_priority"] * 0.997
            continue

        seen_signatures.add(sig)
        slips.append(slip)

        used_idx = working.index[working["fixture_key"].isin(slip["fixture_key"].tolist())]
        working.loc[used_idx, "leg_priority"] = working.loc[used_idx, "leg_priority"] * 0.88

    slip_rows = []
    legs_rows = []
    for i, slip in enumerate(slips, start=1):
        slip_id = f"S{i:05d}"
        s = slip.copy()
        s["slip_id"] = slip_id
        s["template_name"] = template_name
        s["recommended_stake_multiplier"] = float(spec.get("recommended_stake_multiplier", 1.0))
        legs_rows.append(s)

        odds_raw = pd.to_numeric(s["odds"], errors="coerce")
        slip_rows.append(
            {
                "slip_id": slip_id,
                "template_name": template_name,
                "calendar_regime": regime,
                "k": int(len(s)),
                "recommended_stake_multiplier": float(spec.get("recommended_stake_multiplier", 1.0)),
                "avg_leg_priority": float(pd.to_numeric(s["leg_priority"], errors="coerce").mean()),
                "avg_model_p": float(pd.to_numeric(s["model_p"], errors="coerce").mean()),
                "avg_odds": float(odds_raw.mean()),
                "slip_odds": float(np.prod(odds_raw.to_numpy())) if (odds_raw.notna() & (odds_raw > 1.0001)).all() else np.nan,
                "unique_leagues": int(s["league"].nunique(dropna=True)),
                "unique_dates": int(s["match_date"].nunique(dropna=True)),
                "market_mix": "|".join(sorted(s["market"].astype(str).str.lower().tolist())),
            }
        )

    slips_df = pd.DataFrame(slip_rows)
    legs_df = pd.concat(legs_rows, ignore_index=True) if legs_rows else pd.DataFrame()
    runtime = {
        "template_name": template_name,
        "calendar_regime": regime,
        "effective_k": int(spec.get("effective_k", spec.get("base_k", 6))),
        "recommended_stake_multiplier": float(spec.get("recommended_stake_multiplier", 1.0)),
        "template_description": spec.get("description", ""),
        "template_class": str(spec.get("template_class", "unknown")),
        "failure_diagnostics": build_v2_failure_diagnostics(template_pool, spec, template_name, regime) if int(len(slips_df)) == 0 else None,
        "pool_rows_after_template": int(len(template_pool)),
        "slips_built": int(len(slips_df)),
    }
    return slips_df, legs_df, runtime

def _parse_int_csv(value: object, default: list[int]) -> list[int]:
    if value is None:
        return list(default)
    raw = str(value).strip()
    if not raw:
        return list(default)
    out: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            v = int(token)
        except Exception:
            continue
        if v > 0:
            out.append(v)
    return out or list(default)


def build_ranked_leg_board(pool: pd.DataFrame, template_name: str, top_n: int = 20) -> pd.DataFrame:
    template_pool, spec, regime = build_v2_template_pool(pool, template_name)
    ranked = template_pool.copy()
    ranked = ranked.sort_values(["leg_priority", "model_p", "odds"], ascending=[False, False, True]).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["template_name"] = template_name
    ranked["template_description"] = spec.get("description", "")
    ranked["recommended_stake_multiplier"] = float(spec.get("recommended_stake_multiplier", 1.0))
    ranked["effective_k"] = int(spec.get("effective_k", spec.get("base_k", 6)))
    ranked["calendar_regime"] = regime
    if int(top_n) > 0:
        ranked = ranked.head(int(top_n)).copy()
    return ranked


def summarize_ranked_leg_board(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "template_name",
            "calendar_regime",
            "rows",
            "unique_fixtures",
            "avg_leg_priority",
            "avg_model_p",
            "avg_odds",
            "markets",
            "tiers",
        ])

    summary = pd.DataFrame([
        {
            "template_name": str(df["template_name"].iloc[0]) if "template_name" in df.columns and len(df) else "",
            "calendar_regime": str(df["calendar_regime"].iloc[0]) if "calendar_regime" in df.columns and len(df) else "NORMAL",
            "rows": int(len(df)),
            "unique_fixtures": int(df["fixture_key"].astype("string").fillna("").nunique()),
            "avg_leg_priority": float(pd.to_numeric(df.get("leg_priority", np.nan), errors="coerce").mean()),
            "avg_model_p": float(pd.to_numeric(df.get("model_p", np.nan), errors="coerce").mean()),
            "avg_odds": float(pd.to_numeric(df.get("odds", np.nan), errors="coerce").mean()),
            "markets": "|".join(sorted(df["market"].astype("string").fillna("").str.lower().unique().tolist())),
            "tiers": "|".join(sorted(safe_upper_str_series(df.get("source_tier", pd.Series("UNKNOWN", index=df.index))).unique().tolist())),
        }
    ])
    return summary


# ----------------------------------------
# Fixture-independent board helpers
# ----------------------------------------

def build_fixture_independent_board(pool: pd.DataFrame) -> pd.DataFrame:
    """Collapse a template pool to one winning row per fixture.

    Winner logic:
    1. lowest acca_builder_priority wins (ranked beats 999)
    2. then lowest acca_builder_rank
    3. then highest leg_priority
    4. then highest model_p
    5. then lowest odds
    """
    if pool is None or pool.empty:
        return pd.DataFrame()

    df = pool.copy()
    if "fixture_key" not in df.columns:
        return pd.DataFrame()

    df["fixture_key"] = df["fixture_key"].astype("string").fillna("").str.strip()
    df = df[df["fixture_key"].ne("")].copy()
    if df.empty:
        return pd.DataFrame()

    if "acca_builder_priority" not in df.columns:
        df["acca_builder_priority"] = 999
    if "acca_builder_rank" not in df.columns:
        df["acca_builder_rank"] = 999
    if "acca_builder_bucket" not in df.columns:
        df["acca_builder_bucket"] = ""

    df["acca_builder_priority"] = pd.to_numeric(df["acca_builder_priority"], errors="coerce").fillna(999).astype(int)
    df["acca_builder_rank"] = pd.to_numeric(df["acca_builder_rank"], errors="coerce").fillna(999).astype(int)
    df["leg_priority"] = pd.to_numeric(df.get("leg_priority", np.nan), errors="coerce").fillna(-999999.0)
    df["model_p"] = pd.to_numeric(df.get("model_p", np.nan), errors="coerce").fillna(-999999.0)
    df["odds"] = pd.to_numeric(df.get("odds", np.nan), errors="coerce")
    df["__odds_sort"] = df["odds"].fillna(999999.0)
    df["acca_builder_bucket"] = df["acca_builder_bucket"].astype("string").fillna("").str.strip()

    df = df.sort_values(
        ["fixture_key", "acca_builder_priority", "acca_builder_rank", "leg_priority", "model_p", "__odds_sort"],
        ascending=[True, True, True, False, False, True],
    ).copy()

    winner = df.drop_duplicates(subset=["fixture_key"], keep="first").copy()
    winner = winner.drop(columns=["__odds_sort"], errors="ignore")
    winner["fixture_independent_rank"] = np.arange(1, len(winner) + 1)
    return winner.reset_index(drop=True)


def summarize_fixture_independent_board(
    board: pd.DataFrame,
    *,
    template_name: str,
    template_class: str,
    calendar_regime: str,
    pool_rows: int,
    pool_unique_fixtures: int,
) -> pd.DataFrame:
    """Summarize real acca capacity after one-market-per-fixture collapse."""
    if board is None:
        board = pd.DataFrame()

    if board.empty:
        return pd.DataFrame([
            {
                "template_name": template_name,
                "template_class": template_class,
                "calendar_regime": calendar_regime,
                "pool_rows": int(pool_rows),
                "pool_unique_fixtures": int(pool_unique_fixtures),
                "fixture_independent_rows": 0,
                "fixture_independent_ranked_rows": 0,
                "fixture_independent_unranked_rows": 0,
                "ranked_capacity_ge_6": 0,
                "ranked_capacity_ge_8": 0,
                "ranked_capacity_ge_10": 0,
                "ranked_capacity_ge_12": 0,
                "market_mix": "",
                "bucket_mix": "",
            }
        ])

    acca_pri = pd.to_numeric(board.get("acca_builder_priority", 999), errors="coerce").fillna(999).astype(int)
    ranked_mask = acca_pri.lt(999)

    market_mix = (
        board.get("market", pd.Series("", index=board.index))
        .astype("string")
        .fillna("")
        .str.lower()
        .str.strip()
        .value_counts(dropna=False)
        .to_dict()
    )
    bucket_mix = (
        board.get("acca_builder_bucket", pd.Series("", index=board.index))
        .astype("string")
        .fillna("NA")
        .replace("", "NA")
        .value_counts(dropna=False)
        .to_dict()
    )

    ranked_n = int(ranked_mask.sum())
    return pd.DataFrame([
        {
            "template_name": template_name,
            "template_class": template_class,
            "calendar_regime": calendar_regime,
            "pool_rows": int(pool_rows),
            "pool_unique_fixtures": int(pool_unique_fixtures),
            "fixture_independent_rows": int(len(board)),
            "fixture_independent_ranked_rows": ranked_n,
            "fixture_independent_unranked_rows": int(len(board) - ranked_n),
            "ranked_capacity_ge_6": int(ranked_n >= 6),
            "ranked_capacity_ge_8": int(ranked_n >= 8),
            "ranked_capacity_ge_10": int(ranked_n >= 10),
            "ranked_capacity_ge_12": int(ranked_n >= 12),
            "market_mix": json.dumps(market_mix, sort_keys=True),
            "bucket_mix": json.dumps(bucket_mix, sort_keys=True),
        }
    ])


def build_v2_failure_diagnostics(pool: pd.DataFrame, spec: dict, template_name: str, regime: str) -> dict:
    """Diagnostic snapshot for zero-slip template/K runs."""
    df = pool.copy() if isinstance(pool, pd.DataFrame) else pd.DataFrame()
    if df.empty:
        return {
            "template_name": template_name,
            "template_class": str(spec.get("template_class", "unknown")),
            "calendar_regime": regime,
            "pool_rows": 0,
            "unique_fixtures": 0,
            "effective_k": int(spec.get("effective_k", spec.get("base_k", 0)) or 0),
            "allow_same_fixture_multi_market": bool(spec.get("allow_same_fixture_multi_market", False)),
            "max_per_league": int(spec.get("max_per_league", 0) or 0),
            "max_per_date": int(spec.get("max_per_date", 0) or 0),
            "market_min": dict(spec.get("market_min", {}) or {}),
            "market_max": dict(spec.get("market_max", {}) or {}),
            "by_league": {},
            "by_match_date": {},
            "by_market": {},
        }

    if "fixture_key" in df.columns:
        unique_fixtures = int(df["fixture_key"].astype("string").fillna("").str.strip().nunique())
    else:
        unique_fixtures = 0

    by_league = {}
    if "league" in df.columns:
        by_league = (
            df["league"].astype("string").fillna("UNKNOWN_LEAGUE").str.strip().replace("", "UNKNOWN_LEAGUE").value_counts(dropna=False).to_dict()
        )

    by_match_date = {}
    if "match_date" in df.columns:
        s_date = pd.to_datetime(df["match_date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("NA")
        by_match_date = s_date.value_counts(dropna=False).to_dict()

    by_market = {}
    if "market" in df.columns:
        by_market = (
            df["market"].astype("string").fillna("UNKNOWN").str.lower().str.strip().replace("", "UNKNOWN").value_counts(dropna=False).to_dict()
        )

    return {
        "template_name": template_name,
        "template_class": str(spec.get("template_class", "unknown")),
        "calendar_regime": regime,
        "pool_rows": int(len(df)),
        "unique_fixtures": int(unique_fixtures),
        "effective_k": int(spec.get("effective_k", spec.get("base_k", 0)) or 0),
        "allow_same_fixture_multi_market": bool(spec.get("allow_same_fixture_multi_market", False)),
        "max_per_league": int(spec.get("max_per_league", 0) or 0),
        "max_per_date": int(spec.get("max_per_date", 0) or 0),
        "market_min": dict(spec.get("market_min", {}) or {}),
        "market_max": dict(spec.get("market_max", {}) or {}),
        "by_league": by_league,
        "by_match_date": by_match_date,
        "by_market": by_market,
    }


def render_v2_investor_markdown(
    template_name: str,
    ranked_df: pd.DataFrame,
    slips_df: pd.DataFrame,
    runtime: dict,
) -> str:
    lines: list[str] = []
    lines.append(f"# Acca Investor Summary — {template_name}")
    lines.append("")
    lines.append(f"- Calendar regime: {runtime.get('calendar_regime', 'NORMAL')}")
    lines.append(f"- Effective slip size: {runtime.get('effective_k', '')}")
    lines.append(f"- Recommended stake multiplier: {runtime.get('recommended_stake_multiplier', 1.0):.2f}")
    lines.append(f"- Pool rows after template shaping: {runtime.get('pool_rows_after_template', 0)}")
    lines.append(f"- Slips built: {runtime.get('slips_built', 0)}")
    lines.append("")
    lines.append("## Template description")
    lines.append("")
    lines.append(str(runtime.get("template_description", "")))
    lines.append("")

    if ranked_df is not None and not ranked_df.empty:
        lines.append("## Top ranked legs")
        lines.append("")
        top_cols = [
            c for c in [
                "rank",
                "league",
                "match_date",
                "home_team",
                "away_team",
                "market",
                "selection",
                "odds",
                "model_p",
                "leg_priority",
                "source_tier",
                "calendar_regime",
            ]
            if c in ranked_df.columns
        ]
        lines.append(_df_to_markdown_safe(ranked_df.loc[:, top_cols].head(20), index=False))
        lines.append("")

    if slips_df is not None and not slips_df.empty:
        lines.append("## Built slips")
        lines.append("")
        slip_cols = [
            c for c in [
                "slip_id",
                "template_name",
                "calendar_regime",
                "k",
                "recommended_stake_multiplier",
                "avg_leg_priority",
                "avg_model_p",
                "avg_odds",
                "slip_odds",
                "unique_leagues",
                "unique_dates",
                "market_mix",
            ]
            if c in slips_df.columns
        ]
        lines.append(_df_to_markdown_safe(slips_df.loc[:, slip_cols], index=False))
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def cmd_build_v2_slips(args: argparse.Namespace) -> None:
    deploy_inputs = list(args.deploy_input or [])
    if not deploy_inputs:
        raise ValueError("build_v2_slips requires at least one --deploy-input path (file or directory).")

    include_tiers = tuple(x.strip().upper() for x in str(args.include_tiers).split(",") if x.strip())
    exclude_tiers = tuple(x.strip().upper() for x in str(args.exclude_tiers).split(",") if x.strip())

    pool = build_canonical_v2_pool(
        deploy_inputs=deploy_inputs,
        include_tiers=include_tiers or DEFAULT_INCLUDE_TIERS_V2,
        exclude_tiers=exclude_tiers or DEFAULT_EXCLUDE_TIERS_V2,
        allow_observe=bool(args.allow_observe),
    )

    if args.only_markets:
        keep = {m.strip().lower() for m in str(args.only_markets).split(",") if m.strip()}
        pool = pool[pool["market"].isin(keep)].copy()

    template_names = [x.strip().upper() for x in str(args.templates).split(",") if x.strip()]
    if not template_names:
        raise ValueError("No templates provided to build_v2_slips.")

    out_dir = args.out_dir or "."
    os.makedirs(out_dir, exist_ok=True)

    k_values = _parse_int_csv(args.k_values, [8, 10, 12])
    top_n_legs = int(args.top_n_legs)
    slips_per_k = int(args.slips_per_k)

    all_summary_rows: list[pd.DataFrame] = []
    all_slips_rows: list[pd.DataFrame] = []
    all_legs_rows: list[pd.DataFrame] = []
    all_ranked_rows: list[pd.DataFrame] = []

    for template_name in template_names:
        template_pool_for_audit, template_spec_for_audit, template_regime_for_audit = build_v2_template_pool(pool, template_name)
        ranked_df = build_ranked_leg_board(pool, template_name, top_n=top_n_legs)
        ranked_summary = summarize_ranked_leg_board(ranked_df)
        fixture_board_df = build_fixture_independent_board(template_pool_for_audit)
        fixture_board_summary_df = summarize_fixture_independent_board(
            fixture_board_df,
            template_name=template_name,
            template_class=str(template_spec_for_audit.get("template_class", "unknown")),
            calendar_regime=str(template_regime_for_audit),
            pool_rows=int(len(template_pool_for_audit)),
            pool_unique_fixtures=int(template_pool_for_audit["fixture_key"].astype("string").fillna("").str.strip().nunique()) if (not template_pool_for_audit.empty and "fixture_key" in template_pool_for_audit.columns) else 0,
        )

        ranked_path = os.path.join(out_dir, f"acca_pool_{args.tag}__{template_name}.csv")
        fixture_board_path = os.path.join(out_dir, f"acca_fixture_board_{args.tag}__{template_name}.csv")
        fixture_board_summary_path = os.path.join(out_dir, f"acca_fixture_board_summary_{args.tag}__{template_name}.csv")
        ranked_df.to_csv(ranked_path, index=False)
        fixture_board_df.to_csv(fixture_board_path, index=False)
        fixture_board_summary_df.to_csv(fixture_board_summary_path, index=False)

        for k in k_values:
            slips_df, legs_df, runtime = build_v2_slips(
                pool=pool,
                template_name=template_name,
                n_slips=slips_per_k,
                requested_k=int(k),
                deterministic=not bool(args.stochastic),
                seed=int(args.seed) + int(k),
            )

            if not slips_df.empty:
                slips_df = slips_df.copy()
                slips_df["requested_k"] = int(k)
                slips_df["template_tag"] = f"{template_name}_K{k}"
                slips_df["slip_id"] = slips_df["template_tag"].astype(str) + "__" + slips_df["slip_id"].astype(str)
            if not legs_df.empty:
                legs_df = legs_df.copy()
                legs_df["requested_k"] = int(k)
                legs_df["template_tag"] = f"{template_name}_K{k}"
                legs_df["slip_id"] = legs_df["template_tag"].astype(str) + "__" + legs_df["slip_id"].astype(str)

            failure_diag_path = os.path.join(out_dir, f"acca_failure_diag_{args.tag}__{template_name}_K{k}.json")
            failure_diag_csv_path = os.path.join(out_dir, f"acca_failure_diag_{args.tag}__{template_name}_K{k}.csv")
            if runtime.get("failure_diagnostics"):
                _write_json(failure_diag_path, runtime["failure_diagnostics"])
                pd.DataFrame([{
                    "template_name": runtime["failure_diagnostics"].get("template_name", template_name),
                    "template_class": runtime["failure_diagnostics"].get("template_class", "unknown"),
                    "calendar_regime": runtime["failure_diagnostics"].get("calendar_regime", "NORMAL"),
                    "pool_rows": runtime["failure_diagnostics"].get("pool_rows", 0),
                    "unique_fixtures": runtime["failure_diagnostics"].get("unique_fixtures", 0),
                    "effective_k": runtime["failure_diagnostics"].get("effective_k", int(k)),
                    "max_per_league": runtime["failure_diagnostics"].get("max_per_league", 0),
                    "max_per_date": runtime["failure_diagnostics"].get("max_per_date", 0),
                    "by_league_json": json.dumps(runtime["failure_diagnostics"].get("by_league", {}), sort_keys=True),
                    "by_match_date_json": json.dumps(runtime["failure_diagnostics"].get("by_match_date", {}), sort_keys=True),
                    "by_market_json": json.dumps(runtime["failure_diagnostics"].get("by_market", {}), sort_keys=True),
                    "market_min_json": json.dumps(runtime["failure_diagnostics"].get("market_min", {}), sort_keys=True),
                    "market_max_json": json.dumps(runtime["failure_diagnostics"].get("market_max", {}), sort_keys=True),
                }]).to_csv(failure_diag_csv_path, index=False)

            slips_path = os.path.join(out_dir, f"acca_slips_{args.tag}__{template_name}_K{k}.csv")
            legs_path = os.path.join(out_dir, f"acca_slip_legs_{args.tag}__{template_name}_K{k}.csv")
            summary_path = os.path.join(out_dir, f"acca_summary_{args.tag}__{template_name}_K{k}.csv")
            investor_md_path = os.path.join(out_dir, f"acca_investor_summary_{args.tag}__{template_name}_K{k}.md")

            slips_df.to_csv(slips_path, index=False)
            legs_df.to_csv(legs_path, index=False)

            summary_df = pd.DataFrame([
                {
                    "tag": args.tag,
                    "template_name": template_name,
                    "template_class": runtime.get("template_class", "unknown"),
                    "requested_k": int(k),
                    "calendar_regime": runtime.get("calendar_regime", "NORMAL"),
                    "effective_k": int(runtime.get("effective_k", k)),
                    "allow_same_fixture_multi_market": bool(runtime.get("failure_diagnostics", {}).get("allow_same_fixture_multi_market", False)) if runtime.get("failure_diagnostics") else bool(template_spec_for_audit.get("allow_same_fixture_multi_market", False)),
                    "recommended_stake_multiplier": float(runtime.get("recommended_stake_multiplier", 1.0)),
                    "pool_rows_after_template": int(runtime.get("pool_rows_after_template", 0)),
                    "slips_built": int(runtime.get("slips_built", 0)),
                    "top_ranked_legs": int(len(ranked_df)),
                    "template_description": runtime.get("template_description", ""),
                    "failure_diagnostics_json": json.dumps(runtime.get("failure_diagnostics", {}), sort_keys=True) if runtime.get("failure_diagnostics") else "",
                }
            ])
            summary_df.to_csv(summary_path, index=False)

            investor_md = render_v2_investor_markdown(template_name, ranked_df, slips_df, runtime)
            with open(investor_md_path, "w", encoding="utf-8") as fh:
                fh.write(investor_md)

            print(f"[OK] wrote: {ranked_path}")
            print(f"[OK] wrote: {fixture_board_path}")
            print(f"[OK] wrote: {fixture_board_summary_path}")
            print(f"[OK] wrote: {slips_path}")
            print(f"[OK] wrote: {legs_path}")
            print(f"[OK] wrote: {summary_path}")
            print(f"[OK] wrote: {investor_md_path}")

            if runtime.get("failure_diagnostics"):
                print(f"[OK] wrote: {failure_diag_path}")
                print(f"[OK] wrote: {failure_diag_csv_path}")

            all_summary_rows.append(summary_df)
            all_ranked_rows.append(ranked_df.assign(requested_k=int(k), template_tag=f"{template_name}_K{k}"))
            if not slips_df.empty:
                all_slips_rows.append(slips_df)
            if not legs_df.empty:
                all_legs_rows.append(legs_df)


        all_summary_rows.append(ranked_summary.assign(tag=args.tag, requested_k=np.nan))

    if all_summary_rows:
        master_summary = pd.concat(all_summary_rows, ignore_index=True, sort=False)
        master_summary_path = os.path.join(out_dir, f"acca_summary_{args.tag}.csv")
        master_summary.to_csv(master_summary_path, index=False)
        print(f"[OK] wrote: {master_summary_path}")

    if all_ranked_rows:
        master_ranked = pd.concat(all_ranked_rows, ignore_index=True, sort=False)
        master_ranked_path = os.path.join(out_dir, f"acca_pool_{args.tag}.csv")
        master_ranked.to_csv(master_ranked_path, index=False)
        print(f"[OK] wrote: {master_ranked_path}")

    if all_slips_rows:
        master_slips = pd.concat(all_slips_rows, ignore_index=True, sort=False)
        master_slips_path = os.path.join(out_dir, f"acca_slips_{args.tag}.csv")
        master_slips.to_csv(master_slips_path, index=False)
        print(f"[OK] wrote: {master_slips_path}")

    if all_legs_rows:
        master_legs = pd.concat(all_legs_rows, ignore_index=True, sort=False)
        master_legs_path = os.path.join(out_dir, f"acca_slip_legs_{args.tag}.csv")
        master_legs.to_csv(master_legs_path, index=False)
        print(f"[OK] wrote: {master_legs_path}")

    run_json_path = os.path.join(out_dir, f"acca_summary_{args.tag}__RUN.json")
    _write_json(
        run_json_path,
        {
            "cmd": "build_v2_slips",
            "created_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "tag": args.tag,
            "deploy_inputs": [os.path.abspath(x) for x in deploy_inputs],
            "templates": template_names,
            "k_values": k_values,
            "top_n_legs": top_n_legs,
            "slips_per_k": slips_per_k,
            "include_tiers": list(include_tiers or DEFAULT_INCLUDE_TIERS_V2),
            "exclude_tiers": list(exclude_tiers or DEFAULT_EXCLUDE_TIERS_V2),
            "allow_observe": bool(args.allow_observe),
            "only_markets": args.only_markets,
            "out_dir": os.path.abspath(out_dir),
            "git_commit": _git_commit_hash(),
            "args": vars(args),
        },
    )
    print(f"[OK] wrote: {run_json_path}")

# -----------------------------
# Slip building
# -----------------------------

@dataclass
class BuildConfig:
    # --- Required (non-default) fields ---
    k: int
    n_slips: int
    glue_lo: float
    glue_hi: float
    anchor_min: float
    monster_min: float
    require_anchor: bool
    allow_monster: bool
    max_per_league: int
    max_per_date: int
    decay: float
    seed: int
    deterministic: bool

    # --- Optional (defaulted) fields ---

    # Max anchor legs per slip (0 = disabled). Anchor is defined as odds >= anchor_min.
    # Use 1 together with --require-anchor to force exactly one anchor per slip.
    max_anchor_legs: int = 0

    # Optional shape constraints (min/max legs per market). 0 means "no constraint".
    min_ftr: int = 0
    min_over25: int = 0
    min_under25: int = 0
    min_btts: int = 0
    min_btts_no: int = 0

    max_ftr: int = 0
    max_over25: int = 0
    max_under25: int = 0
    max_btts: int = 0
    max_btts_no: int = 0


def _eligible_by_constraints(
    cand: pd.Series,
    chosen: List[pd.Series],
    league_counts: Dict[str, int],
    date_counts: Dict[object, int],
    cfg: BuildConfig,
) -> bool:
    # unique fixture_key
    fk = cand["fixture_key"]
    for c in chosen:
        if c["fixture_key"] == fk:
            return False

    # --- Hard cap monster legs (at most 1 monster per slip) ---
    try:
        if bool(getattr(cfg, "allow_monster", False)):
            chosen_odds = pd.to_numeric(pd.Series([c.get("odds", np.nan) for c in chosen]), errors="coerce")
            cand_od = float(pd.to_numeric(cand.get("odds", np.nan), errors="coerce"))
            if np.isfinite(cand_od) and cand_od >= float(cfg.monster_min):
                already_monsters = int((chosen_odds >= float(cfg.monster_min)).sum())
                if already_monsters >= 1:
                    return False
    except Exception:
        pass

    # --- Per-market max constraints (0 means disabled) ---
    try:
        cand_mkt = str(cand.get("market", "")).strip().lower()
        chosen_mkts = [str(c.get("market", "")).strip().lower() for c in chosen]
        mkt_counts = pd.Series(chosen_mkts).value_counts().to_dict() if chosen_mkts else {}

        # If we add this candidate, would we exceed the max for that market?
        def _max_for(m: str) -> int:
            return int(getattr(cfg, f"max_{m}", 0) or 0)

        mmax = _max_for(cand_mkt)
        if mmax > 0 and int(mkt_counts.get(cand_mkt, 0)) >= mmax:
            return False
    except Exception:
        pass

    league = cand["league"]
    d = cand["match_date"]

    if cfg.max_per_league > 0 and league_counts.get(league, 0) >= cfg.max_per_league:
        return False

    if cfg.max_per_date > 0 and pd.notna(d) and date_counts.get(d, 0) >= cfg.max_per_date:
        return False

    return True

def build_single_slip(df_pool: pd.DataFrame, cfg: BuildConfig, rng: np.random.Generator) -> Optional[pd.DataFrame]:
    """
    Build one slip.

    Key fix: always pick from the *currently eligible* subset.
    The old version could stall by repeatedly selecting the same top ineligible row.
    """
    chosen_rows: List[pd.Series] = []
    chosen_fks: set[str] = set()
    league_counts: Dict[str, int] = {}
    date_counts: Dict[object, int] = {}

    if df_pool is None or df_pool.empty:
        return None

    for req in ("fixture_key", "league", "match_date", "odds", "score"):
        if req not in df_pool.columns:
            return None

    def eligible_subset(cands: pd.DataFrame) -> pd.DataFrame:
        if cands is None or cands.empty:
            return cands

        out = cands

        # exclude used fixture keys (fast path; avoid pandas .isin/.astype in tight loop)
        if chosen_fks:
            fks = out["fixture_key"].astype("string").fillna("").to_numpy()
            keep_mask = np.fromiter((fk not in chosen_fks for fk in fks), dtype=bool, count=len(fks))
            out = out.loc[keep_mask]

        # league cap (fast: avoid pandas .map(dict) inside tight loop)
        if cfg.max_per_league and cfg.max_per_league > 0:
            max_lg = int(cfg.max_per_league)
            leagues = out["league"].astype("string").fillna("").astype(str).to_numpy()
            lc = np.fromiter((int(league_counts.get(lg, 0)) for lg in leagues), dtype=np.int32, count=len(leagues))
            out = out[lc < max_lg].copy()

        # date cap (allow missing dates) (fast: avoid pandas .map(dict) inside tight loop)
        if cfg.max_per_date and cfg.max_per_date > 0:
            max_dt = int(cfg.max_per_date)
            md = out["match_date"]
            dates = md.to_numpy()
            dc = np.fromiter((int(date_counts.get(d, 0)) if pd.notna(d) else 0 for d in dates), dtype=np.int32, count=len(dates))
            out = out[md.isna() | (dc < max_dt)].copy()

        # --- Anchor / monster caps ---
        try:
            odds_s = pd.to_numeric(out.get("odds", np.nan), errors="coerce")

            # Count anchors/monsters already chosen
            chosen_odds = pd.to_numeric(pd.Series([c.get("odds", np.nan) for c in chosen_rows]), errors="coerce")
            n_anchor = int((chosen_odds >= float(cfg.anchor_min)).sum()) if len(chosen_rows) else 0
            n_monster = int((chosen_odds >= float(cfg.monster_min)).sum()) if len(chosen_rows) else 0

            # If monsters are NOT allowed, exclude them entirely from eligibility
            if not bool(getattr(cfg, "allow_monster", False)):
                out = out[~(odds_s >= float(cfg.monster_min))].copy()
            else:
                # If monsters are allowed, hard-cap to at most 1 monster per slip
                if n_monster >= 1:
                    out = out[~(odds_s >= float(cfg.monster_min))].copy()

            # If max_anchor_legs is set (>0), cap anchor legs per slip
            max_anchor = int(getattr(cfg, "max_anchor_legs", 0) or 0)
            if max_anchor > 0 and n_anchor >= max_anchor:
                out = out[~(odds_s >= float(cfg.anchor_min))].copy()
        except Exception:
            pass

        # --- Per-market max constraints (0 = disabled) ---
        try:
            chosen_mkts = [str(c.get("market", "")).strip().lower() for c in chosen_rows]
            mkt_counts = pd.Series(chosen_mkts).value_counts().to_dict() if chosen_mkts else {}

            def _mmax(m: str) -> int:
                return int(getattr(cfg, f"max_{m}", 0) or 0)

            # Build an exclude list for markets already at cap, then filter once.
            exclude_mkts: list[str] = []
            for m in ["ftr", "over25", "under25", "btts", "btts_no"]:
                mm = _mmax(m)
                if mm > 0 and int(mkt_counts.get(m, 0)) >= mm:
                    exclude_mkts.append(m)

            if exclude_mkts:
                mk = out["market"].astype("string").fillna("").str.strip().str.lower().to_numpy()
                keep_mask = np.fromiter((m not in exclude_mkts for m in mk), dtype=bool, count=len(mk))
                out = out.loc[keep_mask]
        except Exception:
            pass

        return out

    def pick_from(cands: pd.DataFrame) -> Optional[pd.Series]:
        c = eligible_subset(cands)
        if c is None or c.empty:
            return None

        # deterministic: best eligible
        if cfg.deterministic:
            return c.sort_values("score", ascending=False).iloc[0]

        # stochastic: sample from top-N eligible
        top_n = min(40, len(c))
        top = c.nlargest(top_n, "score")
        w = np.exp(top["score"].to_numpy(dtype=float) - float(top["score"].max()))
        wsum = float(w.sum())
        if not np.isfinite(wsum) or wsum <= 0:
            return top.iloc[0]
        w = w / wsum
        idx = int(rng.choice(np.arange(len(top)), p=w))
        return top.iloc[idx]

    def add(cand: pd.Series) -> None:
        chosen_rows.append(cand)
        fk = str(cand.get("fixture_key", "")).strip()
        chosen_fks.add(fk)

        lg = str(cand.get("league", "UNKNOWN")).strip()
        league_counts[lg] = league_counts.get(lg, 0) + 1

        d = cand.get("match_date", pd.NaT)
        if pd.notna(d):
            date_counts[d] = date_counts.get(d, 0) + 1

    # 1) Anchor first (if required)
    if cfg.require_anchor:
        anchors = df_pool[pd.to_numeric(df_pool["odds"], errors="coerce") >= float(cfg.anchor_min)].copy()
        picked = pick_from(anchors)
        if picked is None:
            return None
        add(picked)

    # 2) Optional monster (at most 1)
    if cfg.allow_monster:
        # Only attempt to add a monster if we don't already have one (anchor might already be a monster)
        try:
            already_monster = any(
                (pd.to_numeric(c.get("odds", np.nan), errors="coerce") >= float(cfg.monster_min))
                for c in chosen_rows
            )
        except Exception:
            already_monster = False

        if not already_monster:
            monsters = df_pool[pd.to_numeric(df_pool["odds"], errors="coerce") >= float(cfg.monster_min)].copy()
            picked = pick_from(monsters)
            if picked is not None:
                add(picked)

    # 3) Fill with glue
    glue = df_pool[
        (pd.to_numeric(df_pool["odds"], errors="coerce") >= float(cfg.glue_lo))
        & (pd.to_numeric(df_pool["odds"], errors="coerce") <= float(cfg.glue_hi))
    ].copy()

    while len(chosen_rows) < int(cfg.k):
        picked = pick_from(glue)
        if picked is None:
            break
        add(picked)

    # 3b) Try to satisfy minimum market counts before generic fallback filling.
    # This reduces the chance we build a slip that later fails the min_* constraints.
    try:
        def _need_min(m: str) -> int:
            return int(getattr(cfg, f"min_{m}", 0) or 0)

        def _have(m: str) -> int:
            return int(sum(1 for c in chosen_rows if str(c.get("market", "")).strip().lower() == m))

        # Markets we explicitly support in config
        markets_order = ["ftr", "over25", "under25", "btts", "btts_no"]

        for m in markets_order:
            need = _need_min(m)
            while need > 0 and _have(m) < need and len(chosen_rows) < int(cfg.k):
                picked = pick_from(df_pool[df_pool["market"].astype(str).str.lower().str.strip() == m])
                if picked is None:
                    break
                add(picked)
    except Exception:
        pass

    # 4) Fallback fill from best remaining (any odds)
    while len(chosen_rows) < int(cfg.k):
        picked = pick_from(df_pool)
        if picked is None:
            return None
        add(picked)

    slip = pd.DataFrame(chosen_rows).copy()

    # enforce unique fixtures
    slip = slip.drop_duplicates(subset=["fixture_key"], keep="first")

    if len(slip) != int(cfg.k):
        return None

    # Enforce max anchors per slip (anchor defined as odds >= anchor_min)
    try:
        max_anchor = int(getattr(cfg, "max_anchor_legs", 0) or 0)
        if max_anchor > 0:
            od = pd.to_numeric(slip.get("odds", np.nan), errors="coerce")
            n_anchor = int((od >= float(cfg.anchor_min)).sum())
            if n_anchor > max_anchor:
                return None
    except Exception:
        pass

    # -----------------------------
    # Optional market-shape constraints
    # -----------------------------
    try:
        mk = slip["market"].astype(str).str.lower().str.strip()
        vc = mk.value_counts().to_dict()

        def _get(name: str) -> int:
            return int(vc.get(name, 0))

        # mins
        if int(getattr(cfg, "min_ftr", 0)) > 0 and _get("ftr") < int(cfg.min_ftr):
            return None
        if int(getattr(cfg, "min_over25", 0)) > 0 and _get("over25") < int(cfg.min_over25):
            return None
        if int(getattr(cfg, "min_under25", 0)) > 0 and _get("under25") < int(cfg.min_under25):
            return None
        if int(getattr(cfg, "min_btts", 0)) > 0 and _get("btts") < int(cfg.min_btts):
            return None
        if int(getattr(cfg, "min_btts_no", 0)) > 0 and _get("btts_no") < int(cfg.min_btts_no):
            return None

        # maxes (0 means disabled)
        if int(getattr(cfg, "max_ftr", 0)) > 0 and _get("ftr") > int(cfg.max_ftr):
            return None
        if int(getattr(cfg, "max_over25", 0)) > 0 and _get("over25") > int(cfg.max_over25):
            return None
        if int(getattr(cfg, "max_under25", 0)) > 0 and _get("under25") > int(cfg.max_under25):
            return None
        if int(getattr(cfg, "max_btts", 0)) > 0 and _get("btts") > int(cfg.max_btts):
            return None
        if int(getattr(cfg, "max_btts_no", 0)) > 0 and _get("btts_no") > int(cfg.max_btts_no):
            return None
    except Exception:
        # If anything goes wrong, don't crash; just accept the slip.
        pass

    return slip


def build_slips(df_pool: pd.DataFrame, cfg: BuildConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    df_pool = df_pool.copy()

    # compute score if missing
    if "score" not in df_pool.columns:
        df_pool["score"] = compute_pick_score(df_pool)

    slips: List[pd.DataFrame] = []
    seen_signatures = set()

    # score decay across slips to encourage diversity
    working = df_pool.copy()

    attempts = 0
    max_attempts = max(5000, cfg.n_slips * 200)

    while len(slips) < cfg.n_slips and attempts < max_attempts:
        attempts += 1
        slip = build_single_slip(working, cfg, rng)
        if slip is None:
            continue

        sig = "|".join(sorted((slip["fixture_key"].astype(str) + ":" + slip["market"].astype(str)).tolist()))
        if sig in seen_signatures:
            continue

        seen_signatures.add(sig)
        slips.append(slip)

        # decay scores for legs used to diversify next slips
        used_idx = working.index[working["fixture_key"].isin(slip["fixture_key"].tolist())]
        working.loc[used_idx, "score"] = working.loc[used_idx, "score"] * cfg.decay

    if not slips:
        odds = pd.to_numeric(df_pool.get("odds"), errors="coerce")
        mk = df_pool["market"].value_counts().to_dict() if "market" in df_pool.columns else {}
        raise RuntimeError(
            "Failed to build any slips. "
            f"rows={len(df_pool)} uniq_fixtures={df_pool['fixture_key'].nunique()} "
            f"glue={(odds.between(cfg.glue_lo, cfg.glue_hi)).sum()} "
            f"anchors={(odds >= cfg.anchor_min).sum()} monsters={(odds >= cfg.monster_min).sum()} "
            f"markets={mk} k={cfg.k} max_per_league={cfg.max_per_league} max_per_date={cfg.max_per_date} "
            f"require_anchor={cfg.require_anchor} allow_monster={cfg.allow_monster}"
        )

    legs = []
    slip_rows = []
    for i, slip in enumerate(slips, start=1):
        slip_id = f"S{i:05d}"

        odds_raw = pd.to_numeric(slip["odds"], errors="coerce")
        priced_mask = odds_raw.notna() & (odds_raw > 1.0001)
        priced_legs = int(priced_mask.sum())
        unpriced_legs = int((~priced_mask).sum())

        # only compute slip_odds when *all* legs are priced
        slip_odds = float(np.prod(odds_raw.to_numpy())) if unpriced_legs == 0 else np.nan

        avg_p = float(pd.to_numeric(slip["model_p"], errors="coerce").mean(skipna=True)) if "model_p" in slip.columns else np.nan
        avg_gap = float(pd.to_numeric(slip["gap"], errors="coerce").mean(skipna=True)) if "gap" in slip.columns else np.nan
        min_score = float(pd.to_numeric(slip["score"], errors="coerce").min(skipna=True))

        anchor_odds = float(odds_raw.max()) if len(odds_raw) else np.nan

        glue_legs = int(((odds_raw >= cfg.glue_lo) & (odds_raw <= cfg.glue_hi)).sum())
        anchor_legs = int((odds_raw >= cfg.anchor_min).sum())
        monster_legs = int((odds_raw >= cfg.monster_min).sum())

        if monster_legs >= 1:
            slip_type = "monster_anchor"
        elif anchor_legs >= 1:
            slip_type = "anchor"
        else:
            slip_type = "glue_only"

        slip_score = float(slip["score"].sum()) + 0.5 * min_score

        # dates/league diversity metrics
        league_n = int(slip["league"].nunique(dropna=True))
        date_n = int(slip["match_date"].nunique(dropna=True))

        slip_rows.append(
            {
                "slip_id": slip_id,
                "k": cfg.k,
                "slip_odds": slip_odds,
                "slip_score": slip_score,
                "avg_model_p": avg_p,
                "avg_gap": avg_gap,
                "max_leg_odds": anchor_odds,
                "unique_leagues": league_n,
                "unique_dates": date_n,
                "priced_legs": priced_legs,
                "unpriced_legs": unpriced_legs,
                "slip_priced": int(unpriced_legs == 0),
                "glue_legs": glue_legs,
                "anchor_legs": anchor_legs,
                "monster_legs": monster_legs,
                "slip_type": slip_type,
            }
        )

        slegs = slip.copy()
        slegs["slip_id"] = slip_id

        # Optional: attach a human-readable correct-score shortlist note per leg
        try:
            slegs["cs_note"] = slegs.apply(format_cs_note, axis=1)
        except Exception:
            slegs["cs_note"] = ""

        legs.append(slegs)

    slips_df = pd.DataFrame(slip_rows).sort_values(["slip_score", "slip_odds"], ascending=[False, False]).reset_index(drop=True)
    legs_df = pd.concat(legs, ignore_index=True)

    # reorder legs by slip then score
    legs_df = legs_df.sort_values(["slip_id", "score"], ascending=[True, False]).reset_index(drop=True)

    return slips_df, legs_df


# -----------------------------
# Backtesting
# -----------------------------

def _is_realised_row(m: pd.Series) -> bool:
    # realised if goals exist (robust) AND not explicitly "postponed/cancelled"
    hg = m.get("home_goals", np.nan)
    ag = m.get("away_goals", np.nan)
    if pd.isna(hg) or pd.isna(ag):
        return False

    status = str(m.get("status", "")).lower()
    if re.search(r"(postpon|cancel|aband|void)", status):
        return False
    return True


def _ftr_result(hg: float, ag: float) -> str:
    if hg > ag:
        return "HOME"
    if hg < ag:
        return "AWAY"
    return "DRAW"


def _btts_result(hg: float, ag: float) -> bool:
    return (hg >= 1) and (ag >= 1)


def _over25_result(hg: float, ag: float) -> bool:
    return (hg + ag) >= 3


def _parse_team_side(selection: str) -> Optional[str]:
    s = selection.upper()
    if "HOME" in s:
        return "HOME"
    if "AWAY" in s:
        return "AWAY"
    return None


def _infer_threshold(market: str, selection: str) -> Optional[int]:
    m = market.lower()
    s = selection.upper()

    # common naming
    if "ge3" in m or "tg25" in m or ">=3" in s or "GE3" in s:
        return 3
    if "ge2" in m or "tg15" in m or ">=2" in s or "GE2" in s:
        return 2

    # patterns like HOME_TG_15 or OVER1.5
    if "1.5" in m or "15" in m or "1.5" in s:
        return 2
    if "2.5" in m or "25" in m or "2.5" in s:
        return 3

    return None


def leg_is_correct(market: str, selection: str, hg: float, ag: float) -> Optional[bool]:
    market = str(market).lower().strip()
    selection = str(selection).upper().strip()

    if pd.isna(hg) or pd.isna(ag):
        return None

    if market == "ftr":
        res = _ftr_result(hg, ag)
        # normalize selection variants
        sel = selection
        if sel in ["H", "1", "HOME_WIN", "HOMEWIN"]:
            sel = "HOME"
        elif sel in ["A", "2", "AWAY_WIN", "AWAYWIN"]:
            sel = "AWAY"
        elif sel in ["D", "X", "DRAW", "TIE"]:
            sel = "DRAW"
        return res == sel

    if market == "over25":
        return _over25_result(hg, ag) is True
    if market == "under25":
        return _over25_result(hg, ag) is False

    if market == "btts":
        # YES
        return _btts_result(hg, ag) is True
    if market == "btts_no":
        return _btts_result(hg, ag) is False

    # Team-goals style markets
    if market in ("tg15", "tg25", "home_ge2", "away_ge2", "home_ge3", "away_ge3", "home_tg15", "away_tg15", "home_tg25", "away_tg25"):
        side = _parse_team_side(selection) or ("HOME" if market.startswith("home_") else "AWAY" if market.startswith("away_") else None)
        thr = _infer_threshold(market, selection)
        if side is None or thr is None:
            return None
        g = hg if side == "HOME" else ag
        return g >= thr

    # Win To Nil style
    if market == "wtn":
        # selection expects HOME or AWAY
        sel = selection
        if sel in ["H", "HOME_WIN", "HOMEWIN"]:
            sel = "HOME"
        elif sel in ["A", "AWAY_WIN", "AWAYWIN"]:
            sel = "AWAY"
        res = _ftr_result(hg, ag)
        if sel == "HOME":
            return (res == "HOME") and (ag == 0)
        if sel == "AWAY":
            return (res == "AWAY") and (hg == 0)
        return None

    return None


# -----------------------------
# OG-style fixture key fallback (YYYY_MM_DD_HOME_AWAY)
# -----------------------------

def _norm_team_token(x: object) -> str:
    """Normalize a team token into an OG-safe key fragment.

    - keeps A–Z / a–z / 0–9
    - converts everything else to underscores
    - collapses repeated underscores
    """
    s = str(x or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _fallback_fixture_key(df: pd.DataFrame) -> pd.Series:
    """OG-style fallback: YYYY_MM_DD_<HOME>_<AWAY> (case-preserving tokens)."""
    d = pd.to_datetime(df["match_date"], errors="coerce", utc=True)
    ds = d.dt.strftime("%Y_%m_%d")
    ds = ds.fillna("").astype(str)

    h = df["home_team"].astype(str).map(_norm_team_token)
    a = df["away_team"].astype(str).map(_norm_team_token)

    out = (ds + "_" + h + "_" + a).astype(str)
    out = out.str.replace(r"_+", "_", regex=True).str.strip("_")
    return out


def load_matches(matches_root: str, multi_season: bool = True) -> pd.DataFrame:
    # load all csvs under Matches/**.csv
    pattern = os.path.join(matches_root, "**", "*.csv") if multi_season else os.path.join(matches_root, "*", "*.csv")
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"No match CSV files found under: {matches_root}")

    dfs = []
    for fp in files:
        try:
            d = pd.read_csv(fp)
        except Exception:
            continue

        # Normalize columns first (goal aliases etc.)
        d = normalise_matches_columns(d)

        # Infer league from filesystem path if missing/unknown (Matches/<League>/...)
        try:
            pth = Path(fp)
            lg = ""
            parts = list(pth.parts)
            if "Matches" in parts:
                i = parts.index("Matches")
                if i + 1 < len(parts):
                    lg = str(parts[i + 1])
            if not lg:
                lg = str(pth.parent.name)

            if "league" not in d.columns:
                d["league"] = lg
            else:
                s_lg = d["league"].astype("string").fillna("").str.strip()
                d["league"] = s_lg.mask(s_lg.eq("") | s_lg.eq("UNKNOWN_LEAGUE"), lg)
        except Exception:
            pass

        d["__src"] = fp
        dfs.append(d)

    if not dfs:
        raise RuntimeError("Matches load failed (no readable CSVs).")

    m = pd.concat(dfs, ignore_index=True)

    # Always compute an OG-style key (used for fallback joins)
    m["fixture_key_og"] = _fallback_fixture_key(m)

    # Ensure fixture_key exists; if missing/blank, fill from fixture_key_og
    if "fixture_key" not in m.columns:
        m["fixture_key"] = ""
    m["fixture_key"] = m["fixture_key"].astype("string").fillna("").str.strip()
    m.loc[m["fixture_key"].eq(""), "fixture_key"] = m.loc[m["fixture_key"].eq(""), "fixture_key_og"]

    # realized only
    m = m[m.apply(_is_realised_row, axis=1)].copy()

    # Diagnostic: if we filtered everything out, the goals columns likely weren't detected.
    if m.empty:
        try:
            # Count readable CSVs and show a hint
            print("[acca_builder] WARNING: no realised rows after filtering; check goal column aliases in MATCHES_COL_ALIASES")
        except Exception:
            pass

    # keep minimal columns needed
    keep = ["fixture_key", "fixture_key_og", "league", "match_date", "home_team", "away_team", "home_goals", "away_goals", "status", "__src"]
    keep = [c for c in keep if c in m.columns]
    # Deduplicate defensively. Prefer (league, fixture_key) to avoid cross-league collisions.
    subset = [c for c in ["league", "fixture_key"] if c in m.columns]
    if not subset:
        subset = ["fixture_key"]
    return m[keep].drop_duplicates(subset=subset, keep="first").reset_index(drop=True)


def backtest_slips(
    slips_df: pd.DataFrame,
    legs_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    stake: float = 1.0,
    system_k: int = 0,
    system_stake_mode: str = "per-slip",
    system_max_combos: int = 5000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      slip_bt (per slip)
      summary (single-row df)
    """
    legs = legs_df.copy()
    matches = matches_df.copy()

    # join (prefer league+fixture_key when available)
    if ("league" in legs.columns) and ("league" in matches.columns):
        j = legs.merge(
            matches[["league", "fixture_key", "home_goals", "away_goals"]],
            on=["league", "fixture_key"],
            how="left",
        )
        j["matched_by"] = np.where(j["home_goals"].notna() & j["away_goals"].notna(), "league_fixture_key", "none")
    else:
        j = legs.merge(matches[["fixture_key", "home_goals", "away_goals"]], on="fixture_key", how="left")
        j["matched_by"] = np.where(j["home_goals"].notna() & j["away_goals"].notna(), "fixture_key", "none")

    # fallback join by OG-style date+teams key (drift insurance)
    can_fallback = all(c in legs.columns for c in ["match_date", "home_team", "away_team"]) and all(
        c in matches.columns for c in ["match_date", "home_team", "away_team"]
    )

    if can_fallback:
        matches_fb = matches.copy()
        if "fixture_key_og" not in matches_fb.columns:
            matches_fb["fixture_key_og"] = _fallback_fixture_key(matches_fb)

        # de-duplicate to avoid many-to-one explosions
        matches_fb["fixture_key_og"] = matches_fb["fixture_key_og"].astype("string").fillna("").str.strip()
        matches_fb = matches_fb[matches_fb["fixture_key_og"].ne("")].copy()

        if "league" in matches_fb.columns:
            matches_fb["league"] = matches_fb["league"].astype("string").fillna("").str.strip()
            matches_fb = matches_fb.drop_duplicates(subset=["league", "fixture_key_og"], keep="first")
        else:
            matches_fb = matches_fb.drop_duplicates(subset=["fixture_key_og"], keep="first")

        miss = j["matched_by"].eq("none")
        if miss.any():
            j_miss = j.loc[miss].copy()
            j_miss["fixture_key_og"] = _fallback_fixture_key(j_miss)
            j_miss["fixture_key_og"] = j_miss["fixture_key_og"].astype("string").fillna("").str.strip()

            left_cols = ["slip_id", "fixture_key", "fixture_key_og"]
            if "league" in j_miss.columns and "league" in matches_fb.columns:
                left_cols = ["slip_id", "league", "fixture_key", "fixture_key_og"]

            if "league" in j_miss.columns and "league" in matches_fb.columns:
                j_fb = j_miss.loc[:, left_cols].merge(
                    matches_fb[["league", "fixture_key_og", "home_goals", "away_goals"]],
                    on=["league", "fixture_key_og"],
                    how="left",
                )
            else:
                j_fb = j_miss.loc[:, left_cols].merge(
                    matches_fb[["fixture_key_og", "home_goals", "away_goals"]],
                    on="fixture_key_og",
                    how="left",
                )

            found_fb = j_fb["home_goals"].notna() & j_fb["away_goals"].notna()
            if found_fb.any():
                key_cols = ["slip_id", "fixture_key"]
                if "league" in j.columns and "league" in j_fb.columns:
                    key_cols = ["slip_id", "league", "fixture_key"]
                j = j.merge(
                    j_fb.loc[found_fb, key_cols + ["home_goals", "away_goals"]],
                    on=key_cols,
                    how="left",
                    suffixes=("", "__ogfill"),
                )
                j["home_goals"] = j["home_goals"].fillna(j["home_goals__ogfill"])
                j["away_goals"] = j["away_goals"].fillna(j["away_goals__ogfill"])
                j = j.drop(columns=["home_goals__ogfill", "away_goals__ogfill"], errors="ignore")
                j["matched_by"] = np.where(
                    j["home_goals"].notna() & j["away_goals"].notna(),
                    np.where(j["matched_by"].eq("none"), "fixture_key_og", j["matched_by"]),
                    j["matched_by"],
                )

    # leg correctness
    j["leg_correct"] = j.apply(
        lambda r: leg_is_correct(r.get("market", ""), r.get("selection", ""), r.get("home_goals", np.nan), r.get("away_goals", np.nan)),
        axis=1
    )
    j["leg_found"] = ~j["home_goals"].isna() & ~j["away_goals"].isna()

    # -----------------------------
    # Optional system-bet settlement (k-of-n combos)
    # -----------------------------
    def _system_settle_one_slip(sub: pd.DataFrame) -> dict:
        """Return system-bet settlement stats for one slip.

        Uses priced legs only; if any leg is unpriced, system metrics are NaN.
        """
        try:
            from itertools import combinations
        except Exception:
            combinations = None

        od = pd.to_numeric(sub.get("odds", np.nan), errors="coerce")
        priced_mask = od.notna() & (od > 1.0001)
        if not bool(priced_mask.all()):
            return {
                "system_k": int(system_k),
                "system_combos": np.nan,
                "system_combos_won": np.nan,
                "system_total_stake": np.nan,
                "system_profit": np.nan,
                "system_roi": np.nan,
                "system_any_win": np.nan,
            }

        n = int(len(sub))
        k = int(system_k)
        if k <= 0 or k > n or combinations is None:
            return {
                "system_k": int(system_k),
                "system_combos": np.nan,
                "system_combos_won": np.nan,
                "system_total_stake": np.nan,
                "system_profit": np.nan,
                "system_roi": np.nan,
                "system_any_win": np.nan,
            }

        combos_idx = list(combinations(range(n), k))
        if len(combos_idx) > int(system_max_combos):
            combos_idx = combos_idx[: int(system_max_combos)]

        n_combos = int(len(combos_idx))
        if n_combos <= 0:
            return {
                "system_k": int(system_k),
                "system_combos": 0,
                "system_combos_won": 0,
                "system_total_stake": 0.0,
                "system_profit": 0.0,
                "system_roi": 0.0,
                "system_any_win": 0,
            }

        mode = str(system_stake_mode or "per-slip").strip().lower()
        if mode not in ("per-slip", "per-combo"):
            mode = "per-slip"

        if mode == "per-combo":
            stake_per_combo = float(stake)
            total_stake = stake_per_combo * float(n_combos)
        else:
            total_stake = float(stake)
            stake_per_combo = total_stake / float(n_combos)

        odds_v = pd.to_numeric(sub["odds"], errors="coerce").to_numpy(dtype=float)
        corr_v = np.array([True if v is True else False for v in sub["leg_correct"].tolist()], dtype=bool)

        combos_won = 0
        profit = 0.0
        for idxs in combos_idx:
            idxs = list(idxs)
            ok = bool(corr_v[idxs].all())
            if ok:
                combos_won += 1
                o = float(np.prod(odds_v[idxs]))
                profit += (o * stake_per_combo) - stake_per_combo
            else:
                profit -= stake_per_combo

        roi = profit / total_stake if total_stake > 0 else float("nan")
        return {
            "system_k": int(system_k),
            "system_combos": int(n_combos),
            "system_combos_won": int(combos_won),
            "system_total_stake": float(total_stake),
            "system_profit": float(profit),
            "system_roi": float(roi),
            "system_any_win": int(combos_won > 0),
        }

    # slip win = all legs correct AND all legs found
    slip_grp = j.groupby("slip_id", dropna=False)
    slip_eval = slip_grp.agg(
        legs_total=("fixture_key", "count"),
        legs_found=("leg_found", "sum"),
        legs_correct=("leg_correct", lambda x: int(np.nansum(np.array([1 if v is True else 0 for v in x])))),
        any_missing=("leg_found", lambda x: int((~x).any())),
        any_wrong=("leg_correct", lambda x: int(any(v is False for v in x))),
    ).reset_index()

    slip_eval["slip_win"] = (slip_eval["legs_found"] == slip_eval["legs_total"]) & (slip_eval["legs_correct"] == slip_eval["legs_total"])

    # slip odds
    odds = legs_df.copy()
    odds["odds"] = pd.to_numeric(odds["odds"], errors="coerce")
    odds["priced"] = odds["odds"].notna() & (odds["odds"] > 1.0001)

    slip_prices = odds.groupby("slip_id").agg(
        slip_odds=("odds", lambda x: float(np.prod(x.to_numpy())) if (x.notna() & (x > 1.0001)).all() else np.nan),
        priced_legs=("priced", "sum"),
        legs_total=("priced", "count"),
    ).reset_index()
    slip_prices["unpriced_legs"] = slip_prices["legs_total"] - slip_prices["priced_legs"]
    slip_prices["slip_priced"] = (slip_prices["unpriced_legs"] == 0).astype(int)

    slip_eval = slip_eval.merge(slip_prices[["slip_id", "slip_odds", "priced_legs", "unpriced_legs", "slip_priced"]],
                                on="slip_id", how="left")

    # profit & ROI: only defined for priced slips
    slip_eval["stake"] = float(stake)
    slip_eval["profit"] = np.where(
        slip_eval["slip_priced"].eq(1),
        np.where(slip_eval["slip_win"], (slip_eval["slip_odds"] * float(stake)) - float(stake), -float(stake)),
        np.nan,
    )
    slip_eval["roi"] = slip_eval["profit"] / float(stake)

    # System-bet settlement overrides profit/roi when enabled
    if int(system_k) > 0:
        sys_rows = []
        for slip_id, sub in j.groupby("slip_id", dropna=False):
            sys_rows.append({"slip_id": slip_id, **_system_settle_one_slip(sub)})
        sys_df = pd.DataFrame(sys_rows)
        slip_eval = slip_eval.merge(sys_df, on="slip_id", how="left")
        slip_eval["profit"] = slip_eval["system_profit"].where(slip_eval["system_profit"].notna(), slip_eval["profit"])
        slip_eval["roi"] = slip_eval["system_roi"].where(slip_eval["system_roi"].notna(), slip_eval["roi"])

    # order by slip_score if available
    if "slip_score" in slips_df.columns:
        slip_eval = slip_eval.merge(slips_df[["slip_id", "slip_score"]], on="slip_id", how="left")
        slip_eval = slip_eval.sort_values("slip_score", ascending=False)
    else:
        slip_eval = slip_eval.sort_values("slip_odds", ascending=False)

    # drawdown / losing streak metrics over the ordered list
    profits = slip_eval["profit"].to_numpy()
    cum = np.cumsum(profits)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_drawdown = float(dd.min()) if len(dd) else 0.0

    # longest losing streak
    lose = (~slip_eval["slip_win"]).to_numpy()
    longest = 0
    cur = 0
    for v in lose:
        if v:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0

    priced_mask = slip_eval["slip_priced"].eq(1)
    priced_eval = slip_eval.loc[priced_mask].copy()

    if int(system_k) > 0 and ("system_any_win" in priced_eval.columns):
        hit_rate = float(pd.to_numeric(priced_eval["system_any_win"], errors="coerce").fillna(0).astype(int).mean()) if len(priced_eval) else 0.0
    else:
        hit_rate = float(priced_eval["slip_win"].mean()) if len(priced_eval) else 0.0
    mean_roi = float(priced_eval["roi"].mean()) if len(priced_eval) else 0.0
    median_roi = float(priced_eval["roi"].median()) if len(priced_eval) else 0.0
    roi_std = float(priced_eval["roi"].std(ddof=0)) if len(priced_eval) else 0.0
    pct_positive_roi = float((priced_eval["roi"] > 0).mean()) if len(priced_eval) else 0.0
    attempts_per_win = float(1.0 / hit_rate) if hit_rate > 0 else float("inf")

    # join health at leg level
    leg_total = len(j)
    pct_fk = float((j["matched_by"] == "fixture_key").mean()) if leg_total else 0.0
    pct_fb = float((j["matched_by"] == "fixture_key_og").mean()) if leg_total else 0.0
    pct_nf = float((j["matched_by"] == "none").mean()) if leg_total else 0.0

    priced_slips = int(priced_mask.sum())
    unpriced_slips = int((~priced_mask).sum())

    summary = pd.DataFrame(
        [{
            "n_slips": int(len(slip_eval)),
            "priced_slips": priced_slips,
            "unpriced_slips": unpriced_slips,
            "hit_rate": hit_rate,
            "mean_roi": mean_roi,
            "median_roi": median_roi,
            "roi_std": roi_std,
            "pct_positive_roi": pct_positive_roi,
            "attempts_per_win": attempts_per_win,
            "total_profit": float(priced_eval["profit"].sum()) if len(priced_eval) else 0.0,
            "max_drawdown": max_drawdown,
            "longest_losing_streak": int(longest),
            "missing_legs_slips": int((slip_eval["any_missing"] > 0).sum()),
            "pct_matched_fixture_key": pct_fk,
            "pct_matched_fallback": pct_fb,
            "pct_not_found": pct_nf,
            "system_k": int(system_k),
            "system_stake_mode": str(system_stake_mode),
            "system_mean_combos": float(pd.to_numeric(priced_eval.get("system_combos", np.nan), errors="coerce").mean(skipna=True)) if (int(system_k) > 0 and len(priced_eval)) else 0.0,
            "system_mean_total_stake": float(pd.to_numeric(priced_eval.get("system_total_stake", np.nan), errors="coerce").mean(skipna=True)) if (int(system_k) > 0 and len(priced_eval)) else float(stake),
        }]
    )

    return slip_eval.reset_index(drop=True), summary


# -----------------------------
# CLI
# -----------------------------

def cmd_build(args: argparse.Namespace) -> None:
    # Legacy builder path retained for backward compatibility only.
    # Live / production shortlist construction should now use `build_v2`,
    # which is tier-aware, deploy-aware, template-driven, and calendar-sensitive.
    pool = pd.read_csv(args.pool)
    pool = normalise_pool_columns(pool)

    # Drop blank fixture keys (cannot build valid slips)
    pool["fixture_key"] = pool["fixture_key"].astype(str).str.strip()
    pool = pool[pool["fixture_key"].ne("")].copy()

    pool["odds"] = pd.to_numeric(pool["odds"], errors="coerce")

    # optionally exclude certain odds provenance
    if args.exclude_od_source:
        pat = re.compile(args.exclude_od_source, flags=re.IGNORECASE)
        pool = pool[~pool["od_source"].astype(str).apply(lambda x: bool(pat.search(x)))].copy()

    # default: exclude unpriced legs (prevents mixing priced/unpriced in ROI tests)
    if not args.allow_unpriced:
        if "odds" in pool.columns:
            pool["odds"] = pd.to_numeric(pool["odds"], errors="coerce")
            pool = pool[pool["odds"].notna() & (pool["odds"] > 1.0001)].copy()

    # optional filter: only allow specified markets
    if args.only_markets:
        keep = set([m.strip().lower() for m in args.only_markets.split(",") if m.strip()])
        pool = pool[pool["market"].isin(keep)].copy()

    pool["score"] = compute_pick_score(pool)

    cfg = BuildConfig(
        k=int(args.k),
        n_slips=int(args.slips),
        glue_lo=float(args.glue_lo),
        glue_hi=float(args.glue_hi),
        anchor_min=float(args.anchor_min),
        max_anchor_legs=int(args.max_anchor_legs),
        monster_min=float(args.monster_min),
        require_anchor=bool(args.require_anchor),
        allow_monster=bool(args.allow_monster),
        max_per_league=int(args.max_per_league),
        max_per_date=int(args.max_per_date),
        decay=float(args.decay),
        seed=int(args.seed),
        deterministic=not bool(args.stochastic),
        min_ftr=int(args.min_ftr),
        min_over25=int(args.min_over25),
        min_under25=int(args.min_under25),
        min_btts=int(args.min_btts),
        min_btts_no=int(args.min_btts_no),
        max_ftr=int(args.max_ftr),
        max_over25=int(args.max_over25),
        max_under25=int(args.max_under25),
        max_btts=int(args.max_btts),
        max_btts_no=int(args.max_btts_no),
    )

    slips_df, legs_df = build_slips(pool, cfg)

    out_dir = args.out_dir or "."
    os.makedirs(out_dir, exist_ok=True)

    slips_path = os.path.join(out_dir, f"slips_{args.tag}.csv")
    legs_path = os.path.join(out_dir, f"slips_legs_{args.tag}.csv")

    slips_df.to_csv(slips_path, index=False)
    legs_df.to_csv(legs_path, index=False)

    print(f"[OK] wrote: {slips_path}")
    print(f"[OK] wrote: {legs_path}")
    print(f"[INFO] built slips: {len(slips_df)}  (attempted {args.slips})")

    run_json_path = os.path.join(out_dir, f"slips_{args.tag}__RUN.json")
    _write_json(
        run_json_path,
        {
            "cmd": "build",
            "created_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "tag": args.tag,
            "pool_path": os.path.abspath(args.pool),
            "out_dir": os.path.abspath(out_dir),
            "git_commit": _git_commit_hash(),
            "args": vars(args),
            "pool_rows": int(len(pool)),
            "slips_rows": int(len(slips_df)),
            "legs_rows": int(len(legs_df)),
        },
    )
    print(f"[OK] wrote: {run_json_path}")


def cmd_backtest(args: argparse.Namespace) -> None:
    slips_df = pd.read_csv(args.slips_csv)
    legs_df = pd.read_csv(args.legs_csv)
    legs_df = normalise_pool_columns(legs_df)  # legs file should already be normalized shape

    matches_df = load_matches(args.matches_root, multi_season=bool(args.multi_season))

    slip_bt, summary = backtest_slips(
        slips_df=slips_df,
        legs_df=legs_df,
        matches_df=matches_df,
        stake=float(args.stake),
        system_k=int(getattr(args, "system_k", 0) or 0),
        system_stake_mode=str(getattr(args, "system_stake_mode", "per-slip") or "per-slip"),
        system_max_combos=int(getattr(args, "system_max_combos", 5000) or 5000),
    )

    out_dir = args.out_dir or "."
    os.makedirs(out_dir, exist_ok=True)

    bt_path = os.path.join(out_dir, f"slip_backtest_{args.tag}.csv")
    sum_path = os.path.join(out_dir, f"slip_backtest_summary_{args.tag}.csv")

    slip_bt.to_csv(bt_path, index=False)
    summary.to_csv(sum_path, index=False)

    print(f"[OK] wrote: {bt_path}")
    print(f"[OK] wrote: {sum_path}")
    print(summary.to_string(index=False))

    if "slip_type" in slips_df.columns and "slip_id" in slips_df.columns:
        merged = slip_bt.merge(slips_df[["slip_id", "slip_type"]], on="slip_id", how="left")
        # compute only on priced slips
        priced = merged[merged["slip_priced"].eq(1)].copy()
        if not priced.empty:
            type_summary = priced.groupby("slip_type").agg(
                n_slips=("slip_id", "count"),
                hit_rate=("slip_win", "mean"),
                mean_roi=("roi", "mean"),
                median_roi=("roi", "median"),
                roi_std=("roi", lambda x: float(x.std(ddof=0))),
                pct_positive_roi=("roi", lambda x: float((x > 0).mean())),
            ).reset_index()
            type_path = os.path.join(out_dir, f"slip_backtest_type_summary_{args.tag}.csv")
            type_summary.to_csv(type_path, index=False)
            print(f"[OK] wrote: {type_path}")

    run_json_path = os.path.join(out_dir, f"slip_backtest_{args.tag}__RUN.json")
    _write_json(
        run_json_path,
        {
            "cmd": "backtest",
            "created_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "tag": args.tag,
            "slips_csv": os.path.abspath(args.slips_csv),
            "legs_csv": os.path.abspath(args.legs_csv),
            "matches_root": os.path.abspath(args.matches_root),
            "out_dir": os.path.abspath(out_dir),
            "git_commit": _git_commit_hash(),
            "args": vars(args),
            "matches_rows": int(len(matches_df)),
            "slip_backtest_rows": int(len(slip_bt)),
        },
    )
    print(f"[OK] wrote: {run_json_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Acca Builder + Slip Backtester (OG/BAWA)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # build_v2_slips
    v2s = sub.add_parser("build_v2_slips", help="Build ranked acca pools and slips directly from deploy_rulebook tier outputs")
    v2s.add_argument("--deploy-input", action="append", required=True, help="Deploy tier CSV or directory containing deploy tier CSVs. Repeatable.")
    v2s.add_argument("--tag", required=True, help="Tag used in output filenames")
    v2s.add_argument("--out-dir", default=None, help="Output directory")
    v2s.add_argument("--templates", default="COMBINED_ALLMARKETS,FTR_ONLY,BTTS_ONLY,OU25_ONLY", help="Comma-separated v2 template names")
    v2s.add_argument("--k-values", default="8,10,12", help="Comma-separated acca sizes to build, e.g. 8,10,12 or 12,14,16")
    v2s.add_argument("--slips-per-k", type=int, default=6, help="Number of slips to build per template / k combination")
    v2s.add_argument("--top-n-legs", type=int, default=20, help="Export the top N ranked legs per template")
    v2s.add_argument("--include-tiers", default=",".join(DEFAULT_INCLUDE_TIERS_V2), help="Comma-separated source tiers to include")
    v2s.add_argument("--exclude-tiers", default=",".join(DEFAULT_EXCLUDE_TIERS_V2), help="Comma-separated source tiers to exclude")
    v2s.add_argument("--allow-observe", action="store_true", help="Allow OBSERVE rows into the v2 pool")
    v2s.add_argument("--only-markets", default=None, help="Optional comma-separated market filter applied before template shaping")
    v2s.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility")
    v2s.add_argument("--stochastic", action="store_true", help="Enable stochastic template slip construction")
    v2s.set_defaults(func=cmd_build_v2_slips)

    # build
    b = sub.add_parser("build", help="Build slips from pool CSV")
    b.add_argument("--pool", required=True, help="Input pool CSV (deploy output or merged picks). Must contain fixture_key.")
    b.add_argument("--tag", required=True, help="Tag used in output filenames")
    b.add_argument("--out-dir", default=None, help="Output directory")
    b.add_argument("--k", type=int, default=7, help="Slip size (legs) e.g. 6/7/8/9")
    b.add_argument("--slips", type=int, default=50, help="Number of slips to generate")
    b.add_argument("--only-markets", default=None, help="Comma list filter e.g. ftr,over25,btts,btts_no")
    b.add_argument("--glue-lo", type=float, default=1.30, help="Glue leg odds minimum")
    b.add_argument("--glue-hi", type=float, default=1.90, help="Glue leg odds maximum")
    b.add_argument("--anchor-min", type=float, default=2.50, help="Anchor odds minimum")
    b.add_argument("--max-anchor-legs", type=int, default=0, help="Max anchor legs per slip (odds>=anchor-min). 0=disabled. Use 1 with --require-anchor to force exactly one anchor per slip.")
    b.add_argument("--monster-min", type=float, default=3.50, help="Monster odds minimum")
    b.add_argument("--require-anchor", action="store_true", help="Require exactly one anchor leg (>= anchor-min)")
    b.add_argument("--allow-monster", action="store_true", help="Allow at most one monster leg (>= monster-min)")
    b.add_argument("--max-per-league", type=int, default=2, help="Max legs from same league per slip (0=disabled)")
    b.add_argument("--max-per-date", type=int, default=4, help="Max legs from same match_date per slip (0=disabled)")
    b.add_argument("--decay", type=float, default=0.85, help="Score decay factor for used fixtures across slips")
    b.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility")
    b.add_argument("--stochastic", action="store_true", help="Enable stochastic sampling (default is deterministic)")
    b.add_argument("--allow-unpriced", action="store_true", help="Allow legs with missing odds (accuracy-only; ROI becomes NaN)")
    b.add_argument("--exclude-od-source", default=None, help="Regex to exclude legs by od_source (e.g. 'fallback|lambda')")

    # Optional market-shape constraints
    b.add_argument("--min-ftr", type=int, default=0, help="Minimum number of FTR legs per slip (0=disabled)")
    b.add_argument("--min-over25", type=int, default=0, help="Minimum number of OVER25 legs per slip (0=disabled)")
    b.add_argument("--min-under25", type=int, default=0, help="Minimum number of UNDER25 legs per slip (0=disabled)")
    b.add_argument("--min-btts", type=int, default=0, help="Minimum number of BTTS YES legs per slip (0=disabled)")
    b.add_argument("--min-btts-no", type=int, default=0, help="Minimum number of BTTS NO legs per slip (0=disabled)")

    b.add_argument("--max-ftr", type=int, default=0, help="Maximum number of FTR legs per slip (0=disabled)")
    b.add_argument("--max-over25", type=int, default=0, help="Maximum number of OVER25 legs per slip (0=disabled)")
    b.add_argument("--max-under25", type=int, default=0, help="Maximum number of UNDER25 legs per slip (0=disabled)")
    b.add_argument("--max-btts", type=int, default=0, help="Maximum number of BTTS YES legs per slip (0=disabled)")
    b.add_argument("--max-btts-no", type=int, default=0, help="Maximum number of BTTS NO legs per slip (0=disabled)")
    b.set_defaults(func=cmd_build)

    # build_v2
    b2 = sub.add_parser("build_v2", help="Build a canonical deploy-driven acca pool from deploy_rulebook tier outputs")
    b2.add_argument("--deploy-input", action="append", required=True, help="Deploy tier CSV or directory containing deploy tier CSVs. Repeatable.")
    b2.add_argument("--tag", required=True, help="Tag used in output filenames")
    b2.add_argument("--out-dir", default=None, help="Output directory")
    b2.add_argument("--include-tiers", default="ELITE,STANDARD", help="Comma list of tiers to include. Default: ELITE,STANDARD")
    b2.add_argument("--exclude-tiers", default="OBSERVE", help="Comma list of tiers to exclude. Default: OBSERVE")
    b2.add_argument("--allow-observe", action="store_true", help="Allow OBSERVE rows into the canonical v2 pool.")
    b2.add_argument("--only-markets", default=None, help="Comma list filter e.g. ftr,over25,btts,btts_no")
    b2.add_argument("--template", default="HYBRID_BALANCED", choices=sorted(V2_TEMPLATE_SPECS.keys()), help="Template-driven shortlist type to build from the canonical pool.")
    b2.add_argument("--slips", type=int, default=0, help="Optional override for number of slips to build. 0 uses template default.")
    b2.add_argument("--seed", type=int, default=7, help="Random seed for v2 shortlist generation.")
    b2.add_argument("--stochastic", action="store_true", help="Enable stochastic sampling for v2 slip construction.")
    b2.set_defaults(func=cmd_build_v2)

    # backtest
    t = sub.add_parser("backtest", help="Backtest slips using Matches root")
    t.add_argument("--slips-csv", required=True, help="slips_<tag>.csv")
    t.add_argument("--legs-csv", required=True, help="slips_legs_<tag>.csv")
    t.add_argument("--matches-root", required=True, help="Matches/ root")
    t.add_argument("--multi-season", action="store_true", help="Search Matches/**.csv recursively")
    t.add_argument("--stake", type=float, default=1.0, help="Flat stake per slip")
    t.add_argument(
        "--system-k",
        type=int,
        default=0,
        help="System-bet mode: evaluate k-of-n combos per slip (e.g. 7 for an 8-leg slip). 0=disabled.",
    )
    t.add_argument(
        "--system-stake-mode",
        choices=["per-slip", "per-combo"],
        default="per-slip",
        help="System-bet staking: per-slip splits --stake across combos; per-combo uses --stake for each combo line.",
    )
    t.add_argument(
        "--system-max-combos",
        type=int,
        default=5000,
        help="Safety cap: maximum combos per slip to evaluate in system mode (default: 5000).",
    )
    t.add_argument("--out-dir", default=None, help="Output directory")
    t.add_argument("--tag", required=True, help="Tag used in output filenames")
    t.set_defaults(func=cmd_backtest)

    return p


# -----------------------------
# Helper functions (module-level)
# -----------------------------

def _git_commit_hash() -> Optional[str]:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        if r.returncode != 0:
            return None
        return r.stdout.strip() or None
    except Exception:
        return None


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
 
def _value_counts_dict(series: pd.Series) -> dict[str, int]:
    if series is None:
        return {}
    s = series.astype("string").fillna("NA").replace("", "NA")
    vc = s.value_counts(dropna=False)
    return {str(k): int(v) for k, v in vc.items()}


def build_v2_failure_diagnostics(template_pool: pd.DataFrame, spec: dict, template_name: str, regime: str) -> dict:
    pool = template_pool.copy() if template_pool is not None else pd.DataFrame()
    if pool.empty:
        return {
            "template_name": str(template_name),
            "template_class": str(spec.get("template_class", "unknown")),
            "calendar_regime": str(regime),
            "pool_rows": 0,
            "unique_fixtures": 0,
            "by_league": {},
            "by_match_date": {},
            "by_market": {},
            "effective_k": int(spec.get("effective_k", spec.get("base_k", 0)) or 0),
            "max_per_league": int(spec.get("max_per_league", 0) or 0),
            "max_per_date": int(spec.get("max_per_date", 0) or 0),
            "market_min": dict(spec.get("market_min", {}) or {}),
            "market_max": dict(spec.get("market_max", {}) or {}),
        }

    out = {
        "template_name": str(template_name),
        "template_class": str(spec.get("template_class", "unknown")),
        "calendar_regime": str(regime),
        "pool_rows": int(len(pool)),
        "unique_fixtures": int(pool["fixture_key"].astype("string").fillna("").str.strip().nunique()),
        "by_league": _value_counts_dict(pool.get("league", pd.Series("UNKNOWN_LEAGUE", index=pool.index))),
        "by_match_date": _value_counts_dict(pool.get("match_date", pd.Series("", index=pool.index))),
        "by_market": _value_counts_dict(pool.get("market", pd.Series("", index=pool.index))),
        "effective_k": int(spec.get("effective_k", spec.get("base_k", 0)) or 0),
        "max_per_league": int(spec.get("max_per_league", 0) or 0),
        "max_per_date": int(spec.get("max_per_date", 0) or 0),
        "market_min": dict(spec.get("market_min", {}) or {}),
        "market_max": dict(spec.get("market_max", {}) or {}),
    }
    return out


def build_fixture_independent_board(pool: pd.DataFrame) -> pd.DataFrame:
    if pool is None or pool.empty:
        return pd.DataFrame(
            columns=[
                "fixture_key",
                "market",
                "selection",
                "league",
                "match_date",
                "home_team",
                "away_team",
                "source_tier",
                "acca_builder_bucket",
                "acca_builder_priority",
                "acca_builder_rank",
                "leg_priority",
                "model_p",
                "odds",
            ]
        )

    df = pool.copy()
    df["fixture_key"] = df["fixture_key"].astype("string").fillna("").str.strip()
    df = df[df["fixture_key"].ne("")].copy()

    df["__sort_acca_priority"] = pd.to_numeric(df.get("acca_builder_priority", 999999), errors="coerce").fillna(999999).astype(int)
    df["__sort_acca_rank"] = pd.to_numeric(df.get("acca_builder_rank", 999999), errors="coerce").fillna(999999).astype(int)
    df["__sort_leg_priority"] = pd.to_numeric(df.get("leg_priority", np.nan), errors="coerce").fillna(-999999.0)
    df["__sort_model_p"] = pd.to_numeric(df.get("model_p", np.nan), errors="coerce").fillna(-999999.0)
    df["__sort_odds"] = pd.to_numeric(df.get("odds", np.nan), errors="coerce").fillna(999999.0)

    df = df.sort_values(
        ["__sort_acca_priority", "__sort_acca_rank", "__sort_leg_priority", "__sort_model_p", "__sort_odds"],
        ascending=[True, True, False, False, True],
    ).copy()

    df = df.drop_duplicates(subset=["fixture_key"], keep="first").copy()
    df = df.drop(columns=["__sort_acca_priority", "__sort_acca_rank", "__sort_leg_priority", "__sort_model_p", "__sort_odds"], errors="ignore")
    return df.reset_index(drop=True)


def summarize_fixture_independent_board(board: pd.DataFrame, *, template_name: str, template_class: str, calendar_regime: str, pool_rows: int, pool_unique_fixtures: int) -> pd.DataFrame:
    if board is None:
        board = pd.DataFrame()

    return pd.DataFrame([
        {
            "template_name": str(template_name),
            "template_class": str(template_class),
            "calendar_regime": str(calendar_regime),
            "pool_rows": int(pool_rows),
            "pool_unique_fixtures": int(pool_unique_fixtures),
            "fixture_board_rows": int(len(board)),
            "fixture_board_unique_fixtures": int(board["fixture_key"].astype("string").fillna("").str.strip().nunique()) if (not board.empty and "fixture_key" in board.columns) else 0,
            "winning_markets": json.dumps(_value_counts_dict(board.get("market", pd.Series("", index=board.index))), sort_keys=True),
            "winning_buckets": json.dumps(_value_counts_dict(board.get("acca_builder_bucket", pd.Series("", index=board.index))), sort_keys=True),
            "winning_priorities": json.dumps(_value_counts_dict(board.get("acca_builder_priority", pd.Series("", index=board.index))), sort_keys=True),
        }
    ])