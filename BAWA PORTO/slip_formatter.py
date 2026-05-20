"""
slip_formatter.py
─────────────────────────────────────────────────────────────────────────────
Thin formatter over deploy_rulebook output.

NO filler.
NO rescue.
NO forced completion.
NO dependence on acca_builder buckets.

Architecture:
    deploy_rulebook output CSV(s) → slip_formatter.py → ranked board + slips

Default philosophy:
    - use deploy_rulebook routed outputs directly
    - treat ELITE / STANDARD rows as already-filtered valid picks
    - do not require acca_builder columns
    - do not require profit_first_keep unless explicitly requested
    - split by market when needed
    - one leg per fixture
    - rank top-down from real routed quality

Usage examples:

    python3 slip_formatter.py \
        --input /path/to/BOOKIE_IMP20_ALLMARKETS_2026-03-13_to_2026-03-17__DEPLOY_TIER_ELITE__PRESET_V1__FTR_accuracy.csv \
        --outdir ./slips

    python3 slip_formatter.py \
        --inputs \
          /path/to/ELITE.csv \
          /path/to/STANDARD.csv \
        --outdir ./slips \
        --block-regime-flags \
        --max-per-family 6 \
        --monster-only-accas

    python3 slip_formatter.py \
        --inputs /path/to/ELITE.csv /path/to/STANDARD.csv \
        --outdir ./slips \
        --max-per-league 4
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


# ── COLUMN CANDIDATES ────────────────────────────────────────────────────────

COL_FIXTURE_KEY = ["fixture_key"]
COL_LEAGUE = ["league"]
COL_HOME = ["home_team_name", "home_team"]
COL_AWAY = ["away_team_name", "away_team"]
COL_MATCH_DATE = ["match_date"]
COL_MARKET = ["market"]
COL_SELECTION = ["selection", "bookie_pick"]
COL_BOOKIE_PICK = ["bookie_pick", "selection"]
COL_MODEL_TOP_PICK = ["model_top_pick"]
COL_MODEL_P = ["model_p_for_bookie", "model_p"]
COL_ODDS = ["bookie_od", "odds"]
COL_BOOKIE_IMPLIED = ["bookie_implied", "bookie_implied_used"]

COL_DEPLOY_TIER = ["deploy_tier", "tier"]
COL_POOL_TIER = ["pool_tier"]

COL_KEEP = ["profit_first_keep"]
COL_REVIEW_FLAG = ["profit_first_review_flag"]
COL_REVIEW_REASON = ["profit_first_review_reason"]
COL_STD_BUCKET = ["standard_reporting_bucket"]
COL_CONTEXT = ["context_reason_codes"]

COL_PRODUCT = ["product"]
COL_PRODUCT_PROFILE = ["product_profile"]
COL_PRODUCT_LANE = ["product_lane"]
COL_VALUE_EDGE = ["value_edge"]
COL_VALUE_EDGE_BPS = ["value_edge_bps"]
COL_VALUE_GAP = ["value_gap_pct_points"]
COL_VALUE_EDGE_TIER = ["value_edge_tier"]
COL_META_TIER = ["meta_tier"]
COL_META_LABEL = ["meta_overlay_label"]
COL_TEAM_INTEL_ACTION = ["team_intel_overlay_action"]
COL_TEAM_INTEL_UPLIFT = ["team_intel_overlay_rank_uplift_bps"]
COL_TEAM_INTEL_PENALTY = ["team_intel_overlay_rank_penalty_bps"]
COL_TEAM_INTEL_CAUTION = ["team_intel_overlay_slip_caution_flag"]
COL_TEAM_INTEL_AVOID_IN_ACCA = ["team_intel_overlay_avoid_in_acca_flag"]
COL_TEAM_INTEL_PREMIUM_TAG = ["team_intel_overlay_premium_tag_flag"]
COL_TEAM_INTEL_REASON = ["team_intel_overlay_reason"]
COL_TEAM_INTEL_MARKET_FIT = ["team_intel_overlay_market_fit_score"]
COL_TEAM_CONTEXT_LABEL = ["team_context_label"]
COL_TEAM_CONTEXT_INTERACTION = ["team_fixture_interaction"]
COL_TEAM_CONTEXT_FAMILY = ["team_context_filter_family"]
COL_P_HOME_GE2 = ["p_home_ge2", "home_ge2_confidence"]
COL_P_AWAY_GE2 = ["p_away_ge2", "away_ge2_confidence"]
COL_PHASE9B_GE2_SHADOW = ["phase9b_premium_ge2_test_lane"]
COL_CS_TOP1_SCORELINE = ["cs_top1_scoreline", "cs1"]
COL_CS_TOP1_MASS = ["cs_top1_mass", "cs1_p"]
COL_CS_TOP2_SCORELINE = ["cs_top2_scoreline", "cs2"]
COL_CS_TOP2_MASS = ["cs_top2_mass", "cs2_p"]
COL_CS_TOP3_SCORELINE = ["cs_top3_scoreline", "cs3"]
COL_CS_TOP3_MASS = ["cs_top3_mass", "cs3_p"]
COL_CS_TOP3_TOTAL_MASS = ["cs_top3_total_mass"]
COL_CS_ENTROPY = ["cs_entropy"]
COL_CS_CONCENTRATION_BUCKET = ["cs_concentration_bucket"]
COL_CS_BANKER_FLAG = ["cs_banker_flag"]
COL_CS_CLUSTER_FAMILY = ["cs_cluster_family"]
COL_CS_ALIGNS_SELECTION = ["cs_aligns_selection_flag"]
COL_CS_ALIGNS_TOP1 = ["cs_aligns_top1_selection_flag"]
COL_CS_ALIGNS_MASS = ["cs_aligns_mass_selection_flag"]
COL_CS_ALIGNMENT_COUNT = ["cs_alignment_count"]
COL_CS_FTR_PICK_SIDE_MASS = ["cs_ftr_pick_side_mass"]
COL_CS_FTR_PICK_SIDE_MARGIN = ["cs_ftr_pick_side_margin"]
COL_CS_FTR_ALIGNMENT = ["cs_ftr_alignment"]
COL_CS_BTTS_MARGIN = ["cs_btts_margin"]
COL_CS_BTTS_ALIGNMENT = ["cs_btts_alignment"]
COL_CS_OU25_MARGIN = ["cs_ou25_margin"]
COL_CS_OU25_ALIGNMENT = ["cs_ou25_alignment"]
COL_CS_DIFFUSE_FLAG = ["cs_diffuse_flag"]
COL_CS_ONE_SIDED_FLAG = ["cs_one_sided_flag"]
COL_CS_DRAW_FAMILY_FLAG = ["cs_draw_family_flag"]
COL_CS_FRAGILITY_FLAG = ["cs_fragility_flag"]
COL_CS_SUPPORTED_FLAG = ["cs_supported_flag"]
COL_CS_FRAGILITY_PENALTY = ["cs_fragility_penalty"]
COL_CS_BANKER_BONUS = ["cs_banker_bonus"]
COL_LARGE_ACCA_CS_SAFE = ["large_acca_cs_safe"]

COL_FTR_PRIORITY = ["ftr_priority", "ftr_priority_tier"]

COL_DET_WARN = ["deterministic_warn_reason"]
COL_DET_VETO = ["deterministic_veto_reason"]
COL_DEPLOY_VETO = ["deploy_veto_reason"]

COL_LEARNED_VETO = ["learned_veto_reason"]
COL_SIGNAL_BTTS = ["signal_btts"]
COL_FTS_MAX = ["fts_max"]
COL_BTTS_YES_GE2_MIN = ["btts_yes_ge2_min"]
COL_LAMBDA_HOME = ["lambda_home"]
COL_LAMBDA_AWAY = ["lambda_away"]
COL_P_HW_HGE2 = ["p_hw_and_hge2"]
COL_P_AW_AGE2 = ["p_aw_and_age2"]
COL_GE2_GAP = ["ge2_gap"]
COL_AVG_BTTS_RATE = ["avg_btts_rate"]
COL_CS_OVER25_MASS = ["cs_over25_mass"]
COL_TOP3_OVER_COUNT = ["top3_over_count"]
COL_DRAW_WARNING_TOKEN = ["draw_warning_token"]


# ── FAMILY ROUTING ───────────────────────────────────────────────────────────
# These are derived from real deploy_rulebook outputs, not acca_builder.

FAMILY_PRIORITY = {
    "FTR_CS_PROMOTED": 1,
    "BTTS_STRONG": 2,
    "BTTS_NO_STRONG": 2,
    "HW_HGE2_COMBINED": 3,
    "AW_AGE2_COMBINED": 4,
    "OU25_PREMIUM": 5,
    "OU25_SHADOW_PREMIUM": 6,
    "BTTS_SHADOW_STRONG": 7,
    "FTR_POWER_SHADOW": 8,
    "FTR_BASE": 99,
    "OTHER": 999,
}

MONSTER_ACCA_FAMILIES = {
    "FTR_CS_PROMOTED",
    "BTTS_STRONG",
    "BTTS_NO_STRONG",
    "HW_HGE2_COMBINED",
}


DEPLOY_TIER_PRIORITY = {
    "ELITE": 1,
    "STANDARD": 2,
    "OBSERVE": 3,
    "": 9,
}

FTR_PRIORITY_ORDER = {
    "CONSENSUS_ELITE": 1,
    "CAT_ELITE": 2,
    "STANDARD": 3,
    "": 9,
}

# Priority buckets and risk policy constants
PRIORITY_BUCKET_ORDER = {
    "PRIORITY_1_FTR_CS_PROMOTED": 1,
    "PRIORITY_2_BTTS_STRONG": 2,
    "PRIORITY_3_HW_HGE2": 3,
    "PRIORITY_4_AW_AGE2": 4,
    "PRIORITY_5_OU25_PREMIUM": 5,
    "PRIORITY_5B_OU25_SHADOW_PREMIUM": 6,
    "PRIORITY_2B_BTTS_SHADOW_STRONG": 7,
    "PRIORITY_6_FTR_BASE": 8,
    "PRIORITY_6B_FTR_POWER_SHADOW": 9,
    "UNPRIORITISED": 999,
}

HW_HGE2_DENYLIST = {
    "England Championship",
    "England EFL League 1",
    "England FA Cup",
    "Japan J1",
}

HW_HGE2_MONITOR_LEAGUES = {
    "England Premier League",
    "Belgium Pro",
    "Italy Serie A",
}

AW_AGE2_DENYLIST = {
    "England Championship",
    "England EFL League 1",
    "England FA Cup",
    "Spain La Liga",
    "Belgium Pro",
}

AW_AGE2_STRICT_LAMBDA_LEAGUES = {
    "Italy Serie A",
    "Netherlands Eredivisie",
}

OU25_DENYLIST = {
    "Scotland Premiership",
    "England Championship",
    "Belgium Pro",
    "England Premier League",
}

BTTS_SHADOW_DENYLIST = {
    "Brazil Serie A",
}

FTR_POWER_SHADOW_MIN_MODEL_P = 0.52
FTR_POWER_SHADOW_MIN_MARGIN = 0.20
FTR_POWER_SHADOW_MIN_MASS = 0.20


FIXED_ACCA_SIZES = [3, 4, 5, 6, 8, 10, 12, 14, 16, 18]
DOUBLES_MAX_LEGS = 12
TREBLES_MAX_LEGS = 10


# ── BASIC HELPERS ────────────────────────────────────────────────────────────

def safe_str(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def safe_float(value, default=None):
    try:
        s = safe_str(value)
        if s == "":
            return default
        return float(s)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=None):
    try:
        s = safe_str(value)
        if s == "":
            return default
        return int(float(s))
    except (TypeError, ValueError):
        return default


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def first_present(row: dict, keys: list[str], default=""):
    for k in keys:
        if k in row:
            v = row.get(k)
            if safe_str(v) != "":
                return v
    return default


def normalize_upper(value) -> str:
    return safe_str(value).upper()


def normalize_market(value) -> str:
    return safe_str(value).lower()


def parse_bool_keep(value) -> bool:
    return safe_int(value, 0) == 1


def split_reason_tokens(text: str) -> list[str]:
    s = safe_str(text)
    if not s or s.lower() == "nan":
        return []
    return [x.strip() for x in s.split("|") if x.strip()]


# ── REASON TOKEN HELPERS ─────────────────────────────────────────────────────

def has_any_reason_token(text: str, tokens: set[str]) -> bool:
    vals = {x.upper() for x in split_reason_tokens(text)}
    return any(tok in vals for tok in tokens)


def row_has_promoted_standard_context(row: dict) -> bool:
    std_bucket = normalize_upper(first_present(row, COL_STD_BUCKET))
    context_tokens = {x.upper() for x in split_reason_tokens(first_present(row, COL_CONTEXT))}

    if std_bucket == "STANDARD_FTR_CS_PROMOTED_ALIGNED":
        return True

    if std_bucket.startswith("STANDARD_FTR_COMBO_HW"):
        return True

    if std_bucket.startswith("STANDARD_FTR_COMBO_AW"):
        return True

    if "FTR_CS_PROMOTE_STANDARD" in context_tokens:
        return True

    if "FTR_OBSERVE_RESCUE_STANDARD" in context_tokens:
        return True

    return False


def row_reason_token_set(row: dict) -> set[str]:
    tokens: set[str] = set()
    for col in (
        "review_reason",
        "context_reason_codes",
        "deterministic_warn_reason",
        "deterministic_veto_reason",
        "deploy_veto_reason",
        "learned_veto_reason",
    ):
        tokens.update(x.upper() for x in split_reason_tokens(safe_str(row.get(col, ""))))
    return {t for t in tokens if t}


def is_ou25_shadow_candidate(row: dict) -> bool:
    if normalize_market(row.get("market", "")) != "ou25":
        return False
    if normalize_upper(row.get("selection", "")) != "OVER25":
        return False

    model_p = row.get("model_p")
    cs_mass = row.get("cs_over25_mass")
    top3 = row.get("top3_over_count")
    avg_btts = row.get("avg_btts_rate")

    if model_p is None or model_p < 0.62:
        return False
    if cs_mass is None or cs_mass < 0.12:
        return False
    if top3 is None or top3 < 1:
        return False
    if avg_btts is None or avg_btts < 0.45:
        return False

    return True


def is_btts_shadow_candidate(row: dict) -> bool:
    if normalize_market(row.get("market", "")) != "btts":
        return False
    if normalize_upper(row.get("selection", "")) != "YES":
        return False
    if safe_str(row.get("league", "")) in BTTS_SHADOW_DENYLIST:
        return False
    if normalize_upper(row.get("signal_btts", "")) != "STRONG_YES":
        return False

    model_p = row.get("model_p")
    if model_p is None or model_p < 0.62:
        return False

    fts_max = row.get("fts_max")
    if fts_max is not None and fts_max >= 0.35:
        return False

    btts_yes_ge2_min = row.get("btts_yes_ge2_min")
    if btts_yes_ge2_min is not None and btts_yes_ge2_min < 0.22:
        return False

    return True


def parse_scoreline(score: str) -> tuple[int | None, int | None]:
    txt = safe_str(score)
    if not txt or "-" not in txt:
        return None, None
    try:
        a, b = txt.split("-", 1)
        return int(a.strip()), int(b.strip())
    except Exception:
        return None, None


def is_ftr_power_shadow_candidate(row: dict) -> bool:
    if normalize_market(row.get("market", "")) != "ftr":
        return False

    tokens = row_reason_token_set(row)
    if "FTR_ELITE_BLOCK_POWER" not in tokens:
        return False
    if "FTR_ELITE_BLOCK_SELECTION_MISMATCH" in tokens:
        return False
    if "FTR_ELITE_BLOCK_DOUBLE_CHAOS_WARN" in tokens:
        return False

    selection = normalize_upper(row.get("selection", ""))
    power_diff = row.get("power_diff")
    if power_diff is None:
        return False

    direction_ok = (
        (selection == "AWAY" and power_diff < 0)
        or (selection == "HOME" and power_diff > 0)
    )
    if not direction_ok:
        return False

    model_p = row.get("model_p")
    ftr_margin = row.get("ftr_margin")
    mass = row.get("pick_side_mass_top3")

    if model_p is None or model_p < FTR_POWER_SHADOW_MIN_MODEL_P:
        return False
    if ftr_margin is None or ftr_margin < FTR_POWER_SHADOW_MIN_MARGIN:
        return False
    if mass is None or mass < FTR_POWER_SHADOW_MIN_MASS:
        return False

    return True


# ── FAMILY DERIVATION ────────────────────────────────────────────────────────

def derive_family(row: dict) -> str:
    market = normalize_market(first_present(row, COL_MARKET))
    selection = normalize_upper(first_present(row, COL_SELECTION))
    std_bucket = normalize_upper(first_present(row, COL_STD_BUCKET))
    review_reason = normalize_upper(first_present(row, COL_REVIEW_REASON))
    product_profile = normalize_upper(first_present(row, COL_PRODUCT_PROFILE))
    product_lane = normalize_upper(first_present(row, COL_PRODUCT_LANE))

    if std_bucket == "STANDARD_FTR_CS_PROMOTED_ALIGNED":
        return "FTR_CS_PROMOTED"

    if std_bucket.startswith("STANDARD_FTR_COMBO_HW"):
        return "HW_HGE2_COMBINED"

    if std_bucket.startswith("STANDARD_FTR_COMBO_AW"):
        return "AW_AGE2_COMBINED"

    if std_bucket == "STANDARD_BTTS":
        if selection == "NO" or product_lane == "BTTS_NO" or product_profile == "BTTS_NO":
            return "BTTS_NO_STRONG"
        return "BTTS_STRONG"

    if std_bucket == "STANDARD_OU25":
        return "OU25_PREMIUM"

    if std_bucket == "STANDARD_FTR_BASE":
        return "FTR_BASE"

    if review_reason == "FTR_PRIORITY_KEEP":
        return "FTR_BASE"

    if review_reason == "FTR_COMBO_PRIORITY_KEEP":
        return "HW_HGE2_COMBINED"

    if review_reason == "BTTS_PRIORITY_KEEP":
        if selection == "NO":
            return "BTTS_NO_STRONG"
        return "BTTS_STRONG"

    # Insert shadow family logic here
    if is_ou25_shadow_candidate(row):
        return "OU25_SHADOW_PREMIUM"

    if is_btts_shadow_candidate(row):
        return "BTTS_SHADOW_STRONG"

    if is_ftr_power_shadow_candidate(row):
        return "FTR_POWER_SHADOW"

    if market == "btts":
        if selection == "NO":
            return "BTTS_NO_STRONG"
        return "BTTS_STRONG"

    if market == "ou25":
        return "OU25_PREMIUM"

    if market == "ftr":
        return "FTR_BASE"

    return "OTHER"
def derive_priority_bucket(row: dict) -> str:
    league = safe_str(row.get("league", ""))
    family = safe_str(row.get("family", ""))
    market = normalize_market(row.get("market", ""))
    selection = normalize_upper(row.get("selection", ""))
    std_bucket = normalize_upper(row.get("standard_reporting_bucket", ""))
    draw_warning_token = normalize_upper(row.get("draw_warning_token", ""))
    signal_btts = normalize_upper(row.get("signal_btts", ""))

    model_p = row.get("model_p")
    lambda_home = row.get("lambda_home")
    lambda_away = row.get("lambda_away")
    p_hw_and_hge2 = row.get("p_hw_and_hge2")
    p_aw_and_age2 = row.get("p_aw_and_age2")
    ge2_gap = row.get("ge2_gap")
    fts_max = row.get("fts_max")
    btts_yes_ge2_min = row.get("btts_yes_ge2_min")
    avg_btts_rate = row.get("avg_btts_rate")
    cs_over25_mass = row.get("cs_over25_mass")
    top3_over_count = row.get("top3_over_count")

    if std_bucket == "STANDARD_FTR_CS_PROMOTED_ALIGNED":
        if model_p is not None and model_p >= 0.43 and draw_warning_token != "NO_DRAWWARN":
            return "PRIORITY_1_FTR_CS_PROMOTED"

    if market == "btts" and selection == "YES" and league != "Brazil Serie A":
        if signal_btts in {"VERY_STRONG_YES", "VERY_STRONG"}:
            if (fts_max is not None and fts_max < 0.35) and (btts_yes_ge2_min is not None and btts_yes_ge2_min >= 0.22):
                return "PRIORITY_2_BTTS_STRONG"

    if family == "HW_HGE2_COMBINED":
        if league not in HW_HGE2_DENYLIST:
            lambda_home_ok = (lambda_home is not None and lambda_home >= 2.75)
            p_combo_ok = (p_hw_and_hge2 is not None and p_hw_and_hge2 >= 0.65)
            ge2_gap_ok = (ge2_gap is not None and ge2_gap < 0.25)
            away_cap_ok = (lambda_away is not None and lambda_away < 1.20)

            base_ok = lambda_home_ok and p_combo_ok and ge2_gap_ok and away_cap_ok
            bucket_whitelist = std_bucket.startswith("STANDARD_FTR_COMBO_HW")

            if base_ok or bucket_whitelist:
                if league in HW_HGE2_MONITOR_LEAGUES and (lambda_home is None or lambda_home < 3.00):
                    pass
                else:
                    return "PRIORITY_3_HW_HGE2"

    if family == "AW_AGE2_COMBINED":
        if league not in AW_AGE2_DENYLIST:
            lambda_away_ok = (lambda_away is not None and lambda_away >= 2.75)
            p_combo_ok = (p_aw_and_age2 is not None and p_aw_and_age2 >= 0.65)
            ge2_gap_ok = (ge2_gap is not None and ge2_gap < 0.25)
            bucket_whitelist = std_bucket.startswith("STANDARD_FTR_COMBO_AW")

            if (lambda_away_ok and p_combo_ok and ge2_gap_ok) or bucket_whitelist:
                if league in AW_AGE2_STRICT_LAMBDA_LEAGUES and (lambda_away is None or lambda_away < 3.00):
                    pass
                else:
                    return "PRIORITY_4_AW_AGE2"

    if std_bucket == "STANDARD_OU25" or family == "OU25_PREMIUM":
        if league not in OU25_DENYLIST:
            if (
                avg_btts_rate is not None and avg_btts_rate >= 0.50
                and cs_over25_mass is not None and cs_over25_mass >= 0.15
                and top3_over_count is not None and top3_over_count >= 2
            ):
                return "PRIORITY_5_OU25_PREMIUM"
            if std_bucket == "STANDARD_OU25":
                return "PRIORITY_5_OU25_PREMIUM"

    # Insert shadow priority bucket logic here
    if family == "OU25_SHADOW_PREMIUM":
        return "PRIORITY_5B_OU25_SHADOW_PREMIUM"

    if family == "BTTS_SHADOW_STRONG":
        return "PRIORITY_2B_BTTS_SHADOW_STRONG"

    if family == "FTR_POWER_SHADOW":
        return "PRIORITY_6B_FTR_POWER_SHADOW"

    if family == "FTR_BASE":
        return "PRIORITY_6_FTR_BASE"

    return "UNPRIORITISED"


def family_is_allowed(family: str, exclude_ftr_base: bool, exclude_ou25: bool) -> bool:
    if family == "OTHER":
        return False
    if family == "FTR_BASE" and exclude_ftr_base:
        return False
    if family in {"OU25_PREMIUM", "OU25_SHADOW_PREMIUM"} and exclude_ou25:
        return False
    return True


# ── BLOCKING / RISK POLICY ───────────────────────────────────────────────────

def is_regime_flagged(row: dict) -> bool:
    reasons = []
    reasons.extend(split_reason_tokens(first_present(row, COL_REVIEW_REASON)))
    reasons.extend(split_reason_tokens(first_present(row, COL_CONTEXT)))
    reasons.extend(split_reason_tokens(first_present(row, COL_DET_WARN)))
    reasons.extend(split_reason_tokens(first_present(row, COL_DET_VETO)))
    reasons.extend(split_reason_tokens(first_present(row, COL_DEPLOY_VETO)))
    reasons.extend(split_reason_tokens(first_present(row, COL_LEARNED_VETO)))

    tokens = {safe_str(x).upper() for x in reasons if safe_str(x)}

    if row_has_promoted_standard_context(row):
        allowed_with_promotion = {
            "DEMOTED_TO_OBSERVE",
            "FTR_CS_PROMOTE_STANDARD",
            "FTR_OBSERVE_RESCUE_STANDARD",
        }
        tokens = {t for t in tokens if t not in allowed_with_promotion}

    hard_block_tokens = {
        "DOWNGRADE_REGIME",
        "BLOCKED_LIVE",
        "BLOCKED_POWER",
        "BLOCKED_POWER_DIRECTION",
        "BLOCK_PICK_SIDE_MARGIN",
        "SELECTION_MISMATCH",
        "DOUBLE_CHAOS_WARN",
        "SHORTPRICE_MODEL_LT_0P41_BLOCK",
    }

    return any(tok in tokens for tok in hard_block_tokens)


def tier_allowed(row: dict, include_observe: bool) -> bool:
    tier = normalize_upper(first_present(row, COL_DEPLOY_TIER))
    if tier in {"ELITE", "STANDARD"}:
        return True
    if include_observe and tier == "OBSERVE":
        return True
    return False


def compute_rank_key(row: dict):
    family = row["family"]
    tier = normalize_upper(row["deploy_tier"])
    priority_bucket = safe_str(row.get("priority_bucket", "UNPRIORITISED"))
    priority_rank = PRIORITY_BUCKET_ORDER.get(priority_bucket, 999)
    family_pri = FAMILY_PRIORITY.get(family, 999)
    tier_pri = DEPLOY_TIER_PRIORITY.get(tier, 9)
    ftr_priority = normalize_upper(row.get("ftr_priority", ""))
    ftr_pri = FTR_PRIORITY_ORDER.get(ftr_priority, 9)
    slip_leg_score = row.get("slip_leg_score", 0.0)
    model_p = row["model_p"]
    odds = row["odds"]
    if row.get("market") == "ftr":
        return (-slip_leg_score, priority_rank, ftr_pri, family_pri, tier_pri, -model_p, odds)
    return (-slip_leg_score, priority_rank, family_pri, tier_pri, -model_p, odds)


def _value_component(row: dict) -> float:
    edge_bps = row.get("value_edge_bps")
    if edge_bps is not None:
        return clamp(float(edge_bps) / 250.0, 0.0, 1.0) * 20.0

    tier = normalize_upper(row.get("value_edge_tier", ""))
    if tier == "PREMIUM":
        return 20.0
    if tier == "STRONG":
        return 14.0
    if tier == "STANDARD":
        return 8.0

    gap = row.get("value_gap_pct_points")
    if gap is not None:
        return clamp(float(gap) / 0.10, 0.0, 1.0) * 12.0
    return 0.0


def _conviction_component(row: dict) -> float:
    meta_tier = normalize_upper(row.get("meta_tier", ""))
    family = safe_str(row.get("family", ""))
    std_bucket = normalize_upper(row.get("standard_reporting_bucket", ""))
    ftr_priority = normalize_upper(row.get("ftr_priority", ""))

    if meta_tier == "META_ELITE_PLUS":
        base = 15.0
    elif meta_tier == "META_ELITE":
        base = 12.0
    elif meta_tier == "META_STANDARD":
        base = 7.0
    else:
        base = 0.0

    # Family-aware conviction bonuses so structurally strong routed rows
    # are not flattened by a generic probability-first scoring rule.
    if family == "FTR_CS_PROMOTED":
        base += 14.0
    elif family == "BTTS_NO_STRONG":
        base += 7.0
    elif family == "BTTS_STRONG":
        base += 5.0
    elif family == "OU25_PREMIUM":
        base += 4.0

    if std_bucket == "STANDARD_FTR_CS_PROMOTED_ALIGNED":
        base += 8.0

    if ftr_priority == "CONSENSUS_ELITE":
        base += 4.0
    elif ftr_priority == "CAT_ELITE":
        base += 2.0

    return min(base, 25.0)


def _team_intel_component(row: dict) -> tuple[float, int, int]:
    uplift = row.get("team_intel_overlay_rank_uplift_bps")
    penalty = row.get("team_intel_overlay_rank_penalty_bps")
    caution = safe_int(row.get("team_intel_overlay_slip_caution_flag"), 0) or 0
    avoid_in_acca = safe_int(row.get("team_intel_overlay_avoid_in_acca_flag"), 0) or 0

    score = 0.0
    if uplift is not None:
        score += clamp(float(uplift) / 10.0, 0.0, 10.0)
    if penalty is not None:
        score -= clamp(float(penalty) / 10.0, 0.0, 10.0)

    if caution:
        score -= 2.0
    if avoid_in_acca:
        score -= 4.0

    # Fallback to raw Team Intel context when overlay fields do not exist yet.
    if score == 0.0 and not caution and not avoid_in_acca:
        interaction = normalize_upper(row.get("team_fixture_interaction", ""))
        if interaction in {"HOME_GE2_POCKET", "AWAY_GE2_POCKET"}:
            score += 3.0
        elif interaction == "SCORER_VS_SCORER":
            score += 2.0
        elif interaction == "CS_VS_CS":
            score -= 1.0

    return score, caution, avoid_in_acca


def _correct_score_component(row: dict) -> float:
    bucket = normalize_upper(row.get("cs_concentration_bucket", ""))
    banker = safe_int(row.get("cs_banker_flag"), 0) or 0
    supported = safe_int(row.get("cs_supported_flag"), 0) or 0
    fragility_penalty = safe_float(row.get("cs_fragility_penalty"), 0.0) or 0.0
    banker_bonus = safe_float(row.get("cs_banker_bonus"), 0.0) or 0.0
    market = normalize_market(row.get("market", ""))
    if market == "ftr":
        alignment = normalize_upper(row.get("cs_ftr_alignment", ""))
    elif market == "btts":
        alignment = normalize_upper(row.get("cs_btts_alignment", ""))
    elif market == "ou25":
        alignment = normalize_upper(row.get("cs_ou25_alignment", ""))
    else:
        alignment = ""
    diffuse = safe_int(row.get("cs_diffuse_flag"), 0) or 0
    one_sided = safe_int(row.get("cs_one_sided_flag"), 0) or 0
    draw_family = safe_int(row.get("cs_draw_family_flag"), 0) or 0
    fragile = safe_int(row.get("cs_fragility_flag"), 0) or 0
    entropy = row.get("cs_entropy")

    score = 0.0
    if bucket == "ELITE":
        score += 4.0
    elif bucket == "STRONG":
        score += 2.5
    elif bucket == "MODERATE":
        score += 1.0

    if banker:
        score += 1.5

    if alignment == "STRONG_ALIGN":
        score += 5.0
    elif alignment == "SOFT_ALIGN":
        score += 2.5
    elif alignment == "NEUTRAL":
        score += 0.0
    elif alignment == "FRAGILE":
        score -= 2.5
    elif alignment == "CONFLICT":
        score -= 5.0

    if entropy is not None and entropy >= 3.2:
        score -= 1.0
    if fragile:
        score -= 1.0
    if market == "btts" and one_sided:
        score -= 2.0
    if market == "ou25" and draw_family:
        score += 1.0
    if market == "ftr" and one_sided:
        score += 0.5
    if supported:
        score += 1.5
    score += banker_bonus
    score -= fragility_penalty * 1.5
    return clamp(score, -6.0, 12.0)


def _market_trust_component(row: dict) -> float:
    market = normalize_market(row.get("market", ""))
    if market == "btts":
        return 9.0
    if market == "ftr":
        return 8.0
    if market == "ou25":
        return 7.0
    return 5.0


def _odds_shape_component(row: dict) -> float:
    odds = row.get("odds")
    if odds is None:
        return 0.0
    odds = float(odds)
    if 1.45 <= odds <= 2.40:
        return 10.0
    if 2.41 <= odds <= 3.20:
        return 7.0
    if 1.30 <= odds <= 1.44:
        return 6.0
    if odds < 1.30:
        return 2.0
    return 3.0


def _trap_penalty(row: dict) -> float:
    penalty = 0.0
    tokens = row_reason_token_set(row)
    hard = {
        "DOWNGRADE_REGIME",
        "BLOCKED_LIVE",
        "BLOCKED_POWER",
        "SELECTION_MISMATCH",
        "DOUBLE_CHAOS_WARN",
        "SHORTPRICE_MODEL_LT_0P41_BLOCK",
    }
    moderate = {
        "DRAWTRAP",
        "FTR_NOT_GLUE_WARN",
        "DEMOTED_TO_OBSERVE",
    }
    if any(tok in tokens for tok in hard):
        penalty += 25.0
    if any(tok in tokens for tok in moderate):
        penalty += 12.0

    if safe_int(row.get("phase9b_premium_ge2_test_lane"), 0) == 1:
        penalty -= 2.0

    return max(penalty, 0.0)


def _modern_confidence_bonus(row: dict) -> float:
    market = normalize_market(row.get("market", ""))
    home_ge2 = row.get("p_home_ge2")
    away_ge2 = row.get("p_away_ge2")
    vals = [v for v in (home_ge2, away_ge2) if v is not None]
    if not vals:
        return 0.0
    top_ge2 = max(vals)
    if market == "ftr":
        return clamp((top_ge2 - 0.55) / 0.25, 0.0, 1.0) * 5.0
    if market == "btts":
        return clamp((min(vals) - 0.20) / 0.25, 0.0, 1.0) * 3.0
    if market == "ou25":
        return clamp((top_ge2 - 0.55) / 0.25, 0.0, 1.0) * 4.0
    return 0.0


def _derive_cs_composite_support(row: dict) -> str:
    market = normalize_market(row.get("market", ""))
    alignment = ""
    if market == "ftr":
        alignment = normalize_upper(row.get("cs_ftr_alignment", ""))
    elif market == "btts":
        alignment = normalize_upper(row.get("cs_btts_alignment", ""))
    elif market == "ou25":
        alignment = normalize_upper(row.get("cs_ou25_alignment", ""))

    cluster = normalize_upper(row.get("cs_cluster_family", ""))
    banker = safe_int(row.get("cs_banker_flag"), 0) or 0
    supported = safe_int(row.get("cs_supported_flag"), 0) or 0

    if supported and banker and alignment == "STRONG_ALIGN":
        return "BANKER_SUPPORT"
    if market == "ftr" and alignment in {"STRONG_ALIGN", "SOFT_ALIGN"} and cluster in {
        "CS_CLUSTER_1_0_2_0_HOME",
        "CS_CLUSTER_2_1_3_1_HOME",
        "CS_CLUSTER_0_1_0_2_AWAY",
        "CS_CLUSTER_2_1_3_1_AWAY",
    }:
        return "FTR_EXPRESSION_VALIDATED"
    if market == "btts" and alignment in {"STRONG_ALIGN", "SOFT_ALIGN"} and cluster in {
        "CS_CLUSTER_1_1_2_2_BTTS_OVER",
        "CS_CLUSTER_HIGH_EVENT",
    }:
        return "BTTS_EXPRESSION_VALIDATED"
    if market == "ou25" and alignment in {"STRONG_ALIGN", "SOFT_ALIGN"} and cluster in {
        "CS_CLUSTER_HIGH_EVENT",
        "CS_CLUSTER_1_1_2_2_BTTS_OVER",
        "CS_CLUSTER_2_1_3_1_HOME",
        "CS_CLUSTER_2_1_3_1_AWAY",
    }:
        return "OU25_EXPRESSION_VALIDATED"
    if supported:
        return "GENERAL_SUPPORT"
    if normalize_upper(alignment) == "FRAGILE":
        return "FRAGILE_SUPPORT"
    if normalize_upper(alignment) == "CONFLICT":
        return "CONFLICT_SUPPORT"
    return "NEUTRAL_SUPPORT"


def _market_cs_alignment(row: dict) -> str:
    market = normalize_market(row.get("market", ""))
    if market == "ftr":
        return normalize_upper(row.get("cs_ftr_alignment", ""))
    if market == "btts":
        return normalize_upper(row.get("cs_btts_alignment", ""))
    if market == "ou25":
        return normalize_upper(row.get("cs_ou25_alignment", ""))
    return ""


def _parse_match_ts(value: object) -> datetime | None:
    text = safe_str(value)
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(text[:19], fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _season_phase_from_ts(ts: datetime | None) -> str:
    if ts is None:
        return "UNKNOWN"
    month = ts.month
    if month in {8, 9}:
        return "EARLY_SEASON"
    if month in {10, 11, 12}:
        return "AUTUMN_DENSE"
    if month in {1, 2}:
        return "WINTER_RESET"
    if month in {3, 4, 5}:
        return "RUN_IN"
    return "SUMMER_TRANSITION"


def _rank_band(rank: int) -> str:
    if rank <= 8:
        return "TOP_8"
    if rank <= 10:
        return "RANK_9_10"
    return "RANK_11_PLUS"


def _timing_risk_profile(row: dict) -> dict:
    ts = _parse_match_ts(row.get("match_date"))
    season_phase = _season_phase_from_ts(ts)
    month = ts.month if ts is not None else None
    champions_league_phase_flag = int(month in {9, 10, 11, 12, 2, 3, 4}) if month is not None else 0
    international_break_zone_flag = int(month in {3, 9, 10, 11}) if month is not None else 0

    modifier = 0.0
    notes: list[str] = []
    market = normalize_market(row.get("market", ""))
    selection = normalize_upper(row.get("selection", ""))
    cs_support = normalize_upper(row.get("cs_composite_support", ""))
    team_action = normalize_upper(row.get("team_intel_overlay_action", ""))

    if market == "btts":
        if season_phase == "AUTUMN_DENSE" and month == 10:
            modifier -= 2.0
            notes.append("AUTUMN_DENSE_OCTOBER_BTTS")
        if season_phase == "AUTUMN_DENSE" and month == 11:
            modifier -= 2.0
            notes.append("AUTUMN_DENSE_NOVEMBER_BTTS")
        if season_phase == "EARLY_SEASON" and month == 9:
            modifier -= 1.5
            notes.append("EARLY_SEASON_SEPT_BTTS")
        if season_phase == "RUN_IN" and month == 4:
            modifier -= 1.5
            notes.append("RUN_IN_APRIL_BTTS")
        if season_phase == "RUN_IN" and month in {3, 5}:
            modifier -= 1.75
            notes.append("RUN_IN_MAR_MAY_BTTS")
        if season_phase == "WINTER_RESET" and month == 2:
            modifier -= 1.75
            notes.append("WINTER_RESET_FEB_BTTS")
        if season_phase == "SUMMER_TRANSITION" and month == 6:
            modifier -= 1.5
            notes.append("SUMMER_TRANSITION_JUNE_BTTS")
        if month in {3, 6, 11}:
            modifier -= 1.0
            notes.append("KNOWN_POISON_MONTH")
        if selection == "YES" and cs_support in {"GENERAL_SUPPORT", "NEUTRAL_SUPPORT"}:
            modifier -= 1.5
            notes.append("BTTS_YES_GENERAL_SUPPORT")
        if selection == "YES" and cs_support == "CONFLICT_SUPPORT":
            modifier -= 3.0
            notes.append("BTTS_YES_CONFLICT_SUPPORT")
        if selection == "NO" and normalize_upper(row.get("cs_btts_alignment", "")) == "SOFT_ALIGN":
            modifier -= 1.0
            notes.append("BTTS_NO_SOFT_ALIGN")
        if team_action in {"", "NEUTRAL"}:
            modifier -= 0.75
            notes.append("BTTS_NEUTRAL_OVERLAY")
        if selection == "YES" and team_action in {"", "NEUTRAL"} and normalize_upper(row.get("cs_btts_alignment", "")) in {"SOFT_ALIGN", "CONFLICT"}:
            modifier -= 1.25
            notes.append("BTTS_YES_NEUTRAL_SOFT_OR_CONFLICT")

    return {
        "window_month": month if month is not None else "",
        "season_phase": season_phase,
        "champions_league_phase_flag": champions_league_phase_flag,
        "international_break_zone_flag": international_break_zone_flag,
        "timing_risk_modifier": modifier,
        "timing_risk_notes": "|".join(notes),
    }


def _audit_policy_adjustment(row: dict) -> dict:
    """First audit-driven slip policy.

    This is intentionally slip-only. It does not change deploy validity; it only
    adds ranking pressure and acca-safety hints from historical weak-link audits.
    """
    market = normalize_market(row.get("market", ""))
    std_bucket = normalize_upper(row.get("standard_reporting_bucket", ""))
    interaction = normalize_upper(row.get("team_fixture_interaction", ""))
    context_family = normalize_upper(row.get("team_context_filter_family", ""))
    cs_supported = safe_int(row.get("cs_supported_flag"), 0) or 0
    cs_fragile = safe_int(row.get("cs_fragility_flag"), 0) or 0
    cs_alignment = _market_cs_alignment(row)

    policy = {
        "action": "NONE",
        "reason": "",
        "score_delta": 0.0,
        "force_avoid_in_acca": 0,
        "force_safe_large_acca_block": 0,
    }

    # Preserve historically strong buckets.
    preserve_bucket = (
        (market == "ftr" and std_bucket == "STANDARD_FTR_CS_PROMOTED_ALIGNED")
        or (market == "btts" and std_bucket == "STANDARD_BTTS")
        or (market == "ou25" and std_bucket == "STANDARD_OU25")
    )
    preserve_ok = (
        preserve_bucket
        and cs_supported == 1
        and cs_fragile == 0
        and cs_alignment not in {"CONFLICT", "FRAGILE"}
    )

    if preserve_ok:
        policy.update(
            {
                "action": "PRESERVE",
                "reason": "AUDIT_PRESERVE_STRONG_BUCKET",
                "score_delta": 3.0,
            }
        )
        return policy

    # First denylist bucket.
    if (
        market == "ou25"
        and std_bucket == "STANDARD_OTHER"
        and interaction == "AWAY_GE2_POCKET"
        and context_family == "AWAY_TEAM_GOALS"
    ):
        policy.update(
            {
                "action": "DENYLIST",
                "reason": "AUDIT_DENYLIST_OU25_AWAY_GE2_STANDARD_OTHER",
                "score_delta": -12.0,
                "force_avoid_in_acca": 1,
                "force_safe_large_acca_block": 1,
            }
        )
        return policy

    # First rank-down buckets.
    if (
        market == "ftr"
        and std_bucket == "STANDARD_OTHER"
        and (
            (interaction == "OTHER" and context_family == "GENERAL")
            or (interaction == "AWAY_SCORER_VS_HOME_CS" and context_family == "DIRECTIONAL_CLASH")
            or (interaction == "CS_VS_CS" and context_family == "LOW_EVENT")
        )
    ):
        reason = {
            "OTHER": "AUDIT_RANKDOWN_FTR_STANDARD_OTHER_GENERAL",
            "AWAY_SCORER_VS_HOME_CS": "AUDIT_RANKDOWN_FTR_AWAY_SCORER_VS_HOME_CS",
            "CS_VS_CS": "AUDIT_RANKDOWN_FTR_CS_VS_CS",
        }[interaction]
        policy.update(
            {
                "action": "RANK_DOWN",
                "reason": reason,
                "score_delta": -5.0 if interaction == "OTHER" else -6.0,
                "force_safe_large_acca_block": 1,
            }
        )
        return policy

    if (
        market == "ou25"
        and std_bucket == "STANDARD_OTHER"
        and (
            (interaction == "OTHER" and context_family == "GENERAL")
            or (interaction == "CS_VS_CS" and context_family == "LOW_EVENT")
        )
    ):
        reason = {
            "OTHER": "AUDIT_RANKDOWN_OU25_STANDARD_OTHER_GENERAL",
            "CS_VS_CS": "AUDIT_RANKDOWN_OU25_CS_VS_CS",
        }[interaction]
        policy.update(
            {
                "action": "RANK_DOWN",
                "reason": reason,
                "score_delta": -5.0,
                "force_safe_large_acca_block": 1,
            }
        )
        return policy

    return policy


def compute_slip_leg_score(row: dict) -> dict:
    model_p = row.get("model_p")
    market = normalize_market(row.get("market", ""))
    family = safe_str(row.get("family", ""))
    std_bucket = normalize_upper(row.get("standard_reporting_bucket", ""))
    ftr_priority = normalize_upper(row.get("ftr_priority", ""))
    ftr_margin = row.get("ftr_margin")

    base_quality = 0.0
    if market == "ftr":
        # FTR needs a gentler curve than BTTS/OU25 because promoted FTR rows
        # often express conviction via structure and margin, not only raw p.
        if model_p is not None:
            base_quality += clamp((float(model_p) - 0.42) / 0.20, 0.0, 1.0) * 18.0
        if ftr_margin is not None:
            base_quality += clamp(float(ftr_margin) / 0.35, 0.0, 1.0) * 10.0
        if family == "FTR_CS_PROMOTED":
            base_quality += 8.0
        if std_bucket == "STANDARD_FTR_CS_PROMOTED_ALIGNED":
            base_quality += 6.0
        if ftr_priority == "CONSENSUS_ELITE":
            base_quality += 4.0
        elif ftr_priority == "CAT_ELITE":
            base_quality += 2.0
        base_quality = min(base_quality, 35.0)
    else:
        if model_p is not None:
            base_quality = clamp((float(model_p) - 0.50) / 0.35, 0.0, 1.0) * 35.0

    value_component = _value_component(row)
    conviction_component = _conviction_component(row)
    team_intel_component, caution, avoid_in_acca = _team_intel_component(row)
    correct_score_component = _correct_score_component(row)
    market_trust_component = _market_trust_component(row)
    odds_shape_component = _odds_shape_component(row)
    modern_confidence_component = _modern_confidence_bonus(row)
    timing_profile = _timing_risk_profile(row)
    timing_component = safe_float(timing_profile.get("timing_risk_modifier"), 0.0) or 0.0
    trap_penalty = _trap_penalty(row)
    audit_policy = _audit_policy_adjustment(row)
    audit_policy_score_delta = safe_float(audit_policy.get("score_delta"), 0.0) or 0.0

    slip_leg_score = (
        base_quality
        + value_component
        + conviction_component
        + team_intel_component
        + correct_score_component
        + market_trust_component
        + odds_shape_component
        + modern_confidence_component
        + timing_component
        + audit_policy_score_delta
        - trap_penalty
    )
    slip_leg_score = clamp(slip_leg_score, 0.0, 100.0)

    if slip_leg_score >= 90:
        bucket = "P1"
    elif slip_leg_score >= 80:
        bucket = "P2"
    elif slip_leg_score >= 70:
        bucket = "P3"
    elif slip_leg_score >= 60:
        bucket = "REVIEW"
    else:
        bucket = "SHADOW"

    row["slip_leg_base_quality"] = round(base_quality, 4)
    row["slip_leg_value_component"] = round(value_component, 4)
    row["slip_leg_conviction_component"] = round(conviction_component, 4)
    row["slip_leg_team_intel_component"] = round(team_intel_component, 4)
    row["slip_leg_correct_score_component"] = round(correct_score_component, 4)
    row["slip_leg_audit_policy_component"] = round(audit_policy_score_delta, 4)
    row["slip_leg_market_trust_component"] = round(market_trust_component, 4)
    row["slip_leg_odds_shape_component"] = round(odds_shape_component, 4)
    row["slip_leg_modern_confidence_component"] = round(modern_confidence_component, 4)
    row["slip_leg_timing_component"] = round(timing_component, 4)
    row["slip_leg_trap_penalty"] = round(trap_penalty, 4)
    row["slip_leg_score"] = round(slip_leg_score, 4)
    row["slip_leg_bucket"] = bucket
    row["slip_leg_caution_flag"] = caution
    row["slip_leg_avoid_in_acca_flag"] = avoid_in_acca
    meta_tier = normalize_upper(row.get("meta_tier", ""))
    team_intel_support = int(team_intel_component > 0)
    cs_support = int(safe_int(row.get("cs_supported_flag"), 0) == 1)
    meta_support = int(meta_tier in {"META_ELITE_PLUS", "META_ELITE", "META_STANDARD"})
    value_support = int(value_component >= 8.0)
    row["agreement_count_plus_cs"] = int(meta_support + team_intel_support + value_support + cs_support)
    row["cs_composite_support"] = _derive_cs_composite_support(row)
    row.update(timing_profile)
    row["audit_policy_action"] = safe_str(audit_policy.get("action", ""))
    row["audit_policy_reason"] = safe_str(audit_policy.get("reason", ""))
    row["avoid_in_acca_flag"] = int(
        avoid_in_acca == 1
        or bucket == "SHADOW"
        or safe_int(row.get("cs_fragility_flag"), 0) == 1
        or _market_cs_alignment(row) == "CONFLICT"
        or safe_int(audit_policy.get("force_avoid_in_acca"), 0) == 1
    )
    row["safe_for_small_acca_flag"] = int(
        bucket in {"P1", "P2", "P3"}
        and row["avoid_in_acca_flag"] == 0
        and trap_penalty <= 8.0
        and _market_cs_alignment(row) not in {"CONFLICT", "FRAGILE"}
    )
    row["safe_for_large_acca_flag"] = 0
    row["slip_role_hint"] = "SINGLE" if bucket in {"P1", "P2"} else "SUPPORT_ONLY"
    return row


# ── LOADING ──────────────────────────────────────────────────────────────────

def load_rows_from_file(path: Path, args) -> list[dict]:
    rows: list[dict] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        required_groups = [
            ("fixture_key", COL_FIXTURE_KEY),
            ("market", COL_MARKET),
            ("selection", COL_SELECTION),
            ("model_p", COL_MODEL_P),
            ("odds", COL_ODDS),
        ]

        missing_groups = []
        for label, candidates in required_groups:
            if not any(c in fieldnames for c in candidates):
                missing_groups.append(label)

        if missing_groups:
            raise ValueError(f"{path.name} missing required logical columns: {missing_groups}")

        for raw in reader:
            fixture_key = safe_str(first_present(raw, COL_FIXTURE_KEY))
            if not fixture_key:
                continue

            market = normalize_market(first_present(raw, COL_MARKET))
            selection = normalize_upper(first_present(raw, COL_SELECTION))
            model_p = safe_float(first_present(raw, COL_MODEL_P), None)
            odds = safe_float(first_present(raw, COL_ODDS), None)
            deploy_tier = normalize_upper(first_present(raw, COL_DEPLOY_TIER))
            pool_tier = normalize_upper(first_present(raw, COL_POOL_TIER))
            ftr_priority = normalize_upper(first_present(raw, COL_FTR_PRIORITY))
            keep = parse_bool_keep(first_present(raw, COL_KEEP))
            review_flag = safe_int(first_present(raw, COL_REVIEW_FLAG), 0)

            if model_p is None or odds is None:
                continue

            row = {
                "source_file": path.name,
                "fixture_key": fixture_key,
                "league": safe_str(first_present(raw, COL_LEAGUE)),
                "home": safe_str(first_present(raw, COL_HOME)),
                "away": safe_str(first_present(raw, COL_AWAY)),
                "match_date": safe_str(first_present(raw, COL_MATCH_DATE)),
                "market": market,
                "selection": selection,
                "bookie_pick": normalize_upper(first_present(raw, COL_BOOKIE_PICK)),
                "model_top_pick": normalize_upper(first_present(raw, COL_MODEL_TOP_PICK)),
                "model_p": model_p,
                "odds": odds,
                "bookie_implied": safe_float(first_present(raw, COL_BOOKIE_IMPLIED), None),
                "deploy_tier": deploy_tier,
                "pool_tier": pool_tier,
                "ftr_priority": ftr_priority,
                "keep": keep,
                "review_flag": review_flag,
                "review_reason": safe_str(first_present(raw, COL_REVIEW_REASON)),
                "standard_reporting_bucket": safe_str(first_present(raw, COL_STD_BUCKET)),
                "context_reason_codes": safe_str(first_present(raw, COL_CONTEXT)),
                "product": safe_str(first_present(raw, COL_PRODUCT)),
                "product_profile": safe_str(first_present(raw, COL_PRODUCT_PROFILE)),
                "product_lane": safe_str(first_present(raw, COL_PRODUCT_LANE)),
                "value_edge": safe_float(first_present(raw, COL_VALUE_EDGE), None),
                "value_edge_bps": safe_float(first_present(raw, COL_VALUE_EDGE_BPS), None),
                "value_gap_pct_points": safe_float(first_present(raw, COL_VALUE_GAP), None),
                "value_edge_tier": safe_str(first_present(raw, COL_VALUE_EDGE_TIER)),
                "meta_tier": safe_str(first_present(raw, COL_META_TIER)),
                "meta_overlay_label": safe_str(first_present(raw, COL_META_LABEL)),
                "team_intel_overlay_action": safe_str(first_present(raw, COL_TEAM_INTEL_ACTION)),
                "team_intel_overlay_rank_uplift_bps": safe_float(first_present(raw, COL_TEAM_INTEL_UPLIFT), None),
                "team_intel_overlay_rank_penalty_bps": safe_float(first_present(raw, COL_TEAM_INTEL_PENALTY), None),
                "team_intel_overlay_slip_caution_flag": safe_int(first_present(raw, COL_TEAM_INTEL_CAUTION), 0),
                "team_intel_overlay_avoid_in_acca_flag": safe_int(first_present(raw, COL_TEAM_INTEL_AVOID_IN_ACCA), 0),
                "team_intel_overlay_premium_tag_flag": safe_int(first_present(raw, COL_TEAM_INTEL_PREMIUM_TAG), 0),
                "team_intel_overlay_reason": safe_str(first_present(raw, COL_TEAM_INTEL_REASON)),
                "team_intel_overlay_market_fit_score": safe_float(first_present(raw, COL_TEAM_INTEL_MARKET_FIT), None),
                "team_context_label": safe_str(first_present(raw, COL_TEAM_CONTEXT_LABEL)),
                "team_fixture_interaction": safe_str(first_present(raw, COL_TEAM_CONTEXT_INTERACTION)),
                "team_context_filter_family": safe_str(first_present(raw, COL_TEAM_CONTEXT_FAMILY)),
                "p_home_ge2": safe_float(first_present(raw, COL_P_HOME_GE2), None),
                "p_away_ge2": safe_float(first_present(raw, COL_P_AWAY_GE2), None),
                "phase9b_premium_ge2_test_lane": safe_int(first_present(raw, COL_PHASE9B_GE2_SHADOW), 0),
                "cs_top1_scoreline": safe_str(first_present(raw, COL_CS_TOP1_SCORELINE)),
                "cs_top1_mass": safe_float(first_present(raw, COL_CS_TOP1_MASS), None),
                "cs_top2_scoreline": safe_str(first_present(raw, COL_CS_TOP2_SCORELINE)),
                "cs_top2_mass": safe_float(first_present(raw, COL_CS_TOP2_MASS), None),
                "cs_top3_scoreline": safe_str(first_present(raw, COL_CS_TOP3_SCORELINE)),
                "cs_top3_mass": safe_float(first_present(raw, COL_CS_TOP3_MASS), None),
                "cs_top3_total_mass": safe_float(first_present(raw, COL_CS_TOP3_TOTAL_MASS), None),
                "cs_entropy": safe_float(first_present(raw, COL_CS_ENTROPY), None),
                "cs_concentration_bucket": safe_str(first_present(raw, COL_CS_CONCENTRATION_BUCKET)),
                "cs_banker_flag": safe_int(first_present(raw, COL_CS_BANKER_FLAG), 0),
                "cs_cluster_family": safe_str(first_present(raw, COL_CS_CLUSTER_FAMILY)),
                "cs_aligns_selection_flag": safe_int(first_present(raw, COL_CS_ALIGNS_SELECTION), 0),
                "cs_aligns_top1_selection_flag": safe_int(first_present(raw, COL_CS_ALIGNS_TOP1), 0),
                "cs_aligns_mass_selection_flag": safe_int(first_present(raw, COL_CS_ALIGNS_MASS), 0),
                "cs_alignment_count": safe_int(first_present(raw, COL_CS_ALIGNMENT_COUNT), 0),
                "cs_ftr_pick_side_mass": safe_float(first_present(raw, COL_CS_FTR_PICK_SIDE_MASS), None),
                "cs_ftr_pick_side_margin": safe_float(first_present(raw, COL_CS_FTR_PICK_SIDE_MARGIN), None),
                "cs_ftr_alignment": safe_str(first_present(raw, COL_CS_FTR_ALIGNMENT)),
                "cs_btts_margin": safe_float(first_present(raw, COL_CS_BTTS_MARGIN), None),
                "cs_btts_alignment": safe_str(first_present(raw, COL_CS_BTTS_ALIGNMENT)),
                "cs_ou25_margin": safe_float(first_present(raw, COL_CS_OU25_MARGIN), None),
                "cs_ou25_alignment": safe_str(first_present(raw, COL_CS_OU25_ALIGNMENT)),
                "cs_diffuse_flag": safe_int(first_present(raw, COL_CS_DIFFUSE_FLAG), 0),
                "cs_one_sided_flag": safe_int(first_present(raw, COL_CS_ONE_SIDED_FLAG), 0),
                "cs_draw_family_flag": safe_int(first_present(raw, COL_CS_DRAW_FAMILY_FLAG), 0),
                "cs_fragility_flag": safe_int(first_present(raw, COL_CS_FRAGILITY_FLAG), 0),
                "cs_supported_flag": safe_int(first_present(raw, COL_CS_SUPPORTED_FLAG), 0),
                "cs_fragility_penalty": safe_float(first_present(raw, COL_CS_FRAGILITY_PENALTY), 0.0),
                "cs_banker_bonus": safe_float(first_present(raw, COL_CS_BANKER_BONUS), 0.0),
                "large_acca_cs_safe": safe_int(first_present(raw, COL_LARGE_ACCA_CS_SAFE), 0),
                "deterministic_warn_reason": safe_str(first_present(raw, COL_DET_WARN)),
                "deterministic_veto_reason": safe_str(first_present(raw, COL_DET_VETO)),
                "deploy_veto_reason": safe_str(first_present(raw, COL_DEPLOY_VETO)),
                "learned_veto_reason": safe_str(first_present(raw, COL_LEARNED_VETO)),
                "signal_btts": safe_str(first_present(raw, COL_SIGNAL_BTTS)),
                "fts_max": safe_float(first_present(raw, COL_FTS_MAX), None),
                "btts_yes_ge2_min": safe_float(first_present(raw, COL_BTTS_YES_GE2_MIN), None),
                "lambda_home": safe_float(first_present(raw, COL_LAMBDA_HOME), None),
                "lambda_away": safe_float(first_present(raw, COL_LAMBDA_AWAY), None),
                "p_hw_and_hge2": safe_float(first_present(raw, COL_P_HW_HGE2), None),
                "p_aw_and_age2": safe_float(first_present(raw, COL_P_AW_AGE2), None),
                "ge2_gap": safe_float(first_present(raw, COL_GE2_GAP), None),
                "avg_btts_rate": safe_float(first_present(raw, COL_AVG_BTTS_RATE), None),
                "cs_over25_mass": safe_float(first_present(raw, COL_CS_OVER25_MASS), None),
                "top3_over_count": safe_int(first_present(raw, COL_TOP3_OVER_COUNT), None),
                "draw_warning_token": safe_str(first_present(raw, COL_DRAW_WARNING_TOKEN)),
            }

            row["family"] = derive_family(row)
            row["priority_bucket"] = derive_priority_bucket(row)
            row = compute_slip_leg_score(row)

            if not tier_allowed(row, include_observe=args.include_observe):
                continue

            if args.keep_only and not row["keep"]:
                continue

            if args.block_review_flags and row["review_flag"] == -1:
                continue

            if args.block_regime_flags and is_regime_flagged(row):
                row["blocked_by_regime_policy"] = 1
                continue
            else:
                row["blocked_by_regime_policy"] = 0

            if not family_is_allowed(
                row["family"],
                exclude_ftr_base=args.exclude_ftr_base,
                exclude_ou25=args.exclude_ou25,
            ):
                continue

            if args.min_model_p is not None and row["model_p"] < args.min_model_p:
                continue

            rows.append(row)

    return rows


def load_ranked_rows(paths: Iterable[Path], args) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        rows.extend(load_rows_from_file(path, args))
    rows.sort(key=compute_rank_key)
    return rows


# ── RISK POLICY AFTER LOAD ───────────────────────────────────────────────────

def apply_risk_policy(rows: list[dict], args) -> list[dict]:
    seen_fixtures: set[str] = set()
    family_counts: defaultdict[str, int] = defaultdict(int)
    league_counts: defaultdict[str, int] = defaultdict(int)
    result: list[dict] = []

    for row in rows:
        fx = row["fixture_key"]
        fam = row["family"]
        lg = row["league"]

        if fx in seen_fixtures:
            continue

        if args.max_per_family is not None and family_counts[fam] >= args.max_per_family:
            continue

        if args.max_per_league is not None and league_counts[lg] >= args.max_per_league:
            continue

        seen_fixtures.add(fx)
        family_counts[fam] += 1
        league_counts[lg] += 1
        result.append(row)

    return result


def build_ranked_board(rows: list[dict], args) -> list[dict]:
    board = [dict(rank=i + 1, **r) for i, r in enumerate(rows)]

    for row in board:
        rank = safe_int(row.get("rank"), 9999) or 9999
        rank_band = _rank_band(rank)
        bucket = normalize_upper(row.get("slip_leg_bucket", ""))
        score = safe_float(row.get("slip_leg_score"), 0.0) or 0.0
        std_bucket = normalize_upper(row.get("standard_reporting_bucket", ""))
        market_fit = safe_float(row.get("team_intel_overlay_market_fit_score"), None)
        market_fit = -1.0 if market_fit is None else float(market_fit)
        cs_supported = safe_int(row.get("cs_supported_flag"), 0) or 0
        cs_fragile = safe_int(row.get("cs_fragility_flag"), 0) or 0
        cs_alignment = _market_cs_alignment(row)
        cs_entropy = safe_float(row.get("cs_entropy"), None)
        cs_entropy = 99.0 if cs_entropy is None else float(cs_entropy)
        agreement = safe_int(row.get("agreement_count_plus_cs"), 0) or 0
        audit_action = normalize_upper(row.get("audit_policy_action", ""))
        team_action = normalize_upper(row.get("team_intel_overlay_action", ""))
        trap_penalty = safe_float(row.get("slip_leg_trap_penalty"), 0.0) or 0.0
        base_avoid = safe_int(row.get("avoid_in_acca_flag"), 0) or 0
        safe_small = safe_int(row.get("safe_for_small_acca_flag"), 0) or 0
        timing_risk = safe_float(row.get("timing_risk_modifier"), 0.0) or 0.0
        season_phase = normalize_upper(row.get("season_phase", ""))
        window_month = safe_int(row.get("window_month"), 0) or 0
        market = normalize_market(row.get("market", ""))
        selection = normalize_upper(row.get("selection", ""))
        cs_support_label = normalize_upper(row.get("cs_composite_support", ""))

        lower_p3 = bucket == "P3" and score < 76.0
        review_or_worse = bucket in {"REVIEW", "SHADOW"}
        standard_other_general = std_bucket == "STANDARD_OTHER" and normalize_upper(row.get("team_context_filter_family", "")) == "GENERAL"
        weak_team_fit = market_fit >= 0.0 and market_fit < 0.58
        very_weak_team_fit = market_fit >= 0.0 and market_fit < 0.48
        weak_cs_support = cs_supported == 0 or cs_alignment in {"NEUTRAL", "FRAGILE", "CONFLICT"}
        low_agreement = agreement <= 2
        very_low_agreement = agreement <= 1
        neutral_overlay = team_action in {"", "NEUTRAL"}
        high_entropy = cs_entropy >= 3.15
        rank_down_bucket = audit_action == "RANK_DOWN"
        lower_half_large_acca = rank > 8
        tail_large_acca = rank > 10
        btts_strong_but_general = (
            market == "btts"
            and std_bucket == "STANDARD_BTTS"
            and cs_support_label in {"GENERAL_SUPPORT", "NEUTRAL_SUPPORT"}
        )
        btts_timing_risk = (
            market == "btts"
            and (
                season_phase in {"AUTUMN_DENSE", "EARLY_SEASON", "RUN_IN"}
                or safe_int(row.get("international_break_zone_flag"), 0) == 1
                or safe_int(row.get("champions_league_phase_flag"), 0) == 1
            )
        )
        btts_late_board_needs_more_proof = (
            market == "btts"
            and rank_band in {"RANK_9_10", "RANK_11_PLUS"}
            and (
                neutral_overlay
                or cs_alignment in {"SOFT_ALIGN", "NEUTRAL"}
                or cs_support_label in {"GENERAL_SUPPORT", "NEUTRAL_SUPPORT"}
                or agreement <= 3
            )
        )
        btts_yes_conflict_neutral = (
            market == "btts"
            and selection == "YES"
            and cs_support_label == "CONFLICT_SUPPORT"
            and team_action in {"", "NEUTRAL"}
        )

        if (
            base_avoid == 1
            or (rank_down_bucket and lower_half_large_acca)
            or (review_or_worse and tail_large_acca)
            or (standard_other_general and neutral_overlay and low_agreement and weak_cs_support)
            or (very_weak_team_fit and weak_cs_support and low_agreement)
            or (cs_fragile == 1 and lower_half_large_acca)
            or (btts_late_board_needs_more_proof and rank_band == "RANK_11_PLUS")
            or (btts_timing_risk and rank_band == "RANK_11_PLUS" and selection == "YES" and cs_support_label in {"GENERAL_SUPPORT", "NEUTRAL_SUPPORT"})
            or (btts_yes_conflict_neutral and rank_band in {"RANK_9_10", "RANK_11_PLUS"})
        ):
            row["avoid_in_acca_flag"] = 1

        if (
            btts_yes_conflict_neutral
            and rank_band == "TOP_8"
            and season_phase in {"AUTUMN_DENSE", "WINTER_RESET", "RUN_IN"}
        ):
            row["slip_leg_score"] = round(max(0.0, (safe_float(row.get("slip_leg_score"), 0.0) or 0.0) - 6.0), 4)
            row["slip_leg_audit_policy_component"] = round((safe_float(row.get("slip_leg_audit_policy_component"), 0.0) or 0.0) - 6.0, 4)
            score = safe_float(row.get("slip_leg_score"), 0.0) or 0.0
            bucket = "P1" if score >= 90 else "P2" if score >= 80 else "P3" if score >= 70 else "REVIEW" if score >= 60 else "SHADOW"
            row["slip_leg_bucket"] = bucket

        strict_large_safe = (
            row["avoid_in_acca_flag"] == 0
            and safe_small == 1
            and bucket in {"P1", "P2", "P3"}
            and not review_or_worse
            and not lower_p3
            and trap_penalty <= 4.0
            and safe_int(row.get("large_acca_cs_safe"), 1) == 1
            and (
                safe_int(row.get("cs_diffuse_flag"), 0) == 0
                or (cs_supported == 1 and cs_alignment in {"STRONG_ALIGN", "SOFT_ALIGN"})
            )
            and timing_risk > -2.5
        )
        strict_large_safe = bool(strict_large_safe)

        if strict_large_safe:
            if lower_half_large_acca:
                strict_large_safe = (
                    bucket in {"P1", "P2"}
                    and agreement >= 4
                    and market_fit >= 0.62
                    and cs_supported == 1
                    and cs_alignment in {"STRONG_ALIGN", "SOFT_ALIGN"}
                    and team_action not in {"", "NEUTRAL", "CAUTION"}
                    and audit_action not in {"RANK_DOWN", "DENYLIST"}
                    and not standard_other_general
                    and not high_entropy
                    and timing_risk > -1.5
                )
            else:
                strict_large_safe = (
                    agreement >= 3
                    and market_fit >= 0.55
                    and cs_alignment not in {"CONFLICT", "FRAGILE"}
                    and audit_action not in {"RANK_DOWN", "DENYLIST"}
                    and not (std_bucket == "STANDARD_OTHER" and neutral_overlay and low_agreement)
                    and not (btts_strong_but_general and btts_timing_risk and selection == "YES")
                )

        if strict_large_safe and market == "btts":
            if rank_band == "RANK_9_10":
                strict_large_safe = (
                    agreement >= 4
                    and cs_alignment == "STRONG_ALIGN"
                    and cs_support_label in {"BANKER_SUPPORT", "BTTS_EXPRESSION_VALIDATED"}
                    and team_action == "UPLIFT"
                    and timing_risk > -1.25
                )
            elif rank_band == "RANK_11_PLUS":
                strict_large_safe = False

        row["safe_for_large_acca_flag"] = int(bool(strict_large_safe))

        # V5: separate 10+ fold safety from the 5-8 fold safety regime.
        # The deep-slip policy should be narrower and more targeted, not just
        # a harsher version of the same global rule set.
        has_monster_proof = (
            (cs_alignment == "STRONG_ALIGN" and cs_support_label in {"BANKER_SUPPORT", "BTTS_EXPRESSION_VALIDATED", "FTR_EXPRESSION_VALIDATED", "OU25_EXPRESSION_VALIDATED"})
            or (cs_alignment == "SOFT_ALIGN" and cs_support_label in {"BANKER_SUPPORT", "BTTS_EXPRESSION_VALIDATED", "FTR_EXPRESSION_VALIDATED", "OU25_EXPRESSION_VALIDATED"} and agreement >= 2)
            or team_action == "UPLIFT"
            or agreement >= 2
            or market_fit >= 0.60
        )
        monster_rank_gate_ok = (
            rank <= 8
            or (
                rank > 8
                and has_monster_proof
                and cs_alignment not in {"CONFLICT", "FRAGILE"}
                and not (selection == "YES" and market == "btts" and cs_support_label == "CONFLICT_SUPPORT" and neutral_overlay)
            )
        )

        safe_for_monster_acca = (
            row["avoid_in_acca_flag"] == 0
            and safe_small == 1
            and trap_penalty <= 4.0
            and bucket in {"P1", "P2", "P3"}
            and not review_or_worse
            and monster_rank_gate_ok
            and cs_alignment not in {"CONFLICT", "FRAGILE"}
        )

        if market == "btts":
            if selection == "YES" and cs_support_label == "CONFLICT_SUPPORT" and neutral_overlay:
                safe_for_monster_acca = False
            elif selection == "YES" and rank_band in {"RANK_9_10", "RANK_11_PLUS"}:
                safe_for_monster_acca = bool(
                    safe_for_monster_acca
                    and team_action in {"UPLIFT", "CAUTION"}
                    and agreement >= 2
                    and cs_alignment in {"STRONG_ALIGN", "SOFT_ALIGN"}
                    and cs_support_label in {"BTTS_EXPRESSION_VALIDATED", "BANKER_SUPPORT"}
                    and market_fit >= 0.60
                )
            elif selection == "YES" and rank_band == "TOP_8" and season_phase in {"WINTER_RESET", "RUN_IN", "AUTUMN_DENSE"}:
                if cs_support_label == "CONFLICT_SUPPORT" and neutral_overlay:
                    safe_for_monster_acca = False

        if rank > 8 and market != "btts":
            safe_for_monster_acca = bool(
                safe_for_monster_acca
                and has_monster_proof
                and agreement >= 2
                and market_fit >= 0.56
            )

        monster_timing_penalty = 0.0
        if season_phase == "RUN_IN" and window_month in {4, 5}:
            monster_timing_penalty += 2.5
        if season_phase == "AUTUMN_DENSE" and window_month in {10, 11}:
            monster_timing_penalty += 2.5
        if season_phase == "EARLY_SEASON" and window_month == 9:
            monster_timing_penalty += 2.0
        if season_phase == "WINTER_RESET" and window_month == 2:
            monster_timing_penalty += 2.0

        monster_late_board_penalty = 0.0
        monster_specific_caution = False
        monster_caution_reasons: list[str] = []
        if rank_band == "RANK_11_PLUS":
            monster_late_board_penalty += 4.5
            monster_caution_reasons.append("late_board_rank_11_plus")
        elif rank_band == "RANK_9_10":
            monster_late_board_penalty += 2.0

        if market == "ftr" and rank > 8 and cs_support_label in {"FTR_EXPRESSION_VALIDATED", "BANKER_SUPPORT"}:
            monster_specific_caution = True
            monster_late_board_penalty += 4.0 if rank_band == "RANK_11_PLUS" else 2.5
            monster_caution_reasons.append("late_board_ftr_expression")

        if market == "ou25" and selection == "OVER25" and rank > 8:
            monster_specific_caution = True
            monster_late_board_penalty += 3.5 if rank_band == "RANK_11_PLUS" else 2.0
            monster_caution_reasons.append("late_board_ou25_over25")

        if market == "btts" and selection == "YES" and rank > 8 and team_action == "NEUTRAL":
            monster_specific_caution = True
            monster_late_board_penalty += 2.5
            monster_caution_reasons.append("late_board_btts_yes_neutral")

        if season_phase == "RUN_IN" and window_month in {4, 5}:
            monster_caution_reasons.append("timing_run_in_4_5")
        if season_phase == "AUTUMN_DENSE" and window_month in {10, 11}:
            monster_caution_reasons.append("timing_autumn_dense_10_11")
        if season_phase == "EARLY_SEASON" and window_month == 9:
            monster_caution_reasons.append("timing_early_season_9")
        if season_phase == "WINTER_RESET" and window_month == 2:
            monster_caution_reasons.append("timing_winter_reset_2")

        avoid_in_monster_acca = int(
            row["avoid_in_acca_flag"] == 1
            or cs_fragile == 1
            or (
                market == "btts"
                and selection == "YES"
                and cs_support_label == "CONFLICT_SUPPORT"
                and neutral_overlay
                and rank > 8
            )
        )

        monster_candidate_eligible = int(
            avoid_in_monster_acca == 0
            and cs_fragile == 0
            and not (market == "btts" and selection == "YES" and cs_support_label == "CONFLICT_SUPPORT" and neutral_overlay and rank > 8)
        )
        monster_candidate_score = (
            (safe_float(row.get("slip_leg_score"), 0.0) or 0.0)
            + max(0.0, agreement - 1) * 3.0
            + max(0.0, market_fit) * 10.0
            + (6.0 if cs_alignment == "STRONG_ALIGN" else 3.0 if cs_alignment == "SOFT_ALIGN" else 0.0)
            + (4.0 if team_action == "UPLIFT" else 1.0 if team_action == "CAUTION" else 0.0)
            + (3.0 if cs_support_label in {"BANKER_SUPPORT", "BTTS_EXPRESSION_VALIDATED", "FTR_EXPRESSION_VALIDATED", "OU25_EXPRESSION_VALIDATED"} else 0.0)
            + (safe_float(row.get("timing_risk_modifier"), 0.0) or 0.0)
            - monster_timing_penalty
            - monster_late_board_penalty
        )
        if market == "btts" and selection == "YES" and cs_support_label in {"GENERAL_SUPPORT", "CONFLICT_SUPPORT"}:
            monster_candidate_score -= 5.0
        if monster_specific_caution:
            monster_candidate_score -= 2.0

        extension_policy_notes: list[str] = []
        strong_extension_support = (
            agreement >= 3
            and market_fit >= 0.60
            and cs_alignment in {"STRONG_ALIGN", "SOFT_ALIGN"}
            and cs_support_label in {
                "BANKER_SUPPORT",
                "BTTS_EXPRESSION_VALIDATED",
                "FTR_EXPRESSION_VALIDATED",
                "OU25_EXPRESSION_VALIDATED",
            }
            and team_action in {"UPLIFT", "CAUTION"}
        )
        weak_extension_support = (
            agreement <= 2
            or market_fit < 0.58
            or cs_alignment in {"NEUTRAL", "FRAGILE", "CONFLICT"}
            or cs_support_label in {"GENERAL_SUPPORT", "NEUTRAL_SUPPORT", "CONFLICT_SUPPORT", ""}
            or team_action in {"", "NEUTRAL"}
        )
        bad_extension_timing = monster_timing_penalty > 0.0 or timing_risk <= -1.0

        validated_support_labels = {
            "BANKER_SUPPORT",
            "BTTS_EXPRESSION_VALIDATED",
            "FTR_EXPRESSION_VALIDATED",
            "OU25_EXPRESSION_VALIDATED",
        }
        strong_btts_extension_shape = (
            cs_alignment in {"STRONG_ALIGN", "SOFT_ALIGN"}
            and cs_support_label in {"BTTS_EXPRESSION_VALIDATED", "BANKER_SUPPORT"}
            and agreement >= 3
            and market_fit >= 0.68
            and not bad_extension_timing
        )
        weak_ftr_extension_shape = (
            agreement <= 2
            or market_fit < 0.60
            or team_action in {"", "NEUTRAL"}
            or cs_support_label not in validated_support_labels
            or bad_extension_timing
        )
        weak_over25_extension_shape = (
            market_fit < 0.60
            or cs_alignment not in {"STRONG_ALIGN", "SOFT_ALIGN"}
            or cs_support_label not in {"OU25_EXPRESSION_VALIDATED", "BANKER_SUPPORT"}
            or agreement <= 2
            or bad_extension_timing
        )

        if rank_band == "RANK_9_10":
            if market == "btts" and selection == "YES":
                if strong_extension_support:
                    monster_candidate_score += 3.0
                    extension_policy_notes.append("ext_9_10_btts_yes_supported")
                elif weak_extension_support or bad_extension_timing:
                    monster_candidate_score -= 2.0
                    extension_policy_notes.append("ext_9_10_btts_yes_needs_support")
            elif market == "ftr":
                if strong_extension_support:
                    monster_candidate_score += 1.5
                    extension_policy_notes.append("ext_9_10_ftr_supported")
                elif weak_extension_support or bad_extension_timing:
                    monster_candidate_score -= 3.0
                    extension_policy_notes.append("ext_9_10_ftr_weak_support")
            elif market == "ou25" and selection == "OVER25":
                if strong_extension_support:
                    monster_candidate_score += 1.0
                    extension_policy_notes.append("ext_9_10_over25_supported")
                elif weak_extension_support or bad_extension_timing:
                    monster_candidate_score -= 2.5
                    extension_policy_notes.append("ext_9_10_over25_weak_support")

        elif rank_band == "RANK_11_PLUS":
            if market == "btts" and selection == "YES":
                if strong_btts_extension_shape:
                    monster_candidate_score += 5.0
                    extension_policy_notes.append("ext_11_plus_btts_yes_template_preferred")
                else:
                    monster_candidate_score -= 3.0
                    extension_policy_notes.append("ext_11_plus_btts_yes_template_not_met")
            elif market == "ftr":
                monster_candidate_score -= 4.0
                extension_policy_notes.append("ext_11_plus_ftr_base_penalty")
                if weak_ftr_extension_shape:
                    monster_candidate_score -= 4.0
                    extension_policy_notes.append("ext_11_plus_ftr_template_penalty")
            elif market == "ou25" and selection == "OVER25":
                monster_candidate_score -= 2.5
                extension_policy_notes.append("ext_11_plus_over25_base_penalty")
                if weak_over25_extension_shape:
                    monster_candidate_score -= 2.5
                    extension_policy_notes.append("ext_11_plus_over25_template_penalty")

        monster_caution_flag = int(bool(monster_caution_reasons) and avoid_in_monster_acca == 0)
        if monster_caution_flag == 1:
            # Prefer non-caution monster candidates in close decisions without
            # excluding caution rows from the admissible pool.
            monster_candidate_score -= 8.0

        monster_extension_pool_a_eligible = int(monster_candidate_eligible == 1)
        monster_deep_tail_eligible = int(
            monster_candidate_eligible == 1
            and (
                (
                    market == "btts"
                    and selection == "YES"
                    and cs_support_label in {"BTTS_EXPRESSION_VALIDATED", "BANKER_SUPPORT"}
                    and agreement >= 3
                    and market_fit >= 0.64
                    and not bad_extension_timing
                )
                or (
                    market == "ftr"
                    and cs_support_label in validated_support_labels
                    and agreement >= 2
                    and market_fit >= 0.58
                    and not neutral_overlay
                    and not bad_extension_timing
                )
                or (
                    market == "ou25"
                    and selection == "OVER25"
                    and cs_support_label in {"OU25_EXPRESSION_VALIDATED", "BANKER_SUPPORT"}
                    and agreement >= 2
                    and market_fit >= 0.56
                    and not bad_extension_timing
                )
            )
        )
        monster_deep_tail_soft_eligible = int(
            monster_candidate_eligible == 1
            and (
                (
                    market == "btts"
                    and selection == "YES"
                    and cs_support_label in {"BTTS_EXPRESSION_VALIDATED", "BANKER_SUPPORT"}
                    and agreement >= 2
                    and market_fit >= 0.60
                    and not bad_extension_timing
                )
                or (
                    market == "ftr"
                    and cs_support_label in validated_support_labels
                    and agreement >= 2
                    and market_fit >= 0.56
                    and not neutral_overlay
                    and not bad_extension_timing
                )
                or (
                    market == "ou25"
                    and selection == "OVER25"
                    and cs_support_label in {"OU25_EXPRESSION_VALIDATED", "BANKER_SUPPORT"}
                    and agreement >= 2
                    and market_fit >= 0.54
                    and not bad_extension_timing
                )
            )
        )

        row["safe_for_monster_acca_flag"] = int(bool(safe_for_monster_acca))
        row["avoid_in_monster_acca_flag"] = avoid_in_monster_acca
        row["monster_caution_flag"] = monster_caution_flag
        row["monster_caution_reason"] = "|".join(dict.fromkeys(monster_caution_reasons))
        row["monster_extension_policy_notes"] = "|".join(dict.fromkeys(extension_policy_notes))
        row["monster_extension_pool_a_eligible"] = monster_extension_pool_a_eligible
        row["monster_deep_tail_eligible"] = monster_deep_tail_eligible
        row["monster_deep_tail_soft_eligible"] = monster_deep_tail_soft_eligible
        row["monster_candidate_eligible"] = monster_candidate_eligible
        row["monster_candidate_score"] = round(monster_candidate_score, 4)
        row["rank_band"] = rank_band

        if row["avoid_in_acca_flag"] == 1 or cs_fragile == 1:
            row["slip_role_hint"] = "FRAGILE"
        elif row["safe_for_monster_acca_flag"] == 1:
            row["slip_role_hint"] = "LARGE_ACCA_SAFE"
        elif row["safe_for_large_acca_flag"] == 1:
            row["slip_role_hint"] = "LARGE_ACCA_SAFE"
        elif safe_small == 1:
            row["slip_role_hint"] = "SAFE_SMALL_ACCA"
        elif bucket in {"P1", "P2"}:
            row["slip_role_hint"] = "SINGLE"
        else:
            row["slip_role_hint"] = "SUPPORT_ONLY"

    if args.min_rank is not None and args.min_rank > 1:
        board = [r for r in board if r["rank"] >= args.min_rank]

    if args.max_rank is not None:
        board = [r for r in board if r["rank"] <= args.max_rank]

    return board


# ── SLIP BUILDERS ────────────────────────────────────────────────────────────

def _combo_respects_correlation_caps(combo: list[dict], combo_size: int, args, deep_tail_relaxed: bool = False) -> bool:
    league_counts: defaultdict[str, int] = defaultdict(int)
    market_counts: defaultdict[str, int] = defaultdict(int)
    context_counts: defaultdict[str, int] = defaultdict(int)
    eff_league_cap = args.max_slip_per_league
    eff_market_cap = args.max_slip_per_market
    eff_context_cap = args.max_slip_per_context_family
    if combo_size >= 10:
        if eff_league_cap is not None:
            eff_league_cap = max(eff_league_cap, 4)
        if eff_market_cap is not None:
            eff_market_cap = max(eff_market_cap, 5)
        if eff_context_cap is not None:
            eff_context_cap = max(eff_context_cap, 5)
        if deep_tail_relaxed:
            if eff_league_cap is not None:
                eff_league_cap = max(eff_league_cap, 5)
            if eff_market_cap is not None:
                eff_market_cap = max(eff_market_cap, 6)
            if eff_context_cap is not None:
                eff_context_cap = max(eff_context_cap, 6)

    for row in combo:
        lg = safe_str(row.get("league", ""))
        mk = normalize_market(row.get("market", ""))
        ctx = normalize_upper(row.get("team_context_filter_family", "")) or "GENERAL"
        league_counts[lg] += 1
        market_counts[mk] += 1
        context_counts[ctx] += 1

        if eff_league_cap is not None and league_counts[lg] > eff_league_cap:
            return False
        if eff_market_cap is not None and market_counts[mk] > eff_market_cap:
            return False
        if eff_context_cap is not None and context_counts[ctx] > eff_context_cap:
            return False

    if combo_size >= 10:
        if any(safe_int(r.get("monster_candidate_eligible"), 0) != 1 for r in combo):
            return False
    elif combo_size >= args.large_acca_threshold:
        if any(safe_int(r.get("safe_for_large_acca_flag"), 0) != 1 for r in combo):
            return False
    return True


def build_fixed_acca(rows: list[dict], k: int, args, monster_only: bool = False) -> list[dict]:
    pool = [r for r in rows if (not monster_only or r["family"] in MONSTER_ACCA_FAMILIES)]
    if k >= 10:
        pool = [r for r in pool if safe_int(r.get("monster_candidate_eligible"), 0) == 1]
        pool.sort(key=lambda r: (-safe_float(r.get("monster_candidate_score"), 0.0), r.get("rank", 9999)))
    legs: list[dict] = []

    if k >= 10:
        core_target = min(k, 8)
        extension_target = min(max(k - core_target, 0), 2)
        deep_tail_target = max(k - core_target - extension_target, 0)

        def _consume_stage(stage_pool: list[dict], needed: int, deep_tail_relaxed: bool = False) -> bool:
            nonlocal legs
            if needed <= 0:
                return True
            for row in stage_pool:
                if any(safe_str(existing.get("fixture_key", "")) == safe_str(row.get("fixture_key", "")) for existing in legs):
                    continue
                candidate = legs + [row]
                if not _combo_respects_correlation_caps(candidate, k, args, deep_tail_relaxed=deep_tail_relaxed):
                    continue
                legs = candidate
                if len(legs) >= needed:
                    break
            return len(legs) >= needed

        core_pool = pool
        extension_pool = [r for r in pool if safe_int(r.get("monster_extension_pool_a_eligible"), 0) == 1]
        deep_tail_pool = [r for r in pool if safe_int(r.get("monster_deep_tail_eligible"), 0) == 1]
        deep_tail_soft_pool = [
            r for r in pool
            if safe_int(r.get("monster_deep_tail_soft_eligible"), 0) == 1
            and safe_str(r.get("fixture_key", "")) not in {safe_str(d.get("fixture_key", "")) for d in deep_tail_pool}
        ]
        deep_tail_with_fallback_pool = deep_tail_pool + deep_tail_soft_pool + [
            r for r in extension_pool
            if safe_str(r.get("fixture_key", "")) not in {
                safe_str(d.get("fixture_key", "")) for d in (deep_tail_pool + deep_tail_soft_pool)
            }
        ]

        if not _consume_stage(core_pool, core_target):
            return []
        if not _consume_stage(extension_pool, core_target + extension_target):
            return []
        if not _consume_stage(deep_tail_with_fallback_pool, core_target + extension_target + deep_tail_target, deep_tail_relaxed=True):
            return []
    else:
        for row in pool:
            candidate = legs + [row]
            if not _combo_respects_correlation_caps(candidate, k, args):
                continue
            legs = candidate
            if len(legs) >= k:
                break
    if len(legs) < k:
        return []

    odds_product = math.prod(r["odds"] for r in legs)
    avg_model_p = sum(r["model_p"] for r in legs) / len(legs)
    families = "|".join(r["family"] for r in legs)
    leagues = "|".join(r["league"] for r in legs)

    out = []
    for i, leg in enumerate(legs, start=1):
        out.append({
            "slip_id": f"ACCA_{k}__SLIP",
            "slip_type": f"ACCA_{k}",
            "legs": k,
            "slip_odds": round(odds_product, 4),
            "avg_model_p": round(avg_model_p, 4),
            "family_mix": families,
            "league_mix": leagues,
            "leg_index": i,
            "leg_rank": leg["rank"],
            "leg_fixture": leg["fixture_key"],
            "leg_league": leg["league"],
            "leg_home": leg["home"],
            "leg_away": leg["away"],
            "leg_date": leg["match_date"],
            "leg_market": leg["market"],
            "leg_selection": leg["selection"],
            "leg_family": leg["family"],
            "leg_std_bucket": leg["standard_reporting_bucket"],
            "leg_model_p": leg["model_p"],
            "leg_odds": leg["odds"],
            "leg_deploy_tier": leg["deploy_tier"],
            "leg_source_file": leg["source_file"],
        })
    return out


def build_combination_slip(
    rows: list[dict],
    combo_size: int,
    slip_type: str,
    max_legs: int,
    args,
    monster_only: bool = False,
) -> list[dict]:
    pool = [r for r in rows if (not monster_only or r["family"] in MONSTER_ACCA_FAMILIES)]
    if combo_size >= 10:
        pool = [r for r in pool if safe_int(r.get("monster_candidate_eligible"), 0) == 1]
        pool.sort(key=lambda r: (-safe_float(r.get("monster_candidate_score"), 0.0), r.get("rank", 9999)))
    pool = pool[:max_legs]
    if len(pool) < combo_size:
        return []

    lines = []
    for i, combo in enumerate(itertools.combinations(pool, combo_size), start=1):
        combo_list = list(combo)
        if not _combo_respects_correlation_caps(combo_list, combo_size, args):
            continue
        odds_product = math.prod(r["odds"] for r in combo)
        avg_model_p = sum(r["model_p"] for r in combo) / len(combo)

        lines.append({
            "line_id": f"{slip_type}__L{i:05d}",
            "slip_type": slip_type,
            "combo_size": combo_size,
            "line_odds": round(odds_product, 4),
            "avg_model_p": round(avg_model_p, 4),
            "family_mix": "|".join(r["family"] for r in combo),
            "fixtures": "|".join(r["fixture_key"] for r in combo),
            "legs": "|".join(
                f"{r['home']} v {r['away']} [{r['market']} {r['selection']}] [{r['family']}] [{r['deploy_tier']}]"
                for r in combo
            ),
        })
    return lines


def build_heinz(rows: list[dict], args) -> list[dict]:
    pool = rows[:6]
    lines = []
    for size in range(2, 7):
        lines.extend(build_combination_slip(pool, size, f"HEINZ_{size}FOLD", max_legs=6, args=args))
    return lines


def build_yankee(rows: list[dict], args) -> list[dict]:
    pool = rows[:4]
    lines = []
    for size in range(2, 5):
        lines.extend(build_combination_slip(pool, size, f"YANKEE_{size}FOLD", max_legs=4, args=args))
    return lines


# ── SUMMARIES ────────────────────────────────────────────────────────────────

def build_family_summary(board: list[dict]) -> list[dict]:
    fams = sorted({r["family"] for r in board})
    out = []
    for fam in fams:
        sub = [r for r in board if r["family"] == fam]
        if not sub:
            continue
        out.append({
            "family": fam,
            "rows": len(sub),
            "avg_model_p": round(sum(r["model_p"] for r in sub) / len(sub), 4),
            "avg_odds": round(sum(r["odds"] for r in sub) / len(sub), 4),
            "best_rank": min(r["rank"] for r in sub),
            "worst_rank": max(r["rank"] for r in sub),
            "tier_mix": "|".join(sorted({r["deploy_tier"] for r in sub})),
            "std_bucket_mix": "|".join(sorted({r["standard_reporting_bucket"] for r in sub})),
            "priority_bucket_mix": "|".join(sorted({safe_str(r.get("priority_bucket", "")) for r in sub if safe_str(r.get("priority_bucket", ""))})),
            "shadow_rows": sum(1 for r in sub if "SHADOW" in safe_str(r.get("family", ""))),
        })
    return out


def build_singles(board: list[dict]) -> list[dict]:
    out = []
    for r in board:
        out.append({
            "rank": r["rank"],
            "fixture_key": r["fixture_key"],
            "fixture": f"{r['home']} v {r['away']}",
            "date": r["match_date"],
            "league": r["league"],
            "market": r["market"],
            "selection": r["selection"],
            "family": r["family"],
            "priority_bucket": r.get("priority_bucket", ""),
            "blocked_by_regime_policy": r.get("blocked_by_regime_policy", 0),
            "is_shadow": int("SHADOW" in safe_str(r.get("family", ""))),
            "family_priority": FAMILY_PRIORITY.get(r["family"], 999),
            "deploy_tier": r["deploy_tier"],
            "standard_reporting_bucket": r["standard_reporting_bucket"],
            "profit_first_keep": int(r["keep"]),
            "profit_first_review_flag": r["review_flag"],
            "profit_first_review_reason": r["review_reason"],
            "context_reason_codes": r["context_reason_codes"],
            "product_profile": r["product_profile"],
            "product_lane": r["product_lane"],
            "team_context_label": r.get("team_context_label", ""),
            "team_fixture_interaction": r.get("team_fixture_interaction", ""),
            "team_context_filter_family": r.get("team_context_filter_family", ""),
            "team_intel_overlay_action": r.get("team_intel_overlay_action", ""),
            "team_intel_overlay_reason": r.get("team_intel_overlay_reason", ""),
            "team_intel_overlay_slip_caution_flag": r.get("team_intel_overlay_slip_caution_flag", 0),
            "team_intel_overlay_avoid_in_acca_flag": r.get("team_intel_overlay_avoid_in_acca_flag", 0),
            "agreement_count_plus_cs": r.get("agreement_count_plus_cs", 0),
            "cs_composite_support": r.get("cs_composite_support", ""),
            "slip_role_hint": r.get("slip_role_hint", ""),
            "audit_policy_action": r.get("audit_policy_action", ""),
            "audit_policy_reason": r.get("audit_policy_reason", ""),
            "safe_for_large_acca_flag": r.get("safe_for_large_acca_flag", 0),
            "avoid_in_acca_flag": r.get("avoid_in_acca_flag", 0),
            "value_edge_bps": r.get("value_edge_bps"),
            "value_edge_tier": r.get("value_edge_tier", ""),
            "meta_tier": r.get("meta_tier", ""),
            "meta_overlay_label": r.get("meta_overlay_label", ""),
            "slip_leg_score": r.get("slip_leg_score", 0.0),
            "slip_leg_bucket": r.get("slip_leg_bucket", ""),
            "slip_leg_base_quality": r.get("slip_leg_base_quality", 0.0),
            "slip_leg_value_component": r.get("slip_leg_value_component", 0.0),
            "slip_leg_conviction_component": r.get("slip_leg_conviction_component", 0.0),
            "slip_leg_team_intel_component": r.get("slip_leg_team_intel_component", 0.0),
            "slip_leg_correct_score_component": r.get("slip_leg_correct_score_component", 0.0),
            "slip_leg_audit_policy_component": r.get("slip_leg_audit_policy_component", 0.0),
            "slip_leg_market_trust_component": r.get("slip_leg_market_trust_component", 0.0),
            "slip_leg_odds_shape_component": r.get("slip_leg_odds_shape_component", 0.0),
            "slip_leg_modern_confidence_component": r.get("slip_leg_modern_confidence_component", 0.0),
            "slip_leg_trap_penalty": r.get("slip_leg_trap_penalty", 0.0),
            "cs_top1_scoreline": r.get("cs_top1_scoreline", ""),
            "cs_top1_mass": r.get("cs_top1_mass"),
            "cs_top2_scoreline": r.get("cs_top2_scoreline", ""),
            "cs_top2_mass": r.get("cs_top2_mass"),
            "cs_top3_scoreline": r.get("cs_top3_scoreline", ""),
            "cs_top3_mass": r.get("cs_top3_mass"),
            "cs_top3_total_mass": r.get("cs_top3_total_mass"),
            "cs_entropy": r.get("cs_entropy"),
            "cs_concentration_bucket": r.get("cs_concentration_bucket", ""),
            "cs_banker_flag": r.get("cs_banker_flag", 0),
            "cs_cluster_family": r.get("cs_cluster_family", ""),
            "cs_aligns_selection_flag": r.get("cs_aligns_selection_flag", 0),
            "cs_aligns_top1_selection_flag": r.get("cs_aligns_top1_selection_flag", 0),
            "cs_aligns_mass_selection_flag": r.get("cs_aligns_mass_selection_flag", 0),
            "cs_alignment_count": r.get("cs_alignment_count", 0),
            "cs_ftr_pick_side_mass": r.get("cs_ftr_pick_side_mass"),
            "cs_ftr_pick_side_margin": r.get("cs_ftr_pick_side_margin"),
            "cs_ftr_alignment": r.get("cs_ftr_alignment", ""),
            "cs_btts_margin": r.get("cs_btts_margin"),
            "cs_btts_alignment": r.get("cs_btts_alignment", ""),
            "cs_ou25_margin": r.get("cs_ou25_margin"),
            "cs_ou25_alignment": r.get("cs_ou25_alignment", ""),
            "cs_diffuse_flag": r.get("cs_diffuse_flag", 0),
            "cs_one_sided_flag": r.get("cs_one_sided_flag", 0),
            "cs_draw_family_flag": r.get("cs_draw_family_flag", 0),
            "cs_fragility_flag": r.get("cs_fragility_flag", 0),
            "cs_supported_flag": r.get("cs_supported_flag", 0),
            "cs_fragility_penalty": r.get("cs_fragility_penalty", 0.0),
            "cs_banker_bonus": r.get("cs_banker_bonus", 0.0),
            "large_acca_cs_safe": r.get("large_acca_cs_safe", 0),
            "model_p": r["model_p"],
            "odds": r["odds"],
            "source_file": r["source_file"],
        })
    return out


# ── MARKET HELPERS ──────────────────────────────────────────────────────────


def filter_board_by_market(board: list[dict], market: str) -> list[dict]:
    market_norm = safe_str(market).lower()
    return [r for r in board if safe_str(r.get("market", "")).lower() == market_norm]


def add_market_rank(board: list[dict], market: str) -> list[dict]:
    sub = filter_board_by_market(board, market)
    out: list[dict] = []
    for i, row in enumerate(sub, start=1):
        new_row = dict(row)
        new_row["market_rank"] = i
        out.append(new_row)
    return out


def filter_rows_for_acca_market(
    board: list[dict],
    market_name: str,
    monster_only_accas: bool,
    monster_ftr_no_base: bool,
) -> list[dict]:
    if market_name == "mixed":
        rows = list(board)
        if monster_only_accas:
            rows = [r for r in rows if r.get("family") in MONSTER_ACCA_FAMILIES]
        if monster_ftr_no_base:
            rows = [
                r for r in rows
                if not (
                    safe_str(r.get("market", "")).lower() == "ftr"
                    and safe_str(r.get("family", "")) == "FTR_BASE"
                )
            ]
        return rows

    rows = filter_board_by_market(board, market_name)
    if market_name == "ftr" and monster_ftr_no_base:
        rows = [r for r in rows if safe_str(r.get("family", "")) != "FTR_BASE"]
    return rows



def write_market_ranked_boards(board: list[dict], outdir: Path, tag: str) -> dict[str, Path]:
    market_map = {
        "ftr": add_market_rank(board, "ftr"),
        "btts": add_market_rank(board, "btts"),
        "ou25": add_market_rank(board, "ou25"),
    }
    out_paths: dict[str, Path] = {}
    for market_name, sub_board in market_map.items():
        if not sub_board:
            continue
        out_path = outdir / f"ranked_board_{market_name}_{tag}.csv"
        write_csv(sub_board, out_path)
        out_paths[market_name] = out_path
    return out_paths



def write_market_fixed_accas(
    board: list[dict],
    outdir: Path,
    tag: str,
    args,
    monster_only_accas: bool,
    monster_ftr_no_base: bool,
    max_acca_size: int | None,
    selected_market: str,
) -> list[Path]:
    written: list[Path] = []

    market_boards = {
        "ftr": filter_rows_for_acca_market(board, "ftr", monster_only_accas, monster_ftr_no_base),
        "btts": filter_rows_for_acca_market(board, "btts", monster_only_accas, monster_ftr_no_base),
        "mixed": filter_rows_for_acca_market(board, "mixed", monster_only_accas, monster_ftr_no_base),
    }

    if selected_market in {"ftr", "btts", "ou25", "mixed"}:
        if selected_market == "ou25":
            market_boards = {
                "ou25": filter_rows_for_acca_market(board, "ou25", monster_only_accas, monster_ftr_no_base)
            }
        else:
            market_boards = {selected_market: market_boards.get(selected_market, [])}

    for market_name, sub_board in market_boards.items():
        if not sub_board:
            continue
        for k in sorted(FIXED_ACCA_SIZES, reverse=True):
            if max_acca_size is not None and k > max_acca_size:
                continue
            slip_rows = build_fixed_acca(
                sub_board,
                k,
                args=args,
                monster_only=False,
            )
            if slip_rows:
                out_path = outdir / f"slips_acca_{market_name}_{k:02d}_{tag}.csv"
                write_csv(slip_rows, out_path)
                written.append(out_path)
    return written


# ── WRITER ───────────────────────────────────────────────────────────────────

def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {len(rows):>5} rows → {path}")


# ── CLI ──────────────────────────────────────────────────────────────────────
# CLI policy:
# - By default, ELITE and STANDARD routed rows are treated as valid picks.
# - profit_first_keep is optional, not mandatory.
# - FTR_BASE and OU25 are included by default unless explicitly excluded.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thin slip formatter over deploy_rulebook outputs")

    parser.add_argument("--input", default="", help="Single deploy CSV input")
    parser.add_argument("--inputs", nargs="*", default=[], help="Multiple deploy CSV inputs")
    parser.add_argument("--outdir", default="./slips", help="Output directory")

    parser.add_argument("--keep-only", action="store_true", help="Optional: keep only profit_first_keep == 1 rows")
    parser.add_argument("--include-observe", action="store_true", help="Allow OBSERVE rows")
    parser.add_argument("--exclude-ftr-base", action="store_true", help="Exclude STANDARD_FTR_BASE style rows")
    parser.add_argument("--exclude-ou25", action="store_true", help="Exclude OU25 premium rows")
    parser.add_argument("--block-review-flags", action="store_true", help="Exclude rows with profit_first_review_flag = -1")
    parser.set_defaults(keep_only=False)
    parser.add_argument("--block-regime-flags", action="store_true", help="Exclude regime/demotion/blocked rows using reason fields")
    parser.add_argument("--monster-only-accas", action="store_true", help="Fixed accas only use monster families")
    parser.add_argument("--monster-ftr-no-base", action="store_true", help="Exclude FTR_BASE rows from FTR and mixed acca products")
    parser.add_argument("--market", choices=["ftr", "btts", "ou25", "mixed"], default="mixed", help="Limit acca product generation to a specific market board")
    parser.add_argument("--max-acca-size", type=int, default=None, help="Optional maximum fixed acca size to write")

    parser.add_argument("--min-rank", type=int, default=1, help="Keep ranks >= this")
    parser.add_argument("--max-rank", type=int, default=None, help="Keep ranks <= this")
    parser.add_argument("--min-model-p", type=float, default=None, help="Optional minimum model probability")
    parser.add_argument("--max-per-family", type=int, default=None, help="Cap per family")
    parser.add_argument("--max-per-league", type=int, default=None, help="Cap per league")
    parser.add_argument("--max-slip-per-league", type=int, default=2, help="Correlation cap per league inside slip products")
    parser.add_argument("--max-slip-per-market", type=int, default=3, help="Correlation cap per market inside slip products")
    parser.add_argument("--max-slip-per-context-family", type=int, default=2, help="Correlation cap per team-context family inside slip products")
    parser.add_argument("--large-acca-threshold", type=int, default=5, help="Slip size from which safe_for_large_acca_flag becomes mandatory")

    return parser.parse_args()


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    input_paths: list[Path] = []
    if args.input:
        input_paths.append(Path(args.input))
    input_paths.extend(Path(x) for x in args.inputs)

    input_paths = [p for p in input_paths if str(p).strip()]
    if not input_paths:
        print("No input files supplied.")
        sys.exit(1)

    missing = [str(p) for p in input_paths if not p.exists()]
    if missing:
        print("Missing input files:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\nslip_formatter.py")
    print("  inputs:")
    for p in input_paths:
        print(f"    - {p}")
    print(f"  outdir: {outdir}")
    print(
        f"  flags: keep_only={args.keep_only} "
        f"include_observe={args.include_observe} "
        f"exclude_ftr_base={args.exclude_ftr_base} "
        f"exclude_ou25={args.exclude_ou25} "
        f"block_review_flags={args.block_review_flags} "
        f"block_regime_flags={args.block_regime_flags} "
        f"monster_only_accas={args.monster_only_accas} "
        f"monster_ftr_no_base={args.monster_ftr_no_base} "
        f"market={args.market} "
        f"max_acca_size={args.max_acca_size} "
        f"max_slip_per_league={args.max_slip_per_league} "
        f"max_slip_per_market={args.max_slip_per_market} "
        f"max_slip_per_context_family={args.max_slip_per_context_family} "
        f"large_acca_threshold={args.large_acca_threshold}"
    )
    print()

    raw_rows = load_ranked_rows(input_paths, args)
    raw_family_counts: defaultdict[str, int] = defaultdict(int)
    raw_market_counts: defaultdict[str, int] = defaultdict(int)
    raw_priority_counts: defaultdict[str, int] = defaultdict(int)
    raw_shadow_count = 0
    for r in raw_rows:
        raw_family_counts[r["family"]] += 1
        raw_market_counts[r["market"]] += 1
        raw_priority_counts[safe_str(r.get("priority_bucket", "UNPRIORITISED"))] += 1
        raw_shadow_count += int("SHADOW" in safe_str(r.get("family", "")))
    print(f"  loaded {len(raw_rows)} routed candidate rows")
    print(f"  raw markets: {dict(sorted(raw_market_counts.items()))}")
    print(f"  raw families: {dict(sorted(raw_family_counts.items()))}")
    print(f"  raw priorities: {dict(sorted(raw_priority_counts.items()))}")
    print(f"  raw shadow rows: {raw_shadow_count}")

    rows = apply_risk_policy(raw_rows, args)
    print(f"  {len(rows)} rows after risk policy")

    if not rows:
        print("  no qualifying rows — exiting")
        sys.exit(0)

    board = build_ranked_board(rows, args)
    if not board:
        print("  no rows left after rank filters — exiting")
        sys.exit(0)

    ranked_board_path = outdir / f"ranked_board_{tag}.csv"
    family_summary_path = outdir / f"ranked_board_family_summary_{tag}.csv"
    singles_path = outdir / f"slips_singles_{tag}.csv"
    doubles_path = outdir / f"slips_doubles_{tag}.csv"
    trebles_path = outdir / f"slips_trebles_{tag}.csv"
    yankee_path = outdir / f"slips_yankee_{tag}.csv"
    heinz_path = outdir / f"slips_heinz_{tag}.csv"

    write_csv(board, ranked_board_path)
    market_ranked_board_paths = write_market_ranked_boards(board, outdir, tag)
    write_csv(build_family_summary(board), family_summary_path)
    write_csv(build_singles(board), singles_path)

    doubles = build_combination_slip(
        board,
        2,
        "DOUBLES",
        max_legs=min(len(board), DOUBLES_MAX_LEGS),
        args=args,
        monster_only=False,
    )
    write_csv(doubles, doubles_path)

    trebles = build_combination_slip(
        board,
        3,
        "TREBLES",
        max_legs=min(len(board), TREBLES_MAX_LEGS),
        args=args,
        monster_only=False,
    )
    write_csv(trebles, trebles_path)

    yankee = build_yankee(board, args)
    write_csv(yankee, yankee_path)

    heinz = build_heinz(board, args)
    write_csv(heinz, heinz_path)

    primary_acca_board = filter_rows_for_acca_market(
        board,
        args.market,
        args.monster_only_accas,
        args.monster_ftr_no_base,
    )

    if args.market == "mixed":
        for k in sorted(FIXED_ACCA_SIZES, reverse=True):
            if args.max_acca_size is not None and k > args.max_acca_size:
                continue
            slip_rows = build_fixed_acca(primary_acca_board, k, args=args, monster_only=False)
            if slip_rows:
                write_csv(slip_rows, outdir / f"slips_acca{k:02d}_{tag}.csv")

    market_acca_paths = write_market_fixed_accas(
        board,
        outdir,
        tag,
        args,
        monster_only_accas=args.monster_only_accas,
        monster_ftr_no_base=args.monster_ftr_no_base,
        max_acca_size=args.max_acca_size,
        selected_market=args.market,
    )

    family_counts: defaultdict[str, int] = defaultdict(int)
    for r in board:
        family_counts[r["family"]] += 1
    shadow_rows = sum(1 for r in board if "SHADOW" in safe_str(r.get("family", "")))
    live_rows = len(board) - shadow_rows

    market_counts: defaultdict[str, int] = defaultdict(int)
    for r in board:
        market_counts[r["market"]] += 1

    priority_counts: defaultdict[str, int] = defaultdict(int)
    for r in board:
        priority_counts[safe_str(r.get("priority_bucket", "UNPRIORITISED"))] += 1

    eligible_for_monster = primary_acca_board
    largest_eligible_acca = max((k for k in FIXED_ACCA_SIZES if len(eligible_for_monster) >= k), default=0)

    written_acca_sizes = []
    for k in FIXED_ACCA_SIZES:
        if args.max_acca_size is not None and k > args.max_acca_size:
            continue
        if len(primary_acca_board) >= k:
            written_acca_sizes.append(k)
    largest_written_acca = max(written_acca_sizes, default=0)

    mixed_board = filter_rows_for_acca_market(
        board,
        "mixed",
        args.monster_only_accas,
        args.monster_ftr_no_base,
    )
    ftr_board = filter_rows_for_acca_market(
        board,
        "ftr",
        args.monster_only_accas,
        args.monster_ftr_no_base,
    )
    btts_board = filter_rows_for_acca_market(
        board,
        "btts",
        args.monster_only_accas,
        args.monster_ftr_no_base,
    )
    ou25_board = filter_rows_for_acca_market(
        board,
        "ou25",
        args.monster_only_accas,
        args.monster_ftr_no_base,
    )

    largest_mixed_acca = max((k for k in FIXED_ACCA_SIZES if len(mixed_board) >= k), default=0)
    largest_ftr_acca = max((k for k in FIXED_ACCA_SIZES if len(ftr_board) >= k), default=0)
    largest_btts_acca = max((k for k in FIXED_ACCA_SIZES if len(btts_board) >= k), default=0)
    largest_ou25_acca = max((k for k in FIXED_ACCA_SIZES if len(ou25_board) >= k), default=0)

    print(f"""
  ── SUMMARY ──────────────────────────────────────────────────
  Ranked board:    {len(board)} legs
  Markets:         {dict(sorted(market_counts.items()))}
  Families:        {dict(sorted(family_counts.items()))}
  Live rows:       {live_rows}
  Shadow rows:     {shadow_rows}
  Priorities:      {dict(sorted(priority_counts.items()))}
  FTR rows:        {len(filter_board_by_market(board, 'ftr'))}
  BTTS rows:       {len(filter_board_by_market(board, 'btts'))}
  OU25 rows:       {len(filter_board_by_market(board, 'ou25'))}
  Leagues:         {len({r['league'] for r in board})} unique
  Dates:           {sorted({r['match_date'] for r in board})}
  Largest eligible acca: {largest_eligible_acca}-fold
  Largest written acca:  {largest_written_acca}-fold
  Largest mixed acca:    {largest_mixed_acca}-fold
  Largest FTR acca:      {largest_ftr_acca}-fold
  Largest BTTS acca:     {largest_btts_acca}-fold
  Largest OU25 acca:     {largest_ou25_acca}-fold
  Double lines:    {len(doubles)}
  Treble lines:    {len(trebles)}
  ─────────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()
