#!/usr/bin/env python3
"""Build row-level Phase 8H rebuild replay sweeps.

Research-only helper for the narrow-six combined replay work. It loads deploy
tier files from one or more walk-forward roots, joins realised goals, scores
FTR / BTTS / OU25 rows, and writes:

- variant scorecards
- market x league x tier scorecards
- reason-token hit tables
- per-league threshold grids
- FTR + team-goal combo proof tables

This script does not alter live policy or production rulebooks.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtest_deploy_csv import _compute_correct, _load_matches


DEFAULT_LEAGUES = [
    "USA MLS",
    "Belgium Pro",
    "Norway Eliteserien",
    "Spain La Liga",
    "Netherlands Eredivisie",
    "Germany Bundesliga",
]

BASE_COLUMNS = [
    "league",
    "fixture_key",
    "match_date",
    "home_team_name",
    "away_team_name",
    "market",
    "bookie_pick",
    "selection",
    "bookie_od",
    "bookie_implied",
    "bookie_implied_used",
    "model_p_for_bookie",
    "deploy_tier",
    "tier",
    "context_reason_codes",
    "reason_codes",
    "standard_reporting_bucket",
    "value_edge",
    "value_edge_tier",
    "signal_btts",
    "signal_over25",
    "prob_btts",
    "prob_btts_v2",
    "prob_over25",
    "prob_over25_v2",
    "p_meta_btts",
    "p_meta_ou25",
    "p_meta_ftr",
    "exp_goals_sum",
    "bookie_lambda_total_fit",
    "p00_est",
    "p_home_fts",
    "p_away_fts",
    "cs_mass_btts_yes",
    "cs_mass_btts_no",
    "cs_mass_over25",
    "cs_mass_under25",
    "cs_btts_margin",
    "cs_ou25_margin",
    "cs_mass_home_win",
    "cs_mass_draw",
    "cs_mass_away_win",
    "cs_entropy",
    "both_teams_2plus_mass",
    "mass_over25_via_one_sided_rout",
    "mass_0_goals",
    "mass_1_goal",
    "mass_2_goals",
    "mass_3_goals",
    "mass_4plus_goals",
    "grid_vs_cat_btts_gap",
    "grid_vs_xgb_btts_gap",
    "grid_vs_cat_ou25_gap",
    "grid_vs_xgb_ou25_gap",
    "cat_xgb_grid_btts_agreement_count",
    "cat_xgb_grid_ou25_agreement_count",
    "top3_over_count",
    "ou25_over_struct_pass",
    "ftr_margin",
    "power_diff",
    "pick_side_margin_top3",
    "pick_side_mass_top3",
    "ftr_combo_live_product",
    "ftr_combo_live_tier",
    "ftr_combo_live_allowed",
    "p_hw_and_hge2",
    "p_aw_and_age2",
    "hw_hge2_combo_prob",
    "aw_age2_combo_prob",
    "hw_hge2_combo_lambda",
    "aw_age2_combo_lambda",
    "hw_hge2_ge2_gap",
    "aw_age2_ge2_gap",
    "home_team_high_scoring_flag",
    "away_team_high_scoring_flag",
    "home_team_cs_specialist_flag",
    "away_team_cs_specialist_flag",
    "home_team_fts_risk_flag",
    "away_team_fts_risk_flag",
    "home_team_ge2_candidate_flag",
    "away_team_ge2_candidate_flag",
    "source_tier_file",
    "ftr_hit",
    "ou25_hit",
    "btts_yes_hit",
    "btts_no_hit",
    "home_team_goal_count",
    "away_team_goal_count",
    "actual_hw_and_hge2",
    "actual_aw_and_age2",
    "hw_and_hge2_hit",
    "aw_and_age2_hit",
]

PHASE8E_META_ELITE_THRESHOLDS = {
    "btts": {
        "Spain La Liga": 0.80,
        "Norway Eliteserien": 0.80,
        "Belgium Pro": 0.88,
        "Netherlands Eredivisie": 0.88,
        "USA MLS": 0.88,
    },
    "ou25": {
        "Spain La Liga": 0.80,
        "USA MLS": 0.80,
        "Norway Eliteserien": 0.85,
        "Germany Bundesliga": 0.88,
        "Netherlands Eredivisie": 0.88,
    },
}

FEATURES_BY_MARKET = {
    "btts": [
        ("model_p_for_bookie", ">="),
        ("prob_btts", ">="),
        ("prob_btts_v2", ">="),
        ("p_meta_btts", ">="),
        ("phase8e_meta_gap", ">="),
        ("cs_mass_btts_yes", ">="),
        ("cs_mass_btts_no", ">="),
        ("cs_btts_margin", ">="),
        ("both_teams_2plus_mass", ">="),
        ("grid_vs_cat_btts_gap", "<="),
        ("grid_vs_xgb_btts_gap", "<="),
        ("cat_xgb_grid_btts_agreement_count", ">="),
        ("exp_goals_sum", ">="),
        ("bookie_lambda_total_fit", ">="),
        ("value_edge", ">="),
        ("btts_fts_sum", "<="),
        ("btts_fts_max", "<="),
        ("p00_est", "<="),
    ],
    "ou25": [
        ("model_p_for_bookie", ">="),
        ("prob_over25", ">="),
        ("prob_over25_v2", ">="),
        ("p_meta_ou25", ">="),
        ("phase8e_meta_gap", ">="),
        ("cs_mass_over25", ">="),
        ("cs_mass_under25", ">="),
        ("cs_ou25_margin", ">="),
        ("mass_over25_via_one_sided_rout", ">="),
        ("grid_vs_cat_ou25_gap", "<="),
        ("grid_vs_xgb_ou25_gap", "<="),
        ("cat_xgb_grid_ou25_agreement_count", ">="),
        ("exp_goals_sum", ">="),
        ("bookie_lambda_total_fit", ">="),
        ("top3_over_count", ">="),
        ("value_edge", ">="),
    ],
    "ftr": [
        ("model_p_for_bookie", ">="),
        ("p_meta_ftr", ">="),
        ("ftr_margin", ">="),
        ("abs_power_diff", ">="),
        ("pick_side_margin_top3", ">="),
        ("pick_side_mass_top3", ">="),
        ("value_edge", ">="),
    ],
}

COMBO_FEATURES = [
    ("combo_prob", ">="),
    ("combo_lambda", ">="),
    ("combo_ge2_gap", ">="),
    ("p_meta_ftr", ">="),
    ("ftr_margin", ">="),
    ("abs_power_diff", ">="),
    ("pick_side_margin_top3", ">="),
    ("pick_side_mass_top3", ">="),
]

PAIR_GATE_FEATURES_BY_MARKET = {
    "btts": {
        "anchors": [
            ("p_meta_btts", ">="),
            ("phase8e_meta_gap", ">="),
            ("model_p_for_bookie", ">="),
        ],
        "confirms": [
            ("cs_mass_btts_yes", ">="),
            ("cs_btts_margin", ">="),
            ("both_teams_2plus_mass", ">="),
            ("grid_vs_cat_btts_gap", "<="),
            ("grid_vs_xgb_btts_gap", "<="),
            ("cat_xgb_grid_btts_agreement_count", ">="),
            ("btts_fts_sum", "<="),
            ("p00_est", "<="),
            ("exp_goals_sum", ">="),
        ],
    },
    "ou25": {
        "anchors": [
            ("p_meta_ou25", ">="),
            ("phase8e_meta_gap", ">="),
            ("model_p_for_bookie", ">="),
        ],
        "confirms": [
            ("cs_mass_over25", ">="),
            ("cs_ou25_margin", ">="),
            ("mass_over25_via_one_sided_rout", ">="),
            ("grid_vs_cat_ou25_gap", "<="),
            ("grid_vs_xgb_ou25_gap", "<="),
            ("cat_xgb_grid_ou25_agreement_count", ">="),
            ("top3_over_count", ">="),
            ("bookie_lambda_total_fit", ">="),
            ("exp_goals_sum", ">="),
        ],
    },
    "ftr": {
        "anchors": [
            ("p_meta_ftr", ">="),
            ("model_p_for_bookie", ">="),
        ],
        "confirms": [
            ("ftr_margin", ">="),
            ("abs_power_diff", ">="),
            ("pick_side_margin_top3", ">="),
            ("pick_side_mass_top3", ">="),
            ("value_edge", ">="),
        ],
    },
}


def _safe_num(s: pd.Series | float | int) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _split_label_root(value: str) -> tuple[str, Path]:
    if "=" not in value:
        p = Path(value)
        return p.name, p
    label, root = value.split("=", 1)
    return label.strip(), Path(root)


def _tier_from_path(path: Path) -> str:
    text = path.name.upper()
    if "DEPLOY_TIER_ELITE" in text:
        return "ELITE"
    if "DEPLOY_TIER_STANDARD" in text:
        return "STANDARD"
    if "DEPLOY_TIER_OBSERVE" in text:
        return "OBSERVE"
    return ""


def _window_from_path(path: Path) -> str:
    for part in path.parts:
        if re.match(r"^w\d{3}_", part):
            return part
    return ""


def _load_deploy_rows(root: Path, label: str, include_observe: bool = False) -> pd.DataFrame:
    tiers = ["ELITE", "STANDARD"]
    if include_observe:
        tiers.append("OBSERVE")

    files: list[Path] = []
    for tier in tiers:
        files.extend(sorted(root.glob(f"*/02_deploy/*DEPLOY_TIER_{tier}__*.csv")))

    frames = []
    for path in files:
        try:
            header = pd.read_csv(path, nrows=0).columns.tolist()
        except Exception:
            continue
        usecols = [c for c in BASE_COLUMNS if c in header]
        required = {"league", "fixture_key", "market", "bookie_pick"}
        if not required.issubset(set(usecols)):
            continue
        try:
            df = pd.read_csv(path, usecols=usecols, low_memory=False)
        except Exception:
            continue
        if df.empty:
            continue
        if "deploy_tier" not in df.columns:
            df["deploy_tier"] = _tier_from_path(path)
        df["source_tier_file"] = _tier_from_path(path)
        df["window_id"] = _window_from_path(path)
        df["variant"] = label
        df["source_file"] = str(path)
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["deploy_tier"] = out["deploy_tier"].astype("string").fillna("")
    out.loc[out["deploy_tier"].eq(""), "deploy_tier"] = out["source_tier_file"]
    return out


def _load_scored_rows(root: Path, label: str, include_observe: bool = False) -> pd.DataFrame:
    files = sorted(root.glob("*/03_scored/DEPLOY_COMBINED_SCORED_*.csv"))
    frames = []
    allowed_tiers = {"ELITE", "STANDARD"}
    if include_observe:
        allowed_tiers.add("OBSERVE")

    for path in files:
        try:
            header = pd.read_csv(path, nrows=0).columns.tolist()
        except Exception:
            continue
        usecols = [c for c in BASE_COLUMNS if c in header]
        required = {"league", "fixture_key", "market", "bookie_pick"}
        if not required.issubset(set(usecols)):
            continue
        try:
            df = pd.read_csv(path, usecols=usecols, low_memory=False)
        except Exception:
            continue
        if df.empty:
            continue
        if "deploy_tier" not in df.columns:
            if "tier" in df.columns:
                df["deploy_tier"] = df["tier"]
            elif "source_tier_file" in df.columns:
                df["deploy_tier"] = df["source_tier_file"]
            else:
                df["deploy_tier"] = ""
        df["deploy_tier"] = df["deploy_tier"].astype("string").fillna("")
        if "source_tier_file" not in df.columns:
            df["source_tier_file"] = df["deploy_tier"]
        df = df[df["deploy_tier"].isin(allowed_tiers)].copy()
        if df.empty:
            continue
        df["window_id"] = _window_from_path(path)
        df["variant"] = label
        df["source_file"] = str(path)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["correct"] = np.nan
    market = out["market"].astype("string").str.lower().str.strip()
    pick = out["bookie_pick"].astype("string").str.upper().str.strip()
    out.loc[market.eq("ftr"), "correct"] = _safe_num(out.get("ftr_hit", np.nan)).loc[market.eq("ftr")]
    out.loc[market.eq("ou25"), "correct"] = _safe_num(out.get("ou25_hit", np.nan)).loc[market.eq("ou25")]
    yes_mask = market.eq("btts") & pick.eq("YES")
    no_mask = market.eq("btts") & pick.eq("NO")
    out.loc[yes_mask, "correct"] = _safe_num(out.get("btts_yes_hit", np.nan)).loc[yes_mask]
    out.loc[no_mask, "correct"] = _safe_num(out.get("btts_no_hit", np.nan)).loc[no_mask]
    return _add_derived_fields(out)


def _add_derived_fields(out: pd.DataFrame) -> pd.DataFrame:
    out = out.copy()
    out["market_norm"] = out["market"].astype("string").str.lower().str.strip()
    out["btts_fts_sum"] = _safe_num(out.get("p_home_fts", np.nan)) + _safe_num(out.get("p_away_fts", np.nan))
    out["btts_fts_max"] = pd.concat(
        [_safe_num(out.get("p_home_fts", np.nan)), _safe_num(out.get("p_away_fts", np.nan))],
        axis=1,
    ).max(axis=1)
    out["abs_power_diff"] = _safe_num(out.get("power_diff", np.nan)).abs()
    phase8e_floor = []
    for market, league in zip(out["market_norm"], out["league"].astype("string")):
        phase8e_floor.append(PHASE8E_META_ELITE_THRESHOLDS.get(str(market), {}).get(str(league), np.nan))
    out["phase8e_meta_floor"] = phase8e_floor
    out["phase8e_meta_gap"] = np.nan
    btts = out["market_norm"].eq("btts")
    ou25 = out["market_norm"].eq("ou25")
    out.loc[btts, "phase8e_meta_gap"] = _safe_num(out.get("p_meta_btts", np.nan)).loc[btts] - _safe_num(
        out.loc[btts, "phase8e_meta_floor"]
    )
    out.loc[ou25, "phase8e_meta_gap"] = _safe_num(out.get("p_meta_ou25", np.nan)).loc[ou25] - _safe_num(
        out.loc[ou25, "phase8e_meta_floor"]
    )
    out["phase8e_meta_floor_pass"] = _safe_num(out["phase8e_meta_gap"]).ge(0).astype(float)
    out.loc[_safe_num(out["phase8e_meta_floor"]).isna(), "phase8e_meta_floor_pass"] = np.nan
    out["reason_text"] = (
        out.get("context_reason_codes", pd.Series("", index=out.index)).astype("string").fillna("")
        + "|"
        + out.get("reason_codes", pd.Series("", index=out.index)).astype("string").fillna("")
    )
    return out


def _score_rows(df: pd.DataFrame, matches_root: Path) -> pd.DataFrame:
    if df.empty:
        return df

    scored_parts = []
    truth_cache: dict[str, pd.DataFrame | None] = {}

    for league, part in df.groupby("league", dropna=False):
        league_name = str(league)
        if league_name not in truth_cache:
            truth_cache[league_name] = _load_matches(matches_root, league_name)
        truth = truth_cache[league_name]
        if truth is None:
            scored = part.copy()
            scored["home_team_goal_count"] = np.nan
            scored["away_team_goal_count"] = np.nan
        else:
            scored = part.merge(truth, on="fixture_key", how="left")
        scored_parts.append(scored)

    out = pd.concat(scored_parts, ignore_index=True)
    out["correct"] = _compute_correct(out)
    return _add_derived_fields(out)


def _metric_summary(group: pd.DataFrame, hit_col: str = "correct") -> dict[str, object]:
    hit = _safe_num(group.get(hit_col, np.nan))
    graded = int(hit.notna().sum())
    wins = float((hit == 1).sum())
    odds = _safe_num(group.get("bookie_od", np.nan))
    profit = np.where(hit == 1, odds - 1.0, np.where(hit == 0, -1.0, np.nan))
    return {
        "rows": int(len(group)),
        "graded": graded,
        "wins": wins,
        "losses": int((hit == 0).sum()),
        "hit_rate": float(wins / graded) if graded else np.nan,
        "avg_bookie_od": float(odds.mean()) if odds.notna().any() else np.nan,
        "avg_model_p": float(_safe_num(group.get("model_p_for_bookie", np.nan)).mean()),
        "profit": float(np.nansum(profit)),
        "roi": float(np.nansum(profit) / graded) if graded else np.nan,
    }


def _group_summary(df: pd.DataFrame, group_cols: list[str], hit_col: str = "correct") -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(_metric_summary(group, hit_col=hit_col))
        rows.append(row)
    return pd.DataFrame(rows)


def _token_rows(df: pd.DataFrame, min_rows: int) -> pd.DataFrame:
    rows = []
    live = df[df["correct"].notna()].copy()
    for (variant, market, tier, league), group in live.groupby(
        ["variant", "market_norm", "deploy_tier", "league"], dropna=False
    ):
        base = _metric_summary(group)
        tokens: set[str] = set()
        for text in group["reason_text"].fillna("").astype(str):
            for token in re.split(r"[|,;]+", text):
                token = token.strip()
                if token:
                    tokens.add(token)
        for token in sorted(tokens):
            mask = group["reason_text"].fillna("").astype(str).str.contains(token, regex=False)
            n = int(mask.sum())
            if n < min_rows:
                continue
            sub = group.loc[mask]
            row = {
                "variant": variant,
                "market": market,
                "tier": tier,
                "league": league,
                "token": token,
                "base_graded": base["graded"],
                "base_hit_rate": base["hit_rate"],
            }
            row.update(_metric_summary(sub))
            row["hit_delta_vs_base"] = row["hit_rate"] - row["base_hit_rate"]
            rows.append(row)
    return pd.DataFrame(rows)


def _threshold_values(series: pd.Series, quantiles: Iterable[float]) -> list[float]:
    s = _safe_num(series).dropna()
    if s.empty:
        return []
    vals = set()
    for q in quantiles:
        try:
            val = float(s.quantile(q))
        except Exception:
            continue
        if np.isfinite(val):
            vals.add(round(val, 6))
    return sorted(vals)


def _apply_op(series: pd.Series, op: str, threshold: float) -> pd.Series:
    s = _safe_num(series)
    if op == "<=":
        return (s <= threshold).fillna(False)
    return (s >= threshold).fillna(False)


def _threshold_grid(df: pd.DataFrame, min_rows: int, quantiles: list[float]) -> pd.DataFrame:
    rows = []
    live = df[df["correct"].notna() & df["deploy_tier"].isin(["ELITE", "STANDARD"])].copy()
    for (variant, market, tier, league), group in live.groupby(
        ["variant", "market_norm", "deploy_tier", "league"], dropna=False
    ):
        features = FEATURES_BY_MARKET.get(str(market), [])
        if not features:
            continue
        base = _metric_summary(group)
        if int(base["graded"]) < min_rows:
            continue
        for feature, op in features:
            if feature not in group.columns:
                continue
            values = _threshold_values(group[feature], quantiles)
            for threshold in values:
                mask = _apply_op(group[feature], op, threshold)
                sub = group.loc[mask]
                if int(sub["correct"].notna().sum()) < min_rows:
                    continue
                row = {
                    "variant": variant,
                    "market": market,
                    "tier": tier,
                    "league": league,
                    "feature": feature,
                    "op": op,
                    "threshold": threshold,
                    "base_graded": base["graded"],
                    "base_hit_rate": base["hit_rate"],
                }
                row.update(_metric_summary(sub))
                row["coverage_vs_base"] = row["graded"] / base["graded"] if base["graded"] else np.nan
                row["hit_delta_vs_base"] = row["hit_rate"] - base["hit_rate"]
                rows.append(row)
    return pd.DataFrame(rows)


def _pair_threshold_grid(df: pd.DataFrame, min_rows: int, quantiles: list[float]) -> pd.DataFrame:
    rows = []
    live = df[df["correct"].notna() & df["deploy_tier"].isin(["ELITE", "STANDARD"])].copy()
    for (variant, market, tier, league), group in live.groupby(
        ["variant", "market_norm", "deploy_tier", "league"], dropna=False
    ):
        config = PAIR_GATE_FEATURES_BY_MARKET.get(str(market))
        if not config:
            continue
        base = _metric_summary(group)
        if int(base["graded"]) < min_rows:
            continue

        anchor_specs = [(f, op) for f, op in config["anchors"] if f in group.columns]
        confirm_specs = [(f, op) for f, op in config["confirms"] if f in group.columns]
        for anchor_feature, anchor_op in anchor_specs:
            for anchor_threshold in _threshold_values(group[anchor_feature], quantiles):
                anchor_mask = _apply_op(group[anchor_feature], anchor_op, anchor_threshold)
                if int(group.loc[anchor_mask, "correct"].notna().sum()) < min_rows:
                    continue
                for confirm_feature, confirm_op in confirm_specs:
                    if confirm_feature == anchor_feature:
                        continue
                    for confirm_threshold in _threshold_values(group.loc[anchor_mask, confirm_feature], quantiles):
                        mask = anchor_mask & _apply_op(group[confirm_feature], confirm_op, confirm_threshold)
                        sub = group.loc[mask]
                        if int(sub["correct"].notna().sum()) < min_rows:
                            continue
                        row = {
                            "variant": variant,
                            "market": market,
                            "tier": tier,
                            "league": league,
                            "anchor_feature": anchor_feature,
                            "anchor_op": anchor_op,
                            "anchor_threshold": anchor_threshold,
                            "confirm_feature": confirm_feature,
                            "confirm_op": confirm_op,
                            "confirm_threshold": confirm_threshold,
                            "base_graded": base["graded"],
                            "base_hit_rate": base["hit_rate"],
                        }
                        row.update(_metric_summary(sub))
                        row["coverage_vs_base"] = row["graded"] / base["graded"] if base["graded"] else np.nan
                        row["hit_delta_vs_base"] = row["hit_rate"] - base["hit_rate"]
                        rows.append(row)
    return pd.DataFrame(rows)


def _phase8e_prior_grid(df: pd.DataFrame, min_rows: int) -> pd.DataFrame:
    rows = []
    live = df[df["correct"].notna() & df["deploy_tier"].isin(["ELITE", "STANDARD"])].copy()
    for (variant, market, tier, league), group in live.groupby(
        ["variant", "market_norm", "deploy_tier", "league"], dropna=False
    ):
        threshold = PHASE8E_META_ELITE_THRESHOLDS.get(str(market), {}).get(str(league))
        feature = {"btts": "p_meta_btts", "ou25": "p_meta_ou25"}.get(str(market))
        if threshold is None or feature is None or feature not in group.columns:
            continue
        base = _metric_summary(group)
        if int(base["graded"]) < min_rows:
            continue
        sub = group.loc[_safe_num(group[feature]).ge(threshold)]
        if int(sub["correct"].notna().sum()) < min_rows:
            continue
        row = {
            "variant": variant,
            "market": market,
            "tier": tier,
            "league": league,
            "feature": feature,
            "op": ">=",
            "threshold": threshold,
            "base_graded": base["graded"],
            "base_hit_rate": base["hit_rate"],
            "prior_source": "PHASE8E_META_ELITE_LOCKED",
        }
        row.update(_metric_summary(sub))
        row["coverage_vs_base"] = row["graded"] / base["graded"] if base["graded"] else np.nan
        row["hit_delta_vs_base"] = row["hit_rate"] - base["hit_rate"]
        rows.append(row)
    return pd.DataFrame(rows)


def _score_combo_rows(df: pd.DataFrame) -> pd.DataFrame:
    ftr = df[df["market_norm"].eq("ftr")].copy()
    if ftr.empty or "ftr_combo_live_product" not in ftr.columns:
        return pd.DataFrame()

    hg = _safe_num(ftr.get("home_team_goal_count", np.nan))
    ag = _safe_num(ftr.get("away_team_goal_count", np.nan))
    known = hg.notna() & ag.notna()
    product = ftr["ftr_combo_live_product"].astype("string").fillna("")

    ftr["combo_correct"] = np.nan
    home_mask = product.eq("HOME_WIN_AND_HOME_GE2") & known
    away_mask = product.eq("AWAY_WIN_AND_AWAY_GE2") & known
    ftr.loc[home_mask, "combo_correct"] = ((hg > ag) & (hg >= 2)).astype(float).loc[home_mask]
    ftr.loc[away_mask, "combo_correct"] = ((ag > hg) & (ag >= 2)).astype(float).loc[away_mask]

    allowed = _safe_num(ftr.get("ftr_combo_live_allowed", np.nan)).eq(1)
    ftr = ftr.loc[allowed & ftr["combo_correct"].notna()].copy()
    if ftr.empty:
        return ftr

    product = ftr["ftr_combo_live_product"].astype("string").fillna("")
    ftr["combo_prob"] = np.where(
        product.eq("HOME_WIN_AND_HOME_GE2"),
        _safe_num(ftr.get("p_hw_and_hge2", np.nan)),
        _safe_num(ftr.get("p_aw_and_age2", np.nan)),
    )
    ftr["combo_lambda"] = np.where(
        product.eq("HOME_WIN_AND_HOME_GE2"),
        _safe_num(ftr.get("hw_hge2_combo_lambda", np.nan)),
        _safe_num(ftr.get("aw_age2_combo_lambda", np.nan)),
    )
    ftr["combo_ge2_gap"] = np.where(
        product.eq("HOME_WIN_AND_HOME_GE2"),
        _safe_num(ftr.get("hw_hge2_ge2_gap", np.nan)),
        _safe_num(ftr.get("aw_age2_ge2_gap", np.nan)),
    )
    return ftr


def _combo_threshold_grid(combo: pd.DataFrame, min_rows: int, quantiles: list[float]) -> pd.DataFrame:
    if combo.empty:
        return pd.DataFrame()

    rows = []
    for (variant, league, product, combo_tier), group in combo.groupby(
        ["variant", "league", "ftr_combo_live_product", "ftr_combo_live_tier"], dropna=False
    ):
        base = _metric_summary(group, hit_col="combo_correct")
        if int(base["graded"]) < min_rows:
            continue
        for feature, op in COMBO_FEATURES:
            if feature not in group.columns:
                continue
            for threshold in _threshold_values(group[feature], quantiles):
                mask = _apply_op(group[feature], op, threshold)
                sub = group.loc[mask]
                if int(sub["combo_correct"].notna().sum()) < min_rows:
                    continue
                row = {
                    "variant": variant,
                    "league": league,
                    "combo_product": product,
                    "combo_tier": combo_tier,
                    "feature": feature,
                    "op": op,
                    "threshold": threshold,
                    "base_graded": base["graded"],
                    "base_hit_rate": base["hit_rate"],
                }
                row.update(_metric_summary(sub, hit_col="combo_correct"))
                row["coverage_vs_base"] = row["graded"] / base["graded"] if base["graded"] else np.nan
                row["hit_delta_vs_base"] = row["hit_rate"] - base["hit_rate"]
                rows.append(row)
    return pd.DataFrame(rows)


def _write_markdown(
    path: Path,
    variant_summary: pd.DataFrame,
    league_scorecard: pd.DataFrame,
    phase8e_prior: pd.DataFrame,
    threshold_best: pd.DataFrame,
    pair_best: pd.DataFrame,
    combo_summary: pd.DataFrame,
) -> None:
    def table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No data._"
        show = df.copy()
        for col in show.columns:
            if pd.api.types.is_float_dtype(show[col]):
                show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
            else:
                show[col] = show[col].map(lambda x: "" if pd.isna(x) else str(x))
        headers = list(show.columns)
        rows = show.values.tolist()
        widths = [
            max([len(str(headers[i]))] + [len(str(row[i])) for row in rows])
            for i in range(len(headers))
        ]

        def fmt(row: list[object]) -> str:
            return "| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |"

        lines = [fmt(headers), "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"]
        lines.extend(fmt(row) for row in rows)
        return "\n".join(lines)

    lines = [
        "# Phase 8H Replay Sweep Summary",
        "",
        "Research-only row-level sweep output. No production policy files changed.",
        "",
        "## Variant Summary",
        table(variant_summary),
        "",
        "## Weak Live Lanes",
    ]

    weak = league_scorecard[
        league_scorecard["tier"].isin(["ELITE", "STANDARD"])
        & league_scorecard["hit_rate"].notna()
        & (league_scorecard["graded"] >= 20)
    ].sort_values(["hit_rate", "graded"], ascending=[True, False]).head(20)
    lines.append(table(weak) if not weak.empty else "_No weak lanes with minimum sample._")

    lines.extend(["", "## Best Threshold Candidates"])
    cols = [
        "variant",
        "market",
        "tier",
        "league",
        "feature",
        "op",
        "threshold",
        "graded",
        "hit_rate",
        "base_graded",
        "base_hit_rate",
        "hit_delta_vs_base",
        "coverage_vs_base",
    ]
    show = threshold_best[[c for c in cols if c in threshold_best.columns]].head(40)
    lines.append(table(show) if not show.empty else "_No threshold candidates._")

    lines.extend(["", "## Best Two-Feature Gate Candidates"])
    pair_cols = [
        "variant",
        "market",
        "tier",
        "league",
        "anchor_feature",
        "anchor_op",
        "anchor_threshold",
        "confirm_feature",
        "confirm_op",
        "confirm_threshold",
        "graded",
        "hit_rate",
        "base_graded",
        "base_hit_rate",
        "hit_delta_vs_base",
        "coverage_vs_base",
    ]
    pair_show = pair_best[[c for c in pair_cols if c in pair_best.columns]].head(40)
    lines.append(table(pair_show) if not pair_show.empty else "_No two-feature gate candidates._")

    lines.extend(["", "## Phase 8E Prior Gate Replay"])
    prior_cols = [
        "variant",
        "market",
        "tier",
        "league",
        "feature",
        "op",
        "threshold",
        "graded",
        "hit_rate",
        "base_graded",
        "base_hit_rate",
        "hit_delta_vs_base",
        "coverage_vs_base",
    ]
    prior_show = phase8e_prior[[c for c in prior_cols if c in phase8e_prior.columns]].sort_values(
        ["market", "league", "variant", "tier"], na_position="last"
    )
    lines.append(table(prior_show) if not prior_show.empty else "_No Phase 8E prior gates met the minimum sample._")

    lines.extend(["", "## Combo Lane Summary"])
    lines.append(table(combo_summary) if not combo_summary.empty else "_No combo rows._")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Phase 8H replay row-level sweeps")
    ap.add_argument("--root", action="append", required=True, help="label=/path/to/walkforward_root")
    ap.add_argument("--matches-root", default="Matches")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--leagues", default=",".join(DEFAULT_LEAGUES))
    ap.add_argument("--min-rows", type=int, default=20)
    ap.add_argument("--include-observe", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    matches_root = Path(args.matches_root)
    leagues = {x.strip() for x in args.leagues.split(",") if x.strip()}
    quantiles = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    all_rows = []
    for root_arg in args.root:
        label, root = _split_label_root(root_arg)
        if not root.exists():
            raise SystemExit(f"Missing root for {label}: {root}")
        rows = _load_scored_rows(root, label, include_observe=args.include_observe)
        if rows.empty:
            rows = _load_deploy_rows(root, label, include_observe=args.include_observe)
            if rows.empty:
                raise SystemExit(f"No scored or deploy tier rows loaded for {label}: {root}")
            rows = rows[rows["league"].astype("string").isin(leagues)].copy()
            rows = _score_rows(rows, matches_root)
        else:
            rows = rows[rows["league"].astype("string").isin(leagues)].copy()
        all_rows.append(rows)

    scored = pd.concat(all_rows, ignore_index=True)
    scored.to_csv(outdir / "phase8h_replay_row_level_scored.csv", index=False)

    live = scored[scored["deploy_tier"].isin(["ELITE", "STANDARD"])].copy()

    variant_summary = _group_summary(live, ["variant", "market_norm", "deploy_tier"])
    variant_summary = variant_summary.rename(columns={"market_norm": "market", "deploy_tier": "tier"})
    variant_summary.to_csv(outdir / "phase8h_replay_variant_scorecard.csv", index=False)

    league_scorecard = _group_summary(live, ["variant", "market_norm", "deploy_tier", "league"])
    league_scorecard = league_scorecard.rename(columns={"market_norm": "market", "deploy_tier": "tier"})
    league_scorecard.to_csv(outdir / "phase8h_replay_market_league_tier_scorecard.csv", index=False)

    token_hits = _token_rows(scored, args.min_rows)
    token_hits.to_csv(outdir / "phase8h_replay_reason_token_hits.csv", index=False)

    grid = _threshold_grid(scored, args.min_rows, quantiles)
    grid.to_csv(outdir / "phase8h_replay_threshold_grid.csv", index=False)
    if grid.empty:
        best = grid
    else:
        best = grid.sort_values(
            ["hit_rate", "hit_delta_vs_base", "graded", "coverage_vs_base"],
            ascending=[False, False, False, False],
        ).groupby(["variant", "market", "tier", "league"], dropna=False).head(5)
    best.to_csv(outdir / "phase8h_replay_threshold_grid_best.csv", index=False)

    pair_grid = _pair_threshold_grid(scored, args.min_rows, quantiles)
    pair_grid.to_csv(outdir / "phase8h_replay_pair_gate_grid.csv", index=False)
    if pair_grid.empty:
        pair_best = pair_grid
    else:
        pair_best = pair_grid.sort_values(
            ["hit_rate", "hit_delta_vs_base", "graded", "coverage_vs_base"],
            ascending=[False, False, False, False],
        ).groupby(["variant", "market", "tier", "league"], dropna=False).head(10)
    pair_best.to_csv(outdir / "phase8h_replay_pair_gate_grid_best.csv", index=False)

    phase8e_prior = _phase8e_prior_grid(scored, args.min_rows)
    phase8e_prior.to_csv(outdir / "phase8h_replay_phase8e_prior_gate.csv", index=False)

    combo = _score_combo_rows(scored)
    combo.to_csv(outdir / "phase8h_replay_combo_rows_scored.csv", index=False)
    combo_summary = (
        _group_summary(combo, ["variant", "league", "ftr_combo_live_product", "ftr_combo_live_tier"], "combo_correct")
        if not combo.empty
        else pd.DataFrame()
    )
    combo_summary.to_csv(outdir / "phase8h_replay_combo_lane_proof.csv", index=False)

    combo_grid = _combo_threshold_grid(combo, args.min_rows, quantiles)
    combo_grid.to_csv(outdir / "phase8h_replay_combo_threshold_grid.csv", index=False)
    if combo_grid.empty:
        combo_best = combo_grid
    else:
        combo_best = combo_grid.sort_values(
            ["hit_rate", "hit_delta_vs_base", "graded", "coverage_vs_base"],
            ascending=[False, False, False, False],
        ).groupby(["variant", "league", "combo_product", "combo_tier"], dropna=False).head(5)
    combo_best.to_csv(outdir / "phase8h_replay_combo_threshold_grid_best.csv", index=False)

    _write_markdown(
        outdir / "phase8h_replay_sweep_summary.md",
        variant_summary,
        league_scorecard,
        phase8e_prior,
        best,
        pair_best,
        combo_summary,
    )

    print(f"[ok] wrote {outdir}")


if __name__ == "__main__":
    main()
