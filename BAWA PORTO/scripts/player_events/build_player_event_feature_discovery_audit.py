#!/usr/bin/env python3
"""
Research-only feature discovery audit for player-event intelligence.

Consumes scored player-event audit outputs and ranks available feature signals
by market, league, role, and feature family. It also records missing feature
families we should not pretend are available.

No production model artifacts, deploy routes, or priced prop probabilities are
written.
"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings(
    "ignore",
    message="Skipping features without any observed values",
    category=UserWarning,
)


ROOT = Path(__file__).resolve().parents[2]
MARKET_AUDIT_DIR = ROOT / "reports" / "2026-05-06" / "player_events_market_goal_spine_model_audit"
TACKLES_AUDIT_DIR = ROOT / "reports" / "2026-05-06" / "player_events_goal_spine_model_audit"
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-06" / "player_event_feature_discovery_audit"

MIN_FEATURE_ROWS = 250
MAX_MODEL_FEATURES = 60


@dataclass(frozen=True)
class MarketSpec:
    market: str
    display_market: str
    input_path: Path
    hit_col: str


MARKETS = {
    "shots": MarketSpec(
        "shots",
        "Player Shots 0.5+",
        MARKET_AUDIT_DIR / "player_events_goal_spine_shots_joined_scored.csv",
        "actual_hit_ge1",
    ),
    "shots_ge2": MarketSpec(
        "shots_ge2",
        "Player Shots 1.5+",
        MARKET_AUDIT_DIR / "player_events_goal_spine_shots_ge2_joined_scored.csv",
        "actual_hit_ge2",
    ),
    "shots_ge3": MarketSpec(
        "shots_ge3",
        "Player Shots 2.5+",
        MARKET_AUDIT_DIR / "player_events_goal_spine_shots_ge3_joined_scored.csv",
        "actual_hit_ge3",
    ),
    "shots_on_target": MarketSpec(
        "shots_on_target",
        "Player SOT 0.5+",
        MARKET_AUDIT_DIR / "player_events_goal_spine_shots_on_target_joined_scored.csv",
        "actual_hit_ge1",
    ),
    "sot_ge2_attackers": MarketSpec(
        "sot_ge2_attackers",
        "Striker/Winger SOT 1.5+",
        MARKET_AUDIT_DIR / "player_events_goal_spine_sot_ge2_attackers_joined_scored.csv",
        "actual_hit_ge2",
    ),
    "sot_ge3_attackers": MarketSpec(
        "sot_ge3_attackers",
        "Striker/Winger SOT 2.5+",
        MARKET_AUDIT_DIR / "player_events_goal_spine_sot_ge3_attackers_joined_scored.csv",
        "actual_hit_ge3",
    ),
    "fouls_committed": MarketSpec(
        "fouls_committed",
        "Player Fouls 1.5+",
        MARKET_AUDIT_DIR / "player_events_goal_spine_fouls_committed_joined_scored.csv",
        "actual_hit_ge2",
    ),
    "tackles": MarketSpec(
        "tackles",
        "Player Tackles 1.5+",
        TACKLES_AUDIT_DIR / "player_events_goal_spine_tackles_joined_scored.csv",
        "actual_hit_ge2",
    ),
}


FEATURE_FAMILIES = {
    "opponent_allowance_by_role": [
        "opp_tackles_allowed_def_l10",
        "opp_tackles_allowed_mid_l10",
        "opp_tackles_allowed_pos_l10",
        "opp_possession_share_l10",
        "opp_dribble_attempts_l10",
    ],
    "opponent_attack_allowance_by_role": [
        "opp_attack_allowed_role_source_matches",
        "opp_attack_allowed_attacker_any_source_matches",
        "opp_attack_allowed_role_shots_per_match_l5",
        "opp_attack_allowed_role_sot_per_match_l5",
        "opp_attack_allowed_role_shots_per_player_l5",
        "opp_attack_allowed_role_sot_per_player_l5",
        "opp_attack_allowed_role_player_shot_ge1_rate_l5",
        "opp_attack_allowed_role_player_shot_ge2_rate_l5",
        "opp_attack_allowed_role_player_sot_ge1_rate_l5",
        "opp_attack_allowed_role_player_sot_ge2_rate_l5",
        "opp_attack_allowed_role_shots_per_match_l10",
        "opp_attack_allowed_role_sot_per_match_l10",
        "opp_attack_allowed_role_shots_per_player_l10",
        "opp_attack_allowed_role_sot_per_player_l10",
        "opp_attack_allowed_role_player_shot_ge1_rate_l10",
        "opp_attack_allowed_role_player_shot_ge2_rate_l10",
        "opp_attack_allowed_role_player_sot_ge1_rate_l10",
        "opp_attack_allowed_role_player_sot_ge2_rate_l10",
        "opp_attack_allowed_attacker_any_shots_per_match_l5",
        "opp_attack_allowed_attacker_any_sot_per_match_l5",
        "opp_attack_allowed_attacker_any_shots_per_player_l5",
        "opp_attack_allowed_attacker_any_sot_per_player_l5",
        "opp_attack_allowed_attacker_any_player_shot_ge1_rate_l5",
        "opp_attack_allowed_attacker_any_player_shot_ge2_rate_l5",
        "opp_attack_allowed_attacker_any_player_sot_ge1_rate_l5",
        "opp_attack_allowed_attacker_any_player_sot_ge2_rate_l5",
        "opp_attack_allowed_attacker_any_shots_per_match_l10",
        "opp_attack_allowed_attacker_any_sot_per_match_l10",
        "opp_attack_allowed_attacker_any_shots_per_player_l10",
        "opp_attack_allowed_attacker_any_sot_per_player_l10",
        "opp_attack_allowed_attacker_any_player_shot_ge1_rate_l10",
        "opp_attack_allowed_attacker_any_player_shot_ge2_rate_l10",
        "opp_attack_allowed_attacker_any_player_sot_ge1_rate_l10",
        "opp_attack_allowed_attacker_any_player_sot_ge2_rate_l10",
    ],
    "defensive_matchup_pressure": [
        "weaker_side_under_pressure_flag",
        "power_gap_directional_pressure_score",
        "weak_flank_overload_flag",
        "weak_left_flank_overload_flag",
        "weak_right_flank_overload_flag",
        "weak_midfield_overload_flag",
        "weak_territory_protection_flag",
        "fixture_territorial_stress_score",
        "fixture_wide_duel_score",
        "cb_duel_pressure_score",
        "cb_front_foot_duel_flag",
        "home_team_dribbled_past_l5",
        "away_team_dribbled_past_l5",
    ],
    "game_state_goal_environment": [
        "team_power_edge",
        "team_power_rating",
        "opponent_power_rating",
        "og_pre_match_xg_home",
        "og_pre_match_xg_away",
        "og_xg_total",
        "og_xg_weaker_side",
        "og_over25_pre",
        "og_btts_pre",
        "og_goal_environment_score",
        "og_battle_on_score",
        "og_power_gap_abs",
        "starting_xi_attack_quality_score",
        "starting_xi_quality_edge",
    ],
    "set_piece_corner_pressure": [
        "fixture_corner_pressure_score",
        "home_team_corners_for_l5",
        "home_team_corners_for_l10",
        "away_team_corners_for_l5",
        "away_team_corners_for_l10",
        "home_team_corners_against_l5",
        "away_team_corners_against_l5",
        "cb_duel_pressure_score",
    ],
    "referee_foul_ecosystem": [
        "fixture_foul_density_score",
        "ref_cards_per_match",
        "ref_foul_to_card_ratio",
        "ref_dissent_strictness",
        "ref_timewasting_strictness",
        "rivalry_flag",
        "match_stakes_score",
        "team_avg_fouls",
        "team_avg_yellows",
        "h2h_total_fouls_l5",
        "h2h_total_fouls_l10",
    ],
    "lineup_role_certainty": [
        "expected_start_flag",
        "expected_minutes",
        "expected_minutes_audit",
        "expected_start_prob",
        "expected_minutes_proof",
        "attacker_recent_starts_l5",
        "attacker_recent_starts_l8",
        "attacker_recent_start_share_l5",
        "attacker_recent_start_share_l8",
        "minutes_last_3_matches",
        "days_rest",
        "recent_injury_return_flag",
    ],
    "home_away_player_splits": [
        "player_team_side",
        "attacker_recent_home_shots_per90_l5",
        "attacker_recent_home_sot_per90_l5",
        "attacker_recent_away_shots_per90_l5",
        "attacker_recent_away_sot_per90_l5",
        "attacker_recent_home_shots_per90_l8",
        "attacker_recent_home_sot_per90_l8",
        "attacker_recent_away_shots_per90_l8",
        "attacker_recent_away_sot_per90_l8",
    ],
    "team_tactical_shape": [
        "team_formation",
        "opponent_formation",
        "formation_matchup_label",
        "formation_pressure_score",
        "formation_wide_overload_flag",
        "formation_midfield_grind_flag",
        "formation_left_wide_overload_score",
        "formation_right_wide_overload_score",
        "lineup_formation_attack_delta",
        "lineup_formation_defence_delta",
        "tactical_role",
        "position_group",
        "fixture_attacking_style_label",
        "fixture_style_label",
    ],
    "player_involvement_recent_form": [
        "shots_per90",
        "shots_on_target_per90",
        "goals_per90",
        "assists_per90",
        "key_passes_per90",
        "player_form_rating_l5",
        "player_quality_score_l5",
        "attacker_recent_shots_l5",
        "attacker_recent_sot_l5",
        "attacker_recent_goals_l5",
        "attacker_recent_assists_l5",
        "attacker_recent_goal_contributions_l5",
        "attacker_recent_key_passes_l5",
        "attacker_recent_shots_per90_l5",
        "attacker_recent_sot_per90_l5",
        "attacker_recent_goal_contributions_per90_l5",
        "attacker_recent_shots_l8",
        "attacker_recent_sot_l8",
        "attacker_recent_goal_contributions_l8",
        "attacker_recent_shots_per90_l8",
        "attacker_recent_sot_per90_l8",
        "attacker_recent_goal_contributions_per90_l8",
    ],
    "competition_league_style": [
        "league_tag",
        "season_tag",
        "competition",
        "home_team_shots_l5",
        "away_team_shots_l5",
        "home_team_shots_on_goal_l5",
        "away_team_shots_on_goal_l5",
        "home_team_tackles_l5",
        "away_team_tackles_l5",
        "fixture_tackle_density_score",
        "fixture_midfield_grind_score",
        "fixture_attack_pressure_score",
    ],
}

MISSING_FEATURES = {
    "player_expected_goals": "not present in normalized match_player_stats or feature inputs",
    "player_expected_assists": "not present in normalized match_player_stats or feature inputs",
    "big_chances": "not present in normalized match_player_stats or events",
    "touches_in_box": "not present in normalized match_player_stats or events",
    "shots_from_set_pieces": "not separated in normalized player stats",
    "teammate_high_usage_absence": "not built as a current fixture feature yet",
    "l3_minus_l8_involvement_delta": "can be built next; L5/L8 currently available",
    "fullback_specific_shots_allowed": "not built separately; current attack allowance covers central strikers, wide forwards, and wide midfielders/wingers",
    "striker_winger_shots_allowed": "built as PLAYER_EVENT_OPPONENT_ATTACK_ALLOWANCE_FEATURES with role-matched and attacker-any L5/L10 shots/SOT allowed",
}


def num(series: Any) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def read_market(spec: MarketSpec) -> pd.DataFrame:
    if not spec.input_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(spec.input_path, low_memory=False)
    df["market"] = spec.market
    df["display_market"] = spec.display_market
    if "league_tag" not in df.columns:
        df["league_tag"] = np.nan
    if "league" in df.columns:
        league_from_name = df["league"].fillna("").astype(str).str.replace(" ", "_", regex=False)
        df["league_tag"] = df["league_tag"].where(df["league_tag"].notna() & df["league_tag"].astype(str).ne(""), league_from_name)
    if "match_date" in df.columns:
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    if "season_tag" not in df.columns and "match_date" in df.columns:
        df["season_tag"] = df["match_date"].dt.year
    df[spec.hit_col] = num(df[spec.hit_col])
    return df.dropna(subset=[spec.hit_col]).copy()


def available_family_features(df: pd.DataFrame, family: str) -> list[str]:
    return [col for col in FEATURE_FAMILIES[family] if col in df.columns]


def numeric_feature_screen(df: pd.DataFrame, spec: MarketSpec) -> pd.DataFrame:
    rows = []
    y = df[spec.hit_col].astype(float)
    feature_to_family = {
        feature: family
        for family, features in FEATURE_FAMILIES.items()
        for feature in features
    }
    candidates = sorted(set(feature_to_family).intersection(df.columns))
    for feature in candidates:
        x = num(df[feature])
        valid = pd.DataFrame({"x": x, "y": y}).dropna()
        if len(valid) < MIN_FEATURE_ROWS or valid["x"].nunique() < 5 or valid["y"].nunique() < 2:
            continue
        q20 = valid["x"].quantile(0.20)
        q80 = valid["x"].quantile(0.80)
        low = valid[valid["x"].le(q20)]
        high = valid[valid["x"].ge(q80)]
        if low.empty or high.empty:
            continue
        corr = valid["x"].corr(valid["y"])
        try:
            auc = roc_auc_score(valid["y"], valid["x"])
            auc_oriented = max(float(auc), 1.0 - float(auc))
        except Exception:
            auc_oriented = np.nan
        rows.append(
            {
                "market": spec.market,
                "display_market": spec.display_market,
                "feature_family": feature_to_family[feature],
                "feature": feature,
                "rows": int(len(valid)),
                "coverage": float(len(valid) / len(df)),
                "corr_with_hit": float(corr) if not pd.isna(corr) else np.nan,
                "abs_corr": float(abs(corr)) if not pd.isna(corr) else np.nan,
                "bottom_quintile_hit": float(low["y"].mean()),
                "top_quintile_hit": float(high["y"].mean()),
                "top_minus_bottom_hit": float(high["y"].mean() - low["y"].mean()),
                "abs_top_bottom_lift": float(abs(high["y"].mean() - low["y"].mean())),
                "oriented_auc": auc_oriented,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["market", "abs_top_bottom_lift"], ascending=[True, False])


def categorical_feature_screen(df: pd.DataFrame, spec: MarketSpec) -> pd.DataFrame:
    rows = []
    categorical_features = [
        "league_tag",
        "competition",
        "season_tag",
        "player_team_side",
        "position_group",
        "tactical_role",
        "team_formation",
        "opponent_formation",
        "formation_matchup_label",
        "fixture_style_label",
        "fixture_attacking_style_label",
        "player_form_tier",
        "og_goal_environment_label",
    ]
    y = df[spec.hit_col].astype(float)
    baseline = float(y.mean())
    for feature in [c for c in categorical_features if c in df.columns]:
        work = pd.DataFrame({"category": df[feature].fillna("UNKNOWN").astype(str), "y": y}).dropna()
        if len(work) < MIN_FEATURE_ROWS or work["category"].nunique() < 2:
            continue
        grouped = (
            work.groupby("category", dropna=False)
            .agg(rows=("y", "size"), hit_rate=("y", "mean"))
            .reset_index()
        )
        grouped = grouped[grouped["rows"].ge(max(50, len(work) * 0.01))].copy()
        if grouped.empty:
            continue
        top = grouped.sort_values(["hit_rate", "rows"], ascending=[False, False]).head(1).iloc[0]
        bottom = grouped.sort_values(["hit_rate", "rows"], ascending=[True, False]).head(1).iloc[0]
        rows.append(
            {
                "market": spec.market,
                "display_market": spec.display_market,
                "feature": feature,
                "rows": int(len(work)),
                "categories": int(work["category"].nunique()),
                "baseline_hit": baseline,
                "top_category": top["category"],
                "top_category_rows": int(top["rows"]),
                "top_category_hit": float(top["hit_rate"]),
                "bottom_category": bottom["category"],
                "bottom_category_rows": int(bottom["rows"]),
                "bottom_category_hit": float(bottom["hit_rate"]),
                "top_minus_bottom_hit": float(top["hit_rate"] - bottom["hit_rate"]),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["market", "top_minus_bottom_hit"], ascending=[True, False])


def family_summary(screen: pd.DataFrame, spec: MarketSpec) -> pd.DataFrame:
    rows = []
    for family, features in FEATURE_FAMILIES.items():
        available = [f for f in features if f in screen["feature"].values] if not screen.empty else []
        family_screen = screen[screen["feature_family"].eq(family)].copy() if not screen.empty else pd.DataFrame()
        rows.append(
            {
                "market": spec.market,
                "display_market": spec.display_market,
                "feature_family": family,
                "requested_features": len(features),
                "screened_numeric_features": int(len(family_screen)),
                "mean_abs_lift": float(family_screen["abs_top_bottom_lift"].mean()) if not family_screen.empty else np.nan,
                "max_abs_lift": float(family_screen["abs_top_bottom_lift"].max()) if not family_screen.empty else np.nan,
                "top_feature": family_screen.iloc[0]["feature"] if not family_screen.empty else "",
                "top_feature_lift": float(family_screen.iloc[0]["top_minus_bottom_hit"]) if not family_screen.empty else np.nan,
                "available_screened_features": "|".join(available),
            }
        )
    return pd.DataFrame(rows).sort_values(["market", "max_abs_lift"], ascending=[True, False])


def model_importance(df: pd.DataFrame, spec: MarketSpec, screen: pd.DataFrame) -> pd.DataFrame:
    if screen.empty or "season_tag" not in df.columns:
        return pd.DataFrame()
    numeric_features = (
        screen.sort_values("abs_top_bottom_lift", ascending=False)["feature"]
        .drop_duplicates()
        .head(MAX_MODEL_FEATURES)
        .tolist()
    )
    categorical_features = [
        c
        for c in ["league_tag", "player_team_side", "position_group", "tactical_role", "formation_matchup_label"]
        if c in df.columns
    ]
    features = numeric_features + categorical_features
    work = df[features + [spec.hit_col, "season_tag"]].copy()
    for feature in numeric_features:
        work[feature] = num(work[feature])
    work = work.dropna(subset=[spec.hit_col, "season_tag"])
    if len(work) < 1500 or work[spec.hit_col].nunique() < 2:
        return pd.DataFrame()
    seasons = sorted(num(work["season_tag"]).dropna().unique())
    if len(seasons) < 2:
        return pd.DataFrame()
    test_season = seasons[-1]
    train = work[num(work["season_tag"]).lt(test_season)].copy()
    test = work[num(work["season_tag"]).eq(test_season)].copy()
    if len(train) < 1000 or len(test) < 300 or train[spec.hit_col].nunique() < 2 or test[spec.hit_col].nunique() < 2:
        return pd.DataFrame()
    transformer = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
        ],
        remainder="drop",
    )
    model = Pipeline(
        [
            ("prep", transformer),
            ("logit", LogisticRegression(max_iter=1000, C=0.5, solver="liblinear")),
        ]
    )
    try:
        model.fit(train[features], train[spec.hit_col].astype(int))
        result = permutation_importance(
            model,
            test[features],
            test[spec.hit_col].astype(int),
            scoring="roc_auc",
            n_repeats=5,
            random_state=42,
        )
    except Exception:
        return pd.DataFrame()
    rows = []
    for feature, mean, std in zip(features, result.importances_mean, result.importances_std):
        rows.append(
            {
                "market": spec.market,
                "display_market": spec.display_market,
                "test_season": int(test_season),
                "feature": feature,
                "permutation_auc_drop_mean": float(mean),
                "permutation_auc_drop_std": float(std),
            }
        )
    return pd.DataFrame(rows).sort_values("permutation_auc_drop_mean", ascending=False)


def segment_winners(df: pd.DataFrame, spec: MarketSpec, screen: pd.DataFrame, segment_col: str) -> pd.DataFrame:
    if screen.empty or segment_col not in df.columns:
        return pd.DataFrame()
    top_features = screen.sort_values("abs_top_bottom_lift", ascending=False)["feature"].drop_duplicates().head(30).tolist()
    rows = []
    for segment, group in df.groupby(segment_col, dropna=False):
        if len(group) < 500 or group[spec.hit_col].nunique() < 2:
            continue
        local = numeric_feature_screen(group, spec)
        if local.empty:
            continue
        local = local[local["feature"].isin(top_features)].copy()
        if local.empty:
            continue
        best = local.sort_values("abs_top_bottom_lift", ascending=False).head(1).iloc[0]
        rows.append(
            {
                "market": spec.market,
                "display_market": spec.display_market,
                "segment_type": segment_col,
                "segment": segment,
                "rows": int(len(group)),
                "baseline_hit": float(group[spec.hit_col].mean()),
                "top_feature": best["feature"],
                "top_feature_family": best["feature_family"],
                "top_minus_bottom_hit": float(best["top_minus_bottom_hit"]),
                "abs_top_bottom_lift": float(best["abs_top_bottom_lift"]),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["market", "abs_top_bottom_lift"], ascending=[True, False])


def variant_ablation_rows(spec: MarketSpec) -> pd.DataFrame:
    paths = [
        MARKET_AUDIT_DIR / f"player_events_goal_spine_{spec.market}_variant_metrics_common_oof.csv",
        TACKLES_AUDIT_DIR / "player_events_goal_spine_tackles_variant_metrics_common_oof.csv",
    ]
    for path in paths:
        if path.exists():
            df = pd.read_csv(path)
            df["market"] = spec.market
            df["display_market"] = spec.display_market
            df["source_file"] = str(path)
            return df
    return pd.DataFrame()


def feature_registry(df: pd.DataFrame, spec: MarketSpec, screen: pd.DataFrame, family: pd.DataFrame) -> pd.DataFrame:
    rows = []
    screened = set(screen["feature"]) if not screen.empty else set()
    for family_name, features in FEATURE_FAMILIES.items():
        for feature in features:
            status = "KEEP" if feature in screened else "MISSING"
            if feature in df.columns and feature not in screened:
                valid = num(df[feature]).notna().sum()
                status = "MAYBE" if valid >= MIN_FEATURE_ROWS else "DROP"
            rows.append(
                {
                    "market": spec.market,
                    "feature_family": family_name,
                    "feature": feature,
                    "status": status,
                    "available_in_rows": int(num(df[feature]).notna().sum()) if feature in df.columns else 0,
                    "note": "screened numeric feature" if feature in screened else "",
                }
            )
    for feature, reason in MISSING_FEATURES.items():
        rows.append(
            {
                "market": spec.market,
                "feature_family": "missing_or_future",
                "feature": feature,
                "status": "MISSING",
                "available_in_rows": 0,
                "note": reason,
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    work = df.head(max_rows)
    cols = list(work.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in work.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                value = round(value, 6)
            if pd.isna(value):
                value = ""
            values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    outdir: Path,
    numeric: pd.DataFrame,
    categorical: pd.DataFrame,
    family: pd.DataFrame,
    importance: pd.DataFrame,
    league: pd.DataFrame,
    role: pd.DataFrame,
    registry: pd.DataFrame,
) -> None:
    lines = [
        "# Player Event Feature Discovery Audit",
        "",
        "Research-only audit of available player-event feature signals.",
        "",
        "## Safety",
        "- No production model artifact written.",
        "- No deploy routing or tiers changed.",
        "- No priced player-prop odds created.",
        "",
        "## Strongest Numeric Signals",
        markdown_table(
            numeric[
                [
                    "market",
                    "feature_family",
                    "feature",
                    "rows",
                    "top_minus_bottom_hit",
                    "abs_top_bottom_lift",
                    "oriented_auc",
                ]
            ].sort_values(["market", "abs_top_bottom_lift"], ascending=[True, False]),
            max_rows=35,
        ),
        "",
        "## Feature Family Leaders",
        markdown_table(
            family[
                [
                    "market",
                    "feature_family",
                    "screened_numeric_features",
                    "mean_abs_lift",
                    "max_abs_lift",
                    "top_feature",
                    "top_feature_lift",
                ]
            ].sort_values(["market", "max_abs_lift"], ascending=[True, False]),
            max_rows=35,
        ),
        "",
        "## Categorical Shape",
        markdown_table(
            categorical[
                [
                    "market",
                    "feature",
                    "top_category",
                    "top_category_rows",
                    "top_category_hit",
                    "bottom_category",
                    "bottom_category_hit",
                    "top_minus_bottom_hit",
                ]
            ].sort_values(["market", "top_minus_bottom_hit"], ascending=[True, False]),
            max_rows=35,
        ),
        "",
        "## Permutation Importance",
        markdown_table(
            importance[
                ["market", "test_season", "feature", "permutation_auc_drop_mean", "permutation_auc_drop_std"]
            ].sort_values(["market", "permutation_auc_drop_mean"], ascending=[True, False]),
            max_rows=35,
        ),
        "",
        "## League Winners",
        markdown_table(league, max_rows=35),
        "",
        "## Role Winners",
        markdown_table(role, max_rows=35),
        "",
        "## Missing / Future Registry",
        markdown_table(
            registry[registry["status"].eq("MISSING")][["market", "feature_family", "feature", "note"]],
            max_rows=40,
        ),
        "",
        "## Read",
        "- Use this as a prioritization map, not a deploy decision.",
        "- Strong feature screens should be followed by chronological ablation, then live-shadow repeat QA.",
        "- Missing xG/xA/big-chance/touches-in-box fields are not currently in local API-Football normalized files.",
    ]
    (outdir / "PLAYER_EVENT_FEATURE_DISCOVERY_AUDIT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--markets",
        default="shots,shots_ge2,shots_ge3,shots_on_target,sot_ge2_attackers,sot_ge3_attackers,fouls_committed,tackles",
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    numeric_frames = []
    categorical_frames = []
    family_frames = []
    importance_frames = []
    league_frames = []
    role_frames = []
    registry_frames = []
    ablation_frames = []

    for market in [part.strip() for part in args.markets.split(",") if part.strip()]:
        if market not in MARKETS:
            raise SystemExit(f"Unsupported market: {market}")
        spec = MARKETS[market]
        df = read_market(spec)
        if df.empty:
            print(f"SKIP {market}: missing {spec.input_path}")
            continue
        numeric = numeric_feature_screen(df, spec)
        categorical = categorical_feature_screen(df, spec)
        family = family_summary(numeric, spec)
        importance = model_importance(df, spec, numeric)
        league = segment_winners(df, spec, numeric, "league_tag")
        role = segment_winners(df, spec, numeric, "tactical_role")
        registry = feature_registry(df, spec, numeric, family)
        ablation = variant_ablation_rows(spec)

        numeric_frames.append(numeric)
        categorical_frames.append(categorical)
        family_frames.append(family)
        importance_frames.append(importance)
        league_frames.append(league)
        role_frames.append(role)
        registry_frames.append(registry)
        ablation_frames.append(ablation)

    outputs = {
        "player_event_market_feature_screen.csv": pd.concat(numeric_frames, ignore_index=True) if numeric_frames else pd.DataFrame(),
        "player_event_categorical_feature_screen.csv": pd.concat(categorical_frames, ignore_index=True) if categorical_frames else pd.DataFrame(),
        "player_event_feature_family_summary.csv": pd.concat(family_frames, ignore_index=True) if family_frames else pd.DataFrame(),
        "player_event_model_permutation_importance.csv": pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame(),
        "player_event_league_feature_winners.csv": pd.concat(league_frames, ignore_index=True) if league_frames else pd.DataFrame(),
        "player_event_role_feature_winners.csv": pd.concat(role_frames, ignore_index=True) if role_frames else pd.DataFrame(),
        "player_event_feature_registry.csv": pd.concat(registry_frames, ignore_index=True) if registry_frames else pd.DataFrame(),
        "player_event_feature_family_ablation_metrics.csv": pd.concat(ablation_frames, ignore_index=True) if ablation_frames else pd.DataFrame(),
    }
    for filename, df in outputs.items():
        df.to_csv(args.outdir / filename, index=False)

    write_report(
        args.outdir,
        outputs["player_event_market_feature_screen.csv"],
        outputs["player_event_categorical_feature_screen.csv"],
        outputs["player_event_feature_family_summary.csv"],
        outputs["player_event_model_permutation_importance.csv"],
        outputs["player_event_league_feature_winners.csv"],
        outputs["player_event_role_feature_winners.csv"],
        outputs["player_event_feature_registry.csv"],
    )

    print(f"WROTE {args.outdir}")
    print(
        outputs["player_event_market_feature_screen.csv"][
            ["market", "feature_family", "feature", "abs_top_bottom_lift", "oriented_auc"]
        ]
        .sort_values(["market", "abs_top_bottom_lift"], ascending=[True, False])
        .groupby("market")
        .head(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
