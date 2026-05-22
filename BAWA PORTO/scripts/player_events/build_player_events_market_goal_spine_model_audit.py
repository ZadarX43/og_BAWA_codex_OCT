#!/usr/bin/env python3
"""
Research-only player-event goal-spine audit.

Builds a proof-aligned EPL/La Liga player-event feature table, then runs the
same chronological audit shape for fouls, shots, and shots on target.

No production artifacts, deploy tiers, or priced player-prop probabilities are
written. Outputs are beta/intelligence boards for manual research only.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = ROOT / "data_sources" / "api_football" / "features" / "player_events"
NORMALIZED_DIR = ROOT / "data_sources" / "api_football" / "normalized"
DEFAULT_GOAL_SPINE = (
    ROOT
    / "reports"
    / "2026-05-05"
    / "dominance_overlay_walkforward_audit_v2_fullweekly"
    / "dominance_overlay_walkforward_scored.csv"
)
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-06" / "player_events_market_goal_spine_model_audit"
DEFAULT_ATTACKER_RECENT_FORM = (
    ROOT
    / "reports"
    / "2026-05-06"
    / "player_attacker_recent_form_features"
    / "player_attacker_recent_form_features.csv"
)
DEFAULT_OPPONENT_ATTACK_ALLOWANCE = (
    ROOT
    / "reports"
    / "2026-05-06"
    / "player_event_opponent_attack_allowance_features"
    / "player_event_opponent_attack_allowance_features.csv"
)

DEFAULT_LEAGUES = ("England_Premier_League", "Spain_La_Liga")
DEFAULT_SEASONS = (2022, 2023, 2024)
DEFAULT_EVAL_SEASONS = (2023, 2024)

EPS = 1e-6

RESTORED_PHASE8H_COLS = [
    "p_meta_ou25",
    "cs_mass_over25",
    "mass_4plus_goals",
    "p_meta_btts",
    "cs_mass_btts_yes",
    "p00_est",
    "p_meta_ftr",
    "pick_side_margin_top3",
    "pick_side_mass_top3",
    "ftr_margin",
    "p_home_ge2",
    "p_away_ge2",
    "p_home_ge3",
    "p_away_ge3",
    "hw_hge2_combo_prob",
    "aw_age2_combo_prob",
]

PROOF_ALIGNED_GOAL_CONTEXT_COLS = [
    "og_pre_match_xg_home",
    "og_pre_match_xg_away",
    "og_xg_total",
    "og_xg_weaker_side",
    "og_btts_pre",
    "og_over25_pre",
    "og_snap_over25_avg",
    "og_home_power_rating",
    "og_away_power_rating",
    "og_power_gap_abs",
    "og_balance_score",
    "og_goal_environment_score",
    "og_battle_on_score",
    "og_goal_support_flag",
    "og_battle_on_flag",
    "team_power_rating",
    "opponent_power_rating",
    "team_power_edge",
    "starting_xi_attack_quality_score",
    "starting_xi_quality_edge",
]

MATCH_STYLE_COLS = [
    "fixture_foul_density_score",
    "fixture_tackle_density_score",
    "fixture_midfield_grind_score",
    "fixture_wide_duel_score",
    "fixture_attack_pressure_score",
    "fixture_corner_pressure_score",
    "fixture_territorial_stress_score",
    "formation_pressure_score",
    "match_stakes_score",
    "ref_cards_per_match",
    "ref_foul_to_card_ratio",
    "ref_dissent_strictness",
    "ref_timewasting_strictness",
    "opponent_possession_projection",
    "left_flank_dominance",
    "right_flank_dominance",
    "central_attack_dominance",
]

PLAYER_LAYER_COLS = [
    "expected_minutes_audit",
    "fouls_per90",
    "fouls_won_per90",
    "yellow_cards_per90",
    "booking_efficiency",
    "tackles_per90",
    "interceptions_per90",
    "ground_duel_loss_rate",
    "aerial_duel_loss_rate",
    "dribbles_faced_per90",
    "shots_per90",
    "shots_on_target_per90",
    "goals_per90",
    "assists_per90",
    "key_passes_per90",
    "player_form_rating_l5",
    "player_quality_score_l5",
    "pass_accuracy_pct_l5",
    "minutes_last_3_matches",
    "days_rest",
    "team_avg_fouls",
    "team_avg_yellows",
    "home_team_fouls_l5",
    "away_team_fouls_l5",
    "home_team_tackles_l5",
    "away_team_tackles_l5",
    "home_team_shots_l5",
    "away_team_shots_l5",
    "home_team_shots_on_goal_l5",
    "away_team_shots_on_goal_l5",
    "home_team_corners_for_l5",
    "away_team_corners_for_l5",
    "h2h_total_fouls_l5",
    "h2h_total_tackles_l5",
    "h2h_total_shots_l5",
    "h2h_total_shots_on_goal_l5",
]

ATTACKER_RECENT_FORM_COLS = [
    "attacker_recent_history_matches",
    "attacker_recent_apps_l5",
    "attacker_recent_starts_l5",
    "attacker_recent_minutes_l5",
    "attacker_recent_shots_l5",
    "attacker_recent_sot_l5",
    "attacker_recent_goals_l5",
    "attacker_recent_assists_l5",
    "attacker_recent_goal_contributions_l5",
    "attacker_recent_key_passes_l5",
    "attacker_recent_shots_per90_l5",
    "attacker_recent_sot_per90_l5",
    "attacker_recent_goals_per90_l5",
    "attacker_recent_assists_per90_l5",
    "attacker_recent_goal_contributions_per90_l5",
    "attacker_recent_key_passes_per90_l5",
    "attacker_recent_start_share_l5",
    "attacker_recent_sot_share_l5",
    "attacker_recent_apps_l8",
    "attacker_recent_starts_l8",
    "attacker_recent_minutes_l8",
    "attacker_recent_shots_l8",
    "attacker_recent_sot_l8",
    "attacker_recent_goals_l8",
    "attacker_recent_assists_l8",
    "attacker_recent_goal_contributions_l8",
    "attacker_recent_key_passes_l8",
    "attacker_recent_shots_per90_l8",
    "attacker_recent_sot_per90_l8",
    "attacker_recent_goals_per90_l8",
    "attacker_recent_assists_per90_l8",
    "attacker_recent_goal_contributions_per90_l8",
    "attacker_recent_key_passes_per90_l8",
    "attacker_recent_start_share_l8",
    "attacker_recent_sot_share_l8",
    "attacker_recent_home_shots_per90_l5",
    "attacker_recent_home_sot_per90_l5",
    "attacker_recent_away_shots_per90_l5",
    "attacker_recent_away_sot_per90_l5",
    "attacker_recent_home_shots_per90_l8",
    "attacker_recent_home_sot_per90_l8",
    "attacker_recent_away_shots_per90_l8",
    "attacker_recent_away_sot_per90_l8",
    "attacker_role_attack_flag",
]

OPPONENT_ATTACK_ALLOWANCE_COLS = [
    "opp_attack_allowed_role_source_matches",
    "opp_attack_allowed_attacker_any_source_matches",
    "opp_attack_allowed_role_matches_l5",
    "opp_attack_allowed_role_players_l5",
    "opp_attack_allowed_role_shots_l5",
    "opp_attack_allowed_role_sot_l5",
    "opp_attack_allowed_role_shots_per_match_l5",
    "opp_attack_allowed_role_sot_per_match_l5",
    "opp_attack_allowed_role_shots_per_player_l5",
    "opp_attack_allowed_role_sot_per_player_l5",
    "opp_attack_allowed_role_player_shot_ge1_rate_l5",
    "opp_attack_allowed_role_player_shot_ge2_rate_l5",
    "opp_attack_allowed_role_player_sot_ge1_rate_l5",
    "opp_attack_allowed_role_player_sot_ge2_rate_l5",
    "opp_attack_allowed_role_matches_l10",
    "opp_attack_allowed_role_players_l10",
    "opp_attack_allowed_role_shots_l10",
    "opp_attack_allowed_role_sot_l10",
    "opp_attack_allowed_role_shots_per_match_l10",
    "opp_attack_allowed_role_sot_per_match_l10",
    "opp_attack_allowed_role_shots_per_player_l10",
    "opp_attack_allowed_role_sot_per_player_l10",
    "opp_attack_allowed_role_player_shot_ge1_rate_l10",
    "opp_attack_allowed_role_player_shot_ge2_rate_l10",
    "opp_attack_allowed_role_player_sot_ge1_rate_l10",
    "opp_attack_allowed_role_player_sot_ge2_rate_l10",
    "opp_attack_allowed_attacker_any_matches_l5",
    "opp_attack_allowed_attacker_any_players_l5",
    "opp_attack_allowed_attacker_any_shots_l5",
    "opp_attack_allowed_attacker_any_sot_l5",
    "opp_attack_allowed_attacker_any_shots_per_match_l5",
    "opp_attack_allowed_attacker_any_sot_per_match_l5",
    "opp_attack_allowed_attacker_any_shots_per_player_l5",
    "opp_attack_allowed_attacker_any_sot_per_player_l5",
    "opp_attack_allowed_attacker_any_player_shot_ge1_rate_l5",
    "opp_attack_allowed_attacker_any_player_shot_ge2_rate_l5",
    "opp_attack_allowed_attacker_any_player_sot_ge1_rate_l5",
    "opp_attack_allowed_attacker_any_player_sot_ge2_rate_l5",
    "opp_attack_allowed_attacker_any_matches_l10",
    "opp_attack_allowed_attacker_any_players_l10",
    "opp_attack_allowed_attacker_any_shots_l10",
    "opp_attack_allowed_attacker_any_sot_l10",
    "opp_attack_allowed_attacker_any_shots_per_match_l10",
    "opp_attack_allowed_attacker_any_sot_per_match_l10",
    "opp_attack_allowed_attacker_any_shots_per_player_l10",
    "opp_attack_allowed_attacker_any_sot_per_player_l10",
    "opp_attack_allowed_attacker_any_player_shot_ge1_rate_l10",
    "opp_attack_allowed_attacker_any_player_shot_ge2_rate_l10",
    "opp_attack_allowed_attacker_any_player_sot_ge1_rate_l10",
    "opp_attack_allowed_attacker_any_player_sot_ge2_rate_l10",
]

CATEGORICAL_COLS = [
    "league_tag",
    "season_tag",
    "competition",
    "player_team_side",
    "position_group",
    "tactical_role",
    "formation_matchup_label",
    "fixture_style_label",
    "fixture_attacking_style_label",
    "player_form_tier",
    "og_goal_environment_label",
]


@dataclass(frozen=True)
class MarketConfig:
    name: str
    actual_col: str
    rate_col: str
    target_line: int
    row_filter_positions: tuple[str, ...] | None = None
    row_filter_tactical_roles: tuple[str, ...] | None = None


MARKETS = {
    "fouls_committed": MarketConfig(
        name="fouls_committed",
        actual_col="fouls_committed",
        rate_col="fouls_per90",
        target_line=2,
        row_filter_positions=("Defender", "Midfielder", "Forward"),
    ),
    "shots": MarketConfig(
        name="shots",
        actual_col="shots_total",
        rate_col="shots_per90",
        target_line=1,
        row_filter_positions=("Midfielder", "Forward"),
    ),
    "shots_ge2": MarketConfig(
        name="shots_ge2",
        actual_col="shots_total",
        rate_col="shots_per90",
        target_line=2,
        row_filter_positions=("Midfielder", "Forward"),
    ),
    "shots_ge3": MarketConfig(
        name="shots_ge3",
        actual_col="shots_total",
        rate_col="shots_per90",
        target_line=3,
        row_filter_positions=("Midfielder", "Forward"),
    ),
    "shots_on_target": MarketConfig(
        name="shots_on_target",
        actual_col="shots_on_target",
        rate_col="shots_on_target_per90",
        target_line=1,
        row_filter_positions=("Midfielder", "Forward"),
    ),
    "sot_ge2_attackers": MarketConfig(
        name="sot_ge2_attackers",
        actual_col="shots_on_target",
        rate_col="shots_on_target_per90",
        target_line=2,
        row_filter_positions=("Midfielder", "Forward"),
        row_filter_tactical_roles=("Central striker", "Wide forward", "Wide midfielder / winger"),
    ),
    "sot_ge3_attackers": MarketConfig(
        name="sot_ge3_attackers",
        actual_col="shots_on_target",
        rate_col="shots_on_target_per90",
        target_line=3,
        row_filter_positions=("Midfielder", "Forward"),
        row_filter_tactical_roles=("Central striker", "Wide forward", "Wide midfielder / winger"),
    ),
}


def norm_text(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    aliases = {"cf": "", "fc": "", "sc": "", "afc": "", "cd": "", "ud": ""}
    text = " ".join(aliases.get(part, part) for part in text.split())
    return re.sub(r"\s+", " ", text).strip()


def num(series: Any) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def logit(series: pd.Series) -> pd.Series:
    values = num(series).clip(EPS, 1 - EPS)
    return np.log(values / (1 - values))


def poisson_survival(lam: pd.Series, target_line: int) -> pd.Series:
    lam = num(lam).fillna(0.0).clip(lower=0.0, upper=20.0)
    cdf = pd.Series(0.0, index=lam.index, dtype=float)
    for k in range(target_line):
        cdf = cdf + np.exp(-lam) * np.power(lam, k) / math.factorial(k)
    return (1.0 - cdf).clip(EPS, 1 - EPS)


def make_join_key(df: pd.DataFrame) -> pd.Series:
    date = pd.to_datetime(df["match_date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    home = df["home_team_name"].map(norm_text)
    away = df["away_team_name"].map(norm_text)
    return date + "__" + home + "__" + away


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def load_fixture_inputs(leagues: tuple[str, ...], seasons: tuple[int, ...]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for league in leagues:
        for season in seasons:
            path = FEATURES_DIR / f"player_events_fixture_input__{league}__{season}.csv"
            df = read_csv_if_exists(path)
            if df.empty:
                continue
            df["league_tag"] = league
            df["season_tag"] = season
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["match_date"] = pd.to_datetime(out["match_date"], errors="coerce")
    out["expected_minutes_audit"] = num(out.get("expected_minutes", np.nan))
    return out


def load_actuals(leagues: tuple[str, ...], seasons: tuple[int, ...]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for league in leagues:
        for season in seasons:
            fixtures_path = NORMALIZED_DIR / f"fixtures_master__{league}__{season}.csv"
            stats_path = NORMALIZED_DIR / f"match_player_stats__{league}__{season}.csv"
            fixtures = read_csv_if_exists(fixtures_path)
            stats = read_csv_if_exists(stats_path)
            if fixtures.empty or stats.empty:
                continue
            fixture_cols = [
                "fixture_id",
                "fixture_key",
                "home_team_id",
                "away_team_id",
                "home_team_name",
                "away_team_name",
                "match_date",
                "kickoff_ts_utc",
            ]
            merged = stats.merge(fixtures[fixture_cols], on="fixture_id", how="left")
            merged["team_name"] = np.where(
                num(merged["team_id"]).eq(num(merged["home_team_id"])),
                merged["home_team_name"],
                merged["away_team_name"],
            )
            merged["league_tag"] = league
            merged["season_tag"] = season
            frames.append(merged)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["match_date"] = pd.to_datetime(out["match_date"], errors="coerce")
    out["kickoff_ts_utc"] = pd.to_datetime(out["kickoff_ts_utc"], errors="coerce", utc=True)
    out["actual_minutes"] = num(out.get("minutes", np.nan))
    return out


def load_restored_goal_spine(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["_fixture_join_key", "restored_goal_spine_feature_count"])
    header = pd.read_csv(path, nrows=0)
    base_cols = ["fixture_key", "match_date", "league", "home_team_name", "away_team_name", "market"]
    available = [c for c in base_cols + RESTORED_PHASE8H_COLS if c in header.columns]
    raw = pd.read_csv(path, usecols=available, low_memory=False)
    raw["_fixture_join_key"] = make_join_key(raw)
    for col in RESTORED_PHASE8H_COLS:
        if col not in raw.columns:
            raw[col] = np.nan
        raw[col] = num(raw[col])
    aggregations = {col: "max" for col in RESTORED_PHASE8H_COLS}
    aggregations["fixture_key"] = "first"
    aggregations["market"] = lambda s: "|".join(sorted(set(s.dropna().astype(str))))
    out = raw.groupby("_fixture_join_key", dropna=False).agg(aggregations).reset_index()
    out = out.rename(columns={"fixture_key": "restored_goal_fixture_key", "market": "restored_goal_markets_present"})
    out["restored_goal_spine_feature_count"] = out[RESTORED_PHASE8H_COLS].notna().sum(axis=1)
    return out


def load_attacker_recent_form(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    header = pd.read_csv(path, nrows=0)
    key_cols = ["fixture_id", "team_id", "player_id", "league_tag", "season_tag"]
    available = [c for c in key_cols + ATTACKER_RECENT_FORM_COLS if c in header.columns]
    recent = pd.read_csv(path, usecols=available, low_memory=False)
    for col in key_cols:
        if col in recent.columns:
            recent[col] = num(recent[col])
    return recent


def load_opponent_attack_allowance(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    header = pd.read_csv(path, nrows=0)
    key_cols = ["fixture_id", "team_id", "player_id", "league_tag", "season_tag"]
    available = [c for c in key_cols + OPPONENT_ATTACK_ALLOWANCE_COLS if c in header.columns]
    opp = pd.read_csv(path, usecols=available, low_memory=False)
    for col in key_cols:
        if col in opp.columns:
            opp[col] = num(opp[col])
    return opp


def build_feature_table(
    leagues: tuple[str, ...],
    seasons: tuple[int, ...],
    restored_goal_spine_path: Path,
    attacker_recent_form_path: Path,
    opponent_attack_allowance_path: Path,
) -> pd.DataFrame:
    inputs = load_fixture_inputs(leagues, seasons)
    actuals = load_actuals(leagues, seasons)
    if inputs.empty:
        return inputs

    actual_cols = [
        "fixture_key",
        "team_name",
        "player_name",
        "league_tag",
        "season_tag",
        "fixture_id",
        "team_id",
        "player_id",
        "actual_minutes",
        "started_flag",
        "subbed_on_flag",
        "fouls_committed",
        "shots_total",
        "shots_on_target",
        "tackles",
        "interceptions",
        "duels_total",
        "yellow_cards",
    ]
    available_actual_cols = [c for c in actual_cols if c in actuals.columns]
    joined = inputs.merge(
        actuals[available_actual_cols],
        on=["fixture_key", "team_name", "player_name", "league_tag", "season_tag"],
        how="left",
        suffixes=("", "_actual"),
    )
    joined["_fixture_join_key"] = make_join_key(joined)
    restored = load_restored_goal_spine(restored_goal_spine_path)
    joined = joined.merge(restored, on="_fixture_join_key", how="left")
    joined["restored_goal_spine_matched"] = joined["restored_goal_fixture_key"].notna().astype(int)
    joined["proof_aligned_goal_feature_count"] = joined[[c for c in PROOF_ALIGNED_GOAL_CONTEXT_COLS if c in joined.columns]].notna().sum(axis=1)
    joined["proof_aligned_goal_feature_any"] = joined["proof_aligned_goal_feature_count"].gt(0).astype(int)
    joined["proof_aligned_goal_feature_all"] = joined["proof_aligned_goal_feature_count"].eq(
        len([c for c in PROOF_ALIGNED_GOAL_CONTEXT_COLS if c in joined.columns])
    ).astype(int)
    recent = load_attacker_recent_form(attacker_recent_form_path)
    if not recent.empty:
        for col in ["fixture_id", "team_id", "player_id", "league_tag", "season_tag"]:
            if col in joined.columns:
                joined[col] = num(joined[col])
        joined = joined.merge(
            recent,
            on=["fixture_id", "team_id", "player_id", "league_tag", "season_tag"],
            how="left",
        )
    opp_allowance = load_opponent_attack_allowance(opponent_attack_allowance_path)
    if not opp_allowance.empty:
        for col in ["fixture_id", "team_id", "player_id", "league_tag", "season_tag"]:
            if col in joined.columns:
                joined[col] = num(joined[col])
        joined = joined.merge(
            opp_allowance,
            on=["fixture_id", "team_id", "player_id", "league_tag", "season_tag"],
            how="left",
        )
    joined["attacker_role_attack_flag"] = (
        joined.get("tactical_role", "")
        .astype(str)
        .isin(["Central striker", "Wide forward", "Wide midfielder / winger"])
        .astype(int)
    )
    joined["eval_month"] = pd.to_datetime(joined["match_date"], errors="coerce").dt.to_period("M").astype(str)
    joined = joined.sort_values(["match_date", "fixture_key", "team_name", "player_name"]).reset_index(drop=True)
    return joined


def ece_score(y_true: pd.Series, y_prob: pd.Series, bins: int = 10) -> float:
    work = pd.DataFrame({"y": y_true, "p": y_prob}).dropna()
    if work.empty:
        return np.nan
    work["bin"] = pd.cut(work["p"].clip(0, 1), np.linspace(0, 1, bins + 1), include_lowest=True)
    total = len(work)
    ece = 0.0
    for _, group in work.groupby("bin", observed=False):
        if group.empty:
            continue
        ece += len(group) / total * abs(group["p"].mean() - group["y"].mean())
    return float(ece)


def top_decile_precision(y_true: pd.Series, y_prob: pd.Series) -> float:
    work = pd.DataFrame({"y": y_true, "p": y_prob}).dropna()
    if work.empty:
        return np.nan
    n = max(1, math.ceil(len(work) * 0.10))
    return float(work.nlargest(n, "p")["y"].mean())


def metric_row(name: str, y_true: pd.Series, y_prob: pd.Series, total_rows: int | None = None) -> dict[str, Any]:
    total_rows = total_rows or len(y_true)
    work = pd.DataFrame({"y": y_true, "p": y_prob}).dropna()
    if work.empty or work["y"].nunique() < 2:
        return {
            "variant": name,
            "rows": int(len(work)),
            "coverage": float(len(work) / total_rows) if total_rows else 0.0,
            "hit_rate": np.nan,
            "mean_pred": np.nan,
            "brier": np.nan,
            "logloss": np.nan,
            "ece_10bin": np.nan,
            "top_decile_precision": np.nan,
        }
    probs = work["p"].clip(EPS, 1 - EPS)
    return {
        "variant": name,
        "rows": int(len(work)),
        "coverage": float(len(work) / total_rows) if total_rows else 0.0,
        "hit_rate": float(work["y"].mean()),
        "mean_pred": float(probs.mean()),
        "brier": float(brier_score_loss(work["y"], probs)),
        "logloss": float(log_loss(work["y"], probs)),
        "ece_10bin": ece_score(work["y"], probs),
        "top_decile_precision": top_decile_precision(work["y"], probs),
    }


def build_feature_matrix(df: pd.DataFrame, feature_cols: list[str], categorical_cols: list[str]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    cats = [c for c in categorical_cols if c in df.columns and c in feature_cols]
    numeric_cols = [c for c in feature_cols if c in df.columns and c not in cats]
    if numeric_cols:
        pieces.append(df[numeric_cols].apply(num))
    if cats:
        pieces.append(pd.get_dummies(df[cats].fillna("UNKNOWN").astype(str), prefix=cats, dummy_na=False))
    if not pieces:
        return pd.DataFrame(index=df.index)
    return pd.concat(pieces, axis=1)


def temporal_logistic_oof(
    df: pd.DataFrame,
    y_col: str,
    feature_cols: list[str],
    categorical_cols: list[str] | None = None,
    min_train_rows: int = 2500,
) -> pd.Series:
    categorical_cols = categorical_cols or []
    X_all = build_feature_matrix(df, feature_cols, categorical_cols)
    y = df[y_col].astype(int)
    preds = pd.Series(np.nan, index=df.index, dtype=float)
    months = sorted(df["eval_month"].dropna().unique())

    for month in months:
        test_idx = df.index[df["eval_month"].eq(month)]
        if len(test_idx) == 0:
            continue
        first_test_date = df.loc[test_idx, "match_date"].min()
        train_idx = df.index[df["match_date"].lt(first_test_date)]
        train_idx = train_idx[y.loc[train_idx].notna()]
        if len(train_idx) < min_train_rows or y.loc[train_idx].nunique() < 2:
            continue
        x_train = X_all.loc[train_idx]
        x_test = X_all.loc[test_idx]
        non_empty_cols = x_train.columns[x_train.notna().any()]
        x_train = x_train[non_empty_cols]
        x_test = x_test.reindex(columns=non_empty_cols)
        if x_train.shape[1] == 0:
            continue
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
                ("logit", LogisticRegression(max_iter=1000, C=0.5, solver="liblinear")),
            ]
        )
        try:
            model.fit(x_train, y.loc[train_idx])
            preds.loc[test_idx] = model.predict_proba(x_test)[:, 1]
        except Exception:
            continue
    return preds.clip(0, 1)


def temporal_isotonic_oof(
    df: pd.DataFrame,
    y_col: str,
    prob_col: str,
    min_train_rows: int = 2500,
) -> pd.Series:
    y = df[y_col].astype(int)
    p = num(df[prob_col]).clip(EPS, 1 - EPS)
    preds = pd.Series(np.nan, index=df.index, dtype=float)
    months = sorted(df["eval_month"].dropna().unique())
    for month in months:
        test_idx = df.index[df["eval_month"].eq(month)]
        if len(test_idx) == 0:
            continue
        first_test_date = df.loc[test_idx, "match_date"].min()
        train_idx = df.index[df["match_date"].lt(first_test_date)]
        train = pd.DataFrame({"p": p.loc[train_idx], "y": y.loc[train_idx]}).dropna()
        if len(train) < min_train_rows or train["y"].nunique() < 2:
            continue
        try:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(train["p"], train["y"])
            preds.loc[test_idx] = iso.predict(p.loc[test_idx])
        except Exception:
            continue
    return preds.clip(0, 1)


def feature_family_coverage(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    families = {
        "proof_aligned_goal_context": PROOF_ALIGNED_GOAL_CONTEXT_COLS,
        "restored_phase8h_goal_spine": RESTORED_PHASE8H_COLS,
        "match_style": MATCH_STYLE_COLS,
        "player_layer": PLAYER_LAYER_COLS,
        "attacker_recent_form": ATTACKER_RECENT_FORM_COLS,
        "opponent_attack_allowance": OPPONENT_ATTACK_ALLOWANCE_COLS,
    }
    for family, cols in families.items():
        available = [c for c in cols if c in df.columns]
        rows.append(
            {
                "feature_family": family,
                "requested_features": len(cols),
                "available_features": len(available),
                "rows_with_any_feature": int(df[available].notna().any(axis=1).sum()) if available else 0,
                "rows_with_all_features": int(df[available].notna().all(axis=1).sum()) if available else 0,
                "coverage_any": float(df[available].notna().any(axis=1).mean()) if available else 0.0,
                "coverage_all": float(df[available].notna().all(axis=1).mean()) if available else 0.0,
                "available_feature_names": "|".join(available),
            }
        )
    return pd.DataFrame(rows)


def feature_screen(df: pd.DataFrame, y_col: str, cols: list[str]) -> pd.DataFrame:
    rows = []
    y = df[y_col].astype(float)
    for col in cols:
        if col not in df.columns:
            continue
        x = num(df[col])
        valid = pd.DataFrame({"x": x, "y": y}).dropna()
        if len(valid) < 100 or valid["y"].nunique() < 2:
            continue
        q80 = valid["x"].quantile(0.80)
        q20 = valid["x"].quantile(0.20)
        rows.append(
            {
                "feature": col,
                "rows": int(len(valid)),
                "coverage": float(len(valid) / len(df)),
                "corr_with_hit": float(valid["x"].corr(valid["y"])) if valid["x"].nunique() > 1 else np.nan,
                "bottom_quintile_hit": float(valid[valid["x"].le(q20)]["y"].mean()),
                "top_quintile_hit": float(valid[valid["x"].ge(q80)]["y"].mean()),
                "top_minus_bottom_hit": float(
                    valid[valid["x"].ge(q80)]["y"].mean() - valid[valid["x"].le(q20)]["y"].mean()
                ),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("top_minus_bottom_hit", ascending=False)


def run_market_audit(feature_table: pd.DataFrame, config: MarketConfig, min_train_rows: int) -> dict[str, pd.DataFrame]:
    df = feature_table.copy()
    if config.row_filter_positions:
        df = df[df["position_group"].astype(str).isin(config.row_filter_positions)].copy()
    if config.row_filter_tactical_roles:
        df = df[df["tactical_role"].astype(str).isin(config.row_filter_tactical_roles)].copy()
    df[config.actual_col] = num(df.get(config.actual_col, np.nan))
    df["expected_minutes_audit"] = num(df.get("expected_minutes_audit", np.nan))
    df[config.rate_col] = num(df.get(config.rate_col, np.nan))
    df = df.dropna(subset=["match_date", config.actual_col, "expected_minutes_audit", config.rate_col]).copy()
    df = df[df["expected_minutes_audit"].ge(30.0)].copy()
    df[f"actual_hit_ge{config.target_line}"] = df[config.actual_col].ge(config.target_line).astype(int)
    df["raw_lambda"] = (df[config.rate_col].clip(lower=0.0) * df["expected_minutes_audit"].clip(lower=0.0) / 90.0).clip(0, 20)
    df["raw_rate_poisson_p"] = poisson_survival(df["raw_lambda"], config.target_line)
    df["raw_rate_logit"] = logit(df["raw_rate_poisson_p"])
    y_col = f"actual_hit_ge{config.target_line}"

    variants: dict[str, pd.Series] = {
        "RAW_RATE_POISSON": num(df["raw_rate_poisson_p"]),
        "RATE_ISOTONIC_OOF": temporal_isotonic_oof(df, y_col, "raw_rate_poisson_p", min_train_rows=min_train_rows),
        "RATE_LOGISTIC_CAL_OOF": temporal_logistic_oof(df, y_col, ["raw_rate_logit"], min_train_rows=min_train_rows),
        "RATE_PLUS_PROOF_ALIGNED_GOAL_CONTEXT": temporal_logistic_oof(
            df,
            y_col,
            ["raw_rate_logit"] + PROOF_ALIGNED_GOAL_CONTEXT_COLS,
            min_train_rows=min_train_rows,
        ),
        "RATE_PLUS_RESTORED_PHASE8H_GOAL_SPINE": temporal_logistic_oof(
            df,
            y_col,
            ["raw_rate_logit"] + RESTORED_PHASE8H_COLS,
            min_train_rows=min_train_rows,
        ),
        "RATE_PLUS_MATCH_STYLE": temporal_logistic_oof(
            df,
            y_col,
            ["raw_rate_logit"] + MATCH_STYLE_COLS,
            min_train_rows=min_train_rows,
        ),
        "RATE_PLUS_PLAYER_LAYER": temporal_logistic_oof(
            df,
            y_col,
            ["raw_rate_logit"] + PLAYER_LAYER_COLS,
            min_train_rows=min_train_rows,
        ),
        "RATE_PLUS_ATTACKER_RECENT_FORM": temporal_logistic_oof(
            df,
            y_col,
            ["raw_rate_logit"] + ATTACKER_RECENT_FORM_COLS,
            min_train_rows=min_train_rows,
        ),
        "RATE_PLUS_OPPONENT_ATTACK_ALLOWANCE": temporal_logistic_oof(
            df,
            y_col,
            ["raw_rate_logit"] + OPPONENT_ATTACK_ALLOWANCE_COLS,
            min_train_rows=min_train_rows,
        ),
        "RATE_PLUS_ALL_CONTEXT": temporal_logistic_oof(
            df,
            y_col,
            ["raw_rate_logit"]
            + PROOF_ALIGNED_GOAL_CONTEXT_COLS
            + RESTORED_PHASE8H_COLS
            + MATCH_STYLE_COLS
            + PLAYER_LAYER_COLS
            + ATTACKER_RECENT_FORM_COLS
            + OPPONENT_ATTACK_ALLOWANCE_COLS
            + CATEGORICAL_COLS,
            categorical_cols=CATEGORICAL_COLS,
            min_train_rows=min_train_rows,
        ),
    }

    scored = df.copy()
    for name, pred in variants.items():
        scored[f"pred_{name.lower()}"] = pred

    metrics = pd.DataFrame(
        [metric_row(name, scored[y_col], pred, total_rows=len(scored)) for name, pred in variants.items()]
    ).sort_values(["coverage", "brier"], ascending=[False, True])

    common_mask = variants["RATE_LOGISTIC_CAL_OOF"].notna()
    metrics_common = pd.DataFrame(
        [
            metric_row(name, scored.loc[common_mask, y_col], pred.loc[common_mask], total_rows=int(common_mask.sum()))
            for name, pred in variants.items()
        ]
    ).sort_values("brier", ascending=True)

    restored_mask = scored["restored_goal_spine_matched"].eq(1)
    metrics_restored = pd.DataFrame(
        [
            metric_row(
                name,
                scored.loc[restored_mask, y_col],
                pred.loc[restored_mask],
                total_rows=int(restored_mask.sum()),
            )
            for name, pred in variants.items()
        ]
    ).sort_values(["coverage", "brier"], ascending=[False, True])

    screen = feature_screen(
        scored,
        y_col,
        PROOF_ALIGNED_GOAL_CONTEXT_COLS
        + RESTORED_PHASE8H_COLS
        + MATCH_STYLE_COLS
        + PLAYER_LAYER_COLS
        + ATTACKER_RECENT_FORM_COLS
        + OPPONENT_ATTACK_ALLOWANCE_COLS,
    )
    return {
        "scored": scored,
        "metrics": metrics,
        "metrics_common_oof": metrics_common,
        "metrics_restored_matched": metrics_restored,
        "feature_screen": screen,
    }


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    work = df.head(max_rows) if max_rows is not None else df
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
    feature_table: pd.DataFrame,
    coverage: pd.DataFrame,
    market_outputs: dict[str, dict[str, pd.DataFrame]],
    restored_goal_spine_path: Path,
    eval_seasons: tuple[int, ...],
) -> None:
    eval_table = feature_table[feature_table["season_tag"].isin(eval_seasons)].copy()
    lines = [
        "# Player Events Market Goal-Spine Model Audit",
        "",
        "Research-only beta/intelligence audit for fouls, shots, and shots on target.",
        "",
        "## Safety",
        "- No production model artifacts written.",
        "- No deploy routing, tiers, slips, or live picks changed.",
        "- Outputs are not priced player-prop odds; they are calibration research boards.",
        "",
        "## Proof-Aligned Feature Table",
        f"- all feature rows: `{len(feature_table)}`",
        f"- eval-season rows ({', '.join(map(str, eval_seasons))}): `{len(eval_table)}`",
        f"- proof-aligned goal context any coverage: `{eval_table['proof_aligned_goal_feature_any'].mean():.2%}`",
        f"- proof-aligned goal context all coverage: `{eval_table['proof_aligned_goal_feature_all'].mean():.2%}`",
        f"- restored Phase 8H goal-spine match rate: `{eval_table['restored_goal_spine_matched'].mean():.2%}`",
        f"- restored Phase 8H source: `{restored_goal_spine_path}`",
        "",
        "## Feature-Family Coverage",
        markdown_table(coverage),
        "",
        "## Market Reads",
    ]
    for market, outputs in market_outputs.items():
        lines.extend(
            [
                "",
                f"### {market}",
                "",
                "Full coverage:",
                markdown_table(outputs["metrics"]),
                "",
                "Common OOF rows:",
                markdown_table(outputs["metrics_common_oof"]),
                "",
                "Restored Phase 8H matched rows:",
                markdown_table(outputs["metrics_restored_matched"]),
                "",
                "Top feature screen:",
                markdown_table(outputs["feature_screen"], max_rows=15),
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "- Treat proof-aligned goal context as the fair near-full-coverage test.",
            "- Treat restored Phase 8H goal-spine results as directional only until coverage is rebuilt by fixture.",
            "- Prefer Brier/logloss improvements for calibration, and top-decile precision for shortlist usefulness.",
            "- Cards remain a separate hazard model; do not copy count-rate logic blindly.",
        ]
    )
    (outdir / "PLAYER_EVENTS_MARKET_GOAL_SPINE_MODEL_AUDIT.md").write_text("\n".join(lines) + "\n")


def parse_csv_tuple(value: str, cast=str) -> tuple:
    return tuple(cast(part.strip()) for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leagues", default=",".join(DEFAULT_LEAGUES))
    parser.add_argument("--seasons", default=",".join(map(str, DEFAULT_SEASONS)))
    parser.add_argument("--eval-seasons", default=",".join(map(str, DEFAULT_EVAL_SEASONS)))
    parser.add_argument(
        "--markets",
        default="fouls_committed,shots,shots_ge2,shots_ge3,shots_on_target,sot_ge2_attackers,sot_ge3_attackers",
    )
    parser.add_argument("--restored-goal-spine", type=Path, default=DEFAULT_GOAL_SPINE)
    parser.add_argument("--attacker-recent-form", type=Path, default=DEFAULT_ATTACKER_RECENT_FORM)
    parser.add_argument("--opponent-attack-allowance", type=Path, default=DEFAULT_OPPONENT_ATTACK_ALLOWANCE)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--min-train-rows", type=int, default=2500)
    args = parser.parse_args()

    leagues = parse_csv_tuple(args.leagues, str)
    seasons = parse_csv_tuple(args.seasons, int)
    eval_seasons = parse_csv_tuple(args.eval_seasons, int)
    markets = parse_csv_tuple(args.markets, str)

    args.outdir.mkdir(parents=True, exist_ok=True)
    feature_table = build_feature_table(
        leagues,
        seasons,
        args.restored_goal_spine,
        args.attacker_recent_form,
        args.opponent_attack_allowance,
    )
    if feature_table.empty:
        raise SystemExit("No feature table rows found.")

    feature_table_path = args.outdir / "player_events_goal_spine_feature_table_epl_laliga_2022_2024.csv"
    coverage_path = args.outdir / "player_events_goal_spine_feature_table_coverage.csv"
    feature_table.to_csv(feature_table_path, index=False)
    coverage = feature_family_coverage(feature_table)
    coverage.to_csv(coverage_path, index=False)

    market_outputs: dict[str, dict[str, pd.DataFrame]] = {}
    for market in markets:
        if market not in MARKETS:
            raise SystemExit(f"Unsupported market: {market}. Supported: {', '.join(MARKETS)}")
        outputs = run_market_audit(feature_table, MARKETS[market], args.min_train_rows)
        market_outputs[market] = outputs
        outputs["scored"].to_csv(args.outdir / f"player_events_goal_spine_{market}_joined_scored.csv", index=False)
        outputs["metrics"].to_csv(args.outdir / f"player_events_goal_spine_{market}_variant_metrics.csv", index=False)
        outputs["metrics_common_oof"].to_csv(
            args.outdir / f"player_events_goal_spine_{market}_variant_metrics_common_oof.csv",
            index=False,
        )
        outputs["metrics_restored_matched"].to_csv(
            args.outdir / f"player_events_goal_spine_{market}_variant_metrics_restored_matched.csv",
            index=False,
        )
        outputs["feature_screen"].to_csv(args.outdir / f"player_events_goal_spine_{market}_feature_screen.csv", index=False)

    write_report(args.outdir, feature_table, coverage, market_outputs, args.restored_goal_spine, eval_seasons)
    print(f"WROTE {args.outdir}")
    print(f"feature_rows={len(feature_table)}")
    print(coverage.to_string(index=False))
    for market, outputs in market_outputs.items():
        print()
        print(market)
        print(outputs["metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
