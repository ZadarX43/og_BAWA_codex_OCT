#!/usr/bin/env python3
"""Build PLAYER_FOULED recent-form x context policy proof tables.

Research-only beta/intelligence audit. This is not a priced player-prop model
and it does not write deploy artifacts. It tests whether fouls-drawn outcomes
become more reliable when player recent fouls-won form, opponent role allowance,
referee/contact ecosystem, and lineup role certainty agree.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.api_football.team_name_map import normalize_team_name
from scripts.build_player_event_live_interaction_features import load_historical_actuals, norm_text, role_group


PLAYER_EVENTS_DIR = ROOT / "data_sources" / "api_football" / "features" / "player_events"
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-07" / "player_fouled_interaction_policy_audit"

FOUNDATION_LEAGUES = (
    "Belgium_Pro",
    "Brazil_Serie_A",
    "England_Championship",
    "England_EFL_League_1",
    "England_Premier_League",
    "France_Ligue_1",
    "Germany_Bundesliga",
    "Italy_Serie_A",
    "Netherlands_Eredivisie",
    "Norway_Eliteserien",
    "Portugal_Liga",
    "Scotland_Premiership",
    "Spain_La_Liga",
    "USA_MLS",
)
DEFAULT_SEASONS = (2022, 2023, 2024)
RECENT_QUANTILES = (0.75,)
OPPONENT_QUANTILES = (0.75, 0.85, 0.90)
CONTEXT_QUANTILES = (0.65,)
MIN_MONTH_ROWS = 12


@dataclass(frozen=True)
class FouledSpec:
    market: str
    display_market: str
    target_label: str
    hit_col: str
    threshold: int
    beta_core_min_hit: float
    research_ready_min_hit: float
    watch_min_hit: float


MARKETS = {
    "fouled_ge1": FouledSpec(
        market="fouled_ge1",
        display_market="Player Fouled 0.5+",
        target_label="fouls_drawn >= 1",
        hit_col="actual_fouled_ge1",
        threshold=1,
        beta_core_min_hit=0.68,
        research_ready_min_hit=0.62,
        watch_min_hit=0.56,
    ),
    "fouled_ge2": FouledSpec(
        market="fouled_ge2",
        display_market="Player Fouled 1.5+",
        target_label="fouls_drawn >= 2",
        hit_col="actual_fouled_ge2",
        threshold=2,
        beta_core_min_hit=0.38,
        research_ready_min_hit=0.32,
        watch_min_hit=0.26,
    ),
    "fouled_ge3": FouledSpec(
        market="fouled_ge3",
        display_market="Player Fouled 2.5+",
        target_label="fouls_drawn >= 3",
        hit_col="actual_fouled_ge3",
        threshold=3,
        beta_core_min_hit=0.20,
        research_ready_min_hit=0.16,
        watch_min_hit=0.12,
    ),
}

TARGET_COLS = [
    "fixture_key",
    "match_date",
    "competition",
    "league",
    "home_team_name",
    "away_team_name",
    "team_name",
    "player_name",
    "player_team_side",
    "position_group",
    "tactical_role",
    "expected_start_flag",
    "expected_minutes",
    "team_formation",
    "opponent_formation",
    "flank_zone",
    "fixture_foul_density_score",
    "fixture_wide_duel_score",
    "fixture_territorial_stress_score",
    "fixture_attack_pressure_score",
    "fixture_tackle_density_score",
    "fixture_midfield_grind_score",
    "ref_cards_per_match",
    "ref_foul_to_card_ratio",
    "ref_dissent_strictness",
    "ref_timewasting_strictness",
    "match_stakes_score",
    "rivalry_flag",
    "opponent_possession_projection",
    "formation_pressure_score",
    "fouls_won_per90",
    "fouls_per90",
    "tackles_per90",
    "dribbles_faced_per90",
    "days_rest",
    "recent_injury_return_flag",
    "temperament_flag",
    "suspension_risk_flag",
]

RECENT_FEATURES = (
    "attacker_recent_fouls_won_per90_l8",
    "attacker_recent_fouls_won_per90_l5",
    "fouls_won_per90",
)

OPPONENT_FEATURES = (
    "opp_attack_allowed_role_fouls_drawn_per_player_l10",
    "opp_attack_allowed_role_fouls_drawn_per_match_l10",
    "opp_attack_allowed_role_player_fouled_ge1_rate_l10",
    "opp_attack_allowed_role_player_fouled_ge2_rate_l10",
)

CONTEXT_FEATURES = (
    "fixture_foul_density_score",
    "fixture_wide_duel_score",
    "fixture_territorial_stress_score",
    "ref_cards_per_match",
    "formation_pressure_score",
)


def num(values: Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def norm_team(value: Any, league_tag: Any = None) -> str:
    return norm_text(normalize_team_name(value, str(league_tag) if league_tag is not None else None))


def parse_csv_set(value: str, cast=str) -> set:
    return {cast(part.strip()) for part in value.split(",") if part.strip()}


def parse_tag(path: Path) -> tuple[str, int] | None:
    match = re.match(r"player_events_fixture_input__(.+)__(\d{4})\.csv$", path.name)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def read_csv_selected(path: Path, requested: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    usecols = [col for col in requested if col in header.columns]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    for col in requested:
        if col not in df.columns:
            df[col] = np.nan
    return df


def load_targets(input_dir: Path, leagues: set[str], seasons: set[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(input_dir.glob("player_events_fixture_input__*.csv")):
        parsed = parse_tag(path)
        if parsed is None:
            continue
        league_tag, season_tag = parsed
        if league_tag not in leagues or season_tag not in seasons:
            continue
        df = read_csv_selected(path, TARGET_COLS)
        if df.empty:
            continue
        df["league_tag"] = league_tag
        df["season_tag"] = season_tag
        df["source_file"] = path.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["match_date"] = pd.to_datetime(out["match_date"], errors="coerce")
    out["player_name_norm"] = out["player_name"].map(norm_text)
    out["team_name_norm"] = [norm_team(team, tag) for team, tag in zip(out["team_name"], out["league_tag"])]
    out["home_team_norm"] = [norm_team(team, tag) for team, tag in zip(out["home_team_name"], out["league_tag"])]
    out["away_team_norm"] = [norm_team(team, tag) for team, tag in zip(out["away_team_name"], out["league_tag"])]
    out["opponent_team_name"] = np.where(
        out["team_name_norm"].eq(out["home_team_norm"]),
        out["away_team_name"],
        out["home_team_name"],
    )
    out["opponent_team_norm"] = [norm_team(team, tag) for team, tag in zip(out["opponent_team_name"], out["league_tag"])]
    out["attack_role_group"] = [
        role_group(role, pos) for role, pos in zip(out.get("tactical_role", ""), out.get("position_group", ""))
    ]
    for col in [
        "expected_start_flag",
        "expected_minutes",
        "fixture_foul_density_score",
        "fixture_wide_duel_score",
        "fixture_territorial_stress_score",
        "fixture_attack_pressure_score",
        "fixture_tackle_density_score",
        "fixture_midfield_grind_score",
        "ref_cards_per_match",
        "ref_foul_to_card_ratio",
        "match_stakes_score",
        "rivalry_flag",
        "opponent_possession_projection",
        "formation_pressure_score",
        "fouls_won_per90",
    ]:
        out[col] = num(out[col]).fillna(0.0)
    out["eval_month"] = out["match_date"].dt.to_period("M").astype(str)
    return out.dropna(subset=["fixture_key", "match_date", "team_name", "player_name"]).copy()


def actuals_from_history(history: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "fixture_key",
        "league_tag",
        "season_tag",
        "team_name",
        "player_name",
        "team_name_norm",
        "player_name_norm",
        "fouls_drawn",
        "fouls_committed",
        "minutes",
        "started_flag",
    ]
    out = history[[c for c in cols if c in history.columns]].copy()
    out["fouls_drawn"] = num(out.get("fouls_drawn", 0)).fillna(0.0)
    out["actual_fouled_ge1"] = out["fouls_drawn"].ge(1).astype(float)
    out["actual_fouled_ge2"] = out["fouls_drawn"].ge(2).astype(float)
    out["actual_fouled_ge3"] = out["fouls_drawn"].ge(3).astype(float)
    return out.drop_duplicates(["fixture_key", "league_tag", "team_name_norm", "player_name_norm"], keep="last")


def add_join_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_join_fixture_key"] = out["fixture_key"].fillna("").astype(str)
    out["_join_team_name_norm"] = out["team_name_norm"].fillna("").astype(str)
    out["_join_player_name_norm"] = out["player_name_norm"].fillna("").astype(str)
    return out


def add_recent_features(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.sort_values(["league_tag", "player_name_norm", "match_date", "fixture_key"]).copy()
    grouped = out.groupby(["league_tag", "player_name_norm"], dropna=False)
    for n in (5, 8):
        foul_sum = grouped["fouls_drawn"].transform(lambda s: s.shift().rolling(n, min_periods=1).sum())
        minutes_sum = grouped["minutes"].transform(lambda s: s.shift().rolling(n, min_periods=1).sum())
        starts_sum = grouped["started_flag"].transform(lambda s: s.shift().rolling(n, min_periods=1).sum())
        apps = grouped["fouls_drawn"].transform(lambda s: s.shift().rolling(n, min_periods=1).count())
        out[f"attacker_recent_fouls_won_l{n}"] = foul_sum.fillna(0.0)
        out[f"attacker_recent_minutes_l{n}"] = minutes_sum.fillna(0.0)
        out[f"attacker_recent_starts_l{n}"] = starts_sum.fillna(0.0)
        out[f"attacker_recent_apps_l{n}"] = apps.fillna(0.0)
        out[f"attacker_recent_fouls_won_per90_l{n}"] = np.where(
            minutes_sum.gt(0),
            foul_sum * 90.0 / minutes_sum,
            0.0,
        )
        out[f"attacker_recent_start_share_l{n}"] = np.where(apps.gt(0), starts_sum / apps, 0.0)
    return out


def add_allowed_rollups(allowed: pd.DataFrame, group_cols: list[str], prefix: str) -> pd.DataFrame:
    work = allowed.sort_values(group_cols + ["match_date", "fixture_key"]).copy()
    grouped = work.groupby(group_cols, dropna=False)
    for n in (5, 10):
        players = grouped["players"].transform(lambda s: s.shift().rolling(n, min_periods=1).sum())
        fouls = grouped["fouls_drawn"].transform(lambda s: s.shift().rolling(n, min_periods=1).sum())
        ge1 = grouped["fouled_ge1_players"].transform(lambda s: s.shift().rolling(n, min_periods=1).sum())
        ge2 = grouped["fouled_ge2_players"].transform(lambda s: s.shift().rolling(n, min_periods=1).sum())
        matches = grouped["fixture_key"].transform(lambda s: s.shift().rolling(n, min_periods=1).count())
        work[f"{prefix}_matches_l{n}"] = matches.fillna(0.0)
        work[f"{prefix}_players_l{n}"] = players.fillna(0.0)
        work[f"{prefix}_fouls_drawn_l{n}"] = fouls.fillna(0.0)
        work[f"{prefix}_fouls_drawn_per_match_l{n}"] = np.where(matches.gt(0), fouls / matches, 0.0)
        work[f"{prefix}_fouls_drawn_per_player_l{n}"] = np.where(players.gt(0), fouls / players, 0.0)
        work[f"{prefix}_player_fouled_ge1_rate_l{n}"] = np.where(players.gt(0), ge1 / players, 0.0)
        work[f"{prefix}_player_fouled_ge2_rate_l{n}"] = np.where(players.gt(0), ge2 / players, 0.0)
    return work


def add_opponent_allowance_features(scored: pd.DataFrame) -> pd.DataFrame:
    attack = scored[
        scored["attack_role_group"].isin(["central_striker", "wide_forward", "wide_midfielder_winger"])
    ].copy()
    if attack.empty:
        return scored
    attack["fouled_ge1"] = attack["fouls_drawn"].ge(1).astype(float)
    attack["fouled_ge2"] = attack["fouls_drawn"].ge(2).astype(float)
    role_allowed = (
        attack.groupby(["league_tag", "match_date", "fixture_key", "opponent_team_norm", "attack_role_group"], dropna=False)
        .agg(
            players=("player_name", "size"),
            fouls_drawn=("fouls_drawn", "sum"),
            fouled_ge1_players=("fouled_ge1", "sum"),
            fouled_ge2_players=("fouled_ge2", "sum"),
        )
        .reset_index()
    )
    any_allowed = (
        attack.groupby(["league_tag", "match_date", "fixture_key", "opponent_team_norm"], dropna=False)
        .agg(
            players=("player_name", "size"),
            fouls_drawn=("fouls_drawn", "sum"),
            fouled_ge1_players=("fouled_ge1", "sum"),
            fouled_ge2_players=("fouled_ge2", "sum"),
        )
        .reset_index()
    )
    role_allowed = add_allowed_rollups(
        role_allowed,
        ["league_tag", "opponent_team_norm", "attack_role_group"],
        "opp_attack_allowed_role",
    )
    any_allowed = add_allowed_rollups(
        any_allowed,
        ["league_tag", "opponent_team_norm"],
        "opp_attack_allowed_attacker_any",
    )
    role_cols = [
        c
        for c in role_allowed.columns
        if c.startswith("opp_attack_allowed_role_")
    ]
    any_cols = [
        c
        for c in any_allowed.columns
        if c.startswith("opp_attack_allowed_attacker_any_")
    ]
    out = scored.merge(
        role_allowed[
            ["league_tag", "fixture_key", "opponent_team_norm", "attack_role_group"] + role_cols
        ].drop_duplicates(["league_tag", "fixture_key", "opponent_team_norm", "attack_role_group"]),
        on=["league_tag", "fixture_key", "opponent_team_norm", "attack_role_group"],
        how="left",
    )
    out = out.merge(
        any_allowed[["league_tag", "fixture_key", "opponent_team_norm"] + any_cols].drop_duplicates(
            ["league_tag", "fixture_key", "opponent_team_norm"]
        ),
        on=["league_tag", "fixture_key", "opponent_team_norm"],
        how="left",
    )
    for col in role_cols + any_cols:
        out[col] = num(out[col]).fillna(0.0)
    return out


def build_scored_frame(targets: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    actuals = actuals_from_history(history)
    scored = add_join_keys(targets)
    actuals = add_join_keys(actuals)
    actual_cols = [
        "fouls_drawn",
        "fouls_committed",
        "actual_fouled_ge1",
        "actual_fouled_ge2",
        "actual_fouled_ge3",
        "minutes",
        "started_flag",
    ]
    scored = scored.merge(
        actuals[["_join_fixture_key", "_join_team_name_norm", "_join_player_name_norm"] + actual_cols],
        on=["_join_fixture_key", "_join_team_name_norm", "_join_player_name_norm"],
        how="inner",
        suffixes=("", "_actual"),
    )
    scored = add_recent_features(scored)
    scored = add_opponent_allowance_features(scored)
    for col in RECENT_FEATURES + OPPONENT_FEATURES + CONTEXT_FEATURES:
        if col not in scored.columns:
            scored[col] = np.nan
        scored[col] = num(scored[col])
    for col in ["fouls_drawn", "actual_fouled_ge1", "actual_fouled_ge2", "actual_fouled_ge3", "minutes_actual"]:
        if col in scored.columns:
            scored[col] = num(scored[col]).fillna(0.0)
    scored["lineup_role_ready_flag"] = (
        num(scored.get("expected_start_flag", 0)).ge(1)
        & num(scored.get("expected_minutes", 0)).ge(60)
        & scored.get("attack_role_group", pd.Series("", index=scored.index)).isin(
            ["central_striker", "wide_forward", "wide_midfielder_winger"]
        )
    )
    return scored.drop(columns=["_join_fixture_key", "_join_team_name_norm", "_join_player_name_norm"], errors="ignore")


def threshold_value(series: pd.Series, quantile: float) -> float:
    values = num(series).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return np.nan
    return float(values.quantile(quantile))


def stable_month_share(subset: pd.DataFrame, hit_col: str, compare_hit: float) -> tuple[int, int, float]:
    monthly = subset.groupby("eval_month", dropna=False).agg(rows=(hit_col, "size"), hit_rate=(hit_col, "mean")).reset_index()
    monthly = monthly[monthly["rows"].ge(MIN_MONTH_ROWS)].copy()
    if monthly.empty:
        return 0, 0, np.nan
    stable = int(monthly["hit_rate"].ge(compare_hit).sum())
    return int(len(monthly)), stable, float(stable / len(monthly))


def label_candidate(
    spec: FouledSpec,
    rows: int,
    hit_rate: float,
    lift_vs_baseline: float,
    lift_vs_recent_only: float,
    lift_vs_opponent_only: float,
    months_ge_min: int,
    stable_vs_recent: float,
) -> str:
    if rows < 80 or months_ge_min < 3 or math.isnan(hit_rate):
        return "DO_NOT_USE"
    if (
        rows >= 350
        and months_ge_min >= 8
        and hit_rate >= spec.beta_core_min_hit
        and lift_vs_baseline >= 0.08
        and lift_vs_recent_only >= 0.01
        and lift_vs_opponent_only >= 0.01
        and stable_vs_recent >= 0.55
    ):
        return "FOULED_INTERACTION_CORE"
    if (
        rows >= 180
        and months_ge_min >= 6
        and hit_rate >= spec.research_ready_min_hit
        and lift_vs_baseline >= 0.05
        and lift_vs_opponent_only >= 0.01
        and lift_vs_recent_only >= -0.01
    ):
        return "FOULED_RESEARCH_READY"
    if rows >= 120 and months_ge_min >= 5 and hit_rate >= spec.watch_min_hit and lift_vs_baseline >= 0.03:
        return "FOULED_WATCH"
    return "DO_NOT_USE"


def summarize_candidate(
    df: pd.DataFrame,
    spec: FouledSpec,
    recent_feature: str,
    opponent_feature: str,
    context_feature: str,
    recent_quantile: float,
    opponent_quantile: float,
    context_quantile: float,
    require_lineup_role: bool,
) -> dict[str, Any] | None:
    feature_cols = [recent_feature, opponent_feature, context_feature]
    valid = df[["eval_month", spec.hit_col, "league_tag", "season_tag", "tactical_role", "attack_role_group", "lineup_role_ready_flag"] + feature_cols].dropna().copy()
    if len(valid) < 250 or valid[spec.hit_col].nunique() < 2:
        return None
    recent_t = threshold_value(valid[recent_feature], recent_quantile)
    opp_t = threshold_value(valid[opponent_feature], opponent_quantile)
    context_t = threshold_value(valid[context_feature], context_quantile)
    if math.isnan(recent_t) or math.isnan(opp_t) or math.isnan(context_t):
        return None
    if opp_t <= 0:
        return None
    baseline_hit = float(valid[spec.hit_col].mean())
    recent_mask = num(valid[recent_feature]).ge(recent_t)
    opponent_mask = num(valid[opponent_feature]).ge(opp_t)
    context_mask = num(valid[context_feature]).ge(context_t)
    lineup_mask = valid["lineup_role_ready_flag"].astype(bool) if require_lineup_role else pd.Series(True, index=valid.index)
    recent_only = valid[recent_mask].copy()
    opponent_only = valid[opponent_mask].copy()
    context_only = valid[context_mask].copy()
    combo = valid[recent_mask & opponent_mask & context_mask & lineup_mask].copy()
    if combo.empty:
        return None
    combo_hit = float(combo[spec.hit_col].mean())
    recent_hit = float(recent_only[spec.hit_col].mean()) if not recent_only.empty else np.nan
    opponent_hit = float(opponent_only[spec.hit_col].mean()) if not opponent_only.empty else np.nan
    context_hit = float(context_only[spec.hit_col].mean()) if not context_only.empty else np.nan
    months_ge_min, stable_baseline, stable_vs_baseline = stable_month_share(combo, spec.hit_col, baseline_hit)
    _, stable_recent, stable_vs_recent = stable_month_share(combo, spec.hit_col, recent_hit)
    _, stable_opp, stable_vs_opp = stable_month_share(combo, spec.hit_col, opponent_hit)
    lift_vs_baseline = combo_hit - baseline_hit
    lift_vs_recent = combo_hit - recent_hit
    lift_vs_opp = combo_hit - opponent_hit
    return {
        "market": spec.market,
        "display_market": spec.display_market,
        "target_label": spec.target_label,
        "recent_feature": recent_feature,
        "opponent_feature": opponent_feature,
        "context_feature": context_feature,
        "recent_quantile": recent_quantile,
        "opponent_quantile": opponent_quantile,
        "context_quantile": context_quantile,
        "recent_threshold": recent_t,
        "opponent_threshold": opp_t,
        "context_threshold": context_t,
        "require_lineup_role": int(require_lineup_role),
        "baseline_rows": int(len(valid)),
        "baseline_hit": baseline_hit,
        "recent_only_rows": int(len(recent_only)),
        "recent_only_hit": recent_hit,
        "opponent_only_rows": int(len(opponent_only)),
        "opponent_only_hit": opponent_hit,
        "context_only_rows": int(len(context_only)),
        "context_only_hit": context_hit,
        "interaction_rows": int(len(combo)),
        "interaction_hit": combo_hit,
        "lift_vs_baseline": lift_vs_baseline,
        "lift_vs_recent_only": lift_vs_recent,
        "lift_vs_opponent_only": lift_vs_opp,
        "lift_vs_context_only": combo_hit - context_hit,
        "months_ge_min": months_ge_min,
        "stable_months_vs_baseline": stable_baseline,
        "stable_month_share_vs_baseline": stable_vs_baseline,
        "stable_months_vs_recent_only": stable_recent,
        "stable_month_share_vs_recent_only": stable_vs_recent,
        "stable_months_vs_opponent_only": stable_opp,
        "stable_month_share_vs_opponent_only": stable_vs_opp,
        "interaction_label": label_candidate(
            spec,
            int(len(combo)),
            combo_hit,
            lift_vs_baseline,
            lift_vs_recent,
            lift_vs_opp,
            months_ge_min,
            stable_vs_recent,
        ),
    }


def build_grid(scored: pd.DataFrame, specs: dict[str, FouledSpec]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in specs.values():
        for recent_feature in RECENT_FEATURES:
            if recent_feature not in scored.columns:
                continue
            for opponent_feature in OPPONENT_FEATURES:
                if opponent_feature not in scored.columns:
                    continue
                for context_feature in CONTEXT_FEATURES:
                    if context_feature not in scored.columns:
                        continue
                    for recent_q in RECENT_QUANTILES:
                        for opponent_q in OPPONENT_QUANTILES:
                            for context_q in CONTEXT_QUANTILES:
                                for require_lineup_role in (False, True):
                                    row = summarize_candidate(
                                        scored,
                                        spec,
                                        recent_feature,
                                        opponent_feature,
                                        context_feature,
                                        recent_q,
                                        opponent_q,
                                        context_q,
                                        require_lineup_role,
                                    )
                                    if row is not None:
                                        rows.append(row)
    return pd.DataFrame(rows)


def policy_candidates(grid: pd.DataFrame) -> pd.DataFrame:
    if grid.empty:
        return grid
    rank = {
        "FOULED_INTERACTION_CORE": 0,
        "FOULED_RESEARCH_READY": 1,
        "FOULED_WATCH": 2,
        "DO_NOT_USE": 9,
    }
    out = grid.copy()
    out["_rank"] = out["interaction_label"].map(rank).fillna(9)
    out = out.sort_values(
        [
            "_rank",
            "interaction_hit",
            "lift_vs_recent_only",
            "lift_vs_opponent_only",
            "interaction_rows",
            "stable_month_share_vs_recent_only",
        ],
        ascending=[True, False, False, False, False, False],
    )
    # Quantile sweeps can collapse to identical numeric thresholds; keep the
    # strongest rendering of each distinct policy so the report stays readable.
    for col in ["recent_threshold", "opponent_threshold", "context_threshold"]:
        out[f"_{col}_key"] = num(out[col]).round(6)
    dedupe_cols = [
        "market",
        "recent_feature",
        "opponent_feature",
        "context_feature",
        "_recent_threshold_key",
        "_opponent_threshold_key",
        "_context_threshold_key",
        "interaction_rows",
        "interaction_hit",
        "lift_vs_recent_only",
        "lift_vs_opponent_only",
    ]
    out = out.drop_duplicates(dedupe_cols, keep="first")
    return (
        out.drop(columns=["_rank", "_recent_threshold_key", "_opponent_threshold_key", "_context_threshold_key"])
        .groupby("market", as_index=False, group_keys=False)
        .head(25)
        .reset_index(drop=True)
    )


def row_mask(scored: pd.DataFrame, candidate: pd.Series) -> pd.Series:
    mask = (
        num(scored[candidate["recent_feature"]]).ge(float(candidate["recent_threshold"]))
        & num(scored[candidate["opponent_feature"]]).ge(float(candidate["opponent_threshold"]))
        & num(scored[candidate["context_feature"]]).ge(float(candidate["context_threshold"]))
    )
    if int(candidate.get("require_lineup_role", 0)) == 1:
        mask = mask & scored["lineup_role_ready_flag"].astype(bool)
    return mask


def breakdown(scored: pd.DataFrame, candidates: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    keep = {"FOULED_INTERACTION_CORE", "FOULED_RESEARCH_READY", "FOULED_WATCH"}
    rows: list[dict[str, Any]] = []
    for _, candidate in candidates[candidates["interaction_label"].isin(keep)].iterrows():
        spec = MARKETS[str(candidate["market"])]
        subset = scored[row_mask(scored, candidate)].copy()
        if subset.empty:
            continue
        for key, group in subset.groupby(group_cols, dropna=False):
            if len(group) < 20:
                continue
            if not isinstance(key, tuple):
                key = (key,)
            hit_rate = float(group[spec.hit_col].mean())
            record = candidate[
                [
                    "market",
                    "display_market",
                    "recent_feature",
                    "opponent_feature",
                    "context_feature",
                    "interaction_label",
                    "baseline_hit",
                    "recent_only_hit",
                    "opponent_only_hit",
                    "context_only_hit",
                ]
            ].to_dict()
            record.update(dict(zip(group_cols, key)))
            rows.append(
                {
                    **record,
                    "rows": int(len(group)),
                    "hit_rate": hit_rate,
                    "lift_vs_baseline": hit_rate - float(candidate["baseline_hit"]),
                    "lift_vs_recent_only": hit_rate - float(candidate["recent_only_hit"]),
                    "lift_vs_opponent_only": hit_rate - float(candidate["opponent_only_hit"]),
                    "lift_vs_context_only": hit_rate - float(candidate["context_only_hit"]),
                }
            )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
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
    scored: pd.DataFrame,
    grid: pd.DataFrame,
    candidates: pd.DataFrame,
    league: pd.DataFrame,
    role: pd.DataFrame,
) -> None:
    label_counts = (
        grid.groupby(["market", "interaction_label"], dropna=False)
        .size()
        .reset_index(name="candidate_count")
        .sort_values(["market", "interaction_label"])
        if not grid.empty
        else pd.DataFrame()
    )
    top_cols = [
        "market",
        "recent_feature",
        "opponent_feature",
        "context_feature",
        "recent_quantile",
        "opponent_quantile",
        "context_quantile",
        "require_lineup_role",
        "interaction_rows",
        "interaction_hit",
        "lift_vs_baseline",
        "lift_vs_recent_only",
        "lift_vs_opponent_only",
        "stable_month_share_vs_recent_only",
        "interaction_label",
    ]
    lines = [
        "# Player Fouled Interaction Policy Audit",
        "",
        "Research-only beta/intelligence audit for `Player To Be Fouled` style markets.",
        "",
        "## Safety",
        "- No priced player-prop odds.",
        "- No deploy routing, tiers, slips, or production rulebook changes.",
        "- Candidate rows are policy proof inputs for live-shadow review only.",
        "",
        "## Input",
        f"- scored proof rows: `{len(scored)}`",
        f"- fixtures: `{scored['fixture_key'].nunique() if not scored.empty else 0}`",
        f"- players: `{scored['player_name'].nunique() if not scored.empty else 0}`",
        f"- leagues: `{scored['league_tag'].nunique() if not scored.empty else 0}`",
        "",
        "## Label Counts",
        markdown_table(label_counts, max_rows=40),
        "",
        "## Best Policy Candidates",
        markdown_table(candidates[[c for c in top_cols if c in candidates.columns]], max_rows=35),
        "",
        "## Best League Breakdowns",
        markdown_table(
            league.sort_values(["hit_rate", "rows"], ascending=[False, False]).head(35)
            if not league.empty
            else league
        ),
        "",
        "## Best Role Breakdowns",
        markdown_table(
            role.sort_values(["hit_rate", "rows"], ascending=[False, False]).head(35)
            if not role.empty
            else role
        ),
        "",
        "## Interpretation",
        "- `FOULED_INTERACTION_CORE` means recent fouls-won, opponent fouls-drawn allowance, and context all agree with useful lift.",
        "- `FOULED_RESEARCH_READY` means the cell is strong enough for live-shadow priority, not promotion.",
        "- `FOULED_WATCH` is a weak/smaller proof cell for monitoring only.",
        "- Prefer cells with positive lift versus recent-only; otherwise the opponent/context stack is not adding enough.",
    ]
    (outdir / "PLAYER_FOULED_INTERACTION_POLICY_AUDIT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PLAYER_EVENTS_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--leagues", default=",".join(FOUNDATION_LEAGUES))
    parser.add_argument("--seasons", default=",".join(str(season) for season in DEFAULT_SEASONS))
    parser.add_argument("--markets", default="fouled_ge1,fouled_ge2,fouled_ge3")
    parser.add_argument("--min-minutes", type=float, default=45.0)
    args = parser.parse_args()

    leagues = parse_csv_set(args.leagues, str)
    seasons = parse_csv_set(args.seasons, int)
    markets = parse_csv_set(args.markets, str)
    unknown = markets - set(MARKETS)
    if unknown:
        raise SystemExit(f"Unsupported markets: {sorted(unknown)}")
    selected_specs = {market: MARKETS[market] for market in sorted(markets)}

    args.outdir.mkdir(parents=True, exist_ok=True)
    targets = load_targets(args.input_dir, leagues, seasons)
    if targets.empty:
        raise SystemExit("No historical player-event fixture input rows found.")
    print(f"[stage] targets rows={len(targets)} fixtures={targets['fixture_key'].nunique()}", flush=True)
    history = load_historical_actuals(leagues, seasons)
    if history.empty:
        raise SystemExit("No historical match_player_stats rows found.")
    print(f"[stage] history rows={len(history)} fixtures={history['fixture_key'].nunique()}", flush=True)
    scored = build_scored_frame(targets, history)
    print(f"[stage] scored pre-filter rows={len(scored)} fixtures={scored['fixture_key'].nunique()}", flush=True)
    scored = scored[num(scored.get("minutes", scored.get("minutes_actual", 0))).ge(args.min_minutes)].copy()
    if scored.empty:
        raise SystemExit("No scored rows after minutes filter.")
    print(f"[stage] scored filtered rows={len(scored)} fixtures={scored['fixture_key'].nunique()}", flush=True)

    grid = build_grid(scored, selected_specs)
    print(f"[stage] grid rows={len(grid)}", flush=True)
    candidates = policy_candidates(grid)
    league = breakdown(scored, candidates, ["league_tag", "season_tag"]) if not candidates.empty else pd.DataFrame()
    role = breakdown(scored, candidates, ["tactical_role"]) if not candidates.empty else pd.DataFrame()

    scored.to_csv(args.outdir / "player_fouled_interaction_scored_rows.csv", index=False)
    grid.to_csv(args.outdir / "player_fouled_interaction_policy_grid.csv", index=False)
    candidates.to_csv(args.outdir / "player_fouled_interaction_policy_candidates.csv", index=False)
    league.to_csv(args.outdir / "player_fouled_interaction_league_breakdown.csv", index=False)
    role.to_csv(args.outdir / "player_fouled_interaction_role_breakdown.csv", index=False)
    write_report(args.outdir, scored, grid, candidates, league, role)

    print(f"WROTE {args.outdir}")
    print(f"scored_rows={len(scored)} fixtures={scored['fixture_key'].nunique()} players={scored['player_name'].nunique()}")
    if not candidates.empty:
        cols = [
            "market",
            "recent_feature",
            "opponent_feature",
            "context_feature",
            "interaction_rows",
            "interaction_hit",
            "lift_vs_recent_only",
            "interaction_label",
        ]
        print(candidates[cols].head(40).to_string(index=False))


if __name__ == "__main__":
    main()
