#!/usr/bin/env python3
"""Score intelligence sidecar flags against scored walk-forward deploy rows.

Research-only. This is a post-hoc overlay scorer for FTR/BTTS/OU25 rows. It
does not change `deploy_rulebook.py`, live routing, ModelStore artifacts, or
existing walk-forward outputs.

The script intentionally separates:
- pick/result spine: existing scored walk-forward CSVs
- sidecar evidence: lagged team/player/API feature coverage where locally present
- policy simulation: retained/blocked rows by research variant
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_SCORED_ROOT = Path("predictions_output/walk_forward_greenlist_btts_no_recent_regime_2026_05_06")
DEFAULT_FEATURES_DIR = Path("data_sources/api_football/features")
DEFAULT_OUTDIR = Path("reports/latest/walkforward_intelligence_overlay_backtest")
MARKETS = {"ftr", "btts", "ou25"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def markdown_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "_No rows._"
    text = df.head(max_rows).copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            text[col] = text[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            text[col] = text[col].astype("string").fillna("")
    lines = [
        "| " + " | ".join(str(c) for c in text.columns) + " |",
        "| " + " | ".join(["---"] * len(text.columns)) + " |",
    ]
    for _, row in text.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in text.columns) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(df)} rows._")
    return "\n".join(lines)


def num(series: Any) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def truthy(series: Any) -> pd.Series:
    s = pd.Series(series).astype("string").str.strip().str.lower()
    return s.isin({"1", "1.0", "true", "yes", "y", "t"})


def norm_market(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"over25", "over_25", "over_2_5"}:
        return "ou25"
    return text


def league_tag(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def canonical_team(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    replacements = {
        "man utd": "manchester united",
        "man united": "manchester united",
        "man city": "manchester city",
        "spurs": "tottenham",
        "tottenham hotspur": "tottenham",
        "psg": "paris saint germain",
        "inter": "inter milan",
        "ac milan": "milan",
        "ny red bulls": "new york red bulls",
        "nycfc": "new york city",
        "nyc": "new york city",
        "sj earthquakes": "san jose earthquakes",
        "st louis city": "saint louis city",
        "brighton hove albion": "brighton",
        "wolverhampton wanderers": "wolves",
        "west ham united": "west ham",
        "leeds united": "leeds",
        "newcastle united": "newcastle",
        "sheffield united": "sheffield united",
        "olympique marseille": "marseille",
        "olympique lyonnais": "lyon",
        "bayer 04 leverkusen": "bayer leverkusen",
        "borussia monchengladbach": "borussia m gladbach",
        "borussia moenchengladbach": "borussia m gladbach",
        "1 fc koln": "koln",
        "fc koln": "koln",
        "athletic club bilbao": "athletic club",
        "atletico madrid": "atletico madrid",
    }
    text = replacements.get(text, text)
    parts = [part for part in text.split() if part not in {"fc", "cf", "sc", "afc", "club", "the"}]
    return " ".join(parts)


def fixture_join_key(match_date: Any, home: Any, away: Any) -> str:
    date = str(match_date or "").strip()[:10]
    return f"{date}|{canonical_team(home)}|{canonical_team(away)}"


def load_fixture_identity_map(path: Path) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    required = {"league_tag", "calendar_year", "scored_fixture_key", "api_fixture_key", "api_fixture_id", "auto_usable_flag"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()
    usable = truthy(df["auto_usable_flag"])
    out = df[usable].copy()
    if out.empty:
        return pd.DataFrame()
    out["season"] = pd.to_numeric(out["calendar_year"], errors="coerce").astype("Int64")
    out = out.rename(
        columns={
            "scored_fixture_key": "fixture_key",
            "api_fixture_key": "fixture_identity_api_fixture_key",
            "api_fixture_id": "fixture_identity_api_fixture_id",
            "match_tier": "fixture_identity_match_tier",
            "confidence": "fixture_identity_confidence",
            "coverage_caveat": "fixture_identity_coverage_caveat",
        }
    )
    keep = [
        "league_tag",
        "season",
        "fixture_key",
        "fixture_identity_api_fixture_key",
        "fixture_identity_api_fixture_id",
        "fixture_identity_match_tier",
        "fixture_identity_confidence",
        "fixture_identity_coverage_caveat",
    ]
    return out[[col for col in keep if col in out.columns]].drop_duplicates(["league_tag", "season", "fixture_key"], keep="last")


def attach_fixture_identity(rows: pd.DataFrame, identity_map: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    if identity_map.empty:
        out["fixture_identity_join_hit"] = False
        out["fixture_identity_match_tier"] = ""
        out["fixture_identity_confidence"] = np.nan
        out["fixture_identity_coverage_caveat"] = ""
        out["fixture_identity_api_fixture_key"] = ""
        out["fixture_identity_api_fixture_id"] = np.nan
        return out
    out = out.merge(identity_map, on=["league_tag", "season", "fixture_key"], how="left")
    out["fixture_identity_join_hit"] = out["fixture_identity_api_fixture_key"].astype("string").fillna("").ne("")
    return out


def parse_tag(path: Path, family: str) -> tuple[str, int] | None:
    match = re.match(rf"api_{re.escape(family)}__(.+)__(\d{{4}})\.csv$", path.name)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def scored_files(root: Path, max_files: int = 0) -> list[Path]:
    files = sorted(root.glob("w*/03_scored/DEPLOY_COMBINED_SCORED_*.csv"))
    if max_files > 0:
        return files[:max_files]
    return files


def read_scored(path: Path) -> pd.DataFrame:
    wanted = {
        "window_id",
        "league",
        "match_date",
        "home_team_name",
        "away_team_name",
        "fixture_key",
        "market",
        "bookie_pick",
        "selection",
        "bookie_od",
        "bookie_implied",
        "model_p_for_bookie",
        "model_top_pick",
        "deploy_tier",
        "tier",
        "source_tier_file",
        "ftr_margin",
        "confidence_home",
        "confidence_draw",
        "confidence_away",
        "imp_home",
        "imp_draw",
        "imp_away",
        "p_home_ge2",
        "p_away_ge2",
        "home_team_ge2_candidate_flag",
        "away_team_ge2_candidate_flag",
        "home_ge2_confidence",
        "away_ge2_confidence",
        "ftr_drawtrap_flag",
        "draw_risk_flag",
        "close_match_flag",
        "chaos_risk_flag",
        "draw_chaos_score",
        "cs_mass_draw",
        "cs_mass_home_win",
        "cs_mass_away_win",
        "pick_side_mass_top3",
        "pick_side_margin_top3",
        "cs_ftr_alignment",
        "cs_draw_family_flag",
        "cs_fragility_flag",
        "cs_diffuse_flag",
        "cs_ou25_alignment",
        "cs_btts_alignment",
        "team_intel_overlay_active_flag",
        "team_intel_overlay_action",
        "team_intel_overlay_slip_caution_flag",
        "team_intel_overlay_avoid_in_acca_flag",
        "team_intel_overlay_reason",
        "context_reason_codes",
        "uefa_rotation_any",
        "uefa_home_rotation_risk",
        "uefa_away_rotation_risk",
        "uefa_live_table_volatility",
        "uefa_pride_only_flag",
        "uefa_home_eliminated",
        "uefa_away_eliminated",
        "home_team_goal_count",
        "away_team_goal_count",
        "status",
        "ftr_hit",
        "ou25_hit",
        "btts_yes_hit",
        "btts_no_hit",
    }
    df = pd.read_csv(path, usecols=lambda c: c in wanted, low_memory=False)
    if "window_id" not in df.columns:
        df["window_id"] = path.parts[-3] if len(path.parts) >= 3 else ""
    df["__source_scored_file"] = str(path)
    df["market_norm"] = df["market"].map(norm_market)
    return df[df["market_norm"].isin(MARKETS)].copy()


def load_scored(root: Path, max_files: int = 0) -> pd.DataFrame:
    frames = [read_scored(path) for path in scored_files(root, max_files=max_files)]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    df["match_date_dt"] = pd.to_datetime(df.get("match_date"), errors="coerce")
    df["season"] = df["match_date_dt"].dt.year.astype("Int64")
    df["league_tag"] = df["league"].map(league_tag)
    df["deploy_tier_norm"] = df.get("deploy_tier", df.get("tier", "")).astype("string").fillna("")
    df.loc[df["deploy_tier_norm"].eq(""), "deploy_tier_norm"] = df.get("tier", "").astype("string").fillna("")
    return df.sort_values(["window_id", "match_date_dt", "league", "fixture_key", "market_norm"]).reset_index(drop=True)


def feature_frame(features_dir: Path, family: str, keep_cols: set[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(features_dir.glob(f"api_{family}__*.csv")):
        parsed = parse_tag(path, family)
        if parsed is None:
            continue
        tag, season = parsed
        cols = keep_cols | {"fixture_key", "league", "season", "match_date", "home_team_name", "away_team_name"}
        try:
            frame = pd.read_csv(path, usecols=lambda c: c in cols, low_memory=False)
        except Exception:
            continue
        if frame.empty or "fixture_key" not in frame.columns:
            continue
        frame["league_tag"] = tag
        frame["season"] = int(season)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["league_tag", "season", "fixture_key"])
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["api_fixture_join_key"] = out.apply(
        lambda row: fixture_join_key(row.get("match_date"), row.get("home_team_name"), row.get("away_team_name")),
        axis=1,
    )
    return out.drop_duplicates(["league_tag", "season", "fixture_key"], keep="last")


def merge_sidecar(base: pd.DataFrame, frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if frame.empty:
        return base
    out = base.copy()
    if "api_fixture_join_key" not in out.columns:
        out["api_fixture_join_key"] = out.apply(
            lambda row: fixture_join_key(row.get("match_date"), row.get("home_team_name"), row.get("away_team_name")),
            axis=1,
        )

    identity_cols = {
        "league_tag",
        "season",
        "fixture_key",
        "api_fixture_join_key",
        "league",
        "league_id",
        "fixture_id",
        "match_date",
        "home_team_id",
        "away_team_id",
        "home_team_name",
        "away_team_name",
    }
    rename = {col: f"api_{col}" for col in frame.columns if col not in identity_cols}
    renamed = frame.rename(columns=rename)
    api_cols = [col for col in renamed.columns if col.startswith("api_") and col != "api_fixture_join_key"]

    out[f"{prefix}_identity_map_join_hit"] = False
    if "fixture_identity_api_fixture_key" in out.columns:
        by_api_fixture = renamed.drop_duplicates(["league_tag", "season", "fixture_key"], keep="last")
        mapped = out["fixture_identity_api_fixture_key"].astype("string").fillna("").ne("")
        if mapped.any():
            mapped_join = out.loc[mapped, ["league_tag", "season", "fixture_identity_api_fixture_key"]].merge(
                by_api_fixture[["league_tag", "season", "fixture_key", *api_cols]],
                left_on=["league_tag", "season", "fixture_identity_api_fixture_key"],
                right_on=["league_tag", "season", "fixture_key"],
                how="left",
            )
            mapped_join.index = out.loc[mapped].index
            for col in api_cols:
                out.loc[mapped, col] = mapped_join[col].values
            out.loc[mapped, f"{prefix}_identity_map_join_hit"] = mapped_join[api_cols].notna().any(axis=1).values if api_cols else False

    by_fixture = renamed.drop_duplicates(["league_tag", "season", "fixture_key"], keep="last")
    unresolved_before_fixture = ~out[f"{prefix}_identity_map_join_hit"]
    fixture_source = out.loc[unresolved_before_fixture].merge(
        by_fixture[["league_tag", "season", "fixture_key", *api_cols]],
        on=["league_tag", "season", "fixture_key"],
        how="left",
        suffixes=("", "__fixture"),
    )
    fixture_source.index = out.loc[unresolved_before_fixture].index
    for col in api_cols:
        fixture_col = f"{col}__fixture"
        if fixture_col in fixture_source.columns:
            out.loc[unresolved_before_fixture, col] = out.loc[unresolved_before_fixture, col].where(
                out.loc[unresolved_before_fixture, col].notna(),
                fixture_source[fixture_col],
            )
        elif col in fixture_source.columns:
            out.loc[unresolved_before_fixture, col] = out.loc[unresolved_before_fixture, col].where(
                out.loc[unresolved_before_fixture, col].notna(),
                fixture_source[col],
            )
    out[f"{prefix}_fixture_key_join_hit"] = False
    out.loc[unresolved_before_fixture, f"{prefix}_fixture_key_join_hit"] = fixture_source[api_cols].notna().any(axis=1).values if api_cols else False

    unresolved = ~(out[f"{prefix}_identity_map_join_hit"] | out[f"{prefix}_fixture_key_join_hit"])
    if unresolved.any() and "api_fixture_join_key" in renamed.columns:
        fallback = renamed.drop_duplicates(["league_tag", "season", "api_fixture_join_key"], keep="last")
        fallback_cols = ["league_tag", "season", "api_fixture_join_key", *api_cols]
        joined = out.loc[unresolved, ["league_tag", "season", "api_fixture_join_key"]].merge(
            fallback[fallback_cols],
            on=["league_tag", "season", "api_fixture_join_key"],
            how="left",
        )
        joined.index = out.loc[unresolved].index
        for col in api_cols:
            out.loc[unresolved, col] = out.loc[unresolved, col].where(out.loc[unresolved, col].notna(), joined[col])
        out[f"{prefix}_fallback_join_hit"] = False
        out.loc[unresolved, f"{prefix}_fallback_join_hit"] = joined[api_cols].notna().any(axis=1).values if api_cols else False
    else:
        out[f"{prefix}_fallback_join_hit"] = False
    out[f"{prefix}_join_hit"] = out[f"{prefix}_identity_map_join_hit"] | out[f"{prefix}_fixture_key_join_hit"] | out[f"{prefix}_fallback_join_hit"]
    return out


def attach_api_sidecars(rows: pd.DataFrame, features_dir: Path, *, allow_lineup_shadow: bool, fixture_identity_map: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    out = attach_fixture_identity(rows, fixture_identity_map)
    injury_cols = {
        "home_missing_attackers_count",
        "away_missing_attackers_count",
        "home_missing_defenders_count",
        "away_missing_defenders_count",
        "home_missing_goalkeepers_count",
        "away_missing_goalkeepers_count",
        "home_missing_minutes_l5_total",
        "away_missing_minutes_l5_total",
        "home_missing_goals_per90_l5",
        "away_missing_goals_per90_l5",
        "home_absence_severity_score",
        "away_absence_severity_score",
        "absence_severity_delta",
    }
    matchup_cols = {
        "balanced_strength_flag",
        "high_volatility_balanced_flag",
        "goal_environment_interaction",
        "mutual_scoring_interaction",
        "mutual_conceding_interaction",
        "style_conflict_index",
        "midfield_control_conflict_index",
    }
    team_cols = {
        "home_team_draw_rate_l5",
        "away_team_draw_rate_l5",
        "combined_btts_rate_l5",
        "combined_over25_rate_l5",
        "home_scored_rate_l5",
        "away_scored_rate_l5",
        "home_conceded_rate_l5",
        "away_conceded_rate_l5",
        "combined_total_goals_l5",
    }
    lineup_cols = {
        "home_starting_xi_avg_rating_l5",
        "away_starting_xi_avg_rating_l5",
        "home_starting_xi_minutes_l5",
        "away_starting_xi_minutes_l5",
        "home_starting_xi_goals_per90_l5",
        "away_starting_xi_goals_per90_l5",
        "xi_rating_delta",
        "xi_minutes_delta",
        "xi_goal_power_delta",
        "home_formation",
        "away_formation",
        "formation_mismatch_flag",
    }
    for family, cols in [
        ("injury_features", injury_cols),
        ("matchup_interaction_features", matchup_cols),
        ("team_rolling_features", team_cols),
    ]:
        frame = feature_frame(features_dir, family, cols)
        if not frame.empty:
            out = merge_sidecar(out, frame, prefix=family)
    if allow_lineup_shadow:
        frame = feature_frame(features_dir, "lineup_features", lineup_cols)
        if not frame.empty:
            out = merge_sidecar(out, frame, prefix="lineup_features")
            out["lineup_shadow_mode"] = "CONFIRMED_OR_PROVIDER_LINEUP_SHADOW"
        else:
            out["lineup_shadow_mode"] = "NO_LINEUP_FEATURES"
    else:
        out["lineup_shadow_mode"] = "LINEUP_FEATURES_EXCLUDED_TIME_SAFE_DEFAULT"
    return out


def pick_side(row: pd.Series) -> str:
    text = f"{row.get('bookie_pick', '')} {row.get('selection', '')} {row.get('model_top_pick', '')}".lower()
    home = str(row.get("home_team_name") or "").lower()
    away = str(row.get("away_team_name") or "").lower()
    if row.get("market_norm") != "ftr":
        return ""
    if "home" in text or text.strip() in {"h", "1"} or (home and home in text):
        return "home"
    if "away" in text or text.strip() in {"a", "2"} or (away and away in text):
        return "away"
    if "draw" in text or text.strip() in {"d", "x"}:
        return "draw"
    return ""


def compute_result(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    hit = pd.Series(np.nan, index=out.index, dtype="float64")
    ftr = out["market_norm"].eq("ftr")
    ou25 = out["market_norm"].eq("ou25")
    btts = out["market_norm"].eq("btts")
    if "ftr_hit" in out.columns:
        hit.loc[ftr] = num(out.loc[ftr, "ftr_hit"])
    if "ou25_hit" in out.columns:
        hit.loc[ou25] = num(out.loc[ou25, "ou25_hit"])
    if "btts_yes_hit" in out.columns:
        pick_text = (out.get("bookie_pick", "").astype("string") + " " + out.get("selection", "").astype("string")).str.lower()
        no_pick = pick_text.str.contains("no|under|false", regex=True, na=False)
        hit.loc[btts & no_pick] = num(out.loc[btts & no_pick, "btts_no_hit"])
        hit.loc[btts & ~no_pick] = num(out.loc[btts & ~no_pick, "btts_yes_hit"])
    out["settled_flag"] = hit.isin([0, 1])
    out["hit_flag"] = hit
    odds = num(out.get("bookie_od", np.nan))
    out["profit_units"] = np.where(out["hit_flag"].eq(1), odds - 1.0, np.where(out["hit_flag"].eq(0), -1.0, np.nan))
    out.loc[out["profit_units"].isna() & out["hit_flag"].eq(1), "profit_units"] = 1.0
    return out


def add_rolling_health(rows: pd.DataFrame, *, min_prior: int) -> pd.DataFrame:
    out = rows.copy()
    out["settled_int"] = out["settled_flag"].astype(int)
    out["win_int"] = out["hit_flag"].eq(1).astype(int)
    prior_settled = []
    prior_wins = []
    for _, idx in out.groupby(["league", "market_norm", "deploy_tier_norm"], dropna=False).groups.items():
        group = out.loc[idx].sort_values(["window_id", "match_date_dt", "fixture_key"])
        prior_settled.extend(zip(group.index, group["settled_int"].cumsum().shift(1).fillna(0).astype(int)))
        prior_wins.extend(zip(group.index, group["win_int"].cumsum().shift(1).fillna(0).astype(int)))
    out["rolling_prior_settled"] = 0
    out["rolling_prior_wins"] = 0
    for idx, val in prior_settled:
        out.at[idx, "rolling_prior_settled"] = int(val)
    for idx, val in prior_wins:
        out.at[idx, "rolling_prior_wins"] = int(val)
    out["rolling_prior_hit_rate"] = np.where(
        out["rolling_prior_settled"].gt(0),
        out["rolling_prior_wins"] / out["rolling_prior_settled"],
        np.nan,
    )

    def bucket(row: pd.Series) -> str:
        n = int(row.get("rolling_prior_settled", 0))
        hr = row.get("rolling_prior_hit_rate", np.nan)
        market = str(row.get("market_norm") or "")
        if n < min_prior or pd.isna(hr):
            return "UNKNOWN_LOW_SAMPLE"
        if market == "ftr":
            if hr >= 0.58:
                return "GREEN"
            if hr >= 0.50:
                return "AMBER"
            return "RED"
        if hr >= 0.62:
            return "GREEN"
        if hr >= 0.55:
            return "AMBER"
        return "RED"

    out["league_health_bucket"] = out.apply(bucket, axis=1)
    return out


def bool_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return truthy(df[col])


def text_contains(df: pd.DataFrame, col: str, pattern: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].astype("string").fillna("").str.lower().str.contains(pattern, regex=True)


def add_flags(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["picked_side"] = out.apply(pick_side, axis=1)
    out["unsafe_league_health_flag"] = out["league_health_bucket"].isin(["RED"])
    out["low_sample_health_flag"] = out["league_health_bucket"].eq("UNKNOWN_LOW_SAMPLE")

    p_draw = num(out.get("confidence_draw", np.nan)).where(num(out.get("confidence_draw", np.nan)).notna(), num(out.get("imp_draw", np.nan)))
    close_ftr = bool_col(out, "close_match_flag") | num(out.get("ftr_margin", np.nan)).lt(0.08)
    api_balanced = bool_col(out, "api_balanced_strength_flag") | bool_col(out, "api_high_volatility_balanced_flag")
    out["draw_stalemate_risk_flag"] = out["market_norm"].eq("ftr") & (
        bool_col(out, "ftr_drawtrap_flag")
        | bool_col(out, "draw_risk_flag")
        | bool_col(out, "cs_draw_family_flag")
        | api_balanced
        | (close_ftr & p_draw.ge(0.27))
        | num(out.get("api_home_team_draw_rate_l5", np.nan)).ge(0.35)
        | num(out.get("api_away_team_draw_rate_l5", np.nan)).ge(0.35)
    )

    cs_align = out.get("cs_ftr_alignment", pd.Series("", index=out.index)).astype("string").str.upper()
    out["cs_conflict_flag"] = out["market_norm"].eq("ftr") & (
        bool_col(out, "cs_fragility_flag")
        | bool_col(out, "cs_diffuse_flag")
        | (cs_align.ne("") & ~cs_align.str.contains("ALIGN|SUPPORT|MATCH", regex=True))
        | (num(out.get("cs_mass_draw", np.nan)).gt(num(out.get("pick_side_mass_top3", np.nan)).fillna(-1)))
    )

    home_attack_shock = (
        num(out.get("api_home_missing_attackers_count", np.nan)).ge(1)
        | num(out.get("api_home_missing_goals_per90_l5", np.nan)).ge(0.35)
        | num(out.get("api_home_absence_severity_score", np.nan)).ge(1.5)
    )
    away_attack_shock = (
        num(out.get("api_away_missing_attackers_count", np.nan)).ge(1)
        | num(out.get("api_away_missing_goals_per90_l5", np.nan)).ge(0.35)
        | num(out.get("api_away_absence_severity_score", np.nan)).ge(1.5)
    )
    home_defence_shock = (
        num(out.get("api_home_missing_defenders_count", np.nan)).ge(2)
        | num(out.get("api_home_missing_goalkeepers_count", np.nan)).ge(1)
        | num(out.get("api_home_absence_severity_score", np.nan)).ge(2.0)
    )
    away_defence_shock = (
        num(out.get("api_away_missing_defenders_count", np.nan)).ge(2)
        | num(out.get("api_away_missing_goalkeepers_count", np.nan)).ge(1)
        | num(out.get("api_away_absence_severity_score", np.nan)).ge(2.0)
    )
    out["home_attack_shock_flag"] = home_attack_shock.fillna(False)
    out["away_attack_shock_flag"] = away_attack_shock.fillna(False)
    out["home_defence_shock_flag"] = home_defence_shock.fillna(False)
    out["away_defence_shock_flag"] = away_defence_shock.fillna(False)
    out["picked_side_attack_shock_flag"] = (
        out["picked_side"].eq("home") & out["home_attack_shock_flag"]
    ) | (
        out["picked_side"].eq("away") & out["away_attack_shock_flag"]
    )
    out["picked_side_defence_shock_flag"] = (
        out["picked_side"].eq("home") & out["home_defence_shock_flag"]
    ) | (
        out["picked_side"].eq("away") & out["away_defence_shock_flag"]
    )
    out["injury_lineup_shock_flag"] = out["picked_side_attack_shock_flag"] | out["picked_side_defence_shock_flag"]

    out["lineup_uncertainty_flag"] = False
    if "api_home_starting_xi_minutes_l5" in out.columns:
        out["lineup_uncertainty_flag"] = (
            num(out.get("api_home_starting_xi_minutes_l5", np.nan)).lt(4500)
            | num(out.get("api_away_starting_xi_minutes_l5", np.nan)).lt(4500)
            | bool_col(out, "api_formation_mismatch_flag")
        )

    out["motivation_volatility_flag"] = (
        bool_col(out, "uefa_rotation_any")
        | bool_col(out, "uefa_home_rotation_risk")
        | bool_col(out, "uefa_away_rotation_risk")
        | bool_col(out, "uefa_live_table_volatility")
        | bool_col(out, "uefa_pride_only_flag")
        | text_contains(out, "context_reason_codes", "motivation|rotation|end_season|dead_rubber|title|relegation|cup")
        | text_contains(out, "team_intel_overlay_reason", "motivation|rotation|end|injur|lineup")
    )

    home_tg15_support = num(out.get("p_home_ge2", np.nan)).where(num(out.get("p_home_ge2", np.nan)).notna(), num(out.get("home_ge2_confidence", np.nan)))
    away_tg15_support = num(out.get("p_away_ge2", np.nan)).where(num(out.get("p_away_ge2", np.nan)).notna(), num(out.get("away_ge2_confidence", np.nan)))
    picked_tg15 = np.where(out["picked_side"].eq("home"), home_tg15_support, np.where(out["picked_side"].eq("away"), away_tg15_support, np.nan))
    out["tg15_no_help_flag"] = out["market_norm"].eq("ftr") & pd.Series(picked_tg15, index=out.index).lt(0.54)

    model_p = num(out.get("model_p_for_bookie", np.nan))
    implied = num(out.get("bookie_implied", np.nan))
    out["market_price_weakness_flag"] = (
        out["market_norm"].eq("ftr") & (model_p.lt(0.48) | implied.lt(0.44) | num(out.get("ftr_margin", np.nan)).lt(0.07))
    )

    goal_market = out["market_norm"].isin(["btts", "ou25"])
    out["goal_market_attack_shock_caution_flag"] = goal_market & (
        (out["home_attack_shock_flag"] & out["away_attack_shock_flag"])
        | (
            out["market_norm"].eq("ou25")
            & (home_tg15_support.lt(0.52) | away_tg15_support.lt(0.48))
            & (out["home_attack_shock_flag"] | out["away_attack_shock_flag"])
        )
        | (
            out["market_norm"].eq("btts")
            & ((home_tg15_support.lt(0.50) & out["home_attack_shock_flag"]) | (away_tg15_support.lt(0.50) & out["away_attack_shock_flag"]))
        )
    )
    out["team_intel_caution_flag"] = (
        bool_col(out, "team_intel_overlay_slip_caution_flag")
        | bool_col(out, "team_intel_overlay_avoid_in_acca_flag")
        | text_contains(out, "team_intel_overlay_action", "avoid|caution|downgrade")
    )
    risk_cols = [
        "unsafe_league_health_flag",
        "draw_stalemate_risk_flag",
        "cs_conflict_flag",
        "injury_lineup_shock_flag",
        "lineup_uncertainty_flag",
        "motivation_volatility_flag",
        "tg15_no_help_flag",
        "market_price_weakness_flag",
        "goal_market_attack_shock_caution_flag",
        "team_intel_caution_flag",
    ]
    for col in risk_cols:
        out[col] = out[col].fillna(False).astype(bool)
    out["intelligence_risk_score"] = sum(out[col].astype(int) for col in risk_cols)
    out["intelligence_risk_band"] = pd.cut(
        out["intelligence_risk_score"],
        bins=[-1, 0, 1, 2, 99],
        labels=["INTEL_SAFE", "INTEL_CAUTION", "INTEL_HIGH_RISK", "INTEL_BLOCK_RESEARCH"],
    ).astype("string")
    out["reason_codes"] = out.apply(reason_codes, axis=1)
    return out


def reason_codes(row: pd.Series) -> str:
    mapping = [
        ("unsafe_league_health_flag", "LOW_ROLLING_LEAGUE_HEALTH"),
        ("draw_stalemate_risk_flag", "DRAW_STALEMATE_RISK"),
        ("cs_conflict_flag", "CS_CONFLICT"),
        ("injury_lineup_shock_flag", "INJURY_LINEUP_SHOCK"),
        ("lineup_uncertainty_flag", "LINEUP_UNCERTAINTY"),
        ("motivation_volatility_flag", "MOTIVATION_VOLATILITY"),
        ("tg15_no_help_flag", "TG15_NO_HELP"),
        ("market_price_weakness_flag", "MARKET_PRICE_WEAKNESS"),
        ("goal_market_attack_shock_caution_flag", "GOAL_MARKET_ATTACK_SHOCK"),
        ("team_intel_caution_flag", "TEAM_INTEL_CAUTION"),
    ]
    return "|".join(code for col, code in mapping if bool(row.get(col, False)))


def variant_masks(rows: pd.DataFrame) -> dict[str, pd.Series]:
    all_true = pd.Series(True, index=rows.index)
    ftr = rows["market_norm"].eq("ftr")
    return {
        "baseline_no_overlay": all_true,
        "ftr_block_red_health": ~(ftr & rows["unsafe_league_health_flag"]),
        "ftr_block_draw_stalemate": ~(ftr & rows["draw_stalemate_risk_flag"]),
        "ftr_block_injury_shock": ~(ftr & rows["injury_lineup_shock_flag"]),
        "ftr_block_cs_conflict": ~(ftr & rows["cs_conflict_flag"]),
        "ftr_safe_or_caution_only": ~(ftr & rows["intelligence_risk_band"].isin(["INTEL_HIGH_RISK", "INTEL_BLOCK_RESEARCH"])),
        "goal_market_attack_shock_caution": ~rows["goal_market_attack_shock_caution_flag"],
        "all_markets_high_risk_research_block": ~rows["intelligence_risk_band"].eq("INTEL_BLOCK_RESEARCH"),
    }


def summarize_slice(df: pd.DataFrame) -> dict[str, Any]:
    settled = df[df["settled_flag"]].copy()
    wins = int(settled["hit_flag"].eq(1).sum())
    losses = int(settled["hit_flag"].eq(0).sum())
    profit = float(settled["profit_units"].sum()) if not settled.empty else 0.0
    return {
        "rows": int(len(df)),
        "settled": int(len(settled)),
        "wins": wins,
        "losses": losses,
        "hit_rate": round(wins / len(settled), 4) if len(settled) else np.nan,
        "profit_units": round(profit, 4),
        "roi": round(profit / len(settled), 4) if len(settled) else np.nan,
    }


def build_variant_summary(rows: pd.DataFrame) -> pd.DataFrame:
    baseline = rows.copy()
    masks = variant_masks(rows)
    out: list[dict[str, Any]] = []
    for name, mask in masks.items():
        retained = rows[mask].copy()
        blocked = rows[~mask].copy()
        base_metrics = summarize_slice(baseline)
        ret_metrics = summarize_slice(retained)
        blocked_settled = blocked[blocked["settled_flag"]]
        out.append(
            {
                "variant": name,
                "original_rows": base_metrics["rows"],
                "original_settled": base_metrics["settled"],
                "original_hit_rate": base_metrics["hit_rate"],
                "original_roi": base_metrics["roi"],
                "retained_rows": ret_metrics["rows"],
                "retained_settled": ret_metrics["settled"],
                "blocked_rows": int(len(blocked)),
                "retained_hit_rate": ret_metrics["hit_rate"],
                "retained_roi": ret_metrics["roi"],
                "prevented_losses": int(blocked_settled["hit_flag"].eq(0).sum()),
                "missed_winners": int(blocked_settled["hit_flag"].eq(1).sum()),
                "blocked_unsettled": int((~blocked["settled_flag"]).sum()),
            }
        )
    return pd.DataFrame(out)


def build_group_summary(rows: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for name, mask in variant_masks(rows).items():
        retained = rows[mask].copy()
        if retained.empty:
            continue
        group = (
            retained.groupby(group_cols, dropna=False)
            .apply(lambda g: pd.Series(summarize_slice(g)), include_groups=False)
            .reset_index()
        )
        group.insert(0, "variant", name)
        parts.append(group)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def build_reason_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        codes = [code for code in str(row.get("reason_codes") or "").split("|") if code]
        if not codes:
            codes = ["NO_FLAG"]
        for code in codes:
            records.append(
                {
                    "reason_code": code,
                    "market": row.get("market_norm"),
                    "deploy_tier": row.get("deploy_tier_norm"),
                    "settled_flag": row.get("settled_flag"),
                    "hit_flag": row.get("hit_flag"),
                    "profit_units": row.get("profit_units"),
                }
            )
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    return (
        frame.groupby(["reason_code", "market", "deploy_tier"], dropna=False)
        .apply(lambda g: pd.Series(summarize_slice(g.rename(columns={"market": "market_norm"}))), include_groups=False)
        .reset_index()
        .sort_values(["losses", "rows"], ascending=False)
    )


def write_outputs(
    outdir: Path,
    rows: pd.DataFrame,
    *,
    scored_root: Path,
    allow_lineup_shadow: bool,
    max_files: int,
    trusted_fixture_only: bool,
    pre_filter_rows: int,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    summary = build_variant_summary(rows)
    by_market = build_group_summary(rows, ["market_norm"])
    by_market_tier = build_group_summary(rows, ["market_norm", "deploy_tier_norm"])
    by_league = build_group_summary(rows, ["market_norm", "league"])
    by_reason = build_reason_summary(rows)
    decisions_cols = [
        "window_id",
        "match_date",
        "league",
        "home_team_name",
        "away_team_name",
        "fixture_key",
        "market_norm",
        "bookie_pick",
        "deploy_tier_norm",
        "hit_flag",
        "profit_units",
        "league_health_bucket",
        "rolling_prior_settled",
        "rolling_prior_hit_rate",
        "intelligence_risk_score",
        "intelligence_risk_band",
        "reason_codes",
        "lineup_shadow_mode",
        "fixture_identity_join_hit",
        "fixture_identity_match_tier",
        "fixture_identity_confidence",
        "fixture_identity_coverage_caveat",
    ]
    rows.to_csv(outdir / "WALKFORWARD_INTELLIGENCE_OVERLAY_DECISIONS.csv", index=False)
    rows[[col for col in decisions_cols if col in rows.columns]].to_csv(
        outdir / "WALKFORWARD_INTELLIGENCE_OVERLAY_DECISION_SLIM.csv",
        index=False,
    )
    summary.to_csv(outdir / "WALKFORWARD_INTELLIGENCE_OVERLAY_SUMMARY.csv", index=False)
    by_market.to_csv(outdir / "WALKFORWARD_INTELLIGENCE_OVERLAY_BY_MARKET.csv", index=False)
    by_market_tier.to_csv(outdir / "WALKFORWARD_INTELLIGENCE_OVERLAY_BY_MARKET_TIER.csv", index=False)
    by_league.to_csv(outdir / "WALKFORWARD_INTELLIGENCE_OVERLAY_BY_LEAGUE.csv", index=False)
    by_reason.to_csv(outdir / "WALKFORWARD_INTELLIGENCE_OVERLAY_BY_REASON.csv", index=False)

    meta = {
        "generated_at": utc_now(),
        "scored_root": str(scored_root),
        "allow_lineup_shadow": allow_lineup_shadow,
        "trusted_fixture_only": trusted_fixture_only,
        "max_files": max_files,
        "pre_filter_rows": pre_filter_rows,
        "rows": int(len(rows)),
        "settled": int(rows["settled_flag"].sum()),
        "outputs": {
            "summary_csv": str(outdir / "WALKFORWARD_INTELLIGENCE_OVERLAY_SUMMARY.csv"),
            "decisions_csv": str(outdir / "WALKFORWARD_INTELLIGENCE_OVERLAY_DECISIONS.csv"),
            "summary_md": str(outdir / "SUMMARY.md"),
        },
    }
    lines = [
        "# Walk-Forward Intelligence Overlay Backtest",
        "",
        f"Generated: `{meta['generated_at']}`",
        "",
        "Research-only. No deploy routing or live gates changed.",
        "",
        "## Inputs",
        f"- scored root: `{scored_root}`",
        f"- pre-filter scored rows: `{pre_filter_rows}`",
        f"- scored rows: `{len(rows)}`",
        f"- settled rows: `{int(rows['settled_flag'].sum())}`",
        f"- lineup feature mode: `{'shadow-confirmed enabled' if allow_lineup_shadow else 'excluded by default for timestamp safety'}`",
        f"- trusted fixture only: `{trusted_fixture_only}`",
        "",
        "## Variant Summary",
        markdown_table(summary),
        "",
        "## Market Summary",
        markdown_table(by_market),
        "",
        "## Reason-Code Summary",
        markdown_table(by_reason.head(40) if not by_reason.empty else by_reason),
        "",
        "## Interpretation Rules",
        "- This proves sidecar risk detection only; it does not prove a production policy.",
        "- Rolling league health is computed from prior settled rows only inside the scored walk-forward order.",
        "- API team/player rolling features are treated as snapshot-proxy evidence.",
        "- If a fixture identity map is supplied, trusted mapped API fixture keys are used before date/team fallback joins.",
        "- Current lineup feature files are excluded unless `--allow-lineup-shadow` is explicitly used, because confirmed-lineup timing must be proven separately.",
        "- Team-goals 1.5 is handled by the existing model-only shadow backtest and should be read beside this report.",
    ]
    (outdir / "summary.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-root", default=str(DEFAULT_SCORED_ROOT))
    parser.add_argument("--features-dir", default=str(DEFAULT_FEATURES_DIR))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--fixture-identity-map", default="", help="Optional trusted fixture identity map CSV from scripts/build_fixture_identity_map.py.")
    parser.add_argument("--trusted-fixture-only", action="store_true", help="Restrict overlay scoring to rows with trusted fixture identity coverage.")
    parser.add_argument("--min-prior-settled", type=int, default=15)
    parser.add_argument("--allow-lineup-shadow", action="store_true")
    parser.add_argument("--max-files", type=int, default=0, help="Smoke mode: score only the first N scored files.")
    args = parser.parse_args()

    scored_root = Path(args.scored_root)
    rows = load_scored(scored_root, max_files=args.max_files)
    if rows.empty:
        raise SystemExit(f"No scored FTR/BTTS/OU25 rows found under {scored_root}")
    rows = compute_result(rows)
    rows = add_rolling_health(rows, min_prior=args.min_prior_settled)
    identity_map = load_fixture_identity_map(Path(args.fixture_identity_map)) if args.fixture_identity_map else pd.DataFrame()
    rows = attach_api_sidecars(
        rows,
        Path(args.features_dir),
        allow_lineup_shadow=args.allow_lineup_shadow,
        fixture_identity_map=identity_map,
    )
    pre_filter_rows = len(rows)
    if args.trusted_fixture_only:
        if identity_map.empty:
            raise SystemExit("--trusted-fixture-only requires --fixture-identity-map with usable rows.")
        rows = rows[truthy(rows["fixture_identity_join_hit"])].copy()
        if rows.empty:
            raise SystemExit("No rows remained after trusted fixture identity filtering.")
    rows = add_flags(rows)
    write_outputs(
        Path(args.outdir),
        rows,
        scored_root=scored_root,
        allow_lineup_shadow=args.allow_lineup_shadow,
        max_files=args.max_files,
        trusted_fixture_only=args.trusted_fixture_only,
        pre_filter_rows=pre_filter_rows,
    )
    print(f"Rows: {len(rows)}")
    print(f"Settled: {int(rows['settled_flag'].sum())}")
    print(f"Outputs: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
