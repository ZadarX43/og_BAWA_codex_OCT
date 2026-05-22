#!/usr/bin/env python3
"""Track outcomes for expanded player-intelligence live shadow rows.

Research-only sidecar. This consumes the unified live shadow dashboard and
grades supported player/team event rows against normalized API-Football actuals
when available. It does not create priced odds, deploy picks, slips, or
production routing changes.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.api_football.team_name_map import normalize_team_name
from scripts.build_player_event_live_interaction_features import norm_text


DEFAULT_SHADOW_BOARD = (
    ROOT
    / "reports"
    / "2026-05-07"
    / "live_shadow_research_dashboard_with_keeper_saves"
    / "2026_05_02_to_2026_05_04"
    / "2026_05_02_to_2026_05_04__LIVE_SHADOW_RESEARCH_DASHBOARD.csv"
)
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-07" / "player_event_shadow_outcome_tracker_v2"
DEFAULT_ACTUAL_ROOTS = (
    ROOT / "reports" / "2026-05-07" / "api_current_player_window_2026_04_24_to_2026_05_04" / "normalized",
    ROOT / "data_sources" / "api_football" / "normalized",
)


PLAYER_MARKET_TARGETS = {
    "PLAYER_CARD_0_5_SIDE_AWARE_WATCH": ("player_cards_total", 1, "actual_player_card_ge1"),
    "PLAYER_FOULED_0_5_INTERACTION_WATCH": ("fouls_drawn", 1, "actual_fouled_ge1"),
    "PLAYER_FOULED_1_5_INTERACTION_WATCH": ("fouls_drawn", 2, "actual_fouled_ge2"),
    "PLAYER_SHOTS_1_5_INTERACTION_WATCH": ("shots_total", 2, "actual_shots_ge2"),
    "PLAYER_SHOTS_2_5_INTERACTION_WATCH": ("shots_total", 3, "actual_shots_ge3"),
    "PLAYER_SOT_0_5_INTERACTION_WATCH": ("shots_on_target", 1, "actual_sot_ge1"),
    "PLAYER_SHOTS_1_5_CONFIRMED_STARTER_ATTACK_WATCH": ("shots_total", 2, "actual_shots_ge2"),
    "PLAYER_SHOTS_2_5_CONFIRMED_STARTER_ATTACK_WATCH": ("shots_total", 3, "actual_shots_ge3"),
    "PLAYER_SOT_0_5_CONFIRMED_STARTER_ATTACK_WATCH": ("shots_on_target", 1, "actual_sot_ge1"),
    "PLAYER_SOT_1_5_CONFIRMED_STARTER_ATTACK_WATCH": ("shots_on_target", 2, "actual_sot_ge2"),
    "PLAYER_TACKLES_1_5_LIVE_SHADOW": ("tackles", 2, "actual_tackles_ge2"),
    "PLAYER_TACKLES_2_5_LIVE_SHADOW": ("tackles", 3, "actual_tackles_ge3"),
    "KEEPER_SAVES_1_5_LIVE_SHADOW": ("saves", 2, "actual_keeper_saves_ge2"),
    "KEEPER_SAVES_2_5_LIVE_SHADOW": ("saves", 3, "actual_keeper_saves_ge3"),
    "KEEPER_SAVES_3_5_LIVE_SHADOW": ("saves", 4, "actual_keeper_saves_ge4"),
    "KEY_PASSES_0_5_LIVE_SHADOW": ("passes_key", 1, "actual_key_passes_ge1"),
    "KEY_PASSES_1_5_LIVE_SHADOW": ("passes_key", 2, "actual_key_passes_ge2"),
    "ASSIST_0_5_LIVE_WATCH": ("assists", 1, "actual_assists_ge1"),
}

TEAM_MARKET_TARGETS = {
    "TEAM_CORNERS_4_5_LIVE_SHADOW": ("corners_for", 5, "actual_team_corners_ge5"),
    "TEAM_CORNERS_5_5_LIVE_SHADOW": ("corners_for", 6, "actual_team_corners_ge6"),
    "TEAM_CARDS_1_5_SIDE_AWARE_WATCH": ("team_cards_total", 2, "actual_team_cards_ge2"),
}

SUPPORTED_STAGES = set(PLAYER_MARKET_TARGETS) | set(TEAM_MARKET_TARGETS)


def num(values: Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def norm_team(value: Any, league_tag: Any = None) -> str:
    text = norm_text(normalize_team_name(value, str(league_tag) if league_tag is not None else None))
    text = re.sub(r"\b(fc|afc|vfb|vfl|1899)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_file_tag(path: Path, prefix: str) -> tuple[str, int] | None:
    match = re.match(rf"{re.escape(prefix)}__(.+)__(\d{{4}})\.csv$", path.name)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def read_selected(path: Path, cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    header = pd.read_csv(path, nrows=0)
    usecols = [col for col in cols if col in header.columns]
    if not usecols:
        return pd.DataFrame()
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def source_priority(root: Path) -> int:
    return 0 if "reports/2026-05-07/api_current_player_window" in str(root) else 1


def load_fixtures(root: Path, league_tag: str, season_tag: int) -> pd.DataFrame:
    fixture_cols = [
        "fixture_id",
        "fixture_key",
        "match_date",
        "home_team_id",
        "away_team_id",
        "home_team_name",
        "away_team_name",
        "status",
    ]
    return read_selected(root / f"fixtures_master__{league_tag}__{season_tag}.csv", fixture_cols)


def enrich_fixture_team_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["fixture_id", "team_id", "home_team_id", "away_team_id"]:
        if col in out.columns:
            out[col] = num(out[col]).astype("Int64")
    out["team_name"] = np.where(
        out["team_id"].eq(out["home_team_id"]),
        out["home_team_name"],
        out["away_team_name"],
    )
    out["player_team_side"] = np.where(out["team_id"].eq(out["home_team_id"]), "HOME", "AWAY")
    out["match_date"] = pd.to_datetime(out["match_date"], errors="coerce").dt.date.astype("string")
    out["home_team_norm"] = [norm_team(team) for team in out["home_team_name"]]
    out["away_team_norm"] = [norm_team(team) for team in out["away_team_name"]]
    out["team_name_norm"] = [norm_team(team) for team in out["team_name"]]
    return out


def load_player_actuals(actual_roots: list[Path]) -> pd.DataFrame:
    stats_cols = [
        "fixture_id",
        "team_id",
        "player_id",
        "player_name",
        "position",
        "minutes",
        "started_flag",
        "subbed_on_flag",
        "subbed_off_flag",
        "shots_total",
        "shots_on_target",
        "passes_key",
        "assists",
        "fouls_drawn",
        "fouls_committed",
        "tackles",
        "yellow_cards",
        "red_cards",
        "saves",
        "goals_conceded",
    ]
    frames: list[pd.DataFrame] = []
    for root in actual_roots:
        for stats_path in sorted(root.glob("match_player_stats__*.csv")):
            parsed = parse_file_tag(stats_path, "match_player_stats")
            if parsed is None:
                continue
            league_tag, season_tag = parsed
            stats = read_selected(stats_path, stats_cols)
            fixtures = load_fixtures(root, league_tag, season_tag)
            if stats.empty or fixtures.empty:
                continue
            merged = enrich_fixture_team_fields(stats.merge(fixtures, on="fixture_id", how="left"))
            merged["league_tag"] = league_tag
            merged["season_tag"] = season_tag
            merged["player_name_norm"] = merged["player_name"].map(norm_text)
            merged["position_norm"] = merged.get("position", "").astype("string").fillna("").str.upper().str.strip()
            for col in [
                "minutes",
                "started_flag",
                "subbed_on_flag",
                "subbed_off_flag",
                "shots_total",
                "shots_on_target",
                "passes_key",
                "assists",
                "fouls_drawn",
                "fouls_committed",
                "tackles",
                "yellow_cards",
                "red_cards",
                "saves",
                "goals_conceded",
            ]:
                if col not in merged.columns:
                    merged[col] = 0.0
                merged[col] = num(merged[col]).fillna(0.0)
            merged["player_cards_total"] = (
                num(merged.get("yellow_cards", pd.Series(0, index=merged.index))).fillna(0)
                + num(merged.get("red_cards", pd.Series(0, index=merged.index))).fillna(0)
            )
            fixture_red_cards = merged.groupby("fixture_id")["red_cards"].transform("sum")
            team_red_cards = merged.groupby(["fixture_id", "team_id"])["red_cards"].transform("sum")
            merged["fixture_red_cards_total"] = fixture_red_cards
            merged["team_red_cards_total"] = team_red_cards
            merged["opponent_red_cards_total"] = fixture_red_cards - team_red_cards
            merged["_source_priority"] = source_priority(root)
            frames.append(merged)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.sort_values("_source_priority").drop_duplicates(["fixture_id", "team_id", "player_id"], keep="first")
    return out.drop(columns=["_source_priority"], errors="ignore")


def load_team_actuals(actual_roots: list[Path]) -> pd.DataFrame:
    stats_cols = [
        "fixture_id",
        "team_id",
        "team_name",
        "is_home",
        "shots_total",
        "shots_on_goal",
        "corners_for",
        "fouls_for",
        "yellow_cards",
        "red_cards",
    ]
    frames: list[pd.DataFrame] = []
    for root in actual_roots:
        for stats_path in sorted(root.glob("match_team_stats__*.csv")):
            parsed = parse_file_tag(stats_path, "match_team_stats")
            if parsed is None:
                continue
            league_tag, season_tag = parsed
            stats = read_selected(stats_path, stats_cols)
            fixtures = load_fixtures(root, league_tag, season_tag)
            if stats.empty or fixtures.empty:
                continue
            merged = enrich_fixture_team_fields(stats.merge(fixtures, on="fixture_id", how="left"))
            merged["league_tag"] = league_tag
            merged["season_tag"] = season_tag
            for col in ["shots_total", "shots_on_goal", "corners_for", "fouls_for", "yellow_cards"]:
                if col not in merged.columns:
                    merged[col] = np.nan
                merged[col] = num(merged[col])
            if "red_cards" not in merged.columns:
                merged["red_cards"] = np.nan
            merged["red_cards"] = num(merged["red_cards"])
            merged["team_cards_total"] = (
                num(merged.get("yellow_cards", pd.Series(0, index=merged.index))).fillna(0)
                + num(merged.get("red_cards", pd.Series(0, index=merged.index))).fillna(0)
            )
            merged["_source_priority"] = source_priority(root)
            frames.append(merged)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.sort_values("_source_priority").drop_duplicates(["fixture_id", "team_id"], keep="first")
    return out.drop(columns=["_source_priority"], errors="ignore")


def prepare_shadow(shadow: pd.DataFrame) -> pd.DataFrame:
    stage = shadow.get("shadow_stage", pd.Series("", index=shadow.index)).astype(str)
    out = shadow[stage.isin(SUPPORTED_STAGES)].copy()
    if out.empty:
        return out
    for col in ["player_name", "team_name", "home_team_name", "away_team_name", "league", "fixture_key"]:
        if col not in out.columns:
            out[col] = ""
    out["match_date"] = pd.to_datetime(out["match_date"], errors="coerce").dt.date.astype("string")
    out["home_team_norm"] = [norm_team(team) for team in out["home_team_name"]]
    out["away_team_norm"] = [norm_team(team) for team in out["away_team_name"]]
    out["team_name_norm"] = [norm_team(team) for team in out["team_name"]]
    out["player_name_norm"] = out["player_name"].map(norm_text)
    out["_shadow_row_id"] = np.arange(len(out))
    return out


def actual_keep_cols(kind: str) -> list[str]:
    common = [
        "fixture_key",
        "match_date",
        "home_team_norm",
        "away_team_norm",
        "team_name_norm",
        "fixture_id",
        "league_tag",
        "season_tag",
        "status",
    ]
    if kind == "player":
        return common + [
            "player_name_norm",
            "player_name",
            "position",
            "position_norm",
            "minutes",
            "started_flag",
            "subbed_on_flag",
            "subbed_off_flag",
            "shots_total",
            "shots_on_target",
            "passes_key",
            "assists",
            "fouls_drawn",
            "fouls_committed",
            "tackles",
            "yellow_cards",
            "red_cards",
            "player_cards_total",
            "fixture_red_cards_total",
            "team_red_cards_total",
            "opponent_red_cards_total",
            "saves",
            "goals_conceded",
        ]
    return common + [
        "shots_total",
        "shots_on_goal",
        "corners_for",
        "fouls_for",
        "yellow_cards",
        "red_cards",
        "team_cards_total",
    ]


def coalesce_actual_columns(out: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    def one_dim_frame_value(frame: pd.DataFrame, col: str) -> pd.Series:
        values = frame[col]
        if isinstance(values, pd.DataFrame):
            return values.bfill(axis=1).iloc[:, 0]
        return values

    for col in cols:
        actual_col = f"{col}_actual"
        if actual_col in out.columns:
            if col not in out.columns:
                out[col] = np.nan
            current = one_dim_frame_value(out, col)
            actual = one_dim_frame_value(out, actual_col)
            out = out.loc[:, ~out.columns.duplicated()].copy()
            out[col] = current.where(current.notna(), actual)
    return out


def join_player_actuals(shadow: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    player_shadow = shadow[shadow["shadow_stage"].astype(str).isin(PLAYER_MARKET_TARGETS)].copy()
    if player_shadow.empty:
        return player_shadow
    if actuals.empty:
        out = player_shadow.copy()
        out["outcome_status"] = "PENDING_NO_PLAYER_ACTUALS"
        return out

    actual_keep = actuals[[c for c in actual_keep_cols("player") if c in actuals.columns]].copy()
    by_fixture_key = player_shadow.merge(
        actual_keep,
        on=["fixture_key", "team_name_norm", "player_name_norm"],
        how="left",
        suffixes=("", "_actual"),
    )
    matched = by_fixture_key["fixture_id"].notna()
    exact = by_fixture_key[matched].copy()
    leftovers = by_fixture_key[~matched].drop(columns=[c for c in by_fixture_key.columns if c.endswith("_actual")], errors="ignore")

    fallback = leftovers.merge(
        actual_keep.drop(columns=["fixture_key"], errors="ignore"),
        on=["match_date", "home_team_norm", "away_team_norm", "team_name_norm", "player_name_norm"],
        how="left",
        suffixes=("", "_actual"),
    )
    fallback = coalesce_actual_columns(fallback, actual_keep_cols("player"))

    still_missing = fallback[fallback["fixture_id"].isna()].copy()
    fallback_exact = fallback[fallback["fixture_id"].notna()].copy()
    keeper_fill = join_keeper_fallback(still_missing, actual_keep)
    out = pd.concat([exact, fallback_exact, keeper_fill], ignore_index=True, sort=False)
    out = coalesce_actual_columns(out, actual_keep_cols("player"))
    out = out.sort_values("_shadow_row_id").drop_duplicates("_shadow_row_id", keep="first")
    out["outcome_status"] = np.where(out["fixture_id"].notna(), "GRADED", "PENDING_NO_PLAYER_MATCH")
    return out


def join_keeper_fallback(shadow: pd.DataFrame, actual_keep: pd.DataFrame) -> pd.DataFrame:
    if shadow.empty:
        return shadow
    is_keeper = shadow["shadow_stage"].astype(str).str.startswith("KEEPER_SAVES_")
    non_keeper = shadow[~is_keeper].copy()
    keeper_shadow = shadow[is_keeper].copy()
    if keeper_shadow.empty:
        return non_keeper

    gk = actual_keep.copy()
    if "position_norm" in gk.columns:
        gk = gk[gk["position_norm"].astype(str).eq("G")]
    elif "saves" in gk.columns:
        gk = gk[num(gk["saves"]).gt(0)]
    if gk.empty:
        return pd.concat([non_keeper, keeper_shadow], ignore_index=True, sort=False)
    gk = gk.sort_values(["started_flag", "minutes", "saves"], ascending=[False, False, False])
    gk = gk.drop_duplicates(["fixture_key", "team_name_norm"], keep="first")

    joined = keeper_shadow.merge(
        gk,
        on=["fixture_key", "team_name_norm"],
        how="left",
        suffixes=("", "_actual"),
    )
    joined = coalesce_actual_columns(joined, actual_keep_cols("player"))
    missing = joined[joined["fixture_id"].isna()].drop(
        columns=[c for c in joined.columns if c.endswith("_actual")],
        errors="ignore",
    )
    matched = joined[joined["fixture_id"].notna()].copy()
    if missing.empty:
        return pd.concat([non_keeper, matched], ignore_index=True, sort=False)

    gk_fallback = actual_keep.copy()
    if "position_norm" in gk_fallback.columns:
        gk_fallback = gk_fallback[gk_fallback["position_norm"].astype(str).eq("G")]
    gk_fallback = gk_fallback.sort_values(["started_flag", "minutes", "saves"], ascending=[False, False, False])
    gk_fallback = gk_fallback.drop_duplicates(
        ["match_date", "home_team_norm", "away_team_norm", "team_name_norm"],
        keep="first",
    )
    date_joined = missing.merge(
        gk_fallback.drop(columns=["fixture_key"], errors="ignore"),
        on=["match_date", "home_team_norm", "away_team_norm", "team_name_norm"],
        how="left",
        suffixes=("", "_actual"),
    )
    date_joined = coalesce_actual_columns(date_joined, actual_keep_cols("player"))
    return pd.concat([non_keeper, matched, date_joined], ignore_index=True, sort=False)


def join_team_actuals(shadow: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    team_shadow = shadow[shadow["shadow_stage"].astype(str).isin(TEAM_MARKET_TARGETS)].copy()
    if team_shadow.empty:
        return team_shadow
    if actuals.empty:
        out = team_shadow.copy()
        out["outcome_status"] = "PENDING_NO_TEAM_ACTUALS"
        return out

    actual_keep = actuals[[c for c in actual_keep_cols("team") if c in actuals.columns]].copy()
    by_fixture_key = team_shadow.merge(
        actual_keep,
        on=["fixture_key", "team_name_norm"],
        how="left",
        suffixes=("", "_actual"),
    )
    matched = by_fixture_key["fixture_id"].notna()
    exact = by_fixture_key[matched].copy()
    leftovers = by_fixture_key[~matched].drop(columns=[c for c in by_fixture_key.columns if c.endswith("_actual")], errors="ignore")
    fallback = leftovers.merge(
        actual_keep.drop(columns=["fixture_key"], errors="ignore"),
        on=["match_date", "home_team_norm", "away_team_norm", "team_name_norm"],
        how="left",
        suffixes=("", "_actual"),
    )
    out = pd.concat([exact, fallback], ignore_index=True, sort=False)
    out = coalesce_actual_columns(out, actual_keep_cols("team"))
    out = out.sort_values("_shadow_row_id").drop_duplicates("_shadow_row_id", keep="first")
    no_team_stats = actuals.empty
    pending_label = "PENDING_NO_TEAM_ACTUALS" if no_team_stats else "PENDING_NO_TEAM_MATCH"
    out["outcome_status"] = np.where(out["fixture_id"].notna(), "GRADED", pending_label)
    return out


def score_outcomes(tracked: pd.DataFrame) -> pd.DataFrame:
    out = tracked.copy()
    out["actual_stat_col"] = ""
    out["actual_threshold"] = np.nan
    out["actual_stat_value"] = np.nan
    out["actual_hit"] = np.nan
    targets = {**PLAYER_MARKET_TARGETS, **TEAM_MARKET_TARGETS}
    for stage, (stat_col, threshold, hit_col) in targets.items():
        mask = out["shadow_stage"].astype(str).eq(stage)
        out.loc[mask, "actual_stat_col"] = stat_col
        out.loc[mask, "actual_threshold"] = threshold
        if stat_col not in out.columns:
            out[stat_col] = np.nan
        values = num(out.loc[mask, stat_col])
        out.loc[mask, "actual_stat_value"] = values
        graded = mask & out["outcome_status"].astype(str).eq("GRADED")
        out.loc[graded, "actual_hit"] = values.ge(threshold).astype(float)
        out.loc[mask, hit_col] = out.loc[mask, "actual_hit"]
    out = add_state_change_flags(out)
    return out


def add_state_change_flags(tracked: pd.DataFrame) -> pd.DataFrame:
    out = tracked.copy()
    if out.empty:
        return out

    is_player_stage = out.get("shadow_stage", pd.Series("", index=out.index)).astype(str).isin(PLAYER_MARKET_TARGETS)
    minutes = num(out.get("minutes", pd.Series(np.nan, index=out.index)))
    started = num(out.get("started_flag", pd.Series(np.nan, index=out.index))).fillna(0)
    subbed_on = num(out.get("subbed_on_flag", pd.Series(np.nan, index=out.index))).fillna(0)
    subbed_off = num(out.get("subbed_off_flag", pd.Series(np.nan, index=out.index))).fillna(0)

    out["substitution_context_flag"] = "NOT_PLAYER_MARKET"
    out.loc[is_player_stage, "substitution_context_flag"] = "NO_SUB_CONTEXT_AVAILABLE"
    out.loc[is_player_stage & minutes.ge(80), "substitution_context_flag"] = "FULL_OR_NEAR_FULL_MINUTES"
    out.loc[is_player_stage & started.eq(1) & minutes.ge(60) & minutes.lt(80), "substitution_context_flag"] = "NORMAL_SUB_OFF_WINDOW"
    out.loc[is_player_stage & started.eq(1) & minutes.lt(60), "substitution_context_flag"] = "EARLY_SUB_OFF_CONTEXT"
    out.loc[is_player_stage & subbed_off.eq(1) & minutes.lt(60), "substitution_context_flag"] = "EARLY_SUB_OFF_CONTEXT"
    out.loc[is_player_stage & subbed_on.eq(1), "substitution_context_flag"] = "SUB_APPEARANCE_CONTEXT"
    out.loc[is_player_stage & started.eq(0) & subbed_on.eq(0) & minutes.fillna(0).le(0), "substitution_context_flag"] = "DID_NOT_PLAY_CONTEXT"

    player_reds = num(out.get("red_cards", pd.Series(0, index=out.index))).fillna(0)
    fixture_reds = num(out.get("fixture_red_cards_total", pd.Series(0, index=out.index))).fillna(0)
    team_reds = num(out.get("team_red_cards_total", pd.Series(0, index=out.index))).fillna(0)
    opponent_reds = num(out.get("opponent_red_cards_total", pd.Series(0, index=out.index))).fillna(0)
    out["red_card_context_flag"] = "NO_RED_CARD_CONTEXT"
    out.loc[fixture_reds.gt(0), "red_card_context_flag"] = "FIXTURE_RED_CARD_CONTEXT"
    out.loc[team_reds.gt(0), "red_card_context_flag"] = "TEAM_RED_CARD_CONTEXT"
    out.loc[opponent_reds.gt(0), "red_card_context_flag"] = "OPPONENT_RED_CARD_CONTEXT"
    out.loc[player_reds.gt(0), "red_card_context_flag"] = "PLAYER_RED_CARD_CONTEXT"

    out["named_player_grading_mode"] = np.where(is_player_stage, "NAMED_PLAYER_ONLY_DEFAULT", "FIXTURE_OR_TEAM_MARKET")
    out["player_sub_swap_review_candidate_flag"] = (
        is_player_stage
        & out["substitution_context_flag"].isin(["EARLY_SUB_OFF_CONTEXT", "SUB_APPEARANCE_CONTEXT", "DID_NOT_PLAY_CONTEXT"])
    ).astype(int)
    out["player_sub_swap_review_mode"] = np.where(
        out["player_sub_swap_review_candidate_flag"].eq(1),
        "SEPARATE_ROLE_CHAIN_GRADING_REQUIRED",
        "NOT_SUB_SWAP_REVIEW_ROW",
    )
    return out


def summarize(tracked: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if tracked.empty:
        return pd.DataFrame(columns=group_cols + ["rows", "graded", "hits", "hit_rate", "pending"])
    rows = []
    for key, group in tracked.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        graded_mask = group["outcome_status"].astype(str).eq("GRADED")
        graded = group[graded_mask]
        hits = float(num(graded.get("actual_hit", pd.Series(dtype=float))).sum()) if not graded.empty else 0.0
        rows.append(
            {
                **dict(zip(group_cols, key)),
                "rows": int(len(group)),
                "graded": int(len(graded)),
                "hits": int(hits),
                "hit_rate": float(hits / len(graded)) if len(graded) else np.nan,
                "pending": int((~graded_mask).sum()),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["graded", "rows"], ascending=[False, False]).reset_index(drop=True)


def markdown_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df.empty:
        return "_No rows._"
    work = df.head(max_rows).copy()
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
        else:
            work[col] = work[col].astype("string").fillna("")
    lines = ["| " + " | ".join(work.columns) + " |", "| " + " | ".join(["---"] * len(work.columns)) + " |"]
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "/") for col in work.columns) + " |")
    return "\n".join(lines)


def write_report(
    outdir: Path,
    tracked: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    shadow_board: Path,
    actual_roots: list[Path],
) -> None:
    graded = int(tracked["outcome_status"].astype(str).eq("GRADED").sum()) if not tracked.empty else 0
    pending = int(len(tracked) - graded)
    lines = [
        "# Player Event Shadow Outcome Tracker V2",
        "",
        "Research-only outcome tracker for expanded live shadow intelligence rows.",
        "",
        "## Safety",
        "- No priced player-prop odds.",
        "- No deploy routing, tiers, slips, or production rulebook changes.",
        "- Pending rows mean API actuals were not joinable yet.",
        "- Normal player outcomes are named-player only; Player Sub Swap requires separate role-chain grading.",
        "- Red-card and substitution context flags are attached for review and should not be mixed into clean pre-match evidence blindly.",
        "",
        "## Inputs",
        f"- shadow board: `{shadow_board}`",
        *[f"- actual root: `{root}`" for root in actual_roots],
        "",
        "## Supported Shadow Stages",
        *[f"- `{stage}`" for stage in sorted(SUPPORTED_STAGES)],
        "",
        "## Overall",
        f"- tracked rows: `{len(tracked)}`",
        f"- graded rows: `{graded}`",
        f"- pending rows: `{pending}`",
        "",
        "## By Stage / Priority",
        markdown_table(summaries.get("stage_priority", pd.DataFrame()), max_rows=80),
        "",
        "## By Stage / League",
        markdown_table(summaries.get("stage_league", pd.DataFrame()), max_rows=100),
        "",
        "## By Shadow Family",
        markdown_table(summaries.get("family", pd.DataFrame()), max_rows=60),
        "",
        "## Outcome Status",
        markdown_table(summaries.get("status", pd.DataFrame()), max_rows=60),
        "",
        "## By Tactical Role",
        markdown_table(summaries.get("role", pd.DataFrame()), max_rows=80),
    ]
    (outdir / "PLAYER_EVENT_SHADOW_OUTCOME_TRACKER_V2.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-board", type=Path, default=DEFAULT_SHADOW_BOARD)
    parser.add_argument("--actuals-root", type=Path, action="append", default=[])
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    actual_roots = args.actuals_root or list(DEFAULT_ACTUAL_ROOTS)
    args.outdir.mkdir(parents=True, exist_ok=True)

    if not args.shadow_board.exists():
        raise SystemExit(f"Missing shadow board: {args.shadow_board}")
    shadow = prepare_shadow(pd.read_csv(args.shadow_board, low_memory=False))
    player_actuals = load_player_actuals(actual_roots)
    team_actuals = load_team_actuals(actual_roots)

    tracked = pd.concat(
        [
            join_player_actuals(shadow, player_actuals),
            join_team_actuals(shadow, team_actuals),
        ],
        ignore_index=True,
        sort=False,
    )
    if not tracked.empty:
        tracked = tracked.sort_values("_shadow_row_id").reset_index(drop=True)
    tracked = score_outcomes(tracked)

    summaries = {
        "stage_priority": summarize(tracked, ["shadow_stage", "watch_priority"]),
        "stage_league": summarize(tracked, ["shadow_stage", "league"]),
        "family": summarize(tracked, ["shadow_family"]),
        "status": summarize(tracked, ["shadow_stage", "outcome_status"]),
        "role": summarize(tracked, ["shadow_stage", "tactical_role"]) if "tactical_role" in tracked.columns else pd.DataFrame(),
    }

    tracked.to_csv(args.outdir / "PLAYER_EVENT_SHADOW_OUTCOME_TRACKER_ROWS.csv", index=False)
    for name, summary in summaries.items():
        summary.to_csv(args.outdir / f"PLAYER_EVENT_SHADOW_OUTCOME_{name.upper()}_SUMMARY.csv", index=False)
    write_report(args.outdir, tracked, summaries, args.shadow_board, actual_roots)

    print(f"WROTE {args.outdir}")
    print(f"tracked_rows={len(tracked)} graded={int(tracked['outcome_status'].astype(str).eq('GRADED').sum()) if not tracked.empty else 0}")
    if not summaries["stage_priority"].empty:
        print(summaries["stage_priority"].to_string(index=False))


if __name__ == "__main__":
    main()
