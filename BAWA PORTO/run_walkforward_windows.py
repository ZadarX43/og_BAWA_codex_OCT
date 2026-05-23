#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ============================================================================
# Config
# ============================================================================
# Adjust these only if your repo uses different filenames.
BOOKIE_SCRIPT = "bookie_allmarkets.py"
DEPLOY_SCRIPT = "deploy_rulebook.py"
ACCA_BUILDER_SCRIPT = "acca_builder.py"

# Where all walk-forward outputs will be written.
DEFAULT_BASE_OUTDIR = Path("predictions_output/walk_forward")

# Where the native scripts write their own CSVs.
DEFAULT_PREDICTIONS_DIR = Path("predictions_output")
DEFAULT_MERGED_DIR = Path("Matches/__merged__")
DEFAULT_CALENDAR_FLAGS_DIR = Path("predictions_output/calendar_flags")

# Markets to summarize from scored deploy output.
SUPPORTED_MARKETS = {
    "ftr": {
        "hit_col": "ftr_hit",
        "actual_col": "actual_ftr",
        "selection_col": "selection",
        "default_selection_filter": None,
    },
    "ou25": {
        "hit_col": "ou25_hit",
        "actual_col": "actual_over25",
        "selection_col": "selection",
        "default_selection_filter": "OVER25",
    },
    "btts": {
        "hit_col": "btts_yes_hit",
        "actual_col": "actual_btts_yes",
        "selection_col": "selection",
        "default_selection_filter": "YES",
    },
    "tg15": {
        "hit_col": "tg15_hit",
        "actual_col": "actual_tg15",
        "selection_col": "selection",
        "default_selection_filter": None,
    },
    "tg25": {
        "hit_col": "tg25_hit",
        "actual_col": "actual_tg25",
        "selection_col": "selection",
        "default_selection_filter": None,
    },
}
CALENDAR_REGIME_DEFAULT = "NORMAL"

WINDOW_REGIME_MAP = {
    "w05_2024_09_13_2024_09_17": "POST_FIFA_BREAK",
    "w06_2024_11_08_2024_11_12": "POST_UCL_LEAGUE_PHASE",
    "w07_2024_11_29_2024_12_03": "POST_UCL_LEAGUE_PHASE",
    "w08_2024_12_13_2024_12_17": "POST_UCL_LEAGUE_PHASE",
    "w13_2025_09_12_2025_09_16": "POST_FIFA_BREAK",
    "w14_2025_10_17_2025_10_21": "POST_FIFA_BREAK",
    "w15_2025_11_21_2025_11_25": "NOV_FIFA_PLUS_PRE_UCL",
    "w17_2026_02_13_2026_02_17": "PRE_UCL_KNOCKOUT_FIRST_LEG",
    "w18_2026_03_13_2026_03_17": "POST_R16_SECOND_LEG_MARCH",
}
CALENDAR_OVERLAY_RULES = [
    {
        "calendar_regime": "POST_FIFA_BREAK",
        "btts_leg_multiplier": 0.75,
        "ftr_leg_multiplier": 1.00,
        "stake_multiplier": 0.90,
        "overlay_note": "Reduce BTTS exposure after FIFA/international breaks; FTR unchanged.",
    },
    {
        "calendar_regime": "PRE_UCL_KNOCKOUT_FIRST_LEG",
        "btts_leg_multiplier": 1.00,
        "ftr_leg_multiplier": 0.75,
        "stake_multiplier": 0.90,
        "overlay_note": "Reduce FTR exposure before UCL knockout first legs; BTTS unchanged.",
    },
    {
        "calendar_regime": "NOV_FIFA_PLUS_PRE_UCL",
        "btts_leg_multiplier": 0.70,
        "ftr_leg_multiplier": 0.70,
        "stake_multiplier": 0.75,
        "overlay_note": "Highest-risk annual overlap: reduce both BTTS and FTR, lower stake size.",
    },
    {
        "calendar_regime": "POST_UCL_LEAGUE_PHASE",
        "btts_leg_multiplier": 1.00,
        "ftr_leg_multiplier": 1.00,
        "stake_multiplier": 1.00,
        "overlay_note": "Deploy normally after UCL league-phase matchdays.",
    },
    {
        "calendar_regime": "POST_R16_SECOND_LEG_MARCH",
        "btts_leg_multiplier": 0.75,
        "ftr_leg_multiplier": 1.00,
        "stake_multiplier": 0.90,
        "overlay_note": "Reduce BTTS after UCL R16 second legs in March; FTR unchanged.",
    },
]


CALENDAR_OVERLAY_DEFAULT = {
    "calendar_regime": "NORMAL",
    "btts_leg_multiplier": 1.00,
    "ftr_leg_multiplier": 1.00,
    "stake_multiplier": 1.00,
    "overlay_note": "No calendar overlay adjustment.",
}

# ============================================================================
# ACCA Products Config
# ============================================================================
ACCA_PRODUCTS = [
    {"template": "BTTS_ONLY", "k": 6, "slips_per_k": 6},
    {"template": "OU25_ONLY", "k": 6, "slips_per_k": 6},
    {"template": "COMBINED_ALLMARKETS", "k": 6, "slips_per_k": 6},
    {"template": "COMBINED_ALLMARKETS", "k": 8, "slips_per_k": 6},
    {"template": "FTR_ONLY", "k": 10, "slips_per_k": 6},
    {"template": "COMBINED_ALLMARKETS", "k": 10, "slips_per_k": 6},
]

def acca_template_tag(template: str, k: int) -> str:
    return f"{template}_K{k}"

# ============================================================================
# Data models
# ============================================================================
@dataclass
class WindowSpec:
    window_id: str
    date_from: str
    date_to: str

    @property
    def range_key(self) -> tuple[str, str]:
        return (self.date_from, self.date_to)


# ============================================================================
# Small helpers
# ============================================================================
def log(msg: str) -> None:
    print(msg, flush=True)


def safe_series_str(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    return df.get(col, pd.Series(default, index=df.index)).astype("string").fillna(default)


def safe_series_num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df.get(col, pd.Series(np.nan, index=df.index)), errors="coerce")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_calendar_regime(window_id: str, cli_override: str | None = None) -> str:
    override = str(cli_override or "").strip().upper()
    if override:
        return override
    return WINDOW_REGIME_MAP.get(str(window_id).strip(), CALENDAR_REGIME_DEFAULT)


def maybe_stamp_calendar_overlay(df: pd.DataFrame, regime: str | None) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    if df.empty:
        out = df.copy()
        for col in [
            "calendar_regime",
            "btts_leg_multiplier",
            "ftr_leg_multiplier",
            "stake_multiplier",
            "overlay_note",
        ]:
            if col not in out.columns:
                out[col] = pd.Series(dtype="object")
        return out

    out = df.copy()
    regime_key = str(regime or "").strip().upper() or CALENDAR_REGIME_DEFAULT

    regime_table = {
        "NORMAL": {
            "btts_leg_multiplier": 1.00,
            "ftr_leg_multiplier": 1.00,
            "stake_multiplier": 1.00,
            "overlay_note": "Normal deployment window; no calendar adjustment applied.",
        },
        "NONE": {
            "btts_leg_multiplier": 1.00,
            "ftr_leg_multiplier": 1.00,
            "stake_multiplier": 1.00,
            "overlay_note": "No calendar regime applied.",
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

    payload = regime_table.get(regime_key, regime_table[CALENDAR_REGIME_DEFAULT])
    out["calendar_regime"] = regime_key
    out["btts_leg_multiplier"] = float(payload["btts_leg_multiplier"])
    out["ftr_leg_multiplier"] = float(payload["ftr_leg_multiplier"])
    out["stake_multiplier"] = float(payload["stake_multiplier"])
    out["overlay_note"] = str(payload["overlay_note"])
    return out

def get_calendar_overlay_rule(calendar_regime: object) -> dict[str, object]:
    regime = str(calendar_regime or "").strip().upper()
    for rule in CALENDAR_OVERLAY_RULES:
        if str(rule["calendar_regime"]).strip().upper() == regime:
            return dict(rule)
    return dict(CALENDAR_OVERLAY_DEFAULT)


def stamp_calendar_overlay_fields(df: pd.DataFrame, calendar_regime: str) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()

    out = df.copy()
    if out.empty:
        out["calendar_regime"] = calendar_regime
        out["btts_leg_multiplier"] = CALENDAR_OVERLAY_DEFAULT["btts_leg_multiplier"]
        out["ftr_leg_multiplier"] = CALENDAR_OVERLAY_DEFAULT["ftr_leg_multiplier"]
        out["stake_multiplier"] = CALENDAR_OVERLAY_DEFAULT["stake_multiplier"]
        out["overlay_note"] = CALENDAR_OVERLAY_DEFAULT["overlay_note"]
        return out

    rule = get_calendar_overlay_rule(calendar_regime)

    out["calendar_regime"] = str(rule["calendar_regime"])
    out["btts_leg_multiplier"] = float(rule["btts_leg_multiplier"])
    out["ftr_leg_multiplier"] = float(rule["ftr_leg_multiplier"])
    out["stake_multiplier"] = float(rule["stake_multiplier"])
    out["overlay_note"] = str(rule["overlay_note"])
    return out

def newest_file(paths: Iterable[Path]) -> Path | None:
    files = [p for p in paths if p.exists() and p.is_file()]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def copy_if_needed(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if src.resolve() == dst.resolve():
        return
    dst.write_bytes(src.read_bytes())


def slugify_league_name(name: str) -> str:
    text = str(name or "").strip()
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    parts = [p for p in text.split() if p]
    return "_".join(parts)


def build_merged_candidates(merged_dir: Path, league_name: str) -> list[Path]:
    raw_name = str(league_name or "").strip()
    slug = slugify_league_name(raw_name)
    hyphen_preserving = raw_name.replace("/", " ").replace(" ", "_")

    ordered_slugs: list[str] = []
    for candidate in [slug, hyphen_preserving, raw_name.replace(" ", "_")]:
        candidate = str(candidate or "").strip()
        if candidate and candidate not in ordered_slugs:
            ordered_slugs.append(candidate)

    paths: list[Path] = []
    for candidate_slug in ordered_slugs:
        paths.append(merged_dir / f"{candidate_slug}__merged.csv")
        paths.append(merged_dir / f"{candidate_slug}.csv")
    return paths


def normalize_fixture_key_series(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def normalize_team_series(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("")
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("ascii")
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )


def normalize_date_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


# Inserted: build_completed_mask
def build_completed_mask(df: pd.DataFrame) -> pd.Series:
    status = safe_series_str(df, "status").str.lower().str.strip()
    if status.eq("").all():
        return pd.Series(True, index=df.index)

    complete_tokens = {
        "complete", "completed", "finished", "final", "full time", "full-time", "ft",
    }
    incomplete_tokens = {
        "incomplete", "scheduled", "not started", "postponed", "postponed/cancelled",
        "cancelled", "canceled", "abandoned", "suspended", "live", "ongoing",
        "in progress", "pending",
    }

    is_complete = status.isin(complete_tokens)
    is_incomplete = status.isin(incomplete_tokens)

    # Fallback: if status values are unfamiliar, treat rows with both goal columns present
    # as completed only when the status is not explicitly incomplete.
    return is_complete | (~is_incomplete)


def build_actuals_from_merged(merged_df: pd.DataFrame) -> pd.DataFrame:
    out = merged_df.copy()

    out["match_date_norm"] = normalize_date_series(out.get("match_date", out.get("date_GMT", pd.Series(np.nan, index=out.index))))
    out["fixture_key_norm"] = normalize_fixture_key_series(out.get("fixture_key", pd.Series("", index=out.index)))
    out["home_team_norm"] = normalize_team_series(out.get("home_team_name", pd.Series("", index=out.index)))
    out["away_team_norm"] = normalize_team_series(out.get("away_team_name", pd.Series("", index=out.index)))
    out["home_team_goal_count"] = pd.to_numeric(out.get("home_team_goal_count", np.nan), errors="coerce")
    out["away_team_goal_count"] = pd.to_numeric(out.get("away_team_goal_count", np.nan), errors="coerce")
    out["status"] = safe_series_str(out, "status")

    home_goals = pd.to_numeric(out.get("home_team_goal_count", np.nan), errors="coerce")
    away_goals = pd.to_numeric(out.get("away_team_goal_count", np.nan), errors="coerce")
    total_goals = pd.to_numeric(out.get("total_goal_count", np.nan), errors="coerce")
    completed_mask = build_completed_mask(out)

    out["actual_ftr"] = np.where(
        completed_mask & home_goals.gt(away_goals), "HOME",
        np.where(
            completed_mask & home_goals.lt(away_goals), "AWAY",
            np.where(completed_mask & home_goals.eq(away_goals), "DRAW", "")
        )
    )
    out["actual_over25"] = np.where(
        completed_mask & total_goals.notna(),
        (total_goals > 2.5).astype(float),
        np.nan,
    )
    out["actual_btts_yes"] = np.where(
        completed_mask & home_goals.notna() & away_goals.notna(),
        ((home_goals > 0) & (away_goals > 0)).astype(float),
        np.nan,
    )
    out["actual_tg15_home"] = np.where(
        completed_mask & home_goals.notna(),
        (home_goals >= 2).astype(float),
        np.nan,
    )
    out["actual_tg15_away"] = np.where(
        completed_mask & away_goals.notna(),
        (away_goals >= 2).astype(float),
        np.nan,
    )
    out["actual_tg25_home"] = np.where(
        completed_mask & home_goals.notna(),
        (home_goals >= 3).astype(float),
        np.nan,
    )
    out["actual_tg25_away"] = np.where(
        completed_mask & away_goals.notna(),
        (away_goals >= 3).astype(float),
        np.nan,
    )

    keep_cols = [
        "league",
        "match_date_norm",
        "fixture_key_norm",
        "home_team_norm",
        "away_team_norm",
        "actual_ftr",
        "actual_over25",
        "actual_btts_yes",
        "actual_tg15_home",
        "actual_tg15_away",
        "actual_tg25_home",
        "actual_tg25_away",
        "home_team_goal_count",
        "away_team_goal_count",
        "status",
    ]
    out = out[keep_cols].copy()

    # Prefer rows that actually have graded outcomes before falling back to the latest duplicate.
    out["__has_actual"] = (
        out["actual_ftr"].astype("string").fillna("").ne("")
        | out["actual_over25"].notna()
        | out["actual_btts_yes"].notna()
        | out["actual_tg15_home"].notna()
        | out["actual_tg15_away"].notna()
        | out["actual_tg25_home"].notna()
        | out["actual_tg25_away"].notna()
    ).astype(int)
    out = out.sort_values(["league", "fixture_key_norm", "__has_actual"])
    out = out.drop_duplicates(subset=["league", "fixture_key_norm"], keep="last")
    out = out.drop(columns=["__has_actual"])
    return out


def load_window_merged_actuals(source_df: pd.DataFrame, merged_dir: Path) -> pd.DataFrame:
    if "league" not in source_df.columns:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    leagues = sorted({str(x).strip() for x in source_df["league"].dropna().unique() if str(x).strip()})
    missing: list[str] = []

    for league_name in leagues:
        chosen: Path | None = None
        for candidate in build_merged_candidates(merged_dir, league_name):
            if candidate.exists() and candidate.is_file():
                chosen = candidate
                break
        if chosen is None:
            missing.append(league_name)
            continue

        merged_df = pd.read_csv(chosen, low_memory=False)
        if "status" in merged_df.columns:
            status_counts = (
                safe_series_str(merged_df, "status")
                .str.lower()
                .str.strip()
                .value_counts(dropna=False)
                .head(10)
                .to_dict()
            )
            log(f"[merged] {league_name} status sample: {status_counts}")
        if "league" not in merged_df.columns:
            merged_df["league"] = league_name
        frames.append(build_actuals_from_merged(merged_df))

    if missing:
        log(f"[merged] missing merged CSVs for leagues: {missing}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=["league", "fixture_key_norm"], keep="last")
    return combined


def enrich_with_merged_actuals(df: pd.DataFrame, merged_actuals: pd.DataFrame) -> pd.DataFrame:
    if merged_actuals.empty:
        return df.copy()

    out = df.copy()
    out["match_date_norm"] = normalize_date_series(out.get("match_date", pd.Series(np.nan, index=out.index)))
    out["fixture_key_norm"] = normalize_fixture_key_series(out.get("fixture_key", pd.Series("", index=out.index)))
    out["home_team_norm"] = normalize_team_series(out.get("home_team_name", pd.Series("", index=out.index)))
    out["away_team_norm"] = normalize_team_series(out.get("away_team_name", pd.Series("", index=out.index)))
    out["home_team_goal_count"] = pd.to_numeric(out.get("home_team_goal_count", np.nan), errors="coerce")
    out["away_team_goal_count"] = pd.to_numeric(out.get("away_team_goal_count", np.nan), errors="coerce")
    out["status"] = safe_series_str(out, "status")

    actual_cols = [
        "actual_ftr",
        "actual_over25",
        "actual_btts_yes",
        "actual_tg15_home",
        "actual_tg15_away",
        "actual_tg25_home",
        "actual_tg25_away",
    ]
    extra_merge_cols = [
        "home_team_goal_count",
        "away_team_goal_count",
        "status",
    ]
    for col in actual_cols:
        if col not in out.columns:
            out[col] = np.nan

    merge_on_fixture = out.merge(
        merged_actuals,
        on=["league", "fixture_key_norm"],
        how="left",
        suffixes=("", "__merged"),
    )

    still_missing_fixture = (
        merge_on_fixture["actual_ftr__merged"].isna()
        & merge_on_fixture["actual_over25__merged"].isna()
        & merge_on_fixture["actual_btts_yes__merged"].isna()
        & merge_on_fixture["actual_tg15_home__merged"].isna()
        & merge_on_fixture["actual_tg15_away__merged"].isna()
        & merge_on_fixture["actual_tg25_home__merged"].isna()
        & merge_on_fixture["actual_tg25_away__merged"].isna()
    )

    if still_missing_fixture.any():
        fallback_left = merge_on_fixture.loc[still_missing_fixture, [
            "league", "match_date_norm", "home_team_norm", "away_team_norm"
        ]].copy()
        fallback_left["__row_id"] = fallback_left.index
        fallback_join = fallback_left.merge(
            merged_actuals,
            on=["league", "match_date_norm", "home_team_norm", "away_team_norm"],
            how="left",
        )
        fallback_join = fallback_join.set_index("__row_id")
        for col in actual_cols:
            merged_col = f"{col}__merged"
            merge_on_fixture.loc[fallback_join.index, merged_col] = merge_on_fixture.loc[fallback_join.index, merged_col].fillna(fallback_join[col])

    for col in actual_cols:
        merged_col = f"{col}__merged"
        if col == "actual_ftr":
            out[col] = out[col].astype("string")
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        candidate = merge_on_fixture[merged_col]
        if col == "actual_ftr":
            current = out[col].astype("string").fillna("")
            fillv = candidate.astype("string").fillna("")
            out[col] = current.mask(current.eq(""), fillv)
        else:
            current = pd.to_numeric(out[col], errors="coerce")
            fillv = pd.to_numeric(candidate, errors="coerce")
            out[col] = current.fillna(fillv)

    for col in extra_merge_cols:
        merged_col = f"{col}__merged"
        candidate = merge_on_fixture[merged_col]
        if col == "status":
            current = safe_series_str(out, col)
            fillv = candidate.astype("string").fillna("")
            out[col] = current.mask(current.eq(""), fillv)
        else:
            current = pd.to_numeric(out.get(col, np.nan), errors="coerce")
            fillv = pd.to_numeric(candidate, errors="coerce")
            out[col] = current.fillna(fillv)

    out = out.drop(columns=[c for c in ["match_date_norm", "fixture_key_norm", "home_team_norm", "away_team_norm"] if c in out.columns])
    return out

def build_calendar_flag_candidates(calendar_dir: Path, window: WindowSpec) -> list[Path]:
    return [
        calendar_dir / f"CALENDAR_FLAGS_{window.date_from}_to_{window.date_to}.csv",
        calendar_dir / f"MATCH_CALENDAR_FLAGS_{window.date_from}_to_{window.date_to}.csv",
        calendar_dir / window.window_id / f"CALENDAR_FLAGS_{window.date_from}_to_{window.date_to}.csv",
        calendar_dir / window.window_id / f"MATCH_CALENDAR_FLAGS_{window.date_from}_to_{window.date_to}.csv",
    ]


def resolve_calendar_flags_csv(window: WindowSpec, calendar_dir: Path) -> Path | None:
    for candidate in build_calendar_flag_candidates(calendar_dir, window):
        if candidate.exists() and candidate.is_file():
            return candidate

    patterns = [
        f"*{window.date_from}_to_{window.date_to}*.csv",
        f"*{window.window_id}*.csv",
    ]
    hits: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for hit in calendar_dir.rglob(pattern):
            if not hit.exists() or not hit.is_file():
                continue
            resolved = hit.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            hits.append(hit)

    if not hits:
        return None
    return max(hits, key=lambda p: p.stat().st_mtime)


def enrich_with_calendar_flags(df: pd.DataFrame, calendar_flags: pd.DataFrame) -> pd.DataFrame:
    if df.empty or calendar_flags.empty:
        return df.copy()

    out = df.copy()
    cal = calendar_flags.copy()

    out["match_date_norm"] = normalize_date_series(out.get("match_date", pd.Series(np.nan, index=out.index)))
    out["fixture_key_norm"] = normalize_fixture_key_series(out.get("fixture_key", pd.Series("", index=out.index)))
    out["home_team_norm"] = normalize_team_series(out.get("home_team_name", pd.Series("", index=out.index)))
    out["away_team_norm"] = normalize_team_series(out.get("away_team_name", pd.Series("", index=out.index)))

    cal["match_date_norm"] = normalize_date_series(cal.get("match_date", cal.get("date_GMT", pd.Series(np.nan, index=cal.index))))
    cal["fixture_key_norm"] = normalize_fixture_key_series(cal.get("fixture_key", pd.Series("", index=cal.index)))
    cal["home_team_norm"] = normalize_team_series(cal.get("home_team_name", pd.Series("", index=cal.index)))
    cal["away_team_norm"] = normalize_team_series(cal.get("away_team_name", pd.Series("", index=cal.index)))

    cal_payload = cal.copy()

    merged = out.merge(
        cal_payload.drop_duplicates(subset=["fixture_key_norm"], keep="last"),
        on=["fixture_key_norm"],
        how="left",
        suffixes=("", "__calendar"),
    )

    probe_cols = [c for c in merged.columns if c.endswith("__calendar")]
    missing_calendar = merged[probe_cols].isna().all(axis=1) if probe_cols else pd.Series(False, index=merged.index)

    if missing_calendar.any():
        fallback_left = merged.loc[missing_calendar, ["match_date_norm", "home_team_norm", "away_team_norm"]].copy()
        fallback_left["__row_id"] = fallback_left.index
        fallback_join = fallback_left.merge(
            cal_payload.drop_duplicates(subset=["match_date_norm", "home_team_norm", "away_team_norm"], keep="last"),
            on=["match_date_norm", "home_team_norm", "away_team_norm"],
            how="left",
            suffixes=("", "__calendar"),
        ).set_index("__row_id")

        for col in cal_payload.columns:
            if col in {"match_date_norm", "fixture_key_norm", "home_team_norm", "away_team_norm"}:
                continue
            target_col = col if col not in merged.columns else f"{col}__calendar"
            if target_col in fallback_join.columns:
                merged.loc[fallback_join.index, target_col] = (
                    merged.loc[fallback_join.index, target_col].fillna(fallback_join[target_col])
                )

    for col in cal_payload.columns:
        if col in {"match_date_norm", "fixture_key_norm", "home_team_norm", "away_team_norm"}:
            continue
        calendar_col = f"{col}__calendar"
        if calendar_col not in merged.columns:
            continue
        if col not in merged.columns:
            merged[col] = merged[calendar_col]
        else:
            current = merged[col]
            if pd.api.types.is_numeric_dtype(current):
                merged[col] = pd.to_numeric(current, errors="coerce").fillna(
                    pd.to_numeric(merged[calendar_col], errors="coerce")
                )
            else:
                current_text = current.astype("string").fillna("")
                fill_text = merged[calendar_col].astype("string").fillna("")
                merged[col] = current_text.mask(current_text.eq(""), fill_text)

    merged = merged.drop(columns=[c for c in merged.columns if c.endswith("__calendar")], errors="ignore")
    merged = merged.drop(columns=[c for c in ["match_date_norm", "fixture_key_norm", "home_team_norm", "away_team_norm"] if c in merged.columns])
    return merged

def run_cmd(cmd: list[str], log_path: Path, cwd: Path | None = None) -> None:
    ensure_dir(log_path.parent)
    rendered = " ".join(shlex.quote(c) for c in cmd)
    log(f"\n[run] {rendered}")
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(f"COMMAND: {rendered}\n\n")
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {rendered}\nSee log: {log_path}")

def run_acca_builder_for_window(
    deploy_input: str,
    out_dir: str,
    tag: str,
    templates: list[str],
    k_values: list[int],
    slips_per_k: int = 6,
    top_n_legs: int = 20,
    log_path: Path | None = None,
    cwd: Path | None = None,
) -> None:
    templates_clean = [str(x).strip() for x in templates if str(x).strip()]
    k_values_clean = sorted({int(x) for x in k_values if int(x) > 0})
    if not templates_clean or not k_values_clean:
        raise ValueError("run_acca_builder_for_window requires at least one template and one positive k value.")

    cmd = [
        sys.executable,
        ACCA_BUILDER_SCRIPT,
        "build_v2_slips",
        "--deploy-input", str(deploy_input),
        "--tag", str(tag),
        "--templates", ",".join(sorted(set(templates_clean))),
        "--k-values", ",".join(str(x) for x in k_values_clean),
        "--slips-per-k", str(int(slips_per_k)),
        "--top-n-legs", str(int(top_n_legs)),
        "--out-dir", str(out_dir),
    ]

    if log_path is None:
        subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)
        return

    run_cmd(cmd, log_path=log_path, cwd=cwd)


def build_acca_window_product_args(
    templates_override: str | None = None,
    k_values_override: str | None = None,
    slips_per_k_override: int | None = None,
) -> tuple[list[str], list[int], int]:
    templates = sorted({str(x["template"]).strip() for x in ACCA_PRODUCTS if str(x.get("template", "")).strip()})
    k_values = sorted({int(x["k"]) for x in ACCA_PRODUCTS if int(x.get("k", 0)) > 0})
    slips_per_k = max(int(x.get("slips_per_k", 6)) for x in ACCA_PRODUCTS) if ACCA_PRODUCTS else 6

    override_templates = [x.strip() for x in str(templates_override or "").split(",") if x.strip()]
    if override_templates:
        templates = sorted(set(override_templates))

    override_k_values = []
    for raw in str(k_values_override or "").split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value > 0:
            override_k_values.append(value)
    if override_k_values:
        k_values = sorted(set(override_k_values))

    if slips_per_k_override is not None:
        try:
            slips_per_k = max(1, int(slips_per_k_override))
        except Exception:
            pass

    return templates, k_values, slips_per_k

# ============================================================================
# Manifest loading
# ============================================================================
def load_manifest(manifest_path: Path) -> list[WindowSpec]:
    rows: list[WindowSpec] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"window_id", "date_from", "date_to"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest must contain columns: {sorted(required)}")
        for row in reader:
            rows.append(
                WindowSpec(
                    window_id=row["window_id"].strip(),
                    date_from=row["date_from"].strip(),
                    date_to=row["date_to"].strip(),
                )
            )
    if not rows:
        raise ValueError("Manifest is empty.")
    return dedupe_manifest_windows(rows)


# Dedupe function for manifest windows
def dedupe_manifest_windows(windows: list[WindowSpec]) -> list[WindowSpec]:
    seen_window_ids: set[str] = set()
    seen_ranges: dict[tuple[str, str], str] = {}
    deduped: list[WindowSpec] = []

    for window in windows:
        if window.window_id in seen_window_ids:
            log(f"[manifest] duplicate window_id skipped: {window.window_id}")
            continue
        seen_window_ids.add(window.window_id)

        if window.range_key in seen_ranges:
            kept_window_id = seen_ranges[window.range_key]
            log(
                f"[manifest] duplicate date range skipped: {window.window_id} duplicates "
                f"{kept_window_id} for {window.date_from} -> {window.date_to}"
            )
            continue

        seen_ranges[window.range_key] = window.window_id
        deduped.append(window)

    if not deduped:
        raise ValueError("Manifest contains no unique windows after duplicate removal.")

    return deduped


# ============================================================================
# Paths
# ============================================================================
def window_paths(base_outdir: Path, window: WindowSpec) -> dict[str, Path]:
    root = base_outdir / window.window_id
    paths = {
        "root": root,
        "source_dir": root / "01_source",
        "deploy_dir": root / "02_deploy",
        "scored_dir": root / "03_scored",
        "reports_dir": root / "04_reports",
        "logs_dir": root / "logs",
    }
    for p in paths.values():
        if p.suffix == "":
            ensure_dir(p)
    return paths


# ============================================================================
# Pipeline steps
# ============================================================================
def _normalize_source_implied_min_for_bookie(source_implied_min: str | None) -> str:
    raw = str(source_implied_min or "").strip()
    if raw == "":
        return raw
    try:
        value = float(raw)
    except Exception:
        return raw
    if value > 1.0:
        value = value / 100.0
    normalized = f"{value:.6f}".rstrip("0").rstrip(".")
    return normalized or "0"

# === Added: normalize legacy deploy extra args ===
def _normalize_deploy_extra_args(extra_args: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(extra_args):
        token = str(extra_args[i])

        if token == "--selector":
            selector_value = ""
            if (i + 1) < len(extra_args):
                selector_value = str(extra_args[i + 1]).strip()
                i += 2
            else:
                i += 1

            if selector_value:
                out.extend(["--ftr-profile", selector_value])
                log(
                    f"[deploy-build] normalized legacy --selector {selector_value} -> --ftr-profile {selector_value}"
                )
            else:
                log("[deploy-build] dropped legacy --selector with no value")
            continue

        if token.startswith("--selector="):
            selector_value = token.split("=", 1)[1].strip()
            if selector_value:
                out.extend(["--ftr-profile", selector_value])
                log(
                    f"[deploy-build] normalized legacy --selector {selector_value} -> --ftr-profile {selector_value}"
                )
            else:
                log("[deploy-build] dropped legacy --selector= with no value")
            i += 1
            continue

        out.append(token)
        i += 1

    return out
def build_source_command(
    window: WindowSpec,
    extra_args: list[str],
    implied_min: str | None = None,
    source_outdir: Path | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        BOOKIE_SCRIPT,
        "--date-from", window.date_from,
        "--date-to", window.date_to,
    ]
    implied_min_clean = str(implied_min or "").strip()
    if implied_min_clean:
        normalized_implied_min = _normalize_source_implied_min_for_bookie(implied_min)
        cmd.extend(["--implied-min", normalized_implied_min])
        log(
            f"[source-build] source_implied_min_raw={implied_min} "
            f"normalized_for_bookie={normalized_implied_min}"
        )
    extra_args_lower = [str(token).lower() for token in extra_args]
    has_explicit_outdir = any(
        token in {"--outdir", "--run-dir", "--run-tag"}
        or token.startswith("--outdir=")
        or token.startswith("--run-dir=")
        or token.startswith("--run-tag=")
        for token in extra_args_lower
    )
    if source_outdir is not None and not has_explicit_outdir:
        cmd.extend(["--outdir", str(source_outdir)])
        log(f"[source-build] auto-wired bookie output dir -> {source_outdir}")
    cmd.extend(extra_args)
    return cmd


def build_deploy_command(source_csv: Path, outdir: Path, extra_args: list[str]) -> list[str]:
    cmd = [
        sys.executable,
        DEPLOY_SCRIPT,
        "--src", str(source_csv),
        "--outdir", str(outdir),
    ]
    normalized_extra_args = _normalize_deploy_extra_args(extra_args)
    cmd.extend(normalized_extra_args)
    return cmd

def resolve_source_csv(window: WindowSpec, predictions_dir: Path, implied_min: str | None) -> Path:
    day_dir = predictions_dir / window.date_to
    prefix_parts = ["BOOKIE"]
    if implied_min:
        prefix_parts.append(f"IMP{implied_min}")
    prefix_parts.append("ALLMARKETS")
    prefix = "_".join(prefix_parts)

    exact_name = f"{prefix}_{window.date_from}_to_{window.date_to}.csv"

    strict_patterns = [
        exact_name,
        f"{prefix}_{window.date_from}_to_{window.date_to}*.csv",
    ]

    fallback_patterns = [
        f"BOOKIE_*_ALLMARKETS_{window.date_from}_to_{window.date_to}*.csv",
        f"*{window.date_from}_to_{window.date_to}*.csv",
    ]

    strict_hits: list[Path] = []
    seen: set[Path] = set()

    def _maybe_add(pool: list[Path], path: Path) -> None:
        if not path.exists() or not path.is_file():
            return
        name_upper = path.name.upper()
        if (
            "DEPLOY_TIER_ELITE" in name_upper
            or "DEPLOY_TIER_STANDARD" in name_upper
            or "DEPLOY_TIER_OBSERVE" in name_upper
            or "DEPLOY_CANDIDATES_AFTER_GATES" in name_upper
            or "DEPLOY_CANDIDATES_RAW" in name_upper
            or "DEPLOY_COMBINED" in name_upper
        ):
            return
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        pool.append(path)

    for pattern in strict_patterns:
        if any(ch in pattern for ch in "*?[]"):
            for hit in day_dir.glob(pattern):
                _maybe_add(strict_hits, hit)
        else:
            _maybe_add(strict_hits, day_dir / pattern)

    for pattern in strict_patterns:
        for hit in predictions_dir.rglob(pattern):
            _maybe_add(strict_hits, hit)

    if strict_hits:
        return max(strict_hits, key=lambda p: p.stat().st_mtime)

    fallback_hits: list[Path] = []
    for pattern in fallback_patterns:
        for hit in day_dir.glob(pattern):
            _maybe_add(fallback_hits, hit)
    for pattern in fallback_patterns:
        for hit in predictions_dir.rglob(pattern):
            _maybe_add(fallback_hits, hit)

    if fallback_hits:
        return max(fallback_hits, key=lambda p: p.stat().st_mtime)

    raise FileNotFoundError(
        f"Could not find source CSV for window {window.window_id} under {predictions_dir}. "
        f"Expected something like {exact_name}."
    )


# Inserted: resolve_source_csv_after_run
def resolve_source_csv_after_run(
    window: WindowSpec,
    predictions_dir: Path,
    implied_min: str | None,
    run_started_ts: float,
) -> Path:
    day_dir = predictions_dir / window.date_to
    prefix_parts = ["BOOKIE"]
    if implied_min:
        prefix_parts.append(f"IMP{implied_min}")
    prefix_parts.append("ALLMARKETS")
    prefix = "_".join(prefix_parts)

    exact_name = f"{prefix}_{window.date_from}_to_{window.date_to}.csv"
    strict_patterns = [
        exact_name,
        f"{prefix}_{window.date_from}_to_{window.date_to}*.csv",
    ]
    fallback_patterns = [
        f"BOOKIE_*_ALLMARKETS_{window.date_from}_to_{window.date_to}*.csv",
        f"*{window.date_from}_to_{window.date_to}*.csv",
    ]

    def _collect_candidates(patterns: list[str]) -> list[Path]:
        candidates: list[Path] = []
        seen: set[Path] = set()

        def _maybe_add(path: Path) -> None:
            if not path.exists() or not path.is_file():
                return
            name_upper = path.name.upper()
            if (
                "DEPLOY_TIER_ELITE" in name_upper
                or "DEPLOY_TIER_STANDARD" in name_upper
                or "DEPLOY_TIER_OBSERVE" in name_upper
                or "DEPLOY_CANDIDATES_AFTER_GATES" in name_upper
                or "DEPLOY_CANDIDATES_RAW" in name_upper
                or "DEPLOY_COMBINED" in name_upper
            ):
                return
            resolved = path.resolve()
            if resolved in seen:
                return
            seen.add(resolved)
            candidates.append(path)

        for pattern in patterns:
            if any(ch in pattern for ch in "*?[]"):
                for hit in day_dir.glob(pattern):
                    _maybe_add(hit)
            else:
                _maybe_add(day_dir / pattern)

        for pattern in patterns:
            for hit in predictions_dir.rglob(pattern):
                _maybe_add(hit)

        return candidates

    strict_candidates = _collect_candidates(strict_patterns)
    if strict_candidates:
        strict_fresh = [p for p in strict_candidates if p.stat().st_mtime >= (run_started_ts - 2.0)]
        strict_pool = strict_fresh if strict_fresh else strict_candidates
        return max(strict_pool, key=lambda p: p.stat().st_mtime)

    fallback_candidates = _collect_candidates(fallback_patterns)
    if fallback_candidates:
        fallback_fresh = [p for p in fallback_candidates if p.stat().st_mtime >= (run_started_ts - 2.0)]
        fallback_pool = fallback_fresh if fallback_fresh else fallback_candidates
        return max(fallback_pool, key=lambda p: p.stat().st_mtime)

    raise FileNotFoundError(
        f"Could not find source CSV for window {window.window_id} under {predictions_dir}. "
        f"Expected something like {exact_name}, but also searched recursively for strict prefix matches and broader fallbacks "
        f"for *{window.date_from}_to_{window.date_to}*.csv."
    )


def _source_log_indicates_no_rows(log_path: Path) -> bool:
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return "No rows produced." in text or "No rows produced" in text


# New helpers for deploy csv resolution
def _is_tier_deploy_filename(path: Path) -> bool:
    name = path.name.upper()
    return (
        "DEPLOY_TIER_ELITE" in name
        or "DEPLOY_TIER_STANDARD" in name
        or "DEPLOY_TIER_OBSERVE" in name
    )

def _is_candidate_deploy_filename(path: Path) -> bool:
    name = path.name.upper()
    return (
        "DEPLOY_CANDIDATES_AFTER_GATES" in name
        or "DEPLOY_CANDIDATES_RAW" in name
    )

# New helper for explicit candidate deploy files
def _append_explicit_candidate_deploy_files(
    candidates: list[Path],
    seen: set[Path],
    source_csv: Path,
    *dirs: Path,
) -> None:
    explicit_names = [
        "DEPLOY_CANDIDATES_AFTER_GATES.csv",
        "DEPLOY_CANDIDATES_RAW.csv",
    ]
    for dir_path in dirs:
        if not dir_path.exists() or not dir_path.is_dir():
            continue
        for name in explicit_names:
            hit = dir_path / name
            if not hit.exists() or not hit.is_file() or hit == source_csv:
                continue
            resolved = hit.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(hit)


def _dedupe_and_sort_deploy_paths(paths: list[Path]) -> list[Path]:
    deduped = {p.resolve(): p for p in paths}

    def _priority(p: Path) -> tuple[int, str, str]:
        name_upper = p.name.upper()
        if _is_tier_deploy_filename(p):
            if p.parent.name == "02_deploy":
                bucket = 0
            elif p.parent.name == "01_source":
                bucket = 1
            else:
                bucket = 2
        elif _is_candidate_deploy_filename(p):
            if p.parent.name == "02_deploy":
                bucket = 3
            elif p.parent.name == "01_source":
                bucket = 4
            else:
                bucket = 5
        else:
            bucket = 6
        return (bucket, p.name, str(p.parent))

    return sorted(deduped.values(), key=_priority)

def resolve_deploy_csvs(window: WindowSpec, predictions_dir: Path, source_csv: Path) -> list[Path]:
    source_dir = source_csv.parent
    requested_profile = _infer_requested_ftr_profile_from_source_csv(source_csv)
    deploy_dir_candidate = source_dir.parent / "02_deploy"
    legacy_walkforward_dir = predictions_dir / "walk_forward" / window.window_id / "02_deploy"

    candidates: list[Path] = []
    seen: set[Path] = set()

    def _maybe_add(path: Path) -> None:
        if not path.exists() or not path.is_file() or path == source_csv:
            return

        is_tier = _is_tier_deploy_filename(path)
        is_candidate = _is_candidate_deploy_filename(path)

        if not is_tier and not is_candidate:
            return

        if is_tier and not _deploy_csv_matches_requested_profile(path, requested_profile):
            return

        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append(path)

    exact_tier_patterns = [
        f"*{window.date_from}_to_{window.date_to}*DEPLOY_TIER_ELITE*.csv",
        f"*{window.date_from}_to_{window.date_to}*DEPLOY_TIER_STANDARD*.csv",
        f"*{window.date_from}_to_{window.date_to}*DEPLOY_TIER_OBSERVE*.csv",
    ]

    candidate_patterns_exact = [
        "DEPLOY_CANDIDATES_AFTER_GATES.csv",
        "DEPLOY_CANDIDATES_RAW.csv",
    ]

    candidate_patterns_broad = [
        "*DEPLOY_CANDIDATES_AFTER_GATES*.csv",
        "*DEPLOY_CANDIDATES_RAW*.csv",
    ]

    broader_tier_patterns = [
        "*DEPLOY_TIER_ELITE*.csv",
        "*DEPLOY_TIER_STANDARD*.csv",
        "*DEPLOY_TIER_OBSERVE*.csv",
    ]

    if deploy_dir_candidate.exists() and deploy_dir_candidate.is_dir():
        for pattern in exact_tier_patterns:
            for hit in sorted(deploy_dir_candidate.glob(pattern)):
                _maybe_add(hit)
        for pattern in candidate_patterns_exact:
            for hit in sorted(deploy_dir_candidate.glob(pattern)):
                _maybe_add(hit)
        _append_explicit_candidate_deploy_files(
            candidates,
            seen,
            source_csv,
            deploy_dir_candidate,
        )

    if legacy_walkforward_dir.exists() and legacy_walkforward_dir.is_dir():
        for pattern in exact_tier_patterns:
            for hit in sorted(legacy_walkforward_dir.glob(pattern)):
                _maybe_add(hit)
        for pattern in candidate_patterns_exact:
            for hit in sorted(legacy_walkforward_dir.glob(pattern)):
                _maybe_add(hit)
        _append_explicit_candidate_deploy_files(
            candidates,
            seen,
            source_csv,
            legacy_walkforward_dir,
        )

    if candidates:
        tier_hits = sum(1 for p in candidates if _is_tier_deploy_filename(p))
        candidate_hits = sum(1 for p in candidates if _is_candidate_deploy_filename(p))
        log(
            f"[resolved] deploy_candidates_pre_dedupe total={len(candidates)} tier={tier_hits} candidate={candidate_hits} profile={requested_profile}"
        )
        return _dedupe_and_sort_deploy_paths(candidates)

    if deploy_dir_candidate.exists() and deploy_dir_candidate.is_dir():
        for pattern in broader_tier_patterns:
            for hit in sorted(deploy_dir_candidate.glob(pattern)):
                _maybe_add(hit)
        for pattern in candidate_patterns_broad:
            for hit in sorted(deploy_dir_candidate.glob(pattern)):
                _maybe_add(hit)
        _append_explicit_candidate_deploy_files(
            candidates,
            seen,
            source_csv,
            deploy_dir_candidate,
        )

    if legacy_walkforward_dir.exists() and legacy_walkforward_dir.is_dir():
        for pattern in broader_tier_patterns:
            for hit in sorted(legacy_walkforward_dir.glob(pattern)):
                _maybe_add(hit)
        for pattern in candidate_patterns_broad:
            for hit in sorted(legacy_walkforward_dir.glob(pattern)):
                _maybe_add(hit)
        _append_explicit_candidate_deploy_files(
            candidates,
            seen,
            source_csv,
            legacy_walkforward_dir,
        )

    if candidates:
        tier_hits = sum(1 for p in candidates if _is_tier_deploy_filename(p))
        candidate_hits = sum(1 for p in candidates if _is_candidate_deploy_filename(p))
        log(
            f"[resolved] deploy_candidates_pre_dedupe total={len(candidates)} tier={tier_hits} candidate={candidate_hits} profile={requested_profile}"
        )
        return _dedupe_and_sort_deploy_paths(candidates)

    for pattern in exact_tier_patterns:
        for hit in sorted(source_dir.glob(pattern)):
            _maybe_add(hit)
    for pattern in candidate_patterns_exact:
        for hit in sorted(source_dir.glob(pattern)):
            _maybe_add(hit)
    _append_explicit_candidate_deploy_files(
        candidates,
        seen,
        source_csv,
        source_dir,
    )
    if candidates:
        tier_hits = sum(1 for p in candidates if _is_tier_deploy_filename(p))
        candidate_hits = sum(1 for p in candidates if _is_candidate_deploy_filename(p))
        log(
            f"[resolved] deploy_candidates_pre_dedupe total={len(candidates)} tier={tier_hits} candidate={candidate_hits} profile={requested_profile}"
        )
        return _dedupe_and_sort_deploy_paths(candidates)

    for pattern in broader_tier_patterns:
        for hit in sorted(source_dir.glob(pattern)):
            _maybe_add(hit)
    for pattern in candidate_patterns_broad:
        for hit in sorted(source_dir.glob(pattern)):
            _maybe_add(hit)
    _append_explicit_candidate_deploy_files(
        candidates,
        seen,
        source_csv,
        source_dir,
    )
    if candidates:
        tier_hits = sum(1 for p in candidates if _is_tier_deploy_filename(p))
        candidate_hits = sum(1 for p in candidates if _is_candidate_deploy_filename(p))
        log(
            f"[resolved] deploy_candidates_pre_dedupe total={len(candidates)} tier={tier_hits} candidate={candidate_hits} profile={requested_profile}"
        )
        return _dedupe_and_sort_deploy_paths(candidates)

    candidate_patterns = [
        f"*{window.date_from}_to_{window.date_to}*DEPLOY*.csv",
        f"*{window.date_from}_to_{window.date_to}*RULEBOOK*.csv",
        f"*{window.date_from}_to_{window.date_to}*ROUTED*.csv",
    ]

    for pattern in candidate_patterns:
        for hit in sorted(source_dir.glob(pattern)):
            _maybe_add(hit)

    if deploy_dir_candidate.exists() and deploy_dir_candidate.is_dir():
        for pattern in candidate_patterns:
            for hit in sorted(deploy_dir_candidate.glob(pattern)):
                _maybe_add(hit)

    if legacy_walkforward_dir.exists() and legacy_walkforward_dir.is_dir():
        for pattern in candidate_patterns:
            for hit in sorted(legacy_walkforward_dir.glob(pattern)):
                _maybe_add(hit)

    if candidates:
        tier_hits = sum(1 for p in candidates if _is_tier_deploy_filename(p))
        candidate_hits = sum(1 for p in candidates if _is_candidate_deploy_filename(p))
        log(
            f"[resolved] deploy_candidates_pre_dedupe total={len(candidates)} tier={tier_hits} candidate={candidate_hits} profile={requested_profile}"
        )
        return _dedupe_and_sort_deploy_paths(candidates)

    recursive_patterns = exact_tier_patterns + broader_tier_patterns
    for pattern in recursive_patterns:
        for hit in predictions_dir.rglob(pattern):
            _maybe_add(hit)
    for explicit_name in ["DEPLOY_CANDIDATES_AFTER_GATES.csv", "DEPLOY_CANDIDATES_RAW.csv"]:
        for hit in predictions_dir.rglob(explicit_name):
            _maybe_add(hit)
    if candidates:
        tier_hits = sum(1 for p in candidates if _is_tier_deploy_filename(p))
        candidate_hits = sum(1 for p in candidates if _is_candidate_deploy_filename(p))
        log(
            f"[resolved] deploy_candidates_pre_dedupe total={len(candidates)} tier={tier_hits} candidate={candidate_hits} profile={requested_profile}"
        )
        return _dedupe_and_sort_deploy_paths(candidates)

    raise FileNotFoundError(
        f"Could not find deploy outputs for window {window.window_id} near {source_csv}. "
        f"Expected tiered files matching {window.date_from}_to_{window.date_to}."
    )

def _infer_requested_ftr_profile_from_source_csv(source_csv: Path) -> str:
    name = source_csv.name.lower()
    if "valueev_aggressive" in name:
        return "valueev_aggressive"
    if "valueev_balanced" in name:
        return "valueev_balanced"
    return "accuracy"


def _deploy_csv_matches_requested_profile(path: Path, requested_profile: str) -> bool:
    name = path.name.lower()

    # Ignore preset/union views and raw candidate dumps.
    if "__deploy_preset__" in name:
        return False
    if "deploy_candidates_after_gates" in name or "deploy_candidates_raw" in name:
        return False

    # Tiered files with no explicit profile suffix are treated as accuracy.
    if requested_profile == "accuracy":
        return ("valueev_aggressive" not in name) and ("valueev_balanced" not in name)

    return requested_profile in name

def resolve_deploy_csvs_after_run(
    window: WindowSpec,
    predictions_dir: Path,
    source_csv: Path,
    run_started_ts: float,
) -> list[Path]:
    source_dir = source_csv.parent
    requested_profile = _infer_requested_ftr_profile_from_source_csv(source_csv)
    deploy_dir_candidate = source_dir.parent / "02_deploy"
    legacy_walkforward_dir = predictions_dir / "walk_forward" / window.window_id / "02_deploy"

    candidates: list[Path] = []
    seen: set[Path] = set()

    def _maybe_add(path: Path) -> None:
        if not path.exists() or not path.is_file() or path == source_csv:
            return

        is_tier = _is_tier_deploy_filename(path)
        is_candidate = _is_candidate_deploy_filename(path)

        if not is_tier and not is_candidate:
            return

        if is_tier and not _deploy_csv_matches_requested_profile(path, requested_profile):
            return

        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append(path)

    def _scan_dir(dir_path: Path, patterns: list[str], recent_only: bool = False) -> None:
        if not dir_path.exists() or not dir_path.is_dir():
            return
        for pattern in patterns:
            for hit in sorted(dir_path.glob(pattern)):
                if recent_only:
                    try:
                        if hit.stat().st_mtime < (run_started_ts - 2.0):
                            continue
                    except OSError:
                        continue
                _maybe_add(hit)

    exact_patterns = [
        f"*{window.date_from}_to_{window.date_to}*DEPLOY_TIER_ELITE*.csv",
        f"*{window.date_from}_to_{window.date_to}*DEPLOY_TIER_STANDARD*.csv",
        f"*{window.date_from}_to_{window.date_to}*DEPLOY_TIER_OBSERVE*.csv",
    ]

    candidate_patterns_exact = [
        "DEPLOY_CANDIDATES_AFTER_GATES.csv",
        "DEPLOY_CANDIDATES_RAW.csv",
    ]

    candidate_patterns_broad = [
        "*DEPLOY_CANDIDATES_AFTER_GATES*.csv",
        "*DEPLOY_CANDIDATES_RAW*.csv",
    ]
    broad_patterns = [
        f"*{window.date_from}_to_{window.date_to}*DEPLOY*.csv",
        f"*{window.date_from}_to_{window.date_to}*RULEBOOK*.csv",
        f"*{window.date_from}_to_{window.date_to}*ROUTED*.csv",
    ]
    outdir_tier_patterns = [
        "*DEPLOY_TIER_ELITE*.csv",
        "*DEPLOY_TIER_STANDARD*.csv",
        "*DEPLOY_TIER_OBSERVE*.csv",
    ]
    outdir_broad_patterns = [
        "*DEPLOY*.csv",
        "*RULEBOOK*.csv",
        "*ROUTED*.csv",
    ]

    # 1) First prefer exact tier files written into explicit walk-forward outdir.
    _scan_dir(deploy_dir_candidate, exact_patterns, recent_only=True)
    if not candidates:
        _scan_dir(legacy_walkforward_dir, exact_patterns, recent_only=False)

    if candidates:
        fresh = [p for p in candidates if p.stat().st_mtime >= (run_started_ts - 2.0)]
        pool = fresh if fresh else candidates
        tier_hits = sum(1 for p in pool if _is_tier_deploy_filename(p))
        candidate_hits = sum(1 for p in pool if _is_candidate_deploy_filename(p))
        result = _dedupe_and_sort_deploy_paths(pool)
        log(
            f"[resolved] deploy_candidates_count={len(result)} tier={tier_hits} candidate={candidate_hits} profile={requested_profile}"
        )
        return result

    # 2) Then prefer exact tier files written beside the source.
    _scan_dir(source_dir, exact_patterns, recent_only=True)

    if candidates:
        fresh = [p for p in candidates if p.stat().st_mtime >= (run_started_ts - 2.0)]
        pool = fresh if fresh else candidates
        tier_hits = sum(1 for p in pool if _is_tier_deploy_filename(p))
        candidate_hits = sum(1 for p in pool if _is_candidate_deploy_filename(p))
        result = _dedupe_and_sort_deploy_paths(pool)
        log(
            f"[resolved] deploy_candidates_count={len(result)} tier={tier_hits} candidate={candidate_hits} profile={requested_profile}"
        )
        return result

    # 3) If filenames were rewritten without exact date token, still prefer tier files.
    _scan_dir(deploy_dir_candidate, outdir_tier_patterns, recent_only=True)
    if not candidates:
        _scan_dir(legacy_walkforward_dir, outdir_tier_patterns, recent_only=False)
    if not candidates:
        _scan_dir(source_dir, outdir_tier_patterns, recent_only=True)

    if candidates:
        fresh = [p for p in candidates if p.stat().st_mtime >= (run_started_ts - 2.0)]
        pool = fresh if fresh else candidates
        tier_hits = sum(1 for p in pool if _is_tier_deploy_filename(p))
        candidate_hits = sum(1 for p in pool if _is_candidate_deploy_filename(p))
        result = _dedupe_and_sort_deploy_paths(pool)
        log(
            f"[resolved] deploy_candidates_count={len(result)} tier={tier_hits} candidate={candidate_hits} profile={requested_profile}"
        )
        return result

    # 4) Only if no tier files were found, fall back to candidate files.
    _scan_dir(deploy_dir_candidate, candidate_patterns_exact, recent_only=True)
    _append_explicit_candidate_deploy_files(
        candidates,
        seen,
        source_csv,
        deploy_dir_candidate,
    )
    if not candidates:
        _scan_dir(legacy_walkforward_dir, candidate_patterns_exact, recent_only=False)
        _append_explicit_candidate_deploy_files(
            candidates,
            seen,
            source_csv,
            legacy_walkforward_dir,
        )
    if not candidates:
        _scan_dir(source_dir, candidate_patterns_exact, recent_only=True)
        _append_explicit_candidate_deploy_files(
            candidates,
            seen,
            source_csv,
            source_dir,
        )

    if not candidates:
        _scan_dir(deploy_dir_candidate, candidate_patterns_broad, recent_only=True)
        _append_explicit_candidate_deploy_files(
            candidates,
            seen,
            source_csv,
            deploy_dir_candidate,
        )
    if not candidates:
        _scan_dir(legacy_walkforward_dir, candidate_patterns_broad, recent_only=False)
        _append_explicit_candidate_deploy_files(
            candidates,
            seen,
            source_csv,
            legacy_walkforward_dir,
        )
    if not candidates:
        _scan_dir(source_dir, candidate_patterns_broad, recent_only=True)
        _append_explicit_candidate_deploy_files(
            candidates,
            seen,
            source_csv,
            source_dir,
        )

    # 5) Recursive fresh tier-file fallback under predictions_output.
    if not candidates:
        recursive_patterns = [
            f"*{window.date_from}_to_{window.date_to}*DEPLOY_TIER_ELITE*.csv",
            f"*{window.date_from}_to_{window.date_to}*DEPLOY_TIER_STANDARD*.csv",
            f"*{window.date_from}_to_{window.date_to}*DEPLOY_TIER_OBSERVE*.csv",
        ]
        for pattern in recursive_patterns:
            for hit in predictions_dir.rglob(pattern):
                try:
                    if hit.stat().st_mtime < (run_started_ts - 2.0):
                        continue
                except OSError:
                    continue
                _maybe_add(hit)
        for explicit_name in ["DEPLOY_CANDIDATES_AFTER_GATES.csv", "DEPLOY_CANDIDATES_RAW.csv"]:
            for hit in predictions_dir.rglob(explicit_name):
                try:
                    if hit.stat().st_mtime < (run_started_ts - 2.0):
                        continue
                except OSError:
                    continue
                _maybe_add(hit)

    # 6) Last-resort recursive broad fallback under predictions_output.
    if not candidates:
        for pattern in broad_patterns:
            for hit in predictions_dir.rglob(pattern):
                try:
                    if hit.stat().st_mtime < (run_started_ts - 2.0):
                        continue
                except OSError:
                    continue
                _maybe_add(hit)

    if not candidates:
        debug_outdir_listing: list[str] = []
        debug_source_listing: list[str] = []
        debug_legacy_listing: list[str] = []

        if deploy_dir_candidate.exists() and deploy_dir_candidate.is_dir():
            debug_outdir_listing = sorted(p.name for p in deploy_dir_candidate.glob("*.csv"))
        if legacy_walkforward_dir.exists() and legacy_walkforward_dir.is_dir():
            debug_legacy_listing = sorted(p.name for p in legacy_walkforward_dir.glob("*.csv"))

        raise FileNotFoundError(
            f"Could not find deploy outputs for window {window.window_id} near {source_csv}. "
            f"Expected tiered files matching {window.date_from}_to_{window.date_to}. "
            f"Outdir CSVs: {debug_outdir_listing} | Source dir CSVs: {debug_source_listing}"
        )
    fresh = [p for p in candidates if p.stat().st_mtime >= (run_started_ts - 2.0)]
    pool = fresh if fresh else candidates
    tier_hits = sum(1 for p in pool if _is_tier_deploy_filename(p))
    candidate_hits = sum(1 for p in pool if _is_candidate_deploy_filename(p))
    result = _dedupe_and_sort_deploy_paths(pool)
    log(
        f"[resolved] deploy_candidates_count={len(result)} tier={tier_hits} candidate={candidate_hits} profile={requested_profile}"
    )
    return result

def score_deploy_file(deploy_csv: Path, scored_csv: Path, merged_actuals: pd.DataFrame | None = None) -> pd.DataFrame:
    df = pd.read_csv(deploy_csv, low_memory=False)
    if merged_actuals is not None and not merged_actuals.empty:
        df = enrich_with_merged_actuals(df, merged_actuals)
    return score_deploy_df(df, scored_csv)


def _infer_deploy_tier_from_path(csv_path: Path) -> str:
    upper_name = csv_path.stem.upper()
    for candidate in ("ELITE", "STANDARD", "OBSERVE"):
        if f"DEPLOY_TIER_{candidate}" in upper_name:
            return candidate
    return "UNKNOWN"


def _value_counts_safe(series: pd.Series) -> dict[str, int]:
    if series is None or series.empty:
        return {}
    return series.astype("string").fillna("").replace("", "EMPTY").value_counts(dropna=False).to_dict()


def log_runner_self_check(
    *,
    native_source_csv: Path,
    source_csv: Path,
    native_deploy_csvs: list[Path],
    combined_deploy_df: pd.DataFrame,
    scored_df: pd.DataFrame,
) -> dict[str, object]:
    log(f"[self-check] source_used={native_source_csv}")
    log(f"[self-check] source_copied={source_csv}")

    tier_resolved = 0
    candidate_resolved = 0
    if native_deploy_csvs:
        tier_counts: dict[str, int] = {}
        for csv_path in native_deploy_csvs:
            tier = _infer_deploy_tier_from_path(csv_path)
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            if tier != "UNKNOWN":
                tier_resolved += 1
            if _is_candidate_deploy_filename(csv_path):
                candidate_resolved += 1
        log(f"[self-check] tier_files_resolved count={len(native_deploy_csvs)} by_tier={tier_counts}")
        for csv_path in native_deploy_csvs:
            log(f"[self-check] tier_file {csv_path}")
    else:
        log("[self-check] tier_files_resolved count=0 by_tier={}")

    if isinstance(combined_deploy_df, pd.DataFrame) and not combined_deploy_df.empty:
        market_counts = _value_counts_safe(safe_series_str(combined_deploy_df, "market").str.lower().str.strip())
        league_counts = _value_counts_safe(safe_series_str(combined_deploy_df, "league").str.strip())
        if "source_tier_file" in combined_deploy_df.columns:
            tier_series = safe_series_str(combined_deploy_df, "source_tier_file").str.upper().str.strip()
        elif "deploy_tier" in combined_deploy_df.columns:
            tier_series = safe_series_str(combined_deploy_df, "deploy_tier").str.upper().str.strip()
        else:
            tier_series = safe_series_str(combined_deploy_df, "tier").str.upper().str.strip()
        tier_counts = _value_counts_safe(tier_series)
        log(f"[self-check] combined_composition rows={len(combined_deploy_df)} tiers={tier_counts}")
        log(f"[self-check] combined_markets={market_counts}")
        log(f"[self-check] combined_leagues={league_counts}")
    else:
        log("[self-check] combined_composition rows=0 tiers={} markets={} leagues={}")

    graded_any = 0
    scored_rows = 0
    if isinstance(scored_df, pd.DataFrame) and not scored_df.empty:
        scored_rows = int(len(scored_df))
        ftr_nonblank = safe_series_str(scored_df, "actual_ftr").ne("") if "actual_ftr" in scored_df.columns else pd.Series(False, index=scored_df.index)
        ou25_nonnull = pd.to_numeric(scored_df.get("actual_over25", np.nan), errors="coerce").notna()
        btts_nonnull = pd.to_numeric(scored_df.get("actual_btts_yes", np.nan), errors="coerce").notna()
        graded_any = (ftr_nonblank | ou25_nonnull | btts_nonnull).sum()
        log(
            f"[self-check] scored_vs_deploy deploy_rows={len(combined_deploy_df)} scored_rows={len(scored_df)} graded_any={int(graded_any)}"
        )
    else:
        log(f"[self-check] scored_vs_deploy deploy_rows={len(combined_deploy_df)} scored_rows=0 graded_any=0")

    warn_bits: list[str] = []
    if tier_resolved == 0:
        warn_bits.append("NO_TIER_FILES")
    if tier_resolved == 0 and candidate_resolved > 0:
        warn_bits.append("CANDIDATE_ONLY")
    status = "OK"
    if not warn_bits:
        log("[self-check] status=OK")
    else:
        status = "WARN"
        log(f"[self-check] status=WARN flags={warn_bits}")

    return {
        "status": status,
        "flags": "|".join(warn_bits),
        "tier_files_count": int(len(native_deploy_csvs)),
        "tier_files_resolved": int(tier_resolved),
        "candidate_files_resolved": int(candidate_resolved),
        "combined_rows": int(len(combined_deploy_df)) if isinstance(combined_deploy_df, pd.DataFrame) else 0,
        "scored_rows": int(scored_rows),
        "graded_any": int(graded_any),
        "source_used": str(native_source_csv),
        "source_copied": str(source_csv),
    }


def append_self_check_row(base_outdir: Path, window_id: str, info: dict[str, object]) -> None:
    out_path = base_outdir / "_RUN_SELF_CHECK.csv"
    is_new = not out_path.exists()
    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "window_id",
                "status",
                "flags",
                "tier_files_count",
                "tier_files_resolved",
                "candidate_files_resolved",
                "combined_rows",
                "scored_rows",
                "graded_any",
                "source_used",
                "source_copied",
            ],
        )
        if is_new:
            writer.writeheader()
        row = {"window_id": window_id}
        row.update({k: info.get(k, "") for k in writer.fieldnames if k != "window_id"})
        writer.writerow(row)


def combine_deploy_files(deploy_csvs: list[Path], combined_csv: Path) -> pd.DataFrame:
    return combine_deploy_files_with_policy(deploy_csvs, combined_csv, tiers_only=False)


def combine_deploy_files_with_policy(
    deploy_csvs: list[Path],
    combined_csv: Path,
    *,
    tiers_only: bool = False,
) -> pd.DataFrame:
    def _union_missing_rows(base_df: pd.DataFrame, extra_df: pd.DataFrame, label: str) -> pd.DataFrame:
        if extra_df is None or extra_df.empty:
            return base_df

        extra_df = _ensure_source_tier_file(extra_df)

        if base_df is None or base_df.empty:
            log(f"[combine] rescued combined deploy rows from {label} count={len(extra_df)}")
            return extra_df.copy()

        extra_tier = safe_series_str(extra_df, "source_tier_file").str.upper().str.strip()
        base_tier = safe_series_str(base_df, "source_tier_file").str.upper().str.strip()
        include_tier_in_dedupe = not (
            extra_tier.eq("").all()
            or extra_tier.eq("UNSPECIFIED").all()
            or base_tier.eq("").all()
            or base_tier.eq("UNSPECIFIED").all()
        )

        dedupe_priority = ["league", "fixture_key", "market", "selection", "bookie_pick"]
        if include_tier_in_dedupe:
            dedupe_priority.append("source_tier_file")

        dedupe_cols = [
            c for c in dedupe_priority
            if c in base_df.columns and c in extra_df.columns
        ]
        if "fixture_key" not in dedupe_cols:
            fallback_priority = [
                "league",
                "match_date",
                "home_team_name",
                "away_team_name",
                "market",
                "selection",
                "bookie_pick",
            ]
            if include_tier_in_dedupe:
                fallback_priority.append("source_tier_file")
            dedupe_cols = [
                c for c in fallback_priority
                if c in base_df.columns and c in extra_df.columns
            ]

        if not dedupe_cols:
            unioned = pd.concat([base_df, extra_df], ignore_index=True, sort=False)
            log(f"[combine] appended {label} rows without dedupe keys count={len(extra_df)}")
            return unioned

        existing_keys = set(
            base_df.loc[:, dedupe_cols]
            .astype("string")
            .fillna("")
            .agg("||".join, axis=1)
            .tolist()
        )
        extra_keys = (
            extra_df.loc[:, dedupe_cols]
            .astype("string")
            .fillna("")
            .agg("||".join, axis=1)
        )
        missing_extra = extra_df.loc[~extra_keys.isin(existing_keys)].copy()

        if missing_extra.empty:
            return base_df

        log(f"[combine] unioned missing rows from {label} count={len(missing_extra)}")
        return pd.concat([base_df, missing_extra], ignore_index=True, sort=False)
    if not deploy_csvs:
        raise ValueError("No deploy CSVs provided for combination.")

    frames: list[pd.DataFrame] = []
    preset_frames: list[pd.DataFrame] = []
    seen_parent_dirs: set[Path] = set()
    gated_candidate_frames: list[pd.DataFrame] = []
    raw_candidate_frames: list[pd.DataFrame] = []
    raw_frames: list[pd.DataFrame] = []

    def _infer_tier_name_from_path(csv_path: Path) -> str:
        upper_name = csv_path.stem.upper()
        for candidate in ["ELITE", "STANDARD", "OBSERVE"]:
            if f"DEPLOY_TIER_{candidate}" in upper_name:
                return candidate
        return "UNKNOWN"

    def _ensure_source_tier_file(frame: pd.DataFrame, csv_path: Path | None = None) -> pd.DataFrame:
        out = frame.copy()

        inferred_from_path = _infer_tier_name_from_path(csv_path) if csv_path is not None else "UNKNOWN"
        csv_name_upper = csv_path.name.upper() if csv_path is not None else ""
        deploy_tier = safe_series_str(out, "deploy_tier").str.upper().str.strip()
        tier = safe_series_str(out, "tier").str.upper().str.strip()
        current = safe_series_str(out, "source_tier_file").str.upper().str.strip()

        resolved = current.mask(current.eq(""), deploy_tier)
        resolved = resolved.mask(resolved.eq(""), tier)
        resolved = resolved.mask(resolved.eq("UNKNOWN"), deploy_tier)
        resolved = resolved.mask(resolved.eq("UNKNOWN"), tier)
        resolved = resolved.mask(resolved.eq("UNSPECIFIED"), deploy_tier)
        resolved = resolved.mask(resolved.eq("UNSPECIFIED"), tier)

        if inferred_from_path != "UNKNOWN":
            resolved = resolved.mask(resolved.eq(""), inferred_from_path)
            resolved = resolved.mask(resolved.eq("UNKNOWN"), inferred_from_path)
            resolved = resolved.mask(resolved.eq("UNSPECIFIED"), inferred_from_path)

        if "DEPLOY_CANDIDATES_AFTER_GATES" in csv_name_upper or "DEPLOY_CANDIDATES_RAW" in csv_name_upper:
            resolved = resolved.mask(resolved.eq(""), "CANDIDATE")
            resolved = resolved.mask(resolved.eq("UNKNOWN"), "CANDIDATE")
            resolved = resolved.mask(resolved.eq("UNSPECIFIED"), "CANDIDATE")

        resolved = resolved.mask(resolved.eq(""), "UNSPECIFIED")
        resolved = resolved.mask(resolved.eq("UNKNOWN"), "UNSPECIFIED")
        out["source_tier_file"] = resolved
        return out

    for csv_path in deploy_csvs:
        frame = pd.read_csv(csv_path, low_memory=False)
        frame = _ensure_source_tier_file(frame, csv_path)
        frame["__source_deploy_file"] = csv_path.name
        raw_frames.append(frame)

        upper_name = csv_path.stem.upper()
        if "DEPLOY_CANDIDATES_AFTER_GATES" in upper_name:
            gated_candidate_frames.append(frame)
            seen_parent_dirs.add(csv_path.parent.resolve())
            continue
        if "DEPLOY_CANDIDATES_RAW" in upper_name:
            raw_candidate_frames.append(frame)
            seen_parent_dirs.add(csv_path.parent.resolve())
            continue
        if "DEPLOY_PRESET" in upper_name:
            preset_frames.append(frame)
            seen_parent_dirs.add(csv_path.parent.resolve())
            continue
        if frame["source_tier_file"].astype("string").fillna("").str.upper().eq("UNSPECIFIED").all():
            continue
        frames.append(frame)
        seen_parent_dirs.add(csv_path.parent.resolve())

    # Also look for sibling DEPLOY_PRESET files beside the resolved tier files.
    for parent_dir in sorted(seen_parent_dirs):
        for preset_path in sorted(parent_dir.glob("*DEPLOY_PRESET*.csv")):
            upper_name = preset_path.stem.upper()
            if "DEPLOY_CANDIDATES_AFTER_GATES" in upper_name or "DEPLOY_CANDIDATES_RAW" in upper_name:
                continue
            preset_df = pd.read_csv(preset_path, low_memory=False)
            preset_df = _ensure_source_tier_file(preset_df, preset_path)
            preset_df["__source_deploy_file"] = preset_path.name
            preset_frames.append(preset_df)

    if not frames and not preset_frames and not gated_candidate_frames and not raw_candidate_frames:
        if raw_frames and all(df.empty for df in raw_frames):
            # All tier files were present but empty; emit an empty combined CSV and continue.
            cols: list[str] = []
            for df in raw_frames:
                for c in df.columns:
                    if c not in cols:
                        cols.append(c)
            empty_combined = pd.DataFrame(columns=cols)
            empty_combined.to_csv(combined_csv, index=False)
            log(
                "[combine] all deploy tier files were empty; wrote empty combined deploy CSV "
                f"rows=0 cols={len(cols)}"
            )
            return empty_combined
        raise ValueError(
            f"No valid deploy CSVs remained after filtering for combination: {[str(p) for p in deploy_csvs]}"
        )

    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

    if combined.empty:
        fallback_frames = gated_candidate_frames if gated_candidate_frames else raw_candidate_frames
        if fallback_frames:
            combined = pd.concat(fallback_frames, ignore_index=True, sort=False)
            combined = _ensure_source_tier_file(combined)
            log(
                "[combine] rescued combined deploy rows from "
                f"{'DEPLOY_CANDIDATES_AFTER_GATES' if gated_candidate_frames else 'DEPLOY_CANDIDATES_RAW'} "
                f"count={len(combined)}"
            )

    if gated_candidate_frames:
        gated_combined = pd.concat(gated_candidate_frames, ignore_index=True, sort=False)
        if not tiers_only:
            combined = _union_missing_rows(combined, gated_combined, "DEPLOY_CANDIDATES_AFTER_GATES")

    if raw_candidate_frames:
        raw_combined = pd.concat(raw_candidate_frames, ignore_index=True, sort=False)
        if not tiers_only:
            combined = _union_missing_rows(combined, raw_combined, "DEPLOY_CANDIDATES_RAW")

    if not combined.empty:
        league_count = int(safe_series_str(combined, "league").str.strip().nunique()) if "league" in combined.columns else 0
        market_count = int(safe_series_str(combined, "market").str.lower().str.strip().nunique()) if "market" in combined.columns else 0
        tier_series = safe_series_str(combined, "source_tier_file").str.upper().str.strip() if "source_tier_file" in combined.columns else pd.Series(dtype="string")
        tier_count = int(tier_series.nunique()) if "source_tier_file" in combined.columns else 0
        tier_breakdown = tier_series.value_counts(dropna=False).to_dict() if "source_tier_file" in combined.columns else {}
        log(
            f"[combine] combined rows={len(combined)} leagues={league_count} markets={market_count} tiers={tier_count} tier_breakdown={tier_breakdown}"
        )

    # Critical rescue: TG rows can exist in DEPLOY_PRESET even when tier-file resolution misses them.
    # Pull TG rows from preset files and append only rows not already present in combined.
    if preset_frames:
        preset_combined = pd.concat(preset_frames, ignore_index=True, sort=False)
        preset_market = safe_series_str(preset_combined, "market").str.lower().str.strip()
        tg_rows = preset_combined.loc[preset_market.isin(["tg15", "tg25"])].copy()

        if not tg_rows.empty:
            if combined.empty:
                combined = tg_rows.copy()
            else:
                dedupe_cols = [
                    c for c in ["league", "fixture_key", "market", "bookie_pick"]
                    if c in combined.columns and c in tg_rows.columns
                ]
                if dedupe_cols:
                    existing_keys = set(
                        combined.loc[:, dedupe_cols]
                        .astype("string")
                        .fillna("")
                        .agg("||".join, axis=1)
                        .tolist()
                    )
                    tg_keys = (
                        tg_rows.loc[:, dedupe_cols]
                        .astype("string")
                        .fillna("")
                        .agg("||".join, axis=1)
                    )
                    tg_rows = tg_rows.loc[~tg_keys.isin(existing_keys)].copy()
                if not tg_rows.empty:
                    combined = pd.concat([combined, tg_rows], ignore_index=True, sort=False)

    if combined.empty:
        raise ValueError(
            f"No valid deploy rows remained after combining tier and preset files: {[str(p) for p in deploy_csvs]}"
        )

    combined.to_csv(combined_csv, index=False)
    return combined


# ============================================================================
# Helper functions for acca backtesting
# ============================================================================


def build_results_frame(matches_df: pd.DataFrame) -> pd.DataFrame:
    df = matches_df.copy()

    fixture_series = df.get("fixture_key", pd.Series("", index=df.index))
    if "home_goals" in df.columns:
        home_series = df.get("home_goals", pd.Series(np.nan, index=df.index))
    else:
        home_series = df.get("home_team_goal_count", pd.Series(np.nan, index=df.index))

    if "away_goals" in df.columns:
        away_series = df.get("away_goals", pd.Series(np.nan, index=df.index))
    else:
        away_series = df.get("away_team_goal_count", pd.Series(np.nan, index=df.index))

    df["fixture_key"] = fixture_series.astype("string").fillna("").str.strip()
    df["home_goals"] = pd.to_numeric(home_series, errors="coerce")
    df["away_goals"] = pd.to_numeric(away_series, errors="coerce")

    df = df.loc[
        df["fixture_key"].ne("")
        & df["home_goals"].notna()
        & df["away_goals"].notna()
    ].copy()

    return df[["fixture_key", "home_goals", "away_goals"]].drop_duplicates("fixture_key", keep="last")


# === Inserted: build_acca_results_frame ===
def build_acca_results_frame(scored_df: pd.DataFrame) -> pd.DataFrame:
    df = scored_df.copy()

    df["fixture_key"] = df.get("fixture_key", pd.Series("", index=df.index)).astype("string").fillna("").str.strip()

    if "home_goals" not in df.columns:
        df["home_goals"] = pd.to_numeric(df.get("home_team_goal_count", np.nan), errors="coerce")
    else:
        df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")

    if "away_goals" not in df.columns:
        df["away_goals"] = pd.to_numeric(df.get("away_team_goal_count", np.nan), errors="coerce")
    else:
        df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")

    keep = [
        c for c in [
            "fixture_key",
            "home_goals",
            "away_goals",
            "actual_ftr",
            "actual_over25",
            "actual_btts_yes",
            "actual_tg15_home",
            "actual_tg15_away",
            "actual_tg25_home",
            "actual_tg25_away",
        ]
        if c in df.columns
    ]

    df = df[keep].copy()
    if "actual_ftr" in df.columns:
        df["actual_ftr"] = df["actual_ftr"].astype("string").fillna("").str.strip().str.upper()
    if "actual_over25" in df.columns:
        df["actual_over25"] = pd.to_numeric(df["actual_over25"], errors="coerce")
    if "actual_btts_yes" in df.columns:
        df["actual_btts_yes"] = pd.to_numeric(df["actual_btts_yes"], errors="coerce")
    df = df.loc[df["fixture_key"].ne("")].drop_duplicates("fixture_key", keep="last")
    return df


def grade_leg(row: pd.Series) -> int:
    market = str(row.get("market", "")).strip().lower()
    sel = str(row.get("selection", "")).strip().upper()

    # Preferred path: use explicit actual outcome columns when present.
    actual_ftr = str(row.get("actual_ftr", "")).strip().upper()
    actual_over25 = pd.to_numeric(row.get("actual_over25", np.nan), errors="coerce")
    actual_btts_yes = pd.to_numeric(row.get("actual_btts_yes", np.nan), errors="coerce")

    if market == "ftr" and actual_ftr:
        return int(sel == actual_ftr)

    if market == "btts" and pd.notna(actual_btts_yes):
        return int(sel == "YES" and int(actual_btts_yes) == 1)

    if market == "btts_no" and pd.notna(actual_btts_yes):
        return int(sel == "NO" and int(actual_btts_yes) == 0)

    if market == "over25" and pd.notna(actual_over25):
        return int(sel == "OVER25" and int(actual_over25) == 1)

    if market == "under25" and pd.notna(actual_over25):
        return int(sel == "UNDER25" and int(actual_over25) == 0)

    # Fallback path: derive from goals if actual_* columns are unavailable.
    hg = pd.to_numeric(row.get("home_goals", None), errors="coerce")
    ag = pd.to_numeric(row.get("away_goals", None), errors="coerce")

    if pd.isna(hg) or pd.isna(ag):
        return 0

    hg = int(hg)
    ag = int(ag)

    if market == "ftr":
        if hg > ag and sel == "HOME":
            return 1
        if hg == ag and sel == "DRAW":
            return 1
        if hg < ag and sel == "AWAY":
            return 1
        return 0

    if market == "btts":
        both_scored = (hg > 0 and ag > 0)
        return int(both_scored and sel == "YES")

    if market == "btts_no":
        both_scored = (hg > 0 and ag > 0)
        return int((not both_scored) and sel == "NO")

    if market == "over25":
        return int((hg + ag) > 2 and sel == "OVER25")

    if market == "under25":
        return int((hg + ag) < 3 and sel == "UNDER25")

    return 0

def backtest_acca_legs(legs_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    df = legs_df.copy()
    results = build_acca_results_frame(results_df)

    if df.empty:
        out = df.copy()
        if "leg_correct" not in out.columns:
            out["leg_correct"] = pd.Series(dtype="int64")
        return out

    df["fixture_key"] = df.get("fixture_key", pd.Series("", index=df.index)).astype("string").fillna("").str.strip()
    results["fixture_key"] = results.get("fixture_key", pd.Series("", index=results.index)).astype("string").fillna("").str.strip()

    df = df.merge(results, on="fixture_key", how="left")

    if "actual_ftr" in df.columns:
        df["actual_ftr"] = df["actual_ftr"].astype("string").fillna("").str.strip().str.upper()
    if "actual_over25" in df.columns:
        df["actual_over25"] = pd.to_numeric(df["actual_over25"], errors="coerce")
    if "actual_btts_yes" in df.columns:
        df["actual_btts_yes"] = pd.to_numeric(df["actual_btts_yes"], errors="coerce")

    # Normalise fallback goals if only merged-style columns exist
    if "home_goals" not in df.columns and "home_team_goal_count" in df.columns:
        df["home_goals"] = pd.to_numeric(df["home_team_goal_count"], errors="coerce")
    else:
        df["home_goals"] = pd.to_numeric(df.get("home_goals", df.get("home_team_goal_count", np.nan)), errors="coerce")

    if "away_goals" not in df.columns and "away_team_goal_count" in df.columns:
        df["away_goals"] = pd.to_numeric(df["away_team_goal_count"], errors="coerce")
    else:
        df["away_goals"] = pd.to_numeric(df.get("away_goals", df.get("away_team_goal_count", np.nan)), errors="coerce")

    df["leg_correct"] = df.apply(grade_leg, axis=1).astype(int)
    return df


# === Inserted: ACCA aggregation and summary helpers ===

def aggregate_acca_slips(
    graded_legs_df: pd.DataFrame,
    slips_df: pd.DataFrame,
    flat_stake_per_slip: float = 10.0,
) -> pd.DataFrame:
    legs = graded_legs_df.copy()
    slips = slips_df.copy()

    if slips.empty:
        out = slips.copy()
        for col in [
            "stake",
            "legs_total",
            "legs_correct",
            "legs_wrong",
            "slip_result",
            "total_return",
            "pnl",
            "hit_flag",
            "premium_share",
            "best_leg_odds",
        ]:
            if col not in out.columns:
                out[col] = pd.Series(dtype="float64" if col not in {"slip_result"} else "string")
        return out

    if legs.empty:
        out = slips.copy()
        out["stake"] = float(flat_stake_per_slip)
        out["legs_total"] = 0
        out["legs_correct"] = 0
        out["legs_wrong"] = 0
        out["slip_result"] = "LOSS"
        out["total_return"] = 0.0
        out["pnl"] = -float(flat_stake_per_slip)
        out["hit_flag"] = 0
        out["premium_share"] = np.nan
        out["best_leg_odds"] = np.nan
        return out

    legs["slip_id"] = legs.get("slip_id", pd.Series("", index=legs.index)).astype("string").fillna("").str.strip()
    legs["leg_correct"] = pd.to_numeric(legs.get("leg_correct", pd.Series(0, index=legs.index)), errors="coerce").fillna(0).astype(int)
    legs["odds"] = pd.to_numeric(legs.get("odds", pd.Series(np.nan, index=legs.index)), errors="coerce")
    acca_priority = pd.to_numeric(
        legs.get("acca_builder_priority", pd.Series(np.nan, index=legs.index)),
        errors="coerce",
    )

    acca_bucket = legs.get("acca_builder_bucket", pd.Series("", index=legs.index)).astype("string").fillna("").str.strip()
    reporting_bucket = legs.get("standard_reporting_bucket", pd.Series("", index=legs.index)).astype("string").fillna("").str.strip()
    combo_bucket = legs.get("combo_overlay_bucket", pd.Series("", index=legs.index)).astype("string").fillna("").str.strip()

    ranked_from_priority = acca_priority.fillna(999).lt(999)
    ranked_from_bucket = acca_bucket.ne("") & acca_bucket.ne("NA")
    ranked_from_reporting = reporting_bucket.str.startswith("STANDARD_FTR_CS_PROMOTED") | reporting_bucket.str.startswith("STANDARD_FTR_COMBO")
    ranked_from_combo = combo_bucket.ne("") & combo_bucket.ne("NA")

    legs["is_ranked_premium"] = (
        ranked_from_priority
        | ranked_from_bucket
        | ranked_from_reporting
        | ranked_from_combo
    ).astype(int)

    slip_rollup = (
        legs.groupby("slip_id", dropna=False)
        .agg(
            legs_total=("leg_correct", "size"),
            legs_correct=("leg_correct", "sum"),
            avg_leg_odds=("odds", "mean"),
            best_leg_odds=("odds", "max"),
            premium_share=("is_ranked_premium", "mean"),
        )
        .reset_index()
    )
    slip_rollup["legs_wrong"] = slip_rollup["legs_total"] - slip_rollup["legs_correct"]
    slip_rollup["hit_flag"] = (slip_rollup["legs_wrong"] == 0).astype(int)
    slip_rollup["slip_result"] = np.where(slip_rollup["hit_flag"].eq(1), "WIN", "LOSS")

    slip_odds_rollup = (
        legs.groupby("slip_id", dropna=False)["odds"]
        .apply(lambda s: float(np.prod(s.dropna().to_numpy())) if not s.dropna().empty else np.nan)
        .reset_index(name="recomputed_slip_odds")
    )
    slip_rollup = slip_rollup.merge(slip_odds_rollup, on="slip_id", how="left")

    out = slips.merge(slip_rollup, on="slip_id", how="left")
    out["stake"] = float(flat_stake_per_slip)

    slip_odds_existing = pd.to_numeric(out.get("slip_odds", pd.Series(np.nan, index=out.index)), errors="coerce")
    recomputed_slip_odds = pd.to_numeric(out.get("recomputed_slip_odds", pd.Series(np.nan, index=out.index)), errors="coerce")
    effective_slip_odds = slip_odds_existing.fillna(recomputed_slip_odds)

    out["total_return"] = np.where(
        pd.to_numeric(out.get("hit_flag", pd.Series(0, index=out.index)), errors="coerce").fillna(0).eq(1)
        & effective_slip_odds.notna(),
        float(flat_stake_per_slip) * effective_slip_odds,
        0.0,
    )
    out["pnl"] = out["total_return"] - out["stake"]

    return out


def _max_consecutive_losses(flags: pd.Series) -> int:
    streak = 0
    best = 0
    for value in pd.to_numeric(flags, errors="coerce").fillna(0).astype(int).tolist():
        if value == 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return int(best)

def compute_max_losing_streak(hit_flags: pd.Series) -> int:
    streak = 0
    best = 0
    for value in pd.to_numeric(hit_flags, errors="coerce").fillna(0).astype(int).tolist():
        if value == 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return int(best)


def compute_duplicate_fixture_concentration(legs_df: pd.DataFrame) -> float:
    if legs_df is None or legs_df.empty or "fixture_key" not in legs_df.columns:
        return np.nan

    fixture_key = (
        legs_df["fixture_key"]
        .astype("string")
        .fillna("")
        .str.strip()
    )
    fixture_key = fixture_key[fixture_key.ne("")]
    if fixture_key.empty:
        return np.nan

    counts = fixture_key.value_counts(dropna=False)
    total = int(counts.sum())
    if total <= 0:
        return np.nan

    return float(counts.max() / total)

def summarize_acca_products(
    graded_slips_df: pd.DataFrame,
    window_id: str,
) -> pd.DataFrame:
    slips = graded_slips_df.copy()
    if slips.empty:
        return pd.DataFrame(columns=[
            "window_id",
            "template_name",
            "template_tag",
            "requested_k",
            "slips_built",
            "total_stake",
            "total_return",
            "roi",
            "hit_rate",
            "premium_share",
            "best_slip_odds",
            "best_winning_slip_odds",
            "max_losing_streak",
            "template_rank_for_week",
        ])

    slips["template_name"] = slips.get("template_name", pd.Series("", index=slips.index)).astype("string").fillna("").str.strip()
    slips["template_tag"] = slips.get("template_tag", pd.Series("", index=slips.index)).astype("string").fillna("").str.strip()
    slips["requested_k"] = pd.to_numeric(slips.get("requested_k", pd.Series(np.nan, index=slips.index)), errors="coerce")
    slips["stake"] = pd.to_numeric(slips.get("stake", pd.Series(0.0, index=slips.index)), errors="coerce").fillna(0.0)
    slips["total_return"] = pd.to_numeric(slips.get("total_return", pd.Series(0.0, index=slips.index)), errors="coerce").fillna(0.0)
    slips["pnl"] = pd.to_numeric(slips.get("pnl", pd.Series(0.0, index=slips.index)), errors="coerce").fillna(0.0)
    slips["hit_flag"] = pd.to_numeric(slips.get("hit_flag", pd.Series(0, index=slips.index)), errors="coerce").fillna(0).astype(int)
    slips["premium_share"] = pd.to_numeric(slips.get("premium_share", pd.Series(np.nan, index=slips.index)), errors="coerce")
    slips["slip_odds"] = pd.to_numeric(slips.get("slip_odds", pd.Series(np.nan, index=slips.index)), errors="coerce")

    rows: list[dict[str, object]] = []
    group_cols = ["template_name", "template_tag", "requested_k"]
    for keys, grp in slips.groupby(group_cols, dropna=False):
        template_name, template_tag, requested_k = keys
        grp = grp.sort_values(["hit_flag", "slip_odds"], ascending=[False, False]).copy()
        total_stake = float(grp["stake"].sum())
        total_return = float(grp["total_return"].sum())
        slips_built = int(len(grp))
        winners = grp.loc[grp["hit_flag"].eq(1)].copy()

        rows.append(
            {
                "window_id": window_id,
                "template_name": template_name,
                "template_tag": template_tag,
                "requested_k": int(requested_k) if pd.notna(requested_k) else np.nan,
                "slips_built": slips_built,
                "total_stake": total_stake,
                "total_return": total_return,
                "roi": ((total_return - total_stake) / total_stake) if total_stake > 0 else np.nan,
                "hit_rate": float(grp["hit_flag"].mean()) if slips_built > 0 else np.nan,
                "premium_share": float(grp["premium_share"].mean()) if slips_built > 0 else np.nan,
                "best_slip_odds": float(grp["slip_odds"].max()) if slips_built > 0 else np.nan,
                "best_winning_slip_odds": float(winners["slip_odds"].max()) if not winners.empty else np.nan,
                "max_losing_streak": _max_consecutive_losses(grp["hit_flag"]),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(["roi", "hit_rate", "premium_share", "best_slip_odds"], ascending=[False, False, False, False]).reset_index(drop=True)
    out["template_rank_for_week"] = np.arange(1, len(out) + 1)
    return out

def summarize_all_windows(master_window_df: pd.DataFrame) -> pd.DataFrame:
    if master_window_df is None or master_window_df.empty:
        return pd.DataFrame(columns=[
            "template_tag",
            "template_name",
            "requested_k",
            "windows",
            "total_stake",
            "total_return",
            "pnl",
            "roi",
            "avg_hit_rate",
            "avg_premium_share",
            "best_week_roi",
            "worst_week_roi",
        ])

    rows: list[dict[str, object]] = []
    for template_tag, g in master_window_df.groupby("template_tag", dropna=False):
        total_stake = float(pd.to_numeric(g.get("total_stake", pd.Series(0.0, index=g.index)), errors="coerce").fillna(0.0).sum())
        total_return = float(pd.to_numeric(g.get("total_return", pd.Series(0.0, index=g.index)), errors="coerce").fillna(0.0).sum())
        pnl = total_return - total_stake
        roi = pnl / total_stake if total_stake > 0 else np.nan

        requested_k_series = pd.to_numeric(g.get("requested_k", pd.Series(np.nan, index=g.index)), errors="coerce")
        requested_k_value = requested_k_series.dropna().iloc[0] if requested_k_series.notna().any() else np.nan

        hit_rate_series = pd.to_numeric(g.get("hit_rate", pd.Series(np.nan, index=g.index)), errors="coerce")
        premium_share_series = pd.to_numeric(g.get("premium_share", pd.Series(np.nan, index=g.index)), errors="coerce")
        roi_series = pd.to_numeric(g.get("roi", pd.Series(np.nan, index=g.index)), errors="coerce")

        rows.append({
            "template_tag": template_tag,
            "template_name": str(g["template_name"].iloc[0]) if "template_name" in g.columns and len(g) else "",
            "requested_k": int(requested_k_value) if pd.notna(requested_k_value) else np.nan,
            "windows": int(len(g)),
            "total_stake": total_stake,
            "total_return": total_return,
            "pnl": pnl,
            "roi": roi,
            "avg_hit_rate": float(hit_rate_series.mean()) if len(g) else np.nan,
            "avg_premium_share": float(premium_share_series.mean()) if len(g) else np.nan,
            "best_week_roi": float(roi_series.max()) if roi_series.notna().any() else np.nan,
            "worst_week_roi": float(roi_series.min()) if roi_series.notna().any() else np.nan,
        })

    return pd.DataFrame(rows).sort_values("roi", ascending=False).reset_index(drop=True)

def build_acca_product_audit(
    all_windows_df: pd.DataFrame,
    all_slips_df: pd.DataFrame,
    all_legs_df: pd.DataFrame,
) -> pd.DataFrame:
    if all_slips_df is None or all_slips_df.empty:
        return pd.DataFrame(columns=[
            "template_tag",
            "template_name",
            "requested_k",
            "windows",
            "slips",
            "wins",
            "win_rate",
            "total_stake",
            "total_return",
            "pnl",
            "roi",
            "max_losing_streak",
            "avg_winning_slip_odds",
            "median_winning_slip_odds",
            "duplicate_fixture_concentration_score",
        ])

    slips = all_slips_df.copy()
    windows = all_windows_df.copy() if all_windows_df is not None else pd.DataFrame()
    legs = all_legs_df.copy() if all_legs_df is not None else pd.DataFrame()

    slips["template_tag"] = slips.get("template_tag", pd.Series("", index=slips.index)).astype("string").fillna("").str.strip()
    slips["template_name"] = slips.get("template_name", pd.Series("", index=slips.index)).astype("string").fillna("").str.strip()
    slips["requested_k"] = pd.to_numeric(slips.get("requested_k", pd.Series(np.nan, index=slips.index)), errors="coerce")
    slips["window_id"] = slips.get("window_id", pd.Series("", index=slips.index)).astype("string").fillna("").str.strip()
    slips["stake"] = pd.to_numeric(slips.get("stake", pd.Series(0.0, index=slips.index)), errors="coerce").fillna(0.0)
    slips["total_return"] = pd.to_numeric(slips.get("total_return", pd.Series(0.0, index=slips.index)), errors="coerce").fillna(0.0)
    slips["hit_flag"] = pd.to_numeric(slips.get("hit_flag", pd.Series(0, index=slips.index)), errors="coerce").fillna(0).astype(int)
    slips["slip_odds"] = pd.to_numeric(slips.get("slip_odds", pd.Series(np.nan, index=slips.index)), errors="coerce")

    if not windows.empty:
        windows["template_tag"] = windows.get("template_tag", pd.Series("", index=windows.index)).astype("string").fillna("").str.strip()
        windows["window_id"] = windows.get("window_id", pd.Series("", index=windows.index)).astype("string").fillna("").str.strip()

    if not legs.empty:
        legs["template_tag"] = legs.get("template_tag", pd.Series("", index=legs.index)).astype("string").fillna("").str.strip()
        legs["window_id"] = legs.get("window_id", pd.Series("", index=legs.index)).astype("string").fillna("").str.strip()

    rows = []
    for template_tag, g in slips.groupby("template_tag", dropna=False):
        if str(template_tag).strip() == "":
            continue

        wins_df = g.loc[g["hit_flag"].eq(1)].copy()
        product_legs = legs.loc[legs["template_tag"].eq(template_tag)].copy() if not legs.empty else pd.DataFrame()

        if not windows.empty and "template_tag" in windows.columns:
            windows_count = int(windows.loc[windows["template_tag"].eq(template_tag), "window_id"].nunique())
        else:
            windows_count = int(g["window_id"].nunique()) if "window_id" in g.columns else np.nan

        hit_sequence = g.sort_values(["window_id", "slip_id"], ascending=[True, True])["hit_flag"]
        total_stake = float(g["stake"].sum())
        total_return = float(g["total_return"].sum())
        pnl = total_return - total_stake
        roi = (pnl / total_stake) if total_stake > 0 else np.nan

        requested_k_series = g["requested_k"].dropna()
        requested_k_value = int(requested_k_series.iloc[0]) if not requested_k_series.empty else np.nan

        rows.append({
            "template_tag": template_tag,
            "template_name": str(g["template_name"].iloc[0]) if "template_name" in g.columns and len(g) else "",
            "requested_k": requested_k_value,
            "windows": windows_count,
            "slips": int(len(g)),
            "wins": int(g["hit_flag"].sum()),
            "win_rate": float(g["hit_flag"].mean()) if len(g) else np.nan,
            "total_stake": total_stake,
            "total_return": total_return,
            "pnl": pnl,
            "roi": roi,
            "max_losing_streak": compute_max_losing_streak(hit_sequence),
            "avg_winning_slip_odds": float(wins_df["slip_odds"].mean()) if not wins_df.empty else np.nan,
            "median_winning_slip_odds": float(wins_df["slip_odds"].median()) if not wins_df.empty else np.nan,
            "duplicate_fixture_concentration_score": compute_duplicate_fixture_concentration(product_legs),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values(["roi", "win_rate", "wins"], ascending=[False, False, False]).reset_index(drop=True)

def write_master_acca_backtest_outputs(
    master_window_df: pd.DataFrame,
    out_dir: Path,
) -> tuple[Path, Path]:
    ensure_dir(out_dir)
    all_windows_path = out_dir / "acca_backtest_all_windows.csv"
    product_summary_path = out_dir / "acca_backtest_product_summary.csv"

    master_window_df.to_csv(all_windows_path, index=False)
    summarize_all_windows(master_window_df).to_csv(product_summary_path, index=False)
    return all_windows_path, product_summary_path


def _test_grade_leg() -> None:
    tests = [
        ({"market": "ftr", "selection": "HOME", "home_goals": 2, "away_goals": 1}, 1),
        ({"market": "ftr", "selection": "DRAW", "home_goals": 1, "away_goals": 1}, 1),
        ({"market": "ftr", "selection": "AWAY", "home_goals": 0, "away_goals": 2}, 1),
        ({"market": "btts", "selection": "YES", "home_goals": 1, "away_goals": 1}, 1),
        ({"market": "btts", "selection": "YES", "home_goals": 2, "away_goals": 0}, 0),
        ({"market": "btts_no", "selection": "NO", "home_goals": 2, "away_goals": 0}, 1),
        ({"market": "btts_no", "selection": "NO", "home_goals": 0, "away_goals": 0}, 1),
        ({"market": "btts_no", "selection": "NO", "home_goals": 2, "away_goals": 1}, 0),
        ({"market": "over25", "selection": "OVER25", "home_goals": 2, "away_goals": 1}, 1),
        ({"market": "over25", "selection": "OVER25", "home_goals": 1, "away_goals": 1}, 0),
        ({"market": "under25", "selection": "UNDER25", "home_goals": 1, "away_goals": 1}, 1),
        ({"market": "under25", "selection": "UNDER25", "home_goals": 2, "away_goals": 1}, 0),
    ]

    for payload, expected in tests:
        got = grade_leg(pd.Series(payload))
        assert got == expected, f"grade_leg failed: payload={payload}, expected={expected}, got={got}"

def write_acca_backtest_outputs(
    graded_legs_df: pd.DataFrame,
    graded_slips_df: pd.DataFrame,
    product_summary_df: pd.DataFrame,
    reports_dir: Path,
    tag: str,
) -> None:
    ensure_dir(reports_dir)
    graded_legs_df.to_csv(reports_dir / f"ACCA_LEGS_GRADED__{tag}.csv", index=False)
    graded_slips_df.to_csv(reports_dir / f"ACCA_SLIPS_GRADED__{tag}.csv", index=False)
    product_summary_df.to_csv(reports_dir / f"ACCA_PRODUCTS_SUMMARY__{tag}.csv", index=False)

    # lowercase compatibility outputs
    graded_legs_df.to_csv(reports_dir / f"acca_graded_legs_{tag}.csv", index=False)
    graded_slips_df.to_csv(reports_dir / f"acca_backtest_slips_{tag}.csv", index=False)
    product_summary_df.to_csv(reports_dir / f"acca_backtest_summary_{tag}.csv", index=False)


def score_deploy_df(df: pd.DataFrame, scored_csv: Path) -> pd.DataFrame:
    df = df.copy()

    market = safe_series_str(df, "market").str.lower().str.strip()
    selection = safe_series_str(df, "selection").str.upper().str.strip()

    if "actual_ftr" in df.columns and "ftr_hit" not in df.columns:
        actual_ftr = safe_series_str(df, "actual_ftr").str.upper().str.strip()
        df["ftr_hit"] = np.where(
            market.eq("ftr") & actual_ftr.ne("") & selection.ne(""),
            (selection == actual_ftr).astype(float),
            np.nan,
        )

    if "actual_over25" in df.columns and "ou25_hit" not in df.columns:
        actual_ou = pd.to_numeric(df["actual_over25"], errors="coerce")
        df["ou25_hit"] = np.where(
            market.eq("ou25") & selection.isin(["OVER25", "UNDER25"]) & actual_ou.notna(),
            np.where(selection.eq("OVER25"), (actual_ou == 1), (actual_ou == 0)).astype(float),
            np.nan,
        )

    if "actual_btts_yes" in df.columns and "btts_yes_hit" not in df.columns:
        actual_btts = pd.to_numeric(df["actual_btts_yes"], errors="coerce")
        df["btts_yes_hit"] = np.where(
            market.eq("btts") & selection.eq("YES") & actual_btts.notna(),
            (actual_btts == 1).astype(float),
            np.nan,
        )
    if "actual_btts_yes" in df.columns and "btts_no_hit" not in df.columns:
        actual_btts = pd.to_numeric(df["actual_btts_yes"], errors="coerce")
        df["btts_no_hit"] = np.where(
            market.eq("btts") & selection.eq("NO") & actual_btts.notna(),
            (actual_btts == 0).astype(float),
            np.nan,
        )

    if "actual_tg15" not in df.columns:
        df["actual_tg15"] = np.nan
    if "actual_tg25" not in df.columns:
        df["actual_tg25"] = np.nan

    home_goals = pd.to_numeric(df.get("home_team_goal_count", pd.Series(np.nan, index=df.index)), errors="coerce")
    away_goals = pd.to_numeric(df.get("away_team_goal_count", pd.Series(np.nan, index=df.index)), errors="coerce")
    resolved_goals = home_goals.notna() & away_goals.notna()

    actual_tg15 = pd.Series(np.nan, index=df.index, dtype="float64")
    actual_tg25 = pd.Series(np.nan, index=df.index, dtype="float64")

    is_home_tg15 = market.eq("tg15") & selection.eq("HOME_TG15")
    is_away_tg15 = market.eq("tg15") & selection.eq("AWAY_TG15")
    is_home_tg25 = market.eq("tg25") & selection.eq("HOME_TG25")
    is_away_tg25 = market.eq("tg25") & selection.eq("AWAY_TG25")

    # Preferred path: derive TG outcomes directly from raw goal columns when they are present.
    actual_tg15 = actual_tg15.mask(is_home_tg15 & resolved_goals, (home_goals >= 2).astype(float))
    actual_tg15 = actual_tg15.mask(is_away_tg15 & resolved_goals, (away_goals >= 2).astype(float))
    actual_tg25 = actual_tg25.mask(is_home_tg25 & resolved_goals, (home_goals >= 3).astype(float))
    actual_tg25 = actual_tg25.mask(is_away_tg25 & resolved_goals, (away_goals >= 3).astype(float))

    # Robust fallback: use merged actual-side columns when raw goals are not carried on the deploy/scored frame.
    actual_tg15_home = pd.to_numeric(df.get("actual_tg15_home", pd.Series(np.nan, index=df.index)), errors="coerce")
    actual_tg15_away = pd.to_numeric(df.get("actual_tg15_away", pd.Series(np.nan, index=df.index)), errors="coerce")
    actual_tg25_home = pd.to_numeric(df.get("actual_tg25_home", pd.Series(np.nan, index=df.index)), errors="coerce")
    actual_tg25_away = pd.to_numeric(df.get("actual_tg25_away", pd.Series(np.nan, index=df.index)), errors="coerce")

    actual_tg15 = actual_tg15.mask(is_home_tg15 & actual_tg15.isna(), actual_tg15_home)
    actual_tg15 = actual_tg15.mask(is_away_tg15 & actual_tg15.isna(), actual_tg15_away)
    actual_tg25 = actual_tg25.mask(is_home_tg25 & actual_tg25.isna(), actual_tg25_home)
    actual_tg25 = actual_tg25.mask(is_away_tg25 & actual_tg25.isna(), actual_tg25_away)

    existing_tg15 = pd.to_numeric(df.get("actual_tg15", pd.Series(np.nan, index=df.index)), errors="coerce")
    existing_tg25 = pd.to_numeric(df.get("actual_tg25", pd.Series(np.nan, index=df.index)), errors="coerce")
    df["actual_tg15"] = existing_tg15.fillna(actual_tg15)
    df["actual_tg25"] = existing_tg25.fillna(actual_tg25)

    if "tg15_hit" not in df.columns:
        actual_tg15_num = pd.to_numeric(df["actual_tg15"], errors="coerce")
        df["tg15_hit"] = np.where(
            market.eq("tg15") & selection.isin(["HOME_TG15", "AWAY_TG15"]) & actual_tg15_num.notna(),
            (actual_tg15_num == 1).astype(float),
            np.nan,
        )

    if "tg25_hit" not in df.columns:
        actual_tg25_num = pd.to_numeric(df["actual_tg25"], errors="coerce")
        df["tg25_hit"] = np.where(
            market.eq("tg25") & selection.isin(["HOME_TG25", "AWAY_TG25"]) & actual_tg25_num.notna(),
            (actual_tg25_num == 1).astype(float),
            np.nan,
        )
    actual_ftr_combo = safe_series_str(df, "actual_ftr").str.upper().str.strip()
    actual_tg15_home_combo = pd.to_numeric(df.get("actual_tg15_home", pd.Series(np.nan, index=df.index)), errors="coerce")
    actual_tg15_away_combo = pd.to_numeric(df.get("actual_tg15_away", pd.Series(np.nan, index=df.index)), errors="coerce")

    combo_home_from_raw = pd.Series(np.nan, index=df.index, dtype="float64")
    combo_away_from_raw = pd.Series(np.nan, index=df.index, dtype="float64")

    combo_home_from_raw = combo_home_from_raw.mask(
        resolved_goals,
        ((home_goals > away_goals) & (home_goals >= 2)).astype(float),
    )
    combo_away_from_raw = combo_away_from_raw.mask(
        resolved_goals,
        ((away_goals > home_goals) & (away_goals >= 2)).astype(float),
    )

    combo_home_from_actuals = pd.Series(np.nan, index=df.index, dtype="float64")
    combo_away_from_actuals = pd.Series(np.nan, index=df.index, dtype="float64")

    combo_home_resolved = actual_ftr_combo.ne("") & actual_tg15_home_combo.notna()
    combo_away_resolved = actual_ftr_combo.ne("") & actual_tg15_away_combo.notna()

    combo_home_from_actuals = combo_home_from_actuals.mask(
        combo_home_resolved,
        ((actual_ftr_combo.eq("HOME")) & (actual_tg15_home_combo.eq(1))).astype(float),
    )
    combo_away_from_actuals = combo_away_from_actuals.mask(
        combo_away_resolved,
        ((actual_ftr_combo.eq("AWAY")) & (actual_tg15_away_combo.eq(1))).astype(float),
    )

    if "actual_hw_and_hge2" not in df.columns:
        df["actual_hw_and_hge2"] = combo_home_from_raw.fillna(combo_home_from_actuals)
    else:
        existing_hw_combo = pd.to_numeric(df.get("actual_hw_and_hge2", pd.Series(np.nan, index=df.index)), errors="coerce")
        df["actual_hw_and_hge2"] = existing_hw_combo.fillna(combo_home_from_raw).fillna(combo_home_from_actuals)

    if "actual_aw_and_age2" not in df.columns:
        df["actual_aw_and_age2"] = combo_away_from_raw.fillna(combo_away_from_actuals)
    else:
        existing_aw_combo = pd.to_numeric(df.get("actual_aw_and_age2", pd.Series(np.nan, index=df.index)), errors="coerce")
        df["actual_aw_and_age2"] = existing_aw_combo.fillna(combo_away_from_raw).fillna(combo_away_from_actuals)

    if "hw_and_hge2_hit" not in df.columns:
        actual_hw_and_hge2_num = pd.to_numeric(df["actual_hw_and_hge2"], errors="coerce")
        df["hw_and_hge2_hit"] = np.where(
            market.eq("ftr") & selection.eq("HOME") & actual_hw_and_hge2_num.notna(),
            (actual_hw_and_hge2_num == 1).astype(float),
            np.nan,
        )

    if "aw_and_age2_hit" not in df.columns:
        actual_aw_and_age2_num = pd.to_numeric(df["actual_aw_and_age2"], errors="coerce")
        df["aw_and_age2_hit"] = np.where(
            market.eq("ftr") & selection.eq("AWAY") & actual_aw_and_age2_num.notna(),
            (actual_aw_and_age2_num == 1).astype(float),
            np.nan,
        )
    if "bookie_od" not in df.columns:
        df["bookie_od"] = pd.to_numeric(df.get("bookie_od", np.nan), errors="coerce")
        odds = pd.Series(np.nan, index=df.index)
        odds = np.where(market.eq("ftr"), np.nan, odds)
        if "odds_ft_over25" in df.columns:
            odds = np.where(market.eq("ou25") & selection.eq("OVER25"), safe_series_num(df, "odds_ft_over25"), odds)
        if "odds_btts_yes" in df.columns:
            odds = np.where(market.eq("btts") & selection.eq("YES"), safe_series_num(df, "odds_btts_yes"), odds)
        df["bookie_od"] = pd.to_numeric(odds, errors="coerce")

    df.to_csv(scored_csv, index=False)
    return df


# ============================================================================
# Reporting
# ============================================================================
def calc_level_stake_profit(sub: pd.DataFrame, hit_col: str) -> tuple[float, int]:
    market_name = ""
    if isinstance(sub, pd.DataFrame) and (not sub.empty) and ("market" in sub.columns):
        try:
            market_name = str(
                sub["market"].astype("string").fillna("").str.lower().str.strip().iloc[0]
            )
        except Exception:
            market_name = ""

    if market_name in {"tg15", "tg25"}:
        return (np.nan, 0)

    if hit_col not in sub.columns:
        return (0.0, 0)
    graded_mask = sub[hit_col].notna()
    if not graded_mask.any():
        return (0.0, 0)
    graded = sub.loc[graded_mask].copy()
    odds = pd.to_numeric(graded.get("bookie_od", np.nan), errors="coerce")
    hits = pd.to_numeric(graded[hit_col], errors="coerce").fillna(0)
    profit = np.where(hits == 1, odds - 1.0, -1.0)
    profit = pd.Series(profit).replace([np.inf, -np.inf], np.nan).fillna(-1.0)
    return float(profit.sum()), int(len(graded))


def summarize_slice(df: pd.DataFrame, hit_col: str) -> dict[str, float | int]:
    market_name = ""
    if isinstance(df, pd.DataFrame) and (not df.empty) and ("market" in df.columns):
        try:
            market_name = str(
                df["market"].astype("string").fillna("").str.lower().str.strip().iloc[0]
            )
        except Exception:
            market_name = ""

    is_tg_market = market_name in {"tg15", "tg25"}

    graded_mask = df[hit_col].notna() if hit_col in df.columns else pd.Series(False, index=df.index)
    graded = int(graded_mask.sum())
    wins = float(pd.to_numeric(df.loc[graded_mask, hit_col], errors="coerce").fillna(0).sum()) if graded else 0.0
    losses = int(graded - wins) if graded else 0
    profit, stake_n = calc_level_stake_profit(df, hit_col)
    roi = (profit / stake_n) if stake_n > 0 else np.nan
    avg_od = float(pd.to_numeric(df.get("bookie_od", np.nan), errors="coerce").mean()) if len(df) else np.nan
    avg_mp = float(pd.to_numeric(df.get("model_p_for_bookie", np.nan), errors="coerce").mean()) if len(df) else np.nan
    return {
        "rows": int(len(df)),
        "graded": graded,
        "wins": wins,
        "losses": losses,
        "hit_rate": float(wins / graded) if graded else np.nan,
        "roi_level_stake": np.nan if is_tg_market else roi,
        "level_stake_profit": np.nan if is_tg_market else profit,
        "avg_bookie_od": np.nan if is_tg_market else avg_od,
        "avg_model_p_for_bookie": avg_mp,
        "evaluation_mode": "ACCURACY_ONLY" if is_tg_market else "ROI_AND_ACCURACY",
    }


def market_base_filter(df: pd.DataFrame, market_name: str) -> pd.DataFrame:
    cfg = SUPPORTED_MARKETS[market_name]
    market = safe_series_str(df, "market").str.lower().str.strip()
    selection = safe_series_str(df, cfg["selection_col"]).str.upper().str.strip()
    out = df.loc[market.eq(market_name)].copy()
    filt = cfg["default_selection_filter"]
    if filt is not None:
        out = out.loc[selection.loc[out.index].eq(filt)].copy()
    return out


def make_market_tier_summary(df: pd.DataFrame, window_id: str) -> pd.DataFrame:
    rows = []
    for market_name, cfg in SUPPORTED_MARKETS.items():
        sub = market_base_filter(df, market_name)
        if sub.empty:
            continue
        tier_col = "source_tier_file" if "source_tier_file" in sub.columns else "tier"
        if tier_col not in sub.columns:
            tmp = sub.copy()
            tmp[tier_col] = "ALL"
            sub = tmp
        for tier_value, grp in sub.groupby(tier_col, dropna=False):
            row = {
                "window_id": window_id,
                "market": market_name,
                "source_tier_file": tier_value,
                **summarize_slice(grp, cfg["hit_col"]),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def make_market_bucket_summary(df: pd.DataFrame, window_id: str) -> pd.DataFrame:
    rows = []
    bucket_col = "standard_reporting_bucket"
    if bucket_col not in df.columns:
        return pd.DataFrame(columns=["window_id", "market", bucket_col, "rows", "graded", "wins", "losses", "hit_rate"])
    for market_name, cfg in SUPPORTED_MARKETS.items():
        sub = market_base_filter(df, market_name)
        if sub.empty:
            continue
        for bucket_value, grp in sub.groupby(bucket_col, dropna=False):
            rows.append({
                "window_id": window_id,
                "market": market_name,
                bucket_col: bucket_value,
                **summarize_slice(grp, cfg["hit_col"]),
            })
    return pd.DataFrame(rows)


def make_market_league_summary(df: pd.DataFrame, window_id: str) -> pd.DataFrame:
    rows = []
    if "league" not in df.columns:
        return pd.DataFrame(columns=["window_id", "market", "league", "rows", "graded", "wins", "losses", "hit_rate"])
    for market_name, cfg in SUPPORTED_MARKETS.items():
        sub = market_base_filter(df, market_name)
        if sub.empty:
            continue
        for league_value, grp in sub.groupby("league", dropna=False):
            rows.append({
                "window_id": window_id,
                "market": market_name,
                "league": league_value,
                **summarize_slice(grp, cfg["hit_col"]),
            })
    return pd.DataFrame(rows)


def make_market_signal_summary(df: pd.DataFrame, window_id: str) -> pd.DataFrame:
    rows = []
    signal_map = {
        "ftr": ["context_reason_codes", "model_top_pick"],
        "ou25": ["signal_over25", "ou25_runtime_lane", "ou25_policy_branch", "ou25_policy_state"],
        "btts": ["signal_btts", "signal_btts_runtime", "btts_alignment"],
    }
    for market_name, cfg in SUPPORTED_MARKETS.items():
        sub = market_base_filter(df, market_name)
        if sub.empty:
            continue
        for signal_col in signal_map.get(market_name, []):
            if signal_col not in sub.columns:
                continue
            for sig_value, grp in sub.groupby(signal_col, dropna=False):
                rows.append({
                    "window_id": window_id,
                    "market": market_name,
                    "signal_col": signal_col,
                    "signal_group": sig_value,
                    **summarize_slice(grp, cfg["hit_col"]),
                })
    return pd.DataFrame(rows)


def write_market_detail_and_splits(df: pd.DataFrame, reports_dir: Path) -> None:
    for market_name, cfg in SUPPORTED_MARKETS.items():
        sub = market_base_filter(df, market_name)
        if sub.empty:
            continue
        hit_col = cfg["hit_col"]
        prefix = {
            "ftr": "FTR",
            "ou25": "OU25_OVER",
            "btts": "BTTS_YES",
            "tg15": "TG15",
            "tg25": "TG25",
        }[market_name]
        sub.to_csv(reports_dir / f"{prefix}__DETAIL.csv", index=False)
        if hit_col in sub.columns:
            sub.loc[sub[hit_col] == 1].to_csv(reports_dir / f"{prefix}__WINS.csv", index=False)
            sub.loc[sub[hit_col] == 0].to_csv(reports_dir / f"{prefix}__LOSSES.csv", index=False)


def write_per_market_tier_files(df: pd.DataFrame, reports_dir: Path) -> None:
    tier_col_candidates = ["source_tier_file", "tier"]
    for market_name, cfg in SUPPORTED_MARKETS.items():
        sub = market_base_filter(df, market_name)
        if sub.empty:
            continue
        prefix = {
            "ftr": "FTR",
            "ou25": "OU25_OVER",
            "btts": "BTTS_YES",
            "tg15": "TG15",
            "tg25": "TG25",
        }[market_name]
        for col in tier_col_candidates:
            if col in sub.columns:
                for tier_value, grp in sub.groupby(col, dropna=False):
                    tier_slug = str(tier_value).replace(" ", "_")
                    grp.to_csv(reports_dir / f"{prefix}__{tier_slug}.csv", index=False)
                break

# New helper: write_btts_signal_splits
def write_btts_signal_splits(df: pd.DataFrame, reports_dir: Path) -> None:
    if df is None or df.empty:
        return

    market = safe_series_str(df, "market").str.lower().str.strip()
    pick = safe_series_str(df, "bookie_pick").str.upper().str.strip()
    sig = safe_series_str(df, "signal_btts_runtime").str.upper().str.strip()
    league = safe_series_str(df, "league").str.strip()

    bt = df.loc[market.eq("btts") & pick.eq("YES")].copy()
    if bt.empty:
        return

    strong_yes = bt.loc[sig.loc[bt.index].eq("STRONG_YES")].copy()
    very_strong_yes = bt.loc[sig.loc[bt.index].eq("VERY_STRONG_YES")].copy()
    very_strong_yes_brazil = bt.loc[
        sig.loc[bt.index].eq("VERY_STRONG_YES")
        & league.loc[bt.index].eq("Brazil Serie A")
    ].copy()
    very_strong_yes_non_brazil = bt.loc[
        sig.loc[bt.index].eq("VERY_STRONG_YES")
        & (~league.loc[bt.index].eq("Brazil Serie A"))
    ].copy()

    live_core = strong_yes.copy()
    live_aggressive = very_strong_yes_non_brazil.copy()
    live_combined = pd.concat(
        [live_core, live_aggressive],
        ignore_index=True,
        sort=False,
    )

    bt.to_csv(reports_dir / "BTTS_YES__ALL.csv", index=False)
    strong_yes.to_csv(reports_dir / "BTTS_YES__STRONG_YES.csv", index=False)
    very_strong_yes.to_csv(reports_dir / "BTTS_YES__VERY_STRONG_YES.csv", index=False)
    very_strong_yes_brazil.to_csv(
        reports_dir / "BTTS_YES__VERY_STRONG_YES__BRAZIL_SERIE_A.csv",
        index=False,
    )
    very_strong_yes_non_brazil.to_csv(
        reports_dir / "BTTS_YES__VERY_STRONG_YES__NON_BRAZIL.csv",
        index=False,
    )

    live_core.to_csv(reports_dir / "BTTS_YES__LIVE_CORE.csv", index=False)
    live_aggressive.to_csv(reports_dir / "BTTS_YES__LIVE_AGGRESSIVE.csv", index=False)
    live_combined.to_csv(reports_dir / "BTTS_YES__LIVE_COMBINED.csv", index=False)


def summarize_window(scored_df: pd.DataFrame, window_id: str, reports_dir: Path) -> pd.DataFrame:
    ensure_dir(reports_dir)

    market_tier = make_market_tier_summary(scored_df, window_id)
    market_bucket = make_market_bucket_summary(scored_df, window_id)
    market_league = make_market_league_summary(scored_df, window_id)
    market_signal = make_market_signal_summary(scored_df, window_id)

    market_tier.to_csv(reports_dir / "SUMMARY__MARKET_TIER.csv", index=False)
    market_bucket.to_csv(reports_dir / "SUMMARY__BUCKET.csv", index=False)
    market_league.to_csv(reports_dir / "SUMMARY__LEAGUE.csv", index=False)
    market_signal.to_csv(reports_dir / "SUMMARY__SIGNAL.csv", index=False)
    scored_df.to_csv(reports_dir / "DETAIL__SCORED.csv", index=False)

    write_market_detail_and_splits(scored_df, reports_dir)
    write_per_market_tier_files(scored_df, reports_dir)
    write_btts_signal_splits(scored_df, reports_dir)

    # Optional duplicated names matching your preferred pack.
    export_named_summaries(reports_dir, market_tier, market_bucket, market_league, market_signal)
    return market_tier


def export_named_summaries(reports_dir: Path, market_tier: pd.DataFrame, market_bucket: pd.DataFrame,
                           market_league: pd.DataFrame, market_signal: pd.DataFrame) -> None:
    def _safe_market_filter(df: pd.DataFrame, market_name: str) -> pd.DataFrame:
        if df is None or df.empty:
            return df if df is not None else pd.DataFrame()
        if "market" not in df.columns:
            return df.iloc[0:0].copy()
        return df.loc[df["market"] == market_name].copy()

    name_map = {
        "ftr": "FTR",
        "ou25": "OU25_OVER",
        "btts": "BTTS_YES",
        "tg15": "TG15",
        "tg25": "TG25",
    }
    for market_name, prefix in name_map.items():
        _safe_market_filter(market_tier, market_name).to_csv(
            reports_dir / f"{prefix}__TIER_SUMMARY.csv", index=False
        )
        _safe_market_filter(market_bucket, market_name).to_csv(
            reports_dir / f"{prefix}__BUCKET_SUMMARY.csv", index=False
        )
        _safe_market_filter(market_league, market_name).to_csv(
            reports_dir / f"{prefix}__LEAGUE_SUMMARY.csv", index=False
        )
        _safe_market_filter(market_signal, market_name).to_csv(
            reports_dir / f"{prefix}__SIGNAL_SUMMARY.csv", index=False
        )


# ============================================================================
# Master aggregation
# ============================================================================
def build_master_rollups(base_outdir: Path, windows: list[WindowSpec]) -> None:
    master_dir = base_outdir / "_MASTER"
    ensure_dir(master_dir)

    scorecard_path = master_dir / "_ALL_WINDOWS__SCORECARD.csv"
    if not scorecard_path.exists():
        log("[master] No scorecard found yet. Skipping rollups.")
        return

    scorecard = pd.read_csv(scorecard_path)
    if scorecard.empty:
        return

    if "graded" in scorecard.columns:
        graded_vals = pd.to_numeric(scorecard["graded"], errors="coerce").fillna(0)
    else:
        graded_vals = pd.Series([0] * len(scorecard))
    graded_scorecard = scorecard.loc[graded_vals > 0].copy()
    graded_scorecard.to_csv(master_dir / "_ALL_WINDOWS__SCORECARD_GRADED_ONLY.csv", index=False)

    if graded_scorecard.empty:
        log("[master] Scorecard exists but no graded rows were found. Skipping graded rollups.")
        return

    market_rollup = (
        graded_scorecard.groupby(["market", "source_tier_file"], dropna=False)
        .agg(
            windows=("window_id", "nunique"),
            total_rows=("rows", "sum"),
            total_graded=("graded", "sum"),
            total_wins=("wins", "sum"),
            mean_window_hit_rate=("hit_rate", "mean"),
            median_window_hit_rate=("hit_rate", "median"),
            min_window_hit_rate=("hit_rate", "min"),
            max_window_hit_rate=("hit_rate", "max"),
            stdev_window_hit_rate=("hit_rate", "std"),
            total_level_stake_profit=("level_stake_profit", "sum"),
        )
        .reset_index()
    )
    market_rollup["overall_hit_rate"] = market_rollup["total_wins"] / market_rollup["total_graded"].replace(0, np.nan)
    market_rollup["evaluation_mode"] = np.where(
        market_rollup["market"].astype("string").fillna("").str.lower().isin(["tg15", "tg25"]),
        "ACCURACY_ONLY",
        "ROI_AND_ACCURACY",
    )
    market_rollup.loc[
        market_rollup["market"].astype("string").fillna("").str.lower().isin(["tg15", "tg25"]),
        "total_level_stake_profit",
    ] = np.nan
    market_rollup.to_csv(master_dir / "_ALL_WINDOWS__MARKET_ROLLUP.csv", index=False)

    best_windows = graded_scorecard.sort_values(["market", "hit_rate"], ascending=[True, False]).copy()
    worst_windows = graded_scorecard.sort_values(["market", "hit_rate"], ascending=[True, True]).copy()
    tg_mask_best = best_windows["market"].astype("string").fillna("").str.lower().isin(["tg15", "tg25"])
    tg_mask_worst = worst_windows["market"].astype("string").fillna("").str.lower().isin(["tg15", "tg25"])
    for _col in ["roi_level_stake", "level_stake_profit", "avg_bookie_od"]:
        if _col in best_windows.columns:
            best_windows.loc[tg_mask_best, _col] = np.nan
        if _col in worst_windows.columns:
            worst_windows.loc[tg_mask_worst, _col] = np.nan
    if "evaluation_mode" not in best_windows.columns:
        best_windows["evaluation_mode"] = np.where(tg_mask_best, "ACCURACY_ONLY", "ROI_AND_ACCURACY")
    else:
        best_windows.loc[tg_mask_best, "evaluation_mode"] = "ACCURACY_ONLY"
        best_windows.loc[~tg_mask_best, "evaluation_mode"] = best_windows.loc[~tg_mask_best, "evaluation_mode"].fillna("ROI_AND_ACCURACY")
    if "evaluation_mode" not in worst_windows.columns:
        worst_windows["evaluation_mode"] = np.where(tg_mask_worst, "ACCURACY_ONLY", "ROI_AND_ACCURACY")
    else:
        worst_windows.loc[tg_mask_worst, "evaluation_mode"] = "ACCURACY_ONLY"
        worst_windows.loc[~tg_mask_worst, "evaluation_mode"] = worst_windows.loc[~tg_mask_worst, "evaluation_mode"].fillna("ROI_AND_ACCURACY")
    best_windows.to_csv(master_dir / "_ALL_WINDOWS__BEST_WINDOWS.csv", index=False)
    worst_windows.to_csv(master_dir / "_ALL_WINDOWS__WORST_WINDOWS.csv", index=False)

    graded_window_ids = set(graded_scorecard["window_id"].astype("string").tolist())
    graded_windows = [w for w in windows if w.window_id in graded_window_ids]
    build_btts_signal_audits(base_outdir, graded_windows)
    build_btts_live_product_audits(base_outdir, graded_windows)
    build_master_loss_audits(base_outdir, graded_windows)
    build_ftr_draw_risk_audits(base_outdir, graded_windows)
    build_ftr_variant_tests(base_outdir, graded_windows)
    build_ftr_standard_residual_audits(base_outdir, graded_windows)
    build_tg_master_audits(base_outdir, graded_windows)
    build_ftr_combo_master_audits(base_outdir, graded_windows)
    build_experimental_test_family_audits(base_outdir, graded_windows)
    
def build_model_probability_band(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    bands = pd.cut(
        vals,
        bins=[-np.inf, 0.40, 0.50, 0.60, 0.70, 0.80, np.inf],
        labels=["<=0.40", "0.40-0.50", "0.50-0.60", "0.60-0.70", "0.70-0.80", ">0.80"],
        right=True,
    )
    return bands.astype("string").fillna("UNBANDABLE")

def build_combo_probability_band(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    bands = pd.cut(
        vals,
        bins=[-np.inf, 0.40, 0.50, 0.60, 0.65, 0.70, 0.80, np.inf],
        labels=["<=0.40", "0.40-0.50", "0.50-0.60", "0.60-0.65", "0.65-0.70", "0.70-0.80", ">0.80"],
        right=True,
    )
    return bands.astype("string").fillna("UNBANDABLE")


def build_ge2_gap_band(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    bands = pd.cut(
        vals,
        bins=[-np.inf, 0.10, 0.15, 0.20, 0.25, 0.35, np.inf],
        labels=["<=0.10", "0.10-0.15", "0.15-0.20", "0.20-0.25", "0.25-0.35", ">0.35"],
        right=True,
    )
    return bands.astype("string").fillna("UNBANDABLE")

def build_implied_odds_band(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    bands = pd.cut(
        vals,
        bins=[-np.inf, 1.40, 1.60, 1.80, 2.00, 2.50, np.inf],
        labels=["<=1.40", "1.40-1.60", "1.60-1.80", "1.80-2.00", "2.00-2.50", ">2.50"],
        right=True,
    )
    return bands.astype("string").fillna("UNBANDABLE")


def make_grouped_count_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame(columns=[group_col, "rows"])
    return (
        df.groupby(group_col, dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["rows", group_col], ascending=[False, True])
        .reset_index(drop=True)
    )
DRAW_WARNING_TOKENS = [
    "DRAW",
    "CHAOS",
    "NOT_GLUE",
    "DEMOTED",
    "OBSERVE_RESCUE",
    "CS_PROMOTE",
    "ULTRASHORT",
]

def make_grouped_outcome_summary(df: pd.DataFrame, group_col: str, hit_col: str) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame(columns=[group_col, "rows", "graded_rows", "wins", "losses", "hit_rate", "avg_bookie_od", "avg_model_p_for_bookie"])

    grouped = (
        df.groupby(group_col, dropna=False)
        .agg(
            rows=(group_col, "size"),
            graded_rows=(hit_col, lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            wins=(hit_col, lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
            avg_bookie_od=("bookie_od", lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
            avg_model_p_for_bookie=("model_p_for_bookie", lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
        )
        .reset_index()
    )
    grouped["losses"] = grouped["graded_rows"] - grouped["wins"]
    grouped["hit_rate"] = grouped["wins"] / grouped["graded_rows"].replace(0, np.nan)
    return grouped.sort_values(["rows", "graded_rows"], ascending=[False, False]).reset_index(drop=True)

def make_grouped_tg_accuracy_summary(df: pd.DataFrame, group_col: str, hit_col: str) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame(columns=[group_col, "rows", "graded_rows", "wins", "losses", "hit_rate", "avg_model_p_for_bookie"])

    grouped = (
        df.groupby(group_col, dropna=False)
        .agg(
            rows=(group_col, "size"),
            graded_rows=(hit_col, lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            wins=(hit_col, lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
            avg_model_p_for_bookie=("model_p_for_bookie", lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
        )
        .reset_index()
    )
    grouped["losses"] = grouped["graded_rows"] - grouped["wins"]
    grouped["hit_rate"] = grouped["wins"] / grouped["graded_rows"].replace(0, np.nan)
    grouped["evaluation_mode"] = "ACCURACY_ONLY"
    return grouped.sort_values(["rows", "graded_rows"], ascending=[False, False]).reset_index(drop=True)

def make_grouped_combo_accuracy_summary(df: pd.DataFrame, group_col, hit_col: str) -> pd.DataFrame:
    group_cols = [group_col] if isinstance(group_col, str) else list(group_col)

    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        cols = group_cols + ["rows", "graded_rows", "wins", "losses", "hit_rate", "avg_model_p_for_bookie"]
        return pd.DataFrame(columns=cols)

    if hit_col not in df.columns:
        out = (
            df.groupby(group_cols, dropna=False)
            .agg(
                rows=(group_cols[0], "size"),
                avg_model_p_for_bookie=("model_p_for_bookie", lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
            )
            .reset_index()
        )
        out["graded_rows"] = 0
        out["wins"] = 0.0
        out["losses"] = 0.0
        out["hit_rate"] = np.nan
        out["evaluation_mode"] = "ACCURACY_ONLY"
        return out.sort_values(["rows"], ascending=[False]).reset_index(drop=True)

    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            rows=(group_cols[0], "size"),
            graded_rows=(hit_col, lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            wins=(hit_col, lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
            avg_model_p_for_bookie=("model_p_for_bookie", lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
        )
        .reset_index()
    )
    grouped["losses"] = grouped["graded_rows"] - grouped["wins"]
    grouped["hit_rate"] = grouped["wins"] / grouped["graded_rows"].replace(0, np.nan)
    grouped["evaluation_mode"] = "ACCURACY_ONLY"
    return grouped.sort_values(["rows", "graded_rows"], ascending=[False, False]).reset_index(drop=True)

def build_tg_master_audits(base_outdir: Path, windows: list[WindowSpec]) -> None:
    master_dir = base_outdir / "_MASTER"
    audit_dir = master_dir / "TG_MASTER_AUDITS"
    ensure_dir(audit_dir)

    frames: list[pd.DataFrame] = []

    for window in windows:
        scored_path = (
            base_outdir
            / window.window_id
            / "03_scored"
            / f"DEPLOY_COMBINED_SCORED_{window.date_from}_to_{window.date_to}.csv"
        )
        if not scored_path.exists():
            continue

        df = pd.read_csv(scored_path, low_memory=False)
        market = safe_series_str(df, "market").str.lower().str.strip()
        tg = df.loc[market.isin(["tg15", "tg25"])].copy()
        if tg.empty:
            continue

        tg["window_id"] = window.window_id
        tg["league"] = safe_series_str(tg, "league").str.strip()
        tg["selection"] = safe_series_str(tg, "selection").str.upper().str.strip()
        tg["source_tier_file"] = safe_series_str(tg, "source_tier_file").str.upper().str.strip()
        tg["model_probability_band"] = build_model_probability_band(
            pd.to_numeric(tg.get("model_p_for_bookie", np.nan), errors="coerce")
        )
        tg["evaluation_mode"] = "ACCURACY_ONLY"
        frames.append(tg)

    if not frames:
        log("[tg-master-audit] no scored TG files found.")
        return

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(audit_dir / "TG__COMBINED.csv", index=False)

    summary_rows: list[dict[str, object]] = []
    grouped_specs = [
        ("BY_LEAGUE", "league"),
        ("BY_SIDE", "selection"),
        ("BY_CONFIDENCE_BAND", "model_probability_band"),
        ("BY_TIER", "source_tier_file"),
    ]

    for market_name in ["tg15", "tg25"]:
        cfg = SUPPORTED_MARKETS[market_name]
        hit_col = cfg["hit_col"]
        sub = combined.loc[safe_series_str(combined, "market").str.lower().str.strip().eq(market_name)].copy()
        if sub.empty:
            continue

        overall = summarize_slice(sub, hit_col)
        overall["market"] = market_name
        overall["evaluation_mode"] = "ACCURACY_ONLY"
        summary_rows.append(overall)

        sub.to_csv(audit_dir / f"{market_name.upper()}__COMBINED.csv", index=False)

        wins = sub.loc[pd.to_numeric(sub.get(hit_col, np.nan), errors="coerce") == 1].copy()
        losses = sub.loc[pd.to_numeric(sub.get(hit_col, np.nan), errors="coerce") == 0].copy()
        wins.to_csv(audit_dir / f"{market_name.upper()}__WINS.csv", index=False)
        losses.to_csv(audit_dir / f"{market_name.upper()}__LOSSES.csv", index=False)

        detail_cols = [
            c for c in [
                "window_id",
                "league",
                "match_date",
                "home_team_name",
                "away_team_name",
                "market",
                "selection",
                "source_tier_file",
                "model_p_for_bookie",
                "model_probability_band",
                "score",
                "actual_tg15",
                "actual_tg25",
                "tg15_hit",
                "tg25_hit",
                "actual_tg15_home",
                "actual_tg15_away",
                "actual_tg25_home",
                "actual_tg25_away",
                "evaluation_mode",
            ]
            if c in sub.columns
        ]
        sub[detail_cols].to_csv(audit_dir / f"{market_name.upper()}__DETAIL.csv", index=False)

        for label, group_col in grouped_specs:
            grouped = make_grouped_tg_accuracy_summary(sub, group_col, hit_col)
            grouped.to_csv(audit_dir / f"{market_name.upper()}__{label}.csv", index=False)

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(audit_dir / "TG__SUMMARY.csv", index=False)
        
def build_ftr_combo_master_audits(base_outdir: Path, windows: list[WindowSpec]) -> None:
    master_dir = base_outdir / "_MASTER"
    audit_dir = master_dir / "FTR_COMBO_MASTER_AUDITS"
    ensure_dir(audit_dir)

    frames: list[pd.DataFrame] = []

    for window in windows:
        scored_path = (
            base_outdir
            / window.window_id
            / "03_scored"
            / f"DEPLOY_COMBINED_SCORED_{window.date_from}_to_{window.date_to}.csv"
        )
        if not scored_path.exists():
            continue

        df = pd.read_csv(scored_path, low_memory=False)
        market = safe_series_str(df, "market").str.lower().str.strip()
        ftr = df.loc[market.eq("ftr")].copy()
        if ftr.empty:
            continue

        ftr["window_id"] = window.window_id
        ftr["league"] = safe_series_str(ftr, "league").str.strip()
        ftr["selection"] = safe_series_str(ftr, "selection").str.upper().str.strip()
        ftr["source_tier_file"] = safe_series_str(ftr, "source_tier_file").str.upper().str.strip()
        ftr["model_probability_band"] = build_model_probability_band(
            pd.to_numeric(ftr.get("model_p_for_bookie", np.nan), errors="coerce")
        )
        ftr["power_diff_band"] = build_power_diff_band(
            pd.to_numeric(ftr.get("power_diff", np.nan), errors="coerce")
        )
        ftr["combo_side"] = np.where(
            ftr["selection"].eq("HOME"),
            "HOME_WIN_AND_HOME_GE2",
            np.where(ftr["selection"].eq("AWAY"), "AWAY_WIN_AND_AWAY_GE2", "OTHER"),
        )

        home_lambda_num = pd.to_numeric(ftr.get("lambda_home", np.nan), errors="coerce").clip(lower=0.0, upper=10.0)
        away_lambda_num = pd.to_numeric(ftr.get("lambda_away", np.nan), errors="coerce").clip(lower=0.0, upper=10.0)

        direct_home_combo = pd.to_numeric(
            ftr.get("p_hw_and_hge2", pd.Series(np.nan, index=ftr.index)),
            errors="coerce",
        )
        if not isinstance(direct_home_combo, pd.Series):
            direct_home_combo = pd.Series(direct_home_combo, index=ftr.index, dtype="float64")

        direct_away_combo = pd.to_numeric(
            ftr.get("p_aw_and_age2", pd.Series(np.nan, index=ftr.index)),
            errors="coerce",
        )
        if not isinstance(direct_away_combo, pd.Series):
            direct_away_combo = pd.Series(direct_away_combo, index=ftr.index, dtype="float64")

        # Audit-side fallback: reconstruct combo probabilities from lambdas when older scored files
        # do not yet carry p_hw_and_hge2 / p_aw_and_age2.
        max_goals = 8
        factorials = np.array([1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0], dtype="float64")

        home_prob_cols = []
        away_prob_cols = []
        exp_home = np.exp(-home_lambda_num)
        exp_away = np.exp(-away_lambda_num)

        for k in range(max_goals + 1):
            home_prob_cols.append((exp_home * (home_lambda_num ** k) / factorials[k]).astype("float64"))
            away_prob_cols.append((exp_away * (away_lambda_num ** k) / factorials[k]).astype("float64"))

        home_probs = pd.concat(home_prob_cols, axis=1)
        away_probs = pd.concat(away_prob_cols, axis=1)

        poisson_home_combo = pd.Series(0.0, index=ftr.index, dtype="float64")
        poisson_away_combo = pd.Series(0.0, index=ftr.index, dtype="float64")

        for h in range(2, max_goals + 1):
            for a in range(0, h):
                poisson_home_combo = poisson_home_combo + (home_probs.iloc[:, h] * away_probs.iloc[:, a])

        for a in range(2, max_goals + 1):
            for h in range(0, a):
                poisson_away_combo = poisson_away_combo + (home_probs.iloc[:, h] * away_probs.iloc[:, a])

        home_combo_final = direct_home_combo.where(direct_home_combo.notna(), poisson_home_combo.clip(lower=0.0, upper=1.0))
        away_combo_final = direct_away_combo.where(direct_away_combo.notna(), poisson_away_combo.clip(lower=0.0, upper=1.0))

        ftr["combo_prob"] = np.where(
            ftr["selection"].eq("HOME"),
            home_combo_final,
            np.where(
                ftr["selection"].eq("AWAY"),
                away_combo_final,
                np.nan,
            ),
        )
        ftr["combo_prob"] = pd.to_numeric(ftr["combo_prob"], errors="coerce")
        ftr["combo_probability_band"] = build_combo_probability_band(pd.to_numeric(ftr["combo_prob"], errors="coerce"))
        if not bool(pd.to_numeric(ftr["combo_prob"], errors="coerce").notna().any()):
            log("[ftr-combo-audit] combo_prob still all-NaN after combo enrichment")

        ftr["combo_lambda"] = np.where(
            ftr["selection"].eq("HOME"),
            home_lambda_num,
            np.where(
                ftr["selection"].eq("AWAY"),
                away_lambda_num,
                np.nan,
            ),
        )
        ftr["combo_lambda"] = pd.to_numeric(ftr["combo_lambda"], errors="coerce")
        ftr["combo_lambda_band"] = build_lambda_band(pd.to_numeric(ftr["combo_lambda"], errors="coerce"))

        home_ge2_head_direct = pd.to_numeric(
            ftr.get("p_home_ge2", pd.Series(np.nan, index=ftr.index)),
            errors="coerce",
        )
        if not isinstance(home_ge2_head_direct, pd.Series):
            home_ge2_head_direct = pd.Series(home_ge2_head_direct, index=ftr.index, dtype="float64")

        away_ge2_head_direct = pd.to_numeric(
            ftr.get("p_away_ge2", pd.Series(np.nan, index=ftr.index)),
            errors="coerce",
        )
        if not isinstance(away_ge2_head_direct, pd.Series):
            away_ge2_head_direct = pd.Series(away_ge2_head_direct, index=ftr.index, dtype="float64")

        home_ge2_head_conf = pd.to_numeric(
            ftr.get("home_ge2_confidence", pd.Series(np.nan, index=ftr.index)),
            errors="coerce",
        )
        if not isinstance(home_ge2_head_conf, pd.Series):
            home_ge2_head_conf = pd.Series(home_ge2_head_conf, index=ftr.index, dtype="float64")

        away_ge2_head_conf = pd.to_numeric(
            ftr.get("away_ge2_confidence", pd.Series(np.nan, index=ftr.index)),
            errors="coerce",
        )
        if not isinstance(away_ge2_head_conf, pd.Series):
            away_ge2_head_conf = pd.Series(away_ge2_head_conf, index=ftr.index, dtype="float64")

        exp_home = np.exp(-home_lambda_num)
        exp_away = np.exp(-away_lambda_num)
        home_ge2_poisson = (1.0 - (exp_home * (1.0 + home_lambda_num))).clip(lower=0.0, upper=1.0)
        away_ge2_poisson = (1.0 - (exp_away * (1.0 + away_lambda_num))).clip(lower=0.0, upper=1.0)

        home_ge2_head = home_ge2_head_direct.where(home_ge2_head_direct.notna(), home_ge2_head_conf)
        away_ge2_head = away_ge2_head_direct.where(away_ge2_head_direct.notna(), away_ge2_head_conf)

        home_ge2_head = home_ge2_head.where(home_ge2_head.notna(), home_ge2_poisson)
        away_ge2_head = away_ge2_head.where(away_ge2_head.notna(), away_ge2_poisson)

        ftr["ge2_head_prob"] = np.where(
            ftr["selection"].eq("HOME"),
            home_ge2_head,
            np.where(ftr["selection"].eq("AWAY"), away_ge2_head, np.nan),
        )
        ftr["ge2_head_prob"] = pd.to_numeric(ftr["ge2_head_prob"], errors="coerce")

        ftr["ge2_gap"] = (
            pd.to_numeric(ftr["combo_prob"], errors="coerce")
            - pd.to_numeric(ftr["ge2_head_prob"], errors="coerce")
        ).abs()
        ftr["ge2_gap"] = pd.to_numeric(ftr["ge2_gap"], errors="coerce")
        ftr["ge2_gap_band"] = build_ge2_gap_band(pd.to_numeric(ftr["ge2_gap"], errors="coerce"))
        if not bool(pd.to_numeric(ftr["ge2_gap"], errors="coerce").notna().any()):
            log("[ftr-combo-audit] ge2_gap still all-NaN after combo enrichment")
        ftr["ge2_consistent_flag"] = np.where(
            pd.to_numeric(ftr["ge2_gap"], errors="coerce").notna()
            & (pd.to_numeric(ftr["ge2_gap"], errors="coerce") < 0.25),
            "YES",
            np.where(pd.to_numeric(ftr["ge2_gap"], errors="coerce").notna(), "NO", "UNBANDABLE"),
        )

        for combo_col in [
            "actual_hw_and_hge2",
            "actual_aw_and_age2",
            "hw_and_hge2_hit",
            "aw_and_age2_hit",
        ]:
            if combo_col not in ftr.columns:
                ftr[combo_col] = pd.Series(np.nan, index=ftr.index, dtype="float64")

        combo_bucket_native = safe_series_str(ftr, "combo_overlay_bucket").str.upper().str.strip()
        std_bucket = safe_series_str(ftr, "standard_reporting_bucket").str.upper().str.strip()
        selection = safe_series_str(ftr, "selection").str.upper().str.strip()

        combo_prob_num = pd.to_numeric(ftr.get("combo_prob", np.nan), errors="coerce")
        combo_lambda_num = pd.to_numeric(ftr.get("combo_lambda", np.nan), errors="coerce")
        ge2_gap_num = pd.to_numeric(ftr.get("ge2_gap", np.nan), errors="coerce")

        audit_bucket = pd.Series("", index=ftr.index, dtype="string")

        audit_bucket = audit_bucket.mask(
            combo_bucket_native.eq("HOME_WIN_AND_HOME_GE2_VERY_STRONG_HW"),
            "HOME_WIN_AND_HOME_GE2_VERY_STRONG_HW",
        )
        audit_bucket = audit_bucket.mask(
            combo_bucket_native.eq("HOME_WIN_AND_HOME_GE2_STRONG_HW"),
            "HOME_WIN_AND_HOME_GE2_STRONG_HW",
        )
        audit_bucket = audit_bucket.mask(
            combo_bucket_native.eq("AWAY_WIN_AND_AWAY_GE2_VERY_STRONG_AW"),
            "AWAY_WIN_AND_AWAY_GE2_VERY_STRONG_AW",
        )
        audit_bucket = audit_bucket.mask(
            combo_bucket_native.eq("AWAY_WIN_AND_AWAY_GE2_STRONG_AW"),
            "AWAY_WIN_AND_AWAY_GE2_STRONG_AW",
        )
        audit_bucket = audit_bucket.mask(
            combo_bucket_native.eq("FTR_CS_PROMOTED") | std_bucket.eq("STANDARD_FTR_CS_PROMOTED_ALIGNED"),
            "FTR_CS_PROMOTED",
        )

        m_empty = audit_bucket.eq("")

        audit_bucket = audit_bucket.mask(
            m_empty
            & selection.eq("HOME")
            & combo_prob_num.ge(0.80)
            & combo_lambda_num.ge(2.50)
            & ge2_gap_num.lt(0.25),
            "HOME_WIN_AND_HOME_GE2_VERY_STRONG_HW",
        )

        audit_bucket = audit_bucket.mask(
            audit_bucket.eq("")
            & selection.eq("HOME")
            & combo_prob_num.ge(0.65)
            & combo_lambda_num.ge(2.00)
            & ge2_gap_num.lt(0.35),
            "HOME_WIN_AND_HOME_GE2_STRONG_HW",
        )

        audit_bucket = audit_bucket.mask(
            audit_bucket.eq("")
            & selection.eq("AWAY")
            & combo_prob_num.ge(0.80)
            & combo_lambda_num.ge(2.50)
            & ge2_gap_num.lt(0.25),
            "AWAY_WIN_AND_AWAY_GE2_VERY_STRONG_AW",
        )

        audit_bucket = audit_bucket.mask(
            audit_bucket.eq("")
            & selection.eq("AWAY")
            & combo_prob_num.ge(0.65)
            & combo_lambda_num.ge(2.00)
            & ge2_gap_num.lt(0.35),
            "AWAY_WIN_AND_AWAY_GE2_STRONG_AW",
        )

        ftr["audit_bucket"] = audit_bucket.astype("string").fillna("")
        ftr["evaluation_mode"] = "ACCURACY_ONLY"
        frames.append(ftr)

    if not frames:
        log("[ftr-combo-audit] no scored FTR files found.")
        return

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(audit_dir / "FTR_COMBO__COMBINED.csv", index=False)

    audit_sub = combined.loc[safe_series_str(combined, "audit_bucket").ne("")].copy()
    if not audit_sub.empty:
        bucket_hit = pd.Series(np.nan, index=audit_sub.index, dtype="float64")
        bucket_hit = bucket_hit.mask(
            audit_sub["audit_bucket"].eq("HOME_WIN_AND_HOME_GE2_VERY_STRONG_HW"),
            pd.to_numeric(audit_sub.get("hw_and_hge2_hit", np.nan), errors="coerce"),
        )
        bucket_hit = bucket_hit.mask(
            audit_sub["audit_bucket"].eq("HOME_WIN_AND_HOME_GE2_STRONG_HW"),
            pd.to_numeric(audit_sub.get("hw_and_hge2_hit", np.nan), errors="coerce"),
        )
        bucket_hit = bucket_hit.mask(
            audit_sub["audit_bucket"].eq("AWAY_WIN_AND_AWAY_GE2_VERY_STRONG_AW"),
            pd.to_numeric(audit_sub.get("aw_and_age2_hit", np.nan), errors="coerce"),
        )
        bucket_hit = bucket_hit.mask(
            audit_sub["audit_bucket"].eq("AWAY_WIN_AND_AWAY_GE2_STRONG_AW"),
            pd.to_numeric(audit_sub.get("aw_and_age2_hit", np.nan), errors="coerce"),
        )
        bucket_hit = bucket_hit.mask(
            audit_sub["audit_bucket"].eq("FTR_CS_PROMOTED"),
            pd.to_numeric(audit_sub.get("ftr_hit", np.nan), errors="coerce"),
        )
        audit_sub["audit_hit"] = pd.to_numeric(bucket_hit, errors="coerce")

        bucket_summary = (
            audit_sub.groupby("audit_bucket", dropna=False)
            .agg(
                rows=("audit_bucket", "size"),
                graded_rows=("audit_hit", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
                wins=("audit_hit", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
            )
            .reset_index()
        )
        bucket_summary["losses"] = bucket_summary["graded_rows"] - bucket_summary["wins"]
        bucket_summary["hit_rate"] = bucket_summary["wins"] / bucket_summary["graded_rows"].replace(0, np.nan)
        bucket_summary = bucket_summary.sort_values(["rows", "graded_rows", "audit_bucket"], ascending=[False, False, True]).reset_index(drop=True)
        bucket_summary.to_csv(audit_dir / "FTR_COMBO__AUDIT_BUCKET_SUMMARY.csv", index=False)

        audit_sub.to_csv(audit_dir / "FTR_COMBO__AUDIT_BUCKET_DETAIL.csv", index=False)
        audit_sub.loc[pd.to_numeric(audit_sub["audit_hit"], errors="coerce") == 1].to_csv(
            audit_dir / "FTR_COMBO__AUDIT_BUCKET_WINS.csv", index=False
        )
        audit_sub.loc[pd.to_numeric(audit_sub["audit_hit"], errors="coerce") == 0].to_csv(
            audit_dir / "FTR_COMBO__AUDIT_BUCKET_LOSSES.csv", index=False
        )

        grouped_specs = [
            ("BY_BUCKET", "audit_bucket"),
            ("BY_BUCKET_LEAGUE", ["audit_bucket", "league"]),
            ("BY_BUCKET_TIER", ["audit_bucket", "source_tier_file"]),
            ("BY_BUCKET_CONFIDENCE_BAND", ["audit_bucket", "model_probability_band"]),
            ("BY_BUCKET_POWER_DIFF_BAND", ["audit_bucket", "power_diff_band"]),
            ("BY_BUCKET_COMBO_PROBABILITY_BAND", ["audit_bucket", "combo_probability_band"]),
            ("BY_BUCKET_LAMBDA_BAND", ["audit_bucket", "combo_lambda_band"]),
            ("BY_BUCKET_GE2_GAP_BAND", ["audit_bucket", "ge2_gap_band"]),
            ("BY_BUCKET_GE2_CONSISTENT_FLAG", ["audit_bucket", "ge2_consistent_flag"]),
        ]
        for group_label, group_col in grouped_specs:
            grouped = make_grouped_combo_accuracy_summary(audit_sub, group_col, "audit_hit")
            grouped.to_csv(audit_dir / f"FTR_COMBO__AUDIT_{group_label}.csv", index=False)

    combo_specs = [
        ("HOME_WIN_AND_HOME_GE2", "HOME", "hw_and_hge2_hit"),
        ("AWAY_WIN_AND_AWAY_GE2", "AWAY", "aw_and_age2_hit"),
    ]

    for label, selection_value, hit_col in combo_specs:
        sub = combined.loc[safe_series_str(combined, "selection").str.upper().eq(selection_value)].copy()
        if sub.empty:
            continue

        if hit_col not in sub.columns:
            sub[hit_col] = pd.Series(np.nan, index=sub.index, dtype="float64")
        graded = pd.to_numeric(sub.get(hit_col, pd.Series(np.nan, index=sub.index)), errors="coerce")
        if not isinstance(graded, pd.Series):
            graded = pd.Series(graded, index=sub.index, dtype="float64")
        wins = float(graded.fillna(0).sum())
        graded_rows = int(graded.notna().sum())
        losses = graded_rows - wins

        summary = pd.DataFrame([{
            "combo_market": label,
            "rows": int(len(sub)),
            "graded": graded_rows,
            "wins": wins,
            "losses": losses,
            "hit_rate": float(wins / graded_rows) if graded_rows else np.nan,
            "avg_model_p_for_bookie": float(pd.to_numeric(sub.get("model_p_for_bookie", np.nan), errors="coerce").mean()) if len(sub) else np.nan,
            "evaluation_mode": "ACCURACY_ONLY",
        }])
        summary.to_csv(audit_dir / f"{label}__SUMMARY.csv", index=False)

        sub.to_csv(audit_dir / f"{label}__COMBINED.csv", index=False)
        hit_series = pd.to_numeric(sub.get(hit_col, pd.Series(np.nan, index=sub.index)), errors="coerce")
        if not isinstance(hit_series, pd.Series):
            hit_series = pd.Series(hit_series, index=sub.index, dtype="float64")
        hit_series = hit_series.reindex(sub.index)

        sub.loc[hit_series.eq(1)].to_csv(
            audit_dir / f"{label}__WINS.csv", index=False
        )
        sub.loc[hit_series.eq(0)].to_csv(
            audit_dir / f"{label}__LOSSES.csv", index=False
        )

        detail_cols = [
            c for c in [
                "window_id",
                "league",
                "match_date",
                "home_team_name",
                "away_team_name",
                "selection",
                "source_tier_file",
                "standard_reporting_bucket",
                "combo_overlay_bucket",
                "audit_bucket",
                "model_p_for_bookie",
                "model_probability_band",
                "power_diff_band",
                "combo_side",
                "p_hw_and_hge2",
                "p_aw_and_age2",
                "combo_prob",
                "combo_probability_band",
                "combo_lambda",
                "combo_lambda_band",
                "ge2_head_prob",
                "ge2_gap",
                "ge2_gap_band",
                "ge2_consistent_flag",
                "bookie_od",
                "actual_ftr",
                "ftr_hit",
                "actual_hw_and_hge2",
                "actual_aw_and_age2",
                "hw_and_hge2_hit",
                "aw_and_age2_hit",
                "evaluation_mode",
            ]
            if c in sub.columns
        ]
        sub[detail_cols].to_csv(audit_dir / f"{label}__DETAIL.csv", index=False)

        grouped_specs = [
            ("BY_LEAGUE", "league"),
            ("BY_CONFIDENCE_BAND", "model_probability_band"),
            ("BY_TIER", "source_tier_file"),
            ("BY_BUCKET", "standard_reporting_bucket"),
            ("BY_AUDIT_BUCKET", "audit_bucket"),
            ("BY_POWER_DIFF_BAND", "power_diff_band"),
            ("BY_SIDE", "combo_side"),
            ("BY_COMBO_PROBABILITY_BAND", "combo_probability_band"),
            ("BY_LAMBDA_BAND", "combo_lambda_band"),
            ("BY_GE2_GAP_BAND", "ge2_gap_band"),
            ("BY_GE2_CONSISTENT_FLAG", "ge2_consistent_flag"),
            ("BY_LEAGUE_SIDE", ["league", "combo_side"]),
            ("BY_LEAGUE_CONFIDENCE_BAND", ["league", "model_probability_band"]),
            ("BY_SIDE_CONFIDENCE_BAND", ["combo_side", "model_probability_band"]),
            ("BY_LEAGUE_SIDE_CONFIDENCE_BAND", ["league", "combo_side", "model_probability_band"]),
            ("BY_LEAGUE_COMBO_PROBABILITY_BAND", ["league", "combo_probability_band"]),
            ("BY_SIDE_COMBO_PROBABILITY_BAND", ["combo_side", "combo_probability_band"]),
            ("BY_LEAGUE_SIDE_COMBO_PROBABILITY_BAND", ["league", "combo_side", "combo_probability_band"]),
            ("BY_LEAGUE_LAMBDA_BAND", ["league", "combo_lambda_band"]),
            ("BY_SIDE_LAMBDA_BAND", ["combo_side", "combo_lambda_band"]),
            ("BY_LEAGUE_SIDE_LAMBDA_BAND", ["league", "combo_side", "combo_lambda_band"]),
            ("BY_LEAGUE_GE2_GAP_BAND", ["league", "ge2_gap_band"]),
            ("BY_SIDE_GE2_GAP_BAND", ["combo_side", "ge2_gap_band"]),
            ("BY_LEAGUE_SIDE_GE2_GAP_BAND", ["league", "combo_side", "ge2_gap_band"]),
            ("BY_LEAGUE_GE2_CONSISTENT_FLAG", ["league", "ge2_consistent_flag"]),
        ]

        for group_label, group_col in grouped_specs:
            grouped = make_grouped_combo_accuracy_summary(sub, group_col, hit_col)
            grouped.to_csv(audit_dir / f"{label}__{group_label}.csv", index=False)

    shortlist_rows: list[dict[str, object]] = []
    combined_market = safe_series_str(combined, "market").str.lower().str.strip()
    combined_selection = safe_series_str(combined, "selection").str.upper().str.strip()
    combined_model_p = pd.to_numeric(combined.get("model_p_for_bookie", np.nan), errors="coerce")
    combined_combo_p = pd.to_numeric(combined.get("combo_prob", np.nan), errors="coerce")
    combined_combo_lambda = pd.to_numeric(combined.get("combo_lambda", np.nan), errors="coerce")
    combined_ge2_gap = pd.to_numeric(combined.get("ge2_gap", np.nan), errors="coerce")
    combined_power_band = safe_series_str(combined, "power_diff_band")
    combined_league = safe_series_str(combined, "league").str.strip()

    shortlist_specs = [
        ("HOME_WIN_AND_HOME_GE2", combined_selection.eq("HOME"), "hw_and_hge2_hit"),
        ("AWAY_WIN_AND_AWAY_GE2", combined_selection.eq("AWAY"), "aw_and_age2_hit"),
    ]

    for combo_name, base_mask, hit_col in shortlist_specs:
        if hit_col not in combined.columns:
            continue

        for combo_p_floor in [0.50, 0.60, 0.65, 0.70, 0.75, 0.80]:
            sub_combo_p = combined.loc[base_mask & combined_combo_p.ge(combo_p_floor)].copy()
            if sub_combo_p.empty:
                continue
            stats = summarize_slice(sub_combo_p, hit_col)
            shortlist_rows.append({
                "combo_market": combo_name,
                "filter_family": "COMBO_P_FLOOR",
                "filter_value": f">={combo_p_floor:.2f}",
                **stats,
                "evaluation_mode": "ACCURACY_ONLY",
            })

        for lambda_floor in [2.0, 2.25, 2.50, 2.75, 3.0]:
            sub_lambda = combined.loc[base_mask & combined_combo_lambda.ge(lambda_floor)].copy()
            if sub_lambda.empty:
                continue
            stats = summarize_slice(sub_lambda, hit_col)
            shortlist_rows.append({
                "combo_market": combo_name,
                "filter_family": "LAMBDA_FLOOR",
                "filter_value": f">={lambda_floor:.2f}",
                **stats,
                "evaluation_mode": "ACCURACY_ONLY",
            })

        for gap_cap in [0.35, 0.25, 0.20, 0.15, 0.10]:
            sub_gap = combined.loc[base_mask & combined_ge2_gap.le(gap_cap)].copy()
            if sub_gap.empty:
                continue
            stats = summarize_slice(sub_gap, hit_col)
            shortlist_rows.append({
                "combo_market": combo_name,
                "filter_family": "GE2_GAP_CAP",
                "filter_value": f"<={gap_cap:.2f}",
                **stats,
                "evaluation_mode": "ACCURACY_ONLY",
            })

        for band_value in ["10-15", "15-20", "20-30", ">30"]:
            sub_band = combined.loc[base_mask & combined_power_band.eq(band_value)].copy()
            if sub_band.empty:
                continue
            stats = summarize_slice(sub_band, hit_col)
            shortlist_rows.append({
                "combo_market": combo_name,
                "filter_family": "POWER_DIFF_BAND",
                "filter_value": band_value,
                **stats,
                "evaluation_mode": "ACCURACY_ONLY",
            })

        league_counts = (
            combined.loc[base_mask]
            .groupby(combined_league.loc[base_mask], dropna=False)
            .size()
            .sort_values(ascending=False)
        )
        top_leagues = [idx for idx in league_counts.index.tolist() if str(idx).strip()][:16]
        for league_name in top_leagues:
            sub_league = combined.loc[base_mask & combined_league.eq(league_name)].copy()
            if sub_league.empty:
                continue
            stats = summarize_slice(sub_league, hit_col)
            shortlist_rows.append({
                "combo_market": combo_name,
                "filter_family": "LEAGUE",
                "filter_value": league_name,
                **stats,
                "evaluation_mode": "ACCURACY_ONLY",
            })

        sub_structural = combined.loc[
            base_mask
            & combined_combo_lambda.ge(2.50)
            & combined_combo_p.ge(0.65)
            & combined_ge2_gap.lt(0.25)
        ].copy()
        if not sub_structural.empty:
            stats = summarize_slice(sub_structural, hit_col)
            shortlist_rows.append({
                "combo_market": combo_name,
                "filter_family": "STRUCTURAL_3_FILTER",
                "filter_value": "lambda>=2.50 & combo_p>=0.65 & ge2_gap<0.25",
                **stats,
                "evaluation_mode": "ACCURACY_ONLY",
            })

        sub_domination = combined.loc[
            base_mask
            & combined_combo_lambda.ge(2.50)
            & combined_combo_p.ge(0.65)
            & combined_ge2_gap.lt(0.25)
            & combined_power_band.isin(["20-30", ">30"])
        ].copy()
        if not sub_domination.empty:
            stats = summarize_slice(sub_domination, hit_col)
            shortlist_rows.append({
                "combo_market": combo_name,
                "filter_family": "DOMINATION_4_FILTER",
                "filter_value": "lambda>=2.50 & combo_p>=0.65 & ge2_gap<0.25 & power_diff>=20-band",
                **stats,
                "evaluation_mode": "ACCURACY_ONLY",
            })

        sub_elite = combined.loc[
            base_mask
            & combined_combo_p.ge(0.80)
            & combined_ge2_gap.lt(0.25)
            & combined_power_band.isin(["15-20", "20-30", ">30"])
        ].copy()
        if not sub_elite.empty:
            stats = summarize_slice(sub_elite, hit_col)
            shortlist_rows.append({
                "combo_market": combo_name,
                "filter_family": "ELITE_COMBO_SHORTLIST",
                "filter_value": "combo_p>=0.80 & ge2_gap<0.25 & power_diff>=15-band",
                **stats,
                "evaluation_mode": "ACCURACY_ONLY",
            })

        sub_wrong_signal = combined.loc[base_mask & combined_model_p.ge(0.50)].copy()
        if not sub_wrong_signal.empty:
            stats = summarize_slice(sub_wrong_signal, hit_col)
            shortlist_rows.append({
                "combo_market": combo_name,
                "filter_family": "LEGACY_MODEL_P_FLOOR",
                "filter_value": ">=0.50",
                **stats,
                "evaluation_mode": "ACCURACY_ONLY",
            })

    if shortlist_rows:
        pd.DataFrame(shortlist_rows).to_csv(
            audit_dir / "FTR_COMBO__SHORTLIST_FILTER_SWEEPS.csv",
            index=False,
        )

def build_power_diff_band(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    bands = pd.cut(
        vals,
        bins=[-np.inf, -30, -18, -8, 8, 18, 30, np.inf],
        labels=["<=-30", "-30_to_-18", "-18_to_-8", "-8_to_8", "8_to_18", "18_to_30", ">30"],
        right=True,
    )
    return bands.astype("string").fillna("UNBANDABLE")

def build_lambda_band(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    bands = pd.cut(
        vals,
        bins=[-np.inf, 1.5, 2.0, 2.5, 3.0, 3.5, np.inf],
        labels=["<=1.5", "1.5-2.0", "2.0-2.5", "2.5-3.0", "3.0-3.5", ">3.5"],
        right=True,
    )
    return bands.astype("string").fillna("UNBANDABLE")


def build_combo_probability_band(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    bands = pd.cut(
        vals,
        bins=[-np.inf, 0.40, 0.50, 0.60, 0.65, 0.70, 0.80, np.inf],
        labels=["<=0.40", "0.40-0.50", "0.50-0.60", "0.60-0.65", "0.65-0.70", "0.70-0.80", ">0.80"],
        right=True,
    )
    return bands.astype("string").fillna("UNBANDABLE")


def build_ge2_gap_band(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").abs()
    bands = pd.cut(
        vals,
        bins=[-np.inf, 0.10, 0.15, 0.20, 0.25, 0.35, np.inf],
        labels=["<=0.10", "0.10-0.15", "0.15-0.20", "0.20-0.25", "0.25-0.35", ">0.35"],
        right=True,
    )
    return bands.astype("string").fillna("UNBANDABLE")

def build_ppg_diff_band(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    bands = pd.cut(
        vals,
        bins=[-np.inf, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, np.inf],
        labels=["<=-1.0", "-1.0_to_-0.6", "-0.6_to_-0.2", "-0.2_to_0.2", "0.2_to_0.6", "0.6_to_1.0", ">1.0"],
        right=True,
    )
    return bands.astype("string").fillna("UNBANDABLE")


def extract_draw_warning_token_bucket(series: pd.Series) -> pd.Series:
    reason_codes = parse_reason_codes_to_upper(series)
    out = pd.Series("NO_DRAWWARN", index=reason_codes.index, dtype="string")

    out = out.mask(reason_codes.str.contains("OBSERVE_RESCUE", regex=False), "OBSERVE_RESCUE")
    out = out.mask(reason_codes.str.contains("CS_PROMOTE", regex=False), "CS_PROMOTED")
    out = out.mask(reason_codes.str.contains("ULTRASHORT", regex=False), "ULTRASHORT")
    out = out.mask(reason_codes.str.contains("DEMOTED", regex=False), "DEMOTED")
    out = out.mask(reason_codes.str.contains("NOT_GLUE", regex=False), "NOT_GLUE")
    out = out.mask(reason_codes.str.contains("CHAOS", regex=False), "CHAOS")
    out = out.mask(reason_codes.str.contains("DRAW", regex=False), "DRAW_TOKEN")

    multi_mask = reason_codes.str.contains("OBSERVE_RESCUE", regex=False) & reason_codes.str.contains("CHAOS", regex=False)
    out = out.mask(multi_mask, "OBSERVE_RESCUE_PLUS_CHAOS")
    return out.astype("string").fillna("NO_DRAWWARN")


VARIANT_TEST_SPECS = [
    {
        "variant_name": "FTR_STANDARD_BASELINE",
        "market": "ftr",
        "tier": "STANDARD",
        "description": "Baseline STANDARD FTR rows with no extra variant filtering.",
    },
    {
        "variant_name": "FTR_STANDARD_OBSERVE_RESCUE_MODELP_GE_0P41",
        "market": "ftr",
        "tier": "STANDARD",
        "description": "Keep STANDARD FTR rows, but require OBSERVE_RESCUE rows to have model_p_for_bookie >= 0.41.",
    },
    {
        "variant_name": "FTR_STANDARD_PATCH_A_LITE_ONLY",
        "market": "ftr",
        "tier": "STANDARD",
        "description": "Keep STANDARD FTR rows, but demote OBSERVE_RESCUE rows when model_p_for_bookie < 0.41.",
    },
    {
        "variant_name": "FTR_STANDARD_OBSERVE_RESCUE_STRICT_TRIAD",
        "market": "ftr",
        "tier": "STANDARD",
        "description": "Keep STANDARD FTR rows, but require OBSERVE_RESCUE rows to clear model_p >= 0.41, |power_diff| >= 18, and |ppg_diff_pre| >= 0.60.",
    },
    {
        "variant_name": "FTR_STANDARD_CS_PROMOTED_MODELP_GE_0P41_NO_DRAWWARN",
        "market": "ftr",
        "tier": "STANDARD",
        "description": "Keep STANDARD FTR rows, but require CS_PROMOTED rows to clear model_p >= 0.41 and have no draw-warning markers.",
    },
    {
        "variant_name": "FTR_STANDARD_SHORTPRICE_MODEL_LT_0P41_BLOCK",
        "market": "ftr",
        "tier": "STANDARD",
        "description": "Keep STANDARD FTR rows, but block short-price rows with odds <= 1.60 and model_p_for_bookie < 0.41.",
    },
    {
        "variant_name": "FTR_STANDARD_PATCH_C_ONLY",
        "market": "ftr",
        "tier": "STANDARD",
        "description": "Keep STANDARD FTR rows, but demote short-price rows with odds <= 1.60 and model_p_for_bookie < 0.41.",
    },
    {
        "variant_name": "FTR_STANDARD_BASE_DRAW_GAP_LE_0P15_AND_WEAK_POWER_BLOCK",
        "market": "ftr",
        "tier": "STANDARD",
        "description": "Keep STANDARD FTR rows, but block STANDARD_FTR_BASE rows when selected_minus_draw <= 0.15 and |power_diff| <= 18.",
    },
    {
        "variant_name": "FTR_STANDARD_BASE_NO_DRAWWARN_DRAW_GAP_LE_0P15_AND_WEAK_POWER_BLOCK",
        "market": "ftr",
        "tier": "STANDARD",
        "description": "Keep STANDARD FTR rows, but block STANDARD_FTR_BASE rows with NO_DRAWWARN when selected_minus_draw <= 0.15 and |power_diff| <= 18.",
    },
    {
        "variant_name": "FTR_STANDARD_BASE_DEMOTED_DRAW_GAP_LE_0P15_AND_WEAK_POWER_BLOCK",
        "market": "ftr",
        "tier": "STANDARD",
        "description": "Keep STANDARD FTR rows, but block STANDARD_FTR_BASE rows with DEMOTED draw-warning bucket when selected_minus_draw <= 0.15 and |power_diff| <= 18.",
    },
    {
        "variant_name": "FTR_STANDARD_PATCH_C_PLUS_PATCH_D_COMBO",
        "market": "ftr",
        "tier": "STANDARD",
        "description": "Keep STANDARD FTR rows, but apply Patch C short-price low-model blocking plus Patch D base draw-gap weak-power blocking.",
    },
    {
        "variant_name": "FTR_STANDARD_PATCH_C_PLUS_PATCH_D_NO_DRAWWARN_ONLY",
        "market": "ftr",
        "tier": "STANDARD",
        "description": "Keep STANDARD FTR rows, but apply Patch C short-price low-model blocking plus Patch D NO_DRAWWARN-only base draw-gap weak-power blocking.",
    },
    {
        "variant_name": "FTR_STANDARD_PATCH_CA_LITE_COMBO",
        "market": "ftr",
        "tier": "STANDARD",
        "description": "Keep STANDARD FTR rows, but demote short-price low-model rows and OBSERVE_RESCUE rows with model_p_for_bookie < 0.41.",
    },
    {
        "variant_name": "FTR_STANDARD_COMBINED_PATCH_ABC",
        "market": "ftr",
        "tier": "STANDARD",
        "description": "Combined STANDARD patch: tighter OBSERVE_RESCUE, tighter CS_PROMOTED, and short-price low-model block.",
    },
    {
        "variant_name": "FTR_ELITE_BASELINE",
        "market": "ftr",
        "tier": "ELITE",
        "description": "Baseline ELITE FTR rows with no extra variant filtering.",
    },
    {
        "variant_name": "FTR_ELITE_MIN_MODELP_GE_0P58",
        "market": "ftr",
        "tier": "ELITE",
        "description": "Keep ELITE FTR rows only when model_p_for_bookie >= 0.58.",
    },
    {
        "variant_name": "FTR_ELITE_MIN_MODELP_GE_0P60",
        "market": "ftr",
        "tier": "ELITE",
        "description": "Keep ELITE FTR rows only when model_p_for_bookie >= 0.60.",
    },
    {
        "variant_name": "FTR_ELITE_SHORT_HOME_DRAW_TRAP_BLOCK",
        "market": "ftr",
        "tier": "ELITE",
        "description": "Keep ELITE FTR rows, but block short HOME favourites with odds <= 1.65, model_p < 0.60, and draw-warning markers.",
    },
    {
        "variant_name": "FTR_ELITE_COMBINED_PATCH_DE",
        "market": "ftr",
        "tier": "ELITE",
        "description": "Combined ELITE patch: model_p >= 0.58 plus short HOME draw-trap block.",
    },
]
EXPERIMENTAL_TEST_FAMILY_SPECS = [
    {
        "family_name": "POIS_HOME_GE1",
        "description": "Experimental scaffold for Poisson home goals >= 1 once p_home_ge1 is present in scored deploy outputs.",
        "required_cols": ["p_home_ge1"],
        "probability_col": "p_home_ge1",
        "threshold": 0.50,
    },
    {
        "family_name": "POIS_HOME_GE2",
        "description": "Experimental scaffold for Poisson home goals >= 2 once p_home_ge2 is present in scored deploy outputs.",
        "required_cols": ["p_home_ge2"],
        "probability_col": "p_home_ge2",
        "threshold": 0.40,
    },
    {
        "family_name": "POIS_HOME_GE3",
        "description": "Experimental scaffold for Poisson home goals >= 3 once p_home_ge3 is present in scored deploy outputs.",
        "required_cols": ["p_home_ge3"],
        "probability_col": "p_home_ge3",
        "threshold": 0.22,
    },
    {
        "family_name": "POIS_AWAY_GE1",
        "description": "Experimental scaffold for Poisson away goals >= 1 once p_away_ge1 is present in scored deploy outputs.",
        "required_cols": ["p_away_ge1"],
        "probability_col": "p_away_ge1",
        "threshold": 0.42,
    },
    {
        "family_name": "POIS_AWAY_GE2",
        "description": "Experimental scaffold for Poisson away goals >= 2 once p_away_ge2 is present in scored deploy outputs.",
        "required_cols": ["p_away_ge2"],
        "probability_col": "p_away_ge2",
        "threshold": 0.24,
    },
    {
        "family_name": "POIS_HOME_GE4",
        "description": "Experimental scaffold for Poisson home goals >= 4 once p_home_ge4 is present in scored deploy outputs.",
        "required_cols": ["p_home_ge4"],
        "probability_col": "p_home_ge4",
        "threshold": 0.12,
    },
]

FTR_DRAW_AUDIT_GROUP_COLS = [
    "league",
    "standard_reporting_bucket",
    "signal_class",
    "model_probability_band",
    "implied_odds_band",
    "draw_warning_flag",
    "draw_warning_count",
    "draw_risk_band",
    "is_short_price",
    "is_low_model_probability",
    "is_draw_outcome",
]
FTR_STANDARD_RESIDUAL_AUDIT_GROUP_COLS = [
    "standard_reporting_bucket",
    "actual_ftr",
    "selection",
    "bookie_od_band",
    "model_probability_band",
    "power_diff_band",
    "ppg_diff_pre_band",
    "draw_warning_token_bucket",
]


def safe_upper_text_series(df: pd.DataFrame, col: str) -> pd.Series:
    return safe_series_str(df, col).str.upper().str.strip()


def safe_float_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df.get(col, pd.Series(np.nan, index=df.index)), errors="coerce")


def parse_reason_codes_to_upper(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.upper().str.strip()


def build_draw_warning_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    reason_codes = parse_reason_codes_to_upper(out.get("context_reason_codes", pd.Series("", index=out.index)))
    bucket = safe_upper_text_series(out, "standard_reporting_bucket")

    warning_counts = pd.Series(0, index=out.index, dtype="int64")
    for token in DRAW_WARNING_TOKENS:
        warning_counts = warning_counts + reason_codes.str.contains(token, regex=False).astype(int)
        warning_counts = warning_counts + bucket.str.contains(token, regex=False).astype(int)

    out["draw_warning_count"] = warning_counts
    out["draw_warning_flag"] = np.where(out["draw_warning_count"] > 0, "YES", "NO")
    out["draw_risk_band"] = np.where(
        out["draw_warning_count"] >= 3,
        "HIGH",
        np.where(out["draw_warning_count"] >= 1, "MEDIUM", "LOW")
    )
    return out


def enrich_ftr_variant_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["model_p_for_bookie_num"] = safe_float_series(out, "model_p_for_bookie")
    out["bookie_od_num"] = safe_float_series(out, "bookie_od")
    out["power_diff_num"] = safe_float_series(out, "power_diff")
    out["ppg_diff_pre_num"] = safe_float_series(out, "ppg_diff_pre")
    out["signal_class"] = safe_upper_text_series(out, "model_top_pick")
    out["standard_reporting_bucket"] = safe_upper_text_series(out, "standard_reporting_bucket")
    out["actual_ftr"] = safe_upper_text_series(out, "actual_ftr")
    out = build_draw_warning_features(out)
    out["model_probability_band"] = build_model_probability_band(out["model_p_for_bookie_num"])
    out["implied_odds_band"] = build_implied_odds_band(out["bookie_od_num"])
    out["bookie_od_band"] = out["implied_odds_band"]
    out["power_diff_band"] = build_power_diff_band(out["power_diff_num"])
    out["ppg_diff_pre_band"] = build_ppg_diff_band(out["ppg_diff_pre_num"])
    out["is_short_price"] = np.where(out["bookie_od_num"] <= 1.60, "YES", "NO")
    out["is_low_model_probability"] = np.where(out["model_p_for_bookie_num"] < 0.41, "YES", "NO")
    out["is_draw_outcome"] = np.where(out["actual_ftr"] == "DRAW", "YES", "NO")
    out["draw_warning_token_bucket"] = extract_draw_warning_token_bucket(out.get("context_reason_codes", pd.Series("", index=out.index)))

    selection = safe_upper_text_series(out, "selection")
    confidence_home = safe_float_series(out, "confidence_home")
    confidence_draw = safe_float_series(out, "confidence_draw")
    confidence_away = safe_float_series(out, "confidence_away")
    out["selected_minus_draw"] = np.where(
        selection.eq("HOME"),
        confidence_home - confidence_draw,
        np.where(
            selection.eq("AWAY"),
            confidence_away - confidence_draw,
            np.nan,
        ),
    )
    return out


def apply_ftr_variant_rules(df: pd.DataFrame, variant_name: str) -> pd.DataFrame:
    out = df.copy()
    out["variant_name"] = variant_name
    out["variant_keep"] = True
    out["variant_reject_reason"] = ""

    bucket = safe_upper_text_series(out, "standard_reporting_bucket")
    signal_class = safe_upper_text_series(out, "signal_class")
    model_p = safe_float_series(out, "model_p_for_bookie_num")
    bookie_od = safe_float_series(out, "bookie_od_num")
    power_diff_abs = safe_float_series(out, "power_diff_num").abs()
    ppg_diff_abs = safe_float_series(out, "ppg_diff_pre_num").abs()
    selected_minus_draw = safe_float_series(out, "selected_minus_draw")
    draw_warn = safe_upper_text_series(out, "draw_warning_flag").eq("YES")
    patch_c_bucket_ok = bucket.isin(["STANDARD_FTR_BASE", "STANDARD_FTR_OBSERVE_RESCUE"])

    def reject(mask: pd.Series, reason: str) -> None:
        mask = mask.fillna(False)
        out.loc[mask, "variant_keep"] = False
        out.loc[mask & out["variant_reject_reason"].eq(""), "variant_reject_reason"] = reason

    if variant_name == "FTR_STANDARD_BASELINE":
        return out

    if variant_name == "FTR_STANDARD_OBSERVE_RESCUE_MODELP_GE_0P41":
        reject(bucket.eq("STANDARD_FTR_OBSERVE_RESCUE") & model_p.lt(0.41), "OBSERVE_RESCUE_MODEL_P_LT_0P41")
        return out

    if variant_name == "FTR_STANDARD_PATCH_A_LITE_ONLY":
        reject(bucket.eq("STANDARD_FTR_OBSERVE_RESCUE") & model_p.lt(0.41), "OBSERVE_RESCUE_MODEL_P_LT_0P41")
        return out

    if variant_name == "FTR_STANDARD_OBSERVE_RESCUE_STRICT_TRIAD":
        reject(
            bucket.eq("STANDARD_FTR_OBSERVE_RESCUE") & (
                model_p.lt(0.41) | power_diff_abs.lt(18.0) | ppg_diff_abs.lt(0.60)
            ),
            "OBSERVE_RESCUE_TRIAD_FAIL",
        )
        return out

    if variant_name == "FTR_STANDARD_CS_PROMOTED_MODELP_GE_0P41_NO_DRAWWARN":
        reject(
            bucket.eq("STANDARD_FTR_CS_PROMOTED_ALIGNED") & (model_p.lt(0.41) | draw_warn),
            "CS_PROMOTED_LOW_MODEL_OR_DRAWWARN",
        )
        return out

    if variant_name == "FTR_STANDARD_SHORTPRICE_MODEL_LT_0P41_BLOCK":
        reject(patch_c_bucket_ok & bookie_od.le(1.60) & model_p.lt(0.41), "SHORTPRICE_LOW_MODEL_BLOCK")
        return out

    if variant_name == "FTR_STANDARD_PATCH_C_ONLY":
        reject(patch_c_bucket_ok & bookie_od.le(1.60) & model_p.lt(0.41), "SHORTPRICE_LOW_MODEL_BLOCK")
        return out

    if variant_name == "FTR_STANDARD_BASE_DRAW_GAP_LE_0P15_AND_WEAK_POWER_BLOCK":
        reject(
            bucket.eq("STANDARD_FTR_BASE")
            & selected_minus_draw.notna()
            & selected_minus_draw.le(0.15)
            & power_diff_abs.notna()
            & power_diff_abs.le(18.0),
            "BASE_DRAW_GAP_LE_0P15_AND_WEAK_POWER_BLOCK",
        )
        return out

    if variant_name == "FTR_STANDARD_BASE_NO_DRAWWARN_DRAW_GAP_LE_0P15_AND_WEAK_POWER_BLOCK":
        reject(
            bucket.eq("STANDARD_FTR_BASE")
            & safe_upper_text_series(out, "draw_warning_token_bucket").eq("NO_DRAWWARN")
            & selected_minus_draw.notna()
            & selected_minus_draw.le(0.15)
            & power_diff_abs.notna()
            & power_diff_abs.le(18.0),
            "BASE_NO_DRAWWARN_DRAW_GAP_LE_0P15_AND_WEAK_POWER_BLOCK",
        )
        return out

    if variant_name == "FTR_STANDARD_BASE_DEMOTED_DRAW_GAP_LE_0P15_AND_WEAK_POWER_BLOCK":
        reject(
            bucket.eq("STANDARD_FTR_BASE")
            & safe_upper_text_series(out, "draw_warning_token_bucket").eq("DEMOTED")
            & selected_minus_draw.notna()
            & selected_minus_draw.le(0.15)
            & power_diff_abs.notna()
            & power_diff_abs.le(18.0),
            "BASE_DEMOTED_DRAW_GAP_LE_0P15_AND_WEAK_POWER_BLOCK",
        )
        return out

    if variant_name == "FTR_STANDARD_PATCH_C_PLUS_PATCH_D_COMBO":
        reject(
            patch_c_bucket_ok & bookie_od.le(1.60) & model_p.lt(0.41),
            "SHORTPRICE_LOW_MODEL_BLOCK",
        )
        reject(
            bucket.eq("STANDARD_FTR_BASE")
            & selected_minus_draw.notna()
            & selected_minus_draw.le(0.15)
            & power_diff_abs.notna()
            & power_diff_abs.le(18.0),
            "BASE_DRAW_GAP_LE_0P15_AND_WEAK_POWER_BLOCK",
        )
        return out

    if variant_name == "FTR_STANDARD_PATCH_C_PLUS_PATCH_D_NO_DRAWWARN_ONLY":
        reject(
            patch_c_bucket_ok & bookie_od.le(1.60) & model_p.lt(0.41),
            "SHORTPRICE_LOW_MODEL_BLOCK",
        )
        reject(
            bucket.eq("STANDARD_FTR_BASE")
            & safe_upper_text_series(out, "draw_warning_token_bucket").eq("NO_DRAWWARN")
            & selected_minus_draw.notna()
            & selected_minus_draw.le(0.15)
            & power_diff_abs.notna()
            & power_diff_abs.le(18.0),
            "BASE_NO_DRAWWARN_DRAW_GAP_LE_0P15_AND_WEAK_POWER_BLOCK",
        )
        return out

    if variant_name == "FTR_STANDARD_PATCH_CA_LITE_COMBO":
        reject(bucket.eq("STANDARD_FTR_OBSERVE_RESCUE") & model_p.lt(0.41), "OBSERVE_RESCUE_MODEL_P_LT_0P41")
        reject(patch_c_bucket_ok & bookie_od.le(1.60) & model_p.lt(0.41), "SHORTPRICE_LOW_MODEL_BLOCK")
        return out

    if variant_name == "FTR_STANDARD_COMBINED_PATCH_ABC":
        reject(
            bucket.eq("STANDARD_FTR_OBSERVE_RESCUE") & (
                model_p.lt(0.41) | power_diff_abs.lt(18.0) | ppg_diff_abs.lt(0.60)
            ),
            "OBSERVE_RESCUE_TRIAD_FAIL",
        )
        reject(
            bucket.eq("STANDARD_FTR_CS_PROMOTED_ALIGNED") & (model_p.lt(0.41) | draw_warn),
            "CS_PROMOTED_LOW_MODEL_OR_DRAWWARN",
        )
        reject(patch_c_bucket_ok & bookie_od.le(1.60) & model_p.lt(0.41), "SHORTPRICE_LOW_MODEL_BLOCK")
        return out

    if variant_name == "FTR_ELITE_BASELINE":
        return out

    if variant_name == "FTR_ELITE_MIN_MODELP_GE_0P58":
        reject(model_p.lt(0.58), "ELITE_MODEL_P_LT_0P58")
        return out

    if variant_name == "FTR_ELITE_MIN_MODELP_GE_0P60":
        reject(model_p.lt(0.60), "ELITE_MODEL_P_LT_0P60")
        return out

    if variant_name == "FTR_ELITE_SHORT_HOME_DRAW_TRAP_BLOCK":
        reject(signal_class.eq("HOME") & bookie_od.le(1.65) & model_p.lt(0.60) & draw_warn, "ELITE_SHORT_HOME_DRAWTRAP")
        return out

    if variant_name == "FTR_ELITE_COMBINED_PATCH_DE":
        reject(model_p.lt(0.58), "ELITE_MODEL_P_LT_0P58")
        reject(signal_class.eq("HOME") & bookie_od.le(1.65) & model_p.lt(0.60) & draw_warn, "ELITE_SHORT_HOME_DRAWTRAP")
        return out

    raise ValueError(f"Unknown variant_name: {variant_name}")


def make_variant_summary_row(df: pd.DataFrame, variant_name: str, description: str, market_name: str, tier_name: str) -> dict[str, object]:
    kept = df.loc[df["variant_keep"]].copy()
    rejected = df.loc[~df["variant_keep"]].copy()
    hit_col = SUPPORTED_MARKETS[market_name]["hit_col"]
    graded_mask = kept[hit_col].notna() if hit_col in kept.columns else pd.Series(False, index=kept.index)
    graded = kept.loc[graded_mask].copy()
    wins = float(pd.to_numeric(graded.get(hit_col, np.nan), errors="coerce").fillna(0).sum()) if not graded.empty else 0.0
    losses = int(len(graded) - wins) if not graded.empty else 0
    profit, stake_n = calc_level_stake_profit(kept, hit_col)
    return {
        "variant_name": variant_name,
        "description": description,
        "market": market_name,
        "tier": tier_name,
        "input_rows": int(len(df)),
        "kept_rows": int(len(kept)),
        "rejected_rows": int(len(rejected)),
        "graded_rows": int(len(graded)),
        "wins": wins,
        "losses": losses,
        "hit_rate": float(wins / len(graded)) if len(graded) else np.nan,
        "roi_level_stake": float(profit / stake_n) if stake_n else np.nan,
        "level_stake_profit": profit,
        "avg_bookie_od": float(pd.to_numeric(kept.get("bookie_od", np.nan), errors="coerce").mean()) if len(kept) else np.nan,
        "avg_model_p_for_bookie": float(pd.to_numeric(kept.get("model_p_for_bookie", np.nan), errors="coerce").mean()) if len(kept) else np.nan,
        "draw_rate_kept": float((safe_upper_text_series(kept, "actual_ftr") == "DRAW").mean()) if len(kept) else np.nan,
        "draw_rate_rejected": float((safe_upper_text_series(rejected, "actual_ftr") == "DRAW").mean()) if len(rejected) else np.nan,
    }


def build_variant_grouped_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            rows=("variant_keep", "size"),
            kept_rows=("variant_keep", "sum"),
            rejected_rows=("variant_keep", lambda s: int((~s.astype(bool)).sum())),
            graded_rows=("ftr_hit", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            wins=("ftr_hit", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
            avg_bookie_od=("bookie_od", lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
            avg_model_p_for_bookie=("model_p_for_bookie", lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
            draw_rate=("actual_ftr", lambda s: float((s.astype("string").fillna("").str.upper() == "DRAW").mean())),
        )
        .reset_index()
    )
    grouped["losses"] = grouped["graded_rows"] - grouped["wins"]
    grouped["hit_rate"] = grouped["wins"] / grouped["graded_rows"].replace(0, np.nan)
    return grouped.sort_values(["rows", "graded_rows"], ascending=[False, False]).reset_index(drop=True)
LOSS_AUDIT_SPECS = [
    ("FTR_STANDARD", "ftr", "STANDARD"),
    ("FTR_ELITE", "ftr", "ELITE"),
    ("OU25_STANDARD", "ou25", "STANDARD"),
]
BTTS_SIGNAL_AUDIT_SPECS = [
    ("STRONG_YES", "BTTS_YES__STRONG_YES.csv"),
    ("VERY_STRONG_YES", "BTTS_YES__VERY_STRONG_YES.csv"),
    ("VERY_STRONG_YES__BRAZIL_SERIE_A", "BTTS_YES__VERY_STRONG_YES__BRAZIL_SERIE_A.csv"),
]

def build_master_loss_audits(base_outdir: Path, windows: list[WindowSpec]) -> None:
    master_dir = base_outdir / "_MASTER"
    audit_dir = master_dir / "LOSS_AUDITS"
    ensure_dir(audit_dir)

    grouped_cols = [
        "league",
        "standard_reporting_bucket",
        "signal_class",
        "model_probability_band",
        "implied_odds_band",
    ]

    for label, market_name, tier_name in LOSS_AUDIT_SPECS:
        frames: list[pd.DataFrame] = []

        for window in windows:
            scored_path = (
                base_outdir
                / window.window_id
                / "03_scored"
                / f"DEPLOY_COMBINED_SCORED_{window.date_from}_to_{window.date_to}.csv"
            )
            if not scored_path.exists():
                continue

            df = pd.read_csv(scored_path, low_memory=False)
            sub = market_base_filter(df, market_name)
            if sub.empty or "source_tier_file" not in sub.columns:
                continue

            sub = sub.loc[safe_series_str(sub, "source_tier_file").str.upper().eq(tier_name)].copy()
            if sub.empty:
                continue

            hit_col = SUPPORTED_MARKETS[market_name]["hit_col"]
            if hit_col not in sub.columns:
                continue

            losses = sub.loc[pd.to_numeric(sub[hit_col], errors="coerce") == 0].copy()
            if losses.empty:
                continue

            losses["window_id"] = window.window_id
            if market_name == "ftr":
                losses["signal_class"] = safe_series_str(losses, "model_top_pick")
            else:
                losses["signal_class"] = safe_series_str(losses, "signal_over25")

            losses["model_probability_band"] = build_model_probability_band(losses.get("model_p_for_bookie", np.nan))
            losses["implied_odds_band"] = build_implied_odds_band(losses.get("bookie_od", np.nan))
            frames.append(losses)

        if not frames:
            continue

        combined = pd.concat(frames, ignore_index=True, sort=False)
        combined = combined.sort_values(["window_id", "league", "match_date", "home_team_name", "away_team_name"], na_position="last")
        combined.to_csv(audit_dir / f"{label}__LOSSES_COMBINED.csv", index=False)

        for group_col in grouped_cols:
            summary = make_grouped_count_summary(combined, group_col)
            summary.to_csv(audit_dir / f"{label}__LOSSES_BY_{group_col.upper()}.csv", index=False)

def build_btts_signal_audits(base_outdir: Path, windows: list[WindowSpec]) -> None:
    master_dir = base_outdir / "_MASTER"
    audit_dir = master_dir / "BTTS_SIGNAL_AUDITS"
    ensure_dir(audit_dir)

    frames: list[pd.DataFrame] = []
    for window in windows:
        scored_path = (
            base_outdir
            / window.window_id
            / "03_scored"
            / f"DEPLOY_COMBINED_SCORED_{window.date_from}_to_{window.date_to}.csv"
        )
        if not scored_path.exists():
            continue

        df = pd.read_csv(scored_path, low_memory=False)
        bt = market_base_filter(df, "btts")
        if bt.empty:
            continue

        bt["__window"] = window.window_id
        bt["signal_btts_runtime"] = safe_series_str(bt, "signal_btts_runtime").str.upper().str.strip()
        bt["league"] = safe_series_str(bt, "league").str.strip()
        frames.append(bt)

    if not frames:
        log("[btts-audit] no scored BTTS files found.")
        return

    combined = pd.concat(frames, ignore_index=True, sort=False)

    def _summarize(sub: pd.DataFrame) -> dict[str, float | int]:
        return summarize_slice(sub, "btts_yes_hit")

    def _write_group(label: str, sub: pd.DataFrame) -> None:
        sub = sub.copy()
        sub.to_csv(audit_dir / f"{label}__COMBINED.csv", index=False)
        sub.loc[pd.to_numeric(sub.get("btts_yes_hit", np.nan), errors="coerce") == 1].to_csv(
            audit_dir / f"{label}__WINS.csv", index=False
        )
        sub.loc[pd.to_numeric(sub.get("btts_yes_hit", np.nan), errors="coerce") == 0].to_csv(
            audit_dir / f"{label}__LOSSES.csv", index=False
        )

        losses = sub.loc[pd.to_numeric(sub.get("btts_yes_hit", np.nan), errors="coerce") == 0].copy()
        focus_cols = [
            c for c in [
                "__window", "league", "match_date", "home_team_name", "away_team_name",
                "signal_btts", "signal_btts_runtime", "model_p_for_bookie", "bookie_od",
                "home_ge2_confidence", "away_ge2_confidence",
                "p_home_fts", "p_away_fts",
                "cs_home", "cs_away", "cs_max",
                "p00_est", "exp_goals_sum",
                "cs00_top3_mass", "btts_top3_mass", "top3_any_00",
                "fts_sum", "cs_sum",
                "btts_yes_cs_shadow_block",
                "btts_yes_double_blank_shadow_block",
                "btts_yes_hit",
            ]
            if c in losses.columns
        ]
        losses[focus_cols].to_csv(audit_dir / f"{label}__LOSSES_FOCUS.csv", index=False)

        by_league = []
        if "league" in sub.columns:
            for league_value, grp in sub.groupby("league", dropna=False):
                by_league.append({
                    "league": league_value,
                    **_summarize(grp),
                })
        pd.DataFrame(by_league).sort_values(["hit_rate", "rows"], ascending=[False, False]).to_csv(
            audit_dir / f"{label}__BY_LEAGUE.csv", index=False
        )

        by_window = []
        if "__window" in sub.columns:
            for window_value, grp in sub.groupby("__window", dropna=False):
                by_window.append({
                    "__window": window_value,
                    **_summarize(grp),
                })
        pd.DataFrame(by_window).sort_values("__window").to_csv(
            audit_dir / f"{label}__BY_WINDOW.csv", index=False
        )

    strong_yes = combined.loc[
        combined["signal_btts_runtime"].eq("STRONG_YES")
    ].copy()

    very_strong_yes = combined.loc[
        combined["signal_btts_runtime"].eq("VERY_STRONG_YES")
    ].copy()

    very_strong_yes_brazil = combined.loc[
        combined["signal_btts_runtime"].eq("VERY_STRONG_YES")
        & combined["league"].eq("Brazil Serie A")
    ].copy()

    very_strong_yes_non_brazil = combined.loc[
        combined["signal_btts_runtime"].eq("VERY_STRONG_YES")
        & (~combined["league"].eq("Brazil Serie A"))
    ].copy()

    summary_rows = []
    for label, sub in [
        ("STRONG_YES", strong_yes),
        ("VERY_STRONG_YES", very_strong_yes),
        ("VERY_STRONG_YES__BRAZIL_SERIE_A", very_strong_yes_brazil),
        ("VERY_STRONG_YES__NON_BRAZIL", very_strong_yes_non_brazil),
    ]:
        if sub.empty:
            continue
        _write_group(label, sub)
        summary_rows.append({
            "signal_group": label,
            **_summarize(sub),
        })

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            audit_dir / "BTTS_SIGNAL_GROUPS__SUMMARY.csv",
            index=False,
        )


# New: build_btts_live_product_audits
def build_btts_live_product_audits(base_outdir: Path, windows: list[WindowSpec]) -> None:
    master_dir = base_outdir / "_MASTER"
    audit_dir = master_dir / "BTTS_LIVE_PRODUCT_AUDITS"
    ensure_dir(audit_dir)

    frames: list[pd.DataFrame] = []
    for window in windows:
        scored_path = (
            base_outdir
            / window.window_id
            / "03_scored"
            / f"DEPLOY_COMBINED_SCORED_{window.date_from}_to_{window.date_to}.csv"
        )
        if not scored_path.exists():
            continue

        df = pd.read_csv(scored_path, low_memory=False)
        bt = market_base_filter(df, "btts")
        if bt.empty:
            continue

        bt["__window"] = window.window_id
        bt["signal_btts_runtime"] = safe_series_str(bt, "signal_btts_runtime").str.upper().str.strip()
        bt["league"] = safe_series_str(bt, "league").str.strip()
        frames.append(bt)

    if not frames:
        log("[btts-live-audit] no scored BTTS files found.")
        return

    combined = pd.concat(frames, ignore_index=True, sort=False)

    strong_yes = combined.loc[
        combined["signal_btts_runtime"].eq("STRONG_YES")
    ].copy()

    very_strong_yes_non_brazil = combined.loc[
        combined["signal_btts_runtime"].eq("VERY_STRONG_YES")
        & (~combined["league"].eq("Brazil Serie A"))
    ].copy()

    live_combined = pd.concat(
        [strong_yes, very_strong_yes_non_brazil],
        ignore_index=True,
        sort=False,
    )

    def _summarize(sub: pd.DataFrame) -> dict[str, float | int]:
        return summarize_slice(sub, "btts_yes_hit")

    def _write_group(label: str, sub: pd.DataFrame) -> None:
        sub = sub.copy()
        sub.to_csv(audit_dir / f"{label}__COMBINED.csv", index=False)
        sub.loc[pd.to_numeric(sub.get("btts_yes_hit", np.nan), errors="coerce") == 1].to_csv(
            audit_dir / f"{label}__WINS.csv", index=False
        )
        sub.loc[pd.to_numeric(sub.get("btts_yes_hit", np.nan), errors="coerce") == 0].to_csv(
            audit_dir / f"{label}__LOSSES.csv", index=False
        )

        # Inserted: Write BTTS_YES__LOSSES_FOCUS.csv
        losses = sub.loc[pd.to_numeric(sub.get("btts_yes_hit", np.nan), errors="coerce") == 0].copy()
        focus_cols = [
            c for c in [
                "__window", "league", "match_date", "home_team_name", "away_team_name",
                "signal_btts", "signal_btts_runtime", "model_p_for_bookie", "bookie_od",
                "home_ge2_confidence", "away_ge2_confidence",
                "p_home_fts", "p_away_fts",
                "cs_home", "cs_away", "cs_max",
                "p00_est", "exp_goals_sum",
                "cs00_top3_mass", "btts_top3_mass", "top3_any_00",
                "fts_sum", "cs_sum",
                "btts_yes_cs_shadow_block",
                "btts_yes_double_blank_shadow_block",
                "btts_yes_hit",
            ]
            if c in losses.columns
        ]
        losses[focus_cols].to_csv(
            audit_dir / f"{label}__LOSSES_FOCUS.csv",
            index=False,
        )

        by_league = []
        if "league" in sub.columns:
            for league_value, grp in sub.groupby("league", dropna=False):
                by_league.append({
                    "league": league_value,
                    **_summarize(grp),
                })
        pd.DataFrame(by_league).sort_values(["hit_rate", "rows"], ascending=[False, False]).to_csv(
            audit_dir / f"{label}__BY_LEAGUE.csv", index=False
        )

        by_window = []
        if "__window" in sub.columns:
            for window_value, grp in sub.groupby("__window", dropna=False):
                by_window.append({
                    "__window": window_value,
                    **_summarize(grp),
                })
        pd.DataFrame(by_window).sort_values("__window").to_csv(
            audit_dir / f"{label}__BY_WINDOW.csv", index=False
        )

    summary_rows = []
    for label, sub in [
        ("BTTS_YES__LIVE_CORE", strong_yes),
        ("BTTS_YES__LIVE_AGGRESSIVE", very_strong_yes_non_brazil),
        ("BTTS_YES__LIVE_COMBINED", live_combined),
    ]:
        if sub.empty:
            continue
        _write_group(label, sub)
        summary_rows.append({
            "signal_group": label,
            **_summarize(sub),
        })

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            audit_dir / "BTTS_LIVE_PRODUCT__SUMMARY.csv",
            index=False,
        )

def build_ftr_variant_tests(base_outdir: Path, windows: list[WindowSpec]) -> None:
    master_dir = base_outdir / "_MASTER"
    variant_dir = master_dir / "VARIANT_TESTS"
    ensure_dir(variant_dir)

    frames: list[pd.DataFrame] = []
    for window in windows:
        scored_path = (
            base_outdir
            / window.window_id
            / "03_scored"
            / f"DEPLOY_COMBINED_SCORED_{window.date_from}_to_{window.date_to}.csv"
        )
        if not scored_path.exists():
            continue
        df = pd.read_csv(scored_path, low_memory=False)
        ftr_df = market_base_filter(df, "ftr")
        if ftr_df.empty or "source_tier_file" not in ftr_df.columns:
            continue
        ftr_df["window_id"] = window.window_id
        ftr_df = enrich_ftr_variant_features(ftr_df)
        frames.append(ftr_df)

    if not frames:
        log("[variant] no scored FTR files found for variant testing.")
        return

    combined_ftr = pd.concat(frames, ignore_index=True, sort=False)
    combined_ftr.to_csv(variant_dir / "FTR__BASE_COMBINED.csv", index=False)

    summary_rows: list[dict[str, object]] = []

    for spec in VARIANT_TEST_SPECS:
        variant_name = spec["variant_name"]
        market_name = spec["market"]
        tier_name = spec["tier"]
        description = spec["description"]

        sub = combined_ftr.loc[safe_upper_text_series(combined_ftr, "source_tier_file").eq(tier_name)].copy()
        if sub.empty:
            continue

        tested = apply_ftr_variant_rules(sub, variant_name)
        tested.to_csv(variant_dir / f"{variant_name}__ROW_LEVEL.csv", index=False)

        kept = tested.loc[tested["variant_keep"]].copy()
        rejected = tested.loc[~tested["variant_keep"]].copy()

        summary_rows.append(make_variant_summary_row(tested, variant_name, description, market_name, tier_name))

        if not rejected.empty:
            reject_counts = make_grouped_count_summary(rejected, "variant_reject_reason")
            reject_counts.to_csv(variant_dir / f"{variant_name}__REJECT_REASONS.csv", index=False)
        else:
            pd.DataFrame(columns=["variant_reject_reason", "rows"]).to_csv(
                variant_dir / f"{variant_name}__REJECT_REASONS.csv", index=False
            )

        kept_losses = kept.loc[pd.to_numeric(kept.get("ftr_hit", np.nan), errors="coerce") == 0].copy()
        kept_wins = kept.loc[pd.to_numeric(kept.get("ftr_hit", np.nan), errors="coerce") == 1].copy()
        kept_losses.to_csv(variant_dir / f"{variant_name}__KEPT_LOSSES.csv", index=False)
        kept_wins.to_csv(variant_dir / f"{variant_name}__KEPT_WINS.csv", index=False)

        grouped = build_variant_grouped_summary(
            tested,
            ["variant_name", "source_tier_file", "standard_reporting_bucket", "signal_class", "model_probability_band", "implied_odds_band"],
        )
        grouped.to_csv(variant_dir / f"{variant_name}__GROUPED_SUMMARY.csv", index=False)

        kept_loss_draws = kept_losses.loc[safe_upper_text_series(kept_losses, "actual_ftr").eq("DRAW")].copy()
        kept_loss_draws.to_csv(variant_dir / f"{variant_name}__KEPT_LOSS_DRAWS.csv", index=False)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values(["tier", "variant_name"]).reset_index(drop=True)
        summary_df.to_csv(variant_dir / "FTR__VARIANT_SUMMARY.csv", index=False)


def build_ftr_draw_risk_audits(base_outdir: Path, windows: list[WindowSpec]) -> None:
    master_dir = base_outdir / "_MASTER"
    audit_dir = master_dir / "DRAW_RISK_AUDITS"
    ensure_dir(audit_dir)

    frames: list[pd.DataFrame] = []
    for window in windows:
        scored_path = (
            base_outdir
            / window.window_id
            / "03_scored"
            / f"DEPLOY_COMBINED_SCORED_{window.date_from}_to_{window.date_to}.csv"
        )
        if not scored_path.exists():
            continue
        df = pd.read_csv(scored_path, low_memory=False)
        ftr_df = market_base_filter(df, "ftr")
        if ftr_df.empty or "source_tier_file" not in ftr_df.columns:
            continue
        ftr_df = ftr_df.loc[safe_upper_text_series(ftr_df, "source_tier_file").isin(["STANDARD", "ELITE"])].copy()
        if ftr_df.empty:
            continue
        ftr_df["window_id"] = window.window_id
        ftr_df = enrich_ftr_variant_features(ftr_df)
        frames.append(ftr_df)

    if not frames:
        log("[draw-audit] no scored FTR files found for draw-risk auditing.")
        return

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(audit_dir / "FTR__STANDARD_AND_ELITE__COMBINED.csv", index=False)

    standard = combined.loc[safe_upper_text_series(combined, "source_tier_file").eq("STANDARD")].copy()
    elite = combined.loc[safe_upper_text_series(combined, "source_tier_file").eq("ELITE")].copy()

    for label, sub in [("FTR_STANDARD", standard), ("FTR_ELITE", elite)]:
        if sub.empty:
            continue

        winners = sub.loc[pd.to_numeric(sub.get("ftr_hit", np.nan), errors="coerce") == 1].copy()
        losses = sub.loc[pd.to_numeric(sub.get("ftr_hit", np.nan), errors="coerce") == 0].copy()
        draw_losses = losses.loc[safe_upper_text_series(losses, "actual_ftr").eq("DRAW")].copy()
        short_low_model = sub.loc[sub["bookie_od_num"].le(1.60) & sub["model_p_for_bookie_num"].lt(0.41)].copy()

        winners.to_csv(audit_dir / f"{label}__WINNERS_COMBINED.csv", index=False)
        losses.to_csv(audit_dir / f"{label}__LOSSES_COMBINED.csv", index=False)
        draw_losses.to_csv(audit_dir / f"{label}__DRAW_LOSSES_COMBINED.csv", index=False)
        short_low_model.to_csv(audit_dir / f"{label}__SHORTPRICE_LOW_MODEL_COMBINED.csv", index=False)

        for group_col in FTR_DRAW_AUDIT_GROUP_COLS:
            make_grouped_count_summary(losses, group_col).to_csv(
                audit_dir / f"{label}__LOSSES_BY_{group_col.upper()}.csv", index=False
            )
            make_grouped_count_summary(draw_losses, group_col).to_csv(
                audit_dir / f"{label}__DRAW_LOSSES_BY_{group_col.upper()}.csv", index=False
            )
            make_grouped_count_summary(short_low_model, group_col).to_csv(
                audit_dir / f"{label}__SHORTPRICE_LOW_MODEL_BY_{group_col.upper()}.csv", index=False
            )
            make_grouped_count_summary(winners, group_col).to_csv(
                audit_dir / f"{label}__WINNERS_BY_{group_col.upper()}.csv", index=False
            )

def build_ftr_standard_residual_audits(base_outdir: Path, windows: list[WindowSpec]) -> None:
    master_dir = base_outdir / "_MASTER"
    audit_dir = master_dir / "STANDARD_RESIDUAL_AUDITS"
    ensure_dir(audit_dir)

    frames: list[pd.DataFrame] = []
    for window in windows:
        scored_path = (
            base_outdir
            / window.window_id
            / "03_scored"
            / f"DEPLOY_COMBINED_SCORED_{window.date_from}_to_{window.date_to}.csv"
        )
        if not scored_path.exists():
            continue

        df = pd.read_csv(scored_path, low_memory=False)
        ftr_df = market_base_filter(df, "ftr")
        if ftr_df.empty or "source_tier_file" not in ftr_df.columns:
            continue

        ftr_df = ftr_df.loc[safe_upper_text_series(ftr_df, "source_tier_file").eq("STANDARD")].copy()
        if ftr_df.empty:
            continue

        ftr_df["window_id"] = window.window_id
        ftr_df = enrich_ftr_variant_features(ftr_df)
        frames.append(ftr_df)

    if not frames:
        log("[standard-residual-audit] no scored STANDARD FTR files found for residual auditing.")
        return

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(audit_dir / "FTR_STANDARD__CURRENT_LIVE__COMBINED.csv", index=False)

    hit_col = SUPPORTED_MARKETS["ftr"]["hit_col"]
    winners = combined.loc[pd.to_numeric(combined.get(hit_col, np.nan), errors="coerce") == 1].copy()
    losses = combined.loc[pd.to_numeric(combined.get(hit_col, np.nan), errors="coerce") == 0].copy()

    winners.to_csv(audit_dir / "FTR_STANDARD__CURRENT_LIVE__WINS_COMBINED.csv", index=False)
    losses.to_csv(audit_dir / "FTR_STANDARD__CURRENT_LIVE__LOSSES_COMBINED.csv", index=False)

    actual_ftr_summary = make_grouped_outcome_summary(combined, "actual_ftr", hit_col)
    actual_ftr_summary.to_csv(audit_dir / "FTR_STANDARD__CURRENT_LIVE__OUTCOME_BY_ACTUAL_FTR.csv", index=False)

    selection_summary = make_grouped_outcome_summary(combined, "selection", hit_col)
    selection_summary.to_csv(audit_dir / "FTR_STANDARD__CURRENT_LIVE__OUTCOME_BY_SELECTION.csv", index=False)

    draw_rate_summary = pd.DataFrame(
        {
            "subset": ["wins", "losses", "all_standard"],
            "rows": [int(len(winners)), int(len(losses)), int(len(combined))],
            "draw_rate": [
                float((safe_upper_text_series(winners, "actual_ftr") == "DRAW").mean()) if len(winners) else np.nan,
                float((safe_upper_text_series(losses, "actual_ftr") == "DRAW").mean()) if len(losses) else np.nan,
                float((safe_upper_text_series(combined, "actual_ftr") == "DRAW").mean()) if len(combined) else np.nan,
            ],
        }
    )
    draw_rate_summary.to_csv(audit_dir / "FTR_STANDARD__CURRENT_LIVE__DRAW_RATE_COMPARISON.csv", index=False)

    for group_col in FTR_STANDARD_RESIDUAL_AUDIT_GROUP_COLS:
        make_grouped_outcome_summary(combined, group_col, hit_col).to_csv(
            audit_dir / f"FTR_STANDARD__CURRENT_LIVE__OUTCOME_BY_{group_col.upper()}.csv",
            index=False,
        )
        make_grouped_count_summary(winners, group_col).to_csv(
            audit_dir / f"FTR_STANDARD__CURRENT_LIVE__WINS_BY_{group_col.upper()}.csv",
            index=False,
        )
        make_grouped_count_summary(losses, group_col).to_csv(
            audit_dir / f"FTR_STANDARD__CURRENT_LIVE__LOSSES_BY_{group_col.upper()}.csv",
            index=False,
        )

def summarize_experimental_probability_family(df: pd.DataFrame, probability_col: str, threshold: float) -> pd.DataFrame:
    out = df.copy()
    out[probability_col] = pd.to_numeric(out.get(probability_col, np.nan), errors="coerce")
    out = out.loc[out[probability_col].notna()].copy()
    if out.empty:
        return pd.DataFrame()

    out["passes_threshold"] = out[probability_col] >= float(threshold)
    out["model_probability_band"] = build_model_probability_band(out[probability_col])
    out["implied_odds_band"] = build_implied_odds_band(out.get("bookie_od", np.nan))

    grouped = (
        out.groupby(["passes_threshold", "model_probability_band", "implied_odds_band"], dropna=False)
        .agg(
            rows=(probability_col, "size"),
            avg_probability=(probability_col, "mean"),
            avg_bookie_od=("bookie_od", lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
        )
        .reset_index()
        .sort_values(["passes_threshold", "rows", "avg_probability"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    return grouped


def build_experimental_test_family_audits(base_outdir: Path, windows: list[WindowSpec]) -> None:
    master_dir = base_outdir / "_MASTER"
    audit_dir = master_dir / "EXPERIMENTAL_TEST_FAMILIES"
    ensure_dir(audit_dir)

    frames: list[pd.DataFrame] = []
    for window in windows:
        scored_path = (
            base_outdir
            / window.window_id
            / "03_scored"
            / f"DEPLOY_COMBINED_SCORED_{window.date_from}_to_{window.date_to}.csv"
        )
        if not scored_path.exists():
            continue

        df = pd.read_csv(scored_path, low_memory=False)
        df["window_id"] = window.window_id
        frames.append(df)

    if not frames:
        log("[experimental-test-families] no scored deploy files found.")
        return

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(audit_dir / "EXPERIMENTAL__BASE_COMBINED.csv", index=False)

    availability_rows: list[dict[str, object]] = []

    for spec in EXPERIMENTAL_TEST_FAMILY_SPECS:
        family_name = str(spec["family_name"])
        description = str(spec["description"])
        required_cols = list(spec["required_cols"])
        probability_col = str(spec["probability_col"])
        threshold = float(spec["threshold"])

        missing_cols = [c for c in required_cols if c not in combined.columns]
        availability_rows.append(
            {
                "family_name": family_name,
                "description": description,
                "probability_col": probability_col,
                "threshold": threshold,
                "is_available": len(missing_cols) == 0,
                "missing_cols": "|".join(missing_cols),
            }
        )

        if missing_cols:
            continue

        sub = combined.copy()
        sub[probability_col] = pd.to_numeric(sub.get(probability_col, np.nan), errors="coerce")
        sub = sub.loc[sub[probability_col].notna()].copy()
        if sub.empty:
            continue

        sub["family_name"] = family_name
        sub["family_description"] = description
        sub["passes_threshold"] = sub[probability_col] >= threshold
        sub = sub.sort_values([probability_col, "window_id"], ascending=[False, True]).reset_index(drop=True)
        sub.to_csv(audit_dir / f"{family_name}__ROW_LEVEL.csv", index=False)

        summary_df = summarize_experimental_probability_family(sub, probability_col, threshold)
        summary_df.to_csv(audit_dir / f"{family_name}__SUMMARY.csv", index=False)

        top_cols = [
            c for c in [
                "window_id",
                "league",
                "match_date",
                "home_team_name",
                "away_team_name",
                "market",
                "selection",
                "bookie_od",
                "model_p_for_bookie",
                probability_col,
                "passes_threshold",
            ]
            if c in sub.columns
        ]
        sub.loc[:, top_cols].head(100).to_csv(audit_dir / f"{family_name}__TOP100.csv", index=False)

    pd.DataFrame(availability_rows).to_csv(
        audit_dir / "EXPERIMENTAL_TEST_FAMILY_AVAILABILITY.csv",
        index=False,
    )

# ============================================================================
# Main runner
# ============================================================================
def append_to_master_scorecard(base_outdir: Path, tier_summary: pd.DataFrame) -> None:
    master_dir = base_outdir / "_MASTER"
    ensure_dir(master_dir)
    scorecard_path = master_dir / "_ALL_WINDOWS__SCORECARD.csv"

    incoming = tier_summary.copy()
    if isinstance(incoming, pd.DataFrame) and "graded" in incoming.columns:
        graded_series = pd.to_numeric(incoming["graded"], errors="coerce").fillna(0)
    else:
        graded_series = pd.Series([0])
    if not isinstance(incoming, pd.DataFrame):
        incoming = pd.DataFrame()
    if incoming.empty:
        incoming = pd.DataFrame({"window_id": [], "market": [], "source_tier_file": [], "graded": []})
    incoming["is_graded_window"] = graded_series > 0

    if scorecard_path.exists():
        existing = pd.read_csv(scorecard_path)
        if "is_graded_window" not in existing.columns:
            if "graded" in existing.columns:
                existing["is_graded_window"] = pd.to_numeric(existing["graded"], errors="coerce").fillna(0) > 0
            else:
                existing["is_graded_window"] = False
        combined = pd.concat([existing, incoming], ignore_index=True)
        combined = combined.drop_duplicates(subset=["window_id", "market", "source_tier_file"], keep="last")
    else:
        combined = incoming.copy()

    combined.to_csv(scorecard_path, index=False)


def aggregate_btts_gate_audits(base_outdir: Path) -> None:
    master_dir = base_outdir / "_MASTER"
    ensure_dir(master_dir)
    audit_paths = [p for p in base_outdir.rglob("BTTS_GATE_AUDIT__*.csv") if "_MASTER" not in p.parts]
    if not audit_paths:
        log("[btts-gate-audit] no per-window audit files found.")
        return

    frames: list[pd.DataFrame] = []
    for path in audit_paths:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        frames.append(df)

    if not frames:
        log("[btts-gate-audit] audit files empty after load.")
        return

    all_df = pd.concat(frames, ignore_index=True, sort=False)

    missing_gate_cols = [
        "btts_yes_ge2_missing_prelabel",
        "btts_yes_fts_missing_prelabel",
        "btts_yes_cs_missing_prelabel",
        "btts_yes_ge2_missing",
        "btts_yes_fts_missing",
        "btts_yes_cs_missing",
    ]
    gate_cols = [
        "btts_yes_label_fail",
        "btts_yes_brazil_block",
        "btts_yes_model_floor_fail",
        "btts_yes_ge2_fail",
        "btts_yes_fts_fail",
        "btts_yes_csmax_fail",
        "btts_yes_double_blank_fail",
        "btts_yes_cs_shadow_flag",
        "btts_yes_double_blank_flag",
    ]

    for col in missing_gate_cols + gate_cols:
        if col not in all_df.columns:
            all_df[col] = 0
    yes_input = pd.to_numeric(
        all_df.get("btts_yes_input", pd.Series(0, index=all_df.index)),
        errors="coerce",
    ).replace(0, np.nan)
    all_df["btts_yes_ge2_missing_prelabel_rate"] = (
        pd.to_numeric(all_df.get("btts_yes_ge2_missing_prelabel", 0), errors="coerce") / yes_input
    )
    all_df["btts_yes_fts_missing_prelabel_rate"] = (
        pd.to_numeric(all_df.get("btts_yes_fts_missing_prelabel", 0), errors="coerce") / yes_input
    )
    all_df["btts_yes_cs_missing_prelabel_rate"] = (
        pd.to_numeric(all_df.get("btts_yes_cs_missing_prelabel", 0), errors="coerce") / yes_input
    )
    all_df["btts_yes_ge2_missing_rate"] = (
        pd.to_numeric(all_df.get("btts_yes_ge2_missing", 0), errors="coerce") / yes_input
    )
    all_df["btts_yes_fts_missing_rate"] = (
        pd.to_numeric(all_df.get("btts_yes_fts_missing", 0), errors="coerce") / yes_input
    )
    all_df["btts_yes_cs_missing_rate"] = (
        pd.to_numeric(all_df.get("btts_yes_cs_missing", 0), errors="coerce") / yes_input
    )
    prelabel_any = (
        pd.to_numeric(
            all_df.get("btts_yes_ge2_missing_prelabel", pd.Series(0, index=all_df.index)),
            errors="coerce",
        ).fillna(0)
        + pd.to_numeric(
            all_df.get("btts_yes_fts_missing_prelabel", pd.Series(0, index=all_df.index)),
            errors="coerce",
        ).fillna(0)
        + pd.to_numeric(
            all_df.get("btts_yes_cs_missing_prelabel", pd.Series(0, index=all_df.index)),
            errors="coerce",
        ).fillna(0)
    )
    all_df["btts_yes_missing_prelabel_any_rate"] = prelabel_any / yes_input
    rescue_any = (
        pd.to_numeric(
            all_df.get("btts_yes_ge2_missing", pd.Series(0, index=all_df.index)),
            errors="coerce",
        ).fillna(0)
        + pd.to_numeric(
            all_df.get("btts_yes_fts_missing", pd.Series(0, index=all_df.index)),
            errors="coerce",
        ).fillna(0)
        + pd.to_numeric(
            all_df.get("btts_yes_cs_missing", pd.Series(0, index=all_df.index)),
            errors="coerce",
        ).fillna(0)
    )
    all_df["btts_yes_missing_rescue_any_rate"] = rescue_any / yes_input
    all_df["btts_yes_missing_prelabel_minus_rescue_rate"] = (
        all_df["btts_yes_missing_prelabel_any_rate"] - all_df["btts_yes_missing_rescue_any_rate"]
    )

    all_path = master_dir / "BTTS_GATE_AUDIT__ALL_WINDOWS.csv"
    all_df.to_csv(all_path, index=False)
    log(f"[btts-gate-audit] wrote all-windows audit: {all_path}")

    total_yes = float(
        pd.to_numeric(
            all_df.get("btts_yes_input", pd.Series(0, index=all_df.index)),
            errors="coerce",
        ).fillna(0).sum()
    )
    total_yes = total_yes if total_yes > 0 else 1.0

    summary_rows: list[dict[str, object]] = []
    for col in missing_gate_cols + gate_cols:
        if col not in all_df.columns:
            continue
        vals = pd.to_numeric(all_df[col], errors="coerce").fillna(0)
        fail_count = float(vals.sum())
        windows_with_fail = int((vals > 0).sum())
        section = "suppressor"
        if col in missing_gate_cols:
            section = "missing_prelabel" if col.endswith("_prelabel") else "missing_rescue"
        summary_rows.append(
            {
                "section": section,
                "gate": col,
                "fail_count": int(fail_count),
                "fail_share_of_yes_input": float(fail_count / total_yes),
                "windows_with_fail": windows_with_fail,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df["section_order"] = summary_df["section"].map({
            "missing_prelabel": 0,
            "missing_rescue": 1,
            "suppressor": 2,
        }).fillna(3)
        summary_df = summary_df.sort_values(["section_order", "fail_count"], ascending=[True, False]).drop(columns=["section_order"])
    summary_path = master_dir / "BTTS_GATE_AUDIT__SUMMARY.csv"
    summary_df.to_csv(summary_path, index=False)
    log(f"[btts-gate-audit] wrote summary: {summary_path}")

    # Per-gate breakdown by league
    league_paths = [p for p in base_outdir.rglob("BTTS_GATE_AUDIT_BY_LEAGUE__*.csv") if "_MASTER" not in p.parts]
    if not league_paths:
        log("[btts-gate-audit] no per-window league audit files found.")
        return

    league_frames: list[pd.DataFrame] = []
    for path in league_paths:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        league_frames.append(df)

    if not league_frames:
        log("[btts-gate-audit] league audit files empty after load.")
        return

    league_all = pd.concat(league_frames, ignore_index=True, sort=False)
    if "btts_yes_input" in league_all.columns:
        league_all["fail_share_of_yes_input_league"] = (
            pd.to_numeric(
                league_all.get("fail_count", pd.Series(0, index=league_all.index)),
                errors="coerce",
            ).fillna(0)
            / pd.to_numeric(
                league_all.get("btts_yes_input", pd.Series(0, index=league_all.index)),
                errors="coerce",
            ).replace(0, np.nan)
        )
    league_all_path = master_dir / "BTTS_GATE_AUDIT_BY_LEAGUE__ALL_WINDOWS.csv"
    league_all.to_csv(league_all_path, index=False)
    log(f"[btts-gate-audit] wrote league breakdown: {league_all_path}")

    league_yes = (
        league_all.loc[league_all["gate"].eq("btts_yes_input")]
        .groupby("league", dropna=False)
        .agg(league_yes_input=("fail_count", "sum"))
        .reset_index()
    )
    league_summary = (
        league_all.groupby(["gate", "league"], dropna=False)
        .agg(fail_count=("fail_count", "sum"), windows=("window_id", "nunique"))
        .reset_index()
        .merge(league_yes, on="league", how="left")
    )
    league_summary["league_yes_input"] = pd.to_numeric(
        league_summary.get("league_yes_input", pd.Series(0, index=league_summary.index)),
        errors="coerce",
    ).fillna(0)
    league_summary["fail_share_of_yes_input_league"] = league_summary["fail_count"] / league_summary["league_yes_input"].replace(0, np.nan)
    league_summary["fail_share_of_yes_input"] = (
        league_summary["fail_count"] / (total_yes if total_yes > 0 else 1.0)
    )
    league_summary["section"] = league_summary["gate"].map(
        {g: ("missing_prelabel" if g.endswith("_prelabel") else "missing_rescue") for g in missing_gate_cols}
    ).fillna("suppressor")
    league_summary["section_order"] = league_summary["section"].map({
        "missing_prelabel": 0,
        "missing_rescue": 1,
        "suppressor": 2,
    }).fillna(3)
    league_summary = league_summary.sort_values(["section_order", "fail_count"], ascending=[True, False]).drop(columns=["section_order"])
    league_summary_path = master_dir / "BTTS_GATE_AUDIT_BY_LEAGUE__SUMMARY.csv"
    league_summary.to_csv(league_summary_path, index=False)
    log(f"[btts-gate-audit] wrote league summary: {league_summary_path}")


def aggregate_ou25_gate_audits(base_outdir: Path) -> None:
    master_dir = base_outdir / "_MASTER"
    ensure_dir(master_dir)
    audit_paths = [p for p in base_outdir.rglob("OU25_GATE_AUDIT__*.csv") if "_MASTER" not in p.parts]
    if not audit_paths:
        log("[ou25-gate-audit] no per-window audit files found.")
        return

    frames: list[pd.DataFrame] = []
    for path in audit_paths:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        frames.append(df)

    if not frames:
        log("[ou25-gate-audit] audit files empty after load.")
        return

    all_df = pd.concat(frames, ignore_index=True, sort=False)
    all_path = master_dir / "OU25_GATE_AUDIT__ALL_WINDOWS.csv"
    all_df.to_csv(all_path, index=False)
    log(f"[ou25-gate-audit] wrote all-windows audit: {all_path}")

    total_ou = float(
        pd.to_numeric(
            all_df.get("ou25_input", pd.Series(0, index=all_df.index)),
            errors="coerce",
        ).fillna(0).sum()
    )
    total_ou = total_ou if total_ou > 0 else 1.0

    missing_gate_cols: list[str] = [
        "ou25_over_model_p_missing",
        "ou25_over_struct_inputs_missing",
        "ou25_under_model_p_missing",
        "ou25_under_struct_inputs_missing",
        "ou25_under_low_goal_inputs_missing",
    ]
    gate_cols = [
        "ou25_over_signal_fail",
        "ou25_over_model_floor_fail",
        "ou25_over_struct_combo_fail",
        "ou25_over_top3_fail",
        "ou25_over_one_sided_veto",
        "ou25_under_model_floor_fail",
        "ou25_under_struct_fail",
        "ou25_under_low_goal_fail",
    ]

    summary_rows: list[dict[str, object]] = []
    for col in missing_gate_cols + gate_cols:
        if col not in all_df.columns:
            continue
        vals = pd.to_numeric(all_df[col], errors="coerce").fillna(0)
        fail_count = float(vals.sum())
        windows_with_fail = int((vals > 0).sum())
        summary_rows.append(
            {
                "section": "missing_rate" if col in missing_gate_cols else "suppressor",
                "gate": col,
                "fail_count": int(fail_count),
                "fail_share_of_ou25_input": float(fail_count / total_ou),
                "windows_with_fail": windows_with_fail,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df["section_order"] = summary_df["section"].map({"missing_rate": 0, "suppressor": 1}).fillna(2)
        summary_df = summary_df.sort_values(["section_order", "fail_count"], ascending=[True, False]).drop(columns=["section_order"])
    summary_path = master_dir / "OU25_GATE_AUDIT__SUMMARY.csv"
    summary_df.to_csv(summary_path, index=False)
    log(f"[ou25-gate-audit] wrote summary: {summary_path}")

    league_paths = [p for p in base_outdir.rglob("OU25_GATE_AUDIT_BY_LEAGUE__*.csv") if "_MASTER" not in p.parts]
    if not league_paths:
        log("[ou25-gate-audit] no per-window league audit files found.")
        return

    league_frames: list[pd.DataFrame] = []
    for path in league_paths:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        league_frames.append(df)

    if not league_frames:
        log("[ou25-gate-audit] league audit files empty after load.")
        return

    league_all = pd.concat(league_frames, ignore_index=True, sort=False)
    if "ou25_input" in league_all.columns:
        league_all["fail_share_of_ou25_input_league"] = (
            pd.to_numeric(
                league_all.get("fail_count", pd.Series(0, index=league_all.index)),
                errors="coerce",
            ).fillna(0)
            / pd.to_numeric(
                league_all.get("ou25_input", pd.Series(0, index=league_all.index)),
                errors="coerce",
            ).replace(0, np.nan)
        )
    league_all_path = master_dir / "OU25_GATE_AUDIT_BY_LEAGUE__ALL_WINDOWS.csv"
    league_all.to_csv(league_all_path, index=False)
    log(f"[ou25-gate-audit] wrote league breakdown: {league_all_path}")

    league_ou = (
        league_all.loc[league_all["gate"].eq("ou25_input")]
        .groupby("league", dropna=False)
        .agg(league_ou25_input=("fail_count", "sum"))
        .reset_index()
    )
    league_summary = (
        league_all.groupby(["gate", "league"], dropna=False)
        .agg(fail_count=("fail_count", "sum"), windows=("window_id", "nunique"))
        .reset_index()
        .merge(league_ou, on="league", how="left")
    )
    league_summary["league_ou25_input"] = pd.to_numeric(
        league_summary.get("league_ou25_input", pd.Series(0, index=league_summary.index)),
        errors="coerce",
    ).fillna(0)
    league_summary["fail_share_of_ou25_input_league"] = league_summary["fail_count"] / league_summary["league_ou25_input"].replace(0, np.nan)
    league_summary["fail_share_of_ou25_input"] = (
        league_summary["fail_count"] / (total_ou if total_ou > 0 else 1.0)
    )
    league_summary["section"] = league_summary["gate"].map(
        {g: "missing_rate" for g in missing_gate_cols}
    ).fillna("suppressor")
    league_summary["section_order"] = league_summary["section"].map({"missing_rate": 0, "suppressor": 1}).fillna(2)
    league_summary = league_summary.sort_values(["section_order", "fail_count"], ascending=[True, False]).drop(columns=["section_order"])
    league_summary_path = master_dir / "OU25_GATE_AUDIT_BY_LEAGUE__SUMMARY.csv"
    league_summary.to_csv(league_summary_path, index=False)
    log(f"[ou25-gate-audit] wrote league summary: {league_summary_path}")


def _safe_read_first_row(path: Path) -> dict[str, object] | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    return df.iloc[0].to_dict()


def log_gate_top_suppressor(outdir: Path, window: WindowSpec) -> None:
    date_key = f"{window.date_from}_to_{window.date_to}"
    btts_path = outdir / f"BTTS_GATE_AUDIT__{date_key}.csv"
    ou25_path = outdir / f"OU25_GATE_AUDIT__{date_key}.csv"
    if btts_path.exists():
        row = _safe_read_first_row(btts_path)
        if row:
            log(
                "[gate-audit] BTTS top1="
                f"{row.get('top1_suppressor_gate','')} "
                f"count={row.get('top1_suppressor_count','')} "
                f"share={row.get('top1_suppressor_share_of_yes_input','')}"
            )
    if ou25_path.exists():
        row = _safe_read_first_row(ou25_path)
        if row:
            log(
                "[gate-audit] OU25 top1="
                f"{row.get('top1_suppressor_gate','')} "
                f"count={row.get('top1_suppressor_count','')} "
                f"share={row.get('top1_suppressor_share_of_ou25_input','')}"
            )


def build_observe_uplift_report(base_outdir: Path) -> None:
    master_dir = base_outdir / "_MASTER"
    ensure_dir(master_dir)
    scored_paths = sorted(base_outdir.rglob("DEPLOY_COMBINED_SCORED_*.csv"))
    if not scored_paths:
        log("[observe-uplift] no scored deploy files found.")
        return

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
    bin_edges = [0.0, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.01]
    bin_labels = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(bin_edges) - 1)]

    uplift_rows: list[dict[str, object]] = []
    bin_rows: list[dict[str, object]] = []
    uplift_league_rows: list[dict[str, object]] = []
    uplift_vs_standard_rows: list[dict[str, object]] = []

    for path in scored_paths:
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if df is None or df.empty:
            continue

        mk = df.get("market", pd.Series("", index=df.index)).astype("string").fillna("").str.lower().str.strip()
        tier = df.get("source_tier_file", df.get("deploy_tier", df.get("tier", pd.Series("", index=df.index))))
        tier = tier.astype("string").fillna("").str.upper().str.strip()
        is_observe = tier.eq("OBSERVE")

        pm = pd.to_numeric(df.get("model_p_for_bookie", np.nan), errors="coerce")
        league = df.get("league", pd.Series("", index=df.index)).astype("string").fillna("").str.strip()
        is_standard = tier.eq("STANDARD")

        for market_key, cfg in SUPPORTED_MARKETS.items():
            hit_col = cfg.get("hit_col", "")
            if hit_col not in df.columns:
                continue
            sub_mask = mk.eq(market_key) & is_observe
            if not bool(sub_mask.any()):
                continue

            sub = df.loc[sub_mask].copy()
            hits = pd.to_numeric(sub.get(hit_col, np.nan), errors="coerce")
            graded_mask = hits.notna()

            graded_n = int(graded_mask.sum())
            wins = int(hits.fillna(0).sum())
            uplift_rows.append(
                {
                    "market": market_key,
                    "graded": graded_n,
                    "wins": wins,
                    "hit_rate": (wins / graded_n) if graded_n > 0 else float("nan"),
                    "threshold": "ALL",
                }
            )

            # Standard baseline for uplift-vs-standard
            std_mask = mk.eq(market_key) & is_standard
            std_hits = pd.to_numeric(df.loc[std_mask, hit_col], errors="coerce")
            std_graded = int(std_hits.notna().sum())
            std_wins = int(std_hits.fillna(0).sum())
            std_hit_rate = (std_wins / std_graded) if std_graded > 0 else float("nan")

            for t in thresholds:
                t_mask = graded_mask & (pm.loc[sub.index] >= t)
                t_n = int(t_mask.sum())
                t_w = int(hits.loc[t_mask].fillna(0).sum())
                uplift_rows.append(
                    {
                        "market": market_key,
                        "graded": t_n,
                        "wins": t_w,
                        "hit_rate": (t_w / t_n) if t_n > 0 else float("nan"),
                        "threshold": f">={t:.2f}",
                    }
                )

                uplift_vs_standard_rows.append(
                    {
                        "market": market_key,
                        "threshold": f">={t:.2f}",
                        "observe_graded": t_n,
                        "observe_wins": t_w,
                        "observe_hit_rate": (t_w / t_n) if t_n > 0 else float("nan"),
                        "standard_graded": std_graded,
                        "standard_wins": std_wins,
                        "standard_hit_rate": std_hit_rate,
                        "delta_hit_rate": ((t_w / t_n) - std_hit_rate) if (t_n > 0 and std_graded > 0) else float("nan"),
                    }
                )

            # bin stats
            bin_idx = pd.cut(pm.loc[sub.index], bins=bin_edges, labels=bin_labels, include_lowest=True)
            for b in bin_labels:
                b_mask = graded_mask & (bin_idx == b)
                b_n = int(b_mask.sum())
                b_w = int(hits.loc[b_mask].fillna(0).sum())
                if b_n == 0:
                    continue
                bin_rows.append(
                    {
                        "market": market_key,
                        "bin": b,
                        "graded": b_n,
                        "wins": b_w,
                        "hit_rate": (b_w / b_n) if b_n > 0 else float("nan"),
                    }
                )

            # per-league uplift bins (ALL threshold + >=0.60)
            for lg in league.loc[sub.index].astype("string").fillna("").replace("", "UNKNOWN").unique():
                lg_mask = (league.loc[sub.index] == lg)
                lg_hits = hits.loc[lg_mask]
                lg_graded = int(lg_hits.notna().sum())
                lg_wins = int(lg_hits.fillna(0).sum())
                uplift_league_rows.append(
                    {
                        "market": market_key,
                        "league": lg,
                        "threshold": "ALL",
                        "graded": lg_graded,
                        "wins": lg_wins,
                        "hit_rate": (lg_wins / lg_graded) if lg_graded > 0 else float("nan"),
                    }
                )
                t = 0.60
                lg_t_mask = lg_mask & (pm.loc[sub.index] >= t)
                lg_t_hits = hits.loc[lg_t_mask]
                lg_tn = int(lg_t_hits.notna().sum())
                lg_tw = int(lg_t_hits.fillna(0).sum())
                uplift_league_rows.append(
                    {
                        "market": market_key,
                        "league": lg,
                        "threshold": f">={t:.2f}",
                        "graded": lg_tn,
                        "wins": lg_tw,
                        "hit_rate": (lg_tw / lg_tn) if lg_tn > 0 else float("nan"),
                    }
                )

    if uplift_rows:
        uplift_df = pd.DataFrame(uplift_rows)
        uplift_path = master_dir / "OBSERVE_UPLIFT__SUMMARY.csv"
        uplift_df.to_csv(uplift_path, index=False)
        log(f"[observe-uplift] wrote summary: {uplift_path}")

    if bin_rows:
        bin_df = pd.DataFrame(bin_rows)
        bin_path = master_dir / "OBSERVE_UPLIFT__BY_BIN.csv"
        bin_df.to_csv(bin_path, index=False)
        log(f"[observe-uplift] wrote bins: {bin_path}")

    if uplift_league_rows:
        league_df = pd.DataFrame(uplift_league_rows)
        league_path = master_dir / "OBSERVE_UPLIFT__BY_LEAGUE.csv"
        league_df.to_csv(league_path, index=False)
        log(f"[observe-uplift] wrote league split: {league_path}")

    if uplift_vs_standard_rows:
        vs_df = pd.DataFrame(uplift_vs_standard_rows)
        vs_path = master_dir / "OBSERVE_UPLIFT__VS_STANDARD.csv"
        vs_df.to_csv(vs_path, index=False)
        log(f"[observe-uplift] wrote vs-standard: {vs_path}")


def build_odds_audits(base_outdir: Path) -> None:
    master_dir = base_outdir / "_MASTER"
    ensure_dir(master_dir)
    scored_paths = sorted(base_outdir.rglob("DEPLOY_COMBINED_SCORED_*.csv"))
    if not scored_paths:
        log("[odds-audit] no scored deploy files found.")
        return

    rows: list[dict[str, object]] = []
    bin_rows: list[dict[str, object]] = []
    tier_rows: list[dict[str, object]] = []
    market_year_tier_rows: list[dict[str, object]] = []
    odds_edges = [0.0, 1.40, 1.60, 1.80, 2.00, 2.25, 2.50, 3.00, 5.00, 10.00, 100.0]
    odds_labels = [f"{odds_edges[i]:.2f}-{odds_edges[i+1]:.2f}" for i in range(len(odds_edges) - 1)]

    for path in scored_paths:
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if df is None or df.empty:
            continue

        mk = df.get("market", pd.Series("", index=df.index)).astype("string").fillna("").str.lower().str.strip()
        league = df.get("league", pd.Series("", index=df.index)).astype("string").fillna("").str.strip()
        tier = df.get("source_tier_file", df.get("deploy_tier", df.get("tier", pd.Series("", index=df.index))))
        tier = tier.astype("string").fillna("").str.upper().str.strip()
        odds = pd.to_numeric(df.get("bookie_od", np.nan), errors="coerce")

        match_date = df.get("match_date", pd.Series("", index=df.index)).astype("string").fillna("").str.strip()
        year = match_date.str.slice(0, 4)
        year = year.where(year.str.match(r"^\\d{4}$"), pd.Series("", index=year.index))
        year = year.replace("", "UNKNOWN")

        for market_key, cfg in SUPPORTED_MARKETS.items():
            hit_col = cfg.get("hit_col", "")
            if hit_col not in df.columns:
                continue
            sub_mask = mk.eq(market_key)
            if not bool(sub_mask.any()):
                continue
            sub = df.loc[sub_mask].copy()
            hits = pd.to_numeric(sub.get(hit_col, np.nan), errors="coerce")
            graded_mask = hits.notna()
            if not bool(graded_mask.any()):
                continue

            sub_odds = odds.loc[sub.index]
            sub_year = year.loc[sub.index]
            sub_league = league.loc[sub.index]
            sub_tier = tier.loc[sub.index]

            # Market + league + year summary
            for (lg, yr), grp_idx in sub.groupby([sub_league, sub_year]).groups.items():
                g_hits = hits.loc[grp_idx]
                g_odds = sub_odds.loc[grp_idx]
                g_mask = g_hits.notna() & g_odds.notna()
                if not bool(g_mask.any()):
                    continue
                wins = int(g_hits.loc[g_mask].fillna(0).sum())
                graded = int(g_mask.sum())
                avg_od = float(g_odds.loc[g_mask].mean())
                avg_od_win = float(g_odds.loc[g_mask & (g_hits == 1)].mean()) if bool((g_mask & (g_hits == 1)).any()) else float("nan")
                avg_od_loss = float(g_odds.loc[g_mask & (g_hits == 0)].mean()) if bool((g_mask & (g_hits == 0)).any()) else float("nan")
                hit_rate = wins / graded if graded > 0 else float("nan")
                roi = float(((g_odds.loc[g_mask] - 1.0) * g_hits.loc[g_mask] - (1 - g_hits.loc[g_mask])).mean())
                rows.append(
                    {
                        "market": market_key,
                        "league": lg,
                        "year": yr,
                        "graded": graded,
                        "wins": wins,
                        "hit_rate": hit_rate,
                        "avg_odds": avg_od,
                        "avg_odds_wins": avg_od_win,
                        "avg_odds_losses": avg_od_loss,
                        "roi_per_unit": roi,
                    }
                )

            # Tier summary (all years)
            for tr, grp_idx in sub.groupby(sub_tier).groups.items():
                g_hits = hits.loc[grp_idx]
                g_odds = sub_odds.loc[grp_idx]
                g_mask = g_hits.notna() & g_odds.notna()
                if not bool(g_mask.any()):
                    continue
                wins = int(g_hits.loc[g_mask].fillna(0).sum())
                graded = int(g_mask.sum())
                avg_od = float(g_odds.loc[g_mask].mean())
                hit_rate = wins / graded if graded > 0 else float("nan")
                roi = float(((g_odds.loc[g_mask] - 1.0) * g_hits.loc[g_mask] - (1 - g_hits.loc[g_mask])).mean())
                tier_rows.append(
                    {
                        "market": market_key,
                        "tier": tr,
                        "graded": graded,
                        "wins": wins,
                        "hit_rate": hit_rate,
                        "avg_odds": avg_od,
                        "roi_per_unit": roi,
                    }
                )

            # Market + year + tier summary
            for (yr, tr), grp_idx in sub.groupby([sub_year, sub_tier]).groups.items():
                g_hits = hits.loc[grp_idx]
                g_odds = sub_odds.loc[grp_idx]
                g_mask = g_hits.notna() & g_odds.notna()
                if not bool(g_mask.any()):
                    continue
                wins = int(g_hits.loc[g_mask].fillna(0).sum())
                graded = int(g_mask.sum())
                avg_od = float(g_odds.loc[g_mask].mean())
                hit_rate = wins / graded if graded > 0 else float("nan")
                roi = float(((g_odds.loc[g_mask] - 1.0) * g_hits.loc[g_mask] - (1 - g_hits.loc[g_mask])).mean())
                market_year_tier_rows.append(
                    {
                        "market": market_key,
                        "year": yr,
                        "tier": tr,
                        "graded": graded,
                        "wins": wins,
                        "hit_rate": hit_rate,
                        "avg_odds": avg_od,
                        "roi_per_unit": roi,
                    }
                )

            # Odds bin summary by market (all leagues)
            odds_bin = pd.cut(sub_odds, bins=odds_edges, labels=odds_labels, include_lowest=True)
            for b in odds_labels:
                b_mask = graded_mask & (odds_bin == b)
                if not bool(b_mask.any()):
                    continue
                wins = int(hits.loc[b_mask].fillna(0).sum())
                graded = int(b_mask.sum())
                avg_od = float(sub_odds.loc[b_mask].mean())
                hit_rate = wins / graded if graded > 0 else float("nan")
                roi = float(((sub_odds.loc[b_mask] - 1.0) * hits.loc[b_mask] - (1 - hits.loc[b_mask])).mean())
                bin_rows.append(
                    {
                        "market": market_key,
                        "odds_bin": b,
                        "graded": graded,
                        "wins": wins,
                        "hit_rate": hit_rate,
                        "avg_odds": avg_od,
                        "roi_per_unit": roi,
                    }
                )

    if rows:
        summary_df = pd.DataFrame(rows)
        out_path = master_dir / "ODDS_AUDIT__BY_MARKET_LEAGUE_YEAR.csv"
        summary_df.to_csv(out_path, index=False)
        log(f"[odds-audit] wrote market/league/year summary: {out_path}")

    if bin_rows:
        bin_df = pd.DataFrame(bin_rows)
        bin_path = master_dir / "ODDS_AUDIT__BY_ODDS_BIN.csv"
        bin_df.to_csv(bin_path, index=False)
        log(f"[odds-audit] wrote odds-bin summary: {bin_path}")

    if tier_rows:
        tier_df = pd.DataFrame(tier_rows)
        tier_path = master_dir / "ODDS_AUDIT__BY_TIER.csv"
        tier_df.to_csv(tier_path, index=False)
        log(f"[odds-audit] wrote tier summary: {tier_path}")

    if market_year_tier_rows:
        myt_df = pd.DataFrame(market_year_tier_rows)
        myt_path = master_dir / "ODDS_AUDIT__BY_MARKET_YEAR_TIER.csv"
        myt_df.to_csv(myt_path, index=False)
        log(f"[odds-audit] wrote market/year/tier summary: {myt_path}")


def aggregate_universe_audits(base_outdir: Path) -> None:
    master_dir = base_outdir / "_MASTER"
    ensure_dir(master_dir)

    base_paths = [p for p in base_outdir.rglob("UNIVERSE_BASE_RATES__*.csv") if "_MASTER" not in p.parts]
    if not base_paths:
        log("[universe-audit] no per-window base rate files found.")
        return

    base_frames: list[pd.DataFrame] = []
    for path in base_paths:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        base_frames.append(df)

    base_all = pd.concat(base_frames, ignore_index=True, sort=False)
    base_all_path = master_dir / "UNIVERSE_BASE_RATES__ALL_WINDOWS.csv"
    base_all.to_csv(base_all_path, index=False)
    log(f"[universe-audit] wrote all-windows base: {base_all_path}")

    summary = (
        base_all.groupby(["league", "year"], dropna=False)
        .agg(
            total_matches=("total_matches", "sum"),
            btts_yes_count=("btts_yes_count", "sum"),
            ou25_yes_count=("ou25_yes_count", "sum"),
        )
        .reset_index()
    )
    summary["btts_yes_rate"] = summary["btts_yes_count"] / summary["total_matches"]
    summary["ou25_yes_rate"] = summary["ou25_yes_count"] / summary["total_matches"]
    summary_path = master_dir / "UNIVERSE_BASE_RATES__SUMMARY.csv"
    summary.to_csv(summary_path, index=False)
    log(f"[universe-audit] wrote summary: {summary_path}")

    compare_paths = [p for p in base_outdir.rglob("UNIVERSE_COMPARE__*.csv") if "_MASTER" not in p.parts]
    if not compare_paths:
        log("[universe-audit] no per-window compare files found.")
        return

    comp_frames: list[pd.DataFrame] = []
    for path in compare_paths:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        comp_frames.append(df)

    comp_all = pd.concat(comp_frames, ignore_index=True, sort=False)
    comp_all_path = master_dir / "UNIVERSE_COMPARE__ALL_WINDOWS.csv"
    comp_all.to_csv(comp_all_path, index=False)
    log(f"[universe-audit] wrote all-windows compare: {comp_all_path}")

    comp_summary = (
        comp_all.groupby(["market", "league"], dropna=False)
        .agg(
            graded=("graded", "sum"),
            wins=("wins", "sum"),
            hit_rate=("hit_rate", "mean"),
            avg_odds=("avg_odds", "mean"),
            roi_per_unit=("roi_per_unit", "mean"),
            base_btts_yes_rate=("base_btts_yes_rate", "mean"),
            base_ou25_yes_rate=("base_ou25_yes_rate", "mean"),
        )
        .reset_index()
    )
    comp_summary_path = master_dir / "UNIVERSE_COMPARE__SUMMARY.csv"
    comp_summary.to_csv(comp_summary_path, index=False)
    log(f"[universe-audit] wrote compare summary: {comp_summary_path}")


def aggregate_feature_audits(base_outdir: Path) -> None:
    master_dir = base_outdir / "_MASTER"
    ensure_dir(master_dir)

    corr_paths = [p for p in base_outdir.rglob("FEATURE_OUTCOME_CORR__*.csv") if "_MASTER" not in p.parts]
    if corr_paths:
        frames = []
        for path in corr_paths:
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if df is None or df.empty:
                continue
            frames.append(df)
        if frames:
            all_df = pd.concat(frames, ignore_index=True, sort=False)
            all_path = master_dir / "FEATURE_OUTCOME_CORR__ALL_WINDOWS.csv"
            all_df.to_csv(all_path, index=False)
            log(f"[feature-audit] wrote corr all: {all_path}")
            summary = (
                all_df.groupby(["market", "league", "feature"], dropna=False)
                .agg(n=("n", "sum"), corr=("corr", "mean"))
                .reset_index()
                .sort_values(["corr"], ascending=[False])
            )
            summary_path = master_dir / "FEATURE_OUTCOME_CORR__SUMMARY.csv"
            summary.to_csv(summary_path, index=False)
            log(f"[feature-audit] wrote corr summary: {summary_path}")

            # Per-league top-5 feature ranking (by absolute correlation)
            try:
                ranking = summary.copy()
                ranking["abs_corr"] = ranking["corr"].abs()
                ranking = ranking.sort_values(["market", "league", "abs_corr"], ascending=[True, True, False])
                ranking = ranking.groupby(["market", "league"]).head(5).copy()
                ranking_path = master_dir / "FEATURE_OUTCOME_CORR__TOP5_BY_LEAGUE.csv"
                ranking.to_csv(ranking_path, index=False)
                log(f"[feature-audit] wrote top5 by league: {ranking_path}")
            except Exception:
                pass

            # Year-over-year drift (corr by year)
            try:
                if "year" in all_df.columns:
                    drift = (
                        all_df.groupby(["market", "league", "feature", "year"], dropna=False)
                        .agg(n=("n", "sum"), corr=("corr", "mean"))
                        .reset_index()
                        .sort_values(["market", "league", "feature", "year"])
                    )
                    drift_path = master_dir / "FEATURE_OUTCOME_CORR__BY_YEAR.csv"
                    drift.to_csv(drift_path, index=False)
                    log(f"[feature-audit] wrote corr by year: {drift_path}")
                    # Feature stability scoring (low variance = reliable)
                    stability = (
                        drift.groupby(["market", "league", "feature"], dropna=False)
                        .agg(
                            n_years=("year", "nunique"),
                            corr_mean=("corr", "mean"),
                            corr_std=("corr", "std"),
                        )
                        .reset_index()
                    )
                    stability["corr_std"] = stability["corr_std"].fillna(0.0)
                    stability["stability_score"] = 1.0 / (1.0 + stability["corr_std"])
                    stability = stability.sort_values(["stability_score", "corr_mean"], ascending=[False, False])
                    stability_path = master_dir / "FEATURE_OUTCOME_CORR__STABILITY.csv"
                    stability.to_csv(stability_path, index=False)
                    log(f"[feature-audit] wrote stability: {stability_path}")

                    # Global stability rank (top-N features across all leagues)
                    try:
                        global_rank = (
                            stability.groupby(["market", "feature"], dropna=False)
                            .agg(
                                leagues=("league", "nunique"),
                                n_years=("n_years", "mean"),
                                corr_mean=("corr_mean", "mean"),
                                corr_std=("corr_std", "mean"),
                                stability_score=("stability_score", "mean"),
                            )
                            .reset_index()
                        )
                        global_rank = global_rank.sort_values(["stability_score", "corr_mean"], ascending=[False, False])
                        global_path = master_dir / "FEATURE_OUTCOME_CORR__GLOBAL_RANK.csv"
                        global_rank.to_csv(global_path, index=False)
                        log(f"[feature-audit] wrote global rank: {global_path}")
                    except Exception:
                        pass

                    # Stability vs performance quadrant table
                    try:
                        q = global_rank.copy()
                        if not q.empty:
                            stab_med = float(q["stability_score"].median())
                            perf_med = float(q["corr_mean"].median())
                            q["stability_bucket"] = q["stability_score"].ge(stab_med).map({True: "HIGH", False: "LOW"})
                            q["performance_bucket"] = q["corr_mean"].ge(perf_med).map({True: "HIGH", False: "LOW"})
                            q["quadrant"] = q["stability_bucket"] + "_" + q["performance_bucket"]
                            quad = (
                                q.groupby(["market", "quadrant"], dropna=False)
                                .agg(count=("feature", "count"))
                                .reset_index()
                                .sort_values(["market", "count"], ascending=[True, False])
                            )
                            quad_path = master_dir / "FEATURE_OUTCOME_CORR__QUADRANT.csv"
                            quad.to_csv(quad_path, index=False)
                            log(f"[feature-audit] wrote quadrant: {quad_path}")

                            # Top-N features per quadrant (stable + predictive candidates)
                            topn = q.sort_values(["stability_score", "corr_mean"], ascending=[False, False])
                            topn = topn.groupby(["market", "quadrant"]).head(10).copy()
                            topn_path = master_dir / "FEATURE_OUTCOME_CORR__QUADRANT_TOP10.csv"
                            topn.to_csv(topn_path, index=False)
                            log(f"[feature-audit] wrote quadrant top10: {topn_path}")
                    except Exception:
                        pass
            except Exception:
                pass
    else:
        log("[feature-audit] no correlation files found.")

    thresh_paths = [p for p in base_outdir.rglob("FEATURE_OUTCOME_THRESH__*.csv") if "_MASTER" not in p.parts]
    if thresh_paths:
        frames = []
        for path in thresh_paths:
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if df is None or df.empty:
                continue
            frames.append(df)
        if frames:
            all_df = pd.concat(frames, ignore_index=True, sort=False)
            all_path = master_dir / "FEATURE_OUTCOME_THRESH__ALL_WINDOWS.csv"
            all_df.to_csv(all_path, index=False)
            log(f"[feature-audit] wrote thresh all: {all_path}")
            summary = (
                all_df.groupby(["market", "league", "feature", "quantile"], dropna=False)
                .agg(n=("n", "sum"), hit_rate=("hit_rate", "mean"))
                .reset_index()
                .sort_values(["hit_rate"], ascending=[False])
            )
            summary_path = master_dir / "FEATURE_OUTCOME_THRESH__SUMMARY.csv"
            summary.to_csv(summary_path, index=False)
            log(f"[feature-audit] wrote thresh summary: {summary_path}")
    else:
        log("[feature-audit] no threshold files found.")


def write_universe_base_audit(
    merged_actuals: pd.DataFrame,
    window: WindowSpec,
    reports_dir: Path,
) -> Path | None:
    if merged_actuals is None or merged_actuals.empty:
        return None

    df = merged_actuals.copy()
    league = df.get("league", pd.Series("", index=df.index)).astype("string").fillna("").str.strip()
    if "match_date" in df.columns:
        year = df["match_date"].astype("string").fillna("").str.slice(0, 4)
    else:
        year = pd.Series(str(window.date_to)[:4], index=df.index)
    year = year.where(year.str.match(r"^\\d{4}$"), "UNKNOWN")

    btts = pd.to_numeric(df.get("actual_btts_yes", np.nan), errors="coerce")
    ou25 = pd.to_numeric(df.get("actual_over25", np.nan), errors="coerce")
    btts_yes = btts.eq(1)
    ou25_yes = ou25.eq(1)

    base_rows: list[dict[str, object]] = []
    for (lg, yr), idx in df.groupby([league, year]).groups.items():
        n = int(len(idx))
        b_yes = int(btts_yes.loc[idx].sum()) if btts is not None else 0
        o_yes = int(ou25_yes.loc[idx].sum()) if ou25 is not None else 0
        base_rows.append(
            {
                "window_id": window.window_id,
                "date_from": window.date_from,
                "date_to": window.date_to,
                "league": lg if lg else "UNKNOWN",
                "year": yr if yr else "UNKNOWN",
                "total_matches": n,
                "btts_yes_count": b_yes,
                "btts_yes_rate": (b_yes / n) if n > 0 else float("nan"),
                "ou25_yes_count": o_yes,
                "ou25_yes_rate": (o_yes / n) if n > 0 else float("nan"),
            }
        )

    # Add ALL league per year + ALL years rollup
    for yr, idx in df.groupby(year).groups.items():
        n = int(len(idx))
        b_yes = int(btts_yes.loc[idx].sum())
        o_yes = int(ou25_yes.loc[idx].sum())
        base_rows.append(
            {
                "window_id": window.window_id,
                "date_from": window.date_from,
                "date_to": window.date_to,
                "league": "ALL",
                "year": yr if yr else "UNKNOWN",
                "total_matches": n,
                "btts_yes_count": b_yes,
                "btts_yes_rate": (b_yes / n) if n > 0 else float("nan"),
                "ou25_yes_count": o_yes,
                "ou25_yes_rate": (o_yes / n) if n > 0 else float("nan"),
            }
        )

    n_all = int(len(df))
    b_all = int(btts_yes.sum())
    o_all = int(ou25_yes.sum())
    base_rows.append(
        {
            "window_id": window.window_id,
            "date_from": window.date_from,
            "date_to": window.date_to,
            "league": "ALL",
            "year": "ALL",
            "total_matches": n_all,
            "btts_yes_count": b_all,
            "btts_yes_rate": (b_all / n_all) if n_all > 0 else float("nan"),
            "ou25_yes_count": o_all,
            "ou25_yes_rate": (o_all / n_all) if n_all > 0 else float("nan"),
        }
    )

    out_path = reports_dir / f"UNIVERSE_BASE_RATES__{window.date_from}_to_{window.date_to}.csv"
    pd.DataFrame(base_rows).to_csv(out_path, index=False)
    return out_path


def write_universe_compare_audit(
    merged_actuals: pd.DataFrame,
    scored_df: pd.DataFrame,
    window: WindowSpec,
    reports_dir: Path,
) -> Path | None:
    if merged_actuals is None or merged_actuals.empty:
        return None
    if scored_df is None or scored_df.empty:
        return None

    base = merged_actuals.copy()
    base_league = base.get("league", pd.Series("", index=base.index)).astype("string").fillna("").str.strip()
    btts = pd.to_numeric(base.get("actual_btts_yes", np.nan), errors="coerce")
    ou25 = pd.to_numeric(base.get("actual_over25", np.nan), errors="coerce")
    base_rows = (
        pd.DataFrame(
            {
                "league": base_league.replace("", "UNKNOWN"),
                "btts_yes": btts,
                "ou25_yes": ou25,
            }
        )
        .groupby("league", dropna=False)
        .agg(
            base_matches=("btts_yes", "count"),
            base_btts_yes=("btts_yes", "sum"),
            base_ou25_yes=("ou25_yes", "sum"),
        )
        .reset_index()
    )
    base_rows["base_btts_yes_rate"] = base_rows["base_btts_yes"] / base_rows["base_matches"]
    base_rows["base_ou25_yes_rate"] = base_rows["base_ou25_yes"] / base_rows["base_matches"]

    mk = scored_df.get("market", pd.Series("", index=scored_df.index)).astype("string").fillna("").str.lower().str.strip()
    league = scored_df.get("league", pd.Series("", index=scored_df.index)).astype("string").fillna("").str.strip()
    odds = pd.to_numeric(scored_df.get("bookie_od", np.nan), errors="coerce")

    comp_rows: list[dict[str, object]] = []
    for market_key in ["btts", "ou25"]:
        if market_key == "btts":
            hit_col = "btts_hit"
        else:
            hit_col = "ou25_hit"
        if hit_col not in scored_df.columns:
            continue
        sub_mask = mk.eq(market_key)
        if not bool(sub_mask.any()):
            continue
        sub = scored_df.loc[sub_mask].copy()
        hits = pd.to_numeric(sub.get(hit_col, np.nan), errors="coerce")
        graded = hits.notna()
        if not bool(graded.any()):
            continue
        sub_league = league.loc[sub.index].replace("", "UNKNOWN")
        for lg, idx in sub.groupby(sub_league).groups.items():
            g_hits = hits.loc[idx]
            g_odds = odds.loc[idx]
            g_mask = g_hits.notna()
            if not bool(g_mask.any()):
                continue
            wins = int(g_hits.loc[g_mask].fillna(0).sum())
            total = int(g_mask.sum())
            avg_od = float(g_odds.loc[g_mask].mean())
            hit_rate = wins / total if total > 0 else float("nan")
            roi = float(((g_odds.loc[g_mask] - 1.0) * g_hits.loc[g_mask] - (1 - g_hits.loc[g_mask])).mean())
            comp_rows.append(
                {
                    "window_id": window.window_id,
                    "date_from": window.date_from,
                    "date_to": window.date_to,
                    "market": market_key,
                    "league": lg,
                    "graded": total,
                    "wins": wins,
                    "hit_rate": hit_rate,
                    "avg_odds": avg_od,
                    "roi_per_unit": roi,
                }
            )

    if not comp_rows:
        return None

    comp_df = pd.DataFrame(comp_rows)
    comp_df = comp_df.merge(base_rows, on="league", how="left")
    out_path = reports_dir / f"UNIVERSE_COMPARE__{window.date_from}_to_{window.date_to}.csv"
    comp_df.to_csv(out_path, index=False)
    return out_path


def write_feature_outcome_audit(
    source_df: pd.DataFrame,
    merged_actuals: pd.DataFrame,
    window: WindowSpec,
    reports_dir: Path,
) -> tuple[Path | None, Path | None]:
    if source_df is None or source_df.empty:
        return None, None
    if merged_actuals is None or merged_actuals.empty:
        return None, None

    df = enrich_with_merged_actuals(source_df.copy(), merged_actuals)
    mk = df.get("market", pd.Series("", index=df.index)).astype("string").fillna("").str.lower().str.strip()
    league = df.get("league", pd.Series("", index=df.index)).astype("string").fillna("").str.strip().replace("", "UNKNOWN")
    if "match_date" in df.columns:
        year = df["match_date"].astype("string").fillna("").str.slice(0, 4)
    else:
        year = pd.Series(str(window.date_to)[:4], index=df.index)
    year = year.where(year.str.match(r"^\\d{4}$"), "UNKNOWN")

    feature_sets = {
        "btts": [
            "model_p_for_bookie",
            "exp_goals_sum",
            "bookie_lambda_total_fit",
            "p_home_fts",
            "p_away_fts",
            "home_ge2_confidence",
            "away_ge2_confidence",
            "cs_max",
            "p00_est",
            "btts_top3_mass",
            "cs00_top3_mass",
            "fts_sum",
            "cs_sum",
        ],
        "ou25": [
            "model_p_for_bookie",
            "exp_goals_sum",
            "bookie_lambda_total_fit",
            "avg_btts_rate",
            "cs_over25_mass",
            "top3_over_count",
            "average_goals_per_match_pre_match",
            "xg_sum_pre_match",
            "over_25_percentage_pre_match",
            "goaliness_avg_5_home",
            "goaliness_avg_5_away",
            "under25_rate_5_home",
            "under25_rate_5_away",
        ],
    }

    corr_rows: list[dict[str, object]] = []
    thresh_rows: list[dict[str, object]] = []
    quantiles = [0.6, 0.7, 0.8, 0.9]

    for market_key, features in feature_sets.items():
        if market_key == "btts":
            y = pd.to_numeric(df.get("actual_btts_yes", np.nan), errors="coerce")
        else:
            y = pd.to_numeric(df.get("actual_over25", np.nan), errors="coerce")

        sub_mask = mk.eq(market_key) & y.notna()
        if not bool(sub_mask.any()):
            continue

        sub = df.loc[sub_mask].copy()
        y_sub = y.loc[sub.index]
        lg_series = league.loc[sub.index]

        for lg, idx in sub.groupby(lg_series).groups.items():
            y_lg = y_sub.loc[idx]
            if y_lg.notna().sum() < 20:
                continue
            for feat in features:
                if feat not in sub.columns:
                    continue
                x = pd.to_numeric(sub.loc[idx, feat], errors="coerce")
                mask = x.notna() & y_lg.notna()
                if mask.sum() < 20:
                    continue
                xv = x.loc[mask]
                yv = y_lg.loc[mask]
                if xv.nunique() < 2:
                    continue
                corr = xv.corr(yv)
                corr_rows.append(
                    {
                        "window_id": window.window_id,
                        "date_from": window.date_from,
                        "date_to": window.date_to,
                        "market": market_key,
                        "league": lg,
                        "year": year.loc[idx].iloc[0] if "year" in locals() else "UNKNOWN",
                        "feature": feat,
                        "n": int(mask.sum()),
                        "corr": float(corr) if corr is not None else float("nan"),
                    }
                )

                for q in quantiles:
                    thr = float(xv.quantile(q))
                    sel = xv >= thr
                    n_sel = int(sel.sum())
                    if n_sel < 10:
                        continue
                    hit_rate = float(yv.loc[sel].mean())
                    thresh_rows.append(
                        {
                            "window_id": window.window_id,
                            "date_from": window.date_from,
                            "date_to": window.date_to,
                            "market": market_key,
                            "league": lg,
                            "year": year.loc[idx].iloc[0] if "year" in locals() else "UNKNOWN",
                            "feature": feat,
                            "quantile": q,
                            "threshold": thr,
                            "n": n_sel,
                            "hit_rate": hit_rate,
                        }
                    )

    corr_path = None
    thresh_path = None
    if corr_rows:
        corr_path = reports_dir / f"FEATURE_OUTCOME_CORR__{window.date_from}_to_{window.date_to}.csv"
        pd.DataFrame(corr_rows).to_csv(corr_path, index=False)
    if thresh_rows:
        thresh_path = reports_dir / f"FEATURE_OUTCOME_THRESH__{window.date_from}_to_{window.date_to}.csv"
        pd.DataFrame(thresh_rows).to_csv(thresh_path, index=False)
    return corr_path, thresh_path


def build_counterfactual_uplift(base_outdir: Path) -> None:
    master_dir = base_outdir / "_MASTER"
    ensure_dir(master_dir)
    scored_paths = sorted(base_outdir.rglob("DEPLOY_COMBINED_SCORED_*.csv"))
    if not scored_paths:
        log("[counterfactual] no scored deploy files found.")
        return

    features = [
        "model_p_for_bookie",
        "exp_goals_sum",
        "bookie_lambda_total_fit",
        "avg_btts_rate",
        "cs_over25_mass",
        "top3_over_count",
        "p_home_fts",
        "p_away_fts",
        "home_ge2_confidence",
        "away_ge2_confidence",
        "cs_max",
        "p00_est",
        "btts_top3_mass",
        "cs00_top3_mass",
    ]
    quantiles = [0.6, 0.7, 0.8, 0.9]

    rows: list[dict[str, object]] = []
    league_rows: list[dict[str, object]] = []

    for path in scored_paths:
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if df is None or df.empty:
            continue

        mk = df.get("market", pd.Series("", index=df.index)).astype("string").fillna("").str.lower().str.strip()
        tier = df.get("source_tier_file", df.get("deploy_tier", df.get("tier", pd.Series("", index=df.index))))
        tier = tier.astype("string").fillna("").str.upper().str.strip()
        league = df.get("league", pd.Series("", index=df.index)).astype("string").fillna("").str.strip().replace("", "UNKNOWN")

        for market_key, cfg in SUPPORTED_MARKETS.items():
            hit_col = cfg.get("hit_col", "")
            if hit_col not in df.columns:
                continue
            obs_mask = mk.eq(market_key) & tier.eq("OBSERVE")
            std_mask = mk.eq(market_key) & tier.eq("STANDARD")
            if not bool(obs_mask.any()):
                continue
            hits_obs = pd.to_numeric(df.loc[obs_mask, hit_col], errors="coerce")
            hits_std = pd.to_numeric(df.loc[std_mask, hit_col], errors="coerce")
            std_rate = float(hits_std.mean()) if hits_std.notna().any() else float("nan")

            for feat in features:
                if feat not in df.columns:
                    continue
                x = pd.to_numeric(df.loc[obs_mask, feat], errors="coerce")
                if x.notna().sum() < 20:
                    continue
                for q in quantiles:
                    thr = float(x.quantile(q))
                    sel = x >= thr
                    n_sel = int(sel.sum())
                    if n_sel < 10:
                        continue
                    y_sel = hits_obs.loc[sel]
                    hit_rate = float(y_sel.mean())
                    rows.append(
                        {
                            "market": market_key,
                            "feature": feat,
                            "quantile": q,
                            "threshold": thr,
                            "observe_graded": n_sel,
                            "observe_hit_rate": hit_rate,
                            "standard_hit_rate": std_rate,
                            "delta_hit_rate": (hit_rate - std_rate) if std_rate == std_rate else float("nan"),
                        }
                    )

                    # per league
                    lg_series = league.loc[obs_mask]
                    for lg, idx in x.groupby(lg_series).groups.items():
                        xv = x.loc[idx]
                        if xv.notna().sum() < 10:
                            continue
                        thr_l = float(xv.quantile(q))
                        sel_l = xv >= thr_l
                        n_l = int(sel_l.sum())
                        if n_l < 10:
                            continue
                        y_l = hits_obs.loc[idx].loc[sel_l]
                        hit_l = float(y_l.mean())
                        league_rows.append(
                            {
                                "market": market_key,
                                "league": lg,
                                "feature": feat,
                                "quantile": q,
                                "threshold": thr_l,
                                "observe_graded": n_l,
                                "observe_hit_rate": hit_l,
                                "standard_hit_rate": std_rate,
                                "delta_hit_rate": (hit_l - std_rate) if std_rate == std_rate else float("nan"),
                            }
                        )

    if rows:
        out_path = master_dir / "COUNTERFACTUAL_UPLIFT__SUMMARY.csv"
        pd.DataFrame(rows).sort_values(["delta_hit_rate", "observe_graded"], ascending=[False, False]).to_csv(out_path, index=False)
        log(f"[counterfactual] wrote summary: {out_path}")

    if league_rows:
        out_path = master_dir / "COUNTERFACTUAL_UPLIFT__BY_LEAGUE.csv"
        pd.DataFrame(league_rows).sort_values(["delta_hit_rate", "observe_graded"], ascending=[False, False]).to_csv(out_path, index=False)
        log(f"[counterfactual] wrote league summary: {out_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run multi-window walk-forward deploy backtests.")
    ap.add_argument("--manifest", required=True, help="CSV with window_id,date_from,date_to")
    ap.add_argument("--base-outdir", default=str(DEFAULT_BASE_OUTDIR))
    ap.add_argument("--predictions-dir", default=str(DEFAULT_PREDICTIONS_DIR))
    ap.add_argument("--merged-dir", default=str(DEFAULT_MERGED_DIR), help="Directory holding league __merged.csv files used to build actual result columns.")
    ap.add_argument("--calendar-flags-dir", default=str(DEFAULT_CALENDAR_FLAGS_DIR), help="Directory holding per-window calendar / congestion flag CSVs.")
    ap.add_argument("--skip-calendar-enrich", action="store_true", help="Skip optional calendar flag enrichment of scored deploy CSVs.")
    ap.add_argument("--source-implied-min", default="20", help="Used only to resolve native BOOKIE_IMPxx filenames.")
    ap.add_argument("--run-meta-super-score", action="store_true", help="Run FTR meta super-score walkforward audit after windows complete.")
    ap.add_argument("--meta-outdir", default=None, help="Output directory for meta super-score reports (default: <base>/_MASTER/META)")
    ap.add_argument("--bookie-extra-args", default="", help="Extra args passed to bookie_allmarkets.py")
    ap.add_argument("--deploy-extra-args", default="", help="Extra args passed to deploy_rulebook.py")
    ap.add_argument("--skip-source", action="store_true")
    ap.add_argument("--skip-deploy", action="store_true")
    ap.add_argument("--skip-score", action="store_true")
    ap.add_argument("--score-candidates", action="store_true", help="Also score candidate deploy files (raw/after gates).")
    ap.add_argument("--write-window-file-manifest", action="store_true", help="Write a per-window file manifest of key artifacts.")
    ap.add_argument(
        "--no-tiers-only",
        action="store_true",
        help="Disable tiers-only combined output (allow candidate-union when tiers exist).",
    )
    ap.add_argument(
        "--no-btts-gate-audit",
        action="store_true",
        help="Disable BTTS YES gate audit output from deploy_rulebook.py.",
    )
    ap.add_argument(
        "--no-ou25-gate-audit",
        action="store_true",
        help="Disable OU25 gate audit output from deploy_rulebook.py.",
    )
    ap.add_argument(
        "--calendar-regime",
        default="",
        help="Optional override regime to stamp onto all windows for testing. When omitted, each window uses WINDOW_REGIME_MAP or NORMAL.",
    )
    ap.add_argument("--skip-summary", action="store_true")
    ap.add_argument("--include-acca", action="store_true", help="Run acca builder and acca backtest pipeline for each window")
    ap.add_argument("--acca-stake", type=float, default=10.0, help="Flat stake per acca slip for backtesting")
    ap.add_argument("--acca-templates", default="", help="Optional comma-separated acca templates override, e.g. BTTS_ONLY,OU25_ONLY,FTR_ONLY")
    ap.add_argument("--acca-k-values", default="", help="Optional comma-separated acca k-values override, e.g. 6,8,10")
    ap.add_argument("--acca-slips-per-k", type=int, default=6, help="Acca slips to build per template/k per window")
    return ap.parse_args()


def main() -> None:
    _test_grade_leg()
    args = parse_args()
    include_acca = bool(getattr(args, "include_acca", False))
    acca_stake = float(getattr(args, "acca_stake", 10.0) or 10.0)
    acca_slips_per_k = int(getattr(args, "acca_slips_per_k", 6) or 6)
    acca_templates_cli = str(getattr(args, "acca_templates", "") or "").strip()
    acca_k_values_cli = str(getattr(args, "acca_k_values", "") or "").strip()

    manifest_path = Path(args.manifest)
    base_outdir = Path(args.base_outdir)
    predictions_dir = Path(args.predictions_dir)
    merged_dir = Path(args.merged_dir)
    calendar_flags_dir = Path(args.calendar_flags_dir)
    windows = load_manifest(manifest_path)

    bookie_extra_args = shlex.split(args.bookie_extra_args)
    deploy_extra_args = shlex.split(args.deploy_extra_args)
    acca_window_summary_frames: list[pd.DataFrame] = []
    acca_window_slip_frames: list[pd.DataFrame] = []
    acca_window_leg_frames: list[pd.DataFrame] = []

    run_started_ts = time.time()
    total_windows = len(windows)
    for idx, window in enumerate(windows, start=1):
        log("\n" + "=" * 100)
        elapsed_min = (time.time() - run_started_ts) / 60.0
        log(f"[progress] window {idx}/{total_windows} elapsed_min={elapsed_min:.1f}")
        log(f"WINDOW: {window.window_id}  [{window.date_from} -> {window.date_to}]")
        log("=" * 100)
        resolved_calendar_regime = resolve_calendar_regime(
            window.window_id,
            getattr(args, "calendar_regime", ""),
        )
        paths = window_paths(base_outdir, window)

        combined_deploy_csv = paths["deploy_dir"] / f"DEPLOY_COMBINED_{window.date_from}_to_{window.date_to}.csv"
        scored_csv = paths["scored_dir"] / f"DEPLOY_COMBINED_SCORED_{window.date_from}_to_{window.date_to}.csv"
        scored_calendar_csv = paths["scored_dir"] / f"DEPLOY_COMBINED_SCORED_CALENDAR_{window.date_from}_to_{window.date_to}.csv"
        native_source_csv: Path | None = None
        native_deploy_csvs: list[Path] = []

        source_run_started_ts = 0.0
        if args.skip_source:
            native_source_csv = resolve_source_csv(window, predictions_dir, args.source_implied_min)
        else:
            source_run_started_ts = time.time()
            source_log_path = paths["logs_dir"] / "01_bookie_allmarkets.log"
            run_cmd(
                build_source_command(
                    window,
                    bookie_extra_args,
                    args.source_implied_min,
                    source_outdir=predictions_dir,
                ),
                source_log_path,
            )
            log(f"[log] source command log={source_log_path}")
            try:
                native_source_csv = resolve_source_csv_after_run(
                    window,
                    predictions_dir,
                    args.source_implied_min,
                    source_run_started_ts,
                )
            except FileNotFoundError:
                if _source_log_indicates_no_rows(source_log_path):
                    log(f"[skip] source produced no rows for {window.window_id}; skipping window.")
                    continue
                raise

        source_csv = paths["source_dir"] / native_source_csv.name
        copy_if_needed(native_source_csv, source_csv)
        log(f"[source-copy] native={native_source_csv}")
        log(f"[source-copy] copied={source_csv}")

        if args.skip_source:
            log(f"[log] source command log={paths['logs_dir'] / '01_bookie_allmarkets.log'}")

        source_df_for_actuals = pd.read_csv(source_csv, low_memory=False)
        if source_df_for_actuals.empty:
            log(f"[skip] source CSV empty for {window.window_id}; skipping window.")
            continue
        merged_actuals = load_window_merged_actuals(source_df_for_actuals, merged_dir)
        if merged_actuals.empty:
            log("[merged] no merged actuals resolved for this window; scoring will rely on existing actual_* columns only.")
        else:
            graded_ftr = int(merged_actuals["actual_ftr"].astype("string").fillna("").ne("").sum())
            graded_ou25 = int(pd.to_numeric(merged_actuals["actual_over25"], errors="coerce").notna().sum())
            graded_btts = int(pd.to_numeric(merged_actuals["actual_btts_yes"], errors="coerce").notna().sum())
            log(f"[merged] resolved actual rows: {len(merged_actuals)} from {merged_dir}")
            log(f"[merged] graded actual coverage: ftr={graded_ftr} ou25={graded_ou25} btts={graded_btts}")
            ensure_dir(paths["reports_dir"])
            base_path = write_universe_base_audit(merged_actuals, window, paths["reports_dir"])
            if base_path is not None:
                log(f"[universe-audit] wrote base rates: {base_path}")
            corr_path, thresh_path = write_feature_outcome_audit(
                source_df_for_actuals,
                merged_actuals,
                window,
                paths["reports_dir"],
            )
            if corr_path is not None:
                log(f"[feature-audit] wrote correlations: {corr_path}")
            if thresh_path is not None:
                log(f"[feature-audit] wrote thresholds: {thresh_path}")

        deploy_run_started_ts = 0.0
        if not args.skip_deploy:
            deploy_run_started_ts = time.time()
            deploy_args = list(deploy_extra_args)
            if not bool(getattr(args, "no_btts_gate_audit", False)):
                deploy_args.extend(["--btts-gate-audit", "--window-id", window.window_id])
            if not bool(getattr(args, "no_ou25_gate_audit", False)):
                deploy_args.extend(["--ou25-gate-audit", "--window-id", window.window_id])
            run_cmd(
                build_deploy_command(source_csv, paths["deploy_dir"], deploy_args),
                paths["logs_dir"] / "02_deploy_rulebook.log",
            )
        log(f"[log] deploy command log={paths['logs_dir'] / '02_deploy_rulebook.log'}")

        if args.skip_deploy and combined_deploy_csv.exists():
            native_deploy_csvs = sorted(
                paths["deploy_dir"].glob(f"*{window.date_from}_to_{window.date_to}*DEPLOY_TIER_*.csv")
            )
            if not native_deploy_csvs:
                native_deploy_csvs = resolve_deploy_csvs(window, predictions_dir, native_source_csv)
        else:
            try:
                native_deploy_csvs = (
                    resolve_deploy_csvs_after_run(window, predictions_dir, source_csv, deploy_run_started_ts)
                    if not args.skip_deploy
                    else resolve_deploy_csvs(window, predictions_dir, source_csv)
                )
            except FileNotFoundError:
                native_deploy_csvs = resolve_deploy_csvs(window, predictions_dir, source_csv)
        log(
            f"[resolved] deploy_candidates_count={len(native_deploy_csvs)} "
            f"profile={_infer_requested_ftr_profile_from_source_csv(native_source_csv)}"
        )
        copied_deploy_csvs: list[Path] = []
        for native_csv in native_deploy_csvs:
            dst = paths["deploy_dir"] / native_csv.name
            copy_if_needed(native_csv, dst)
            copied_deploy_csvs.append(dst)

        tiers_only = not bool(getattr(args, "no_tiers_only", False))
        if tiers_only:
            log("[combine] tiers_only=1 (candidate union disabled when tiers exist)")
        else:
            log("[combine] tiers_only=0 (candidate union enabled)")
        combined_deploy_df = combine_deploy_files_with_policy(
            copied_deploy_csvs,
            combined_deploy_csv,
            tiers_only=tiers_only,
        )
        candidate_csvs = [p for p in copied_deploy_csvs if _is_candidate_deploy_filename(p)]
        if not candidate_csvs:
            # Fallback: explicitly pull candidate deploy files even when tier files were preferred.
            explicit_names = [
                "DEPLOY_CANDIDATES_AFTER_GATES.csv",
                "DEPLOY_CANDIDATES_RAW.csv",
            ]
            search_dirs = [
                paths["deploy_dir"],
                native_source_csv.parent.parent / "02_deploy",
                native_source_csv.parent,
            ]
            seen_candidates: set[Path] = set()
            for dir_path in search_dirs:
                if not dir_path.exists() or not dir_path.is_dir():
                    continue
                for name in explicit_names:
                    hit = dir_path / name
                    if not hit.exists() or not hit.is_file():
                        continue
                    resolved = hit.resolve()
                    if resolved in seen_candidates:
                        continue
                    seen_candidates.add(resolved)
                    candidate_csvs.append(hit)
            if candidate_csvs:
                log(f"[resolved] explicit candidate deploy files recovered: {len(candidate_csvs)}")
        if include_acca:
            acca_dir = paths["root"] / "05_acca"
            ensure_dir(acca_dir)

            acca_templates, acca_k_values, acca_slips_per_k = build_acca_window_product_args(
                templates_override=getattr(args, "acca_templates", ""),
                k_values_override=getattr(args, "acca_k_values", ""),
                slips_per_k_override=getattr(args, "acca_slips_per_k", 6),
            )
            run_acca_builder_for_window(
                deploy_input=str(paths["deploy_dir"]),
                out_dir=str(acca_dir),
                tag=window.window_id,
                templates=acca_templates,
                k_values=acca_k_values,
                slips_per_k=acca_slips_per_k,
                top_n_legs=20,
                log_path=paths["logs_dir"] / "acca_builder.log",
            )


        calendar_flags_csv = None
        calendar_flags_df = pd.DataFrame()
        if not args.skip_calendar_enrich:
            calendar_flags_csv = resolve_calendar_flags_csv(window, calendar_flags_dir)
            if calendar_flags_csv is not None:
                calendar_flags_df = pd.read_csv(calendar_flags_csv, low_memory=False)
                log(f"[calendar] resolved flags={calendar_flags_csv}")
            else:
                log(f"[calendar] no calendar flags found for {window.window_id} under {calendar_flags_dir}")

        if not args.skip_score:
            combined_deploy_df = enrich_with_merged_actuals(combined_deploy_df, merged_actuals)
            if not calendar_flags_df.empty:
                combined_deploy_df = enrich_with_calendar_flags(combined_deploy_df, calendar_flags_df)
            scored_df = score_deploy_df(combined_deploy_df, scored_csv)
            scored_df = maybe_stamp_calendar_overlay(scored_df, resolved_calendar_regime)
            scored_df.to_csv(scored_csv, index=False)
            if not calendar_flags_df.empty:
                scored_df.to_csv(scored_calendar_csv, index=False)

            if bool(getattr(args, "score_candidates", False)) and candidate_csvs:
                for cand_path in candidate_csvs:
                    cand_df = pd.read_csv(cand_path, low_memory=False)
                    cand_df = enrich_with_merged_actuals(cand_df, merged_actuals)
                    if not calendar_flags_df.empty:
                        cand_df = enrich_with_calendar_flags(cand_df, calendar_flags_df)
                    cand_scored_name = f"{cand_path.stem}_SCORED_{window.date_from}_to_{window.date_to}.csv"
                    cand_scored_path = paths["scored_dir"] / cand_scored_name
                    cand_scored_df = score_deploy_df(cand_df, cand_scored_path)
                    cand_scored_df = maybe_stamp_calendar_overlay(cand_scored_df, resolved_calendar_regime)
                    cand_scored_df.to_csv(cand_scored_path, index=False)
                    if not calendar_flags_df.empty:
                        cand_scored_calendar = paths["scored_dir"] / f"{cand_path.stem}_SCORED_CALENDAR_{window.date_from}_to_{window.date_to}.csv"
                        cand_scored_df.to_csv(cand_scored_calendar, index=False)
        else:
            scored_df = pd.read_csv(scored_csv, low_memory=False)
            if not calendar_flags_df.empty:
                scored_df = enrich_with_calendar_flags(scored_df, calendar_flags_df)
            scored_df = maybe_stamp_calendar_overlay(scored_df, resolved_calendar_regime)
            scored_df.to_csv(scored_csv, index=False)
            if not calendar_flags_df.empty:
                scored_df.to_csv(scored_calendar_csv, index=False)
        compare_path = write_universe_compare_audit(merged_actuals, scored_df, window, paths["reports_dir"])
        if compare_path is not None:
            log(f"[universe-audit] wrote compare: {compare_path}")

        log(f"[resolved] source={native_source_csv}")
        log(f"[resolved] merged_dir={merged_dir}")
        log(f"[resolved] calendar_flags_dir={calendar_flags_dir}")
        log(f"[resolved] calendar_regime={resolved_calendar_regime}")
        if calendar_flags_csv is not None:
            log(f"[resolved] calendar_flags={calendar_flags_csv}")
        for native_csv in native_deploy_csvs:
            log(f"[resolved] deploy={native_csv}")
        log(f"[combined] deploy={combined_deploy_csv}")
        log(f"[combined] scored={scored_csv}")
        if scored_calendar_csv.exists():
            log(f"[combined] scored_calendar={scored_calendar_csv}")
        if bool(getattr(args, "write_window_file_manifest", False)):
            manifest_rows = []
            def _add_manifest(path: Path, kind: str) -> None:
                if path is None:
                    return
                manifest_rows.append({"kind": kind, "path": str(path)})
            _add_manifest(source_csv, "source")
            for p in copied_deploy_csvs:
                _add_manifest(p, "deploy")
            _add_manifest(combined_deploy_csv, "deploy_combined")
            _add_manifest(scored_csv, "scored_combined")
            if scored_calendar_csv.exists():
                _add_manifest(scored_calendar_csv, "scored_combined_calendar")
            if bool(getattr(args, "score_candidates", False)):
                for p in candidate_csvs:
                    _add_manifest(paths["scored_dir"] / f"{p.stem}_SCORED_{window.date_from}_to_{window.date_to}.csv", "scored_candidate")
                    cand_cal = paths["scored_dir"] / f"{p.stem}_SCORED_CALENDAR_{window.date_from}_to_{window.date_to}.csv"
                    if cand_cal.exists():
                        _add_manifest(cand_cal, "scored_candidate_calendar")
            if manifest_rows:
                manifest_path = paths["reports_dir"] / "WINDOW_FILE_MANIFEST.csv"
                pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
                log(f"[manifest] wrote {manifest_path}")
        log(f"[combined] merged_actual_rows={0 if merged_actuals.empty else len(merged_actuals)}")
        log_gate_top_suppressor(paths["deploy_dir"], window)
        if not scored_df.empty:
            ftr_nonblank = int(safe_series_str(scored_df, "actual_ftr").ne("").sum()) if "actual_ftr" in scored_df.columns else 0
            ou25_nonnull = int(pd.to_numeric(scored_df.get("actual_over25", np.nan), errors="coerce").notna().sum())
            btts_nonnull = int(pd.to_numeric(scored_df.get("actual_btts_yes", np.nan), errors="coerce").notna().sum())
            log(f"[scored] actual coverage in combined deploy: ftr={ftr_nonblank} ou25={ou25_nonnull} btts={btts_nonnull}")

        self_check_info = log_runner_self_check(
            native_source_csv=native_source_csv,
            source_csv=source_csv,
            native_deploy_csvs=native_deploy_csvs,
            combined_deploy_df=combined_deploy_df,
            scored_df=scored_df,
        )
        append_self_check_row(base_outdir, window.window_id, self_check_info)
        if self_check_info.get("status") == "WARN":
            flags = str(self_check_info.get("flags", ""))
            combined_rows = int(self_check_info.get("combined_rows", 0) or 0)
            candidate_only = "CANDIDATE_ONLY" in flags
            if "NO_TIER_FILES" in flags and not candidate_only:
                log("[self-check] FAIL_FAST: missing tier files (NO_TIER_FILES). Exiting with status 2.")
                raise SystemExit(2)
            if "NO_TIER_FILES" in flags and candidate_only and combined_rows <= 0:
                log("[self-check] FAIL_FAST: candidate-only window produced no combined rows. Exiting with status 2.")
                raise SystemExit(2)
            if "NO_TIER_FILES" in flags and candidate_only and combined_rows > 0:
                log(
                    "[self-check] WARN_ONLY: candidate-only window accepted because combined rows were rescued successfully."
                )
        if include_acca:
            acca_dir = paths["root"] / "05_acca"
            acca_legs_csv = acca_dir / f"acca_slip_legs_{window.window_id}.csv"
            acca_slips_csv = acca_dir / f"acca_slips_{window.window_id}.csv"

            if acca_legs_csv.exists() and acca_slips_csv.exists():
                acca_legs_df = pd.read_csv(acca_legs_csv, low_memory=False)
                acca_slips_df = pd.read_csv(acca_slips_csv, low_memory=False)

                acca_results_df = build_acca_results_frame(scored_df)
                acca_graded_legs_df = backtest_acca_legs(acca_legs_df, acca_results_df)
                acca_graded_slips_df = aggregate_acca_slips(
                    acca_graded_legs_df,
                    acca_slips_df,
                    flat_stake_per_slip=acca_stake,
                )
                acca_products_summary_df = summarize_acca_products(
                    acca_graded_slips_df,
                    window_id=window.window_id,
                )

                write_acca_backtest_outputs(
                    acca_graded_legs_df,
                    acca_graded_slips_df,
                    acca_products_summary_df,
                    reports_dir=acca_dir,
                    tag=window.window_id,
                )

                log(f"[acca] wrote graded legs: {acca_dir / f'ACCA_LEGS_GRADED__{window.window_id}.csv'}")
                log(f"[acca] wrote graded slips: {acca_dir / f'ACCA_SLIPS_GRADED__{window.window_id}.csv'}")
                log(f"[acca] wrote product summary: {acca_dir / f'ACCA_PRODUCTS_SUMMARY__{window.window_id}.csv'}")

                if not acca_products_summary_df.empty:
                    acca_window_summary_frames.append(acca_products_summary_df.copy())
                if not acca_graded_slips_df.empty:
                    acca_window_slip_frames.append(acca_graded_slips_df.assign(window_id=window.window_id).copy())
                if not acca_graded_legs_df.empty:
                    acca_window_leg_frames.append(acca_graded_legs_df.assign(window_id=window.window_id).copy())
        if not args.skip_summary:
            tier_summary = summarize_window(scored_df, window.window_id, paths["reports_dir"])
            if isinstance(tier_summary, pd.DataFrame) and "graded" in tier_summary.columns:
                graded_series = pd.to_numeric(tier_summary["graded"], errors="coerce").fillna(0)
            else:
                graded_series = pd.Series([0])
            graded_window_count = int(graded_series.sum())
            if graded_window_count == 0:
                log(f"[summary] window {window.window_id} has zero graded results; keeping scorecard rows but excluding it from graded rollups.")
            append_to_master_scorecard(base_outdir, tier_summary)

    if acca_window_summary_frames:
        master_acca_window_df = pd.concat(acca_window_summary_frames, ignore_index=True, sort=False)
        master_acca_dir = base_outdir / "acca_backtests"
        all_windows_path, product_summary_path = write_master_acca_backtest_outputs(
            master_acca_window_df,
            out_dir=master_acca_dir,
        )
        log(f"[acca] wrote all-window summary: {all_windows_path}")
        log(f"[acca] wrote product summary: {product_summary_path}")

    if acca_window_slip_frames:
        master_acca_slips_df = pd.concat(acca_window_slip_frames, ignore_index=True, sort=False)
        master_acca_slips_path = base_outdir / "acca_backtests" / "acca_backtest_slips_all_windows.csv"
        ensure_dir(master_acca_slips_path.parent)
        master_acca_slips_df.to_csv(master_acca_slips_path, index=False)
        log(f"[acca] wrote all-window slip ledger: {master_acca_slips_path}")

    if acca_window_summary_frames and acca_window_slip_frames:
        master_acca_window_df = pd.concat(acca_window_summary_frames, ignore_index=True, sort=False)
        master_acca_slips_df = pd.concat(acca_window_slip_frames, ignore_index=True, sort=False)
        master_acca_legs_df = (
            pd.concat(acca_window_leg_frames, ignore_index=True, sort=False)
            if acca_window_leg_frames else pd.DataFrame()
        )
        master_acca_dir = base_outdir / "acca_backtests"
        ensure_dir(master_acca_dir)

        acca_product_audit_df = build_acca_product_audit(
            master_acca_window_df,
            master_acca_slips_df,
            master_acca_legs_df,
        )
        acca_product_audit_path = master_acca_dir / "acca_backtest_product_audit.csv"
        acca_product_audit_df.to_csv(acca_product_audit_path, index=False)
        log(f"[acca] wrote product audit: {acca_product_audit_path}")

    build_master_rollups(base_outdir, windows)
    aggregate_btts_gate_audits(base_outdir)
    aggregate_ou25_gate_audits(base_outdir)
    build_observe_uplift_report(base_outdir)
    build_odds_audits(base_outdir)
    aggregate_universe_audits(base_outdir)
    aggregate_feature_audits(base_outdir)
    build_counterfactual_uplift(base_outdir)
    if args.run_meta_super_score:
        meta_outdir = Path(args.meta_outdir) if args.meta_outdir else base_outdir / "_MASTER" / "META"
        ensure_dir(meta_outdir)
        meta_cmd = [
            sys.executable,
            "build_ftr_meta_super_score_walkforward.py",
            "--base",
            str(base_outdir),
            "--outdir",
            str(meta_outdir),
            "--manifest",
            str(args.manifest),
            "--merged-dir",
            str(args.merged_dir),
            "--source-implied-min",
            str(args.source_implied_min),
        ]
        run_cmd(meta_cmd, log_path=base_outdir / "_MASTER" / "logs" / "meta_super_score.log")
        log(f"[meta] wrote meta super-score outputs to {meta_outdir}")

        # Post-pass: merge meta scores into tier outputs for each window
        try:
            for window in windows:
                window_dir = base_outdir / window.window_id
                meta_path = meta_outdir / f"FTR_META_SUPER_SCORE__WINDOW_{window.window_id}.csv"
                if not meta_path.exists():
                    continue
                meta_df = pd.read_csv(meta_path)
                if meta_df.empty:
                    continue
                tier_files = sorted((window_dir / "02_deploy").glob("*__DEPLOY_TIER_*.csv"))
                for tf in tier_files:
                    df_t = pd.read_csv(tf)
                    join_cols = [c for c in ["fixture_key", "league", "market"] if c in df_t.columns and c in meta_df.columns]
                    if not join_cols:
                        continue
                    merged = df_t.merge(meta_df, on=join_cols, how="left")
                    merged.to_csv(tf, index=False)
        except Exception as e:
            log(f"[meta] warning: failed to merge meta scores into tier outputs: {e}")

    log("\nDONE.")
    log(f"Master outputs: {base_outdir / '_MASTER'}")
    log("Use --bookie-extra-args and --deploy-extra-args to pass league lists, markets, implied mins, preset, strict, debug, etc.")


if __name__ == "__main__":
    main()
