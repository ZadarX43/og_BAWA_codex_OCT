#!/usr/bin/env python3
"""
build_merged.py

Build per-league merged CSVs into:
  Matches/__merged__/<League_With_Underscores>__merged.csv

Key features:
- Prefers synth odds file if present:
    fd_odds_enriched_synth.csv > fd_odds_enriched.csv > fallback CSVs
- Skips fixtures/prediction/report artifacts
- Sanitises invalid odds placeholders (0.0 / <=1.0001 -> NaN) for all odds-ish cols
- Ensures league + od_source columns exist
- Stamps leak-safe rolling defence/rate fields directly into merged output
- Optional: compute rolling press features directly in merged output
- Optional: write a per-league health summary CSV with --write-report

Usage examples:

  # Build merged for a single league
  python build_merged.py --league "England EFL League 1"

  # Build merged for a set of leagues
  python build_merged.py --leagues "England Championship" "Japan J1" "Brazil Serie A"

  # Build all leagues under Matches/ (skipping __merged__, Upcoming Fixtures)
  python build_merged.py --all

  # Build all + compute rolling press features
  python build_merged.py --all --rolling-press

  # Build recursively (if your season CSVs are nested)
  python build_merged.py --all --recursive

  # Build a league and write a health report
  python build_merged.py --league "England EFL League 1" --write-report

  # Build all and stamp defence/rate features used by specialist audits
  python build_merged.py --all --recursive --rolling-press --write-report --dedupe-report
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any


import numpy as np
import pandas as pd

try:
    from streaks_module import attach_team_rates as _attach_team_rates  # type: ignore
except Exception:
    _attach_team_rates = None


SKIP_DIRS_DEFAULT = {"__merged__", "Upcoming Fixtures"}
SKIP_NAME_SUBSTRINGS = (
    "upcoming", "fixture", "fixtures", "prediction", "predictions", "report",
    "__merged__", "backtest", "audit", "output", "train_multiseason"
)

SEASON_MATCH_FILE_RE = re.compile(
    r"^[a-z0-9-]+-matches-\d{4}-to-\d{4}-stats(?: \(\d+\))?\.csv$",
    flags=re.IGNORECASE,
)
SEASON_SNAPSHOT_MATCH_FILE_RE = re.compile(
    r"^[a-z0-9-]+-matches-\d{4}-to-\d{4}-stats(?: \(\d+\))?__team_snapshot_matchups\.csv$",
    flags=re.IGNORECASE,
)
SEASON_GENERIC_FILE_RE = re.compile(
    r"^[a-z0-9-]+-(matches|teams|players)-\d{4}-to-\d{4}-stats(?: \(\d+\))?\.csv$",
    flags=re.IGNORECASE,
)

DUPLICATE_SUFFIX_RE = re.compile(
    r"\s\(\d+\)(?=(?:__team_snapshot_matchups)?\.csv$)",
    flags=re.IGNORECASE,
)

# Additional regex and date formats
SEASON_YEAR_SPAN_RE = re.compile(r"-(\d{4})-to-(\d{4})-stats(?: \(\d+\))?\.csv$", flags=re.IGNORECASE)
DATE_GMT_FORMATS = (
    "%b %d %Y - %I:%M%p",
    "%b %d %Y - %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
)

# if present, prefer these (in order)
PREFERRED_FILES = (
    "fd_odds_enriched_synth.csv",
    "fd_odds_enriched.csv",
)

ROLLING_PRESS_COLS = (
    "rolling5_home_press_intensity",
    "rolling5_away_press_intensity",
    "rolling5_press_intensity_diff",
    "rolling5_home_press_z",
    "rolling5_away_press_z",
    "rolling5_press_z_diff",
)

PRESS_BASE_COLS = (
    "home_press_intensity",
    "away_press_intensity",
)

SYNTH_ENABLED_LEAGUES = {
    "England Premier League",
    "Scotland Premiership",
}

DEDUPE_REPORT_DEFAULT_NAME = "__merged_dedupe_report__.csv"

ODDS_COL_HINTS = ("odds", "B365", "Avg", "Max", "PC", "P>", "P<", "BTTS")

ROLLING_DEFENCE_BASE_COLS = (
    "conceded_rate_5_home",
    "conceded_rate_5_away",
    "clean_sheet_rate_5_home",
    "clean_sheet_rate_5_away",
    "xg_against_avg_5_home",
    "xg_against_avg_5_away",
)

ROLLING_DEFENCE_DERIVED_COLS = (
    "rolling5_home_gc",
    "rolling5_away_gc",
    "gapm_diff",
    "clean_sheet_rate_diff",
    "home_xg_against_idx",
    "away_xg_against_idx",
    "defence_diff",
)

SNAPSHOT_RAW_BACKFILL_COLS = (
    "odds_ft_home_team_win",
    "odds_ft_draw",
    "odds_ft_away_team_win",
    "odds_ft_over25",
    "odds_ft_under25",
    "odds_btts_yes",
    "odds_btts_no",
)


def tagify(name: str) -> str:
    return str(name).strip().replace(" ", "_")


def clean_duplicate_suffix(filename: str) -> str:
    return DUPLICATE_SUFFIX_RE.sub("", str(filename).strip())


def canonical_filename(filename: str) -> str:
    return clean_duplicate_suffix(os.path.basename(str(filename))).lower()


# --- new: season_sort_key and parse_date_gmt_series ---
def season_sort_key(path: str) -> tuple[int, int, str]:
    """Sort season exports by start/end year, then canonical filename.

    This keeps season files in chronological order regardless of macOS duplicate
    suffixes like " (1)" / " (2)".
    """
    base = canonical_filename(path)
    m = SEASON_YEAR_SPAN_RE.search(base)
    if not m:
        return (9999, 9999, base)
    return (int(m.group(1)), int(m.group(2)), base)


def parse_date_gmt_series(series: pd.Series) -> pd.Series:
    """Parse common FootyStats `date_GMT` formats without noisy inference warnings."""
    s = series.astype("string").fillna("").str.strip()
    out = pd.Series(pd.NaT, index=series.index)

    for fmt in DATE_GMT_FORMATS:
        remaining = out.isna() & s.ne("")
        if not remaining.any():
            break
        parsed = pd.to_datetime(s.loc[remaining], format=fmt, errors="coerce")
        out.loc[remaining] = parsed

    # Final permissive fallback for odd edge cases.
    remaining = out.isna() & s.ne("")
    if remaining.any():
        out.loc[remaining] = pd.to_datetime(s.loc[remaining], errors="coerce")

    return out


def is_season_csv_name(filename: str, require_matches: bool = True) -> bool:
    name = canonical_filename(filename)
    if require_matches:
        return (
            SEASON_MATCH_FILE_RE.match(name) is not None
            or SEASON_SNAPSHOT_MATCH_FILE_RE.match(name) is not None
        )
    return SEASON_GENERIC_FILE_RE.match(name) is not None


def season_identity_key(filename: str) -> str:
    """Return a stable logical season key shared by base and snapshot exports."""
    name = canonical_filename(filename)
    name = re.sub(r"__team_snapshot_matchups\.csv$", ".csv", name, flags=re.IGNORECASE)
    return name


# --- module-level team slugifier for stable deploy keys ---
def _slugify_team_name(s: pd.Series) -> pd.Series:
    """Slugify team names to match deploy/backtest fixture_key style.

    IMPORTANT: Do NOT lowercase. Existing deploy fixture_keys preserve original casing
    (e.g. 2024_09_14_Inter_Miami_Philadelphia_Union). If we lowercase here, truth keys
    will never match deploy keys.

    Rules:
    - strip
    - replace any run of non-alphanumeric characters with '_'
    - collapse multiple underscores
    - strip leading/trailing underscores

    This intentionally turns accented chars into '_' (e.g. 'Víkingur' -> 'V_kingur'),
    matching the historical deploy-style keys in the repo.
    """
    s = s.astype("string").fillna("").str.strip()
    s = s.str.replace(r"[^A-Za-z0-9]+", "_", regex=True)
    s = s.str.replace(r"_+", "_", regex=True)
    s = s.str.strip("_")
    return s.astype("string")


def should_skip_path(path: str) -> bool:
    n = canonical_filename(path)

    if any(k in n for k in SKIP_NAME_SUBSTRINGS):
        return True

    # hard-exclude known junk / compatibility artefacts
    if n in {
        "matches.csv",
        "teams.csv",
        "players.csv",
        "__train_multiseason_completed.csv",
        "fd_odds_enriched.csv",
        "fd_odds_enriched_synth.csv",
        "fd_ou25_novig.csv",
        "odds_synth_tables.json",
    }:
        return True

    # only allow true season exports when we are in fallback mode
    if n.endswith(".csv") and not is_season_csv_name(n, require_matches=True):
        return True

    return False


def list_league_folders(matches_root: str, skip_dirs: set[str]) -> List[Tuple[str, str]]:
    out = []
    for name in sorted(os.listdir(matches_root)):
        if name.startswith("."):
            continue
        if name in skip_dirs:
            continue
        p = os.path.join(matches_root, name)
        if os.path.isdir(p):
            out.append((name, p))
    return out


def pick_input_csvs(league_dir: str, recursive: bool) -> List[str]:
    # Only take true season match exports (optionally recursive).
    # Prefer the per-season snapshot-enriched export when it exists for a season,
    # otherwise fall back to the raw per-season match export.
    if recursive:
        candidates = glob.glob(os.path.join(league_dir, "**", "*.csv"), recursive=True)
    else:
        candidates = glob.glob(os.path.join(league_dir, "*.csv"))

    candidates = sorted(candidates, key=season_sort_key)

    chosen_by_season: Dict[str, str] = {}

    for c in candidates:
        base = os.path.basename(c)
        if should_skip_path(base):
            continue

        if not is_season_csv_name(base, require_matches=True):
            continue

        season_key = season_identity_key(base)
        prev = chosen_by_season.get(season_key)
        cur_is_snapshot = "__team_snapshot_matchups.csv" in canonical_filename(base)

        if prev is None:
            chosen_by_season[season_key] = c
            continue

        prev_is_snapshot = "__team_snapshot_matchups.csv" in canonical_filename(os.path.basename(prev))

        # Prefer snapshot-enriched season exports over raw season exports.
        if cur_is_snapshot and not prev_is_snapshot:
            chosen_by_season[season_key] = c

    csvs = [chosen_by_season[k] for k in sorted(chosen_by_season.keys(), key=season_sort_key)]

    return csvs


def ensure_league_and_source(df: pd.DataFrame, league_name: str) -> pd.DataFrame:
    if "league" not in df.columns:
        df["league"] = league_name
    else:
        # fill blanks
        df["league"] = df["league"].astype("string").fillna("").str.strip()
        df.loc[df["league"] == "", "league"] = league_name

    if "od_source" not in df.columns:
        df["od_source"] = "*"
    else:
        df["od_source"] = df["od_source"].astype("string").fillna("").str.strip()
        df.loc[df["od_source"] == "", "od_source"] = "*"

    return df


def sanitize_odds_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert invalid odds placeholders to NaN:
    - 0.0, 1.0, 1.0001 etc are not valid decimal odds for our purposes
    Applies to columns that look like odds columns.
    """
    derived_safe = {
        "draw_implied",
        "implied_prob_diff",
        "odds_parity",
        "odds_skew",
        "odds_diff",
    }
    cols = []
    for c in df.columns:
        cs = str(c)
        if c in derived_safe:
            continue
        if any(h in cs for h in ODDS_COL_HINTS):
            cols.append(c)

    if not cols:
        return df

    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        s = s.where(s > 1.0001, np.nan)
        df[c] = s
    return df


def _snapshot_raw_csv_path(snapshot_csv_path: str) -> Optional[Path]:
    snap = Path(snapshot_csv_path)
    if "__team_snapshot_matchups.csv" not in snap.name:
        return None
    raw_name = snap.name.replace("__team_snapshot_matchups.csv", ".csv")
    raw_path = snap.with_name(raw_name)
    if not raw_path.exists():
        return None
    return raw_path


def _snapshot_join_key(df: pd.DataFrame) -> Optional[pd.Series]:
    required = {"home_team_name", "away_team_name"}
    if not required.issubset(df.columns):
        return None

    home = df["home_team_name"].astype("string").fillna("").str.strip().str.lower()
    away = df["away_team_name"].astype("string").fillna("").str.strip().str.lower()

    date_col = None
    for candidate in ("fixture_key", "date_GMT", "match_date", "timestamp"):
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is None:
        return home + "||" + away

    date_part = df[date_col].astype("string").fillna("").str.strip()
    return date_part + "||" + home + "||" + away


def backfill_snapshot_odds_from_raw(df: pd.DataFrame, csv_path: str) -> pd.DataFrame:
    raw_path = _snapshot_raw_csv_path(csv_path)
    if raw_path is None:
        return df

    snap_key = _snapshot_join_key(df)
    if snap_key is None:
        return df

    wanted_cols = [
        c
        for c in dict.fromkeys(
            ["fixture_key", "date_GMT", "match_date", "timestamp", "home_team_name", "away_team_name"]
            + list(SNAPSHOT_RAW_BACKFILL_COLS)
        )
        if c in pd.read_csv(raw_path, nrows=0).columns
    ]
    if not wanted_cols:
        return df

    raw_df = pd.read_csv(raw_path, usecols=wanted_cols, low_memory=False)
    raw_key = _snapshot_join_key(raw_df)
    if raw_key is None:
        return df

    raw_df = raw_df.assign(__snapshot_join_key=raw_key)
    raw_df = raw_df.drop_duplicates(subset=["__snapshot_join_key"], keep="last")

    raw_subset_cols = ["__snapshot_join_key"] + [c for c in SNAPSHOT_RAW_BACKFILL_COLS if c in raw_df.columns]
    if len(raw_subset_cols) == 1:
        return df

    merged = df.assign(__snapshot_join_key=snap_key).merge(
        raw_df[raw_subset_cols],
        how="left",
        on="__snapshot_join_key",
        suffixes=("", "__raw"),
    )

    fill_summary: List[str] = []
    fill_counts: Dict[str, int] = {}
    for col in SNAPSHOT_RAW_BACKFILL_COLS:
        raw_col = f"{col}__raw"
        if raw_col not in merged.columns:
            continue

        current = pd.to_numeric(merged[col], errors="coerce") if col in merged.columns else pd.Series(np.nan, index=merged.index)
        raw_vals = pd.to_numeric(merged[raw_col], errors="coerce")
        fill_mask = raw_vals.gt(1.0001) & ~current.gt(1.0001)
        if not fill_mask.any():
            continue

        if col not in merged.columns:
            merged[col] = np.nan
            current = pd.to_numeric(merged[col], errors="coerce")

        merged.loc[fill_mask, col] = raw_vals.loc[fill_mask]
        fill_count = int(fill_mask.sum())
        fill_summary.append(f"{col}={fill_count}")
        fill_counts[col] = fill_count

    merged = merged.drop(columns=["__snapshot_join_key"] + [c for c in merged.columns if c.endswith("__raw")], errors="ignore")
    merged.attrs["snapshot_backfill_counts"] = fill_counts
    merged.attrs["snapshot_backfill_source"] = os.path.basename(csv_path)

    if fill_summary:
        print(
            "  ↺ backfilled snapshot odds from raw:",
            os.path.basename(csv_path),
            "|",
            ", ".join(fill_summary),
        )

    return merged


# --- helper: promote synth under25 odds into canonical under25 ---
def promote_under25_from_synth(df: pd.DataFrame) -> pd.DataFrame:
    """Promote synth Under 2.5 odds into canonical column when canonical is missing.

    Why this exists:
    - Some leagues only have OU25 as Over 2.5 odds historically.
    - `odds_synth.py apply` creates `odds_ft_under25_synth` as a usable estimate.
    - Downstream code expects canonical `odds_ft_under25`.

    Behaviour:
    1) If canonical `odds_ft_under25` is missing and synth exists, fill canonical from synth.
    2) Maintain `odds_source_under25` provenance:
       - If we filled from synth: `synth_ou25`
       - If canonical already existed but matches synth (within eps) and source is blank: `synth_ou25`
       - Otherwise, if canonical exists and source is blank: `orig`

    This avoids the confusing case where canonical is already pre-filled (e.g. by prior steps)
    but we still want to label it as synth-derived.
    """
    if "odds_ft_under25_synth" not in df.columns:
        return df

    # ensure canonical exists
    if "odds_ft_under25" not in df.columns:
        df["odds_ft_under25"] = np.nan

    if "odds_source_under25" not in df.columns:
        df["odds_source_under25"] = pd.Series("", index=df.index, dtype="string")
    else:
        df["odds_source_under25"] = df["odds_source_under25"].astype("string").fillna("").str.strip()

    u25 = pd.to_numeric(df["odds_ft_under25"], errors="coerce")
    u25s = pd.to_numeric(df["odds_ft_under25_synth"], errors="coerce")

    # normalise source strings
    src = df["odds_source_under25"].astype("string").fillna("").str.strip()

    # (1) Fill missing canonical from synth
    fill_mask = u25.isna() & u25s.notna()
    if fill_mask.any():
        df.loc[fill_mask, "odds_ft_under25"] = u25s.loc[fill_mask]
        df.loc[fill_mask, "odds_source_under25"] = "synth_ou25"

    # (2) If canonical already exists and synth exists, but source is blank,
    # label as synth when they match closely (covers cases where canonical was pre-filled earlier)
    eps = 1e-6
    u25_after = pd.to_numeric(df["odds_ft_under25"], errors="coerce")
    src_after = df["odds_source_under25"].astype("string").fillna("").str.strip()

    match_mask = (
        u25_after.notna()
        & u25s.notna()
        & (src_after == "")
        & ((u25_after - u25s).abs() <= eps)
    )
    if match_mask.any():
        df.loc[match_mask, "odds_source_under25"] = "synth_ou25"

    # (3) Anything else with canonical present but no source gets 'orig'
    src_final = df["odds_source_under25"].astype("string").fillna("").str.strip()
    have_u25 = pd.to_numeric(df["odds_ft_under25"], errors="coerce").notna()
    df.loc[have_u25 & (src_final == ""), "odds_source_under25"] = "orig"

    return df



def parse_dt(df: pd.DataFrame) -> pd.Series:
    """
    Best-effort datetime parser with fallback priority:
      1) match_date
      2) date_GMT
      3) timestamp (unix seconds)

    Returns a Series aligned to df.index.
    """
    out = pd.Series(pd.NaT, index=df.index)

    if "match_date" in df.columns:
        dt = pd.to_datetime(df["match_date"], errors="coerce")
        out = out.fillna(dt)

    if "date_GMT" in df.columns:
        dt = parse_date_gmt_series(df["date_GMT"])
        out = out.fillna(dt)

    if "timestamp" in df.columns:
        ts_num = pd.to_numeric(df["timestamp"], errors="coerce")
        dt = pd.to_datetime(ts_num, unit="s", errors="coerce")
        out = out.fillna(dt)

    return out


# --- helper: ensure match_date, fixture_key, __src_csv ---
def ensure_match_date_and_fixture_key(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common audit/debug keys across leagues.

    Adds/repairs:
    - `match_date`: populated from the best available parsed datetime (usually `date_GMT`)
    - `fixture_key`: deterministic key from deploy/backtest-compatible format
    - `__src_csv`: alias of `__src` when present (for compatibility with older audits)

    Notes:
    - We keep `date_GMT` as-is and do not force timezone conversions.
    - `fixture_key` is only created when home/away columns are available.
    - Existing non-blank values are preserved.
    """
    df = df.copy()

    dt = parse_dt(df)

    # 1) ensure match_date exists and backfill it from the parsed datetime.
    # Use normalized date-only form so fixture keys stay stable across feeds.
    if "match_date" not in df.columns:
        df["match_date"] = pd.Series(pd.NA, index=df.index, dtype="string")

    match_date_existing = df["match_date"].astype("string").fillna("").str.strip()
    dt_date_only = dt.dt.strftime("%Y-%m-%d").astype("string")
    fill_match_date = (match_date_existing == "") & dt.notna()
    if fill_match_date.any():
        df.loc[fill_match_date, "match_date"] = dt_date_only.loc[fill_match_date]

    # 2) add __src_csv alias for audit scripts that expect it
    if "__src_csv" not in df.columns and "__src" in df.columns:
        df["__src_csv"] = df["__src"]

    # 3) ensure fixture_key exists when we have enough columns
    if {"home_team_name", "away_team_name"}.issubset(df.columns):
        if "fixture_key" not in df.columns:
            df["fixture_key"] = pd.Series(pd.NA, index=df.index, dtype="string")

        fk_existing = df["fixture_key"].astype("string").fillna("").str.strip()
        # Build deploy-style fixture_key from normalized date + home + away.
        # Prefer the parsed datetime fallback chain so we recover keys even when
        # historical rows have blank match_date but valid date_GMT / timestamp.
        dt_date = dt.dt.strftime("%Y_%m_%d").astype("string").fillna("")

        home_slug = _slugify_team_name(df["home_team_name"])
        away_slug = _slugify_team_name(df["away_team_name"])
        fk_new = (dt_date + "_" + home_slug + "_" + away_slug).astype("string")
        fill_fk = (
            (fk_existing == "")
            & (dt_date != "")
            & (home_slug != "")
            & (away_slug != "")
        )
        if fill_fk.any():
            df.loc[fill_fk, "fixture_key"] = fk_new.loc[fill_fk]

    return df


def ensure_match_id(df: pd.DataFrame) -> pd.DataFrame:
    if "match_id" in df.columns:
        return df
    if {"home_team_name", "away_team_name"}.issubset(df.columns):
        dt = parse_dt(df).dt.strftime("%Y_%m_%d").astype("string").fillna("")
        home_slug = _slugify_team_name(df["home_team_name"])
        away_slug = _slugify_team_name(df["away_team_name"])
        key = dt + "_" + home_slug + "_" + away_slug
        df["match_id"] = key.factorize()[0]
        return df
    df["match_id"] = pd.RangeIndex(0, len(df))
    return df


def zscore(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    mu = float(v.mean(skipna=True)) if v.notna().any() else 0.0
    sd = float(v.std(ddof=0, skipna=True)) if v.notna().any() else 0.0
    if not np.isfinite(sd) or sd == 0.0:
        return pd.Series(0.0, index=v.index)
    out = (v - mu) / sd
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _coerce_num_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def add_team_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach leak-safe rolling team-rate fields from `streaks_module.attach_team_rates`
    when available. This stamps the merged dataframe with the rolling concession /
    clean-sheet / xG-against ingredients needed by downstream specialist defence logic.
    """
    if not callable(_attach_team_rates):
        return df

    try:
        out = _attach_team_rates(df.copy(), lookbacks=(5, 10))
        return out if isinstance(out, pd.DataFrame) else df
    except Exception:
        return df


def add_defence_layer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build merged-level defence features from leak-safe rolling team-rate inputs.

    Produces, when ingredients are available:
      - rolling5_home_gc / rolling5_away_gc
      - gapm_diff
      - clean_sheet_rate_diff
      - home_xg_against_idx / away_xg_against_idx
      - defence_diff

    Notes:
    - `rolling5_*_gc` is reconstructed from conceded-rate when explicit rolling GC is absent.
    - `home_xg_against_idx` / `away_xg_against_idx` are simple aliases of the rolling
      xG-against averages so downstream code can use the canonical names it already expects.
    - `defence_diff` is signed as home_defence_strength - away_defence_strength, where
      stronger defence means fewer recent concessions, stronger clean-sheet rate, and lower
      xG against.
    """
    out = df.copy()

    conceded_h = _coerce_num_series(out, "conceded_rate_5_home")
    conceded_a = _coerce_num_series(out, "conceded_rate_5_away")
    cs_h = _coerce_num_series(out, "clean_sheet_rate_5_home")
    cs_a = _coerce_num_series(out, "clean_sheet_rate_5_away")
    xga_h = _coerce_num_series(out, "xg_against_avg_5_home")
    xga_a = _coerce_num_series(out, "xg_against_avg_5_away")

    if "rolling5_home_gc" not in out.columns:
        out["rolling5_home_gc"] = conceded_h * 5.0
    else:
        out["rolling5_home_gc"] = pd.to_numeric(out["rolling5_home_gc"], errors="coerce")

    if "rolling5_away_gc" not in out.columns:
        out["rolling5_away_gc"] = conceded_a * 5.0
    else:
        out["rolling5_away_gc"] = pd.to_numeric(out["rolling5_away_gc"], errors="coerce")

    home_gc = pd.to_numeric(out["rolling5_home_gc"], errors="coerce")
    away_gc = pd.to_numeric(out["rolling5_away_gc"], errors="coerce")

    out["gapm_diff"] = (home_gc - away_gc) / 5.0
    out["clean_sheet_rate_diff"] = cs_h - cs_a

    out["home_xg_against_idx"] = xga_h
    out["away_xg_against_idx"] = xga_a

    home_defence_strength = (
        (-home_gc / 5.0)
        + cs_h.fillna(0.0)
        - xga_h.fillna(0.0)
    )
    away_defence_strength = (
        (-away_gc / 5.0)
        + cs_a.fillna(0.0)
        - xga_a.fillna(0.0)
    )
    out["defence_diff"] = home_defence_strength - away_defence_strength

    return out


def add_rolling_press_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Computes:
      rolling5_home_press_intensity
      rolling5_away_press_intensity
      rolling5_press_intensity_diff
      rolling5_home_press_z
      rolling5_away_press_z
      rolling5_press_z_diff
    Uses shift(1) to prevent leakage (current match not included in its own rolling stats).
    """
    df = df.copy()
    df = ensure_match_id(df)
    dt = parse_dt(df)
    df["_dt"] = dt

    if not {"home_team_name", "away_team_name"}.issubset(df.columns):
        # cannot roll safely
        return df

    # fallback intensity from baseline if present
    if "home_press_intensity" not in df.columns and "home_press_baseline" in df.columns:
        df["home_press_intensity"] = df["home_press_baseline"]
    if "away_press_intensity" not in df.columns and "away_press_baseline" in df.columns:
        df["away_press_intensity"] = df["away_press_baseline"]

    if "home_press_intensity" not in df.columns or "away_press_intensity" not in df.columns:
        return df

    df["_dt_rank"] = df["_dt"].fillna(pd.Timestamp.max)
    df = df.sort_values(["_dt_rank", "match_id"], ascending=True).reset_index(drop=True)

    w = int(window)

    df["rolling5_home_press_intensity"] = (
        df.groupby("home_team_name")["home_press_intensity"]
          .apply(lambda s: pd.to_numeric(s, errors="coerce").shift(1).rolling(w, min_periods=1).mean())
          .reset_index(level=0, drop=True)
    )
    df["rolling5_away_press_intensity"] = (
        df.groupby("away_team_name")["away_press_intensity"]
          .apply(lambda s: pd.to_numeric(s, errors="coerce").shift(1).rolling(w, min_periods=1).mean())
          .reset_index(level=0, drop=True)
    )

    df["rolling5_press_intensity_diff"] = df["rolling5_home_press_intensity"] - df["rolling5_away_press_intensity"]
    df["rolling5_home_press_z"] = zscore(df["rolling5_home_press_intensity"])
    df["rolling5_away_press_z"] = zscore(df["rolling5_away_press_intensity"])
    df["rolling5_press_z_diff"] = df["rolling5_home_press_z"] - df["rolling5_away_press_z"]

    df = df.drop(columns=["_dt", "_dt_rank"], errors="ignore")
    return df


def build_report_row(
    league_name: str,
    merged_path: str,
    df: pd.DataFrame,
    input_files: List[str],
    rolling_press_requested: bool,
    snapshot_backfill_counts: Optional[Dict[str, int]] = None,
) -> dict:
    """Create a compact health row for this league's merged dataframe."""
    n_rows = int(len(df))
    n_cols = int(len(df.columns))

    # timeline coverage
    dt = parse_dt(df)
    timeline_nonnull = int(dt.notna().sum()) if dt is not None else 0
    timeline_rate = float(timeline_nonnull / n_rows) if n_rows else 0.0

    def nn_rate(col: str) -> tuple[int, float]:
        if col not in df.columns:
            return 0, 0.0

        if col in {"fixture_key", "home_team_name", "away_team_name", "status", "league", "od_source", "odds_source_under25", "match_date", "__src", "__src_csv"}:
            nn = int(df[col].astype("string").fillna("").str.strip().ne("").sum())
        else:
            if pd.api.types.is_numeric_dtype(df[col]):
                nn = int(pd.to_numeric(df[col], errors="coerce").notna().sum())
            else:
                nn = int(df[col].notna().sum())

        rate = float(nn / n_rows) if n_rows else 0.0
        return nn, rate

    # Under 2.5 coverage (canonical + synth)
    u25_nn, u25_rate = nn_rate("odds_ft_under25")
    u25s_nn, u25s_rate = nn_rate("odds_ft_under25_synth")

    # Over 2.5 coverage (to catch known_missing blocks)
    o25_nn, o25_rate = nn_rate("odds_ft_over25")

    # Rolling press coverage (if requested)
    press_base_present = sum(1 for c in PRESS_BASE_COLS if c in df.columns)
    press_base_min_rate = None
    if press_base_present:
        rates = []
        for c in PRESS_BASE_COLS:
            if c in df.columns:
                _, r = nn_rate(c)
                rates.append(r)
        press_base_min_rate = float(min(rates)) if rates else 0.0

    roll_present = sum(1 for c in ROLLING_PRESS_COLS if c in df.columns)
    roll_min_rate = None
    if roll_present:
        rates = []
        for c in ROLLING_PRESS_COLS:
            if c in df.columns:
                _, r = nn_rate(c)
                rates.append(r)
        roll_min_rate = float(min(rates)) if rates else 0.0

    backfill_counts = snapshot_backfill_counts or {}
    backfill_total = int(sum(backfill_counts.values()))
    backfill_cols = "|".join(f"{k}:{v}" for k, v in sorted(backfill_counts.items())) if backfill_counts else ""

    warnings: List[str] = []
    if backfill_total > 0:
        warnings.append(f"snapshot_backfill_rescue={backfill_total}")
    if o25_rate == 0.0:
        warnings.append("over25_missing")
    if u25_rate == 0.0 and u25s_rate == 0.0:
        warnings.append("under25_missing")
    if rolling_press_requested and (roll_present == 0 or (roll_min_rate is not None and roll_min_rate == 0.0)):
        warnings.append("rolling_press_missing")
    elif rolling_press_requested and roll_min_rate is not None and roll_min_rate < 0.5:
        warnings.append(f"rolling_press_sparse={round(roll_min_rate, 3)}")

    defence_base_present = sum(1 for c in ROLLING_DEFENCE_BASE_COLS if c in df.columns)
    defence_base_min_rate = None
    if defence_base_present:
        rates = []
        for c in ROLLING_DEFENCE_BASE_COLS:
            if c in df.columns:
                _, r = nn_rate(c)
                rates.append(r)
        defence_base_min_rate = float(min(rates)) if rates else 0.0

    defence_derived_present = sum(1 for c in ROLLING_DEFENCE_DERIVED_COLS if c in df.columns)
    defence_derived_min_rate = None
    if defence_derived_present:
        rates = []
        for c in ROLLING_DEFENCE_DERIVED_COLS:
            if c in df.columns:
                _, r = nn_rate(c)
                rates.append(r)
        defence_derived_min_rate = float(min(rates)) if rates else 0.0

    row = {
        "league": league_name,
        "merged_path": os.path.basename(merged_path),
        "rows": n_rows,
        "cols": n_cols,
        "inputs": "|".join([os.path.basename(x) for x in input_files]),
        "timeline_nonnull": timeline_nonnull,
        "timeline_rate": round(timeline_rate, 6),
        "press_base_cols_present": int(press_base_present),
        "press_base_min_rate": (round(press_base_min_rate, 6) if press_base_min_rate is not None else ""),
        "over25_nn": o25_nn,
        "over25_rate": round(o25_rate, 6),
        "under25_nn": u25_nn,
        "under25_rate": round(u25_rate, 6),
        "under25_synth_nn": u25s_nn,
        "under25_synth_rate": round(u25s_rate, 6),
        "snapshot_backfill_total": backfill_total,
        "snapshot_backfill_cols": backfill_cols,
        "rolling_press_requested": int(bool(rolling_press_requested)),
        "rolling_press_cols_present": int(roll_present),
        "rolling_press_min_rate": (round(roll_min_rate, 6) if roll_min_rate is not None else ""),
        "defence_base_cols_present": int(defence_base_present),
        "defence_base_min_rate": (round(defence_base_min_rate, 6) if defence_base_min_rate is not None else ""),
        "defence_derived_cols_present": int(defence_derived_present),
        "defence_derived_min_rate": (round(defence_derived_min_rate, 6) if defence_derived_min_rate is not None else ""),
        "warnings": "|".join(warnings),
    }
    return row


def build_dedupe_report_row(
    league_name: str,
    merged_path: str,
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """Create a compact duplicate audit row for a merged dataframe."""
    row: Dict[str, Any] = {
        "league": league_name,
        "merged_path": os.path.basename(merged_path),
        "exists": True,
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
    }

    if "fixture_key" in df.columns:
        fk = df["fixture_key"].astype("string").fillna("").str.strip()
        row["fixture_key_nonblank"] = int(fk.ne("").sum())
        row["fixture_key_dup_rows"] = int(fk[fk.ne("") & fk.duplicated(keep=False)].shape[0])
    else:
        row["fixture_key_nonblank"] = 0
        row["fixture_key_dup_rows"] = 0

    row["exact_dup_rows"] = int(df.duplicated(keep=False).sum())

    if {"home_team_name", "away_team_name"}.issubset(df.columns):
        dt = parse_dt(df)
        date_key = dt.dt.strftime("%Y-%m-%d").astype("string").fillna("")
        home = df["home_team_name"].astype("string").fillna("").str.strip()
        away = df["away_team_name"].astype("string").fillna("").str.strip()
        combo = date_key + "||" + home + "||" + away
        dup_mask = date_key.ne("") & home.ne("") & away.ne("") & combo.duplicated(keep=False)
        row["home_away_date_dup_rows"] = int(dup_mask.sum())
    else:
        row["home_away_date_dup_rows"] = 0

    return row


def build_missing_dedupe_report_row(league_name: str, merged_path: str) -> Dict[str, Any]:
    return {
        "league": league_name,
        "merged_path": os.path.basename(merged_path),
        "exists": False,
        "rows": 0,
        "cols": 0,
        "fixture_key_nonblank": 0,
        "fixture_key_dup_rows": 0,
        "exact_dup_rows": 0,
        "home_away_date_dup_rows": 0,
    }


def build_one_league(
    matches_root: str,
    merged_root: str,
    league_name: str,
    recursive: bool,
    rolling_press: bool,
    window: int,
) -> Tuple[bool, Optional[dict], Optional[dict]]:
    league_dir = os.path.join(matches_root, league_name)
    if not os.path.isdir(league_dir):
        print("❌ missing folder:", league_dir)
        return False, None, None

    csvs = pick_input_csvs(league_dir, recursive=recursive)
    if not csvs:
        print("❌ no csvs found:", league_name)
        return False, None, None

    dfs = []
    snapshot_backfill_counts: Dict[str, int] = {}
    for fp in csvs:
        try:
            df = pd.read_csv(fp, low_memory=False)
            df = backfill_snapshot_odds_from_raw(df, fp)
            for col, count in (df.attrs.get("snapshot_backfill_counts") or {}).items():
                snapshot_backfill_counts[col] = snapshot_backfill_counts.get(col, 0) + int(count)
            if len(df) > 0:
                df["__src"] = os.path.basename(fp)
                dfs.append(df)
        except Exception as e:
            print("  ⚠️ skipped:", os.path.basename(fp), "|", e)

    if not dfs:
        print("❌ could not read any csvs:", league_name)
        return False, None, None

    df = pd.concat(dfs, ignore_index=True, sort=False)

    # remove exact duplicate rows that sometimes appear inside raw season exports
    df = df.drop_duplicates().reset_index(drop=True)
    # Raw cup / European exports sometimes contain the exact same fixture twice
    # inside a single season file. After we backfill `match_date` and `fixture_key`
    # below, we will drop duplicate fixture rows deterministically.

    df = ensure_league_and_source(df, league_name)
    df = sanitize_odds_placeholders(df)

    # promote synth under25 into canonical under25 when needed
    df = promote_under25_from_synth(df)

    # normalize key audit/debug columns
    df = ensure_match_date_and_fixture_key(df)

    # drop duplicated fixtures after key repair/backfill
    dedupe_keys = [c for c in ["fixture_key", "match_date", "home_team_name", "away_team_name"] if c in df.columns]
    if "fixture_key" in dedupe_keys:
        fk = df["fixture_key"].astype("string").fillna("").str.strip()
        nonblank_fk = fk.ne("")
        if nonblank_fk.any():
            keep_nonblank = df.loc[nonblank_fk].drop_duplicates(subset=["fixture_key"], keep="first")
            keep_blank = df.loc[~nonblank_fk]
            df = pd.concat([keep_nonblank, keep_blank], ignore_index=True, sort=False)
    elif len(dedupe_keys) >= 3:
        df = df.drop_duplicates(subset=dedupe_keys, keep="first").reset_index(drop=True)

    # attach leak-safe rolling team-rate ingredients used by specialist defence logic
    df = add_team_rate_features(df)
    df = add_defence_layer_features(df)

    # optional rolling press features
    if rolling_press:
        df = add_rolling_press_features(df, window=window)

    out_path = os.path.join(merged_root, f"{tagify(league_name)}__merged.csv")
    Path(merged_root).mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    picked = ", ".join([clean_duplicate_suffix(os.path.basename(x)) for x in csvs])
    print(f"✅ merged: {league_name} | rows={len(df)} cols={len(df.columns)} | inputs={picked} -> {os.path.basename(out_path)}")

    report = build_report_row(
        league_name=league_name,
        merged_path=out_path,
        df=df,
        input_files=csvs,
        rolling_press_requested=rolling_press,
        snapshot_backfill_counts=snapshot_backfill_counts,
    )
    dedupe_report = build_dedupe_report_row(
        league_name=league_name,
        merged_path=out_path,
        df=df,
    )
    return True, report, dedupe_report


def main() -> None:
    ap = argparse.ArgumentParser(description="Build per-league merged CSVs into Matches/__merged__")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--league", type=str, help="Single league folder name under Matches/")
    g.add_argument("--leagues", nargs="+", help="Multiple league folder names under Matches/")
    g.add_argument("--all", action="store_true", help="Build all league folders under Matches/ (skipping __merged__ and Upcoming Fixtures)")

    ap.add_argument("--matches-root", default="Matches", help="Path to Matches root (default: Matches)")
    ap.add_argument("--merged-root", default=None, help="Path to Matches/__merged__ (default: <matches-root>/__merged__)")
    ap.add_argument("--write-report", action="store_true", help="Write a compact health report CSV after building")
    ap.add_argument("--report-path", default=None, help="Override report output path (default: <merged-root>/__merge_report__.csv)")
    ap.add_argument("--dedupe-report", action="store_true", help="Write a duplicate audit CSV after building")
    ap.add_argument("--dedupe-report-path", default=None, help=f"Override dedupe report output path (default: <merged-root>/{DEDUPE_REPORT_DEFAULT_NAME})")
    ap.add_argument("--recursive", action="store_true", help="Search CSVs recursively inside each league folder")
    ap.add_argument("--rolling-press", action="store_true", help="Compute rolling press features into merged output")
    ap.add_argument("--rolling-window", type=int, default=5, help="Rolling window size (default 5)")
    ap.add_argument("--skip-dir", action="append", default=[], help="Extra directory names under Matches/ to skip (repeatable)")
    ap.add_argument("--summary-json-path", default="/tmp/build_merged_summary.json", help="Path for post-run JSON summary (default: /tmp/build_merged_summary.json)")
    ap.add_argument("--no-summary-json", action="store_true", help="Print the post-run summary but do not write the JSON file")

    args = ap.parse_args()

    matches_root = str(args.matches_root)
    merged_root = str(args.merged_root) if args.merged_root else os.path.join(matches_root, "__merged__")

    skip_dirs = set(SKIP_DIRS_DEFAULT)
    skip_dirs.update(args.skip_dir or [])

    if args.league:
        targets = [args.league]
    elif args.leagues:
        targets = list(args.leagues)
    else:
        targets = [n for n, _ in list_league_folders(matches_root, skip_dirs=skip_dirs)]

    report_rows: List[dict] = []
    dedupe_report_rows: List[dict] = []
    processed_leagues: List[str] = []

    ok = 0
    fail = 0
    for league_name in targets:
        if league_name in skip_dirs:
            continue
        success, rep, dedupe_rep = build_one_league(
            matches_root=matches_root,
            merged_root=merged_root,
            league_name=league_name,
            recursive=bool(args.recursive),
            rolling_press=bool(args.rolling_press),
            window=int(args.rolling_window),
        )
        if rep is not None:
            report_rows.append(rep)
        if dedupe_rep is not None:
            dedupe_report_rows.append(dedupe_rep)
        elif bool(args.dedupe_report):
            missing_path = os.path.join(merged_root, f"{tagify(league_name)}__merged.csv")
            dedupe_report_rows.append(build_missing_dedupe_report_row(league_name, missing_path))
        ok += 1 if success else 0
        fail += 0 if success else 1
        if success:
            processed_leagues.append(league_name)

    if args.write_report:
        outp = str(args.report_path) if args.report_path else os.path.join(merged_root, "__merge_report__.csv")
        try:
            Path(os.path.dirname(outp) or ".").mkdir(parents=True, exist_ok=True)
            pd.DataFrame(report_rows).to_csv(outp, index=False)
            print(f"✅ wrote report: {outp} | leagues={len(report_rows)}")
        except Exception as e:
            print("❌ failed to write report:", outp, "|", e)

    if args.dedupe_report:
        outp = str(args.dedupe_report_path) if args.dedupe_report_path else os.path.join(merged_root, DEDUPE_REPORT_DEFAULT_NAME)
        try:
            Path(os.path.dirname(outp) or ".").mkdir(parents=True, exist_ok=True)
            pd.DataFrame(dedupe_report_rows).to_csv(outp, index=False)
            print(f"✅ wrote dedupe report: {outp} | leagues={len(dedupe_report_rows)}")
        except Exception as e:
            print("❌ failed to write dedupe report:", outp, "|", e)

    backfill_leagues = []
    press_populated_leagues = []
    rolling_press_leagues = []
    warning_entries = []

    for row in report_rows:
        league = row.get("league", "UNKNOWN")
        if int(row.get("snapshot_backfill_total", 0) or 0) > 0:
            backfill_leagues.append(
                {
                    "league": league,
                    "snapshot_backfill_total": int(row.get("snapshot_backfill_total", 0) or 0),
                    "snapshot_backfill_cols": row.get("snapshot_backfill_cols", ""),
                }
            )
        press_rate = row.get("press_base_min_rate", "")
        if press_rate != "" and float(press_rate) > 0.0:
            press_populated_leagues.append(league)
        rolling_rate = row.get("rolling_press_min_rate", "")
        if rolling_rate != "" and float(rolling_rate) > 0.0:
            rolling_press_leagues.append(league)
        warnings = str(row.get("warnings", "") or "").strip()
        if warnings:
            warning_entries.append({"league": league, "warnings": warnings.split("|")})

    summary = {
        "targets_requested": len(targets),
        "leagues_processed": len(report_rows),
        "ok": ok,
        "fail": fail,
        "merged_root": merged_root,
        "leagues_with_odds_backfill_rescue": backfill_leagues,
        "leagues_with_press_columns_populated": sorted(press_populated_leagues),
        "leagues_with_rolling_press": sorted(rolling_press_leagues),
        "leagues_with_warnings": warning_entries,
    }

    print("\nPOST-RUN SUMMARY")
    print(f"Leagues processed: {summary['leagues_processed']}")
    print(f"Leagues with odds backfill rescue: {len(backfill_leagues)}")
    if backfill_leagues:
        for item in backfill_leagues:
            print(f"  - {item['league']}: total={item['snapshot_backfill_total']} | {item['snapshot_backfill_cols']}")
    print(f"Leagues with press columns populated: {len(press_populated_leagues)}")
    if press_populated_leagues:
        print("  - " + ", ".join(sorted(press_populated_leagues)))
    print(f"Leagues with rolling press: {len(rolling_press_leagues)}")
    if rolling_press_leagues:
        print("  - " + ", ".join(sorted(rolling_press_leagues)))
    print(f"Leagues with warnings: {len(warning_entries)}")
    if warning_entries:
        for item in warning_entries:
            print(f"  - {item['league']}: {', '.join(item['warnings'])}")

    if not args.no_summary_json:
        try:
            summary_path = str(args.summary_json_path)
            Path(os.path.dirname(summary_path) or ".").mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"✅ wrote summary json: {summary_path}")
        except Exception as e:
            print("❌ failed to write summary json:", args.summary_json_path, "|", e)

    rebuilt_synth_leagues = [league for league in processed_leagues if league in SYNTH_ENABLED_LEAGUES]
    if rebuilt_synth_leagues:
        leagues_arg = ",".join(rebuilt_synth_leagues)
        patch_script = Path(__file__).parent / "patch_merge_add_synth_odds.py"
        repo_root = Path(__file__).parent
        print(f"\n[POST-MERGE] Reapplying synth patch for synth-enabled leagues: {leagues_arg}")
        result = subprocess.run(
            [
                sys.executable,
                str(patch_script),
                "--root",
                str(repo_root),
                "--leagues",
                leagues_arg,
                "--harmonize-duplicates",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("[POST-MERGE] Synth patch applied successfully.")
        else:
            print("[POST-MERGE] WARNING: synth patch failed — rerun manually:")
            print(
                f'  python3 patch_merge_add_synth_odds.py --root "{repo_root}" '
                f'--leagues "{leagues_arg}" --harmonize-duplicates'
            )
            if result.stderr:
                print(result.stderr[:500])

    print(f"\nDONE | ok={ok} fail={fail} | merged_root={merged_root}")


if __name__ == "__main__":
    main()
