# team_ratings.py
"""Per-league team power ratings (0–100 scale) built from historical matches.

Usage from project root:

    python -m team_ratings --league "England Premier League"

This will look under Matches/<League>/*matches*.csv, aggregate per-team stats,
compute a strength score, map it to [0,100], and write:

    ModelStore/<LeagueTag>_team_ratings.csv

where <LeagueTag> = league with spaces replaced by underscores.

These ratings can then be loaded and merged into match frames as
home_power_rating / away_power_rating / power_diff features for FTR and
side-market models.
"""


import argparse
import glob
import os
from typing import Optional


import numpy as np
import pandas as pd

# Canonical key/date helpers (best-effort)
try:
    from prediction_overlay import _match_key, _coalesce_match_date_series  # type: ignore
except Exception:
    _match_key = None
    _coalesce_match_date_series = None



def _pick_col(cols, candidates):
    """Return first column from `candidates` that exists in `cols`."""
    for c in candidates:
        if c in cols:
            return c
    return None


# Load canonical merged file for a league if present
def load_merged_for_league(league: str, merged_root: str = "Matches/__merged__") -> pd.DataFrame:
    """Load canonical merged file for a league if present."""
    league_tag = str(league).replace(" ", "_")
    path = os.path.join(merged_root, f"{league_tag}__merged.csv")

    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"⚠️ team_ratings: could not read merged file {path}: {e}")
        return pd.DataFrame()

    # Ensure canonical date/key fields
    if "match_date" in df.columns:
        try:
            md0 = df["match_date"].astype("string").str.strip()
            df["match_date"] = md0.mask(md0.eq(""), pd.NA)
        except Exception:
            pass

    try:
        if callable(_coalesce_match_date_series):
            df["match_date"] = _coalesce_match_date_series(df)
    except Exception:
        pass

    if callable(_match_key) and {"home_team_name", "away_team_name"}.issubset(df.columns):
        try:
            df["fixture_key"] = df.apply(_match_key, axis=1)
        except Exception:
            pass

    if "fixture_key" in df.columns:
        df["fixture_key"] = df["fixture_key"].astype("string").fillna("").str.strip()
        df = df[df["fixture_key"].ne("")].copy()

    # Coerce goal columns if present
    for c in ("home_team_goal_count", "away_team_goal_count"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # final dedupe
    if "fixture_key" in df.columns and not df.empty:
        df = df.drop_duplicates(subset=["fixture_key"], keep="first").reset_index(drop=True)

    print(f"✅ team_ratings: loaded merged source for {league} → {len(df)} rows")
    return df


def load_matches_for_league(league: str,
                            matches_root: str = "Matches") -> pd.DataFrame:
    """Load all match CSVs for a league from Matches/<League>/*matches*.csv.

    Returns a concatenated DataFrame with columns normalised (where possible)
    to these canonical names:
      - home_team_name, away_team_name
      - home_team_goal_count, away_team_goal_count
      - xg_home, xg_away (optional)
    """
    # We want historical *and* current-season fixtures. Some leagues have a single
    # current-season `*matches*.csv` that contains mostly upcoming fixtures (no goals),
    # while older seasons may be stored in other CSVs. So we load a union of:
    #   • Matches/<League>/*matches*.csv
    #   • Matches/<League>/*.csv (excluding obvious non-match artefacts)
    matches_pattern = os.path.join(matches_root, league, "*matches*.csv")
    any_pattern = os.path.join(matches_root, league, "*.csv")

    files_matches = sorted(glob.glob(matches_pattern))
    files_any = sorted(glob.glob(any_pattern))

    # Exclude odds-only / artefact files that commonly live alongside match CSVs.
    # (We still rely on column normalisation + goal presence later, but this avoids
    # reading huge irrelevant files.)
    exclude_substrings = [
        "fd_odds_",           # odds tables
        "odds_enriched",      # odds-enriched artefacts
        "enriched_synth",     # synthetic odds artefacts
        "player_team_totals", # press-intensity intermediate
        "press_intensity",    # press-intensity output
        "team_totals",        # generic totals
    ]

    def _keep_file(path: str) -> bool:
        base = os.path.basename(path).lower()
        return not any(s in base for s in exclude_substrings)

    files_any = [p for p in files_any if _keep_file(p)]

    # Union + stable order
    files = sorted(set(files_matches + files_any))

    print(f"🔎 team_ratings: matches_pattern={matches_pattern} → {len(files_matches)} files")
    print(f"🔎 team_ratings: any_pattern={any_pattern} → {len(files_any)} files (after excludes)")
    print(f"🔎 team_ratings: total files to load={len(files)}")

    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            print(f"  → loading {f}")
            frames.append(pd.read_csv(f, low_memory=False))
        except Exception as e:
            print(f"  ⚠️ could not read {f}: {e}")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Normalise some common headers to the canonical names we expect.
    rename_map: dict[str, str] = {}
    cols = df.columns

    # Team names
    home_name = _pick_col(cols, ["home_team_name", "Home", "home_team"])
    away_name = _pick_col(cols, ["away_team_name", "Away", "away_team"])
    if home_name and home_name != "home_team_name":
        rename_map[home_name] = "home_team_name"
    if away_name and away_name != "away_team_name":
        rename_map[away_name] = "away_team_name"

    # Goals
    home_goals = _pick_col(
        cols,
        ["home_team_goal_count", "home_goals", "FTHG", "home_team_score"],
    )
    away_goals = _pick_col(
        cols,
        ["away_team_goal_count", "away_goals", "FTAG", "away_team_score"],
    )
    if home_goals and home_goals != "home_team_goal_count":
        rename_map[home_goals] = "home_team_goal_count"
    if away_goals and away_goals != "away_team_goal_count":
        rename_map[away_goals] = "away_team_goal_count"

    # Pre-match xG (your CSVs use prettified headers)
    xg_home = _pick_col(
        cols,
        [
            "pre_match_xg_home",
            "Home Team Pre-Match xG",
            "team_a_xg",
            "xg_home",
        ],
    )
    xg_away = _pick_col(
        cols,
        [
            "pre_match_xg_away",
            "Away Team Pre-Match xG",
            "team_b_xg",
            "xg_away",
        ],
    )
    if xg_home and xg_home != "xg_home":
        rename_map[xg_home] = "xg_home"
    if xg_away and xg_away != "xg_away":
        rename_map[xg_away] = "xg_away"

    if rename_map:
        print(f"🔤 team_ratings: normalising columns {rename_map}")
        df = df.rename(columns=rename_map)

    # Ensure a stable chronological order to avoid any accidental leakage
    # in downstream processes that might later use rolling windows.
    try:
        if "timestamp" in df.columns:
            # Timestamps in your CSVs are Unix epoch seconds.
            df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)
        elif "date_GMT" in df.columns:
            # Fallback: parse the date_GMT string to datetime and sort.
            dt = pd.to_datetime(df["date_GMT"], errors="coerce")
            df = df.assign(_dt=dt).sort_values("_dt", ascending=True).drop(columns=["_dt"]).reset_index(drop=True)
    except Exception as e:
        print(f"⚠️ team_ratings: could not sort by timestamp/date_GMT: {e}")

    # Ensure canonical match_date string (YYYY-MM-DD) if possible
    # Treat blank match_date as missing so the coalescer can fill from date_GMT/timestamp.
    if "match_date" in df.columns:
        try:
            md0 = df["match_date"].astype("string").str.strip()
            df["match_date"] = md0.mask(md0.eq(""), pd.NA)
        except Exception:
            pass

    try:
        if callable(_coalesce_match_date_series):
            df["match_date"] = _coalesce_match_date_series(df)
    except Exception:
        pass

    # Always compute canonical fixture_key (do NOT trust existing fixture_key columns)
    if callable(_match_key) and {"home_team_name", "away_team_name"}.issubset(df.columns):
        try:
            df["fixture_key"] = df.apply(_match_key, axis=1)
        except Exception:
            df["fixture_key"] = ""
    else:
        df["fixture_key"] = ""

    df["fixture_key"] = df["fixture_key"].astype("string").fillna("").str.strip()
    df = df[df["fixture_key"].ne("")].copy()

    df["fixture_key"] = df["fixture_key"].astype("string").fillna("").str.strip()

    # Coerce numeric goals (needed for realised + dedupe scoring)
    for c in ("home_team_goal_count", "away_team_goal_count"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Dedupe: keep best row per fixture_key (prevents double-updates in Elo)
    if "fixture_key" in df.columns and (df["fixture_key"] != "").any():
        nn = df.notna().sum(axis=1)
        goals_present = df[["home_team_goal_count", "away_team_goal_count"]].notna().all(axis=1)
        status_complete = (
            df["status"].astype(str).str.lower().eq("complete")
            if "status" in df.columns else pd.Series(False, index=df.index)
        )
        score = (status_complete.astype(int) * 1000) + (goals_present.astype(int) * 100) + nn
        df = df.assign(__score=score).sort_values(["fixture_key", "__score"], ascending=[True, False])
        df = df.drop_duplicates(subset=["fixture_key"], keep="first").drop(columns=["__score"], errors="ignore")

    return df


def build_team_ratings_from_matches(
    league: str,
    matches_dir: str = "Matches",
    out_path: Optional[str] = None,
    *,
    min_matches: int = 10,
) -> pd.DataFrame:
    """High-level helper: load Matches/<League>/*.csv, compute per-team rating,
    and optionally write to CSV.

    Rating logic v1 (Elo-ish without time decay):
      - Build per-team aggregates over all matches in the sample:
          * matches played
          * goals_for, goals_against, goal_diff per match
          * points (3/1/0) per match and points-per-game (PPG)
          * xG for / xG against, xG diff per match (if available)
      - Compute z-scores for PPG, goal_diff_per_match, xg_diff_per_match
      - Combine into a single strength score:
            strength = 0.5*z_ppg + 0.3*z_gd + 0.2*z_xg
      - Map strength to a 0–100 rating (affine transform & clip).

    The returned DataFrame has columns:
      [
        'league', 'team_name', 'matches', 'points', 'ppg',
        'goals_for', 'goals_against', 'gd_per_match',
        'xg_for', 'xg_against', 'xg_diff_per_match',
        'rating'
      ]

    If out_path is provided, the ratings are written there.
    If out_path is None, a default of
      ModelStore/<LeagueTag>_team_ratings.csv
    is used and the path is printed.
    """
    df_matches = load_merged_for_league(league)
    if df_matches.empty:
        print(f"ℹ️ team_ratings: merged source missing for {league}; falling back to raw season files")
        df_matches = load_matches_for_league(league, matches_root=matches_dir)
    if df_matches is None or df_matches.empty:
        print("⚠️ team_ratings: no matches loaded; nothing to rate.")
        return pd.DataFrame()

    # Restrict to completed fixtures when a status column exists.
    # Your match CSVs use a 'status' column with values like 'complete' or 'incomplete'.
    df = df_matches.copy()
    if "status" in df.columns:
        before = len(df)
        mask_complete = df["status"].astype(str).str.lower().eq("complete")
        df = df.loc[mask_complete].copy()
        print(
            f"🔎 team_ratings: filtered to completed fixtures via status column "
            f"({len(df)}/{before} rows kept)"
        )
        if df.empty:
            print("⚠️ team_ratings: no completed fixtures after status filter; aborting.")
            return pd.DataFrame()

    cols = df.columns
    required = {
        "home_team_name",
        "away_team_name",
        "home_team_goal_count",
        "away_team_goal_count",
    }
    if not required.issubset(cols):
        print(f"⚠️ team_ratings: missing required columns {required - set(cols)}")
        return pd.DataFrame()

    # Ensure numeric goal counts
    df["home_team_goal_count"] = pd.to_numeric(
        df["home_team_goal_count"], errors="coerce"
    )
    df["away_team_goal_count"] = pd.to_numeric(
        df["away_team_goal_count"], errors="coerce"
    )

    # Home and away team-rows
    home_rows = pd.DataFrame(
        {
            "team": df["home_team_name"],
            "gf": df["home_team_goal_count"],
            "ga": df["away_team_goal_count"],
        }
    )
    away_rows = pd.DataFrame(
        {
            "team": df["away_team_name"],
            "gf": df["away_team_goal_count"],
            "ga": df["home_team_goal_count"],
        }
    )

    # Points per team (3/1/0)
    gd = df["home_team_goal_count"] - df["away_team_goal_count"]
    home_points = np.where(gd > 0, 3, np.where(gd == 0, 1, 0))
    away_points = np.where(gd < 0, 3, np.where(gd == 0, 1, 0))
    home_rows["points"] = home_points
    away_rows["points"] = away_points

    # xG for/against if available
    if "xg_home" in df.columns and "xg_away" in df.columns:
        df["xg_home"] = pd.to_numeric(df["xg_home"], errors="coerce")
        df["xg_away"] = pd.to_numeric(df["xg_away"], errors="coerce")
        home_rows["xg_for"] = df["xg_home"]
        home_rows["xg_against"] = df["xg_away"]
        away_rows["xg_for"] = df["xg_away"]
        away_rows["xg_against"] = df["xg_home"]
    else:
        home_rows["xg_for"] = np.nan
        home_rows["xg_against"] = np.nan
        away_rows["xg_for"] = np.nan
        away_rows["xg_against"] = np.nan

    all_rows = pd.concat([home_rows, away_rows], axis=0, ignore_index=True)

    grouped = (
        all_rows
        .groupby("team", dropna=True)
        .agg(
            matches=("gf", "count"),
            goals_for=("gf", "sum"),
            goals_against=("ga", "sum"),
            points=("points", "sum"),
            xg_for=("xg_for", "sum"),
            xg_against=("xg_against", "sum"),
        )
        .reset_index()
    )

    grouped = grouped[grouped["matches"] >= int(min_matches)].copy()
    if grouped.empty:
        print("⚠️ team_ratings: no teams with sufficient matches.")
        return grouped

    grouped["ppg"] = grouped["points"] / grouped["matches"].clip(lower=1)
    grouped["gd_per_match"] = (
        (grouped["goals_for"] - grouped["goals_against"]) /
        grouped["matches"].clip(lower=1)
    )

    with np.errstate(invalid="ignore"):
        grouped["xg_diff_per_match"] = (
            (grouped["xg_for"] - grouped["xg_against"]) /
            grouped["matches"].clip(lower=1)
        )

    def zscore(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        m = s.mean(skipna=True)
        sd = s.std(skipna=True)
        if not np.isfinite(sd) or sd <= 1e-9:
            return pd.Series(0.0, index=s.index)
        return (s - m) / sd

    grouped["z_ppg"] = zscore(grouped["ppg"])
    grouped["z_gd"] = zscore(grouped["gd_per_match"])
    grouped["z_xg"] = zscore(grouped["xg_diff_per_match"])

    # Combined strength (weights can be tuned later)
    grouped["strength"] = (
        0.5 * grouped["z_ppg"] +
        0.3 * grouped["z_gd"] +
        0.2 * grouped["z_xg"]
    )

    # Map strength to [0,100]
    s = grouped["strength"]
    s_min, s_max = float(s.min()), float(s.max())
    if not np.isfinite(s_min) or not np.isfinite(s_max) or s_max <= s_min:
        grouped["rating"] = 50.0
    else:
        rating = 20.0 + 60.0 * (s - s_min) / (s_max - s_min)
        grouped["rating"] = rating.clip(0.0, 100.0)

    grouped["league"] = league
    grouped = grouped.rename(columns={"team": "team_name"})
    cols_out = [
        "league",
        "team_name",
        "matches",
        "points",
        "ppg",
        "goals_for",
        "goals_against",
        "gd_per_match",
        "xg_for",
        "xg_against",
        "xg_diff_per_match",
        "rating",
    ]
    grouped = grouped[cols_out]

    # Decide output path if needed
    if out_path is None:
        league_tag = str(league).replace(" ", "_")
        model_dir = "ModelStore"
        os.makedirs(model_dir, exist_ok=True)
        out_path = os.path.join(model_dir, f"{league_tag}_team_ratings.csv")

    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        grouped.to_csv(out_path, index=False)
        print(f"✅ team_ratings: wrote {len(grouped)} rows → {out_path}")
    except Exception as e:
        print(f"⚠️ team_ratings: could not write ratings CSV: {e}")

    return grouped


def build_rolling_power_ratings(
    df_matches: pd.DataFrame,
    *,
    base_rating: float = 1500.0,
    k: float = 20.0,
) -> pd.DataFrame:
    """Build per-match pre-kickoff power ratings (Elo-style) for one league.

    For each row (match), we:
      • look up current ratings for home/away (default = base_rating),
      • store those as home_power_rating / away_power_rating,
      • then update the underlying ratings store using the *result* of this match.

    This guarantees that the rating attached to a given row only depends on
    matches that occurred strictly before that row (no self-leak, no future
    information).
    """
    df = df_matches.copy()

    # Preserve the full fixture list for output, but only let completed historical
    # matches update the underlying ratings state. Upcoming/incomplete fixtures
    # should still receive pre-match ratings in the exported file.
    status_series = (
        df["status"].astype(str).str.lower().str.strip()
        if "status" in df.columns else pd.Series("", index=df.index)
    )

    # Ensure a stable chronological order by the best available time column.
    time_col = None
    if "match_date" in df.columns:
        time_col = "match_date"
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
        df = df.sort_values("match_date").reset_index(drop=True)
    elif "timestamp" in df.columns:
        time_col = "timestamp"
        df = df.sort_values("timestamp").reset_index(drop=True)
    elif "date_GMT" in df.columns:
        time_col = "date_GMT"
        df["date_GMT"] = pd.to_datetime(df["date_GMT"], errors="coerce")
        df = df.sort_values("date_GMT").reset_index(drop=True)
    else:
        # Fallback: use index order, but keep it stable.
        df = df.sort_index().reset_index(drop=True)

    # Required columns for rating updates
    required = {"home_team_name", "away_team_name", "home_team_goal_count", "away_team_goal_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"build_rolling_power_ratings: missing required columns {missing}")

    # Current ratings store (Elo-like)
    ratings: dict[str, float] = {}

    # Columns to hold pre-match ratings
    df["home_power_rating_raw"] = np.nan
    df["away_power_rating_raw"] = np.nan

    # Optional post-match ratings (debug/acceptance-test only; never used as features)
    df["home_power_rating_post_raw"] = np.nan
    df["away_power_rating_post_raw"] = np.nan

    for idx, row in df.iterrows():
        home = row["home_team_name"]
        away = row["away_team_name"]

        # Look up current ratings (pre-match); default to base_rating on first sight
        r_h = ratings.get(home, base_rating)
        r_a = ratings.get(away, base_rating)

        # Store PRE-MATCH ratings as features for this match
        df.at[idx, "home_power_rating_raw"] = r_h
        df.at[idx, "away_power_rating_raw"] = r_a

        # Only completed historical rows should update the ratings state.
        # Upcoming/incomplete fixtures still keep their pre-match ratings in output,
        # but must not affect later rows.
        row_status = str(status_series.iloc[idx]).lower().strip()
        if row_status != "complete":
            continue

        # Now update ratings based on the completed match result, if we have goals
        try:
            hg = float(row["home_team_goal_count"])
            ag = float(row["away_team_goal_count"])
        except Exception:
            continue

        if np.isnan(hg) or np.isnan(ag):
            continue

        if hg > ag:
            s_h, s_a = 1.0, 0.0
        elif hg < ag:
            s_h, s_a = 0.0, 1.0
        else:
            s_h, s_a = 0.5, 0.5

        # Expected scores (logistic Elo)
        q_h = 10.0 ** (r_h / 400.0)
        q_a = 10.0 ** (r_a / 400.0)
        denom = q_h + q_a
        if denom <= 0:
            continue
        e_h = q_h / denom
        e_a = q_a / denom

        # Update underlying ratings for next matches
        ratings[home] = r_h + k * (s_h - e_h)
        ratings[away] = r_a + k * (s_a - e_a)

        # Store POST-match ratings (debug/acceptance test only)
        df.at[idx, "home_power_rating_post_raw"] = ratings.get(home, r_h)
        df.at[idx, "away_power_rating_post_raw"] = ratings.get(away, r_a)

    # Normalise raw ratings to a [20, 80] range for UI-friendly scale.
    raw = df[["home_power_rating_raw", "away_power_rating_raw"]].astype(float)
    r_min = float(raw.min().min()) if not raw.empty else np.nan
    r_max = float(raw.max().max()) if not raw.empty else np.nan

    if not np.isfinite(r_min) or not np.isfinite(r_max) or r_max <= r_min:
        df["home_power_rating"] = 50.0
        df["away_power_rating"] = 50.0
    else:
        span = r_max - r_min
        df["home_power_rating"] = 20.0 + 60.0 * (df["home_power_rating_raw"] - r_min) / span
        df["away_power_rating"] = 20.0 + 60.0 * (df["away_power_rating_raw"] - r_min) / span

    df["power_diff"] = df["home_power_rating"] - df["away_power_rating"]

    return df


def build_and_write_rolling_power(
    league: str,
    matches_dir: str = "Matches",
    out_path: Optional[str] = None,
) -> pd.DataFrame:
    """Build per-match rolling power ratings for a league and write to CSV.

    The output CSV contains all original match columns plus:
      • home_power_rating_raw, away_power_rating_raw
      • home_power_rating, away_power_rating, power_diff
    """
    df = load_merged_for_league(league)
    if df.empty:
        print("⚠️ team_ratings: merged file missing, falling back to raw season files.")
        df = load_matches_for_league(league, matches_root=matches_dir)

    if df.empty:
        print("⚠️ team_ratings: no matches loaded; cannot build rolling power ratings.")
        return df

    df_with_power = build_rolling_power_ratings(df)

    if out_path is None:
        league_tag = str(league).replace(" ", "_")
        model_dir = "ModelStore"
        os.makedirs(model_dir, exist_ok=True)
        out_path = os.path.join(model_dir, f"{league_tag}_match_power_ratings.csv")

    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df_with_power.to_csv(out_path, index=False)
        print(f"✅ team_ratings: wrote {len(df_with_power)} match-level power rows → {out_path}")
    except Exception as e:
        print(f"⚠️ team_ratings: could not write match-level power CSV: {e}")

    return df_with_power


def load_match_power_ratings(league: str,
                             model_dir: str = "ModelStore") -> pd.DataFrame:
    """Load precomputed per-match power ratings for a league, if available.

    This reads ModelStore/<LeagueTag>_match_power_ratings.csv as produced by
    :func:`build_and_write_rolling_power`. Returns an empty DataFrame if the
    file is missing or cannot be read.
    """
    league_tag = str(league).replace(" ", "_")
    path = os.path.join(model_dir, f"{league_tag}_match_power_ratings.csv")
    if not os.path.exists(path):
        print(f"⚠️ team_ratings: no match-level power file found at {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"⚠️ team_ratings: could not read match-level power file {path}: {e}")
        return pd.DataFrame()


def load_team_ratings(league: str,
                      model_dir: str = "ModelStore") -> pd.DataFrame:
    """Read ModelStore/<LeagueTag>_team_ratings.csv if present.

    Returns an empty DataFrame if the file does not exist.
    """
    league_tag = str(league).replace(" ", "_")
    path = os.path.join(model_dir, f"{league_tag}_team_ratings.csv")
    if not os.path.exists(path):
        print(f"⚠️ team_ratings: no ratings file found at {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"⚠️ team_ratings: could not read {path}: {e}")
        return pd.DataFrame()


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Build power ratings for a league (team summary or per-match rolling)."
    )
    parser.add_argument(
        "--league",
        required=True,
        help="League name (e.g. 'England Premier League')",
    )
    parser.add_argument(
        "--matches-root",
        default="Matches",
        help="Root directory for match CSVs (default: Matches)",
    )
    parser.add_argument(
        "--model-dir",
        default="ModelStore",
        help="Directory to write ratings CSV into (default: ModelStore)",
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=10,
        help="Minimum matches per team to include in summary ratings (default: 10)",
    )
    parser.add_argument(
        "--mode",
        choices=["summary", "rolling"],
        default="summary",
        help=(
            "Which type of output to produce: "
            "'summary' = one row per team (legacy), "
            "'rolling' = per-match pre-kickoff power ratings."
        ),
    )

    args = parser.parse_args()

    league_tag = str(args.league).replace(" ", "_")
    os.makedirs(args.model_dir, exist_ok=True)

    if args.mode == "summary":
        df_matches = load_merged_for_league(args.league)
        if df_matches.empty:
            print("⚠️ team_ratings: merged file missing, falling back to raw season files.")
            df_matches = load_matches_for_league(args.league, matches_root=args.matches_root)

        if df_matches.empty:
            print("⚠️ team_ratings: no matches loaded; aborting.")
            return
        out_path = os.path.join(args.model_dir, f"{league_tag}_team_ratings.csv")
        build_team_ratings_from_matches(
            league=args.league,
            matches_dir=args.matches_root,
            out_path=out_path,
            min_matches=args.min_matches,
        )
    else:
        out_path = os.path.join(args.model_dir, f"{league_tag}_match_power_ratings.csv")
        build_and_write_rolling_power(
            league=args.league,
            matches_dir=args.matches_root,
            out_path=out_path,
        )


if __name__ == "__main__":
    _main()