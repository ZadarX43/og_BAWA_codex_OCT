# streaks_module.py
# Utilities to attach team streaks, H2H streaks and implied-vs-actual signals
# to a matches dataframe. Defensive against missing columns; works on mixed
# slates containing both completed and upcoming fixtures.

from __future__ import annotations

import math
from typing import Iterable, Tuple, Sequence, Optional


import numpy as np
import pandas as pd


# --- robust datetime helper ---
def _to_datetime_mixed(s: pd.Series, *, utc: bool = False) -> pd.Series:
    """Robust datetime parse.

    Uses pandas' format='mixed' + cache when available (suppresses 'Could not infer format' warnings),
    and falls back to plain to_datetime for older pandas.
    """
    try:
        # pandas >= 2.0
        return pd.to_datetime(s, errors="coerce", utc=utc, format="mixed", cache=True)
    except TypeError:
        # older pandas
        return pd.to_datetime(s, errors="coerce", utc=utc)


# ------------------------------- helpers ------------------------------------ #a

def _coerce_dt(s: pd.Series, fallback_cols: Sequence[str] = ("match_date", "Date")) -> pd.Series:
    """Return a datetime series. If s is missing, try common fallback columns on the df."""
    if isinstance(s, pd.Series):
        return _to_datetime_mixed(s, utc=False)
    return pd.Series(pd.NaT, index=pd.RangeIndex(0))


def _get_date_col(df: pd.DataFrame) -> str:
    for c in ("match_date", "Date", "date_GMT", "timestamp"):
        if c in df.columns:
            return c
    # Allow downstream creation; caller will recheck
    return "match_date"


def _to_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _sigmoid(x: pd.Series | float, k: float = 1.0) -> pd.Series | float:
    return 1.0 / (1.0 + np.exp(-k * (x)))


def _rolling_sum_shifted(g: pd.Series, window: int) -> pd.Series:
    """Rolling sum over prior N rows (shifted by 1 to exclude current)."""
    return g.astype(float).shift(1).rolling(window=window, min_periods=1).sum()


# New helper: rolling mean shifted
def _rolling_mean_shifted(g: pd.Series, window: int) -> pd.Series:
    """Rolling mean over prior N rows (shifted by 1 to exclude current)."""
    return g.astype(float).shift(1).rolling(window=window, min_periods=1).mean()


def _streak_consecutive(g: pd.Series) -> pd.Series:
    """Consecutive True-run length up to previous row (shifted)."""
    # Reset counter when condition is False
    block = (~g.astype(bool)).cumsum()
    streak = g.astype(bool).groupby(block).cumcount() + 1
    streak = streak.where(g.astype(bool), 0)
    return streak.shift(1).fillna(0).astype(int)


def _pair_key(h: pd.Series, a: pd.Series) -> pd.Series:
    left = h.astype(str)
    right = a.astype(str)
    return np.where(left < right, left + "||" + right, right + "||" + left)


def _ensure_required_cols(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        for c in missing:
            df[c] = np.nan


# --------------------------- long-form team table ---------------------------- #

def _to_team_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a long table with one row per (team, match). Builds robust boolean signals
    used for streaks. Only uses columns if present.
    """
    req = [
        "home_team_name", "away_team_name",
        "home_team_goal_count", "away_team_goal_count",
    ]
    _ensure_required_cols(df, req)
    dt_col = _get_date_col(df)
    if dt_col not in df.columns:
        # try to create from timestamp or date_GMT if present
        for fallback in ("timestamp", "date_GMT"):
            if fallback in df.columns:
                df[dt_col] = df[fallback]
                break
        if dt_col not in df.columns:
            df[dt_col] = pd.NaT

    # Coerce numerics
    hgf = _to_num(df, "home_team_goal_count")
    agf = _to_num(df, "away_team_goal_count")
    total_goals = (hgf + agf)

    # Pre-match xG (safe) — used for rolling xG_for/xG_against features
    def _first_xg(cands):
        for c in cands:
            if c in df.columns:
                return pd.to_numeric(df[c], errors="coerce")
        return pd.Series(np.nan, index=df.index)

    xg_home = _first_xg(["Home Team Pre-Match xG", "pre_match_xg_home", "home_pre_match_xg", "team_a_xg"])
    xg_away = _first_xg(["Away Team Pre-Match xG", "pre_match_xg_away", "away_pre_match_xg", "team_b_xg"])

    # Cards / corners if available
    hc = _to_num(df, "home_team_corner_count")
    ac = _to_num(df, "away_team_corner_count")
    total_corners = (hc + ac)

    hy = _to_num(df, "home_team_yellow_cards")
    ay = _to_num(df, "away_team_yellow_cards")
    hr = _to_num(df, "home_team_red_cards")
    ar = _to_num(df, "away_team_red_cards")
    # treat red as 2 cards by convention if we need a total proxy
    total_cards = hy.add(ay, fill_value=0).add(hr.mul(2), fill_value=0).add(ar.mul(2), fill_value=0)

    # First-to-concede approximation: prefer goal-timings if present, else GA>0 & GF==0 proxy
    ht_t = df.get("home_team_goal_timings", pd.Series(index=df.index, dtype=object))
    at_t = df.get("away_team_goal_timings", pd.Series(index=df.index, dtype=object))

    def _earliest_minute(s):
        # timings like '12,45,83' or empty; robust parse
        try:
            if pd.isna(s) or s == "" or (isinstance(s, str) and s.strip() == ""):
                return np.nan
            if isinstance(s, (int, float)):
                return float(s)
            parts = [p for p in str(s).replace("'", "").split(",") if p.strip() != ""]
            parts = [float(p) for p in parts]
            return min(parts) if parts else np.nan
        except Exception:
            return np.nan

    h_first = ht_t.apply(_earliest_minute) if "home_team_goal_timings" in df.columns else pd.Series(np.nan, index=df.index)
    a_first = at_t.apply(_earliest_minute) if "away_team_goal_timings" in df.columns else pd.Series(np.nan, index=df.index)

    # Long-form rows for home
    home_rows = pd.DataFrame({
        "team": df["home_team_name"].astype(str),
        "opponent": df["away_team_name"].astype(str),
        "is_home": True,
        "date": _to_datetime_mixed(df[dt_col], utc=False),
        "gf": hgf,
        "ga": agf,
        "total_goals": total_goals,
        "xg_for_pm": xg_home,
        "xg_against_pm": xg_away,
        "total_corners": total_corners,
        "total_cards": total_cards,
    })
    home_rows["result_w"] = (home_rows["gf"] > home_rows["ga"])
    home_rows["result_d"] = (home_rows["gf"] == home_rows["ga"])
    home_rows["result_l"] = (home_rows["gf"] < home_rows["ga"])
    home_rows["no_loss"] = (home_rows["result_w"] | home_rows["result_d"])
    home_rows["clean_sheet"] = (home_rows["ga"] == 0)
    home_rows["over25"] = (home_rows["total_goals"] > 2)
    home_rows["btts"] = (home_rows["gf"] > 0) & (home_rows["ga"] > 0)
    home_rows["under25"] = (home_rows["total_goals"] <= 2)

    # first-to-concede (home-side perspective)
    home_rows["first_to_concede"] = False
    if "home_team_goal_timings" in df.columns or "away_team_goal_timings" in df.columns:
        home_rows["first_to_concede"] = (a_first < h_first)
    else:
        # fallback proxy: conceded and did not score early info → if conceded and scored 0, assume conceded first
        home_rows["first_to_concede"] = (home_rows["ga"] > 0) & (home_rows["gf"] == 0)

    # Long-form rows for away
    away_rows = pd.DataFrame({
        "team": df["away_team_name"].astype(str),
        "opponent": df["home_team_name"].astype(str),
        "is_home": False,
        "date": _to_datetime_mixed(df[dt_col], utc=False),
        "gf": agf,
        "ga": hgf,
        "total_goals": total_goals,
        "xg_for_pm": xg_away,
        "xg_against_pm": xg_home,
        "total_corners": total_corners,
        "total_cards": total_cards,
    })
    away_rows["result_w"] = (away_rows["gf"] > away_rows["ga"])
    away_rows["result_d"] = (away_rows["gf"] == away_rows["ga"])
    away_rows["result_l"] = (away_rows["gf"] < away_rows["ga"])
    away_rows["no_loss"] = (away_rows["result_w"] | away_rows["result_d"])
    away_rows["clean_sheet"] = (away_rows["ga"] == 0)
    away_rows["over25"] = (away_rows["total_goals"] > 2)
    away_rows["btts"] = (away_rows["gf"] > 0) & (away_rows["ga"] > 0)
    away_rows["under25"] = (away_rows["total_goals"] <= 2)

    away_rows["first_to_concede"] = False
    if "home_team_goal_timings" in df.columns or "away_team_goal_timings" in df.columns:
        away_rows["first_to_concede"] = (h_first < a_first)
    else:
        away_rows["first_to_concede"] = (away_rows["ga"] > 0) & (away_rows["gf"] == 0)

    long_df = pd.concat([home_rows, away_rows], axis=0, ignore_index=True)
    long_df = long_df.sort_values(["team", "date"], kind="mergesort").reset_index(drop=True)
    return long_df


# ----------------------------- team streaks --------------------------------- #

def attach_team_streaks(df: pd.DataFrame, lookbacks: Tuple[int, int] = (5, 10)) -> pd.DataFrame:
    """
    Compute team-level streak counters and last-N counts, then join back:
      - no_losses_streak, without_clean_sheet_streak
      - over_2_5_streak, btts_streak
      - first_to_concede_streak
      - under_4_5_cards_streak, under_10_5_corners_streak
    Also emits last-N counts: e.g., no_losses_5, btts_10, etc.
    Returns a copy with *_home/*_away columns stamped.
    """
    out = df.copy()
    long_df = _to_team_long(out)

    # Boolean signals for streak logic
    s_no_loss = long_df["no_loss"]
    s_no_cs = ~long_df["clean_sheet"]
    s_o25 = long_df["over25"]
    s_btts = long_df["btts"]
    s_ftc = long_df["first_to_concede"]
    s_u45c = long_df["total_cards"] < 4.5
    s_u105corn = long_df["total_corners"] < 10.5

    # Consecutive streaks (till previous)
    long_df["no_losses_streak"] = s_no_loss.groupby(long_df["team"]).apply(_streak_consecutive).reset_index(level=0, drop=True)
    long_df["without_clean_sheet_streak"] = s_no_cs.groupby(long_df["team"]).apply(_streak_consecutive).reset_index(level=0, drop=True)
    long_df["over_2_5_streak"] = s_o25.groupby(long_df["team"]).apply(_streak_consecutive).reset_index(level=0, drop=True)
    long_df["btts_streak"] = s_btts.groupby(long_df["team"]).apply(_streak_consecutive).reset_index(level=0, drop=True)
    long_df["first_to_concede_streak"] = s_ftc.groupby(long_df["team"]).apply(_streak_consecutive).reset_index(level=0, drop=True)
    long_df["under_4_5_cards_streak"] = s_u45c.groupby(long_df["team"]).apply(_streak_consecutive).reset_index(level=0, drop=True)
    long_df["under_10_5_corners_streak"] = s_u105corn.groupby(long_df["team"]).apply(_streak_consecutive).reset_index(level=0, drop=True)

    # Last-N counts
    for N in lookbacks:
        long_df[f"no_losses_{N}"] = s_no_loss.groupby(long_df["team"]).apply(lambda g: _rolling_sum_shifted(g, N)).reset_index(level=0, drop=True)
        long_df[f"without_clean_sheet_{N}"] = s_no_cs.groupby(long_df["team"]).apply(lambda g: _rolling_sum_shifted(g, N)).reset_index(level=0, drop=True)
        long_df[f"over_2_5_{N}"] = s_o25.groupby(long_df["team"]).apply(lambda g: _rolling_sum_shifted(g, N)).reset_index(level=0, drop=True)
        long_df[f"btts_{N}"] = s_btts.groupby(long_df["team"]).apply(lambda g: _rolling_sum_shifted(g, N)).reset_index(level=0, drop=True)
        long_df[f"first_to_concede_{N}"] = s_ftc.groupby(long_df["team"]).apply(lambda g: _rolling_sum_shifted(g, N)).reset_index(level=0, drop=True)
        long_df[f"under_4_5_cards_{N}"] = s_u45c.groupby(long_df["team"]).apply(lambda g: _rolling_sum_shifted(g, N)).reset_index(level=0, drop=True)
        long_df[f"under_10_5_corners_{N}"] = s_u105corn.groupby(long_df["team"]).apply(lambda g: _rolling_sum_shifted(g, N)).reset_index(level=0, drop=True)

    # Prepare join back
    dt_col = _get_date_col(out)
    out[dt_col] = _to_datetime_mixed(out[dt_col], utc=False)

    # Home join
    home_cols = [c for c in long_df.columns if c not in (
        "team", "opponent", "date", "is_home",
        "gf", "ga", "total_goals", "total_corners", "total_cards",
        "result_w", "result_d", "result_l", "no_loss",
        "clean_sheet", "over25", "btts", "first_to_concede", "under25"
    )]
    home_map = long_df.loc[
        long_df["is_home"],
        ["team", "date"] + home_cols
    ].rename(columns={c: f"{c}_home" for c in home_cols})
    home_map = home_map.rename(columns={"team": "home_team_name", "date": dt_col})
    # 🔧 Guard: ensure at most one row per (home_team_name, date)
    home_map = home_map.drop_duplicates(subset=["home_team_name", dt_col])

    # Away join
    away_map = long_df.loc[
        ~long_df["is_home"],
        ["team", "date"] + home_cols
    ].rename(columns={c: f"{c}_away" for c in home_cols})
    away_map = away_map.rename(columns={"team": "away_team_name", "date": dt_col})
    # 🔧 Guard: ensure at most one row per (away_team_name, date)
    away_map = away_map.drop_duplicates(subset=["away_team_name", dt_col])

    out = out.merge(home_map, on=["home_team_name", dt_col], how="left")
    out = out.merge(away_map, on=["away_team_name", dt_col], how="left")

    return out


# ----------------------------- team rate features ---------------------------- #

def attach_team_rates(df: pd.DataFrame, lookbacks: Tuple[int, int] = (5, 10)) -> pd.DataFrame:
    """Attach leak-safe (shifted) rolling team rate features split by home/away.

    For each team, over prior N matches (shifted by 1 so current match never leaks):
      - scored_rate_{N}      := mean(gf > 0)
      - conceded_rate_{N}    := mean(ga > 0)
      - clean_sheet_rate_{N} := mean(ga == 0)
      - btts_rate_{N}        := mean(btts)
      - over25_rate_{N}      := mean(over25)
      - under25_rate_{N}     := mean(under25)
      - goaliness_avg_{N}    := mean(total_goals)

    Stamps back onto match rows as *_home for the home team (home fixtures) and *_away
    for the away team (away fixtures).
    """
    out = df.copy()
    long_df = _to_team_long(out)

    long_home = long_df[long_df["is_home"]].copy()
    long_away = long_df[~long_df["is_home"]].copy()

    def _compute_rates(ldf: pd.DataFrame) -> pd.DataFrame:
        if ldf is None or ldf.empty:
            return ldf

        gf = pd.to_numeric(ldf.get("gf"), errors="coerce")
        ga = pd.to_numeric(ldf.get("ga"), errors="coerce")

        # Only compute signals on realised rows (where goals are known).
        known = gf.notna() & ga.notna()
        tot = gf + ga

        s_scored = (gf > 0).astype(float).where(known)
        s_conceded = (ga > 0).astype(float).where(known)
        s_cs = (ga == 0).astype(float).where(known)
        s_btts = ((gf > 0) & (ga > 0)).astype(float).where(known)
        s_o25 = (tot > 2).astype(float).where(tot.notna())
        s_u25 = (tot <= 2).astype(float).where(tot.notna())
        s_goaliness = pd.to_numeric(tot, errors="coerce")

        # Rolling xG (pre-match) for/against
        s_xgf = pd.to_numeric(ldf.get("xg_for_pm"), errors="coerce")
        s_xga = pd.to_numeric(ldf.get("xg_against_pm"), errors="coerce")

        for N in lookbacks:
            ldf[f"scored_rate_{N}"] = s_scored.groupby(ldf["team"]).apply(lambda g: _rolling_mean_shifted(g, N)).reset_index(level=0, drop=True)
            ldf[f"conceded_rate_{N}"] = s_conceded.groupby(ldf["team"]).apply(lambda g: _rolling_mean_shifted(g, N)).reset_index(level=0, drop=True)
            ldf[f"clean_sheet_rate_{N}"] = s_cs.groupby(ldf["team"]).apply(lambda g: _rolling_mean_shifted(g, N)).reset_index(level=0, drop=True)
            ldf[f"btts_rate_{N}"] = s_btts.groupby(ldf["team"]).apply(lambda g: _rolling_mean_shifted(g, N)).reset_index(level=0, drop=True)
            ldf[f"over25_rate_{N}"] = s_o25.groupby(ldf["team"]).apply(lambda g: _rolling_mean_shifted(g, N)).reset_index(level=0, drop=True)
            ldf[f"under25_rate_{N}"] = s_u25.groupby(ldf["team"]).apply(lambda g: _rolling_mean_shifted(g, N)).reset_index(level=0, drop=True)
            ldf[f"goaliness_avg_{N}"] = s_goaliness.groupby(ldf["team"]).apply(lambda g: _rolling_mean_shifted(g, N)).reset_index(level=0, drop=True)
            ldf[f"xg_for_avg_{N}"] = s_xgf.groupby(ldf["team"]).apply(lambda g: _rolling_mean_shifted(g, N)).reset_index(level=0, drop=True)
            ldf[f"xg_against_avg_{N}"] = s_xga.groupby(ldf["team"]).apply(lambda g: _rolling_mean_shifted(g, N)).reset_index(level=0, drop=True)

        return ldf

    long_home = _compute_rates(long_home)
    long_away = _compute_rates(long_away)

    dt_col = _get_date_col(out)
    out[dt_col] = _to_datetime_mixed(out[dt_col], utc=False)

    expected = []
    for N in lookbacks:
        expected.extend([
            f"scored_rate_{N}",
            f"conceded_rate_{N}",
            f"clean_sheet_rate_{N}",
            f"btts_rate_{N}",
            f"over25_rate_{N}",
            f"under25_rate_{N}",
            f"goaliness_avg_{N}",
            f"xg_for_avg_{N}",
            f"xg_against_avg_{N}",
        ])

    # Drop any existing stamped columns to avoid merge suffix issues
    for c in expected:
        hc = f"{c}_home"
        ac = f"{c}_away"
        if hc in out.columns:
            out = out.drop(columns=[hc])
        if ac in out.columns:
            out = out.drop(columns=[ac])

    home_cols = [c for c in expected if c in long_home.columns]
    away_cols = [c for c in expected if c in long_away.columns]

    home_map = long_home.loc[:, ["team", "date"] + home_cols].rename(columns={c: f"{c}_home" for c in home_cols})
    home_map = home_map.rename(columns={"team": "home_team_name", "date": dt_col})
    home_map = home_map.drop_duplicates(subset=["home_team_name", dt_col])

    away_map = long_away.loc[:, ["team", "date"] + away_cols].rename(columns={c: f"{c}_away" for c in away_cols})
    away_map = away_map.rename(columns={"team": "away_team_name", "date": dt_col})
    away_map = away_map.drop_duplicates(subset=["away_team_name", dt_col])

    out = out.merge(home_map, on=["home_team_name", dt_col], how="left")
    out = out.merge(away_map, on=["away_team_name", dt_col], how="left")

    return out


# ------------------------------- H2H streaks -------------------------------- #

def attach_h2h_streaks(df: pd.DataFrame,
                       lookbacks: Tuple[int, int] = (5, 8),
                       time_decay: bool = True,
                       decay_lambda: float = 0.35) -> pd.DataFrame:
    """
    Compute H2H streaks/priors:
      - h2h_no_losses_streak_home / _away (consecutive not-lost vs opponent)
      - h2h_btts_streak_pair (consecutive BTTS across the pair)
      - h2h_without_clean_sheet_streak_pair (consecutive not-CS across the pair)
      - h2h_goal_diff_avg (from current home perspective; last-K avg, shifted)
      - h2h_home_advantage_factor (pair-level prior home-win rate across prior K)
    """
    out = df.copy()
    # Required basics
    req = ["home_team_name", "away_team_name", "home_team_goal_count", "away_team_goal_count"]
    _ensure_required_cols(out, req)
    dt_col = _get_date_col(out)
    if dt_col not in out.columns:
        out[dt_col] = pd.NaT
    out[dt_col] = _to_datetime_mixed(out[dt_col], utc=False)

    # Pair key and per-row signals
    pair = _pair_key(out["home_team_name"], out["away_team_name"])
    # IMPORTANT: do NOT fill missing goals with 0; that contaminates H2H signals for future fixtures.
    hgf = _to_num(out, "home_team_goal_count")
    agf = _to_num(out, "away_team_goal_count")
    known = hgf.notna() & agf.notna()

    tot_goals = (hgf + agf).where(known)
    over25 = (tot_goals > 2).where(known)

    # Pair-level booleans (masked to realised rows)
    btts = ((hgf > 0) & (agf > 0)) & known
    not_cs_either = btts.copy()  # historically this file used ~((hgf==0)|(agf==0)) which equals BTTS
    gd_pair = (hgf - agf).where(known)

    tmp = pd.DataFrame({
        "pair": pair,
        "date": out[dt_col],
        "home": out["home_team_name"].astype(str),
        "away": out["away_team_name"].astype(str),
        "gd_pair": gd_pair,
        "home_not_lost": (hgf >= agf),
        "away_not_lost": (agf >= hgf),
        "btts": btts,
        "not_cs_either": ~not_cs_either,  # "either team failed CS" => True when NOT clean sheet by at least one side
        "known": known,
        "tot_goals": tot_goals,
        "over25": over25,
    }).sort_values(["pair", "date"], kind="mergesort")

    # Pair-level consecutive streaks (shifted)
    def _pair_streak(s: pd.Series) -> pd.Series:
        return _streak_consecutive(s)

    tmp["btts_streak_pair"] = tmp.groupby("pair")["btts"].apply(_pair_streak).reset_index(level=0, drop=True)
    tmp["no_cs_streak_pair"] = tmp.groupby("pair")["not_cs_either"].apply(_pair_streak).reset_index(level=0, drop=True)

    # From current-home perspective:
    # If current row's home equals alphabetical first element -> sign = +1 else -1
    first = np.where(tmp["home"] < tmp["away"], tmp["home"], tmp["away"])
    sign = np.where(tmp["home"] == first, 1.0, -1.0)
    tmp["gd_for_current_home"] = sign * tmp["gd_pair"]

    # Rolling avg of prior goal diffs
    for K in lookbacks:
        tmp[f"h2h_gd_avg_{K}"] = tmp.groupby("pair")["gd_for_current_home"].apply(
            lambda g: g.shift(1).rolling(K, min_periods=1).mean()
        ).reset_index(level=0, drop=True)

    # H2H no-loss streaks from perspective of *current* home / away
    # Build booleans for "current-home did not lose" on each past row
    # On each row r, the relevant prior boolean series is:
    #   if prev.home == current.home  -> prev.home_not_lost
    #   else                          -> prev.away_not_lost
    # We approximate using: home_not_lost if prev.home==first, away_not_lost if prev.away==first; then select by sign
    tmp["first_not_lost"] = np.where(tmp["home"] == first, tmp["home_not_lost"], tmp["away_not_lost"])
    tmp["second_not_lost"] = np.where(tmp["home"] != first, tmp["home_not_lost"], tmp["away_not_lost"])

    # Streaks for first/second identities
    tmp["first_no_loss_streak"] = tmp.groupby("pair")["first_not_lost"].apply(_pair_streak).reset_index(level=0, drop=True)
    tmp["second_no_loss_streak"] = tmp.groupby("pair")["second_not_lost"].apply(_pair_streak).reset_index(level=0, drop=True)

    # Map to current-home/away
    tmp["h2h_no_losses_streak_home"] = np.where(tmp["home"] == first, tmp["first_no_loss_streak"], tmp["second_no_loss_streak"])
    tmp["h2h_no_losses_streak_away"] = np.where(tmp["home"] == first, tmp["second_no_loss_streak"], tmp["first_no_loss_streak"])

    # Home-advantage factor (pair-level past home-win rate)
    tmp["prev_home_win"] = (tmp["gd_pair"] > 0)
    for K in lookbacks:
        tmp[f"h2h_home_advantage_factor_{K}"] = tmp.groupby("pair")["prev_home_win"].apply(
            lambda g: g.shift(1).rolling(K, min_periods=1).mean()
        ).reset_index(level=0, drop=True)

    # Rolling H2H rates (shifted) and sample size over the same lookbacks
    tmp["known_f"] = tmp["known"].astype(float)
    tmp["btts_f"] = tmp["btts"].astype(float).where(tmp["known"].astype(bool))
    tmp["over25_f"] = tmp["over25"].astype(float).where(tmp["known"].astype(bool))
    tmp["tot_goals_f"] = pd.to_numeric(tmp["tot_goals"], errors="coerce")

    for K in lookbacks:
        tmp[f"h2h_n_{K}"] = tmp.groupby("pair")["known_f"].apply(
            lambda s: s.shift(1).rolling(K, min_periods=1).sum()
        ).reset_index(level=0, drop=True)

        tmp[f"h2h_btts_rate_{K}"] = tmp.groupby("pair")["btts_f"].apply(
            lambda s: s.shift(1).rolling(K, min_periods=1).mean()
        ).reset_index(level=0, drop=True)

        tmp[f"h2h_over25_rate_{K}"] = tmp.groupby("pair")["over25_f"].apply(
            lambda s: s.shift(1).rolling(K, min_periods=1).mean()
        ).reset_index(level=0, drop=True)

        tmp[f"h2h_goaliness_avg_{K}"] = tmp.groupby("pair")["tot_goals_f"].apply(
            lambda s: s.shift(1).rolling(K, min_periods=1).mean()
        ).reset_index(level=0, drop=True)

    # Join back
    join_cols = [
        "pair", "date",
        "btts_streak_pair", "no_cs_streak_pair",
        "h2h_no_losses_streak_home", "h2h_no_losses_streak_away"
    ] + [c for c in tmp.columns if c.startswith("h2h_gd_avg_")] + [c for c in tmp.columns if c.startswith("h2h_home_advantage_factor_")] \
      + [c for c in tmp.columns if c.startswith("h2h_n_")] \
      + [c for c in tmp.columns if c.startswith("h2h_btts_rate_")] \
      + [c for c in tmp.columns if c.startswith("h2h_over25_rate_")] \
      + [c for c in tmp.columns if c.startswith("h2h_goaliness_avg_")]

    mapper = tmp[join_cols].copy()
    mapper = mapper.rename(columns={"date": dt_col})
    mapper["__pair_key"] = mapper["pair"]

    # 🔧 Guard: ensure at most one row per (pair, date)
    mapper = mapper.drop_duplicates(subset=[dt_col, "__pair_key"])

    out["__pair_key"] = pair
    out = out.merge(mapper.drop(columns=["pair"]), on=[dt_col, "__pair_key"], how="left")
    out = out.drop(columns=["__pair_key"])

    # Rename a couple of generic columns to friendlier defaults
    # pick the smallest K for gd_avg/home_advantage as defaults
    if lookbacks:
        k0 = lookbacks[0]
        if f"h2h_gd_avg_{k0}" in out.columns and "h2h_goal_diff_avg" not in out.columns:
            out = out.rename(columns={f"h2h_gd_avg_{k0}": "h2h_goal_diff_avg"})
        if f"h2h_home_advantage_factor_{k0}" in out.columns and "h2h_home_advantage_factor" not in out.columns:
            out = out.rename(columns={f"h2h_home_advantage_factor_{k0}": "h2h_home_advantage_factor"})
        if f"h2h_n_{k0}" in out.columns and "h2h_n" not in out.columns:
            out = out.rename(columns={f"h2h_n_{k0}": "h2h_n"})
        if f"h2h_btts_rate_{k0}" in out.columns and "h2h_btts_rate" not in out.columns:
            out = out.rename(columns={f"h2h_btts_rate_{k0}": "h2h_btts_rate"})
        if f"h2h_over25_rate_{k0}" in out.columns and "h2h_over25_rate" not in out.columns:
            out = out.rename(columns={f"h2h_over25_rate_{k0}": "h2h_over25_rate"})
        if f"h2h_goaliness_avg_{k0}" in out.columns and "h2h_goaliness_avg" not in out.columns:
            out = out.rename(columns={f"h2h_goaliness_avg_{k0}": "h2h_goaliness_avg"})

    return out


# -------------- implied vs actual winrate (team rolling) -------------------- #

def attach_implied_vs_actual(df: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    """
    Compute rolling actual win rates per team (last `window` games, excluding current)
    and gaps vs 1x2 implied probabilities.
    Emits:
      - actual_winrate_last_{window}_home, _away
      - implied_vs_actual_gap_home, _away
    """
    out = df.copy()
    dt_col = _get_date_col(out)
    out[dt_col] = _to_datetime_mixed(out[dt_col], utc=False)

    # Implied probs from 1x2 odds (defensive if missing)
    def _impl(odds_col):
        o = _to_num(out, odds_col)
        return (1.0 / o).replace([np.inf, -np.inf], np.nan)

    impl_home = _impl("odds_ft_home_team_win")
    impl_draw = _impl("odds_ft_draw")
    impl_away = _impl("odds_ft_away_team_win")

    # Normalize implied to sum<=1 if all available
    total_impl = impl_home + impl_draw + impl_away
    with np.errstate(invalid="ignore", divide="ignore"):
        impl_home = impl_home / total_impl
        impl_away = impl_away / total_impl

    out["__impl_home"] = impl_home
    out["__impl_away"] = impl_away

    # Build long form with results W/L per team
    long_df = _to_team_long(out)
    long_df["win"] = long_df["result_w"].astype(float)

    # Rolling win-rate excluding current
    for side in ("home", "away"):
        mask = long_df["is_home"] if side == "home" else ~long_df["is_home"]
        grp = long_df[mask].sort_values(["team", "date"])
        roll = grp.groupby("team")["win"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean()).reset_index(level=0, drop=True)
        col = f"actual_winrate_last_{window}_{side}"
        grp[col] = roll
        # Map back to matches by (team,date,is_home)
        key_cols = ["team", "date"]
        grp_map = grp[key_cols + [col]].copy()
        if side == "home":
            grp_map = grp_map.rename(columns={"team": "home_team_name", "date": dt_col})
            out = out.merge(grp_map, on=["home_team_name", dt_col], how="left")
        else:
            grp_map = grp_map.rename(columns={"team": "away_team_name", "date": dt_col})
            out = out.merge(grp_map, on=["away_team_name", dt_col], how="left")

    # Gaps vs implied
    out["implied_vs_actual_gap_home"] = out.get("__impl_home") - out.get(f"actual_winrate_last_{window}_home")
    out["implied_vs_actual_gap_away"] = out.get("__impl_away") - out.get(f"actual_winrate_last_{window}_away")

    # Clean temp cols
    out = out.drop(columns=[c for c in ("__impl_home", "__impl_away") if c in out.columns], errors="ignore")
    return out


# ----------------------------- composites ----------------------------------- #

def attach_composites(df: pd.DataFrame, team_window_for_ga: int = 5) -> pd.DataFrame:
    """
    Compose a few higher-order indicators from streaks:
      - Momentum_Index_home/away
      - Match_Volatility_Score
      - H2H_Psych_Factor
    """
    out = df.copy()
    # Proxy for defensive pressure denominator: rolling GA avg from long-form
    long_df = _to_team_long(out)
    long_df["ga_avg5"] = long_df.groupby("team")["ga"].apply(lambda s: s.shift(1).rolling(team_window_for_ga, min_periods=1).mean()).reset_index(level=0, drop=True)

    dt_col = _get_date_col(out)
    long_df = long_df[["team", "date", "ga_avg5"]].rename(columns={"team": "home_team_name", "date": dt_col, "ga_avg5": "__ga5_home"})
    out = out.merge(long_df, on=["home_team_name", dt_col], how="left")
    long_df2 = long_df.rename(columns={"home_team_name": "away_team_name", "__ga5_home": "__ga5_away"})
    out = out.merge(long_df2, on=["away_team_name", dt_col], how="left")

    def _safe_den(x):
        return np.where(pd.isna(x) | (x <= 0), 0.5, x)

    # Momentum Index: (no_losses_streak – first_to_concede_5) / avg GA (last 5)
    if "no_losses_streak_home" in out.columns and "first_to_concede_5_home" in out.columns:
        out["Momentum_Index_home"] = (out["no_losses_streak_home"] - out["first_to_concede_5_home"]) / _safe_den(out["__ga5_home"])
    if "no_losses_streak_away" in out.columns and "first_to_concede_5_away" in out.columns:
        out["Momentum_Index_away"] = (out["no_losses_streak_away"] - out["first_to_concede_5_away"]) / _safe_den(out["__ga5_away"])

    # Match Volatility Score: mean of over25_5, btts_5 and pair-level btts streak scaled
    parts = []
    if "over_2_5_5_home" in out.columns and "btts_5_home" in out.columns:
        parts.append((out["over_2_5_5_home"] / 5.0 + out["btts_5_home"] / 5.0) / 2.0)
    if "btts_streak_pair" in out.columns:
        # scale to [0,1] by a cap of 5
        parts.append(np.minimum(1.0, out["btts_streak_pair"] / 5.0))
    if parts:
        out["Match_Volatility_Score"] = np.nanmean(np.vstack([p.to_numpy(dtype=float) for p in parts]), axis=0)

    # H2H-Psych-Factor = sigmoid(h2h_no_losses_streak_home – h2h_goal_diff_avg)
    if "h2h_no_losses_streak_home" in out.columns and "h2h_goal_diff_avg" in out.columns:
        out["H2H_Psych_Factor"] = _sigmoid((out["h2h_no_losses_streak_home"] - out["h2h_goal_diff_avg"]).astype(float), k=0.6)

    # Cleanup temp
    out = out.drop(columns=[c for c in ("__ga5_home", "__ga5_away") if c in out.columns], errors="ignore")
    return out


# ---------------------------- orchestrator ---------------------------------- #

def attach_streaks_and_h2h(df: pd.DataFrame,
                           team_lookbacks: Tuple[int, int] = (5, 10),
                           h2h_lookbacks: Tuple[int, int] = (5, 8),
                           include_implied_vs_actual: bool = True,
                           include_composites: bool = True) -> pd.DataFrame:
    """
    One-call attachment:
      1) team streaks
      2) h2h streaks
      3) implied vs actual winrate gap (optional)
      4) composites (optional)
    """
    out = attach_team_streaks(df, lookbacks=team_lookbacks)
    out = attach_team_rates(out, lookbacks=team_lookbacks)
    out = attach_h2h_streaks(out, lookbacks=h2h_lookbacks)
    if include_implied_vs_actual:
        out = attach_implied_vs_actual(out, window=team_lookbacks[0])
    if include_composites:
        out = attach_composites(out, team_window_for_ga=team_lookbacks[0])
    return out