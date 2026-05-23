#!/usr/bin/env python3
"""uefa_context.py

UEFA context features for OG/BAWA.

Goal
----
Turn UEFA table incentives ("must win", "goal hunt", "rotation risk", "volatile cut line")
into deterministic, joinable features that can be stamped onto ALLMARKETS.

This module is intentionally v0.1:
- uses existing TEAM CSVs (and optionally MATCH CSVs for max-round context)
- no web scraping
- deterministic, explainable flags

Typical usage (inside bookie_allmarkets):
    from uefa_context import attach_uefa_context_for_league
    df = attach_uefa_context_for_league(df, league="Champions League", teams_root=Path("TEAMS"), matches_root=Path("Matches"))

Key conventions
---------------
- We prefer exact integer points computed from wins/draws when present.
- `gap_to24 = cut24_points - points` (positive => outside; negative => cushion inside).
- `state_bucket` uses league_position when available, otherwise points ordering fallback.

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class UefaContextConfig:
    # Cut lines for league-phase format
    cut8: int = 8
    cut24: int = 24

    # How many matches left before we treat pressure as urgent
    urgent_matches_remaining: int = 2

    # Cushion thresholds (points) for considering a team "safe" (rotation risk)
    rot_safe_top8_cushion: int = 3
    rot_safe_top24_cushion: int = 5

    # Band (points) around cut lines used to compute live-table volatility
    volatility_band_points: int = 2

    # If a team is outside the cut by at least this many points, call it "must win big"
    must_win_big_points: int = 2


DEFAULT_CFG = UefaContextConfig()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _s(df: pd.DataFrame, col: str) -> pd.Series:
    return df.get(col, pd.Series("", index=df.index, dtype="string")).astype("string").fillna("")


def _n(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df.get(col, pd.Series(np.nan, index=df.index)), errors="coerce")


def _norm_team_name(s: pd.Series) -> pd.Series:
    """Conservative team-name normalization for joins.

    - lower
    - strip
    - drop punctuation
    - collapse whitespace
    - remove common tokens that drift across sources

    NOTE: keep conservative; do not over-normalize.
    """
    x = s.astype("string").fillna("").str.lower().str.strip()
    # drop very common tokens
    x = x.str.replace(r"\bfc\b", "", regex=True)
    x = x.str.replace(r"\bsc\b", "", regex=True)
    x = x.str.replace(r"\bcd\b", "", regex=True)
    x = x.str.replace(r"\bclub\b", "", regex=True)
    # punctuation -> space
    x = x.str.replace(r"[^a-z0-9]+", " ", regex=True)
    x = x.str.replace(r"\s+", " ", regex=True).str.strip()
    return x


def _infer_points(df: pd.DataFrame) -> pd.Series:
    """Prefer exact integer points = wins*3 + draws when possible."""
    wins = _n(df, "wins")
    draws = _n(df, "draws")
    if wins.notna().any() and draws.notna().any():
        pts = (wins.fillna(0) * 3.0 + draws.fillna(0)).astype(float)
        return pts
    # fallback: ppg * matches_played
    ppg = _n(df, "points_per_game")
    mp = _n(df, "matches_played")
    return (ppg * mp).astype(float)


def _safe_int(s: pd.Series, default: int = 0) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    out = out.fillna(default).astype(int)
    return out


def _max_round_from_matches(matches_csv: Path) -> Optional[int]:
    """Infer max round from a UEFA matches CSV using `Game Week` when present."""
    try:
        if matches_csv is None or not Path(matches_csv).exists():
            return None
        m = pd.read_csv(matches_csv, low_memory=False)
        if m is None or m.empty:
            return None
        gw = pd.to_numeric(m.get("Game Week", np.nan), errors="coerce")
        gw = gw.dropna()
        if gw.empty:
            return None
        mx = int(np.nanmax(gw.to_numpy(dtype=float)))
        if mx <= 0:
            return None
        return mx
    except Exception:
        return None


def _fallback_table_order(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback ordering when league_position is missing/broken.

    Sort by:
    - points desc
    - goal_difference desc
    - goals_scored desc
    """
    out = df.copy()
    out["__points"] = _infer_points(out)
    gd = _n(out, "goal_difference")
    gf = _n(out, "goals_scored")
    out["__gd"] = gd.fillna(-1e9)
    out["__gf"] = gf.fillna(-1e9)
    out = out.sort_values(["__points", "__gd", "__gf"], ascending=[False, False, False])
    out["__rank_fallback"] = np.arange(1, len(out) + 1)
    return out


# -----------------------------------------------------------------------------
# Core computations
# -----------------------------------------------------------------------------

def compute_uefa_team_snapshot(
    teams_df: pd.DataFrame,
    *,
    matches_csv: Optional[Path] = None,
    cfg: UefaContextConfig = DEFAULT_CFG,
    debug: bool = False,
) -> pd.DataFrame:
    """Compute a UEFA table snapshot + incentive flags from the TEAM stats CSV.

    Parameters
    ----------
    teams_df:
        The league TEAM table (one row per team) from europe-uefa-*-teams-*-stats.csv
    matches_csv:
        Optional path to the league matches CSV. Used only to infer max round (`Game Week`).
    cfg:
        Config thresholds.

    Returns
    -------
    DataFrame with (at minimum):
        team_name, team_norm, points_est,
        cut8_points_est, cut24_points_est,
        gap_to8, gap_to24,
        state_bucket,
        must_win_flag, must_win_big_flag, must_avoid_loss_flag,
        rotation_risk_flag,
        volatility_band_n, volatility_ratio,
        matches_remaining,
        eliminated_flag, pride_only_flag

    Notes
    -----
    This is deliberately simple and deterministic. We can tune thresholds once backtests exist.
    """

    if teams_df is None or teams_df.empty:
        return teams_df

    df = teams_df.copy()

    # Identity / join key
    if "team_name" not in df.columns and "common_name" in df.columns:
        df["team_name"] = df["common_name"]
    df["team_name"] = _s(df, "team_name")
    df["team_norm"] = _norm_team_name(df["team_name"])

    # Core numeric fields
    df["points_est"] = _infer_points(df)
    df["matches_played"] = _n(df, "matches_played")
    df["league_position"] = _n(df, "league_position")

    # Determine max_round and matches_remaining
    max_round = _max_round_from_matches(matches_csv) if matches_csv else None
    if max_round is None:
        # League phase default (safe): 8 rounds
        max_round = 8
    df["max_round"] = int(max_round)

    mp = pd.to_numeric(df["matches_played"], errors="coerce").fillna(0).astype(int)
    df["matches_remaining"] = (int(max_round) - mp).clip(lower=0).astype(int)

    # Use league_position when it looks valid; else compute fallback
    lp = pd.to_numeric(df["league_position"], errors="coerce")
    lp_valid = lp.notna() & (lp > 0)

    if bool(lp_valid.any()):
        # If some rows have invalid positions, fill them with fallback ranks
        base = df.copy()
        fb = _fallback_table_order(base)
        fb_rank = fb["__rank_fallback"].reindex(base.index)
        lp2 = lp.where(lp_valid, fb_rank)
        df["league_position_eff"] = pd.to_numeric(lp2, errors="coerce").fillna(999).astype(int)
    else:
        fb = _fallback_table_order(df)
        df = fb.drop(columns=["__points", "__gd", "__gf"], errors="ignore")
        df["league_position_eff"] = pd.to_numeric(df.get("__rank_fallback"), errors="coerce").fillna(999).astype(int)

    # Cut lines
    def _cut_points(pos: int) -> float:
        try:
            sub = df.loc[df["league_position_eff"].eq(int(pos))]
            if sub.empty:
                return float("nan")
            return float(pd.to_numeric(sub["points_est"], errors="coerce").iloc[0])
        except Exception:
            return float("nan")

    cut8_pts = _cut_points(cfg.cut8)
    cut24_pts = _cut_points(cfg.cut24)

    df["cut8_points_est"] = cut8_pts
    df["cut24_points_est"] = cut24_pts

    # Tie-break proxies (for "win big" and volatility shading)
    df["goal_difference"] = _n(df, "goal_difference")
    df["goals_scored"] = _n(df, "goals_scored")

    cut24_gd = float(pd.to_numeric(df.loc[df["league_position_eff"].eq(cfg.cut24), "goal_difference"], errors="coerce").iloc[0]) if bool(df["league_position_eff"].eq(cfg.cut24).any()) else float("nan")
    cut24_gf = float(pd.to_numeric(df.loc[df["league_position_eff"].eq(cfg.cut24), "goals_scored"], errors="coerce").iloc[0]) if bool(df["league_position_eff"].eq(cfg.cut24).any()) else float("nan")

    df["cut24_goal_difference"] = cut24_gd
    df["cut24_goals_scored"] = cut24_gf

    # Gaps (positive => outside; negative => cushion)
    pts = pd.to_numeric(df["points_est"], errors="coerce")
    df["gap_to8"] = (cut8_pts - pts).astype(float) if np.isfinite(cut8_pts) else np.nan
    df["gap_to24"] = (cut24_pts - pts).astype(float) if np.isfinite(cut24_pts) else np.nan

    # State bucket (prefer league_position_eff when available)
    pos_eff = pd.to_numeric(df["league_position_eff"], errors="coerce").fillna(999).astype(int)

    state = pd.Series("OUTSIDE", index=df.index, dtype="string")
    state = state.mask(pos_eff <= int(cfg.cut24), "TOP24")
    state = state.mask(pos_eff <= int(cfg.cut8), "TOP8")

    # If positions are nonsense but cut points exist, also use points test
    if np.isfinite(cut24_pts):
        state = state.mask((pts >= cut24_pts) & (state.eq("OUTSIDE")), "TOP24")
    if np.isfinite(cut8_pts):
        state = state.mask((pts >= cut8_pts) & (state.ne("TOP8")), "TOP8")

    df["state_bucket"] = state

    # Elimination / pride-only
    # If even winning out cannot reach cut24, call eliminated.
    mr = pd.to_numeric(df["matches_remaining"], errors="coerce").fillna(0).astype(int)
    max_points_possible = pts + (3.0 * mr)
    eliminated = pd.Series(False, index=df.index)
    if np.isfinite(cut24_pts):
        eliminated = max_points_possible < float(cut24_pts)
    df["eliminated_flag"] = eliminated.astype(int)
    df["pride_only_flag"] = (eliminated & (mr <= cfg.urgent_matches_remaining)).astype(int)

    # Incentive flags
    # Must-win: outside top24 with urgent remaining matches
    must_win = (df["state_bucket"].eq("OUTSIDE") & (mr <= int(cfg.urgent_matches_remaining)) & (~eliminated)).astype(int)
    df["must_win_flag"] = must_win

    # Must-avoid-loss: barely inside top24 (small cushion) with urgent remaining
    cushion24 = (-pd.to_numeric(df["gap_to24"], errors="coerce"))  # points above cut24
    must_avoid = (df["state_bucket"].eq("TOP24") & (df["state_bucket"].ne("TOP8")) & (mr <= int(cfg.urgent_matches_remaining)) & (cushion24 <= 1.0)).astype(int)
    df["must_avoid_loss_flag"] = must_avoid

    # Must-win-big: outside and (a) needs >=2 points OR (b) needs 1 point but tie-break is behind
    gd = pd.to_numeric(df["goal_difference"], errors="coerce")
    gf = pd.to_numeric(df["goals_scored"], errors="coerce")
    gap24 = pd.to_numeric(df["gap_to24"], errors="coerce")

    tie_break_behind = pd.Series(False, index=df.index)
    if np.isfinite(cut24_gd):
        tie_break_behind = tie_break_behind | (gd.notna() & (gd < float(cut24_gd)))
    if np.isfinite(cut24_gf):
        tie_break_behind = tie_break_behind | ((gd.notna() & np.isfinite(cut24_gd) & (gd == float(cut24_gd))) & (gf.notna() & (gf < float(cut24_gf))))

    need_points = gap24.fillna(999)
    must_win_big = (
        (must_win.astype(bool))
        & (
            (need_points >= float(cfg.must_win_big_points))
            | ((need_points >= 1.0) & tie_break_behind)
        )
    ).astype(int)
    df["must_win_big_flag"] = must_win_big

    # Rotation risk: safe + low incentive (late in phase)
    cushion8 = (-pd.to_numeric(df["gap_to8"], errors="coerce"))

    rot = pd.Series(False, index=df.index)
    rot = rot | (
        df["state_bucket"].eq("TOP8")
        & (mr <= int(cfg.urgent_matches_remaining))
        & (cushion8 >= float(cfg.rot_safe_top8_cushion))
    )
    rot = rot | (
        df["state_bucket"].eq("TOP24")
        & (df["state_bucket"].ne("TOP8"))
        & (mr <= int(cfg.urgent_matches_remaining))
        & (cushion24 >= float(cfg.rot_safe_top24_cushion))
    )
    df["rotation_risk_flag"] = rot.astype(int)

    # Volatility band around cut lines
    n_teams = int(len(df))
    band = float(cfg.volatility_band_points)

    vol24_n = 0
    vol8_n = 0
    if np.isfinite(cut24_pts):
        vol24_n = int(((pts - float(cut24_pts)).abs() <= band).sum())
    if np.isfinite(cut8_pts):
        vol8_n = int(((pts - float(cut8_pts)).abs() <= band).sum())

    df["volatility_band_24_n"] = vol24_n
    df["volatility_band_8_n"] = vol8_n
    df["volatility_band_n"] = int(vol24_n + vol8_n)
    df["volatility_ratio"] = float((vol24_n + vol8_n) / max(n_teams, 1))

    if bool(debug):
        try:
            print(
                f"[uefa_context] snapshot: n={n_teams} max_round={max_round} "
                f"cut8_pts={cut8_pts} cut24_pts={cut24_pts} "
                f"vol_band_n={int(vol24_n + vol8_n)} vol_ratio={float((vol24_n + vol8_n)/max(n_teams,1)):.3f}"
            )
            # quick state counts
            print("[uefa_context] state counts:", df["state_bucket"].value_counts(dropna=False).to_dict())
            print(
                "[uefa_context] flags counts:",
                {
                    "must_win": int(df["must_win_flag"].sum()),
                    "must_win_big": int(df["must_win_big_flag"].sum()),
                    "must_avoid_loss": int(df["must_avoid_loss_flag"].sum()),
                    "rotation_risk": int(df["rotation_risk_flag"].sum()),
                    "eliminated": int(df["eliminated_flag"].sum()),
                },
            )
        except Exception:
            pass

    # Keep only useful columns (but leave raw columns too, caller can drop)
    return df


def attach_uefa_context(
    df: pd.DataFrame,
    team_snapshot: pd.DataFrame,
    *,
    home_col: str = "home_team_name",
    away_col: str = "away_team_name",
    prefix_home: str = "uefa_home_",
    prefix_away: str = "uefa_away_",
    debug: bool = False,
) -> pd.DataFrame:
    """Attach computed UEFA context features onto a match-level dataframe.

    Designed to join into ALLMARKETS exports.

    Adds match-level fields:
      - uefa_home_state / uefa_away_state
      - uefa_home_gap24 / uefa_away_gap24 / uefa_gap24_diff
      - uefa_home_rotation_risk / uefa_away_rotation_risk
      - uefa_both_must_win / uefa_goal_hunt_flag / uefa_pride_only_flag
      - uefa_live_table_volatility

    It also attaches a handful of per-team debug fields (points_est, matches_remaining).
    """

    if df is None or df.empty or team_snapshot is None or team_snapshot.empty:
        return df

    out = df.copy()

    snap = team_snapshot.copy()
    if "team_name" not in snap.columns and "common_name" in snap.columns:
        snap["team_name"] = snap["common_name"]
    snap["team_name"] = _s(snap, "team_name")
    snap["team_norm"] = snap.get("team_norm", _norm_team_name(snap["team_name"]))

    # de-dupe snapshot on team_norm
    snap = snap.sort_values(["team_norm"]).drop_duplicates(subset=["team_norm"], keep="first")

    # Prepare join keys
    out["__home_norm"] = _norm_team_name(_s(out, home_col))
    out["__away_norm"] = _norm_team_name(_s(out, away_col))

    keep_cols = [
        "team_norm",
        "state_bucket",
        "points_est",
        "gap_to8",
        "gap_to24",
        "must_win_flag",
        "must_win_big_flag",
        "must_avoid_loss_flag",
        "rotation_risk_flag",
        "matches_remaining",
        "eliminated_flag",
        "pride_only_flag",
        "volatility_ratio",
    ]
    keep_cols = [c for c in keep_cols if c in snap.columns]

    s2 = snap[keep_cols].copy()

    # Join home
    home = s2.rename(columns={
        "team_norm": "__home_norm",
        "state_bucket": f"{prefix_home}state",
        "points_est": f"{prefix_home}points_est",
        "gap_to8": f"{prefix_home}gap8",
        "gap_to24": f"{prefix_home}gap24",
        "must_win_flag": f"{prefix_home}must_win",
        "must_win_big_flag": f"{prefix_home}must_win_big",
        "must_avoid_loss_flag": f"{prefix_home}must_avoid_loss",
        "rotation_risk_flag": f"{prefix_home}rotation_risk",
        "matches_remaining": f"{prefix_home}matches_remaining",
        "eliminated_flag": f"{prefix_home}eliminated",
        "pride_only_flag": f"{prefix_home}pride_only",
        "volatility_ratio": "__uefa_vol_ratio",
    })

    out = out.merge(home, on="__home_norm", how="left")

    # Join away
    away = s2.rename(columns={
        "team_norm": "__away_norm",
        "state_bucket": f"{prefix_away}state",
        "points_est": f"{prefix_away}points_est",
        "gap_to8": f"{prefix_away}gap8",
        "gap_to24": f"{prefix_away}gap24",
        "must_win_flag": f"{prefix_away}must_win",
        "must_win_big_flag": f"{prefix_away}must_win_big",
        "must_avoid_loss_flag": f"{prefix_away}must_avoid_loss",
        "rotation_risk_flag": f"{prefix_away}rotation_risk",
        "matches_remaining": f"{prefix_away}matches_remaining",
        "eliminated_flag": f"{prefix_away}eliminated",
        "pride_only_flag": f"{prefix_away}pride_only",
        "volatility_ratio": "__uefa_vol_ratio2",
    })

    out = out.merge(away, on="__away_norm", how="left")

    # Match-level derived fields
    h_gap24 = pd.to_numeric(out.get(f"{prefix_home}gap24", np.nan), errors="coerce")
    a_gap24 = pd.to_numeric(out.get(f"{prefix_away}gap24", np.nan), errors="coerce")

    out["uefa_gap24_diff"] = (h_gap24 - a_gap24)

    h_mw = pd.to_numeric(out.get(f"{prefix_home}must_win", 0), errors="coerce").fillna(0).astype(int)
    a_mw = pd.to_numeric(out.get(f"{prefix_away}must_win", 0), errors="coerce").fillna(0).astype(int)
    out["uefa_both_must_win"] = ((h_mw.eq(1)) & (a_mw.eq(1))).astype(int)

    h_big = pd.to_numeric(out.get(f"{prefix_home}must_win_big", 0), errors="coerce").fillna(0).astype(int)
    a_big = pd.to_numeric(out.get(f"{prefix_away}must_win_big", 0), errors="coerce").fillna(0).astype(int)
    out["uefa_goal_hunt_flag"] = ((h_big.eq(1)) | (a_big.eq(1))).astype(int)

    h_pride = pd.to_numeric(out.get(f"{prefix_home}pride_only", 0), errors="coerce").fillna(0).astype(int)
    a_pride = pd.to_numeric(out.get(f"{prefix_away}pride_only", 0), errors="coerce").fillna(0).astype(int)
    out["uefa_pride_only_flag"] = ((h_pride.eq(1)) | (a_pride.eq(1))).astype(int)

    # Live table volatility (league-level constant)
    v1 = pd.to_numeric(out.get("__uefa_vol_ratio", np.nan), errors="coerce")
    v2 = pd.to_numeric(out.get("__uefa_vol_ratio2", np.nan), errors="coerce")
    out["uefa_live_table_volatility"] = v1.fillna(v2)

    # Cleanup internal columns
    out = out.drop(columns=["__home_norm", "__away_norm", "__uefa_vol_ratio", "__uefa_vol_ratio2"], errors="ignore")

    if bool(debug):
        try:
            # coverage diagnostics
            cov_home = out.get(f"{prefix_home}state", pd.Series("", index=out.index)).notna().mean()
            cov_away = out.get(f"{prefix_away}state", pd.Series("", index=out.index)).notna().mean()
            print(f"[uefa_context] attach coverage: home={cov_home:.3f} away={cov_away:.3f} rows={len(out)}")
        except Exception:
            pass

    return out


# -----------------------------------------------------------------------------
# File discovery helpers (repo layout)
# -----------------------------------------------------------------------------

def _find_one_csv(root: Path, pattern: str) -> Optional[Path]:
    try:
        root = Path(root)
        hits = sorted(root.rglob(pattern))
        return hits[0] if hits else None
    except Exception:
        return None


def discover_uefa_csvs(
    *,
    teams_root: Path,
    matches_root: Path,
    league: str,
    season_token: str = "2025-to-2026",
) -> Tuple[Optional[Path], Optional[Path]]:
    """Find TEAM and MATCH csv paths for a given UEFA league.

    Expected repo layout (as described by user):
      BAWA PORTO/TEAMS/<league>/europe-uefa-*-teams-<season>-stats.csv
      BAWA PORTO/Matches/<league>/europe-uefa-*-matches-<season>-stats.csv

    Returns (teams_csv, matches_csv) where each may be None.
    """

    league_dir_teams = Path(teams_root) / league
    league_dir_matches = Path(matches_root) / league

    teams_pat = f"*teams*{season_token}*stats.csv"
    matches_pat = f"*matches*{season_token}*stats.csv"

    teams_csv = _find_one_csv(league_dir_teams, teams_pat)
    matches_csv = _find_one_csv(league_dir_matches, matches_pat)

    return teams_csv, matches_csv


def build_snapshot_for_league(
    *,
    league: str,
    teams_root: Path,
    matches_root: Optional[Path] = None,
    season_token: str = "2025-to-2026",
    cfg: UefaContextConfig = DEFAULT_CFG,
    debug: bool = False,
) -> pd.DataFrame:
    """Convenience: load league TEAM CSV (and optional MATCH CSV) and build snapshot."""

    teams_csv, matches_csv = discover_uefa_csvs(
        teams_root=teams_root,
        matches_root=(matches_root or Path(".")),
        league=league,
        season_token=season_token,
    )

    if teams_csv is None or not teams_csv.exists():
        if debug:
            print(f"[uefa_context] teams CSV not found for league={league} under {teams_root}")
        return pd.DataFrame()

    tdf = pd.read_csv(teams_csv, low_memory=False)
    return compute_uefa_team_snapshot(tdf, matches_csv=matches_csv if matches_root else None, cfg=cfg, debug=debug)


def attach_uefa_context_for_league(
    df: pd.DataFrame,
    *,
    league: str,
    teams_root: Path,
    matches_root: Optional[Path] = None,
    season_token: str = "2025-to-2026",
    cfg: UefaContextConfig = DEFAULT_CFG,
    debug: bool = False,
) -> pd.DataFrame:
    """Convenience: build snapshot for league and attach to df.

    This does NOT filter rows by league; caller can do that if desired.
    """
    snap = build_snapshot_for_league(
        league=league,
        teams_root=teams_root,
        matches_root=matches_root,
        season_token=season_token,
        cfg=cfg,
        debug=debug,
    )
    if snap is None or snap.empty:
        return df
    return attach_uefa_context(df, snap, debug=debug)