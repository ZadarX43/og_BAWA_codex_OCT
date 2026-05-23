#!/usr/bin/env python3
"""backtest_ftr_consensus.py

Walk-forward backtest harness for OG FTR consensus/XOG.

Goal
- Produce a *true* backtest where:
  (1) training uses ONLY matches strictly before the test window
  (2) features used at inference are leak-safe (rolling/shifted)
  (3) we generate FTR model probabilities for the test window, then run the
      ftr_consensus consensus/XOG scoring and evaluate against realised outcomes.

This script is intentionally separate from deploy_rulebook.py and from your
production V2/V3 deploy path.

Outputs
- predictions_output/backtests/<LEAGUE_TAG>/<STAMP>/
    - preds.csv              (per-fixture predictions + consensus + XOG)
    - folds.csv              (per-fold summary)
    - summary.json           (overall summary)

Usage examples

Single window (range):
    python backtest_ftr_consensus.py --league "England Premier League" \
      --fold range --from 2025-08-01 --to 2025-12-31

Monthly folds:
    python backtest_ftr_consensus.py --league "England Premier League" \
      --fold monthly --from 2025-08-01 --to 2025-12-31

Weekly folds:
    python backtest_ftr_consensus.py --league "Spain La Liga" \
      --fold weekly --from 2025-08-01 --to 2025-12-31

Notes
- Uses CatBoost MultiClass by default (same family as V2).
- Feature frame selection is leak-aware (drops target/outcome columns and common post-match tokens).
- Rolling power ratings are computed per row using past matches only.
- Rolling team rates/H2H are attached if streaks_module is available (expected leak-safe).
"""


import argparse
import json
import os
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

# --- Optional: CatBoost (preferred for parity with V2) ---
try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None  # type: ignore

# --- Optional: canonical key/date helpers ---
try:
    from prediction_overlay import _match_key, _coalesce_match_date_series
except Exception:
    _match_key = None
    _coalesce_match_date_series = None

# --- Optional: rolling power ratings ---
try:
    from team_ratings import build_rolling_power_ratings as _build_rolling_power_ratings
except Exception:
    _build_rolling_power_ratings = None

# --- Optional: leak-safe rolling rates + h2h ---
try:
    from streaks_module import attach_team_rates as _attach_team_rates
except Exception:
    _attach_team_rates = None

try:
    from streaks_module import attach_h2h_streaks as _attach_h2h_streaks
except Exception:
    _attach_h2h_streaks = None

# --- Import consensus scoring from ftr_consensus (we reuse your proven scorer) ---
import ftr_consensus as _fc


# -----------------------------
# Helpers
# -----------------------------

def _league_tag(league: str) -> str:
    return str(league).strip().replace(" ", "_")


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _ensure_fixture_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "fixture_key" in out.columns:
        out["fixture_key"] = out["fixture_key"].astype("string").fillna("").str.strip()
    else:
        out["fixture_key"] = ""

    if callable(_match_key) and {"home_team_name", "away_team_name"}.issubset(out.columns):
        try:
            out["fixture_key"] = out.apply(_match_key, axis=1)
            out["fixture_key"] = out["fixture_key"].astype("string").fillna("").str.strip()
        except Exception:
            pass

    if out["fixture_key"].astype("string").fillna("").str.strip().eq("").any():
        # final fallback
        md = _coalesce_dt(out)
        out["fixture_key"] = (
            out.get("league", "").astype(str) + "||" +
            out.get("home_team_name", "").astype(str) + "||" +
            out.get("away_team_name", "").astype(str) + "||" +
            md.astype(str)
        )
        out["fixture_key"] = out["fixture_key"].astype("string").fillna("").str.strip()

    out = out[out["fixture_key"].ne("")].copy()
    return out


def _coalesce_dt(df: pd.DataFrame) -> pd.Series:
    # Prefer project coalescer
    if callable(_coalesce_match_date_series):
        try:
            dt = _coalesce_match_date_series(df)
            return pd.to_datetime(dt, errors="coerce", utc=True)
        except Exception:
            pass

    for c in ("match_date", "date_GMT", "date", "Date", "timestamp"):
        if c not in df.columns:
            continue
        if c == "timestamp":
            ts = pd.to_numeric(df[c], errors="coerce")
            if ts.notna().any():
                return pd.to_datetime(ts, errors="coerce", unit="s", utc=True)
        dt = pd.to_datetime(df[c], errors="coerce", utc=True)
        if dt.notna().any():
            return dt
    return pd.Series(pd.NaT, index=df.index)


def _actual_ftr(hg: float, ag: float) -> str:
    if hg > ag:
        return "HOME"
    if hg < ag:
        return "AWAY"
    return "DRAW"


def _poisson_1x2(lh: float, la: float, max_goals: int = 10) -> Tuple[float, float, float, float]:
    """Return (p_home, p_draw, p_away, p00)."""
    lh = float(max(0.05, lh))
    la = float(max(0.05, la))

    # pmf arrays
    k = np.arange(0, max_goals + 1)
    denom = np.array([math.factorial(int(i)) for i in k], dtype=float)
    denom = np.maximum(1.0, denom)
    ph = np.exp(-lh) * np.power(lh, k) / denom
    pa = np.exp(-la) * np.power(la, k) / denom

    # outer product for score matrix
    mat = np.outer(ph, pa)
    p_draw = float(np.trace(mat))
    p_home = float(np.tril(mat, k=-1).sum())
    p_away = float(np.triu(mat, k=1).sum())
    p00 = float(mat[0, 0])

    s = p_home + p_draw + p_away
    if s > 0:
        p_home, p_draw, p_away = p_home / s, p_draw / s, p_away / s
    return float(p_home), float(p_draw), float(p_away), float(p00)


def _derive_odds_cols(df: pd.DataFrame) -> Tuple[str, str, str]:
    cols = set(df.columns)
    # Most common in your feeds
    c_h = "odds_ft_home_team_win" if "odds_ft_home_team_win" in cols else ("od_home" if "od_home" in cols else "")
    c_d = "odds_ft_draw" if "odds_ft_draw" in cols else ("od_draw" if "od_draw" in cols else "")
    c_a = "odds_ft_away_team_win" if "odds_ft_away_team_win" in cols else ("od_away" if "od_away" in cols else "")
    return c_h, c_d, c_a


def _attach_market_microstructure(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c_h, c_d, c_a = _derive_odds_cols(out)

    if not c_h or not c_d or not c_a:
        # leave NaNs; consensus will fallback
        out["od_home"] = out.get("od_home", np.nan)
        out["od_draw"] = out.get("od_draw", np.nan)
        out["od_away"] = out.get("od_away", np.nan)
        return out

    out["od_home"] = _to_num(out[c_h])
    out["od_draw"] = _to_num(out[c_d])
    out["od_away"] = _to_num(out[c_a])

    inv_h = 1.0 / out["od_home"].replace(0, np.nan)
    inv_d = 1.0 / out["od_draw"].replace(0, np.nan)
    inv_a = 1.0 / out["od_away"].replace(0, np.nan)

    s = (inv_h + inv_d + inv_a)
    out["imp_home"] = inv_h / s
    out["imp_draw"] = inv_d / s
    out["imp_away"] = inv_a / s

    # closeness diagnostics
    pmax = pd.concat([out["imp_home"], out["imp_draw"], out["imp_away"]], axis=1).max(axis=1)
    p2 = pd.concat([out["imp_home"], out["imp_draw"], out["imp_away"]], axis=1).apply(lambda r: np.sort(r.to_numpy(dtype=float))[-2] if np.isfinite(r.to_numpy(dtype=float)).any() else np.nan, axis=1)
    out["bookie_spread"] = pmax - p2

    out["implied_prob_diff"] = (out["imp_home"] - out["imp_away"]).abs()
    out["odds_diff"] = (out["od_home"] - out["od_away"]).abs()

    return out


def _attach_basic_strength(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Ensure pre-match xG columns exist
    if "pre_match_xg_home" not in out.columns:
        if "Home Team Pre-Match xG" in out.columns:
            out["pre_match_xg_home"] = _to_num(out["Home Team Pre-Match xG"])
    if "pre_match_xg_away" not in out.columns:
        if "Away Team Pre-Match xG" in out.columns:
            out["pre_match_xg_away"] = _to_num(out["Away Team Pre-Match xG"])

    # xg_diff_abs
    xgh = _to_num(out.get("pre_match_xg_home", pd.Series(np.nan, index=out.index)))
    xga = _to_num(out.get("pre_match_xg_away", pd.Series(np.nan, index=out.index)))
    out["xg_diff_abs"] = (xgh - xga).abs()

    # PPG diff if available (prefer explicit home/away pre-match columns)
    ph = None
    pa = None
    for c in ("ppg_home_pre", "pre_match_ppg_home", "Pre-Match PPG (Home)"):
        if c in out.columns:
            ph = _to_num(out[c])
            break
    for c in ("ppg_away_pre", "pre_match_ppg_away", "Pre-Match PPG (Away)"):
        if c in out.columns:
            pa = _to_num(out[c])
            break

    if ph is not None and pa is not None:
        out["ppg_home_pre"] = ph
        out["ppg_away_pre"] = pa
        out["ppg_diff_pre"] = (ph - pa)
    else:
        out["ppg_diff_pre"] = _to_num(out.get("ppg_diff_pre", pd.Series(np.nan, index=out.index)))

    return out


def _compute_risk_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Compute lightweight risk flags when the dataset doesn't already have them."""
    out = df.copy()

    bs = _to_num(out.get("bookie_spread", pd.Series(np.nan, index=out.index)))
    ipd = _to_num(out.get("implied_prob_diff", pd.Series(np.nan, index=out.index)))
    mgn = _to_num(out.get("ftr_margin", pd.Series(np.nan, index=out.index)))
    pdiff = _to_num(out.get("power_diff", pd.Series(np.nan, index=out.index)))
    xgd = _to_num(out.get("xg_diff_abs", pd.Series(np.nan, index=out.index)))
    ppgd = _to_num(out.get("ppg_diff_pre", out.get("ppg_diff_pre", pd.Series(np.nan, index=out.index)))).abs()

    close_score = pd.Series(0.0, index=out.index)
    close_score = close_score + (0.20 - bs).clip(lower=0.0, upper=0.20) / 0.20
    close_score = close_score + (0.20 - ipd).clip(lower=0.0, upper=0.20) / 0.20
    close_score = (close_score / 2.0).clip(0.0, 1.0)

    flat_score = ((0.12 - mgn).clip(lower=0.0, upper=0.12) / 0.12).fillna(0.0)

    struct_flat = pd.Series(0.0, index=out.index)
    struct_flat = struct_flat + (1.0 - (pdiff.abs() / 15.0).clip(0.0, 1.0)).fillna(0.0)
    struct_flat = struct_flat + (1.0 - (xgd / 0.6).clip(0.0, 1.0)).fillna(0.0)
    struct_flat = struct_flat + (1.0 - (ppgd / 0.7).clip(0.0, 1.0)).fillna(0.0)
    struct_flat = (struct_flat / 3.0).clip(0.0, 1.0)

    dcs = (0.45 * close_score + 0.35 * flat_score + 0.20 * struct_flat).clip(0.0, 1.0)

    out["draw_chaos_score"] = _to_num(out.get("draw_chaos_score", dcs)).fillna(dcs)
    out["chaos_risk_flag"] = _to_num(out.get("chaos_risk_flag", (out["draw_chaos_score"] >= 0.65).astype(int))).fillna(0).astype(int)
    out["draw_risk_flag"] = _to_num(out.get("draw_risk_flag", ((close_score >= 0.60) & (out.get("exp_goals_sum", pd.Series(np.nan, index=out.index)) <= 2.70)).astype(int))).fillna(0).astype(int)
    out["not_glue_flag"] = _to_num(out.get("not_glue_flag", (close_score >= 0.50).astype(int))).fillna(0).astype(int)

    return out


def _make_feature_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Leak-aware feature selection for training FTR.

    Keeps numeric + select categorical IDs; drops outcome/obvious post-match columns.
    """
    work = df.copy()

    # Drop true targets/outcomes
    drop = {
        "home_team_goal_count",
        "away_team_goal_count",
        "total_goal_count",
        "home_team_goal_count_half_time",
        "away_team_goal_count_half_time",
        "total_goals_at_half_time",
        "home_team_goal_timings",
        "away_team_goal_timings",
        "home_team_corner_count",
        "away_team_corner_count",
        "home_team_yellow_cards",
        "away_team_yellow_cards",
        "home_team_red_cards",
        "away_team_red_cards",
        "status",
    }

    # Drop post-match / in-play tokens unless clearly safe
    leak_tokens = (
        "shots",
        "possession",
        "corner",
        "yellow",
        "red",
        "foul",
        "half_time",
        "first_half",
        "second_half",
        "goal_timings",
    )

    safe_prefixes = ("rolling", "streak_", "h2h_", "press_", "power_", "team_")

    def is_safe(col: str) -> bool:
        lc = col.lower()
        if lc.startswith(safe_prefixes):
            return True
        if "pre_match" in lc or lc.endswith("_pre") or lc.endswith("_pre_match"):
            return True
        if lc in ("bookie_spread", "implied_prob_diff", "odds_diff", "od_home", "od_draw", "od_away", "imp_home", "imp_draw", "imp_away"):
            return True
        return False

    for c in list(work.columns):
        if c in drop:
            continue
        lc = str(c).lower()
        if any(t in lc for t in leak_tokens) and (not is_safe(c)):
            drop.add(c)

    work = work.drop(columns=[c for c in drop if c in work.columns], errors="ignore")

    cat_cols = [c for c in ("home_team_name", "away_team_name") if c in work.columns]

    num_cols = work.select_dtypes(include=["number", "bool"]).columns.tolist()

    feats: List[str] = []
    feats.extend(cat_cols)
    for c in num_cols:
        if c not in feats:
            feats.append(c)

    feats = [c for c in feats if c in work.columns and not work[c].isna().all()]

    X = work[feats].copy()

    # Coerce
    for c in cat_cols:
        X[c] = X[c].astype("string").fillna("NA")
    for c in feats:
        if c in cat_cols:
            continue
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

    return X, feats


def _train_ftr_catboost(X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray, cat_cols: List[str], *, iters: int, seed: int) -> Any:
    if CatBoostClassifier is None:
        raise SystemExit("CatBoost is not installed. Install catboost or switch trainer.")

    cat_idx = [X_tr.columns.get_loc(c) for c in cat_cols if c in X_tr.columns]

    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="Accuracy",
        iterations=int(iters),
        learning_rate=0.05,
        depth=8,
        random_seed=int(seed),
        verbose=False,
        allow_writing_files=False,
    )

    model.fit(
        X_tr,
        y_tr,
        cat_features=cat_idx or None,
        eval_set=(X_va, y_va),
        use_best_model=True,
    )

    return model


def _folds(df: pd.DataFrame, mode: str, dt_col: str, start: pd.Timestamp, end: pd.Timestamp) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Return list of (test_start, test_end) windows."""
    mode = str(mode).lower()
    if mode == "range":
        return [(start, end)]

    d = pd.to_datetime(df[dt_col], errors="coerce", utc=True)
    d = d.dropna()
    if d.empty:
        return [(start, end)]

    # clamp
    start2 = max(start, d.min())
    end2 = min(end, d.max())

    windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

    if mode == "monthly":
        cur = pd.Timestamp(year=start2.year, month=start2.month, day=1, tz="UTC")
        while cur <= end2:
            nxt = (cur + pd.offsets.MonthBegin(1))
            if getattr(nxt, "tz", None) is None:
                nxt = nxt.tz_localize("UTC")
            else:
                nxt = nxt.tz_convert("UTC")
            w0 = cur
            w1 = min(end2, nxt - pd.Timedelta(seconds=1))
            if w1 >= start2:
                windows.append((max(w0, start2), w1))
            cur = nxt
        return windows

    if mode == "weekly":
        # align to Monday
        cur = (start2 - pd.Timedelta(days=int(start2.dayofweek))).normalize().tz_convert("UTC")
        while cur <= end2:
            nxt = cur + pd.Timedelta(days=7)
            w0 = cur
            w1 = min(end2, nxt - pd.Timedelta(seconds=1))
            if w1 >= start2:
                windows.append((max(w0, start2), w1))
            cur = nxt
        return windows

    # fallback
    return [(start2, end2)]


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward backtest for FTR consensus/XOG")
    ap.add_argument("--league", required=True)
    ap.add_argument("--matches-root", default="Matches")
    ap.add_argument("--out-root", default="predictions_output/backtests")

    ap.add_argument("--fold", choices=["range", "monthly", "weekly"], default="monthly")
    ap.add_argument("--from", dest="date_from", required=True, help="YYYY-MM-DD")
    ap.add_argument("--to", dest="date_to", required=True, help="YYYY-MM-DD")

    ap.add_argument("--min-train", type=int, default=400, help="Minimum training rows before running a fold")
    ap.add_argument("--min-test", type=int, default=15, help="Minimum test rows to score a fold")

    ap.add_argument("--iters", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)

    # Consensus knobs
    ap.add_argument("--tau", type=float, default=1.50)
    ap.add_argument("--xog-k", type=float, default=0.25)
    ap.add_argument("--close-spread-max", type=float, default=0.09)
    ap.add_argument("--close-ipd-max", type=float, default=0.11)
    ap.add_argument("--side-spread-min", type=float, default=0.16)
    ap.add_argument("--side-ipd-min", type=float, default=0.20)

    args = ap.parse_args()

    league = str(args.league).strip()
    tag = _league_tag(league)

    date_from = pd.Timestamp(str(args.date_from), tz="UTC")
    date_to = pd.Timestamp(str(args.date_to), tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    # Load matches
    mdir = Path(args.matches_root) / league
    if not mdir.exists():
        raise SystemExit(f"Matches folder not found: {mdir}")

    # Prefer matches.csv; else all csvs
    files = [mdir / "matches.csv"] if (mdir / "matches.csv").exists() else sorted(mdir.glob("*.csv"))
    frames = []
    for p in files:
        try:
            df = pd.read_csv(p)
            df["__src_csv"] = p.name
            frames.append(df)
        except Exception:
            continue

    if not frames:
        raise SystemExit(f"No CSVs found under {mdir}")

    df0 = pd.concat(frames, ignore_index=True, sort=False)
    df0["league"] = league

    # Canonical team columns
    if "home_team_name" not in df0.columns and "Home" in df0.columns:
        df0["home_team_name"] = df0["Home"]
    if "away_team_name" not in df0.columns and "Away" in df0.columns:
        df0["away_team_name"] = df0["Away"]

    # Goals
    for c in ("home_team_goal_count", "away_team_goal_count"):
        if c in df0.columns:
            df0[c] = _to_num(df0[c])

    # Date + key
    df0["match_date"] = _coalesce_dt(df0)
    df0 = _ensure_fixture_key(df0)

    # Realised-only rows (require goals)
    if {"home_team_goal_count", "away_team_goal_count"}.issubset(df0.columns):
        df_done = df0[df0[["home_team_goal_count", "away_team_goal_count"]].notna().all(axis=1)].copy()
    else:
        raise SystemExit("Missing goal columns for backtest")

    # Restrict to overall range (train can use earlier history; test is inside range)
    df_done = df_done.sort_values("match_date", ascending=True).reset_index(drop=True)

    # Attach rolling power ratings (pre-match) if available
    if callable(_build_rolling_power_ratings):
        try:
            df_done = _build_rolling_power_ratings(df_done)
            # drop post rating debug columns
            df_done = df_done.drop(columns=["home_power_rating_post_raw", "away_power_rating_post_raw"], errors="ignore")
        except Exception as e:
            print(f"⚠️ power_ratings attach failed: {e}")

    # Attach rolling team rates / h2h (best-effort)
    try:
        if callable(_attach_team_rates):
            df_done = _attach_team_rates(df_done)
    except Exception as e:
        print(f"ℹ️ attach_team_rates skipped: {e}")

    try:
        if callable(_attach_h2h_streaks):
            df_done = _attach_h2h_streaks(df_done, lookbacks=(5, 8))
    except Exception as e:
        print(f"ℹ️ attach_h2h_streaks skipped: {e}")

    # Market microstructure
    df_done = _attach_market_microstructure(df_done)

    # Basic strength cols
    df_done = _attach_basic_strength(df_done)

    # Compute simple lambdas from pre-match xG (fallbacks if missing)
    xgh = _to_num(df_done.get("pre_match_xg_home", pd.Series(np.nan, index=df_done.index)))
    xga = _to_num(df_done.get("pre_match_xg_away", pd.Series(np.nan, index=df_done.index)))
    if xgh.isna().all() or xga.isna().all():
        # fallback to rolling xg_for
        xgh = _to_num(df_done.get("xg_for_avg_5_home", pd.Series(1.35, index=df_done.index))).fillna(1.35)
        xga = _to_num(df_done.get("xg_for_avg_5_away", pd.Series(1.15, index=df_done.index))).fillna(1.15)

    df_done["lambda_home"] = xgh.clip(lower=0.20, upper=3.50)
    df_done["lambda_away"] = xga.clip(lower=0.20, upper=3.50)
    df_done["exp_goals_sum"] = (df_done["lambda_home"] + df_done["lambda_away"]).clip(lower=0.20, upper=7.0)

    # Poisson masses + p00
    ph, pdw, pa, p00 = [], [], [], []
    for lh, la in zip(df_done["lambda_home"].to_numpy(dtype=float), df_done["lambda_away"].to_numpy(dtype=float)):
        a, b, c, d = _poisson_1x2(lh, la, max_goals=10)
        ph.append(a); pdw.append(b); pa.append(c); p00.append(d)
    df_done["p_home_pois"] = ph
    df_done["p_draw_pois"] = pdw
    df_done["p_away_pois"] = pa
    df_done["p00_est"] = p00

    # Risk flags
    df_done = _compute_risk_flags(df_done)

    # Targets
    hg = _to_num(df_done["home_team_goal_count"])
    ag = _to_num(df_done["away_team_goal_count"])
    y = np.where(hg > ag, 0, np.where(hg == ag, 1, 2)).astype(int)

    # Feature frame for model
    X_all, feats = _make_feature_frame(df_done)
    cat_cols = [c for c in ("home_team_name", "away_team_name") if c in X_all.columns]

    # Folds
    df_done = df_done.assign(__dt=pd.to_datetime(df_done["match_date"], errors="coerce", utc=True))
    windows = _folds(df_done, args.fold, "__dt", date_from, date_to)

    stamp = pd.Timestamp.utcnow().strftime("%Y-%m-%d_%H%M%S")
    outdir = Path(args.out_root) / tag / stamp
    outdir.mkdir(parents=True, exist_ok=True)

    fold_rows: List[Dict[str, Any]] = []
    pred_rows: List[pd.DataFrame] = []

    for w0, w1 in windows:
        test_mask = (df_done["__dt"] >= w0) & (df_done["__dt"] <= w1)
        test_idx = df_done.index[test_mask]
        if len(test_idx) < int(args.min_test):
            continue

        train_mask = (df_done["__dt"] < w0)
        train_idx = df_done.index[train_mask]
        if len(train_idx) < int(args.min_train):
            continue

        X_tr = X_all.loc[train_idx]
        y_tr = y[train_idx]

        X_te = X_all.loc[test_idx]
        y_te = y[test_idx]

        # Train model
        # Use a small validation slice from the end of train for early stopping
        tr_dt = df_done.loc[train_idx, "__dt"]
        order = tr_dt.sort_values(kind="mergesort").index
        cut = int(max(1, round(0.85 * len(order))))
        tr_i = order[:cut]
        va_i = order[cut:]

        X_tr2 = X_all.loc[tr_i]
        y_tr2 = y[tr_i]
        X_va2 = X_all.loc[va_i]
        y_va2 = y[va_i]

        model = _train_ftr_catboost(X_tr2, y_tr2, X_va2, y_va2, cat_cols, iters=int(args.iters), seed=int(args.seed))

        proba = np.asarray(model.predict_proba(X_te), dtype=float)
        # Ensure shape (n,3)
        if proba.ndim != 2 or proba.shape[1] != 3:
            continue

        # Attach model probs to test rows
        te = df_done.loc[test_idx].copy()
        te["confidence_home"] = proba[:, 0]
        te["confidence_draw"] = proba[:, 1]
        te["confidence_away"] = proba[:, 2]
        te["model_top_pick"] = np.argmax(proba, axis=1)
        te["model_strength"] = np.max(proba, axis=1)
        srt = np.sort(proba, axis=1)
        te["ftr_margin"] = srt[:, -1] - srt[:, -2]
        te["actual_pick"] = ["HOME" if v == 0 else ("DRAW" if v == 1 else "AWAY") for v in y_te]

        # Run consensus scorer (reuse ftr_consensus functions)
        out_rows: List[Dict[str, Any]] = []

        # Enumerate to keep a stable row->proba alignment
        for i_row, (_, r) in enumerate(te.iterrows()):
            # Lane
            lane = _fc._lane_for_row(
                bookie_spread=float(r.get("bookie_spread", np.nan)),
                implied_prob_diff=float(r.get("implied_prob_diff", np.nan)),
                power_diff=float(r.get("power_diff", np.nan)),
                xg_diff_abs=float(r.get("xg_diff_abs", np.nan)),
                ppg_diff_abs=float(abs(float(r.get("ppg_diff_pre", np.nan))) if np.isfinite(r.get("ppg_diff_pre", np.nan)) else np.nan),
                ftr_margin=float(r.get("ftr_margin", np.nan)),
                close_spread_max=float(args.close_spread_max),
                close_ipd_max=float(args.close_ipd_max),
                side_spread_min=float(args.side_spread_min),
                side_ipd_min=float(args.side_ipd_min),
            )

            # p_model
            p_model = (
                float(r.get("confidence_home", np.nan)),
                float(r.get("confidence_draw", np.nan)),
                float(r.get("confidence_away", np.nan)),
            )

            # p_bk
            p_bk = (
                float(r.get("imp_home", np.nan)),
                float(r.get("imp_draw", np.nan)),
                float(r.get("imp_away", np.nan)),
            )

            # xg diff sign
            pre_xg_h = float(r.get("pre_match_xg_home", np.nan))
            pre_xg_a = float(r.get("pre_match_xg_away", np.nan))
            xg_diff = (pre_xg_h - pre_xg_a) if (np.isfinite(pre_xg_h) and np.isfinite(pre_xg_a)) else np.nan

            scores, rc, _ = _fc._score_outcomes(
                p_model=p_model,
                p_bk=p_bk,
                power_diff=float(r.get("power_diff", np.nan)),
                ppg_diff=float(r.get("ppg_diff_pre", np.nan)),
                xg_diff=float(xg_diff),
                exp_goals_sum=float(r.get("exp_goals_sum", np.nan)),
                p00_est=float(r.get("p00_est", np.nan)),
                p_home_fts=float(r.get("p_home_fts", np.nan)),
                p_away_fts=float(r.get("p_away_fts", np.nan)),
                home_ge2=float(r.get("home_ge2_confidence", np.nan)),
                away_ge2=float(r.get("away_ge2_confidence", np.nan)),
                home_ge3=float(r.get("home_ge3_confidence", np.nan)),
                away_ge3=float(r.get("away_ge3_confidence", np.nan)),
                bookie_spread=float(r.get("bookie_spread", np.nan)),
                implied_prob_diff=float(r.get("implied_prob_diff", np.nan)),
                p_home_pois=float(r.get("p_home_pois", np.nan)),
                p_draw_pois=float(r.get("p_draw_pois", np.nan)),
                p_away_pois=float(r.get("p_away_pois", np.nan)),
                scored_rate_5_home=float(r.get("scored_rate_5_home", np.nan)),
                scored_rate_5_away=float(r.get("scored_rate_5_away", np.nan)),
                conceded_rate_5_home=float(r.get("conceded_rate_5_home", np.nan)),
                conceded_rate_5_away=float(r.get("conceded_rate_5_away", np.nan)),
                goaliness_avg_5_home=float(r.get("goaliness_avg_5_home", np.nan)),
                goaliness_avg_5_away=float(r.get("goaliness_avg_5_away", np.nan)),
                h2h_goaliness_avg=float(r.get("h2h_goaliness_avg", np.nan)),
                draw_risk_flag=float(r.get("draw_risk_flag", np.nan)),
                chaos_risk_flag=float(r.get("chaos_risk_flag", np.nan)),
                draw_chaos_score=float(r.get("draw_chaos_score", np.nan)),
                not_glue_flag=float(r.get("not_glue_flag", np.nan)),
                return_components=False,
            )

            probs = _fc._softmax(scores, tau=float(args.tau))
            top_i = int(np.argmax(probs))
            top_p = float(probs[top_i])
            second_p = float(np.sort(probs)[-2])
            margin = float(top_p - second_p)

            consensus_pick = "DRAW" if top_i == 1 else ("HOME" if top_i == 0 else "AWAY")

            # XOG
            k = float(args.xog_k)
            xog_home = 3.0 * float(_fc._sigmoid(np.array([(probs[0] - 1/3) / k]))[0])
            xog_draw = 3.0 * float(_fc._sigmoid(np.array([(probs[1] - 1/3) / k]))[0])
            xog_away = 3.0 * float(_fc._sigmoid(np.array([(probs[2] - 1/3) / k]))[0])
            xogs = np.array([xog_home, xog_draw, xog_away], dtype=float)
            i_best = int(np.argmax(xogs))
            xog_pick = "DRAW" if i_best == 1 else ("HOME" if i_best == 0 else "AWAY")
            x_sorted = np.sort(xogs)
            xog_pick_score = float(x_sorted[-1])
            xog_spread = float(x_sorted[-1] - x_sorted[-2])

            xog_tier = _fc._xog_tier(xog_pick_score, xog_spread, lane)

            hit = int(consensus_pick == str(r.get("actual_pick", "")).upper())

            # Raw model pick label (IMPORTANT: do NOT use `or 1` because 0 is falsy)
            _mtp = pd.to_numeric(r.get("model_top_pick", np.nan), errors="coerce")
            if np.isfinite(_mtp):
                mtp_idx = int(_mtp)
            else:
                mtp_idx = 1
            model_pick = "HOME" if mtp_idx == 0 else ("DRAW" if mtp_idx == 1 else "AWAY")

            actual_pick_u = str(r.get("actual_pick", "")).upper().strip()
            model_hit = int(model_pick == actual_pick_u)

            out_rows.append({
                "league": league,
                "test_start": str(w0),
                "test_end": str(w1),
                "fixture_key": str(r.get("fixture_key", "")),
                "match_date": str(r.get("match_date", "")),
                "home_team_name": str(r.get("home_team_name", "")),
                "away_team_name": str(r.get("away_team_name", "")),
                "actual_pick": actual_pick_u,

                # Raw model diagnostics
                "confidence_home": float(r.get("confidence_home", np.nan)),
                "confidence_draw": float(r.get("confidence_draw", np.nan)),
                "confidence_away": float(r.get("confidence_away", np.nan)),
                "model_strength": float(r.get("model_strength", np.nan)),
                "ftr_margin": float(r.get("ftr_margin", np.nan)),
                "model_top_pick": model_pick,
                "model_hit": model_hit,

                "consensus_lane": lane,
                "consensus_pick": consensus_pick,
                "consensus_confidence": top_p,
                "consensus_margin": margin,
                "xog_pick": xog_pick,
                "xog_pick_score": xog_pick_score,
                "xog_spread": xog_spread,
                "xog_tier": xog_tier,
                "reason_codes": rc,
                "hit": hit,

                # Diagnostics
                "bookie_spread": float(r.get("bookie_spread", np.nan)),
                "implied_prob_diff": float(r.get("implied_prob_diff", np.nan)),
                "power_diff": float(r.get("power_diff", np.nan)),
                "ppg_diff_pre": float(r.get("ppg_diff_pre", np.nan)),
                "xg_diff_abs": float(r.get("xg_diff_abs", np.nan)),
                "exp_goals_sum": float(r.get("exp_goals_sum", np.nan)),
                "p00_est": float(r.get("p00_est", np.nan)),
                "p_draw_pois": float(r.get("p_draw_pois", np.nan)),
                "draw_chaos_score": float(r.get("draw_chaos_score", np.nan)),

                "p_model_home": float(proba[i_row, 0]) if (i_row < proba.shape[0]) else np.nan,
                "p_model_draw": float(proba[i_row, 1]) if (i_row < proba.shape[0]) else np.nan,
                "p_model_away": float(proba[i_row, 2]) if (i_row < proba.shape[0]) else np.nan,
            })

        pred_df = pd.DataFrame(out_rows)
        pred_rows.append(pred_df)

        # Fold metrics
        acc = float(pred_df["hit"].mean()) if len(pred_df) else 0.0
        fold_rows.append({
            "league": league,
            "test_start": str(w0),
            "test_end": str(w1),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "accuracy": acc,
        })

    if not pred_rows:
        raise SystemExit("No folds produced (check date range and min-train/min-test)")

    preds = pd.concat(pred_rows, axis=0, ignore_index=True)
    folds_df = pd.DataFrame(fold_rows)

    preds_path = outdir / "preds.csv"
    folds_path = outdir / "folds.csv"
    summary_path = outdir / "summary.json"

    preds.to_csv(preds_path, index=False)
    folds_df.to_csv(folds_path, index=False)

    overall = {
        "league": league,
        "tag": tag,
        "fold_mode": args.fold,
        "date_from": str(date_from),
        "date_to": str(date_to),
        "n_preds": int(len(preds)),
        "acc": float(preds["hit"].mean()) if len(preds) else 0.0,
        "lane_counts": preds["consensus_lane"].value_counts().to_dict(),
        "xog_tier_counts": preds["xog_tier"].value_counts().to_dict(),
    }

    summary_path.write_text(json.dumps(overall, indent=2), encoding="utf-8")

    print("WROTE:", preds_path)
    print("WROTE:", folds_path)
    print("WROTE:", summary_path)
    print("overall_acc:", round(overall["acc"], 3), "| n=", overall["n_preds"])


if __name__ == "__main__":
    main()