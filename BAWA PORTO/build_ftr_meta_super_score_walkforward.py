#!/usr/bin/env python3
"""
Build a walkforward meta-model (logistic) to produce a "super-score" for FTR picks.

Features:
- Cat prob (model_p_for_bookie)
- XGB prob (model_p_for_bookie_xgb)
- OU25 prob (prob_over25_v2)
- BTTS prob (prob_btts_v2)
- odds band (binned implied prob)
- league prior (historical hit rate)

Output:
- FTR_META_SUPER_SCORE__WINDOW_SUMMARY.csv
- FTR_META_SUPER_SCORE__BY_LEAGUE.csv
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


@dataclass
class Window:
    window_id: str
    date_from: str
    date_to: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Walkforward meta super-score for FTR.")
    ap.add_argument("--base", required=True, help="predictions_output/walk_forward")
    ap.add_argument("--outdir", required=True, help="output directory for meta summary files")
    ap.add_argument("--manifest", default=None, help="Walkforward manifest CSV (optional)")
    ap.add_argument("--merged-dir", default="Matches/__merged__", help="Merged CSV directory")
    ap.add_argument("--source-implied-min", default="20", help="Used to resolve BOOKIE_IMPxx filenames")
    ap.add_argument("--min-train-rows", type=int, default=800)
    ap.add_argument("--min-window-rows", type=int, default=200)
    ap.add_argument("--odds-bins", default="0,0.2,0.3,0.4,0.5,0.6,0.7,1.0")
    return ap.parse_args()


def load_manifest(manifest_path: Path) -> List[Window]:
    df = pd.read_csv(manifest_path)
    req = {"window_id", "date_from", "date_to"}
    if not req.issubset(set(df.columns)):
        raise ValueError(f"Manifest missing columns: {req - set(df.columns)}")
    return [Window(r.window_id, r.date_from, r.date_to) for r in df.itertuples(index=False)]


def infer_windows_from_dirs(base: Path) -> List[Window]:
    windows = []
    for p in sorted(base.glob("w*/")):
        name = p.name
        # format: w001_YYYY_MM_DD_YYYY_MM_DD
        parts = name.split("_")
        if len(parts) >= 7:
            window_id = name
            date_from = f"{parts[1]}-{parts[2]}-{parts[3]}"
            date_to = f"{parts[4]}-{parts[5]}-{parts[6]}"
            windows.append(Window(window_id, date_from, date_to))
    return windows


def league_to_merged_path(merged_dir: Path, league: str) -> Path:
    league_tag = league.replace(" ", "_")
    return merged_dir / f"{league_tag}__merged.csv"


def compute_actual_ftr(df: pd.DataFrame) -> pd.Series:
    h = pd.to_numeric(df["home_team_goal_count"], errors="coerce")
    a = pd.to_numeric(df["away_team_goal_count"], errors="coerce")
    out = np.where(h > a, "HOME", np.where(h < a, "AWAY", "DRAW"))
    return pd.Series(out, index=df.index)


def load_predictions(window_dir: Path, source_implied_min: str) -> Path:
    # Prefer IMPxx if present
    pattern = f"BOOKIE_IMP{source_implied_min}_ALLMARKETS_*.csv"
    candidates = sorted((window_dir / "01_source").glob(pattern))
    if candidates:
        return candidates[0]
    # fallback to BOOKIE_ALLMARKETS
    candidates = sorted((window_dir / "01_source").glob("BOOKIE_ALLMARKETS_*.csv"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No source predictions found in {window_dir / '01_source'}")


def load_deploy_tiers(window_dir: Path, source_implied_min: str) -> pd.DataFrame:
    pattern = f"BOOKIE_IMP{source_implied_min}_ALLMARKETS_*__DEPLOY_TIER_*.csv"
    files = sorted((window_dir / "02_deploy").glob(pattern))
    if not files:
        # fallback to 01_source tier outputs if present
        files = sorted((window_dir / "01_source").glob(pattern))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_csv(p) for p in files]
    return pd.concat(frames, ignore_index=True, sort=False)


def build_training_frame(
    pred_df: pd.DataFrame,
    merged_cache: Dict[str, pd.DataFrame],
    merged_dir: Path,
) -> pd.DataFrame:
    # FTR only
    df = pred_df[pred_df["market"] == "ftr"].copy()
    if df.empty:
        return df

    # join actuals
    leagues = sorted(df["league"].dropna().unique().tolist())
    actual_frames = []
    for league in leagues:
        if league not in merged_cache:
            mp = league_to_merged_path(merged_dir, league)
            if not mp.exists():
                continue
            merged_cache[league] = pd.read_csv(mp, low_memory=False)
        mdf = merged_cache[league]
        keep_cols = ["fixture_key", "home_team_goal_count", "away_team_goal_count"]
        subset = mdf[keep_cols].copy()
        subset["league"] = league
        actual_frames.append(subset)

    if not actual_frames:
        return pd.DataFrame()

    actual_df = pd.concat(actual_frames, ignore_index=True)
    df = df.merge(actual_df, on=["fixture_key", "league"], how="left")
    df["actual_ftr"] = compute_actual_ftr(df)
    df["is_correct"] = (df["model_top_pick"] == df["actual_ftr"]).astype(int)
    df = df[df["actual_ftr"].notna()]
    return df


def add_features(
    df: pd.DataFrame,
    odds_bins: List[float],
    league_prior: Dict[str, float],
    *,
    all_odds_band_cols: List[str] | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()

    # base features
    for col in [
        "model_p_for_bookie",
        "model_p_for_bookie_xgb",
        "prob_over25_v2",
        "prob_btts_v2",
    ]:
        if col not in out.columns:
            out[col] = np.nan

    implied_col = "bookie_implied_novig" if "bookie_implied_novig" in out.columns else "bookie_implied"
    out["implied_prob"] = pd.to_numeric(out.get(implied_col), errors="coerce")

    out["odds_band"] = pd.cut(out["implied_prob"], bins=odds_bins, include_lowest=True)
    out["odds_band"] = out["odds_band"].astype(str)

    out["league_prior"] = out["league"].map(league_prior).fillna(np.nan)

    feat_cols = [
        "model_p_for_bookie",
        "model_p_for_bookie_xgb",
        "prob_over25_v2",
        "prob_btts_v2",
        "implied_prob",
        "league_prior",
    ]

    # one-hot odds band
    odds_dummies = pd.get_dummies(out["odds_band"], prefix="odds_band")
    if all_odds_band_cols is not None:
        for c in all_odds_band_cols:
            if c not in odds_dummies.columns:
                odds_dummies[c] = 0
        odds_dummies = odds_dummies[all_odds_band_cols]
    out = pd.concat([out, odds_dummies], axis=1)
    feat_cols.extend(list(odds_dummies.columns))

    return out, feat_cols


def fit_logistic(X: np.ndarray, y: pd.Series) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


def _fit_platt(x: np.ndarray, y: np.ndarray) -> LogisticRegression | None:
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 50:
        return None
    if len(np.unique(y)) < 2:
        return None
    mdl = LogisticRegression(max_iter=1000)
    mdl.fit(x.reshape(-1, 1), y)
    return mdl


def _fit_isotonic(x: np.ndarray, y: np.ndarray) -> IsotonicRegression | None:
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 80:
        return None
    if len(np.unique(y)) < 2:
        return None
    # isotonic requires at least some variation in x
    if len(np.unique(x)) < 10:
        return None
    mdl = IsotonicRegression(out_of_bounds="clip")
    mdl.fit(x, y)
    return mdl


def _apply_calibrator(x: np.ndarray, iso: IsotonicRegression | None, platt: LogisticRegression | None) -> tuple[np.ndarray, str]:
    if iso is not None:
        return iso.transform(x), "isotonic"
    if platt is not None:
        return platt.predict_proba(x.reshape(-1, 1))[:, 1], "platt"
    return x, "none"


def _align_matrix(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    parts = []
    for c in cols:
        if c in df.columns:
            col = df[c]
            # If duplicate column names exist, pandas returns a DataFrame; take first column.
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            s = pd.to_numeric(col, errors="coerce").fillna(0.0).to_numpy()
        else:
            s = np.zeros(len(df), dtype="float64")
        parts.append(s)
    if not parts:
        return np.empty((len(df), 0))
    return np.column_stack(parts)


def _pick_odds_from_1x2(df: pd.DataFrame) -> pd.Series:
    # Prefer bookie_od if available
    if "bookie_od" in df.columns:
        return pd.to_numeric(df["bookie_od"], errors="coerce")
    # Fallback: choose from 1X2 based on pick
    pick = df.get("bookie_pick", df.get("selection", df.get("model_top_pick", "")))
    pick = pick.astype("string").fillna("").str.upper().str.strip()
    pick = pick.replace({"1": "HOME", "X": "DRAW", "2": "AWAY", "H": "HOME", "D": "DRAW", "A": "AWAY"})
    od_home = pd.to_numeric(df.get("od_home", np.nan), errors="coerce")
    od_draw = pd.to_numeric(df.get("od_draw", np.nan), errors="coerce")
    od_away = pd.to_numeric(df.get("od_away", np.nan), errors="coerce")
    odds = od_home.where(pick.eq("HOME"), np.nan)
    odds = odds.fillna(od_draw.where(pick.eq("DRAW"), np.nan))
    odds = odds.fillna(od_away.where(pick.eq("AWAY"), np.nan))
    return odds


def compute_roi(df: pd.DataFrame) -> Tuple[float, float]:
    # ROI based on bookie_od or 1X2 fallback
    odds = _pick_odds_from_1x2(df)
    correct = df["is_correct"].astype(float)
    profit = np.where(correct == 1, odds - 1.0, -1.0)
    profit = profit[np.isfinite(profit)]
    if len(profit) == 0:
        return 0.0, 0.0
    roi = float(profit.mean())
    hit = float(correct.mean())
    return hit, roi


def main() -> None:
    args = parse_args()
    base = Path(args.base)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest) if args.manifest else None
    if manifest_path and manifest_path.exists():
        windows = load_manifest(manifest_path)
    else:
        windows = infer_windows_from_dirs(base)

    if not windows:
        raise SystemExit("No windows found.")

    merged_dir = Path(args.merged_dir)
    merged_cache: Dict[str, pd.DataFrame] = {}

    odds_bins = [float(x) for x in args.odds_bins.split(",")]

    train_frames: List[pd.DataFrame] = []
    window_rows = []
    league_rows = []

    for idx, w in enumerate(windows):
        window_dir = base / w.window_id
        if not window_dir.exists():
            continue

        pred_path = load_predictions(window_dir, args.source_implied_min)
        pred_df = pd.read_csv(pred_path)
        df = build_training_frame(pred_df, merged_cache, merged_dir)
        if df.empty:
            continue

        # current window features
        if train_frames:
            train_df = pd.concat(train_frames, ignore_index=True, sort=False)
        else:
            train_df = pd.DataFrame()

        if len(train_df) < args.min_train_rows or len(df) < args.min_window_rows:
            # not enough history yet
            train_frames.append(df)
            continue

        # league prior from training data
        league_prior = train_df.groupby("league")["is_correct"].mean().to_dict()
        # Ensure consistent odds-band columns across train/val
        train_df, feat_cols = add_features(train_df, odds_bins, league_prior)
        odds_band_cols = [c for c in feat_cols if c.startswith("odds_band_")]
        df, _ = add_features(df, odds_bins, league_prior, all_odds_band_cols=odds_band_cols)

        # drop NA in features
        train_df = train_df.dropna(subset=feat_cols + ["is_correct"])
        df = df.dropna(subset=feat_cols + ["is_correct"])
        if len(train_df) < args.min_train_rows or len(df) < args.min_window_rows:
            train_frames.append(df)
            continue

        X_tr = _align_matrix(train_df, feat_cols)
        y_tr = train_df["is_correct"]
        model = fit_logistic(X_tr, y_tr)

        # Align validation features to training columns/order
        X_va = _align_matrix(df, feat_cols)
        df["meta_super_score"] = model.predict_proba(X_va)[:, 1]

        # ---- Calibration (per-league where possible, global fallback) ----
        # Fit global calibrators on training
        train_scores = model.predict_proba(X_tr)[:, 1]
        y_tr_np = y_tr.to_numpy()
        global_iso = _fit_isotonic(train_scores, y_tr_np)
        global_platt = _fit_platt(train_scores, y_tr_np)

        df["meta_super_score_cal"] = df["meta_super_score"].to_numpy()
        df["meta_calibration_method"] = "none"

        # per-league calibration from training
        for lg, g in train_df.groupby("league"):
            if "meta_super_score" in g.columns:
                lg_scores = g["meta_super_score"].to_numpy()
            else:
                g_aligned = _align_matrix(g, feat_cols)
                lg_scores = model.predict_proba(g_aligned)[:, 1]
            lg_y = g["is_correct"].to_numpy()
            lg_iso = _fit_isotonic(lg_scores, lg_y)
            lg_platt = _fit_platt(lg_scores, lg_y)
            use_iso = lg_iso is not None
            use_platt = lg_platt is not None

            mask = df["league"] == lg
            if not mask.any():
                continue

            if use_iso or use_platt:
                cal, method = _apply_calibrator(df.loc[mask, "meta_super_score"].to_numpy(), lg_iso, lg_platt)
            else:
                cal, method = _apply_calibrator(df.loc[mask, "meta_super_score"].to_numpy(), global_iso, global_platt)

            df.loc[mask, "meta_super_score_cal"] = cal
            df.loc[mask, "meta_calibration_method"] = method

        # Consensus lane from deploy tiers
        tiers_df = load_deploy_tiers(window_dir, args.source_implied_min)
        tiers_df = tiers_df[tiers_df["market"] == "ftr"] if not tiers_df.empty else pd.DataFrame()
        if not tiers_df.empty and "ftr_priority" in tiers_df.columns:
            consensus = tiers_df[tiers_df["ftr_priority"] == "CONSENSUS_ELITE"].copy()
            # merge actuals
            consensus = consensus.merge(
                df[["fixture_key", "league", "actual_ftr", "is_correct", "bookie_od"]],
                on=["fixture_key", "league"],
                how="left",
            )
            consensus = consensus.dropna(subset=["is_correct"])
            consensus_n = len(consensus)
        else:
            consensus_n = 0
            consensus = pd.DataFrame()

        # meta top-k to match consensus coverage
        if consensus_n > 0:
            meta_pick = df.sort_values("meta_super_score_cal", ascending=False).head(consensus_n)
        else:
            meta_pick = df.sort_values("meta_super_score", ascending=False).head(0)

        cons_hit, cons_roi = compute_roi(consensus) if not consensus.empty else (0.0, 0.0)
        meta_hit, meta_roi = compute_roi(meta_pick) if not meta_pick.empty else (0.0, 0.0)

        window_rows.append({
            "window_id": w.window_id,
            "date_from": w.date_from,
            "date_to": w.date_to,
            "ftr_rows": len(df),
            "train_rows": len(train_df),
            "consensus_rows": consensus_n,
            "meta_rows": len(meta_pick),
            "consensus_hit_rate": cons_hit,
            "consensus_roi": cons_roi,
            "meta_hit_rate": meta_hit,
            "meta_roi": meta_roi,
        })

        # league summary
        if consensus_n > 0:
            cons_league = consensus.groupby("league").apply(lambda g: pd.Series({
                "consensus_rows": len(g),
                "consensus_hit_rate": compute_roi(g)[0],
                "consensus_roi": compute_roi(g)[1],
            })).reset_index()
        else:
            cons_league = pd.DataFrame(columns=["league","consensus_rows","consensus_hit_rate","consensus_roi"])

        meta_league = meta_pick.groupby("league").apply(lambda g: pd.Series({
            "meta_rows": len(g),
            "meta_hit_rate": compute_roi(g)[0],
            "meta_roi": compute_roi(g)[1],
        })).reset_index()

        league = pd.merge(cons_league, meta_league, on="league", how="outer")
        league["window_id"] = w.window_id
        league_rows.append(league)

        # write per-window meta scores for downstream merge
        out_window = df[[
            "fixture_key",
            "league",
            "market",
            "meta_super_score",
            "meta_super_score_cal",
            "meta_calibration_method",
        ]].copy()
        out_window["window_id"] = w.window_id
        out_window_path = outdir / f"FTR_META_SUPER_SCORE__WINDOW_{w.window_id}.csv"
        out_window.to_csv(out_window_path, index=False)

        train_frames.append(df)

    # write outputs
    if window_rows:
        dfw = pd.DataFrame(window_rows)
        dfw.to_csv(outdir / "FTR_META_SUPER_SCORE__WINDOW_SUMMARY.csv", index=False)
    if league_rows:
        dfl = pd.concat(league_rows, ignore_index=True, sort=False)
        dfl.to_csv(outdir / "FTR_META_SUPER_SCORE__BY_LEAGUE.csv", index=False)

    print(f"Wrote: {outdir / 'FTR_META_SUPER_SCORE__WINDOW_SUMMARY.csv'}")
    print(f"Wrote: {outdir / 'FTR_META_SUPER_SCORE__BY_LEAGUE.csv'}")


if __name__ == "__main__":
    main()
