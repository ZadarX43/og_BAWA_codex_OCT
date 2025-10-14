"""
side_prob_models.py
──────────────────────────────────────────────────────────────────────────
Light‑weight “side‑probability” helpers that replace the neutral
placeholder columns (`draw_prone`, `home_under15_prob`,
`away_under15_prob`, `under15_combined_prob`, `under25_combined_prob`)
with data‑driven estimates.

v1 Design
---------
* one pickle per (league, label) under  ModelStore/side_probs/
* Gradient‑Boosting + isotonic calibration  ➜ decent out‑of‑box accuracy
* Fast to train; can be swapped for XGBoost/LightGBM later
"""
from __future__ import annotations
import os   
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from sklearn.ensemble import GradientBoostingClassifier

# --- Robust calibrator (sklearn >=1.5) -------------------------------------
if "_make_calibrator" not in globals():
    def _make_calibrator(base_estimator, method: str = "isotonic", cv: int = 3):
        """Return CalibratedClassifierCV using modern signature (estimator=...)."""
        try:
            from sklearn.utils.fixes import _wrap_in_frozen_estimator as _Freeze  # type: ignore
            base_estimator = _Freeze(base_estimator)
        except Exception:
            pass
        from sklearn.calibration import CalibratedClassifierCV
        return CalibratedClassifierCV(estimator=base_estimator, method=method, cv=cv)

# --- OOF helper imports --------------------------------------------
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone


# --------------------------------------------------------------------
# Lazy imports of heavy helpers — avoids circular‑import with
# 00_baseline_ftr_pipeline.py
# --------------------------------------------------------------------
import importlib

def _strip_leaks_lazy(df: pd.DataFrame) -> pd.DataFrame:
    """Call baseline_ftr_pipeline.strip_leaks only when first needed."""
    strip_leaks = importlib.import_module("_baseline_ftr_pipeline").strip_leaks
    return strip_leaks(df)

def _drop_constant_cols_lazy(df: pd.DataFrame) -> pd.DataFrame:
    """Same for drop_constant_cols."""
    drop_constant_cols = importlib.import_module("_baseline_ftr_pipeline").drop_constant_cols
    return drop_constant_cols(df)

# --------------------------------------------------------------------
#  Out‑of‑fold probability helper – prevents target leakage when the
#  side‑prob columns are generated on the same dataframe that will be
#  used for training downstream models.
# --------------------------------------------------------------------
def _oof_probs(base_model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> np.ndarray:
    """
    Return leak‑free out‑of‑fold probabilities for the positive class.

    Parameters
    ----------
    base_model : *unfitted* sklearn classifier supporting predict_proba
    X, y       : feature matrix and binary target
    n_splits   : StratifiedKFold splits (default 5)

    Notes
    -----
    A fresh clone of *base_model* is trained in each fold so the caller
    can safely pass an already‑fitted estimator without it retaining
    state between folds.
    """
    oof = np.zeros(len(X), dtype=float)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr_idx, val_idx in cv.split(X, y):
        clf = clone(base_model)
        clf.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof[val_idx] = clf.predict_proba(X.iloc[val_idx])[:, 1]
    return oof

# Project‑wide constants -------------------------------------------------
from constants import MODEL_DIR

# ----------------------------------------------------------------------------
# Central registry of the side‑prob models we want
# ----------------------------------------------------------------------------
SIDE_MODELS: Dict[str, List[str]] = {
    # label                    richer feature seed (λ̂ + team-season priors)
    "draw_prone": [
        # parity & volatility
        "ppg_diff","elo_diff","rest_diff","xg_diff_abs","rolling10_press_intensity_diff",
        # team-season priors & diffs
        "xg_net_diff","xg_ratio_diff","home_advantage_diff",
        "avg_total_goals_diff","gfpm_diff","gapm_diff",
        "cards_per_match_diff","corners_per_match_diff",
        "btts_rate_diff","over25_rate_diff",
        # half-time profile
        "ht_draw_rate_diff","ht_gd_diff",
    ],

    "home_under15_prob":     [
        "home_goals_pred","lambda_home",
        "gapm_diff","clean_sheet_rate_diff","home_xg_against_idx"
    ],
    "away_under15_prob":     [
        "away_goals_pred","lambda_away",
        "gapm_diff","clean_sheet_rate_diff","away_xg_against_idx"
    ],
    "under15_combined_prob": [
        "home_goals_pred","away_goals_pred","lambda_home","lambda_away",
        "gapm_diff","clean_sheet_rate_diff"
    ],
    "under25_combined_prob": [
        "home_goals_pred","away_goals_pred","lambda_home","lambda_away",
        "avg_total_goals_diff","btts_rate_diff"
    ],

    "over15":    [
        "lambda_home","lambda_away","exp_goals_sum","lam_parity",
        "ppg_diff","xg_diff_abs","xg_net_diff","avg_total_goals_diff","btts_rate_diff",
        "rolling5_home_sot_ratio","rolling5_away_sot_ratio"
    ],
    "over35":    [
        "lambda_home","lambda_away","exp_goals_sum","lam_parity",
        "ppg_diff","xg_diff_abs","xg_net_diff","avg_total_goals_diff","btts_rate_diff",
        "rolling5_home_sot_ratio","rolling5_away_sot_ratio"
    ],
    "over45":    [
        "lambda_home","lambda_away","exp_goals_sum","lam_parity",
        "ppg_diff","xg_diff_abs","xg_net_diff","avg_total_goals_diff","btts_rate_diff",
        "rolling5_home_sot_ratio","rolling5_away_sot_ratio"
    ],
    "btts": [
        # λ̂ + team priors helpful for BTTS
        "lambda_home","lambda_away","exp_goals_sum",
        "ppg_diff","xg_diff_abs","xg_net_diff",
        "avg_total_goals_diff","btts_rate_diff",
        "rolling5_home_sot_ratio","rolling5_away_sot_ratio"
    ],
    "over25": [
        "lambda_home","lambda_away","exp_goals_sum","lam_parity",
        "ppg_diff","xg_diff_abs","xg_net_diff","avg_total_goals_diff","btts_rate_diff",
        "rolling5_home_sot_ratio","rolling5_away_sot_ratio"
    ],
    # First-half heads (prob only; no odds needed)
    "over15_fh": [
        "lambda_home","lambda_away","ppg_diff",
        "rolling5_home_sot_ratio","rolling5_away_sot_ratio",
        "rolling5_press_z_diff","ewma_ppg_diff",
        # half-time priors
        "ht_draw_rate_diff","ht_gd_diff"
    ],
    "btts_fh":   [
        "lambda_home","lambda_away","ppg_diff",
        "rolling5_home_sot_ratio","rolling5_away_sot_ratio",
        "rolling5_press_z_diff","ewma_ppg_diff",
        # half-time priors
        "ht_draw_rate_diff","ht_gd_diff"
    ],
}

# Where we persist pickles
SIDE_PROB_DIR = Path(MODEL_DIR) / "side_probs"
SIDE_PROB_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _model_path(label: str, league: str) -> Path:
    """Return Path to the cached pickle for *label* & *league*."""
    tag = league.replace(" ", "_")
    return SIDE_PROB_DIR / f"{tag}_{label}.pkl"

def _safe_X(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """
    Numeric, finite, NaN-free feature matrix plus constant-/leak-pruning.
    Tolerates empty/constant matrices and falls back gracefully.
    """
    import pandas as _pd
    X = df[feats].copy()
    # Sanitize numerics
    X = X.apply(_pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Drop constant columns; if all constant, keep at least one to avoid downstream errors
    try:
        X = _drop_constant_cols_lazy(X)
    except Exception:
        try:
            var = X.var(axis=0)
            keep = var[var > 0.0].index
            if len(keep) == 0:
                keep = X.columns[:1]  # keep the first column to avoid empty feature set
            X = X[keep]
        except Exception:
            pass
    # Leak strip (best effort)
    try:
        X = _strip_leaks_lazy(X)
    except Exception:
        pass
    return X

# Determine if we can derive a training target for the given label on this frame
def _has_target_for_label(label: str, df: pd.DataFrame) -> bool:
    if label == "draw_prone":
        return ("FTR" in df.columns) or ("home_team_goal_count" in df.columns and "away_team_goal_count" in df.columns)
    if label in ("home_under15_prob", "away_under15_prob", "under15_combined_prob", "under25_combined_prob",
                 "over15", "over35", "over45", "over25", "btts"):
        # full‑time targets rely on final goals
        need = {"home_team_goal_count", "away_team_goal_count"}
        return need.issubset(df.columns)
    if label in ("over15_fh", "btts_fh"):
        # first‑half targets rely on HT goals
        need = {"total_goals_at_half_time"} if label == "over15_fh" else {"home_team_goal_count_half_time", "away_team_goal_count_half_time"}
        return need.issubset(df.columns)
    return False

def _derive_target(label: str, df: pd.DataFrame) -> pd.Series:
    """Create a binary target for each side‑prob label."""
    if label == "draw_prone":
        # Prefer explicit FTR when available
        if "FTR" in df.columns:
            s = df["FTR"]
            # numeric 0/1/2 mapping or string mapping (H/D/A)
            try:
                sn = pd.to_numeric(s, errors="coerce")
                if sn.notna().any():
                    return (sn == 1).astype(int)
            except Exception:
                pass
            m = {"H":0, "D":1, "A":2, "h":0, "d":1, "a":2,
                 "home":0, "draw":1, "away":2}
            return s.astype(str).str.strip().map(m).eq(1).astype(int)
        # Fallback: derive from final goals (draw iff equal)
        ht = pd.to_numeric(df.get("home_team_goal_count"), errors="coerce")
        at = pd.to_numeric(df.get("away_team_goal_count"), errors="coerce")
        return (ht.eq(at) & ht.notna() & at.notna()).astype(int)

    if label == "home_under15_prob":
        return (df["home_team_goal_count"] <= 1).astype(int)

    if label == "away_under15_prob":
        return (df["away_team_goal_count"] <= 1).astype(int)

    if label == "under15_combined_prob":
        return (
            (df["home_team_goal_count"] + df["away_team_goal_count"]) <= 1
        ).astype(int)

    if label == "under25_combined_prob":
        return (
            (df["home_team_goal_count"] + df["away_team_goal_count"]) <= 2
        ).astype(int)

    if label == "over15":
        return (df["total_goal_count"] >= 2).astype(int)
    if label == "over35":
        return (df["total_goal_count"] >= 4).astype(int)
    if label == "over45":
        return (df["total_goal_count"] >= 5).astype(int)
    if label == "over25":
        return (df["total_goal_count"] >= 3).astype(int)
    if label == "btts":
        return ((df["home_team_goal_count"] > 0) & (df["away_team_goal_count"] > 0)).astype(int)
    if label == "over15_fh":
        return (df["total_goals_at_half_time"] >= 2).astype(int)
    if label == "btts_fh":
        ht = df.get("home_team_goal_count_half_time")
        at = df.get("away_team_goal_count_half_time")
        return ((ht > 0) & (at > 0)).astype(int)

    raise ValueError(f"Unknown side‑prob label: {label}")

# ----------------------------------------------------------------------------
# Core training / loading
# ----------------------------------------------------------------------------
def load_or_train_side_model(
    label: str,
    feature_cols: list[str],
    league: str = "generic",
    df: pd.DataFrame | None = None,
):
    """
    Load a cached side‑prob model or fit & cache a new one.

    Parameters
    ----------
    label        : one of SIDE_MODELS keys
    feature_cols : list of column names to feed the classifier
    league       : league string used to name the pickle
    df           : training dataframe (needed only when pickle absent)
    """
    if label not in SIDE_MODELS:
        raise ValueError(f"label must be one of {list(SIDE_MODELS.keys())}")

    path = _model_path(label, league)

    # Fast‑path → already cached
    if path.exists():
        try:
            return joblib.load(path)
        except Exception:
            # Corrupted / incompatible pickle – fall through to retrain
            pass

    # Need data to train
    if df is None or df.empty:
        raise RuntimeError(
            f"Side‑prob model '{label}' not found on disk and no training "
            "data provided."
        )

    # Filter to available features (caller should already have resolved)
    feature_cols = [c for c in feature_cols if (df is not None and c in df.columns)]
    if df is None or len(feature_cols) == 0:
        raise RuntimeError(
            f"Cannot train '{label}' – no usable features provided"
        )

    X = _safe_X(df, feature_cols)
    feature_cols = list(X.columns)               # keep the model schema in sync
    y = _derive_target(label, df)

    # Simple GBM + isotonic calib
    # Note: calibration is kept; SMOTE is intentionally not used here.
    base = GradientBoostingClassifier(random_state=42)
    model = _make_calibrator(base, method="isotonic", cv=3)
    model.fit(X, y)
    # expose the learned schema for safe inference
    model.feature_names_in_ = np.array(feature_cols, dtype=object)

    # Cache for future runs
    try:
        joblib.dump(model, path)
    except Exception:
        # Non‑fatal – continue without caching
        pass

    return model

# ----------------------------------------------------------------------------
def attach_side_probabilities(
    match_df: pd.DataFrame,
    league_tag: str | None = None
) -> pd.DataFrame:
    """
    Add/overwrite the five side‑prob columns with model predictions.

    Missing features → column left unchanged (neutral placeholder).
    """
    if match_df.empty:
        return match_df

    league = (league_tag or match_df.get("league_name", "generic")).replace(" ", "_")
    out = match_df.copy()

    for label, feats in SIDE_MODELS.items():
        # Resolve available features with sensible minima per label
        avail = [f for f in feats if f in out.columns]
        # Allow leaner minimums for some labels in sparse contexts
        if label in ("home_under15_prob","away_under15_prob"):
            min_feats = 1
        elif label == "draw_prone":
            min_feats = int(os.getenv("SIDE_MIN_FEATS_DRAW", "2"))  # default 2
        else:
            min_feats = int(os.getenv("SIDE_MIN_FEATS_DEFAULT", "3"))
        if len(avail) < min_feats:
            if os.getenv("VERBOSE_QUICK","0") == "1":
                print(f"⏭️ side-prob '{label}': insufficient features ({len(avail)}/{min_feats}) in {league}")
            continue

        model = load_or_train_side_model(label, avail, league, df=out)
        X_pred = _safe_X(out, list(model.feature_names_in_))      # align & sanitise

        # Determine whether we are in *training* context (target can be derived)
        training_mode = _has_target_for_label(label, out)

        live_col = f"prob_{label}"
        oof_col  = f"oof_prob_{label}"
        if training_mode:
            y_bin = _derive_target(label, out)
            # Use a fresh base estimator for OOF to avoid leakage; calibrator is used for live inference
            base = GradientBoostingClassifier(random_state=42)
            oof_probs_for_label = _oof_probs(base, X_pred, y_bin)
            out[oof_col] = oof_probs_for_label
            # Backward compat: also expose under legacy name
            out[label] = out[oof_col]
            # FH alias without underscore sometimes expected by allowlist
            if label == "over15_fh":
                if "oof_prob_over15fh" not in out.columns:
                    out["oof_prob_over15fh"] = out[oof_col].values
        else:
            # Live / inference – calibrated model
            live_probs = model.predict_proba(X_pred)[:, 1]
            out[live_col] = live_probs
            # Backward compat: also expose under legacy name
            out[label] = out[live_col]

    return out

__all__ = ["load_or_train_side_model", "attach_side_probabilities", "SIDE_MODELS"]