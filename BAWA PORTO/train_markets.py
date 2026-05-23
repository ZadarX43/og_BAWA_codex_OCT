import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import argparse
import pickle

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, accuracy_score

# Reuse preprocessing helpers from the overlay so train/infer match
try:
    from prediction_overlay import (
        apply_safe_renames_and_whitelist,
        _coerce_numeric_like,
        _force_drop_leaky_cols,
    )
except Exception:
    # Soft fallback if overlay is not importable in this context
    def apply_safe_renames_and_whitelist(df: pd.DataFrame) -> pd.DataFrame:
        return df
    def _coerce_numeric_like(df: pd.DataFrame, odds_whitelist=None) -> pd.DataFrame:
        return df
    def _force_drop_leaky_cols(df: pd.DataFrame) -> pd.DataFrame:
        return df


# ------------------------------
# Picklable probability calibration helpers
# ------------------------------


def _safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Stable logit transform for probabilities."""
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _make_logistic_regression(
    *,
    max_iter: int,
    random_state: int,
    class_weight=None,
    solver: str = "lbfgs",
    multinomial: bool = False,
):
    """Create a LogisticRegression instance compatibly across sklearn versions.

    Some environments reject `multi_class=` at construction time. We only request
    multinomial behaviour for FTR, and gracefully fall back to the default
    constructor signature if `multi_class` is unsupported.
    """
    kwargs = {
        "max_iter": int(max_iter),
        "random_state": int(random_state),
        "solver": solver,
    }
    if class_weight is not None:
        kwargs["class_weight"] = class_weight

    if multinomial:
        try:
            return LogisticRegression(multi_class="multinomial", **kwargs)
        except TypeError:
            return LogisticRegression(**kwargs)

    return LogisticRegression(**kwargs)



class CalibratedBinaryWrapper:
    """Pickle-safe wrapper that applies a calibrator to a base binary classifier.

    We avoid sklearn's CalibratedClassifierCV pickling issues by:
      - training a base model,
      - fitting a simple calibrator on out-of-fold probabilities,
      - storing both as plain sklearn estimators.

    The wrapper exposes predict_proba/predict like a normal classifier.
    """

    def __init__(self, base_model, calibrator=None, method: str = "sigmoid"):
        self.base_model = base_model
        self.calibrator = calibrator
        self.method = str(method or "sigmoid")

    def predict_proba(self, X):
        p = self.base_model.predict_proba(X)
        # Expect binary proba shape (n,2)
        if p is None or getattr(p, "shape", (0, 0))[1] != 2 or self.calibrator is None:
            return p

        p1 = np.asarray(p[:, 1], dtype=float)
        # Calibrator expects a 2D feature matrix
        if self.method == "isotonic":
            p1c = np.asarray(self.calibrator.predict(p1), dtype=float)
        else:
            z = _safe_logit(p1).reshape(-1, 1)
            p1c = np.asarray(self.calibrator.predict_proba(z)[:, 1], dtype=float)

        p1c = np.clip(p1c, 0.0, 1.0)
        return np.column_stack([1.0 - p1c, p1c])

    def predict(self, X):
        p = self.predict_proba(X)
        if p is None:
            return self.base_model.predict(X)
        return (np.asarray(p[:, 1], dtype=float) >= 0.5).astype(int)

# --- Pickle/JOBLIB compatibility (important) ---
# When `train_markets.py` is executed as a script, `__name__` is `__main__`, and
# joblib may pickle custom classes under `__main__.*`. That breaks later loads
# when models are loaded from another entrypoint (e.g. a python -c / <stdin> run).
# We make the class resolvable under a stable module name and also alias the
# current module as `train_markets` when run as a script.
import sys as _sys
_sys.modules.setdefault("train_markets", _sys.modules[__name__])
try:
    CalibratedBinaryWrapper.__module__ = "train_markets"
except Exception:
    pass

# ------------------------------
# Model root & completed-status filter
# ------------------------------
try:
    from constants import MODEL_DIR as _MODEL_DIR
except Exception:
    _MODEL_DIR = "ModelStore"

# Single source of truth for where models live
MODEL_ROOT = _MODEL_DIR

# Common tokens seen for completed fixtures in CSVs
COMPLETED_STATUS_PATTERNS = (
    r"\bft\b", r"full\s*time", r"finished", r"final",
    r"match\s*finished", r"aet", r"after\s*extra\s*time",
    r"pens?", r"penalt(?:y|ies)", r"awarded", r"ended"
)
# Tokens that indicate the game is NOT a completed result
INCOMPLETE_STATUS_PATTERNS = (
    r"postp|postponed|abandon|suspend|void|cancel|walkover|WO|NS|not\s*started|live|in\s*play"
)

def _future_fixture_mask(df: pd.DataFrame, *, grace_hours: float = 0.0) -> pd.Series:
    """Return True for rows that look like future-dated fixtures.

    This guards against datasets that include upcoming fixtures with placeholder
    goal counts (e.g., 0-0) which would otherwise appear "complete".

    We parse any of these columns if present (first non-null wins):
      - match_date
      - date_GMT
      - date
      - timestamp

    Parsing is done with `utc=True` to avoid tz-naive vs tz-aware comparison bugs.

    Parameters
    ----------
    grace_hours : float
        Optional grace window to tolerate small clock/ingestion offsets.
        Rows are considered future only if dt > now_utc + grace_hours.
    """
    idx = df.index

    # Build a best-effort UTC datetime series from multiple possible columns.
    dt = pd.Series(pd.NaT, index=idx, dtype="datetime64[ns, UTC]")
    for col in ("match_date", "date_GMT", "date", "timestamp"):
        if col not in df.columns:
            continue
        try:
            cand = pd.to_datetime(df[col], errors="coerce", utc=True)
        except Exception:
            # If a column is a weird mixed type, coerce via string first
            try:
                cand = pd.to_datetime(df[col].astype(str), errors="coerce", utc=True)
            except Exception:
                continue
        dt = dt.fillna(cand)

    # If we couldn't parse anything, we cannot assert the row is future.
    if dt.isna().all():
        return pd.Series(False, index=idx)

    now_utc = pd.Timestamp.now(tz="UTC")
    if grace_hours and grace_hours > 0:
        now_utc = now_utc + pd.Timedelta(hours=float(grace_hours))

    # Only mark as future when dt is valid and strictly greater than now_utc.
    return dt.notna() & (dt > now_utc)

def _completed_mask(df: pd.DataFrame, *, include_future: bool = False) -> pd.Series:
    """Vectorised mask for completed fixtures.

    Uses BOTH:
      1) status text heuristics (if present), and
      2) presence of full-time goals (more robust across sources).

    A row is considered completed if either (1) or (2) is true, and then we remove
    any rows that explicitly match incomplete tokens (postponed, suspended, etc.).

    Additionally, we drop rows that are future-dated (common in multi-season files
    where upcoming fixtures exist with placeholder goal counts like 0-0).

    When `include_future=True`, returns the completion logic before applying the future-date guard.
    """
    idx = df.index

    # (2) Label-based completeness (presence of both FT goal counts)
    ht = df.get("home_team_goal_count")
    at = df.get("away_team_goal_count")
    if isinstance(ht, pd.Series):
        ht = pd.to_numeric(ht, errors="coerce")
    else:
        ht = pd.Series(np.nan, index=idx)
    if isinstance(at, pd.Series):
        at = pd.to_numeric(at, errors="coerce")
    else:
        at = pd.Series(np.nan, index=idx)
    label_complete = ~(ht.isna() | at.isna())

    # (1) Status-based completeness
    status_complete = pd.Series(False, index=idx)
    if "status" in df.columns:
        s = df["status"].astype(str).str.lower().str.strip()
        for pat in COMPLETED_STATUS_PATTERNS:
            status_complete = status_complete | s.str.contains(pat, regex=True)
        # Explicitly mark known incomplete rows
        incomplete = s.str.contains(INCOMPLETE_STATUS_PATTERNS, regex=True)
        status_complete = status_complete & ~incomplete

    completed = (status_complete | label_complete)
    # If requested, return completion without the future-date guard.
    # Useful for diagnostics so we can report how many rows were excluded
    # specifically because they were future-dated.
    if include_future:
        return completed

    # Future-dated guard (prevents upcoming 0-0 placeholders from entering training)
    future = _future_fixture_mask(df)

    return completed & ~future


def _filter_completed_training_rows(df: pd.DataFrame) -> pd.DataFrame:
    # Baseline completion (status/label heuristics), before applying the future-date guard
    completed_baseline = _completed_mask(df, include_future=True)

    # Future mask is computed separately (UTC-aware parsing) to avoid tz issues
    future_mask = _future_fixture_mask(df)

    # Final training completion mask: completed AND not future-dated
    completed_mask = completed_baseline & ~future_mask

    out = df.loc[completed_mask].copy()

    try:
        dropped_total = int(len(df) - len(out))
        # Count only rows that *would have been kept* if not for being future-dated
        dropped_future = int((future_mask & completed_baseline).sum())
        # Note: dropped_total includes future-dropped rows plus any other non-completed rows.
        print(
            f"🧹 Train filter: completed fixtures {len(out)}/{len(df)} "
            f"(dropped_total={dropped_total}, future_dropped={dropped_future})."
        )
    except Exception:
        pass

    return out


# ---------------------------------------------------------------
# Trained model auto-loader and scorers (for train_markets.py pkls)
# ---------------------------------------------------------------

def _market_model_path(league: str, market: str, model_root: str | None = None) -> str:
    root = model_root or MODEL_ROOT
    return os.path.join(root, league.replace(" ", "_"), f"{market}.pkl")


def _load_market_model(league: str, market: str, model_root: str | None = None) -> Optional[dict]:
    """Return {model, features, ...} bundle if a saved model exists, else None."""
    root = model_root or MODEL_ROOT
    path = _market_model_path(league, market, root)
    if os.path.exists(path):
        try:
            # Compatibility: older pickles may reference __main__.CalibratedBinaryWrapper.
            # Ensure it exists in the current __main__ module before loading.
            try:
                import __main__ as _main  # type: ignore
                if not hasattr(_main, "CalibratedBinaryWrapper"):
                    _main.CalibratedBinaryWrapper = CalibratedBinaryWrapper
            except Exception:
                pass

            return joblib.load(path)
        except Exception as e:
            print(f"⚠️ Could not load model for {market} @ {path}: {e}")
    return None


def _ensure_feature_frame(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Select and align feature columns used at training time.
    Missing columns are created with zeros, then all cast to float.
    """
    X = pd.DataFrame(index=df.index)
    for c in features:
        if c in df.columns:
            X[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            X[c] = 0.0
    try:
        missing = [c for c in features if c not in df.columns]
        if missing:
            print(f"⚠️ Feature alignment: {len(missing)} missing columns filled with 0.0 (e.g. {missing[:5]})")
    except Exception:
        pass
    return X.astype(float)


def _score_binary_market(df: pd.DataFrame, league: str, market: str, model_root: str | None = None) -> pd.DataFrame:
    root = model_root or MODEL_ROOT
    """Load a binary classifier and attach `<market>_confidence` + `<market>_pred`.
    Special cases:
      • home_fts → write `p_home_fts` and `home_fts_pred`
      • away_fts → write `p_away_fts` and `away_fts_pred`
    """
    bundle = _load_market_model(league, market, root)
    if not bundle:
        return df
    clf = bundle.get("model")
    feats = bundle.get("feature_cols") or bundle.get("features", [])
    if clf is None or not feats:
        return df
    X = _ensure_feature_frame(df, feats)
    try:
        proba = clf.predict_proba(X)[:, 1]
    except Exception as e:
        print(f"⚠️ Inference failed for {market}: {e}")
        return df

    # Column naming
    if market == "home_fts":
        conf_col, pred_col = "p_home_fts", "home_fts_pred"
    elif market == "away_fts":
        conf_col, pred_col = "p_away_fts", "away_fts_pred"
    else:
        conf_col, pred_col = f"{market}_confidence", f"{market}_pred"

    df[conf_col] = pd.to_numeric(proba, errors="coerce").clip(0, 1)

    # Prefer per-model threshold saved at training time; fallback to attrs/default.
    thr = None
    try:
        thr = bundle.get("threshold", None)
    except Exception:
        thr = None
    if thr is None:
        thr = df.attrs.get(f"thr_{market}", 0.5)
    try:
        thr = float(thr)
    except Exception:
        thr = 0.5
    thr = float(min(0.99, max(0.01, thr)))

    # Stash threshold so downstream layers can read it consistently.
    df.attrs[f"thr_{market}"] = thr

    df[pred_col] = (df[conf_col] >= thr).astype(int)
    df.attrs.setdefault("used_trained_models", []).append(market)
    return df


def _score_multiclass_ftr(df: pd.DataFrame, league: str, model_root: str | None = None) -> pd.DataFrame:
    root = model_root or MODEL_ROOT
    """Load multinomial FTR model and attach confidence_home/draw/away and ftr_pred_outcome (0/1/2)."""
    bundle = _load_market_model(league, "ftr", root)
    if not bundle:
        return df
    clf = bundle.get("model")
    feats = bundle.get("feature_cols") or bundle.get("features", [])
    if clf is None or not feats:
        return df
    X = _ensure_feature_frame(df, feats)
    try:
        proba = clf.predict_proba(X)
        pred = clf.predict(X)
    except Exception as e:
        print(f"⚠️ Inference failed for FTR: {e}")
        return df

    if proba.shape[1] == 3:
        df["confidence_home"] = pd.to_numeric(proba[:, 0], errors="coerce").clip(0, 1)
        df["confidence_draw"] = pd.to_numeric(proba[:, 1], errors="coerce").clip(0, 1)
        df["confidence_away"] = pd.to_numeric(proba[:, 2], errors="coerce").clip(0, 1)
    else:
        # Unexpected class order/size – best effort: fill NaNs
        df["confidence_home"] = np.nan
        df["confidence_draw"] = np.nan
        df["confidence_away"] = np.nan

    df["ftr_pred_outcome"] = pd.to_numeric(pred, errors="coerce").fillna(1).astype(int)
    df.attrs.setdefault("used_trained_models", []).append("ftr")
    return df


def score_trained_markets(df: pd.DataFrame, league: str, markets: 'Optional[List[str]]' = None, model_root: str | None = None) -> pd.DataFrame:
    root = model_root or MODEL_ROOT
    """Attempt to load and score any available trained models for the requested markets.
    Safe no-op if models are missing.
    """
    # Normalize column names/types so inference sees the same feature names as training
    try:
        df = apply_safe_renames_and_whitelist(df.copy())
    except Exception:
        df = df.copy()
    try:
        df = _coerce_numeric_like(df)
    except Exception:
        pass
    mkts = list(markets) if markets else [
        "btts", "over25", "btts_fh",
        "home_ge2", "away_ge2", "home_ge3", "away_ge3",
        "home_fts", "away_fts",
    ]
    for m in mkts:
        if m == "ftr":
            df = _score_multiclass_ftr(df, league, model_root=root)
        else:
            df = _score_binary_market(df, league, m, model_root=root)
    return df


# ------------------------------
# Config (one source of truth)
# ------------------------------
DEFAULT_CFG = {
    "random_state": 42,
    "cv_folds": 5,
    "test_size": 0.15,  # used if no explicit season split available
    "calibration": "sigmoid",  # for binary models via CalibratedClassifierCV
    # markets we'll train
    "markets": [
        "ftr",            # multinomial (Home/Draw/Away)
        "over25",         # binary
        "btts",           # binary
        "btts_fh",        # binary (first-half BTTS)
        "home_ge2",       # binary (home team scores >=2)
        "away_ge2",       # binary
        "home_ge3",
        "away_ge3",
        "home_fts",       # binary (home fail to score)
        "away_fts",       # binary
    ],
}

# Where models will be saved/loaded from → MODEL_ROOT (derived from constants.MODEL_DIR or 'ModelStore')


# ------------------------------
# Feature building
# ------------------------------
PREMATCH_FEATURES_CANDIDATES = [
    # ------------------------------
    # Core pre-match strength / quality
    # ------------------------------
    "pre_match_ppg_home", "pre_match_ppg_away",
    "ppg_home_pre", "ppg_away_pre",
    "pre_match_xg_home", "pre_match_xg_away",
    "xg_home", "xg_away",
    "xg_net_diff", "xg_diff_abs", "xg_ratio_diff",
    "gfpm_diff", "gapm_diff",
    "average_goals_per_match_pre_match",
    "btts_percentage_pre_match",
    "over_25_percentage_pre_match", "over_15_percentage_pre_match",
    "avg_total_goals_diff", "over25_rate_diff", "btts_rate_diff",
    "clean_sheet_rate_diff",

    # ------------------------------
    # Attack / defence specialist context
    # ------------------------------
    "home_attack_score", "away_attack_score",
    "selected_attack_score", "opponent_attack_score",
    "home_defence_score", "away_defence_score",
    "selected_defence_score", "opponent_defence_score",
    "home_xg_against_idx", "away_xg_against_idx",
    "defence_diff",

    # ------------------------------
    # Shot volume / shot quality context
    # ------------------------------
    "shot_volume_diff", "sot_quality_diff",
    "rolling5_home_sot_ratio", "rolling5_away_sot_ratio",

    # ------------------------------
    # Press / pressure / volatility (prematch-safe proxies)
    # ------------------------------
    # Prefer the explicit prematch proxy columns produced by etl_press_intensity.py
    "pre_match_press_intensity_home", "pre_match_press_intensity_away",
    # Keep baselines for backward compatibility / older merged files
    "home_press_baseline", "away_press_baseline",
    # Optional rolling prematch-safe press context
    "rolling5_home_press_intensity", "rolling5_away_press_intensity",
    "rolling5_press_intensity_diff",
    "rolling5_home_press_z", "rolling5_away_press_z",
    "rolling5_press_z_diff",
    "rolling10_press_intensity_diff",
    "press_volatility_score",

    # ------------------------------
    # Strength / control / result-context
    # ------------------------------
    "ppg_diff",
    "power_diff",
    "home_power_rating", "away_power_rating",
    "elo_diff",
    "home_advantage_diff",
    "selected_ppg_pre", "opponent_ppg_pre",
    "selected_xg_pre", "opponent_xg_pre",
    "selected_power_edge",
    "prob_ftr_home", "prob_ftr_away",
    "selected_win_prob", "opponent_win_prob",

    # ------------------------------
    # Lambda / Poisson goal environment
    # ------------------------------
    "lambda_home", "lambda_away",
    "selected_lambda", "opponent_lambda",
    "exp_goals_sum",
    "p_home_pois", "p_away_pois",

    # ------------------------------
    # Market odds / bookmaker priors
    # ------------------------------
    "odds_ft_home_team_win", "odds_ft_draw", "odds_ft_away_team_win",
    "odds_btts_yes", "odds_btts_no",
    "odds_ft_over15", "odds_ft_over25", "odds_ft_over35", "odds_ft_over45",
    # Under 2.5 aliases after renames / synth promotion
    "odds_ft_under25", "odds_under25", "odds_under_2_5", "odds_ft_u25",

    # ------------------------------
    # Existing prematch-safe model context
    # ------------------------------
    "home_ge2_confidence", "away_ge2_confidence",
    "home_ge3_confidence", "away_ge3_confidence",
    "p_home_fts", "p_away_fts",
]

TARGET_COLS_NEEDED = [
    # We derive targets from these if present
    "home_team_goal_count", "away_team_goal_count",
    "home_team_goal_count_half_time", "away_team_goal_count_half_time",
]


def _select_feature_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []

    # 1) Start from the explicit allow-list above.
    for c in PREMATCH_FEATURES_CANDIDATES:
        if c in df.columns and df[c].dtype.kind in "if":
            cols.append(c)

    # 2) Pull in other prematch-safe numeric columns that follow known naming shapes.
    dynamic_suffixes = (
        "_pre_match",
        "_percentage_pre_match",
        "_diff",
        "_score",
        "_rate",
        "_ratio",
        "_idx",
        "_edge",
        "_prob",
        "_confidence",
        "_lambda",
        "_pois",
    )
    dynamic_exact = {
        "lambda_home",
        "lambda_away",
        "exp_goals_sum",
        "elo_diff",
        "power_diff",
        "gfpm_diff",
        "gapm_diff",
        "defence_diff",
        "shot_volume_diff",
        "sot_quality_diff",
        "clean_sheet_rate_diff",
        "avg_total_goals_diff",
        "over25_rate_diff",
        "btts_rate_diff",
        "press_volatility_score",
        "prob_ftr_home",
        "prob_ftr_away",
        "selected_win_prob",
        "opponent_win_prob",
        "p_home_pois",
        "p_away_pois",
        "p_home_fts",
        "p_away_fts",
    }
    odds_prefixes = ("odds_",)

    for c in df.columns:
        if c in cols:
            continue
        if df[c].dtype.kind not in "if":
            continue
        if c in dynamic_exact:
            cols.append(c)
            continue
        if c.startswith(odds_prefixes):
            cols.append(c)
            continue
        if c.endswith(dynamic_suffixes):
            cols.append(c)
            continue

    # 3) Final hard leak filter.
    leak_like = {
        "total_goal_count",
        "home_team_shots", "away_team_shots",
        "home_team_shots_on_target", "away_team_shots_on_target",
        "home_team_shots_off_target", "away_team_shots_off_target",
        "home_team_goal_count", "away_team_goal_count",
        "home_team_goal_count_half_time", "away_team_goal_count_half_time",
        "team_a_xg", "team_b_xg",
        "home_goals", "away_goals",
        "ft_home_goals", "ft_away_goals",
        "result", "full_time_result",
        "won", "draw", "lost",
        "target",
    }
    cols = [c for c in cols if c not in leak_like]
    return sorted(list(dict.fromkeys(cols)))


def build_features(raw_df: pd.DataFrame, league: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Normalise headers & types and build the feature matrix X for training.

    IMPORTANT: we preserve label columns (goal counts) for target derivation,
    and drop leaky columns only from the *feature* frame.
    Returns:
      X_df (features after leak-drop) and df_labels (original frame for targets).
    """
    df0 = raw_df.copy()
    df0 = apply_safe_renames_and_whitelist(df0)
    df0 = _coerce_numeric_like(df0)

    # Keep a copy with labels intact for target derivation
    df_labels = df0.copy()

    # Build the feature frame with leaky columns removed
    df_feat = _force_drop_leaky_cols(df0)

    # ensure key feature columns exist (fill NAs to median) on feature frame
    for c in PREMATCH_FEATURES_CANDIDATES:
        if c in df_feat.columns:
            df_feat[c] = pd.to_numeric(df_feat[c], errors="coerce")

    # Press features: treat exact 0.0 as missing (older/bad seasons can be all-zero)
    press_cols = [
        "pre_match_press_intensity_home",
        "pre_match_press_intensity_away",
        "home_press_baseline",
        "away_press_baseline",
        "rolling5_home_press_intensity",
        "rolling5_away_press_intensity",
        "rolling5_press_intensity_diff",
        "rolling5_home_press_z",
        "rolling5_away_press_z",
        "rolling5_press_z_diff",
        "rolling10_press_intensity_diff",
        "press_volatility_score",
    ]
    for c in press_cols:
        if c in df_feat.columns:
            df_feat[c] = pd.to_numeric(df_feat[c], errors="coerce")
            # Treat exact zeros as missing so we impute from valid distribution
            df_feat.loc[df_feat[c].fillna(0.0).abs() < 1e-12, c] = np.nan
            # Missingness flag can be useful to the model
            miss_flag = f"{c}_missing"
            if miss_flag not in df_feat.columns:
                df_feat[miss_flag] = df_feat[c].isna().astype(int)

    # Lambda / Poisson features are specialist-critical; exact zeros usually mean
    # missing upstream build rather than a genuine prematch expectation.
    lambda_cols = [
        "lambda_home",
        "lambda_away",
        "selected_lambda",
        "opponent_lambda",
        "exp_goals_sum",
        "p_home_pois",
        "p_away_pois",
    ]
    for c in lambda_cols:
        if c in df_feat.columns:
            df_feat[c] = pd.to_numeric(df_feat[c], errors="coerce")
            if c != "exp_goals_sum":
                df_feat.loc[df_feat[c].fillna(0.0).abs() < 1e-12, c] = np.nan
            miss_flag = f"{c}_missing"
            if miss_flag not in df_feat.columns:
                df_feat[miss_flag] = df_feat[c].isna().astype(int)

    num_cols = [c for c in df_feat.columns if df_feat[c].dtype.kind in "if"]
    if num_cols:
        df_feat[num_cols] = df_feat[num_cols].fillna(df_feat[num_cols].median())

    X_cols = _select_feature_columns(df_feat)
    if X_cols:
        X = df_feat[X_cols].copy()
    else:
        X = pd.DataFrame(index=df_feat.index)
    # Ensure stable float dtype
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    if len(X.columns):
        X = X.fillna(0.0).astype(float)
    return X, df_labels


def _extract_feature_names_from_estimator(clf, fallback: List[str]) -> List[str]:
    """Best-effort extraction of feature names used by the fitted model.

    We prefer sklearn's `feature_names_in_` when available (ensures stable mapping
    even if columns get re-ordered elsewhere). Fall back to the provided list.

    Supports:
      - Pipeline
      - CalibratedClassifierCV (uses base estimator)
    """
    try:
        est = clf
        # Unwrap our pickle-safe calibration wrapper if present
        if hasattr(est, "base_model") and est.base_model is not None:
            est = est.base_model
        # CalibratedClassifierCV wraps a base estimator
        if hasattr(est, "estimator") and est.estimator is not None:
            est = est.estimator
        elif hasattr(est, "base_estimator") and est.base_estimator is not None:
            est = est.base_estimator

        # Pipeline may carry feature_names_in_ itself, or on the final step
        if hasattr(est, "feature_names_in_"):
            return [str(x) for x in list(est.feature_names_in_)]
        if hasattr(est, "named_steps"):
            last = None
            try:
                # prefer final estimator step
                last = list(est.named_steps.values())[-1]
            except Exception:
                last = None
            if last is not None and hasattr(last, "feature_names_in_"):
                return [str(x) for x in list(last.feature_names_in_)]
    except Exception:
        pass
    return [str(x) for x in list(fallback)]


# ------------------------------
# Best F1 threshold helper
# ------------------------------
def _threshold_bounds_for_market(market: str, pos_rate: float) -> Tuple[float, float]:
    """Reasonable guard-rails for prediction rate vs true positive rate.

    We do NOT want threshold selection to explode predicted positive rate far above
    the empirical holdout base rate, which is exactly what plain max-F1 can do on
    skewed football side-markets.

    Returns:
        (min_pred_pos_rate, max_pred_pos_rate)
    """
    m = str(market or "").strip().lower()
    pr = float(np.clip(pos_rate, 0.0, 1.0))

    # Sensible default band.
    lo_mult = 0.60
    hi_mult = 1.40
    lo_floor = 0.02
    hi_floor = 0.08

    # Rare-event markets need tighter upper control.
    if m in {"home_ge3", "away_ge3", "home_fts", "away_fts", "btts_fh"}:
        lo_mult = 0.50
        hi_mult = 1.20
        lo_floor = 0.01
        hi_floor = 0.05
    elif m in {"home_ge2", "away_ge2"}:
        lo_mult = 0.65
        hi_mult = 1.30
        lo_floor = 0.03
        hi_floor = 0.10
    elif m in {"btts", "over25"}:
        lo_mult = 0.75
        hi_mult = 1.25
        lo_floor = 0.05
        hi_floor = 0.15

    lo = max(lo_floor, pr * lo_mult)
    hi = min(0.98, max(hi_floor, pr * hi_mult))
    if lo > hi:
        lo = min(lo, hi)
    return float(lo), float(hi)



def _best_f1_threshold(
    y_true: pd.Series,
    proba: np.ndarray,
    market: str = "",
) -> float | None:
    """Pick a safer decision threshold than raw max-F1.

    Plain max-F1 often drives thresholds far too low on skewed specialist markets,
    which creates absurd positive prediction rates. Here we:
      1) score all PR thresholds,
      2) keep only candidates whose predicted positive rate sits inside a
         market-aware band around the true holdout positive rate,
      3) choose the candidate with the best F1 among the valid set,
      4) fall back to global max-F1 only if nothing survives.
    """
    try:
        y = pd.to_numeric(y_true, errors="coerce").fillna(0).astype(int).values
        p = pd.to_numeric(pd.Series(proba), errors="coerce").fillna(0.0).clip(0.0, 1.0).values
        if len(y) == 0:
            return None
        if int(np.unique(y).size) < 2:
            return None

        prec, rec, thr = precision_recall_curve(y, p)
        if thr is None or len(thr) == 0:
            return None

        f1 = (2.0 * prec[1:] * rec[1:]) / (prec[1:] + rec[1:] + 1e-12)
        pos_rate = float(np.mean(y))
        min_pred_rate, max_pred_rate = _threshold_bounds_for_market(market, pos_rate)

        pred_rates = np.array([(p >= float(t)).mean() for t in thr], dtype=float)
        valid = (pred_rates >= min_pred_rate) & (pred_rates <= max_pred_rate)

        if valid.any():
            candidate_idx = np.where(valid)[0]
            best_local = candidate_idx[int(np.nanargmax(f1[candidate_idx]))]
            best_thr = float(thr[best_local])
        else:
            best_i = int(np.nanargmax(f1))
            best_thr = float(thr[best_i])

        return float(min(0.99, max(0.01, best_thr)))
    except Exception:
        return None


# ------------------------------
# Targets per market
# ------------------------------

def _derive_targets(df: pd.DataFrame) -> Dict[str, pd.Series]:
    idx = df.index

    def _to_series(col):
        if isinstance(col, pd.Series):
            return pd.to_numeric(col, errors="coerce")
        # Return an all-NaN Series aligned to df if column missing or scalar
        return pd.Series(np.nan, index=idx)

    ht = _to_series(df.get("home_team_goal_count"))
    at = _to_series(df.get("away_team_goal_count"))
    htht = _to_series(df.get("home_team_goal_count_half_time"))
    at_ht = _to_series(df.get("away_team_goal_count_half_time"))

    targets: Dict[str, pd.Series] = {}

    have_ft = not (ht.isna().all() or at.isna().all())
    have_ht = not (htht.isna().all() or at_ht.isna().all())

    if have_ft:
        # FTR as multiclass: 0=Home, 1=Draw, 2=Away
        ftr = np.where(ht > at, 0, np.where(ht == at, 1, 2))
        targets["ftr"] = pd.Series(ftr, index=idx)

        # Over 2.5 and BTTS
        targets["over25"] = ((ht + at) >= 3).astype(int)
        targets["btts"] = ((ht > 0) & (at > 0)).astype(int)

        # Home/Away >=2 and FTS
        targets["home_ge2"] = (ht >= 2).astype(int)
        targets["home_ge3"] = (ht >= 3).astype(int)
        targets["home_fts"] = (ht == 0).astype(int)

        targets["away_ge2"] = (at >= 2).astype(int)
        targets["away_ge3"] = (at >= 3).astype(int)
        targets["away_fts"] = (at == 0).astype(int)

    if have_ht:
        targets["btts_fh"] = ((htht > 0) & (at_ht > 0)).astype(int)

    return targets


# ------------------------------
# Training per market
# ------------------------------

def _train_binary(X: pd.DataFrame, y: pd.Series, cfg: dict, market: str = ""):
    """Train a binary classifier.

    Important:
    - For rare targets (GE3/FTS) we use class_weight='balanced'.
    - For calibration, we avoid CalibratedClassifierCV and instead fit a simple
      Platt (sigmoid) or isotonic calibrator on out-of-fold probabilities.
      This is far more pickle-safe in mixed environments.
    """

    mkt = str(market or "").strip().lower()
    # Rare / skewed targets where calibration matters most
    rare_markets = {"home_ge3", "away_ge3", "home_fts", "away_fts", "btts_fh"}
    cw = "balanced" if mkt in rare_markets else None

    base = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "logit",
            _make_logistic_regression(
                max_iter=200,
                random_state=cfg["random_state"],
                class_weight=cw,
                solver="lbfgs",
                multinomial=False,
            ),
        ),
    ])

    cal = str(cfg.get("calibration", "")).strip().lower()
    if cal in ("none", "off", "false", "0", ""):
        base.fit(X, y)
        return base

    # Build out-of-fold probs for robust calibration
    y_int = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
    if int(np.unique(y_int).size) < 2:
        # Single-class edge case: no calibration possible
        base.fit(X, y_int)
        return base

    skf = StratifiedKFold(n_splits=int(cfg.get("cv_folds", 5)), shuffle=True, random_state=cfg["random_state"])
    oof_p = np.zeros(len(y_int), dtype=float)

    for tr, te in skf.split(X, y_int):
        b = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "logit",
                _make_logistic_regression(
                    max_iter=200,
                    random_state=cfg["random_state"],
                    class_weight=cw,
                    solver="lbfgs",
                    multinomial=False,
                ),
            ),
        ])
        b.fit(X.iloc[tr], y_int.iloc[tr])
        oof_p[te] = b.predict_proba(X.iloc[te])[:, 1]

    # Fit calibrator
    if cal == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(oof_p, y_int.values)
        # Fit final base on all data
        base.fit(X, y_int)
        return CalibratedBinaryWrapper(base_model=base, calibrator=calibrator, method="isotonic")

    # Default: sigmoid / platt scaling
    z = _safe_logit(oof_p).reshape(-1, 1)
    calibrator = _make_logistic_regression(
        max_iter=200,
        random_state=cfg["random_state"],
        solver="lbfgs",
        multinomial=False,
    )
    calibrator.fit(z, y_int.values)

    # Fit final base on all data
    base.fit(X, y_int)
    return CalibratedBinaryWrapper(base_model=base, calibrator=calibrator, method="sigmoid")


def _train_multiclass(X: pd.DataFrame, y: pd.Series, cfg: dict):
    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "logit",
            _make_logistic_regression(
                max_iter=400,
                random_state=cfg["random_state"],
                solver="lbfgs",
                multinomial=True,
            ),
        ),
    ])
    clf.fit(X, y)
    return clf


def _cv_metrics_binary(X: pd.DataFrame, y: pd.Series, cfg: dict, market: str = "") -> Dict[str, float]:
    skf = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=cfg["random_state"])
    ap_scores, roc_scores = [], []
    for tr, te in skf.split(X, y):
        clf = _train_binary(X.iloc[tr], y.iloc[tr], cfg, market=market)
        proba = clf.predict_proba(X.iloc[te])[:, 1]
        ap_scores.append(average_precision_score(y.iloc[te], proba))
        # ROC can fail if only one class in fold; guard
        try:
            roc_scores.append(roc_auc_score(y.iloc[te], proba))
        except Exception:
            pass
    return {
        "ap_mean": float(np.mean(ap_scores)) if ap_scores else np.nan,
        "roc_mean": float(np.mean(roc_scores)) if roc_scores else np.nan,
    }


def _cv_metrics_multiclass(X: pd.DataFrame, y: pd.Series, cfg: dict) -> Dict[str, float]:
    skf = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=cfg["random_state"])
    acc_scores = []
    for tr, te in skf.split(X, y):
        clf = _train_multiclass(X.iloc[tr], y.iloc[tr], cfg)
        pred = clf.predict(X.iloc[te])
        acc_scores.append(accuracy_score(y.iloc[te], pred))
    return {"acc_mean": float(np.mean(acc_scores)) if acc_scores else np.nan}


def _holdout_split(df: pd.DataFrame, cfg: dict) -> pd.Series:
    # try season-based holdout if a season column exists
    if "season" in df.columns:
        seasons = sorted([s for s in df["season"].dropna().unique()])
        if len(seasons) >= 2:
            last = seasons[-1]
            return (df["season"] != last).astype(bool)
    # else do a time split if a date exists
    for dc in ("date_GMT", "match_date", "timestamp"):
        if dc in df.columns:
            dt = pd.to_datetime(df[dc], errors="coerce")
            if dt.notna().any():
                order = dt.rank(method="first")
                cutoff = order.quantile(1.0 - cfg["test_size"])
                return (order <= cutoff).astype(bool)
    # fallback random mask
    rng = np.random.RandomState(cfg["random_state"])
    return pd.Series(rng.rand(len(df)) <= (1.0 - cfg["test_size"]), index=df.index).astype(bool)


def train_all_markets_for_league(df_raw: pd.DataFrame, league: str, cfg: dict | None = None) -> pd.DataFrame:
    cfg = {**DEFAULT_CFG, **(cfg or {})}

    # Enforce completed-only rows for training
    df_source = _filter_completed_training_rows(df_raw.copy())

    X, df = build_features(df_source, league)
    targets = _derive_targets(df)

    # build mask for train rows where targets are not NA
    metrics_rows: List[Dict[str, object]] = []

    # train/val split indices (single split for holdout reporting)
    train_mask = _holdout_split(df, cfg)
    if not isinstance(train_mask, pd.Series):
        train_mask = pd.Series(train_mask, index=df.index)
    train_mask = train_mask.astype(bool)

    os.makedirs(MODEL_ROOT, exist_ok=True)
    league_dir = os.path.join(MODEL_ROOT, league.replace(" ", "_"))
    os.makedirs(league_dir, exist_ok=True)

    for market in cfg["markets"]:
        if market not in targets:
            metrics_rows.append({
                "league": league,
                "market": market,
                "status": "skipped (no target)",
            })
            continue
        y = targets[market]
        valid = ~y.isna()
        if valid.sum() < max(100, 10 * cfg["cv_folds"]):
            metrics_rows.append({
                "league": league,
                "market": market,
                "status": f"skipped (too few rows: {int(valid.sum())})",
            })
            continue

        Xv = X.loc[valid]
        yv = y.loc[valid]

        # choose binary vs multiclass
        is_multiclass = (market == "ftr")
        if is_multiclass:
            cvm = _cv_metrics_multiclass(Xv, yv, cfg)
            clf = _train_multiclass(Xv.loc[train_mask[valid]], yv.loc[train_mask[valid]], cfg)
            y_pred = clf.predict(Xv.loc[~train_mask[valid]])
            hold_acc = accuracy_score(yv.loc[~train_mask[valid]], y_pred)
            metrics = {
                "cv_acc": cvm.get("acc_mean"),
                "hold_acc": float(hold_acc),
            }
            # For multiclass, no thresholding/diagnostics
            best_thr = None
            hold_acc_diag = np.nan
            hold_pos_rate = np.nan
            pred_pos_rate = np.nan
        else:
            cvm = _cv_metrics_binary(Xv, yv, cfg, market=market)
            clf = _train_binary(Xv.loc[train_mask[valid]], yv.loc[train_mask[valid]], cfg, market=market)
            proba = clf.predict_proba(Xv.loc[~train_mask[valid]])[:, 1]
            hold_ap = average_precision_score(yv.loc[~train_mask[valid]], proba)
            try:
                hold_roc = roc_auc_score(yv.loc[~train_mask[valid]], proba)
            except Exception:
                hold_roc = np.nan

            # Compute best F1 threshold and holdout diagnostics
            y_hold = yv.loc[~train_mask[valid]]
            best_thr = _best_f1_threshold(y_hold, proba, market=market)
            if best_thr is not None:
                pred_hold = (pd.to_numeric(pd.Series(proba), errors="coerce").fillna(0.0) >= float(best_thr)).astype(int).values
                try:
                    hold_acc_diag = float(accuracy_score(y_hold, pred_hold))
                except Exception:
                    hold_acc_diag = np.nan
                try:
                    hold_pos_rate = float(pd.to_numeric(y_hold, errors="coerce").mean())
                except Exception:
                    hold_pos_rate = np.nan
                try:
                    pred_pos_rate = float(np.mean(pred_hold))
                except Exception:
                    pred_pos_rate = np.nan
            else:
                pred_hold = None
                hold_acc_diag = np.nan
                hold_pos_rate = np.nan
                pred_pos_rate = np.nan

            metrics = {
                "cv_ap": cvm.get("ap_mean"),
                "cv_roc": cvm.get("roc_mean"),
                "hold_ap": float(hold_ap),
                "hold_roc": float(hold_roc) if not np.isnan(hold_roc) else np.nan,
                "val_accuracy": float(hold_acc_diag) if not np.isnan(hold_acc_diag) else np.nan,
                "val_pos_rate": float(hold_pos_rate) if not np.isnan(hold_pos_rate) else np.nan,
                "pred_pos_rate": float(pred_pos_rate) if not np.isnan(pred_pos_rate) else np.nan,
                "pred_pos_rate_ratio": (
                    float(pred_pos_rate / hold_pos_rate)
                    if (not np.isnan(pred_pos_rate) and not np.isnan(hold_pos_rate) and float(hold_pos_rate) > 0)
                    else np.nan
                ),
                "threshold": float(best_thr) if best_thr is not None else None,
                "thr_method": "f1_bounded" if best_thr is not None else None,
                "n_train": int((train_mask[valid]).sum()),
                "n_val": int((~train_mask[valid]).sum()),
            }

        # save model
        model_path = os.path.join(league_dir, f"{market}.pkl")
        model_path_v2 = os.path.join(league_dir, f"{market}_v2.pkl")

        # Persist features used at training time (critical for explainability + consistency)
        train_feature_cols = [str(c) for c in list(Xv.columns)]
        saved_feature_cols = _extract_feature_names_from_estimator(clf, train_feature_cols)
        feature_hash = str(pd.util.hash_pandas_object(pd.Index(saved_feature_cols), index=False).sum())
        train_feature_hash = str(pd.util.hash_pandas_object(pd.Index(train_feature_cols), index=False).sum())

        bundle = {
            "model": clf,
            # Feature columns used at training time
            "features": train_feature_cols,          # legacy key used by existing inference code
            "feature_cols": saved_feature_cols,      # preferred key (may come from sklearn)
            "n_features": int(len(train_feature_cols)),
            "feature_hash": feature_hash,
            "train_feature_hash": train_feature_hash,

            # Training / validation diagnostics (persisted for tooling + reports)
            # NOTE: these come from the `metrics` dict built above.
            "threshold": (metrics.get("threshold") if isinstance(metrics, dict) else None),
            "thr_method": (metrics.get("thr_method") if isinstance(metrics, dict) else None),
            "val_accuracy": (metrics.get("val_accuracy") if isinstance(metrics, dict) else None),
            "val_pos_rate": (metrics.get("val_pos_rate") if isinstance(metrics, dict) else None),
            "pred_pos_rate": (metrics.get("pred_pos_rate") if isinstance(metrics, dict) else None),
            "n_train": (metrics.get("n_train") if isinstance(metrics, dict) else None),
            "n_val": (metrics.get("n_val") if isinstance(metrics, dict) else None),
            "cv_ap": (metrics.get("cv_ap") if isinstance(metrics, dict) else None),
            "cv_roc": (metrics.get("cv_roc") if isinstance(metrics, dict) else None),
            "hold_ap": (metrics.get("hold_ap") if isinstance(metrics, dict) else None),
            "hold_roc": (metrics.get("hold_roc") if isinstance(metrics, dict) else None),
            "cv_acc": (metrics.get("cv_acc") if isinstance(metrics, dict) else None),
            "hold_acc": (metrics.get("hold_acc") if isinstance(metrics, dict) else None),

            "config": cfg,
            "league": league,
            "market": market,
            "calibration": str(cfg.get("calibration", "sigmoid")),
            "calibration_wrapper": bool(isinstance(clf, CalibratedBinaryWrapper)),
        }

        try:
            joblib.dump(bundle, model_path)
            joblib.dump(bundle, model_path_v2)
            print(f"💾 Saved model → {model_path}")
            print(f"💾 Saved model → {model_path_v2}")
        except pickle.PicklingError as e:
            # Some environments can fail to pickle CalibratedClassifierCV due to class identity mismatches.
            # IMPORTANT: do NOT save `clf.estimator` / `clf.base_estimator` here, because those are template
            # estimators and may be unfitted. Instead, refit a fresh uncalibrated Pipeline on the training fold.
            print(
                f"⚠️ Pickle failed for {league} {market} (likely calibrated). "
                f"Refitting UNCALIBRATED pipeline and retrying. Error: {e}"
            )

            # Refit an uncalibrated model on the same training fold used above
            try:
                X_fit = Xv.loc[train_mask[valid]]
                y_fit = yv.loc[train_mask[valid]]
            except Exception:
                # Fallback to fitting on all valid rows if masks are unavailable
                X_fit = Xv
                y_fit = yv

            if market == "ftr":
                uncal = Pipeline([
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    (
                        "logit",
                        _make_logistic_regression(
                            max_iter=400,
                            random_state=cfg["random_state"],
                            solver="lbfgs",
                            multinomial=True,
                        ),
                    ),
                ])
            else:
                uncal = Pipeline([
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    (
                        "logit",
                        _make_logistic_regression(
                            max_iter=200,
                            random_state=cfg["random_state"],
                            solver="lbfgs",
                            multinomial=False,
                        ),
                    ),
                ])

            uncal.fit(X_fit, y_fit)

            bundle["model"] = uncal
            bundle["calibration_dropped"] = True
            bundle["calibration_error"] = str(e)

            joblib.dump(bundle, model_path)
            joblib.dump(bundle, model_path_v2)
            print(f"💾 Saved UNCALIBRATED (refit) model → {model_path}")
            print(f"💾 Saved UNCALIBRATED (refit) model → {model_path_v2}")

        metrics_rows.append({
            "league": league,
            "market": market,
            "status": "ok",
            **metrics,
            "rows": int(valid.sum()),
            "model_path": model_path,
            "model_path_v2": model_path_v2,
        })

    # Write a flat market-thresholds json for signal_layers (mid thresholds), merge-safe.
    try:
        tag = league.replace(" ", "_")
        th_path = os.path.join(MODEL_ROOT, f"{tag}_market_thresholds.json")

        # Load existing thresholds if present (do NOT wipe investor-v2 thresholds)
        th: Dict[str, float] = {}
        if os.path.exists(th_path):
            try:
                with open(th_path, "r", encoding="utf-8") as f:
                    old = json.load(f)
                if isinstance(old, dict):
                    for k, v in old.items():
                        try:
                            th[str(k)] = float(v)
                        except Exception:
                            continue
            except Exception:
                pass

        # Pull latest thresholds from the metrics rows we just computed
        # (only markets trained in this run will be updated)
        for r in metrics_rows:
            if r.get("status") != "ok":
                continue
            mkt = str(r.get("market"))
            t = r.get("threshold")
            if t is None:
                continue
            try:
                th[mkt] = float(t)
            except Exception:
                continue

        # Complements (derive if base exists in merged map)
        if "btts" in th:
            try:
                th.setdefault("btts_no", float(1.0 - float(th["btts"])))
            except Exception:
                pass
        if "over25" in th:
            try:
                th.setdefault("under25", float(1.0 - float(th["over25"])))
            except Exception:
                pass

        # Keep a sensible default for ftr mid if not present
        th.setdefault("ftr", 0.4)

        if th:
            with open(th_path, "w", encoding="utf-8") as f:
                json.dump(th, f, indent=2, sort_keys=True)
            print(f"🧾 Wrote market thresholds → {th_path}")
    except Exception as _e:
        print(f"⚠️ Could not write market thresholds json: {_e}")

    metrics_df = pd.DataFrame(metrics_rows)
    return metrics_df

# Inside run_for_league(...) function (not shown here), replace the block starting with:
# # 4) Generate BTTS/Over (or fallback to odds), with optional volatility modifiers
# with the following:

#     # 4) Generate BTTS/Over (or fallback to odds), with optional volatility modifiers
#     has_models = (btts_model is not None and over_model is not None)
#     if has_models:
#         df = generate_btts_and_over_preds(df, btts_model, over_model)
#         if not no_volatility_adjust:
#             df = adjust_with_volatility_modifiers(df)
#     else:
#         # Try auto-loaded trained models first (from ModelStore)
#         df_before = df.copy()
#         df = score_trained_markets(df, league_name, markets=list(markets))
#         used = set(df.attrs.get("used_trained_models", []))
#         has_trained_both = ("btts" in used and "over25" in used)
#         if has_trained_both:
#             if not no_volatility_adjust:
#                 df = adjust_with_volatility_modifiers(df)
#         else:
#             if infer_from_odds:
#                 print("ℹ️ No BTTS/Over side models → inferring probabilities from bookmaker odds (and using any trained models available for other markets).")
#                 df = infer_probs_from_odds(df)
#                 # create simple binary preds for these markets if missing
#                 thr_map = {
#                     "btts_confidence": float(df.attrs.get("thr_btts", 0.55)),
#                     "over25_confidence": float(df.attrs.get("thr_over25", 0.32)),
#                     "under25_confidence": float(df.attrs.get("thr_under25", 0.60)),
#                     "btts_no_confidence": float(df.attrs.get("thr_btts_no", 0.60)),
#                 }
#                 for col, thr in thr_map.items():
#                     if col in df.columns:
#                         pred_col = col.replace("_confidence", "_pred")
#                         if pred_col not in df.columns:
#                             df[pred_col] = (pd.to_numeric(df[col], errors="coerce").fillna(0.0) >= thr).astype(int)
#             else:
#                 print("ℹ️ BTTS/Over models not available and --infer-from-odds is off; pruning markets: btts, over25, under25, btts_no.")
#                 markets[:] = [m for m in markets if m not in ("btts","over25","under25","btts_no")]

# Then, after this block and before market filtering starts, add:

#     # Also attach any other trained market probabilities if available (FTR, FTS, GE2, BTTS_FH)
#     df = score_trained_markets(df, league_name, markets=["ftr", "home_fts", "away_fts", "home_ge2", "away_ge2", "btts_fh"])

# ------------------------------
# CLI runner
# ------------------------------
def _comma_list(s: str) -> List[str]:
    if not s:
        return []
    if isinstance(s, list):
        return s
    return [x.strip() for x in str(s).split(',') if x.strip()]

def _print_summary(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        print("No metrics to display.")
        return
    cols = [c for c in [
        "league","market","status","rows",
        "cv_acc","hold_acc","cv_ap","cv_roc","hold_ap","hold_roc",
        "val_accuracy","val_pos_rate","pred_pos_rate","pred_pos_rate_ratio",
        "threshold","thr_method",
        "model_path_v2","model_path"
    ] if c in df.columns]
    with pd.option_context('display.max_rows', None, 'display.max_colwidth', 120):
        print("\n==== Training Summary ====")
        print(df[cols].to_string(index=False))

def main():
    global MODEL_ROOT
    parser = argparse.ArgumentParser(description="Train market models for a single league")
    parser.add_argument('--league', required=True, help='League name (e.g., "England Premier League")')
    parser.add_argument(
        "--matches-csv",
        default=None,
        help=(
            "Optional path to a matches CSV for this league. If omitted, defaults to the merged canonical file: "
            "Matches/__merged__/<League_With_Underscores>__merged.csv (relative to the project root)."
        ),
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help=(
            "Optional project root override. Defaults to the directory containing this script. "
            "Used to resolve the default merged CSV path when --matches-csv is not provided."
        ),
    )
    parser.add_argument('--outdir', default=MODEL_ROOT, help='Root directory to save models/metrics')
    parser.add_argument('--markets', default=','.join(DEFAULT_CFG['markets']), help='Comma-separated list of markets to train')
    parser.add_argument('--cv-folds', type=int, default=DEFAULT_CFG['cv_folds'])
    parser.add_argument('--test-size', type=float, default=DEFAULT_CFG['test_size'])
    parser.add_argument('--calibration', choices=['sigmoid','isotonic','none'], default=DEFAULT_CFG['calibration'])
    parser.add_argument('--random-state', type=int, default=DEFAULT_CFG['random_state'])
    args = parser.parse_args()

    # Resolve training source CSV.
    # If the user does not supply --matches-csv, we train from the merged canonical dataset:
    #   Matches/__merged__/<League_With_Underscores>__merged.csv
    # This aligns training with the ODDS_SYNTH + build_merged.py pipeline.
    project_root = args.project_root
    if not project_root:
        project_root = os.path.dirname(os.path.abspath(__file__))

    def _default_merged_csv_for_league(league_name: str) -> str:
        fname = f"{league_name.replace(' ', '_')}__merged.csv"
        return os.path.join(project_root, "Matches", "__merged__", fname)

    csv_path = args.matches_csv or _default_merged_csv_for_league(args.league)

    # Load data
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "Matches CSV not found: "
            f"{csv_path}\n\n"
            "If you intended to use the merged canonical dataset, run build_merged.py first so it creates: "
            "Matches/__merged__/<League_With_Underscores>__merged.csv\n"
            "Or pass an explicit file via --matches-csv."
        )
    raw_df = pd.read_csv(csv_path, low_memory=False)

    # Build cfg overrides
    cfg = {
        "cv_folds": args.cv_folds,
        "test_size": args.test_size,
        "calibration": args.calibration,
        "random_state": args.random_state,
        "markets": _comma_list(args.markets) if args.markets else DEFAULT_CFG['markets'],
    }

    # Train
    MODEL_ROOT = args.outdir  # allow redirecting model root via CLI
    print(f"📥 Training source CSV: {csv_path}")
    metrics_df = train_all_markets_for_league(raw_df, args.league, cfg)

    # Ensure league dir and save metrics
    league_dir = os.path.join(args.outdir, args.league.replace(' ', '_'))
    os.makedirs(league_dir, exist_ok=True)
    metrics_path = os.path.join(league_dir, f"{args.league.replace(' ', '_')}_market_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print("ℹ️ Threshold logic: bounded F1 with market-aware predicted-positive-rate guard rails.")

    print(f"\n✅ Saved metrics to: {metrics_path}")
    _print_summary(metrics_df)

if __name__ == '__main__':
    main()