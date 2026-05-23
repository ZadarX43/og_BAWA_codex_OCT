#!/usr/bin/env python3
"""train_investor_leagues_v2.py

Batch-train and persist V2 market models (FTR / Over2.5 / Under2.5 / BTTS) for a list of leagues.

Outputs (per league):
  ModelStore/<LEAGUE_TAG>/{ftr_v2.pkl, over25_v2.pkl, under25_v2.pkl, btts_v2.pkl}
  ModelStore/<LEAGUE_TAG>_market_thresholds.json

Notes
- Uses time-based split by match_date when available.
- Uses CatBoostClassifier for binary markets; FTR can be CatBoost or XGBoost (see --ftr-engine).
- Applies project leakage stripping if available: `_baseline_ftr_pipeline.strip_leaks`.
- Saves a *bundle dict* {model, features, league, metrics, threshold,...}.
- IMPORTANT: adds sklearn-like attrs to CatBoost model (`feature_names_in_`, `n_features_in_`, `best_threshold_`)
  so your existing overlay `_strict_align` can align and CatBoost feature-name checks don't explode.
"""

from __future__ import annotations


# --- Canonical fixture key and date coalescer from overlay (for best dedupe) ---
from prediction_overlay import _match_key
from prediction_overlay import _coalesce_match_date_series

import argparse
import os
import datetime as _dt
import json
import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


import numpy as np
import pandas as pd

# --- Optional: strict pre-match team power ratings (rolling Elo, leak-safe) ---
try:
    from team_ratings import build_rolling_power_ratings as _build_rolling_power_ratings  # type: ignore
except Exception:
    _build_rolling_power_ratings = None

# --- Optional project imports (best-effort) ---------------------------------
try:
    # Preferred: project leak-stripper
    from _baseline_ftr_pipeline import strip_leaks as _strip_leaks  # type: ignore
except Exception:
    _strip_leaks = None

try:
    # Optional: rename helper already used in your overlay
    from prediction_overlay import apply_safe_renames_and_whitelist as _apply_safe_renames  # type: ignore
except Exception:
    _apply_safe_renames = None


try:
    from catboost import CatBoostClassifier
except Exception as e:
    raise SystemExit(
        "CatBoost is required for this trainer. Install: pip install catboost\n"
        f"Import error: {e}"
    )

# --- Optional: XGBoost (for parallel FTR engines) ---------------------------
try:
    from xgboost import XGBClassifier as _XGBClassifier  # type: ignore
except Exception:
    _XGBClassifier = None  # type: ignore

# --- Optional: rolling / streak / H2H engineered features (leak-safe when shifted) ---
try:
    from streaks_module import attach_streaks_and_h2h as _attach_streaks_and_h2h  # type: ignore
except Exception:
    _attach_streaks_and_h2h = None

# --- Optional: rolling / streak / H2H engineered features (leak-safe when shifted) ---
try:
    from _baseline_ftr_pipeline import build_team_rolling_metrics as _build_team_rolling_metrics  # type: ignore
except Exception:
    _build_team_rolling_metrics = None

# --- Optional: λ (goal) feature attachment (best-effort, leak-safe when run on realised df_done) ---
try:
    from _baseline_ftr_pipeline import add_goal_predictions_to_complete_data as _add_goal_predictions_to_complete_data  # type: ignore
except Exception:
    _add_goal_predictions_to_complete_data = None

try:
    from _baseline_ftr_pipeline import attach_lambda_features_inplace as _attach_lambda_features_inplace  # type: ignore
except Exception:
    _attach_lambda_features_inplace = None

# --- Optional calibration (isotonic) ---------------------------------------
try:
    from sklearn.isotonic import IsotonicRegression as _Iso  # type: ignore
except Exception:
    _Iso = None

try:
    import joblib as _joblib  # type: ignore
except Exception:
    _joblib = None

# --- Optional: local goal-ensemble loader (to enforce leak-safe year caps) ---
try:
    from catboost import CatBoostRegressor  # type: ignore
except Exception:
    CatBoostRegressor = None  # type: ignore

def _infer_holdout_year_for_lambda(df_done: pd.DataFrame) -> Optional[int]:
    """Infer the holdout (most recent) season start year for leak-safe λ selection.

    Priority:
      1) season column start-year (e.g. 2024/2025 -> 2024)
      2) max(match_date).year
    """
    try:
        if df_done is None or df_done.empty:
            return None
    except Exception:
        return None

    # 1) season-based
    try:
        if "season" in df_done.columns:
            seasons = df_done["season"].astype("string").fillna("").str.strip()
            yrs = seasons.map(_season_start_year)
            if pd.to_numeric(yrs, errors="coerce").notna().any():
                y = int(pd.to_numeric(yrs, errors="coerce").max())
                if y >= 0:
                    return y
    except Exception:
        pass

    # 2) date-based
    try:
        md = pd.to_datetime(df_done.get("match_date"), errors="coerce")
        if md.notna().any():
            return int(md.max().year)
    except Exception:
        pass

    return None

def _goal_ensemble_path(modelstore: Path, league: str, side: str, year: Optional[int] = None) -> Path:
    tag = _modelstore_tag(league)
    base = modelstore / tag / "goal_ensembles"
    if year is None:
        return base / f"{side}_goals_fold5.pkl"
    return base / f"{int(year)}_{side}_goals_fold5.pkl"

def _try_load_goal_ensemble(modelstore: Path, league: str, side: str, *, max_year: Optional[int]) -> Optional[List[Any]]:
    """Load a CatBoost goal-ensemble list for `side` ('home'/'away').

    Policy:
      - If `max_year` is provided: prefer the newest available versioned file `YYYY_<side>_goals_fold5.pkl`
        where `YYYY <= max_year` (searching downward).
      - If no versioned file exists under the cap: warn and fall back to the unversioned alias
        `<side>_goals_fold5.pkl` if present.
      - If `max_year` is None: use the unversioned alias.

    This function prints which file is selected to make model selection auditable.
    """
    if _joblib is None:
        return None

    tag = _modelstore_tag(league)
    base = modelstore / tag / "goal_ensembles"

    # Year-capped load: pick newest available <= max_year
    if max_year is not None:
        chosen: Optional[Path] = None
        try:
            for y in range(int(max_year), 1990, -1):
                p = base / f"{y}_{side}_goals_fold5.pkl"
                if p.exists():
                    chosen = p
                    break
        except Exception:
            chosen = None

        if chosen is None:
            try:
                print(
                    f"⚠️ goal_ensemble[{league}][{side}]: cap<= {int(max_year)} but no versioned file found; trying unversioned alias"
                )
            except Exception:
                pass

            # Last resort: unversioned alias
            p = base / f"{side}_goals_fold5.pkl"
            if not p.exists():
                return None
            try:
                obj = _joblib.load(str(p))
                if isinstance(obj, list) and len(obj) > 0:
                    try:
                        print(f"⚽ goal_ensemble[{league}][{side}]: using {p.name} (unversioned fallback)")
                    except Exception:
                        pass
                    return obj
            except Exception:
                return None
            return None

        # Load chosen versioned file
        try:
            obj = _joblib.load(str(chosen))
            if isinstance(obj, list) and len(obj) > 0:
                try:
                    print(f"⚽ goal_ensemble[{league}][{side}]: using {chosen.name} (cap<= {int(max_year)})")
                except Exception:
                    pass
                return obj
        except Exception:
            return None

        return None

    # Uncapped load -> unversioned alias
    p = base / f"{side}_goals_fold5.pkl"
    if not p.exists():
        return None

    try:
        obj = _joblib.load(str(p))
        if isinstance(obj, list) and len(obj) > 0:
            try:
                print(f"⚽ goal_ensemble[{league}][{side}]: using {p.name}")
            except Exception:
                pass
            return obj
    except Exception:
        return None

    return None

def _score_goal_ensemble(df_tmp: pd.DataFrame, models: List[Any]) -> pd.Series:
    """Score df_tmp with a list of CatBoostRegressor models and return mean prediction."""
    if df_tmp is None or df_tmp.empty:
        return pd.Series([], dtype=float)

    m0 = models[0]

    # Feature name extraction (CatBoost stores these variously)
    fn: List[str] = []
    try:
        fn = list(getattr(m0, "feature_names_in_", []))
    except Exception:
        fn = []
    if not fn:
        try:
            fn = list(getattr(m0, "feature_names_", []))
        except Exception:
            fn = []
    if not fn:
        # As a last resort, use all non-target columns
        fn = [c for c in df_tmp.columns if c not in _TARGET_COLS]

    X = df_tmp.copy()

    # Ensure all required columns exist
    for c in fn:
        if c not in X.columns:
            X[c] = 0.0

    X = X[fn].copy()

    # Basic dtype hygiene: strings for objects, floats for numerics
    for c in X.columns:
        try:
            if pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c]):
                X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
            else:
                X[c] = X[c].astype("string").fillna("NA")
        except Exception:
            try:
                X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
            except Exception:
                X[c] = X[c].astype("string").fillna("NA")

    preds: List[np.ndarray] = []
    for m in models:
        try:
            p = np.asarray(m.predict(X), dtype=float)
            preds.append(p)
        except Exception:
            # If any single model fails, skip it (best-effort)
            continue

    if not preds:
        return pd.Series([np.nan] * len(df_tmp), index=df_tmp.index, dtype=float)

    P = np.vstack([p.reshape(1, -1) for p in preds])
    mu = np.nanmean(P, axis=0)
    return pd.Series(mu, index=df_tmp.index, dtype=float)

# Best-effort: reuse overlay's calibrator path logic if exposed
try:
    from prediction_overlay import _cal_path as _overlay_cal_path  # type: ignore
except Exception:
    _overlay_cal_path = None


def _calibrator_path(modelstore: Path, league: str, market: str) -> Path:
    """Resolve where to write/read calibrators for a league/market.

    Prefers the overlay's `_cal_path()` if available so runtime loading matches.
    Fallback: ModelStore/<LeagueTag>/calibrators/<market>.joblib
    """
    if callable(_overlay_cal_path):
        try:
            return Path(str(_overlay_cal_path(league, market)))
        except Exception:
            pass
    tag = _modelstore_tag(league)
    return modelstore / tag / "calibrators" / f"{market}.joblib"


def _fit_and_save_calibrators_for_league(
    league: str,
    modelstore: Path,
    *,
    models: Dict[str, Any],
    X_va: pd.DataFrame,
    y_btts_va: Optional[pd.Series] = None,
    y_over_va: Optional[pd.Series] = None,
    y_under_va: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """Fit isotonic calibrators on the (realised) validation slice and persist.

    Writes per-market calibrators to the same location runtime uses.
    Creates a light metadata JSON at ModelStore/<LeagueTag>_calibrators.json.

    This is intentionally best-effort and leak-safe: we only fit on the holdout
    slice `X_va` derived from `df_done` (already realised-only).
    """
    info: Dict[str, Any] = {
        "league": str(league),
        "tag": _modelstore_tag(league),
        "markets_fitted": [],
        "paths": {},
        "skipped": [],
    }

    if _Iso is None or _joblib is None:
        info["skipped"].append("sklearn_or_joblib_missing")
        return info

    import hashlib as _hashlib
    try:
        cols = list(X_va.columns)
        info["n_features"] = int(len(cols))
        info["features_head"] = cols[:5]
        info["features_tail"] = cols[-5:] if len(cols) > 5 else cols
        info["features_sha1"] = _hashlib.sha1("|".join(cols).encode("utf-8")).hexdigest()
    except Exception:
        pass

    def _align_X_to_model(mkt: str, mdl: Any, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure `X` matches the feature frame the model was trained on.

        Policy:
          - If the model exposes feature names, require an exact match of the *set* of columns.
          - If only order differs, reorder `X` to the model's feature order.
          - If there are missing/extra columns, raise (so drift cannot be silent).

        Returns an aligned view/copy of X.
        """
        try:
            model_feats = getattr(mdl, "feature_names_in_", None)
            if model_feats is None or len(model_feats) == 0:
                model_feats = getattr(mdl, "feature_names_", None)
        except Exception:
            model_feats = None

        if model_feats is None or len(model_feats) == 0:
            # If we can't introspect features, we cannot assert alignment safely.
            # Fail closed to prevent silent drift.
            raise ValueError(f"{mkt}: model has no feature names; cannot assert feature alignment")

        mf = [str(x) for x in list(model_feats)]
        xc = [str(x) for x in list(X.columns)]

        set_mf = set(mf)
        set_xc = set(xc)

        if set_mf != set_xc:
            missing = sorted(list(set_mf - set_xc))
            extra = sorted(list(set_xc - set_mf))
            raise ValueError(
                f"{mkt}: feature mismatch (cannot calibrate) | "
                f"missing={missing[:12]}{'...' if len(missing)>12 else ''} | "
                f"extra={extra[:12]}{'...' if len(extra)>12 else ''}"
            )

        # Same set; ensure same order
        if xc != mf:
            try:
                info.setdefault("warnings", []).append(f"{mkt}: X_va column order differed; reordering to model feature order")
            except Exception:
                pass
            X = X.reindex(columns=mf)

        return X

    def _fit_one(mkt: str, mdl: Any, y_va: pd.Series) -> None:
        yv = pd.to_numeric(y_va, errors="coerce").fillna(0).astype(int).to_numpy()
        if np.unique(yv).size < 2:
            info["skipped"].append(f"{mkt}:single_class_val")
            return
        try:
            X_va_use = _align_X_to_model(mkt, mdl, X_va)
        except Exception as e:
            info["skipped"].append(f"{mkt}:feature_mismatch:{e}")
            return

        try:
            p_raw = np.asarray(mdl.predict_proba(X_va_use)[:, 1], dtype=float)
        except Exception:
            info["skipped"].append(f"{mkt}:predict_failed")
            return

        p_raw = np.clip(p_raw, 0.0, 1.0)
        iso = _Iso(out_of_bounds="clip")
        try:
            iso.fit(p_raw, yv)
        except Exception:
            info["skipped"].append(f"{mkt}:fit_failed")
            return

        path = _calibrator_path(modelstore, league, mkt)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _joblib.dump(iso, str(path))
            info["markets_fitted"].append(mkt)
            info["paths"][mkt] = str(path)
        except Exception:
            info["skipped"].append(f"{mkt}:dump_failed")
            return

    def _fit_under_from_over(mdl_over: Any, y_va_under: pd.Series) -> None:
        """Fit an under25 calibrator from the over25 model by using p_under_raw = 1 - p_over_raw."""
        yv = pd.to_numeric(y_va_under, errors="coerce").fillna(0).astype(int).to_numpy()
        if np.unique(yv).size < 2:
            info["skipped"].append("under25:single_class_val")
            return
        try:
            X_va_use = _align_X_to_model("under25(from_over25)", mdl_over, X_va)
        except Exception as e:
            info["skipped"].append(f"under25:feature_mismatch:{e}")
            return

        try:
            p_over_raw = np.asarray(mdl_over.predict_proba(X_va_use)[:, 1], dtype=float)
        except Exception:
            info["skipped"].append("under25:predict_failed")
            return
        p_under_raw = 1.0 - np.clip(p_over_raw, 0.0, 1.0)
        p_under_raw = np.clip(p_under_raw, 0.0, 1.0)
        iso = _Iso(out_of_bounds="clip")
        try:
            iso.fit(p_under_raw, yv)
        except Exception:
            info["skipped"].append("under25:fit_failed")
            return
        path = _calibrator_path(modelstore, league, "under25")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _joblib.dump(iso, str(path))
            info["markets_fitted"].append("under25")
            info["paths"]["under25"] = str(path)
        except Exception:
            info["skipped"].append("under25:dump_failed")
            return

    # Fit calibrators for binary markets only (FTR multiclass calibration is separate)
    if "btts" in models and y_btts_va is not None:
        _fit_one("btts", models["btts"], y_btts_va)

    if "over25" in models and y_over_va is not None:
        _fit_one("over25", models["over25"], y_over_va)

    # Under25 calibrator: prefer a dedicated under25 model if present; otherwise derive from over25 via 1-p.
    if y_under_va is not None:
        if "under25" in models:
            _fit_one("under25", models["under25"], y_under_va)
        elif "over25" in models:
            _fit_under_from_over(models["over25"], y_under_va)

    # Write a small metadata file (optional; runtime currently only requires the joblib files)
    if info["markets_fitted"]:
        try:
            tag = _modelstore_tag(league)
            meta_path = modelstore / tag / "calibrators" / "calibrators_meta.json"
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            payload: Dict[str, Any] = {
                "league": str(league),
                "tag": tag,
                "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                "n_features": info.get("n_features"),
                "features_head": info.get("features_head"),
                "features_tail": info.get("features_tail"),
                "features_sha1": info.get("features_sha1"),
                "markets": {},
            }
            for mkt in info["markets_fitted"]:
                payload["markets"][mkt] = {"path": info["paths"].get(mkt, "")}
            meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            info["meta_path"] = str(meta_path)
        except Exception:
            pass

    return info

# --- Canonical renames fallback (if overlay helper not importable) -----------
_SAFE_COL_RENAMES = {
    'Pre-Match PPG (Home)': 'pre_match_ppg_home',
    'Pre-Match PPG (Away)': 'pre_match_ppg_away',
    'Home Team Pre-Match xG': 'pre_match_xg_home',
    'Away Team Pre-Match xG': 'pre_match_xg_away',
    'Average Goals Per Match (Pre-Match)': 'average_goals_per_match_pre_match',
    'Average Corners Per Match (Pre-Match)': 'average_corners_per_match_pre_match',
    'Average Cards Per Match (Pre-Match)': 'average_cards_per_match_pre_match',
    'BTTS % (Pre-Match)': 'btts_percentage_pre_match',
    'BTTS Percentage (Pre-Match)': 'btts_percentage_pre_match',
    'Over 2.5 % (Pre-Match)': 'over_25_percentage_pre_match',
    'Game Week': 'game_week',
}

# Targets + always-not-features
_TARGET_COLS = {
    'home_team_goal_count',
    'away_team_goal_count',
    'total_goal_count',
}

# Columns we never want as training features
_NON_FEATURE_COLS = {
    # identifiers / admin
    'match_id', 'fixture_id', 'id',
    'status', 'referee', 'attendance',
    # explicit outcomes (some feeds)
    'ftr_pred', 'ftr_pred_outcome',
    # (TEMP) ensure FTR never a feature (for rolling metrics compatibility)
    'FTR',
    'is_home_win',
    'is_away_win',
}


def _league_tag(league: str) -> str:
    return str(league).strip().replace(' ', '_')

def _modelstore_tag(league: str) -> str:
    league = str(league).strip()
    overrides = {
        "Australia A-League": "Australia_A_League",
        "England EFL League 1": "England_EFL_League_1",
    }
    if league in overrides:
        return overrides[league]
    return league.replace(' ', '_')

import re as _re

def _pick_latest_csv(matches_dir: Path) -> Optional[Path]:
    """DEPRECATED: do not use single-file selection in this trainer.

    This trainer is intended to be multi-season. Use `_load_all_matches_csvs()`.
    """
    return None



# -----------------------------
# Realised guards (strict training truth boundary)
# -----------------------------
_COMPLETED_RE = r"\bcomplete(?:d)?\b|\bft\b|full\s*time|finished|final|match\s*finished|aet|after\s*extra\s*time|pens?|penalt(?:y|ies)|awarded|ended"
_INCOMPLETE_RE = r"\bincomplete\b|postp|postponed|abandon|suspend|void|cancel|walkover|\bwo\b|\bns\b|not\s*started|live|in\s*play"


def _status_is_complete(s: pd.Series) -> pd.Series:
    """True for status strings that strongly imply a completed fixture."""
    x = s.astype("string").fillna("").str.strip().str.lower()
    return x.str.contains(_COMPLETED_RE, regex=True)


def _status_is_incomplete(s: pd.Series) -> pd.Series:
    """True for status strings that imply the fixture is NOT completed."""
    x = s.astype("string").fillna("").str.strip().str.lower()
    return x.str.contains(_INCOMPLETE_RE, regex=True)


def _future_fixture_mask(df: pd.DataFrame, *, grace_hours: float = 0.0) -> pd.Series:
    """Return True for rows that look like future-dated fixtures (UTC-aware parsing)."""
    if df is None or df.empty:
        return pd.Series(False, index=getattr(df, "index", None))

    idx = df.index
    dt = pd.Series(pd.NaT, index=idx, dtype="datetime64[ns, UTC]")

    for col in ("match_date", "date_GMT", "date", "Date", "timestamp"):
        if col not in df.columns:
            continue
        try:
            try:
                cand = pd.to_datetime(df[col], errors="coerce", utc=True, format="mixed")
            except TypeError:
                cand = pd.to_datetime(df[col], errors="coerce", utc=True)
        except Exception:
            try:
                try:
                    cand = pd.to_datetime(df[col].astype(str), errors="coerce", utc=True, format="mixed")
                except TypeError:
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

    return dt.notna() & (dt > now_utc)


def _realised_mask(df: pd.DataFrame) -> pd.Series:
    """Strict realised-mask: exclude future rows and explicit incomplete statuses.

    A row is "realised" if:
      - status says complete, OR (status unknown/blank) AND goals present AND date known
    AND:
      - goals present
      - status is not explicitly incomplete
      - fixture is not future-dated
    """
    if df is None or df.empty:
        return pd.Series(False, index=getattr(df, "index", None))

    idx = df.index

    hg = pd.to_numeric(df.get("home_team_goal_count", pd.Series(np.nan, index=idx)), errors="coerce")
    ag = pd.to_numeric(df.get("away_team_goal_count", pd.Series(np.nan, index=idx)), errors="coerce")
    goals_present = ~(hg.isna() | ag.isna())

    status_txt = (
        df["status"].astype("string").fillna("").str.strip().str.lower()
        if "status" in df.columns
        else pd.Series("", index=idx, dtype="string")
    )
    st_complete = _status_is_complete(status_txt) if "status" in df.columns else pd.Series(False, index=idx)
    st_incomp = _status_is_incomplete(status_txt) if "status" in df.columns else pd.Series(False, index=idx)

    # Only treat status as "known" if it hits either complete or incomplete regex
    st_known = status_txt.ne("") & (st_complete | st_incomp)

    # Require a parseable date for fallback path (avoids undated placeholder 0-0 rows)
    dt = pd.Series(pd.NaT, index=idx, dtype="datetime64[ns, UTC]")
    for col in ("match_date", "date_GMT", "date", "Date", "timestamp"):
        if col not in df.columns:
            continue
        try:
            try:
                cand = pd.to_datetime(df[col], errors="coerce", utc=True, format="mixed")
            except TypeError:
                cand = pd.to_datetime(df[col], errors="coerce", utc=True)
        except Exception:
            try:
                try:
                    cand = pd.to_datetime(df[col].astype(str), errors="coerce", utc=True, format="mixed")
                except TypeError:
                    cand = pd.to_datetime(df[col].astype(str), errors="coerce", utc=True)
            except Exception:
                continue
        dt = dt.fillna(cand)
    date_known = dt.notna()

    # completed if:
    #  - status says complete
    #  - OR status is unknown/blank AND goals are present AND date is known
    base = (st_complete | ((~st_known) & goals_present & date_known)) & goals_present

    future = _future_fixture_mask(df)
    return base & ~st_incomp & ~future




def _load_all_matches_csvs(matches_dir: Path) -> pd.DataFrame:
    """Read + concat ALL CSVs under Matches/<League>/ (multi-season)."""
    if not matches_dir.exists():
        return pd.DataFrame()

    # Robustly collect CSVs. `glob()` can return broken symlinks or files that
    # disappear between listing and stat; skip anything we can't stat.
    _cands_with_mtime: List[Tuple[float, Path]] = []
    _skipped_missing: List[str] = []
    for p in matches_dir.glob("*.csv"):
        try:
            mt = float(p.stat().st_mtime)
        except FileNotFoundError:
            _skipped_missing.append(str(p))
            continue
        except Exception:
            # If stat fails for any other reason, skip quietly.
            continue
        _cands_with_mtime.append((mt, p))

    cands = [p for _, p in sorted(_cands_with_mtime, key=lambda t: t[0], reverse=False)]

    if not cands:
        if _skipped_missing:
            try:
                print(f"⚠️  {matches_dir}: skipped missing/broken files (first 5): {_skipped_missing[:5]}")
            except Exception:
                pass
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for p in cands:
        try:
            df = pd.read_csv(p)
            df["__src_csv"] = p.name
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, axis=0, ignore_index=True, sort=False)

    # Canonical renames + numeric coercion + match_date parse
    df = _apply_renames(df)
    df = _coerce_numbers(df)

    # Robust match_date parsing (UTC-naive) to avoid tz/mixed-dtype warnings in pandas.
    df = _ensure_match_date(df)

    # Ensure team name columns exist (some feeds use Home/Away)
    if "home_team_name" not in df.columns and "Home" in df.columns:
        df["home_team_name"] = df["Home"]
    if "away_team_name" not in df.columns and "Away" in df.columns:
        df["away_team_name"] = df["Away"]

    # Canonical fixture_key for dedupe (match overlay/meta/backtests)
    df["fixture_key"] = pd.NA
    try:
        df["fixture_key"] = df.apply(_match_key, axis=1)
    except Exception:
        # If canonical keying fails, do NOT invent an alternate format.
        # Drop rows we cannot key safely.
        pass

    # Drop rows without a usable fixture_key (prevents silent join drift + bad dedupe)
    df["fixture_key"] = df["fixture_key"].astype("string").fillna("").str.strip()
    bad_key = df["fixture_key"].eq("")
    if bool(bad_key.any()):
        df = df.loc[~bad_key].copy()

    # Ensure numeric goals exist for scoring
    for c in ("home_team_goal_count", "away_team_goal_count"):
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Best-row scoring: completed status wins, then goals_present, then more filled columns; future rows lose hard
    nn = df.notna().sum(axis=1)

    status_txt = (
        df["status"].astype("string").fillna("").str.strip().str.lower()
        if "status" in df.columns else pd.Series("", index=df.index, dtype="string")
    )
    status_complete = _status_is_complete(status_txt) if "status" in df.columns else pd.Series(False, index=df.index)

    goals_present = df[["home_team_goal_count", "away_team_goal_count"]].notna().all(axis=1)

    future = _future_fixture_mask(df)

    score = (status_complete.astype(int) * 1000) + (goals_present.astype(int) * 100) + nn - (future.astype(int) * 5000)

    df = df.assign(__score=score).sort_values(["fixture_key", "__score"], ascending=[True, False])
    df = df.drop_duplicates(subset=["fixture_key"], keep="first").drop(columns=["__score"], errors="ignore")

    # Re-assert canonical key (belt + braces) so all downstream joins work
    try:
        df["fixture_key"] = df.apply(_match_key, axis=1)
    except Exception:
        # If this fails, leave as-is; we already dropped bad keys above.
        pass

    return df


def _ensure_match_date(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort match_date normalisation.

    Goals:
      - Prefer stable parsing for common ISO formats first (fast + deterministic).
      - Fall back to general parsing only for leftovers.
      - Support epoch timestamps (seconds/ms) when `timestamp` is present.

    Produces `match_date` as pandas datetime64[ns] (UTC-naive).
    """
    out = df.copy()

    def _parse_series(s: pd.Series) -> pd.Series:
        # Work with strings where appropriate; keep NaNs
        s_str = s.astype("string").str.strip()

        def _parse_utc_naive(ss: pd.Series, fmt: Optional[str] = None) -> pd.Series:
            """Parse as UTC, then drop tz -> datetime64[ns] (UTC-naive)."""
            try:
                dt = pd.to_datetime(ss, errors="coerce", format=fmt, utc=True)
            except TypeError:
                # Older pandas compatibility
                dt = pd.to_datetime(ss, errors="coerce", format=fmt)
                dt = pd.to_datetime(dt, errors="coerce", utc=True)
            except Exception:
                dt = pd.to_datetime(ss, errors="coerce", utc=True)

            # dt is typically a Series[datetime64[ns, UTC]]; convert to UTC-naive
            try:
                if hasattr(dt, "dt"):
                    dt = dt.dt.tz_convert(None)
            except Exception:
                pass

            # Ensure a writable Series aligned to ss.index
            if not isinstance(dt, pd.Series):
                dt = pd.Series(dt, index=ss.index)
            else:
                dt = dt.reindex(ss.index)

            # Force UTC-naive dtype
            try:
                dt = pd.to_datetime(dt, errors="coerce")
            except Exception:
                pass
            return dt

        # Start with strict ISO date
        parsed = _parse_utc_naive(s_str, fmt="%Y-%m-%d")

        # Strict ISO datetime for remaining
        m = parsed.isna() & s_str.ne("")
        if bool(m.any()):
            parsed2 = _parse_utc_naive(s_str[m], fmt="%Y-%m-%d %H:%M:%S")
            # Assign using numpy array to avoid pandas datetimelike setter warnings
            parsed.loc[m] = pd.to_datetime(parsed2, errors="coerce").to_numpy(dtype="datetime64[ns]")

        # General parse for remaining
        m = parsed.isna() & s_str.ne("")
        if bool(m.any()):
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Could not infer format*")
                parsed3 = _parse_utc_naive(s_str[m], fmt=None)
            parsed.loc[m] = pd.to_datetime(parsed3, errors="coerce").to_numpy(dtype="datetime64[ns]")

        return parsed

    # 1) match_date if present
    if "match_date" in out.columns:
        md = _parse_series(out["match_date"])
        if md.notna().any():
            out["match_date"] = md
            return out

    # 2) common alternatives
    for alt in ("date_GMT", "Date", "date"):
        if alt not in out.columns:
            continue
        md = _parse_series(out[alt])
        if md.notna().any():
            out["match_date"] = md
            return out

    # 3) epoch timestamps (often seconds/ms)
    if "timestamp" in out.columns:
        ts = pd.to_numeric(out["timestamp"], errors="coerce")
        if ts.notna().any():
            try:
                med = float(ts.dropna().median())
            except Exception:
                med = 0.0
            unit = "ms" if med > 1e12 else "s"
            md = pd.to_datetime(ts, unit=unit, errors="coerce", utc=True)
            try:
                md = md.dt.tz_convert(None)
            except Exception:
                pass
            if md.notna().any():
                out["match_date"] = md
                return out

    # nothing usable
    out["match_date"] = pd.NaT
    return out


def _apply_renames(df: pd.DataFrame) -> pd.DataFrame:
    if _apply_safe_renames is not None:
        try:
            return _add_recent_btts_regime_features(_apply_safe_renames(df))
        except Exception:
            pass

    out = df.copy()
    to_rename = {k: v for k, v in _SAFE_COL_RENAMES.items() if k in out.columns and v not in out.columns}
    if to_rename:
        out = out.rename(columns=to_rename)
    out = _add_recent_btts_regime_features(out)
    return out


def _pct_from_series(s: pd.Series) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce")
    if vals.dropna().empty:
        return vals
    try:
        max_abs = float(vals.dropna().abs().max())
    except Exception:
        max_abs = float("nan")
    if np.isfinite(max_abs) and max_abs <= 1.000001:
        return vals * 100.0
    return vals


def _add_recent_btts_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add inference-safe recent BTTS regime blends from existing rolling columns.

    Prefer true rolling/API columns when available, but fall back to the merged
    side-specific BTTS rolling rates already present in the canonical estate.
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    def _first_present(*names: str) -> Optional[str]:
        for name in names:
            if name in out.columns:
                return name
        return None

    h5 = _first_present("home_btts_rate_l5", "btts_rate_5_home")
    a5 = _first_present("away_btts_rate_l5", "btts_rate_5_away")
    h10 = _first_present("home_btts_rate_l10", "btts_rate_10_home")
    a10 = _first_present("away_btts_rate_l10", "btts_rate_10_away")
    c5 = _first_present("combined_btts_rate_l5")

    if c5 is not None:
        out["recent_btts_regime_blend_l5"] = pd.to_numeric(_pct_from_series(out[c5]), errors="coerce").round(4)
    elif h5 is not None and a5 is not None:
        out["recent_btts_regime_blend_l5"] = (
            pd.concat([_pct_from_series(out[h5]), _pct_from_series(out[a5])], axis=1)
            .mean(axis=1, skipna=True)
            .round(4)
        )

    if h10 is not None and a10 is not None:
        out["recent_btts_regime_blend_l10"] = (
            pd.concat([_pct_from_series(out[h10]), _pct_from_series(out[a10])], axis=1)
            .mean(axis=1, skipna=True)
            .round(4)
        )

    if "recent_btts_regime_blend_l5" in out.columns:
        out["recent_no_btts_regime_blend_l5"] = (100.0 - pd.to_numeric(out["recent_btts_regime_blend_l5"], errors="coerce")).round(4)
    if "recent_btts_regime_blend_l10" in out.columns:
        out["recent_no_btts_regime_blend_l10"] = (100.0 - pd.to_numeric(out["recent_btts_regime_blend_l10"], errors="coerce")).round(4)
    return out


def _merged_already_has_streak_h2h_family(df: pd.DataFrame) -> bool:
    """True when the canonical merged input already carries the stage-4 streak/H2H block."""
    if df is None or df.empty:
        return False
    required = {
        "btts_streak_home",
        "btts_streak_away",
        "h2h_btts_rate",
        "xg_for_pm_home",
        "xg_for_pm_away",
    }
    return required.issubset(set(map(str, df.columns)))


def _collapse_xy_feature_suffixes(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicated pandas merge suffixes back onto canonical unsuffixed names."""
    if df is None or df.empty:
        return df

    out = df.copy()
    suffix_cols = [c for c in out.columns if isinstance(c, str) and (c.endswith("_x") or c.endswith("_y"))]
    if not suffix_cols:
        return out

    grouped: Dict[str, List[str]] = {}
    for col in suffix_cols:
        grouped.setdefault(col[:-2], []).append(col)

    for base, dup_cols in grouped.items():
        merged = out[base].copy() if base in out.columns else out[dup_cols[0]].copy()
        for col in dup_cols:
            try:
                merged = merged.where(merged.notna(), out[col])
            except Exception:
                pass
        out[base] = merged
        out = out.drop(columns=dup_cols, errors="ignore")

    return out


_TRAINING_ONLY_BINARY_FEATURE_CANDIDATES = {
    "rolling5_home_shots",
    "rolling5_away_shots",
    "rolling10_home_shots",
    "rolling10_away_shots",
    "rolling5_home_sot_ratio",
    "rolling5_away_sot_ratio",
    "rolling10_home_sot_ratio",
    "rolling10_away_sot_ratio",
    "rolling10_home_press_intensity",
    "rolling10_away_press_intensity",
    "rolling10_press_intensity_diff",
    "rolling10_home_press_z",
    "rolling10_away_press_z",
    "rolling10_press_z_diff",
}

def _prune_training_only_binary_features(
    df: pd.DataFrame,
    *,
    source_cols: set[str],
) -> pd.DataFrame:
    """Drop binary-market features that were only created in-memory and are not inference-safe yet."""
    if df is None or df.empty:
        return df

    out = df.copy()
    drop_cols = [c for c in _TRAINING_ONLY_BINARY_FEATURE_CANDIDATES if c in out.columns and c not in source_cols]
    if drop_cols:
        try:
            print(
                f"🧹 binary_contract: dropping {len(drop_cols)} training-only cols absent from merged source: "
                f"{drop_cols[:8]}{'...' if len(drop_cols) > 8 else ''}"
            )
        except Exception:
            pass
        out = out.drop(columns=drop_cols, errors="ignore")
    return out


def _coerce_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric-like columns safely.

    - Avoids deprecated `errors='ignore'`.
    - Prevents accidental coercion of categorical identifiers (teams, refs, etc.).
    - Converts object columns *only if* they are mostly numeric-like after coercion.
    """
    out = df.copy()

    # Heuristic: never coerce true identifier/text columns.
    # IMPORTANT: do NOT blanket-skip tokens like "home"/"away" because that blocks
    # numeric feature cols such as home_goals_pred, p_home_fts, home_power_rating, etc.
    _never_coerce_exact = {
        "match_id", "fixture_id", "id",
        "status", "referee", "attendance",
        "league", "league_name", "country",
        "season", "round", "game_week",
        "home_team_name", "away_team_name",
        "home", "away",
        "venue", "stadium",
        "__src_csv",
    }
    _never_coerce_suffixes = ("_name", "_id")

    for c in list(out.columns):
        lc = str(c).lower()

        # Skip obvious identifier columns
        if (lc in _never_coerce_exact) or lc.endswith(_never_coerce_suffixes):
            continue

        s = out[c]

        # If already numeric/bool, just coerce and continue
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            out[c] = pd.to_numeric(s, errors="coerce")
            continue

        # Only attempt object/string coercion when it looks numeric-like
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            # Normalise common numeric formatting
            s_str = s.astype("string").str.strip()
            # remove percent sign; keep as numeric (we do not scale to 0..1 here)
            s_str = s_str.str.replace("%", "", regex=False)
            # European decimal comma
            s_str = s_str.str.replace(",", ".", regex=False)

            num = pd.to_numeric(s_str, errors="coerce")

            # Decide whether this column is truly numeric-like
            non_null = s_str.notna() & s_str.ne("")
            denom = int(non_null.sum())
            if denom == 0:
                continue
            ratio = float(num.notna().sum()) / float(denom)

            # If most entries convert, treat it as numeric; else leave it alone
            if ratio >= 0.80:
                out[c] = num

    # Final pass: ensure any numeric dtypes are proper numeric with NaNs for bad values
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out



def _completed_only(df: pd.DataFrame) -> pd.DataFrame:
    """Strict training-time filter: realised fixtures only (leak-safe)."""
    if df is None or df.empty:
        return df.iloc[0:0].copy()

    before = int(len(df))
    m = _realised_mask(df)
    out = df.loc[m].copy()

    try:
        future_goals = _future_fixture_mask(df) & df[["home_team_goal_count", "away_team_goal_count"]].notna().all(axis=1)
        print(f"🧹 TRAIN realised_only kept {len(out)}/{before} | future_goals_dropped={int(future_goals.sum())}")
    except Exception:
        pass

    return out


def _build_targets(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    hg = pd.to_numeric(df['home_team_goal_count'], errors='coerce')
    ag = pd.to_numeric(df['away_team_goal_count'], errors='coerce')

    y_btts  = ((hg >= 1) & (ag >= 1)).astype(int)
    y_over  = ((hg + ag) >= 3).astype(int)
    y_under = ((hg + ag) <= 2).astype(int)

    # FTR: 0=Home, 1=Draw, 2=Away
    diff = (hg - ag)
    y_ftr = diff.apply(lambda d: 0 if d > 0 else (2 if d < 0 else 1)).astype(int)

    return y_btts, y_over, y_under, y_ftr


def _time_split(df: pd.DataFrame, val_frac: float = 0.2) -> Tuple[pd.Index, pd.Index]:
    if 'match_date' in df.columns:
        md = pd.to_datetime(df['match_date'], errors='coerce')
        if md.notna().any():
            order = md.sort_values(kind='mergesort').index
            n = len(order)
            cut = max(1, int(round((1.0 - val_frac) * n)))
            tr = order[:cut]
            va = order[cut:]
            return tr, va

    idx = df.index.to_list()
    n = len(idx)
    cut = max(1, int(round((1.0 - val_frac) * n)))
    return pd.Index(idx[:cut]), pd.Index(idx[cut:])



# --- Season-based holdout helpers ---
def _season_start_year(x: str) -> int:
    # Handles "2024/2025", "2024-2025", "2024", etc.
    m = _re.search(r"(19|20)\d{2}", str(x))
    return int(m.group(0)) if m else -1


def _time_split_most_recent_season(df: pd.DataFrame, val_frac: float = 0.2) -> Tuple[pd.Index, pd.Index]:
    """
    Prefer: hold out the most recent season if a season column exists.
    Fallback: last val_frac portion by match_date.
    """
    # 1) Season-based holdout
    if "season" in df.columns:
        seasons = df["season"].astype("string").fillna("").str.strip()
        yrs = seasons.map(_season_start_year)
        if yrs.max() >= 0 and yrs.nunique() >= 2:
            recent_year = int(yrs.max())
            is_val = (yrs == recent_year)
            va = df.index[is_val]
            tr = df.index[~is_val]
            if len(tr) > 0 and len(va) > 0:
                return tr, va

    # 2) Date-based holdout (last val_frac by match_date)
    if "match_date" in df.columns:
        md = pd.to_datetime(df["match_date"], errors="coerce")
        if md.notna().any():
            order = md.sort_values(kind="mergesort").index
            n = len(order)
            cut = max(1, int(round((1.0 - val_frac) * n)))
            return order[:cut], order[cut:]

    # 3) Fallback to index split
    idx = df.index.to_list()
    n = len(idx)
    cut = max(1, int(round((1.0 - val_frac) * n)))
    return pd.Index(idx[:cut]), pd.Index(idx[cut:])


# --- Explicit holdout split with diagnostics ---
def _compute_holdout_split(df: pd.DataFrame, val_frac: float = 0.2) -> Tuple[pd.Index, pd.Index, Dict[str, Any]]:
    """Return (train_idx, val_idx, info) for the holdout split.

    Preference order:
      1) Hold out most recent season (if season column exists with >=2 unique years)
      2) Hold out last `val_frac` by match_date
      3) Fallback to index split

    The returned `info` is printed by the caller to avoid silent mis-splits.
    """
    info: Dict[str, Any] = {"method": "index", "val_frac": float(val_frac)}

    # 1) Season-based
    if "season" in df.columns:
        seasons = df["season"].astype("string").fillna("").str.strip()
        yrs = seasons.map(_season_start_year)
        uniq_years = sorted([int(x) for x in pd.unique(yrs) if int(x) >= 0])
        if len(uniq_years) >= 2:
            recent_year = int(max(uniq_years))
            is_val = (yrs == recent_year)
            va = df.index[is_val]
            tr = df.index[~is_val]
            if len(tr) > 0 and len(va) > 0:
                info.update({
                    "method": "season",
                    "season_years": uniq_years,
                    "recent_year": recent_year,
                    "n_train": int(len(tr)),
                    "n_val": int(len(va)),
                })
                return tr, va, info

    # 2) Date-based
    if "match_date" in df.columns:
        md = pd.to_datetime(df["match_date"], errors="coerce")

        # ✅ NaT dates never go to validation (they break chronological logic)
        valid = md.notna()
        if bool(valid.any()):
            order = md.loc[valid].sort_values(kind="mergesort").index
            n = len(order)
            cut = max(1, int(round((1.0 - val_frac) * n)))

            tr = order[:cut]
            va = order[cut:]

            # push NaT rows into train
            nat_idx = df.index[~valid]
            if len(nat_idx):
                tr = tr.append(nat_idx)

            boundary = md.loc[va[0]] if len(va) else pd.NaT

            info.update({
                "method": "date",
                "date_min": str(md.min()),
                "date_max": str(md.max()),
                "boundary": str(boundary),
                "n_train": int(len(tr)),
                "n_val": int(len(va)),
                "n_nat_train": int(len(nat_idx)),
            })
            return tr, va, info

    # 3) Index fallback
    idx = df.index.to_list()
    n = len(idx)
    cut = max(1, int(round((1.0 - val_frac) * n)))
    tr = pd.Index(idx[:cut])
    va = pd.Index(idx[cut:])
    info.update({"n_train": int(len(tr)), "n_val": int(len(va))})
    return tr, va, info


def _make_feature_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[int]]:
    """Return (X, feature_names, cat_feature_indices).

    Policy:
      - Drop true post-match / in-play leak columns (shots, corners, cards, HT stats, timings, etc.)
      - KEEP leak-safe engineered features even if token-matching:
          rolling*, streak_*, h2h_*, press_*, power_*, team_*, *pre_match*
      - For Teams-derived aggregates that start with home_team_/away_team_, only treat them as safe
        when they look like aggregates (per_match / percentage / avg / rate / ratio / idx) —
        never keep raw match stat columns like home_team_shots.
    """
    work = df.copy()

    import os
    dbg_lambda = os.getenv("OG_LAMBDA_DEBUG", "").lower() in ("1", "true", "yes", "y")

    def _lambda_nn_rate(frame: pd.DataFrame, col: str) -> float:
        try:
            if col not in frame.columns:
                return float("nan")
            s = pd.to_numeric(frame[col], errors="coerce")
            return float(s.notna().mean())
        except Exception:
            return float("nan")

    # Preserve λ/goal-projection columns even if `_strip_leaks` is overly aggressive.
    #
    # We preserve in two ways:
    #   A) by a stable row id (__og_row_id) that usually survives strip_leaks (best)
    #   B) by a stable match key (fixture_key / computed _match_key) as a fallback
    _lambda_keep_cols = [
        "home_goals_pred",
        "away_goals_pred",
        "lambda_home",
        "lambda_away",
        "exp_goals_sum",
        "p00_est",
    ]

    # A) Row-id preservation (most robust when strip_leaks keeps columns)
    if "__og_row_id" not in work.columns:
        work["__og_row_id"] = np.arange(len(work), dtype=np.int64)
    # ✅ Make row-id the canonical index (survives merges/reorders)
    work = work.set_index("__og_row_id", drop=False).sort_index()
    _lambda_keep_df: Optional[pd.DataFrame] = None
    try:
        _lambda_keep_df = work[["__og_row_id"]].copy()
        for _c in _lambda_keep_cols:
            if _c in work.columns:
                _lambda_keep_df[_c] = pd.to_numeric(work[_c], errors="coerce")
    except Exception:
        _lambda_keep_df = None

    if dbg_lambda:
        try:
            rates = {c: _lambda_nn_rate(work, c) for c in _lambda_keep_cols if c in work.columns}
            print(f"[train_investor_leagues_v2] lambda pre_strip: n={len(work)} cols={list(rates.keys())} nn_rate={rates}")
        except Exception:
            pass

    # B) Key-based preservation (fallback)
    def _stable_lambda_key(frame: pd.DataFrame) -> pd.Series:
        """Return a stable join key for restoring λ columns after leak stripping.

        Preference order:
          1) fixture_key when present and non-empty
          2) computed _match_key from match_date + team names
          3) index string
        """
        if frame is None or frame.empty:
            return pd.Series([], dtype="string")

        # 1) fixture_key when usable
        if "fixture_key" in frame.columns:
            try:
                fk = frame["fixture_key"].astype("string").fillna("").str.strip()
                if bool(fk.ne("").any()):
                    return fk.where(fk.ne(""), pd.Series(frame.index.astype(str), index=frame.index)).astype("string")
            except Exception:
                pass

        # 2) compute match key from match_date + teams
        if {"match_date", "home_team_name", "away_team_name"}.issubset(frame.columns):
            try:
                tmp = frame[["match_date", "home_team_name", "away_team_name"]].copy()
                tmp["match_date"] = pd.to_datetime(tmp["match_date"], errors="coerce", utc=True)
                try:
                    tmp["match_date"] = tmp["match_date"].dt.tz_convert(None)
                except Exception:
                    pass
                k = tmp.apply(_match_key, axis=1)
                k = k.astype("string").fillna("").str.strip()
                return k.where(k.ne(""), pd.Series(frame.index.astype(str), index=frame.index)).astype("string")
            except Exception:
                pass

        # 3) final fallback
        return pd.Series(frame.index.astype(str), index=frame.index, dtype="string")

    _lambda_key: Optional[pd.Series] = None
    _lambda_keep_by_key: Dict[str, pd.Series] = {}

    try:
        _lambda_key = _stable_lambda_key(work)
        for _c in _lambda_keep_cols:
            if _c in work.columns:
                _lambda_keep_by_key[_c] = pd.to_numeric(work[_c], errors="coerce")
    except Exception:
        _lambda_key = None
        _lambda_keep_by_key = {}

    if callable(_strip_leaks):
        try:
            work = _strip_leaks(work)
        except Exception:
            pass

    # If strip_leaks dropped __og_row_id but preserved index alignment, restore it
    if "__og_row_id" not in work.columns:
        if isinstance(_lambda_keep_df, pd.DataFrame):
            try:
                if work.index.equals(_lambda_keep_df.index):
                    work["__og_row_id"] = pd.to_numeric(_lambda_keep_df["__og_row_id"], errors="coerce")
            except Exception:
                pass
        # If __og_row_id survived as an index name, recover it
        try:
            if "__og_row_id" not in work.columns and getattr(work.index, "name", None) == "__og_row_id":
                work["__og_row_id"] = pd.to_numeric(pd.Index(work.index), errors="coerce")
        except Exception:
            pass

    if dbg_lambda:
        try:
            rates = {c: _lambda_nn_rate(work, c) for c in _lambda_keep_cols if c in work.columns}
            print(f"[train_investor_leagues_v2] lambda post_strip: n={len(work)} has_row_id={'__og_row_id' in work.columns} nn_rate={rates}")
        except Exception:
            pass

    # Restore preserved λ cols:
    # 1) Prefer row-id merge when possible.
    # 2) Otherwise fall back to key-based mapping.
    if isinstance(_lambda_keep_df, pd.DataFrame) and (not work.empty) and ("__og_row_id" in work.columns):
        try:
            work = work.merge(_lambda_keep_df, on="__og_row_id", how="left", suffixes=("", "_lamkeep"))
            work = work.set_index("__og_row_id", drop=False).sort_index()
            for _c in _lambda_keep_cols:
                cu = f"{_c}_lamkeep"
                if cu in work.columns:
                    # Coalesce: prefer existing values, else take lamkeep
                    if _c in work.columns:
                        a = pd.to_numeric(work[_c], errors="coerce")
                        b = pd.to_numeric(work[cu], errors="coerce")
                        work[_c] = a.where(a.notna(), b)
                    else:
                        work[_c] = pd.to_numeric(work[cu], errors="coerce")
                    work = work.drop(columns=[cu], errors="ignore")

            # Some merges bring λ columns in without suffix if they weren't present on the left.
            # Ensure they are numeric so they land in num_cols.
            for _c in _lambda_keep_cols:
                if _c in work.columns:
                    work[_c] = pd.to_numeric(work[_c], errors="coerce")

        except Exception:
            pass
    elif _lambda_keep_by_key and (_lambda_key is not None) and (not work.empty):
        try:
            cur_key = _stable_lambda_key(work)

            base = pd.DataFrame({"__k": _lambda_key.astype("string")})
            for _c, _v in _lambda_keep_by_key.items():
                base[_c] = pd.to_numeric(_v, errors="coerce").to_numpy()

            base["__k"] = base["__k"].astype("string").fillna("").str.strip()
            base = base[base["__k"].ne("")].copy()
            base = base.drop_duplicates(subset=["__k"], keep="first")
            base = base.set_index("__k", drop=True)

            for _c in _lambda_keep_cols:
                if _c in base.columns:
                    mapped = cur_key.map(base[_c])
                    if _c in work.columns:
                        a = pd.to_numeric(work[_c], errors="coerce")
                        work[_c] = a.where(a.notna(), pd.to_numeric(mapped, errors="coerce"))
                    else:
                        work[_c] = pd.to_numeric(mapped, errors="coerce")
        except Exception:
            pass

    # Ensure numeric dtype for λ columns (so they land in `num_cols`).
    for _c in ("home_goals_pred", "away_goals_pred", "lambda_home", "lambda_away", "exp_goals_sum", "p00_est"):
        if _c in work.columns:
            work[_c] = pd.to_numeric(work[_c], errors="coerce")

    # Convenience derivations when only partial λ is present
    if ("exp_goals_sum" not in work.columns) and ("home_goals_pred" in work.columns) and ("away_goals_pred" in work.columns):
        work["exp_goals_sum"] = pd.to_numeric(work["home_goals_pred"], errors="coerce") + pd.to_numeric(work["away_goals_pred"], errors="coerce")
    if ("p00_est" not in work.columns) and ("exp_goals_sum" in work.columns):
        work["p00_est"] = np.exp(-pd.to_numeric(work["exp_goals_sum"], errors="coerce").clip(lower=0.0))

    if dbg_lambda:
        try:
            rates = {c: _lambda_nn_rate(work, c) for c in _lambda_keep_cols if c in work.columns}
            print(f"[train_investor_leagues_v2] lambda post_restore: n={len(work)} nn_rate={rates}")
        except Exception:
            pass

    drop = set(_TARGET_COLS) | set(_NON_FEATURE_COLS)

    # Extra hardening: drop common outcome-encoding aliases that some feeds include.
    # These can encode the final result directly under different names and yield near-perfect val acc.
    _outcome_alias_cols = {
        # generic
        "result", "match_result", "full_time_result", "ft_result", "winner", "match_winner", "winning_team",
        "final_score", "ft_score", "full_time_score", "half_time_score", "ht_score",
        # home/away variants
        "home_score", "away_score", "home_team_score", "away_team_score",
        "home_goals", "away_goals", "home_team_goals", "away_team_goals",
        "goals_home", "goals_away",
        "home_ft_goals", "away_ft_goals", "home_ht_goals", "away_ht_goals",
        "home_full_time_goals", "away_full_time_goals", "home_half_time_goals", "away_half_time_goals",
        # team_a/team_b variants
        "team_a_score", "team_b_score", "team_a_goals", "team_b_goals",
        "team_a_goal_count", "team_b_goal_count",
    }
    for _c in list(work.columns):
        try:
            if str(_c).strip().lower() in _outcome_alias_cols:
                drop.add(_c)
        except Exception:
            continue

    _leak_tokens = (
        "goal_timings",
        "shots",
        "possession",
        "corner",
        "yellow",
        "red",
        "foul",
        "half_time",
        "first_half",
        "second_half",
        "total_goals_at_half",
        "goal_count_half",
    )

    _safe_prefixes = (
        "rolling", "streak_", "h2h_", "press_", "power_", "team_", "recent_",
    )

    # Only allow home_team_/away_team_ columns that look like aggregates, and never the raw match stat names.
    _team_agg_markers = (
        "per_match", "percentage", "avg", "mean", "rate", "ratio", "idx", "index",
        "xg", "ppg", "elo", "power", "baseline", "points", "goals", "gf", "ga",
        "xgf", "xga", "btts", "over", "under", "cards", "corners",
    )
    _raw_match_team_cols = {
        "home_team_shots", "home_team_shots_on_target", "home_team_shots_off_target",
        "away_team_shots", "away_team_shots_on_target", "away_team_shots_off_target",
        "home_team_possession", "away_team_possession",
        "home_team_corner_count", "away_team_corner_count",
        "home_team_fouls", "away_team_fouls",
        "home_team_yellow_cards", "away_team_yellow_cards",
        "home_team_red_cards", "away_team_red_cards",
        "total_goals_at_half_time",
        "home_team_goal_count_half_time", "away_team_goal_count_half_time",
    }

    def _is_safe_feature(col: str) -> bool:
        lc = str(col).lower()
        if lc in _raw_match_team_cols:
            return False
        if lc.startswith(_safe_prefixes):
            return True
        if "pre_match" in lc or lc.endswith("_pre_match"):
            return True
        if lc.startswith("home_team_") or lc.startswith("away_team_"):
            # only allow if clearly pre-match / rolling / streak / h2h derived
            if ("pre_match" in lc) or lc.endswith("_pre_match"):
                return True
            if ("rolling" in lc) or lc.startswith("rolling"):
                return True
            if ("streak" in lc) or lc.startswith("streak_"):
                return True
            if ("h2h" in lc) or lc.startswith("h2h_"):
                return True
            return False
        return False

    for c in list(work.columns):
        if c in drop:
            continue
        lc = str(c).lower()
        if any(tok in lc for tok in _leak_tokens) and (not _is_safe_feature(c)):
            drop.add(c)

    # --- SAFEGUARD: never drop λ goal-projection columns ---
    # These are *explicitly* engineered pre-match signals and must survive feature selection.
    # If they appear in `drop`, log why (debug) and remove them.
    _lam_force_keep = [
        "home_goals_pred",
        "away_goals_pred",
        "lambda_home",
        "lambda_away",
        "exp_goals_sum",
        "p00_est",
    ]

    if dbg_lambda:
        try:
            reasons = {}
            for _c in _lam_force_keep:
                if _c in drop:
                    r = []
                    if _c in _TARGET_COLS:
                        r.append("TARGET_COLS")
                    if _c in _NON_FEATURE_COLS:
                        r.append("NON_FEATURE_COLS")
                    if not r:
                        r.append("TOKEN_OR_OTHER")
                    reasons[_c] = r
            if reasons:
                print(f"[train_investor_leagues_v2] lambda drop_reasons: {reasons}")
        except Exception:
            pass

    for _c in _lam_force_keep:
        if (_c in drop) and (_c not in _TARGET_COLS) and (_c not in _NON_FEATURE_COLS):
            drop.discard(_c)

    if dbg_lambda:
        try:
            _chk = ["home_goals_pred", "away_goals_pred", "exp_goals_sum", "p00_est", "__og_row_id"]
            _present = {c: (c in work.columns) for c in _chk}
            _in_drop = {c: (c in drop) for c in _chk}
            print(f"[train_investor_leagues_v2] lambda pre_drop: present={_present} in_drop={_in_drop}")
        except Exception:
            pass

    work = work.drop(columns=[c for c in drop if c in work.columns], errors="ignore")

    if dbg_lambda:
        try:
            _chk = ["home_goals_pred", "away_goals_pred", "exp_goals_sum", "p00_est", "__og_row_id"]
            _present = {c: (c in work.columns) for c in _chk}
            print(f"[train_investor_leagues_v2] lambda post_drop: present={_present}")
        except Exception:
            pass

    # FINAL λ restore (after all drop/leak stages):
    # Always (re)merge from the preserved row-id table when possible, then coalesce.
    # This prevents home/away goal preds disappearing between post_restore and feature selection.
    if isinstance(_lambda_keep_df, pd.DataFrame) and (not work.empty) and ("__og_row_id" in work.columns):
        try:
            tmp = _lambda_keep_df[["__og_row_id"] + [c for c in _lambda_keep_cols if c in _lambda_keep_df.columns]].copy()
            work = work.merge(tmp, on="__og_row_id", how="left", suffixes=("", "_lamkeep2"))
            work = work.set_index("__og_row_id", drop=False).sort_index()

            for _c in _lambda_keep_cols:
                cu = f"{_c}_lamkeep2"
                if cu in work.columns:
                    if _c in work.columns:
                        a = pd.to_numeric(work[_c], errors="coerce")
                        b = pd.to_numeric(work[cu], errors="coerce")
                        work[_c] = a.where(a.notna(), b)
                    else:
                        work[_c] = pd.to_numeric(work[cu], errors="coerce")
                    work = work.drop(columns=[cu], errors="ignore")

            # Ensure numeric dtype for λ columns so they land in num_cols
            for _c in _lambda_keep_cols:
                if _c in work.columns:
                    work[_c] = pd.to_numeric(work[_c], errors="coerce")

        except Exception:
            pass

    elif _lambda_keep_by_key and (_lambda_key is not None) and (not work.empty):
        # Fallback: restore by stable match key mapping when row-id isn't available
        try:
            cur_key = _stable_lambda_key(work)

            base = pd.DataFrame({"__k": _lambda_key.astype("string")})
            for _c, _v in _lambda_keep_by_key.items():
                base[_c] = pd.to_numeric(_v, errors="coerce").to_numpy()

            base["__k"] = base["__k"].astype("string").fillna("").str.strip()
            base = base[base["__k"].ne("")].copy()
            base = base.drop_duplicates(subset=["__k"], keep="first").set_index("__k", drop=True)

            for _c in _lambda_keep_cols:
                if _c in base.columns:
                    mapped = cur_key.map(base[_c])
                    if _c in work.columns:
                        a = pd.to_numeric(work[_c], errors="coerce")
                        work[_c] = a.where(a.notna(), pd.to_numeric(mapped, errors="coerce"))
                    else:
                        work[_c] = pd.to_numeric(mapped, errors="coerce")
        except Exception:
            pass

    # Never allow internal row-id to become a model feature
    work = work.drop(columns=["__og_row_id"], errors="ignore")
    # Keep categorical IDs
    cat_keep: List[str] = []
    for c in ("home_team_name", "away_team_name", "League", "league", "league_name"):
        if c in work.columns:
            cat_keep.append(c)

    # Numeric/bool features
    num_cols = work.select_dtypes(include=["number", "bool"]).columns.tolist()

    # Force include key λ columns as numeric features if present.
    # (Some upstream transforms can leave them as object dtype even when values are numeric.)
    for _c in ("home_goals_pred", "away_goals_pred", "lambda_home", "lambda_away", "exp_goals_sum", "p00_est"):
        if _c in work.columns and _c not in num_cols:
            try:
                work[_c] = pd.to_numeric(work[_c], errors="coerce")
                num_cols.append(_c)
            except Exception:
                pass

    feats: List[str] = []
    feats.extend(cat_keep)
    for c in num_cols:
        if c not in feats:
            feats.append(c)

    # Drop all-NaN columns
    feats = [c for c in feats if c in work.columns and not work[c].isna().all()]

    # Debug: confirm λ cols survive into the feature list
    if dbg_lambda:
        try:
            lam_in = [c for c in ("home_goals_pred", "away_goals_pred", "exp_goals_sum", "p00_est") if c in work.columns]
            lam_in_feats = [c for c in lam_in if c in feats]
            lam_dtypes = {c: str(work[c].dtype) for c in lam_in}
            print(f"[train_investor_leagues_v2] lambda in_feats={lam_in_feats} dtypes={lam_dtypes}")
        except Exception:
            pass

    X = work[feats].copy()

    cat_idx: List[int] = []
    for i, c in enumerate(feats):
        if c in cat_keep or X[c].dtype == object or pd.api.types.is_string_dtype(X[c]):
            X[c] = X[c].astype("string").fillna("NA")
            cat_idx.append(i)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0).astype(np.float32)

    return X, feats, cat_idx


from typing import Tuple

def _best_threshold_f1(y_true: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    """Return (best_threshold, best_f1) on validation."""
    y_true = y_true.astype(int)
    best_thr = 0.5
    best_f1 = -1.0
    p = np.clip(p, 1e-9, 1 - 1e-9)

    for thr in np.linspace(0.05, 0.95, 91):
        y_hat = (p >= thr).astype(int)
        tp = int(((y_hat == 1) & (y_true == 1)).sum())
        fp = int(((y_hat == 1) & (y_true == 0)).sum())
        fn = int(((y_hat == 0) & (y_true == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = (2 * prec * rec) / max(prec + rec, 1e-12)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)

    return float(best_thr), float(best_f1)


def _best_threshold_balanced_accuracy(y_true: np.ndarray, p: np.ndarray) -> float:
    """Return threshold that maximises balanced accuracy on validation."""
    y_true = y_true.astype(int)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    best_thr = 0.5
    best_ba = -1.0

    for thr in np.linspace(0.05, 0.95, 91):
        y_hat = (p >= thr).astype(int)
        tp = int(((y_hat == 1) & (y_true == 1)).sum())
        fp = int(((y_hat == 1) & (y_true == 0)).sum())
        tn = int(((y_hat == 0) & (y_true == 0)).sum())
        fn = int(((y_hat == 0) & (y_true == 1)).sum())
        tpr = tp / max(tp + fn, 1)
        tnr = tn / max(tn + fp, 1)
        ba = 0.5 * (tpr + tnr)
        if ba > best_ba:
            best_ba = float(ba)
            best_thr = float(thr)

    return float(best_thr)


def _train_binary(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    cat_idx: List[int],
    seed: int,
    iters: int,
    threads: int,
) -> Tuple[CatBoostClassifier, Dict[str, float], float]:
    model = CatBoostClassifier(
        loss_function='Logloss',
        eval_metric='AUC',
        iterations=int(iters),
        learning_rate=0.05,
        depth=7,
        random_seed=int(seed),
        verbose=False,
        allow_writing_files=False,
        thread_count=int(max(1, int(threads))),
        od_type='Iter',
        od_wait=75,
    )

    model.fit(
        X_tr,
        y_tr,
        cat_features=cat_idx or None,
        eval_set=(X_va, y_va),
        use_best_model=True,
    )

    p_va = model.predict_proba(X_va)[:, 1]

    # If validation has only one class, thresholds are meaningless; use 0.5.
    yv = y_va.to_numpy(dtype=int)
    if np.unique(yv).size < 2:
        thr = 0.5
        method = "default_0.5"
    else:
        pos_rate = float(np.mean(yv))

        thr_f1, best_f1 = _best_threshold_f1(yv, p_va)
        # Evaluate the implied decision rate at the F1 threshold
        pred_rate_f1 = float((p_va >= thr_f1).mean())
        pred_rate_gap = abs(pred_rate_f1 - pos_rate)

        # If F1 is degenerate, threshold is extreme, OR decision-rate is wildly misaligned,
        # fall back to a balanced-accuracy threshold (more stable for deploy + banding).
        if (best_f1 <= 1e-9) or (thr_f1 < 0.10) or (thr_f1 > 0.90) or (pred_rate_gap > 0.25):
            thr = _best_threshold_balanced_accuracy(yv, p_va)
            method = "balanced_accuracy"
        else:
            thr = thr_f1
            method = "f1"

    y_hat = (p_va >= thr).astype(int)
    acc = float((y_hat == yv).mean())
    pos_rate = float(np.mean(yv))

    metrics = {
        'val_accuracy': acc,
        'val_pos_rate': pos_rate,
        'thr_method': method,
    }
    return model, metrics, float(thr)


def _train_binary_xgb(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    feats: List[str],
    cat_idx: List[int],
    seed: int,
    iters: int,
    threads: int,
    *,
    enable_categorical: bool,
) -> Tuple[Any, Dict[str, float], float, List[str]]:
    if _XGBClassifier is None:
        raise RuntimeError("xgboost not available")

    enable_cat = bool(enable_categorical)
    X_tr_xgb, feats_xgb = _prep_xgb_frame(X_tr, feats, cat_idx, enable_categorical=enable_cat)
    X_va_xgb, feats_xgb = _prep_xgb_frame(X_va, feats_xgb, cat_idx=[], enable_categorical=enable_cat)

    try:
        y_tr = pd.to_numeric(y_tr, errors="coerce").fillna(0).astype(int)
        y_va = pd.to_numeric(y_va, errors="coerce").fillna(0).astype(int)
    except Exception:
        pass

    try:
        X_tr_xgb = X_tr_xgb.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        X_va_xgb = X_va_xgb.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    except Exception:
        X_tr_xgb = np.asarray(X_tr_xgb).astype(np.float32, copy=False)
        X_va_xgb = np.asarray(X_va_xgb).astype(np.float32, copy=False)

    params = dict(
        n_estimators=int(iters),
        max_depth=6,
        learning_rate=0.05,
        subsample=0.80,
        colsample_bytree=0.80,
        min_child_weight=2.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        gamma=0.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=int(max(1, int(threads))),
        random_state=int(seed),
    )
    if enable_cat:
        params["enable_categorical"] = True

    model = _XGBClassifier(**params)
    model.fit(
        X_tr_xgb,
        y_tr,
        eval_set=[(X_va_xgb, y_va)],
        verbose=False,
    )

    p_va = np.asarray(model.predict_proba(X_va_xgb), dtype=float)[:, 1]
    yv = y_va.to_numpy(dtype=int)
    if np.unique(yv).size < 2:
        thr = 0.5
        method = "default_0.5"
    else:
        pos_rate = float(np.mean(yv))
        thr_f1, best_f1 = _best_threshold_f1(yv, p_va)
        pred_rate_f1 = float((p_va >= thr_f1).mean())
        pred_rate_gap = abs(pred_rate_f1 - pos_rate)
        if (best_f1 <= 1e-9) or (thr_f1 < 0.10) or (thr_f1 > 0.90) or (pred_rate_gap > 0.25):
            thr = _best_threshold_balanced_accuracy(yv, p_va)
            method = "balanced_accuracy"
        else:
            thr = thr_f1
            method = "f1"

    y_hat = (p_va >= thr).astype(int)
    acc = float((y_hat == yv).mean())
    pos_rate = float(np.mean(yv))
    metrics = {
        "val_accuracy": acc,
        "val_pos_rate": pos_rate,
        "thr_method": method,
        "xgb_enable_categorical": bool(enable_cat),
        "n_features": int(len(feats_xgb)),
    }
    return model, metrics, float(thr), feats_xgb


def _train_ftr(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    cat_idx: List[int],
    seed: int,
    iters: int,
    threads: int,
) -> Tuple[CatBoostClassifier, Dict[str, float]]:
    model = CatBoostClassifier(
        loss_function='MultiClass',
        eval_metric='Accuracy',
        iterations=int(iters),
        learning_rate=0.05,
        depth=8,
        random_seed=int(seed),
        verbose=False,
        allow_writing_files=False,
        thread_count=int(max(1, int(threads))),
        od_type='Iter',
        od_wait=75,
    )

    model.fit(
        X_tr,
        y_tr,
        cat_features=cat_idx or None,
        eval_set=(X_va, y_va),
        use_best_model=True,
    )

    proba = model.predict_proba(X_va)
    yhat = np.asarray(proba).argmax(axis=1)
    acc = float((yhat == y_va.to_numpy(dtype=int)).mean())

    metrics = {'val_accuracy': acc}
    return model, metrics


def _supports_xgb_categorical() -> bool:
    """Best-effort check for XGBoost categorical support."""
    if _XGBClassifier is None:
        return False
    try:
        _XGBClassifier(enable_categorical=True)
        return True
    except Exception:
        return False


def _prep_xgb_frame(
    X: pd.DataFrame,
    feats: List[str],
    cat_idx: List[int],
    *,
    enable_categorical: bool,
) -> Tuple[pd.DataFrame, List[str]]:
    """Prepare a feature frame for XGBoost.

    If enable_categorical is True, categorical columns are cast to pandas 'category'.
    Otherwise categorical columns are dropped.
    """
    feats = list(feats)
    cat_cols = [feats[i] for i in (cat_idx or []) if 0 <= int(i) < len(feats)]

    Xx = X.copy()

    if enable_categorical:
        # Explicitly cast known categorical columns
        if cat_cols:
            for c in cat_cols:
                if c in Xx.columns:
                    try:
                        Xx[c] = Xx[c].astype("category")
                    except Exception:
                        # If category conversion fails, fall back to string category
                        Xx[c] = Xx[c].astype("string").fillna("NA").astype("category")
        # Also convert any remaining string/object columns to categorical
        for c in list(Xx.columns):
            try:
                if pd.api.types.is_string_dtype(Xx[c]) or pd.api.types.is_object_dtype(Xx[c]):
                    Xx[c] = Xx[c].astype("string").fillna("NA").astype("category")
            except Exception:
                pass
    else:
        # Drop categorical columns when categorical support is unavailable
        if cat_cols:
            Xx = Xx.drop(columns=cat_cols, errors="ignore")
            feats = [c for c in feats if c not in set(cat_cols)]

    # Ensure numeric (or categorical) for remaining columns
    for c in list(Xx.columns):
        if pd.api.types.is_bool_dtype(Xx[c]):
            Xx[c] = Xx[c].astype(np.int8)
        elif pd.api.types.is_numeric_dtype(Xx[c]):
            Xx[c] = pd.to_numeric(Xx[c], errors="coerce").fillna(0.0).astype(np.float32)
        elif pd.api.types.is_categorical_dtype(Xx[c]):
            if not enable_categorical:
                # encode category as integer codes
                Xx[c] = Xx[c].cat.codes.astype(np.int32)
        elif pd.api.types.is_string_dtype(Xx[c]) or pd.api.types.is_object_dtype(Xx[c]):
            if enable_categorical:
                try:
                    Xx[c] = Xx[c].astype("string").fillna("NA").astype("category")
                except Exception:
                    Xx[c] = Xx[c].astype("category")
            else:
                # factorize into numeric codes
                try:
                    codes, _ = pd.factorize(Xx[c].astype("string").fillna("NA"))
                    Xx[c] = codes.astype(np.int32)
                except Exception:
                    Xx[c] = 0
        else:
            # Try numeric coercion first (handles mixed/object numeric columns)
            try:
                coerced = pd.to_numeric(Xx[c], errors="coerce")
                Xx[c] = coerced.fillna(0.0).astype(np.float32)
                continue
            except Exception:
                # Last-resort: zero-fill
                try:
                    Xx[c] = 0.0
                except Exception:
                    pass

    # Final cleanup: eliminate any lingering pandas string dtype
    for c in list(Xx.columns):
        try:
            if str(Xx[c].dtype).startswith("string"):
                if enable_categorical:
                    Xx[c] = Xx[c].astype("string").fillna("NA").astype("category")
                else:
                    codes, _ = pd.factorize(Xx[c].astype("string").fillna("NA"))
                    Xx[c] = codes.astype(np.int32)
        except Exception:
            # last-resort: hard zero-fill
            try:
                Xx[c] = 0.0
            except Exception:
                pass

    # Optional debug: report any residual string dtypes
    try:
        if os.getenv("OG_XGB_DEBUG_DTYPE", "0").strip().lower() in ("1", "true", "yes", "y"):
            bad = [c for c in Xx.columns if "string" in str(Xx[c].dtype)]
            if bad:
                print(f"[xgb-debug] lingering string dtypes: {bad}")
    except Exception:
        pass

    # Ensure columns order matches feats
    if feats:
        Xx = Xx.reindex(columns=feats)
    return Xx, feats


def _train_ftr_xgb(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    feats: List[str],
    cat_idx: List[int],
    seed: int,
    iters: int,
    threads: int,
    *,
    enable_categorical: bool,
) -> Tuple[Any, Dict[str, float], List[str]]:
    if _XGBClassifier is None:
        raise RuntimeError("xgboost not available")

    enable_cat = bool(enable_categorical)
    X_tr_xgb, feats_xgb = _prep_xgb_frame(X_tr, feats, cat_idx, enable_categorical=enable_cat)
    X_va_xgb, feats_xgb = _prep_xgb_frame(X_va, feats_xgb, cat_idx=[], enable_categorical=enable_cat)

    # Final sanitize: no string dtypes should remain
    def _sanitize_frame(Xf: pd.DataFrame) -> pd.DataFrame:
        Xf = Xf.copy()
        for c in list(Xf.columns):
            try:
                if str(Xf[c].dtype).startswith("string") or pd.api.types.is_object_dtype(Xf[c]):
                    if enable_cat:
                        Xf[c] = Xf[c].astype("string").fillna("NA").astype("category")
                    else:
                        codes, _ = pd.factorize(Xf[c].astype("string").fillna("NA"))
                        Xf[c] = codes.astype(np.int32)
            except Exception:
                pass
        return Xf

    X_tr_xgb = _sanitize_frame(X_tr_xgb)
    X_va_xgb = _sanitize_frame(X_va_xgb)

    # Ensure y is numeric int
    try:
        y_tr = pd.to_numeric(y_tr, errors="coerce").fillna(0).astype(int)
        y_va = pd.to_numeric(y_va, errors="coerce").fillna(0).astype(int)
    except Exception:
        pass

    # Force numeric numpy arrays to avoid pandas string dtype issues
    try:
        X_tr_xgb = X_tr_xgb.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        X_va_xgb = X_va_xgb.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    except Exception:
        # Fallback: best-effort coercion
        X_tr_xgb = np.asarray(X_tr_xgb).astype(np.float32, copy=False)
        X_va_xgb = np.asarray(X_va_xgb).astype(np.float32, copy=False)

    params = dict(
        n_estimators=int(iters),
        max_depth=6,
        learning_rate=0.05,
        subsample=0.80,
        colsample_bytree=0.80,
        min_child_weight=2.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        gamma=0.0,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=int(max(1, int(threads))),
        random_state=int(seed),
    )
    if enable_cat:
        params["enable_categorical"] = True

    model = _XGBClassifier(**params)
    model.fit(
        X_tr_xgb,
        y_tr,
        eval_set=[(X_va_xgb, y_va)],
        verbose=False,
    )

    proba = np.asarray(model.predict_proba(X_va_xgb))
    yhat = proba.argmax(axis=1)
    acc = float((yhat == y_va.to_numpy(dtype=int)).mean())

    metrics = {
        "val_accuracy": acc,
        "xgb_enable_categorical": bool(enable_cat),
        "n_features": int(len(feats_xgb)),
    }
    return model, metrics, feats_xgb


def _attach_alignment_attrs(model: Any, features: List[str], threshold: Optional[float] = None) -> None:
    """Make CatBoost behave with sklearn-ish alignment in your overlay."""
    try:
        model.feature_names_in_ = np.array(list(features))
    except Exception:
        pass
    try:
        model.n_features_in_ = int(len(features))
    except Exception:
        pass
    if threshold is not None:
        try:
            model.best_threshold_ = float(threshold)
        except Exception:
            pass
        try:
            model.opt_threshold_ = float(threshold)
        except Exception:
            pass


def _save_bundle(path: Path, bundle: Dict[str, Any]) -> None:
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)

def _save_bundle_subdir(outdir: Path, subdir: str, filename: str, bundle: Dict[str, Any]) -> None:
    """Optional subdir save for engine-separated bundles."""
    sub = str(subdir or "").strip()
    if not sub:
        return
    _save_bundle(outdir / sub / filename, bundle)

def _write_binary_bundle_variants(
    outdir: Path,
    stem: str,
    bundle: Dict[str, Any],
    mver: str,
    write_v2_compat: bool,
) -> None:
    """Write versioned binary market bundles and naming aliases.

    Writes:
      - <stem>_<mver>.pkl always
      - <stem>_v2.pkl if write_v2_compat and mver != v2
      - OU aliases:
          over25 -> ou25
          under25 -> u25
    """
    mver = str(mver).strip().lower()
    if mver not in {"v2", "v3"}:
        mver = "v3"

    # Primary versioned write
    _save_bundle(outdir / f"{stem}_{mver}.pkl", bundle)

    # Optional compat copy
    if write_v2_compat and mver != "v2":
        _save_bundle(outdir / f"{stem}_v2.pkl", bundle)

    # Aliases for OU naming
    if stem == "over25":
        _save_bundle(outdir / f"ou25_{mver}.pkl", bundle)
        if write_v2_compat and mver != "v2":
            _save_bundle(outdir / "ou25_v2.pkl", bundle)

    if stem == "under25":
        _save_bundle(outdir / f"u25_{mver}.pkl", bundle)
        if write_v2_compat and mver != "v2":
            _save_bundle(outdir / "u25_v2.pkl", bundle)

def _save_thresholds_json(path: Path, thresholds: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(thresholds, fh, indent=2)


def train_one_league(
    league: str,
    *,
    matches_df: pd.DataFrame,
    source_is_merged: bool = False,
    modelstore: Path,
    markets: Tuple[str, ...],
    val_frac: float,
    seed: int,
    iters: int,
    overwrite: bool,
    ftr_version: str = "v2",
    ftr_engine: str = "catboost",
    xgb_categorical: str = "auto",
    ftr_use_lambda: bool = False,
    market_version: str = "v3",
    binary_cat_markets: Optional[Tuple[str, ...]] = None,
    binary_xgb_markets: Tuple[str, ...] = (),
    write_v2_compat: bool = False,
    ftr_engine_subdir: bool = False,
    threads: int = 4,
    lite: bool = False,
) -> Dict[str, Any]:
    tag = _modelstore_tag(league)
    outdir = modelstore / tag

    df_raw = matches_df.copy()
    df_raw = _apply_renames(df_raw)
    df_raw = _coerce_numbers(df_raw)
    df_raw = _ensure_match_date(df_raw)

    df_done = _completed_only(df_raw)
    if df_done.empty:
        print(f"⚠️ {league}: no completed fixtures with goals after multi-season load — skipping")
        return {'league': league, 'skipped': True, 'reason': 'no_completed'}

    # Reset index ONCE (prevents holdout index/key errors and keeps splits stable)
    df_done = df_done.reset_index(drop=True).copy()
    source_cols_after_hygiene = set(map(str, df_done.columns))

    # --- HARD SAFETY: eliminate undated (NaT) rows before any split/training ---
    # Undated rows can create train/val contamination because the same fixture can appear
    # twice (one dated, one undated) and evade fixture_key de-dupe.
    # We first try to repair match_date via the overlay coalescer, then drop remaining NaT.
    try:
        md = pd.to_datetime(df_done.get("match_date"), errors="coerce")
    except Exception:
        md = pd.Series(pd.NaT, index=df_done.index)

    if bool(md.isna().any()):
        # Try to repair from other date-like columns using the overlay coalescer
        try:
            s_co = _coalesce_match_date_series(df_done)
            md2 = pd.to_datetime(s_co, errors="coerce", utc=True)
            try:
                md2 = md2.dt.tz_convert(None)
            except Exception:
                pass
            md = md.fillna(pd.to_datetime(md2, errors="coerce"))
        except Exception:
            pass

    nat = md.isna()
    if bool(nat.any()):
        try:
            print(f"🧹 {league}: dropping {int(nat.sum())} rows with NaT match_date (prevents train/val contamination)")
        except Exception:
            pass
        df_done = df_done.loc[~nat].copy()
        md = md.loc[~nat]

    # Ensure match_date is the repaired, non-NaT series
    try:
        df_done["match_date"] = pd.to_datetime(md, errors="coerce")
    except Exception:
        pass

    # Re-reset index ONCE more after dropping NaT rows (keeps split indices stable)
    df_done = df_done.reset_index(drop=True).copy()
    # --- Team ratings: strict pre-match rolling power (leak-safe) ---
    try:
        if callable(_build_rolling_power_ratings):
            df_done = _build_rolling_power_ratings(df_done)

            # Drop post-match debug columns (must never enter training features)
            df_done = df_done.drop(
                columns=[
                    "home_power_rating_post_raw",
                    "away_power_rating_post_raw",
                ],
                errors="ignore",
            )

            # Log coverage
            try:
                jr = float(pd.to_numeric(df_done.get("power_diff"), errors="coerce").notna().mean()) if len(df_done) else 0.0
                print(f"⚡ team_ratings[{league}]: power_join_rate={jr:.3f}")
            except Exception:
                pass
    except Exception as _e:
        print(f"⚠️ team_ratings[{league}]: rolling attach failed: {_e}")

    # --- Optional: leak-safe rolling/streak/H2H features (computed pre-match / shifted) ---
    try:
        if "match_date" in df_done.columns:
            md_sort = pd.to_datetime(df_done["match_date"], errors="coerce")
            if md_sort.notna().any():
                df_done = df_done.loc[md_sort.sort_values(kind="mergesort").index].copy()
    except Exception:
        pass

    if not bool(lite):
        try:
            if callable(_attach_streaks_and_h2h):
                if source_is_merged and _merged_already_has_streak_h2h_family(df_done):
                    try:
                        print(f"🧩 streaks[{league}]: merged source already has streak/H2H family — skipping duplicate attach")
                    except Exception:
                        pass
                else:
                    df_done = _attach_streaks_and_h2h(df_done)
                df_done = _collapse_xy_feature_suffixes(df_done)
        except Exception as _e:
            print(f"⚠️ streaks[{league}]: attach failed: {_e}")

    if not bool(lite):
        try:
            if callable(_build_team_rolling_metrics):
                # Some legacy rolling builders expect an integer-coded FTR column (0=H,1=D,2=A).
                # We derive it from final goals on df_done (already realised-only) and drop it after.
                if "FTR" not in df_done.columns:
                    hg = pd.to_numeric(df_done.get("home_team_goal_count"), errors="coerce")
                    ag = pd.to_numeric(df_done.get("away_team_goal_count"), errors="coerce")
                    df_done["FTR"] = np.where(hg > ag, 0, np.where(hg == ag, 1, 2)).astype(int)

                # build_team_rolling_metrics signature is (df, windows=(...)); do NOT pass league as positional arg.
                try:
                    df_done = _build_team_rolling_metrics(df_done)
                except TypeError:
                    # Some variants may require explicit windows
                    df_done = _build_team_rolling_metrics(df_done, windows=(5, 10))

                # Ensure the temporary outcome label never becomes a model feature.
                df_done = df_done.drop(columns=["FTR"], errors="ignore")
                # These are derived from the current match outcome in the legacy builder (leaky); never keep.
                df_done = df_done.drop(columns=["is_home_win", "is_away_win"], errors="ignore")
                df_done = _collapse_xy_feature_suffixes(df_done)
                if source_is_merged:
                    df_done = _prune_training_only_binary_features(
                        df_done,
                        source_cols=source_cols_after_hygiene,
                    )
        except Exception as _e:
            print(f"⚠️ rolling_metrics[{league}]: attach failed: {_e}")

    # --- FINAL SAFETY: ensure match_date remains valid after feature builders ---
    # Some builders (or merges) can accidentally blank/overwrite match_date.
    # Any NaT rows here can create train/val contamination or unstable boundaries.
    try:
        md_final = pd.to_datetime(df_done.get("match_date"), errors="coerce")
    except Exception:
        md_final = pd.Series(pd.NaT, index=df_done.index)

    if bool(md_final.isna().any()):
        # Try to repair from other date-like columns using the overlay coalescer
        try:
            s_co = _coalesce_match_date_series(df_done)
            md2 = pd.to_datetime(s_co, errors="coerce", utc=True)
            try:
                md2 = md2.dt.tz_convert(None)
            except Exception:
                pass
            md_final = md_final.fillna(pd.to_datetime(md2, errors="coerce"))
        except Exception:
            pass

    nat_final = md_final.isna()
    if bool(nat_final.any()):
        try:
            print(f"🧹 {league}: FINAL drop {int(nat_final.sum())} rows with NaT match_date (post-feature-build)")
        except Exception:
            pass
        df_done = df_done.loc[~nat_final].copy()
        md_final = md_final.loc[~nat_final]

    # Re-assign and re-index once to keep split indices stable
    try:
        df_done["match_date"] = pd.to_datetime(md_final, errors="coerce")
    except Exception:
        pass
    df_done = df_done.reset_index(drop=True).copy()

    # ------------------------------------------------------------------
    # Optional: λ (goal) feature attachment for FTR v3
    # - Best-effort: uses baseline goal ensemble scorer when available.
    # - Runs on a goal-blanked copy of realised df_done to prevent leakage.
    # ------------------------------------------------------------------
    use_lambda = bool(str(ftr_version).strip().lower() == "v3") or bool(ftr_use_lambda)
    # Log λ intent (helps catch runs where --ftr-version wasn't set)
    try:
        print(f"🧩 lambda_feats[{league}]: intent use_lambda={use_lambda} ftr_version={str(ftr_version).strip().lower()} ftr_use_lambda={bool(ftr_use_lambda)}")
    except Exception:
        pass
    lambda_feature_cols: List[str] = []
    uses_lambda_features = False
    lambda_source: str = "none"

    if use_lambda:
        try:
            before_cols = set(df_done.columns)

            # IMPORTANT: prevent goal-engine leakage during training.
            # Some goal helper paths can (directly or indirectly) use realised goal columns.
            # That makes FTR trivially learnable (near-perfect val acc). We therefore:
            #   1) build λ on a copy with goal columns blanked
            #   2) merge λ back by a stable row id
            # This matches inference-time reality (future fixtures have no realised goals).

            df_tmp = df_done.copy()
            if "__og_row_id" not in df_tmp.columns:
                df_tmp["__og_row_id"] = np.arange(len(df_tmp), dtype=np.int64)

            # Blank realised goal columns so the λ generator cannot copy/derive truth.
            for gc in ("home_team_goal_count", "away_team_goal_count", "total_goal_count"):
                if gc in df_tmp.columns:
                    df_tmp[gc] = np.nan

            # Strip leak-prone post-match columns BEFORE λ generation.
            # Otherwise the goal engine can implicitly use post-match stats (xG/shots/cards/etc)
            # and produce near-perfect FTR signals on realised data.
            if callable(_strip_leaks):
                try:
                    _rid = df_tmp["__og_row_id"].copy()
                    df_tmp = _strip_leaks(df_tmp)
                    # Ensure row id survives for the merge-back step.
                    if "__og_row_id" not in df_tmp.columns:
                        df_tmp["__og_row_id"] = _rid.to_numpy()
                except Exception:
                    pass

            # --- Leak-safe year-capped goal ensembles (preferred) ---
            # If holdout is 2025, we must cap goal models to max_year=2024, etc.
            holdout_year = _infer_holdout_year_for_lambda(df_done)
            max_year = (int(holdout_year) - 1) if (holdout_year is not None and int(holdout_year) >= 1) else None
            try:
                print(f"🧩 lambda_feats[{league}]: cap holdout_year={holdout_year} -> max_year={max_year}")
            except Exception:
                pass

            # Try local ensemble scoring first so we control max_year selection.
            try:
                h_models = _try_load_goal_ensemble(modelstore, league, "home", max_year=max_year)
                a_models = _try_load_goal_ensemble(modelstore, league, "away", max_year=max_year)
                if h_models is not None and a_models is not None:
                    df_tmp["home_goals_pred"] = _score_goal_ensemble(df_tmp, h_models).clip(lower=0.0)
                    df_tmp["away_goals_pred"] = _score_goal_ensemble(df_tmp, a_models).clip(lower=0.0)
                    if max_year is not None:
                        lambda_source = "ensemble_year_capped"
                    else:
                        lambda_source = "ensemble_unversioned"
            except Exception:
                pass

            # Preferred path: per-league goal ensemble scorer
            if callable(_add_goal_predictions_to_complete_data):
                out2 = None
                try:
                    out2 = _add_goal_predictions_to_complete_data(df_tmp, league, required_features=[])
                except TypeError:
                    try:
                        out2 = _add_goal_predictions_to_complete_data(df_tmp, league, [])
                    except TypeError:
                        try:
                            out2 = _add_goal_predictions_to_complete_data(df_tmp, league_name=league, required_features=[])
                        except Exception:
                            out2 = None
                if isinstance(out2, pd.DataFrame) and (not out2.empty):
                    df_tmp = out2
                    if lambda_source == "none":
                        lambda_source = "baseline_helper"

            # Fallback path: attach_lambda_features_inplace (may persist ridge models)
            # Only call fallback if we still don't have goal preds.
            need_h = ("home_goals_pred" not in df_tmp.columns) or pd.to_numeric(df_tmp.get("home_goals_pred"), errors="coerce").isna().all()
            need_a = ("away_goals_pred" not in df_tmp.columns) or pd.to_numeric(df_tmp.get("away_goals_pred"), errors="coerce").isna().all()
            if (need_h or need_a) and callable(_attach_lambda_features_inplace):
                out3 = None
                try:
                    out3 = _attach_lambda_features_inplace(df_tmp, league, holdout_year=holdout_year)
                except TypeError:
                    try:
                        out3 = _attach_lambda_features_inplace(df_tmp, league_name=league)
                    except Exception:
                        out3 = None
                if isinstance(out3, pd.DataFrame) and (not out3.empty):
                    df_tmp = out3
                    if lambda_source == "none":
                        lambda_source = "baseline_helper"

            # Last-resort fallback: seed λ from pre-match xG (safe, always available)
            if ("home_goals_pred" not in df_tmp.columns) or pd.to_numeric(df_tmp.get("home_goals_pred"), errors="coerce").isna().all():
                hxg = pd.to_numeric(df_tmp.get("pre_match_xg_home", df_tmp.get("Home Team Pre-Match xG", np.nan)), errors="coerce")
                df_tmp["home_goals_pred"] = hxg.clip(lower=0.0).fillna(1.2)
            if ("away_goals_pred" not in df_tmp.columns) or pd.to_numeric(df_tmp.get("away_goals_pred"), errors="coerce").isna().all():
                axg = pd.to_numeric(df_tmp.get("pre_match_xg_away", df_tmp.get("Away Team Pre-Match xG", np.nan)), errors="coerce")
                df_tmp["away_goals_pred"] = axg.clip(lower=0.0).fillna(1.0)
            if lambda_source == "none":
                lambda_source = "xg_fallback"

            # Normalize common λ columns if present
            if "lambda_home" in df_tmp.columns and (("home_goals_pred" not in df_tmp.columns) or pd.to_numeric(df_tmp.get("home_goals_pred"), errors="coerce").isna().all()):
                df_tmp["home_goals_pred"] = pd.to_numeric(df_tmp["lambda_home"], errors="coerce")
            if "lambda_away" in df_tmp.columns and (("away_goals_pred" not in df_tmp.columns) or pd.to_numeric(df_tmp.get("away_goals_pred"), errors="coerce").isna().all()):
                df_tmp["away_goals_pred"] = pd.to_numeric(df_tmp["lambda_away"], errors="coerce")

            # Ensure exp_goals_sum / p00_est
            if "exp_goals_sum" not in df_tmp.columns:
                df_tmp["exp_goals_sum"] = pd.to_numeric(df_tmp["home_goals_pred"], errors="coerce") + pd.to_numeric(df_tmp["away_goals_pred"], errors="coerce")
            if "p00_est" not in df_tmp.columns:
                df_tmp["p00_est"] = np.exp(-pd.to_numeric(df_tmp["exp_goals_sum"], errors="coerce").clip(lower=0.0))

            # Pull λ cols back onto df_done by stable row id
            if "__og_row_id" not in df_done.columns:
                df_done["__og_row_id"] = np.arange(len(df_done), dtype=np.int64)

            keep_cols = ["__og_row_id", "home_goals_pred", "away_goals_pred", "exp_goals_sum", "p00_est", "lambda_home", "lambda_away"]
            keep_cols = [c for c in keep_cols if c in df_tmp.columns]
            lam_keep = df_tmp[keep_cols].copy()

            df_done = df_done.merge(lam_keep, on="__og_row_id", how="left", suffixes=("", "_lam"))
            for c in ("home_goals_pred", "away_goals_pred", "exp_goals_sum", "p00_est", "lambda_home", "lambda_away"):
                cu = c + "_lam"
                if cu in df_done.columns:
                    if c in df_done.columns:
                        a = pd.to_numeric(df_done[c], errors="coerce")
                        b = pd.to_numeric(df_done[cu], errors="coerce")
                        df_done[c] = a.where(a.notna(), b)
                    else:
                        df_done[c] = pd.to_numeric(df_done[cu], errors="coerce")
                    df_done = df_done.drop(columns=[cu], errors="ignore")

            # Clean internal row id
            df_done = df_done.drop(columns=["__og_row_id"], errors="ignore")

            # Track which λ cols exist and are usable
            cand_cols = ["home_goals_pred", "away_goals_pred", "lambda_home", "lambda_away", "exp_goals_sum", "p00_est"]
            lambda_feature_cols = [c for c in cand_cols if c in df_done.columns]

            h0 = pd.to_numeric(df_done.get("home_goals_pred", np.nan), errors="coerce")
            a0 = pd.to_numeric(df_done.get("away_goals_pred", np.nan), errors="coerce")
            uses_lambda_features = bool(h0.notna().any()) and bool(a0.notna().any())

            added_cols = sorted(list(set(df_done.columns) - before_cols))
            if added_cols:
                print(f"🧩 lambda_feats[{league}]: added_cols={added_cols[:12]}{'...' if len(added_cols)>12 else ''}")
            print(f"🧩 lambda_feats[{league}]: uses_lambda_features={uses_lambda_features} cols={lambda_feature_cols}")
        except Exception as _e:
            import traceback
            try:
                print(f"⚠️ lambda_feats[{league}]: attach failed: {_e}")
                print("".join(traceback.format_exc(limit=8)))
            except Exception:
                pass
            lambda_feature_cols = []
            uses_lambda_features = False

    y_btts, y_over, y_under, y_ftr = _build_targets(df_done)
    # Build one canonical feature frame for binary markets + calibration
    X_all_base, feats_base, cat_idx_base = _make_feature_frame(df_done)

    # --- Feature manifest (critical for upstream scoring / audits) ---
    # We persist the exact feature list + categorical feature names so we can later
    # verify "new tricks" are included (power_ratings, rolling metrics, lambda cols, etc.)
    try:
        cat_feats = []
        try:
            cat_feats = [str(feats_base[i]) for i in (cat_idx_base or []) if 0 <= int(i) < len(feats_base)]
        except Exception:
            cat_feats = []

        manifest = {
            "league": str(league),
            "tag": str(tag),
            "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "lite": bool(lite),
            "uses_lambda_features": bool(uses_lambda_features),
            "lambda_source": str(lambda_source) if "lambda_source" in locals() else "none",
            "lambda_feature_cols": list(lambda_feature_cols or []),
            "n_features": int(len(feats_base)),
            "n_cat_features": int(len(cat_feats)),
            "cat_features": cat_feats,
            "features": list(map(str, feats_base)),
        }

        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "features_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        # Also print a compact preview for fast sanity checks in logs
        try:
            preview = [f for f in (lambda_feature_cols or []) if f in feats_base]
            if not preview:
                preview = [f for f in ("power_diff", "home_power_rating", "away_power_rating", "exp_goals_sum", "p00_est") if f in feats_base]
            print(f"🧾 features[{league}]: n={len(feats_base)} cat={len(cat_feats)} lambda={bool(uses_lambda_features)} preview={preview[:8]}")
        except Exception:
            pass
    except Exception:
        pass

    tr_idx, va_idx, split_info = _compute_holdout_split(df_done, val_frac=val_frac)
    try:
        # show where data came from (multi-season loader stamps __src_csv)
        n_src = int(df_raw.get("__src_csv").nunique()) if "__src_csv" in df_raw.columns else None
        src_msg = f" | src_csvs={n_src}" if n_src is not None else ""
        if split_info.get("method") == "season":
            print(
                f"🧪 holdout[{league}] method=season recent_year={split_info.get('recent_year')} "
                f"years={split_info.get('season_years')} n_train={split_info.get('n_train')} n_val={split_info.get('n_val')}" + src_msg
            )
        elif split_info.get("method") == "date":
            print(
                f"🧪 holdout[{league}] method=date boundary={split_info.get('boundary')} "
                f"n_train={split_info.get('n_train')} n_val={split_info.get('n_val')}" + src_msg
            )
        else:
            print(
                f"🧪 holdout[{league}] method=index n_train={split_info.get('n_train')} n_val={split_info.get('n_val')}" + src_msg
            )
    except Exception:
        pass
    X_tr = X_all_base.loc[tr_idx]
    X_va = X_all_base.loc[va_idx]

    out: Dict[str, Any] = {
        'league': league,
        'tag': tag,
        'n_total': int(len(df_done)),
        'n_train': int(len(tr_idx)),
        'n_val': int(len(va_idx)),
    }

    trained_models: Dict[str, Any] = {}

    binary_cat_markets = None if binary_cat_markets is None else tuple(str(m).strip().lower() for m in (binary_cat_markets or ()))
    binary_xgb_markets = tuple(str(m).strip().lower() for m in (binary_xgb_markets or ()))

    def _should_train_binary_cat(market: str) -> bool:
        if binary_cat_markets is None:
            return True
        return str(market).strip().lower() in set(binary_cat_markets)

    if 'btts' in markets:
        mver = str(market_version).strip().lower() if market_version else "v3"
        if mver not in ("v2", "v3"):
            mver = "v3"
        X_tr_btts, feats_btts, cat_idx_btts = X_tr, feats_base, cat_idx_base
        X_va_btts = X_va
        p_primary = outdir / f"btts_{mver}.pkl"
        if _should_train_binary_cat("btts") and (overwrite or (not p_primary.exists())):
            mdl, met, thr = _train_binary(X_tr_btts, y_btts.loc[tr_idx], X_va_btts, y_btts.loc[va_idx], cat_idx_btts, seed, iters, int(threads))
            trained_models["btts"] = mdl
            _attach_alignment_attrs(mdl, feats_btts, threshold=thr)
            bundle = {
                'model': mdl,
                'features': feats_btts,
                'league': league,
                'val_accuracy': met.get('val_accuracy'),
                'val_accuracy_trainer': met.get('val_accuracy'),
                'val_pos_rate': met.get('val_pos_rate'),
                'thr_method': met.get('thr_method'),
                'threshold': float(thr),
                'n_train': int(len(tr_idx)),
                'n_val': int(len(va_idx)),
                'created_utc': _dt.datetime.now(_dt.timezone.utc).isoformat(),
            }
            _write_binary_bundle_variants(outdir, "btts", bundle, mver, bool(write_v2_compat))
            print(f"✅ Saved BTTS {mver} → {outdir}")
            out['btts_thr'] = float(thr)

        if "btts" in binary_xgb_markets:
            p_xgb_primary = outdir / "xgb" / f"btts_{mver}.pkl"
            if overwrite or (not p_xgb_primary.exists()):
                xgb_cat_policy = str(xgb_categorical or "auto").strip().lower()
                enable_cat = False
                if xgb_cat_policy == "true":
                    enable_cat = True
                elif xgb_cat_policy == "false":
                    enable_cat = False
                else:
                    enable_cat = _supports_xgb_categorical()
                if enable_cat and not _supports_xgb_categorical():
                    print("⚠️ BTTS xgboost categorical requested but not supported; disabling.")
                    enable_cat = False
                try:
                    mdl_xgb, met_xgb, thr_xgb, feats_xgb = _train_binary_xgb(
                        X_tr_btts,
                        y_btts.loc[tr_idx],
                        X_va_btts,
                        y_btts.loc[va_idx],
                        feats_btts,
                        cat_idx_btts,
                        seed,
                        iters,
                        int(threads),
                        enable_categorical=enable_cat,
                    )
                    _attach_alignment_attrs(mdl_xgb, feats_xgb, threshold=thr_xgb)
                    try:
                        cat_cols = [feats_btts[i] for i in (cat_idx_btts or []) if 0 <= int(i) < len(feats_btts)]
                        cat_cols = [c for c in cat_cols if c in feats_xgb]
                    except Exception:
                        cat_cols = []
                    bundle_xgb = {
                        'model': mdl_xgb,
                        'features': feats_xgb,
                        'league': league,
                        'val_accuracy': met_xgb.get('val_accuracy'),
                        'val_accuracy_trainer': met_xgb.get('val_accuracy'),
                        'val_pos_rate': met_xgb.get('val_pos_rate'),
                        'thr_method': met_xgb.get('thr_method'),
                        'threshold': float(thr_xgb),
                        'n_train': int(len(tr_idx)),
                        'n_val': int(len(va_idx)),
                        'created_utc': _dt.datetime.now(_dt.timezone.utc).isoformat(),
                        'engine': 'xgboost',
                        'xgb_enable_categorical': bool(enable_cat),
                        'xgb_cat_features': list(map(str, cat_cols)),
                    }
                    _save_bundle(p_xgb_primary, bundle_xgb)
                    _save_bundle(outdir / f"btts_{mver}_xgb.pkl", bundle_xgb)
                    if write_v2_compat and mver != "v2":
                        _save_bundle(outdir / "btts_v2_xgb.pkl", bundle_xgb)
                    print(f"✅ Saved BTTS {mver} (xgboost) → {p_xgb_primary}")
                    out['btts_xgb_thr'] = float(thr_xgb)
                except Exception as _e:
                    print(f"⚠️ BTTS xgboost train failed: {_e}")

    if 'over25' in markets:
        mver = str(market_version).strip().lower() if market_version else "v3"
        if mver not in ("v2", "v3"):
            mver = "v3"
        p_primary = outdir / f"over25_{mver}.pkl"
        if _should_train_binary_cat("over25") and (overwrite or (not p_primary.exists())):
            mdl, met, thr = _train_binary(X_tr, y_over.loc[tr_idx], X_va, y_over.loc[va_idx], cat_idx_base, seed, iters, int(threads))
            trained_models["over25"] = mdl
            _attach_alignment_attrs(mdl, feats_base, threshold=thr)
            bundle = {
                'model': mdl,
                'features': feats_base,
                'league': league,
                'val_accuracy': met.get('val_accuracy'),
                'val_pos_rate': met.get('val_pos_rate'),
                'thr_method': met.get('thr_method'),
                'threshold': float(thr),
                'n_train': int(len(tr_idx)),
                'n_val': int(len(va_idx)),
                'created_utc': _dt.datetime.now(_dt.timezone.utc).isoformat(),
            }
            _write_binary_bundle_variants(outdir, "over25", bundle, mver, bool(write_v2_compat))
            print(f"✅ Saved OVER25 {mver} → {outdir}")
            out['over25_thr'] = float(thr)
        if "over25" in binary_xgb_markets:
            p_xgb_primary = outdir / "xgb" / f"over25_{mver}.pkl"
            if overwrite or (not p_xgb_primary.exists()):
                xgb_cat_policy = str(xgb_categorical or "auto").strip().lower()
                enable_cat = False
                if xgb_cat_policy == "true":
                    enable_cat = True
                elif xgb_cat_policy == "false":
                    enable_cat = False
                else:
                    enable_cat = _supports_xgb_categorical()
                if enable_cat and not _supports_xgb_categorical():
                    print("⚠️ OU25 xgboost categorical requested but not supported; disabling.")
                    enable_cat = False
                try:
                    mdl_xgb, met_xgb, thr_xgb, feats_xgb = _train_binary_xgb(
                        X_tr,
                        y_over.loc[tr_idx],
                        X_va,
                        y_over.loc[va_idx],
                        feats_base,
                        cat_idx_base,
                        seed,
                        iters,
                        int(threads),
                        enable_categorical=enable_cat,
                    )
                    _attach_alignment_attrs(mdl_xgb, feats_xgb, threshold=thr_xgb)
                    cat_cols = []
                    if enable_cat:
                        cat_cols = [feats_base[idx] for idx in cat_idx_base if idx < len(feats_base)]
                        cat_cols = [c for c in cat_cols if c in feats_xgb]
                    bundle_xgb = {
                        'model': mdl_xgb,
                        'features': feats_xgb,
                        'league': league,
                        'val_accuracy': met_xgb.get('val_accuracy'),
                        'val_accuracy_trainer': met_xgb.get('val_accuracy'),
                        'val_pos_rate': met_xgb.get('val_pos_rate'),
                        'thr_method': met_xgb.get('thr_method'),
                        'threshold': float(thr_xgb),
                        'n_train': int(len(tr_idx)),
                        'n_val': int(len(va_idx)),
                        'created_utc': _dt.datetime.now(_dt.timezone.utc).isoformat(),
                        'engine': 'xgboost',
                        'xgb_enable_categorical': bool(enable_cat),
                        'xgb_cat_features': list(map(str, cat_cols)),
                    }
                    _save_bundle(p_xgb_primary, bundle_xgb)
                    _save_bundle(outdir / f"over25_{mver}_xgb.pkl", bundle_xgb)
                    _save_bundle(outdir / f"ou25_{mver}_xgb.pkl", bundle_xgb)
                    if bool(write_v2_compat):
                        _save_bundle(outdir / "over25_v2_xgb.pkl", bundle_xgb)
                        _save_bundle(outdir / "ou25_v2_xgb.pkl", bundle_xgb)
                    print(f"✅ Saved OVER25 {mver} (xgboost) → {p_xgb_primary}")
                    out['over25_xgb_thr'] = float(thr_xgb)
                except Exception as _e:
                    print(f"⚠️ OU25 xgboost train failed: {_e}")

    if 'under25' in markets:
        mver = str(market_version).strip().lower() if market_version else "v3"
        if mver not in ("v2", "v3"):
            mver = "v3"
        p_primary = outdir / f"under25_{mver}.pkl"
        if _should_train_binary_cat("under25") and (overwrite or (not p_primary.exists())):
            mdl, met, thr = _train_binary(
                X_tr, y_under.loc[tr_idx],
                X_va, y_under.loc[va_idx],
                cat_idx_base, seed, iters, int(threads)
            )
            trained_models["under25"] = mdl
            _attach_alignment_attrs(mdl, feats_base, threshold=thr)
            bundle = {
                'model': mdl,
                'features': feats_base,
                'league': league,
                'val_accuracy': met.get('val_accuracy'),
                'val_pos_rate': met.get('val_pos_rate'),
                'thr_method': met.get('thr_method'),
                'threshold': float(thr),
                'n_train': int(len(tr_idx)),
                'n_val': int(len(va_idx)),
                'created_utc': _dt.datetime.now(_dt.timezone.utc).isoformat(),
            }
            _write_binary_bundle_variants(outdir, "under25", bundle, mver, bool(write_v2_compat))
            print(f"✅ Saved UNDER25 {mver} → {outdir}")
            out['under25_thr'] = float(thr)
        else:
            # Not retraining, but we still want the dedicated under25 model available
            # so calibrator fitting prefers it (better in low-scoring leagues).
            try:
                import joblib
                existing = joblib.load(str(p_primary))
                if isinstance(existing, dict) and "model" in existing:
                    trained_models["under25"] = existing["model"]
            except Exception:
                pass

    if 'ftr' in markets:
        # Versioning:
        #   - v2: existing behavior -> ftr_v2.pkl
        #   - v3: λ-enabled feature set (when available) -> ftr_v3.pkl
        fver = str(ftr_version).strip().lower() if ftr_version else "v2"
        if fver not in ("v2", "v3"):
            fver = "v2"

        fname = "ftr_v3.pkl" if fver == "v3" else "ftr_v2.pkl"
        p_primary = outdir / fname

        # FTR feature frame:
        # - v2: reuse the canonical base frame used by binary markets + calibration
        # - v3: rebuild (in case future policy diverges / λ gating differs)
        if fver == "v3":
            X_all_ftr, feats_ftr, cat_idx_ftr = _make_feature_frame(df_done)
            X_tr_ftr = X_all_ftr.loc[tr_idx]
            X_va_ftr = X_all_ftr.loc[va_idx]
        else:
            X_all_ftr = None
            feats_ftr = feats_base
            cat_idx_ftr = cat_idx_base
            X_tr_ftr = X_tr
            X_va_ftr = X_va

        engine = str(ftr_engine or "catboost").strip().lower()
        if engine not in ("catboost", "xgboost", "both"):
            engine = "catboost"

        def _build_ftr_bundle(mdl: Any, feats_used: List[str], engine_name: str, met: Dict[str, Any]) -> Dict[str, Any]:
            # Recompute validation accuracy explicitly on the TRUE val slice with the TRUE feature order
            if engine_name == "xgboost":
                X_va_use, feats_used = _prep_xgb_frame(X_va_ftr, feats_used, cat_idx=[], enable_categorical=False)
                proba_va = np.asarray(mdl.predict_proba(X_va_use), dtype=float)
            else:
                X_va_use = X_va_ftr[feats_used] if feats_used else X_va_ftr
                proba_va = np.asarray(mdl.predict_proba(X_va_use), dtype=float)
            yhat_va = np.asarray(proba_va).argmax(axis=1)
            ytrue_va = y_ftr.loc[va_idx].to_numpy(dtype=int)
            val_acc_recomputed = float((yhat_va == ytrue_va).mean())

            # Track class balance in the validation slice
            try:
                _vc = pd.Series(y_ftr.loc[va_idx]).value_counts().sort_index()
                val_class_counts = {int(k): int(v) for k, v in _vc.to_dict().items()}
            except Exception:
                val_class_counts = {}

            _attach_alignment_attrs(mdl, feats_used, threshold=None)

            bundle = {
                'model': mdl,
                'features': feats_used,
                'league': league,
                'val_accuracy': float(val_acc_recomputed),
                'split_info': split_info,
                'val_class_counts': val_class_counts,
                'lambda_feature_cols_in_features': [c for c in list(lambda_feature_cols) if c in list(feats_used)],
                'n_train': int(len(tr_idx)),
                'n_val': int(len(va_idx)),
                'created_utc': _dt.datetime.now(_dt.timezone.utc).isoformat(),
                # v3 metadata (harmless for v2; always included)
                'ftr_version': ("v3" if fver == "v3" else "v2"),
                'uses_lambda_features': bool(uses_lambda_features) if 'uses_lambda_features' in locals() else False,
                'lambda_feature_cols': list(lambda_feature_cols) if 'lambda_feature_cols' in locals() else [],
                'lambda_source': str(lambda_source) if 'lambda_source' in locals() else "none",
                'engine': str(engine_name),
            }
            # merge any engine-specific metrics
            try:
                for k, v in (met or {}).items():
                    if k not in bundle:
                        bundle[k] = v
            except Exception:
                pass
            return bundle

        if overwrite or (not p_primary.exists()):
            # Primary engine write
            if engine in ("catboost", "both"):
                mdl, met = _train_ftr(
                    X_tr_ftr,
                    y_ftr.loc[tr_idx],
                    X_va_ftr,
                    y_ftr.loc[va_idx],
                    cat_idx_ftr,
                    seed,
                    iters,
                    int(threads),
                )
                bundle = _build_ftr_bundle(mdl, feats_ftr, "catboost", met)
                _save_bundle(p_primary, bundle)
                if ftr_engine_subdir:
                    _save_bundle_subdir(outdir, "cat", fname, bundle)
                print(f"✅ Saved FTR {bundle['ftr_version']} (catboost) → {p_primary}")

            if engine in ("xgboost", "both"):
                try:
                    # Resolve categorical policy
                    xgb_cat_policy = str(xgb_categorical or "auto").strip().lower()
                    enable_cat = False
                    if xgb_cat_policy == "true":
                        enable_cat = True
                    elif xgb_cat_policy == "false":
                        enable_cat = False
                    else:
                        # auto: enable only if supported
                        enable_cat = _supports_xgb_categorical()

                    if enable_cat and not _supports_xgb_categorical():
                        print("⚠️ xgboost categorical requested but not supported by installed xgboost; disabling.")
                        enable_cat = False

                    mdl_xgb, met_xgb, feats_xgb = _train_ftr_xgb(
                        X_tr_ftr,
                        y_ftr.loc[tr_idx],
                        X_va_ftr,
                        y_ftr.loc[va_idx],
                        feats_ftr,
                        cat_idx_ftr,
                        seed,
                        iters,
                        int(threads),
                        enable_categorical=enable_cat,
                    )
                    bundle_xgb = _build_ftr_bundle(mdl_xgb, feats_xgb, "xgboost", met_xgb)
                    # Record categorical metadata for inference casting
                    try:
                        cat_cols = [feats_ftr[i] for i in (cat_idx_ftr or []) if 0 <= int(i) < len(feats_ftr)]
                        cat_cols = [c for c in cat_cols if c in feats_xgb]
                        bundle_xgb["xgb_enable_categorical"] = bool(enable_cat)
                        bundle_xgb["xgb_cat_features"] = list(map(str, cat_cols))
                    except Exception:
                        bundle_xgb["xgb_enable_categorical"] = bool(enable_cat)
                        bundle_xgb["xgb_cat_features"] = []
                    # If xgboost is primary, write to the canonical filename; else write an alt.
                    if engine == "xgboost":
                        _save_bundle(p_primary, bundle_xgb)
                        if ftr_engine_subdir:
                            _save_bundle_subdir(outdir, "xgb", fname, bundle_xgb)
                        print(f"✅ Saved FTR {bundle_xgb['ftr_version']} (xgboost) → {p_primary}")
                    else:
                        alt = outdir / (fname.replace(".pkl", "_xgb.pkl"))
                        _save_bundle(alt, bundle_xgb)
                        if ftr_engine_subdir:
                            _save_bundle_subdir(outdir, "xgb", fname, bundle_xgb)
                        print(f"✅ Saved FTR {bundle_xgb['ftr_version']} (xgboost) → {alt}")
                except Exception as _e:
                    print(f"⚠️ FTR xgboost train failed: {_e}")

            try:
                try:
                    if X_all_ftr is not None:
                        del X_all_ftr
                    del X_tr_ftr, X_va_ftr
                except Exception:
                    pass
            except Exception:
                pass
            try:
                gc.collect()
            except Exception:
                pass

    thr_map: Dict[str, float] = {}
    if 'btts_thr' in out:
        thr_map['btts'] = float(out['btts_thr'])
        _inv = 1.0 - float(out['btts_thr'])
        thr_map['btts_no'] = float(np.clip(_inv, 0.05, 0.95))

    if 'over25_thr' in out:
        thr_map['over25'] = float(out['over25_thr'])
        if 'under25_thr' not in out:
            thr_map['under25'] = float(np.clip(1.0 - float(out['over25_thr']), 0.05, 0.95))

    if 'under25_thr' in out:
        thr_map['under25'] = float(out['under25_thr'])

    thr_map.setdefault('ftr', 0.40)

    try:
        thr_map["lambda_source"] = str(lambda_source) if "lambda_source" in locals() else "none"
    except Exception:
        pass

    thr_path = outdir / "market_thresholds.json"
    _save_thresholds_json(thr_path, thr_map)
    print(f"✅ Saved thresholds → {thr_path}")

    # --- Calibration pass (fit isotonic calibrators on realised holdout slice) ---
    if not bool(lite):
        try:
            y_btts_va = y_btts.loc[va_idx] if 'btts' in markets else None
            y_over_va = y_over.loc[va_idx] if 'over25' in markets else None
            y_under_va = y_under.loc[va_idx] if ('over25' in markets or 'under25' in markets) else None
            cal_info = _fit_and_save_calibrators_for_league(
                league,
                modelstore,
                models=trained_models,
                X_va=X_va,
                y_btts_va=y_btts_va,
                y_over_va=y_over_va,
                y_under_va=y_under_va,
            )
            out['calibrators'] = cal_info
            if cal_info.get('markets_fitted'):
                print(f"🧪 Calibrators fitted for {league}: {cal_info.get('markets_fitted')}")
        except Exception as _e:
            # best-effort: training should not fail if calibration libraries aren't available
            out['calibrators'] = {'error': str(_e)}
    else:
        out['calibrators'] = {'skipped': True, 'reason': 'lite_mode'}

    out['thresholds_path'] = str(thr_path)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description='Batch train V2 models for investor leagues')
    ap.add_argument('--lite', action='store_true', help='Lite mode: skip heavy feature builders (streaks/rolling metrics) + skip calibration to reduce RAM.')
    ap.add_argument('--leagues', default=None, help='Comma-separated leagues (e.g. "Italy Serie A,Spain La Liga")')
    ap.add_argument('--matches-root', default='Matches', help='Root matches folder')
    ap.add_argument('--use-merged', action='store_true', help='Load per-league merged CSVs from Matches/__merged__/<LEAGUE_TAG>__merged.csv instead of scanning Matches/<League>/ folder CSVs.')
    ap.add_argument('--merged-root', default='Matches/__merged__', help='Root folder containing merged per-league CSVs (default: Matches/__merged__).')
    ap.add_argument('--modelstore', default='ModelStore', help='ModelStore output folder')
    ap.add_argument('--markets', default='btts,over25,under25,ftr', help='Comma list: btts,over25,under25,ftr')
    ap.add_argument('--ftr-version', default='v2', choices=['v2','v3'], help='FTR bundle version: v2 (default) or v3 (λ-enabled).')
    ap.add_argument('--ftr-engine', default='catboost', choices=['catboost','xgboost','both'], help='FTR engine: catboost (default), xgboost, or both (write _xgb alt).')
    ap.add_argument('--xgb-categorical', default='auto', choices=['auto','true','false'], help='XGBoost categorical handling: auto (enable if supported), true, or false.')
    ap.add_argument('--ftr-engine-subdir', action='store_true', help='Also save FTR bundles into ModelStore/<LeagueTag>/(cat|xgb)/<ftr_vX>.pkl')
    ap.add_argument('--ftr-use-lambda', action='store_true', help='Best-effort attach λ goal features into training (implied by --ftr-version v3).')
    ap.add_argument('--market-version', default='v3', choices=['v2','v3'], help='Binary market bundle version tag: v2 or v3 (btts/over25/under25).')
    ap.add_argument('--binary-cat-markets', default=None, help='Comma list of binary markets to train as CatBoost engines. Omit to keep legacy behavior (all binary markets). Pass empty string to skip CatBoost binary training.')
    ap.add_argument('--binary-xgb-markets', default='', help='Comma list of binary markets to also train as XGBoost engines (e.g. btts,over25).')
    ap.add_argument('--write-v2-compat', action='store_true', help='Also write *_v2.pkl compat copies for runtime compatibility (and ou25/u25 aliases).')
    ap.add_argument('--val-frac', type=float, default=0.2, help='Validation fraction (time split)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--iters', type=int, default=1500)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--threads', type=int, default=int(os.getenv('OG_TRAIN_THREADS', '4')), help='CatBoost thread_count (lower reduces RAM spikes).')

    args = ap.parse_args()

    # Default batch list. IMPORTANT: names must match folder names under Matches/ exactly.
    # You can always override this with --leagues "League A,League B".
    investor_default = [
        'Champions League',
        'Europa League',
        'Europa Conference',
        'Germany Bundesliga',
        'Germany Bundesliga 2',
        'France Ligue 1',
        'Italy Serie A',
        'Spain La Liga',
        'England Premier League',
        'England Championship',
        'England EFL League 1',
        'England FA Cup',
        'Portugal Liga',
        'USA MLS',
        'Brazil Serie A',
        'Scotland Premiership',
        'Belgium Pro',
        'Netherlands Eredivisie',
        'Norway Eliteserien',
        'Japan J1',
        'Australia A-League',
        'Austria Bundesliga',
        'Czech First League',
        'Denmark Superliga',
        'Saudi Pro League',
        'South Korea K League',
        'Sweden Allsvenskan',
        'Swiss Super League',
        'Turkey Super Lig',
    ]

    leagues: List[str]
    if args.leagues:
        leagues = [s.strip() for s in str(args.leagues).split(',') if s.strip()]
    else:
        leagues = investor_default

    markets = tuple([m.strip().lower() for m in str(args.markets).split(',') if m.strip()])

    matches_root = Path(args.matches_root)
    modelstore = Path(args.modelstore)

    print(f"📦 Training markets={markets} | modelstore={modelstore}")

    results: List[Dict[str, Any]] = []

    merged_root = Path(getattr(args, "merged_root", "Matches/__merged__"))
    use_merged = bool(getattr(args, "use_merged", False))

    for lg in leagues:
        if use_merged:
            tag = _league_tag(lg)
            mp = merged_root / f"{tag}__merged.csv"
            if not mp.exists():
                print(f"⚠️ {lg}: missing merged file {mp} (falling back to league folder CSVs)")
                mdir = matches_root / lg
                df_all = _load_all_matches_csvs(mdir)
            else:
                df_all = pd.read_csv(mp, low_memory=False)
                # stamp source for diagnostics (keeps existing reporting stable)
                try:
                    df_all["__src_csv"] = mp.name
                except Exception:
                    pass
                # Keep the same hygiene steps your multi-season loader does
                df_all = _apply_renames(df_all)
                df_all = _coerce_numbers(df_all)
                df_all = _ensure_match_date(df_all)
                # Ensure fixture_key exists (trainer relies on it)
                if "fixture_key" not in df_all.columns:
                    try:
                        df_all["fixture_key"] = df_all.apply(_match_key, axis=1)
                    except Exception:
                        df_all["fixture_key"] = pd.NA
        else:
            mdir = matches_root / lg
            df_all = _load_all_matches_csvs(mdir)

        if df_all.empty:
            print(f"⚠️ {lg}: no matches CSVs under {matches_root / lg}")
            continue

        try:
            src = "MERGED" if (use_merged and 'mp' in locals() and mp.exists()) else "FOLDER"
            print(f"\n➡️  {lg}: loaded rows={len(df_all)} source={src}")
        except Exception:
            pass

        res = train_one_league(
            lg,
            matches_df=df_all,
            source_is_merged=bool(use_merged and 'mp' in locals() and mp.exists()),
            modelstore=modelstore,
            markets=markets,
            val_frac=float(args.val_frac),
            seed=int(args.seed),
            iters=int(args.iters),
            overwrite=bool(args.overwrite),
            ftr_version=str(getattr(args, 'ftr_version', 'v2')),
            ftr_engine=str(getattr(args, 'ftr_engine', 'catboost')),
            xgb_categorical=str(getattr(args, 'xgb_categorical', 'auto')),
            ftr_use_lambda=bool(getattr(args, 'ftr_use_lambda', False)),
            market_version=str(getattr(args, 'market_version', 'v3')),
            binary_cat_markets=None if getattr(args, 'binary_cat_markets', None) is None else tuple(m.strip().lower() for m in str(getattr(args, 'binary_cat_markets', '')).split(',') if m.strip()),
            binary_xgb_markets=tuple(m.strip().lower() for m in str(getattr(args, 'binary_xgb_markets', '')).split(',') if m.strip()),
            write_v2_compat=bool(getattr(args, 'write_v2_compat', False)),
            ftr_engine_subdir=bool(getattr(args, 'ftr_engine_subdir', False)),
            threads=int(getattr(args, 'threads', 4)),
            lite=bool(getattr(args, 'lite', False)),
        )
        results.append(res)
        try:
            del df_all
        except Exception:
            pass
        try:
            gc.collect()
        except Exception:
            pass

    stamp = _dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%d_%H%M%S')

    # Write a per-league batch summary inside each league folder (keeps ModelStore tidy)
    try:
        for res in results:
            if not isinstance(res, dict):
                continue
            lg = str(res.get('league', '')).strip()
            tag = str(res.get('tag', '')).strip() or (_modelstore_tag(lg) if lg else '')
            if not tag:
                continue

            outdir = modelstore / tag
            outdir.mkdir(parents=True, exist_ok=True)

            one = {
                'created_utc': _dt.datetime.now(_dt.timezone.utc).isoformat(),
                'stamp': stamp,
                'args': vars(args),
                'per_league': {lg if lg else tag: res},
            }

            outp = outdir / f"batch_summary_{stamp}.json"
            with open(outp, 'w', encoding='utf-8') as fh:
                json.dump(one, fh, indent=2, default=str)

        print("\n🧾 Batch summary → per-league files written into each ModelStore/<LeagueTag>/ folder")
    except Exception:
        # If anything fails, don't break training
        pass


if __name__ == '__main__':
    main()
