# --- CatBoost pickle shim (ensures `catboost.core` exists for old pickles) ---
try:
    import sys as __sys, types as __types
    import catboost as __cb  # requires `pip install catboost`
    if 'catboost.core' not in __sys.modules:
        __core = __types.ModuleType('catboost.core')
        try:
            __core.CatBoostClassifier = __cb.CatBoostClassifier  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            __core.CatBoostRegressor  = __cb.CatBoostRegressor   # type: ignore[attr-defined]
        except Exception:
            pass
        __sys.modules['catboost.core'] = __core
except Exception:
    # If CatBoost isn't installed, downstream loaders should already fall back safely
    pass

import numpy as np
import pandas as pd
import re
from typing import Any
import math
from typing import Any, Optional, cast

# --- Helper utilities for safe numeric casting and env parsing ---
import os as _os_glob

def _to_float(x, default=np.nan) -> float:
    try:
        return float(pd.to_numeric(x, errors="coerce"))
    except Exception:
        try:
            return float(default)
        except Exception:
            return float("nan")

def _to_int(x, default=0) -> int:
    try:
        return int(pd.to_numeric(x, errors="coerce"))
    except Exception:
        try:
            return int(default)
        except Exception:
            return 0

def _getenv_float(key: str, default: float) -> float:
    raw = _os_glob.getenv(key, str(default))
    try:
        return float(raw)
    except Exception:
        return float(default)

def _getenv_int(key: str, default: int) -> int:
    raw = _os_glob.getenv(key, str(default))
    try:
        return int(raw)
    except Exception:
        return int(default)
# Prefer YAML mix via the pipeline (fallback to LOCKED only if needed)
from _baseline_ftr_pipeline import load_mix_params as _load_mix_params
# Global accumulator for pre-gate candidate pools (fallback dumps use this)
all_cands: dict[str, pd.DataFrame] = {}

# Safe fallbacks for optional symbols used later
if "POST_MATCH_TAGS" not in globals():
    POST_MATCH_TAGS: list[str] = []
if "_align_dataframe_to_model" not in globals():
    def _align_dataframe_to_model(df, *args, **kwargs):
        return df

# ---- Normalize per-market prob/od columns to canonical names -----------------
def _map_prob_od_for_market(df, market: str):
    """
    Guarantee presence of:
      - p_model : model probability for the selection being TRUE
      - od      : decimal odds for that selection
    by mapping from known historical column names for each market.
    """
    import pandas as _pd
    if df is None or getattr(df, "empty", True):
        return df
    out = df.copy()

    def _first(cols):
        for c in cols:
            if c in out.columns:
                return c
        return None

    m = (market or "").lower()

    if m == "btts":
        # Probability: prefer adjusted first, then raw/confidence/oof
        pcol = _first([
            "p_model",
            "adjusted_btts_confidence",
            "btts_confidence",
            "oof_prob_btts",
            "p_btts_yes",
            "prob_btts_yes",
            "proba_yes",
        ])
        if pcol:
            _curp = _pd.to_numeric(out.get("p_model"), errors="coerce") if "p_model" in out.columns else None
            if (_curp is None) or (_curp.notna().sum() == 0):
                out["p_model"] = _pd.to_numeric(out[pcol], errors="coerce")

        # Odds: prefer market odds first, then implied, then any existing generic 'od'
        ocol = _first([
            "odds_btts_yes",
            "imp_odds_btts_yes",
            "od",
        ])
        if ocol:
            _curod = _pd.to_numeric(out.get("od"), errors="coerce") if "od" in out.columns else None
            if (_curod is None) or (_curod.le(0).all() or _curod.isna().all()):
                out["od"] = _pd.to_numeric(out[ocol], errors="coerce")

    elif m in ("over25", "over_25", "o2_5"):
        pcol = _first([
            "p_model",
            "adjusted_over25_confidence",
            "over25_confidence",
            "oof_prob_over25",
            "prob_over25",
        ])
        if pcol:
            _curp = _pd.to_numeric(out.get("p_model"), errors="coerce") if "p_model" in out.columns else None
            if (_curp is None) or (_curp.notna().sum() == 0):
                out["p_model"] = _pd.to_numeric(out[pcol], errors="coerce")

        ocol = _first([
            "odds_ft_over25",
            "odds_over25",
            "imp_odds_over25",
            "od",
        ])
        if ocol:
            _curod = _pd.to_numeric(out.get("od"), errors="coerce") if "od" in out.columns else None
            if (_curod is None) or (_curod.le(0).all() or _curod.isna().all()):
                out["od"] = _pd.to_numeric(out[ocol], errors="coerce")

    elif m == "under25":
        # Probability: explicit under, else invert over25/adjusted_over25
        pcol = _first(["p_model","prob_under25","__prob_under25_from_over","__prob_under25_from_adj","under25_confidence"])
        if pcol is None:
            if "prob_over25" in out.columns:
                out["__prob_under25_from_over"] = 1.0 - _pd.to_numeric(out["prob_over25"], errors="coerce")
                pcol = "__prob_under25_from_over"
            elif "adjusted_over25_confidence" in out.columns:
                out["__prob_under25_from_adj"] = 1.0 - _pd.to_numeric(out["adjusted_over25_confidence"], errors="coerce")
                pcol = "__prob_under25_from_adj"
        if pcol and "p_model" not in out.columns:
            out["p_model"] = _pd.to_numeric(out[pcol], errors="coerce")
        ocol = _first(["od","odds_ft_under25","odds_under25","odds_ft_u25","odds_under_2_5","under_2.5_odds","imp_odds_under25"])
        if ocol and "od" not in out.columns:
            out["od"] = _pd.to_numeric(out[ocol], errors="coerce")

    elif m == "btts_no":
        # Probability: explicit no, else invert yes/adjusted yes
        pcol = _first(["p_model","prob_btts_no","__prob_btts_no_from_yes","__prob_btts_no_from_adj","btts_no_confidence"])
        if pcol is None:
            if "prob_btts" in out.columns:
                out["__prob_btts_no_from_yes"] = 1.0 - _pd.to_numeric(out["prob_btts"], errors="coerce")
                pcol = "__prob_btts_no_from_yes"
            elif "adjusted_btts_confidence" in out.columns:
                out["__prob_btts_no_from_adj"] = 1.0 - _pd.to_numeric(out["adjusted_btts_confidence"], errors="coerce")
                pcol = "__prob_btts_no_from_adj"
        if pcol and "p_model" not in out.columns:
            out["p_model"] = _pd.to_numeric(out[pcol], errors="coerce")
        ocol = _first(["od","odds_btts_no","btts_no_odds","imp_odds_btts_no"])
        if ocol and "od" not in out.columns:
            out["od"] = _pd.to_numeric(out[ocol], errors="coerce")

    elif m in ("wtn","wtn_home"):
        pcol = _first(["p_model","prob_home_win_to_nil"])
        if pcol and "p_model" not in out.columns:
            out["p_model"] = _pd.to_numeric(out[pcol], errors="coerce")
        ocol = _first(["od","odds_home_win_to_nil","odds_home_to_nil","home_win_to_nil_odds"])
        if ocol and "od" not in out.columns:
            out["od"] = _pd.to_numeric(out[ocol], errors="coerce")

    elif m == "wtn_away":
        pcol = _first(["p_model","prob_away_win_to_nil"])
        if pcol and "p_model" not in out.columns:
            out["p_model"] = _pd.to_numeric(out[pcol], errors="coerce")
        ocol = _first(["od","odds_away_win_to_nil","odds_away_to_nil","away_win_to_nil_odds"])
        if ocol and "od" not in out.columns:
            out["od"] = _pd.to_numeric(out[ocol], errors="coerce")

    elif m in ("ftr_home","ftr_draw","ftr_away"):
        side = {"ftr_home": "home", "ftr_draw": "draw", "ftr_away": "away"}[m]
        # map prob from confidence_* → p_model
        pcol = _first(["p_model", f"confidence_{side}"])
        if pcol:
            _curp = _pd.to_numeric(out.get("p_model"), errors="coerce") if "p_model" in out.columns else None
            if (_curp is None) or (_curp.notna().sum() == 0):
                out["p_model"] = _pd.to_numeric(out[pcol], errors="coerce")
        # map odds from odds_ft_* → od, prefer specific FTR odds column first
        odds_map = {
            "home": "odds_ft_home_team_win",
            "draw": "odds_ft_draw",
            "away": "odds_ft_away_team_win",
        }
        ocol = _first([odds_map[side], "od"])
        if ocol:
            _curod = _pd.to_numeric(out.get("od"), errors="coerce") if "od" in out.columns else None
            if (_curod is None) or (_curod.le(0).all() or _curod.isna().all()):
                out["od"] = _pd.to_numeric(out[ocol], errors="coerce")
        # After assigning real market odds, set od_source to 'market' if positive odds
        try:
            _chk = _pd.to_numeric(out.get("od"), errors="coerce")
            if _chk is not None and bool((_chk > 0).any()):
                out["od_source"] = out.get("od_source").fillna("market") if "od_source" in out.columns else "market"
        except Exception:
            pass
        # If odds still missing or non-positive, synthesize from p_model when allowed
        try:
            import os as _os
            if "od" in out.columns:
                _odz = _pd.to_numeric(out["od"], errors="coerce")
            else:
                _odz = None
            _allow_syn = str(_os.getenv("ALLOW_SYNTH_ODDS","0")).strip().lower() in ("1","true","yes","y","on")
            if _allow_syn and ("p_model" in out.columns) and (_odz is None or _odz.le(0).all() or _odz.isna().all()):
                _pm = _pd.to_numeric(out["p_model"], errors="coerce").replace(0, _pd.NA)
                out["od"] = (1.0 / _pm).clip(lower=1.01)
                out["od_source"] = "synth"
        except Exception:
            pass
    return out

def _match_key(row: pd.Series) -> str:
    """
    Stable key for fixture rows using date + teams when available.
    Falls back to index if anything is missing.
    """
    try:
        md = str(row.get("match_date", "")).strip()
        h  = str(row.get("home_team_name", "")).strip()
        a  = str(row.get("away_team_name", "")).strip()
        if md or h or a:
            return re.sub(r"[^A-Za-z0-9_]+", "_", f"{md}_{h}_{a}").strip("_") or str(row.name)
    except Exception:
        pass
    return str(row.name)

def _get_mix(league: str) -> dict:
    """Prefer YAML via pipeline loader; fallback to LOCKED_FTR_MIX safely."""
    try:
        return _load_mix_params(league)
    except Exception:
        try:
            from constants import LOCKED_FTR_MIX as _LOCKED
            # exact key, slug, or default
            m = (_LOCKED or {}).get(league)
            if not m:
                tag = str(league).replace(" ", "_")
                m = (_LOCKED or {}).get(tag)
            return m or {"alpha": 0.50, "beta": 0.10, "cap": 0.33, "gate_scale": 1.00}
        except Exception:
            return {"alpha": 0.50, "beta": 0.10, "cap": 0.33, "gate_scale": 1.00}
        
from etl_press_intensity import ensure_press_intensity_on_disk
def attach_synth_odds(df: pd.DataFrame, market: str | None = None) -> pd.DataFrame:
    """
    No-op stub so composer paths don’t crash if synth-odds injection is toggled.
    Replace with the real implementation when ready.
    """
    return df

# --- Per-league draw temperature resolver -----------------------------------
if "_resolve_draw_temp" not in globals():
    def _resolve_draw_temp(league: object, default: float | None = None) -> float:
        """Resolve draw temperature T with a per-league override.
        Uses DRAW_TEMP_<LEAGUE_TAG> if set; otherwise falls back to DRAW_TEMP; else 1.4.
        """
        import os, re
        try:
            base = float(os.getenv("DRAW_TEMP", str(1.4 if default is None else default)))
        except Exception:
            base = 1.4 if default is None else float(default)
        if not league:
            return base
        try:
            tag = re.sub(r"[^A-Za-z0-9_]+", "_", str(league).upper()).strip("_")
        except Exception:
            tag = ""
        try:
            return float(os.getenv(f"DRAW_TEMP_{tag}", str(base)))
        except Exception:
            return base

# --- Per-league resolver for gate thresholds (IMP/PAR/MARGIN) ---------------
if "_resolve_gate_param" not in globals():
    def _resolve_gate_param(base_key: str, league: object, default: float) -> float:
        """Resolve a numeric gate parameter with optional per-league override.
        Looks for BASE_KEY (global) then BASE_KEY_<LEAGUE_TAG> (per-league),
        where LEAGUE_TAG uppercases and underscores the league string.
        """
        import os, re
        try:
            base = float(os.getenv(base_key, str(default)))
        except Exception:
            base = float(default)
        if not league:
            return base
        try:
            tag = re.sub(r"[^A-Za-z0-9_]+", "_", str(league).upper()).strip("_")
        except Exception:
            tag = ""
        try:
            return float(os.getenv(f"{base_key}_{tag}", str(base)))
        except Exception:
            return base



# === OG: canonical draw-artifact resolver (newest wins) ===
def _og_slug(league):
    import re
    return re.sub(r"[^a-z0-9]+","_", str(league).lower()).strip("_") or "unknown_league"

def _og_resolve_draw_artifacts(league_name: str) -> tuple[str, str, str, str]:
    """
    Resolve ModelStore artifact paths for draw components for the given league.
    Returns (bundle_path, alt_bundle_path, clf_pickle_path, threshold_json_path).
    """
    import os
    slug = _og_slug(league_name)
    # Proper-cased underscore form (e.g., "England_Premier_League")
    proper = slug.replace("_", " ").title().replace(" ", "_")
    base = os.path.join(os.path.dirname(__file__), "ModelStore")
    bundle = os.path.join(base, f"{slug}__draw_bundle.joblib")
    alt    = os.path.join(base, f"{proper}_draw_bundle.joblib")
    pkl    = os.path.join(base, f"{proper}_draw_clf.pkl")
    thr    = os.path.join(base, f"{proper}_draw_threshold.json")
    return bundle, alt, pkl, thr
# === Draw model bridge: reuse main pipeline mixer (λ + supervised draw) ===
def attach_ftr_with_draw_mix(df, league):
    """
    Mix the existing λ-Poisson FTR with a supervised draw model and league mix params.
    Safe no-op if the draw model bundle or threshold JSON are missing.
    Writes: confidence_home/draw/away and ftr_pred_outcome (0/1/2).
    """
    import os as _os
    if _os.getenv("VERBOSE_DRAW","0") == "1" or _os.getenv("VERBOSE_QUICK","0") == "1":
        try:
            print(f"🔧 overlay: attach_ftr_with_draw_mix from {__file__} | league={league}")
        except Exception:
            pass
    import os, json
    import numpy as np
    import pandas as pd
    # Temperature soften supervised draw (env‑tunable; per‑league override)
    T = _resolve_draw_temp(league)
    import os as _os
    if _os.getenv("VERBOSE_QUICK", "0") == "1":
        try:
            _tag = str(league).replace(" ", "_")
            print(f"🧪 draw temp T={T:.3f} ({_tag})")
        except Exception:
            pass
    
    # Safe imports from overlay for schema/placeholder helpers
    try:
        from prediction_overlay import apply_safe_renames_and_whitelist, ensure_realized_goal_placeholders
    except Exception:
        apply_safe_renames_and_whitelist = None
        ensure_realized_goal_placeholders = None
    try:
        import joblib
    except Exception:
        joblib = None
    # Late imports so this helper can be dropped in without touching top-level imports
    from constants import MODEL_DIR
    try:
        from _baseline_ftr_pipeline import ensure_draw_ready_features, predict_FTR_adaptive
    except Exception:
        # fallback alias if the file is named differently on disk
        from _baseline_ftr_pipeline import ensure_draw_ready_features, predict_FTR_adaptive

    # Resolve newest/canonical artifacts first
    bundle_path, alt_bundle_path, pkl_path, thr_path = _og_resolve_draw_artifacts(league)
    draw_bundle = None
    draw_thr = None
    thr_payload = {}
    tag = str(league).replace(" ", "_")
    # Debug: show candidates & existence
    try:
        print(
            "🧭 draw-bundle candidates:\n"
            f"  bundle={bundle_path} (exists={os.path.exists(bundle_path) if bundle_path else False})\n"
            f"  alt={alt_bundle_path} (exists={os.path.exists(alt_bundle_path)})\n"
            f"  pkl={pkl_path} (exists={os.path.exists(pkl_path)})\n"
            f"  thr={thr_path} (exists={os.path.exists(thr_path) if thr_path else False})"
        )
    except Exception:
        pass

    # Load learned deployment threshold (and keep payload for features_used fallback)
    if thr_path and os.path.exists(thr_path):
        try:
            with open(thr_path, "r") as fh:
                thr_payload = json.load(fh) or {}
            if "threshold" in thr_payload and thr_payload["threshold"] is not None:
                draw_thr = float(thr_payload["threshold"])
        except Exception as e:
            print(f"⚠️ draw threshold load failed: {e}")

    # Canonical newest-first loader: load the resolved bundle_path if present
    if draw_bundle is None and joblib:
        try:
            if bundle_path and os.path.exists(bundle_path):
                draw_bundle = joblib.load(bundle_path)
                if (_os.getenv("VERBOSE_DRAW","0") == "1" or _os.getenv("VERBOSE_QUICK","0") == "1"):
                    print(f"🧩 using draw bundle: {os.path.basename(bundle_path)}")
        except Exception as e:
            print(f"⚠️ draw bundle load failed: {e}")

    # Legacy pkl fallback (wrap or unwrap accordingly)
    if draw_bundle is None and joblib and os.path.exists(pkl_path):
        try:
            legacy_obj = joblib.load(pkl_path)
            feats_json = thr_payload.get("features_used") or thr_payload.get("features") or []
            if isinstance(legacy_obj, dict) and "model" in legacy_obj:
                # already a bundle – add features if missing
                draw_bundle = dict(legacy_obj)
                if not draw_bundle.get("features"):
                    draw_bundle["features"] = list(feats_json)
            else:
                # plain estimator – wrap it
                draw_bundle = {"model": legacy_obj, "features": list(feats_json)}
            if (_os.getenv("VERBOSE_DRAW","0") == "1" or _os.getenv("VERBOSE_QUICK","0") == "1"):
                print(f"🧩 using legacy draw pkl: {os.path.basename(pkl_path)} "
                    f"(features from JSON: n={len(draw_bundle.get('features',[]))})")
        except Exception as e:
            print(f"⚠️ draw pkl load failed: {e}")

    # Bail if still no model
    if not isinstance(draw_bundle, dict) or ("model" not in draw_bundle):
        return df

    base_clf  = draw_bundle["model"]
    draw_feats = list(draw_bundle.get("features", []))

    # --- Canonicalize bundle path: persist to <tag>__draw_bundle.joblib once ---
    try:
        _canon = os.path.join(MODEL_DIR, f"{tag}__draw_bundle.joblib")
        if joblib and not os.path.exists(_canon):
            joblib.dump(draw_bundle, _canon)
            if (_os.getenv("VERBOSE_DRAW","0") == "1" or _os.getenv("VERBOSE_QUICK","0") == "1"):
                print(f"💾 normalized draw bundle → {os.path.basename(_canon)}")
    except Exception as _e:
        try:
            if (_os.getenv("VERBOSE_DRAW","0") == "1" or _os.getenv("VERBOSE_QUICK","0") == "1"):
                print(f"⚠️ could not normalize bundle: {_e}")
        except Exception:
            pass

    # Unwrap any nested bundle dict accidentally stored under 'model'
    if isinstance(base_clf, dict) and "model" in base_clf:
        base_clf = base_clf["model"]

    # Pass the calibrated draw model straight through; temperature is handled in the mixer
    draw_clf = base_clf

    # --- Wrap the classifier to scrub any NaN/±Inf on predict_proba, and proxy attrs ----
    class _NoNaNPredictProba:
        def __init__(self, inner):
            self.inner = inner
            # Copy common sklearn attributes if present so upstream code can read them
            for attr in ("classes_", "feature_names_in_", "n_features_in_"):
                if hasattr(inner, attr):
                    setattr(self, attr, getattr(inner, attr))

        def __getattr__(self, name):
            # Proxy any other attribute/method lookups to the inner estimator
            return getattr(self.inner, name)

        def predict_proba(self, X):
            import numpy as _np
            # Accept DataFrame/Series/ndarray → numpy float64
            try:
                A = X.to_numpy(dtype=_np.float64, copy=True)
            except Exception:
                A = _np.array(X, dtype=_np.float64, copy=True)
            # Defensive scrub
            A = _np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
            return self.inner.predict_proba(A)

    # Use the safe wrapper for all downstream calls (vectorized + row-wise)
    draw_clf = _NoNaNPredictProba(base_clf)
    # Ensure sklearn-like attributes exist for downstream code
    try:
        if not hasattr(draw_clf, "feature_names_in_"):
            draw_clf.feature_names_in_ = np.array(draw_feats)
        if not hasattr(draw_clf, "n_features_in_"):
            draw_clf.n_features_in_ = len(draw_feats)
    except Exception:
        pass
    # NOTE: Overlay does not apply temperature scaling to the draw classifier; mixer handles T.
    T = _resolve_draw_temp(league)

    # (VERBOSE_QUICK print now handled above)

    # League-specific mix params (YAML via pipeline; fallback to LOCKED only if needed)
    mix = _get_mix(league)
    alpha = float(mix.get("alpha", 0.50))
    beta  = float(mix.get("beta",  0.10))
    cap   = float(mix.get("cap",   0.33))
    gate_scale = float(mix.get("gate_scale", 1.00))
    import os as _os
    use_gate = (_os.getenv("NO_DRAW_GATE", "0") != "1")
    if not use_gate and _os.getenv("VERBOSE_DRAW", "0") == "1":
        try:
            _tag = str(league).replace(" ", "_")
            print(f"ℹ️ draw gate disabled via NO_DRAW_GATE=1 ({_tag})")
        except Exception:
            pass
    # --- Gate thresholds (env-tunable), scaled by gate_scale ---
    imp_floor = _resolve_gate_param("DRAW_GATE_IMP_MIN", league, 0.26)
    par_floor = _resolve_gate_param("DRAW_GATE_PAR_MIN", league, 0.65)
    margin_max = _resolve_gate_param("DRAW_GATE_MARGIN_MAX", league, 0.02)
    # Scale thresholds: gate_scale > 1.0 makes gate harder to trigger
    imp_thr_scaled    = imp_floor * gate_scale
    par_thr_scaled    = par_floor * gate_scale
    margin_thr_scaled = margin_max / max(gate_scale, 1e-6)

    # Build leak-safe draw features and align to model schema
    work = ensure_draw_ready_features(df.copy())
    # Ensure specialist side probabilities exist for upcoming frames (maps prob_* → oof_prob_*)
    try:
        from _baseline_ftr_pipeline import _ensure_upcoming_side_probs as _ensure_side
        work = _ensure_side(work, league)
    except Exception:
        pass
    # If draw_implied is missing/scarce, derive it from 1x2 odds (normalized)
    try:
        need_implied = ("draw_implied" not in work.columns) or (
            pd.to_numeric(work.get("draw_implied"), errors="coerce").isna().mean() > 0.2
        )
        if need_implied:
            # Use the overlay's odds→probs helper to get confidence_draw
            try:
                _w2 = infer_ftr_from_odds(work.copy())
                cd = pd.to_numeric(_w2.get("confidence_draw"), errors="coerce")
            except Exception:
                cd = pd.Series(np.nan, index=work.index, dtype="float64")

            # Backfill draw_implied where missing
            if "draw_implied" in work.columns:
                di = pd.to_numeric(work["draw_implied"], errors="coerce")
                work["draw_implied"] = di.fillna(cd)
            else:
                work["draw_implied"] = cd

            if (_os.getenv("VERBOSE_DRAW","0") == "1" or _os.getenv("VERBOSE_QUICK","0") == "1"):
                avail = int(pd.to_numeric(work["draw_implied"], errors="coerce").notna().sum())
                print(f"🧪 backfilled draw_implied from odds: avail={avail}/{len(work)}")
    except Exception as _e:
        if (_os.getenv("VERBOSE_DRAW","0") == "1" or _os.getenv("VERBOSE_QUICK","0") == "1"):
            print(f"ℹ️ draw_implied backfill skipped: {_e}")

    # If implied_prob_diff is missing/scarce, derive it from normalized 1x2 implied probs
    try:
        need_mrg = ("implied_prob_diff" not in work.columns) or (
            pd.to_numeric(work.get("implied_prob_diff"), errors="coerce").isna().mean() > 0.2
        )
        if need_mrg:
            # Resolve 1x2 odds columns
            home_col = _resolve_odds(work, ODD_COLUMN_WHITELIST.get("home_win", []), label="FTR HOME odds")
            draw_col = _resolve_odds(work, ODD_COLUMN_WHITELIST.get("draw", []),     label="FTR DRAW odds")
            away_col = _resolve_odds(work, ODD_COLUMN_WHITELIST.get("away_win", []), label="FTR AWAY odds")
            if (isinstance(home_col, str) and home_col in work.columns and
                isinstance(draw_col, str) and draw_col in work.columns and
                isinstance(away_col, str) and away_col in work.columns):
                oh = pd.to_numeric(work[home_col], errors="coerce")
                od = pd.to_numeric(work[draw_col], errors="coerce")
                oa = pd.to_numeric(work[away_col], errors="coerce")
                qh = (1.0 / oh).replace({0.0: np.nan})
                qd = (1.0 / od).replace({0.0: np.nan})
                qa = (1.0 / oa).replace({0.0: np.nan})
                tot = (qh + qd + qa).replace({0.0: np.nan})
                ph = (qh / tot).clip(0, 1)
                pdw = (qd / tot).clip(0, 1)
                pa = (qa / tot).clip(0, 1)
                imp_m = (pdw - np.maximum(ph, pa)).astype("float64")
                if "implied_prob_diff" in work.columns:
                    cur = pd.to_numeric(work["implied_prob_diff"], errors="coerce")
                    work["implied_prob_diff"] = cur.fillna(imp_m)
                else:
                    work["implied_prob_diff"] = imp_m
                if (_os.getenv("VERBOSE_DRAW","0") == "1" or _os.getenv("VERBOSE_QUICK","0") == "1"):
                    avail_m = int(pd.to_numeric(work["implied_prob_diff"], errors="coerce").notna().sum())
                    print(f"🧪 backfilled implied_prob_diff from odds: avail={avail_m}/{len(work)}")
    except Exception as _e:
        if (_os.getenv("VERBOSE_DRAW","0") == "1" or _os.getenv("VERBOSE_QUICK","0") == "1"):
            print(f"ℹ️ implied_prob_diff backfill skipped: {_e}")
            
    # Debug: list any draw features missing from the frame we’ll score
    if (_os.getenv("VERBOSE_DRAW","0") == "1" or _os.getenv("VERBOSE_QUICK","0") == "1"):
        missing = [c for c in draw_feats if c not in work.columns]
        try:
            print(
                f"🧪 draw-ready: created {missing if missing else []} | "
                f"have {sum(c in work.columns for c in draw_feats)}/{len(draw_feats)} core cols"
            )
        except Exception:
            pass

    # --- Build draw feature matrix robustly (no NaNs, strict order) ---
    X = pd.DataFrame(index=work.index)
    for c in draw_feats:
        if c in work.columns:
            s = pd.to_numeric(work[c], errors="coerce")
        else:
            s = pd.Series(0.0, index=work.index)  # absent feature → zeros
        X[c] = s

    # Final sanitize (DataFrame)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float64")
    # Strict feature order that the model was trained with
    X = X.loc[:, list(draw_feats)]

    # (verbose) surface anything unexpected before we convert to NumPy
    if (_os.getenv("VERBOSE_DRAW","0") == "1" or _os.getenv("VERBOSE_QUICK","0") == "1"):
        if bool(X.isna().any().any()):
            bad = [c for c in draw_feats if bool(X[c].isna().any())]
            print(f"⚠️ overlay: NaNs remain in draw X after fill; cols={bad}")
        try:
            dvc = X.dtypes.value_counts()
            print(f"🧪 X dtypes: {dict((str(k), int(v)) for k,v in dvc.items())}")
        except Exception:
            pass

    # Convert to a NumPy array and hard-sanitize again for GBM safety
    X_arr = X.to_numpy(dtype=np.float64, copy=True)
    bad_nan = int(np.isnan(X_arr).sum())
    bad_inf = int(np.isfinite(X_arr).size - np.isfinite(X_arr).sum())
    if (_os.getenv("VERBOSE_DRAW","0") == "1" or _os.getenv("VERBOSE_QUICK","0") == "1") and (bad_nan or bad_inf):
        print(f"⚠️ overlay: pre-scrub NaN={bad_nan} ±Inf={bad_inf} in X_arr → scrubbing")
    X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Supervised draw probability with defensive fallback
    try:
        p_draw_sup = draw_clf.predict_proba(X_arr)[:, 1]
    except Exception as e:
        # Fall back to market-implied draw so overlay can proceed & gate can print
        print(f"⚠️ draw predict failed: {e} → falling back to market-implied draw")
        # Prefer draw_implied if available; else safe prior 0.25
        if "draw_implied" in work.columns:
            p_draw_sup = pd.to_numeric(work["draw_implied"], errors="coerce").fillna(0.25).to_numpy()
        else:
            p_draw_sup = np.full(len(work), 0.25, dtype=np.float64)
    
    # Candidate columns from features for gate diagnostics
    draw_implied = pd.to_numeric(work.get("draw_implied"), errors="coerce") if "draw_implied" in work.columns else None
    lam_parity   = pd.to_numeric(work.get("lam_parity"),   errors="coerce") if "lam_parity"   in work.columns else None
    imp_diff     = pd.to_numeric(work.get("implied_prob_diff"), errors="coerce") if "implied_prob_diff" in work.columns else None

    import os as _os
    # Gate off-switch and predicate strictness
    try:
        no_gate = str(_os.getenv("NO_DRAW_GATE", "0")).strip().lower() in ("1","true","yes","y","on")
    except Exception:
        no_gate = False
    try:
        require_all_predicates = str(_os.getenv("DRAW_GATE_REQUIRE_ALL", "0")).strip().lower() in ("1","true","yes","y","on")
    except Exception:
        require_all_predicates = False
    # Convenience alias for summary
    use_gate = (not no_gate)

    # Predicate pass counts (treat NaNs as False so they don't inflate)
    _cnt_imp_ok = int((draw_implied >= imp_thr_scaled).fillna(False).sum()) if isinstance(draw_implied, pd.Series) else 0
    _cnt_par_ok = int((lam_parity   >= par_thr_scaled).fillna(False).sum())  if isinstance(lam_parity,   pd.Series) else 0
    _cnt_mrg_ok = int((imp_diff     <= margin_thr_scaled).fillna(False).sum()) if isinstance(imp_diff,     pd.Series) else 0

    # Availability counts (diagnostics)
    _cnt_imp_avail = int(draw_implied.notna().sum()) if isinstance(draw_implied, pd.Series) else 0
    _cnt_par_avail = int(lam_parity.notna().sum())   if isinstance(lam_parity,   pd.Series) else 0
    _cnt_mrg_avail = int(imp_diff.notna().sum())     if isinstance(imp_diff,     pd.Series) else 0

    # Total rows for this slate (use the working frame)
    n_total = len(work)

    # Verbosity flag (off-switch managed via use_gate)
    verbose_draw = False
    try:
        verbose_draw = (str(os.getenv("VERBOSE_DRAW", "0")) == "1") or (str(os.getenv("VERBOSE_QUICK", "0")) == "1")
    except Exception:
        pass

    # Predicate activation & pass-rate heuristics
    use_imp = True
    use_par = True
    use_mrg = True

    # Compute pass rates (treat missing as False)
    imp_ok_rate = (_cnt_imp_ok / max(_cnt_imp_avail, 1)) if _cnt_imp_avail else 0.0
    mrg_ok_rate = (_cnt_mrg_ok / max(_cnt_mrg_avail, 1)) if _cnt_mrg_avail else 0.0

    # Auto-disable implied predicate if pass rate is extremely low on this slate
    _min_imp_rate = _getenv_float("DRAW_GATE_IMP_MIN_PASS_RATE", 0.05)
    if imp_ok_rate < _min_imp_rate:
        use_imp = False
        if verbose_draw:
            print("ℹ️ disabling implied predicate for this slate (pass rate too low)")

    # Auto-disable margin predicate if availability is tiny
    _min_mrg_avail_frac = _getenv_float("DRAW_GATE_MRG_MIN_AVAIL_FRAC", 0.20)  # 20% default
    if _cnt_mrg_avail < int(_min_mrg_avail_frac * n_total):
        use_mrg = False
        if verbose_draw:
            print("ℹ️ disabling margin predicate (insufficient availability)")

    # Minimum number of active predicates that must pass when REQUIRE_ALL is False
    min_active = _getenv_int("DRAW_GATE_MIN_ACTIVE", 1)

    # Relax REQUIRE_ALL if implied availability is low on this slate
    try:
        if require_all_predicates and _cnt_imp_avail < int(0.8 * n_total):
            require_all_predicates = False
            if verbose_draw:
                print("ℹ️ REQUIRE_ALL relaxed (insufficient implied availability)")
    except Exception:
        pass

    # Precompute a boolean mask for classifier-agreeing rows (AND gating)
    cls_mask = None
    try:
        if draw_thr is not None:
            cls_mask = (pd.to_numeric(pd.Series(p_draw_sup, index=df.index), errors="coerce") >= _to_float(draw_thr))
    except Exception:
        cls_mask = None

    out = df.copy()
    # Ensure row-level access to draw features for predict_FTR_adaptive:
    # copy any present features from the prepared 'work' frame; zero-fill the rest
    try:
        for _c in draw_feats:
            if _c in out.columns:
                # already present — coerce numeric
                out[_c] = pd.to_numeric(out[_c], errors="coerce").fillna(0.0)
            elif _c in work.columns:
                out[_c] = pd.to_numeric(work[_c], errors="coerce").fillna(0.0)
            else:
                out[_c] = 0.0
    except Exception:
        # best-effort; mixer will still run with zeros if anything slips
        for _c in draw_feats:
            if _c not in out.columns:
                out[_c] = 0.0

    if "home_goals_pred" in out.columns:
        h = pd.to_numeric(out["home_goals_pred"], errors="coerce")
    else:
        h = pd.Series(np.nan, index=out.index, dtype="float64")

    if "away_goals_pred" in out.columns:
        a = pd.to_numeric(out["away_goals_pred"], errors="coerce")
    else:
        a = pd.Series(np.nan, index=out.index, dtype="float64")

    # Introduce counters for diagnostics
    _n_rows = len(df)
    _ser_p  = pd.Series(p_draw_sup, index=df.index)
    _cnt_cls_agree = int(_ser_p.ge(_to_float(draw_thr)).sum()) if (draw_thr is not None) else 0 
    _cnt_imp_ok = int(draw_implied.ge(imp_thr_scaled).sum()) if isinstance(draw_implied, pd.Series) else 0
    _cnt_par_ok = int(lam_parity.ge(par_thr_scaled).sum())   if isinstance(lam_parity,   pd.Series) else 0
    _cnt_mrg_ok = int(imp_diff.le(margin_thr_scaled).sum())  if isinstance(imp_diff,     pd.Series) else 0
    _cnt_gate_used = 0

    ph, pdw, pa = [], [], []
    for i, row in out.iterrows():
        _gate_row = None
        try:
            if (not no_gate) and (draw_thr is not None):
                # Classifier agreement is mandatory
                _ok = bool(cls_mask.loc[i]) if (cls_mask is not None) else True

                # Track individual predicate passes
                imp_pass = True
                par_pass = True
                mrg_pass = True

                # draw_implied ≥ threshold (fail if missing when require_all_predicates, else ignore)
                if use_imp and isinstance(draw_implied, pd.Series):
                    _vi = draw_implied.loc[i]
                    imp_pass = (not pd.isna(_vi)) and (_to_float(_vi) >= _to_float(imp_thr_scaled))
                    if require_all_predicates and not imp_pass:
                        _ok = False

                # lam_parity ≥ threshold
                if use_par and isinstance(lam_parity, pd.Series):
                    _vp = lam_parity.loc[i]
                    par_pass = (not pd.isna(_vp)) and (_to_float(_vp) >= _to_float(par_thr_scaled))
                    if require_all_predicates and not par_pass:
                        _ok = False

                # implied_prob_diff ≤ threshold
                if use_mrg and isinstance(imp_diff, pd.Series):
                    _vm = imp_diff.loc[i]
                    mrg_pass = (not pd.isna(_vm)) and (_to_float(_vm) <= _to_float(margin_thr_scaled))
                    if require_all_predicates and not mrg_pass:
                        _ok = False

                # In relaxed mode, require at least `min_active` of the active predicates to pass
                if not require_all_predicates:
                    passed = (
                        (1 if (use_imp and imp_pass) else 0)
                        + (1 if (use_par and par_pass) else 0)
                        + (1 if (use_mrg and mrg_pass) else 0)
                    )
                    if passed < _to_int(min_active):
                        _ok = False

                if _ok:
                    _gate_row = _to_float(draw_thr)
        except Exception:
            _gate_row = _to_float(draw_thr) if (not no_gate and draw_thr is not None) else None

        if _gate_row is not None:
            _cnt_gate_used += 1

        p_home, p_draw, p_away = predict_FTR_adaptive(
            row,
            _to_float(h.get(i, np.nan)) if h is not None else np.nan,
            _to_float(a.get(i, np.nan)) if a is not None else np.nan,
            draw_classifier=draw_clf,
            draw_features=draw_feats,
            alpha=alpha, beta=beta, overlay_cap=cap, gate_scale=gate_scale,
            draw_gate=_gate_row
        )
        ph.append(p_home); pdw.append(p_draw); pa.append(p_away)

    if verbose_draw:
        try:
            _no_gate_i = 1 if no_gate else 0
            _require_all_i = 1 if require_all_predicates else 0
            _use_imp_i = 1 if use_imp else 0
            _use_par_i = 1 if use_par else 0
            _use_mrg_i = 1 if use_mrg else 0
            try:
                _min_active_i = int(min_active)
            except Exception:
                _min_active_i = _to_int(min_active)
            tag = str(league).replace(" ", "_")
            print(
                f"🧰 draw-gate[{tag}]: "
                f"thr={_to_float(draw_thr):.3f} | gate_scale={_to_float(gate_scale):.2f} | "
                f"cls>=thr {_cnt_cls_agree}/{_n_rows} | "
                f"imp>= {imp_thr_scaled:.3f} ({_cnt_imp_ok}/{_cnt_imp_avail} avail) | "
                f"par>= {par_thr_scaled:.3f} ({_cnt_par_ok}/{_cnt_par_avail} avail) | "
                f"margin<= {margin_thr_scaled:.3f} ({_cnt_mrg_ok}/{_cnt_mrg_avail} avail) | "
                f"gate_used {_cnt_gate_used}/{_n_rows} | "
                f"NO_DRAW_GATE={_no_gate_i} | REQUIRE_ALL={_require_all_i} | "
                f"use[imp={_use_imp_i},par={_use_par_i},mrg={_use_mrg_i}] min_active={_min_active_i}"
            )
        except Exception:
            pass

    out["confidence_home"] = np.clip(ph, 0, 1)
    out["confidence_draw"] = np.clip(pdw, 0, 1)
    out["confidence_away"] = np.clip(pa, 0, 1)
    out["ftr_pred_outcome"] = out[["confidence_home","confidence_draw","confidence_away"]].values.argmax(axis=1).astype(int)
    return out

# ==== NEW: thresholds + trained-model scoring helpers =====================

def _load_market_thresholds_json(league_name: str) -> dict:
    """
    Try a few common paths to load per-market thresholds for a league.
    Expected schema example:
      {"btts": 0.56, "over25": 0.36, "under25": 0.61, "btts_no": 0.58, "clean_sheet": 0.33}
    Returns {} if not found or unreadable.
    """
    try:
        import os, json
        from constants import MODEL_DIR
        safe = str(league_name).replace(" ", "_")
        candidates = [
            os.path.join(MODEL_DIR, f"{safe}_market_thresholds.json"),
            os.path.join(MODEL_DIR, safe, "market_thresholds.json"),
        ]
        for p in candidates:
            try:
                if os.path.exists(p):
                    with open(p, "r") as fh:
                        data = json.load(fh)
                    if isinstance(data, dict):
                        return data
            except Exception:
                continue
    except Exception:
        pass
    return {}

def _apply_market_thresholds_to_attrs(df, league_name: str):
    """
    Load market thresholds JSON (if present) and mirror values into df.attrs
    using the same keys the overlay already looks for (e.g., 'thr_btts').
    Non-destructive: existing df.attrs overrides win.
    """
    try:
        th = _load_market_thresholds_json(league_name)
        if not th:
            return df
        mapping = {
            "btts":        ("thr_btts", "thr_btts_default"),
            "over25":      ("thr_over25", "thr_over25_default"),
            "under25":     ("thr_under25", "thr_under25_default"),
            "btts_no":     ("thr_btts_no", "thr_btts_no_default"),
            "clean_sheet": ("thr_clean_sheet", "thr_clean_sheet_default"),
        }
        for mk, (k_primary, k_fallback) in mapping.items():
            if k_primary in df.attrs:     # explicit run-time override wins
                continue
            val = None
            if mk in th:
                val = th.get(mk)
            elif k_primary in th:
                val = th.get(k_primary)
            elif k_fallback in th:
                val = th.get(k_fallback)
            if val is not None:
                try:
                    df.attrs[k_primary] = float(val)
                except Exception:
                    pass
    except Exception:
        pass
    return df
# --- NEW: public helper to load per-league market thresholds → df.attrs keys ---
from constants import MODEL_DIR as _CONST_MODEL_DIR  # safe reimport; used inside helper

def load_market_thresholds_for_league(league: str) -> dict:
    """
    Return a dict of df.attrs keys → threshold floats, e.g.
    {"thr_btts": 0.57, "thr_over25": 0.48, ...}. Safe no-op if JSON missing.
    Handles both schemas:
      1) {"markets": {"btts": {"threshold": 0.57, ...}, ...}}
      2) {"btts": 0.57, "over25": 0.48, ...}
    """
    import os, json, re

    def _slug(s):
        return re.sub(r"[^A-Za-z0-9_]+", "", re.sub(r"\s+", "_", str(s).strip()))

    league_tag = _slug(league)
    path = os.path.join(_CONST_MODEL_DIR, f"{league_tag}_market_thresholds.json")
    out: dict[str, float] = {}
    keymap = {
        "btts": "thr_btts",
        "over25": "thr_over25",
        "under25": "thr_under25",
        "btts_no": "thr_btts_no",
        "home_fts": "thr_home_fts",
        "away_fts": "thr_away_fts",
        "home_ge2": "thr_home_ge2",
        "away_ge2": "thr_away_ge2",
        "btts_fh": "thr_btts_fh",
        "clean_sheet": "thr_clean_sheet",
        "wtn": "thr_wtn",
        "ftr": "thr_ftr",
    }
    try:
        with open(path, "r") as fh:
            payload = json.load(fh)
        # Schema 1: nested under "markets"
        if isinstance(payload, dict) and isinstance(payload.get("markets"), dict):
            for mkt, info in payload["markets"].items():
                try:
                    thr = info.get("threshold") if isinstance(info, dict) else None
                    k = keymap.get(str(mkt))
                    if k is not None and isinstance(thr, (int, float)):
                        out[k] = float(thr)
                except Exception:
                    continue
        # Schema 2: flat mapping {market: threshold}
        elif isinstance(payload, dict):
            for mkt, thr in payload.items():
                k = keymap.get(str(mkt))
                if k is not None:
                    try:
                        out[k] = float(thr)
                    except Exception:
                        continue
    except Exception:
        pass
    return out

# ----------------------------------------------------------------------------
# Side‑market calibration store (isotonic per league)
# ----------------------------------------------------------------------------
from pathlib import Path as _Path

try:
    from sklearn.isotonic import IsotonicRegression as _Iso
    import joblib as _joblib
except Exception:
    _Iso = None
    _joblib = None

_CAL_MARKET_DEFAULTS = (
    "btts","over25","under25","btts_no",
    "home_ge2","away_ge2","home_fts","away_fts",
    "ah_home_minus15","ah_home_minus25",
)

# Map market → (prob_col, target_deriver)
# NOTE: fit_* functions require completed fixtures (use on training/holdout frames, not upcoming).

def _derive_target_from_goals(df: pd.DataFrame, *, kind: str) -> pd.Series:
    hg = pd.to_numeric(df.get("home_team_goal_count"), errors="coerce")
    ag = pd.to_numeric(df.get("away_team_goal_count"), errors="coerce")
    if hg is None or ag is None:
        return pd.Series([np.nan]*len(df), index=df.index, dtype=float)
    if kind == "btts":
        return ((hg >= 1) & (ag >= 1)).astype(int)
    if kind == "over25":
        return ((hg + ag) >= 3).astype(int)
    if kind == "under25":
        return ((hg + ag) <= 2).astype(int)
    if kind == "btts_no":
        return ((hg == 0) | (ag == 0)).astype(int)
    if kind == "home_ge2":
        return (hg >= 2).astype(int)
    if kind == "away_ge2":
        return (ag >= 2).astype(int)
    if kind == "home_fts":
        return (hg == 0).astype(int)
    if kind == "away_fts":
        return (ag == 0).astype(int)
    if kind == "ah_home_minus15":
        return ((hg - ag) >= 2).astype(int)
    if kind == "ah_home_minus25":
        return ((hg - ag) >= 3).astype(int)
    return pd.Series([np.nan]*len(df), index=df.index, dtype=float)

_CAL_PROB_COLS = {
    "btts":              ("prob_btts",              lambda d: _derive_target_from_goals(d, kind="btts")),
    "over25":            ("prob_over25",            lambda d: _derive_target_from_goals(d, kind="over25")),
    "under25":           ("prob_under25",          lambda d: _derive_target_from_goals(d, kind="under25")),
    "btts_no":           ("prob_btts_no",          lambda d: _derive_target_from_goals(d, kind="btts_no")),
    "home_ge2":          ("prob_home_ge2",         lambda d: _derive_target_from_goals(d, kind="home_ge2")),
    "away_ge2":          ("prob_away_ge2",         lambda d: _derive_target_from_goals(d, kind="away_ge2")),
    "home_fts":          ("prob_home_fts",         lambda d: _derive_target_from_goals(d, kind="home_fts")),
    "away_fts":          ("prob_away_fts",         lambda d: _derive_target_from_goals(d, kind="away_fts")),
    "ah_home_minus15":   ("prob_ah_home_minus15",  lambda d: _derive_target_from_goals(d, kind="ah_home_minus15")),
    "ah_home_minus25":   ("prob_ah_home_minus25",  lambda d: _derive_target_from_goals(d, kind="ah_home_minus25")),
}

def _cal_path(league_name: str, market: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(league_name)).strip("_")
    try:
        from constants import MODEL_DIR as __MDIR
    except Exception:
        __MDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "ModelStore"))
    return os.path.join(__MDIR, f"{safe}_{market}_cal.joblib")


def fit_and_save_calibrators(df: pd.DataFrame, league_name: str,
                             markets: tuple[str, ...] = _CAL_MARKET_DEFAULTS) -> dict:
    """Fit isotonic calibrators per market for a league and persist to ModelStore.
    Requires completed fixtures (with realized goal counts). Returns dict of saved paths.
    """
    saved: dict[str, str] = {}
    if _Iso is None or _joblib is None:
        print("⚠️ Isotonic/joblib unavailable; cannot fit calibrators.")
        return saved
    if df is None or df.empty:
        return saved
    for m in markets:
        prob_col, y_fn = _CAL_PROB_COLS.get(m, (None, None))
        if (prob_col is None) or (prob_col not in df.columns) or (y_fn is None):
            continue
        y = y_fn(df)
        if y is None or y.isna().all():
            continue
        y = pd.to_numeric(y, errors="coerce")
        p = pd.to_numeric(df[prob_col], errors="coerce").clip(0, 1)
        mask = (~p.isna()) & (~y.isna())
        p, y = p[mask], y[mask].astype(int)
        # need both classes
        if y.nunique() < 2 or len(y) < 100:
            continue
        try:
            iso = _Iso(out_of_bounds="clip")
            iso.fit(p.values, y.values)
            path = _cal_path(league_name, m)
            _joblib.dump(iso, path)
            saved[m] = path
        except Exception:
            continue
    return saved



def apply_saved_calibrators(df: pd.DataFrame, league_name: str) -> pd.DataFrame:
    """Apply isotonic calibrators if present; overwrite prob columns in place.
    Safe no-op if calibrators are missing or libraries unavailable.
    """
    if _Iso is None or _joblib is None:
        return df
    if df is None or df.empty:
        return df
    out: pd.DataFrame = df.copy()
    for m, (prob_col, _) in _CAL_PROB_COLS.items():
        if prob_col not in out.columns:
            continue
        path = _cal_path(league_name, m)
        if not os.path.exists(path):
            continue
        try:
            iso = _joblib.load(path)
            p = pd.to_numeric(out[prob_col], errors="coerce").clip(0, 1)
            out[prob_col] = pd.Series(iso.predict(p.fillna(0.5).values), index=out.index).clip(0, 1)
        except Exception:
            continue
    # Mark the frame as calibrated so candidate builder can prefer prob_* first
    out.attrs["calibrated"] = True
    return out

# ----------------------------------------------------------------------------
# "Decisive Overs" scorer (Over 2.5 / BTTS) — slate-normalized, rule-aware
# ----------------------------------------------------------------------------
def _minmax01(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mn, mx = float(s.min(skipna=True)), float(s.max(skipna=True))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return pd.Series(0.5, index=s.index, dtype="float64")
    return ((s - mn) / (mx - mn)).clip(0, 1)

def _resolve_implied(df: pd.DataFrame, market_key: str, *, label: str) -> pd.Series:
    """Resolve implied prob 1/odds for a key present in ODD_COLUMN_WHITELIST."""
    try:
        col = _resolve_odds(df, ODD_COLUMN_WHITELIST.get(market_key, []), label=label)
        if isinstance(col, str) and col in df.columns:
            o = pd.to_numeric(df[col], errors="coerce")
            return (1.0 / o.replace(0, np.nan)).astype("float64")
    except Exception:
        pass
    return pd.Series(np.nan, index=df.index, dtype="float64")

def attach_decisive_over_btts_scores(df: pd.DataFrame, league_name: str | None = None) -> pd.DataFrame:
    """
    Compute Decisive Over 2.5 / BTTS scores (0..100) using only signals we already have:
    - O2.5 / BTTS implied from odds, 1X2 implied favourite strength
    - pre-match xG (home+away) & pace proxy
    - pre-match O2.5% / BTTS%
    - drawish trap from model draw confidence (confidence_draw)
    Writes: decisive_over_score, decisive_btts_score, decisive_grade (A/A-/B/Pass).
    """
    out = df.copy()

    # Odds → implied
    o25_implied   = _resolve_implied(out, "over25", label="Over 2.5 odds")
    btts_implied  = _resolve_implied(out, "btts_yes", label="BTTS YES odds")

    # 1X2 → favourite strength
    pH = _resolve_implied(out, "home_win", label="FTR HOME odds")
    pD = _resolve_implied(out, "draw",     label="FTR DRAW odds")
    pA = _resolve_implied(out, "away_win", label="FTR AWAY odds")
    probs = pd.concat([pH.rename("H"), pD.rename("D"), pA.rename("A")], axis=1)
    fav_strength = probs.max(axis=1) - probs.apply(lambda r: np.partition(r.values, -2)[-2] if r.notna().sum()>=2 else np.nan, axis=1)
    fav_strength = fav_strength.clip(0, 1).fillna(0.0)

    # Pre-match xG and pace proxies
    xg_home = pd.to_numeric(out.get("pre_match_xg_home"), errors="coerce")
    xg_away = pd.to_numeric(out.get("pre_match_xg_away"), errors="coerce")
    xg_sum  = (xg_home.fillna(0) + xg_away.fillna(0)).astype("float64")
    pace_index = pd.to_numeric(out.get("average_goals_per_match_pre_match"), errors="coerce")

    # Slate normalization
    xg_sum_norm     = _minmax01(xg_sum)
    pace_index_norm = _minmax01(pace_index)
    att_index_norm  = xg_sum_norm           # proxy
    def_leak_norm   = pace_index_norm       # proxy

    # Team priors (fixture-level columns)
    def _rate01(col):
        val = out.get(col)
        if val is None:
            return pd.Series(np.nan, index=out.index, dtype="float64")
        v = pd.to_numeric(val, errors="coerce")
        v = v / 100.0 if (pd.notna(v.max()) and float(v.max()) > 1.5) else v
        return v.clip(0, 1)

    o25_team_prior  = _rate01("over_25_percentage_pre_match")
    btts_team_prior = _rate01("btts_percentage_pre_match")

    # Drawish trap via model draw confidence
    try:
        draw_trap_min = float(os.getenv("DRAW_TRAP_MIN", "0.35"))
    except Exception:
        draw_trap_min = 0.35
    cD = pd.to_numeric(out.get("confidence_draw"), errors="coerce")
    drawish_trap_flag = (cD >= draw_trap_min).astype(int).fillna(0)

    # "Decisive over": strong over price and not draw-friendly
    decisive_over_flag = ((o25_implied >= 0.62) & (drawish_trap_flag.eq(0))).astype(int)

    # Weights (env-overridable)
    def _w(name, dflt):
        try:
            return float(os.getenv(name, str(dflt)))
        except Exception:
            return float(dflt)

    w = {
        "O_O25_IMP": _w("W_O_O25_IMP", 18), "O_DECISIVE": _w("W_O_DECISIVE", 14),
        "O_PRIOR":  _w("W_O_PRIOR", 12),    "O_XG":       _w("W_O_XG", 10),
        "O_ATT":    _w("W_O_ATT", 8),       "O_DEF":      _w("W_O_DEF", 8),
        "O_PACE":   _w("W_O_PACE", 6),      "O_H2H":      _w("W_O_H2H", 4),
        "O_PREV":   _w("W_O_PREV", 4),      "O_FAV":      _w("W_O_FAV", 6),
        "O_TRAP":   _w("W_O_TRAP", -12),

        "B_IMP":  _w("W_B_IMP", 16),  "B_PRIOR": _w("W_B_PRIOR", 14),
        "B_BAL":  _w("W_B_BAL", 10),  "B_DEF":   _w("W_B_DEF", 8),
        "B_H2H":  _w("W_B_H2H", 6),   "B_PACE":  _w("W_B_PACE", 6),
        "B_PREV": _w("W_B_PREV", 5),  "B_XG":    _w("W_B_XG", 4),
        "B_FAV":  _w("W_B_FAV", -6),  "B_CS":    _w("W_B_CS", -6),
    }

    # (Optional) H2H placeholders
    h2h_o25_prior  = _rate01("h2h_o25_prior")  if "h2h_o25_prior"  in out.columns else pd.Series(0.0, index=out.index)
    h2h_btts_prior = _rate01("h2h_btts_prior") if "h2h_btts_prior" in out.columns else pd.Series(0.0, index=out.index)

    # BTTS: balance & CS bias
    def _sigmoid(x): 
        try: return 1.0 / (1.0 + np.exp(-x))
        except Exception: return 0.5
    att_balance = 1.0 - _sigmoid(2.0 * fav_strength.fillna(0.0))
    p_h_fts = pd.to_numeric(out.get("p_home_fts"), errors="coerce")
    p_a_fts = pd.to_numeric(out.get("p_away_fts"), errors="coerce")
    clean_sheet_bias = pd.concat([p_h_fts, p_a_fts], axis=1).max(axis=1).fillna(0.0)

    # No external previews in data yet → zeros
    preview_goal_sum_norm = pd.Series(0.0, index=out.index, dtype="float64")

    # Scores
    over_score = (
        w["O_O25_IMP"] * o25_implied.fillna(0) + w["O_DECISIVE"] * decisive_over_flag +
        w["O_PRIOR"]   * o25_team_prior.fillna(0) + w["O_XG"] * xg_sum_norm.fillna(0) +
        w["O_ATT"]     * att_index_norm.fillna(0) + w["O_DEF"] * def_leak_norm.fillna(0) +
        w["O_PACE"]    * pace_index_norm.fillna(0) + w["O_H2H"] * h2h_o25_prior.fillna(0) +
        w["O_PREV"]    * preview_goal_sum_norm + w["O_FAV"] * fav_strength.fillna(0) +
        w["O_TRAP"]    * drawish_trap_flag
    )
    over_score = np.where(xg_sum_norm.fillna(0) < 0.35, np.minimum(over_score, 65.0), over_score)
    over_score = np.clip(over_score, 0, 100)

    btts_score = (
        w["B_IMP"] * btts_implied.fillna(0) + w["B_PRIOR"] * btts_team_prior.fillna(0) +
        w["B_BAL"] * att_balance + w["B_DEF"] * def_leak_norm.fillna(0) +
        w["B_H2H"] * h2h_btts_prior.fillna(0) + w["B_PACE"] * pace_index_norm.fillna(0) +
        w["B_PREV"] * preview_goal_sum_norm + w["B_XG"] * xg_sum_norm.fillna(0) +
        w["B_FAV"] * fav_strength.fillna(0) + w["B_CS"] * clean_sheet_bias.fillna(0)
    )
    btts_score = np.clip(btts_score, 0, 100)

    out["decisive_over_score"] = pd.Series(over_score, index=out.index, dtype="float64")
    out["decisive_btts_score"] = pd.Series(btts_score, index=out.index, dtype="float64")

    # Grade from OverScore
    g = out["decisive_over_score"]
    grade = np.where(g >= 72, "A", np.where(g >= 66, "A-", np.where(g >= 60, "B", "Pass")))
    # downgrade if drawish trap and A/A-
    grade = np.where(((pd.to_numeric(drawish_trap_flag) == 1) & (g >= 66)),
                     np.where(g >= 72, "A-", "B"), grade)
    out["decisive_grade"] = grade
    return out
# === Candidate preparation with EV/edge gates + decisive/threshold fallbacks ===
def _prepare_candidates_legacy(df: pd.DataFrame, market: str, min_edge: float, max_odds: float | None = None) -> pd.DataFrame:
    """LEGACY: superseded by the guarded _prepare_candidates below; retained only for reference.

    Build an EV‑ranked candidate pool for a given market.
    - Computes per‑leg EV = p*odds − 1 and edge = p − 1/odds
    - Applies EV_MIN (env) and min_edge
    - Optional CLV_MIN (env) filter per market
    - Optional max_odds cap
    - If empty, falls back to probability thresholds (env/attrs) and “Decisive” score gates

    Returns a DataFrame tagged with attrs: _prob_col, _odds_col and sorted by EV desc.
    """
    import os as _os
    m = str(market).lower()
    df = df.copy()
    df0 = df.copy()  # keep a copy for fallbacks

    # ---- Map market → probability/odds columns (robust fallbacks) ----
    prob_col: Optional[str] = None
    ocol: Optional[str] = None

    # Helper: choose first present column from a list
    def _first_present(cols):
        for c in cols:
            if c in df.columns:
                return c
        return None

    if m == "over25":
        # Prefer calibrated prob_ if frame is flagged as calibrated
        if bool(df.attrs.get("calibrated")) and "prob_over25" in df.columns:
            prob_col = "prob_over25"
        else:
            prob_col = _first_present(["adjusted_over25_confidence","over25_confidence","prob_over25"])
        ocol = _first_present(["odds_over25","odds_ft_over25","odds_ft_over_25","over25_odds"]) or "odds_over25"

    elif m == "btts":
        # Robust probability column fallback for BTTS YES
        if "prob_btts" in df.columns:
            prob_col = "prob_btts"
        elif "adjusted_btts_confidence" in df.columns:
            prob_col = "adjusted_btts_confidence"
        elif "btts_confidence" in df.columns:
            prob_col = "btts_confidence"
        else:
            prob_col = None
        # Odds column for BTTS YES (prefer market, then implied, then existing 'od')
        ocol = _first_present(["odds_btts_yes","imp_odds_btts_yes","od"]) or "odds_btts_yes"

    elif m == "ftr_home":
        prob_col = _first_present([
            "prob_ftr_home","confidence_home","p_home","home_win_proba"
        ])
        # If we only have a single selected_confidence + selector, create a temp filtered column
        if prob_col is None and ("selected_confidence" in df.columns and "selected_outcome" in df.columns):
            tmp = pd.to_numeric(df["selected_confidence"], errors="coerce")
            mask = df["selected_outcome"].astype(str).str.lower().isin(["home","h","1","home win","home_win"])
            df["__tmp_prob_home"] = tmp.where(mask, np.nan)
            prob_col = "__tmp_prob_home"
        if prob_col is None and ("selected_confidence" in df.columns and "ftr_pred_outcome" in df.columns):
            tmp = pd.to_numeric(df["selected_confidence"], errors="coerce")
            mask = pd.to_numeric(df["ftr_pred_outcome"], errors="coerce").astype("Int64") == 0
            df["__tmp_prob_home"] = tmp.where(mask, np.nan)
            prob_col = "__tmp_prob_home"
        ocol = _first_present([
            "odds_ft_home_team_win","odds_home_win","odds_ft_home_win","home_win_odds",
            "imp_odds_ft_home_team_win"
        ]) or "odds_ft_home_team_win"

    elif m == "ftr_draw":
        prob_col = _first_present([
            "prob_ftr_draw","confidence_draw","p_draw","draw_win_proba"
        ])
        if prob_col is None and ("selected_confidence" in df.columns and "selected_outcome" in df.columns):
            tmp = pd.to_numeric(df["selected_confidence"], errors="coerce")
            mask = df["selected_outcome"].astype(str).str.lower().isin(["draw","d","x"])
            df["__tmp_prob_draw"] = tmp.where(mask, np.nan)
            prob_col = "__tmp_prob_draw"
        if prob_col is None and ("selected_confidence" in df.columns and "ftr_pred_outcome" in df.columns):
            tmp = pd.to_numeric(df["selected_confidence"], errors="coerce")
            mask = pd.to_numeric(df["ftr_pred_outcome"], errors="coerce").astype("Int64") == 1
            df["__tmp_prob_draw"] = tmp.where(mask, np.nan)
            prob_col = "__tmp_prob_draw"
        ocol = _first_present(["odds_ft_draw","odds_draw","imp_odds_ft_draw"]) or "odds_ft_draw"

    elif m == "ftr_away":
        prob_col = _first_present([
            "prob_ftr_away","confidence_away","p_away","away_win_proba"
        ])
        if prob_col is None and ("selected_confidence" in df.columns and "selected_outcome" in df.columns):
            tmp = pd.to_numeric(df["selected_confidence"], errors="coerce")
            mask = df["selected_outcome"].astype(str).str.lower().isin(["away","a","2","away win","away_win"])
            df["__tmp_prob_away"] = tmp.where(mask, np.nan)
            prob_col = "__tmp_prob_away"
        if prob_col is None and ("selected_confidence" in df.columns and "ftr_pred_outcome" in df.columns):
            tmp = pd.to_numeric(df["selected_confidence"], errors="coerce")
            mask = pd.to_numeric(df["ftr_pred_outcome"], errors="coerce").astype("Int64") == 2
            df["__tmp_prob_away"] = tmp.where(mask, np.nan)
            prob_col = "__tmp_prob_away"
        ocol = _first_present([
            "odds_ft_away_team_win","odds_away_win","odds_ft_away_win","away_win_odds",
            "imp_odds_ft_away_team_win"
        ]) or "odds_ft_away_team_win"

    elif m == "ah_home_minus15":
        prob_col = "prob_ah_home_minus15" if "prob_ah_home_minus15" in df.columns else None
        ocol = _first_present(["odds_ah_home_minus15","home_ah_-1_5","ah_home_-1_5","odds_home_ah_-1_5"])
    elif m == "ah_home_minus25":
        prob_col = "prob_ah_home_minus25" if "prob_ah_home_minus25" in df.columns else None
        ocol = _first_present(["odds_ah_home_minus25","home_ah_-2_5","ah_home_-2_5","odds_home_ah_-2_5"])

    elif m == "under25":
        # Prob from explicit under25, else invert over25/adjusted_over25
        if "prob_under25" in df.columns:
            prob_col = "prob_under25"
        elif "prob_over25" in df.columns:
            df["__prob_under25_from_over"] = 1.0 - pd.to_numeric(df["prob_over25"], errors="coerce")
            prob_col = "__prob_under25_from_over"
        elif "adjusted_over25_confidence" in df.columns:
            df["__prob_under25_from_adj"] = 1.0 - pd.to_numeric(df["adjusted_over25_confidence"], errors="coerce")
            prob_col = "__prob_under25_from_adj"
        else:
            prob_col = None
        # Prefer real under odds; otherwise accept implied or synthesized 'od'
        ocol = _first_present([
            "odds_ft_under25","odds_under25","odds_ft_u25",
            "odds_under_2_5","under_2.5_odds","imp_odds_under25","od"
        ]) or "od"

    elif m == "btts_no":
        # Prob from explicit btts_no, else invert btts/adjusted_btts
        if "prob_btts_no" in df.columns:
            prob_col = "prob_btts_no"
        elif "prob_btts" in df.columns:
            df["__prob_btts_no_from_yes"] = 1.0 - pd.to_numeric(df["prob_btts"], errors="coerce")
            prob_col = "__prob_btts_no_from_yes"
        elif "adjusted_btts_confidence" in df.columns:
            df["__prob_btts_no_from_adj"] = 1.0 - pd.to_numeric(df["adjusted_btts_confidence"], errors="coerce")
            prob_col = "__prob_btts_no_from_adj"
        else:
            prob_col = None
        ocol = _first_present(["odds_btts_no","btts_no_odds","imp_odds_btts_no","od"]) or "od"

    elif m == "wtn_home":
        # Home Win-To-Nil from proxy probs (attach_win_to_nil_proxy)
        prob_col = "prob_home_win_to_nil" if "prob_home_win_to_nil" in df.columns else None
        ocol = _first_present(["odds_home_win_to_nil","odds_home_to_nil","home_win_to_nil_odds"]) or "od"

    elif m == "wtn_away":
        # Away Win-To-Nil from proxy probs (attach_win_to_nil_proxy)
        prob_col = "prob_away_win_to_nil" if "prob_away_win_to_nil" in df.columns else None
        ocol = _first_present(["odds_away_win_to_nil","odds_away_to_nil","away_win_to_nil_odds"]) or "od"

    else:
        raise ValueError(f"Unknown market {market}")

    # Bail out early if we still don't have a probability column
    if prob_col is None or prob_col not in df.columns:
        return pd.DataFrame(columns=["Date","Time","League","Home","Away","ev","edge"])

    # ---- Optional CLV filter per market (env: CLV_MIN) ----
    try:
        clv_min_raw = _os.getenv("CLV_MIN", None)
        clv_min: float | None = float(clv_min_raw) if clv_min_raw is not None else None
        if clv_min is not None:
            clv_map = {
                "over25": "clv_over25",
                "btts": "clv_btts",
                "ftr_home": "clv_ftr_home",
                "ftr_draw": "clv_ftr_draw",
                "ftr_away": "clv_ftr_away",
                "ah_home_minus15": "clv_ah_home_minus15",
                "ah_home_minus25": "clv_ah_home_minus25",
            }
            ccol = clv_map.get(m)
            if isinstance(ccol, str) and ccol in df.columns:
                df = df[pd.to_numeric(df[ccol], errors="coerce") >= clv_min].copy()
    except Exception:
        pass

    # ---- Compute EV + edge ----
    try:
        df["ev"]   = pd.to_numeric(df[prob_col], errors="coerce") * pd.to_numeric(df[ocol], errors="coerce") - 1.0
        df["edge"] = pd.to_numeric(df[prob_col], errors="coerce") - (1.0 / pd.to_numeric(df[ocol], errors="coerce").replace(0, np.nan))
    except Exception:
        return pd.DataFrame(columns=["Date","Time","League","Home","Away","ev","edge"])

    # --- Ensure usable odds: synthesize if allowed, then drop non-positive
    try:
        _odv = pd.to_numeric(df[ocol], errors="coerce")
        if (_odv.le(0).all() or _odv.isna().all()) and str(_os.getenv("ALLOW_SYNTH_ODDS","0")).strip().lower() in ("1","true","yes","y","on"):
            syn = (1.0 / pd.to_numeric(df[prob_col], errors="coerce").replace(0, np.nan)).clip(lower=1.01)
            df[ocol] = syn
            df["ev"]   = pd.to_numeric(df[prob_col], errors="coerce") * syn - 1.0
            df["edge"] = pd.to_numeric(df[prob_col], errors="coerce") - (1.0 / syn)
        # finally drop rows with non-positive odds
        df = df[pd.to_numeric(df[ocol], errors="coerce") > 0].copy()
    except Exception:
        pass

    # ---- EV_MIN gate + min_edge ----
    try:
        _ev_min = float(_os.getenv("EV_MIN", "0.0"))
    except Exception:
        _ev_min = 0.0

    if max_odds is not None:
        try:
            df = df[pd.to_numeric(df[ocol], errors="coerce") <= float(max_odds)]
        except Exception:
            pass

    mask_ev   = pd.to_numeric(df["ev"], errors="coerce")   >= _ev_min
    mask_edge = pd.to_numeric(df["edge"], errors="coerce") >= float(min_edge)
    df = df[mask_ev & mask_edge].copy()

    # ---- Fallback: probability thresholds + decisive score gates if empty ----
    if df.empty:
        try:
            import os as _os
            # env-first fallback precedence
            thr = None
            if m == "over25":
                thr = float(_os.getenv(
                    "FALLBACK_OVER25_PROB_MIN",
                    str(df0.attrs.get("thr_over25", _os.getenv("FALLBACK_PROB_MIN", "0")))
                ))
            elif m == "under25":
                thr = float(_os.getenv(
                    "FALLBACK_UNDER25_PROB_MIN",
                    str(df0.attrs.get("thr_under25", _os.getenv("FALLBACK_PROB_MIN", "0")))
                ))
            elif m == "btts":
                thr = float(_os.getenv(
                    "FALLBACK_BTTS_PROB_MIN",
                    str(df0.attrs.get("thr_btts", _os.getenv("FALLBACK_PROB_MIN", "0")))
                ))
            elif m == "btts_no":
                thr = float(_os.getenv(
                    "FALLBACK_BTTS_NO_PROB_MIN",
                    str(df0.attrs.get("thr_btts_no", _os.getenv("FALLBACK_PROB_MIN", "0")))
                ))
            elif m in ("ftr_home","ftr_draw","ftr_away"):
                thr = float(_os.getenv(
                    "FALLBACK_FTR_PROB_MIN",
                    str(df0.attrs.get("thr_ftr", _os.getenv("FALLBACK_PROB_MIN", "0")))
                ))

            # select a safe prob column for mask and EV if primary is missing
            pc: Optional[str] = prob_col
            if pc is None:
                if m == "btts":
                    if "prob_btts" in df0.columns: pc = "prob_btts"
                    elif "adjusted_btts_confidence" in df0.columns: pc = "adjusted_btts_confidence"
                    elif "btts_confidence" in df0.columns: pc = "btts_confidence"
                elif m == "btts_no":
                    if "prob_btts_no" in df0.columns: pc = "prob_btts_no"
                    elif "__prob_btts_no_from_yes" in df0.columns: pc = "__prob_btts_no_from_yes"
                    elif "__prob_btts_no_from_adj" in df0.columns: pc = "__prob_btts_no_from_adj"
                elif m == "over25":
                    if "prob_over25" in df0.columns: pc = "prob_over25"
                    elif "adjusted_over25_confidence" in df0.columns: pc = "adjusted_over25_confidence"
                    elif "over25_confidence" in df0.columns: pc = "over25_confidence"
                elif m == "under25":
                    if "prob_under25" in df0.columns: pc = "prob_under25"
                    elif "__prob_under25_from_over" in df0.columns: pc = "__prob_under25_from_over"
                    elif "__prob_under25_from_adj" in df0.columns: pc = "__prob_under25_from_adj"
                elif m == "ftr_home":
                    if "prob_ftr_home" in df0.columns: pc = "prob_ftr_home"
                    elif "confidence_home" in df0.columns: pc = "confidence_home"
                elif m == "ftr_draw":
                    if "prob_ftr_draw" in df0.columns: pc = "prob_ftr_draw"
                    elif "confidence_draw" in df0.columns: pc = "confidence_draw"
                elif m == "ftr_away":
                    if "prob_ftr_away" in df0.columns: pc = "prob_ftr_away"
                    elif "confidence_away" in df0.columns: pc = "confidence_away"

            # Build probability mask with the best available column
            _fallback_df = pd.DataFrame(index=df0.index)
            _pcol_for_mask: Optional[str] = prob_col if (prob_col and prob_col in df0.columns) else pc

            # Fallback ladders per market if resolver returns None
            if thr is not None and _pcol_for_mask is not None and _pcol_for_mask in df0.columns:
                _pmask = pd.to_numeric(df0[_pcol_for_mask], errors="coerce") >= _to_float(thr)
                _fallback_df = df0.loc[_pmask].copy()

                # Decisive score OR-gates for over25 / btts are handled elsewhere if present

                if not _fallback_df.empty:
                    try:
                        print(f"ℹ️ fallback threshold used for {m}: thr={_to_float(thr)}")
                    except Exception:
                        pass
                    if "ev" not in _fallback_df.columns or "edge" not in _fallback_df.columns:
                        try:
                            if (_pcol_for_mask is not None) and (_pcol_for_mask in _fallback_df.columns) and (ocol in _fallback_df.columns):
                                _fallback_df["ev"] = (
                                    pd.to_numeric(_fallback_df[_pcol_for_mask], errors="coerce")
                                    * pd.to_numeric(_fallback_df[ocol], errors="coerce")
                                    - 1.0
                                )
                                _fallback_df["edge"] = (
                                    pd.to_numeric(_fallback_df[_pcol_for_mask], errors="coerce")
                                    - (1.0 / pd.to_numeric(_fallback_df[ocol], errors="coerce").replace(0, np.nan))
                                )
                        except Exception:
                            pass
                    # Drop rows with non-positive odds; canonicalize 'od' for writer
                    try:
                        if ocol in _fallback_df.columns:
                            _odv = pd.to_numeric(_fallback_df[ocol], errors="coerce")
                            _fallback_df = _fallback_df[_odv > 0].copy()
                            if "od" not in _fallback_df.columns:
                                _fallback_df["od"] = _odv
                    except Exception:
                        pass
                    df = _fallback_df
                    if _pcol_for_mask is not None:
                        prob_col = cast(str, _pcol_for_mask)
        except Exception:
            pass

    # --- Canonicalize for composer/writer: ensure generic p_model/od exist ---
    try:
        _curp = pd.to_numeric(df.get("p_model"), errors="coerce") if "p_model" in df.columns else None
        if (_curp is None) or (_curp.notna().sum() == 0):
            if prob_col and prob_col in df.columns:
                df["p_model"] = pd.to_numeric(df[prob_col], errors="coerce")
    except Exception:
        pass
    try:
        _curod = pd.to_numeric(df.get("od"), errors="coerce") if "od" in df.columns else None
        if (_curod is None) or (_curod.le(0).all() or _curod.isna().all()):
            if ocol and ocol in df.columns:
                df["od"] = pd.to_numeric(df[ocol], errors="coerce")
                df["od_source"] = "market"
    except Exception:
        pass
    df.attrs["_prob_col"] = prob_col
    df.attrs["_odds_col"] = ocol
    df.attrs["_od"]       = "od"
    return df.sort_values(["ev","edge",prob_col], ascending=[False, False, False])
# ----------------------------------------------------------------------------
# Walk-forward reporting: Accuracy / LogLoss / Brier / ECE by time period
# ----------------------------------------------------------------------------

def time_split_report(df: pd.DataFrame, league_name: str,
                      markets: tuple[str, ...] = ("ftr","btts","over25","ah15","ah25"),
                      *,
                      freq: str = "M",  # 'M' monthly; 'W' weekly
                      n_bins: int = 10) -> dict:
    """
    Compute Accuracy / LogLoss / Brier / ECE for selected markets over time slices
    (e.g., by month), write a CSV under predictions_output/<date>/reports/ and
    return a summary dict.

    Safe no-op when realized goals are missing for the chosen markets.
    """
    import os, datetime
    out_summary: dict[str, Any] = {"league": league_name, "markets": {}, "path": None}
    if df is None or df.empty:
        return out_summary

    # Require a parseable match_date to group by time period
    md = pd.to_datetime(df.get("match_date"), errors="coerce") if "match_date" in df.columns else None
    if md is None or md.notna().sum() == 0:
        return out_summary
    period = md.dt.to_period(freq).astype(str)

    # Helper: reliability / ECE for binary
    def _ece_binary(p: pd.Series, y: pd.Series, bins: int = n_bins) -> float:
        try:
            p = pd.to_numeric(p, errors="coerce").clip(0,1)
            y = pd.to_numeric(y, errors="coerce").astype(int)
            edges = np.linspace(0, 1, bins+1)
            idx = np.digitize(p.fillna(0.5), edges, right=True)
            ece = 0.0
            n = len(p)
            for b in range(1, bins+1):
                mask = (idx == b)
                if not bool(np.any(mask)):
                    continue
                pb = float(p[mask].mean())
                yb = float(y[mask].mean())
                ece += (mask.mean()) * abs(pb - yb)
            return float(ece)
        except Exception:
            return float("nan")

    # Helper: logloss/brier for binary
    def _metrics_binary(p: pd.Series, y: pd.Series) -> dict:
        eps = 1e-9
        p = pd.to_numeric(p, errors="coerce").clip(0,1).fillna(0.5)
        y = pd.to_numeric(y, errors="coerce").astype(int)
        acc = float(((p >= 0.5) == (y == 1)).mean())
        ll  = float((- (y*np.log(p+eps) + (1-y)*np.log(1-p+eps))).mean())
        br  = float(((p - y)**2).mean())
        ece = _ece_binary(p, y, n_bins)
        return {"acc":acc, "logloss":ll, "brier":br, "ece":ece}

    # Helper: multiclass (FTR)
    def _metrics_ftr(ph: pd.Series, pdw: pd.Series, pa: pd.Series, y_int: pd.Series) -> dict:
        eps = 1e-9
        try:
            ph = pd.to_numeric(ph, errors="coerce").clip(0,1).fillna(1/3)
            pdw = pd.to_numeric(pdw, errors="coerce").clip(0,1).fillna(1/3)
            pa = pd.to_numeric(pa, errors="coerce").clip(0,1).fillna(1/3)
            tot = (ph + pdw + pa).replace(0, np.nan)
            ph, pdw, pa = (ph/tot).fillna(1/3), (pdw/tot).fillna(1/3), (pa/tot).fillna(1/3)
            y = pd.to_numeric(y_int, errors="coerce").astype(int)
            acc = float((np.argmax(np.vstack([ph, pdw, pa]).T, axis=1) == y).mean())
            # multiclass logloss
            onehot = np.zeros((len(y), 3), dtype=float)
            rows = (~y.isna()) & (y.between(0,2))
            onehot[np.where(rows)[0], y[rows].values] = 1.0
            probs = np.vstack([ph, pdw, pa]).T
            ll = float((- (onehot*np.log(probs + eps))).sum(axis=1).mean())
            br = float(((probs - onehot)**2).sum(axis=1).mean())
            # simple ECE: bins on max prob vs correctness
            pmax = probs.max(axis=1)
            yhat = probs.argmax(axis=1)
            correct = (yhat == y.values)
            ece = _ece_binary(pd.Series(pmax), pd.Series(correct.astype(int)), bins=n_bins)
            return {"acc":acc, "logloss":ll, "brier":br, "ece":ece}
        except Exception:
            return {"acc": float("nan"), "logloss": float("nan"), "brier": float("nan"), "ece": float("nan")}

    rows = []
    # Precompute targets from realized goals if present
    has_goals = {"home_team_goal_count","away_team_goal_count"}.issubset(df.columns)
    if not has_goals:
        return out_summary

    # FTR targets
    hg = pd.to_numeric(df["home_team_goal_count"], errors="coerce")
    ag = pd.to_numeric(df["away_team_goal_count"], errors="coerce")
    y_ftr = (hg - ag).apply(lambda d: 0 if d>0 else (2 if d<0 else 1))  # 0 H,1 D,2 A

    # Market → (prob columns or function)
    market_map = {
        "ftr":  ("confidence_home","confidence_draw","confidence_away"),
        "btts": ("prob_btts", "btts"),
        "over25": ("prob_over25", "over25"),
        "ah15": ("prob_ah_home_minus15", "ah_home_minus15"),
        "ah25": ("prob_ah_home_minus25", "ah_home_minus25"),
    }

    # Iterate by period and compute metrics
    for mkt in markets:
        try:
            if mkt == "ftr":
                ph, pdw, pa = (df.get(market_map[mkt][0]), df.get(market_map[mkt][1]), df.get(market_map[mkt][2]))
                if ph is None or pdw is None or pa is None:
                    continue
                sub = pd.DataFrame({
                    "period": period, "ph": ph, "pd": pdw, "pa": pa, "y": y_ftr
                }).dropna()
                if sub.empty:
                    continue
                for g, gdf in sub.groupby("period"):
                    met = _metrics_ftr(gdf["ph"], gdf["pd"], gdf["pa"], gdf["y"])
                    rows.append({"period": g, "market": mkt, **met, "n": int(len(gdf))})
            else:
                prob_col, kind = market_map.get(mkt, (None, None))
                if (prob_col is None) or (prob_col not in df.columns) or (kind is None):
                    continue
                y = _derive_target_from_goals(df, kind=kind)
                sub = pd.DataFrame({"period": period, "p": df[prob_col], "y": y}).dropna()
                if sub.empty:
                    continue
                for g, gdf in sub.groupby("period"):
                    met = _metrics_binary(gdf["p"], gdf["y"])
                    rows.append({"period": g, "market": mkt, **met, "n": int(len(gdf))})
        except Exception:
            continue

    if not rows:
        return out_summary

    rep = pd.DataFrame(rows).sort_values(["period","market"]).reset_index(drop=True)
    # Write CSV
    dstdir = os.path.join("predictions_output", datetime.datetime.utcnow().strftime("%Y-%m-%d"), "reports")
    try:
        os.makedirs(dstdir, exist_ok=True)
        safe = __re.sub(r"[^A-Za-z0-9_]+","_", str(league_name)).strip("_")
        path = os.path.join(dstdir, f"{safe}_walkforward_metrics.csv")
        rep.to_csv(path, index=False)
        out_summary["path"] = path
    except Exception:
        pass

    # Market summaries (overall means weighted by n) — avoid groupby.apply warning
    try:
        rows = []
        for mkt, d in rep.groupby("market"):
            try:
                rows.append({
                    "market": mkt,
                    "acc": float(np.average(d["acc"], weights=d["n"])),
                    "logloss": float(np.average(d["logloss"], weights=d["n"])),
                    "brier": float(np.average(d["brier"], weights=d["n"])),
                    "ece": float(np.average(d["ece"], weights=d["n"])),
                    "n": int(d["n"].sum()),
                })
            except Exception:
                continue
        if rows:
            agg = pd.DataFrame(rows)
            out_summary["markets"] = agg.to_dict(orient="records")
    except Exception:
        pass

    return out_summary

# ----------------------------------------------------------------------------
# Plot helpers (matplotlib only): time-series & reliability curves
# ----------------------------------------------------------------------------

def _reports_dir() -> str:
    import os, datetime
    dstdir = os.path.join("predictions_output", datetime.datetime.utcnow().strftime("%Y-%m-%d"), "reports")
    os.makedirs(dstdir, exist_ok=True)
    return dstdir


# --- Confidence/coverage summary helper ---
def summarize_confidence_and_write(df: pd.DataFrame, league_name: str, out_dir: str | None = None) -> dict | None:
    """
    Lightweight confidence/coverage snapshot writer.
    Writes a markdown summary and a details CSV when REPORT_CONFIDENCE=1.
    Returns a small dict of paths or None if disabled.
    """
    import os, re
    if str(os.getenv("REPORT_CONFIDENCE", "0")).strip() != "1":
        return None
    if df is None or df.empty:
        return None
    try:
        os.makedirs(out_dir or _reports_dir(), exist_ok=True)
    except Exception:
        pass
    dstdir = out_dir or _reports_dir()

    # Resolve thresholds
    try: over_min = float(os.getenv("OVER_SCORE_MIN", "66"))
    except Exception: over_min = 66.0
    try: btts_min = float(os.getenv("BTTS_SCORE_MIN", "60"))
    except Exception: btts_min = 60.0

    # Decide league tag
    safe_tag = re.sub(r"[^A-Za-z0-9_]+", "_", str(league_name)).strip("_") or "league"

    # Build gates
    o_scores = pd.to_numeric(df.get("decisive_over_score"), errors="coerce")
    b_scores = pd.to_numeric(df.get("decisive_btts_score"), errors="coerce")
    over_ok  = (o_scores >= over_min) if o_scores is not None else pd.Series(False, index=df.index)
    btts_ok  = (b_scores >= btts_min) if b_scores is not None else pd.Series(False, index=df.index)

    n_total = int(len(df))
    n_over  = int(over_ok.fillna(False).sum())
    n_btts  = int(btts_ok.fillna(False).sum())

    pct = lambda n: (100.0 * n / max(n_total, 1))

    # Try to surface any market probability thresholds present in attrs
    thr_over = df.attrs.get("thr_over25", None)
    thr_btts = df.attrs.get("thr_btts", None)

    # Markdown summary
    summary_lines = [
        f"# {league_name} — Confidence Coverage",
        "",
        f"- Rows analysed: **{n_total}**",
        f"- Decisive Over≥{over_min:.0f}: **{n_over}** ({pct(n_over):.1f}%)",
        f"- Decisive BTTS≥{btts_min:.0f}: **{n_btts}** ({pct(n_btts):.1f}%)",
    ]
    if thr_over is not None or thr_btts is not None:
        summary_lines.append("")
        summary_lines.append("**Probability thresholds (from attrs if present):**")
        if thr_over is not None:
            summary_lines.append(f"- thr_over25: {float(thr_over):.3f}")
        if thr_btts is not None:
            summary_lines.append(f"- thr_btts: {float(thr_btts):.3f}")

    # Details CSV (union of rows passing either decisive gate)
    try:
        mask_any = (over_ok.fillna(False) | btts_ok.fillna(False))
    except Exception:
        mask_any = pd.Series(False, index=df.index)
    cols_pref = [
        "Date","Time","League","Home","Away","__mkt",
        "p_model","od","ev","edge",
        "prob_over25","prob_btts",
        "decisive_over_score","decisive_btts_score","decisive_grade"
    ]
    keep = [c for c in cols_pref if c in df.columns]
    details = df.loc[mask_any, keep].copy() if keep else pd.DataFrame()

    # Write files
    paths = {}
    try:
        md_path = os.path.join(dstdir, f"{safe_tag}_confidence_summary.md")
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(summary_lines) + "\n")
        paths["summary_md"] = md_path
    except Exception:
        pass
    try:
        csv_path = os.path.join(dstdir, f"{safe_tag}_confidence_details.csv")
        if not details.empty:
            details.to_csv(csv_path, index=False)
        else:
            # write an empty stub with header if possible
            pd.DataFrame(columns=keep or ["info"]).to_csv(csv_path, index=False)
        paths["details_csv"] = csv_path
    except Exception:
        pass

    if paths:
        try:
            print(f"📊 Wrote confidence coverage → {dstdir}")
        except Exception:
            pass
    return paths


def plot_walkforward_timeseries(league_name: str,
                                *,
                                csv_path: str | None = None,
                                markets: tuple[str, ...] = ("ftr","btts","over25","ah15","ah25"),
                                metrics: tuple[str, ...] = ("acc","brier","ece","logloss")) -> dict:
    """
    Read the walk-forward CSV for `league_name` and plot per-metric time series.
    Saves PNGs to the same reports directory. Returns dict of {metric: path}.
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import re as _re

    out: dict[str, Any] = {}
    rep_dir = _reports_dir()
    if not csv_path:
        safe = _re.sub(r"[^A-Za-z0-9_]+","_", str(league_name)).strip("_")
        csv_path = os.path.join(rep_dir, f"{safe}_walkforward_metrics.csv")
    if not os.path.exists(csv_path):
        print(f"⚠️ walk-forward csv not found: {csv_path}")
        return out

    df = pd.read_csv(csv_path)
    if df.empty or not {"period","market"}.issubset(df.columns):
        return out

    # Ensure sorted periods for a clean line
    try:
        # try to sort Year-Month strings
        _p = pd.PeriodIndex(df["period"], freq="M")
        df = df.assign(_p=_p).sort_values(["_p","market"]).drop(columns=["_p"])
    except Exception:
        df = df.sort_values(["period","market"])  # fallback

    for metric in metrics:
        if metric not in df.columns:
            continue
        fig, ax = plt.subplots()
        for mkt in markets:
            sub = df[df["market"] == mkt]
            if sub.empty:
                continue
            ax.plot(sub["period"], sub[metric], marker="o", label=mkt)
        ax.set_xlabel("Period")
        ax.set_ylabel(metric)
        ax.set_title(f"{league_name} — {metric} by period")
        ax.legend()
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        out_png = os.path.join(rep_dir, f"{_re.sub(r'[^A-Za-z0-9_]+','_',league_name)}_timeseries_{metric}.png")
        try:
            fig.savefig(out_png, dpi=120)
            out[metric] = out_png
        except Exception:
            pass
        plt.close(fig)
    return out


def plot_reliability_curve(df: pd.DataFrame,
                           prob_col: str,
                           target_col: str,
                           *,
                           bins: int = 10,
                           out_path: str | None = None,
                           title: str | None = None) -> str | None:
    """
    Plot a reliability curve (mean predicted vs empirical) for a binary market
    from raw frame (requires realized goals target). Saves one PNG. Returns path or None.
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    p = pd.to_numeric(df.get(prob_col), errors="coerce").clip(0,1)
    y = pd.to_numeric(df.get(target_col), errors="coerce").astype(float)
    mask = (~p.isna()) & (~y.isna())
    p, y = p[mask], y[mask]
    if len(p) < 100:
        print("⚠️ insufficient rows for reliability curve (need ~100+)")
        return None

    edges = np.linspace(0,1,bins+1)
    idx = np.digitize(p, edges, right=True)
    x_hat, y_hat = [], []
    for b in range(1,bins+1):
        m = (idx == b)
        if not bool(np.any(m)):
            continue
        x_hat.append(float(p[m].mean()))
        y_hat.append(float(y[m].mean()))

    fig, ax = plt.subplots()
    ax.plot([0,1],[0,1], linestyle='--')
    ax.plot(x_hat, y_hat, marker='o')
    ax.set_xlabel('Mean predicted prob')
    ax.set_ylabel('Empirical rate')
    ax.set_title(title or f"Reliability: {prob_col}")
    plt.tight_layout()

    if out_path is None:
        out_path = os.path.join(_reports_dir(), f"reliability_{prob_col}.png")
    try:
        fig.savefig(out_path, dpi=120)
    except Exception:
        out_path = None
    plt.close(fig)
    return out_path


def plot_reliability_for_market(df: pd.DataFrame,
                                league_name: str,
                                *,
                                market: str = "btts",
                                bins: int = 10,
                                out_path: str | None = None) -> str | None:
    """Convenience: derive target from goals by market, select prob_col (with robust fallbacks), and plot reliability."""
    import os, re as _re
    import pandas as pd
    m = (market or "").lower()

    # Prefer the same resolver used elsewhere
    try:
        pc, _oc = _pick_prob_odds_for_market(df, m)
    except Exception:
        pc, _oc = (None, None)
    # Fallback ladders per market if resolver returns None
    if pc is None or pc not in df.columns:
        if m == "btts":
            for cand in ("prob_btts","adjusted_btts_confidence","btts_confidence","oof_prob_btts"):
                if cand in df.columns:
                    pc = cand; break
        elif m == "over25":
            for cand in ("prob_over25","adjusted_over25_confidence","over25_confidence"):
                if cand in df.columns:
                    pc = cand; break
        elif m == "under25":
            if "prob_under25" in df.columns:
                pc = "prob_under25"
            elif "prob_over25" in df.columns:
                df["__prob_under25_from_over"] = 1.0 - pd.to_numeric(df["prob_over25"], errors="coerce")
                pc = "__prob_under25_from_over"
            elif "adjusted_over25_confidence" in df.columns:
                df["__prob_under25_from_adj"] = 1.0 - pd.to_numeric(df["adjusted_over25_confidence"], errors="coerce")
                pc = "__prob_under25_from_adj"
        elif m == "btts_no":
            if "prob_btts_no" in df.columns:
                pc = "prob_btts_no"
            elif "prob_btts" in df.columns:
                df["__prob_btts_no_from_yes"] = 1.0 - pd.to_numeric(df["prob_btts"], errors="coerce")
                pc = "__prob_btts_no_from_yes"
            elif "adjusted_btts_confidence" in df.columns:
                df["__prob_btts_no_from_adj"] = 1.0 - pd.to_numeric(df["adjusted_btts_confidence"], errors="coerce")
                pc = "__prob_btts_no_from_adj"
        elif m in ("ah15","ah25"):
            # keep existing names if present
            _lookup = {"ah15":"prob_ah_home_minus15","ah25":"prob_ah_home_minus25"}
            cand = _lookup.get(m)
            if cand and cand in df.columns:
                pc = cand

    if pc is None or pc not in df.columns:
        print(f"⚠️ missing prob column for market={market}")
        return None
    from typing import cast
    prob_col = cast(str, pc)
    from typing import cast
    prob_col = cast(str, pc)
    # Target kind mapping
    kind_map = {
        "btts": "btts",
        "over25": "over25",
        "under25": "under25",
        "btts_no": "btts_no",
        "ah15": "ah_home_minus15",
        "ah25": "ah_home_minus25",
    }
    kind = kind_map.get(m, m)
    y = _derive_target_from_goals(df, kind=kind)

    title = f"{league_name} — reliability {market}"
    safe = _re.sub(r"[^A-Za-z0-9_]+","_", f"{league_name}_{market}").strip("_")
    path = out_path or os.path.join(_reports_dir(), f"{safe}_reliability.png")
    return plot_reliability_curve(df.assign(**{kind: y}), prob_col, kind, bins=bins, out_path=path, title=title)

def _load_draw_threshold_json(league_name: str) -> dict:
    """Load the per-league draw threshold JSON produced by train_draw_classifier.py."""
    try:
        import os, json
        from constants import MODEL_DIR
        safe = str(league_name).replace(" ", "_")
        p = os.path.join(MODEL_DIR, f"{safe}_draw_threshold.json")
        if os.path.exists(p):
            with open(p, "r") as fh:
                return json.load(fh) or {}
    except Exception:
        pass
    return {}

def _load_draw_context_into(df, league_name: str):
    """
    Populate df.attrs with draw threshold/mode and set global BEST_K_PCT / BASELINE_ALPHA_USED
    if available in the saved JSON. Non-destructive to existing overrides.
    """
    try:
        payload = _load_draw_threshold_json(league_name)
        if not payload:
            return df
        if "thr_draw" not in df.attrs and "threshold" in payload:
            try:
                df.attrs["thr_draw"] = float(payload.get("threshold"))
            except Exception:
                pass
        if "mode_draw" not in df.attrs and "mode" in payload:
            try:
                df.attrs["mode_draw"] = str(payload.get("mode"))
            except Exception:
                pass
        try:
            if "baseline_blend" in payload:
                df.attrs["baseline_blend"] = float(payload["baseline_blend"])
                globals()["BASELINE_ALPHA_USED"] = float(payload["baseline_blend"])
        except Exception:
            pass
        try:
            if "best_k_pct" in payload:
                globals()["BEST_K_PCT"] = float(payload["best_k_pct"])
        except Exception:
            pass
    except Exception:
        pass
    return df

def _attach_trained_market_scores_if_available(df, league_name: str, markets=None):
    """
    If train_markets.score_trained_markets is available and matching models exist,
    attach trained probabilities/preds to the dataframe. Returns (df, used_trained: bool).
    """
    if bool(df.attrs.get("_markets_scored", False)):
        return df, False
    mkts = markets or ["btts","over25","under25","btts_no","ftr","wtn","clean_sheet",
                       "home_fts","away_fts","home_ge2","away_ge2"]
    try:
        import importlib
        tm = importlib.import_module("train_markets")
        scorer = getattr(tm, "score_trained_markets", None)
        if scorer is None:
            return df, False
        try:
            df2 = scorer(df.copy(), league_name=league_name, markets=mkts)
        except TypeError:
            df2 = scorer(df.copy(), league=league_name, markets=mkts)
        df2.attrs["_markets_scored"] = True
        return df2, True
    except Exception:
        return df, False
# ==== /NEW helpers ========================================================
# ------------------------------------------------------------------
# NaN-handling helpers (lazy import avoids circular-import issues)
# ------------------------------------------------------------------
from importlib import import_module as _imp
# Try the renamed pipeline module first; fall back to legacy names
try:
    _bfp = _imp("_baseline_ftr_pipeline")
except Exception:
    try:
        _bfp = _imp("baseline_ftr_pipeline")
    except Exception:
        _bfp = _imp("00_baseline_ftr_pipeline")
safe_fill = getattr(_bfp, "safe_fill", None)
SENTINEL  = getattr(_bfp, "SENTINEL", -999)
# ------------------------------------------------------------------
# Shared feature‑lists for Poisson-derived helpers
# ------------------------------------------------------------------
try:
    OVER25_FEATURES = getattr(_bfp, "OVER25_FEATURES")
except AttributeError:
    OVER25_FEATURES = []

try:
    EXPANDED_BTTS_FEATURES = getattr(_bfp, "EXPANDED_BTTS_FEATURES")
except AttributeError:
    EXPANDED_BTTS_FEATURES = []
import os, datetime
# ------------------------------------------------------------------
# Path to ModelStore – defined once at import time
# ------------------------------------------------------------------
MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "ModelStore")
)
# ------------------------------------------------------------------
# Path to ModelStore – prefer constants.MODEL_DIR, fallback to local
# ------------------------------------------------------------------
try:
    from constants import MODEL_DIR as _MODEL_DIR
except Exception:
    _MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "ModelStore"))
MODEL_DIR = _MODEL_DIR
# --- Inference globals set by prepare_league_for_inference -----------------
# When you call prepare_league_for_inference(league), we capture the
# chosen baseline blend (alpha) and best top-k% so downstream exports
# can annotate CSVs consistently.
BEST_K_PCT: float | None = None
BASELINE_ALPHA_USED: float | None = None

# ────────────────────────────────────────────────────────────
# Lightweight config dict (can be overwritten by caller)
# ────────────────────────────────────────────────────────────

config = {
    "stake_per_acca": 10,           # flat £ stake per acca
    "acca_sizes": [2, 3],           # folds to simulate
    "roi_n_trials": 10_000,         # Monte‑Carlo trials
    # Kelly‑staking parameters
    "kelly_frac": 0.5,              # use ½‑Kelly stake sizing
    "max_kelly_pct": 0.05,           # cap any Kelly bet at 5 % of bankroll
    # MC-sim tail calming
    "min_pool_x": 8,            # require at least this multiple of acca_size in the candidate pool
    "prob_clip_low": 0.05,      # lower bound for per-leg probability in sims
    "prob_clip_high": 0.95,     # upper bound for per-leg probability in sims
    "max_leg_odds": 12.0,       # optionally cap per-leg odds (e.g., 12.0); None disables
    # Optional global cap for folds (also settable via weekend_smoke_test flags)
    "max_acca_size": 3,
    "quiet_skips": True,
    "write_all_leagues_csv": True,
    # Default classification thresholds (used if models lack calibrated cutoffs)
    "thr_btts_default": 0.50,
    "thr_over25_default": 0.50,
    # Inverse-market defaults
    "thr_under25_default": 0.60,
    "thr_btts_no_default": 0.60,
    "thr_ftr_default": 0.40,
    "thr_wtn_default": 0.25,
    "thr_clean_sheet_default": 0.25,
    "verbose": False,
    "log_level": "INFO",      # INFO | DEBUG | WARN
    "rng_seed": 42,           # default RNG seed for MC sims
    "dual_persist": False,    # also write dated predictions_output CSVs when True
    "allow_synth_odds": False,
    "objective": "roi",           # 'roi' | 'accuracy' | 'hybrid'
    "sort_key": "ev",             # 'ev' | 'edge' | 'prob'
}

# Export a mutable alias that callers (e.g., weekend_smoke_test) can tweak at runtime
overlay_config = config
# Lightweight logging helpers (env overrides config)
LOG_LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30}
_ENV_LEVEL = os.getenv("OVERLAY_LOG_LEVEL", str(config.get("log_level", "INFO"))).upper()
LOG_LEVEL = LOG_LEVELS.get(_ENV_LEVEL, 20)

def _log(level: str, msg: str) -> None:
    try:
        lvl = LOG_LEVELS.get(str(level).upper(), 20)
        if lvl >= LOG_LEVEL:
            print(msg)
    except Exception:
        pass

def _vprint(msg: str) -> None:
    if bool(config.get("verbose", False)):
        try:
            print(msg)
        except Exception:
            pass

# --- Back-compat thresholds mapping (module-level) -------------------------
# Some callers refer to a module-scope `thresholds` dict. Keep it in sync
# with the lightweight config so imports like `thresholds.get('ftr', 0.40)`
# do not crash at import time.
thresholds = {
    'btts': float(config.get('thr_btts_default', 0.50)),
    'over25': float(config.get('thr_over25_default', 0.50)),
    'under25': float(config.get('thr_under25_default', 0.60)),
    'btts_no': float(config.get('thr_btts_no_default', 0.60)),
    'ftr': float(config.get('thr_ftr_default', 0.40)),
    'wtn': float(config.get('thr_wtn_default', 0.25)),
    'clean_sheet': float(config.get('thr_clean_sheet_default', 0.25)),
}
# Defensive: make sure a thresholds dict always exists even if import order shifts
if not isinstance(globals().get('thresholds'), dict):
    thresholds = {
        'btts': float(config.get('thr_btts_default', 0.55)),
        'over25': float(config.get('thr_over25_default', 0.32)),
        'under25': float(config.get('thr_under25_default', 0.60)),
        'btts_no': float(config.get('thr_btts_no_default', 0.60)),
        'ftr': float(config.get('thr_ftr_default', 0.40)),
        'wtn': float(config.get('thr_wtn_default', 0.25)),
        'clean_sheet': float(config.get('thr_clean_sheet_default', 0.25)),
    }
import pandas as pd

# Fallback: if the pipeline didn't expose safe_fill, provide a minimal local version
if safe_fill is None:
    def safe_fill(df: pd.DataFrame, cols: list, fill_value):
        df = df.copy()
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(fill_value)
            else:
                df[c] = fill_value
        return df

# --------------------------------------------------------------
# Hard drop list for leaky/post‑match columns (in addition to POST_MATCH_TAGS)
# --------------------------------------------------------------
LEAKY_ALWAYS_DROP = {
    # final outcomes/events
    'home_team_goal_count', 'away_team_goal_count', 'total_goal_count',
    'total_goals_at_half_time', 'home_team_goal_count_half_time', 'away_team_goal_count_half_time',
    'home_team_goal_timings', 'away_team_goal_timings',
    # post‑match performance stats
    'home_team_shots', 'away_team_shots',
    'home_team_shots_on_target', 'away_team_shots_on_target',
    'home_team_shots_off_target', 'away_team_shots_off_target',
    'home_team_fouls', 'away_team_fouls',
    'home_team_corner_count', 'away_team_corner_count',
    'home_team_yellow_cards', 'away_team_yellow_cards',
    'home_team_red_cards', 'away_team_red_cards',
    'home_team_first_half_cards', 'home_team_second_half_cards',
    'away_team_first_half_cards', 'away_team_second_half_cards',
    'home_team_possession', 'away_team_possession',
    # post‑match xG (distinct from pre‑match xG)
    'team_a_xg', 'team_b_xg',
    # admin / scraped after kickoff
    'status', 'attendance', 'referee',
}
# --------------------------------------------------------------
# Canonicalise messy CSV column names → training schema snake_case
# --------------------------------------------------------------
SAFE_COL_RENAMES = {
    # PPG & xG (messy display headers → canonical snake_case)
    'Pre-Match PPG (Home)': 'pre_match_ppg_home',
    'Pre-Match PPG (Away)': 'pre_match_ppg_away',
    'Home Team Pre-Match xG': 'pre_match_xg_home',
    'Away Team Pre-Match xG': 'pre_match_xg_away',

    # Averages (some sources use prettified title-case)
    'Average Goals Per Match (Pre-Match)': 'average_goals_per_match_pre_match',
    'Average Corners Per Match (Pre-Match)': 'average_corners_per_match_pre_match',
    'Average Cards Per Match (Pre-Match)': 'average_cards_per_match_pre_match',

    # Percentages (BTTS/Over family – prettified variants)
    'BTTS % (Pre-Match)': 'btts_percentage_pre_match',
    'BTTS Percentage (Pre-Match)': 'btts_percentage_pre_match',
    'Over 0.5 % (1st Half, Pre-Match)': 'over_05_HT_FHG_percentage_pre_match',
    'Over 1.5 % (1st Half, Pre-Match)': 'over_15_HT_FHG_percentage_pre_match',
    'Over 0.5 % (2nd Half, Pre-Match)': 'over_05_2HG_percentage_pre_match',
    'Over 1.5 % (2nd Half, Pre-Match)': 'over_15_2HG_percentage_pre_match',
    'Over 1.5 % (Pre-Match)': 'over_15_percentage_pre_match',
    'Over 2.5 % (Pre-Match)': 'over_25_percentage_pre_match',
    'Over 3.5 % (Pre-Match)': 'over_35_percentage_pre_match',
    'Over 4.5 % (Pre-Match)': 'over_45_percentage_pre_match',

    # Misc frequently seen prettified headers
    'Game Week': 'game_week',
}

# --------------------------------------------------------------
# Numeric coercion and hygiene helpers
# --------------------------------------------------------------
import re as _re

def _looks_numeric_string(val: str) -> bool:
    if not isinstance(val, str):
        return False
    s = val.strip()
    if s.upper() == 'N/A' or s == '':
        return True
    # digits, optional decimal comma/dot, optional percent sign
    return bool(_re.fullmatch(r"[+-]?[0-9]+(?:[\.,][0-9]+)?%?", s))


def apply_safe_renames_and_whitelist(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known messy headers to the canonical names used in training.
    Does not drop columns unless both legacy and canonical are present.
    """
    if df is None or df.empty:
        return df
    df = df.copy()

    # Only rename if canonical target doesn't already exist
    to_rename = {old: new for old, new in SAFE_COL_RENAMES.items()
                 if old in df.columns and new not in df.columns}
    if to_rename:
        try:
            _log("INFO", f"🔤 Renaming columns → canonical schema: {to_rename}")
        except Exception:
            pass
        df.rename(columns=to_rename, inplace=True)

    # If both old and new exist, drop the legacy header to avoid ambiguity
    dupes = [old for old, new in SAFE_COL_RENAMES.items()
             if old in df.columns and new in df.columns]
    if dupes:
        try:
            _log("INFO", f"ℹ️ Dropping duplicate legacy headers now replaced by canonical names: {dupes}")
        except Exception:
            pass
        df.drop(columns=dupes, inplace=True, errors='ignore')

    return df


# --------------------------------------------------------------
# Quick sanity‑print so we can see which constants file is loaded
# --------------------------------------------------------------
import constants, os
_log("DEBUG", f">>> constants loaded from: {os.path.abspath(constants.__file__)}")

_log("DEBUG", f">>> USA‑draw gate        : {constants.LOCKED_THRESHOLDS['USA MLS']['draw']}")




# Odds column whitelist per market (strict candidate order)
ODD_COLUMN_WHITELIST = {
    'btts_yes':  ['odds_btts_yes','odds_btts','btts_yes_odds','odds_btts_y','odds_btts_gg'],
    'btts_no':   ['odds_btts_no','btts_no_odds','odds_btts_n','odds_btts_ng'],
    'over25':    ['odds_ft_over25','odds_over25','odds_ft_over_25','over25_odds'],
    'under25':   ['odds_ft_under25','odds_under25','odds_ft_under_25','under25_odds', 'odds_under_2_5','odds_ft_u25','under_2.5_odds','u2_5_odds'],
    'home_win':  ['odds_ft_home_team_win','odds_home_win','odds_ft_home_win','home_win_odds'],
    'draw':      ['odds_ft_draw','odds_draw'],
    'away_win':  ['odds_ft_away_team_win','odds_away_win','odds_ft_away_win','away_win_odds'],
    'home_wtn':  ['odds_home_win_to_nil','odds_home_to_nil','home_win_to_nil_odds','odds_home_cs_win','odds_home_win_nil','odds_home_win_to_nill'],
    'away_wtn':  ['odds_away_win_to_nil','odds_away_to_nil','away_win_to_nil_odds','odds_away_cs_win','odds_away_win_nil','odds_away_win_to_nill'],
    'home_cs':   ['odds_home_cs','home_clean_sheet_odds','odds_home_to_nil','odds_home_cs_win'],
    'away_cs':   ['odds_away_cs','away_clean_sheet_odds','odds_away_to_nil','odds_away_cs_win'],
    'over15':    ['odds_ft_over15','odds_over15','odds_ft_over_15','over15_odds','odds_o15'],
    'over35':    ['odds_ft_over35','odds_over35','odds_ft_over_35','over35_odds','odds_o35'],
    'over45':    ['odds_ft_over45','odds_over45','odds_ft_over_45','over45_odds','odds_o45'],
}
# === Extend odds whitelist to cover Under 2.5, BTTS No, and AH (home -1.5/-2.5) aliases
_try_extend = {
    'under25': [
        "under25_odds","odds_ft_under25","odds_under_2_5","odds_ft_u25",
    ],
    'btts_no': [
        "btts_no_odds","odds_btts_n","odds_btts_ng",
    ],
    'ah_home_minus15': [
        "odds_ah_home_minus15","home_ah_-1_5","ah_home_-1_5","odds_home_ah_-1_5",
    ],
    'ah_home_minus25': [
        "odds_ah_home_minus25","home_ah_-2_5","ah_home_-2_5","odds_home_ah_-2_5",
    ],
}
for _k, _vals in _try_extend.items():
    if _k not in ODD_COLUMN_WHITELIST:
        ODD_COLUMN_WHITELIST[_k] = []
    for _alias in _vals:
        if _alias not in ODD_COLUMN_WHITELIST[_k]:
            ODD_COLUMN_WHITELIST[_k].append(_alias)
# Session-scoped registry of disabled markets (per process run)
DISABLED_MARKETS: set[str] = set()

def _disable_market(market: str, reason: str) -> None:
    """Log once and disable a market for this run."""
    try:
        if market not in DISABLED_MARKETS:
            _log("WARN", f"🚫 Disabling market '{market}' for this run: {reason}")
    except Exception:
        pass
    DISABLED_MARKETS.add(str(market))

def report_disabled_markets() -> set:
    """Print a one-shot summary of disabled markets at the end of the run."""
    if not DISABLED_MARKETS:
        _log("INFO", "✅ All markets enabled this run.")
    else:
        _log("INFO", "🚫 Disabled markets this run: " + ", ".join(sorted(DISABLED_MARKETS)))
    return set(DISABLED_MARKETS)

# Auto-print a summary when the process exits
import atexit as _atexit
_atexit.register(report_disabled_markets)
_percent_like_re = _re.compile(r"(percentage|_pct|_prob(?!a)|_confidence)$", _re.IGNORECASE)


def _force_drop_leaky_cols(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in LEAKY_ALWAYS_DROP if c in df.columns]
    if present:
        print(f"ℹ️ Dropping {len(present)} always-drop columns: {present}")
        df = df.drop(columns=present, errors='ignore')
    return df


def _coerce_numeric_like(df: pd.DataFrame, odds_whitelist: dict | None = None) -> pd.DataFrame:
    """Coerce common numeric-like columns (object dtype) to float.
    Handles: European decimal commas, 'N/A', blanks, and percent strings.
    If *odds_whitelist* is provided, also coerces any column listed there.
    """
    df = df.copy()
    # columns to coerce by regex or known prefixes
    candidates = set()
    # odds/probability/percentage/confidence columns
    for c in df.columns:
        if df[c].dtype == object:
            lc = c.lower()
            if any(tok in lc for tok in ("odds", "prob", "confidence")) or _percent_like_re.search(lc):
                candidates.add(c)
    # include explicit whitelist odds columns
    if odds_whitelist:
        for lst in odds_whitelist.values():
            for c in lst:
                if c in df.columns:
                    candidates.add(c)
    # coercion
    for c in candidates:
        # Always convert to pandas "string" dtype first so `.str` is safe
        s = df[c].astype("string")

        # Trim whitespace
        s = s.str.strip()

        # Normalise common NA tokens to <NA>
        s = s.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "N/A": pd.NA, "n/a": pd.NA})

        # Prepare masks
        pct_mask = s.fillna("").str.endswith('%')
        non_pct  = ~pct_mask

        # Ensure the destination column is numeric float64 before masked assignments
        try:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float64')
        except Exception:
            # If conversion fails, create a float64 column with NaNs
            df[c] = pd.Series(pd.to_numeric(df[c], errors='coerce'), index=df.index, dtype='float64')

        # Percent strings like '62%' → 0.62
        if bool(np.any(pct_mask)):
            s_pct = s[pct_mask].str.rstrip('%').str.replace(',', '.', regex=False)
            vals_pct = pd.to_numeric(s_pct, errors='coerce') / 100.0
            # Assign as float64 to avoid dtype downcast warnings
            df.loc[pct_mask, c] = vals_pct.astype('float64')

        # Non‑percent values: handle European comma decimals on the remainder
        if bool(np.any(non_pct)):
            s_np = s[non_pct].str.replace(',', '.', regex=False)
            vals_np = pd.to_numeric(s_np, errors='coerce')
            df.loc[non_pct, c] = vals_np.astype('float64')

        # Final tidy: ensure column is numeric float64
        try:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float64')
        except Exception:
            pass

        # If values look like 0..100 for probabilities, scale to 0..1
        try:
            col = df[c]
            if bool(np.any(getattr(col, "notna", lambda: pd.Series([], dtype=bool))())):
                mx = float(col.max()) if pd.notna(col.max()) else 0.0
                if 1.5 < mx <= 100:
                    df[c] = (col / 100.0).astype('float64')
        except Exception:
            pass
    return df


def _resolve_odds(df: pd.DataFrame, candidates: list[str], *, label: str = "odds") -> str | None:
    # mypy: allow dynamic attribute on logger function
    _log_any = cast(Any, _log)
    if not hasattr(_log_any, "_logged_labels"):
        setattr(_log_any, "_logged_labels", set())
    # Track which labels have been logged to avoid duplicate logs per run
    if not hasattr(_resolve_odds, "_logged_labels"):
        _resolve_odds._logged_labels = set()
    logged_labels = _resolve_odds._logged_labels
    for c in candidates:
        if c in df.columns:
            # ensure numeric and drop invalid odds (≤ 1.0)
            df[c] = pd.to_numeric(df[c], errors='coerce')
            df[c] = df[c].where(df[c] > 1.0)
            if label not in logged_labels:
                _log("INFO", f"🔎 Using {label} column: '{c}'")
                logged_labels.add(label)
            return c
    if label not in logged_labels:
        _log("WARN", f"⚠️ Missing {label} columns; tried {candidates}.")
        logged_labels.add(label)
    return None

# --------------------------------------------------------------
# Estimate bookmaker margin (overround) from 1X2 odds
# --------------------------------------------------------------

def _estimate_book_margin(df: pd.DataFrame) -> float:
    """Estimate book margin from the FTR (1X2) market.
    Returns a non-negative float (e.g., 0.06 == 6% margin) and stores it in df.attrs['book_margin'].
    Safe no-op if any 1X2 leg is missing."""
    try:
        h = _resolve_odds(df, ODD_COLUMN_WHITELIST['home_win'], label='FTR HOME odds')
        d = _resolve_odds(df, ODD_COLUMN_WHITELIST['draw'],     label='FTR DRAW odds')
        a = _resolve_odds(df, ODD_COLUMN_WHITELIST['away_win'], label='FTR AWAY odds')
        if not (h and d and a):
            return float(df.attrs.get('book_margin', 0.0))
        oh = pd.to_numeric(df[h], errors='coerce')
        od = pd.to_numeric(df[d], errors='coerce')
        oa = pd.to_numeric(df[a], errors='coerce')
        qh = (1.0 / oh).replace({0.0: np.nan})
        qd = (1.0 / od).replace({0.0: np.nan})
        qa = (1.0 / oa).replace({0.0: np.nan})
        s  = (qh + qd + qa).median(skipna=True)
        if pd.isna(s):
            s = (qh + qd + qa).mean(skipna=True)
        margin = float(max((s - 1.0), 0.0)) if pd.notna(s) else 0.0
        df.attrs['book_margin'] = margin
        return margin
    except Exception:
        return float(df.attrs.get('book_margin', 0.0))


# ------------------------------------------------------------------
# Helper: synthesise odds column from a probability column if needed
# ------------------------------------------------------------------

def _synthesise_odds_from_probs(df: pd.DataFrame, prob_col: str, out_col: str) -> str | None:
    """Create a synthetic decimal-odds column from a probability column when
    bookmaker odds are missing. Returns the name of the created column or None.
    We set odds = 1 / p, with NaN where p is invalid. Caller is responsible for
    any fair-odds haircut if needed (handled later using df.attrs['probs_from_odds']).
    """
    if prob_col not in df.columns:
        return None
    try:
        s = pd.to_numeric(df[prob_col], errors='coerce')
        s = s.clip(lower=1e-9, upper=1.0)
        df[out_col] = 1.0 / s.replace({0.0: np.nan})
        try:
            _log("DEBUG", f"🧪 Synthesised odds '{out_col}' from probabilities '{prob_col}' (odds = 1/p)")
        except Exception:
            pass
        return out_col
    except Exception:
        return None
# ------------------------------------------------------------------
# Multi-book odds aggregation (open/close) and CLV attachment
# ------------------------------------------------------------------
import re as __re

_DEF_CLV_KEYS = ("over25","btts","ftr_home","ftr_draw","ftr_away","ah_home_minus15","ah_home_minus25")

def _find_cols(df: pd.DataFrame, key: str, phase: str) -> list[str]:
    """Find odds columns resembling odds_<key>_<phase> or book-specific variants.
    Example matches: odds_over25_open, odds_over25_pin_open, odds_over25_b365_close, open_odds_over25
    """
    cols = []
    pat = __re.compile(rf"^(?:odds_)?{__re.escape(key)}_(?:{phase}|(?:[A-Za-z0-9]+)_{phase})$", __re.IGNORECASE)
    alt = __re.compile(rf"^(?:odds_)?{phase}_(?:{__re.escape(key)})$", __re.IGNORECASE)
    for c in df.columns:
        lc = c.lower()
        if pat.match(lc) or alt.match(lc):
            cols.append(c)
    return cols

def attach_multi_book_odds(df: pd.DataFrame, *, market_keys: tuple[str,...] = _DEF_CLV_KEYS) -> pd.DataFrame:
    """Attach consensus open/close odds per market if multiple book columns exist.
    Consensus is median across any discovered book-specific columns.
    Safe no-op if none found.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    out = df.copy()
    for key in market_keys:
        base = f"odds_{key}"
        c_open  = _find_cols(out, key, "open")
        c_close = _find_cols(out, key, "close")
        if c_open:
            out[f"{base}_open"]  = pd.concat([pd.to_numeric(out[c], errors="coerce") for c in c_open], axis=1).median(axis=1, skipna=True)
        if c_close:
            out[f"{base}_close"] = pd.concat([pd.to_numeric(out[c], errors="coerce") for c in c_close], axis=1).median(axis=1, skipna=True)
    return out

def attach_clv(df: pd.DataFrame, *, market_keys: tuple[str,...] = _DEF_CLV_KEYS) -> pd.DataFrame:
    """Compute simple CLV = close/open - 1 for each market if open/close are present.
    Produces clv_<key> columns. Safe no-op if data missing.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    out = df.copy()
    for key in market_keys:
        base = f"odds_{key}"
        ocol = f"{base}_open"; ccol = f"{base}_close"
        if ocol in out.columns and ccol in out.columns:
            try:
                o = pd.to_numeric(out[ocol], errors="coerce")
                c = pd.to_numeric(out[ccol], errors="coerce")
                out[f"clv_{key}"] = (c / o) - 1.0
            except Exception:
                continue
    return out
# === NEW: ensure standard odds + implied probs + edge per market ===
def _ensure_odds_and_edges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    def _ens(colnames, out_col):
        for c in colnames:
            if c in df.columns and bool(np.any(df[c].notna())):
                df[out_col] = pd.to_numeric(df[c], errors="coerce")
                return
        df[out_col] = np.nan

    # FTR
    _ens(["odds_ft_home_team_win","home_odds","H_odds","odds_home"], "odds_ft_home_team_win")
    _ens(["odds_ft_draw","D_odds","draw_odds"], "odds_ft_draw")
    _ens(["odds_ft_away_team_win","away_odds","A_odds","odds_away"], "odds_ft_away_team_win")

    # Totals & BTTS
    _ens(["odds_over25","over_2_5_odds","o25_odds"], "odds_over25")
    _ens(["odds_btts_yes","btts_yes_odds","btts_odds"], "odds_btts_yes")
    _ens(["odds_under25","under25_odds","odds_ft_under25","odds_under_2_5","odds_ft_u25"], "odds_under25")
    _ens(["odds_btts_no","btts_no_odds","odds_btts_n","odds_btts_ng"], "odds_btts_no")
    # AH (forward-compatible): home -1.5 / -2.5 if present
    _ens(["odds_ah_home_minus15","home_ah_-1_5","ah_home_-1_5","odds_home_ah_-1_5"], "odds_ah_home_minus15")
    _ens(["odds_ah_home_minus25","home_ah_-2_5","ah_home_-2_5","odds_home_ah_-2_5"], "odds_ah_home_minus25")

    # Implied & edge
    for oc, pc in [
        ("odds_over25",  "prob_over25"),
        ("odds_btts_yes","prob_btts"),
        ("odds_ft_home_team_win","prob_ftr_home"),
        ("odds_ft_draw",        "prob_ftr_draw"),
        ("odds_ft_away_team_win","prob_ftr_away"),
        ("odds_under25",  "prob_under25"),
        ("odds_btts_no",  "prob_btts_no"),
        ("odds_ah_home_minus15", "prob_ah_home_minus15"),
        ("odds_ah_home_minus25", "prob_ah_home_minus25"),
    ]:
        if oc in df.columns:
            df[f"imp_{oc}"] = 1.0 / pd.to_numeric(df[oc], errors="coerce").replace(0, np.nan)
        if pc in df.columns and oc in df.columns:
            df[f"edge_{pc}"] = pd.to_numeric(df[pc], errors="coerce") - df[f"imp_{oc}"]
    return df

# ----------------------------------------------------------------------------
# ScenarioSupport — coherence score from available side-heads
# ----------------------------------------------------------------------------
def compute_scenario_support(df: pd.DataFrame, weights: dict[str,float] | None = None) -> pd.Series:
    """
    Light-weight coherence score in [0,1] combining available heads:
      - prob_ftr_home (or confidence_home)
      - prob_over25
      - prob_btts (helps 3-1 stories)
      - prob_ah_home_minus15 / prob_ah_home_minus25
      - prob_home_fts / prob_away_fts (if present)
      - lambda_home / lambda_away (fallback tails)
    Score = weighted average of present components (weights re-normalised).
    """
    if df is None or len(df) == 0:
        return pd.Series([], dtype=float)

    w = {"ftr_home":0.25,"over25":0.20,"ah15":0.20,"ah25":0.10,"away_fts":0.15,"home_fts":0.00,"btts":0.10}
    if isinstance(weights, dict):
        w.update({k: float(v) for k,v in weights.items() if k in w})

    comps: list[tuple[pd.Series,float]] = []

    # FTR home
    p_fh = pd.to_numeric(df.get("prob_ftr_home"), errors="coerce") if "prob_ftr_home" in df.columns else pd.to_numeric(df.get("confidence_home"), errors="coerce")
    if p_fh is not None and bool(np.any(pd.notna(p_fh))):
        comps.append((p_fh.fillna(0).clip(0,1), w["ftr_home"]))

    # Over 2.5
    if "prob_over25" in df.columns:
        comps.append((pd.to_numeric(df["prob_over25"], errors="coerce").fillna(0).clip(0,1), w["over25"]))

    # AH −1.5 / −2.5
    if "prob_ah_home_minus15" in df.columns:
        comps.append((pd.to_numeric(df["prob_ah_home_minus15"], errors="coerce").fillna(0).clip(0,1), w["ah15"]))
    if "prob_ah_home_minus25" in df.columns:
        comps.append((pd.to_numeric(df["prob_ah_home_minus25"], errors="coerce").fillna(0).clip(0,1), w["ah25"]))

    # Away fails to score → boosts 2-0 stories
    if "prob_away_fts" in df.columns:
        comps.append((pd.to_numeric(df["prob_away_fts"], errors="coerce").fillna(0).clip(0,1), w["away_fts"]))

    # BTTS supports 3-1 stories (low weight)
    if "prob_btts" in df.columns:
        comps.append((pd.to_numeric(df["prob_btts"], errors="coerce").fillna(0).clip(0,1), w["btts"]))

    # Lambda fallback tails
    try:
        if ("lambda_home" in df.columns) and ("lambda_away" in df.columns):
            lh = pd.to_numeric(df["lambda_home"], errors="coerce").fillna(1.2)
            la = pd.to_numeric(df["lambda_away"], errors="coerce").fillna(1.0)
            p_h_ge2 = 1.0 - np.exp(-lh) * (1.0 + lh)
            p_a_ge1 = 1.0 - np.exp(-la)
            comps.append(((1.0 - p_a_ge1).clip(0,1), w["away_fts"]*0.5))  # extra boost for 2-0
            comps.append((p_h_ge2.clip(0,1), w["over25"]*0.5))            # mild support for goals
    except Exception:
        pass

    if not comps:
        return pd.Series([0.0]*len(df), index=df.index, dtype=float)

    weights_arr = np.array([wt for _, wt in comps], dtype=float)
    weights_arr = weights_arr/weights_arr.sum() if weights_arr.sum() > 0 else np.ones_like(weights_arr)
    vals = np.vstack([np.asarray(val) for val,_ in comps])
    score = (weights_arr[:,None] * vals).sum(axis=0)
    return pd.Series(np.clip(score, 0, 1), index=df.index, dtype=float)

def _attach_scenario_support(df: pd.DataFrame, *, min_scenario: float = 0.0, weights: dict[str,float] | None = None) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    out = df.copy()
    try:
        sc = compute_scenario_support(out, weights=weights)
        out["scenario_support"] = sc
        if min_scenario > 0:
            out = out.loc[sc >= float(min_scenario)].copy()
    except Exception:
        pass
    return out

# Attach standard EV columns for a (prob, odds) pair
if "_attach_ev_columns" not in globals():
    def _attach_ev_columns(df: pd.DataFrame,
                           prob_col: str,
                           odds_col: str,
                           *,
                           margin: float = 0.0,
                           probs_from_odds: bool = False,
                           synth_odds: bool = False) -> pd.DataFrame:
        """Attach p_model, odds, edge, kelly_frac, expected_value (with fair-odds haircut)."""
        df = df.copy()
        p = pd.to_numeric(df[prob_col], errors='coerce')
        o = pd.to_numeric(df[odds_col], errors='coerce')
        # Fair-odds adjustment only for expected_value
        if probs_from_odds or synth_odds:
            fair = o / (1.0 + max(margin, 0.0))
        else:
            fair = o * 0.95
        df["p_model"] = p
        df["odds"] = o
        with np.errstate(divide='ignore', invalid='ignore'):
            implied = 1.0 / o.replace({0.0: np.nan})
        df["edge"] = (p - implied).astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            k = (p * o - 1.0) / (o - 1.0)
        df["kelly_frac"] = k.clip(lower=0.0).fillna(0.0)
        df["expected_value"] = p * fair - 1.0
        return df

# ------------------------------------------------------------------
# Confidence coverage & decisive-score reporting (per slate/league)
# ------------------------------------------------------------------
def _pick_prob_odds_for_market(df: pd.DataFrame, market: str) -> tuple[str|None, str|None]:
    """Return (prob_col, odds_col) using the same precedence as _prepare_candidates.
    Safe-returns (None, None) when unresolved."""
    m = str(market).lower()
    pc: str | None = None
    oc: str | None = None
    # Mirror the robust selection used in _prepare_candidates
    if m == "over25":
        if bool(df.attrs.get("calibrated")) and "prob_over25" in df.columns:
            pc = "prob_over25"
        elif "adjusted_over25_confidence" in df.columns:
            pc = "adjusted_over25_confidence"
        elif "over25_confidence" in df.columns:
            pc = "over25_confidence"
        else:
            pc = "prob_over25" if "prob_over25" in df.columns else None
        oc = "odds_over25" if "odds_over25" in df.columns else ("odds_ft_over25" if "odds_ft_over25" in df.columns else None)
        return pc, oc
    if m == "btts":
        if bool(df.attrs.get("calibrated")) and "prob_btts" in df.columns:
            pc = "prob_btts"
        elif "adjusted_btts_confidence" in df.columns:
            pc = "adjusted_btts_confidence"
        elif "btts_confidence" in df.columns:
            pc = "btts_confidence"
        else:
            pc = "prob_btts" if "prob_btts" in df.columns else None
        return pc, ("odds_btts_yes" if "odds_btts_yes" in df.columns else None)
    if m in ("ftr_home","ftr_draw","ftr_away"):
        # side-specific
        side = m.split("_",1)[1]
        # probability ladders
        ladders = {
            "home": ("prob_ftr_home","confidence_home","p_home","home_win_proba"),
            "draw": ("prob_ftr_draw","confidence_draw","p_draw","draw_win_proba"),
            "away": ("prob_ftr_away","confidence_away","p_away","away_win_proba"),
        }
        pc = None
        for cand in ladders.get(side, ()):
            if cand in df.columns:
                pc = cand; break
        # single selected confidence + outcome fallback
        if pc is None and "selected_confidence" in df.columns:
            if "selected_outcome" in df.columns:
                tmp = pd.to_numeric(df["selected_confidence"], errors="coerce")
                mask_map = {"home":["home","h","1","home win","home_win"],
                            "draw":["draw","d","x"],
                            "away":["away","a","2","away win","away_win"]}
                mask = df["selected_outcome"].astype(str).str.lower().isin(mask_map.get(side, []))
                df[f"__tmp_prob_{side}"] = tmp.where(mask, np.nan)
                pc = f"__tmp_prob_{side}"
            elif "ftr_pred_outcome" in df.columns:
                tmp = pd.to_numeric(df["selected_confidence"], errors="coerce")
                code = {"home":0,"draw":1,"away":2}[side]
                mask = pd.to_numeric(df["ftr_pred_outcome"], errors="coerce").astype("Int64") == code
                df[f"__tmp_prob_{side}"] = tmp.where(mask, np.nan)
                pc = f"__tmp_prob_{side}"
        # odds
        oc_map = {"home":"odds_ft_home_team_win","draw":"odds_ft_draw","away":"odds_ft_away_team_win"}
        oc = oc_map.get(side)
        return pc, (oc if oc in df.columns else None)
    return None, None


def _summarize_confidence_and_write_dup(df: pd.DataFrame,
                                   league_name: str,
                                   out_dir: str | None,
                                   markets: tuple[str,...] = ("over25","btts","ftr_home","ftr_draw","ftr_away"),
                                   top_n: int = 100) -> None:
    """
    Produce a quick 'confidence coverage' snapshot for a league slate:
      - counts above prob threshold (env/attrs),
      - counts meeting decisive score gates (if present),
      - counts passing EV/edge gating (if odds & probs present).
    Writes CSVs under out_dir:
      * <League>_confidence_summary.csv
      * <League>_confidence_details.csv  (top-N per market by EV, includes decisive scores)
    Safe no-op if df empty.
    """
    import os, math
    if not isinstance(df, pd.DataFrame) or df.empty:
        return
    dstdir = out_dir or os.path.join("predictions_output", datetime.datetime.utcnow().strftime("%Y-%m-%d"))
    os.makedirs(dstdir, exist_ok=True)

    rows = []
    details = []

    # Resolve slate-level gate defaults from env / attrs / config
    def _thr_for(mkt:str, df: pd.DataFrame) -> float:
        m = mkt.lower()
        if m == "over25":
            return float(os.getenv("FALLBACK_OVER25_PROB_MIN",
                                   df.attrs.get("thr_over25", os.getenv("FALLBACK_PROB_MIN","0"))))
        if m == "btts":
            return float(os.getenv("FALLBACK_BTTS_PROB_MIN",
                                   df.attrs.get("thr_btts", os.getenv("FALLBACK_PROB_MIN","0"))))
        if m.startswith("ftr"):
            return float(os.getenv("FALLBACK_FTR_PROB_MIN",
                                   df.attrs.get("thr_ftr", os.getenv("FALLBACK_PROB_MIN","0"))))
        return 0.0

    try:
        ev_min = float(os.getenv("EV_MIN","0.0"))
    except Exception:
        ev_min = 0.0
    try:
        min_edge = float(os.getenv("MIN_EDGE","0.0"))
    except Exception:
        min_edge = 0.0
    try:
        max_odds = float(os.getenv("MAX_ODDS", str(config.get("max_leg_odds", 12.0))))
    except Exception:
        max_odds = 12.0
    over_min = float(os.getenv("OVER_SCORE_MIN","66"))
    btts_min = float(os.getenv("BTTS_SCORE_MIN","60"))

    for m in markets:
        pc, oc = _pick_prob_odds_for_market(df, m)
        n_total = len(df)
        n_prob = n_dec = n_ev = 0

        # Probability threshold coverage
        if pc and pc in df.columns:
            n_prob = int((pd.to_numeric(df[pc], errors="coerce") >= _thr_for(m, df)).sum())

        # Decisive score coverage
        if m == "over25" and "decisive_over_score" in df.columns:
            n_dec = int((pd.to_numeric(df["decisive_over_score"], errors="coerce") >= over_min).sum())
        elif m == "btts" and "decisive_btts_score" in df.columns:
            n_dec = int((pd.to_numeric(df["decisive_btts_score"], errors="coerce") >= btts_min).sum())

        # EV/edge coverage (requires prob+odds)
        if pc and oc and pc in df.columns and oc in df.columns:
            p = pd.to_numeric(df[pc], errors="coerce")
            o = pd.to_numeric(df[oc], errors="coerce")
            implied = 1.0 / o.replace({0.0: np.nan})
            ev = p*o - 1.0
            edge = p - implied
            mask = (ev >= ev_min) & (edge >= min_edge) & (o <= max_odds)
            n_ev = int(mask.fillna(False).sum())

            # Collect top-N details for this market
            take = df.loc[mask].copy()
            if not take.empty:
                # attach EV/edge columns for export
                take["__ev"] = ev
                take["__edge"] = edge
                take["__prob"] = p
                take["__odds"] = o
                # bring decisive columns if present
                keep_cols = [c for c in ("match_date","League","home_team_name","away_team_name",
                                         "__prob","__odds","__ev","__edge",
                                         "decisive_over_score","decisive_btts_score","decisive_grade") if c in take.columns]
                top = take.sort_values("__ev", ascending=False).head(int(top_n))[keep_cols].copy()
                top.insert(0, "market", m)
                details.append(top)

        rows.append({"league": league_name, "market": m,
                     "n_rows": n_total, "prob_gate": n_prob,
                     "decisive_gate": n_dec, "ev_gate": n_ev})

    # Write summary CSV
    try:
        import pandas as _pd
        summ = _pd.DataFrame(rows)
        tag = league_name.replace(" ","_")
        summ.to_csv(os.path.join(dstdir, f"{tag}_confidence_summary.csv"), index=False)
        if details:
            det = _pd.concat(details, axis=0, ignore_index=True)
            det.to_csv(os.path.join(dstdir, f"{tag}_confidence_details.csv"), index=False)
        print(f"📊 Wrote confidence coverage → {dstdir}")
    except Exception:
        pass

    # Prepare EV-ranked candidate rows for a single market
    if "_prepare_candidates" not in globals():
        def _prepare_candidates(df: pd.DataFrame, market: str, min_edge: float, max_odds: float | None = None) -> pd.DataFrame:
            m = market.lower()
            df = df.copy()
            df0 = df.copy() 
            # Map market to probability/odds columns with robust fallbacks
            prob_col, odds_col = None, None

            if m == "ftr_home":
                if "prob_ftr_home" in df.columns:
                    prob_col = "prob_ftr_home"
                elif "confidence_home" in df.columns:
                    prob_col = "confidence_home"
                elif "p_home" in df.columns:
                    prob_col = "p_home"
                elif "home_win_proba" in df.columns:
                    prob_col = "home_win_proba"
                elif {"selected_confidence","selected_outcome"}.issubset(df.columns):
                    tmp = pd.to_numeric(df["selected_confidence"], errors="coerce")
                    mask = df["selected_outcome"].astype(str).str.lower().isin(["home","h","1","home win","home_win"])
                    df["__tmp_prob_home"] = tmp.where(mask, np.nan)
                    prob_col = "__tmp_prob_home"
                elif {"selected_confidence","ftr_pred_outcome"}.issubset(df.columns):
                    tmp = pd.to_numeric(df["selected_confidence"], errors="coerce")
                    mask = pd.to_numeric(df["ftr_pred_outcome"], errors="coerce").astype("Int64") == 0
                    df["__tmp_prob_home"] = tmp.where(mask, np.nan)
                    prob_col = "__tmp_prob_home"
                odds_col = "odds_ft_home_team_win"

            elif m == "ftr_draw":
                if "prob_ftr_draw" in df.columns:
                    prob_col = "prob_ftr_draw"
                elif "confidence_draw" in df.columns:
                    prob_col = "confidence_draw"
                elif "p_draw" in df.columns:
                    prob_col = "p_draw"
                elif "draw_win_proba" in df.columns:
                    prob_col = "draw_win_proba"
                elif {"selected_confidence","selected_outcome"}.issubset(df.columns):
                    tmp = pd.to_numeric(df["selected_confidence"], errors="coerce")
                    mask = df["selected_outcome"].astype(str).str.lower().isin(["draw","d","x"])
                    df["__tmp_prob_draw"] = tmp.where(mask, np.nan)
                    prob_col = "__tmp_prob_draw"
                elif {"selected_confidence","ftr_pred_outcome"}.issubset(df.columns):
                    tmp = pd.to_numeric(df["selected_confidence"], errors="coerce")
                    mask = pd.to_numeric(df["ftr_pred_outcome"], errors="coerce").astype("Int64") == 1
                    df["__tmp_prob_draw"] = tmp.where(mask, np.nan)
                    prob_col = "__tmp_prob_draw"
                odds_col = "odds_ft_draw"

            elif m == "ftr_away":
                if "prob_ftr_away" in df.columns:
                    prob_col = "prob_ftr_away"
                elif "confidence_away" in df.columns:
                    prob_col = "confidence_away"
                elif "p_away" in df.columns:
                    prob_col = "p_away"
                elif "away_win_proba" in df.columns:
                    prob_col = "away_win_proba"
                elif {"selected_confidence","selected_outcome"}.issubset(df.columns):
                    tmp = pd.to_numeric(df["selected_confidence"], errors="coerce")
                    mask = df["selected_outcome"].astype(str).str.lower().isin(["away","a","2","away win","away_win"])
                    df["__tmp_prob_away"] = tmp.where(mask, np.nan)
                    prob_col = "__tmp_prob_away"
                elif {"selected_confidence","ftr_pred_outcome"}.issubset(df.columns):
                    tmp = pd.to_numeric(df["selected_confidence"], errors="coerce")
                    mask = pd.to_numeric(df["ftr_pred_outcome"], errors="coerce").astype("Int64") == 2
                    df["__tmp_prob_away"] = tmp.where(mask, np.nan)
                    prob_col = "__tmp_prob_away"
                odds_col = "odds_ft_away_team_win"

            elif m == "btts":
                if "adjusted_btts_confidence" in df.columns:
                    prob_col, odds_col = "adjusted_btts_confidence", "odds_btts_yes"
                elif "btts_confidence" in df.columns:
                    prob_col, odds_col = "btts_confidence", "odds_btts_yes"
                else:
                    prob_col, odds_col = ("prob_btts" if "prob_btts" in df.columns else None), "odds_btts_yes"

            elif m == "under25":
                if "prob_under25" in df.columns:
                    prob_col = "prob_under25"
                elif "adjusted_under25_confidence" in df.columns:
                    prob_col = "adjusted_under25_confidence"
                elif "under25_confidence" in df.columns:
                    prob_col = "under25_confidence"
                else:
                    prob_col = None
                odds_col = "odds_under25" if "odds_under25" in df.columns else "odds_ft_under25"

            elif m == "over25":
                if "prob_over25" in df.columns:
                    prob_col = "prob_over25"
                elif "adjusted_over25_confidence" in df.columns:
                    prob_col = "adjusted_over25_confidence"
                elif "over25_confidence" in df.columns:
                    prob_col = "over25_confidence"
                else:
                    prob_col = None
                odds_col = "odds_over25" if "odds_over25" in df.columns else "odds_ft_over25"

            elif m == "btts_no":
                if "prob_btts_no" in df.columns:
                    prob_col, odds_col = "prob_btts_no", "odds_btts_no"
                elif "btts_no_confidence" in df.columns:
                    prob_col, odds_col = "btts_no_confidence", "odds_btts_no"
                else:
                    prob_col, odds_col = (None, "odds_btts_no")

            elif m == "wtn_home":
                if "prob_wtn_home" in df.columns:
                    prob_col, odds_col = "prob_wtn_home", "odds_home_wtn"
                elif "wtn_home_confidence" in df.columns:
                    prob_col, odds_col = "wtn_home_confidence", "odds_home_wtn"
                else:
                    prob_col, odds_col = (None, "odds_home_wtn")

            elif m == "wtn_away":
                if "prob_wtn_away" in df.columns:
                    prob_col, odds_col = "prob_wtn_away", "odds_away_wtn"
                elif "wtn_away_confidence" in df.columns:
                    prob_col, odds_col = "wtn_away_confidence", "odds_away_wtn"
                else:
                    prob_col, odds_col = (None, "odds_away_wtn")

            elif m == "ah_home_minus15":
                prob_col, odds_col = "prob_ah_home_minus15", "odds_ah_home_minus15"

            elif m == "ah_home_minus25":
                prob_col, odds_col = "prob_ah_home_minus25", "odds_ah_home_minus25"

            else:
                raise ValueError(f"Unknown market {market}")

            # Keep a copy for fallbacks AFTER mapping
            df0 = df.copy()
            # Early exit if we still don't have a prob column
            if prob_col is None or prob_col not in df.columns:
                return pd.DataFrame(columns=["Date","Time","League","Home","Away","ev","edge"])

            # Optional CLV filter if clv_min is set
            try:
                import os as _os
                clv_min_raw = _os.getenv("CLV_MIN", None)
                clv_min: float | None = float(clv_min_raw) if clv_min_raw is not None else None
                if clv_min is not None:
                    clv_map = {
                        "over25": "clv_over25",
                        "btts": "clv_btts",
                        "ftr_home": "clv_ftr_home",
                        "ftr_draw": "clv_ftr_draw",
                        "ftr_away": "clv_ftr_away",
                        "ah_home_minus15": "clv_ah_home_minus15",
                        "ah_home_minus25": "clv_ah_home_minus25",
                    }
                    clv_col = clv_map.get(m)
                    if isinstance(clv_col, str) and clv_col in df.columns:
                        df = df[pd.to_numeric(df[clv_col], errors="coerce") >= clv_min].copy()
            except Exception:
                pass
            # Compute EV + edge columns
            try:
                df["ev"] = pd.to_numeric(df[prob_col], errors="coerce") * pd.to_numeric(df[odds_col], errors="coerce") - 1.0
                df["edge"] = pd.to_numeric(df[prob_col], errors="coerce") - (1.0 / pd.to_numeric(df[odds_col], errors="coerce").replace(0, np.nan))
            except Exception:
                return pd.DataFrame(columns=["Date","Time","League","Home","Away","ev","edge"])  # empty

            # EV gate (allow debug-time negative EV via EV_MIN)
            try:
                import os as _os
                _ev_min = float(_os.getenv("EV_MIN", "0.0"))
            except Exception:
                _ev_min = 0.0

            # Optional odds cap
            if max_odds is not None:
                try:
                    df = df[pd.to_numeric(df[odds_col], errors="coerce") <= float(max_odds)]
                except Exception:
                    pass

            # EV/edge gate (using _ev_min)
            mask_ev   = pd.to_numeric(df["ev"], errors="coerce")   >= _ev_min
            mask_edge = pd.to_numeric(df["edge"], errors="coerce") >= float(min_edge)
            df = df[mask_ev & mask_edge].copy()
            # Fallback: if EV/edge gating yields empty, use calibrated market thresholds to seed a pool
            if df.empty:
                try:
                    import os as _os
                    _thr = None
                    if m == "over25":
                        _thr = float(_os.getenv("FALLBACK_OVER25_PROB_MIN",
                                                df0.attrs.get("thr_over25",
                                                              _os.getenv("FALLBACK_PROB_MIN", "0"))))
                    elif m == "btts":
                        _thr = float(_os.getenv("FALLBACK_BTTS_PROB_MIN",
                                                df0.attrs.get("thr_btts",
                                                              _os.getenv("FALLBACK_PROB_MIN", "0"))))
                    elif m in ("ftr_home","ftr_draw","ftr_away"):
                        _thr = float(_os.getenv("FALLBACK_FTR_PROB_MIN",
                                                df0.attrs.get("thr_ftr",
                                                              _os.getenv("FALLBACK_PROB_MIN", "0"))))
                    # Build probability mask if possible
                    _fallback_df = pd.DataFrame(index=df0.index)
                    if _thr is not None and prob_col is not None and prob_col in df0.columns:
                        _pmask = pd.to_numeric(df0[prob_col], errors="coerce") >= float(_thr)
                        _fallback_df = df0.loc[_pmask].copy()

                    # Add Decisive score gates (env: OVER_SCORE_MIN / BTTS_SCORE_MIN)
                    if m == "over25" and "decisive_over_score" in df0.columns:
                        try:
                            over_min = float(_os.getenv("OVER_SCORE_MIN", "66"))
                        except Exception:
                            over_min = 66.0
                        _mask_o = pd.to_numeric(df0["decisive_over_score"], errors="coerce") >= over_min
                        _fallback_df = pd.concat([_fallback_df, df0.loc[_mask_o]], axis=0).drop_duplicates()

                    if m == "btts" and "decisive_btts_score" in df0.columns:
                        try:
                            btts_min = float(_os.getenv("BTTS_SCORE_MIN", "60"))
                        except Exception:
                            btts_min = 60.0
                        _mask_b = pd.to_numeric(df0["decisive_btts_score"], errors="coerce") >= btts_min
                        _fallback_df = pd.concat([_fallback_df, df0.loc[_mask_b]], axis=0).drop_duplicates()

                    if not _fallback_df.empty:
                        try:
                            print(f"ℹ️ fallback threshold used for {m}: thr={_thr}")
                        except Exception:
                            pass
                        # recompute EV/edge if needed
                        if "ev" not in _fallback_df.columns or "edge" not in _fallback_df.columns:
                            try:
                                _fallback_df["ev"] = (
                                    pd.to_numeric(_fallback_df[prob_col], errors="coerce")
                                    * pd.to_numeric(_fallback_df[odds_col], errors="coerce")
                                    - 1.0
                                )
                                _fallback_df["edge"] = (
                                    pd.to_numeric(_fallback_df[prob_col], errors="coerce")
                                    - (1.0 / pd.to_numeric(_fallback_df[odds_col], errors="coerce").replace(0, np.nan))
                                )
                            except Exception:
                                pass
                        df = _fallback_df
                        if _pcol_for_mask is not None:
                            prob_col = cast(str, _pcol_for_mask)
                except Exception:
                    pass
            # Tag prob/odds columns for composer context
            df.attrs["_prob_col"] = prob_col
            df.attrs["_odds_col"] = odds_col
            # Order by EV desc, then edge, then prob
            return df.sort_values(["ev","edge",prob_col], ascending=[False, False, False])

# Compose best accas from a single pool
if "compose_best_accas" not in globals():
    def compose_best_accas(candidates: pd.DataFrame, acca_size: int, top_k: int = 10) -> list[dict]:
        if candidates is None or len(candidates) == 0:
            return []

        pool = candidates.copy()
        prob_attr = pool.attrs.get("_prob_col")
        odds_attr = pool.attrs.get("_odds_col")

        # Row-wise preferences if attrs are missing
        PROB_PREF = (
            "p_model",
            "prob_over25","adjusted_over25_confidence","over25_confidence",
            "prob_btts","adjusted_btts_confidence","btts_confidence",
            "prob_ftr_home","confidence_home",
            "prob_ftr_draw","confidence_draw",
            "prob_ftr_away","confidence_away",
        )
        ODDS_PREF = (
            "od","odds",
            "odds_over25","odds_ft_over25",
            "odds_btts_yes",
            "odds_ft_home_team_win","odds_ft_draw","odds_ft_away_team_win",
        )

        def _first_present_row(row: pd.Series, keys: tuple[str, ...]) -> str | None:
            for k in keys:
                if k in row.index and pd.notna(row[k]):
                    return k
            return None

        accas: list[dict] = []

        for _, seed in pool.head(100).iterrows():
            acc = [seed]
            # Greedy fill
            for _, row in pool.iterrows():
                if len(acc) >= int(acca_size):
                    break
                if row is seed:
                    continue
                acc.append(row)

            if len(acc) == int(acca_size):
                acc_df = pd.DataFrame(acc).copy()

                # Ensure canonical probability column per leg
                if "p_model" not in acc_df.columns or acc_df["p_model"].isna().any():
                    # Attr-first
                    if prob_attr and prob_attr in acc_df.columns:
                        if "p_model" not in acc_df.columns:
                            acc_df["p_model"] = pd.to_numeric(acc_df[prob_attr], errors="coerce")
                        else:
                            acc_df["p_model"] = acc_df["p_model"].fillna(pd.to_numeric(acc_df[prob_attr], errors="coerce"))
                    # Fallbacks
                    for _c in PROB_PREF:
                        if _c in acc_df.columns:
                            if "p_model" not in acc_df.columns:
                                acc_df["p_model"] = pd.to_numeric(acc_df[_c], errors="coerce")
                            else:
                                acc_df["p_model"] = acc_df["p_model"].fillna(pd.to_numeric(acc_df[_c], errors="coerce"))
                    # Row-wise last resort
                    if "p_model" not in acc_df.columns or acc_df["p_model"].isna().any():
                        def _row_prob(r):
                            k = _first_present_row(r, PROB_PREF)
                            return pd.to_numeric(r.get(k, pd.NA), errors="coerce") if k else pd.NA
                        if "p_model" not in acc_df.columns:
                            acc_df["p_model"] = pd.NA
                        acc_df.loc[acc_df["p_model"].isna(), "p_model"] = acc_df.apply(_row_prob, axis=1)

                # Ensure canonical odds column per leg
                if "od" not in acc_df.columns or acc_df["od"].isna().any():
                    # Attr-first
                    if odds_attr and odds_attr in acc_df.columns:
                        if "od" not in acc_df.columns:
                            acc_df["od"] = pd.to_numeric(acc_df[odds_attr], errors="coerce")
                        else:
                            acc_df["od"] = acc_df["od"].fillna(pd.to_numeric(acc_df[odds_attr], errors="coerce"))
                    # Common alias
                    if "od" not in acc_df.columns and "odds" in acc_df.columns:
                        acc_df["od"] = pd.to_numeric(acc_df["odds"], errors="coerce")
                    # Fallbacks
                    if "od" not in acc_df.columns or acc_df["od"].isna().any():
                        def _row_od(r):
                            k = _first_present_row(r, ODDS_PREF)
                            return pd.to_numeric(r.get(k, pd.NA), errors="coerce") if k else pd.NA
                        if "od" not in acc_df.columns:
                            acc_df["od"] = pd.NA
                        acc_df.loc[acc_df["od"].isna(), "od"] = acc_df.apply(_row_od, axis=1)

                # Drop rows lacking either p_model or od to avoid NaNs in products
                acc_df = acc_df[
                    pd.to_numeric(acc_df.get("p_model"), errors="coerce").notna() &
                    pd.to_numeric(acc_df.get("od"), errors="coerce").notna()
                ].copy()

                # Refill if short: top up from best remaining rows that have both p_model and od
                if len(acc_df) < int(acca_size):
                    try:
                        # Track used match keys to avoid duplicate fixtures
                        used_mks = set()
                        for _, rr in acc_df.iterrows():
                            try:
                                used_mks.add(_match_key(rr))
                            except Exception:
                                used_mks.add(rr.name)

                        # Walk the master candidate list (already EV-ordered) to top up
                        for _, r in pool.iterrows():
                            if len(acc_df) >= int(acca_size):
                                break
                            try:
                                mk = _match_key(r)
                            except Exception:
                                mk = r.name
                            if mk in used_mks:
                                continue

                            pv = pd.to_numeric(r.get("p_model"), errors="coerce")
                            ov = pd.to_numeric(r.get("od"), errors="coerce") if "od" in r else pd.to_numeric(r.get("odds"), errors="coerce")
                            if pd.notna(pv) and pd.notna(ov):
                                add = r.copy()
                                # ensure 'od' exists even if only 'odds' is present
                                if "od" not in add and "odds" in add:
                                    add["od"] = pd.to_numeric(add["odds"], errors="coerce")
                                acc_df = pd.concat([acc_df, pd.DataFrame([add])], ignore_index=True)
                                used_mks.add(mk)
                    except Exception:
                        pass

                # If still short, abandon this slip
                if len(acc_df) != int(acca_size):
                    continue

                # Per-leg EV/edge if missing (helps legs table)
                if "ev" not in acc_df.columns:
                    acc_df["ev"] = acc_df["p_model"].astype(float) * acc_df["od"].astype(float) - 1.0
                if "edge" not in acc_df.columns:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        acc_df["edge"] = acc_df["p_model"].astype(float) - (1.0 / acc_df["od"].astype(float))

                # Slip-level prob/odds/EV
                try:
                    p = float(np.prod(pd.to_numeric(acc_df["p_model"], errors="coerce").clip(0, 1).values))
                    o = float(np.prod(pd.to_numeric(acc_df["od"], errors="coerce").values))
                except Exception:
                    continue
                if not np.isfinite(o) or o <= 0:
                    continue
                ev = p * o - 1.0

                # Canonical leg columns for printing/export
                KEEP_COLS = [
                    "Date","Time","League","Home","Away",
                    "p_model","od","ev","edge","__mkt",
                    "adjusted_over25_confidence","over25_confidence","prob_over25",
                    "adjusted_btts_confidence","btts_confidence","prob_btts",
                    "prob_ftr_home","prob_ftr_draw","prob_ftr_away",
                    "decisive_over_score","decisive_btts_score","decisive_grade",
                ]
                legs = acc_df[[c for c in KEEP_COLS if c in acc_df.columns]].copy()

                accas.append({
                    "legs": legs,
                    "prob": p,
                    "odds": o,
                    "ev": ev
                })
            if len(accas) >= int(top_k):
                break

        return sorted(accas, key=lambda x: x["ev"], reverse=True)

# Write markdown + json betslips to disk

if "write_betslips" not in globals():
    def write_betslips(accas: list[dict], dst_dir: str, stake: float | None = None, title: str = "best_betslips", bankroll: float | None = None):
        _cstr: dict[str, Any] = {}
        import os, json, datetime
        os.makedirs(dst_dir, exist_ok=True)
        stake_f: float = _to_float(stake if stake is not None else overlay_config.get("stake_per_acca", 10))
        md_lines = [f"# {title}\n"]
        # Show active constraints (env overrides config)
        try:
            import os, json
            _cstr = {
                "MAX_AH_LEGS": int(os.getenv("MAX_AH_LEGS", str(overlay_config.get("max_ah_legs", 2)))),
                "MAX_PER_LEAGUE": int(os.getenv("MAX_PER_LEAGUE", str(overlay_config.get("max_per_league", 999)))),
                "MAX_PER_MARKET": os.getenv("MAX_PER_MARKET", json.dumps(overlay_config.get("max_per_market", {}))),
                "MAX_COMBINED_ODDS": float(os.getenv("MAX_COMBINED_ODDS", str(overlay_config.get("max_combined_odds", 1e12)))),
                "MIN_COMBINED_PROB": float(os.getenv("MIN_COMBINED_PROB", str(overlay_config.get("min_combined_prob", 0.0)))),
                "MAX_LONGSHOTS": int(os.getenv("MAX_LONGSHOTS", str(overlay_config.get("max_longshots", 999)))),
                "LONGSHOT_ODDS": float(os.getenv("LONGSHOT_ODDS", str(overlay_config.get("longshot_odds", 3.0)))),
            }
            md_lines.append("\n**Constraints:** " + ", ".join(f"{k}={v}" for k,v in _cstr.items()) + "\n")
        except Exception:
            pass

        # 1/2-Kelly slip stake suggestion if bankroll provided
        try:
            k_frac = _to_float(overlay_config.get("kelly_frac", 0.5))
            k_cap  = _to_float(overlay_config.get("max_kelly_pct", 0.05))
            show_kelly = bankroll is not None and bankroll > 0
        except Exception:
            show_kelly = False

        for i, a in enumerate(accas, 1):
            line = f"## Slip {i} — {len(a['legs'])}-Fold — Combined odds {a['odds']:.2f} — EV {(a['ev']*100):.1f}%"
            if show_kelly:
                p = max(0.0, min(1.0, float(a.get("prob", 0.0))))
                o = float(a.get("odds", 0.0))
                k = 0.0 if o <= 1.0 else max(0.0, (p*o - 1.0)/(o - 1.0))
                k *= k_frac
                k = min(k, k_cap)
                suggested = k * float(bankroll)
                line += f" — ½-Kelly stake £{suggested:.2f} (bankroll £{bankroll:.2f}, cap {k_cap*100:.0f}%)"
            elif stake_f is not None and stake_f > 0:
                implied_return = stake_f * float(a["odds"])
                line += f" — Stake £{stake_f:.2f} — Implied Return £{implied_return:,.2f}"

            # Surface decisive scores/grade in the legs table
            md_lines.append(line)

            legs = a["legs"].copy()

            # Ensure canonical odds/prob columns for writer & EV fallback
            try:
                import numpy as _np, pandas as _pd
                if "od" not in legs.columns and "odds" in legs.columns:
                    legs["od"] = _pd.to_numeric(legs["odds"], errors="coerce")
                if "p_model" not in legs.columns:
                    # try common fallbacks
                    for k in ("prob","p","prob_over25","prob_btts","prob_ftr_home","prob_ftr_draw","prob_ftr_away"):
                        if k in legs.columns:
                            legs["p_model"] = _pd.to_numeric(legs[k], errors="coerce")
                            break
                # If slip EV/odds missing or zero, recompute from legs
                need_recompute = False
                try:
                    need_recompute = (a.get("ev") is None) or (not _np.isfinite(float(a.get("ev", 0.0))))
                except Exception:
                    need_recompute = True
                if need_recompute and "p_model" in legs.columns and ("od" in legs.columns or "odds" in legs.columns):
                    ocol = "od" if "od" in legs.columns else "odds"
                    p_prod = float(_np.prod(_pd.to_numeric(legs["p_model"], errors="coerce").dropna().astype(float)))
                    o_prod = float(_np.prod(_pd.to_numeric(legs[ocol], errors="coerce").dropna().astype(float)))
                    a["prob"] = p_prod
                    a["odds"] = o_prod
                    a["ev"]   = p_prod * o_prod - 1.0
            except Exception:
                pass

            # Add a per-row 'decisive_score' convenience column for display
            try:
                import numpy as _np
                if "__mkt" in legs.columns:
                    def _dec_row(r):
                        m = str(r.get("__mkt", "")).lower()
                        if "over" in m and "decisive_over_score" in legs.columns:
                            return r.get("decisive_over_score")
                        if "btts" in m and "decisive_btts_score" in legs.columns:
                            return r.get("decisive_btts_score")
                        return _np.nan
                    if "decisive_score" not in legs.columns:
                        legs["decisive_score"] = legs.apply(_dec_row, axis=1)
            except Exception:
                pass

            # Ensure od_source exists for display
            try:
                if "od_source" not in legs.columns:
                    legs["od_source"] = "market"
            except Exception:
                pass

            # Restrict to a readable set of columns, if present
            SHOW_COLS = [
                "Date","Time","League","Home","Away",
                "p_model","od","od_source","ev","edge","__mkt",
                "decisive_score","decisive_over_score","decisive_btts_score","decisive_grade",
                "adjusted_over25_confidence","over25_confidence",
                "adjusted_btts_confidence","btts_confidence"
            ]
            try:
                out = legs[[c for c in SHOW_COLS if c in legs.columns]].copy()
            except Exception:
                out = legs

            # Write legs with a graceful fallback if markdown needs 'tabulate'
            try:
                md_lines.append(out.to_markdown(index=False))
            except Exception:
                md_lines.append("```\n" + out.to_string(index=False) + "\n```")
            md_lines.append("")
        md_path = os.path.join(dst_dir, f"{title}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        with open(os.path.join(dst_dir, f"{title}.json"), "w", encoding="utf-8") as f:
            json.dump([{k:(str(v) if k=="legs" else v) for k,v in d.items()} for d in accas], f, indent=2)
        print(f"✅ Wrote betslips → {md_path}")

# === Mixed-pool composer with market-mix constraints ========================

def compose_best_accas_mixed(pools: list[tuple[str, pd.DataFrame]],
                              *,
                              acca_size: int,
                              top_k: int = 10,
                              max_ah_legs: int = 2,
                              diversify_by: str | None = None) -> list[dict]:
    """Compose accas from multiple candidate pools with constraints.
    Constraint sources (env overrides overlay_config):
      MAX_PER_LEAGUE (int)
      MAX_PER_MARKET (json mapping, e.g. {"over25":4,"btts":3,"ah":2})
      MAX_COMBINED_ODDS (float)
      MIN_COMBINED_PROB (float)
      MAX_LONGSHOTS (int) + LONGSHOT_ODDS (float)
    """
    import os, json

    # ---- Resolve constraints (env → overlay_config → defaults) ----
    def _env_int(key, dflt):
        raw = os.getenv(key, None)
        try:
            return int(raw) if raw is not None else int(dflt)
        except Exception:
            return int(dflt)

    def _env_float(key, dflt):
        raw = os.getenv(key, None)
        try:
            return float(raw) if raw is not None else float(dflt)
        except Exception:
            return float(dflt)

    def _env_json(key, dflt: dict):
        raw = os.getenv(key, None)
        if not raw:
            return dflt
        try:
            val = json.loads(raw)
            return val if isinstance(val, dict) else dflt
        except Exception:
            return dflt

    max_per_league  = _env_int("MAX_PER_LEAGUE", int(overlay_config.get("max_per_league", 999)))
    max_combined_odds = _env_float("MAX_COMBINED_ODDS", float(overlay_config.get("max_combined_odds", 1e12)))
    min_combined_prob = _env_float("MIN_COMBINED_PROB", float(overlay_config.get("min_combined_prob", 0.0)))
    max_longshots   = _env_int("MAX_LONGSHOTS", int(overlay_config.get("max_longshots", 999)))
    longshot_odds   = _env_float("LONGSHOT_ODDS", float(overlay_config.get("longshot_odds", 3.0)))
    max_per_market_map = _env_json("MAX_PER_MARKET", overlay_config.get("max_per_market", {}))

    # ---- Canonical per-row probability and odds preferences ----
    PROB_PREF = (
        "p_model",
        "adjusted_over25_confidence","over25_confidence","prob_over25",
        "adjusted_btts_confidence","btts_confidence","prob_btts",
        "prob_ftr_home","prob_ftr_draw","prob_ftr_away",
        "p_home","p_draw","p_away","home_win_proba","draw_win_proba"
    )
    ODDS_PREF = (
        "od","odds",
        "odds_ft_over25","odds_btts_yes",
        "odds_ft_home_team_win","odds_ft_draw","odds_ft_away_team_win",
        "odds_ah_home_minus15","odds_ah_home_minus25",
    )
    def _first_present_row(row: pd.Series, keys: tuple[str, ...]) -> str | None:
        for k in keys:
            if k in row.index and pd.notna(row[k]):
                return k
        return None

    # Treat both AH lines as one bucket 'ah'
    def _mkt_bucket(mkt: str) -> str:
        m = str(mkt or "").lower()
        return "ah" if m.startswith("ah") else m

    # ---- Merge and sort pools by EV ----
    merged: list[pd.DataFrame] = []
    for mkt, df in pools:
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        tmp = df.copy()
        tmp["__mkt"] = mkt
        # Normalize prob/od columns to canonical names for safety
        try:
            tmp = _map_prob_od_for_market(tmp, mkt)
        except Exception:
            pass
        merged.append(tmp)
    if not merged:
        return []
    all_cands = pd.concat(merged, axis=0, ignore_index=True)
    # --- Backfill canonical p_model/od across merged pools if missing/empty ---
    try:
        # p_model: fill when missing or entirely NaN
        if ("p_model" not in all_cands.columns) or pd.to_numeric(all_cands["p_model"], errors="coerce").isna().all():
            all_cands["p_model"] = all_cands.apply(lambda r: _row_prob(r), axis=1)

        # od: fill when missing or non‑positive/NaN
        need_od = False
        if "od" not in all_cands.columns:
            need_od = True
        else:
            _odv = pd.to_numeric(all_cands["od"], errors="coerce")
            need_od = _odv.le(0).all() or _odv.isna().all()
        if need_od:
            all_cands["od"] = all_cands.apply(lambda r: _row_odds(r), axis=1)

        # mark market odds where positive
        try:
            _odv = pd.to_numeric(all_cands.get("od"), errors="coerce")
            if _odv is not None:
                if "od_source" not in all_cands.columns:
                    all_cands["od_source"] = ""
                all_cands.loc[_odv > 0, "od_source"] = all_cands.loc[_odv > 0, "od_source"].replace("", "market")
        except Exception:
            pass

        # compute ev/edge if missing
        if "ev" not in all_cands.columns:
            with np.errstate(invalid="ignore"):
                all_cands["ev"] = pd.to_numeric(all_cands.get("p_model"), errors="coerce") * pd.to_numeric(all_cands.get("od"), errors="coerce") - 1.0
        if "edge" not in all_cands.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                pm = pd.to_numeric(all_cands.get("p_model"), errors="coerce")
                od = pd.to_numeric(all_cands.get("od"), errors="coerce")
                all_cands["edge"] = pm - (1.0 / od.replace(0, np.nan))
    except Exception:
        pass
    sort_cols = [c for c in ("ev","edge") if c in all_cands.columns]
    if sort_cols:
        all_cands = all_cands.sort_values(sort_cols, ascending=[False]*len(sort_cols)).reset_index(drop=True)
    # Debug: coverage of canonical prob/od columns
    try:
        import os
        if str(os.getenv("DEBUG_COMPOSER","0")).lower() in ("1","true","yes","y","on") or \
        str(os.getenv("VERBOSE_COMPOSER","0")).lower() in ("1","true","yes","y","on"):
            miss_p = pd.to_numeric(all_cands.get("p_model"), errors="coerce").isna().mean() if "p_model" in all_cands.columns else 1.0
            _odcol = "od" if "od" in all_cands.columns else ("odds" if "odds" in all_cands.columns else None)
            miss_o = pd.to_numeric(all_cands.get(_odcol), errors="coerce").isna().mean() if _odcol else 1.0
            print(f"🧪 composer coverage: p_model NaN={miss_p:.1%}, od NaN={miss_o:.1%}, rows={len(all_cands)}")
    except Exception:
        pass
    # Auto-relax diversify_by when the slate is effectively single-league
    try:
        n_leagues_present = pd.Series(all_cands.get("League")).nunique(dropna=True)
        if n_leagues_present is not None and n_leagues_present <= 1 and diversify_by in ("league","league_date"):
            print("ℹ️ composer: single-league slate detected → relaxing diversify_by constraints")
            diversify_by = None if diversify_by == "league" else "date"
    except Exception:
        pass

    # ---- Helpers to read per-row prob & odds ----
    PROB_PREF = (
        "p_model",
        "prob_over25","adjusted_over25_confidence","over25_confidence",
        "prob_btts","adjusted_btts_confidence","btts_confidence",
        "prob_ftr_home","confidence_home",
        "prob_ftr_draw","confidence_draw",
        "prob_ftr_away","confidence_away",
        "prob_ah_home_minus15","prob_ah_home_minus25",
    )

    ODDS_PREF = (
        "od","odds",
        "odds_over25","odds_ft_over25",
        "odds_btts_yes",
        "odds_ft_home_team_win","odds_ft_draw","odds_ft_away_team_win",
        "odds_ah_home_minus15","odds_ah_home_minus25",
    )
    def _row_prob(row):
        for c in PROB_PREF:
            if c in row.index and pd.notna(row[c]):
                try: return float(row[c])
                except Exception: continue
        return None
    def _row_odds(row):
        for c in ODDS_PREF:
            if c in row.index and pd.notna(row[c]):
                try: return float(row[c])
                except Exception: continue
        return None

    def _match_key(row):
        if {"home_team_name","away_team_name"}.issubset(row.index):
            return (str(row["home_team_name"]).lower(), str(row["away_team_name"]).lower())
        if {"Home","Away"}.issubset(row.index):
            return (str(row["Home"]).lower(), str(row["Away"]).lower())
        return (row.name,)

    def _group_key(row):
        if diversify_by is None: return None
        dl = diversify_by.lower()
        if dl == "league": return str(row.get("League"," ")).lower()
        if dl == "date": return str(row.get("Date", row.get("match_date"," "))).lower()
        if dl == "league_date": return (str(row.get("League"," ")), str(row.get("Date", row.get("match_date"," "))))
        return None

    accas: list[dict] = []
    seen = set()
    for idx, seed in all_cands.head(300).iterrows():
        sk = _match_key(seed)
        if sk in seen: continue
        seen.add(sk)
        acc = [seed]
        # per-slip counters
        ah_cnt = 1 if _mkt_bucket(seed.get("__mkt","")) == "ah" else 0
        market_cnt = {_mkt_bucket(seed.get("__mkt","")): 1}
        league_cnt = {}
        lg = str(seed.get("League"," ")).lower()
        if lg.strip(): league_cnt[lg] = 1
        longshots = 1 if (_to_float(_row_odds(seed)) > _to_float(longshot_odds)) else 0
        used_matches = {sk}
        used_groups = set([_group_key(seed)]) if diversify_by else set()

        for jdx, row in all_cands.iterrows():
            if len(acc) >= int(acca_size): break
            if jdx == idx: continue
            mk = _match_key(row)
            if mk in used_matches: continue
            gk = _group_key(row)
            if diversify_by and gk in used_groups and gk is not None: continue
            bucket = _mkt_bucket(row.get("__mkt",""))
            # per-market cap
            cap_mkt = int(max_per_market_map.get(bucket, 999))
            if market_cnt.get(bucket, 0) >= cap_mkt: continue
            # AH cap
            if bucket == "ah" and ah_cnt >= int(max_ah_legs): continue
            # per-league cap
            rlg = str(row.get("League"," ")).lower()
            if rlg.strip() and league_cnt.get(rlg, 0) >= int(max_per_league): continue
            # longshot cap
            ro = _row_odds(row) or 0.0
            if (_to_float(ro) > _to_float(longshot_odds)) and (longshots >= int(max_longshots)):
                continue
            # add row
            acc.append(row)
            used_matches.add(mk)
            if diversify_by and gk is not None: used_groups.add(gk)
            if bucket == "ah": ah_cnt += 1
            market_cnt[bucket] = market_cnt.get(bucket, 0) + 1
            if rlg.strip(): league_cnt[rlg] = league_cnt.get(rlg, 0) + 1
            if _to_float(ro) > _to_float(longshot_odds): longshots += 1

        # Second pass: relax grouping to fill remaining legs if needed
        if len(acc) < int(acca_size):
            try:
                for kdx, row2 in all_cands.iterrows():
                    if len(acc) >= int(acca_size):
                        break
                    if kdx == idx:
                        continue
                    mk2 = _match_key(row2)
                    if mk2 in used_matches:
                        continue
                    # ignore grouping constraint on this pass; still enforce AH/market/league caps
                    bucket2 = _mkt_bucket(row2.get("__mkt",""))
                    cap_mkt2 = int(max_per_market_map.get(bucket2, 999))
                    if market_cnt.get(bucket2, 0) >= cap_mkt2:
                        continue
                    is_ah2 = str(row2.get("__mkt","")).startswith("ah")
                    if is_ah2 and ah_cnt >= int(max_ah_legs):
                        continue
                    rlg2 = str(row2.get("League"," ")).lower()
                    if rlg2.strip() and league_cnt.get(rlg2, 0) >= int(max_per_league):
                        continue

                    acc.append(row2)
                    used_matches.add(mk2)
                    if is_ah2:
                        ah_cnt += 1
                    market_cnt[bucket2] = market_cnt.get(bucket2, 0) + 1
                    if rlg2.strip():
                        league_cnt[rlg2] = league_cnt.get(rlg2, 0) + 1
            except Exception:
                pass

        if len(acc) == int(acca_size):
            # Build a DataFrame for the selected legs
            acc_df = pd.DataFrame(acc).copy()

            # Ensure canonical probability and odds columns per leg
            if "p_model" not in acc_df.columns:
                acc_df["p_model"] = acc_df.apply(
                    lambda r: pd.to_numeric(r.get(_first_present_row(r, PROB_PREF), pd.NA), errors="coerce"),
                    axis=1
                )
            if "od" not in acc_df.columns:
                acc_df["od"] = acc_df.apply(
                    lambda r: pd.to_numeric(r.get(_first_present_row(r, ODDS_PREF), pd.NA), errors="coerce"),
                    axis=1
                )

            # Drop rows lacking either p_model or od to avoid NaNs in products
            acc_df = acc_df[
                pd.to_numeric(acc_df.get("p_model"), errors="coerce").notna() &
                pd.to_numeric(acc_df.get("od"), errors="coerce").notna()
            ].copy()

            # Refill if short: top up from best remaining rows that have both p_model and od,
            # respecting market/league/AH/longshot caps
            if len(acc_df) < int(acca_size):
                try:
                    # Track used match keys to avoid duplicate fixtures
                    used_mks = set()
                    for _, rr in acc_df.iterrows():
                        try:
                            used_mks.add(_match_key(rr))
                        except Exception:
                            used_mks.add(rr.name)

                    for _, r in all_cands.iterrows():
                        if len(acc_df) >= int(acca_size):
                            break

                        # Skip if already present by match
                        try:
                            mk = _match_key(r)
                        except Exception:
                            mk = r.name
                        if mk in used_mks:
                            continue

                        # Require both prob and odds
                        pv = pd.to_numeric(r.get("p_model"), errors="coerce")
                        ov = pd.to_numeric(r.get("od"), errors="coerce") if "od" in r else pd.to_numeric(r.get("odds"), errors="coerce")
                        if pd.isna(pv) or pd.isna(ov):
                            continue

                        # Respect per-market / AH / league / longshot caps
                        bucket = _mkt_bucket(r.get("__mkt", "")) if "__mkt" in r else ""
                        cap_mkt = int(max_per_market_map.get(bucket, 999))
                        if market_cnt.get(bucket, 0) >= cap_mkt:
                            continue

                        is_ah = (bucket == "ah")
                        if is_ah and ah_cnt >= int(max_ah_legs):
                            continue

                        rlg = str(r.get("League", " ")).lower()
                        if rlg.strip() and league_cnt.get(rlg, 0) >= int(max_per_league):
                            continue

                        # Longshot cap
                        try:
                            ro = _row_odds(r) or ov
                        except Exception:
                            ro = ov
                        if (ro is not None) and (_to_float(ro) > _to_float(longshot_odds)) and (longshots >= int(max_longshots)):
                            continue

                        # All good → add the row
                        add = r.copy()
                        if "od" not in add and "odds" in add:
                            add["od"] = pd.to_numeric(add["odds"], errors="coerce")
                        acc_df = pd.concat([acc_df, pd.DataFrame([add])], ignore_index=True)

                        # Update used match keys / counters
                        used_mks.add(mk)
                        market_cnt[bucket] = market_cnt.get(bucket, 0) + 1
                        if is_ah:
                            ah_cnt += 1
                        if rlg.strip():
                            league_cnt[rlg] = league_cnt.get(rlg, 0) + 1
                        try:
                            if _to_float(ro) > _to_float(longshot_odds):
                                longshots += 1
                        except Exception:
                            pass
                except Exception:
                    pass

            # If still short, abandon this slip
            if len(acc_df) != int(acca_size):
                continue

            # Compute per-leg EV/edge if missing
            if "ev" not in acc_df.columns:
                acc_df["ev"] = acc_df["p_model"].astype(float) * acc_df["od"].astype(float) - 1.0
            if "edge" not in acc_df.columns:
                with np.errstate(divide="ignore", invalid="ignore"):
                    acc_df["edge"] = acc_df["p_model"].astype(float) - (1.0 / acc_df["od"].astype(float))

            # Combined slip prob/odds/EV
            from math import prod
            p = float(prod(acc_df["p_model"].astype(float).tolist()))
            o = float(prod(acc_df["od"].astype(float).tolist()))
            ev = p * o - 1.0

            # Final slip-level constraints
            if o > float(max_combined_odds): 
                continue
            if p < float(min_combined_prob): 
                continue

            # Canonical leg columns for printing/export
            KEEP_COLS = [
                "Date","Time","League","Home","Away",
                "p_model","od","ev","edge","__mkt",
                "adjusted_over25_confidence","over25_confidence","prob_over25",
                "adjusted_btts_confidence","btts_confidence","prob_btts",
                "prob_ftr_home","prob_ftr_draw","prob_ftr_away",
                "decisive_over_score","decisive_btts_score","decisive_grade",
            ]
            legs = acc_df[[c for c in KEEP_COLS if c in acc_df.columns]].copy()

            accas.append({
                "legs": legs,
                "prob": p,
                "odds": o,   # writer expects 'odds' for the header line
                "ev": ev,
                "_mix": {"ah": ah_cnt, "per_mkt": market_cnt, "per_league": league_cnt}
            })
        if len(accas) >= int(top_k): break
    return sorted(accas, key=lambda a: a["ev"], reverse=True)

# === Mixed-pool composer with market-mix constraints ========================
if "compose_best_accas_mixed" not in globals():
    def compose_best_accas_mixed(pools: list[tuple[str, pd.DataFrame]],
                                  *,
                                  acca_size: int,
                                  top_k: int = 10,
                                  max_ah_legs: int = 2,
                                  diversify_by: str | None = None) -> list[dict]:
        """Compose accas from multiple candidate pools with constraints.
        - pools: list of (market_key, candidates_df) where each df carries ev/edge and attrs _prob_col/_odds_col.
        - Enforces: at most `max_ah_legs` where market_key startswith 'ah'.
        - Uniqueness: avoid duplicate matches (by (Home, Away) or (home_team_name, away_team_name)).
        - Optional diversification: avoid adding multiple legs from same (League, Date) when diversify_by is set.
        """
        merged: list[pd.DataFrame] = []
        for mkt, df in pools:
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            tmp = df.copy()
            tmp["__mkt"] = mkt
            # Normalize prob/od columns to canonical names for safety
            try:
                tmp = _map_prob_od_for_market(tmp, mkt)
            except Exception:
                pass
            tmp.attrs["_prob_col"] = df.attrs.get("_prob_col")
            tmp.attrs["_odds_col"] = df.attrs.get("_odds_col")
            merged.append(tmp)
        if not merged:
            return []

        all_cands = pd.concat(merged, axis=0, ignore_index=True)
        # --- Backfill canonical p_model/od across merged pools if missing/empty ---
        try:
            # p_model: fill when missing or entirely NaN
            def _row_prob(row):
                for c in (
                    "p_model",
                    "prob_over25","adjusted_over25_confidence","over25_confidence",
                    "prob_btts","adjusted_btts_confidence","btts_confidence",
                    "prob_ftr_home","confidence_home",
                    "prob_ftr_draw","confidence_draw",
                    "prob_ftr_away","confidence_away",
                    "prob_ah_home_minus15","prob_ah_home_minus25",
                ):
                    if c in row.index and pd.notna(row[c]):
                        try: return float(row[c])
                        except Exception: continue
                return None
            def _row_odds(row):
                for c in (
                    "od","odds",
                    "odds_over25","odds_ft_over25",
                    "odds_btts_yes",
                    "odds_ft_home_team_win","odds_ft_draw","odds_ft_away_team_win",
                    "odds_ah_home_minus15","odds_ah_home_minus25",
                ):
                    if c in row.index and pd.notna(row[c]):
                        try: return float(row[c])
                        except Exception: continue
                return None
            if ("p_model" not in all_cands.columns) or pd.to_numeric(all_cands["p_model"], errors="coerce").isna().all():
                all_cands["p_model"] = all_cands.apply(lambda r: _row_prob(r), axis=1)
            need_od = False
            if "od" not in all_cands.columns:
                need_od = True
            else:
                _odv = pd.to_numeric(all_cands["od"], errors="coerce")
                need_od = _odv.le(0).all() or _odv.isna().all()
            if need_od:
                all_cands["od"] = all_cands.apply(lambda r: _row_odds(r), axis=1)
            try:
                _odv = pd.to_numeric(all_cands.get("od"), errors="coerce")
                if _odv is not None:
                    if "od_source" not in all_cands.columns:
                        all_cands["od_source"] = ""
                    all_cands.loc[_odv > 0, "od_source"] = all_cands.loc[_odv > 0, "od_source"].replace("", "market")
            except Exception:
                pass
            if "ev" not in all_cands.columns:
                with np.errstate(invalid="ignore"):
                    all_cands["ev"] = pd.to_numeric(all_cands.get("p_model"), errors="coerce") * pd.to_numeric(all_cands.get("od"), errors="coerce") - 1.0
            if "edge" not in all_cands.columns:
                with np.errstate(divide="ignore", invalid="ignore"):
                    pm = pd.to_numeric(all_cands.get("p_model"), errors="coerce")
                    od = pd.to_numeric(all_cands.get("od"), errors="coerce")
                    all_cands["edge"] = pm - (1.0 / od.replace(0, np.nan))
        except Exception:
            pass
        # Sort by EV desc, then edge, then prob if present
        any_prob = next((d.attrs.get("_prob_col") for d in merged if d.attrs.get("_prob_col")), None)
        sort_cols = [c for c in ("ev","edge", any_prob) if c in all_cands.columns]
        all_cands = all_cands.sort_values(sort_cols, ascending=[False]*len(sort_cols)).reset_index(drop=True)
        # Debug: coverage of canonical prob/od columns
        try:
            import os
            if str(os.getenv("DEBUG_COMPOSER","0")).lower() in ("1","true","yes","y","on") or \
            str(os.getenv("VERBOSE_COMPOSER","0")).lower() in ("1","true","yes","y","on"):
                miss_p = pd.to_numeric(all_cands.get("p_model"), errors="coerce").isna().mean() if "p_model" in all_cands.columns else 1.0
                _odcol = "od" if "od" in all_cands.columns else ("odds" if "odds" in all_cands.columns else None)
                miss_o = pd.to_numeric(all_cands.get(_odcol), errors="coerce").isna().mean() if _odcol else 1.0
                print(f"🧪 composer coverage: p_model NaN={miss_p:.1%}, od NaN={miss_o:.1%}, rows={len(all_cands)}")
        except Exception:
            pass

        def _match_key(row):
            if {"home_team_name","away_team_name"}.issubset(row.index):
                return (str(row["home_team_name"]).lower(), str(row["away_team_name"]).lower())
            if {"Home","Away"}.issubset(row.index):
                return (str(row["Home"]).lower(), str(row["Away"]).lower())
            return (row.name,)

        def _group_key(row):
            if diversify_by is None:
                return None
            dl = diversify_by.lower()
            if dl == "league":
                return str(row.get("League","")).lower()
            if dl == "date":
                return str(row.get("Date", row.get("match_date",""))).lower()
            if dl == "league_date":
                return (str(row.get("League","")), str(row.get("Date", row.get("match_date",""))))
            return None

        accas: list[dict] = []
        seen = set()
        for idx, seed in all_cands.head(300).iterrows():
            sk = _match_key(seed)
            if sk in seen:
                continue
            seen.add(sk)
            acc = [seed]
            ah_cnt = 1 if str(seed.get("__mkt",""))[:2] == "ah" else 0
            used_matches = {sk}
            used_groups = set([_group_key(seed)]) if diversify_by else set()
            for jdx, row in all_cands.iterrows():
                if len(acc) >= int(acca_size):
                    break
                if jdx == idx:
                    continue
                mk = _match_key(row)
                if mk in used_matches:
                    continue
                gk = _group_key(row)
                if diversify_by and gk in used_groups and gk is not None:
                    continue
                is_ah = str(row.get("__mkt",""))[:2] == "ah"
                if is_ah and ah_cnt >= int(max_ah_legs):
                    continue
                acc.append(row)
                used_matches.add(mk)
                if diversify_by and gk is not None:
                    used_groups.add(gk)
                if is_ah:
                    ah_cnt += 1
            if len(acc) == int(acca_size):
                # --- Canonical slip aggregation using p_model + od across mixed markets ---
                acc_df = pd.DataFrame(acc).copy()

                # Ensure canonical prob column per row
                if "p_model" not in acc_df.columns:
                    _prob_candidates = (
                        "p_model",
                        "prob_over25","prob_btts","prob_ftr_home","prob_ftr_draw","prob_ftr_away",
                        "adjusted_over25_confidence","adjusted_btts_confidence",
                        "confidence_home","confidence_draw","confidence_away",
                    )
                    for _c in _prob_candidates:
                        if _c in acc_df.columns:
                            acc_df["p_model"] = pd.to_numeric(acc_df[_c], errors="coerce")
                            break

                # Ensure canonical odds column per row
                if "od" not in acc_df.columns:
                    if "odds" in acc_df.columns:
                        acc_df["od"] = pd.to_numeric(acc_df["odds"], errors="coerce")
                    else:
                        for _oc in ("odds_over25","odds_btts_yes",
                                    "odds_ft_home_team_win","odds_ft_draw","odds_ft_away_team_win"):
                            if _oc in acc_df.columns:
                                acc_df["od"] = pd.to_numeric(acc_df[_oc], errors="coerce")
                                break

                # Drop rows lacking either p_model or od to avoid NaNs in products
                acc_df = acc_df[
                    pd.to_numeric(acc_df.get("p_model"), errors="coerce").notna() &
                    pd.to_numeric(acc_df.get("od"), errors="coerce").notna()
                ].copy()
                if len(acc_df) != int(acca_size):
                    continue

                # Compute slip-level prob & odds
                try:
                    p = float(np.prod(pd.to_numeric(acc_df["p_model"], errors="coerce").clip(0,1).values))
                    o = float(np.prod(pd.to_numeric(acc_df["od"], errors="coerce").values))
                except Exception:
                    continue
                if not np.isfinite(o) or o <= 0:
                    continue
                ev = p * o - 1.0

                # Legs payload (keep decisive & canonical cols)
                base_cols = (
                    "Date","Time","League","Home","Away","__mkt",
                    "ev","edge","p_model","od",
                    "prob_over25","prob_btts","prob_ftr_home","prob_ftr_draw","prob_ftr_away",
                    "adjusted_over25_confidence","adjusted_btts_confidence",
                    "decisive_over_score","decisive_btts_score","decisive_grade",
                )
                keep_cols = [c for c in base_cols if c in acc_df.columns]
                legs = acc_df[keep_cols].copy()

                accas.append({
                    "legs": legs,
                    "prob": p,
                    "odds": o,
                    "ev": ev
                })
            if len(accas) >= int(top_k):
                break
        return sorted(accas, key=lambda a: a["ev"], reverse=True)

# === Convenience: compose and write betslips including AH pools ==============
if "compose_betslips_with_ah" not in globals():
    def compose_betslips_with_ah(df: pd.DataFrame,
                             league_name: str,
                             out_dir: str,
                             *,
                             min_edge: float = 0.02,
                             max_odds: float | None = 3.50,
                             sizes: tuple[int, ...] | None = None,   # <— changed to optional
                             top_k_per_size: int = 5,
                             stake: float | None = None,
                             max_ah_legs_per_slip: int = 2,
                             diversify_by: str | None = "league_date") -> None:
        """Build candidate pools (Over 2.5, BTTS, FTR home, AH -1.5/-2.5),
        compose accas per size with market-mix constraint, and write betslips.
        """
        try:
            work = df.copy()
            # Prefer trained AH cover models; fallback to Poisson attach if missing
            try:
                from side_prob_models import attach_ah_probs_from_models as _attach_ah_models
                if isinstance(league_name, str) and league_name.strip():
                    work = _attach_ah_models(work, league_name=league_name)
            except Exception:
                pass
            # Ensure AH probs exist even if no trained model is available
            try:
                if "prob_ah_home_minus15" not in work.columns:
                    work = attach_prob_ah_minus15(work)
                if "prob_ah_home_minus25" not in work.columns:
                    work = attach_prob_ah_minus25(work)
            except Exception:
                pass
            # Normalise odds & implied/edge
            work = _ensure_odds_and_edges(work)

            # WTN proxies & synth odds before decisive scorer
            try:
                work = attach_win_to_nil_proxy(work)
            except Exception:
                pass
            try:
                work = attach_synth_odds(work)
            except Exception:
                pass

            # Compute Decisive Over/BTTS scores once so pools & writer can use them
            try:
                work = attach_decisive_over_btts_scores(work, league_name=league_name)
            except Exception:
                pass

            # ScenarioSupport gate (honor DISABLE_SCENARIO=1 to skip)
            try:
                import os
                _disable_scn = str(os.getenv("DISABLE_SCENARIO", "0")).strip().lower() in ("1","true","yes","y","on")
                _min_scn = float(os.getenv("SCENARIO_MIN", "0.0"))
            except Exception:
                _disable_scn, _min_scn = False, 0.0
            if not _disable_scn:
                work = _attach_scenario_support(work, min_scenario=_min_scn)
            # Multi-book consensus (if columns present) + CLV
            try:
                work = attach_multi_book_odds(work)
                work = attach_clv(work)
            except Exception:
                pass
            # NOTE: decisive_over_score / decisive_btts_score / decisive_grade are already on `work` from the single scorer call above
            pools: list[tuple[str, pd.DataFrame]] = []
            all_cands: dict[str, pd.DataFrame] = {}
            for key in (
                "over25", "under25",
                "btts", "btts_no",
                "ftr_home", "ftr_draw", "ftr_away",
                "ah_home_minus15", "ah_home_minus25",
                "wtn_home", "wtn_away",
            ):
                try:
                    cand = _prepare_candidates(work, key, min_edge, max_odds)
                    # Preserve the full pre-gate pool for fallback candidate dumps
                    try:
                        if isinstance(cand, pd.DataFrame):
                            try:
                                all_cands  # ensure it exists in this scope
                            except NameError:
                                all_cands = {}
                            all_cands[key] = cand.copy()
                    except Exception:
                        pass
                    if isinstance(cand, pd.DataFrame):
                        all_cands[key] = cand.copy()
                        _sz = len(cand)
                        if _sz > 0:
                            try:
                                print(f"🧪 pool[{key}] size={_sz}")
                            except Exception:
                                pass
                            # --- Canonicalize prob/odds on pool rows and carry diagnostics ---
                            tmp = cand.copy()
                            _pc = cand.attrs.get("_prob_col")
                            _oc = cand.attrs.get("_odds_col")
                            try:
                                if _pc and _pc in tmp.columns:
                                    tmp["p_model"] = pd.to_numeric(tmp[_pc], errors="coerce")
                                if _oc and _oc in tmp.columns:
                                    tmp["odds"] = pd.to_numeric(tmp[_oc], errors="coerce")
                                    # ensure 'od' alias exists for composers that prefer 'od'
                                    if "od" not in tmp.columns:
                                        tmp["od"] = tmp["odds"]
                            except Exception:
                                pass

                            # Market label used by composer & betslip formatter
                            if "__mkt" not in tmp.columns:
                                tmp["__mkt"] = key

                            # Propagate decisive diagnostics (if present)
                            try:
                                for col in (
                                    "decisive_over_score", "decisive_btts_score", "decisive_grade",
                                    "adjusted_over25_confidence", "adjusted_btts_confidence",
                                ):
                                    if col in cand.columns and col not in tmp.columns:
                                        tmp[col] = cand[col]
                            except Exception:
                                pass

                            pools.append((key, tmp))
                except Exception as _e:
                    try:
                        print(f"⚠️ pool[{key}] build failed: {_e}")
                    except Exception:
                        pass
                    continue
            if not pools:
                # Fallback: dump candidate pools for debugging when none are usable
                import datetime as _dt
                dstdir = out_dir or os.path.join("predictions_output", _dt.datetime.utcnow().strftime("%Y-%m-%d"))
                try:
                    import os
                    os.makedirs(dstdir, exist_ok=True)
                    tag = str(league_name).replace(" ", "_")
                    for _k, _df in (all_cands or {}).items():
                        if isinstance(_df, pd.DataFrame) and not _df.empty:
                            cand_df = _df.copy()
                            mkt = _k
                            # --- Normalize candidate pool before writing: ensure p_model/od/ev/edge present ---
                            try:
                                # 1) Map to canonical columns
                                cand_df = _map_prob_od_for_market(cand_df, mkt)

                                # 2) Fill canonical from attrs if present
                                _pcol = cand_df.attrs.get("_prob_col")
                                _ocol = cand_df.attrs.get("_odds_col")
                                import pandas as _pd, numpy as _np, os as _os

                                if ("p_model" not in cand_df.columns) and _pcol and _pcol in cand_df.columns:
                                    cand_df["p_model"] = _pd.to_numeric(cand_df[_pcol], errors="coerce")
                                if ("od" not in cand_df.columns) and _ocol and _ocol in cand_df.columns:
                                    cand_df["od"] = _pd.to_numeric(cand_df[_ocol], errors="coerce")

                                # 3) Synthesize odds if allowed and still missing/non-positive
                                _allow_syn = str(_os.getenv("ALLOW_SYNTH_ODDS","0")).strip().lower() in ("1","true","yes","y","on")
                                _odz = _pd.to_numeric(cand_df["od"], errors="coerce") if "od" in cand_df.columns else None
                                if _allow_syn and ("p_model" in cand_df.columns) and (_odz is None or _odz.le(0).all() or _odz.isna().all()):
                                    _pm = _pd.to_numeric(cand_df["p_model"], errors="coerce").replace(0, _pd.NA)
                                    cand_df["od"] = (1.0 / _pm).clip(lower=1.01)
                                    cand_df["od_source"] = "synth"

                                # 4) Default od_source to market when not set
                                if "od_source" not in cand_df.columns:
                                    cand_df["od_source"] = "market"
                                else:
                                    cand_df["od_source"] = cand_df["od_source"].fillna("market")

                                # 5) Recompute EV/edge if missing
                                if "p_model" in cand_df.columns and "od" in cand_df.columns:
                                    if "ev" not in cand_df.columns:
                                        cand_df["ev"] = _pd.to_numeric(cand_df["p_model"], errors="coerce") * _pd.to_numeric(cand_df["od"], errors="coerce") - 1.0
                                    if "edge" not in cand_df.columns:
                                        cand_df["edge"] = _pd.to_numeric(cand_df["p_model"], errors="coerce") - (1.0 / _pd.to_numeric(cand_df["od"], errors="coerce").replace(0, _np.nan))

                                # 6) Drop rows with non-positive odds
                                if "od" in cand_df.columns:
                                    cand_df = cand_df[_pd.to_numeric(cand_df["od"], errors="coerce") > 0].copy()

                            except Exception:
                                # Fail-open: write whatever we have, but we've tried to normalize
                                pass

                            # Debug summary for candidate pool after normalization
                            try:
                                _p_ok  = int(_pd.to_numeric(cand_df.get("p_model"), errors="coerce").notna().sum()) if "p_model" in cand_df.columns else 0
                                _od_ok = int((_pd.to_numeric(cand_df.get("od"), errors="coerce") > 0).sum()) if "od" in cand_df.columns else 0
                                print(f"🧪 candidates[{mkt}] → rows={len(cand_df)} | p_model_non_na={_p_ok} | od_pos={_od_ok} | od_source={cand_df.get('od_source').dropna().unique().tolist() if 'od_source' in cand_df.columns else []}")
                            except Exception:
                                pass

                            cand_df.to_csv(os.path.join(dstdir, f"{tag}_candidates_{_k}.csv"), index=False)
                    print(f"ℹ️ no accas composed (no pools); dumped candidate pools → {dstdir}")
                except Exception:
                    pass
                try:
                    import os
                    if os.getenv("REPORT_CONFIDENCE", "0") == "1":
                        summarize_confidence_and_write(work, league_name, dstdir)
                except Exception:
                    pass
                return
            # Decide acca sizes from overlay_config (preferred over the function argument)
            try:
                sizes_cfg = overlay_config.get("acca_sizes", [])
                if isinstance(sizes_cfg, (list, tuple)):
                    sizes_to_use = sorted({int(s) for s in sizes_cfg if isinstance(s, (int, float)) and int(s) > 0})
                else:
                    sizes_to_use = []
                if not sizes_to_use:
                    sizes_to_use = [6, 7]  # sensible default when not configured
            except Exception:
                sizes_to_use = [6, 7]
            try:
                print(f"🎛️ acca sizes in use: {sizes_to_use}")
            except Exception:
                pass

            # Normalize diversify_by if string "none" sneaks through
            if isinstance(diversify_by, str) and diversify_by.strip().lower() == "none":
                diversify_by = None

            accas: list[dict] = []
            for size in sizes_to_use:   # use sizes from overlay_config
                accas += compose_best_accas_mixed(
                    pools,
                    acca_size=int(size),
                    top_k=int(top_k_per_size),
                    max_ah_legs=int(max_ah_legs_per_slip),
                    diversify_by=diversify_by,
                )
            if not accas:
                # Fallback: dump candidate pools when no accas were produced
                try:
                    import os, datetime as _dt
                    dstdir = out_dir or os.path.join("predictions_output", _dt.datetime.utcnow().strftime("%Y-%m-%d"))
                    os.makedirs(dstdir, exist_ok=True)
                    tag = str(league_name).replace(" ", "_")
                    for _k, _df in (all_cands or {}).items():
                        if isinstance(_df, pd.DataFrame) and not _df.empty:
                            cand_df = _df.copy()
                            mkt = _k
                            # --- Normalize candidate pool before writing: ensure p_model/od/ev/edge present ---
                            try:
                                # 1) Map to canonical columns
                                cand_df = _map_prob_od_for_market(cand_df, mkt)

                                # 2) Fill canonical from attrs if present
                                _pcol = cand_df.attrs.get("_prob_col")
                                _ocol = cand_df.attrs.get("_odds_col")
                                import pandas as _pd, numpy as _np, os as _os

                                if ("p_model" not in cand_df.columns) and _pcol and _pcol in cand_df.columns:
                                    cand_df["p_model"] = _pd.to_numeric(cand_df[_pcol], errors="coerce")
                                if ("od" not in cand_df.columns) and _ocol and _ocol in cand_df.columns:
                                    cand_df["od"] = _pd.to_numeric(cand_df[_ocol], errors="coerce")

                                # 3) Synthesize odds if allowed and still missing/non-positive
                                _allow_syn = str(_os.getenv("ALLOW_SYNTH_ODDS","0")).strip().lower() in ("1","true","yes","y","on")
                                _odz = _pd.to_numeric(cand_df["od"], errors="coerce") if "od" in cand_df.columns else None
                                if _allow_syn and ("p_model" in cand_df.columns) and (_odz is None or _odz.le(0).all() or _odz.isna().all()):
                                    _pm = _pd.to_numeric(cand_df["p_model"], errors="coerce").replace(0, _pd.NA)
                                    cand_df["od"] = (1.0 / _pm).clip(lower=1.01)
                                    cand_df["od_source"] = "synth"

                                # 4) Default od_source to market when not set
                                if "od_source" not in cand_df.columns:
                                    cand_df["od_source"] = "market"
                                else:
                                    cand_df["od_source"] = cand_df["od_source"].fillna("market")

                                # 5) Recompute EV/edge if missing
                                if "p_model" in cand_df.columns and "od" in cand_df.columns:
                                    if "ev" not in cand_df.columns:
                                        cand_df["ev"] = _pd.to_numeric(cand_df["p_model"], errors="coerce") * _pd.to_numeric(cand_df["od"], errors="coerce") - 1.0
                                    if "edge" not in cand_df.columns:
                                        cand_df["edge"] = _pd.to_numeric(cand_df["p_model"], errors="coerce") - (1.0 / _pd.to_numeric(cand_df["od"], errors="coerce").replace(0, _np.nan))

                                # 6) Drop rows with non-positive odds
                                if "od" in cand_df.columns:
                                    cand_df = cand_df[_pd.to_numeric(cand_df["od"], errors="coerce") > 0].copy()

                            except Exception:
                                # Fail-open: write whatever we have, but we've tried to normalize
                                pass
                            # Debug summary for candidate pool after normalization
                            try:
                                _p_ok  = int(_pd.to_numeric(cand_df.get("p_model"), errors="coerce").notna().sum()) if "p_model" in cand_df.columns else 0
                                _od_ok = int((_pd.to_numeric(cand_df.get("od"), errors="coerce") > 0).sum()) if "od" in cand_df.columns else 0
                                print(f"🧪 candidates[{mkt}] → rows={len(cand_df)} | p_model_non_na={_p_ok} | od_pos={_od_ok} | od_source={cand_df.get('od_source').dropna().unique().tolist() if 'od_source' in cand_df.columns else []}")
                            except Exception:
                                pass
                            cand_df.to_csv(os.path.join(dstdir, f"{tag}_candidates_{_k}.csv"), index=False)
                    print(f"ℹ️ no accas composed; dumped candidate pools → {dstdir}")
                except Exception:
                    pass
                return
            accas = sorted(accas, key=lambda a: a["ev"], reverse=True)
            import os, datetime
            out_path = out_dir or os.path.join("predictions_output", datetime.datetime.utcnow().strftime("%Y-%m-%d"))
            os.makedirs(out_path, exist_ok=True)
            # Confidence snapshot for QA (opt-in)
            try:
                if os.getenv("REPORT_CONFIDENCE", "0") == "1":
                    summarize_confidence_and_write(work, league_name, out_path)
            except Exception:
                pass
            try:
                bankroll = float(os.getenv("KELLY_BANKROLL", "0"))
                bankroll = bankroll if bankroll > 0 else None
            except Exception:
                bankroll = None
            _st_raw = stake if stake is not None else overlay_config.get("stake_per_acca", 10)
            _st_f: float = _to_float(_st_raw)
            write_betslips(
                accas[:10], out_path,
                stake=_st_f,
                title=f"{league_name}_betslips",
                bankroll=bankroll
            )
        except Exception as _e:
            try:
                print(f"⚠️ compose_betslips_with_ah skipped: {_e}")
            except Exception:
                pass

# ------------------------------------------------------------------
# EV/edge/Kelly helper for consistent export columns
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# AH helper: attach probs/odds for a given line and return EV-ranked shortlist
# ------------------------------------------------------------------

def generate_ah_candidates(df: pd.DataFrame,
                            line: str = "-1.5",
                            *,
                            min_edge: float | None = None,
                            max_odds: float | None = None,
                            top_n: int = 50) -> pd.DataFrame:
    """
    Build an EV-ranked shortlist for Asian Handicap home lines (-1.5 or -2.5).
    Steps:
      1) attach prob_ah_home_minus{15|25} from lambdas.
      2) resolve/synth odds_ah_home_minus{15|25} using whitelist or 1/p.
      3) compute EV/edge/Kelly; filter with min_edge/max_odds; return top_n by EV.
    Safe no-op on empty frames.
    """
    import os
    import pandas as _pd

    if df is None or len(df) == 0:
        return _pd.DataFrame(columns=["League","match_date","Home","Away","prob","odds","edge","kelly_frac","expected_value"])  # empty

    out = df.copy()

    # Defaults from env/config
    if min_edge is None:
        try: min_edge = float(os.getenv("MIN_EDGE_AH", os.getenv("MIN_EDGE", str(config.get("min_edge", 0.0)))))
        except Exception: min_edge = 0.0
    if max_odds is None:
        try: max_odds = float(os.getenv("MAX_ODDS_AH", os.getenv("MAX_ODDS", str(config.get("max_leg_odds", 12.0)))))
        except Exception: max_odds = 12.0

    # Normalise numeric-likes & existing odds
    try: out = _coerce_numeric_like(out, odds_whitelist=ODD_COLUMN_WHITELIST)
    except Exception: pass

    # Attach probabilities for requested AH line
    line = str(line).strip()
    if line == "-1.5":
        out = attach_prob_ah_minus15(out)
        prob_col = "prob_ah_home_minus15"
        odds_key = "ah_home_minus15"
        default_odds_col = "odds_ah_home_minus15"
        odds_label = "AH -1.5 odds"
    elif line == "-2.5":
        out = attach_prob_ah_minus25(out)
        prob_col = "prob_ah_home_minus25"
        odds_key = "ah_home_minus25"
        default_odds_col = "odds_ah_home_minus25"
        odds_label = "AH -2.5 odds"
    else:
        # Unsupported line for now
        return _pd.DataFrame(columns=["League","match_date","Home","Away","prob","odds","edge","kelly_frac","expected_value"])  # empty

    # Ensure odds columns exist or synthesise if allowed
    try:
        odds_col = _resolve_odds(out, ODD_COLUMN_WHITELIST.get(odds_key, []), label=odds_label)
    except Exception:
        odds_col = None

    synth_used = False
    if not odds_col and bool(config.get("allow_synth_odds", False)) and prob_col in out.columns:
        try:
            created = _synthesise_odds_from_probs(out, prob_col, default_odds_col)
            odds_col = created or odds_col
            synth_used = bool(created)
        except Exception:
            pass

    if not odds_col or prob_col not in out.columns:
        # Nothing to rank
        return _pd.DataFrame(columns=["League","match_date","Home","Away","prob","odds","edge","kelly_frac","expected_value"])  # empty

    # Estimate book margin from 1X2 to optionally haircut fair odds in EV
    try:
        margin = _estimate_book_margin(out)
    except Exception:
        margin = 0.0

    # Attach EV columns (generic names), then map into AH-specific view
    tmp = out[[prob_col, odds_col]].copy()
    tmp = _attach_ev_columns(tmp, prob_col, odds_col, margin=margin,
                             probs_from_odds=False, synth_odds=synth_used)

    # Merge back minimal identity for output
    cols_id = [c for c in ("League","match_date","Home","Away") if c in out.columns]
    res = out[cols_id].copy() if cols_id else _pd.DataFrame(index=out.index)
    res["prob"] = tmp["p_model"].astype(float)
    res["odds"] = tmp["odds"].astype(float)
    res["edge"] = tmp["edge"].astype(float)
    res["kelly_frac"] = tmp["kelly_frac"].astype(float)
    res["expected_value"] = tmp["expected_value"].astype(float)

    # EV/odds gates
    try:
        o = _pd.to_numeric(res["odds"], errors="coerce")
        e = _pd.to_numeric(res["expected_value"], errors="coerce")
        keep = (e >= float(min_edge)) & (o <= float(max_odds))
        res = res.loc[keep].copy()
    except Exception:
        pass

    # Rank & return top_n
    res = res.sort_values(["expected_value","edge","prob"], ascending=[False, False, False])
    if isinstance(top_n, int) and top_n > 0:
        res = res.head(top_n)
    res.reset_index(drop=True, inplace=True)
    return res
# --- Ensure Poisson totals probs exist (overlay variant uses λ̂ or exp_goals_sum) ---
if "_ensure_poisson_over_probs_inplace_overlay" not in globals():
    def _ensure_poisson_over_probs_inplace_overlay(df: pd.DataFrame) -> pd.DataFrame:
        import pandas as _pd, numpy as _np
        if not isinstance(df, _pd.DataFrame) or df.empty:
            return df
        # Prefer exp_goals_sum; else sum lambdas if available
        exg = _pd.to_numeric(df.get("exp_goals_sum"), errors="coerce")
        if exg is None or exg.isna().all():
            h = _pd.to_numeric(df.get("home_goals_pred"), errors="coerce")
            a = _pd.to_numeric(df.get("away_goals_pred"), errors="coerce")
            if h is not None and a is not None:
                exg = (h.fillna(0) + a.fillna(0))
        if exg is None or (hasattr(exg, "isna") and exg.isna().all()):
            return df
        with _np.errstate(over="ignore", invalid="ignore"):
            eL = _np.exp(-exg)
            P0 = eL
            P1 = eL * exg
            P2 = eL * (exg**2) / 2.0
            P3 = eL * (exg**3) / 6.0
            P4 = eL * (exg**4) / 24.0
            p_over15 = 1.0 - (P0 + P1)                  # ≥2 goals
            p_over35 = 1.0 - (P0 + P1 + P2 + P3)        # ≥4 goals
            p_over45 = 1.0 - (P0 + P1 + P2 + P3 + P4)   # ≥5 goals
        def _set(name, vals):
            if (name not in df.columns) or _pd.to_numeric(df.get(name), errors="coerce").isna().all():
                df[name] = vals
        _set("prob_over15", p_over15)
        _set("prob_over35", p_over35)
        _set("prob_over45", p_over45)
        return df

# === NEW: Poisson-based cover prob for AH -1.5 (win by 2+) ===
def prob_cover_minus15(lambda_home: float, lambda_away: float, max_goals: int = 10) -> float:
    p = 0.0
    # sum P(h - a >= 2)
    for h in range(0, max_goals + 1):
        # Poisson prob for home h
        ph = math.exp(-lambda_home) * (lambda_home ** h) / math.factorial(h)
        for a in range(0, max_goals + 1):
            if h - a >= 2:
                # Poisson prob for away a
                pa = math.exp(-lambda_away) * (lambda_away ** a) / math.factorial(a)
                p += ph * pa
    # Clamp into [0,1]
    return max(0.0, min(1.0, p))

# Vectoriser: attach P(cover -1.5) as 'prob_ah_home_minus15' if lambdas exist

def attach_prob_ah_minus15(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    if "lambda_home" not in df.columns or "lambda_away" not in df.columns:
        return df
    lh = pd.to_numeric(df["lambda_home"], errors="coerce").fillna(1.2)
    la = pd.to_numeric(df["lambda_away"], errors="coerce").fillna(1.0)
    vals = [prob_cover_minus15(float(h), float(a)) for h, a in zip(lh, la)]
    df = df.copy()
    df["prob_ah_home_minus15"] = np.clip(np.asarray(vals, dtype=float), 0.0, 1.0)
    return df

# === NEW: Poisson-based cover prob for AH -2.5 (win by 3+) ===

def prob_cover_minus25(lambda_home: float, lambda_away: float, max_goals: int = 10) -> float:
    p = 0.0
    # sum P(h - a >= 3)
    for h in range(0, max_goals + 1):
        ph = math.exp(-lambda_home) * (lambda_home ** h) / math.factorial(h)
        for a in range(0, max_goals + 1):
            if h - a >= 3:
                pa = math.exp(-lambda_away) * (lambda_away ** a) / math.factorial(a)
                p += ph * pa
    return max(0.0, min(1.0, p))

# Vectoriser: attach P(cover -2.5) as 'prob_ah_home_minus25' if lambdas exist

def attach_prob_ah_minus25(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    if "lambda_home" not in df.columns or "lambda_away" not in df.columns:
        return df
    lh = pd.to_numeric(df["lambda_home"], errors="coerce").fillna(1.2)
    la = pd.to_numeric(df["lambda_away"], errors="coerce").fillna(1.0)
    vals = [prob_cover_minus25(float(h), float(a)) for h, a in zip(lh, la)]
    df = df.copy()
    df["prob_ah_home_minus25"] = np.clip(np.asarray(vals, dtype=float), 0.0, 1.0)
    return df
    
# --- Public: apply EV gating for O1.5 / O3.5 / O4.5 (overlay exports) ---
if "apply_ev_filters_for_o15_o35_o45" not in globals():
    def apply_ev_filters_for_o15_o35_o45(df: pd.DataFrame, *, min_edge: float | None = None, max_odds: float | None = None) -> pd.DataFrame:
        import os
        import pandas as _pd
        if df is None or len(df) == 0:
            return df
        df = df.copy()
        # Pull defaults from env or overlay config
        if min_edge is None:
            try: min_edge = float(os.getenv("MIN_EDGE", str(config.get("min_edge", 0.0))))
            except Exception: min_edge = 0.0
        if max_odds is None:
            try: max_odds = float(os.getenv("MAX_ODDS", str(config.get("max_leg_odds", 12.0))))
            except Exception: max_odds = 12.0
        # Ensure numeric coercions and Poisson fallback
        try: df = _coerce_numeric_like(df, odds_whitelist=ODD_COLUMN_WHITELIST)
        except Exception: pass
        try: df = _ensure_poisson_over_probs_inplace_overlay(df)
        except Exception: pass
        # Helper to attach EV and mask non-keepers
        def _ev_filter(prob_col: str, odds_key: str, ev_col: str):
            odds_col = _resolve_odds(df, ODD_COLUMN_WHITELIST.get(odds_key, []), label=f"{odds_key.upper()} odds")
            if odds_col is None or prob_col not in df.columns:
                return
            p = _pd.to_numeric(df[prob_col], errors="coerce").fillna(0.0)
            o = _pd.to_numeric(df[odds_col], errors="coerce")
            df[ev_col] = p * (o - 1.0) - (1.0 - p)
            keep = (df[ev_col] >= float(min_edge)) & (o <= float(max_odds))
            # Mask out-of-gate rows by nulling the EV and leaving other columns intact
            df.loc[~keep, ev_col] = _pd.NA
        # Apply per market
        _ev_filter("prob_over15", "over15", "ev_o15")
        _ev_filter("prob_over35", "over35", "ev_o35")
        _ev_filter("prob_over45", "over45", "ev_o45")
        return df
# --- Convenience: prepare a single deployable overlay frame ------------------
if "prepare_deployable_overlay" not in globals():
    pass
def prepare_deployable_overlay(df: pd.DataFrame,
                               *,
                               min_edge: float | None = None,
                               max_odds: float | None = None,
                               include_markets: tuple[str, ...] = ("o15","o25","o35","o45","btts","ah15","ah25")) -> pd.DataFrame:
        """
        Build a deployable view from a raw league dataframe using overlay-only steps.
        Pipeline:
          1) ensure_minimal_signals (FTR probs + λ̂ seeds)
          2) _ensure_poisson_over_probs_inplace_overlay (prob_over15/35/45)
          3) apply_ev_filters_for_o15_o35_o45 (ev_o15/o35/o45 with MIN_EDGE/MAX_ODDS)
          4) EV for O2.5 and BTTS (ev_o25, ev_btts) with same thresholds
        Returns a filtered dataframe with rows that pass at least one EV gate.
        """
        import os
        import pandas as _pd

        if df is None or len(df) == 0:
            return df

        out = df.copy()

        # Defaults
        if min_edge is None:
            try: min_edge = float(os.getenv("MIN_EDGE", str(config.get("min_edge", 0.0))))
            except Exception: min_edge = 0.0
        if max_odds is None:
            try: max_odds = float(os.getenv("MAX_ODDS", str(config.get("max_leg_odds", 12.0))))
            except Exception: max_odds = 12.0

        # 1) Minimal signals (no models required)
        try: out = apply_safe_renames_and_whitelist(out)
        except Exception: pass
        try: out = _coerce_numeric_like(out, odds_whitelist=ODD_COLUMN_WHITELIST)
        except Exception: pass
        try: out = ensure_minimal_signals(out)
        except Exception: pass
        # --- Ensure standard odds + implied/edge columns for all markets
        try: out = _ensure_odds_and_edges(out)
        except Exception: pass

        # 2) Poisson totals fallback for tails (prob_over15/35/45)
        try: out = _ensure_poisson_over_probs_inplace_overlay(out)
        except Exception: pass

        # 3) Attach EV masks for O1.5/O3.5/O4.5
        try: out = apply_ev_filters_for_o15_o35_o45(out, min_edge=min_edge, max_odds=max_odds)
        except Exception: pass

        # 4) EV for O2.5 and BTTS using available probs/odds
        try: out = _ensure_report_probs_inplace(out)
        except Exception: pass
        try: out = _ensure_roi_inputs_inplace(out)
        except Exception: pass

        # Resolve odds columns for O2.5/BTTS
        o25_col = _resolve_odds(out, ODD_COLUMN_WHITELIST.get("over25", []), label="Over 2.5 odds")
        bt_col  = _resolve_odds(out, ODD_COLUMN_WHITELIST.get("btts_yes", []), label="BTTS YES odds")

        # Compute EVs with gates; mask non-keepers to NA
        if o25_col and "prob_over25" in out.columns:
            p = _pd.to_numeric(out["prob_over25"], errors="coerce").fillna(0.0)
            o = _pd.to_numeric(out[o25_col], errors="coerce")
            out["ev_o25"] = p * (o - 1.0) - (1.0 - p)
            keep = (out["ev_o25"] >= float(min_edge)) & (o <= float(max_odds))
            out.loc[~keep, "ev_o25"] = _pd.NA

        if bt_col and "prob_btts" in out.columns:
            p = _pd.to_numeric(out["prob_btts"], errors="coerce").fillna(0.0)
            o = _pd.to_numeric(out[bt_col], errors="coerce")
            out["ev_btts"] = p * (o - 1.0) - (1.0 - p)
            keep = (out["ev_btts"] >= float(min_edge)) & (o <= float(max_odds))
            out.loc[~keep, "ev_btts"] = _pd.NA

        # 5) EV for AH Home -1.5 and -2.5
        # Prefer trained AH models for probabilities if possible (fallback to Poisson later)
        try:
            # Resolve league name best-effort
            _ln = df.attrs.get("league_name") or df.attrs.get("league")
            if not _ln and "League" in out.columns:
                try:
                    _ln = str(out["League"].mode().iloc[0])
                except Exception:
                    _ln = None
            if _ln:
                from side_prob_models import attach_ah_probs_from_models as _attach_ah_models
                out = _attach_ah_models(out, league_name=str(_ln))
        except Exception:
            pass

        # Attach probabilities from lambdas if still missing
        try:
            if "prob_ah_home_minus15" not in out.columns:
                out = attach_prob_ah_minus15(out)
            if "prob_ah_home_minus25" not in out.columns:
                out = attach_prob_ah_minus25(out)
        except Exception:
            pass

        # Resolve/synth odds for AH -1.5
        ah15_col = None
        try:
            ah15_col = _resolve_odds(out, ODD_COLUMN_WHITELIST.get("ah_home_minus15", []), label="AH -1.5 odds")
            if not ah15_col and bool(config.get("allow_synth_odds", False)) and ("prob_ah_home_minus15" in out.columns):
                created = _synthesise_odds_from_probs(out, "prob_ah_home_minus15", "odds_ah_home_minus15")
                ah15_col = created or ah15_col
        except Exception:
            ah15_col = None

        if ah15_col and "prob_ah_home_minus15" in out.columns:
            p = _pd.to_numeric(out["prob_ah_home_minus15"], errors="coerce").fillna(0.0)
            o = _pd.to_numeric(out[ah15_col], errors="coerce")
            out["ev_ah15"] = p * (o - 1.0) - (1.0 - p)
            keep = (out["ev_ah15"] >= float(min_edge)) & (o <= float(max_odds))
            out.loc[~keep, "ev_ah15"] = _pd.NA

        # Resolve/synth odds for AH -2.5
        ah25_col = None
        try:
            ah25_col = _resolve_odds(out, ODD_COLUMN_WHITELIST.get("ah_home_minus25", []), label="AH -2.5 odds")
            if not ah25_col and bool(config.get("allow_synth_odds", False)) and ("prob_ah_home_minus25" in out.columns):
                created = _synthesise_odds_from_probs(out, "prob_ah_home_minus25", "odds_ah_home_minus25")
                ah25_col = created or ah25_col
        except Exception:
            ah25_col = None

        if ah25_col and "prob_ah_home_minus25" in out.columns:
            p = _pd.to_numeric(out["prob_ah_home_minus25"], errors="coerce").fillna(0.0)
            o = _pd.to_numeric(out[ah25_col], errors="coerce")
            out["ev_ah25"] = p * (o - 1.0) - (1.0 - p)
            keep = (out["ev_ah25"] >= float(min_edge)) & (o <= float(max_odds))
            out.loc[~keep, "ev_ah25"] = _pd.NA

        # Build a single deployable view: union of chosen markets
        ev_cols = []
        if "o15" in include_markets: ev_cols.append("ev_o15")
        if "o25" in include_markets: ev_cols.append("ev_o25")
        if "o35" in include_markets: ev_cols.append("ev_o35")
        if "o45" in include_markets: ev_cols.append("ev_o45")
        if "btts" in include_markets: ev_cols.append("ev_btts")
        if "ah15" in include_markets: ev_cols.append("ev_ah15")
        if "ah25" in include_markets: ev_cols.append("ev_ah25")
        present = [c for c in ev_cols if c in out.columns]
        if not present:
            return out.iloc[0:0].copy()

        mask = _pd.DataFrame({c: _pd.to_numeric(out[c], errors="coerce") for c in present}).notna().any(axis=1)
        deploy = out.loc[mask].copy()

        # Optional summariser: which markets passed per row
        try:
            def _mk(row):
                ks = [k for k in present if _pd.notna(row.get(k))]
                return ",".join(ks) if ks else ""
            deploy["deploy_markets"] = deploy[present].apply(lambda r: _mk(r), axis=1)
        except Exception:
            pass

        return deploy
# ------------------------------------------------------------------
# Helper: robustly coalesce/format match_date (row + series variants)
# ------------------------------------------------------------------
def _coalesce_match_date_row(row) -> str:
    """
    Best-effort extraction of a usable YYYY-MM-DD date string from a row.
    Priority:
      1) match_date
      2) date_GMT
      3) date
      4) timestamp
      5) any column containing 'date' or 'time' that parses
    Returns "" if nothing usable is found.
    """
    import pandas as _pd
    def _is_blank(x):
        if x is None:
            return True
        try:
            if _pd.isna(x):
                return True
        except Exception:
            pass
        return str(x).strip() == ""

    for key in ("match_date", "date_GMT", "date", "timestamp"):
        v = row.get(key, None)
        if not _is_blank(v):
            try:
                return _pd.to_datetime(v).date().isoformat()
            except Exception:
                s = str(v).strip()
                if s:
                    return s

    try:
        for k, v in row.items():
            kl = str(k).lower()
            if ("date" in kl or "time" in kl) and not _is_blank(v):
                try:
                    return _pd.to_datetime(v).date().isoformat()
                except Exception:
                    s = str(v).strip()
                    if s:
                        return s
    except Exception:
        pass
    return ""

def _coalesce_match_date_series(df: pd.DataFrame) -> pd.Series:
    """Series-level version: returns a string Series (YYYY-MM-DD where parseable)."""
    import pandas as _pd
    if df is None or df.empty:
        return _pd.Series([], dtype="object")
    md = df.get("match_date")
    if md is None:
        md = _pd.Series([None] * len(df), index=df.index, dtype="object")
    else:
        md = md.astype(object)
    try:
        md = md.where(~md.astype(str).str.strip().eq(""), _pd.NA)
    except Exception:
        pass
    for _fb in ("date_GMT", "date", "timestamp"):
        if _fb in df.columns:
            cand = df[_fb]
            try:
                cand = cand.where(~cand.astype(str).str.strip().eq(""), _pd.NA)
            except Exception:
                pass
            md = md.where(~md.isna(), cand)
    if bool(md.isna().any()):
        for c in df.columns:
            cl = str(c).lower()
            if ("date" in cl or "time" in cl) and c not in ("match_date","date_GMT","date","timestamp"):
                cand = df[c]
                try:
                    cand = cand.where(~cand.astype(str).str.strip().eq(""), _pd.NA)
                except Exception:
                    pass
                md = md.where(~md.isna(), cand)
    try:
        parsed = _pd.to_datetime(md, errors="coerce")
        out = parsed.dt.date.astype("string")
        leftover = md.where(parsed.isna(), None)
        out = out.where(~out.isna(), leftover.astype("string"))
        out = out.fillna("")
        return out.astype(object)
    except Exception:
        return md.astype(str).fillna("")
    
# ------------------------------------------------------------------
# Ensure draw context + match_date are present for cross-league exports
# ------------------------------------------------------------------
def _ensure_draw_context_for_export(df: pd.DataFrame, league_name: str) -> pd.DataFrame:
    """Fill minimal draw context columns used by cross-league CSV when we
    haven't explicitly run the draw pipeline. Creates/fills:
        - p_draw  (from confidence_draw if missing)
        - draw_flag, thr_draw, mode_draw  (from per-league JSON)
        - draw_flag_topk, is_topk_best    (using BEST_K_PCT or per-league JSON)
        - match_date from fallback columns if empty
        - alpha_used, best_k_pct from module globals
    Safe no-op when inputs are unavailable.
    """
    if df is None or df.empty:
        return df
    out = df.copy()

    # ---- match_date fallback (per-row coalesce: date_GMT → date → timestamp) ----
    # We fill *per row* so partially blank columns get fixed too.
    if "match_date" not in out.columns:
        out["match_date"] = np.nan
    # treat empty strings as NaN for the purposes of coalescing
    _md = out["match_date"].astype(object)
    try:
        _md = _md.where(~_md.astype(str).str.strip().eq(""), np.nan)
    except Exception:
        pass
    out["match_date"] = _md
    for _fb in ("date_GMT", "date", "timestamp"):
        if _fb in out.columns:
            _cand = out[_fb]
            # only fill where match_date is missing/blank
            out["match_date"] = out["match_date"].where(~out["match_date"].isna(), _cand)
            try:
                out["match_date"] = out["match_date"].where(~out["match_date"].astype(str).str.strip().eq(""), _cand)
            except Exception:
                pass

    # Normalise to string date where possible (YYYY-MM-DD), keep raw otherwise
    try:
        out["match_date"] = _coalesce_match_date_series(out)
    except Exception:
        pass

    # ---- p_draw from confidence_draw when not provided ---------------------
    if "p_draw" not in out.columns or out["p_draw"].isna().all():
        if "confidence_draw" in out.columns:
            out["p_draw"] = pd.to_numeric(out["confidence_draw"], errors="coerce").clip(0, 1)

    # ---- draw threshold + flags -------------------------------------------
    if "p_draw" in out.columns:
        # Ensure per-league threshold/mode are available as attrs
        default_gate = constants.LOCKED_THRESHOLDS.get(league_name, {}).get("draw", 0.50)
        thr, mode = _load_draw_threshold(league_name, default_thr=float(default_gate))
        out.attrs["thr_draw"], out.attrs["mode_draw"] = float(thr), str(mode)

        # draw_flag (thresholded)
        try:
            out.loc[:, "draw_flag"] = (pd.to_numeric(out["p_draw"], errors="coerce") >= float(thr)).astype(int)
        except Exception:
            out.loc[:, "draw_flag"] = 0

        # top-k flag; use cached BEST_K_PCT if available
        try:
            k_pct = globals().get("BEST_K_PCT")
            if k_pct is None:
                k_pct = _load_best_topk_percent(league_name, default_pct=0.10)
            k = max(1, int(round(len(out) * float(k_pct))))
            order = np.argsort(-pd.to_numeric(out["p_draw"], errors="coerce").fillna(0.0).to_numpy())
            idx_top = set(out.index.take(order[:k]))
            out.loc[:, "draw_flag_topk"] = out.index.to_series().apply(lambda i: 1 if i in idx_top else 0).astype(int)
            out.loc[:, "is_topk_best"] = out["draw_flag_topk"].astype(int)
            out.attrs["topk_pct_draw"] = float(k_pct)
        except Exception:
            # fallbacks if anything goes sideways
            if "draw_flag_topk" not in out.columns:
                out.loc[:, "draw_flag_topk"] = 0
            if "is_topk_best" not in out.columns:
                out.loc[:, "is_topk_best"] = out["draw_flag_topk"].astype(int)

        # materialise thr_draw/mode_draw as columns for export when missing
        if "thr_draw" not in out.columns:
            out["thr_draw"] = float(out.attrs.get("thr_draw", thr))
        if "mode_draw" not in out.columns:
            out["mode_draw"] = str(out.attrs.get("mode_draw", mode))

    # ---- annotate training/inference context -------------------------------
    _alpha_used = globals().get("BASELINE_ALPHA_USED", None)
    _best_k_pct = globals().get("BEST_K_PCT", out.attrs.get("topk_pct_draw", None))
    if _alpha_used is not None and "alpha_used" not in out.columns:
        out["alpha_used"] = float(_alpha_used)
    if _best_k_pct is not None and "best_k_pct" not in out.columns:
        out["best_k_pct"] = float(_best_k_pct)

    return out


# --------------------------------------------------------------
# Odds resolution preview (debug logger)
# --------------------------------------------------------------

def log_resolved_odds_columns(df: pd.DataFrame) -> None:
    """Print which odds columns will be used for each supported market.
    Safe no-op if columns are missing. Does not modify df."""
    try:
        # BTTS YES / NO
        _ = _resolve_odds(df, ODD_COLUMN_WHITELIST['btts_yes'], label='BTTS YES odds')
        _ = _resolve_odds(df, ODD_COLUMN_WHITELIST['btts_no'],  label='BTTS NO odds')
        # Over/Under 2.5
        _ = _resolve_odds(df, ODD_COLUMN_WHITELIST['over25'],  label='Over 2.5 odds')
        _ = _resolve_odds(df, ODD_COLUMN_WHITELIST['under25'], label='Under 2.5 odds')
        # FTR legs
        _ = _resolve_odds(df, ODD_COLUMN_WHITELIST['home_win'], label='FTR HOME odds')
        _ = _resolve_odds(df, ODD_COLUMN_WHITELIST['draw'],     label='FTR DRAW odds')
        _ = _resolve_odds(df, ODD_COLUMN_WHITELIST['away_win'], label='FTR AWAY odds')
        # Win-to-Nil
        _ = _resolve_odds(df, ODD_COLUMN_WHITELIST['home_wtn'], label='HOME WTN odds')
        _ = _resolve_odds(df, ODD_COLUMN_WHITELIST['away_wtn'], label='AWAY WTN odds')
        # Clean sheet
        _ = _resolve_odds(df, ODD_COLUMN_WHITELIST['home_cs'],  label='HOME clean sheet odds')
        _ = _resolve_odds(df, ODD_COLUMN_WHITELIST['away_cs'],  label='AWAY clean sheet odds')
        # Over tails (1.5 / 3.5 / 4.5)
        _ = _resolve_odds(df, ODD_COLUMN_WHITELIST.get('over15', []), label='Over 1.5 odds')
        _ = _resolve_odds(df, ODD_COLUMN_WHITELIST.get('over35', []), label='Over 3.5 odds')
        _ = _resolve_odds(df, ODD_COLUMN_WHITELIST.get('over45', []), label='Over 4.5 odds')
    except Exception:
        # Be silent on any errors – this is diagnostics-only
        pass

# --- fallback: infer BTTS/Over/Under probabilities from bookmaker odds ---
def infer_probs_from_odds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer market probabilities from bookmaker odds, with de-vig where possible.

    Sets:
      - btts_confidence, btts_no_confidence
      - over25_confidence, under25_confidence
      - confidence_home, confidence_draw, confidence_away  (FTR three-way)
      - home_goals_pred, away_goals_pred (seeded from pre-match xG when available)

    Tags df.attrs['probs_from_odds'] = True so EV logic avoids double penalisation.
    """
    import numpy as _np
    import pandas as _pd

    def _num(s):
        return _pd.to_numeric(s, errors="coerce")

    def _pick(colnames):
        for c in colnames:
            if c in df.columns:
                return c
        return None
    
    # Guard against post-match leakage
    df = _force_drop_leaky_cols(df)

    # mark that we inferred from odds (affects EV computation)
    df.attrs["probs_from_odds"] = True

    # ---- BTTS YES/NO ---------------------------------------------------------
    yes_col = _pick(ODD_COLUMN_WHITELIST.get("btts_yes", []))
    no_col  = _pick(ODD_COLUMN_WHITELIST.get("btts_no", []))
    if yes_col is not None or no_col is not None:
        q_yes = 1.0 / _num(df[yes_col]) if yes_col is not None else _pd.Series(_np.nan, index=df.index)
        q_no  = 1.0 / _num(df[no_col])  if no_col  is not None else _pd.Series(_np.nan, index=df.index)

        both = ~(q_yes.isna() | q_no.isna())
        p_yes = _pd.Series(_np.nan, index=df.index)
        p_no  = _pd.Series(_np.nan, index=df.index)
        p_yes.loc[both] = (q_yes[both] / (q_yes[both] + q_no[both])).clip(0, 1)
        p_no.loc[both]  = (q_no[both]  / (q_yes[both] + q_no[both])).clip(0, 1)

        # single-side fallbacks (guarded when only one side of odds is present)
        if yes_col is not None:
            p_yes.loc[p_yes.isna() & ~q_yes.isna()] = (
                (1.0 / _num(df[yes_col]))[p_yes.isna() & ~q_yes.isna()]
            ).clip(0, 1)
        if no_col is not None:
            p_no.loc[p_no.isna() & ~q_no.isna()] = (
                (1.0 / _num(df[no_col]))[p_no.isna() & ~q_no.isna()]
            ).clip(0, 1)

        df["btts_confidence"]    = p_yes.fillna(0.5).astype(float)
        df["btts_no_confidence"] = p_no.fillna(1.0 - df["btts_confidence"]).astype(float)

    # ---- OVER/UNDER 2.5 ------------------------------------------------------
    over_col  = _pick(ODD_COLUMN_WHITELIST.get("over25", []))
    under_col = _pick(ODD_COLUMN_WHITELIST.get("under25", []))
    if over_col is not None or under_col is not None:
        q_over  = 1.0 / _num(df[over_col])  if over_col  is not None else _pd.Series(_np.nan, index=df.index)
        q_under = 1.0 / _num(df[under_col]) if under_col is not None else _pd.Series(_np.nan, index=df.index)

        both = ~(q_over.isna() | q_under.isna())
        p_over  = _pd.Series(_np.nan, index=df.index)
        p_under = _pd.Series(_np.nan, index=df.index)
        p_over.loc[both]  = (q_over[both]  / (q_over[both] + q_under[both])).clip(0, 1)
        p_under.loc[both] = (q_under[both] / (q_over[both] + q_under[both])).clip(0, 1)

        # single-side fallbacks (guarded when only one side of odds is present)
        if over_col is not None:
            p_over.loc[p_over.isna() & ~q_over.isna()] = (
                (1.0 / _num(df[over_col]))[p_over.isna() & ~q_over.isna()]
            ).clip(0, 1)
        if under_col is not None:
            p_under.loc[p_under.isna() & ~q_under.isna()] = (
                (1.0 / _num(df[under_col]))[p_under.isna() & ~q_under.isna()]
            ).clip(0, 1)

        df["over25_confidence"]  = p_over.fillna(0.5).astype(float)
        df["under25_confidence"] = p_under.fillna(1.0 - df["over25_confidence"]).astype(float)

    # ---- FTR 1X2 from three-way odds ----------------------------------------
    home_col = _pick(ODD_COLUMN_WHITELIST.get("home_win", []))
    draw_col = _pick(ODD_COLUMN_WHITELIST.get("draw", []))
    away_col = _pick(ODD_COLUMN_WHITELIST.get("away_win", []))
    if home_col and draw_col and away_col:
        qh = 1.0 / _num(df[home_col])
        qd = 1.0 / _num(df[draw_col])
        qa = 1.0 / _num(df[away_col])
        denom = (qh + qd + qa).replace({0.0: _np.nan})
        df["confidence_home"] = (qh / denom).clip(0, 1).fillna(1/3)
        df["confidence_draw"] = (qd / denom).clip(0, 1).fillna(1/3)
        df["confidence_away"] = (qa / denom).clip(0, 1).fillna(1/3)

    # ---- Seed Poisson lambdas from pre-match xG (as defaults) ----------------
    if "home_goals_pred" not in df.columns or df["home_goals_pred"].isna().all() \
       or "away_goals_pred" not in df.columns or df["away_goals_pred"].isna().all():
        df = seed_goal_lambda_from_prematch(df)

    # Fill match_date if blank using date_GMT
    if "match_date" in df.columns and "date_GMT" in df.columns:
        df["match_date"] = df["match_date"].replace({"N/A": _np.nan}).fillna(df["date_GMT"])
    # Estimate and cache book margin from 1X2 for conservative EV adjustments later
    try:
        _ = _estimate_book_margin(df)
    except Exception:
        pass

    return df


# --- FTR implied probabilities + Poisson λ seeders (fallbacks) ---------------

def infer_ftr_from_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Infer FTR probabilities (home/draw/away) from bookmaker odds.
    Normalises implied probabilities to remove overround.
    Creates: confidence_home, confidence_draw, confidence_away, ftr_pred.
    Safe no-op if required odds columns are missing.
    """
    if df is None or df.empty:
        return df
    # Guard against post-match leakage
    df = _force_drop_leaky_cols(df)
    # Ensure numeric-ish coercions first and canonical headers
    df = _coerce_numeric_like(df, odds_whitelist=ODD_COLUMN_WHITELIST)
    df = apply_safe_renames_and_whitelist(df)

    home_col = _resolve_odds(df, ODD_COLUMN_WHITELIST['home_win'], label='FTR HOME odds')
    draw_col = _resolve_odds(df, ODD_COLUMN_WHITELIST['draw'],     label='FTR DRAW odds')
    away_col = _resolve_odds(df, ODD_COLUMN_WHITELIST['away_win'], label='FTR AWAY odds')
    if not all([home_col, draw_col, away_col]):
        return df

    o_home = pd.to_numeric(df[home_col], errors='coerce')
    o_draw = pd.to_numeric(df[draw_col], errors='coerce')
    o_away = pd.to_numeric(df[away_col], errors='coerce')

    p_home = (1.0 / o_home).clip(0, 1)
    p_draw = (1.0 / o_draw).clip(0, 1)
    p_away = (1.0 / o_away).clip(0, 1)

    total = (p_home + p_draw + p_away).replace({0.0: np.nan})
    p_home_n = (p_home / total).fillna(0.0).clip(0, 1)
    p_draw_n = (p_draw / total).fillna(0.0).clip(0, 1)
    p_away_n = (p_away / total).fillna(0.0).clip(0, 1)

    df['confidence_home'] = p_home_n
    df['confidence_draw'] = p_draw_n
    df['confidence_away'] = p_away_n

    # argmax: 0=home,1=draw,2=away (for quick debug/export only)
    df['ftr_pred'] = np.argmax(
        np.vstack([p_home_n.to_numpy(), p_draw_n.to_numpy(), p_away_n.to_numpy()]).T,
        axis=1
    )
    return df


def seed_goal_lambda_from_prematch(df: pd.DataFrame) -> pd.DataFrame:
    """Create home_goals_pred and away_goals_pred if missing.
    Priority: pre-match xG; else split average total goals by FTR strengths.
    """
    if df is None or df.empty:
        return df
    need_home = 'home_goals_pred' not in df.columns
    need_away = 'away_goals_pred' not in df.columns
    if not (need_home or need_away):
        return df

    df = apply_safe_renames_and_whitelist(df)  # ensure xG headers are canonical
    h_xg = df.get('pre_match_xg_home')
    a_xg = df.get('pre_match_xg_away')
    if h_xg is not None and a_xg is not None:
        df['home_goals_pred'] = pd.to_numeric(h_xg, errors='coerce').clip(lower=0).fillna(1.0)
        df['away_goals_pred'] = pd.to_numeric(a_xg, errors='coerce').clip(lower=0).fillna(1.0)
        return df

    # Ensure Series types (df.get may return a scalar if column is missing)
    _tot_raw = df.get('average_goals_per_match_pre_match', None)
    tot = pd.to_numeric(_tot_raw, errors='coerce') if isinstance(_tot_raw, pd.Series) else pd.Series(np.nan, index=df.index)

    _ph_raw = df.get('confidence_home', None)
    ph = pd.to_numeric(_ph_raw, errors='coerce') if isinstance(_ph_raw, pd.Series) else pd.Series(np.nan, index=df.index)

    _pa_raw = df.get('confidence_away', None)
    pa = pd.to_numeric(_pa_raw, errors='coerce') if isinstance(_pa_raw, pd.Series) else pd.Series(np.nan, index=df.index)

    if tot.isna().all():
        tot = pd.Series(2.6, index=df.index)  # broad prior
    if ph.isna().all() or pa.isna().all():
        ph = pd.Series(0.55, index=df.index)
        pa = pd.Series(0.45, index=df.index)

    denom   = (ph + pa).replace({0.0: np.nan})
    share_h = (ph / denom).fillna(0.5)
    share_a = (pa / denom).fillna(0.5)

    df['home_goals_pred'] = (tot * share_h).clip(lower=0.1)
    df['away_goals_pred'] = (tot * share_a).clip(lower=0.1)
    return df
def attach_fts_from_poisson(df: pd.DataFrame) -> pd.DataFrame:
    """Derive Fail-To-Score probabilities from Poisson goal lambdas.
    Requires: home_goals_pred, away_goals_pred.
    Creates (if missing):
      - p_home_fts = P(home scores 0) = exp(-home_goals_pred)
      - p_away_fts = P(away scores 0) = exp(-away_goals_pred)
    Safe no-op if lambdas are missing.
    """
    if df is None or df.empty:
        return df
    if "home_goals_pred" not in df.columns or "away_goals_pred" not in df.columns:
        return df

    if "p_home_fts" not in df.columns:
        try:
            df["p_home_fts"] = np.exp(-pd.to_numeric(df["home_goals_pred"], errors="coerce").clip(lower=0))
        except Exception:
            df["p_home_fts"] = np.nan
    if "p_away_fts" not in df.columns:
        try:
            df["p_away_fts"] = np.exp(-pd.to_numeric(df["away_goals_pred"], errors="coerce").clip(lower=0))
        except Exception:
            df["p_away_fts"] = np.nan
    return df


# --- Ensure realized goal columns exist for downstream consumers (e.g., reports) ---
def ensure_realized_goal_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure realized goal columns exist for downstream consumers (e.g., reports).
    For upcoming fixtures these do not exist; add them as NaN so any accidental
    label derivations won't KeyError.
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    if 'home_team_goal_count' not in df.columns:
        df['home_team_goal_count'] = np.nan
    if 'away_team_goal_count' not in df.columns:
        df['away_team_goal_count'] = np.nan
    return df

# --- Compatibility helper for ROI odds/prob columns (idempotent) ---
if "_ensure_roi_inputs_inplace" not in globals():
    def _ensure_roi_inputs_inplace(df):
        """
        Ensure ROI-required odds columns exist in *df*:
          • odds_ft_over25  (aliases: odds_ft_over2_5, odds_over_2_5, odds_over25, odds_o25, odds_ft_over_25)
          • odds_ft_btts    (aliases: odds_btts_yes, odds_btts, btts_odds_yes, odds_yes_btts, odds_btts_y, odds_btts_gg)
          • odds_ft_over25_btts (combo; optional)
        If none are present, create NaN columns so ROI code can skip rows without KeyError.
        """
        import pandas as pd, numpy as np
        if not isinstance(df, pd.DataFrame):
            return df

        alias_map = {
            "odds_ft_over25": ["odds_ft_over2_5","odds_over_2_5","odds_over25","odds_o25","over25_odds","odds_ft_over_25"],
            "odds_ft_btts":   ["odds_btts_yes","odds_btts","btts_odds_yes","odds_yes_btts","odds_btts_y","odds_btts_gg"],
            "odds_ft_over25_btts": ["odds_over25_btts","odds_ft_over25_btts","odds_o25_btts","odds_btts_over25","odds_combo_over25_btts"],
        }

        for target, alts in alias_map.items():
            if target not in df.columns:
                for a in alts:
                    if a in df.columns:
                        df[target] = pd.to_numeric(df[a], errors="coerce")
                        break
                else:
                    df[target] = np.nan
        return df

# --- Report-probability helper (idempotent) ---
if "_ensure_report_probs_inplace" not in globals():
    def _ensure_report_probs_inplace(df):
        import pandas as pd
        if not isinstance(df, pd.DataFrame):
            return df

        def _as_series(val):
            return val if isinstance(val, pd.Series) else pd.Series([val]*len(df), index=df.index)

        if "prob_over25" not in df.columns:
            src = df.get("adjusted_over25_confidence", df.get("over25_confidence", 0.5))
            s = pd.to_numeric(_as_series(src), errors="coerce").fillna(0.5).clip(0, 1)
            df["prob_over25"] = s
        if "prob_btts" not in df.columns:
            src = df.get("adjusted_btts_confidence", df.get("btts_confidence", 0.5))
            s = pd.to_numeric(_as_series(src), errors="coerce").fillna(0.5).clip(0, 1)
            df["prob_btts"] = s
        if "Over25" not in df.columns:
            df["Over25"] = (pd.to_numeric(df["prob_over25"], errors="coerce") > 0.5).astype(int)
        if "BTTS" not in df.columns:
            df["BTTS"] = (pd.to_numeric(df["prob_btts"], errors="coerce") > 0.5).astype(int)
        return df

# --- Guarded ROI printer used by overlay & pipeline ---
if "_print_roi_snapshot" not in globals():
    def _print_roi_snapshot(df, stake: float = 10.0):
        import numpy as np
        import pandas as pd
        if not isinstance(df, pd.DataFrame) or df.empty:
            print("---- ROI snapshot (£10 stakes) ----------------------------")
            print("Over 2.5 : {'bets': 0, 'profit': 0.0, 'roi': 0.0}")
            print("BTTS     : {'bets': 0, 'profit': 0.0, 'roi': 0.0}")
            print("Combo    : {'bets': 0, 'profit': 0.0, 'roi': 0.0}")
            print("----------------------------------------------------------")
            return {"over25": {"bets": 0, "profit": 0.0, "roi": 0.0},
                    "btts":   {"bets": 0, "profit": 0.0, "roi": 0.0},
                    "combo":  {"bets": 0, "profit": 0.0, "roi": 0.0}}

        # Ensure the few columns ROI relies on exist (safe no-ops)
        try: _ensure_report_probs_inplace(df)
        except Exception: pass
        try: _ensure_roi_inputs_inplace(df)
        except Exception: pass

        # Realized label mask
        yh = pd.to_numeric(df.get("home_team_goal_count"), errors="coerce")
        ya = pd.to_numeric(df.get("away_team_goal_count"), errors="coerce")
        realized = yh.notna() & ya.notna()

        # ---- Over 2.5 ROI ----
        odds_o25 = pd.to_numeric(df.get("odds_ft_over25"), errors="coerce")
        valid_o25 = realized & odds_o25.notna() & np.isfinite(odds_o25) & (odds_o25 >= 1.01)
        y_over = (yh.add(ya) > 2) & valid_o25
        bets_over = int(valid_o25.sum())
        profit_over = np.where(y_over, (odds_o25 - 1.0) * stake, -stake)
        profit_over = float(np.nan_to_num(np.where(valid_o25, profit_over, 0.0)).sum())
        roi_over = (profit_over / (bets_over * stake)) if bets_over else 0.0

        # ---- BTTS ROI ----
        odds_btts = pd.to_numeric(df.get("odds_ft_btts"), errors="coerce")
        valid_btts = realized & odds_btts.notna() & np.isfinite(odds_btts) & (odds_btts >= 1.01)
        y_btts = ((yh > 0) & (ya > 0)) & valid_btts
        bets_btts = int(valid_btts.sum())
        profit_btts = np.where(y_btts, (odds_btts - 1.0) * stake, -stake)
        profit_btts = float(np.nan_to_num(np.where(valid_btts, profit_btts, 0.0)).sum())
        roi_btts = (profit_btts / (bets_btts * stake)) if bets_btts else 0.0

        # ---- Combo (optional) ----
        if "odds_ft_over25_btts" in df.columns:
            odds_combo = pd.to_numeric(df.get("odds_ft_over25_btts"), errors="coerce")
            valid_combo = realized & odds_combo.notna() & np.isfinite(odds_combo) & (odds_combo >= 1.01)
            y_combo = (y_over & y_btts) & valid_combo
            bets_combo = int(valid_combo.sum())
            profit_combo = np.where(y_combo, (odds_combo - 1.0) * stake, -stake)
            profit_combo = float(np.nan_to_num(np.where(valid_combo, profit_combo, 0.0)).sum())
            roi_combo = (profit_combo / (bets_combo * stake)) if bets_combo else 0.0
        else:
            bets_combo = 0
            profit_combo = 0.0
            roi_combo = 0.0

        print("---- ROI snapshot (£10 stakes) ----------------------------")
        print(f"Over 2.5 : {{'bets': {bets_over}, 'profit': {profit_over:.1f}, 'roi': {roi_over:.1%}}}")
        print(f"BTTS     : {{'bets': {bets_btts}, 'profit': {profit_btts:.1f}, 'roi': {roi_btts:.1%}}}")
        print(f"Combo    : {{'bets': {bets_combo}, 'profit': {profit_combo:.1f}, 'roi': {roi_combo:.1%}}}")
        print("----------------------------------------------------------")

        # Memory hygiene
        try:
            del yh, ya, realized, odds_o25, valid_o25, y_over, odds_btts, valid_btts, y_btts
        except Exception:
            pass
        try:
            del odds_combo, valid_combo, y_combo  # may not exist
        except Exception:
            pass
        try:
            import gc; gc.collect()
        except Exception:
            pass

        return {
            "over25": {"bets": bets_over, "profit": profit_over, "roi": roi_over},
            "btts":   {"bets": bets_btts, "profit": profit_btts, "roi": roi_btts},
            "combo":  {"bets": bets_combo, "profit": profit_combo, "roi": roi_combo},
        }

# --- Deployable summary helper (rows, draw share, gate) ----------------------
def print_deployable_summary(df) -> None:
    """Print a one‑liner with rows, draw share, and gate used (FTR_PICK_GATE)."""
    try:
        import os
        gate = float(os.getenv("FTR_PICK_GATE", "0.68"))
    except Exception:
        gate = 0.68
    try:
        import pandas as _pd
        if isinstance(df, _pd.DataFrame) and not df.empty and ("selected_outcome" in df.columns):
            ds = float(df["selected_outcome"].astype(str).str.lower().eq("draw").mean())
        else:
            ds = 0.0
        print(f"🎯 deployable summary [overlay]: rows={len(df)} | draw_share={ds:.3f} | gate={gate:.2f}")
    except Exception:
        pass
        
def ensure_minimal_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure we have FTR probs and goal lambdas so WTN/Clean-Sheet/Poisson
    helpers can operate even without trained side models."""
    df = infer_ftr_from_odds(df)
    # Prefer linear λ-from-FTR; fall back to prematch xG splitter
    df = _estimate_lambdas_from_ftr(df)
    df = seed_goal_lambda_from_prematch(df)
    df = attach_fts_from_poisson(df)
    return df

# --- Tiny linear λ-from-FTR helper ------------------------------------------
def _estimate_lambdas_from_ftr(df: pd.DataFrame, a: float = 1.6, b: float = 0.2) -> pd.DataFrame:
    """Lightweight mapping from FTR implied probs to goal lambdas.
    λ_home = a * p_home + b; λ_away = a * p_away + b.
    If an average total-goals prior is available, rescale so λ_h+λ_a matches it.
    Returns df with home_goals_pred/away_goals_pred filled where missing."""
    if df is None or df.empty:
        return df
    if 'confidence_home' not in df.columns or 'confidence_away' not in df.columns:
        return df
    if 'home_goals_pred' in df.columns and 'away_goals_pred' in df.columns \
       and bool(df['home_goals_pred'].notna().any()) and bool(df['away_goals_pred'].notna().any()):
        return df

    ph = pd.to_numeric(df.get('confidence_home'), errors='coerce').clip(0, 1).fillna(1/3)
    pa = pd.to_numeric(df.get('confidence_away'), errors='coerce').clip(0, 1).fillna(1/3)
    lam_h = (a * ph + b).clip(lower=0.05)
    lam_a = (a * pa + b).clip(lower=0.05)

    # Optional rescale to match a known average total goals if present
    tot_prior = pd.to_numeric(df.get('average_goals_per_match_pre_match'), errors='coerce')
    have_tot  = tot_prior.notna()
    sum_la    = (lam_h + lam_a).replace({0.0: np.nan})
    scale     = (tot_prior / sum_la).where(have_tot & sum_la.notna(), 1.0)
    lam_h = (lam_h * scale).clip(lower=0.05)
    lam_a = (lam_a * scale).clip(lower=0.05)

    df['home_goals_pred'] = df.get('home_goals_pred', lam_h).where(df.get('home_goals_pred').notna() if 'home_goals_pred' in df.columns else False, lam_h)
    df['away_goals_pred'] = df.get('away_goals_pred', lam_a).where(df.get('away_goals_pred').notna() if 'away_goals_pred' in df.columns else False, lam_a)
    return df
def _normalise_prob_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Scale any percentage/prob columns that arrive as 0..100 to 0..1.
    Uses name heuristics and value ranges.
    """
    df = df.copy()
    for c in df.columns:
        lc = c.lower()
        if _percent_like_re.search(lc) or any(tok in lc for tok in ("prob", "confidence")):
            try:
                s = pd.to_numeric(df[c], errors='coerce')
                if bool(s.notna().any()) and float(s.max()) > 1.5 and float(s.max()) <= 100:
                    df[c] = s / 100.0
            except Exception:
                continue
    return df

# --------------------------------------------------------------
# Exports
# --------------------------------------------------------------
__all__ = [
    "prepare_league_for_inference",
    "apply_draw_threshold_flag",
    "mark_topk_draws",
    "generate_btts_and_over_preds",
    "adjust_with_volatility_modifiers",
    "log_prediction_changes",
    "attach_win_to_nil_proxy",
    "simulate_accumulator_roi",
    "generate_accumulator_recommendations",
    "generate_correct_score_candidates",
    "validate_market_coherence",
    "simulate_walk_forward_pnl",
    "log_resolved_odds_columns",
    "infer_probs_from_odds",
    "apply_safe_renames_and_whitelist",
    "infer_ftr_from_odds",
    "seed_goal_lambda_from_prematch",
    "ensure_minimal_signals",
    "attach_fts_from_poisson",
    "ensure_realized_goal_placeholders",
    "overlay_config",
    "load_market_thresholds",
    "apply_market_thresholds",
    "load_market_thresholds_for_league", 
    "score_trained_markets_if_available",
    "enrich_with_models_or_odds",
    "attach_specialist_probs_for_inference",
    "_resolve_draw_temp",
    "print_deployable_summary",
    "apply_ev_filters_for_o15_o35_o45",
    "prepare_deployable_overlay",
]
# --- Inference-time specialist probs → 'oof_prob_*' columns (no OOF splits) ---
if "attach_specialist_probs_for_inference" not in globals():
    def attach_specialist_probs_for_inference(df: pd.DataFrame,
                                              league_name: str,
                                              markets: tuple[str, ...] = ("over25","btts","home_over15tg","away_over15tg","home_fts","away_fts")) -> pd.DataFrame:
        """
        For upcoming/inference data, attach specialist probabilities into
        columns named like the training OOF features (e.g., 'oof_prob_over25').
        Uses fitted models (no OOF splitting).
        Safe no-op if a model or features are missing.
        """
        import numpy as _np, pandas as _pd
        if df is None or len(df) == 0:
            return df
        out = df.copy()

        for m in markets:
            try:
                mdl = _load_market_model(league_name, m)
            except Exception:
                mdl = None
            if mdl is None:
                continue
            try:
                X = _strict_align(out, mdl).replace([_np.inf, -_np.inf], _np.nan)
                # Only fill numeric columns with sentinel to avoid datetime issues
                try:
                    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
                    X = safe_fill(X, num_cols, SENTINEL)
                except Exception:
                    pass
                proba = mdl.predict_proba(X)
                # Binary models: take the positive class column
                p = proba[:, 1] if getattr(proba, "ndim", 1) == 2 and proba.shape[1] >= 2 else _np.asarray(proba, dtype=float).ravel()
                col = f"oof_prob_{m}"
                if col not in out.columns or out[col].isna().all():
                    out[col] = _np.clip(p, 0.0, 1.0)
            except Exception as _e:
                try:
                    _log("WARN", f"⚠️ Could not score {m} specialist for inference: {_e}")
                except Exception:
                    pass
                continue
        return out

# --------------------------------------------------------------
# λ-shrink constants and historical draw rates
# --------------------------------------------------------------
from constants import N0_SHRINK, N0_SHRINK_PER_LEAGUE

# Fallback dictionary – you can overwrite / inject a richer mapping at runtime
HISTORICAL_DRAW_RATE = {
    "Champions League": 0.24,
    "England Premier League": 0.23,
    "Europa Conference": 0.27,
    "Europa League": 0.21,
    "Germany Bundesliga": 0.25,
    "Italy Serie A": 0.28,
    "Portugal Liga": 0.23,
    "Spain La Liga": 0.26,
    "USA MLS": 0.25,
}

# ------------------------------------------------------------------
# Strict alignment: ensure columns are in *exact* training order
# ------------------------------------------------------------------
def _strict_align(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Return a dataframe whose columns exactly match
    `model.feature_names_in_` (order & presence).
    Any missing predictor is filled with zeros.
    """
    cols = list(getattr(model, "feature_names_in_", []))
    if not cols:   # fallback if the attribute is missing
        numeric = (
            df.select_dtypes(include=['number'])
              .replace([np.inf, -np.inf], np.nan)
        )
        return safe_fill(numeric, numeric.columns.tolist(), SENTINEL)
    aligned = df.copy()
    for c in cols:
        if c not in aligned.columns:
            aligned[c] = 0.0
    return aligned.loc[:, cols].astype(float)

# ------------------------------------------------------------------
# λ-shrink helper – draw‑probability prior‑shrink toward long‑run rate
# ------------------------------------------------------------------
def _apply_lambda_shrink(p_draw: np.ndarray,
                         *,
                         league: str,
                         matches_seen: int,
                         n0: int = N0_SHRINK) -> np.ndarray:
    """
    Empirical‑Bayes shrinkage of draw probability toward the league’s
    historical draw rate.

        p* = (N / (N + n0)) · p  +  (n0 / (N + n0)) · π

    Parameters
    ----------
    p_draw : 1‑D numpy array of raw draw probabilities.
    league : League name string.
    matches_seen : How many fixtures from this league are in the sample.
    n0 : Prior‑strength in ‘matches’ (defaults to constants.N0_SHRINK).

    Returns
    -------
    np.ndarray – same shape as *p_draw* with stabilised probabilities.
    """
    # Override global n0 with any league‑specific setting
    n0 = N0_SHRINK_PER_LEAGUE.get(league, n0)
    if matches_seen >= n0:      # enough data → no shrink
        return p_draw
    hist = constants.LOCKED_THRESHOLDS.get(league, {}).get("draw", 0.28)
    lam  = n0 / (n0 + max(matches_seen, 1))          # avoid divide‑by‑zero
    return (1.0 - lam) * p_draw + lam * hist

# ------------------------------------------------------------------
# Draw-threshold + top-k helpers (per-league, JSON-backed)
# ------------------------------------------------------------------
import json as _json

# Lazy hook to pipeline's loader if present
try:
    _load_draw_thr_from_pipeline = getattr(_bfp, "load_league_draw_threshold")
except Exception:
    _load_draw_thr_from_pipeline = None

try:
    _load_best_topk_percent_from_pipeline = getattr(_bfp, "load_best_topk_percent")
except Exception:
    _load_best_topk_percent_from_pipeline = None


def _load_draw_threshold_local(league_name: str, default_thr: float = 0.5):
    """
    Read <MODEL_DIR>/<League>_draw_threshold.json and return (thr, mode).
    Falls back to `default_thr` and mode 'default' if file is missing.
    """
    league_tag = str(league_name).replace(" ", "_")
    path = os.path.join(MODEL_DIR, f"{league_tag}_draw_threshold.json")
    try:
        with open(path, "r") as fh:
            data = _json.load(fh)
        thr  = float(data.get("threshold", default_thr))
        mode = str(data.get("mode", "youden"))
        return thr, mode
    except Exception:
        return float(default_thr), "default"


def _load_draw_threshold(league_name: str, default_thr: float = 0.5):
    """Unified wrapper that prefers the pipeline helper when available."""
    if callable(_load_draw_thr_from_pipeline):
        try:
            return _load_draw_thr_from_pipeline(league_name, default_thr)
        except Exception:
            pass
    return _load_draw_threshold_local(league_name, default_thr)


def _load_best_topk_percent_local(league_name: str, default_pct: float = 0.10) -> float:
    """Return the historically best top-k percent from the league JSON if present."""
    league_tag = str(league_name).replace(" ", "_")
    path = os.path.join(MODEL_DIR, f"{league_tag}_draw_threshold.json")
    try:
        with open(path, "r") as fh:
            data = _json.load(fh)
        best = data.get("best_k_for_precision")
        if isinstance(best, dict) and "k_pct" in best:
            return float(best["k_pct"])
        # Back-compat: some versions stored a list under `precision_at_k`
        pakk = data.get("precision_at_k")
        if isinstance(pakk, list) and pakk:
            # pick k with highest precision
            best_row = max(
                (r for r in pakk if isinstance(r, dict) and "precision" in r and "k_pct" in r),
                key=lambda r: r.get("precision", 0.0),
                default=None
            )
            if best_row:
                return float(best_row.get("k_pct", default_pct))
    except Exception:
        pass
    return float(default_pct)


def _load_best_topk_percent(league_name: str, default_pct: float = 0.10) -> float:
    if callable(_load_best_topk_percent_from_pipeline):
        try:
            return float(_load_best_topk_percent_from_pipeline(league_name, default_pct))
        except Exception:
            pass
    return _load_best_topk_percent_local(league_name, default_pct)

# ─────────────────────────────────────────────────────────────────────────────
# Per‑market thresholds loader + trained‑model scoring helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_market_thresholds(league_name: str, model_dir: str = MODEL_DIR) -> dict:
    """Read ModelStore/<League>_market_thresholds.json if present.
    Returns a dict {market: threshold} with float values; empty dict if missing."""
    league_tag = str(league_name).replace(" ", "_")
    path = os.path.join(model_dir, f"{league_tag}_market_thresholds.json")
    try:
        with open(path, "r") as fh:
            data = _json.load(fh)
        out: dict[str, float] = {}
        # Schema 1: nested payload under "markets": {mkt: {threshold: x, ...}}
        if isinstance(data, dict) and isinstance(data.get("markets"), dict):
            for mkt, info in data["markets"].items():
                try:
                    thr = info.get("threshold") if isinstance(info, dict) else None
                    if isinstance(thr, (int, float)):
                        out[str(mkt)] = float(thr)
                except Exception:
                    continue
        # Schema 2: flat mapping {market: threshold}
        elif isinstance(data, dict):
            for k, v in data.items():
                try:
                    out[str(k)] = float(v)
                except Exception:
                    continue
        return out
    except Exception:
        return {}

def apply_market_thresholds(df: pd.DataFrame, league_name: str, markets: list[str] | None = None) -> pd.DataFrame:
    """Apply per‑league thresholds to df by setting df.attrs['thr_<market>'] and, when
    a matching probability column exists, compute the binary *pred* column via that threshold.
    Also updates `config["thr_<market>_default"]` so downstream components pick up the gates.
    Safe no‑op if the JSON is not present."""
    thr_map = load_market_thresholds(league_name)
    if not thr_map:
        return df
    df = df.copy()

    # Map each market to (probability_column, pred_column) used by the overlay
    mapping = {
        'btts':      ('btts_confidence',   'btts_pred'),
        'over25':    ('over25_confidence', 'over25_pred'),
        'under25':   ('under25_confidence','under25_pred'),
        'btts_no':   ('btts_no_confidence','btts_no_pred'),
        'wtn':       ('selected_wtn_confidence','wtn_pred'),      # when present
        'clean_sheet': ('selected_cs_confidence','clean_sheet_pred'),
        # FTR is multiclass → no single thresholded pred here
        'ftr':       (None, None),
    }
    if markets is None:
        markets = list(thr_map.keys())

    for m in markets:
        if m not in thr_map:
            continue
        thr = float(thr_map[m])
        # Persist to attrs + config defaults
        df.attrs[f"thr_{m}"] = thr
        try:
            config[f"thr_{m}_default"] = thr
        except Exception:
            pass
        # If a probability column exists, threshold to a *_pred column
        prob_col, pred_col = mapping.get(m, (None, None))
        if prob_col and pred_col and prob_col in df.columns:
            try:
                df[pred_col] = (pd.to_numeric(df[prob_col], errors='coerce') >= thr).astype(int)
            except Exception:
                pass
    return df

def _find_model_path(league_name: str, market: str, model_dir: str = MODEL_DIR) -> str | None:
    """Best‑effort search for a saved model file.
    Tries common patterns under ModelStore/ and ModelStore/<LeagueTag>/ and falls back
    to a lightweight substring scan."""
    league_tag = str(league_name).replace(" ", "_")
    market_tag = str(market).lower()
    # Common stems and extensions (require `_model` to avoid picking evaluation artifacts)
    stems = [
        f"{league_tag}_{market_tag}_model",
        f"{market_tag}_model",
    ]
    exts = [".pkl", ".joblib"]
    roots = [os.path.join(model_dir, league_tag), model_dir]
    for root in roots:
        for st in stems:
            for ex in exts:
                p = os.path.join(root, st + ex)
                if os.path.exists(p):
                    return p
    # Last-resort scan: stricter filter to avoid picking eval artifacts (y_test, metrics, etc.)
    try:
        tok_league = league_tag.lower()
        tok_market = market_tag.replace("over25", "over").lower()
        banned = ('y_test','ytrain','x_test','xtrain','pred','proba','prob','metric','metrics',
                'threshold','roc','pr','summary','calib','calibrated')
        for r, _d, files in os.walk(model_dir):
            for fn in files:
                low = fn.lower()
                if not (low.endswith('.pkl') or low.endswith('.joblib')):
                    continue
                if any(b in low for b in banned):
                    continue
                # require “model” token in filename to reduce false positives
                if "model" not in low:
                    continue
                if tok_league in low and (
                    tok_market in low or (market_tag == 'over25' and ('over' in low and '25' in low))
                ):
                    return os.path.join(r, fn)
    except Exception:
        pass
    return None

def _load_market_model(league_name: str, market: str, model_dir: str = MODEL_DIR):
    """Load a trained model if present; returns the estimator or None."""
    path = _find_model_path(league_name, market, model_dir)
    if not path:
        return None
    try:
        try:
            import joblib
            model = joblib.load(path)
        except Exception:
            import pickle
            with open(path, 'rb') as fh:
                model = pickle.load(fh)
        # Require predict_proba to be present; otherwise treat as not-a-model (e.g., y_test series)
        if not hasattr(model, 'predict_proba'):
            _log("WARN", f"⚠️ Loaded artifact lacks predict_proba; ignoring: {os.path.basename(path)}")
            return None
        _log("INFO", f"🧩 Loaded {market} model for {league_name}: {os.path.relpath(path, MODEL_DIR)}")
        return model
    except Exception as e:
        _log("WARN", f"⚠️ Could not load {market} model for {league_name}: {e}")
        return None

def score_trained_markets_if_available(df: pd.DataFrame,
                                       league_name: str,
                                       *,
                                       markets: tuple[str, ...] = ("ftr", "btts", "over25")) -> pd.DataFrame:
    """Attach trained model probabilities when available. Falls back to existing
    odds‑based inference for any missing model. Returns a new dataframe."""
    if df is None or df.empty:
        return df
    out = df.copy()

    # Prefer trained FTR (multiclass) if present
    if "ftr" in markets:
        mdl_ftr = _load_market_model(league_name, "ftr")
        if mdl_ftr is not None:
            try:
                X = _strict_align(out, mdl_ftr).replace([np.inf, -np.inf], np.nan)
                X = safe_fill(X, X.columns.tolist(), SENTINEL)
                proba = mdl_ftr.predict_proba(X)
                classes = list(getattr(mdl_ftr, 'classes_', [0, 1, 2]))
                # Map to home/draw/away indices as best we can
                idx_home = idx_draw = idx_away = None
                for i, c in enumerate(classes):
                    s = str(c).lower()
                    if s in ("home", "h", "1", "home_win"): idx_home = i
                    elif s in ("draw", "d", "x", "0"):      idx_draw = i
                    elif s in ("away", "a", "2", "away_win"): idx_away = i
                if idx_home is None or idx_draw is None or idx_away is None:
                    idx_home, idx_draw, idx_away = 0, 1, 2
                out["confidence_home"] = np.clip(proba[:, idx_home], 0, 1)
                out["confidence_draw"] = np.clip(proba[:, idx_draw], 0, 1)
                out["confidence_away"] = np.clip(proba[:, idx_away], 0, 1)
                out["ftr_pred"] = np.argmax(proba, axis=1)
            except Exception as e:
                _log("WARN", f"⚠️ FTR model present but scoring failed: {e}")
        else:
            # keep prior behaviour
            out = infer_ftr_from_odds(out)

    # BTTS & Over 2.5 as a pair: require both to be present for model path
    mdl_btts = _load_market_model(league_name, "btts") if "btts" in markets else None
    mdl_over = _load_market_model(league_name, "over25") if "over25" in markets else None
    if mdl_btts is not None and mdl_over is not None:
        try:
            out = generate_btts_and_over_preds(out, mdl_btts, mdl_over)
        except Exception as e:
            _log("WARN", f"⚠️ BTTS/Over models present but scoring failed: {e}")
            out = ensure_minimal_signals(out)
    else:
        out = ensure_minimal_signals(out)

    # Ensure downstream consumers won't crash if they expect realized-goal columns
    out = ensure_realized_goal_placeholders(out)

    # Also attach inference-time specialist probabilities as 'oof_prob_*' cols
    try:
        out = attach_specialist_probs_for_inference(out, league_name)
    except Exception:
        pass
    return out

def enrich_with_models_or_odds(df: pd.DataFrame, league_name: str,
                               *, markets: tuple[str, ...] = ("ftr","btts","over25")) -> pd.DataFrame:
    """One‑shot: prefer trained models when available, then apply any learned
    per‑market thresholds from ModelStore. This is the call you want just before
    generating picks/accas."""
    out = score_trained_markets_if_available(df, league_name, markets=markets)
    # --- Fill required market probabilities without overwriting model outputs ---
    have_btts = ("btts_confidence" in out.columns)
    have_over = ("over25_confidence" in out.columns)

    # If main probs are present (usually from trained models), derive complements
    if have_btts and "btts_no_confidence" not in out.columns:
        src = (
            out["adjusted_btts_confidence"]
            if "adjusted_btts_confidence" in out.columns
            else out["btts_confidence"]
        )
        out["btts_no_confidence"] = (1.0 - pd.to_numeric(src, errors="coerce")).clip(0, 1)

    if have_over and "under25_confidence" not in out.columns:
        src = (
            out["adjusted_over25_confidence"]
            if "adjusted_over25_confidence" in out.columns
            else out["over25_confidence"]
        )
        out["under25_confidence"] = (1.0 - pd.to_numeric(src, errors="coerce")).clip(0, 1)

    # If either main probability is still missing, infer all from odds
    if (not have_btts) or (not have_over):
        try:
            out = infer_probs_from_odds(out)
        except Exception:
            pass

    # NEW: pre-load per-league market thresholds into attrs + config defaults
    try:
        _attrs_map = load_market_thresholds_for_league(league_name)
        if _attrs_map:
            out.attrs.update(_attrs_map)
            # keep config defaults in sync for helpers that read config
            overlay_config["thr_btts_default"]       = out.attrs.get("thr_btts",       overlay_config.get("thr_btts_default", 0.5))
            overlay_config["thr_over25_default"]     = out.attrs.get("thr_over25",     overlay_config.get("thr_over25_default", 0.5))
            overlay_config["thr_under25_default"]    = out.attrs.get("thr_under25",    overlay_config.get("thr_under25_default", 0.6))
            overlay_config["thr_btts_no_default"]    = out.attrs.get("thr_btts_no",    overlay_config.get("thr_btts_no_default", 0.6))
            overlay_config["thr_clean_sheet_default"] = out.attrs.get("thr_clean_sheet", overlay_config.get("thr_clean_sheet_default", 0.25))
            overlay_config["thr_wtn_default"]        = out.attrs.get("thr_wtn",        overlay_config.get("thr_wtn_default", 0.25))
    except Exception:
        pass

    out = apply_market_thresholds(out, league_name)
    # Ensure downstream consumers won't crash if they expect realized-goal columns
    out = ensure_realized_goal_placeholders(out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Helper to wire up baseline_blend + ensure press intensity + print best k%
# ─────────────────────────────────────────────────────────────────────────────

def prepare_league_for_inference(league_name: str,
                                 matches_root: str = "Matches",
                                 *,
                                 default_alpha: float = 0.60,
                                 overwrite_baseline: bool = False,
                                 overwrite_intensity: bool = False) -> tuple[float, float]:
    """
    Read per-league training JSON to get `baseline_blend`, rebuild press baseline/intensity
    for this league (cached), and load the chosen top-k% used historically.
    Returns (alpha, best_k_pct) and sets globals BEST_K_PCT / BASELINE_ALPHA_USED.
    """
    global BEST_K_PCT, BASELINE_ALPHA_USED

    league_tag = str(league_name).replace(" ", "_")
    thr_json = os.path.join(MODEL_DIR, f"{league_tag}_draw_threshold.json")
    alpha = float(default_alpha)
    try:
        with open(thr_json, "r") as fh:
            data = _json.load(fh)
            alpha = float(data.get("baseline_blend", alpha))
    except Exception:
        # keep default
        pass

    match_dir = os.path.join(matches_root, league_name)
    try:
        ensure_press_intensity_on_disk(
            match_dir,
            force=False,
            baseline_blend=alpha,
            use_cache=True,
            overwrite_baseline=overwrite_baseline,
            overwrite_intensity=overwrite_intensity,
        )
    except Exception as e:
        print(f"⚠️ ensure_press_intensity_on_disk failed for {match_dir}: {e}")

    # Load best k% from JSON (fall back to 10%)
    try:
        k_pct = _load_best_topk_percent(league_name, default_pct=0.10)
    except Exception:
        k_pct = 0.10

    # Pre‑load per‑market thresholds for this league (if present) and
    # store as runtime defaults so downstream code picks them up.
    try:
        _thr_map = load_market_thresholds(league_name)
        if _thr_map:
            globals().setdefault("PER_MARKET_THRESHOLDS", {})[league_name] = dict(_thr_map)
            for _mk, _thr in _thr_map.items():
                try:
                    config[f"thr_{_mk}_default"] = float(_thr)
                except Exception:
                    pass
    except Exception:
        pass

    # Set module-level globals for downstream consumers/exports
    BASELINE_ALPHA_USED = float(alpha)
    BEST_K_PCT = float(k_pct)

    print(f"ℹ️ Using baseline_blend α={alpha:.2f} for inference")
    print(f"ℹ️ Using top-k draw percent k={k_pct:.2%} (historical best) for flags/exports")
    return float(alpha), float(k_pct)


def apply_draw_threshold_flag(df: pd.DataFrame,
                              league_name: str,
                              *,
                              prob_col: str = "p_draw",
                              out_col: str = "draw_flag",
                              default_gate: float | None = None) -> pd.DataFrame:
    """
    Apply the per-league calibrated draw threshold to create a boolean flag.
    Stores `thr_draw` and `mode_draw` in df.attrs.
    """
    if prob_col not in df.columns:
        # nothing to do
        return df

    # default gate from constants if available
    if default_gate is None:
        default_gate = constants.LOCKED_THRESHOLDS.get(league_name, {}).get("draw", 0.50)

    thr, mode = _load_draw_threshold(league_name, default_thr=float(default_gate))
    df.loc[:, out_col] = (df[prob_col].astype(float) >= float(thr)).astype(int)
    df.attrs["thr_draw"], df.attrs["mode_draw"] = float(thr), str(mode)
    return df


def mark_topk_draws(df: pd.DataFrame,
                    league_name: str,
                    *,
                    prob_col: str = "p_draw",
                    out_col: str = "draw_flag_topk",
                    k_pct: float | None = None) -> pd.DataFrame:
    """
    Mark the top-k% fixtures by P(draw) with a separate flag.
    If k_pct is None, load the historically best k% for the league.
    """
    if prob_col not in df.columns or df.empty:
        return df

    if k_pct is None:
        k_pct = _load_best_topk_percent(league_name, default_pct=0.10)
    k = max(1, int(round(len(df) * float(k_pct))))

    # argsort descending by prob
    order = np.argsort(-df[prob_col].to_numpy(dtype=float))
    idx_top = set(df.index.take(order[:k]))

    flag = df.index.to_series().apply(lambda i: 1 if i in idx_top else 0)
    df.loc[:, out_col] = flag.astype(int)
    # Provide a friendly alias for downstream consumers
    try:
        df.loc[:, "is_topk_best"] = df[out_col].astype(int)
    except Exception:
        # Defensive fallback in case out_col is missing (should not happen)
        df.loc[:, "is_topk_best"] = 0
    df.attrs["topk_pct_draw"] = float(k_pct)
    return df

def generate_btts_and_over_preds(df: pd.DataFrame,
                                 btts_model,
                                 over_model,
                                 *,
                                 gate_scale: float = 0.60) -> pd.DataFrame:
    """
    Adds / updates in‑place:
        • btts_confidence,   over25_confidence   – calibrated probabilities
        • btts_pred,         over25_pred         – binary labels

    The binary labels respect each model’s calibrated probability
    cut‑off stored as `.best_threshold_` (falls back to 0.50).
    gate_scale defines the p_draw threshold (default 0.60) used for the draw gate later in the function.
    """
    # --- lazy import avoids circular dependency with the main pipeline ----
    global POST_MATCH_TAGS, _align_dataframe_to_model, safe_fill, SENTINEL
    if "POST_MATCH_TAGS" not in globals():
        from importlib import import_module
        try:
            _pipeline = import_module("_baseline_ftr_pipeline")
        except Exception:
            try:
                _pipeline = import_module("baseline_ftr_pipeline")
            except Exception:
                _pipeline = import_module("00_baseline_ftr_pipeline")
        POST_MATCH_TAGS = _pipeline.POST_MATCH_TAGS
        _align_dataframe_to_model = _pipeline._align_dataframe_to_model
        safe_fill = _pipeline.safe_fill
        SENTINEL  = getattr(_pipeline, "SENTINEL", -999)
    # --- guard against duplicate column names -------------------
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # ── Drop any post‑match leakage columns before inference ──
    df = df.drop(
        columns=[c for c in df.columns if any(tag in c for tag in POST_MATCH_TAGS)],
        errors="ignore"
    )
    # Always drop leaky columns and coerce numeric-likes (odds/percent/prob)
    df = _force_drop_leaky_cols(df)
    df = _coerce_numeric_like(df, odds_whitelist=ODD_COLUMN_WHITELIST)
    df = _normalise_prob_columns(df)
    # Canonicalise messy CSV headers to training schema
    df = apply_safe_renames_and_whitelist(df)
    # Log which odds columns will be used (diagnostics)
    try:
        log_resolved_odds_columns(df)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 🆕 Poisson‑derived probabilities from the goal λ̂ predictions
    # ------------------------------------------------------------------
    try:
        from scipy.stats import poisson
    except ImportError:     # defensive guard – continue without Poisson boost
        poisson = None

    if poisson is not None and {"home_goals_pred", "away_goals_pred"}.issubset(df.columns):
        λ_home = df["home_goals_pred"].clip(lower=0, upper=5)
        λ_away = df["away_goals_pred"].clip(lower=0, upper=5)
        λ_tot  = λ_home + λ_away

        df["poisson_over25_prob"] = 1.0 - poisson.cdf(2, λ_tot)
        df["poisson_btts_prob"]   = (1.0 - np.exp(-λ_home)) * (1.0 - np.exp(-λ_away))

        # make sure downstream feature lists pick them up
        for _c in ("poisson_over25_prob", "poisson_btts_prob"):
            try:
                if _c not in OVER25_FEATURES:
                    OVER25_FEATURES.append(_c)
            except NameError:
                pass
            try:
                if _c not in EXPANDED_BTTS_FEATURES:
                    EXPANDED_BTTS_FEATURES.append(_c)
            except NameError:
                pass
    # ── ensure target columns exist so assignment is safe ──
    for col in ["btts_confidence", "over25_confidence",
                "btts_pred",       "over25_pred"]:
        if col not in df.columns:
            df[col] = 0.0 if "confidence" in col else 0

    # ---------- probability inference ----------
    # Align the dataframe to each model’s training schema (strict order)
    X_btts = _strict_align(df, btts_model).replace([np.inf, -np.inf], np.nan)
    X_btts = safe_fill(X_btts, X_btts.columns.tolist(), SENTINEL)

    X_over = _strict_align(df, over_model).replace([np.inf, -np.inf], np.nan)
    X_over = safe_fill(X_over, X_over.columns.tolist(), SENTINEL)

    prob_btts = np.clip(btts_model.predict_proba(X_btts)[:, 1], 0, 1)
    prob_over = np.clip(over_model.predict_proba(X_over)[:, 1], 0, 1)

    # ----- sanity guard: model must return non‑trivial probabilities -----
    if np.isnan(prob_btts).all() or prob_btts.max() == 0:
        raise RuntimeError("BTTS side‑model produced empty probabilities – check feature alignment")
    if np.isnan(prob_over).all() or prob_over.max() == 0:
        raise RuntimeError("Over2.5 side‑model produced empty probabilities – check feature alignment")

    df.loc[:, "btts_confidence"]   = prob_btts
    df.loc[:, "over25_confidence"] = prob_over

    # ---------- calibrated thresholds ----------
    # use model‑specific calibrated cut‑offs if present, else config defaults
    thr_btts = getattr(btts_model, "best_threshold_", getattr(btts_model, "opt_threshold_", config.get("thr_btts_default", 0.50)))
    thr_over = getattr(over_model, "best_threshold_", getattr(over_model, "opt_threshold_", config.get("thr_over25_default", 0.50)))

    # propagate thresholds for downstream diagnostics
    df.attrs["thr_btts"]    = float(thr_btts)
    df.attrs["thr_over25"]  = float(thr_over)
    # back-compat alias used by some older helpers
    df.attrs["thr_over"]    = float(thr_over)

    df.loc[:, "btts_pred"]   = (prob_btts > thr_btts).astype(int)
    df.loc[:, "over25_pred"] = (prob_over > thr_over).astype(int)

    # --- Derived inverse markets: UNDER 2.5 and BTTS NO -------------------
    # Use adjusted_* when available, else fall back to raw *_confidence (default 0.5)
    over_src = (
        df["adjusted_over25_confidence"]
        if "adjusted_over25_confidence" in df.columns
        else df.get("over25_confidence", pd.Series(0.5, index=df.index))
    )
    btts_src = (
        df["adjusted_btts_confidence"]
        if "adjusted_btts_confidence" in df.columns
        else df.get("btts_confidence", pd.Series(0.5, index=df.index))
    )
    df["under25_confidence"] = (
        1.0 - pd.to_numeric(over_src, errors="coerce")
        .fillna(0.5)
        .astype(float)
        .clip(0, 1)
    )
    df["btts_no_confidence"] = (
        1.0 - pd.to_numeric(btts_src, errors="coerce")
        .fillna(0.5)
        .astype(float)
        .clip(0, 1)
    )

    # thresholds for inverse markets (single source of truth: config)
    thr_under25 = float(config.get("thr_under25_default", 0.60))
    thr_btts_no = float(config.get("thr_btts_no_default", 0.60))
    df.attrs["thr_under25"] = thr_under25
    df.attrs["thr_btts_no"] = thr_btts_no

    # binary labels for inverse markets
    df["under25_pred"] = (df["under25_confidence"] >= thr_under25).astype(int)
    df["btts_no_pred"] = (df["btts_no_confidence"] >= thr_btts_no).astype(int)

    return df

def adjust_with_volatility_modifiers(df):
    if 'btts_confidence' in df.columns and 'btts_overlay_modifier' in df.columns:
        df['adjusted_btts_confidence'] = df['btts_confidence'] + df['btts_overlay_modifier'] * 0.1
    else:
        df['adjusted_btts_confidence'] = df.get('btts_confidence', 0.5)

    if 'over25_confidence' in df.columns and 'over25_overlay_modifier' in df.columns:
        df['adjusted_over25_confidence'] = df['over25_confidence'] + df['over25_overlay_modifier'] * 0.1
    else:
        df['adjusted_over25_confidence'] = df.get('over25_confidence', 0.5)

        # never let modifiers push probabilities outside [0,1]
    df["adjusted_btts_confidence"]   = df["adjusted_btts_confidence"].clip(0, 1)
    df["adjusted_over25_confidence"] = df["adjusted_over25_confidence"].clip(0, 1)

    return df


def log_prediction_changes(df):
    thr_btts = df.attrs.get("thr_btts", 0.5)
    thr_over = df.attrs.get("thr_over", 0.5)

    df['btts_changed']   = df['btts_pred'] != (df['adjusted_btts_confidence']  > thr_btts).astype(int)
    df['over25_changed'] = df['over25_pred'] != (df['adjusted_over25_confidence'] > thr_over).astype(int)
    return df

# --------------------------------------------------------------
# Win-to-Nil proxy from FTS (Fail-To-Score) probabilities
# --------------------------------------------------------------
def attach_win_to_nil_proxy(df: pd.DataFrame,
                            *,
                            home_win_prob_col: str = "confidence_home",
                            away_win_prob_col: str = "confidence_away",
                            home_fts_prob_cols = ("home_fts_confidence", "prob_home_fts", "p_home_fts"),
                            away_fts_prob_cols = ("away_fts_confidence", "prob_away_fts", "p_away_fts")) -> pd.DataFrame:
    """
    Approximate:
        P(Home win to nil) ≈ P(Home Win) × P(Away FTS)
        P(Away win to nil) ≈ P(Away Win) × P(Home FTS)
    Creates columns if inputs exist:
        - p_home_win_to_nil
        - p_away_win_to_nil
    """
    import pandas as _pd

    if home_win_prob_col not in df.columns or away_win_prob_col not in df.columns:
        return df

    def _first_present(cols):
        for c in cols:
            if c in df.columns:
                return c
        return None

    hfts_col = _first_present(home_fts_prob_cols)
    afts_col = _first_present(away_fts_prob_cols)
    if hfts_col is None or afts_col is None:
        return df

    df["p_home_win_to_nil"] = (
        _pd.to_numeric(df[home_win_prob_col], errors="coerce").fillna(0.0)
        * _pd.to_numeric(df[afts_col], errors="coerce").fillna(0.0)
    ).clip(0, 1)
    df["p_away_win_to_nil"] = (
        _pd.to_numeric(df[away_win_prob_col], errors="coerce").fillna(0.0)
        * _pd.to_numeric(df[hfts_col], errors="coerce").fillna(0.0)
    ).clip(0, 1)
    return df
def _ensure_report_probs_inplace_dup(df):
    """
    Create compatibility probability columns used by older ROI code paths:
      • prob_over25, prob_btts  (from adjusted_* → raw *_confidence → 0.5)
      • Over25, BTTS            (binary flags from those probs when missing)
    Works in-place and returns df.
    """
    import pandas as pd
    if not isinstance(df, pd.DataFrame):
        return df
    # --- SAFE: ensure Series fallback so .fillna() exists ---
    if "prob_over25" not in df.columns:
        src_over25 = (
            df["adjusted_over25_confidence"] if "adjusted_over25_confidence" in df.columns
            else (df["over25_confidence"] if "over25_confidence" in df.columns
                  else pd.Series(0.5, index=df.index))
        )
        df["prob_over25"] = pd.to_numeric(src_over25, errors="coerce").fillna(0.5).clip(0, 1)
    # --- SAFE: ensure Series fallback so .fillna() exists ---
    if "prob_btts" not in df.columns:
        src_btts = (
            df["adjusted_btts_confidence"] if "adjusted_btts_confidence" in df.columns
            else (df["btts_confidence"] if "btts_confidence" in df.columns
                  else pd.Series(0.5, index=df.index))
        )
        df["prob_btts"] = pd.to_numeric(src_btts, errors="coerce").fillna(0.5).clip(0, 1)
    if "Over25" not in df.columns:
        df["Over25"] = (df["prob_over25"] > 0.5).astype(int)
    if "BTTS" not in df.columns:
        df["BTTS"] = (df["prob_btts"] > 0.5).astype(int)
    return df

#############################################
# Monte‑Carlo ROI simulation utilities
#############################################
def _simulate_acca_roi(cand: pd.DataFrame,
                       conf_col: str,
                       odds_col: str,
                       acca_size: int,
                       n_trials: int | None = None,
                       stake: float | None = None) -> dict:
    """
    Vectorised Monte‑Carlo ROI simulation of random accumulators.
    Returns mean ROI and 95 % CI (default: 10‑fold).
    """
    if n_trials is None:
        n_trials = config.get("roi_n_trials", 10_000)
    if stake is None:
        stake = config.get("stake_per_acca", 10)

    # --- normalize simulation sizes to plain ints for typing/runtime safety ---
    try:
        n_trials_i = _to_int(n_trials if n_trials is not None else config.get("roi_n_trials", 10_000))
    except Exception:
        n_trials_i = _to_int(10_000)

    try:
        acca_i = _to_int(acca_size)
    except Exception:
        acca_i = _to_int(2)

    # Tail-calming knobs from config (with safe defaults) — normalized to concrete types
    min_pool_mult_i = _to_int(config.get("min_pool_x", 5))
    prob_lo_f = _to_float(config.get("prob_clip_low", 0.03))
    prob_hi_f = _to_float(config.get("prob_clip_high", 0.97))
    _max_leg_raw = config.get("max_leg_odds", None)
    max_leg_odds_f = (_to_float(_max_leg_raw) if _max_leg_raw is not None else None)
    _max_acca_raw = config.get("max_acca_size", None)
    max_acca_size_i = (_to_int(_max_acca_raw) if _max_acca_raw is not None else None)

    # Respect optional global folds cap
    if max_acca_size_i is not None:
        if acca_i > max_acca_size_i:
            return {"mean_roi": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                    "std_roi": np.nan, "var95": np.nan, "sharpe_like": np.nan}

    # Require a reasonable pool size to stabilise tails
    if len(cand) < max(3, min_pool_mult_i * acca_i):
        return {"mean_roi": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "std_roi": np.nan, "var95": np.nan, "sharpe_like": np.nan}

    acca_i = min(acca_i, len(cand))  # fallback if not enough picks
    # guard against empty candidate set after down‑sampling
    if acca_i == 0:
        return {
            "mean_roi": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "std_roi": np.nan,
            "var95": np.nan,
            "sharpe_like": np.nan
        }

    prob = cand[conf_col].to_numpy(dtype=float)
    # Clip probabilities to calm tails
    prob = np.clip(prob, prob_hi_f if prob_hi_f is not None else 0.97, prob_hi_f if prob_hi_f is not None else 0.97)
    # Re-clip lower bound explicitly using prob_lo_f
    prob = np.maximum(prob, prob_lo_f if prob_lo_f is not None else 0.03)

    # Clip per-leg odds to avoid extreme tails in MC
    odds = cand[odds_col].to_numpy(dtype=float)
    if (max_leg_odds_f is not None) and (max_leg_odds_f > 0):
        odds = np.minimum(odds, float(max_leg_odds_f))

    rng  = np.random.default_rng(int(config.get("rng_seed", 42)))
    # ------------------------------------------------------------------
    # Sampling logic
    #   • Prefer fast vectorised `choice` when the fold is smaller than
    #     the population.
    #   • If `choice` ever complains (rare edge‑case with multi‑dim size),
    #     or when fold==population, fall back to a per‑trial permutation
    #     which guarantees unique legs inside each acca.
    # ------------------------------------------------------------------
    if acca_i < len(prob):
        try:
            idxs = rng.choice(
                len(prob),
                size=(n_trials_i, acca_i),
                replace=False
            )
        except ValueError:
            # Safety fallback – still keep trials independent
            idxs = np.vstack([
                rng.permutation(len(prob))[:acca_i]
                for _ in range(n_trials_i)
            ])
    else:  # acca_i equals the full candidate pool
        idxs = np.vstack([
            rng.permutation(len(prob))[:acca_i]
            for _ in range(n_trials_i)
        ])

    # Safe log-sum (avoid -inf from log(0))
    _plo = prob_lo_f if prob_lo_f is not None else 0.03
    logp = np.log(prob, where=(prob > 0), out=np.full_like(prob, np.log(_plo)))
    acca_prob = np.exp(logp[idxs].sum(axis=1))
    acca_odds = np.exp(np.log(odds)[idxs].sum(axis=1))
    payout    = stake * acca_odds * (rng.random(n_trials_i) < acca_prob)
    roi_vec   = (payout - stake) / stake

    mean_roi  = float(roi_vec.mean())
    lo, hi    = map(float, np.percentile(roi_vec, [2.5, 97.5]))
    std_roi  = float(roi_vec.std(ddof=0))
    var95    = float(np.percentile(roi_vec, 5))   # 5‑percent VaR
    sharpe   = float(mean_roi / std_roi) if std_roi > 0 else np.nan
    return {
        "mean_roi": mean_roi,
        "ci_low": lo,
        "ci_high": hi,
        "std_roi": std_roi,
        "var95": var95,
        "sharpe_like": sharpe
    }

def simulate_accumulator_roi(df: pd.DataFrame, league_name: str | None = None, stake: float = 10.0) -> dict:
    """Print and return a simple ROI snapshot for Over 2.5, BTTS, and (optionally) Over2.5+BTTS combo.
    Only settles bets for rows that have *both* realized outcomes and valid odds (≥1.01).
    """
    # Ensure minimal shape & columns the ROI logic relies on
    try:
        df = ensure_realized_goal_placeholders(df)
    except Exception:
        pass
    try:
        _ensure_report_probs_inplace(df)
    except Exception:
        pass
    try:
        _ensure_roi_inputs_inplace(df)
    except Exception:
        pass

    # Delegate to the guarded printer (also returns a results dict)
    return _print_roi_snapshot(df, stake=stake)


#
# --------------------------------------------------------------
# Helper: attach p_model, odds, edge, kelly_frac, expected_value
# --------------------------------------------------------------

# --------------------------------------------------------------
# File helpers for ROI/P&L and global-picks exports
# --------------------------------------------------------------

def _write_roi_csv(league_name: str, tag: str, stats: dict) -> None:
    """
    Persist a one-row CSV with ROI sim stats.
    Example filename: predictions_output/<YYYY-MM-DD>/<League>_<tag>_roi.csv
    """
    try:
        outdir = os.path.join("predictions_output", datetime.datetime.utcnow().strftime("%Y-%m-%d"))
        os.makedirs(outdir, exist_ok=True)
        fname = f"{league_name.replace(' ', '_')}_{tag}_roi.csv"
        fpath = os.path.join(outdir, fname)
        cols = [
            "folds", "mean_roi", "ci_low", "ci_high",
            "std_roi", "var95", "sharpe_like"
        ]
        row = {k: stats.get(k, "") for k in cols}
        pd.DataFrame([row], columns=cols).to_csv(fpath, index=False)
    except Exception as e:
        try:
            print(f"⚠️ Could not write ROI CSV for {league_name} {tag}: {e}")
        except Exception:
            pass

def _write_pnl_csv(league_name: str, market: str, pnl: dict) -> None:
    """
    Persist a one-row CSV with walk-forward P&L metrics.
    Example filename: predictions_output/<YYYY-MM-DD>/<League>_<market>_walkforward_pnl.csv
    """
    try:
        outdir = os.path.join("predictions_output", datetime.datetime.utcnow().strftime("%Y-%m-%d"))
        os.makedirs(outdir, exist_ok=True)
        fname = f"{league_name.replace(' ', '_')}_{market}_walkforward_pnl.csv"
        fpath = os.path.join(outdir, fname)
        # Preserve key order where possible; fall back to sorted keys
        base_cols = [
            "final_roi", "max_dd", "num_bets", "win_rate", "avg_odds",
            "avg_edge", "total_return", "strategy"
        ]
        cols = [c for c in base_cols if c in pnl] or sorted(pnl.keys())
        pd.DataFrame([{k: pnl.get(k, "") for k in cols}], columns=cols).to_csv(fpath, index=False)
    except Exception as e:
        try:
            print(f"⚠️ Could not write walk-forward P&L CSV for {league_name} {market}: {e}")
        except Exception:
            pass

# --- ModelStore CSV writers (write to ModelStore/...) ---
def write_modelstore_roi_csv(league_name: str, tag: str, stats: dict) -> None:
    """
    Persist a one-row CSV with ROI sim stats to ModelStore.
    Example filename: ModelStore/<League>_<tag>_roi.csv
    """
    try:
        outdir = MODEL_DIR
        os.makedirs(outdir, exist_ok=True)
        fname = f"{league_name.replace(' ', '_')}_{tag}_roi.csv"
        fpath = os.path.join(outdir, fname)
        cols = [
            "folds", "mean_roi", "ci_low", "ci_high",
            "std_roi", "var95", "sharpe_like"
        ]
        row = {k: stats.get(k, "") for k in cols}
        pd.DataFrame([row], columns=cols).to_csv(fpath, index=False)
    except Exception as e:
        try:
            print(f"⚠️ Could not write ModelStore ROI CSV for {league_name} {tag}: {e}")
        except Exception:
            pass

def write_modelstore_pnl_csv(league_name: str, market: str, pnl: dict) -> None:
    """
    Persist a one-row CSV with walk-forward P&L metrics to ModelStore.
    Example filename: ModelStore/<League>_<market>_walkforward_pnl.csv
    """
    try:
        outdir = MODEL_DIR
        os.makedirs(outdir, exist_ok=True)
        fname = f"{league_name.replace(' ', '_')}_{market}_walkforward_pnl.csv"
        fpath = os.path.join(outdir, fname)
        base_cols = [
            "final_roi", "max_dd", "num_bets", "win_rate", "avg_odds",
            "avg_edge", "total_return", "strategy"
        ]
        cols = [c for c in base_cols if c in pnl] or sorted(pnl.keys())
        pd.DataFrame([{k: pnl.get(k, "") for k in cols}], columns=cols).to_csv(fpath, index=False)
    except Exception as e:
        try:
            print(f"⚠️ Could not write ModelStore walk-forward P&L CSV for {league_name} {market}: {e}")
        except Exception:
            pass


def generate_accumulator_recommendations(df,
                                         target_type: str = 'btts',
                                         top_n: int = 10,
                                         league_name: str = 'Unknown',
                                         **kwargs):
    """
    Returns a dataframe of the top‑N selections by expected value (EV).
    EV is calculated per‑row as  (model_prob * decimal_odds) − 1.
    Rows with missing odds or negative EV are discarded.

    Supports accumulators for: BTTS, Over2.5, FTR, WTN (Win-To-Nil), UNDER2.5, BTTS NO, CLEAN_SHEET
    """
    # Early bail if this market was disabled earlier in the run
    if target_type in globals().get("DISABLED_MARKETS", set()):
        print(f"⏭️ market '{target_type}' disabled for this run; skipping.")
        return pd.DataFrame()
    
    # Single-source-of-truth thresholds from config
    thresholds = {
        'btts': float(config.get("thr_btts_default", 0.55)),
        'over25': float(config.get("thr_over25_default", 0.32)),
        'ftr': float(config.get('thr_ftr_default', 0.40)),
        'wtn': float(config.get('thr_wtn_default', 0.25)),
        'under25': float(config.get("thr_under25_default", 0.60)),
        'btts_no': float(config.get("thr_btts_no_default", 0.60)),
        'clean_sheet': float(config.get('thr_clean_sheet_default', 0.25)),
    }
        # === NEW: ingest saved thresholds/context and prefer trained models ===
    try:
        df = _load_draw_context_into(df, league_name)
    except Exception:
        pass
    try:
        df = _apply_market_thresholds_to_attrs(df, league_name)
    except Exception:
        pass
    # Attach trained-market probabilities once per dataframe if available
    try:
        if not bool(df.attrs.get("_markets_scored", False)):
            _df2, _used = _attach_trained_market_scores_if_available(
                df, league_name,
                markets=["btts","over25","under25","btts_no","ftr","wtn","clean_sheet"]
            )
            if _used:
                df = _df2
                try:
                    _log("INFO", "🔮 Using trained market models where available (ModelStore).")
                except Exception:
                    pass
            df.attrs["_markets_scored"] = True
    except Exception:
        pass
    # Resolve per-market threshold, preferring CLI/attrs overrides when present
    def _resolve_min_conf(market: str) -> float:
        override_keys = {
            'btts': 'thr_btts',
            'over25': 'thr_over25',
            'under25': 'thr_under25',
            'btts_no': 'thr_btts_no',
            'ftr': 'thr_ftr',
            'wtn': 'thr_wtn',
            'clean_sheet': 'thr_clean_sheet',
        }
        key = override_keys.get(market)
        if key is not None and key in df.attrs:
            try:
                return float(df.attrs.get(key))
            except Exception:
                pass
        return float(thresholds.get(market, 0.5))

    def _any_present(cols: list[str]) -> bool:
        """Return True if any column in *cols* exists in df.columns."""
        try:
            return any(c in df.columns for c in cols)
        except Exception:
            return False

    # Normalise any 0..100 percentage/prob columns to 0..1 (defensive)
    df = _normalise_prob_columns(_coerce_numeric_like(df, odds_whitelist=ODD_COLUMN_WHITELIST))
    df = _ensure_report_probs_inplace(df)

    def _first_present(cols):
        for c in cols:
            if c in df.columns:
                return c
        return None

    # mapping for confidence/odds columns per market
    conf_col_map = {
        'btts': 'adjusted_btts_confidence',
        'over25': 'adjusted_over25_confidence'
    }
    odds_col_map = {
        'btts': 'odds_btts_yes',
        'over25': 'odds_ft_over25'
    }
    conf_col_map.update({
        'under25': 'under25_confidence',
        'btts_no': 'btts_no_confidence',
    })
    odds_col_map.update({
        'under25': 'odds_ft_under25',
        'btts_no': 'odds_btts_no',
    })

    # Resolve odds columns and select branch per market
    min_conf = _resolve_min_conf(target_type)
    try:
        _log("INFO", f"📊 Filtering {target_type.upper()} predictions with threshold: {min_conf:.2f}")
    except Exception:
        pass

    # Guard + objective knobs
    allow_synth = bool(
        globals().get("overlay_config", {}).get(
            "allow_synth_odds", config.get("allow_synth_odds", False)
        )
    )
    probs_from_odds_flag = bool(df.attrs.get("probs_from_odds", False))
    forbid_synth = (probs_from_odds_flag and not allow_synth)

    objective = str(globals().get("overlay_config", {}).get("objective", config.get("objective", "roi"))).lower()
    sort_key  = str(globals().get("overlay_config", {}).get("sort_key",  config.get("sort_key",  "ev"))).lower()
        
        # Simple markets first: BTTS / Over2.5 / BTTS NO / Under 2.5
    if target_type in ("btts", "over25", "btts_no", "under25"):
        # resolve per-market columns
        if target_type == 'btts':
            odds_col = _resolve_odds(df, ODD_COLUMN_WHITELIST['btts_yes'], label='BTTS YES odds')
            conf_col = 'adjusted_btts_confidence' if 'adjusted_btts_confidence' in df.columns else (
                'btts_confidence' if 'btts_confidence' in df.columns else None
            )
            pred_col = 'btts_pred'
            canonical = 'odds_btts_yes'
        elif target_type == 'over25':
            odds_col = _resolve_odds(df, ODD_COLUMN_WHITELIST['over25'], label='Over 2.5 odds')
            conf_col = 'adjusted_over25_confidence' if 'adjusted_over25_confidence' in df.columns else (
                'over25_confidence' if 'over25_confidence' in df.columns else None
            )
            pred_col = 'over25_pred'
            canonical = 'odds_ft_over25'
        elif target_type == 'btts_no':
            odds_col = _resolve_odds(df, ODD_COLUMN_WHITELIST['btts_no'], label='BTTS NO odds')
            conf_col = 'btts_no_confidence'
            pred_col = 'btts_no_pred'
            canonical = 'odds_btts_no'
        else:  # under25
            odds_col = _resolve_odds(df, ODD_COLUMN_WHITELIST['under25'], label='Under 2.5 odds')
            conf_col = 'under25_confidence'
            pred_col = 'under25_pred'
            canonical = 'odds_ft_under25'

        # need a probability column for this market
        if conf_col is None or conf_col not in df.columns:
            _disable_market(target_type, f"missing {target_type.upper()} confidence column(s)")
            return pd.DataFrame()

        # If odds are missing → either forbid (when probs came from odds & synth disabled) or synthesise
        if odds_col is None:
            if forbid_synth:
                _disable_market(
                    target_type,
                    "missing odds and synthesis disabled (probs_from_odds & allow_synth_odds=False)"
                )
                return pd.DataFrame()
            synth_name = {
                'btts': 'synth_btts_yes_odds',
                'over25': 'synth_over25_odds',
                'btts_no': 'synth_btts_no_odds',
                'under25': 'synth_under25_odds',
            }[target_type]
            maybe = _synthesise_odds_from_probs(df, conf_col, synth_name)
            if maybe is None:
                _disable_market(target_type, f"no odds columns found for {target_type} and could not synthesise")
                return pd.DataFrame()
            odds_col = maybe
            if canonical not in df.columns:
                df[canonical] = df[odds_col]

        # attach market odds and compute expected value (EV)
        conf_series = pd.to_numeric(df.get(conf_col), errors='coerce')
        pred_series = pd.to_numeric(df.get(pred_col), errors='coerce').fillna(0).astype(int)
        odds_series = pd.to_numeric(df.get(odds_col), errors='coerce')
        mask = (
            (conf_series >= min_conf) &
            (pred_series == 1) &
            (odds_series.notna())
        )
        candidates = df.loc[mask].copy()

        candidates.loc[:, odds_col] = pd.to_numeric(candidates[odds_col], errors="coerce")
        candidates = candidates.dropna(subset=[odds_col])

        candidates = _ensure_draw_context_for_export(candidates, league_name)

        # EV with conservative fair-odds tweak
        margin = float(df.attrs.get('book_margin', 0.0))
        probs_from_odds = bool(df.attrs.get("probs_from_odds", False))
        synth_odds = str(odds_col).startswith('synth_')

        candidates = _attach_ev_columns(
            candidates,
            conf_col,
            odds_col,
            margin=margin,
            probs_from_odds=probs_from_odds,
            synth_odds=synth_odds,
        )

        ev_cushion = 0.02 if (probs_from_odds or synth_odds) else 0.0
        candidates = candidates[candidates["expected_value"] > -ev_cushion]

        # ROI sims
        min_pool_mult = int(config.get("min_pool_x", 5))
        max_acca_size = config.get("max_acca_size", None)
        for sz in config.get("acca_sizes", [10]):
            if isinstance(max_acca_size, (int, float)) and max_acca_size is not None and sz > int(max_acca_size):
                if not config.get("quiet_skips", False):
                    _log("INFO", f"⏭️ Skipped {sz}-fold {target_type.upper()} sim (capped by max_acca_size={int(max_acca_size)}).")
                continue
            required = max(3, int(min_pool_mult) * int(sz))
            if len(candidates) < required:
                if not config.get("quiet_skips", False):
                    _log("INFO", f"⏭️ Skipped {sz}-fold {target_type.upper()} sim (pool {len(candidates)} < {int(min_pool_mult)}×{int(sz)}).")
                continue
            roi_stats = simulate_accumulator_roi(candidates, target_type, sz)
            if np.isfinite(roi_stats.get("mean_roi", np.nan)):
                _log("INFO",
                    f"⚡ Sim {sz}-fold {target_type.upper()} ROI: "
                    f"{roi_stats['mean_roi']*100: .2f}%  "
                    f"[95% CI: {roi_stats['ci_low']*100: .1f} … {roi_stats['ci_high']*100: .1f}]  "
                    f"σ={roi_stats['std_roi']*100: .1f}%  VaR5={roi_stats['var95']*100: .1f}%"
                )
                write_modelstore_roi_csv(league_name, f"{target_type}_{sz}x", roi_stats if isinstance(roi_stats, dict) else roi_stats.to_dict())
            else:
                if not config.get("quiet_skips", False):
                    print(f"⏭️ Skipped {sz}-fold {target_type.upper()} sim (insufficient pool or capped).")

        # Walk-forward P&L (flat vs ½-Kelly)
        for _kelly in (False, True):
            # normalize stake to float|None for typing
            _stake_val = config.get("stake_per_acca", 100)
            try:
                stake_f: float | None = None if _stake_val is None else _to_float(_stake_val)
            except Exception:
                stake_f = None if _stake_val is None else _to_float(_stake_val)

            pnl = simulate_walk_forward_pnl(candidates, target_type,
                                            stake=stake_f,
                                            kelly=_kelly)
            if pnl:
                tag = "½-Kelly" if _kelly else "flat"
                print(f"📈 Walk-forward {target_type.upper()} ROI ({tag}): "
                      f"{pnl['final_roi']*100: .1f}% | Max DD {pnl['max_dd']*100: .1f}%")
                write_modelstore_pnl_csv(league_name, target_type, pnl)

        # Build accuracy/watchlist shortlist (probability-only gating)
        acc_conf_col = conf_col
        acc_pred_col = pred_col
        try:
            acc_mask = pd.to_numeric(df.get(acc_conf_col), errors="coerce") >= float(min_conf)
        except Exception:
            acc_mask = pd.Series(False, index=df.index)
        if acc_pred_col in df.columns:
            try:
                acc_mask = acc_mask & (pd.to_numeric(df.get(acc_pred_col), errors="coerce").fillna(0).astype(int) == 1)
            except Exception:
                pass
        accuracy_df = df.loc[acc_mask].copy()
        accuracy_df = _ensure_draw_context_for_export(accuracy_df, league_name)
        accuracy_df["thr_effective"] = float(min_conf)

        # Export accuracy shortlist
        try:
            if league_name:
                outdir_acc = os.path.join("predictions_output", datetime.datetime.utcnow().strftime("%Y-%m-%d"))
                os.makedirs(outdir_acc, exist_ok=True)
                outfile_acc = os.path.join(outdir_acc, f"{league_name.replace(' ', '_')}_top_{target_type}_accuracy.csv")
                export_cols_acc = ['home_team_name','away_team_name','match_date', acc_conf_col, 'thr_effective']
                for _extra in ("p_draw","draw_flag","draw_flag_topk","thr_draw","mode_draw","alpha_used","best_k_pct","is_topk_best"):
                    if _extra in accuracy_df.columns and _extra not in export_cols_acc:
                        export_cols_acc.append(_extra)
                (accuracy_df
                    .sort_values(by=[acc_conf_col], ascending=False)
                    .head(top_n)
                    .to_csv(outfile_acc, index=False, columns=[c for c in export_cols_acc if c in accuracy_df.columns]))
                _log("INFO", f"📤 Saved top {target_type.upper()} accuracy picks to: {outfile_acc}")
        except Exception as e:
            _log("WARN", f"⚠️ Accuracy export failed for {target_type.upper()}: {e}")

        # Final return depends on objective
        if objective == "accuracy" or (probs_from_odds_flag and objective != "hybrid"):
            return (
                accuracy_df
                .sort_values(by=[acc_conf_col], ascending=False)
                .head(top_n)
            )
        elif objective == "hybrid":
            sort_cols = []
            if sort_key in ("edge","ev") and "edge" in candidates.columns:
                sort_cols.append("edge")
            if "expected_value" in candidates.columns:
                sort_cols.append("expected_value")
            if acc_conf_col in candidates.columns:
                sort_cols.append(acc_conf_col)
            sort_cols = sort_cols or [acc_conf_col]
            return candidates.sort_values(by=sort_cols, ascending=False).head(top_n)
        else:
            return (
                candidates
                .sort_values(by=["expected_value", conf_col], ascending=False)
                .head(top_n)
            )
            
    elif target_type == 'wtn':
        prob_home_col = 'p_home_win_to_nil'
        prob_away_col = 'p_away_win_to_nil'
        if prob_home_col not in df.columns or prob_away_col not in df.columns:
            _disable_market('wtn', 'missing Win-To-Nil probability columns; run attach_win_to_nil_proxy() first')
            _log("WARN", "⚠️ Missing Win-To-Nil probability columns; run attach_win_to_nil_proxy() first. Skipping WTN.")
            return pd.DataFrame()

        odds_home_col = _resolve_odds(df, ODD_COLUMN_WHITELIST['home_wtn'], label='HOME WTN odds')
        odds_away_col = _resolve_odds(df, ODD_COLUMN_WHITELIST['away_wtn'], label='AWAY WTN odds')
        if odds_home_col is None or odds_away_col is None:
            if forbid_synth:
                _disable_market('wtn', 'missing WTN odds and synthesis disabled (probs_from_odds & allow_synth_odds=False)')
                return pd.DataFrame()
            if odds_home_col is None:
                odds_home_col = _synthesise_odds_from_probs(df, prob_home_col, 'synth_home_wtn_odds')
            if odds_away_col is None:
                odds_away_col = _synthesise_odds_from_probs(df, prob_away_col, 'synth_away_wtn_odds')
            if odds_home_col is None or odds_away_col is None:
                _disable_market('wtn', 'missing WTN odds columns and could not synthesise from probabilities')
                return pd.DataFrame()

        def _select_wtn(row):
            ph = float(row.get(prob_home_col, 0) or 0)
            pa = float(row.get(prob_away_col, 0) or 0)
            oh = float(row.get(odds_home_col, float('nan')))
            oa = float(row.get(odds_away_col, float('nan')))
            ev_h = ph * oh - 1.0 if not pd.isna(oh) else -1
            ev_a = pa * oa - 1.0 if not pd.isna(oa) else -1
            if ev_h >= ev_a:
                return pd.Series([ph, oh, 'home'])
            else:
                return pd.Series([pa, oa, 'away'])

        df[['selected_wtn_confidence', 'selected_wtn_odds', 'wtn_side']] = df.apply(_select_wtn, axis=1)

        margin = float(df.attrs.get('book_margin', 0.0))
        probs_from_odds = bool(df.attrs.get('probs_from_odds', False))
        synth_odds = str(odds_home_col).startswith('synth_') or str(odds_away_col).startswith('synth_')

        candidates = _attach_ev_columns(
            df.copy(),
            'selected_wtn_confidence',
            'selected_wtn_odds',
            margin=margin,
            probs_from_odds=probs_from_odds,
            synth_odds=synth_odds,
        )

        min_conf = _resolve_min_conf('wtn')
        ev_cushion = 0.02 if (probs_from_odds or synth_odds) else 0.0
        candidates = candidates[(candidates['selected_wtn_confidence'] >= min_conf) & (candidates['expected_value'] > -ev_cushion)].copy()
        candidates.dropna(subset=['selected_wtn_odds'], inplace=True)
        candidates = _ensure_draw_context_for_export(candidates, league_name)

        # --- ROI sims ---
        min_pool_mult = int(config.get("min_pool_x", 5))
        max_acca_size = config.get("max_acca_size", None)
        for sz in config.get("acca_sizes", [10]):
            if isinstance(max_acca_size, (int, float)) and max_acca_size is not None and sz > int(max_acca_size):
                if not config.get("quiet_skips", False):
                    _log("INFO", f"⏭️ Skipped {sz}-fold WTN sim (capped by max_acca_size={int(max_acca_size)}).")
                continue
            required = max(3, int(min_pool_mult) * int(sz))
            if len(candidates) < required:
                if not config.get("quiet_skips", False):
                    _log("INFO", f"⏭️ Skipped {sz}-fold WTN sim (pool {len(candidates)} < {int(min_pool_mult)}×{int(sz)}).")
                continue
            roi_stats = simulate_accumulator_roi(candidates, 'wtn', sz)
            print(f"⚡ Sim {sz}-fold WTN ROI: {roi_stats['mean_roi']*100: .2f}%  "
                  f"[95% CI: {roi_stats['ci_low']*100: .1f} … {roi_stats['ci_high']*100: .1f}]  "
                  f"σ={roi_stats['std_roi']*100: .1f}%  VaR5={roi_stats['var95']*100: .1f}%")
            _write_roi_csv(league_name, f"wtn_{sz}x", roi_stats if isinstance(roi_stats, dict) else roi_stats.to_dict())

        # Walk-forward P&L (flat vs ½-Kelly)
        for _kelly in (False, True):
            # normalize stake to float|None for typing
            _stake_val = config.get("stake_per_acca", 100)
            try:
                stake_f: float | None = None if _stake_val is None else _to_float(_stake_val)
            except Exception:
                stake_f = None if _stake_val is None else _to_float(_stake_val)

            pnl = simulate_walk_forward_pnl(candidates, "wtn",
                                            stake=stake_f,
                                            kelly=_kelly)
            if pnl:
                tag = "½-Kelly" if _kelly else "flat"
                print(f"📈 Walk-forward WTN ROI ({tag}): "
                      f"{pnl['final_roi']*100: .1f}% | Max DD {pnl['max_dd']*100: .1f}%")
                _write_pnl_csv(league_name, 'wtn', pnl)

        
    elif target_type == 'clean_sheet':
        # Clean Sheet accumulator. Try explicit CS probabilities first; else derive from FTS.
        p_home_cs_col = None
        p_away_cs_col = None
        for cand in ('p_home_cs','prob_home_clean_sheet','home_cs_confidence'):
            if cand in df.columns:
                p_home_cs_col = cand; break
        for cand in ('p_away_cs','prob_away_clean_sheet','away_cs_confidence'):
            if cand in df.columns:
                p_away_cs_col = cand; break

        # derive from FTS if not explicitly provided
        if p_home_cs_col is None and 'p_away_fts' in df.columns:
            p_home_cs_col = 'p_away_fts'
        if p_away_cs_col is None and 'p_home_fts' in df.columns:
            p_away_cs_col = 'p_home_fts'
        if p_home_cs_col is None or p_away_cs_col is None:
            _disable_market('clean_sheet', 'missing clean sheet or FTS probabilities')
            return pd.DataFrame()

        odds_home_cs = _resolve_odds(df, ODD_COLUMN_WHITELIST['home_cs'], label='HOME clean sheet odds')
        odds_away_cs = _resolve_odds(df, ODD_COLUMN_WHITELIST['away_cs'], label='AWAY clean sheet odds')
        if odds_home_cs is None or odds_away_cs is None:
            if forbid_synth:
                _disable_market('clean_sheet', 'missing clean sheet odds and synthesis disabled (probs_from_odds & allow_synth_odds=False)')
                return pd.DataFrame()
            if odds_home_cs is None:
                odds_home_cs = _synthesise_odds_from_probs(df, p_home_cs_col, 'synth_home_cs_odds')
            if odds_away_cs is None:
                odds_away_cs = _synthesise_odds_from_probs(df, p_away_cs_col, 'synth_away_cs_odds')
            if odds_home_cs is None or odds_away_cs is None:
                _disable_market('clean_sheet', 'missing clean sheet odds columns and could not synthesise from probabilities')
                return pd.DataFrame()
            
        def _row_select_cs(row):
            phcs = float(pd.to_numeric(row.get(p_home_cs_col, 0), errors='coerce') or 0)
            pacs = float(pd.to_numeric(row.get(p_away_cs_col, 0), errors='coerce') or 0)
            ohcs = float(pd.to_numeric(row.get(odds_home_cs, np.nan), errors='coerce'))
            oacs = float(pd.to_numeric(row.get(odds_away_cs, np.nan), errors='coerce'))
            ev_h = phcs * ohcs - 1.0 if not pd.isna(ohcs) else -1
            ev_a = pacs * oacs - 1.0 if not pd.isna(oacs) else -1
            if ev_h >= ev_a:
                return pd.Series([phcs, ohcs, 'home'])
            else:
                return pd.Series([pacs, oacs, 'away'])

        df[['selected_cs_confidence', 'selected_cs_odds', 'cs_side']] = df.apply(_row_select_cs, axis=1)

        # EV/edge/Kelly attach
        margin = float(df.attrs.get('book_margin', 0.0))
        probs_from_odds = bool(df.attrs.get('probs_from_odds', False))
        synth_odds = str(odds_home_cs).startswith('synth_') or str(odds_away_cs).startswith('synth_')

        candidates = _attach_ev_columns(
            df.copy(),
            'selected_cs_confidence',
            'selected_cs_odds',
            margin=margin,
            probs_from_odds=probs_from_odds,
            synth_odds=synth_odds,
        )

        # Threshold & EV cushion
        min_conf = _resolve_min_conf('clean_sheet')
        ev_cushion = 0.02 if (probs_from_odds or synth_odds) else 0.0
        candidates = candidates[(candidates['selected_cs_confidence'] >= min_conf) &
                                (candidates['expected_value'] > -ev_cushion)].copy()
        candidates.dropna(subset=['selected_cs_odds'], inplace=True)
        candidates = _ensure_draw_context_for_export(candidates, league_name)

        # ROI sims
        min_pool_mult = int(config.get("min_pool_x", 5))
        max_acca_size = config.get("max_acca_size", None)
        for sz in config.get("acca_sizes", [10]):
            if isinstance(max_acca_size, (int, float)) and max_acca_size is not None and sz > int(max_acca_size):
                if not config.get("quiet_skips", False):
                    _log("INFO", f"⏭️ Skipped {sz}-fold CLEAN_SHEET sim (capped by max_acca_size={int(max_acca_size)}).")
                continue
            required = max(3, int(min_pool_mult) * int(sz))
            if len(candidates) < required:
                if not config.get("quiet_skips", False):
                    _log("INFO", f"⏭️ Skipped {sz}-fold CLEAN_SHEET sim (pool {len(candidates)} < {int(min_pool_mult)}×{int(sz)}).")
                continue
            roi_stats = simulate_accumulator_roi(candidates, 'clean_sheet', sz)
            if np.isfinite(roi_stats.get("mean_roi", np.nan)):
                print(f"⚡ Sim {sz}-fold CLEAN_SHEET ROI: {roi_stats['mean_roi']*100: .2f}%  "
                      f"[95% CI: {roi_stats['ci_low']*100: .1f} … {roi_stats['ci_high']*100: .1f}]  "
                      f"σ={roi_stats['std_roi']*100: .1f}%  VaR5={roi_stats['var95']*100: .1f}%")
                _write_roi_csv(league_name, f"clean_sheet_{sz}x", roi_stats if isinstance(roi_stats, dict) else roi_stats.to_dict())
            else:
                if not config.get("quiet_skips", False):
                    _log("INFO", f"⏭️ Skipped {sz}-fold CLEAN_SHEET sim (insufficient pool or capped).")

        # Walk-forward P&L
        for _kelly in (False, True):
            # normalize stake to float|None for typing
            _stake_val = config.get("stake_per_acca", 100)
            try:
                stake_f: float | None = None if _stake_val is None else _to_float(_stake_val)
            except Exception:
                stake_f = None if _stake_val is None else _to_float(_stake_val)

            pnl = simulate_walk_forward_pnl(
                candidates, 'clean_sheet',
                stake=stake_f,
                kelly=_kelly
            )
            if pnl:
                tag = "½-Kelly" if _kelly else "flat"
                print(f"📈 Walk-forward CLEAN_SHEET ROI ({tag}): "
                      f"{pnl['final_roi']*100: .1f}% | Max DD {pnl['max_dd']*100: .1f}%")
                _write_pnl_csv(league_name, 'clean_sheet', pnl)

        # Accuracy/watchlist shortlist
        acc_conf_col = 'selected_cs_confidence'
        try:
            accuracy_df = df.copy()
            thr_eff = float(_resolve_min_conf('clean_sheet'))
            accuracy_df = accuracy_df[pd.to_numeric(accuracy_df.get(acc_conf_col), errors='coerce') >= thr_eff].copy()
            accuracy_df = _ensure_draw_context_for_export(accuracy_df, league_name)
            accuracy_df['thr_effective'] = thr_eff

            if league_name:
                outdir_acc = os.path.join('predictions_output', datetime.datetime.utcnow().strftime('%Y-%m-%d'))
                os.makedirs(outdir_acc, exist_ok=True)
                outfile_acc = os.path.join(outdir_acc, f"{league_name.replace(' ', '_')}_top_clean_sheet_accuracy.csv")
                export_cols_acc = ['home_team_name','away_team_name','match_date', acc_conf_col, 'thr_effective']
                for _extra in ('p_draw','draw_flag','draw_flag_topk','thr_draw','mode_draw','alpha_used','best_k_pct','is_topk_best'):
                    if _extra in accuracy_df.columns and _extra not in export_cols_acc:
                        export_cols_acc.append(_extra)
                (accuracy_df
                    .sort_values(by=[acc_conf_col], ascending=False)
                    .head(top_n)
                    .to_csv(outfile_acc, index=False, columns=[c for c in export_cols_acc if c in accuracy_df.columns]))
                _log('INFO', f"📤 Saved top CLEAN_SHEET accuracy picks to: {outfile_acc}")
        except Exception as e:
            _log('WARN', f"⚠️ Accuracy export failed for CLEAN_SHEET: {e}")

        # Export high-EV picks
        if not candidates.empty and league_name:
            outdir = os.path.join("predictions_output", datetime.datetime.utcnow().strftime("%Y-%m-%d"))
            os.makedirs(outdir, exist_ok=True)
            outfile = os.path.join(outdir, f"{league_name.replace(' ', '_')}_top_{target_type}_picks.csv")

            thr_draw_attr = df.attrs.get("thr_draw", None)
            mode_draw_attr = df.attrs.get("mode_draw", None)
            if thr_draw_attr is not None:
                candidates["thr_draw"] = float(thr_draw_attr)
            if mode_draw_attr is not None:
                candidates["mode_draw"] = str(mode_draw_attr)

            conf_col = 'selected_cs_confidence'
            odds_col = 'selected_cs_odds'
            candidates["thr_effective"] = float(min_conf)

            export_cols = ['home_team_name', 'away_team_name', 'match_date',
                           'expected_value', 'p_model', 'odds', 'edge', 'kelly_frac',
                           conf_col, odds_col, 'thr_effective']
            for _extra in ("p_draw", "draw_flag", "draw_flag_topk", "thr_draw", "mode_draw"):
                if _extra in candidates.columns and _extra not in export_cols:
                    export_cols.append(_extra)
            if "draw_flag_topk" not in candidates.columns:
                candidates["draw_flag_topk"] = 0
            if "is_topk_best" not in candidates.columns:
                candidates["is_topk_best"] = candidates["draw_flag_topk"].astype(int)
            for _must in ("draw_flag_topk", "is_topk_best"):
                if _must not in export_cols:
                    export_cols.append(_must)

            _alpha_used = globals().get("BASELINE_ALPHA_USED", None)
            _best_k_pct = globals().get("BEST_K_PCT", df.attrs.get("topk_pct_draw", None))
            if _alpha_used is not None:
                candidates["alpha_used"] = float(_alpha_used)
                if "alpha_used" not in export_cols:
                    export_cols.append("alpha_used")
            if _best_k_pct is not None:
                candidates["best_k_pct"] = float(_best_k_pct)
                if "best_k_pct" not in export_cols:
                    export_cols.append("best_k_pct")

            candidates.to_csv(outfile, index=False, columns=[c for c in export_cols if c in candidates.columns])
            print(f"📤 Saved top {target_type.upper()} picks to: {outfile}")

            # Cross-league append
            try:
                global_path = os.path.join(
                    "predictions_output",
                    datetime.datetime.utcnow().strftime("%Y-%m-%d"),
                    f"ALL_LEAGUES_top_{target_type}_picks.csv",
                )
                use_extended = _inspect_and_upgrade_global_csv(global_path)
                header_needed = not os.path.exists(global_path)

                with open(global_path, "a") as fh:
                    if use_extended:
                        if header_needed:
                            fh.write(
                                "league,home_team_name,away_team_name,match_date,expected_value,confidence,odds,thr_effective,"
                                "p_draw,draw_flag,draw_flag_topk,thr_draw,mode_draw,alpha_used,best_k_pct,is_topk_best\n"
                            )
                        for _, row in candidates.iterrows():
                            team_home  = str(row.get("home_team_name", ""))
                            team_away  = str(row.get("away_team_name", ""))
                            match_date = _coalesce_match_date_row(row)
                            ev   = float(row.get("expected_value", np.nan))
                            conf = float(row.get(conf_col, np.nan))
                            odds = float(row.get(odds_col, np.nan))
                            thr_effective = float(row.get("thr_effective", min_conf))
                            pdraw_val = row.get("p_draw", None)
                            if (pdraw_val is None) or (str(pdraw_val) == "") or (pd.isna(pdraw_val)):
                                pdraw_val = row.get("confidence_draw", "")
                            p_draw = pdraw_val
                            dflag  = row.get("draw_flag", "")
                            dtopk  = row.get("draw_flag_topk", "")
                            thr    = row.get("thr_draw", candidates.attrs.get("thr_draw", ""))
                            mode   = row.get("mode_draw", candidates.attrs.get("mode_draw", ""))
                            alpha_used = row.get("alpha_used", globals().get("BASELINE_ALPHA_USED", ""))
                            best_k_pct = row.get("best_k_pct", globals().get("BEST_K_PCT", ""))
                            is_topk    = row.get("is_topk_best", row.get("draw_flag_topk", ""))
                            fh.write(
                                f"{league_name},{team_home},{team_away},{match_date},"
                                f"{ev:.4f},{conf:.4f},{odds:.4f},{thr_effective},"
                                f"{p_draw},{dflag},{dtopk},{thr},{mode},{alpha_used},{best_k_pct},{is_topk}\n"
                            )
                    else:
                        if header_needed:
                            fh.write("league,home_team_name,away_team_name,match_date,expected_value,confidence,odds\n")
                        for _, row in candidates.iterrows():
                            team_home  = str(row.get("home_team_name", ""))
                            team_away  = str(row.get("away_team_name", ""))
                            match_date = _coalesce_match_date_row(row)
                            ev   = float(row.get("expected_value", np.nan))
                            conf = float(row.get(conf_col, np.nan))
                            odds = float(row.get(odds_col, np.nan))
                            fh.write(
                                f"{league_name},{team_home},{team_away},{match_date},{ev:.4f},{conf:.4f},{odds:.4f}\n"
                            )
            except Exception as e:
                print(f"⚠️ Could not append to cross-league {target_type} CSV: {e}")

        # Final return depends on objective
        if objective == 'accuracy' or (probs_from_odds and objective != 'hybrid'):
            return (
                accuracy_df
                .sort_values(by=[acc_conf_col], ascending=False)
                .head(top_n)
            )
        elif objective == 'hybrid':
            sort_cols = []
            if sort_key in ('edge','ev') and 'edge' in candidates.columns:
                sort_cols.append('edge')
            if 'expected_value' in candidates.columns:
                sort_cols.append('expected_value')
            if acc_conf_col in candidates.columns:
                sort_cols.append(acc_conf_col)
            sort_cols = sort_cols or [acc_conf_col]
            return candidates.sort_values(by=sort_cols, ascending=False).head(top_n)
        else:
            return (
                candidates
                .sort_values(by=['expected_value', acc_conf_col], ascending=False)
                .head(top_n)
            )
        
    elif target_type == 'ftr':
        # Required probability columns
        prob_cols = ['confidence_home', 'confidence_draw', 'confidence_away']
        missing_probs = [c for c in prob_cols if c not in df.columns]
        if missing_probs:
            _disable_market('ftr', f"missing probability columns: {missing_probs}")
            _log("WARN", f"⚠️ Missing probability cols for FTR accumulator: {missing_probs}; skipping.")
            return pd.DataFrame()

        # Resolve bookmaker odds column names; synth if missing
        odds_home_col = _resolve_odds(df, ODD_COLUMN_WHITELIST['home_win'], label='FTR HOME odds')
        odds_draw_col = _resolve_odds(df, ODD_COLUMN_WHITELIST['draw'],     label='FTR DRAW odds')
        odds_away_col = _resolve_odds(df, ODD_COLUMN_WHITELIST['away_win'], label='FTR AWAY odds')
        if odds_home_col is None or odds_draw_col is None or odds_away_col is None:
            if forbid_synth:
                _disable_market('ftr', 'missing FTR odds and synthesis disabled (probs_from_odds & allow_synth_odds=False)')
                return pd.DataFrame()
            _log("INFO", "ℹ️ FTR odds missing → will use synthetic selected_odds = 1/p for EV ranking.")
            odds_home_col = odds_home_col or 'synth_home_ftr_odds'
            odds_draw_col = odds_draw_col or 'synth_draw_ftr_odds'
            odds_away_col = odds_away_col or 'synth_away_ftr_odds'
            df[odds_home_col] = 1.0 / pd.to_numeric(df['confidence_home'], errors='coerce').clip(lower=1e-9)
            df[odds_draw_col] = 1.0 / pd.to_numeric(df['confidence_draw'], errors='coerce').clip(lower=1e-9)
            df[odds_away_col] = 1.0 / pd.to_numeric(df['confidence_away'], errors='coerce').clip(lower=1e-9)

        outcome_cols = [
            ('confidence_home', odds_home_col),
            ('confidence_draw', odds_draw_col),
            ('confidence_away', odds_away_col)
        ]

        def _row_select(row):
            probs = [row[c[0]] for c in outcome_cols]
            odds  = [row[c[1]] for c in outcome_cols]
            if any(pd.isna(probs)) or any(pd.isna(odds)):
                return pd.Series([np.nan, np.nan, np.nan])
            pred = int(np.argmax(probs))
            return pd.Series([probs[pred], odds[pred], pred])

        df[['selected_confidence', 'selected_odds', 'ftr_pred_outcome']] = df.apply(_row_select, axis=1)

        # EV/edge/Kelly attach
        margin = float(df.attrs.get('book_margin', 0.0))
        probs_from_odds = bool(df.attrs.get('probs_from_odds', False))
        synth_odds = any(str(c).startswith('synth_') for c in (odds_home_col, odds_draw_col, odds_away_col))

        candidates = _attach_ev_columns(
            df.copy(),
            'selected_confidence',
            'selected_odds',
            margin=margin,
            probs_from_odds=probs_from_odds,
            synth_odds=synth_odds,
        )

        # Confidence + EV gate
        min_conf = _resolve_min_conf('ftr')
        ev_cushion = 0.02 if (probs_from_odds or synth_odds) else 0.0
        candidates = candidates[(candidates['selected_confidence'] >= min_conf) &
                                (candidates['expected_value'] > -ev_cushion)].copy()

        # ROI sims
        min_pool_mult = int(config.get("min_pool_x", 5))
        max_acca_size = config.get("max_acca_size", None)
        for sz in config.get("acca_sizes", [10]):
            if isinstance(max_acca_size, (int, float)) and max_acca_size is not None and sz > int(max_acca_size):
                if not config.get("quiet_skips", False):
                    _log("INFO", f"⏭️ Skipped {sz}-fold FTR sim (capped by max_acca_size={int(max_acca_size)}).")
                continue
            required = max(3, int(min_pool_mult) * int(sz))
            if len(candidates) < required:
                if not config.get("quiet_skips", False):
                    _log("INFO", f"⏭️ Skipped {sz}-fold FTR sim (pool {len(candidates)} < {int(min_pool_mult)}×{int(sz)}).")
                continue
            roi_stats = simulate_accumulator_roi(candidates, 'ftr', sz)
            if np.isfinite(roi_stats.get("mean_roi", np.nan)):
                print(f"⚡ Sim {sz}-fold FTR ROI: "
                      f"{roi_stats['mean_roi']*100: .2f}%  "
                      f"[95% CI: {roi_stats['ci_low']*100: .1f} … {roi_stats['ci_high']*100: .1f}]  "
                      f"σ={roi_stats['std_roi']*100: .1f}%  VaR5={roi_stats['var95']*100: .1f}%")
                _write_roi_csv(league_name, f"ftr_{sz}x", roi_stats if isinstance(roi_stats, dict) else roi_stats.to_dict())
            else:
                if not config.get("quiet_skips", False):
                    _log("INFO", f"⏭️ Skipped {sz}-fold FTR sim (insufficient pool or capped).")

        # Walk-forward P&L
        for _kelly in (False, True):
            # normalize stake to float|None for typing
            _stake_val = config.get("stake_per_acca", 100)
            try:
                stake_f: float | None = None if _stake_val is None else _to_float(_stake_val)
            except Exception:
                stake_f = None if _stake_val is None else _to_float(_stake_val)

            pnl = simulate_walk_forward_pnl(candidates, 'ftr',
                                            stake=stake_f,
                                            kelly=_kelly)
            if pnl:
                tag = "½-Kelly" if _kelly else "flat"
                print(f"📈 Walk-forward FTR ROI ({tag}): "
                      f"{pnl['final_roi']*100: .1f}% | Max DD {pnl['max_dd']*100: .1f}%")
                _write_pnl_csv(league_name, 'ftr', pnl)

        # Export top FTR EV picks
        try:
            if not candidates.empty and league_name:
                outdir = os.path.join("predictions_output", datetime.datetime.utcnow().strftime("%Y-%m-%d"))
                os.makedirs(outdir, exist_ok=True)
                outfile = os.path.join(outdir, f"{league_name.replace(' ', '_')}_top_ftr_picks.csv")

                thr_draw_attr = df.attrs.get("thr_draw", None)
                mode_draw_attr = df.attrs.get("mode_draw", None)
                if thr_draw_attr is not None:
                    candidates["thr_draw"] = float(thr_draw_attr)
                if mode_draw_attr is not None:
                    candidates["mode_draw"] = str(mode_draw_attr)

                export_cols = ['home_team_name','away_team_name','match_date','expected_value',
                               'p_model','odds','edge','kelly_frac',
                               'selected_confidence','selected_odds','ftr_pred_outcome','thr_effective']
                for _extra in ("p_draw","draw_flag","draw_flag_topk","thr_draw","mode_draw"):
                    if _extra in candidates.columns and _extra not in export_cols:
                        export_cols.append(_extra)
                if "draw_flag_topk" not in candidates.columns:
                    candidates["draw_flag_topk"] = 0
                if "is_topk_best" not in candidates.columns:
                    candidates["is_topk_best"] = candidates["draw_flag_topk"].astype(int)
                for _must in ("draw_flag_topk","is_topk_best"):
                    if _must not in export_cols:
                        export_cols.append(_must)

                _alpha_used = globals().get("BASELINE_ALPHA_USED", None)
                _best_k_pct = globals().get("BEST_K_PCT", df.attrs.get("topk_pct_draw", None))
                if _alpha_used is not None:
                    candidates["alpha_used"] = float(_alpha_used)
                    if "alpha_used" not in export_cols:
                        export_cols.append("alpha_used")
                if _best_k_pct is not None:
                    candidates["best_k_pct"] = float(_best_k_pct)
                    if "best_k_pct" not in export_cols:
                        export_cols.append("best_k_pct")

                candidates["thr_effective"] = float(min_conf)
                candidates.to_csv(outfile, index=False, columns=[c for c in export_cols if c in candidates.columns])
                print(f"📤 Saved top FTR picks to: {outfile}")

                # Cross-league append
                try:
                    global_path = os.path.join(
                        "predictions_output",
                        datetime.datetime.utcnow().strftime("%Y-%m-%d"),
                        "ALL_LEAGUES_top_ftr_picks.csv",
                    )
                    use_extended = _inspect_and_upgrade_global_csv(global_path)
                    header_needed = not os.path.exists(global_path)
                    with open(global_path, "a") as fh:
                        if use_extended and header_needed:
                            fh.write(
                                "league,home_team_name,away_team_name,match_date,expected_value,confidence,odds,thr_effective,"
                                "p_draw,draw_flag,draw_flag_topk,thr_draw,mode_draw,alpha_used,best_k_pct,is_topk_best\n"
                            )
                        if use_extended:
                            for _, row in candidates.iterrows():
                                team_home  = str(row.get("home_team_name", ""))
                                team_away  = str(row.get("away_team_name", ""))
                                match_date = _coalesce_match_date_row(row)
                                ev   = float(row.get("expected_value", np.nan))
                                conf = float(row.get("selected_confidence", np.nan))
                                odds = float(row.get("selected_odds", np.nan))
                                thr_effective = float(row.get("thr_effective", min_conf))
                                p_draw = row.get("p_draw", "")
                                dflag  = row.get("draw_flag", "")
                                dtopk  = row.get("draw_flag_topk", "")
                                thr    = row.get("thr_draw", candidates.attrs.get("thr_draw", ""))
                                mode   = row.get("mode_draw", candidates.attrs.get("mode_draw", ""))
                                alpha_used = row.get("alpha_used", globals().get("BASELINE_ALPHA_USED", ""))
                                best_k_pct = row.get("best_k_pct", globals().get("BEST_K_PCT", ""))
                                is_topk    = row.get("is_topk_best", row.get("draw_flag_topk", ""))
                                fh.write(
                                    f"{league_name},{team_home},{team_away},{match_date},"
                                    f"{ev:.4f},{conf:.4f},{odds:.4f},{thr_effective},"
                                    f"{p_draw},{dflag},{dtopk},{thr},{mode},{alpha_used},{best_k_pct},{is_topk}\n"
                                )
                        else:
                            if header_needed:
                                fh.write("league,home_team_name,away_team_name,match_date,expected_value,confidence,odds\n")
                            for _, row in candidates.iterrows():
                                team_home  = str(row.get("home_team_name", ""))
                                team_away  = str(row.get("away_team_name", ""))
                                match_date = _coalesce_match_date_row(row)
                                ev   = float(row.get("expected_value", np.nan))
                                conf = float(row.get("selected_confidence", np.nan))
                                odds = float(row.get("selected_odds", np.nan))
                                fh.write(
                                    f"{league_name},{team_home},{team_away},{match_date},{ev:.4f},{conf:.4f},{odds:.4f}\n"
                                )
                except Exception as e:
                    print(f"⚠️ Could not append to cross-league FTR CSV: {e}")
        except Exception as e:
            print(f"⚠️ FTR export failed: {e}")

        # Accuracy/watchlist shortlist (FTR)
        try:
            acc_conf_col = "selected_confidence"
            accuracy_df = df.loc[pd.to_numeric(df.get(acc_conf_col), errors="coerce") >= float(min_conf)].copy()
            accuracy_df = _ensure_draw_context_for_export(accuracy_df, league_name)
            accuracy_df["thr_effective"] = float(min_conf)
            if league_name:
                outdir_acc = os.path.join("predictions_output", datetime.datetime.utcnow().strftime("%Y-%m-%d"))
                os.makedirs(outdir_acc, exist_ok=True)
                outfile_acc = os.path.join(outdir_acc, f"{league_name.replace(' ', '_')}_top_ftr_accuracy.csv")
                export_cols_acc = ['home_team_name','away_team_name','match_date', acc_conf_col, 'thr_effective']
                for _extra in ("p_draw","draw_flag","draw_flag_topk","thr_draw","mode_draw","alpha_used","best_k_pct","is_topk_best"):
                    if _extra in accuracy_df.columns and _extra not in export_cols_acc:
                        export_cols_acc.append(_extra)
                (accuracy_df
                    .sort_values(by=[acc_conf_col], ascending=False)
                    .head(top_n)
                    .to_csv(outfile_acc, index=False, columns=[c for c in export_cols_acc if c in accuracy_df.columns]))
                _log("INFO", f"📤 Saved top FTR accuracy picks to: {outfile_acc}")
        except Exception as e:
            _log("WARN", f"⚠️ Accuracy export failed for FTR: {e}")

        # Final return depends on objective
        if objective == "accuracy" or (probs_from_odds and objective != "hybrid"):
            return accuracy_df.sort_values(by=[acc_conf_col], ascending=False).head(top_n)
        elif objective == "hybrid":
            sort_cols = []
            if sort_key in ("edge","ev") and "edge" in candidates.columns:
                sort_cols.append("edge")
            if "expected_value" in candidates.columns:
                sort_cols.append("expected_value")
            if acc_conf_col in candidates.columns:
                sort_cols.append(acc_conf_col)
            sort_cols = sort_cols or [acc_conf_col]
            return candidates.sort_values(by=sort_cols, ascending=False).head(top_n)
        else:
            return candidates

# --------------------------------------------------------------
# Correct score shortlist using independent Poisson with FTS boost
# --------------------------------------------------------------

def generate_correct_score_candidates(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Generate top-N correct-score candidates using Poisson lambdas per match."""
    import numpy as _np
    import pandas as _pd
    from math import factorial

    def _poisson_pmf(k, lam):
        try:
            k = int(k)
            lam = float(lam)
            if lam < 0:
                return _np.nan
            return _np.exp(-lam) * (lam ** k) / factorial(k)
        except Exception:
            return _np.nan

    if df is None or df.empty:
        return _pd.DataFrame(columns=[
            "row_idx", "home_team_name", "away_team_name", "match_date",
            "scoreline", "p_cs", "cs_odds", "ev_cs"
        ])

    df2 = df.copy()
    if "home_goals_pred" not in df2.columns or "away_goals_pred" not in df2.columns:
        return _pd.DataFrame(columns=[
            "row_idx", "home_team_name", "away_team_name", "match_date",
            "scoreline", "p_cs", "cs_odds", "ev_cs"
        ])

    home_fts_col = next((c for c in ("p_home_fts", "prob_home_fts", "home_fts_confidence") if c in df2.columns), None)
    away_fts_col = next((c for c in ("p_away_fts", "prob_away_fts", "away_fts_confidence") if c in df2.columns), None)
    fts_boost = (home_fts_col is not None) or (away_fts_col is not None)

    rows = []
    for idx, row in df2.iterrows():
        lam_h = float(_pd.to_numeric(row.get("home_goals_pred", _np.nan), errors="coerce"))
        lam_a = float(_pd.to_numeric(row.get("away_goals_pred", _np.nan), errors="coerce"))
        if _np.isnan(lam_h) or _np.isnan(lam_a):
            continue

        p_home_fts = float(_pd.to_numeric(row.get(home_fts_col, _np.nan), errors="coerce")) if home_fts_col else _np.nan
        p_away_fts = float(_pd.to_numeric(row.get(away_fts_col, _np.nan), errors="coerce")) if away_fts_col else _np.nan

        for h in range(0, 5):
            for a in range(0, 5):
                p_h = _poisson_pmf(h, lam_h)
                p_a = _poisson_pmf(a, lam_a)
                if _np.isnan(p_h) or _np.isnan(p_a):
                    continue
                p_cs = p_h * p_a
                if fts_boost:
                    if h == 0 and not _np.isnan(p_away_fts):
                        p_cs = (p_cs * p_away_fts) ** 0.5
                    if a == 0 and not _np.isnan(p_home_fts):
                        p_cs = (p_cs * p_home_fts) ** 0.5
                rows.append({
                    "row_idx": idx,
                    "home_team_name": row.get("home_team_name", ""),
                    "away_team_name": row.get("away_team_name", ""),
                    "match_date": _coalesce_match_date_row(row),
                    "scoreline": f"{h}-{a}",
                    "p_cs": float(max(0.0, min(1.0, p_cs))),
                    "cs_odds": _np.nan,
                    "ev_cs": _np.nan,
                })

    cand = _pd.DataFrame(rows)
    if cand.empty:
        return cand
    return cand.sort_values("p_cs", ascending=False).head(int(top_n)).reset_index(drop=True)
# --------------------------------------------------------------
# Coherence checks between related markets
# --------------------------------------------------------------

def validate_market_coherence(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Run lightweight checks: Over/Under add to ~1, BTTS vs clean-sheet consistency, etc.
    Marks boolean columns like 'incoherent_over25' when discrepancies > 0.15.
    """
    tol = 0.15
    try:
        if {'adjusted_over25_confidence','under25_confidence'}.issubset(df.columns):
            incoh = (df['adjusted_over25_confidence'] + df['under25_confidence'] - 1.0).abs() > tol
            df['incoherent_over25'] = incoh.astype(int)
            if verbose and bool(np.any(incoh)):
                n = int(incoh.sum())
                print(f"⚠️ {n} rows show Over/Under incoherence > {tol}")
        if {'adjusted_btts_confidence','btts_no_confidence'}.issubset(df.columns):
            incoh = (df['adjusted_btts_confidence'] + df['btts_no_confidence'] - 1.0).abs() > tol
            df['incoherent_btts'] = incoh.astype(int)
        # Clean sheet vs BTTS sanity: if BTTS YES is very high, clean sheets should be low
        if 'adjusted_btts_confidence' in df.columns:
            for col in ('p_home_win_to_nil','p_home_cs','p_away_cs','p_away_win_to_nil'):
                if col in df.columns:
                    df[f'incoherent_{col}'] = ((df['adjusted_btts_confidence'] > 0.75) & (df[col] > 0.40)).astype(int)
    except Exception as e:
        if verbose:
            print(f"ℹ️ Coherence validation skipped: {e}")
    return df

# --------------------------------------------------------------
# Walk‑forward P&L simulation (chronological equity curve)
# --------------------------------------------------------------
def simulate_walk_forward_pnl(candidates: pd.DataFrame,
                              market: str,
                              stake: float = 100.0,
                              kelly: bool = False,
                              kelly_frac: float | None = None,
                              max_pct: float | None = None) -> dict:
    """
    Simulates placing a single‑selection bet on *each* candidate row
    in chronological order. Bankroll starts at 10 000.
    Flat stake or Kelly fraction.

    Returns dict with final ROI, max drawdown, and equity curve arrays.
    """
    if candidates.empty:
        return {}

    # Sort by fixture date if present else keep current order
    date_col = next((c for c in candidates.columns if "date" in c.lower()), None)
    if date_col:
        candidates = candidates.sort_values(date_col)

    # fall back to global settings if not supplied
    if kelly_frac is None:
        kelly_frac = config.get("kelly_frac", 0.5)
    if max_pct is None:
        max_pct = config.get("max_kelly_pct", 0.05)

    bankroll = 10_000.0
    peak     = bankroll
    max_dd   = 0.0
    curve    = []

    conf_map = {
        'btts': 'adjusted_btts_confidence',
        'btts_no': 'btts_no_confidence',
        'over25': 'adjusted_over25_confidence',
        'under25': 'under25_confidence',
        'ftr': 'selected_confidence',
        'wtn': 'selected_wtn_confidence',
        'clean_sheet': 'selected_cs_confidence',
    }
    odds_map = {
        'btts': 'odds_btts_yes',
        'btts_no': 'odds_btts_no',
        'over25': 'odds_ft_over25',
        'under25': 'odds_ft_under25',
        'ftr': 'selected_odds',
        'wtn': 'selected_wtn_odds',
        'clean_sheet': 'selected_cs_odds',
    }
    conf_col = conf_map[market]
    odds_col = odds_map[market]

    for _, row in candidates.iterrows():
        p   = row[conf_col]
        dec = row[odds_col]
        if pd.isna(p) or pd.isna(dec):
            continue
        fair_ev = p * dec - 1.0
        if fair_ev <= 0:
            continue  # skip negative EV

        if kelly:
            if p > 0 and dec > 1:
                k_frac = max(0.0, (p * (dec - 1) - (1 - p)) / (dec - 1))
                w = bankroll * k_frac * kelly_frac
                # never risk more than max_pct of bankroll
                w = min(w, bankroll * max_pct)
            else:
                continue  # Kelly stake would be zero – skip bet
        else:
            w = stake

        w = min(w, bankroll)  # can’t bet more than we have
        win = np.random.random() < p
        bankroll += w * (dec - 1) if win else -w
        peak = max(peak, bankroll)
        max_dd = min(max_dd, (bankroll - peak) / peak)
        curve.append(bankroll)

    final_roi = (bankroll - 10_000) / 10_000
    method = f"kelly{kelly_frac}" if kelly else "flat"
    return {
        "final_roi": final_roi,
        "max_dd": max_dd,
        "equity": curve,
        "method": method
    }


# --------------------------------------------------------------
# Legacy → extended schema upgrade for cross-league picks CSV
# --------------------------------------------------------------
def _inspect_and_upgrade_global_csv(global_path: str):
    """Detect whether the cross-league CSV should use the extended header
    (draw context + thr_effective) and upgrade the existing file in place
    if any of those columns are missing."""
    import csv as _csv
    import os as _os
    import tempfile as _tempfile
    import shutil as _shutil

    # Canonical extended header (now includes thr_effective)
    extended = [
        "league","home_team_name","away_team_name","match_date","expected_value",
        "confidence","odds","thr_effective","p_draw","draw_flag","draw_flag_topk","thr_draw",
        "mode_draw","alpha_used","best_k_pct","is_topk_best"
    ]

    # Brand-new file → we plan to write the extended header
    if not _os.path.exists(global_path):
        return True

    # Read current header
    try:
        with open(global_path, "r", newline="") as fh:
            reader = _csv.reader(fh)
            header = next(reader, [])
    except Exception:
        header = []

    header_set   = set(header)
    extended_set = set(extended)

    # Need upgrade if thr_effective (or any extended col) missing, or order differs
    missing_thr_effective = "thr_effective" not in header_set
    missing_any_required  = not extended_set.issubset(header_set)
    needs_upgrade         = missing_thr_effective or missing_any_required or (header != extended)

    if not needs_upgrade:
        return True

    # Rewrite to extended schema
    try:
        with open(global_path, "r", newline="") as fh_in:
            rdict = _csv.DictReader(fh_in)
            fd, tmp_path = _tempfile.mkstemp(prefix="_upgrade_", suffix=".csv")
        _os.close(fd)

        with open(tmp_path, "w", newline="") as fh_out:
            writer = _csv.DictWriter(fh_out, fieldnames=extended)
            writer.writeheader()
            for r in rdict:
                writer.writerow({
                    "league": r.get("league", ""),
                    "home_team_name": r.get("home_team_name", ""),
                    "away_team_name": r.get("away_team_name", ""),
                    "match_date": r.get("match_date", ""),
                    "expected_value": r.get("expected_value", ""),
                    "confidence": r.get("confidence", ""),
                    "odds": r.get("odds", ""),
                    # legacy rows can’t recover this: leave blank
                    "thr_effective": "",
                    # best-effort carry-through for draw context if present
                    "p_draw": r.get("p_draw", ""),
                    "draw_flag": r.get("draw_flag", ""),
                    "draw_flag_topk": r.get("draw_flag_topk", ""),
                    "thr_draw": r.get("thr_draw", ""),
                    "mode_draw": r.get("mode_draw", ""),
                    "alpha_used": r.get("alpha_used", ""),
                    "best_k_pct": r.get("best_k_pct", ""),
                    # prefer explicit column; fall back to old top-k flag
                    "is_topk_best": r.get("is_topk_best", r.get("draw_flag_topk", "")),
                })

        _shutil.move(tmp_path, global_path)
        try:
            print(f"🛠️ Upgraded global CSV to extended header (added thr_effective): {global_path}")
        except Exception:
            pass
    except Exception as e:
        try:
            print(f"⚠️ Failed to upgrade global CSV {global_path}: {e}")
        except Exception:
            pass

    return True
# Re‑export for pipeline import convenience
__all__ += [
    "_apply_lambda_shrink",
    "generate_btts_and_over_preds",
    "adjust_with_volatility_modifiers",
    "log_prediction_changes",
    "generate_accumulator_recommendations",
    "apply_draw_threshold_flag",
    "mark_topk_draws",
    "prepare_league_for_inference",
    "BEST_K_PCT",
    "BASELINE_ALPHA_USED",
    "attach_win_to_nil_proxy",
    "generate_correct_score_candidates",
    "validate_market_coherence",
]

# Hint for linters/tests: we reference the helpers once so they’re not flagged
_unused_exports = (
    generate_btts_and_over_preds,
    adjust_with_volatility_modifiers,
    log_prediction_changes,
    generate_accumulator_recommendations,
)
# Module-scope EV attachment removed; handled inside FTR branch of generate_accumulator_recommendations()
# --------------------------------------------------------------
# Minimal smoke tests for EV/sorting paths (manual runner)
# --------------------------------------------------------------

def _make_dummy_df_for_smoke(case: str = "modelled") -> pd.DataFrame:
    """
    Build a tiny dataframe with the minimal columns needed to exercise
    BTTS, Over2.5 and FTR accumulator flows.

    case = "modelled"  → has explicit probabilities + preds + odds
    case = "odds"      → has odds only; probabilities are inferred
    """
    rows = [
        {
            "home_team_name": "Alpha FC", "away_team_name": "Bravo",
            "match_date": "2025-08-16",
            # Model-style confidences (only used in 'modelled' case)
            "adjusted_btts_confidence": 0.65, "btts_pred": 1,
            "adjusted_over25_confidence": 0.62, "over25_pred": 1,
            # FTR probabilities (only used in 'modelled' case)
            "confidence_home": 0.50, "confidence_draw": 0.24, "confidence_away": 0.26,
            # Odds (used in both cases)
            "odds_btts_yes": 1.91, "odds_ft_over25": 1.80,
            "odds_ft_home_team_win": 2.20, "odds_ft_draw": 3.60, "odds_ft_away_team_win": 3.40,
        },
        {
            "home_team_name": "Charlie", "away_team_name": "Delta",
            "match_date": "2025-08-16",
            "adjusted_btts_confidence": 0.58, "btts_pred": 1,
            "adjusted_over25_confidence": 0.55, "over25_pred": 1,
            "confidence_home": 0.44, "confidence_draw": 0.30, "confidence_away": 0.26,
            "odds_btts_yes": 2.05, "odds_ft_over25": 1.95,
            "odds_ft_home_team_win": 2.40, "odds_ft_draw": 3.20, "odds_ft_away_team_win": 3.10,
        },
        {
            "home_team_name": "Echo", "away_team_name": "Foxtrot",
            "match_date": "2025-08-17",
            "adjusted_btts_confidence": 0.70, "btts_pred": 1,
            "adjusted_over25_confidence": 0.48, "over25_pred": 0,
            "confidence_home": 0.39, "confidence_draw": 0.30, "confidence_away": 0.31,
            "odds_btts_yes": 1.80, "odds_ft_over25": 2.10,
            "odds_ft_home_team_win": 2.80, "odds_ft_draw": 3.10, "odds_ft_away_team_win": 2.75,
        },
    ]
    df = pd.DataFrame(rows)

    if case == "odds":
        # Strip out explicit probability columns to force odds→probs path
        for c in ("adjusted_btts_confidence","btts_pred",
                  "adjusted_over25_confidence","over25_pred",
                  "confidence_home","confidence_draw","confidence_away"):
            if c in df.columns:
                df.drop(columns=[c], inplace=True)
    return df


def _assert_has_cols(name: str, frame: pd.DataFrame, cols: set[str]) -> None:
    missing = cols - set(frame.columns)
    if missing:
        raise AssertionError(f"{name}: missing expected column(s): {sorted(missing)}")


def run_overlay_smoke_tests(league_name: str = "England Premier League") -> dict:
    """
    Quick, deterministic smoke tests for:
      • BTTS accumulator (EV columns + non-empty)
      • Over2.5 accumulator (EV columns + non-empty)
      • FTR accumulator (EV columns + non-empty)
      • Odds-derived path (infer_probs_from_odds)
    Returns a small dict of counts so CI/logs can assert quickly.
    """
    # Save + tweak overlay knobs for the test
    _bak = dict(overlay_config)
    overlay_config.update({"objective": "hybrid", "sort_key": "ev", "allow_synth_odds": False})

    try:
        required_ev_cols = {"expected_value", "p_model", "odds", "edge", "kelly_frac"}

        # ── Case 1: model-style inputs (explicit probabilities present) ──
        df_model = _make_dummy_df_for_smoke("modelled")
        # Ensure any draw context is available for exports (safe no-op if constants absent)
        df_model = _ensure_draw_context_for_export(df_model, league_name)

        btts = generate_accumulator_recommendations(df_model.copy(), "btts", top_n=5, league_name=league_name)
        over = generate_accumulator_recommendations(df_model.copy(), "over25", top_n=5, league_name=league_name)
        ftr  = generate_accumulator_recommendations(df_model.copy(), "ftr", top_n=5, league_name=league_name)

        if btts.empty or over.empty or ftr.empty:
            raise AssertionError("Modelled-path candidates unexpectedly empty (btts/over/ftr).")

        _assert_has_cols("BTTS(modelled)", btts, required_ev_cols)
        _assert_has_cols("Over25(modelled)", over, required_ev_cols)
        _assert_has_cols("FTR(modelled)", ftr, required_ev_cols)

        # ── Case 2: odds-only inputs (probabilities inferred) ──
        df_odds = _make_dummy_df_for_smoke("odds")
        df_odds = infer_probs_from_odds(df_odds)   # tags df.attrs['probs_from_odds']=True
        df_odds = _ensure_draw_context_for_export(df_odds, league_name)

        btts_o = generate_accumulator_recommendations(df_odds.copy(), "btts", top_n=5, league_name=league_name)
        over_o = generate_accumulator_recommendations(df_odds.copy(), "over25", top_n=5, league_name=league_name)
        ftr_o  = generate_accumulator_recommendations(df_odds.copy(), "ftr", top_n=5, league_name=league_name)

        if btts_o.empty or over_o.empty or ftr_o.empty:
            raise AssertionError("Odds-derived-path candidates unexpectedly empty (btts/over/ftr).")

        _assert_has_cols("BTTS(odds)", btts_o, required_ev_cols)
        _assert_has_cols("Over25(odds)", over_o, required_ev_cols)
        _assert_has_cols("FTR(odds)", ftr_o, required_ev_cols)

        # Lightweight return for logs
        result = {
            "btts_n": int(len(btts)),
            "over_n": int(len(over)),
            "ftr_n": int(len(ftr)),
            "btts_odds_n": int(len(btts_o)),
            "over_odds_n": int(len(over_o)),
            "ftr_odds_n": int(len(ftr_o)),
        }
        print("✅ overlay smoke tests passed:", result)
        return result
    finally:
        # Restore config
        overlay_config.clear()
        overlay_config.update(_bak)

# Ensure the helper is importable without touching the earlier __all__ block
try:
    __all__.append("run_overlay_smoke_tests")
except Exception:
    pass

if __name__ == "__main__":
    try:
        run_overlay_smoke_tests()
    except Exception as _e:
        print(f"❌ overlay smoke tests failed: {_e}")
        raise
    
    
    # === Draw model + FTR remix bridge (non-destructive) ===
try:
    import os, json, joblib
    import numpy as np
    import pandas as pd
except Exception:
    pass

# --- Per‑league draw temperature resolver -----------------------------------
if "_resolve_draw_temp" not in globals():
    def _resolve_draw_temp(league: object, default: float | None = None) -> float:
        """Resolve draw temperature T with a per‑league override.
        Uses DRAW_TEMP_<LEAGUE_TAG> if set; otherwise falls back to DRAW_TEMP; else 1.4.
        """
        import os, re
        try:
            base = float(os.getenv("DRAW_TEMP", str(1.4 if default is None else default)))
        except Exception:
            base = 1.4 if default is None else float(default)
        if not league:
            return base
        try:
            tag = re.sub(r"[^A-Za-z0-9_]+", "_", str(league).upper()).strip("_")
        except Exception:
            tag = ""
        try:
            return float(os.getenv(f"DRAW_TEMP_{tag}", str(base)))
        except Exception:
            return base

# bring constants with fallbacks
try:
    from constants import MODEL_DIR
except Exception:
    MODEL_DIR = "ModelStore"
try:
    from constants import HISTORICAL_DRAW_RATE, HISTORICAL_DRAW_RATE_DEFAULT
except Exception:
    HISTORICAL_DRAW_RATE, HISTORICAL_DRAW_RATE_DEFAULT = {}, 0.25
try:
    from constants import N0_SHRINK, N0_SHRINK_PER_LEAGUE
except Exception:
    N0_SHRINK, N0_SHRINK_PER_LEAGUE = 20, {}
try:
    from constants import LOCKED_FTR_MIX
except Exception:
    LOCKED_FTR_MIX = {}

from typing import List

def _coerce_float(s):
    import pandas as pd
    return pd.to_numeric(s, errors="coerce")

def _draw_json_path(league: str) -> str:
    slug = league.replace(" ", "_")
    return os.path.join(MODEL_DIR, f"{slug}_draw_threshold.json")

def _draw_model_path(league: str) -> str:
    slug = league.replace(" ", "_")
    # Prefer new per-league folder structure, fall back to legacy flat path
    p_new = os.path.join(MODEL_DIR, slug, "draw.pkl")
    p_legacy = os.path.join(MODEL_DIR, f"{slug}_draw_clf.pkl")
    return p_new if os.path.exists(p_new) else p_legacy

def _load_draw_thresholds(league: str) -> dict:
    try:
        with open(_draw_json_path(league), "r") as fh:
            return json.load(fh)
    except Exception:
        return {}

def _load_draw_model(league: str):
    p = _draw_model_path(league)
    if os.path.exists(p):
        try:
            return joblib.load(p)
        except Exception as e:
            print(f"⚠️ Could not load draw model for {league}: {e}")
    return None

def _ensure_feature_frame(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in features:
        if c in df.columns:
            X[c] = _coerce_float(df[c]).fillna(0.0)
        else:
            X[c] = 0.0
    return X.astype(float)

def _shrink_to_league_prior(p: pd.Series, league: str) -> pd.Series:
    prior = float(HISTORICAL_DRAW_RATE.get(league, HISTORICAL_DRAW_RATE_DEFAULT or 0.25))
    n0 = int(N0_SHRINK_PER_LEAGUE.get(league, N0_SHRINK or 20))
    return (p + n0 * prior) / (1.0 + n0)

def _logit(x):
    x = np.clip(x, 1e-9, 1-1e-9)
    return np.log(x/(1-x))

def _sigm(z):
    return 1.0/(1.0+np.exp(-z))

def apply_market_thresholds_to_attrs(df: pd.DataFrame, league: str) -> pd.DataFrame:
    slug = league.replace(" ", "_")
    p = os.path.join(MODEL_DIR, f"{slug}_market_thresholds.json")
    if not os.path.exists(p):
        return df
    try:
        with open(p,"r") as fh:
            payload = json.load(fh)
        mkts = payload.get("markets", {})
        for k, v in mkts.items():
            thr = v.get("threshold", None)
            if thr is not None:
                df.attrs[f"thr_{k}"] = float(thr)
    except Exception as e:
        print(f"⚠️ could not load thresholds: {e}")
    return df

# optional ETL alignment helper
try:
    from etl_press_intensity import ensure_press_intensity_on_disk
    def _ensure_press_for_inference(league: str, matches_dir: str):
        j = _load_draw_thresholds(league)
        alpha = float(j.get("baseline_blend", 0.70))
        try:
            ensure_press_intensity_on_disk(
                matches_dir,
                force=False,
                baseline_blend=alpha,
                use_cache=True,
                overwrite_baseline=False,
                overwrite_intensity=False
            )
        except TypeError:
            ensure_press_intensity_on_disk(matches_dir)
except Exception:
    def _ensure_press_for_inference(league: str, matches_dir: str):
        return  # silently skip if ETL not available

def attach_p_draw_and_mix_ftr(df: pd.DataFrame, league: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    bundle = _load_draw_model(league)
    if bundle is None:
        print("ℹ️ draw model missing → leaving FTR as-is")
        return df
    # Support both dict bundle and bare model
    if isinstance(bundle, dict):
        clf = bundle.get("model", None) or bundle.get("clf", None)
        feats = list(bundle.get("features", []))
    else:
        clf = bundle
        feats = []
    if clf is None:
        print("ℹ️ draw bundle incomplete → leaving FTR as-is")
        return df

    X = _ensure_feature_frame(df, feats) if feats else df.select_dtypes(include="number")
    try:
        p_draw_raw = clf.predict_proba(X)[:, 1]
    except Exception as e:
        print(f"⚠️ draw inference failed: {e}")
        return df

    df["p_draw_model_raw"] = _coerce_float(p_draw_raw).clip(0,1)
    # Per‑league temperature soften on raw draw proba (then shrink to league prior)
    try:
        T = _resolve_draw_temp(league)
    except Exception:
        T = 1.4
    try:
        import os as _os
        if _os.getenv("VERBOSE_QUICK","0") == "1":
            try:
                _tag = str(league).replace(" ", "_")
                print(f"🧪 draw temp T={T:.3f} ({_tag})")
            except Exception:
                pass
    except Exception:
        pass
    # temperature scaling in logit space
    try:
        _logit_p = _logit(df["p_draw_model_raw"])  # uses local helper
        _p_temp  = _sigm(_logit_p / float(T))       # uses local helper
    except Exception:
        _p_temp = df["p_draw_model_raw"]
    df["p_draw_model"] = _shrink_to_league_prior(_p_temp, league)

    j = _load_draw_thresholds(league)
    thr = float(j.get("threshold", 0.5))
    # support either explicit best_k_pct or precision_at_k map
    if "best_k_pct" in j:
        best_k = float(j.get("best_k_pct", 0.10))
    else:
        best_k = float(j.get("precision_at_k", {}).get("0.10", 0.10)) if isinstance(j, dict) else 0.10

    df["pred_draw_flag"] = (df["p_draw_model"] >= thr).astype(int)
    if len(df) > 0:
        k = max(1, int(len(df) * best_k))
        order = df["p_draw_model"].rank(method="first")
        df["is_topk_draw"] = (order >= (len(df) - k + 1)).astype(int)
    else:
        df["is_topk_draw"] = 0

    # base side confidences (keep your existing side signal)
    cand_home = [c for c in ("confidence_home","p_home","p_home_win","home_confidence") if c in df.columns]
    cand_away = [c for c in ("confidence_away","p_away","p_away_win","away_confidence") if c in df.columns]
    if not cand_home or not cand_away:
        print("ℹ️ base side confidence cols not found → leaving FTR unchanged")
        return df
    ph_base = _coerce_float(df[cand_home[0]]).clip(1e-6, 1-1e-6)
    pa_base = _coerce_float(df[cand_away[0]]).clip(1e-6, 1-1e-6)
    side_mass = (ph_base + pa_base).replace(0, 1e-6)
    S = (ph_base / side_mass).clip(1e-6, 1-1e-6)

    mix = LOCKED_FTR_MIX.get(league, {"alpha":1.0,"beta":0.10,"cap":0.45})
    alpha = float(mix.get("alpha",1.0))
    beta  = float(mix.get("beta",0.1))
    cap   = float(mix.get("cap",0.45))

    s_sharp = _sigm(alpha * _logit(S) + beta * (ph_base - pa_base))

    p_draw = np.minimum(df["p_draw_model"].values, cap)
    remain = np.maximum(0.0, 1.0 - p_draw)
    p_home_new = remain * s_sharp
    p_away_new = 1.0 - p_home_new - p_draw

    # write final confidences (preserving existing column names)
    df["confidence_home"] = np.clip(p_home_new, 0.0, 1.0)
    df["confidence_draw"] = np.clip(p_draw, 0.0, 1.0)
    df["confidence_away"] = np.clip(p_away_new, 0.0, 1.0)
    df["ftr_pred_outcome"] = np.vstack([
        df["confidence_home"].values,
        df["confidence_draw"].values,
        df["confidence_away"].values
    ]).T.argmax(axis=1)
    return df

# ---- Hard-disable legacy inline mixer name (use modern adaptive path) ----
def _DEPRECATED_attach_p_draw_and_mix_ftr(*args, **kwargs):
    raise RuntimeError(
        "Deprecated inline FTR mixer path. Use attach_ftr_with_draw_mix → predict_FTR_adaptive."
    )

# Rebind old symbol to the deprecated stub so accidental calls are loud


def generate_prediction_report(df, league_name, *args, **kwargs):
    """
    Build a robust predictions CSV per league.
    BTTS/Over2.5 labels prefer realized goals when available; otherwise they fall back
    to model probabilities using per-league thresholds (df.attrs or config defaults).
    Also annotates label sources: 'realized' vs 'model' to aid downstream QA.
    """
    import os
    import pandas as _pd

    # --- guard: ensure realized goal columns exist and are numeric (upcoming fixtures safe) ---
    import pandas as pd
    for _c in ("home_team_goal_count", "away_team_goal_count"):
        if _c not in df.columns:
            # create as zeros so boolean logic works without NA casting
            df[_c] = 0
        df[_c] = pd.to_numeric(df[_c], errors="coerce").fillna(0).astype(int)

    # Compute BTTS/Over25 directly from realized goals (single‑line, NA‑safe)
    df["BTTS"] = (
        (_pd.to_numeric(df.get("home_team_goal_count"), errors="coerce").fillna(0) > 0)
        & (_pd.to_numeric(df.get("away_team_goal_count"), errors="coerce").fillna(0) > 0)
    ).astype(int)
    df["Over25"] = (
        (_pd.to_numeric(df.get("home_team_goal_count"), errors="coerce").fillna(0)
         + _pd.to_numeric(df.get("away_team_goal_count"), errors="coerce").fillna(0)) > 2
    ).astype(int)

    # ── Export to PredictionReports/<League>_predictions.csv ──
    outdir = os.path.join("PredictionReports")
    try:
        os.makedirs(outdir, exist_ok=True)
    except Exception:
        pass
    outpath = os.path.join(outdir, f"{str(league_name).replace(' ', '_')}_predictions.csv")

    # Choose export columns (only those that exist)
    base_cols = [
        "home_team_name", "away_team_name", "match_date",
        "BTTS", "Over25",
        "adjusted_btts_confidence", "btts_confidence",
        "adjusted_over25_confidence", "over25_confidence",
    ]
    extra_cols = []
    for c in ("p_draw", "draw_flag", "draw_flag_topk", "thr_draw", "mode_draw"):
        if c in df.columns:
            extra_cols.append(c)
    export_cols = [c for c in (base_cols + extra_cols) if c in df.columns]

    try:
        df.to_csv(outpath, index=False, columns=export_cols)
        print_deployable_summary(df)
        print(f"📁 Saved prediction report to: {os.path.abspath(outpath)}")
    except Exception as _e:
        try:
            print(f"⚠️ Could not write prediction report for {league_name}: {_e}")
        except Exception:
            pass

    return df
# --- Deprecation guard: legacy inline FTR mixers -----------------------------
# Any older overlay functions that mixed FTR directly (without predict_FTR_adaptive)
# should not be callable anymore. If they exist in this module, replace them with
# a hard error to prevent accidental use.
try:
    _LEGACY_INLINE_MIXERS = (
        "attach_p_draw_and_mix_ftr",
        "attach_p_draw_mix_ftr",
        "inline_mix_ftr",
        "mix_ftr_inline",
        "attach_ftr_inline",
    )
    def _make_deprecated(name: str):
        def _raise(*args, **kwargs):
            raise RuntimeError(
                "Deprecated inline FTR mixer path. Use attach_ftr_with_draw_mix → predict_FTR_adaptive.")
        _raise.__name__ = f"_DEPRECATED_{name}"
        return _raise
    for _nm in _LEGACY_INLINE_MIXERS:
        fn = globals().get(_nm)
        if callable(fn):
            globals()[_nm] = _make_deprecated(_nm)
except Exception:
    pass