#!/usr/bin/env python3
"""
bookie_allmarkets.py

Generate BOOKIE implied-probability filtered picks (NO EV), and compare to model.

Always writes:
  - model_strength  (val_accuracy from the relevant v2 bundle)
  - ftr_margin      (top1-top2 from FTR 3-way probs; NaN for non-FTR markets)
  - bookie_overround, bookie_implied_novig, gap_novig, bookie_spread (FTR)

Outputs:
  predictions_output/<today>/BOOKIE_IMP{IMP68}_ALLMARKETS_<date_from>_to_<date_to>.csv
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import logging
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "og_mplconfig"))

import numpy as np
import pandas as pd
import joblib
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
logging.getLogger("matplotlib").setLevel(logging.ERROR)

_IMPORT_DIAGNOSTICS = str(os.getenv("OG_IMPORT_DIAGNOSTICS", "0")).strip().lower() in {"1", "true", "yes", "on"}

# -----------------------------------------------------------------------------
# Phase 8 meta scorers (BTTS / OU25 / FTR)
# -----------------------------------------------------------------------------
_META_MODEL_DIR = Path(os.getenv("META_MODEL_DIR", Path(__file__).resolve().parent / "ModelStore" / "_META"))
DRAW_LAYER_ENABLED = str(os.getenv("OG_ENABLE_DRAW_LAYER", "0")).strip().lower() in {"1", "true", "yes", "on"}

_DEFAULT_META_FEATURES: Dict[str, List[str]] = {
    "btts": [
        "cat_xgb_grid_btts_agreement_count",
        "model_p_for_bookie",
        "model_p_for_bookie_xgb",
        "cs_mass_btts_yes",
        "cs_mass_over25",
        "cs_entropy",
        "both_teams_2plus_mass",
        "mass_over25_via_one_sided_rout",
        "mass_0_goals",
        "mass_2_goals",
        "mass_3_goals",
        "mass_4plus_goals",
        "grid_vs_cat_btts_gap",
        "grid_vs_xgb_btts_gap",
        "avg_bookie_od",
        "league",
        "selection",
    ],
    "ou25": [
        "cat_xgb_grid_ou25_agreement_count",
        "model_p_for_bookie",
        "model_p_for_bookie_xgb",
        "cs_mass_btts_yes",
        "cs_mass_over25",
        "cs_entropy",
        "both_teams_2plus_mass",
        "mass_over25_via_one_sided_rout",
        "mass_0_goals",
        "mass_2_goals",
        "mass_3_goals",
        "mass_4plus_goals",
        "grid_vs_cat_ou25_gap",
        "grid_vs_xgb_ou25_gap",
        "avg_bookie_od",
        "league",
        "selection",
    ],
    "ftr": [
        "cat_xgb_grid_ftr_agreement_count",
        "model_p_for_bookie",
        "model_p_for_bookie_xgb",
        "grid_vs_cat_ftr_gap",
        "grid_vs_xgb_ftr_gap",
        "cs_mass_home_win",
        "cs_mass_away_win",
        "cs_entropy",
        "lambda_gap_abs",
        "avg_bookie_od",
        "league",
        "selection",
    ],
}

VALUE_EDGE_TIERS: Dict[str, float] = {
    "PREMIUM": 0.10,
    "STRONG": 0.08,
    "STANDARD": 0.06,
    "MARGINAL": 0.04,
}

TEAM_PROFILE_STAGE0_FILENAME = "TEAM_GOAL_PROFILES_STAGE0.csv"
TEAM_PROFILE_FLAG_COLUMNS = [
    "home_team_high_scoring_flag",
    "away_team_high_scoring_flag",
    "home_team_cs_specialist_flag",
    "away_team_cs_specialist_flag",
    "home_team_fts_risk_flag",
    "away_team_fts_risk_flag",
    "home_team_ge2_candidate_flag",
    "away_team_ge2_candidate_flag",
]
TEAM_FIXTURE_INTERACTION_LABELS = {
    "OTHER",
    "SCORER_VS_SCORER",
    "CS_VS_CS",
    "HOME_GE2_POCKET",
    "AWAY_GE2_POCKET",
    "HOME_SCORER_VS_AWAY_CS",
    "AWAY_SCORER_VS_HOME_CS",
}
_TEAM_PROFILE_STAGE0_CACHE: Optional[Dict[str, Any]] = None
TEAM_CONTEXT_PRODUCT_MAP: Dict[str, Dict[str, str]] = {
    "OTHER": {
        "label": "General Fixture",
        "filter": "GENERAL",
        "bias": "NONE",
    },
    "SCORER_VS_SCORER": {
        "label": "High Scorers Clash",
        "filter": "GOAL_RACE",
        "bias": "BTTS_YES|OU25_OVER",
    },
    "CS_VS_CS": {
        "label": "Clean-Sheet Duel",
        "filter": "LOW_EVENT",
        "bias": "BTTS_NO|UNDER25",
    },
    "HOME_GE2_POCKET": {
        "label": "Home GE2 Pocket",
        "filter": "HOME_TEAM_GOALS",
        "bias": "HOME_WIN|HOME_GE2",
    },
    "AWAY_GE2_POCKET": {
        "label": "Away GE2 Pocket",
        "filter": "AWAY_TEAM_GOALS",
        "bias": "AWAY_WIN|AWAY_GE2",
    },
    "HOME_SCORER_VS_AWAY_CS": {
        "label": "Home Attack vs Away Defence",
        "filter": "DIRECTIONAL_CLASH",
        "bias": "HOME_PRESSURE",
    },
    "AWAY_SCORER_VS_HOME_CS": {
        "label": "Away Attack vs Home Defence",
        "filter": "DIRECTIONAL_CLASH",
        "bias": "AWAY_PRESSURE",
    },
}


def _candidate_team_profile_stage0_paths() -> List[Path]:
    root = Path(__file__).resolve().parent
    candidates: List[Path] = [root / "ModelStore" / "_TEAM" / TEAM_PROFILE_STAGE0_FILENAME]
    reports_dir = root / "reports"
    if reports_dir.exists():
        report_candidates = sorted(reports_dir.glob(f"*/{TEAM_PROFILE_STAGE0_FILENAME}"), reverse=True)
        candidates.extend(report_candidates)
    return candidates


def _maybe_reexec_into_project_venv() -> None:
    """Re-exec into the project venv if core bundle deps are missing there.

    This protects direct `python3 bookie_allmarkets.py ...` runs from silently
    using a base interpreter that cannot unpickle CatBoost/XGBoost bundles.
    """
    if os.getenv("OG_SKIP_PROJECT_VENV_REEXEC", "0").strip().lower() in {"1", "true", "yes", "on"}:
        return

    here = Path(__file__).resolve().parent
    venv_python = here / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return

    try:
        current_prefix = Path(getattr(sys, "prefix", "")).resolve()
        venv_prefix = (here / ".venv").resolve()
        if current_prefix == venv_prefix:
            return
    except Exception:
        pass

    missing = []
    for mod in ("catboost", "xgboost"):
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(mod)
    if not missing:
        return

    try:
        probe = subprocess.run(
            [str(venv_python), "-c", "import catboost, xgboost"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return
    if probe.returncode != 0:
        return

    print(
        "[bookie_allmarkets] re-exec into project venv:",
        {
            "missing_modules": missing,
            "from_python": sys.executable,
            "to_python": str(venv_python),
        },
    )
    env = os.environ.copy()
    env["OG_SKIP_PROJECT_VENV_REEXEC"] = "1"
    argv = [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]]
    try:
        os.execve(str(venv_python), argv, env)
    except Exception:
        proc = subprocess.run(argv, env=env, check=False)
        raise SystemExit(proc.returncode)


def _load_team_profile_stage0_tables() -> Optional[Dict[str, Any]]:
    global _TEAM_PROFILE_STAGE0_CACHE

    if _TEAM_PROFILE_STAGE0_CACHE is not None:
        return _TEAM_PROFILE_STAGE0_CACHE

    required_cols = {
        "league",
        "team",
        "venue",
        "high_scoring_flag",
        "clean_sheet_specialist_flag",
        "fts_risk_flag",
        "ge2_candidate_flag",
    }

    for path in _candidate_team_profile_stage0_paths():
        if not path.exists():
            continue
        try:
            prof = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if not required_cols.issubset(prof.columns):
            continue

        use_cols = [
            "league",
            "team",
            "venue",
            "high_scoring_flag",
            "clean_sheet_specialist_flag",
            "fts_risk_flag",
            "ge2_candidate_flag",
        ]
        prof = prof.loc[:, use_cols].copy()
        prof["league"] = prof["league"].astype("string").fillna("").str.strip()
        prof["team"] = prof["team"].astype("string").fillna("").str.strip()
        prof["venue"] = prof["venue"].astype("string").fillna("").str.upper().str.strip()
        for col in [
            "high_scoring_flag",
            "clean_sheet_specialist_flag",
            "fts_risk_flag",
            "ge2_candidate_flag",
        ]:
            prof[col] = pd.to_numeric(prof[col], errors="coerce").fillna(0).astype(int)

        home = prof.loc[prof["venue"].eq("HOME")].copy().rename(
            columns={
                "team": "home_team_name",
                "high_scoring_flag": "home_team_high_scoring_flag",
                "clean_sheet_specialist_flag": "home_team_cs_specialist_flag",
                "fts_risk_flag": "home_team_fts_risk_flag",
                "ge2_candidate_flag": "home_team_ge2_candidate_flag",
            }
        )
        away = prof.loc[prof["venue"].eq("AWAY")].copy().rename(
            columns={
                "team": "away_team_name",
                "high_scoring_flag": "away_team_high_scoring_flag",
                "clean_sheet_specialist_flag": "away_team_cs_specialist_flag",
                "fts_risk_flag": "away_team_fts_risk_flag",
                "ge2_candidate_flag": "away_team_ge2_candidate_flag",
            }
        )

        _TEAM_PROFILE_STAGE0_CACHE = {
            "source_path": str(path),
            "home": home[
                [
                    "league",
                    "home_team_name",
                    "home_team_high_scoring_flag",
                    "home_team_cs_specialist_flag",
                    "home_team_fts_risk_flag",
                    "home_team_ge2_candidate_flag",
                ]
            ].copy(),
            "away": away[
                [
                    "league",
                    "away_team_name",
                    "away_team_high_scoring_flag",
                    "away_team_cs_specialist_flag",
                    "away_team_fts_risk_flag",
                    "away_team_ge2_candidate_flag",
                ]
            ].copy(),
        }
        print(f"[team-intel] loaded team profile stage0 tables: {path}")
        return _TEAM_PROFILE_STAGE0_CACHE

    return None


def _load_phase8_meta_models() -> Dict[str, Dict[str, Any]]:
    """Load optional Phase 8 meta scorers from ModelStore/_META."""
    models: Dict[str, Dict[str, Any]] = {}
    for market in ("btts", "ou25", "ftr"):
        path = _META_MODEL_DIR / f"{market}_meta_v1.pkl"
        if not path.exists():
            continue
        try:
            obj = joblib.load(path)
            cfg = obj if isinstance(obj, dict) else {"model": obj}
            cfg.setdefault("features", list(_DEFAULT_META_FEATURES.get(market, [])))
            cfg["_path"] = str(path)
            models[market] = cfg
            if _IMPORT_DIAGNOSTICS:
                print(f"[meta] loaded {market} meta model: {path}")
        except Exception as e:
            print(f"[meta] WARNING: failed to load {market} meta model from {path}: {e}")
    return models


_PHASE8_META_MODELS = _load_phase8_meta_models()


def _load_draw_meta_model() -> Optional[Dict[str, Any]]:
    """Load the optional draw meta scorer used for synthetic FTR DRAW rows."""
    path = _META_MODEL_DIR / "draw_meta_v1.pkl"
    if not path.exists():
        return None
    try:
        obj = joblib.load(path)
        cfg = obj if isinstance(obj, dict) else {"model": obj}
        cfg["_path"] = str(path)
        if _IMPORT_DIAGNOSTICS:
            print(f"[draw] loaded draw meta model: {path}")
        return cfg
    except Exception as e:
        print(f"[draw] WARNING: failed to load draw meta model from {path}: {e}")
        return None


_DRAW_META_MODEL = _load_draw_meta_model() if DRAW_LAYER_ENABLED else None

# -----------------------------------------------------------------------------
# BTTS XGB consensus routing (Phase 7A initial rollout)
# -----------------------------------------------------------------------------
BTTS_CONSENSUS_LEAGUES = {
    "Belgium Pro",
    "Brazil Serie A",
    "England Championship",
    "England Premier League",
    "Europa Conference",
    "France Ligue 1",
    "Italy Serie A",
    "Japan J1",
    "Netherlands Eredivisie",
    "Norway Eliteserien",
    "Portugal Liga",
    "Scotland Premiership",
    "Spain La Liga",
    "USA MLS",
}

BTTS_CAT_ONLY_LEAGUES = {
    # CatBoost is already near-perfect here; do not require XGB agreement.
    "England FA Cup",
}

BTTS_XGB_FALLBACK_LEAGUES = {
    # Tiny sample in the first audit; keep Cat as the fallback lane for now.
    "England EFL League 1",
}


def _resolve_ou25_priority(cat_pick: str, xgb_pick: str) -> str:
    """First-pass OU25 priority tier.

    Phase 7B starts as an audit layer only, so we stamp the consensus status
    but do not yet wire league-aware live routing in deploy_rulebook.
    """
    cat = str(cat_pick or "").upper().strip()
    xgb = str(xgb_pick or "").upper().strip()
    if cat and xgb and cat == xgb:
        return "CONSENSUS_ELITE"
    return "CAT_ELITE"


def _priority_rank(label: str) -> int:
    s = str(label or "").upper().strip()
    if s == "CONSENSUS_ELITE":
        return 3
    if s == "CAT_ELITE":
        return 2
    return 1


def _stamp_cross_market_consensus_support(df: pd.DataFrame) -> pd.DataFrame:
    """Propagate best BTTS / OU25 consensus context to sibling rows in the same fixture.

    This is Phase 7C scaffolding only:
      - no tier changes
      - no live rescue routing
      - just stable cross-market context fields for audit + downstream rulebook logic
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    if ("league" not in out.columns) or ("fixture_key" not in out.columns):
        return out

    mk = out.get("market", pd.Series("", index=out.index)).astype("string").fillna("").str.lower().str.strip()
    league = out["league"].astype("string").fillna("").str.strip()
    fixture = out["fixture_key"].astype("string").fillna("").str.strip()

    def _best_support(src: pd.DataFrame, *, priority_col: str, signal_col: str, prefix: str) -> pd.DataFrame:
        if src.empty:
            return pd.DataFrame(columns=["league", "fixture_key"])
        s = src.copy()
        if prefix == "btts":
            # bookie_allmarkets exports BTTS signal labels as signal_btts; the
            # rulebook later derives signal_btts_runtime. For Phase 7C support
            # stamping we need the export-side BTTS labels here.
            signal_series = s.get(
                signal_col,
                s.get("signal_btts", s.get("signal_btts_side", pd.Series("", index=s.index))),
            )
        else:
            signal_series = s.get(signal_col, pd.Series("", index=s.index))
        s["__priority_rank"] = s.get(priority_col, pd.Series("", index=s.index)).map(_priority_rank).astype(int)
        s["__model_p"] = pd.to_numeric(s.get("model_p_for_bookie", np.nan), errors="coerce").fillna(-1.0)
        s["__model_p_xgb"] = pd.to_numeric(
            s.get(
                "model_p_for_bookie_xgb_btts" if prefix == "btts" else "model_p_for_bookie_xgb_ou25",
                np.nan,
            ),
            errors="coerce",
        ).fillna(-1.0)
        s = s.sort_values(
            ["league", "fixture_key", "__priority_rank", "__model_p", "__model_p_xgb"],
            ascending=[True, True, False, False, False],
            kind="mergesort",
        )
        s = s.drop_duplicates(subset=["league", "fixture_key"], keep="first")
        keep = pd.DataFrame({
            "league": s["league"].astype("string").fillna("").str.strip(),
            "fixture_key": s["fixture_key"].astype("string").fillna("").str.strip(),
            f"support_{prefix}_priority": s.get(priority_col, pd.Series("", index=s.index)).astype("string").fillna("").str.upper().str.strip(),
            f"support_{prefix}_priority_rank": s["__priority_rank"].astype(int),
            f"support_{prefix}_pick": s.get("selection", s.get("bookie_pick", pd.Series("", index=s.index))).astype("string").fillna("").str.upper().str.strip(),
            f"support_{prefix}_model_p": pd.to_numeric(s.get("model_p_for_bookie", np.nan), errors="coerce"),
            f"support_{prefix}_model_p_xgb": pd.to_numeric(
                s.get(
                    "model_p_for_bookie_xgb_btts" if prefix == "btts" else "model_p_for_bookie_xgb_ou25",
                    np.nan,
                ),
                errors="coerce",
            ),
            f"support_{prefix}_signal": signal_series.astype("string").fillna("").str.upper().str.strip(),
        })
        keep[f"support_{prefix}_consensus_flag"] = keep[f"support_{prefix}_priority"].eq("CONSENSUS_ELITE").astype(int)
        return keep

    bt_src = out.loc[mk.eq("btts")].copy()
    ou_src = out.loc[mk.eq("ou25")].copy()

    bt_best = _best_support(bt_src, priority_col="btts_priority", signal_col="signal_btts_runtime", prefix="btts")
    ou_best = _best_support(ou_src, priority_col="ou25_priority", signal_col="signal_over25", prefix="ou25")

    if not bt_best.empty:
        out = out.merge(bt_best, on=["league", "fixture_key"], how="left")
    if not ou_best.empty:
        out = out.merge(ou_best, on=["league", "fixture_key"], how="left")

    for prefix in ("btts", "ou25"):
        str_cols = [
            f"support_{prefix}_priority",
            f"support_{prefix}_pick",
            f"support_{prefix}_signal",
        ]
        num_cols = [
            f"support_{prefix}_priority_rank",
            f"support_{prefix}_model_p",
            f"support_{prefix}_model_p_xgb",
            f"support_{prefix}_consensus_flag",
        ]
        for c in str_cols:
            if c not in out.columns:
                out[c] = pd.Series("", index=out.index, dtype="string")
            else:
                out[c] = out[c].astype("string").fillna("").str.strip()
        for c in num_cols:
            if c not in out.columns:
                out[c] = 0 if c.endswith("_flag") or c.endswith("_rank") else np.nan
            if c.endswith("_flag") or c.endswith("_rank"):
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
            else:
                out[c] = pd.to_numeric(out[c], errors="coerce")

    return out

# Optional: per-league close-match thresholds (used for close_match_flag)
try:
    from constants import DRAW_THRESHOLD_PARAMS  # type: ignore
except Exception:
    DRAW_THRESHOLD_PARAMS = {
        "DEFAULT": {"ppg_diff": 0.5, "xg_diff": 0.4, "implied_prob_diff": 0.25, "odds_diff": 3.0}
    }

# Optional: UEFA table-pressure / rotation context (UCL/UEL/UECL)
try:
    from uefa_context import build_snapshot_for_league as _uefa_build_snapshot_for_league  # type: ignore
except Exception:
    _uefa_build_snapshot_for_league = None

# --- Poisson λ + correct-score shortlist helpers (for ALLMARKETS downstream) ---
# --- Poisson λ + correct-score shortlist helpers (for ALLMARKETS downstream) ---
import importlib
import math
import unicodedata

# Shared model-bundle path resolver (keeps loading rules identical across scripts)
try:
    from og_model_paths import resolve_market_bundle_path  # type: ignore
except Exception:
    resolve_market_bundle_path = None


def _bundle_resolver_debug_enabled() -> bool:
    """Enable resolver-path debug logging via --debug or env flag.

    Env flags accepted (truthy):
      - OG_DEBUG_BUNDLES
      - OG_DEBUG_MODEL_RESOLVER
    """
    try:
        if os.getenv("OG_DEBUG_BUNDLES", "0").strip().lower() in ("1", "true", "yes", "y"):
            return True
        if os.getenv("OG_DEBUG_MODEL_RESOLVER", "0").strip().lower() in ("1", "true", "yes", "y"):
            return True
    except Exception:
        pass
    try:
        argv = [str(a).strip().lower() for a in getattr(sys, "argv", [])]
        return "--debug" in argv
    except Exception:
        return False


def _resolve_btts_priority(league: str, cat_pick: str, xgb_pick: str) -> str:
    """Return the BTTS routing tier for a single scored row.

    Phase 7A policy:
      - consensus leagues: only Cat == XGB earns CONSENSUS_ELITE
      - cat-only leagues: preserve Cat ELITE without requiring XGB
      - fallback/unvalidated leagues: keep Cat ELITE until audited
    """
    league_name = str(league or "").strip()
    cat = str(cat_pick or "").upper().strip()
    xgb = str(xgb_pick or "").upper().strip()

    if league_name in BTTS_CAT_ONLY_LEAGUES:
        return "CAT_ELITE"

    if league_name in BTTS_CONSENSUS_LEAGUES:
        if cat and xgb and cat == xgb:
            return "CONSENSUS_ELITE"
        return "CAT_ELITE"

    if league_name in BTTS_XGB_FALLBACK_LEAGUES:
        return "CAT_ELITE"

    # Default conservative fallback for leagues that have not yet been
    # explicitly promoted into the consensus allowlist.
    return "CAT_ELITE"

def _ensure_goal_preds_window(df: pd.DataFrame, league: str) -> pd.DataFrame:
    """
    Ensure df has home_goals_pred / away_goals_pred.
    1) Try to call a helper from prediction_overlay if present (name-agnostic).
    2) Fallback: seed from pre-match xG columns if available.
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    # If already present, keep
    if ("home_goals_pred" in out.columns) and ("away_goals_pred" in out.columns):
        h = pd.to_numeric(out["home_goals_pred"], errors="coerce")
        a = pd.to_numeric(out["away_goals_pred"], errors="coerce")
        if bool(h.notna().any()) and bool(a.notna().any()):
            return out

    # 0) Prefer canonical goal-model scorer from _baseline_ftr_pipeline (per-league ensembles)
    # This is the *real* goal engine; only fall back to xG if models aren't available.
    try:
        base = importlib.import_module("_baseline_ftr_pipeline")

        # A) Preferred: add_goal_predictions_to_complete_data(df, league, required_features)
        if hasattr(base, "add_goal_predictions_to_complete_data"):
            fn = getattr(base, "add_goal_predictions_to_complete_data")
            if callable(fn):
                out2 = None
                try:
                    out2 = fn(out, league, required_features=[])
                except TypeError:
                    try:
                        out2 = fn(out, league, [])
                    except TypeError:
                        try:
                            out2 = fn(out, league_name=league, required_features=[])
                        except Exception:
                            out2 = None
                if isinstance(out2, pd.DataFrame):
                    out = out2

        # B) Fallback: attach_lambda_features_inplace(df, league) -> lambda_home/lambda_away
        if (
            ("home_goals_pred" not in out.columns or out["home_goals_pred"].isna().all())
            or ("away_goals_pred" not in out.columns or out["away_goals_pred"].isna().all())
        ):
            if hasattr(base, "attach_lambda_features_inplace"):
                fn2 = getattr(base, "attach_lambda_features_inplace")
                if callable(fn2):
                    out3 = None
                    try:
                        out3 = fn2(out, league)
                    except TypeError:
                        try:
                            out3 = fn2(out, league_name=league)
                        except Exception:
                            out3 = None
                    if isinstance(out3, pd.DataFrame):
                        out = out3

        # If we only got lambda_* from the fallback, expose home/away_goals_pred too.
        if "lambda_home" in out.columns and ("home_goals_pred" not in out.columns or out["home_goals_pred"].isna().all()):
            out["home_goals_pred"] = pd.to_numeric(out["lambda_home"], errors="coerce")
        if "lambda_away" in out.columns and ("away_goals_pred" not in out.columns or out["away_goals_pred"].isna().all()):
            out["away_goals_pred"] = pd.to_numeric(out["lambda_away"], errors="coerce")

        # If we successfully produced goal preds, we can skip the overlay helper path.
        if ("home_goals_pred" in out.columns) and ("away_goals_pred" in out.columns):
            h0 = pd.to_numeric(out["home_goals_pred"], errors="coerce")
            a0 = pd.to_numeric(out["away_goals_pred"], errors="coerce")
            if bool(h0.notna().any()) and bool(a0.notna().any()):
                # Keep values sane
                out["home_goals_pred"] = h0.clip(lower=0.0, upper=5.0)
                out["away_goals_pred"] = a0.clip(lower=0.0, upper=5.0)
    except Exception:
        pass

    # 1) Try prediction_overlay helper(s) if available
    try:
        mod = importlib.import_module("prediction_overlay")
        for fn_name in [
            # common/likely names (we don't assume exact)
            "create_goal_predictions_if_missing",
            "ensure_goal_predictions_if_missing",
            "ensure_goal_preds_if_missing",
            "ensure_goal_preds",
            "create_goal_preds_if_missing",
            "fill_goal_predictions_if_missing",
        ]:
            if hasattr(mod, fn_name):
                fn = getattr(mod, fn_name)
                if callable(fn):
                    try:
                        out2 = fn(out)
                        if isinstance(out2, pd.DataFrame):
                            out = out2
                    except TypeError:
                        # some variants might require league_name
                        try:
                            out2 = fn(out, league_name=league)
                            if isinstance(out2, pd.DataFrame):
                                out = out2
                        except Exception:
                            pass
                break
    except Exception:
        pass

    # 2) Fallback: seed from pre-match xG (most stable cheap λ proxy)
    if ("home_goals_pred" not in out.columns) or out["home_goals_pred"].isna().all():
        hxg = pd.to_numeric(out.get("pre_match_xg_home", out.get("Home Team Pre-Match xG", np.nan)), errors="coerce")
        out["home_goals_pred"] = hxg.clip(lower=0.0).fillna(1.2)

    if ("away_goals_pred" not in out.columns) or out["away_goals_pred"].isna().all():
        axg = pd.to_numeric(out.get("pre_match_xg_away", out.get("Away Team Pre-Match xG", np.nan)), errors="coerce")
        out["away_goals_pred"] = axg.clip(lower=0.0).fillna(1.0)

    # Keep values sane
    out["home_goals_pred"] = pd.to_numeric(out["home_goals_pred"], errors="coerce").clip(lower=0.0, upper=5.0)
    out["away_goals_pred"] = pd.to_numeric(out["away_goals_pred"], errors="coerce").clip(lower=0.0, upper=5.0)

    # Also expose lambda_* aliases used elsewhere in the stack
    out["lambda_home"] = out["home_goals_pred"]
    out["lambda_away"] = out["away_goals_pred"]

    # --------------------------------------------------------------
    # Trusted-total rescale (OU25): if bookie_lambda_total_fit exists and
    # model λ totals are absurdly low, rescale λ_home/λ_away by the model shares.
    # This fixes regimes where λ_home+λ_away << bookie total.
    # Gate:
    #   - lam_sum < 1.0 OR (lam_sum / bookie_lambda_total_fit) < 0.80
    # Only uses bookie totals when they look sane (1.0–6.0).
    # --------------------------------------------------------------
    try:
        if "bookie_lambda_total_fit" in out.columns:
            bk = pd.to_numeric(out.get("bookie_lambda_total_fit", np.nan), errors="coerce")
            bk_sane = bk.where(bk.between(1.0, 6.0))

            lh = pd.to_numeric(out.get("lambda_home", np.nan), errors="coerce")
            la = pd.to_numeric(out.get("lambda_away", np.nan), errors="coerce")
            lam_sum = lh + la

            ratio = (lam_sum / bk_sane)
            m_ou = out.get("market", "").astype("string").fillna("").str.lower().str.strip().eq("ou25")
            m_rescale = m_ou & bk_sane.notna() & ((lam_sum < 1.0) | (ratio < 0.80))

            if bool(m_rescale.any()):
                denom = lam_sum.where(lam_sum > 1e-9)
                share_h = (lh / denom).clip(lower=0.0, upper=1.0)
                share_a = (la / denom).clip(lower=0.0, upper=1.0)

                # If denom is missing/zero for a row, default to 50/50 split
                share_h = share_h.where(denom.notna(), 0.5)
                share_a = share_a.where(denom.notna(), 0.5)

                # Normalize shares so they sum to 1
                share_sum = (share_h + share_a).where((share_h + share_a) > 1e-9)
                share_h = (share_h / share_sum).where(share_sum.notna(), 0.5)
                share_a = (share_a / share_sum).where(share_sum.notna(), 0.5)

                new_lh = (bk_sane * share_h).clip(lower=0.0)
                new_la = (bk_sane * share_a).clip(lower=0.0)

                out.loc[m_rescale, "lambda_home"] = new_lh.loc[m_rescale].astype(float)
                out.loc[m_rescale, "lambda_away"] = new_la.loc[m_rescale].astype(float)

                # Keep goal-pred aliases consistent
                out.loc[m_rescale, "home_goals_pred"] = pd.to_numeric(out.loc[m_rescale, "lambda_home"], errors="coerce")
                out.loc[m_rescale, "away_goals_pred"] = pd.to_numeric(out.loc[m_rescale, "lambda_away"], errors="coerce")

                # Optional debug via env flag (keeps function signature stable)
                if os.getenv("OG_DEBUG_LAMBDA_RESCALE", "0").strip().lower() in ("1", "true", "yes", "y"):
                    try:
                        before_ratio = pd.to_numeric(ratio.loc[m_rescale], errors="coerce")
                        after_sum = pd.to_numeric(out.loc[m_rescale, "lambda_home"], errors="coerce") + pd.to_numeric(out.loc[m_rescale, "lambda_away"], errors="coerce")
                        after_ratio = (after_sum / bk_sane.loc[m_rescale]).astype(float)
                        print(
                            f"[LAMBDA_RESCALE] league={league} rows={int(m_rescale.sum())} "
                            f"ratio_before(min/med/max)={float(before_ratio.min()):.3f}/{float(before_ratio.median()):.3f}/{float(before_ratio.max()):.3f} "
                            f"ratio_after(min/med/max)={float(after_ratio.min()):.3f}/{float(after_ratio.median()):.3f}/{float(after_ratio.max()):.3f}"
                        )
                    except Exception:
                        pass
    except Exception:
        # Never break generation due to rescale logic
        pass

    out["exp_goals_sum"] = pd.to_numeric(out["lambda_home"], errors="coerce") + pd.to_numeric(out["lambda_away"], errors="coerce")
    out["p00_est"] = np.exp(-pd.to_numeric(out["exp_goals_sum"], errors="coerce").clip(lower=0.0))

    return out


def _apply_absurd_lambda_and_fts_sanity(
    df: pd.DataFrame,
    league: str = "",
    *,
    debug: bool = False,
) -> pd.DataFrame:
    """Repair clearly broken goal/FTS states before market rows are emitted.

    Why:
      - Some fixtures arrive with absurdly tiny model lambdas even when raw pre-match
        xG is perfectly sane (e.g. exp_goals_sum ~ 0.10 with xG sum > 3.0).
      - Specialist FTS heads can also disagree violently with the current goal lambdas,
        which can invert BTTS picks.

    Policy:
      - Only intervene on clearly pathological rows.
      - Prefer raw pre-match xG as a conservative fallback for broken λ totals.
      - If trained FTS probabilities contradict the current Poisson blank rates too
        aggressively, fall back to the Poisson blank rates for those rows.
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    lam_h = pd.to_numeric(out.get("home_goals_pred", out.get("lambda_home", np.nan)), errors="coerce")
    lam_a = pd.to_numeric(out.get("away_goals_pred", out.get("lambda_away", np.nan)), errors="coerce")
    raw_hxg = pd.to_numeric(
        out.get("pre_match_xg_home", out.get("Home Team Pre-Match xG", out.get("xg_home", np.nan))),
        errors="coerce",
    )
    raw_axg = pd.to_numeric(
        out.get("pre_match_xg_away", out.get("Away Team Pre-Match xG", out.get("xg_away", np.nan))),
        errors="coerce",
    )

    lam_sum = lam_h + lam_a
    raw_sum = raw_hxg + raw_axg

    absurd_lambda_mask = (
        lam_sum.notna()
        & raw_sum.notna()
        & (lam_sum < 0.75)
        & (raw_sum >= 2.0)
    )

    if bool(absurd_lambda_mask.any()):
        new_h = raw_hxg.clip(lower=0.05, upper=5.0)
        new_a = raw_axg.clip(lower=0.05, upper=5.0)
        out.loc[absurd_lambda_mask, "home_goals_pred"] = new_h.loc[absurd_lambda_mask].astype(float)
        out.loc[absurd_lambda_mask, "away_goals_pred"] = new_a.loc[absurd_lambda_mask].astype(float)
        out.loc[absurd_lambda_mask, "lambda_home"] = new_h.loc[absurd_lambda_mask].astype(float)
        out.loc[absurd_lambda_mask, "lambda_away"] = new_a.loc[absurd_lambda_mask].astype(float)

    lam_h = pd.to_numeric(out.get("home_goals_pred", out.get("lambda_home", np.nan)), errors="coerce").clip(lower=0.0, upper=10.0)
    lam_a = pd.to_numeric(out.get("away_goals_pred", out.get("lambda_away", np.nan)), errors="coerce").clip(lower=0.0, upper=10.0)

    out["exp_goals_sum"] = lam_h + lam_a
    out["p00_est"] = np.exp(-pd.to_numeric(out["exp_goals_sum"], errors="coerce").clip(lower=0.0))

    pois_home_fts = np.exp(-lam_h)
    pois_away_fts = np.exp(-lam_a)
    cur_home_fts = pd.to_numeric(out.get("p_home_fts", pd.Series(np.nan, index=out.index)), errors="coerce")
    cur_away_fts = pd.to_numeric(out.get("p_away_fts", pd.Series(np.nan, index=out.index)), errors="coerce")

    bad_home_fts = (
        lam_h.notna()
        & (
            cur_home_fts.isna()
            | ((cur_home_fts - pois_home_fts).abs() > 0.45)
            | ((cur_home_fts < 0.02) & (pois_home_fts > 0.12))
            | ((cur_home_fts > 0.85) & (pois_home_fts < 0.45))
        )
    )
    bad_away_fts = (
        lam_a.notna()
        & (
            cur_away_fts.isna()
            | ((cur_away_fts - pois_away_fts).abs() > 0.45)
            | ((cur_away_fts < 0.02) & (pois_away_fts > 0.12))
            | ((cur_away_fts > 0.85) & (pois_away_fts < 0.45))
        )
    )

    if bool(bad_home_fts.any()):
        out.loc[bad_home_fts, "p_home_fts"] = pois_home_fts.loc[bad_home_fts].astype(float)
    if bool(bad_away_fts.any()):
        out.loc[bad_away_fts, "p_away_fts"] = pois_away_fts.loc[bad_away_fts].astype(float)

    if debug and (bool(absurd_lambda_mask.any()) or bool(bad_home_fts.any()) or bool(bad_away_fts.any())):
        try:
            print(
                f"[GOAL_SANITY] {league}: absurd_lambda_rows={int(absurd_lambda_mask.sum())} "
                f"fts_home_overrides={int(bad_home_fts.sum())} fts_away_overrides={int(bad_away_fts.sum())}"
            )
        except Exception:
            pass

    return out


def _attach_poisson_cs_top3(df: pd.DataFrame, *, max_goals: int = 6) -> pd.DataFrame:
    """
    Attach:
      cs1/cs1_p, cs2/cs2_p, cs3/cs3_p,
      p_home_pois/p_draw_pois/p_away_pois,
      cs_trunc_mass_0_6
    Requires: home_goals_pred, away_goals_pred
    """
    if df is None or df.empty:
        return df
    if ("home_goals_pred" not in df.columns) or ("away_goals_pred" not in df.columns):
        return df

    out = df.copy()
    grid = list(range(0, int(max_goals) + 1))
    W = int(max_goals) + 1

    def _pmf(k: int, lam: float) -> float:
        if not np.isfinite(lam) or lam < 0:
            return 0.0
        return float(math.exp(-lam) * (lam ** int(k)) / math.factorial(int(k)))

    cs1, cs1p, cs2, cs2p, cs3, cs3p = [], [], [], [], [], []
    pH, pD, pA, tmass = [], [], [], []

    for _, r in out.iterrows():
        lam_h = float(pd.to_numeric(r.get("home_goals_pred", np.nan), errors="coerce"))
        lam_a = float(pd.to_numeric(r.get("away_goals_pred", np.nan), errors="coerce"))

        if (not np.isfinite(lam_h)) or (not np.isfinite(lam_a)) or lam_h < 0 or lam_a < 0:
            cs1.append(""); cs1p.append(np.nan)
            cs2.append(""); cs2p.append(np.nan)
            cs3.append(""); cs3p.append(np.nan)
            pH.append(np.nan); pD.append(np.nan); pA.append(np.nan); tmass.append(np.nan)
            continue

        lam_h = float(np.clip(lam_h, 0.0, 10.0))
        lam_a = float(np.clip(lam_a, 0.0, 10.0))

        h_probs = np.array([_pmf(k, lam_h) for k in grid], dtype=float)
        a_probs = np.array([_pmf(k, lam_a) for k in grid], dtype=float)
        mat = np.outer(h_probs, a_probs)

        ph = float(np.sum(np.tril(mat, k=-1)))
        pd_ = float(np.sum(np.diag(mat)))
        pa = float(np.sum(np.triu(mat, k=1)))
        msum = float(mat.sum())

        flat = mat.ravel()
        if flat.size == 0:
            top_scores = ["", "", ""]
            top_probs = [np.nan, np.nan, np.nan]
        else:
            kk = 3
            top_idx = np.argpartition(-flat, range(min(kk, flat.size)))[:min(kk, flat.size)]
            top_idx = top_idx[np.argsort(-flat[top_idx])]
            top_scores, top_probs = [], []
            for idx in top_idx:
                h = int(idx // W)
                a = int(idx % W)
                top_scores.append(f"{h}-{a}")
                top_probs.append(float(flat[idx]))
            while len(top_scores) < 3:
                top_scores.append("")
                top_probs.append(np.nan)

        cs1.append(top_scores[0]); cs1p.append(top_probs[0])
        cs2.append(top_scores[1]); cs2p.append(top_probs[1])
        cs3.append(top_scores[2]); cs3p.append(top_probs[2])
        pH.append(ph); pD.append(pd_); pA.append(pa); tmass.append(msum)

    out["cs1"] = cs1; out["cs1_p"] = cs1p
    out["cs2"] = cs2; out["cs2_p"] = cs2p
    out["cs3"] = cs3; out["cs3_p"] = cs3p
    out["p_home_pois"] = pH
    out["p_draw_pois"] = pD
    out["p_away_pois"] = pA
    out["cs_trunc_mass_0_6"] = tmass
    return out

def _attach_team_poisson_tails(df: pd.DataFrame) -> pd.DataFrame:
    """Attach per-team Poisson zero-goal and tail probabilities from λ (home_goals_pred/away_goals_pred).

    Adds:
      - cs_home, cs_away
      - p_home_ge1, p_away_ge1
      - p_home_ge2, p_away_ge2
      - p_home_ge3, p_away_ge3
      - p_home_ge4, p_away_ge4

    Backward-compat aliases retained:
      - pois_home_ge2, pois_away_ge2
      - pois_home_ge3, pois_away_ge3
    """
    if df is None or df.empty:
        return df
    if ("home_goals_pred" not in df.columns) or ("away_goals_pred" not in df.columns):
        return df

    out = df.copy()

    lh = pd.to_numeric(out.get("home_goals_pred"), errors="coerce").clip(lower=0.0, upper=10.0)
    la = pd.to_numeric(out.get("away_goals_pred"), errors="coerce").clip(lower=0.0, upper=10.0)

    eh = np.exp(-lh)
    ea = np.exp(-la)

    # Zero-goal probabilities
    out["cs_home"] = eh.clip(0.0, 1.0)
    out["cs_away"] = ea.clip(0.0, 1.0)

    # P(X >= 1) = 1 - e^-λ
    out["p_home_ge1"] = (1.0 - out["cs_home"]).clip(0.0, 1.0)
    out["p_away_ge1"] = (1.0 - out["cs_away"]).clip(0.0, 1.0)

    # P(X >= 2) = 1 - e^-λ (1 + λ)
    out["p_home_ge2"] = (1.0 - (eh * (1.0 + lh))).clip(0.0, 1.0)
    out["p_away_ge2"] = (1.0 - (ea * (1.0 + la))).clip(0.0, 1.0)

    # P(X >= 3) = 1 - e^-λ (1 + λ + λ^2/2)
    out["p_home_ge3"] = (1.0 - (eh * (1.0 + lh + 0.5 * lh * lh))).clip(0.0, 1.0)
    out["p_away_ge3"] = (1.0 - (ea * (1.0 + la + 0.5 * la * la))).clip(0.0, 1.0)

    # P(X >= 4) = 1 - e^-λ (1 + λ + λ^2/2 + λ^3/6)
    out["p_home_ge4"] = (1.0 - (eh * (1.0 + lh + 0.5 * lh * lh + (lh * lh * lh) / 6.0))).clip(0.0, 1.0)
    out["p_away_ge4"] = (1.0 - (ea * (1.0 + la + 0.5 * la * la + (la * la * la) / 6.0))).clip(0.0, 1.0)

    # Backward-compat aliases for existing TG / coherence logic
    out["pois_home_ge2"] = out["p_home_ge2"]
    out["pois_away_ge2"] = out["p_away_ge2"]
    out["pois_home_ge3"] = out["p_home_ge3"]
    out["pois_away_ge3"] = out["p_away_ge3"]

    return out


def _build_poisson_cs_grid(lambda_home: float, lambda_away: float, max_goals: int = 6) -> np.ndarray:
    """Build a normalized Poisson correct-score grid with tail mass folded into the last bucket."""
    W = int(max_goals) + 1
    grid = np.full((W, W), np.nan, dtype=float)

    try:
        lam_h = float(pd.to_numeric(lambda_home, errors="coerce"))
        lam_a = float(pd.to_numeric(lambda_away, errors="coerce"))
    except Exception:
        return grid

    if (not np.isfinite(lam_h)) or (not np.isfinite(lam_a)) or lam_h < 0 or lam_a < 0:
        return grid

    lam_h = float(np.clip(lam_h, 0.0, 10.0))
    lam_a = float(np.clip(lam_a, 0.0, 10.0))

    def _pmf(k: int, lam: float) -> float:
        return float(math.exp(-lam) * (lam ** int(k)) / math.factorial(int(k)))

    h_probs = np.array([_pmf(k, lam_h) for k in range(max_goals)], dtype=float)
    a_probs = np.array([_pmf(k, lam_a) for k in range(max_goals)], dtype=float)
    h_tail = max(0.0, 1.0 - float(h_probs.sum()))
    a_tail = max(0.0, 1.0 - float(a_probs.sum()))

    h_all = np.append(h_probs, h_tail)
    a_all = np.append(a_probs, a_tail)
    grid = np.outer(h_all, a_all).astype(float, copy=False)

    grid_sum = float(np.nansum(grid))
    if np.isfinite(grid_sum) and grid_sum > 0.0:
        grid = grid / grid_sum
    return grid


def _compute_phase8a_grid_features_from_grid(
    cs_grid: np.ndarray,
    *,
    lambda_home: float,
    lambda_away: float,
) -> dict[str, float]:
    """Extract first-pass Phase 8A fixture-level grid features."""
    feat_names = [
        "cs_mass_btts_yes",
        "cs_mass_btts_no",
        "cs_mass_over25",
        "cs_mass_under25",
        "cs_mass_home_win",
        "cs_mass_draw",
        "cs_mass_away_win",
        "cs_entropy",
        "both_teams_2plus_mass",
        "mass_over25_via_one_sided_rout",
        "mass_0_goals",
        "mass_1_goal",
        "mass_2_goals",
        "mass_3_goals",
        "mass_4plus_goals",
    ]
    out = {k: np.nan for k in feat_names}

    if not isinstance(cs_grid, np.ndarray) or cs_grid.ndim != 2 or cs_grid.shape[0] != cs_grid.shape[1]:
        return out

    G = np.array(cs_grid, dtype=float, copy=True)
    if not np.isfinite(G).any():
        return out

    G = np.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)
    G = np.clip(G, 0.0, None)
    grid_sum = float(G.sum())
    if grid_sum <= 0.0:
        return out
    G = G / grid_sum

    n = G.shape[0]
    totals = np.add.outer(np.arange(n), np.arange(n))
    ii = np.arange(n)[:, None]
    jj = np.arange(n)[None, :]

    flat = np.sort(G.ravel())[::-1]

    out["cs_mass_btts_yes"] = float(G[1:, 1:].sum())
    out["cs_mass_btts_no"] = float(G[(ii == 0) | (jj == 0)].sum())
    out["cs_mass_over25"] = float(G[totals >= 3].sum())
    out["cs_mass_under25"] = float(G[totals <= 2].sum())
    out["cs_mass_home_win"] = float(G[ii > jj].sum())
    out["cs_mass_draw"] = float(np.trace(G))
    out["cs_mass_away_win"] = float(G[ii < jj].sum())

    out["cs_entropy"] = float(max(0.0, -np.sum(G[G > 0.0] * np.log(G[G > 0.0]))))
    out["both_teams_2plus_mass"] = float(G[2:, 2:].sum()) if n > 2 else 0.0
    out["mass_over25_via_one_sided_rout"] = float(G[((ii == 0) | (jj == 0)) & (totals >= 3)].sum())

    out["mass_0_goals"] = float(G[0, 0])
    out["mass_1_goal"] = float(G[totals == 1].sum())
    out["mass_2_goals"] = float(G[totals == 2].sum())
    out["mass_3_goals"] = float(G[totals == 3].sum())
    out["mass_4plus_goals"] = float(G[totals >= 4].sum())

    for k, v in list(out.items()):
        if k != "cs_entropy":
            out[k] = float(np.clip(v, 0.0, 1.0)) if np.isfinite(v) else np.nan
    return out


def _attach_phase8a_grid_features(df: pd.DataFrame, max_goals: int = 6) -> pd.DataFrame:
    """Attach fixture-level Phase 8A Poisson grid features to all market rows."""
    if df is None or df.empty:
        return df

    out = df.copy()
    required = ["league", "fixture_key", "lambda_home", "lambda_away"]
    if any(c not in out.columns for c in required):
        return out

    feat_cols = [
        "cs_mass_btts_yes",
        "cs_mass_btts_no",
        "cs_mass_over25",
        "cs_mass_under25",
        "cs_mass_home_win",
        "cs_mass_draw",
        "cs_mass_away_win",
        "cs_entropy",
        "both_teams_2plus_mass",
        "mass_over25_via_one_sided_rout",
        "mass_0_goals",
        "mass_1_goal",
        "mass_2_goals",
        "mass_3_goals",
        "mass_4plus_goals",
    ]

    fx = out[["league", "fixture_key", "lambda_home", "lambda_away"]].copy()
    fx["league"] = fx["league"].astype("string").fillna("").str.strip()
    fx["fixture_key"] = fx["fixture_key"].astype("string").fillna("").str.strip()
    fx["lambda_home"] = pd.to_numeric(fx["lambda_home"], errors="coerce")
    fx["lambda_away"] = pd.to_numeric(fx["lambda_away"], errors="coerce")
    fx = fx.loc[
        fx["league"].ne("")
        & fx["fixture_key"].ne("")
        & fx["lambda_home"].notna()
        & fx["lambda_away"].notna()
    ].drop_duplicates(subset=["league", "fixture_key"], keep="first")

    if fx.empty:
        for c in feat_cols:
            if c not in out.columns:
                out[c] = np.nan
        return out

    records = []
    for r in fx.itertuples(index=False):
        grid = _build_poisson_cs_grid(r.lambda_home, r.lambda_away, max_goals=max_goals)
        feats = _compute_phase8a_grid_features_from_grid(
            grid,
            lambda_home=r.lambda_home,
            lambda_away=r.lambda_away,
        )
        feats["league"] = r.league
        feats["fixture_key"] = r.fixture_key
        records.append(feats)

    feat_df = pd.DataFrame.from_records(records)
    if feat_df.empty:
        for c in feat_cols:
            if c not in out.columns:
                out[c] = np.nan
        return out

    feat_df = feat_df.drop_duplicates(subset=["league", "fixture_key"], keep="first")
    out = out.drop(columns=[c for c in feat_cols if c in out.columns], errors="ignore")
    out = out.merge(feat_df, on=["league", "fixture_key"], how="left")
    for c in feat_cols:
        out[c] = pd.to_numeric(out.get(c, np.nan), errors="coerce")
    return out


def _attach_phase8b_coherence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach row-aware coherence features between grid, Cat, and XGB market signals."""
    if df is None or df.empty:
        return df

    out = df.copy()
    mk = out.get("market", pd.Series("", index=out.index)).astype("string").fillna("").str.lower().str.strip()
    pick = out.get("bookie_pick", out.get("selection", pd.Series("", index=out.index))).astype("string").fillna("").str.upper().str.strip()

    feat_cols = [
        "grid_vs_cat_btts_gap",
        "grid_vs_xgb_btts_gap",
        "grid_vs_cat_ou25_gap",
        "grid_vs_xgb_ou25_gap",
        "grid_vs_cat_ftr_gap",
        "grid_vs_xgb_ftr_gap",
        "cat_xgb_grid_btts_agreement_count",
        "cat_xgb_grid_ou25_agreement_count",
        "cat_xgb_grid_ftr_agreement_count",
    ]
    for c in feat_cols:
        if c not in out.columns:
            out[c] = np.nan

    cat_p = pd.to_numeric(out.get("model_p_for_bookie", np.nan), errors="coerce")

    # BTTS coherence
    is_btts = mk.eq("btts")
    btts_yes_mass = pd.to_numeric(out.get("cs_mass_btts_yes", np.nan), errors="coerce")
    btts_no_mass = pd.to_numeric(out.get("cs_mass_btts_no", np.nan), errors="coerce")
    btts_grid_p = pd.Series(np.nan, index=out.index, dtype="float64")
    btts_grid_p.loc[is_btts & pick.eq("YES")] = btts_yes_mass.loc[is_btts & pick.eq("YES")]
    btts_grid_p.loc[is_btts & pick.eq("NO")] = btts_no_mass.loc[is_btts & pick.eq("NO")]
    btts_xgb_p = pd.to_numeric(out.get("model_p_for_bookie_xgb_btts", np.nan), errors="coerce")

    out.loc[is_btts, "grid_vs_cat_btts_gap"] = (btts_grid_p.loc[is_btts] - cat_p.loc[is_btts]).abs()
    out.loc[is_btts, "grid_vs_xgb_btts_gap"] = (btts_grid_p.loc[is_btts] - btts_xgb_p.loc[is_btts]).abs()

    btts_cat_agree = is_btts & cat_p.ge(0.50)
    btts_xgb_agree = is_btts & btts_xgb_p.ge(0.50)
    btts_grid_agree = is_btts & btts_grid_p.ge(0.50)
    out.loc[is_btts, "cat_xgb_grid_btts_agreement_count"] = (
        btts_cat_agree.loc[is_btts].astype(int)
        + btts_xgb_agree.loc[is_btts].astype(int)
        + btts_grid_agree.loc[is_btts].astype(int)
    )

    # OU25 coherence
    is_ou25 = mk.eq("ou25")
    ou25_over_mass = pd.to_numeric(out.get("cs_mass_over25", np.nan), errors="coerce")
    ou25_under_mass = pd.to_numeric(out.get("cs_mass_under25", np.nan), errors="coerce")
    ou25_grid_p = pd.Series(np.nan, index=out.index, dtype="float64")
    ou25_grid_p.loc[is_ou25 & pick.eq("OVER25")] = ou25_over_mass.loc[is_ou25 & pick.eq("OVER25")]
    ou25_grid_p.loc[is_ou25 & pick.eq("UNDER25")] = ou25_under_mass.loc[is_ou25 & pick.eq("UNDER25")]
    ou25_xgb_p = pd.to_numeric(out.get("model_p_for_bookie_xgb_ou25", np.nan), errors="coerce")

    out.loc[is_ou25, "grid_vs_cat_ou25_gap"] = (ou25_grid_p.loc[is_ou25] - cat_p.loc[is_ou25]).abs()
    out.loc[is_ou25, "grid_vs_xgb_ou25_gap"] = (ou25_grid_p.loc[is_ou25] - ou25_xgb_p.loc[is_ou25]).abs()

    ou25_cat_agree = is_ou25 & cat_p.ge(0.50)
    ou25_xgb_agree = is_ou25 & ou25_xgb_p.ge(0.50)
    ou25_grid_agree = is_ou25 & ou25_grid_p.ge(0.50)
    out.loc[is_ou25, "cat_xgb_grid_ou25_agreement_count"] = (
        ou25_cat_agree.loc[is_ou25].astype(int)
        + ou25_xgb_agree.loc[is_ou25].astype(int)
        + ou25_grid_agree.loc[is_ou25].astype(int)
    )

    # FTR coherence
    is_ftr = mk.eq("ftr")
    ftr_home_mass = pd.to_numeric(out.get("cs_mass_home_win", np.nan), errors="coerce")
    ftr_draw_mass = pd.to_numeric(out.get("cs_mass_draw", np.nan), errors="coerce")
    ftr_away_mass = pd.to_numeric(out.get("cs_mass_away_win", np.nan), errors="coerce")
    ftr_grid_p = pd.Series(np.nan, index=out.index, dtype="float64")
    ftr_grid_p.loc[is_ftr & pick.eq("HOME")] = ftr_home_mass.loc[is_ftr & pick.eq("HOME")]
    ftr_grid_p.loc[is_ftr & pick.eq("DRAW")] = ftr_draw_mass.loc[is_ftr & pick.eq("DRAW")]
    ftr_grid_p.loc[is_ftr & pick.eq("AWAY")] = ftr_away_mass.loc[is_ftr & pick.eq("AWAY")]
    ftr_xgb_p = pd.to_numeric(out.get("model_p_for_bookie_xgb", np.nan), errors="coerce")

    out.loc[is_ftr, "grid_vs_cat_ftr_gap"] = (ftr_grid_p.loc[is_ftr] - cat_p.loc[is_ftr]).abs()
    out.loc[is_ftr, "grid_vs_xgb_ftr_gap"] = (ftr_grid_p.loc[is_ftr] - ftr_xgb_p.loc[is_ftr]).abs()

    # Count how many of Cat, XGB, and grid support this row's selected side.
    ftr_cat_pick = pd.Series(pd.NA, index=out.index, dtype="string")
    ftr_cat_pick.loc[is_ftr & pick.eq("HOME") & cat_p.ge(0.50)] = "HOME"
    ftr_cat_pick.loc[is_ftr & pick.eq("DRAW") & cat_p.ge(0.50)] = "DRAW"
    ftr_cat_pick.loc[is_ftr & pick.eq("AWAY") & cat_p.ge(0.50)] = "AWAY"

    ftr_xgb_pick = pd.Series(pd.NA, index=out.index, dtype="string")
    ftr_xgb_pick.loc[is_ftr & pick.eq("HOME") & ftr_xgb_p.ge(0.50)] = "HOME"
    ftr_xgb_pick.loc[is_ftr & pick.eq("DRAW") & ftr_xgb_p.ge(0.50)] = "DRAW"
    ftr_xgb_pick.loc[is_ftr & pick.eq("AWAY") & ftr_xgb_p.ge(0.50)] = "AWAY"

    ftr_grid_stack = np.column_stack([
        ftr_home_mass.fillna(-np.inf).to_numpy(),
        ftr_draw_mass.fillna(-np.inf).to_numpy(),
        ftr_away_mass.fillna(-np.inf).to_numpy(),
    ])
    ftr_grid_argmax = np.argmax(ftr_grid_stack, axis=1)
    ftr_grid_pick = pd.Series(np.array(["HOME", "DRAW", "AWAY"], dtype=object)[ftr_grid_argmax], index=out.index, dtype="string")
    ftr_grid_pick.loc[~is_ftr] = pd.NA
    ftr_grid_pick.loc[is_ftr & ftr_home_mass.isna() & ftr_draw_mass.isna() & ftr_away_mass.isna()] = pd.NA

    ftr_cat_agree = is_ftr & ftr_cat_pick.eq(pick)
    ftr_xgb_agree = is_ftr & ftr_xgb_pick.eq(pick)
    ftr_grid_agree = is_ftr & ftr_grid_pick.eq(pick)
    out.loc[is_ftr, "cat_xgb_grid_ftr_agreement_count"] = (
        ftr_cat_agree.loc[is_ftr].fillna(False).astype(int)
        + ftr_xgb_agree.loc[is_ftr].fillna(False).astype(int)
        + ftr_grid_agree.loc[is_ftr].fillna(False).astype(int)
    )

    for c in feat_cols:
        out[c] = pd.to_numeric(out.get(c, np.nan), errors="coerce")
    return out


def _stamp_phase8_meta_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Score rows with optional Phase 8 meta models and stamp p_meta_* columns."""
    if df is None or df.empty:
        return df

    out = df.copy()
    for market in ("btts", "ou25", "ftr"):
        col = f"p_meta_{market}"
        if col not in out.columns:
            out[col] = np.nan

    if not _PHASE8_META_MODELS:
        return out

    market_series = out.get("market", pd.Series("", index=out.index)).astype("string").fillna("").str.lower().str.strip()
    selection_series = out.get(
        "selection",
        out.get("bookie_pick", pd.Series("", index=out.index)),
    ).astype("string").fillna("").str.upper().str.strip()

    for market, cfg in _PHASE8_META_MODELS.items():
        model = cfg.get("model", cfg)
        features = list(cfg.get("features") or _DEFAULT_META_FEATURES.get(market, []))
        if model is None or not features:
            continue

        mask = market_series.eq(market)
        if market == "ftr":
            mask = mask & selection_series.isin(["HOME", "AWAY"])

        if not bool(mask.any()):
            continue

        X = out.loc[mask].copy()

        # Canonicalise market-specific XGB probability columns to the generic
        # meta-training feature name expected by BTTS / OU25 scorers.
        if market == "btts":
            if "model_p_for_bookie_xgb" not in X.columns:
                X["model_p_for_bookie_xgb"] = np.nan
            if "model_p_for_bookie_xgb_btts" in X.columns:
                gx = pd.to_numeric(X["model_p_for_bookie_xgb"], errors="coerce")
                sx = pd.to_numeric(X["model_p_for_bookie_xgb_btts"], errors="coerce")
                X["model_p_for_bookie_xgb"] = gx.where(gx.notna(), sx)

        if market == "ou25":
            if "model_p_for_bookie_xgb" not in X.columns:
                X["model_p_for_bookie_xgb"] = np.nan
            if "model_p_for_bookie_xgb_ou25" in X.columns:
                gx = pd.to_numeric(X["model_p_for_bookie_xgb"], errors="coerce")
                sx = pd.to_numeric(X["model_p_for_bookie_xgb_ou25"], errors="coerce")
                X["model_p_for_bookie_xgb"] = gx.where(gx.notna(), sx)

        meta_X = pd.DataFrame(index=X.index)
        for feat in features:
            if feat in X.columns:
                meta_X[feat] = X[feat]
            else:
                meta_X[feat] = np.nan

        try:
            proba = np.asarray(model.predict_proba(meta_X))
            if proba.ndim == 2 and proba.shape[1] >= 2:
                scores = proba[:, 1].astype(float)
            else:
                scores = np.asarray(proba).reshape(-1).astype(float)
            out.loc[mask, f"p_meta_{market}"] = scores
        except Exception as e:
            path = cfg.get("_path", "(unknown)")
            print(f"[meta] WARNING: scoring failed for {market} using {path}: {e}")

    return out


def _generate_draw_rows(df: pd.DataFrame, draw_model_pkg: Optional[Dict[str, Any]]) -> pd.DataFrame:
    """
    Generate one synthetic FTR DRAW row per fixture and score it with the saved
    draw meta pipeline.

    The saved draw model already contains its own imputer/calibration logic, so
    we pass the raw reindexed feature frame directly into `predict_proba`.
    """
    if (not DRAW_LAYER_ENABLED) or df is None or df.empty or not draw_model_pkg:
        return pd.DataFrame()

    model = draw_model_pkg.get("model")
    feature_cols = list(draw_model_pkg.get("features") or [])
    if model is None or not feature_cols:
        return pd.DataFrame()

    ftr = df[
        df.get("market", pd.Series("", index=df.index))
        .astype("string")
        .fillna("")
        .str.lower()
        .str.strip()
        .eq("ftr")
    ].copy()
    if ftr.empty:
        return pd.DataFrame()

    fixture_id_cols = [c for c in ("league", "fixture_key") if c in ftr.columns]
    if not fixture_id_cols:
        fixture_id_cols = [c for c in ("league", "home_team_name", "away_team_name", "match_date") if c in ftr.columns]
    if not fixture_id_cols:
        return pd.DataFrame()

    draw_base = ftr.drop_duplicates(subset=fixture_id_cols, keep="first").copy()

    draw_base["selection"] = "DRAW"
    draw_base["bookie_pick"] = "DRAW"
    draw_base["market"] = "ftr"
    draw_base["is_synthetic_draw_row"] = 1
    draw_base["draw_meta_source"] = "draw_meta_v1"

    odds_source_col = None
    for draw_odds_col in ("od_draw", "odds_draw_decimal", "bookie_od_draw"):
        if draw_odds_col in draw_base.columns:
            odds_source_col = draw_odds_col
            draw_base["odds_draw_decimal"] = pd.to_numeric(draw_base[draw_odds_col], errors="coerce")
            break

    if "odds_draw_decimal" not in draw_base.columns:
        draw_base["odds_draw_decimal"] = np.nan
    if "odds_draw_implied" not in draw_base.columns:
        draw_base["odds_draw_implied"] = np.nan
    if "bookie_od" not in draw_base.columns:
        draw_base["bookie_od"] = np.nan

    draw_base["odds_draw_implied"] = 1.0 / draw_base["odds_draw_decimal"].replace(0, np.nan)
    draw_base["bookie_od"] = draw_base["odds_draw_decimal"]

    draw_enc = pd.get_dummies(draw_base.copy(), columns=["league"], drop_first=True)
    meta_X = draw_enc.reindex(columns=feature_cols, fill_value=np.nan)

    try:
        proba = np.asarray(model.predict_proba(meta_X))
        if proba.ndim == 2 and proba.shape[1] >= 2:
            scores = proba[:, 1].astype(float)
        else:
            scores = np.asarray(proba).reshape(-1).astype(float)
    except Exception as e:
        path = draw_model_pkg.get("_path", "(unknown)")
        print(f"[draw] WARNING: scoring synthetic DRAW rows failed using {path}: {e}")
        return pd.DataFrame()

    draw_base["p_draw"] = scores
    draw_base["p_meta_ftr_draw"] = scores

    draw_base = draw_base[
        draw_base["odds_draw_decimal"].notna()
        & (draw_base["odds_draw_decimal"] > 1.0)
    ].copy()

    if odds_source_col is not None and "od_source" in draw_base.columns:
        draw_base["od_source"] = "bookie_pick"

    return draw_base.reset_index(drop=True)

def _reconcile_fixture_lambdas_from_ou25(
    df: pd.DataFrame,
    *,
    debug: bool = False,
    ou_market: str = "ou25",
    sane_total_min: float = 2.0,
    broken_total_max: float = 0.75,
    broken_ratio_max: float = 0.35,
) -> pd.DataFrame:
    """Propagate repaired OU25 lambdas back onto clearly-broken sibling rows.

    Why:
      - OU25 rows can be repaired late via trusted-total rescale.
      - FTR / BTTS rows for the same fixture may still carry stale tiny lambdas.
      - This helper builds a fixture-level map from repaired OU25 rows and overwrites
        non-OU25 rows only when their totals are clearly broken relative to OU25.

    Recomputes on repaired rows:
      - home_goals_pred / away_goals_pred
      - exp_goals_sum
      - p00_est
      - Poisson CS shortlist / side probs
      - team poisson tails
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    needed = ["league", "fixture_key", "market", "lambda_home", "lambda_away"]
    if any(c not in out.columns for c in needed):
        return out

    out["league"] = out["league"].astype("string").fillna("").str.strip()
    out["fixture_key"] = out["fixture_key"].astype("string").fillna("").str.strip()
    mk = out["market"].astype("string").fillna("").str.lower().str.strip()

    m_ou = mk.eq(str(ou_market).lower().strip())
    if not bool(m_ou.any()):
        return out

    # Build fixture-level canonical map ONLY from repaired/sane OU25 rows
    ou = out.loc[m_ou, ["league", "fixture_key", "lambda_home", "lambda_away"]].copy()
    ou["ou_lambda_home"] = pd.to_numeric(ou["lambda_home"], errors="coerce")
    ou["ou_lambda_away"] = pd.to_numeric(ou["lambda_away"], errors="coerce")
    ou["ou_xgsum"] = ou["ou_lambda_home"] + ou["ou_lambda_away"]
    ou = ou.loc[ou["ou_xgsum"].notna() & (ou["ou_xgsum"] >= float(sane_total_min))].copy()
    if ou.empty:
        return out

    ou = ou[["league", "fixture_key", "ou_lambda_home", "ou_lambda_away", "ou_xgsum"]].copy()
    ou = ou.drop_duplicates(subset=["league", "fixture_key"], keep="first")

    out = out.merge(ou, on=["league", "fixture_key"], how="left")

    row_lh = pd.to_numeric(out.get("lambda_home", np.nan), errors="coerce")
    row_la = pd.to_numeric(out.get("lambda_away", np.nan), errors="coerce")
    row_xgsum = row_lh + row_la
    ou_xgsum = pd.to_numeric(out.get("ou_xgsum", np.nan), errors="coerce")
    ratio = (row_xgsum / ou_xgsum)

    repair_mask = (
        (~m_ou)
        & out["league"].ne("")
        & out["fixture_key"].ne("")
        & ou_xgsum.notna()
        & row_xgsum.notna()
        & (
            (row_xgsum < float(broken_total_max))
            | (ratio < float(broken_ratio_max))
        )
    )

    if not bool(repair_mask.any()):
        out = out.drop(columns=["ou_lambda_home", "ou_lambda_away", "ou_xgsum"], errors="ignore")
        return out

    out.loc[repair_mask, "lambda_home"] = pd.to_numeric(
        out.loc[repair_mask, "ou_lambda_home"], errors="coerce"
    )
    out.loc[repair_mask, "lambda_away"] = pd.to_numeric(
        out.loc[repair_mask, "ou_lambda_away"], errors="coerce"
    )

    if "home_goals_pred" in out.columns:
        out.loc[repair_mask, "home_goals_pred"] = pd.to_numeric(
            out.loc[repair_mask, "lambda_home"], errors="coerce"
        )
    else:
        out["home_goals_pred"] = pd.to_numeric(out.get("lambda_home", np.nan), errors="coerce")

    if "away_goals_pred" in out.columns:
        out.loc[repair_mask, "away_goals_pred"] = pd.to_numeric(
            out.loc[repair_mask, "lambda_away"], errors="coerce"
        )
    else:
        out["away_goals_pred"] = pd.to_numeric(out.get("lambda_away", np.nan), errors="coerce")

    repaired_xgsum = (
        pd.to_numeric(out.loc[repair_mask, "lambda_home"], errors="coerce")
        + pd.to_numeric(out.loc[repair_mask, "lambda_away"], errors="coerce")
    )
    out.loc[repair_mask, "exp_goals_sum"] = repaired_xgsum
    out.loc[repair_mask, "p00_est"] = np.exp(
        -pd.to_numeric(out.loc[repair_mask, "exp_goals_sum"], errors="coerce").clip(lower=0.0)
    )

    # Refresh Poisson-derived fields so CS / tails align to final lambdas
    out = _attach_poisson_cs_top3(out, max_goals=6)
    out = _attach_team_poisson_tails(out)
    out = _attach_phase8a_grid_features(out, max_goals=6)

    if debug:
        try:
            print(
                "[FIXTURE LAMBDA RECONCILE]",
                {
                    "rows_repaired": int(repair_mask.sum()),
                    "sane_total_min": float(sane_total_min),
                    "broken_total_max": float(broken_total_max),
                    "broken_ratio_max": float(broken_ratio_max),
                },
            )
        except Exception:
            pass

    out = out.drop(columns=["ou_lambda_home", "ou_lambda_away", "ou_xgsum"], errors="ignore")
    return out

def _attach_draw_chaos_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Stamp Draw/Chaos Risk onto the output pool (FTR rows only).

    Produces:
      - draw_risk_flag   (0/1)
      - chaos_risk_flag  (0/1)
      - not_glue_flag    (0/1)  # warning-first soft flag
      - hard_not_glue_flag (0/1)  # stricter live-risk flag
      - draw_chaos_score (0..1)

    Uses ONLY existing output columns:
      confidence_home/confidence_draw/confidence_away, ftr_margin,
      and rolling-rate columns (scored/conceded/clean_sheet/btts/goaliness).
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    # Ensure destination columns exist
    for c in ("draw_risk_flag", "not_glue_flag", "hard_not_glue_flag", "draw_chaos_score", "chaos_risk_flag"):
        if c not in out.columns:
            out[c] = np.nan

    if "market" not in out.columns:
        out["draw_risk_flag"] = 0
        out["chaos_risk_flag"] = 0
        out["not_glue_flag"] = 0
        out["hard_not_glue_flag"] = 0
        out["draw_chaos_score"] = np.nan

    # Note: we compute chaos/draw risk for ALL rows. Some components (e.g. confidence_draw/ftr_margin)
    # only exist on FTR rows; when missing they gracefully fall back and do not trigger flags.
    m_ftr = out["market"].astype(str).str.lower().str.strip().eq("ftr")

    # Helper: always return a Series aligned to out.index (never a scalar)
    def _col(name: str, default=np.nan) -> pd.Series:
        if name in out.columns:
            return out[name]
        return pd.Series(default, index=out.index)
    def _as_series(x) -> pd.Series:
        if isinstance(x, pd.Series):
            return x
        return pd.Series(x, index=out.index)
    # Core model probs + margin
    pD = pd.to_numeric(_col("confidence_draw"), errors="coerce").clip(0.0, 1.0)
    margin = pd.to_numeric(_col("ftr_margin"), errors="coerce")

    # Rolling-rate proxies (safe if missing)
    scored_h = pd.to_numeric(_col("scored_rate_5_home"), errors="coerce").clip(0.0, 1.0)
    scored_a = pd.to_numeric(_col("scored_rate_5_away"), errors="coerce").clip(0.0, 1.0)

    conceded_h = pd.to_numeric(_col("conceded_rate_5_home"), errors="coerce").clip(0.0, 1.0)
    conceded_a = pd.to_numeric(_col("conceded_rate_5_away"), errors="coerce").clip(0.0, 1.0)

    cs_h = pd.to_numeric(_col("clean_sheet_rate_5_home"), errors="coerce").clip(0.0, 1.0)
    cs_a = pd.to_numeric(_col("clean_sheet_rate_5_away"), errors="coerce").clip(0.0, 1.0)

    btts_h = pd.to_numeric(_col("btts_rate_5_home"), errors="coerce").clip(0.0, 1.0)
    btts_a = pd.to_numeric(_col("btts_rate_5_away"), errors="coerce").clip(0.0, 1.0)

    # Goaliness is in avg-goals units; scale into 0..1 (2.2 -> ~0, 4.0 -> ~1)
    goal_h = pd.to_numeric(_col("goaliness_avg_5_home"), errors="coerce")
    goal_a = pd.to_numeric(_col("goaliness_avg_5_away"), errors="coerce")

    cs_mean = pd.concat([cs_h, cs_a], axis=1).mean(axis=1, skipna=True)
    conceded_mean = pd.concat([conceded_h, conceded_a], axis=1).mean(axis=1, skipna=True)
    btts_mean = pd.concat([btts_h, btts_a], axis=1).mean(axis=1, skipna=True)
    scored_mean = pd.concat([scored_h, scored_a], axis=1).mean(axis=1, skipna=True)
    goal_mean = pd.concat([goal_h, goal_a], axis=1).mean(axis=1, skipna=True)

    goal_scaled = ((goal_mean - 2.2) / 1.8).clip(0.0, 1.0)

    # Chaos profile: high scoring + high conceding + low CS + high BTTS + high goaliness
    chaos_components = pd.concat([
        (1.0 - cs_mean),
        conceded_mean,
        btts_mean,
        goal_scaled,
        scored_mean,
    ], axis=1)
    chaos_score = chaos_components.mean(axis=1, skipna=True).clip(0.0, 1.0)

    # Margin risk: small margin => unstable
    margin_risk = ((0.25 - margin) / 0.25).clip(0.0, 1.0)

    # Draw pressure scaled: ~0.33 is “very live”
    pD_scaled = (pD / 0.33).clip(0.0, 1.0).fillna(0.0)

    # Final risk score (0..1)
    score = (0.40 * pD_scaled) + (0.35 * chaos_score.fillna(0.5)) + (0.25 * margin_risk.fillna(0.5))
    score = score.clip(0.0, 1.0)

    # Draw-risk flag: only when draw prob is genuinely elevated,
    # or when draw prob is moderate AND margin is tight (knife-edge).
    draw_flag = (
        (pD >= 0.29)
        | ((pD >= 0.25) & (margin.notna()) & (margin <= 0.10))
    ).fillna(False).astype(int)

    # Chaos-only flag: high-variance profile even if draw prob isn't extreme.
    # (Uses the rolling-rate signals you already have.)
    chaos_flag = (
        (chaos_score >= 0.62)
        | ((chaos_score >= 0.55) & (margin.notna()) & (margin <= 0.08))
        | ((conceded_mean >= 0.60) & (btts_mean >= 0.60) & (cs_mean <= 0.40))
    ).fillna(False).astype(int)

    # Warning-first NOT-GLUE: broad soft flag for draw-risk OR chaos-risk.
    # Keep this intentionally sensitive for diagnostics / observe demotions.
    not_glue = ((draw_flag == 1) | (chaos_flag == 1)).fillna(False).astype(int)

    # Hard NOT-GLUE: stricter live-risk layer.
    # Only trip when draw risk is genuinely present AND the combined profile is hot,
    # or when both draw-risk and chaos-risk fire together.
    hard_not_glue = (
        ((draw_flag == 1) & score.notna() & (score >= 0.60))
        | ((draw_flag == 1) & (chaos_flag == 1))
    ).fillna(False).astype(int)

    # Stamp on ALL rows (FTR rows will also carry pD/margin-driven effects)
    out["draw_chaos_score"] = score
    out["draw_risk_flag"] = draw_flag
    out["chaos_risk_flag"] = chaos_flag
    out["not_glue_flag"] = not_glue
    out["hard_not_glue_flag"] = hard_not_glue

    out["draw_risk_flag"] = pd.to_numeric(out["draw_risk_flag"], errors="coerce").fillna(0).astype(int)
    out["chaos_risk_flag"] = pd.to_numeric(out["chaos_risk_flag"], errors="coerce").fillna(0).astype(int)
    out["not_glue_flag"] = pd.to_numeric(out["not_glue_flag"], errors="coerce").fillna(0).astype(int)
    out["hard_not_glue_flag"] = pd.to_numeric(out["hard_not_glue_flag"], errors="coerce").fillna(0).astype(int)
    out["draw_chaos_score"] = pd.to_numeric(out["draw_chaos_score"], errors="coerce")

    return out

def _attach_close_match_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Stamp a close-match router flag onto the output pool (FTR rows only).

    Adds/updates:
      - close_match_flag (0/1)
      - xg_diff_abs
      - implied_prob_diff
      - odds_diff

    Uses league-specific thresholds from constants.DRAW_THRESHOLD_PARAMS when available.
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    # Ensure destination columns exist
    for c in ("close_match_flag", "xg_diff_abs", "implied_prob_diff", "odds_diff"):
        if c not in out.columns:
            out[c] = np.nan

    if "market" not in out.columns:
        out["close_match_flag"] = 0
        return out

    m_ftr = out["market"].astype("string").fillna("").str.lower().str.strip().eq("ftr")
    if not bool(m_ftr.any()):
        out["close_match_flag"] = pd.to_numeric(out.get("close_match_flag", 0), errors="coerce").fillna(0).astype(int)
        return out

    # --- Inputs (safe numeric) ---
    ppg = pd.to_numeric(out.loc[m_ftr].get("ppg_diff_pre", np.nan), errors="coerce").abs()

    xg_h = pd.to_numeric(out.loc[m_ftr].get("pre_match_xg_home", np.nan), errors="coerce")
    xg_a = pd.to_numeric(out.loc[m_ftr].get("pre_match_xg_away", np.nan), errors="coerce")
    xg_diff = (xg_h - xg_a).abs()
    out.loc[m_ftr, "xg_diff_abs"] = xg_diff

    od_h = pd.to_numeric(out.loc[m_ftr].get("od_home", np.nan), errors="coerce")
    od_d = pd.to_numeric(out.loc[m_ftr].get("od_draw", np.nan), errors="coerce")
    od_a = pd.to_numeric(out.loc[m_ftr].get("od_away", np.nan), errors="coerce")
    odds_diff = (od_h - od_a).abs()
    out.loc[m_ftr, "odds_diff"] = odds_diff

    imp_h = pd.to_numeric(out.loc[m_ftr].get("imp_home", np.nan), errors="coerce")
    imp_d = pd.to_numeric(out.loc[m_ftr].get("imp_draw", np.nan), errors="coerce")
    imp_a = pd.to_numeric(out.loc[m_ftr].get("imp_away", np.nan), errors="coerce")

    # Fallback implieds from odds if needed
    imp_h = imp_h.where(imp_h.notna(), (1.0 / od_h).where(od_h > 1.0))
    imp_d = imp_d.where(imp_d.notna(), (1.0 / od_d).where(od_d > 1.0))
    imp_a = imp_a.where(imp_a.notna(), (1.0 / od_a).where(od_a > 1.0))

    s = (imp_h + imp_d + imp_a)
    imp_h_nv = (imp_h / s).where(s > 0)
    imp_a_nv = (imp_a / s).where(s > 0)
    implied_prob_diff = (imp_h_nv - imp_a_nv).abs()
    out.loc[m_ftr, "implied_prob_diff"] = implied_prob_diff

    # --- League thresholds (from constants.py) ---
    leagues = out.loc[m_ftr].get("league", "DEFAULT").astype("string").fillna("DEFAULT")

    def _thr_series(col: str, default: float) -> pd.Series:
        def _get(lg: str) -> float:
            base = DRAW_THRESHOLD_PARAMS.get(lg, DRAW_THRESHOLD_PARAMS.get("DEFAULT", {}))
            try:
                return float(base.get(col, default))
            except Exception:
                return float(default)
        return leagues.map(_get)

    t_ppg = _thr_series("ppg_diff", 0.5)
    t_xg  = _thr_series("xg_diff", 0.4)
    t_ipd = _thr_series("implied_prob_diff", 0.25)
    t_od  = _thr_series("odds_diff", 3.0)

    flag = (ppg <= t_ppg) & (xg_diff <= t_xg) & (implied_prob_diff <= t_ipd) & (odds_diff <= t_od)
    out.loc[m_ftr, "close_match_flag"] = flag.fillna(False).astype(int)

    # Normalize type
    out["close_match_flag"] = pd.to_numeric(out.get("close_match_flag", 0), errors="coerce").fillna(0).astype(int)

    return out

def _coalesce_match_date_series(df: pd.DataFrame) -> pd.Series:
    """Return the best available date-like series for windowing.

    Important: do NOT prefer `match_date` if it is empty/NA (some league files only have `date_GMT`).
    """
    # 1) If match_date exists AND contains any usable values, use it
    if "match_date" in df.columns:
        try:
            s = df["match_date"].astype("string").str.strip()
            s = s.mask(s.eq(""), pd.NA)
            if bool(s.notna().any()):
                return df["match_date"]
        except Exception:
            try:
                if bool(pd.to_datetime(df["match_date"], errors="coerce", utc=True).notna().any()):
                    return df["match_date"]
            except Exception:
                pass

    # 2) Otherwise fall back to the raw date columns
    for c in ("date_GMT", "date", "Date", "timestamp"):
        if c in df.columns:
            return df[c]

    return pd.Series(pd.NA, index=df.index)

def _norm_team_token(x: object) -> str:
    import re
    s = str(x or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def _ascii_fold(s: object) -> str:
    """Best-effort ASCII folding (removes diacritics)."""
    t = str(s or "").strip()
    if not t:
        return ""
    try:
        return "".join(ch for ch in unicodedata.normalize("NFKD", t) if not unicodedata.combining(ch))
    except Exception:
        return t

def _norm_team_token_ascii(x: object) -> str:
    """Normalize team token but first ASCII-fold diacritics (Qarabağ -> Qarabag)."""
    return _norm_team_token(_ascii_fold(x))

# OG-style key: YYYY_MM_DD_HOME_AWAY
def _match_key(row: pd.Series) -> str:
    # OG-style key: YYYY_MM_DD_HOME_AWAY
    md = pd.to_datetime(row.get("match_date", None), errors="coerce", utc=True)
    ds = md.strftime("%Y_%m_%d") if pd.notna(md) else ""
    h = _norm_team_token(row.get("home_team_name", row.get("HomeTeam", "")))
    a = _norm_team_token(row.get("away_team_name", row.get("AwayTeam", "")))
    key = f"{ds}_{h}_{a}".strip("_")
    return key

# ASCII-folded version of _match_key (diacritics-safe)
def _match_key_ascii(row: pd.Series) -> str:
    md = pd.to_datetime(row.get("match_date", None), errors="coerce", utc=True)
    ds = md.strftime("%Y_%m_%d") if pd.notna(md) else ""

    def _fold(v: object) -> str:
        try:
            s = str(v or "").strip()
        except Exception:
            return ""
        if not s:
            return ""
        try:
            s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
        except Exception:
            pass
        return s

    h = _norm_team_token(_fold(row.get("home_team_name", row.get("HomeTeam", ""))))
    a = _norm_team_token(_fold(row.get("away_team_name", row.get("AwayTeam", ""))))
    key = f"{ds}_{h}_{a}".strip("_")
    return key
# Optional: signal-label attachment (VERY_STRONG/STRONG/NEUTRAL) for side markets.
# This powers deploy_gates side-label gates instead of confidence fallbacks.
try:
    from signal_layers import attach_signal_layers as _attach_signal_layers  # type: ignore
except Exception:
    _attach_signal_layers = None

try:
    from prediction_overlay import attach_signal_layers_if_available as _attach_signal_layers_if_available  # type: ignore
except Exception:
    _attach_signal_layers_if_available = None

# Optional: leak-safe rolling team rates (shifted) for BTTS/Ou2.5 gating
# Prefer the light `attach_team_rates()`; fall back to the orchestrator if needed.
try:
    from streaks_module import attach_team_rates as _attach_team_rates  # type: ignore
except Exception:
    _attach_team_rates = None

try:
    from streaks_module import attach_streaks_and_h2h as _attach_streaks_and_h2h  # type: ignore
except Exception:
    _attach_streaks_and_h2h = None

try:
    from streaks_module import attach_h2h_streaks as _attach_h2h_streaks  # type: ignore
except Exception:
    _attach_h2h_streaks = None
# Optional: specialist market heads trained by train_markets.py (GE2/GE3/FTS)
try:
    from train_markets import score_trained_markets as _score_trained_markets  # type: ignore
except Exception:
    _score_trained_markets = None

DEFAULT_LEAGUES = [
    "Champions League",
    "Europa League",
    "Europa Conference",
    "Germany Bundesliga",
    "Germany Bundesliga 2",
    "France Ligue 1",
    "Italy Serie A",
    "Spain La Liga",
    "England Premier League",
    "England Championship",
    "England EFL League 1",
    "England FA Cup",
    "Portugal Liga",
    "USA MLS",
    "Brazil Serie A",
    "Scotland Premiership",
    "Belgium Pro",
    "Netherlands Eredivisie",
    "Norway Eliteserien",
    "Japan J1",
    "Australia A-League",
    "Austria Bundesliga",
    "Czech First League",
    "Denmark Superliga",
    "Saudi Pro League",
    "South Korea K League",
    "Sweden Allsvenskan",
    "Swiss Super League",
    "Turkey Super Lig",
]


def _league_tag(league: str) -> str:
    league = str(league).strip()
    overrides = {
        "Australia A-League": "Australia_A_League",
        # Canonical tag uses underscores throughout; the legacy space-tag
        # breaks merged/modelstore resolution for EFL1.
        "England EFL League 1": "England_EFL_League_1",
    }
    if league in overrides:
        return overrides[league]
    return league.replace(" ", "_")

def _load_bundle(modelstore: Path, league: str, market: str, *, engine: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load a per-league model bundle using the shared resolver.

    Core markets are STRICT V3-only here:
      - ftr
      - btts
      - ou25 / over25 alias

    V2 fallback is intentionally disallowed for those core markets.
    (V2 is only permitted for side/specialist models elsewhere in the stack.)

    Notes:
      - We persist `_bundle_path` and `_bundle_file` onto dict bundles for --debug reporting.
      - If the shared resolver is unavailable, we fall back to a minimal strict-V3 loader.
    """
    tag = _league_tag(league)
    mkt = str(market or "").strip().lower()

    # Shared resolver path (preferred): identical loading rules across scripts.
    if callable(resolve_market_bundle_path):
        # Core markets must be strict V3-only.
        core_markets = {"ftr", "btts", "ou25", "over25"}
        is_core = mkt in core_markets

        # Keep env knobs for non-core/specialist markets only.
        allow_v2_env = os.getenv("OG_ALLOW_V2_FALLBACK", "0").strip().lower() in ("1", "true", "yes", "y")
        pref_ftr = os.getenv("OG_FTR_VERSION", "v3").strip().lower()
        if pref_ftr not in ("auto", "v2", "v3"):
            pref_ftr = "v3"
        pref_side = os.getenv("OG_SIDE_VERSION", "v3").strip().lower()
        if pref_side not in ("auto", "v2", "v3"):
            pref_side = "v3"

        prefer = pref_ftr if mkt == "ftr" else pref_side
        # Force strict V3 for core markets (ignore v2 env fallback there)
        if is_core:
            prefer = "v3"
            allow_v2 = False

        # Special-case: allow XGB FTR bundles to fall back to v2 for side-by-side audits
        if mkt == "ftr" and str(engine or "").strip().lower() in ("xgb", "xgboost"):
            prefer = os.getenv("OG_FTR_XGB_VERSION", "v2").strip().lower()
            if prefer not in ("v2", "v3"):
                prefer = "v2"
            allow_v2 = True
        else:
            allow_v2 = allow_v2_env

        try:
            res = resolve_market_bundle_path(modelstore, tag, mkt, prefer=prefer, allow_v2=allow_v2, engine=engine)
        except TypeError:
            # Back-compat if resolver signature differs slightly
            res = resolve_market_bundle_path(modelstore, tag, mkt)

        p = None
        if isinstance(res, dict):
            p_raw = res.get("path")
            exists = bool(res.get("exists", False))
            if p_raw:
                p = Path(p_raw)
                if not exists:
                    p = None
        elif isinstance(res, (str, Path)):
            p = Path(res)

        if p is not None and p.exists():
            try:
                b = joblib.load(p)
                if isinstance(b, dict):
                    b["_bundle_path"] = str(p)
                    b["_bundle_file"] = p.name
                    # Optional resolver metadata for debugging / auditing
                    if isinstance(res, dict):
                        b.setdefault("_bundle_resolver_version", res.get("version_loaded"))
                        b.setdefault("_bundle_resolver_alias", res.get("alias_used"))

                    if _bundle_resolver_debug_enabled():
                        print(
                            f"[BUNDLE_RESOLVER] league={league} market={mkt} "
                            f"path={b.get('_bundle_path')} "
                            f"alias={b.get('_bundle_resolver_alias')} "
                            f"version={b.get('_bundle_resolver_version')}"
                        )
                return b
            except Exception as e:
                if _bundle_resolver_debug_enabled():
                    print(
                        f"[BUNDLE_RESOLVER] LOAD_FAIL league={league} market={mkt} "
                        f"path={p} error={type(e).__name__}: {e}"
                    )
                return None

        # Debug print for unsuccessful/missing resolution
        if callable(resolve_market_bundle_path) and _bundle_resolver_debug_enabled():
            try:
                if isinstance(res, dict):
                    print(
                        f"[BUNDLE_RESOLVER] MISS league={league} market={mkt} "
                        f"path={res.get('path')} alias={res.get('alias_used')} "
                        f"version={res.get('version_loaded')} exists={res.get('exists')}"
                    )
                    cands = res.get("candidates")
                    if cands:
                        try:
                            cand_str = " | ".join(str(x) for x in cands)
                        except Exception:
                            cand_str = str(cands)
                        print(f"[BUNDLE_RESOLVER] CANDIDATES league={league} market={mkt} tried={cand_str}")
            except Exception:
                pass

    # Minimal fallback if shared resolver import is unavailable (strict V3-only)
    if mkt == "ou25":
        cands = [
            modelstore / tag / "over25_v3.pkl",
            modelstore / tag / "ou25_v3.pkl",
        ]
    else:
        cands = [modelstore / tag / f"{mkt}_v3.pkl"]

    for p in cands:
        if not p.exists():
            continue
        try:
            b = joblib.load(p)
            if isinstance(b, dict):
                b["_bundle_path"] = str(p)
                b["_bundle_file"] = p.name
            return b
        except Exception as e:
            if _bundle_resolver_debug_enabled():
                print(
                    f"[BUNDLE_RESOLVER] FALLBACK_LOAD_FAIL league={league} market={mkt} "
                    f"path={p} error={type(e).__name__}: {e}"
                )
            continue

    return None

def _filter_bundle_engine(bundle: Optional[Dict[str, Any]], want_engine: Optional[str]) -> Optional[Dict[str, Any]]:
    """Ensure bundle engine matches requested engine (xgb/cat). Cat allows legacy bundles without engine tag."""
    if not isinstance(bundle, dict):
        return None
    want = str(want_engine or "").strip().lower()
    if not want:
        return bundle
    eng = str(bundle.get("engine", "")).strip().lower()
    if want in ("xgb", "xgboost"):
        return bundle if eng in ("xgb", "xgboost") else None
    if want in ("cat", "catboost"):
        # allow missing engine tag for legacy cat bundles
        return None if eng in ("xgb", "xgboost") else bundle
    return bundle

def _model_strength_from_bundle(bundle: Optional[Dict[str, Any]]) -> float:
    if not isinstance(bundle, dict):
        return np.nan
    v = bundle.get("val_accuracy", np.nan)
    try:
        v = float(v)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan

def _bundle_debug_id(bundle: Optional[Dict[str, Any]]) -> str:
    if not isinstance(bundle, dict):
        return "<none>"
    fn = str(bundle.get("_bundle_file") or "").strip()
    fp = str(bundle.get("_bundle_path") or "").strip()
    if fn and fp:
        return f"{fn} ({fp})"
    if fn:
        return fn
    if fp:
        return fp
    ver = str(bundle.get("version") or bundle.get("bundle_version") or "").strip()
    if ver:
        return f"<bundle ver={ver}>"
    return "<bundle>"

# --- Begin: Power ratings helpers ---

def _load_match_power_ratings(modelstore: Path, league: str) -> pd.DataFrame:
    """Load per-match power ratings for a league if available.

    Primary expected file:
      - ModelStore/<LeagueTag>_match_power_ratings.csv
        Must contain: fixture_key and (home_power_rating, away_power_rating, power_diff)

    Fallback:
      - ModelStore/<LeagueTag>_team_ratings.csv
        Must contain a team name column + a rating column.
        This is mapped onto match frames via home_team_name/away_team_name.

    Returns:
      - If per-match file exists: DataFrame keyed by fixture_key.
      - Else if per-team file exists: DataFrame keyed by `team_key`.
      - Else: empty DataFrame.
    """
    tag = _league_tag(league)

    # 1) Preferred: per-match ratings keyed by fixture_key
    p_match = modelstore / f"{tag}_match_power_ratings.csv"
    if p_match.exists():
        try:
            pr = pd.read_csv(p_match, low_memory=False)
        except Exception:
            pr = pd.DataFrame()

        if pr is None or pr.empty:
            pr = pd.DataFrame()
        else:
            if "fixture_key" not in pr.columns:
                pr = pd.DataFrame()
            else:
                pr = pr.copy()
                pr["fixture_key"] = pr["fixture_key"].astype("string").fillna("").str.strip()
                pr = pr[pr["fixture_key"].ne("")].copy()

                keep = ["fixture_key"]
                for c in (
                    "home_power_rating", "away_power_rating", "power_diff",
                    "home_power_rating_raw", "away_power_rating_raw",
                ):
                    if c in pr.columns:
                        keep.append(c)
                pr = pr[keep].copy()

                try:
                    pr = pr.drop_duplicates(subset=["fixture_key"], keep="first")
                except Exception:
                    pass

                for c in (
                    "home_power_rating", "away_power_rating", "power_diff",
                    "home_power_rating_raw", "away_power_rating_raw",
                ):
                    if c in pr.columns:
                        pr[c] = pd.to_numeric(pr[c], errors="coerce")

        if pr is not None and not pr.empty:
            return pr

    # 2) Fallback: per-team ratings keyed by team name
    # team_ratings.py writes: ModelStore/<LeagueTag>_team_ratings.csv
    p_team = modelstore / f"{tag}_team_ratings.csv"
    if not p_team.exists():
        return pd.DataFrame()

    try:
        tr = pd.read_csv(p_team)
    except Exception:
        return pd.DataFrame()

    if tr is None or tr.empty:
        return pd.DataFrame()

    # Identify likely team column
    team_col = None
    for c in ("team", "team_name", "Team", "name", "club", "squad"):
        if c in tr.columns:
            team_col = c
            break
    if not team_col:
        return pd.DataFrame()

    # Identify likely rating columns (prefer already-canonical names)
    # We accept multiple schemas and normalize to: home/away_power_rating(+_raw)
    rating_col = None
    raw_col = None
    for c in (
        "power_rating", "team_power_rating", "rating", "strength", "power", "score",
        "home_power_rating",  # sometimes people store a generic as this
    ):
        if c in tr.columns:
            rating_col = c
            break

    for c in (
        "power_rating_raw", "raw_rating", "raw_strength", "strength_raw",
        "home_power_rating_raw",
    ):
        if c in tr.columns:
            raw_col = c
            break

    if not rating_col:
        return pd.DataFrame()

    out = tr.copy()
    out[team_col] = out[team_col].astype("string").fillna("").str.strip()
    out = out[out[team_col].ne("")].copy()

    # Normalize team name -> key (diacritics-safe)
    out["team_key"] = out[team_col].map(_norm_team_token_ascii).astype("string")
    out["team_key"] = out["team_key"].fillna("").str.strip()
    out = out[out["team_key"].ne("")].copy()

    out["power_rating"] = pd.to_numeric(out[rating_col], errors="coerce")
    if raw_col:
        out["power_rating_raw"] = pd.to_numeric(out[raw_col], errors="coerce")
    else:
        out["power_rating_raw"] = out["power_rating"]

    out = out[["team_key", "power_rating", "power_rating_raw"]].copy()
    try:
        out = out.drop_duplicates(subset=["team_key"], keep="first")
    except Exception:
        pass

    return out


def _attach_power_ratings(df: pd.DataFrame, league: str, modelstore: Path) -> pd.DataFrame:
    """Attach power ratings onto a match frame.

    Strategy:
      A) If ModelStore/<LeagueTag>_match_power_ratings.csv exists, join on fixture_key.
      B) Else if ModelStore/<LeagueTag>_team_ratings.csv exists, map by team names.

    Adds (always present after this function):
      - home_power_rating, away_power_rating, power_diff
      - home_power_rating_raw, away_power_rating_raw
    """
    if df is None or df.empty:
        return df

    pr = _load_match_power_ratings(modelstore, league)
    out = df.copy()

    # Optional debug: verify duplicates BEFORE attaching power ratings
    # Enable with: OG_DEBUG_POWER=1
    if os.getenv("OG_DEBUG_POWER", "0").strip().lower() in ("1", "true", "yes", "y"):
        try:
            if "fixture_key" in out.columns:
                vc = out["fixture_key"].astype("string").fillna("").str.strip()
                vc = vc[vc.ne("")].value_counts()
                print(f"[POWER_DEBUG] league={league} base rows={len(out)}")
                print(f"[POWER_DEBUG] duplicates: {(vc > 1).sum()} | max dup count: {int(vc.max()) if len(vc) else 0}")
                print("[POWER_DEBUG] top dup fixture_keys:\n", vc.head(10))
            else:
                print(f"[POWER_DEBUG] league={league} base rows={len(out)} (no fixture_key column)")
        except Exception as e:
            print(f"[POWER_DEBUG] duplicate check failed: {e}")

    def _power_debug_after(_df: pd.DataFrame, stage: str) -> None:
        """Optional debug: verify power columns after attach."""
        if os.getenv("OG_DEBUG_POWER", "0").strip().lower() not in ("1", "true", "yes", "y"):
            return
        try:
            cols = [
                "home_power_rating",
                "away_power_rating",
                "power_diff",
                "home_power_rating_raw",
                "away_power_rating_raw",
            ]
            present = [c for c in cols if c in _df.columns]
            if not present:
                print(f"[POWER_DEBUG] {stage}: no power cols present")
                return
            print(f"[POWER_DEBUG] {stage}: nonnull share")
            print(_df[present].notna().mean())
            leftovers = [c for c in _df.columns if str(c).endswith("_power")]
            print("[POWER_DEBUG] any *_power left?", leftovers)
            head_cols = [c for c in ["fixture_key", "home_team_name", "away_team_name"] if c in _df.columns] + present
            if head_cols:
                print(_df[head_cols].head(5))
        except Exception as e:
            print(f"[POWER_DEBUG] after-check failed: {e}")


    if pr is None or pr.empty:
        return out

    # Case A: per-match join by fixture_key
    if "fixture_key" in pr.columns:
        if "fixture_key" not in out.columns or out["fixture_key"].astype("string").fillna("").str.strip().eq("").all():
            try:
                out["fixture_key"] = out.apply(_match_key_ascii, axis=1)
            except Exception:
                try:
                    out["fixture_key"] = out.apply(_match_key, axis=1)
                except Exception:
                    out["fixture_key"] = ""

        out["fixture_key"] = out["fixture_key"].astype("string").fillna("").str.strip()
        out = out.merge(pr, on="fixture_key", how="left", suffixes=("", "_power"))

        # Promote *_power columns into canonical columns (handles cases where `out`
        # already had placeholder power columns, causing pandas to suffix incoming values).
        for c in (
            "home_power_rating", "away_power_rating", "power_diff",
            "home_power_rating_raw", "away_power_rating_raw",
        ):
            cp = f"{c}_power"
            if cp in out.columns:
                if c in out.columns:
                    out[c] = pd.to_numeric(out[c], errors="coerce").fillna(pd.to_numeric(out[cp], errors="coerce"))
                else:
                    out[c] = pd.to_numeric(out[cp], errors="coerce")
                out = out.drop(columns=[cp], errors="ignore")

        # Ensure destination columns exist for schema stability (after promotion)
        for c in (
            "home_power_rating", "away_power_rating", "power_diff",
            "home_power_rating_raw", "away_power_rating_raw",
        ):
            if c not in out.columns:
                out[c] = np.nan

        # Drop any remaining power-rating helper columns only
        drop_cols = [
            c for c in out.columns
            if c.endswith("_power") and ("power_rating" in str(c) or str(c) == "power_diff_power")
        ]
        if drop_cols:
            out = out.drop(columns=drop_cols, errors="ignore")

        # If per-match file didn't provide raw columns, backfill raw from scaled
        if ("home_power_rating_raw" not in out.columns) or out["home_power_rating_raw"].isna().all():
            out["home_power_rating_raw"] = out.get("home_power_rating", np.nan)
        if ("away_power_rating_raw" not in out.columns) or out["away_power_rating_raw"].isna().all():
            out["away_power_rating_raw"] = out.get("away_power_rating", np.nan)

        # Compute diff if missing
        if ("power_diff" not in out.columns) or out["power_diff"].isna().all():
            hp = pd.to_numeric(out.get("home_power_rating"), errors="coerce")
            ap = pd.to_numeric(out.get("away_power_rating"), errors="coerce")
            out["power_diff"] = hp - ap

        # Fallback: if per-match PR exists but some fixtures have missing values,
        # backfill missing rows from per-team ratings (by normalized team keys).
        try:
            need_fill = (
                pd.to_numeric(out.get("home_power_rating"), errors="coerce").isna()
                | pd.to_numeric(out.get("away_power_rating"), errors="coerce").isna()
            )
            if bool(need_fill.any()):
                tag = _league_tag(league)
                p_team = modelstore / f"{tag}_team_ratings.csv"
                if p_team.exists():
                    tr = pd.read_csv(p_team)

                    # Identify team column
                    team_col = None
                    for c in ("team", "team_name", "Team", "name", "club", "squad"):
                        if c in tr.columns:
                            team_col = c
                            break

                    # Identify rating columns
                    rating_col = None
                    raw_col = None
                    for c in (
                        "power_rating", "team_power_rating", "rating", "strength", "power", "score",
                        "home_power_rating",
                    ):
                        if c in tr.columns:
                            rating_col = c
                            break

                    for c in (
                        "power_rating_raw", "raw_rating", "raw_strength", "strength_raw",
                        "home_power_rating_raw",
                    ):
                        if c in tr.columns:
                            raw_col = c
                            break

                    if team_col and rating_col:
                        tr = tr.copy()
                        tr[team_col] = tr[team_col].astype("string").fillna("").str.strip()
                        tr = tr[tr[team_col].ne("")].copy()
                        tr["team_key"] = tr[team_col].map(_norm_team_token_ascii).astype("string").fillna("").str.strip()
                        tr = tr[tr["team_key"].ne("")].copy()

                        tr["power_rating"] = pd.to_numeric(tr[rating_col], errors="coerce")
                        if raw_col:
                            tr["power_rating_raw"] = pd.to_numeric(tr[raw_col], errors="coerce")
                        else:
                            tr["power_rating_raw"] = tr["power_rating"]

                        tr = tr[["team_key", "power_rating", "power_rating_raw"]].drop_duplicates(subset=["team_key"], keep="first")
                        mp = tr.set_index("team_key")["power_rating"].to_dict()
                        mr = tr.set_index("team_key")["power_rating_raw"].to_dict()

                        # Ensure names exist
                        if "home_team_name" not in out.columns:
                            out["home_team_name"] = out.get("HomeTeam", "")
                        if "away_team_name" not in out.columns:
                            out["away_team_name"] = out.get("AwayTeam", "")

                        hk = out["home_team_name"].map(_norm_team_token_ascii).astype("string")
                        ak = out["away_team_name"].map(_norm_team_token_ascii).astype("string")

                        hp_fb = pd.to_numeric(hk.map(mp), errors="coerce")
                        ap_fb = pd.to_numeric(ak.map(mp), errors="coerce")
                        hr_fb = pd.to_numeric(hk.map(mr), errors="coerce")
                        ar_fb = pd.to_numeric(ak.map(mr), errors="coerce")

                        # Fill only missing
                        out.loc[need_fill, "home_power_rating"] = pd.to_numeric(out.loc[need_fill, "home_power_rating"], errors="coerce").fillna(hp_fb.loc[need_fill])
                        out.loc[need_fill, "away_power_rating"] = pd.to_numeric(out.loc[need_fill, "away_power_rating"], errors="coerce").fillna(ap_fb.loc[need_fill])
                        out.loc[need_fill, "home_power_rating_raw"] = pd.to_numeric(out.loc[need_fill, "home_power_rating_raw"], errors="coerce").fillna(hr_fb.loc[need_fill])
                        out.loc[need_fill, "away_power_rating_raw"] = pd.to_numeric(out.loc[need_fill, "away_power_rating_raw"], errors="coerce").fillna(ar_fb.loc[need_fill])

                        # Recompute power_diff for rows we filled
                        hp2 = pd.to_numeric(out.get("home_power_rating"), errors="coerce")
                        ap2 = pd.to_numeric(out.get("away_power_rating"), errors="coerce")
                        out.loc[need_fill, "power_diff"] = (hp2 - ap2).loc[need_fill]
        except Exception:
            pass

        _power_debug_after(out, stage="after per-match attach")
        return out

    # Case B: per-team map by normalized team keys
    if "team_key" in pr.columns:
        # Ensure team names exist
        if "home_team_name" not in out.columns:
            out["home_team_name"] = out.get("HomeTeam", "")
        if "away_team_name" not in out.columns:
            out["away_team_name"] = out.get("AwayTeam", "")

        hk = out["home_team_name"].map(_norm_team_token_ascii).astype("string")
        ak = out["away_team_name"].map(_norm_team_token_ascii).astype("string")

        # Create mapping dicts
        mp = pr.set_index("team_key")["power_rating"].to_dict()
        mr = pr.set_index("team_key")["power_rating_raw"].to_dict()

        out["home_power_rating"] = pd.to_numeric(hk.map(mp), errors="coerce")
        out["away_power_rating"] = pd.to_numeric(ak.map(mp), errors="coerce")
        out["home_power_rating_raw"] = pd.to_numeric(hk.map(mr), errors="coerce")
        out["away_power_rating_raw"] = pd.to_numeric(ak.map(mr), errors="coerce")

        hp = pd.to_numeric(out.get("home_power_rating"), errors="coerce")
        ap = pd.to_numeric(out.get("away_power_rating"), errors="coerce")
        out["power_diff"] = hp - ap

        _power_debug_after(out, stage="after per-team attach")
        return out

    # Unknown schema => return with empty cols
    _power_debug_after(out, stage="after attach (unknown schema)")
    return out

# --- End: Power ratings helpers ---


def _predict_proba(bundle: Dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    mdl = bundle.get("model")
    feats = bundle.get("features", [])
    if mdl is None or not feats:
        raise RuntimeError("bundle missing model/features")

    eng = str(bundle.get("engine", "")).strip().lower()
    feats = [str(c) for c in feats]
    X2 = X.copy()

    cat_feature_names: set[str] = set()
    try:
        if hasattr(mdl, "get_cat_feature_indices"):
            cat_idxs = mdl.get_cat_feature_indices()
            cat_feature_names = {
                feats[int(i)]
                for i in cat_idxs
                if isinstance(i, (int, np.integer)) and 0 <= int(i) < len(feats)
            }
    except Exception:
        cat_feature_names = set()

    # Avoid pandas PerformanceWarning (highly fragmented frame) by adding missing
    # columns in one shot instead of repeated `X2[c] = ...` inserts.
    missing = [c for c in feats if c not in X2.columns]
    if missing:
        cat_cols = {"home_team_name", "away_team_name", "League", "league", "league_name"} | cat_feature_names
        add_data: dict[str, object] = {}
        for c in missing:
            add_data[c] = "NA" if c in cat_cols else 0.0
        add_df = pd.DataFrame(add_data, index=X2.index)
        X2 = pd.concat([X2, add_df], axis=1)

    # Order/select features and de-fragment once
    X2 = X2.reindex(columns=feats)
    X2 = X2.copy()

    # CatBoost bundles may learn additional categorical features such as
    # home_formation / away_formation. If those are missing upstream, a raw
    # numeric 0.0 filler will crash predict_proba. Keep categorical columns
    # explicitly string-typed with a stable NA token.
    if cat_feature_names:
        for c in cat_feature_names:
            if c in X2.columns:
                X2[c] = X2[c].astype("string").fillna("NA")

    # XGBoost expects numeric input; coerce to float32
    if eng in ("xgb", "xgboost"):
        try:
            X2 = X2.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        except Exception:
            X2 = np.asarray(X2, dtype=np.float32)

    return np.asarray(mdl.predict_proba(X2))

def ff_first_nonnull(g: pd.Series) -> pd.Series:
    v = g.dropna()
    if len(v) == 0:
        return g  # remains NaN
    return pd.Series([v.iloc[0]] * len(g), index=g.index)

def _canon_btts_market_selection(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalise BTTS market/selection.

    Convention:
      - market is always 'btts'
      - selection is 'YES' or 'NO'

    Back-compat:
      - if older rows come in as market=='btts_no', convert them.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    if "market" not in out.columns:
        return out

    m = out["market"].astype("string").fillna("").str.lower().str.strip()

    # Ensure selection column exists
    if "selection" not in out.columns:
        out["selection"] = pd.NA

    sel = out["selection"].astype("string").fillna("").str.upper().str.strip()

    # Back-compat: market == btts_no  => market=btts, selection=NO
    mask_no = m.eq("btts_no")
    if bool(mask_no.any()):
        out.loc[mask_no, "market"] = "btts"
        empty_sel = sel.eq("") | sel.isna()
        out.loc[mask_no & empty_sel, "selection"] = "NO"

    # If selection empty on BTTS rows but bookie_pick exists, use it
    mask_btts = out["market"].astype("string").fillna("").str.lower().str.strip().eq("btts")
    if "bookie_pick" in out.columns:
        bp = out["bookie_pick"].astype("string").fillna("").str.upper().str.strip()
        empty_sel2 = out["selection"].astype("string").fillna("").str.strip().eq("")
        out.loc[mask_btts & empty_sel2 & bp.isin(["YES", "NO"]), "selection"] = bp

    # Final cleanup: enforce YES/NO tokens
    out.loc[mask_btts, "selection"] = (
        out.loc[mask_btts, "selection"]
        .astype("string")
        .fillna("")
        .str.upper()
        .str.strip()
        .replace({"Y": "YES", "N": "NO"})
    )

    return out

# Helper: canonical odds schema stamping (canonicalise side-market columns for export)
def _stamp_canonical_odds_schema(df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """Final, single-source schema stamping for canonical odds fields.

    Goal: keep export schema stable even if picker outputs change.

    Canonical columns:
        - OU25: odds_ft_over25 / odds_ft_under25
        - BTTS: odds_btts_yes / odds_btts_no
    Side-market raw columns:
        - OU25: od_over / od_under
        - BTTS: od_yes / od_no

    Stamping rules:
        - If market is OU25-like and canonical odds are missing, fill from od_over/od_under.
        - If market is BTTS-like and canonical odds are missing, fill from od_yes/od_no.

    Debug mode:
        - Assert canonical columns exist
        - Print per-market non-null coverage for canonical columns
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df

    # Ensure required columns exist (always)
    canon_cols = [
        "od_over",
        "od_under",
        "od_yes",
        "od_no",
        "odds_ft_over25",
        "odds_ft_under25",
        "odds_btts_yes",
        "odds_btts_no",
    ]

    for c in canon_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Coalesce canonical odds from merge-suffixed columns if present.
    # Some pipelines merge RM/market frames and end up with odds_*_rm columns.
    for base, alt in [
        ("odds_ft_over25", "odds_ft_over25_rm"),
        ("odds_ft_under25", "odds_ft_under25_rm"),
        ("odds_btts_yes", "odds_btts_yes_rm"),
        ("odds_btts_no", "odds_btts_no_rm"),
    ]:
        if alt in df.columns:
            try:
                b = pd.to_numeric(df.get(base, np.nan), errors="coerce")
                a = pd.to_numeric(df.get(alt, np.nan), errors="coerce")
                # Fill only where base is missing
                df[base] = b.fillna(a)
            except Exception:
                pass

    # Normalise market for stamping (do not mutate selection here)
    m = df.get("market", "").astype("string").fillna("").str.strip().str.lower()

    # Coerce odds columns to numeric so fills behave deterministically
    for c in canon_cols:
        try:
            df[c] = pd.to_numeric(df.get(c, np.nan), errors="coerce")
        except Exception:
            # If coercion fails, leave as-is
            pass

    # OU25-like markets: ou25 + (legacy) over25/under25/totals aliases
    m_ou = m.isin(["ou25", "over25", "under25", "ou_25", "o/u25", "ou_2_5", "totals_25", "totals"])
    if bool(m_ou.any()):
        # Fill canonical from side-market cols if canonical missing
        miss_over = m_ou & df["odds_ft_over25"].isna() & df["od_over"].notna()
        miss_under = m_ou & df["odds_ft_under25"].isna() & df["od_under"].notna()
        if bool(miss_over.any()):
            df.loc[miss_over, "odds_ft_over25"] = df.loc[miss_over, "od_over"]
        if bool(miss_under.any()):
            df.loc[miss_under, "odds_ft_under25"] = df.loc[miss_under, "od_under"]

    # BTTS-like markets: btts + legacy btts_no
    m_bt = m.isin(["btts", "btts_no"])
    if bool(m_bt.any()):
        miss_yes = m_bt & df["odds_btts_yes"].isna() & df["od_yes"].notna()
        miss_no = m_bt & df["odds_btts_no"].isna() & df["od_no"].notna()
        if bool(miss_yes.any()):
            df.loc[miss_yes, "odds_btts_yes"] = df.loc[miss_yes, "od_yes"]
        if bool(miss_no.any()):
            df.loc[miss_no, "odds_btts_no"] = df.loc[miss_no, "od_no"]

    if bool(debug):
        # Assertions: canonical columns exist
        missing = [c for c in canon_cols if c not in df.columns]
        if missing:
            raise AssertionError(f"Missing canonical odds columns after stamping: {missing}")

        # Coverage by market
        try:
            _mk = df.get("market", "").astype("string").fillna("").str.strip().str.lower()
            _tmp = df.copy()
            _tmp["_market"] = _mk

            def _cov(col: str) -> pd.Series:
                return _tmp.groupby("_market")[col].apply(lambda s: float(s.notna().mean()) if len(s) else float("nan"))

            cov_over = _cov("odds_ft_over25")
            cov_under = _cov("odds_ft_under25")
            cov_yes = _cov("odds_btts_yes")
            cov_no = _cov("odds_btts_no")

            # Print as % for quick reading
            print("\n[SCHEMA_STAMP] canonical odds coverage by market (% non-null)")
            for mk in sorted(set(_tmp["_market"].tolist())):
                if not str(mk).strip():
                    continue
                po = cov_over.get(mk, float("nan")) * 100.0
                pu = cov_under.get(mk, float("nan")) * 100.0
                py = cov_yes.get(mk, float("nan")) * 100.0
                pn = cov_no.get(mk, float("nan")) * 100.0
                print(f"  {mk}: ft_over25={po:5.1f}% ft_under25={pu:5.1f}% btts_yes={py:5.1f}% btts_no={pn:5.1f}%")
        except Exception as _e_cov:
            print(f"ℹ️ [SCHEMA_STAMP] coverage print skipped: {_e_cov}")

    return df

# ------------------------------------------------------------------
# Export fixups: ensure canonical side probs exist (OU25 no-vig + prob_* aliases)
# ------------------------------------------------------------------

def _ensure_export_side_probs(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee OU25 no-vig columns and canonical prob columns exist for export.

    Fixes:
    - p_over25_novig / p_under25_novig / ou25_overround all-NA in export
    - prob_over25 / prob_btts missing/all-NA when only *_v2 is populated

    Runs AFTER canonical odds stamping so odds_ft_* / odds_btts_* are present.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df

    out = df.copy()

    # 1) Ensure canonical odds columns are filled where possible
    try:
        out = _stamp_canonical_odds_schema(out, debug=False)
    except Exception:
        pass

    # 2) Ensure destination columns exist
    for c in ("ou25_overround", "p_over25_novig", "p_under25_novig", "prob_over25", "prob_btts"):
        if c not in out.columns:
            out[c] = np.nan

    # 3) Compute OU25 overround + no-vig probs if missing/all-NA
    try:
        need_any = (
            out["ou25_overround"].isna().all()
            or out["p_over25_novig"].isna().all()
            or out["p_under25_novig"].isna().all()
            or out["ou25_overround"].isna().any()
            or out["p_over25_novig"].isna().any()
            or out["p_under25_novig"].isna().any()
        )
        if need_any and ("odds_ft_over25" in out.columns) and ("odds_ft_under25" in out.columns):
            oo = pd.to_numeric(out["odds_ft_over25"], errors="coerce")
            ou = pd.to_numeric(out["odds_ft_under25"], errors="coerce")
            imp_o = (1.0 / oo).where(oo > 1.0)
            imp_u = (1.0 / ou).where(ou > 1.0)
            ov = (imp_o + imp_u)

            # Fill row-level gaps where possible (not just all-NA)
            if out["ou25_overround"].isna().any():
                out["ou25_overround"] = out["ou25_overround"].where(
                    out["ou25_overround"].notna(), ov.where(ov > 0)
                )
            if out["p_over25_novig"].isna().any():
                out["p_over25_novig"] = out["p_over25_novig"].where(
                    out["p_over25_novig"].notna(), (imp_o / ov).where(ov > 0)
                )
            if out["p_under25_novig"].isna().any():
                out["p_under25_novig"] = out["p_under25_novig"].where(
                    out["p_under25_novig"].notna(), (imp_u / ov).where(ov > 0)
                )

            out["p_over25_novig"] = pd.to_numeric(out["p_over25_novig"], errors="coerce").clip(0.0, 1.0)
            out["p_under25_novig"] = pd.to_numeric(out["p_under25_novig"], errors="coerce").clip(0.0, 1.0)
            out["ou25_overround"] = pd.to_numeric(out["ou25_overround"], errors="coerce")
    except Exception:
        pass

    # 4) Backfill bookie_implied_novig for OU25 if missing but novig side-probs exist
    try:
        mk = out.get("market", pd.Series(pd.NA, index=out.index, dtype="string")).astype("string").fillna("").str.lower().str.strip()
        sel = out.get("selection", out.get("bookie_pick", pd.Series(pd.NA, index=out.index, dtype="string"))).astype("string").fillna("").str.upper().str.strip()
        m_ou = mk.eq("ou25")
        if "bookie_implied_novig" in out.columns:
            need_bi = m_ou & out["bookie_implied_novig"].isna()
            if bool(need_bi.any()):
                out.loc[need_bi & sel.eq("OVER25"), "bookie_implied_novig"] = out.loc[need_bi & sel.eq("OVER25"), "p_over25_novig"]
                out.loc[need_bi & sel.eq("UNDER25"), "bookie_implied_novig"] = out.loc[need_bi & sel.eq("UNDER25"), "p_under25_novig"]
    except Exception:
        pass

    # 5) Backfill canonical probs from *_v2 if needed
    try:
        if out["prob_over25"].isna().all() and ("prob_over25_v2" in out.columns):
            out["prob_over25"] = pd.to_numeric(out["prob_over25_v2"], errors="coerce")
        if out["prob_btts"].isna().all() and ("prob_btts_v2" in out.columns):
            out["prob_btts"] = pd.to_numeric(out["prob_btts_v2"], errors="coerce")

        out["prob_over25"] = pd.to_numeric(out["prob_over25"], errors="coerce").clip(0.0, 1.0)
        out["prob_btts"] = pd.to_numeric(out["prob_btts"], errors="coerce").clip(0.0, 1.0)
    except Exception:
        pass

    return out

def _coalesce_side_prob_cols_for_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure canonical prob columns exist early enough for signal layers and runtime gates.
    If only v2 aliases exist, backfill:
      - prob_over25 <- prob_over25_v2
      - prob_btts   <- prob_btts_v2
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df

    out = df.copy()

    for c in ("prob_over25", "prob_btts"):
        if c not in out.columns:
            out[c] = np.nan

    try:
        if out["prob_over25"].isna().all() and ("prob_over25_v2" in out.columns):
            out["prob_over25"] = pd.to_numeric(out["prob_over25_v2"], errors="coerce")
        if out["prob_btts"].isna().all() and ("prob_btts_v2" in out.columns):
            out["prob_btts"] = pd.to_numeric(out["prob_btts_v2"], errors="coerce")

        out["prob_over25"] = pd.to_numeric(out["prob_over25"], errors="coerce").clip(0.0, 1.0)
        out["prob_btts"] = pd.to_numeric(out["prob_btts"], errors="coerce").clip(0.0, 1.0)
    except Exception:
        pass

    return out


def _stamp_value_edge_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Stamp additive value-gap fields on scored rows.

    This is descriptive only: it does not alter routing or candidate inclusion.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df

    out = df.copy()

    def _num_col(name: str) -> pd.Series:
        if name in out.columns:
            return pd.to_numeric(out[name], errors="coerce")
        return pd.Series(np.nan, index=out.index)

    model_p = _num_col("model_p_for_bookie")
    implied = _num_col("bookie_implied")

    # Fall back to odds-derived implied probability when needed.
    if implied.isna().all() or implied.isna().any():
        bookie_od = _num_col("bookie_od")
        implied_from_odds = (1.0 / bookie_od).where(bookie_od > 1.0)
        implied = implied.where(implied.notna(), implied_from_odds)

    out["bookie_implied"] = implied
    out["value_edge"] = model_p - implied
    out["value_edge_bps"] = out["value_edge"] * 10000.0
    out["value_gap_pct_points"] = out["value_edge"] * 100.0
    out["value_edge_tier"] = pd.Series("NONE", index=out.index, dtype="string")

    for tier, threshold in sorted(VALUE_EDGE_TIERS.items(), key=lambda kv: kv[1], reverse=True):
        mask = (
            model_p.notna()
            & implied.notna()
            & out["value_edge"].ge(float(threshold))
            & out["value_edge_tier"].astype("string").fillna("").eq("NONE")
        )
        out.loc[mask, "value_edge_tier"] = tier

    return out


def _stamp_team_goal_intelligence_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Attach descriptive team-profile flags and fixture interaction labels.

    This is Phase 1 descriptive context only; it does not alter routing.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df

    out = df.copy()
    if "team_context_label" not in out.columns:
        out["team_context_label"] = pd.Series("", index=out.index, dtype="string")
    if "team_context_filter_family" not in out.columns:
        out["team_context_filter_family"] = pd.Series("GENERAL", index=out.index, dtype="string")
    if "team_context_market_bias" not in out.columns:
        out["team_context_market_bias"] = pd.Series("NONE", index=out.index, dtype="string")
    if "team_context_active_flag" not in out.columns:
        out["team_context_active_flag"] = pd.Series(0, index=out.index, dtype="int64")

    required_cols = {"league", "home_team_name", "away_team_name"}
    if not required_cols.issubset(out.columns):
        for col in TEAM_PROFILE_FLAG_COLUMNS:
            if col not in out.columns:
                out[col] = pd.Series(0, index=out.index, dtype="int64")
        if "team_fixture_interaction" not in out.columns:
            out["team_fixture_interaction"] = pd.Series("OTHER", index=out.index, dtype="string")
        if "team_goal_profile_source" not in out.columns:
            out["team_goal_profile_source"] = pd.Series("", index=out.index, dtype="string")
        out["team_context_label"] = "General Fixture"
        out["team_context_filter_family"] = "GENERAL"
        out["team_context_market_bias"] = "NONE"
        out["team_context_active_flag"] = 0
        return out

    tables = _load_team_profile_stage0_tables()
    if not tables:
        for col in TEAM_PROFILE_FLAG_COLUMNS:
            if col not in out.columns:
                out[col] = pd.Series(0, index=out.index, dtype="int64")
        if "team_fixture_interaction" not in out.columns:
            out["team_fixture_interaction"] = pd.Series("OTHER", index=out.index, dtype="string")
        if "team_goal_profile_source" not in out.columns:
            out["team_goal_profile_source"] = pd.Series("", index=out.index, dtype="string")
        out["team_context_label"] = "General Fixture"
        out["team_context_filter_family"] = "GENERAL"
        out["team_context_market_bias"] = "NONE"
        out["team_context_active_flag"] = 0
        return out

    out["league"] = out["league"].astype("string").fillna("").str.strip()
    out["home_team_name"] = out["home_team_name"].astype("string").fillna("").str.strip()
    out["away_team_name"] = out["away_team_name"].astype("string").fillna("").str.strip()
    out = out.merge(tables["home"], on=["league", "home_team_name"], how="left")
    out = out.merge(tables["away"], on=["league", "away_team_name"], how="left")

    for col in TEAM_PROFILE_FLAG_COLUMNS:
        out[col] = pd.to_numeric(out.get(col, pd.Series(0, index=out.index)), errors="coerce").fillna(0).astype(int)

    home_high = out["home_team_high_scoring_flag"].eq(1)
    away_high = out["away_team_high_scoring_flag"].eq(1)
    home_cs = out["home_team_cs_specialist_flag"].eq(1)
    away_cs = out["away_team_cs_specialist_flag"].eq(1)
    home_fts = out["home_team_fts_risk_flag"].eq(1)
    away_fts = out["away_team_fts_risk_flag"].eq(1)
    home_ge2 = out["home_team_ge2_candidate_flag"].eq(1)
    away_ge2 = out["away_team_ge2_candidate_flag"].eq(1)

    out["team_fixture_interaction"] = pd.Series("OTHER", index=out.index, dtype="string")
    out.loc[home_high & away_high, "team_fixture_interaction"] = "SCORER_VS_SCORER"
    out.loc[home_high & away_cs, "team_fixture_interaction"] = "HOME_SCORER_VS_AWAY_CS"
    out.loc[away_high & home_cs, "team_fixture_interaction"] = "AWAY_SCORER_VS_HOME_CS"
    out.loc[home_cs & away_cs, "team_fixture_interaction"] = "CS_VS_CS"
    out.loc[home_ge2 & away_fts, "team_fixture_interaction"] = "HOME_GE2_POCKET"
    out.loc[away_ge2 & home_fts, "team_fixture_interaction"] = "AWAY_GE2_POCKET"
    out["team_goal_profile_source"] = str(tables.get("source_path", ""))
    out["team_fixture_interaction"] = (
        out["team_fixture_interaction"].astype("string").fillna("OTHER").str.upper().str.strip()
    )
    out.loc[~out["team_fixture_interaction"].isin(TEAM_FIXTURE_INTERACTION_LABELS), "team_fixture_interaction"] = "OTHER"
    out["team_context_label"] = out["team_fixture_interaction"].map(
        lambda x: TEAM_CONTEXT_PRODUCT_MAP.get(str(x), TEAM_CONTEXT_PRODUCT_MAP["OTHER"])["label"]
    ).astype("string")
    out["team_context_filter_family"] = out["team_fixture_interaction"].map(
        lambda x: TEAM_CONTEXT_PRODUCT_MAP.get(str(x), TEAM_CONTEXT_PRODUCT_MAP["OTHER"])["filter"]
    ).astype("string")
    out["team_context_market_bias"] = out["team_fixture_interaction"].map(
        lambda x: TEAM_CONTEXT_PRODUCT_MAP.get(str(x), TEAM_CONTEXT_PRODUCT_MAP["OTHER"])["bias"]
    ).astype("string")
    out["team_context_active_flag"] = out["team_fixture_interaction"].ne("OTHER").astype(int)

    return out


def _btts_side_signal_from_pick_prob(selection: Any, p_pick: Any) -> str:
    """
    Strict row-aware BTTS signal.

    YES row -> label strength on the YES side only
    NO row  -> label strength on the NO side only

    This keeps side labels aligned with the emitted row selection.
    Fixture-level directional disagreement is preserved separately via:
      - model_top_pick
      - signal_btts_fixture
    """
    try:
        pick = str(selection).upper().strip()
        p = float(pd.to_numeric(p_pick, errors="coerce"))
    except Exception:
        return "NEUTRAL"

    if pick not in ("YES", "NO") or (not np.isfinite(p)):
        return "NEUTRAL"

    # Clamp for safety
    p = float(np.clip(p, 0.0, 1.0))

    if p >= 0.70:
        return f"VERY_STRONG_{pick}"
    if p >= 0.60:
        return f"STRONG_{pick}"
    if p >= 0.55:
        return f"WEAK_{pick}"
    return "NEUTRAL"
def _enforce_row_aware_ou25_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure `signal_over25` is ROW-aware for OU25-like rows.

    Rules:
      - If market == 'ou25': direction is derived from emitted pick (bookie_pick OVER25/UNDER25, else selection).
      - If market is legacy ('over25'/'under25'/'o25'/'u25'): direction is derived from the market token.

    This is a rewrite pass: it overwrites signal_over25 for OU25-like rows only, using prob_over25.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df

    out = df.copy()
    if "market" not in out.columns:
        return out

    mk = out["market"].astype("string").fillna("").str.lower().str.strip()

    # OU25-like markets include canonical and legacy tokens.
    m_any = mk.isin(["ou25", "over25", "under25", "o25", "u25", "ou_25", "o/u25", "ou_2_5", "totals_25", "totals"])
    if not bool(m_any.any()):
        return out

    # Prefer bookie_pick, but if blank fall back to selection.
    bp = out.get("bookie_pick", pd.Series(pd.NA, index=out.index, dtype="string")).astype("string").fillna("").str.upper().str.strip()
    sel = out.get("selection", pd.Series(pd.NA, index=out.index, dtype="string")).astype("string").fillna("").str.upper().str.strip()
    pick = bp.mask(bp.eq(""), sel)

    # Side inference
    m_ou = mk.eq("ou25")
    is_over = (m_ou & pick.eq("OVER25")) | mk.isin(["over25", "o25"])
    is_under = (m_ou & pick.eq("UNDER25")) | mk.isin(["under25", "u25"])

    # Mutual exclusivity
    is_under = is_under & (~is_over)

    # Pull best available over-prob
    p_over = pd.to_numeric(out.get("prob_over25", np.nan), errors="coerce")
    if p_over.isna().all() and ("prob_over25_v2" in out.columns):
        p_over = pd.to_numeric(out.get("prob_over25_v2", np.nan), errors="coerce")
    p_over = p_over.clip(0.0, 1.0)
    p_under = (1.0 - p_over).clip(0.0, 1.0)

    if "signal_over25" not in out.columns:
        out["signal_over25"] = "NEUTRAL"

    over_vs = (p_over >= 0.70)
    over_s = (p_over >= 0.62) & (p_over < 0.70)
    over_w = (p_over >= 0.56) & (p_over < 0.62)

    under_vs = (p_under >= 0.70)
    under_s = (p_under >= 0.62) & (p_under < 0.70)
    under_w = (p_under >= 0.56) & (p_under < 0.62)

    # Overwrite only for OU25-like rows
    out.loc[m_any & is_over & over_vs, "signal_over25"] = "VERY_STRONG_OVER"
    out.loc[m_any & is_over & over_s, "signal_over25"] = "STRONG_OVER"
    out.loc[m_any & is_over & over_w, "signal_over25"] = "WEAK_OVER"
    out.loc[m_any & is_over & ~(over_vs | over_s | over_w), "signal_over25"] = "NEUTRAL"

    out.loc[m_any & is_under & under_vs, "signal_over25"] = "VERY_STRONG_UNDER"
    out.loc[m_any & is_under & under_s, "signal_over25"] = "STRONG_UNDER"
    out.loc[m_any & is_under & under_w, "signal_over25"] = "WEAK_UNDER"
    out.loc[m_any & is_under & ~(under_vs | under_s | under_w), "signal_over25"] = "NEUTRAL"

    return out

def _run_strict_qc_asserts(df: pd.DataFrame, *, csv_path: Optional[Path] = None) -> None:
    """Strict QC: hard-fail if key export columns are missing or inconsistent.

    Checks:
      - OU25 rows must have: bookie_implied_novig, p_over25_novig, p_under25_novig, ou25_overround
        and p_over25_novig + p_under25_novig must sum to ~1.
      - BTTS rows must have: bookie_implied_novig, gap_novig, prob_btts, od_yes, od_no
      - FTR rows must have: bookie_implied_novig, imp_home, imp_draw, imp_away, model_top_pick, model_p_for_bookie
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise SystemExit("[STRICT_QC] Empty output dataframe")

    if "market" not in df.columns:
        raise SystemExit("[STRICT_QC] Missing required column: market")

    mk = df["market"].astype("string").fillna("").str.lower().str.strip()

    def _req_nonnull(market: str, cols: list[str]) -> None:
        sub = df.loc[mk.eq(market)]
        if sub.empty:
            return  # market not present in this run
        missing_cols = [c for c in cols if c not in sub.columns]
        if missing_cols:
            raise SystemExit(f"[STRICT_QC] {market}: missing required columns: {missing_cols}")
        miss = {c: int(sub[c].isna().sum()) for c in cols}
        bad = {c: v for c, v in miss.items() if v > 0}
        if bad:
            raise SystemExit(f"[STRICT_QC] {market}: missing non-null values: {bad}")
        print(f"[OK] {market} non-null checks passed for {cols}")

    # OU25 musts + sanity
    # Only enforce novig fields on rows where both sides are present (>1.0).
    ou = df.loc[mk.eq("ou25")]
    if not ou.empty:
        if ("odds_ft_over25" in ou.columns) and ("odds_ft_under25" in ou.columns):
            oo = pd.to_numeric(ou["odds_ft_over25"], errors="coerce")
            uu = pd.to_numeric(ou["odds_ft_under25"], errors="coerce")
            valid = (oo > 1.0) & (uu > 1.0)
            ou_chk = ou.loc[valid]
        else:
            ou_chk = ou

        if not ou_chk.empty:
            missing_cols = [c for c in ["bookie_implied_novig", "p_over25_novig", "p_under25_novig", "ou25_overround"] if c not in ou_chk.columns]
            if missing_cols:
                raise SystemExit(f"[STRICT_QC] ou25: missing required columns: {missing_cols}")
            miss = {c: int(ou_chk[c].isna().sum()) for c in ["bookie_implied_novig", "p_over25_novig", "p_under25_novig", "ou25_overround"]}
            bad = {c: v for c, v in miss.items() if v > 0}
            if bad:
                bad_rows = ou_chk[ou_chk[["bookie_implied_novig", "p_over25_novig", "p_under25_novig", "ou25_overround"]].isna().any(axis=1)]
                print(f"[STRICT_QC] WARNING ou25: dropping {len(bad_rows)} rows with missing novig fields: {bad}")
                if len(bad_rows):
                    print(
                        bad_rows[["league", "match_date", "home_team_name", "away_team_name", "odds_ft_over25", "odds_ft_under25"]]
                        .to_string(index=False)
                    )
                    df.drop(index=bad_rows.index, inplace=True)
            else:
                print("[OK] ou25 non-null checks passed for novig columns")

            s = (pd.to_numeric(ou_chk["p_over25_novig"], errors="coerce") +
                 pd.to_numeric(ou_chk["p_under25_novig"], errors="coerce"))
            max_err = float((s - 1.0).abs().max())
            if not np.isfinite(max_err) or max_err > 1e-6:
                raise SystemExit(f"[STRICT_QC] ou25: novig probs do not sum to 1 (max_err={max_err})")
            print("[OK] ou25 novig probs sum to 1")

    # BTTS musts
    btts = df.loc[mk.eq("btts")].copy()
    if not btts.empty:
        try:
            prod_bt = btts.get("product", pd.Series(pd.NA, index=btts.index, dtype="string")).astype("string").fillna("").str.strip()
            lane_bt = btts.get("model_lane", pd.Series(pd.NA, index=btts.index, dtype="string")).astype("string").fillna("").str.strip()
            btts_model = btts.loc[prod_bt.eq("BTTS_MODEL") & lane_bt.eq("btts_model")].copy()
        except Exception:
            btts_model = btts.copy()

        if not btts_model.empty:
            missing_cols = [
                c for c in [
                    "bookie_implied_novig", "gap_novig", "prob_btts", "od_yes", "od_no",
                    "signal_btts", "signal_btts_fixture", "signal_btts_side",
                ] if c not in btts_model.columns
            ]
            if missing_cols:
                raise SystemExit(f"[STRICT_QC] btts: missing required columns: {missing_cols}")

            miss = {
                c: int(btts_model[c].isna().sum())
                for c in [
                    "bookie_implied_novig", "gap_novig", "prob_btts", "od_yes", "od_no",
                    "signal_btts", "signal_btts_fixture", "signal_btts_side",
                ]
            }
            bad = {c: v for c, v in miss.items() if v > 0}
            if bad:
                raise SystemExit(f"[STRICT_QC] btts: missing non-null values on BTTS_MODEL rows: {bad}")
            print("[OK] btts non-null checks passed for BTTS_MODEL rows")

    # FTR musts
    _req_nonnull("ftr", ["bookie_implied_novig", "imp_home", "imp_draw", "imp_away", "model_top_pick", "model_p_for_bookie"])

    out_path = str(csv_path) if csv_path is not None else "<in-memory>"
    print(f"\nALL ASSERTS PASSED ✅ | strict QC | {out_path}")
    
def _to_decimal_odds(s: pd.Series) -> pd.Series:
    """Parse odds that may be decimal strings or fractional like '7/2' into decimal odds.
    Returns float series with NaN on parse failure.
    """
    if s is None:
        return pd.Series(dtype=float)

    # Fast path: numeric coercion
    out = pd.to_numeric(s, errors="coerce")

    # Fractional fallback: '7/2' => 1 + 7/2
    try:
        ss = s.astype("string").fillna("").str.strip()
        mask = out.isna() & ss.str.contains("/", regex=False)
        if mask.any():
            parts = ss.where(mask, "").str.split("/", n=1, expand=True)
            num = pd.to_numeric(parts[0], errors="coerce")
            den = pd.to_numeric(parts[1], errors="coerce")
            dec = (num / den) + 1.0
            out = out.where(~mask, dec)
    except Exception:
        pass

    return out

def _fill_ftr_bookie_od_from_1x2(df: pd.DataFrame) -> pd.DataFrame:
    """If FTR bookie_od is missing, fill it from od_home/od_draw/od_away based on bookie_pick."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df

    out = df.copy()
    if "market" not in out.columns:
        return out

    mk = out["market"].astype("string").fillna("").str.lower().str.strip()
    m_ftr = mk.eq("ftr")
    if not bool(m_ftr.any()):
        return out

    for c in ("bookie_pick", "bookie_od", "od_home", "od_draw", "od_away", "bookie_implied"):
        if c not in out.columns:
            out[c] = np.nan

    bp = out.loc[m_ftr, "bookie_pick"].astype("string").fillna("").str.upper().str.strip()
    bod = pd.to_numeric(out.loc[m_ftr, "bookie_od"], errors="coerce")

    oh = pd.to_numeric(out.loc[m_ftr, "od_home"], errors="coerce")
    od = pd.to_numeric(out.loc[m_ftr, "od_draw"], errors="coerce")
    oa = pd.to_numeric(out.loc[m_ftr, "od_away"], errors="coerce")

    need = bod.isna()
    if bool(need.any()):
        filled = bod.copy()

        fill_home = need & bp.eq("HOME") & oh.notna()
        fill_draw = need & bp.eq("DRAW") & od.notna()
        fill_away = need & bp.eq("AWAY") & oa.notna()

        filled.loc[fill_home] = oh.loc[fill_home]
        filled.loc[fill_draw] = od.loc[fill_draw]
        filled.loc[fill_away] = oa.loc[fill_away]

        out.loc[m_ftr, "bookie_od"] = filled

    # backfill implied if needed
    try:
        bod2 = pd.to_numeric(out.loc[m_ftr, "bookie_od"], errors="coerce")
        imp = pd.to_numeric(out.loc[m_ftr, "bookie_implied"], errors="coerce")
        imp2 = (1.0 / bod2).where(bod2 > 1.0)
        out.loc[m_ftr, "bookie_implied"] = imp.fillna(imp2)
    except Exception:
        pass

    return out

# ------------------------------------------------------------------
# Final output dtype normalisation (call once right before writing)
# ------------------------------------------------------------------

_INT_FLAG_COLS = [
    # generic pool flags
    "is_fixture_primary",
    "is_market_primary",
    "is_candidate",
    "close_match_flag",
    "candidate_rank",
    "ou25_is_shadow",
    "ou25_is_premium_candidate",
    # FTR glue / trap
    "ftr_ppg_glue_ok",
    "ftr_drawtrap_flag",
    "draw_risk_flag",
    "chaos_risk_flag",
    "not_glue_flag",
    # UEFA flags
    "uefa_rotation_any",
    "uefa_rotation_both",
    "uefa_both_must_win",
    "uefa_goal_hunt_flag",
    "uefa_pride_only_flag",
    "uefa_home_must_win",
    "uefa_away_must_win",
    "uefa_home_must_avoid_loss",
    "uefa_away_must_avoid_loss",
    "uefa_home_eliminated",
    "uefa_away_eliminated",
    "uefa_home_rotation_risk",
    "uefa_away_rotation_risk",
    # bookie/model fit flags
    "bookie_goaliness_fit_ok",
    "tg_pois_ok",
]

_NUMERIC_COLS = [
    # core probs / margins / gaps
    "model_strength",
    "model_p_for_bookie",
    "ftr_margin",
    "gap_novig",
    "bookie_overround",
    "bookie_implied",
    "bookie_implied_novig",
    "bookie_spread",
    "bookie_od",
    # odds / implieds
    "od_home",
    "od_draw",
    "od_away",
    "imp_home",
    "imp_draw",
    "imp_away",
    "od_over",
    "od_under",
    "od_yes",
    "od_no",
    # canonical odds columns (must survive export)
    "odds_ft_over25",
    "odds_ft_under25",
    "odds_btts_yes",
    "odds_btts_no",
    # poisson / goal preds
    "home_goals_pred",
    "away_goals_pred",
    "lambda_home",
    "lambda_away",
    "exp_goals_sum",
    "p00_est",
    "p_home_pois",
    "p_draw_pois",
    "p_away_pois",
    "cs1_p",
    "cs2_p",
    "cs3_p",
    "cs_trunc_mass_0_6",
    "cs_mass_btts_yes",
    "cs_mass_btts_no",
    "cs_mass_over25",
    "cs_mass_under25",
    "cs_mass_home_win",
    "cs_mass_draw",
    "cs_mass_away_win",
    "cs_entropy",
    "both_teams_2plus_mass",
    "mass_over25_via_one_sided_rout",
    "mass_0_goals",
    "mass_1_goal",
    "mass_2_goals",
    "mass_3_goals",
    "mass_4plus_goals",
    "grid_vs_cat_btts_gap",
    "grid_vs_xgb_btts_gap",
    "grid_vs_cat_ou25_gap",
    "grid_vs_xgb_ou25_gap",
    "grid_vs_cat_ftr_gap",
    "grid_vs_xgb_ftr_gap",
    "cat_xgb_grid_btts_agreement_count",
    "cat_xgb_grid_ou25_agreement_count",
    "cat_xgb_grid_ftr_agreement_count",
    # power / diffs
    "home_power_rating",
    "away_power_rating",
    "power_diff",
    "xg_diff_abs",
    "implied_prob_diff",
    "odds_diff",
    # derived canonical side-market probs (version-agnostic)
    "p_over25_novig",
    "p_under25_novig",
    "ou25_overround",
    "prob_over25",
    "prob_btts",
    "prob_over25",
    "prob_btts",
    # legacy aliases kept for backward compatibility (may actually be v3-derived)
    "prob_over25_v2",
    "prob_btts_v2",
    # UEFA numeric context
    "uefa_home_gap24",
    "uefa_away_gap24",
    "uefa_gap24_diff",
    "uefa_live_table_volatility",
    "uefa_vol_band_n",
    "uefa_pressure_sum",
    "uefa_pressure_asym",
    # H2H columns for export
    "h2h_n",
    "h2h_btts_rate",
    "h2h_over25_rate",
    "h2h_goaliness_avg",
]

_STR_COLS = [
    "league",
    "fixture_key",
    "match_date",
    "home_team_name",
    "away_team_name",
    "market",
    "selection",
    "signal_over25",
    "bookie_pick",
    "signal_btts_fixture",
    "signal_btts_side",
    "od_source",
    "pool_tier",
    "ou25_policy_mode",
    "ou25_policy_branch",
    "ou25_policy_state",
    "ou25_shadow_mode",
    "ou25_shadow_model",
    "ou25_runtime_lane",
    "signal_btts",
    "cs1",
    "cs2",
    "cs3",
]


def _coerce_int_flags(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for c in _INT_FLAG_COLS:
        if c not in out.columns:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
    return out


def _coerce_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for c in _NUMERIC_COLS:
        if c not in out.columns:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _coerce_str_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for c in _STR_COLS:
        if c not in out.columns:
            continue
        out[c] = out[c].astype("string").fillna("").str.strip()
    return out


def _finalise_output_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """One-shot schema stabiliser. Call once right before writing."""
    if df is None or df.empty:
        return df
    out = df
    out = _coerce_int_flags(out)
    out = _coerce_numeric_cols(out)
    out = _coerce_str_cols(out)
    return out

def _restore_preserved_signal_cols(df: pd.DataFrame, preserved: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Restore late-stage signal columns after export fixups."""
    if df is None or df.empty or preserved is None or preserved.empty:
        return df

    out = df.copy()

    key_cols = [c for c in ("league", "fixture_key", "market", "bookie_pick", "product", "model_lane") if c in out.columns and c in preserved.columns]
    if len(key_cols) < 6:
        return out

    # IMPORTANT: do NOT restore signal_over25 from preserved.
    # OU25 signals are row-aware and must reflect the current row selection (OVER25 vs UNDER25).
    # Preserved signal_over25 can be fixture-level and may reintroduce direction mismatches.
    keep_cols = key_cols + [c for c in ("signal_btts", "signal_btts_fixture", "signal_btts_side") if c in preserved.columns]
    if len(keep_cols) <= len(key_cols):
        return out

    p = preserved[keep_cols].copy()

    for c in key_cols:
        p[c] = p[c].astype("string").fillna("").str.strip()
        out[c] = out[c].astype("string").fillna("").str.strip()

    try:
        p = p.drop_duplicates(subset=key_cols, keep="first")
    except Exception:
        pass

    merged = out.merge(p, on=key_cols, how="left", suffixes=("", "__preserved"))

    for c in ("signal_btts", "signal_btts_fixture", "signal_btts_side"):
        cp = f"{c}__preserved"
        if cp in merged.columns:
            if c in merged.columns:
                cur = merged[c].astype("string").fillna("").str.strip()
                prv = merged[cp].astype("string").fillna("").str.strip()
                merged[c] = cur.mask(prv.ne(""), prv)
            else:
                merged[c] = merged[cp]
            merged = merged.drop(columns=[cp], errors="ignore")

    return merged

def _best_prob_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a sane `match_date` column exists.

    Some league CSVs do not have `match_date` but do have `date_GMT` (e.g., UEFA comps).
    Previous logic created an all-NA `match_date` and then re-selected it, causing empty windows.
    """
    out = df.copy()

    # If match_date exists, normalize empties to NA; otherwise we'll fill it from other date columns.
    if "match_date" in out.columns:
        try:
            s = out["match_date"].astype("string").str.strip()
            out["match_date"] = s.mask(s.eq(""), pd.NA)
        except Exception:
            pass

    # Always (re)populate match_date from the best available date-like column
    out["match_date"] = _coalesce_match_date_series(out)
    return out

def _filter_primary_btts_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep exported BTTS rows to the canonical upstream deploy contract only.

    Contract for BTTS export:
      - exactly one BTTS row per fixture
      - market == 'btts'
      - selection/bookie_pick in {'YES', 'NO'}
      - is_fixture_primary == 1
      - product forced to 'BTTS_MODEL'
      - model_lane forced to 'btts_model'

    Non-BTTS rows pass through unchanged.
    """
    if df is None or df.empty or ("market" not in df.columns):
        return df

    out = df.copy()

    m_bt = out["market"].astype("string").fillna("").str.lower().str.strip().eq("btts")
    bt = out.loc[m_bt].copy()
    non_bt = out.loc[~m_bt].copy()

    if bt.empty:
        return out

    # Normalise canonical BTTS identity first
    if "selection" not in bt.columns:
        bt["selection"] = bt.get("bookie_pick", pd.NA)
    bt["selection"] = (
        bt["selection"]
        .astype("string")
        .fillna("")
        .str.upper()
        .str.strip()
        .replace({"Y": "YES", "N": "NO"})
    )

    if "bookie_pick" not in bt.columns:
        bt["bookie_pick"] = bt["selection"]
    bt["bookie_pick"] = (
        bt["bookie_pick"]
        .astype("string")
        .fillna("")
        .str.upper()
        .str.strip()
    )
    bt.loc[bt["selection"].isin(["YES", "NO"]), "bookie_pick"] = bt.loc[
        bt["selection"].isin(["YES", "NO"]), "selection"
    ]

    prim = pd.to_numeric(bt.get("is_fixture_primary", 0), errors="coerce").fillna(0).astype(int)

    # Keep only the canonical primary decision row per fixture.
    bt = bt.loc[
        prim.eq(1)
        & bt["selection"].isin(["YES", "NO"])
        & bt["bookie_pick"].isin(["YES", "NO"])
    ].copy()

    if bt.empty:
        return pd.concat([bt, non_bt], ignore_index=True)

    # Force canonical BTTS contract; BTTS_VALUEEV is retired.
    bt["product"] = "BTTS_MODEL"
    bt["model_lane"] = "btts_model"
    bt["market"] = "btts"
    bt["is_fixture_primary"] = 1

    dedupe_keys = [
        c for c in [
            "league",
            "fixture_key",
            "market",
        ] if c in bt.columns
    ]
    if dedupe_keys:
        for c in dedupe_keys:
            bt[c] = bt[c].astype("string").fillna("").str.strip()
        bt = bt.drop_duplicates(subset=dedupe_keys, keep="first")

    return pd.concat([bt, non_bt], ignore_index=True)

def _apply_feature_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight feature alias / rename layer.

    Purpose:
      - Some league merge files use legacy headers (e.g. "Pre-Match PPG (Home)")
        while model bundles expect canonical snake_case feature names.
      - This layer standardizes the minimum core features used across bundles.

    Policy:
      - Never delete columns.
      - Only add canonical columns when missing.
      - Prefer already-canonical columns when present.
      - Keep operations cheap (vectorized; no row-wise apply).
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    def _first_present(*names: str) -> str:
        for n in names:
            if n and (n in out.columns):
                return n
        return ""

    def _ensure_numeric(dst: str, *srcs: str) -> None:
        if dst in out.columns and out[dst].notna().any():
            return
        src = _first_present(*srcs)
        if not src:
            return
        out[dst] = pd.to_numeric(out[src], errors="coerce")

    def _ensure_str(dst: str, *srcs: str) -> None:
        if dst in out.columns and out[dst].astype("string").fillna("").str.strip().ne("").any():
            return
        src = _first_present(*srcs)
        if not src:
            return
        out[dst] = out[src].astype("string")

    def _pct_from_src(src: str) -> pd.Series:
        vals = pd.to_numeric(out[src], errors="coerce")
        if vals.dropna().empty:
            return vals
        try:
            max_abs = float(vals.dropna().abs().max())
        except Exception:
            max_abs = float("nan")
        if np.isfinite(max_abs) and max_abs <= 1.000001:
            return vals * 100.0
        return vals

    # --- Core team name + date aliases ---
    _ensure_str("home_team_name", "home_team_name", "HomeTeam", "home", "home_team", "home_team_name_x")
    _ensure_str("away_team_name", "away_team_name", "AwayTeam", "away", "away_team", "away_team_name_x")

    # --- match_date coalesce (row-wise) ---
    # Some files have a partially-populated match_date but a fully-populated date_GMT.
    # Keep existing match_date where present; otherwise fill from the best available date-like column.
    if "match_date" not in out.columns:
        out["match_date"] = pd.NA

    # Normalize blanks to NA so fill works deterministically
    try:
        md0 = out["match_date"].astype("string").str.strip()
        out["match_date"] = md0.mask(md0.eq(""), pd.NA)
    except Exception:
        pass

    # Best available fallback series (date_GMT/date/timestamp etc)
    md_fallback = _coalesce_match_date_series(out)

    # Fill only missing/blank rows
    try:
        md_fb = pd.Series(md_fallback, index=out.index)
        out["match_date"] = out["match_date"].combine_first(md_fb)
    except Exception:
        # best effort
        out["match_date"] = out["match_date"].fillna(md_fallback)

    # --- Fixture key (needed for power ratings join) ---
    # Ensure we have an OG-style fixture_key: YYYY_MM_DD_HOME_AWAY (ASCII-folded tokens)
    if "fixture_key" not in out.columns:
        out["fixture_key"] = pd.NA

    fk = out["fixture_key"].astype("string").fillna("").str.strip()
    need_fk = fk.eq("") | fk.isna()

    if bool(need_fk.any()):
        # Parse match_date into YYYY_MM_DD
        try:
            md = pd.to_datetime(out["match_date"], errors="coerce", utc=True, format="mixed", cache=True)
        except TypeError:
            md = pd.to_datetime(out["match_date"], errors="coerce", utc=True, cache=True)
        ds = md.dt.strftime("%Y_%m_%d").fillna("")

        # Normalize team names (diacritics-safe)
        h = out["home_team_name"].map(_norm_team_token_ascii).astype("string").fillna("").str.strip()
        a = out["away_team_name"].map(_norm_team_token_ascii).astype("string").fillna("").str.strip()

        out.loc[need_fk, "fixture_key"] = (ds + "_" + h + "_" + a).str.strip("_")

    # --- Pre-match PPG aliases ---
    _ensure_numeric("pre_match_ppg_home", "pre_match_ppg_home", "Pre-Match PPG (Home)", "Pre-Match PPG Home", "ppg_home_pre", "home_ppg", "ppg_home", "pre_match_ppg_h")
    _ensure_numeric("pre_match_ppg_away", "pre_match_ppg_away", "Pre-Match PPG (Away)", "Pre-Match PPG Away", "ppg_away_pre", "away_ppg", "ppg_away", "pre_match_ppg_a")

    # --- Pre-match xG aliases ---
    _ensure_numeric("pre_match_xg_home", "pre_match_xg_home", "Home Team Pre-Match xG", "team_a_xg", "home_xg", "xg_home")
    _ensure_numeric("pre_match_xg_away", "pre_match_xg_away", "Away Team Pre-Match xG", "team_b_xg", "away_xg", "xg_away")
    _ensure_numeric("btts_percentage_pre_match", "btts_percentage_pre_match", "btts_pct_pre", "avg_btts_rate")

    # --- Recent BTTS regime counterweights (prefer true rolling/API fields; fallback to merged rolling rates) ---
    if ("recent_btts_regime_blend_l5" not in out.columns) or out["recent_btts_regime_blend_l5"].isna().all():
        src = _first_present("combined_btts_rate_l5")
        if src:
            out["recent_btts_regime_blend_l5"] = _pct_from_src(src).round(4)
        else:
            h5 = _first_present("home_btts_rate_l5", "btts_rate_5_home")
            a5 = _first_present("away_btts_rate_l5", "btts_rate_5_away")
            if h5 and a5:
                out["recent_btts_regime_blend_l5"] = pd.concat([_pct_from_src(h5), _pct_from_src(a5)], axis=1).mean(axis=1, skipna=True).round(4)

    if ("recent_btts_regime_blend_l10" not in out.columns) or out["recent_btts_regime_blend_l10"].isna().all():
        h10 = _first_present("home_btts_rate_l10", "btts_rate_10_home")
        a10 = _first_present("away_btts_rate_l10", "btts_rate_10_away")
        if h10 and a10:
            out["recent_btts_regime_blend_l10"] = pd.concat([_pct_from_src(h10), _pct_from_src(a10)], axis=1).mean(axis=1, skipna=True).round(4)

    if "recent_btts_regime_blend_l5" in out.columns and (("recent_no_btts_regime_blend_l5" not in out.columns) or out["recent_no_btts_regime_blend_l5"].isna().all()):
        out["recent_no_btts_regime_blend_l5"] = (100.0 - pd.to_numeric(out["recent_btts_regime_blend_l5"], errors="coerce")).round(4)
    if "recent_btts_regime_blend_l10" in out.columns and (("recent_no_btts_regime_blend_l10" not in out.columns) or out["recent_no_btts_regime_blend_l10"].isna().all()):
        out["recent_no_btts_regime_blend_l10"] = (100.0 - pd.to_numeric(out["recent_btts_regime_blend_l10"], errors="coerce")).round(4)

    # --- OU25 no-vig / overround (needed by some bundles) ---
    # If missing, compute from OU25 odds when possible.
    # Ensure canonical odds columns are populated even if merges created *_rm suffixes.
    try:
        out = _stamp_canonical_odds_schema(out, debug=False)
    except Exception:
        pass

    need_p = ("p_over25_novig" not in out.columns) or out["p_over25_novig"].isna().all()
    need_u = ("p_under25_novig" not in out.columns) or out["p_under25_novig"].isna().all()
    need_or = ("ou25_overround" not in out.columns) or out["ou25_overround"].isna().all()

    if (need_p or need_u or need_or) and ("odds_ft_over25" in out.columns) and ("odds_ft_under25" in out.columns):
        oo = pd.to_numeric(out["odds_ft_over25"], errors="coerce")
        ou = pd.to_numeric(out["odds_ft_under25"], errors="coerce")
        imp_o = (1.0 / oo).where(oo > 1.0)
        imp_u = (1.0 / ou).where(ou > 1.0)
        ov = (imp_o + imp_u)

        if need_or:
            out["ou25_overround"] = ov.where(ov > 0)
        if need_p:
            out["p_over25_novig"] = (imp_o / ov).where(ov > 0)
        if need_u:
            out["p_under25_novig"] = (imp_u / ov).where(ov > 0)

    # --- Early de-duplication by fixture_key ---
    # Some league sources contain repeated rows per fixture (often ~11x duplicates).
    # Keep one row per fixture_key, preferring rows with a real match_date,
    # then tie-break by "most odds columns non-null", else keep first.
    if "fixture_key" in out.columns:
        fk2 = out["fixture_key"].astype("string").fillna("").str.strip()
        m_fk = fk2.ne("")
        if bool(m_fk.any()):
            tmp = out.loc[m_fk].copy()

            # Prefer rows where match_date parses
            md_parsed = pd.to_datetime(tmp.get("match_date", pd.NA), errors="coerce", utc=True, format="mixed")
            tmp["_has_md"] = md_parsed.notna().astype(int)

            # Odds completeness proxy: count non-null across common odds columns present
            odds_cols = [
                c for c in tmp.columns
                if (
                    ("odds" in str(c).lower())
                    or str(c).lower().startswith("fd_odds_")
                    or str(c).lower().startswith("od_")
                    or str(c).lower() in ("od_home", "od_draw", "od_away", "od_over", "od_under", "od_yes", "od_no")
                )
            ]
            if odds_cols:
                tmp["_odds_nnz"] = tmp[odds_cols].notna().sum(axis=1).astype(int)
            else:
                tmp["_odds_nnz"] = 0

            # Stable sort then keep best row per fixture
            tmp["_fk"] = fk2.loc[m_fk].astype("string")
            tmp = tmp.sort_values(["_fk", "_has_md", "_odds_nnz"], ascending=[True, False, False], kind="mergesort")
            tmp = tmp.drop_duplicates(subset=["_fk"], keep="first")

            # Recombine with any blank-fixture_key rows (kept as-is)
            tmp = tmp.drop(columns=["_fk", "_has_md", "_odds_nnz"], errors="ignore")
            out_blank = out.loc[~m_fk].copy()
            out = pd.concat([tmp, out_blank], axis=0, ignore_index=True)

    return out

def _pick_best_date_col(df: pd.DataFrame) -> Optional[str]:
    """Pick the best available date-like column for windowing."""

    if df is None:
        return None

    # Header-only / empty frame: we cannot inspect value presence.
    # Prefer raw source date columns over match_date because many CSVs
    # include a blank/placeholder match_date column.
    try:
        if getattr(df, "empty", False) or (hasattr(df, "shape") and int(df.shape[0]) == 0):
            for c in ("date_GMT", "date", "Date", "timestamp", "match_date"):
                if c in df.columns:
                    return c
            return None
    except Exception:
        pass

    # Non-empty frames: use match_date only if it has usable values.
    if "match_date" in df.columns:
        try:
            s0 = df["match_date"].astype("string").str.strip()
            s0 = s0.mask(s0.eq(""), pd.NA)
            if bool(s0.notna().any()):
                return "match_date"
        except Exception:
            pass

    # Otherwise fall back to raw date columns
    for c in ("date_GMT", "date", "Date", "timestamp"):
        if c in df.columns:
            return c

    return None


def _parse_window_dates(s: pd.Series, date_col: str) -> pd.Series:
    """Robust date parser shared by window filter + source-file scoring."""
    if str(date_col) == "timestamp":
        ts = pd.to_numeric(s, errors="coerce")
        mx = float(ts.max()) if ts.notna().any() else float("nan")
        if np.isfinite(mx):
            unit = "ms" if mx > 1.0e11 else "s"
            return pd.to_datetime(ts, errors="coerce", utc=True, unit=unit)
        try:
            return pd.to_datetime(s, errors="coerce", utc=True, format="mixed", cache=True)
        except TypeError:
            return pd.to_datetime(s, errors="coerce", utc=True, cache=True)

    try:
        return pd.to_datetime(s, errors="coerce", utc=True, format="mixed", cache=True)
    except TypeError:
        return pd.to_datetime(s, errors="coerce", utc=True, cache=True)
    
def _filter_window(df: pd.DataFrame, date_from: str, date_to: str) -> pd.DataFrame:
    date_col = _pick_best_date_col(df)
    if not date_col:
        return df.iloc[0:0].copy()

    s = df[date_col]
    md = _parse_window_dates(s, date_col)

    lo = pd.Timestamp(date_from, tz="UTC")
    hi_ex = pd.Timestamp(date_to, tz="UTC") + pd.Timedelta(days=1)  # inclusive date_to

    m = md.notna() & (md >= lo) & (md < hi_ex)
    out = df.loc[m].copy()

    # Standardize a match_date column for downstream logic
    out["match_date"] = md.loc[m].dt.strftime("%Y-%m-%d")
    return out


def _count_rows_in_window(csv_path: Path, date_from: str, date_to: str) -> int:
    """Best-effort count of rows in [date_from, date_to] for a given CSV.

    We only read a single date-like column to keep this cheap.
    Returns 0 if no date column is present or parsing fails.
    """
    try:
        if (csv_path is None) or (not csv_path.exists()) or (not csv_path.is_file()):
            return 0

        # Read header only to discover columns
        hdr = pd.read_csv(csv_path, nrows=0)
        cols = [str(c).strip() for c in hdr.columns]

        # REFACTORED: Use _pick_best_date_col and _parse_window_dates
        tmp_hdr = pd.DataFrame(columns=cols)
        date_col = _pick_best_date_col(tmp_hdr)
        if not date_col:
            return 0

        s = pd.read_csv(csv_path, usecols=[date_col], low_memory=False)[date_col]
        md = _parse_window_dates(s, date_col)

        lo = pd.Timestamp(date_from, tz="UTC")
        hi_ex = pd.Timestamp(date_to, tz="UTC") + pd.Timedelta(days=1)

        m = md.notna() & (md >= lo) & (md < hi_ex)
        return int(m.sum())
    except Exception:
        return 0

def _count_valid_1x2_rows_in_window(csv_path: Path, date_from: str, date_to: str) -> int:
    """Count rows in [date_from, date_to] where 1X2 odds are present and > 1.0.

    IMPORTANT:
      Uses _resolve_odds_cols on the file's header columns so we count valid 1X2 rows
      even when odds are stored under aliases (odds_home/odds_draw/odds_away etc).
    """
    try:
        hdr = pd.read_csv(csv_path, nrows=0)
        cols = [str(c).strip() for c in hdr.columns]
    except Exception:
        return 0

    tmp_hdr = pd.DataFrame(columns=cols)
    date_col = _pick_best_date_col(tmp_hdr)
    if not date_col:
        return 0

    # Resolve which columns represent 1X2 in THIS file
    try:
        tmp = pd.DataFrame(columns=cols)
        odc = _resolve_odds_cols(tmp)
        hcol = str(odc.get("ftr_home") or "")
        dcol = str(odc.get("ftr_draw") or "")
        acol = str(odc.get("ftr_away") or "")
        if (not hcol) or (not dcol) or (not acol):
            return 0
    except Exception:
        return 0

    usecols = [c for c in (date_col, hcol, dcol, acol) if c in cols]
    if len(usecols) < 4:
        return 0

    try:
        df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
        if df.empty:
            return 0

        md = _parse_window_dates(df[date_col], date_col)

        lo = pd.Timestamp(date_from, tz="UTC")
        hi_ex = pd.Timestamp(date_to, tz="UTC") + pd.Timedelta(days=1)
        inwin = md.notna() & (md >= lo) & (md < hi_ex)
        if not bool(inwin.any()):
            return 0

        oh = pd.to_numeric(df[hcol], errors="coerce")
        od = pd.to_numeric(df[dcol], errors="coerce")
        oa = pd.to_numeric(df[acol], errors="coerce")

        good = inwin & oh.notna() & od.notna() & oa.notna() & (oh > 1.0) & (od > 1.0) & (oa > 1.0)
        return int(good.sum())
    except Exception:
        return 0

def _count_valid_side_rows_in_window(csv_path: Path, date_from: str, date_to: str) -> int:
    """Score rows in [date_from, date_to] with usable side-market odds.

    This is intentionally richer than a simple row count:
      - +1 for usable O2.5 odds
      - +1 for usable U2.5 odds
      - +1 for usable BTTS YES odds
      - +1 for usable BTTS NO odds

    Higher scores indicate files that are better suited for BTTS/OU25 source use.
    """
    try:
        hdr = pd.read_csv(csv_path, nrows=0)
        cols = [str(c).strip() for c in hdr.columns]
    except Exception:
        return 0

    tmp_hdr = pd.DataFrame(columns=cols)
    date_col = _pick_best_date_col(tmp_hdr)
    if not date_col:
        return 0

    try:
        odc = _resolve_odds_cols(tmp_hdr)
    except Exception:
        return 0

    side_cols = [
        str(odc.get("ou25_over") or ""),
        str(odc.get("ou25_under") or ""),
        str(odc.get("btts_yes") or ""),
        str(odc.get("btts_no") or ""),
    ]
    side_cols = [c for c in side_cols if c]
    if not side_cols:
        return 0

    usecols = [c for c in [date_col, *side_cols] if c in cols]
    if len(usecols) <= 1:
        return 0

    try:
        df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
        if df.empty:
            return 0

        md = _parse_window_dates(df[date_col], date_col)
        lo = pd.Timestamp(date_from, tz="UTC")
        hi_ex = pd.Timestamp(date_to, tz="UTC") + pd.Timedelta(days=1)
        inwin = md.notna() & (md >= lo) & (md < hi_ex)
        if not bool(inwin.any()):
            return 0

        score = 0
        for c in side_cols:
            if c not in df.columns:
                continue
            odds = pd.to_numeric(df[c], errors="coerce")
            score += int((inwin & odds.notna() & (odds > 1.0)).sum())
        return int(score)
    except Exception:
        return 0

def _pick_latest_matches_csv(
    matches_root: Path,
    league: str,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    *,
    require_ftr_1x2: bool = False,
) -> Optional[Path]:
    """Pick the best matches CSV for a league.

    Previous behavior always preferred `fd_odds_enriched.csv` when present.
    That can accidentally drop forward windows if that file doesn't actually contain
    the requested date range.

    New behavior:
      - If date_from/date_to are provided, choose the file with the most rows inside
        the requested window (tie-break by preference order).
      - Otherwise, prefer `fd_odds_enriched_synth.csv` (if present), then
        `fd_odds_enriched.csv`, then the newest *.csv in the folder.
      - When require_ftr_1x2=True (and a window is provided), prefer the file with the most valid 1X2 odds rows in-window.
    """
    # NOTE: league folder names are not always identical to the league label.
    # Try a small set of safe fallbacks (e.g. "UEFA Champions League" -> "Champions League").
    league = str(league).strip()
    mdir = matches_root / league

    if not mdir.exists():
        alts: list[str] = []
        # common prefix noise
        if league.lower().startswith("uefa "):
            alts.append(league[5:].strip())
        # common naming variants
        alts.append(league.replace("UEFA ", "").strip())
        alts.append(league.replace("UEFA", "").strip())

        # de-dupe while preserving order
        seen = set([league])
        for alt in alts:
            alt2 = str(alt or "").strip()
            if not alt2 or alt2 in seen:
                continue
            seen.add(alt2)
            cand = matches_root / alt2
            if cand.exists():
                mdir = cand
                league = alt2
                break

    if not mdir.exists():
        return None

    # Optional debug: show which folder we resolved
    try:
        if os.getenv("OG_DEBUG_PICK", "0").strip().lower() in ("1", "true", "yes", "y"):
            print(f"[pick_csv] league_label='{league}' resolved_dir='{mdir}'")
    except Exception:
        pass
    # Candidate list: newest first
    cands = sorted([p for p in mdir.glob("*.csv") if p.is_file()], key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        return None

    # Stable preference order (lower is better) for tie-breaks.
    # IMPORTANT:
    #   - For FTR (require_ftr_1x2=True), prefer the *real* enriched file first because
    #     synth files frequently lack 1X2 odds coverage.
    #   - For side markets / general windows, keep the original preference for synth.
    if require_ftr_1x2:
        pref_rank: dict[str, int] = {
            "fd_odds_enriched.csv": 0,
            "fd_odds_enriched_synth.csv": 1,
        }
    else:
        pref_rank = {
            "fd_odds_enriched_synth.csv": 0,
            "fd_odds_enriched.csv": 1,
        }

    def _rank(p: Path) -> int:
        return int(pref_rank.get(p.name, 10_000))

    # If a window is provided, pick the best file for that window.
    # When require_ftr_1x2=True, prefer the file with the most *valid* 1X2 rows.
    if date_from and date_to:
        best = None
        best_valid_1x2 = -1
        best_n = -1
        best_r = 10_000

        for p in cands:
            n = _count_rows_in_window(p, str(date_from), str(date_to))
            if n <= 0:
                continue
            r = _rank(p)
            v = _count_valid_1x2_rows_in_window(p, str(date_from), str(date_to)) if require_ftr_1x2 else 0

            # Primary: valid 1X2 rows when required
            if require_ftr_1x2:
                if (v > best_valid_1x2) or (v == best_valid_1x2 and n > best_n) or (v == best_valid_1x2 and n == best_n and r < best_r):
                    best = p
                    best_valid_1x2 = v
                    best_n = n
                    best_r = r
            else:
                # Original behavior: most rows in window (tie-break by preference)
                if (n > best_n) or (n == best_n and r < best_r):
                    best = p
                    best_n = n
                    best_r = r

        # Trust the window-based pick only if it actually contains usable rows.
        if best is not None:
            if require_ftr_1x2:
                # Prefer a file with at least one valid 1X2 row in-window
                if best_valid_1x2 > 0:
                    return best
                # Fallback: still return the best window file so side markets can run
                # and any partially-odds-populated rows can still emit.
                if best_n > 0:
                    return best
            else:
                if best_n > 0:
                    return best

        # No candidate file contains any rows in the requested window
        return None

    # Fallback: prefer synth-enriched, then fd-enriched, else newest file
    for name in ("fd_odds_enriched_synth.csv", "fd_odds_enriched.csv"):
        p = mdir / name
        if p.exists() and p.is_file():
            return p

    return cands[0]


def _resolve_odds_cols(df: pd.DataFrame) -> Dict[str, str]:
    """Resolve odds column names for the current dataframe.

    Different leagues/files can use slightly different headers.
    We choose the first candidate that exists for each required odds field.
    """

    def pick(*cands: str) -> str:
        for c in cands:
            if c and c in df.columns:
                return c
        return ""  # empty => not available

    # NOTE: We include legacy/synth alias names here because some league CSVs
    # store UNDER2.5 (and sometimes other odds) under alternative headers.
    return {
        # 1X2
        "ftr_home": pick(
            "odds_ft_home_team_win",
            "odds_home",
            "odds_ft_home",
            "odds_1",
            "odds_ft_1",
        ),
        "ftr_draw": pick(
            "odds_ft_draw",
            "odds_draw",
            "odds_ft_x",
            "odds_x",
        ),
        "ftr_away": pick(
            "odds_ft_away_team_win",
            "odds_away",
            "odds_ft_away",
            "odds_2",
            "odds_ft_2",
        ),
        # Over 2.5
        "ou25_over": pick(
            "odds_ft_over25",
            "odds_over25",
            "odds_over_2_5",
            "odds_ft_over_2_5",
            "odds_ou_25_over",
            # legacy / synth aliases
            # "odds_source_over25",
            # "odds_source_over_2_5",
            # "odds_source_u25_over",
            "odds_u25_over",
            "od_u25_over",
            "synth_odds_ft_over25",
            "synth_odds_over25",
        ),
        # Under 2.5
        "ou25_under": pick(
            "odds_ft_under25",
            "odds_under25",
            "odds_under_2_5",
            "odds_ft_under_2_5",
            "odds_ou_25_under",
            # legacy / synth aliases
            # "odds_source_under25",
            # "odds_source_under_2_5",
            # "odds_source_u25_under",
            "odds_u25_under",
            "od_u25_under",
            "synth_odds_ft_under25",
            "synth_odds_under25",
        ),
        # BTTS Yes
        "btts_yes": pick(
            "odds_btts_yes",
            "odds_ft_btts_yes",
            "odds_btts",
            "odds_btts_y",
            # legacy / synth aliases
            # "odds_source_btts_yes",
            # "odds_source_btts",
        ),
        # BTTS No
        "btts_no": pick(
            "odds_btts_no",
            "odds_ft_btts_no",
            "odds_btts_n",
            "odds_btts_noo",
            # legacy / synth aliases
            # "odds_source_btts_no",
        ),
        # Totals (for goaliness fit)
        "tg_over15": pick(
            "odds_ft_over15",
            "odds_over15",
            "odds_over_1_5",
            "odds_ft_over_1_5",
        ),
        "tg_over35": pick(
            "odds_ft_over35",
            "odds_over35",
            "odds_over_3_5",
            "odds_ft_over_3_5",
        ),
        "tg_over45": pick(
            "odds_ft_over45",
            "odds_over45",
            "odds_over_4_5",
            "odds_ft_over_4_5",
        ),
    }


# --- Begin: Goaliness and lambda fit helpers ---

def _num_from_row(row: pd.Series, *cands: str) -> float:
    """Best-effort numeric fetch from a row for any of the candidate column names."""
    for c in cands:
        if not c:
            continue
        try:
            v = pd.to_numeric(row.get(c, np.nan), errors="coerce")
            v = float(v) if v is not None else np.nan
        except Exception:
            v = np.nan
        if np.isfinite(v):
            return float(v)
    return float("nan")


def _resolve_pre_match_ppg_home(row: pd.Series) -> float:
    return _num_from_row(
        row,
        "pre_match_ppg_home",
        "Pre-Match PPG (Home)",
        "Pre-Match PPG Home",
        "ppg_home_pre",
        "home_ppg",
        "ppg_home",
    )


def _resolve_pre_match_ppg_away(row: pd.Series) -> float:
    return _num_from_row(
        row,
        "pre_match_ppg_away",
        "Pre-Match PPG (Away)",
        "Pre-Match PPG Away",
        "ppg_away_pre",
        "away_ppg",
        "ppg_away",
    )


def _resolve_pre_match_xg_home(row: pd.Series) -> float:
    return _num_from_row(
        row,
        "pre_match_xg_home",
        "Home Team Pre-Match xG",
        "team_a_xg",
        "home_xg",
        "xg_home",
    )


def _resolve_pre_match_xg_away(row: pd.Series) -> float:
    return _num_from_row(
        row,
        "pre_match_xg_away",
        "Away Team Pre-Match xG",
        "team_b_xg",
        "away_xg",
        "xg_away",
    )


def _resolve_btts_pct_pre(row: pd.Series) -> float:
    return _num_from_row(
        row,
        "btts_percentage_pre_match",
        "btts_pct_pre",
        "avg_btts_rate",
    )

# --- H2H leak-safe rate helper ---
def _h2h_rate(row: pd.Series, rate_col: str, n_col: str = "h2h_n", min_n: int = 3) -> float:
    """Return a H2H rate only when sample size is sufficient; else NaN."""
    try:
        n = pd.to_numeric(row.get(n_col, np.nan), errors="coerce")
        n = float(n) if n is not None else np.nan
    except Exception:
        n = np.nan
    if not np.isfinite(n) or n < float(min_n):
        return float("nan")
    return _num_from_row(row, rate_col)

def _norm_pct01(x: float) -> float:
    """Normalize a percentage-like value to 0..1 when it looks like 0..100."""
    try:
        xv = float(x)
    except Exception:
        return np.nan
    if not np.isfinite(xv):
        return np.nan
    if xv > 1.0 and xv <= 100.0:
        return float(xv / 100.0)
    return float(xv)

def _poisson_tail_ge(lam: float, k: int) -> float:
    """P(X >= k) for X~Poisson(lam). k is integer >= 0."""
    try:
        lam = float(lam)
        k = int(k)
    except Exception:
        return np.nan
    if not np.isfinite(lam) or lam < 0:
        return np.nan
    if k <= 0:
        return 1.0
    # Compute CDF(k-1) via iterative terms to avoid scipy dependency.
    term = np.exp(-lam)  # i=0
    cdf = term
    for i in range(1, k):
        term = term * lam / float(i)
        cdf += term
    p = 1.0 - cdf
    # numeric safety
    if p < 0:
        p = 0.0
    if p > 1:
        p = 1.0
    return float(p)


# Helper: Vig-free-ish P(TG>=3) from a fitted Poisson total-goals lambda.
def _poisson_p_ge3(lam: float) -> float:
    """Vig-free-ish P(TG>=3) from a fitted Poisson total-goals lambda."""
    try:
        lam = float(lam)
    except Exception:
        return np.nan
    if not np.isfinite(lam) or lam <= 0:
        return np.nan
    return float(1.0 - np.exp(-lam) * (1.0 + lam + 0.5 * lam * lam))

def _fit_lambda_total_from_over_odds(o15: float, o25: float, o35: float, o45: float) -> tuple[float, bool]:
    """Fit a Poisson lambda_total from vigged OVER odds for 1.5/2.5/3.5/4.5.

    Uses implied probabilities for P(TG>=2), P(TG>=3), P(TG>=4), P(TG>=5).
    Returns (lambda_total_fit, ok).
    """
    pts: list[tuple[int, float]] = []

    def _imp(od: float) -> float:
        try:
            od = float(od)
        except Exception:
            return np.nan
        if not np.isfinite(od) or od <= 1.0:
            return np.nan
        return float(1.0 / od)

    p2 = _imp(o15)
    p3 = _imp(o25)
    p4 = _imp(o35)
    p5 = _imp(o45)

    if np.isfinite(p2):
        pts.append((2, float(p2)))
    if np.isfinite(p3):
        pts.append((3, float(p3)))
    if np.isfinite(p4):
        pts.append((4, float(p4)))
    if np.isfinite(p5):
        pts.append((5, float(p5)))

    # Need at least 2 points for a meaningful fit.
    if len(pts) < 2:
        return (np.nan, False)

    best_lam = np.nan
    best_err = float("inf")

    # Grid search is fast enough here.
    for lam in np.linspace(0.50, 5.50, 501):
        err = 0.0
        for k, p_obs in pts:
            p_hat = _poisson_tail_ge(lam, k)
            if not np.isfinite(p_hat):
                err = float("inf")
                break
            d = (p_hat - p_obs)
            err += float(d * d)
        if err < best_err:
            best_err = err
            best_lam = float(lam)

    ok = np.isfinite(best_lam) and best_err < float("inf")
    return (best_lam, bool(ok))
# --- End: Goaliness and lambda fit helpers ---


def _bookie_pick_ftr(row: pd.Series, cols: Dict[str, str], implied_min: float) -> Optional[Dict[str, Any]]:
    if not cols.get("ftr_home") or not cols.get("ftr_draw") or not cols.get("ftr_away"):
        return None

    oh = float(pd.to_numeric(row.get(cols["ftr_home"]), errors="coerce") or np.nan)
    od = float(pd.to_numeric(row.get(cols["ftr_draw"]), errors="coerce") or np.nan)
    oa = float(pd.to_numeric(row.get(cols["ftr_away"]), errors="coerce") or np.nan)
    if not (np.isfinite(oh) and oh > 1.0 and np.isfinite(od) and od > 1.0 and np.isfinite(oa) and oa > 1.0):
        return None

    imp_h = 1.0 / oh
    imp_d = 1.0 / od
    imp_a = 1.0 / oa

    # Raw (vigged) implieds
    imps = {"HOME": imp_h, "DRAW": imp_d, "AWAY": imp_a}
    pick = max(imps, key=lambda k: imps[k])
    imp = float(imps[pick])
    if imp < implied_min:
        return None

    # No-vig implied + overround
    overround = float(imp_h + imp_d + imp_a)
    imp_novig_h = imp_h / overround if np.isfinite(overround) and overround > 0 else np.nan
    imp_novig_d = imp_d / overround if np.isfinite(overround) and overround > 0 else np.nan
    imp_novig_a = imp_a / overround if np.isfinite(overround) and overround > 0 else np.nan

    imp_novig_map = {"HOME": imp_novig_h, "DRAW": imp_novig_d, "AWAY": imp_novig_a}
    imp_novig_pick = float(imp_novig_map.get(pick, np.nan))

    # Bookie sharpness: top1 - top2 on NO-VIG implieds
    novig_vals = [v for v in [imp_novig_h, imp_novig_d, imp_novig_a] if np.isfinite(v)]
    if len(novig_vals) >= 2:
        s = sorted(novig_vals)
        bookie_spread = float(s[-1] - s[-2])
    else:
        bookie_spread = np.nan

    return {
        "bookie_pick": pick,
        "bookie_od": {"HOME": oh, "DRAW": od, "AWAY": oa}[pick],
        "bookie_implied": imp,
        "bookie_overround": overround,
        "bookie_implied_novig": imp_novig_pick,
        "bookie_spread": bookie_spread,
        "od_home": oh,
        "od_draw": od,
        "od_away": oa,
        # Add goaliness/lambda placeholders for schema consistency
        "bookie_lambda_total_fit": np.nan,
        "bookie_goaliness_fit_ok": False,
    }


def _bookie_pick_ou25(row: pd.Series, cols: Dict[str, str], implied_min: float) -> Optional[Dict[str, Any]]:
    """Pick the bookie's stronger OU2.5 side (OVER25 or UNDER25) when both odds exist.

    - If both OVER and UNDER odds exist, choose the side with higher raw implied (vigged)
      and compute a proper no-vig implied + overround.
    - If UNDER odds are missing/invalid, fall back to OVER25 only and (optionally)
      approximate no-vig via lambda-fit.
    """
    if not cols.get("ou25_over"):
        return None

    o_over = float(pd.to_numeric(row.get(cols["ou25_over"]), errors="coerce") or np.nan)
    if not (np.isfinite(o_over) and o_over > 1.0):
        return None

    # Optional under odds
    o_under = np.nan
    if cols.get("ou25_under"):
        o_under = float(pd.to_numeric(row.get(cols["ou25_under"]), errors="coerce") or np.nan)

    imp_over = float(1.0 / o_over)
    imp_under = float(1.0 / o_under) if (np.isfinite(o_under) and o_under > 1.0) else np.nan

    # Choose side (prefer the bookie's higher implied if both are present)
    if np.isfinite(imp_under):
        pick = "OVER25" if imp_over >= imp_under else "UNDER25"
        imp_pick = imp_over if pick == "OVER25" else imp_under
        od_pick = o_over if pick == "OVER25" else float(o_under)

        if imp_pick < implied_min:
            return None

        overround = float(imp_over + imp_under)
        imp_novig = float(imp_pick / overround) if np.isfinite(overround) and overround > 0 else np.nan

    else:
        # No UNDER odds => only consider OVER25
        pick = "OVER25"
        imp_pick = imp_over
        od_pick = o_over

        if imp_pick < implied_min:
            return None

        overround = np.nan
        imp_novig = np.nan

    # Bookie goaliness proxy (fit lambda_total from totals over odds when available)
    o15 = np.nan
    o35 = np.nan
    o45 = np.nan
    if cols.get("tg_over15"):
        o15 = float(pd.to_numeric(row.get(cols["tg_over15"]), errors="coerce") or np.nan)
    if cols.get("tg_over35"):
        o35 = float(pd.to_numeric(row.get(cols["tg_over35"]), errors="coerce") or np.nan)
    if cols.get("tg_over45"):
        o45 = float(pd.to_numeric(row.get(cols["tg_over45"]), errors="coerce") or np.nan)

    lam_fit, lam_ok = _fit_lambda_total_from_over_odds(o15, o_over, o35, o45)

    # If we don't have UNDER odds (no overround), approximate no-vig using lambda.
    # Priority:
    #   1) Bookie-fitted totals lambda (from over15/25/35/45) when available.
    #   2) Model total-goals lambda proxy (exp_goals_sum or lambda_home+lambda_away) when bookie totals aren't published yet.
    # For OVER25: P(TG>=3). For UNDER25: 1 - P(TG>=3).

    # 1) Bookie lambda-fit fallback
    if (not np.isfinite(imp_novig)) and bool(lam_ok) and np.isfinite(lam_fit):
        p_ge3 = _poisson_p_ge3(lam_fit)
        if np.isfinite(p_ge3):
            imp_novig = float(p_ge3) if pick == "OVER25" else float(1.0 - p_ge3)

    # 2) Model lambda total fallback (ONLY if no-vig still missing)
    if not np.isfinite(imp_novig):
        lam_model = float("nan")
        try:
            lam_model = float(pd.to_numeric(row.get("exp_goals_sum", np.nan), errors="coerce"))
        except Exception:
            lam_model = float("nan")

        if not np.isfinite(lam_model):
            try:
                lh = float(pd.to_numeric(row.get("lambda_home", np.nan), errors="coerce"))
                la = float(pd.to_numeric(row.get("lambda_away", np.nan), errors="coerce"))
                if np.isfinite(lh) and np.isfinite(la):
                    lam_model = float(lh + la)
            except Exception:
                lam_model = float("nan")

        if np.isfinite(lam_model) and (lam_model > 0.0):
            p_ge3_m = _poisson_p_ge3(lam_model)
            if np.isfinite(p_ge3_m):
                imp_novig = float(p_ge3_m) if pick == "OVER25" else float(1.0 - p_ge3_m)

    # If UNDER odds were missing, synthesize a fair UNDER price so downstream columns aren't empty.
    # Priority:
    #   1) Use bookie lambda-fit when available.
    #   2) Else use model lambda total proxy.
    # (These are NOT bookie prices; they are fair-odds proxies for schema completeness.)
    if not np.isfinite(o_under):
        lam_for_under = float("nan")
        if bool(lam_ok) and np.isfinite(lam_fit):
            lam_for_under = float(lam_fit)
        else:
            # Use model proxy
            try:
                lam_for_under = float(pd.to_numeric(row.get("exp_goals_sum", np.nan), errors="coerce"))
            except Exception:
                lam_for_under = float("nan")
            if not np.isfinite(lam_for_under):
                try:
                    lh = float(pd.to_numeric(row.get("lambda_home", np.nan), errors="coerce"))
                    la = float(pd.to_numeric(row.get("lambda_away", np.nan), errors="coerce"))
                    if np.isfinite(lh) and np.isfinite(la):
                        lam_for_under = float(lh + la)
                except Exception:
                    lam_for_under = float("nan")

        if np.isfinite(lam_for_under) and (lam_for_under > 0.0):
            try:
                p_ge3_u = _poisson_p_ge3(lam_for_under)
                p_u25 = float(1.0 - p_ge3_u) if np.isfinite(p_ge3_u) else float("nan")
                if np.isfinite(p_u25) and (p_u25 > 0.0) and (p_u25 < 1.0):
                    o_under = float(1.0 / p_u25)
            except Exception:
                pass

    return {
        "bookie_pick": pick,
        "bookie_od": float(od_pick),
        "bookie_implied": float(imp_pick),
        "bookie_overround": overround,
        "bookie_implied_novig": imp_novig,
        "bookie_spread": np.nan,

        "od_over": float(o_over),
        "od_under": float(o_under) if np.isfinite(o_under) else np.nan,

        # canonical OU25 odds (bookie OR synth proxy)
        "odds_ft_over25": float(o_over),
        "odds_ft_under25": float(o_under) if np.isfinite(o_under) else np.nan,

        "bookie_lambda_total_fit": lam_fit,
        "bookie_goaliness_fit_ok": bool(lam_ok),
    }


def _bookie_pick_btts(row: pd.Series, cols: Dict[str, str], implied_min: float) -> Optional[Dict[str, Any]]:
    if not cols.get("btts_yes"):
        return None

    o_yes = float(pd.to_numeric(row.get(cols["btts_yes"]), errors="coerce") or np.nan)
    if not (np.isfinite(o_yes) and o_yes > 1.0):
        return None

    # Optional NO odds
    o_no = np.nan
    if cols.get("btts_no"):
        o_no = float(pd.to_numeric(row.get(cols["btts_no"]), errors="coerce") or np.nan)

    imp_yes = float(1.0 / o_yes)
    imp_no = float(1.0 / o_no) if (np.isfinite(o_no) and o_no > 1.0) else np.nan

    # Choose side (prefer higher implied if both present)
    if np.isfinite(imp_no):
        pick = "YES" if imp_yes >= imp_no else "NO"
        imp_pick = imp_yes if pick == "YES" else imp_no
        od_pick = o_yes if pick == "YES" else float(o_no)

        if imp_pick < implied_min:
            return None

        overround = float(imp_yes + imp_no)
        imp_novig = float(imp_pick / overround) if np.isfinite(overround) and overround > 0 else np.nan
    else:
        # If NO odds missing, fall back to YES only
        pick = "YES"
        imp_pick = imp_yes
        od_pick = o_yes

        if imp_pick < implied_min:
            return None

        overround = np.nan
        imp_novig = np.nan

    # Bookie goaliness proxy (reuse totals odds if present)
    o15 = np.nan
    o25 = np.nan
    o35 = np.nan
    o45 = np.nan
    if cols.get("tg_over15"):
        o15 = float(pd.to_numeric(row.get(cols["tg_over15"]), errors="coerce") or np.nan)
    if cols.get("ou25_over"):
        o25 = float(pd.to_numeric(row.get(cols["ou25_over"]), errors="coerce") or np.nan)
    if cols.get("tg_over35"):
        o35 = float(pd.to_numeric(row.get(cols["tg_over35"]), errors="coerce") or np.nan)
    if cols.get("tg_over45"):
        o45 = float(pd.to_numeric(row.get(cols["tg_over45"]), errors="coerce") or np.nan)

    lam_fit, lam_ok = _fit_lambda_total_from_over_odds(o15, o25, o35, o45)

    return {
        # Canonical BTTS convention: market is always 'btts', selection is YES/NO
        "market": "btts",
        "selection": pick,
        

        "bookie_pick": pick,
        "bookie_od": float(od_pick),
        "bookie_implied": float(imp_pick),
        "bookie_overround": overround,
        "bookie_implied_novig": imp_novig,
        "bookie_spread": np.nan,
        "od_yes": float(o_yes),
        "od_no": float(o_no) if np.isfinite(o_no) else np.nan,

        # canonical BTTS odds
        "odds_btts_yes": float(o_yes),
        "odds_btts_no": float(o_no) if np.isfinite(o_no) else np.nan,

        "bookie_lambda_total_fit": lam_fit,
        "bookie_goaliness_fit_ok": bool(lam_ok),
    }


def main() -> None:
    _maybe_reexec_into_project_venv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--date-from", required=True)
    ap.add_argument("--date-to", required=True)
    ap.add_argument("--implied-min", type=float, default=0.68)
    # FTR product profile (lane)
    ap.add_argument(
        "--ftr-profile",
        default=os.getenv("FTR_PROFILE", "accuracy"),
        choices=["accuracy", "valueev_balanced", "valueev_aggressive"],
        help="FTR lane profile: accuracy=discipline favourites; valueev_* = pick from ANCHOR_CAND with edge ranking.",
    )

    # Optional: edge metric
    ap.add_argument(
        "--ftr-edge-metric",
        default=os.getenv("FTR_EDGE_METRIC", "gap_novig"),
        choices=["gap_novig", "ratio_novig"],
        help="ValueEV edge metric: gap_novig=(p_model - imp_nv); ratio_novig=(p_model / imp_nv - 1).",
    )

    # Optional: per-league top-k for valueEV lanes
    ap.add_argument(
        "--ftr-valueev-topk-per-league",
        type=int,
        default=int(os.getenv("FTR_VALUEEV_TOPK_PER_LEAGUE", "2")),
        help="ValueEV lanes: keep at most this many FTR rows per league (default 2).",
    )
    ap.add_argument(
        "--ftr-implied-min",
        type=float,
        default=None,
        help="Override implied-min for FTR (1X2) only. Default: use --implied-min.",
    )
    ap.add_argument(
        "--ou25-implied-min",
        type=float,
        default=None,
        help="Separate implied floor for OU25 rows. Defaults to --implied-min when omitted.",
    )
    ap.add_argument(
        "--btts-implied-min",
        type=float,
        default=None,
        help="Override implied-min for BTTS only (useful to allow NO picks). Default: use --implied-min.",
    )
    ap.add_argument("--matches-root", default="Matches")
    ap.add_argument("--teams-root", default="Teams", help="Root folder for UEFA team CSVs (default: Teams)")
    ap.add_argument("--modelstore", default="ModelStore")
    ap.add_argument("--outdir", default=None)
    ap.add_argument(
        "--run-tag",
        default=None,
        help="Optional run folder name under predictions_output (e.g. 2024-10). Used when --outdir is not set.",
    )
    ap.add_argument(
        "--run-dir",
        default=None,
        help="Explicit output directory path. Overrides --outdir and the default dated folder.",
    )
    ap.add_argument("--leagues", default=None, help='Comma-separated; default = investor 17')
    ap.add_argument("--markets", default="ftr,ou25,btts", help="Comma-separated: ftr,ou25,btts,tg15,tg25")
    ap.add_argument("--tg15-pmin", type=float, default=0.65, help="Team goals 1.5 (>=2) min model prob (default: 0.65)")
    ap.add_argument("--tg25-pmin", type=float, default=0.45, help="Team goals 2.5 (>=3) min model prob (default: 0.45)")
    ap.add_argument(
        "--tg-pois-ge2-min",
        type=float,
        default=0.12,
        help="TG coherence: require Poisson P(team>=2) >= this for TG15 candidates (default: 0.12).",
    )
    ap.add_argument(
        "--tg-pois-ge3-min",
        type=float,
        default=0.08,
        help="TG coherence: require Poisson P(team>=3) >= this for TG25 candidates (default: 0.08).",
    )
    ap.add_argument(
    "--tg-pois-gap-max-ge2",
    type=float,
    default=0.50,
    help="TG coherence: veto if (GE2_model - Poisson_ge2) > this for TG15 candidates (default: 0.50).",
    )
    ap.add_argument(
        "--tg-pois-gap-max-ge3",
        type=float,
        default=0.50,
        help="TG coherence: veto if (GE3_model - Poisson_ge3) > this for TG25 candidates (default: 0.50).",
    )
    # TG directional sanity gates (recommended)
    ap.add_argument(
        "--tg-use-dir-gate",
        action="store_true",
        help="TG: enable directional PPG/power_diff gating (prevents nonsense TG picks).",
    )
    ap.add_argument(
        "--tg-ppg-home-min",
        type=float,
        default=0.35,
        help="TG dir gate: HOME_TG requires ppg_diff_pre >= this (default: 0.35).",
    )
    ap.add_argument(
        "--tg-ppg-away-max",
        type=float,
        default=-0.35,
        help="TG dir gate: AWAY_TG requires ppg_diff_pre <= this (default: -0.35).",
    )
    ap.add_argument(
        "--tg-pd-home-min",
        type=float,
        default=5.0,
        help="TG dir gate: HOME_TG requires power_diff >= this when available (default: 5.0).",
    )
    ap.add_argument(
        "--tg-pd-away-max",
        type=float,
        default=-5.0,
        help="TG dir gate: AWAY_TG requires power_diff <= this when available (default: -5.0).",
    )
    ap.add_argument(
        "--tg-opp-ppg-max",
        type=float,
        default=1.20,
        help="TG dir gate: opponent PPG must be <= this when available (default: 1.20).",
    )
    ap.add_argument(
        "--tg-ambig-delta",
        type=float,
        default=0.05,
        help="TG: if both sides pass and |p1-p2| < this, skip as ambiguous (default: 0.05).",
    )
    ap.add_argument("--debug", action="store_true", help="Print per-league diagnostics for attached rolling rate columns")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail fast if export QC checks are not green (non-null + sanity asserts)",
    )
    ap.add_argument("--enable-h2h", action="store_true", help="Attach H2H rate features (h2h_n, h2h_btts_rate, h2h_over25_rate, h2h_goaliness_avg)")
    ap.add_argument(
        "--emit-ftr-candidates",
        action="store_true",
        help="Emit extra FTR candidate rows (HOME/DRAW/AWAY) for each fixture (anchor-pool mode).",
    )
    ap.add_argument(
        "--ftr-cand-od-min",
        type=float,
        default=2.50,
        help="FTR candidate rows: minimum odds (default: 2.50).",
    )
    ap.add_argument(
        "--ftr-cand-od-max",
        type=float,
        default=6.00,
        help="FTR candidate rows: maximum odds (default: 6.00). Use a higher value if you want long-shot monsters.",
    )
    ap.add_argument(
        "--ftr-cand-pmin",
        type=float,
        default=0.20,
        help="FTR candidate rows: minimum model probability for that outcome (default: 0.20).",
    )
    ap.add_argument(
        "--ftr-cand-margin-min",
        type=float,
        default=0.05,
        help="FTR candidate rows: minimum ftr_margin (top1-top2) on the fixture (default: 0.05).",
    )
    ap.add_argument(
        "--ftr-cand-max-per-fixture",
        type=int,
        default=2,
        help="FTR candidate rows: keep at most this many candidates per fixture after filtering (default: 2).",
    )
    ap.add_argument(
        "--ftr-cand-max-per-league",
        type=int,
        default=0,
        help="FTR candidate rows: optional cap per league after filtering (0=disabled; default: 0).",
    )
    ap.add_argument(
        "--ftr-cand-gap-min",
        type=float,
        default=-0.05,
        help="FTR candidate rows: minimum (model_p - no-vig implied) for that outcome (default: -0.05).",
    )

    ap.add_argument(
        "--ftr-cand-use-ppg",
        action="store_true",
        help="FTR candidate rows: enable directional PPG gating (recommended to prevent nonsensical underdog anchors).",
    )
    ap.add_argument(
        "--ftr-cand-ppg-home-min",
        type=float,
        default=0.35,
        help="FTR candidate rows (PPG gate): if pick==HOME require ppg_diff_pre >= this (default: 0.35).",
    )
    ap.add_argument(
        "--ftr-cand-ppg-away-max",
        type=float,
        default=-0.35,
        help="FTR candidate rows (PPG gate): if pick==AWAY require ppg_diff_pre <= this (default: -0.35).",
    )
    ap.add_argument(
        "--ftr-cand-ppg-draw-abs-max",
        type=float,
        default=0.35,
        help="FTR candidate rows (PPG gate): if pick==DRAW require abs(ppg_diff_pre) <= this (default: 0.35).",
    )
    ap.add_argument(
        "--ftr-cand-ppg-opp-max",
        type=float,
        default=1.20,
        help="FTR candidate rows (PPG gate): optional opponent PPG cap for HOME/AWAY candidates (default: 1.20).",
    )

    ap.add_argument(
        "--ftr-glue-use-ppg",
        action="store_true",
        help="FTR base rows: only keep favourites that pass PPG glue gate (ppg_diff>=thr and opp_ppg<=thr).",
    )
    ap.add_argument(
        "--ftr-glue-ppg-diff-min",
        type=float,
        default=0.70,
        help="FTR base rows: PPG glue gate diff min (default: 0.70).",
    )
    ap.add_argument(
        "--ftr-glue-ppg-opp-max",
        type=float,
        default=1.00,
        help="FTR base rows: PPG glue gate opponent max (default: 1.00).",
    )

    ap.add_argument(
        "--ftr-drawtrap-veto",
        action="store_true",
        help="FTR base rows: veto ultra-short favourites vs decent opponents (draw trap).",
    )
    ap.add_argument(
        "--ftr-drawtrap-od-max",
        type=float,
        default=1.30,
        help="FTR base rows: draw trap odds max (default: 1.30).",
    )
    ap.add_argument(
        "--ftr-drawtrap-opp-ppg-min",
        type=float,
        default=1.20,
        help="FTR base rows: draw trap opponent PPG min (default: 1.20).",
    )
    ap.add_argument(
        "--merged-subdir",
        default="__merged__",
        help="Subdirectory under --matches-root containing merged league CSVs (default: __merged__).",
    )
    args = ap.parse_args()

    # Allow a separate implied gate for BTTS so BTTS_NO can appear (it is rarely priced as short as BTTS_YES)
    btts_implied_min = float(args.btts_implied_min) if args.btts_implied_min is not None else float(args.implied_min)
    # Allow a separate implied gate for FTR if provided
    ftr_implied_min = float(args.ftr_implied_min) if args.ftr_implied_min is not None else float(args.implied_min)
    # Allow a separate implied gate for OU25 (totals 2.5) so totals can survive when FTR is strict.
    ou25_implied_min = float(args.ou25_implied_min) if args.ou25_implied_min is not None else float(args.implied_min)

    leagues = DEFAULT_LEAGUES if not args.leagues else [s.strip() for s in args.leagues.split(",") if s.strip()]
    markets = [m.strip().lower() for m in args.markets.split(",") if m.strip()]

    matches_root = Path(args.matches_root)
    modelstore = Path(args.modelstore)

    teams_root = Path(getattr(args, "teams_root", "Teams"))

    def _load_league_source(league_name: str, *, require_ftr_1x2: bool) -> tuple[pd.DataFrame, Optional[Path]]:
        """Load the best available source dataframe for a league.

        Weekend-safe policy:
          - Prefer merged league files when available (Matches/__merged__/<TAG>__merged.csv),
            because they contain engineered columns (e.g. rolling press) that trained bundles may expect.
          - For side markets, prefer synth-enriched odds when available.
          - For FTR, prefer a source that actually contains valid in-window 1X2 odds.
          - Fall back to the best available per-league folder CSVs.

        Returns: (df, src_path_or_None)
        """
        lg = str(league_name or "").strip()
        tag = _league_tag(lg)

        matches_root = Path(str(getattr(args, "matches_root", "Matches")))
        merged_root = matches_root / str(getattr(args, "merged_subdir", "__merged__"))
        league_dir = matches_root / lg

        merged_proxy_csv = merged_root / f"{tag}__merged__proxy_enriched.csv"
        merged_csv = merged_root / f"{tag}__merged.csv"
        synth_csv = league_dir / "fd_odds_enriched_synth.csv"
        enriched_csv = league_dir / "fd_odds_enriched.csv"

        def _read_csv(p: Path) -> pd.DataFrame:
            df = pd.read_csv(p, low_memory=False)
            if "__src_csv" not in df.columns:
                df["__src_csv"] = p.name

            # existing hygiene helpers (best-effort)
            try:
                df = _apply_feature_aliases(df)
            except Exception:
                pass
            try:
                df = _best_prob_cols(df)
            except Exception:
                pass

            # Ensure fixture_key exists (needed for joins / power ratings / dedupe)
            if "fixture_key" not in df.columns or df["fixture_key"].astype("string").fillna("").str.strip().eq("").all():
                try:
                    df["fixture_key"] = df.apply(_match_key_ascii, axis=1)
                except Exception:
                    try:
                        df["fixture_key"] = df.apply(_match_key, axis=1)
                    except Exception:
                        df["fixture_key"] = pd.NA

            # Ensure league populated (keeps downstream assumptions stable)
            if "league" not in df.columns:
                df["league"] = lg
            else:
                try:
                    _lg0 = df["league"].astype("string").fillna("").str.strip()
                    df["league"] = _lg0.where(_lg0.ne(""), lg)
                except Exception:
                    pass

            # Standardise team cols early
            if "home_team_name" not in df.columns and "Home" in df.columns:
                df["home_team_name"] = df["Home"]
            if "away_team_name" not in df.columns and "Away" in df.columns:
                df["away_team_name"] = df["Away"]

            return df

        def _valid_ftr_rows_in_window(p: Path) -> int:
            if not require_ftr_1x2:
                return 1
            try:
                return int(
                    _count_valid_1x2_rows_in_window(
                        p,
                        str(getattr(args, "date_from", "")),
                        str(getattr(args, "date_to", "")),
                    )
                )
            except Exception:
                return 0

        def _valid_side_rows_in_window(p: Path) -> int:
            if require_ftr_1x2:
                return 0
            try:
                return int(
                    _count_valid_side_rows_in_window(
                        p,
                        str(getattr(args, "date_from", "")),
                        str(getattr(args, "date_to", "")),
                    )
                )
            except Exception:
                return 0

        # For side markets, choose the file with the strongest in-window side-odds
        # coverage instead of blindly preferring synth/enriched. This protects
        # weekend boards after partial rebuilds where one source family is stale.
        if not require_ftr_1x2:
            side_pref_rank = {
                str(merged_proxy_csv): 0,
                str(merged_csv): 1,
                str(synth_csv): 2,
                str(enriched_csv): 3,
            }
            side_candidates: list[Path] = []
            try:
                picked = _pick_latest_matches_csv(
                    matches_root,
                    lg,
                    str(getattr(args, "date_from", "")),
                    str(getattr(args, "date_to", "")),
                    require_ftr_1x2=False,
                )
                if picked is not None:
                    side_candidates.append(picked)
            except Exception:
                pass

            for p in (merged_proxy_csv, merged_csv, synth_csv, enriched_csv):
                if p.exists():
                    side_candidates.append(p)

            best_side_path: Optional[Path] = None
            best_side_score = -1
            best_side_rows = -1
            best_side_rank = 10_000
            seen_side: set[str] = set()

            for p in side_candidates:
                ps = str(p)
                if ps in seen_side:
                    continue
                seen_side.add(ps)
                score = _valid_side_rows_in_window(p)
                rows = 0
                try:
                    rows = int(
                        _count_rows_in_window(
                            p,
                            str(getattr(args, "date_from", "")),
                            str(getattr(args, "date_to", "")),
                        )
                    )
                except Exception:
                    rows = 0
                rank = int(side_pref_rank.get(ps, 10_000))
                if (
                    (score > best_side_score)
                    or (score == best_side_score and rows > best_side_rows)
                    or (score == best_side_score and rows == best_side_rows and rank < best_side_rank)
                ):
                    best_side_path = p
                    best_side_score = score
                    best_side_rows = rows
                    best_side_rank = rank

            if best_side_path is not None and best_side_rows > 0:
                try:
                    return _read_csv(best_side_path), best_side_path
                except Exception:
                    pass

        # 1) Prefer proxy-enriched merged (best feature coverage for historical tests)
        if merged_proxy_csv.exists() and _valid_ftr_rows_in_window(merged_proxy_csv) > 0:
            try:
                return _read_csv(merged_proxy_csv), merged_proxy_csv
            except Exception:
                pass

        # 2) Fall back to plain merged
        if merged_csv.exists() and _valid_ftr_rows_in_window(merged_csv) > 0:
            try:
                return _read_csv(merged_csv), merged_csv
            except Exception:
                pass

        # 3) For FTR, prefer the best per-league CSV with real 1X2 coverage in-window.
        if require_ftr_1x2:
            try:
                picked = _pick_latest_matches_csv(
                    matches_root,
                    lg,
                    str(getattr(args, "date_from", "")),
                    str(getattr(args, "date_to", "")),
                    require_ftr_1x2=True,
                )
                if picked is not None:
                    return _read_csv(picked), picked
            except Exception:
                pass

        # 4) Prefer synth enriched for side markets / non-FTR flows
        if synth_csv.exists():
            try:
                return _read_csv(synth_csv), synth_csv
            except Exception:
                pass

        # 5) Fall back to enriched
        if enriched_csv.exists():
            try:
                return _read_csv(enriched_csv), enriched_csv
            except Exception:
                pass

        # 6) Last resort: newest CSV in league folder
        try:
            if league_dir.exists():
                cands = sorted(
                    [p for p in league_dir.glob("*.csv") if p.is_file()],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if cands:
                    p = cands[0]
                    return _read_csv(p), p
        except Exception:
            pass

        return (pd.DataFrame(), None)

    if bool(getattr(args, "debug", False)):
        try:
            print(
                f"[bookie_allmarkets] UEFA builder available={callable(_uefa_build_snapshot_for_league)} | "
                f"teams_root={teams_root} | matches_root={matches_root} | "
                f"merged_subdir={getattr(args, 'merged_subdir', '__merged__')}"
            )
        except Exception:
            pass

    # Cache UEFA snapshots per league for this run (prevents repeated CSV reads)
    _uefa_snapshot_cache: Dict[str, pd.DataFrame] = {}

    def _call_uefa_builder(league_name: str) -> Optional[pd.DataFrame]:
        """Best-effort call into uefa_context.build_snapshot_for_league.

        We try a few signatures to stay compatible with older/newer versions.
        Returns a TEAM-LEVEL snapshot (one row per team) or None.
        """
        if not callable(_uefa_build_snapshot_for_league):
            return None

        if league_name in _uefa_snapshot_cache:
            snap = _uefa_snapshot_cache.get(league_name)
            return snap if isinstance(snap, pd.DataFrame) and not snap.empty else None

        snap = None
        # Try multiple signatures; uefa_context.build_snapshot_for_league can vary by version.
        for _args, _kwargs in [
            ((), {"comp": league_name, "teams_root": teams_root, "matches_root": matches_root}),
            ((), {"league": league_name, "teams_root": teams_root, "matches_root": matches_root}),
            ((league_name,), {"teams_root": teams_root, "matches_root": matches_root}),
            ((league_name,), {"teams_root": str(teams_root), "matches_root": str(matches_root)}),
            ((league_name, teams_root, matches_root), {}),
            ((league_name, str(teams_root), str(matches_root)), {}),
            ((league_name, teams_root), {}),
            ((league_name, str(teams_root)), {}),
            ((league_name,), {}),
        ]:
            try:
                snap = _uefa_build_snapshot_for_league(*_args, **_kwargs)  # type: ignore
                break
            except TypeError:
                continue
            except Exception:
                snap = None
                break

        if bool(getattr(args, "debug", False)) and isinstance(snap, pd.DataFrame):
            try:
                print(f"[bookie_allmarkets] {league_name} UEFA snapshot: n={len(snap)}")
            except Exception:
                pass
        # Additional debug: columns, dtypes, volatility_ratio/vol_band_n distributions
        if bool(getattr(args, "debug", False)) and isinstance(snap, pd.DataFrame):
            try:
                cols0 = list(snap.columns)
                key_cols = [
                    c for c in (
                        "team_name",
                        "common_name",
                        "state_bucket",
                        "gap_to24",
                        "gap_to8",
                        "rotation_risk_flag",
                        "must_win_flag",
                        "must_win_big_flag",
                        "must_avoid_loss_flag",
                        "eliminated_flag",
                        "volatility_ratio",
                        "vol_band_n",
                    )
                    if c in cols0
                ]
                dtypes0 = {c: str(snap[c].dtype) for c in key_cols}
                print(f"[bookie_allmarkets] {league_name} UEFA snapshot cols(has): {key_cols}")
                print(f"[bookie_allmarkets] {league_name} UEFA snapshot dtypes: {dtypes0}")

                if "volatility_ratio" in snap.columns:
                    vr = pd.to_numeric(snap["volatility_ratio"], errors="coerce")
                    uniq = sorted(vr.dropna().unique().tolist())[:8]
                    print(f"[bookie_allmarkets] {league_name} UEFA snapshot volatility_ratio: nn={int(vr.notna().sum())} uniq={uniq}")

                if "vol_band_n" in snap.columns:
                    vb = pd.to_numeric(snap["vol_band_n"], errors="coerce")
                    uniq = sorted(vb.dropna().unique().tolist())[:8]
                    print(f"[bookie_allmarkets] {league_name} UEFA snapshot vol_band_n: nn={int(vb.notna().sum())} uniq={uniq}")
            except Exception:
                pass

        if isinstance(snap, pd.DataFrame) and not snap.empty:
            _uefa_snapshot_cache[league_name] = snap
            return snap

        _uefa_snapshot_cache[league_name] = pd.DataFrame()
        if bool(getattr(args, "debug", False)):
            try:
                print(f"[bookie_allmarkets] {league_name} UEFA snapshot unavailable (teams_root={teams_root})")
            except Exception:
                pass
        return None

    def _build_uefa_match_context(league_name: str, dfw: pd.DataFrame) -> pd.DataFrame:
        """Attach UEFA table-pressure / rotation context to the match window frame.

        Adds match-level columns (prefixed):
          - uefa_home_state, uefa_away_state
          - uefa_home_gap24, uefa_away_gap24, uefa_gap24_diff
          - uefa_home_rotation_risk, uefa_away_rotation_risk
          - uefa_both_must_win, uefa_goal_hunt_flag, uefa_pride_only_flag
          - uefa_live_table_volatility

        Also appends a fixture-level map into `uefa_maps` for later merging.
        Safe no-op when snapshot isn't available.
        """
        if dfw is None or dfw.empty:
            return dfw

        snap = _call_uefa_builder(league_name)
        if snap is None or snap.empty:
            return dfw

        out = dfw.copy()

        # Normalise join keys (match files typically use common_name-like tokens)
        def _norm(s: pd.Series) -> pd.Series:
            # 1) fill + to string
            x = s.fillna("").astype(str)

            # 2) strip accents (Qarabağ -> Qarabag)
            def _strip_acc(v: str) -> str:
                try:
                    return "".join(ch for ch in unicodedata.normalize("NFKD", v) if not unicodedata.combining(ch))
                except Exception:
                    return v

            x = x.map(_strip_acc)

            # 3) lowercase + cleanup
            x = pd.Series(x, index=s.index).astype("string").str.lower().str.strip()
            x = x.str.replace(r"[^a-z0-9]+", " ", regex=True)
            # Drop common suffix tokens so "arsenal fc" can match "arsenal"
            x = x.str.replace(r"\b(fc|cf|sc|ac|fk|bk|kv|sk|sv|afc|utd)\b", " ", regex=True)
            x = x.str.replace(r"\s+", " ", regex=True).str.strip()
            return x

        snap2 = snap.copy()

        # Build join-key map from any available name columns.
        # We always require at least `team_name`.
        if "team_name" not in snap2.columns:
            return out

        name_cols = ["team_name"]
        for c in ("common_name", "short_name", "team_short_name", "team_name_short", "club_name", "name"):
            if c in snap2.columns and c not in name_cols:
                name_cols.append(c)

        parts: list[pd.DataFrame] = []
        for c in name_cols:
            tmp = snap2.copy()
            tmp["_team_key"] = _norm(tmp[c])
            parts.append(tmp)

        snap2 = pd.concat(parts, ignore_index=True, sort=False)
        snap2["_team_key"] = snap2["_team_key"].astype("string").fillna("").str.strip()
        snap2 = snap2[snap2["_team_key"].ne("")].copy()
        try:
            snap2 = snap2.drop_duplicates(subset=["_team_key"], keep="first")
        except Exception:
            pass

        if bool(getattr(args, "debug", False)):
            try:
                print(f"[bookie_allmarkets] {league_name} UEFA key-cols used: {name_cols}")
            except Exception:
                pass

        # Create match keys
        out["_home_key"] = _norm(out.get("home_team_name", pd.Series("", index=out.index)))
        out["_away_key"] = _norm(out.get("away_team_name", pd.Series("", index=out.index)))

        # Select columns we expect from snapshot (defensive)
        keep_cols = [
            "_team_key",
            "state_bucket",
            "gap_to24",
            "gap_to8",
            "must_win_flag",
            "must_win_big_flag",
            "must_avoid_loss_flag",
            "rotation_risk_flag",
            "eliminated_flag",
            "volatility_ratio",
            "vol_band_n",
        ]
        for c in keep_cols:
            if c not in snap2.columns:
                snap2[c] = np.nan

        # --- Robust derive of vol_band_n (count of teams in the volatility band) ---
        # Some uefa_context versions only provide `volatility_ratio` (e.g. 0.4878) but not `vol_band_n`.
        # Also: `vol_band_n` can exist but be non-numeric (object/empty), which would map through as NaN.
        comp_vol_band_n = np.nan
        vr0 = np.nan
        nteams = np.nan
        try:
            nteams = int(len(snap)) if isinstance(snap, pd.DataFrame) else int(len(snap2))
        except Exception:
            nteams = np.nan

        # Prefer pulling volatility_ratio from the *original* snapshot if possible
        try:
            if isinstance(snap, pd.DataFrame) and ("volatility_ratio" in snap.columns):
                vr_s = pd.to_numeric(snap["volatility_ratio"], errors="coerce")
            else:
                vr_s = pd.to_numeric(snap2.get("volatility_ratio"), errors="coerce")
            vr0 = float(vr_s.dropna().iloc[0]) if isinstance(vr_s, pd.Series) and (vr_s.dropna().shape[0] > 0) else float("nan")
            if np.isfinite(vr0) and np.isfinite(float(nteams)) and float(nteams) > 0:
                comp_vol_band_n = float(int(round(float(vr0) * float(nteams))))
        except Exception:
            comp_vol_band_n = np.nan

        # Ensure snap2["vol_band_n"] is numeric; if missing/empty, stamp comp-level value
        try:
            vb_num = pd.to_numeric(snap2.get("vol_band_n"), errors="coerce")
            if (not isinstance(vb_num, pd.Series)) or (vb_num.notna().sum() == 0):
                if np.isfinite(comp_vol_band_n):
                    snap2["vol_band_n"] = float(comp_vol_band_n)
            else:
                snap2["vol_band_n"] = vb_num
        except Exception:
            if np.isfinite(comp_vol_band_n):
                snap2["vol_band_n"] = float(comp_vol_band_n)

        if bool(getattr(args, "debug", False)):
            try:
                vb2 = pd.to_numeric(snap2.get("vol_band_n"), errors="coerce")
                nn_vb2 = int(vb2.notna().sum()) if isinstance(vb2, pd.Series) else 0
                uniq_vb2 = sorted(vb2.dropna().unique().tolist())[:8] if isinstance(vb2, pd.Series) else []
                print(
                    f"[bookie_allmarkets] {league_name} UEFA vol_band derive: "
                    f"nteams={nteams} vr0={vr0} comp_vol_band_n={comp_vol_band_n} "
                    f"snap2.vol_band_n nn={nn_vb2} uniq={uniq_vb2}"
                )
            except Exception:
                pass

        base = snap2[keep_cols].copy()

        # Home attach
        h = base.rename(columns={
            "state_bucket": "uefa_home_state",
            "gap_to24": "uefa_home_gap24",
            "gap_to8": "uefa_home_gap8",
            "must_win_flag": "uefa_home_must_win",
            "must_win_big_flag": "uefa_home_must_win_big",
            "must_avoid_loss_flag": "uefa_home_must_avoid_loss",
            "rotation_risk_flag": "uefa_home_rotation_risk",
            "eliminated_flag": "uefa_home_eliminated",
            "volatility_ratio": "uefa_live_table_volatility",
            "vol_band_n": "uefa_vol_band_n",
        }).rename(columns={"_team_key": "_home_key"})
        out = out.merge(h, on="_home_key", how="left")

        # Away attach
        a = base.rename(columns={
            "state_bucket": "uefa_away_state",
            "gap_to24": "uefa_away_gap24",
            "gap_to8": "uefa_away_gap8",
            "must_win_flag": "uefa_away_must_win",
            "must_win_big_flag": "uefa_away_must_win_big",
            "must_avoid_loss_flag": "uefa_away_must_avoid_loss",
            "rotation_risk_flag": "uefa_away_rotation_risk",
            "eliminated_flag": "uefa_away_eliminated",
            "volatility_ratio": "_tmp_vol2",
            "vol_band_n": "_tmp_band2",
        }).rename(columns={"_team_key": "_away_key"})
        out = out.merge(a, on="_away_key", how="left")

        # Prefer a single league-level volatility ratio (same for all teams)
        if "uefa_live_table_volatility" not in out.columns:
            out["uefa_live_table_volatility"] = np.nan
        out["uefa_live_table_volatility"] = pd.to_numeric(out.get("uefa_live_table_volatility"), errors="coerce")
        tmp2 = pd.to_numeric(out.get("_tmp_vol2"), errors="coerce")
        out["uefa_live_table_volatility"] = out["uefa_live_table_volatility"].where(out["uefa_live_table_volatility"].notna(), tmp2)

        # Coalesce uefa_vol_band_n
        if "uefa_vol_band_n" not in out.columns:
            out["uefa_vol_band_n"] = np.nan
        out["uefa_vol_band_n"] = pd.to_numeric(out.get("uefa_vol_band_n"), errors="coerce")
        tmpb = pd.to_numeric(out.get("_tmp_band2"), errors="coerce")
        out["uefa_vol_band_n"] = out["uefa_vol_band_n"].where(out["uefa_vol_band_n"].notna(), tmpb)

        # If still missing everywhere, fall back to the comp-level derived count
        try:
            if ("uefa_vol_band_n" in out.columns) and (pd.to_numeric(out["uefa_vol_band_n"], errors="coerce").notna().sum() == 0):
                if np.isfinite(comp_vol_band_n):
                    out["uefa_vol_band_n"] = float(comp_vol_band_n)
        except Exception:
            pass

        if bool(getattr(args, "debug", False)):
            try:
                vb_out = pd.to_numeric(out.get("uefa_vol_band_n"), errors="coerce")
                nn_vb_out = int(vb_out.notna().sum()) if isinstance(vb_out, pd.Series) else 0
                uniq_vb_out = sorted(vb_out.dropna().unique().tolist())[:8] if isinstance(vb_out, pd.Series) else []
                print(f"[bookie_allmarkets] {league_name} UEFA ctx uefa_vol_band_n: nn={nn_vb_out}/{len(out)} uniq={uniq_vb_out}")
            except Exception:
                pass

        # Match-level derived flags
        hw = pd.to_numeric(out.get("uefa_home_must_win"), errors="coerce").fillna(0).astype(int)
        aw = pd.to_numeric(out.get("uefa_away_must_win"), errors="coerce").fillna(0).astype(int)
        hwb = pd.to_numeric(out.get("uefa_home_must_win_big"), errors="coerce").fillna(0).astype(int)
        awb = pd.to_numeric(out.get("uefa_away_must_win_big"), errors="coerce").fillna(0).astype(int)
        hel = pd.to_numeric(out.get("uefa_home_eliminated"), errors="coerce").fillna(0).astype(int)
        ael = pd.to_numeric(out.get("uefa_away_eliminated"), errors="coerce").fillna(0).astype(int)

        out["uefa_both_must_win"] = ((hw == 1) & (aw == 1)).astype(int)
        out["uefa_goal_hunt_flag"] = ((hwb == 1) | (awb == 1)).astype(int)
        out["uefa_pride_only_flag"] = ((hel == 1) & (ael == 1)).astype(int)

        # Gap diffs
        out["uefa_home_gap24"] = pd.to_numeric(out.get("uefa_home_gap24"), errors="coerce")
        out["uefa_away_gap24"] = pd.to_numeric(out.get("uefa_away_gap24"), errors="coerce")
        out["uefa_gap24_diff"] = out["uefa_home_gap24"] - out["uefa_away_gap24"]

        # Debug print for matched rows
        try:
            if bool(getattr(args, "debug", False)):
                nh = int(pd.to_numeric(out.get("uefa_home_gap24"), errors="coerce").notna().sum())
                na = int(pd.to_numeric(out.get("uefa_away_gap24"), errors="coerce").notna().sum())
                print(f"[bookie_allmarkets] {league_name} UEFA ctx attach: home_matched={nh}/{len(out)} away_matched={na}/{len(out)}")
        except Exception:
            pass

        # Build a fixture-level map for later merge (only when fixture_key exists)
        if "fixture_key" in out.columns:
            try:
                cols_map = [
                    "fixture_key",

                    "uefa_home_state", "uefa_away_state",
                    "uefa_home_gap24", "uefa_away_gap24", "uefa_gap24_diff",

                    "uefa_home_rotation_risk", "uefa_away_rotation_risk",
                    "uefa_home_must_win", "uefa_away_must_win",
                    "uefa_home_must_avoid_loss", "uefa_away_must_avoid_loss",
                    "uefa_home_eliminated", "uefa_away_eliminated",

                    "uefa_both_must_win", "uefa_goal_hunt_flag", "uefa_pride_only_flag",

                    "uefa_live_table_volatility",
                    "uefa_vol_band_n",
                ]
                m = out.reindex(columns=cols_map).copy()
                m["league"] = league_name
                uefa_maps.append(m.drop_duplicates(subset=["league", "fixture_key"], keep="first"))
            except Exception:
                pass

        # Cleanup
        out = out.drop(columns=["_home_key", "_away_key", "_tmp_vol2", "_tmp_band2"], errors="ignore")
        return out

    # Output directory resolution priority:
    #   1) --run-dir (explicit path)
    #   2) --outdir (explicit path)
    #   3) --run-tag (predictions_output/<tag>)
    #   4) default (predictions_output/<YYYY-MM-DD> in UTC)
    if getattr(args, "run_dir", None):
        outdir = Path(str(args.run_dir))
    elif getattr(args, "outdir", None):
        outdir = Path(str(args.outdir))
    elif getattr(args, "run_tag", None):
        outdir = Path("predictions_output") / str(args.run_tag)
    else:
        outdir = Path("predictions_output") / dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    outdir.mkdir(parents=True, exist_ok=True)
    if bool(getattr(args, "debug", False)):
        try:
            print(f"[bookie_allmarkets] outdir={outdir}")
        except Exception:
            pass

    rows: List[Dict[str, Any]] = []
    rate_maps: List[pd.DataFrame] = []  # carry rate cols by fixture_key into ALLMARKETS
    lambda_maps: List[pd.DataFrame] = []  # carry λ cols by fixture_key into ALLMARKETS
    uefa_maps: List[pd.DataFrame] = []

    for lg in leagues:
        # Load two sources:
        #  - FTR prefers real 1X2 coverage
        #  - Side markets prefer synth-first (for OU25 under/BTTS no etc)
        raw_ftr, src_ftr = _load_league_source(lg, require_ftr_1x2=True)
        raw_side, src_side = _load_league_source(lg, require_ftr_1x2=False)
        # Alias names for debug readability (requested as p_side/p_ftr)
        p_side = src_side
        p_ftr = src_ftr

        # fallback reuse if one side is missing
        if raw_side is None or raw_side.empty:
            raw_side = raw_ftr.copy() if isinstance(raw_ftr, pd.DataFrame) else pd.DataFrame()
            src_side = src_ftr
        if raw_ftr is None or raw_ftr.empty:
            raw_ftr = raw_side.copy() if isinstance(raw_side, pd.DataFrame) else pd.DataFrame()
            src_ftr = src_side

        if (raw_side is None or raw_side.empty) and (raw_ftr is None or raw_ftr.empty):
            if bool(getattr(args, "debug", False)):
                try:
                    if bool(getattr(args, "debug", False)) or os.getenv("OG_DEBUG_PICK", "0").strip().lower() in ("1","true","yes","y"):
                        print(f"[bookie_allmarkets] {lg}: p_side={str(p_side) if 'p_side' in locals() else '<na>'} | p_ftr={str(p_ftr) if 'p_ftr' in locals() else '<na>'}")
                        if 'raw_side' in locals() and isinstance(raw_side, pd.DataFrame):
                            print(f"[bookie_allmarkets] {lg}: raw_side rows={len(raw_side)} cols={len(raw_side.columns)}")
                        if 'raw_ftr' in locals() and isinstance(raw_ftr, pd.DataFrame):
                            print(f"[bookie_allmarkets] {lg}: raw_ftr rows={len(raw_ftr)} cols={len(raw_ftr.columns)}")
                except Exception:
                    pass
                print(f"[bookie_allmarkets] {lg}: no rows available (sources empty)")
            continue

        if bool(getattr(args, "debug", False)):
            s_side = src_side.name if src_side is not None else "<none>"
            s_ftr  = src_ftr.name  if src_ftr  is not None else "<none>"
            print(f"[bookie_allmarkets] {lg}: src_side={s_side} | src_ftr={s_ftr}")

            if "ftr" in markets and src_ftr is not None:
                try:
                    v1x2 = _count_valid_1x2_rows_in_window(src_ftr, args.date_from, args.date_to)
                    print(f"[bookie_allmarkets] {lg}: ftr_source={src_ftr.name} | window_valid_1x2_rows={v1x2}")
                    if int(v1x2) == 0:
                        print(
                            f"[bookie_allmarkets] {lg}: ⚠️ no valid 1X2 odds rows in-window; "
                            f"FTR rows may not emit (bookie_pick requires od_home/od_draw/od_away > 1.0)."
                        )
                except Exception:
                    pass

        # Helpful debug: confirms we're actually consuming fd_odds_enriched.csv when available
        try:
            if (src_side is not None) and (src_side.name == "fd_odds_enriched.csv"):
                print(f"🧩 Using FD-enriched matches for {lg} (side): {src_side}")
            if (src_ftr is not None) and (src_ftr.name == "fd_odds_enriched.csv"):
                print(f"🧩 Using FD-enriched matches for {lg} (ftr): {src_ftr}")
        except Exception:
            pass

        # IMPORTANT:
        # Run streaks/H2H/rates on FULL history (raw_side) before window filtering.
        raw = raw_side

        # --- Optional: attach leak-safe rolling team rates BEFORE window filter ---
        # Important: compute on the full league frame so shift(1) has full context.
        # Ensure stable chronological ordering first.
        try:
            # Avoid noisy pandas warnings when formats vary across source CSVs
            try:
                _md_dt = pd.to_datetime(raw.get("match_date"), errors="coerce", utc=True, format="mixed", cache=True)
            except TypeError:
                _md_dt = pd.to_datetime(raw.get("match_date"), errors="coerce", utc=True, cache=True)
            if isinstance(_md_dt, pd.Series) and _md_dt.notna().any():
                raw = (
                    raw.assign(__md_dt=_md_dt)
                       .sort_values(["__md_dt", "home_team_name", "away_team_name"], kind="mergesort")
                       .drop(columns=["__md_dt"], errors="ignore")
                )
        except Exception:
            pass

        # Prefer lightweight rate attachment; fall back to the orchestrator if needed.
        if callable(_attach_team_rates):
            try:
                raw = _attach_team_rates(raw, lookbacks=(5, 10))
            except Exception as _e:
                print(f"ℹ️ attach_team_rates skipped for {lg}: {_e}")
        elif callable(_attach_streaks_and_h2h):
            try:
                # Orchestrator can also stamp the rate fields; keep extras minimal.
                raw = _attach_streaks_and_h2h(
                    raw,
                    team_lookbacks=(5, 10),
                    h2h_lookbacks=(5, 8),
                    include_implied_vs_actual=False,
                    include_composites=False,
                )
            except Exception as _e:
                print(f"ℹ️ attach_streaks_and_h2h skipped for {lg}: {_e}")

        # --- Optional: attach H2H streaks BEFORE window filter ---
        if getattr(args, "enable_h2h", False) and callable(_attach_h2h_streaks):
            try:
                raw = _attach_h2h_streaks(raw, lookbacks=(5, 8))
            except Exception as _e:
                print(f"ℹ️ attach_h2h_streaks skipped for {lg}: {_e}")
        # --- Optional: attach trained specialist heads (GE2/GE3/FTS) BEFORE window filter ---
        if callable(_score_trained_markets):
            try:
                raw = _score_trained_markets(
                raw,
                lg,
                markets=["btts_fh", "home_ge2", "away_ge2", "home_ge3", "away_ge3", "home_fts", "away_fts"],
                model_root=str(modelstore),
            )
            except Exception as _e:
                print(f"ℹ️ score_trained_markets skipped for {lg}: {_e}")
        # ------------------------------------------------------------------
        # TG sanity: some runs show GE3 probabilities inverted (near-1.0 for most rows),
        # which produces impossible outputs (HOME_TG25 and AWAY_TG25 both ~1).
        # If the GE3 confidence distribution looks inverted, flip: p = 1 - p.
        # This is a defensive runtime fix until train_markets.py is audited.
        # ------------------------------------------------------------------
        try:
            def _maybe_flip_prob(col: str, *, mean_hi: float, q90_hi: float) -> None:
                if col not in raw.columns:
                    return
                s = pd.to_numeric(raw[col], errors="coerce").clip(0.0, 1.0)
                finite = s[np.isfinite(s)]
                if len(finite) < 30:
                    raw[col] = s
                    return
                mu = float(finite.mean())
                q90 = float(finite.quantile(0.90))
                if (mu > float(mean_hi)) and (q90 > float(q90_hi)):
                    raw[col] = (1.0 - s).clip(0.0, 1.0)
                    if bool(getattr(args, "debug", False)):
                        print(f"[bookie_allmarkets] {lg}: flipped {col} (mu={mu:.3f}, q90={q90:.3f})")
                else:
                    raw[col] = s

            # GE3 should be rare; if it looks common, it is likely inverted.
            _maybe_flip_prob("home_ge3_confidence", mean_hi=0.60, q90_hi=0.80)
            _maybe_flip_prob("away_ge3_confidence", mean_hi=0.60, q90_hi=0.80)

            # GE2 is less rare; only flip if it's extremely skewed toward 1.
            _maybe_flip_prob("home_ge2_confidence", mean_hi=0.80, q90_hi=0.95)
            _maybe_flip_prob("away_ge2_confidence", mean_hi=0.80, q90_hi=0.95)
        except Exception:
            pass
        if getattr(args, "debug", False):
            _dbg_cols = [
                "scored_rate_5_home", "scored_rate_5_away",
                "clean_sheet_rate_5_home", "clean_sheet_rate_5_away",
            ]
            parts = []
            for c in _dbg_cols:
                if c in raw.columns:
                    nn = pd.to_numeric(raw[c], errors="coerce").notna().mean()
                    parts.append(f"{c}:nn={nn:.3f}")
                else:
                    parts.append(f"{c}:MISSING")
            print(f"[bookie_allmarkets] {lg} rate-cols(raw) -> " + ", ".join(parts))

        cols = _resolve_odds_cols(raw)
        cols_ftr = _resolve_odds_cols(raw_ftr) if ("ftr" in markets) else cols
        # --- OU25 safeguard ---
        # Some leagues/files may not have an UNDER25 odds column (or it may exist but be empty/invalid).
        # In that case `_bookie_pick_ou25()` already falls back to OVER25-only picks.
        # Here we also treat an all-missing/invalid under column as "missing" to avoid confusion.
        try:
            if "ou25" in markets:
                ucol = str(cols.get("ou25_under", "") or "")
                u_ok = False
                if ucol and (ucol in raw.columns):
                    uu = pd.to_numeric(raw[ucol], errors="coerce")
                    # consider it usable only if we have at least one finite odds > 1.0
                    u_ok = bool((uu.notna() & (uu > 1.0)).any())
                if not u_ok:
                    cols["ou25_under"] = ""
                    if bool(getattr(args, "debug", False)):
                        print(f"[bookie_allmarkets] {lg}: OU25 UNDER odds not present in source -> will synthesize fair UNDER via lambda/model proxy; picks may still be OVER25-only.")
        except Exception:
            pass

        # --------------------------------------------------------------
        # Attach per-match power ratings BEFORE window filtering.
        # Reason: FTR rows are emitted from the FTR-preferred frame; if we
        # only attach power to the side-market frame, FTR coverage becomes 0%.
        # We also canonicalise/alias columns first so fixture_key/team/date
        # fields are stable for the power join.
        # --------------------------------------------------------------
        try:
            if raw is not None and isinstance(raw, pd.DataFrame) and not raw.empty:
                raw = _apply_feature_aliases(raw)
                raw = _attach_power_ratings(raw, lg, modelstore)
            if raw_ftr is not None and isinstance(raw_ftr, pd.DataFrame) and not raw_ftr.empty:
                raw_ftr = _apply_feature_aliases(raw_ftr)
                raw_ftr = _attach_power_ratings(raw_ftr, lg, modelstore)
        except Exception:
            pass
        # Ensure aliases + power ratings are attached BEFORE window filter
        # (FTR pool was missing power columns because aliases/PR were attached only on the side path)
        try:
            raw = _attach_power_ratings(_apply_feature_aliases(raw), lg, modelstore)
        except Exception:
            try:
                raw = _apply_feature_aliases(raw)
            except Exception:
                pass

        try:
            raw_ftr = _attach_power_ratings(_apply_feature_aliases(raw_ftr), lg, modelstore)
        except Exception:
            try:
                raw_ftr = _apply_feature_aliases(raw_ftr)
            except Exception:
                pass
        # Window filter AFTER streak/H2H/rates are attached
        dfw_side = _filter_window(raw, args.date_from, args.date_to)
        dfw_ftr  = _filter_window(raw_ftr, args.date_from, args.date_to)

        # fallback reuse if one side is missing
        if dfw_side is None or dfw_side.empty:
            dfw_side = dfw_ftr.copy()
        if dfw_ftr is None or dfw_ftr.empty:
            dfw_ftr = dfw_side.copy()

        # Ensure unique, stable row indexing for downstream get_loc/index-based logic
        # (non-unique indices can cause get_loc to return slices/arrays)
        try:
            if dfw_side is not None and not dfw_side.empty:
                dfw_side = dfw_side.reset_index(drop=True)
            if dfw_ftr is not None and not dfw_ftr.empty:
                dfw_ftr = dfw_ftr.reset_index(drop=True)
        except Exception:
            pass

        # Canonicalise column names + ensure fixture_key/match_date are present on WINDOW frames
        # (needed for power join + consistent downstream feature access)
        try:
            if dfw_side is not None and not dfw_side.empty:
                dfw_side = _apply_feature_aliases(dfw_side)
            if dfw_ftr is not None and not dfw_ftr.empty:
                dfw_ftr = _apply_feature_aliases(dfw_ftr)
        except Exception:
            pass

        # If H2H is enabled but key H2H cols are missing/empty on the window frame,
        # try a best-effort attach on the window itself.
        try:
            if getattr(args, "enable_h2h", False) and callable(_attach_h2h_streaks):
                _need_h2h = False
                for _c in ("h2h_n", "h2h_btts_rate", "h2h_over25_rate", "h2h_goaliness_avg"):
                    if (_c not in dfw_side.columns) or (pd.to_numeric(dfw_side.get(_c), errors="coerce").notna().sum() == 0):
                        _need_h2h = True
                        break
                if _need_h2h and (dfw_side is not None) and (not dfw_side.empty):
                    dfw_side = _attach_h2h_streaks(dfw_side, lookbacks=(5, 8))
        except Exception:
            pass

        if (dfw_side is None or dfw_side.empty) and (dfw_ftr is None or dfw_ftr.empty):
            if bool(getattr(args, "debug", False)):
                print(f"[bookie_allmarkets] {lg}: no rows in window {args.date_from}..{args.date_to}")
            continue

        # preserve existing downstream logic (dfw default is side markets)
        dfw = dfw_side
        # Ensure dfw also has a clean RangeIndex
        try:
            if dfw is not None and not dfw.empty:
                dfw = dfw.reset_index(drop=True)
        except Exception:
            pass
        if getattr(args, "debug", False) and (dfw is not None) and (not dfw.empty):
            _dbg_cols = [
                "scored_rate_5_home", "scored_rate_5_away",
                "clean_sheet_rate_5_home", "clean_sheet_rate_5_away",
            ]
            parts = []
            for c in _dbg_cols:
                if c in dfw.columns:
                    nn = pd.to_numeric(dfw[c], errors="coerce").notna().mean()
                    parts.append(f"{c}:nn={nn:.3f}")
                else:
                    parts.append(f"{c}:MISSING")
            print(f"[bookie_allmarkets] {lg} rate-cols(window) -> " + ", ".join(parts))
        if dfw.empty:
            continue
        
        # Build fixture keys early so fixture-level carry maps can use the canonical join key.
        try:
            dfw["fixture_key"] = dfw.apply(_match_key, axis=1)
        except Exception:
            dfw["fixture_key"] = ""
        dfw["fixture_key"] = dfw["fixture_key"].astype("string").fillna("").str.strip()
        try:
            dfw["fixture_key_ascii"] = dfw.apply(_match_key_ascii, axis=1)
        except Exception:
            dfw["fixture_key_ascii"] = ""
        dfw["fixture_key_ascii"] = dfw["fixture_key_ascii"].astype("string").fillna("").str.strip()
        dfw["league"] = lg
        try:
            if ("fixture_key" not in dfw_ftr.columns) or dfw_ftr["fixture_key"].astype("string").fillna("").str.strip().eq("").all():
                dfw_ftr["fixture_key"] = dfw_ftr.apply(_match_key, axis=1)
            if ("fixture_key_ascii" not in dfw_ftr.columns) or dfw_ftr["fixture_key_ascii"].astype("string").fillna("").str.strip().eq("").all():
                dfw_ftr["fixture_key_ascii"] = dfw_ftr.apply(_match_key_ascii, axis=1)
            dfw_ftr["fixture_key"] = dfw_ftr["fixture_key"].astype("string").fillna("").str.strip()
            dfw_ftr["fixture_key_ascii"] = dfw_ftr["fixture_key_ascii"].astype("string").fillna("").str.strip()
        except Exception:
            pass

        # Build a fixture-level lookup for deploy-safe context fields from the canonical
        # merged file. Side-market sources such as fd_odds_enriched(_synth).csv often do
        # not carry matchup snapshot or H2H features, but deploy lanes need them.
        merged_context_cols = [
            "snap_timing_early_goal_pressure",
            "snap_timing_second_half_acceleration",
            "snap_home_first_to_score_edge",
            "snap_ht_goal_regime_blend",
            "snap_ou25_over_regime_blend",
            "h2h_n",
            "h2h_btts_rate",
            "h2h_over25_rate",
            "h2h_goaliness_avg",
        ]
        merged_context_lookup: dict[str, dict[str, float]] = {}
        merged_context_lookup_ascii: dict[str, dict[str, float]] = {}
        try:
            merged_timing_path = matches_root / str(getattr(args, "merged_subdir", "__merged__")) / f"{tag}__merged.csv"
            if merged_timing_path.exists():
                wanted = {
                    "match_date",
                    "home_team_name",
                    "away_team_name",
                    "fixture_key",
                    "fixture_key_ascii",
                    *merged_context_cols,
                }
                mt = pd.read_csv(merged_timing_path, usecols=lambda c: c in wanted)
                mt = _apply_feature_aliases(mt)
                mt = _filter_window(mt, args.date_from, args.date_to)
                if mt is not None and not mt.empty:
                    if ("fixture_key" not in mt.columns) or mt["fixture_key"].astype("string").fillna("").str.strip().eq("").all():
                        try:
                            mt["fixture_key"] = mt.apply(_match_key, axis=1)
                        except Exception:
                            mt["fixture_key"] = ""
                    if ("fixture_key_ascii" not in mt.columns) or mt["fixture_key_ascii"].astype("string").fillna("").str.strip().eq("").all():
                        try:
                            mt["fixture_key_ascii"] = mt.apply(_match_key_ascii, axis=1)
                        except Exception:
                            mt["fixture_key_ascii"] = ""
                    mt["fixture_key"] = mt["fixture_key"].astype("string").fillna("").str.strip()
                    mt["fixture_key_ascii"] = mt["fixture_key_ascii"].astype("string").fillna("").str.strip()
                    keep_cols = ["fixture_key", "fixture_key_ascii", *[c for c in merged_context_cols if c in mt.columns]]
                    mt = mt[keep_cols].drop_duplicates(subset=["fixture_key"], keep="last")
                    for _, rr in mt.iterrows():
                        payload = {c: _num_from_row(rr, c) for c in merged_context_cols}
                        fk = str(rr.get("fixture_key", "") or "").strip()
                        fk_ascii = str(rr.get("fixture_key_ascii", "") or "").strip()
                        if fk:
                            merged_context_lookup[fk] = payload
                        if fk_ascii:
                            merged_context_lookup_ascii[fk_ascii] = payload
        except Exception:
            merged_context_lookup = {}
            merged_context_lookup_ascii = {}

        def _apply_merged_context_lookup(df_part: pd.DataFrame) -> pd.DataFrame:
            if df_part is None or df_part.empty:
                return df_part
            out_part = df_part.copy()
            for _c in merged_context_cols:
                if _c not in out_part.columns:
                    out_part[_c] = np.nan
            for _ix, _rr in out_part.iterrows():
                _fk = str(_rr.get("fixture_key", "") or "").strip()
                _fk_ascii = str(_rr.get("fixture_key_ascii", "") or "").strip()
                _fb = merged_context_lookup.get(_fk) or merged_context_lookup_ascii.get(_fk_ascii) or {}
                if not _fb:
                    continue
                for _c in merged_context_cols:
                    _cur = _num_from_row(_rr, _c)
                    _new = _fb.get(_c, np.nan)
                    if (not np.isfinite(_cur)) and np.isfinite(_new):
                        out_part.at[_ix, _c] = float(_new)
            return out_part

        dfw = _apply_merged_context_lookup(dfw)
        dfw_ftr = _apply_merged_context_lookup(dfw_ftr)

        # --- Attach λ (goal preds) + correct-score shortlist on the WINDOW frame ---
        dfw = _ensure_goal_preds_window(dfw, lg)
        dfw = _apply_absurd_lambda_and_fts_sanity(dfw, lg, debug=bool(getattr(args, "debug", False)))
        dfw = _attach_poisson_cs_top3(dfw, max_goals=6)
        dfw = _attach_team_poisson_tails(dfw)
        dfw = _attach_phase8a_grid_features(dfw, max_goals=6)
        # --------------------------------------------------------------
        # FTR odds source (post-enrichment):
        # Keep ALL enrichments/keys from dfw (side window), but source 1X2 odds
        # from dfw_ftr (FTR-preferred window) via fixture_key.
        # IMPORTANT: preserve dfw's original index so model arrays stay aligned.
        # --------------------------------------------------------------
        dfw_ftr_ready = dfw
        try:
            if ("ftr" in markets) and (dfw_ftr is not None) and (not dfw_ftr.empty):
                # Ensure fixture_key exists on dfw_ftr
                if ("fixture_key" not in dfw_ftr.columns) or (
                    dfw_ftr["fixture_key"].astype("string").fillna("").str.strip().eq("").all()
                ):
                    try:
                        dfw_ftr["fixture_key"] = dfw_ftr.apply(_match_key, axis=1)
                    except Exception:
                        dfw_ftr["fixture_key"] = ""
                dfw_ftr["fixture_key"] = dfw_ftr["fixture_key"].astype("string").fillna("").str.strip()

                # Build an ASCII-safe key too. The merged window frame often carries
                # folded team names while season/source files can preserve diacritics,
                # so a fixture_key-only merge can silently drop valid FTR odds.
                if ("fixture_key_ascii" not in dfw_ftr.columns) or (
                    dfw_ftr["fixture_key_ascii"].astype("string").fillna("").str.strip().eq("").all()
                ):
                    try:
                        dfw_ftr["fixture_key_ascii"] = dfw_ftr.apply(_match_key_ascii, axis=1)
                    except Exception:
                        dfw_ftr["fixture_key_ascii"] = ""
                dfw_ftr["fixture_key_ascii"] = dfw_ftr["fixture_key_ascii"].astype("string").fillna("").str.strip()

                # Identify the actual 1X2 odds columns present on dfw_ftr
                try:
                    _odc_ftr = _resolve_odds_cols(dfw_ftr)
                except Exception:
                    _odc_ftr = {}

                _ftr_cols = []
                for _k in ("ftr_home", "ftr_draw", "ftr_away"):
                    _c = str(_odc_ftr.get(_k) or "").strip()
                    if _c and (_c in dfw_ftr.columns):
                        _ftr_cols.append(_c)
                _ftr_cols = list(dict.fromkeys(_ftr_cols))

                if _ftr_cols:
                    # Keep dfw index alignment stable
                    _lhs = dfw.copy()
                    _lhs["__row_id"] = _lhs.index

                    _rhs = dfw_ftr[["fixture_key", "fixture_key_ascii"] + _ftr_cols].copy()
                    _rhs_key = _rhs.drop_duplicates(subset=["fixture_key"], keep="first")
                    _rhs_ascii = _rhs.drop_duplicates(subset=["fixture_key_ascii"], keep="first")

                    _m = _lhs.merge(_rhs_key, on="fixture_key", how="left", suffixes=("", "_ftrsrc"))
                    _m = _m.set_index("__row_id")
                    _m = _m.reindex(dfw.index)

                    # If dfw already had those odds cols, keep its values and fill gaps from FTR source.
                    for _c in _ftr_cols:
                        _src = f"{_c}_ftrsrc"
                        if _src in _m.columns:
                            if _c in dfw.columns:
                                _m[_c] = _m[_c].where(_m[_c].notna(), _m[_src])
                            else:
                                _m[_c] = _m[_src]
                            _m = _m.drop(columns=[_src], errors="ignore")

                    # Second pass: resolve any remaining missing odds rows via ASCII-safe fixture identity.
                    _missing_mask = pd.Series(False, index=_m.index)
                    for _c in _ftr_cols:
                        if _c in _m.columns:
                            _missing_mask = _missing_mask | pd.to_numeric(_m[_c], errors="coerce").isna()
                    if bool(_missing_mask.any()):
                        _lhs_missing = _m.loc[_missing_mask].copy()
                        if "fixture_key_ascii" in _lhs_missing.columns and _lhs_missing["fixture_key_ascii"].astype("string").fillna("").str.strip().ne("").any():
                            _lhs_missing["__row_id_ascii"] = _lhs_missing.index
                            _m2 = _lhs_missing.merge(
                                _rhs_ascii,
                                on="fixture_key_ascii",
                                how="left",
                                suffixes=("", "_ftrascii"),
                            ).set_index("__row_id_ascii")
                            for _c in _ftr_cols:
                                _src2 = f"{_c}_ftrascii"
                                if _src2 in _m2.columns:
                                    _m.loc[_m2.index, _c] = pd.to_numeric(
                                        _m.loc[_m2.index, _c], errors="coerce"
                                    ).where(
                                        pd.to_numeric(_m.loc[_m2.index, _c], errors="coerce").notna(),
                                        pd.to_numeric(_m2[_src2], errors="coerce"),
                                    )

                    dfw_ftr_ready = _m
        except Exception:
            dfw_ftr_ready = dfw

        # Resolve FTR odds cols from the FTR-ready frame (used by _bookie_pick_ftr)
        try:
            cols_ftr = _resolve_odds_cols(dfw_ftr_ready) if ("ftr" in markets) else cols
        except Exception:
            cols_ftr = cols
        # --- DEBUG: sanity-check FTR 1X2 odds presence on dfw_ftr_ready ---
        if bool(getattr(args, "debug", False)) and ("ftr" in markets):
            try:
                print(
                    f"[bookie_allmarkets] {lg}: cols_ftr keys={ {k: cols_ftr.get(k) for k in ('ftr_home','ftr_draw','ftr_away')} }"
                )
                ch, cd, ca = cols_ftr.get("ftr_home", ""), cols_ftr.get("ftr_draw", ""), cols_ftr.get("ftr_away", "")
                if ch and cd and ca and all(c in dfw_ftr_ready.columns for c in (ch, cd, ca)):
                    oh = pd.to_numeric(dfw_ftr_ready[ch], errors="coerce")
                    od = pd.to_numeric(dfw_ftr_ready[cd], errors="coerce")
                    oa = pd.to_numeric(dfw_ftr_ready[ca], errors="coerce")
                    v = ((oh > 1.0) & (od > 1.0) & (oa > 1.0)).sum()
                    print(
                        f"[bookie_allmarkets] {lg}: dfw_ftr_ready valid 1X2 rows (odds>1.0) = {int(v)}/{len(dfw_ftr_ready)}"
                    )
                else:
                    print(f"[bookie_allmarkets] {lg}: ⚠️ cols_ftr columns not present on dfw_ftr_ready")
            except Exception as _e:
                print(f"[bookie_allmarkets] {lg}: cols_ftr sanity-check failed: {_e}")
        # --------------------------------------------------------------
        # Fixture-level λ map (shared across all markets)
        # Ensures every emitted market row (incl. FTR) inherits λ columns.
        # --------------------------------------------------------------
        try:
            _lam_cols = [
                "home_goals_pred",
                "away_goals_pred",
                "lambda_home",
                "lambda_away",
                "exp_goals_sum",
                "p00_est",
                "cs_home",
                "cs_away",
                "p_home_ge1",
                "p_away_ge1",
                "p_home_ge2",
                "p_away_ge2",
                "p_home_ge3",
                "p_away_ge3",
                "p_home_ge4",
                "p_away_ge4",
                "pois_home_ge2",
                "pois_away_ge2",
                "pois_home_ge3",
                "pois_away_ge3",
            ]
            _mlam = dfw.copy()
            for _c in _lam_cols:
                if _c not in _mlam.columns:
                    _mlam[_c] = np.nan
            _mlam = _mlam[["fixture_key"] + _lam_cols].copy()
            _mlam["league"] = lg
            lambda_maps.append(_mlam.drop_duplicates(subset=["league", "fixture_key"], keep="first"))
        except Exception:
            pass

        # --- UEFA context (table pressure / rotation / must-win flags) ---
        # Only applies when Teams/<league> exists (e.g. Champions League, Europa League).
        try:
            dfw = _build_uefa_match_context(lg, dfw)
        except Exception as _uefa_e:
            if bool(getattr(args, "debug", False)):
                print(f"[bookie_allmarkets] {lg} UEFA ctx attach FAILED: {_uefa_e}")

        # Capture any leak-safe rolling rate columns (if present) so the final ALLMARKETS
        # CSV can carry them even if individual rows.append dicts forget to include them.
        try:
            _rate_cols = [
                "scored_rate_5_home", "scored_rate_5_away",
                "clean_sheet_rate_5_home", "clean_sheet_rate_5_away",
                "conceded_rate_5_home", "conceded_rate_5_away",
                "rolling5_home_gc", "rolling5_away_gc",
                "btts_rate_5_home", "btts_rate_5_away",
                "over25_rate_5_home", "over25_rate_5_away",
                "under25_rate_5_home", "under25_rate_5_away",
                "goaliness_avg_5_home", "goaliness_avg_5_away",
                "xg_for_avg_5_home", "xg_for_avg_5_away",
                "xg_against_avg_5_home", "xg_against_avg_5_away",
                "gapm_diff", "clean_sheet_rate_diff",
                "home_xg_against_idx", "away_xg_against_idx",
                "defence_diff",
                "home_ge2_confidence", "away_ge2_confidence",
                "home_ge3_confidence", "away_ge3_confidence",
                "p_home_fts", "p_away_fts",
                "btts_fh_confidence", "btts_fh_pred",
                "home_goals_pred","away_goals_pred","lambda_home","lambda_away","exp_goals_sum","p00_est",
                "cs1","cs1_p","cs2","cs2_p","cs3","cs3_p","p_home_pois","p_draw_pois","p_away_pois","cs_trunc_mass_0_6",
                "pois_home_ge2","pois_away_ge2","pois_home_ge3","pois_away_ge3",
                "cs_mass_btts_yes","cs_mass_btts_no","cs_mass_over25","cs_mass_under25",
                "cs_mass_home_win","cs_mass_draw","cs_mass_away_win","cs_entropy",
                "both_teams_2plus_mass","mass_over25_via_one_sided_rout",
                "mass_0_goals","mass_1_goal","mass_2_goals","mass_3_goals","mass_4plus_goals",
                # UEFA context (if attached earlier in this league window)
                "uefa_home_state", "uefa_away_state",
                "uefa_home_gap24", "uefa_away_gap24", "uefa_gap24_diff",
                "uefa_home_rotation_risk", "uefa_away_rotation_risk",
                "uefa_both_must_win", "uefa_goal_hunt_flag", "uefa_pride_only_flag",
                "uefa_live_table_volatility", "uefa_vol_band_n",
                "uefa_home_must_win", "uefa_away_must_win",
                "uefa_home_must_avoid_loss", "uefa_away_must_avoid_loss",
                "uefa_home_eliminated", "uefa_away_eliminated",
            ]
            _h2h_cols = [
                "h2h_n", "h2h_btts_rate", "h2h_over25_rate", "h2h_goaliness_avg"
            ]
            _rm = dfw.copy()
            for _c in _rate_cols + _h2h_cols:
                if _c not in _rm.columns:
                    _rm[_c] = np.nan
            _rm = _rm[["fixture_key"] + _rate_cols + _h2h_cols].copy()
            _rm["league"] = lg
            rate_maps.append(_rm.drop_duplicates(subset=["league", "fixture_key"]))
        except Exception:
            pass

        # Build fixture_key once (used for power join + output)
        # Only compute if missing/blank to avoid overwriting canonicalised keys.
        if "fixture_key" not in dfw.columns:
            dfw["fixture_key"] = ""
        _fk0 = dfw["fixture_key"].astype("string").fillna("").str.strip()
        if bool((_fk0.eq("") | _fk0.isna()).any()):
            try:
                dfw.loc[_fk0.eq("") | _fk0.isna(), "fixture_key"] = dfw.loc[_fk0.eq("") | _fk0.isna()].apply(_match_key, axis=1)
            except Exception:
                # best effort
                pass
        dfw["fixture_key"] = dfw["fixture_key"].astype("string").fillna("").str.strip()

        # Also build an ASCII-folded fixture key for league-safe rescue joins (diacritics-safe)
        if "fixture_key_ascii" not in dfw.columns:
            dfw["fixture_key_ascii"] = ""
        _fka0 = dfw["fixture_key_ascii"].astype("string").fillna("").str.strip()
        if bool((_fka0.eq("") | _fka0.isna()).any()):
            try:
                dfw.loc[_fka0.eq("") | _fka0.isna(), "fixture_key_ascii"] = dfw.loc[_fka0.eq("") | _fka0.isna()].apply(_match_key_ascii, axis=1)
            except Exception:
                # best effort
                pass
        dfw["fixture_key_ascii"] = dfw["fixture_key_ascii"].astype("string").fillna("").str.strip()

        # Attach per-match power ratings (if available) BEFORE constructing dfw_ftr_ready
        # so FTR rows inherit power columns too.
        try:
            dfw = _attach_power_ratings(dfw, lg, modelstore)
        except Exception:
            pass

        # Prepare minimal X frame for models: numeric+string columns are OK; alignment adds missing zeros
        X = dfw.copy()
        # Track where this league's emitted rows start (for per-league diagnostics)
        _rows_start = len(rows)

        # load bundles for this league
        b_ftr = _load_bundle(modelstore, lg, "ftr", engine="cat") if "ftr" in markets else None
        b_ftr = _filter_bundle_engine(b_ftr, "cat")
        b_ftr_xgb = _load_bundle(modelstore, lg, "ftr", engine="xgb") if "ftr" in markets else None
        b_ftr_xgb = _filter_bundle_engine(b_ftr_xgb, "xgb")
        b_ou  = _load_bundle(modelstore, lg, "over25") if "ou25" in markets else None
        b_ou_xgb = _load_bundle(modelstore, lg, "over25", engine="xgb") if "ou25" in markets else None
        b_ou_xgb = _filter_bundle_engine(b_ou_xgb, "xgb")
        b_u25 = _load_bundle(modelstore, lg, "under25") if "ou25" in markets else None
        b_bt  = _load_bundle(modelstore, lg, "btts") if "btts" in markets else None
        b_bt_xgb = _load_bundle(modelstore, lg, "btts", engine="xgb") if "btts" in markets else None
        b_bt_xgb = _filter_bundle_engine(b_bt_xgb, "xgb")
        # --- Attach per-fixture side-market probabilities to dfw (version-agnostic; may be v3) ---
        def _pos_prob(p):
            p = np.asarray(p)
            if p.ndim == 1:
                return p.astype(float)
            if p.ndim == 2 and p.shape[1] >= 2:
                return p[:, 1].astype(float)
            return np.nanmax(p, axis=1).astype(float)

        if dfw is not None and not dfw.empty:
            # Canonical, version-agnostic probability columns.
            # NOTE: legacy *_v2 columns are still populated for backward compatibility,
            # but they may actually come from v3 bundles.
            if "prob_over25" not in dfw.columns:
                dfw["prob_over25"] = np.nan
            if "prob_btts" not in dfw.columns:
                dfw["prob_btts"] = np.nan
            if "prob_over25_v2" not in dfw.columns:
                dfw["prob_over25_v2"] = np.nan
            if "prob_btts_v2" not in dfw.columns:
                dfw["prob_btts_v2"] = np.nan

            # OVER25 prob
            if "ou25" in markets:
                if isinstance(b_ou, dict):
                    try:
                        p_over = _pos_prob(_predict_proba(b_ou, dfw))
                        dfw["prob_over25"] = pd.Series(p_over, index=dfw.index)
                    except Exception:
                        pass
                elif isinstance(b_u25, dict):
                    # recover OVER25 = 1 - P(UNDER25)
                    try:
                        p_u = _pos_prob(_predict_proba(b_u25, dfw))
                        dfw["prob_over25"] = 1.0 - pd.Series(p_u, index=dfw.index)
                    except Exception:
                        pass

            # BTTS YES prob
            if "btts" in markets and isinstance(b_bt, dict):
                try:
                    p_yes = _pos_prob(_predict_proba(b_bt, dfw))
                    dfw["prob_btts"] = pd.Series(p_yes, index=dfw.index)
                except Exception:
                    pass

            dfw["prob_over25"] = pd.to_numeric(dfw["prob_over25"], errors="coerce").clip(0, 1)
            dfw["prob_btts"]  = pd.to_numeric(dfw["prob_btts"], errors="coerce").clip(0, 1)

            # Back-compat aliases (may be v3-derived)
            dfw["prob_over25_v2"] = dfw["prob_over25"]
            dfw["prob_btts_v2"] = dfw["prob_btts"]

        # --- DEBUG: report which FTR bundle we actually loaded (prevents silent v2/v3 regressions) ---
        if bool(getattr(args, "debug", False)) and ("ftr" in markets):
            tag = _league_tag(lg)
            if isinstance(b_ftr, dict):
                pth = str(b_ftr.get("_bundle_path", ""))
                pth_disp = pth if pth else f"ModelStore/{tag}/(unknown)"
                fv = str(b_ftr.get("ftr_version") or ("v3" if pth.endswith("ftr_v3.pkl") else "v2"))
                uses_lam = bool(b_ftr.get("uses_lambda_features")) if ("uses_lambda_features" in b_ftr) else False
                nfeat = len(b_ftr.get("features", []) or [])
                lam_cols = b_ftr.get("lambda_feature_cols", [])
                print(
                    f"[bookie_allmarkets] {lg}: FTR bundle loaded={fv} "
                    f"path={pth_disp} n_features={nfeat} uses_lambda={uses_lam} lambda_cols={lam_cols}"
                )
            else:
                print(
                    f"[bookie_allmarkets] {lg}: FTR bundle MISSING "
                    f"(no ftr_v2.pkl/ftr_v3.pkl found under ModelStore/{tag}/)"
                )
        # --- DEBUG: report which SIDE bundles we actually loaded (prevents silent v2/v3 regressions) ---
        if bool(getattr(args, "debug", False)):
            tag = _league_tag(lg)

            # OU25 heads (can be over25/under25 legacy naming)
            if "ou25" in markets:
                if isinstance(b_ou, dict):
                    pth = str(b_ou.get("_bundle_path", ""))
                    pth_disp = pth if pth else f"ModelStore/{tag}/(unknown)"
                    fv = "v3" if pth.endswith("_v3.pkl") else ("v2" if pth.endswith("_v2.pkl") else "?")
                    nfeat = len(b_ou.get("features", []) or [])
                    print(f"[bookie_allmarkets] {lg}: OU25(OVER) bundle loaded={fv} path={pth_disp} n_features={nfeat}")
                if isinstance(b_ou_xgb, dict):
                    pth = str(b_ou_xgb.get("_bundle_path", ""))
                    pth_disp = pth if pth else f"ModelStore/{tag}/xgb/(unknown)"
                    fv = "v3" if pth.endswith("_v3.pkl") else ("v2" if pth.endswith("_v2.pkl") else "?")
                    nfeat = len(b_ou_xgb.get("features", []) or [])
                    print(f"[bookie_allmarkets] {lg}: OU25(OVER) XGB bundle loaded={fv} path={pth_disp} n_features={nfeat}")
                elif isinstance(b_u25, dict):
                    pth = str(b_u25.get("_bundle_path", ""))
                    pth_disp = pth if pth else f"ModelStore/{tag}/(unknown)"
                    fv = "v3" if pth.endswith("_v3.pkl") else ("v2" if pth.endswith("_v2.pkl") else "?")
                    nfeat = len(b_u25.get("features", []) or [])
                    print(f"[bookie_allmarkets] {lg}: OU25(UNDER) bundle loaded={fv} path={pth_disp} n_features={nfeat}")
                else:
                    print(f"[bookie_allmarkets] {lg}: OU25 bundle MISSING (no over25/under25 ou25 bundles found under ModelStore/{tag}/)")

            # BTTS head
            if "btts" in markets:
                if isinstance(b_bt, dict):
                    pth = str(b_bt.get("_bundle_path", ""))
                    pth_disp = pth if pth else f"ModelStore/{tag}/(unknown)"
                    fv = "v3" if pth.endswith("_v3.pkl") else ("v2" if pth.endswith("_v2.pkl") else "?")
                    nfeat = len(b_bt.get("features", []) or [])
                    print(f"[bookie_allmarkets] {lg}: BTTS bundle loaded={fv} path={pth_disp} n_features={nfeat}")
                else:
                    print(f"[bookie_allmarkets] {lg}: BTTS bundle MISSING (no btts bundles found under ModelStore/{tag}/)")
                if isinstance(b_bt_xgb, dict):
                    pth = str(b_bt_xgb.get("_bundle_path", ""))
                    pth_disp = pth if pth else f"ModelStore/{tag}/xgb/(unknown)"
                    fv = "v3" if pth.endswith("_v3.pkl") else ("v2" if pth.endswith("_v2.pkl") else "?")
                    nfeat = len(b_bt_xgb.get("features", []) or [])
                    print(f"[bookie_allmarkets] {lg}: BTTS XGB bundle loaded={fv} path={pth_disp} n_features={nfeat}")
            # --- FTR ---
        if b_ftr:
            try:
                P = _predict_proba(b_ftr, X)  # (n,3)
                # assume classes order 0/1/2 => HOME/DRAW/AWAY in your trainer
                conf_home = P[:, 0].astype(float)
                conf_draw = P[:, 1].astype(float)
                conf_away = P[:, 2].astype(float)
                top1 = np.max(P, axis=1)
                top2 = np.sort(P, axis=1)[:, -2]
                margin = (top1 - top2).astype(float)
            except Exception:
                conf_home = conf_draw = conf_away = margin = None

            # Optional XGBoost parallel FTR head
            conf_home_xgb = conf_draw_xgb = conf_away_xgb = margin_xgb = None
            if b_ftr_xgb:
                try:
                    P_xgb = _predict_proba(b_ftr_xgb, X)
                    conf_home_xgb = P_xgb[:, 0].astype(float)
                    conf_draw_xgb = P_xgb[:, 1].astype(float)
                    conf_away_xgb = P_xgb[:, 2].astype(float)
                    top1_xgb = np.max(P_xgb, axis=1)
                    top2_xgb = np.sort(P_xgb, axis=1)[:, -2]
                    margin_xgb = (top1_xgb - top2_xgb).astype(float)
                except Exception as _e:
                    if os.getenv("OG_XGB_INFER_DEBUG", "0").strip().lower() in ("1", "true", "yes", "y"):
                        print(f"[xgb-infer] {lg}: predict_proba failed: {_e}")
                    conf_home_xgb = conf_draw_xgb = conf_away_xgb = margin_xgb = None
            if os.getenv("OG_XGB_INFER_DEBUG", "0").strip().lower() in ("1", "true", "yes", "y"):
                if conf_home_xgb is None:
                    print(f"[xgb-infer] {lg}: xgb predictions missing (conf_home_xgb=None)")
                else:
                    try:
                        print(f"[xgb-infer] {lg}: xgb predictions rows={len(conf_home_xgb)}")
                    except Exception:
                        print(f"[xgb-infer] {lg}: xgb predictions rows=unknown")

            def _timing_context_row(r0) -> dict:
                """Carry deploy-safe timing/context fields from merged inputs into the board."""
                out_ctx = {
                    "snap_timing_early_goal_pressure": _num_from_row(r0, "snap_timing_early_goal_pressure"),
                    "snap_timing_second_half_acceleration": _num_from_row(r0, "snap_timing_second_half_acceleration"),
                    "snap_home_first_to_score_edge": _num_from_row(r0, "snap_home_first_to_score_edge"),
                    "snap_ht_goal_regime_blend": _num_from_row(r0, "snap_ht_goal_regime_blend"),
                    "snap_ou25_over_regime_blend": _num_from_row(r0, "snap_ou25_over_regime_blend"),
                }
                if all(not np.isfinite(v) for v in out_ctx.values()):
                    fk = str(r0.get("fixture_key", "") or "").strip()
                    fk_ascii = str(r0.get("fixture_key_ascii", "") or "").strip()
                    fb = merged_context_lookup.get(fk) or merged_context_lookup_ascii.get(fk_ascii) or {}
                    for k, v in fb.items():
                        if k in out_ctx and not np.isfinite(out_ctx[k]) and np.isfinite(v):
                            out_ctx[k] = float(v)
                return out_ctx

            for i, r in dfw_ftr_ready.iterrows():
                b = _bookie_pick_ftr(r, cols_ftr, ftr_implied_min)
                if not b or conf_home is None:
                    continue
                pick = b["bookie_pick"]
                # Pre-match PPG (used for glue/anchor filters)
                ppg_home_pre = _num_from_row(r, "Pre-Match PPG (Home)", "home_ppg", "Pre-Match PPG Home")
                ppg_away_pre = _num_from_row(r, "Pre-Match PPG (Away)", "away_ppg", "Pre-Match PPG Away")
                try:
                    ppg_diff_pre = float(ppg_home_pre - ppg_away_pre)
                except Exception:
                    ppg_diff_pre = np.nan

                # --- Derived debug flags (always computed, even if gates are off) ---
                _glue_dmin = float(getattr(args, "ftr_glue_ppg_diff_min", 0.70))
                _glue_opp_max = float(getattr(args, "ftr_glue_ppg_opp_max", 1.00))

                glue_ok = False
                if pick == "HOME":
                    glue_ok = (
                        np.isfinite(ppg_diff_pre)
                        and (ppg_diff_pre >= _glue_dmin)
                        and np.isfinite(ppg_away_pre)
                        and (float(ppg_away_pre) <= _glue_opp_max)
                    )
                elif pick == "AWAY":
                    glue_ok = (
                        np.isfinite(ppg_diff_pre)
                        and (ppg_diff_pre <= -_glue_dmin)
                        and np.isfinite(ppg_home_pre)
                        and (float(ppg_home_pre) <= _glue_opp_max)
                    )

                # draw-trap flag (only meaningful for HOME/AWAY favourites)
                _dt_od_max = float(getattr(args, "ftr_drawtrap_od_max", 1.30))
                _dt_opp_min = float(getattr(args, "ftr_drawtrap_opp_ppg_min", 1.20))
                try:
                    _bookie_od_sel = float(pd.to_numeric(b.get("bookie_od", np.nan), errors="coerce"))
                except Exception:
                    _bookie_od_sel = np.nan
                _opp_ppg = ppg_away_pre if pick == "HOME" else (ppg_home_pre if pick == "AWAY" else np.nan)
                drawtrap_flag = bool(
                    (pick in ("HOME", "AWAY"))
                    and np.isfinite(_bookie_od_sel)
                    and (_bookie_od_sel <= _dt_od_max)
                    and np.isfinite(_opp_ppg)
                    and (float(_opp_ppg) >= _dt_opp_min)
)

                # Optional PPG glue gate (favourites)
                if bool(getattr(args, "ftr_glue_use_ppg", False)) and pick in ("HOME", "AWAY"):
                    dmin = float(getattr(args, "ftr_glue_ppg_diff_min", 0.70))
                    opp_max = float(getattr(args, "ftr_glue_ppg_opp_max", 1.00))
                    if pick == "HOME":
                        if (not np.isfinite(ppg_diff_pre)) or (ppg_diff_pre < dmin) or (not np.isfinite(ppg_away_pre)) or (ppg_away_pre > opp_max):
                            continue
                    if pick == "AWAY":
                        if (not np.isfinite(ppg_diff_pre)) or (ppg_diff_pre > -dmin) or (not np.isfinite(ppg_home_pre)) or (ppg_home_pre > opp_max):
                            continue

                # Optional draw-trap veto for ultra-short favourites vs decent opponents
                if bool(getattr(args, "ftr_drawtrap_veto", False)) and pick in ("HOME", "AWAY"):
                    od_max = float(getattr(args, "ftr_drawtrap_od_max", 1.30))
                    opp_min = float(getattr(args, "ftr_drawtrap_opp_ppg_min", 1.20))
                    try:
                        bookie_od_sel = float(pd.to_numeric(b.get("bookie_od", np.nan), errors="coerce"))
                    except Exception:
                        bookie_od_sel = np.nan
                    opp_ppg = ppg_away_pre if pick == "HOME" else ppg_home_pre
                    if (np.isfinite(bookie_od_sel) and (bookie_od_sel <= od_max)) and (np.isfinite(opp_ppg) and (float(opp_ppg) >= opp_min)):
                        continue

                # Convenience fields (picked-side PPG)
                ppg_pick = ppg_home_pre if pick == "HOME" else (ppg_away_pre if pick == "AWAY" else np.nan)
                ppg_opp = ppg_away_pre if pick == "HOME" else (ppg_home_pre if pick == "AWAY" else np.nan)

                idx = dfw.index.get_loc(i)
                mp = {"HOME": conf_home[idx], "DRAW": conf_draw[idx], "AWAY": conf_away[idx]}[pick]
                model_top = ["HOME","DRAW","AWAY"][int(np.argmax([conf_home[idx], conf_draw[idx], conf_away[idx]]))]

                # XGB-derived fields (if available)
                if conf_home_xgb is not None:
                    mp_xgb = {"HOME": conf_home_xgb[idx], "DRAW": conf_draw_xgb[idx], "AWAY": conf_away_xgb[idx]}[pick]
                    model_top_xgb = ["HOME","DRAW","AWAY"][int(np.argmax([conf_home_xgb[idx], conf_draw_xgb[idx], conf_away_xgb[idx]]))]
                else:
                    mp_xgb = np.nan
                    model_top_xgb = ""
                rows.append({
                    "league": lg,
                    "match_date": r.get("match_date",""),
                    "home_team_name": r.get("home_team_name",""),
                    "away_team_name": r.get("away_team_name",""),
                    "fixture_key": r.get("fixture_key", ""),
                    "fixture_key_ascii": r.get("fixture_key_ascii", ""),
                    "is_fixture_primary": 1,
                    "pool_tier": "GLUE",
                    "od_source": "bookie_pick",
                    "market": "ftr",
                    "bookie_pick": pick,
                    "selection": pick,
                    "bookie_implied": b["bookie_implied"],
                    "bookie_implied_novig": b.get("bookie_implied_novig", np.nan),
                    "bookie_overround": b.get("bookie_overround", np.nan),
                    "bookie_spread": b.get("bookie_spread", np.nan),
                    # Export all 1X2 odds so slip-builder / EV-anchor logic can use non-favourite outcomes
                    "od_home": float(pd.to_numeric(b.get("od_home", np.nan), errors="coerce")),
                    "od_draw": float(pd.to_numeric(b.get("od_draw", np.nan), errors="coerce")),
                    "od_away": float(pd.to_numeric(b.get("od_away", np.nan), errors="coerce")),
                    "imp_home": (1.0 / float(b.get("od_home"))) if np.isfinite(pd.to_numeric(b.get("od_home", np.nan), errors="coerce")) and float(b.get("od_home")) > 1.0 else np.nan,
                    "imp_draw": (1.0 / float(b.get("od_draw"))) if np.isfinite(pd.to_numeric(b.get("od_draw", np.nan), errors="coerce")) and float(b.get("od_draw")) > 1.0 else np.nan,
                    "imp_away": (1.0 / float(b.get("od_away"))) if np.isfinite(pd.to_numeric(b.get("od_away", np.nan), errors="coerce")) and float(b.get("od_away")) > 1.0 else np.nan,
                    "model_top_pick": model_top,
                    "model_p_for_bookie": float(mp),
                    "agree_model_vs_bookie": int(str(model_top).upper() == str(pick).upper()),
                    "model_top_pick_xgb": model_top_xgb,
                    "ftr_pick_xgb": model_top_xgb,
                    "model_p_for_bookie_xgb": float(mp_xgb) if np.isfinite(mp_xgb) else np.nan,
                    "agree_model_vs_bookie_xgb": int(str(model_top_xgb).upper() == str(pick).upper()) if model_top_xgb else 0,
                    "model_strength": _model_strength_from_bundle(b_ftr),
                    "confidence_home": float(conf_home[dfw.index.get_loc(i)]),
                    "confidence_draw": float(conf_draw[dfw.index.get_loc(i)]),
                    "confidence_away": float(conf_away[dfw.index.get_loc(i)]),
                    "ftr_margin": float(margin[dfw.index.get_loc(i)]),
                    "confidence_home_xgb": float(conf_home_xgb[idx]) if conf_home_xgb is not None else np.nan,
                    "confidence_draw_xgb": float(conf_draw_xgb[idx]) if conf_draw_xgb is not None else np.nan,
                    "confidence_away_xgb": float(conf_away_xgb[idx]) if conf_away_xgb is not None else np.nan,
                    "ftr_p_home_xgb": float(conf_home_xgb[idx]) if conf_home_xgb is not None else np.nan,
                    "ftr_p_draw_xgb": float(conf_draw_xgb[idx]) if conf_draw_xgb is not None else np.nan,
                    "ftr_p_away_xgb": float(conf_away_xgb[idx]) if conf_away_xgb is not None else np.nan,
                    "ftr_margin_xgb": float(margin_xgb[idx]) if margin_xgb is not None else np.nan,
                    "home_power_rating": float(pd.to_numeric(r.get("home_power_rating", np.nan), errors="coerce")),
                    "away_power_rating": float(pd.to_numeric(r.get("away_power_rating", np.nan), errors="coerce")),
                    "power_diff": float(pd.to_numeric(r.get("power_diff", np.nan), errors="coerce")),
                    # Goaliness fields and lambda fit placeholders
                    "average_goals_per_match_pre_match": _num_from_row(r, "average_goals_per_match_pre_match", "Average Goals Per Match (Pre-Match)"),
                    "pre_match_xg_home": _num_from_row(r, "pre_match_xg_home", "Home Team Pre-Match xG", "xg_home"),
                    "pre_match_xg_away": _num_from_row(r, "pre_match_xg_away", "Away Team Pre-Match xG", "xg_away"),
                    "over_25_percentage_pre_match": _norm_pct01(_num_from_row(r, "over_25_percentage_pre_match", "Over 2.5 % (Pre-Match)")),
                    "xg_sum_pre_match": float(
                        np.nansum([
                            _num_from_row(r, "pre_match_xg_home", "Home Team Pre-Match xG", "xg_home"),
                            _num_from_row(r, "pre_match_xg_away", "Away Team Pre-Match xG", "xg_away"),
                        ])
                    ),
                    "ppg_home_pre": ppg_home_pre,
                    "ppg_away_pre": ppg_away_pre,
                    "ppg_diff_pre": ppg_diff_pre,
                    "ppg_pick": ppg_pick,
                    "ppg_opp": ppg_opp,
                    "ftr_ppg_glue_ok": int(bool(glue_ok)),
                    "ftr_drawtrap_flag": int(bool(drawtrap_flag)),
                    **_timing_context_row(r),
                    # Specialist head probabilities
                    "home_ge2_confidence": _num_from_row(r, "home_ge2_confidence"),
                    "away_ge2_confidence": _num_from_row(r, "away_ge2_confidence"),
                    "home_ge3_confidence": _num_from_row(r, "home_ge3_confidence"),
                    "away_ge3_confidence": _num_from_row(r, "away_ge3_confidence"),
                    "p_home_fts": _num_from_row(r, "p_home_fts"),
                    "p_away_fts": _num_from_row(r, "p_away_fts"),
                    "btts_fh_confidence": _num_from_row(r, "btts_fh_confidence"),
                    "btts_fh_pred": _num_from_row(r, "btts_fh_pred"),   
                    # Leak-safe rolling team rates (if available)
                    "scored_rate_5_home": _num_from_row(r, "scored_rate_5_home"),
                    "scored_rate_5_away": _num_from_row(r, "scored_rate_5_away"),
                    "clean_sheet_rate_5_home": _num_from_row(r, "clean_sheet_rate_5_home"),
                    "clean_sheet_rate_5_away": _num_from_row(r, "clean_sheet_rate_5_away"),
                    "conceded_rate_5_home": _num_from_row(r, "conceded_rate_5_home"),
                    "conceded_rate_5_away": _num_from_row(r, "conceded_rate_5_away"),
                    "btts_rate_5_home": _num_from_row(r, "btts_rate_5_home"),
                    "btts_rate_5_away": _num_from_row(r, "btts_rate_5_away"),
                    "over25_rate_5_home": _num_from_row(r, "over25_rate_5_home"),
                    "over25_rate_5_away": _num_from_row(r, "over25_rate_5_away"),
                    "under25_rate_5_home": _num_from_row(r, "under25_rate_5_home"),
                    "under25_rate_5_away": _num_from_row(r, "under25_rate_5_away"),
                    "goaliness_avg_5_home": _num_from_row(r, "goaliness_avg_5_home"),
                    "goaliness_avg_5_away": _num_from_row(r, "goaliness_avg_5_away"),
                    "xg_for_avg_5_home": _num_from_row(r, "xg_for_avg_5_home"),
                    "xg_for_avg_5_away": _num_from_row(r, "xg_for_avg_5_away"),
                    "xg_against_avg_5_home": _num_from_row(r, "xg_against_avg_5_home"),
                    "xg_against_avg_5_away": _num_from_row(r, "xg_against_avg_5_away"),
                    "rolling5_home_gc": _num_from_row(r, "rolling5_home_gc"),
                    "rolling5_away_gc": _num_from_row(r, "rolling5_away_gc"),
                    "gapm_diff": _num_from_row(r, "gapm_diff"),
                    "clean_sheet_rate_diff": _num_from_row(r, "clean_sheet_rate_diff"),
                    "home_xg_against_idx": _num_from_row(r, "home_xg_against_idx"),
                    "away_xg_against_idx": _num_from_row(r, "away_xg_against_idx"),
                    "defence_diff": _num_from_row(r, "defence_diff"),
                    "h2h_n": _num_from_row(r, "h2h_n"),
                    "h2h_btts_rate": _h2h_rate(r, "h2h_btts_rate"),
                    "h2h_over25_rate": _h2h_rate(r, "h2h_over25_rate"),
                    "h2h_goaliness_avg": _h2h_rate(r, "h2h_goaliness_avg"),
                    "bookie_lambda_total_fit": float(pd.to_numeric(b.get("bookie_lambda_total_fit", np.nan), errors="coerce")),
                    "bookie_goaliness_fit_ok": bool(b.get("bookie_goaliness_fit_ok", False)),
                    # canonical odds columns (non-applicable for FTR)
                    "od_over": np.nan,
                    "od_under": np.nan,
                    "od_yes": np.nan,
                    "od_no": np.nan,
                    "odds_ft_over25": np.nan,
                    "odds_ft_under25": np.nan,
                    "odds_btts_yes": np.nan,
                    "odds_btts_no": np.nan,
                })
        def _specialist_common_row(r, lg, *, od_source="model_only") -> dict:
            return {
                "league": lg,
                "match_date": r.get("match_date", ""),
                "home_team_name": r.get("home_team_name", ""),
                "away_team_name": r.get("away_team_name", ""),
                "fixture_key": r.get("fixture_key", ""),
                "fixture_key_ascii": r.get("fixture_key_ascii", ""),
                "is_fixture_primary": 1,
                "pool_tier": "GLUE",
                "od_source": od_source,
                "bookie_od": np.nan,
                "bookie_implied": np.nan,
                "bookie_implied_novig": np.nan,
                "bookie_overround": np.nan,
                "bookie_spread": np.nan,
                "model_strength": np.nan,
                "ftr_margin": np.nan,
                "home_power_rating": float(pd.to_numeric(r.get("home_power_rating", np.nan), errors="coerce")),
                "away_power_rating": float(pd.to_numeric(r.get("away_power_rating", np.nan), errors="coerce")),
                "power_diff": float(pd.to_numeric(r.get("power_diff", np.nan), errors="coerce")),
                "average_goals_per_match_pre_match": _num_from_row(r, "average_goals_per_match_pre_match", "Average Goals Per Match (Pre-Match)"),
                "pre_match_xg_home": _num_from_row(r, "pre_match_xg_home", "Home Team Pre-Match xG", "xg_home"),
                "pre_match_xg_away": _num_from_row(r, "pre_match_xg_away", "Away Team Pre-Match xG", "xg_away"),
                "over_25_percentage_pre_match": _norm_pct01(_num_from_row(r, "over_25_percentage_pre_match", "Over 2.5 % (Pre-Match)")),
                "xg_sum_pre_match": float(np.nansum([
                    _num_from_row(r, "pre_match_xg_home", "Home Team Pre-Match xG", "xg_home"),
                    _num_from_row(r, "pre_match_xg_away", "Away Team Pre-Match xG", "xg_away"),
                ])),
                "ppg_home_pre": _num_from_row(r, "Pre-Match PPG (Home)", "home_ppg", "Pre-Match PPG Home"),
                "ppg_away_pre": _num_from_row(r, "Pre-Match PPG (Away)", "away_ppg", "Pre-Match PPG Away"),
                "ppg_diff_pre": float(
                    _num_from_row(r, "Pre-Match PPG (Home)", "home_ppg", "Pre-Match PPG Home")
                    - _num_from_row(r, "Pre-Match PPG (Away)", "away_ppg", "Pre-Match PPG Away")
                ),
                **_timing_context_row(r),
                "home_ge2_confidence": _num_from_row(r, "home_ge2_confidence"),
                "away_ge2_confidence": _num_from_row(r, "away_ge2_confidence"),
                "home_ge3_confidence": _num_from_row(r, "home_ge3_confidence"),
                "away_ge3_confidence": _num_from_row(r, "away_ge3_confidence"),
                "p_home_fts": _num_from_row(r, "p_home_fts"),
                "p_away_fts": _num_from_row(r, "p_away_fts"),
                "scored_rate_5_home": _num_from_row(r, "scored_rate_5_home"),
                "scored_rate_5_away": _num_from_row(r, "scored_rate_5_away"),
                "clean_sheet_rate_5_home": _num_from_row(r, "clean_sheet_rate_5_home"),
                "clean_sheet_rate_5_away": _num_from_row(r, "clean_sheet_rate_5_away"),
                "conceded_rate_5_home": _num_from_row(r, "conceded_rate_5_home"),
                "conceded_rate_5_away": _num_from_row(r, "conceded_rate_5_away"),
                "btts_rate_5_home": _num_from_row(r, "btts_rate_5_home"),
                "btts_rate_5_away": _num_from_row(r, "btts_rate_5_away"),
                "over25_rate_5_home": _num_from_row(r, "over25_rate_5_home"),
                "over25_rate_5_away": _num_from_row(r, "over25_rate_5_away"),
                "under25_rate_5_home": _num_from_row(r, "under25_rate_5_home"),
                "under25_rate_5_away": _num_from_row(r, "under25_rate_5_away"),
                "goaliness_avg_5_home": _num_from_row(r, "goaliness_avg_5_home"),
                "goaliness_avg_5_away": _num_from_row(r, "goaliness_avg_5_away"),
                "xg_for_avg_5_home": _num_from_row(r, "xg_for_avg_5_home"),
                "xg_for_avg_5_away": _num_from_row(r, "xg_for_avg_5_away"),
                "xg_against_avg_5_home": _num_from_row(r, "xg_against_avg_5_home"),
                "xg_against_avg_5_away": _num_from_row(r, "xg_against_avg_5_away"),
                "rolling5_home_gc": _num_from_row(r, "rolling5_home_gc"),
                "rolling5_away_gc": _num_from_row(r, "rolling5_away_gc"),
                "gapm_diff": _num_from_row(r, "gapm_diff"),
                "clean_sheet_rate_diff": _num_from_row(r, "clean_sheet_rate_diff"),
                "home_xg_against_idx": _num_from_row(r, "home_xg_against_idx"),
                "away_xg_against_idx": _num_from_row(r, "away_xg_against_idx"),
                "defence_diff": _num_from_row(r, "defence_diff"),
                "h2h_n": _num_from_row(r, "h2h_n"),
                "h2h_btts_rate": _h2h_rate(r, "h2h_btts_rate"),
                "h2h_over25_rate": _h2h_rate(r, "h2h_over25_rate"),
                "h2h_goaliness_avg": _h2h_rate(r, "h2h_goaliness_avg"),
                "bookie_lambda_total_fit": np.nan,
                "bookie_goaliness_fit_ok": False,
                "od_yes": np.nan,
                "od_no": np.nan,
                "od_over": np.nan,
                "od_under": np.nan,
                "odds_btts_yes": np.nan,
                "odds_btts_no": np.nan,
                "odds_ft_over25": np.nan,
                "odds_ft_under25": np.nan,
            }
        # --- OU25 ---
        if (b_ou is not None) or (b_u25 is not None):
            try:
                if b_u25 is not None:
                    # Prefer the dedicated UNDER25 head; keep coherence.
                    P_under = _predict_proba(b_u25, X)[:, 1].astype(float)  # prob UNDER25
                    P_under = np.clip(P_under, 0.0, 1.0)
                    P_over = (1.0 - P_under).astype(float)
                else:
                    # Fallback to OVER25 head; derive under as complement.
                    P_over = _predict_proba(b_ou, X)[:, 1].astype(float)  # prob OVER25
                    P_over = np.clip(P_over, 0.0, 1.0)
                    P_under = (1.0 - P_over).astype(float)
            except Exception:
                P_over = None
                P_under = None

            try:
                if b_ou_xgb is not None:
                    P_over_xgb = _predict_proba(b_ou_xgb, X)[:, 1].astype(float)  # prob OVER25
                    P_over_xgb = np.clip(P_over_xgb, 0.0, 1.0)
                    P_under_xgb = (1.0 - P_over_xgb).astype(float)
                else:
                    P_over_xgb = None
                    P_under_xgb = None
            except Exception:
                P_over_xgb = None
                P_under_xgb = None

            ou25_skip_counts: Dict[str, int] = {}
            ou25_skips: List[Dict[str, object]] = []
            ou25_emitted_n = 0

            for i, r in dfw.iterrows():
                b = _bookie_pick_ou25(r, cols, ou25_implied_min)

                ou25_skip_reason = None
                if not b:
                    ou25_skip_reason = "NO_BOOKIE_PICK"
                elif (P_over is None) or (P_under is None):
                    ou25_skip_reason = "NO_MODEL_PROBS"

                if ou25_skip_reason is not None:
                    ou25_skip_counts[ou25_skip_reason] = int(ou25_skip_counts.get(ou25_skip_reason, 0)) + 1
                    try:
                        ou25_skips.append({
                            "league": lg,
                            "match_date": r.get("match_date", ""),
                            "home_team_name": r.get("home_team_name", ""),
                            "away_team_name": r.get("away_team_name", ""),
                            "fixture_key": r.get("fixture_key", ""),
                            "fixture_key_ascii": r.get("fixture_key_ascii", ""),
                            "ou25_skip_reason": ou25_skip_reason,
                        })
                    except Exception:
                        pass

                    if bool(getattr(args, "debug", False)):
                        print(
                            f"[bookie_allmarkets][OU25_SKIP] "
                            f"league={lg} "
                            f"fixture={r.get('home_team_name','')} vs {r.get('away_team_name','')} "
                            f"reason={ou25_skip_reason}"
                        )
                    continue

                p_over = float(P_over[dfw.index.get_loc(i)])
                p_under = float(P_under[dfw.index.get_loc(i)])
                model_top = "OVER25" if p_over >= 0.5 else "UNDER25"
                if P_over_xgb is not None and P_under_xgb is not None:
                    p_over_xgb = float(P_over_xgb[dfw.index.get_loc(i)])
                    p_under_xgb = float(P_under_xgb[dfw.index.get_loc(i)])
                    ou25_pick_xgb = "OVER25" if p_over_xgb >= 0.5 else "UNDER25"
                else:
                    p_over_xgb = np.nan
                    p_under_xgb = np.nan
                    ou25_pick_xgb = ""

                pick = str(b.get("bookie_pick", "OVER25")).upper().strip()
                if pick not in ("OVER25", "UNDER25"):
                    pick = "OVER25"

                # Use the probability for the bookie-selected side
                p_for_bookie = p_over if pick == "OVER25" else p_under
                p_for_bookie_xgb = p_over_xgb if pick == "OVER25" else p_under_xgb

                # Model strength should reflect the head actually used for this row
                bundle_for_strength = None
                if (pick == "UNDER25") and (b_u25 is not None):
                    bundle_for_strength = b_u25
                else:
                    bundle_for_strength = b_ou if b_ou is not None else b_u25
                ou25_priority = _resolve_ou25_priority(model_top, ou25_pick_xgb)

                rows.append({
                    "league": lg,
                    "match_date": r.get("match_date", ""),
                    "home_team_name": r.get("home_team_name", ""),
                    "away_team_name": r.get("away_team_name", ""),
                    "fixture_key": r.get("fixture_key", ""),
                    "fixture_key_ascii": r.get("fixture_key_ascii", ""),
                    "is_fixture_primary": 1,
                    "pool_tier": "GLUE",
                    "od_source": "bookie_pick",
                    "market": "ou25",
                    "bookie_pick": pick,
                    "selection": pick,
                    "bookie_od": b["bookie_od"],
                    "bookie_implied": b["bookie_implied"],
                    "bookie_implied_novig": b.get("bookie_implied_novig", np.nan),
                    "bookie_overround": b.get("bookie_overround", np.nan),
                    "bookie_spread": b.get("bookie_spread", np.nan),
                    "model_top_pick": model_top,
                    "model_top_pick_xgb": ou25_pick_xgb,
                    "model_p_for_bookie": float(p_for_bookie),
                    "ou25_pick_xgb": ou25_pick_xgb,
                    "model_p_for_bookie_xgb_ou25": float(p_for_bookie_xgb) if np.isfinite(p_for_bookie_xgb) else np.nan,
                    "agree_model_vs_bookie_xgb_ou25": int(float(p_for_bookie_xgb) >= 0.50) if np.isfinite(p_for_bookie_xgb) else 0,
                    "ou25_priority": ou25_priority,
                    # Use model_p_for_bookie >= 0.50 for agree_model_vs_bookie (stability improvement)
                    "agree_model_vs_bookie": int(float(p_for_bookie) >= 0.50),
                    "model_strength": _model_strength_from_bundle(bundle_for_strength),
                    "ftr_margin": np.nan,
                    "home_power_rating": float(pd.to_numeric(r.get("home_power_rating", np.nan), errors="coerce")),
                    "away_power_rating": float(pd.to_numeric(r.get("away_power_rating", np.nan), errors="coerce")),
                    "power_diff": float(pd.to_numeric(r.get("power_diff", np.nan), errors="coerce")),
                    # Goaliness fields and lambda fit
                    "average_goals_per_match_pre_match": _num_from_row(r, "average_goals_per_match_pre_match", "Average Goals Per Match (Pre-Match)"),
                    "pre_match_xg_home": _num_from_row(r, "pre_match_xg_home", "Home Team Pre-Match xG", "xg_home"),
                    "pre_match_xg_away": _num_from_row(r, "pre_match_xg_away", "Away Team Pre-Match xG", "xg_away"),
                    "over_25_percentage_pre_match": _norm_pct01(_num_from_row(r, "over_25_percentage_pre_match", "Over 2.5 % (Pre-Match)")),
                    "xg_sum_pre_match": float(
                        np.nansum([
                            _num_from_row(r, "pre_match_xg_home", "Home Team Pre-Match xG", "xg_home"),
                            _num_from_row(r, "pre_match_xg_away", "Away Team Pre-Match xG", "xg_away"),
                        ])
                    ),
                    "ppg_home_pre": _num_from_row(r, "Pre-Match PPG (Home)", "home_ppg", "Pre-Match PPG Home"),
                    "ppg_away_pre": _num_from_row(r, "Pre-Match PPG (Away)", "away_ppg", "Pre-Match PPG Away"),
                    "ppg_diff_pre": float(
                        _num_from_row(r, "Pre-Match PPG (Home)", "home_ppg", "Pre-Match PPG Home")
                        - _num_from_row(r, "Pre-Match PPG (Away)", "away_ppg", "Pre-Match PPG Away")
                    ),
                    **_timing_context_row(r),
                    # Specialist head probabilities
                    "home_ge2_confidence": _num_from_row(r, "home_ge2_confidence"),
                    "away_ge2_confidence": _num_from_row(r, "away_ge2_confidence"),
                    "home_ge3_confidence": _num_from_row(r, "home_ge3_confidence"),
                    "away_ge3_confidence": _num_from_row(r, "away_ge3_confidence"),
                    "p_home_fts": _num_from_row(r, "p_home_fts"),
                    "p_away_fts": _num_from_row(r, "p_away_fts"),
                    # Leak-safe rolling team rates (if available)
                    "scored_rate_5_home": _num_from_row(r, "scored_rate_5_home"),
                    "scored_rate_5_away": _num_from_row(r, "scored_rate_5_away"),
                    "clean_sheet_rate_5_home": _num_from_row(r, "clean_sheet_rate_5_home"),
                    "clean_sheet_rate_5_away": _num_from_row(r, "clean_sheet_rate_5_away"),
                    "conceded_rate_5_home": _num_from_row(r, "conceded_rate_5_home"),
                    "conceded_rate_5_away": _num_from_row(r, "conceded_rate_5_away"),
                    "btts_rate_5_home": _num_from_row(r, "btts_rate_5_home"),
                    "btts_rate_5_away": _num_from_row(r, "btts_rate_5_away"),
                    "over25_rate_5_home": _num_from_row(r, "over25_rate_5_home"),
                    "over25_rate_5_away": _num_from_row(r, "over25_rate_5_away"),
                    "under25_rate_5_home": _num_from_row(r, "under25_rate_5_home"),
                    "under25_rate_5_away": _num_from_row(r, "under25_rate_5_away"),
                    "goaliness_avg_5_home": _num_from_row(r, "goaliness_avg_5_home"),
                    "goaliness_avg_5_away": _num_from_row(r, "goaliness_avg_5_away"),
                    "xg_for_avg_5_home": _num_from_row(r, "xg_for_avg_5_home"),
                    "xg_for_avg_5_away": _num_from_row(r, "xg_for_avg_5_away"),
                    "xg_against_avg_5_home": _num_from_row(r, "xg_against_avg_5_home"),
                    "xg_against_avg_5_away": _num_from_row(r, "xg_against_avg_5_away"),
                    "rolling5_home_gc": _num_from_row(r, "rolling5_home_gc"),
                    "rolling5_away_gc": _num_from_row(r, "rolling5_away_gc"),
                    "gapm_diff": _num_from_row(r, "gapm_diff"),
                    "clean_sheet_rate_diff": _num_from_row(r, "clean_sheet_rate_diff"),
                    "home_xg_against_idx": _num_from_row(r, "home_xg_against_idx"),
                    "away_xg_against_idx": _num_from_row(r, "away_xg_against_idx"),
                    "defence_diff": _num_from_row(r, "defence_diff"),
                    "h2h_n": _num_from_row(r, "h2h_n"),
                    "h2h_btts_rate": _h2h_rate(r, "h2h_btts_rate"),
                    "h2h_over25_rate": _h2h_rate(r, "h2h_over25_rate"),
                    "h2h_goaliness_avg": _h2h_rate(r, "h2h_goaliness_avg"),
                    "bookie_lambda_total_fit": float(pd.to_numeric(b.get("bookie_lambda_total_fit", np.nan), errors="coerce")),
                    "bookie_goaliness_fit_ok": bool(b.get("bookie_goaliness_fit_ok", False)),
                    # canonical OU25 odds (bookie OR synth proxy)
                    "od_over": float(pd.to_numeric(b.get("od_over", np.nan), errors="coerce")),
                    "od_under": float(pd.to_numeric(b.get("od_under", np.nan), errors="coerce")),
                    "odds_ft_over25": float(pd.to_numeric(b.get("odds_ft_over25", b.get("od_over", np.nan)), errors="coerce")),
                    "odds_ft_under25": float(pd.to_numeric(b.get("odds_ft_under25", b.get("od_under", np.nan)), errors="coerce")),

                    # canonical BTTS odds (non-applicable for OU25)
                    "od_yes": np.nan,
                    "od_no": np.nan,
                    "odds_btts_yes": np.nan,
                    "odds_btts_no": np.nan,
                })
                ou25_emitted_n += 1
        # --- GE2 (HOME/AWAY) ---
        if "ge2" in markets:
            for i, r in dfw.iterrows():
                p_home_ge2 = _num_from_row(r, "home_ge2_confidence")
                p_away_ge2 = _num_from_row(r, "away_ge2_confidence")

                if np.isfinite(p_home_ge2):
                    common = _specialist_common_row(r, lg, od_source="model_only")
                    model_top = "HOME"
                    if np.isfinite(p_away_ge2) and (p_away_ge2 > p_home_ge2):
                        model_top = "AWAY"

                    rows.append({
                        **common,
                        "market": "ge2",
                        "bookie_pick": "HOME",
                        "selection": "HOME",
                        "model_top_pick": model_top,
                        "model_p_for_bookie": float(p_home_ge2),
                        "agree_model_vs_bookie": int(model_top == "HOME"),
                    })

                if np.isfinite(p_away_ge2):
                    common = _specialist_common_row(r, lg, od_source="model_only")
                    model_top = "AWAY"
                    if np.isfinite(p_home_ge2) and (p_home_ge2 > p_away_ge2):
                        model_top = "HOME"

                    rows.append({
                        **common,
                        "market": "ge2",
                        "bookie_pick": "AWAY",
                        "selection": "AWAY",
                        "model_top_pick": model_top,
                        "model_p_for_bookie": float(p_away_ge2),
                        "agree_model_vs_bookie": int(model_top == "AWAY"),
                    })

        # --- GE3 (HOME/AWAY) ---
        if "ge3" in markets:
            for i, r in dfw.iterrows():
                p_home_ge3 = _num_from_row(r, "home_ge3_confidence")
                p_away_ge3 = _num_from_row(r, "away_ge3_confidence")

                if np.isfinite(p_home_ge3):
                    common = _specialist_common_row(r, lg, od_source="model_only")
                    model_top = "HOME"
                    if np.isfinite(p_away_ge3) and (p_away_ge3 > p_home_ge3):
                        model_top = "AWAY"

                    rows.append({
                        **common,
                        "market": "ge3",
                        "bookie_pick": "HOME",
                        "selection": "HOME",
                        "model_top_pick": model_top,
                        "model_p_for_bookie": float(p_home_ge3),
                        "agree_model_vs_bookie": int(model_top == "HOME"),
                    })

                if np.isfinite(p_away_ge3):
                    common = _specialist_common_row(r, lg, od_source="model_only")
                    model_top = "AWAY"
                    if np.isfinite(p_home_ge3) and (p_home_ge3 > p_away_ge3):
                        model_top = "HOME"

                    rows.append({
                        **common,
                        "market": "ge3",
                        "bookie_pick": "AWAY",
                        "selection": "AWAY",
                        "model_top_pick": model_top,
                        "model_p_for_bookie": float(p_away_ge3),
                        "agree_model_vs_bookie": int(model_top == "AWAY"),
                    })

        # --- BTTS (YES/NO) ---
        if b_bt:
            try:
                P_yes = _predict_proba(b_bt, X)[:, 1].astype(float)  # prob YES
                P_yes = np.clip(P_yes, 0.0, 1.0)
            except Exception:
                P_yes = None
            try:
                P_yes_xgb = _predict_proba(b_bt_xgb, X)[:, 1].astype(float) if b_bt_xgb else None
                if P_yes_xgb is not None:
                    P_yes_xgb = np.clip(P_yes_xgb, 0.0, 1.0)
            except Exception:
                P_yes_xgb = None
            btts_valueev_od_min = 1.33
            btts_valueev_edge_min = 1.02
            for i, r in dfw.iterrows():
                if P_yes is None:
                    continue

                b = _bookie_pick_btts(r, cols, btts_implied_min)

                pos = int(dfw.index.get_loc(i))

                p_yes = float(P_yes[pos])
                p_no = float(1.0 - p_yes)
                model_top = "YES" if p_yes >= 0.5 else "NO"
                if P_yes_xgb is not None:
                    p_yes_xgb = float(P_yes_xgb[pos])
                    p_no_xgb = float(1.0 - p_yes_xgb)
                    btts_pick_xgb = "YES" if p_yes_xgb >= 0.5 else "NO"
                else:
                    p_yes_xgb = np.nan
                    p_no_xgb = np.nan
                    btts_pick_xgb = ""

                # IMPORTANT:
                # Build YES/NO side rows from raw match-row odds first, not from the
                # chosen-side helper payload. Some helper paths preserve only the chosen
                # side cleanly, which traps BTTS output inside YES rows.
                od_yes = float(pd.to_numeric(
                    r.get(
                        "od_yes",
                        r.get(
                            "odds_btts_yes",
                            r.get(
                                "odds_btts_yes_rm",
                                (b.get("od_yes", b.get("odds_btts_yes", np.nan)) if b else np.nan)
                            )
                        )
                    ),
                    errors="coerce",
                ))
                od_no = float(pd.to_numeric(
                    r.get(
                        "od_no",
                        r.get(
                            "odds_btts_no",
                            r.get(
                                "odds_btts_no_rm",
                                (b.get("od_no", b.get("odds_btts_no", np.nan)) if b else np.nan)
                            )
                        )
                    ),
                    errors="coerce",
                ))

                if (not np.isfinite(od_yes)) or (od_yes <= 1.0):
                    for _cand in (
                        cols.get("btts_yes"),
                        "od_yes",
                        "odds_btts_yes",
                        "odds_btts_yes_rm",
                    ):
                        if _cand and (_cand in r.index):
                            _v = float(pd.to_numeric(r.get(_cand, np.nan), errors="coerce"))
                            if np.isfinite(_v) and (_v > 1.0):
                                od_yes = _v
                                break

                if (not np.isfinite(od_no)) or (od_no <= 1.0):
                    for _cand in (
                        cols.get("btts_no"),
                        "od_no",
                        "odds_btts_no",
                        "odds_btts_no_rm",
                    ):
                        if _cand and (_cand in r.index):
                            _v = float(pd.to_numeric(r.get(_cand, np.nan), errors="coerce"))
                            if np.isfinite(_v) and (_v > 1.0):
                                od_no = _v
                                break

                imp_yes = float((1.0 / od_yes)) if np.isfinite(od_yes) and (od_yes > 1.0) else float("nan")
                imp_no = float((1.0 / od_no)) if np.isfinite(od_no) and (od_no > 1.0) else float("nan")

                overround_bt = float(imp_yes + imp_no) if np.isfinite(imp_yes) and np.isfinite(imp_no) else float("nan")
                imp_yes_nv = float(imp_yes / overround_bt) if np.isfinite(overround_bt) and (overround_bt > 0.0) else float("nan")
                imp_no_nv = float(imp_no / overround_bt) if np.isfinite(overround_bt) and (overround_bt > 0.0) else float("nan")

                side_rows = []
                if np.isfinite(od_yes) and (od_yes > 1.0):
                    side_rows.append({
                        "pick": "YES",
                        "bookie_od": od_yes,
                        "bookie_implied": imp_yes,
                        "bookie_implied_novig": imp_yes_nv,
                        "p_pick": p_yes,
                    })

                if np.isfinite(od_no) and (od_no > 1.0):
                    side_rows.append({
                        "pick": "NO",
                        "bookie_od": od_no,
                        "bookie_implied": imp_no,
                        "bookie_implied_novig": imp_no_nv,
                        "p_pick": p_no,
                    })
                # Last-resort fallback: if raw YES/NO odds still failed to materialise,
                # keep the legacy chosen-side row so the market is not lost completely.
                if (not side_rows) and (b is not None):
                    legacy_pick = str(b.get("bookie_pick", "YES")).upper().strip()
                    if legacy_pick not in ("YES", "NO"):
                        legacy_pick = "YES"
                    legacy_p_pick = p_yes if legacy_pick == "YES" else p_no
                    legacy_imp_pick = float(pd.to_numeric(b.get("bookie_implied", np.nan), errors="coerce"))
                    legacy_imp_pick_nv = float(pd.to_numeric(b.get("bookie_implied_novig", np.nan), errors="coerce"))
                    legacy_bookie_od_pick = float(pd.to_numeric(b.get("bookie_od", np.nan), errors="coerce"))
                    side_rows.append({
                        "pick": legacy_pick,
                        "bookie_od": legacy_bookie_od_pick,
                        "bookie_implied": legacy_imp_pick,
                        "bookie_implied_novig": legacy_imp_pick_nv,
                        "p_pick": float(legacy_p_pick),
                    })

                if not side_rows:
                    continue
                
                for side_info in side_rows:
                    pick = str(side_info["pick"]).upper().strip()
                    p_pick = float(side_info["p_pick"])
                    bookie_od_pick = float(side_info["bookie_od"])
                    imp_pick = float(side_info["bookie_implied"])
                    imp_pick_nv = float(side_info["bookie_implied_novig"])

                    edge_raw = float(p_pick - imp_pick) if pd.notna(p_pick) and pd.notna(imp_pick) else np.nan
                    edge_novig = float(p_pick - imp_pick_nv) if pd.notna(p_pick) and pd.notna(imp_pick_nv) else np.nan
                    edge_value = edge_novig if pd.notna(edge_novig) else edge_raw

                    valueev_ok = bool(
                        pd.notna(bookie_od_pick)
                        and (bookie_od_pick >= float(btts_valueev_od_min))
                        and pd.notna(edge_value)
                        and (edge_value >= float(btts_valueev_edge_min) / 100.0)
                    )
                    p_pick_xgb = p_yes_xgb if pick == "YES" else p_no_xgb
                    btts_priority = _resolve_btts_priority(lg, model_top, btts_pick_xgb)

                    rows.append({
                        "league": lg,
                        "match_date": r.get("match_date",""),
                        "home_team_name": r.get("home_team_name",""),
                        "away_team_name": r.get("away_team_name",""),
                        "fixture_key": r.get("fixture_key", ""),
                        "fixture_key_ascii": r.get("fixture_key_ascii", ""),
                        "is_fixture_primary": 1,
                        "pool_tier": "GLUE",
                        "od_source": "bookie_pick",
                        "market": "btts",
                        "bookie_pick": pick,
                        "selection": pick,
                        "product": "BTTS_MODEL",
                        "model_lane": "btts_model",
                        "source_prob_col": "p_pick",
                        "signal_btts_fixture": pd.NA,
                        "signal_btts_side": _btts_side_signal_from_pick_prob(pick, p_pick),
                        "bookie_od": float(bookie_od_pick),
                        "bookie_implied": float(imp_pick),
                        "bookie_implied_novig": float(imp_pick_nv) if pd.notna(imp_pick_nv) else np.nan,
                        "bookie_overround": float(overround_bt) if pd.notna(overround_bt) else np.nan,
                        "bookie_spread": np.nan,
                        "model_top_pick": model_top,
                        "model_top_pick_xgb": btts_pick_xgb,
                        "model_p_for_bookie": float(p_pick),
                        "agree_model_vs_bookie": int(float(p_pick) >= 0.50),
                        "btts_pick_xgb": btts_pick_xgb,
                        "model_p_for_bookie_xgb_btts": float(p_pick_xgb) if np.isfinite(p_pick_xgb) else np.nan,
                        "agree_model_vs_bookie_xgb_btts": int(float(p_pick_xgb) >= 0.50) if np.isfinite(p_pick_xgb) else 0,
                        "btts_priority": btts_priority,
                        "model_strength": _model_strength_from_bundle(b_bt),
                        "ftr_margin": np.nan,
                        "home_power_rating": float(pd.to_numeric(r.get("home_power_rating", np.nan), errors="coerce")),
                        "away_power_rating": float(pd.to_numeric(r.get("away_power_rating", np.nan), errors="coerce")),
                        "power_diff": float(pd.to_numeric(r.get("power_diff", np.nan), errors="coerce")),
                        "average_goals_per_match_pre_match": _num_from_row(r, "average_goals_per_match_pre_match", "Average Goals Per Match (Pre-Match)"),
                        "pre_match_ppg_home": _resolve_pre_match_ppg_home(r),
                        "pre_match_ppg_away": _resolve_pre_match_ppg_away(r),
                        "pre_match_xg_home": _resolve_pre_match_xg_home(r),
                        "pre_match_xg_away": _resolve_pre_match_xg_away(r),
                        "btts_percentage_pre_match": _resolve_btts_pct_pre(r),
                        "over_25_percentage_pre_match": _norm_pct01(_num_from_row(r, "over_25_percentage_pre_match", "Over 2.5 % (Pre-Match)")),
                        "xg_sum_pre_match": float(
                            np.nansum([
                                _resolve_pre_match_xg_home(r),
                                _resolve_pre_match_xg_away(r),
                            ])
                        ),
                        "ppg_home_pre": _resolve_pre_match_ppg_home(r),
                        "ppg_away_pre": _resolve_pre_match_ppg_away(r),
                        "ppg_diff_pre": float(
                            _resolve_pre_match_ppg_home(r)
                            - _resolve_pre_match_ppg_away(r)
                        ),
                        **_timing_context_row(r),
                        "home_ge2_confidence": _num_from_row(r, "home_ge2_confidence"),
                        "away_ge2_confidence": _num_from_row(r, "away_ge2_confidence"),
                        "home_ge3_confidence": _num_from_row(r, "home_ge3_confidence"),
                        "away_ge3_confidence": _num_from_row(r, "away_ge3_confidence"),
                        "p_home_fts": _num_from_row(r, "p_home_fts"),
                        "p_away_fts": _num_from_row(r, "p_away_fts"),
                        "scored_rate_5_home": _num_from_row(r, "scored_rate_5_home"),
                        "scored_rate_5_away": _num_from_row(r, "scored_rate_5_away"),
                        "clean_sheet_rate_5_home": _num_from_row(r, "clean_sheet_rate_5_home"),
                        "clean_sheet_rate_5_away": _num_from_row(r, "clean_sheet_rate_5_away"),
                        "conceded_rate_5_home": _num_from_row(r, "conceded_rate_5_home"),
                        "conceded_rate_5_away": _num_from_row(r, "conceded_rate_5_away"),
                        "btts_rate_5_home": _num_from_row(r, "btts_rate_5_home"),
                        "btts_rate_5_away": _num_from_row(r, "btts_rate_5_away"),
                        "recent_btts_regime_blend_l5": _num_from_row(r, "recent_btts_regime_blend_l5"),
                        "recent_btts_regime_blend_l10": _num_from_row(r, "recent_btts_regime_blend_l10"),
                        "recent_no_btts_regime_blend_l5": _num_from_row(r, "recent_no_btts_regime_blend_l5"),
                        "recent_no_btts_regime_blend_l10": _num_from_row(r, "recent_no_btts_regime_blend_l10"),
                        "over25_rate_5_home": _num_from_row(r, "over25_rate_5_home"),
                        "over25_rate_5_away": _num_from_row(r, "over25_rate_5_away"),
                        "under25_rate_5_home": _num_from_row(r, "under25_rate_5_home"),
                        "under25_rate_5_away": _num_from_row(r, "under25_rate_5_away"),
                        "goaliness_avg_5_home": _num_from_row(r, "goaliness_avg_5_home"),
                        "goaliness_avg_5_away": _num_from_row(r, "goaliness_avg_5_away"),
                        "xg_for_avg_5_home": _num_from_row(r, "xg_for_avg_5_home"),
                        "xg_for_avg_5_away": _num_from_row(r, "xg_for_avg_5_away"),
                        "xg_against_avg_5_home": _num_from_row(r, "xg_against_avg_5_home"),
                        "xg_against_avg_5_away": _num_from_row(r, "xg_against_avg_5_away"),
                        "rolling5_home_gc": _num_from_row(r, "rolling5_home_gc"),
                        "rolling5_away_gc": _num_from_row(r, "rolling5_away_gc"),
                        "gapm_diff": _num_from_row(r, "gapm_diff"),
                        "clean_sheet_rate_diff": _num_from_row(r, "clean_sheet_rate_diff"),
                        "home_xg_against_idx": _num_from_row(r, "home_xg_against_idx"),
                        "away_xg_against_idx": _num_from_row(r, "away_xg_against_idx"),
                        "defence_diff": _num_from_row(r, "defence_diff"),
                        "h2h_n": _num_from_row(r, "h2h_n"),
                        "h2h_btts_rate": _h2h_rate(r, "h2h_btts_rate"),
                        "h2h_over25_rate": _h2h_rate(r, "h2h_over25_rate"),
                        "h2h_goaliness_avg": _h2h_rate(r, "h2h_goaliness_avg"),
                        "bookie_lambda_total_fit": float(pd.to_numeric(b.get("bookie_lambda_total_fit", np.nan), errors="coerce")) if b is not None else np.nan,
                        "bookie_goaliness_fit_ok": bool(b.get("bookie_goaliness_fit_ok", False)) if b is not None else False,
                        "od_yes": float(od_yes) if np.isfinite(od_yes) else np.nan,
                        "od_no": float(od_no) if np.isfinite(od_no) else np.nan,
                        "odds_btts_yes": float(od_yes) if np.isfinite(od_yes) else np.nan,
                        "odds_btts_no": float(od_no) if np.isfinite(od_no) else np.nan,
                        "od_over": np.nan,
                        "od_under": np.nan,
                        "odds_ft_over25": np.nan,
                        "odds_ft_under25": np.nan,
                        "p_pick": float(p_pick),
                        "edge": float(edge_value) if pd.notna(edge_value) else np.nan,
                    })

                    if valueev_ok:
                        rows.append({
                            "league": lg,
                            "match_date": r.get("match_date",""),
                            "home_team_name": r.get("home_team_name",""),
                            "away_team_name": r.get("away_team_name",""),
                            "fixture_key": r.get("fixture_key", ""),
                            "fixture_key_ascii": r.get("fixture_key_ascii", ""),
                            "is_fixture_primary": 0,
                            "pool_tier": "GLUE",
                            "od_source": "bookie_pick",
                            "market": "btts",
                            "bookie_pick": pick,
                            "selection": pick,
                            "product": "BTTS_VALUEEV",
                            "model_lane": "btts_valueev",
                            "source_prob_col": "p_pick",
                            "signal_btts_fixture": pd.NA,
                            "signal_btts_side": _btts_side_signal_from_pick_prob(pick, p_pick),
                            "bookie_od": float(bookie_od_pick),
                            "bookie_implied": float(imp_pick),
                            "bookie_implied_novig": float(imp_pick_nv) if pd.notna(imp_pick_nv) else np.nan,
                            "bookie_overround": float(overround_bt) if pd.notna(overround_bt) else np.nan,
                            "bookie_spread": np.nan,
                            "model_top_pick": model_top,
                            "model_p_for_bookie": float(p_pick),
                            "agree_model_vs_bookie": int(float(p_pick) >= 0.50),
                            "model_strength": _model_strength_from_bundle(b_bt),
                            "ftr_margin": np.nan,
                            "home_power_rating": float(pd.to_numeric(r.get("home_power_rating", np.nan), errors="coerce")),
                            "away_power_rating": float(pd.to_numeric(r.get("away_power_rating", np.nan), errors="coerce")),
                            "power_diff": float(pd.to_numeric(r.get("power_diff", np.nan), errors="coerce")),
                            "average_goals_per_match_pre_match": _num_from_row(r, "average_goals_per_match_pre_match", "Average Goals Per Match (Pre-Match)"),
                            "pre_match_ppg_home": _resolve_pre_match_ppg_home(r),
                            "pre_match_ppg_away": _resolve_pre_match_ppg_away(r),
                            "pre_match_xg_home": _resolve_pre_match_xg_home(r),
                            "pre_match_xg_away": _resolve_pre_match_xg_away(r),
                            "btts_percentage_pre_match": _resolve_btts_pct_pre(r),
                            "over_25_percentage_pre_match": _norm_pct01(_num_from_row(r, "over_25_percentage_pre_match", "Over 2.5 % (Pre-Match)")),
                            "xg_sum_pre_match": float(
                                np.nansum([
                                    _resolve_pre_match_xg_home(r),
                                    _resolve_pre_match_xg_away(r),
                                ])
                            ),
                            "ppg_home_pre": _resolve_pre_match_ppg_home(r),
                            "ppg_away_pre": _resolve_pre_match_ppg_away(r),
                            "ppg_diff_pre": float(
                                _resolve_pre_match_ppg_home(r)
                                - _resolve_pre_match_ppg_away(r)
                            ),
                            **_timing_context_row(r),
                            "home_ge2_confidence": _num_from_row(r, "home_ge2_confidence"),
                            "away_ge2_confidence": _num_from_row(r, "away_ge2_confidence"),
                            "home_ge3_confidence": _num_from_row(r, "home_ge3_confidence"),
                            "away_ge3_confidence": _num_from_row(r, "away_ge3_confidence"),
                            "p_home_fts": _num_from_row(r, "p_home_fts"),
                            "p_away_fts": _num_from_row(r, "p_away_fts"),
                            "scored_rate_5_home": _num_from_row(r, "scored_rate_5_home"),
                            "scored_rate_5_away": _num_from_row(r, "scored_rate_5_away"),
                            "clean_sheet_rate_5_home": _num_from_row(r, "clean_sheet_rate_5_home"),
                            "clean_sheet_rate_5_away": _num_from_row(r, "clean_sheet_rate_5_away"),
                            "conceded_rate_5_home": _num_from_row(r, "conceded_rate_5_home"),
                            "conceded_rate_5_away": _num_from_row(r, "conceded_rate_5_away"),
                            "btts_rate_5_home": _num_from_row(r, "btts_rate_5_home"),
                            "btts_rate_5_away": _num_from_row(r, "btts_rate_5_away"),
                            "recent_btts_regime_blend_l5": _num_from_row(r, "recent_btts_regime_blend_l5"),
                            "recent_btts_regime_blend_l10": _num_from_row(r, "recent_btts_regime_blend_l10"),
                            "recent_no_btts_regime_blend_l5": _num_from_row(r, "recent_no_btts_regime_blend_l5"),
                            "recent_no_btts_regime_blend_l10": _num_from_row(r, "recent_no_btts_regime_blend_l10"),
                            "over25_rate_5_home": _num_from_row(r, "over25_rate_5_home"),
                            "over25_rate_5_away": _num_from_row(r, "over25_rate_5_away"),
                            "under25_rate_5_home": _num_from_row(r, "under25_rate_5_home"),
                            "under25_rate_5_away": _num_from_row(r, "under25_rate_5_away"),
                            "goaliness_avg_5_home": _num_from_row(r, "goaliness_avg_5_home"),
                            "goaliness_avg_5_away": _num_from_row(r, "goaliness_avg_5_away"),
                            "xg_for_avg_5_home": _num_from_row(r, "xg_for_avg_5_home"),
                            "xg_for_avg_5_away": _num_from_row(r, "xg_for_avg_5_away"),
                            "xg_against_avg_5_home": _num_from_row(r, "xg_against_avg_5_home"),
                            "xg_against_avg_5_away": _num_from_row(r, "xg_against_avg_5_away"),
                            "rolling5_home_gc": _num_from_row(r, "rolling5_home_gc"),
                            "rolling5_away_gc": _num_from_row(r, "rolling5_away_gc"),
                            "gapm_diff": _num_from_row(r, "gapm_diff"),
                            "clean_sheet_rate_diff": _num_from_row(r, "clean_sheet_rate_diff"),
                            "home_xg_against_idx": _num_from_row(r, "home_xg_against_idx"),
                            "away_xg_against_idx": _num_from_row(r, "away_xg_against_idx"),
                            "defence_diff": _num_from_row(r, "defence_diff"),
                            "h2h_n": _num_from_row(r, "h2h_n"),
                            "h2h_btts_rate": _h2h_rate(r, "h2h_btts_rate"),
                            "h2h_over25_rate": _h2h_rate(r, "h2h_over25_rate"),
                            "h2h_goaliness_avg": _h2h_rate(r, "h2h_goaliness_avg"),
                            "bookie_lambda_total_fit": float(pd.to_numeric(b.get("bookie_lambda_total_fit", np.nan), errors="coerce")) if b is not None else np.nan,
                            "bookie_goaliness_fit_ok": bool(b.get("bookie_goaliness_fit_ok", False)) if b is not None else False,
                            "od_yes": float(od_yes) if np.isfinite(od_yes) else np.nan,
                            "od_no": float(od_no) if np.isfinite(od_no) else np.nan,
                            "odds_btts_yes": float(od_yes) if np.isfinite(od_yes) else np.nan,
                            "odds_btts_no": float(od_no) if np.isfinite(od_no) else np.nan,
                            "od_over": np.nan,
                            "od_under": np.nan,
                            "odds_ft_over25": np.nan,
                            "odds_ft_under25": np.nan,
                            "p_pick": float(p_pick),
                            "edge": float(edge_value) if pd.notna(edge_value) else np.nan,
                        })

        # --- FTS (HOME/AWAY) ---
        if "fts" in markets:
            for i, r in dfw.iterrows():
                p_home_fts = _num_from_row(r, "p_home_fts")
                p_away_fts = _num_from_row(r, "p_away_fts")

                if np.isfinite(p_home_fts):
                    common = _specialist_common_row(r, lg, od_source="model_only")
                    model_top = "HOME"
                    if np.isfinite(p_away_fts) and (p_away_fts > p_home_fts):
                        model_top = "AWAY"

                    rows.append({
                        **common,
                        "market": "fts",
                        "bookie_pick": "HOME",
                        "selection": "HOME",
                        "model_top_pick": model_top,
                        "model_p_for_bookie": float(p_home_fts),
                        "agree_model_vs_bookie": int(model_top == "HOME"),
                    })

                if np.isfinite(p_away_fts):
                    common = _specialist_common_row(r, lg, od_source="model_only")
                    model_top = "AWAY"
                    if np.isfinite(p_home_fts) and (p_home_fts > p_away_fts):
                        model_top = "HOME"

                    rows.append({
                        **common,
                        "market": "fts",
                        "bookie_pick": "AWAY",
                        "selection": "AWAY",
                        "model_top_pick": model_top,
                        "model_p_for_bookie": float(p_away_fts),
                        "agree_model_vs_bookie": int(model_top == "AWAY"),
                    })
                    
        # --- WTN (HOME/AWAY proxy) ---
        # Honest first-pass proxy emitter only:
        #   - not a priced WTN market
        #   - not true prob_wtn_home / prob_wtn_away heads yet
        #   - derived from win tendency + opponent fail-to-score tendency
        if "wtn" in markets:
            for i, r in dfw.iterrows():
                p_home = _num_from_row(r, "confidence_home", "p_home_pois_norm", "p_home_pois")
                p_away = _num_from_row(r, "confidence_away", "p_away_pois_norm", "p_away_pois")
                p_home_fts = _num_from_row(r, "p_home_fts")
                p_away_fts = _num_from_row(r, "p_away_fts")

                p_home_wtn = np.nan
                p_away_wtn = np.nan

                if np.isfinite(p_home) and np.isfinite(p_away_fts):
                    p_home_wtn = float(np.sqrt(float(p_home) * float(p_away_fts)))
                if np.isfinite(p_away) and np.isfinite(p_home_fts):
                    p_away_wtn = float(np.sqrt(float(p_away) * float(p_home_fts)))

                if np.isfinite(p_home_wtn):
                    common = _specialist_common_row(r, lg, od_source="model_only")
                    model_top = "HOME"
                    if np.isfinite(p_away_wtn) and (p_away_wtn > p_home_wtn):
                        model_top = "AWAY"

                    rows.append({
                        **common,
                        "market": "wtn",
                        "bookie_pick": "HOME",
                        "selection": "HOME",
                        "model_top_pick": model_top,
                        "model_p_for_bookie": float(p_home_wtn),
                        "agree_model_vs_bookie": int(model_top == "HOME"),
                    })

                if np.isfinite(p_away_wtn):
                    common = _specialist_common_row(r, lg, od_source="model_only")
                    model_top = "AWAY"
                    if np.isfinite(p_home_wtn) and (p_home_wtn > p_away_wtn):
                        model_top = "HOME"

                    rows.append({
                        **common,
                        "market": "wtn",
                        "bookie_pick": "AWAY",
                        "selection": "AWAY",
                        "model_top_pick": model_top,
                        "model_p_for_bookie": float(p_away_wtn),
                        "agree_model_vs_bookie": int(model_top == "AWAY"),
                    })
        if bool(getattr(args, "debug", False)):
            try:
                if "ou25" in markets:
                    total_skipped = int(sum(ou25_skip_counts.values())) if "ou25_skip_counts" in locals() else 0
                    print(
                        f"[bookie_allmarkets][OU25_SUMMARY] league={lg} emitted={int(ou25_emitted_n) if 'ou25_emitted_n' in locals() else 0} "
                        f"skipped={total_skipped} skip_counts={ou25_skip_counts if 'ou25_skip_counts' in locals() else {}}"
                    )
            except Exception:
                pass
        # ---------------------------
        # Per-league diagnostic (debug)
        # ---------------------------
        if bool(getattr(args, "debug", False)):
            try:
                df_dbg = pd.DataFrame(rows[_rows_start:])
                if isinstance(df_dbg, pd.DataFrame) and not df_dbg.empty:
                    # Basic counts
                    n_rows = int(len(df_dbg))
                    mkt = df_dbg.get("market", pd.Series([], dtype="object")).astype(str).str.lower().str.strip()
                    try:
                        mkt_counts = mkt.value_counts(dropna=False).to_dict()
                    except Exception:
                        mkt_counts = {}

                    # Existing numeric summaries (keep as lightweight health signals)
                    gap = pd.to_numeric(df_dbg.get("gap_novig", np.nan), errors="coerce")
                    mp = pd.to_numeric(df_dbg.get("model_p_for_bookie", np.nan), errors="coerce")
                    bod = pd.to_numeric(df_dbg.get("bookie_od", np.nan), errors="coerce")

                    def _mean_or_nan(s: pd.Series) -> float:
                        try:
                            return float(s.mean()) if s.notna().any() else float("nan")
                        except Exception:
                            return float("nan")

                    # Compact market summary string e.g. ftr=12, ou25=8, btts=6
                    try:
                        mkt_parts = [f"{k}={int(v)}" for k, v in mkt_counts.items()]
                        mkt_summary = ", ".join(mkt_parts)
                    except Exception:
                        mkt_summary = ""

                    print(
                        f"[LEAGUE_DIAG] {lg}: rows_total={n_rows} "
                        f"markets=({mkt_summary}) "
                        f"gap_novig_mean={_mean_or_nan(gap):.4f} "
                        f"model_p_mean={_mean_or_nan(mp):.4f} "
                        f"bookie_od_mean={_mean_or_nan(bod):.4f}"
                    )
            except Exception as _e_diag:
                print(f"[LEAGUE_DIAG] {lg}: skipped ({_e_diag})")

        # --- Team Goals markets (TG15/TG25) ---
        # These are model-only markets (no bookmaker team-goals odds in fd_odds_enriched.csv).
        # We output them for acca-building and accuracy tracking; ROI requires real odds later.
        if ("tg15" in markets) or ("tg25" in markets):
            tg15_pmin = float(getattr(args, "tg15_pmin", 0.65))
            tg25_pmin = float(getattr(args, "tg25_pmin", 0.45))

            for i, r in dfw.iterrows():
                # Pull specialist head probabilities (may be NaN if head not attached)
                p_h_ge2 = _num_from_row(r, "home_ge2_confidence")
                p_a_ge2 = _num_from_row(r, "away_ge2_confidence")
                p_h_ge3 = _num_from_row(r, "home_ge3_confidence")
                p_a_ge3 = _num_from_row(r, "away_ge3_confidence")
                # Poisson coherence tails (from λ). Used to veto nonsensical TG picks.
                pois_h_ge2 = _num_from_row(r, "pois_home_ge2")
                pois_a_ge2 = _num_from_row(r, "pois_away_ge2")
                pois_h_ge3 = _num_from_row(r, "pois_home_ge3")
                pois_a_ge3 = _num_from_row(r, "pois_away_ge3")

                tg_pois_ge2_min = float(getattr(args, "tg_pois_ge2_min", 0.12))
                tg_pois_ge3_min = float(getattr(args, "tg_pois_ge3_min", 0.08))

                tg_pois_gap_max_ge2 = float(getattr(args, "tg_pois_gap_max_ge2", 0.50))
                tg_pois_gap_max_ge3 = float(getattr(args, "tg_pois_gap_max_ge3", 0.50))

                # Common fields shared across TG rows
                common = {
                    "league": lg,
                    "match_date": r.get("match_date", ""),
                    "home_team_name": r.get("home_team_name", ""),
                    "away_team_name": r.get("away_team_name", ""),
                    "fixture_key": r.get("fixture_key", ""),
                    "fixture_key_ascii": r.get("fixture_key_ascii", ""),
                    "is_fixture_primary": 1,
                    "pool_tier": "GLUE",
                    "od_source": "model_only",
                    "bookie_od": np.nan,
                    "bookie_implied": np.nan,
                    "bookie_implied_novig": np.nan,
                    "bookie_overround": np.nan,
                    "bookie_spread": np.nan,
                    "model_strength": np.nan,
                    "ftr_margin": np.nan,
                    "home_power_rating": float(pd.to_numeric(r.get("home_power_rating", np.nan), errors="coerce")),
                    "away_power_rating": float(pd.to_numeric(r.get("away_power_rating", np.nan), errors="coerce")),
                    "power_diff": float(pd.to_numeric(r.get("power_diff", np.nan), errors="coerce")),
                    "average_goals_per_match_pre_match": _num_from_row(r, "average_goals_per_match_pre_match", "Average Goals Per Match (Pre-Match)"),
                    "pre_match_xg_home": _num_from_row(r, "pre_match_xg_home", "Home Team Pre-Match xG", "xg_home"),
                    "pre_match_xg_away": _num_from_row(r, "pre_match_xg_away", "Away Team Pre-Match xG", "xg_away"),
                    "over_25_percentage_pre_match": _norm_pct01(_num_from_row(r, "over_25_percentage_pre_match", "Over 2.5 % (Pre-Match)")),
                    "xg_sum_pre_match": float(
                        np.nansum([
                            _num_from_row(r, "pre_match_xg_home", "Home Team Pre-Match xG", "xg_home"),
                            _num_from_row(r, "pre_match_xg_away", "Away Team Pre-Match xG", "xg_away"),
                        ])
                    ),
                    "ppg_home_pre": _num_from_row(r, "Pre-Match PPG (Home)", "home_ppg", "Pre-Match PPG Home"),
                    "ppg_away_pre": _num_from_row(r, "Pre-Match PPG (Away)", "away_ppg", "Pre-Match PPG Away"),
                    "ppg_diff_pre": float(
                        _num_from_row(r, "Pre-Match PPG (Home)", "home_ppg", "Pre-Match PPG Home")
                        - _num_from_row(r, "Pre-Match PPG (Away)", "away_ppg", "Pre-Match PPG Away")
                    ),
                    **_timing_context_row(r),
                    # Specialist head probabilities
                    "home_ge2_confidence": p_h_ge2,
                    "away_ge2_confidence": p_a_ge2,
                    "home_ge3_confidence": p_h_ge3,
                    "away_ge3_confidence": p_a_ge3,
                    "p_home_fts": _num_from_row(r, "p_home_fts"),
                    "p_away_fts": _num_from_row(r, "p_away_fts"),
                    # Rolling rates (if present)
                    "scored_rate_5_home": _num_from_row(r, "scored_rate_5_home"),
                    "scored_rate_5_away": _num_from_row(r, "scored_rate_5_away"),
                    "clean_sheet_rate_5_home": _num_from_row(r, "clean_sheet_rate_5_home"),
                    "clean_sheet_rate_5_away": _num_from_row(r, "clean_sheet_rate_5_away"),
                    "conceded_rate_5_home": _num_from_row(r, "conceded_rate_5_home"),
                    "conceded_rate_5_away": _num_from_row(r, "conceded_rate_5_away"),
                    "btts_rate_5_home": _num_from_row(r, "btts_rate_5_home"),
                    "btts_rate_5_away": _num_from_row(r, "btts_rate_5_away"),
                    "over25_rate_5_home": _num_from_row(r, "over25_rate_5_home"),
                    "over25_rate_5_away": _num_from_row(r, "over25_rate_5_away"),
                    "under25_rate_5_home": _num_from_row(r, "under25_rate_5_home"),
                    "under25_rate_5_away": _num_from_row(r, "under25_rate_5_away"),
                    "goaliness_avg_5_home": _num_from_row(r, "goaliness_avg_5_home"),
                    "goaliness_avg_5_away": _num_from_row(r, "goaliness_avg_5_away"),
                    "xg_for_avg_5_home": _num_from_row(r, "xg_for_avg_5_home"),
                    "xg_for_avg_5_away": _num_from_row(r, "xg_for_avg_5_away"),
                    "xg_against_avg_5_home": _num_from_row(r, "xg_against_avg_5_home"),
                    "xg_against_avg_5_away": _num_from_row(r, "xg_against_avg_5_away"),
                    "rolling5_home_gc": _num_from_row(r, "rolling5_home_gc"),
                    "rolling5_away_gc": _num_from_row(r, "rolling5_away_gc"),
                    "gapm_diff": _num_from_row(r, "gapm_diff"),
                    "clean_sheet_rate_diff": _num_from_row(r, "clean_sheet_rate_diff"),
                    "home_xg_against_idx": _num_from_row(r, "home_xg_against_idx"),
                    "away_xg_against_idx": _num_from_row(r, "away_xg_against_idx"),
                    "defence_diff": _num_from_row(r, "defence_diff"),
                    "h2h_n": _num_from_row(r, "h2h_n"),
                    "h2h_btts_rate": _h2h_rate(r, "h2h_btts_rate"),
                    "h2h_over25_rate": _h2h_rate(r, "h2h_over25_rate"),
                    "h2h_goaliness_avg": _h2h_rate(r, "h2h_goaliness_avg"),
                    "bookie_lambda_total_fit": np.nan,
                    "bookie_goaliness_fit_ok": False,
                }
                # --- TG directional gate helper ---
                def _tg_dir_ok(pick_tg: str) -> bool:
                    if not bool(getattr(args, "tg_use_dir_gate", False)):
                        return True

                    pdiff = float(pd.to_numeric(common.get("ppg_diff_pre", np.nan), errors="coerce"))
                    p_home = float(pd.to_numeric(common.get("ppg_home_pre", np.nan), errors="coerce"))
                    p_away = float(pd.to_numeric(common.get("ppg_away_pre", np.nan), errors="coerce"))
                    powdiff = float(pd.to_numeric(common.get("power_diff", np.nan), errors="coerce"))

                    pick_tg = str(pick_tg).upper().strip()
                    opp_ppg = np.nan

                    if pick_tg.startswith("HOME"):
                        opp_ppg = p_away
                        if np.isfinite(pdiff) and pdiff < float(getattr(args, "tg_ppg_home_min", 0.35)):
                            return False
                        if np.isfinite(powdiff) and powdiff < float(getattr(args, "tg_pd_home_min", 5.0)):
                            return False
                    elif pick_tg.startswith("AWAY"):
                        opp_ppg = p_home
                        if np.isfinite(pdiff) and pdiff > float(getattr(args, "tg_ppg_away_max", -0.35)):
                            return False
                        if np.isfinite(powdiff) and powdiff > float(getattr(args, "tg_pd_away_max", -5.0)):
                            return False

                    # optional opponent cap
                    if np.isfinite(opp_ppg) and opp_ppg > float(getattr(args, "tg_opp_ppg_max", 1.20)):
                        return False

                    return True
                # --- TG15 ---
                if "tg15" in markets:
                    cand = []

                    # HOME_TG15 uses GE2
                    if np.isfinite(p_h_ge2) and (p_h_ge2 >= tg15_pmin) and _tg_dir_ok("HOME_TG15"):
                        ok_pois = (np.isfinite(pois_h_ge2) and (pois_h_ge2 >= tg_pois_ge2_min))
                        gap_pois = (float(p_h_ge2) - float(pois_h_ge2)) if np.isfinite(pois_h_ge2) else np.nan

                        # mismatch veto: model says huge, Poisson says tiny
                        if ok_pois and np.isfinite(gap_pois) and (gap_pois > tg_pois_gap_max_ge2):
                            ok_pois = False

                        if ok_pois:
                            cand.append(("HOME_TG15", float(p_h_ge2), True, float(gap_pois) if np.isfinite(gap_pois) else np.nan))

                    # AWAY_TG15 uses GE2
                    if np.isfinite(p_a_ge2) and (p_a_ge2 >= tg15_pmin) and _tg_dir_ok("AWAY_TG15"):
                        ok_pois = (np.isfinite(pois_a_ge2) and (pois_a_ge2 >= tg_pois_ge2_min))
                        gap_pois = (float(p_a_ge2) - float(pois_a_ge2)) if np.isfinite(pois_a_ge2) else np.nan

                        if ok_pois and np.isfinite(gap_pois) and (gap_pois > tg_pois_gap_max_ge2):
                            ok_pois = False

                        if ok_pois:
                            cand.append(("AWAY_TG15", float(p_a_ge2), True, float(gap_pois) if np.isfinite(gap_pois) else np.nan))

                    if len(cand) == 2:
                        cand = sorted(cand, key=lambda x: x[1], reverse=True)
                        if abs(cand[0][1] - cand[1][1]) < float(getattr(args, "tg_ambig_delta", 0.05)):
                            cand = []
                        else:
                            cand = [cand[0]]

                    for pick_tg, p_tg, tg_pois_ok, tg_pois_gap in cand:
                        rows.append({
                            **common,
                            "market": "tg15",
                            "bookie_pick": pick_tg,
                            "selection": pick_tg,
                            "model_top_pick": pick_tg,
                            "model_p_for_bookie": float(p_tg),
                            "agree_model_vs_bookie": 1,
                            "tg_pois_ok": int(bool(tg_pois_ok)),
                            "tg_pois_gap": float(tg_pois_gap) if np.isfinite(tg_pois_gap) else np.nan,
                        })

                # --- TG25 ---
                if "tg25" in markets:
                    cand = []

                    # HOME_TG25 uses GE3
                    if np.isfinite(p_h_ge3) and (p_h_ge3 >= tg25_pmin) and _tg_dir_ok("HOME_TG25"):
                        ok_pois = (np.isfinite(pois_h_ge3) and (pois_h_ge3 >= tg_pois_ge3_min))
                        gap_pois = (float(p_h_ge3) - float(pois_h_ge3)) if np.isfinite(pois_h_ge3) else np.nan

                        if ok_pois and np.isfinite(gap_pois) and (gap_pois > tg_pois_gap_max_ge3):
                            ok_pois = False

                        if ok_pois:
                            cand.append(("HOME_TG25", float(p_h_ge3), True, float(gap_pois) if np.isfinite(gap_pois) else np.nan))

                    # AWAY_TG25 uses GE3
                    if np.isfinite(p_a_ge3) and (p_a_ge3 >= tg25_pmin) and _tg_dir_ok("AWAY_TG25"):
                        ok_pois = (np.isfinite(pois_a_ge3) and (pois_a_ge3 >= tg_pois_ge3_min))
                        gap_pois = (float(p_a_ge3) - float(pois_a_ge3)) if np.isfinite(pois_a_ge3) else np.nan

                        if ok_pois and np.isfinite(gap_pois) and (gap_pois > tg_pois_gap_max_ge3):
                            ok_pois = False

                        if ok_pois:
                            cand.append(("AWAY_TG25", float(p_a_ge3), True, float(gap_pois) if np.isfinite(gap_pois) else np.nan))

                    if len(cand) == 2:
                        cand = sorted(cand, key=lambda x: x[1], reverse=True)
                        if abs(cand[0][1] - cand[1][1]) < float(getattr(args, "tg_ambig_delta", 0.05)):
                            cand = []
                        else:
                            cand = [cand[0]]

                    for pick_tg, p_tg, tg_pois_ok, tg_pois_gap in cand:
                        rows.append({
                            **common,
                            "market": "tg25",
                            "bookie_pick": pick_tg,
                            "selection": pick_tg,
                            "model_top_pick": pick_tg,
                            "model_p_for_bookie": float(p_tg),
                            "agree_model_vs_bookie": 1,
                            "tg_pois_ok": int(bool(tg_pois_ok)),
                            "tg_pois_gap": float(tg_pois_gap) if np.isfinite(tg_pois_gap) else np.nan,
                        })

    out = pd.DataFrame(rows)
    # Final canonical odds schema stamp (single source of truth)
    out = _stamp_canonical_odds_schema(out, debug=bool(getattr(args, "debug", False)))
    if out.empty:
        print("No rows produced.")
        return
    # Canonicalise BTTS representation (market='btts', selection in {'YES','NO'})
    out = _canon_btts_market_selection(out)
    # ------------------------------------------------------------------
    # Early backfill: ensure FTR bookie_od is populated from 1X2 odds
    # as soon as the canonical schema exists.
    # ------------------------------------------------------------------
    try:
        out = _fill_ftr_bookie_od_from_1x2(out)
    except Exception:
        pass
    # ------------------------------------------------------------------
    # Ensure fixture_key_ascii exists for audit/debug and truth-join safety.
    # This is a diacritics-safe key built from match_date + team names.
    # ------------------------------------------------------------------
    try:
        if ("fixture_key_ascii" not in out.columns) or (
            out["fixture_key_ascii"].astype("string").fillna("").str.strip().eq("").all()
        ):
            out["fixture_key_ascii"] = out.apply(_match_key_ascii, axis=1)
            out["fixture_key_ascii"] = out["fixture_key_ascii"].astype("string").fillna("").str.strip()
    except Exception:
        # Last-resort: fold the already-built fixture_key string.
        try:
            out["fixture_key_ascii"] = out.get("fixture_key", "").astype("string").fillna("").map(_ascii_fold).astype("string")
        except Exception:
            pass
    # ------------------------------------------------------------------
    # Optional: emit extra FTR candidate rows (HOME/DRAW/AWAY) per fixture.
    #
    # Why: the default FTR path only emits the bookie's strongest side.
    # For slip-building we need higher-odds anchor legs (often DRAW/AWAY)
    # when the model probability is decent.
    #
    # This uses the existing FTR rows as the per-fixture carrier of:
    #   - od_home / od_draw / od_away
    #   - confidence_home / confidence_draw / confidence_away
    # and expands them into additional rows with bookie_pick set to each outcome.
    # ------------------------------------------------------------------
    try:
        if bool(getattr(args, "emit_ftr_candidates", False)):
            m_ftr = out.get("market", "").astype(str).str.lower().eq("ftr") if "market" in out.columns else pd.Series(False, index=out.index)
            ftr = out.loc[m_ftr].copy()

            need_cols = [
                "league",
                "fixture_key",
                "od_home",
                "od_draw",
                "od_away",
                "confidence_home",
                "confidence_draw",
                "confidence_away",
            ]

            # Guard: only build FTR candidate rows when odds + model confidences exist and are finite.
            missing = [c for c in need_cols if c not in ftr.columns]
            if missing:
                if bool(getattr(args, "debug", False)):
                    print(f"[FTR_CAND_GUARD] skip candidates: missing cols={missing}")
                ftr = ftr.iloc[0:0].copy()
            else:
                oh = pd.to_numeric(ftr["od_home"], errors="coerce")
                od = pd.to_numeric(ftr["od_draw"], errors="coerce")
                oa = pd.to_numeric(ftr["od_away"], errors="coerce")
                pH = pd.to_numeric(ftr["confidence_home"], errors="coerce")
                pD = pd.to_numeric(ftr["confidence_draw"], errors="coerce")
                pA = pd.to_numeric(ftr["confidence_away"], errors="coerce")

                good_odds = oh.gt(1.0) & od.gt(1.0) & oa.gt(1.0)
                good_probs = pH.between(0.0, 1.0) & pD.between(0.0, 1.0) & pA.between(0.0, 1.0)
                good = (good_odds & good_probs).fillna(False)

                if not bool(good.any()):
                    if bool(getattr(args, "debug", False)):
                        print("[FTR_CAND_GUARD] skip candidates: no rows with valid odds+probs")
                    ftr = ftr.iloc[0:0].copy()
                else:
                    ftr = ftr.loc[good].copy()

            if not ftr.empty:
                # Coerce odds + probs
                for c in ("od_home", "od_draw", "od_away"):
                    ftr[c] = pd.to_numeric(ftr[c], errors="coerce")
                for c in ("confidence_home", "confidence_draw", "confidence_away"):
                    ftr[c] = pd.to_numeric(ftr[c], errors="coerce").clip(0.0, 1.0)

                # Implieds + no-vig implieds
                imp_h = (1.0 / ftr["od_home"]).where(ftr["od_home"] > 1.0)
                imp_d = (1.0 / ftr["od_draw"]).where(ftr["od_draw"] > 1.0)
                imp_a = (1.0 / ftr["od_away"]).where(ftr["od_away"] > 1.0)
                overround = (imp_h + imp_d + imp_a)
                imp_nv_h = (imp_h / overround).where(overround > 0)
                imp_nv_d = (imp_d / overround).where(overround > 0)
                imp_nv_a = (imp_a / overround).where(overround > 0)

                # Bookie spread (no-vig top1 - top2)
                try:
                    vv = pd.concat([imp_nv_h, imp_nv_d, imp_nv_a], axis=1).to_numpy(dtype=float)
                    vv = np.where(np.isfinite(vv), vv, -1e9)
                    vv = np.sort(vv, axis=1)
                    spread = (vv[:, -1] - vv[:, -2]).astype(float)
                    ftr["bookie_spread"] = pd.Series(spread, index=ftr.index)
                except Exception:
                    pass

                def _mk(side: str, od_col: str, p_col: str, imp: pd.Series, imp_nv: pd.Series) -> pd.DataFrame:
                    dfc = ftr.copy()
                    dfc["is_fixture_primary"] = 0
                    dfc["market"] = "ftr"
                    dfc["bookie_pick"] = side
                    dfc["bookie_od"] = pd.to_numeric(dfc[od_col], errors="coerce")
                    dfc["bookie_implied"] = pd.to_numeric(imp, errors="coerce")
                    dfc["bookie_overround"] = pd.to_numeric(overround, errors="coerce")
                    dfc["bookie_implied_novig"] = pd.to_numeric(imp_nv, errors="coerce")
                    dfc["model_p_for_bookie"] = pd.to_numeric(dfc[p_col], errors="coerce").clip(0.0, 1.0)
                    dfc["is_fixture_primary"] = 0

                    # keep model_top_pick as the argmax of the original 3-way distribution
                    if "model_top_pick" not in dfc.columns:
                        dfc["model_top_pick"] = ""
                    dfc["agree_model_vs_bookie"] = (dfc["model_top_pick"].astype(str).str.upper() == side).astype(int)

                    dfc["pool_tier"] = "ANCHOR_CAND"
                    dfc["od_source"] = "ftr_candidate"
                    return dfc

                c_home = _mk("HOME", "od_home", "confidence_home", imp_h, imp_nv_h)
                c_draw = _mk("DRAW", "od_draw", "confidence_draw", imp_d, imp_nv_d)
                c_away = _mk("AWAY", "od_away", "confidence_away", imp_a, imp_nv_a)

                cands = pd.concat([c_home, c_draw, c_away], ignore_index=True, sort=False)
                # Candidate rows must never be counted as primary rows in evaluation
                cands["is_fixture_primary"] = 0
                # Ensure identity columns exist on candidate rows
                if "pool_tier" not in cands.columns:
                    cands["pool_tier"] = "ANCHOR_CAND"
                if "od_source" not in cands.columns:
                    cands["od_source"] = "ftr_candidate"
                # --- Derived debug flags for candidate rows (per-candidate side) ---
                try:
                    side_s = cands.get("bookie_pick", "").astype(str).str.upper().str.strip()
                    ppg_home_s = pd.to_numeric(cands.get("ppg_home_pre", np.nan), errors="coerce")
                    ppg_away_s = pd.to_numeric(cands.get("ppg_away_pre", np.nan), errors="coerce")
                    ppg_diff_s = pd.to_numeric(cands.get("ppg_diff_pre", (ppg_home_s - ppg_away_s)), errors="coerce")

                    _glue_dmin = float(getattr(args, "ftr_glue_ppg_diff_min", 0.70))
                    _glue_opp_max = float(getattr(args, "ftr_glue_ppg_opp_max", 1.00))

                    opp_ppg_s = pd.Series(np.nan, index=cands.index, dtype=float)
                    opp_ppg_s = opp_ppg_s.mask(side_s.eq("HOME"), ppg_away_s)
                    opp_ppg_s = opp_ppg_s.mask(side_s.eq("AWAY"), ppg_home_s)

                    glue_ok_s = (
                        (side_s.eq("HOME") & (ppg_diff_s >= _glue_dmin) & (opp_ppg_s <= _glue_opp_max))
                        | (side_s.eq("AWAY") & (ppg_diff_s <= -_glue_dmin) & (opp_ppg_s <= _glue_opp_max))
                    )

                    _dt_od_max = float(getattr(args, "ftr_drawtrap_od_max", 1.30))
                    _dt_opp_min = float(getattr(args, "ftr_drawtrap_opp_ppg_min", 1.20))
                    od_sel_s = pd.to_numeric(cands.get("bookie_od", np.nan), errors="coerce")
                    drawtrap_s = (
                        side_s.isin(["HOME", "AWAY"])
                        & od_sel_s.notna()
                        & (od_sel_s <= _dt_od_max)
                        & opp_ppg_s.notna()
                        & (opp_ppg_s >= _dt_opp_min)
                    )

                    cands["ftr_ppg_glue_ok"] = glue_ok_s.fillna(False).astype(int)
                    cands["ftr_drawtrap_flag"] = drawtrap_s.fillna(False).astype(int)
                except Exception:
                    cands["ftr_ppg_glue_ok"] = 0
                    cands["ftr_drawtrap_flag"] = 0

                # Filter candidates (anchor-like)
                od_min = float(getattr(args, "ftr_cand_od_min", 2.50))
                od_max = float(getattr(args, "ftr_cand_od_max", 6.00))
                pmin = float(getattr(args, "ftr_cand_pmin", 0.20))
                gap_min = float(getattr(args, "ftr_cand_gap_min", -0.05))
                margin_min = float(getattr(args, "ftr_cand_margin_min", 0.05))
                max_per_fixture = int(getattr(args, "ftr_cand_max_per_fixture", 2))
                max_per_league = int(getattr(args, "ftr_cand_max_per_league", 0))

                od = pd.to_numeric(cands.get("bookie_od"), errors="coerce")
                pm = pd.to_numeric(cands.get("model_p_for_bookie"), errors="coerce")
                inv = pd.to_numeric(cands.get("bookie_implied_novig"), errors="coerce")
                gap_nv = pm - inv
                # Candidate filtering (anchor-ish): odds band + model prob + not-too-negative gap + stability margin
                ftr_margin_s = pd.to_numeric(cands.get("ftr_margin", np.nan), errors="coerce").fillna(0.0)

                keep = (
                    od.notna()
                    & (od >= od_min)
                    & (od <= od_max)
                    & pm.notna()
                    & (pm >= pmin)
                    & inv.notna()
                    & (gap_nv >= gap_min)
                    & (ftr_margin_s >= margin_min)
                )


                cands = cands.loc[keep].copy()

                # Optional directional PPG gate (kills nonsense like AWAY @ 13.0 when ppg_diff is strongly HOME)
                if not cands.empty and bool(getattr(args, "ftr_cand_use_ppg", False)):
                    ppg_diff = pd.to_numeric(cands.get("ppg_diff_pre", np.nan), errors="coerce")
                    ppg_home = pd.to_numeric(cands.get("ppg_home_pre", np.nan), errors="coerce")
                    ppg_away = pd.to_numeric(cands.get("ppg_away_pre", np.nan), errors="coerce")

                    # opponent ppg depends on the candidate side
                    pick_s = cands.get("bookie_pick", "").astype("string").fillna("").str.upper().str.strip()
                    ppg_opp = pd.Series(np.nan, index=cands.index)
                    ppg_opp.loc[pick_s.eq("HOME")] = ppg_away.loc[pick_s.eq("HOME")]
                    ppg_opp.loc[pick_s.eq("AWAY")] = ppg_home.loc[pick_s.eq("AWAY")]
                    # For DRAW, define opponent as the stronger side (conservative)
                    ppg_opp.loc[pick_s.eq("DRAW")] = np.maximum(ppg_home.loc[pick_s.eq("DRAW")], ppg_away.loc[pick_s.eq("DRAW")])

                    # Persist ppg_opp into the candidate rows for downstream debugging
                    cands["ppg_opp"] = pd.to_numeric(ppg_opp, errors="coerce")

                    ppg_home_min = float(getattr(args, "ftr_cand_ppg_home_min", 0.35))
                    ppg_away_max = float(getattr(args, "ftr_cand_ppg_away_max", -0.35))
                    ppg_draw_abs_max = float(getattr(args, "ftr_cand_ppg_draw_abs_max", 0.35))
                    ppg_opp_max = float(getattr(args, "ftr_cand_ppg_opp_max", 1.20))

                    m_home = pick_s.eq("HOME") & (ppg_diff >= ppg_home_min) & (ppg_opp <= ppg_opp_max)
                    m_away = pick_s.eq("AWAY") & (ppg_diff <= ppg_away_max) & (ppg_opp <= ppg_opp_max)
                    m_draw = pick_s.eq("DRAW") & (ppg_diff.abs() <= ppg_draw_abs_max)

                    cands = cands[(m_home | m_away | m_draw).fillna(False)].copy()

                # Ensure debug flags exist on candidate rows (avoid NaNs downstream) WITHOUT clobbering
                if not cands.empty:
                    cands["ftr_ppg_glue_ok"] = pd.to_numeric(cands.get("ftr_ppg_glue_ok", 0), errors="coerce").fillna(0).astype(int)
                    cands["ftr_drawtrap_flag"] = pd.to_numeric(cands.get("ftr_drawtrap_flag", 0), errors="coerce").fillna(0).astype(int)

                # Rank candidates: prefer higher gap_nv, then higher model prob, then lower odds (less extreme)
                if not cands.empty:
                    cands["gap_cand_novig"] = gap_nv.loc[cands.index].astype(float)
                    cands["p_cand"] = pm.loc[cands.index].astype(float)
                    cands["od_cand"] = od.loc[cands.index].astype(float)

                    cands = cands.sort_values(
                        ["league", "fixture_key", "gap_cand_novig", "p_cand", "od_cand"],
                        ascending=[True, True, False, False, True],
                    ).reset_index(drop=True)

                    # NEW: rank candidates within each fixture (1 = best)
                    cands["candidate_rank"] = (
                        cands.groupby(["league", "fixture_key"]).cumcount() + 1
                    ).astype(int)

                    # Keep top-N candidates per fixture (preserving explicit rank)
                    if max_per_fixture > 0:
                        cands = cands[cands["candidate_rank"] <= int(max_per_fixture)].copy()

                    # Optional: cap total candidate rows per league
                    if max_per_league and max_per_league > 0:
                        cands = cands.groupby(["league"], as_index=False, group_keys=False).head(max_per_league)

                    # Clean helper cols
                    cands = cands.drop(columns=["gap_cand_novig", "p_cand", "od_cand"], errors="ignore")

                if not cands.empty:
                    out = pd.concat([out, cands], ignore_index=True, sort=False)

                    # De-dupe on the canonical identity
                    for c in ("league", "fixture_key", "market", "bookie_pick"):
                        if c in out.columns:
                            out[c] = out[c].astype("string").fillna("").str.strip()
                    subset = [c for c in ("league", "fixture_key", "market", "bookie_pick") if c in out.columns]
                    if subset:
                        out = out.drop_duplicates(subset=subset, keep="first")

                    if bool(getattr(args, "debug", False)):
                        try:
                            od2 = pd.to_numeric(out.get("bookie_od"), errors="coerce")
                            n25 = int((od2 >= 2.5).sum()) if od2 is not None else 0
                            print(
                                f"[bookie_allmarkets] emitted FTR candidates: +{len(cands)} rows "
                                f"| band=[{od_min:.2f},{od_max:.2f}] pmin={pmin:.2f} gapmin={gap_min:+.2f} margin_min={margin_min:.2f} "
                                f"| bookie_od>=2.5 now n={n25}"
                            )
                        except Exception:
                            pass
    except Exception as _e:
        if bool(getattr(args, "debug", False)):
            print(f"ℹ️ emit_ftr_candidates skipped: {_e}")
    # --- CloseMatch router flag (must be AFTER candidates are appended) ---
    try:
        out = _attach_close_match_flag(out)
    except Exception as _e:
        if bool(getattr(args, "debug", False)):
            print(f"ℹ️ close_match_flag stamp skipped: {_e}")
    # ------------------------------------------------------------------
    # Derive canonical side-market probability columns for signal banding.
    #
    # signal_layers expects per-fixture probabilities for:
    #   - P(OVER25)  -> prob_over25_v2
    #   - P(BTTS YES)-> prob_btts_v2
    #
    # Our ALLMARKETS rows store `model_p_for_bookie` which is P(bookie_pick).
    # For ou25 and btts markets, we can safely invert to recover the canonical
    # probability because we define complements (UNDER=1-OVER, NO=1-YES).
    # ------------------------------------------------------------------
    def _norm_market_selection(df: pd.DataFrame) -> pd.DataFrame:
        """Canonicalise market + selection for OU25 and BTTS.

        Output conventions used by downstream prob-derivation + signal attach:
          - OU25: market='ou25', selection in {'OVER25','UNDER25'}
          - BTTS: market='btts', selection in {'YES','NO'}

        This function is intentionally small and defensive; it does NOT try to
        normalise FTR or TG markets.
        """
        out = df.copy()

        # Ensure columns exist
        if "market" not in out.columns:
            out["market"] = ""
        if "selection" not in out.columns:
            # fall back to bookie_pick if present
            out["selection"] = out.get("bookie_pick", "")

        m = out["market"].astype("string").fillna("").str.strip().str.lower()
        s = out["selection"].astype("string").fillna("").str.strip().str.upper()

        # --------------------
        # OU25 canonicalisation
        # --------------------
        m_ou_alias = m.isin(["ou25", "ou_25", "o/u25", "ou_2_5", "totals_25", "totals"])
        m_over = m.eq("over25")
        m_under = m.eq("under25")

        if bool((m_ou_alias | m_over | m_under).any()):
            # force market to 'ou25'
            m2 = m.copy()
            m2 = m2.mask(m_ou_alias | m_over | m_under, "ou25")

            # normalise selection
            s_over = s.isin(["OVER", "OVER25", "OVER 2.5", "O2.5", "O2_5", "O25"]) | m_over
            s_under = s.isin(["UNDER", "UNDER25", "UNDER 2.5", "U2.5", "U2_5", "U25"]) | m_under

            s2 = s.copy()
            s2 = s2.mask(m2.eq("ou25") & s_over, "OVER25")
            s2 = s2.mask(m2.eq("ou25") & s_under, "UNDER25")

            out["market"] = m2
            out["selection"] = s2

            # refresh locals
            m = out["market"].astype("string").fillna("").str.strip().str.lower()
            s = out["selection"].astype("string").fillna("").str.strip().str.upper()

        # ----------------------
        # BTTS canonicalisation
        # ----------------------
        # Delegate BTTS normalisation to the single canonical rule to avoid drift.
        # Convention:
        #   - market == 'btts'
        #   - selection in {'YES','NO'}
        # Back-compat: legacy 'btts_no' rows become market='btts', selection='NO'.
        out = _canon_btts_market_selection(out)

        # Keep bookie_pick aligned with canonical BTTS selection whenever selection is valid.
        # This prevents later YES-only drift when downstream code falls back to bookie_pick.
        if "bookie_pick" in out.columns:
            m_bt = out["market"].astype("string").fillna("").str.strip().str.lower().eq("btts")
            sel_bt = out["selection"].astype("string").fillna("").str.strip().str.upper()
            out.loc[m_bt & sel_bt.isin(["YES", "NO"]), "bookie_pick"] = sel_bt.loc[m_bt & sel_bt.isin(["YES", "NO"])]

        return out

    try:
        # Make sure base columns are clean
        out["market"] = out.get("market", "").astype("string").fillna("").str.strip().str.lower()
        out["bookie_pick"] = out.get("bookie_pick", "").astype("string").fillna("").str.strip().str.upper()
        out["model_p_for_bookie"] = pd.to_numeric(out.get("model_p_for_bookie", np.nan), errors="coerce")

        # Apply per-league FTR calibration (isotonic/platt) before gates
        try:
            _cal_models = _load_ftr_calibration_models()
            if _cal_models:
                out = _apply_ftr_calibration(out, _cal_models)
        except Exception as _e_cal:
            if bool(getattr(args, "debug", False)):
                print(f"ℹ️ FTR calibration skipped: {_e_cal}")

        # Ensure the canonical prob columns exist
        if "prob_over25_v2" not in out.columns:
            out["prob_over25_v2"] = np.nan
        if "prob_btts_v2" not in out.columns:
            out["prob_btts_v2"] = np.nan

        # Build a tiny canonical view without mutating the main market labels
        tmp = pd.DataFrame({
            "market": out.get("market", ""),
            "selection": out.get("selection", out.get("bookie_pick", "")),
            "model_p_for_bookie": out.get("model_p_for_bookie", np.nan),
        })
        tmp = _norm_market_selection(tmp)

        # OU25 rows -> prob_over25_v2 (canonical is P(OVER25))
        m_ou = tmp["market"].astype(str).str.lower().eq("ou25")
        if bool(m_ou.any()):
            p = pd.to_numeric(tmp.loc[m_ou, "model_p_for_bookie"], errors="coerce").astype(float)
            sel = tmp.loc[m_ou, "selection"].astype(str).str.upper().str.strip()

            # When selection is OVER25, model_p_for_bookie is P(OVER25)
            out.loc[m_ou & sel.eq("OVER25"), "prob_over25_v2"] = p.loc[sel.eq("OVER25")]
            # When selection is UNDER25, model_p_for_bookie is P(UNDER25) => P(OVER25)=1-P(UNDER25)
            out.loc[m_ou & sel.eq("UNDER25"), "prob_over25_v2"] = (1.0 - p.loc[sel.eq("UNDER25")])

        # BTTS rows -> prob_btts_v2 (canonical is P(YES))
        m_bt = tmp["market"].astype(str).str.lower().eq("btts")
        if bool(m_bt.any()):
            bt_idx = tmp.index[m_bt]
            p_bt = pd.to_numeric(tmp.loc[bt_idx, "model_p_for_bookie"], errors="coerce").astype(float)
            sel_bt = tmp.loc[bt_idx, "selection"].astype(str).str.upper().str.strip()

            yes_idx = bt_idx[sel_bt.eq("YES")]
            no_idx = bt_idx[sel_bt.eq("NO")]

            if len(yes_idx):
                out.loc[yes_idx, "prob_btts_v2"] = p_bt.loc[yes_idx]
            if len(no_idx):
                out.loc[no_idx, "prob_btts_v2"] = (1.0 - p_bt.loc[no_idx])

        # Clip to sane bounds
        out["prob_over25_v2"] = pd.to_numeric(out["prob_over25_v2"], errors="coerce").clip(0.0, 1.0)
        out["prob_btts_v2"] = pd.to_numeric(out["prob_btts_v2"], errors="coerce").clip(0.0, 1.0)

        # --- DEBUG: coverage of derived canonical side-market probs ---
        if bool(getattr(args, "debug", False)):
            try:
                total_rows = int(len(out))
                needed = int(out["fixture_key"].nunique()) if "fixture_key" in out.columns else total_rows

                m_btts = out["market"].astype("string").fillna("").str.lower().str.strip().eq("btts")
                m_ou25 = out["market"].astype("string").fillna("").str.lower().str.strip().eq("ou25")

                # Prefer canonical columns; fall back to legacy *_v2 aliases if needed.
                col_btts = "prob_btts" if "prob_btts" in out.columns else ("prob_btts_v2" if "prob_btts_v2" in out.columns else "")
                col_o25  = "prob_over25" if "prob_over25" in out.columns else ("prob_over25_v2" if "prob_over25_v2" in out.columns else "")

                btts_ok = int(out.loc[m_btts, col_btts].notna().sum()) if (col_btts and bool(m_btts.any())) else 0
                o25_ok  = int(out.loc[m_ou25, col_o25].notna().sum()) if (col_o25 and bool(m_ou25.any())) else 0

                print(
                    f"[BTTS/O25 DERIVED] fixtures={needed} rows={total_rows} "
                    f"btts_rows={int(m_btts.sum())} btts_prob_nonnull={btts_ok} "
                    f"ou25_rows={int(m_ou25.sum())} ou25_prob_nonnull={o25_ok}"
                )

                # Market-scoped per-league coverage (avoid averaging over non-market rows)
                if "league" in out.columns:
                    print("[BTTS/O25 DERIVED] BTTS prob coverage by league (BTTS rows only)")
                    if bool(m_btts.any()) and bool(col_btts):
                        print(
                            out.loc[m_btts]
                            .assign(ok=out.loc[m_btts, col_btts].notna())
                            .groupby("league")["ok"].mean()
                            .sort_values(ascending=False)
                        )
                    else:
                        print("No BTTS rows emitted this run (likely implied-min filter) or prob column missing.")

                    print("[BTTS/O25 DERIVED] OU25 prob coverage by league (OU25 rows only)")
                    if bool(m_ou25.any()) and bool(col_o25):
                        print(
                            out.loc[m_ou25]
                            .assign(ok=out.loc[m_ou25, col_o25].notna())
                            .groupby("league")["ok"].mean()
                            .sort_values(ascending=False)
                        )
                    else:
                        print("No OU25 rows emitted this run (likely implied-min filter) or prob column missing.")

            except Exception as _e_cov:
                print(f"ℹ️ [BTTS/O25 DERIVED] coverage log skipped: {_e_cov}")
    except Exception:
        # Non-fatal: signal attachment below will fall back to NEUTRAL
        pass

    # --- Attach signal labels (side markets) on the final ALLMARKETS frame ---
    # Goal: provide `signal_over25` and `signal_btts` for deploy_gates (label-based gates).
    # We normalise to the V2 META conventions (over25/under25 + btts) on a temp frame
    # so signal-layer code works even when the source market is 'ou25'.
    try:
        sig_df = out.copy()

        # Required columns often expected by signal-layer code
        if "selection" not in sig_df.columns:
            sig_df["selection"] = sig_df.get("bookie_pick", "")
        if "p_model" not in sig_df.columns:
            sig_df["p_model"] = sig_df.get("model_p_for_bookie", np.nan)
        if "confidence" not in sig_df.columns:
            sig_df["confidence"] = sig_df["p_model"]

        # Normalise market + selection with one canonical helper
        sig_df["market"] = sig_df.get("market", "").astype("string").fillna("").str.strip().str.lower()
        sig_df["selection"] = sig_df.get("selection", "").astype("string").fillna("").str.strip().str.upper()
        sig_df = _norm_market_selection(sig_df)

        # Keep BTTS bookie_pick aligned with canonical selection inside the signal frame too.
        # Some downstream helpers may still inspect bookie_pick even when selection exists.
        if "bookie_pick" in sig_df.columns:
            m_bt_sig = sig_df["market"].astype("string").fillna("").str.strip().str.lower().eq("btts")
            sel_bt_sig = sig_df["selection"].astype("string").fillna("").str.strip().str.upper()
            sig_df.loc[m_bt_sig & sel_bt_sig.isin(["YES", "NO"]), "bookie_pick"] = sel_bt_sig.loc[m_bt_sig & sel_bt_sig.isin(["YES", "NO"])]

        # For signal-layer compatibility, split OU25 into over25/under25 markets
        m_ou = sig_df["market"].astype(str).str.lower().eq("ou25")
        if bool(m_ou.any()):
            sel = sig_df.loc[m_ou, "selection"].astype(str).str.upper().str.strip()
            sig_df.loc[m_ou & sel.eq("OVER25"), "market"] = "over25"
            sig_df.loc[m_ou & sel.eq("UNDER25"), "market"] = "under25"

        # BTTS stays market='btts' with selection YES/NO (no btts_no market)

        # ------------------------------------------------------------------
        # Side-prob coalesce: ensure signal-layer inputs exist even when
        # side-prob feature columns were disabled / missing.
        # This prevents OU25 signals collapsing to all-NEUTRAL.
        # ------------------------------------------------------------------
        try:
            sig_df = _coalesce_side_prob_cols_for_signals(sig_df)
        except Exception:
            pass

        used = None

        def _fallback_attach_side_signals(df_: pd.DataFrame) -> pd.DataFrame:
            """Fallback signal labels when signal_bands.json is missing/stale.

            Uses only probability columns already present on the ALLMARKETS frame.
            Produces:
              - signal_over25 for markets over25/under25
              - signal_btts for market btts with selection YES/NO
            """
            if df_ is None or df_.empty:
                return df_
            z = df_.copy()
            if "market" not in z.columns:
                return z

            mk = z["market"].astype("string").fillna("").str.lower().str.strip()
            if "selection" not in z.columns:
                z["selection"] = z.get("bookie_pick", "")
            sel = z["selection"].astype("string").fillna("").str.upper().str.strip()

            # Ensure prob cols exist
            if "prob_over25_v2" not in z.columns:
                z["prob_over25_v2"] = pd.to_numeric(z.get("prob_over25", np.nan), errors="coerce")
            if "prob_btts_v2" not in z.columns:
                z["prob_btts_v2"] = pd.to_numeric(z.get("prob_btts", np.nan), errors="coerce")

            p_over = pd.to_numeric(z.get("prob_over25_v2", np.nan), errors="coerce").clip(0.0, 1.0)
            p_btts = pd.to_numeric(z.get("prob_btts_v2", np.nan), errors="coerce").clip(0.0, 1.0)

            # Output cols
            if "signal_over25" not in z.columns:
                z["signal_over25"] = "NEUTRAL"
            if "signal_btts" not in z.columns:
                z["signal_btts"] = "NEUTRAL"

            # --- OU25 signals (ROW-AWARE; derive direction from emitted pick) ---
            # Priority:
            #   1) If bookie_pick/selection explicitly says OVER25/UNDER25, trust it.
            #   2) Otherwise, fall back to market tokens (ou25/over25/under25 aliases).
            mk = z.get("market", "").astype("string").fillna("").str.lower().str.strip()
            pick = (
                z.get("bookie_pick", z.get("selection", ""))
                .astype("string")
                .fillna("")
                .str.upper()
                .str.strip()
            )

            m_ou = mk.eq("ou25")

            pick_is_over = pick.eq("OVER25")
            pick_is_under = pick.eq("UNDER25")

            # 1) Explicit pick wins (works for both ou25 + legacy market tokens)
            is_over25 = pick_is_over
            is_under25 = pick_is_under

            # 2) If pick is NOT explicit, fall back to market tokens
            no_explicit_pick = ~(pick_is_over | pick_is_under)
            is_over25 = is_over25 | (no_explicit_pick & (m_ou | mk.isin(["over25", "o25"])))
            is_under25 = is_under25 | (no_explicit_pick & (m_ou | mk.isin(["under25", "u25"])))

            # Ensure mutual exclusivity
            is_under25 = is_under25 & (~is_over25)

            # Over side strength from p_over
            over_vs = (p_over >= 0.70)
            over_s  = (p_over >= 0.62) & (p_over < 0.70)
            over_w  = (p_over >= 0.56) & (p_over < 0.62)

            # Under side strength from p_under = 1 - p_over
            p_under = (1.0 - p_over).clip(0.0, 1.0)
            under_vs = (p_under >= 0.70)
            under_s  = (p_under >= 0.62) & (p_under < 0.70)
            under_w  = (p_under >= 0.56) & (p_under < 0.62)

            z.loc[is_over25 & over_vs, "signal_over25"] = "VERY_STRONG_OVER"
            z.loc[is_over25 & over_s,  "signal_over25"] = "STRONG_OVER"
            z.loc[is_over25 & over_w,  "signal_over25"] = "WEAK_OVER"
            z.loc[is_over25 & ~(over_vs | over_s | over_w), "signal_over25"] = "NEUTRAL"

            z.loc[is_under25 & under_vs, "signal_over25"] = "VERY_STRONG_UNDER"
            z.loc[is_under25 & under_s,  "signal_over25"] = "STRONG_UNDER"
            z.loc[is_under25 & under_w,  "signal_over25"] = "WEAK_UNDER"
            z.loc[is_under25 & ~(under_vs | under_s | under_w), "signal_over25"] = "NEUTRAL"

            # --- BTTS signals (market='btts', selection YES/NO) ---
            is_btts = mk.eq("btts")
            p_no = (1.0 - p_btts).clip(0.0, 1.0)

            yes_vs = is_btts & sel.eq("YES") & (p_btts >= 0.70)
            yes_s  = is_btts & sel.eq("YES") & (p_btts >= 0.62) & (p_btts < 0.70)
            yes_w  = is_btts & sel.eq("YES") & (p_btts >= 0.56) & (p_btts < 0.62)

            no_vs = is_btts & sel.eq("NO") & (p_no >= 0.70)
            no_s  = is_btts & sel.eq("NO") & (p_no >= 0.62) & (p_no < 0.70)
            no_w  = is_btts & sel.eq("NO") & (p_no >= 0.56) & (p_no < 0.62)

            z.loc[yes_vs, "signal_btts"] = "VERY_STRONG_YES"
            z.loc[yes_s,  "signal_btts"] = "STRONG_YES"
            z.loc[yes_w,  "signal_btts"] = "WEAK_YES"
            z.loc[is_btts & sel.eq("YES") & ~(yes_vs | yes_s | yes_w), "signal_btts"] = "NEUTRAL"

            z.loc[no_vs, "signal_btts"] = "VERY_STRONG_NO"
            z.loc[no_s,  "signal_btts"] = "STRONG_NO"
            z.loc[no_w,  "signal_btts"] = "WEAK_NO"
            z.loc[is_btts & sel.eq("NO") & ~(no_vs | no_s | no_w), "signal_btts"] = "NEUTRAL"

            return z

        import inspect as _inspect

        def _call_attach(fn, df_):
            nonlocal used
            used = getattr(fn, "__module__", "") + "." + getattr(fn, "__name__", str(fn))
            try:
                params = _inspect.signature(fn).parameters
            except Exception:
                params = {}

            # Try common param names for modelstore path
            for pname in ("modelstore", "model_store", "model_root", "model_dir", "modelstore_path"):
                if pname in params:
                    try:
                        return fn(df_, **{pname: str(modelstore)})
                    except TypeError:
                        pass

            return fn(df_)

        # Prefer explicit probability columns we derived above.
        if callable(_attach_signal_layers_if_available):
            try:
                sig_df = _attach_signal_layers_if_available(
                    sig_df,
                    league_col="league",
                    over25_col="prob_over25_v2",
                    btts_col="prob_btts_v2",
                )
                used = getattr(_attach_signal_layers_if_available, "__module__", "") + "." + getattr(_attach_signal_layers_if_available, "__name__", str(_attach_signal_layers_if_available))
            except TypeError:
                sig_df = _call_attach(_attach_signal_layers_if_available, sig_df)
        elif callable(_attach_signal_layers):
            try:
                # Side-prob coalesce: ensure signal-layer inputs exist even when
                # side-prob feature columns were disabled / missing.
                try:
                    sig_df = _coalesce_side_prob_cols_for_signals(sig_df)
                except Exception:
                    pass
                sig_df = _attach_signal_layers(
                    sig_df,
                    league_col="league",
                    over25_col="prob_over25_v2",
                    btts_col="prob_btts_v2",
                )
                used = getattr(_attach_signal_layers, "__module__", "") + "." + getattr(_attach_signal_layers, "__name__", str(_attach_signal_layers))
            except TypeError:
                sig_df = _call_attach(_attach_signal_layers, sig_df)



        # Fallback when external signal bands are missing/stale or attachment leaves side signals blank.
        need_fallback_over = ("signal_over25" not in sig_df.columns) or sig_df.get("signal_over25", pd.Series(index=sig_df.index, dtype="object")).astype("string").fillna("").str.strip().eq("").all() or sig_df.get("signal_over25", pd.Series(index=sig_df.index, dtype="object")).astype("string").fillna("").str.upper().str.strip().eq("NEUTRAL").all()
        need_fallback_btts = ("signal_btts" not in sig_df.columns) or sig_df.get("signal_btts", pd.Series(index=sig_df.index, dtype="object")).astype("string").fillna("").str.strip().eq("").all() or sig_df.get("signal_btts", pd.Series(index=sig_df.index, dtype="object")).astype("string").fillna("").str.upper().str.strip().eq("NEUTRAL").all()

        if need_fallback_over or need_fallback_btts:
            sig_df = _fallback_attach_side_signals(sig_df)
            used = (used + " + fallback") if used else "fallback_side_probs"
        # Re-canonicalise BTTS just before writing signals back, and keep bookie_pick aligned.
        sig_df = _norm_market_selection(sig_df)
        if "bookie_pick" in sig_df.columns:
            m_bt_sig = sig_df["market"].astype("string").fillna("").str.strip().str.lower().eq("btts")
            sel_bt_sig = sig_df["selection"].astype("string").fillna("").str.strip().str.upper()
            sig_df.loc[m_bt_sig & sel_bt_sig.isin(["YES", "NO"]), "bookie_pick"] = sel_bt_sig.loc[m_bt_sig & sel_bt_sig.isin(["YES", "NO"])]

        # Bring signals back to the output frame
        if "signal_over25" in sig_df.columns:
            out["signal_over25"] = sig_df["signal_over25"]
        if "signal_btts" in sig_df.columns:
            out["signal_btts"] = sig_df["signal_btts"]

        # Ensure columns exist for downstream consistency, even if attach failed
        if "signal_over25" not in out.columns:
            out["signal_over25"] = "NEUTRAL"
        # Ensure OU25 signal direction is ROW-aware (OVER25 vs UNDER25) after signal attachment.
        # This must run AFTER bookie_pick has been emitted and AFTER any fixture-level signal layers.
        try:
            out = _enforce_row_aware_ou25_signal(out)
        except Exception:
            pass
        if "signal_btts" not in out.columns:
            out["signal_btts"] = "NEUTRAL"

        # --------------------------------------------------------------
        # BTTS signal split
        #   signal_btts_fixture = legacy / fixture-level label
        #   signal_btts_side    = row-aware label from selection + p_pick
        #   signal_btts         = side-aware label for deploy logic
        # --------------------------------------------------------------
        try:
            m_btts_final = (
                out["market"]
                .astype("string")
                .fillna("")
                .str.lower()
                .str.strip()
                .eq("btts")
            )
            prod_bt = out.get("product", pd.Series(pd.NA, index=out.index, dtype="string")).astype("string").fillna("").str.strip()
            lane_bt = out.get("model_lane", pd.Series(pd.NA, index=out.index, dtype="string")).astype("string").fillna("").str.strip()
            m_btts_model = m_btts_final & prod_bt.eq("BTTS_MODEL") & lane_bt.eq("btts_model")

            if "signal_btts_fixture" not in out.columns:
                out["signal_btts_fixture"] = pd.NA
            if "signal_btts_side" not in out.columns:
                out["signal_btts_side"] = pd.NA

            # Preserve legacy fixture-level label for diagnostics on primary BTTS rows only
            if "signal_btts" in out.columns:
                out.loc[m_btts_model, "signal_btts_fixture"] = out.loc[m_btts_model, "signal_btts"]

            # Build row-aware side label from emitted side probability on primary BTTS rows only
            _sel_bt = out.loc[m_btts_model, "selection"] if "selection" in out.columns else pd.Series(index=out.index[m_btts_model], dtype="object")
            _pp_bt = out.loc[m_btts_model, "model_p_for_bookie"] if "model_p_for_bookie" in out.columns else pd.Series(index=out.index[m_btts_model], dtype="float64")

            out.loc[m_btts_model, "signal_btts_side"] = [
                _btts_side_signal_from_pick_prob(sel, pp)
                for sel, pp in zip(_sel_bt.tolist(), _pp_bt.tolist())
            ]

            # Make BTTS signal row-aware for downstream gating on primary BTTS rows only
            out.loc[m_btts_model, "signal_btts"] = out.loc[m_btts_model, "signal_btts_side"]
            # Hard-mask BTTS signal columns to BTTS rows only
            for c in ["signal_btts", "signal_btts_fixture", "signal_btts_side"]:
                if c not in out.columns:
                    out[c] = pd.NA
                out.loc[~m_btts_final, c] = pd.NA
        except Exception:
            pass
        if bool(getattr(args, "debug", False)):
            try:
                so = out["signal_over25"].astype(str).value_counts().to_dict() if "signal_over25" in out.columns else {}
                sb = out["signal_btts"].astype(str).value_counts().to_dict() if "signal_btts" in out.columns else {}
                print(f"[bookie_allmarkets] signal attach used: {used}")
                print(f"[bookie_allmarkets] signal_over25 counts: {so}")
                print(f"[bookie_allmarkets] signal_btts counts:  {sb}")

                # Helpful: check for per-league signal-bands artefacts existence
                leagues_present = sorted(set(out["league"].astype(str).fillna("").tolist()))
                leagues_present = [x for x in leagues_present if x.strip()]
                missing = []
                found = 0
                for lg2 in leagues_present:
                    tag2 = _league_tag(lg2)
                    cands = [
                        modelstore / f"{tag2}_signal_bands.json",
                        modelstore / f"{tag2}_signal_bands_v2.json",
                        modelstore / tag2 / "signal_bands.json",
                        modelstore / tag2 / f"{tag2}_signal_bands.json",
                    ]
                    if any(p.exists() for p in cands):
                        found += 1
                    else:
                        missing.append(lg2)

                if leagues_present:
                    print(f"[bookie_allmarkets] signal_bands artefacts found for {found}/{len(leagues_present)} leagues")
                    if missing:
                        print(f"[bookie_allmarkets] signal_bands missing (first 10): {missing[:10]}")
            except Exception:
                pass
    except Exception as _e:
        # Do not fail the window runner if label attachment is unavailable.
        if "signal_over25" not in out.columns:
            out["signal_over25"] = "NEUTRAL"
        if "signal_btts" not in out.columns:
            out["signal_btts"] = "NEUTRAL"
        if bool(getattr(args, "debug", False)):
            print(f"ℹ️ signal label attach skipped: {_e}")
    # ------------------------------------------------------------------
    # Merge fixture-level λ columns (by league + fixture_key)
    # This makes λ available on every market row (incl. FTR).
    # ------------------------------------------------------------------
    try:
        if lambda_maps:
            lm = pd.concat(lambda_maps, ignore_index=True)
            lm["league"] = lm["league"].astype("string").fillna("").str.strip()
            lm["fixture_key"] = lm["fixture_key"].astype("string").fillna("").str.strip()

            out["league"] = out["league"].astype("string").fillna("").str.strip()
            out["fixture_key"] = out["fixture_key"].astype("string").fillna("").str.strip()

            out = out.merge(lm, on=["league", "fixture_key"], how="left", suffixes=("", "_lam"))

            for c in [
                "home_goals_pred",
                "away_goals_pred",
                "lambda_home",
                "lambda_away",
                "exp_goals_sum",
                "p00_est",
                "cs_home",
                "cs_away",
                "p_home_ge1",
                "p_away_ge1",
                "p_home_ge2",
                "p_away_ge2",
                "p_home_ge3",
                "p_away_ge3",
                "p_home_ge4",
                "p_away_ge4",
                "pois_home_ge2",
                "pois_away_ge2",
                "pois_home_ge3",
                "pois_away_ge3",
            ]:
                cu = c + "_lam"
                if cu in out.columns:
                    a = pd.to_numeric(out.get(c, np.nan), errors="coerce")
                    b = pd.to_numeric(out.get(cu, np.nan), errors="coerce")
                    out[c] = a.where(a.notna(), b)
                    out = out.drop(columns=[cu], errors="ignore")

        # Ensure schema stability even when λ map is empty
        for c in [
            "home_goals_pred",
            "away_goals_pred",
            "lambda_home",
            "lambda_away",
            "exp_goals_sum",
            "p00_est",
            "cs_home",
            "cs_away",
            "p_home_ge1",
            "p_away_ge1",
            "p_home_ge2",
            "p_away_ge2",
            "p_home_ge3",
            "p_away_ge3",
            "p_home_ge4",
            "p_away_ge4",
            "pois_home_ge2",
            "pois_away_ge2",
            "pois_home_ge3",
            "pois_away_ge3",
        ]:
            if c not in out.columns:
                out[c] = np.nan
    except Exception as _e:
        if bool(getattr(args, "debug", False)):
            try:
                print(f"ℹ️ lambda merge skipped: {_e}")
            except Exception:
                pass

    # Phase 8A: compute grid features on the fully merged output pool as a final
    # fixture-level pass. This avoids relying on upstream carry maps.
    try:
        out = _attach_phase8a_grid_features(out, max_goals=6)
    except Exception:
        pass
    # Merge in any captured rate columns (by league + fixture_key) so the output CSV
    # contains them even if they are all-NaN for now.
    try:
        if rate_maps:
            rate_maps = [x for x in rate_maps if isinstance(x, pd.DataFrame) and (not x.empty)]
            # Avoid pandas FutureWarning: concat of empty/all-NA frames
            _rms = [
                f for f in (rate_maps or [])
                if isinstance(f, pd.DataFrame) and (not f.empty) and (not f.isna().all().all())
            ]
            _rms_clean = []
            for _x in (_rms or []):
                if not isinstance(_x, pd.DataFrame):
                    continue
                if _x.empty:
                    continue
                # prevent concat warning: drop all-NA columns inside each fragment
                _x2 = _x.dropna(axis=1, how="all")
                if _x2.empty:
                    continue
                _rms_clean.append(_x2)

            rm = pd.concat(_rms_clean, ignore_index=True) if _rms_clean else pd.DataFrame()
            # ------------------------------------------------------------------
            # Export fixups: ensure canonical side probs exist (OU25 no-vig + prob_* aliases)
            # ------------------------------------------------------------------

            # Apply export fixups early on the concatenated pool
            try:
                rm = _ensure_export_side_probs(rm)
            except Exception:
                pass
            
            rm["league"] = rm["league"].astype("string").fillna("").str.strip()
            rm["fixture_key"] = rm["fixture_key"].astype("string").fillna("").str.strip()

            out["league"] = out["league"].astype("string").fillna("").str.strip()
            out["fixture_key"] = out["fixture_key"].astype("string").fillna("").str.strip()

            out = out.merge(rm, on=["league", "fixture_key"], how="left", suffixes=("", "_rm"))

            for c in (
                "scored_rate_5_home", "scored_rate_5_away",
                "clean_sheet_rate_5_home", "clean_sheet_rate_5_away",
                "conceded_rate_5_home", "conceded_rate_5_away",
                "btts_rate_5_home", "btts_rate_5_away",
                "over25_rate_5_home", "over25_rate_5_away",
                "under25_rate_5_home", "under25_rate_5_away",
                "goaliness_avg_5_home", "goaliness_avg_5_away",
                "xg_for_avg_5_home", "xg_for_avg_5_away",
                "xg_against_avg_5_home", "xg_against_avg_5_away",
                "home_ge2_confidence", "away_ge2_confidence",
                "home_ge3_confidence", "away_ge3_confidence",
                "p_home_fts", "p_away_fts",
                "h2h_n", "h2h_btts_rate", "h2h_over25_rate", "h2h_goaliness_avg",
                # Phase 8A grid features
                "cs_mass_btts_yes", "cs_mass_btts_no",
                "cs_mass_over25", "cs_mass_under25",
                "cs_mass_home_win", "cs_mass_draw", "cs_mass_away_win",
                "cs_entropy", "both_teams_2plus_mass",
                "mass_over25_via_one_sided_rout",
                "mass_0_goals", "mass_1_goal", "mass_2_goals", "mass_3_goals", "mass_4plus_goals",
                # Phase 8B coherence features
                "grid_vs_cat_btts_gap", "grid_vs_xgb_btts_gap",
                "grid_vs_cat_ou25_gap", "grid_vs_xgb_ou25_gap",
                "grid_vs_cat_ftr_gap", "grid_vs_xgb_ftr_gap",
                "cat_xgb_grid_btts_agreement_count",
                "cat_xgb_grid_ou25_agreement_count",
                "cat_xgb_grid_ftr_agreement_count",
                # UEFA context (carried via rate_maps)
                "uefa_home_state", "uefa_away_state",
                "uefa_home_gap24", "uefa_away_gap24", "uefa_gap24_diff",
                "uefa_home_rotation_risk", "uefa_away_rotation_risk",
                "uefa_both_must_win", "uefa_goal_hunt_flag", "uefa_pride_only_flag",
                "uefa_live_table_volatility", "uefa_vol_band_n",
                "uefa_home_must_win", "uefa_away_must_win",
                "uefa_home_must_avoid_loss", "uefa_away_must_avoid_loss",
                "uefa_home_eliminated", "uefa_away_eliminated",
                "uefa_home_state", "uefa_away_state",
            ):
                if (c not in out.columns) and (c + "_rm" in out.columns):
                    out[c] = out[c + "_rm"]
                elif (c in out.columns) and (c + "_rm" in out.columns):
                    # Special-case string state columns (numeric coercion would wipe them)
                    if c in ("uefa_home_state", "uefa_away_state"):
                        a = out[c].astype("string").fillna("").str.strip()
                        b = out[c + "_rm"].astype("string").fillna("").str.strip()
                        out[c] = a.where(a.ne(""), b)
                    else:
                        a = pd.to_numeric(out[c], errors="coerce")
                        b = pd.to_numeric(out[c + "_rm"], errors="coerce")
                        out[c] = a.where(a.notna(), b)
                if c + "_rm" in out.columns:
                    out = out.drop(columns=[c + "_rm"])
    except Exception as _e:
        print(f"ℹ️ rate merge skipped: {_e}")

    # Merge in UEFA context columns (by league + fixture_key) so outputs can use table-pressure logic.
    try:
        if uefa_maps:
            um = pd.concat(uefa_maps, ignore_index=True)
            um["league"] = um["league"].astype("string").fillna("").str.strip()
            um["fixture_key"] = um["fixture_key"].astype("string").fillna("").str.strip()

            out["league"] = out["league"].astype("string").fillna("").str.strip()
            out["fixture_key"] = out["fixture_key"].astype("string").fillna("").str.strip()

            out = out.merge(um, on=["league", "fixture_key"], how="left", suffixes=("", "_uefa"))

            # Coalesce any duplicated columns defensively
            for c in list(um.columns):
                if c in ("league", "fixture_key"):
                    continue
                cu = c + "_uefa"
                if cu in out.columns:
                    if c in out.columns:
                        try:
                            a = out[c]
                            b = out[cu]
                            if pd.api.types.is_numeric_dtype(b):
                                out[c] = pd.to_numeric(a, errors="coerce").where(
                                    pd.to_numeric(a, errors="coerce").notna(),
                                    pd.to_numeric(b, errors="coerce")
                                )
                            else:
                                out[c] = a.astype("string").where(
                                    a.astype("string").notna(),
                                    b.astype("string")
                                )
                        except Exception:
                            pass
                    else:
                        out[c] = out[cu]
                    out = out.drop(columns=[cu], errors="ignore")
    except Exception as _ue:
        try:
            print("ℹ️ UEFA context merge skipped: " + str(_ue))
        except Exception:
            pass
    # ------------------------------------------------------------------
    # Derived UEFA aggregates (match-level)
    # ------------------------------------------------------------------
    try:
        # rotation_any = max(home_rotation_risk, away_rotation_risk)
        if "uefa_home_rotation_risk" in out.columns:
            hr = pd.to_numeric(out["uefa_home_rotation_risk"], errors="coerce").fillna(0).astype(int)
        else:
            hr = pd.Series(0, index=out.index, dtype=int)

        if "uefa_away_rotation_risk" in out.columns:
            ar = pd.to_numeric(out["uefa_away_rotation_risk"], errors="coerce").fillna(0).astype(int)
        else:
            ar = pd.Series(0, index=out.index, dtype=int)

        out["uefa_rotation_any"] = np.maximum(hr, ar).astype(int)
        out["uefa_rotation_both"] = (hr.astype(int) & ar.astype(int)).astype(int)

        # pressure_sum = home_must_win + away_must_win + home_must_avoid_loss + away_must_avoid_loss
        if "uefa_home_must_win" in out.columns:
            hmw = pd.to_numeric(out["uefa_home_must_win"], errors="coerce").fillna(0).astype(int)
        else:
            hmw = pd.Series(0, index=out.index, dtype=int)

        if "uefa_away_must_win" in out.columns:
            amw = pd.to_numeric(out["uefa_away_must_win"], errors="coerce").fillna(0).astype(int)
        else:
            amw = pd.Series(0, index=out.index, dtype=int)

        if "uefa_home_must_avoid_loss" in out.columns:
            hmal = pd.to_numeric(out["uefa_home_must_avoid_loss"], errors="coerce").fillna(0).astype(int)
        else:
            hmal = pd.Series(0, index=out.index, dtype=int)

        if "uefa_away_must_avoid_loss" in out.columns:
            amal = pd.to_numeric(out["uefa_away_must_avoid_loss"], errors="coerce").fillna(0).astype(int)
        else:
            amal = pd.Series(0, index=out.index, dtype=int)

        out["uefa_pressure_sum"] = (hmw + amw + hmal + amal).astype(int)

        # pressure_asym = abs(home_gap24 - away_gap24)
        if "uefa_home_gap24" in out.columns:
            hg = pd.to_numeric(out["uefa_home_gap24"], errors="coerce")
        else:
            hg = pd.Series(np.nan, index=out.index, dtype=float)

        if "uefa_away_gap24" in out.columns:
            ag = pd.to_numeric(out["uefa_away_gap24"], errors="coerce")
        else:
            ag = pd.Series(np.nan, index=out.index, dtype=float)

        out["uefa_pressure_asym"] = (hg - ag).abs()
    except Exception as _uefa_agg_e:
        try:
            print(f"ℹ️ UEFA aggregate features skipped: {_uefa_agg_e}")
        except Exception:
            pass
    # --- Draw/Chaos Risk flags (stamp before gap/score) ---
    try:
        out = _attach_draw_chaos_risk(out)
    except Exception as _e:
        if bool(getattr(args, "debug", False)):
            print(f"ℹ️ draw/chaos risk stamp skipped: {_e}")

    # gap = model - implied (handy for your tolerance filter)
    out["gap"] = pd.to_numeric(out["model_p_for_bookie"], errors="coerce") - pd.to_numeric(out["bookie_implied"], errors="coerce")

    # no-vig gap (preferred when available)
    out["bookie_implied_novig"] = pd.to_numeric(out.get("bookie_implied_novig"), errors="coerce")
    out["gap_novig"] = pd.to_numeric(out["model_p_for_bookie"], errors="coerce") - out["bookie_implied_novig"]

    # ------------------------------------------------------------------
    # Backfill bookie total-goals lambda-fit for OU25 rows when odds exist
    # but the upstream totals-fit did not run (e.g., missing in fd_odds_enriched).
    #
    # We fit a Poisson mean `mu` for total goals such that:
    #   P(Total Goals > 2.5) ~= P_over_novig
    # where P_over_novig is derived from the OVER/UNDER odds and the overround.
    #
    # This is intentionally lightweight (bisection) and only runs for OU25 rows
    # where `bookie_lambda_total_fit` is NaN AND valid odds exist.
    # ------------------------------------------------------------------
    def _poisson_cdf_2(mu: float) -> float:
        """P(X <= 2) for X ~ Poisson(mu)."""
        if not np.isfinite(mu) or mu <= 0:
            return float("nan")
        # e^-mu * (1 + mu + mu^2/2)
        try:
            return float(np.exp(-mu) * (1.0 + mu + (mu * mu) / 2.0))
        except Exception:
            return float("nan")

    def _fit_total_lambda_from_p_over(p_over: float, lo: float = 0.01, hi: float = 8.0, iters: int = 60) -> float:
        """Solve for mu such that 1 - P(Pois(mu) <= 2) ~= p_over via bisection."""
        if not np.isfinite(p_over):
            return float("nan")
        # keep within open interval to avoid pathological edges
        p_over = float(np.clip(p_over, 1e-6, 1.0 - 1e-6))

        def f(mu: float) -> float:
            pu = _poisson_cdf_2(mu)
            if not np.isfinite(pu):
                return float("nan")
            return (1.0 - pu) - p_over

        flo = f(lo)
        fhi = f(hi)
        if not (np.isfinite(flo) and np.isfinite(fhi)):
            return float("nan")
        # If not bracketed, expand hi a bit (rare), else give up.
        if flo * fhi > 0:
            for _ in range(6):
                hi *= 1.5
                fhi = f(hi)
                if np.isfinite(fhi) and flo * fhi <= 0:
                    break
            else:
                return float("nan")

        a, b = float(lo), float(hi)
        fa, fb = float(flo), float(fhi)
        for _ in range(int(iters)):
            m = 0.5 * (a + b)
            fm = f(m)
            if not np.isfinite(fm):
                return float("nan")
            if abs(fm) < 1e-10:
                return float(m)
            # Maintain bracket
            if fa * fm <= 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        return float(0.5 * (a + b))

    def _backfill_ou25_bookie_lambda_fit(df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
        """Backfill bookie_lambda_total_fit + bookie_goaliness_fit_ok for OU25 rows when possible."""
        out_df = df
        if not isinstance(out_df, pd.DataFrame) or out_df.empty:
            return out_df
        if "market" not in out_df.columns:
            return out_df

        m_ou = out_df["market"].astype("string").fillna("").str.lower().str.strip().eq("ou25")
        if not bool(m_ou.any()):
            return out_df

        # Ensure required columns exist
        if "bookie_lambda_total_fit" not in out_df.columns:
            out_df["bookie_lambda_total_fit"] = np.nan
        if "bookie_goaliness_fit_ok" not in out_df.columns:
            out_df["bookie_goaliness_fit_ok"] = 0

        # Need valid odds + overround
        oo = _to_decimal_odds(out_df.get("odds_ft_over25", np.nan))
        ou = _to_decimal_odds(out_df.get("odds_ft_under25", np.nan))
        ov = pd.to_numeric(out_df.get("ou25_overround", np.nan), errors="coerce")

        need = (
            m_ou
            & pd.to_numeric(out_df["bookie_lambda_total_fit"], errors="coerce").isna()
            & oo.notna() & ou.notna() & (oo > 1.0) & (ou > 1.0)
            & ov.notna() & (ov > 0)
        )

        # Even if we don't need to backfill missing rows, we may still want to
        # rescale absurdly-low model lambdas to the trusted bookie total.
        # So we do NOT early-return here.
        if not bool(need.any()):
            need = need  # no-op

        # Compute no-vig probabilities from the odds pair
        imp_over = (1.0 / oo).where(oo > 1.0)
        imp_under = (1.0 / ou).where(ou > 1.0)
        p_over_nv = (imp_over / ov).where(ov > 0)

        # Fit mu per row (bisection) – only on the needed subset
        idx = out_df.index[need]
        mus = []
        for i in idx:
            mu = _fit_total_lambda_from_p_over(float(p_over_nv.loc[i]))
            mus.append(mu)

        mu_s = pd.Series(mus, index=idx, dtype=float)
        ok = mu_s.notna() & np.isfinite(mu_s)

        # Pre-coerce flag dtype BEFORE assignment (avoids pandas dtype FutureWarning)
        out_df["bookie_goaliness_fit_ok"] = (
            pd.to_numeric(out_df.get("bookie_goaliness_fit_ok", 0), errors="coerce")
            .fillna(0)
            .astype(int)
        )

        # Stamp results
        idx_ok = idx[ok.to_numpy()]
        if len(idx_ok):
            out_df.loc[idx_ok, "bookie_lambda_total_fit"] = mu_s.loc[idx_ok].astype(float)
            out_df.loc[idx_ok, "bookie_goaliness_fit_ok"] = 1

        if debug:
            try:
                n_try = int(len(idx))
                n_ok = int(ok.sum())
                print(f"[OU25 LAMBDA BACKFILL] attempted={n_try} solved={n_ok} (from odds_ft_* + ou25_overround)")
            except Exception:
                pass

        rescaled_mask = pd.Series(False, index=out_df.index)

        # --------------------------------------------------------------
        # Trusted-total rescale (fix absurdly-low model lambdas)
        # --------------------------------------------------------------
        try:
            # Only operate on OU25 rows where a sane trusted total exists
            bk = pd.to_numeric(out_df.get("bookie_lambda_total_fit", np.nan), errors="coerce")
            bk_sane = bk.where(bk.between(1.0, 6.0))

            # Prefer lambda_* if present; otherwise fall back to home/away_goals_pred.
            # (At this stage in bookie_allmarkets, fixture-level λ may not be merged yet.)
            lh = pd.to_numeric(out_df.get("lambda_home", out_df.get("home_goals_pred", np.nan)), errors="coerce")
            la = pd.to_numeric(out_df.get("lambda_away", out_df.get("away_goals_pred", np.nan)), errors="coerce")
            lam_sum = lh + la

            # Gate: only rescale when the model total is clearly broken
            ratio = (lam_sum / bk_sane)
            m_rescale = m_ou & bk_sane.notna() & (
                (lam_sum < 1.0) | (ratio < 0.60)
            )

            if bool(m_rescale.any()):
                rescaled_mask = rescaled_mask | m_rescale.fillna(False)
                # Shares (handle divide-by-zero / missing)
                denom = lam_sum.where(lam_sum > 1e-9)
                share_h = (lh / denom).clip(lower=0.0, upper=1.0)
                share_a = (la / denom).clip(lower=0.0, upper=1.0)

                # If denom is missing/zero for a row, default to 50/50 split
                share_h = share_h.where(denom.notna(), 0.5)
                share_a = share_a.where(denom.notna(), 0.5)

                # Normalize shares so they sum to 1 (guarding drift)
                share_sum = (share_h + share_a).where((share_h + share_a) > 1e-9)
                share_h = (share_h / share_sum).where(share_sum.notna(), 0.5)
                share_a = (share_a / share_sum).where(share_sum.notna(), 0.5)

                new_lh = (bk_sane * share_h).clip(lower=0.0)
                new_la = (bk_sane * share_a).clip(lower=0.0)

                out_df.loc[m_rescale, "lambda_home"] = new_lh.loc[m_rescale].astype(float)
                out_df.loc[m_rescale, "lambda_away"] = new_la.loc[m_rescale].astype(float)

                # Keep goal-pred aliases consistent (used by Poisson + CS helpers)
                if "home_goals_pred" in out_df.columns:
                    out_df.loc[m_rescale, "home_goals_pred"] = pd.to_numeric(
                        out_df.loc[m_rescale, "lambda_home"], errors="coerce"
                    )
                if "away_goals_pred" in out_df.columns:
                    out_df.loc[m_rescale, "away_goals_pred"] = pd.to_numeric(
                        out_df.loc[m_rescale, "lambda_away"], errors="coerce"
                    )

                # By construction, the rescaled total should equal bk_sane on these rows.
                out_df.loc[m_rescale, "exp_goals_sum"] = bk_sane.loc[m_rescale].astype(float)

                out_df.loc[m_rescale, "p00_est"] = np.exp(
                    -pd.to_numeric(out_df.loc[m_rescale, "exp_goals_sum"], errors="coerce").clip(lower=0.0)
                )

                if debug:
                    try:
                        n_rs = int(m_rescale.sum())
                        before_ratio = pd.to_numeric(ratio.loc[m_rescale], errors="coerce")
                        after_sum = (
                            pd.to_numeric(out_df.loc[m_rescale, "lambda_home"], errors="coerce")
                            + pd.to_numeric(out_df.loc[m_rescale, "lambda_away"], errors="coerce")
                        )
                        after_ratio = (after_sum / bk_sane.loc[m_rescale]).astype(float)

                        print(
                            "[OU25 LAMBDA RESCALE]",
                            {
                                "rows": n_rs,
                                "ratio_before_min/med/max": (
                                    float(before_ratio.min()),
                                    float(before_ratio.median()),
                                    float(before_ratio.max()),
                                ),
                                "ratio_after_min/med/max": (
                                    float(after_ratio.min()),
                                    float(after_ratio.median()),
                                    float(after_ratio.max()),
                                ),
                            },
                        )
                    except Exception:
                        pass
        except Exception:
            # Never break generation due to rescale logic
            pass

        # --------------------------------------------------------------
        # Post-backfill trusted-total rescale (OU25):
        # At this stage `bookie_lambda_total_fit` is guaranteed for OU25 rows.
        # If model λ totals still disagree meaningfully with the trusted bookie total,
        # rescale λ_home/λ_away by the model shares.
        # Gate:
        #   - lam_sum < 1.0 OR ratio < 0.80 OR ratio > 1.20
        # Only uses bookie totals when they look sane (1.0–6.0).
        # --------------------------------------------------------------
        try:
            bk = pd.to_numeric(out_df.get("bookie_lambda_total_fit", np.nan), errors="coerce")
            bk_sane = bk.where(bk.between(1.0, 6.0))

            lh = pd.to_numeric(out_df.get("lambda_home", np.nan), errors="coerce")
            la = pd.to_numeric(out_df.get("lambda_away", np.nan), errors="coerce")
            lam_sum = lh + la

            ratio = (lam_sum / bk_sane)

            m_rescale2 = m_ou & bk_sane.notna() & (
                (lam_sum < 1.0)
                | (ratio < 0.80)
                | (ratio > 1.20)
            )

            if bool(m_rescale2.any()):
                rescaled_mask = rescaled_mask | m_rescale2.fillna(False)
                denom = lam_sum.where(lam_sum > 1e-9)
                share_h = (lh / denom).clip(lower=0.0, upper=1.0)
                share_a = (la / denom).clip(lower=0.0, upper=1.0)

                # If denom is missing/zero for a row, default to 50/50 split
                share_h = share_h.where(denom.notna(), 0.5)
                share_a = share_a.where(denom.notna(), 0.5)

                # Normalize shares so they sum to 1
                share_sum = (share_h + share_a).where((share_h + share_a) > 1e-9)
                share_h = (share_h / share_sum).where(share_sum.notna(), 0.5)
                share_a = (share_a / share_sum).where(share_sum.notna(), 0.5)

                new_lh = (bk_sane * share_h).clip(lower=0.0)
                new_la = (bk_sane * share_a).clip(lower=0.0)

                out_df.loc[m_rescale2, "lambda_home"] = new_lh.loc[m_rescale2].astype(float)
                out_df.loc[m_rescale2, "lambda_away"] = new_la.loc[m_rescale2].astype(float)

                # Keep goal-pred aliases consistent
                if "home_goals_pred" in out_df.columns:
                    out_df.loc[m_rescale2, "home_goals_pred"] = pd.to_numeric(
                        out_df.loc[m_rescale2, "lambda_home"], errors="coerce"
                    )
                if "away_goals_pred" in out_df.columns:
                    out_df.loc[m_rescale2, "away_goals_pred"] = pd.to_numeric(
                        out_df.loc[m_rescale2, "lambda_away"], errors="coerce"
                    )

                # Keep exp_goals_sum / p00_est consistent
                if "exp_goals_sum" in out_df.columns:
                    out_df.loc[m_rescale2, "exp_goals_sum"] = (
                        pd.to_numeric(out_df.loc[m_rescale2, "lambda_home"], errors="coerce")
                        + pd.to_numeric(out_df.loc[m_rescale2, "lambda_away"], errors="coerce")
                    )
                if "p00_est" in out_df.columns:
                    out_df.loc[m_rescale2, "p00_est"] = np.exp(
                        -pd.to_numeric(out_df.loc[m_rescale2, "exp_goals_sum"], errors="coerce").clip(lower=0.0)
                    )

                if debug:
                    try:
                        before_ratio = pd.to_numeric(ratio.loc[m_rescale2], errors="coerce")
                        after_sum = (
                            pd.to_numeric(out_df.loc[m_rescale2, "lambda_home"], errors="coerce")
                            + pd.to_numeric(out_df.loc[m_rescale2, "lambda_away"], errors="coerce")
                        )
                        after_ratio = (after_sum / bk_sane.loc[m_rescale2]).astype(float)
                        print(
                            "[OU25 LAMBDA RESCALE POST]",
                            {
                                "rows": int(m_rescale2.sum()),
                                "ratio_before_min/med/max": (
                                    float(before_ratio.min()),
                                    float(before_ratio.median()),
                                    float(before_ratio.max()),
                                ),
                                "ratio_after_min/med/max": (
                                    float(after_ratio.min()),
                                    float(after_ratio.median()),
                                    float(after_ratio.max()),
                                ),
                            },
                        )
                    except Exception:
                        pass
        except Exception:
            # Never break generation due to rescale logic
            pass

                # --------------------------------------------------------------
        # Reconcile fixture-level lambdas from repaired OU25 rows.
        # This propagates sane OU25 goal environments back onto clearly-broken
        # FTR / BTTS sibling rows for the same fixture, then refreshes CS/tails.
        # --------------------------------------------------------------
        try:
            out_df = _reconcile_fixture_lambdas_from_ou25(
                out_df,
                debug=bool(debug),
                ou_market="ou25",
                sane_total_min=2.0,
                broken_total_max=0.75,
                broken_ratio_max=0.35,
            )
        except Exception:
            # Never break generation due to fixture lambda reconciliation
            pass

        # --------------------------------------------------------------
        # Refresh logging after any late OU25 rescale.
        # The reconciliation helper already refreshes Poisson CS / tails.
        # --------------------------------------------------------------
        try:
            if bool(rescaled_mask.fillna(False).any()):
                if debug:
                    try:
                        print(
                            "[OU25 CS/TAILS REFRESH]",
                            {"rows": int(rescaled_mask.fillna(False).sum())},
                        )
                    except Exception:
                        pass
        except Exception:
            # Never break generation due to CS/tails refresh logging
            pass
        return out_df

    # --- Coverage stats: no-vig availability (tells you if under/no odds exist) ---
    try:
        # Align coverage diagnostics with export fixups (OU25 no-vig + canonical prob columns)
        out = _ensure_export_side_probs(out)
        out = _backfill_ou25_bookie_lambda_fit(out, debug=bool(getattr(args, "debug", False)))
        cov = out.copy()
        cov["_has_novig"] = cov["bookie_implied_novig"].notna()
        # Overround coverage is market-aware:
        #   - OU25 uses ou25_overround
        #   - other priced markets use bookie_overround
        _mkt = cov.get("market", "").astype(str).str.lower().str.strip()
        cov["_has_overround"] = np.where(
            _mkt.eq("ou25"),
            pd.to_numeric(cov.get("ou25_overround", np.nan), errors="coerce").notna(),
            pd.to_numeric(cov.get("bookie_overround", np.nan), errors="coerce").notna(),
        )

        # Bookie-derived total-goals lambda fit (from totals odds) — may be absent for FTR.
        cov["_has_bookie_lambda_fit"] = pd.to_numeric(
            cov.get("bookie_lambda_total_fit", np.nan),
            errors="coerce"
        ).notna()

        # Model-derived lambda presence (from goal engine) — should exist for all markets after lambda_maps merge.
        lh = pd.to_numeric(cov.get("lambda_home", np.nan), errors="coerce")
        la = pd.to_numeric(cov.get("lambda_away", np.nan), errors="coerce")
        eg = pd.to_numeric(cov.get("exp_goals_sum", np.nan), errors="coerce")
        cov["_has_model_lambda"] = eg.notna() | (lh.notna() & la.notna())

        cov_by_mkt = (
            cov.groupby("market", dropna=False)
            .agg(
                n=("market", "size"),
                novig_n=("_has_novig", "sum"),
                novig_rate=("_has_novig", "mean"),
                overround_n=("_has_overround", "sum"),
                overround_rate=("_has_overround", "mean"),
                bookie_lambda_fit_n=("_has_bookie_lambda_fit", "sum"),
                bookie_lambda_fit_rate=("_has_bookie_lambda_fit", "mean"),
                model_lambda_n=("_has_model_lambda", "sum"),
                model_lambda_rate=("_has_model_lambda", "mean"),
            )
            .reset_index()
            .sort_values(["market"], ascending=True)
        )

        # TG markets are model-only (no bookie odds), so blank out bookie coverage fields
        try:
            _m = cov_by_mkt["market"].astype(str).str.lower().str.strip()
            _is_tg = _m.isin(["tg15", "tg25", "tg_15", "tg_25"]) | _m.str.startswith("tg")
            for _c in [
                "novig_n", "novig_rate",
                "overround_n", "overround_rate",
                "bookie_lambda_fit_n", "bookie_lambda_fit_rate",
            ]:
                if _c in cov_by_mkt.columns:
                    cov_by_mkt.loc[_is_tg, _c] = np.nan
        except Exception:
            pass
        
        # --- EXPORT DIAG: inspect final export columns (not rate-map helper frames) ---
        try:
            _cols_export = list(out.columns) if isinstance(out, pd.DataFrame) else []
        except Exception:
            _cols_export = []

        print("\n[EXPORT DIAG] odds_ft_* present:", [c for c in _cols_export if c.startswith("odds_ft_")])
        print("[EXPORT DIAG] odds_btts_* present:", [c for c in _cols_export if c.startswith("odds_btts_")])
        print("[EXPORT DIAG] sample cols:", sorted([c for c in _cols_export if "odds" in str(c).lower()])[:80])
        print("\n🧾 Coverage by market (bookie no-vig where applicable + model lambdas):")
        print("Note: TG markets (tg15/tg25) are model-only; bookie no-vig/overround/lambda-fit are not expected.")
        print(cov_by_mkt.to_string(index=False))

        # If any market has low no-vig coverage, show where it's missing (league breakdown)
        low = cov_by_mkt[cov_by_mkt["novig_rate"] < 0.90]
        # TG markets are model-only (no odds), so no-vig is missing by design
        low = low[~low["market"].astype(str).str.lower().isin(["tg15", "tg25"])]
        if len(low):
            miss = cov[(~cov["_has_novig"])& (~cov["market"].astype(str).str.lower().isin(["tg15", "tg25"]))].copy()
            if not miss.empty:
                miss_by = (
                    miss.groupby(["market", "league"], dropna=False)
                    .size()
                    .reset_index(name="missing_n")
                    .sort_values(["market", "missing_n"], ascending=[True, False])
                )
                print("\n⚠️ No-vig missing (top league contributors):")
                print(miss_by.head(30).to_string(index=False))
    except Exception as _e:
        print(f"ℹ️ No-vig coverage stats skipped: {_e}")

    # Ensure key downstream columns always exist (bulk add; avoids fragmentation)
    _required = [
        # (keep your full list exactly as you already have it)
        "model_strength",
        "ftr_margin",
        "home_power_rating",
        "away_power_rating",
        "power_diff",
        "bookie_overround",
        "bookie_implied_novig",
        "bookie_spread",
        "gap_novig",
        "bookie_lambda_total_fit",
        "bookie_goaliness_fit_ok",
        "average_goals_per_match_pre_match",
        "pre_match_xg_home",
        "pre_match_xg_away",
        "xg_sum_pre_match",
        "over_25_percentage_pre_match",
        "snap_timing_early_goal_pressure",
        "snap_timing_second_half_acceleration",
        "snap_home_first_to_score_edge",
        "snap_ht_goal_regime_blend",
        "snap_ou25_over_regime_blend",
        "ppg_home_pre",
        "ppg_away_pre",
        "ppg_diff_pre",
        # Specialist head probabilities (always exist)
        "home_ge2_confidence",
        "away_ge2_confidence",
        "home_ge3_confidence",
        "away_ge3_confidence",
        "p_home_fts",
        "p_away_fts",
        # Leak-safe rolling team rates (if available)
        "scored_rate_5_home",
        "scored_rate_5_away",
        "clean_sheet_rate_5_home",
        "clean_sheet_rate_5_away",
        "conceded_rate_5_home",
        "conceded_rate_5_away",
        "btts_rate_5_home",
        "btts_rate_5_away",
        "over25_rate_5_home",
        "over25_rate_5_away",
        "under25_rate_5_home",
        "under25_rate_5_away",
        "goaliness_avg_5_home",
        "goaliness_avg_5_away",
        "xg_for_avg_5_home",
        "xg_for_avg_5_away",
        "xg_against_avg_5_home",
        "xg_against_avg_5_away",
        "h2h_n",
        "h2h_btts_rate",
        "h2h_over25_rate",
        "h2h_goaliness_avg",
        # 1X2 odds and implieds for slip-builder/EV logic
        "od_home",
        "od_draw",
        "od_away",
        "imp_home",
        "imp_draw",
        "imp_away",
        # Side-market odds (schema stability)
        "od_over",
        "od_under",
        "od_yes",
        "od_no",
        # Canonical odds columns (synth/bookie)
        "odds_ft_over25",
        "odds_ft_under25",
        "odds_btts_yes",
        "odds_btts_no",
        # Draw/chaos + routing fields
        "ftr_ppg_glue_ok",
        "ftr_drawtrap_flag",
        "draw_risk_flag",
        "not_glue_flag",
        "draw_chaos_score",
        "chaos_risk_flag",
        "selection",
        "product",
        "model_lane",
        "source_prob_col",
        "ou25_policy_mode",
        "ou25_policy_branch",
        "ou25_policy_state",
        "ou25_shadow_mode",
        "ou25_shadow_model",
        "ou25_runtime_lane",
        "ou25_is_shadow",
        "ou25_is_premium_candidate",
        "pool_tier",
        "od_source",
        "is_fixture_primary",
        "is_market_primary",
        "is_candidate",
        # Lambda / Poisson / CS core
        "home_goals_pred",
        "away_goals_pred",
        "lambda_home",
        "lambda_away",
        "exp_goals_sum",
        "p00_est",
        "cs_home",
        "cs_away",
        "p_home_ge1",
        "p_away_ge1",
        "p_home_ge2",
        "p_away_ge2",
        "p_home_ge3",
        "p_away_ge3",
        "p_home_ge4",
        "p_away_ge4",
        "cs1",
        "cs1_p",
        "cs2",
        "cs2_p",
        "cs3",
        "cs3_p",
        "p_home_pois",
        "p_draw_pois",
        "p_away_pois",
        "cs_trunc_mass_0_6",
        # Backward-compatible Poisson alias fields
        "pois_home_ge2",
        "pois_away_ge2",
        "pois_home_ge3",
        "pois_away_ge3",
        "close_match_flag",
        "candidate_rank",
        "xg_diff_abs",
        "implied_prob_diff",
        "odds_diff",
        "p_pick",
        "edge",
        "btts_fh_confidence",
        "btts_fh_pred",
        # UEFA context
        "uefa_home_state",
        "uefa_away_state",
        "uefa_home_gap24",
        "uefa_away_gap24",
        "uefa_gap24_diff",
        "uefa_home_rotation_risk",
        "uefa_away_rotation_risk",
        "uefa_both_must_win",
        "uefa_goal_hunt_flag",
        "uefa_pride_only_flag",
        "uefa_live_table_volatility",
        "uefa_vol_band_n",
        "uefa_rotation_any",
        "uefa_rotation_both",
        "uefa_pressure_sum",
        "uefa_pressure_asym",
        "uefa_home_must_win",
        "uefa_away_must_win",
        "uefa_home_must_avoid_loss",
        "uefa_away_must_avoid_loss",
        "uefa_home_eliminated",
        "uefa_away_eliminated",
    ]

    # Remove duplicates while preserving first occurrence order.
    seen = set()
    _required = [x for x in _required if not (x in seen or seen.add(x))]

    _missing = [c for c in _required if c not in out.columns]
    if _missing:
        out = out.reindex(columns=list(out.columns) + _missing)

    # Stamp stable BTTS lane identity for downstream deploy routing.
    try:
        m_bt = out.get("market", "").astype("string").fillna("").str.lower().str.strip().eq("btts")
        if bool(m_bt.any()):
            if "product" not in out.columns:
                out["product"] = pd.NA
            if "model_lane" not in out.columns:
                out["model_lane"] = pd.NA
            if "source_prob_col" not in out.columns:
                out["source_prob_col"] = pd.NA

            prod = out["product"].astype("string").fillna("").str.strip()
            lane = out["model_lane"].astype("string").fillna("").str.strip()
            spc = out["source_prob_col"].astype("string").fillna("").str.strip()

            out.loc[m_bt & prod.eq(""), "product"] = "BTTS_MODEL"
            out.loc[m_bt & lane.eq(""), "model_lane"] = "btts_model"
            out.loc[m_bt & spc.eq(""), "source_prob_col"] = "p_pick"

            if "selection" in out.columns:
                sel_bt = out["selection"].astype("string").fillna("").str.strip().str.upper()
                out.loc[m_bt & sel_bt.isin(["YES", "NO"]), "bookie_pick"] = sel_bt.loc[m_bt & sel_bt.isin(["YES", "NO"])]
    except Exception:
        pass

    # Stamp stable OU25 policy identity for downstream deploy routing.
    try:
        m_ou = out.get("market", "").astype("string").fillna("").str.lower().str.strip().eq("ou25")
        if bool(m_ou.any()):
            for _c in [
                "ou25_policy_mode",
                "ou25_policy_branch",
                "ou25_policy_state",
                "ou25_shadow_mode",
                "ou25_shadow_model",
                "ou25_runtime_lane",
            ]:
                if _c not in out.columns:
                    out[_c] = pd.Series(pd.NA, index=out.index, dtype="string")
                else:
                    out[_c] = out[_c].astype("string")

            for _c in ["ou25_is_shadow", "ou25_is_premium_candidate"]:
                if _c not in out.columns:
                    out[_c] = 0

            pick_ou = out.get("bookie_pick", "").astype("string").fillna("").str.upper().str.strip()
            lg_ou = out.get("league", "").astype("string").fillna("").str.strip()
            lane_ou = out.get("model_lane", "").astype("string").fillna("").str.strip().str.lower()
            product_ou = out.get("product", "").astype("string").fillna("").str.strip().str.upper()
            sig_ou = out.get("signal_over25", "").astype("string").fillna("").str.upper().str.strip()
            pm_ou = pd.to_numeric(out.get("model_p_for_bookie", np.nan), errors="coerce")

            is_over = m_ou & pick_ou.eq("OVER25")
            is_under = m_ou & pick_ou.eq("UNDER25")
            is_epl = m_ou & lg_ou.eq("England Premier League")
            is_shadow = m_ou & (
                lane_ou.eq("dedicated_over25_model")
                | product_ou.eq("OVER25_MODEL")
                | product_ou.eq("OU25_MODEL")
            )

            premium_p = (pm_ou >= 0.60).fillna(False)
            premium_mask = (is_over & (~is_shadow) & premium_p).fillna(False)

            out.loc[m_ou, "ou25_policy_mode"] = "observe"
            out.loc[m_ou, "ou25_policy_branch"] = "ou25_band1_124_176"
            out.loc[m_ou, "ou25_policy_state"] = "live"
            out.loc[m_ou, "ou25_shadow_mode"] = "dedicated_over25_model"
            out.loc[m_ou, "ou25_shadow_model"] = "over25_v3"
            out.loc[m_ou, "ou25_runtime_lane"] = "ou25_observe"
            out.loc[m_ou, "ou25_is_shadow"] = 0
            out.loc[m_ou, "ou25_is_premium_candidate"] = 0

            out.loc[is_over & (~is_shadow), "ou25_policy_mode"] = "filtered_ou25_over"
            out.loc[is_over & (~is_shadow), "ou25_policy_branch"] = "ou25_band1_124_176"
            out.loc[is_over & (~is_shadow), "ou25_runtime_lane"] = "ou25_live"

            try:
                _dbg = pd.DataFrame({
                    "league": lg_ou,
                    "bookie_pick": pick_ou,
                    "signal_over25": sig_ou,
                    "model_p_for_bookie": pm_ou,
                    "is_over": is_over,
                    "is_shadow": is_shadow,
                    "premium_p": premium_p,
                    "premium_mask": premium_mask,
                })
                _dbg = _dbg.loc[m_ou].copy()
                if (not _dbg.empty) and bool(getattr(args, "debug", False)):
                    # NOTE: OU25_POLICY_DEBUG is printed later (post signal-restore + direction enforcement)
                    pass
            except Exception:
                pass
            out.loc[premium_mask, "ou25_policy_branch"] = "ou25_combined_topq_080"
            out.loc[premium_mask, "ou25_runtime_lane"] = "ou25_premium"
            out.loc[premium_mask, "ou25_is_premium_candidate"] = 1

            out.loc[is_under, "ou25_policy_mode"] = "observe"
            out.loc[is_under, "ou25_policy_branch"] = "ou25_under_watch"
            out.loc[is_under, "ou25_runtime_lane"] = "ou25_observe"

            out.loc[is_shadow, "ou25_policy_mode"] = "dedicated_over25_model"
            out.loc[is_shadow, "ou25_runtime_lane"] = "ou25_shadow"
            out.loc[is_shadow, "ou25_is_shadow"] = 1

            out.loc[is_epl, "ou25_policy_state"] = "review"
            out.loc[is_epl & (~is_shadow), "ou25_runtime_lane"] = "ou25_review"
    except Exception:
        pass
    # Normalize debug flags to 0/1 ints
    out["ftr_ppg_glue_ok"] = pd.to_numeric(out.get("ftr_ppg_glue_ok", 0), errors="coerce").fillna(0).astype(int)
    out["ftr_drawtrap_flag"] = pd.to_numeric(out.get("ftr_drawtrap_flag", 0), errors="coerce").fillna(0).astype(int)

    # Normalize CLOSE fields
    out["candidate_rank"] = pd.to_numeric(out.get("candidate_rank", 0), errors="coerce").fillna(0).astype(int)
    out["close_match_flag"] = pd.to_numeric(out.get("close_match_flag", 0), errors="coerce").fillna(0).astype(int)

    # Normalize draw/chaos risk flags (0/1 ints) + score
    out["draw_risk_flag"] = pd.to_numeric(out.get("draw_risk_flag", 0), errors="coerce").fillna(0).astype(int)
    out["not_glue_flag"] = pd.to_numeric(out.get("not_glue_flag", 0), errors="coerce").fillna(0).astype(int)
    out["chaos_risk_flag"] = pd.to_numeric(out.get("chaos_risk_flag", 0), errors="coerce").fillna(0).astype(int)
    out["draw_chaos_score"] = pd.to_numeric(out.get("draw_chaos_score", np.nan), errors="coerce")

    for _c in (
        "uefa_home_rotation_risk",
        "uefa_away_rotation_risk",
        "uefa_both_must_win",
        "uefa_goal_hunt_flag",
        "uefa_pride_only_flag",
        "uefa_rotation_any",
        "uefa_rotation_both",
        "uefa_pressure_sum",
        "uefa_home_must_win",
        "uefa_away_must_win",
        "uefa_home_must_avoid_loss",
        "uefa_away_must_avoid_loss",
        "uefa_home_eliminated",
        "uefa_away_eliminated",
    ):
        if _c in out.columns:
            out[_c] = pd.to_numeric(out[_c], errors="coerce").fillna(0).astype(int)

    # ------------------------------------------------------------------
    # Primary flags split:
    #   - is_candidate: 1 for ANCHOR_CAND rows (never primary)
    #   - is_market_primary: 1 row per (league, fixture_key, market) for GLUE rows
    #     where od_source is bookie_pick or model_only (TG is model_only)
    #   - is_fixture_primary: backward-compatible alias of is_market_primary
    # ------------------------------------------------------------------

    # Normalize pool identity columns (bulletproof defaults)
    out["pool_tier"] = out.get("pool_tier", "").astype("string").fillna("").str.strip()
    out.loc[out["pool_tier"].eq(""), "pool_tier"] = "GLUE"

    out["od_source"] = out.get("od_source", "").astype("string").fillna("").str.strip()
    blank = out["od_source"].eq("")
    out.loc[blank & out["pool_tier"].str.upper().eq("ANCHOR_CAND"), "od_source"] = "ftr_candidate"
    out.loc[blank, "od_source"] = "bookie_pick"

    # If a leg is unpriced, mark it model_only — market-aware (FTR != bookie_od)
    try:
        m = out.get("market", "").astype("string").fillna("").str.lower().str.strip()
        priced = pd.Series(False, index=out.index)

        # FTR priced if ANY 1X2 odds exist (decimal or fractional) and are > 1.0
        oh = _to_decimal_odds(out.get("od_home", np.nan))
        od = _to_decimal_odds(out.get("od_draw", np.nan))
        oa = _to_decimal_odds(out.get("od_away", np.nan))
        priced = priced | ((m == "ftr") & (oh.gt(1.0) | od.gt(1.0) | oa.gt(1.0)))

        # OU25 priced if totals odds exist (decimal or fractional)
        oo = _to_decimal_odds(out.get("odds_ft_over25", np.nan))
        ou = _to_decimal_odds(out.get("odds_ft_under25", np.nan))
        priced = priced | ((m == "ou25") & (oo.gt(1.0) | ou.gt(1.0)))

        # BTTS priced if btts odds exist (decimal or fractional)
        oy = _to_decimal_odds(out.get("odds_btts_yes", np.nan))
        on = _to_decimal_odds(out.get("odds_btts_no", np.nan))
        priced = priced | ((m == "btts") & (oy.gt(1.0) | on.gt(1.0)))

        # TG markets are always model_only
        out.loc[m.isin(["tg15", "tg25"]), "od_source"] = "model_only"

        # If a row is priced but currently labelled model_only, promote it to bookie_pick.
        # This fixes cases like FTR where odds live in od_home/od_draw/od_away (not bookie_od)
        # and earlier row-build steps may default od_source to model_only.
        try:
            od_source_s = out.get("od_source", "").astype("string").fillna("").str.lower().str.strip()
            pt_u = out.get("pool_tier", "").astype("string").fillna("").str.upper().str.strip()
            is_cand = pt_u.eq("ANCHOR_CAND") | os.eq("ftr_candidate")
            promote = priced & ~m.isin(["tg15", "tg25"]) & ~is_cand & os.eq("model_only")
            out.loc[promote, "od_source"] = "bookie_pick"
        except Exception:
            pass

        # Non-TG markets: only mark model_only when not priced
        out.loc[~priced & ~m.isin(["tg15", "tg25"]), "od_source"] = "model_only"

    except Exception:
        pass

    # Candidate flag
    try:
        out["is_candidate"] = out.get("pool_tier", "").astype(str).str.upper().eq("ANCHOR_CAND").astype(int)
    except Exception:
        out["is_candidate"] = 0
    # OU25 premium/live candidate override
    # Promote live premium OVER25 rows into candidate posture even when they are not
    # ANCHOR_CAND pool rows. This keeps premium filtered-over rows visible to downstream
    # deploy/tiering instead of leaving them as plain GLUE rows with is_candidate=0.
    try:
        mk = out.get("market", pd.Series("", index=out.index)).astype("string").fillna("").str.lower().str.strip()
        pick = out.get("bookie_pick", out.get("selection", pd.Series("", index=out.index))).astype("string").fillna("").str.upper().str.strip()
        state = out.get("ou25_policy_state", pd.Series("", index=out.index)).astype("string").fillna("").str.lower().str.strip()
        branch = out.get("ou25_policy_branch", pd.Series("", index=out.index)).astype("string").fillna("").str.strip()
        sig = out.get("signal_over25", pd.Series("", index=out.index)).astype("string").fillna("").str.upper().str.strip()

        sh = pd.to_numeric(out.get("ou25_is_shadow", 0), errors="coerce").fillna(0).astype(int)
        prem = pd.to_numeric(out.get("ou25_is_premium_candidate", 0), errors="coerce").fillna(0).astype(int)

        m_ou25_premium_live = (
            mk.eq("ou25")
            & pick.eq("OVER25")
            & state.eq("live")
            & sh.eq(0)
            & (
                prem.ge(1)
                | branch.eq("ou25_combined_topq_080")
            )
            & sig.isin(["STRONG_OVER", "VERY_STRONG_OVER"])
        )

        if bool(m_ou25_premium_live.any()):
            out.loc[m_ou25_premium_live, "is_candidate"] = 1

            if "candidate_rank" not in out.columns:
                out["candidate_rank"] = 0

            rk = pd.to_numeric(out.get("candidate_rank", 0), errors="coerce").fillna(0).astype(int)
            out.loc[m_ou25_premium_live & rk.le(0), "candidate_rank"] = 1
    except Exception:
        pass
    # Initial market-primary flag (GLUE rows only; allow bookie_pick + model_only)
    try:
        pt = out.get("pool_tier", "").astype(str).str.upper().str.strip()
        osrc = out.get("od_source", "").astype(str).str.lower().str.strip()
        out["is_market_primary"] = (
            pt.eq("GLUE")
            & osrc.isin(["bookie_pick", "model_only"])
            & (pd.to_numeric(out.get("is_candidate", 0), errors="coerce").fillna(0).astype(int) == 0)
        ).astype(int)
    except Exception:
        out["is_market_primary"] = 0

    # Explicit premium OU25 candidate identity:
    # premium live OVER25 rows must always be candidate rows, never market-primary.
    try:
        mk = out.get("market", pd.Series("", index=out.index)).astype("string").fillna("").str.lower().str.strip()
        pick = out.get("bookie_pick", out.get("selection", pd.Series("", index=out.index))).astype("string").fillna("").str.upper().str.strip()
        branch = out.get("ou25_policy_branch", pd.Series("", index=out.index)).astype("string").fillna("").str.strip()
        state = out.get("ou25_policy_state", pd.Series("", index=out.index)).astype("string").fillna("").str.lower().str.strip()

        shadow = pd.to_numeric(out.get("ou25_is_shadow", 0), errors="coerce").fillna(0).astype(int)
        premium = pd.to_numeric(out.get("ou25_is_premium_candidate", 0), errors="coerce").fillna(0).astype(int)

        premium_live_ou25 = (
            mk.eq("ou25")
            & pick.eq("OVER25")
            & shadow.eq(0)
            & (~state.eq("review"))
            & (
                branch.eq("ou25_combined_topq_080")
                | premium.ge(1)
            )
        )

        if bool(premium_live_ou25.any()):
            out.loc[premium_live_ou25, "is_candidate"] = 1

            if "candidate_rank" not in out.columns:
                out["candidate_rank"] = 0
            rk = pd.to_numeric(out.get("candidate_rank", 0), errors="coerce").fillna(0).astype(int)
            out.loc[premium_live_ou25, "candidate_rank"] = np.where(
                rk.loc[premium_live_ou25].gt(0),
                rk.loc[premium_live_ou25],
                1,
            )

            if "is_market_primary" not in out.columns:
                out["is_market_primary"] = 0
            out.loc[premium_live_ou25, "is_market_primary"] = 0
    except Exception:
        pass

    # Enforce exactly one primary per (league, fixture_key, market) deterministically
    try:
        m_base = pd.to_numeric(out.get("is_market_primary", 0), errors="coerce").fillna(0).astype(int).eq(1)
        if bool(m_base.any()):
            # reset then re-pick winners
            out.loc[m_base, "is_market_primary"] = 0

            tmp = out.loc[m_base, [
                "league", "fixture_key", "market",
                "ftr_margin", "model_p_for_bookie", "bookie_implied_novig"
            ]].copy()

            tmp["ftr_margin"] = pd.to_numeric(tmp.get("ftr_margin", np.nan), errors="coerce").fillna(0.0)
            tmp["model_p_for_bookie"] = pd.to_numeric(tmp.get("model_p_for_bookie", np.nan), errors="coerce").fillna(0.0)
            tmp["bookie_implied_novig"] = pd.to_numeric(tmp.get("bookie_implied_novig", np.nan), errors="coerce").fillna(-1.0)

            tmp = tmp.sort_values(
                ["league", "fixture_key", "market", "ftr_margin", "model_p_for_bookie", "bookie_implied_novig"],
                ascending=[True, True, True, False, False, False],
                kind="mergesort",
            )

            idx_keep = tmp.groupby(["league", "fixture_key", "market"], as_index=False).head(1).index
            out.loc[idx_keep, "is_market_primary"] = 1
    except Exception:
        pass

    # Backward-compatible alias for downstream code
    out["is_market_primary"] = pd.to_numeric(out.get("is_market_primary", 0), errors="coerce").fillna(0).astype(int)
    out["is_candidate"] = pd.to_numeric(out.get("is_candidate", 0), errors="coerce").fillna(0).astype(int)
    out["is_fixture_primary"] = out["is_market_primary"].astype(int)

    # ------------------------------------------------------------------
    # Explicit premium OU25 identity assignment
    # Premium OVER25 rows should always become candidate rows.
    # Do not key this off score; branch/state/shadow govern identity.
    # ------------------------------------------------------------------
    try:
        mk = out.get("market", "").astype("string").fillna("").str.lower().str.strip()
        pick = out.get("bookie_pick", out.get("selection", "")).astype("string").fillna("").str.upper().str.strip()
        br = out.get("ou25_policy_branch", "").astype("string").fillna("").str.strip()
        state = out.get("ou25_policy_state", "").astype("string").fillna("").str.lower().str.strip()
        sh = pd.to_numeric(out.get("ou25_is_shadow", 0), errors="coerce").fillna(0).astype(int)
        prem = pd.to_numeric(out.get("ou25_is_premium_candidate", 0), errors="coerce").fillna(0).astype(int)

        premium_ou25 = (
            mk.eq("ou25")
            & pick.eq("OVER25")
            & (~state.eq("review"))
            & sh.eq(0)
            & (br.eq("ou25_combined_topq_080") | prem.ge(1))
        )

        if bool(premium_ou25.any()):
            out.loc[premium_ou25, "is_candidate"] = 1
            out.loc[premium_ou25, "candidate_rank"] = 1
            out.loc[premium_ou25, "is_market_primary"] = 0
            out.loc[premium_ou25, "is_fixture_primary"] = 0
    except Exception:
        pass

    # Deterministic ranking score:
    # - Side markets: score = gap (model_p_for_bookie - bookie_implied)
    # - FTR: score = ftr_margin + gap (margin adds stability signal)
    out["gap"] = pd.to_numeric(out.get("gap"), errors="coerce")
    out["gap_novig"] = pd.to_numeric(out.get("gap_novig"), errors="coerce")
    out["ftr_margin"] = pd.to_numeric(out.get("ftr_margin"), errors="coerce")

    # Prefer no-vig gap when available; fall back to raw gap
    gap_used = out["gap_novig"].where(out["gap_novig"].notna(), out["gap"])

    mkt = out["market"].astype(str).str.lower().str.strip() if "market" in out.columns else ""
    out["score"] = gap_used

    try:
        is_ftr = mkt.eq("ftr")
        out.loc[is_ftr, "score"] = gap_used.loc[is_ftr].fillna(0.0) + out.loc[is_ftr, "ftr_margin"].fillna(0.0)
    except Exception:
        out["score"] = gap_used

    # Team-goals markets have no bookie implied; rank them by model probability
    try:
        is_tg = mkt.isin(["tg15", "tg25"])
        out.loc[is_tg, "score"] = pd.to_numeric(out.loc[is_tg, "model_p_for_bookie"], errors="coerce")
    except Exception:
        pass

    imp_tag = int(round(float(args.implied_min) * 100))

    # ------------------------------------------------------------------
    # Deterministic ranking across pandas versions:
    # NaN ordering in tie-break columns can vary, so fill them for sorting.
    # Use stable mergesort to preserve deterministic ordering.
    # ------------------------------------------------------------------
    if "bookie_implied" in out.columns:
        out["bookie_implied"] = pd.to_numeric(out["bookie_implied"], errors="coerce")
    else:
        out["bookie_implied"] = np.nan

    if "bookie_implied_novig" in out.columns:
        out["bookie_implied_novig"] = pd.to_numeric(out["bookie_implied_novig"], errors="coerce")
    else:
        out["bookie_implied_novig"] = np.nan

    out["bookie_implied_novig_f"] = out["bookie_implied_novig"].fillna(-1.0)
    out["bookie_implied_f"] = out["bookie_implied"].fillna(-1.0)

    # Prefer no-vig implied for tie-breaks when it exists; otherwise fall back to implied.
    if out["bookie_implied_novig"].notna().any():
        out = out.sort_values(
            ["market", "score", "bookie_implied_novig_f", "bookie_implied_f"],
            ascending=[True, False, False, False],
            kind="mergesort",
        )
    else:
        out = out.sort_values(
            ["market", "score", "bookie_implied_f"],
            ascending=[True, False, False],
            kind="mergesort",
        )

    # Keep output schema stable: drop helper tie-break columns
    out = out.drop(columns=["bookie_implied_novig_f", "bookie_implied_f"], errors="ignore")
    # ------------------------------------------------------------------
    # Propagate per-fixture BTTS/O25 probabilities onto ALL market rows.
    # NOTE: signal_over25 is SIDE/ROW-aware (depends on selection OVER25 vs UNDER25).
    # Do NOT propagate it across fixtures via groupby/transform.
    # ------------------------------------------------------------------
    try:
        if "fixture_key" in out.columns:
            gcols = ["league", "fixture_key"] if ("league" in out.columns) else ["fixture_key"]
            for c in ["prob_btts_v2", "prob_over25_v2"]:
                if c in out.columns:
                    out[c] = out.groupby(gcols)[c].transform(ff_first_nonnull)
    except Exception:
        pass
    # ------------------------------------------------------------------
    # BTTS-only cleanup before export
    # ------------------------------------------------------------------
    try:
        m_btts_final = (
            out["market"]
            .astype("string")
            .fillna("")
            .str.lower()
            .str.strip()
            .eq("btts")
        )

        for c in ["signal_btts", "signal_btts_fixture", "signal_btts_side"]:
            if c not in out.columns:
                out[c] = pd.NA
            out.loc[~m_btts_final, c] = pd.NA
    except Exception:
        pass

    # Main export should only carry primary BTTS model rows.
    # BTTS_VALUEEV remains useful internally, but must not pollute the
    # exported ALLMARKETS contract consumed by wrappers / proxy audits.
    try:
        out = _filter_primary_btts_rows(out)
    except Exception:
        pass
    try:
        out = _attach_phase8b_coherence_features(out)
    except Exception:
        pass
    # Preserve side-signal columns across late sort/recombine/export fixup steps.
    # We observed OU25 labels are correct through policy + primary enforcement,
    # then collapse to NEUTRAL in the final export path. Snapshot them here and
    # restore by stable row identity just before write.
    try:
        _sig_restore_cols = [c for c in ["signal_btts", "signal_btts_fixture", "signal_btts_side"] if c in out.columns]
        _sig_restore_keys = [c for c in ["league", "fixture_key", "market", "bookie_pick", "product", "model_lane"] if c in out.columns]
        if _sig_restore_cols and len(_sig_restore_keys) == 6:
            _sig_restore = out[_sig_restore_keys + _sig_restore_cols].copy()
            for _c in _sig_restore_keys:
                _sig_restore[_c] = _sig_restore[_c].astype("string").fillna("").str.strip()
            _sig_restore = _sig_restore.drop_duplicates(subset=_sig_restore_keys, keep="first")
        else:
            _sig_restore = None
    except Exception:
        _sig_restore = None

    # Phase 7C: propagate fixture-level BTTS/OU25 consensus context onto sibling rows.
    try:
        out = _stamp_cross_market_consensus_support(out)
    except Exception:
        pass

    # Final output dtype normalisation (stop dtype regressions across runs)
    out = _finalise_output_dtypes(out)

    # Preserve last known-good side signal labels before late export fixups.
    try:
        _preserved_signal_cols = [
            c for c in (
                "league", "fixture_key", "market", "bookie_pick", "product", "model_lane",
                "signal_over25", "signal_btts", "signal_btts_fixture", "signal_btts_side",
            ) if c in out.columns
        ]
        out_signal_preserve = out[_preserved_signal_cols].copy() if len(_preserved_signal_cols) >= 5 else None
    except Exception:
        out_signal_preserve = None

    dst = outdir / f"BOOKIE_IMP{imp_tag}_ALLMARKETS_{args.date_from}_to_{args.date_to}.csv"
    try:
        out = _ensure_export_side_probs(out)
    except Exception:
        pass

    # Restore preserved signal labels after export-side fixups.
    try:
        out = _restore_preserved_signal_cols(out, out_signal_preserve)
    except Exception:
        pass

    # Restore preserved side-signal labels after late export fixups.
    try:
        if _sig_restore is not None:
            _restore_keys = ["league", "fixture_key", "market", "bookie_pick", "product", "model_lane"]
            for _c in _restore_keys:
                if _c in out.columns:
                    out[_c] = out[_c].astype("string").fillna("").str.strip()

            # Merge with suffix so we can COALESCE instead of overwriting.
            out = out.merge(_sig_restore, on=_restore_keys, how="left", suffixes=("", "__preserved"))

            # Only backfill missing/blank signals from preserved snapshot.
            for _c in ["signal_btts", "signal_btts_fixture", "signal_btts_side"]:
                cp = f"{_c}__preserved"
                if cp in out.columns:
                    if _c in out.columns:
                        cur = out[_c].astype("string").fillna("").str.strip()
                        prv = out[cp].astype("string").fillna("").str.strip()
                        # Fill only where current is blank
                        out[_c] = cur.mask(cur.eq("") & prv.ne(""), prv)
                    else:
                        out[_c] = out[cp]
                    out = out.drop(columns=[cp], errors="ignore")
            # Re-enforce row-aware OU25 signals after late restores/merges.
            try:
                out = _enforce_row_aware_ou25_signal(out)
            except Exception:
                pass
            # Final guard: OU25 direction check AFTER all restores (no mutation)
            try:
                if "signal_over25" in out.columns:
                    mk = out.get("market", "").astype("string").fillna("").str.lower().str.strip()
                    m_ou = mk.eq("ou25")
                    if bool(m_ou.any()):
                        pick_u = (
                            out.get("bookie_pick", out.get("selection", ""))
                            .astype("string")
                            .fillna("")
                            .str.upper()
                            .str.strip()
                        )
                        is_over_row = m_ou & pick_u.eq("OVER25")
                        is_under_row = m_ou & pick_u.eq("UNDER25")
                        is_under_row = is_under_row & (~is_over_row)

                        so2 = out.get(
                            "signal_over25",
                            pd.Series(index=out.index, dtype="string"),
                        ).astype("string").fillna("").str.upper().str.strip()

                        post_bad_over = is_over_row & so2.str.contains("UNDER", regex=False)
                        post_bad_under = is_under_row & so2.str.contains("OVER", regex=False)

                        if bool(getattr(args, "debug", False)):
                            print(
                                f"[bookie_allmarkets] final OU25 direction check: "
                                f"post_bad_over={int(post_bad_over.sum())} post_bad_under={int(post_bad_under.sum())}"
                            )

                        assert int(post_bad_over.sum()) == 0 and int(post_bad_under.sum()) == 0, (
                            "OU25 signal direction mismatch persists at export time: "
                            f"bad_over={int(post_bad_over.sum())} bad_under={int(post_bad_under.sum())}"
                        )
            except AssertionError:
                raise
            except Exception:
                pass
    except AssertionError:
        raise
    except Exception:
        pass
    # ------------------------------------------------------------------
    # Backfill FTR bookie_od from 1X2 odds (HOME/DRAW/AWAY) so deploy gates
    # that expect bookie_od do not drop valid FTR rows.
    # ------------------------------------------------------------------
    try:
        out = _fill_ftr_bookie_od_from_1x2(out)
    except Exception:
        pass
    # Debug sanity: confirm FTR bookie_od is populated at export time
    if bool(getattr(args, "debug", False)):
        try:
            m_ftr_dbg = out.get("market", "").astype("string").fillna("").str.lower().str.strip().eq("ftr")
            ftr_dbg = out.loc[m_ftr_dbg].copy() if bool(m_ftr_dbg.any()) else out.iloc[0:0].copy()
            n_ftr = int(len(ftr_dbg))
            n_nan = int(pd.to_numeric(ftr_dbg.get("bookie_od", np.nan), errors="coerce").isna().sum()) if n_ftr else 0
            if n_ftr:
                print(f"[bookie_allmarkets] EXPORT CHECK: ftr_rows={n_ftr} bookie_od_nan={n_nan}")
        except Exception:
            pass

    # Synthetic draw rows: one FTR DRAW row per fixture, scored with the saved
    # draw meta pipeline so deploy can route explicit draw candidates.
    try:
        draw_rows = _generate_draw_rows(out, _DRAW_META_MODEL)
        if draw_rows is not None and not draw_rows.empty:
            out = pd.concat([out, draw_rows], ignore_index=True)
            print(f"[draw] appended {len(draw_rows)} synthetic DRAW rows")
    except Exception as e:
        print(f"[draw] WARNING: synthetic DRAW row generation failed: {e}")

    # Final Phase 8 refresh on the fully materialized export pool.
    try:
        out = _attach_phase8a_grid_features(out, max_goals=6)
        out = _attach_phase8b_coherence_features(out)
        out = _stamp_phase8_meta_scores(out)
    except Exception:
        pass

    try:
        out = _stamp_team_goal_intelligence_fields(out)
    except Exception:
        pass

    try:
        out = _stamp_value_edge_fields(out)
    except Exception:
        pass

    # Strict export QC: fail fast if key markets are not "QC green"
    if bool(getattr(args, "strict", False)):
        _run_strict_qc_asserts(out, csv_path=dst)

    out.to_csv(dst, index=False)

    # Lightweight sanity (behind --debug): OU25 summary only
    if bool(getattr(args, "debug", False)):
        try:
            m_ou_dbg = out.get("market", "").astype("string").fillna("").str.lower().str.strip().eq("ou25")
            if bool(m_ou_dbg.any()):
                ou_dbg = out.loc[m_ou_dbg].copy()
                print("\n[OU25_DEBUG_SUMMARY]")
                print(f"ou25_rows={len(ou_dbg)}")
                print("signal_over25 counts:")
                print(
                    ou_dbg.get("signal_over25", pd.Series([], dtype="string"))
                    .astype("string")
                    .value_counts(dropna=False)
                )

                if "ou25_policy_branch" in ou_dbg.columns and "signal_over25" in ou_dbg.columns:
                    print("policy branch x signal:")
                    try:
                        print(pd.crosstab(ou_dbg["ou25_policy_branch"], ou_dbg["signal_over25"], dropna=False))
                    except Exception:
                        pass
        except Exception as e:
            print(f"[OU25_DEBUG_SUMMARY failed] {e}")

    print("WROTE:", dst)
    print("rows:", len(out), "| leagues:", out["league"].nunique(), "| markets:", out["market"].value_counts().to_dict())


# ------------------------------------------------------------
# Optional FTR calibration (per-league isotonic / platt)
# ------------------------------------------------------------
def _load_ftr_calibration_models() -> dict:
    """Load calibration models from JSON (if present)."""
    try:
        if str(os.environ.get("OG_FTR_CALIBRATION", "1")).strip() in ("0", "false", "False"):
            return {}
        path = os.environ.get(
            "OG_FTR_CALIBRATION_JSON",
            "predictions_output/walk_forward/_MASTER/CALIBRATION/FTR_CALIBRATION__MODELS.json",
        )
        if not path:
            return {}
        if not os.path.exists(path):
            return {}
        import json

        with open(path, "r", encoding="utf-8") as fh:
            rows = json.load(fh)
        # index by league
        out = {}
        for r in rows:
            lg = str(r.get("league", "")).strip()
            if not lg:
                continue
            out[lg] = r
        return out
    except Exception:
        return {}


def _apply_ftr_calibration(df: pd.DataFrame, calib: dict) -> pd.DataFrame:
    """Apply isotonic/platt calibration to model_p_for_bookie for FTR rows."""
    if df is None or df.empty or not calib:
        return df
    if "market" not in df.columns or "model_p_for_bookie" not in df.columns:
        return df

    out = df.copy()
    mk = out["market"].astype("string").fillna("").str.lower().str.strip()
    is_ftr = mk.eq("ftr")
    if not bool(is_ftr.any()):
        return out

    p = pd.to_numeric(out.get("model_p_for_bookie", np.nan), errors="coerce").astype(float)

    # Preserve raw
    out["model_p_for_bookie_raw"] = p
    out["model_p_for_bookie_cal"] = p.copy()
    out["calibration_method"] = ""

    for lg, info in calib.items():
        m_lg = is_ftr & out.get("league", "").astype("string").fillna("").str.strip().eq(lg)
        if not bool(m_lg.any()):
            continue
        p_lg = p.loc[m_lg].astype(float)
        if p_lg.empty:
            continue

        # Prefer isotonic if present
        iso_x = info.get("isotonic_x")
        iso_y = info.get("isotonic_y")
        if isinstance(iso_x, list) and isinstance(iso_y, list) and len(iso_x) >= 2 and len(iso_y) == len(iso_x):
            try:
                cal = np.interp(p_lg.values, np.asarray(iso_x, dtype="float64"), np.asarray(iso_y, dtype="float64"))
                out.loc[m_lg, "model_p_for_bookie_cal"] = cal
                out.loc[m_lg, "calibration_method"] = "isotonic"
                continue
            except Exception:
                pass

        # Fallback to platt
        try:
            a = float(info.get("platt_coef", 0.0))
            b = float(info.get("platt_intercept", 0.0))
            cal = 1.0 / (1.0 + np.exp(-(a * p_lg.values + b)))
            out.loc[m_lg, "model_p_for_bookie_cal"] = cal
            out.loc[m_lg, "calibration_method"] = "platt"
        except Exception:
            out.loc[m_lg, "calibration_method"] = "none"

    # Only overwrite for FTR rows (so gates use calibrated prob)
    out.loc[is_ftr, "model_p_for_bookie"] = out.loc[is_ftr, "model_p_for_bookie_cal"]
    # Non-FTR rows: keep empty cal fields
    out.loc[~is_ftr, "model_p_for_bookie_cal"] = np.nan
    out.loc[~is_ftr, "calibration_method"] = ""

    return out


if __name__ == "__main__":
    main()
