import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, average_precision_score, precision_recall_curve, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import HistGradientBoostingClassifier
import pandas as pd
from leak_tests import assert_no_shuffle_leak
import os
import json
import joblib
from datetime import datetime

from etl_press_intensity import ensure_press_intensity_on_disk

# --- Robust calibrator for draw model (sklearn >=1.5) ----------------------
if "_make_calibrator" not in globals():
    def _make_calibrator(base_estimator, method: str = "isotonic", cv: int = 3):
        try:
            from sklearn.utils.fixes import _wrap_in_frozen_estimator as _Freeze  # type: ignore
            base_estimator = _Freeze(base_estimator)
        except Exception:
            pass
        from sklearn.calibration import CalibratedClassifierCV
        return CalibratedClassifierCV(estimator=base_estimator, method=method, cv=cv)

# Additional imports for logistic regression and oversampling
from sklearn.linear_model import LogisticRegression

# Try to import RandomOverSampler; fall back gracefully if unavailable
try:
    from imblearn.over_sampling import RandomOverSampler
    _HAS_IMBLEARN = True
except Exception:
    print("⚠️  imblearn not installed – oversampling will be skipped. Install via `pip install imblearn` for better class balance.")
    _HAS_IMBLEARN = False

from constants import (
    MODEL_DIR,
    # threshold learning / gating controls
    N0_SHRINK,
    N0_SHRINK_PER_LEAGUE,
    LOCKED_THRESHOLDS,
    # new autos for per-league draw threshold learning
    AUTO_DRAW_THRESHOLD,
    DRAW_THRESHOLD_MODE_DEFAULT,
    DRAW_PRECISION_TARGET_DEFAULT,
    DRAW_THRESHOLD_MIN_SAMPLES,
    DRAW_THRESHOLD_FALLBACK,
    USE_LOCKED_DRAW_THRESHOLDS,
)
# Optional per-league overrides (guard with getattr for backward compat)
try:
    from constants import DRAW_THRESHOLD_MODE_PER_LEAGUE  # dict[str,str]
except Exception:
    DRAW_THRESHOLD_MODE_PER_LEAGUE = {}
try:
    from constants import DRAW_MIN_SAMPLES_PER_LEAGUE  # dict[str,int]
except Exception:
    DRAW_MIN_SAMPLES_PER_LEAGUE = {}

# --- training row filter: only completed fixtures ---------------------------
COMPLETED_STATUS_PATTERNS = (
    r"\bft\b",
    r"full[\s-]?time",
    r"finished",
    r"\bfinal\b",
    r"match[\s-]?finished",
    r"\baet\b",
    r"after[\s-]?extra[\s-]?time",
    r"pens?",
    r"penalt(?:y|ies)",
    r"\bpso\b",
)

INCOMPLETE_STATUS_PATTERNS = (
    r"postp|postponed|abandon|suspend|void|cancel",
    r"\bwo\b|walkover",
    r"\bns\b|not\s*started",
    r"live|in\s*play",
    r"\bht\b|half[\s-]?time|\b1h\b|\b2h\b",
)

def _filter_completed_training_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep completed fixtures for training.
    Robust logic:
      • If a 'status'-like column exists, consider a row completed when it matches any COMPLETED pattern
        and does NOT match any INCOMPLETE pattern.
      • Also keep any row that clearly has labels available (FTR or both goal counts present).
      • Union of (status_complete) OR (labels_present). If the union is empty, fall back to labels_present.
    """
    import pandas as _pd
    before = len(df)

    # Label-availability mask (safe fallback)
    has_label = _pd.Series(False, index=df.index)
    if "FTR" in df.columns:
        has_label = has_label | df["FTR"].notna()
    hg_col = next((c for c in ("home_team_goal_count", "home_goals", "goals_home") if c in df.columns), None)
    ag_col = next((c for c in ("away_team_goal_count", "away_goals", "goals_away") if c in df.columns), None)
    if hg_col and ag_col:
        has_label = has_label | (df[hg_col].notna() & df[ag_col].notna())

    # Status-based mask (if a status-like column exists)
    status_cols = [c for c in df.columns if "status" in c.lower()]
    status_complete = _pd.Series(False, index=df.index)
    if status_cols:
        # Use the first 'status' column found
        sc = status_cols[0]
        s = df[sc].astype(str).str.lower().str.strip()
        # normalise some separators
        s = s.str.replace("-", " ", regex=False)

        # Completed patterns
        for pat in COMPLETED_STATUS_PATTERNS:
            status_complete = status_complete | s.str.contains(pat, regex=True, na=False)
        # Incomplete patterns
        status_bad = _pd.Series(False, index=df.index)
        for pat in INCOMPLETE_STATUS_PATTERNS:
            status_bad = status_bad | s.str_contains(pat, regex=True, na=False) if hasattr(s, "str_contains") else status_bad | s.str.contains(pat, regex=True, na=False)
        status_complete = status_complete & ~status_bad
    else:
        sc = None

    keep = status_complete | has_label
    if keep.sum() == 0 and has_label.any():
        keep = has_label

    out = df.loc[keep].copy()
    try:
        msg_bits = []
        if sc:
            msg_bits.append(f"status-match={int(status_complete.sum())}")
        msg_bits.append(f"label-derived={int(has_label.sum())}")
        dropped = before - len(out)
        print(f"🧹 Train filter: completed fixtures {len(out)}/{before} via {'status' if sc else 'labels'} ({', '.join(msg_bits)}; {dropped} dropped)")
    except Exception:
        pass
    return out
# --- helper: ensure/derive FTR column --------------------------------
def _ensure_ftr_col(df):
    """
    Ensure df['FTR'] exists and is encoded 0=Home, 1=Draw, 2=Away.
    Tries common aliases first; otherwise derives from goal counts.
    """
    import numpy as np
    import pandas as pd

    if "FTR" in df.columns:
        return df

    # Common alias: sometimes results are "H/D/A" in another column
    for alias in ("Full Time Result", "result", "full_time_result"):
        if alias in df.columns:
            m = df[alias].astype(str).str.upper().map({"H": 0, "D": 1, "A": 2})
            if m.notna().any():
                df["FTR"] = m.fillna(1).astype(int)
                return df

    # Derive from goals if present
    home_g = None
    away_g = None
    for h in ("home_team_goal_count", "home_goals", "goals_home"):
        if h in df.columns:
            home_g = h
            break
    for a in ("away_team_goal_count", "away_goals", "goals_away"):
        if a in df.columns:
            away_g = a
            break

    if home_g and away_g:
        hg = df[home_g]
        ag = df[away_g]
        ftr = np.where(hg > ag, 0, np.where(hg < ag, 2, 1))
        df["FTR"] = ftr.astype(int)
        return df

    # Could not derive
    return df

# --- tiny validator: report press-intensity enrichment coverage -----------------
from pathlib import Path

def _report_press_intensity_coverage(league_folder: Path) -> None:
    """
    Print a short coverage summary for the given *Matches/<League>* folder:
      • how many match CSVs already contain home/away_press_intensity
      • how many player-season CSVs exist under Players/<League>
    This is non-fatal and only reads headers (nrows=1) for speed.
    """
    try:
        import glob
        import os
        import pandas as _pd
    except Exception:
        return

    # --- find match CSVs (skip fixtures/prediction/report artifacts) ---
    def _skip(p: str) -> bool:
        name = os.path.basename(p).lower()
        if any(k in name for k in ("upcoming", "fixture", "fixtures", "prediction", "predictions", "report")):
            return True
        if any(k in p.lower() for k in ("predictions_output", "modelstore")):
            return True
        return False

    match_paths = sorted(
        p for p in glob.glob(str(league_folder / "**" / "*.csv"), recursive=True)
        if not _skip(p)
    )

    total = len(match_paths)
    enriched = 0
    missing = 0
    for p in match_paths:
        try:
            hdr = _pd.read_csv(p, nrows=1)
            if {"home_press_intensity", "away_press_intensity"}.issubset(hdr.columns):
                enriched += 1
            else:
                missing += 1
        except Exception:
            missing += 1

    # --- count available Players/<League> season files ---
    # Derive Players path next to Matches
    lf = Path(league_folder)
    league_name = lf.name
    # parent of Matches/<League> is Matches; parent of that is project root
    players_root = lf.parent.parent / "Players"
    players_folder = players_root / league_name
    if players_folder.exists():
        player_csvs = list(players_folder.glob("**/*.csv"))
    else:
        player_csvs = []

    print(
        f"🧪 Coverage: press-intensity present in {enriched}/{total} match CSVs | "
        f"Players season files: {len(player_csvs)} under {players_folder if players_folder.exists() else players_root / '[missing]'}"
    )
# --- helper: league-/season-normalised press intensity (z-scores) ---------
def _add_press_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add within-season z-scores for rolling-press features, if present.
    Creates: rolling5_home_press_z, rolling5_away_press_z, rolling5_press_z_diff.
    Groups by 'season' when available; otherwise normalises globally.
    """
    home_col = "rolling5_home_press_intensity"
    away_col = "rolling5_away_press_intensity"
    if home_col not in df.columns or away_col not in df.columns:
        return df

    grp_key = None
    for cand in ("season", "Season", "season_name"):
        if cand in df.columns:
            grp_key = cand
            break

    def _z(s: pd.Series) -> pd.Series:
        mu = s.mean()
        sd = s.std(ddof=0)
        if pd.isna(sd) or sd == 0:
            return pd.Series(0.0, index=s.index)
        return (s - mu) / sd

    if grp_key:
        df["rolling5_home_press_z"] = df.groupby(grp_key)[home_col].transform(_z)
        df["rolling5_away_press_z"] = df.groupby(grp_key)[away_col].transform(_z)
    else:
        df["rolling5_home_press_z"] = _z(df[home_col])
        df["rolling5_away_press_z"] = _z(df[away_col])

    df["rolling5_press_z_diff"] = df["rolling5_home_press_z"] - df["rolling5_away_press_z"]
    return df
# --- per-league threshold settings & persistence -------------------------
def _get_league_threshold_settings(league_name: str, cli_mode: str | None, cli_prec: float | None):
    """Resolve threshold learning settings for a league using constants + CLI.
    Returns (mode, precision_target, min_samples).
    CLI args win if explicitly set; otherwise constants drive the default.
    """
    # base from constants
    mode_const = DRAW_THRESHOLD_MODE_PER_LEAGUE.get(league_name, DRAW_THRESHOLD_MODE_DEFAULT)
    prec_const = float(DRAW_PRECISION_TARGET_DEFAULT)
    min_s_const = int(DRAW_MIN_SAMPLES_PER_LEAGUE.get(league_name, DRAW_THRESHOLD_MIN_SAMPLES))

    # CLI overrides when provided (non-None)
    mode = (cli_mode or mode_const).lower()
    precision_target = float(cli_prec if cli_prec is not None else prec_const)
    min_samples = int(min_s_const)
    return mode, precision_target, min_samples

def _threshold_store_path(league_name: str) -> str:
    safe = str(league_name).replace(" ", "_")
    return os.path.join(MODEL_DIR, f"{safe}_draw_threshold.json")

# --- helper: persist calibrated draw model for inference bridge ---
def _draw_model_flat_path(league_name: str, model_dir: str = MODEL_DIR) -> str:
    tag = str(league_name).replace(" ", "_")
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, f"{tag}_draw_clf.pkl")

# --- helper: persist joblib bundle with conventional downstream name ---
def _draw_bundle_joblib_path(league_name: str, model_dir: str = MODEL_DIR) -> str:
    tag = str(league_name).replace(" ", "_").lower()
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, f"{tag}__draw_bundle.joblib")

def save_draw_model_bundle(league_name, model, features, *, model_type: str = "gbm") -> None:
    """Write ModelStore/<LeagueTag>_draw_clf.pkl with {model, features, league, model_type}."""
    try:
        path = _draw_model_flat_path(league_name)
        payload = {
            "model": model,
            "features": list(features or []),
            "league": str(league_name),
            "model_type": str(model_type),
        }
        joblib.dump(payload, path)
        print(f"💾 Saved draw model → {path}")

        # Also save a joblib bundle with a conventional name for downstream tools
        bundle_path = _draw_bundle_joblib_path(league_name)
        joblib.dump(payload, bundle_path)
        print(f"💾 Saved draw bundle → {bundle_path}")
    except Exception as e:
        print(f"⚠️ Could not save draw model bundle: {e}")

# --- helper: load best k_pct for league from persisted threshold JSON ---
def load_best_k_pct_for_league(league_name: str, default_k_pct: float = 0.10):
    """
    Loads the best-performing k_pct (fraction) for a league, as determined by the highest precision
    among the "precision_at_k_list" in the per-league threshold JSON.
    Returns (k_pct: float, precision: float). If missing, returns (default_k_pct, None).
    """
    json_path = _threshold_store_path(league_name)
    try:
        with open(json_path, "r") as fh:
            payload = json.load(fh)
        prec_list = payload.get("precision_at_k_list", None)
        if not prec_list or not isinstance(prec_list, list):
            return (default_k_pct, None)
        # Find entry with highest precision
        best = max(prec_list, key=lambda d: d.get("precision", float("-inf")))
        k_pct = float(best.get("k_pct", default_k_pct))
        precision = float(best.get("precision", None))
        return (k_pct, precision)
    except Exception:
        return (default_k_pct, None)
# -----------------------------
# Explicit prematch allowlist (ETL‑backed + odds)
# -----------------------------
SAFE_PREMATCH_WHITELIST = [
    # team strength + parity
    "Pre-Match PPG (Home)", "Pre-Match PPG (Away)",
    "ppg_diff",
    "elo_diff",
    "rest_diff",
    "Home Team Pre-Match xG", "Away Team Pre-Match xG",
    "xg_diff_abs", "recent_xg_diff_abs",
    # fixture history + low-form flag
    "h2h_draw_rate", "both_slumping",
    # market (1X2) + derived parity metrics
    "odds_ft_home_team_win", "odds_ft_draw", "odds_ft_away_team_win",
    "draw_implied", "odds_parity", "odds_skew", "implied_prob_diff", "odds_diff",
    # optional market percentages when present
    "btts_percentage_pre_match", "over_15_percentage_pre_match", "over_25_percentage_pre_match",
    # team draw form / parity
    "home_draw_rate_rolling", "away_draw_rate_rolling", "team_draw_parity",
    # ETL-backed rolling press-intensity proxies (lagged)
    "rolling5_home_press_intensity", "rolling5_away_press_intensity", "rolling5_press_intensity_diff",
    # league-normalised press (z-scores)
    "rolling5_home_press_z", "rolling5_away_press_z", "rolling5_press_z_diff",
    # goal-model derived (optional; safe at inference)
    "exp_goals_sum", "p00_est",
]

# --- Additional safe lag features ---
_SAFE_LAG_FEATURES = []
_SAFE_LAG_FEATURES += [
    "ppg_diff",
    "elo_diff",
    "rest_diff",
    "xg_diff_abs",
]

ALIAS_MAP = {
    # if any “clean” names missing, map alternates here
    "Pre-Match PPG (Home)": ["home_ppg", "pre_match_home_ppg"],
    "Pre-Match PPG (Away)": ["away_ppg", "pre_match_away_ppg"],
    "Home Team Pre-Match xG": ["home_team_pre_match_xg", "home_pre_xg"],
    "Away Team Pre-Match xG": ["away_team_pre_match_xg", "away_pre_xg"],
}

def _resolve_whitelist(df, base_list, alias_map):
    keep = []
    for c in base_list:
        if c in df.columns:
            keep.append(c)
            continue
        for alt in alias_map.get(c, []):
            if alt in df.columns:
                keep.append(alt)
                break
    # drop anything not present
    return [c for c in keep if c in df.columns]

def train_draw_classifier(training_data, feature_cols, *, league_name: str = "generic", threshold_mode: str | None = None, precision_target: float | None = None, top_k_pcts: list[float] | float = 0.10, model_type: str = "gbm"):
    import numpy as np  # ensure NumPy is in scope for NaN handling
    # Resolve threshold-learning settings (constants + CLI)
    mode_resolved, precision_target_resolved, min_samples_required = _get_league_threshold_settings(
        str(league_name), threshold_mode, precision_target
    )
    # ── guard‑rail: drop draw features absent from the current dataframe ──
    feature_cols_present = [c for c in feature_cols if c in training_data.columns]
    missing = set(feature_cols) - set(feature_cols_present)
    if missing:
        print(f"⚠️  Skipping {len(missing)} missing draw features: "
              f"{', '.join(list(missing)[:6])}{'…' if len(missing) > 6 else ''}")
    # Use the pruned list from here on
    feature_cols = feature_cols_present

    # Prefer an explicit prematch allowlist (leak-safe) over ad-hoc pruning
    safe_cols = _resolve_whitelist(training_data, SAFE_PREMATCH_WHITELIST, ALIAS_MAP)
    if len(safe_cols) < 6:
        print(f"⚠️  Too few safe prematch cols ({len(safe_cols)}). Adding market trio for stability.")
        for c in ("odds_ft_home_team_win", "odds_ft_draw", "odds_ft_away_team_win"):
            if c in training_data.columns and c not in safe_cols:
                safe_cols.append(c)
    # If we have any safe prematch columns, use them as the feature set
    if len(safe_cols) > 0:
        feature_cols = safe_cols

    # Prepare training data
    X = training_data[feature_cols].replace([np.inf, -np.inf], np.nan)
    # ── per‑model leak blacklist (high shuffle‑corr culprits) ──────────
    # strip any surrounding whitespace on col‑headers first so matching is robust
    X.columns = X.columns.str.strip()

    # --- SAFE core features that must never be blacklisted ---
    SAFE_CORE = {
        "Pre-Match PPG (Home)",
        "Pre-Match PPG (Away)",
        "Home Team Pre-Match xG",
        "Away Team Pre-Match xG",
    }

    # In‑play patterns used by multiple guards
    INPLAY_PATTERNS = (
        "shots", "possession", "fouls", "cards", "goal",
        "minute", "half_time", "red", "yellow"
    )

    DRAW_BLACKLIST = {
        "Home Team Pre-Match xG",
        "Away Team Pre-Match xG",
        "Pre-Match PPG (Home)",
        "Pre-Match PPG (Away)",
        "Game Week",
        "home_team_shots", "away_team_shots",
        "home_team_shots_on_target", "away_team_shots_on_target",
        "home_team_shots_off_target", "away_team_shots_off_target",
        "home_team_possession", "away_team_possession",
        "total_goals_at_half_time",
        "home_team_yellow_cards", "away_team_yellow_cards",
        "home_team_red_cards",    "away_team_red_cards",
    }
    # Never blacklist the SAFE core columns
    DRAW_BLACKLIST.difference_update(SAFE_CORE)

    # 1️⃣  Drop black‑listed columns from the design‑matrix
    X = X.drop(columns=[c for c in X.columns if c in DRAW_BLACKLIST],
               errors="ignore")

    # 2️⃣  Also prune them from the feature list used for importances later
    feature_cols = [c for c in feature_cols if c not in DRAW_BLACKLIST]

    # ---  Fallback guard  ------------------------------------------------
    # If every requested feature was either missing or black‑listed we’d
    # end up with an **empty** matrix which would crash scikit‑learn.
    # Instead, build a *pre‑match‑only* numeric feature set.
    if X.shape[1] == 0:
        # ① candidate numeric columns (exclude labels)
        candidates = (
            training_data.select_dtypes(include="number")
                         .drop(columns=["FTR", "is_draw"], errors="ignore")
                         .columns
        )

        # ② heuristics – filter out obvious in‑play stats
        INPLAY_PATTERNS = (
            "shots", "possession", "fouls", "cards", "goal",
            "minute", "half_time", "red", "yellow"
        )
        pre_match_cols = [
            c for c in candidates
            if not any(pat in c.lower() for pat in INPLAY_PATTERNS)
        ]

        # Add the two foul columns to the draw blacklist so they are excluded downstream
        DRAW_BLACKLIST.update({"home_team_fouls", "away_team_fouls"})

        # ⬅️ NEW – strip season/time proxies such as “Game Week”
        pre_match_cols = [
            c for c in pre_match_cols
            if c.lower() not in {"game week", "week"}
        ]

        # ③ ensure we still have a usable set; if not, relax the filter
        if len(pre_match_cols) < 2:
            print(
                f"⚠️  Too few safe pre‑match columns ({len(pre_match_cols)}). "
                "Falling back to the first 10 numeric candidates."
            )
            # keep the original order so we don’t leak future info
            pre_match_cols = list(candidates[:10])
            # re‑apply the in‑play pattern filter so card/shot stats do not
            # sneak back in via the relaxed fallback path
            pre_match_cols = [
                c for c in pre_match_cols
                if not any(pat in c.lower() for pat in INPLAY_PATTERNS)
            ]
            # still guard against an empty list (extreme edge‑case)
            if len(pre_match_cols) == 0:
                raise ValueError(
                    "❌  No usable pre‑match features available after pruning. "
                    "Provide a cleaned DRAW_FEATURES list instead."
                )

        print(
            f"⚠️  All requested draw features absent – "
            f"falling back to {len(pre_match_cols)} *pre‑match* columns."
        )
        X = training_data[pre_match_cols].replace([np.inf, -np.inf], np.nan)

        # Re‑apply the draw blacklist to the fallback matrix
        X = X.drop(columns=[c for c in X.columns if c in DRAW_BLACKLIST],
                   errors="ignore")
        feature_cols = [c for c in pre_match_cols if c not in DRAW_BLACKLIST]
    # FINAL guard – after all blacklist / pattern filters we may still
    # end up with an empty matrix which would crash scikit‑learn. If so,
    # re‑insert the *first* numeric candidate so at least one column
    # survives.
    if X.shape[1] == 0:
        # pick the first purely numeric column that is not the label **and** not black‑listed
        for col in training_data.select_dtypes(include="number").columns:
            if (
                col not in {"FTR", "is_draw"}
                and col not in DRAW_BLACKLIST
                and col.lower() not in {"game week", "week"}
                and not any(pat in col.lower() for pat in INPLAY_PATTERNS)
            ):
                X[col] = training_data[col]
                feature_cols = [col]
                print(f"⚠️  All columns pruned – falling back to single "
                      f"numeric column: {col}")
                break
        # absolute last‑resort – still empty → abort early
        if X.shape[1] == 0:
            raise ValueError(
                "❌  No usable feature columns remain after all pruning "
                "steps.  Provide a cleaned DRAW_FEATURES list instead."
            )
    # -------------------- target label --------------------
    # Use explicit `is_draw` column if present; otherwise derive
    # it from the FTR outcome (1 = draw, 0 = home/away win).
    if 'is_draw' in training_data.columns:
        y = training_data['is_draw'].astype(int)
    elif 'FTR' in training_data.columns:
        y = (training_data['FTR'] == 1).astype(int)
    else:
        raise KeyError(
            "Neither 'is_draw' nor 'FTR' column present in the "
            "training dataframe – cannot build the target label."
        )

    # Ensure NaNs are handled and capture the feature list at fit time
    X = X.fillna(0)
    trained_feature_cols = list(X.columns)
    model_type = (model_type or "gbm").lower().strip()
    print(f"🔧 Model type: {model_type}")

    # ──────────────────────────────────────────────────────────────
    # Out‑of‑fold (TimeSeriesSplit) training to avoid look‑ahead
    # ──────────────────────────────────────────────────────────────
    # Robust TimeSeriesSplit: ensure we don't request more folds than samples
    n_splits_req = 5
    n_samples = len(X)
    if n_samples <= 2:
        raise ValueError(f"❌  Too few training samples after filtering: {n_samples}")
    n_splits = min(n_splits_req, max(2, n_samples - 1))
    tss = TimeSeriesSplit(n_splits=n_splits)
    oof = np.zeros(len(X))            # out‑of‑fold hard labels (0/1)
    oof_proba = np.zeros(len(X))      # out‑of‑fold P(draw)
    for tr_idx, va_idx in tss.split(X):
        # Train/val split
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va = X.iloc[va_idx]

        # Oversample only the training fold if available
        if _HAS_IMBLEARN:
            sampler = RandomOverSampler(random_state=42)
            X_tr, y_tr = sampler.fit_resample(X_tr, y_tr)

        # Fit model with internal CV calibration (avoids deprecated cv='prefit')
        if model_type == "histgb":
            base_est = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_depth=None,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=0.0,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42,
            )
        else:
            base_est = GradientBoostingClassifier(
                n_estimators=1200,
                learning_rate=0.025,
                max_depth=4,
                min_samples_leaf=20,
                subsample=0.9,
                random_state=42,
            )
        calibrated = _make_calibrator(base_est, method="isotonic", cv=3)
        calibrated.fit(X_tr, y_tr)

        # OOF outputs
        proba_va = calibrated.predict_proba(X_va)[:, 1]
        oof_proba[va_idx] = proba_va
        oof[va_idx] = (proba_va >= 0.5).astype(int)

    # Fit a final model on the full data for downstream use
    X_fit, y_fit = X, y
    if _HAS_IMBLEARN:
        sampler = RandomOverSampler(random_state=42)
        X_fit, y_fit = sampler.fit_resample(X_fit, y_fit)

    # Final model: backbone + isotonic calibration via internal CV
    if model_type == "histgb":
        base_final = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=None,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
        )
    else:
        base_final = GradientBoostingClassifier(
            n_estimators=1200,
            learning_rate=0.025,
            max_depth=4,
            min_samples_leaf=20,
            subsample=0.9,
            random_state=42,
        )
    clf = _make_calibrator(base_final, method="isotonic", cv=3)
    clf.fit(X_fit, y_fit)

    base_acc = accuracy_score(y, oof)
    draw_rate = float(y.mean())
    maj_acc = max(draw_rate, 1.0 - draw_rate)
    try:
        oof_auc = roc_auc_score(y, oof_proba)
    except Exception:
        oof_auc = float('nan')
    print(f"OOF accuracy: {base_acc:.3f} | OOF ROC AUC: {oof_auc:.3f} | class balance (draws): {draw_rate:.3f} | majority baseline: {maj_acc:.3f}")
    # ---- Precision-Recall diagnostics & threshold candidates ----
    try:
        pr_auc = average_precision_score(y, oof_proba)
    except Exception:
        pr_auc = float('nan')
    print(f"PR AUC (Average Precision): {pr_auc:.3f}")

    # Best F1 threshold from PR curve
    try:
        prec, rec, thr_pr = precision_recall_curve(y, oof_proba)
        # precision_recall_curve returns thresholds of length n-1
        f1 = np.zeros_like(thr_pr)
        # use prec[1:], rec[1:] to align with thr_pr
        with np.errstate(divide='ignore', invalid='ignore'):
            f1 = 2 * (prec[1:] * rec[1:]) / (prec[1:] + rec[1:])
        best_f1_idx = int(np.nanargmax(f1)) if len(f1) else 0
        best_f1_thr = float(thr_pr[best_f1_idx]) if len(thr_pr) else 0.5
        best_f1 = float(f1[best_f1_idx]) if len(f1) else float('nan')
    except Exception:
        best_f1_thr = 0.5
        best_f1 = float('nan')

    # Threshold to hit a target precision (e.g., 0.60) if achievable
    target_precision = float(precision_target_resolved)
    thr_p_target = None
    if 'prec' in locals() and 'thr_pr' in locals() and len(thr_pr):
        # iterate from high precision to low (prec is aligned with thr as prec[1:])
        candidates = [(p, r, t) for p, r, t in zip(prec[1:], rec[1:], thr_pr) if np.isfinite(p) and p >= target_precision]
        if candidates:
            # pick the one with highest recall among those meeting precision
            p_sel, r_sel, t_sel = max(candidates, key=lambda x: x[1])
            thr_p_target = float(t_sel)
            print(f"Target precision {target_precision:.2f} achievable at thr={thr_p_target:.3f} (precision={p_sel:.3f}, recall={r_sel:.3f})")
        else:
            print(f"Target precision {target_precision:.2f} not reachable on OOF.")

    # Keep Youden-J as primary threshold but report F1-optimal too
    print(f"Best F1 threshold: {best_f1_thr:.3f} (F1={best_f1:.3f})")
    if base_acc < maj_acc:
        print("⚠️  OOF accuracy is below the majority baseline — model likely underfitting; consider widening features or tuning GBM.")

    # Best threshold via Youden's J statistic from OOF probabilities
    try:
        fpr, tpr, thr = roc_curve(y, oof_proba)
        j = tpr - fpr
        best_idx = int(np.nanargmax(j)) if len(j) else 0
        best_thr = float(thr[best_idx]) if len(thr) else 0.5
    except Exception:
        best_thr = 0.5

    # Select deployment threshold based on threshold_mode
    deploy_thr = best_thr
    deploy_mode = "youden"
    mode = (mode_resolved or "youden").strip().lower()
    if mode == "f1":
        deploy_thr = float(best_f1_thr)
        deploy_mode = "f1"
    elif mode == "precision":
        if 'thr_p_target' in locals() and thr_p_target is not None:
            deploy_thr = float(thr_p_target)
            deploy_mode = f"precision@{target_precision:.2f}"
        else:
            print(f"⚠️  Target precision {target_precision:.2f} not achievable; falling back to Youden-J.")
            deploy_thr = float(best_thr)
            deploy_mode = "youden"
    else:
        deploy_thr = float(best_thr)
        deploy_mode = "youden"

    setattr(clf, "best_threshold_", deploy_thr)
    setattr(clf, "threshold_mode_", deploy_mode)
    setattr(clf, "precision_target_", float(precision_target_resolved))

    # Recompute OOF accuracy at the selected deployment threshold
    oof_pred_thr = (oof_proba >= deploy_thr).astype(int)
    acc_at_thr = accuracy_score(y, oof_pred_thr)
    print(f"• OOF accuracy @ selected threshold ({deploy_mode}) {deploy_thr:.3f}: {acc_at_thr:.3f}")

    # OOF precision/recall/F1 at the selected deployment threshold
    oof_prec = precision_score(y, oof_pred_thr, zero_division=0)
    oof_rec  = recall_score(y, oof_pred_thr, zero_division=0)
    oof_f1   = f1_score(y, oof_pred_thr, zero_division=0)
    print(f"• OOF precision/recall/F1 @ {deploy_mode} {deploy_thr:.3f}: {oof_prec:.3f} / {oof_rec:.3f} / {oof_f1:.3f}")

    # Confusion matrix at selected deployment threshold
    try:
        tn, fp, fn, tp = confusion_matrix(y, oof_pred_thr, labels=[0, 1]).ravel()
    except Exception:
        tn = fp = fn = tp = 0

    # Also show accuracy at Youden-J and F1 thresholds (for reference)
    if deploy_mode != "youden":
        yj_pred = (oof_proba >= best_thr).astype(int)
        print(f"• OOF accuracy @ Youden-J {best_thr:.3f}: {accuracy_score(y, yj_pred):.3f}")
    if 'best_f1_thr' in locals() and deploy_thr != float(best_f1_thr):
        f1_pred = (oof_proba >= best_f1_thr).astype(int)
        print(f"• OOF accuracy @ F1-optimal {best_f1_thr:.3f}: {accuracy_score(y, f1_pred):.3f}")

    # Rank-based view: precision among top-k% most draw-like fixtures (support multiple k's)
    # Normalize input to a list of floats
    if isinstance(top_k_pcts, (int, float)):
        _topk_list = [float(top_k_pcts)]
    else:
        _topk_list = [float(x) for x in (top_k_pcts or [0.10])]

    precision_at_k_list = []
    for k_pct in _topk_list:
        k_pct = max(0.001, min(0.999, k_pct))  # clamp to sensible bounds
        k = max(1, int(len(oof_proba) * k_pct))
        top_idx = np.argsort(oof_proba)[-k:]
        top_precision = float(y.iloc[top_idx].mean()) if hasattr(y, 'iloc') else float(np.mean(y[top_idx]))
        precision_at_k_list.append({"k_pct": float(k_pct), "k": int(k), "precision": float(top_precision)})
        print(f"• Precision among top {int(k_pct*100)}% (k={k}) by P(draw): {top_precision:.3f}")

    # —— Persist learned per-league threshold if enabled and we have enough samples ——
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        n_samples = int(len(y))
        if AUTO_DRAW_THRESHOLD and n_samples >= min_samples_required and not USE_LOCKED_DRAW_THRESHOLDS:
            payload = {
                "league": str(league_name),
                "threshold": float(deploy_thr),
                "mode": deploy_mode,
                "model_type": model_type,
                "precision_target": float(precision_target_resolved),
                "n_samples": n_samples,
                "draw_rate": float(draw_rate),
                # OOF metrics
                "oof_roc_auc": float(oof_auc) if 'oof_auc' in locals() else None,
                "oof_pr_auc": float(pr_auc) if 'pr_auc' in locals() else None,
                "oof_precision": float(oof_prec),
                "oof_recall": float(oof_rec),
                "oof_f1": float(oof_f1),
                # Confusion matrix at selected threshold
                "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
                # Precision among top-k most draw-like fixtures (list)
                "precision_at_k_list": precision_at_k_list,
                # Features actually used at fit time
                "features_used": trained_feature_cols,
                "feature_count": int(len(trained_feature_cols)),
                "generated_at": datetime.utcnow().isoformat() + "Z",
            }
            # Ensure baseline_blend and a reference to the model bundle path are present
            BLEND_DEFAULT = float(os.getenv("BASELINE_BLEND_DEFAULT", "0.70"))
            payload["baseline_blend"] = float(BLEND_DEFAULT)
            try:
                payload["model_bundle_path"] = _draw_bundle_joblib_path(league_name)
            except Exception:
                pass
            # Compute and store best_k_pct for convenience
            try:
                if precision_at_k_list:
                    _best = max(precision_at_k_list, key=lambda d: d.get("precision", float("-inf")))
                    payload["best_k_pct"] = float(_best.get("k_pct", 0.10))
            except Exception:
                pass
            thr_path = _threshold_store_path(str(league_name))
            with open(thr_path, "w") as fh:
                json.dump(payload, fh, indent=2)
            print(f"💾 Saved learned draw threshold → {thr_path} ({deploy_mode} = {deploy_thr:.3f})")
            # Save calibrated model bundle alongside the threshold JSON
            save_draw_model_bundle(league_name, clf, trained_feature_cols, model_type=model_type)
        elif USE_LOCKED_DRAW_THRESHOLDS:
            print("ℹ️  Skipping save: USE_LOCKED_DRAW_THRESHOLDS is True (constants).")
        else:
            print(f"ℹ️  Skipping save: only {n_samples} samples (< {min_samples_required} required).")
    except Exception as _e:
        print(f"⚠️  Could not persist learned threshold: {_e}")

    # --------------------------------------------------------------
    # === shuffle‑label leak probe =================================
    # --------------------------------------------------------------
    y_shuf = np.random.permutation(y.values)
    # Use probabilistic OOF predictions and ROC‑AUC against shuffled labels.
    # For a leak‑free model this should be ~0.5 (random). We flag only if it
    # meaningfully exceeds random chance.
    try:
        auc_shuf = roc_auc_score(y_shuf, oof_proba)
    except Exception:
        # If AUC cannot be computed (e.g., only one class present), treat as random
        auc_shuf = 0.5
    if auc_shuf > 0.58:
        print(f"❌  Shuffle‑label AUC {auc_shuf:.3f} > 0.58 → leakage suspected.")
    else:
        print(f"✅  Shuffle‑label AUC {auc_shuf:.3f} ≈ random (no leak).")

    # ---- top |corr| with shuffled labels (index‑aligned) ---------
    y_shuf_pd = pd.Series(y_shuf, index=X.index)
    shuffle_corr = (
        X.corrwith(y_shuf_pd, method="pearson")
          .abs()
          .sort_values(ascending=False)
    )

    # ── auto‑drop high‑corr leakage suspects ──
    SUSPECT_CUTOFF = 0.08  # was 0.05; tighten so we don’t false‑flag stable prematch features
    raw_suspect = shuffle_corr[shuffle_corr > SUSPECT_CUTOFF].index.tolist()
    # Never drop the core prematch features (even if noisy in small samples)
    suspect = [c for c in raw_suspect if c not in SAFE_CORE]
    if suspect:
        print(f"🚫  Auto‑dropping {len(suspect)} of {len(raw_suspect)} leak‑suspect cols (SAFE_CORE protected):", suspect[:8], "…")
        X.drop(columns=suspect, inplace=True, errors="ignore")
        feature_cols = [c for c in feature_cols if c not in suspect]

    print("\n[leak‑probe] Top‑20 |corr| with shuffled FTR:")
    print(shuffle_corr.head(20).to_string())

    # Persist full vector for deeper inspection
    os.makedirs("ModelStore", exist_ok=True)
    _ltag = str(league_name).replace(" ", "_")
    out_csv = os.path.join("ModelStore", f"{_ltag}_draw_shuffle_corr.csv")
    shuffle_corr.to_csv(out_csv)
    print(f"Full shuffle-corr written to {out_csv}")

    # —— Model summary: features, importances, threshold ——
    used_n = len(trained_feature_cols)

    def _extract_feature_importances_from_calibrator(calibrated, n_features):
        mats = []
        # sklearn>=1.6 exposes `estimator`; older had `base_estimator`
        if hasattr(calibrated, "calibrated_classifiers_") and calibrated.calibrated_classifiers_:
            for cc in calibrated.calibrated_classifiers_:
                est = getattr(cc, "estimator", None)
                if est is None:
                    est = getattr(cc, "base_estimator", None)
                if est is not None and hasattr(est, "feature_importances_"):
                    mats.append(est.feature_importances_)
        if mats:
            try:
                return np.vstack(mats).mean(axis=0)
            except Exception:
                pass
        # Fallback: single underlying estimator path
        est = getattr(calibrated, "estimator", None)
        if est is None:
            est = getattr(calibrated, "base_estimator_", None)
        if est is not None and hasattr(est, "feature_importances_"):
            fi = np.asarray(est.feature_importances_)
            return fi[:n_features]
        return None

    importances_arr = _extract_feature_importances_from_calibrator(clf, len(trained_feature_cols))

    if importances_arr is not None:
        imp_ser = pd.Series(importances_arr, index=trained_feature_cols).sort_values(ascending=False)
        top10 = imp_ser.head(10)
        print("\n📊 Feature summary:")
        print(f"• Features used: {used_n}")
        print("• Top-10 importances:")
        for k, v in top10.items():
            print(f"   - {k}: {v:.4f}")
        os.makedirs("ModelStore", exist_ok=True)
        _ltag = str(league_name).replace(" ", "_")
        top10.to_csv(os.path.join("ModelStore", f"{_ltag}_draw_feature_importances_top10.csv"))
    else:
        print(f"\n📊 Feature summary: Features used: {used_n} (importances unavailable)")

    print(f"• Deployment threshold [{getattr(clf, 'threshold_mode_', 'youden')}]: {getattr(clf, 'best_threshold_', 0.5):.3f}")

    # —— Persist the actually-used feature list for reproducibility ——
    try:
        league_tag = str(training_data.get("league_name", "generic")).replace(" ", "_")
    except Exception:
        league_tag = str(league_name).replace(" ", "_")
    _feat_path = os.path.join(MODEL_DIR, f"{league_tag}_draw_features.txt")
    try:
        with open(_feat_path, "w") as _fh:
            _fh.write("\n".join(trained_feature_cols))
    except Exception as _e:
        print(f"⚠️  Could not persist draw features to {_feat_path}: {_e}")

    # Save full importances if available
    if importances_arr is not None:
        _ltag = str(league_name).replace(" ", "_")
        pd.Series(importances_arr, index=trained_feature_cols).to_csv(
            os.path.join("ModelStore", f"{_ltag}_draw_feature_importances.csv")
        )

    # Package scores for sweep usage
    precision_at_k_map = {float(d.get("k_pct", 0.0)): float(d.get("precision", float("nan"))) for d in precision_at_k_list}
    scores = {
        "oof_auc": float(oof_auc) if 'oof_auc' in locals() else None,
        "pr_auc": float(pr_auc) if 'pr_auc' in locals() else None,
        "oof_precision": float(oof_prec),
        "oof_recall": float(oof_rec),
        "oof_f1": float(oof_f1),
        "precision_at_k": precision_at_k_map,   # {k_pct: precision}
        "deploy_thr": float(deploy_thr),
        "deploy_mode": deploy_mode,
        "trained_feature_cols": trained_feature_cols,
        "model_type": model_type,
    }

    return clf, scores

# ---- bridge used by the pipeline (project API) ----
def ensure_draw_bundle(df, league: str, row_cap: int | None = None, n_jobs: int = 1, perm_importance: bool = False):
    """
    Train a calibrated P(draw) model and return a bundle the pipeline can consume:
      {"model": fitted_calibrated_model, "feature_selector": fn(DataFrame)->X}
    """
    import importlib, numpy as np, pandas as pd

    # 0) make sure FTR exists and restrict to completed rows
    df = _ensure_ftr_col(df.copy())
    df = _filter_completed_training_rows(df)

    # 1) optional: add engineered draw features from the pipeline if available
    try:
        _pipe = importlib.import_module("_baseline_ftr_pipeline")
        if hasattr(_pipe, "ensure_draw_ready_features"):
            df = _pipe.ensure_draw_ready_features(df)
    except Exception:
        pass

    # 2) pick a safe prematch allowlist (no leaks)
    feats = _resolve_whitelist(df, SAFE_PREMATCH_WHITELIST, ALIAS_MAP)

    # 3) train calibrated model (GBM backbone)
    clf, scores = train_draw_classifier(
        df, feats, league_name=league,
        threshold_mode=None,   # constants decide; or "f1"/"precision"
        precision_target=None, # constants decide; e.g. 0.60
        top_k_pcts=[0.10], model_type="gbm"
    )

    # 4) feature selector used at inference
    used = list(scores.get("trained_feature_cols", feats))
    def _selector(d: pd.DataFrame):
        X = d.reindex(columns=used, fill_value=np.nan)
        return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return {"model": clf, "feature_selector": _selector}

# ----------------------------------------------------------------------
# Optional CLI usage:
#   python train_draw_classifier.py --league "England Premier League" \
#                                   --data-dir Matches/England_Premier_League
# ----------------------------------------------------------------------
import argparse, os, pathlib, textwrap
# --- dynamic import because the module name used to start with a digit ---
import importlib
# Try the legacy numeric module first; if it fails for any reason (missing, SyntaxError, etc.),
# fall back to the renamed modules.
try:
    _pipeline = importlib.import_module("00_baseline_ftr_pipeline")
except Exception:
    try:
        _pipeline = importlib.import_module("_baseline_ftr_pipeline")
    except Exception:
        _pipeline = importlib.import_module("baseline_ftr_pipeline")

# pull the required symbols (regardless of which import succeeded)
load_multiple_seasons = getattr(_pipeline, "load_multiple_seasons", None)
if load_multiple_seasons is None:
    raise ImportError("load_multiple_seasons not found in pipeline module")
DRAW_FEATURES = getattr(_pipeline, "DRAW_FEATURES", [])

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Quick utility to retrain ONLY the draw‑classifier in isolation.
            It loads all seasons for the chosen league, uses a safe
            allowlisted pre‑match feature set (no global strip_leaks),
            prints shuffle‑leak diagnostics, and writes:
                • draw_feature_importances.csv
                • draw_shuffle_corr.csv
            to the local ModelStore.
        """)
    )
    ap.add_argument(
        "--league",
        default="England Premier League",
        help="League name used for path resolution (default: %(default)s)"
    )
    ap.add_argument(
        "--data-dir",
        default="Matches",
        help="Root folder containing league sub‑directories (default: %(default)s)"
    )
    ap.add_argument(
        "--threshold-mode", choices=["youden", "f1", "precision"], default="f1",
        help="Which threshold strategy to use for deployment: 'youden', 'f1', or 'precision' (default: %(default)s)"
    )
    ap.add_argument(
        "--precision-target", type=float, default=0.60,
        help="Target precision used when --threshold-mode=precision (default: %(default)s)"
    )
    ap.add_argument(
        "--topk-pct", type=float, action="append", default=[0.10],
        help="Top-k fraction(s) for rank-based precision diagnostics and JSON. Can be repeated, e.g. --topk-pct 0.05 --topk-pct 0.10 --topk-pct 0.20 (default: 0.10)"
    )
    ap.add_argument(
        "--blend-sweep", type=str, default="",
        help="Comma-separated list of prev-season blend weights to try, e.g. '0.6,0.7,0.8'. If empty, skip sweep."
    )
    ap.add_argument(
        "--topk-pcts", type=str, default="0.05,0.10,0.20",
        help="Comma-separated top-k percentages to score precision at (default: 0.05,0.10,0.20)."
    )
    ap.add_argument(
        "--model", choices=["gbm", "histgb"], default="gbm",
        help="Model backbone: classic GradientBoosting ('gbm') or HistGradientBoosting ('histgb') with early stopping (default: %(default)s)"
    )
    args = ap.parse_args()

    # Resolve league folder robustly
    from pathlib import Path
    league_tag = args.league.replace(" ", "_")
    base = Path(args.data_dir)

    # Case 1: --data-dir already points at the league folder
    if base.name in (args.league, league_tag) and base.exists():
        league_folder = base
    # Case 2: try common subpaths under --data-dir
    else:
        candidates = [
            base / league_tag,
            base / args.league,
        ]
        # If user passed a path that already ends with the league tag/name, accept it
        if str(base).lower().endswith(league_tag.lower()) or str(base).lower().endswith(args.league.lower()):
            candidates.insert(0, base)
        league_folder = None
        for cand in candidates:
            if Path(cand).exists():
                league_folder = Path(cand)
                break
        if league_folder is None:
            raise SystemExit(
                "❌  League folder not found. Tried: "
                + ", ".join(str(c) for c in candidates)
            )

    print(f"ℹ️  Using league folder: {league_folder}")

    # Quick visibility into enrichment coverage
    try:
        _report_press_intensity_coverage(league_folder)
    except Exception as _e:
        print(f"ℹ️  coverage check skipped: {_e}")

    print(f"ℹ️  Loading CSVs from  {league_folder}")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Parse sweep grid and top-k list
    if getattr(args, "blend_sweep", "").strip():
        try:
            blend_grid = [float(x) for x in args.blend_sweep.split(",") if x.strip()]
        except Exception:
            print(f"⚠️  Could not parse --blend-sweep='{args.blend_sweep}', skipping sweep.")
            blend_grid = []
    else:
        blend_grid = []

    try:
        topk_pcts = [float(x) for x in (args.topk_pcts.split(",") if isinstance(args.topk_pcts, str) else args.topk_pcts)]
    except Exception:
        # fall back to legacy --topk-pct list
        topk_pcts = [float(x) for x in (args.topk_pct or [0.10])]

    matches_dir = str(league_folder)

    # Helper to load data, prep, and train once (returns (clf, scores))
    def _train_and_score_once():
        df_local = load_multiple_seasons(league_folder)
        if df_local.empty:
            raise SystemExit(f"❌  No CSVs under {league_folder}")
        if "match_date" in df_local.columns:
            df_local = df_local.sort_values("match_date")
        df_local = _ensure_ftr_col(df_local)
        try:
            df_local = _pipeline.ensure_draw_ready_features(df_local)
        except Exception as e:
            print(f"ℹ️  ensure_draw_ready_features skipped: {e}")

        # Add press z-score features (within-season normalisation)
        try:
            df_local = _add_press_zscores(df_local)
        except Exception as _e:
            print(f"ℹ️  press z-score enrichment skipped: {_e}")

        # Restrict to completed fixtures only for training
        try:
            df_local = _filter_completed_training_rows(df_local)
        except Exception as _e:
            print(f"ℹ️ completed‑status train filter skipped: {_e}")

        return train_draw_classifier(
            df_local.copy(),
            DRAW_FEATURES,
            league_name=args.league,
            threshold_mode=args.threshold_mode,
            precision_target=args.precision_target,
            top_k_pcts=topk_pcts,
            model_type=args.model,   # ⬅️ NEW
        )

    chosen_alpha = None
    sweep_log = []
    chosen_scores = None

    if blend_grid:
        best = None
        k0 = float(topk_pcts[0]) if topk_pcts else 0.10
        for alpha in blend_grid:
            print(f"\n🔁 Baseline-blend sweep: trying prev-season weight α={alpha:.2f}")
            # Rebuild baseline and intensity for this alpha (overwrite to ensure consistency)
            try:
                ensure_press_intensity_on_disk(
                    matches_dir,
                    force=False,
                    baseline_blend=alpha,
                    use_cache=True,
                    overwrite_baseline=True,
                    overwrite_intensity=True,
                )
            except TypeError:
                # Backward-compat if ensure_press_intensity_on_disk signature lacks new params
                ensure_press_intensity_on_disk(matches_dir)

            _, scores = _train_and_score_once()
            sel = float(scores.get("precision_at_k", {}).get(k0, float("nan")))
            row = {
                "alpha": float(alpha),
                "precision_at_k": scores.get("precision_at_k", {}),
                "pr_auc": scores.get("pr_auc"),
                "oof_auc": scores.get("oof_auc"),
                "f1": scores.get("oof_f1"),
            }
            sweep_log.append(row)

            if best is None:
                best = (alpha, scores)
            else:
                prev_sel = float(best[1].get("precision_at_k", {}).get(k0, -1))
                prev_pr  = float(best[1].get("pr_auc", float("nan")))
                if (sel, scores.get("pr_auc", float("nan"))) > (prev_sel, prev_pr):
                    best = (alpha, scores)

        print("\n📊 Baseline-blend sweep results (α = prev-season weight):")
        for row in sweep_log:
            short = ", ".join([f"@{int(k*100)}%={float(row['precision_at_k'].get(k, float('nan'))):.3f}" for k in topk_pcts])
            print(f"  α={row['alpha']:.2f} | PR-AUC={float(row['pr_auc']):.3f} | {short}")

        chosen_alpha, chosen_scores = best
        print(f"\n✅ Chosen α={chosen_alpha:.2f} (by precision@{int(k0*100)}% then PR-AUC)")

        # Rebuild once more with the chosen α so final model aligns
        try:
            ensure_press_intensity_on_disk(
                matches_dir,
                force=False,
                baseline_blend=chosen_alpha,
                use_cache=True,
                overwrite_baseline=True,
                overwrite_intensity=True,
            )
        except TypeError:
            ensure_press_intensity_on_disk(matches_dir)

    # Final training run (aligned with chosen alpha if we swept)
    clf, final_scores = _train_and_score_once()

    # Persist chosen alpha + sweep log into the threshold JSON (append/update)
    try:
        thr_json = _threshold_store_path(args.league)
        if os.path.exists(thr_json):
            with open(thr_json, "r") as fh:
                payload = json.load(fh)
        else:
            payload = {}
        if chosen_alpha is not None:
            payload["baseline_blend"] = float(chosen_alpha)
            payload["blend_sweep"] = sweep_log
        else:
            # set a sensible default if not already present
            payload.setdefault("baseline_blend", float(os.getenv("BASELINE_BLEND_DEFAULT", "0.70")))
        # store the final precision_at_k for convenience
        payload["precision_at_k"] = {str(k): float(final_scores.get("precision_at_k", {}).get(k)) for k in topk_pcts}
        # ensure list-of-dicts is present for overlay's best_k logic
        pak_map = final_scores.get("precision_at_k", {})
        if pak_map:
            payload.setdefault("precision_at_k_list", [
                {"k_pct": float(k), "k": None, "precision": float(v)} for k, v in pak_map.items()
            ])
            # store explicit best_k_pct as well
            try:
                best_k = max(pak_map.items(), key=lambda kv: kv[1])[0]
                payload.setdefault("best_k_pct", float(best_k))
            except Exception:
                pass
        payload["model_type"] = args.model
        # include a reference to the joblib bundle path
        try:
            payload["model_bundle_path"] = _draw_bundle_joblib_path(args.league)
        except Exception:
            pass
        # write back
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(thr_json, "w") as fh:
            json.dump(payload, fh, indent=2)
        # Final CLI summary of artefact paths
        try:
            bundle_path_cli = _draw_bundle_joblib_path(args.league)
            print("\n✅ Artefacts written:")
            print(f"   • Threshold JSON: {thr_json}")
            print(f"   • Draw bundle:    {bundle_path_cli}")
        except Exception as _e2:
            print(f"ℹ️  Artefact summary print skipped: {_e2}")
    except Exception as _e:
        print(f"⚠️  Could not append blend sweep info to threshold JSON: {_e}")