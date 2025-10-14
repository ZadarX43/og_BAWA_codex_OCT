# constants.py
# ------------------------------------------------------------------
import os

print(f"🔧 constants loaded from {__file__}")
# --- Forward declarations for static analysers (avoid undefined before use) ---
if "PER_LEAGUE" not in globals():
    PER_LEAGUE: dict[str, dict[str, str]] = {}
if "DRAW_THRESHOLD_MODE_PER_LEAGUE" not in globals():
    DRAW_THRESHOLD_MODE_PER_LEAGUE: dict[str, str] = {}

# Helper to parse boolean-like env flags
_DEF_TRUE = ("1", "true", "yes", "y", "on")
_DEF_FALSE = ("0", "false", "no", "n", "off")

# --- Canonical FTR feature allowlist (pre-match + OOF + λ + odds) -------------
# Used by _baseline_ftr_pipeline._resolve_ftr_allowlist() when present.
# No need for FTR_FEATURES_PATH: this is the single source of truth.
ALLOWED_FTR = [
    # OOF specialists (stackers)
    "oof_prob_over25","oof_prob_btts",
    "oof_prob_home_over15tg","oof_prob_away_over15tg",
    "oof_prob_home_fts","oof_prob_away_fts",
    # λ̂ and Poisson helpers
    "lambda_home","lambda_away","exp_goals_sum","p00_est",
    "lam_diff","lam_parity","tg_diff","fts_diff","attack_lean",
    # Pre-match strength / context
    "Pre-Match PPG (Home)","Pre-Match PPG (Away)","ppg_diff",
    "Home Team Pre-Match xG","Away Team Pre-Match xG","xg_diff_abs",
    "elo_diff","rest_diff","ewma_ppg_diff",
    # Press-intensity (rolling + z)
    "rolling5_home_press_intensity","rolling5_away_press_intensity","rolling5_press_intensity_diff",
    "rolling5_home_press_z","rolling5_away_press_z","rolling5_press_z_diff",
    # Markets & priors (leak-safe pre-match)
    "over_25_percentage_pre_match","btts_percentage_pre_match",
    # Odds / parity
    "odds_ft_home_team_win","odds_ft_draw","odds_ft_away_team_win",
    "draw_implied","implied_prob_diff","odds_diff","odds_parity","odds_skew",
    # Draw bridge signal
    "prob_draw_model",
]
# Back-compat alias some earlier helpers look for
ALLOWED_FTR_FEATURES = list(ALLOWED_FTR)

def _env_bool(name: str, default: str = "1") -> bool:
    val = os.getenv(name, default)
    if val is None:
        return default in _DEF_TRUE
    v = str(val).strip().lower()
    if v in _DEF_TRUE:
        return True
    if v in _DEF_FALSE:
        return False
    return default.strip().lower() in _DEF_TRUE
# ------------------------------------------------------------------
FORCE_LOCKED = False  # optimiser is allowed for all leagues except those still in the lock dicts
ENABLE_SIDE_PROB_ABLATION = False  # strip bookmaker-derived side‑prob columns during training

VAR_THRESH_NUM = 1e-5             # low‑variance pruning threshold for numeric features
# Frozen / “good-enough” probability cut-offs that shouldn’t get
# overwritten by the automatic threshold-tuner.  Add or adjust as
# you confirm stronger numbers.

# NOTE:
#  • Trainer ignores these when USE_LOCKED_DRAW_THRESHOLDS is False (default).
#  • Inference uses learned per-league JSON thresholds when available.
#  • Some utilities may read LOCKED_THRESHOLDS[league]["draw"] only as a fallback
#    default gate if no JSON exists. Keep values sensible but do not rely on them.

LOCKED_THRESHOLDS: dict[str, dict[str, float]] = {
    "Portugal Liga": {
        "over25": 0.32,
        "btts":   0.56,
        "draw": 0.02,  # Note: "draw" here is only for deployment locks, not for historical priors
        "home": 0.380,
        "away": 0.395
    },
    "Italy Serie A": {
        "over25": 0.30,
        "home": 0.405,
        "away": 0.380
    },
    "USA MLS": {
        "over25": 0.34,
        "draw": 0.02,  # Note: "draw" here is only for deployment locks, not for historical priors
        "home": 0.395,
        "away": 0.430
    },
    "Europa League": {
        "over25": 0.32,
        "draw": 0.02,  # Note: "draw" here is only for deployment locks, not for historical priors
        "home": 0.395,
        "away": 0.385
    },
    "Champions League": {
        "over25": 0.30,
        "draw": 0.05,  # Note: "draw" here is only for deployment locks, not for historical priors
        "home": 0.38,
        "away": 0.42
    },
    "Spain La Liga": {
        "btts":   0.55,
        "home": 0.410,
        "away": 0.385
    },
    "England Premier League": {
        "draw": 0.02,  # Note: "draw" here is only for deployment locks, not for historical priors
        "home": 0.34,
        "away": 0.34
    },
    "Germany Bundesliga": {
        "home": 0.375,
        "away": 0.370
    }
}

HISTORICAL_DRAW_RATE = {
    "England Premier League": 0.27,
    "Germany Bundesliga": 0.25,
    "Spain La Liga": 0.24,
    "Italy Serie A": 0.26,
    "Portugal Liga": 0.27,
    "USA MLS": 0.21,
    "Europa League": 0.23,
    "Champions League": 0.22,
}

HISTORICAL_DRAW_RATE_DEFAULT = 0.26

# --- unlock Premier League FTR thresholds ---
LOCKED_THRESHOLDS.pop("England Premier League", None)

# ───── Fixed FTR mixing parameters (α / β / cap) ────────────────  
# Freeze α, β, and the overlay cap when provided.
# ───── Fixed FTR mixing parameters (α / β / cap) ────────────────  
# Freeze α, β, and the overlay cap when provided.
# ─────────── Locked α/β/cap per‑league (post‑debug‑draw) ────────────
LOCKED_FTR_MIX: dict[str, dict[str, float]] = {
    "England Premier League": {"alpha": 1.40, "beta": 0.10, "cap": 0.50},
    "Germany Bundesliga": {"alpha": 2.00, "beta": 0.15, "cap": 0.48},
    "Portugal Liga": {"alpha": 1.60, "beta": 0.10, "cap": 0.60},
    "Spain La Liga": {"alpha": 1.80, "beta": 0.10, "cap": 0.45},
    "Europa League": {"alpha": 1.30, "beta": 0.20, "cap": 0.45},
    "USA MLS": {"alpha": 1.20, "beta": 0.05, "cap": 0.50},
    "Italy Serie A": {"alpha": 2.00, "beta": 0.15, "cap": 0.40},
    "Champions League": {"alpha": 2.00, "beta": 0.15, "cap": 0.45},
    # "Italy Serie A":          {"alpha": 0.90, "beta": 0.00, "cap": 0.40},
    # "Europa League":          {"alpha": 1.30, "beta": 0.00, "cap": 0.40},
    # "USA MLS":                {"alpha": 0.70, "beta": 0.00, "cap": 0.40},
    # "Germany Bundesliga":     {"alpha": 2.20, "beta": 0.00, "cap": 0.40},
    # "Portugal Liga":          {"alpha": 2.40, "beta": 0.00, "cap": 0.40},
    # "Spain La Liga":          {"alpha": 2.70, "beta": 0.00, "cap": 0.40},
    # "Champions League":       {"alpha": 1.90, "beta": 0.00, "cap": 0.40},
}
# Ensure Champions League has a sensible starting mix
LOCKED_FTR_MIX.setdefault("Champions League", {"alpha": 1.00, "beta": 0.08, "cap": 0.45})

# Default gate-scale for mixers if not provided explicitly
GATE_SCALE_DEFAULT = float(os.getenv("GATE_SCALE_DEFAULT", "0.60"))

# Normalise LOCKED_FTR_MIX to ensure gate_scale is present
for _lg, _mix in list(LOCKED_FTR_MIX.items()):
    if "gate_scale" not in _mix:
        _mix["gate_scale"] = GATE_SCALE_DEFAULT

# Draw-prone tuning parameters per league
DRAW_THRESHOLD_PARAMS: dict[str, dict[str, float]] = {
    'Champions League': {
        'ppg_diff': 0.6,
        'xg_diff': 0.4,
        'implied_prob_diff': 0.25,
        'odds_diff': 3.25
    },
    'England Premier League': {
        'ppg_diff': 0.70,
        'xg_diff': 0.35,
        'implied_prob_diff': 0.35,
        'odds_diff': 2.4
    },
    'Europa Conference': {
        'ppg_diff': 0.75,
        'xg_diff': 0.40,
        'implied_prob_diff': 0.35,
        'odds_diff': 2.60
    },
    'Europa League': {
        'ppg_diff': 0.85,
        'xg_diff': 0.45,
        'implied_prob_diff': 0.4,
        'odds_diff': 2.7
    },
    'Germany Bundesliga': {
        'ppg_diff': 0.58,
        'xg_diff': 0.50,
        'implied_prob_diff': 0.36,
        'odds_diff': 2.50
    },
    'Italy Serie A': {
        'ppg_diff': 0.65,
        'xg_diff': 0.45,
        'implied_prob_diff': 0.4,
        'odds_diff': 3.2
    },
    'Portugal Liga': {
        'ppg_diff': 0.60,
        'xg_diff': 0.4,
        'implied_prob_diff': 0.20,
        'odds_diff': 2.0
    },
    'Spain La Liga': {
        'ppg_diff': 0.80,
        'xg_diff': 0.38,
        'implied_prob_diff': 0.28,
        'odds_diff': 2.40
    },

    'USA MLS': {
        'ppg_diff': 0.8,
        'xg_diff': 0.525,
        'implied_prob_diff': 0.35,
        'odds_diff': 2.8
    }
}

# ----- default mix so unseen leagues don't explode -----

# ------------------------------------------------------------------
# Centralised path to the on‑disk ModelStore for all modules
# ------------------------------------------------------------------


# Global prior‑shrink parameter for draw calibration
N0_SHRINK = 30   # lower = stronger shrink toward historical draw rate

# Per‑league prior strength (matches); falls back to N0_SHRINK when absent
N0_SHRINK_PER_LEAGUE = {
    "England Premier League": 6,
    "Germany Bundesliga":     10,
    "USA MLS":                15,
    "Portugal Liga":          10,
    "Europa League": 8,
    # Add more leagues here as needed
}

DEFAULT_FTR_MIX = {"alpha": 1.0, "beta": 0.10, "cap": 0.45}
MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "ModelStore")
)

# --- Prefer on-disk YAML/JSON artefacts over locks when both exist (env-guarded)
PREFER_YAML_OVER_LOCKS = _env_bool("PREFER_YAML_OVER_LOCKS", "1")

try:
    # Check for per-league mix YAMLs that would supersede LOCKED_FTR_MIX
    for _league in list(LOCKED_FTR_MIX.keys()):
        _tag = _league.replace(" ", "_")
        _mix_yaml = os.path.join(MODEL_DIR, f"{_tag}_mix.yaml")
        if os.path.exists(_mix_yaml):
            if PREFER_YAML_OVER_LOCKS:
                LOCKED_FTR_MIX.pop(_league, None)
                print(f"ℹ️  Using YAML mix over LOCKED for '{_league}' → {_mix_yaml}")
            else:
                print(f"⚠️  YAML mix exists for '{_league}' but LOCKED_FTR_MIX will be used; set PREFER_YAML_OVER_LOCKS=1 to prefer YAML.")

    # Check for learned draw-threshold JSONs that would supersede LOCKED_THRESHOLDS
    for _league in list(LOCKED_THRESHOLDS.keys()):
        _tag = _league.replace(" ", "_")
        _thr_json = os.path.join(MODEL_DIR, f"{_tag}_draw_threshold.json")
        if os.path.exists(_thr_json):
            if PREFER_YAML_OVER_LOCKS:
                # Do not delete thresholds for other markets; just note that draw gate will come from JSON in training/inference
                print(f"ℹ️  Learned draw-threshold JSON present for '{_league}' → {_thr_json}; loader will prefer it over LOCKED.")
            else:
                print(f"⚠️  Draw-threshold JSON exists for '{_league}' but LOCKED thresholds may be consulted; set PREFER_YAML_OVER_LOCKS=1 to prefer JSON.")
except Exception:
    # Never crash at import time over filesystem checks
    pass

# --- Self-check: warn if any locked beta exceeds the loader cap ---------------
MIX_BETA_CAP = float(os.getenv("MIX_BETA_CAP", "1.00"))  # default off
try:
    _warn_cap = os.getenv("WARN_MIX_BETA_CAP", "0") == "1"
    if _warn_cap:
        for _league, _mix in list(LOCKED_FTR_MIX.items()):
            try:
                _b = float(_mix.get("beta", 0.0))
                if _b > MIX_BETA_CAP:
                    print(
                        f"⚠️  LOCKED_FTR_MIX beta {_b:.2f} > cap {MIX_BETA_CAP:.2f} for '{_league}' — consider raising MIX_BETA_CAP or updating YAML.")
            except Exception:
                pass
except Exception:
    # Never crash at import time for diagnostics
    pass

# --- Self-check: suspicious neutral FTR thresholds in LOCKED_THRESHOLDS -------
try:
    EPS = float(os.getenv("NEUTRAL_THRESH_EPS", "0.005"))
    NEUTRALS = [float(x) for x in os.getenv("NEUTRAL_THRESHOLDS", "0.34,0.3333,0.333,0.35").split(",") if x.strip()]

    def _near_neutral(v: float):
        for n in NEUTRALS:
            try:
                if abs(float(v) - n) <= EPS:
                    return n
            except Exception:
                pass
        return None

    for _league, _thr in list(LOCKED_THRESHOLDS.items()):
        h = _thr.get("home"); a = _thr.get("away")
        if h is None or a is None:
            continue
        try:
            hv = float(h); av = float(a)
        except Exception:
            continue
        if abs(hv - av) <= EPS:
            nn = _near_neutral(hv)
            if nn is not None:
                print(
                    f"⚠️  LOCKED_THRESHOLDS for '{_league}' look neutralish: home={hv:.3f}, away={av:.3f} ~ {nn:.3f} — consider relying on learned per-league thresholds or adjusting locks.")
except Exception:
    # Never crash at import time for diagnostics
    pass

# --- Self-check: missing learned draw JSON for active/frequently used leagues -
try:
    # Define "active" leagues via env (comma-separated). If not provided, fall back
    # to a sensible union of known sets (locks + threshold modes + historical rates).
    raw_active = os.getenv("ACTIVE_LEAGUES", "").strip()
    if raw_active:
        ACTIVE = [s.strip() for s in raw_active.split(",") if s.strip()]
    else:
        ACTIVE = sorted(set(list(LOCKED_FTR_MIX.keys()) +
                            list((globals().get("DRAW_THRESHOLD_MODE_PER_LEAGUE", {}) or {}).keys()) +
                            list(HISTORICAL_DRAW_RATE.keys())))

    for _league in ACTIVE:
        _tag = _league.replace(" ", "_")
        _thr_json = os.path.join(MODEL_DIR, f"{_tag}_draw_threshold.json")
        if not os.path.exists(_thr_json):
            print(
                f"⚠️  No learned draw-threshold JSON for '{_league}' → {_thr_json}. "
                f"Prioritise training: python train_draw_classifier.py --league \"{_league}\" --data-dir Matches/{_tag}")
except Exception:
    # Never crash at import time for diagnostics
    pass

# ==================================================================
# Draw threshold learning & deployment defaults (league-aware)
# ==================================================================
# If True, training code should learn a per-league draw threshold
# from OOF predictions and persist it to ModelStore/<league>_draw_threshold.json.
# Inference then loads the saved threshold; if missing, it should
# fall back to DRAW_THRESHOLD_FALLBACK or a model-embedded default.
AUTO_DRAW_THRESHOLD: bool = True

# Default selection rule for deployment threshold when training.
# Allowed: "f1", "precision", "youden".
DRAW_THRESHOLD_MODE_DEFAULT: str = "f1"

# If DRAW_THRESHOLD_MODE_DEFAULT == "precision", this is the target precision
# the trainer should try to meet or exceed when selecting the threshold.
DRAW_PRECISION_TARGET_DEFAULT: float = 0.60

# Don’t update/save a per-league threshold unless you have at least this many
# OOF samples for that league.  Prevents noisy early-season calibration.
DRAW_THRESHOLD_MIN_SAMPLES: int = 200

# Hard fallback used when no learned threshold exists and no league lock applies.
DRAW_THRESHOLD_FALLBACK: float = 0.05

# Some code paths may still consult LOCKED_THRESHOLDS[league]["draw"].
# When False, draw locks should be ignored unless FORCE_LOCKED is True.
USE_LOCKED_DRAW_THRESHOLDS: bool = False

# Optional per-league default modes; trainer can override globally via CLI.
# Example: make Serie A default to precision targeting; EPL to F1.
DRAW_THRESHOLD_MODE_PER_LEAGUE: dict[str, str] = {
    "England Premier League": "f1",
    "Italy Serie A": "precision",
    "Spain La Liga": "f1",
    "Germany Bundesliga": "f1",
    "Portugal Liga": "f1",
    "USA MLS": "f1",
    "Europa League": "f1",
    "Champions League": "f1",
}

# Optional: minimum samples per league before trusting a learned threshold.
DRAW_MIN_SAMPLES_PER_LEAGUE: dict[str, int] = {
    "England Premier League": 400,
    "Germany Bundesliga":     300,
    "Spain La Liga":          300,
    "Italy Serie A":          300,
    "Portugal Liga":          250,
    "USA MLS":                250,
}

# Backward-compatibility alias (common typo: letter-O vs zero)
NO_SHRINK_PER_LEAGUE = N0_SHRINK_PER_LEAGUE
# --- Side-model toggles (default ON unless explicitly disabled) -------------
globals().setdefault("ENABLE_SPEC_OVER25", 1)
globals().setdefault("ENABLE_SPEC_BTTS",   1)
globals().setdefault("ENABLE_SPEC_FTS",    1)
globals().setdefault("ENABLE_SPEC_GE2TG",  1)

# --- Per-league defaults for FTR calibration & gating -----------------------
import os as _os, json as _json, re as _re
try:
    from pathlib import Path as _Path
except Exception:
    _Path = None  # portable fallback

# If a PER_LEAGUE already exists, keep it; otherwise create and seed defaults
PER_LEAGUE = globals().get('PER_LEAGUE', {})

# ⚠️ EDIT these as you finalize tuning; these are current working values
PER_LEAGUE.setdefault("Italy Serie A", {
    "FTR_CAL_MODE":     "isotonic",
    "FTR_CONF_SHARPEN": "1.48",     # gamma
    "FTR_PICK_GATE":    "0.435"     # gate
})
PER_LEAGUE.setdefault("France Ligue 1", {
    "FTR_CAL_MODE":     "isotonic",
    "FTR_CONF_SHARPEN": "1.55",
    "FTR_PICK_GATE":    "0.445"
})
PER_LEAGUE.setdefault("Germany Bundesliga", {
    "FTR_CAL_MODE":     "isotonic",
    "FTR_CONF_SHARPEN": "1.44",
    "FTR_PICK_GATE":    "0.455"
})

def _slug(_league: str) -> str:
    return _re.sub(r"[^a-z0-9]+", "_", str(_league).lower()).strip("_")

def apply_per_league_env(league_name: str, prefer_tuned: bool = True) -> None:
    """
    Set env vars for the given league. If prefer_tuned=True and a tuned JSON exists
    in ModelStore/<slug>_gate_gamma.json, use it; otherwise fall back to PER_LEAGUE.
    """
    try:
        # Prefer tuned JSON saved by the report tuner
        if prefer_tuned and _Path is not None:
            model_dir = _os.getenv("MODEL_DIR", "ModelStore")
            path = _Path(model_dir) / f"{_slug(league_name)}_gate_gamma.json"
            if path.exists():
                try:
                    d = _json.loads(path.read_text(encoding="utf-8"))
                    if "gamma" in d: _os.environ["FTR_CONF_SHARPEN"] = str(d["gamma"])
                    if "gate"  in d: _os.environ["FTR_PICK_GATE"]    = str(d["gate"])
                except Exception:
                    pass
        # Apply per-league defaults without clobbering tuned values
        cfg = PER_LEAGUE.get(league_name, {})
        for k in ("FTR_CAL_MODE", "FTR_CONF_SHARPEN", "FTR_PICK_GATE"):
            v = cfg.get(k)
            if v is not None:
                _os.environ.setdefault(k, str(v))
    except Exception:
        # never fail hard on config
        pass