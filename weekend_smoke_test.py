#!/usr/bin/env python3
"""
Weekend smoke test for accumulator markets.

Runs:
- prepare_league_for_inference()  -> rebuilds press baseline/intensity, loads alpha & best_k
- generates BTTS/Over2.5 probs, inverse markets (Under2.5, BTTS NO)
- optional volatility adjustment
- WTN & Clean Sheet selection (needs FTS columns)
- FTR accumulator (if FTR probs + odds present)
- Correct Score shortlist (if home/away lambda preds present)
- ROI + walk-forward P&L (flat vs 1/2-Kelly) for each market
- Exports per-league & cross-league CSVs (handled inside prediction_overlay)

Usage (example):
  python scripts/weekend_smoke_test.py \
    --league "England Premier League" \
    --matches-csv "Matches/England Premier League/england-premier-league-matches-2025-to-2025-stats.csv" \
    --btts-model "ModelStore/England_Premier_League_BTTS_model.joblib" \
    --over-model "ModelStore/England_Premier_League_Over25_model.joblib"
"""

import os, json, argparse, datetime, glob, re
import pandas as pd
import numpy as np
import re

# --- robust parser for free-form date_GMT strings (silences per-element fallback warning)
def _parse_match_date_series(series: pd.Series) -> pd.Series:
    """Try common formats used in our match files and return the parse with
    the most non-NaNs; fall back to pandas' generic parser."""
    if series is None:
        return pd.Series(pd.NaT, index=[])
    fmts = (
        "%b %d %Y - %I:%M%p",  # e.g., "Jun 27 2023 - 1:00pm"
        "%b %d %Y - %H:%M",    # e.g., "Jun 27 2023 - 13:00"
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    )
    best = None
    best_nonnull = -1
    for fmt in fmts:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        nn = int(parsed.notna().sum())
        if nn > best_nonnull:
            best_nonnull = nn
            best = parsed
    if best is not None and best_nonnull > 0:
        return best
    return pd.to_datetime(series, errors="coerce")
try:
    from prediction_overlay import overlay_config
except Exception:
    overlay_config = {"acca_sizes": [2, 3, 5, 10], "volatility_adjust": True, "allow_synth_odds": False}
    print("⚠️ overlay_config not exported by prediction_overlay; using local defaults (flags may not affect overlay).")
    # Prefer trained models when available
# ---- Status & window filtering -------------------------------------------------
# Common end-of-match markers observed across our datasets
COMPLETED_STATUS_PATTERNS = (
    "ft", "full time", "finished", "after extra time", "aet",
    "pens", "penalties", "final", "match finished"
)

def _detect_status_column(df: pd.DataFrame) -> str | None:
    """Return the first column that looks like a match status field (case-insensitive)."""
    for c in df.columns:
        if "status" in str(c).lower():
            return c
    return None

def _is_completed_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower()
    mask = pd.Series(False, index=s.index)
    for pat in COMPLETED_STATUS_PATTERNS:
        mask = mask | s.str.contains(pat, na=False)
    return mask

def _load_yaml_config_if_any() -> dict:
    """Best-effort loader for ./config.yaml or ./config.yml. Safe no-op on failure."""
    import os
    cfg_paths = ["config.yaml", "config.yml"]
    for p in cfg_paths:
        if os.path.exists(p):
            try:
                import yaml  # type: ignore
                with open(p, "r") as fh:
                    return yaml.safe_load(fh) or {}
            except Exception:
                # YAML may not be installed; fall back to empty
                return {}
    return {}

def _filter_for_prediction_window(df: pd.DataFrame,
                                  start: str | None = None,
                                  end: str | None = None,
                                  only_upcoming: bool = True) -> pd.DataFrame:
    """Filter dataframe to a date window [start, end] (inclusive) and optionally
    drop rows that appear to be completed fixtures by `status`.
    Expects `match_date` to be a YYYY-MM-DD string column (caller ensures creation)."""
    out = df.copy()
    # 1) Upcoming-only by status (if a status column exists)
    if only_upcoming:
        col = _detect_status_column(out)
        if col is not None:
            try:
                mask_comp = _is_completed_series(out[col])
                before = len(out)
                out = out.loc[~mask_comp].copy()
                if before != len(out):
                    print(f"🧹 Dropped completed fixtures by status: {before-len(out)}")
            except Exception:
                pass
    # 2) Window by match_date if provided
    if start or end:
        md = pd.to_datetime(out.get("match_date"), errors="coerce")
        if start:
            try:
                s = pd.to_datetime(str(start), errors="coerce")
                out = out.loc[md >= s].copy()
            except Exception:
                pass
        if end:
            try:
                e = pd.to_datetime(str(end), errors="coerce")
                out = out.loc[md <= e].copy()
            except Exception:
                pass
    return out

overlay_config["prefer_trained_models"] = True
# ------------------------------------------------------------------
# Robust season CSV picker (handles many naming schemes)
# ------------------------------------------------------------------
def _pick_season_csv(match_dir: str) -> str | None:
    """Return the best season CSV path under `match_dir`.
    Understands names like:
      - "England Premier League 2024-25.csv"
      - "spain-la-liga-matches-2024-to-2025-stats (2).csv"
      - single-year variants like "2025" (MLS style)
    Falls back to newest (mtime) CSV if no season parse succeeds.
    Skips fixtures/predictions/report/model outputs.
    """
    if not os.path.isdir(match_dir):
        return None

    def _skip(path: str) -> bool:
        name = os.path.basename(path).lower()
        if any(k in name for k in ("upcoming", "fixture", "fixtures", "prediction", "predictions", "report")):
            return True
        if any(k in path.lower() for k in ("predictions_output", "modelstore")):
            return True
        return False

    paths = [p for p in glob.glob(os.path.join(match_dir, "**", "*.csv"), recursive=True) if not _skip(p)]
    if not paths:
        return None

    # Parse season from filename
    season_candidates: list[tuple[str, int, int, str]] = []  # (path, y1, y2, base)
    for p in paths:
        base = os.path.basename(p)
        s = base.lower()
        # two-year forms: 2024-25, 2024/25, 2024_to_2025, 2024-2025
        m = re.search(r"((?:19|20)\d{2})\s*(?:[-/_–]|\s+to\s+)\s*((?:19|20)?\d{2})", s)
        if m:
            y1 = int(m.group(1))
            y2_raw = m.group(2)
            y2 = int(y2_raw) if len(y2_raw) == 4 else (y1 // 100) * 100 + int(y2_raw)
            season_candidates.append((p, y1, y2, base))
            continue
        # single-year form: use that year for both ends (MLS style files occasionally)
        m1 = re.findall(r"((?:19|20)\d{2})", s)
        if m1:
            y = int(m1[-1])
            season_candidates.append((p, y, y, base))

    if season_candidates:
        # choose by (y1,y2) descending
        season_candidates.sort(key=lambda t: (t[1], t[2], t[3]), reverse=True)
        return season_candidates[0][0]

    # Fallback to latest modified time
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths[0] if paths else None

# Back-compat: some helpers reference `config`; keep it as an alias.
config = overlay_config

from prediction_overlay import (
    prepare_league_for_inference,
    generate_btts_and_over_preds,
    adjust_with_volatility_modifiers,
    validate_market_coherence,
    generate_accumulator_recommendations,
    generate_correct_score_candidates,
    log_resolved_odds_columns,
    _force_drop_leaky_cols,
    _coerce_numeric_like,
    _normalise_prob_columns,
    infer_probs_from_odds,
    apply_safe_renames_and_whitelist,
    infer_ftr_from_odds,
    seed_goal_lambda_from_prematch,
    ensure_minimal_signals,
    attach_win_to_nil_proxy,
)

# Try to import overlay helpers for auto-trained models, fallback to None if not present
try:
    from prediction_overlay import score_trained_markets_if_available, apply_market_thresholds
except Exception:
    score_trained_markets_if_available = None
    apply_market_thresholds = None

try:
    from joblib import load as joblib_load
except Exception:
    joblib_load = None


def _maybe_load(path: str):
    if not path:
        return None
    if not os.path.exists(path):
        print(f"⚠️ model missing: {path}")
        return None
    if joblib_load is None:
        print("⚠️ joblib not available; cannot load models.")
        return None
    try:
        return joblib_load(path)
    except Exception as e:
        print(f"⚠️ failed to load {path}: {e}")
        return None


# -------------------- helpers for auto path resolution & batching --------------------
LEAGUE_DEFAULT_BATCH = [
    "England Premier League",
    "Spain La Liga",
    "Italy Serie A",
    "Germany Bundesliga",
    "Portugal Liga",
    "USA MLS",
]


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")

# --- Ensure exported CSVs have match_date by patching overlays on-disk ---
def _backfill_match_date_in_exports(league: str, src_df: pd.DataFrame, outdir: str | None) -> None:
    """Some overlay helpers write their own CSVs and may drop passthrough columns
    like `match_date`. After markets run, scan the league's output files and
    left-join `match_date` from the original dataframe on (home_team_name, away_team_name).
    Safe no-op if nothing to patch.
    """
    try:
        # Resolve destination dir used by this run (same logic as below)
        dstdir = outdir or os.path.join("predictions_output", datetime.datetime.utcnow().strftime("%Y-%m-%d"))
        if not os.path.isdir(dstdir):
            return
        prefix = league.replace(" ", "_") + "_"
        # Consider any per-market picks CSVs for this league
        pattern = os.path.join(dstdir, f"{prefix}*_picks.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            return
        # Build small lookup table from source df
        if not {"home_team_name", "away_team_name", "match_date"}.issubset(src_df.columns):
            return
        lut = (
            src_df[["home_team_name", "away_team_name", "match_date"]]
                .dropna(subset=["home_team_name", "away_team_name"]).drop_duplicates()
        )
        # Normalise match_date to string (no NaT)
        lut["match_date"] = (
            pd.to_datetime(lut["match_date"], errors="coerce")
              .dt.strftime("%Y-%m-%d").fillna("")
        )
        for fp in files:
            try:
                out = pd.read_csv(fp)
            except Exception:
                continue
            # Only patch if missing or blank
            needs_col = "match_date" not in out.columns
            needs_fill = False
            if not needs_col:
                try:
                    needs_fill = out["match_date"].astype(str).str.strip().eq("").any()
                except Exception:
                    needs_fill = False
            if not (needs_col or needs_fill):
                continue
            if not {"home_team_name", "away_team_name"}.issubset(out.columns):
                continue
            merged = out.merge(lut, on=["home_team_name", "away_team_name"], how="left", suffixes=("", "_src"))
            # If column existed and has blanks, backfill from _src
            if "match_date" in out.columns:
                m = merged["match_date"].astype(str).str.strip()
                src = merged["match_date_src"].astype(str).str.strip() if "match_date_src" in merged.columns else ""
                merged.loc[m.eq("") & src.ne(""), "match_date"] = merged.loc[m.eq("") & src.ne(""), "match_date_src"]
            else:
                # Column missing: just rename source in
                if "match_date_src" in merged.columns:
                    merged.rename(columns={"match_date_src": "match_date"}, inplace=True)
            if "match_date_src" in merged.columns:
                merged.drop(columns=["match_date_src"], inplace=True)
            try:
                merged.to_csv(fp, index=False)
                print(f"🩹 Patched match_date → {os.path.basename(fp)}")
            except Exception:
                pass
    except Exception:
        # Never fail the run for patching
        return

def find_latest_matches_csv(league: str) -> str | None:
    """Pick the latest matches CSV for a league under Matches/<League>.
    We now consider any .csv in the directory (some leagues don't include the
    word 'matches' in filenames). We sort by season-like year tokens when present,
    otherwise by modification time, then lexicographic name.
    """
    mdir = os.path.join("Matches", league)
    if not os.path.isdir(mdir):
        print(f"⚠️ Matches dir not found for {league}: {mdir}")
        return None

    # consider ALL CSVs in the league's Matches dir
    cands = [
        os.path.join(mdir, f)
        for f in os.listdir(mdir)
        if f.lower().endswith(".csv")
    ]
    if not cands:
        print(f"⚠️ No CSV files under {mdir}")
        return None

    def _season_key(p: str):
        fn = os.path.basename(p)
        # extract 4-digit years; take the last two as (start, end)
        nums = re.findall(r"(20\d{2}|19\d{2})", fn)
        if len(nums) >= 2:
            return (int(nums[-2]), int(nums[-1]), 1)
        elif len(nums) == 1:
            return (int(nums[0]), int(nums[0]), 1)
        else:
            # fall back to file modification time (older first), mark as 0 to sort after year-tagged
            try:
                mtime = os.path.getmtime(p)
            except Exception:
                mtime = 0
            return (0, 0, mtime)

    # sort by season key then filename for stability
    cands.sort(key=lambda p: (_season_key(p), os.path.basename(p)))
    chosen = cands[-1]
    print(f"🗂️  Auto-selected matches CSV for {league}: {os.path.basename(chosen)}")
    return chosen

def default_model_paths(league: str) -> tuple[str | None, str | None]:
    """Return (btts_model_path, over_model_path) if they exist under ModelStore."""
    slug = _slug(league)
    ms = "ModelStore"
    btts_guess = os.path.join(ms, f"{slug}_BTTS_model.joblib")
    over_guess = os.path.join(ms, f"{slug}_Over25_model.joblib")
    btts = btts_guess if os.path.exists(btts_guess) else None
    over = over_guess if os.path.exists(over_guess) else None
    # fallback via glob if exact name missing
    if btts is None:
        g = glob.glob(os.path.join(ms, f"*{slug}*BTTS*joblib"))
        btts = g[0] if g else None
    if over is None:
        g = glob.glob(os.path.join(ms, f"*{slug}*Over25*joblib"))
        over = g[0] if g else None
    return btts, over


def run_for_league(league: str, matches_csv: str, btts_model_path: str | None, over_model_path: str | None,
                   outdir: str | None, markets: list[str], top_n: int, show_odds_map: bool,
                   no_volatility_adjust: bool, infer_from_odds: bool,
                   thr_btts: float | None = None, thr_over25: float | None = None,
                   thr_under25: float | None = None, thr_btts_no: float | None = None,
                   window_start: str | None = None, window_end: str | None = None, only_upcoming: bool = True):
    """Single-league execution (extracted from main). Returns summary dict."""
    # Work on a local copy of the markets list to avoid mutating the caller's list across leagues.
    markets = list(markets or [])
    # 1) Rebuild press baseline/intensity and load α & best-k
    alpha, best_k = prepare_league_for_inference(league, overwrite_baseline=False, overwrite_intensity=False)

    # 2) Load data
    df = pd.read_csv(matches_csv)
    # hygiene: drop leak, coerce numeric, normalise percents
    df = _force_drop_leaky_cols(df)
    df = _coerce_numeric_like(df)
    df = _normalise_prob_columns(df)

    # Ensure match_date exists and is populated (YYYY-MM-DD) for downstream exports
    # Prefer date_GMT (free-form), else epoch `timestamp`. Only fill where empty/NaN.
    def _is_empty_col(s):
        try:
            return s.isna().all() or s.astype(str).str.strip().eq("").all()
        except Exception:
            return False

    if "match_date" not in df.columns:
        # Create from date_GMT or timestamp
        if "date_GMT" in df.columns:
            _dt = _parse_match_date_series(df["date_GMT"])  # robust parse; avoids per-element warning
            df["match_date"] = _dt.dt.strftime("%Y-%m-%d").fillna("")
        elif "timestamp" in df.columns:
            _ts = pd.to_numeric(df["timestamp"], errors="coerce")
            _dt = pd.to_datetime(_ts, unit="s", errors="coerce")
            df["match_date"] = _dt.dt.strftime("%Y-%m-%d").fillna("")
        else:
            df["match_date"] = ""
    else:
        # Fill only missing/blank entries
        _need_fill = df["match_date"].isna() | df["match_date"].astype(str).str.strip().eq("")
        if _need_fill.any():
            _filled = pd.Series([None] * len(df), index=df.index, dtype="object")
            if "date_GMT" in df.columns:
                _dt = _parse_match_date_series(df["date_GMT"])  # robust parse; avoids per-element warning
                _filled = _dt.dt.strftime("%Y-%m-%d")
            elif "timestamp" in df.columns:
                _ts = pd.to_numeric(df["timestamp"], errors="coerce")
                _dt = pd.to_datetime(_ts, unit="s", errors="coerce")
                _filled = _dt.dt.strftime("%Y-%m-%d")
            df.loc[_need_fill, "match_date"] = _filled[_need_fill].fillna("")
            # Normalize any 'NaT' string remnants and ensure no NaN/NaT
            df["match_date"] = df["match_date"].astype(str).replace({"NaT": ""}).fillna("")

    # Apply prediction window & upcoming filter
    df_before = len(df)
    df = _filter_for_prediction_window(df,
                                       start=window_start,
                                       end=window_end,
                                       only_upcoming=only_upcoming)
    if (window_start or window_end) or only_upcoming:
        kept = len(df)
        wtxt = f"[{window_start or '-'} → {window_end or '-'}]" if (window_start or window_end) else "[all dates]"
        print(f"📅 Window {wtxt} | upcoming_only={'yes' if only_upcoming else 'no'} | kept {kept}/{df_before}")

    # Optional threshold overrides for overlay helpers (picked up via df.attrs)
    if thr_btts is not None:
        df.attrs["thr_btts"] = float(thr_btts)
    if thr_over25 is not None:
        df.attrs["thr_over25"] = float(thr_over25)
    if thr_under25 is not None:
        df.attrs["thr_under25"] = float(thr_under25)
    if thr_btts_no is not None:
        df.attrs["thr_btts_no"] = float(thr_btts_no)

    if show_odds_map:
        try:
            log_resolved_odds_columns(df)
        except Exception:
            pass

    # 3) Prefer trained models via overlay helper (if available)
    used_overlay_scoring = False
    if score_trained_markets_if_available is not None:
        try:
            df2 = score_trained_markets_if_available(df.copy(), league)
            if isinstance(df2, pd.DataFrame):
                df = df2
                if bool(df.attrs.get("used_trained_models")):
                    used_overlay_scoring = True
        except Exception:
            pass

    # 3.1) Load side models explicitly (optional CLI paths still supported)
    btts_model = _maybe_load(btts_model_path)
    over_model = _maybe_load(over_model_path)

    # 4) Generate BTTS/Over (or fallback to odds), with optional volatility modifiers
    # Detect whether BTTS/Over confidences already exist from overlay scoring
    has_btts_over_cols = any(c in df.columns for c in (
        'btts_confidence', 'over25_confidence', 'adjusted_btts_confidence', 'adjusted_over25_confidence'
    ))

    has_models = (btts_model is not None and over_model is not None)
    if not has_btts_over_cols and has_models:
        # We don't have BTTS/Over yet — use the explicit models passed in
        df = generate_btts_and_over_preds(df, btts_model, over_model)
        if not no_volatility_adjust:
            try:
                df = adjust_with_volatility_modifiers(df)
            except Exception:
                pass
    elif has_btts_over_cols:
        # Already have model confidences (likely from overlay auto-scoring). Optionally apply volatility adjustment.
        if not no_volatility_adjust and not all(c in df.columns for c in ('adjusted_btts_confidence','adjusted_over25_confidence')):
            try:
                df = adjust_with_volatility_modifiers(df)
            except Exception:
                pass
    else:
        # No models and no overlay scoring → fallback path
        if infer_from_odds:
            print("ℹ️ No BTTS/Over models → inferring probabilities from bookmaker odds.")
            df = infer_probs_from_odds(df)
            # Ensure we also have FTR probabilities + goal lambdas for downstream markets
            df = ensure_minimal_signals(df)
            # Softer selection thresholds when using odds‑derived probabilities (unless CLI overrode earlier)
            if 'thr_btts' not in df.attrs:
                df.attrs['thr_btts'] = 0.45
            if 'thr_over25' not in df.attrs:
                df.attrs['thr_over25'] = 0.40
            if 'thr_under25' not in df.attrs:
                df.attrs['thr_under25'] = 0.55
            if 'thr_btts_no' not in df.attrs:
                df.attrs['thr_btts_no'] = 0.55

            # Backfill inverse confidence columns when only one side exists
            if 'under25_confidence' not in df.columns and 'over25_confidence' in df.columns:
                df['under25_confidence'] = 1 - pd.to_numeric(df['over25_confidence'], errors='coerce')
            if 'btts_no_confidence' not in df.columns and 'btts_confidence' in df.columns:
                df['btts_no_confidence'] = 1 - pd.to_numeric(df['btts_confidence'], errors='coerce')

            # If we have lambdas but no explicit FTS, derive FTS via Poisson no-goal probability
            if 'home_goals_pred' in df.columns and 'p_home_fts' not in df.columns:
                df['p_home_fts'] = np.exp(-pd.to_numeric(df['home_goals_pred'], errors='coerce').fillna(1.0))
            if 'away_goals_pred' in df.columns and 'p_away_fts' not in df.columns:
                df['p_away_fts'] = np.exp(-pd.to_numeric(df['away_goals_pred'], errors='coerce').fillna(1.0))

            # Compose Win-To-Nil proxies now that inputs exist
            df = attach_win_to_nil_proxy(df)
            # create simple binary preds for these markets if missing
            thr_map = {
                "btts_confidence": float(df.attrs.get("thr_btts", 0.55)),
                "over25_confidence": float(df.attrs.get("thr_over25", 0.32)),
                "under25_confidence": float(df.attrs.get("thr_under25", 0.60)),
                "btts_no_confidence": float(df.attrs.get("thr_btts_no", 0.60)),
            }
            for col, thr in thr_map.items():
                if col in df.columns:
                    pred_col = col.replace("_confidence", "_pred")
                    if pred_col not in df.columns:
                        df[pred_col] = (pd.to_numeric(df[col], errors="coerce").fillna(0.0) >= thr).astype(int)

            # If probabilities came from odds and we don't allow synthetic odds,
            # prune markets that would require fabricating odds. This avoids
            # noisy "missing odds" warnings and pointless synthesis attempts.
            if bool(df.attrs.get("probs_from_odds", False)) and not bool(overlay_config.get("allow_synth_odds", False)):
                preferred = ("btts", "over25", "btts_no", "ftr", "cs")
                pruned = [m for m in markets if m not in preferred]
                if pruned:
                    print(f"ℹ️ Synth odds disabled & probs-from-odds → pruning markets: {', '.join(pruned)}.")
                    markets[:] = [m for m in markets if m in preferred]
        else:
            print("ℹ️ BTTS/Over models not both available and --infer-from-odds is off; pruning markets: btts, over25, under25, btts_no.")
            markets[:] = [m for m in markets if m not in ("btts","over25","under25","btts_no")]

    # 4.1) Apply learned per‑market thresholds if overlay exposes them
    if apply_market_thresholds is not None:
        try:
            df = apply_market_thresholds(df, league)
        except Exception:
            pass

    # Integrate supervised draw model into FTR mix (keeps the main pipeline logic)
    from prediction_overlay import (
    attach_ftr_with_draw_mix,    # backward-compat
    attach_p_draw_and_mix_ftr,  # new bridge
    apply_market_thresholds_to_attrs,
    _ensure_press_for_inference,
    )
    import os
    try:
        _league_for_mix = league_name
    except NameError:
        try:
            _league_for_mix = league
        except NameError:
            _league_for_mix = config.get("league") if "config" in globals() else None
    if _league_for_mix:
            # Load per-market thresholds into df.attrs (BTTS/Over/FTS/GE2, etc.)
    df = apply_market_thresholds_to_attrs(df, _league_for_mix)

    # Ensure press-intensity ETL uses the same alpha chosen at draw training
    try:
        _matches_dir = config.get("matches_dir", os.path.join("Matches", _league_for_mix.replace(" ", "_")))
    except Exception:
        _matches_dir = os.path.join("Matches", _league_for_mix.replace(" ", "_"))
    _ensure_press_for_inference(_league_for_mix, _matches_dir)
    df = attach_ftr_with_draw_mix(df, _league_for_mix)

    # 5) Coherence checks
    df = validate_market_coherence(df, verbose=True)

    # 5.1) Surface which trained models (if any) were used by the overlay
    _used = df.attrs.get("used_trained_models")
    if _used:
        if isinstance(_used, (set, list, tuple)):
            _used_list = sorted(list(_used))
        else:
            _used_list = [str(_used)]
        print(f"🧠 Trained models in use: {', '.join(_used_list)}")

    # 6) Markets → generate + export (handled inside overlay helpers)
    summary = {"league": league, "alpha_used": alpha, "best_k_pct": best_k, "markets": {}}
    for market in markets:
        try:
            if market == "cs":
                cs = generate_correct_score_candidates(df, top_n=top_n)
                summary["markets"]["correct_score"] = {
                    "n": int(len(cs)),
                    "preview": cs.head(min(5, len(cs))).to_dict(orient="records"),
                }
                if len(cs):
                    dstdir = outdir or os.path.join("predictions_output", datetime.datetime.utcnow().strftime("%Y-%m-%d"))
                    os.makedirs(dstdir, exist_ok=True)
                    cs_out = os.path.join(dstdir, f"{league.replace(' ','_')}_correct_score_shortlist.csv")
                    cs.to_csv(cs_out, index=False)
                    print(f"📤 Saved CS shortlist → {cs_out}")
                continue
            out = generate_accumulator_recommendations(df, target_type=market, top_n=top_n, league_name=league)
            # Some overlay helpers may return None to signal "no candidates"; normalise to empty DataFrame
            if out is None:
                summary["markets"][market] = {"n": 0, "preview": []}
                continue
            # Ensure match_date survives overlay exports (some helpers drop passthrough cols)
            if "match_date" not in out.columns or out["match_date"].isna().any() or out["match_date"].astype(str).str.strip().eq("").any():
                base_cols = [c for c in ["home_team_name", "away_team_name", "match_date"] if c in df.columns]
                if set(["home_team_name", "away_team_name"]).issubset(base_cols) and "match_date" in base_cols:
                    out = out.merge(
                        df[base_cols].drop_duplicates(),
                        on=["home_team_name", "away_team_name"],
                        how="left",
                        suffixes=("", "_src")
                    )
                    # prefer any non-empty match_date from source
                    if "match_date_src" in out.columns:
                        needs = out["match_date"].isna() | out["match_date"].astype(str).str.strip().eq("")
                        out.loc[needs, "match_date"] = out.loc[needs, "match_date_src"]
                        out.drop(columns=["match_date_src"], inplace=True)
            summary["markets"][market] = {"n": int(len(out)), "preview": out.head(min(5, len(out))).to_dict(orient="records")}
        except Exception as e:
            print(f"⚠️ {market} failed: {e}")
            summary["markets"][market] = {"error": str(e)}

    # Post-process overlay exports on disk to ensure match_date is present
    _backfill_match_date_in_exports(league, df, outdir)

    # 7) Save per-league summary JSON
    dstdir = outdir or os.path.join("predictions_output", datetime.datetime.utcnow().strftime("%Y-%m-%d"))
    os.makedirs(dstdir, exist_ok=True)
    jpath = os.path.join(dstdir, f"{league.replace(' ', '_')}_weekend_smoke_summary.json")
    with open(jpath, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"✅ Summary JSON → {jpath}")

    # Compact per-league banner
    try:
        active_markets = [m for m, info in summary.get("markets", {}).items() if isinstance(info, dict) and int(info.get("n", 0)) > 0]
        if used_overlay_scoring or (btts_model_path and over_model_path):
            models_mode = "trained"
        elif infer_from_odds:
            models_mode = "odds"
        else:
            models_mode = "mixed"
        sims_on = bool(overlay_config.get("acca_sizes"))
        # Format best_k as a percentage if numeric (e.g., 0.10 -> "10%"), else show N/A
        if isinstance(best_k, (int, float)):
            best_k_txt = f"{best_k:.0%}"
        else:
            best_k_txt = "N/A"
        print(
            f"🧾 Summary → {league}: active=[{', '.join(active_markets or ['none'])}] | "
            f"models={models_mode} | sims={'on' if sims_on else 'off'} | α={alpha:.2f} | best_k={best_k_txt}"
        )
    except Exception:
        pass

    return summary



def main():
    ap = argparse.ArgumentParser()
    # Single-league explicit arguments
    ap.add_argument("--league", help="League name (when running single league)")
    ap.add_argument("--matches-csv", help="Input matches CSV for the league")
    ap.add_argument("--btts-model", default=None, help="Path to BTTS model .joblib")
    ap.add_argument("--over-model", default=None, help="Path to Over 2.5 model .joblib")
    # Batch mode
    ap.add_argument("--batch-leagues", default=None,
                    help="Comma list of leagues to run in batch. If omitted, uses a sensible default set.")
    ap.add_argument("--auto-paths", action="store_true",
                    help="Auto-resolve matches CSV (latest) and model paths from folder naming.")

    ap.add_argument("--outdir", default=None, help="Override output dir (default: predictions_output/YYYY-MM-DD)")
    ap.add_argument("--no-volatility-adjust", action="store_true", help="Skip adjusted_* confidence step")
    ap.add_argument("--markets", default="btts,over25,under25,btts_no,ftr,wtn,clean_sheet,cs",
                    help="Comma list of markets to run. 'cs' means correct score shortlist.")
    ap.add_argument("--top-n", type=int, default=50, help="Top-N to print per market (exports still handled inside)")
    ap.add_argument("--show-odds-map", action="store_true", help="Log which odds columns were found/resolved.")
    ap.add_argument("--no-sims", action="store_true",
                    help="Disable ROI simulations (accumulator ROI + walk-forward) inside overlay helpers.")
    ap.add_argument("--max-acca-size", type=int, default=None,
                    help="Cap the largest simulated accumulator fold (e.g., 3 = up to trebles).")
    ap.add_argument("--infer-from-odds", action="store_true",
                    help="If BTTS/Over/FTR models are missing, infer probs from bookmaker odds.")
    ap.add_argument("--allow-synth-odds", action="store_true",
                    help="Permit synthesising odds from probabilities even when probabilities came from bookmaker odds (off by default).")
    ap.add_argument("--thr-btts", type=float, default=None,
                    help="Override BTTS YES selection threshold (default 0.55).")
    ap.add_argument("--thr-over25", type=float, default=None,
                    help="Override Over 2.5 selection threshold (default 0.32).")
    ap.add_argument("--thr-under25", type=float, default=None,
                    help="Override Under 2.5 selection threshold (default 0.60).")
    ap.add_argument("--thr-btts-no", type=float, default=None,
                    help="Override BTTS NO selection threshold (default 0.60).")
    ap.add_argument("--print-config", action="store_true", help="Print effective overlay_config at startup")
    ap.add_argument("--window-start", default=None, help="Filter predictions: inclusive start date YYYY-MM-DD")
    ap.add_argument("--window-end", default=None, help="Filter predictions: inclusive end date YYYY-MM-DD")
    ap.add_argument("--include-completed", action="store_true", help="Do not drop completed fixtures by status")
    args = ap.parse_args()

    # Optionally pull a prediction_window from config.yaml when CLI not supplied
    if not args.window_start or not args.window_end:
        _cfg = _load_yaml_config_if_any()
        try:
            _pw = (_cfg or {}).get("prediction_window") or {}
            args.window_start = args.window_start or _pw.get("start")
            args.window_end   = args.window_end   or _pw.get("end")
        except Exception:
            pass

    # Apply simulation controls to prediction_overlay's config
    try:
        # Start from whatever overlay_config currently has (fallback defaults if missing)
        default_sizes = overlay_config.get("acca_sizes", [2, 3, 5, 10])
        # Normalise & sort
        sizes = sorted({int(s) for s in (default_sizes or []) if isinstance(s, (int, float)) and int(s) > 0})
        if not sizes:
            sizes = [2, 3, 5, 10]

        if args.no_sims:
            overlay_config["acca_sizes"] = []  # disable ROI sims entirely
        else:
            if args.max_acca_size is not None:
                sizes = [s for s in sizes if s <= int(args.max_acca_size)]
            overlay_config["acca_sizes"] = sizes

        # Propagate no-volatility-adjust into overlay config as a soft flag (helpers may read it)
        if args.no_volatility_adjust:
            overlay_config["volatility_adjust"] = False
        # Propagate allow_synth_odds into overlay config
        overlay_config["allow_synth_odds"] = bool(getattr(args, "allow_synth_odds", False))
        overlay_config["prefer_trained_models"] = True
    except Exception:
        # Non-fatal; keep running with defaults
        pass

    if args.print_config:
        print(f"🔧 Effective overlay_config: {overlay_config}")
        overlay_config["prefer_trained_models"] = True

    markets = [m.strip() for m in (args.markets or "").split(",") if m.strip()]

    # Decide mode: batch vs single
    if args.batch_leagues:
        leagues = [s.strip() for s in args.batch_leagues.split(",") if s.strip()]
    elif not args.league:
        leagues = LEAGUE_DEFAULT_BATCH[:]  # default batch
    else:
        leagues = []

    summaries = []

    if leagues:
        print(f"🔁 Batch run for {len(leagues)} leagues")
        for lg in leagues:
            # Resolve paths (auto if --auto-paths OR no explicit --matches-csv)
            use_auto = args.auto_paths or not args.matches_csv
            if use_auto:
                matches_csv = _pick_season_csv(os.path.join("Matches", lg))
                bm, om = default_model_paths(lg)
            else:
                matches_csv = args.matches_csv
                bm, om = args.btts_model, args.over_model
            if not matches_csv:
                mdir = os.path.join("Matches", lg)
                try:
                    available = [
                        os.path.relpath(p, mdir)
                        for p in glob.glob(os.path.join(mdir, "**", "*.csv"), recursive=True)
                    ]
                except Exception:
                    available = []
                print(f"⚠️ No matches CSV auto-found for {lg} under {mdir}; skipping.")
                if available:
                    print("   CSVs I can see:")
                    for f in sorted(available):
                        print(f"   - {f}")
                continue
            print(f"➡️  {lg}: matches={os.path.basename(matches_csv)}")
            
            # Determine per-league infer_from_odds: force True if both models are missing
            league_infer_from_odds = args.infer_from_odds
            if bm is None and om is None:
                league_infer_from_odds = True
            summaries.append(
                run_for_league(
                    league=lg,
                    matches_csv=matches_csv,
                    btts_model_path=bm,
                    over_model_path=om,
                    outdir=args.outdir,
                    markets=markets.copy(),
                    top_n=args.top_n,
                    show_odds_map=args.show_odds_map,
                    no_volatility_adjust=args.no_volatility_adjust,
                    infer_from_odds=league_infer_from_odds,
                    thr_btts=args.thr_btts,
                    thr_over25=args.thr_over25,
                    thr_under25=args.thr_under25,
                    thr_btts_no=args.thr_btts_no,
                    window_start=args.window_start,
                    window_end=args.window_end,
                    only_upcoming=not bool(args.include_completed),
                )
            )
        # Write a combined summary file
        dstdir = args.outdir or os.path.join("predictions_output", datetime.datetime.utcnow().strftime("%Y-%m-%d"))
        os.makedirs(dstdir, exist_ok=True)
        jpath = os.path.join(dstdir, "BATCH_weekend_smoke_summary.json")
        with open(jpath, "w") as fh:
            json.dump({"runs": summaries}, fh, indent=2)
        _winfo = None
        if args.window_start or args.window_end:
            _winfo = f"{args.window_start or '-'}→{args.window_end or '-'}"
        if _winfo:
            print(f"📅 Active prediction window: {_winfo} | upcoming_only={'yes' if (not args.include_completed) else 'no'}")
        print(f"✅ Batch summary JSON → {jpath}")
        return

    # Single-league mode
    if not args.league:
        ap.error("--league is required when not running --batch-leagues")
    league = args.league
    matches_csv = args.matches_csv or (find_latest_matches_csv(league) if args.auto_paths else None)
    if not matches_csv:
        mdir = os.path.join("Matches", league)
        try:
            available = [f for f in os.listdir(mdir) if f.lower().endswith('.csv')]
        except Exception:
            available = []
        if args.auto_paths:
            print(f"⚠️ Could not auto-resolve a matches CSV under {mdir}.")
            if available:
                print("   CSVs I can see:")
                for f in sorted(available):
                    print(f"   - {f}")
        ap.error("--matches-csv is required unless --auto-paths can find a file in Matches/<League>.")

    btts_model = args.btts_model
    over_model = args.over_model
    if args.auto_paths:
        bm, om = default_model_paths(league)
        btts_model = btts_model or bm
        over_model = over_model or om

    run_for_league(
        league=league,
        matches_csv=matches_csv,
        btts_model_path=btts_model,
        over_model_path=over_model,
        outdir=args.outdir,
        markets=markets.copy(),
        top_n=args.top_n,
        show_odds_map=args.show_odds_map,
        no_volatility_adjust=args.no_volatility_adjust,
        infer_from_odds=args.infer_from_odds,
        thr_btts=args.thr_btts,
        thr_over25=args.thr_over25,
        thr_under25=args.thr_under25,
        thr_btts_no=args.thr_btts_no,
        window_start=args.window_start,
        window_end=args.window_end,
        only_upcoming=not bool(args.include_completed),
    )


if __name__ == "__main__":
    main()