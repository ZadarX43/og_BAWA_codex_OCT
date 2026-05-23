from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import re

import numpy as np
import pandas as pd

from prediction_overlay import _match_key, _coalesce_match_date_series


# -----------------------------
# Realised guards (same logic you standardised elsewhere)
# -----------------------------
_COMPLETED_RE = r"\bcomplete(?:d)?\b|\bft\b|full\s*time|finished|final|match\s*finished|aet|after\s*extra\s*time|pens?|penalt(?:y|ies)|awarded|ended"
_INCOMPLETE_RE = r"\bincomplete\b|postp|postponed|abandon|suspend|void|cancel|walkover|\bwo\b|\bns\b|not\s*started|live|in\s*play"

def _status_is_complete(s: pd.Series) -> pd.Series:
    x = s.astype("string").fillna("").str.strip().str.lower()
    return x.str.contains(_COMPLETED_RE, regex=True)

def _status_is_incomplete(s: pd.Series) -> pd.Series:
    x = s.astype("string").fillna("").str.strip().str.lower()
    return x.str.contains(_INCOMPLETE_RE, regex=True)

def _future_fixture_mask(df: pd.DataFrame, *, grace_hours: float = 0.0) -> pd.Series:
    idx = df.index
    dt = pd.Series(pd.NaT, index=idx, dtype="datetime64[ns, UTC]")
    for col in ("match_date", "date_GMT", "date", "timestamp"):
        if col not in df.columns:
            continue
        try:
            cand = pd.to_datetime(df[col], errors="coerce", utc=True, format="mixed")
        except TypeError:
            cand = pd.to_datetime(df[col], errors="coerce", utc=True)
        except Exception:
            try:
                cand = pd.to_datetime(df[col].astype(str), errors="coerce", utc=True, format="mixed")
            except TypeError:
                cand = pd.to_datetime(df[col].astype(str), errors="coerce", utc=True)
        dt = dt.fillna(cand)

    if dt.isna().all():
        return pd.Series(False, index=idx)

    now_utc = pd.Timestamp.now(tz="UTC")
    if grace_hours and grace_hours > 0:
        now_utc = now_utc + pd.Timedelta(hours=float(grace_hours))
    return dt.notna() & (dt > now_utc)

def _realised_mask(df: pd.DataFrame) -> pd.Series:
    idx = df.index
    hg = pd.to_numeric(df.get("home_team_goal_count", pd.Series(np.nan, index=idx)), errors="coerce")
    ag = pd.to_numeric(df.get("away_team_goal_count", pd.Series(np.nan, index=idx)), errors="coerce")
    goals_present = ~(hg.isna() | ag.isna())

    status_txt = (
        df["status"].astype("string").fillna("").str.strip().str.lower()
        if "status" in df.columns else pd.Series("", index=idx, dtype="string")
    )
    st_complete = _status_is_complete(status_txt) if "status" in df.columns else pd.Series(False, index=idx)
    st_incomp   = _status_is_incomplete(status_txt) if "status" in df.columns else pd.Series(False, index=idx)

    st_known = status_txt.ne("") & (st_complete | st_incomp)

    # Require date_known for fallback path
    dt = pd.Series(pd.NaT, index=idx, dtype="datetime64[ns, UTC]")
    for col in ("match_date", "date_GMT", "date", "timestamp"):
        if col not in df.columns:
            continue
        try:
            cand = pd.to_datetime(df[col], errors="coerce", utc=True, format="mixed")
        except TypeError:
            cand = pd.to_datetime(df[col], errors="coerce", utc=True)
        except Exception:
            try:
                cand = pd.to_datetime(df[col].astype(str), errors="coerce", utc=True, format="mixed")
            except TypeError:
                cand = pd.to_datetime(df[col].astype(str), errors="coerce", utc=True)
        dt = dt.fillna(cand)
    date_known = dt.notna()

    base = (st_complete | ((~st_known) & goals_present & date_known)) & goals_present
    future = _future_fixture_mask(df)

    # Some feeds incorrectly label finished matches as 'incomplete'.
    # Allow those only when we have strong evidence of a realised match:
    # - goals present AND date known AND not future
    # - AND (non-zero goals) OR (any other match stats present beyond goals)
    stats_present = pd.Series(False, index=idx)
    for c in (
        "home_team_shots", "away_team_shots",
        "home_team_shots_on_target", "away_team_shots_on_target",
        "home_team_shots_off_target", "away_team_shots_off_target",
        "home_team_corner_count", "away_team_corner_count",
        "home_team_yellow_cards", "away_team_yellow_cards",
        "home_team_red_cards", "away_team_red_cards",
        "home_team_possession", "away_team_possession",
        "home_team_fouls", "away_team_fouls",
        "team_a_xg", "team_b_xg",
        "total_goal_count",
    ):
        if c in df.columns:
            stats_present |= df[c].notna()

    goals_sum = (hg.fillna(0) + ag.fillna(0))
    nonzero_goals = goals_sum > 0

    incomp_ok = st_incomp & goals_present & date_known & (~future) & (stats_present | nonzero_goals)

    return (base & (~st_incomp) & (~future)) | incomp_ok


# -----------------------------
# Multi-season matches loader (canonical fixture_key)
# -----------------------------
def _load_all_matches_csvs(matches_root: Path, league: str) -> pd.DataFrame:
    mdir = matches_root / league
    if not mdir.exists():
        return pd.DataFrame()

    files = sorted(mdir.glob("*.csv"))
    if not files:
        return pd.DataFrame()

    frames = []
    for p in files:
        try:
            df = pd.read_csv(p)
            df["__src_csv"] = p.name
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True, sort=False)

    # Canonical match_date: treat blank strings as missing so coalescer can fill from date_GMT/timestamp
    if "match_date" not in df.columns:
        df["match_date"] = pd.NA
    else:
        try:
            md0 = df["match_date"].astype("string").str.strip()
            df["match_date"] = md0.mask(md0.eq(""), pd.NA)
        except Exception:
            pass

    df["match_date"] = _coalesce_match_date_series(df)

    # Standardize to string for stable fixture_keying
    try:
        md1 = df["match_date"].astype("string").str.strip()
        df["match_date"] = md1.mask(md1.eq(""), pd.NA)
    except Exception:
        pass

    # Fallback fill: if coalescer left NAs, try common date columns directly
    try:
        if df["match_date"].isna().any():
            for col in ("date_GMT", "date", "Date", "timestamp"):
                if col not in df.columns:
                    continue
                s = df[col]
                try:
                    cand = pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
                except TypeError:
                    cand = pd.to_datetime(s, errors="coerce", utc=True)
                # keep only where we can parse
                df["match_date"] = df["match_date"].fillna(cand)

            # Re-standardize to string after filling
            md2 = df["match_date"].astype("string").str.strip()
            df["match_date"] = md2.mask(md2.eq(""), pd.NA)
    except Exception:
        pass

    if "home_team_name" not in df.columns and "Home" in df.columns:
        df["home_team_name"] = df["Home"]
    if "away_team_name" not in df.columns and "Away" in df.columns:
        df["away_team_name"] = df["Away"]

    # canonical fixture_key
    df["fixture_key"] = df.apply(_match_key, axis=1)

    # Drop rows without a usable fixture_key (prevents silent join drift)
    try:
        fk = df["fixture_key"].astype("string").fillna("").str.strip()
        bad = fk.eq("")
        if bool(bad.any()):
            df = df.loc[~bad].copy()
    except Exception:
        pass

    # If match_date is missing, fixture_keying is unreliable; drop those rows.
    if "match_date" in df.columns:
        try:
            md = df["match_date"].astype("string").fillna("").str.strip()
            bad_md = md.eq("")
            if bool(bad_md.any()):
                df = df.loc[~bad_md].copy()
        except Exception:
            pass

    # ensure numeric goals exist
    for c in ("home_team_goal_count", "away_team_goal_count"):
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Dedup scoring: completed wins, then goals_present, then non-null, future loses hard
    nn = df.notna().sum(axis=1)
    status_txt = df["status"].astype("string").fillna("").str.strip().str.lower() if "status" in df.columns else pd.Series("", index=df.index, dtype="string")
    status_complete = _status_is_complete(status_txt) if "status" in df.columns else pd.Series(False, index=df.index)
    goals_present = df[["home_team_goal_count", "away_team_goal_count"]].notna().all(axis=1)
    future = _future_fixture_mask(df)
    score = (status_complete.astype(int) * 1000) + (goals_present.astype(int) * 100) + nn - (future.astype(int) * 5000)

    df = df.assign(__score=score).sort_values(["fixture_key", "__score"], ascending=[True, False])
    df = df.drop_duplicates(subset=["fixture_key"], keep="first").drop(columns=["__score"], errors="ignore")

    # re-assert canonical key (belt + braces)
    df["fixture_key"] = df.apply(_match_key, axis=1)

    return df


# -----------------------------
# FTR band parsing
# -----------------------------
def _band_lower(s: str) -> float:
    if s is None:
        return np.nan
    s = str(s).strip()
    if s.upper() in ("ALL", "NAN", ""):
        return np.nan
    # handles "<0.45" and "≥0.65" and "0.55–0.60" (en dash)
    s = s.replace("—", "–").replace("-", "–")
    if s.startswith("<"):
        try:
            return 0.0
        except Exception:
            return 0.0
    if s.startswith("≥"):
        try:
            return float(s.replace("≥", "").strip())
        except Exception:
            return np.nan
    if "–" in s:
        try:
            lo = s.split("–")[0].strip()
            return float(lo)
        except Exception:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


# -----------------------------
# Side correctness from goals
# -----------------------------
def _correct_from_goals(mkt: str, hg: pd.Series, ag: pd.Series) -> Optional[pd.Series]:
    g = hg + ag
    if mkt == "over25":
        return (g >= 3)
    if mkt == "under25":
        return (g <= 2)
    if mkt == "btts":
        return ((hg >= 1) & (ag >= 1))
    if mkt == "btts_no":
        return ~((hg >= 1) & (ag >= 1))
    return None


def _side_gate_candidates(mkt: str) -> List[Tuple[str, ...]]:
    if mkt == "over25":
        return [("VERY_STRONG_OVER",), ("VERY_STRONG_OVER","STRONG_OVER")]
    if mkt == "under25":
        return [("VERY_STRONG_UNDER",), ("VERY_STRONG_UNDER","STRONG_UNDER")]
    if mkt == "btts":
        # keep both unless VS is clearly worse; we’ll evaluate strict then expanded
        return [("VERY_STRONG_YES",), ("VERY_STRONG_YES","STRONG_YES")]
    if mkt == "btts_no":
        # WEAK_NO vs STRONG_NO is league-sensitive; evaluate both + union
        return [("STRONG_NO",), ("WEAK_NO",), ("STRONG_NO","WEAK_NO")]
    return []



# -----------------------------
# Sanity gates for SIDE markets
# (pre-deploy vetoes to prevent known shape-mismatch traps)
# -----------------------------
# BTTS YES sanity (market == "btts")
BTTS_YES_FTS_MAX = 0.55
BTTS_YES_MIN_BTTS_FROM_FTS = 0.20
BTTS_YES_MAX_ABS_POWER_DIFF = 22.0

# OVER 2.5 sanity (market == "over25")
OVER25_MAX_ABS_POWER_DIFF = 22.0
OVER25_MAX_ABS_PPG_DIFF = 0.70
OVER25_MIN_POIS_P_OVER = 0.58  # P(total goals >= 3) from Poisson(lam_total)

# OU25 learned-gate policy interaction
# Learned gates may still analyse/report OU25 label performance, but they must
# not produce deployable learned gates from rows the runtime policy would never
# allow live.
OU25_POLICY_BLOCK_REVIEW = True
OU25_POLICY_BLOCK_SHADOW = True
OU25_POLICY_ALLOW_UNDER_LEARNED_GATES = False


def _series_num(df: pd.DataFrame, col: str) -> pd.Series:
    """Numeric series helper; returns NaN series if col missing."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _poisson_p_total_ge3(lam_total: pd.Series) -> pd.Series:
    """P(TotalGoals >= 3) under Poisson(lam_total)."""
    lam = pd.to_numeric(lam_total, errors="coerce")
    lam = lam.clip(lower=0)
    # 1 - P(0) - P(1) - P(2)
    return 1.0 - (np.exp(-lam) * (1.0 + lam + 0.5 * lam * lam))


def _btts_yes_sanity_mask(df: pd.DataFrame) -> pd.Series:
    """Return True for rows that pass BTTS YES sanity gates.

    Uses blank-risk proxies (p_home_fts/p_away_fts) + optional power_diff.
    Missing columns do NOT veto.
    """
    idx = df.index
    ph = _series_num(df, "p_home_fts")
    pa = _series_num(df, "p_away_fts")

    # fts_max = max(p_home_fts, p_away_fts)
    fts_max = pd.concat([ph, pa], axis=1).max(axis=1)

    # btts_from_fts = P(home scores >=1) * P(away scores >=1)
    btts_from_fts = (1.0 - ph) * (1.0 - pa)

    # Optional power gate
    abs_power = _series_num(df, "power_diff").abs()

    c1 = btts_from_fts.isna() | (btts_from_fts >= float(BTTS_YES_MIN_BTTS_FROM_FTS))
    c2 = fts_max.isna() | (fts_max <= float(BTTS_YES_FTS_MAX))
    c3 = abs_power.isna() | (abs_power <= float(BTTS_YES_MAX_ABS_POWER_DIFF))

    out = (c1 & c2 & c3)
    return out.reindex(idx, fill_value=True)


def _over25_sanity_mask(df: pd.DataFrame) -> pd.Series:
    """Return True for rows that pass OVER 2.5 sanity gates.

    Uses power/PPG balance + Poisson total>=3 check.
    Missing columns do NOT veto.
    """
    idx = df.index

    # Power + PPG balance
    abs_power = _series_num(df, "power_diff").abs()
    abs_ppg = _series_num(df, "ppg_diff_pre").abs()

    c_power = abs_power.isna() | (abs_power <= float(OVER25_MAX_ABS_POWER_DIFF))
    c_ppg = abs_ppg.isna() | (abs_ppg <= float(OVER25_MAX_ABS_PPG_DIFF))

    # Poisson total goals probability (use a trusted lambda_total)
    # Priority:
    #   1) lambda_home + lambda_away (only when sane)
    #   2) bookie_lambda_total_fit (only when sane)
    #   3) exp_goals_sum (only when sane; last resort)
    lam_home = _series_num(df, "lambda_home")
    lam_away = _series_num(df, "lambda_away")

    lam_total = lam_home + lam_away
    lam_total = lam_total.where(lam_total.between(1.0, 6.0))

    if lam_total.isna().all():
        bl = _series_num(df, "bookie_lambda_total_fit")
        lam_total = bl.where(bl.between(1.0, 6.0))

    if lam_total.isna().all():
        eg = _series_num(df, "exp_goals_sum")
        lam_total = eg.where(eg.between(1.0, 6.0))

    p_over = _poisson_p_total_ge3(lam_total)
    c_pois = p_over.isna() | (p_over >= float(OVER25_MIN_POIS_P_OVER))

    out = (c_power & c_ppg & c_pois)
    return out.reindex(idx, fill_value=True)


def _ou25_policy_eligible_mask(df: pd.DataFrame, mkt: str) -> pd.Series:
    """Rows eligible for OU25 learned-gate selection under runtime policy.

    Important:
      - This helper is ONLY for learned-gate selection in `build_side_gates`.
      - It does NOT stop per-label reporting; reporting can still inspect the
        full realised OU25 sample.
      - Default posture:
          * over25 => block shadow rows + review-state rows
          * under25 => blocked entirely unless explicitly allowed
    """
    idx = df.index
    out = pd.Series(True, index=idx)

    if str(mkt).strip().lower() not in ("over25", "under25"):
        return out

    shadow = pd.Series(False, index=idx)
    if "ou25_is_shadow" in df.columns:
        shadow = pd.to_numeric(df["ou25_is_shadow"], errors="coerce").fillna(0).astype(int).ge(1)
    else:
        lane = df.get("model_lane", pd.Series("", index=idx)).astype("string").fillna("").str.lower().str.strip()
        prod = df.get("product", pd.Series("", index=idx)).astype("string").fillna("").str.upper().str.strip()
        shadow = lane.eq("dedicated_over25_model") | prod.isin(["OVER25_MODEL", "OU25_MODEL"])

    review = pd.Series(False, index=idx)
    if "ou25_policy_state" in df.columns:
        review = df["ou25_policy_state"].astype("string").fillna("").str.lower().str.strip().eq("review")

    if bool(OU25_POLICY_BLOCK_SHADOW):
        out &= ~shadow
    if bool(OU25_POLICY_BLOCK_REVIEW):
        out &= ~review

    if str(mkt).strip().lower() == "under25" and not bool(OU25_POLICY_ALLOW_UNDER_LEARNED_GATES):
        out &= False

    return out.reindex(idx, fill_value=False)


def _ou25_policy_note(df: pd.DataFrame, mkt: str) -> str:
    """Compact note describing OU25 learned-gate policy posture."""
    if str(mkt).strip().lower() == "over25":
        parts: List[str] = []
        if bool(OU25_POLICY_BLOCK_SHADOW):
            parts.append("block_shadow")
        if bool(OU25_POLICY_BLOCK_REVIEW):
            parts.append("block_review")
        return "OU25_OVER_POLICY[" + ",".join(parts) + "]" if parts else ""
    if str(mkt).strip().lower() == "under25":
        if not bool(OU25_POLICY_ALLOW_UNDER_LEARNED_GATES):
            return "OU25_UNDER_POLICY[blocked]"
        parts: List[str] = []
        if bool(OU25_POLICY_BLOCK_SHADOW):
            parts.append("block_shadow")
        if bool(OU25_POLICY_BLOCK_REVIEW):
            parts.append("block_review")
        return "OU25_UNDER_POLICY[" + ",".join(parts) + "]" if parts else ""
    return ""

def _center(s: str, width: int = 86) -> str:
    return s.center(width)

# -----------------------------
# Fallback join key helpers (for when fixture_key drifts)
# -----------------------------

def _norm_team_series(s: pd.Series) -> pd.Series:
    x = s.astype("string").fillna("").str.lower().str.strip()
    x = x.str.replace(r"\bfc\b", "", regex=True)
    x = x.str.replace(r"[^a-z0-9]+", " ", regex=True)
    x = x.str.replace(r"\s+", " ", regex=True).str.strip()
    return x


def _date_only_series(s: pd.Series) -> pd.Series:
    try:
        dt = pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
    except TypeError:
        dt = pd.to_datetime(s, errors="coerce", utc=True)
    # Use date-only for stable joins
    return dt.dt.strftime("%Y-%m-%d").astype("string")


def _fallback_join_key(df: pd.DataFrame) -> pd.Series:
    md = df.get("match_date")
    if md is None:
        md = df.get("date_GMT", df.get("date", df.get("timestamp", pd.Series(pd.NA, index=df.index))))
    d = _date_only_series(pd.Series(md, index=df.index))

    ht = df.get("home_team_name", df.get("Home", pd.Series("", index=df.index)))
    at = df.get("away_team_name", df.get("Away", pd.Series("", index=df.index)))

    h = _norm_team_series(pd.Series(ht, index=df.index))
    a = _norm_team_series(pd.Series(at, index=df.index))

    return (d.fillna("") + "|" + h.fillna("") + "|" + a.fillna("")).astype("string")

# -----------------------------
# FTR correctness + margin helpers
# -----------------------------

def _real_ftr_from_goals(hg: pd.Series, ag: pd.Series) -> pd.Series:
    hg = pd.to_numeric(hg, errors="coerce")
    ag = pd.to_numeric(ag, errors="coerce")
    diff = hg - ag
    out = pd.Series("DRAW", index=hg.index, dtype="string")
    out = out.mask(diff > 0, "HOME")
    out = out.mask(diff < 0, "AWAY")
    return out


def _compute_ftr_top1_and_margin(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Return (top1_prob, margin=top1-top2) from confidence_home/draw/away.

    Falls back to df['confidence'] with margin=NaN if the 3-way columns are missing.
    """
    if {"confidence_home", "confidence_draw", "confidence_away"}.issubset(df.columns):
        ch = pd.to_numeric(df["confidence_home"], errors="coerce")
        cd = pd.to_numeric(df["confidence_draw"], errors="coerce")
        ca = pd.to_numeric(df["confidence_away"], errors="coerce")
        P = np.vstack([
            ch.fillna(-1.0).to_numpy(dtype=float),
            cd.fillna(-1.0).to_numpy(dtype=float),
            ca.fillna(-1.0).to_numpy(dtype=float),
        ]).T
        # top1 and top2 over the 3-way probs
        top1 = np.max(P, axis=1)
        top2 = np.sort(P, axis=1)[:, -2]
        margin = top1 - top2
        top1_s = pd.Series(top1, index=df.index)
        margin_s = pd.Series(margin, index=df.index)
        return top1_s.clip(0, 1), margin_s.clip(lower=0)

    # Fallback: we can gate on confidence, but margin is unknown
    if "confidence" in df.columns:
        top1_s = pd.to_numeric(df["confidence"], errors="coerce").clip(0, 1)
    else:
        top1_s = pd.Series(np.nan, index=df.index)
    margin_s = pd.Series(np.nan, index=df.index)
    return top1_s, margin_s


def build_ftr_gates(ftr_stats_csv: Path, leagues: Optional[List[str]], min_hit: float, min_n: int) -> pd.DataFrame:
    df = pd.read_csv(ftr_stats_csv)
    df["conf_band"] = df["conf_band"].astype(str)
    df = df[df["conf_band"].str.upper() != "ALL"].copy()

    if leagues:
        df = df[df["league"].isin(leagues)].copy()

    df["band_lo"] = df["conf_band"].map(_band_lower)
    df["hit_rate"] = pd.to_numeric(df["hit_rate"], errors="coerce")
    df["n"] = pd.to_numeric(df["n"], errors="coerce").fillna(0).astype(int)

    out_rows = []
    for league, g in df.groupby("league"):
        g = g.dropna(subset=["band_lo","hit_rate"]).copy()
        g = g.sort_values("band_lo")  # min band first

        pick = g[(g["hit_rate"] >= min_hit) & (g["n"] >= min_n)].head(1)
        if pick.empty:
            # fallback: best hit_rate among bands with enough n
            pick2 = g[g["n"] >= min_n].sort_values(["hit_rate","band_lo"], ascending=[False, True]).head(1)
            if pick2.empty:
                out_rows.append({"league": league, "ftr_gate_min_conf": np.nan, "conf_band": "", "n": 0, "hit_rate": np.nan, "avg_confidence": np.nan, "avg_odds": np.nan, "status": "SKIP"})
            else:
                r = pick2.iloc[0]
                out_rows.append({"league": league, "ftr_gate_min_conf": float(r["band_lo"]), "conf_band": str(r["conf_band"]), "n": int(r["n"]), "hit_rate": float(r["hit_rate"]), "avg_confidence": float(r.get("avg_confidence", np.nan)), "avg_odds": float(r.get("avg_odds", np.nan)), "status": "BEST_EFFORT"})
        else:
            r = pick.iloc[0]
            out_rows.append({"league": league, "ftr_gate_min_conf": float(r["band_lo"]), "conf_band": str(r["conf_band"]), "n": int(r["n"]), "hit_rate": float(r["hit_rate"]), "avg_confidence": float(r.get("avg_confidence", np.nan)), "avg_odds": float(r.get("avg_odds", np.nan)), "status": "OK"})

    out = pd.DataFrame(out_rows).sort_values("league")
    return out


# ------------------------------------------------
# FTR margin gate builder
# ------------------------------------------------
def build_ftr_margin_gates(
    meta_path: Path,
    matches_root: Path,
    leagues: Optional[List[str]],
    ftr_gates_df: pd.DataFrame,
    *,
    min_hit: float,
    min_n: int,
    multi_season: bool,
    margin_candidates: List[float],
    topk: int = 50,
) -> pd.DataFrame:
    """Choose a per-league FTR margin gate given an existing min_conf gate.

    We evaluate realised joins from META (market=ftr) against realised matches.
    """
    try:
        meta = pd.read_csv(meta_path)
    except Exception:
        return pd.DataFrame()

    if "market" not in meta.columns:
        return pd.DataFrame()

    meta["market"] = meta["market"].astype(str).str.strip().str.lower()
    ftr = meta[meta["market"] == "ftr"].copy()
    if leagues:
        ftr = ftr[ftr["league"].isin(leagues)].copy()
    if ftr.empty:
        return pd.DataFrame()

    # predicted label
    pred_lab = ftr.get("selection", ftr.get("ftr_outcome_label", ""))
    ftr["pred_outcome"] = pred_lab.astype(str).str.strip().str.upper()

    # compute top1 + margin (used for gating)
    top1, margin = _compute_ftr_top1_and_margin(ftr)
    ftr["_top1"] = pd.to_numeric(top1, errors="coerce")
    ftr["_margin"] = pd.to_numeric(margin, errors="coerce")

    # use odds for reporting if present
    if "od" in ftr.columns:
        ftr["_od"] = pd.to_numeric(ftr["od"], errors="coerce")
    else:
        ftr["_od"] = np.nan

    # build a quick lookup for min_conf per league
    conf_map: Dict[str, float] = {}
    for _, r in ftr_gates_df.iterrows():
        try:
            conf_map[str(r["league"])] = float(r["ftr_gate_min_conf"]) if pd.notna(r["ftr_gate_min_conf"]) else 0.0
        except Exception:
            conf_map[str(r.get("league", ""))] = 0.0

    rows: List[Dict[str, Any]] = []

    # normalize candidate list
    cands = sorted({float(x) for x in margin_candidates if x is not None and np.isfinite(float(x)) and float(x) >= 0.0})
    if not cands:
        cands = [0.0]

    for league in sorted(ftr["league"].dropna().unique().tolist()):
        sub = ftr[ftr["league"] == league].copy()
        if sub.empty:
            continue

        # load realised matches
        matches = _load_all_matches_csvs(matches_root, league)
        if matches is None or matches.empty:
            rows.append({"league": league, "ftr_gate_min_margin": np.nan, "n_margin": 0, "hit_rate_margin": np.nan, "avg_confidence_margin": np.nan, "avg_odds_margin": np.nan, "status_margin": "SKIP"})
            continue

        matches = matches.loc[_realised_mask(matches)].copy()
        if matches.empty:
            rows.append({"league": league, "ftr_gate_min_margin": np.nan, "n_margin": 0, "hit_rate_margin": np.nan, "avg_confidence_margin": np.nan, "avg_odds_margin": np.nan, "status_margin": "SKIP"})
            continue

        # join
        joined = sub.merge(
            matches[["fixture_key", "home_team_goal_count", "away_team_goal_count"]],
            on="fixture_key",
            how="left",
        )
        joined = joined[joined["home_team_goal_count"].notna() & joined["away_team_goal_count"].notna()].copy()
        if joined.empty:
            rows.append({"league": league, "ftr_gate_min_margin": np.nan, "n_margin": 0, "hit_rate_margin": np.nan, "avg_confidence_margin": np.nan, "avg_odds_margin": np.nan, "status_margin": "SKIP"})
            continue

        # realised outcome
        real = _real_ftr_from_goals(joined["home_team_goal_count"], joined["away_team_goal_count"]).astype(str).str.upper()
        pred = joined["pred_outcome"].astype(str).str.upper()
        correct = (pred == real)

        # base conf gate from ftr_gates_df (already band-selected)
        min_conf = float(conf_map.get(str(league), 0.0))

        # take a stable top-k slice like the FTR meta backtest
        joined["_top1"] = pd.to_numeric(joined["_top1"], errors="coerce")
        joined["_margin"] = pd.to_numeric(joined["_margin"], errors="coerce")
        joined["_od"] = pd.to_numeric(joined.get("_od"), errors="coerce")

        joined = joined.sort_values("_top1", ascending=False).head(int(topk)).copy()
        correct = correct.loc[joined.index]

        # evaluate candidates: pick the smallest margin that meets (min_hit, min_n)
        best_row = None
        for mg in cands:
            m = (joined["_top1"] >= min_conf) & (joined["_margin"].fillna(-1.0) >= float(mg))
            jj = joined[m].copy()
            if jj.empty:
                continue
            n = int(len(jj))
            hit = float(correct.loc[jj.index].mean())
            if (n >= int(min_n)) and (hit >= float(min_hit)):
                best_row = {"league": league, "ftr_gate_min_margin": float(mg), "n_margin": n, "hit_rate_margin": hit,
                            "avg_confidence_margin": float(pd.to_numeric(jj["_top1"], errors="coerce").mean()),
                            "avg_odds_margin": float(pd.to_numeric(jj["_od"], errors="coerce").mean()),
                            "status_margin": "OK"}
                break

        # best-effort if nothing met constraints
        if best_row is None:
            scored: List[Tuple[float, int, float]] = []
            for mg in cands:
                m = (joined["_top1"] >= min_conf) & (joined["_margin"].fillna(-1.0) >= float(mg))
                jj = joined[m].copy()
                if jj.empty:
                    continue
                n = int(len(jj))
                hit = float(correct.loc[jj.index].mean())
                scored.append((float(mg), n, hit))

            if not scored:
                best_row = {"league": league, "ftr_gate_min_margin": np.nan, "n_margin": 0, "hit_rate_margin": np.nan, "avg_confidence_margin": np.nan, "avg_odds_margin": np.nan, "status_margin": "SKIP"}
            else:
                # prefer: hit desc, n desc, margin asc (smaller margin keeps coverage)
                scored.sort(key=lambda t: (-t[2], -t[1], t[0]))
                mg, n, hit = scored[0]
                # compute averages for that chosen mg
                m = (joined["_top1"] >= min_conf) & (joined["_margin"].fillna(-1.0) >= float(mg))
                jj = joined[m].copy()
                best_row = {"league": league, "ftr_gate_min_margin": float(mg), "n_margin": int(len(jj)), "hit_rate_margin": float(correct.loc[jj.index].mean()),
                            "avg_confidence_margin": float(pd.to_numeric(jj["_top1"], errors="coerce").mean()),
                            "avg_odds_margin": float(pd.to_numeric(jj["_od"], errors="coerce").mean()),
                            "status_margin": "BEST_EFFORT"}

        rows.append(best_row)

    return pd.DataFrame(rows).sort_values("league")


def build_side_gates(meta_path: Path, matches_root: Path, leagues: Optional[List[str]], min_hit: float, min_n: int, multi_season: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    meta = pd.read_csv(meta_path)
    meta["market"] = meta["market"].astype(str).str.strip().str.lower()

    # Normalize ALLMARKETS OU25 rows into deploy-gates side markets.
    # bookie_allmarkets exports market='ou25' with directional bookie_pick,
    # while the learned side-gate engine expects over25 / under25 labels.
    if "bookie_pick" not in meta.columns:
        meta["bookie_pick"] = ""
    pick_norm = meta["bookie_pick"].astype("string").fillna("").str.upper().str.strip()

    is_ou25 = meta["market"].eq("ou25")
    meta.loc[is_ou25 & pick_norm.eq("OVER25"), "market"] = "over25"
    meta.loc[is_ou25 & pick_norm.eq("UNDER25"), "market"] = "under25"

    # Only side markets
    side = meta[meta["market"].isin(["over25", "under25", "btts", "btts_no"])].copy()
    if leagues:
        side = side[side["league"].isin(leagues)].copy()

    if side.empty:
        return pd.DataFrame(), pd.DataFrame()

    gates_rows = []
    per_label_rows = []

    # Ensure stable output schema even when empty
    GATES_COLS = [
        "league","market","allowed_labels","n","hit_rate","baseline_rate","lift_pp","status",
        "policy_total_n","policy_eligible_n","policy_blocked_n","policy_note",
    ]
    LABEL_COLS = ["league","market","label","n","hit_rate"]

    for league in sorted(side["league"].unique().tolist()):
        league_side = side[side["league"] == league].copy()
        if league_side.empty:
            continue

        if multi_season:
            matches = _load_all_matches_csvs(matches_root, league)
        else:
            # If you ever want non-multi-season here, you can add a latest-csv loader.
            matches = _load_all_matches_csvs(matches_root, league)

        if matches is None or matches.empty:
            continue

        matches = matches.loc[_realised_mask(matches)].copy()
        if matches.empty:
            continue

        # Baselines (context)
        hg_all = pd.to_numeric(matches["home_team_goal_count"], errors="coerce").fillna(0)
        ag_all = pd.to_numeric(matches["away_team_goal_count"], errors="coerce").fillna(0)
        g_all = hg_all + ag_all
        base_over = float((g_all >= 3).mean())
        base_under = float((g_all <= 2).mean())
        base_btts = float(((hg_all >= 1) & (ag_all >= 1)).mean())
        base_no = 1.0 - base_btts

        for mkt in ["over25", "under25", "btts", "btts_no"]:
            sub = league_side[league_side["market"] == mkt].copy()
            if sub.empty:
                continue
            # Primary join on canonical fixture_key
            joined = sub.merge(
                matches[["fixture_key", "home_team_goal_count", "away_team_goal_count"]],
                on="fixture_key",
                how="left",
            )
            joined = joined[joined["home_team_goal_count"].notna() & joined["away_team_goal_count"].notna()].copy()

            # Fallback join if fixture_key drifted (use date-only + normalized team names)
            if joined.empty:
                sub2 = sub.copy()
                m2 = matches.copy()
                sub2["__jk"] = _fallback_join_key(sub2)
                m2["__jk"] = _fallback_join_key(m2)
                joined = sub2.merge(
                    m2[["__jk", "home_team_goal_count", "away_team_goal_count"]],
                    on="__jk",
                    how="left",
                )
                joined = joined[joined["home_team_goal_count"].notna() & joined["away_team_goal_count"].notna()].copy()

            if joined.empty:
                continue

            hg = pd.to_numeric(joined["home_team_goal_count"], errors="coerce").fillna(0)
            ag = pd.to_numeric(joined["away_team_goal_count"], errors="coerce").fillna(0)
            ok = _correct_from_goals(mkt, hg, ag)
            if ok is None:
                continue

            # Apply sanity gates for known failure modes (BTTS YES + OVER2.5)
            sanity = pd.Series(True, index=joined.index)
            if mkt == "btts":
                sanity = _btts_yes_sanity_mask(joined)
            elif mkt == "over25":
                sanity = _over25_sanity_mask(joined)

            if sanity is not None and bool((~sanity).any()):
                joined = joined.loc[sanity].copy()
                ok = ok.loc[joined.index]
                if joined.empty:
                    continue

            # Keep the full realised/sanity-passed sample for transparency reporting,
            # but restrict learned-gate selection for OU25 to rows allowed by policy.
            joined_report = joined.copy()
            ok_report = ok.loc[joined_report.index]

            policy_total_n = int(len(joined_report))
            policy_eligible_n = int(len(joined_report))
            policy_blocked_n = 0
            policy_note = ""

            if mkt in ("over25", "under25"):
                policy_mask = _ou25_policy_eligible_mask(joined_report, mkt).fillna(False)
                policy_note = _ou25_policy_note(joined_report, mkt)
                policy_eligible_n = int(policy_mask.sum())
                policy_blocked_n = int((~policy_mask).sum())
                joined = joined_report.loc[policy_mask].copy()
                ok = ok_report.loc[joined.index]
            else:
                joined = joined_report.copy()
                ok = ok_report.loc[joined.index]

            label_col = "signal_over25" if mkt in ("over25","under25") else "signal_btts"

            # --- Fallback: if signal labels are missing OR unusable, gate by a confidence/probability threshold instead ---
            # Usability rule: labels are considered unusable when they are all NA/blank/NEUTRAL.
            labels_present = (label_col in joined.columns)
            labels_usable = False
            if labels_present:
                try:
                    lab_u = joined[label_col].astype("string").fillna("").str.upper().str.strip()
                    lab_u = lab_u.replace({"NAN": "", "NONE": "", "NA": ""})
                    # If *any* row has a directional strong label, treat labels as usable.
                    labels_usable = bool((~lab_u.isin(["", "NEUTRAL"])) .any())
                except Exception:
                    labels_usable = False

            if (not labels_present) or (not labels_usable):
                # Prefer explicit confidence columns, else fall back to ALLMARKETS-side prob columns.
                conf = pd.to_numeric(
                    joined.get(
                        "confidence",
                        joined.get(
                            "p_model",
                            joined.get("model_p_for_bookie", np.nan),
                        ),
                    ),
                    errors="coerce",
                )

                # If no confidence exists, we can't build any gate
                if conf.isna().all():
                    continue

                # Candidate thresholds (match your side_meta_backtest bands)
                thr_grid = [0.70, 0.80, 0.85, 0.90]

                best = None
                for thr in thr_grid:
                    jj = joined[conf >= thr].copy()
                    if jj.empty:
                        continue
                    hit = float(ok.loc[jj.index].mean())
                    n = int(len(jj))
                    if (n >= min_n) and (hit >= min_hit):
                        best = (thr, n, hit, "OK")
                        break

                # Best-effort if none meet constraints
                if best is None:
                    scored = []
                    for thr in thr_grid:
                        jj = joined[conf >= thr].copy()
                        if jj.empty:
                            continue
                        hit = float(ok.loc[jj.index].mean())
                        n = int(len(jj))
                        scored.append((thr, n, hit))
                    if scored:
                        scored.sort(key=lambda t: (t[2], t[1], -t[0]), reverse=True)  # hit desc, n desc, thr asc-ish
                        thr, n, hit = scored[0]
                        status = "BEST_EFFORT_NO_LABELS"
                    else:
                        thr, n, hit = (np.nan, 0, np.nan)
                        status = "SKIP_NO_LABELS"
                else:
                    thr, n, hit, status = best

                base = {"over25": base_over, "under25": base_under, "btts": base_btts, "btts_no": base_no}.get(mkt, np.nan)
                lift = (hit - base) if np.isfinite(hit) and np.isfinite(base) else np.nan

                if mkt in ("over25", "under25") and int(policy_eligible_n) == 0:
                    status = "POLICY_BLOCKED"
                    thr = np.nan
                    n = 0
                    hit = np.nan
                    lift = np.nan

                gates_rows.append({
                    "league": league,
                    "market": mkt,
                    "allowed_labels": f"PMODEL>={thr:.2f}" if np.isfinite(thr) else "",
                    "n": int(n),
                    "hit_rate": float(hit) if np.isfinite(hit) else np.nan,
                    "baseline_rate": float(base) if np.isfinite(base) else np.nan,
                    "lift_pp": float(lift) if np.isfinite(lift) else np.nan,
                    "status": status,
                    "policy_total_n": int(policy_total_n),
                    "policy_eligible_n": int(policy_eligible_n),
                    "policy_blocked_n": int(policy_blocked_n),
                    "policy_note": policy_note,
                })
                continue

            joined_report[label_col] = joined_report[label_col].astype(str)
            joined_report[label_col] = joined_report[label_col].replace({"": "NA", "nan": "NA", "None": "NA"})

            if label_col in joined.columns:
                joined[label_col] = joined[label_col].astype(str)
                joined[label_col] = joined[label_col].replace({"": "NA", "nan": "NA", "None": "NA"})

            # per-label stats (for transparency) stay on the full realised sample,
            # even when policy blocks rows from learned-gate eligibility.
            grp = joined_report.assign(correct=ok_report.astype(int)).groupby(label_col)["correct"].agg(["count","mean"]).reset_index()
            grp = grp.rename(columns={label_col: "label", "count": "n", "mean": "hit_rate"})
            for _, rr in grp.iterrows():
                per_label_rows.append({
                    "league": league,
                    "market": mkt,
                    "label": str(rr["label"]),
                    "n": int(rr["n"]),
                    "hit_rate": float(rr["hit_rate"]),
                })

            # choose gate
            candidates = _side_gate_candidates(mkt)
            best = None

            if not joined.empty:
                for labels in candidates:
                    mask = joined[label_col].isin(list(labels))
                    jj = joined[mask].copy()
                    if jj.empty:
                        continue
                    hit = float(ok.loc[jj.index].mean())
                    n = int(len(jj))
                    if (n >= min_n) and (hit >= min_hit):
                        best = (labels, n, hit)
                        break  # strict-first

            # If nothing meets constraints, choose best-effort (max hit, then max n) among candidate sets
            if best is None:
                scored = []
                if not joined.empty:
                    for labels in candidates:
                        mask = joined[label_col].isin(list(labels))
                        jj = joined[mask].copy()
                        if jj.empty:
                            continue
                        hit = float(ok.loc[jj.index].mean())
                        n = int(len(jj))
                        scored.append((labels, n, hit))
                if scored:
                    scored.sort(key=lambda x: (x[2], x[1], -len(x[0])), reverse=True)
                    labels, n, hit = scored[0]
                    status = "BEST_EFFORT"
                else:
                    labels, n, hit = tuple(), 0, np.nan
                    status = "POLICY_BLOCKED" if (mkt in ("over25", "under25") and int(policy_eligible_n) == 0) else "SKIP"
            else:
                labels, n, hit = best
                status = "OK"

            # lift vs baseline
            base = {"over25": base_over, "under25": base_under, "btts": base_btts, "btts_no": base_no}.get(mkt, np.nan)
            lift = (hit - base) if np.isfinite(hit) and np.isfinite(base) else np.nan

            gates_rows.append({
                "league": league,
                "market": mkt,
                "allowed_labels": ",".join(labels) if labels else "",
                "n": int(n),
                "hit_rate": float(hit) if np.isfinite(hit) else np.nan,
                "baseline_rate": float(base) if np.isfinite(base) else np.nan,
                "lift_pp": float(lift) if np.isfinite(lift) else np.nan,
                "status": status,
                "policy_total_n": int(policy_total_n),
                "policy_eligible_n": int(policy_eligible_n),
                "policy_blocked_n": int(policy_blocked_n),
                "policy_note": policy_note,
            })

    gates_df = pd.DataFrame(gates_rows).reindex(columns=GATES_COLS)
    per_label_df = pd.DataFrame(per_label_rows).reindex(columns=LABEL_COLS)

    if gates_df.empty:
        return gates_df, per_label_df

    return (
        gates_df.sort_values(["league","market"]),
        per_label_df.sort_values(["league","market","label"]) if not per_label_df.empty else per_label_df,
    )


def write_report(outdir: Path, ftr_gates: pd.DataFrame, side_gates: pd.DataFrame) -> Path:
    lines = []
    lines.append(_center("DEPLOY GATES SUMMARY"))
    lines.append(_center("FTR + SIDE MARKETS (BAND / SIGNAL LABELS)"))
    lines.append("")
    lines.append("Legend: hit_rate is on realised joins; lift_pp is vs league baseline for that market.")
    lines.append("")
    lines.append("NOTE: Sanity gates ACTIVE for side markets: BTTS YES (fts_max<=0.55 & (1-p_home_fts)*(1-p_away_fts)>=0.20 & abs(power_diff)<=22) and OVER2.5 (abs(power_diff)<=22 & abs(ppg_diff_pre)<=0.70 & Poisson P(total>=3)>=0.58).")
    lines.append("")

    leagues = sorted(set(ftr_gates["league"].tolist()) | set(side_gates["league"].tolist()))
    for league in leagues:
        lines.append("=" * 86)
        lines.append(_center(f"LEAGUE: {league.upper()}"))
        lines.append("=" * 86)

        # FTR
        fg = ftr_gates[ftr_gates["league"] == league]
        if fg.empty:
            lines.append("FTR: (no data)")
        else:
            r = fg.iloc[0]
            min_conf = r["ftr_gate_min_conf"] if pd.notna(r.get("ftr_gate_min_conf")) else "NA"
            min_margin = r.get("ftr_gate_min_margin", np.nan)
            # Prefer margin-filtered metrics only when they are present AND not NaN
            hit_m = r.get("hit_rate_margin", np.nan)
            n_m = r.get("n_margin", np.nan)
            st_m = r.get("status_margin", "")

            hit_show = hit_m if pd.notna(hit_m) else r.get("hit_rate", np.nan)
            n_show = n_m if pd.notna(n_m) else r.get("n", 0)
            status_show = st_m if (isinstance(st_m, str) and st_m.strip()) else r.get("status", "")

            # Robust formatting: NaNs can appear for n/hit when a league is SKIP
            try:
                n_show_int = int(n_show) if pd.notna(n_show) else 0
            except Exception:
                n_show_int = 0

            try:
                hit_show_val = float(hit_show) if pd.notna(hit_show) else np.nan
            except Exception:
                hit_show_val = np.nan

            if pd.notna(min_margin):
                lines.append(
                    f"FTR  | conf≥{min_conf} | margin≥{float(min_margin):.2f} | band={r.get('conf_band','')} | "
                    f"n={n_show_int} | hit={hit_show_val:.3f} | status={status_show}"
                )
            else:
                lines.append(
                    f"FTR  | conf≥{min_conf} | band={r.get('conf_band','')} | n={n_show_int} | "
                    f"hit={hit_show_val:.3f} | status={status_show}"
                )

        # Side markets
        sg = side_gates[side_gates["league"] == league]
        if sg.empty:
            lines.append("SIDE | (no data)")
        else:
            for mkt in ["over25","under25","btts","btts_no"]:
                rr = sg[sg["market"] == mkt]
                if rr.empty:
                    continue
                r = rr.iloc[0]
                if pd.isna(r["hit_rate"]):
                    lines.append(f"{mkt.upper():7} | SKIP")
                else:
                    lines.append(
                        f"{mkt.upper():7} | labels=[{r['allowed_labels']}] | n={int(r['n']) if pd.notna(r['n']) else 0} | "
                        f"hit={r['hit_rate']:.3f} | base={r['baseline_rate']:.3f} | lift={r['lift_pp']:+.3f} | {r['status']}"
                    )
        lines.append("")

    out_path = outdir / "DEPLOY_GATES_REPORT.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-path", required=True)
    ap.add_argument("--matches-root", default="Matches")
    ap.add_argument("--ftr-stats", required=True, help="Path to predictions_output/FTR_META_backtest_stats.csv")
    ap.add_argument(
        "--side-stats",
        default="predictions_output/SIDE_META_backtest_stats.csv",
        help="Path to SIDE_META_backtest_stats.csv (optional; for compatibility; current script computes side gates from META)",
    )
    ap.add_argument("--leagues", default=None, help="Comma-separated leagues")
    ap.add_argument("--outdir", default="predictions_output")

    ap.add_argument("--multi-season", action="store_true")

    ap.add_argument("--min-hit-ftr", type=float, default=0.85, help="Target hit-rate for FTR gates (default 0.85; override for experimental runs)")
    ap.add_argument("--min-n-ftr", type=int, default=10, help="Minimum sample size per FTR band/gate")

    ap.add_argument("--min-hit-side", type=float, default=0.70)
    ap.add_argument("--min-n-side", type=int, default=10)

    args = ap.parse_args()

    leagues = None
    if args.leagues:
        leagues = [s.strip() for s in str(args.leagues).split(",") if s.strip()]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Optional: load side stats CSV (for compatibility with earlier CLI usage).
    # NOTE: This script currently computes side gates from META + realised matches,
    # so side-stats is not required. We still accept it to avoid CLI errors and
    # to sanity-check that the file exists.
    try:
        side_stats_path = Path(args.side_stats)
        if side_stats_path.exists():
            _side_df = pd.read_csv(side_stats_path)
            # Light validation only
            if isinstance(_side_df, pd.DataFrame) and len(_side_df):
                print(f"ℹ️ Loaded side stats: {side_stats_path} (rows={len(_side_df)})")
        else:
            print(f"ℹ️ side-stats not found (ok): {side_stats_path}")
    except Exception as _e:
        print(f"ℹ️ side-stats load skipped: {_e}")

    ftr_gates = build_ftr_gates(Path(args.ftr_stats), leagues, args.min_hit_ftr, args.min_n_ftr)
    # Optional: derive a per-league FTR margin gate (top1-top2) using META + realised matches
    margin_candidates = [0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
    ftr_margin = build_ftr_margin_gates(
        Path(args.meta_path),
        Path(args.matches_root),
        leagues,
        ftr_gates,
        min_hit=float(args.min_hit_ftr),
        min_n=int(args.min_n_ftr),
        multi_season=bool(args.multi_season),
        margin_candidates=margin_candidates,
        topk=50,
    )
    if ftr_margin is not None and not ftr_margin.empty:
        ftr_gates = ftr_gates.merge(ftr_margin, on="league", how="left")

    side_gates, side_by_label = build_side_gates(Path(args.meta_path), Path(args.matches_root), leagues, args.min_hit_side, args.min_n_side, args.multi_season)

    ftr_out = outdir / "DEPLOY_GATES_FTR.csv"
    side_out = outdir / "DEPLOY_GATES_SIDE.csv"
    side_label_out = outdir / "DEPLOY_GATES_SIDE_BY_LABEL.csv"

    ftr_gates.to_csv(ftr_out, index=False)
    side_gates.to_csv(side_out, index=False)
    side_by_label.to_csv(side_label_out, index=False)

    report_path = write_report(outdir, ftr_gates, side_gates)

    print("✅ Wrote:")
    print(" -", ftr_out)
    print(" -", side_out)
    print(" -", side_label_out)
    print(" -", report_path)



# ============================================================================
# Runtime deploy policy gates (RulePack v1.1)
#
# These are NOT learned / backtested gates. They are deterministic safety +
# policy rules applied at deploy time using columns already present in
# ALLMARKETS exports.
#
# Output columns added by `compute_deterministic_deploy_vetoes(df)`:
#   - signal_mismatch_flag (0/1)
#   - deterministic_pass (0/1)
#   - deterministic_veto_reason (pipe-separated string)
#   - deterministic_adjustment (-1/0/+1)
#   - deterministic_adjust_reason (pipe-separated string)
#
# IMPORTANT: These constants are prefixed with DEPLOY_ to avoid colliding with
# the earlier "learned gates" / backtest helper constants used elsewhere in
# this file.
# ============================================================================

# --- Signal mismatch policy ---
DEPLOY_SIG_EMPTY_COUNTS_AS_MISMATCH = False

# --- BTTS YES ---
DEPLOY_BTTS_YES_FTS_MAX_VETO = 0.55
DEPLOY_BTTS_YES_CS1_TRAP_SET = {"0-0", "1-0", "0-1"}
DEPLOY_BTTS_YES_CS1P_MIN_VETO = 0.18
DEPLOY_BTTS_YES_DOWNGRADE_XGSUM_MAX = 2.75
DEPLOY_BTTS_YES_DOWNGRADE_P00_MIN = 0.065

# --- BTTS NO ---
DEPLOY_BTTS_NO_VETO_XGSUM_MIN = 3.00
DEPLOY_BTTS_NO_VETO_P00_MAX = 0.060
DEPLOY_BTTS_NO_VETO_PROB_BTTS_V2_MIN = 0.53
DEPLOY_BTTS_NO_BOOST_FTS_MAX_MIN = 0.33
DEPLOY_BTTS_NO_BOOST_P00_MIN = 0.085

# --- OU2.5 OVER ---
DEPLOY_OVER25_VETO_P00_MIN = 0.18
DEPLOY_OVER25_DOWNGRADE_CS1_SET = {"1-1", "2-0", "0-2"}
DEPLOY_OVER25_DOWNGRADE_CS1P_MIN = 0.105


# --- FTR ---
DEPLOY_FTR_CLOSE_PD_ABS_MAX = 10.0
DEPLOY_FTR_CLOSE_MARGIN_MAX = 0.10
DEPLOY_FTR_CHAOS_MARGIN_MAX = 0.12

# Optional dominant override (off by default for v1.1)
DEPLOY_FTR_USE_DOMINANT_OVERRIDE = False
DEPLOY_FTR_OVERRIDE_PD_ABS_MIN = 25.0
DEPLOY_FTR_OVERRIDE_MARGIN_MIN = 0.18


# --- FTR_CONSENSUS brutal eligibility (tier/lane/margin/p00) ---
# Applied ONLY when the row looks like it came from a consensus export.
# (Detected via od_source containing 'consensus' or presence of consensus_* columns.)
DEPLOY_FTR_CONS_TIER_ALLOWED = {"ELITE", "MEDIUM"}
DEPLOY_FTR_CONS_LANE_VETO = {"CLOSE"}
DEPLOY_FTR_CONS_MARGIN_MIN_DEFAULT = 0.08
DEPLOY_FTR_CONS_MARGIN_MIN_EPL_L1 = 0.10
DEPLOY_FTR_CONS_P00_MAX = 0.09
DEPLOY_FTR_CONS_IMPLIED_PROB_DIFF_MIN = 0.10   # coinflip microstructure veto
DEPLOY_FTR_CONS_BOOKIE_SPREAD_MIN = 0.12       # very tight 1X2 market
DEPLOY_FTR_CONS_ODDS_DIFF_MIN = 1.00           # small odds separation
DEPLOY_FTR_CONS_TRUE_CLOSE_REASON_VETO = True  # veto if reason_codes contains TRUE_CLOSE
DEPLOY_FTR_CONS_CLOSE_OVERRIDE_MARGIN = 0.20  # only used if you later allow CLOSE overrides


def _s(df: pd.DataFrame, col: str) -> pd.Series:
    """String series helper (never fails)."""
    if col not in df.columns:
        return pd.Series("", index=df.index, dtype="string")
    return df[col].astype("string").fillna("")


def _n(df: pd.DataFrame, col: str) -> pd.Series:
    """Numeric series helper (never fails)."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _infer_ou_dir_from_signal(sig: pd.Series) -> pd.Series:
    """Infer OVER/UNDER direction from signal_over25 labels."""
    x = sig.astype("string").fillna("").str.upper().str.strip()
    x = x.replace({"NAN": "", "NONE": "", "NA": ""})
    out = pd.Series("", index=x.index, dtype="string")
    out = out.mask(x.str.contains("OVER", regex=False), "OVER")
    out = out.mask(x.str.contains("UNDER", regex=False), "UNDER")
    return out


def _infer_btts_dir_from_signal(sig: pd.Series) -> pd.Series:
    """Infer YES/NO direction from signal_btts labels."""
    x = sig.astype("string").fillna("").str.upper().str.strip()
    x = x.replace({"NAN": "", "NONE": "", "NA": ""})
    out = pd.Series("", index=x.index, dtype="string")
    out = out.mask(x.str.contains("YES", regex=False), "YES")
    out = out.mask(x.str.contains("NO", regex=False), "NO")
    return out


def _is_strong_yes_signal(sig: pd.Series) -> pd.Series:
    """True when signal_btts indicates a strong YES lane."""
    x = sig.astype("string").fillna("").str.upper().str.strip()
    # Examples: VERY_STRONG_YES, STRONG_YES
    return (x.str.contains("YES", regex=False) & x.str.contains("STRONG", regex=False))


def compute_deterministic_deploy_vetoes(df: pd.DataFrame) -> pd.DataFrame:
    """Compute deterministic deploy veto + adjustment reasons (RulePack v1.1).

    Returns a copy of df with:
      - signal_mismatch_flag (0/1)
      - deterministic_pass (0/1)
      - deterministic_veto_reason (pipe-separated string)
      - deterministic_adjustment (-1/0/+1)
      - deterministic_adjust_reason (pipe-separated string)

    Notes:
      - Missing columns do NOT veto unless explicitly stated.
      - Signal mismatch uses the existing signal columns when present.
      - Adjustments (DOWNGRADE/BOOST) are informational; they do not veto.
    """
    if df is None or df.empty:
        out0 = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
        if isinstance(out0, pd.DataFrame) and not out0.empty:
            out0["signal_mismatch_flag"] = 0
            out0["deterministic_pass"] = 1
            out0["deterministic_veto_reason"] = ""
            out0["deterministic_adjustment"] = 0
            out0["deterministic_adjust_reason"] = ""
        return out0

    out = df.copy()

    # Demotion marker (used by deploy_rulebook tiering). Not a hard veto.
    if "demote_to_observe" not in out.columns:
        out["demote_to_observe"] = 0
    if "deterministic_warn_reason" not in out.columns:
        out["deterministic_warn_reason"] = ""

    market = _s(out, "market").str.lower().str.strip()
    pick = _s(out, "bookie_pick").str.upper().str.strip()

    # --- Base numeric features (safe if missing) ---
    ph_fts = _n(out, "p_home_fts")
    pa_fts = _n(out, "p_away_fts")
    fts_max = pd.concat([ph_fts, pa_fts], axis=1).max(axis=1)

    p00 = _n(out, "p00_est")
    xgsum = _n(out, "exp_goals_sum")

    cs1 = _s(out, "cs1").str.strip()
    cs1_p = _n(out, "cs1_p")

    prob_btts_v2 = _n(out, "prob_btts_v2")

    # --- Signal mismatch veto (OU25 + BTTS) ---
    has_sig_over = "signal_over25" in out.columns
    has_sig_btts = "signal_btts" in out.columns

    sig_over = _s(out, "signal_over25")
    sig_btts = _s(out, "signal_btts")

    ou_dir = _infer_ou_dir_from_signal(sig_over)
    btts_dir = _infer_btts_dir_from_signal(sig_btts)

    is_ou = market.eq("ou25") & pick.isin(["OVER25", "UNDER25"])
    expected_ou = pd.Series("", index=out.index, dtype="string")
    expected_ou = expected_ou.mask(is_ou & pick.eq("OVER25"), "OVER")
    expected_ou = expected_ou.mask(is_ou & pick.eq("UNDER25"), "UNDER")

    is_btts = market.eq("btts") & pick.isin(["YES", "NO"])
    expected_btts = pd.Series("", index=out.index, dtype="string")
    expected_btts = expected_btts.mask(is_btts & pick.eq("YES"), "YES")
    expected_btts = expected_btts.mask(is_btts & pick.eq("NO"), "NO")

    ou_mismatch = pd.Series(False, index=out.index)
    btts_mismatch = pd.Series(False, index=out.index)

    if bool(is_ou.any()) and has_sig_over:
        if bool(DEPLOY_SIG_EMPTY_COUNTS_AS_MISMATCH):
            ou_mismatch = is_ou & ((ou_dir.eq("")) | (ou_dir.ne(expected_ou)))
        else:
            ou_mismatch = is_ou & (ou_dir.ne("") & ou_dir.ne(expected_ou))

    if bool(is_btts.any()) and has_sig_btts:
        if bool(DEPLOY_SIG_EMPTY_COUNTS_AS_MISMATCH):
            btts_mismatch = is_btts & ((btts_dir.eq("")) | (btts_dir.ne(expected_btts)))
        else:
            btts_mismatch = is_btts & (btts_dir.ne("") & btts_dir.ne(expected_btts))

    signal_mismatch = ou_mismatch | btts_mismatch

    # --- VETO rules (RulePack v1.1) ---
    veto_flags: List[pd.Series] = []

    # 0) Signal mismatch veto
    veto_flags.append(signal_mismatch)

    # BTTS YES
    is_btts_yes = market.eq("btts") & pick.eq("YES")
    btts_yes_veto_fts = is_btts_yes & fts_max.notna() & (fts_max >= float(DEPLOY_BTTS_YES_FTS_MAX_VETO))
    # Guard: cs1_p should be a true correct-score probability (typically <= ~0.35).
    # If cs1_p is mis-scaled (e.g. copied from another column), do not blanket-veto.
    cs1_p_ok = cs1_p.notna() & (cs1_p >= 0) & (cs1_p <= 0.35)
    btts_yes_veto_cs1 = is_btts_yes & cs1.ne("") & cs1.isin(list(DEPLOY_BTTS_YES_CS1_TRAP_SET)) & cs1_p_ok & (cs1_p >= float(DEPLOY_BTTS_YES_CS1P_MIN_VETO))

    veto_flags.append(btts_yes_veto_fts)
    veto_flags.append(btts_yes_veto_cs1)

    # BTTS NO
    is_btts_no = market.eq("btts") & pick.eq("NO")
    btts_no_veto_dead = is_btts_no & xgsum.notna() & p00.notna() & (xgsum >= float(DEPLOY_BTTS_NO_VETO_XGSUM_MIN)) & (p00 <= float(DEPLOY_BTTS_NO_VETO_P00_MAX))

    btts_no_veto_prob = pd.Series(False, index=out.index)
    if "prob_btts_v2" in out.columns:
        btts_no_veto_prob = is_btts_no & prob_btts_v2.notna() & (prob_btts_v2 >= float(DEPLOY_BTTS_NO_VETO_PROB_BTTS_V2_MIN))

    btts_no_veto_strong_yes = pd.Series(False, index=out.index)
    if "signal_btts" in out.columns:
        btts_no_veto_strong_yes = is_btts_no & _is_strong_yes_signal(sig_btts)

    veto_flags.append(btts_no_veto_dead)
    veto_flags.append(btts_no_veto_prob | btts_no_veto_strong_yes)

    # OU2.5 OVER
    is_over25 = market.eq("ou25") & pick.eq("OVER25")
    # Guard: p00_est should be a true 0-0 probability (typically <= ~0.35).
    # If p00_est is mis-scaled, do not blanket-veto OVER25.
    p00_ok = p00.notna() & (p00 >= 0) & (p00 <= 0.35)
    over25_veto_p00 = is_over25 & p00_ok & (p00 >= float(DEPLOY_OVER25_VETO_P00_MIN))
    veto_flags.append(over25_veto_p00)

    # FTR
    is_ftr_side = market.eq("ftr") & pick.isin(["HOME", "AWAY"])
    close_flag = _n(out, "close_match_flag")
    chaos_flag = _n(out, "chaos_risk_flag")
    ftr_margin = _n(out, "ftr_margin")
    abs_pd = _n(out, "power_diff").abs()

    veto_ftr_close = is_ftr_side & (close_flag.fillna(0).astype(float) >= 1.0) & abs_pd.notna() & ftr_margin.notna() & (abs_pd < float(DEPLOY_FTR_CLOSE_PD_ABS_MAX)) & (ftr_margin < float(DEPLOY_FTR_CLOSE_MARGIN_MAX))
    veto_ftr_chaos = is_ftr_side & (chaos_flag.fillna(0).astype(float) >= 1.0) & ftr_margin.notna() & (ftr_margin < float(DEPLOY_FTR_CHAOS_MARGIN_MAX))

    # --- FTR_CONSENSUS brutal eligibility rules ---
    # Detect consensus rows safely (so we don't accidentally veto ALLMARKETS/GLUE rows).
    od_source = _s(out, "od_source").str.lower().str.strip()
    has_cons_cols = any(c in out.columns for c in ("consensus_tier", "consensus_lane", "consensus_margin"))
    is_consensus_src = od_source.str.contains("consensus", regex=False) | has_cons_cols

    # Tier / lane / margin columns (use whatever exists)
    tier_raw = _s(out, "consensus_tier")
    if tier_raw.eq("").all() and "pool_tier" in out.columns:
        tier_raw = _s(out, "pool_tier")
    if tier_raw.eq("").all() and "tier" in out.columns:
        tier_raw = _s(out, "tier")
    tier = tier_raw.str.upper().str.strip()

    lane_raw = _s(out, "consensus_lane")
    if lane_raw.eq("").all() and "lane" in out.columns:
        lane_raw = _s(out, "lane")
    lane = lane_raw.str.upper().str.strip()

    margin_raw = _n(out, "consensus_margin")
    if margin_raw.isna().all():
        margin_raw = _n(out, "consensus_margin_value")
    if margin_raw.isna().all():
        # Fallback: in many exports margin is already `ftr_margin`
        margin_raw = ftr_margin

    # League-specific margin minimums (EPL + Ligue 1 slightly stricter)
    league_txt = _s(out, "league").str.lower().str.strip()
    is_epl_or_l1 = league_txt.str.contains("premier", regex=False) | league_txt.str.contains("epl", regex=False) | league_txt.str.contains("ligue 1", regex=False) | league_txt.str.contains("ligue1", regex=False)
    margin_min = pd.Series(float(DEPLOY_FTR_CONS_MARGIN_MIN_DEFAULT), index=out.index)
    margin_min = margin_min.mask(is_epl_or_l1, float(DEPLOY_FTR_CONS_MARGIN_MIN_EPL_L1))

    # Apply only to FTR side picks and only when row is from consensus source
    cons_scope = is_ftr_side & is_consensus_src

    # 1) Tier eligibility: only enforce when the tier values look like a consensus tier system
    # (avoids vetoing pools like GLUE / BOOKIE_PICK etc)
    tier_known = tier.isin(["ELITE", "MEDIUM", "WEAK", "LOW", "HIGH"])  # extend safely as needed
    veto_cons_tier = cons_scope & tier_known & (~tier.isin(list(DEPLOY_FTR_CONS_TIER_ALLOWED)))

    # 2) Lane eligibility: do not deploy side picks when lane == CLOSE
    veto_cons_lane = cons_scope & lane.ne("") & lane.isin(list(DEPLOY_FTR_CONS_LANE_VETO))

    # 3) Margin eligibility
    veto_cons_margin = cons_scope & margin_raw.notna() & (margin_raw < margin_min)

    # 4) Draw-magnet guard via p00
    veto_cons_p00 = cons_scope & p00.notna() & (p00 >= float(DEPLOY_FTR_CONS_P00_MAX))

    # 5) Coinflip microstructure veto (consensus only)
    ipd = _n(out, "implied_prob_diff")
    bs  = _n(out, "bookie_spread")
    odf = _n(out, "odds_diff")

    veto_cons_micro = cons_scope & (
        (ipd.notna() & (ipd < float(DEPLOY_FTR_CONS_IMPLIED_PROB_DIFF_MIN))) |
        (bs.notna()  & (bs  < float(DEPLOY_FTR_CONS_BOOKIE_SPREAD_MIN))) |
        (odf.notna() & (odf < float(DEPLOY_FTR_CONS_ODDS_DIFF_MIN)))
    )

    veto_cons_true_close = pd.Series(False, index=out.index)
    if bool(DEPLOY_FTR_CONS_TRUE_CLOSE_REASON_VETO) and ("reason_codes" in out.columns):
        rc = _s(out, "reason_codes").str.upper()
        veto_cons_true_close = cons_scope & rc.str.contains("TRUE_CLOSE", regex=False)

    if bool(DEPLOY_FTR_USE_DOMINANT_OVERRIDE):
        dom = is_ftr_side & abs_pd.notna() & ftr_margin.notna() & (abs_pd >= float(DEPLOY_FTR_OVERRIDE_PD_ABS_MIN)) & (ftr_margin >= float(DEPLOY_FTR_OVERRIDE_MARGIN_MIN))
        veto_ftr_close = veto_ftr_close & (~dom)
        veto_ftr_chaos = veto_ftr_chaos & (~dom)

    veto_flags.append(veto_ftr_close)

    # NOTE: do NOT hard-veto chaos-margin FTRs; mark for OBSERVE demotion instead.
    demote_ftr_chaos = veto_ftr_chaos.copy()
    if bool(demote_ftr_chaos.any()):
        out.loc[demote_ftr_chaos, "demote_to_observe"] = 1
        # Append warn reason (pipe-safe)
        prev = out.loc[demote_ftr_chaos, "deterministic_warn_reason"].astype("string").fillna("")
        add = pd.Series("FTR_NO_SIDE_CHAOS_MARGIN", index=prev.index, dtype="string")
        out.loc[demote_ftr_chaos, "deterministic_warn_reason"] = prev.where(prev.eq(""), prev + "|") + add

    veto_flags.append(veto_cons_tier)
    veto_flags.append(veto_cons_lane)
    veto_flags.append(veto_cons_margin)
    veto_flags.append(veto_cons_p00)
    veto_flags.append(veto_cons_micro)
    veto_flags.append(veto_cons_true_close)

    # --- Adjustment rules (DOWNGRADE/BOOST; informational) ---
    adj = pd.Series(0, index=out.index, dtype=int)
    adj_reason_parts: List[pd.Series] = []

    # BTTS YES DOWNGRADE
    btts_yes_downgrade = is_btts_yes & xgsum.notna() & p00.notna() & (xgsum <= float(DEPLOY_BTTS_YES_DOWNGRADE_XGSUM_MAX)) & (p00 >= float(DEPLOY_BTTS_YES_DOWNGRADE_P00_MIN))
    adj = adj.where(~btts_yes_downgrade, -1)
    adj_reason_parts.append(pd.Series(np.where(btts_yes_downgrade, "BTTS_YES_DOWNGRADE_REGIME", ""), index=out.index, dtype="string"))

    # BTTS NO BOOST
    btts_no_boost = is_btts_no & ((fts_max.notna() & (fts_max >= float(DEPLOY_BTTS_NO_BOOST_FTS_MAX_MIN))) | (p00.notna() & (p00 >= float(DEPLOY_BTTS_NO_BOOST_P00_MIN))))
    # If already downgraded, keep downgrade; else allow boost
    adj = adj.mask((adj == 0) & btts_no_boost, 1)
    adj_reason_parts.append(pd.Series(np.where(btts_no_boost, "BTTS_NO_BOOST_BLANKY", ""), index=out.index, dtype="string"))

    # OVER25 DOWNGRADE
    over25_downgrade = is_over25 & cs1.ne("") & cs1.isin(list(DEPLOY_OVER25_DOWNGRADE_CS1_SET)) & cs1_p.notna() & (cs1_p >= float(DEPLOY_OVER25_DOWNGRADE_CS1P_MIN))
    adj = adj.where(~over25_downgrade, -1)
    adj_reason_parts.append(pd.Series(np.where(over25_downgrade, "OVER25_DOWNGRADE_CS1_UNDERMAGNET", ""), index=out.index, dtype="string"))

    # --- Build veto reason strings (ordered, deduped) ---
    # --- Build veto reason strings (ordered, deduped) ---
    veto_reason_parts: List[pd.Series] = []
    veto_reason_parts.append(pd.Series(np.where(signal_mismatch, "SIGNAL_MISMATCH", ""), index=out.index, dtype="string"))

    veto_reason_parts.append(pd.Series(np.where(btts_yes_veto_fts, "BTTS_YES_FTS_TOO_HIGH", ""), index=out.index, dtype="string"))
    veto_reason_parts.append(pd.Series(np.where(btts_yes_veto_cs1, "BTTS_YES_CS1_TRAP", ""), index=out.index, dtype="string"))

    veto_reason_parts.append(pd.Series(np.where(btts_no_veto_dead, "BTTS_NO_HIGH_GOALS_LOW_P00", ""), index=out.index, dtype="string"))
    veto_reason_parts.append(pd.Series(np.where(btts_no_veto_prob, "BTTS_NO_PROB_BTTS_TOO_HIGH", ""), index=out.index, dtype="string"))
    veto_reason_parts.append(pd.Series(np.where(btts_no_veto_strong_yes, "BTTS_NO_SIGNAL_STRONG_YES", ""), index=out.index, dtype="string"))

    veto_reason_parts.append(pd.Series(np.where(over25_veto_p00, "OVER25_P00_TOO_HIGH", ""), index=out.index, dtype="string"))

    veto_reason_parts.append(pd.Series(np.where(veto_ftr_close, "FTR_NO_SIDE_TRUE_CLOSE", ""), index=out.index, dtype="string"))

    # --- Build warning reason strings (non-veto; used for OBSERVE demotions) ---
    warn_reason_parts: List[pd.Series] = []
    warn_reason_parts.append(pd.Series(np.where(demote_ftr_chaos, "FTR_NO_SIDE_CHAOS_MARGIN", ""), index=out.index, dtype="string"))

    veto_reason_parts.append(pd.Series(np.where(veto_cons_tier, "FTR_CONS_TIER_BLOCK", ""), index=out.index, dtype="string"))
    veto_reason_parts.append(pd.Series(np.where(veto_cons_lane, "FTR_CONS_LANE_CLOSE", ""), index=out.index, dtype="string"))
    veto_reason_parts.append(pd.Series(np.where(veto_cons_margin, "FTR_CONS_MARGIN_TOO_LOW", ""), index=out.index, dtype="string"))
    veto_reason_parts.append(pd.Series(np.where(veto_cons_p00, "FTR_CONS_DRAW_MAGNET_P00", ""), index=out.index, dtype="string"))
    veto_reason_parts.append(pd.Series(np.where(veto_cons_micro, "FTR_CONS_COINFLIP_MICRO", ""), index=out.index, dtype="string"))
    veto_reason_parts.append(pd.Series(np.where(veto_cons_true_close, "FTR_CONS_TRUE_CLOSE", ""), index=out.index, dtype="string"))

    def _combine_tokens(parts: List[pd.Series]) -> pd.Series:
        def _combine_row(i: int) -> str:
            toks: List[str] = []
            for s in parts:
                v = str(s.iat[i])
                if v and v != "nan":
                    toks.append(v)
            if not toks:
                return ""
            seen = set()
            out_t: List[str] = []
            for t in toks:
                if t in seen:
                    continue
                seen.add(t)
                out_t.append(t)
            return "|".join(out_t)

        return pd.Series([_combine_row(i) for i in range(len(out))], index=out.index, dtype="string")

    veto_reason = _combine_tokens(veto_reason_parts)
    adj_reason = _combine_tokens(adj_reason_parts)
    warn_reason = _combine_tokens(warn_reason_parts) if "warn_reason_parts" in locals() else pd.Series("", index=out.index, dtype="string")

    # Overall pass/fail
    deterministic_pass = veto_reason.eq("")

    out["signal_mismatch_flag"] = signal_mismatch.astype(int)
    out["deterministic_veto_reason"] = veto_reason
    out["deterministic_warn_reason"] = warn_reason
    out["deterministic_pass"] = deterministic_pass.astype(int)
    out["deterministic_adjustment"] = adj.astype(int)
    out["deterministic_adjust_reason"] = adj_reason

    return out


if __name__ == "__main__":
    main()
