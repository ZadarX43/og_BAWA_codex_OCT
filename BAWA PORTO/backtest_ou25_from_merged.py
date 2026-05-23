#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import unicodedata
import json
import argparse

PRED_PATH = Path("predictions_output/dec2025_ou25_backtest_fix/BOOKIE_IMP68_ALLMARKETS_2025-12-01_to_2025-12-31.csv")
MERGED_DIR = Path("Matches/__merged__")

# Optional: filter predictions using an OVER-only policy derived from OU25 walkforward audits.
# If enabled, we only score the subset of OU25 rows that match the policy's deployable (league, branch)
# and we also require bookie_pick == OVER25.
POLICY_PATH = Path("over25_from_ou25_policy.json")
APPLY_OVER25_POLICY = False  # default Pass A (raw engine). Use --apply-policy for Pass B.

DATE_FROM = "2024-11-01"
DATE_TO   = "2025-04-30"
# Default knobs (can be overridden by CLI)
OVER_ONLY_DEFAULT = True       # apples-to-apples with OVER-only audits
BAND1_MIN_P_DEFAULT = 0.60     # band1 gate threshold for Pass B

# ----------------------------
# Deploy-v0 defaults
DEPLOY_BRANCH_DEFAULT = "ou25_combined_topq_080"
DEPLOY_SIGNALS_DEFAULT = {"STRONG_OVER", "VERY_STRONG_OVER"}
DEPLOY_TOPQ_Q_DEFAULT = 0.80
DEPLOY_SIGNAL_RECOMPUTE_DEFAULT = "auto"  # auto|always|never
DEPLOY_SIGNAL_STRONG_P_DEFAULT = 0.78
DEPLOY_SIGNAL_VSTRONG_P_DEFAULT = 0.85
# Quantile banding defaults for deploy-v0
DEPLOY_SIGNAL_BANDING_DEFAULT = "fixed"  # fixed|quantile
DEPLOY_SIGNAL_STRONG_Q_DEFAULT = 0.80     # per-league strong quantile when banding=quantile
DEPLOY_SIGNAL_VSTRONG_Q_DEFAULT = 0.90    # per-league very-strong quantile when banding=quantile

# Deploy profile presets
DEPLOY_PROFILE_DEFAULT = "premium"  # premium|elite

# Elite premium hard whitelist (only these leagues are kept when --deploy-profile elite)
ELITE_KEEP_LEAGUES_DEFAULT: set[str] = {
    "Germany Bundesliga",
    "Netherlands Eredivisie",
    "Italy Serie A",
    "Portugal Liga",
    "Champions League",
    "USA MLS",
    "France Ligue 1",
    "Belgium Pro",
}
# ----------------------------
# Quantile signal recompute helper
# ----------------------------
def _recompute_signal_over25_quantile_by_league(
    df: pd.DataFrame,
    league_col: str = "league",
    p_col: str = "model_p_for_bookie",
    strong_q: float = DEPLOY_SIGNAL_STRONG_Q_DEFAULT,
    vstrong_q: float = DEPLOY_SIGNAL_VSTRONG_Q_DEFAULT,
    per_league_q: dict[str, tuple[float, float]] | None = None,
) -> tuple[pd.Series, dict[str, tuple[float, float]]]:
    """Recompute signal_over25 using per-league *rank percentiles*.

    Why rank-percentiles (instead of raw quantiles):
      - Some leagues have heavy probability ties/rounding which can make
        quantile thresholds identical (e.g., strong_thr == vstrong_thr).
      - Rank-percentiles remain well-defined under ties.

    Rules (stationary by league):
      - VERY_STRONG_OVER if rank_pct(p) >= vstrong_q
      - STRONG_OVER if rank_pct(p) >= strong_q and < vstrong_q
      - NEUTRAL otherwise

    `per_league_q` can override (strong_q, vstrong_q) for specific leagues.
    Example: {"England Championship": (0.85, 0.93)}

    Returns (signal_series, thresholds_by_league) where thresholds_by_league maps
    league -> (approx_strong_thr, approx_vstrong_thr) for logging/debug.
    """
    if df is None or df.empty:
        return pd.Series([], dtype="string"), {}

    lg = df.get(league_col, "").astype("string").fillna("").str.strip()
    p = pd.to_numeric(df.get(p_col, np.nan), errors="coerce")

    # Work only on rows with usable league + p
    tmp = pd.DataFrame({"league": lg, "p": p})
    tmp = tmp[tmp["league"].ne("") & tmp["p"].notna()].copy()

    thr_map: dict[str, tuple[float, float]] = {}
    sig = pd.Series("NEUTRAL", index=df.index, dtype="string")

    if tmp.empty:
        return sig.astype("string"), thr_map

    # Normalize override map
    q_over: dict[str, tuple[float, float]] = {}
    if isinstance(per_league_q, dict):
        for k, v in per_league_q.items():
            try:
                lk = str(k).strip()
                if not lk:
                    continue
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    q_over[lk] = (float(v[0]), float(v[1]))
            except Exception:
                continue

    # Compute signals league-by-league using rank percentiles (tie-safe)
    for lg_name, df_lg in tmp.groupby("league", dropna=False):
        lg_key = str(lg_name).strip()
        if not lg_key:
            continue

        sq, vq = (float(strong_q), float(vstrong_q))
        if lg_key in q_over:
            try:
                sq, vq = q_over[lg_key]
            except Exception:
                sq, vq = (float(strong_q), float(vstrong_q))

        # guard: ensure vq >= sq
        if vq < sq:
            vq = sq

        p_ser = pd.to_numeric(df_lg["p"], errors="coerce")
        if p_ser.notna().sum() == 0:
            continue

        rp = p_ser.rank(pct=True, method="average")
        idx = df_lg.index

        m_vs = rp.ge(vq).fillna(False)
        m_s = rp.ge(sq).fillna(False) & rp.lt(vq).fillna(False)

        # Apply onto the original df index (tmp index matches original df index)
        sig.loc[idx[m_vs]] = "VERY_STRONG_OVER"
        sig.loc[idx[m_s]] = "STRONG_OVER"

        # Approx numeric thresholds for logs only
        try:
            s_thr = float(p_ser.quantile(sq))
            v_thr = float(p_ser.quantile(vq))
            if np.isfinite(s_thr) and np.isfinite(v_thr) and (v_thr < s_thr):
                v_thr = s_thr
            thr_map[lg_key] = (s_thr, v_thr)
        except Exception:
            pass

    return sig.astype("string"), thr_map

# Per-league overrides for deploy-v0 (only applied when --deploy-v0 is used)
# Tune these without touching the rest of the policy code.
LEAGUE_OVERRIDES_DEFAULT: dict[str, dict[str, object]] = {
    # Conference is noisy: park it for now
    "Europa Conference": {"exclude": True},
    # Brazil Serie A has truth/quality volatility in this slice: park it for now
    "Brazil Serie A": {"exclude": True},
    # Japan J1 is currently too thin/noisy for premium deploy: park it for now
    "Japan J1": {"exclude": True},

    # Known-bad / structurally weak for OU25 premium lane in this validation window
    "England Premier League": {"exclude": True},
    "Spain La Liga": {"exclude": True},
    "England EFL League 1": {"exclude": True},
    # Soft-avoid FA Cup as excluded by default for premium lane
    "England FA Cup": {"exclude": True},

    # Belgium Pro underperforms in strict pass: tighten slightly by default
    "Belgium Pro": {"strong_q": 0.85, "vstrong_q": 0.93, "signals": {"VERY_STRONG_OVER"}},

    # Championship: tighten quantile banding for premium (improves deployment reliability)
    "England Championship": {"strong_q": 0.85, "vstrong_q": 0.93},
}

# Quantile points for band1 autopsy reporting
BAND1_AUTOPSY_QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

# ----------------------------
# Date parsing helpers
# ----------------------------
def _first_present_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first candidate column name that exists in df, else ''."""
    for c in candidates:
        if c in df.columns:
            return c
    return ""

def pick_best_date_col(df: pd.DataFrame) -> str:
    """Pick the best available date-like column.

    Priority is NOT hard-coded.
    We avoid selecting a column that exists but is mostly empty/NaN (common in some __merged__ files).

    Heuristic:
      - Prefer `match_date` only if it is reasonably populated.
      - Otherwise fall back to `date_GMT`, then `date/Date`, then `timestamp`.

    NOTE: We deliberately keep this lightweight (no full parsing here) to avoid slowdown.
    """
    cols = [str(c).strip() for c in df.columns]

    # If match_date exists but is largely empty, do not pick it.
    if "match_date" in cols:
        try:
            s = df["match_date"].astype("string").str.strip()
            s = s.mask(s.eq(""), pd.NA)
            non_null = int(s.notna().sum())
            total = int(len(s)) if len(s) else 0
            frac = (non_null / total) if total else 0.0
            # Require at least 20% populated (or at least 500 rows) to be considered reliable.
            if (non_null >= 500) or (frac >= 0.20):
                return "match_date"
        except Exception:
            # If anything goes wrong, do not trust match_date blindly.
            pass

    for c in ("date_GMT", "date", "Date", "timestamp"):
        if c in cols:
            return c

    return ""

def parse_window_dates(s: pd.Series, date_col: str) -> pd.Series:
    """Parse a date-like series into UTC timestamps (best effort)."""
    if s is None:
        return pd.Series(pd.NaT, index=pd.RangeIndex(0))
    if date_col == "timestamp":
        ts = pd.to_numeric(s, errors="coerce")
        mx = float(ts.max()) if ts.notna().any() else float("nan")
        if np.isfinite(mx):
            # seconds ~ 1e9, millis ~ 1e12
            unit = "ms" if mx > 1.0e11 else "s"
            return pd.to_datetime(ts, errors="coerce", utc=True, unit=unit)
        return pd.to_datetime(s, errors="coerce", utc=True)

    # string-ish dates (match_date/date_GMT/etc.)
    # Some merged files contain formats like "Dec 6 2025 - 7:45pm".
    try:
        return pd.to_datetime(s, errors="coerce", utc=True, format="mixed", cache=True)
    except TypeError:
        # Older pandas: no format="mixed"
        return pd.to_datetime(s, errors="coerce", utc=True, cache=True)
    except Exception:
        # Last resort: stringify (handles mixed object types)
        return pd.to_datetime(s.astype(str), errors="coerce", utc=True)

def filter_window(df: pd.DataFrame, date_from: str, date_to: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df.iloc[0:0].copy()

    date_col = pick_best_date_col(df)
    if not date_col:
        return df.iloc[0:0].copy()

    md = parse_window_dates(df[date_col], date_col)
    lo = pd.Timestamp(date_from, tz="UTC")
    hi_ex = pd.Timestamp(date_to, tz="UTC") + pd.Timedelta(days=1)
    m = md.notna() & (md >= lo) & (md < hi_ex)

    out = df.loc[m].copy()
    out["match_date"] = md.loc[m].dt.strftime("%Y-%m-%d")
    return out


# ----------------------------
# Prediction window filter helpers
# ----------------------------

def _parse_pred_match_date(pred: pd.DataFrame) -> pd.Series:
    """Best-effort match_date for predictions.

    Priority:
      1) existing `match_date` column (already YYYY-MM-DD in many exports)
      2) derive from fixture_key prefix like `YYYY_MM_DD_...`

    Returns a UTC Timestamp series (NaT where unknown).
    """
    if pred is None or pred.empty:
        return pd.Series(pd.NaT, index=pd.RangeIndex(0))

    if "match_date" in pred.columns:
        try:
            s = pred["match_date"].astype("string").fillna("").str.strip()
            s = s.mask(s.eq(""), pd.NA)
            md = pd.to_datetime(s, errors="coerce", utc=True)
            if bool(md.notna().any()):
                return md
        except Exception:
            pass

    # Try parse from fixture_key: YYYY_MM_DD_...
    if "fixture_key" in pred.columns:
        try:
            fk = pred["fixture_key"].astype("string").fillna("").str.strip()
            # take first 10 chars: YYYY_MM_DD
            head = fk.str.slice(0, 10)
            head = head.str.replace("_", "-", regex=False)
            md = pd.to_datetime(head, errors="coerce", utc=True)
            return md
        except Exception:
            return pd.Series(pd.NaT, index=pred.index)

    return pd.Series(pd.NaT, index=pred.index)


def _filter_pred_window(pred: pd.DataFrame, date_from: str, date_to: str) -> pd.DataFrame:
    """Filter predictions to the same window as truth (using best-effort match_date)."""
    if pred is None or pred.empty:
        return pred.iloc[0:0].copy()

    md = _parse_pred_match_date(pred)
    lo = pd.Timestamp(str(date_from).strip(), tz="UTC")
    hi_ex = pd.Timestamp(str(date_to).strip(), tz="UTC") + pd.Timedelta(days=1)

    m = md.notna() & (md >= lo) & (md < hi_ex)
    out = pred.loc[m].copy()
    # persist a normalized match_date string for downstream debug/fallback joins
    out["match_date"] = md.loc[m].dt.strftime("%Y-%m-%d")
    return out


# ----------------------------
# Fixture key helpers
# ----------------------------

def _ascii_fold(x: object) -> str:
    """Best-effort ASCII fold (remove diacritics) for stable keys."""
    try:
        s = str(x or "").strip()
    except Exception:
        return ""
    if not s:
        return ""
    # Normalize then strip combining marks
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s


def _fixture_key_ascii(x: object) -> str:
    """Normalize a fixture_key for join safety.

    - ASCII-fold (remove diacritics)
    - lowercase
    - strip

    We intentionally do NOT rebuild tokens here; we only normalize so keys
    built by different code paths still join (e.g. rebuilt keys are lowercase).
    """
    s = _ascii_fold(x)
    try:
        return str(s or "").strip().lower()
    except Exception:
        return ""



def _norm_team_token(x: object) -> str:
    """Lowercase + underscore token for team names (lightweight, deterministic)."""
    try:
        s = _ascii_fold(x).strip().lower()
    except Exception:
        s = ""
    # keep alnum, convert everything else to underscore
    out = []
    prev_us = False
    for ch in s:
        if ch.isalnum():
            out.append(ch)
            prev_us = False
        else:
            if not prev_us:
                out.append("_")
                prev_us = True
    tok = "".join(out).strip("_")
    # collapse any double underscores
    while "__" in tok:
        tok = tok.replace("__", "_")
    return tok


# More forgiving team token normalizer for join fallback.
def _norm_team_token_fuzzy(x: object) -> str:
    """More forgiving team token normalizer for join fallback.

    - ASCII-fold
    - lowercase
    - remove common punctuation
    - collapse whitespace/underscores

    Intended only for fallback joins when fixture_key drift exists.
    """
    s = _ascii_fold(x)
    try:
        s = str(s or "").strip().lower()
    except Exception:
        s = ""
    if not s:
        return ""

    # Remove dots and apostrophes, normalize hyphens to spaces
    s = s.replace(".", " ").replace("'", " ").replace("’", " ").replace("-", " ")

    # Keep alnum, convert everything else to underscore
    out = []
    prev_us = False
    for ch in s:
        if ch.isalnum():
            out.append(ch)
            prev_us = False
        else:
            if not prev_us:
                out.append("_")
                prev_us = True
    tok = "".join(out).strip("_")
    while "__" in tok:
        tok = tok.replace("__", "_")

    # A few common abbreviations / aliases
    # (kept intentionally minimal and safe)
    tok = tok.replace("atletico_mineiro", "atletico_mg")
    tok = tok.replace("atletico_mg", "atletico_mg")

    # Drop common club suffix tokens to reduce drift between feeds
    # e.g. "reading_fc" vs "reading", "fk_bodo_glimt" vs "bodo_glimt"
    for suf in ("_fc", "_cf", "_sc", "_afc", "_fk"):
        if tok.endswith(suf):
            tok = tok[: -len(suf)]

    # Also remove standalone suffix tokens inside the name
    for mid in ("_fc_", "_cf_", "_sc_", "_afc_", "_fk_"):
        tok = tok.replace(mid, "_")

    tok = tok.strip("_")
    while "__" in tok:
        tok = tok.replace("__", "_")

    return tok


# Extra-fuzzy token for stubborn join drift (EFL/UK team suffixes etc.)
# Only intended for fallback joins; DO NOT use for primary keys.
def _norm_team_token_extra_fuzzy(x: object) -> str:
    """Extra-fuzzy token normalizer.

    Builds on _norm_team_token_fuzzy and then removes common club suffix words
    that often differ across feeds (e.g., "city", "town", "county", "rovers").

    Intended ONLY for fallback join recovery.
    """
    tok = _norm_team_token_fuzzy(x)
    if not tok:
        return ""

    parts = [p for p in tok.split("_") if p]
    # Common suffixes/stopwords seen in English/British club naming
    stop = {
        "fc", "afc", "cf", "sc", "fk",
        "the",
        "city", "town", "county",
        "united", "utd",
        "rovers", "rover",
        "athletic", "ath",
        "wanderers", "wand",
        "hotspur", "spurs",
        "albion",
        "forest",
        "rangers",
        "wednesday",
        "saturday",
        "saints",
    }

    parts2 = [p for p in parts if p not in stop]
    if not parts2:
        # If we stripped everything, fall back to the original fuzzy token
        return tok

    out = "_".join(parts2).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out


def build_fixture_key_from_row(match_date_str: object, home: object, away: object) -> str:
    """Build OG-style fixture key: YYYY_MM_DD_HOME_AWAY (best-effort)."""
    # match_date_str is already standardized to YYYY-MM-DD in windowed frames
    md = pd.to_datetime(match_date_str, errors="coerce", utc=True)
    ds = md.strftime("%Y_%m_%d") if pd.notna(md) else ""
    h = _norm_team_token(home)
    a = _norm_team_token(away)
    key = f"{ds}_{h}_{a}".strip("_")
    return key


# ----------------------------
# Goal/OU25 truth helpers
# ----------------------------
def coalesce_numeric(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    """Return first candidate numeric series that has any non-null values."""
    for c in candidates:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                return s
    return pd.Series(np.nan, index=df.index)

def build_total_goals(df: pd.DataFrame) -> pd.Series:
    """
    Best-effort total goals:
      1) total_goal_count / total_goals / FT_total_goals
      2) home_team_goal_count + away_team_goal_count (or aliases)
    """
    # direct total columns
    total = coalesce_numeric(df, [
        "total_goal_count",
        "total_goals",
        "ft_total_goals",
        "FT_total_goals",
        "goals_total",
    ])
    if total.notna().any():
        return total

    # home/away goals
    hg = coalesce_numeric(df, [
        "home_team_goal_count",
        "home_goals",
        "HG",
        "FTHG",
        "home_score",
    ])
    ag = coalesce_numeric(df, [
        "away_team_goal_count",
        "away_goals",
        "AG",
        "FTAG",
        "away_score",
    ])
    tot = hg + ag
    return tot


# ----------------------------
# Completion mask helper
# ----------------------------
def _is_complete_mask(df: pd.DataFrame) -> pd.Series:
    """Best-effort mask for whether a match is completed in merged truth.

    Many merged files include future fixtures where goal counts are 0/0 and
    `status` is not complete. Those must NOT be scored as 0 goals.

    Rules:
      - If a status-like column exists (even with whitespace/case drift), require it to indicate completion.
      - If no status exists, infer completion only when we can observe a non-null goals signal (total or home/away)
        OR a result/score-like field that contains a plausible FT score.
    """
    if df is None or df.empty:
        return pd.Series(False, index=pd.RangeIndex(0))

    # Build a normalized column map: stripped+lower -> original
    try:
        col_map = {str(c).strip().lower(): c for c in df.columns}
    except Exception:
        col_map = {}

    # 1) status-based (preferred)
    status_key = None
    for k in ("status", "match_status", "fixture_status"):
        if k in col_map:
            status_key = col_map[k]
            break

    if status_key is not None:
        s = df[status_key].astype("string").fillna("").str.strip().str.lower()
        # common completed markers
        complete = s.isin([
            "complete", "completed", "finished", "ft", "full-time", "full time",
            "played", "final", "ended",
        ])
        # also accept anything containing these tokens
        complete = complete | s.str.contains(r"\b(?:complete|completed|finished|full|ft)\b", regex=True)
        return complete.fillna(False)

    # 1b) score/result-based fallback (some feeds omit status but include a result like "2-1")
    for k in ("result", "ft", "ft_score", "full_time", "full_time_score"):
        if k in col_map:
            rr = df[col_map[k]].astype("string").fillna("").str.strip()
            # Accept patterns like 0-0, 1-2, 10-9
            m_score = rr.str.match(r"^\d{1,2}\s*-\s*\d{1,2}$", na=False)
            if bool(m_score.any()):
                return m_score.fillna(False)

    # 2) no status column: infer completion from goals columns
    tot = coalesce_numeric(df, [
        "total_goal_count", "total_goals", "ft_total_goals", "FT_total_goals", "goals_total",
    ])
    hg = coalesce_numeric(df, [
        "home_team_goal_count", "home_goals", "HG", "FTHG", "home_score",
    ])
    ag = coalesce_numeric(df, [
        "away_team_goal_count", "away_goals", "AG", "FTAG", "away_score",
    ])

    # completed if total is present OR either side goals are present
    return (tot.notna() | hg.notna() | ag.notna()).fillna(False)


# ----------------------------
# Policy helper
# ----------------------------

def _load_policy_json(path: Path) -> tuple[dict, str]:
    """Load policy JSON dict (for metadata + allowlist extraction)."""
    if (path is None) or (not Path(path).exists()):
        return {}, "missing"
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data, "ok"
        return {}, "not_a_dict"
    except Exception:
        return {}, "unreadable"


def _load_over25_policy_allowlist(path: Path) -> tuple[set[tuple[str, str]], str]:
    """Return allowlist of (league, branch) pairs for OVER-only scoring.

    We prefer policy['deployable_league_rows'] when present.
    Fallback: allow all branches in policy['branch_leaderboard'].
    Returns (allow_pairs, mode) where mode is a short string for logging.
    """
    data, st = _load_policy_json(path)
    if st != "ok":
        return set(), st

    allow: set[tuple[str, str]] = set()

    dlr = data.get("deployable_league_rows")
    if isinstance(dlr, list) and dlr:
        for r in dlr:
            try:
                lg = str(r.get("league", "")).strip()
                br = str(r.get("branch", "")).strip()
                if lg and br:
                    allow.add((lg, br))
            except Exception:
                continue
        return allow, "deployable_league_rows"

    blb = data.get("branch_leaderboard")
    if isinstance(blb, list) and blb:
        for r in blb:
            try:
                br = str(r.get("branch", "")).strip()
                if br:
                    allow.add(("*", br))
            except Exception:
                continue
        return allow, "branch_leaderboard"

    return set(), "empty"


# ----------------------------
# Policy metadata print & band1 autopsy
# ----------------------------

def _print_policy_metadata(policy_path: Path) -> None:
    """Print policy provenance + allowlist sizes so the wrong JSON can't slip in silently."""
    data, st = _load_policy_json(policy_path)
    if st != "ok":
        print(f"[POLICY_META] policy={policy_path} status={st}")
        return

    src = str(data.get("source", "")).strip()
    run_root = str(data.get("run_root", "")).strip()

    blb = data.get("branch_leaderboard")
    dlr = data.get("deployable_league_rows")
    blb_n = len(blb) if isinstance(blb, list) else 0
    dlr_n = len(dlr) if isinstance(dlr, list) else 0

    allow_pairs, mode = _load_over25_policy_allowlist(policy_path)
    allow_n = len(allow_pairs)

    print(
        "[POLICY_META] "
        f"policy={policy_path} | source={src or 'n/a'} | run_root={run_root or 'n/a'} | "
        f"branch_leaderboard_n={blb_n} | deployable_league_rows_n={dlr_n} | allowlist_mode={mode} | allowlist_n={allow_n}"
    )


def _band1_autopsy(scored: pd.DataFrame) -> None:
    """Band1 autopsy: per-league performance + p-quantiles + side distribution."""
    if scored is None or scored.empty:
        print("[BAND1_AUTOPSY] no scored rows")
        return
    if "ou25_policy_branch" not in scored.columns:
        print("[BAND1_AUTOPSY] missing ou25_policy_branch")
        return

    b1 = scored[scored["ou25_policy_branch"].astype(str).str.strip().eq("ou25_band1_124_176")].copy()
    if b1.empty:
        print("[BAND1_AUTOPSY] no band1 rows")
        return

    # A) performance by league
    try:
        print("\n=== BAND1 AUTOPSY: accuracy by league ===")
        g = b1.groupby("league")["correct"].agg(["count", "mean"]).sort_values("count", ascending=False)
        g["acc_pct"] = (g["mean"] * 100).round(2)
        print(g[["count", "acc_pct"]].to_string())
    except Exception:
        print("(failed band1 league accuracy)")

    # B) p-quantiles by league
    if "model_p_for_bookie" in b1.columns:
        try:
            print("\n=== BAND1 AUTOPSY: model_p_for_bookie quantiles by league ===")
            p = pd.to_numeric(b1["model_p_for_bookie"], errors="coerce")
            b1 = b1.assign(__p=p)
            rows = []
            for lg, df_lg in b1.groupby("league"):
                s = pd.to_numeric(df_lg["__p"], errors="coerce").dropna()
                if s.empty:
                    continue
                qv = s.quantile(BAND1_AUTOPSY_QUANTILES).to_dict()
                out = {"league": lg, "n": int(len(df_lg))}
                for q in BAND1_AUTOPSY_QUANTILES:
                    out[f"q{int(q*100):02d}"] = float(qv.get(q, np.nan))
                rows.append(out)
            if rows:
                qdf = pd.DataFrame(rows).sort_values("n", ascending=False)
                print(qdf.to_string(index=False))
            else:
                print("(no quantile rows)")
        except Exception:
            print("(failed band1 p-quantiles)")
    else:
        print("\n[BAND1_AUTOPSY] model_p_for_bookie not present; skipping quantiles")

    # C) side distribution within band1
    try:
        print("\n=== BAND1 AUTOPSY: side distribution (bookie_pick) ===")
        gb = b1.groupby("bookie_pick")["correct"].agg(["count", "mean"]).sort_values("count", ascending=False)
        gb["acc_pct"] = (gb["mean"] * 100).round(2)
        print(gb[["count", "acc_pct"]].to_string())
    except Exception:
        print("(failed band1 side distribution)")


# ----------------------------
# Signal recompute helpers
# ----------------------------

def _recompute_signal_over25_from_p(
    df: pd.DataFrame,
    p_col: str = "model_p_for_bookie",
    strong_p: float = DEPLOY_SIGNAL_STRONG_P_DEFAULT,
    vstrong_p: float = DEPLOY_SIGNAL_VSTRONG_P_DEFAULT,
) -> pd.Series:
    """Recompute signal_over25 from a probability column.

    Rules (over-only oriented):
      - VERY_STRONG_OVER if p >= vstrong_p
      - STRONG_OVER if strong_p <= p < vstrong_p
      - NEUTRAL otherwise (including NaNs)

    Returns an uppercase string Series.
    """
    if df is None or df.empty:
        return pd.Series([], dtype="string")

    p = pd.to_numeric(df.get(p_col, np.nan), errors="coerce")
    sig = pd.Series("NEUTRAL", index=df.index, dtype="string")
    sig = sig.mask(p.ge(vstrong_p).fillna(False), "VERY_STRONG_OVER")
    sig = sig.mask(p.ge(strong_p).fillna(False) & p.lt(vstrong_p).fillna(False), "STRONG_OVER")
    return sig.astype("string")


def _should_recompute_signal(df: pd.DataFrame, mode: str) -> bool:
    """Decide whether to recompute signal_over25.

    mode:
      - always: always recompute
      - never: never recompute
      - auto: recompute if column missing OR >=90% of non-null values are NEUTRAL
    """
    m = str(mode or "auto").strip().lower()
    if m == "always":
        return True
    if m == "never":
        return False

    # auto
    if df is None or df.empty:
        return False
    if "signal_over25" not in df.columns:
        return True

    s = df["signal_over25"].astype("string").fillna("").str.strip().str.upper()
    s = s[s.ne("")]
    if s.empty:
        return True

    neutral_share = float((s.eq("NEUTRAL")).mean())
    return neutral_share >= 0.90


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest OU25 from __merged__ truth")
    # Guardrail: print which script file is executing (helps avoid editing one copy but running another)
    try:
        print(f"[SCRIPT] running: {__file__}")
    except Exception:
        pass
    parser.add_argument("--pred-path", default=str(PRED_PATH), help="Path to BOOKIE_IMP*_ALLMARKETS predictions CSV")
    parser.add_argument("--merged-dir", default=str(MERGED_DIR), help="Directory containing __merged__ league CSVs")
    parser.add_argument("--date-from", default=DATE_FROM, help="Window start (YYYY-MM-DD)")
    parser.add_argument("--date-to", default=DATE_TO, help="Window end (YYYY-MM-DD)")

    # Pass selection
    parser.add_argument("--apply-policy", action="store_true", help="Pass B: apply OVER-only deploy policy filter")
    parser.add_argument("--no-policy", action="store_true", help="Force Pass A: do not apply policy filter")

    # Apples-to-apples with OVER-only audit
    parser.add_argument("--over-only", dest="over_only", action="store_true", default=OVER_ONLY_DEFAULT,
                        help="Require bookie_pick == OVER25 even when policy is off (default: on)")
    parser.add_argument("--all-sides", dest="over_only", action="store_false",
                        help="Do not force OVER25-only (score OVER+UNDER)")

    # Band1 gate tuning (only matters when policy is on)
    parser.add_argument("--band1-min-p", type=float, default=BAND1_MIN_P_DEFAULT,
                        help="Band1 gate: require model_p_for_bookie >= this value (default: 0.60)")

    # Optional: strict deploy simulation (drop NEUTRAL/UNDER signals)
    parser.add_argument(
        "--strict-over-signals",
        action="store_true",
        help="Keep only signal_over25 in {STRONG_OVER, VERY_STRONG_OVER} after policy/gates (deploy-style)"
    )

    parser.add_argument(
        "--deploy-profile",
        default=DEPLOY_PROFILE_DEFAULT,
        choices=["premium", "elite"],
        help="Deploy profile preset: premium=default gating; elite=hard whitelist of top leagues",
    )
    # Tip: for stationary signals by league, use:
    #   --deploy-v0 --deploy-signal-recompute always --deploy-signal-banding quantile --deploy-strong-q 0.80 --deploy-vstrong-q 0.90
    # Deploy-v0: enforce topq + strong signals, with per-league overrides
    parser.add_argument(
        "--deploy-v0",
        action="store_true",
        help="Apply deploy-v0 filter: topq branch + strong over signals + optional per-league overrides"
    )
    parser.add_argument(
        "--deploy-branch",
        default=DEPLOY_BRANCH_DEFAULT,
        help=f"Deploy-v0 branch (default: {DEPLOY_BRANCH_DEFAULT})"
    )
    parser.add_argument(
        "--deploy-min-p",
        type=float,
        default=0.0,
        help="Deploy-v0 global minimum model_p_for_bookie (default: 0.0)"
    )
    parser.add_argument(
        "--deploy-topq-q",
        type=float,
        default=DEPLOY_TOPQ_Q_DEFAULT,
        help=f"Deploy-v0 TopQ quantile for model_p_for_bookie when ou25_policy_branch is missing (default: {DEPLOY_TOPQ_Q_DEFAULT})"
    )
    parser.add_argument(
        "--deploy-override-json",
        default="",
        help="Optional path to JSON dict of per-league overrides (min_p, signals). If omitted, uses built-in defaults"
    )
    parser.add_argument(
        "--deploy-signal-mode",
        default="strict",
        choices=["strict", "relaxed", "none"],
        help=(
            "Deploy-v0 signal gating mode: "
            "strict=STRONG/VERY_STRONG only; "
            "relaxed=STRONG/VERY_STRONG OR (NEUTRAL with p>=TopQ threshold); "
            "none=do not filter by signal_over25"
        ),
    )
    parser.add_argument(
        "--deploy-print-signal-dist",
        action="store_true",
        help="When --deploy-v0 is active, print signal_over25 distribution before/after deploy-v0 filtering",
    )
    parser.add_argument(
        "--deploy-signal-recompute",
        default=DEPLOY_SIGNAL_RECOMPUTE_DEFAULT,
        choices=["auto", "always", "never"],
        help=(
            "Recompute signal_over25 from model_p_for_bookie for deploy-v0. "
            "auto=recompute if missing or mostly NEUTRAL; always=force recompute; never=use existing column."
        ),
    )
    parser.add_argument(
        "--deploy-signal-banding",
        default=DEPLOY_SIGNAL_BANDING_DEFAULT,
        choices=["fixed", "quantile"],
        help=(
            "Deploy-v0 signal banding mode: fixed=global p thresholds (strong/vstrong); "
            "quantile=per-league quantile thresholds (stationary by league)."
        ),
    )
    parser.add_argument(
        "--deploy-strong-p",
        type=float,
        default=DEPLOY_SIGNAL_STRONG_P_DEFAULT,
        help=f"Signal recompute: STRONG_OVER if p >= this (default: {DEPLOY_SIGNAL_STRONG_P_DEFAULT})",
    )
    parser.add_argument(
        "--deploy-vstrong-p",
        type=float,
        default=DEPLOY_SIGNAL_VSTRONG_P_DEFAULT,
        help=f"Signal recompute: VERY_STRONG_OVER if p >= this (default: {DEPLOY_SIGNAL_VSTRONG_P_DEFAULT})",
    )
    parser.add_argument(
        "--deploy-strong-q",
        type=float,
        default=DEPLOY_SIGNAL_STRONG_Q_DEFAULT,
        help=f"Quantile banding: STRONG_OVER if p >= league quantile q (default: {DEPLOY_SIGNAL_STRONG_Q_DEFAULT})",
    )
    parser.add_argument(
        "--deploy-vstrong-q",
        type=float,
        default=DEPLOY_SIGNAL_VSTRONG_Q_DEFAULT,
        help=f"Quantile banding: VERY_STRONG_OVER if p >= league quantile q (default: {DEPLOY_SIGNAL_VSTRONG_Q_DEFAULT})",
    )

    # Optional: drop leagues with poor truth coverage in-window (keeps KPIs honest)
    parser.add_argument(
        "--min-league-coverage",
        type=float,
        default=0.0,
        help="Minimum per-league scored/pred coverage (0..1). If >0, leagues below this are excluded from scoring (default: 0.0)"
    )

    # Optional: extra side diagnostics (useful when running --all-sides)
    parser.add_argument(
        "--side-diagnostics",
        action="store_true",
        help="Print accuracy by (branch, bookie_pick) and (league, branch, bookie_pick) to diagnose band1 side behaviour"
    )

    parser.add_argument(
        "--band1-autopsy",
        action="store_true",
        help="Print band1 autopsy report (league accuracy, p-quantiles, side distribution)"
    )

    parser.add_argument(
        "--autofix-misses",
        action="store_true",
        help="Attempt additional fuzzy join recovery for missed rows using tokens (incl. swapped teams); expands miss cap to 500"
    )

    parser.add_argument(
        "--miss-autopsy",
        action="store_true",
        help="Print per-league diagnostics for join-miss rows (samples + token availability)"
    )

    args = parser.parse_args()

    # Resolve runtime knobs
    pred_path = Path(args.pred_path)
    merged_dir = Path(args.merged_dir)
    date_from = str(args.date_from).strip()
    date_to = str(args.date_to).strip()

    apply_policy = bool(args.apply_policy) and (not bool(args.no_policy))
    # If neither flag is set, fall back to file default
    if (not bool(args.apply_policy)) and (not bool(args.no_policy)):
        apply_policy = bool(APPLY_OVER25_POLICY)

    band1_min_p = float(args.band1_min_p)
    over_only = bool(args.over_only)
    strict_over_signals = bool(getattr(args, "strict_over_signals", False))
    min_league_coverage = float(getattr(args, "min_league_coverage", 0.0) or 0.0)
    side_diagnostics = bool(getattr(args, "side_diagnostics", False))
    deploy_v0 = bool(getattr(args, "deploy_v0", False))
    deploy_branch = str(getattr(args, "deploy_branch", DEPLOY_BRANCH_DEFAULT)).strip() or DEPLOY_BRANCH_DEFAULT
    deploy_min_p = float(getattr(args, "deploy_min_p", 0.0) or 0.0)
    deploy_topq_q = float(getattr(args, "deploy_topq_q", DEPLOY_TOPQ_Q_DEFAULT) or DEPLOY_TOPQ_Q_DEFAULT)
    deploy_signal_mode = str(getattr(args, "deploy_signal_mode", "strict") or "strict").strip().lower()
    deploy_print_signal_dist = bool(getattr(args, "deploy_print_signal_dist", False))
    deploy_signal_recompute = str(getattr(args, "deploy_signal_recompute", DEPLOY_SIGNAL_RECOMPUTE_DEFAULT) or DEPLOY_SIGNAL_RECOMPUTE_DEFAULT).strip().lower()
    deploy_signal_banding = str(getattr(args, "deploy_signal_banding", DEPLOY_SIGNAL_BANDING_DEFAULT) or DEPLOY_SIGNAL_BANDING_DEFAULT).strip().lower()
    deploy_strong_p = float(getattr(args, "deploy_strong_p", DEPLOY_SIGNAL_STRONG_P_DEFAULT) or DEPLOY_SIGNAL_STRONG_P_DEFAULT)
    deploy_vstrong_p = float(getattr(args, "deploy_vstrong_p", DEPLOY_SIGNAL_VSTRONG_P_DEFAULT) or DEPLOY_SIGNAL_VSTRONG_P_DEFAULT)
    deploy_strong_q = float(getattr(args, "deploy_strong_q", DEPLOY_SIGNAL_STRONG_Q_DEFAULT) or DEPLOY_SIGNAL_STRONG_Q_DEFAULT)
    deploy_vstrong_q = float(getattr(args, "deploy_vstrong_q", DEPLOY_SIGNAL_VSTRONG_Q_DEFAULT) or DEPLOY_SIGNAL_VSTRONG_Q_DEFAULT)
    band1_autopsy = bool(getattr(args, "band1_autopsy", False))
    autofix_misses = bool(getattr(args, "autofix_misses", False))
    miss_autopsy = bool(getattr(args, "miss_autopsy", False))
    deploy_profile = str(getattr(args, "deploy_profile", DEPLOY_PROFILE_DEFAULT) or DEPLOY_PROFILE_DEFAULT).strip().lower()
    elite_keep_leagues: set[str] = set(ELITE_KEEP_LEAGUES_DEFAULT)
    # (removed duplicate assignments of deploy_signal_mode and deploy_print_signal_dist)

    # Load league overrides (optional external JSON), otherwise use defaults
    league_overrides = dict(LEAGUE_OVERRIDES_DEFAULT)
    ov_path = str(getattr(args, "deploy_override_json", "") or "").strip()
    if ov_path:
        try:
            _ov = json.loads(Path(ov_path).read_text(encoding="utf-8"))
            if isinstance(_ov, dict):
                # normalize: ensure keys are strings
                league_overrides = {str(k).strip(): (v if isinstance(v, dict) else {}) for k, v in _ov.items()}
                print(f"[DEPLOY_V0] loaded overrides from {ov_path} (n={len(league_overrides)})")
            else:
                print(f"[DEPLOY_V0] overrides json is not a dict: {ov_path}")
        except Exception:
            print(f"[DEPLOY_V0] failed to load overrides json: {ov_path}")
    # --- Load predictions (OU25 primary only)
    try:
        pred = pd.read_csv(pred_path, low_memory=False)
    except FileNotFoundError:
        print(f"[PRED_LOAD] ERROR: file not found: {pred_path}")
        # Best-effort hints
        try:
            parent = pred_path.parent
            if parent.exists():
                # show a few nearby CSVs to help the user spot the correct filename
                cands = sorted([p for p in parent.glob("*.csv")])
                if cands:
                    print("[PRED_LOAD] Nearby CSVs (first 20):")
                    for p in cands[:20]:
                        print("  -", p)
            else:
                # if the parent folder doesn't exist, search under predictions_output for similar suffix
                root = Path("predictions_output")
                if root.exists():
                    tail = pred_path.name
                    hits = list(root.rglob(tail))
                    if hits:
                        print("[PRED_LOAD] Found matching filename(s) under predictions_output:")
                        for h in hits[:20]:
                            print("  -", h)
                    else:
                        # loose search on key tokens
                        toks = [t for t in ["OU25", "COMBINED", "TOPQ", "BOOKIE_IMP40"] if t]
                        loose = []
                        for h in root.rglob("*.csv"):
                            hn = h.name.upper()
                            if all(t in hn for t in toks):
                                loose.append(h)
                                if len(loose) >= 20:
                                    break
                        if loose:
                            print("[PRED_LOAD] Similar CSVs under predictions_output (first 20):")
                            for h in loose:
                                print("  -", h)
        except Exception:
            pass
        raise

    # Always print a minimal load summary (this prevents silent 0-row mysteries)
    try:
        print(f"[PRED_LOAD] file={pred_path} | raw_rows={len(pred)} | cols_n={len(pred.columns)}")
        print("[PRED_LOAD] columns:", sorted([str(c) for c in pred.columns]))
    except Exception:
        pass

    # --------
    # Column harmonization (frozen/filtered artifacts are not consistent)
    # --------
    def _pick_col(cands: list[str]) -> str:
        for c in cands:
            if c in pred.columns:
                return c
        return ""

    league_col = _pick_col(["league", "League", "competition", "Competition", "tournament", "Tournament"])
    fixture_col = _pick_col(["fixture_key", "fixture", "match_key", "game_key", "fixtureId", "fixture_id"])
    bookie_col = _pick_col([
        "bookie_pick", "selection", "pick", "bet_pick", "pred_ou25_label", "pred_label", "prediction", "side"
    ])

    # Normalize/construct core columns
    if league_col:
        pred["league"] = pred[league_col].astype(str).str.strip()
    else:
        pred["league"] = ""

    if fixture_col:
        pred["fixture_key"] = pred[fixture_col].astype(str).str.strip()
    else:
        pred["fixture_key"] = ""

    # Normalize bookie_pick values if we can find a column that represents the side
    if bookie_col:
        bp = pred[bookie_col].astype(str).str.upper().str.strip()
        bp_alias = {
            "OVER2.5": "OVER25",
            "OVER_25": "OVER25",
            "OVER 2.5": "OVER25",
            "OVER 2,5": "OVER25",
            "O25": "OVER25",
            "O2.5": "OVER25",
            "OVER": "OVER25",
            "UNDER2.5": "UNDER25",
            "UNDER_25": "UNDER25",
            "UNDER 2.5": "UNDER25",
            "UNDER 2,5": "UNDER25",
            "U25": "UNDER25",
            "U2.5": "UNDER25",
            "UNDER": "UNDER25",
        }
        pred["bookie_pick"] = bp.replace(bp_alias)
    else:
        pred["bookie_pick"] = ""

    # --------
    # Market filtering
    # --------
    # Some exports omit `market` entirely; some encode OU25 as over25/over2.5/ou_25.
    # If `market` exists, normalize and filter to OU25.
    if "market" in pred.columns:
        m0 = pred["market"].astype(str).str.lower().str.strip()
        _alias_map = {
            "over25": "ou25",
            "over2.5": "ou25",
            "over_25": "ou25",
            "ou_25": "ou25",
            "ou2.5": "ou25",
            "ou_2.5": "ou25",
            "o25": "ou25",
        }
        pred["market"] = m0.replace(_alias_map)
        before_mkt = len(pred)
        pred = pred[pred["market"].eq("ou25")].copy()
        after_mkt = len(pred)
        try:
            print(f"[PRED_LOAD] market filter ou25: {before_mkt} -> {after_mkt}")
        except Exception:
            pass
        if after_mkt == 0 and before_mkt > 0:
            try:
                vc = m0.value_counts(dropna=False).head(25)
                print("[PRED_LOAD] market value_counts (top25):")
                print(vc.to_string())
            except Exception:
                pass

    else:
        print("[PRED_LOAD] NOTE: no `market` column; assuming file is already OU25-only")

    # --------
    # Primary filtering
    # --------
    # Frozen branch CSVs often do NOT include is_market_primary. If missing, assume primary.
    if "is_market_primary" in pred.columns:
        pred["is_market_primary"] = pd.to_numeric(pred["is_market_primary"], errors="coerce").fillna(0).astype(int)
        before_prim = len(pred)
        pred = pred[pred["is_market_primary"].eq(1)].copy()
        after_prim = len(pred)
        print(f"[PRED_LOAD] is_market_primary==1 filter: {before_prim} -> {after_prim}")
    else:
        pred["is_market_primary"] = 1
        print("[PRED_LOAD] NOTE: no `is_market_primary`; assuming primary for all rows")

    # Helpful side distribution before over_only filter
    try:
        if "bookie_pick" in pred.columns and len(pred):
            vc_bp = pred["bookie_pick"].astype(str).str.upper().str.strip().replace({"": np.nan}).value_counts(dropna=False).head(10)
            print("[PRED_LOAD] bookie_pick dist (top10):")
            print(vc_bp.to_string())
    except Exception:
        pass

    # If we can't determine the side, do NOT apply over_only (it would zero everything)
    if over_only:
        if "bookie_pick" not in pred.columns or pred["bookie_pick"].astype(str).str.strip().eq("").all():
            print("[PRED_LOAD] WARNING: over_only requested but no usable pick column found; skipping over_only filter")
        else:
            before_over = len(pred)
            pred = pred[pred["bookie_pick"].eq("OVER25")].copy()
            after_over = len(pred)
            print(f"[PRED_LOAD] over_only OVER25 filter: {before_over} -> {after_over}")

    # Final debug: if empty here, show a few rows + key columns
    if pred.empty:
        try:
            print("[PRED_LOAD] EMPTY after prediction normalization/filters.")
            # print a small preview of the raw file (not filtered) to diagnose naming
            raw_preview = pd.read_csv(pred_path, low_memory=False).head(5)
            print("[PRED_LOAD] raw head(5):")
            print(raw_preview.to_string(index=False))
        except Exception:
            pass
    # Pass A / apples-to-apples: force OVER-only if requested (policy file is OVER-only)
    if over_only:
        pred = pred[pred["bookie_pick"].eq("OVER25")].copy()

    # IMPORTANT: filter predictions to the requested window (otherwise Dec-2025 preds will never join to 2024-11..2025-04 truth)
    before_win = len(pred)

    # Pre-compute parsed dates so we can print min/max when the window filters everything out.
    md_all = _parse_pred_match_date(pred)
    md_min = md_all.min() if isinstance(md_all, pd.Series) and md_all.notna().any() else pd.NaT
    md_max = md_all.max() if isinstance(md_all, pd.Series) and md_all.notna().any() else pd.NaT

    pred = _filter_pred_window(pred, date_from, date_to)
    after_win = len(pred)

    if after_win < before_win:
        msg = f"[PRED_WINDOW] Filtered preds to window {date_from}..{date_to}: {before_win} -> {after_win}"
        if pd.notna(md_min) and pd.notna(md_max):
            msg += f" | pred_date_range={md_min.date()}..{md_max.date()}"
        else:
            msg += " | pred_date_range=unknown (could not parse match_date/fixture_key)"
        print(msg)

    # Guardrail: if the user passed a predictions file outside the scoring window, don't proceed with empty frames.
    if before_win > 0 and after_win == 0:
        print(
            "[PRED_WINDOW] No prediction rows fall inside the requested window. "
            "This usually means --pred-path points to a different month/season than --date-from/--date-to.\n"
            "            Fix: pass the correct windowed ALLMARKETS export via --pred-path, or adjust --date-from/--date-to."
        )
        raise SystemExit(2)

    pred["fixture_key_ascii"] = pred["fixture_key"].map(_fixture_key_ascii).astype(str).str.strip()

    # Fallback join keys (team-token based) for rare fixture_key drift cases
    if "match_date" in pred.columns:
        pred["match_date"] = pred["match_date"].astype(str).str.strip()
    _hc = _first_present_col(pred, ["home_team_name","home_team","home","HomeTeam","team_home"])
    _ac = _first_present_col(pred, ["away_team_name","away_team","away","AwayTeam","team_away"])
    if _hc:
        pred["home_tok"] = pred[_hc].map(_norm_team_token_fuzzy).astype(str).str.strip()
    else:
        pred["home_tok"] = ""
    if _ac:
        pred["away_tok"] = pred[_ac].map(_norm_team_token_fuzzy).astype(str).str.strip()
    else:
        pred["away_tok"] = ""

    # Extra-fuzzy tokens for stubborn join drift (e.g., EFL naming differences)
    if _hc:
        pred["home_tok_x"] = pred[_hc].map(_norm_team_token_extra_fuzzy).astype(str).str.strip()
    else:
        pred["home_tok_x"] = ""
    if _ac:
        pred["away_tok_x"] = pred[_ac].map(_norm_team_token_extra_fuzzy).astype(str).str.strip()
    else:
        pred["away_tok_x"] = ""

    if apply_policy or deploy_v0:
        _print_policy_metadata(POLICY_PATH)

    # Optional: OVER-only policy filter (from over25_from_ou25_policy.json)
    if apply_policy:
        allow_pairs, mode = _load_over25_policy_allowlist(POLICY_PATH)
        if not allow_pairs:
            print(f"[POLICY] APPLY_OVER25_POLICY=True but policy allowlist is {mode}; no filtering applied.")
        else:
            # Require OVER25 picks only (policy is OVER-only)
            before_n = len(pred)
            pred = pred[pred["bookie_pick"].eq("OVER25")].copy()

            # Prefer league+branch allowlist when available
            if "ou25_policy_branch" in pred.columns:
                lg_s = pred["league"].astype(str).str.strip()
                br_s = pred["ou25_policy_branch"].astype(str).str.strip()

                # If allowlist is wildcard-by-branch ("*", branch), match on branch only.
                wildcard_branches = {b for (lg, b) in allow_pairs if lg == "*"}
                if wildcard_branches:
                    m_allow = br_s.isin(list(wildcard_branches))
                else:
                    m_allow = pd.Series(False, index=pred.index)
                    for (lg, br) in allow_pairs:
                        m_allow = m_allow | ((lg_s == lg) & (br_s == br))

                pred = pred[m_allow.fillna(False)].copy()

            after_n = len(pred)
            print(f"[POLICY] Applied OVER-only policy ({mode}): {before_n} -> {after_n} rows")
            # Extra safety: band1 rows must be high-confidence OVER to be deploy-scored
            try:
                if "ou25_policy_branch" in pred.columns:
                    b1 = pred["ou25_policy_branch"].astype(str).str.strip().eq("ou25_band1_124_176")
                    pm = pd.to_numeric(pred.get("model_p_for_bookie", np.nan), errors="coerce")
                    sig = pred.get("signal_over25", "").astype(str).str.strip().str.upper()

                    keep_b1 = (~b1) | (
                        pm.ge(band1_min_p).fillna(False)
                        & sig.isin(["STRONG_OVER", "VERY_STRONG_OVER"])
                    )

                    before_gate = len(pred)
                    pred = pred[keep_b1].copy()
                    after_gate = len(pred)
                    if after_gate < before_gate:
                        print(f"[POLICY] band1 gate (p>={band1_min_p:.2f} & STRONG+/VS+ OVER): {before_gate} -> {after_gate}")
            except Exception:
                pass

    # Deploy-v0 filter: topq + strong signals + per-league overrides
    if deploy_v0:
        before_d = len(pred)

        # Elite profile: hard whitelist leagues before any further deploy filtering
        if deploy_profile == "elite":
            _before_elite = len(pred)
            pred = pred[pred["league"].astype(str).str.strip().isin(elite_keep_leagues)].copy()
            _after_elite = len(pred)
            if _after_elite < _before_elite:
                print(f"[DEPLOY_V0] elite whitelist applied: {_before_elite} -> {_after_elite} rows | leagues={sorted(elite_keep_leagues)}")

        # Optional: recompute signal_over25 from model_p_for_bookie for stability across historical exports.
        # Two modes:
        #   - fixed: global thresholds (deploy_strong_p / deploy_vstrong_p)
        #   - quantile: per-league quantile thresholds (deploy_strong_q / deploy_vstrong_q)
        try:
            if _should_recompute_signal(pred, deploy_signal_recompute):
                if "model_p_for_bookie" in pred.columns:
                    if deploy_signal_banding == "quantile":
                        # Allow per-league quantile overrides from league_overrides (e.g. Championship tightening)
                        per_lg_q: dict[str, tuple[float, float]] = {}
                        try:
                            for _lgk, _cfg in league_overrides.items():
                                if not isinstance(_cfg, dict):
                                    continue
                                if ("strong_q" in _cfg) or ("vstrong_q" in _cfg):
                                    sq = float(_cfg.get("strong_q", deploy_strong_q))
                                    vq = float(_cfg.get("vstrong_q", deploy_vstrong_q))
                                    per_lg_q[str(_lgk).strip()] = (sq, vq)
                        except Exception:
                            per_lg_q = {}

                        sig_new, thr_map = _recompute_signal_over25_quantile_by_league(
                            pred,
                            league_col="league",
                            p_col="model_p_for_bookie",
                            strong_q=deploy_strong_q,
                            vstrong_q=deploy_vstrong_q,
                            per_league_q=per_lg_q,
                        )
                        pred["signal_over25"] = sig_new
                        print(
                            f"[DEPLOY_V0] signal_over25 recomputed (banding=quantile, mode={deploy_signal_recompute}, "
                            f"profile={deploy_profile}, strong_q={deploy_strong_q:.2f}, vstrong_q={deploy_vstrong_q:.2f})"
                        )
                        # Show a few thresholds for transparency (largest leagues by row count)
                        try:
                            lg_counts = pred["league"].astype(str).str.strip().value_counts().head(10)
                            for lg_name, cnt in lg_counts.items():
                                th = thr_map.get(str(lg_name).strip())
                                if th and np.isfinite(th[0]) and np.isfinite(th[1]):
                                    print(f"[DEPLOY_V0]   sig_thr[{lg_name}] strong={th[0]:.4f} vstrong={th[1]:.4f} | n={int(cnt)}")
                        except Exception:
                            pass
                    else:
                        pred["signal_over25"] = _recompute_signal_over25_from_p(
                            pred,
                            p_col="model_p_for_bookie",
                            strong_p=deploy_strong_p,
                            vstrong_p=deploy_vstrong_p,
                        )
                        print(
                            f"[DEPLOY_V0] signal_over25 recomputed from model_p_for_bookie "
                            f"(banding=fixed, mode={deploy_signal_recompute}, strong_p={deploy_strong_p:.2f}, vstrong_p={deploy_vstrong_p:.2f})"
                        )
                else:
                    print("[DEPLOY_V0] signal_over25 recompute requested but model_p_for_bookie is missing")
        except Exception:
            pass

        if deploy_print_signal_dist and ("signal_over25" in pred.columns):
            try:
                _sd0 = pred["signal_over25"].astype(str).str.strip().str.upper().value_counts(dropna=False)
                print("[DEPLOY_V0] signal_over25 dist (pre):")
                print(_sd0.to_string())
            except Exception:
                pass
        if deploy_signal_recompute == "never" and ("signal_over25" in pred.columns):
            try:
                _s0 = pred["signal_over25"].astype("string").fillna("").str.strip().str.upper()
                _s0 = _s0[_s0.ne("")]
                if len(_s0) and float((_s0.eq("NEUTRAL")).mean()) >= 0.90:
                    print("[DEPLOY_V0] WARNING: signal_over25 is >=90% NEUTRAL; consider --deploy-signal-recompute auto/always")
            except Exception:
                pass

        # Require branch (preferred) or fallback to TopQ per-league when branch isn't available
        if "ou25_policy_branch" in pred.columns:
            pred = pred[pred["ou25_policy_branch"].astype(str).str.strip().eq(deploy_branch)].copy()
            print(f"[DEPLOY_V0] branch_enforced=ou25_policy_branch:{deploy_branch}")
        else:
            # Many __BACKTEST_SCORED exports don't carry policy-branch columns.
            # In that case, recompute TopQ from model_p_for_bookie per-league.
            if "model_p_for_bookie" not in pred.columns:
                print("[DEPLOY_V0] missing ou25_policy_branch and model_p_for_bookie; cannot compute TopQ")
            else:
                p_tmp = pd.to_numeric(pred.get("model_p_for_bookie", np.nan), errors="coerce")
                lg_tmp = pred["league"].astype(str).str.strip()

                # Compute per-league quantile threshold
                thr_by_lg = (
                    pred.assign(__p=p_tmp, __lg=lg_tmp)
                        .groupby("__lg", dropna=False)["__p"]
                        .quantile(deploy_topq_q)
                )

                # Keep rows at/above threshold (NaNs drop out)
                thr = lg_tmp.map(thr_by_lg)
                m_topq = p_tmp.notna() & thr.notna() & p_tmp.ge(thr)
                before_topq = len(pred)
                pred = pred[m_topq].copy()

                # Helpful log: show thresholds for the largest 10 leagues by remaining row count
                try:
                    keep_counts = pred["league"].astype(str).str.strip().value_counts().head(10)
                    if not keep_counts.empty:
                        print(f"[DEPLOY_V0] branch_enforced=fallback:TopQ@q={deploy_topq_q:.2f} on model_p_for_bookie (ou25_policy_branch missing)")
                        for lg_name, cnt in keep_counts.items():
                            t = float(thr_by_lg.get(str(lg_name).strip(), np.nan))
                            if np.isfinite(t):
                                print(f"[DEPLOY_V0]   topq_thr[{lg_name}]={t:.4f} | kept_n={int(cnt)}")
                except Exception:
                    print(f"[DEPLOY_V0] branch_enforced=fallback:TopQ@q={deploy_topq_q:.2f} on model_p_for_bookie (ou25_policy_branch missing)")

                print(f"[DEPLOY_V0] TopQ recompute (per-league) applied: {before_topq} -> {len(pred)}")

        # Require OVER25 (deploy-v0 is over-only)
        pred = pred[pred["bookie_pick"].astype(str).str.upper().str.strip().eq("OVER25")].copy()

        # Require allowed signals (global) or per-league override
        sig_raw = pred.get("signal_over25", "").astype("string").fillna("").str.strip().str.upper()
        p_raw = pd.to_numeric(pred.get("model_p_for_bookie", np.nan), errors="coerce")

        # Build row-wise mask with overrides
        m_keep = pd.Series(True, index=pred.index)

        # Global min p
        if deploy_min_p and deploy_min_p > 0.0:
            m_keep &= p_raw.ge(deploy_min_p).fillna(False)

        # Global signals
        # NOTE: historical __BACKTEST_SCORED exports often have most rows tagged NEUTRAL.
        # `strict` replicates your premium-only lane; `relaxed` allows NEUTRAL when p is already TopQ;
        # `none` disables signal gating entirely.
        if deploy_signal_mode == "none":
            pass
        elif deploy_signal_mode == "relaxed":
            # Allow STRONG/VERY_STRONG always; additionally allow NEUTRAL when p is already >= TopQ threshold.
            # We compute `topq_thr` per-league above when ou25_policy_branch is missing.
            # If ou25_policy_branch exists (branch enforced), we do not have per-league `topq_thr` here,
            # so relaxed behaves like strict in that case.
            _is_strong = sig_raw.isin(list(DEPLOY_SIGNALS_DEFAULT)).fillna(False)
            _is_neutral = sig_raw.eq("NEUTRAL")
            _neutral_ok = pd.Series(False, index=pred.index)
            if "ou25_policy_branch" not in pred.columns:
                try:
                    # Recompute per-league TopQ threshold on the *current* pred frame
                    _lg2 = pred["league"].astype(str).str.strip()
                    _p2 = p_raw
                    _thr2 = (
                        pred.assign(__p=_p2, __lg=_lg2)
                            .groupby("__lg", dropna=False)["__p"]
                            .quantile(deploy_topq_q)
                    )
                    _neutral_ok = _p2.notna() & _lg2.map(_thr2).notna() & _p2.ge(_lg2.map(_thr2))
                except Exception:
                    _neutral_ok = pd.Series(False, index=pred.index)
            m_keep &= (_is_strong | (_is_neutral & _neutral_ok))
        else:
            # strict
            m_keep &= sig_raw.isin(list(DEPLOY_SIGNALS_DEFAULT)).fillna(False)

        # Per-league overrides
        lg_s = pred["league"].astype(str).str.strip()
        for lg, cfg in league_overrides.items():
            try:
                lg = str(lg).strip()
                if not lg:
                    continue
                if not isinstance(cfg, dict):
                    continue

                # league mask (must be defined before any per-league action)
                m_lg = lg_s.eq(lg)
                if not bool(m_lg.any()):
                    continue

                # Allow per-league hard exclusion (park volatile leagues)
                o_excl = cfg.get("exclude", False)
                if bool(o_excl):
                    m_keep = m_keep & (~m_lg)
                    continue

                o_min_p = cfg.get("min_p", None)
                o_sigs = cfg.get("signals", None)

                m_cfg = pd.Series(True, index=pred.index)

                if o_min_p is not None:
                    try:
                        o_min_p_f = float(o_min_p)
                        m_cfg &= p_raw.ge(o_min_p_f).fillna(False)
                    except Exception:
                        pass

                if o_sigs is not None:
                    try:
                        if isinstance(o_sigs, (list, set, tuple)):
                            _ss = {str(x).strip().upper() for x in o_sigs}
                        else:
                            _ss = {str(o_sigs).strip().upper()}
                        m_cfg &= sig_raw.isin(list(_ss)).fillna(False)
                    except Exception:
                        pass

                # Apply override mask only to that league
                m_keep = m_keep & (~m_lg | m_cfg)
            except Exception:
                continue

        # Log which leagues were excluded via overrides
        try:
            excl_leagues = [str(k).strip() for k, v in league_overrides.items() if isinstance(v, dict) and bool(v.get("exclude", False))]
            excl_leagues = [x for x in excl_leagues if x]
            if excl_leagues:
                print(f"[DEPLOY_V0] excluded leagues via overrides: {sorted(set(excl_leagues))}")
        except Exception:
            pass

        pred = pred[m_keep.fillna(False)].copy()

        if deploy_print_signal_dist and ("signal_over25" in pred.columns):
            try:
                _sd1 = pred["signal_over25"].astype(str).str.strip().str.upper().value_counts(dropna=False)
                print("[DEPLOY_V0] signal_over25 dist (post):")
                print(_sd1.to_string())
            except Exception:
                pass

        after_d = len(pred)
        print(f"[DEPLOY_V0] Applied deploy-v0 filter: {before_d} -> {after_d} rows")

    # Optional: stricter deploy simulation: only allow STRONG_OVER / VERY_STRONG_OVER signals
    if strict_over_signals:
        try:
            before_sig = len(pred)
            sig = pred.get("signal_over25", "").astype(str).str.strip().str.upper()
            pred = pred[sig.isin(["STRONG_OVER", "VERY_STRONG_OVER"])].copy()
            after_sig = len(pred)
            if after_sig < before_sig:
                print(f"[STRICT] signal_over25 STRONG/VSTRONG OVER only: {before_sig} -> {after_sig}")
        except Exception:
            pass

    print("Pred OU25 primary rows:", len(pred))
    leagues = sorted(pred["league"].unique().tolist())
    print("Leagues in preds:", leagues)

    # --- Load merged truth for leagues present
    truth_frames = []
    for lg in leagues:
        guess = merged_dir / f"{lg.replace(' ', '_')}__merged.csv"
        p = guess if guess.exists() else None
        if p is None:
            hits = list(merged_dir.glob(f"*{lg.replace(' ', '_')}*__merged.csv"))
            if hits:
                p = hits[0]
        if p is None or (not p.exists()):
            print(f"[WARN] merged file not found for league='{lg}' (expected {guess.name})")
            continue

        df = pd.read_csv(p, low_memory=False)
        # Guardrail: some __merged__ files can be stale (stop at season end).
        # If the merged file doesn't reach DATE_FROM, we can't score that league in this window.
        try:
            # Try best column first, but if it is sparse/unparseable, fall back to other candidates.
            cand_cols = []
            _dc0 = pick_best_date_col(df)
            if _dc0:
                cand_cols.append(_dc0)
            for c in ("date_GMT", "date", "Date", "timestamp", "match_date"):
                if c in df.columns and c not in cand_cols:
                    cand_cols.append(c)

            _mx = pd.NaT
            for _dc in cand_cols:
                _md_all = parse_window_dates(df[_dc], _dc)
                if isinstance(_md_all, pd.Series) and _md_all.notna().any():
                    _mx = _md_all.max()
                    if pd.notna(_mx):
                        break

            if pd.notna(_mx) and _mx < pd.Timestamp(date_from, tz="UTC"):
                print(f"[WARN] {lg} merged truth appears stale (max={_mx.date()}); skipping league for this window")
                continue
        except Exception:
            pass

        # ensure league column exists for join
        if "league" not in df.columns:
            df["league"] = lg
        df["league"] = df["league"].astype(str).str.strip()

        # window filter (handles match_date/date_GMT/timestamp)
        dfw = filter_window(df, date_from, date_to)
        if dfw.empty:
            print(f"[WARN] no rows in window for {lg} from {p.name}")
            continue

        # League-specific key hygiene: some merged feeds carry inconsistent/duplicated fixture_key values.
        # For these leagues we rebuild fixture_key deterministically from (match_date, home_team_name, away_team_name)
        # for ALL rows to maximize join stability.
        if str(lg).strip() in {"England EFL League 1"}:
            if all(c in dfw.columns for c in ["match_date", "home_team_name", "away_team_name"]):
                dfw["fixture_key"] = dfw.apply(
                    lambda r: build_fixture_key_from_row(
                        r.get("match_date"), r.get("home_team_name"), r.get("away_team_name")
                    ),
                    axis=1,
                )
                dfw["fixture_key"] = dfw["fixture_key"].astype("string").fillna("").str.strip()
                print(f"[INFO] {lg}: rebuilt fixture_key for ALL rows (league-specific override)")

        # ensure fixture_key exists and is populated
        # Some __merged__ frames have the column but it is blank for many rows.
        need_cols = all(c in dfw.columns for c in ["match_date", "home_team_name", "away_team_name"])

        if "fixture_key" not in dfw.columns:
            dfw["fixture_key"] = pd.NA

        # Normalize existing keys
        try:
            fk0 = dfw["fixture_key"].astype("string").fillna("").str.strip()
        except Exception:
            fk0 = dfw["fixture_key"].astype(str).fillna("").astype(str).str.strip()

        # Rebuild only missing/blank keys
        if need_cols:
            need_fk = fk0.eq("") | fk0.isna()
            if bool(need_fk.any()):
                dfw.loc[need_fk, "fixture_key"] = dfw.loc[need_fk].apply(
                    lambda r: build_fixture_key_from_row(
                        r.get("match_date"), r.get("home_team_name"), r.get("away_team_name")
                    ),
                    axis=1,
                )

        # Final normalize + guard
        dfw["fixture_key"] = dfw["fixture_key"].astype("string").fillna("").str.strip()
        if dfw["fixture_key"].eq("").all():
            print(f"[WARN] {lg} missing fixture_key in {p.name} (rebuild failed or source has no team/date cols)")
            continue
        else:
            # Only log when we actually rebuilt some keys
            try:
                rebuilt_n = int((fk0.eq("") | fk0.isna()).sum())
            except Exception:
                rebuilt_n = 0
            if rebuilt_n:
                print(f"[INFO] {lg}: rebuilt fixture_key for {rebuilt_n} rows")

        dfw["fixture_key_ascii"] = dfw["fixture_key"].map(_fixture_key_ascii).astype(str).str.strip()
        _thc = _first_present_col(dfw, ["home_team_name","home_team","home","HomeTeam","team_home"])
        _tac = _first_present_col(dfw, ["away_team_name","away_team","away","AwayTeam","team_away"])
        if _thc:
            dfw["home_tok"] = dfw[_thc].map(_norm_team_token_fuzzy).astype(str).str.strip()
        else:
            dfw["home_tok"] = ""
        if _tac:
            dfw["away_tok"] = dfw[_tac].map(_norm_team_token_fuzzy).astype(str).str.strip()
        else:
            dfw["away_tok"] = ""
        # Extra-fuzzy tokens for stubborn join drift (e.g., EFL naming differences)
        if _thc:
            dfw["home_tok_x"] = dfw[_thc].map(_norm_team_token_extra_fuzzy).astype(str).str.strip()
        else:
            dfw["home_tok_x"] = ""
        if _tac:
            dfw["away_tok_x"] = dfw[_tac].map(_norm_team_token_extra_fuzzy).astype(str).str.strip()
        else:
            dfw["away_tok_x"] = ""

        # compute total goals
        dfw["total_goals_calc"] = build_total_goals(dfw)

        # Debug: Europa League completion detection (helps explain total_goals_calc being nulled)
        try:
            if str(lg).strip() == "Europa League":
                col_map = {str(c).strip().lower(): c for c in dfw.columns}
                status_key = None
                for k in ("status", "match_status", "fixture_status"):
                    if k in col_map:
                        status_key = col_map[k]
                        break
                if status_key is not None:
                    vv = dfw[status_key].astype("string").fillna("").str.strip().str.lower().value_counts().head(10)
                    print("[DEBUG] Europa League truth status values (top10):")
                    print(vv.to_string())
                else:
                    print("[DEBUG] Europa League truth: no status column detected; inferring completion from goals/result fields")
        except Exception:
            pass

        # IMPORTANT: do not score future/incomplete fixtures as 0 goals.
        # If a merged file includes scheduled fixtures, they often have 0/0 goals.
        try:
            m_done = _is_complete_mask(dfw)
            before_nonnull = int(pd.to_numeric(dfw.get("total_goals_calc", np.nan), errors="coerce").notna().sum())
            dfw.loc[~m_done, "total_goals_calc"] = np.nan
            after_nonnull = int(pd.to_numeric(dfw.get("total_goals_calc", np.nan), errors="coerce").notna().sum())
            if after_nonnull < before_nonnull:
                print(f"[INFO] {lg}: dropped non-complete totals {before_nonnull} -> {after_nonnull}")
        except Exception:
            pass

        # Guardrail: if the merged truth window contains NO completed totals, it is effectively a fixture list.
        # Skip this league so it doesn't pollute join-miss stats or scoring denominators.
        try:
            _nonnull = pd.to_numeric(dfw.get("total_goals_calc", np.nan), errors="coerce").notna()
            if not bool(_nonnull.any()):
                # Helpful status distribution when available
                try:
                    _col_map2 = {str(c).strip().lower(): c for c in dfw.columns}
                    _sk = _col_map2.get("status") or _col_map2.get("match_status") or _col_map2.get("fixture_status")
                    if _sk is not None:
                        _vv = dfw[_sk].astype("string").fillna("").str.strip().str.lower().value_counts().head(10)
                        print(f"[WARN] {lg} has no completed goal totals in-window; skipping league (status top10 shown below)")
                        print(_vv.to_string())
                    else:
                        print(f"[WARN] {lg} has no completed goal totals in-window; skipping league")
                except Exception:
                    print(f"[WARN] {lg} has no completed goal totals in-window; skipping league")
                continue
        except Exception:
            pass

        # Dedupe truth to one row per (league, fixture_key) to avoid one-to-many joins.
        # Some merged sources can contain repeated rows per fixture.
        try:
            # Prefer stable ordering by parsed match_date, then team names
            _md_dt = pd.to_datetime(dfw.get("match_date"), errors="coerce", utc=True)
            if isinstance(_md_dt, pd.Series):
                dfw = (
                    dfw.assign(__md_dt=_md_dt)
                       .sort_values(["__md_dt", "home_team_name", "away_team_name"], kind="mergesort")
                       .drop(columns=["__md_dt"], errors="ignore")
                )
        except Exception:
            pass
        try:
            before_n = len(dfw)
            dfw = dfw.drop_duplicates(subset=["league", "fixture_key"], keep="first").copy()
            after_n = len(dfw)
            if after_n < before_n:
                print(f"[INFO] {lg}: deduped truth rows {before_n} -> {after_n}")
        except Exception:
            pass

        # keep only necessary columns
        keep = [c for c in [
            "league", "fixture_key", "fixture_key_ascii", "match_date",
            "home_team_name", "away_team_name",
            "home_tok", "away_tok", "home_tok_x", "away_tok_x",
            "total_goals_calc"
        ] if c in dfw.columns]
        truth_frames.append(dfw[keep].copy())

    truth = pd.concat(truth_frames, ignore_index=True) if truth_frames else pd.DataFrame()
    print("Truth rows in window:", len(truth))

    if truth.empty:
        raise SystemExit("No truth rows loaded. Check merged filenames/date columns/window.")

    # Final truth de-dupe to guarantee 1-row-per-(league,fixture_key_ascii)
    try:
        # Ensure fixture_key_ascii exists; if blank, rebuild from (match_date, teams)
        base_fk = truth.get("fixture_key_ascii", pd.Series(pd.NA, index=truth.index))
        if base_fk is None or len(base_fk) != len(truth):
            base_fk = pd.Series(pd.NA, index=truth.index)

        base_fk = pd.Series(base_fk, index=truth.index).astype("string").fillna("").str.strip()

        # Start from existing fixture_key when present
        fk_src = truth.get("fixture_key", pd.Series(pd.NA, index=truth.index))
        fk_src = pd.Series(fk_src, index=truth.index).astype("string").fillna("").str.strip()

        fk_ascii = fk_src.map(_fixture_key_ascii).astype("string").fillna("").str.strip()

        # If still blank, rebuild from match_date + team names
        can_rebuild = all(c in truth.columns for c in ["match_date", "home_team_name", "away_team_name"])
        if can_rebuild:
            need = fk_ascii.eq("") | fk_ascii.isna()
            if bool(need.any()):
                rebuilt = truth.loc[need].apply(
                    lambda r: build_fixture_key_from_row(
                        r.get("match_date"), r.get("home_team_name"), r.get("away_team_name")
                    ),
                    axis=1,
                )
                fk_ascii.loc[need] = rebuilt.map(_fixture_key_ascii).astype("string").fillna("").str.strip()

        truth["fixture_key_ascii"] = fk_ascii
        truth["league"] = truth["league"].astype(str).str.strip()
        truth = truth.drop_duplicates(subset=["league", "fixture_key_ascii"], keep="first").copy()
    except Exception:
        pass

    # Only score leagues that actually have truth coverage in this window
    try:
        covered_leagues = set(truth["league"].astype(str).str.strip().unique().tolist())
        before_pred = len(pred)
        pred = pred[pred["league"].astype(str).str.strip().isin(covered_leagues)].copy()
        after_pred = len(pred)
        if after_pred < before_pred:
            print(f"[COVERAGE] Dropped preds with no truth coverage: {before_pred} -> {after_pred}")
    except Exception:
        pass

    # --- Join (prefer ASCII-safe fixture key)
    j = pred.merge(
        truth,
        on=["league", "fixture_key_ascii"],
        how="left",
        suffixes=("", "_truth"),
    )

    miss = int(pd.to_numeric(j.get("total_goals_calc", np.nan), errors="coerce").isna().sum())
    print("Join misses:", miss, f"({miss/len(j):.1%})")

    try:
        m_miss = pd.to_numeric(j.get("total_goals_calc", np.nan), errors="coerce").isna()
        if bool(m_miss.any()):
            miss_by_lg = j.loc[m_miss].groupby("league").size().sort_values(ascending=False)
            print("Join misses by league:")
            print(miss_by_lg.to_string())
            # Extra debug: if ALL misses are Europa League, show a few candidate truth rows by date
            try:
                if (len(miss_by_lg) == 1) and (miss_by_lg.index.astype(str).str.strip().tolist()[0] == "Europa League"):
                    sample = j.loc[m_miss].head(8).copy()
                    print("\n[DEBUG] Europa League miss sample:")
                    cols = [c for c in ["match_date", "home_team_name", "away_team_name", "fixture_key", "fixture_key_ascii"] if c in sample.columns]
                    print(sample[cols].to_string(index=False))

                    if not truth.empty and "league" in truth.columns:
                        t_el = truth[truth["league"].astype(str).str.strip().eq("Europa League")].copy()
                        if not t_el.empty and "match_date" in t_el.columns:
                            md0 = str(sample.get("match_date", "").iloc[0]) if "match_date" in sample.columns else ""
                            cand = t_el[t_el["match_date"].astype(str).str.strip().eq(md0)].copy() if md0 else t_el.head(20).copy()
                            print("\n[DEBUG] Europa League truth candidates (same match_date if available):")
                            cols2 = [c for c in ["match_date", "home_team_name", "away_team_name", "fixture_key", "fixture_key_ascii", "total_goals_calc"] if c in cand.columns]
                            print(cand[cols2].head(20).to_string(index=False))
            except Exception:
                pass
    except Exception:
        pass

    # Focus debug for small residual misses (usually team token drift)
    try:
        m_miss = pd.to_numeric(j.get("total_goals_calc", np.nan), errors="coerce").isna()
        if int(m_miss.sum()) and int(m_miss.sum()) <= 10:
            print("\n[DEBUG] Small miss set (show rows):")
            cols = [c for c in [
                "league", "match_date", "fixture_key", "fixture_key_ascii",
                "home_team_name", "away_team_name", "bookie_pick"
            ] if c in j.columns]
            print(j.loc[m_miss, cols].head(20).to_string(index=False))
    except Exception:
        pass

    # Fallback: token/date-based join for missed rows (optionally expanded + swapped teams)
    try:
        m_miss = pd.to_numeric(j.get("total_goals_calc", np.nan), errors="coerce").isna()
        n_miss = int(m_miss.sum()) if hasattr(m_miss, "sum") else 0

        if miss_autopsy and n_miss:
            print("\n=== JOIN MISS AUTOPSY (pre-fix) ===")
            miss_by_lg = j.loc[m_miss].groupby("league").size().sort_values(ascending=False)
            print(miss_by_lg.to_string())
            for lg, cnt in miss_by_lg.head(8).items():
                samp = j.loc[m_miss & j["league"].astype(str).str.strip().eq(str(lg).strip())].head(5).copy()
                cols = [c for c in ["league","match_date","home_team_name","away_team_name","fixture_key","home_tok","away_tok","bookie_pick"] if c in samp.columns]
                print(f"\n[miss sample] {lg} (n={int(cnt)})")
                if cols:
                    print(samp[cols].to_string(index=False))

        max_miss = 500 if autofix_misses else 25
        if n_miss and n_miss <= max_miss:
            tcols = [c for c in ["league","match_date","home_tok","away_tok","total_goals_calc"] if c in truth.columns]
            have_pred_keys = all(c in j.columns for c in ["league","match_date","home_tok","away_tok"])
            if len(tcols) == 5 and have_pred_keys:
                truth_f = truth[tcols].copy()
                truth_f["league"] = truth_f["league"].astype(str).str.strip()
                truth_f["match_date"] = truth_f["match_date"].astype(str).str.strip()
                truth_f["home_tok"] = truth_f["home_tok"].astype(str).str.strip()
                truth_f["away_tok"] = truth_f["away_tok"].astype(str).str.strip()

                miss_keys = j.loc[m_miss, ["league","match_date","home_tok","away_tok"]].copy()
                miss_keys["league"] = miss_keys["league"].astype(str).str.strip()
                miss_keys["match_date"] = miss_keys["match_date"].astype(str).str.strip()
                miss_keys["home_tok"] = miss_keys["home_tok"].astype(str).str.strip()
                miss_keys["away_tok"] = miss_keys["away_tok"].astype(str).str.strip()

                # 1) direct token join
                fb = miss_keys.merge(truth_f, on=["league","match_date","home_tok","away_tok"], how="left")
                rec = fb["total_goals_calc"].copy()
                got = int(pd.to_numeric(rec, errors="coerce").notna().sum())
                if got:
                    j.loc[m_miss, "total_goals_calc"] = rec.values
                    print(f"[FALLBACK_JOIN] recovered {got}/{n_miss} totals via (league,match_date,home_tok,away_tok)")

                # 2) swapped team join for remaining misses
                m_still = pd.to_numeric(j.get("total_goals_calc", np.nan), errors="coerce").isna()
                n_still = int(m_still.sum()) if hasattr(m_still, "sum") else 0
                if n_still:
                    miss_keys2 = j.loc[m_still, ["league","match_date","home_tok","away_tok"]].copy()
                    miss_keys2["league"] = miss_keys2["league"].astype(str).str.strip()
                    miss_keys2["match_date"] = miss_keys2["match_date"].astype(str).str.strip()
                    miss_keys2["home_tok"] = miss_keys2["home_tok"].astype(str).str.strip()
                    miss_keys2["away_tok"] = miss_keys2["away_tok"].astype(str).str.strip()

                    miss_keys2_sw = miss_keys2.rename(columns={"home_tok":"away_tok","away_tok":"home_tok"})
                    fb2 = miss_keys2_sw.merge(truth_f, on=["league","match_date","home_tok","away_tok"], how="left")
                    rec2 = fb2["total_goals_calc"].copy()
                    got2 = int(pd.to_numeric(rec2, errors="coerce").notna().sum())
                    if got2:
                        j.loc[m_still, "total_goals_calc"] = rec2.values
                        print(f"[FALLBACK_JOIN_SWAP] recovered {got2}/{n_still} totals via swapped tokens")

                # 3) extra-fuzzy token join for remaining misses (common for EFL naming drift)
                m_still2 = pd.to_numeric(j.get("total_goals_calc", np.nan), errors="coerce").isna()
                n_still2 = int(m_still2.sum()) if hasattr(m_still2, "sum") else 0
                if n_still2:
                    tcols_x = [c for c in ["league","match_date","home_tok_x","away_tok_x","total_goals_calc"] if c in truth.columns]
                    have_pred_x = all(c in j.columns for c in ["league","match_date","home_tok_x","away_tok_x"])
                    if len(tcols_x) == 5 and have_pred_x:
                        truth_fx = truth[tcols_x].copy()
                        truth_fx["league"] = truth_fx["league"].astype(str).str.strip()
                        truth_fx["match_date"] = truth_fx["match_date"].astype(str).str.strip()
                        truth_fx["home_tok_x"] = truth_fx["home_tok_x"].astype(str).str.strip()
                        truth_fx["away_tok_x"] = truth_fx["away_tok_x"].astype(str).str.strip()

                        miss_keys_x = j.loc[m_still2, ["league","match_date","home_tok_x","away_tok_x"]].copy()
                        miss_keys_x["league"] = miss_keys_x["league"].astype(str).str.strip()
                        miss_keys_x["match_date"] = miss_keys_x["match_date"].astype(str).str.strip()
                        miss_keys_x["home_tok_x"] = miss_keys_x["home_tok_x"].astype(str).str.strip()
                        miss_keys_x["away_tok_x"] = miss_keys_x["away_tok_x"].astype(str).str.strip()

                        fb3 = miss_keys_x.merge(truth_fx, on=["league","match_date","home_tok_x","away_tok_x"], how="left")
                        rec3 = fb3["total_goals_calc"].copy()
                        got3 = int(pd.to_numeric(rec3, errors="coerce").notna().sum())
                        if got3:
                            j.loc[m_still2, "total_goals_calc"] = rec3.values
                            print(f"[FALLBACK_JOIN_X] recovered {got3}/{n_still2} totals via (league,match_date,home_tok_x,away_tok_x)")

                        # 4) swapped teams, extra-fuzzy
                        m_still3 = pd.to_numeric(j.get("total_goals_calc", np.nan), errors="coerce").isna()
                        n_still3 = int(m_still3.sum()) if hasattr(m_still3, "sum") else 0
                        if n_still3:
                            mk2 = j.loc[m_still3, ["league","match_date","home_tok_x","away_tok_x"]].copy()
                            mk2["league"] = mk2["league"].astype(str).str.strip()
                            mk2["match_date"] = mk2["match_date"].astype(str).str.strip()
                            mk2["home_tok_x"] = mk2["home_tok_x"].astype(str).str.strip()
                            mk2["away_tok_x"] = mk2["away_tok_x"].astype(str).str.strip()

                            mk2_sw = mk2.rename(columns={"home_tok_x":"away_tok_x","away_tok_x":"home_tok_x"})
                            fb4 = mk2_sw.merge(truth_fx, on=["league","match_date","home_tok_x","away_tok_x"], how="left")
                            rec4 = fb4["total_goals_calc"].copy()
                            got4 = int(pd.to_numeric(rec4, errors="coerce").notna().sum())
                            if got4:
                                j.loc[m_still3, "total_goals_calc"] = rec4.values
                                print(f"[FALLBACK_JOIN_X_SWAP] recovered {got4}/{n_still3} totals via swapped extra-fuzzy tokens")

                # 5) date-slack join for remaining misses (handles rare +/-1 day drift)
                # Some feeds can shift the calendar day via timezone parsing. For stubborn misses (often EFL),
                # try matching by (league, home_tok_x, away_tok_x) within +/- 1 day.
                m_still4 = pd.to_numeric(j.get("total_goals_calc", np.nan), errors="coerce").isna()
                n_still4 = int(m_still4.sum()) if hasattr(m_still4, "sum") else 0
                if n_still4:
                    try:
                        # Build a compact truth lookup keyed by (league, home_tok_x, away_tok_x)
                        if all(c in truth.columns for c in ["league", "match_date", "home_tok_x", "away_tok_x", "total_goals_calc"]):
                            t = truth[["league", "match_date", "home_tok_x", "away_tok_x", "total_goals_calc"]].copy()
                            t["league"] = t["league"].astype(str).str.strip()
                            t["match_date"] = t["match_date"].astype(str).str.strip()
                            t["home_tok_x"] = t["home_tok_x"].astype(str).str.strip()
                            t["away_tok_x"] = t["away_tok_x"].astype(str).str.strip()
                            t["__md_dt"] = pd.to_datetime(t["match_date"], errors="coerce", utc=True)

                            # Index: key -> list of (md_dt, total)
                            lookup: dict[tuple[str, str, str], list[tuple[pd.Timestamp, float]]] = {}
                            for r in t.itertuples(index=False):
                                key = (str(r.league), str(r.home_tok_x), str(r.away_tok_x))
                                md = getattr(r, "__md_dt")
                                tot = getattr(r, "total_goals_calc")
                                if pd.isna(md):
                                    continue
                                try:
                                    tot_f = float(tot)
                                except Exception:
                                    continue
                                lookup.setdefault(key, []).append((md, tot_f))

                            # Attempt to fill remaining misses
                            filled = 0
                            filled_sw = 0
                            slack_days = 1
                            for idx, rr in j.loc[m_still4, ["league", "match_date", "home_tok_x", "away_tok_x"]].iterrows():
                                lg = str(rr.get("league", "")).strip()
                                md0 = pd.to_datetime(str(rr.get("match_date", "")).strip(), errors="coerce", utc=True)
                                h = str(rr.get("home_tok_x", "")).strip()
                                a = str(rr.get("away_tok_x", "")).strip()
                                if not lg or pd.isna(md0) or (not h) or (not a):
                                    continue

                                # direct
                                cand = lookup.get((lg, h, a), [])
                                best = None
                                best_dt = None
                                if cand:
                                    for md_c, tot_c in cand:
                                        dd = abs((md_c - md0).days)
                                        if dd <= slack_days:
                                            if (best is None) or (dd < best_dt):
                                                best = tot_c
                                                best_dt = dd
                                if best is not None:
                                    j.at[idx, "total_goals_calc"] = best
                                    filled += 1
                                    continue

                                # swapped teams
                                cand2 = lookup.get((lg, a, h), [])
                                best2 = None
                                best2_dt = None
                                if cand2:
                                    for md_c, tot_c in cand2:
                                        dd = abs((md_c - md0).days)
                                        if dd <= slack_days:
                                            if (best2 is None) or (dd < best2_dt):
                                                best2 = tot_c
                                                best2_dt = dd
                                if best2 is not None:
                                    j.at[idx, "total_goals_calc"] = best2
                                    filled_sw += 1

                            if (filled + filled_sw) > 0:
                                print(f"[FALLBACK_JOIN_DATE_SLACK] recovered {filled+filled_sw}/{n_still4} totals via (league,home_tok_x,away_tok_x) within +/-{slack_days} day(s) (swapped={filled_sw})")
                    except Exception:
                        pass

        if miss_autopsy:
            m_after = pd.to_numeric(j.get("total_goals_calc", np.nan), errors="coerce").isna()
            n_after = int(m_after.sum()) if hasattr(m_after, "sum") else 0
            if n_after:
                print("\n=== JOIN MISS AUTOPSY (post-fix) ===")
                miss_by_lg2 = j.loc[m_after].groupby("league").size().sort_values(ascending=False)
                print(miss_by_lg2.to_string())
            else:
                print("\n[JOIN_MISS] no remaining misses after autofix")
    except Exception:
        pass

    # --- Coverage diagnostics (per-league) so denominators stay honest
    try:
        cov = (
            j.groupby("league", dropna=False)
             .agg(
                 pred_n=("league", "size"),
                 scored_n=("total_goals_calc", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
                 miss_n=("total_goals_calc", lambda s: int(pd.to_numeric(s, errors="coerce").isna().sum())),
             )
        )
        cov["coverage_pct"] = ((cov["scored_n"] / cov["pred_n"]) * 100.0).round(2)
        print("\n=== Truth coverage by league (pred vs scored) ===")
        print(cov.sort_values(["pred_n", "coverage_pct"], ascending=[False, True]).to_string())

        # Optional: drop low-coverage leagues from scoring
        if min_league_coverage and (min_league_coverage > 0.0):
            thr_pct = float(min_league_coverage) * 100.0
            keep_leagues = cov[(cov["pred_n"] > 0) & ((cov["scored_n"] / cov["pred_n"]) >= float(min_league_coverage))].index
            before_drop = len(j)
            j = j[j["league"].isin(list(keep_leagues))].copy()
            after_drop = len(j)
            if after_drop < before_drop:
                print(f"[COVERAGE_DROP] Dropped leagues with coverage < {thr_pct:.1f}%: {before_drop} -> {after_drop}")
    except Exception:
        pass

    # --- Score
    # Actual market label from truth totals (OU25 threshold)
    j["actual_ou25_label"] = np.where(j["total_goals_calc"].notna() & j["total_goals_calc"].ge(3), "OVER25", "UNDER25")
    # Pred label is bookie_pick (expected OVER25/UNDER25)
    j["pred_ou25_label"] = j["bookie_pick"].astype(str).str.upper().str.strip()

    j["correct"] = (j["pred_ou25_label"] == j["actual_ou25_label"]) & j["total_goals_calc"].notna()

    scored = j[j["total_goals_calc"].notna()].copy()
    acc = float(scored["correct"].mean()) if len(scored) else float("nan")

    print("\n=== OU25 BACKTEST (primary only) ===")
    print("Scored rows:", len(scored), "/", len(j))
    print("Accuracy:", round(acc * 100, 2), "%")

    # breakdowns
    if "signal_over25" in j.columns:
        print("\n=== Accuracy by signal_over25 ===")
        g = scored.groupby("signal_over25")["correct"].agg(["count", "mean"]).sort_values("count", ascending=False)
        g["acc_pct"] = (g["mean"] * 100).round(2)
        print(g[["count", "acc_pct"]])

    if "ou25_policy_branch" in j.columns:
        print("\n=== Accuracy by ou25_policy_branch ===")
        g = scored.groupby("ou25_policy_branch")["correct"].agg(["count", "mean"]).sort_values("count", ascending=False)
        g["acc_pct"] = (g["mean"] * 100).round(2)
        print(g[["count", "acc_pct"]])

    if "ou25_runtime_lane" in j.columns:
        print("\n=== Accuracy by ou25_runtime_lane ===")
        g = scored.groupby("ou25_runtime_lane")["correct"].agg(["count", "mean"]).sort_values("count", ascending=False)
        g["acc_pct"] = (g["mean"] * 100).round(2)
        print(g[["count", "acc_pct"]])

    # --- Extra summaries (scored only)
    print("\n=== Accuracy by league (scored only) ===")
    try:
        g = scored.groupby("league")["correct"].agg(["count", "mean"]).sort_values("count", ascending=False)
        g["acc_pct"] = (g["mean"] * 100).round(2)
        print(g[["count", "acc_pct"]].to_string())
    except Exception:
        print("(failed to compute)")

    if "ou25_policy_branch" in scored.columns:
        print("\n=== Accuracy by (league, ou25_policy_branch) top 20 by count ===")
        try:
            gb = (
                scored.groupby(["league", "ou25_policy_branch"])["correct"]
                .agg(["count", "mean"])
                .sort_values("count", ascending=False)
                .head(20)
            )
            gb["acc_pct"] = (gb["mean"] * 100).round(2)
            print(gb[["count", "acc_pct"]].to_string())
        except Exception:
            print("(failed to compute)")

    # sample wrong rows
    print("\n=== Sample incorrect rows ===")
    cols_show = [c for c in [
        "league","match_date","fixture_key","home_team_name","away_team_name",
        "bookie_pick","pred_ou25_label","actual_ou25_label","total_goals_calc","signal_over25","ou25_policy_branch","ou25_runtime_lane"
    ] if c in j.columns]
    bad = scored[~scored["correct"]].copy()
    print(bad[cols_show].head(30).to_string(index=False))

    # --- Side diagnostics (useful for diagnosing band1 anti-signal when running --all-sides)
    if side_diagnostics:
        try:
            print("\n=== Accuracy by (ou25_policy_branch, bookie_pick) ===")
            gb = (
                scored.groupby(["ou25_policy_branch", "bookie_pick"], dropna=False)["correct"]
                .agg(["count", "mean"])
                .sort_values("count", ascending=False)
            )
            gb["acc_pct"] = (gb["mean"] * 100).round(2)
            print(gb[["count", "acc_pct"]].to_string())
        except Exception:
            print("(failed to compute branch x side)")

        try:
            print("\n=== Accuracy by (league, ou25_policy_branch, bookie_pick) top 25 by count ===")
            glb = (
                scored.groupby(["league", "ou25_policy_branch", "bookie_pick"], dropna=False)["correct"]
                .agg(["count", "mean"])
                .sort_values("count", ascending=False)
                .head(25)
            )
            glb["acc_pct"] = (glb["mean"] * 100).round(2)
            print(glb[["count", "acc_pct"]].to_string())
        except Exception:
            print("(failed to compute league x branch x side)")

    if band1_autopsy:
        _band1_autopsy(scored)

    # write a scored join file for later analysis
    out_path = Path("predictions_output") / "ou25_backtest_joined_dec2025.csv"
    j.to_csv(out_path, index=False)
    print("\nWROTE joined scored file:", out_path)

if __name__ == "__main__":
    main()
