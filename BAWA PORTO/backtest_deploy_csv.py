#!/usr/bin/env python3
"""backtest_deploy_csv.py

Backtest a deploy CSV by joining to realised goals from per-league
Matches/<League>/fd_odds_enriched.csv (multi-season).

Outputs:
- <deploy>__BACKTEST.csv (row-level joined + scored)
- <deploy>__BACKTEST_SUMMARY.csv (aggregate metrics)

Metrics:
- hit rate
- avg_od
- ROI = mean(correct * odds - 1) using 1u flat staking

Designed to be leak-safe: uses realised goals from Matches files only.

Notes
- Supports markets: ftr, ou25, btts, fts, ge2, ge3, wtn, tg15 (HOME_TG15/AWAY_TG15), tg25 (HOME_TG25/AWAY_TG25).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import unicodedata


import numpy as np
import pandas as pd

# Status heuristics (used to avoid scoring scheduled/placeholder rows as if final)

_COMPLETED_STATUS_RE = r"\bcomplete(?:d)?\b|\bft\b|full\s*time|finished|final|match\s*finished|aet|after\s*extra\s*time|pens?|penalt(?:y|ies)|awarded|ended"
_INCOMPLETE_STATUS_RE = r"incomplete|postp|postponed|abandon|suspend|void|cancel|walkover|\bwo\b|\bns\b|not\s*started|scheduled|upcoming|fixture|tbd|live|in\s*play"

# UEFA competitions often have qualifiers / prelim rounds that can be filed under a different UEFA folder
# than the deploy row's `league` value. For those leagues only, we allow truth lookups across the
# three UEFA competition CSVs to reduce `correct` NaNs caused by fixture_key being present in a sibling file.
_UEFA_COMP_LEAGUES = {"Champions League", "Europa League", "Europa Conference"}


def _coalesce_match_dt_utc(m: pd.DataFrame) -> pd.Series:
    """Best-effort UTC datetime series from common date columns."""
    idx = m.index
    dt = pd.Series(pd.NaT, index=idx, dtype="datetime64[ns, UTC]")

    for col in ("match_date", "date_GMT", "date", "Date"):
        if col in m.columns:
            try:
                cand = pd.to_datetime(m[col], errors="coerce", utc=True)
            except Exception:
                cand = pd.to_datetime(m[col].astype(str), errors="coerce", utc=True)
            dt = dt.fillna(cand)

    if "timestamp" in m.columns:
        ts = pd.to_numeric(m["timestamp"], errors="coerce")
        if ts.notna().any():
            try:
                med = float(ts.dropna().median())
            except Exception:
                med = 0.0
            unit = "ms" if med > 1e12 else "s"
            cand = pd.to_datetime(ts, errors="coerce", utc=True, unit=unit)
            dt = dt.fillna(cand)

    return dt


def _mask_unrealised_rows(m: pd.DataFrame) -> pd.Series:
    """Mask rows that should not be scored as realised results.

    We treat rows as unrealised when:
      - status indicates incomplete/live/scheduled/postponed, OR
      - match datetime is in the future, OR
      - status is blank AND scoreline is 0-0 (common placeholder)
    and status does NOT strongly indicate completion.
    """
    idx = m.index
    hg = pd.to_numeric(m.get("home_team_goal_count"), errors="coerce")
    ag = pd.to_numeric(m.get("away_team_goal_count"), errors="coerce")

    if "status" in m.columns:
        st = m["status"].astype(str).str.lower().str.strip()
        st_complete = st.str.contains(_COMPLETED_STATUS_RE, regex=True)
        st_incomp = st.str.contains(_INCOMPLETE_STATUS_RE, regex=True)
    else:
        st = pd.Series("", index=idx, dtype="string")
        st_complete = pd.Series(False, index=idx)
        st_incomp = pd.Series(False, index=idx)

    dt = _coalesce_match_dt_utc(m)
    now = pd.Timestamp.now(tz="UTC")
    is_future = dt.notna() & (dt > now)

    # Ambiguous placeholders: blank status + 0-0.
    # Important: real matches can finish 0-0, so only treat 0-0 as a placeholder when
    # (a) the row has very little other match information, AND
    # (b) the match date is unknown or very recent (to avoid masking historical 0-0s).
    st_blank = st.fillna("").eq("")
    is_00 = hg.fillna(0).eq(0) & ag.fillna(0).eq(0)

    info_cols = [
        "home_team_shots",
        "away_team_shots",
        "home_team_corner_count",
        "away_team_corner_count",
        "home_team_yellow_cards",
        "away_team_yellow_cards",
        "attendance",
    ]
    present = [c for c in info_cols if c in m.columns]
    if present:
        info = pd.concat([pd.to_numeric(m[c], errors="coerce") for c in present], axis=1)
        low_info = info.fillna(0).sum(axis=1).eq(0)
    else:
        low_info = pd.Series(True, index=idx)

    recent_cutoff = now - pd.Timedelta(days=2)
    ambiguous_00 = st_blank & is_00 & low_info & (dt.isna() | (dt > recent_cutoff))

    unrealised = (st_incomp | is_future | ambiguous_00) & (~st_complete)
    return unrealised.fillna(False)


def _to_num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _ascii_slug(s: object) -> str:
    """Slug used inside fixture_key (deploy-compatible).

    Important: this intentionally does *not* transliterate accents.
    It mirrors the deploy key style where non [A-Za-z0-9] characters
    become underscores (e.g. 'München' -> 'M_nchen').
    """
    if s is None:
        return ""
    txt = str(s)

    # Remove apostrophes (deploy keys typically drop them)
    txt = txt.replace("'", "")

    # Replace anything not ASCII alnum with underscores (do NOT normalize to ASCII)
    txt = re.sub(r"[^A-Za-z0-9]+", "_", txt)

    # Collapse repeats and trim
    txt = re.sub(r"_+", "_", txt).strip("_")
    return txt


def _synth_fixture_key(m: pd.DataFrame) -> pd.Series:
    """Best-effort synthesis of deploy fixture_key: YYYY_MM_DD_Home_Away."""
    dt = _coalesce_match_dt_utc(m)
    # Use UTC date component
    d = dt.dt.strftime("%Y_%m_%d")

    home = pd.Series("", index=m.index, dtype="string")
    away = pd.Series("", index=m.index, dtype="string")
    if "home_team_name" in m.columns:
        home = m["home_team_name"].astype(str)
    if "away_team_name" in m.columns:
        away = m["away_team_name"].astype(str)

    home_slug = home.map(_ascii_slug)
    away_slug = away.map(_ascii_slug)

    fk = d.astype("string") + "_" + home_slug.astype("string") + "_" + away_slug.astype("string")
    # If we couldn't form a date, return NA
    fk = fk.where(d.notna(), pd.NA)
    return fk

def _load_matches(matches_root: Path, league: str) -> Optional[pd.DataFrame]:
    """Load realised goals for a league.

    Supports two layouts:
      A) Matches/__merged__/<LeagueName>__merged.csv   (optional layout)
      B) Matches/<LeagueName>/fd_odds_enriched.csv     (current/legacy layout)

    Extra robustness:
      - For UEFA competitions (UCL/UEL/UECL), qualifiers and prelims can sometimes be filed under a
        sibling UEFA folder. When the merged layout is not available, we union truth rows across all
        UEFA competition folders to reduce `correct` NaNs.
    """
    league = str(league).strip()

    # 1) Optional merged layout: Matches/__merged__/
    merged_dir = matches_root / "__merged__"
    merged_candidates = [
        merged_dir / f"{league.replace(' ', '_')}__merged.csv",
        merged_dir / f"{league}__merged.csv",  # in case spaces are kept
        merged_dir / "fd_odds_enriched_synth.csv",  # last-resort global file (rare)
    ]

    merged_path: Optional[Path] = None
    for p in merged_candidates:
        if p.exists():
            merged_path = p
            break

    # 2) Legacy layout: Matches/<League>/fd_odds_enriched.csv
    legacy_path = matches_root / league / "fd_odds_enriched.csv"

    # If we have a merged file, use it as the single source of truth.
    if merged_path is not None and merged_path.exists():
        paths_to_load = [merged_path]
    else:
        # Otherwise load the league folder file, plus (for UEFA comps) sibling UEFA files.
        paths_to_load: List[Path] = []
        if legacy_path.exists():
            paths_to_load.append(legacy_path)

        if league in _UEFA_COMP_LEAGUES:
            for other in sorted(_UEFA_COMP_LEAGUES):
                p = matches_root / other / "fd_odds_enriched.csv"
                if p.exists() and p not in paths_to_load:
                    paths_to_load.append(p)

    if not paths_to_load:
        return None

    frames: List[pd.DataFrame] = []
    for path in paths_to_load:
        try:
            frames.append(pd.read_csv(path))
        except Exception:
            continue

    if not frames:
        return None

    m = pd.concat(frames, ignore_index=True)

    # Normalise column names (some CSVs may carry trailing spaces / BOM)
    try:
        m.columns = [str(c).strip().lstrip("\ufeff") for c in m.columns]
    except Exception:
        pass

    # Ensure fixture_key is present (case/space robustness)
    if "fixture_key" not in m.columns:
        fk = None
        for c in m.columns:
            norm = str(c).strip().lstrip("\ufeff").lower().replace(" ", "_")
            if norm in ("fixture_key", "fixturekey"):
                fk = c
                break
        if fk is not None:
            m = m.rename(columns={fk: "fixture_key"})

    # Build deploy-style fixture key (YYYY_MM_DD_Home_Away) from date + team names.
    # We compute this even if the file already has a fixture_key, because some merged
    # outputs use other IDs (e.g. numeric match_id) that won't match deploy CSV keys.
    try:
        fk_synth = _synth_fixture_key(m)
    except Exception:
        fk_synth = pd.Series(pd.NA, index=m.index, dtype="string")

    # If fixture_key is missing, use the synthesized one.
    if "fixture_key" not in m.columns:
        m["fixture_key"] = fk_synth
    else:
        # If fixture_key exists but is not deploy-style for many rows, overwrite those rows
        # with the synthesized deploy-style key.
        fk_ser = m["fixture_key"].astype(str)
        fk_ser = fk_ser.replace({"nan": "", "None": "", "<NA>": ""}).str.strip()

        # Deploy-style pattern: YYYY_MM_DD_...
        deploy_like = fk_ser.str.match(r"^\d{4}_\d{2}_\d{2}_")

        # Overwrite rows that are blank or not deploy-like, when we have a synthesized value.
        bad = (~deploy_like) | (fk_ser == "")
        if bad.any():
            m.loc[bad & fk_synth.notna(), "fixture_key"] = fk_synth.loc[bad & fk_synth.notna()]

    # Last-resort fallback: some truth files only carry match_id / fixture_id
    if "fixture_key" not in m.columns or m["fixture_key"].isna().all():
        for alt in ("match_id", "fixture_id", "fixtureid", "id"):
            if alt in m.columns:
                m["fixture_key"] = m[alt].astype(str)
                break

    if "fixture_key" not in m.columns:
        return None

    # Normalise fixture_key formatting
    m["fixture_key"] = m["fixture_key"].astype(str).str.strip()
    m.loc[m["fixture_key"].isin(["", "nan", "None", "<NA>"]), "fixture_key"] = pd.NA

    m["home_team_goal_count"] = _to_num(m.get("home_team_goal_count"))
    m["away_team_goal_count"] = _to_num(m.get("away_team_goal_count"))

    # Prevent scoring scheduled/placeholder rows as realised results
    try:
        unrealised = _mask_unrealised_rows(m)
        if bool(unrealised.any()):
            m.loc[unrealised, "home_team_goal_count"] = np.nan
            m.loc[unrealised, "away_team_goal_count"] = np.nan
    except Exception:
        pass

    # Keep last occurrence per fixture_key (multi-season duplicates can exist)
    sort_cols = [c for c in ["match_date", "date_GMT", "timestamp"] if c in m.columns]
    if sort_cols:
        m = m.sort_values(sort_cols)
    m = m.drop_duplicates(subset=["fixture_key"], keep="last")

    return m[["fixture_key", "home_team_goal_count", "away_team_goal_count"]].copy()

def _compute_correct(df: pd.DataFrame) -> pd.Series:
    mk = df["market"].astype(str).str.lower().str.strip()
    pick = df.get("bookie_pick", "").astype(str).str.upper().str.strip()

    hg = _to_num(df.get("home_team_goal_count"))
    ag = _to_num(df.get("away_team_goal_count"))

    known = hg.notna() & ag.notna()
    tot = hg + ag

    correct = pd.Series(np.nan, index=df.index, dtype="float")

    # FTR
    m_ftr = known & mk.eq("ftr")
    if m_ftr.any():
        real = np.where(hg > ag, "HOME", np.where(hg < ag, "AWAY", "DRAW"))
        correct.loc[m_ftr] = (
            pick.loc[m_ftr].to_numpy()
            == pd.Series(real, index=df.index).loc[m_ftr].to_numpy()
        ).astype(float)

    # OU25
    m_ou = known & mk.eq("ou25")
    if m_ou.any():
        p = pick.loc[m_ou]
        t = tot.loc[m_ou]
        out = np.full(len(p), np.nan)
        p_arr = p.to_numpy()
        t_arr = t.to_numpy()
        m_over = p_arr == "OVER25"
        m_under = p_arr == "UNDER25"
        rhs_over = (t_arr >= 3).astype(float)
        rhs_under = (t_arr <= 2).astype(float)
        if m_over.any():
            out[m_over] = rhs_over[m_over]
        if m_under.any():
            out[m_under] = rhs_under[m_under]
        correct.loc[m_ou] = out

    # BTTS
    m_btts = known & mk.eq("btts")
    if m_btts.any():
        yes = (hg >= 1) & (ag >= 1)
        p = pick.loc[m_btts]
        y = yes.loc[m_btts]
        out = np.full(len(p), np.nan)
        p_arr = p.to_numpy()
        y_arr = y.to_numpy()
        m_yes = p_arr == "YES"
        m_no = p_arr == "NO"
        rhs_yes = y_arr.astype(float)
        rhs_no = (~y_arr).astype(float)
        if m_yes.any():
            out[m_yes] = rhs_yes[m_yes]
        if m_no.any():
            out[m_no] = rhs_no[m_no]
        correct.loc[m_btts] = out

    # FTS (Fail To Score)
    m_fts = known & mk.eq("fts")
    if m_fts.any():
        p = pick.loc[m_fts]
        hh = hg.loc[m_fts]
        aa = ag.loc[m_fts]
        out = np.full(len(p), np.nan)
        p_arr = p.to_numpy()
        hh_arr = hh.to_numpy()
        aa_arr = aa.to_numpy()
        m_home = p_arr == "HOME"
        m_away = p_arr == "AWAY"
        rhs_home = (hh_arr == 0).astype(float)
        rhs_away = (aa_arr == 0).astype(float)
        if m_home.any():
            out[m_home] = rhs_home[m_home]
        if m_away.any():
            out[m_away] = rhs_away[m_away]
        correct.loc[m_fts] = out

    # GE2 (team >=2 goals)
    m_ge2 = known & mk.eq("ge2")
    if m_ge2.any():
        p = pick.loc[m_ge2]
        hh = hg.loc[m_ge2]
        aa = ag.loc[m_ge2]
        out = np.full(len(p), np.nan)
        p_arr = p.to_numpy()
        hh_arr = hh.to_numpy()
        aa_arr = aa.to_numpy()
        m_home = p_arr == "HOME"
        m_away = p_arr == "AWAY"
        rhs_home = (hh_arr >= 2).astype(float)
        rhs_away = (aa_arr >= 2).astype(float)
        if m_home.any():
            out[m_home] = rhs_home[m_home]
        if m_away.any():
            out[m_away] = rhs_away[m_away]
        correct.loc[m_ge2] = out

    # GE3 (team >=3 goals)
    m_ge3 = known & mk.eq("ge3")
    if m_ge3.any():
        p = pick.loc[m_ge3]
        hh = hg.loc[m_ge3]
        aa = ag.loc[m_ge3]
        out = np.full(len(p), np.nan)
        p_arr = p.to_numpy()
        hh_arr = hh.to_numpy()
        aa_arr = aa.to_numpy()
        m_home = p_arr == "HOME"
        m_away = p_arr == "AWAY"
        rhs_home = (hh_arr >= 3).astype(float)
        rhs_away = (aa_arr >= 3).astype(float)
        if m_home.any():
            out[m_home] = rhs_home[m_home]
        if m_away.any():
            out[m_away] = rhs_away[m_away]
        correct.loc[m_ge3] = out

    # WTN (Win To Nil)
    m_wtn = known & mk.eq("wtn")
    if m_wtn.any():
        p = pick.loc[m_wtn]
        hh = hg.loc[m_wtn]
        aa = ag.loc[m_wtn]
        out = np.full(len(p), np.nan)
        p_arr = p.to_numpy()
        hh_arr = hh.to_numpy()
        aa_arr = aa.to_numpy()
        m_home = np.isin(p_arr, ["HOME_WTN", "HOME"])
        m_away = np.isin(p_arr, ["AWAY_WTN", "AWAY"])
        rhs_home = ((hh_arr > aa_arr) & (aa_arr == 0)).astype(float)
        rhs_away = ((aa_arr > hh_arr) & (hh_arr == 0)).astype(float)
        if m_home.any():
            out[m_home] = rhs_home[m_home]
        if m_away.any():
            out[m_away] = rhs_away[m_away]
        correct.loc[m_wtn] = out

    # Team Goals TG15 (team >=2 goals)
    m_tg15 = known & mk.eq("tg15")
    if m_tg15.any():
        p = pick.loc[m_tg15]
        hh = hg.loc[m_tg15]
        aa = ag.loc[m_tg15]
        out = np.full(len(p), np.nan)
        p_arr = p.to_numpy()
        hh_arr = hh.to_numpy()
        aa_arr = aa.to_numpy()
        m_home = p_arr == "HOME_TG15"
        m_away = p_arr == "AWAY_TG15"
        rhs_home = (hh_arr >= 2).astype(float)
        rhs_away = (aa_arr >= 2).astype(float)
        if m_home.any():
            out[m_home] = rhs_home[m_home]
        if m_away.any():
            out[m_away] = rhs_away[m_away]
        correct.loc[m_tg15] = out

    # Team Goals TG25 (team >=3 goals)
    m_tg25 = known & mk.eq("tg25")
    if m_tg25.any():
        p = pick.loc[m_tg25]
        hh = hg.loc[m_tg25]
        aa = ag.loc[m_tg25]
        out = np.full(len(p), np.nan)
        p_arr = p.to_numpy()
        hh_arr = hh.to_numpy()
        aa_arr = aa.to_numpy()
        m_home = p_arr == "HOME_TG25"
        m_away = p_arr == "AWAY_TG25"
        rhs_home = (hh_arr >= 3).astype(float)
        rhs_away = (aa_arr >= 3).astype(float)
        if m_home.any():
            out[m_home] = rhs_home[m_home]
        if m_away.any():
            out[m_away] = rhs_away[m_away]
        correct.loc[m_tg25] = out

    return correct


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy-csv", required=True, help="Path to deploy CSV")
    ap.add_argument("--matches-root", default="Matches", help="Matches root folder (default: Matches)")
    ap.add_argument("--outdir", default=None, help="Output directory (default: same dir as deploy file)")
    ap.add_argument("--leagues", default=None, help="Optional comma-separated league whitelist")
    ap.add_argument(
        "--min-n-src-col-print",
        type=int,
        default=15,
        help="Minimum n to print MARKET_SRC_COL lines to console (default: 15)",
    )
    args = ap.parse_args()

    deploy_path = Path(args.deploy_csv)
    if not deploy_path.exists():
        raise SystemExit(f"Deploy CSV not found: {deploy_path}")

    matches_root = Path(args.matches_root)
    outdir = Path(args.outdir) if args.outdir else deploy_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(deploy_path)

    for c in ["league", "fixture_key", "market", "bookie_pick"]:
        if c not in df.columns:
            raise SystemExit(f"Deploy CSV missing required column: {c}")

    # bookie_od is optional (WTN doesn’t have real odds)
    if "bookie_od" not in df.columns:
        df["bookie_od"] = np.nan

    # bookie_od_source indicates whether odds are real bookmaker odds or model-derived fair/proxy odds.
    # filter_deploy_by_rulebook.py should populate this, but we make it robust here.
    if "bookie_od_source" not in df.columns:
        df["bookie_od_source"] = pd.NA

    # bookie_od_column indicates which input odds column was used per-row (auditable provenance)
    # Populated upstream by filter_deploy_by_rulebook.py; we keep this robust here.
    if "bookie_od_column" not in df.columns:
        df["bookie_od_column"] = pd.NA

    df["league"] = df["league"].astype(str)
    df["fixture_key"] = df["fixture_key"].astype(str)
    df["market"] = df["market"].astype(str).str.lower().str.strip()
    df["bookie_pick"] = df["bookie_pick"].astype(str).str.upper().str.strip()
    df["bookie_od"] = _to_num(df["bookie_od"])

    # Normalize source field (if present) but don't overwrite meaningful values.
    df["bookie_od_source"] = df["bookie_od_source"].astype("string")
    df["bookie_od_source"] = df["bookie_od_source"].str.lower().str.strip()
    df.loc[df["bookie_od_source"].isin(["", "nan", "none", "<na>"]), "bookie_od_source"] = pd.NA

    # Normalize bookie_od_column (if present) to a stable, comparable form
    df["bookie_od_column"] = df["bookie_od_column"].astype("string")
    df["bookie_od_column"] = df["bookie_od_column"].str.strip()
    df.loc[df["bookie_od_column"].isin(["", "nan", "none", "<na>"]), "bookie_od_column"] = pd.NA

    # --- Fill missing bookie_od for FTR from market-specific odds columns (if present) ---
    # Many deploy CSVs carry FTR prices in columns like od_home/od_draw/od_away (or odds_ft_*).
    # If bookie_od is missing for FTR, infer it from those columns based on the bookie_pick.
    def _first_present(cols):
        for c in cols:
            if c in df.columns:
                return c
        return None

    ftr_home_col = _first_present(["od_home", "odds_ft_home", "odds_ft_home_team_win", "odds_ft_home_win"])
    ftr_draw_col = _first_present(["od_draw", "odds_ft_draw"])
    ftr_away_col = _first_present(["od_away", "odds_ft_away", "odds_ft_away_team_win", "odds_ft_away_win"])

    if ftr_home_col and ftr_draw_col and ftr_away_col:
        # Coerce to numeric (safe if already numeric)
        df[ftr_home_col] = _to_num(df[ftr_home_col])
        df[ftr_draw_col] = _to_num(df[ftr_draw_col])
        df[ftr_away_col] = _to_num(df[ftr_away_col])

        m_ftr = df["market"].eq("ftr")
        m_missing = df["bookie_od"].isna()
        m = m_ftr & m_missing
        if m.any():
            picks = df.loc[m, "bookie_pick"].astype(str).str.upper().str.strip()

            # Default NaN; fill by pick
            filled = pd.Series(np.nan, index=picks.index, dtype="float")
            m_home = picks.eq("HOME")
            m_draw = picks.eq("DRAW")
            m_away = picks.eq("AWAY")

            if m_home.any():
                filled.loc[m_home] = df.loc[filled.loc[m_home].index, ftr_home_col].to_numpy(dtype=float)
            if m_draw.any():
                filled.loc[m_draw] = df.loc[filled.loc[m_draw].index, ftr_draw_col].to_numpy(dtype=float)
            if m_away.any():
                filled.loc[m_away] = df.loc[filled.loc[m_away].index, ftr_away_col].to_numpy(dtype=float)

            df.loc[filled.index, "bookie_od"] = filled
            # These columns are bookmaker-provided odds.
            df.loc[filled.index[pd.notna(filled)], "bookie_od_source"] = df.loc[
                filled.index[pd.notna(filled)], "bookie_od_source"
            ].fillna("bookmaker")

    # --- Fill missing bookie_od for OU25 from market-specific odds columns (if present) ---
    # Deploy CSVs often carry OU25 prices in od_over/od_under (or odds_ft_over25/odds_ft_under25).
    # If bookie_od is missing for OU25, infer it from those columns based on the bookie_pick.
    ou_over_col = _first_present(["od_over", "odds_ft_over25", "odds_ft_over25_rm", "odds_over25"])
    ou_under_col = _first_present(["od_under", "odds_ft_under25", "odds_ft_under25_rm", "odds_under25"])

    ou25_filled_n = 0
    if ou_over_col and ou_under_col:
        df[ou_over_col] = _to_num(df[ou_over_col])
        df[ou_under_col] = _to_num(df[ou_under_col])

        m_ou = df["market"].eq("ou25")
        m_missing = df["bookie_od"].isna()
        m = m_ou & m_missing
        if m.any():
            picks = df.loc[m, "bookie_pick"].astype(str).str.upper().str.strip()

            filled = pd.Series(np.nan, index=picks.index, dtype="float")
            m_over = picks.eq("OVER25")
            m_under = picks.eq("UNDER25")

            if m_over.any():
                idx_over = filled.loc[m_over].index
                filled.loc[idx_over] = df.loc[idx_over, ou_over_col].to_numpy(dtype=float)
            if m_under.any():
                idx_under = filled.loc[m_under].index
                filled.loc[idx_under] = df.loc[idx_under, ou_under_col].to_numpy(dtype=float)

            df.loc[filled.index, "bookie_od"] = filled
            # These columns are bookmaker-provided odds.
            df.loc[filled.index[pd.notna(filled)], "bookie_od_source"] = df.loc[
                filled.index[pd.notna(filled)], "bookie_od_source"
            ].fillna("bookmaker")
            ou25_filled_n = int(pd.notna(filled).sum())

    # --- Fill missing bookie_od for BTTS from market-specific odds columns (if present) ---
    # Deploy CSVs often carry BTTS prices in od_yes/od_no (or odds_btts_yes/odds_btts_no).
    # If bookie_od is missing for BTTS, infer it from those columns based on the bookie_pick.
    btts_yes_col = _first_present(["od_yes", "odds_btts_yes", "odds_btts_yes_rm"])
    btts_no_col = _first_present(["od_no", "odds_btts_no", "odds_btts_no_rm"])

    btts_filled_n = 0
    if btts_yes_col and btts_no_col:
        df[btts_yes_col] = _to_num(df[btts_yes_col])
        df[btts_no_col] = _to_num(df[btts_no_col])

        m_btts = df["market"].eq("btts")
        m_missing = df["bookie_od"].isna()
        m = m_btts & m_missing
        if m.any():
            picks = df.loc[m, "bookie_pick"].astype(str).str.upper().str.strip()

            filled = pd.Series(np.nan, index=picks.index, dtype="float")
            m_yes = picks.eq("YES")
            m_no = picks.eq("NO")

            if m_yes.any():
                idx_yes = filled.loc[m_yes].index
                filled.loc[idx_yes] = df.loc[idx_yes, btts_yes_col].to_numpy(dtype=float)
            if m_no.any():
                idx_no = filled.loc[m_no].index
                filled.loc[idx_no] = df.loc[idx_no, btts_no_col].to_numpy(dtype=float)

            df.loc[filled.index, "bookie_od"] = filled
            # These columns are bookmaker-provided odds.
            df.loc[filled.index[pd.notna(filled)], "bookie_od_source"] = df.loc[
                filled.index[pd.notna(filled)], "bookie_od_source"
            ].fillna("bookmaker")
            btts_filled_n = int(pd.notna(filled).sum())

    # --- Quick sanity prints: are these odds columns present and populated? ---
    # Keep this lightweight; it helps confirm why ROI might still be NaN for a market.
    def _nn_count(col: str) -> int:
        if not col or col not in df.columns:
            return 0
        return int(pd.to_numeric(df[col], errors="coerce").notna().sum())

    def _nn_count_market(col: str, mk: str) -> int:
        if not col or col not in df.columns:
            return 0
        sub = df[df["market"].eq(mk)]
        if sub.empty:
            return 0
        return int(pd.to_numeric(sub[col], errors="coerce").notna().sum())

    # FTR columns already computed above; reuse their names if present.
    if ftr_home_col and ftr_draw_col and ftr_away_col:
        print(
            f"ODDS_COLS FTR: home={ftr_home_col}({ _nn_count_market(ftr_home_col, 'ftr') }/{ int((df['market']=='ftr').sum()) }) "
            f"draw={ftr_draw_col}({ _nn_count_market(ftr_draw_col, 'ftr') }/{ int((df['market']=='ftr').sum()) }) "
            f"away={ftr_away_col}({ _nn_count_market(ftr_away_col, 'ftr') }/{ int((df['market']=='ftr').sum()) })"
        )

    if ou_over_col and ou_under_col:
        print(
            f"ODDS_COLS OU25: over={ou_over_col}({ _nn_count_market(ou_over_col, 'ou25') }/{ int((df['market']=='ou25').sum()) }) "
            f"under={ou_under_col}({ _nn_count_market(ou_under_col, 'ou25') }/{ int((df['market']=='ou25').sum()) }) "
            f"filled_bookie_od={ou25_filled_n}"
        )
    else:
        print("ODDS_COLS OU25: missing (no usable over/under columns found)")

    if btts_yes_col and btts_no_col:
        print(
            f"ODDS_COLS BTTS: yes={btts_yes_col}({ _nn_count_market(btts_yes_col, 'btts') }/{ int((df['market']=='btts').sum()) }) "
            f"no={btts_no_col}({ _nn_count_market(btts_no_col, 'btts') }/{ int((df['market']=='btts').sum()) }) "
            f"filled_bookie_od={btts_filled_n}"
        )
    else:
        print("ODDS_COLS BTTS: missing (no usable yes/no columns found)")

    # If a row has odds but no source, infer a sensible default.
    # TG markets typically use model-derived fair/proxy odds unless explicitly provided.
    m_has_od = df["bookie_od"].notna()
    m_src_missing = df["bookie_od_source"].isna()
    if bool((m_has_od & m_src_missing).any()):
        m_tg = df["market"].isin(["tg15", "tg25"])
        df.loc[m_has_od & m_src_missing & m_tg, "bookie_od_source"] = "model_fair"
        df.loc[m_has_od & m_src_missing & ~m_tg, "bookie_od_source"] = "bookmaker"

    # Normalize any remaining NA sources (useful for WTN or missing odds)
    df["bookie_od_source"] = df["bookie_od_source"].fillna("unknown")
    if args.leagues:
        allow = {x.strip() for x in args.leagues.split(",") if x.strip()}
        df = df[df["league"].isin(allow)].copy()

    joined_parts: List[pd.DataFrame] = []
    missing: List[str] = []

    for lg in sorted(df["league"].dropna().unique()):
        m = _load_matches(matches_root, lg)
        if m is None:
            missing.append(lg)
            continue
        sub = df[df["league"] == lg].copy()
        sub = sub.merge(m, on="fixture_key", how="left")
        joined_parts.append(sub)

    if not joined_parts:
        raise SystemExit("No leagues could be joined to Matches/<League>/fd_odds_enriched.csv (missing file or fixture_key).")

    out = pd.concat(joined_parts, ignore_index=True)
    # Ensure odds-source column exists for downstream grouping
    if "bookie_od_source" not in out.columns:
        out["bookie_od_source"] = "unknown"
    # Ensure odds-column provenance exists for downstream grouping
    if "bookie_od_column" not in out.columns:
        out["bookie_od_column"] = "unknown"

    out["correct"] = _compute_correct(out)

    # Add outcome hit columns for OU25 UNDER and BTTS NO audits (independent of pick)
    mk = out["market"].astype(str).str.lower().str.strip()
    hg = _to_num(out.get("home_team_goal_count"))
    ag = _to_num(out.get("away_team_goal_count"))
    known = hg.notna() & ag.notna()
    tot = hg + ag
    btts_yes = (hg >= 1) & (ag >= 1)

    out["under25_hit"] = np.where(known & mk.eq("ou25"), (tot <= 2).astype(float), np.nan)
    out["btts_no_hit"] = np.where(known & mk.eq("btts"), (~btts_yes).astype(float), np.nan)

    def _product(market: str, pick_: str) -> str:
        market = (market or "").lower()
        pick_ = (pick_ or "").upper()
        if market == "ou25":
            return f"OU25_{pick_}"
        if market == "btts":
            return f"BTTS_{pick_}"
        if market == "ftr":
            return f"FTR_{pick_}"
        if market == "fts":
            return f"FTS_{pick_}"
        if market == "ge2":
            return f"GE2_{pick_}"
        if market == "ge3":
            return f"GE3_{pick_}"
        if market == "wtn":
            return f"WTN_{pick_}"
        if market == "tg15":
            return f"TG15_{pick_}"
        if market == "tg25":
            return f"TG25_{pick_}"
        return f"{market.upper()}_{pick_}"

    out["product"] = [_product(m, p) for m, p in zip(out["market"].astype(str), out["bookie_pick"].astype(str))]

    scored_hit = out.dropna(subset=["correct"]).copy()
    scored_roi = scored_hit.dropna(subset=["bookie_od"]).copy()
    unscored = out[out["correct"].isna()].copy()

    # Option A: keep investor-facing backtest clean.
    # - __BACKTEST.csv: scored-only rows (no NaN correct)
    # - __BACKTEST_UNSCORED.csv: excluded rows (kept for forensic/diagnostics)
    out_csv = outdir / f"{deploy_path.stem}__BACKTEST.csv"
    out_csv_unscored = outdir / f"{deploy_path.stem}__BACKTEST_UNSCORED.csv"
    scored_hit.to_csv(out_csv, index=False)
    unscored.to_csv(out_csv_unscored, index=False)

    # Back-compat: keep writing a scored file (now identical to __BACKTEST.csv)
    out_scored_csv = outdir / f"{deploy_path.stem}__BACKTEST_SCORED.csv"
    scored_hit.to_csv(out_scored_csv, index=False)

    def _summ_pair(hit_df: pd.DataFrame, roi_df: pd.DataFrame) -> Tuple[int, float, float, float, int]:
        n_hit = len(hit_df)
        hit = float(hit_df["correct"].mean()) if n_hit else float("nan")
        n_roi = len(roi_df)
        if n_roi:
            avg_od = float(roi_df["bookie_od"].mean())
            roi = float((roi_df["correct"] * roi_df["bookie_od"] - 1).mean())
        else:
            avg_od = float("nan")
            roi = float("nan")
        return n_hit, hit, roi, avg_od, n_roi

    rows: List[Dict[str, object]] = []

    n_hit, hit, roi, avg_od, n_roi = _summ_pair(scored_hit, scored_roi)
    rows.append({"group": "ALL", "n": n_hit, "hit": hit, "roi": roi, "avg_od": avg_od, "n_roi": n_roi})

    for mk_val, g_hit in scored_hit.groupby("market", dropna=False):
        g_roi = scored_roi[scored_roi["market"] == mk_val]
        n_hit, hit, roi, avg_od, n_roi = _summ_pair(g_hit, g_roi)
        rows.append({"group": f"MARKET::{mk_val}", "n": n_hit, "hit": hit, "roi": roi, "avg_od": avg_od, "n_roi": n_roi})

    # --- Market + pick splits ---
    # Useful for OU25/BTTS where pick-level performance tells a clearer story.
    for (mk_val, pick_val), g_hit in scored_hit.groupby(["market", "bookie_pick"], dropna=False):
        g_roi = scored_roi[(scored_roi["market"] == mk_val) & (scored_roi["bookie_pick"] == pick_val)]
        n_hit, hit, roi, avg_od, n_roi = _summ_pair(g_hit, g_roi)
        rows.append(
            {
                "group": f"MARKET_PICK::{mk_val}::{pick_val}",
                "n": n_hit,
                "hit": hit,
                "roi": roi,
                "avg_od": avg_od,
                "n_roi": n_roi,
            }
        )

    # --- Market + source splits (normalized) ---
    if "bookie_od_source" in scored_hit.columns:
        sh = scored_hit.copy()
        sr = scored_roi.copy()

        sh["bookie_od_source_norm"] = (
            sh["bookie_od_source"].astype(str).str.strip().str.lower()
            .replace({
                "bookie": "bookmaker",
                "real": "bookmaker",
                "bk": "bookmaker",
                "": "unknown",
                "nan": "unknown",
                "none": "unknown",
                "<na>": "unknown",
                "<NA>": "unknown",
            })
        )
        sr["bookie_od_source_norm"] = (
            sr["bookie_od_source"].astype(str).str.strip().str.lower()
            .replace({
                "bookie": "bookmaker",
                "real": "bookmaker",
                "bk": "bookmaker",
                "": "unknown",
                "nan": "unknown",
                "none": "unknown",
                "<na>": "unknown",
                "<NA>": "unknown",
            })
        )

        for (mk_val, src_val), g_hit in sh.groupby(["market", "bookie_od_source_norm"], dropna=False):
            g_roi = sr[(sr["market"] == mk_val) & (sr["bookie_od_source_norm"] == src_val)]
            n_hit, hit, roi, avg_od, n_roi = _summ_pair(g_hit, g_roi)
            rows.append(
                {
                    "group": f"MARKET_SRC::{mk_val}::{src_val}",
                    "n": n_hit,
                    "hit": hit,
                    "roi": roi,
                    "avg_od": avg_od,
                    "n_roi": n_roi,
                }
            )

    # --- Market + source + odds-column splits (normalized) ---
    if ("bookie_od_source" in scored_hit.columns) and ("bookie_od_column" in scored_hit.columns):
        shc = scored_hit.copy()
        src = scored_roi.copy()

        shc["bookie_od_source_norm"] = (
            shc["bookie_od_source"].astype(str).str.strip().str.lower()
            .replace({
                "bookie": "bookmaker",
                "real": "bookmaker",
                "bk": "bookmaker",
                "": "unknown",
                "nan": "unknown",
                "none": "unknown",
                "<na>": "unknown",
                "<NA>": "unknown",
            })
        )
        src["bookie_od_source_norm"] = (
            src["bookie_od_source"].astype(str).str.strip().str.lower()
            .replace({
                "bookie": "bookmaker",
                "real": "bookmaker",
                "bk": "bookmaker",
                "": "unknown",
                "nan": "unknown",
                "none": "unknown",
                "<na>": "unknown",
                "<NA>": "unknown",
            })
        )

        shc["bookie_od_column_norm"] = (
            shc["bookie_od_column"].astype(str).str.strip()
            .replace({"": "unknown", "nan": "unknown", "none": "unknown", "<na>": "unknown", "<NA>": "unknown"})
        )
        src["bookie_od_column_norm"] = (
            src["bookie_od_column"].astype(str).str.strip()
            .replace({"": "unknown", "nan": "unknown", "none": "unknown", "<na>": "unknown", "<NA>": "unknown"})
        )

        for (mk_val, src_val, col_val), g_hit in shc.groupby(
            ["market", "bookie_od_source_norm", "bookie_od_column_norm"], dropna=False
        ):
            g_roi = src[
                (src["market"] == mk_val)
                & (src["bookie_od_source_norm"] == src_val)
                & (src["bookie_od_column_norm"] == col_val)
            ]
            n_hit, hit, roi, avg_od, n_roi = _summ_pair(g_hit, g_roi)
            rows.append(
                {
                    "group": f"MARKET_SRC_COL::{mk_val}::{src_val}::{col_val}",
                    "n": n_hit,
                    "hit": hit,
                    "roi": roi,
                    "avg_od": avg_od,
                    "n_roi": n_roi,
                }
            )

    for prod_val, g_hit in scored_hit.groupby("product", dropna=False):
        g_roi = scored_roi[scored_roi["product"] == prod_val]
        n_hit, hit, roi, avg_od, n_roi = _summ_pair(g_hit, g_roi)
        rows.append({"group": f"PRODUCT::{prod_val}", "n": n_hit, "hit": hit, "roi": roi, "avg_od": avg_od, "n_roi": n_roi})
        
    # --- Product + source splits (normalized) ---
    if "bookie_od_source" in scored_hit.columns:
        # Reuse normalized sources if we already created them above
        if "bookie_od_source_norm" not in scored_hit.columns:
            sh = scored_hit.copy()
            sh["bookie_od_source_norm"] = (
                sh["bookie_od_source"].astype(str).str.strip().str.lower()
                .replace({
                    "bookie": "bookmaker",
                    "real": "bookmaker",
                    "bk": "bookmaker",
                    "": "unknown",
                    "nan": "unknown",
                    "none": "unknown",
                    "<na>": "unknown",
                    "<NA>": "unknown",
                })
            )
            sr = scored_roi.copy()
            sr["bookie_od_source_norm"] = (
                sr["bookie_od_source"].astype(str).str.strip().str.lower()
                .replace({
                    "bookie": "bookmaker",
                    "real": "bookmaker",
                    "bk": "bookmaker",
                    "": "unknown",
                    "nan": "unknown",
                    "none": "unknown",
                    "<na>": "unknown",
                    "<NA>": "unknown",
                })
            )
        else:
            # If a caller later refactors and leaves this column on the frame, keep behaviour stable
            sh = scored_hit
            sr = scored_roi

        for (prod_val, src_val), g_hit in sh.groupby(["product", "bookie_od_source_norm"], dropna=False):
            g_roi = sr[(sr["product"] == prod_val) & (sr["bookie_od_source_norm"] == src_val)]
            n_hit, hit, roi, avg_od, n_roi = _summ_pair(g_hit, g_roi)
            rows.append(
                {
                    "group": f"PRODUCT_SRC::{prod_val}::{src_val}",
                    "n": n_hit,
                    "hit": hit,
                    "roi": roi,
                    "avg_od": avg_od,
                    "n_roi": n_roi,
                }
            )

    # --- Product + source + odds-column splits (normalized) ---
    if ("bookie_od_source" in scored_hit.columns) and ("bookie_od_column" in scored_hit.columns):
        shp = scored_hit.copy()
        srp = scored_roi.copy()

        shp["bookie_od_source_norm"] = (
            shp["bookie_od_source"].astype(str).str.strip().str.lower()
            .replace({
                "bookie": "bookmaker",
                "real": "bookmaker",
                "bk": "bookmaker",
                "": "unknown",
                "nan": "unknown",
                "none": "unknown",
                "<na>": "unknown",
                "<NA>": "unknown",
            })
        )
        srp["bookie_od_source_norm"] = (
            srp["bookie_od_source"].astype(str).str.strip().str.lower()
            .replace({
                "bookie": "bookmaker",
                "real": "bookmaker",
                "bk": "bookmaker",
                "": "unknown",
                "nan": "unknown",
                "none": "unknown",
                "<na>": "unknown",
                "<NA>": "unknown",
            })
        )

        shp["bookie_od_column_norm"] = (
            shp["bookie_od_column"].astype(str).str.strip()
            .replace({"": "unknown", "nan": "unknown", "none": "unknown", "<na>": "unknown", "<NA>": "unknown"})
        )
        srp["bookie_od_column_norm"] = (
            srp["bookie_od_column"].astype(str).str.strip()
            .replace({"": "unknown", "nan": "unknown", "none": "unknown", "<na>": "unknown", "<NA>": "unknown"})
        )

        for (prod_val, src_val, col_val), g_hit in shp.groupby(
            ["product", "bookie_od_source_norm", "bookie_od_column_norm"], dropna=False
        ):
            g_roi = srp[
                (srp["product"] == prod_val)
                & (srp["bookie_od_source_norm"] == src_val)
                & (srp["bookie_od_column_norm"] == col_val)
            ]
            n_hit, hit, roi, avg_od, n_roi = _summ_pair(g_hit, g_roi)
            rows.append(
                {
                    "group": f"PRODUCT_SRC_COL::{prod_val}::{src_val}::{col_val}",
                    "n": n_hit,
                    "hit": hit,
                    "roi": roi,
                    "avg_od": avg_od,
                    "n_roi": n_roi,
                }
            )

    for lg_val, g_hit in scored_hit.groupby("league", dropna=False):
        g_roi = scored_roi[scored_roi["league"] == lg_val]
        n_hit, hit, roi, avg_od, n_roi = _summ_pair(g_hit, g_roi)
        rows.append({"group": f"LEAGUE::{lg_val}", "n": n_hit, "hit": hit, "roi": roi, "avg_od": avg_od, "n_roi": n_roi})

    # --- League + source splits (normalized) ---
    if "bookie_od_source" in scored_hit.columns:
        # Reuse normalized sources if we already created them above
        if "bookie_od_source_norm" not in scored_hit.columns:
            sh = scored_hit.copy()
            sh["bookie_od_source_norm"] = (
                sh["bookie_od_source"].astype(str).str.strip().str.lower()
                .replace({
                    "bookie": "bookmaker",
                    "real": "bookmaker",
                    "bk": "bookmaker",
                    "": "unknown",
                    "nan": "unknown",
                    "none": "unknown",
                    "<na>": "unknown",
                    "<NA>": "unknown",
                })
            )
            sr = scored_roi.copy()
            sr["bookie_od_source_norm"] = (
                sr["bookie_od_source"].astype(str).str.strip().str.lower()
                .replace({
                    "bookie": "bookmaker",
                    "real": "bookmaker",
                    "bk": "bookmaker",
                    "": "unknown",
                    "nan": "unknown",
                    "none": "unknown",
                    "<na>": "unknown",
                    "<NA>": "unknown",
                })
            )
        else:
            sh = scored_hit
            sr = scored_roi

        for (lg_val, src_val), g_hit in sh.groupby(["league", "bookie_od_source_norm"], dropna=False):
            g_roi = sr[(sr["league"] == lg_val) & (sr["bookie_od_source_norm"] == src_val)]
            n_hit, hit, roi, avg_od, n_roi = _summ_pair(g_hit, g_roi)
            rows.append(
                {
                    "group": f"LEAGUE_SRC::{lg_val}::{src_val}",
                    "n": n_hit,
                    "hit": hit,
                    "roi": roi,
                    "avg_od": avg_od,
                    "n_roi": n_roi,
                }
            )

    # --- League + source + odds-column splits (normalized) ---
    if ("bookie_od_source" in scored_hit.columns) and ("bookie_od_column" in scored_hit.columns):
        shl = scored_hit.copy()
        srl = scored_roi.copy()

        shl["bookie_od_source_norm"] = (
            shl["bookie_od_source"].astype(str).str.strip().str.lower()
            .replace({
                "bookie": "bookmaker",
                "real": "bookmaker",
                "bk": "bookmaker",
                "": "unknown",
                "nan": "unknown",
                "none": "unknown",
                "<na>": "unknown",
                "<NA>": "unknown",
            })
        )
        srl["bookie_od_source_norm"] = (
            srl["bookie_od_source"].astype(str).str.strip().str.lower()
            .replace({
                "bookie": "bookmaker",
                "real": "bookmaker",
                "bk": "bookmaker",
                "": "unknown",
                "nan": "unknown",
                "none": "unknown",
                "<na>": "unknown",
                "<NA>": "unknown",
            })
        )

        shl["bookie_od_column_norm"] = (
            shl["bookie_od_column"].astype(str).str.strip()
            .replace({"": "unknown", "nan": "unknown", "none": "unknown", "<na>": "unknown", "<NA>": "unknown"})
        )
        srl["bookie_od_column_norm"] = (
            srl["bookie_od_column"].astype(str).str.strip()
            .replace({"": "unknown", "nan": "unknown", "none": "unknown", "<na>": "unknown", "<NA>": "unknown"})
        )

        for (lg_val, src_val, col_val), g_hit in shl.groupby(
            ["league", "bookie_od_source_norm", "bookie_od_column_norm"], dropna=False
        ):
            g_roi = srl[
                (srl["league"] == lg_val)
                & (srl["bookie_od_source_norm"] == src_val)
                & (srl["bookie_od_column_norm"] == col_val)
            ]
            n_hit, hit, roi, avg_od, n_roi = _summ_pair(g_hit, g_roi)
            rows.append(
                {
                    "group": f"LEAGUE_SRC_COL::{lg_val}::{src_val}::{col_val}",
                    "n": n_hit,
                    "hit": hit,
                    "roi": roi,
                    "avg_od": avg_od,
                    "n_roi": n_roi,
                }
            )

    summary = pd.DataFrame(rows).sort_values(["group"])
    out_sum = outdir / f"{deploy_path.stem}__BACKTEST_SUMMARY.csv"
    summary.to_csv(out_sum, index=False)

    print("DEPLOY:", str(deploy_path))
    if missing:
        print("WARNING: missing leagues:", ", ".join(sorted(missing)))
    print("SCORED(hit):", len(scored_hit), "/", len(out))
    print("SCORED(roi):", len(scored_roi), "/", len(out))

    def _fmt(r):
        return f"n={int(r['n']):4d} hit={float(r['hit']):.3f} ROI={float(r['roi']):+.3f} avg_od={float(r['avg_od']):.3f}"

    print("ALL:", _fmt(summary[summary["group"] == "ALL"].iloc[0]))
    for k in ["MARKET::ftr", "MARKET::ou25", "MARKET::btts", "MARKET::fts", "MARKET::ge2", "MARKET::ge3", "MARKET::wtn", "MARKET::tg15", "MARKET::tg25"]:
        if (summary["group"] == k).any():
            print(k + ":", _fmt(summary[summary["group"] == k].iloc[0]))

        # Also print odds-source splits when present
        for src in ["bookmaker", "model_fair", "unknown"]:
            ks = k.replace("MARKET::", "MARKET_SRC::") + f"::{src}"
            if (summary["group"] == ks).any():
                print("  " + ks + ":", _fmt(summary[summary["group"] == ks].iloc[0]))

        # Also print pick-level splits for OU25/BTTS (and only when present)
        # Apply the same small-sample guardrail to pick splits (deck-safe).
        mk_name = k.replace("MARKET::", "")
        min_n = int(getattr(args, "min_n_src_col_print", 15) or 0)

        if mk_name in ("ou25", "btts"):
            pick_prefix = f"MARKET_PICK::{mk_name}::"
            if (summary["group"].astype(str).str.startswith(pick_prefix)).any():
                subp = summary[summary["group"].astype(str).str.startswith(pick_prefix)].copy()
                skipped_pick = 0
                printed_pick = 0
                hidden_pick_labels: List[str] = []
                for _, r in subp.sort_values(["group"]).iterrows():
                    try:
                        n_val = int(r.get("n", 0))
                    except Exception:
                        n_val = 0
                    if n_val < min_n:
                        skipped_pick += 1
                        # Extract label after MARKET_PICK::<mk>::
                        try:
                            label = str(r["group"]).split("::")[-1]
                        except Exception:
                            label = ""
                        if label:
                            hidden_pick_labels.append(label)
                        continue
                    printed_pick += 1
                    print("  " + str(r["group"]) + ":", _fmt(r))

                if skipped_pick:
                    # Show which pick labels were hidden (names only; no metrics)
                    # Example: (...skipped 1 small MARKET_PICK groups; n<15: NO)
                    hidden_labels = ", ".join(sorted({str(x) for x in hidden_pick_labels if str(x)}))
                    suffix = f": {hidden_labels}" if hidden_labels else ""
                    print(f"  (...skipped {skipped_pick} small MARKET_PICK groups; n<{min_n} [--min-n-src-col-print]{suffix})")

        # Also print odds-source + odds-column splits when present
        # Print a one-liner if any groups were hidden by the guardrail.
        prefix = k.replace("MARKET::", "MARKET_SRC_COL::") + "::"
        if (summary["group"].astype(str).str.startswith(prefix)).any():
            sub = summary[summary["group"].astype(str).str.startswith(prefix)].copy()
            skipped_src_col = 0
            printed_src_col = 0
            hidden_src_col_names: List[str] = []
            for _, r in sub.sort_values(["group"]).iterrows():
                try:
                    n_val = int(r.get("n", 0))
                except Exception:
                    n_val = 0
                if n_val < min_n:
                    skipped_src_col += 1
                    # Extract the last segment (column name) from MARKET_SRC_COL::<mk>::<src>::<col>
                    try:
                        colname = str(r["group"]).split("::")[-1]
                    except Exception:
                        colname = ""
                    if colname:
                        hidden_src_col_names.append(colname)
                    continue
                printed_src_col += 1
                print("  " + str(r["group"]) + ":", _fmt(r))

            if skipped_src_col:
                # Show which odds-column names were hidden (names only; no metrics)
                # Example: (...skipped 1 small MARKET_SRC_COL groups; n<15: od_no)
                hidden_cols = ", ".join(sorted({str(x) for x in hidden_src_col_names if str(x)}))
                suffix = f": {hidden_cols}" if hidden_cols else ""
                print(f"  (...skipped {skipped_src_col} small MARKET_SRC_COL groups; n<{min_n} [--min-n-src-col-print]{suffix})")

    print("WROTE:", str(out_csv))
    print("WROTE:", str(out_csv_unscored))
    print("WROTE:", str(out_scored_csv))
    print("WROTE:", str(out_sum))


if __name__ == "__main__":
    main()
