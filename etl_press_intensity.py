"""
etl_press_intensity.py
~~~~~~~~~~~~~~~~~~~~~~

Player → team totals ETL and a match‑level press‑intensity (PPDA‑like) proxy.

Exposes:
    • ensure_player_team_totals_on_disk(match_dir)
    • compute_press_intensity(match_df) -> DataFrame[match_id, home_press_intensity, away_press_intensity]
    • _attach_press_intensity(csv_path)  # in‑place update of a single match CSV
    • ensure_press_intensity_on_disk(match_dir, force=False, max_files=None)
    • CLI: python etl_press_intensity.py --match-dir Matches/<League>
"""
from __future__ import annotations

import os
import glob
import argparse
import numpy as np
import pandas as pd

# Cache location (ModelStore)
try:
    from constants import MODEL_DIR
except Exception:
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "ModelStore")

# Promote season baseline into a per-match proxy if no match-level press_intensity exists
# This creates home_press_intensity/away_press_intensity from the baseline columns when possible.
def _promote_baseline_to_proxy(match_dir: str, *, overwrite: bool = False) -> tuple[int, int]:
    pattern = os.path.join(match_dir, "**", "*.csv")
    paths = sorted(glob.glob(pattern, recursive=True))
    def _skip(p: str) -> bool:
        name = os.path.basename(p).lower()
        return (
            any(k in name for k in ("upcoming", "fixture", "fixtures", "prediction", "predictions", "report"))
            or any(k in p.lower() for k in ("predictions_output", "modelstore"))
        )
    paths = [p for p in paths if not _skip(p)]
    if not paths:
        return (0, 0)

    promoted = 0
    for mp in paths:
        try:
            df = pd.read_csv(mp, low_memory=False)
        except Exception:
            continue
        # if already have real press-intensity, skip unless overwrite=True
        if {"home_press_intensity", "away_press_intensity"}.issubset(df.columns) and not overwrite:
            continue
        # promote only if baseline available and NOT already promoted
        have_bl = {"home_press_baseline", "away_press_baseline"}.issubset(df.columns)
        if not have_bl:
            continue
        # write new columns (copy baseline -> intensity proxy)
        df["home_press_intensity"] = df.get("home_press_baseline")
        df["away_press_intensity"] = df.get("away_press_baseline")
        try:
            df.to_csv(mp, index=False)
            promoted += 1
        except Exception:
            pass
    return (promoted, len(paths))

# Robust free-form date parser used for date_GMT fallbacks
def _parse_date_series(series: pd.Series) -> pd.Series:
    """Try several common formats and pick the parse with most non‑NaNs.
    Falls back to pandas' generic parser with errors='coerce'."""
    if series is None:
        return pd.Series(pd.NaT, index=[])
    fmts = ("%b %d %Y - %I:%M%p", "%Y-%m-%d %H:%M", "%Y-%m-%d")
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
# --- Season-baseline proxy helpers (fallback when no per-match keys) ---

def _season_prev(season: str) -> str:
    """
    '2023/2024' -> '2022/2023'. If not parseable, return input.
    """
    try:
        a, b = season.split("/")
        return f"{int(a)-1}/{int(b)-1}"
    except Exception:
        return season

# --- Helper: normalize season string ---
def _normalize_season_string(season: str) -> str:
    """Normalize season labels to canonical 'YYYY/YYYY' form.
    Accepts '2018', '2018/2019', '2018-2019', '2018–2019', '2018 to 2019'.
    Single-year like '2018' becomes '2018/2018'.
    Unknown strings are returned unchanged.
    """
    import re
    if season is None:
        return ""
    s = str(season).strip()
    if not s:
        return s
    # two-year patterns
    m = re.search(r"^((?:19|20)\d{2})\s*(?:[/_\-–]|\s+to\s+)\s*((?:19|20)\d{2})$", s, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    # single-year pattern
    m = re.search(r"^((?:19|20)\d{2})$", s)
    if m:
        y = m.group(1)
        return f"{y}/{y}"
    return s

def _normalize_team(name: str) -> str:
    """Return a canonical key for team names across seasons/leagues.
    Steps:
      1) lowercase, strip accents, remove punctuation → spaces
      2) map well-known aliases (MLS-heavy)
      3) drop generic tokens (fc/sc/cf/club)
      4) sort remaining tokens alphabetically so order differences don't matter
    """
    import unicodedata

    if name is None:
        return ""

    # 1) lowercase + strip accents
    s = str(name).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))  # remove diacritics

    # remove punctuation / noise → space
    for ch in (".", ",", "'", "-", "(", ")", "/", "\\", "&"):
        s = s.replace(ch, " ")
    while "  " in s:
        s = s.replace("  ", " ")

    # 2) alias map (both directions where likely)
    # canonical forms chosen to be short and stable
    aliases = {
        # LAFC
        "lafc": "los angeles fc",
        "la fc": "los angeles fc",
        "los angeles football club": "los angeles fc",
        # NYCFC
        "nycfc": "new york city",
        "new york city fc": "new york city",
        # Red Bulls
        "ny red bulls": "new york rb",
        "nyrb": "new york rb",
        "rbny": "new york rb",
        "new york red bulls": "new york rb",
        # DCU
        "d c united": "dc united",
        "d.c. united": "dc united",
        # San Jose
        "san jose earthquakes": "sj earthquakes",
        "san jose": "sj earthquakes",
        # Montréal / Montreal
        "cf montreal": "montreal",
        "cf montreal impact": "montreal",
        "montreal impact": "montreal",
        "montreal": "montreal",
        # Sporting
        "sporting kansas city": "sporting kc",
        # Orlando
        "orlando city sc": "orlando city",
        # Chicago / Houston / Seattle / Vancouver / Toronto / Minnesota (FC suffixes)
        "chicago fire fc": "chicago fire",
        "houston dynamo fc": "houston dynamo",
        "seattle sounders fc": "seattle sounders",
        "vancouver whitecaps fc": "vancouver whitecaps",
        "toronto fc": "toronto",
        "minnesota united fc": "minnesota united",
        # St. Louis City
        "st louis city": "st louis city",
        "st. louis city": "st louis city",
        "st louis city sc": "st louis city",
        # Charlotte
        "charlotte fc": "charlotte",
        # Inter Miami
        "inter miami cf": "inter miami",
        # Austin / Cincinnati / Nashville
        "austin fc": "austin",
        "fc cincinnati": "fc cincinnati",
        "cincinnati": "fc cincinnati",
        "nashville": "nashville sc",
        "nashville sc": "nashville sc",
    }

    # apply alias on the raw cleaned string
    if s in aliases:
        s = aliases[s]

    # also apply alias on token-joined form (after dropping extra spaces)
    key_raw = " ".join([t for t in s.split() if t])
    if key_raw in aliases:
        s = aliases[key_raw]

    # 3) remove generic tokens
    drop = {"fc", "sc", "cf", "club"}
    tokens = [t for t in s.split() if t and t not in drop]

    # 4) sort tokens so order doesn't matter ("city new york" == "new york city")
    tokens_sorted = sorted(tokens)

    return " ".join(tokens_sorted)

def _build_team_season_baseline(players_dir: str, *, use_cache: bool = True) -> pd.DataFrame:
    """
    Build team-season 'press propensity baseline' from season-level Players exports.
    Aggregates per-90 defensive actions across the squad:
        baseline = tackles_per_90_overall + interceptions_per_90_overall (+ pressures_per_90_overall if present)
    Returns columns: ['season','team_norm','press_baseline']
    """
    if use_cache:
        try:
            cache_path = _baseline_cache_path(players_dir)
            if os.path.exists(cache_path):
                cache_mtime = os.path.getmtime(cache_path)
                players_mtime = _players_latest_mtime(players_dir)
                # Only trust the cache if it's at least as new as the newest Players CSV
                if cache_mtime >= players_mtime:
                    df = pd.read_csv(cache_path)
                    # Guard against degenerate caches (flat/zero baselines)
                    if {"season", "team_norm", "press_baseline"}.issubset(df.columns) and len(df) > 0:
                        vals = pd.Series(df["press_baseline"]).astype(float)
                        degenerate = (vals.dropna().nunique() <= 1) or (float(vals.fillna(0).abs().sum()) == 0.0)
                        if not degenerate:
                            print(f"🗂️  Loaded press-baseline cache: {os.path.basename(cache_path)}")
                            return df
                        else:
                            print("⚠️  Press-baseline cache appears degenerate; rebuilding…")
                    else:
                        print("⚠️  Press-baseline cache missing columns or empty; rebuilding…")
                else:
                    print("ℹ️  Press-baseline cache is stale vs Players files; rebuilding…")
        except Exception:
            # Any cache issue → rebuild silently
            pass

    frames = []
    needed_any = {"season", "Current Club"}
    stat_cols = [
        "tackles_per_90_overall",
        "interceptions_per_90_overall",
        "pressures_per_90_overall",  # optional
    ]
    # Gather all player CSVs under the league's Players directory
    if not os.path.isdir(players_dir):
        return pd.DataFrame(columns=["season","team_norm","press_baseline"])
    paths = sorted(glob.glob(os.path.join(players_dir, "**", "*.csv"), recursive=True))
    for p in paths:
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception:
            continue
        if not needed_any.issubset(df.columns):
            continue
        cols = [c for c in stat_cols if c in df.columns]
        if not cols:
            continue
        sub = df[["season", "Current Club"] + cols].copy()
        sub.columns = ["season", "club"] + cols
        frames.append(sub)

    if not frames:
        return pd.DataFrame(columns=["season","team_norm","press_baseline"])

    players = pd.concat(frames, ignore_index=True)
    # Normalize season labels coming from Players exports so lookups match match-file inference
    if "season" in players.columns:
        players["season"] = players["season"].astype(str).map(_normalize_season_string)
    for c in stat_cols:
        if c not in players.columns:
            players[c] = 0.0

    players["team_norm"] = players["club"].astype(str).map(_normalize_team)
    # sum per-90 contributions across squad (it’s a crude proxy; acceptable for a season-level baseline)
    players["press_baseline"] = (
        players["tackles_per_90_overall"].fillna(0.0) +
        players["interceptions_per_90_overall"].fillna(0.0) +
        (players["pressures_per_90_overall"].fillna(0.0) if "pressures_per_90_overall" in players.columns else 0.0)
    )

    baseline = (
        players.groupby(["season", "team_norm"], dropna=False)["press_baseline"]
               .sum().reset_index()
    )
    # Guard: if the computed baseline is effectively constant/zero, disable the proxy
    vals = baseline["press_baseline"].astype(float)
    if vals.dropna().nunique() <= 1 or float(vals.fillna(0).abs().sum()) == 0.0:
        return pd.DataFrame(columns=["season","team_norm","press_baseline"])
    baseline["season"] = baseline["season"].astype(str).str.strip()
    baseline["season"] = baseline["season"].astype(str).str.strip()
    try:
        if len(baseline) > 0:
            os.makedirs(MODEL_DIR, exist_ok=True)
            cache_path = _baseline_cache_path(players_dir)
            baseline.to_csv(cache_path, index=False)
            print(f"🗂️  Saved press-baseline cache → {os.path.basename(cache_path)}")
    except Exception:
        pass
    return baseline

def _infer_season_from_filename(path: str) -> str | None:
    """
    Try to pull 'YYYY/YYYY' from filenames. Supports:
      - '2018_2019', '2018-2019', '2018–2019', '2018 to 2019'
      - single-year '2018' -> '2018/2018'
    """
    base = os.path.basename(path)
    import re
    s = base.lower()
    # Two-year season patterns first
    m = re.search(r"((?:19|20)\d{2})\s*(?:[_\-–]|\s+to\s+)\s*((?:19|20)\d{2})", s)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    # Single-year fallback: use the last year-like token in the name
    m1 = re.findall(r"((?:19|20)\d{2})", s)
    if m1:
        y = m1[-1]
        return f"{y}/{y}"
    return None


def _baseline_cache_path(players_dir: str) -> str:
    """Return a per-league cache path for the computed press baselines."""
    league = os.path.basename(players_dir.rstrip(os.sep))
    fname = f"{league.replace(' ', '_')}_press_baseline.csv"
    return os.path.join(MODEL_DIR, fname)
def _players_latest_mtime(players_dir: str) -> float:
    """Return the most recent mtime across all CSVs under players_dir.
    Used to decide if a cache is stale and should be rebuilt."""
    try:
        paths = glob.glob(os.path.join(players_dir, "**", "*.csv"), recursive=True)
        mtimes = [os.path.getmtime(p) for p in paths if os.path.isfile(p)]
        return max(mtimes) if mtimes else 0.0
    except Exception:
        return 0.0
# --- Fuzzy team key helpers ---
from typing import Iterable, Optional

def _tokenize_team_key(k: str) -> set:
    return set([t for t in str(k).split() if t])

def _best_team_key(query: str, candidates: Iterable[str], *, min_jaccard: float = 0.5) -> Optional[str]:
    q = _tokenize_team_key(query)
    best_key, best_score = None, 0.0
    for c in candidates:
        cset = _tokenize_team_key(c)
        if not q or not cset:
            continue
        inter = len(q & cset)
        union = len(q | cset)
        score = inter / union if union else 0.0
        if score > best_score:
            best_key, best_score = c, score
    return best_key if best_score >= min_jaccard else None

# --- Helper: best team key with score, and debug unmatched attach ---
def _best_team_key_with_score(query: str, candidates: Iterable[str]) -> tuple[Optional[str], float]:
    """Like _best_team_key but also returns the Jaccard score."""
    q = _tokenize_team_key(query)
    best_key, best_score = None, 0.0
    for c in candidates:
        cset = _tokenize_team_key(c)
        if not q or not cset:
            continue
        inter = len(q & cset)
        union = len(q | cset)
        score = inter / union if union else 0.0
        if score > best_score:
            best_key, best_score = c, score
    return best_key, best_score

def _debug_unmatched_attach(mp: str,
                            season: str,
                            prev_season: str,
                            prev_keys: Iterable[str],
                            cur_keys: Iterable[str],
                            raw_home_names: pd.Series,
                            raw_away_names: pd.Series) -> None:
    """Print diagnostics when no baseline rows could be attached for a file."""
    try:
        print(f"🔎  Debug attach → {os.path.basename(mp)} | season={season} prev={prev_season}")
        uniq = pd.Index(
            pd.concat([raw_home_names, raw_away_names], ignore_index=True)
              .astype(str).map(_normalize_team).unique()
        ).tolist()
        print(f"   unique normalized team keys in file ({len(uniq)}): {uniq[:12]}{' …' if len(uniq)>12 else ''}")
        prev_keys = list(map(str, prev_keys))
        cur_keys  = list(map(str, cur_keys))
        for t in uniq[:12]:
            p_key, p_s = _best_team_key_with_score(t, prev_keys)
            c_key, c_s = _best_team_key_with_score(t, cur_keys)
            print(f"   • '{t}' → prev: {p_key} (J={p_s:.2f}) | cur: {c_key} (J={c_s:.2f})")
        if len(uniq) > 12:
            print("   (showing first 12 teams)")
    except Exception as _e:
        print(f"   ⚠️  debug failed: {_e}")

def _attach_season_baseline_to_matches(match_dir: str,
                                       baseline: pd.DataFrame,
                                       *,
                                       blend_prev: float = 0.70,
                                       overwrite: bool = False,
                                       debug: bool = False) -> tuple[int,int]:
    """
    For each match CSV:
        1) infer its season (S)
        2) use previous season (S-1) baseline
        3) attach `home_press_baseline` / `away_press_baseline`
    Returns: (updated_count, total_considered)
    """
    pattern = os.path.join(match_dir, "**", "*.csv")
    paths = sorted(glob.glob(pattern, recursive=True))
    def _skip(p: str) -> bool:
        name = os.path.basename(p).lower()
        return (
            any(k in name for k in ("upcoming", "fixture", "fixtures", "prediction", "predictions", "report"))
            or any(k in p.lower() for k in ("predictions_output", "modelstore"))
        )
    paths = [p for p in paths if not _skip(p)]
    if baseline.empty or not paths:
        return (0, 0)

    updated = 0
    for mp in paths:
        try:
            df = pd.read_csv(mp, low_memory=False)
        except Exception:
            continue

        # already has baseline?
        if {"home_press_baseline","away_press_baseline"}.issubset(df.columns) and not overwrite:
            continue

        season = None
        if "season" in df.columns and df["season"].notna().any():
            season = str(df["season"].iloc[0])
        if not season:
            season = _infer_season_from_filename(mp)
        if not season:
            # last resort: derive from first date in file
            if "match_date" in df.columns:
                dt = pd.to_datetime(df["match_date"], errors="coerce")
            elif "date_GMT" in df.columns:
                # try to parse free-form "Aug 11 2023 - 7:00pm"
                dt = _parse_date_series(df["date_GMT"])  # robust free‑form parsing
            else:
                dt = None
            if dt is not None and dt.notna().any():
                y = int(dt.dt.year.mode().iloc[0])
                # Choose a season format that actually exists in the baseline
                cand1 = f"{y}/{y+1}"  # European-style season spanning two years
                cand2 = f"{y}/{y}"    # Single-year season (e.g., MLS)
                try:
                    seasons = set(map(str, baseline["season"].dropna().unique()))
                except Exception:
                    seasons = set()
                if cand1 in seasons:
                    season = cand1
                elif cand2 in seasons:
                    season = cand2
                else:
                    season = cand1

        if not season:
            continue

        prev_season = _season_prev(season)

        bl_prev = baseline[baseline["season"] == prev_season][["team_norm","press_baseline"]].copy()
        bl_cur  = baseline[baseline["season"] == season][["team_norm","press_baseline"]].copy()

        if bl_prev.empty and bl_cur.empty:
            continue

        prev_map = dict(zip(bl_prev["team_norm"].map(str), bl_prev["press_baseline"]))
        cur_map  = dict(zip(bl_cur["team_norm"].map(str),  bl_cur["press_baseline"]))
        prev_keys = list(prev_map.keys())
        cur_keys  = list(cur_map.keys())

        alpha = float(blend_prev)

        def _lookup_blended(raw_team_name: str):
            t = _normalize_team(raw_team_name)
            # direct hits
            pkey = t if t in prev_map else _best_team_key(t, prev_keys)
            ckey = t if t in cur_map  else _best_team_key(t, cur_keys)
            p = prev_map.get(pkey) if pkey else None
            c = cur_map.get(ckey)  if ckey else None
            # blending/fallback: if no prev-season value for expansion teams, use current only
            if p is None and c is None:
                return np.nan
            if p is None:
                return c
            if c is None:
                return p
            return alpha * p + (1.0 - alpha) * c

        df["home_press_baseline"] = df["home_team_name"].apply(_lookup_blended)
        df["away_press_baseline"] = df["away_team_name"].apply(_lookup_blended)

        # count matched (non-null) baseline rows
        matched_rows = int(
            (df[["home_press_baseline","away_press_baseline"]].notna().any(axis=1)).sum()
        )
        if matched_rows == 0 and debug:
            _debug_unmatched_attach(
                mp, season, prev_season, prev_keys, cur_keys,
                df.get("home_team_name", pd.Series(dtype=str)),
                df.get("away_team_name", pd.Series(dtype=str)),
            )

        if matched_rows > 0:
            # write back
            try:
                df.to_csv(mp, index=False)
                updated += 1
                print(f"✅  Attached season press-baseline (prev={prev_season}) → {os.path.basename(mp)}")
            except Exception:
                pass

    return (updated, len(paths))
# ---------------------------------------------------------------
# 1) Player → team totals ETL (to unlock press‑intensity proxy)
# ---------------------------------------------------------------
def ensure_player_team_totals_on_disk(match_dir: str,
                                      *,
                                      baseline_blend: float = 0.70,
                                      use_cache: bool = True,
                                      overwrite_baseline: bool = False,
                                      debug_attach: bool = False) -> None:
    """
    For each season CSV under *match_dir*, if the team‑total columns
    needed by the press‑intensity proxy are missing, try to build them
    from the corresponding player‑level exports in:
        <repo_root>/Players/<league_name>/*.csv

    Expected player CSV columns (row per player per match):
        match_id, side ("home"/"away"),
        passes_total_overall, tackles_total_overall,
        interceptions_total_overall, (optional) pressures_total_overall

    Idempotent: if a matches CSV already has the *_home/*_away total
    columns, it is left untouched.
    """
    try:
        league_name = os.path.basename(os.path.normpath(match_dir))
        # Infer repo root by walking up from the Matches/<league> dir
        root = os.path.dirname(os.path.dirname(os.path.abspath(match_dir)))
        players_dir = os.path.join(root, "Players", league_name)
        if not os.path.isdir(players_dir):
            # fallback to underscore variant (e.g., "England_Premier_League")
            alt = os.path.join(root, "Players", league_name.replace(" ", "_"))
            if os.path.isdir(alt):
                players_dir = alt
            else:
                return

        p_paths = sorted(glob.glob(os.path.join(players_dir, "**", "*.csv"), recursive=True))
        if not p_paths:
            return

        frames: list[pd.DataFrame] = []
        keep_cols = {
            "match_id", "side",
            "passes_total_overall", "tackles_total_overall",
            "interceptions_total_overall", "pressures_total_overall"
        }
        saw_season_aggregate = False
        sample_players_headers: list[str] | None = None
        sample_players_path: str | None = None
        for p in p_paths:
            try:
                df = pd.read_csv(p)
                if sample_players_headers is None:
                    sample_players_headers = list(df.columns)[:20]
                    sample_players_path = os.path.basename(p)
                cols = [c for c in df.columns if c in keep_cols]
                if {"match_id", "side"}.issubset(cols):
                    frames.append(df[cols].copy())
                else:
                    # Heuristic: season aggregate exports tend to include these columns
                    if {"season", "Current Club", "passes_total_overall"}.issubset(set(df.columns)):
                        saw_season_aggregate = True
            except Exception:
                continue
        if not frames:
            if saw_season_aggregate:
                print(
                    "ℹ️  Players files look like SEASON aggregates (no per‑match 'match_id'/'side').\n"
                    f"    Example file: {sample_players_path} | sample headers: {sample_players_headers}\n"
                    "    → Falling back to SEASON baseline proxy (prev‑season team defensive per‑90)."
                )
                try:
                    baseline = _build_team_season_baseline(players_dir, use_cache=use_cache)
                    if not baseline.empty:
                        up, tot = _attach_season_baseline_to_matches(
                            match_dir, baseline, blend_prev=baseline_blend, overwrite=overwrite_baseline, debug=debug_attach
                        )
                        msg = (
                            f"🧪  Season‑baseline proxy attached to {up}/{tot} match CSVs under {match_dir}"
                            if up > 0 else
                            f"🧪  Season‑baseline proxy built, but no files needed updating under {match_dir}"
                        )
                        print(msg)
                    else:
                        print("⚠️  Season‑baseline proxy could not be built (no usable columns found).")
                except Exception as _e:
                    print(f"⚠️  Season‑baseline proxy failed: {_e}")
            return

        players = pd.concat(frames, ignore_index=True)
        # Normalise side labels
        if "side" in players.columns:
            players["side"] = (
                players["side"].astype(str).str.lower().map({
                    "home": "home", "h": "home", "0": "home",
                    "away": "away", "a": "away", "1": "away"
                }).fillna(players["side"])  # keep unknowns as‑is
            )

        agg_cols = [c for c in (
            "passes_total_overall", "tackles_total_overall",
            "interceptions_total_overall", "pressures_total_overall"
        ) if c in players.columns]
        if not agg_cols:
            return

        team_totals = (
            players.dropna(subset=["match_id"]).groupby(["match_id", "side"], dropna=False)[agg_cols]
                   .sum().reset_index()
        )
        # Pivot side → _home/_away
        wide = team_totals.pivot(index="match_id", columns="side")
        wide.columns = [f"{stat}_{side}" for stat, side in wide.columns.to_flat_index()]
        wide = wide.reset_index()

        # Walk all match CSVs and merge where totals are missing
        m_paths = sorted(glob.glob(os.path.join(match_dir, "**", "*.csv"), recursive=True))

        def _skip(p: str) -> bool:
            name = os.path.basename(p).lower()
            if any(k in name for k in ("upcoming", "fixture", "fixtures", "prediction", "predictions", "report")):
                return True
            if any(k in p.lower() for k in ("predictions_output", "modelstore")):
                return True
            return False

        need = {
            "passes_total_overall_home", "passes_total_overall_away",
            "tackles_total_overall_home", "tackles_total_overall_away",
            "interceptions_total_overall_home", "interceptions_total_overall_away",
        }
        for mp in m_paths:
            if _skip(mp):
                continue
            try:
                m = pd.read_csv(mp)
                if "match_id" not in m.columns:
                    continue
                if need.issubset(set(m.columns)):
                    continue
                merged = m.merge(wide, on="match_id", how="left")
                if need.issubset(set(merged.columns)):
                    merged.to_csv(mp, index=False)
                    print(f"✅  Wrote team totals from Players → {os.path.basename(mp)}")
            except Exception:
                continue
    except Exception:
        return


# ---------------------------------------------------------------
# 2) Match‑level press‑intensity proxy (PPDA‑like)
# ---------------------------------------------------------------
def compute_press_intensity(match_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a *match‑level* dataframe (already merged with team totals), return:
        match_id, home_press_intensity, away_press_intensity

    Proxy formula (full‑pitch PPDA):
        press_intensity = opponent_passes / own_def_actions
        own_def_actions = tackles + interceptions (+ pressures if exists)

    Required columns (suffix _home / _away):
        passes_total_overall, tackles_total_overall,
        interceptions_per_game_overall  # FootyStats naming
        pressures_total_overall         # optional
    """
    df = match_df.copy()

    # Robust alias resolution for both side‑specific and reversed spellings
    def _pick(candidates: list[str]) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    for side in ("home", "away"):
        # passes
        _p = _pick([
            f"passes_total_overall_{side}", f"passes_total_{side}", f"total_passes_{side}",
            f"{side}_passes_total", f"{side}_total_passes", "passes_total_overall"
        ])
        if _p and f"passes_total_overall_{side}" not in df.columns:
            df[f"passes_total_overall_{side}"] = df[_p]

        # tackles
        _t = _pick([
            f"tackles_total_overall_{side}", f"tackles_total_{side}", f"total_tackles_{side}",
            f"{side}_tackles_total", f"{side}_total_tackles", "tackles_total_overall"
        ])
        if _t and f"tackles_total_overall_{side}" not in df.columns:
            df[f"tackles_total_overall_{side}"] = df[_t]

        # interceptions (accept per‑game or total)
        _i = _pick([
            f"interceptions_per_game_overall_{side}",
            f"interceptions_total_overall_{side}", f"interceptions_total_{side}",
            f"total_interceptions_{side}", f"{side}_interceptions_total", f"{side}_total_interceptions",
            "interceptions_per_game_overall", "interceptions_total_overall"
        ])
        if _i and f"interceptions_per_game_overall_{side}" not in df.columns:
            # normalise into the per‑game canonical slot even if it's actually totals
            df[f"interceptions_per_game_overall_{side}"] = df[_i]

        # pressures (optional)
        _pr = _pick([
            f"pressures_total_overall_{side}", f"pressures_total_{side}", f"total_pressures_{side}",
            f"{side}_pressures_total", f"{side}_total_pressures", "pressures_total_overall"
        ])
        if _pr and f"pressures_total_overall_{side}" not in df.columns:
            df[f"pressures_total_overall_{side}"] = df[_pr]

    # Ensure match_id exists (attempt to synthesise if absent)
    if "match_id" not in df.columns:
        if {"home_team_name", "away_team_name"}.issubset(df.columns):
            if "match_date" in df.columns:
                _dt = pd.to_datetime(df["match_date"], errors="coerce")
            elif "date_GMT" in df.columns:
                _dt = _parse_date_series(df["date_GMT"])  # robust free‑form parsing
            else:
                _dt = pd.Series(pd.NaT, index=df.index)
            _date_str = _dt.dt.strftime("%Y-%m-%d %H:%M").fillna("")
            synthetic = (
                _date_str + "_" + df["home_team_name"].astype(str) + "_" + df["away_team_name"].astype(str)
            )
            df["match_id"] = synthetic.factorize()[0]
        else:
            raise KeyError("Missing columns for synthetic match_id creation")

    required = [
        "match_id",
        "passes_total_overall_home", "passes_total_overall_away",
        "tackles_total_overall_home", "tackles_total_overall_away",
        # intercept columns are handled below
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for press‑intensity proxy: {missing}")

    for side in ("home", "away"):
        _int_pg = f"interceptions_per_game_overall_{side}"
        _int_tot = f"interceptions_total_overall_{side}"
        _int_col = _int_pg if _int_pg in df.columns else (_int_tot if _int_tot in df.columns else None)

        df[f"{side}_def_actions"] = df[f"tackles_total_overall_{side}"].fillna(0)
        if _int_col:
            df[f"{side}_def_actions"] += df[_int_col].fillna(0)

        press_col = f"pressures_total_overall_{side}"
        if press_col in df.columns:
            df[f"{side}_def_actions"] += df[press_col].fillna(0)

    with np.errstate(divide="ignore", invalid="ignore"):
        df["home_press_intensity"] = (
            df["passes_total_overall_away"] / df["home_def_actions"].replace(0, np.nan)
        )
        df["away_press_intensity"] = (
            df["passes_total_overall_home"] / df["away_def_actions"].replace(0, np.nan)
        )

    return df[["match_id", "home_press_intensity", "away_press_intensity"]]


# ---------------------------------------------------------------
# 3) Single‑file in‑place attachment helper
# ---------------------------------------------------------------
def _attach_press_intensity(match_path: str) -> None:
    """Attach press‑intensity columns to one match CSV in‑place.
    Skips gracefully when inputs are missing or file is a non‑match artifact.
    """
    name = os.path.basename(match_path).lower()
    if any(k in name for k in ("upcoming", "fixture", "fixtures", "prediction", "predictions", "report")):
        return

    df = pd.read_csv(match_path)

    if "match_id" not in df.columns and {"home_team_name", "away_team_name"}.issubset(df.columns):
        if "match_date" in df.columns:
            _date = pd.to_datetime(df["match_date"], errors="coerce")
        elif "date_GMT" in df.columns:
            _date = _parse_date_series(df["date_GMT"])  # robust free‑form parsing
        else:
            _date = pd.Series(pd.NaT, index=df.index)
        _date_str = _date.dt.strftime("%Y-%m-%d %H:%M").fillna("")
        synthetic = _date_str + "_" + df["home_team_name"].astype(str) + "_" + df["away_team_name"].astype(str)
        df["match_id"] = synthetic.factorize()[0]

    try:
        press_df = compute_press_intensity(df)
    except KeyError as err:
        # include a short header sample in the message for debugging
        hdr = ", ".join(list(df.columns)[:18])
        print(f"⏩  Skipping {os.path.basename(match_path)} – '{err} | sample headers: [{hdr} ...]'" )
        return

    out = df.merge(press_df, on="match_id", how="left")
    out.to_csv(match_path, index=False)
    print(f"✅  Wrote press_intensity → {os.path.basename(match_path)}")


# ---------------------------------------------------------------
# 4) Batch runner used by the main pipeline
# ---------------------------------------------------------------
def debug_press_join_sample(match_csv: str, players_csv: str) -> None:
    """Print a quick comparison of available join keys between a match CSV and a players CSV.
    Helps diagnose why Players→Matches enrichment is not happening.
    """
    try:
        m = pd.read_csv(match_csv)
    except Exception as e:
        print(f"❌ Could not read match CSV: {match_csv} — {e}")
        return
    try:
        p = pd.read_csv(players_csv)
    except Exception as e:
        print(f"❌ Could not read players CSV: {players_csv} — {e}")
        return

    print("\n—— DEBUG: join‑key availability ———————————————————")
    print("Match CSV candidate keys present:")
    keys_m = [k for k in ("match_id", "date_GMT", "match_date", "home_team_name", "away_team_name") if k in m.columns]
    print("  ", keys_m)

    print("Players CSV candidate keys present:")
    keys_p = [k for k in ("match_id", "side", "season", "Current Club") if k in p.columns]
    print("  ", keys_p)

    # Synthesize a few example _join_key values on the match side (date + sorted team names)
    if {"home_team_name", "away_team_name"}.issubset(m.columns):
        if "match_date" in m.columns:
            dt = pd.to_datetime(m["match_date"], errors="coerce")
        elif "date_GMT" in m.columns:
            dt = _parse_date_series(m["date_GMT"])  # robust free‑form parsing
        else:
            dt = pd.Series(pd.NaT, index=m.index)
        dstr = dt.dt.strftime("%Y-%m-%d").fillna("")
        h = m["home_team_name"].astype(str).str.strip()
        a = m["away_team_name"].astype(str).str.strip()
        jkey = (dstr + "_" + np.minimum(h, a) + "_" + np.maximum(h, a)).head(5).tolist()
        print("Match _join_key examples:")
        for s in jkey:
            print("   ", s)
    else:
        print("Match CSV missing team name columns; cannot demonstrate _join_key.")

    # On the players side, call out lack of per‑match keys explicitly
    if not {"match_id", "side"}.issubset(p.columns):
        print(
            "\nPlayers CSV lacks per‑MATCH keys ('match_id','side'). It appears to be season‑level.\n"
            "Attachable proxy would need a team‑SEASON mapping instead (e.g., previous season baseline)."
        )
    print("—— END DEBUG ————————————————————————————————————\n")

# ---------------------------------------------------------------
def ensure_press_intensity_on_disk(match_dir: str, *,
                                   force: bool = False,
                                   max_files: int | None = None,
                                   baseline_blend: float = 0.70,
                                   use_cache: bool = True,
                                   overwrite_baseline: bool = False,
                                   overwrite_intensity: bool = False,
                                   debug_attach: bool = False) -> None:
    """Run the press‑intensity ETL across season match CSVs under *match_dir*.
    Skips fixtures/prediction/report CSVs and files missing core columns.
    Also attempts to derive team totals from Players/<league> if needed.
    """
    # First, try to build team totals from player files
    try:
        ensure_player_team_totals_on_disk(match_dir,
                                          baseline_blend=baseline_blend,
                                          use_cache=use_cache,
                                          overwrite_baseline=overwrite_baseline,
                                          debug_attach=debug_attach)
    except Exception as _e:
        print(f"⚠️ player→team totals ETL skipped for {match_dir}: {_e}")
    pattern = os.path.join(match_dir, "**", "*.csv")
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        return

    def _should_skip(path: str) -> bool:
        name = os.path.basename(path).lower()
        if any(k in name for k in ("upcoming", "fixture", "fixtures", "prediction", "predictions", "report")):
            return True
        if any(k in path.lower() for k in ("predictions_output", "modelstore")):
            return True
        return False

    paths = [p for p in paths if not _should_skip(p)]

    updated = 0
    skipped_missing = 0
    skipped_present = 0

    for p in paths:
        try:
            hdr = pd.read_csv(p, nrows=1)
            if not force and {"home_press_intensity", "away_press_intensity"}.issubset(hdr.columns):
                skipped_present += 1
                continue

            hdr_cols = set(hdr.columns)

            def _has_any(opts: list[str]) -> bool:
                return any(o in hdr_cols for o in opts)

            has_passes = _has_any([
                "passes_total_overall_home", "passes_total_overall_away",
                "passes_total_home", "passes_total_away",
                "total_passes_home", "total_passes_away",
                "home_passes_total", "away_passes_total",
                "passes_total_overall"
            ])
            has_tackles = _has_any([
                "tackles_total_overall_home", "tackles_total_overall_away",
                "tackles_total_home", "tackles_total_away",
                "total_tackles_home", "total_tackles_away",
                "home_tackles_total", "away_tackles_total",
                "tackles_total_overall"
            ])
            has_intercepts = _has_any([
                "interceptions_per_game_overall_home", "interceptions_per_game_overall_away",
                "interceptions_total_overall_home", "interceptions_total_overall_away",
                "interceptions_total_home", "interceptions_total_away",
                "total_interceptions_home", "total_interceptions_away",
                "home_interceptions_total", "away_interceptions_total",
                "interceptions_per_game_overall", "interceptions_total_overall"
            ])

            if not (has_passes and has_tackles and has_intercepts):
                skipped_missing += 1
                continue

            # Attempt attachment and count only if we actually wrote the columns
            _attach_press_intensity(p)
            try:
                hdr2 = pd.read_csv(p, nrows=1)
                if {"home_press_intensity", "away_press_intensity"}.issubset(set(hdr2.columns)):
                    updated += 1
            except Exception:
                pass
            if max_files is not None and updated >= max_files:
                break
        except Exception as _e:
            print(f"⏩  Skipping {os.path.basename(p)} – {_e}")

    if updated or skipped_missing or skipped_present:
        print(
            f"🧰  Press-intensity ETL: updated {updated} | "
            f"already-present {skipped_present} | missing-cols {skipped_missing} "
            f"under {match_dir}"
        )

    # If we still lack match-level intensity but have season baseline, promote it as a proxy
    try:
        promoted, considered = _promote_baseline_to_proxy(match_dir, overwrite=overwrite_intensity)
        if considered:
            print(f"🧪  Promoted baseline→intensity for {promoted}/{considered} CSVs under {match_dir}")
    except Exception as _e:
        print(f"⚠️  Baseline→intensity promotion failed: {_e}")

    # Combined coverage snapshot (intensity + baseline)
    try:
        paths2 = sorted(glob.glob(os.path.join(match_dir, "**", "*.csv"), recursive=True))
        def _skip2(p: str) -> bool:
            name2 = os.path.basename(p).lower()
            return (
                any(k in name2 for k in ("upcoming", "fixture", "fixtures", "prediction", "predictions", "report"))
                or any(k in p.lower() for k in ("predictions_output", "modelstore"))
            )
        paths2 = [p for p in paths2 if not _skip2(p)]
        have_intensity = 0
        have_baseline = 0
        for p2 in paths2:
            try:
                hdr2 = pd.read_csv(p2, nrows=1)
                if {"home_press_intensity", "away_press_intensity"}.issubset(hdr2.columns):
                    have_intensity += 1
                if {"home_press_baseline", "away_press_baseline"}.issubset(hdr2.columns):
                    have_baseline += 1
            except Exception:
                continue
        if paths2:
            print(
                f"🧪  Coverage: press-intensity present in {have_intensity}/{len(paths2)} | "
                f"season-baseline present in {have_baseline}/{len(paths2)} under {match_dir}"
            )
    except Exception:
        pass


# ---------------------------------------------------------------
# 5) CLI
# ---------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attach press-intensity metrics to every match CSV under --match-dir (recursively)")
    parser.add_argument("--match-dir", required=True, help="Root directory containing per‑season match CSVs")
    parser.add_argument("--force", action="store_true", help="Force re‑write even if columns already present")
    parser.add_argument("--baseline-blend", type=float, default=0.70,
                        help="Weight for previous season in baseline blend (0..1). 0.70 = 70% prev + 30% current")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable per-league baseline cache for press baselines")
    parser.add_argument("--overwrite-baseline", action="store_true",
                        help="Rewrite home/away_press_baseline even if present")
    parser.add_argument("--overwrite-intensity", action="store_true",
                        help="Rewrite home/away_press_intensity from baseline even if present")
    parser.add_argument("--debug-attach", action="store_true",
                        help="Print diagnostics when no baseline could be attached to a file")
    args = parser.parse_args()
    ensure_press_intensity_on_disk(
        args.match_dir,
        force=args.force,
        baseline_blend=args.baseline_blend,
        use_cache=not args.no_cache,
        overwrite_baseline=args.overwrite_baseline,
        overwrite_intensity=args.overwrite_intensity,
        debug_attach=args.debug_attach,
    )