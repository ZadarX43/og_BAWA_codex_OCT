#!/usr/bin/env python3
"""
patch_merge_add_synth_odds.py

Patch synth Under2.5 (and related metadata) from:
  Matches/<League>/fd_odds_enriched_synth.csv
into:
  Matches/__merged__/<LeagueTag>__merged.csv

Goal:
- Do NOT rebuild merged files.
- Do NOT corrupt other engineered features.
- Only add/fill the synth odds columns in-place.

Join strategy:
1) If both sides have fixture_key -> merge on fixture_key
2) Else merge on a robust fallback key: (match_datetime_rounded_to_minute + home + away)

By default:
- We only fill columns when merged value is missing/blank.
- Use --overwrite to force synth to overwrite existing values.

Run:
  python patch_merge_add_synth_odds.py --root "/Users/hughwade/Documents/Code/OG_master/BAWA PORTO"

Optional:
  --leagues "Germany Bundesliga,England Championship"
  --overwrite
  --harmonize-duplicates
  --audit-dup-variance
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


# -----------------------------
# Helpers
# -----------------------------
def league_tag(name: str) -> str:
    return str(name).strip().replace(" ", "_")

def parse_dt(df: pd.DataFrame) -> pd.Series:
    """Return best-effort datetime series (UTC-naive).

    Strategy (fast -> slow):
      1) match_date (usually ISO)
      2) timestamp epoch (s/ms)
      3) date_GMT with a few known formats
      4) dateutil fallback (slow)

    We return timezone-naive timestamps (UTC-normalised).
    """
    idx = df.index
    out = pd.Series(pd.NaT, index=idx)

    # 1) Preferred explicit match_date (typically ISO-like)
    if "match_date" in df.columns:
        dt = pd.to_datetime(df["match_date"], errors="coerce", utc=True)
        try:
            dt = dt.dt.tz_convert(None)
        except Exception:
            pass
        out = out.fillna(pd.Series(dt, index=idx))

    # 2) Epoch timestamp fallback (fills remaining gaps)
    if "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        if ts.notna().any():
            med = float(ts.dropna().median())
            unit = "ms" if med > 1e12 else "s"
            dt = pd.to_datetime(ts, unit=unit, errors="coerce", utc=True)
            try:
                dt = dt.dt.tz_convert(None)
            except Exception:
                pass
            out = out.fillna(pd.Series(dt, index=idx))

    # 3) date_GMT and other common date columns
    # Try a few known formats first (fast, avoids dateutil warning), then fall back.
    for col in ("date_GMT", "date", "Date"):
        if col not in df.columns:
            continue

        s = df[col]

        # Try known formats for strings like: "Aug 24 2018 - 6:30pm"
        if out.isna().any():
            for fmt in (
                "%b %d %Y - %I:%M%p",
                "%b %d %Y - %I:%M %p",
                "%b %d %Y %I:%M%p",
                "%b %d %Y %I:%M %p",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%Y-%m-%d",
            ):
                if not out.isna().any():
                    break
                dt = pd.to_datetime(s, format=fmt, errors="coerce", utc=True)
                try:
                    dt = dt.dt.tz_convert(None)
                except Exception:
                    pass
                out = out.fillna(pd.Series(dt, index=idx))

        # Final fallback: dateutil/infer (slow, but robust)
        if out.isna().any():
            dt = pd.to_datetime(s, errors="coerce", utc=True)
            try:
                dt = dt.dt.tz_convert(None)
            except Exception:
                pass
            out = out.fillna(pd.Series(dt, index=idx))

    return out

def norm_team(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
         .fillna("")
         .str.strip()
         .str.lower()
         .str.replace(r"\s+", " ", regex=True)
    )


def make_fallback_key(df: pd.DataFrame) -> pd.Series:
    """Key = YYYY-mm-dd HH:MM + home + away (all normalised)."""
    dt = parse_dt(df)
    dt_min = dt.dt.floor("min")
    dt_str = dt_min.dt.strftime("%Y-%m-%d %H:%M").fillna("")

    home = norm_team(df.get("home_team_name", pd.Series("", index=df.index)))
    away = norm_team(df.get("away_team_name", pd.Series("", index=df.index)))

    k = dt_str.astype("string") + "|" + home.astype("string") + "|" + away.astype("string")
    k = k.fillna("").astype("string")
    return k


# --- New helpers: slug team and fixture_key generator ---
def _slug_team_for_key(s: pd.Series) -> pd.Series:
    """Normalize team names into a fixture_key-safe token."""
    return (
        s.astype("string")
         .fillna("")
         .str.strip()
         .str.replace(r"\s+", "_", regex=True)
         .str.replace(r"[^0-9A-Za-z_]+", "_", regex=True)
         .str.replace(r"_+", "_", regex=True)
         .str.strip("_")
    )


def make_fixture_key(df: pd.DataFrame) -> pd.Series:
    """Best-effort construct `fixture_key` like YYYY_MM_DD_HHMM_Home_Away.

    - Includes kickoff time when available (HHMM from parsed datetime).
    - If duplicates still occur within the same dataframe and `match_id` exists,
      we append a slug of match_id to disambiguate.

    This is used to fill blanks in merged files (common in upcoming fixtures).
    We intentionally keep this deterministic and conservative.
    """
    dt = parse_dt(df)

    # Date is required; time is optional.
    date_str = dt.dt.strftime("%Y_%m_%d").fillna("")

    # Kickoff time (HHMM) when available; blank if dt is NaT.
    time_str = dt.dt.strftime("%H%M").fillna("")

    home = _slug_team_for_key(df.get("home_team_name", pd.Series("", index=df.index)))
    away = _slug_team_for_key(df.get("away_team_name", pd.Series("", index=df.index)))

    # Build base key. If time is missing, omit it.
    base = date_str.astype("string")
    with_time = base + "_" + time_str.astype("string")
    fk = (with_time.where(time_str.ne(""), base) + "_" + home.astype("string") + "_" + away.astype("string"))
    fk = fk.str.replace(r"_+", "_", regex=True).str.strip("_")

    # If we couldn't build it (missing date or teams), return empty strings
    bad = date_str.eq("") | home.eq("") | away.eq("")
    fk = fk.mask(bad, "")

    # Disambiguate collisions inside this DF using match_id when available.
    try:
        if "match_id" in df.columns:
            mid = df["match_id"].astype("string").fillna("").str.strip()
            mid = mid.str.replace(r"\s+", "_", regex=True)
            mid = mid.str.replace(r"[^0-9A-Za-z_]+", "_", regex=True)
            mid = mid.str.replace(r"_+", "_", regex=True).str.strip("_")

            # Only for keys that are duplicated (and have a non-empty match_id), append match_id.
            dup = fk.ne("") & fk.duplicated(keep=False)
            use_mid = dup & mid.ne("")
            fk = fk.where(~use_mid, fk + "_" + mid)

            # If still duplicated, append row index to remaining duplicates.
            dup2 = fk.ne("") & fk.duplicated(keep=False)
            if dup2.any():
                fk = fk.where(~dup2, fk + "_" + pd.Series(df.index, index=df.index).astype("string"))
    except Exception:
        pass

    return fk.astype("string")


def pick_first_existing(path_candidates: list[Path]) -> Path | None:
    for p in path_candidates:
        if p.exists():
            return p
    return None


def find_synth_file(league_dir: Path) -> Path | None:
    """Best-effort locate fd_odds_enriched_synth.csv anywhere under a league folder.

    We prefer a top-level file if present, but also support season subfolders or
    intermediate pipelines that write to nested locations.
    """
    # Common direct locations first
    direct = pick_first_existing([
        league_dir / "fd_odds_enriched_synth.csv",
        league_dir / "fd_odds_enriched_synth" / "fd_odds_enriched_synth.csv",
    ])
    if direct is not None:
        return direct

    # Recursive fallback: grab the newest fd_odds_enriched_synth.csv under this league
    try:
        cands = list(league_dir.rglob("fd_odds_enriched_synth.csv"))
        if not cands:
            return None
        # Prefer most recently modified file
        cands = sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)
        return cands[0]
    except Exception:
        return None


# -----------------------------
# Patch logic
# -----------------------------

# Columns we patch from fd_odds_enriched_synth.csv
PATCH_COLS_SYNTH = [
    # core under2.5 odds + provenance
    "odds_ft_under25",
    "odds_source_under25",
    "synth_u25_method",
    "synth_u25_conf",

    # if present (some flows)
    "synth_odds_ft_under25",
    "synth_odds_under25",
]

# Columns we patch from fd_ou25_novig.csv (emitted by make_fd_odds_enriched_synth.py --emit-ou25-novig)
PATCH_COLS_OU25_NOVIG = [
    "p_over25_novig",
    "p_under25_novig",
    "ou25_overround",
]
def find_ou25_novig_file(league_dir: Path) -> Path | None:
    """Locate fd_ou25_novig.csv under a league folder (top-level preferred)."""
    direct = pick_first_existing([
        league_dir / "fd_ou25_novig.csv",
        league_dir / "fd_ou25_novig" / "fd_ou25_novig.csv",
    ])
    if direct is not None:
        return direct

    try:
        cands = list(league_dir.rglob("fd_ou25_novig.csv"))
        if not cands:
            return None
        cands = sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)
        return cands[0]
    except Exception:
        return None


def patch_one_league(root: Path, league_name: str, *, overwrite: bool = False, harmonize_duplicates: bool = False, audit_dup_variance: bool = False) -> dict:
    matches_dir = root / "Matches"
    merged_dir = matches_dir / "__merged__"

    tag = league_tag(league_name)
    merged_path = merged_dir / f"{tag}__merged.csv"

    # synth file lives in league folder (or sometimes in season subfolder; try a few)
    league_dir = matches_dir / league_name
    synth_path = find_synth_file(league_dir)
    novig_path = find_ou25_novig_file(league_dir)

    if not merged_path.exists():
        return {"league": league_name, "ok": False, "reason": f"missing merged: {merged_path}"}
    if synth_path is None:
        return {
            "league": league_name,
            "ok": False,
            "reason": f"missing fd_odds_enriched_synth.csv under: {league_dir}",
        }

    if novig_path is None:
        print(f"ℹ️ [PATCH_SYNTH] {league_name}: fd_ou25_novig.csv not found under {league_dir} (novig cols will not be patched)")

    dfm = pd.read_csv(merged_path, low_memory=False)
    dfs = pd.read_csv(synth_path, low_memory=False)
    dfn = pd.read_csv(novig_path, low_memory=False) if (novig_path is not None and novig_path.exists()) else None

    # --- AUDIT (before): capture baseline non-null coverage for key patch cols ---
    audit_cols = [
        "odds_ft_under25",
        "odds_source_under25",
        "synth_u25_method",
        "synth_u25_conf",
    ]

    def _nonnull_rate(df: pd.DataFrame, col: str) -> tuple[int, int, float]:
        if col not in df.columns:
            return (0, int(len(df)), 0.0)
        s = pd.Series(df[col]).replace("", np.nan)
        nn = int(s.notna().sum())
        tot = int(len(s))
        rate = float(nn / tot) if tot else 0.0
        return (nn, tot, rate)

    before_audit = {c: _nonnull_rate(dfm, c) for c in audit_cols}

    # Keep only patch columns that actually exist in the synth/novig files
    cols_in_synth = [c for c in PATCH_COLS_SYNTH if c in dfs.columns]
    cols_in_novig = [c for c in PATCH_COLS_OU25_NOVIG if (dfn is not None and c in dfn.columns)]
    if (not cols_in_synth) and (not cols_in_novig):
        return {
            "league": league_name,
            "ok": False,
            "reason": f"nothing to patch: synth has none of {PATCH_COLS_SYNTH} and novig has none of {PATCH_COLS_OU25_NOVIG}",
        }

    # ensure team cols exist (some merged files can be different cased; adapt if needed)
    for df in (dfm, dfs) + (() if dfn is None else (dfn,)):
        if "home_team_name" not in df.columns and "Home" in df.columns:
            df["home_team_name"] = df["Home"]
        if "away_team_name" not in df.columns and "Away" in df.columns:
            df["away_team_name"] = df["Away"]

    # --- Fix missing/blank fixture_key (common in UPCOMING fixtures inside merged files) ---
    # We fill blanks so that fixture_key-based joins/audits behave deterministically.
    for _name, _df in (("merged", dfm), ("synth", dfs)) + (() if dfn is None else (("novig", dfn),)):
        try:
            if "fixture_key" not in _df.columns:
                # Only create if we have the required columns
                if {"home_team_name", "away_team_name"}.issubset(_df.columns):
                    _df["fixture_key"] = make_fixture_key(_df)
            else:
                fk = _df["fixture_key"].astype("string").fillna("").str.strip()
                n_blank = int((fk.eq("")).sum())
                if n_blank:
                    gen = make_fixture_key(_df)
                    _df["fixture_key"] = fk.where(fk.ne(""), gen)
                # Optional: print counts for visibility
                fk2 = _df["fixture_key"].astype("string").fillna("").str.strip()
                n_blank2 = int((fk2.eq("")).sum())
                if n_blank or n_blank2:
                    print(f"[PATCH_SYNTH] {league_name}: {_name} fixture_key blank BEFORE={n_blank} AFTER={n_blank2}")
        except Exception as _e_fk:
            print(f"ℹ️ [PATCH_SYNTH] {league_name}: fixture_key fill skipped for {_name}: {_e_fk}")

    # choose join key
    use_fixture_key = ("fixture_key" in dfm.columns) and ("fixture_key" in dfs.columns)
    if use_fixture_key:
        dfm["__k"] = dfm["fixture_key"].astype("string").fillna("").str.strip()
        dfs["__k"] = dfs["fixture_key"].astype("string").fillna("").str.strip()
        if dfn is not None and "fixture_key" in dfn.columns:
            dfn["__k"] = dfn["fixture_key"].astype("string").fillna("").str.strip()
        use_fixture_key = bool(dfs["__k"].ne("").any()) and bool(dfm["__k"].ne("").any())
        if dfn is not None:
            use_fixture_key = use_fixture_key and bool(dfn["__k"].ne("").any())

    if not use_fixture_key:
        dfm["__k"] = make_fallback_key(dfm)
        dfs["__k"] = make_fallback_key(dfs)
        if dfn is not None:
            dfn["__k"] = make_fallback_key(dfn)

    # --- Build RHS lookup from synth ---
    rhs_s = None
    if cols_in_synth:
        rhs_s = dfs[["__k"] + cols_in_synth].copy()
        for c in cols_in_synth:
            rhs_s[c] = rhs_s[c].replace("", np.nan)

        try:
            m_dups = int(dfm["__k"].duplicated().sum())
            s_dups = int(rhs_s["__k"].duplicated().sum())
            m_uniq = int(dfm["__k"].nunique(dropna=False))
            s_uniq = int(rhs_s["__k"].nunique(dropna=False))
            print(f"[PATCH_SYNTH] {league_name}: merged_rows={len(dfm)} merged_dup_keys={m_dups} merged_uniq_keys={m_uniq} | synth_rows={len(dfs)} synth_dup_keys={s_dups} synth_uniq_keys={s_uniq}")
        except Exception:
            pass

        rhs_s["__nn"] = rhs_s[cols_in_synth].notna().sum(axis=1)
        rhs_s = rhs_s.sort_values(["__k", "__nn"], ascending=[True, False]).drop_duplicates("__k", keep="first")
        rhs_s = rhs_s.drop(columns=["__nn"]).set_index("__k")

    # --- Build RHS lookup from novig file ---
    rhs_n = None
    if (dfn is not None) and cols_in_novig:
        rhs_n = dfn[["__k"] + cols_in_novig].copy()
        for c in cols_in_novig:
            rhs_n[c] = rhs_n[c].replace("", np.nan)

        rhs_n["__nn"] = rhs_n[cols_in_novig].notna().sum(axis=1)
        rhs_n = rhs_n.sort_values(["__k", "__nn"], ascending=[True, False]).drop_duplicates("__k", keep="first")
        rhs_n = rhs_n.drop(columns=["__nn"]).set_index("__k")

    before_cols = set(dfm.columns)

    out = dfm.copy()

    filled = {}

    # Patch synth-derived columns
    if rhs_s is not None:
        for c in cols_in_synth:
            try:
                synth_series = rhs_s[c]
            except Exception:
                continue
            synth_vals = out["__k"].map(synth_series)

            if c not in out.columns:
                out[c] = synth_vals
            else:
                if overwrite:
                    out[c] = synth_vals.where(synth_vals.notna(), out[c])
                else:
                    base = pd.Series(out[c])
                    base2 = base.replace("", np.nan)
                    out[c] = base2.where(base2.notna(), synth_vals)

            filled[c] = int(pd.Series(out[c]).replace("", np.nan).notna().sum())

    # Patch novig-derived columns
    if rhs_n is not None:
        for c in cols_in_novig:
            try:
                novig_series = rhs_n[c]
            except Exception:
                continue
            novig_vals = out["__k"].map(novig_series)

            if c not in out.columns:
                out[c] = novig_vals
            else:
                if overwrite:
                    out[c] = novig_vals.where(novig_vals.notna(), out[c])
                else:
                    base = pd.Series(out[c])
                    base2 = base.replace("", np.nan)
                    out[c] = base2.where(base2.notna(), novig_vals)

            filled[c] = int(pd.Series(out[c]).replace("", np.nan).notna().sum())

    # --- Optional: harmonize duplicate merged rows so all rows for the same key carry the same patched values ---
    # This prevents "duplicate drift" where the same fixture_key appears multiple times (from different source CSVs)
    # with different odds_ft_under25 values. By default we do NOT overwrite existing non-null values, so duplicates
    # may legitimately differ. If you want strict per-fixture consistency, enable --harmonize-duplicates.
    if bool(harmonize_duplicates):
        try:
            # Prefer fixture_key grouping when available; otherwise fall back to the join key.
            gkey = "fixture_key" if (use_fixture_key and "fixture_key" in out.columns) else "__k"

            for c in cols_in_synth:
                if c not in out.columns:
                    continue

                # Choose one representative value per key (post-patch), then broadcast to ALL duplicate rows.
                # NOTE: This is intentionally stronger than --overwrite: it enforces per-fixture consistency
                # inside merged files that contain duplicate rows from multiple source CSVs.
                base = pd.Series(out[c]).replace("", np.nan)
                chosen = (
                    base.groupby(out[gkey])
                        .transform(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
                )

                # Only apply where we actually have a chosen value; otherwise leave as-is.
                out[c] = base.where(chosen.isna(), chosen)
        except Exception as _e_h:
            print(f"ℹ️ [PATCH_SYNTH] {league_name}: harmonize_duplicates skipped: {_e_h}")

    # --- Optional: audit duplicate variance for odds_ft_under25 ---
    # Reports how many fixture_keys (or join keys) have >1 distinct non-null under25 value.
    if bool(audit_dup_variance):
        try:
            gkey = "fixture_key" if (use_fixture_key and "fixture_key" in out.columns) else "__k"
            if "odds_ft_under25" in out.columns:
                s = pd.to_numeric(pd.Series(out["odds_ft_under25"]).replace("", np.nan), errors="coerce")
                nunq = s.groupby(out[gkey]).nunique(dropna=True)
                bad_keys = int((nunq > 1).sum())
                tot_keys = int(nunq.shape[0])
                rate = (bad_keys / tot_keys * 100.0) if tot_keys else 0.0
                print(f"[AUDIT_DUP] {league_name}: keys_with_variance_under25={bad_keys}/{tot_keys} ({rate:5.2f}%)")
        except Exception as _e_dv:
            print(f"ℹ️ [AUDIT_DUP] {league_name}: variance audit skipped: {_e_dv}")

    out = out.drop(columns=["__k"], errors="ignore")

    # --- AUDIT (after): non-null coverage for key patch cols ---
    after_audit = {c: _nonnull_rate(out, c) for c in audit_cols}

    def _fmt_trip(trip: tuple[int, int, float]) -> str:
        nn, tot, rate = trip
        return f"{nn}/{tot} ({rate*100.0:5.1f}%)"

    # Always print a compact per-league audit line so we can prove patch landed.
    # Note: some leagues may legitimately have 0% because synth odds are not available yet.
    try:
        b = _fmt_trip(before_audit.get("odds_ft_under25", (0, int(len(dfm)), 0.0)))
        a = _fmt_trip(after_audit.get("odds_ft_under25", (0, int(len(out)), 0.0)))
        print(f"[AUDIT_SYNTH] {league_name}: odds_ft_under25 non-null BEFORE={b} AFTER={a}")
    except Exception:
        pass

    # write back
    out.to_csv(merged_path, index=False)

    added_cols = [c for c in cols_in_synth if c not in before_cols and c in out.columns]
    return {
        "league": league_name,
        "ok": True,
        "merged": str(merged_path),
        "synth": str(synth_path),
        "join": "fixture_key" if use_fixture_key else "dt+teams",
        "added_cols": added_cols,
        "filled_nonnull_counts": filled,
        "rows": int(len(out)),
        "cols": int(len(out.columns)),
        "audit_before": {c: {"nn": int(before_audit[c][0]), "total": int(before_audit[c][1]), "rate": float(before_audit[c][2])} for c in before_audit},
        "audit_after": {c: {"nn": int(after_audit[c][0]), "total": int(after_audit[c][1]), "rate": float(after_audit[c][2])} for c in after_audit},
    }


def discover_leagues(root: Path) -> list[str]:
    matches_dir = root / "Matches"
    merged_dir = matches_dir / "__merged__"
    leagues = []

    if not merged_dir.exists():
        return []

    # derive league names from merged filenames, and map back to folder names by tag match
    merged_files = sorted(merged_dir.glob("*__merged.csv"))
    # build tag->folder lookup
    folders = [p for p in matches_dir.iterdir() if p.is_dir() and p.name != "__merged__"]
    tag_to_folder = {league_tag(p.name): p.name for p in folders}

    for mp in merged_files:
        tag = mp.name.replace("__merged.csv", "")
        if tag in tag_to_folder:
            leagues.append(tag_to_folder[tag])
        else:
            # fallback: best-effort reverse
            leagues.append(tag.replace("_", " "))
    # stable unique
    seen = set()
    out = []
    for x in leagues:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Repo root, e.g. /Users/hughwade/Documents/Code/OG_master/BAWA PORTO")
    ap.add_argument("--leagues", default="", help="Comma-separated league folder names (optional)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing merged values with synth values")
    ap.add_argument("--harmonize-duplicates", action="store_true", help="For duplicate fixture_key (or join key) rows in merged files, propagate a single chosen value across all duplicates for the patched columns.")
    ap.add_argument("--audit-dup-variance", action="store_true", help="Print a per-league report of how many fixture_keys have multiple distinct odds_ft_under25 values after patching (helps detect duplicate drift).")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"root does not exist: {root}")

    if args.leagues.strip():
        leagues = [x.strip() for x in args.leagues.split(",") if x.strip()]
    else:
        leagues = discover_leagues(root)

    print(f"patch cols requested (synth): {PATCH_COLS_SYNTH}")
    print(f"patch cols requested (novig): {PATCH_COLS_OU25_NOVIG}")
    print(f"overwrite={bool(args.overwrite)}")
    print(f"leagues={len(leagues)}")

    ok = 0
    bad = 0
    for lg in leagues:
        res = patch_one_league(root, lg, overwrite=bool(args.overwrite), harmonize_duplicates=bool(args.harmonize_duplicates), audit_dup_variance=bool(args.audit_dup_variance))
        if res.get("ok"):
            ok += 1
            b = res.get("audit_before", {}).get("odds_ft_under25", {})
            a = res.get("audit_after", {}).get("odds_ft_under25", {})
            b_pct = (float(b.get("rate", 0.0)) * 100.0) if b else 0.0
            a_pct = (float(a.get("rate", 0.0)) * 100.0) if a else 0.0
            suffix = ""
            if args.harmonize_duplicates:
                suffix += " | harmonized_dups=YES"
            if args.audit_dup_variance:
                suffix += " | audit_dup=YES"
            print(
                f"✅ {lg}: join={res['join']} added={len(res['added_cols'])} rows={res['rows']} cols={res['cols']} "
                f"| under25_nonnull: {b_pct:5.1f}% -> {a_pct:5.1f}%{suffix}"
            )
        else:
            bad += 1
            print(f"❌ {lg}: {res.get('reason')}")

    print(f"\nDONE ok={ok} bad={bad}")


if __name__ == "__main__":
    main()