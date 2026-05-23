from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Dict, Any, Iterable

import numpy as np
import pandas as pd

from prediction_overlay import _match_key, _coalesce_match_date_series
import re

try:
    from constants import FTR_MARGIN_MIN_PER_LEAGUE as _FTR_MARGIN_MIN_PER_LEAGUE  # type: ignore
except Exception:
    _FTR_MARGIN_MIN_PER_LEAGUE = {}

def _resolve_ftr_margin_min_local(league: object, default: float = 0.0) -> float:
    """Resolve per-league FTR margin gate.

    Precedence:
      1) DEPLOY_GATES_FTR.csv (env: DEPLOY_GATES_FTR_PATH) column `ftr_gate_min_margin` if present and finite.
      2) constants.FTR_MARGIN_MIN_PER_LEAGUE (imported as _FTR_MARGIN_MIN_PER_LEAGUE).

    Safety:
      - Clamp to [0, FTR_MARGIN_MAX_CAP] where FTR_MARGIN_MAX_CAP defaults to 0.15.
        This prevents stale constants (e.g. 0.55/0.80) from nuking whole leagues.
    """
    try:
        cap = float(os.getenv("FTR_MARGIN_MAX_CAP", "0.15"))
    except Exception:
        cap = 0.15
    cap = max(0.0, cap)

    def _clamp(v: float) -> float:
        try:
            fv = float(v)
        except Exception:
            fv = float(default)
        if not np.isfinite(fv):
            fv = float(default)
        fv = max(0.0, fv)
        if cap > 0:
            fv = min(fv, cap)
        return float(fv)

    key = str(league).strip() if league is not None else ""
    if not key:
        return _clamp(default)

    # 1) Prefer deploy-gates margin if available
    try:
        from pathlib import Path as _Path
        p = _Path(os.getenv("DEPLOY_GATES_FTR_PATH", os.path.join("predictions_output", "DEPLOY_GATES_FTR.csv")))
        if p.exists():
            g = pd.read_csv(p)
            if "league" in g.columns and "ftr_gate_min_margin" in g.columns:
                r = g[g["league"].astype(str).str.strip() == key].head(1)
                if len(r):
                    v = pd.to_numeric(r.iloc[0].get("ftr_gate_min_margin"), errors="coerce")
                    if np.isfinite(v):
                        return _clamp(float(v))
    except Exception:
        pass

    # 2) Fall back to constants
    try:
        tag = re.sub(r"[^A-Za-z0-9_]+", "_", key).strip("_")
        for k in (key, tag, key.replace(" ", "_"), tag.replace("_", " ")):
            if k in _FTR_MARGIN_MIN_PER_LEAGUE:
                fv = float(_FTR_MARGIN_MIN_PER_LEAGUE.get(k))
                if np.isfinite(fv):
                    return _clamp(fv)
    except Exception:
        pass

    return _clamp(default)

def _attach_ftr_margin_local(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or getattr(df, "empty", True):
        return df
    if not all(c in df.columns for c in ("confidence_home", "confidence_draw", "confidence_away")):
        return df
    try:
        P = df[["confidence_home", "confidence_draw", "confidence_away"]].to_numpy(dtype=float)
        s = np.sort(P, axis=1)
        df["ftr_margin"] = (s[:, 2] - s[:, 1]).astype(float)
    except Exception:
        pass
    return df
# ----------------------------------------------------------------------
# Realised-fixture guards (exclude placeholders / future rows)
# ----------------------------------------------------------------------
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
            cand = pd.to_datetime(df[col], errors="coerce", utc=True)
        except Exception:
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

    # goals present (numeric)
    hg = pd.to_numeric(df.get("home_team_goal_count", pd.Series(np.nan, index=idx)), errors="coerce")
    ag = pd.to_numeric(df.get("away_team_goal_count", pd.Series(np.nan, index=idx)), errors="coerce")
    goals_present = ~(hg.isna() | ag.isna())

    status_txt = (
        df["status"].astype("string").fillna("").str.strip().str.lower()
        if "status" in df.columns
        else pd.Series("", index=idx, dtype="string")
    )
    st_complete = _status_is_complete(status_txt) if "status" in df.columns else pd.Series(False, index=idx)
    st_incomp   = _status_is_incomplete(status_txt) if "status" in df.columns else pd.Series(False, index=idx)

    # Only treat status as "known" if it hits either complete or incomplete regex
    st_known = status_txt.ne("") & (st_complete | st_incomp)

    # Need a parseable date for fallback path (avoids undated 0-0 garbage)
    dt = pd.Series(pd.NaT, index=idx, dtype="datetime64[ns, UTC]")
    for col in ("match_date", "date_GMT", "date", "timestamp"):
        if col not in df.columns:
            continue
        try:
            cand = pd.to_datetime(df[col], errors="coerce", utc=True)
        except Exception:
            cand = pd.to_datetime(df[col].astype(str), errors="coerce", utc=True)
        dt = dt.fillna(cand)
    date_known = dt.notna()

    # completed if:
    #  - status says complete
    #  - OR status is unknown/blank AND goals are present AND date is known
    base = (st_complete | ((~st_known) & goals_present & date_known)) & goals_present

    # never treat “incomplete” as realised; never allow future rows
    future = _future_fixture_mask(df)
    return base & ~st_incomp & ~future


def _pick_latest_matches_csv(matches_root: Path, league: str) -> Path | None:
    mdir = matches_root / league
    if not mdir.exists():
        return None
    cands = sorted(mdir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def _load_matches_with_fixture_key(matches_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(matches_csv)

    # Ensure match_date & team names
    if "match_date" not in df.columns:
        df["match_date"] = ""
    df["match_date"] = _coalesce_match_date_series(df)
    df["match_date_dt"] = pd.to_datetime(df["match_date"], errors="coerce")

    if "home_team_name" not in df.columns and "Home" in df.columns:
        df["home_team_name"] = df["Home"]
    if "away_team_name" not in df.columns and "Away" in df.columns:
        df["away_team_name"] = df["Away"]

    # fixture_key must match overlay/meta_builder
    df["fixture_key"] = df.apply(_match_key, axis=1)

    # numeric goals
    for c in ("home_team_goal_count", "away_team_goal_count"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan

    return df


def _derive_real_ftr_label(df: pd.DataFrame) -> pd.Series:
    """Return HOME/DRAW/AWAY realised outcome label from goals."""
    hg = pd.to_numeric(df["home_team_goal_count"], errors="coerce")
    ag = pd.to_numeric(df["away_team_goal_count"], errors="coerce")
    diff = hg - ag
    lab = np.where(diff > 0, "HOME", np.where(diff < 0, "AWAY", "DRAW"))
    return pd.Series(lab, index=df.index)


def _band_confidence(conf: float) -> str:
    if not np.isfinite(conf):
        return "nan"
    if conf < 0.45:
        return "<0.45"
    if conf < 0.50:
        return "0.45–0.50"
    if conf < 0.55:
        return "0.50–0.55"
    if conf < 0.60:
        return "0.55–0.60"
    if conf < 0.65:
        return "0.60–0.65"
    return "≥0.65"


def _fixture_key_series(df: pd.DataFrame) -> pd.Series:
    """Stable fixture key: YYYY-MM-DD + home + away (slugged)."""
    md = df.get("match_date")
    if md is None:
        md = pd.Series([""] * len(df), index=df.index, dtype="string")
    else:
        try:
            md = pd.to_datetime(md, errors="coerce")
            md = md.dt.strftime("%Y-%m-%d").fillna("")
        except Exception:
            md = md.astype("string").fillna("").str.strip()

    h = df.get("home_team_name", df.get("Home", ""))
    a = df.get("away_team_name", df.get("Away", ""))
    h = h.astype("string").fillna("").str.strip()
    a = a.astype("string").fillna("").str.strip()

    raw = (md.astype("string") + "_" + h + "_" + a).astype("string").fillna("")
    return raw.str.replace(r"[^A-Za-z0-9_]+", "_", regex=True).str.strip("_")


def _load_all_matches_csvs(matches_root: Path, league: str) -> pd.DataFrame:
    """Read + concat ALL CSVs under Matches/<League>/ (multi-season), then dedupe by fixture_key."""
    mdir = matches_root / league
    if not mdir.exists():
        return pd.DataFrame()

    cands = sorted(mdir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=False)
    if not cands:
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for p in cands:
        try:
            df = pd.read_csv(p)
            df["__src_csv"] = p.name
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, axis=0, ignore_index=True, sort=False)

    # Ensure match_date + team columns, then fixture_key
    if "match_date" not in df.columns:
        df["match_date"] = ""
    df["match_date"] = _coalesce_match_date_series(df)

    # Ensure match_date_dt exists (used to bound realised window)
    try:
        df["match_date_dt"] = pd.to_datetime(df["match_date"], errors="coerce")
    except Exception:
        df["match_date_dt"] = pd.NaT

    if "home_team_name" not in df.columns and "Home" in df.columns:
        df["home_team_name"] = df["Home"]
    if "away_team_name" not in df.columns and "Away" in df.columns:
        df["away_team_name"] = df["Away"]

    df["fixture_key"] = _fixture_key_series(df)

    # Ensure goal columns exist for scoring
    for c in ("home_team_goal_count", "away_team_goal_count"):
        if c not in df.columns:
            df[c] = np.nan

    # De-dupe: prefer the most reliable row per fixture_key.
    nn = df.notna().sum(axis=1)

    status_txt = (
        df["status"].astype("string").fillna("").str.strip().str.lower()
        if "status" in df.columns else pd.Series("", index=df.index, dtype="string")
    )
    status_complete = _status_is_complete(status_txt) if "status" in df.columns else pd.Series(False, index=df.index)

    goals_present = df[["home_team_goal_count", "away_team_goal_count"]].notna().all(axis=1)

    future = _future_fixture_mask(df)

    score = (status_complete.astype(int) * 1000) + (goals_present.astype(int) * 100) + nn - (future.astype(int) * 5000)

    df = df.assign(__score=score).sort_values(["fixture_key", "__score"], ascending=[True, False])
    df = df.drop_duplicates(subset=["fixture_key"], keep="first").drop(columns=["__score"], errors="ignore")

    # Ensure numeric goals
    for c in ("home_team_goal_count", "away_team_goal_count"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def backtest_ftr_meta(
    meta_path: Path,
    matches_root: Path,
    leagues: Iterable[str] | None = None,
    *,
    multi_season: bool = False,
    use_margin_gate: bool = False,
    margin_min: float | None = None,
) -> pd.DataFrame:
    meta = pd.read_csv(meta_path)

    # Filter to FTR rows
    ftr = meta[meta["market"] == "ftr"].copy()
    if ftr.empty:
        print("⚠️ No FTR rows in META; nothing to backtest.")
        return pd.DataFrame()

    ftr["confidence"] = pd.to_numeric(ftr["confidence"], errors="coerce")
    ftr["match_date_dt"] = pd.to_datetime(ftr.get("match_date"), errors="coerce")

    if leagues is None:
        leagues = sorted(ftr["league"].dropna().unique().tolist())

    rows: List[Dict[str, Any]] = []
    for league in leagues:
        sub = ftr[ftr["league"] == league].copy()
        if sub.empty:
            print(f"ℹ️ FTR META: no rows for league={league}")
            continue

        # Optional margin gate (so backtest matches deploy behaviour)
        if bool(use_margin_gate):
            sub = _attach_ftr_margin_local(sub)
            mm = float(margin_min) if margin_min is not None else _resolve_ftr_margin_min_local(league, 0.0)

            if mm and ("ftr_margin" in sub.columns):
                before_sub = len(sub)
                sub_g = sub[pd.to_numeric(sub["ftr_margin"], errors="coerce").fillna(0.0) >= float(mm)].copy()

                # If margin gate wipes the league, fall back to margin=0.0 so we still score this league.
                if sub_g.empty:
                    print(
                        f"ℹ️ No FTR META picks for {league} after margin gate (min_margin={float(mm):.3f}); "
                        f"retrying with margin=0.000"
                    )
                    mm = 0.0
                else:
                    sub = sub_g
                    print(f"🧱 FTR META margin gate {league}: min_margin={float(mm):.3f} kept {len(sub)}/{before_sub}")

        if multi_season:
            matches = _load_all_matches_csvs(matches_root, league)
            if matches is None or matches.empty:
                print(f"⚠️ No matches CSVs for {league} under {matches_root}")
                continue
        else:
            csvp = _pick_latest_matches_csv(matches_root, league)
            if csvp is None:
                print(f"⚠️ No matches CSV for {league} under {matches_root}")
                continue
            matches = _load_matches_with_fixture_key(csvp)

        # Guard: exclude placeholder rows (e.g., status=incomplete with 0-0) and any future fixtures
        before_m = len(matches)
        matches = matches.loc[_realised_mask(matches)].copy()
        try:
            print(f"🧹 {league}: realised-only matches kept {len(matches)}/{before_m}")
        except Exception:
            pass

        # Ensure match_date_dt exists for downstream date bounding (multi-season loader may not have created it)
        try:
            matches["match_date_dt"] = pd.to_datetime(matches.get("match_date"), errors="coerce")
        except Exception:
            matches["match_date_dt"] = pd.NaT

        if matches.empty:
            print(f"ℹ️ {league}: no realised fixtures after status/future guards; skipping.")
            continue

        # Only evaluate picks for fixtures that have realised results
        real_mask = matches["home_team_goal_count"].notna() & matches["away_team_goal_count"].notna()
        if not real_mask.any():
            print(f"ℹ️ {league}: no completed fixtures with goals in matches CSV; skipping backtest.")
            continue
        # Robust: derive from match_date if match_date_dt is missing for any reason
        if "match_date_dt" in matches.columns:
            last_real_date = matches.loc[real_mask, "match_date_dt"].max()
        else:
            last_real_date = pd.to_datetime(matches.loc[real_mask, "match_date"], errors="coerce").max()
        if pd.isna(last_real_date):
            print(f"ℹ️ {league}: could not determine last realised match_date; skipping backtest.")
            continue

        # Restrict META FTR picks to fixtures on or before last_real_date
        sub = sub.copy()
        if "match_date_dt" in sub.columns:
            sub = sub[sub["match_date_dt"] <= last_real_date]
        else:
            sub = sub
        if sub.empty:
            print(f"ℹ️ No FTR META picks for {league} on or before {last_real_date.date()}; skipping.")
            continue

        # join on fixture_key
        joined = sub.merge(
            matches[["fixture_key", "home_team_goal_count", "away_team_goal_count"]],
            on="fixture_key",
            how="left",
            suffixes=("", "_real"),
        )

        # Drop rows without realised results
        mask_real = joined["home_team_goal_count"].notna() & joined["away_team_goal_count"].notna()
        joined = joined[mask_real].copy()
        if joined.empty:
            print(f"ℹ️ No realised results for FTR META picks in {league}")
            continue

        joined["real_outcome"] = _derive_real_ftr_label(joined)

        # correctness
        joined["correct"] = (joined["selection"].astype(str).str.upper() == joined["real_outcome"].astype(str).str.upper())
        joined["conf_band"] = joined["confidence"].apply(_band_confidence)

        # Per-band stats
        for band, gdf in joined.groupby("conf_band"):
            n = len(gdf)
            if n == 0:
                continue
            hit = float(gdf["correct"].mean())
            avg_conf = float(gdf["confidence"].mean())
            avg_od = float(pd.to_numeric(gdf["od"], errors="coerce").mean())
            rows.append({
                "league": league,
                "conf_band": band,
                "n": int(n),
                "hit_rate": hit,
                "avg_confidence": avg_conf,
                "avg_odds": avg_od,
            })

        # Overall row for league
        rows.append({
            "league": league,
            "conf_band": "ALL",
            "n": int(len(joined)),
            "hit_rate": float(joined["correct"].mean()),
            "avg_confidence": float(joined["confidence"].mean()),
            "avg_odds": float(pd.to_numeric(joined["od"], errors="coerce").mean()),
        })

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows)
    return df_out


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest FTR META picks vs realised results.")
    ap.add_argument("--meta-path", required=True, help="Path to META_candidates_...csv")
    ap.add_argument("--matches-root", default="Matches", help="Root matches folder")
    ap.add_argument("--leagues", default=None, help='Comma-separated leagues; defaults to distinct leagues in META')
    ap.add_argument("--multi-season", action="store_true", help="Load ALL CSVs under Matches/<League>/ and dedupe by fixture_key")
    ap.add_argument(
    "--use-margin-gate",
    action="store_true",
    help="Apply per-league FTR margin gate (top1-top2) before backtest stats",
    )
    ap.add_argument(
        "--margin-min",
        type=float,
        default=None,
        help="Override margin min for all leagues. If omitted, uses constants.FTR_MARGIN_MIN_PER_LEAGUE (else 0.0).",
    )
    args = ap.parse_args()
    meta_path = Path(args.meta_path)
    matches_root = Path(args.matches_root)

    if args.leagues:
        leagues = [s.strip() for s in str(args.leagues).split(",") if s.strip()]
    else:
        meta = pd.read_csv(meta_path)
        leagues = sorted(meta["league"].dropna().unique().tolist())

    df_stats = backtest_ftr_meta(
        meta_path,
        matches_root,
        leagues,
        multi_season=bool(args.multi_season),
        use_margin_gate=bool(args.use_margin_gate),
        margin_min=args.margin_min,
    )

    if df_stats.empty:
        print("⚠️ No FTR META backtest stats produced.")
        return

    print("\nFTR META backtest stats:")
    print(df_stats.to_string(index=False))

    out_dir = Path("predictions_output")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "FTR_META_backtest_stats.csv"
    df_stats.to_csv(out_path, index=False)
    print(f"\n📁 Wrote FTR META backtest CSV → {out_path}")


if __name__ == "__main__":
    main()