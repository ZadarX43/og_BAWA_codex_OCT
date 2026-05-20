#!/usr/bin/env python3
"""make_fd_odds_enriched_synth.py

Build `fd_odds_enriched_synth.csv` per league under Matches/<League>/ by synthesizing
missing *Under 2.5* odds (and optionally BTTS NO) from the known side using a
league-specific bin rulebook fitted on historical rows where BOTH sides exist.

Key outputs (added/filled):
  - odds_ft_under25 (filled when missing)
  - odds_source_under25 = {"existing","synth"}
  - synth_u25_method, synth_u25_conf

  - odds_btts_no (filled when missing, optional)
  - odds_source_btts_no = {"existing","synth"}
  - synth_bttsno_method, synth_bttsno_conf

  - Optionally emits `fd_ou25_novig.csv` containing no-vig probabilities for the paired O2.5/U2.5 market.

SAFETY:
  Before writing the synth CSV, we restore *real* odds columns from the base
  `fd_odds_enriched.csv` (same folder) when a synth column looks broken (all zeros
  or no values > 1). Otherwise we only fill missing/non-odds cells.

SOURCE HYGIENE:
  You can exclude certain __src_csv origins from *fitting* the rulebooks (e.g. legacy
  “*_all_seasons” or “__TRAIN_MULTISEASON” files) while still applying synthesis to
  the full dataset. This prevents dirty merges from teaching the synth model.

This file must NOT contain any bookie_allmarkets logic.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd


# -------------------------
# Fit-set hygiene filtering
# -------------------------

def _filter_fit_rows(df: pd.DataFrame, exclude_src_substrings: List[str]) -> pd.DataFrame:
    """Return a filtered frame for fitting rulebooks.

    We only filter the *fit* set, never the output/synthesis frame.
    Filtering is based on __src_csv substring matches.
    """
    if df is None or df.empty:
        return df
    if not exclude_src_substrings:
        return df
    if "__src_csv" not in df.columns:
        return df

    src = df["__src_csv"].astype("string").fillna("")
    m = pd.Series(True, index=df.index)
    for sub in exclude_src_substrings:
        s = str(sub or "").strip()
        if not s:
            continue
        m &= ~src.str.contains(s, case=False, na=False)
    return df.loc[m].copy()


# -------------------------
# Helpers: fixture key
# -------------------------

def _coalesce_match_date_series(df: pd.DataFrame) -> pd.Series:
    for c in ("match_date", "date_GMT", "date", "Date", "timestamp"):
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


def _match_key(row: pd.Series) -> str:
    md = pd.to_datetime(row.get("match_date", None), errors="coerce", utc=True)
    ds = md.strftime("%Y_%m_%d") if pd.notna(md) else ""
    h = _norm_team_token(row.get("home_team_name", row.get("HomeTeam", row.get("Home", ""))))
    a = _norm_team_token(row.get("away_team_name", row.get("AwayTeam", row.get("Away", ""))))
    return f"{ds}_{h}_{a}".strip("_")


# -------------------------
# Bin rulebook synthesis
# -------------------------


def _safe_odds(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    # treat <=1 or 0 as missing
    x = x.where(x > 1.0, np.nan)
    return x


# -------------------------
# O/U2.5 no-vig helpers
# -------------------------

def _novig_from_two_way(od_a: pd.Series, od_b: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return (p_a_novig, p_b_novig, overround) for a two-way market.

    Inputs are odds series; invalid odds (<=1) should already be NaN.
    """
    ia = (1.0 / pd.to_numeric(od_a, errors="coerce")).astype(float)
    ib = (1.0 / pd.to_numeric(od_b, errors="coerce")).astype(float)
    over = (ia + ib).astype(float)
    pa = (ia / over).astype(float)
    pb = (ib / over).astype(float)
    pa = pa.where(np.isfinite(pa), np.nan)
    pb = pb.where(np.isfinite(pb), np.nan)
    over = over.where(np.isfinite(over), np.nan)
    return pa, pb, over


def _emit_fd_ou25_novig(
    league_dir: Path,
    df: pd.DataFrame,
    *,
    conf_gate: float = 0.0,
    future_only: bool = False,
    debug: bool = False,
) -> Dict[str, Any]:
    """Write `fd_ou25_novig.csv` for rows where O2.5 and U2.5 odds are paired.

    - Only uses rows where odds_ft_over25 and odds_ft_under25 are both present (>1.0).
    - If conf_gate > 0 and synth_u25_conf exists, we require synth_u25_conf >= conf_gate for rows where under25 was synthesized.
      (Existing under25 rows are always allowed.)
    - If future_only is True, restrict to match_date >= today (UTC-normalized).

    Returns a small diagnostics dict.
    """
    out = df.copy()

    # robust date parsing (force UTC, then drop tz so comparisons are always valid)
    md_raw = out.get("match_date", pd.Series(pd.NA, index=out.index))
    md = pd.to_datetime(md_raw, errors="coerce", utc=True)
    # Convert to tz-naive datetime64[ns] for safe vectorized comparisons
    try:
        md = md.dt.tz_convert(None)
    except Exception:
        # if already tz-naive
        pass

    today = pd.Timestamp.utcnow()
    # Ensure `today` is tz-naive to match `md`
    try:
        if getattr(today, "tzinfo", None) is not None:
            today = today.tz_convert(None)
    except Exception:
        pass
    today = today.normalize()

    future_mask = md >= today

    # Allow for column name drift across leagues/sources
    def _first_series(cols: List[str]) -> pd.Series:
        for c in cols:
            if c in out.columns:
                return out[c]
        return pd.Series(np.nan, index=out.index)

    o_raw = _first_series(["odds_ft_over25", "odds_ft_over_25", "odds_over25", "odds_o25", "od_over25", "odds_ft_o25"])
    u_raw = _first_series(["odds_ft_under25", "odds_ft_under_25", "odds_under25", "odds_u25", "od_under25", "odds_ft_u25"])

    o = _safe_odds(o_raw)
    u = _safe_odds(u_raw)

    paired = o.notna() & u.notna()

    # If nothing is paired, emit an empty CSV with headers and return diagnostics.
    # (This keeps downstream steps deterministic.)
    if int(paired.sum()) == 0:
        out_path = league_dir / "fd_ou25_novig.csv"
        cols_keep = [
            "match_date",
            "date_GMT",
            "home_team_name",
            "away_team_name",
            "fixture_key",
            "odds_ft_over25",
            "odds_ft_under25",
            "odds_source_under25",
            "synth_u25_conf",
            "synth_u25_method",
        ]
        keep = [c for c in cols_keep if c in out.columns]
        out2 = out.loc[[], keep].copy()
        out2["p_over25_novig"] = pd.Series(dtype=float)
        out2["p_under25_novig"] = pd.Series(dtype=float)
        out2["ou25_overround"] = pd.Series(dtype=float)
        out2.to_csv(out_path, index=False)

        diag = {
            "rows_written": 0,
            "future_rows": int(future_mask.sum()),
            "total_o25_present": int(o.notna().sum()),
            "total_u25_present": int(u.notna().sum()),
            "total_paired": 0,
            "future_paired": 0,
            "future_neither": int((future_mask & (o.isna() & u.isna())).sum()),
            "conf_gate": float(conf_gate),
            "future_only": bool(future_only),
        }
        if debug:
            print(
                f"🧾 {league_dir.name}: WROTE {out_path} rows=0 | "
                f"present_o25={diag['total_o25_present']} present_u25={diag['total_u25_present']} paired=0 | "
                f"future_only={future_only} conf_gate={conf_gate}"
            )
        return diag

    # optional confidence gate for synthesized rows
    if conf_gate and conf_gate > 0 and "odds_source_under25" in out.columns and "synth_u25_conf" in out.columns:
        src = out["odds_source_under25"].astype("string").fillna("")
        conf = pd.to_numeric(out["synth_u25_conf"], errors="coerce")
        # allow existing always; require conf for synth
        conf_ok = (src.ne("synth")) | (conf >= float(conf_gate))
        paired = paired & conf_ok

    if future_only:
        paired = paired & future_mask

    p_over, p_under, overround = _novig_from_two_way(o.where(paired), u.where(paired))

    cols_keep = [
        "match_date",
        "date_GMT",
        "home_team_name",
        "away_team_name",
        "fixture_key",
        "odds_ft_over25",
        "odds_ft_under25",
        "odds_source_under25",
        "synth_u25_conf",
        "synth_u25_method",
    ]
    keep = [c for c in cols_keep if c in out.columns]

    out2 = out.loc[paired, keep].copy()
    out2["p_over25_novig"] = p_over.loc[out2.index]
    out2["p_under25_novig"] = p_under.loc[out2.index]
    out2["ou25_overround"] = overround.loc[out2.index]

    # write
    out_path = league_dir / "fd_ou25_novig.csv"
    out2.to_csv(out_path, index=False)

    # diagnostics
    diag = {
        "rows_written": int(len(out2)),
        "future_rows": int(future_mask.sum()),
        "total_o25_present": int(o.notna().sum()),
        "total_u25_present": int(u.notna().sum()),
        "total_paired": int((o.notna() & u.notna()).sum()),
        "future_paired": int((future_mask & (o.notna() & u.notna())).sum()),
        "future_neither": int((future_mask & (o.isna() & u.isna())).sum()),
        "conf_gate": float(conf_gate),
        "future_only": bool(future_only),
    }
    if debug:
        print(
            f"🧾 {league_dir.name}: WROTE {out_path} rows={diag['rows_written']} | "
            f"present_o25={diag['total_o25_present']} present_u25={diag['total_u25_present']} paired={diag['total_paired']} | "
            f"future_only={future_only} conf_gate={conf_gate}"
        )
    return diag


def _quantile(x: pd.Series, q: float) -> float:
    try:
        v = float(x.quantile(q))
        return v if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def _build_quantile_bins(p_known: pd.Series, n_bins: int) -> Tuple[np.ndarray, pd.Series]:
    pk = pd.to_numeric(p_known, errors="coerce")
    pk = pk[pk.notna()]
    if pk.empty:
        return np.array([]), p_known

    # jitter to avoid duplicated edges
    rng = np.random.default_rng(7)
    pk2 = pk.to_numpy(dtype=float) + rng.normal(0, 1e-8, size=len(pk))
    qs = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.unique(np.quantile(pk2, qs))
    if len(edges) < 3:
        edges = np.array([float(pk2.min() - 1e-6), float(pk2.max() + 1e-6)])
    # return edges and original series
    return edges.astype(float), p_known


def _fit_bin_rulebook(
    df: pd.DataFrame,
    known_col: str,
    miss_col: str,
    *,
    n_bins: int = 10,
    min_n: int = 200,
) -> List[Dict[str, Any]]:
    """Fit a bin rulebook for missing odds from known-side implied.

    Stores per bin:
      - p_lo/p_hi (range of known implied)
      - R_median, R_q25, R_q75 (overround distribution for the 2-way market)
      - miss_q10, miss_q90 (missing-odds clamp range)
      - n
      - iqr (R_q75 - R_q25)
    """
    if df is None or df.empty:
        return []

    ok = _safe_odds(df.get(known_col, pd.Series(np.nan, index=df.index)))
    om = _safe_odds(df.get(miss_col, pd.Series(np.nan, index=df.index)))
    m = ok.notna() & om.notna()
    if int(m.sum()) < int(min_n):
        return []

    pk = 1.0 / ok.loc[m]
    pm = 1.0 / om.loc[m]
    R = pk + pm

    edges, pk_all = _build_quantile_bins(pk, int(n_bins))
    if edges.size < 2:
        return []

    # assign bins on pk
    cats = pd.cut(pk, bins=edges, include_lowest=True, duplicates="drop")
    by_bin = pd.Series(cats, index=pk.index)

    bins: List[Dict[str, Any]] = []

    # threshold for per-bin support
    min_bin = max(30, int(min_n * 0.05))

    for b, idx in by_bin.groupby(by_bin, observed=False).groups.items():
        sel = list(idx)
        n = int(len(sel))
        if n < min_bin:
            continue

        pk_b = pk.loc[sel]
        R_b = R.loc[sel]
        om_b = om.loc[sel]

        R_q25 = _quantile(R_b, 0.25)
        R_q75 = _quantile(R_b, 0.75)
        iqr = (R_q75 - R_q25) if (np.isfinite(R_q25) and np.isfinite(R_q75)) else float("nan")

        bins.append(
            {
                "n": n,
                "p_lo": float(pk_b.min()),
                "p_hi": float(pk_b.max()),
                "R_median": float(_quantile(R_b, 0.50)),
                "R_q25": float(R_q25),
                "R_q75": float(R_q75),
                "iqr": float(iqr) if np.isfinite(iqr) else float("nan"),
                "miss_q10": float(_quantile(om_b, 0.10)),
                "miss_q90": float(_quantile(om_b, 0.90)),
            }
        )

    bins = sorted(bins, key=lambda d: float(d.get("p_lo", 0.0)))
    return bins


def _conf_from_bin(b: Dict[str, Any]) -> float:
    """Conservative confidence score 0..1 from bin support + IQR tightness."""
    try:
        n = int(b.get("n", 0))
    except Exception:
        n = 0
    if n <= 0:
        return 0.0

    iqr = float(b.get("iqr", np.nan))

    # n factor saturates
    n_factor = min(1.0, math.log1p(n) / math.log1p(200.0))

    if not np.isfinite(iqr):
        return float(0.10 + 0.30 * n_factor)

    # typical OU25 overround IQR is small (~0.01-0.03). Larger = less stable.
    iqr_scale = 0.035
    iqr_factor = max(0.0, min(1.0, 1.0 - (iqr / iqr_scale)))

    conf = 0.10 + 0.90 * (0.55 * n_factor + 0.45 * iqr_factor)
    return float(max(0.0, min(1.0, conf)))


def _choose_bin(bins: List[Dict[str, Any]], p_known: float) -> Tuple[int, Optional[Dict[str, Any]]]:
    if not bins:
        return -1, None
    for i, b in enumerate(bins):
        lo = float(b.get("p_lo", -1.0))
        hi = float(b.get("p_hi", 2.0))
        if p_known >= lo and p_known <= hi:
            return i, b

    # out of range -> nearest by center
    best_i, best_b, best_d = 0, bins[0], float("inf")
    for i, b in enumerate(bins):
        lo = float(b.get("p_lo", 0.0))
        hi = float(b.get("p_hi", 0.0))
        ctr = 0.5 * (lo + hi)
        d = abs(p_known - ctr)
        if d < best_d:
            best_i, best_b, best_d = i, b, d
    return best_i, best_b


def _synth_missing_from_known(
    known_od: float,
    bins: List[Dict[str, Any]],
    *,
    conf_min: float = 0.10,
) -> Tuple[float, str, float]:
    """Return (odds_hat, method, conf)."""
    try:
        known_od = float(known_od)
    except Exception:
        return (float("nan"), "", float("nan"))
    if not np.isfinite(known_od) or known_od <= 1.0:
        return (float("nan"), "", float("nan"))

    p_known = 1.0 / known_od
    bi, b = _choose_bin(bins, p_known)
    if b is None:
        return (float("nan"), "", float("nan"))

    conf = _conf_from_bin(b)
    if conf < float(conf_min):
        return (float("nan"), "drop_low_conf", float(conf))

    R_hat = float(b.get("R_median", np.nan))
    if not np.isfinite(R_hat) or R_hat <= 0:
        return (float("nan"), "drop_bad_R", float(conf))

    p_miss = R_hat - p_known
    if not np.isfinite(p_miss) or p_miss <= 1e-6:
        return (float("nan"), "drop_p_nonpositive", float(conf))

    od_hat = 1.0 / p_miss
    method = "bin_overround"

    # clamp to missing-odds distribution
    q10 = float(b.get("miss_q10", np.nan))
    q90 = float(b.get("miss_q90", np.nan))
    if np.isfinite(q10) and np.isfinite(q90) and (q90 > q10 > 1.0):
        if od_hat > q90:
            od_hat = q90
            method = "bin_overround_clamp_hi"
        elif od_hat < q10:
            od_hat = q10
            method = "bin_overround_clamp_lo"

    return (float(od_hat), str(method), float(conf))


# -------------------------
# League runner
# -------------------------

def _ensure_date_and_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "match_date" not in out.columns:
        out["match_date"] = pd.NA
    out["match_date"] = _coalesce_match_date_series(out)
    try:
        out["match_date"] = pd.to_datetime(out["match_date"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
    except Exception:
        pass

    if "home_team_name" not in out.columns and "Home" in out.columns:
        out["home_team_name"] = out["Home"]
    if "away_team_name" not in out.columns and "Away" in out.columns:
        out["away_team_name"] = out["Away"]

    if "fixture_key" not in out.columns or out["fixture_key"].isna().all():
        try:
            out["fixture_key"] = out.apply(_match_key, axis=1)
        except Exception:
            out["fixture_key"] = ""

    out["fixture_key"] = out["fixture_key"].astype("string").fillna("").str.strip()
    return out


def _process_one_league(
    league_dir: Path,
    *,
    n_bins: int,
    min_n: int,
    conf_min: float,
    synth_btts_no: bool,
    debug: bool,
    exclude_src: List[str],
    emit_ou25_novig: bool,
    ou25_conf_gate: float,
    ou25_future_only: bool,
) -> Optional[Dict[str, Any]]:
    base_path = league_dir / "fd_odds_enriched.csv"
    if not base_path.exists():
        return None

    df0 = pd.read_csv(base_path, low_memory=False)
    df0 = _ensure_date_and_key(df0)

    # Fit-set hygiene: optionally exclude dirty source CSVs from teaching the rulebooks
    df_fit = _filter_fit_rows(df0, exclude_src)

    # --- Fit rulebooks (only if enough support) ---
    rb_ou = _fit_bin_rulebook(df_fit, "odds_ft_over25", "odds_ft_under25", n_bins=n_bins, min_n=min_n)
    rb_bt = _fit_bin_rulebook(df_fit, "odds_btts_yes", "odds_btts_no", n_bins=n_bins, min_n=min_n) if synth_btts_no else []

    # quick RB diagnostics
    def _rb_diag(rb: List[Dict[str, Any]]) -> Tuple[int, float, float]:
        if not rb:
            return (0, float("nan"), float("nan"))
        iqrs = [float(b.get("iqr", np.nan)) for b in rb]
        iqrs = [x for x in iqrs if np.isfinite(x)]
        if not iqrs:
            return (len(rb), float("nan"), float("nan"))
        return (len(rb), float(np.nanmedian(iqrs)), float(np.nanpercentile(iqrs, 90)))

    ou_bins, ou_iqr_med, ou_iqr_p90 = _rb_diag(rb_ou)
    bt_bins, bt_iqr_med, bt_iqr_p90 = _rb_diag(rb_bt)

    if debug:
        print(
            f"ℹ️  {league_dir.name}: fit_rows={len(df_fit)}/{len(df0)} | "
            f"rb_ou(bins={ou_bins} iqr_med={ou_iqr_med if np.isfinite(ou_iqr_med) else float('nan'):.4f} iqr_p90={ou_iqr_p90 if np.isfinite(ou_iqr_p90) else float('nan'):.4f}) "
            f"rb_bt(bins={bt_bins} iqr_med={bt_iqr_med if np.isfinite(bt_iqr_med) else float('nan'):.4f} iqr_p90={bt_iqr_p90 if np.isfinite(bt_iqr_p90) else float('nan'):.4f}) "
            f"conf_min={conf_min} exclude_fit_src={exclude_src}"
        )

    df = df0.copy()

    # Ensure synth meta columns exist (force method columns to string dtype to avoid dtype warnings)
    if "synth_u25_method" not in df.columns:
        df["synth_u25_method"] = pd.Series(pd.NA, index=df.index, dtype="string")
    else:
        df["synth_u25_method"] = df["synth_u25_method"].astype("string")

    if "synth_u25_conf" not in df.columns:
        df["synth_u25_conf"] = np.nan

    if "odds_source_under25" not in df.columns:
        df["odds_source_under25"] = pd.Series("", index=df.index, dtype="string")
    else:
        df["odds_source_under25"] = df["odds_source_under25"].astype("string")

    if "synth_bttsno_method" not in df.columns:
        df["synth_bttsno_method"] = pd.Series(pd.NA, index=df.index, dtype="string")
    else:
        df["synth_bttsno_method"] = df["synth_bttsno_method"].astype("string")

    if "synth_bttsno_conf" not in df.columns:
        df["synth_bttsno_conf"] = np.nan

    if "odds_source_btts_no" not in df.columns:
        df["odds_source_btts_no"] = pd.Series("", index=df.index, dtype="string")
    else:
        df["odds_source_btts_no"] = df["odds_source_btts_no"].astype("string")

    # --- Under25 synthesis ---
    u = _safe_odds(df.get("odds_ft_under25", pd.Series(np.nan, index=df.index)))
    o = _safe_odds(df.get("odds_ft_over25", pd.Series(np.nan, index=df.index)))

    need_u = u.isna() & o.notna()

    synth_count_u = 0
    for idx in df.index[need_u].tolist():
        od_known = float(pd.to_numeric(df.at[idx, "odds_ft_over25"], errors="coerce"))
        od_hat, method, conf = _synth_missing_from_known(od_known, rb_ou, conf_min=conf_min)
        if np.isfinite(od_hat) and od_hat > 1.0 and str(method).startswith("bin_overround"):
            df.at[idx, "odds_ft_under25"] = float(od_hat)
            df.at[idx, "synth_u25_method"] = str(method)
            df.at[idx, "synth_u25_conf"] = float(conf)
            df.at[idx, "odds_source_under25"] = "synth"
            synth_count_u += 1

    # Mark existing under25 sources
    try:
        u2 = _safe_odds(df.get("odds_ft_under25", pd.Series(np.nan, index=df.index)))
        src = df.get("odds_source_under25", pd.Series("", index=df.index)).astype("string").fillna("")
        src = src.where(src.ne(""), np.where(u2.notna(), "existing", ""))
        df["odds_source_under25"] = src
    except Exception:
        pass

    # --- BTTS NO synthesis (optional) ---
    synth_count_bt = 0
    if synth_btts_no and rb_bt:
        bn = _safe_odds(df.get("odds_btts_no", pd.Series(np.nan, index=df.index)))
        by = _safe_odds(df.get("odds_btts_yes", pd.Series(np.nan, index=df.index)))
        need_bn = bn.isna() & by.notna()

        for idx in df.index[need_bn].tolist():
            od_known = float(pd.to_numeric(df.at[idx, "odds_btts_yes"], errors="coerce"))
            od_hat, method, conf = _synth_missing_from_known(od_known, rb_bt, conf_min=conf_min)
            if np.isfinite(od_hat) and od_hat > 1.0 and str(method).startswith("bin_overround"):
                df.at[idx, "odds_btts_no"] = float(od_hat)
                df.at[idx, "synth_bttsno_method"] = str(method)
                df.at[idx, "synth_bttsno_conf"] = float(conf)
                df.at[idx, "odds_source_btts_no"] = "synth"
                synth_count_bt += 1

        # Mark existing btts_no sources
        try:
            bn2 = _safe_odds(df.get("odds_btts_no", pd.Series(np.nan, index=df.index)))
            src = df.get("odds_source_btts_no", pd.Series("", index=df.index)).astype("string").fillna("")
            src = src.where(src.ne(""), np.where(bn2.notna(), "existing", ""))
            df["odds_source_btts_no"] = src
        except Exception:
            pass

    # Recompute nn counts for reporting
    u_nn = int(_safe_odds(df.get("odds_ft_under25", pd.Series(np.nan, index=df.index))).notna().sum())
    bttsno_nn = int(_safe_odds(df.get("odds_btts_no", pd.Series(np.nan, index=df.index))).notna().sum())

    # Output path
    out_path = league_dir / "fd_odds_enriched_synth.csv"
    out_csv = str(out_path)

    # ------------------------------------------------------------------
    # SAFETY: Preserve real odds columns from base fd_odds_enriched.csv
    # ------------------------------------------------------------------
    try:
        # Resolve base path next to the synth output
        try:
            _base_path = Path(out_path).with_name("fd_odds_enriched.csv")
        except Exception:
            _base_path = Path(str(out_csv)).with_name("fd_odds_enriched.csv")

        if _base_path.exists():
            _base = pd.read_csv(_base_path, low_memory=False)

            ODDS_COLS = [
                "odds_ft_home_team_win","odds_ft_draw","odds_ft_away_team_win",
                "odds_ft_over15","odds_ft_over25","odds_ft_over35","odds_ft_over45",
                "odds_ft_under25",
                "odds_btts_yes","odds_btts_no",
            ]

            # Ensure fixture_key exists on both frames for stable alignment
            if "fixture_key" in df.columns and "fixture_key" in _base.columns:
                _b = _base.copy()
                _b["fixture_key"] = _b["fixture_key"].astype("string").fillna("").str.strip()
                df["fixture_key"] = df["fixture_key"].astype("string").fillna("").str.strip()

                _keep = ["fixture_key"] + [c for c in ODDS_COLS if c in _b.columns]
                _b = _b[_keep].drop_duplicates(subset=["fixture_key"], keep="first")

                df = df.merge(_b, on="fixture_key", how="left", suffixes=("", "__base"))

                for c in ODDS_COLS:
                    bc = f"{c}__base"
                    if bc not in df.columns:
                        continue

                    # If synth column missing, create it
                    if c not in df.columns:
                        df[c] = np.nan

                    s = pd.to_numeric(df[c], errors="coerce")
                    b = pd.to_numeric(df[bc], errors="coerce")

                    # Detect a "broken" synth column: all zeros or no usable odds (>1)
                    broken = False
                    try:
                        finite = s[np.isfinite(s)]
                        if len(finite) > 0 and float(finite.max()) <= 1.0:
                            broken = True
                        if len(finite) > 0 and float(finite.min()) == 0.0 and float(finite.max()) == 0.0:
                            broken = True
                    except Exception:
                        pass

                    # Restore: for broken columns, take base values; otherwise fill only missing/non-odds
                    if broken:
                        df[c] = b
                    else:
                        # Treat <=1 as missing and fill from base
                        s = s.where(s > 1.0, np.nan)
                        df[c] = s.where(s.notna(), b)

                    # Clean up placeholder zeros
                    df[c] = pd.to_numeric(df[c], errors="coerce").where(pd.to_numeric(df[c], errors="coerce") > 1.0, np.nan)

                # Drop merge suffix columns
                df = df.drop(columns=[c for c in df.columns if str(c).endswith("__base")], errors="ignore")

            else:
                # No fixture_key alignment available: best-effort restore by index
                for c in ODDS_COLS:
                    if c in _base.columns:
                        b = pd.to_numeric(_base[c], errors="coerce")
                        if c not in df.columns:
                            df[c] = b
                        else:
                            s = pd.to_numeric(df[c], errors="coerce")
                            finite = s[np.isfinite(s)]
                            broken = bool(len(finite) > 0 and float(finite.max()) <= 1.0)
                            if broken:
                                df[c] = b

    except Exception:
        # Never fail synth generation on restore logic
        pass

    # Write
    df.to_csv(out_path, index=False)

    # Optional: emit O/U2.5 no-vig probabilities for paired rows
    novig_diag = None
    if emit_ou25_novig:
        try:
            novig_diag = _emit_fd_ou25_novig(
                league_dir,
                df,
                conf_gate=float(ou25_conf_gate),
                future_only=bool(ou25_future_only),
                debug=bool(debug),
            )
        except Exception as e:
            if debug:
                print(f"⚠️  {league_dir.name}: failed to emit fd_ou25_novig.csv: {e}")

    return {
        "league": league_dir.name,
        "u25_nn": int(u_nn),
        "bttsno_nn": int(bttsno_nn),
        "synth_u25": int(synth_count_u),
        "synth_bttsno": int(synth_count_bt),
        "rb_ou_bins": int(ou_bins),
        "rb_bt_bins": int(bt_bins),
        "ou25_novig_rows": int(novig_diag["rows_written"]) if novig_diag else 0,
        "ou25_future_rows": int(novig_diag["future_rows"]) if novig_diag else 0,
        "ou25_future_paired": int(novig_diag["future_paired"]) if novig_diag else 0,
        "ou25_total_o25_present": int(novig_diag["total_o25_present"]) if novig_diag else 0,
        "ou25_total_u25_present": int(novig_diag["total_u25_present"]) if novig_diag else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Create fd_odds_enriched_synth.csv for each league under Matches/")
    ap.add_argument("--matches-root", default="Matches")
    ap.add_argument("--leagues", default=None, help="Comma-separated list. Default: all leagues under Matches/")
    ap.add_argument("--n-bins", type=int, default=12)
    ap.add_argument("--min-n", type=int, default=200)
    ap.add_argument("--conf-min", type=float, default=0.10)
    ap.add_argument("--no-bttsno", action="store_true", help="Disable BTTS NO synthesis")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument(
        "--exclude-fit-src",
        default="_all_seasons,__TRAIN_MULTISEASON",
        help="Comma-separated substrings. Rows whose __src_csv contains any substring are excluded from rulebook fitting only.",
    )
    ap.add_argument("--emit-ou25-novig", action="store_true", help="Also write fd_ou25_novig.csv with no-vig probs for paired O2.5/U2.5")
    ap.add_argument("--ou25-conf-gate", type=float, default=0.0, help="If >0, require synth_u25_conf >= gate for synthesized under25 rows when emitting novig")
    ap.add_argument("--ou25-future-only", action="store_true", help="When emitting novig, restrict to match_date >= today")
    args = ap.parse_args()

    exclude_src = [s.strip() for s in str(args.exclude_fit_src).split(",") if s.strip()]

    matches_root = Path(args.matches_root)
    if args.leagues:
        leagues = [s.strip() for s in str(args.leagues).split(",") if s.strip()]
        league_dirs = [matches_root / lg for lg in leagues]
    else:
        league_dirs = [p for p in sorted(matches_root.iterdir()) if p.is_dir()]

    for ld in league_dirs:
        res = _process_one_league(
            ld,
            n_bins=int(args.n_bins),
            min_n=int(args.min_n),
            conf_min=float(args.conf_min),
            synth_btts_no=(not bool(args.no_bttsno)),
            debug=bool(args.debug),
            exclude_src=exclude_src,
            emit_ou25_novig=bool(args.emit_ou25_novig),
            ou25_conf_gate=float(args.ou25_conf_gate),
            ou25_future_only=bool(args.ou25_future_only),
        )
        if not res:
            continue
        print(
            f"✅ {res['league']}: wrote fd_odds_enriched_synth.csv | "
            f"under25_nn={res['u25_nn']} btts_no_nn={res['bttsno_nn']} | "
            f"synth_u25={res['synth_u25']} synth_bttsno={res['synth_bttsno']}"
            f" | ou25_novig_rows={res.get('ou25_novig_rows', 0)}"
            f" future_rows={res.get('ou25_future_rows', 0)}"
            f" future_paired={res.get('ou25_future_paired', 0)}"
            f" total_o25_present={res.get('ou25_total_o25_present', 0)}"
            f" total_u25_present={res.get('ou25_total_u25_present', 0)}"
        )
        if int(res.get("ou25_future_rows", 0)) > 0 and int(res.get("ou25_future_paired", 0)) == 0:
            print(
                f"⚠️  {res['league']}: future rows exist but paired O/U2.5 odds are still zero. "
                "Check fd_odds_enriched.csv coverage or synth under generation before weekend picks."
            )


if __name__ == "__main__":
    main()
