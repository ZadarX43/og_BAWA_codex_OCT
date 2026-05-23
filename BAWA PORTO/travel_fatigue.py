# travel_fatigue.py
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None


# ---------------------------
# Core math helpers
# ---------------------------

def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized haversine distance in km."""
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    return 6371.0 * c


def daylight_hours(lat_deg: float, dt_utc: pd.Timestamp) -> float:
    """
    Approx daylight duration (hours) using latitude + day-of-year.
    Works well enough for 'Arctic winter shock' flags.
    """
    if pd.isna(lat_deg) or dt_utc is pd.NaT:
        return np.nan

    # day-of-year
    n = int(dt_utc.dayofyear)
    # solar declination (radians)
    dec = math.radians(23.44) * math.sin(2.0 * math.pi * (284 + n) / 365.0)
    lat = math.radians(float(lat_deg))

    # hour angle
    cos_h = -math.tan(lat) * math.tan(dec)
    cos_h = max(-1.0, min(1.0, cos_h))  # polar day/night clamp
    h = math.acos(cos_h)
    return (24.0 / math.pi) * h


@lru_cache(maxsize=4096)
def tz_offset_hours(tz_name: str, dt_utc_iso: str) -> float:
    """Return UTC offset hours for tz_name at dt_utc. Cached."""
    if ZoneInfo is None:
        return np.nan
    try:
        tz = ZoneInfo(str(tz_name))
        dt_utc = pd.Timestamp(dt_utc_iso).to_pydatetime()
        # dt_utc is timezone-aware? ensure UTC:
        if dt_utc.tzinfo is None:
            dt_utc = pd.Timestamp(dt_utc_iso, tz="UTC").to_pydatetime()
        local = dt_utc.astimezone(tz)
        off = local.utcoffset()
        return float(off.total_seconds() / 3600.0) if off is not None else np.nan
    except Exception:
        return np.nan


# ---------------------------
# Scoring config
# ---------------------------

@dataclass
class TravelStressWeights:
    w_dist: float = 0.35
    w_tz: float = 0.20
    w_east: float = 0.10
    w_daylight: float = 0.15
    w_lat: float = 0.10
    w_alt: float = 0.05
    w_rest: float = 0.05


def _clip01(x: pd.Series) -> pd.Series:
    return x.clip(0.0, 1.0)


def compute_travel_overlay(
    df: pd.DataFrame,
    team_geo_csv: str,
    match_date_col: str = "match_date",
    home_col: str = "home_team_name",
    away_col: str = "away_team_name",
    weights: TravelStressWeights = TravelStressWeights(),
) -> pd.DataFrame:
    """
    Attaches:
      travel_km, tz_diff_hours, eastward_hours, lat_diff, daylight_diff_hours, alt_diff_m,
      rest_adv_home (if rest cols exist),
      travel_stress_away (0-100),
      env_shock_flag (0/1)
    """
    out = df.copy()

    geo = pd.read_csv(team_geo_csv)
    geo["team_name"] = geo["team_name"].astype("string").str.strip()

    for c in ["lat", "lon", "alt_m"]:
        if c in geo.columns:
            geo[c] = pd.to_numeric(geo[c], errors="coerce")

    # join home/away geo
    out["_home"] = out[home_col].astype("string").str.strip()
    out["_away"] = out[away_col].astype("string").str.strip()

    home_geo = geo.rename(columns={
        "team_name": "_home",
        "lat": "lat_home",
        "lon": "lon_home",
        "alt_m": "alt_home",
        "timezone": "tz_home",
    })
    away_geo = geo.rename(columns={
        "team_name": "_away",
        "lat": "lat_away",
        "lon": "lon_away",
        "alt_m": "alt_away",
        "timezone": "tz_away",
    })

    out = out.merge(home_geo[["_home", "lat_home", "lon_home", "alt_home", "tz_home"]], on="_home", how="left")
    out = out.merge(away_geo[["_away", "lat_away", "lon_away", "alt_away", "tz_away"]], on="_away", how="left")

    # match date (UTC)
    md = pd.to_datetime(out.get(match_date_col, pd.NaT), errors="coerce", utc=True)
    out["_match_date_utc"] = md

    # travel distance
    out["travel_km"] = np.nan
    ok = out["lat_home"].notna() & out["lon_home"].notna() & out["lat_away"].notna() & out["lon_away"].notna()
    if ok.any():
        out.loc[ok, "travel_km"] = haversine_km(
            out.loc[ok, "lat_away"].to_numpy(),
            out.loc[ok, "lon_away"].to_numpy(),
            out.loc[ok, "lat_home"].to_numpy(),
            out.loc[ok, "lon_home"].to_numpy(),
        )

    # tz offsets at match time (DST-aware)
    out["tz_diff_hours"] = np.nan
    out["eastward_hours"] = np.nan
    if ZoneInfo is not None:
        def _off(row, which: str) -> float:
            tz = row.get(which, None)
            dt = row.get("_match_date_utc", pd.NaT)
            if pd.isna(dt) or tz is None or str(tz).strip() == "":
                return np.nan
            return tz_offset_hours(str(tz), str(pd.Timestamp(dt).isoformat()))

        home_off = out.apply(lambda r: _off(r, "tz_home"), axis=1)
        away_off = out.apply(lambda r: _off(r, "tz_away"), axis=1)
        out["tz_diff_hours"] = (home_off - away_off).abs()

        # east/west direction using longitude delta as proxy for direction
        # (positive if away is west of home => traveling east)
        lon_delta = pd.to_numeric(out["lon_home"], errors="coerce") - pd.to_numeric(out["lon_away"], errors="coerce")
        out["eastward_hours"] = np.where(lon_delta > 0, (home_off - away_off).clip(lower=0), 0.0)

    # latitude/daylight/altitude mismatches
    out["lat_diff"] = (pd.to_numeric(out["lat_home"], errors="coerce") - pd.to_numeric(out["lat_away"], errors="coerce")).abs()

    out["daylight_home"] = out.apply(
        lambda r: daylight_hours(r.get("lat_home", np.nan), r.get("_match_date_utc", pd.NaT)),
        axis=1,
    )
    out["daylight_away"] = out.apply(
        lambda r: daylight_hours(r.get("lat_away", np.nan), r.get("_match_date_utc", pd.NaT)),
        axis=1,
    )
    out["daylight_diff_hours"] = (pd.to_numeric(out["daylight_home"], errors="coerce") - pd.to_numeric(out["daylight_away"], errors="coerce")).abs()

    out["alt_diff_m"] = (pd.to_numeric(out["alt_home"], errors="coerce") - pd.to_numeric(out["alt_away"], errors="coerce")).clip(lower=0)

    # rest advantage if present (optional)
    # expected names if you already compute them somewhere: rest_days_home/rest_days_away
    rest_home = pd.to_numeric(out.get("rest_days_home", np.nan), errors="coerce")
    rest_away = pd.to_numeric(out.get("rest_days_away", np.nan), errors="coerce")
    out["rest_adv_home"] = (rest_home - rest_away)

    # --- normalize components to 0..1 ---
    # distance: log scale around 300km steps
    dist01 = _clip01(np.log1p(pd.to_numeric(out["travel_km"], errors="coerce") / 300.0) / np.log(1 + 3000/300))
    tz01 = _clip01(pd.to_numeric(out["tz_diff_hours"], errors="coerce") / 3.0)
    east01 = _clip01(pd.to_numeric(out["eastward_hours"], errors="coerce") / 3.0)
    day01 = _clip01(pd.to_numeric(out["daylight_diff_hours"], errors="coerce") / 6.0)
    lat01 = _clip01(pd.to_numeric(out["lat_diff"], errors="coerce") / 25.0)
    alt01 = _clip01(pd.to_numeric(out["alt_diff_m"], errors="coerce") / 1200.0)

    # rest penalty: only if away has <= 4 days AND home has more rest
    rest_pen = pd.Series(0.0, index=out.index)
    if np.isfinite(rest_away).any():
        rest_pen = _clip01((4.0 - rest_away) / 4.0) * _clip01((out["rest_adv_home"]) / 4.0)

    score01 = (
        weights.w_dist * dist01.fillna(0.0)
        + weights.w_tz * tz01.fillna(0.0)
        + weights.w_east * east01.fillna(0.0)
        + weights.w_daylight * day01.fillna(0.0)
        + weights.w_lat * lat01.fillna(0.0)
        + weights.w_alt * alt01.fillna(0.0)
        + weights.w_rest * rest_pen.fillna(0.0)
    )

    out["travel_stress_away"] = (100.0 * score01).clip(0.0, 100.0)

    # env shock flag: high latitude winter OR huge daylight mismatch OR high altitude jump
    month = out["_match_date_utc"].dt.month
    lat_home = pd.to_numeric(out["lat_home"], errors="coerce")
    out["env_shock_flag"] = (
        ((lat_home >= 66.5) & month.isin([11, 12, 1, 2, 3])) |
        (pd.to_numeric(out["daylight_diff_hours"], errors="coerce") >= 4.5) |
        (pd.to_numeric(out["alt_diff_m"], errors="coerce") >= 900)
    ).fillna(False).astype(int)

    return out.drop(columns=["_home", "_away"], errors="ignore")


def compute_upset_flags(
    df: pd.DataFrame,
    fav_implied_colset: Tuple[str, str, str] = ("imp_home", "imp_draw", "imp_away"),
    stress_col: str = "travel_stress_away",
    stress_thr: float = 65.0,
) -> pd.DataFrame:
    """
    Adds:
      fav_side (HOME/DRAW/AWAY),
      home_upset_risk_flag (0/1),
      home_upset_risk_score (0..1)
    """
    out = df.copy()

    # Determine favourite by *no-vig implied* if available; else use imp_*.
    ih = pd.to_numeric(out.get(fav_implied_colset[0], np.nan), errors="coerce")
    idr = pd.to_numeric(out.get(fav_implied_colset[1], np.nan), errors="coerce")
    ia = pd.to_numeric(out.get(fav_implied_colset[2], np.nan), errors="coerce")

    vv = pd.concat([ih, idr, ia], axis=1).to_numpy(dtype=float)
    fav_idx = np.nanargmax(np.where(np.isfinite(vv), vv, -1e9), axis=1)
    fav = np.array(["HOME", "DRAW", "AWAY"])[fav_idx]
    out["fav_side"] = fav

    stress = pd.to_numeric(out.get(stress_col, np.nan), errors="coerce").fillna(0.0)

    # basic "home not a minnow" floor using whichever exists
    ppg_h = pd.to_numeric(out.get("ppg_home_pre", np.nan), errors="coerce")
    xg_h = pd.to_numeric(out.get("pre_match_xg_home", np.nan), errors="coerce")
    pdiff = pd.to_numeric(out.get("power_diff", np.nan), errors="coerce")  # home - away

    home_ok = (
        (ppg_h.fillna(0) >= 0.90) |
        (xg_h.fillna(0) >= 1.00) |
        (pdiff.fillna(0) >= -15)
    )

    # risk score (0..1): scaled around threshold
    out["home_upset_risk_score"] = 1.0 / (1.0 + np.exp(-(stress - stress_thr) / 7.5))

    out["home_upset_risk_flag"] = (
        (out["fav_side"].astype(str) == "AWAY") &
        (stress >= stress_thr) &
        home_ok
    ).fillna(False).astype(int)

    return out
