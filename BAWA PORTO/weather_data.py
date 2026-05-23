"""Utilities for enriching football fixtures with historical weather data."""

from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from zoneinfo import ZoneInfo

DEFAULT_HOURLY_PARAMS: Tuple[str, ...] = (
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "snowfall",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "surface_pressure",
    "pressure_msl",
    "weather_code",
    "shortwave_radiation",
)

WEATHER_SOURCE_NAME = "open-meteo (archive)"


@dataclass
class OpenMeteoClient:
    """Small client for Open-Meteo archive and geocoding endpoints."""

    cache_dir: Optional[str] = None
    timeout: int = 20
    session: Optional[requests.Session] = None
    geocode_base: str = "https://geocoding-api.open-meteo.com/v1/search"
    archive_base: str = "https://archive-api.open-meteo.com/v1/archive"

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = requests.Session()
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        self._geo_cache = self._load_cache("geocode")
        self._weather_cache = self._load_cache("weather")

    def _cache_path(self, name: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, f"{name}_cache.json")

    def _load_cache(self, name: str) -> Dict[str, Dict[str, Any]]:
        path = self._cache_path(name)
        if not path or not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {}

    def _persist_cache(self, name: str) -> None:
        path = self._cache_path(name)
        if not path:
            return
        cache = self._geo_cache if name == "geocode" else self._weather_cache
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(cache, fh)
        except Exception:
            pass

    def geocode(self, query: str, *, country: Optional[str] = None) -> Dict[str, Any]:
        query = (query or "").strip()
        if not query:
            return {}
        key = f"{query.lower()}|{(country or '').strip().lower()}"
        if key in self._geo_cache:
            return self._geo_cache[key]

        params = {"name": query, "count": 1, "language": "en", "format": "json"}
        if country:
            params["country"] = country
        resp = self.session.get(self.geocode_base, params=params, timeout=self.timeout)
        resp.raise_for_status()
        payload = resp.json() or {}
        results = payload.get("results") or []
        out: Dict[str, Any] = {}
        if results:
            top = results[0]
            out = {
                "name": top.get("name"),
                "latitude": top.get("latitude"),
                "longitude": top.get("longitude"),
                "country": top.get("country_code") or top.get("country"),
                "timezone": top.get("timezone"),
            }
        self._geo_cache[key] = out
        self._persist_cache("geocode")
        return out

    def get_hourly_weather(
        self,
        latitude: float,
        longitude: float,
        *,
        start: dt.datetime,
        end: dt.datetime,
        hourly_params: Iterable[str] = DEFAULT_HOURLY_PARAMS,
    ) -> Dict[str, Any]:
        hourly_params = tuple(hourly_params)
        lat = float(latitude)
        lon = float(longitude)
        key = f"{lat:.4f}|{lon:.4f}|{start.date()}|{end.date()}|{','.join(hourly_params)}"
        if key in self._weather_cache:
            return self._weather_cache[key]

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
            "hourly": ",".join(hourly_params),
            "timezone": "GMT",
            "timeformat": "unixtime",
        }
        resp = self.session.get(self.archive_base, params=params, timeout=self.timeout)
        resp.raise_for_status()
        payload = resp.json() or {}
        out = {
            "latitude": payload.get("latitude"),
            "longitude": payload.get("longitude"),
            "timezone": payload.get("timezone"),
            "utc_offset_seconds": payload.get("utc_offset_seconds"),
            "data": payload.get("hourly") or {},
            "units": payload.get("hourly_units") or {},
        }
        self._weather_cache[key] = out
        self._persist_cache("weather")
        return out

    def get_weather_summary(
        self,
        latitude: float,
        longitude: float,
        kickoff: dt.datetime,
        *,
        window_hours: float = 2.0,
        hourly_params: Iterable[str] = DEFAULT_HOURLY_PARAMS,
    ) -> Dict[str, Any]:
        window = dt.timedelta(hours=float(window_hours))
        payload = self.get_hourly_weather(
            latitude=latitude,
            longitude=longitude,
            start=kickoff - window,
            end=kickoff + window,
            hourly_params=hourly_params,
        )
        summary = summarise_hourly_weather(payload, kickoff, window_hours=window_hours)
        summary["weather_source"] = WEATHER_SOURCE_NAME
        summary["weather_kickoff_utc"] = kickoff.astimezone(dt.timezone.utc).isoformat()
        summary["requested_latitude"] = float(latitude)
        summary["requested_longitude"] = float(longitude)
        summary["returned_grid_latitude"] = _coerce_float(payload.get("latitude"))
        summary["returned_grid_longitude"] = _coerce_float(payload.get("longitude"))
        summary["weather_timezone"] = payload.get("timezone")
        summary["utc_offset_seconds"] = payload.get("utc_offset_seconds")
        return summary


def summarise_hourly_weather(payload: Dict[str, Any], kickoff: dt.datetime, window_hours: float = 2.0) -> Dict[str, Any]:
    hourly = (payload or {}).get("data") or {}
    if not hourly:
        return {"weather_sample_size": 0, "weather_condition": "unknown"}

    frame = pd.DataFrame(hourly)
    if frame.empty:
        return {"weather_sample_size": 0, "weather_condition": "unknown"}

    if "time" in frame.columns:
        time_vals = pd.to_numeric(frame["time"], errors="coerce")
        frame["time"] = pd.to_datetime(time_vals, unit="s", utc=True, errors="coerce")
    else:
        return {"weather_sample_size": 0, "weather_condition": "unknown"}

    kickoff_utc = kickoff.astimezone(dt.timezone.utc)
    window = pd.Timedelta(hours=float(window_hours))
    subset = frame[(frame["time"] >= kickoff_utc - window) & (frame["time"] <= kickoff_utc + window)].copy()
    if subset.empty:
        subset = frame.copy()

    numeric_cols = [c for c in subset.columns if c != "time"]
    subset[numeric_cols] = subset[numeric_cols].apply(pd.to_numeric, errors="coerce")

    summary = {
        "weather_sample_size": int(len(subset)),
        "temperature_2m": _safe_mean(subset, "temperature_2m"),
        "relative_humidity_2m": _safe_mean(subset, "relative_humidity_2m"),
        "dew_point_2m": _safe_mean(subset, "dew_point_2m"),
        "apparent_temperature": _safe_mean(subset, "apparent_temperature"),
        "precipitation": _safe_sum(subset, "precipitation"),
        "rain": _safe_sum(subset, "rain"),
        "snowfall": _safe_sum(subset, "snowfall"),
        "wind_speed_10m": _safe_mean(subset, "wind_speed_10m"),
        "wind_direction_10m": _safe_mean(subset, "wind_direction_10m"),
        "wind_gusts_10m": _safe_max(subset, "wind_gusts_10m"),
        "cloud_cover": _safe_mean(subset, "cloud_cover"),
        "surface_pressure": _safe_mean(subset, "surface_pressure"),
        "pressure_msl": _safe_mean(subset, "pressure_msl"),
        "shortwave_radiation": _safe_mean(subset, "shortwave_radiation"),
        "weather_code_mode": _safe_mode(subset, "weather_code"),
    }

    renamed = {
        "weather_temp_c_mean": summary["temperature_2m"],
        "weather_relative_humidity_pct": summary["relative_humidity_2m"],
        "weather_dew_point_c": summary["dew_point_2m"],
        "weather_apparent_temp_c": summary["apparent_temperature"],
        "weather_precip_mm": summary["precipitation"],
        "weather_rain_mm": summary["rain"],
        "weather_snow_mm": summary["snowfall"],
        "weather_windspeed_kmh": summary["wind_speed_10m"],
        "weather_wind_direction_deg": summary["wind_direction_10m"],
        "weather_windgust_kmh": summary["wind_gusts_10m"],
        "weather_cloud_cover_pct": summary["cloud_cover"],
        "weather_surface_pressure_hpa": summary["surface_pressure"],
        "weather_pressure_hpa": summary["pressure_msl"],
        "weather_shortwave_radiation": summary["shortwave_radiation"],
        "weather_code": summary["weather_code_mode"],
    }
    renamed.update(classify_conditions(renamed))
    return renamed


def classify_conditions(summary: Dict[str, Any]) -> Dict[str, Any]:
    precip = _coerce_float(summary.get("weather_precip_mm"))
    wind = _coerce_float(summary.get("weather_windspeed_kmh"))
    gust = _coerce_float(summary.get("weather_windgust_kmh"))
    temp = _coerce_float(summary.get("weather_temp_c_mean"))

    conds = []
    if precip is not None and precip >= 0.2:
        conds.append("wet")
    if wind is not None and wind >= 20:
        conds.append("windy")
    if gust is not None and gust >= 35:
        conds.append("gusty")
    if temp is not None and temp <= 4:
        conds.append("cold")
    if temp is not None and temp >= 28:
        conds.append("hot")
    if not conds:
        conds.append("fair")

    severity = 0
    if precip is not None:
        severity += min(3, int(precip >= 0.2) + int(precip >= 2.0) + int(precip >= 5.0))
    if wind is not None:
        severity += min(3, int(wind >= 20) + int(wind >= 30) + int(wind >= 40))
    if temp is not None:
        severity += int(temp <= 2 or temp >= 30)

    return {
        "weather_condition": " & ".join(sorted(set(conds))),
        "weather_is_wet": bool("wet" in conds),
        "weather_is_windy": bool("windy" in conds or "gusty" in conds),
        "weather_is_cold": bool("cold" in conds),
        "weather_is_hot": bool("hot" in conds),
        "heavy_rain_flag": bool(precip is not None and precip >= 2.0),
        "high_wind_flag": bool(wind is not None and wind >= 30),
        "gusty_flag": bool(gust is not None and gust >= 40),
        "weather_severity_score": int(severity),
    }


def fetch_weather_history_series(
    *,
    client: OpenMeteoClient,
    latitude: float,
    longitude: float,
    start: dt.datetime | str,
    end: dt.datetime | str,
    hourly_params: Iterable[str] = DEFAULT_HOURLY_PARAMS,
    timezone: str = "UTC",
    aggregate: Optional[str] = None,
) -> pd.DataFrame:
    start_dt = pd.to_datetime(start, utc=True)
    end_dt = pd.to_datetime(end, utc=True)
    if pd.isna(start_dt) or pd.isna(end_dt):
        raise ValueError("start and end must be parseable datetimes")
    payload = client.get_hourly_weather(
        latitude=latitude,
        longitude=longitude,
        start=start_dt.to_pydatetime(),
        end=end_dt.to_pydatetime(),
        hourly_params=hourly_params,
    )
    frame = pd.DataFrame((payload or {}).get("data") or {})
    if frame.empty:
        return frame
    frame["time"] = pd.to_datetime(pd.to_numeric(frame["time"], errors="coerce"), unit="s", utc=True, errors="coerce")
    frame = frame.dropna(subset=["time"]).sort_values("time")
    if timezone and timezone.upper() != "UTC":
        try:
            frame["time"] = frame["time"].dt.tz_convert(timezone)
        except Exception:
            pass
    num_cols = [c for c in frame.columns if c != "time"]
    frame[num_cols] = frame[num_cols].apply(pd.to_numeric, errors="coerce")
    if aggregate:
        resampled = frame.set_index("time").resample(aggregate).mean(numeric_only=True).reset_index()
        frame = resampled
    frame["weather_source"] = WEATHER_SOURCE_NAME
    frame["requested_latitude"] = float(latitude)
    frame["requested_longitude"] = float(longitude)
    return frame


def attach_weather_features(
    df: pd.DataFrame,
    *,
    client: OpenMeteoClient,
    location_col: Optional[str] = "stadium_name",
    lat_col: Optional[str] = None,
    lon_col: Optional[str] = None,
    country_col: Optional[str] = None,
    timestamp_col: Optional[str] = "timestamp",
    date_col: Optional[str] = "date_GMT",
    timezone_col: Optional[str] = None,
    fallback_location_cols: Optional[Iterable[str]] = None,
    country_hint: Optional[str] = None,
    window_hours: float = 2.0,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    fallback_location_cols = tuple(fallback_location_cols or ())
    rows = {}
    failures = 0
    for idx, row in df.iterrows():
        kickoff = resolve_kickoff(row, timestamp_col=timestamp_col, date_col=date_col, timezone=row.get(timezone_col) if timezone_col else "UTC")
        if kickoff is None:
            failures += 1
            continue
        lat = _coerce_float(row.get(lat_col)) if lat_col else None
        lon = _coerce_float(row.get(lon_col)) if lon_col else None
        if lat is None or lon is None:
            query = pick_location_query(row, location_col, fallback_location_cols)
            country = row.get(country_col) if country_col else None
            geo = client.geocode(query, country=country or country_hint) if query else {}
            lat = _coerce_float(geo.get("latitude"))
            lon = _coerce_float(geo.get("longitude"))
        if lat is None or lon is None:
            failures += 1
            continue
        try:
            rows[idx] = client.get_weather_summary(lat, lon, kickoff, window_hours=window_hours)
        except Exception:
            failures += 1
    if not rows:
        return df.copy(), {"attached_rows": 0, "failures": failures, "columns": []}
    features = pd.DataFrame.from_dict(rows, orient="index")
    out = df.copy()
    for col in features.columns:
        out[col] = features[col]
    return out, {"attached_rows": len(features), "failures": failures, "columns": list(features.columns)}


def resolve_kickoff(
    row: pd.Series,
    *,
    timestamp_col: Optional[str] = "timestamp",
    date_col: Optional[str] = "date_GMT",
    timezone: Any = "UTC",
) -> Optional[dt.datetime]:
    if timestamp_col and timestamp_col in row and pd.notna(row[timestamp_col]):
        ts = pd.to_numeric(row[timestamp_col], errors="coerce")
        if pd.notna(ts):
            return dt.datetime.fromtimestamp(float(ts), tz=dt.timezone.utc)
    if date_col and date_col in row and pd.notna(row[date_col]):
        parsed = pd.to_datetime(row[date_col], errors="coerce", utc=False)
        if pd.isna(parsed):
            return None
        if isinstance(parsed, pd.Timestamp):
            if parsed.tzinfo is None:
                tz_name = timezone if isinstance(timezone, str) and timezone else "UTC"
                try:
                    parsed = parsed.tz_localize(ZoneInfo(tz_name))
                except Exception:
                    parsed = parsed.tz_localize("UTC")
            return parsed.tz_convert("UTC").to_pydatetime()
    return None


def pick_location_query(row: pd.Series, location_col: Optional[str], fallback_cols: Iterable[str]) -> str:
    if location_col and location_col in row:
        value = str(row.get(location_col) or "").strip()
        if value:
            return value
    for col in fallback_cols:
        value = str(row.get(col) or "").strip()
        if value:
            return value
    return ""


def _safe_mean(frame: pd.DataFrame, col: str) -> float:
    if col not in frame:
        return np.nan
    return float(pd.to_numeric(frame[col], errors="coerce").mean())


def _safe_sum(frame: pd.DataFrame, col: str) -> float:
    if col not in frame:
        return np.nan
    return float(pd.to_numeric(frame[col], errors="coerce").sum())


def _safe_max(frame: pd.DataFrame, col: str) -> float:
    if col not in frame:
        return np.nan
    return float(pd.to_numeric(frame[col], errors="coerce").max())


def _safe_mode(frame: pd.DataFrame, col: str) -> float:
    if col not in frame:
        return np.nan
    values = pd.to_numeric(frame[col], errors="coerce").dropna()
    if values.empty:
        return np.nan
    mode = values.mode()
    return float(mode.iloc[0]) if not mode.empty else np.nan


def _coerce_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except Exception:
        return None
    if np.isnan(result):
        return None
    return result


__all__ = [
    "DEFAULT_HOURLY_PARAMS",
    "WEATHER_SOURCE_NAME",
    "OpenMeteoClient",
    "attach_weather_features",
    "classify_conditions",
    "fetch_weather_history_series",
    "resolve_kickoff",
    "summarise_hourly_weather",
]
