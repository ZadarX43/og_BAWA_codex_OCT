#!/usr/bin/env python3
"""Build a World Cup 2026 venue/weather/travel research scaffold.

This is a pre-tournament sidecar. It uses known venue geography and coarse
climate/travel priors, then leaves explicit placeholders for prematch weather
snapshots once match week data is available.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd


DEFAULT_LAUNCH = Path("data_sources/footystats_world_cup/launch_2026/world_cup_2026_launch_scaffold.csv")
DEFAULT_MATRIX = Path("data_sources/footystats_world_cup/research_feature_matrix_2026/world_cup_2026_research_feature_matrix.csv")
DEFAULT_OUTDIR = Path("data_sources/footystats_world_cup/venue_weather_travel_2026")

VENUE_REGISTRY = {
    "arrowhead_stadium": {
        "venue_name": "Arrowhead Stadium",
        "city": "Kansas City",
        "country": "USA",
        "latitude": 39.0489,
        "longitude": -94.4839,
        "altitude_m": 274,
        "roof_flag": 0,
        "heat_humidity_prior": "MODERATE_SUMMER_HEAT",
    },
    "bc_place": {
        "venue_name": "BC Place",
        "city": "Vancouver",
        "country": "Canada",
        "latitude": 49.2768,
        "longitude": -123.1119,
        "altitude_m": 2,
        "roof_flag": 1,
        "heat_humidity_prior": "LOW_WEATHER_BUFFERED",
    },
    "bmo_field": {
        "venue_name": "BMO Field",
        "city": "Toronto",
        "country": "Canada",
        "latitude": 43.6332,
        "longitude": -79.4186,
        "altitude_m": 76,
        "roof_flag": 0,
        "heat_humidity_prior": "MODERATE_SUMMER_HEAT",
    },
    "estadio_akron": {
        "venue_name": "Estadio Akron",
        "city": "Zapopan",
        "country": "Mexico",
        "latitude": 20.6819,
        "longitude": -103.4628,
        "altitude_m": 1566,
        "roof_flag": 0,
        "heat_humidity_prior": "MODERATE_ALTITUDE_HEAT",
    },
    "estadio_azteca": {
        "venue_name": "Estadio Azteca",
        "city": "Mexico City",
        "country": "Mexico",
        "latitude": 19.3029,
        "longitude": -99.1505,
        "altitude_m": 2240,
        "roof_flag": 0,
        "heat_humidity_prior": "HIGH_ALTITUDE",
    },
    "estadio_bbva": {
        "venue_name": "Estadio BBVA",
        "city": "Monterrey",
        "country": "Mexico",
        "latitude": 25.6682,
        "longitude": -100.2440,
        "altitude_m": 540,
        "roof_flag": 0,
        "heat_humidity_prior": "HIGH_SUMMER_HEAT",
    },
    "gillette_stadium": {
        "venue_name": "Gillette Stadium",
        "city": "Foxborough",
        "country": "USA",
        "latitude": 42.0909,
        "longitude": -71.2643,
        "altitude_m": 88,
        "roof_flag": 0,
        "heat_humidity_prior": "MODERATE_SUMMER_HEAT",
    },
    "hard_rock_stadium": {
        "venue_name": "Hard Rock Stadium",
        "city": "Miami Gardens",
        "country": "USA",
        "latitude": 25.9580,
        "longitude": -80.2389,
        "altitude_m": 2,
        "roof_flag": 0,
        "heat_humidity_prior": "HIGH_HEAT_HUMIDITY",
    },
    "lincoln_financial_field": {
        "venue_name": "Lincoln Financial Field",
        "city": "Philadelphia",
        "country": "USA",
        "latitude": 39.9008,
        "longitude": -75.1675,
        "altitude_m": 12,
        "roof_flag": 0,
        "heat_humidity_prior": "MODERATE_SUMMER_HEAT",
    },
    "lumen_field": {
        "venue_name": "Lumen Field",
        "city": "Seattle",
        "country": "USA",
        "latitude": 47.5952,
        "longitude": -122.3316,
        "altitude_m": 52,
        "roof_flag": 0,
        "heat_humidity_prior": "LOW_COASTAL",
    },
    "mercedes_benz_stadium": {
        "venue_name": "Mercedes-Benz Stadium",
        "city": "Atlanta",
        "country": "USA",
        "latitude": 33.7554,
        "longitude": -84.4008,
        "altitude_m": 320,
        "roof_flag": 1,
        "heat_humidity_prior": "HIGH_HEAT_BUFFERED",
    },
    "metlife_stadium": {
        "venue_name": "MetLife Stadium",
        "city": "East Rutherford",
        "country": "USA",
        "latitude": 40.8135,
        "longitude": -74.0745,
        "altitude_m": 3,
        "roof_flag": 0,
        "heat_humidity_prior": "MODERATE_SUMMER_HEAT",
    },
    "nrg_stadium": {
        "venue_name": "NRG Stadium",
        "city": "Houston",
        "country": "USA",
        "latitude": 29.6847,
        "longitude": -95.4107,
        "altitude_m": 13,
        "roof_flag": 1,
        "heat_humidity_prior": "HIGH_HEAT_HUMIDITY_BUFFERED",
    },
    "sofi_stadium": {
        "venue_name": "SoFi Stadium",
        "city": "Inglewood",
        "country": "USA",
        "latitude": 33.9535,
        "longitude": -118.3392,
        "altitude_m": 38,
        "roof_flag": 1,
        "heat_humidity_prior": "LOW_WEATHER_BUFFERED",
    },
}

HOST_COUNTRY_BY_TEAM = {"Canada": "Canada", "Mexico": "Mexico", "USA": "USA"}
LOCAL_CONFED_BY_COUNTRY = {"Canada": "CONCACAF", "Mexico": "CONCACAF", "USA": "CONCACAF"}


def slugify(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def altitude_bucket(altitude_m: float | int | None) -> str:
    if altitude_m is None or pd.isna(altitude_m):
        return "UNKNOWN"
    if altitude_m >= 1800:
        return "HIGH_ALTITUDE"
    if altitude_m >= 1000:
        return "MODERATE_ALTITUDE"
    return "LOW_ALTITUDE"


def confed_travel_load(confed: object, venue_country: object) -> int:
    confed_text = "" if pd.isna(confed) else str(confed).upper()
    country = "" if pd.isna(venue_country) else str(venue_country)
    if not country:
        return 2
    if confed_text == LOCAL_CONFED_BY_COUNTRY.get(country, ""):
        return 0
    if confed_text == "CONMEBOL":
        return 1
    if confed_text in {"UEFA", "CAF"}:
        return 2
    if confed_text in {"AFC", "OFC"}:
        return 3
    return 2


def climate_load(heat_humidity_prior: object, altitude: object, roof_flag: object) -> int:
    if pd.isna(heat_humidity_prior):
        return 1
    text = str(heat_humidity_prior).upper()
    load = 0
    if "HIGH_HEAT" in text or "HIGH_SUMMER" in text:
        load += 2
    elif "MODERATE" in text:
        load += 1
    if not pd.isna(altitude) and float(altitude) >= 1500:
        load += 2
    if pd.to_numeric(pd.Series([roof_flag]), errors="coerce").fillna(0).iloc[0] == 1:
        load = max(0, load - 1)
    return load


def risk_bucket(score: int | float) -> str:
    if score >= 5:
        return "HIGH"
    if score >= 3:
        return "MEDIUM"
    if score >= 1:
        return "LOW"
    return "LOCAL_OR_BUFFERED"


def load_confeds(matrix_path: Path) -> pd.DataFrame:
    if not matrix_path.exists():
        return pd.DataFrame(columns=["team_name", "confederation"])
    matrix = pd.read_csv(matrix_path, low_memory=False)
    rows = []
    for side in ["home", "away"]:
        name = f"api_{side}_team_name"
        confed = f"{side}_confederation"
        if name in matrix.columns and confed in matrix.columns:
            rows.append(matrix[[name, confed]].rename(columns={name: "team_name", confed: "confederation"}))
    if not rows:
        return pd.DataFrame(columns=["team_name", "confederation"])
    out = pd.concat(rows, ignore_index=True).dropna(subset=["team_name"]).drop_duplicates("team_name")
    return out


def build_venue_registry() -> pd.DataFrame:
    rows = []
    for slug, meta in VENUE_REGISTRY.items():
        row = {"venue_slug": slug, **meta}
        row["altitude_bucket"] = altitude_bucket(row["altitude_m"])
        row["indoor_weather_buffer_flag"] = int(row["roof_flag"] == 1)
        row["venue_registry_ready_flag"] = 1
        rows.append(row)
    return pd.DataFrame(rows)


def build_fixture_scaffold(launch: pd.DataFrame, registry: pd.DataFrame, confeds: pd.DataFrame) -> pd.DataFrame:
    base = launch[
        [
            "api_fixture_id",
            "api_date",
            "api_round",
            "api_home_team_name",
            "api_away_team_name",
            "api_venue_name",
            "api_venue_city",
        ]
    ].copy()
    base["venue_slug"] = base["api_venue_name"].map(slugify)
    out = base.merge(registry, on="venue_slug", how="left")
    out["venue_name_resolved"] = out["venue_name"].fillna(out["api_venue_name"])
    out["venue_city_resolved"] = out["city"].fillna(out["api_venue_city"])
    out["venue_known_flag"] = out["venue_registry_ready_flag"].fillna(0).astype(int)
    out["venue_tbd_flag"] = out["api_venue_name"].isna().astype(int)
    out["venue_host_country"] = out["country"]
    out["venue_local_confederation"] = out["venue_host_country"].map(LOCAL_CONFED_BY_COUNTRY)
    out["home_is_host_country_side_flag"] = (
        out["api_home_team_name"].map(HOST_COUNTRY_BY_TEAM).fillna("").eq(out["venue_host_country"].fillna(""))
    ).astype(int)
    out["away_is_host_country_side_flag"] = (
        out["api_away_team_name"].map(HOST_COUNTRY_BY_TEAM).fillna("").eq(out["venue_host_country"].fillna(""))
    ).astype(int)
    out["neutral_venue_flag"] = (
        out["home_is_host_country_side_flag"].eq(0) & out["away_is_host_country_side_flag"].eq(0)
    ).astype(int)

    confed_map = confeds.set_index("team_name")["confederation"].to_dict()
    out["home_confederation"] = out["api_home_team_name"].map(confed_map)
    out["away_confederation"] = out["api_away_team_name"].map(confed_map)
    out["home_travel_load_proxy"] = [
        confed_travel_load(confed, country) for confed, country in zip(out["home_confederation"], out["venue_host_country"])
    ]
    out["away_travel_load_proxy"] = [
        confed_travel_load(confed, country) for confed, country in zip(out["away_confederation"], out["venue_host_country"])
    ]
    out["travel_load_delta_home_minus_away"] = out["home_travel_load_proxy"] - out["away_travel_load_proxy"]
    out["fixture_climate_load_proxy"] = [
        climate_load(heat, alt, roof)
        for heat, alt, roof in zip(out["heat_humidity_prior"], out["altitude_m"], out["roof_flag"])
    ]
    out["home_travel_climate_load_proxy"] = out["home_travel_load_proxy"] + out["fixture_climate_load_proxy"]
    out["away_travel_climate_load_proxy"] = out["away_travel_load_proxy"] + out["fixture_climate_load_proxy"]
    out["travel_climate_load_delta_home_minus_away"] = (
        out["home_travel_climate_load_proxy"] - out["away_travel_climate_load_proxy"]
    )
    out["home_travel_climate_risk_bucket"] = out["home_travel_climate_load_proxy"].map(risk_bucket)
    out["away_travel_climate_risk_bucket"] = out["away_travel_climate_load_proxy"].map(risk_bucket)
    out["venue_weather_travel_scaffold_ready_flag"] = out["venue_known_flag"]
    out["weather_snapshot_status"] = "PENDING_PREMATCH_WEATHER_API"
    out["weather_model_ready_flag"] = 0
    out["travel_model_status"] = "CONFED_PROXY_NEEDS_TEAM_BASE_DISTANCE"
    unknown = out["venue_known_flag"].eq(0)
    for col in [
        "home_travel_load_proxy",
        "away_travel_load_proxy",
        "travel_load_delta_home_minus_away",
        "fixture_climate_load_proxy",
        "home_travel_climate_load_proxy",
        "away_travel_climate_load_proxy",
        "travel_climate_load_delta_home_minus_away",
    ]:
        out.loc[unknown, col] = pd.NA
    out.loc[unknown, "home_travel_climate_risk_bucket"] = "UNKNOWN_VENUE"
    out.loc[unknown, "away_travel_climate_risk_bucket"] = "UNKNOWN_VENUE"
    out.loc[unknown, "weather_snapshot_status"] = "PENDING_VENUE_ASSIGNMENT"
    out.loc[unknown, "travel_model_status"] = "PENDING_VENUE_ASSIGNMENT"
    return out


def write_summary(outdir: Path, registry: pd.DataFrame, fixture: pd.DataFrame) -> None:
    lines = [
        "# World Cup Venue, Weather, and Travel Scaffold",
        "",
        "Research-only venue geography, climate, and travel proxy sidecar for 2026 group-stage fixtures.",
        "",
        "## Outputs",
        "",
        f"- `{outdir / 'world_cup_venue_registry_2026.csv'}`",
        f"- `{outdir / 'world_cup_2026_fixture_venue_travel_weather_scaffold.csv'}`",
        "",
        "## Coverage",
        "",
        f"- Registry venues: {len(registry)}",
        f"- Fixtures: {len(fixture)}",
        f"- Known venue fixtures: {int(fixture['venue_known_flag'].sum())} / {len(fixture)}",
        f"- TBD venue fixtures: {int(fixture['venue_tbd_flag'].sum())} / {len(fixture)}",
        f"- High-altitude fixtures: {int(fixture['altitude_bucket'].eq('HIGH_ALTITUDE').sum())}",
        f"- Roof/weather-buffered fixtures: {int(fixture['indoor_weather_buffer_flag'].fillna(0).sum())}",
        "",
        "## Notes",
        "",
        "- Weather is intentionally marked pending until prematch weather snapshots are pulled.",
        "- Travel load is currently a confederation proxy; team-base distance can replace it later.",
        "- Venue names missing from the API schedule remain explicit TBD rows rather than guessed.",
    ]
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, default=DEFAULT_LAUNCH)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    launch = pd.read_csv(args.launch, low_memory=False)
    confeds = load_confeds(args.matrix)
    registry = build_venue_registry()
    fixture = build_fixture_scaffold(launch, registry, confeds)
    registry.to_csv(args.outdir / "world_cup_venue_registry_2026.csv", index=False)
    fixture.to_csv(args.outdir / "world_cup_2026_fixture_venue_travel_weather_scaffold.csv", index=False)
    write_summary(args.outdir, registry, fixture)
    print(f"[ok] venues={len(registry)} fixtures={len(fixture)} known={int(fixture['venue_known_flag'].sum())}")
    print(f"[ok] wrote {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
