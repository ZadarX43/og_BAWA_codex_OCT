#!/usr/bin/env python3
"""Build calendar-year API-Football sidecars for walk-forward benchmarks.

Research-only. API-Football feature files are normally keyed by provider season
start year. The benchmark walk-forward estate is keyed by fixture calendar year.
This bridge copies/splits already-local API-Football normalized and feature
files into calendar-year files without fetching data or changing production
routing.

The output is intentionally written to separate directories by default so the
provider-season estate remains intact.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_SCORED_ROOT = Path("predictions_output/hybrid_shadow_walkforward_2026_05_01_parity_rebuild")
DEFAULT_FEATURES_DIR = Path("data_sources/api_football/features")
DEFAULT_NORMALIZED_DIR = Path("data_sources/api_football/normalized")
DEFAULT_FEATURES_OUT_DIR = Path("data_sources/api_football/features_calendar_year")
DEFAULT_NORMALIZED_OUT_DIR = Path("data_sources/api_football/normalized_calendar_year")
DEFAULT_OUTDIR = Path("reports/latest/api_football_calendar_year_bridge")

KEYS = [
    "fixture_id",
    "fixture_key",
    "league",
    "league_id",
    "season",
    "match_date",
    "home_team_id",
    "away_team_id",
    "home_team_name",
    "away_team_name",
]

FEATURE_FAMILIES = [
    "team_rolling_features",
    "player_rolling_features",
    "lineup_features",
    "injury_features",
    "event_features",
    "odds_features",
    "h2h_regime_features",
    "referee_profile_features",
    "team_identity_features",
    "enriched_fixture_features",
    "matchup_interaction_features",
]

NORMALIZED_FAMILIES = [
    "fixtures_master",
    "match_team_stats",
    "match_events",
    "match_player_stats",
    "lineups",
    "injuries",
    "sidelined",
    "odds_prematch_long",
]

WINTER_LEAGUE_HINTS = {
    "Belgium_Pro",
    "Champions_League",
    "England_Championship",
    "England_EFL_League_1",
    "England_Premier_League",
    "Europa_Conference",
    "Europa_League",
    "France_Ligue_1",
    "Germany_Bundesliga",
    "Germany_Bundesliga_2",
    "Italy_Serie_A",
    "Netherlands_Eredivisie",
    "Portugal_Liga",
    "Scotland_Premiership",
    "Spain_La_Liga",
    "Swiss_Super_League",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def markdown_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "_No rows._"
    text = df.head(max_rows).copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            text[col] = text[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            text[col] = text[col].astype("string").fillna("")
    lines = [
        "| " + " | ".join(str(c) for c in text.columns) + " |",
        "| " + " | ".join(["---"] * len(text.columns)) + " |",
    ]
    for _, row in text.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in text.columns) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(df)} rows._")
    return "\n".join(lines)


def league_tag(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def canonical_team(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    replacements = {
        "man utd": "manchester united",
        "man united": "manchester united",
        "man city": "manchester city",
        "spurs": "tottenham",
        "tottenham hotspur": "tottenham",
        "psg": "paris saint germain",
        "inter": "inter milan",
        "ac milan": "milan",
        "ny red bulls": "new york red bulls",
        "nycfc": "new york city",
        "nyc": "new york city",
        "sj earthquakes": "san jose earthquakes",
        "st louis city": "saint louis city",
        "brighton hove albion": "brighton",
        "wolverhampton wanderers": "wolves",
        "west ham united": "west ham",
        "leeds united": "leeds",
        "newcastle united": "newcastle",
        "sheffield united": "sheffield united",
        "olympique marseille": "marseille",
        "olympique lyonnais": "lyon",
        "bayer 04 leverkusen": "bayer leverkusen",
        "borussia monchengladbach": "borussia m gladbach",
        "borussia moenchengladbach": "borussia m gladbach",
        "1 fc koln": "koln",
        "fc koln": "koln",
        "athletic club bilbao": "athletic club",
        "atletico madrid": "atletico madrid",
    }
    text = replacements.get(text, text)
    parts = [part for part in text.split() if part not in {"fc", "cf", "sc", "afc", "club", "the"}]
    return " ".join(parts)


def fixture_join_key(match_date: Any, home: Any, away: Any) -> str:
    date = str(match_date or "").strip()[:10]
    return f"{date}|{canonical_team(home)}|{canonical_team(away)}"


def scored_files(root: Path, max_files: int = 0) -> list[Path]:
    files = sorted(root.glob("w*/03_scored/DEPLOY_COMBINED_SCORED_*.csv"))
    if max_files > 0:
        return files[:max_files]
    return files


def load_required_cells(root: Path, max_files: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    wanted = {"league", "match_date", "fixture_key", "home_team_name", "away_team_name"}
    for path in scored_files(root, max_files=max_files):
        frame = pd.read_csv(path, usecols=lambda c: c in wanted, low_memory=False)
        frame["match_date_dt"] = pd.to_datetime(frame.get("match_date"), errors="coerce")
        frame["calendar_year"] = frame["match_date_dt"].dt.year.astype("Int64")
        frame["league_tag"] = frame.get("league", "").map(league_tag)
        frame["scored_join_key"] = frame.apply(
            lambda row: fixture_join_key(row.get("match_date"), row.get("home_team_name"), row.get("away_team_name")),
            axis=1,
        )
        frames.append(frame)
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    rows = pd.concat(frames, ignore_index=True, sort=False)
    cells = (
        rows.dropna(subset=["calendar_year"])
        .groupby(["league", "league_tag", "calendar_year"], dropna=False)
        .agg(required_rows=("fixture_key", "size"), required_fixtures=("fixture_key", "nunique"))
        .reset_index()
    )
    cells["calendar_year"] = cells["calendar_year"].astype(int)
    return cells, rows


def parse_source_file(path: Path, family: str, *, source_layer: str) -> tuple[str, int] | None:
    if source_layer == "features":
        pattern = rf"api_{re.escape(family)}__(.+)__(\d{{4}})\.csv$"
    else:
        pattern = rf"{re.escape(family)}__(.+)__(\d{{4}})\.csv$"
    match = re.match(pattern, path.name)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def source_files(source_dir: Path, family: str, league: str, *, source_layer: str) -> list[Path]:
    prefix = f"api_{family}" if source_layer == "features" else family
    return sorted(source_dir.glob(f"{prefix}__{league}__*.csv"))


def calendar_slice(
    *,
    source_dir: Path,
    family: str,
    league: str,
    calendar_year: int,
    source_layer: str,
) -> tuple[pd.DataFrame, list[int]]:
    frames: list[pd.DataFrame] = []
    seasons: list[int] = []
    for path in source_files(source_dir, family, league, source_layer=source_layer):
        parsed = parse_source_file(path, family, source_layer=source_layer)
        if parsed is None:
            continue
        _, provider_season = parsed
        frame = safe_read_csv(path)
        if frame.empty:
            continue
        if "match_date" not in frame.columns and "fixture_id" in frame.columns:
            fixtures = safe_read_csv(source_dir / f"fixtures_master__{league}__{provider_season}.csv")
            if not fixtures.empty and "fixture_id" in fixtures.columns:
                fixture_cols = [col for col in KEYS if col in fixtures.columns and col not in frame.columns]
                fixture_cols = ["fixture_id", *fixture_cols]
                frame = frame.merge(
                    fixtures[fixture_cols].drop_duplicates("fixture_id"),
                    on="fixture_id",
                    how="left",
                )
        if "match_date" not in frame.columns:
            continue
        dates = pd.to_datetime(frame["match_date"], errors="coerce")
        part = frame[dates.dt.year.eq(calendar_year)].copy()
        if part.empty:
            continue
        part["season"] = calendar_year
        part["calendar_year"] = calendar_year
        part["api_provider_season"] = provider_season
        part["api_calendar_bridge_source"] = f"{path.name}"
        frames.append(part)
        seasons.append(provider_season)
    if not frames:
        return pd.DataFrame(), []
    out = pd.concat(frames, ignore_index=True, sort=False)
    subset = [col for col in ["fixture_id", "fixture_key", "player_id", "team_id"] if col in out.columns]
    if subset:
        out = out.drop_duplicates(subset=subset, keep="last")
    return out, sorted(set(seasons))


def write_calendar_family(
    *,
    source_dir: Path,
    output_dir: Path,
    family: str,
    league: str,
    calendar_year: int,
    source_layer: str,
    dry_run: bool,
) -> dict[str, Any]:
    frame, source_seasons = calendar_slice(
        source_dir=source_dir,
        family=family,
        league=league,
        calendar_year=calendar_year,
        source_layer=source_layer,
    )
    prefix = f"api_{family}" if source_layer == "features" else family
    output_path = output_dir / f"{prefix}__{league}__{calendar_year}.csv"
    if not frame.empty and not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
    return {
        "source_layer": source_layer,
        "family": family,
        "league_tag": league,
        "calendar_year": calendar_year,
        "source_provider_seasons": ",".join(str(v) for v in source_seasons),
        "rows": int(len(frame)),
        "unique_fixtures": int(frame["fixture_key"].nunique(dropna=True)) if "fixture_key" in frame.columns and not frame.empty else 0,
        "output_path": str(output_path),
        "write_status": "DRY_RUN" if dry_run else ("WROTE" if not frame.empty else "NO_SOURCE_ROWS"),
    }


def merge_by_fixture(base: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    if base.empty or other.empty or "fixture_id" not in base.columns or "fixture_id" not in other.columns:
        return base
    bridge_meta = {"calendar_year", "api_provider_season", "api_calendar_bridge_source"}
    cols = [
        col
        for col in other.columns
        if col not in KEYS and col not in bridge_meta and not col.startswith("api_calendar_bridge") and col not in base.columns
    ]
    cols = ["fixture_id", *cols]
    return base.merge(other[cols].drop_duplicates(subset=["fixture_id"]), on="fixture_id", how="left")


def build_enriched_from_calendar(
    *,
    features_out_dir: Path,
    normalized_out_dir: Path,
    league: str,
    calendar_year: int,
    dry_run: bool,
) -> dict[str, Any]:
    fixtures = safe_read_csv(normalized_out_dir / f"fixtures_master__{league}__{calendar_year}.csv")
    if fixtures.empty:
        fixtures = safe_read_csv(features_out_dir / f"api_team_rolling_features__{league}__{calendar_year}.csv")
    if fixtures.empty:
        return {"family": "enriched_fixture_features", "league_tag": league, "calendar_year": calendar_year, "rows": 0, "write_status": "NO_BASE_ROWS"}
    out = fixtures.copy()
    for family in ["team_rolling_features", "event_features", "lineup_features", "injury_features", "odds_features", "team_identity_features", "h2h_regime_features", "referee_profile_features"]:
        other = safe_read_csv(features_out_dir / f"api_{family}__{league}__{calendar_year}.csv")
        out = merge_by_fixture(out, other)
    for col in KEYS:
        if col not in out.columns:
            out[col] = np.nan
    out["season"] = calendar_year
    out["calendar_year"] = calendar_year
    output_path = features_out_dir / f"api_enriched_fixture_features__{league}__{calendar_year}.csv"
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False)
    return {
        "source_layer": "features",
        "family": "enriched_fixture_features",
        "league_tag": league,
        "calendar_year": calendar_year,
        "source_provider_seasons": "calendar_bridge",
        "rows": int(len(out)),
        "unique_fixtures": int(out["fixture_key"].nunique(dropna=True)) if "fixture_key" in out.columns else 0,
        "output_path": str(output_path),
        "write_status": "DRY_RUN" if dry_run else "WROTE",
    }


def ncol(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def scaled(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    mn = float(s.min()) if len(s) else 0.0
    mx = float(s.max()) if len(s) else 0.0
    if mx - mn <= 1e-12:
        return pd.Series(0.5, index=s.index, dtype="float64")
    return ((s - mn) / (mx - mn)).clip(0.0, 1.0)


def build_matchup_from_calendar(
    *,
    features_out_dir: Path,
    league: str,
    calendar_year: int,
    dry_run: bool,
) -> dict[str, Any]:
    team = safe_read_csv(features_out_dir / f"api_team_rolling_features__{league}__{calendar_year}.csv")
    event = safe_read_csv(features_out_dir / f"api_event_features__{league}__{calendar_year}.csv")
    if team.empty:
        return {"family": "matchup_interaction_features", "league_tag": league, "calendar_year": calendar_year, "rows": 0, "write_status": "NO_TEAM_ROWS"}
    out = team[[col for col in KEYS if col in team.columns]].copy()
    for col in KEYS:
        if col not in out.columns:
            out[col] = np.nan
    out["season"] = calendar_year
    if not event.empty and "fixture_id" in event.columns and "fixture_id" in team.columns:
        event_cols = ["fixture_id", *[col for col in event.columns if col not in KEYS]]
        merged = team.merge(event[event_cols].drop_duplicates("fixture_id"), on="fixture_id", how="left")
    else:
        merged = team.copy()
    out["home_attack_vs_away_defence_gap"] = ncol(merged, "home_goals_for_l5") - ncol(merged, "away_goals_against_l5")
    out["away_attack_vs_home_defence_gap"] = ncol(merged, "away_goals_for_l5") - ncol(merged, "home_goals_against_l5")
    out["home_attack_vs_away_restraint_gap"] = out["home_attack_vs_away_defence_gap"]
    out["away_attack_vs_home_restraint_gap"] = out["away_attack_vs_home_defence_gap"]
    out["home_buildup_resistance"] = 0.55 * scaled(ncol(merged, "home_pass_accuracy_l5")) + 0.45 * scaled(ncol(merged, "home_possession_l5"))
    out["away_buildup_resistance"] = 0.55 * scaled(ncol(merged, "away_pass_accuracy_l5")) + 0.45 * scaled(ncol(merged, "away_possession_l5"))
    out["home_press_intensity_proxy"] = 0.50 * scaled(ncol(merged, "home_fouls_for_l5")) + 0.50 * (1.0 - scaled(ncol(merged, "home_possession_l5")))
    out["away_press_intensity_proxy"] = 0.50 * scaled(ncol(merged, "away_fouls_for_l5")) + 0.50 * (1.0 - scaled(ncol(merged, "away_possession_l5")))
    out["home_press_vs_away_buildup_gap"] = out["home_press_intensity_proxy"] - out["away_buildup_resistance"]
    out["away_press_vs_home_buildup_gap"] = out["away_press_intensity_proxy"] - out["home_buildup_resistance"]
    out["press_mismatch_index"] = (out["home_press_vs_away_buildup_gap"] - out["away_press_vs_home_buildup_gap"]).abs()
    out["pressed_vs_pressed"] = out["home_press_intensity_proxy"] * out["away_press_intensity_proxy"]
    out["both_teams_chaos_interaction"] = ncol(merged, "home_chaos_index_l10") * ncol(merged, "away_chaos_index_l10")
    out["style_conflict_index"] = (ncol(merged, "home_possession_l5") - ncol(merged, "away_possession_l5")).abs()
    out["midfield_control_conflict_index"] = (ncol(merged, "home_pass_accuracy_l5") - ncol(merged, "away_pass_accuracy_l5")).abs()
    out["wing_mismatch_index"] = (ncol(merged, "home_corners_for_l5") - ncol(merged, "away_corners_for_l5")).abs()
    out["both_teams_booking_risk"] = ncol(merged, "home_cards_total_l5") + ncol(merged, "away_cards_total_l5") + ncol(merged, "home_fouls_for_l5") + ncol(merged, "away_fouls_for_l5")
    out["booking_pressure_interaction"] = scaled(out["both_teams_booking_risk"]) * scaled(out["both_teams_chaos_interaction"])
    out["goal_environment_interaction"] = (
        ncol(merged, "home_over25_rate_l5") * ncol(merged, "away_over25_rate_l5")
        + ncol(merged, "home_btts_rate_l5") * ncol(merged, "away_btts_rate_l5")
    ) / 2.0
    out["mutual_scoring_interaction"] = ncol(merged, "home_scored_rate_l5") * ncol(merged, "away_scored_rate_l5")
    out["mutual_conceding_interaction"] = ncol(merged, "home_conceded_rate_l5") * ncol(merged, "away_conceded_rate_l5")
    out["conversion_clash_index"] = ncol(merged, "home_shot_accuracy_l5") - ncol(merged, "away_shot_accuracy_l5")
    out["balanced_strength_flag"] = ((ncol(merged, "home_team_ppg_l5") - ncol(merged, "away_team_ppg_l5")).abs() <= 0.35).astype(int)
    out["high_volatility_balanced_flag"] = ((out["balanced_strength_flag"].eq(1)) & (scaled(out["both_teams_chaos_interaction"]) > 0.6)).astype(int)
    output_path = features_out_dir / f"api_matchup_interaction_features__{league}__{calendar_year}.csv"
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False)
    return {
        "source_layer": "features",
        "family": "matchup_interaction_features",
        "league_tag": league,
        "calendar_year": calendar_year,
        "source_provider_seasons": "calendar_bridge",
        "rows": int(len(out)),
        "unique_fixtures": int(out["fixture_key"].nunique(dropna=True)) if "fixture_key" in out.columns else 0,
        "output_path": str(output_path),
        "write_status": "DRY_RUN" if dry_run else "WROTE",
    }


def build_join_audit(scored_rows: pd.DataFrame, inventory: pd.DataFrame, features_out_dir: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if scored_rows.empty:
        return pd.DataFrame()
    for (league_tag_value, year), group in scored_rows.dropna(subset=["calendar_year"]).groupby(["league_tag", "calendar_year"], dropna=False):
        league_tag_text = str(league_tag_value)
        calendar_year = int(year)
        scored_keys = set(group["scored_join_key"].dropna().astype(str))
        for family in ["team_rolling_features", "player_rolling_features", "lineup_features", "injury_features", "event_features", "enriched_fixture_features", "matchup_interaction_features"]:
            path = features_out_dir / f"api_{family}__{league_tag_text}__{calendar_year}.csv"
            frame = safe_read_csv(path)
            api_keys: set[str] = set()
            if not frame.empty:
                needed = {"match_date", "home_team_name", "away_team_name"}
                if needed.issubset(frame.columns):
                    api_keys = set(
                        frame.apply(lambda row: fixture_join_key(row.get("match_date"), row.get("home_team_name"), row.get("away_team_name")), axis=1)
                        .dropna()
                        .astype(str)
                    )
            hits = len(scored_keys & api_keys)
            required = len(scored_keys)
            rate = hits / required if required else np.nan
            source_rows = inventory[
                inventory["source_layer"].eq("features")
                & inventory["family"].eq(family)
                & inventory["league_tag"].eq(league_tag_text)
                & inventory["calendar_year"].eq(calendar_year)
            ]
            caveats: list[str] = []
            if league_tag_text in WINTER_LEAGUE_HINTS and calendar_year >= 2025:
                caveats.append("WINTER_API_SEASON_BRIDGED")
            if rate < 0.80 and required >= 20:
                caveats.append("LOW_DATE_TEAM_JOIN")
            if source_rows.empty or int(source_rows["rows"].sum()) == 0:
                caveats.append("NO_LOCAL_SOURCE_ROWS")
            if league_tag_text in {"Belgium_Pro", "Scotland_Premiership", "Austria_Bundesliga", "Czech_First_League", "Denmark_Superliga", "South_Korea_K_League", "Swiss_Super_League"}:
                caveats.append("POSSIBLE_SPLIT_STAGE_OR_PLAYOFF_SCOPE")
            records.append(
                {
                    "league_tag": league_tag_text,
                    "calendar_year": calendar_year,
                    "feature_family": family,
                    "required_unique_fixtures": required,
                    "date_team_join_hits": hits,
                    "date_team_join_rate": rate,
                    "coverage_caveat": "|".join(caveats) if caveats else "OK",
                }
            )
    return pd.DataFrame(records)


def write_report(outdir: Path, *, scored_root: Path, features_out_dir: Path, normalized_out_dir: Path, inventory: pd.DataFrame, join_audit: pd.DataFrame, dry_run: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(outdir / "API_FOOTBALL_CALENDAR_YEAR_BRIDGE_INVENTORY.csv", index=False)
    join_audit.to_csv(outdir / "API_FOOTBALL_CALENDAR_YEAR_BRIDGE_JOIN_AUDIT.csv", index=False)
    summary = (
        inventory.groupby(["source_layer", "family", "write_status"], dropna=False)
        .agg(files=("output_path", "count"), rows=("rows", "sum"), unique_fixtures=("unique_fixtures", "sum"))
        .reset_index()
        if not inventory.empty
        else pd.DataFrame()
    )
    caveats = (
        join_audit.groupby(["coverage_caveat"], dropna=False)
        .agg(cells=("coverage_caveat", "size"), required_unique_fixtures=("required_unique_fixtures", "sum"), date_team_join_hits=("date_team_join_hits", "sum"))
        .reset_index()
        .sort_values(["required_unique_fixtures", "cells"], ascending=False)
        if not join_audit.empty
        else pd.DataFrame()
    )
    lines = [
        "# API-Football Calendar-Year Bridge",
        "",
        f"Generated: `{utc_now()}`",
        "",
        "Research-only. No deploy gates, ModelStore artifacts, live routing, raw API files, or provider-season files were changed.",
        "",
        "## Inputs",
        f"- scored root: `{scored_root}`",
        f"- calendar features out: `{features_out_dir}`",
        f"- calendar normalized out: `{normalized_out_dir}`",
        f"- dry run: `{dry_run}`",
        "",
        "## Methodology",
        "- Splits local API-Football files by `match_date` calendar year.",
        "- Rewrites the output `season` column to the benchmark calendar year.",
        "- For winter leagues, benchmark `2026` can therefore source provider season `2025`; benchmark `2025` can combine provider seasons `2024` and `2025`.",
        "- Synthetic matchup files are derived from calendar-year team/event rolling context when provider-season matchup files are absent.",
        "- Enriched fixture files are calendar-year merges of available local fixture/team/event/lineup/injury/odds/identity context.",
        "",
        "## Output Summary",
        markdown_table(summary),
        "",
        "## Coverage Caveats",
        markdown_table(caveats),
        "",
        "## Playoff/Split-Stage Warning",
        "- API-Football and FootyStats can handle domestic playoff, championship, relegation, and split-stage fixtures differently.",
        "- Low date/team join cells are not automatically model failures; they may be source-scope mismatches.",
        "- These cells should remain labelled in reports until a league-specific mapping or exclusion decision is made.",
    ]
    meta = {
        "generated_at": utc_now(),
        "scored_root": str(scored_root),
        "features_out_dir": str(features_out_dir),
        "normalized_out_dir": str(normalized_out_dir),
        "dry_run": dry_run,
        "inventory_rows": int(len(inventory)),
        "join_audit_rows": int(len(join_audit)),
    }
    (outdir / "summary.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-root", default=str(DEFAULT_SCORED_ROOT))
    parser.add_argument("--features-dir", default=str(DEFAULT_FEATURES_DIR))
    parser.add_argument("--normalized-dir", default=str(DEFAULT_NORMALIZED_DIR))
    parser.add_argument("--features-out-dir", default=str(DEFAULT_FEATURES_OUT_DIR))
    parser.add_argument("--normalized-out-dir", default=str(DEFAULT_NORMALIZED_OUT_DIR))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scored_root = Path(args.scored_root)
    features_dir = Path(args.features_dir)
    normalized_dir = Path(args.normalized_dir)
    features_out_dir = Path(args.features_out_dir)
    normalized_out_dir = Path(args.normalized_out_dir)
    cells, scored_rows = load_required_cells(scored_root, max_files=args.max_files)
    if cells.empty:
        raise SystemExit(f"No scored cells found under {scored_root}")

    inventory_rows: list[dict[str, Any]] = []
    for _, cell in cells.iterrows():
        league = str(cell["league_tag"])
        calendar_year = int(cell["calendar_year"])
        for family in FEATURE_FAMILIES:
            if family in {"enriched_fixture_features", "matchup_interaction_features"}:
                continue
            inventory_rows.append(
                write_calendar_family(
                    source_dir=features_dir,
                    output_dir=features_out_dir,
                    family=family,
                    league=league,
                    calendar_year=calendar_year,
                    source_layer="features",
                    dry_run=args.dry_run,
                )
            )
        for family in NORMALIZED_FAMILIES:
            inventory_rows.append(
                write_calendar_family(
                    source_dir=normalized_dir,
                    output_dir=normalized_out_dir,
                    family=family,
                    league=league,
                    calendar_year=calendar_year,
                    source_layer="normalized",
                    dry_run=args.dry_run,
                )
            )
        inventory_rows.append(
            build_enriched_from_calendar(
                features_out_dir=features_out_dir,
                normalized_out_dir=normalized_out_dir,
                league=league,
                calendar_year=calendar_year,
                dry_run=args.dry_run,
            )
        )
        inventory_rows.append(
            build_matchup_from_calendar(
                features_out_dir=features_out_dir,
                league=league,
                calendar_year=calendar_year,
                dry_run=args.dry_run,
            )
        )

    inventory = pd.DataFrame(inventory_rows)
    join_audit = build_join_audit(scored_rows, inventory, features_out_dir)
    write_report(
        Path(args.outdir),
        scored_root=scored_root,
        features_out_dir=features_out_dir,
        normalized_out_dir=normalized_out_dir,
        inventory=inventory,
        join_audit=join_audit,
        dry_run=args.dry_run,
    )
    wrote = inventory[inventory["write_status"].astype("string").isin(["WROTE", "DRY_RUN"])] if not inventory.empty else pd.DataFrame()
    print(f"Calendar cells: {len(cells)}")
    print(f"Inventory rows: {len(inventory)}")
    print(f"Wrote/planned files: {len(wrote)}")
    print(f"Outputs: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
