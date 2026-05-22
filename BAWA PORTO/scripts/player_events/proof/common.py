from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NORMALIZED_DIR = REPO_ROOT / "data_sources" / "api_football" / "normalized"
FEATURES_DIR = REPO_ROOT / "data_sources" / "api_football" / "features" / "player_events"
REPORTS_DIR = REPO_ROOT / "reports" / "player_events" / "proof"
MERGED_DIR = REPO_ROOT / "Matches" / "__merged__"
TARGET_LEAGUES = ("England_Premier_League", "Spain_La_Liga")
TARGET_SEASONS = (2022, 2023, 2024)

from scripts.player_events.build_fixture_style_overlay import build_fixture_style_overlay
from scripts.player_events.build_player_form_quality_overlay import build_player_form_quality_overlay
from scripts.player_events.build_referee_profiles import build_referee_profiles
from scripts.player_events.build_og_goal_environment_overlay import build_og_goal_environment_overlay
from scripts.player_events.build_player_events_fixture_input import build_player_events_fixture_input


DEFAULTS = {
    "referee": lambda league, season: FEATURES_DIR / f"referee_profiles__{league}__{season}.csv",
    "style": lambda league, season: FEATURES_DIR / f"fixture_style_overlay__{league}__{season}.csv",
    "quality": lambda league, season: FEATURES_DIR / f"player_form_quality_overlay__{league}__{season}.csv",
    "og": lambda league, season: FEATURES_DIR / f"og_goal_environment_overlay__{league}__{season}.csv",
    "fixture_input": lambda league, season: FEATURES_DIR / f"player_events_fixture_input__{league}__{season}.csv",
}


def normalized_path(prefix: str, league: str, season: int) -> Path:
    return NORMALIZED_DIR / f"{prefix}__{league}__{season}.csv"


def ensure_feature_inputs(league: str, season: int, force: bool = False) -> Path:
    fixtures_csv = normalized_path("fixtures_master", league, season)
    player_stats_csv = normalized_path("match_player_stats", league, season)
    team_stats_csv = normalized_path("match_team_stats", league, season)
    lineups_csv = normalized_path("lineups", league, season)
    injuries_csv = normalized_path("injuries", league, season)
    events_csv = normalized_path("match_events", league, season)
    lineup_features_csv = REPO_ROOT / "data_sources" / "api_football" / "features" / f"api_lineup_features__{league}__{season}.csv"
    manual_side_csv = FEATURES_DIR / f"manual_side_enrichment__{league}__{season}.csv"
    merged_csv = MERGED_DIR / f"{league}__merged.csv"

    referee_csv = DEFAULTS["referee"](league, season)
    style_csv = DEFAULTS["style"](league, season)
    quality_csv = DEFAULTS["quality"](league, season)
    og_csv = DEFAULTS["og"](league, season)
    fixture_input_csv = DEFAULTS["fixture_input"](league, season)

    referee_csv.parent.mkdir(parents=True, exist_ok=True)

    if force or not referee_csv.exists():
        build_referee_profiles(str(fixtures_csv), str(team_stats_csv), str(events_csv), str(referee_csv))
    if force or not style_csv.exists():
        build_fixture_style_overlay(str(fixtures_csv), str(team_stats_csv), str(player_stats_csv), str(style_csv))
    if force or not quality_csv.exists():
        build_player_form_quality_overlay(str(fixtures_csv), str(player_stats_csv), str(lineups_csv), str(quality_csv))
    if (force or not og_csv.exists()) and merged_csv.exists():
        build_og_goal_environment_overlay(league, season, str(merged_csv), str(fixtures_csv), str(og_csv))
    if force or not fixture_input_csv.exists():
        build_player_events_fixture_input(
            league,
            season,
            str(fixtures_csv),
            str(player_stats_csv),
            str(team_stats_csv),
            str(lineups_csv),
            str(injuries_csv),
            str(referee_csv),
            str(og_csv) if og_csv.exists() else "",
            str(style_csv) if style_csv.exists() else "",
            str(quality_csv) if quality_csv.exists() else "",
            str(lineup_features_csv) if lineup_features_csv.exists() else "",
            str(manual_side_csv) if manual_side_csv.exists() else "",
            str(fixture_input_csv),
        )
    return fixture_input_csv


def load_fixture_inputs(leagues: tuple[str, ...] = TARGET_LEAGUES, seasons: tuple[int, ...] = TARGET_SEASONS, force: bool = False) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for league in leagues:
        for season in seasons:
            csv_path = ensure_feature_inputs(league, season, force=force)
            if csv_path.exists():
                df = pd.read_csv(csv_path, low_memory=False)
                df["league_tag"] = league
                df["season_tag"] = season
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["match_date"] = pd.to_datetime(out["match_date"], errors="coerce")
    return out


def safe_team_id(row: pd.Series) -> int | None:
    try:
        side = str(row.get("player_team_side", ""))
        if side == "HOME":
            return int(row.get("home_team_id"))
        if side == "AWAY":
            return int(row.get("away_team_id"))
    except Exception:
        return None
    return None
