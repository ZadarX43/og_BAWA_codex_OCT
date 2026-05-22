#!/usr/bin/env python3
"""Build projected current-board player-event fixture input rows.

Research-only bridge. When current API-Football player/lineup normalized files
are not available yet, this script maps latest historical player-event profiles
onto current deploy-board fixtures so the player-intelligence shadow dashboard
can run against the same fixture keys as the live board.

These rows are projected lineup/profile proxies, not confirmed lineups and not
priced player-prop odds.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY_INPUT_DIR = ROOT / "data_sources" / "api_football" / "features" / "player_events"
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-07" / "player_event_current_board_fixture_inputs"

LEAGUE_TO_TAG = {
    "Belgium Pro": "Belgium_Pro",
    "Brazil Serie A": "Brazil_Serie_A",
    "Champions League": "Champions_League",
    "England Championship": "England_Championship",
    "England EFL League 1": "England_EFL_League_1",
    "England FA Cup": "England_FA_Cup",
    "England Premier League": "England_Premier_League",
    "Europa Conference": "Europa_Conference",
    "Europa League": "Europa_League",
    "France Ligue 1": "France_Ligue_1",
    "Germany Bundesliga": "Germany_Bundesliga",
    "Italy Serie A": "Italy_Serie_A",
    "Japan J1": "Japan_J1",
    "Netherlands Eredivisie": "Netherlands_Eredivisie",
    "Norway Eliteserien": "Norway_Eliteserien",
    "Portugal Liga": "Portugal_Liga",
    "Scotland Premiership": "Scotland_Premiership",
    "Spain La Liga": "Spain_La_Liga",
    "USA MLS": "USA_MLS",
}

PROFILE_DROP_COLS = {"__source_file", "__source_league_tag", "__source_season"}


def norm_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compact_team_key(value: Any) -> str:
    text = norm_text(value)
    text = re.sub(r"\b(fc|cf|sc|afc|cd|ud|rcd|1)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    aliases = {
        "bayern munchen": "bayern munich",
        "bayern m nchen": "bayern munich",
        "fc barcelona": "barcelona",
        "athletic club bilbao": "athletic club",
        "levante ud": "levante",
        "heidenheim": "heidenheim",
        "la galaxy": "los angeles galaxy",
        "new york rb": "new york red bulls",
        "sj earthquakes": "san jose earthquakes",
        "sporting kc": "sporting kansas city",
        "sint truiden": "st truiden",
        "union saint gilloise": "union st gilloise",
    }
    return aliases.get(text, text)


def num(values: Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def discover_board_files(board_dir: Path) -> list[Path]:
    return sorted(board_dir.glob("*__DEPLOY_TIER_*__PRESET_V1__FTR_accuracy.csv"))


def load_current_fixtures(board_dir: Path) -> pd.DataFrame:
    frames = []
    cols = ["fixture_key", "match_date", "league", "home_team_name", "away_team_name"]
    for path in discover_board_files(board_dir):
        header = pd.read_csv(path, nrows=0)
        usecols = [col for col in cols if col in header.columns]
        if "fixture_key" not in usecols:
            continue
        frame = pd.read_csv(path, usecols=usecols, low_memory=False)
        for col in cols:
            if col not in frame.columns:
                frame[col] = ""
        frame["source_board_file"] = path.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=cols)
    fixtures = pd.concat(frames, ignore_index=True, sort=False)
    fixtures = fixtures.dropna(subset=["fixture_key"]).drop_duplicates("fixture_key", keep="first")
    fixtures["match_date"] = pd.to_datetime(fixtures["match_date"], errors="coerce")
    fixtures["league_tag"] = fixtures["league"].map(LEAGUE_TO_TAG)
    fixtures = fixtures[fixtures["league_tag"].notna()].copy()
    return fixtures.sort_values(["match_date", "league", "fixture_key"]).reset_index(drop=True)


def parse_history_file(path: Path) -> tuple[str, int] | None:
    match = re.match(r"player_events_fixture_input__(.+)__(\d{4})\.csv$", path.name)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def load_history_profiles(input_dir: Path, league_tag: str) -> pd.DataFrame:
    frames = []
    for path in sorted(input_dir.glob(f"player_events_fixture_input__{league_tag}__*.csv")):
        parsed = parse_history_file(path)
        if parsed is None:
            continue
        _, season = parsed
        if season >= 2026:
            continue
        frame = pd.read_csv(path, low_memory=False)
        if frame.empty:
            continue
        frame["__source_file"] = path.name
        frame["__source_league_tag"] = league_tag
        frame["__source_season"] = season
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    history = pd.concat(frames, ignore_index=True, sort=False)
    history["match_date"] = pd.to_datetime(history.get("match_date"), errors="coerce")
    history["team_key"] = history.get("team_name", pd.Series("", index=history.index)).map(compact_team_key)
    history["player_key"] = history.get("player_name", pd.Series("", index=history.index)).map(norm_text)
    for col in [
        "expected_start_flag",
        "expected_minutes",
        "player_quality_score_l5",
        "player_form_rating_l5",
        "minutes_last_3_matches",
        "shots_per90",
        "shots_on_target_per90",
        "fouls_won_per90",
        "fouls_per90",
        "tackles_per90",
    ]:
        if col not in history.columns:
            history[col] = 0.0
        history[col] = num(history[col]).fillna(0.0)
    history = history.dropna(subset=["team_key", "player_key"])
    history = history[history["player_key"].astype(str).ne("") & history["team_key"].astype(str).ne("")]
    history = history.sort_values(["match_date", "fixture_key"]).drop_duplicates(["team_key", "player_key"], keep="last")
    history["_profile_rank"] = (
        1000.0 * history["expected_start_flag"]
        + 2.0 * history["expected_minutes"]
        + 20.0 * history["player_quality_score_l5"]
        + 4.0 * history["player_form_rating_l5"]
        + 0.20 * history["minutes_last_3_matches"]
        + 3.0 * history["shots_per90"]
        + 2.0 * history["shots_on_target_per90"]
        + 1.5 * history["fouls_won_per90"]
        + history["tackles_per90"]
    )
    return history.sort_values(["team_key", "_profile_rank"], ascending=[True, False]).reset_index(drop=True)


def lookup_team_profiles(history: pd.DataFrame, team_name: str, max_players: int) -> tuple[pd.DataFrame, str]:
    if history.empty:
        return pd.DataFrame(), "NO_HISTORY"
    key = compact_team_key(team_name)
    exact = history[history["team_key"].eq(key)].copy()
    if not exact.empty:
        return exact.head(max_players), "EXACT_TEAM_PROFILE"
    candidates = []
    for candidate in sorted(history["team_key"].dropna().astype(str).unique()):
        if len(key) >= 5 and (key in candidate or candidate in key):
            candidates.append((abs(len(candidate) - len(key)), candidate))
    if not candidates:
        return pd.DataFrame(), "NO_TEAM_PROFILE_MATCH"
    _, best = sorted(candidates, key=lambda item: item[0])[0]
    return history[history["team_key"].eq(best)].head(max_players).copy(), f"FUZZY_TEAM_PROFILE:{best}"


def project_fixture_rows(fixtures: pd.DataFrame, history_by_league: dict[str, pd.DataFrame], max_players: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    coverage = []
    for _, fixture in fixtures.iterrows():
        league_tag = str(fixture["league_tag"])
        history = history_by_league.get(league_tag, pd.DataFrame())
        fixture_rows = 0
        for side, team_col, opponent_col in [
            ("HOME", "home_team_name", "away_team_name"),
            ("AWAY", "away_team_name", "home_team_name"),
        ]:
            profiles, mode = lookup_team_profiles(history, str(fixture[team_col]), max_players=max_players)
            coverage.append(
                {
                    "fixture_key": fixture["fixture_key"],
                    "league": fixture["league"],
                    "league_tag": league_tag,
                    "team_name": fixture[team_col],
                    "side": side,
                    "projection_match_mode": mode,
                    "projected_players": len(profiles),
                }
            )
            for _, profile in profiles.iterrows():
                rec = {
                    col: profile[col]
                    for col in profile.index
                    if col not in PROFILE_DROP_COLS and not str(col).startswith("_")
                }
                rec.update(
                    {
                        "fixture_key": fixture["fixture_key"],
                        "match_date": fixture["match_date"].date().isoformat() if pd.notna(fixture["match_date"]) else "",
                        "competition": fixture["league"],
                        "league": fixture["league"],
                        "league_tag": league_tag,
                        "home_team_name": fixture["home_team_name"],
                        "away_team_name": fixture["away_team_name"],
                        "team_name": fixture[team_col],
                        "player_team_side": side,
                        "projection_input_mode": "CURRENT_BOARD_LATEST_HISTORICAL_PROFILE_PROXY",
                        "projection_team_match_mode": mode,
                        "lineup_watch_flags": "PROJECTED_LINEUP_PROFILE_NOT_CONFIRMED",
                    }
                )
                if "expected_start_flag" in rec:
                    rec["expected_start_flag"] = int(float(rec.get("expected_start_flag") or 0))
                if "expected_minutes" in rec:
                    rec["expected_minutes"] = max(45.0, float(rec.get("expected_minutes") or 0.0))
                rows.append(rec)
                fixture_rows += 1
        coverage[-1]["fixture_projected_rows"] = fixture_rows
    projected = pd.DataFrame(rows)
    coverage_df = pd.DataFrame(coverage)
    if not projected.empty:
        projected = projected.drop(columns=["team_key", "player_key"], errors="ignore")
        projected = projected.sort_values(["match_date", "fixture_key", "player_team_side", "team_name", "player_name"])
    return projected, coverage_df


def write_report(outdir: Path, fixtures: pd.DataFrame, projected: pd.DataFrame, coverage: pd.DataFrame) -> None:
    league_summary = (
        coverage.groupby(["league", "league_tag"], dropna=False)
        .agg(
            fixtures=("fixture_key", "nunique"),
            team_sides=("team_name", "count"),
            matched_team_sides=("projected_players", lambda s: int((num(s) > 0).sum())),
            projected_players=("projected_players", "sum"),
        )
        .reset_index()
        if not coverage.empty
        else pd.DataFrame()
    )
    lines = [
        "# Current Board Player-Event Fixture Inputs",
        "",
        "Research-only projected player-event fixture inputs for current deploy-board fixture keys.",
        "",
        "## Safety",
        "- Uses latest historical player profiles; does not claim confirmed lineups.",
        "- No priced player-prop odds.",
        "- No deploy promotion or source tier mutation.",
        "",
        "## Counts",
        f"- board fixtures mapped to supported league tags: `{len(fixtures)}`",
        f"- projected player rows: `{len(projected)}`",
        f"- projected fixtures: `{projected['fixture_key'].nunique() if not projected.empty else 0}`",
        "",
        "## League Coverage",
        markdown_table(league_summary),
    ]
    (outdir / "CURRENT_BOARD_PLAYER_EVENT_FIXTURE_INPUTS.md").write_text("\n".join(lines) + "\n")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row.get(col, "")).replace("|", "/") for col in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-dir", required=True)
    parser.add_argument("--history-input-dir", type=Path, default=DEFAULT_HISTORY_INPUT_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--max-players-per-team", type=int, default=14)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    inputs_dir = args.outdir / "fixture_inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    fixtures = load_current_fixtures(Path(args.board_dir))
    if fixtures.empty:
        raise SystemExit(f"No supported current fixtures found in board dir: {args.board_dir}")

    history_by_league = {
        league_tag: load_history_profiles(args.history_input_dir, league_tag)
        for league_tag in sorted(fixtures["league_tag"].dropna().astype(str).unique())
    }
    projected, coverage = project_fixture_rows(fixtures, history_by_league, max_players=args.max_players_per_team)
    if projected.empty:
        raise SystemExit("No projected player-event rows generated.")

    output_paths = []
    for league_tag, group in projected.groupby("league_tag", dropna=False):
        tag = str(league_tag)
        if tag in {"nan", "", "None"}:
            tag = "LIVE_BOARD"
        out = inputs_dir / f"player_events_fixture_input__{tag}__{args.season}.csv"
        group.drop(columns=["__source_league_tag", "__source_season", "__source_file"], errors="ignore").to_csv(out, index=False)
        output_paths.append(out)

    projected.to_csv(args.outdir / "CURRENT_BOARD_PLAYER_EVENT_FIXTURE_INPUTS_ALL.csv", index=False)
    coverage.to_csv(args.outdir / "CURRENT_BOARD_PLAYER_EVENT_FIXTURE_INPUTS_COVERAGE.csv", index=False)
    write_report(args.outdir, fixtures, projected, coverage)

    league_tags = ",".join(sorted({path.name.split("__")[1] for path in output_paths}))
    print(f"WROTE {args.outdir}")
    print(f"fixture_input_dir={inputs_dir}")
    print(f"projected_rows={len(projected)} fixtures={projected['fixture_key'].nunique()}")
    print(f"league_tags={league_tags}")


if __name__ == "__main__":
    main()
