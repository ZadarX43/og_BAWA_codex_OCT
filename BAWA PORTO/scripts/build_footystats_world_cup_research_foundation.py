#!/usr/bin/env python3
"""Build a timestamp-safe FootyStats World Cup research foundation.

This is not the production ingest path. It creates research-only normalized
frames from the World Cup FootyStats drops:
- match spine with inferred tournament stage and graded market outcomes
- lagged in-tournament team state available before each kickoff
- static player/squad profile frame with post-tournament performance columns
  deliberately excluded
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_DROP = Path("/Users/hughwade/Desktop/FOOTYSTATS_DROP")
DEFAULT_OUTDIR = Path("data_sources/footystats_world_cup/research_foundation")

FILE_RE = re.compile(
    r"^international-fifa-world-cup-(?P<host>[a-z0-9-]+)-(?P<kind>matches|teams|players)-(?P<start>\d{4})-to-(?P<end>\d{4})-stats(?: \((?P<dup>\d+)\))?\.csv$",
    re.IGNORECASE,
)

PLAYER_STATIC_COLUMNS = [
    "full_name",
    "age",
    "birthday",
    "birthday_GMT",
    "league",
    "season",
    "position",
    "Current Club",
    "nationality",
]


def parse_file(path: Path) -> dict[str, Any] | None:
    match = FILE_RE.match(path.name)
    if not match:
        return None
    return {
        "path": path,
        "host_slug": match.group("host"),
        "kind": match.group("kind").lower(),
        "season": int(match.group("start")),
        "duplicate_suffix": match.group("dup") or "",
    }


def canonical_files(drop: Path) -> dict[tuple[int, str], Path]:
    candidates: list[dict[str, Any]] = []
    for path in sorted(drop.glob("international-fifa-world-cup*.csv")):
        parsed = parse_file(path)
        if parsed:
            parsed["_clean_rank"] = 0 if parsed["duplicate_suffix"] == "" else 1
            candidates.append(parsed)
    out: dict[tuple[int, str], Path] = {}
    for row in sorted(candidates, key=lambda r: (r["season"], r["kind"], r["_clean_rank"], r["duplicate_suffix"])):
        key = (int(row["season"]), str(row["kind"]))
        out.setdefault(key, row["path"])
    return out


def norm_slug(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+national\s+team$", "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def infer_stage(idx: int) -> str:
    if idx < 48:
        return "GROUP_STAGE"
    if idx < 56:
        return "ROUND_OF_16"
    if idx < 60:
        return "QUARTER_FINAL"
    if idx < 62:
        return "SEMI_FINAL"
    if idx == 62:
        return "THIRD_PLACE"
    return "FINAL"


def result_label(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "HOME"
    if away_goals > home_goals:
        return "AWAY"
    return "DRAW"


def build_match_frame(path: Path, season: int) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["_kickoff_dt"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    df = df.sort_values(["_kickoff_dt", "home_team_name", "away_team_name"]).reset_index(drop=True)
    df["season"] = season
    df["competition"] = "World Cup"
    df["league_tag"] = "World_Cup"
    df["tournament_stage"] = [infer_stage(i) for i in range(len(df))]
    df["group_matchday"] = pd.to_numeric(df.get("Game Week"), errors="coerce").astype("Int64")
    df["fixture_key"] = [
        f"World_Cup_{season}_{int(row.timestamp)}_{norm_slug(row.home_team_name)}_vs_{norm_slug(row.away_team_name)}"
        for row in df.itertuples(index=False)
    ]
    home_goals = pd.to_numeric(df["home_team_goal_count"], errors="coerce").fillna(0).astype(int)
    away_goals = pd.to_numeric(df["away_team_goal_count"], errors="coerce").fillna(0).astype(int)
    total_goals = home_goals + away_goals
    df["actual_ftr"] = [result_label(int(h), int(a)) for h, a in zip(home_goals, away_goals)]
    df["actual_btts"] = ((home_goals > 0) & (away_goals > 0)).map({True: "YES", False: "NO"})
    df["actual_ou25"] = (total_goals > 2.5).map({True: "OVER", False: "UNDER"})
    df["actual_over15"] = (total_goals > 1.5).map({True: "OVER", False: "UNDER"})
    keep = [
        "fixture_key",
        "season",
        "competition",
        "league_tag",
        "timestamp",
        "date_GMT",
        "_kickoff_dt",
        "status",
        "tournament_stage",
        "group_matchday",
        "home_team_name",
        "away_team_name",
        "home_team_goal_count",
        "away_team_goal_count",
        "total_goal_count",
        "actual_ftr",
        "actual_btts",
        "actual_ou25",
        "actual_over15",
        "odds_ft_home_team_win",
        "odds_ft_draw",
        "odds_ft_away_team_win",
        "odds_ft_over15",
        "odds_ft_over25",
        "odds_btts_yes",
        "odds_btts_no",
        "Pre-Match PPG (Home)",
        "Pre-Match PPG (Away)",
        "Home Team Pre-Match xG",
        "Away Team Pre-Match xG",
        "average_goals_per_match_pre_match",
        "btts_percentage_pre_match",
        "over_15_percentage_pre_match",
        "over_25_percentage_pre_match",
        "stadium_name",
        "attendance",
        "referee",
    ]
    return df[[col for col in keep if col in df.columns]].rename(columns={"_kickoff_dt": "kickoff_dt"})


def build_lagged_team_state(matches: pd.DataFrame) -> pd.DataFrame:
    state: dict[tuple[int, str], dict[str, float]] = {}
    rows: list[dict[str, Any]] = []
    ordered = matches.sort_values(["season", "kickoff_dt", "fixture_key"]).copy()
    for row in ordered.itertuples(index=False):
        season = int(row.season)
        home = str(row.home_team_name)
        away = str(row.away_team_name)
        home_goals = int(getattr(row, "home_team_goal_count"))
        away_goals = int(getattr(row, "away_team_goal_count"))
        for side, team, opp, gf, ga in [("home", home, away, home_goals, away_goals), ("away", away, home, away_goals, home_goals)]:
            key = (season, team)
            prior = state.setdefault(
                key,
                {
                    "played": 0,
                    "points": 0,
                    "wins": 0,
                    "draws": 0,
                    "losses": 0,
                    "goals_for": 0,
                    "goals_against": 0,
                    "btts": 0,
                    "over25": 0,
                },
            )
            played = prior["played"]
            rows.append(
                {
                    "fixture_key": row.fixture_key,
                    "season": season,
                    "kickoff_dt": row.kickoff_dt,
                    "team_name": team,
                    "opponent_name": opp,
                    "side": side,
                    "prior_matches_played": int(played),
                    "prior_points": int(prior["points"]),
                    "prior_points_per_match": float(prior["points"] / played) if played else None,
                    "prior_goal_diff": int(prior["goals_for"] - prior["goals_against"]),
                    "prior_goals_for_per_match": float(prior["goals_for"] / played) if played else None,
                    "prior_goals_against_per_match": float(prior["goals_against"] / played) if played else None,
                    "prior_btts_rate": float(prior["btts"] / played) if played else None,
                    "prior_over25_rate": float(prior["over25"] / played) if played else None,
                }
            )
        for team, gf, ga in [(home, home_goals, away_goals), (away, away_goals, home_goals)]:
            prior = state[(season, team)]
            prior["played"] += 1
            prior["goals_for"] += gf
            prior["goals_against"] += ga
            prior["btts"] += int(gf > 0 and ga > 0)
            prior["over25"] += int(gf + ga > 2)
            if gf > ga:
                prior["points"] += 3
                prior["wins"] += 1
            elif gf == ga:
                prior["points"] += 1
                prior["draws"] += 1
            else:
                prior["losses"] += 1
    return pd.DataFrame(rows)


def build_player_static(path: Path, season: int) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    keep = [col for col in PLAYER_STATIC_COLUMNS if col in df.columns]
    out = df[keep].copy()
    out["season"] = season
    out["competition"] = "World Cup"
    out = out.rename(columns={"Current Club": "squad_team_name"})
    out["squad_team_slug"] = out["squad_team_name"].map(norm_slug)
    out["player_slug"] = out["full_name"].map(norm_slug)
    return out


def build_team_static(path: Path, season: int) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    keep = [col for col in ["team_name", "common_name", "season", "country"] if col in df.columns]
    out = df[keep].copy()
    out["season"] = season
    out["competition"] = "World Cup"
    out["team_slug"] = out.get("common_name", out.get("team_name")).map(norm_slug)
    return out


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            view[col] = view[col].astype("string").fillna("")
    lines = [
        "| " + " | ".join(view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "/") for col in view.columns) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drop", default=str(DEFAULT_DROP))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    args = parser.parse_args()

    drop = Path(args.drop)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    files = canonical_files(drop)
    seasons = sorted({season for season, _kind in files})

    match_frames: list[pd.DataFrame] = []
    player_frames: list[pd.DataFrame] = []
    team_frames: list[pd.DataFrame] = []
    for season in seasons:
        if (season, "matches") in files:
            match_frames.append(build_match_frame(files[(season, "matches")], season))
        if (season, "players") in files:
            player_frames.append(build_player_static(files[(season, "players")], season))
        if (season, "teams") in files:
            team_frames.append(build_team_static(files[(season, "teams")], season))

    matches = pd.concat(match_frames, ignore_index=True, sort=False) if match_frames else pd.DataFrame()
    team_state = build_lagged_team_state(matches) if not matches.empty else pd.DataFrame()
    players = pd.concat(player_frames, ignore_index=True, sort=False) if player_frames else pd.DataFrame()
    teams = pd.concat(team_frames, ignore_index=True, sort=False) if team_frames else pd.DataFrame()

    matches.to_csv(outdir / "footystats_world_cup_match_spine.csv", index=False)
    team_state.to_csv(outdir / "footystats_world_cup_lagged_team_state_long.csv", index=False)
    players.to_csv(outdir / "footystats_world_cup_player_static_profiles.csv", index=False)
    teams.to_csv(outdir / "footystats_world_cup_team_static_profiles.csv", index=False)

    season_summary = (
        matches.groupby("season", dropna=False)
        .agg(
            matches=("fixture_key", "size"),
            group_stage=("tournament_stage", lambda s: int((s == "GROUP_STAGE").sum())),
            knockout=("tournament_stage", lambda s: int((s != "GROUP_STAGE").sum())),
            teams=("home_team_name", lambda s: len(set(s).union(set(matches.loc[s.index, "away_team_name"])))),
            btts_rate=("actual_btts", lambda s: float((s == "YES").mean())),
            over25_rate=("actual_ou25", lambda s: float((s == "OVER").mean())),
            home_win_rate=("actual_ftr", lambda s: float((s == "HOME").mean())),
            draw_rate=("actual_ftr", lambda s: float((s == "DRAW").mean())),
        )
        .reset_index()
    )
    season_summary.to_csv(outdir / "footystats_world_cup_foundation_summary.csv", index=False)

    field_policy = pd.DataFrame(
        [
            {"field_family": "match outcomes/goals", "historical_use": "labels_only", "pre_match_safe": "no"},
            {"field_family": "match odds", "historical_use": "pre_match_market_context", "pre_match_safe": "yes_if_timestamped_or_source_assumed_close"},
            {"field_family": "FootyStats pre-match PPG/xG/percentages", "historical_use": "candidate_features", "pre_match_safe": "yes_needs_source_contract"},
            {"field_family": "lagged team state from prior tournament matches", "historical_use": "candidate_features", "pre_match_safe": "yes"},
            {"field_family": "player static profile age/position/squad", "historical_use": "candidate_features", "pre_match_safe": "yes_if_squad_known_pre_match"},
            {"field_family": "player tournament performance aggregates", "historical_use": "post_event_analysis_only", "pre_match_safe": "no"},
            {"field_family": "team tournament aggregate file performance", "historical_use": "post_event_analysis_only", "pre_match_safe": "no"},
        ]
    )
    field_policy.to_csv(outdir / "footystats_world_cup_field_policy.csv", index=False)

    summary = [
        "# FootyStats World Cup Research Foundation",
        "",
        "Research-only normalized files built from the audited FootyStats World Cup drop.",
        "",
        "## Season Summary",
        markdown_table(season_summary),
        "",
        "## Field Policy",
        markdown_table(field_policy),
        "",
        "## Outputs",
        f"- `{outdir / 'footystats_world_cup_match_spine.csv'}`",
        f"- `{outdir / 'footystats_world_cup_lagged_team_state_long.csv'}`",
        f"- `{outdir / 'footystats_world_cup_player_static_profiles.csv'}`",
        f"- `{outdir / 'footystats_world_cup_team_static_profiles.csv'}`",
        f"- `{outdir / 'footystats_world_cup_field_policy.csv'}`",
        "",
        "## Notes",
        "- Knockout stage labels are inferred by chronological position in the 32-team World Cup format: 48 group matches, 8 round-of-16, 4 quarter-finals, 2 semi-finals, third-place match, final.",
        "- This foundation intentionally excludes player and team tournament performance aggregate fields from pre-match feature files.",
        "- First group matches still need external pre-tournament strength: FIFA/Elo, qualifying form, squad club minutes/form, injuries, travel/rest, and market odds.",
    ]
    (outdir / "SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Matches: {len(matches)}")
    print(f"Lagged team-state rows: {len(team_state)}")
    print(f"Player static rows: {len(players)}")
    print(f"Team static rows: {len(teams)}")
    print(f"Output: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
