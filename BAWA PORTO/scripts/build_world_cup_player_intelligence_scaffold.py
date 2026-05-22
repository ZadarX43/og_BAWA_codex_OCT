#!/usr/bin/env python3
"""
Build research-only World Cup player/squad intelligence sidecars.

Inputs are intentionally pre-tournament safe:
- API-Football /players roster pages when available
- historical World Cup squad static priors
- optional external player ratings/values from Kaggle or other documented source

No production routing, training artifacts, or deploy gates are changed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_LAUNCH = Path("data_sources/footystats_world_cup/launch_2026/world_cup_2026_launch_scaffold.csv")
DEFAULT_HISTORICAL_PLAYERS = Path(
    "data_sources/footystats_world_cup/research_foundation/footystats_world_cup_player_static_profiles.csv"
)
DEFAULT_MACRO_TEAMS = Path("data_sources/footystats_world_cup/macro_prior_engine/world_cup_team_macro_strength_2026.csv")
DEFAULT_API_PLAYERS_RAW = Path("data_sources/api_football/raw/players__league_1__season_2026__players.jsonl")
DEFAULT_OUTDIR = Path("data_sources/footystats_world_cup/player_intelligence_2026")


POSITION_BUCKETS = {
    "goalkeeper": "gk",
    "keeper": "gk",
    "gk": "gk",
    "defender": "def",
    "defence": "def",
    "defense": "def",
    "midfielder": "mid",
    "midfield": "mid",
    "attacker": "att",
    "forward": "att",
    "striker": "att",
}


EXTERNAL_PLAYER_NUMERIC_FIELDS = [
    "player_rating",
    "domestic_player_rating",
    "market_value_eur",
    "minutes_last_season",
    "goals_last_season",
    "assists_last_season",
    "xg_last_season",
    "xa_last_season",
    "progressive_actions",
    "defensive_actions",
]


def slugify(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.lower().strip()
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def to_num(series: pd.Series | None, default: float = math.nan) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce").fillna(default)


def bucket_position(value: object) -> str:
    text = slugify(value)
    if not text:
        return "unknown"
    if text in {"g"}:
        return "gk"
    if text in {"d", "df", "cb", "lb", "rb", "lwb", "rwb"}:
        return "def"
    if text in {"m", "mf", "cm", "dm", "am", "lm", "rm"}:
        return "mid"
    if text in {"f", "fw", "st", "cf", "lw", "rw"}:
        return "att"
    for key, bucket in POSITION_BUCKETS.items():
        if key in text:
            return bucket
    return "unknown"


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_api_players(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for payload in iter_jsonl(path):
        season = int((payload.get("parameters") or {}).get("season") or 0)
        for item in payload.get("response") or []:
            player = item.get("player") or {}
            stats = (item.get("statistics") or [{}])[0] or {}
            team = stats.get("team") or {}
            games = stats.get("games") or {}
            rows.append(
                {
                    "season": season,
                    "api_player_id": player.get("id"),
                    "player_name": player.get("name") or "",
                    "player_slug": slugify(player.get("name")),
                    "age": player.get("age"),
                    "nationality": player.get("nationality") or "",
                    "team_id": team.get("id"),
                    "team_name": team.get("name") or "",
                    "team_slug": slugify(team.get("name") or player.get("nationality")),
                    "position": games.get("position") or "",
                    "position_bucket": bucket_position(games.get("position")),
                    "number": games.get("number"),
                    "captain_flag": int(bool(games.get("captain"))),
                    "source": "API_FOOTBALL_PLAYERS",
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "api_player_id",
                "player_name",
                "player_slug",
                "age",
                "nationality",
                "team_id",
                "team_name",
                "team_slug",
                "position",
                "position_bucket",
                "number",
                "captain_flag",
                "source",
            ]
        )
    for col in ["age", "team_id", "number"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.drop_duplicates(subset=["team_slug", "player_slug"], keep="first").reset_index(drop=True)


def build_historical_team_priors(players: pd.DataFrame) -> pd.DataFrame:
    if players.empty:
        return pd.DataFrame(columns=["team_slug"])
    df = players.copy()
    df["team_slug"] = df.get("squad_team_slug", df.get("squad_team_name", "")).map(slugify)
    df["position_bucket"] = df.get("position", "").map(bucket_position)
    df["age"] = pd.to_numeric(df.get("age"), errors="coerce")
    latest_season = df.groupby("team_slug")["season"].transform("max")
    latest = df[df["season"].eq(latest_season)].copy()
    grouped = latest.groupby("team_slug", dropna=False)
    out = grouped.agg(
        hist_last_wc_squad_year=("season", "max"),
        hist_last_wc_squad_players=("player_slug", "nunique"),
        hist_last_wc_squad_avg_age=("age", "mean"),
        hist_last_wc_gk=("position_bucket", lambda s: int((s == "gk").sum())),
        hist_last_wc_def=("position_bucket", lambda s: int((s == "def").sum())),
        hist_last_wc_mid=("position_bucket", lambda s: int((s == "mid").sum())),
        hist_last_wc_att=("position_bucket", lambda s: int((s == "att").sum())),
    ).reset_index()
    return out


def load_external_players(path: Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if "team_slug" not in df.columns:
        team_col = next((c for c in ["team_name", "team", "country", "nation", "national_team"] if c in df.columns), None)
        if not team_col:
            raise SystemExit("External player ratings need team_slug or a team/country/nation column.")
        df["team_slug"] = df[team_col].map(slugify)
    else:
        df["team_slug"] = df["team_slug"].map(slugify)
    if "player_slug" not in df.columns:
        player_col = next((c for c in ["player_name", "player", "name", "full_name"] if c in df.columns), None)
        if not player_col:
            raise SystemExit("External player ratings need player_slug or a player/name/full_name column.")
        df["player_slug"] = df[player_col].map(slugify)
    else:
        df["player_slug"] = df["player_slug"].map(slugify)
    if "position_bucket" not in df.columns:
        pos_col = next((c for c in ["position", "pos", "role"] if c in df.columns), None)
        df["position_bucket"] = df[pos_col].map(bucket_position) if pos_col else "unknown"
    for col in EXTERNAL_PLAYER_NUMERIC_FIELDS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def aggregate_external_player_team_strength(external: pd.DataFrame) -> pd.DataFrame:
    if external.empty:
        return pd.DataFrame(columns=["team_slug"])
    df = external.copy()
    rating_col = next((c for c in ["player_rating", "domestic_player_rating"] if c in df.columns), None)
    value_col = next((c for c in ["market_value_eur", "squad_market_value_eur", "value_eur"] if c in df.columns), None)
    if rating_col:
        df["_rating"] = pd.to_numeric(df[rating_col], errors="coerce")
    else:
        df["_rating"] = math.nan
    if value_col:
        df["_value"] = pd.to_numeric(df[value_col], errors="coerce")
    else:
        df["_value"] = math.nan
    df["_att_rating"] = df["_rating"].where(df["position_bucket"].eq("att"))
    df["_def_rating"] = df["_rating"].where(df["position_bucket"].isin(["gk", "def"]))
    grouped = df.groupby("team_slug", dropna=False)
    out = grouped.agg(
        external_players=("player_slug", "nunique"),
        external_avg_player_rating=("_rating", "mean"),
        external_top11_avg_rating=("_rating", lambda s: pd.to_numeric(s, errors="coerce").dropna().sort_values(ascending=False).head(11).mean()),
        external_top5_avg_rating=("_rating", lambda s: pd.to_numeric(s, errors="coerce").dropna().sort_values(ascending=False).head(5).mean()),
        external_attack_avg_rating=("_att_rating", "mean"),
        external_defence_avg_rating=("_def_rating", "mean"),
        external_squad_market_value_eur=("_value", "sum"),
    ).reset_index()
    out["external_player_prior_joined_flag"] = 1
    return out


def build_team_scaffold(
    launch: pd.DataFrame,
    api_players: pd.DataFrame,
    hist_players: pd.DataFrame,
    macro_teams: pd.DataFrame,
    external_players: pd.DataFrame,
) -> pd.DataFrame:
    teams = []
    for side in ["home", "away"]:
        part = launch[
            [
                f"{side}_team_slug",
                f"{side}_team_name_latest",
                f"{side}_last_wc_squad_avg_age",
                f"{side}_last_wc_squad_players",
            ]
        ].copy()
        part.columns = ["team_slug", "team_name", "launch_last_wc_squad_avg_age", "launch_last_wc_squad_players"]
        teams.append(part)
    out = pd.concat(teams, ignore_index=True).drop_duplicates(subset=["team_slug"]).reset_index(drop=True)
    out["team_slug"] = out["team_slug"].map(slugify)

    if not api_players.empty:
        api = api_players.groupby("team_slug", dropna=False).agg(
            api_2026_roster_players=("player_slug", "nunique"),
            api_2026_roster_avg_age=("age", "mean"),
            api_2026_gk=("position_bucket", lambda s: int((s == "gk").sum())),
            api_2026_def=("position_bucket", lambda s: int((s == "def").sum())),
            api_2026_mid=("position_bucket", lambda s: int((s == "mid").sum())),
            api_2026_att=("position_bucket", lambda s: int((s == "att").sum())),
            api_2026_captains=("captain_flag", "sum"),
        ).reset_index()
    else:
        api = pd.DataFrame(columns=["team_slug"])

    hist = build_historical_team_priors(hist_players)
    external = aggregate_external_player_team_strength(external_players)
    macro_cols = [
        "team_slug",
        "macro_prior_score",
        "macro_prior_percentile",
        "macro_prior_band",
        "has_world_cup_prior",
        "has_2022_prior",
    ]
    macro = macro_teams[[c for c in macro_cols if c in macro_teams.columns]].copy() if not macro_teams.empty else pd.DataFrame(columns=["team_slug"])
    if "team_slug" in macro.columns:
        macro["team_slug"] = macro["team_slug"].map(slugify)

    for frame in [api, hist, macro, external]:
        if not frame.empty:
            out = out.merge(frame, on="team_slug", how="left")

    def numeric_col(name: str, default: float = 0.0) -> pd.Series:
        if name not in out.columns:
            return pd.Series(default, index=out.index)
        return pd.to_numeric(out[name], errors="coerce").fillna(default)

    out["api_2026_roster_joined_flag"] = (numeric_col("api_2026_roster_players") > 0).astype(int)
    out["historical_player_prior_joined_flag"] = (numeric_col("hist_last_wc_squad_players") > 0).astype(int)
    if "external_player_prior_joined_flag" not in out.columns:
        out["external_player_prior_joined_flag"] = 0

    rating = numeric_col("external_top11_avg_rating", math.nan)
    macro_score = numeric_col("macro_prior_score")
    hist_players_count = numeric_col("hist_last_wc_squad_players")
    api_players_count = numeric_col("api_2026_roster_players")
    out["player_intel_depth_score"] = (
        0.45 * out["external_player_prior_joined_flag"].fillna(0)
        + 0.30 * out["api_2026_roster_joined_flag"].fillna(0)
        + 0.15 * out["historical_player_prior_joined_flag"].fillna(0)
        + 0.10 * (hist_players_count.clip(0, 26) / 26)
    )
    out["squad_quality_proxy"] = macro_score + 0.08 * rating.fillna(rating.mean() if rating.notna().any() else 0)
    out["squad_continuity_risk"] = (
        (out["api_2026_roster_joined_flag"].eq(0))
        & (out["historical_player_prior_joined_flag"].eq(0))
    ).astype(int)
    out["api_2026_roster_partial_flag"] = (
        out["api_2026_roster_joined_flag"].eq(1)
        & (api_players_count < 20)
    ).astype(int)
    out["player_intel_status"] = out.apply(
        lambda r: "EXTERNAL_PLAYER_PRIOR"
        if r.get("external_player_prior_joined_flag", 0) == 1
        else "API_2026_PARTIAL_ROSTER"
        if r.get("api_2026_roster_partial_flag", 0) == 1
        else "API_2026_ROSTER"
        if r.get("api_2026_roster_joined_flag", 0) == 1
        else "HISTORICAL_WORLD_CUP_SQUAD_PRIOR"
        if r.get("historical_player_prior_joined_flag", 0) == 1
        else "MISSING_PLAYER_PRIOR",
        axis=1,
    )
    return out.sort_values(["player_intel_status", "team_slug"]).reset_index(drop=True)


def attach_fixture_side(fixtures: pd.DataFrame, teams: pd.DataFrame, side: str) -> pd.DataFrame:
    out = fixtures.copy()
    key = f"{side}_team_slug"
    team = teams.copy()
    team = team.rename(columns={c: f"{side}_{c}" for c in team.columns if c != "team_slug"})
    team = team.rename(columns={"team_slug": key})
    out[key] = out[key].map(slugify)
    return out.merge(team, on=key, how="left")


def build_fixture_matrix(launch: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "season",
        "api_fixture_id",
        "api_date",
        "api_round",
        "api_home_team_name",
        "api_away_team_name",
        "home_team_slug",
        "away_team_slug",
        "coverage_bucket",
    ]
    out = launch[[c for c in base_cols if c in launch.columns]].copy()
    out = attach_fixture_side(out, teams, "home")
    out = attach_fixture_side(out, teams, "away")
    out["diff_squad_quality_proxy"] = pd.to_numeric(out.get("home_squad_quality_proxy"), errors="coerce").fillna(0) - pd.to_numeric(
        out.get("away_squad_quality_proxy"), errors="coerce"
    ).fillna(0)
    out["diff_player_intel_depth_score"] = pd.to_numeric(out.get("home_player_intel_depth_score"), errors="coerce").fillna(0) - pd.to_numeric(
        out.get("away_player_intel_depth_score"), errors="coerce"
    ).fillna(0)
    out["fixture_player_intel_coverage"] = out.apply(
        lambda r: "BOTH_EXTERNAL"
        if r.get("home_external_player_prior_joined_flag", 0) == 1 and r.get("away_external_player_prior_joined_flag", 0) == 1
        else "BOTH_API_ROSTER"
        if r.get("home_api_2026_roster_joined_flag", 0) == 1 and r.get("away_api_2026_roster_joined_flag", 0) == 1
        else "MIXED_OR_HISTORICAL"
        if r.get("home_player_intel_status") != "MISSING_PLAYER_PRIOR" or r.get("away_player_intel_status") != "MISSING_PLAYER_PRIOR"
        else "NO_PLAYER_PRIOR",
        axis=1,
    )
    out["player_intel_lineup_uncertainty_proxy"] = (
        (out.get("home_api_2026_roster_joined_flag", 0).fillna(0) == 0)
        | (out.get("away_api_2026_roster_joined_flag", 0).fillna(0) == 0)
    ).astype(int)
    return out


def write_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "team_slug",
        "team_name",
        "player_slug",
        "player_name",
        "position",
        "position_bucket",
        *EXTERNAL_PLAYER_NUMERIC_FIELDS,
        "source_name",
        "source_url",
        "source_snapshot_date",
        "notes",
    ]
    pd.DataFrame(columns=cols).to_csv(path, index=False)


def write_summary(
    outdir: Path,
    api_players: pd.DataFrame,
    teams: pd.DataFrame,
    fixtures: pd.DataFrame,
    external_path: Path | None,
) -> None:
    coverage = teams["player_intel_status"].value_counts(dropna=False).rename_axis("status").reset_index(name="teams")
    coverage.to_csv(outdir / "world_cup_2026_player_intel_team_coverage.csv", index=False)
    fixture_cov = fixtures["fixture_player_intel_coverage"].value_counts(dropna=False).rename_axis("status").reset_index(name="fixtures")
    fixture_cov.to_csv(outdir / "world_cup_2026_player_intel_fixture_coverage.csv", index=False)
    cov_lines = ["| status | teams |", "|---|---:|"]
    cov_lines.extend(f"| {r.status} | {int(r.teams)} |" for r in coverage.itertuples())
    fixture_lines = ["| status | fixtures |", "|---|---:|"]
    fixture_lines.extend(f"| {r.status} | {int(r.fixtures)} |" for r in fixture_cov.itertuples())
    lines = [
        "# World Cup 2026 Player Intelligence Scaffold",
        "",
        "Research-only sidecar for squad/player priors.",
        "",
        "## Outputs",
        "",
        f"- `{outdir / 'world_cup_2026_api_player_roster.csv'}`",
        f"- `{outdir / 'world_cup_2026_team_player_intelligence_scaffold.csv'}`",
        f"- `{outdir / 'world_cup_2026_fixture_player_intelligence_matrix.csv'}`",
        f"- `{outdir / 'world_cup_external_player_rating_template.csv'}`",
        "",
        "## Inputs",
        "",
        f"- API player rows parsed: {len(api_players)}",
        f"- API player rows matching scheduled 2026 teams: {int(api_players.get('team_slug', pd.Series(dtype=str)).isin(set(teams['team_slug'])).sum()) if not api_players.empty else 0}",
        f"- API player rows outside current 2026 schedule: {int((~api_players.get('team_slug', pd.Series(dtype=str)).isin(set(teams['team_slug']))).sum()) if not api_players.empty else 0}",
        f"- External player ratings: `{external_path}`" if external_path else "- External player ratings: not supplied",
        "",
        "## Team Coverage",
        "",
        *cov_lines,
        "",
        "## Fixture Coverage",
        "",
        *fixture_lines,
        "",
        "## Notes",
        "",
        "- API-Football roster pages may be partial before final squads are announced.",
        "- External domestic player ratings should be joined via the template and must include source snapshot date.",
        "- This sidecar is designed for CatBoost/XGB/goal-mass research inputs, not live deploy routing.",
    ]
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-scaffold", type=Path, default=DEFAULT_LAUNCH)
    parser.add_argument("--historical-players", type=Path, default=DEFAULT_HISTORICAL_PLAYERS)
    parser.add_argument("--macro-teams", type=Path, default=DEFAULT_MACRO_TEAMS)
    parser.add_argument("--api-players-raw", type=Path, default=DEFAULT_API_PLAYERS_RAW)
    parser.add_argument("--external-player-ratings", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    launch = pd.read_csv(args.launch_scaffold, low_memory=False)
    hist_players = pd.read_csv(args.historical_players, low_memory=False) if args.historical_players.exists() else pd.DataFrame()
    macro_teams = pd.read_csv(args.macro_teams, low_memory=False) if args.macro_teams.exists() else pd.DataFrame()
    api_players = parse_api_players(args.api_players_raw)
    external_players = load_external_players(args.external_player_ratings)
    teams = build_team_scaffold(launch, api_players, hist_players, macro_teams, external_players)
    fixtures = build_fixture_matrix(launch, teams)
    scheduled_slugs = set(teams["team_slug"].dropna().astype(str))
    if not api_players.empty:
        api_players[api_players["team_slug"].isin(scheduled_slugs)].to_csv(
            args.outdir / "world_cup_2026_api_player_roster_scheduled_teams_only.csv",
            index=False,
        )
        api_players[~api_players["team_slug"].isin(scheduled_slugs)].to_csv(
            args.outdir / "world_cup_2026_api_player_roster_outside_current_schedule.csv",
            index=False,
        )

    api_players.to_csv(args.outdir / "world_cup_2026_api_player_roster.csv", index=False)
    teams.to_csv(args.outdir / "world_cup_2026_team_player_intelligence_scaffold.csv", index=False)
    fixtures.to_csv(args.outdir / "world_cup_2026_fixture_player_intelligence_matrix.csv", index=False)
    write_template(args.outdir / "world_cup_external_player_rating_template.csv")
    write_summary(args.outdir, api_players, teams, fixtures, args.external_player_ratings)

    print(f"[ok] api_players={len(api_players)} teams={len(teams)} fixtures={len(fixtures)}")
    print(f"[ok] wrote {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
