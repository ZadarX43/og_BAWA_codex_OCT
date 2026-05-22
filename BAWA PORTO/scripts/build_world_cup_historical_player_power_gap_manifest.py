#!/usr/bin/env python3
"""Build a gap manifest for timestamp-safe historical World Cup player power.

This report answers: what player-power evidence exists for 2018/2022, what is
usable as a pre-tournament or pre-match proxy, and what still needs sourcing
before full-stack player intelligence can be treated as a serious historical
accuracy result.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


DEFAULT_DROP = Path("/Users/hughwade/Desktop/FOOTYSTATS_DROP")
DEFAULT_OUTDIR = Path("data_sources/footystats_world_cup/historical_player_power_gap_manifest")
DEFAULT_API_RAW = Path("data_sources/api_football/raw")

TARGET_SEASONS = [2018, 2022]
QUALIFIER_CONFEDS = ["africa", "asia", "concacaf", "europe", "south-america", "ofc", "intercontinental-playoffs"]


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            out[col] = out[col].astype("string").fillna("")
    lines = [
        "| " + " | ".join(out.columns) + " |",
        "| " + " | ".join(["---"] * len(out.columns)) + " |",
    ]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in out.columns) + " |")
    return "\n".join(lines)


def file_years(name: str) -> set[int]:
    return {int(y) for y in re.findall(r"(?:19|20)\d{2}", name)}


def scan_files(root: Path) -> pd.DataFrame:
    rows = []
    if not root.exists():
        return pd.DataFrame(columns=["path", "name", "kind", "years"])
    for path in root.rglob("*.csv"):
        name = path.name.lower()
        kind = "other"
        if "players" in name:
            kind = "players"
        elif "teams" in name:
            kind = "teams"
        elif "matches" in name:
            kind = "matches"
        elif "squads" in name:
            kind = "squads"
        rows.append({"path": str(path), "name": name, "kind": kind, "years": sorted(file_years(name))})
    return pd.DataFrame(rows)


def contains_year(row_years: object, season: int) -> bool:
    years = row_years if isinstance(row_years, list) else []
    if season == 2018:
        return 2018 in years or (2016 in years and 2018 in years)
    return season in years


def matching_paths(files: pd.DataFrame, *, season: int, includes: list[str], kind: str | None = None) -> list[str]:
    if files.empty:
        return []
    mask = files["name"].astype(str).map(lambda n: all(token in n for token in includes))
    mask &= files["years"].map(lambda y: contains_year(y, season))
    if kind:
        mask &= files["kind"].eq(kind)
    return sorted(files.loc[mask, "path"].astype(str).tolist())


def api_exists(api_raw: Path, pattern: str) -> list[str]:
    if not api_raw.exists():
        return []
    return sorted(str(p) for p in api_raw.glob(pattern))


def jsonl_response_rows(paths: list[str]) -> int:
    rows = 0
    for item in paths:
        path = Path(item)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                response = payload.get("response")
                if isinstance(response, list):
                    rows += len(response)
    return rows


def status(found: list[str], good_status: str = "FOUND") -> str:
    return good_status if found else "MISSING"


def build_manifest(drop: Path, api_raw: Path, outdir: Path) -> None:
    files = scan_files(drop)
    rows: list[dict[str, object]] = []
    available_rows: list[dict[str, object]] = []

    for _, row in files.iterrows():
        name = str(row["name"])
        if any(token in name for token in ["world-cup", "wc-qualification", "friendlies", "copa-america", "gold-cup", "nations-league"]):
            available_rows.append(row.to_dict())

    for season in TARGET_SEASONS:
        wc_players = matching_paths(files, season=season, includes=["fifa-world-cup", "players"], kind="players")
        wc_teams = matching_paths(files, season=season, includes=["fifa-world-cup", "teams"], kind="teams")
        rows.append(
            {
                "season": season,
                "layer": "world_cup_player_stats",
                "current_status": status(wc_players, "FOUND_POST_TOURNAMENT_PLAYER_STATS"),
                "found_files": " | ".join(wc_players),
                "usable_for_validation": "NO_DIRECTLY_POST_TOURNAMENT_STATS",
                "needed_source_or_action": "Keep as context only unless columns can be lagged per prior fixture. Do not use final tournament aggregates as pre-match power.",
            }
        )
        rows.append(
            {
                "season": season,
                "layer": "world_cup_team_stats",
                "current_status": status(wc_teams, "FOUND_TEAM_TOURNAMENT_CONTEXT"),
                "found_files": " | ".join(wc_teams),
                "usable_for_validation": "LIMITED_CONTEXT_ONLY",
                "needed_source_or_action": "Use only historical prior/metadata surfaces, not same-tournament final aggregates for pre-match claims.",
            }
        )

        for confed in QUALIFIER_CONFEDS:
            q_players = matching_paths(files, season=season, includes=[f"wc-qualification-{confed}", "players"], kind="players")
            q_matches = matching_paths(files, season=season, includes=[f"wc-qualification-{confed}", "matches"], kind="matches")
            rows.append(
                {
                    "season": season,
                    "layer": f"qualifier_player_power_{confed}",
                    "current_status": status(q_players, "FOUND_QUALIFIER_PLAYER_PROXY"),
                    "found_files": " | ".join(q_players),
                    "usable_for_validation": "YES_IF_FILTERED_BEFORE_WORLD_CUP_KICKOFF" if q_players else "NO",
                    "needed_source_or_action": (
                        "Join to qualified squads and lag by source competition end date."
                        if q_players
                        else f"Source FootyStats/API/player files for WC Qualification {confed} feeding World Cup {season}, or document exclusion."
                    ),
                }
            )
            rows.append(
                {
                    "season": season,
                    "layer": f"qualifier_match_context_{confed}",
                    "current_status": status(q_matches, "FOUND_QUALIFIER_MATCH_PROXY"),
                    "found_files": " | ".join(q_matches),
                    "usable_for_validation": "YES_IF_FILTERED_BEFORE_WORLD_CUP_KICKOFF" if q_matches else "NO",
                    "needed_source_or_action": (
                        "Already usable for team/recent-form proxy if filtered by fixture date."
                        if q_matches
                        else f"Source qualifier match files for {confed} or document confed gap."
                    ),
                }
            )

        api_players = api_exists(api_raw, f"players__league_1__season_{season}__players.jsonl")
        api_fixture_players = api_exists(api_raw, f"fixtures__league_1__season_{season}__fixtures_players.jsonl")
        api_injuries = api_exists(api_raw, f"fixtures__league_1__season_{season}__injuries.jsonl")
        api_injury_rows = jsonl_response_rows(api_injuries)
        api_injury_status = (
            "FOUND_API_INJURY_PAYLOADS_ZERO_ROWS"
            if api_injuries and api_injury_rows == 0
            else status(api_injuries, "FOUND_API_INJURIES")
        )
        api_injury_usable = (
            "NO_HISTORICAL_INJURY_ROWS"
            if api_injuries and api_injury_rows == 0
            else "YES_IF_INJURY_FIRST_SEEN_BEFORE_DEPLOY_OR_KICKOFF"
        )
        api_injury_action = (
            "API-Football returned historical injury payloads but zero injury rows; source external timestamped injury/suspension data for validation."
            if api_injuries and api_injury_rows == 0
            else f"Fetch/cache API-Football injuries for league=1 season={season}; persist first-seen timestamp if possible."
        )
        rows.extend(
            [
                {
                    "season": season,
                    "layer": "api_world_cup_roster_players",
                    "current_status": status(api_players, "FOUND_API_WORLD_CUP_PLAYERS"),
                    "found_files": " | ".join(api_players),
                    "usable_for_validation": "YES_IF_ENDPOINT_IS_PRE_TOURNAMENT_ROSTER_OR_TIMESTAMPED",
                    "needed_source_or_action": f"Fetch/cache API-Football `/players?league=1&season={season}` and record fetch/source timestamp.",
                },
                {
                    "season": season,
                    "layer": "api_fixture_player_ratings",
                    "current_status": status(api_fixture_players, "FOUND_API_FIXTURE_PLAYER_STATS"),
                    "found_files": " | ".join(api_fixture_players),
                    "usable_for_validation": "ONLY_AS_PRIOR_MATCH_HISTORY_AFTER_LAGGING",
                    "needed_source_or_action": f"Fetch API-Football `/fixtures/players` for World Cup {season}; use only prior completed matches for later-round predictions.",
                },
                {
                    "season": season,
                    "layer": "api_injuries_suspensions",
                    "current_status": api_injury_status,
                    "found_files": " | ".join(api_injuries),
                    "usable_for_validation": api_injury_usable,
                    "needed_source_or_action": api_injury_action,
                },
            ]
        )

        squad_archive = matching_paths(files, season=season, includes=["squads"], kind="squads")
        if season == 2018:
            squad_archive += [str(drop / "archive" / "squads.csv")] if (drop / "archive" / "squads.csv").exists() else []
        rows.append(
            {
                "season": season,
                "layer": "official_squad_snapshot",
                "current_status": status(sorted(set(squad_archive)), "FOUND_STATIC_SQUAD_CONTEXT"),
                "found_files": " | ".join(sorted(set(squad_archive))),
                "usable_for_validation": "YES_AS_TOURNAMENT_SQUAD_CONTEXT_IF_NO_POST_MATCH_STATS_INCLUDED" if squad_archive else "NO",
                "needed_source_or_action": "Source official squad list with announcement date and player IDs for all tournament teams.",
            }
        )

        rows.extend(
            [
                {
                    "season": season,
                    "layer": "external_domestic_player_ratings",
                    "current_status": "MISSING_TIME_SAFE_PLAYER_RATING_SNAPSHOT",
                    "found_files": "",
                    "usable_for_validation": "NO",
                    "needed_source_or_action": (
                        f"Find a pre-tournament {season} player rating/value dataset keyed by player and national team "
                        "or club, with rating date before the World Cup."
                    ),
                },
                {
                    "season": season,
                    "layer": "team_elo_or_fifa_ranking_snapshot",
                    "current_status": "MISSING_EXPLICIT_RATING_SNAPSHOT",
                    "found_files": "",
                    "usable_for_validation": "NO",
                    "needed_source_or_action": f"Source FIFA ranking/Elo CSV with country, rating/rank, and date before each {season} fixture.",
                },
                {
                    "season": season,
                    "layer": "player_identity_alias_map",
                    "current_status": "MISSING_VALIDATED_ALIAS_MAP",
                    "found_files": "",
                    "usable_for_validation": "NO",
                    "needed_source_or_action": "Build player/team alias map across FootyStats, API-Football, Fjelstul, and any external ratings source.",
                },
            ]
        )

    manifest = pd.DataFrame(rows)
    available = pd.DataFrame(available_rows)
    if not available.empty:
        available["years"] = available["years"].map(lambda y: ",".join(map(str, y)) if isinstance(y, list) else "")

    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / "world_cup_historical_player_power_gap_manifest.csv"
    available_path = outdir / "world_cup_historical_player_power_available_files.csv"
    manifest.to_csv(manifest_path, index=False)
    available.to_csv(available_path, index=False)

    gap_summary = (
        manifest.groupby(["season", "usable_for_validation", "current_status"], dropna=False)
        .agg(layers=("layer", "count"))
        .reset_index()
        .sort_values(["season", "usable_for_validation", "layers"], ascending=[True, True, False])
    )
    gap_summary_path = outdir / "world_cup_historical_player_power_gap_summary.csv"
    gap_summary.to_csv(gap_summary_path, index=False)

    md = [
        "# World Cup Historical Player-Power Gap Manifest",
        "",
        "Research-only source manifest for making 2018/2022 player-power validation timestamp-safe.",
        "",
        "## Gap Summary",
        "",
        markdown_table(gap_summary),
        "",
        "## Missing Critical Layers",
        "",
        markdown_table(
            manifest[
                manifest["current_status"].astype(str).str.contains("MISSING", na=False)
            ][["season", "layer", "current_status", "needed_source_or_action"]].head(40)
        ),
        "",
        "## Outputs",
        "",
        f"- Gap manifest: `{manifest_path}`",
        f"- Available source inventory: `{available_path}`",
        f"- Gap summary: `{gap_summary_path}`",
        "",
        "## Practical Read",
        "",
        "Qualifier player files can become a useful pre-tournament proxy where present. "
        "The critical missing piece is still a dated, player-level rating or value snapshot for 2018 and 2022, plus API/FIFA/Elo identity mapping.",
        "",
    ]
    (outdir / "SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[ok] manifest_rows={len(manifest)} source_files={len(available)}")
    print(f"[ok] wrote {outdir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--footystats-drop", type=Path, default=DEFAULT_DROP)
    parser.add_argument("--api-raw", type=Path, default=DEFAULT_API_RAW)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    build_manifest(args.footystats_drop, args.api_raw, args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
