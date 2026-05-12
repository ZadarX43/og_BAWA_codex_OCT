#!/usr/bin/env python3
"""Prune published player intelligence to the active-site latest-season footprint."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any


DATA_ROOT = Path("frontend/public/data")
PLAYER_ROOT = DATA_ROOT / "player_intelligence"


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)


def season_sort_key(season: Any) -> tuple[int, str]:
    text = str(season or "").strip()
    years = [int(part) for part in text.replace("-", "/").split("/") if part.isdigit()]
    return (max(years) if years else 0, text)


def active_competition_seasons() -> dict[str, str]:
    lineup_index = read_json(DATA_ROOT / "fixture_lineup_intelligence" / "index.json", [])
    active_competitions = {
        str(row.get("competition_key") or "").strip()
        for row in lineup_index
        if isinstance(row, dict) and row.get("competition_key")
    }
    team_index = read_json(DATA_ROOT / "team_intelligence" / "team_ratings_index.json", [])
    seasons: dict[str, str] = {}
    for row in team_index if isinstance(team_index, list) else []:
        competition_key = str(row.get("competition_key") or "").strip()
        season = str(row.get("season") or "").strip()
        if not competition_key or not season or competition_key not in active_competitions:
            continue
        if competition_key not in seasons or season_sort_key(season) > season_sort_key(seasons[competition_key]):
            seasons[competition_key] = season
    return seasons


def keep_row(row: dict[str, Any], active_seasons: dict[str, str]) -> bool:
    competition_key = str(row.get("competition_key") or "").strip()
    season = str(row.get("season") or "").strip()
    return bool(competition_key and season and active_seasons.get(competition_key) == season)


def prune_json_list(path: Path, active_seasons: dict[str, str]) -> int:
    rows = read_json(path, [])
    kept = [row for row in rows if isinstance(row, dict) and keep_row(row, active_seasons)]
    write_json(path, kept)
    return len(kept)


def rebuild_player_indexes(club_squads_path: Path) -> tuple[int, int]:
    squads = read_json(club_squads_path, [])
    players: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for squad in squads if isinstance(squads, list) else []:
        for player in squad.get("players") or []:
            players.append(player)
            row = {
                "name": player.get("name"),
                "club": player.get("club"),
                "competition": player.get("competition"),
                "competition_key": player.get("competition_key"),
                "season": player.get("season"),
                "position_group": player.get("position_group"),
                "league_overall_rank": (player.get("ranks") or {}).get("league_overall_rank"),
                "position_rank": (player.get("ranks") or {}).get("position_rank"),
                "club_rank": (player.get("ranks") or {}).get("club_rank"),
            }
            row.update(player.get("ratings") or {})
            csv_rows.append(row)
    write_json(PLAYER_ROOT / "player_ratings.json", sorted(players, key=lambda item: (
        item.get("competition_key") or "",
        item.get("season") or "",
        item.get("club") or "",
        item.get("name") or "",
    )))
    csv_path = PLAYER_ROOT / "player_ratings.csv"
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
    return (len(players), len(csv_rows))


def prune_club_payload_dirs(active_seasons: dict[str, str]) -> tuple[int, int]:
    clubs_root = PLAYER_ROOT / "clubs"
    if not clubs_root.exists():
        return (0, 0)

    removed_dirs = 0
    kept_files = 0
    for competition_dir in list(clubs_root.iterdir()):
        if not competition_dir.is_dir():
            continue
        competition_key = competition_dir.name
        active_season = active_seasons.get(competition_key)
        if not active_season:
            shutil.rmtree(competition_dir)
            removed_dirs += 1
            continue

        allowed_parts = active_season.split("/")
        for child in list(competition_dir.iterdir()):
            if child.is_file():
                if child.suffix == ".json" and len(allowed_parts) == 1:
                    kept_files += 1
                else:
                    child.unlink()
                continue
            if not child.is_dir():
                continue
            if child.name != allowed_parts[0]:
                shutil.rmtree(child)
                removed_dirs += 1
                continue
            if len(allowed_parts) == 1:
                kept_files += len(list(child.glob("*.json")))
                continue
            for nested in list(child.iterdir()):
                if nested.is_dir() and nested.name == allowed_parts[1]:
                    kept_files += len(list(nested.glob("*.json")))
                    continue
                if nested.is_dir():
                    shutil.rmtree(nested)
                    removed_dirs += 1
                else:
                    nested.unlink()
    return (removed_dirs, kept_files)


def main() -> None:
    active_seasons = active_competition_seasons()
    club_squads_path = PLAYER_ROOT / "club_squad_ratings.json"
    club_squads = prune_json_list(club_squads_path, active_seasons)
    player_json_rows, player_csv_rows = rebuild_player_indexes(club_squads_path)
    removed_dirs, kept_club_files = prune_club_payload_dirs(active_seasons)
    summary = {
        "active_competitions": len(active_seasons),
        "club_squad_ratings": club_squads,
        "player_ratings": player_json_rows,
        "player_ratings_csv": player_csv_rows,
        "removed_club_dirs": removed_dirs,
        "kept_club_files": kept_club_files,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
