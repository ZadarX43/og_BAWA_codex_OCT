from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from player_rating_engine import build_player_scores, build_ranks, prepare_player_frame
from ratings_engine_utils import clean_columns, ensure_dir, score_to_int, slugify, write_json
from team_rating_engine import build_team_scores, prepare_team_frame


CONFIRMED_LINEUP_UPDATE_MODEL = {
    "trigger": "provider_lineups_available",
    "target_window_minutes_before_kickoff": 50,
    "frontend_contract": (
        "Keep the fixture_lineup_intelligence payload shape stable. Replace predicted "
        "last-fixture starters/bench with confirmed provider starters/bench, set "
        "lineup_status to confirmed_lineup, and republish both D1 and static fallback."
    ),
    "fallback_before_confirmation": "predicted_from_last_fixture",
}


LINEUP_OUTPUT_FIELDS = [
    "attack_unit",
    "midfield_control",
    "defensive_unit",
    "wide_threat",
    "central_threat",
    "discipline_risk",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build publish-safe fixture lineup intelligence.")
    parser.add_argument("--config", help="JSON config listing team/player source families.")
    parser.add_argument("--team-input", help="Single team season CSV.")
    parser.add_argument("--player-input", help="Single player season CSV.")
    parser.add_argument("--lineups-csv", help="Normalized lineup CSV for the same competition-season.")
    parser.add_argument("--features-csv", help="Fixture lineup feature CSV for the same competition-season.")
    parser.add_argument("--competition-name", help="Public competition name override.")
    parser.add_argument("--competition-key", help="Stable competition key override.")
    parser.add_argument("--season", help="Season label override.")
    parser.add_argument(
        "--normalized-lineups-root",
        default="data_sources/api_football/normalized",
        help="Root for normalized lineup CSVs when using --config.",
    )
    parser.add_argument(
        "--features-root",
        default="data_sources/api_football/features",
        help="Root for fixture lineup feature CSVs when using --config.",
    )
    parser.add_argument(
        "--output-root",
        default="frontend/public/data",
        help="Publish output root. Default: frontend/public/data",
    )
    parser.add_argument("--fixture-feed", default=None, help="Optional fixture feed JSON for current fixture-key aliases.")
    return parser.parse_args()


def season_start(value: object) -> str:
    text = str(value or "").strip()
    match = re.match(r"(\d{4})", text)
    return match.group(1) if match else text


def safe_int_value(value: object) -> int | None:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        text = str(value).strip()
        if not re.fullmatch(r"-?\d+", text):
            return None
        return int(text)
    except (TypeError, ValueError):
        return None


def normalize_text(value: object) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_competition_key(value: object) -> str:
    text = normalize_text(value)
    aliases = {
        "swiss super league": "switzerland super league",
    }
    return aliases.get(text, text)


def normalize_team_key(value: object) -> str:
    text = normalize_text(value)
    text = re.sub(r"\b(fc|cf|sc|afc|club)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def team_match_key(value: object) -> str:
    text = normalize_team_key(value)
    replacements = {
        "st": "saint",
        "sint": "saint",
        "hove": "",
        "albion": "",
        "hotspur": "",
        "united": "",
        "wanderers": "",
        "kv": "",
        "kvc": "",
        "kaa": "",
        "krc": "",
        "rsc": "",
        "royal": "",
        "eh": "",
    }
    tokens = [replacements.get(token, token) for token in text.split()]
    text = " ".join(token for token in tokens if token)
    aliases = {
        "wolverhampton": "wolves",
        "atletico mineiro": "atletico mg",
        "atletico pr": "atletico paranaense",
        "bragantino": "rb bragantino",
        "sporting braga": "braga",
        "vitoria guimaraes": "guimaraes",
        "estrela amadora": "estrela",
        "gd estoril praia": "estoril",
        "cd nacional": "nacional",
    }
    return aliases.get(text, text)


def team_match_score(left: object, right: object) -> int:
    left_key = team_match_key(left)
    right_key = team_match_key(right)
    if not left_key or not right_key:
        return 0
    if left_key == right_key:
        return 100
    if len(left_key) >= 4 and len(right_key) >= 4 and (left_key in right_key or right_key in left_key):
        return 86
    left_tokens = set(left_key.split())
    right_tokens = set(right_key.split())
    if not left_tokens or not right_tokens:
        return 0
    overlap = len(left_tokens & right_tokens)
    return int(round((overlap / max(len(left_tokens), len(right_tokens))) * 80))


def person_name_keys(value: object) -> dict[str, str]:
    text = normalize_text(value)
    parts = [part for part in text.split() if part]
    if not parts:
        return {"full": "", "surname": "", "initial_surname": ""}
    surname = parts[-1]
    initial = parts[0][0] if parts[0] else ""
    return {
        "full": " ".join(parts),
        "surname": surname,
        "initial_surname": f"{initial} {surname}".strip(),
    }


def inferred_position_group(value: object) -> str:
    key = str(value or "").strip().upper()
    if key == "G":
        return "goalkeeper"
    if key == "D":
        return "centre_back"
    if key == "M":
        return "central_midfielder"
    if key == "F":
        return "forward"
    return "utility"


def compatibility_bonus(candidate_group: str, expected_group: str) -> int:
    if candidate_group == expected_group:
        return 3
    if expected_group == "centre_back" and candidate_group in {"full_back", "defensive_midfielder"}:
        return 2
    if expected_group == "central_midfielder" and candidate_group in {"defensive_midfielder", "attacking_midfielder"}:
        return 2
    if expected_group == "forward" and candidate_group in {"winger", "attacking_midfielder"}:
        return 2
    return 0


def discover_competition_file(root: Path, prefix: str, competition_name: str, year: str) -> Path | None:
    target = normalize_text(competition_name)
    suffix = f"__{year}.csv"
    for path in root.glob(f"{prefix}__*{suffix}"):
        middle = path.stem[len(prefix) + 2 : -len(suffix[:-4])]
        if normalize_text(middle.replace("_", " ")) == target:
            return path
    return None


def parsed_competition_file(path: Path, prefix: str) -> tuple[str, int] | None:
    match = re.match(rf"{re.escape(prefix)}__(.+)__(\d{{4}})$", path.stem)
    if not match:
        return None
    return normalize_competition_key(match.group(1).replace("_", " ")), int(match.group(2))


def discover_latest_competition_pair(normalized_root: Path, features_root: Path, competition_name: str) -> tuple[Path, Path, int] | None:
    target = normalize_competition_key(competition_name)
    lineups_by_year: dict[int, Path] = {}
    for path in normalized_root.glob("lineups__*.csv"):
        parsed = parsed_competition_file(path, "lineups")
        if parsed and parsed[0] == target:
            lineups_by_year[parsed[1]] = path

    features_by_year: dict[int, Path] = {}
    for path in features_root.glob("api_lineup_features__*.csv"):
        parsed = parsed_competition_file(path, "api_lineup_features")
        if parsed and parsed[0] == target:
            features_by_year[parsed[1]] = path

    common_years = sorted(set(lineups_by_year) & set(features_by_year), reverse=True)
    if not common_years:
        return None
    year = common_years[0]
    return lineups_by_year[year], features_by_year[year], year


def latest_by_competition(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for entry in entries:
        comp_key = entry.get("competition_key")
        if not comp_key:
            continue
        current = latest.get(comp_key)
        if current is None or season_start(entry.get("season")) > season_start(current.get("season")):
            latest[comp_key] = entry
    return latest


def load_entries(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.config:
        payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
        fixture_feed_rows = load_fixture_feed_rows(Path(args.fixture_feed)) if args.fixture_feed else []
        active_competitions = {normalize_competition_key(row.get("league")) for row in fixture_feed_rows if row.get("league")}
        teams = {
            (entry["competition_key"], str(entry["season"])): entry
            for entry in payload.get("teams", [])
        }
        players = {
            (entry["competition_key"], str(entry["season"])): entry
            for entry in payload.get("players", [])
        }
        normalized_root = Path(args.normalized_lineups_root)
        features_root = Path(args.features_root)

        if active_competitions:
            latest_teams = latest_by_competition(
                [entry for entry in payload.get("teams", []) if normalize_competition_key(entry.get("competition_name")) in active_competitions]
            )
            latest_players = latest_by_competition(
                [entry for entry in payload.get("players", []) if normalize_competition_key(entry.get("competition_name")) in active_competitions]
            )
            entries: list[dict[str, Any]] = []
            for competition_key, team_entry in latest_teams.items():
                player_entry = latest_players.get(competition_key)
                if not player_entry:
                    continue
                competition_name = team_entry.get("competition_name") or player_entry.get("competition_name") or competition_key
                latest_pair = discover_latest_competition_pair(normalized_root, features_root, competition_name)
                if not latest_pair:
                    continue
                lineups_csv, features_csv, source_year = latest_pair
                entries.append(
                    {
                        "competition_name": competition_name,
                        "competition_key": competition_key,
                        "season": str(team_entry.get("season")),
                        "snapshot_source_year": str(source_year),
                        "team_source": team_entry["source"],
                        "player_source": player_entry["source"],
                        "lineups_csv": str(lineups_csv),
                        "features_csv": str(features_csv),
                    }
                )
            return entries

        entries: list[dict[str, Any]] = []
        for key, team_entry in teams.items():
            player_entry = players.get(key)
            if not player_entry:
                continue
            competition_name = team_entry.get("competition_name") or player_entry.get("competition_name") or key[0]
            year = season_start(team_entry.get("season"))
            lineups_csv = discover_competition_file(normalized_root, "lineups", competition_name, year)
            features_csv = discover_competition_file(features_root, "api_lineup_features", competition_name, year)
            if not lineups_csv or not features_csv:
                continue
            entries.append(
                {
                    "competition_name": competition_name,
                    "competition_key": team_entry.get("competition_key"),
                    "season": str(team_entry.get("season")),
                    "team_source": team_entry["source"],
                    "player_source": player_entry["source"],
                    "lineups_csv": str(lineups_csv),
                    "features_csv": str(features_csv),
                }
            )
        return entries

    required = [args.team_input, args.player_input, args.lineups_csv, args.features_csv]
    if all(required):
        return [
            {
                "competition_name": args.competition_name,
                "competition_key": args.competition_key or slugify(args.competition_name),
                "season": str(args.season or "unknown"),
                "team_source": args.team_input,
                "player_source": args.player_input,
                "lineups_csv": args.lineups_csv,
                "features_csv": args.features_csv,
            }
        ]
    raise SystemExit("Provide either --config or the direct single-season inputs.")


def build_team_lookup(frame: pd.DataFrame) -> dict[str, pd.Series]:
    lookup: dict[str, pd.Series] = {}
    for _, row in frame.iterrows():
        keys = {
            normalize_team_key(row.get("team")),
            normalize_team_key(row.get("common_name")),
            normalize_team_key(row.get("team_name")),
        }
        for key in keys:
            if key:
                lookup[key] = row
    return lookup


def build_player_lookup(frame: pd.DataFrame) -> dict[str, dict[str, dict[str, list[dict[str, Any]]]]]:
    lookup: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    for _, row in frame.iterrows():
        club_key = normalize_team_key(row.get("club"))
        bucket = lookup.setdefault(club_key, {"full": {}, "surname": {}, "initial_surname": {}})
        keys = person_name_keys(row.get("name"))
        payload = row.to_dict()
        for name_key, field in [("full", "full"), ("surname", "surname"), ("initial_surname", "initial_surname")]:
            value = keys[field]
            if value:
                bucket[name_key].setdefault(value, []).append(payload)
    return lookup


def resolve_player_profile(
    lineup_player_name: object,
    lineup_position: object,
    club_lookup: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, Any] | None:
    keys = person_name_keys(lineup_player_name)
    expected_group = inferred_position_group(lineup_position)
    candidates: list[dict[str, Any]] = []
    for namespace, key in [("full", keys["full"]), ("initial_surname", keys["initial_surname"]), ("surname", keys["surname"])]:
        if key and key in club_lookup[namespace]:
            candidates = list(club_lookup[namespace][key])
            if candidates:
                break
    if not candidates:
        return None
    scored = sorted(
        candidates,
        key=lambda candidate: (
            compatibility_bonus(str(candidate.get("position_group", "")), expected_group),
            float(candidate.get("og_player_power", 0)),
            float(candidate.get("minutes_played_overall", 0) or 0),
        ),
        reverse=True,
    )
    return scored[0]


def weighted_player_score(player: dict[str, Any], weights: dict[str, float]) -> float:
    total = 0.0
    total_weight = 0.0
    for field, weight in weights.items():
        total += float(player.get(field, 50) or 50) * weight
        total_weight += weight
    return total / max(total_weight, 1.0)


def top_mean(players: list[dict[str, Any]], weights: dict[str, float], take: int, fallback: float = 50.0) -> float:
    if not players:
        return fallback
    values = sorted((weighted_player_score(player, weights) for player in players), reverse=True)
    sample = values[: max(1, min(take, len(values)))]
    return sum(sample) / len(sample)


def score_band_note(score: float, high: str, mid: str, low: str) -> str:
    if score >= 80:
        return high
    if score >= 55:
        return mid
    return low


def lineup_side_payload(
    side: str,
    lineup_rows: pd.DataFrame,
    team_row: pd.Series | None,
    player_lookup: dict[str, dict[str, dict[str, list[dict[str, Any]]]]],
) -> tuple[dict[str, int], list[dict[str, Any]], dict[str, int]]:
    team_name = team_row.get("team") if team_row is not None else ""
    club_bucket = player_lookup.get(normalize_team_key(team_name), {"full": {}, "surname": {}, "initial_surname": {}})

    resolved_players: list[dict[str, Any]] = []
    unresolved = 0
    for _, lineup_row in lineup_rows.iterrows():
        profile = resolve_player_profile(lineup_row.get("player_name"), lineup_row.get("position"), club_bucket)
        if profile is None:
            unresolved += 1
            profile = {
                "name": lineup_row.get("player_name") or "Player",
                "pitch_label": str(lineup_row.get("player_name") or "Player").split()[-1],
                "position_group": inferred_position_group(lineup_row.get("position")),
                "og_player_power": 50,
                "goal_threat": 50,
                "creative_spark": 50,
                "midfield_engine": 50,
                "defensive_lock": 50,
                "pressing_heat": 50,
                "ball_progression": 50,
                "aerial_dominance": 50,
                "goalkeeper_shield": 50,
                "discipline_risk": 50,
                "booking_heat": 50,
                "minutes_played_overall": 0,
                "tags": [],
            }
        resolved_players.append(
            {
                "name": profile.get("name") or lineup_row.get("player_name") or "Player",
                "surname": profile.get("pitch_label") or str(profile.get("name") or lineup_row.get("player_name") or "Player").split()[-1],
                "lineup_position": str(lineup_row.get("position") or "").upper(),
                "position_group": profile.get("position_group") or inferred_position_group(lineup_row.get("position")),
                "power": int(score_to_int(profile.get("og_player_power", 50))),
                "goal_threat": int(score_to_int(profile.get("goal_threat", 50))),
                "creative_spark": int(score_to_int(profile.get("creative_spark", 50))),
                "midfield_engine": int(score_to_int(profile.get("midfield_engine", 50))),
                "defensive_lock": int(score_to_int(profile.get("defensive_lock", 50))),
                "pressing_heat": int(score_to_int(profile.get("pressing_heat", 50))),
                "ball_progression": int(score_to_int(profile.get("ball_progression", 50))),
                "aerial_dominance": int(score_to_int(profile.get("aerial_dominance", 50))),
                "goalkeeper_shield": int(score_to_int(profile.get("goalkeeper_shield", 50))),
                "discipline_risk": int(score_to_int(profile.get("discipline_risk", 50))),
                "booking_heat": int(score_to_int(profile.get("booking_heat", 50))),
                "minutes_played": int(float(profile.get("minutes_played_overall", 0) or 0)),
                "tags": list(profile.get("tags") or [])[:3],
            }
        )

    attack_pool = [
        player
        for player in resolved_players
        if player["lineup_position"] == "F" or player["position_group"] in {"forward", "winger", "attacking_midfielder"}
    ]
    midfield_pool = [
        player
        for player in resolved_players
        if player["lineup_position"] == "M" or player["position_group"] in {"central_midfielder", "defensive_midfielder", "attacking_midfielder"}
    ]
    defence_pool = [
        player
        for player in resolved_players
        if player["lineup_position"] in {"G", "D"} or player["position_group"] in {"goalkeeper", "centre_back", "full_back", "defensive_midfielder"}
    ]
    wide_pool = [
        player
        for player in resolved_players
        if player["position_group"] in {"winger", "full_back"} or (player["lineup_position"] in {"M", "F"} and player["ball_progression"] >= 60)
    ]
    central_pool = [
        player
        for player in resolved_players
        if player["position_group"] in {"forward", "attacking_midfielder", "central_midfielder"} or player["lineup_position"] == "F"
    ]

    team_attack = float(team_row.get("attack_flow_rating", 50) if team_row is not None else 50)
    team_first = float(team_row.get("first_strike_rating", 50) if team_row is not None else 50)
    team_control = float(team_row.get("control_rating", 50) if team_row is not None else 50)
    team_defence = float(team_row.get("defensive_lock_rating", 50) if team_row is not None else 50)
    team_corners = float(team_row.get("corner_pressure_rating", 50) if team_row is not None else 50)
    team_cards = float(team_row.get("card_heat_rating", 50) if team_row is not None else 50)
    team_power = float(team_row.get("og_power_rating", 50) if team_row is not None else 50)

    attack_unit = score_to_int(
        0.68
        * top_mean(
            attack_pool,
            {"power": 0.35, "goal_threat": 0.35, "creative_spark": 0.15, "ball_progression": 0.15},
            4,
            fallback=team_attack,
        )
        + 0.32 * ((team_attack * 0.55) + (team_first * 0.25) + (team_power * 0.20))
    )
    midfield_control = score_to_int(
        0.7
        * top_mean(
            midfield_pool,
            {"midfield_engine": 0.4, "ball_progression": 0.25, "pressing_heat": 0.2, "creative_spark": 0.15},
            4,
            fallback=team_control,
        )
        + 0.3 * ((team_control * 0.5) + (team_power * 0.3) + (team_attack * 0.2))
    )
    defensive_unit = score_to_int(
        0.72
        * top_mean(
            defence_pool,
            {"defensive_lock": 0.45, "aerial_dominance": 0.2, "goalkeeper_shield": 0.2, "pressing_heat": 0.15},
            5,
            fallback=team_defence,
        )
        + 0.28 * ((team_defence * 0.6) + (team_control * 0.25) + ((100 - team_cards) * 0.15))
    )
    wide_threat = score_to_int(
        0.75
        * top_mean(
            wide_pool,
            {"creative_spark": 0.35, "ball_progression": 0.35, "pressing_heat": 0.15, "goal_threat": 0.15},
            3,
            fallback=team_corners,
        )
        + 0.25 * ((team_corners * 0.6) + (team_attack * 0.4))
    )
    central_threat = score_to_int(
        0.75
        * top_mean(
            central_pool,
            {"goal_threat": 0.4, "power": 0.2, "creative_spark": 0.15, "ball_progression": 0.1, "midfield_engine": 0.15},
            3,
            fallback=team_attack,
        )
        + 0.25 * ((team_attack * 0.55) + (team_first * 0.25) + (team_power * 0.2))
    )
    discipline_risk = score_to_int(
        0.7
        * top_mean(
            resolved_players,
            {"discipline_risk": 0.6, "booking_heat": 0.4},
            5,
            fallback=team_cards,
        )
        + 0.3 * team_cards
    )

    units = {
        "attack_unit": int(attack_unit),
        "midfield_control": int(midfield_control),
        "defensive_unit": int(defensive_unit),
        "wide_threat": int(wide_threat),
        "central_threat": int(central_threat),
        "discipline_risk": int(discipline_risk),
    }
    resolution = {
        "starting_xi_count": int(len(lineup_rows)),
        "resolved_profiles": int(len(lineup_rows) - unresolved),
        "fallback_profiles": int(unresolved),
    }
    return units, resolved_players, resolution


def matchup_item(
    zone: str,
    home_value: int,
    away_value: int,
    home_team: str,
    away_team: str,
    home_bias_summary: str,
    away_bias_summary: str,
    home_label: str,
    away_label: str,
) -> dict[str, Any]:
    delta = int(home_value - away_value)
    advantage = home_team if delta >= 0 else away_team
    summary = home_bias_summary if delta >= 0 else away_bias_summary
    return {
        "zone": zone,
        "advantage": advantage,
        "mismatch_score": abs(delta),
        "left_label": home_label,
        "left_value": int(home_value),
        "right_label": away_label,
        "right_value": int(away_value),
        "summary": summary,
    }


def build_key_mismatches(home_units: dict[str, int], away_units: dict[str, int], home_team: str, away_team: str) -> list[dict[str, Any]]:
    candidates = [
        matchup_item(
            f"{home_team} attack vs {away_team} defence",
            home_units["attack_unit"],
            away_units["defensive_unit"],
            home_team,
            away_team,
            f"Major {home_team} attacking advantage",
            f"{away_team} defensive resistance is suppressing the home attack",
            f"{home_team} attack",
            f"{away_team} defence",
        ),
        matchup_item(
            f"{away_team} attack vs {home_team} defence",
            away_units["attack_unit"],
            home_units["defensive_unit"],
            away_team,
            home_team,
            f"Major {away_team} attacking advantage",
            f"{home_team} defensive resistance is suppressing the away attack",
            f"{away_team} attack",
            f"{home_team} defence",
        ),
        matchup_item(
            f"{home_team} wide threat vs {away_team} resistance",
            home_units["wide_threat"],
            away_units["defensive_unit"],
            home_team,
            away_team,
            f"{home_team} wide channel has a live attacking edge",
            f"{away_team} can neutralize the wide-channel pressure",
            f"{home_team} wide threat",
            f"{away_team} resistance",
        ),
        matchup_item(
            f"{away_team} wide threat vs {home_team} resistance",
            away_units["wide_threat"],
            home_units["defensive_unit"],
            away_team,
            home_team,
            f"{away_team} wide channel has a live attacking edge",
            f"{home_team} can neutralize the wide-channel pressure",
            f"{away_team} wide threat",
            f"{home_team} resistance",
        ),
        matchup_item(
            "Midfield control battle",
            home_units["midfield_control"],
            away_units["midfield_control"],
            home_team,
            away_team,
            f"{home_team} should control more of the central game-state",
            f"{away_team} should control more of the central game-state",
            f"{home_team} midfield",
            f"{away_team} midfield",
        ),
    ]
    return sorted(candidates, key=lambda item: item["mismatch_score"], reverse=True)[:4]


def best_player(players: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, Any] | None:
    if not players:
        return None
    return max(players, key=lambda player: sum(float(player.get(field, 50) or 50) for field in fields))


def build_player_matchups(
    home_players: list[dict[str, Any]],
    away_players: list[dict[str, Any]],
    home_team: str,
    away_team: str,
) -> list[dict[str, Any]]:
    home_attackers = [player for player in home_players if player["lineup_position"] == "F" or player["position_group"] in {"forward", "winger", "attacking_midfielder"}]
    away_attackers = [player for player in away_players if player["lineup_position"] == "F" or player["position_group"] in {"forward", "winger", "attacking_midfielder"}]
    home_defenders = [player for player in home_players if player["lineup_position"] in {"D", "G"} or player["position_group"] in {"centre_back", "full_back", "goalkeeper", "defensive_midfielder"}]
    away_defenders = [player for player in away_players if player["lineup_position"] in {"D", "G"} or player["position_group"] in {"centre_back", "full_back", "goalkeeper", "defensive_midfielder"}]
    home_creators = [player for player in home_players if player["creative_spark"] >= 55 or player["ball_progression"] >= 55]
    away_creators = [player for player in away_players if player["creative_spark"] >= 55 or player["ball_progression"] >= 55]

    pairs = []
    home_spear = best_player(home_attackers, ("goal_threat", "power"))
    away_anchor = best_player(away_defenders, ("defensive_lock", "aerial_dominance", "goalkeeper_shield"))
    if home_spear and away_anchor:
      pairs.append(
          {
              "zone": f"{home_team} attack spearhead vs {away_team} defensive anchor",
              "home_player": home_spear["surname"],
              "away_player": away_anchor["surname"],
              "advantage": home_team if home_spear["goal_threat"] >= away_anchor["defensive_lock"] else away_team,
              "mismatch_score": abs(int(home_spear["goal_threat"] - away_anchor["defensive_lock"])),
              "home_metric_label": "Goal Threat",
              "home_metric_value": int(home_spear["goal_threat"]),
              "away_metric_label": "Defensive Lock",
              "away_metric_value": int(away_anchor["defensive_lock"]),
              "summary": f"{home_spear['surname']} goal threat against {away_anchor['surname']} defensive lock.",
          }
      )
    away_spear = best_player(away_attackers, ("goal_threat", "power"))
    home_anchor = best_player(home_defenders, ("defensive_lock", "aerial_dominance", "goalkeeper_shield"))
    if away_spear and home_anchor:
      pairs.append(
          {
              "zone": f"{away_team} attack spearhead vs {home_team} defensive anchor",
              "home_player": away_spear["surname"],
              "away_player": home_anchor["surname"],
              "advantage": away_team if away_spear["goal_threat"] >= home_anchor["defensive_lock"] else home_team,
              "mismatch_score": abs(int(away_spear["goal_threat"] - home_anchor["defensive_lock"])),
              "home_metric_label": "Goal Threat",
              "home_metric_value": int(away_spear["goal_threat"]),
              "away_metric_label": "Defensive Lock",
              "away_metric_value": int(home_anchor["defensive_lock"]),
              "summary": f"{away_spear['surname']} goal threat against {home_anchor['surname']} defensive lock.",
          }
      )
    home_creator = best_player(home_creators, ("creative_spark", "ball_progression"))
    away_creator = best_player(away_creators, ("creative_spark", "ball_progression"))
    if home_creator and away_creator:
      pairs.append(
          {
              "zone": "Creative control battle",
              "home_player": home_creator["surname"],
              "away_player": away_creator["surname"],
              "advantage": home_team if home_creator["creative_spark"] >= away_creator["creative_spark"] else away_team,
              "mismatch_score": abs(int(home_creator["creative_spark"] - away_creator["creative_spark"])),
              "home_metric_label": "Creative Spark",
              "home_metric_value": int(home_creator["creative_spark"]),
              "away_metric_label": "Creative Spark",
              "away_metric_value": int(away_creator["creative_spark"]),
              "summary": f"{home_creator['surname']} versus {away_creator['surname']} for the creative control layer.",
          }
      )
    return sorted(pairs, key=lambda item: item["mismatch_score"], reverse=True)[:3]


def lineup_profiles_for_rows(
    lineup_rows: pd.DataFrame,
    team_row: pd.Series | None,
    player_lookup: dict[str, dict[str, dict[str, list[dict[str, Any]]]]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    team_name = team_row.get("team") if team_row is not None else ""
    club_bucket = player_lookup.get(normalize_team_key(team_name), {"full": {}, "surname": {}, "initial_surname": {}})
    resolved_players: list[dict[str, Any]] = []
    unresolved = 0
    for _, lineup_row in lineup_rows.iterrows():
        profile = resolve_player_profile(lineup_row.get("player_name"), lineup_row.get("position"), club_bucket)
        if profile is None:
            unresolved += 1
            profile = {
                "name": lineup_row.get("player_name") or "Player",
                "pitch_label": str(lineup_row.get("player_name") or "Player").split()[-1],
                "position_group": inferred_position_group(lineup_row.get("position")),
                "og_player_power": 50,
                "goal_threat": 50,
                "creative_spark": 50,
                "midfield_engine": 50,
                "defensive_lock": 50,
                "pressing_heat": 50,
                "ball_progression": 50,
                "aerial_dominance": 50,
                "goalkeeper_shield": 50,
                "discipline_risk": 50,
                "booking_heat": 50,
                "minutes_played_overall": 0,
                "tags": [],
            }
        resolved_players.append(
            {
                "name": profile.get("name") or lineup_row.get("player_name") or "Player",
                "surname": profile.get("pitch_label") or str(profile.get("name") or lineup_row.get("player_name") or "Player").split()[-1],
                "lineup_position": str(lineup_row.get("position") or "").upper(),
                "position_group": profile.get("position_group") or inferred_position_group(lineup_row.get("position")),
                "power": int(score_to_int(profile.get("og_player_power", 50))),
                "goal_threat": int(score_to_int(profile.get("goal_threat", 50))),
                "creative_spark": int(score_to_int(profile.get("creative_spark", 50))),
                "midfield_engine": int(score_to_int(profile.get("midfield_engine", 50))),
                "defensive_lock": int(score_to_int(profile.get("defensive_lock", 50))),
                "pressing_heat": int(score_to_int(profile.get("pressing_heat", 50))),
                "ball_progression": int(score_to_int(profile.get("ball_progression", 50))),
                "aerial_dominance": int(score_to_int(profile.get("aerial_dominance", 50))),
                "goalkeeper_shield": int(score_to_int(profile.get("goalkeeper_shield", 50))),
                "discipline_risk": int(score_to_int(profile.get("discipline_risk", 50))),
                "booking_heat": int(score_to_int(profile.get("booking_heat", 50))),
                "minutes_played": int(float(profile.get("minutes_played_overall", 0) or 0)),
                "tags": list(profile.get("tags") or [])[:3],
            }
        )
    return resolved_players, {
        "player_count": int(len(lineup_rows)),
        "resolved_profiles": int(len(lineup_rows) - unresolved),
        "fallback_profiles": int(unresolved),
    }


def build_fixture_payloads(entry: dict[str, Any], output_root: Path, write_output: bool = True) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    team_frame = build_team_scores(prepare_team_frame(Path(entry["team_source"])))
    player_frame = build_ranks(build_player_scores(prepare_player_frame(Path(entry["player_source"]))))
    lineup_frame = clean_columns(pd.read_csv(entry["lineups_csv"]))
    feature_frame = clean_columns(pd.read_csv(entry["features_csv"]))

    team_lookup = build_team_lookup(team_frame)
    player_lookup = build_player_lookup(player_frame)

    records: list[dict[str, Any]] = []
    index_records: list[dict[str, Any]] = []

    for _, fixture in feature_frame.iterrows():
        fixture_id = int(fixture["fixture_id"])
        home_team_name = fixture.get("home_team_name") or "Home"
        away_team_name = fixture.get("away_team_name") or "Away"
        home_team = team_lookup.get(normalize_team_key(home_team_name))
        away_team = team_lookup.get(normalize_team_key(away_team_name))

        fixture_lineups = lineup_frame[(lineup_frame["fixture_id"] == fixture_id) & (lineup_frame["is_starting_xi"] == 1)]
        home_rows = fixture_lineups[fixture_lineups["team_id"] == fixture["home_team_id"]]
        away_rows = fixture_lineups[fixture_lineups["team_id"] == fixture["away_team_id"]]

        home_units, home_players, home_resolution = lineup_side_payload("home", home_rows, home_team, player_lookup)
        away_units, away_players, away_resolution = lineup_side_payload("away", away_rows, away_team, player_lookup)
        key_mismatches = build_key_mismatches(home_units, away_units, home_team_name, away_team_name)
        player_matchups = build_player_matchups(home_players, away_players, home_team_name, away_team_name)

        payload = {
            "fixture_key": fixture.get("fixture_key"),
            "fixture_id": fixture_id,
            "competition": entry["competition_name"],
            "competition_key": entry["competition_key"],
            "season": str(entry["season"]),
            "home_team": home_team_name,
            "away_team": away_team_name,
            "home_team_id": int(fixture.get("home_team_id") or 0),
            "away_team_id": int(fixture.get("away_team_id") or 0),
            "home_formation": fixture.get("home_formation") or "",
            "away_formation": fixture.get("away_formation") or "",
            "home_units": home_units,
            "away_units": away_units,
            "home_resolution": home_resolution,
            "away_resolution": away_resolution,
            "key_mismatches": key_mismatches,
            "player_matchups": player_matchups,
            "home_lineup_profiles": home_players,
            "away_lineup_profiles": away_players,
            "formation_context": {
                "same_formation_flag": int(fixture.get("same_formation_flag") or 0),
                "formation_mismatch_flag": int(fixture.get("formation_mismatch_flag") or 0),
                "formation_attack_delta": float(fixture.get("formation_attack_delta") or 0),
                "formation_defence_delta": float(fixture.get("formation_defence_delta") or 0),
            },
        }
        records.append(payload)
        index_records.append(
            {
                "fixture_key": payload["fixture_key"],
                "fixture_id": payload["fixture_id"],
                "competition": payload["competition"],
                "competition_key": payload["competition_key"],
                "season": payload["season"],
                "home_team": payload["home_team"],
                "away_team": payload["away_team"],
            }
        )

    if write_output:
        base = output_root / "fixture_lineup_intelligence"
        ensure_dir(base)
        for payload in records:
            write_json(base / f"{payload['fixture_key']}.json", payload)

        index_path = base / "index.json"
        existing: list[dict[str, Any]] = []
        if index_path.exists():
            existing = json.loads(index_path.read_text(encoding="utf-8"))
            existing = [
                row
                for row in existing
                if not (
                    row.get("competition_key") == entry["competition_key"]
                    and str(row.get("season")) == str(entry["season"])
                )
            ]
        write_json(
            index_path,
            sorted(existing + index_records, key=lambda row: (row["competition_key"], row["season"], row["fixture_key"])),
        )
    return records, index_records


def latest_team_fixture_rows(feature_frame: pd.DataFrame) -> dict[tuple[str, str], tuple[pd.Series, str]]:
    team_rows: dict[tuple[str, str], tuple[pd.Series, str]] = {}
    sorted_features = feature_frame.sort_values(["match_date", "fixture_id"], ascending=[False, False])
    for _, fixture in sorted_features.iterrows():
        for side in ("home", "away"):
            team_id = str(fixture.get(f"{side}_team_id") or "").strip()
            team_name = str(fixture.get(f"{side}_team_name") or "").strip()
            key = (team_id, normalize_team_key(team_name))
            if not team_id and not key[1]:
                continue
            if key not in team_rows:
                team_rows[key] = (fixture, side)
    return team_rows


def build_team_lineup_snapshots(entry: dict[str, Any]) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    team_frame = build_team_scores(prepare_team_frame(Path(entry["team_source"])))
    player_frame = build_ranks(build_player_scores(prepare_player_frame(Path(entry["player_source"]))))
    lineup_frame = clean_columns(pd.read_csv(entry["lineups_csv"]))
    feature_frame = clean_columns(pd.read_csv(entry["features_csv"]))

    team_lookup = build_team_lookup(team_frame)
    player_lookup = build_player_lookup(player_frame)
    snapshots: dict[tuple[str, str], dict[str, Any]] = {}
    index_rows: list[dict[str, Any]] = []

    for (team_id_key, normalized_name), (fixture, side) in latest_team_fixture_rows(feature_frame).items():
        fixture_id = int(fixture.get("fixture_id") or 0)
        team_id = int(fixture.get(f"{side}_team_id") or 0)
        opponent_side = "away" if side == "home" else "home"
        team_name = fixture.get(f"{side}_team_name") or ""
        opponent_name = fixture.get(f"{opponent_side}_team_name") or ""
        team_row = team_lookup.get(normalize_team_key(team_name))
        fixture_lineups = lineup_frame[lineup_frame["fixture_id"] == fixture_id]
        team_lineups = fixture_lineups[fixture_lineups["team_id"] == team_id]
        starters_rows = team_lineups[team_lineups["is_starting_xi"] == 1]
        bench_rows = team_lineups[team_lineups["is_starting_xi"] != 1]
        if starters_rows.empty and bench_rows.empty:
            continue

        units, starters, starter_resolution = lineup_side_payload(side, starters_rows, team_row, player_lookup)
        bench, bench_resolution = lineup_profiles_for_rows(bench_rows, team_row, player_lookup)
        formation = fixture.get(f"{side}_formation") or (team_lineups["formation"].dropna().astype(str).iloc[0] if not team_lineups.empty else "")
        snapshot_key = slugify(str(team_name))
        payload = {
            "payload_type": "team_latest_lineup_snapshot",
            "team": team_name,
            "team_key": snapshot_key,
            "team_id": team_id,
            "competition": entry["competition_name"],
            "competition_key": entry["competition_key"],
            "season": str(entry["season"]),
            "snapshot_source_year": str(entry.get("snapshot_source_year") or season_start(entry.get("season"))),
            "source_fixture_key": fixture.get("fixture_key"),
            "source_fixture_id": fixture_id,
            "source_match_date": str(fixture.get("match_date") or ""),
            "source_opponent": opponent_name,
            "source_venue": side,
            "formation": formation,
            "lineup_status": "last_fixture_snapshot",
            "lineup_mode": "team_snapshot_last_fixture",
            "lineup_label": "Last fixture lineup snapshot",
            "prediction_use": "Use this compact team snapshot as the predicted lineup source for upcoming fixtures.",
            "confirmed_lineup_update_model": CONFIRMED_LINEUP_UPDATE_MODEL,
            "refresh_policy": "Use this snapshot as the predicted lineup until a confirmed upstream lineup is published close to kickoff.",
            "units": units,
            "starters": starters,
            "bench": bench,
            "starter_resolution": starter_resolution,
            "bench_resolution": bench_resolution,
            "summary": (
                f"Most recent published lineup snapshot for {team_name}: {formation or 'formation pending'} "
                f"against {opponent_name} on {fixture.get('match_date') or 'the latest available fixture'}."
            ),
        }
        lookup_key = (normalize_competition_key(entry["competition_name"]), normalize_team_key(team_name))
        snapshots[lookup_key] = payload
        index_rows.append(
            {
                "team": team_name,
                "team_key": snapshot_key,
                "team_id": team_id,
                "competition": entry["competition_name"],
                "competition_key": entry["competition_key"],
                "season": str(entry["season"]),
                "snapshot_source_year": payload["snapshot_source_year"],
                "source_fixture_key": payload["source_fixture_key"],
                "source_match_date": payload["source_match_date"],
                "formation": payload["formation"],
                "starter_count": len(starters),
                "bench_count": len(bench),
            }
        )
    return snapshots, index_rows


def placeholder_current_fixture_payload(fixture: dict[str, Any]) -> dict[str, Any]:
    return {
        "fixture_key": str(fixture.get("fixture_key") or "").strip(),
        "fixture_id": safe_int_value(fixture.get("api_fixture_id") or fixture.get("fixture_id")),
        "competition": fixture.get("league"),
        "competition_key": slugify(fixture.get("league") or ""),
        "season": str(fixture.get("api_season") or fixture.get("season") or ""),
        "home_team": fixture.get("home_team"),
        "away_team": fixture.get("away_team"),
        "home_formation": "",
        "away_formation": "",
        "home_units": {},
        "away_units": {},
        "home_resolution": {"starting_xi_count": 0, "resolved_profiles": 0, "fallback_profiles": 0},
        "away_resolution": {"starting_xi_count": 0, "resolved_profiles": 0, "fallback_profiles": 0},
        "key_mismatches": [],
        "player_matchups": [],
        "home_lineup_profiles": [],
        "away_lineup_profiles": [],
        "home_bench_profiles": [],
        "away_bench_profiles": [],
        "coverage_status": "unpublished",
        "lineup_status": "unavailable",
        "lineup_mode": "unavailable_fallback",
        "lineup_label": "Lineup unavailable",
        "fallback_mode": "unpublished",
        "confirmed_lineup_update_model": CONFIRMED_LINEUP_UPDATE_MODEL,
        "summary": "No publish-safe last-known lineup snapshot is available yet for at least one team in this fixture.",
    }


def find_team_snapshot(
    snapshots: dict[tuple[str, str], dict[str, Any]],
    competition: str,
    team_name: object,
) -> dict[str, Any] | None:
    exact = snapshots.get((competition, normalize_team_key(team_name)))
    if exact:
        return exact

    best: tuple[int, dict[str, Any] | None] = (0, None)
    for (snapshot_competition, _snapshot_team_key), snapshot in snapshots.items():
        if snapshot_competition != competition:
            continue
        score = team_match_score(team_name, snapshot.get("team"))
        if score > best[0]:
            best = (score, snapshot)
    return best[1] if best[0] >= 80 else None


def build_current_fixture_snapshot_payloads(
    fixture_feed_rows: list[dict[str, Any]],
    snapshots: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    for fixture in fixture_feed_rows:
        fixture_key = str(fixture.get("fixture_key") or "").strip()
        if not fixture_key:
            continue
        competition = normalize_competition_key(fixture.get("league"))
        home_snapshot = find_team_snapshot(snapshots, competition, fixture.get("home_team"))
        away_snapshot = find_team_snapshot(snapshots, competition, fixture.get("away_team"))
        if not home_snapshot or not away_snapshot:
            payload = placeholder_current_fixture_payload(fixture)
        else:
            home_units = dict(home_snapshot.get("units") or {})
            away_units = dict(away_snapshot.get("units") or {})
            home_players = list(home_snapshot.get("starters") or [])
            away_players = list(away_snapshot.get("starters") or [])
            payload = {
                "fixture_key": fixture_key,
                "fixture_id": safe_int_value(fixture.get("api_fixture_id") or fixture.get("fixture_id")),
                "competition": fixture.get("league") or home_snapshot.get("competition"),
                "competition_key": home_snapshot.get("competition_key"),
                "season": str(fixture.get("api_season") or fixture.get("season") or home_snapshot.get("season") or ""),
                "home_team": fixture.get("home_team") or home_snapshot.get("team"),
                "away_team": fixture.get("away_team") or away_snapshot.get("team"),
                "home_team_id": home_snapshot.get("team_id"),
                "away_team_id": away_snapshot.get("team_id"),
                "home_formation": home_snapshot.get("formation") or "",
                "away_formation": away_snapshot.get("formation") or "",
                "home_units": home_units,
                "away_units": away_units,
                "home_resolution": home_snapshot.get("starter_resolution"),
                "away_resolution": away_snapshot.get("starter_resolution"),
                "key_mismatches": build_key_mismatches(home_units, away_units, fixture.get("home_team") or "Home", fixture.get("away_team") or "Away"),
                "player_matchups": build_player_matchups(home_players, away_players, fixture.get("home_team") or "Home", fixture.get("away_team") or "Away"),
                "home_lineup_profiles": home_players,
                "away_lineup_profiles": away_players,
                "home_bench_profiles": list(home_snapshot.get("bench") or []),
                "away_bench_profiles": list(away_snapshot.get("bench") or []),
                "home_snapshot": {
                    "source_fixture_key": home_snapshot.get("source_fixture_key"),
                    "source_match_date": home_snapshot.get("source_match_date"),
                    "source_opponent": home_snapshot.get("source_opponent"),
                    "source_venue": home_snapshot.get("source_venue"),
                },
                "away_snapshot": {
                    "source_fixture_key": away_snapshot.get("source_fixture_key"),
                    "source_match_date": away_snapshot.get("source_match_date"),
                    "source_opponent": away_snapshot.get("source_opponent"),
                    "source_venue": away_snapshot.get("source_venue"),
                },
                "coverage_status": "predicted",
                "lineup_status": "predicted_from_last_fixture",
                "lineup_mode": "predicted_from_last_fixture",
                "lineup_label": "Predicted lineups",
                "fallback_mode": "last_fixture_snapshot",
                "confirmed_lineup_update_model": CONFIRMED_LINEUP_UPDATE_MODEL,
                "refresh_policy": "Replace this last-known prediction with confirmed team sheets when upstream lineups publish close to kickoff.",
                "formation_context": {
                    "same_formation_flag": int((home_snapshot.get("formation") or "") == (away_snapshot.get("formation") or "")),
                    "formation_mismatch_flag": int((home_snapshot.get("formation") or "") != (away_snapshot.get("formation") or "")),
                    "formation_attack_delta": 0.0,
                    "formation_defence_delta": 0.0,
                },
                "summary": "Predicted lineups are built from each team's most recent published lineup and bench snapshot.",
            }
        records.append(payload)
        index_rows.append(
            {
                "fixture_key": fixture_key,
                "fixture_id": payload.get("fixture_id"),
                "competition": payload.get("competition"),
                "competition_key": payload.get("competition_key"),
                "season": payload.get("season"),
                "home_team": payload.get("home_team"),
                "away_team": payload.get("away_team"),
                "coverage_status": payload.get("coverage_status"),
                "lineup_status": payload.get("lineup_status"),
                "lineup_mode": payload.get("lineup_mode"),
            }
        )
    return records, index_rows


def publish_team_snapshots(snapshots: dict[tuple[str, str], dict[str, Any]], index_rows: list[dict[str, Any]], output_root: Path) -> None:
    base = output_root / "fixture_lineup_intelligence" / "team_snapshots"
    ensure_dir(base)
    keep_paths: set[Path] = set()
    for payload in snapshots.values():
        comp_dir = base / str(payload.get("competition_key") or "unknown")
        ensure_dir(comp_dir)
        target = comp_dir / f"{payload.get('team_key')}.json"
        write_json(target, payload)
        keep_paths.add(target)
    for path in base.glob("*/*.json"):
        if path not in keep_paths:
            path.unlink()
    write_json(base / "index.json", sorted(index_rows, key=lambda row: (row["competition_key"], row["team_key"])))


def publish_current_fixture_lineups(records: list[dict[str, Any]], index_rows: list[dict[str, Any]], output_root: Path) -> None:
    base = output_root / "fixture_lineup_intelligence"
    ensure_dir(base)
    keep = {f"{payload['fixture_key']}.json" for payload in records}
    keep.add("index.json")
    for payload in records:
        write_json(base / f"{payload['fixture_key']}.json", payload)
    for path in base.glob("*.json"):
        if path.name not in keep:
            path.unlink()
    write_json(base / "index.json", sorted(index_rows, key=lambda row: (row["competition_key"] or "", row["fixture_key"])))


def load_fixture_feed_rows(path: Path | None) -> list[dict[str, Any]]:
    if not path or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("fixtures") if isinstance(payload, dict) else None
    return list(rows) if isinstance(rows, list) else []


def alias_current_fixture_keys(
    records: list[dict[str, Any]],
    index_records: list[dict[str, Any]],
    fixture_feed_rows: list[dict[str, Any]],
    competition_name: str,
    season: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not fixture_feed_rows:
        return records, index_records

    target_competition = normalize_text(competition_name)
    target_season = season_start(season)
    relevant_rows = [
        row
        for row in fixture_feed_rows
        if normalize_text(row.get("league")) == target_competition
        and (
            not season_start(row.get("api_season") or row.get("season"))
            or season_start(row.get("api_season") or row.get("season")) == target_season
        )
    ]
    if not relevant_rows:
        return records, index_records

    by_fixture_id = {
        str(row.get("api_fixture_id") or row.get("fixture_id") or "").strip(): row
        for row in relevant_rows
        if str(row.get("api_fixture_id") or row.get("fixture_id") or "").strip()
    }
    by_pair = {
        (
            normalize_team_key(row.get("home_team")),
            normalize_team_key(row.get("away_team")),
        ): row
        for row in relevant_rows
    }

    seen_keys = {record["fixture_key"] for record in index_records}
    aliased_records = list(records)
    aliased_index = list(index_records)
    for payload in records:
        fixture_row = None
        fixture_id_key = str(payload.get("fixture_id") or "").strip()
        if fixture_id_key:
            fixture_row = by_fixture_id.get(fixture_id_key)
        if not fixture_row:
            fixture_row = by_pair.get(
                (
                    normalize_team_key(payload.get("home_team")),
                    normalize_team_key(payload.get("away_team")),
                )
            )
        if not fixture_row:
            continue
        current_key = str(fixture_row.get("fixture_key") or "").strip()
        if not current_key or current_key in seen_keys:
            continue
        aliased_payload = dict(payload)
        aliased_payload.update(
            {
                "fixture_key": current_key,
                "fixture_id": int(fixture_row.get("api_fixture_id") or payload.get("fixture_id") or 0) or payload.get("fixture_id"),
                "home_team": fixture_row.get("home_team") or payload.get("home_team"),
                "away_team": fixture_row.get("away_team") or payload.get("away_team"),
                "site_alias_source_fixture_key": payload.get("fixture_key"),
            }
        )
        aliased_records.append(aliased_payload)
        aliased_index.append(
            {
                "fixture_key": current_key,
                "fixture_id": aliased_payload["fixture_id"],
                "competition": payload["competition"],
                "competition_key": payload["competition_key"],
                "season": payload["season"],
                "home_team": aliased_payload["home_team"],
                "away_team": aliased_payload["away_team"],
            }
        )
        seen_keys.add(current_key)
    return aliased_records, aliased_index


def main() -> None:
    args = parse_args()
    entries = load_entries(args)
    output_root = Path(args.output_root)
    fixture_feed_rows = load_fixture_feed_rows(Path(args.fixture_feed)) if args.fixture_feed else []

    if fixture_feed_rows:
        all_snapshots: dict[tuple[str, str], dict[str, Any]] = {}
        snapshot_index_rows: list[dict[str, Any]] = []
        for entry in entries:
            snapshots, rows = build_team_lineup_snapshots(entry)
            all_snapshots.update(snapshots)
            snapshot_index_rows.extend(rows)
            print(
                f"Built {len(snapshots)} team lineup snapshots for "
                f"{entry['competition_name']} from {entry.get('snapshot_source_year') or season_start(entry.get('season'))}"
            )

        records, index_records = build_current_fixture_snapshot_payloads(fixture_feed_rows, all_snapshots)
        publish_team_snapshots(all_snapshots, snapshot_index_rows, output_root)
        publish_current_fixture_lineups(records, index_records, output_root)
        print(f"Published {len(records)} predicted fixture lineup payloads from {len(all_snapshots)} team snapshots.")
        return

    total = 0
    for entry in entries:
        records, _ = build_fixture_payloads(entry, output_root)
        total += len(records)
        print(
            f"Built {len(records)} fixture lineup intelligence profiles for "
            f"{entry['competition_name']} {entry['season']}"
        )
    print(f"Built {total} fixture lineup intelligence profiles in total")


if __name__ == "__main__":
    main()
