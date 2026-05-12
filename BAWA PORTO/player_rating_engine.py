from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ratings_engine_utils import (
    blend_with_neutral,
    clean_columns,
    ensure_dir,
    player_confidence_label,
    player_confidence_multiplier,
    rating_band,
    score_to_int,
    safe_divide,
    slugify,
    weighted_average,
    write_json,
)


PLAYER_RATING_FIELDS = [
    "og_player_power",
    "goal_threat",
    "shot_threat",
    "xg_threat",
    "creative_spark",
    "xa_threat",
    "midfield_engine",
    "defensive_lock",
    "pressing_heat",
    "ball_progression",
    "aerial_dominance",
    "goalkeeper_shield",
    "discipline_risk",
    "booking_heat",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build publish-safe Odds Genius player ratings.")
    parser.add_argument("--input", help="Single player season CSV to process.")
    parser.add_argument("--config", help="JSON config listing player sources.")
    parser.add_argument("--competition-name", help="Public competition name override.")
    parser.add_argument("--competition-key", help="Stable competition key override.")
    parser.add_argument("--season", help="Season label override.")
    parser.add_argument(
        "--output-root",
        default="frontend/public/data",
        help="Publish output root. Default: frontend/public/data",
    )
    return parser.parse_args()


def load_entries(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.config:
        payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
        return list(payload.get("players", []))
    if args.input:
        return [
            {
                "source": args.input,
                "competition_name": args.competition_name,
                "competition_key": args.competition_key,
                "season": args.season,
            }
        ]
    raise SystemExit("Provide either --input or --config.")


def normalize_position_group(position: object) -> str:
    text = str(position or "").strip().lower()
    if "goal" in text:
        return "goalkeeper"
    if "wing" in text:
        return "winger"
    if "forward" in text or "striker" in text or text == "attacker":
        return "forward"
    if "defensive midfielder" in text or "dm" in text:
        return "defensive_midfielder"
    if "attacking midfielder" in text or "am" in text:
        return "attacking_midfielder"
    if "midfielder" in text or text == "midfield":
        return "central_midfielder"
    if "full" in text or "left back" in text or "right back" in text or "wing back" in text:
        return "full_back"
    if "centre back" in text or "center back" in text or "defender" in text or "cb" in text:
        return "centre_back"
    return "utility"


def infer_surname(full_name: object) -> str:
    parts = str(full_name or "").strip().split()
    return parts[-1] if parts else "Player"


def prepare_player_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = clean_columns(df)

    competition = df.get("league", pd.Series("Unknown Competition", index=df.index)).fillna("Unknown Competition")
    club = df.get("current_club", pd.Series("Unknown Club", index=df.index)).fillna("Unknown Club")
    name = df.get("full_name", pd.Series("Unknown Player", index=df.index)).fillna("Unknown Player")
    position = df.get("position", pd.Series("", index=df.index)).fillna("")
    minutes = pd.to_numeric(df.get("minutes_played_overall"), errors="coerce").replace(0, np.nan)

    derived = {
        "competition": competition,
        "club": club,
        "name": name,
        "player_slug": name.map(slugify),
        "club_slug": club.map(slugify),
        "position_group": position.map(normalize_position_group),
        "pitch_label": name.map(infer_surname),
        "progressive_passes_per_90": safe_divide(df.get("progressive_passes_total_overall"), minutes) * 90.0,
        "accurate_crosses_per_90": safe_divide(df.get("accurate_crosses_total_overall"), minutes) * 90.0,
        "yellow_red_card_load": pd.to_numeric(df.get("yellow_cards_overall"), errors="coerce").fillna(0)
        + (
        pd.to_numeric(df.get("red_cards_overall"), errors="coerce").fillna(0) * 2.0
        ),
    }
    return df.assign(**derived).copy()


def build_player_scores(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    confidence = player_confidence_multiplier(pd.to_numeric(df.get("minutes_played_overall"), errors="coerce").fillna(0))

    raw_scores = {
        "goal_threat": weighted_average(
            scored,
            [
                ("goals_per_90_overall", True, 1.2),
                ("xg_per_90_overall", True, 1.1),
                ("npxg_per_90_overall", True, 1.0),
                ("shots_per_90_overall", True, 1.0),
                ("shots_on_target_per_90_overall", True, 1.0),
                ("shot_conversion_rate_overall", True, 0.8),
                ("goals_involved_per_90_overall", True, 0.8),
            ],
        ),
        "shot_threat": weighted_average(
            scored,
            [
                ("shots_per_90_overall", True, 1.2),
                ("shots_on_target_per_90_overall", True, 1.1),
                ("xg_per_90_overall", True, 1.0),
                ("shot_accuraccy_percentage_overall", True, 0.8),
            ],
        ),
        "xg_threat": weighted_average(
            scored,
            [
                ("xg_per_90_overall", True, 1.2),
                ("npxg_per_90_overall", True, 1.1),
                ("shots_on_target_per_90_overall", True, 0.7),
                ("goals_per_90_overall", True, 0.6),
            ],
        ),
        "creative_spark": weighted_average(
            scored,
            [
                ("assists_per_90_overall", True, 1.0),
                ("xa_per_90_overall", True, 1.1),
                ("chances_created_per_90_overall", True, 1.1),
                ("key_passes_per_90_overall", True, 1.0),
                ("through_passes_per_90_overall", True, 0.8),
                ("accurate_crosses_per_90", True, 0.7),
                ("progressive_passes_per_90", True, 0.6),
            ],
        ),
        "xa_threat": weighted_average(
            scored,
            [
                ("xa_per_90_overall", True, 1.2),
                ("key_passes_per_90_overall", True, 1.0),
                ("chances_created_per_90_overall", True, 1.0),
                ("through_passes_per_90_overall", True, 0.8),
            ],
        ),
        "midfield_engine": weighted_average(
            scored,
            [
                ("passes_per_90_overall", True, 1.0),
                ("pass_completion_rate_overall", True, 1.0),
                ("progressive_passes_per_90", True, 0.9),
                ("possession_regained_per_90_overall", True, 0.9),
                ("pressures_per_90_overall", True, 0.8),
                ("tackles_per_90_overall", True, 0.8),
                ("interceptions_per_90_overall", True, 0.8),
                ("distance_travelled_per_90_overall", True, 0.6),
                ("duels_won_percentage_overall", True, 0.6),
            ],
        ),
        "defensive_lock": weighted_average(
            scored,
            [
                ("tackles_successful_per_90_overall", True, 1.0),
                ("interceptions_per_90_overall", True, 1.0),
                ("clearances_per_90_overall", True, 0.9),
                ("blocks_per_90_overall", True, 0.8),
                ("aerial_duels_won_percentage_overall", True, 0.8),
                ("duels_won_percentage_overall", True, 0.8),
                ("dribbled_past_per_90_overall", False, 0.8),
            ],
        ),
        "pressing_heat": weighted_average(
            scored,
            [
                ("pressures_per_90_overall", True, 1.1),
                ("possession_regained_per_90_overall", True, 1.0),
                ("tackles_per_90_overall", True, 0.9),
                ("interceptions_per_90_overall", True, 0.8),
                ("fouls_committed_per_90_overall", True, 0.6),
                ("distance_travelled_per_90_overall", True, 0.8),
            ],
        ),
        "ball_progression": weighted_average(
            scored,
            [
                ("progressive_passes_per_90", True, 1.2),
                ("through_passes_per_90_overall", True, 0.9),
                ("long_passes_per_90_overall", True, 0.7),
                ("dribbles_successful_per_90_overall", True, 0.8),
                ("dribbles_per_90_overall", True, 0.7),
                ("pass_completion_rate_overall", True, 0.7),
                ("key_passes_per_90_overall", True, 0.6),
            ],
        ),
        "aerial_dominance": weighted_average(
            scored,
            [
                ("aerial_duels_per_90_overall", True, 0.8),
                ("aerial_duels_won_per_90_overall", True, 1.0),
                ("aerial_duels_won_percentage_overall", True, 1.0),
                ("clearances_per_90_overall", True, 0.7),
                ("goals_per_90_overall", True, 0.3),
            ],
        ),
        "goalkeeper_shield": weighted_average(
            scored,
            [
                ("saves_per_90_overall", True, 1.0),
                ("save_percentage_overall", True, 1.1),
                ("shots_faced_per_90_overall", True, 0.7),
                ("xg_faced_per_90_overall", True, 0.7),
                ("conceded_per_90_overall", False, 1.0),
                ("clean_sheets_percentage_percentile_overall", True, 0.8),
                ("shots_per_goal_conceded_overall", True, 0.7),
                ("inside_box_saves_total_overall", True, 0.5),
                ("pens_saved_total_overall", True, 0.3),
            ],
        ),
        "discipline_risk": weighted_average(
            scored,
            [
                ("cards_per_90_overall", True, 1.1),
                ("booked_over05_percentage_overall", True, 1.1),
                ("fouls_committed_per_90_overall", True, 1.0),
                ("tackles_per_90_overall", True, 0.7),
                ("pressures_per_90_overall", True, 0.5),
                ("yellow_red_card_load", True, 0.8),
            ],
        ),
        "booking_heat": weighted_average(
            scored,
            [
                ("booked_over05_percentage_overall", True, 1.2),
                ("cards_per_90_overall", True, 1.0),
                ("min_per_card_overall", False, 1.0),
                ("fouls_committed_per_90_overall", True, 0.8),
                ("tackles_per_90_overall", True, 0.7),
                ("pressures_per_90_overall", True, 0.5),
            ],
        ),
    }

    for key, raw in raw_scores.items():
        scored[key] = score_to_int(blend_with_neutral(raw, confidence))

    score_frame = scored
    power_weights = {
        "goalkeeper": {"goalkeeper_shield": 0.45, "defensive_lock": 0.2, "aerial_dominance": 0.1, "discipline_risk": -0.05},
        "centre_back": {"defensive_lock": 0.35, "aerial_dominance": 0.2, "pressing_heat": 0.1, "ball_progression": 0.1},
        "full_back": {"defensive_lock": 0.2, "creative_spark": 0.15, "ball_progression": 0.2, "pressing_heat": 0.15},
        "defensive_midfielder": {"midfield_engine": 0.25, "defensive_lock": 0.2, "pressing_heat": 0.2, "ball_progression": 0.1},
        "central_midfielder": {"midfield_engine": 0.3, "ball_progression": 0.2, "creative_spark": 0.15, "pressing_heat": 0.1},
        "attacking_midfielder": {"creative_spark": 0.3, "xa_threat": 0.2, "ball_progression": 0.15, "goal_threat": 0.1},
        "winger": {"creative_spark": 0.25, "goal_threat": 0.25, "shot_threat": 0.1, "ball_progression": 0.15, "pressing_heat": 0.1},
        "forward": {"goal_threat": 0.35, "shot_threat": 0.2, "xg_threat": 0.2, "creative_spark": 0.05, "aerial_dominance": 0.1},
        "utility": {"midfield_engine": 0.15, "creative_spark": 0.15, "goal_threat": 0.15, "defensive_lock": 0.15, "pressing_heat": 0.1},
    }

    og_power = []
    for _, row in score_frame.iterrows():
        weights = power_weights.get(row["position_group"], power_weights["utility"])
        total = 0.0
        total_weight = 0.0
        for field, weight in weights.items():
            total += float(row[field]) * weight
            total_weight += weight
        total += float(row.get("average_rating_percentile_overall", 50)) * 0.15
        total += float(row.get("minutes_played_percentile_overall", 50)) * 0.1
        total_weight += 0.25
        og_power.append(int(score_to_int(total / max(total_weight, 1.0))))
    score_frame["og_player_power"] = og_power
    return score_frame


def build_ranks(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()
    ranked["league_overall_rank"] = ranked["og_player_power"].rank(method="min", ascending=False).astype(int)
    ranked["position_rank"] = ranked.groupby("position_group")["og_player_power"].rank(method="min", ascending=False).astype(int)
    ranked["club_rank"] = ranked.groupby("club")["og_player_power"].rank(method="min", ascending=False).astype(int)
    for field in ["goal_threat", "creative_spark", "defensive_lock", "pressing_heat", "booking_heat"]:
        ranked[f"{field}_rank"] = ranked[field].rank(method="min", ascending=False).astype(int)
    return ranked


def build_tags(row: pd.Series) -> list[str]:
    tags: list[str] = []
    if row["goal_threat"] >= 85:
        tags.append("Elite Goal Threat")
    if row["creative_spark"] >= 85:
        tags.append("Elite Creator")
    if row["pressing_heat"] >= 80:
        tags.append("Pressing Engine")
    if row["defensive_lock"] >= 85:
        tags.append("Defensive Anchor")
    if row["aerial_dominance"] >= 85:
        tags.append("Aerial Monster")
    if row["booking_heat"] >= 75:
        tags.append("Booking Risk")
    if row["ball_progression"] >= 80:
        tags.append("Progressive Carrier")
    if row["minutes_played_overall"] < 600:
        tags.append("Low Sample")
    return tags[:4] or [f"{str(row['position_group']).replace('_', ' ').title()} Profile"]


def player_explanation(field: str, score: int) -> str:
    label = rating_band(score)
    templates = {
        "og_player_power": f"{label} headline player impact once role, production, and minutes trust are combined.",
        "goal_threat": f"{label} scoring profile with xG and shot support.",
        "shot_threat": f"{label} shooting volume and on-target pressure.",
        "xg_threat": f"{label} chance-quality profile.",
        "creative_spark": f"{label} creation profile for final-third value.",
        "xa_threat": f"{label} assist expectation profile.",
        "midfield_engine": f"{label} central involvement and recycling profile.",
        "defensive_lock": f"{label} defensive action and duel profile.",
        "pressing_heat": f"{label} pressure and regain activity profile.",
        "ball_progression": f"{label} carrying and forward progression profile.",
        "aerial_dominance": f"{label} aerial and set-piece profile.",
        "goalkeeper_shield": f"{label} shot-stopping and resistance profile.",
        "discipline_risk": f"{label} foul and discipline exposure profile.",
        "booking_heat": f"{label} booking-market risk profile.",
    }
    return templates[field]


def export_player_payloads(df: pd.DataFrame, competition_name: str, competition_key: str, season: str, output_root: Path) -> None:
    base = output_root / "player_intelligence"
    club_dir = base / "clubs" / competition_key / str(season)
    csv_path = base / "player_ratings.csv"
    json_path = base / "player_ratings.json"
    club_index_path = base / "club_squad_ratings.json"

    ensure_dir(club_dir)

    players = []
    clubs: dict[str, Any] = {}
    for _, row in df.iterrows():
        ratings = {field: int(row[field]) for field in PLAYER_RATING_FIELDS}
        payload = {
            "player_id": f"{competition_key}_{row['club_slug']}_{row['player_slug']}_{season}",
            "name": row["name"],
            "surname": row["pitch_label"],
            "club": row["club"],
            "club_slug": row["club_slug"],
            "competition": competition_name,
            "competition_key": competition_key,
            "season": str(season),
            "position": row.get("position"),
            "position_group": row["position_group"],
            "minutes_confidence": {
                "label": player_confidence_label(float(row.get("minutes_played_overall") or 0)),
                "minutes_played": int(row.get("minutes_played_overall") or 0),
            },
            "ratings": ratings,
            "rating_bands": {name: rating_band(score) for name, score in ratings.items()},
            "rating_explanations": {name: player_explanation(name, score) for name, score in ratings.items()},
            "ranks": {
                "league_overall_rank": int(row["league_overall_rank"]),
                "position_rank": int(row["position_rank"]),
                "club_rank": int(row["club_rank"]),
                "goal_threat_rank": int(row["goal_threat_rank"]),
                "creative_spark_rank": int(row["creative_spark_rank"]),
                "defensive_lock_rank": int(row["defensive_lock_rank"]),
                "pressing_heat_rank": int(row["pressing_heat_rank"]),
                "booking_heat_rank": int(row["booking_heat_rank"]),
            },
            "tags": build_tags(row),
            "ui": {
                "pitch_label": row["pitch_label"],
                "badge_score": int(row["og_player_power"]),
                "tier": rating_band(int(row["og_player_power"])).upper().replace(" ", "_"),
            },
        }
        players.append(payload)

        club_payload = clubs.setdefault(
            row["club_slug"],
            {
                "club": row["club"],
                "club_slug": row["club_slug"],
                "competition": competition_name,
                "competition_key": competition_key,
                "season": str(season),
                "players": [],
            },
        )
        club_payload["players"].append(payload)

    for club_slug, payload in clubs.items():
        payload["players"] = sorted(payload["players"], key=lambda item: item["ratings"]["og_player_power"], reverse=True)
        payload["leaders"] = {
            "power": [p["surname"] for p in payload["players"][:5]],
            "goal_threat": [p["surname"] for p in sorted(payload["players"], key=lambda item: item["ratings"]["goal_threat"], reverse=True)[:5]],
            "creative_spark": [p["surname"] for p in sorted(payload["players"], key=lambda item: item["ratings"]["creative_spark"], reverse=True)[:5]],
            "discipline_risk": [p["surname"] for p in sorted(payload["players"], key=lambda item: item["ratings"]["discipline_risk"], reverse=True)[:5]],
        }
        write_json(club_dir / f"{club_slug}.json", payload)

    flat = []
    for payload in players:
        row = {
            "name": payload["name"],
            "club": payload["club"],
            "competition": payload["competition"],
            "competition_key": payload["competition_key"],
            "season": payload["season"],
            "position_group": payload["position_group"],
            "league_overall_rank": payload["ranks"]["league_overall_rank"],
            "position_rank": payload["ranks"]["position_rank"],
            "club_rank": payload["ranks"]["club_rank"],
        }
        row.update(payload["ratings"])
        flat.append(row)

    fresh_rows = []
    if csv_path.exists():
        existing_csv = pd.read_csv(csv_path)
        if not existing_csv.empty:
            existing_csv = existing_csv[
                ~(
                    (existing_csv["competition_key"].astype(str) == competition_key)
                    & (existing_csv["season"].astype(str) == str(season))
                )
            ]
            fresh_rows.append(existing_csv)
    fresh_rows.append(pd.DataFrame(flat))
    pd.concat(fresh_rows, ignore_index=True).to_csv(csv_path, index=False)

    existing_players: list[dict[str, Any]] = []
    if json_path.exists():
        existing_players = json.loads(json_path.read_text(encoding="utf-8"))
        existing_players = [
            row
            for row in existing_players
            if not (
                str(row.get("competition_key")) == competition_key
                and str(row.get("season")) == str(season)
            )
        ]
    write_json(
        json_path,
        sorted(existing_players + players, key=lambda item: (item["competition_key"], item["season"], item["club"], item["name"])),
    )

    existing_clubs: list[dict[str, Any]] = []
    if club_index_path.exists():
        existing_clubs = json.loads(club_index_path.read_text(encoding="utf-8"))
        existing_clubs = [
            row
            for row in existing_clubs
            if not (
                str(row.get("competition_key")) == competition_key
                and str(row.get("season")) == str(season)
            )
        ]
    club_index_rows = []
    for payload in clubs.values():
        players = payload.get("players") or []
        club_index_rows.append(
            {
                "club": payload["club"],
                "club_slug": payload["club_slug"],
                "competition": payload["competition"],
                "competition_key": payload["competition_key"],
                "season": payload["season"],
                "player_count": len(players),
                "top_power": players[0]["ratings"]["og_player_power"] if players else None,
                "leaders": payload.get("leaders", {}),
            }
        )
    write_json(
        club_index_path,
        sorted(existing_clubs + club_index_rows, key=lambda item: (item["competition_key"], item["season"], item["club"])),
    )


def process_entry(entry: dict[str, Any], output_root: Path) -> None:
    source = Path(entry["source"])
    df = prepare_player_frame(source)
    competition_name = entry.get("competition_name") or str(df["competition"].iloc[0])
    competition_key = entry.get("competition_key") or slugify(competition_name)
    season = str(entry.get("season") or df.get("season", pd.Series(["unknown"])).iloc[0])
    scored = build_player_scores(df)
    ranked = build_ranks(scored)
    export_player_payloads(ranked, competition_name, competition_key, season, output_root)
    print(f"Built {len(ranked)} player intelligence profiles for {competition_name} {season}")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    entries = load_entries(args)
    for entry in entries:
        process_entry(entry, output_root)


if __name__ == "__main__":
    main()
