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
    market_lean,
    rating_band,
    score_to_int,
    series_or_default,
    safe_divide,
    slugify,
    team_confidence_label,
    team_confidence_multiplier,
    weighted_average,
    write_json,
)


TEAM_RATING_FIELDS = [
    "og_power_rating",
    "attack_flow_rating",
    "defensive_lock_rating",
    "goal_heat_rating",
    "btts_pressure_rating",
    "over25_heat_rating",
    "control_rating",
    "first_strike_rating",
    "corner_pressure_rating",
    "card_heat_rating",
    "chaos_rating",
    "home_fortress_rating",
    "away_threat_rating",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build publish-safe Odds Genius team ratings.")
    parser.add_argument("--input", help="Single team season CSV to process.")
    parser.add_argument("--config", help="JSON config listing team sources.")
    parser.add_argument("--competition-name", help="Public competition name override.")
    parser.add_argument("--competition-key", help="Stable competition key override.")
    parser.add_argument("--season", help="Season label override.")
    parser.add_argument(
        "--output-root",
        default="frontend/public/data",
        help="Publish output root. Default: frontend/public/data",
    )
    return parser.parse_args()


def infer_competition_name(path: Path) -> str:
    stem = path.stem.replace("-", " ").replace("_", " ")
    stem = stem.replace("stats", "").replace("teams", "").strip()
    parts = [part for part in stem.split() if not part.isdigit()]
    if not parts:
        return "Unknown Competition"
    return " ".join(word.upper() if len(word) <= 3 else word.title() for word in parts[:4])


def load_entries(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.config:
        payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
        return list(payload.get("teams", []))
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


def prepare_team_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = clean_columns(df)

    matches = series_or_default(df, "matches_played", 0).replace(0, np.nan)
    matches_home = series_or_default(df, "matches_played_home", 0).replace(0, np.nan)
    matches_away = series_or_default(df, "matches_played_away", 0).replace(0, np.nan)

    team_name = df.get("common_name", df.get("team_name", pd.Series("Unknown", index=df.index)))
    if "team_name" in df.columns:
        team_name = team_name.fillna(df["team_name"])

    derived = {
        "team": team_name.fillna("Unknown"),
        "xg_diff": series_or_default(df, "xg_for_avg_overall") - series_or_default(df, "xg_against_avg_overall"),
        "xg_total": series_or_default(df, "xg_for_avg_overall") + series_or_default(df, "xg_against_avg_overall"),
        "goal_difference_per_match": safe_divide(series_or_default(df, "goal_difference"), matches),
        "shots_per_match": safe_divide(series_or_default(df, "shots"), matches),
        "shots_on_target_per_match": safe_divide(series_or_default(df, "shots_on_target"), matches),
        "shots_off_target_per_match": safe_divide(series_or_default(df, "shots_off_target"), matches),
        "fouls_per_match": safe_divide(series_or_default(df, "fouls"), matches),
        "goals_scored_half_time_per_match": safe_divide(series_or_default(df, "goals_scored_half_time"), matches),
        "goals_conceded_half_time_per_match": safe_divide(series_or_default(df, "goals_conceded_half_time"), matches),
        "wins_home_pct": safe_divide(series_or_default(df, "wins_home"), matches_home),
        "wins_away_pct": safe_divide(series_or_default(df, "wins_away"), matches_away),
        "home_away_balance": 1.0
        / (1.0 + (series_or_default(df, "points_per_game_home") - series_or_default(df, "points_per_game_away")).abs()),
    }
    derived["team_slug"] = derived["team"].map(slugify)
    derived["shot_dominance"] = derived["shots_on_target_per_match"] - derived["shots_off_target_per_match"]
    derived["late_scoring"] = (
        series_or_default(df, "goals_scored_min_71_to_80", 0) + series_or_default(df, "goals_scored_min_81_to_90", 0)
    )
    derived["late_conceding"] = (
        series_or_default(df, "goals_conceded_min_71_to_80", 0) + series_or_default(df, "goals_conceded_min_81_to_90", 0)
    )
    derived["early_scoring"] = (
        series_or_default(df, "goals_scored_min_0_to_10", 0) + series_or_default(df, "goals_scored_min_11_to_20", 0)
    )
    derived["late_goal_activity"] = derived["late_scoring"] + derived["late_conceding"]
    derived["match_instability"] = (
        series_or_default(df, "draw_percentage_overall")
        + series_or_default(df, "loss_percentage_ovearll")
        + (series_or_default(df, "prediction_risk") * 100.0)
    )
    return df.assign(**derived).copy()


def build_team_scores(df: pd.DataFrame) -> pd.DataFrame:
    score_frame = df.copy()
    confidence = team_confidence_multiplier(series_or_default(df, "matches_played", 0))

    raw_scores = {
        "og_power_rating": weighted_average(
            score_frame,
            [
                ("points_per_game", True, 1.4),
                ("league_position", False, 1.0),
                ("goal_difference_per_match", True, 1.1),
                ("xg_diff", True, 1.2),
                ("goals_scored_per_match", True, 0.8),
                ("goals_conceded_per_match", False, 0.9),
                ("shot_dominance", True, 0.8),
                ("clean_sheet_percentage", True, 0.8),
                ("first_team_to_score_percentage", True, 0.7),
                ("home_away_balance", True, 0.6),
            ],
        ),
        "attack_flow_rating": weighted_average(
            score_frame,
            [
                ("goals_scored_per_match", True, 1.2),
                ("xg_for_avg_overall", True, 1.1),
                ("shots_per_match", True, 0.9),
                ("shots_on_target_per_match", True, 1.0),
                ("first_team_to_score_percentage", True, 0.9),
                ("goals_scored_half_time_per_match", True, 0.7),
                ("over15_percentage", True, 0.7),
                ("early_scoring", True, 0.6),
            ],
        ),
        "defensive_lock_rating": weighted_average(
            score_frame,
            [
                ("goals_conceded_per_match", False, 1.2),
                ("xg_against_avg_overall", False, 1.1),
                ("clean_sheet_percentage", True, 1.0),
                ("minutes_per_goal_conceded", True, 0.9),
                ("goals_conceded_half_time_per_match", False, 0.7),
                ("late_conceding", False, 0.8),
                ("btts_percentage", False, 0.8),
                ("clean_sheet_percentage_home", True, 0.5),
                ("clean_sheet_percentage_away", True, 0.5),
            ],
        ),
        "goal_heat_rating": weighted_average(
            score_frame,
            [
                ("average_total_goals_per_match", True, 1.1),
                ("over25_percentage", True, 1.1),
                ("over35_percentage", True, 0.9),
                ("btts_percentage", True, 0.9),
                ("xg_total", True, 1.0),
                ("goals_scored_per_match", True, 0.5),
                ("goals_conceded_per_match", True, 0.5),
                ("late_goal_activity", True, 0.6),
            ],
        ),
        "btts_pressure_rating": weighted_average(
            score_frame,
            [
                ("btts_percentage", True, 1.2),
                ("clean_sheet_percentage", False, 0.9),
                ("fts_percentage", False, 0.9),
                ("goals_scored_per_match", True, 0.8),
                ("goals_conceded_per_match", True, 0.8),
                ("xg_for_avg_overall", True, 0.7),
                ("xg_against_avg_overall", True, 0.7),
                ("btts_percentage_home", True, 0.5),
                ("btts_percentage_away", True, 0.5),
            ],
        ),
        "over25_heat_rating": weighted_average(
            score_frame,
            [
                ("over25_percentage", True, 1.2),
                ("over35_percentage", True, 0.9),
                ("average_total_goals_per_match", True, 1.0),
                ("xg_total", True, 1.0),
                ("late_goal_activity", True, 0.7),
                ("over05_half_time_percentage", True, 0.6),
            ],
        ),
        "control_rating": weighted_average(
            score_frame,
            [
                ("under25_percentage", True, 1.1),
                ("clean_sheet_percentage", True, 0.9),
                ("xg_against_avg_overall", False, 1.0),
                ("btts_percentage", False, 0.8),
                ("average_total_goals_per_match", False, 0.9),
                ("average_possession", True, 0.6),
                ("over05_half_time_percentage", False, 0.6),
            ],
        ),
        "first_strike_rating": weighted_average(
            score_frame,
            [
                ("first_team_to_score_percentage", True, 1.2),
                ("early_scoring", True, 0.9),
                ("leading_at_half_time_percentage", True, 1.0),
                ("goals_scored_half_time_per_match", True, 0.8),
                ("fts_percentage", False, 0.8),
            ],
        ),
        "corner_pressure_rating": weighted_average(
            score_frame,
            [
                ("corners_per_match", True, 1.2),
                ("over85_corners_percentage", True, 0.8),
                ("over95_corners_percentage", True, 0.8),
                ("over105_corners_percentage", True, 0.7),
                ("shots_per_match", True, 0.7),
                ("average_possession", True, 0.6),
                ("xg_for_avg_overall", True, 0.7),
            ],
        ),
        "card_heat_rating": weighted_average(
            score_frame,
            [
                ("cards_per_match", True, 1.2),
                ("fouls_per_match", True, 1.0),
                ("losses", True, 0.7),
                ("late_conceding", True, 0.7),
                ("prediction_risk", True, 0.9),
            ],
        ),
        "chaos_rating": weighted_average(
            score_frame,
            [
                ("btts_percentage", True, 0.9),
                ("over25_percentage", True, 1.0),
                ("clean_sheet_percentage", False, 0.9),
                ("goals_conceded_per_match", True, 0.8),
                ("cards_per_match", True, 0.7),
                ("fouls_per_match", True, 0.7),
                ("prediction_risk", True, 1.0),
                ("match_instability", True, 0.8),
                ("late_goal_activity", True, 0.7),
            ],
        ),
        "home_fortress_rating": weighted_average(
            score_frame,
            [
                ("points_per_game_home", True, 1.3),
                ("wins_home_pct", True, 1.0),
                ("goals_scored_per_match_home", True, 0.8),
                ("goals_conceded_per_match_home", False, 0.9),
                ("xg_for_avg_home", True, 0.8),
                ("xg_against_avg_home", False, 0.8),
                ("clean_sheet_percentage_home", True, 0.7),
                ("first_team_to_score_percentage_home", True, 0.7),
                ("league_position_home", False, 0.7),
            ],
        ),
        "away_threat_rating": weighted_average(
            score_frame,
            [
                ("points_per_game_away", True, 1.3),
                ("wins_away_pct", True, 1.0),
                ("goals_scored_per_match_away", True, 0.8),
                ("goals_conceded_per_match_away", False, 0.9),
                ("xg_for_avg_away", True, 0.8),
                ("xg_against_avg_away", False, 0.8),
                ("clean_sheet_percentage_away", True, 0.7),
                ("first_team_to_score_percentage_away", True, 0.7),
                ("league_position_away", False, 0.7),
            ],
        ),
    }

    for key, raw in raw_scores.items():
        score_frame[key] = score_to_int(blend_with_neutral(raw, confidence))

    return score_frame


def make_profile_tags(row: pd.Series) -> list[str]:
    tags: list[str] = []
    if row["og_power_rating"] >= 90:
        tags.append("Elite Control Side")
    elif row["og_power_rating"] >= 80:
        tags.append("Strong Table Profile")

    if row["defensive_lock_rating"] >= 85:
        tags.append("Defensive Wall")
    if row["attack_flow_rating"] >= 85:
        tags.append("Attack-Led Side")
    if row["first_strike_rating"] >= 80:
        tags.append("Strong First Goal")
    if row["home_fortress_rating"] >= 85:
        tags.append("Home Dominant")
    if row["away_threat_rating"] >= 80:
        tags.append("Travels Well")
    if row["corner_pressure_rating"] >= 80:
        tags.append("Corner Pressure Side")
    if row["goal_heat_rating"] >= 75:
        tags.append("Open Match Profile")
    if row["btts_pressure_rating"] >= 70:
        tags.append("Two-Way Scoring Profile")
    if row["chaos_rating"] <= 35:
        tags.append("Low Chaos")
    elif row["chaos_rating"] >= 75:
        tags.append("High Chaos Side")
    if row["card_heat_rating"] >= 75:
        tags.append("Card Heat Risk")
    return tags[:5] or ["Mixed Team Profile"]


def rating_explanation(name: str, score: int) -> str:
    label = rating_band(score)
    templates = {
        "og_power_rating": f"{label} team strength built from table output, xG shape, and repeatable match control.",
        "attack_flow_rating": f"{label} attacking pressure profile with shot production and first-goal threat support.",
        "defensive_lock_rating": f"{label} defensive resistance with concession control and clean-sheet support.",
        "goal_heat_rating": f"{label} goal-environment profile for open versus controlled match states.",
        "btts_pressure_rating": f"{label} two-way scoring pressure based on scoring access and defensive leakage.",
        "over25_heat_rating": f"{label} 3+ goal tendency profile driven by tempo, scoring rate, and goal timing.",
        "control_rating": f"{label} match-control profile that suppresses chaos and supports lower-event game states.",
        "first_strike_rating": f"{label} early control profile with strong first-goal and halftime leverage.",
        "corner_pressure_rating": f"{label} territory and corner-generation profile.",
        "card_heat_rating": f"{label} discipline and card volatility profile.",
        "chaos_rating": f"{label} volatility profile showing how difficult this side is to trust cleanly.",
        "home_fortress_rating": f"{label} home-ground control profile.",
        "away_threat_rating": f"{label} travelling threat profile away from home.",
    }
    return templates[name]


def build_team_payloads(df: pd.DataFrame, competition_name: str, competition_key: str, season: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    index_records: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        ratings = {field: int(row[field]) for field in TEAM_RATING_FIELDS}
        strengths = sorted(ratings.items(), key=lambda item: item[1], reverse=True)[:3]
        caution = max(
            [("Chaos Rating", ratings["chaos_rating"]), ("Card Heat", ratings["card_heat_rating"])],
            key=lambda item: item[1],
        )
        team_payload: dict[str, Any] = {
            "team": row["team"],
            "team_slug": row["team_slug"],
            "competition": competition_name,
            "competition_key": competition_key,
            "season": str(season),
            "country": row.get("country"),
            "matches_played": int(row.get("matches_played") or 0),
            "sample_confidence": {
                "label": team_confidence_label(float(row.get("matches_played") or 0)),
                "matches_played": int(row.get("matches_played") or 0),
            },
            "ratings": ratings,
            "rating_bands": {name: rating_band(score) for name, score in ratings.items()},
            "rating_explanations": {name: rating_explanation(name, score) for name, score in ratings.items()},
            "profile_tags": make_profile_tags(row),
            "summary": {
                "headline": f"OG Power Rating: {ratings['og_power_rating']}%",
                "profile": rating_explanation("og_power_rating", ratings["og_power_rating"]),
                "primary_strengths": [name.replace("_rating", "").replace("_", " ").title() for name, _ in strengths],
                "main_caution": f"{caution[0]} sits at {caution[1]}%, so this side needs context rather than blind trust.",
            },
            "market_tendencies": {
                "ftr_lean": market_lean(ratings["og_power_rating"]),
                "btts_lean": market_lean(ratings["btts_pressure_rating"]),
                "over25_lean": market_lean(ratings["over25_heat_rating"]),
                "under_control": market_lean(ratings["control_rating"]),
                "team_goals_15": market_lean(ratings["attack_flow_rating"]),
                "corners": market_lean(ratings["corner_pressure_rating"]),
                "cards": market_lean(100 - ratings["card_heat_rating"]),
            },
            "home_profile": {
                "power": ratings["home_fortress_rating"],
                "attack": int(row["attack_flow_rating"]),
                "defence": int(row["defensive_lock_rating"]),
                "goal_heat": int(row["goal_heat_rating"]),
            },
            "away_profile": {
                "power": ratings["away_threat_rating"],
                "attack": int(round((row["attack_flow_rating"] + row.get("away_threat_rating", 50)) / 2)),
                "defence": int(round((row["defensive_lock_rating"] + (100 - row.get("chaos_rating", 50))) / 2)),
                "goal_heat": int(round((row["goal_heat_rating"] + row["over25_heat_rating"]) / 2)),
            },
            "timing_profile": {
                "early_threat": int(score_to_int((row["first_strike_rating"] + row["attack_flow_rating"]) / 2)),
                "half_time_control": int(score_to_int((series_or_default(pd.DataFrame([row]), "leading_at_half_time_percentage").iloc[0] + row["control_rating"]) / 2)),
                "late_surge": int(score_to_int((row["goal_heat_rating"] + row["over25_heat_rating"]) / 2)),
                "late_fragility": int(score_to_int((row["chaos_rating"] + row["card_heat_rating"]) / 2)),
            },
        }
        records.append(team_payload)
        index_records.append(
            {
                "team": row["team"],
                "team_slug": row["team_slug"],
                "competition": competition_name,
                "competition_key": competition_key,
                "season": str(season),
                "headline_rating": ratings["og_power_rating"],
                "headline_band": rating_band(ratings["og_power_rating"]),
                "profile_tags": team_payload["profile_tags"],
            }
        )

    return records, index_records


def export_team_payloads(
    records: list[dict[str, Any]],
    index_records: list[dict[str, Any]],
    competition_key: str,
    season: str,
    output_root: Path,
) -> None:
    base = output_root / "team_intelligence"
    team_dir = base / "teams" / competition_key / str(season)
    competition_path = base / "competitions" / f"{competition_key}__{season}.json"
    csv_path = base / "team_ratings.csv"
    index_path = base / "team_ratings_index.json"

    ensure_dir(team_dir)
    ensure_dir(competition_path.parent)

    for payload in records:
        write_json(team_dir / f"{payload['team_slug']}.json", payload)

    write_json(competition_path, {"competition_key": competition_key, "season": str(season), "teams": records})

    existing_index: list[dict[str, Any]] = []
    if index_path.exists():
        existing_index = json.loads(index_path.read_text(encoding="utf-8"))
        existing_index = [
            row for row in existing_index if not (row.get("competition_key") == competition_key and str(row.get("season")) == str(season))
        ]
    write_json(index_path, sorted(existing_index + index_records, key=lambda row: (row["competition_key"], row["season"], row["team"])))

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

    flat_rows = []
    for payload in records:
        row = {
            "team": payload["team"],
            "team_slug": payload["team_slug"],
            "competition": payload["competition"],
            "competition_key": payload["competition_key"],
            "season": payload["season"],
            "matches_played": payload["sample_confidence"]["matches_played"],
        }
        row.update(payload["ratings"])
        flat_rows.append(row)
    fresh_rows.append(pd.DataFrame(flat_rows))
    pd.concat(fresh_rows, ignore_index=True).to_csv(csv_path, index=False)


def process_entry(entry: dict[str, Any], output_root: Path) -> None:
    source = Path(entry["source"])
    df = prepare_team_frame(source)
    competition_name = entry.get("competition_name") or infer_competition_name(source)
    competition_key = entry.get("competition_key") or slugify(competition_name)
    season = str(entry.get("season") or df.get("season", pd.Series(["unknown"])).iloc[0])
    scored = build_team_scores(df)
    records, index_records = build_team_payloads(scored, competition_name, competition_key, season)
    export_team_payloads(records, index_records, competition_key, season, output_root)
    print(f"Built {len(records)} team intelligence profiles for {competition_name} {season}")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    entries = load_entries(args)
    for entry in entries:
        process_entry(entry, output_root)


if __name__ == "__main__":
    main()
