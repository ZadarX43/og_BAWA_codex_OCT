from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd


def _safe_div(num: float, den: float) -> float:
    if not den:
        return 0.0
    return float(num) / float(den)


def _rolling_mean(records: list[dict], key: str, n: int) -> float:
    sample = records[:n]
    if not sample:
        return 0.0
    return sum(float(r.get(key, 0.0) or 0.0) for r in sample) / len(sample)


def _rolling_sum(records: list[dict], key: str, n: int) -> float:
    sample = records[:n]
    return sum(float(r.get(key, 0.0) or 0.0) for r in sample) if sample else 0.0


def _strictness_score(avg_cards: float, foul_to_card_ratio: float, late_cards: float) -> float:
    cards_component = min(1.0, avg_cards / 6.0)
    ratio_component = min(1.0, _safe_div(8.0, max(foul_to_card_ratio, 1.0)))
    late_component = min(1.0, late_cards / 1.5)
    return round((0.50 * cards_component) + (0.35 * ratio_component) + (0.15 * late_component), 4)


def build_referee_profiles(
    fixtures_csv: str,
    team_stats_csv: str,
    events_csv: str,
    output_csv: str,
    history_window: int = 20,
) -> pd.DataFrame:
    fixtures = pd.read_csv(fixtures_csv)
    team_stats = pd.read_csv(team_stats_csv)
    events = pd.read_csv(events_csv)

    fixtures["kickoff_ts_utc"] = pd.to_datetime(fixtures["kickoff_ts_utc"], errors="coerce", utc=True)

    fixture_totals = (
        team_stats.groupby("fixture_id", as_index=False)
        .agg(
            total_fouls=("fouls_for", "sum"),
            total_yellows=("yellow_cards", "sum"),
            total_reds=("red_cards", "sum"),
        )
    )
    fixture_totals["total_cards"] = fixture_totals["total_yellows"] + fixture_totals["total_reds"]

    card_events = events[events["event_type"].astype("string").str.lower().eq("card")].copy()
    card_events["minute"] = pd.to_numeric(card_events["minute"], errors="coerce").fillna(0.0)
    card_events["late_card_flag"] = (card_events["minute"] >= 75).astype(int)
    event_totals = (
        card_events.groupby("fixture_id", as_index=False)
        .agg(
            raw_card_events=("event_id", "count"),
            late_card_events=("late_card_flag", "sum"),
        )
    )

    merged = fixtures.merge(fixture_totals, on="fixture_id", how="left").merge(event_totals, on="fixture_id", how="left")
    for col in ["total_fouls", "total_yellows", "total_reds", "total_cards", "raw_card_events", "late_card_events"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    merged = merged.sort_values(["kickoff_ts_utc", "fixture_id"]).reset_index(drop=True)

    history: dict[str, list[dict]] = defaultdict(list)
    out_rows: list[dict] = []

    for _, row in merged.iterrows():
        referee_name = str(row.get("referee_name", "") or "").strip()
        prev = list(reversed(history.get(referee_name, []))) if referee_name else []

        avg_cards = _rolling_mean(prev, "total_cards", history_window)
        avg_fouls = _rolling_mean(prev, "total_fouls", history_window)
        avg_late_cards = _rolling_mean(prev, "late_card_events", history_window)
        foul_to_card_ratio = _safe_div(_rolling_sum(prev, "total_fouls", history_window), _rolling_sum(prev, "total_cards", history_window))
        strictness = _strictness_score(avg_cards, foul_to_card_ratio, avg_late_cards) if prev else 0.0

        out_rows.append(
            {
                "fixture_id": int(row["fixture_id"]),
                "fixture_key": row.get("fixture_key", ""),
                "league": row.get("league", ""),
                "league_id": row.get("league_id", ""),
                "season": row.get("season", ""),
                "match_date": row.get("match_date", ""),
                "home_team_id": row.get("home_team_id", ""),
                "away_team_id": row.get("away_team_id", ""),
                "home_team_name": row.get("home_team_name", ""),
                "away_team_name": row.get("away_team_name", ""),
                "venue": row.get("venue_name", ""),
                "referee_name": referee_name,
                "ref_matches_sample_l20": min(len(prev), history_window),
                "ref_cards_per_match": round(avg_cards, 4),
                "ref_fouls_per_match": round(avg_fouls, 4),
                "ref_foul_to_card_ratio": round(foul_to_card_ratio, 4),
                "ref_late_cards_per_match": round(avg_late_cards, 4),
                "ref_dissent_strictness": round(strictness, 4),
                "ref_timewasting_strictness": round(min(1.0, avg_late_cards / 1.5), 4),
                "ref_strictness_score": strictness,
            }
        )

        if referee_name:
            history[referee_name].append(
                {
                    "total_cards": float(row.get("total_cards", 0.0) or 0.0),
                    "total_fouls": float(row.get("total_fouls", 0.0) or 0.0),
                    "late_card_events": float(row.get("late_card_events", 0.0) or 0.0),
                }
            )

    out = pd.DataFrame(out_rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def _default_path(league_tag: str, season: int) -> Path:
    return Path("data_sources/api_football/features/player_events") / f"referee_profiles__{league_tag}__{season}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rolling referee profiles for player-events research.")
    parser.add_argument("--league-tag", required=True, help="League tag like Italy_Serie_A")
    parser.add_argument("--season", type=int, required=True, help="Season integer, e.g. 2024")
    parser.add_argument("--fixtures-csv", default="", help="Override fixtures csv path")
    parser.add_argument("--team-stats-csv", default="", help="Override team stats csv path")
    parser.add_argument("--events-csv", default="", help="Override events csv path")
    parser.add_argument("--output-csv", default="", help="Override output csv path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    normalized = Path("data_sources/api_football/normalized")
    fixtures_csv = args.fixtures_csv or str(normalized / f"fixtures_master__{args.league_tag}__{args.season}.csv")
    team_stats_csv = args.team_stats_csv or str(normalized / f"match_team_stats__{args.league_tag}__{args.season}.csv")
    events_csv = args.events_csv or str(normalized / f"match_events__{args.league_tag}__{args.season}.csv")
    output_csv = args.output_csv or str(_default_path(args.league_tag, args.season))

    df = build_referee_profiles(fixtures_csv, team_stats_csv, events_csv, output_csv)
    print(f"WROTE: {output_csv}")
    print(f"rows: {len(df)} | referees: {df['referee_name'].nunique(dropna=True)}")


if __name__ == "__main__":
    main()
