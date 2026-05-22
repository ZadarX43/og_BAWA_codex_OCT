from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import NORMALIZED_DIR, REPORTS_DIR, TARGET_LEAGUES, TARGET_SEASONS, normalized_path

DEFAULT_START_MINUTES = {"D": 82.0, "M": 78.0, "F": 74.0, "G": 90.0}
DEFAULT_SUB_MINUTES = {"D": 24.0, "M": 26.0, "F": 28.0, "G": 5.0}
RECENCY_WEIGHTS = [5, 4, 3, 2, 1]


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    if not values:
        return 0.0
    w = weights[: len(values)]
    total_w = sum(w)
    return sum(v * wt for v, wt in zip(values, w)) / total_w if total_w else 0.0


def build_for_league_season(league: str, season: int) -> pd.DataFrame:
    fixtures = pd.read_csv(normalized_path("fixtures_master", league, season), low_memory=False)
    player_stats = pd.read_csv(normalized_path("match_player_stats", league, season), low_memory=False)
    lineups = pd.read_csv(normalized_path("lineups", league, season), low_memory=False)

    fixtures["kickoff_ts_utc"] = pd.to_datetime(fixtures["kickoff_ts_utc"], errors="coerce", utc=True)
    fixture_lookup = fixtures[["fixture_id", "fixture_key", "match_date", "kickoff_ts_utc", "home_team_id", "away_team_id", "home_team_name", "away_team_name"]]

    stats = player_stats.merge(fixture_lookup, on="fixture_id", how="left")
    stats["kickoff_ts_utc"] = pd.to_datetime(stats["kickoff_ts_utc"], errors="coerce", utc=True)
    stats["minutes"] = pd.to_numeric(stats["minutes"], errors="coerce").fillna(0.0)
    stats["started_flag"] = pd.to_numeric(stats["started_flag"], errors="coerce").fillna(0).astype(int)
    stats["subbed_on_flag"] = pd.to_numeric(stats["subbed_on_flag"], errors="coerce").fillna(0).astype(int)
    stats = stats.sort_values(["kickoff_ts_utc", "fixture_id", "team_id", "player_id"]).reset_index(drop=True)

    lineups = lineups.merge(fixture_lookup, on="fixture_id", how="left")
    lineups["kickoff_ts_utc"] = pd.to_datetime(lineups["kickoff_ts_utc"], errors="coerce", utc=True)
    lineups["is_starting_xi"] = pd.to_numeric(lineups["is_starting_xi"], errors="coerce").fillna(0).astype(int)
    lineups = lineups.sort_values(["kickoff_ts_utc", "fixture_id", "team_id", "player_id"]).reset_index(drop=True)

    history: dict[int, list[dict]] = {}
    rows: list[dict] = []
    for _, row in stats.iterrows():
        player_id = int(row["player_id"])
        prev = list(reversed(history.get(player_id, [])))
        started_vals = [float(r.get("started_flag", 0.0)) for r in prev[:5]]
        start_prob = _weighted_mean(started_vals, RECENCY_WEIGHTS)

        pos = str(row.get("position", "")).strip().upper()[:1]
        start_minutes = [float(r.get("minutes", 0.0)) for r in prev if int(r.get("started_flag", 0)) == 1][:5]
        sub_minutes = [float(r.get("minutes", 0.0)) for r in prev if int(r.get("started_flag", 0)) == 0 and int(r.get("subbed_on_flag", 0)) == 1][:5]
        mins_if_start = _weighted_mean(start_minutes, RECENCY_WEIGHTS) if start_minutes else DEFAULT_START_MINUTES.get(pos, 75.0)
        mins_if_sub = _weighted_mean(sub_minutes, RECENCY_WEIGHTS) if sub_minutes else DEFAULT_SUB_MINUTES.get(pos, 25.0)
        apps_l5 = min(len(prev), 5)
        expected_minutes = max(0.0, min(90.0, start_prob * mins_if_start + (1.0 - start_prob) * mins_if_sub))
        source_max_date = max([str(r.get("match_date", "")) for r in prev[:5]], default="")
        rows.append(
            {
                "fixture_id": int(row["fixture_id"]),
                "fixture_key": row["fixture_key"],
                "match_date": row["match_date"],
                "team_id": int(row["team_id"]),
                "player_id": player_id,
                "player_name": row["player_name"],
                "position": row.get("position", ""),
                "expected_start_prob": round(float(start_prob), 4),
                "expected_minutes_if_start": round(float(mins_if_start), 4),
                "expected_minutes_if_sub": round(float(mins_if_sub), 4),
                "expected_minutes_proof": round(float(expected_minutes), 4),
                "minutes_history_apps_l5": int(apps_l5),
                "minutes_source_max_date": source_max_date,
                "actual_minutes": round(float(row.get("minutes", 0.0) or 0.0), 4),
                "actual_started_flag": int(row.get("started_flag", 0) or 0),
                "league_tag": league,
                "season_tag": season,
            }
        )
        history.setdefault(player_id, []).append(
            {
                "match_date": row["match_date"],
                "minutes": float(row.get("minutes", 0.0) or 0.0),
                "started_flag": int(row.get("started_flag", 0) or 0),
                "subbed_on_flag": int(row.get("subbed_on_flag", 0) or 0),
            }
        )
    return pd.DataFrame(rows)


def build(output_csv: Path, output_md: Path, leagues: tuple[str, ...] = TARGET_LEAGUES, seasons: tuple[int, ...] = TARGET_SEASONS) -> pd.DataFrame:
    frames = [build_for_league_season(league, season) for league in leagues for season in seasons]
    out = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    lines = [
        "# Minutes Offset Build",
        "",
        "- Proof-only expected minutes estimate used as the count-model offset.",
        "- Formula: `P(start) * E[minutes|start] + (1 - P(start)) * E[minutes|sub]` from the last five appearances with recency weighting.",
        "",
    ]
    if out.empty:
        lines.append("- No rows built.")
    else:
        mae = (pd.to_numeric(out["expected_minutes_proof"], errors="coerce") - pd.to_numeric(out["actual_minutes"], errors="coerce")).abs().mean()
        lines.extend([
            f"- rows={len(out)}",
            f"- minutes MAE vs realized minutes: `{float(mae):.3f}`",
            "",
            "## Features",
            "- `expected_start_prob`",
            "- `expected_minutes_if_start`",
            "- `expected_minutes_if_sub`",
            "- `expected_minutes_proof`",
            "- `minutes_history_apps_l5`",
            "- `minutes_source_max_date`",
        ])
    output_md.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build proof-only expected minutes offsets for tackles modeling.")
    parser.add_argument("--output-csv", default=str(REPORTS_DIR / "minutes_offset.csv"))
    parser.add_argument("--output-md", default=str(REPORTS_DIR / "minutes_offset.md"))
    args = parser.parse_args()
    out = build(Path(args.output_csv), Path(args.output_md))
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")


if __name__ == "__main__":
    main()
