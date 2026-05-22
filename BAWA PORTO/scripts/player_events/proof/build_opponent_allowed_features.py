from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from common import NORMALIZED_DIR, REPORTS_DIR, TARGET_LEAGUES, TARGET_SEASONS, normalized_path

POSITION_MAP = {"D": "DEF", "M": "MID", "F": "FWD", "G": "GK"}


def _mean(records: list[dict], key: str, n: int) -> float:
    sample = records[:n]
    if not sample:
        return 0.0
    return sum(float(r.get(key, 0.0) or 0.0) for r in sample) / len(sample)


def build_for_league_season(league: str, season: int) -> pd.DataFrame:
    fixtures = pd.read_csv(normalized_path("fixtures_master", league, season), low_memory=False)
    player_stats = pd.read_csv(normalized_path("match_player_stats", league, season), low_memory=False)
    team_stats = pd.read_csv(normalized_path("match_team_stats", league, season), low_memory=False)

    fixtures["kickoff_ts_utc"] = pd.to_datetime(fixtures["kickoff_ts_utc"], errors="coerce", utc=True)
    fixtures = fixtures.sort_values(["kickoff_ts_utc", "fixture_id"]).reset_index(drop=True)

    players = player_stats.copy()
    players["position_group"] = players["position"].astype(str).str.upper().map(POSITION_MAP).fillna("OTHER")
    players["minutes"] = pd.to_numeric(players["minutes"], errors="coerce").fillna(0.0)
    players["tackles"] = pd.to_numeric(players["tackles"], errors="coerce").fillna(0.0)
    players["dribbles_attempted"] = pd.to_numeric(players["dribbles_attempted"], errors="coerce").fillna(0.0)

    grouped = (
        players.groupby(["fixture_id", "team_id", "position_group"], as_index=False)
        .agg(minutes=("minutes", "sum"), tackles=("tackles", "sum"), dribbles_attempted=("dribbles_attempted", "sum"))
    )
    grouped["tackles_per90"] = grouped.apply(
        lambda r: (float(r["tackles"]) * 90.0 / float(r["minutes"])) if float(r["minutes"]) > 0 else 0.0,
        axis=1,
    )

    team_stats = team_stats.copy()
    team_stats["possession_pct"] = pd.to_numeric(team_stats["possession_pct"], errors="coerce").fillna(50.0)
    team_stats = team_stats.merge(
        fixtures[["fixture_id", "fixture_key", "match_date", "kickoff_ts_utc", "home_team_id", "away_team_id"]],
        on="fixture_id",
        how="left",
    )

    team_history: dict[int, list[dict]] = defaultdict(list)
    rows: list[dict] = []

    for fixture in fixtures.itertuples(index=False):
        fixture_stats = team_stats[team_stats["fixture_id"].eq(fixture.fixture_id)].copy()
        if len(fixture_stats) != 2:
            continue
        for _, team_row in fixture_stats.iterrows():
            team_id = int(team_row["team_id"])
            opp_team_id = int(fixture.away_team_id if team_id == int(fixture.home_team_id) else fixture.home_team_id)
            opp_history = list(reversed(team_history.get(opp_team_id, [])))
            rows.append(
                {
                    "fixture_id": int(fixture.fixture_id),
                    "fixture_key": fixture.fixture_key,
                    "match_date": fixture.match_date,
                    "team_id": team_id,
                    "opponent_team_id": opp_team_id,
                    "league_tag": league,
                    "season_tag": season,
                    "opp_tackles_allowed_def_l10": round(_mean(opp_history, "allowed_def_tackles_per90", 10), 4),
                    "opp_tackles_allowed_mid_l10": round(_mean(opp_history, "allowed_mid_tackles_per90", 10), 4),
                    "opp_possession_share_l10": round(_mean(opp_history, "conceded_possession_pct", 10), 4),
                    "opp_dribble_attempts_l10": round(_mean(opp_history, "allowed_dribbles_attempted", 10), 4),
                    "opp_allowed_history_matches_l10": int(min(len(opp_history), 10)),
                    "opp_allowed_source_max_date": max([r.get("match_date") for r in opp_history[:10]], default=""),
                }
            )

        for _, team_row in fixture_stats.iterrows():
            team_id = int(team_row["team_id"])
            opp_team_id = int(fixture.away_team_id if team_id == int(fixture.home_team_id) else fixture.home_team_id)
            opp_players = grouped[(grouped["fixture_id"].eq(fixture.fixture_id)) & (grouped["team_id"].eq(opp_team_id))]
            def_tackles = float(opp_players.loc[opp_players["position_group"].eq("DEF"), "tackles_per90"].sum())
            mid_tackles = float(opp_players.loc[opp_players["position_group"].eq("MID"), "tackles_per90"].sum())
            allowed_dribbles = float(opp_players["dribbles_attempted"].sum())
            conceded_possession = 100.0 - float(team_row.get("possession_pct", 50.0) or 50.0)
            team_history[team_id].append(
                {
                    "match_date": fixture.match_date,
                    "allowed_def_tackles_per90": def_tackles,
                    "allowed_mid_tackles_per90": mid_tackles,
                    "allowed_dribbles_attempted": allowed_dribbles,
                    "conceded_possession_pct": conceded_possession,
                }
            )

    return pd.DataFrame(rows)


def build(output_csv: Path, output_md: Path, leagues: tuple[str, ...] = TARGET_LEAGUES, seasons: tuple[int, ...] = TARGET_SEASONS) -> pd.DataFrame:
    frames = [build_for_league_season(league, season) for league in leagues for season in seasons]
    out = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    lines = [
        "# Opponent Allowed Features",
        "",
        "- Proof-only feature build for the tackles rate model.",
        "- Measures what the opponent has recently allowed to opposition defenders and midfielders, plus possession conceded and dribble volume invited.",
        "",
    ]
    if out.empty:
        lines.append("- No rows built.")
    else:
        by_league = out.groupby("league_tag", dropna=False).agg(rows=("fixture_id", "size"), fixtures=("fixture_key", pd.Series.nunique)).reset_index()
        lines.append("## Coverage")
        for _, row in by_league.iterrows():
            lines.append(f"- {row['league_tag']} | rows={int(row['rows'])} | fixtures={int(row['fixtures'])}")
        lines.extend([
            "",
            "## Features",
            "- `opp_tackles_allowed_def_l10`",
            "- `opp_tackles_allowed_mid_l10`",
            "- `opp_possession_share_l10`",
            "- `opp_dribble_attempts_l10`",
            "- `opp_allowed_history_matches_l10`",
            "- `opp_allowed_source_max_date`",
        ])
    output_md.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build proof-only opponent-allowed features for the tackles rate model.")
    parser.add_argument("--output-csv", default=str(REPORTS_DIR / "opponent_allowed_features.csv"))
    parser.add_argument("--output-md", default=str(REPORTS_DIR / "opponent_allowed_features.md"))
    args = parser.parse_args()
    out = build(Path(args.output_csv), Path(args.output_md))
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")


if __name__ == "__main__":
    main()
