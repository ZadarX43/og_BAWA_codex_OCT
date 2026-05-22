from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd


def _safe_div(num: float, den: float) -> float:
    if not den:
        return 0.0
    return float(num) / float(den)


def _sum(records: list[dict], key: str, n: int) -> float:
    sample = records[:n]
    return sum(float(r.get(key, 0.0) or 0.0) for r in sample) if sample else 0.0


def _mean(records: list[dict], key: str, n: int) -> float:
    sample = records[:n]
    return _safe_div(_sum(sample, key, n), len(sample)) if sample else 0.0


def _norm_cap(series: pd.Series, cap: float) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").astype(float)
    return (out.clip(lower=0.0, upper=cap) / cap).fillna(0.0)


def _pair_key(home_team_id: int, away_team_id: int) -> tuple[int, int]:
    return tuple(sorted((int(home_team_id), int(away_team_id))))


def _style_label(
    foul_density_score: float,
    tackle_density_score: float,
    home_fouls_l5: float,
    away_fouls_l5: float,
    central_battle_score: float,
    wide_duel_score: float,
) -> str:
    if foul_density_score >= 0.68 and tackle_density_score >= 0.68:
        return "AGGRESSIVE_BOTH"
    if central_battle_score >= 0.66:
        return "MIDFIELD_GRIND"
    if wide_duel_score >= 0.66:
        return "WIDE_DUEL_GAME"
    if abs(home_fouls_l5 - away_fouls_l5) >= 3.0:
        return "ONE_SIDED_PRESSURE"
    return "BALANCED_CONTACT"


def _attacking_style_label(
    attack_pressure_score: float,
    corner_pressure_score: float,
    territorial_stress_score: float,
) -> str:
    if attack_pressure_score >= 0.70 and territorial_stress_score >= 0.62:
        return "ATTACK_WAVE"
    if corner_pressure_score >= 0.70:
        return "CORNER_SIEGE"
    if territorial_stress_score >= 0.68:
        return "TERRITORY_TILT"
    return "BALANCED_ATTACK"


def build_fixture_style_overlay(
    fixtures_csv: str,
    team_stats_csv: str,
    player_stats_csv: str,
    output_csv: str,
) -> pd.DataFrame:
    fixtures = pd.read_csv(fixtures_csv)
    team_stats = pd.read_csv(team_stats_csv)
    player_stats = pd.read_csv(player_stats_csv)

    fixtures["kickoff_ts_utc"] = pd.to_datetime(fixtures["kickoff_ts_utc"], errors="coerce", utc=True)

    team_tackles = (
        player_stats.groupby(["fixture_id", "team_id"], as_index=False)
        .agg(
            team_tackles=("tackles", "sum"),
            team_interceptions=("interceptions", "sum"),
            team_fouls_players=("fouls_committed", "sum"),
            team_dribbled_past=("dribbled_past", "sum"),
        )
    )

    team_frame = team_stats.merge(team_tackles, on=["fixture_id", "team_id"], how="left")
    for c in ["team_tackles", "team_interceptions", "team_fouls_players", "team_dribbled_past"]:
        team_frame[c] = pd.to_numeric(team_frame[c], errors="coerce").fillna(0.0)
    for c in ["shots_total", "shots_on_goal", "corners_for", "possession_pct", "passes_total"]:
        if c in team_frame.columns:
            team_frame[c] = pd.to_numeric(team_frame[c], errors="coerce").fillna(0.0)
    team_frame = team_frame.merge(
        fixtures[
            [
                "fixture_id",
                "fixture_key",
                "league",
                "season",
                "match_date",
                "kickoff_ts_utc",
                "home_team_id",
                "away_team_id",
                "home_team_name",
                "away_team_name",
            ]
        ],
        on="fixture_id",
        how="left",
    )
    team_frame = team_frame.sort_values(["kickoff_ts_utc", "fixture_id", "team_id"]).reset_index(drop=True)

    team_history: dict[int, list[dict]] = defaultdict(list)
    fixture_rows: list[dict] = []

    for fixture_id, group in team_frame.groupby("fixture_id", sort=False):
        group = group.sort_values("is_home", ascending=False).reset_index(drop=True)
        if len(group) != 2:
            continue
        home = group[group["is_home"].eq(1)].iloc[0]
        away = group[group["is_home"].eq(0)].iloc[0]

        home_prev = list(reversed(team_history.get(int(home["team_id"]), [])))
        away_prev = list(reversed(team_history.get(int(away["team_id"]), [])))

        home_fouls_l5 = _mean(home_prev, "fouls_for", 5)
        home_fouls_l10 = _mean(home_prev, "fouls_for", 10)
        away_fouls_l5 = _mean(away_prev, "fouls_for", 5)
        away_fouls_l10 = _mean(away_prev, "fouls_for", 10)
        home_tackles_l5 = _mean(home_prev, "team_tackles", 5)
        home_tackles_l10 = _mean(home_prev, "team_tackles", 10)
        away_tackles_l5 = _mean(away_prev, "team_tackles", 5)
        away_tackles_l10 = _mean(away_prev, "team_tackles", 10)
        home_interceptions_l5 = _mean(home_prev, "team_interceptions", 5)
        away_interceptions_l5 = _mean(away_prev, "team_interceptions", 5)
        home_dribbled_past_l5 = _mean(home_prev, "team_dribbled_past", 5)
        away_dribbled_past_l5 = _mean(away_prev, "team_dribbled_past", 5)
        home_shots_l5 = _mean(home_prev, "shots_total", 5)
        home_shots_l10 = _mean(home_prev, "shots_total", 10)
        away_shots_l5 = _mean(away_prev, "shots_total", 5)
        away_shots_l10 = _mean(away_prev, "shots_total", 10)
        home_sog_l5 = _mean(home_prev, "shots_on_goal", 5)
        home_sog_l10 = _mean(home_prev, "shots_on_goal", 10)
        away_sog_l5 = _mean(away_prev, "shots_on_goal", 5)
        away_sog_l10 = _mean(away_prev, "shots_on_goal", 10)
        home_corners_for_l5 = _mean(home_prev, "corners_for", 5)
        home_corners_for_l10 = _mean(home_prev, "corners_for", 10)
        away_corners_for_l5 = _mean(away_prev, "corners_for", 5)
        away_corners_for_l10 = _mean(away_prev, "corners_for", 10)
        home_possession_l5 = _mean(home_prev, "possession_pct", 5)
        away_possession_l5 = _mean(away_prev, "possession_pct", 5)
        home_passes_l5 = _mean(home_prev, "passes_total", 5)
        away_passes_l5 = _mean(away_prev, "passes_total", 5)

        fixture_rows.append(
            {
                "fixture_id": int(fixture_id),
                "fixture_key": home["fixture_key"],
                "league": home["league"],
                "season": int(home["season"]),
                "match_date": home["match_date"],
                "home_team_name": home["home_team_name"],
                "away_team_name": home["away_team_name"],
                "home_team_fouls_l5": round(home_fouls_l5, 4),
                "home_team_fouls_l10": round(home_fouls_l10, 4),
                "away_team_fouls_l5": round(away_fouls_l5, 4),
                "away_team_fouls_l10": round(away_fouls_l10, 4),
                "home_team_tackles_l5": round(home_tackles_l5, 4),
                "home_team_tackles_l10": round(home_tackles_l10, 4),
                "away_team_tackles_l5": round(away_tackles_l5, 4),
                "away_team_tackles_l10": round(away_tackles_l10, 4),
                "home_team_interceptions_l5": round(home_interceptions_l5, 4),
                "away_team_interceptions_l5": round(away_interceptions_l5, 4),
                "home_team_dribbled_past_l5": round(home_dribbled_past_l5, 4),
                "away_team_dribbled_past_l5": round(away_dribbled_past_l5, 4),
                "home_team_shots_l5": round(home_shots_l5, 4),
                "home_team_shots_l10": round(home_shots_l10, 4),
                "away_team_shots_l5": round(away_shots_l5, 4),
                "away_team_shots_l10": round(away_shots_l10, 4),
                "home_team_shots_on_goal_l5": round(home_sog_l5, 4),
                "home_team_shots_on_goal_l10": round(home_sog_l10, 4),
                "away_team_shots_on_goal_l5": round(away_sog_l5, 4),
                "away_team_shots_on_goal_l10": round(away_sog_l10, 4),
                "home_team_corners_for_l5": round(home_corners_for_l5, 4),
                "home_team_corners_for_l10": round(home_corners_for_l10, 4),
                "away_team_corners_for_l5": round(away_corners_for_l5, 4),
                "away_team_corners_for_l10": round(away_corners_for_l10, 4),
                "home_team_possession_l5": round(home_possession_l5, 4),
                "away_team_possession_l5": round(away_possession_l5, 4),
                "home_team_passes_l5": round(home_passes_l5, 4),
                "away_team_passes_l5": round(away_passes_l5, 4),
            }
        )

        team_history[int(home["team_id"])].append(home.to_dict())
        team_history[int(away["team_id"])].append(away.to_dict())

    out = pd.DataFrame(fixture_rows)
    if out.empty:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False)
        return out

    pair_history: dict[tuple[int, int], list[dict]] = defaultdict(list)
    fixtures_index = fixtures.set_index("fixture_id")[["home_team_id", "away_team_id"]]
    out = out.sort_values(["match_date", "fixture_key"]).reset_index(drop=True)

    h2h_rows = []
    team_frame_by_fixture = {
        int(fid): grp.copy()
        for fid, grp in team_frame.groupby("fixture_id", sort=False)
    }
    for _, row in out.iterrows():
        fixture_id = int(row["fixture_id"])
        ids = fixtures_index.loc[fixture_id]
        key = _pair_key(int(ids["home_team_id"]), int(ids["away_team_id"]))
        prev = list(reversed(pair_history.get(key, [])))
        h2h_rows.append(
            {
                "fixture_id": fixture_id,
                "h2h_total_fouls_l5": round(_mean(prev, "total_fouls", 5), 4),
                "h2h_total_fouls_l10": round(_mean(prev, "total_fouls", 10), 4),
                "h2h_total_tackles_l5": round(_mean(prev, "total_tackles", 5), 4),
                "h2h_total_tackles_l10": round(_mean(prev, "total_tackles", 10), 4),
                "h2h_total_shots_l5": round(_mean(prev, "total_shots", 5), 4),
                "h2h_total_shots_on_goal_l5": round(_mean(prev, "total_shots_on_goal", 5), 4),
                "h2h_total_corners_l5": round(_mean(prev, "total_corners", 5), 4),
            }
        )
        grp = team_frame_by_fixture.get(fixture_id)
        if grp is not None and len(grp) >= 2:
            pair_history[key].append(
                {
                    "total_fouls": float(pd.to_numeric(grp["fouls_for"], errors="coerce").fillna(0.0).sum()),
                    "total_tackles": float(pd.to_numeric(grp["team_tackles"], errors="coerce").fillna(0.0).sum()),
                    "total_shots": float(pd.to_numeric(grp["shots_total"], errors="coerce").fillna(0.0).sum()),
                    "total_shots_on_goal": float(pd.to_numeric(grp["shots_on_goal"], errors="coerce").fillna(0.0).sum()),
                    "total_corners": float(pd.to_numeric(grp["corners_for"], errors="coerce").fillna(0.0).sum()),
                }
            )

    out = out.merge(pd.DataFrame(h2h_rows), on="fixture_id", how="left")
    out["fixture_foul_density_score"] = (
        0.30 * _norm_cap(out["home_team_fouls_l5"] + out["away_team_fouls_l5"], 28.0)
        + 0.30 * _norm_cap(out["home_team_fouls_l10"] + out["away_team_fouls_l10"], 28.0)
        + 0.40 * _norm_cap(out["h2h_total_fouls_l5"], 30.0)
    )
    out["fixture_tackle_density_score"] = (
        0.35 * _norm_cap(out["home_team_tackles_l5"] + out["away_team_tackles_l5"], 38.0)
        + 0.35 * _norm_cap(out["home_team_tackles_l10"] + out["away_team_tackles_l10"], 38.0)
        + 0.30 * _norm_cap(out["h2h_total_tackles_l5"], 40.0)
    )
    out["fixture_midfield_grind_score"] = (
        0.45 * _norm_cap(out["home_team_interceptions_l5"] + out["away_team_interceptions_l5"], 18.0)
        + 0.35 * _norm_cap(out["fixture_foul_density_score"], 1.0)
        + 0.20 * _norm_cap(out["fixture_tackle_density_score"], 1.0)
    )
    out["fixture_wide_duel_score"] = (
        0.55 * _norm_cap(out["home_team_dribbled_past_l5"] + out["away_team_dribbled_past_l5"], 10.0)
        + 0.45 * _norm_cap(out["fixture_tackle_density_score"], 1.0)
    )
    out["home_team_corners_against_l5"] = out["away_team_corners_for_l5"]
    out["home_team_corners_against_l10"] = out["away_team_corners_for_l10"]
    out["away_team_corners_against_l5"] = out["home_team_corners_for_l5"]
    out["away_team_corners_against_l10"] = out["home_team_corners_for_l10"]
    out["fixture_attack_pressure_score"] = (
        0.25 * _norm_cap(out["home_team_shots_l5"] + out["away_team_shots_l5"], 30.0)
        + 0.20 * _norm_cap(out["home_team_shots_l10"] + out["away_team_shots_l10"], 30.0)
        + 0.20 * _norm_cap(out["home_team_shots_on_goal_l5"] + out["away_team_shots_on_goal_l5"], 10.0)
        + 0.15 * _norm_cap(out["h2h_total_shots_l5"], 32.0)
        + 0.20 * _norm_cap(out["h2h_total_shots_on_goal_l5"], 10.0)
    )
    out["fixture_corner_pressure_score"] = (
        0.35 * _norm_cap(out["home_team_corners_for_l5"] + out["away_team_corners_for_l5"], 14.0)
        + 0.20 * _norm_cap(out["home_team_corners_for_l10"] + out["away_team_corners_for_l10"], 14.0)
        + 0.25 * _norm_cap(out["h2h_total_corners_l5"], 16.0)
        + 0.20 * _norm_cap(out["home_team_corners_against_l5"] + out["away_team_corners_against_l5"], 14.0)
    )
    out["fixture_territorial_stress_score"] = (
        0.40 * _norm_cap(out["home_team_possession_l5"].sub(50.0).abs() + out["away_team_possession_l5"].sub(50.0).abs(), 30.0)
        + 0.35 * _norm_cap(out["home_team_passes_l5"] + out["away_team_passes_l5"], 1100.0)
        + 0.25 * _norm_cap(out["fixture_attack_pressure_score"], 1.0)
    )
    out["fixture_attacking_style_label"] = [
        _attacking_style_label(float(ap), float(cp), float(ts))
        for ap, cp, ts in zip(
            out["fixture_attack_pressure_score"],
            out["fixture_corner_pressure_score"],
            out["fixture_territorial_stress_score"],
        )
    ]
    out["fixture_style_label"] = [
        _style_label(
            foul_density_score=float(fd),
            tackle_density_score=float(td),
            home_fouls_l5=float(hf),
            away_fouls_l5=float(af),
            central_battle_score=float(cb),
            wide_duel_score=float(wd),
        )
        for fd, td, hf, af, cb, wd in zip(
            out["fixture_foul_density_score"],
            out["fixture_tackle_density_score"],
            out["home_team_fouls_l5"],
            out["away_team_fouls_l5"],
            out["fixture_midfield_grind_score"],
            out["fixture_wide_duel_score"],
        )
    ]

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def _default_output_path(league_tag: str, season: int) -> Path:
    return Path("data_sources/api_football/features/player_events") / f"fixture_style_overlay__{league_tag}__{season}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixture-wide foul/tackle density overlay for player-events research.")
    parser.add_argument("--league-tag", required=True)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--fixtures-csv", default="")
    parser.add_argument("--team-stats-csv", default="")
    parser.add_argument("--player-stats-csv", default="")
    parser.add_argument("--output-csv", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = Path("data_sources/api_football/normalized")
    fixtures_csv = args.fixtures_csv or str(base / f"fixtures_master__{args.league_tag}__{args.season}.csv")
    team_stats_csv = args.team_stats_csv or str(base / f"match_team_stats__{args.league_tag}__{args.season}.csv")
    player_stats_csv = args.player_stats_csv or str(base / f"match_player_stats__{args.league_tag}__{args.season}.csv")
    output_csv = args.output_csv or str(_default_output_path(args.league_tag, args.season))
    df = build_fixture_style_overlay(fixtures_csv, team_stats_csv, player_stats_csv, output_csv)
    print(f"WROTE: {output_csv}")
    print(f"rows: {len(df)}")


if __name__ == "__main__":
    main()
