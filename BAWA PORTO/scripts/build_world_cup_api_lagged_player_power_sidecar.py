#!/usr/bin/env python3
"""Build lagged API-Football World Cup player-power sidecars.

Research-only fixture sidecar for 2018/2022 World Cup validation.

The player rolling source is timestamp-safe at the feature level because
`api_player_rolling_features` is built from prior fixture appearances only.
The row universe is still the current fixture's player/lineup payload, so this
script labels the aggregates as late-lineup / research scope until we have
timestamped pre-kickoff lineup publication evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MERGED = Path("Matches/__merged__/World_Cup__merged.csv")
DEFAULT_FEATURES = Path("data_sources/api_football/features")
DEFAULT_NORMALIZED = Path("data_sources/api_football/normalized")
DEFAULT_RAW = Path("data_sources/api_football/raw")
DEFAULT_OUTDIR = Path("data_sources/footystats_world_cup/api_lagged_player_power_backbuild")
DEFAULT_SEASONS = "2018,2022"


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
        "| " + " | ".join(str(c) for c in out.columns) + " |",
        "| " + " | ".join(["---"] * len(out.columns)) + " |",
    ]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in out.columns) + " |")
    return "\n".join(lines)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def weighted_mean(frame: pd.DataFrame, value_col: str, weight_col: str = "player_minutes_l5") -> float:
    if frame.empty or value_col not in frame.columns:
        return np.nan
    values = safe_numeric(frame[value_col])
    weights = safe_numeric(frame.get(weight_col, pd.Series(1.0, index=frame.index))).clip(lower=0)
    valid = values.notna()
    if not valid.any():
        return np.nan
    if weights[valid].sum() > 0:
        return float(np.average(values[valid], weights=weights[valid]))
    return float(values[valid].mean())


def mean_col(frame: pd.DataFrame, col: str) -> float:
    if frame.empty or col not in frame.columns:
        return np.nan
    values = safe_numeric(frame[col])
    return float(values.mean()) if values.notna().any() else np.nan


def sum_col(frame: pd.DataFrame, col: str) -> float:
    if frame.empty or col not in frame.columns:
        return np.nan
    values = safe_numeric(frame[col])
    return float(values.sum()) if values.notna().any() else np.nan


def prior_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(False, index=frame.index)
    minutes = safe_numeric(frame.get("player_minutes_l5", pd.Series(0.0, index=frame.index))).fillna(0)
    rating = safe_numeric(frame.get("player_rating_l5", pd.Series(0.0, index=frame.index))).fillna(0)
    starts = safe_numeric(frame.get("player_start_rate_l5", pd.Series(0.0, index=frame.index))).fillna(0)
    return minutes.gt(0) | rating.gt(0) | starts.gt(0)


def add_power_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    rating = safe_numeric(out.get("player_rating_l5", pd.Series(0.0, index=out.index))).fillna(0)
    minutes = safe_numeric(out.get("player_minutes_l5", pd.Series(0.0, index=out.index))).fillna(0)
    start_rate = safe_numeric(out.get("player_start_rate_l5", pd.Series(0.0, index=out.index))).fillna(0)
    goals90 = safe_numeric(out.get("player_goals_per90_l5", pd.Series(0.0, index=out.index))).fillna(0)
    assists90 = safe_numeric(out.get("player_assists_per90_l5", pd.Series(0.0, index=out.index))).fillna(0)
    shots90 = safe_numeric(out.get("player_shots_per90_l5", pd.Series(0.0, index=out.index))).fillna(0)
    sot90 = safe_numeric(out.get("player_sot_per90_l5", pd.Series(0.0, index=out.index))).fillna(0)
    tackles90 = safe_numeric(out.get("player_tackles_per90_l5", pd.Series(0.0, index=out.index))).fillna(0)
    duel = safe_numeric(out.get("player_duel_win_rate_l5", pd.Series(0.0, index=out.index))).fillna(0)
    cards90 = safe_numeric(out.get("player_cards_per90_l10", pd.Series(0.0, index=out.index))).fillna(0)
    fouls90 = safe_numeric(out.get("player_fouls_committed_per90_l5", pd.Series(0.0, index=out.index))).fillna(0)

    out["api_player_power_component_score"] = (
        rating
        + 0.012 * minutes.clip(0, 90)
        + 0.60 * start_rate.clip(0, 1)
        + 0.35 * goals90.clip(0, 3)
        + 0.25 * assists90.clip(0, 3)
        + 0.04 * shots90.clip(0, 8)
        + 0.08 * sot90.clip(0, 5)
    )
    out["api_player_attack_component_score"] = (
        goals90.clip(0, 3)
        + 0.75 * assists90.clip(0, 3)
        + 0.10 * shots90.clip(0, 8)
        + 0.25 * sot90.clip(0, 5)
    )
    out["api_player_defence_component_score"] = (
        0.50 * tackles90.clip(0, 8)
        + 1.50 * duel.clip(0, 1)
        - 0.35 * cards90.clip(0, 2)
        - 0.10 * fouls90.clip(0, 5)
    )
    return out


def load_merged(path: Path, seasons: list[int]) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df[pd.to_numeric(df["season"], errors="coerce").isin(seasons)].copy()
    for col in ["api_fixture_id", "api_home_team_id", "api_away_team_id", "timestamp", "season"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["season", "timestamp", "fixture_key"]).reset_index(drop=True)


def load_player_sources(features_dir: Path, normalized_dir: Path, seasons: list[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        rolling_path = features_dir / f"api_player_rolling_features__World_Cup__{season}.csv"
        stats_path = normalized_dir / f"match_player_stats__World_Cup__{season}.csv"
        if not rolling_path.exists() or not stats_path.exists():
            continue
        rolling = pd.read_csv(rolling_path, low_memory=False)
        stats = pd.read_csv(
            stats_path,
            low_memory=False,
            usecols=lambda c: c
            in {
                "fixture_id",
                "player_id",
                "team_id",
                "started_flag",
                "minutes",
                "rating",
                "position",
            },
        )
        merged = rolling.merge(
            stats.rename(
                columns={
                    "started_flag": "current_started_flag",
                    "minutes": "current_minutes",
                    "rating": "current_rating",
                    "position": "current_position",
                }
            ),
            on=["fixture_id", "player_id", "team_id"],
            how="left",
        )
        frames.append(add_power_columns(merged))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def load_roster_counts(raw_dir: Path, seasons: list[int]) -> pd.DataFrame:
    rows = []
    for season in seasons:
        path = raw_dir / f"players__league_1__season_{season}__players.jsonl"
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
                for item in payload.get("response") or []:
                    player = item.get("player") or {}
                    for stat in item.get("statistics") or []:
                        team = stat.get("team") or {}
                        games = stat.get("games") or {}
                        rows.append(
                            {
                                "season": season,
                                "team_id": team.get("id"),
                                "team_name": team.get("name"),
                                "player_id": player.get("id"),
                                "player_age": player.get("age"),
                                "player_position": games.get("position"),
                            }
                        )
    if not rows:
        return pd.DataFrame(columns=["season", "team_id", "api_wc_roster_players", "api_wc_roster_avg_age"])
    df = pd.DataFrame(rows)
    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce")
    df["player_age"] = pd.to_numeric(df["player_age"], errors="coerce")
    return (
        df.dropna(subset=["team_id"])
        .groupby(["season", "team_id"], dropna=False)
        .agg(
            api_wc_roster_players=("player_id", "nunique"),
            api_wc_roster_avg_age=("player_age", "mean"),
            api_wc_roster_positions=("player_position", lambda s: int(s.notna().sum())),
        )
        .reset_index()
    )


def side_summary(frame: pd.DataFrame, prefix: str, roster: pd.Series | None = None) -> dict:
    total_rows = int(len(frame))
    prior = frame[prior_mask(frame)].copy()
    starters = frame[safe_numeric(frame.get("current_started_flag", pd.Series(0, index=frame.index))).fillna(0).eq(1)].copy()
    starter_prior = starters[prior_mask(starters)].copy()
    participants = frame[safe_numeric(frame.get("current_minutes", pd.Series(0, index=frame.index))).fillna(0).gt(0)].copy()
    participant_prior = participants[prior_mask(participants)].copy()
    top11 = prior.sort_values(["api_player_power_component_score", "player_minutes_l5"], ascending=False).head(11)

    roster_players = np.nan
    roster_avg_age = np.nan
    if roster is not None and not roster.empty:
        roster_players = pd.to_numeric(pd.Series([roster.get("api_wc_roster_players")]), errors="coerce").iloc[0]
        roster_avg_age = pd.to_numeric(pd.Series([roster.get("api_wc_roster_avg_age")]), errors="coerce").iloc[0]

    return {
        f"{prefix}_api_wc_fixture_player_rows": total_rows,
        f"{prefix}_api_wc_fixture_player_prior_rows": int(len(prior)),
        f"{prefix}_api_wc_fixture_starter_rows": int(len(starters)),
        f"{prefix}_api_wc_fixture_starter_prior_rows": int(len(starter_prior)),
        f"{prefix}_api_wc_fixture_participant_prior_rows": int(len(participant_prior)),
        f"{prefix}_api_wc_roster_players": roster_players,
        f"{prefix}_api_wc_roster_avg_age": roster_avg_age,
        f"{prefix}_api_wc_roster_coverage_rate": float(total_rows / roster_players) if pd.notna(roster_players) and roster_players else np.nan,
        f"{prefix}_api_wc_player_prior_coverage_rate": float(len(prior) / total_rows) if total_rows else np.nan,
        f"{prefix}_api_wc_starter_prior_coverage_rate": float(len(starter_prior) / len(starters)) if len(starters) else np.nan,
        f"{prefix}_api_wc_participant_prior_coverage_rate": float(len(participant_prior) / len(participants)) if len(participants) else np.nan,
        f"{prefix}_api_wc_player_power_score": weighted_mean(prior, "api_player_power_component_score"),
        f"{prefix}_api_wc_player_power_top11_score": mean_col(top11, "api_player_power_component_score"),
        f"{prefix}_api_wc_starter_player_power_score": weighted_mean(starter_prior, "api_player_power_component_score"),
        f"{prefix}_api_wc_participant_player_power_score": weighted_mean(participant_prior, "api_player_power_component_score"),
        f"{prefix}_api_wc_player_attack_score": weighted_mean(prior, "api_player_attack_component_score"),
        f"{prefix}_api_wc_player_attack_top11_score": mean_col(top11, "api_player_attack_component_score"),
        f"{prefix}_api_wc_starter_player_attack_score": weighted_mean(starter_prior, "api_player_attack_component_score"),
        f"{prefix}_api_wc_player_defence_score": weighted_mean(prior, "api_player_defence_component_score"),
        f"{prefix}_api_wc_player_defence_top11_score": mean_col(top11, "api_player_defence_component_score"),
        f"{prefix}_api_wc_starter_player_defence_score": weighted_mean(starter_prior, "api_player_defence_component_score"),
        f"{prefix}_api_wc_player_minutes_score": mean_col(prior, "player_minutes_l5"),
        f"{prefix}_api_wc_player_start_rate": mean_col(prior, "player_start_rate_l5"),
        f"{prefix}_api_wc_player_goal_threat_score": weighted_mean(prior, "player_goals_per90_l5"),
        f"{prefix}_api_wc_player_creativity_score": weighted_mean(prior, "player_assists_per90_l5"),
        f"{prefix}_api_wc_player_shot_volume_score": weighted_mean(prior, "player_shots_per90_l5"),
        f"{prefix}_api_wc_player_sot_score": weighted_mean(prior, "player_sot_per90_l5"),
        f"{prefix}_api_wc_player_cards_score": weighted_mean(prior, "player_cards_per90_l10"),
        f"{prefix}_api_wc_first_fixture_no_player_prior_rate": 1.0 if total_rows and not len(prior) else 0.0,
        f"{prefix}_api_wc_lineup_membership_leakage_risk_score": 1.0,
    }


def add_delta(row: dict, left: str, right: str, metric: str, out_name: str | None = None) -> None:
    out = out_name or f"api_wc_{metric}_diff"
    lval = pd.to_numeric(pd.Series([row.get(f"{left}_{metric}")]), errors="coerce").iloc[0]
    rval = pd.to_numeric(pd.Series([row.get(f"{right}_{metric}")]), errors="coerce").iloc[0]
    row[out] = lval - rval


def build_sidecar(merged: pd.DataFrame, players: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    player_groups = {int(k): v.copy() for k, v in players.groupby("fixture_id", dropna=False) if pd.notna(k)}
    roster_keyed = {
        (int(r.season), int(r.team_id)): pd.Series(r._asdict())
        for r in roster.itertuples(index=False)
        if pd.notna(r.team_id)
    }
    rows = []
    for fx in merged.itertuples(index=False):
        fixture_id = getattr(fx, "api_fixture_id", np.nan)
        home_team_id = getattr(fx, "api_home_team_id", np.nan)
        away_team_id = getattr(fx, "api_away_team_id", np.nan)
        season = int(getattr(fx, "season"))
        frame = player_groups.get(int(fixture_id), pd.DataFrame()) if pd.notna(fixture_id) else pd.DataFrame()
        home = frame[frame["team_id"].eq(int(home_team_id))].copy() if pd.notna(home_team_id) and not frame.empty else pd.DataFrame()
        away = frame[frame["team_id"].eq(int(away_team_id))].copy() if pd.notna(away_team_id) and not frame.empty else pd.DataFrame()
        row = {
            "fixture_key": fx.fixture_key,
            "season": season,
            "match_date": getattr(fx, "match_date", ""),
            "home_team_name": getattr(fx, "home_team_name", ""),
            "away_team_name": getattr(fx, "away_team_name", ""),
            "api_fixture_id": fixture_id,
            "api_wc_player_power_scope_status": "LAGGED_PLAYER_FEATURES_WITH_CURRENT_FIXTURE_LINEUP_MEMBERSHIP",
            "api_wc_player_power_training_policy": "RESEARCH_ONLY_UNTIL_PRE_KICKOFF_LINEUP_TIMESTAMPS_EXIST",
            "api_wc_player_power_feature_timestamp_policy": "PLAYER_VALUES_USE_PRIOR_WORLD_CUP_APPEARANCES_ONLY",
        }
        row.update(side_summary(home, "home", roster_keyed.get((season, int(home_team_id))) if pd.notna(home_team_id) else None))
        row.update(side_summary(away, "away", roster_keyed.get((season, int(away_team_id))) if pd.notna(away_team_id) else None))
        for metric in [
            "api_wc_roster_coverage_rate",
            "api_wc_player_prior_coverage_rate",
            "api_wc_starter_prior_coverage_rate",
            "api_wc_participant_prior_coverage_rate",
            "api_wc_player_power_score",
            "api_wc_player_power_top11_score",
            "api_wc_starter_player_power_score",
            "api_wc_participant_player_power_score",
            "api_wc_player_attack_score",
            "api_wc_player_attack_top11_score",
            "api_wc_starter_player_attack_score",
            "api_wc_player_defence_score",
            "api_wc_player_defence_top11_score",
            "api_wc_starter_player_defence_score",
            "api_wc_player_minutes_score",
            "api_wc_player_start_rate",
            "api_wc_player_goal_threat_score",
            "api_wc_player_creativity_score",
            "api_wc_player_shot_volume_score",
            "api_wc_player_sot_score",
            "api_wc_player_cards_score",
            "api_wc_first_fixture_no_player_prior_rate",
        ]:
            add_delta(row, "home", "away", metric, f"{metric}_diff")
        home_rows = row.get("home_api_wc_fixture_player_rows", 0) or 0
        away_rows = row.get("away_api_wc_fixture_player_rows", 0) or 0
        home_prior = row.get("home_api_wc_fixture_player_prior_rows", 0) or 0
        away_prior = row.get("away_api_wc_fixture_player_prior_rows", 0) or 0
        row["api_wc_player_power_any_prior_rate"] = 1.0 if home_prior and away_prior else 0.0
        row["api_wc_player_power_fixture_coverage_rate"] = float((home_rows > 0) and (away_rows > 0))
        row["api_wc_player_power_missing_rate"] = 1.0 - row["api_wc_player_power_any_prior_rate"]
        rows.append(row)
    return pd.DataFrame(rows)


def coverage_table(sidecar: pd.DataFrame) -> pd.DataFrame:
    checks = []
    for season, frame in sidecar.groupby("season", dropna=False):
        checks.extend(
            [
                {
                    "season": season,
                    "coverage_check": "api_fixture_player_rows_both_sides",
                    "fixtures": int(
                        (
                            pd.to_numeric(frame["home_api_wc_fixture_player_rows"], errors="coerce").gt(0)
                            & pd.to_numeric(frame["away_api_wc_fixture_player_rows"], errors="coerce").gt(0)
                        ).sum()
                    ),
                    "total": len(frame),
                },
                {
                    "season": season,
                    "coverage_check": "lagged_player_prior_both_sides",
                    "fixtures": int(pd.to_numeric(frame["api_wc_player_power_any_prior_rate"], errors="coerce").fillna(0).gt(0).sum()),
                    "total": len(frame),
                },
                {
                    "season": season,
                    "coverage_check": "starter_prior_both_sides",
                    "fixtures": int(
                        (
                            pd.to_numeric(frame["home_api_wc_starter_prior_coverage_rate"], errors="coerce").fillna(0).gt(0)
                            & pd.to_numeric(frame["away_api_wc_starter_prior_coverage_rate"], errors="coerce").fillna(0).gt(0)
                        ).sum()
                    ),
                    "total": len(frame),
                },
            ]
        )
    out = pd.DataFrame(checks)
    if not out.empty:
        out["coverage_rate"] = out["fixtures"] / out["total"]
    return out


def write_summary(outdir: Path, sidecar: pd.DataFrame, coverage: pd.DataFrame) -> None:
    score_cols = [
        "fixture_key",
        "home_team_name",
        "away_team_name",
        "api_wc_player_power_score_diff",
        "api_wc_player_attack_score_diff",
        "api_wc_player_defence_score_diff",
        "api_wc_player_power_any_prior_rate",
        "api_wc_player_power_missing_rate",
    ]
    sample = sidecar[sidecar["season"].eq(2022)][[c for c in score_cols if c in sidecar.columns]].head(12)
    lines = [
        "# World Cup API Lagged Player-Power Sidecar",
        "",
        "Research-only fixture sidecar built from API-Football World Cup player ratings/statistics.",
        "",
        "## Outputs",
        "",
        f"- `{outdir / 'world_cup_api_lagged_player_power_fixture_sidecar.csv'}`",
        f"- `{outdir / 'world_cup_api_lagged_player_power_coverage.csv'}`",
        "",
        "## Coverage",
        "",
        markdown_table(coverage),
        "",
        "## 2022 Sample",
        "",
        markdown_table(sample),
        "",
        "## Guardrails",
        "",
        "- Player values are lagged: they use prior World Cup appearances before the fixture.",
        "- Current fixture player/lineup membership comes from API fixture player payloads, so this is late-lineup research scope unless lineup publish timestamps prove pre-kickoff availability.",
        "- API historical injuries returned zero rows for 2018/2022, so injury shock still needs an external timestamped source for historical validation.",
        "- This does not modify production routing or ModelStore artifacts.",
    ]
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged", type=Path, default=DEFAULT_MERGED)
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--seasons", default=DEFAULT_SEASONS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seasons = [int(x.strip()) for x in str(args.seasons).split(",") if x.strip()]
    args.outdir.mkdir(parents=True, exist_ok=True)
    merged = load_merged(args.merged, seasons)
    players = load_player_sources(args.features_dir, args.normalized_dir, seasons)
    roster = load_roster_counts(args.raw_dir, seasons)
    sidecar = build_sidecar(merged, players, roster)
    coverage = coverage_table(sidecar)
    sidecar_path = args.outdir / "world_cup_api_lagged_player_power_fixture_sidecar.csv"
    coverage_path = args.outdir / "world_cup_api_lagged_player_power_coverage.csv"
    sidecar.to_csv(sidecar_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    write_summary(args.outdir, sidecar, coverage)
    print(f"[ok] fixtures={len(sidecar)} coverage_checks={len(coverage)}")
    print(f"[ok] wrote {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
