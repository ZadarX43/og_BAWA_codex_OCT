from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MARKET_TO_STAT = {
    "shots": "shots_total",
    "shots_on_target": "shots_on_target",
    "fouls_committed": "fouls_committed",
    "tackles": "tackles",
}


def _load_team_lookup(fixtures: pd.DataFrame) -> pd.DataFrame:
    home = fixtures[["fixture_id", "fixture_key", "home_team_id", "home_team_name", "match_date"]].rename(
        columns={"home_team_id": "team_id", "home_team_name": "team_name"}
    )
    away = fixtures[["fixture_id", "fixture_key", "away_team_id", "away_team_name", "match_date"]].rename(
        columns={"away_team_id": "team_id", "away_team_name": "team_name"}
    )
    return pd.concat([home, away], ignore_index=True).drop_duplicates(subset=["fixture_id", "team_id", "team_name"])


def build_backtest(inputs: list[str], family_tags: list[str], fixtures_csv: str, stats_csv: str, output_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    fixtures = pd.read_csv(fixtures_csv, low_memory=False)
    stats = pd.read_csv(stats_csv, low_memory=False)
    team_lookup = _load_team_lookup(fixtures)
    stats = stats.merge(team_lookup, on=["fixture_id", "team_id"], how="left")

    frames = []
    for path, family in zip(inputs, family_tags):
        df = pd.read_csv(path, low_memory=False)
        if df.empty:
            continue
        df = df.copy()
        df["source_family"] = family
        frames.append(df)
    board = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if board.empty:
        empty = pd.DataFrame()
        Path(f"{output_prefix}__family_summary.csv").parent.mkdir(parents=True, exist_ok=True)
        empty.to_csv(f"{output_prefix}__family_summary.csv", index=False)
        empty.to_csv(f"{output_prefix}__family_market_summary.csv", index=False)
        Path(f"{output_prefix}__family_summary.md").write_text("# Specialist Family Backtest Summary\n\nNo rows matched.\n")
        return empty, empty

    board = board.merge(fixtures[["fixture_id", "fixture_key", "match_date"]], on="fixture_key", how="left", suffixes=("", "_fx"))
    board["match_date"] = board["match_date"].fillna(board.get("match_date_fx"))
    board = board.merge(
        team_lookup[["fixture_id", "team_id", "team_name"]],
        on=["fixture_id", "team_name"],
        how="left",
    )
    board = board.merge(
        stats[
            [
                "fixture_id",
                "team_id",
                "player_name",
                "shots_total",
                "shots_on_target",
                "fouls_committed",
                "tackles",
            ]
        ],
        on=["fixture_id", "team_id", "player_name"],
        how="left",
    )
    board["actual_stat"] = board.apply(lambda row: float(row.get(MARKET_TO_STAT.get(str(row.get("market", "")), ""), 0.0) or 0.0), axis=1)
    board["market_hit_flag"] = (board["actual_stat"] >= 1.0).astype(int)
    board["match_month"] = pd.to_datetime(board["match_date"], errors="coerce").dt.to_period("M").astype(str)

    family_summary = (
        board.groupby("source_family", as_index=False)
        .agg(
            rows=("market", "count"),
            fixtures=("fixture_key", "nunique"),
            p1_rows=("priority_bucket", lambda s: int((pd.Series(s) == "P1_SUPER_ELITE").sum())),
            attack_rows=("market", lambda s: int(pd.Series(s).isin(["shots", "shots_on_target"]).sum())),
            contact_rows=("market", lambda s: int(pd.Series(s).isin(["fouls_committed", "tackles"]).sum())),
            row_hit_rate=("market_hit_flag", "mean"),
            fixture_quality_avg=("fixture_quality_score", "mean"),
            market_score_avg=("market_score", "mean"),
        )
        .sort_values(["row_hit_rate", "p1_rows", "fixture_quality_avg"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    family_market_summary = (
        board.groupby(["source_family", "market"], as_index=False)
        .agg(
            rows=("market_hit_flag", "count"),
            hit_rate=("market_hit_flag", "mean"),
            avg_score=("market_score", "mean"),
            avg_fixture_quality=("fixture_quality_score", "mean"),
        )
        .sort_values(["source_family", "hit_rate", "avg_score"], ascending=[True, False, False])
        .reset_index(drop=True)
    )
    family_monthly_summary = (
        board.groupby(["source_family", "match_month"], as_index=False)
        .agg(
            rows=("market_hit_flag", "count"),
            hit_rate=("market_hit_flag", "mean"),
            avg_score=("market_score", "mean"),
        )
        .sort_values(["source_family", "match_month"], ascending=[True, True])
        .reset_index(drop=True)
    )

    family_summary.to_csv(f"{output_prefix}__family_summary.csv", index=False)
    family_market_summary.to_csv(f"{output_prefix}__family_market_summary.csv", index=False)
    family_monthly_summary.to_csv(f"{output_prefix}__family_monthly_summary.csv", index=False)

    lines = ["# Specialist Family Backtest Summary", ""]
    lines.append("## Family Summary")
    for _, row in family_summary.iterrows():
        lines.append(
            f"- {row['source_family']}: hit_rate={row['row_hit_rate']:.3f} | rows={int(row['rows'])} | fixtures={int(row['fixtures'])} | p1_rows={int(row['p1_rows'])} | avg_quality={row['fixture_quality_avg']:.3f}"
        )
    lines.append("")
    lines.append("## Family x Market")
    for _, row in family_market_summary.iterrows():
        lines.append(
            f"- {row['source_family']} | {row['market']}: hit_rate={row['hit_rate']:.3f} | rows={int(row['rows'])} | avg_score={row['avg_score']:.1f}"
        )
    Path(f"{output_prefix}__family_summary.md").write_text("\n".join(lines) + "\n")
    return family_summary, family_market_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest specialist formation families against actual player stats.")
    parser.add_argument("--inputs", required=True, help="Comma-separated family merged board csv paths")
    parser.add_argument("--family-tags", required=True, help="Comma-separated family tags aligned to inputs")
    parser.add_argument("--fixtures-csv", required=True)
    parser.add_argument("--stats-csv", required=True)
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inputs = [x.strip() for x in args.inputs.split(",") if x.strip()]
    tags = [x.strip() for x in args.family_tags.split(",") if x.strip()]
    if len(inputs) != len(tags):
        raise SystemExit("inputs and family-tags must align")
    family_df, market_df = build_backtest(inputs, tags, args.fixtures_csv, args.stats_csv, args.output_prefix)
    print(f"WROTE: {args.output_prefix}__family_summary.csv")
    print(f"families: {len(family_df)} | family_market_rows: {len(market_df)}")
