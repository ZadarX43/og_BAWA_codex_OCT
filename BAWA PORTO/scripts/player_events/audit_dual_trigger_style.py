from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _load_actual_bookings(events_csv: str, fixtures_csv: str, player_stats_csv: str) -> pd.DataFrame:
    events = pd.read_csv(events_csv)
    fixtures = pd.read_csv(fixtures_csv, usecols=["fixture_id", "fixture_key"])
    players = (
        pd.read_csv(player_stats_csv, usecols=["fixture_id", "player_id", "player_name"])
        .drop_duplicates(subset=["fixture_id", "player_id"])
    )
    booked = events[
        (events["event_type"].astype("string").str.lower() == "card")
        & (events["event_detail"].astype("string").str.contains("yellow", case=False, na=False))
    ].copy()
    booked = booked.merge(fixtures, on="fixture_id", how="left").merge(players, on=["fixture_id", "player_id"], how="left")
    booked = booked[["fixture_key", "player_name"]].dropna()
    booked["yc_hit_flag"] = 1
    return booked.drop_duplicates(subset=["fixture_key", "player_name"])


def _load_actual_high_fouls(player_stats_csv: str, fixtures_csv: str, foul_threshold: int) -> pd.DataFrame:
    fixtures = pd.read_csv(fixtures_csv, usecols=["fixture_id", "fixture_key"])
    stats = pd.read_csv(player_stats_csv, usecols=["fixture_id", "player_name", "fouls_committed"])
    merged = stats.merge(fixtures, on="fixture_id", how="left")
    merged["fouls_committed"] = pd.to_numeric(merged["fouls_committed"], errors="coerce").fillna(0.0)
    merged["fouls_hit_flag"] = (merged["fouls_committed"] >= foul_threshold).astype(int)
    return merged[["fixture_key", "player_name", "fouls_hit_flag"]].drop_duplicates(subset=["fixture_key", "player_name"])


def audit_dual_trigger_style(
    board_csv: str,
    yc_events_csv: str,
    yc_player_stats_csv: str,
    yc_fixtures_csv: str,
    fouls_player_stats_csv: str,
    fouls_fixtures_csv: str,
    output_csv: str,
    output_md: str,
    foul_threshold: int = 2,
) -> pd.DataFrame:
    board = pd.read_csv(board_csv)
    if board.empty:
        out = pd.DataFrame()
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Dual Trigger Style Audit\n\nNo rows found.\n")
        return out

    dual = (
        board.groupby(["fixture_key", "player_name"], as_index=False)
        .agg(
            league=("league", "first"),
            match_date=("match_date", "first"),
            fixture_style_label=("fixture_style_label", "first"),
            fixture_attacking_style_label=("fixture_attacking_style_label", "first"),
            fixture_quality_score=("fixture_quality_score", "first"),
            trigger_markets=("trigger_markets", "first"),
        )
    )

    booked = _load_actual_bookings(yc_events_csv, yc_fixtures_csv, yc_player_stats_csv)
    fouls = _load_actual_high_fouls(fouls_player_stats_csv, fouls_fixtures_csv, foul_threshold)
    dual = dual.merge(booked, on=["fixture_key", "player_name"], how="left")
    dual = dual.merge(fouls, on=["fixture_key", "player_name"], how="left")
    dual["yc_hit_flag"] = dual["yc_hit_flag"].fillna(0).astype(int)
    dual["fouls_hit_flag"] = dual["fouls_hit_flag"].fillna(0).astype(int)
    dual["both_hit_flag"] = (dual["yc_hit_flag"].eq(1) & dual["fouls_hit_flag"].eq(1)).astype(int)
    dual["either_hit_flag"] = (dual["yc_hit_flag"].eq(1) | dual["fouls_hit_flag"].eq(1)).astype(int)
    dual["style_attack_combo"] = (
        dual["fixture_style_label"].astype("string").fillna("UNSET").str.upper()
        + " + "
        + dual["fixture_attacking_style_label"].astype("string").fillna("UNSET").str.upper()
    )

    style_summary = (
        dual.groupby(["fixture_style_label"], as_index=False)
        .agg(
            picks=("player_name", "count"),
            fixtures=("fixture_key", "nunique"),
            yc_hit_rate=("yc_hit_flag", "mean"),
            fouls_hit_rate=("fouls_hit_flag", "mean"),
            both_hit_rate=("both_hit_flag", "mean"),
            either_hit_rate=("either_hit_flag", "mean"),
            avg_fixture_quality_score=("fixture_quality_score", "mean"),
        )
        .sort_values(["both_hit_rate", "either_hit_rate", "fouls_hit_rate", "picks"], ascending=[False, False, False, False])
        .reset_index(drop=True)
    )

    league_style_summary = (
        dual.groupby(["league", "fixture_style_label"], as_index=False)
        .agg(
            picks=("player_name", "count"),
            fixtures=("fixture_key", "nunique"),
            yc_hit_rate=("yc_hit_flag", "mean"),
            fouls_hit_rate=("fouls_hit_flag", "mean"),
            both_hit_rate=("both_hit_flag", "mean"),
            either_hit_rate=("either_hit_flag", "mean"),
        )
        .sort_values(["both_hit_rate", "either_hit_rate", "fouls_hit_rate", "picks"], ascending=[False, False, False, False])
        .reset_index(drop=True)
    )
    attack_style_summary = (
        dual.groupby(["fixture_attacking_style_label"], as_index=False)
        .agg(
            picks=("player_name", "count"),
            fixtures=("fixture_key", "nunique"),
            yc_hit_rate=("yc_hit_flag", "mean"),
            fouls_hit_rate=("fouls_hit_flag", "mean"),
            both_hit_rate=("both_hit_flag", "mean"),
            either_hit_rate=("either_hit_flag", "mean"),
            avg_fixture_quality_score=("fixture_quality_score", "mean"),
        )
        .sort_values(["both_hit_rate", "either_hit_rate", "fouls_hit_rate", "picks"], ascending=[False, False, False, False])
        .reset_index(drop=True)
    )
    combo_summary = (
        dual.groupby(["fixture_style_label", "fixture_attacking_style_label", "style_attack_combo"], as_index=False)
        .agg(
            picks=("player_name", "count"),
            fixtures=("fixture_key", "nunique"),
            yc_hit_rate=("yc_hit_flag", "mean"),
            fouls_hit_rate=("fouls_hit_flag", "mean"),
            both_hit_rate=("both_hit_flag", "mean"),
            either_hit_rate=("either_hit_flag", "mean"),
            avg_fixture_quality_score=("fixture_quality_score", "mean"),
        )
        .sort_values(["both_hit_rate", "either_hit_rate", "fouls_hit_rate", "picks"], ascending=[False, False, False, False])
        .reset_index(drop=True)
    )
    league_attack_summary = (
        dual.groupby(["league", "fixture_attacking_style_label"], as_index=False)
        .agg(
            picks=("player_name", "count"),
            fixtures=("fixture_key", "nunique"),
            yc_hit_rate=("yc_hit_flag", "mean"),
            fouls_hit_rate=("fouls_hit_flag", "mean"),
            both_hit_rate=("both_hit_flag", "mean"),
            either_hit_rate=("either_hit_flag", "mean"),
        )
        .sort_values(["both_hit_rate", "either_hit_rate", "fouls_hit_rate", "picks"], ascending=[False, False, False, False])
        .reset_index(drop=True)
    )

    style_summary["scope"] = "cross_league"
    league_style_summary["scope"] = "league_style"
    attack_style_summary["scope"] = "cross_league_attack"
    combo_summary["scope"] = "cross_league_combo"
    league_attack_summary["scope"] = "league_attack_style"
    out = pd.concat([style_summary, league_style_summary, attack_style_summary, combo_summary, league_attack_summary], ignore_index=True, sort=False)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    lines = ["# Dual Trigger Style Audit", "", "## Cross-League Style Ranking"]
    for row in style_summary.itertuples(index=False):
        lines.append(
            f"- {row.fixture_style_label}: picks={row.picks} fixtures={row.fixtures} yc_hit={row.yc_hit_rate:.3f} fouls_hit={row.fouls_hit_rate:.3f} both_hit={row.both_hit_rate:.3f} either_hit={row.either_hit_rate:.3f}"
        )
    lines.extend(["", "## Cross-League Attacking Style Ranking"])
    for row in attack_style_summary.itertuples(index=False):
        lines.append(
            f"- {row.fixture_attacking_style_label}: picks={row.picks} fixtures={row.fixtures} yc_hit={row.yc_hit_rate:.3f} fouls_hit={row.fouls_hit_rate:.3f} both_hit={row.both_hit_rate:.3f} either_hit={row.either_hit_rate:.3f}"
        )
    lines.extend(["", "## Cross-League Contact x Attack Combos"])
    for row in combo_summary.head(20).itertuples(index=False):
        lines.append(
            f"- {row.style_attack_combo}: picks={row.picks} fixtures={row.fixtures} yc_hit={row.yc_hit_rate:.3f} fouls_hit={row.fouls_hit_rate:.3f} both_hit={row.both_hit_rate:.3f} either_hit={row.either_hit_rate:.3f}"
        )
    lines.extend(["", "## League x Style Detail"])
    for row in league_style_summary.itertuples(index=False):
        lines.append(
            f"- {row.league} | {row.fixture_style_label}: picks={row.picks} fixtures={row.fixtures} yc_hit={row.yc_hit_rate:.3f} fouls_hit={row.fouls_hit_rate:.3f} both_hit={row.both_hit_rate:.3f} either_hit={row.either_hit_rate:.3f}"
        )
    lines.extend(["", "## League x Attacking Style Detail"])
    for row in league_attack_summary.itertuples(index=False):
        lines.append(
            f"- {row.league} | {row.fixture_attacking_style_label}: picks={row.picks} fixtures={row.fixtures} yc_hit={row.yc_hit_rate:.3f} fouls_hit={row.fouls_hit_rate:.3f} both_hit={row.both_hit_rate:.3f} either_hit={row.either_hit_rate:.3f}"
        )
    Path(output_md).write_text("\n".join(lines))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit same-player dual-trigger board performance by contact style.")
    parser.add_argument("--board-csv", required=True)
    parser.add_argument("--yc-events-csv", required=True)
    parser.add_argument("--yc-player-stats-csv", required=True)
    parser.add_argument("--yc-fixtures-csv", required=True)
    parser.add_argument("--fouls-player-stats-csv", required=True)
    parser.add_argument("--fouls-fixtures-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--foul-threshold", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = audit_dual_trigger_style(
        board_csv=args.board_csv,
        yc_events_csv=args.yc_events_csv,
        yc_player_stats_csv=args.yc_player_stats_csv,
        yc_fixtures_csv=args.yc_fixtures_csv,
        fouls_player_stats_csv=args.fouls_player_stats_csv,
        fouls_fixtures_csv=args.fouls_fixtures_csv,
        output_csv=args.output_csv,
        output_md=args.output_md,
        foul_threshold=args.foul_threshold,
    )
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")


if __name__ == "__main__":
    main()
