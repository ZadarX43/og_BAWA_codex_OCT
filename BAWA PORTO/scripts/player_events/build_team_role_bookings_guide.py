from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MIN_RECURRING_ROWS = 2
MIN_RECURRING_FIXTURES = 2


def _recurring_mask(df: pd.DataFrame) -> pd.Series:
    rows = pd.to_numeric(df.get("rows"), errors="coerce").fillna(0)
    fixtures = pd.to_numeric(df.get("fixtures"), errors="coerce").fillna(0)
    return rows.ge(MIN_RECURRING_ROWS) | fixtures.ge(MIN_RECURRING_FIXTURES)


def _bookings_opponent_context(family: str, role: str) -> tuple[str, str, str, str]:
    family = str(family)
    role = str(role)
    if role == "Wide defender / wing-back":
        if family == "4231v433":
            return (
                "OPP_WINGER_ISOLATION",
                "Cards tend to come from repeat winger isolation and recovery fouls on the flank.",
                "WINGER_VS_FULLBACK_ISOLATION",
                "Trust this most when an isolated winger is repeatedly attacking the same full-back lane.",
            )
        return (
            "OPP_WIDE_ROTATION_PRESSURE",
            "Cards tend to come from wide overload rotations forcing late wide challenges.",
            "WIDE_ROTATION_VS_FULLBACK",
            "Trust this when wide rotations repeatedly force the full-back to pass runners across.",
        )
    if role == "Holding midfielder":
        if family == "4231v442":
            return (
                "OPP_CENTRAL_SCREEN_STRESS",
                "Cards tend to come from repeated central screen fouls against split attacking lanes.",
                "ADVANCED_8S_VS_HOLDING_MID",
                "Trust this when the holding midfielder is repeatedly screening advanced runners through central lanes.",
            )
        return (
            "OPP_MIDFIELD_COLLISION",
            "Cards tend to come from repeated central duel and second-ball pressure.",
            "MIDFIELD_COLLISION_MATCHUP",
            "Trust this when the game repeatedly collapses into central duel and second-ball collisions.",
        )
    return (
        "OPP_GENERIC_CONTACT_STRESS",
        "Cards are being driven by repeated opponent pressure without a clean role-specific trigger yet.",
        "GENERIC_CONTACT_MATCHUP",
        "Treat this as broad pressure only until the matchup story becomes cleaner.",
    )


def build_guide(bookings_csv: str, team_role_csv: str, team_market_csv: str, output_csv: str, output_md: str) -> pd.DataFrame:
    bookings = pd.read_csv(bookings_csv, low_memory=False)
    team_role = pd.read_csv(team_role_csv, low_memory=False)
    team_market = pd.read_csv(team_market_csv, low_memory=False)

    cards = team_market[team_market["market"].eq("yellow_cards")].copy()
    cards = cards[cards["tactical_role"].astype(str).ne("UNKNOWN")].copy()
    if cards.empty:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        cards.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Team-Role Bookings Guide\n\nNo card rows matched.\n")
        return cards

    family_role = team_role[["team_name", "review_family", "tactical_role", "rows", "fixtures", "hit_rate", "markets"]].rename(
        columns={
            "rows": "role_rows",
            "fixtures": "role_fixtures",
            "hit_rate": "role_hit_rate",
            "markets": "role_markets",
        }
    )
    cards = cards.merge(family_role, on=["team_name", "review_family", "tactical_role"], how="left")

    cards["sample_bucket"] = "EMERGING_ONE_OFF"
    recurring_exact = _recurring_mask(cards)
    recurring_support = (
        pd.to_numeric(cards["role_rows"], errors="coerce").fillna(0).ge(2)
        & cards["role_markets"].fillna("").astype(str).str.contains("yellow_cards")
    )
    cards.loc[recurring_support, "sample_bucket"] = "SUPPORTING_RECURRING"
    cards.loc[recurring_exact, "sample_bucket"] = "SAMPLE_RECURRING_CAUTION"
    cards.loc[recurring_exact & pd.to_numeric(cards["hit_rate"], errors="coerce").fillna(0.0).ge(0.25), "sample_bucket"] = "TRUSTED_RECURRING"
    opponent_context = cards.apply(
        lambda row: _bookings_opponent_context(row["review_family"], row["tactical_role"]),
        axis=1,
        result_type="expand",
    )
    cards["opponent_flank_profile"] = opponent_context[0]
    cards["opponent_role_context_note"] = opponent_context[1]
    cards["player_vs_player_matchup_tag"] = opponent_context[2]
    cards["player_vs_player_matchup_note"] = opponent_context[3]

    cards["guide_priority"] = (
        50.0 * pd.to_numeric(cards["hit_rate"], errors="coerce").fillna(0.0)
        + 0.35 * pd.to_numeric(cards["avg_score"], errors="coerce").fillna(0.0)
        + 4.0 * pd.to_numeric(cards["rows"], errors="coerce").fillna(0.0)
        + 2.0 * pd.to_numeric(cards["role_rows"], errors="coerce").fillna(0.0)
    )
    cards = cards.sort_values(["guide_priority", "rows", "avg_score"], ascending=[False, False, False]).reset_index(drop=True)

    elite_names = bookings[["fixture_key", "team_name", "player_name", "tactical_role", "source_family", "market_score"]].copy()
    elite_names = elite_names.rename(columns={"source_family": "review_family", "market_score": "latest_market_score"})
    summary = (
        elite_names.groupby(["team_name", "review_family", "tactical_role"], as_index=False)
        .agg(
            elite_fixture_examples=("fixture_key", lambda s: "|".join(pd.Series(s).astype(str).head(3))),
            elite_names=("player_name", lambda s: "|".join(pd.Series(s).astype(str).head(4))),
            latest_market_score=("latest_market_score", "max"),
        )
    )
    cards = cards.merge(summary, on=["team_name", "review_family", "tactical_role"], how="left")

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    cards.to_csv(output_csv, index=False)

    lines = ["# Team-Role Bookings Guide", ""]
    lines.append(f"- thresholds: recurring rows/fixtures >= {MIN_RECURRING_ROWS}/{MIN_RECURRING_FIXTURES}")
    lines.append("")
    for bucket in ["TRUSTED_RECURRING", "SAMPLE_RECURRING_CAUTION", "SUPPORTING_RECURRING", "EMERGING_ONE_OFF"]:
        sub = cards[cards["sample_bucket"].eq(bucket)].head(12)
        lines.append(f"## {bucket}")
        if sub.empty:
            lines.append("No rows matched.")
            lines.append("")
            continue
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['team_name']} | {row['review_family']} | {row['tactical_role']}: hit_rate={row['hit_rate']:.3f} | rows={int(row['rows'])} | fixtures={int(row['fixtures'])} | avg_score={row['avg_score']:.1f} | latest_names={row.get('elite_names', '')}"
            )
            lines.append(f"  opponent_context={row['opponent_flank_profile']} | {row['opponent_role_context_note']}")
            lines.append(f"  matchup_tag={row['player_vs_player_matchup_tag']} | {row['player_vs_player_matchup_note']}")
        lines.append("")
    lines.append("## Notes")
    lines.append("- Trust repeated wide-defender and holding-mid card patterns first when they survive the recurring threshold.")
    lines.append("- `SAMPLE_RECURRING_CAUTION` means the team-role repeats enough to matter, but the realized card hit rate is still poor so far.")
    lines.append("- One-off perfect rows stay in the guide as watchlist names, not primary deploy signals.")
    lines.append("")
    Path(output_md).write_text("\n".join(lines) + "\n")
    return cards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a team-role bookings guide with recurring-sample thresholds.")
    parser.add_argument("--bookings-csv", required=True)
    parser.add_argument("--team-role-csv", required=True)
    parser.add_argument("--team-market-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build_guide(args.bookings_csv, args.team_role_csv, args.team_market_csv, args.output_csv, args.output_md)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")
