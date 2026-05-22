from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MIN_RECURRING_ROWS = 2
MIN_RECURRING_FIXTURES = 2
MIN_OPPONENT_CONTEXT_ROWS = 2


def _is_recurring(df: pd.DataFrame) -> pd.Series:
    rows = pd.to_numeric(df.get("rows"), errors="coerce").fillna(0)
    fixtures = pd.to_numeric(df.get("fixtures"), errors="coerce").fillna(0)
    return rows.ge(MIN_RECURRING_ROWS) | fixtures.ge(MIN_RECURRING_FIXTURES)


def build_guide(team_family_role_csv: str, team_market_csv: str, output_md: str, context_csv: str | None = None) -> None:
    team_role = pd.read_csv(team_family_role_csv, low_memory=False)
    team_market = pd.read_csv(team_market_csv, low_memory=False)
    context = pd.read_csv(context_csv, low_memory=False) if context_csv and Path(context_csv).exists() else pd.DataFrame()

    lines = ["# Team-Specific Specialist Deploy Guide", ""]
    if team_role.empty:
        lines.append("No rows matched.")
        Path(output_md).write_text("\n".join(lines) + "\n")
        return

    cleaned = team_role[team_role["tactical_role"].astype(str).ne("UNKNOWN")].copy()
    if cleaned.empty:
        cleaned = team_role.copy()

    recurring = cleaned[_is_recurring(cleaned)].copy()
    emerging = cleaned[~_is_recurring(cleaned)].copy()

    lines.append("## Minimum Sample Thresholds")
    lines.append(
        f"- Treat a pattern as `recurring` only when it has at least `{MIN_RECURRING_ROWS}` rows or `{MIN_RECURRING_FIXTURES}` fixtures."
    )
    lines.append(
        f"- Treat opponent-context or matchup tags as deploy-grade only when they repeat at least `{MIN_OPPONENT_CONTEXT_ROWS}` times."
    )
    lines.append("- One-off perfect rows are kept as watchlist notes, not primary deploy guidance.")
    lines.append("")

    lines.append("## Top Recurring Team x Family x Role Patterns")
    source = recurring if not recurring.empty else cleaned
    for _, row in source.sort_values(["hit_rate", "rows", "avg_score"], ascending=[False, False, False]).head(20).iterrows():
        lines.append(
            f"- {row['team_name']} | {row['review_family']} | {row['tactical_role']}: hit_rate={row['hit_rate']:.3f} | rows={int(row['rows'])} | fixtures={int(row['fixtures'])} | markets={row['markets']}"
        )
    lines.append("")

    if not emerging.empty:
        lines.append("## Emerging One-Off Watchlist")
        for _, row in emerging.sort_values(["hit_rate", "avg_score"], ascending=[False, False]).head(10).iterrows():
            lines.append(
                f"- {row['team_name']} | {row['review_family']} | {row['tactical_role']}: hit_rate={row['hit_rate']:.3f} | rows={int(row['rows'])} | fixtures={int(row['fixtures'])} | markets={row['markets']}"
            )
        lines.append("")

    if not context.empty:
        context_cols = {
            "team_name",
            "review_family",
            "tactical_role",
            "opponent_flank_profile",
            "player_vs_player_matchup_tag",
            "market",
        }
        if context_cols.issubset(set(context.columns)):
            context_summary = (
                context.groupby(
                    [
                        "team_name",
                        "review_family",
                        "tactical_role",
                        "opponent_flank_profile",
                        "player_vs_player_matchup_tag",
                    ],
                    as_index=False,
                )
                .agg(
                    rows=("player_name", "count"),
                    markets=("market", lambda s: "|".join(sorted(pd.Series(s).astype(str).unique()))),
                )
                .sort_values(["rows", "team_name"], ascending=[False, True])
            )
            deployable_context = context_summary[context_summary["rows"].ge(MIN_OPPONENT_CONTEXT_ROWS)].copy()
            lines.append("## Opponent-Context Thresholds")
            if deployable_context.empty:
                lines.append("No opponent-context tags have cleared the minimum repeat threshold yet.")
            else:
                for _, row in deployable_context.head(15).iterrows():
                    lines.append(
                        f"- {row['team_name']} | {row['review_family']} | {row['tactical_role']} | {row['opponent_flank_profile']} | {row['player_vs_player_matchup_tag']}: rows={int(row['rows'])} | markets={row['markets']}"
                    )
            lines.append("")
        striker_cols = {
            "team_name",
            "review_family",
            "tactical_role",
            "opponent_striker_profile",
            "opponent_striker_pressure_tag",
            "market",
        }
        if striker_cols.issubset(set(context.columns)):
            striker_context = context[context["opponent_striker_profile"].astype(str).ne("UNSET")].copy()
            if not striker_context.empty:
                striker_summary = (
                    striker_context.groupby(
                        [
                            "team_name",
                            "review_family",
                            "tactical_role",
                            "opponent_striker_profile",
                            "opponent_striker_pressure_tag",
                        ],
                        as_index=False,
                    )
                    .agg(
                        rows=("player_name", "count"),
                        fixtures=("fixture_key", "nunique"),
                        markets=("market", lambda s: "|".join(sorted(pd.Series(s).astype(str).unique()))),
                        avg_cb_duel_pressure=("cb_duel_pressure_score", "mean"),
                    )
                    .sort_values(["rows", "avg_cb_duel_pressure"], ascending=[False, False])
                )
                deployable_striker = striker_summary[striker_summary["rows"].ge(MIN_OPPONENT_CONTEXT_ROWS)].copy()
                lines.append("## Opponent Striker-Profile Thresholds")
                if deployable_striker.empty:
                    lines.append("No opponent striker-profile tags have cleared the minimum repeat threshold yet.")
                else:
                    for _, row in deployable_striker.head(15).iterrows():
                        lines.append(
                            f"- {row['team_name']} | {row['review_family']} | {row['tactical_role']} | {row['opponent_striker_profile']} | {row['opponent_striker_pressure_tag']}: rows={int(row['rows'])} | fixtures={int(row['fixtures'])} | avg_cb_duel_pressure={row['avg_cb_duel_pressure']:.3f} | markets={row['markets']}"
                        )
                lines.append("")

    for market in ["yellow_cards", "fouls_committed", "tackles"]:
        lines.append(f"## {market}")
        market_rows = team_market[team_market["market"].eq(market)].copy()
        market_rows = market_rows[market_rows["tactical_role"].astype(str).ne("UNKNOWN")] if not market_rows.empty else market_rows
        if market_rows.empty:
            lines.append("No role-clean rows matched.")
            lines.append("")
            continue
        recurring_market = market_rows[_is_recurring(market_rows)].copy()
        ranked = (recurring_market if not recurring_market.empty else market_rows).sort_values(
            ["hit_rate", "rows", "avg_score"], ascending=[False, False, False]
        ).head(12)
        for _, row in ranked.iterrows():
            lines.append(
                f"- {row['team_name']} | {row['review_family']} | {row['tactical_role']}: hit_rate={row['hit_rate']:.3f} | rows={int(row['rows'])} | avg_score={row['avg_score']:.1f}"
            )
        if not recurring_market.empty:
            emerging_market = market_rows[~_is_recurring(market_rows)].copy()
            if not emerging_market.empty:
                lines.append("- Emerging watchlist:")
                for _, row in emerging_market.sort_values(["hit_rate", "avg_score"], ascending=[False, False]).head(5).iterrows():
                    lines.append(
                        f"  {row['team_name']} | {row['review_family']} | {row['tactical_role']}: hit_rate={row['hit_rate']:.3f} | rows={int(row['rows'])} | avg_score={row['avg_score']:.1f}"
                    )
        lines.append("")

    lines.append("## Deployment Notes")
    lines.append("- Prefer rows where a team repeats inside the same family with the same role type, not just a one-off high score.")
    lines.append("- For bookings, wide defenders and holding mids are the first team-role combinations to trust when they recur inside `4231v433` and `4231v442`.")
    lines.append("- For contact markets, treat repeated team-role hits as a stronger override than generic family averages once the sample stops being tiny.")
    lines.append("")
    Path(output_md).write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a team-specific specialist deploy guide from the team/family/role audit outputs.")
    parser.add_argument("--team-family-role-csv", required=True)
    parser.add_argument("--team-market-csv", required=True)
    parser.add_argument("--context-csv")
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_guide(args.team_family_role_csv, args.team_market_csv, args.output_md, args.context_csv)
    print(f"WROTE: {args.output_md}")
