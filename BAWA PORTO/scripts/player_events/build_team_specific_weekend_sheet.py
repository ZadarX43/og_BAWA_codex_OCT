from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MIN_RECURRING_ROWS = 2
MIN_RECURRING_FIXTURES = 2


def _recurring_mask(df: pd.DataFrame, rows_col: str, fixtures_col: str) -> pd.Series:
    rows = pd.to_numeric(df.get(rows_col), errors="coerce").fillna(0)
    fixtures = pd.to_numeric(df.get(fixtures_col), errors="coerce").fillna(0)
    return rows.ge(MIN_RECURRING_ROWS) | fixtures.ge(MIN_RECURRING_FIXTURES)


def _prepare_contact(contact: pd.DataFrame) -> pd.DataFrame:
    if contact.empty:
        return pd.DataFrame()
    out = contact.copy()
    out["review_family"] = out["source_family_tag"].astype(str)
    out["score"] = pd.to_numeric(out["market_score"], errors="coerce").fillna(0.0)
    out["board_lane"] = "CONTACT"
    cols = [
        "fixture_key",
        "match_date",
        "competition",
        "league",
        "home_team_name",
        "away_team_name",
        "team_name",
        "player_name",
        "market",
        "tactical_role",
        "review_family",
        "score",
        "fixture_quality_score",
        "formation_pressure_score",
        "priority_bucket",
        "preset_tier",
        "source_batch_tag",
        "manual_pitch_side",
        "manual_overload_target_side",
        "blocks_per90",
        "duels_total_per90",
        "duels_won_per90",
        "aerial_duel_loss_rate",
        "cb_duel_pressure_score",
        "cb_front_foot_duel_flag",
        "opponent_striker_profile",
        "opponent_striker_pressure_tag",
        "opponent_striker_context_note",
        "opponent_striker_subtype_note",
        "board_lane",
    ]
    return out[cols].copy()


def _prepare_bookings(bookings: pd.DataFrame) -> pd.DataFrame:
    if bookings.empty:
        return pd.DataFrame()
    out = bookings.copy()
    out["market"] = "yellow_cards"
    out["review_family"] = out["source_family"].astype(str)
    out["score"] = pd.to_numeric(out["market_score"], errors="coerce").fillna(0.0)
    out["board_lane"] = "BOOKINGS"
    out["priority_bucket"] = "BOOKINGS_SUPER_ELITE"
    out["preset_tier"] = out["portable_tier"].astype(str)
    out["source_batch_tag"] = out["source_batch"].astype(str).str.upper()
    out["manual_pitch_side"] = "UNSET"
    out["manual_overload_target_side"] = "UNSET"
    out["blocks_per90"] = 0.0
    out["duels_total_per90"] = 0.0
    out["duels_won_per90"] = 0.0
    out["aerial_duel_loss_rate"] = 0.0
    out["cb_duel_pressure_score"] = 0.0
    out["cb_front_foot_duel_flag"] = 0
    out["opponent_striker_profile"] = "UNSET"
    out["opponent_striker_pressure_tag"] = "UNSET"
    out["opponent_striker_context_note"] = "No centre-back duel context attached on the bookings side."
    out["opponent_striker_subtype_note"] = "Subtype note unavailable on the bookings side."
    cols = [
        "fixture_key",
        "match_date",
        "competition",
        "league",
        "home_team_name",
        "away_team_name",
        "team_name",
        "player_name",
        "market",
        "tactical_role",
        "review_family",
        "score",
        "fixture_quality_score",
        "formation_pressure_score",
        "priority_bucket",
        "preset_tier",
        "source_batch_tag",
        "manual_pitch_side",
        "manual_overload_target_side",
        "blocks_per90",
        "duels_total_per90",
        "duels_won_per90",
        "aerial_duel_loss_rate",
        "cb_duel_pressure_score",
        "cb_front_foot_duel_flag",
        "opponent_striker_profile",
        "opponent_striker_pressure_tag",
        "opponent_striker_context_note",
        "opponent_striker_subtype_note",
        "board_lane",
    ]
    return out[cols].copy()


def _opponent_context(row: pd.Series) -> tuple[str, str, str, str]:
    family = str(row.get("review_family", ""))
    role = str(row.get("tactical_role", ""))
    market = str(row.get("market", ""))
    if role == "Wide defender / wing-back":
        if family == "4231v433":
            return (
                "OPP_WIDE_ISOLATION",
                "Against an opponent wide lane that repeatedly isolates the flank defender.",
                "WINGER_VS_FULLBACK_ISOLATION",
                "Treat this like a winger-versus-full-back isolation lane rather than a generic contact spot.",
            )
        if family == "4231v442":
            return (
                "OPP_WIDE_SWITCH_PRESSURE",
                "Against an opponent shape that drags the full-back across wide switches.",
                "WINGER_SWITCH_VS_FULLBACK",
                "Treat this like a wide switch-and-recovery matchup against the full-back lane.",
            )
    if role == "Holding midfielder":
        if family == "4231v442":
            return (
                "OPP_DOUBLE_PIVOT_CENTRAL_PRESSURE",
                "Against a central lane that forces repeat screen-and-foul actions.",
                "ADVANCED_8S_VS_HOLDING_MID",
                "Treat this like advanced midfield runners repeatedly forcing the DM to screen and recover.",
            )
        if family == "3421v4231":
            return (
                "OPP_BOX_MIDFIELD_GRIND",
                "Against a packed central block with repeated midfield duel pressure.",
                "BOX_MIDFIELD_VS_DM_COLLISION",
                "Treat this like a compressed central collision between the box midfield and the holding mid.",
            )
        if family == "4231v433":
            return (
                "OPP_HIGH_8_CHANNEL_PRESSURE",
                "Against advanced midfield runners hitting the DM channels.",
                "HIGH_8S_VS_SINGLE_PIVOT",
                "Treat this like high-eights repeatedly attacking the single-pivot channels.",
            )
    if role == "Centre-back enforcer":
        striker_profile = str(row.get("opponent_striker_profile", "UNSET"))
        striker_note = str(row.get("opponent_striker_context_note", "") or "")
        cb_pressure = float(pd.to_numeric(pd.Series([row.get("cb_duel_pressure_score", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        if striker_profile != "UNSET" and cb_pressure >= 0.58:
            return (
                "OPP_DIRECT_CENTRAL_DUEL",
                striker_note or "Against a central duel profile that forces front-foot recovery defending.",
                "STRIKER_VS_FRONT_FOOT_CB",
                f"Treat this like a centre-back duel lane driven by `{striker_profile}` rather than a generic defender contact spot.",
            )
        return (
            "OPP_CB_DUEL_WATCH",
            striker_note or "Centre-back duel lane exists, but the striker-profile evidence is still too soft for full deployment.",
            "STRIKER_VS_FRONT_FOOT_CB_WATCH",
            "Treat this as a centre-back duel watchlist spot until the striker-profile signal repeats more cleanly.",
        )
    if market == "yellow_cards":
        return (
            "OPP_CONTACT_ESCALATION",
            "Card lane driven by repeated defensive stress rather than isolated volume.",
            "CONTACT_ESCALATION_MATCHUP",
            "Treat this like repeat pressure escalation rather than one isolated matchup duel.",
        )
    return (
        "OPP_GENERIC_ROLE_PRESSURE",
        "Role is under opponent-shape pressure but the specific trigger is still broad.",
        "GENERIC_ROLE_MATCHUP",
        "Treat this as a broad role-pressure matchup until we have cleaner player-to-player tags.",
    )


def build_sheet(
    contact_csv: str,
    bookings_csv: str,
    team_role_csv: str,
    team_market_csv: str,
    output_csv: str,
    output_md: str,
    max_fixtures: int = 12,
    max_rows_per_fixture: int = 3,
) -> pd.DataFrame:
    contact = pd.read_csv(contact_csv, low_memory=False)
    bookings = pd.read_csv(bookings_csv, low_memory=False)
    team_role = pd.read_csv(team_role_csv, low_memory=False)
    team_market = pd.read_csv(team_market_csv, low_memory=False)

    board = pd.concat([_prepare_contact(contact), _prepare_bookings(bookings)], ignore_index=True)
    if board.empty:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        board.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Team-Specific Weekend Sheet\n\nNo rows matched.\n")
        return board

    team_role = team_role.rename(
        columns={
            "rows": "role_rows",
            "fixtures": "role_fixtures",
            "hit_rate": "role_hit_rate",
            "avg_score": "role_avg_score",
            "avg_quality": "role_avg_quality",
            "avg_pressure": "role_avg_pressure",
            "markets": "role_markets",
        }
    )
    team_market = team_market.rename(
        columns={
            "rows": "market_rows",
            "fixtures": "market_fixtures",
            "hit_rate": "market_hit_rate",
            "avg_score": "market_avg_score",
        }
    )

    board = board.merge(
        team_role[
            [
                "team_name",
                "review_family",
                "tactical_role",
                "role_rows",
                "role_fixtures",
                "role_hit_rate",
                "role_avg_score",
                "role_avg_quality",
                "role_avg_pressure",
                "role_markets",
            ]
        ],
        on=["team_name", "review_family", "tactical_role"],
        how="left",
    )
    board = board.merge(
        team_market[
            [
                "team_name",
                "review_family",
                "tactical_role",
                "market",
                "market_rows",
                "market_fixtures",
                "market_hit_rate",
                "market_avg_score",
            ]
        ],
        on=["team_name", "review_family", "tactical_role", "market"],
        how="left",
    )

    board["role_rows"] = pd.to_numeric(board["role_rows"], errors="coerce").fillna(0).astype(int)
    board["role_fixtures"] = pd.to_numeric(board["role_fixtures"], errors="coerce").fillna(0).astype(int)
    board["market_rows"] = pd.to_numeric(board["market_rows"], errors="coerce").fillna(0).astype(int)
    board["market_fixtures"] = pd.to_numeric(board["market_fixtures"], errors="coerce").fillna(0).astype(int)
    board["role_hit_rate"] = pd.to_numeric(board["role_hit_rate"], errors="coerce").fillna(0.0)
    board["market_hit_rate"] = pd.to_numeric(board["market_hit_rate"], errors="coerce").fillna(0.0)
    board["manual_pitch_side"] = board["manual_pitch_side"].fillna("UNSET").astype(str)
    board["manual_overload_target_side"] = board["manual_overload_target_side"].fillna("UNSET").astype(str)

    recurring_market = _recurring_mask(board, "market_rows", "market_fixtures")
    recurring_role = _recurring_mask(board, "role_rows", "role_fixtures")
    multi_market_role = board["role_markets"].fillna("").astype(str).str.contains("\\|")

    board["sample_bucket"] = "ONE_OFF_WATCH"
    board.loc[recurring_role & multi_market_role, "sample_bucket"] = "CROSS_MARKET_RECURRING"
    board.loc[recurring_market, "sample_bucket"] = "RECURRING_TRUST"

    board = board[board["sample_bucket"].ne("ONE_OFF_WATCH")].copy()
    board = board[
        board["market_hit_rate"].ge(0.5)
        | board["role_hit_rate"].ge(0.5)
    ].copy()
    if board.empty:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        board.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Team-Specific Weekend Sheet\n\nNo strong recurring team-role rows matched.\n")
        return board

    board["team_specific_priority"] = (
        board["score"]
        + 12.0 * board["market_hit_rate"]
        + 9.0 * board["role_hit_rate"]
        + 1.5 * board["market_rows"]
        + 1.0 * board["role_rows"]
        + 4.0 * pd.to_numeric(board["fixture_quality_score"], errors="coerce").fillna(0.0)
    )
    opponent_context = board.apply(_opponent_context, axis=1, result_type="expand")
    board["opponent_flank_profile"] = opponent_context[0]
    board["opponent_role_context_note"] = opponent_context[1]
    board["player_vs_player_matchup_tag"] = opponent_context[2]
    board["player_vs_player_matchup_note"] = opponent_context[3]
    board["manual_side_override_active"] = (
        board["manual_pitch_side"].ne("UNSET") | board["manual_overload_target_side"].ne("UNSET")
    ).astype(int)

    board = (
        board.sort_values(
            ["fixture_key", "team_specific_priority", "score"],
            ascending=[True, False, False],
        )
        .groupby("fixture_key", group_keys=False)
        .head(max_rows_per_fixture)
        .reset_index(drop=True)
    )

    fixture_rank = (
        board.groupby(["fixture_key", "home_team_name", "away_team_name"], as_index=False)
        .agg(
            fixture_priority=("team_specific_priority", "sum"),
            fixture_rows=("player_name", "count"),
            best_market_hit=("market_hit_rate", "max"),
            best_role_hit=("role_hit_rate", "max"),
        )
        .sort_values(["fixture_priority", "fixture_rows", "best_market_hit"], ascending=[False, False, False])
        .head(max_fixtures)
    )

    out = board[board["fixture_key"].isin(fixture_rank["fixture_key"])].copy()
    out = out.merge(fixture_rank[["fixture_key", "fixture_priority"]], on="fixture_key", how="left")
    out = out.sort_values(
        ["fixture_priority", "fixture_key", "team_specific_priority"],
        ascending=[False, True, False],
    ).reset_index(drop=True)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    lines = ["# Team-Specific Weekend Sheet", ""]
    lines.append(f"- fixtures: {out['fixture_key'].nunique()} | rows: {len(out)}")
    lines.append(
        f"- recurring thresholds: market rows/fixtures >= {MIN_RECURRING_ROWS}/{MIN_RECURRING_FIXTURES} or role recurrence across markets"
    )
    lines.append("")
    for fixture_key, sub in out.groupby("fixture_key", sort=False):
        first = sub.iloc[0]
        lines.append(f"## {fixture_key}")
        lines.append(
            f"- {first['home_team_name']} vs {first['away_team_name']} | fixture priority={first['fixture_priority']:.1f}"
        )
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['player_name']} ({row['team_name']}) | {row['market']} | {row['tactical_role']} | family={row['review_family']} | sample={row['sample_bucket']} | market_hit={row['market_hit_rate']:.3f} | role_hit={row['role_hit_rate']:.3f} | score={row['score']:.1f}"
            )
            lines.append(f"  opponent_context={row['opponent_flank_profile']} | {row['opponent_role_context_note']}")
            lines.append(f"  matchup_tag={row['player_vs_player_matchup_tag']} | {row['player_vs_player_matchup_note']}")
            if str(row.get("opponent_striker_profile", "UNSET")) != "UNSET":
                lines.append(
                    f"  striker_profile={row['opponent_striker_profile']} | pressure_tag={row.get('opponent_striker_pressure_tag','UNSET')} | cb_duel_pressure={float(row.get('cb_duel_pressure_score', 0.0)):.3f}"
                )
                lines.append(f"  subtype_note={row.get('opponent_striker_subtype_note','UNSET')}")
            if int(row["manual_side_override_active"]) == 1:
                lines.append(
                    f"  manual_override=YES | pitch_side={row['manual_pitch_side']} | overload_target={row['manual_overload_target_side']}"
                )
        lines.append("")
    Path(output_md).write_text("\n".join(lines) + "\n")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a weekend sheet from recurring team x family x role patterns.")
    parser.add_argument("--contact-csv", required=True)
    parser.add_argument("--bookings-csv", required=True)
    parser.add_argument("--team-role-csv", required=True)
    parser.add_argument("--team-market-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--max-fixtures", type=int, default=12)
    parser.add_argument("--max-rows-per-fixture", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build_sheet(
        contact_csv=args.contact_csv,
        bookings_csv=args.bookings_csv,
        team_role_csv=args.team_role_csv,
        team_market_csv=args.team_market_csv,
        output_csv=args.output_csv,
        output_md=args.output_md,
        max_fixtures=args.max_fixtures,
        max_rows_per_fixture=args.max_rows_per_fixture,
    )
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)} | fixtures: {out['fixture_key'].nunique() if not out.empty else 0}")
