from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_watchlist(cb_csv: str, bookings_csv: str, output_csv: str, output_md: str) -> pd.DataFrame:
    cb = pd.read_csv(cb_csv, low_memory=False)
    bookings = pd.read_csv(bookings_csv, low_memory=False)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    if cb.empty or bookings.empty:
        out = pd.DataFrame()
        out.to_csv(output_csv, index=False)
        Path(output_md).write_text("# CB Duel + Bookings Crossover Watchlist\n\nNo rows matched.\n")
        return out

    cb_key = cb[[
        "fixture_key",
        "home_team_name",
        "away_team_name",
        "team_name",
        "player_name",
        "market",
        "opponent_striker_profile",
        "opponent_striker_pressure_tag",
        "opponent_striker_subtype_note",
        "cb_duel_pressure_score",
        "market_hit_rate",
        "role_hit_rate",
        "score",
    ]].copy()
    cb_key = cb_key.rename(
        columns={
            "player_name": "cb_player_name",
            "market": "cb_market",
            "market_hit_rate": "cb_market_hit_rate",
            "role_hit_rate": "cb_role_hit_rate",
            "score": "cb_score",
        }
    )

    book = bookings[[
        "fixture_key",
        "home_team_name",
        "away_team_name",
        "team_name",
        "player_name",
        "market_score",
        "booking_probability_index",
        "portable_tier",
        "tactical_role",
    ]].copy()
    book = book.rename(
        columns={
            "player_name": "booking_player_name",
            "market_score": "booking_score",
            "tactical_role": "booking_role",
        }
    )

    out = cb_key.merge(book, on=["fixture_key", "home_team_name", "away_team_name"], how="inner")
    out = out[out["team_name_x"].astype(str) != out["team_name_y"].astype(str)].copy()
    out = out.rename(columns={"team_name_x": "cb_team_name", "team_name_y": "booking_team_name"})
    if out.empty:
        out.to_csv(output_csv, index=False)
        Path(output_md).write_text("# CB Duel + Bookings Crossover Watchlist\n\nNo crossover rows matched.\n")
        return out

    out["crossover_priority"] = (
        pd.to_numeric(out["cb_score"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["booking_score"], errors="coerce").fillna(0.0)
        + 20.0 * pd.to_numeric(out["cb_duel_pressure_score"], errors="coerce").fillna(0.0)
        + 10.0 * pd.to_numeric(out["booking_probability_index"], errors="coerce").fillna(0.0)
    )
    out = out.sort_values(["crossover_priority", "fixture_key"], ascending=[False, True]).reset_index(drop=True)
    out.to_csv(output_csv, index=False)

    lines = ["# CB Duel + Bookings Crossover Watchlist", "", f"- fixtures: {out['fixture_key'].nunique()} | rows: {len(out)}", ""]
    for fixture_key, sub in out.groupby("fixture_key", sort=False):
        first = sub.iloc[0]
        lines.append(f"## {fixture_key}")
        lines.append(
            f"- {first['home_team_name']} vs {first['away_team_name']} | cb_profile={first['opponent_striker_profile']} | pressure_tag={first['opponent_striker_pressure_tag']}"
        )
        lines.append(f"  subtype_note={first.get('opponent_striker_subtype_note', 'UNSET')}")
        for _, row in sub.head(6).iterrows():
            lines.append(
                f"- CB lane: {row['cb_player_name']} ({row['cb_team_name']}) | {row['cb_market']} | cb_pressure={float(row['cb_duel_pressure_score']):.3f}"
            )
            lines.append(
                f"  bookings support: {row['booking_player_name']} ({row['booking_team_name']}) | role={row['booking_role']} | booking_score={float(row['booking_score']):.1f} | tier={row['portable_tier']}"
            )
        lines.append("")
    Path(output_md).write_text("\n".join(lines) + "\n")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a crossover watchlist linking CB duel fixtures with bookings names in the same game.")
    parser.add_argument("--cb-csv", required=True)
    parser.add_argument("--bookings-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build_watchlist(args.cb_csv, args.bookings_csv, args.output_csv, args.output_md)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)} | fixtures: {out['fixture_key'].nunique() if not out.empty else 0}")
