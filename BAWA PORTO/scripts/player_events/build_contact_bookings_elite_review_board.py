from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CONTACT_MARKETS = {"fouls_committed", "tackles"}


def _prep_contact(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["review_family"] = out["source_family_tag"].astype(str)
    out["review_tier"] = out["preset_tier"].astype(str)
    out["review_score"] = pd.to_numeric(out["market_score"], errors="coerce").fillna(0.0)
    return out[
        [
            "fixture_key",
            "match_date",
            "competition",
            "league",
            "home_team_name",
            "away_team_name",
            "team_name",
            "player_name",
            "market",
            "review_score",
            "fixture_quality_score",
            "formation_pressure_score",
            "starting_xi_quality_edge",
            "review_family",
            "review_tier",
            "priority_bucket",
        ]
    ].copy()


def _prep_bookings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["market"] = "yellow_cards"
    out["review_family"] = out["source_family"].astype(str)
    out["review_tier"] = out["portable_tier"].astype(str)
    out["priority_bucket"] = "BOOKINGS_SUPER_ELITE"
    out["review_score"] = pd.to_numeric(out["market_score"], errors="coerce").fillna(0.0)
    return out[
        [
            "fixture_key",
            "match_date",
            "competition",
            "league",
            "home_team_name",
            "away_team_name",
            "team_name",
            "player_name",
            "market",
            "review_score",
            "fixture_quality_score",
            "formation_pressure_score",
            "starting_xi_quality_edge",
            "review_family",
            "review_tier",
            "priority_bucket",
        ]
    ].copy()


def build_board(contact_csv: str, bookings_csv: str, output_csv: str, output_md: str) -> pd.DataFrame:
    contact = pd.read_csv(contact_csv, low_memory=False)
    bookings = pd.read_csv(bookings_csv, low_memory=False)

    frames = []
    if not contact.empty:
        frames.append(_prep_contact(contact[contact["market"].astype(str).isin(CONTACT_MARKETS)].copy()))
    if not bookings.empty:
        frames.append(_prep_bookings(bookings))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        out.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Contact + Bookings Elite Review Board\n\nNo rows matched.\n")
        return out

    fixture_meta = (
        out.groupby("fixture_key", as_index=False)
        .agg(
            fixture_rows=("player_name", "count"),
            markets_present=("market", lambda s: "|".join(sorted(pd.Series(s).astype(str).unique()))),
            combined_review_score=("review_score", "sum"),
            avg_quality=("fixture_quality_score", "mean"),
        )
    )
    fixture_meta["contact_booking_overlap_flag"] = fixture_meta["markets_present"].astype(str).str.contains("yellow_cards") & (
        fixture_meta["markets_present"].astype(str).str.contains("fouls_committed")
        | fixture_meta["markets_present"].astype(str).str.contains("tackles")
    )
    out = out.merge(fixture_meta, on="fixture_key", how="left")
    out = out.sort_values(
        ["contact_booking_overlap_flag", "combined_review_score", "fixture_key", "review_score"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)
    out.to_csv(output_csv, index=False)

    lines = ["# Contact + Bookings Elite Review Board", "", f"- rows: {len(out)} | fixtures: {out['fixture_key'].nunique()}", ""]
    for fixture_key, sub in out.groupby("fixture_key", sort=False):
        first = sub.iloc[0]
        lines.append(f"## {fixture_key}")
        lines.append(
            f"- {first['home_team_name']} vs {first['away_team_name']} | overlap={bool(first['contact_booking_overlap_flag'])} | markets={first['markets_present']} | combined_score={first['combined_review_score']:.1f}"
        )
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['market']}: {row['player_name']} ({row['team_name']}) | family={row['review_family']} | tier={row['review_tier']} | score={row['review_score']:.1f}"
            )
        lines.append("")
    Path(output_md).write_text("\n".join(lines) + "\n")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a combined contact + bookings elite review board.")
    parser.add_argument("--contact-csv", required=True)
    parser.add_argument("--bookings-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build_board(args.contact_csv, args.bookings_csv, args.output_csv, args.output_md)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)} | fixtures: {out['fixture_key'].nunique() if not out.empty else 0}")
