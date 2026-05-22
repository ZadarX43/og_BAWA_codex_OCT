from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_shortlist(guide_csv: str, elite_csv: str, output_csv: str, output_md: str) -> pd.DataFrame:
    guide = pd.read_csv(guide_csv, low_memory=False)
    elite = pd.read_csv(elite_csv, low_memory=False)
    if guide.empty or elite.empty:
        out = pd.DataFrame()
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Team-Specific Bookings Doubles Shortlist\n\nNo rows matched.\n")
        return out

    allowed = guide[guide["sample_bucket"].isin(["TRUSTED_RECURRING", "SUPPORTING_RECURRING"])].copy()
    if allowed.empty:
        allowed.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Team-Specific Bookings Doubles Shortlist\n\nNo recurring booking-role rows matched.\n")
        return allowed

    pairs = elite.merge(
        allowed[["team_name", "review_family", "tactical_role", "sample_bucket", "hit_rate", "avg_score", "opponent_flank_profile", "opponent_role_context_note"]],
        left_on=["team_name", "source_family", "tactical_role"],
        right_on=["team_name", "review_family", "tactical_role"],
        how="inner",
    )
    if pairs.empty:
        pairs.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Team-Specific Bookings Doubles Shortlist\n\nNo elite rows matched the trusted team-role guide.\n")
        return pairs

    fixture_counts = (
        pairs.groupby(["fixture_key", "home_team_name", "away_team_name"], as_index=False)
        .agg(
            names=("player_name", "count"),
            best_hit_rate=("hit_rate", "max"),
            avg_score=("market_score", "mean"),
        )
    )
    fixture_counts = fixture_counts[fixture_counts["names"].ge(2)].copy()
    out = pairs[pairs["fixture_key"].isin(fixture_counts["fixture_key"])].copy()
    if out.empty:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False)
        Path(output_md).write_text("# Team-Specific Bookings Doubles Shortlist\n\nNo same-fixture doubles survived the team-role filter.\n")
        return out

    fixture_counts["fixture_priority"] = 40.0 * pd.to_numeric(fixture_counts["best_hit_rate"], errors="coerce").fillna(0.0) + pd.to_numeric(fixture_counts["avg_score"], errors="coerce").fillna(0.0)
    out = out.merge(fixture_counts[["fixture_key", "fixture_priority"]], on="fixture_key", how="left")
    out = out.sort_values(["fixture_priority", "fixture_key", "market_score"], ascending=[False, True, False]).reset_index(drop=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    lines = ["# Team-Specific Bookings Doubles Shortlist", "", f"- fixtures: {out['fixture_key'].nunique()} | rows: {len(out)}", ""]
    for fixture_key, sub in out.groupby("fixture_key", sort=False):
        first = sub.iloc[0]
        lines.append(f"## {fixture_key}")
        lines.append(f"- {first['home_team_name']} vs {first['away_team_name']} | fixture priority={first['fixture_priority']:.1f}")
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['player_name']} ({row['team_name']}) | {row['tactical_role']} | family={row['source_family']} | guide={row['sample_bucket']} | hit_rate={row['hit_rate']:.3f} | score={row['market_score']:.1f}"
            )
            lines.append(f"  opponent_context={row['opponent_flank_profile']} | {row['opponent_role_context_note']}")
        lines.append("")
    Path(output_md).write_text("\n".join(lines) + "\n")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a same-fixture bookings doubles shortlist from trusted team-role card patterns.")
    parser.add_argument("--guide-csv", required=True)
    parser.add_argument("--elite-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build_shortlist(args.guide_csv, args.elite_csv, args.output_csv, args.output_md)
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)} | fixtures: {out['fixture_key'].nunique() if not out.empty else 0}")
