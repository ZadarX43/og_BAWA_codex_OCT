from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_leaderboard(
    pack_csv: str,
    audit_csv: str,
    super_csv: str,
    super_audit_csv: str,
    output_md: str,
) -> None:
    pack = pd.read_csv(pack_csv, low_memory=False)
    audit = pd.read_csv(audit_csv, low_memory=False)
    super_board = pd.read_csv(super_csv, low_memory=False)
    super_audit = pd.read_csv(super_audit_csv, low_memory=False)

    lines = ["# Bookings Family Leaderboard", ""]
    if audit.empty:
        lines.append("No bookings family rows matched.")
        Path(output_md).write_text("\n".join(lines) + "\n")
        return

    merged = audit.merge(
        super_audit.rename(
            columns={
                "rows": "super_rows",
                "fixtures": "super_fixtures",
                "hit_rate": "super_hit_rate",
                "avg_score": "super_avg_score",
                "avg_fixture_quality": "super_avg_fixture_quality",
            }
        ),
        on="source_family",
        how="left",
    )
    merged["super_rows"] = merged["super_rows"].fillna(0).astype(int)
    merged["super_fixtures"] = merged["super_fixtures"].fillna(0).astype(int)
    merged["super_hit_rate"] = merged["super_hit_rate"].fillna(0.0)
    merged = merged.sort_values(["hit_rate", "super_hit_rate", "rows"], ascending=[False, False, False]).reset_index(drop=True)

    for idx, row in merged.iterrows():
        family = row["source_family"]
        fam_pack = pack[pack["source_family"].eq(family)].copy()
        fam_super = super_board[super_board["source_family"].eq(family)].copy()
        best_fixture = "None"
        if not fam_pack.empty:
            best_fx = (
                fam_pack.groupby(["fixture_key", "home_team_name", "away_team_name"], as_index=False)
                .agg(top_score=("market_score", "max"), rows=("player_name", "count"))
                .sort_values(["top_score", "rows"], ascending=[False, False])
                .iloc[0]
            )
            best_fixture = f"{best_fx['fixture_key']} ({best_fx['home_team_name']} vs {best_fx['away_team_name']})"
        strongest_names = ", ".join(fam_super["player_name"].astype(str).head(3).tolist()) if not fam_super.empty else "None"
        lines.extend(
            [
                f"## {idx + 1}. {family}",
                f"- sample size: {int(row['rows'])} rows across {int(row['fixtures'])} fixtures",
                f"- hit rate: {row['hit_rate']:.3f}",
                f"- super-elite follow-up: {int(row['super_rows'])} rows | hit rate {row['super_hit_rate']:.3f}",
                f"- strongest markets: yellow_cards family pack, with avg score {row['avg_score']:.1f}",
                f"- best example fixture: {best_fixture}",
                f"- best super-elite names: {strongest_names}",
                "",
            ]
        )

    Path(output_md).write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a cards-only bookings family leaderboard.")
    parser.add_argument("--pack-csv", required=True)
    parser.add_argument("--audit-csv", required=True)
    parser.add_argument("--super-csv", required=True)
    parser.add_argument("--super-audit-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_leaderboard(
        pack_csv=args.pack_csv,
        audit_csv=args.audit_csv,
        super_csv=args.super_csv,
        super_audit_csv=args.super_audit_csv,
        output_md=args.output_md,
    )
    print(f"WROTE: {args.output_md}")
