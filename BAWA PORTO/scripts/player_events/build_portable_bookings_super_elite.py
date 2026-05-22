from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_super_elite(input_csv: str, output_prefix: str, max_fixtures: int = 24, max_per_fixture: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.read_csv(input_csv, low_memory=False)
    if base.empty:
        empty = pd.DataFrame()
        Path(f"{output_prefix}.csv").parent.mkdir(parents=True, exist_ok=True)
        empty.to_csv(f"{output_prefix}.csv", index=False)
        empty.to_csv(f"{output_prefix}__audit.csv", index=False)
        Path(f"{output_prefix}.md").write_text("# Portable Bookings Super-Elite\n\nNo rows matched.\n")
        Path(f"{output_prefix}__audit.md").write_text("# Portable Bookings Super-Elite Audit\n\nNo rows matched.\n")
        return empty, empty

    core_mask = (
        base["portable_tier"].astype(str).eq("CORE_PORTABLE_BOOKINGS")
        & pd.to_numeric(base["market_score"], errors="coerce").fillna(0.0).ge(84.0)
        & pd.to_numeric(base["fixture_quality_score"], errors="coerce").fillna(0.0).ge(0.80)
        & pd.to_numeric(base["formation_pressure_score"], errors="coerce").fillna(0.0).ge(0.65)
    )
    conditional_mask = (
        base["portable_tier"].astype(str).eq("CONDITIONAL_PORTABLE_BOOKINGS")
        & pd.to_numeric(base["market_score"], errors="coerce").fillna(0.0).ge(90.0)
        & pd.to_numeric(base["fixture_quality_score"], errors="coerce").fillna(0.0).ge(0.79)
        & pd.to_numeric(base["formation_pressure_score"], errors="coerce").fillna(0.0).ge(0.55)
        & pd.to_numeric(base["power_gap_directional_pressure_score"], errors="coerce").fillna(0.0).ge(0.62)
        & pd.to_numeric(base["starting_xi_quality_edge"], errors="coerce").fillna(0.0).le(-3.0)
    )
    elite = base[core_mask | conditional_mask].copy()
    if elite.empty:
        Path(f"{output_prefix}.csv").parent.mkdir(parents=True, exist_ok=True)
        elite.to_csv(f"{output_prefix}.csv", index=False)
        elite.to_csv(f"{output_prefix}__audit.csv", index=False)
        Path(f"{output_prefix}.md").write_text("# Portable Bookings Super-Elite\n\nNo rows matched.\n")
        Path(f"{output_prefix}__audit.md").write_text("# Portable Bookings Super-Elite Audit\n\nNo rows matched.\n")
        return elite, pd.DataFrame()

    elite["super_elite_priority"] = (
        0.55 * pd.to_numeric(elite["market_score"], errors="coerce").fillna(0.0)
        + 16.0 * pd.to_numeric(elite["fixture_quality_score"], errors="coerce").fillna(0.0)
        + 14.0 * pd.to_numeric(elite["formation_pressure_score"], errors="coerce").fillna(0.0)
        + 8.0 * pd.to_numeric(elite["power_gap_directional_pressure_score"], errors="coerce").fillna(0.0)
    )
    elite = (
        elite.sort_values(["fixture_key", "super_elite_priority", "market_score"], ascending=[True, False, False])
        .groupby("fixture_key", group_keys=False)
        .head(max_per_fixture)
        .reset_index(drop=True)
    )
    fixture_rank = (
        elite.groupby("fixture_key", as_index=False)
        .agg(
            fixture_super_score=("super_elite_priority", "sum"),
            fixture_rows=("player_name", "count"),
            top_score=("market_score", "max"),
            avg_quality=("fixture_quality_score", "mean"),
        )
        .sort_values(["fixture_super_score", "fixture_rows", "top_score"], ascending=[False, False, False])
        .head(max_fixtures)
    )
    elite = elite[elite["fixture_key"].isin(fixture_rank["fixture_key"])].copy()
    elite = elite.merge(fixture_rank[["fixture_key", "fixture_super_score"]], on="fixture_key", how="left")
    elite = elite.sort_values(
        ["fixture_super_score", "fixture_key", "super_elite_priority", "market_score"],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)
    elite.to_csv(f"{output_prefix}.csv", index=False)

    audit = (
        elite.groupby(["source_family", "portable_tier"], as_index=False)
        .agg(
            rows=("player_name", "count"),
            fixtures=("fixture_key", "nunique"),
            avg_score=("market_score", "mean"),
            avg_quality=("fixture_quality_score", "mean"),
            avg_pressure=("formation_pressure_score", "mean"),
        )
        .sort_values(["avg_score", "rows"], ascending=[False, False])
        .reset_index(drop=True)
    )
    audit.to_csv(f"{output_prefix}__audit.csv", index=False)

    lines = ["# Portable Bookings Super-Elite", "", f"- rows: {len(elite)} | fixtures: {elite['fixture_key'].nunique()}", ""]
    for fixture_key, sub in elite.groupby("fixture_key", sort=False):
        first = sub.iloc[0]
        lines.append(f"## {fixture_key}")
        lines.append(
            f"- {first['home_team_name']} vs {first['away_team_name']} | family score {first['fixture_super_score']:.1f} | quality {first['fixture_quality_score']:.3f}"
        )
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['portable_tier']}: {row['player_name']} ({row['team_name']}) | family={row['source_family']} | score={row['market_score']:.1f} | pressure={row['formation_pressure_score']:.3f}"
            )
        lines.append("")
    Path(f"{output_prefix}.md").write_text("\n".join(lines) + "\n")

    audit_lines = ["# Portable Bookings Super-Elite Audit", ""]
    for _, row in audit.iterrows():
        audit_lines.append(
            f"- {row['source_family']} | {row['portable_tier']}: rows={int(row['rows'])} | fixtures={int(row['fixtures'])} | avg_score={row['avg_score']:.1f} | avg_quality={row['avg_quality']:.3f}"
        )
    Path(f"{output_prefix}__audit.md").write_text("\n".join(audit_lines) + "\n")
    return elite, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tighten portable bookings elite into a super-elite subset.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--max-fixtures", type=int, default=24)
    parser.add_argument("--max-per-fixture", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    elite, audit = build_super_elite(args.input_csv, args.output_prefix, args.max_fixtures, args.max_per_fixture)
    print(f"WROTE: {args.output_prefix}.csv")
    print(f"rows: {len(elite)} | fixtures: {elite['fixture_key'].nunique() if not elite.empty else 0}")
