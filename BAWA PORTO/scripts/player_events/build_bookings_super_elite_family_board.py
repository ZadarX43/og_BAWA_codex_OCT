from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_bookings_specialist_family_pack import build_pack, _load_booked_actuals


def build_super_elite(
    input_csv: str,
    fixtures_csv: str,
    stats_csv: str,
    events_csv: str,
    output_prefix: str,
    max_per_fixture: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pack, _ = build_pack(
        input_csv=input_csv,
        fixtures_csv=fixtures_csv,
        stats_csv=stats_csv,
        events_csv=events_csv,
        output_prefix=f"{output_prefix}__tmp",
        max_per_fixture=2,
    )
    if pack.empty:
        empty = pd.DataFrame()
        Path(f"{output_prefix}.csv").parent.mkdir(parents=True, exist_ok=True)
        empty.to_csv(f"{output_prefix}.csv", index=False)
        empty.to_csv(f"{output_prefix}__audit.csv", index=False)
        Path(f"{output_prefix}.md").write_text("# Bookings Super-Elite Family Board\n\nNo rows matched.\n")
        Path(f"{output_prefix}__audit.md").write_text("# Bookings Super-Elite Family Audit\n\nNo rows matched.\n")
        return empty, empty

    elite = pack[
        pd.to_numeric(pack["fixture_quality_score"], errors="coerce").fillna(0.0).ge(0.78)
        & pd.to_numeric(pack["formation_pressure_score"], errors="coerce").fillna(0.0).ge(0.50)
        & pd.to_numeric(pack["market_score"], errors="coerce").fillna(0.0).ge(82.0)
        & pd.to_numeric(pack["starting_xi_quality_edge"], errors="coerce").fillna(0.0).le(2.5)
    ].copy()
    elite = (
        elite.sort_values(["fixture_key", "family_priority_score"], ascending=[True, False])
        .groupby("fixture_key", group_keys=False)
        .head(max_per_fixture)
        .reset_index(drop=True)
    )
    elite["priority_bucket"] = "BOOKINGS_SUPER_ELITE"
    elite.to_csv(f"{output_prefix}.csv", index=False)

    actuals = _load_booked_actuals(events_csv, stats_csv, fixtures_csv)
    audit = elite.merge(
        actuals[["fixture_key", "player_name", "booked_flag"]].drop_duplicates(),
        on=["fixture_key", "player_name"],
        how="left",
    )
    audit["booked_flag"] = audit["booked_flag"].fillna(0).astype(int)
    family_audit = (
        audit.groupby("source_family", as_index=False)
        .agg(
            rows=("player_name", "count"),
            fixtures=("fixture_key", "nunique"),
            hit_rate=("booked_flag", "mean"),
            avg_score=("market_score", "mean"),
            avg_fixture_quality=("fixture_quality_score", "mean"),
        )
        .sort_values(["hit_rate", "rows", "avg_score"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    family_audit.to_csv(f"{output_prefix}__audit.csv", index=False)

    lines = ["# Bookings Super-Elite Family Board", ""]
    for _, row in elite.iterrows():
        lines.append(
            f"- {row['fixture_key']}: {row['player_name']} ({row['team_name']}) | family={row['source_family']} | score={row['market_score']:.1f} | quality={row['fixture_quality_score']:.3f} | pressure={row['formation_pressure_score']:.3f}"
        )
    Path(f"{output_prefix}.md").write_text("\n".join(lines) + "\n")

    audit_lines = ["# Bookings Super-Elite Family Audit", ""]
    for _, row in family_audit.iterrows():
        audit_lines.append(
            f"- {row['source_family']}: hit_rate={row['hit_rate']:.3f} | rows={int(row['rows'])} | fixtures={int(row['fixtures'])} | avg_score={row['avg_score']:.1f}"
        )
    Path(f"{output_prefix}__audit.md").write_text("\n".join(audit_lines) + "\n")
    return elite, family_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a tighter bookings super-elite family board.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--fixtures-csv", required=True)
    parser.add_argument("--stats-csv", required=True)
    parser.add_argument("--events-csv", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--max-per-fixture", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    elite, audit = build_super_elite(
        input_csv=args.input_csv,
        fixtures_csv=args.fixtures_csv,
        stats_csv=args.stats_csv,
        events_csv=args.events_csv,
        output_prefix=args.output_prefix,
        max_per_fixture=args.max_per_fixture,
    )
    print(f"WROTE: {args.output_prefix}.csv")
    print(f"rows: {len(elite)} | families: {audit['source_family'].nunique() if not audit.empty else 0}")
