from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from render_yellow_card_report import _build_bpi


DEFAULT_FAMILY_FORMATIONS = {
    "4231v442": {"4-2-3-1 vs 4-4-2", "4-4-2 vs 4-2-3-1"},
    "4231v433": {"4-2-3-1 vs 4-3-3", "4-3-3 vs 4-2-3-1"},
    "3421v4231": {"3-4-2-1 vs 4-2-3-1", "4-2-3-1 vs 3-4-2-1"},
}


def _ensure_fixture_quality(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "fixture_quality_score" not in out.columns:
        out["fixture_quality_score"] = (
            0.30 * pd.to_numeric(out.get("og_goal_environment_score", 0.0), errors="coerce").fillna(0.0)
            + 0.25 * pd.to_numeric(out.get("fixture_foul_density_score", 0.0), errors="coerce").fillna(0.0)
            + 0.20 * pd.to_numeric(out.get("fixture_attack_pressure_score", 0.0), errors="coerce").fillna(0.0)
            + 0.15 * pd.to_numeric(out.get("fixture_tackle_density_score", 0.0), errors="coerce").fillna(0.0)
            + 0.10 * pd.to_numeric(out.get("formation_pressure_score", 0.0), errors="coerce").fillna(0.0)
        ).clip(lower=0.0, upper=1.0)
    return out


def _load_booked_actuals(events_csv: str, stats_csv: str, fixtures_csv: str) -> pd.DataFrame:
    events = pd.read_csv(events_csv, low_memory=False)
    stats = pd.read_csv(stats_csv, low_memory=False, usecols=["fixture_id", "player_id", "player_name"]).drop_duplicates()
    fixtures = pd.read_csv(fixtures_csv, low_memory=False, usecols=["fixture_id", "fixture_key"])
    booked = events[
        (events["event_type"].astype("string").str.lower() == "card")
        & (events["event_detail"].astype("string").str.contains("yellow", case=False, na=False))
    ].copy()
    booked = booked.merge(fixtures, on="fixture_id", how="left").merge(stats, on=["fixture_id", "player_id"], how="left")
    booked = booked[["fixture_key", "player_name"]].dropna().drop_duplicates()
    booked["booked_flag"] = 1
    return booked


def build_pack(input_csv: str, fixtures_csv: str, stats_csv: str, events_csv: str, output_prefix: str, max_per_fixture: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.read_csv(input_csv, low_memory=False)
    if base.empty:
        empty = pd.DataFrame()
        Path(f"{output_prefix}.csv").parent.mkdir(parents=True, exist_ok=True)
        empty.to_csv(f"{output_prefix}.csv", index=False)
        empty.to_csv(f"{output_prefix}__audit.csv", index=False)
        Path(f"{output_prefix}.md").write_text("# Bookings Specialist Family Pack\n\nNo rows matched.\n")
        Path(f"{output_prefix}__audit.md").write_text("# Bookings Specialist Family Audit\n\nNo rows matched.\n")
        return empty, empty

    base = _ensure_fixture_quality(base)
    scored = _build_bpi(base).copy()
    scored["market"] = "yellow_cards"
    scored["market_score"] = scored["booking_probability_index"]
    scored["market_confidence"] = scored["confidence_label"]
    scored["source_family"] = "OTHER"
    for family, labels in DEFAULT_FAMILY_FORMATIONS.items():
        mask = scored["formation_matchup_label"].isin(labels)
        scored.loc[mask, "source_family"] = family
    scored = scored[scored["source_family"].isin(DEFAULT_FAMILY_FORMATIONS)].copy()
    scored = scored[
        scored["fixture_style_label"].astype("string").str.upper().isin(["AGGRESSIVE_BOTH", "MIDFIELD_GRIND"])
        & pd.to_numeric(scored["fixture_quality_score"], errors="coerce").fillna(0.0).ge(0.72)
        & pd.to_numeric(scored["booking_probability_index"], errors="coerce").fillna(0.0).ge(48.0)
        & (
            pd.to_numeric(scored["starting_xi_quality_edge"], errors="coerce").fillna(0.0).le(-1.0)
            | pd.to_numeric(scored["weaker_side_under_pressure_flag"], errors="coerce").fillna(0).ge(1)
        )
    ].copy()

    scored["family_priority_score"] = (
        0.45 * pd.to_numeric(scored["booking_probability_index"], errors="coerce").fillna(0.0)
        + 18.0 * pd.to_numeric(scored["formation_pressure_score"], errors="coerce").fillna(0.0)
        + 10.0 * pd.to_numeric(scored["power_gap_directional_pressure_score"], errors="coerce").fillna(0.0)
        + 7.0 * pd.to_numeric(scored["fixture_quality_score"], errors="coerce").fillna(0.0)
    )
    scored = (
        scored.sort_values(["fixture_key", "family_priority_score"], ascending=[True, False])
        .groupby(["fixture_key", "source_family"], group_keys=False)
        .head(max_per_fixture)
        .reset_index(drop=True)
    )
    keep_cols = [
        "fixture_key",
        "match_date",
        "competition",
        "league",
        "home_team_name",
        "away_team_name",
        "team_name",
        "player_name",
        "tactical_role",
        "source_family",
        "formation_matchup_label",
        "fixture_style_label",
        "fixture_quality_score",
        "formation_pressure_score",
        "starting_xi_quality_edge",
        "player_quality_score_l5",
        "booking_probability_index",
        "market_score",
        "market_confidence",
        "family_priority_score",
    ]
    pack = scored[keep_cols].copy()
    pack.to_csv(f"{output_prefix}.csv", index=False)

    actuals = _load_booked_actuals(events_csv, stats_csv, fixtures_csv)
    audit = pack.merge(actuals, on=["fixture_key", "player_name"], how="left")
    audit["booked_flag"] = audit["booked_flag"].fillna(0).astype(int)
    family_audit = (
        audit.groupby("source_family", as_index=False)
        .agg(
            rows=("player_name", "count"),
            fixtures=("fixture_key", "nunique"),
            hit_rate=("booked_flag", "mean"),
            avg_score=("market_score", "mean"),
            avg_fixture_quality=("fixture_quality_score", "mean"),
            strongest_fixture=("fixture_key", "first"),
        )
        .sort_values(["hit_rate", "rows", "avg_score"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    family_audit.to_csv(f"{output_prefix}__audit.csv", index=False)

    lines = ["# Bookings Specialist Family Pack", ""]
    for family, sub in pack.groupby("source_family", sort=False):
        lines.append(f"## {family}")
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['fixture_key']}: {row['player_name']} ({row['team_name']}) | score={row['market_score']:.1f} | quality={row['fixture_quality_score']:.3f} | edge={row['starting_xi_quality_edge']:.1f}"
            )
        lines.append("")
    Path(f"{output_prefix}.md").write_text("\n".join(lines) + "\n")

    audit_lines = ["# Bookings Specialist Family Audit", ""]
    for _, row in family_audit.iterrows():
        audit_lines.append(
            f"- {row['source_family']}: hit_rate={row['hit_rate']:.3f} | rows={int(row['rows'])} | fixtures={int(row['fixtures'])} | avg_score={row['avg_score']:.1f} | best_fixture={row['strongest_fixture']}"
        )
    Path(f"{output_prefix}__audit.md").write_text("\n".join(audit_lines) + "\n")
    return pack, family_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a bookings specialist family pack across the strongest formation families.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--fixtures-csv", required=True)
    parser.add_argument("--stats-csv", required=True)
    parser.add_argument("--events-csv", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--max-per-fixture", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pack, audit = build_pack(
        args.input_csv,
        args.fixtures_csv,
        args.stats_csv,
        args.events_csv,
        args.output_prefix,
        max_per_fixture=args.max_per_fixture,
    )
    print(f"WROTE: {args.output_prefix}.csv")
    print(f"pack_rows: {len(pack)} | families: {audit['source_family'].nunique() if not audit.empty else 0}")
