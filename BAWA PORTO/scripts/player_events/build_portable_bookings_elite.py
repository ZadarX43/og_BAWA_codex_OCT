from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_bookings_specialist_family_pack import DEFAULT_FAMILY_FORMATIONS, _ensure_fixture_quality, _load_booked_actuals
from render_yellow_card_report import _build_bpi
from run_greenlist_specialist_family_batch import BATCHES


REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = REPO_ROOT / "data_sources" / "api_football" / "features" / "player_events"
NORMALIZED_DIR = REPO_ROOT / "data_sources" / "api_football" / "normalized"

BOOKINGS_PORTABLE_MAP = {
    "4231v433": "CORE_PORTABLE_BOOKINGS",
    "4231v442": "CONDITIONAL_PORTABLE_BOOKINGS",
}


def _family_for_label(label: str) -> str:
    for family, labels in DEFAULT_FAMILY_FORMATIONS.items():
        if label in labels:
            return family
    return "OTHER"


def _merge_batch_frames(batch_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    input_frames = []
    fixture_frames = []
    stats_frames = []
    event_frames = []
    for batch in batch_names:
        input_path = FEATURES_DIR / f"player_events_fixture_input__{batch.upper()}__2024.csv"
        if input_path.exists():
            df = pd.read_csv(input_path, low_memory=False)
            if not df.empty:
                df["source_batch"] = batch
                input_frames.append(df)
        for league in BATCHES.get(batch, []):
            fx = NORMALIZED_DIR / f"fixtures_master__{league}__2024.csv"
            st = NORMALIZED_DIR / f"match_player_stats__{league}__2024.csv"
            ev = NORMALIZED_DIR / f"match_events__{league}__2024.csv"
            if fx.exists():
                fixture_frames.append(pd.read_csv(fx, low_memory=False))
            if st.exists():
                stats_frames.append(pd.read_csv(st, low_memory=False))
            if ev.exists():
                event_frames.append(pd.read_csv(ev, low_memory=False))
    inputs = pd.concat(input_frames, ignore_index=True) if input_frames else pd.DataFrame()
    fixtures = pd.concat(fixture_frames, ignore_index=True) if fixture_frames else pd.DataFrame()
    stats = pd.concat(stats_frames, ignore_index=True) if stats_frames else pd.DataFrame()
    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    return inputs, fixtures, stats, events


def build_elite(batch_names: list[str], output_prefix: str, max_per_fixture: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    inputs, fixtures, stats, events = _merge_batch_frames(batch_names)
    if inputs.empty:
        empty = pd.DataFrame()
        Path(f"{output_prefix}.csv").parent.mkdir(parents=True, exist_ok=True)
        empty.to_csv(f"{output_prefix}.csv", index=False)
        empty.to_csv(f"{output_prefix}__audit.csv", index=False)
        Path(f"{output_prefix}.md").write_text("# Portable Bookings Elite\n\nNo rows matched.\n")
        Path(f"{output_prefix}__audit.md").write_text("# Portable Bookings Elite Audit\n\nNo rows matched.\n")
        return empty, empty

    base = _ensure_fixture_quality(inputs)
    scored = _build_bpi(base).copy()
    scored["market"] = "yellow_cards"
    scored["market_score"] = scored["booking_probability_index"]
    scored["market_confidence"] = scored["confidence_label"]
    scored["source_family"] = scored["formation_matchup_label"].astype(str).map(_family_for_label)
    scored["portable_tier"] = scored["source_family"].map(BOOKINGS_PORTABLE_MAP).fillna("")
    scored = scored[scored["portable_tier"].astype(str).ne("")].copy()
    if scored.empty:
        Path(f"{output_prefix}.csv").parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(f"{output_prefix}.csv", index=False)
        scored.to_csv(f"{output_prefix}__audit.csv", index=False)
        Path(f"{output_prefix}.md").write_text("# Portable Bookings Elite\n\nNo rows matched.\n")
        Path(f"{output_prefix}__audit.md").write_text("# Portable Bookings Elite Audit\n\nNo rows matched.\n")
        return scored, pd.DataFrame()

    scored = scored[
        scored["fixture_style_label"].astype("string").str.upper().isin(["AGGRESSIVE_BOTH", "MIDFIELD_GRIND"])
        & pd.to_numeric(scored["fixture_quality_score"], errors="coerce").fillna(0.0).ge(0.76)
        & pd.to_numeric(scored["booking_probability_index"], errors="coerce").fillna(0.0).ge(58.0)
        & pd.to_numeric(scored["formation_pressure_score"], errors="coerce").fillna(0.0).ge(0.46)
        & (
            pd.to_numeric(scored["starting_xi_quality_edge"], errors="coerce").fillna(0.0).le(-1.5)
            | pd.to_numeric(scored["weaker_side_under_pressure_flag"], errors="coerce").fillna(0).ge(1)
        )
    ].copy()

    core_mask = scored["portable_tier"].eq("CORE_PORTABLE_BOOKINGS")
    conditional_mask = scored["portable_tier"].eq("CONDITIONAL_PORTABLE_BOOKINGS")
    scored = scored[
        (
            core_mask
            & pd.to_numeric(scored["booking_probability_index"], errors="coerce").fillna(0.0).ge(62.0)
            & pd.to_numeric(scored["starting_xi_quality_edge"], errors="coerce").fillna(0.0).le(0.5)
        )
        | (
            conditional_mask
            & pd.to_numeric(scored["booking_probability_index"], errors="coerce").fillna(0.0).ge(72.0)
            & pd.to_numeric(scored["starting_xi_quality_edge"], errors="coerce").fillna(0.0).le(-3.0)
            & pd.to_numeric(scored["power_gap_directional_pressure_score"], errors="coerce").fillna(0.0).ge(0.40)
        )
    ].copy()

    scored["elite_priority_score"] = (
        0.50 * pd.to_numeric(scored["booking_probability_index"], errors="coerce").fillna(0.0)
        + 20.0 * pd.to_numeric(scored["formation_pressure_score"], errors="coerce").fillna(0.0)
        + 10.0 * pd.to_numeric(scored["fixture_quality_score"], errors="coerce").fillna(0.0)
        + 8.0 * pd.to_numeric(scored["power_gap_directional_pressure_score"], errors="coerce").fillna(0.0)
    )

    scored = (
        scored.sort_values(
            ["fixture_key", "portable_tier", "elite_priority_score", "market_score"],
            ascending=[True, True, False, False],
        )
        .groupby("fixture_key", group_keys=False)
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
        "source_batch",
        "source_family",
        "portable_tier",
        "formation_matchup_label",
        "fixture_style_label",
        "fixture_quality_score",
        "formation_pressure_score",
        "power_gap_directional_pressure_score",
        "starting_xi_quality_edge",
        "player_quality_score_l5",
        "booking_probability_index",
        "market_score",
        "market_confidence",
        "elite_priority_score",
    ]
    elite = scored[keep_cols].copy()
    elite.to_csv(f"{output_prefix}.csv", index=False)

    actuals = _load_booked_actuals(
        events_csv=str((NORMALIZED_DIR / "__tmp_events__portable_bookings_elite.csv")),
        stats_csv=str((NORMALIZED_DIR / "__tmp_stats__portable_bookings_elite.csv")),
        fixtures_csv=str((NORMALIZED_DIR / "__tmp_fixtures__portable_bookings_elite.csv")),
    ) if False else None

    fixtures.to_csv(NORMALIZED_DIR / "__tmp_fixtures__portable_bookings_elite.csv", index=False)
    stats.to_csv(NORMALIZED_DIR / "__tmp_stats__portable_bookings_elite.csv", index=False)
    events.to_csv(NORMALIZED_DIR / "__tmp_events__portable_bookings_elite.csv", index=False)
    actuals = _load_booked_actuals(
        events_csv=str(NORMALIZED_DIR / "__tmp_events__portable_bookings_elite.csv"),
        stats_csv=str(NORMALIZED_DIR / "__tmp_stats__portable_bookings_elite.csv"),
        fixtures_csv=str(NORMALIZED_DIR / "__tmp_fixtures__portable_bookings_elite.csv"),
    )
    audit = elite.merge(actuals[["fixture_key", "player_name", "booked_flag"]].drop_duplicates(), on=["fixture_key", "player_name"], how="left")
    audit["booked_flag"] = audit["booked_flag"].fillna(0).astype(int)
    family_audit = (
        audit.groupby(["source_family", "portable_tier"], as_index=False)
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

    lines = ["# Portable Bookings Elite", "", f"- batches: {', '.join(batch_names)}", f"- rows: {len(elite)} | fixtures: {elite['fixture_key'].nunique() if not elite.empty else 0}", ""]
    for family, sub in elite.groupby("source_family", sort=False):
        lines.append(f"## {family}")
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['portable_tier']}: {row['fixture_key']} | {row['player_name']} ({row['team_name']}) | score={row['market_score']:.1f} | quality={row['fixture_quality_score']:.3f} | edge={row['starting_xi_quality_edge']:.1f}"
            )
        lines.append("")
    Path(f"{output_prefix}.md").write_text("\n".join(lines) + "\n")

    audit_lines = ["# Portable Bookings Elite Audit", ""]
    for _, row in family_audit.iterrows():
        audit_lines.append(
            f"- {row['source_family']} | {row['portable_tier']}: hit_rate={row['hit_rate']:.3f} | rows={int(row['rows'])} | fixtures={int(row['fixtures'])} | avg_score={row['avg_score']:.1f}"
        )
    Path(f"{output_prefix}__audit.md").write_text("\n".join(audit_lines) + "\n")
    return elite, family_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a portable bookings elite lane using the specialist family deploy logic.")
    parser.add_argument("--batch-names", default="greenlist_batch1,greenlist_batch2,greenlist_batch3,greenlist_batch4,greenlist_batch5")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--max-per-fixture", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batch_names = [x.strip() for x in args.batch_names.split(",") if x.strip()]
    elite, audit = build_elite(batch_names=batch_names, output_prefix=args.output_prefix, max_per_fixture=args.max_per_fixture)
    print(f"WROTE: {args.output_prefix}.csv")
    print(f"rows: {len(elite)} | fixtures: {elite['fixture_key'].nunique() if not elite.empty else 0} | families: {audit['source_family'].nunique() if not audit.empty else 0}")
