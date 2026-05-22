from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_bookings_specialist_family_pack import _load_booked_actuals
from run_greenlist_specialist_family_batch import BATCHES


REPO_ROOT = Path(__file__).resolve().parents[2]
NORMALIZED_DIR = REPO_ROOT / "data_sources" / "api_football" / "normalized"

CONTACT_STAT_MAP = {
    "fouls_committed": "fouls_committed",
    "tackles": "tackles",
}


def _merge_batch_actual_sources(batch_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fixtures_frames = []
    stats_frames = []
    events_frames = []
    for batch in batch_names:
        for league in BATCHES.get(batch, []):
            fx = NORMALIZED_DIR / f"fixtures_master__{league}__2024.csv"
            st = NORMALIZED_DIR / f"match_player_stats__{league}__2024.csv"
            ev = NORMALIZED_DIR / f"match_events__{league}__2024.csv"
            if fx.exists():
                fixtures_frames.append(pd.read_csv(fx, low_memory=False))
            if st.exists():
                stats_frames.append(pd.read_csv(st, low_memory=False))
            if ev.exists():
                events_frames.append(pd.read_csv(ev, low_memory=False))
    fixtures = pd.concat(fixtures_frames, ignore_index=True) if fixtures_frames else pd.DataFrame()
    stats = pd.concat(stats_frames, ignore_index=True) if stats_frames else pd.DataFrame()
    events = pd.concat(events_frames, ignore_index=True) if events_frames else pd.DataFrame()
    return fixtures, stats, events


def _build_contact_actuals(stats: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    if stats.empty or fixtures.empty:
        return pd.DataFrame(columns=["fixture_key", "player_name", "market", "actual_stat", "actual_hit"])
    fixture_lookup = fixtures[["fixture_id", "fixture_key"]].drop_duplicates()
    merged = stats.merge(fixture_lookup, on="fixture_id", how="left")
    frames = []
    for market, stat_col in CONTACT_STAT_MAP.items():
        if stat_col not in merged.columns:
            continue
        sub = merged[["fixture_key", "player_name", stat_col]].copy()
        sub["market"] = market
        sub["actual_stat"] = pd.to_numeric(sub[stat_col], errors="coerce").fillna(0.0)
        sub["actual_hit"] = (sub["actual_stat"] >= 1.0).astype(int)
        frames.append(sub[["fixture_key", "player_name", "market", "actual_stat", "actual_hit"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["fixture_key", "player_name", "market", "actual_stat", "actual_hit"])


def build_audit(contact_csv: str, bookings_csv: str, batch_names: list[str], output_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    contact = pd.read_csv(contact_csv, low_memory=False)
    bookings = pd.read_csv(bookings_csv, low_memory=False)
    fixtures, stats, events = _merge_batch_actual_sources(batch_names)

    contact_board = pd.DataFrame()
    if not contact.empty:
        contact_board = contact.copy()
        contact_board["review_family"] = contact_board["source_family_tag"].astype(str)
        contact_board["review_tier"] = contact_board["preset_tier"].astype(str)
        contact_board["score"] = pd.to_numeric(contact_board["market_score"], errors="coerce").fillna(0.0)
        if "tactical_role" not in contact_board.columns:
            contact_board["tactical_role"] = "UNKNOWN"
        contact_board = contact_board[
            [
                "fixture_key",
                "team_name",
                "player_name",
                "market",
                "tactical_role",
                "review_family",
                "review_tier",
                "score",
                "fixture_quality_score",
                "formation_pressure_score",
            ]
        ].copy()

    bookings_board = pd.DataFrame()
    if not bookings.empty:
        bookings_board = bookings.copy()
        bookings_board["market"] = "yellow_cards"
        bookings_board["review_family"] = bookings_board["source_family"].astype(str)
        bookings_board["review_tier"] = bookings_board["portable_tier"].astype(str)
        bookings_board["score"] = pd.to_numeric(bookings_board["market_score"], errors="coerce").fillna(0.0)
        bookings_board = bookings_board[
            [
                "fixture_key",
                "team_name",
                "player_name",
                "market",
                "tactical_role",
                "review_family",
                "review_tier",
                "score",
                "fixture_quality_score",
                "formation_pressure_score",
            ]
        ].copy()

    board = pd.concat([contact_board, bookings_board], ignore_index=True)
    if board.empty:
        empty = pd.DataFrame()
        Path(f"{output_prefix}.csv").parent.mkdir(parents=True, exist_ok=True)
        empty.to_csv(f"{output_prefix}.csv", index=False)
        empty.to_csv(f"{output_prefix}__team_market.csv", index=False)
        Path(f"{output_prefix}.md").write_text("# Team x Family x Role Audit\n\nNo rows matched.\n")
        return empty, empty

    contact_actuals = _build_contact_actuals(stats, fixtures)
    fixtures.to_csv(NORMALIZED_DIR / "__tmp_team_role_audit_fixtures.csv", index=False)
    stats.to_csv(NORMALIZED_DIR / "__tmp_team_role_audit_stats.csv", index=False)
    events.to_csv(NORMALIZED_DIR / "__tmp_team_role_audit_events.csv", index=False)
    bookings_actuals = _load_booked_actuals(
        events_csv=str(NORMALIZED_DIR / "__tmp_team_role_audit_events.csv"),
        stats_csv=str(NORMALIZED_DIR / "__tmp_team_role_audit_stats.csv"),
        fixtures_csv=str(NORMALIZED_DIR / "__tmp_team_role_audit_fixtures.csv"),
    )
    bookings_actuals["market"] = "yellow_cards"
    bookings_actuals["actual_hit"] = bookings_actuals["booked_flag"].fillna(0).astype(int)
    bookings_actuals["actual_stat"] = bookings_actuals["actual_hit"].astype(float)
    bookings_actuals = bookings_actuals[["fixture_key", "player_name", "market", "actual_stat", "actual_hit"]]

    actuals = pd.concat([contact_actuals, bookings_actuals], ignore_index=True)
    audit = board.merge(actuals, on=["fixture_key", "player_name", "market"], how="left")
    audit["actual_stat"] = pd.to_numeric(audit["actual_stat"], errors="coerce").fillna(0.0)
    audit["actual_hit"] = pd.to_numeric(audit["actual_hit"], errors="coerce").fillna(0).astype(int)

    team_family_role = (
        audit.groupby(["team_name", "review_family", "tactical_role"], as_index=False)
        .agg(
            rows=("player_name", "count"),
            fixtures=("fixture_key", "nunique"),
            markets=("market", lambda s: "|".join(sorted(pd.Series(s).astype(str).unique()))),
            hit_rate=("actual_hit", "mean"),
            avg_score=("score", "mean"),
            avg_quality=("fixture_quality_score", "mean"),
            avg_pressure=("formation_pressure_score", "mean"),
        )
        .sort_values(["hit_rate", "rows", "avg_score"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    team_market = (
        audit.groupby(["team_name", "review_family", "tactical_role", "market"], as_index=False)
        .agg(
            rows=("player_name", "count"),
            fixtures=("fixture_key", "nunique"),
            hit_rate=("actual_hit", "mean"),
            avg_score=("score", "mean"),
        )
        .sort_values(["market", "hit_rate", "rows"], ascending=[True, False, False])
        .reset_index(drop=True)
    )
    team_family_role.to_csv(f"{output_prefix}.csv", index=False)
    team_market.to_csv(f"{output_prefix}__team_market.csv", index=False)

    lines = ["# Team x Family x Role Audit", "", f"- batches: {', '.join(batch_names)}", ""]
    top_overall = team_family_role.head(20).copy()
    if not top_overall.empty:
        lines.append("## Top Team x Family x Role Patterns")
        for _, row in top_overall.iterrows():
            lines.append(
                f"- {row['team_name']} | {row['review_family']} | {row['tactical_role']}: hit_rate={row['hit_rate']:.3f} | rows={int(row['rows'])} | fixtures={int(row['fixtures'])} | markets={row['markets']}"
            )
        lines.append("")
    for market in ["yellow_cards", "fouls_committed", "tackles"]:
        sub = team_market[team_market["market"].eq(market)].head(10)
        lines.append(f"## Top {market} Patterns")
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['team_name']} | {row['review_family']} | {row['tactical_role']}: hit_rate={row['hit_rate']:.3f} | rows={int(row['rows'])}"
            )
        lines.append("")
    Path(f"{output_prefix}.md").write_text("\n".join(lines) + "\n")

    for name in ["__tmp_team_role_audit_fixtures.csv", "__tmp_team_role_audit_stats.csv", "__tmp_team_role_audit_events.csv"]:
        p = NORMALIZED_DIR / name
        if p.exists():
            p.unlink()
    return team_family_role, team_market


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit team x family x role patterns across contact and bookings elite boards.")
    parser.add_argument("--contact-csv", required=True)
    parser.add_argument("--bookings-csv", required=True)
    parser.add_argument("--batch-names", default="greenlist_batch1,greenlist_batch2,greenlist_batch3,greenlist_batch4,greenlist_batch5")
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batch_names = [x.strip() for x in args.batch_names.split(",") if x.strip()]
    team_role, team_market = build_audit(args.contact_csv, args.bookings_csv, batch_names, args.output_prefix)
    print(f"WROTE: {args.output_prefix}.csv")
    print(f"rows: {len(team_role)} | team_market_rows: {len(team_market)}")
