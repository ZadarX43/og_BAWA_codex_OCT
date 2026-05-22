from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LEAGUE_TAGS = {
    "Serie A": "Italy_Serie_A",
    "La Liga": "Spain_La_Liga",
    "UEFA Europa League": "Europa_League",
}


def _league_tables(league_name: str) -> tuple[str, str, str]:
    tag = LEAGUE_TAGS[league_name]
    base = Path("data_sources/api_football/normalized")
    fixtures = str(base / f"fixtures_master__{tag}__2024.csv")
    events = str(base / f"match_events__{tag}__2024.csv")
    player_stats = str(base / f"match_player_stats__{tag}__2024.csv")
    return fixtures, events, player_stats


def _load_actuals_for_league(league_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    fixtures_csv, events_csv, player_stats_csv = _league_tables(league_name)
    fixtures = pd.read_csv(fixtures_csv, usecols=["fixture_id", "fixture_key"])
    ps = pd.read_csv(
        player_stats_csv,
        usecols=["fixture_id", "player_id", "player_name", "fouls_committed", "tackles"],
    ).drop_duplicates(subset=["fixture_id", "player_id", "player_name"])
    events = pd.read_csv(events_csv, usecols=["fixture_id", "player_id", "event_type", "event_detail"])

    booked = events[
        (events["event_type"].astype("string").str.lower() == "card")
        & (events["event_detail"].astype("string").str.contains("yellow", case=False, na=False))
    ].merge(fixtures, on="fixture_id", how="left").merge(ps[["fixture_id", "player_id", "player_name"]], on=["fixture_id", "player_id"], how="left")
    booked = booked[["fixture_key", "player_name"]].dropna().drop_duplicates()
    booked["actual_booked_flag"] = 1

    fouls = ps.merge(fixtures, on="fixture_id", how="left")
    fouls["fouls_committed"] = pd.to_numeric(fouls["fouls_committed"], errors="coerce").fillna(0.0)
    fouls["tackles"] = pd.to_numeric(fouls["tackles"], errors="coerce").fillna(0.0)
    fouls["actual_high_foul_flag"] = (fouls["fouls_committed"] >= 2).astype(int)
    fouls = fouls[["fixture_key", "player_name", "fouls_committed", "tackles", "actual_high_foul_flag"]]
    return booked, fouls


def audit_combined_board(board_csv: str, output_md: str, output_csv: str, sample_size: int = 25) -> tuple[pd.DataFrame, pd.DataFrame]:
    board = pd.read_csv(board_csv)
    frames = []
    for league_name in sorted(board["league"].dropna().astype(str).unique()):
        if league_name not in LEAGUE_TAGS:
            continue
        booked, fouls = _load_actuals_for_league(league_name)
        part = board[board["league"].astype(str).eq(league_name)].copy()
        part = part.merge(booked, on=["fixture_key", "player_name"], how="left")
        part = part.merge(fouls, on=["fixture_key", "player_name"], how="left")
        frames.append(part)
    combined = pd.concat(frames, ignore_index=True) if frames else board.copy()
    combined["actual_booked_flag"] = combined.get("actual_booked_flag", 0).fillna(0).astype(int)
    combined["actual_high_foul_flag"] = combined.get("actual_high_foul_flag", 0).fillna(0).astype(int)
    combined["fouls_committed"] = combined.get("fouls_committed", 0).fillna(0.0)
    combined["tackles"] = combined.get("tackles", 0).fillna(0.0)

    fixture_order = (
        combined[["fixture_key", "match_date"]]
        .drop_duplicates()
        .assign(match_date_ts=lambda x: pd.to_datetime(x["match_date"], errors="coerce"))
        .sort_values(["match_date_ts", "fixture_key"], ascending=[False, False])
        .head(sample_size)
    )
    keep = fixture_order["fixture_key"].tolist()
    sample = combined[combined["fixture_key"].isin(keep)].copy()

    rows = []
    for fixture_key, group in sample.groupby("fixture_key", sort=False):
        yc = group[group["market"].eq("yellow_card")].copy()
        fouls = group[group["market"].eq("fouls_committed")].copy()
        actual_booked = sample[(sample["fixture_key"].eq(fixture_key)) & (sample["actual_booked_flag"].eq(1))]["player_name"].dropna().astype(str).drop_duplicates()
        actual_high_foul = sample[(sample["fixture_key"].eq(fixture_key)) & (sample["actual_high_foul_flag"].eq(1))]["player_name"].dropna().astype(str).drop_duplicates()
        row = {
            "fixture_key": fixture_key,
            "match_date": group["match_date"].iloc[0],
            "league": group["league"].iloc[0],
            "home_team_name": group["home_team_name"].iloc[0],
            "away_team_name": group["away_team_name"].iloc[0],
            "fixture_quality_score": group["fixture_quality_score"].iloc[0],
            "og_battle_on_score": group["og_battle_on_score"].iloc[0],
            "fixture_quality_reason_codes": group["fixture_quality_reason_codes"].iloc[0],
            "yc_pick": yc["player_name"].iloc[0] if not yc.empty else "",
            "yc_pick_hit": int(not yc.empty and int(yc["actual_booked_flag"].fillna(0).iloc[0]) == 1),
            "fouls_pick": fouls["player_name"].iloc[0] if not fouls.empty else "",
            "fouls_pick_hit": int(not fouls.empty and int(fouls["actual_high_foul_flag"].fillna(0).iloc[0]) == 1),
            "yc_dual_trigger_flag": int(not yc.empty and int(yc["same_player_dual_trigger_flag"].iloc[0]) == 1),
            "fouls_dual_trigger_flag": int(not fouls.empty and int(fouls["same_player_dual_trigger_flag"].iloc[0]) == 1),
            "actual_booked_names": " | ".join(actual_booked.tolist()),
            "actual_high_foul_names": " | ".join(actual_high_foul.tolist()),
        }
        rows.append(row)

    fixture_audit = pd.DataFrame(rows).sort_values(["match_date", "fixture_key"], ascending=[False, False])
    summary = pd.DataFrame(
        [
            {
                "fixtures_audited": len(fixture_audit),
                "yc_pick_hit_rate": round(float(fixture_audit["yc_pick_hit"].mean()), 4) if len(fixture_audit) else 0.0,
                "fouls_pick_hit_rate": round(float(fixture_audit["fouls_pick_hit"].mean()), 4) if len(fixture_audit) else 0.0,
                "dual_trigger_yc_share": round(float(fixture_audit["yc_dual_trigger_flag"].mean()), 4) if len(fixture_audit) else 0.0,
                "dual_trigger_fouls_share": round(float(fixture_audit["fouls_dual_trigger_flag"].mean()), 4) if len(fixture_audit) else 0.0,
                "avg_fixture_quality_score": round(float(fixture_audit["fixture_quality_score"].mean()), 4) if len(fixture_audit) else 0.0,
                "avg_og_battle_on_score": round(float(fixture_audit["og_battle_on_score"].mean()), 4) if len(fixture_audit) else 0.0,
            }
        ]
    )

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    fixture_audit.to_csv(output_csv, index=False)
    md_path = Path(output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Internal Combined Board Manual Audit",
        "",
        "## Summary",
    ]
    for col, val in summary.iloc[0].items():
        lines.append(f"- {col}: {val}")
    lines.extend(["", "## Fixture Audit"])
    for row in fixture_audit.itertuples(index=False):
        lines.extend(
            [
                f"### {row.fixture_key}",
                f"- fixture: {row.home_team_name} vs {row.away_team_name} ({row.league})",
                f"- fixture quality: {row.fixture_quality_score:.3f}",
                f"- battle on score: {row.og_battle_on_score:.3f}",
                f"- reasons: {row.fixture_quality_reason_codes}",
                f"- yellow-card pick: {row.yc_pick or 'none'} | hit={row.yc_pick_hit} | dual_trigger={row.yc_dual_trigger_flag}",
                f"- fouls pick: {row.fouls_pick or 'none'} | hit={row.fouls_pick_hit} | dual_trigger={row.fouls_dual_trigger_flag}",
                f"- actual booked: {row.actual_booked_names or 'none'}",
                f"- actual high-foul names: {row.actual_high_foul_names or 'none'}",
                "",
            ]
        )
    md_path.write_text("\n".join(lines))
    return summary, fixture_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual audit pack for the ultra-strict combined internal board.")
    parser.add_argument("--board-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--sample-size", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, audit = audit_combined_board(args.board_csv, args.output_md, args.output_csv, args.sample_size)
    print(summary.to_string(index=False))
    print(audit.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
