from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def audit_goal_sot_combo_board(board_csv: str, fixtures_csv: str, player_stats_csv: str, outdir: str, sample_size: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    board = pd.read_csv(board_csv)
    fixtures = pd.read_csv(fixtures_csv, usecols=["fixture_id", "fixture_key", "match_date"])
    actual = pd.read_csv(player_stats_csv, usecols=["fixture_id", "player_name", "goals", "shots_on_target"])
    actual = actual.merge(fixtures, on="fixture_id", how="left")
    actual["goals"] = pd.to_numeric(actual["goals"], errors="coerce").fillna(0.0)
    actual["shots_on_target"] = pd.to_numeric(actual["shots_on_target"], errors="coerce").fillna(0.0)
    actual["goal_hit_flag"] = (actual["goals"] >= 1).astype(int)
    actual["sot_hit_flag"] = (actual["shots_on_target"] >= 1).astype(int)
    board = board.merge(actual[["fixture_key", "player_name", "goal_hit_flag", "sot_hit_flag"]], on=["fixture_key", "player_name"], how="left")
    board[["goal_hit_flag", "sot_hit_flag"]] = board[["goal_hit_flag", "sot_hit_flag"]].fillna(0).astype(int)
    order = board[["fixture_key", "match_date"]].drop_duplicates().assign(match_date_ts=lambda x: pd.to_datetime(x["match_date"], errors="coerce")).sort_values(["match_date_ts", "fixture_key"], ascending=[False, False])
    keep = order.head(sample_size)["fixture_key"].tolist()
    board = board[board["fixture_key"].isin(keep)].copy()

    rows = []
    for fixture_key, group in board.groupby("fixture_key", sort=False):
        goal_rows = group[group["market"].eq("goal")]
        sot_rows = group[group["market"].eq("shots_on_target")]
        goal_hit = int(not goal_rows.empty and int(goal_rows["goal_hit_flag"].max()) == 1)
        sot_hit = int(not sot_rows.empty and int(sot_rows["sot_hit_flag"].max()) == 1)
        same_player_both = 0
        dual_group = group[group["same_player_dual_trigger_flag"].eq(1)]
        if not dual_group.empty:
            player_hits = dual_group.groupby("player_name", as_index=False).agg(goal_hit=("goal_hit_flag", "max"), sot_hit=("sot_hit_flag", "max"))
            same_player_both = int(((player_hits["goal_hit"] == 1) & (player_hits["sot_hit"] == 1)).any())
        rows.append({
            "fixture_key": fixture_key,
            "fixture_attacking_style_label": group["fixture_attacking_style_label"].iloc[0],
            "goal_pick_hit": goal_hit,
            "sot_pick_hit": sot_hit,
            "both_markets_hit": int(goal_hit == 1 and sot_hit == 1),
            "same_player_dual_both_hit": same_player_both,
            "fixture_attack_quality_score": float(group["fixture_attack_quality_score"].iloc[0]),
        })
    fixture_df = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        "fixtures_audited": len(fixture_df),
        "goal_pick_hit_rate": round(float(fixture_df["goal_pick_hit"].mean()), 4) if len(fixture_df) else 0.0,
        "sot_pick_hit_rate": round(float(fixture_df["sot_pick_hit"].mean()), 4) if len(fixture_df) else 0.0,
        "both_markets_hit_rate": round(float(fixture_df["both_markets_hit"].mean()), 4) if len(fixture_df) else 0.0,
        "same_player_dual_both_hit_rate": round(float(fixture_df["same_player_dual_both_hit"].mean()), 4) if len(fixture_df) else 0.0,
        "avg_attack_quality_score": round(float(fixture_df["fixture_attack_quality_score"].mean()), 4) if len(fixture_df) else 0.0,
    }])
    style_df = fixture_df.groupby("fixture_attacking_style_label", as_index=False).agg(
        fixtures=("fixture_key", "count"),
        goal_pick_hit_rate=("goal_pick_hit", "mean"),
        sot_pick_hit_rate=("sot_pick_hit", "mean"),
        both_markets_hit_rate=("both_markets_hit", "mean"),
        same_player_dual_both_hit_rate=("same_player_dual_both_hit", "mean"),
    ).sort_values(["same_player_dual_both_hit_rate", "both_markets_hit_rate", "fixtures"], ascending=[False, False, False])
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(board_csv).stem
    summary.to_csv(out / f"{stem}__goal_sot_audit_summary_last{sample_size}.csv", index=False)
    style_df.to_csv(out / f"{stem}__goal_sot_audit_style_last{sample_size}.csv", index=False)
    return summary, style_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a goal + SOT combo board.")
    parser.add_argument("--board-csv", required=True)
    parser.add_argument("--fixtures-csv", required=True)
    parser.add_argument("--player-stats-csv", required=True)
    parser.add_argument("--outdir", default="reports/player_events/combined_boards")
    parser.add_argument("--sample-size", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, style_df = audit_goal_sot_combo_board(args.board_csv, args.fixtures_csv, args.player_stats_csv, args.outdir, args.sample_size)
    print("WROTE:", args.outdir)
    print(summary.to_string(index=False))
    print(style_df.to_string(index=False))


if __name__ == "__main__":
    main()
