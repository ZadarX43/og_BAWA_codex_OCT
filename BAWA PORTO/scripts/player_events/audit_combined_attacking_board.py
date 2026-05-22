from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def audit_combined_attacking_board(
    board_csv: str,
    fixtures_csv: str,
    player_stats_csv: str,
    outdir: str,
    sample_size: int = 100,
    shot_threshold: int = 2,
    sot_threshold: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    board = pd.read_csv(board_csv)
    fixtures = pd.read_csv(fixtures_csv, usecols=["fixture_id", "fixture_key", "match_date"])
    actual = pd.read_csv(player_stats_csv, usecols=["fixture_id", "player_name", "shots_total", "shots_on_target"])
    actual = actual.merge(fixtures, on="fixture_id", how="left")
    actual["shots_total"] = pd.to_numeric(actual["shots_total"], errors="coerce").fillna(0.0)
    actual["shots_on_target"] = pd.to_numeric(actual["shots_on_target"], errors="coerce").fillna(0.0)
    actual["shots_hit_flag"] = (actual["shots_total"] >= shot_threshold).astype(int)
    actual["sot_hit_flag"] = (actual["shots_on_target"] >= sot_threshold).astype(int)
    board = board.merge(
        actual[["fixture_key", "player_name", "shots_total", "shots_on_target", "shots_hit_flag", "sot_hit_flag"]],
        on=["fixture_key", "player_name"],
        how="left",
    )
    board["shots_hit_flag"] = board["shots_hit_flag"].fillna(0).astype(int)
    board["sot_hit_flag"] = board["sot_hit_flag"].fillna(0).astype(int)

    fixture_order = (
        board[["fixture_key", "match_date"]]
        .drop_duplicates()
        .assign(match_date_ts=lambda x: pd.to_datetime(x["match_date"], errors="coerce"))
        .sort_values(["match_date_ts", "fixture_key"], ascending=[False, False])
    )
    keep_keys = fixture_order.head(sample_size)["fixture_key"].tolist()
    board = board[board["fixture_key"].isin(keep_keys)].copy()

    rows = []
    for fixture_key, group in board.groupby("fixture_key", sort=False):
        shot_rows = group[group["market"].eq("shots")]
        sot_rows = group[group["market"].eq("shots_on_target")]
        shot_hit = int(not shot_rows.empty and int(shot_rows["shots_hit_flag"].max()) == 1)
        sot_hit = int(not sot_rows.empty and int(sot_rows["sot_hit_flag"].max()) == 1)
        dual_group = group[group["same_player_dual_trigger_flag"].eq(1)]
        dual_both_hit = 0
        dual_either_hit = 0
        if not dual_group.empty:
            player_hits = (
                dual_group.groupby("player_name", as_index=False)
                .agg(shots_hit=("shots_hit_flag", "max"), sot_hit=("sot_hit_flag", "max"))
            )
            dual_both_hit = int(((player_hits["shots_hit"] == 1) & (player_hits["sot_hit"] == 1)).any())
            dual_either_hit = int(((player_hits["shots_hit"] == 1) | (player_hits["sot_hit"] == 1)).any())
        rows.append(
            {
                "fixture_key": fixture_key,
                "league": group["league"].iloc[0],
                "fixture_attacking_style_label": group["fixture_attacking_style_label"].iloc[0],
                "combo_reason_bucket": group["combo_reason_bucket"].iloc[0] if "combo_reason_bucket" in group.columns else "",
                "same_player_dual_trigger_fixture_flag": int(group["same_player_dual_trigger_flag"].max()),
                "shot_pick_hit": shot_hit,
                "sot_pick_hit": sot_hit,
                "both_markets_hit": int(shot_hit == 1 and sot_hit == 1),
                "either_market_hit": int(shot_hit == 1 or sot_hit == 1),
                "dual_same_player_both_hit": dual_both_hit,
                "dual_same_player_either_hit": dual_either_hit,
                "fixture_attack_quality_score": float(group["fixture_attack_quality_score"].iloc[0]),
            }
        )

    fixture_df = pd.DataFrame(rows)
    summary = pd.DataFrame([
        {
            "fixtures_audited": len(fixture_df),
            "shot_threshold": shot_threshold,
            "sot_threshold": sot_threshold,
            "shot_pick_hit_rate": round(float(fixture_df["shot_pick_hit"].mean()), 4) if len(fixture_df) else 0.0,
            "sot_pick_hit_rate": round(float(fixture_df["sot_pick_hit"].mean()), 4) if len(fixture_df) else 0.0,
            "both_markets_hit_rate": round(float(fixture_df["both_markets_hit"].mean()), 4) if len(fixture_df) else 0.0,
            "either_market_hit_rate": round(float(fixture_df["either_market_hit"].mean()), 4) if len(fixture_df) else 0.0,
            "dual_same_player_both_hit_rate": round(float(fixture_df["dual_same_player_both_hit"].mean()), 4) if len(fixture_df) else 0.0,
            "dual_same_player_either_hit_rate": round(float(fixture_df["dual_same_player_either_hit"].mean()), 4) if len(fixture_df) else 0.0,
            "avg_attack_quality_score": round(float(fixture_df["fixture_attack_quality_score"].mean()), 4) if len(fixture_df) else 0.0,
        }
    ])
    style_df = (
        fixture_df.groupby("fixture_attacking_style_label", as_index=False)
        .agg(
            fixtures=("fixture_key", "count"),
            shot_pick_hit_rate=("shot_pick_hit", "mean"),
            sot_pick_hit_rate=("sot_pick_hit", "mean"),
            both_markets_hit_rate=("both_markets_hit", "mean"),
            either_market_hit_rate=("either_market_hit", "mean"),
            dual_same_player_both_hit_rate=("dual_same_player_both_hit", "mean"),
            dual_same_player_either_hit_rate=("dual_same_player_either_hit", "mean"),
            avg_attack_quality_score=("fixture_attack_quality_score", "mean"),
        )
        .sort_values(["dual_same_player_both_hit_rate", "both_markets_hit_rate", "fixtures"], ascending=[False, False, False])
    )
    combo_df = (
        fixture_df.groupby("combo_reason_bucket", as_index=False)
        .agg(
            fixtures=("fixture_key", "count"),
            shot_pick_hit_rate=("shot_pick_hit", "mean"),
            sot_pick_hit_rate=("sot_pick_hit", "mean"),
            both_markets_hit_rate=("both_markets_hit", "mean"),
            either_market_hit_rate=("either_market_hit", "mean"),
            dual_same_player_both_hit_rate=("dual_same_player_both_hit", "mean"),
            dual_same_player_either_hit_rate=("dual_same_player_either_hit", "mean"),
        )
        .sort_values(["dual_same_player_both_hit_rate", "both_markets_hit_rate", "fixtures"], ascending=[False, False, False])
    )

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(board_csv).stem
    summary.to_csv(out / f"{stem}__attacking_audit_summary_last{sample_size}.csv", index=False)
    style_df.to_csv(out / f"{stem}__attacking_style_audit_last{sample_size}.csv", index=False)
    combo_df.to_csv(out / f"{stem}__attacking_combo_audit_last{sample_size}.csv", index=False)
    fixture_df.to_csv(out / f"{stem}__attacking_fixture_audit_last{sample_size}.csv", index=False)
    return summary, style_df, combo_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a combined attacking board against actual shots + SOT outcomes.")
    parser.add_argument("--board-csv", required=True)
    parser.add_argument("--fixtures-csv", required=True)
    parser.add_argument("--player-stats-csv", required=True)
    parser.add_argument("--outdir", default="reports/player_events/combined_boards")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--shot-threshold", type=int, default=2)
    parser.add_argument("--sot-threshold", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, style_df, combo_df = audit_combined_attacking_board(
        board_csv=args.board_csv,
        fixtures_csv=args.fixtures_csv,
        player_stats_csv=args.player_stats_csv,
        outdir=args.outdir,
        sample_size=args.sample_size,
        shot_threshold=args.shot_threshold,
        sot_threshold=args.sot_threshold,
    )
    print("WROTE:", args.outdir)
    print(summary.to_string(index=False))
    print(style_df.to_string(index=False))
    print(combo_df.to_string(index=False))


if __name__ == "__main__":
    main()
