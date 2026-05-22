#!/usr/bin/env python3
"""Build a 2026 World Cup pre-tournament player-event intelligence board.

This is not priced betting output. It ranks player-event candidates using the
same market-specific score families as the historical backtest, then labels
each market with the 2018->2022 evidence currently available.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_world_cup_player_event_markets import MARKETS, add_scores, eligible_mask


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "data_sources"
    / "footystats_world_cup"
    / "player_event_fixture_inputs"
    / "player_events_fixture_input__World_Cup__2026_pre_tournament.csv"
)
DEFAULT_BACKTEST = (
    ROOT
    / "data_sources"
    / "footystats_world_cup"
    / "player_event_backtests"
    / "world_cup_player_event_backtest_summary.csv"
)
DEFAULT_OUTDIR = ROOT / "data_sources" / "footystats_world_cup" / "player_event_board_2026"


MARKET_FAMILIES = {
    "shots_0_5": "likely_shooters",
    "shots_1_5": "likely_shooters",
    "sot_0_5": "sot_candidates",
    "keeper_saves_1_5": "keeper_save_pressure_spots",
    "keeper_saves_2_5": "keeper_save_pressure_spots",
    "tackles_1_5": "tackle_contact_hotspots",
    "fouls_1_5": "foul_contact_hotspots",
    "cards_0_5": "bookings_hazard_watch",
}


def load_market_evidence(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, low_memory=False)
    df = df[df["split"].eq("test_2022")].copy()
    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        lift = pd.to_numeric(row.get("lift_vs_baseline"), errors="coerce")
        hit = pd.to_numeric(row.get("hit_rate"), errors="coerce")
        selected = int(pd.to_numeric(row.get("selected_rows"), errors="coerce") or 0)
        if pd.notna(lift) and lift > 0.04 and selected >= 8:
            status = "WATCH_CANDIDATE"
        elif pd.notna(lift) and lift > 0 and selected >= 5:
            status = "LIGHT_WATCH"
        else:
            status = "RESEARCH_ONLY"
        out[str(row["market"])] = {
            "evidence_status": status,
            "holdout_hit_rate": float(hit) if pd.notna(hit) else pd.NA,
            "holdout_lift_vs_baseline": float(lift) if pd.notna(lift) else pd.NA,
            "backtest_score_threshold": float(pd.to_numeric(row.get("score_threshold"), errors="coerce") or 0.0),
            "holdout_selected_rows": selected,
        }
    return out


def board_rows(features: pd.DataFrame, evidence: dict[str, dict[str, Any]], per_market_limit: int) -> pd.DataFrame:
    scored = add_scores(features)
    rows: list[pd.DataFrame] = []
    for market, spec in MARKETS.items():
        score_col = f"score_{market}"
        work = scored[eligible_mask(scored, spec["eligible"])].copy()
        if work.empty:
            continue
        ev = evidence.get(market, {})
        threshold = float(ev.get("backtest_score_threshold", work[score_col].quantile(0.82)))
        work = work.sort_values(["fixture_key", score_col], ascending=[True, False]).copy()
        work["fixture_market_rank"] = work.groupby("fixture_key").cumcount() + 1
        work = work[work["fixture_market_rank"].le(per_market_limit)].copy()
        work["market"] = market
        work["market_label"] = spec["label"]
        work["market_family"] = MARKET_FAMILIES.get(market, "player_event_watch")
        work["score"] = work[score_col]
        work["evidence_status"] = ev.get("evidence_status", "NO_HOLDOUT_EVIDENCE")
        work["holdout_hit_rate"] = ev.get("holdout_hit_rate", pd.NA)
        work["holdout_lift_vs_baseline"] = ev.get("holdout_lift_vs_baseline", pd.NA)
        work["backtest_score_threshold"] = threshold
        work["clears_backtest_threshold_flag"] = work["score"].ge(threshold).astype(int)
        work["board_priority"] = work.apply(priority_label, axis=1)
        rows.append(work)
    if not rows:
        return pd.DataFrame()
    board = pd.concat(rows, ignore_index=True, sort=False)
    keep = [
        "market_family",
        "market",
        "market_label",
        "board_priority",
        "evidence_status",
        "clears_backtest_threshold_flag",
        "score",
        "fixture_market_rank",
        "fixture_key",
        "match_date",
        "home_team_name",
        "away_team_name",
        "team_name",
        "opponent_team_name",
        "player_name",
        "position_group",
        "tactical_role",
        "player_team_side",
        "expected_minutes",
        "shots_per90",
        "shots_on_target_per90",
        "tackles_per90",
        "fouls_per90",
        "yellow_cards_per90",
        "fixture_attack_pressure_score",
        "fixture_tackle_density_score",
        "fixture_foul_density_score",
        "team_power_edge",
        "holdout_hit_rate",
        "holdout_lift_vs_baseline",
        "backtest_score_threshold",
        "world_cup_scope",
        "world_cup_lineup_scope",
        "analyst_notes",
    ]
    for col in keep:
        if col not in board.columns:
            board[col] = pd.NA
    return board[keep].sort_values(["board_priority", "market_family", "score"], ascending=[True, True, False])


def priority_label(row: pd.Series) -> str:
    evidence = str(row.get("evidence_status", ""))
    clears = int(row.get("clears_backtest_threshold_flag", 0) or 0)
    score = float(pd.to_numeric(row.get("score"), errors="coerce") or 0.0)
    if evidence == "WATCH_CANDIDATE" and clears:
        return "A_WATCH_CANDIDATE"
    if evidence in {"WATCH_CANDIDATE", "LIGHT_WATCH"} and score >= 0.75:
        return "B_STRONG_RESEARCH"
    if clears or score >= 0.70:
        return "C_RESEARCH_WATCH"
    return "D_CONTEXT_ONLY"


def write_summary(outdir: Path, board: pd.DataFrame) -> None:
    lines = [
        "# 2026 World Cup Player-Event Intelligence Board",
        "",
        "- Pre-tournament research board only. Not priced bets and not deploy routing.",
        "- Built from API 2026 roster scaffolds, FootyStats additions context, macro/player-power fixture context, and 2018->2022 holdout labels.",
        "- Official squads, injuries, confirmed lineups, and same-tournament player ratings should replace or upgrade this during the tournament.",
        "",
        "## Board Counts",
    ]
    if board.empty:
        lines.append("- No board rows.")
    else:
        counts = board.groupby(["market_family", "board_priority"], dropna=False).size().reset_index(name="rows")
        for _, row in counts.iterrows():
            lines.append(f"- {row['market_family']} | {row['board_priority']} | rows={int(row['rows'])}")
    lines += [
        "",
        "## Current Read",
        "- Use this to prepare match-by-match research notes and launch content.",
        "- Treat bookings/fouls as watch/hazard context until referee and lineup truth matures.",
        "- Keeper saves and SOT candidates should sharpen materially after each World Cup matchday because same-tournament SOT and player-rating signals become real rather than projected.",
    ]
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build 2026 World Cup player-event intelligence board.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--backtest-summary", default=str(DEFAULT_BACKTEST))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--per-market-fixture-limit", type=int, default=5)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(args.input, low_memory=False)
    evidence = load_market_evidence(Path(args.backtest_summary))
    board = board_rows(features, evidence, args.per_market_fixture_limit)
    board.to_csv(outdir / "world_cup_2026_player_event_intelligence_board.csv", index=False)
    if not board.empty:
        for family, group in board.groupby("market_family", dropna=False):
            family_name = str(family).replace("/", "_")
            group.to_csv(outdir / f"world_cup_2026_{family_name}.csv", index=False)
    write_summary(outdir, board)
    print(f"[ok] rows={len(board)} wrote {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
