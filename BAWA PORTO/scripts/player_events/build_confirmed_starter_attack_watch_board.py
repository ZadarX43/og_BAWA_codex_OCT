#!/usr/bin/env python3
"""Build confirmed-starter shots/SOT attack-watch rows.

Research-only sidecar. This is deliberately separate from the strict
recent-form x opponent-allowance interaction board: it catches practical live
shots/SOT candidates where the player is confirmed starting and the attacking
environment is strong, even if the exact opponent-allowance proof cell does
not fire.

No priced odds, deploy routing, slips, or production rulebook changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.player_events.build_confirmed_lineup_player_event_shortlist_demo import (  # noqa: E402
    build_fixture_matches,
    markdown_table,
    player_key,
    read_many_csv,
)


DEFAULT_BOARD = (
    ROOT
    / "reports"
    / "2026-05-08"
    / "player_event_exact_interaction_shadow_refresh_2026_05_09"
    / "player_event_live_feature_join"
    / "PLAYER_EVENT_HITRATE_BAND_DASHBOARD__WITH_INTERACTION_FEATURES.csv"
)
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-09" / "confirmed_starter_attack_watch_board"

ATTACK_WATCH_MARKETS = {
    ("PLAYER_SHOTS", 1.5): {
        "shadow_stage": "PLAYER_SHOTS_1_5_CONFIRMED_STARTER_ATTACK_WATCH",
        "expression": "Player Shots 1.5+",
        "min_hit_pct": 66.0,
        "min_recent_per90": 1.40,
        "recent_col": "attacker_recent_shots_per90_l8",
        "priority_hit_pct": 74.0,
    },
    ("PLAYER_SHOTS", 2.5): {
        "shadow_stage": "PLAYER_SHOTS_2_5_CONFIRMED_STARTER_ATTACK_WATCH",
        "expression": "Player Shots 2.5+",
        "min_hit_pct": 52.0,
        "min_recent_per90": 2.05,
        "recent_col": "attacker_recent_shots_per90_l8",
        "priority_hit_pct": 58.0,
    },
    ("PLAYER_SOT", 0.5): {
        "shadow_stage": "PLAYER_SOT_0_5_CONFIRMED_STARTER_ATTACK_WATCH",
        "expression": "Player SOT 0.5+",
        "min_hit_pct": 58.0,
        "min_recent_per90": 0.45,
        "recent_col": "attacker_recent_sot_per90_l8",
        "priority_hit_pct": 70.0,
    },
    ("PLAYER_SOT", 1.5): {
        "shadow_stage": "PLAYER_SOT_1_5_CONFIRMED_STARTER_ATTACK_WATCH",
        "expression": "Player SOT 1.5+",
        "min_hit_pct": 44.0,
        "min_recent_per90": 0.80,
        "recent_col": "attacker_recent_sot_per90_l8",
        "priority_hit_pct": 50.0,
    },
}

ATTACKING_ROLES = {
    "Forward",
    "Midfielder",
}

SHADOW_COLUMNS = [
    "shadow_family",
    "shadow_stage",
    "fixture_key",
    "match_date",
    "league",
    "home_team_name",
    "away_team_name",
    "expression",
    "source_market",
    "source_selection",
    "source_deploy_tier",
    "source_tier",
    "combo_product",
    "combo_tier",
    "bookie_od",
    "model_prob",
    "value_edge",
    "value_edge_tier",
    "backtest_hit_rate",
    "backtest_graded",
    "league_stability_bucket",
    "team_stability_bucket",
    "watch_priority",
    "watch_flag",
    "guardrail",
    "reason",
    "player_name",
    "team_name",
    "player_team_side",
    "position_group",
    "tactical_role",
    "predicted_hit_rate",
    "predicted_hit_rate_pct",
    "confidence_label",
    "support_score",
    "context_reason_codes",
    "lineup_watch_flags",
    "expected_minutes",
    "formation_matchup_label",
    "formation_pressure_score",
    "fixture_style_label",
    "fixture_attacking_style_label",
    "fixture_foul_density_score",
    "fixture_wide_duel_score",
    "fixture_territorial_stress_score",
    "fixture_attack_pressure_score",
    "referee_name",
    "ref_cards_per_match",
    "interaction_match_mode",
    "interaction_label",
    "fouled_context_cell",
    "fouled_context_cell_label",
    "lineup_confirmation_status",
    "api_fixture_id",
    "api_kickoff_local",
    "attack_watch_recent_feature",
    "attack_watch_recent_value",
    "attack_watch_gate_label",
    "player_sub_swap_review_mode",
]


def num(values: Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def scalar(row: pd.Series, col: str, default: Any = "") -> Any:
    if col not in row.index:
        return default
    value = row[col]
    if pd.isna(value):
        return default
    return value


def load_starter_keys(lineups: pd.DataFrame) -> dict[int, set[str]]:
    work = lineups.copy()
    work["is_starting_xi"] = num(work.get("is_starting_xi", 0)).fillna(0).astype(int)
    work["player_key"] = work.get("player_name", pd.Series("", index=work.index)).map(player_key)
    return work[work["is_starting_xi"].eq(1)].groupby("fixture_id")["player_key"].apply(set).to_dict()


def enrich_lineup_matches(
    board: pd.DataFrame,
    fixtures: pd.DataFrame,
    lineups: pd.DataFrame,
    kickoff_times: set[str],
) -> pd.DataFrame:
    fixture_matches = build_fixture_matches(board, fixtures, kickoff_times)
    lineups = lineups.copy()
    lineups["is_starting_xi"] = num(lineups.get("is_starting_xi", 0)).fillna(0).astype(int)
    lineup_counts = lineups.groupby("fixture_id").size().rename("api_lineup_rows").reset_index()
    starter_counts = (
        lineups[lineups["is_starting_xi"].eq(1)]
        .groupby("fixture_id")
        .size()
        .rename("api_starter_rows")
        .reset_index()
    )
    out = (
        fixture_matches.merge(lineup_counts, left_on="api_fixture_id", right_on="fixture_id", how="left")
        .merge(starter_counts, left_on="api_fixture_id", right_on="fixture_id", how="left", suffixes=("", "_starter"))
        .drop(columns=["fixture_id", "fixture_id_starter"], errors="ignore")
    )
    out["api_lineup_rows"] = num(out.get("api_lineup_rows", 0)).fillna(0).astype(int)
    out["api_starter_rows"] = num(out.get("api_starter_rows", 0)).fillna(0).astype(int)
    out["lineup_coverage_status"] = out["api_starter_rows"].map(
        lambda value: "API_CONFIRMED_LINEUP" if int(value) >= 18 else "API_LINEUP_MISSING_OR_INCOMPLETE"
    )
    return out


def gate_label(row: pd.Series, config: dict[str, Any]) -> str:
    hit_pct = float(row.get("predicted_hit_rate_pct_num", 0) or 0)
    recent_value = float(row.get("_recent_value", 0) or 0)
    attack_pressure = float(row.get("fixture_attack_pressure_score", 0) or 0)
    support = float(row.get("support_score", 0) or 0)
    if hit_pct >= config["priority_hit_pct"] and attack_pressure >= 0.72 and support >= 2:
        return "ATTACK_WATCH_CORE"
    if hit_pct >= config["priority_hit_pct"] or (attack_pressure >= 0.72 and recent_value >= config["min_recent_per90"]):
        return "ATTACK_WATCH_CONFIRM"
    return "ATTACK_WATCH"


def build_rows(
    board: pd.DataFrame,
    fixtures: pd.DataFrame,
    lineups: pd.DataFrame,
    *,
    target_leagues: set[str],
    target_date: str,
    kickoff_times: set[str],
    max_rows_per_fixture: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = board.copy()
    work["match_date"] = pd.to_datetime(work.get("match_date"), errors="coerce").dt.date.astype(str)
    work = work[work["league"].isin(target_leagues) & work["match_date"].eq(target_date)].copy()
    work["threshold_num"] = num(work.get("threshold", np.nan)).round(3)
    work["predicted_hit_rate_pct_num"] = num(work.get("predicted_hit_rate_pct", np.nan))
    fixture_matches = enrich_lineup_matches(work, fixtures, lineups, kickoff_times)
    starter_keys = load_starter_keys(lineups)

    work = work.merge(
        fixture_matches[
            [
                "fixture_key",
                "api_fixture_id",
                "api_kickoff_local",
                "api_match_score",
                "api_lineup_rows",
                "api_starter_rows",
                "lineup_coverage_status",
            ]
        ],
        on="fixture_key",
        how="left",
    )
    work["player_key"] = work.get("player_name", pd.Series("", index=work.index)).map(player_key)
    work["lineup_confirmation_status"] = "LINEUP_PENDING"
    for idx, row in work.iterrows():
        fixture_id = row.get("api_fixture_id")
        if pd.isna(fixture_id):
            continue
        starters = starter_keys.get(int(fixture_id), set())
        if row.get("player_key") in starters:
            work.at[idx, "lineup_confirmation_status"] = "CONFIRMED_STARTER"
        elif starters:
            work.at[idx, "lineup_confirmation_status"] = "API_LINEUP_AVAILABLE_NOT_STARTING"

    selected_frames: list[pd.DataFrame] = []
    for (market_family, threshold), config in ATTACK_WATCH_MARKETS.items():
        recent_col = str(config["recent_col"])
        if recent_col not in work.columns:
            work[recent_col] = np.nan
        rows = work[
            work.get("market_family", pd.Series("", index=work.index)).astype(str).eq(market_family)
            & work["threshold_num"].eq(float(threshold))
            & work["lineup_confirmation_status"].eq("CONFIRMED_STARTER")
            & work["predicted_hit_rate_pct_num"].ge(float(config["min_hit_pct"]))
            & num(work.get("expected_minutes", 0)).ge(55)
            & work.get("position_group", pd.Series("", index=work.index)).astype(str).isin(ATTACKING_ROLES)
        ].copy()
        rows["_recent_value"] = num(rows[recent_col])
        rows = rows[
            rows["_recent_value"].ge(float(config["min_recent_per90"]))
            | num(rows.get("fixture_attack_pressure_score", 0)).ge(0.70)
            | rows.get("fixture_attacking_style_label", pd.Series("", index=rows.index)).astype(str).str.contains(
                "ATTACK", case=False, na=False
            )
        ].copy()
        if rows.empty:
            continue
        rows["shadow_stage"] = config["shadow_stage"]
        rows["expression_override"] = config["expression"]
        rows["attack_watch_recent_feature"] = recent_col
        rows["attack_watch_gate_label"] = rows.apply(lambda row: gate_label(row, config), axis=1)
        selected_frames.append(rows)

    selected = pd.concat(selected_frames, ignore_index=True, sort=False) if selected_frames else pd.DataFrame()
    if selected.empty:
        return pd.DataFrame(columns=SHADOW_COLUMNS), fixture_matches

    selected["_priority_rank"] = selected["attack_watch_gate_label"].map(
        {"ATTACK_WATCH_CORE": 3, "ATTACK_WATCH_CONFIRM": 2, "ATTACK_WATCH": 1}
    ).fillna(0)
    selected = selected.sort_values(
        ["fixture_key", "_priority_rank", "predicted_hit_rate_pct_num", "support_score"],
        ascending=[True, False, False, False],
    )
    selected = selected.groupby("fixture_key", group_keys=False).head(max_rows_per_fixture).reset_index(drop=True)

    records: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        gate = str(row.get("attack_watch_gate_label", "ATTACK_WATCH"))
        priority = "PRIORITY_CORE" if gate == "ATTACK_WATCH_CORE" else "PRIORITY_CONFIRM" if gate == "ATTACK_WATCH_CONFIRM" else "WATCH_ONLY_NOT_PROMOTION"
        records.append(
            {
                "shadow_family": "CONFIRMED_STARTER_ATTACK_WATCH",
                "shadow_stage": scalar(row, "shadow_stage"),
                "fixture_key": scalar(row, "fixture_key"),
                "match_date": scalar(row, "match_date"),
                "league": scalar(row, "league", scalar(row, "competition")),
                "home_team_name": scalar(row, "home_team_name"),
                "away_team_name": scalar(row, "away_team_name"),
                "expression": scalar(row, "expression_override"),
                "source_market": scalar(row, "market_family"),
                "source_selection": scalar(row, "player_name"),
                "source_deploy_tier": "PLAYER_EVENT_BETA",
                "source_tier": scalar(row, "confidence_label"),
                "combo_product": "",
                "combo_tier": gate,
                "bookie_od": np.nan,
                "model_prob": scalar(row, "predicted_hit_rate", np.nan),
                "value_edge": np.nan,
                "value_edge_tier": "",
                "backtest_hit_rate": scalar(row, "predicted_hit_rate", np.nan),
                "backtest_graded": np.nan,
                "league_stability_bucket": "",
                "team_stability_bucket": "",
                "watch_priority": priority,
                "watch_flag": True,
                "guardrail": (
                    "PLAYER_EVENT_BETA_ONLY|NO_PRICED_ODDS|NO_DEPLOY_PROMOTION|"
                    "CONFIRMED_STARTER_ATTACK_WATCH|SEPARATE_FROM_INTERACTION_CORE"
                ),
                "reason": (
                    f"{gate}|confirmed_starter=1|"
                    f"recent={row.get('attack_watch_recent_feature')}:{row.get('_recent_value')}|"
                    f"attack_pressure={row.get('fixture_attack_pressure_score')}|"
                    "opponent_allowance_not_required"
                ),
                "player_name": scalar(row, "player_name"),
                "team_name": scalar(row, "team_name"),
                "player_team_side": scalar(row, "player_team_side", scalar(row, "player_team_side_x")),
                "position_group": scalar(row, "position_group"),
                "tactical_role": scalar(row, "tactical_role"),
                "predicted_hit_rate": scalar(row, "predicted_hit_rate", np.nan),
                "predicted_hit_rate_pct": scalar(row, "predicted_hit_rate_pct", np.nan),
                "confidence_label": scalar(row, "confidence_label"),
                "support_score": scalar(row, "support_score", np.nan),
                "context_reason_codes": scalar(row, "context_reason_codes"),
                "lineup_watch_flags": scalar(row, "lineup_watch_flags"),
                "expected_minutes": scalar(row, "expected_minutes", np.nan),
                "formation_matchup_label": scalar(row, "formation_matchup_label"),
                "formation_pressure_score": scalar(row, "formation_pressure_score", np.nan),
                "fixture_style_label": scalar(row, "fixture_style_label"),
                "fixture_attacking_style_label": scalar(row, "fixture_attacking_style_label"),
                "fixture_foul_density_score": scalar(row, "fixture_foul_density_score", np.nan),
                "fixture_wide_duel_score": scalar(row, "fixture_wide_duel_score", np.nan),
                "fixture_territorial_stress_score": scalar(row, "fixture_territorial_stress_score", np.nan),
                "fixture_attack_pressure_score": scalar(row, "fixture_attack_pressure_score", np.nan),
                "referee_name": scalar(row, "referee_name"),
                "ref_cards_per_match": scalar(row, "ref_cards_per_match", np.nan),
                "interaction_match_mode": "CONFIRMED_STARTER_ATTACK_WATCH",
                "interaction_label": gate,
                "fouled_context_cell": "",
                "fouled_context_cell_label": "",
                "lineup_confirmation_status": scalar(row, "lineup_confirmation_status"),
                "api_fixture_id": scalar(row, "api_fixture_id", np.nan),
                "api_kickoff_local": scalar(row, "api_kickoff_local"),
                "attack_watch_recent_feature": scalar(row, "attack_watch_recent_feature"),
                "attack_watch_recent_value": scalar(row, "_recent_value", np.nan),
                "attack_watch_gate_label": gate,
                "player_sub_swap_review_mode": "NAMED_PLAYER_DEFAULT__SUB_SWAP_REQUIRES_SEPARATE_ROLE_CHAIN_GRADING",
            }
        )
    return pd.DataFrame(records).reindex(columns=SHADOW_COLUMNS), fixture_matches


def write_report(outdir: Path, rows: pd.DataFrame, coverage: pd.DataFrame, board_path: Path) -> None:
    counts = (
        rows.groupby(["shadow_stage", "watch_priority"], dropna=False).size().reset_index(name="rows")
        if not rows.empty
        else pd.DataFrame(columns=["shadow_stage", "watch_priority", "rows"])
    )
    top_cols = [
        "api_kickoff_local",
        "league",
        "home_team_name",
        "away_team_name",
        "expression",
        "player_name",
        "team_name",
        "predicted_hit_rate_pct",
        "watch_priority",
        "attack_watch_gate_label",
        "attack_watch_recent_value",
    ]
    lines = [
        "# Confirmed-Starter Attack Watch Board",
        "",
        "Research-only shots/SOT sidecar for confirmed starters.",
        "",
        "## Why This Exists",
        "- Strict `INTERACTION_CORE` remains the proof lane.",
        "- This lane catches shots/SOT candidates where recent attacker form and match attack pressure are strong but exact opponent-allowance gates do not fire.",
        "- Sub-swap products are not mixed into named-player grading; they are flagged for separate role-chain review.",
        "",
        "## Source",
        f"- player-event board: `{board_path}`",
        "",
        "## Counts",
        markdown_table(counts, ["shadow_stage", "watch_priority", "rows"], max_rows=80),
        "",
        "## Top Rows",
        markdown_table(rows, [col for col in top_cols if col in rows.columns], max_rows=80),
        "",
        "## Fixture Coverage",
        markdown_table(
            coverage,
            [
                "api_kickoff_local",
                "league",
                "home_team_name",
                "away_team_name",
                "api_starter_rows",
                "lineup_coverage_status",
            ],
            max_rows=80,
        ),
        "",
        "## Guardrails",
        "- No priced player-event odds.",
        "- No deploy promotion.",
        "- No mutation of source dashboards or production rulebook.",
        "- Normal outcome grading remains named-player only.",
        "- Player Sub Swap requires separate replacement/role-chain grading before it is included in evidence.",
    ]
    (outdir / "CONFIRMED_STARTER_ATTACK_WATCH_BOARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player-event-board", type=Path, default=DEFAULT_BOARD)
    parser.add_argument("--fixtures-csvs", required=True)
    parser.add_argument("--lineups-csvs", required=True)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--target-leagues", default="England Premier League,Germany Bundesliga")
    parser.add_argument("--target-date", default="2026-05-09")
    parser.add_argument("--kickoff-times", default="14:30,15:00")
    parser.add_argument("--max-rows-per-fixture", type=int, default=5)
    args = parser.parse_args()

    if not args.player_event_board.exists():
        raise SystemExit(f"Missing player-event board: {args.player_event_board}")
    board = pd.read_csv(args.player_event_board, low_memory=False)
    fixtures = read_many_csv(args.fixtures_csvs)
    lineups = read_many_csv(args.lineups_csvs)
    if fixtures.empty or lineups.empty:
        raise SystemExit("Fixtures and lineups inputs must be non-empty.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    target_leagues = {item.strip() for item in args.target_leagues.split(",") if item.strip()}
    kickoff_times = {item.strip() for item in args.kickoff_times.split(",") if item.strip()}
    rows, coverage = build_rows(
        board,
        fixtures,
        lineups,
        target_leagues=target_leagues,
        target_date=args.target_date,
        kickoff_times=kickoff_times,
        max_rows_per_fixture=args.max_rows_per_fixture,
    )
    rows.to_csv(args.outdir / "CONFIRMED_STARTER_ATTACK_WATCH_BOARD.csv", index=False)
    coverage.to_csv(args.outdir / "CONFIRMED_STARTER_ATTACK_WATCH_FIXTURE_COVERAGE.csv", index=False)
    write_report(args.outdir, rows, coverage, args.player_event_board)
    print(f"WROTE {args.outdir / 'CONFIRMED_STARTER_ATTACK_WATCH_BOARD.csv'}")
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
