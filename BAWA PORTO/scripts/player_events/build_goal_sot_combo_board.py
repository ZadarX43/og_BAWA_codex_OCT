from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_combined_attacking_board import _build_fixture_gate
from render_player_to_score_report import _build_gsi
from render_player_shots_on_target_report import _build_soti


def _prep_market(df: pd.DataFrame, market: str) -> pd.DataFrame:
    if market == "goal":
        scored = _build_gsi(df.copy())
        scored = scored.rename(columns={"player_goal_index": "market_score", "confidence_label": "market_confidence"})
        scored["market"] = "goal"
    elif market == "shots_on_target":
        scored = _build_soti(df.copy())
        scored = scored.rename(columns={"player_sot_index": "market_score", "confidence_label": "market_confidence"})
        scored["market"] = "shots_on_target"
    else:
        raise ValueError(f"Unsupported market: {market}")
    return scored


def build_board(
    input_csv: str,
    output_csv: str,
    output_md: str,
    goal_min_score: float = 86.0,
    sot_min_score: float = 90.0,
    max_goal_per_fixture: int = 1,
    max_sot_per_fixture: int = 1,
    leagues_filter: list[str] | None = None,
    min_fixture_quality_score: float = 0.74,
    require_attack_support: bool = True,
    require_goal_env_support: bool = True,
    allowed_attacking_style_labels: list[str] | None = None,
    dual_trigger_only: bool = True,
) -> pd.DataFrame:
    base = pd.read_csv(input_csv)
    if leagues_filter:
        lf = {x.strip().lower() for x in leagues_filter if x.strip()}
        base = base[base["league"].astype("string").str.lower().isin(lf)].copy()

    fixture_gate = _build_fixture_gate(base, min_fixture_quality_score=min_fixture_quality_score)
    gate_cols = [
        "fixture_key",
        "fixture_attack_pressure_score",
        "fixture_corner_pressure_score",
        "fixture_territorial_stress_score",
        "fixture_attacking_style_label",
        "fixture_style_label",
        "og_goal_environment_score",
        "og_battle_on_score",
        "og_goal_environment_label",
        "fixture_attack_quality_score",
        "strong_attack_support_flag",
        "strong_goal_env_support_flag",
        "fixture_attack_quality_pass_flag",
        "fixture_attack_reason_codes",
    ]

    goal = _prep_market(base, "goal")
    sot = _prep_market(base, "shots_on_target")
    for df in (goal, sot):
        overlap_cols = [c for c in gate_cols if c != "fixture_key" and c in df.columns]
        if overlap_cols:
            df.drop(columns=overlap_cols, inplace=True)
    goal = goal.merge(fixture_gate[gate_cols], on="fixture_key", how="left")
    sot = sot.merge(fixture_gate[gate_cols], on="fixture_key", how="left")

    goal = goal[goal["market_score"].ge(goal_min_score)].copy()
    sot = sot[sot["market_score"].ge(sot_min_score)].copy()
    goal = goal[goal["fixture_attack_quality_pass_flag"].eq(1)].copy()
    sot = sot[sot["fixture_attack_quality_pass_flag"].eq(1)].copy()
    if require_attack_support:
        goal = goal[goal["strong_attack_support_flag"].eq(1)].copy()
        sot = sot[sot["strong_attack_support_flag"].eq(1)].copy()
    if require_goal_env_support:
        goal = goal[goal["strong_goal_env_support_flag"].eq(1)].copy()
        sot = sot[sot["strong_goal_env_support_flag"].eq(1)].copy()
    if allowed_attacking_style_labels:
        allowed = {x.strip().upper() for x in allowed_attacking_style_labels if x.strip()}
        goal = goal[goal["fixture_attacking_style_label"].astype("string").str.upper().isin(allowed)].copy()
        sot = sot[sot["fixture_attacking_style_label"].astype("string").str.upper().isin(allowed)].copy()

    goal = goal.sort_values(["fixture_key", "market_score"], ascending=[True, False]).groupby("fixture_key", as_index=False, group_keys=False).head(max_goal_per_fixture)
    sot = sot.sort_values(["fixture_key", "market_score"], ascending=[True, False]).groupby("fixture_key", as_index=False, group_keys=False).head(max_sot_per_fixture)

    keep_cols = [
        "fixture_key", "match_date", "competition", "league", "home_team_name", "away_team_name",
        "team_name", "player_name", "player_team_side", "position_group", "tactical_role",
        "goals_per90", "shots_per90", "shots_on_target_per90", "assists_per90", "key_passes_per90",
        "expected_minutes", "fixture_style_label", "fixture_attacking_style_label",
        "fixture_attack_pressure_score", "fixture_corner_pressure_score", "fixture_territorial_stress_score",
        "og_goal_environment_score", "og_battle_on_score", "og_goal_environment_label",
        "fixture_attack_quality_score", "strong_attack_support_flag", "strong_goal_env_support_flag",
        "fixture_attack_quality_pass_flag", "fixture_attack_reason_codes",
        "market", "market_score", "market_confidence", "analyst_notes",
    ]
    out = pd.concat([goal[keep_cols], sot[keep_cols]], ignore_index=True)
    if not out.empty:
        trigger_counts = out.groupby(["fixture_key", "player_name"], as_index=False).agg(
            trigger_count=("market", "nunique"),
            trigger_markets=("market", lambda s: "|".join(sorted(pd.Series(s).astype(str).unique()))),
        )
        out = out.merge(trigger_counts, on=["fixture_key", "player_name"], how="left")
        out["same_player_dual_trigger_flag"] = out["trigger_count"].ge(2).astype(int)
        out["combo_reason_bucket"] = (
            out["fixture_attacking_style_label"].astype("string").str.upper() + "__" +
            out["trigger_markets"].astype("string").str.replace("|", "_", regex=False).str.upper()
        )
    else:
        out["trigger_count"] = pd.Series(dtype=int)
        out["trigger_markets"] = pd.Series(dtype="string")
        out["same_player_dual_trigger_flag"] = pd.Series(dtype=int)
        out["combo_reason_bucket"] = pd.Series(dtype="string")

    if dual_trigger_only:
        out = out[out["same_player_dual_trigger_flag"].eq(1)].copy()

    out = out.sort_values(["match_date", "fixture_key", "market", "market_score"], ascending=[False, True, True, False]).reset_index(drop=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    lines = [
        "# Goal + SOT Combo Board", "", "## Rules",
        f"- goal minimum score: {goal_min_score}",
        f"- SOT minimum score: {sot_min_score}",
        f"- dual-trigger only: {dual_trigger_only}",
        f"- allowed attacking styles: {', '.join(allowed_attacking_style_labels) if allowed_attacking_style_labels else 'all'}", "",
        "## Summary",
        f"- rows: {len(out)}",
        f"- fixtures: {out['fixture_key'].nunique() if not out.empty else 0}",
        f"- goal rows: {int((out['market']=='goal').sum()) if not out.empty else 0}",
        f"- SOT rows: {int((out['market']=='shots_on_target').sum()) if not out.empty else 0}",
    ]
    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md).write_text("\n".join(lines))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a player-to-score + SOT combo board.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--goal-min-score", type=float, default=86.0)
    parser.add_argument("--sot-min-score", type=float, default=90.0)
    parser.add_argument("--max-goal-per-fixture", type=int, default=1)
    parser.add_argument("--max-sot-per-fixture", type=int, default=1)
    parser.add_argument("--leagues", default="")
    parser.add_argument("--min-fixture-quality-score", type=float, default=0.74)
    parser.add_argument("--require-attack-support", action="store_true")
    parser.add_argument("--require-goal-env-support", action="store_true")
    parser.add_argument("--allowed-attacking-style-labels", default="")
    parser.add_argument("--dual-trigger-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    leagues = [x.strip() for x in args.leagues.split(",") if x.strip()]
    styles = [x.strip() for x in args.allowed_attacking_style_labels.split(",") if x.strip()]
    out = build_board(
        input_csv=args.input,
        output_csv=args.output_csv,
        output_md=args.output_md,
        goal_min_score=args.goal_min_score,
        sot_min_score=args.sot_min_score,
        max_goal_per_fixture=args.max_goal_per_fixture,
        max_sot_per_fixture=args.max_sot_per_fixture,
        leagues_filter=leagues or None,
        min_fixture_quality_score=args.min_fixture_quality_score,
        require_attack_support=args.require_attack_support,
        require_goal_env_support=args.require_goal_env_support,
        allowed_attacking_style_labels=styles or None,
        dual_trigger_only=args.dual_trigger_only,
    )
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")
    print(f"fixtures: {out['fixture_key'].nunique() if not out.empty else 0}")
    print(f"goal_rows: {int((out['market']=='goal').sum()) if not out.empty else 0}")
    print(f"sot_rows: {int((out['market']=='shots_on_target').sum()) if not out.empty else 0}")


if __name__ == "__main__":
    main()
