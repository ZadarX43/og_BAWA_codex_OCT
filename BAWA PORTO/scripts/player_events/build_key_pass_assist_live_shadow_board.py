#!/usr/bin/env python3
"""Build current-board key-pass / assist intelligence shadow rows.

Research-only sidecar. Emits player key-pass and assist-watch rows for the
unified live shadow dashboard without priced odds or deploy promotion.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "reports"
    / "2026-05-07"
    / "player_event_current_board_fixture_inputs_clean"
    / "CURRENT_BOARD_PLAYER_EVENT_FIXTURE_INPUTS_ALL.csv"
)
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-07" / "key_pass_assist_live_shadow_board"


def num(values: Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def pct_rank(values: pd.Series) -> pd.Series:
    numeric = num(values).fillna(0.0)
    if numeric.nunique(dropna=False) <= 1:
        return pd.Series(0.5, index=values.index)
    return numeric.rank(pct=True)


def safe_01(values: pd.Series, scale: float = 1.0) -> pd.Series:
    return (num(values).fillna(0.0) / scale).clip(0.0, 1.0)


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["key_pass_per90_pct"] = pct_rank(out.get("key_passes_per90", pd.Series(0, index=out.index)))
    out["assist_per90_pct"] = pct_rank(out.get("assists_per90", pd.Series(0, index=out.index)))
    out["expected_minutes_pct"] = pct_rank(out.get("expected_minutes", pd.Series(0, index=out.index)))
    out["player_quality_pct"] = safe_01(out.get("player_quality_score_l5", pd.Series(0, index=out.index)), 100.0)
    out["attack_pressure_01"] = safe_01(out.get("fixture_attack_pressure_score", pd.Series(0, index=out.index)))
    out["corner_pressure_01"] = safe_01(out.get("fixture_corner_pressure_score", pd.Series(0, index=out.index)))
    out["territory_stress_01"] = safe_01(out.get("fixture_territorial_stress_score", pd.Series(0, index=out.index)))
    role_boost = out.get("position_group", pd.Series("", index=out.index)).map(
        {"Midfielder": 0.06, "Forward": 0.05, "Defender": -0.02}
    ).fillna(0.0)
    out["key_pass_live_score"] = (
        0.34 * out["key_pass_per90_pct"]
        + 0.14 * out["assist_per90_pct"]
        + 0.15 * out["expected_minutes_pct"]
        + 0.13 * out["player_quality_pct"]
        + 0.10 * out["attack_pressure_01"]
        + 0.07 * out["corner_pressure_01"]
        + 0.07 * out["territory_stress_01"]
        + role_boost
    ).clip(0.0, 1.0)
    out["assist_live_score"] = (
        0.28 * out["assist_per90_pct"]
        + 0.24 * out["key_pass_per90_pct"]
        + 0.14 * out["expected_minutes_pct"]
        + 0.12 * out["player_quality_pct"]
        + 0.10 * out["attack_pressure_01"]
        + 0.06 * out["corner_pressure_01"]
        + 0.06 * out["territory_stress_01"]
        + role_boost
    ).clip(0.0, 1.0)
    return out


def source_tier(score: float, market: str) -> tuple[str, str, float, float]:
    if market == "KEY_PASSES_0_5_LIVE_SHADOW":
        if score >= 0.85:
            return "CORE_WATCH", "PRIORITY_CORE", 62.03, 130804
        return "RESEARCH_READY", "PRIORITY_CONFIRM", 58.78, 177461
    if market == "KEY_PASSES_1_5_LIVE_SHADOW":
        if score >= 0.85:
            return "CORE_WATCH", "PRIORITY_CORE", 32.16, 130804
        return "RESEARCH_READY", "PRIORITY_CONFIRM", 28.99, 177461
    return "WATCH", "WATCH_ONLY_NOT_PROMOTION", 11.78, 95707


def build_rows(scored: pd.DataFrame) -> pd.DataFrame:
    eligible = scored[
        num(scored.get("expected_start_flag", pd.Series(0, index=scored.index))).ge(1)
        & num(scored.get("expected_minutes", pd.Series(0, index=scored.index))).ge(55)
        & scored.get("position_group", pd.Series("", index=scored.index)).isin(["Midfielder", "Forward", "Defender"])
    ].copy()
    records: list[dict[str, Any]] = []
    for _, row in eligible.iterrows():
        kp_score = float(row.get("key_pass_live_score", 0.0) or 0.0)
        assist_score = float(row.get("assist_live_score", 0.0) or 0.0)
        stages: list[tuple[str, float, str]] = []
        if kp_score >= 0.78:
            stages.append(("KEY_PASSES_0_5_LIVE_SHADOW", kp_score, "Player key passes 0.5+"))
        if kp_score >= 0.85:
            stages.append(("KEY_PASSES_1_5_LIVE_SHADOW", kp_score, "Player key passes 1.5+"))
        if assist_score >= 0.86 and kp_score >= 0.75:
            stages.append(("ASSIST_0_5_LIVE_WATCH", assist_score, "Player assist watch 0.5+"))
        for stage, score, expression in stages:
            confidence, priority, hit_pct, graded = source_tier(score, stage)
            records.append(
                {
                    "shadow_family": "KEY_PASS_ASSIST_INTELLIGENCE",
                    "shadow_stage": stage,
                    "fixture_key": row.get("fixture_key"),
                    "match_date": row.get("match_date"),
                    "league": row.get("league"),
                    "home_team_name": row.get("home_team_name"),
                    "away_team_name": row.get("away_team_name"),
                    "expression": expression,
                    "source_market": "player_key_pass_assist",
                    "source_selection": row.get("player_name"),
                    "source_deploy_tier": "PLAYER_EVENT_BETA",
                    "source_tier": confidence,
                    "combo_product": expression,
                    "combo_tier": confidence,
                    "model_prob": score,
                    "backtest_hit_rate": hit_pct / 100.0,
                    "backtest_graded": graded,
                    "league_stability_bucket": "LIVE_SHADOW_ONLY",
                    "team_stability_bucket": "LIVE_SHADOW_ONLY",
                    "watch_priority": priority,
                    "watch_flag": True,
                    "guardrail": "PLAYER_EVENT_BETA_ONLY|NO_PRICED_ODDS|NO_DEPLOY_PROMOTION",
                    "reason": (
                        f"key_pass_score={kp_score:.3f}|assist_score={assist_score:.3f}|"
                        f"key_passes_per90={float(row.get('key_passes_per90', 0) or 0):.3f}|"
                        f"expected_minutes={float(row.get('expected_minutes', 0) or 0):.1f}"
                    ),
                    "player_name": row.get("player_name"),
                    "team_name": row.get("team_name"),
                    "player_team_side": row.get("player_team_side"),
                    "position_group": row.get("position_group"),
                    "tactical_role": row.get("tactical_role"),
                    "predicted_hit_rate_pct": hit_pct,
                    "confidence_label": confidence,
                    "lineup_watch_flags": row.get("lineup_watch_flags"),
                    "expected_minutes": row.get("expected_minutes"),
                    "formation_matchup_label": row.get("formation_matchup_label"),
                    "formation_pressure_score": row.get("formation_pressure_score"),
                    "fixture_style_label": row.get("fixture_style_label"),
                    "fixture_attacking_style_label": row.get("fixture_attacking_style_label"),
                    "fixture_foul_density_score": row.get("fixture_foul_density_score"),
                    "fixture_wide_duel_score": row.get("fixture_wide_duel_score"),
                    "fixture_territorial_stress_score": row.get("fixture_territorial_stress_score"),
                    "fixture_attack_pressure_score": row.get("fixture_attack_pressure_score"),
                    "referee_name": row.get("referee_name"),
                    "ref_cards_per_match": row.get("ref_cards_per_match"),
                    "interaction_match_mode": "CURRENT_BOARD_PROFILE_SCORE",
                    "interaction_label": confidence,
                }
            )
    return pd.DataFrame(records)


def markdown_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df.empty:
        return "_No rows._"
    work = df.head(max_rows).copy()
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
        else:
            work[col] = work[col].astype("string").fillna("")
    lines = ["| " + " | ".join(work.columns) + " |", "| " + " | ".join(["---"] * len(work.columns)) + " |"]
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "/") for col in work.columns) + " |")
    return "\n".join(lines)


def write_report(outdir: Path, rows: pd.DataFrame, input_path: Path) -> None:
    summary = (
        rows.groupby(["shadow_stage", "watch_priority"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["shadow_stage", "watch_priority"])
        if not rows.empty
        else pd.DataFrame()
    )
    league_summary = (
        rows.groupby(["shadow_stage", "league"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["shadow_stage", "rows"], ascending=[True, False])
        if not rows.empty
        else pd.DataFrame()
    )
    lines = [
        "# Key Pass / Assist Live Shadow Board",
        "",
        "Research-only current-board intelligence rows.",
        "",
        "## Safety",
        "- No priced player-prop odds.",
        "- No deploy routing, tiers, slips, or production rulebook changes.",
        "- Assist remains a watch signal, not a confident binary prop.",
        "",
        "## Input",
        f"- `{input_path}`",
        "",
        "## Overall",
        f"- rows: `{len(rows)}`",
        "",
        "## Stage Counts",
        markdown_table(summary),
        "",
        "## League Counts",
        markdown_table(league_summary, max_rows=80),
    ]
    (outdir / "KEY_PASS_ASSIST_LIVE_SHADOW_BOARD.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-inputs", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    if not args.fixture_inputs.exists():
        raise SystemExit(f"Missing fixture inputs: {args.fixture_inputs}")
    scored = add_scores(pd.read_csv(args.fixture_inputs, low_memory=False))
    rows = build_rows(scored)
    rows.to_csv(args.outdir / "KEY_PASS_ASSIST_LIVE_SHADOW_BOARD.csv", index=False)
    write_report(args.outdir, rows, args.fixture_inputs)
    print(f"WROTE {args.outdir}")
    print(f"rows={len(rows)}")
    if not rows.empty:
        print(rows.groupby(["shadow_stage", "watch_priority"], dropna=False).size().reset_index(name="rows").to_string(index=False))


if __name__ == "__main__":
    main()
