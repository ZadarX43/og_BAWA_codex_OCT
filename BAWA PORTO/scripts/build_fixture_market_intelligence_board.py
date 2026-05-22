#!/usr/bin/env python3
"""Build fixture-level market intelligence shadow rows.

Research-only sidecar for Bet365-style fixture/team markets:
goal range, winning margin, total shots/SOT, team-most shots/SOT/corners/cards.
It does not create priced odds, deploy picks, slips, or production routing
changes.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-08" / "fixture_market_intelligence_board"
DEFAULT_CONTEXT = (
    ROOT
    / "reports"
    / "2026-05-07"
    / "live_shadow_research_dashboard_with_key_pass_assist"
    / "2026_05_02_to_2026_05_04"
    / "2026_05_02_to_2026_05_04__LIVE_SHADOW_RESEARCH_DASHBOARD.csv"
)
DEFAULT_FIXTURE_INPUTS = (
    ROOT
    / "reports"
    / "2026-05-07"
    / "player_event_current_board_fixture_inputs_clean"
    / "CURRENT_BOARD_PLAYER_EVENT_FIXTURE_INPUTS_ALL.csv"
)
DEFAULT_TACTICAL_REGISTRY = ROOT / "reports" / "2026-05-08" / "tactical_feature_registry" / "TACTICAL_FEATURE_REGISTRY.csv"


def num(values: Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def scalar(row: pd.Series, col: str, default: Any = "") -> Any:
    value = row.get(col, default)
    if pd.isna(value):
        return default
    return value


def latest_allmarkets() -> Path:
    candidates = [
        path
        for path in (ROOT / "predictions_output").glob("**/BOOKIE_IMP20_ALLMARKETS_*.csv")
        if "__DEPLOY_" not in path.name
    ]
    if not candidates:
        raise FileNotFoundError("No BOOKIE_IMP20_ALLMARKETS_*.csv files found under predictions_output")
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def registry_for_stage(registry: pd.DataFrame, stage: str) -> dict[str, str]:
    if registry.empty or "target_shadow_stages" not in registry.columns:
        return {
            "tactical_feature_ids": "",
            "tactical_feature_families": "",
            "tactical_market_hooks": "",
            "tactical_leakage_risk_max": "",
        }
    mask = registry["target_shadow_stages"].astype(str).str.contains(stage, regex=False, na=False)
    subset = registry[mask].copy()
    if subset.empty:
        return {
            "tactical_feature_ids": "",
            "tactical_feature_families": "",
            "tactical_market_hooks": "",
            "tactical_leakage_risk_max": "",
        }
    risks = subset["leakage_risk"].astype(str)
    return {
        "tactical_feature_ids": "|".join(dict.fromkeys(subset["feature_id"].astype(str))),
        "tactical_feature_families": "|".join(dict.fromkeys(subset["family"].astype(str))),
        "tactical_market_hooks": "|".join(dict.fromkeys(subset["target_markets"].astype(str))),
        "tactical_leakage_risk_max": "MEDIUM" if risks.str.contains("MEDIUM", na=False).any() else "LOW",
    }


def fixture_base_from_allmarkets(allmarkets: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "fixture_key",
        "match_date",
        "league",
        "home_team_name",
        "away_team_name",
        "lambda_home",
        "lambda_away",
        "exp_goals_sum",
        "cs_mass_home_win",
        "cs_mass_draw",
        "cs_mass_away_win",
        "mass_0_goals",
        "mass_1_goal",
        "mass_2_goals",
        "mass_3_goals",
        "mass_4plus_goals",
        "p_meta_ou25",
        "p_meta_ftr",
        "ftr_margin",
        "cs_mass_over25",
    ]
    present = [col for col in keep if col in allmarkets.columns]
    if "fixture_key" not in present:
        return pd.DataFrame()
    base = allmarkets[present].drop_duplicates("fixture_key").copy()
    for col in [c for c in present if c not in {"fixture_key", "match_date", "league", "home_team_name", "away_team_name"}]:
        base[col] = num(base[col])
    return base


def fixture_context_from_dashboard(context: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "fixture_key",
        "home_team_expected_shots",
        "away_team_expected_shots",
        "match_expected_shots",
        "home_team_expected_sot",
        "away_team_expected_sot",
        "match_expected_sot",
        "home_team_expected_corners",
        "away_team_expected_corners",
        "match_expected_corners",
        "home_team_shots_profile_labels",
        "away_team_shots_profile_labels",
    ]
    present = [col for col in keep if col in context.columns]
    if "fixture_key" not in present:
        return pd.DataFrame(columns=["fixture_key"])
    out = context[present].drop_duplicates("fixture_key").copy()
    for col in [
        "home_team_expected_shots",
        "away_team_expected_shots",
        "match_expected_shots",
        "home_team_expected_sot",
        "away_team_expected_sot",
        "match_expected_sot",
        "home_team_expected_corners",
        "away_team_expected_corners",
        "match_expected_corners",
    ]:
        if col in out.columns:
            out[col] = num(out[col])
    return out


def fixture_card_context(fixture_inputs: pd.DataFrame) -> pd.DataFrame:
    if fixture_inputs.empty or "fixture_key" not in fixture_inputs.columns:
        return pd.DataFrame(columns=["fixture_key"])
    work = fixture_inputs.copy()
    for col in ["team_avg_yellows", "team_avg_fouls", "ref_cards_per_match", "fixture_foul_density_score"]:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = num(work[col])
    team = (
        work.groupby(["fixture_key", "team_name"], dropna=False)
        .agg(
            team_avg_yellows=("team_avg_yellows", "mean"),
            team_avg_fouls=("team_avg_fouls", "mean"),
            ref_cards_per_match=("ref_cards_per_match", "mean"),
            fixture_foul_density_score=("fixture_foul_density_score", "mean"),
        )
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for fixture_key, group in team.groupby("fixture_key", dropna=False):
        if len(group) < 2:
            continue
        group = group.copy()
        group["projected_card_pressure"] = (
            group["team_avg_yellows"].fillna(0)
            + 0.08 * group["team_avg_fouls"].fillna(0)
            + 0.12 * group["ref_cards_per_match"].fillna(0)
            + 0.20 * group["fixture_foul_density_score"].fillna(0)
        )
        ordered = group.sort_values("projected_card_pressure", ascending=False)
        top = ordered.iloc[0]
        second = ordered.iloc[1]
        rows.append(
            {
                "fixture_key": fixture_key,
                "team_most_cards": top["team_name"],
                "team_most_cards_score": float(top["projected_card_pressure"]),
                "team_most_cards_margin": float(top["projected_card_pressure"] - second["projected_card_pressure"]),
                "match_card_pressure": float(group["projected_card_pressure"].sum()),
            }
        )
    return pd.DataFrame(rows)


def poisson_grid(lambda_home: float, lambda_away: float, max_goals: int = 9) -> pd.DataFrame:
    if not np.isfinite(lambda_home) or not np.isfinite(lambda_away) or lambda_home <= 0 or lambda_away <= 0:
        return pd.DataFrame()
    rows = []
    for home_goals in range(max_goals + 1):
        ph = math.exp(-lambda_home) * lambda_home**home_goals / math.factorial(home_goals)
        for away_goals in range(max_goals + 1):
            pa = math.exp(-lambda_away) * lambda_away**away_goals / math.factorial(away_goals)
            rows.append({"home_goals": home_goals, "away_goals": away_goals, "p": ph * pa})
    grid = pd.DataFrame(rows)
    total = grid["p"].sum()
    if total > 0:
        grid["p"] = grid["p"] / total
    return grid


def margin_probs(row: pd.Series) -> dict[str, float]:
    grid = poisson_grid(float(scalar(row, "lambda_home", np.nan)), float(scalar(row, "lambda_away", np.nan)))
    if grid.empty:
        return {
            "home_margin_2plus": np.nan,
            "away_margin_2plus": np.nan,
            "home_margin_3plus": np.nan,
            "away_margin_3plus": np.nan,
        }
    diff = grid["home_goals"] - grid["away_goals"]
    return {
        "home_margin_2plus": float(grid.loc[diff.ge(2), "p"].sum()),
        "away_margin_2plus": float(grid.loc[diff.le(-2), "p"].sum()),
        "home_margin_3plus": float(grid.loc[diff.ge(3), "p"].sum()),
        "away_margin_3plus": float(grid.loc[diff.le(-3), "p"].sum()),
    }


def add_row(
    records: list[dict[str, Any]],
    row: pd.Series,
    stage: str,
    expression: str,
    score: float,
    selection: str,
    priority: str,
    reason: str,
    registry: pd.DataFrame,
) -> None:
    registry_tags = registry_for_stage(registry, stage)
    records.append(
        {
            "shadow_family": "FIXTURE_MARKET_INTELLIGENCE",
            "shadow_stage": stage,
            "fixture_key": scalar(row, "fixture_key"),
            "match_date": scalar(row, "match_date"),
            "league": scalar(row, "league"),
            "home_team_name": scalar(row, "home_team_name"),
            "away_team_name": scalar(row, "away_team_name"),
            "expression": expression,
            "source_market": "fixture_market_intelligence",
            "source_selection": selection,
            "source_deploy_tier": "FIXTURE_MARKET_BETA",
            "source_tier": priority,
            "combo_product": expression,
            "combo_tier": priority,
            "bookie_od": np.nan,
            "model_prob": score,
            "value_edge": np.nan,
            "value_edge_tier": "",
            "backtest_hit_rate": np.nan,
            "backtest_graded": np.nan,
            "watch_flag": True,
            "guardrail": "FIXTURE_MARKET_BETA_ONLY|NO_PRICED_ODDS|NO_DEPLOY_PROMOTION",
            "reason": reason,
            "watch_priority": priority,
            "confidence_label": priority,
            "fixture_market_score": score,
            **registry_tags,
        }
    )


def build_rows(fixtures: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, row in fixtures.iterrows():
        goal_ranges = {
            "0-1 goals": float(scalar(row, "mass_0_goals", 0.0) or 0.0) + float(scalar(row, "mass_1_goal", 0.0) or 0.0),
            "2-3 goals": float(scalar(row, "mass_2_goals", 0.0) or 0.0) + float(scalar(row, "mass_3_goals", 0.0) or 0.0),
            "4+ goals": float(scalar(row, "mass_4plus_goals", 0.0) or 0.0),
        }
        goal_label, goal_score = max(goal_ranges.items(), key=lambda item: item[1])
        if goal_score >= 0.43 or (goal_label == "4+ goals" and goal_score >= 0.30):
            add_row(
                records,
                row,
                "GOAL_RANGE_SHADOW",
                f"Goal Range: {goal_label}",
                goal_score,
                goal_label,
                "PRIORITY_CONFIRM" if goal_score >= 0.48 else "WATCH_ONLY_NOT_PROMOTION",
                f"goal_range={goal_label}|p={goal_score:.3f}|ou25={float(scalar(row, 'cs_mass_over25', np.nan)):.3f}",
                registry,
            )

        margins = margin_probs(row)
        margin_2 = {
            "HOME 2+": margins["home_margin_2plus"],
            "AWAY 2+": margins["away_margin_2plus"],
        }
        side_2, score_2 = max(margin_2.items(), key=lambda item: -1 if pd.isna(item[1]) else item[1])
        if pd.notna(score_2) and score_2 >= 0.24:
            add_row(
                records,
                row,
                "WINNING_MARGIN_2_PLUS_SHADOW",
                f"Winning Margin: {side_2}",
                score_2,
                side_2,
                "PRIORITY_CONFIRM" if score_2 >= 0.32 else "WATCH_ONLY_NOT_PROMOTION",
                f"home_2plus={margins['home_margin_2plus']:.3f}|away_2plus={margins['away_margin_2plus']:.3f}",
                registry,
            )
        margin_3 = {
            "HOME 3+": margins["home_margin_3plus"],
            "AWAY 3+": margins["away_margin_3plus"],
        }
        side_3, score_3 = max(margin_3.items(), key=lambda item: -1 if pd.isna(item[1]) else item[1])
        if pd.notna(score_3) and score_3 >= 0.12:
            add_row(
                records,
                row,
                "WINNING_MARGIN_3_PLUS_WATCH",
                f"Winning Margin: {side_3}",
                score_3,
                side_3,
                "WATCH_ONLY_NOT_PROMOTION",
                f"home_3plus={margins['home_margin_3plus']:.3f}|away_3plus={margins['away_margin_3plus']:.3f}",
                registry,
            )

        match_shots = float(scalar(row, "match_expected_shots", np.nan))
        if np.isfinite(match_shots) and (match_shots >= 24.5 or match_shots <= 20.0):
            selection = "HIGH" if match_shots >= 24.5 else "LOW"
            score = min(0.99, max(0.0, (match_shots - 18.0) / 12.0)) if selection == "HIGH" else min(0.99, max(0.0, (23.0 - match_shots) / 8.0))
            add_row(records, row, "TOTAL_SHOTS_SHADOW", f"Total Shots: {selection}", score, selection, "PRIORITY_CONFIRM", f"match_expected_shots={match_shots:.2f}", registry)

        match_sot = float(scalar(row, "match_expected_sot", np.nan))
        if np.isfinite(match_sot) and (match_sot >= 8.0 or match_sot <= 6.0):
            selection = "HIGH" if match_sot >= 8.0 else "LOW"
            score = min(0.99, max(0.0, (match_sot - 5.5) / 5.0)) if selection == "HIGH" else min(0.99, max(0.0, (7.0 - match_sot) / 3.5))
            add_row(records, row, "TOTAL_SOT_SHADOW", f"Total SOT: {selection}", score, selection, "PRIORITY_CONFIRM", f"match_expected_sot={match_sot:.2f}", registry)

        for stage, home_col, away_col, label, threshold in [
            ("TEAM_MOST_SHOTS_SHADOW", "home_team_expected_shots", "away_team_expected_shots", "shots", 2.5),
            ("TEAM_MOST_SOT_SHADOW", "home_team_expected_sot", "away_team_expected_sot", "SOT", 0.9),
            ("TEAM_MOST_CORNERS_SHADOW", "home_team_expected_corners", "away_team_expected_corners", "corners", 0.9),
        ]:
            home_value = float(scalar(row, home_col, np.nan))
            away_value = float(scalar(row, away_col, np.nan))
            if not np.isfinite(home_value) or not np.isfinite(away_value):
                continue
            diff = home_value - away_value
            if abs(diff) >= threshold:
                side = "HOME" if diff > 0 else "AWAY"
                team = scalar(row, "home_team_name") if side == "HOME" else scalar(row, "away_team_name")
                score = min(0.99, abs(diff) / (threshold * 3.0))
                add_row(records, row, stage, f"Team Most {label}: {team}", score, team, "PRIORITY_CONFIRM", f"home_{label}={home_value:.2f}|away_{label}={away_value:.2f}|diff={diff:.2f}", registry)

        card_team = scalar(row, "team_most_cards", "")
        card_margin = float(scalar(row, "team_most_cards_margin", np.nan))
        if card_team and np.isfinite(card_margin) and card_margin >= 0.25:
            score = min(0.99, card_margin / 1.25)
            add_row(
                records,
                row,
                "TEAM_MOST_CARDS_WATCH",
                f"Team Most Cards: {card_team}",
                score,
                card_team,
                "WATCH_ONLY_NOT_PROMOTION",
                f"card_pressure_margin={card_margin:.3f}|match_card_pressure={float(scalar(row, 'match_card_pressure', np.nan)):.3f}",
                registry,
            )
    return pd.DataFrame(records)


def markdown_table(df: pd.DataFrame, max_rows: int = 60) -> str:
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


def write_report(outdir: Path, rows: pd.DataFrame, allmarkets_path: Path, context_path: Path, fixture_inputs_path: Path, registry_path: Path) -> None:
    counts = (
        rows.groupby(["shadow_stage", "watch_priority"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["shadow_stage", "watch_priority"])
        if not rows.empty
        else pd.DataFrame()
    )
    sample_cols = ["match_date", "league", "home_team_name", "away_team_name", "shadow_stage", "expression", "model_prob", "watch_priority", "tactical_feature_ids", "tactical_feature_families", "reason"]
    lines = [
        "# Fixture Market Intelligence Board",
        "",
        "Research-only fixture market sidecar for Bet365-style markets.",
        "",
        "## Safety",
        "- No priced odds.",
        "- No deploy routing, tiers, slips, or production rulebook changes.",
        "- Rows are shadow-only notes for manual review and future outcome audit.",
        "",
        "## Inputs",
        f"- allmarkets: `{allmarkets_path}`",
        f"- context dashboard: `{context_path}`",
        f"- fixture inputs: `{fixture_inputs_path}`",
        f"- tactical registry: `{registry_path}`",
        "",
        "## Overall",
        f"- rows: `{len(rows)}`",
        "",
        "## Counts",
        markdown_table(counts),
        "",
        "## Sample Rows",
        markdown_table(rows[[c for c in sample_cols if c in rows.columns]], max_rows=40),
    ]
    (outdir / "FIXTURE_MARKET_INTELLIGENCE_BOARD.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allmarkets", type=Path, default=None)
    parser.add_argument("--context-dashboard", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--fixture-inputs", type=Path, default=DEFAULT_FIXTURE_INPUTS)
    parser.add_argument("--tactical-registry", type=Path, default=DEFAULT_TACTICAL_REGISTRY)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    allmarkets_path = args.allmarkets or latest_allmarkets()
    allmarkets = read_csv(allmarkets_path)
    context = read_csv(args.context_dashboard)
    fixture_inputs = read_csv(args.fixture_inputs)
    registry = read_csv(args.tactical_registry)
    if allmarkets.empty:
        raise SystemExit(f"Missing/empty allmarkets input: {allmarkets_path}")

    fixtures = fixture_base_from_allmarkets(allmarkets)
    ctx = fixture_context_from_dashboard(context)
    cards = fixture_card_context(fixture_inputs)
    if not ctx.empty:
        fixtures = fixtures.merge(ctx, on="fixture_key", how="left")
    if not cards.empty:
        fixtures = fixtures.merge(cards, on="fixture_key", how="left")

    rows = build_rows(fixtures, registry)
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.outdir / "FIXTURE_MARKET_INTELLIGENCE_BOARD.csv", index=False)
    if not rows.empty:
        rows.groupby(["shadow_stage", "watch_priority"], dropna=False).size().reset_index(name="rows").to_csv(
            args.outdir / "FIXTURE_MARKET_INTELLIGENCE_COUNTS.csv",
            index=False,
        )
    write_report(args.outdir, rows, allmarkets_path, args.context_dashboard, args.fixture_inputs, args.tactical_registry)
    print(f"WROTE {args.outdir}")
    print(f"rows={len(rows)}")
    if not rows.empty:
        print(rows.groupby(["shadow_stage", "watch_priority"], dropna=False).size().reset_index(name="rows").to_string(index=False))


if __name__ == "__main__":
    main()
