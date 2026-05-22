#!/usr/bin/env python3
"""Backtest World Cup player-event intelligence selections.

Calibrates score thresholds on 2018 and applies them unchanged to 2022.
This is a research-only market-specific intelligence audit, not priced props
or live deployment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "data_sources" / "footystats_world_cup" / "player_event_fixture_inputs"
DEFAULT_OUTDIR = ROOT / "data_sources" / "footystats_world_cup" / "player_event_backtests"

MARKETS: dict[str, dict[str, Any]] = {
    "shots_0_5": {"label": "Player Shots 0.5+", "actual_col": "actual_shots_total", "threshold": 1, "eligible": "attacker"},
    "shots_1_5": {"label": "Player Shots 1.5+", "actual_col": "actual_shots_total", "threshold": 2, "eligible": "attacker"},
    "sot_0_5": {"label": "Player SOT 0.5+", "actual_col": "actual_shots_on_target", "threshold": 1, "eligible": "attacker"},
    "keeper_saves_1_5": {"label": "Keeper Saves 1.5+", "actual_col": "actual_saves", "threshold": 2, "eligible": "keeper"},
    "keeper_saves_2_5": {"label": "Keeper Saves 2.5+", "actual_col": "actual_saves", "threshold": 3, "eligible": "keeper"},
    "tackles_1_5": {"label": "Player Tackles 1.5+", "actual_col": "actual_tackles", "threshold": 2, "eligible": "contact"},
    "fouls_1_5": {"label": "Player Fouls 1.5+", "actual_col": "actual_fouls_committed", "threshold": 2, "eligible": "contact"},
    "cards_0_5": {"label": "Player Cards 0.5+ Hazard", "actual_col": "actual_yellow_cards", "threshold": 1, "eligible": "contact"},
}


def num(values: Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def safe_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return num(df[col]).fillna(default)


def pct(series: pd.Series) -> pd.Series:
    values = num(series).fillna(0.0)
    if values.nunique(dropna=False) <= 1:
        return pd.Series(0.0, index=series.index)
    return values.rank(pct=True).fillna(0.0)


def eligible_mask(df: pd.DataFrame, family: str) -> pd.Series:
    pos = df.get("position_group", pd.Series("", index=df.index)).astype(str).str.lower()
    expected = safe_series(df, "expected_minutes", 0.0)
    if family == "attacker":
        return expected.ge(35) & pos.isin(["forward", "midfielder"])
    if family == "keeper":
        return expected.ge(45) & pos.eq("goalkeeper")
    return expected.ge(45) & pos.isin(["defender", "midfielder", "forward"])


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    position = out.get("position_group", pd.Series("", index=out.index)).astype(str).str.lower()
    role = out.get("tactical_role", pd.Series("", index=out.index)).astype(str).str.lower()
    minutes = safe_series(out, "expected_minutes", 0.0).clip(0, 90) / 90.0
    attack_context = (
        0.45 * pct(safe_series(out, "fixture_attack_pressure_score"))
        + 0.25 * pct(safe_series(out, "og_xg_total"))
        + 0.20 * pct(safe_series(out, "team_power_edge"))
        + 0.10 * pct(safe_series(out, "shots_per90"))
    ).clip(0, 1)
    contact_context = (
        0.35 * pct(safe_series(out, "fixture_tackle_density_score"))
        + 0.30 * pct(safe_series(out, "fixture_foul_density_score"))
        + 0.20 * pct(safe_series(out, "formation_pressure_score"))
        + 0.15 * pct(safe_series(out, "opponent_possession_projection"))
    ).clip(0, 1)
    pressure_against = (
        0.45 * pct(-safe_series(out, "team_power_edge"))
        + 0.30 * pct(safe_series(out, "fixture_attack_pressure_score"))
        + 0.25 * pct(safe_series(out, "opponent_possession_projection"))
    ).clip(0, 1)
    attacker_role = position.isin(["forward", "midfielder"]).astype(float)
    defensive_role = position.isin(["defender", "midfielder"]).astype(float)
    keeper_role = position.eq("goalkeeper").astype(float)
    holding_role = role.str.contains("holding|central|centre|wide defender|wing").astype(float)

    out["score_shots_0_5"] = (
        0.40 * pct(safe_series(out, "shots_per90"))
        + 0.20 * pct(safe_series(out, "goals_per90") + safe_series(out, "assists_per90"))
        + 0.20 * attack_context
        + 0.10 * minutes
        + 0.10 * attacker_role
    ).clip(0, 1)
    out["score_shots_1_5"] = (
        0.48 * pct(safe_series(out, "shots_per90"))
        + 0.18 * attack_context
        + 0.16 * pct(safe_series(out, "goals_per90"))
        + 0.10 * minutes
        + 0.08 * attacker_role
    ).clip(0, 1)
    out["score_sot_0_5"] = (
        0.48 * pct(safe_series(out, "shots_on_target_per90"))
        + 0.20 * pct(safe_series(out, "shots_per90"))
        + 0.18 * attack_context
        + 0.08 * minutes
        + 0.06 * attacker_role
    ).clip(0, 1)
    out["score_keeper_saves_1_5"] = (
        0.55 * pressure_against
        + 0.20 * pct(safe_series(out, "opponent_power_rating"))
        + 0.15 * pct(safe_series(out, "og_xg_total"))
        + 0.10 * keeper_role
    ).clip(0, 1)
    out["score_keeper_saves_2_5"] = (
        0.62 * pressure_against
        + 0.18 * pct(safe_series(out, "opponent_power_rating"))
        + 0.12 * pct(safe_series(out, "og_xg_total"))
        + 0.08 * keeper_role
    ).clip(0, 1)
    out["score_tackles_1_5"] = (
        0.42 * pct(safe_series(out, "tackles_per90"))
        + 0.25 * contact_context
        + 0.14 * pct(safe_series(out, "dribbles_faced_per90"))
        + 0.11 * defensive_role
        + 0.08 * holding_role
    ).clip(0, 1)
    out["score_fouls_1_5"] = (
        0.42 * pct(safe_series(out, "fouls_per90"))
        + 0.26 * contact_context
        + 0.14 * pct(safe_series(out, "temperament_flag"))
        + 0.10 * holding_role
        + 0.08 * defensive_role
    ).clip(0, 1)
    out["score_cards_0_5"] = (
        0.34 * pct(safe_series(out, "yellow_cards_per90"))
        + 0.22 * pct(safe_series(out, "fouls_per90"))
        + 0.20 * contact_context
        + 0.14 * holding_role
        + 0.10 * pct(safe_series(out, "match_stakes_score"))
    ).clip(0, 1)
    return out


def load_season(input_dir: Path, season: int) -> pd.DataFrame:
    features = pd.read_csv(input_dir / f"player_events_fixture_input__World_Cup__{season}.csv", low_memory=False)
    actuals = pd.read_csv(input_dir / f"world_cup_player_event_actuals__{season}.csv", low_memory=False)
    join_cols = ["season", "fixture_key", "player_name"]
    merged = features.merge(
        actuals[
            [
                "season",
                "fixture_key",
                "player_name",
                "minutes",
                "started_flag",
                "actual_shots_total",
                "actual_shots_on_target",
                "actual_tackles",
                "actual_fouls_committed",
                "actual_yellow_cards",
                "actual_saves",
            ]
        ],
        on=join_cols,
        how="left",
        suffixes=("", "_actual"),
    )
    merged["season"] = season
    return add_scores(merged)


def evaluate_threshold(df: pd.DataFrame, market: str, threshold: float) -> dict[str, Any]:
    spec = MARKETS[market]
    score_col = f"score_{market}"
    elig = eligible_mask(df, spec["eligible"])
    graded = df[elig & num(df[spec["actual_col"]]).notna()].copy()
    selected = graded[graded[score_col].ge(threshold)].copy()
    baseline = float(num(graded[spec["actual_col"]]).ge(float(spec["threshold"])).mean()) if len(graded) else np.nan
    hit_rate = float(num(selected[spec["actual_col"]]).ge(float(spec["threshold"])).mean()) if len(selected) else np.nan
    return {
        "market": market,
        "market_label": spec["label"],
        "score_threshold": threshold,
        "eligible_rows": int(len(graded)),
        "selected_rows": int(len(selected)),
        "hit_rows": int(num(selected[spec["actual_col"]]).ge(float(spec["threshold"])).sum()) if len(selected) else 0,
        "hit_rate": hit_rate,
        "baseline_hit_rate": baseline,
        "lift_vs_baseline": hit_rate - baseline if pd.notna(hit_rate) and pd.notna(baseline) else np.nan,
    }


def calibrate_threshold(train: pd.DataFrame, market: str, min_rows: int) -> float:
    spec = MARKETS[market]
    score_col = f"score_{market}"
    graded = train[eligible_mask(train, spec["eligible"]) & num(train[spec["actual_col"]]).notna()].copy()
    if graded.empty:
        return 1.01
    candidates = sorted(set(float(graded[score_col].quantile(q)) for q in np.arange(0.50, 0.96, 0.05)))
    best_threshold = candidates[0]
    best_key = (-999.0, -999.0, 0)
    for threshold in candidates:
        result = evaluate_threshold(train, market, threshold)
        if result["selected_rows"] < min_rows or pd.isna(result["hit_rate"]):
            continue
        key = (float(result["lift_vs_baseline"]), float(result["hit_rate"]), int(result["selected_rows"]))
        if key > best_key:
            best_key = key
            best_threshold = threshold
    return round(float(best_threshold), 6)


def selected_predictions(df: pd.DataFrame, market: str, threshold: float, split: str) -> pd.DataFrame:
    spec = MARKETS[market]
    score_col = f"score_{market}"
    work = df[eligible_mask(df, spec["eligible"]) & num(df[spec["actual_col"]]).notna()].copy()
    work = work[work[score_col].ge(threshold)].copy()
    work["split"] = split
    work["market"] = market
    work["market_label"] = spec["label"]
    work["score"] = work[score_col]
    work["actual_value"] = num(work[spec["actual_col"]])
    work["actual_hit"] = work["actual_value"].ge(float(spec["threshold"])).astype(int)
    keep = [
        "split",
        "season",
        "fixture_key",
        "match_date",
        "home_team_name",
        "away_team_name",
        "team_name",
        "player_name",
        "position_group",
        "tactical_role",
        "market",
        "market_label",
        "score",
        "actual_value",
        "actual_hit",
        "expected_minutes",
        "shots_per90",
        "shots_on_target_per90",
        "tackles_per90",
        "fouls_per90",
        "yellow_cards_per90",
        "fixture_attack_pressure_score",
        "fixture_tackle_density_score",
        "fixture_foul_density_score",
        "world_cup_scope",
        "world_cup_lineup_scope",
    ]
    for col in keep:
        if col not in work.columns:
            work[col] = pd.NA
    return work[keep].sort_values(["market", "score"], ascending=[True, False])


def write_summary(outdir: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# World Cup Player-Event Backtest",
        "",
        "- Research-only. Thresholds are calibrated on 2018 and applied unchanged to 2022.",
        "- This tests market-specific intelligence selection, not priced prop odds.",
        "- Historical player membership uses API match player payloads, so this is not yet a pre-kickoff lineup-proof product.",
        "",
        "## 2022 Holdout Results",
    ]
    holdout = summary[summary["split"].eq("test_2022")].sort_values(["lift_vs_baseline", "hit_rate"], ascending=False)
    for _, row in holdout.iterrows():
        lift = "" if pd.isna(row["lift_vs_baseline"]) else f"{float(row['lift_vs_baseline']):+.3f}"
        hit = "" if pd.isna(row["hit_rate"]) else f"{float(row['hit_rate']):.3f}"
        base = "" if pd.isna(row["baseline_hit_rate"]) else f"{float(row['baseline_hit_rate']):.3f}"
        lines.append(
            f"- {row['market_label']}: selected={int(row['selected_rows'])}/{int(row['eligible_rows'])} "
            f"hit={hit} baseline={base} lift={lift} threshold={float(row['score_threshold']):.3f}"
        )
    lines += [
        "",
        "## Read",
        "- Positive lift cells are candidates for 2026 watch boards.",
        "- Negative/low-sample cells stay research-only until richer qualifier/domestic player-event priors and confirmed lineups are available.",
    ]
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest World Cup player-event intelligence markets.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--min-train-selected", type=int, default=20)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    train = load_season(input_dir, 2018)
    test = load_season(input_dir, 2022)

    rows: list[dict[str, Any]] = []
    preds: list[pd.DataFrame] = []
    for market in MARKETS:
        threshold = calibrate_threshold(train, market, args.min_train_selected)
        train_result = evaluate_threshold(train, market, threshold)
        train_result["split"] = "train_2018"
        test_result = evaluate_threshold(test, market, threshold)
        test_result["split"] = "test_2022"
        rows.extend([train_result, test_result])
        preds.append(selected_predictions(train, market, threshold, "train_2018"))
        preds.append(selected_predictions(test, market, threshold, "test_2022"))

    summary = pd.DataFrame(rows)
    predictions = pd.concat(preds, ignore_index=True, sort=False) if preds else pd.DataFrame()
    summary.to_csv(outdir / "world_cup_player_event_backtest_summary.csv", index=False)
    predictions.to_csv(outdir / "world_cup_player_event_backtest_selected_predictions.csv", index=False)
    write_summary(outdir, summary)
    print(f"[ok] wrote {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
