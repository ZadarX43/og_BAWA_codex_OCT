#!/usr/bin/env python3
"""Simulate the Phase 8H C3 research gate table over C1/C2 scored rows.

This is a post-scoring research harness. It does not call or modify
`deploy_rulebook.py`; it only applies documented candidate gates to the
row-level C1/C2 replay output.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_ROW_LEVEL = Path(
    "reports/2026-05-06/phase8h_replay_sweeps_c1_c2/phase8h_replay_row_level_scored.csv"
)
DEFAULT_OUTDIR = Path("reports/2026-05-06/phase8h_c3_policy_simulation")

PROTECTED_BENCHMARK_FLOORS = {
    ("btts", "ELITE"): 0.9068,
    ("btts", "STANDARD"): 0.8706,
    ("ftr", "ELITE"): 0.9264,
    ("ftr", "STANDARD"): 0.8332,
    ("ou25", "STANDARD"): 0.9021,
}


def gate(
    name: str,
    lane: str,
    variant: str,
    market: str,
    league: str,
    tier: str | list[str] | None,
    conditions: list[tuple[str, str, Any]],
    priority: int,
    note: str,
) -> dict[str, Any]:
    return {
        "gate_name": name,
        "c3_lane": lane,
        "variant": variant,
        "market": market,
        "league": league,
        "tier": tier,
        "conditions": conditions,
        "priority": priority,
        "note": note,
    }


C3_GATES: list[dict[str, Any]] = [
    gate(
        "btts_belgium_elite_c1_phase8e",
        "BTTS",
        "c1",
        "btts",
        "Belgium Pro",
        "ELITE",
        [("selection", "==", "YES"), ("p_meta_btts", ">=", 0.88)],
        100,
        "Phase 8E Belgium BTTS prior restored on C1.",
    ),
    gate(
        "btts_belgium_standard_c2_fts_confirm",
        "BTTS",
        "c2",
        "btts",
        "Belgium Pro",
        "STANDARD",
        [
            ("selection", "==", "YES"),
            ("p_meta_btts", ">=", 0.8999),
            ("btts_fts_sum", "<=", 0.3969),
        ],
        110,
        "Belgium STANDARD needs FTS confirmation; C2 pair gate was 30/30.",
    ),
    gate(
        "btts_belgium_standard_c1_phase8e_backup",
        "BTTS",
        "c1",
        "btts",
        "Belgium Pro",
        "STANDARD",
        [("selection", "==", "YES"), ("p_meta_btts", ">=", 0.88)],
        111,
        "C1 Phase 8E backup for Belgium STANDARD.",
    ),
    gate(
        "btts_usa_elite_c1_phase8e",
        "BTTS",
        "c1",
        "btts",
        "USA MLS",
        "ELITE",
        [("selection", "==", "YES"), ("p_meta_btts", ">=", 0.88)],
        120,
        "USA BTTS Phase 8E prior remains strong.",
    ),
    gate(
        "btts_spain_elite_c2_base",
        "BTTS",
        "c2",
        "btts",
        "Spain La Liga",
        "ELITE",
        [("selection", "==", "YES")],
        130,
        "Spain C2 BTTS ELITE base was already benchmark-safe.",
    ),
    gate(
        "btts_norway_elite_c2_p00_confirm",
        "BTTS",
        "c2",
        "btts",
        "Norway Eliteserien",
        "ELITE",
        [
            ("selection", "==", "YES"),
            ("p_meta_btts", ">=", 0.8646),
            ("p00_est", "<=", 0.0532),
        ],
        140,
        "Norway needs low-0-0-risk confirmation; Phase 8E 0.80 alone is unsafe.",
    ),
    gate(
        "ou25_usa_standard_c1_phase8e",
        "OU25",
        "c1",
        "ou25",
        "USA MLS",
        "STANDARD",
        [("selection", "==", "OVER25"), ("p_meta_ou25", ">=", 0.80)],
        200,
        "USA OU25 Phase 8E prior restored on C1.",
    ),
    gate(
        "ou25_germany_standard_c1_grid_confirm",
        "OU25",
        "c1",
        "ou25",
        "Germany Bundesliga",
        "STANDARD",
        [
            ("selection", "==", "OVER25"),
            ("model_p_for_bookie", ">=", 0.6895),
            ("grid_vs_cat_ou25_gap", "<=", 0.0810),
        ],
        210,
        "Germany OU25 should use C1 plus grid/Cat coherence; C2 broad output is unsafe.",
    ),
    gate(
        "ou25_spain_standard_c2_grid_expand",
        "OU25",
        "c2",
        "ou25",
        "Spain La Liga",
        "STANDARD",
        [
            ("selection", "==", "OVER25"),
            ("p_meta_ou25", ">=", 0.9450),
            ("cs_mass_over25", ">=", 0.5991),
        ],
        220,
        "Spain C2 strict gate expands beyond the tiny C1 proof.",
    ),
    gate(
        "ou25_spain_standard_c1_phase8e_backup",
        "OU25",
        "c1",
        "ou25",
        "Spain La Liga",
        "STANDARD",
        [("selection", "==", "OVER25"), ("p_meta_ou25", ">=", 0.80)],
        221,
        "C1 Spain OU25 Phase 8E backup.",
    ),
    gate(
        "ou25_norway_standard_c2_grid_confirm",
        "OU25",
        "c2",
        "ou25",
        "Norway Eliteserien",
        "STANDARD",
        [
            ("selection", "==", "OVER25"),
            ("p_meta_ou25", ">=", 0.8995),
            ("grid_vs_cat_ou25_gap", "<=", 0.2755),
        ],
        230,
        "Norway OU25 needs meta plus grid-confirmed shape.",
    ),
    gate(
        "ou25_belgium_standard_c1_micro",
        "OU25",
        "c1",
        "ou25",
        "Belgium Pro",
        "STANDARD",
        [
            ("selection", "==", "OVER25"),
            ("p_meta_ou25", ">=", 0.9838),
            ("cs_mass_over25", ">=", 0.7265),
        ],
        240,
        "Belgium is Phase 8E OU25 never-reaches; only test as ultra-strict micro-lane.",
    ),
    gate(
        "ftr_spain_c2_base",
        "FTR",
        "c2",
        "ftr",
        "Spain La Liga",
        ["ELITE", "STANDARD"],
        [],
        300,
        "Spain C2 FTR base was benchmark-safe across ELITE and STANDARD.",
    ),
    gate(
        "ftr_belgium_c2_strict_margin",
        "FTR",
        "c2",
        "ftr",
        "Belgium Pro",
        ["ELITE", "STANDARD"],
        [("p_meta_ftr", ">=", 0.9977), ("ftr_margin", ">=", 0.2403)],
        310,
        "Belgium FTR broad lanes are weak; use only high-meta/high-margin subset.",
    ),
    gate(
        "ftr_germany_c1_strict_top3_margin",
        "FTR",
        "c1",
        "ftr",
        "Germany Bundesliga",
        ["ELITE", "STANDARD"],
        [("p_meta_ftr", ">=", 0.9773), ("pick_side_margin_top3", ">=", 0.2417)],
        320,
        "Germany FTR uses C1 strict top-3 side-shape confirmation.",
    ),
    gate(
        "ftr_usa_c1_top3_mass",
        "FTR",
        "c1",
        "ftr",
        "USA MLS",
        ["ELITE", "STANDARD"],
        [("pick_side_mass_top3", ">=", 0.3827)],
        330,
        "USA FTR small but useful top-3 mass subset.",
    ),
    gate(
        "combo_spain_c2_home_ge2",
        "TEAM_GOAL_COMBO",
        "c2",
        "ftr",
        "Spain La Liga",
        None,
        [
            ("ftr_combo_live_product", "==", "HOME_WIN_AND_HOME_GE2"),
            ("ftr_combo_live_tier", "in", ["MODERATE_HW", "STRONG_HW", "VERY_STRONG_HW"]),
        ],
        400,
        "Spain home-win-and-home-GE2 combo proof lane.",
    ),
    gate(
        "combo_germany_c2_home_ge2",
        "TEAM_GOAL_COMBO",
        "c2",
        "ftr",
        "Germany Bundesliga",
        None,
        [
            ("ftr_combo_live_product", "==", "HOME_WIN_AND_HOME_GE2"),
            ("ftr_combo_live_tier", "in", ["MODERATE_HW", "STRONG_HW", "VERY_STRONG_HW"]),
        ],
        410,
        "Germany home-win-and-home-GE2 combo proof lane.",
    ),
]


def condition_mask(df: pd.DataFrame, col: str, op: str, value: Any) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    series = df[col]
    if op == "==":
        return series.astype("string") == str(value)
    if op == "in":
        return series.astype("string").isin([str(v) for v in value])

    nums = pd.to_numeric(series, errors="coerce")
    if op == ">=":
        return nums >= float(value)
    if op == "<=":
        return nums <= float(value)
    if op == ">":
        return nums > float(value)
    if op == "<":
        return nums < float(value)
    raise ValueError(f"Unsupported operator: {op}")


def apply_gate(df: pd.DataFrame, rule: dict[str, Any]) -> pd.DataFrame:
    mask = (
        (df["variant"].astype("string") == rule["variant"])
        & (df["market"].astype("string") == rule["market"])
        & (df["league"].astype("string") == rule["league"])
    )

    tiers = rule["tier"]
    if tiers is not None:
        if isinstance(tiers, str):
            tiers = [tiers]
        mask &= df["tier"].astype("string").isin(tiers)

    for col, op, value in rule["conditions"]:
        mask &= condition_mask(df, col, op, value)

    selected = df.loc[mask].copy()
    if selected.empty:
        return selected

    selected["gate_name"] = rule["gate_name"]
    selected["c3_lane"] = rule["c3_lane"]
    selected["gate_priority"] = rule["priority"]
    selected["gate_note"] = rule["note"]
    return selected


def score_rows(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    combo_mask = rows["c3_lane"].eq("TEAM_GOAL_COMBO")
    rows["c3_market"] = np.where(combo_mask, "combo", rows["market"].astype("string"))
    rows["c3_tier"] = np.where(combo_mask, rows["ftr_combo_live_tier"], rows["tier"])
    rows["c3_selection"] = np.where(combo_mask, rows["ftr_combo_live_product"], rows["selection"])

    rows["c3_correct"] = pd.to_numeric(rows["correct"], errors="coerce")
    home_combo = combo_mask & rows["ftr_combo_live_product"].eq("HOME_WIN_AND_HOME_GE2")
    away_combo = combo_mask & rows["ftr_combo_live_product"].eq("AWAY_WIN_AND_AWAY_GE2")
    rows.loc[home_combo, "c3_correct"] = pd.to_numeric(
        rows.loc[home_combo, "hw_and_hge2_hit"], errors="coerce"
    )
    rows.loc[away_combo, "c3_correct"] = pd.to_numeric(
        rows.loc[away_combo, "aw_and_age2_hit"], errors="coerce"
    )

    odds = pd.to_numeric(rows["bookie_od"], errors="coerce")
    rows["c3_profit"] = np.where(rows["c3_correct"].eq(1), odds - 1.0, -1.0)
    rows.loc[rows["c3_correct"].isna(), "c3_profit"] = np.nan
    return rows


def scorecard(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=group_cols
            + [
                "rows",
                "graded",
                "wins",
                "losses",
                "hit_rate",
                "avg_bookie_od",
                "avg_model_p",
                "profit",
                "roi",
            ]
        )

    rows = []
    grouped = df.groupby(group_cols, dropna=False)
    for keys, g in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        graded_mask = g["c3_correct"].notna()
        graded = int(graded_mask.sum())
        wins = float(g.loc[graded_mask, "c3_correct"].sum()) if graded else 0.0
        profit = float(g.loc[graded_mask, "c3_profit"].sum()) if graded else 0.0
        row = dict(zip(group_cols, keys, strict=False))
        row.update(
            {
                "rows": int(len(g)),
                "graded": graded,
                "wins": wins,
                "losses": graded - wins,
                "hit_rate": wins / graded if graded else np.nan,
                "avg_bookie_od": pd.to_numeric(g["bookie_od"], errors="coerce").mean(),
                "avg_model_p": pd.to_numeric(g["model_p_for_bookie"], errors="coerce").mean(),
                "profit": profit,
                "roi": profit / graded if graded else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def benchmark_compare(policy_rows: pd.DataFrame) -> pd.DataFrame:
    tier_card = scorecard(policy_rows, ["c3_market", "c3_tier"])
    if tier_card.empty:
        return tier_card

    floors = []
    for _, row in tier_card.iterrows():
        key = (str(row["c3_market"]), str(row["c3_tier"]))
        floors.append(PROTECTED_BENCHMARK_FLOORS.get(key, np.nan))
    tier_card["benchmark_floor"] = floors
    tier_card["delta_vs_floor"] = tier_card["hit_rate"] - tier_card["benchmark_floor"]
    tier_card["floor_pass"] = np.where(
        tier_card["benchmark_floor"].isna(),
        "",
        np.where(tier_card["delta_vs_floor"] >= 0, "PASS", "FAIL"),
    )
    return tier_card


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows selected._"

    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
        else:
            display[col] = display[col].astype("string").fillna("")

    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def dedupe_policy_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows

    rows = rows.copy()
    rows["dedupe_key"] = (
        rows["league"].astype("string")
        + "||"
        + rows["fixture_key"].astype("string")
        + "||"
        + rows["c3_market"].astype("string")
        + "||"
        + rows["c3_selection"].astype("string")
    )
    rows["model_p_sort"] = pd.to_numeric(rows["model_p_for_bookie"], errors="coerce").fillna(-1)
    rows = rows.sort_values(
        ["dedupe_key", "gate_priority", "model_p_sort", "gate_name"],
        ascending=[True, True, False, True],
    )
    return rows.drop_duplicates("dedupe_key", keep="first").drop(columns=["model_p_sort"])


def write_summary(outdir: Path, all_hits: pd.DataFrame, policy_rows: pd.DataFrame) -> None:
    lane_card = scorecard(policy_rows, ["c3_lane", "c3_market"])
    market_league = scorecard(policy_rows, ["c3_lane", "c3_market", "league", "c3_tier"])
    gate_card = scorecard(all_hits, ["gate_name", "c3_lane", "variant", "league"])
    benchmark_card = benchmark_compare(policy_rows)

    lines = [
        "# Phase 8H C3 Policy Simulation",
        "",
        "Research-only post-scoring simulation. No production policy files changed.",
        "",
        "## Policy Lane Scorecard",
        markdown_table(lane_card),
        "",
        "## Protected Tier Benchmark Compare",
        markdown_table(benchmark_card),
        "",
        "## Market / League / Tier Scorecard",
        markdown_table(market_league),
        "",
        "## Gate Scorecard",
        markdown_table(gate_card),
        "",
        "## Interpretation",
        "",
        "- This file tests the documented C3 gate table against already-scored C1/C2 rows.",
        "- Treat this as a calibration proof table, not live deploy behavior.",
        "- The next loop is to review weak/low-volume cells, then decide whether to harden thresholds or expand validation.",
        "",
    ]
    (outdir / "phase8h_c3_policy_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row-level", type=Path, default=DEFAULT_ROW_LEVEL)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    df = pd.read_csv(args.row_level)
    args.outdir.mkdir(parents=True, exist_ok=True)

    selected = [apply_gate(df, rule) for rule in C3_GATES]
    selected = [s for s in selected if not s.empty]
    all_hits = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    all_hits = score_rows(all_hits) if not all_hits.empty else all_hits
    policy_rows = dedupe_policy_rows(all_hits) if not all_hits.empty else all_hits

    rules_df = pd.DataFrame(
        [
            {
                "gate_name": rule["gate_name"],
                "c3_lane": rule["c3_lane"],
                "variant": rule["variant"],
                "market": rule["market"],
                "league": rule["league"],
                "tier": rule["tier"],
                "priority": rule["priority"],
                "conditions": "; ".join(f"{c} {op} {v}" for c, op, v in rule["conditions"]),
                "note": rule["note"],
            }
            for rule in C3_GATES
        ]
    )

    all_hits.to_csv(args.outdir / "phase8h_c3_all_gate_hits.csv", index=False)
    policy_rows.to_csv(args.outdir / "phase8h_c3_policy_selected_dedup.csv", index=False)
    rules_df.to_csv(args.outdir / "phase8h_c3_gate_table.csv", index=False)

    scorecard(all_hits, ["gate_name", "c3_lane", "variant", "league"]).to_csv(
        args.outdir / "phase8h_c3_gate_scorecard.csv", index=False
    )
    scorecard(policy_rows, ["c3_lane", "c3_market"]).to_csv(
        args.outdir / "phase8h_c3_lane_scorecard.csv", index=False
    )
    scorecard(policy_rows, ["c3_lane", "c3_market", "league", "c3_tier"]).to_csv(
        args.outdir / "phase8h_c3_market_league_tier_scorecard.csv", index=False
    )
    benchmark_compare(policy_rows).to_csv(
        args.outdir / "phase8h_c3_protected_benchmark_compare.csv", index=False
    )

    if not all_hits.empty:
        overlap = (
            all_hits.groupby(["league", "fixture_key", "c3_market", "c3_selection"], dropna=False)
            .agg(
                gate_count=("gate_name", "nunique"),
                variants=("variant", lambda s: ",".join(sorted(set(map(str, s))))),
                gates=("gate_name", lambda s: " | ".join(sorted(set(map(str, s))))),
            )
            .reset_index()
            .query("gate_count > 1")
        )
    else:
        overlap = pd.DataFrame()
    overlap.to_csv(args.outdir / "phase8h_c3_overlap_dedup_audit.csv", index=False)

    write_summary(args.outdir, all_hits, policy_rows)
    print(f"[ok] wrote {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
