#!/usr/bin/env python3
"""Audit formation/context cells inside player-fouled interaction lanes.

Research-only sidecar. Starts from historical scored rows produced by
PLAYER_FOULED_INTERACTION_POLICY_AUDIT, applies the strict live fouled-player
policies, then summarizes which tactical-role / formation / foul-ecosystem
cells look repeatable enough for dashboard watch prominence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCORED = (
    ROOT
    / "reports"
    / "2026-05-07"
    / "player_fouled_interaction_policy_audit_strict_opponent"
    / "player_fouled_interaction_scored_rows.csv"
)
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-07" / "player_fouled_formation_context_cell_audit"
MIN_MONTH_ROWS = 12


POLICIES = {
    "fouled_ge1": {
        "display_market": "Player Fouled 0.5+",
        "hit_col": "actual_fouled_ge1",
        "shadow_stage": "PLAYER_FOULED_0_5_INTERACTION_WATCH",
        "recent_feature": "attacker_recent_fouls_won_per90_l8",
        "recent_threshold": 1.377551,
        "opponent_feature": "opp_attack_allowed_role_fouls_drawn_per_player_l10",
        "opponent_threshold": 1.272727,
        "context_feature": "fixture_foul_density_score",
        "context_threshold": 0.7496,
        "core_hit": 0.78,
        "ready_hit": 0.74,
    },
    "fouled_ge2": {
        "display_market": "Player Fouled 1.5+",
        "hit_col": "actual_fouled_ge2",
        "shadow_stage": "PLAYER_FOULED_1_5_INTERACTION_WATCH",
        "recent_feature": "attacker_recent_fouls_won_per90_l8",
        "recent_threshold": 1.377551,
        "opponent_feature": "opp_attack_allowed_role_fouls_drawn_per_match_l10",
        "opponent_threshold": 2.0,
        "context_feature": "fixture_wide_duel_score",
        "context_threshold": 0.9137,
        "core_hit": 0.46,
        "ready_hit": 0.40,
    },
}


def num(values: Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def bucket(series: pd.Series, low: float, high: float, labels: tuple[str, str, str]) -> pd.Series:
    values = num(series).fillna(0.0)
    return np.select([values.ge(high), values.ge(low)], [labels[2], labels[1]], default=labels[0])


def month_stability(rows: pd.DataFrame, hit_col: str, baseline_hit: float) -> tuple[int, int, float]:
    monthly = rows.groupby("eval_month", dropna=False).agg(rows=(hit_col, "size"), hit_rate=(hit_col, "mean")).reset_index()
    monthly = monthly[monthly["rows"].ge(MIN_MONTH_ROWS)]
    if monthly.empty:
        return 0, 0, np.nan
    stable = int(monthly["hit_rate"].ge(baseline_hit).sum())
    return int(len(monthly)), stable, float(stable / len(monthly))


def apply_policy(scored: pd.DataFrame, market: str, policy: dict[str, Any]) -> pd.DataFrame:
    mask = (
        num(scored[policy["recent_feature"]]).ge(float(policy["recent_threshold"]))
        & num(scored[policy["opponent_feature"]]).ge(float(policy["opponent_threshold"]))
        & num(scored[policy["context_feature"]]).ge(float(policy["context_threshold"]))
    )
    out = scored[mask].copy()
    out["market"] = market
    out["display_market"] = policy["display_market"]
    out["shadow_stage"] = policy["shadow_stage"]
    out["hit_col"] = policy["hit_col"]
    out["eval_month"] = pd.to_datetime(out["match_date"], errors="coerce").dt.to_period("M").astype(str)
    out["formation_pressure_bucket"] = bucket(
        out["formation_pressure_score"],
        0.16,
        0.35,
        ("FORMATION_LOW", "FORMATION_MED", "FORMATION_HIGH"),
    )
    out["foul_density_bucket"] = bucket(
        out["fixture_foul_density_score"],
        0.70,
        0.82,
        ("FOUL_DENSITY_LOW", "FOUL_DENSITY_MED", "FOUL_DENSITY_HIGH"),
    )
    out["wide_duel_bucket"] = bucket(
        out["fixture_wide_duel_score"],
        0.88,
        0.94,
        ("WIDE_DUEL_LOW", "WIDE_DUEL_MED", "WIDE_DUEL_HIGH"),
    )
    out["territory_bucket"] = bucket(
        out["fixture_territorial_stress_score"],
        0.65,
        0.80,
        ("TERRITORY_LOW", "TERRITORY_MED", "TERRITORY_HIGH"),
    )
    out["ref_cards_bucket"] = bucket(
        out["ref_cards_per_match"],
        4.0,
        5.25,
        ("REF_CARDS_LOW", "REF_CARDS_MED", "REF_CARDS_HIGH"),
    )
    out["role_context_cell"] = (
        out["tactical_role"].fillna("UNKNOWN").astype(str)
        + "|"
        + out["formation_pressure_bucket"].astype(str)
        + "|"
        + out["foul_density_bucket"].astype(str)
        + "|"
        + out["wide_duel_bucket"].astype(str)
    )
    return out


def summarize(rows: pd.DataFrame, group_cols: list[str], policy_lookup: dict[str, dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for key, group in rows.groupby(["market"] + group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        market = str(key[0])
        policy = policy_lookup[market]
        hit_col = str(policy["hit_col"])
        baseline = rows[rows["market"].eq(market)]
        baseline_hit = float(baseline[hit_col].mean()) if not baseline.empty else np.nan
        if len(group) < 20:
            continue
        hit_rate = float(group[hit_col].mean())
        months, stable_months, stable_share = month_stability(group, hit_col, baseline_hit)
        label = "DO_NOT_USE"
        lift = hit_rate - baseline_hit
        if len(group) >= 300 and months >= 8 and hit_rate >= policy["core_hit"] and lift >= 0 and stable_share >= 0.60:
            label = "FOULED_CONTEXT_CORE"
        elif len(group) >= 120 and months >= 6 and hit_rate >= policy["ready_hit"] and lift >= 0 and stable_share >= 0.50:
            label = "FOULED_CONTEXT_READY"
        elif len(group) >= 60 and months >= 4 and lift >= 0:
            label = "FOULED_CONTEXT_WATCH"
        record = {
            "market": market,
            "display_market": policy["display_market"],
            **dict(zip(group_cols, key[1:])),
            "rows": int(len(group)),
            "hit_rate": hit_rate,
            "baseline_hit": baseline_hit,
            "lift_vs_policy_lane": lift,
            "months_ge_min": months,
            "stable_months_vs_policy_lane": stable_months,
            "stable_month_share_vs_policy_lane": stable_share,
            "context_cell_label": label,
        }
        records.append(record)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values(
        ["market", "context_cell_label", "hit_rate", "rows"],
        ascending=[True, True, False, False],
    )


def markdown_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "_No rows._"
    work = df.head(max_rows)
    lines = ["| " + " | ".join(work.columns) + " |", "| " + " | ".join(["---"] * len(work.columns)) + " |"]
    for _, row in work.iterrows():
        values = []
        for col in work.columns:
            value = row[col]
            if isinstance(value, float):
                value = round(value, 4)
            if pd.isna(value):
                value = ""
            values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    outdir: Path,
    signal_rows: pd.DataFrame,
    role_cells: pd.DataFrame,
    context_cells: pd.DataFrame,
    formation_cells: pd.DataFrame,
) -> None:
    counts = (
        signal_rows.groupby(["market", "shadow_stage"], dropna=False)
        .agg(rows=("fixture_key", "size"), hit_ge1=("actual_fouled_ge1", "mean"), hit_ge2=("actual_fouled_ge2", "mean"))
        .reset_index()
        if not signal_rows.empty
        else pd.DataFrame()
    )
    lines = [
        "# Player Fouled Formation Context Cell Audit",
        "",
        "Research-only audit for formation/context cells inside strict fouled-player interaction lanes.",
        "",
        "## Safety",
        "- No priced player-prop odds.",
        "- No deploy routing, tiers, slips, or production rulebook changes.",
        "- Context labels are dashboard/watch intelligence only.",
        "",
        "## Strict Policy Lane Counts",
        markdown_table(counts),
        "",
        "## Best Role Context Cells",
        markdown_table(role_cells[role_cells["context_cell_label"].ne("DO_NOT_USE")].head(40)),
        "",
        "## Best Context Buckets",
        markdown_table(context_cells[context_cells["context_cell_label"].ne("DO_NOT_USE")].head(40)),
        "",
        "## Best Formation Buckets",
        markdown_table(formation_cells[formation_cells["context_cell_label"].ne("DO_NOT_USE")].head(40)),
        "",
        "## Interpretation",
        "- Prefer `FOULED_CONTEXT_CORE` only after repeated live outcome tracking agrees.",
        "- `Fouled 0.5+` is the only current dashboard-prominence candidate.",
        "- `Fouled 1.5+` remains secondary watch unless live outcomes improve.",
    ]
    (outdir / "PLAYER_FOULED_FORMATION_CONTEXT_CELL_AUDIT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored", type=Path, default=DEFAULT_SCORED)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    if not args.scored.exists():
        raise SystemExit(f"Missing scored rows: {args.scored}")
    args.outdir.mkdir(parents=True, exist_ok=True)
    scored = pd.read_csv(args.scored, low_memory=False)
    signal_rows = pd.concat([apply_policy(scored, market, policy) for market, policy in POLICIES.items()], ignore_index=True)

    role_cells = summarize(signal_rows, ["tactical_role", "role_context_cell"], POLICIES)
    context_cells = summarize(
        signal_rows,
        ["formation_pressure_bucket", "foul_density_bucket", "wide_duel_bucket", "territory_bucket", "ref_cards_bucket"],
        POLICIES,
    )
    formation_cells = summarize(signal_rows, ["team_formation", "opponent_formation", "formation_pressure_bucket"], POLICIES)

    signal_rows.to_csv(args.outdir / "PLAYER_FOULED_FORMATION_CONTEXT_SIGNAL_ROWS.csv", index=False)
    role_cells.to_csv(args.outdir / "PLAYER_FOULED_FORMATION_CONTEXT_ROLE_CELLS.csv", index=False)
    context_cells.to_csv(args.outdir / "PLAYER_FOULED_FORMATION_CONTEXT_BUCKET_CELLS.csv", index=False)
    formation_cells.to_csv(args.outdir / "PLAYER_FOULED_FORMATION_CONTEXT_FORMATION_CELLS.csv", index=False)
    write_report(args.outdir, signal_rows, role_cells, context_cells, formation_cells)

    print(f"WROTE {args.outdir}")
    print(f"signal_rows={len(signal_rows)}")
    print(
        signal_rows.groupby(["market", "shadow_stage"], dropna=False)
        .size()
        .reset_index(name="rows")
        .to_string(index=False)
    )
    for label, df in [("role", role_cells), ("context", context_cells), ("formation", formation_cells)]:
        print(f"\n[{label}]")
        if df.empty:
            print("no rows")
        else:
            print(df[df["context_cell_label"].ne("DO_NOT_USE")].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
