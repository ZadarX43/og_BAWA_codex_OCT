#!/usr/bin/env python3
"""Classify Phase 8H full-estate recovery rings from the old Phase 8E maps.

This research-only script applies the locked FootyStats-era Phase 8E BTTS/OU25
league threshold maps to a row-level replay export, including OBSERVE rows when
present. It then labels each market/league as:

- RESTORE_NOW
- RESTORE_WITH_CONFIRM
- MICRO_ONLY
- OBSERVE

No production policy files are read or changed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_ROW_LEVEL = Path(
    "reports/2026-05-06/phase8h_full_estate_c4_sweeps/phase8h_replay_row_level_scored.csv"
)
DEFAULT_OUTDIR = Path("reports/2026-05-06/phase8h_c4_recovery_rings")


BTTS_PHASE8E_MAP = {
    "England FA Cup": {"threshold": 0.80, "locked_rows": 439, "locked_hit_rate": 0.9977},
    "Europa Conference": {"threshold": 0.88, "locked_rows": 461, "locked_hit_rate": 0.9826},
    "Champions League": {"threshold": 0.80, "locked_rows": 258, "locked_hit_rate": 0.9806},
    "Brazil Serie A": {"threshold": 0.85, "locked_rows": 83, "locked_hit_rate": 0.9759},
    "Italy Serie A": {"threshold": 0.85, "locked_rows": 179, "locked_hit_rate": 0.9665},
    "Spain La Liga": {"threshold": 0.80, "locked_rows": 248, "locked_hit_rate": 0.9556},
    "France Ligue 1": {"threshold": 0.88, "locked_rows": 121, "locked_hit_rate": 0.9174},
    "Scotland Premiership": {"threshold": 0.80, "locked_rows": 110, "locked_hit_rate": 0.9091},
    "Belgium Pro": {"threshold": 0.88, "locked_rows": 158, "locked_hit_rate": 0.9051},
    "Norway Eliteserien": {"threshold": 0.80, "locked_rows": 138, "locked_hit_rate": 0.8986},
    "Netherlands Eredivisie": {"threshold": 0.88, "locked_rows": 382, "locked_hit_rate": 0.8953},
    "Japan J1": {"threshold": 0.80, "locked_rows": 293, "locked_hit_rate": 0.8908},
    "USA MLS": {"threshold": 0.85, "locked_rows": 512, "locked_hit_rate": 0.8750},
}

OU25_PHASE8E_MAP = {
    "England FA Cup": {"threshold": 0.80, "locked_rows": 532, "locked_hit_rate": 0.9962},
    "Europa Conference": {"threshold": 0.80, "locked_rows": 501, "locked_hit_rate": 0.9741},
    "Germany Bundesliga": {"threshold": 0.88, "locked_rows": 276, "locked_hit_rate": 0.9710},
    "Europa League": {"threshold": 0.85, "locked_rows": 243, "locked_hit_rate": 0.9630},
    "Brazil Serie A": {"threshold": 0.80, "locked_rows": 69, "locked_hit_rate": 0.9565},
    "Champions League": {"threshold": 0.80, "locked_rows": 385, "locked_hit_rate": 0.9532},
    "Portugal Liga": {"threshold": 0.90, "locked_rows": 213, "locked_hit_rate": 0.9531},
    "Netherlands Eredivisie": {"threshold": 0.88, "locked_rows": 301, "locked_hit_rate": 0.9468},
    "Scotland Premiership": {"threshold": 0.90, "locked_rows": 72, "locked_hit_rate": 0.9444},
    "Spain La Liga": {"threshold": 0.80, "locked_rows": 352, "locked_hit_rate": 0.9375},
    "Japan J1": {"threshold": 0.80, "locked_rows": 241, "locked_hit_rate": 0.9336},
    "Norway Eliteserien": {"threshold": 0.85, "locked_rows": 117, "locked_hit_rate": 0.9145},
    "USA MLS": {"threshold": 0.80, "locked_rows": 526, "locked_hit_rate": 0.9106},
}

CONFIRM_FEATURES = {
    "btts": [
        ("p_meta_btts", ">="),
        ("model_p_for_bookie", ">="),
        ("cs_mass_btts_yes", ">="),
        ("cs_btts_margin", ">="),
        ("both_teams_2plus_mass", ">="),
        ("grid_vs_cat_btts_gap", "<="),
        ("grid_vs_xgb_btts_gap", "<="),
        ("cat_xgb_grid_btts_agreement_count", ">="),
        ("btts_fts_sum", "<="),
        ("p00_est", "<="),
        ("exp_goals_sum", ">="),
    ],
    "ou25": [
        ("p_meta_ou25", ">="),
        ("model_p_for_bookie", ">="),
        ("cs_mass_over25", ">="),
        ("cs_ou25_margin", ">="),
        ("mass_over25_via_one_sided_rout", ">="),
        ("grid_vs_cat_ou25_gap", "<="),
        ("grid_vs_xgb_ou25_gap", "<="),
        ("cat_xgb_grid_ou25_agreement_count", ">="),
        ("top3_over_count", ">="),
        ("bookie_lambda_total_fit", ">="),
        ("exp_goals_sum", ">="),
    ],
}

QUANTILES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def num(series: pd.Series | Any) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def metric(group: pd.DataFrame, hit_col: str = "correct") -> dict[str, Any]:
    hit = num(group.get(hit_col, pd.Series(dtype=float)))
    graded = int(hit.notna().sum())
    wins = float((hit == 1).sum())
    odds = num(group.get("bookie_od", pd.Series(dtype=float)))
    profit = np.where(hit == 1, odds - 1.0, np.where(hit == 0, -1.0, np.nan))
    return {
        "rows": int(len(group)),
        "graded": graded,
        "wins": wins,
        "losses": int((hit == 0).sum()),
        "hit_rate": wins / graded if graded else np.nan,
        "avg_bookie_od": odds.mean() if odds.notna().any() else np.nan,
        "profit": float(np.nansum(profit)),
        "roi": float(np.nansum(profit) / graded) if graded else np.nan,
    }


def candidate_rows(df: pd.DataFrame, market: str, league: str, threshold: float) -> pd.DataFrame:
    feat = "p_meta_btts" if market == "btts" else "p_meta_ou25"
    selection = "YES" if market == "btts" else "OVER25"
    g = df[
        df["market_norm"].astype("string").eq(market)
        & df["league"].astype("string").eq(league)
        & num(df.get(feat, np.nan)).ge(threshold)
    ].copy()
    pick = g.get("selection", g.get("bookie_pick", "")).astype("string").str.upper()
    return g[pick.eq(selection)].copy()


def apply_op(series: pd.Series, op: str, threshold: float) -> pd.Series:
    s = num(series)
    if op == "<=":
        return s.le(threshold).fillna(False)
    return s.ge(threshold).fillna(False)


def find_best_confirm(candidate: pd.DataFrame, market: str, target_floor: float) -> dict[str, Any]:
    if candidate.empty:
        return {
            "confirm_feature": "",
            "confirm_op": "",
            "confirm_threshold": np.nan,
            "confirm_graded": 0,
            "confirm_hit_rate": np.nan,
            "confirm_coverage": np.nan,
        }

    base_graded = int(num(candidate["correct"]).notna().sum())
    min_rows = max(20, int(round(base_graded * 0.15)))
    rows = []
    for feature, op in CONFIRM_FEATURES[market]:
        if feature not in candidate.columns:
            continue
        vals = num(candidate[feature]).dropna()
        if vals.empty:
            continue
        for q in QUANTILES:
            threshold = round(float(vals.quantile(q)), 6)
            sub = candidate[apply_op(candidate[feature], op, threshold)]
            m = metric(sub)
            if m["graded"] < min_rows:
                continue
            rows.append(
                {
                    "confirm_feature": feature,
                    "confirm_op": op,
                    "confirm_threshold": threshold,
                    "confirm_graded": m["graded"],
                    "confirm_hit_rate": m["hit_rate"],
                    "confirm_coverage": m["graded"] / base_graded if base_graded else np.nan,
                    "confirm_roi": m["roi"],
                    "confirm_profit": m["profit"],
                    "confirm_pass_floor": bool(m["hit_rate"] >= target_floor)
                    if pd.notna(m["hit_rate"])
                    else False,
                }
            )

    if not rows:
        return {
            "confirm_feature": "",
            "confirm_op": "",
            "confirm_threshold": np.nan,
            "confirm_graded": 0,
            "confirm_hit_rate": np.nan,
            "confirm_coverage": np.nan,
        }

    out = pd.DataFrame(rows)
    out["floor_gap"] = out["confirm_hit_rate"] - target_floor
    out = out.sort_values(
        ["confirm_pass_floor", "confirm_hit_rate", "confirm_graded", "confirm_coverage"],
        ascending=[False, False, False, False],
    )
    return out.iloc[0].to_dict()


def classify_row(row: dict[str, Any], target_floor: float) -> str:
    graded = int(row["graded"])
    hit_rate = row["hit_rate"]
    locked_hit = row["locked_hit_rate"]
    locked_rows = int(row["locked_rows"])
    coverage = row["graded_vs_locked_rows"]
    delta = row["hit_delta_vs_locked"]
    confirm_hit = row.get("confirm_hit_rate", np.nan)
    confirm_graded = int(row.get("confirm_graded", 0) or 0)

    if graded == 0:
        return "OBSERVE"
    if graded < max(20, int(locked_rows * 0.2)):
        if pd.notna(hit_rate) and hit_rate >= target_floor:
            return "MICRO_ONLY"
        return "OBSERVE"
    if pd.notna(hit_rate) and hit_rate >= target_floor and delta >= -0.015:
        return "RESTORE_NOW"
    if pd.notna(confirm_hit) and confirm_hit >= target_floor and confirm_graded >= 20:
        return "RESTORE_WITH_CONFIRM"
    if pd.notna(hit_rate) and hit_rate >= target_floor and coverage < 0.50:
        return "MICRO_ONLY"
    return "OBSERVE"


def replay_map(df: pd.DataFrame, market: str, locked_map: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    target_floor = 0.90 if market == "btts" else 0.91
    for league, cfg in locked_map.items():
        threshold = float(cfg["threshold"])
        cand = candidate_rows(df, market, league, threshold)
        m = metric(cand)
        tier_counts = cand.get("deploy_tier", pd.Series(dtype=str)).value_counts(dropna=False).to_dict()
        row = {
            "market": market,
            "league": league,
            "threshold": threshold,
            "locked_rows": int(cfg["locked_rows"]),
            "locked_hit_rate": float(cfg["locked_hit_rate"]),
            "target_floor": target_floor,
            **m,
            "graded_vs_locked_rows": m["graded"] / float(cfg["locked_rows"]) if cfg["locked_rows"] else np.nan,
            "hit_delta_vs_locked": m["hit_rate"] - float(cfg["locked_hit_rate"])
            if pd.notna(m["hit_rate"])
            else np.nan,
            "elite_rows": int(tier_counts.get("ELITE", 0)),
            "standard_rows": int(tier_counts.get("STANDARD", 0)),
            "observe_rows": int(tier_counts.get("OBSERVE", 0)),
        }
        row.update(find_best_confirm(cand, market, target_floor))
        row["recovery_ring"] = classify_row(row, target_floor)
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data._"
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
        else:
            show[col] = show[col].astype("string").fillna("")
    headers = list(show.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in headers) + " |")
    return "\n".join(lines)


def write_summary(outdir: Path, classified: pd.DataFrame) -> None:
    ring_summary = (
        classified.groupby(["market", "recovery_ring"], dropna=False)
        .agg(
            leagues=("league", "count"),
            graded=("graded", "sum"),
            wins=("wins", "sum"),
            observe_rows=("observe_rows", "sum"),
            locked_rows=("locked_rows", "sum"),
        )
        .reset_index()
    )
    ring_summary["hit_rate"] = ring_summary["wins"] / ring_summary["graded"]
    ring_summary["graded_vs_locked_rows"] = ring_summary["graded"] / ring_summary["locked_rows"]

    total = (
        classified.groupby("market", dropna=False)
        .agg(graded=("graded", "sum"), wins=("wins", "sum"), locked_rows=("locked_rows", "sum"))
        .reset_index()
    )
    total["hit_rate"] = total["wins"] / total["graded"]
    total["graded_vs_locked_rows"] = total["graded"] / total["locked_rows"]

    policy_rows = []
    for _, row in classified.iterrows():
        ring = row["recovery_ring"]
        if ring == "RESTORE_NOW":
            graded = row["graded"]
            wins = row["wins"]
            hit_rate = row["hit_rate"]
            source = "phase8e_threshold"
        elif ring == "RESTORE_WITH_CONFIRM":
            graded = row.get("confirm_graded", 0)
            hit_rate = row.get("confirm_hit_rate", np.nan)
            wins = graded * hit_rate if pd.notna(hit_rate) else np.nan
            source = f"{row.get('confirm_feature', '')} {row.get('confirm_op', '')} {row.get('confirm_threshold', '')}"
        else:
            graded = 0
            wins = 0
            hit_rate = np.nan
            source = "no_restore"
        policy_rows.append(
            {
                "market": row["market"],
                "league": row["league"],
                "recovery_ring": ring,
                "policy_source": source,
                "policy_graded": int(graded) if pd.notna(graded) else 0,
                "policy_wins": float(wins) if pd.notna(wins) else 0.0,
                "policy_hit_rate": hit_rate,
                "locked_rows": row["locked_rows"],
            }
        )
    policy = pd.DataFrame(policy_rows)
    policy_summary = (
        policy.groupby("market", dropna=False)
        .agg(policy_graded=("policy_graded", "sum"), policy_wins=("policy_wins", "sum"), locked_rows=("locked_rows", "sum"))
        .reset_index()
    )
    policy_summary["policy_hit_rate"] = policy_summary["policy_wins"] / policy_summary["policy_graded"]
    policy_summary["policy_vs_locked_rows"] = policy_summary["policy_graded"] / policy_summary["locked_rows"]

    cols = [
        "market",
        "league",
        "recovery_ring",
        "threshold",
        "graded",
        "hit_rate",
        "locked_rows",
        "locked_hit_rate",
        "graded_vs_locked_rows",
        "hit_delta_vs_locked",
        "elite_rows",
        "standard_rows",
        "observe_rows",
        "confirm_feature",
        "confirm_op",
        "confirm_threshold",
        "confirm_graded",
        "confirm_hit_rate",
        "confirm_coverage",
    ]

    lines = [
        "# Phase 8H C4 Recovery Rings",
        "",
        "Research-only classification from the locked Phase 8E FootyStats-era maps applied to the full Phase 8H row-level estate.",
        "",
        "## Market Totals",
        markdown_table(total),
        "",
        "## Ring Summary",
        markdown_table(ring_summary.sort_values(["market", "recovery_ring"])),
        "",
        "## Recommended Ring Policy Rollup",
        markdown_table(policy_summary),
        "",
        "## League Classification",
        markdown_table(classified[cols].sort_values(["market", "recovery_ring", "league"])),
        "",
        "## Read",
        "",
        "- `RESTORE_NOW` means the old Phase 8E league threshold still clears the market floor and is close to the locked league hit rate.",
        "- `RESTORE_WITH_CONFIRM` means the old threshold is not enough by itself, but a C3-style confirmer restores the floor.",
        "- `MICRO_ONLY` means the signal exists but volume is too thin for broad restoration.",
        "- `OBSERVE` means no current restoration proof from this pass.",
        "",
    ]
    (outdir / "phase8h_c4_recovery_rings_summary.md").write_text("\n".join(lines), encoding="utf-8")
    ring_summary.to_csv(outdir / "phase8h_c4_ring_summary.csv", index=False)
    total.to_csv(outdir / "phase8h_c4_market_total_summary.csv", index=False)
    policy.to_csv(outdir / "phase8h_c4_recommended_ring_policy_by_league.csv", index=False)
    policy_summary.to_csv(outdir / "phase8h_c4_recommended_ring_policy_summary.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row-level", type=Path, default=DEFAULT_ROW_LEVEL)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.row_level, low_memory=False)

    classified = pd.concat(
        [
            replay_map(df, "btts", BTTS_PHASE8E_MAP),
            replay_map(df, "ou25", OU25_PHASE8E_MAP),
        ],
        ignore_index=True,
    )
    classified.to_csv(args.outdir / "phase8h_c4_restore_ring_classification.csv", index=False)
    write_summary(args.outdir, classified)
    print(f"[ok] wrote {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
