#!/usr/bin/env python3
"""Build a research-only FTR C5 side-shape ring classifier.

The goal is to recover FTR shape by league using meta probability, correct-score
side support, model margin, and power/side coherence. This script does not
change deploy tiers or production rulebooks.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_SCORED_GLOB = (
    "predictions_output/"
    "walk_forward_phase8h_value_layer_full_relock_2026_04_21_r3/"
    "w*/03_scored/DEPLOY_COMBINED_SCORED_*.csv"
)
DEFAULT_OUTDIR = Path("reports/2026-05-06/ftr_c5_side_shape_ring_classifier")
WINDOW_RE = re.compile(r"^w\d{3}_")

FTR_ELITE_FLOOR = 0.9264
FTR_STANDARD_FLOOR = 0.8336

BASE_COLUMNS = [
    "league",
    "match_date",
    "home_team_name",
    "away_team_name",
    "fixture_key",
    "market",
    "bookie_pick",
    "selection",
    "deploy_tier",
    "tier",
    "source_tier_file",
    "__source_deploy_file",
    "p_meta_ftr",
    "model_p_for_bookie",
    "model_p_for_bookie_xgb",
    "model_top_pick",
    "model_top_pick_xgb",
    "ftr_pick_xgb",
    "ftr_margin",
    "ftr_margin_xgb",
    "pick_side_margin_top3",
    "pick_side_mass_top3",
    "cs_top3_match_ftr",
    "cs_top3_support_side",
    "cs_top3_support_mass",
    "cs_top3_side_margin",
    "cat_xgb_grid_ftr_agreement_count",
    "grid_vs_cat_ftr_gap",
    "grid_vs_xgb_ftr_gap",
    "power_diff",
    "home_power_rating",
    "away_power_rating",
    "value_edge",
    "bookie_od",
    "bookie_implied_used",
    "actual_ftr",
    "ftr_hit",
    "context_reason_codes",
]

RING_CONFIGS = [
    {
        "ring": "C5_META_SIDE_CORE",
        "p_meta_ftr": 0.90,
        "pick_side_margin_top3": 0.04,
        "pick_side_mass_top3": 0.18,
        "ftr_margin": 0.10,
        "side_power_abs": 4.0,
        "agreement_count": 2,
        "grid_gap_max": 0.28,
    },
    {
        "ring": "C5_META_SIDE_CONFIRM",
        "p_meta_ftr": 0.85,
        "pick_side_margin_top3": 0.03,
        "pick_side_mass_top3": 0.16,
        "ftr_margin": 0.08,
        "side_power_abs": 2.0,
        "agreement_count": 2,
        "grid_gap_max": 0.32,
    },
    {
        "ring": "C5_SIDE_SHAPE_STRICT",
        "p_meta_ftr": 0.80,
        "pick_side_margin_top3": 0.07,
        "pick_side_mass_top3": 0.22,
        "ftr_margin": 0.14,
        "side_power_abs": 5.0,
        "agreement_count": 1,
        "grid_gap_max": 0.35,
    },
    {
        "ring": "C5_GRID_META_COHERENT",
        "p_meta_ftr": 0.84,
        "pick_side_margin_top3": 0.02,
        "pick_side_mass_top3": 0.15,
        "ftr_margin": 0.06,
        "side_power_abs": 1.0,
        "agreement_count": 2,
        "grid_gap_max": 0.20,
    },
    {
        "ring": "C5_VALUE_SIDE_SHAPE",
        "p_meta_ftr": 0.82,
        "pick_side_margin_top3": 0.05,
        "pick_side_mass_top3": 0.18,
        "ftr_margin": 0.10,
        "side_power_abs": 3.0,
        "agreement_count": 1,
        "grid_gap_max": 0.35,
        "value_edge": 0.06,
    },
]


def num(values) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def num_col(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return num(df[column])


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    text = df.copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            text[col] = text[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            text[col] = text[col].astype("string").fillna("")
    lines = [
        "| " + " | ".join(str(c) for c in text.columns) + " |",
        "| " + " | ".join(["---"] * len(text.columns)) + " |",
    ]
    for _, row in text.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in text.columns) + " |")
    return "\n".join(lines)


def window_id_from_path(path: Path) -> str:
    for part in path.parts:
        if WINDOW_RE.match(part):
            return part
    return path.stem


def read_ftr_rows(files: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in files:
        header = pd.read_csv(path, nrows=0)
        usecols = [col for col in BASE_COLUMNS if col in header.columns]
        if not usecols:
            continue
        frame = pd.read_csv(path, usecols=usecols, low_memory=False)
        if "market" not in frame.columns:
            continue
        frame = frame[frame["market"].astype("string").str.lower().eq("ftr")].copy()
        if frame.empty:
            continue
        frame["window_id"] = window_id_from_path(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    df["selection_norm"] = (
        df.get("selection", df.get("bookie_pick", ""))
        .astype("string")
        .str.upper()
        .str.strip()
    )
    df["hit"] = num(df.get("ftr_hit"))
    return df[df["hit"].notna()].copy()


def side_power_pass(selection: pd.Series, power_diff: pd.Series, threshold: float) -> pd.Series:
    side = selection.astype("string").str.upper()
    diff = num(power_diff)
    home = side.eq("HOME") & diff.ge(threshold)
    away = side.eq("AWAY") & diff.le(-threshold)
    draw = side.eq("DRAW") & diff.abs().le(max(threshold, 3.0))
    return home | away | draw


def build_ring_candidates(rows: pd.DataFrame) -> pd.DataFrame:
    outputs = []
    for config in RING_CONFIGS:
        mask = pd.Series(True, index=rows.index)
        mask &= num(rows.get("p_meta_ftr")).ge(config["p_meta_ftr"])
        mask &= num(rows.get("pick_side_margin_top3")).ge(config["pick_side_margin_top3"])
        mask &= num(rows.get("pick_side_mass_top3")).ge(config["pick_side_mass_top3"])
        mask &= num(rows.get("ftr_margin")).ge(config["ftr_margin"])
        mask &= side_power_pass(rows["selection_norm"], rows.get("power_diff"), config["side_power_abs"])
        if "cat_xgb_grid_ftr_agreement_count" in rows.columns:
            mask &= num(rows["cat_xgb_grid_ftr_agreement_count"]).ge(config["agreement_count"])
        for gap_col in ["grid_vs_cat_ftr_gap", "grid_vs_xgb_ftr_gap"]:
            if gap_col in rows.columns:
                mask &= num(rows[gap_col]).le(config["grid_gap_max"])
        if "value_edge" in config and "value_edge" in rows.columns:
            mask &= num(rows["value_edge"]).ge(config["value_edge"])

        selected = rows[mask].copy()
        if selected.empty:
            continue
        selected["c5_ring"] = config["ring"]
        selected["c5_p_meta_ftr_floor"] = config["p_meta_ftr"]
        selected["c5_top3_margin_floor"] = config["pick_side_margin_top3"]
        selected["c5_top3_mass_floor"] = config["pick_side_mass_top3"]
        selected["c5_ftr_margin_floor"] = config["ftr_margin"]
        selected["c5_side_power_abs_floor"] = config["side_power_abs"]
        outputs.append(selected)
    if not outputs:
        return pd.DataFrame()
    return pd.concat(outputs, ignore_index=True, sort=False)


def scorecard(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        hit = num(group["hit"])
        graded = int(hit.notna().sum())
        wins = int(hit.eq(1).sum())
        if "bookie_od" in group.columns:
            profit = num(group["bookie_od"]).where(hit.eq(1), 0).sub(1).where(hit.eq(1), -1)
        else:
            profit = pd.Series(dtype=float)
        row = dict(zip(group_cols, keys, strict=False))
        row.update(
            {
                "rows": int(len(group)),
                "graded": graded,
                "wins": wins,
                "losses": int(hit.eq(0).sum()),
                "hit_rate": wins / graded if graded else np.nan,
                "profit_1u": float(profit.sum()) if len(profit) else np.nan,
                "roi_1u": float(profit.mean()) if len(profit) else np.nan,
                "avg_p_meta_ftr": float(num_col(group, "p_meta_ftr").mean()),
                "avg_pick_side_margin_top3": float(num_col(group, "pick_side_margin_top3").mean()),
                "avg_pick_side_mass_top3": float(num_col(group, "pick_side_mass_top3").mean()),
                "avg_ftr_margin": float(num_col(group, "ftr_margin").mean()),
                "avg_abs_power_diff": float(num_col(group, "power_diff").abs().mean()),
                "active_windows": int(group["window_id"].nunique()) if "window_id" in group.columns else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def window_stability(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    by_window = scorecard(df, group_cols + ["window_id"])
    rows = []
    for keys, group in by_window.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        graded = int(group["graded"].sum())
        wins = int(group["wins"].sum())
        row = dict(zip(group_cols, keys, strict=False))
        row.update(
            {
                "active_windows": int(group["window_id"].nunique()),
                "graded": graded,
                "wins": wins,
                "losses": int(group["losses"].sum()),
                "hit_rate": wins / graded if graded else np.nan,
                "profit_1u": float(group["profit_1u"].sum()),
                "roi_1u": float(group["profit_1u"].sum() / graded) if graded else np.nan,
                "median_rows_per_window": float(group["rows"].median()),
                "windows_below_ftr_standard": int(group["hit_rate"].lt(FTR_STANDARD_FLOOR).sum()),
                "windows_below_70": int(group["hit_rate"].lt(0.70).sum()),
                "mean_window_hit_rate": float(group["hit_rate"].mean()),
                "median_window_hit_rate": float(group["hit_rate"].median()),
                "p25_window_hit_rate": float(group["hit_rate"].quantile(0.25)),
                "p10_window_hit_rate": float(group["hit_rate"].quantile(0.10)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def classify_ring(row: pd.Series) -> str:
    graded = int(row.get("graded", 0))
    windows = int(row.get("active_windows", 0))
    hit = float(row.get("hit_rate", np.nan))
    p25 = float(row.get("p25_window_hit_rate", np.nan))
    below_standard = int(row.get("windows_below_ftr_standard", 0))

    if graded >= 60 and windows >= 25 and hit >= 0.90 and p25 >= 0.75 and below_standard <= max(2, windows * 0.20):
        return "RESTORE_NOW_SHADOW"
    if graded >= 30 and windows >= 15 and hit >= FTR_STANDARD_FLOOR and p25 >= 0.65:
        return "RESTORE_WITH_CONFIRM_SHADOW"
    if graded >= 10 and hit >= 0.90:
        return "MICRO_ONLY"
    return "OBSERVE"


def choose_best_policy(league_rings: pd.DataFrame) -> pd.DataFrame:
    if league_rings.empty:
        return league_rings
    priority = {
        "RESTORE_NOW_SHADOW": 0,
        "RESTORE_WITH_CONFIRM_SHADOW": 1,
        "MICRO_ONLY": 2,
        "OBSERVE": 3,
    }
    ranked = league_rings.copy()
    ranked["bucket_priority"] = ranked["c5_bucket"].map(priority).fillna(99)
    ranked = ranked.sort_values(
        ["league", "bucket_priority", "hit_rate", "graded"],
        ascending=[True, True, False, False],
    )
    return ranked.groupby("league", as_index=False).head(1).drop(columns=["bucket_priority"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-glob", default=DEFAULT_SCORED_GLOB)
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    args = parser.parse_args()

    root = Path.cwd()
    files = sorted(root.glob(args.scored_glob))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ftr_rows = read_ftr_rows(files)
    if ftr_rows.empty:
        raise SystemExit("No graded FTR rows found.")

    base_summary = scorecard(ftr_rows, ["deploy_tier", "tier"])
    base_summary.to_csv(outdir / "ftr_c5_source_deploy_tier_scorecard.csv", index=False)

    candidates = build_ring_candidates(ftr_rows)
    candidates.to_csv(outdir / "ftr_c5_side_shape_candidate_rows.csv", index=False)
    if candidates.empty:
        raise SystemExit("No C5 candidate rows generated.")

    ring_summary = scorecard(candidates, ["c5_ring"])
    ring_summary.to_csv(outdir / "ftr_c5_ring_scorecard.csv", index=False)

    league_rings = window_stability(candidates, ["league", "c5_ring"])
    league_rings["c5_bucket"] = league_rings.apply(classify_ring, axis=1)
    league_rings = league_rings.sort_values(
        ["c5_bucket", "hit_rate", "graded"], ascending=[True, False, False]
    )
    league_rings.to_csv(outdir / "ftr_c5_league_ring_classifier.csv", index=False)

    selected = choose_best_policy(league_rings)
    selected.to_csv(outdir / "ftr_c5_recommended_league_policy.csv", index=False)

    side_summary = scorecard(candidates, ["c5_ring", "selection_norm"])
    side_summary.to_csv(outdir / "ftr_c5_side_scorecard.csv", index=False)

    token_rows = []
    if "context_reason_codes" in candidates.columns:
        for _, row in candidates[["league", "c5_ring", "hit", "context_reason_codes"]].iterrows():
            tokens = str(row.get("context_reason_codes", "")).replace("|", ";").split(";")
            for token in tokens:
                token = token.strip()
                if token and token.lower() != "nan":
                    token_rows.append(
                        {
                            "league": row["league"],
                            "c5_ring": row["c5_ring"],
                            "reason_token": token,
                            "hit": row["hit"],
                        }
                    )
    token_df = pd.DataFrame(token_rows)
    if not token_df.empty:
        token_score = scorecard(token_df, ["league", "c5_ring", "reason_token"])
        token_score = token_score[token_score["graded"].ge(10)]
        token_score = token_score.sort_values(["hit_rate", "graded"], ascending=[False, False])
        token_score.to_csv(outdir / "ftr_c5_reason_token_hit_table.csv", index=False)

    bucket_counts = (
        selected.groupby("c5_bucket", dropna=False)
        .size()
        .reset_index(name="leagues")
        .sort_values("c5_bucket")
    )
    summary = [
        "# FTR C5 Side-Shape Ring Classifier",
        "",
        "Research-only classifier over Phase 8H 139-window scored exports.",
        "No deploy tiers, rulebooks, or scored rows were mutated.",
        "",
        "## Source Baseline By Existing Tier",
        markdown_table(base_summary.sort_values(["deploy_tier", "tier"])),
        "",
        "## C5 Ring Scorecard",
        markdown_table(ring_summary.sort_values("hit_rate", ascending=False)),
        "",
        "## Recommended League Policy Counts",
        markdown_table(bucket_counts),
        "",
        "## Recommended League Policy",
        markdown_table(
            selected[
                [
                    "league",
                    "c5_bucket",
                    "c5_ring",
                    "graded",
                    "wins",
                    "hit_rate",
                    "active_windows",
                    "p25_window_hit_rate",
                    "windows_below_ftr_standard",
                    "roi_1u",
                ]
            ].sort_values(["c5_bucket", "hit_rate"], ascending=[True, False])
        ),
        "",
        "## Interpretation",
        (
            "Treat RESTORE_NOW_SHADOW as a forward-facing shadow candidate only. "
            "RESTORE_WITH_CONFIRM_SHADOW needs a second confirmer pass. "
            "MICRO_ONLY is useful for copy/research but not volume restoration. "
            "OBSERVE should not be promoted."
        ),
    ]
    (outdir / "ftr_c5_side_shape_ring_classifier.md").write_text("\n".join(summary), encoding="utf-8")

    print(f"WROTE {outdir}")
    print(f"source_ftr_rows={len(ftr_rows)} candidate_rows={len(candidates)}")


if __name__ == "__main__":
    main()
