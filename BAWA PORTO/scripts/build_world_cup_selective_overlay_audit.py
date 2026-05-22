#!/usr/bin/env python3
"""Audit macro vs full-stack World Cup trainer-native predictions row by row.

Research only. This script does not train models, write ModelStore artifacts, or
touch production routing. It consumes the prediction CSV emitted by
`run_world_cup_trainer_native_research.py` and isolates where the full-stack
sidecar fixed or broke the macro pick.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_PREDICTIONS = Path(
    "data_sources/footystats_world_cup/trainer_native_research_runs/"
    "catboost_xgb_2018_train_2022_test_v2/world_cup_trainer_native_predictions.csv"
)
DEFAULT_BACKBUILT_SIDECAR = Path(
    "data_sources/footystats_world_cup/historical_full_stack_backbuild/"
    "world_cup_historical_backbuilt_fixture_intelligence_sidecar.csv"
)
DEFAULT_API_PLAYER_POWER_SIDECAR = Path(
    "data_sources/footystats_world_cup/api_lagged_player_power_backbuild/"
    "world_cup_api_lagged_player_power_fixture_sidecar.csv"
)
DEFAULT_OUTDIR = Path("data_sources/footystats_world_cup/selective_overlay_audit")

KEY_COLS = [
    "fixture_key",
    "season",
    "match_date",
    "home_team_name",
    "away_team_name",
    "home_team_goal_count",
    "away_team_goal_count",
    "engine",
    "market",
    "target",
]

GROUP_MACRO = "trainer_native_macro"
GROUP_FULL = "trainer_native_full_stack_backbuilt"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            out[col] = out[col].astype("string").fillna("")
    lines = [
        "| " + " | ".join(out.columns) + " |",
        "| " + " | ".join(["---"] * len(out.columns)) + " |",
    ]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in out.columns) + " |")
    return "\n".join(lines)


def confidence_for(row: pd.Series, prefix: str) -> float | None:
    pred = row.get(f"{prefix}_pred")
    if pd.isna(pred):
        return None
    try:
        pred_int = int(pred)
    except (TypeError, ValueError):
        return None
    col = f"{prefix}_prob_class_{pred_int}"
    value = row.get(col)
    if pd.isna(value):
        return None
    return float(value)


def label_prediction(market: str, pred: object) -> str:
    if pd.isna(pred):
        return ""
    try:
        pred_int = int(pred)
    except (TypeError, ValueError):
        return str(pred)
    if market == "ftr":
        return {0: "HOME", 1: "DRAW", 2: "AWAY"}.get(pred_int, str(pred_int))
    return "YES" if pred_int == 1 else "NO"


def classify_flip(row: pd.Series) -> str:
    macro_hit = int(row["macro_hit"])
    full_hit = int(row["full_hit"])
    flipped = bool(row["prediction_flipped_flag"])
    if flipped and macro_hit == 0 and full_hit == 1:
        return "FULLSTACK_FIXES_MACRO_LOSS"
    if flipped and macro_hit == 1 and full_hit == 0:
        return "FULLSTACK_BREAKS_MACRO_WIN"
    if flipped and macro_hit == 1 and full_hit == 1:
        return "DIFFERENT_PICK_BOTH_HIT"
    if flipped and macro_hit == 0 and full_hit == 0:
        return "DIFFERENT_PICK_BOTH_MISS"
    if not flipped and macro_hit == 1 and full_hit == 1:
        return "AGREE_HIT"
    return "AGREE_MISS"


def recommendation(row: pd.Series) -> str:
    net = int(row.get("net_flip_value", 0))
    full_delta = float(row.get("full_minus_macro_accuracy", 0.0))
    flips = int(row.get("prediction_flips", 0))
    if flips == 0:
        return "NO_FLIP_SIGNAL"
    if net > 0 and full_delta > 0:
        return "SELECTIVE_OVERLAY_CANDIDATE"
    if net == 0 and full_delta >= 0:
        return "WATCHLIST_NEEDS_MORE_SAMPLE"
    if net < 0:
        return "MACRO_DEFAULT_FULLSTACK_CAUTION"
    return "AUDIT_ONLY"


def add_sidecar_context(audit: pd.DataFrame, sidecar_paths: list[Path]) -> pd.DataFrame:
    joined = audit.copy()
    joined["sidecar_joined_flag"] = 0
    patterns = [
        "historical_player_intel",
        "historical_venue",
        "source_status",
        "local_h2h",
        "backbuilt_recent",
        "backbuilt_qualifier",
        "api_wc_",
        "_delta",
    ]
    for sidecar_path in sidecar_paths:
        if not sidecar_path.exists():
            continue
        sidecar = pd.read_csv(sidecar_path, low_memory=False)
        before_cols = set(joined.columns)
        context_cols = ["fixture_key"]
        for col in sidecar.columns:
            if col == "fixture_key" or col in joined.columns:
                continue
            if any(p in col for p in patterns):
                context_cols.append(col)
        context_cols = list(dict.fromkeys(context_cols))
        if len(context_cols) <= 1:
            continue
        joined = joined.merge(sidecar[context_cols], on="fixture_key", how="left")
        new_cols = [c for c in joined.columns if c not in before_cols]
        if new_cols:
            joined["sidecar_joined_flag"] = (
                joined["sidecar_joined_flag"].astype(int)
                | joined[new_cols].notna().any(axis=1).astype(int)
            )
    return joined


def _series_or_missing(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series(index=df.index, dtype=object)


def bucketize_conditions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["h2h_available_bucket"] = pd.to_numeric(out.get("historical_local_h2h_match_count"), errors="coerce").fillna(0).map(
        lambda x: "H2H_AVAILABLE" if x > 0 else "NO_H2H"
    )
    qgd = pd.to_numeric(out.get("backbuilt_qualifier_goal_diff_per_match_delta"), errors="coerce")
    rgd = pd.to_numeric(out.get("backbuilt_recent_goal_diff_per_match_delta"), errors="coerce")
    pp = pd.to_numeric(out.get("api_wc_player_power_score_diff"), errors="coerce")
    prior = pd.to_numeric(out.get("api_wc_player_power_any_prior_rate"), errors="coerce")
    out["qualifier_goal_diff_edge_bucket"] = pd.cut(
        qgd,
        bins=[-99, -0.75, -0.25, 0.25, 0.75, 99],
        labels=["AWAY_BIG", "AWAY_SMALL", "BALANCED", "HOME_SMALL", "HOME_BIG"],
    ).astype("string").fillna("UNKNOWN")
    out["recent_goal_diff_edge_bucket"] = pd.cut(
        rgd,
        bins=[-99, -0.75, -0.25, 0.25, 0.75, 99],
        labels=["AWAY_BIG", "AWAY_SMALL", "BALANCED", "HOME_SMALL", "HOME_BIG"],
    ).astype("string").fillna("UNKNOWN")
    out["api_player_power_edge_bucket"] = pd.cut(
        pp,
        bins=[-99, -0.75, -0.25, 0.25, 0.75, 99],
        labels=["AWAY_BIG", "AWAY_SMALL", "BALANCED", "HOME_SMALL", "HOME_BIG"],
    ).astype("string").fillna("NO_API_PLAYER_POWER")
    out["api_player_prior_bucket"] = prior.map(lambda x: "API_PRIOR_READY" if pd.notna(x) and x > 0 else "NO_API_PRIOR")
    out["player_snapshot_bucket"] = _series_or_missing(out, "historical_player_intel_backbuild_status").fillna("MISSING")
    return out


def build_audit(
    predictions_path: Path,
    sidecar_paths: list[Path],
    outdir: Path,
    macro_group: str,
    overlay_group: str,
) -> None:
    preds = pd.read_csv(predictions_path, low_memory=False)
    required = set(KEY_COLS + ["ablation_group", "pred", "hit"])
    missing = sorted(required - set(preds.columns))
    if missing:
        raise SystemExit(f"Prediction file missing required columns: {missing}")

    subset = preds[preds["ablation_group"].isin([macro_group, overlay_group])].copy()
    prob_cols = [c for c in subset.columns if c.startswith("prob_class_")]
    value_cols = ["pred", "hit"] + prob_cols

    macro = subset[subset["ablation_group"].eq(macro_group)][KEY_COLS + value_cols].copy()
    full = subset[subset["ablation_group"].eq(overlay_group)][KEY_COLS + value_cols].copy()
    macro = macro.rename(columns={c: f"macro_{c}" for c in value_cols})
    full = full.rename(columns={c: f"full_{c}" for c in value_cols})

    audit = macro.merge(full, on=KEY_COLS, how="inner")
    audit["prediction_flipped_flag"] = (audit["macro_pred"] != audit["full_pred"]).astype(int)
    audit["macro_confidence"] = audit.apply(lambda r: confidence_for(r, "macro"), axis=1)
    audit["full_confidence"] = audit.apply(lambda r: confidence_for(r, "full"), axis=1)
    audit["confidence_delta_full_minus_macro"] = audit["full_confidence"] - audit["macro_confidence"]
    audit["macro_pick_label"] = audit.apply(lambda r: label_prediction(r["market"], r["macro_pred"]), axis=1)
    audit["full_pick_label"] = audit.apply(lambda r: label_prediction(r["market"], r["full_pred"]), axis=1)
    audit["target_label"] = audit.apply(lambda r: label_prediction(r["market"], r["target"]), axis=1)
    audit["flip_result"] = audit.apply(classify_flip, axis=1)
    audit["fullstack_fix_flag"] = audit["flip_result"].eq("FULLSTACK_FIXES_MACRO_LOSS").astype(int)
    audit["fullstack_break_flag"] = audit["flip_result"].eq("FULLSTACK_BREAKS_MACRO_WIN").astype(int)
    audit["net_flip_value"] = audit["fullstack_fix_flag"] - audit["fullstack_break_flag"]
    audit["macro_group"] = macro_group
    audit["overlay_group"] = overlay_group
    audit = add_sidecar_context(audit, sidecar_paths)
    audit = bucketize_conditions(audit)

    summary = (
        audit.groupby(["engine", "market"], dropna=False)
        .agg(
            rows=("fixture_key", "count"),
            macro_accuracy=("macro_hit", "mean"),
            full_accuracy=("full_hit", "mean"),
            prediction_flips=("prediction_flipped_flag", "sum"),
            fullstack_fixes=("fullstack_fix_flag", "sum"),
            fullstack_breaks=("fullstack_break_flag", "sum"),
            net_flip_value=("net_flip_value", "sum"),
            avg_macro_confidence=("macro_confidence", "mean"),
            avg_full_confidence=("full_confidence", "mean"),
        )
        .reset_index()
    )
    summary["full_minus_macro_accuracy"] = summary["full_accuracy"] - summary["macro_accuracy"]
    summary["selective_overlay_recommendation"] = summary.apply(recommendation, axis=1)
    summary["macro_group"] = macro_group
    summary["overlay_group"] = overlay_group
    summary = summary.sort_values(["net_flip_value", "full_minus_macro_accuracy"], ascending=[False, False])

    condition_summary = (
        audit[audit["prediction_flipped_flag"].eq(1)]
        .groupby(
            [
                "engine",
                "market",
                "h2h_available_bucket",
                "qualifier_goal_diff_edge_bucket",
                "recent_goal_diff_edge_bucket",
                "api_player_power_edge_bucket",
                "api_player_prior_bucket",
                "player_snapshot_bucket",
            ],
            dropna=False,
        )
        .agg(
            flips=("fixture_key", "count"),
            fullstack_fixes=("fullstack_fix_flag", "sum"),
            fullstack_breaks=("fullstack_break_flag", "sum"),
            net_flip_value=("net_flip_value", "sum"),
        )
        .reset_index()
        .sort_values(["net_flip_value", "flips"], ascending=[False, False])
    )

    examples = audit[audit["prediction_flipped_flag"].eq(1)].copy()
    examples = examples.sort_values(["net_flip_value", "engine", "market", "match_date"], ascending=[False, True, True, True])

    outdir.mkdir(parents=True, exist_ok=True)
    audit_path = outdir / "world_cup_selective_overlay_fixture_audit.csv"
    summary_path = outdir / "world_cup_selective_overlay_summary.csv"
    condition_path = outdir / "world_cup_selective_overlay_condition_summary.csv"
    examples_path = outdir / "world_cup_selective_overlay_flip_examples.csv"

    audit.to_csv(audit_path, index=False)
    summary.to_csv(summary_path, index=False)
    condition_summary.to_csv(condition_path, index=False)
    examples.to_csv(examples_path, index=False)

    top_cols = [
        "engine",
        "market",
        "rows",
        "macro_accuracy",
        "full_accuracy",
        "full_minus_macro_accuracy",
        "prediction_flips",
        "fullstack_fixes",
        "fullstack_breaks",
        "net_flip_value",
        "selective_overlay_recommendation",
        "overlay_group",
    ]
    md = [
        "# World Cup Selective Overlay Audit",
        "",
        f"Research-only row-by-row audit comparing `{macro_group}` predictions against `{overlay_group}` predictions.",
        "",
        "## Overlay Summary",
        "",
        markdown_table(summary[top_cols]),
        "",
        "## Best Flip Conditions",
        "",
        markdown_table(condition_summary.head(20)),
        "",
        "## Outputs",
        "",
        f"- Fixture audit: `{audit_path}`",
        f"- Summary: `{summary_path}`",
        f"- Condition summary: `{condition_path}`",
        f"- Flip examples: `{examples_path}`",
        "",
        "## Interpretation Guardrail",
        "",
        "A positive overlay result here means the intelligence layer corrected historical holdout picks in this narrow 2022 test. "
        "It is still not a production policy until the same pattern is stable across backbuilt, timestamp-safe historical windows.",
        "",
    ]
    (outdir / "SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[ok] rows={len(audit)} flips={int(audit['prediction_flipped_flag'].sum())}")
    print(f"[ok] wrote {outdir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--sidecar", type=Path, action="append", default=None)
    parser.add_argument("--macro-group", default=GROUP_MACRO)
    parser.add_argument("--overlay-group", default=GROUP_FULL)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    sidecars = args.sidecar or [DEFAULT_BACKBUILT_SIDECAR, DEFAULT_API_PLAYER_POWER_SIDECAR]
    build_audit(args.predictions, sidecars, args.outdir, args.macro_group, args.overlay_group)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
