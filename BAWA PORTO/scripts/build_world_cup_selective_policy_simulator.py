#!/usr/bin/env python3
"""Simulate selective World Cup research policies over trainer-native predictions.

Research only. This consumes `run_world_cup_trainer_native_research.py` outputs
and tests market-specific selection policies without changing ModelStore,
deploy routing, or production rulebooks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RUN_DIR = Path(
    "data_sources/footystats_world_cup/trainer_native_research_runs/"
    "catboost_xgb_api_player_power_2018_train_2022_test"
)
DEFAULT_MERGED = Path("Matches/__merged__/World_Cup__merged.csv")
DEFAULT_OUTDIR = Path("data_sources/footystats_world_cup/selective_policy_simulator")

GROUP_MACRO = "trainer_native_macro"
GROUP_HIST_FULL = "trainer_native_full_stack_backbuilt"
GROUP_API_ONLY = "trainer_native_api_player_power"
GROUP_FULL_API = "trainer_native_full_stack_api_player_power"

MARKETS = ["ftr", "btts", "over25", "home_ge2", "away_ge2", "any_team_ge2"]


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
        "| " + " | ".join(str(c) for c in out.columns) + " |",
        "| " + " | ".join(["---"] * len(out.columns)) + " |",
    ]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in out.columns) + " |")
    return "\n".join(lines)


def load_predictions(path: Path, merged_path: Path) -> pd.DataFrame:
    preds = pd.read_csv(path, low_memory=False)
    stage_cols = [
        "fixture_key",
        "tournament_stage",
        "group_matchday",
        "is_knockout_stage",
        "is_first_group_match",
    ]
    if merged_path.exists():
        merged = pd.read_csv(merged_path, low_memory=False, usecols=lambda c: c in set(stage_cols))
        preds = preds.merge(merged[stage_cols].drop_duplicates("fixture_key"), on="fixture_key", how="left")
    for col in ["target", "pred", "hit", "home_team_goal_count", "away_team_goal_count"]:
        if col in preds.columns:
            preds[col] = pd.to_numeric(preds[col], errors="coerce")
    return preds


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def best_macro_policy(results: pd.DataFrame) -> pd.DataFrame:
    ok = results[(results["status"].eq("OK")) & (results["ablation_group"].eq(GROUP_MACRO))].copy()
    if ok.empty:
        return pd.DataFrame()
    idx = ok.sort_values(["accuracy", "log_loss"], ascending=[False, True]).groupby("market").head(1).index
    rows = []
    for _, row in ok.loc[idx].iterrows():
        rows.append(
            {
                "policy_name": "macro_best",
                "market": row["market"],
                "engine": row["engine"],
                "ablation_group": GROUP_MACRO,
                "policy_role": "BASELINE_MACRO_BEST_ENGINE",
            }
        )
    return pd.DataFrame(rows)


def build_policy_table(results: pd.DataFrame) -> pd.DataFrame:
    macro_best = best_macro_policy(results)
    if macro_best.empty:
        raise SystemExit("Could not build macro policy: no trainer_native_macro OK rows found.")

    def macro_choice(market: str) -> dict:
        row = macro_best[macro_best["market"].eq(market)].iloc[0].to_dict()
        return {"market": market, "engine": row["engine"], "ablation_group": row["ablation_group"]}

    policies: list[dict[str, object]] = []
    policies.extend(macro_best.to_dict("records"))

    # Strict promotion: only the clearly positive API player-power lane moves.
    for market in MARKETS:
        choice = macro_choice(market)
        role = "MACRO_DEFAULT"
        if market == "away_ge2":
            choice = {"market": market, "engine": "xgboost", "ablation_group": GROUP_FULL_API}
            role = "PROMOTE_FULL_STACK_API_PLAYER_POWER"
        policies.append({"policy_name": "selective_core_away_ge2", **choice, "policy_role": role})

    # Include the small watchlist positives so they can be audited separately.
    for market in MARKETS:
        choice = macro_choice(market)
        role = "MACRO_DEFAULT"
        if market == "away_ge2":
            choice = {"market": market, "engine": "xgboost", "ablation_group": GROUP_FULL_API}
            role = "PROMOTE_FULL_STACK_API_PLAYER_POWER"
        elif market == "btts":
            choice = {"market": market, "engine": "xgboost", "ablation_group": GROUP_FULL_API}
            role = "WATCHLIST_FULL_STACK_API_PLAYER_POWER"
        elif market == "home_ge2":
            choice = {"market": market, "engine": "catboost", "ablation_group": GROUP_FULL_API}
            role = "WATCHLIST_FULL_STACK_API_PLAYER_POWER"
        policies.append({"policy_name": "selective_plus_watchlist", **choice, "policy_role": role})

    # Separate API-only any-team-goal audit. Full-stack API was harmful here.
    for market in MARKETS:
        choice = macro_choice(market)
        role = "MACRO_DEFAULT"
        if market == "away_ge2":
            choice = {"market": market, "engine": "xgboost", "ablation_group": GROUP_FULL_API}
            role = "PROMOTE_FULL_STACK_API_PLAYER_POWER"
        elif market == "any_team_ge2":
            choice = {"market": market, "engine": "xgboost", "ablation_group": GROUP_API_ONLY}
            role = "AUDIT_API_ONLY_PLAYER_POWER"
        policies.append({"policy_name": "selective_plus_any_team_api_only_audit", **choice, "policy_role": role})

    # Deliberately bad comparator: apply full-stack API wherever the current run
    # made it tempting, to show why broad promotion is unsafe.
    for market in MARKETS:
        choice = macro_choice(market)
        role = "MACRO_DEFAULT"
        if market in {"away_ge2", "btts", "ftr", "home_ge2"}:
            choice = {
                "market": market,
                "engine": "xgboost" if market in {"away_ge2", "btts"} else "catboost",
                "ablation_group": GROUP_FULL_API,
            }
            role = "BROAD_FULL_STACK_API_COMPARATOR"
        policies.append({"policy_name": "broad_full_stack_api_comparator", **choice, "policy_role": role})

    out = pd.DataFrame(policies)
    return out.sort_values(["policy_name", "market"]).reset_index(drop=True)


def select_predictions(preds: pd.DataFrame, policy: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, rule in policy.iterrows():
        frame = preds[
            preds["market"].eq(rule["market"])
            & preds["engine"].eq(rule["engine"])
            & preds["ablation_group"].eq(rule["ablation_group"])
        ].copy()
        frame["policy_name"] = rule["policy_name"]
        frame["policy_role"] = rule["policy_role"]
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def selected_for_all_policies(preds: pd.DataFrame, policies: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [select_predictions(preds, rules) for _, rules in policies.groupby("policy_name", sort=False)],
        ignore_index=True,
        sort=False,
    )


def metric_summary(selected: pd.DataFrame, baseline: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    key = ["fixture_key", "market"]
    base = baseline[key + ["pred", "hit"]].rename(columns={"pred": "baseline_pred", "hit": "baseline_hit"})
    joined = selected.merge(base, on=key, how="left")
    joined["prediction_flipped_vs_macro_best"] = joined["pred"].ne(joined["baseline_pred"]).astype(int)
    joined["fix_vs_macro_best"] = (
        joined["prediction_flipped_vs_macro_best"].eq(1) & joined["baseline_hit"].eq(0) & joined["hit"].eq(1)
    ).astype(int)
    joined["break_vs_macro_best"] = (
        joined["prediction_flipped_vs_macro_best"].eq(1) & joined["baseline_hit"].eq(1) & joined["hit"].eq(0)
    ).astype(int)
    joined["net_vs_macro_best"] = joined["fix_vs_macro_best"] - joined["break_vs_macro_best"]

    policy_summary = (
        joined.groupby("policy_name", dropna=False)
        .agg(
            rows=("fixture_key", "count"),
            markets=("market", "nunique"),
            fixtures=("fixture_key", "nunique"),
            accuracy=("hit", "mean"),
            baseline_accuracy=("baseline_hit", "mean"),
            accuracy_delta_vs_macro_best=("hit", lambda s: np.nan),
            flips_vs_macro_best=("prediction_flipped_vs_macro_best", "sum"),
            fixes_vs_macro_best=("fix_vs_macro_best", "sum"),
            breaks_vs_macro_best=("break_vs_macro_best", "sum"),
            net_vs_macro_best=("net_vs_macro_best", "sum"),
        )
        .reset_index()
    )
    policy_summary["accuracy_delta_vs_macro_best"] = policy_summary["accuracy"] - policy_summary["baseline_accuracy"]
    policy_summary = policy_summary.sort_values(["accuracy_delta_vs_macro_best", "net_vs_macro_best"], ascending=[False, False])

    market_summary = (
        joined.groupby(["policy_name", "market", "engine", "ablation_group", "policy_role"], dropna=False)
        .agg(
            rows=("fixture_key", "count"),
            accuracy=("hit", "mean"),
            baseline_accuracy=("baseline_hit", "mean"),
            flips_vs_macro_best=("prediction_flipped_vs_macro_best", "sum"),
            fixes_vs_macro_best=("fix_vs_macro_best", "sum"),
            breaks_vs_macro_best=("break_vs_macro_best", "sum"),
            net_vs_macro_best=("net_vs_macro_best", "sum"),
        )
        .reset_index()
    )
    market_summary["accuracy_delta_vs_macro_best"] = market_summary["accuracy"] - market_summary["baseline_accuracy"]
    market_summary = market_summary.sort_values(
        ["policy_name", "accuracy_delta_vs_macro_best", "net_vs_macro_best"], ascending=[True, False, False]
    )

    stage_summary = (
        joined.groupby(["policy_name", "market", "tournament_stage"], dropna=False)
        .agg(
            rows=("fixture_key", "count"),
            accuracy=("hit", "mean"),
            baseline_accuracy=("baseline_hit", "mean"),
            flips_vs_macro_best=("prediction_flipped_vs_macro_best", "sum"),
            fixes_vs_macro_best=("fix_vs_macro_best", "sum"),
            breaks_vs_macro_best=("break_vs_macro_best", "sum"),
            net_vs_macro_best=("net_vs_macro_best", "sum"),
        )
        .reset_index()
    )
    stage_summary["accuracy_delta_vs_macro_best"] = stage_summary["accuracy"] - stage_summary["baseline_accuracy"]
    return policy_summary, market_summary, stage_summary, joined


def write_summary(
    outdir: Path,
    policy_summary: pd.DataFrame,
    market_summary: pd.DataFrame,
    stage_summary: pd.DataFrame,
) -> None:
    market_view = market_summary[
        market_summary["policy_name"].isin(
            [
                "selective_core_away_ge2",
                "selective_plus_watchlist",
                "selective_plus_any_team_api_only_audit",
            ]
        )
    ][
        [
            "policy_name",
            "market",
            "engine",
            "ablation_group",
            "policy_role",
            "accuracy",
            "baseline_accuracy",
            "accuracy_delta_vs_macro_best",
            "fixes_vs_macro_best",
            "breaks_vs_macro_best",
            "net_vs_macro_best",
        ]
    ]
    away_stage = stage_summary[
        (stage_summary["policy_name"].eq("selective_core_away_ge2")) & (stage_summary["market"].eq("away_ge2"))
    ][
        [
            "tournament_stage",
            "rows",
            "accuracy",
            "baseline_accuracy",
            "accuracy_delta_vs_macro_best",
            "fixes_vs_macro_best",
            "breaks_vs_macro_best",
            "net_vs_macro_best",
        ]
    ]
    lines = [
        "# World Cup Selective Policy Simulator",
        "",
        "Research-only simulation of market-specific policy choices over trainer-native World Cup predictions.",
        "",
        "## Policy Summary",
        "",
        markdown_table(policy_summary),
        "",
        "## Market Decisions",
        "",
        markdown_table(market_view),
        "",
        "## Away GE2 Stage Check",
        "",
        markdown_table(away_stage),
        "",
        "## Outputs",
        "",
        f"- `{outdir / 'world_cup_selective_policy_rules.csv'}`",
        f"- `{outdir / 'world_cup_selective_policy_selected_predictions.csv'}`",
        f"- `{outdir / 'world_cup_selective_policy_summary.csv'}`",
        f"- `{outdir / 'world_cup_selective_policy_market_summary.csv'}`",
        f"- `{outdir / 'world_cup_selective_policy_stage_summary.csv'}`",
        "",
        "## Guardrail",
        "",
        "This is not a deploy policy. It is a 2018-train / 2022-holdout research comparator. "
        "Promotion requires more folds, better pre-tournament player priors for first group matches, and timestamped lineup/injury truth.",
        "",
    ]
    (outdir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--merged", type=Path, default=DEFAULT_MERGED)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions_path = args.predictions or args.run_dir / "world_cup_trainer_native_predictions.csv"
    results_path = args.results or args.run_dir / "world_cup_trainer_native_results.csv"
    preds = load_predictions(predictions_path, args.merged)
    results = load_results(results_path)
    policies = build_policy_table(results)
    selected = selected_for_all_policies(preds, policies)
    baseline_rules = policies[policies["policy_name"].eq("macro_best")]
    baseline = select_predictions(preds, baseline_rules)
    policy_summary, market_summary, stage_summary, selected_joined = metric_summary(selected, baseline)

    args.outdir.mkdir(parents=True, exist_ok=True)
    policies.to_csv(args.outdir / "world_cup_selective_policy_rules.csv", index=False)
    selected_joined.to_csv(args.outdir / "world_cup_selective_policy_selected_predictions.csv", index=False)
    policy_summary.to_csv(args.outdir / "world_cup_selective_policy_summary.csv", index=False)
    market_summary.to_csv(args.outdir / "world_cup_selective_policy_market_summary.csv", index=False)
    stage_summary.to_csv(args.outdir / "world_cup_selective_policy_stage_summary.csv", index=False)
    write_summary(args.outdir, policy_summary, market_summary, stage_summary)
    print(f"[ok] policies={policies['policy_name'].nunique()} selected_rows={len(selected_joined)}")
    print(f"[ok] wrote {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
