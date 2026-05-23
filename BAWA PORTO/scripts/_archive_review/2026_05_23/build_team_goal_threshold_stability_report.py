#!/usr/bin/env python3
"""Build league/team threshold stability report for team-goal shadow markets.

Research-only. Consumes the model-only team-goal shadow backtest outputs and
classifies league/team cells for live shadow monitoring. This does not change
deploy routing or source prediction files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_BACKTEST_DIR = Path("reports/2026-05-06/team_goal_shadow_market_backtest")
DEFAULT_OUTDIR = Path("reports/2026-05-06/team_goal_threshold_stability")

PROMOTION_POLICY_ALLOWLIST = {"TG15_PREMIUM", "TG15_CORE_WATCH"}


def num(values) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


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


def scorecard(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        hit = num(group["correct"])
        graded = int(hit.notna().sum())
        wins = int(hit.eq(1).sum())
        row = dict(zip(group_cols, keys, strict=False))
        row.update(
            {
                "rows": int(len(group)),
                "graded": graded,
                "wins": wins,
                "losses": int(hit.eq(0).sum()),
                "hit_rate": wins / graded if graded else np.nan,
                "avg_model_prob": float(num(group["model_prob"]).mean()),
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
                "rows": int(group["rows"].sum()),
                "graded": graded,
                "wins": wins,
                "losses": int(group["losses"].sum()),
                "hit_rate": wins / graded if graded else np.nan,
                "median_rows_per_window": float(group["rows"].median()),
                "windows_below_80": int(group["hit_rate"].lt(0.80).sum()),
                "windows_below_70": int(group["hit_rate"].lt(0.70).sum()),
                "mean_window_hit_rate": float(group["hit_rate"].mean()),
                "median_window_hit_rate": float(group["hit_rate"].median()),
                "p25_window_hit_rate": float(group["hit_rate"].quantile(0.25)),
                "p10_window_hit_rate": float(group["hit_rate"].quantile(0.10)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def classify_league(row: pd.Series) -> str:
    policy = str(row.get("shadow_policy", ""))
    product = str(row.get("shadow_product", ""))
    graded = int(row.get("graded", 0))
    windows = int(row.get("active_windows", 0))
    hit = float(row.get("hit_rate", np.nan))
    p25 = float(row.get("p25_window_hit_rate", np.nan))
    below70 = int(row.get("windows_below_70", 0))

    if product not in {"HOME_TEAM_OVER_1_5_SHADOW", "AWAY_TEAM_OVER_1_5_SHADOW"}:
        return "WATCH_ONLY_NOT_PROMOTION"
    if policy not in PROMOTION_POLICY_ALLOWLIST:
        return "OBSERVE"
    if graded >= 50 and windows >= 25 and hit >= 0.90 and p25 >= 0.75 and below70 <= max(2, windows * 0.15):
        return "SHADOW_CORE"
    if graded >= 25 and windows >= 15 and hit >= 0.88 and p25 >= 0.65:
        return "SHADOW_CONFIRM"
    if graded >= 10 and hit >= 0.90:
        return "MICRO_ONLY"
    return "OBSERVE"


def classify_team(row: pd.Series) -> str:
    policy = str(row.get("shadow_policy", ""))
    product = str(row.get("shadow_product", ""))
    graded = int(row.get("graded", 0))
    windows = int(row.get("active_windows", 0))
    hit = float(row.get("hit_rate", np.nan))
    p25 = float(row.get("p25_window_hit_rate", np.nan))

    if product not in {"HOME_TEAM_OVER_1_5_SHADOW", "AWAY_TEAM_OVER_1_5_SHADOW"}:
        return "WATCH_ONLY_NOT_PROMOTION"
    if policy not in PROMOTION_POLICY_ALLOWLIST:
        return "OBSERVE"
    if graded >= 20 and windows >= 15 and hit >= 0.90 and p25 >= 0.70:
        return "TEAM_CORE"
    if graded >= 10 and windows >= 8 and hit >= 0.85:
        return "TEAM_CONFIRM"
    if graded >= 5 and hit >= 0.90:
        return "MICRO_ONLY"
    return "OBSERVE"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backtest-dir", default=str(DEFAULT_BACKTEST_DIR))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    args = parser.parse_args()

    backtest_dir = Path(args.backtest_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    candidates_path = backtest_dir / "team_goal_shadow_market_candidates.csv"
    if not candidates_path.exists():
        raise SystemExit(f"Missing candidates file: {candidates_path}")

    candidates = pd.read_csv(candidates_path)
    candidates["shadow_team_name"] = candidates["shadow_team_name"].fillna("")

    league = window_stability(candidates, ["shadow_product", "shadow_policy", "league"])
    league["stability_bucket"] = league.apply(classify_league, axis=1)
    league = league.sort_values(["stability_bucket", "hit_rate", "graded"], ascending=[True, False, False])
    league.to_csv(outdir / "team_goal_threshold_stability_by_league.csv", index=False)

    team = window_stability(candidates, ["shadow_product", "shadow_policy", "league", "shadow_team_name"])
    team = team[team["shadow_team_name"].astype("string").str.len().gt(0)].copy()
    team["stability_bucket"] = team.apply(classify_team, axis=1)
    team = team.sort_values(["stability_bucket", "hit_rate", "graded"], ascending=[True, False, False])
    team.to_csv(outdir / "team_goal_threshold_stability_by_team.csv", index=False)

    bucket_counts = (
        pd.concat(
            [
                league.assign(level="league"),
                team.assign(level="team"),
            ],
            ignore_index=True,
            sort=False,
        )
        .groupby(["level", "stability_bucket"], dropna=False)
        .size()
        .reset_index(name="cells")
        .sort_values(["level", "stability_bucket"])
    )
    bucket_counts.to_csv(outdir / "team_goal_threshold_stability_bucket_counts.csv", index=False)

    league_core = league[league["stability_bucket"].isin(["SHADOW_CORE", "SHADOW_CONFIRM"])].head(40)
    team_core = team[team["stability_bucket"].isin(["TEAM_CORE", "TEAM_CONFIRM"])].head(40)
    observe_risk = league[
        league["shadow_policy"].isin(PROMOTION_POLICY_ALLOWLIST)
        & league["stability_bucket"].eq("OBSERVE")
        & league["graded"].ge(20)
    ].sort_values(["hit_rate", "graded"], ascending=[True, False]).head(25)

    summary = [
        "# Team Goal Threshold Stability Report",
        "",
        "Research-only league/team classifier for model-only team-goal shadow markets.",
        "",
        "## Bucket Counts",
        markdown_table(bucket_counts),
        "",
        "## League Cells To Keep Shadowing",
        markdown_table(
            league_core[
                [
                    "stability_bucket",
                    "shadow_product",
                    "shadow_policy",
                    "league",
                    "graded",
                    "wins",
                    "hit_rate",
                    "active_windows",
                    "p25_window_hit_rate",
                    "windows_below_70",
                ]
            ]
        ),
        "",
        "## Team Cells To Keep Shadowing",
        markdown_table(
            team_core[
                [
                    "stability_bucket",
                    "shadow_product",
                    "shadow_policy",
                    "league",
                    "shadow_team_name",
                    "graded",
                    "wins",
                    "hit_rate",
                    "active_windows",
                    "p25_window_hit_rate",
                ]
            ]
        ),
        "",
        "## Observe Risks",
        markdown_table(
            observe_risk[
                [
                    "shadow_product",
                    "shadow_policy",
                    "league",
                    "graded",
                    "wins",
                    "hit_rate",
                    "active_windows",
                    "p25_window_hit_rate",
                    "windows_below_70",
                ]
            ]
        ),
        "",
        "## Operating Read",
        "",
        "- Keep all cells shadow-only until repeated live-board QA is boring.",
        "- Treat `TG15_PREMIUM` as the first-class team-goal signal.",
        "- Treat `TG15_CORE_WATCH` as a strong watch lane for dominant teams with softer total-goal mass.",
        "- Keep `TG25_MONSTER` and `MATCH_OVER_3_5_SHADOW` out of promotion until much tighter league/team proof exists.",
        "- Use league/team buckets for watch-table prioritisation, not deployment permission.",
    ]
    (outdir / "team_goal_threshold_stability_report.md").write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    print(f"[ok] league_cells={len(league)} team_cells={len(team)}")
    print(f"[ok] wrote {outdir}")


if __name__ == "__main__":
    main()
