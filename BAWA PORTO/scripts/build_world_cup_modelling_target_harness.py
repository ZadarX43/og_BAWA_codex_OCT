#!/usr/bin/env python3
"""Build the World Cup modelling target harness and ablation readiness report.

This is research-only. It reads the canonical World Cup merged adapter for
historical labels and the 2026 feature matrix for current feature-group
coverage. It does not train production models or write ModelStore artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_MERGED = Path("Matches/__merged__/World_Cup__merged.csv")
DEFAULT_MATRIX = Path("data_sources/footystats_world_cup/research_feature_matrix_2026/world_cup_2026_research_feature_matrix.csv")
DEFAULT_OUTDIR = Path("data_sources/footystats_world_cup/modelling_target_harness")


FEATURE_GROUP_PATTERNS = {
    "macro_only": [
        "macro_",
        "home_macro_",
        "away_macro_",
    ],
    "macro_plus_qualifier": [
        "macro_",
        "home_macro_",
        "away_macro_",
        "qualifier_",
        "home_qualifier_",
        "away_qualifier_",
        "qualification_",
        "home_qualification_",
        "away_qualification_",
    ],
    "macro_plus_additions": [
        "macro_",
        "home_macro_",
        "away_macro_",
        "additions_",
        "home_additions_",
        "away_additions_",
    ],
    "macro_plus_fjelstul_history": [
        "macro_",
        "home_macro_",
        "away_macro_",
        "historical_",
        "home_all_wc_",
        "away_all_wc_",
        "home_modern_wc_",
        "away_modern_wc_",
        "home_recent_wc_",
        "away_recent_wc_",
        "fjelstul_",
        "home_fjelstul_",
        "away_fjelstul_",
    ],
    "full_stack": [
        "macro_",
        "home_macro_",
        "away_macro_",
        "qualifier_",
        "home_qualifier_",
        "away_qualifier_",
        "qualification_",
        "home_qualification_",
        "away_qualification_",
        "additions_",
        "home_additions_",
        "away_additions_",
        "historical_",
        "home_all_wc_",
        "away_all_wc_",
        "home_modern_wc_",
        "away_modern_wc_",
        "home_recent_wc_",
        "away_recent_wc_",
        "fjelstul_",
        "home_fjelstul_",
        "away_fjelstul_",
        "recent_history_",
        "home_recent_history_",
        "away_recent_history_",
        "local_h2h_",
        "venue_",
        "weather_",
        "travel_",
        "altitude_",
        "home_travel_",
        "away_travel_",
        "fixture_climate_",
    ],
}


IDENTITY_COLS = [
    "season",
    "api_fixture_id",
    "api_date",
    "api_round",
    "api_home_team_name",
    "api_away_team_name",
]


def market_pick_from_min_odds(row: pd.Series, mapping: dict[str, str]) -> str | None:
    available = {label: row[col] for label, col in mapping.items() if col in row.index and pd.notna(row[col]) and row[col] > 0}
    if not available:
        return None
    return min(available, key=available.get)


def build_targets(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()
    out["target_ftr"] = out["actual_ftr_label"].fillna("")
    out["target_btts_yes"] = pd.to_numeric(out["actual_btts_label"], errors="coerce")
    out["target_ou25_over"] = pd.to_numeric(out["actual_over25_label"], errors="coerce")
    out["target_home_tg15_over"] = (pd.to_numeric(out["home_team_goal_count"], errors="coerce") >= 2).astype(int)
    out["target_away_tg15_over"] = (pd.to_numeric(out["away_team_goal_count"], errors="coerce") >= 2).astype(int)
    out["target_any_team_tg15_over"] = (
        out["target_home_tg15_over"].eq(1) | out["target_away_tg15_over"].eq(1)
    ).astype(int)
    out["market_ftr_favourite"] = out.apply(
        lambda r: market_pick_from_min_odds(
            r,
            {
                "HOME": "odds_ft_home_team_win",
                "DRAW": "odds_ft_draw",
                "AWAY": "odds_ft_away_team_win",
            },
        ),
        axis=1,
    )
    out["market_btts_pick"] = out.apply(
        lambda r: market_pick_from_min_odds(r, {"YES": "odds_btts_yes", "NO": "odds_btts_no"}),
        axis=1,
    )
    out["market_ou25_pick"] = out.apply(
        lambda r: market_pick_from_min_odds(r, {"OVER": "odds_ft_over25", "UNDER": "odds_ft_under25"}),
        axis=1,
    )
    out["target_btts_pick_label"] = out["target_btts_yes"].map({1: "YES", 0: "NO"})
    out["target_ou25_pick_label"] = out["target_ou25_over"].map({1: "OVER", 0: "UNDER"})
    keep = [
        "fixture_key",
        "season",
        "match_date",
        "tournament_stage",
        "group_matchday",
        "home_team_name",
        "away_team_name",
        "home_team_goal_count",
        "away_team_goal_count",
        "total_goal_count",
        "target_ftr",
        "target_btts_yes",
        "target_ou25_over",
        "target_home_tg15_over",
        "target_away_tg15_over",
        "target_any_team_tg15_over",
        "market_ftr_favourite",
        "market_btts_pick",
        "market_ou25_pick",
        "target_btts_pick_label",
        "target_ou25_pick_label",
    ]
    return out[[c for c in keep if c in out.columns]].reset_index(drop=True)


def build_targets_long(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in wide.itertuples(index=False):
        base = {
            "fixture_key": r.fixture_key,
            "season": r.season,
            "match_date": r.match_date,
            "home_team_name": r.home_team_name,
            "away_team_name": r.away_team_name,
        }
        rows.extend(
            [
                {**base, "market": "FTR", "selection": r.target_ftr, "target": 1},
                {**base, "market": "BTTS", "selection": "YES", "target": r.target_btts_yes},
                {**base, "market": "OU25", "selection": "OVER", "target": r.target_ou25_over},
                {**base, "market": "HOME_TG15", "selection": "OVER", "target": r.target_home_tg15_over},
                {**base, "market": "AWAY_TG15", "selection": "OVER", "target": r.target_away_tg15_over},
                {**base, "market": "ANY_TEAM_TG15", "selection": "OVER", "target": r.target_any_team_tg15_over},
            ]
        )
    return pd.DataFrame(rows)


def build_market_baselines(targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baselines = [
        ("FTR_MARKET_FAVOURITE", "target_ftr", "market_ftr_favourite"),
        ("BTTS_MARKET_PRICE_PICK", "target_btts_pick_label", "market_btts_pick"),
        ("OU25_MARKET_PRICE_PICK", "target_ou25_pick_label", "market_ou25_pick"),
    ]
    for name, target_col, pred_col in baselines:
        valid = targets[target_col].notna() & targets[pred_col].notna()
        scoped = targets[valid]
        rows.append(
            {
                "baseline": name,
                "rows": len(scoped),
                "hit_rate": float((scoped[target_col] == scoped[pred_col]).mean()) if len(scoped) else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def columns_for_group(matrix: pd.DataFrame, group: str) -> list[str]:
    patterns = FEATURE_GROUP_PATTERNS[group]
    selected = []
    for col in matrix.columns:
        if col in IDENTITY_COLS:
            continue
        if any(col.startswith(prefix) for prefix in patterns):
            selected.append(col)
    return selected


def build_feature_group_outputs(matrix: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows = []
    for group in FEATURE_GROUP_PATTERNS:
        cols = columns_for_group(matrix, group)
        usable_numeric = [
            c
            for c in cols
            if pd.api.types.is_numeric_dtype(matrix[c]) and matrix[c].notna().sum() > 0
        ]
        coverage = float(matrix[usable_numeric].notna().mean().mean()) if usable_numeric else 0.0
        out_cols = [c for c in IDENTITY_COLS if c in matrix.columns] + cols
        matrix[out_cols].to_csv(outdir / f"world_cup_2026_features__{group}.csv", index=False)
        rows.append(
            {
                "ablation_group": group,
                "feature_columns": len(cols),
                "numeric_feature_columns": len(usable_numeric),
                "mean_numeric_nonnull_coverage": coverage,
                "fixtures": len(matrix),
                "historical_accuracy_backtest_ready": int(group in {"macro_only", "macro_plus_fjelstul_history"}),
                "notes": "Needs time-safe historical sidecar for 2018/2022 before judged by accuracy"
                if group not in {"macro_only", "macro_plus_fjelstul_history"}
                else "Can be compared against historical World_Cup__merged targets with compatible features",
            }
        )
    return pd.DataFrame(rows)


def write_summary(outdir: Path, targets: pd.DataFrame, long_targets: pd.DataFrame, baselines: pd.DataFrame, readiness: pd.DataFrame) -> None:
    target_rates = pd.DataFrame(
        [
            {"target": "BTTS_YES", "positive_rate": targets["target_btts_yes"].mean(), "rows": targets["target_btts_yes"].notna().sum()},
            {"target": "OU25_OVER", "positive_rate": targets["target_ou25_over"].mean(), "rows": targets["target_ou25_over"].notna().sum()},
            {"target": "HOME_TG15_OVER", "positive_rate": targets["target_home_tg15_over"].mean(), "rows": len(targets)},
            {"target": "AWAY_TG15_OVER", "positive_rate": targets["target_away_tg15_over"].mean(), "rows": len(targets)},
            {"target": "ANY_TEAM_TG15_OVER", "positive_rate": targets["target_any_team_tg15_over"].mean(), "rows": len(targets)},
        ]
    )
    target_rates.to_csv(outdir / "world_cup_historical_target_rates.csv", index=False)
    lines = [
        "# World Cup Modelling Target Harness",
        "",
        "Research-only target and feature-group harness for FTR, BTTS, OU25, and TG1.5.",
        "",
        "## Outputs",
        "",
        f"- `{outdir / 'world_cup_historical_targets_wide.csv'}`",
        f"- `{outdir / 'world_cup_historical_targets_long.csv'}`",
        f"- `{outdir / 'world_cup_historical_market_baselines.csv'}`",
        f"- `{outdir / 'world_cup_2026_ablation_readiness.csv'}`",
        f"- `{outdir / 'world_cup_2026_features__<ablation_group>.csv'}`",
        "",
        "## Historical Label Estate",
        "",
        f"- Historical labelled fixtures: {len(targets)}",
        f"- Long target rows: {len(long_targets)}",
        f"- Seasons: {', '.join(str(x) for x in sorted(targets['season'].dropna().unique()))}",
        "",
        "## Market Baselines",
        "",
        "| Baseline | Rows | Hit rate |",
        "|---|---:|---:|",
    ]
    for r in baselines.itertuples(index=False):
        hit = "" if pd.isna(r.hit_rate) else f"{r.hit_rate:.3f}"
        lines.append(f"| {r.baseline} | {int(r.rows)} | {hit} |")
    lines.extend(
        [
            "",
            "## Ablation Readiness",
            "",
            "| Group | Features | Numeric | 2026 coverage | Historical accuracy ready |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for r in readiness.itertuples(index=False):
        lines.append(
            f"| {r.ablation_group} | {int(r.feature_columns)} | {int(r.numeric_feature_columns)} | "
            f"{r.mean_numeric_nonnull_coverage:.3f} | {int(r.historical_accuracy_backtest_ready)} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Historical accuracy ablations for qualifier/additions/full-stack require backbuilt 2018/2022 time-safe sidecars.",
            "- This harness does not write ModelStore artifacts or train production models.",
            "- `TG1.5` is emitted as home, away, and any-team target rows because side selection needs a side-aware modelling lane.",
        ]
    )
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged", type=Path, default=DEFAULT_MERGED)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    merged = pd.read_csv(args.merged, low_memory=False)
    matrix = pd.read_csv(args.matrix, low_memory=False)
    targets = build_targets(merged)
    long_targets = build_targets_long(targets)
    baselines = build_market_baselines(targets)
    readiness = build_feature_group_outputs(matrix, args.outdir)
    targets.to_csv(args.outdir / "world_cup_historical_targets_wide.csv", index=False)
    long_targets.to_csv(args.outdir / "world_cup_historical_targets_long.csv", index=False)
    baselines.to_csv(args.outdir / "world_cup_historical_market_baselines.csv", index=False)
    readiness.to_csv(args.outdir / "world_cup_2026_ablation_readiness.csv", index=False)
    write_summary(args.outdir, targets, long_targets, baselines, readiness)
    print(f"[ok] targets={len(targets)} long={len(long_targets)} groups={len(readiness)}")
    print(f"[ok] wrote {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
