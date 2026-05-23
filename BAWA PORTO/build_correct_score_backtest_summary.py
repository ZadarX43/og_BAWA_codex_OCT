#!/usr/bin/env python3
"""Build a historical Correct Score backtest and product summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from correct_score_product import CSConfig, PREMIUM_LEAGUE_ALLOWLIST, build_fixture_level_cs, read_csvs


THRESHOLDS = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
SCORED_USECOLS = [
    "fixture_key",
    "match_date",
    "league",
    "home_team_name",
    "away_team_name",
    "deploy_tier",
    "market",
    "bookie_pick",
    "selection",
    "cs1",
    "cs1_p",
    "cs2",
    "cs2_p",
    "cs3",
    "cs3_p",
    "cs_top1_scoreline",
    "cs_top2_scoreline",
    "cs_top3_scoreline",
    "cs_entropy",
    "cs_concentration_bucket",
    "cs_banker_flag",
    "cs_banker_bonus",
    "cs_supported_flag",
    "cs_fragility_flag",
    "cs_diffuse_flag",
    "cs_one_sided_flag",
    "cs_draw_family_flag",
    "cs_alignment_count",
    "cs_ftr_alignment",
    "cs_btts_alignment",
    "cs_ou25_alignment",
    "cs_aligns_selection_flag",
    "pick_side_mass_top3",
    "pick_side_margin_top3",
    "home_team_goal_count",
    "away_team_goal_count",
    "actual_ftr",
    "actual_over25",
    "actual_btts_yes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Correct Score backtest summary from scored deploy outputs")
    parser.add_argument("--scored-root", required=True, help="Walk-forward root containing 03_scored deploy files")
    parser.add_argument("--outdir", required=True, help="Directory for Correct Score backtest outputs")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on scored files for smoke testing")
    parser.add_argument("--elite-top1-min", type=float, default=0.11)
    parser.add_argument("--banker-top1-min", type=float, default=0.15)
    parser.add_argument("--rescue-top1-min", type=float, default=0.08)
    return parser.parse_args()


def load_scored_files(scored_root: Path) -> list[Path]:
    return sorted(scored_root.rglob("03_scored/DEPLOY_COMBINED_SCORED_*.csv"))


def build_threshold_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for floor in THRESHOLDS:
        sub = df[df["cs1_p"].ge(floor)].copy()
        rows.append(
            {
                "cs1_p_floor": floor,
                "fixtures": int(sub["fixture_key"].nunique()),
                "exact_hit_rate": sub["exact_hit_flag"].mean(),
                "top3_hit_rate": sub["top3_hit_flag"].mean(),
                "ftr_shape_hit_rate": sub["cs1_ftr_hit_flag"].mean(),
                "btts_shape_hit_rate": sub["cs1_btts_hit_flag"].mean(),
                "ou25_shape_hit_rate": sub["cs1_ou25_hit_flag"].mean(),
                "mean_entropy": sub["cs_entropy"].mean(),
                "mean_support_count": sub["support_count"].mean(),
            }
        )
    return pd.DataFrame(rows)


def write_markdown(
    overall: pd.DataFrame,
    by_class: pd.DataFrame,
    thresholds: pd.DataFrame,
    concentration: pd.DataFrame,
    premium_rules: pd.DataFrame,
    path: Path,
) -> None:
    def md_table(df: pd.DataFrame) -> str:
        frame = df.copy()
        cols = [str(c) for c in frame.columns]
        rows = [cols, ["---"] * len(cols)]
        for _, row in frame.iterrows():
            rendered = []
            for col in frame.columns:
                val = row[col]
                if pd.isna(val):
                    rendered.append("")
                elif isinstance(val, float):
                    rendered.append(f"{val:.3f}")
                else:
                    rendered.append(str(val))
            rows.append(rendered)
        return "\n".join("| " + " | ".join(r) + " |" for r in rows)

    lines = [
        "# Correct Score Product Summary",
        "",
        "## Headline",
        "",
        "- This report measures exact correct-score hit rates and top-3 hit rates from the historical walk-forward scored deploy outputs.",
        "- Where exact historical goal counts are unavailable in the scored deploy export, the report also uses structural hit rates: whether the top CS scoreline got the FTR side, BTTS expression, and OU25 expression right.",
        "- The deploy layer groups fixtures into `CS_BANKER`, `CS_ELITE`, `CS_RESCUE`, and `CS_WATCH`.",
        "- The product view is fixture-level, not market-row-level, so support/veto signals from FTR / BTTS / OU25 are combined into one CS card per fixture.",
        "",
        "## Overall",
        "",
    ]
    if not overall.empty:
        row = overall.iloc[0]
        lines.extend(
            [
                f"- Fixtures: `{int(row['fixtures'])}`",
                f"- Exact top-1 hit rate: `{row['exact_hit_rate']:.3f}`",
                f"- Top-3 hit rate: `{row['top3_hit_rate']:.3f}`",
                f"- CS1 FTR-side hit rate: `{row['ftr_shape_hit_rate']:.3f}`",
                f"- CS1 BTTS-shape hit rate: `{row['btts_shape_hit_rate']:.3f}`",
                f"- CS1 OU25-shape hit rate: `{row['ou25_shape_hit_rate']:.3f}`",
                f"- Mean top-1 probability: `{row['mean_cs1_p']:.3f}`",
                f"- Mean top-3 probability: `{row['mean_top3_prob']:.3f}`",
            ]
        )

    lines.extend(
        [
            "",
            "## By Class",
            "",
            md_table(by_class),
            "",
            "## Thresholded Hit Rates",
            "",
            md_table(thresholds),
            "",
            "## Concentration View",
            "",
            md_table(concentration),
            "",
            "## Premium Rule Stack",
            "",
            md_table(premium_rules),
            "",
            "## Product Use",
            "",
            "- `CS_BANKER`: premium scoreline tag when concentration and support are both strong.",
            "- `CS_ELITE`: public-facing premium shortlist when top-1 probability is high enough and the shape is not diffuse.",
            "- `CS_RESCUE`: fixture where cross-market agreement rescues a weaker raw CS top-1 number.",
            "- `CS_WATCH`: still useful for fixture cards, but not a premium recommendation.",
            "- `PRO_CS_BANKER`: banker + high confidence + low entropy in preferred premium leagues.",
            "- `PRO_CS_BANKER_PLUS`: `PRO_CS_BANKER` with top-3 mass above 0.50.",
            "- `PRO_CS_TRIPLE_ALIGN`: banker with strong FTR + BTTS + OU25 alignment.",
            "- `PRO_CS_ELITE`: high-confidence concentrated premium lane, including strong `CS_WATCH` rows.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    scored_root = Path(args.scored_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = load_scored_files(scored_root)
    if args.max_files is not None:
        files = files[: max(int(args.max_files), 0)]
    if not files:
        raise SystemExit(f"No scored deploy files found under {scored_root}")

    cfg = CSConfig(
        elite_top1_min=args.elite_top1_min,
        banker_top1_min=args.banker_top1_min,
        rescue_top1_min=args.rescue_top1_min,
    )
    scored_df = read_csvs(files, usecols=SCORED_USECOLS)
    fixture_cards = build_fixture_level_cs(scored_df, cfg)
    graded_mask = (
        fixture_cards["actual_score"].astype(str).ne("")
        | fixture_cards["actual_ftr"].astype(str).ne("")
        | fixture_cards["actual_over25"].notna()
        | fixture_cards["actual_btts_yes"].notna()
    )
    fixture_cards = fixture_cards[graded_mask].copy()
    if fixture_cards.empty:
        raise SystemExit("No graded Correct Score fixture rows were produced.")

    overall = pd.DataFrame(
        [
            {
                "fixtures": int(fixture_cards["fixture_key"].nunique()),
                "exact_hit_rate": fixture_cards["exact_hit_flag"].mean(),
                "top3_hit_rate": fixture_cards["top3_hit_flag"].mean(),
                "ftr_shape_hit_rate": fixture_cards["cs1_ftr_hit_flag"].mean(),
                "btts_shape_hit_rate": fixture_cards["cs1_btts_hit_flag"].mean(),
                "ou25_shape_hit_rate": fixture_cards["cs1_ou25_hit_flag"].mean(),
                "mean_cs1_p": fixture_cards["cs1_p"].mean(),
                "mean_top3_prob": fixture_cards["cs_top3_total_prob"].mean(),
            }
        ]
    )
    by_class = (
        fixture_cards.groupby("cs_class", observed=False)
        .agg(
            fixtures=("fixture_key", "nunique"),
            exact_hit_rate=("exact_hit_flag", "mean"),
            top3_hit_rate=("top3_hit_flag", "mean"),
            ftr_shape_hit_rate=("cs1_ftr_hit_flag", "mean"),
            btts_shape_hit_rate=("cs1_btts_hit_flag", "mean"),
            ou25_shape_hit_rate=("cs1_ou25_hit_flag", "mean"),
            mean_cs1_p=("cs1_p", "mean"),
            mean_top3_prob=("cs_top3_total_prob", "mean"),
            mean_entropy=("cs_entropy", "mean"),
            mean_support_count=("support_count", "mean"),
        )
        .reset_index()
    )
    thresholds = build_threshold_table(fixture_cards)
    concentration = (
        fixture_cards.groupby("cs_concentration_bucket", observed=False)
        .agg(
            fixtures=("fixture_key", "nunique"),
            exact_hit_rate=("exact_hit_flag", "mean"),
            top3_hit_rate=("top3_hit_flag", "mean"),
            ftr_shape_hit_rate=("cs1_ftr_hit_flag", "mean"),
            btts_shape_hit_rate=("cs1_btts_hit_flag", "mean"),
            ou25_shape_hit_rate=("cs1_ou25_hit_flag", "mean"),
            mean_cs1_p=("cs1_p", "mean"),
            mean_entropy=("cs_entropy", "mean"),
        )
        .reset_index()
        .sort_values("fixtures", ascending=False)
    )
    premium_shortlist = fixture_cards[fixture_cards["cs_class"].isin(["CS_BANKER", "CS_ELITE", "CS_RESCUE"])].copy()
    pro_shortlist = fixture_cards[fixture_cards["website_segment"].eq("PREMIUM")].copy()
    league_aware_shortlist = pro_shortlist[pro_shortlist["premium_league_flag"].eq(1)].copy()
    premium_rules = pd.DataFrame(
        [
            {
                "premium_rule": "PRO_CS_TRIPLE_ALIGN",
                "fixtures": int(fixture_cards[fixture_cards["pro_cs_triple_align_flag"].eq(1)]["fixture_key"].nunique()),
                "exact_hit_rate": fixture_cards[fixture_cards["pro_cs_triple_align_flag"].eq(1)]["exact_hit_flag"].mean(),
                "top3_hit_rate": fixture_cards[fixture_cards["pro_cs_triple_align_flag"].eq(1)]["top3_hit_flag"].mean(),
            },
            {
                "premium_rule": "PRO_CS_BANKER_PLUS",
                "fixtures": int(fixture_cards[fixture_cards["pro_cs_banker_plus_flag"].eq(1)]["fixture_key"].nunique()),
                "exact_hit_rate": fixture_cards[fixture_cards["pro_cs_banker_plus_flag"].eq(1)]["exact_hit_flag"].mean(),
                "top3_hit_rate": fixture_cards[fixture_cards["pro_cs_banker_plus_flag"].eq(1)]["top3_hit_flag"].mean(),
            },
            {
                "premium_rule": "PRO_CS_BANKER",
                "fixtures": int(fixture_cards[fixture_cards["pro_cs_banker_flag"].eq(1)]["fixture_key"].nunique()),
                "exact_hit_rate": fixture_cards[fixture_cards["pro_cs_banker_flag"].eq(1)]["exact_hit_flag"].mean(),
                "top3_hit_rate": fixture_cards[fixture_cards["pro_cs_banker_flag"].eq(1)]["top3_hit_flag"].mean(),
            },
            {
                "premium_rule": "PRO_CS_ELITE",
                "fixtures": int(fixture_cards[fixture_cards["pro_cs_elite_flag"].eq(1)]["fixture_key"].nunique()),
                "exact_hit_rate": fixture_cards[fixture_cards["pro_cs_elite_flag"].eq(1)]["exact_hit_flag"].mean(),
                "top3_hit_rate": fixture_cards[fixture_cards["pro_cs_elite_flag"].eq(1)]["top3_hit_flag"].mean(),
            },
        ]
    )
    premium_rules["league_allowlist"] = ", ".join(sorted(PREMIUM_LEAGUE_ALLOWLIST))

    overall_path = outdir / "CS_BACKTEST__OVERALL.csv"
    by_class_path = outdir / "CS_BACKTEST__BY_CLASS.csv"
    thresholds_path = outdir / "CS_BACKTEST__THRESHOLD_HIT_RATES.csv"
    concentration_path = outdir / "CS_BACKTEST__CONCENTRATION.csv"
    fixture_cards_path = outdir / "CS_BACKTEST__FIXTURE_CARDS.csv"
    premium_path = outdir / "CS_BACKTEST__PREMIUM_SHORTLIST.csv"
    pro_shortlist_path = outdir / "CS_BACKTEST__PRO_SHORTLIST.csv"
    league_aware_path = outdir / "CS_BACKTEST__LEAGUE_AWARE_PREMIUM_SHORTLIST.csv"
    premium_rules_path = outdir / "CS_BACKTEST__PREMIUM_RULES.csv"
    md_path = outdir / "CS_BACKTEST__SUMMARY.md"

    overall.to_csv(overall_path, index=False)
    by_class.to_csv(by_class_path, index=False)
    thresholds.to_csv(thresholds_path, index=False)
    concentration.to_csv(concentration_path, index=False)
    fixture_cards.to_csv(fixture_cards_path, index=False)
    premium_shortlist.to_csv(premium_path, index=False)
    pro_shortlist.to_csv(pro_shortlist_path, index=False)
    league_aware_shortlist.to_csv(league_aware_path, index=False)
    premium_rules.to_csv(premium_rules_path, index=False)
    write_markdown(overall, by_class, thresholds, concentration, premium_rules, md_path)

    print("WROTE:")
    print(overall_path)
    print(by_class_path)
    print(thresholds_path)
    print(concentration_path)
    print(fixture_cards_path)
    print(premium_path)
    print(pro_shortlist_path)
    print(league_aware_path)
    print(premium_rules_path)
    print(md_path)
    print("\nCS BACKTEST OVERALL\n")
    print(overall.to_string(index=False))
    print("\nCS BACKTEST BY CLASS\n")
    print(by_class.to_string(index=False))
    print("\nCS THRESHOLD HIT RATES\n")
    print(thresholds.to_string(index=False))
    print("\nCS PREMIUM RULES\n")
    print(premium_rules.to_string(index=False))


if __name__ == "__main__":
    main()
