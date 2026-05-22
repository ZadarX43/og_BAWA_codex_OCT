from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TARGET_FORMATIONS = {"4-2-3-1 vs 4-4-2", "4-4-2 vs 4-2-3-1"}
TARGET_MARKETS = ["yellow_cards", "fouls_committed", "tackles", "shots", "shots_on_target"]


def build_pack(input_csv: str, formation_audit_csv: str, output_csv: str, output_md: str) -> None:
    inputs = pd.read_csv(input_csv)
    audit = pd.read_csv(formation_audit_csv)

    inputs = inputs[inputs["formation_matchup_label"].isin(TARGET_FORMATIONS)].copy()
    audit = audit[audit["formation_bucket"].isin(TARGET_FORMATIONS)].copy()
    audit = audit[audit["market"].isin(TARGET_MARKETS)].copy()
    if "fixture_quality_score" not in inputs.columns:
        inputs["fixture_quality_score"] = (
            0.30 * pd.to_numeric(inputs.get("og_goal_environment_score", 0.0), errors="coerce").fillna(0.0)
            + 0.25 * pd.to_numeric(inputs.get("fixture_foul_density_score", 0.0), errors="coerce").fillna(0.0)
            + 0.20 * pd.to_numeric(inputs.get("fixture_attack_pressure_score", 0.0), errors="coerce").fillna(0.0)
            + 0.15 * pd.to_numeric(inputs.get("fixture_tackle_density_score", 0.0), errors="coerce").fillna(0.0)
            + 0.10 * pd.to_numeric(inputs.get("formation_pressure_score", 0.0), errors="coerce").fillna(0.0)
        ).clip(lower=0.0, upper=1.0)

    fixture_rows = (
        inputs[[
            "fixture_key",
            "competition",
            "home_team_name",
            "away_team_name",
            "formation_matchup_label",
            "fixture_style_label",
            "fixture_attacking_style_label",
            "fixture_quality_score",
            "formation_pressure_score",
        ]]
        .drop_duplicates(subset=["fixture_key"]) 
        .sort_values(["fixture_quality_score", "formation_pressure_score"], ascending=[False, False])
    )

    market_summary = (
        audit.groupby("market", dropna=False)
        .agg(
            combo_rows=("rows", "sum"),
            combo_fixtures=("fixtures", "sum"),
            weighted_hit_rate=("hit_rate", lambda s: round((s * audit.loc[s.index, "rows"]).sum() / max(audit.loc[s.index, "rows"].sum(), 1), 4)),
            avg_market_score=("avg_market_score", "mean"),
            avg_formation_pressure=("avg_formation_pressure", "mean"),
        )
        .reset_index()
        .sort_values(["weighted_hit_rate", "combo_rows"], ascending=[False, False])
    )

    top_combos = (
        audit.sort_values(["market", "hit_rate", "rows"], ascending=[True, False, False])
        .groupby("market", dropna=False)
        .head(5)
        .reset_index(drop=True)
    )

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    market_summary.to_csv(output_csv, index=False)
    top_combo_csv = out_path.with_name(out_path.stem + "__top_combos.csv")
    top_combos.to_csv(top_combo_csv, index=False)
    fixture_csv = out_path.with_name(out_path.stem + "__fixture_examples.csv")
    fixture_rows.to_csv(fixture_csv, index=False)

    lines = [
        "# 4-2-3-1 vs 4-4-2 Specialist Audit Pack",
        "",
        f"- formation buckets: {', '.join(sorted(TARGET_FORMATIONS))}",
        f"- fixtures in merged input: {fixture_rows['fixture_key'].nunique()}",
        "",
        "## Market Summary",
        "",
    ]
    if market_summary.empty:
        lines.append("No rows matched the specialist filter.")
    else:
        for _, row in market_summary.iterrows():
            lines.extend([
                f"### {row['market']}",
                f"- weighted hit rate: {row['weighted_hit_rate']:.4f}",
                f"- combo rows: {int(row['combo_rows'])}",
                f"- combo fixtures: {int(row['combo_fixtures'])}",
                f"- avg market score: {float(row['avg_market_score']):.3f}",
                f"- avg formation pressure: {float(row['avg_formation_pressure']):.3f}",
                "",
            ])

        lines.append("## Top Combos")
        lines.append("")
        for market in TARGET_MARKETS:
            sub = top_combos[top_combos['market'].eq(market)]
            if sub.empty:
                continue
            lines.append(f"### {market}")
            for _, row in sub.iterrows():
                lines.append(
                    f"- {row['combo_label']}: hit {float(row['hit_rate']):.4f} | rows {int(row['rows'])} | fixtures {int(row['fixtures'])} | formation pressure {float(row['avg_formation_pressure']):.3f}"
                )
            lines.append("")

        lines.append("## Fixture Examples")
        lines.append("")
        for _, row in fixture_rows.head(20).iterrows():
            lines.append(
                f"- {row['fixture_key']}: {row['competition']} | {row['formation_matchup_label']} | contact {row['fixture_style_label']} | attack {row['fixture_attacking_style_label']} | fixture quality {float(row['fixture_quality_score']):.3f} | formation pressure {float(row['formation_pressure_score']):.3f}"
            )

    Path(output_md).write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 4-2-3-1 vs 4-4-2 specialist audit pack.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--formation-audit-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_pack(args.input, args.formation_audit_csv, args.output_csv, args.output_md)
