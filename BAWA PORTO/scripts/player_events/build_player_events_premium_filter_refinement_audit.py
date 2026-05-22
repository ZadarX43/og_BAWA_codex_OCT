from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKTEST_ROOT = REPO_ROOT / "reports" / "player_events" / "backtests"
TARGET_MARKETS = ("shots_on_target", "fouls_committed")


def _latest_backtest_dir() -> Path:
    dirs = sorted(BACKTEST_ROOT.glob("player_events_3y_backtest__*/"), reverse=True)
    if not dirs:
        raise FileNotFoundError("No player-events 3Y backtest directories found.")
    return dirs[0]


def classify_tier(row: pd.Series) -> str:
    market = str(row.get("market", ""))
    source = str(row.get("lookback_source", ""))
    expected_hit = float(pd.to_numeric(row.get("expected_hit_rate_3y"), errors="coerce") or 0.0)
    lookback_rows = int(pd.to_numeric(row.get("lookback_rows"), errors="coerce") or 0)
    score_delta = float(pd.to_numeric(row.get("score_delta_vs_3y"), errors="coerce") or 0.0)
    selected = int(pd.to_numeric(row.get("selection_gate_flag"), errors="coerce") or 0)

    high_quality_source = source in {"EXACT", "ROLE_MARKET", "FAMILY_ROLE"}
    if market == "shots_on_target":
        if high_quality_source and lookback_rows >= 3 and expected_hit >= 0.60 and score_delta >= 0 and selected == 1:
            return "ELITE"
        if source != "GLOBAL" and lookback_rows >= 3 and expected_hit >= 0.55:
            return "STRONG"
        return "WATCH"
    if market == "fouls_committed":
        if high_quality_source and lookback_rows >= 3 and expected_hit >= 0.60 and score_delta >= 0 and selected == 1:
            return "ELITE"
        if source != "GLOBAL" and lookback_rows >= 3 and expected_hit >= 0.55:
            return "STRONG"
        return "WATCH"
    return "WATCH"


def _reason_tag(row: pd.Series) -> str:
    observed_hit = int(pd.to_numeric(row.get("observed_success_flag"), errors="coerce") or 0)
    if observed_hit == 1:
        return "ACTUAL_HIT"
    player_q = float(pd.to_numeric(row.get("player_quality_score_l5"), errors="coerce") or 0.0)
    pressure = float(pd.to_numeric(row.get("formation_pressure_score"), errors="coerce") or 0.0)
    source = str(row.get("lookback_source", ""))
    role = str(row.get("tactical_role", ""))
    xi_edge = float(pd.to_numeric(row.get("starting_xi_quality_edge"), errors="coerce") or 0.0)

    if source == "MARKET_ONLY":
        return "FALLBACK_SOURCE_WEAKNESS"
    if player_q < 60:
        return "PLAYER_QUALITY_WEAK"
    if pressure < 0.20 and "shots_on_target" == str(row.get("market")):
        return "PRESSURE_TOO_WEAK_FOR_ATTACK"
    if pressure < 0.20 and "fouls_committed" == str(row.get("market")):
        return "PRESSURE_TOO_WEAK_FOR_CONTACT"
    if xi_edge < 0:
        return "XI_CONTEXT_NEGATIVE"
    if "Wide" in role and pressure < 0.30:
        return "ROLE_PRESSURE_MISALIGNMENT"
    return "PREMIUM_FILTER_STILL_TOO_LOOSE"


def build(backtest_dir: Path, output_csv: Path, output_md: Path) -> pd.DataFrame:
    runner_csv = backtest_dir / "player_events_3y_backtest_runner.csv"
    df = pd.read_csv(runner_csv, low_memory=False)
    if df.empty:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        output_md.write_text("# Player Events Premium Filter Refinement Audit\n\nNo runner rows matched.\n")
        return df

    numeric_cols = [
        "expected_hit_rate_3y",
        "lookback_rows",
        "score_delta_vs_3y",
        "selection_gate_flag",
        "observed_success_flag",
        "fixture_quality_score",
        "formation_pressure_score",
        "player_quality_score_l5",
        "starting_xi_quality_edge",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    scoped = df[df["market"].astype(str).isin(TARGET_MARKETS)].copy()
    scoped["premium_tier"] = scoped.apply(classify_tier, axis=1)
    scoped["refinement_reason"] = scoped.apply(_reason_tag, axis=1)

    summary = (
        scoped.groupby(["market", "premium_tier", "refinement_reason"], dropna=False)
        .agg(
            rows=("fixture_key", "size"),
            fixtures=("fixture_key", pd.Series.nunique),
            observed_hit_rate=("observed_success_flag", "mean"),
            avg_expected_hit=("expected_hit_rate_3y", "mean"),
            avg_pressure=("formation_pressure_score", "mean"),
            avg_player_quality=("player_quality_score_l5", "mean"),
        )
        .reset_index()
        .sort_values(["market", "premium_tier", "rows"], ascending=[True, True, False])
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)

    lines = [
        "# Player Events Premium Filter Refinement Audit",
        "",
        "- Focus: explain why premium-looking `shots_on_target` and `fouls_committed` rows are still missing the premium hit-rate standard.",
        "- Uses the latest stable 3Y backtest runner and a research-only premium tier lens.",
        "",
    ]

    for market in TARGET_MARKETS:
        market_rows = scoped[scoped["market"].astype(str) == market].copy()
        lines.append(f"## {market}")
        tier_summary = (
            market_rows.groupby("premium_tier", dropna=False)
            .agg(
                rows=("fixture_key", "size"),
                observed_hit_rate=("observed_success_flag", "mean"),
                avg_expected_hit=("expected_hit_rate_3y", "mean"),
            )
            .reset_index()
            .sort_values(["premium_tier"])
        )
        for _, row in tier_summary.iterrows():
            lines.append(
                f"- tier={row['premium_tier']} | rows={int(row['rows'])} | observed_hit_rate={float(row['observed_hit_rate']):.3f} | avg_expected_hit={float(row['avg_expected_hit']):.3f}"
            )

        premium_failures = market_rows[
            market_rows["premium_tier"].isin(["ELITE", "STRONG"])
            & (pd.to_numeric(market_rows["observed_success_flag"], errors="coerce").fillna(0) == 0)
        ].copy()
        lines.append("")
        lines.append("### Why Premium Rows Still Miss")
        if premium_failures.empty:
            lines.append("- No premium-tier misses found in this market.")
        else:
            for _, row in (
                premium_failures.groupby("refinement_reason", dropna=False)
                .agg(
                    rows=("fixture_key", "size"),
                    avg_pressure=("formation_pressure_score", "mean"),
                    avg_player_quality=("player_quality_score_l5", "mean"),
                )
                .reset_index()
                .sort_values(["rows"], ascending=[False])
                .iterrows()
            ):
                lines.append(
                    f"- {row['refinement_reason']} | rows={int(row['rows'])} | avg_pressure={float(row['avg_pressure']):.3f} | avg_player_quality={float(row['avg_player_quality']):.2f}"
                )

            lines.append("")
            lines.append("### Missed Premium Examples")
            for _, row in premium_failures.head(8).iterrows():
                lines.append(
                    f"- {row['fixture_key']} | {row['player_name']} | tier={row['premium_tier']} | source={row['lookback_source']} | "
                    f"expected_hit={float(row['expected_hit_rate_3y']):.3f} | pressure={float(row['formation_pressure_score']):.3f} | "
                    f"player_q={float(row['player_quality_score_l5']):.1f} | reason={row['refinement_reason']}"
                )
        lines.append("")

    lines.extend(
        [
            "## Takeaway",
            "- If premium rows are still missing, the next improvement is usually filter quality: better role purity, stronger pressure alignment, and less dependence on weak fallback evidence.",
            "- This audit is meant to guide filter refinement, not to justify blanket threshold relaxation.",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a premium-filter refinement audit for shots_on_target and fouls_committed.")
    parser.add_argument("--backtest-dir", default="", help="Optional explicit player-events backtest directory.")
    parser.add_argument(
        "--output-csv",
        default=str(REPO_ROOT / "reports" / "player_events" / "quality_audits" / "player_events_premium_filter_refinement_audit.csv"),
    )
    parser.add_argument(
        "--output-md",
        default=str(REPO_ROOT / "reports" / "player_events" / "quality_audits" / "player_events_premium_filter_refinement_audit.md"),
    )
    args = parser.parse_args()
    backtest_dir = Path(args.backtest_dir) if args.backtest_dir else _latest_backtest_dir()
    out = build(backtest_dir, Path(args.output_csv), Path(args.output_md))
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")


if __name__ == "__main__":
    main()
