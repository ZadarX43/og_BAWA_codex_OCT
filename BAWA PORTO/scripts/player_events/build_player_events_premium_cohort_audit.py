from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKTEST_ROOT = REPO_ROOT / "reports" / "player_events" / "backtests"
CORE_MARKETS = {"shots_on_target", "tackles", "fouls_committed"}


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
    if market == "yellow_cards":
        if high_quality_source and lookback_rows >= 3 and expected_hit >= 0.60 and selected == 1:
            return "BOOKINGS_ELITE"
        if source != "GLOBAL" and lookback_rows >= 3 and expected_hit >= 0.50:
            return "BOOKINGS_STRONG"
        return "BOOKINGS_WATCH"

    if high_quality_source and lookback_rows >= 3 and expected_hit >= 0.60 and score_delta >= 0 and selected == 1:
        return "ELITE"
    if source != "GLOBAL" and lookback_rows >= 3 and expected_hit >= 0.55:
        return "STRONG"
    return "WATCH"


def build(backtest_dir: Path, output_csv: Path, output_md: Path) -> pd.DataFrame:
    runner_csv = backtest_dir / "player_events_3y_backtest_runner.csv"
    df = pd.read_csv(runner_csv, low_memory=False)
    if df.empty:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        output_md.write_text("# Player Events Premium Cohort Audit\n\nNo runner rows matched.\n")
        return df

    df["expected_hit_rate_3y"] = pd.to_numeric(df["expected_hit_rate_3y"], errors="coerce").fillna(0.0)
    df["lookback_rows"] = pd.to_numeric(df["lookback_rows"], errors="coerce").fillna(0).astype(int)
    df["score_delta_vs_3y"] = pd.to_numeric(df["score_delta_vs_3y"], errors="coerce").fillna(0.0)
    df["selection_gate_flag"] = pd.to_numeric(df["selection_gate_flag"], errors="coerce").fillna(0).astype(int)
    df["observed_success_flag"] = pd.to_numeric(df["observed_success_flag"], errors="coerce").fillna(0).astype(int)

    df["premium_tier"] = df.apply(classify_tier, axis=1)
    df["premium_target_band"] = df["market"].astype(str).map(
        {
            "shots_on_target": "70_80",
            "tackles": "70_80",
            "fouls_committed": "70_80",
            "yellow_cards": "SEPARATE_NOISIER",
        }
    ).fillna("OTHER")

    summary = (
        df[df["market"].isin(CORE_MARKETS | {"yellow_cards"})]
        .groupby(["market", "premium_tier"], dropna=False)
        .agg(
            rows=("fixture_key", "size"),
            fixtures=("fixture_key", pd.Series.nunique),
            selected=("selection_gate_flag", "sum"),
            observed_hits=("observed_success_flag", "sum"),
            observed_hit_rate=("observed_success_flag", "mean"),
            avg_expected_hit=("expected_hit_rate_3y", "mean"),
            avg_score_delta=("score_delta_vs_3y", "mean"),
        )
        .reset_index()
    )
    summary["toward_target_call"] = summary.apply(
        lambda row: (
            "HITTING_PREMIUM_STANDARD"
            if row["market"] != "yellow_cards" and float(row["observed_hit_rate"]) >= 0.70
            else "PROMISING_BUT_BELOW_PREMIUM_STANDARD"
            if row["market"] != "yellow_cards" and float(row["observed_hit_rate"]) >= 0.55
            else "BOOKINGS_READ_SEPARATELY"
            if row["market"] == "yellow_cards"
            else "BELOW_PREMIUM_STANDARD"
        ),
        axis=1,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)

    lines = [
        "# Player Events Premium Cohort Audit",
        "",
        "- Research-only audit of whether the best-supported subsets are behaving like true premium cohorts.",
        "- Core target markets: `shots_on_target`, `tackles`, `fouls_committed`.",
        "- `yellow_cards` is shown separately because it is structurally noisier.",
        "",
        "## Core Markets",
    ]
    core = summary[summary["market"].isin(CORE_MARKETS)].sort_values(
        ["market", "premium_tier"],
        ascending=[True, True],
    )
    for _, row in core.iterrows():
        lines.append(
            f"- {row['market']} | tier={row['premium_tier']} | rows={int(row['rows'])} | fixtures={int(row['fixtures'])} | "
            f"selected={int(row['selected'])} | observed_hit_rate={float(row['observed_hit_rate']):.3f} | "
            f"avg_expected_hit={float(row['avg_expected_hit']):.3f} | call={row['toward_target_call']}"
        )

    lines.extend(["", "## Yellow Cards"])
    yc = summary[summary["market"] == "yellow_cards"].sort_values(["premium_tier"])
    for _, row in yc.iterrows():
        lines.append(
            f"- yellow_cards | tier={row['premium_tier']} | rows={int(row['rows'])} | fixtures={int(row['fixtures'])} | "
            f"selected={int(row['selected'])} | observed_hit_rate={float(row['observed_hit_rate']):.3f} | "
            f"avg_expected_hit={float(row['avg_expected_hit']):.3f}"
        )

    elite_rows = df[(df["market"].isin(CORE_MARKETS)) & (df["premium_tier"] == "ELITE")].copy()
    lines.extend(["", "## Elite Examples"])
    if elite_rows.empty:
        lines.append("- No core-market rows currently clear the strict `ELITE` definition.")
    else:
        elite_rows = elite_rows.sort_values(
            ["market", "observed_success_flag", "expected_hit_rate_3y", "score_delta_vs_3y"],
            ascending=[True, False, False, False],
        )
        for _, row in elite_rows.head(12).iterrows():
            lines.append(
                f"- {row['fixture_key']} | {row['player_name']} | {row['market']} | source={row['lookback_source']} | "
                f"expected_hit={float(row['expected_hit_rate_3y']):.3f} | observed_hit={int(row['observed_success_flag'])}"
            )

    lines.extend(
        [
            "",
            "## Read",
            "- Use this to judge whether the top supported subsets are actually approaching the premium hit-rate standard you want, rather than judging the whole research estate by one number.",
            "- If `ELITE` and `STRONG` tiers still sit well below the target band, the next step is usually better cohort filtering, not just more threshold relaxation.",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a premium cohort audit for the player-events 3Y backtest.")
    parser.add_argument("--backtest-dir", default="", help="Optional explicit player-events backtest directory.")
    parser.add_argument(
        "--output-csv",
        default=str(REPO_ROOT / "reports" / "player_events" / "quality_audits" / "player_events_premium_cohort_audit.csv"),
    )
    parser.add_argument(
        "--output-md",
        default=str(REPO_ROOT / "reports" / "player_events" / "quality_audits" / "player_events_premium_cohort_audit.md"),
    )
    args = parser.parse_args()
    backtest_dir = Path(args.backtest_dir) if args.backtest_dir else _latest_backtest_dir()
    out = build(backtest_dir, Path(args.output_csv), Path(args.output_md))
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")


if __name__ == "__main__":
    main()
