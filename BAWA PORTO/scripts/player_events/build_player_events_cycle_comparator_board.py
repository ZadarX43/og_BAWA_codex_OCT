from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def _summarize_run(label: str, runner_csv: Path) -> tuple[pd.DataFrame, dict[str, int | float | str]]:
    df = pd.read_csv(runner_csv, low_memory=False)
    sel = pd.to_numeric(df.get("selection_gate_flag", 0), errors="coerce").fillna(0)
    hit = pd.to_numeric(df.get("observed_success_flag", 0), errors="coerce").fillna(0)
    miss_corr = pd.to_numeric(df.get("missed_correct_flag", 0), errors="coerce").fillna(0)
    near = pd.to_numeric(df.get("near_miss_flag", 0), errors="coerce").fillna(0)
    if "history_gate_blocked_flag" in df.columns:
        hgb = pd.to_numeric(df["history_gate_blocked_flag"], errors="coerce").fillna(0)
    else:
        hgb = pd.Series([0] * len(df))

    overall = {
        "cycle_label": label,
        "rows": int(len(df)),
        "selected": int(sel.sum()),
        "selected_hits": int(((sel == 1) & (hit == 1)).sum()),
        "selected_misses": int(((sel == 1) & (hit == 0)).sum()),
        "missed_correct": int(miss_corr.sum()),
        "near_miss": int(near.sum()),
        "history_gate_blocked": int(hgb.sum()),
    }

    # Recompute selected hit/miss cleanly
    market_rows = []
    for market, sub in df.groupby(df["market"].astype(str)):
        if market not in {"shots", "shots_on_target", "tackles", "fouls_committed", "yellow_cards"}:
            continue
        sub_sel = pd.to_numeric(sub.get("selection_gate_flag", 0), errors="coerce").fillna(0)
        sub_hit = pd.to_numeric(sub.get("observed_success_flag", 0), errors="coerce").fillna(0)
        if "history_gate_blocked_flag" in sub.columns:
            sub_hgb = pd.to_numeric(sub["history_gate_blocked_flag"], errors="coerce").fillna(0)
        else:
            sub_hgb = pd.Series([0] * len(sub))
        market_rows.append(
            {
                "cycle_label": label,
                "market": market,
                "rows": int(len(sub)),
                "selected": int(sub_sel.sum()),
                "selected_hits": int(((sub_sel == 1) & (sub_hit == 1)).sum()),
                "selected_misses": int(((sub_sel == 1) & (sub_hit == 0)).sum()),
                "history_gate_blocked": int(sub_hgb.sum()),
            }
        )
    return pd.DataFrame(market_rows), overall


def build(
    baseline_csv: Path,
    cycle5_csv: Path,
    cycle6_csv: Path,
    output_csv: Path,
    output_md: Path,
) -> pd.DataFrame:
    market_frames = []
    overall_rows = []
    for label, path in [
        ("baseline", baseline_csv),
        ("cycle5_recent_strict", cycle5_csv),
        ("cycle6_recent_relaxed_contact", cycle6_csv),
    ]:
        market_df, overall = _summarize_run(label, path)
        market_frames.append(market_df)
        overall_rows.append(overall)

    overall_df = pd.DataFrame(overall_rows)
    market_df = pd.concat(market_frames, ignore_index=True)
    market_df["section"] = "BY_MARKET"
    overall_df["section"] = "OVERALL"
    output = pd.concat([overall_df, market_df], ignore_index=True, sort=False)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)

    lines = [
        "# Player Events Cycle Comparator Board",
        "",
        "- Compares the baseline runner against recent-involvement cycle 5 and relaxed-contact cycle 6.",
        "- Read this as the compact scoreboard for whether the newer research filters are improving precision without killing too many real hits.",
        "",
        "## Overall",
    ]
    for _, row in overall_df.iterrows():
        lines.append(
            f"- {row['cycle_label']} | selected={int(row['selected'])} | selected_hits={int(row['selected_hits'])} | "
            f"selected_misses={int(row['selected_misses'])} | missed_correct={int(row['missed_correct'])} | "
            f"near_miss={int(row['near_miss'])} | history_gate_blocked={int(row['history_gate_blocked'])}"
        )

    lines.append("")
    lines.append("## By Market")
    for market in ["shots", "shots_on_target", "tackles", "fouls_committed", "yellow_cards"]:
        sub = market_df[market_df["market"] == market].copy()
        if sub.empty:
            continue
        lines.append(f"### {market}")
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['cycle_label']} | rows={int(row['rows'])} | selected={int(row['selected'])} | "
                f"selected_hits={int(row['selected_hits'])} | selected_misses={int(row['selected_misses'])} | "
                f"history_gate_blocked={int(row['history_gate_blocked'])}"
            )
        lines.append("")

    lines.extend(
        [
            "## Call",
            "- Cycle 5 shows the effect of a stricter all-market recent involvement gate.",
            "- Cycle 6 keeps the stricter `shots_on_target` stance but relaxes `tackles` and `fouls_committed` so contact markets are not over-suppressed.",
            "- `shots` remains on the board as the broader attacking pool and should be refined separately from the stricter `shots_on_target` premium lane.",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a tiny player-events cycle comparator board for baseline, cycle 5, and cycle 6.")
    parser.add_argument(
        "--baseline-csv",
        default=str(REPO_ROOT / "reports" / "player_events" / "backtests" / "player_events_3y_backtest__2026-05-03__231829" / "player_events_3y_backtest_runner.csv"),
    )
    parser.add_argument(
        "--cycle5-csv",
        default=str(REPO_ROOT / "reports" / "player_events" / "backtests" / "player_events_3y_backtest__2026-05-04__tuned_cycle_5_recent_involvement" / "player_events_3y_backtest_runner.csv"),
    )
    parser.add_argument(
        "--cycle6-csv",
        default=str(REPO_ROOT / "reports" / "player_events" / "backtests" / "player_events_3y_backtest__2026-05-04__tuned_cycle_6_recent_relaxed_contact" / "player_events_3y_backtest_runner.csv"),
    )
    parser.add_argument(
        "--output-csv",
        default=str(REPO_ROOT / "reports" / "player_events" / "quality_audits" / "player_events_cycle_comparator_board.csv"),
    )
    parser.add_argument(
        "--output-md",
        default=str(REPO_ROOT / "reports" / "player_events" / "quality_audits" / "player_events_cycle_comparator_board.md"),
    )
    args = parser.parse_args()
    out = build(
        Path(args.baseline_csv),
        Path(args.cycle5_csv),
        Path(args.cycle6_csv),
        Path(args.output_csv),
        Path(args.output_md),
    )
    print(f"WROTE: {args.output_csv}")
    print(f"rows: {len(out)}")


if __name__ == "__main__":
    main()
