from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TARGET_MARKETS = ["tackles", "fouls_committed", "shots", "shots_on_target"]


def _load_csv(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path, low_memory=False)


def _markdown_table(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return [header, sep] + rows


def _combine_market_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    valid = [df for df in frames if not df.empty]
    if not valid:
        return pd.DataFrame()
    combined = pd.concat(valid, ignore_index=True)
    combined["weighted_hits"] = pd.to_numeric(combined["hit_rate"], errors="coerce").fillna(0.0) * pd.to_numeric(
        combined["rows"], errors="coerce"
    ).fillna(0.0)
    combined["weighted_score"] = pd.to_numeric(combined["avg_score"], errors="coerce").fillna(0.0) * pd.to_numeric(
        combined["rows"], errors="coerce"
    ).fillna(0.0)
    combined["weighted_quality"] = pd.to_numeric(combined["avg_fixture_quality"], errors="coerce").fillna(0.0) * pd.to_numeric(
        combined["rows"], errors="coerce"
    ).fillna(0.0)
    grouped = (
        combined.groupby(["source_family", "market"], as_index=False)
        .agg(
            rows=("rows", "sum"),
            weighted_hits=("weighted_hits", "sum"),
            weighted_score=("weighted_score", "sum"),
            weighted_quality=("weighted_quality", "sum"),
        )
    )
    grouped["hit_rate"] = grouped["weighted_hits"] / grouped["rows"].replace(0, pd.NA)
    grouped["avg_score"] = grouped["weighted_score"] / grouped["rows"].replace(0, pd.NA)
    grouped["avg_fixture_quality"] = grouped["weighted_quality"] / grouped["rows"].replace(0, pd.NA)
    return grouped.drop(columns=["weighted_hits", "weighted_score", "weighted_quality"]).sort_values(
        ["market", "hit_rate", "rows"], ascending=[True, False, False]
    ).reset_index(drop=True)


def _portability_label(europe_hit: float, batch4_hit: float, batch4_rows: int) -> str:
    if batch4_rows < 20:
        return "NEEDS_MORE_SAMPLE"
    gap = europe_hit - batch4_hit
    if batch4_hit >= 0.65 and gap <= 0.08:
        return "PORTABLE"
    if batch4_hit >= 0.55 and gap <= 0.18:
        return "CONDITIONAL_PORTABLE"
    return "EUROPE_ONLY_SO_FAR"


def build_market_leaderboard(
    batch_names: list[str],
    family_market_csvs: list[str],
    output_md: str,
    output_csv: str | None = None,
    portability_md: str | None = None,
    portability_csv: str | None = None,
) -> None:
    market_map = {batch: _load_csv(path) for batch, path in zip(batch_names, family_market_csvs)}
    combined_market = _combine_market_frames(list(market_map.values()))
    all_export_frames: list[pd.DataFrame] = []

    lines = ["# Cross-Batch Market Family Leaderboard", ""]
    for batch in batch_names:
        batch_df = market_map[batch]
        if not batch_df.empty:
            export_df = batch_df.copy()
            export_df.insert(0, "scope", batch)
            all_export_frames.append(export_df)
        lines.append(f"## {batch}")
        if batch_df.empty:
            lines.append("No rows matched.")
            lines.append("")
            continue
        for market in TARGET_MARKETS:
            market_df = batch_df[batch_df["market"].eq(market)].sort_values(["hit_rate", "rows"], ascending=[False, False])
            lines.append(f"### {market}")
            if market_df.empty:
                lines.append("No rows matched.")
                lines.append("")
                continue
            best = market_df.iloc[0]
            lines.append(
                f"- best family: {best['source_family']} | hit_rate={float(best['hit_rate']):.3f} | rows={int(best['rows'])}"
            )
            market_print = market_df.copy()
            for col in ["hit_rate", "avg_score", "avg_fixture_quality"]:
                market_print[col] = pd.to_numeric(market_print[col], errors="coerce").round(3)
            lines.extend(_markdown_table(market_print))
            lines.append("")

    if not combined_market.empty:
        export_df = combined_market.copy()
        export_df.insert(0, "scope", "combined_overall")
        all_export_frames.append(export_df)
    lines.append("## combined_overall")
    for market in TARGET_MARKETS:
        market_df = combined_market[combined_market["market"].eq(market)].sort_values(["hit_rate", "rows"], ascending=[False, False])
        lines.append(f"### {market}")
        if market_df.empty:
            lines.append("No rows matched.")
            lines.append("")
            continue
        best = market_df.iloc[0]
        lines.append(
            f"- best family: {best['source_family']} | hit_rate={float(best['hit_rate']):.3f} | rows={int(best['rows'])}"
        )
        market_print = market_df.copy()
        for col in ["hit_rate", "avg_score", "avg_fixture_quality"]:
            market_print[col] = pd.to_numeric(market_print[col], errors="coerce").round(3)
        lines.extend(_markdown_table(market_print))
        lines.append("")

    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md).write_text("\n".join(lines) + "\n")
    if output_csv:
        pd.concat(all_export_frames, ignore_index=True).to_csv(output_csv, index=False)

    if portability_md or portability_csv:
        europe_frames = [market_map[b] for b in batch_names if b != "greenlist_batch4" and not market_map[b].empty]
        europe_combined = _combine_market_frames(europe_frames)
        batch4_df = market_map.get("greenlist_batch4", pd.DataFrame())
        portability_rows = []
        for market in TARGET_MARKETS:
            europe_market = europe_combined[europe_combined["market"].eq(market)]
            batch4_market = batch4_df[batch4_df["market"].eq(market)]
            for family in sorted(set(europe_market["source_family"]).union(set(batch4_market["source_family"]))):
                europe_row = europe_market[europe_market["source_family"].eq(family)]
                batch4_row = batch4_market[batch4_market["source_family"].eq(family)]
                europe_hit = float(europe_row["hit_rate"].iloc[0]) if not europe_row.empty else 0.0
                europe_rows = int(europe_row["rows"].iloc[0]) if not europe_row.empty else 0
                batch4_hit = float(batch4_row["hit_rate"].iloc[0]) if not batch4_row.empty else 0.0
                batch4_rows = int(batch4_row["rows"].iloc[0]) if not batch4_row.empty else 0
                portability_rows.append(
                    {
                        "market": market,
                        "source_family": family,
                        "europe_hit_rate": europe_hit,
                        "europe_rows": europe_rows,
                        "batch4_hit_rate": batch4_hit,
                        "batch4_rows": batch4_rows,
                        "hit_rate_gap": europe_hit - batch4_hit,
                        "portability_label": _portability_label(europe_hit, batch4_hit, batch4_rows),
                    }
                )
        portability_df = pd.DataFrame(portability_rows).sort_values(
            ["market", "portability_label", "batch4_hit_rate"], ascending=[True, True, False]
        ).reset_index(drop=True)

        portability_lines = ["# Portable vs Europe-Only Family Split", ""]
        portability_lines.append(
            "Provisional labels: `PORTABLE` means batch 4 stayed close to the Europe-core hit rate on a decent sample; `CONDITIONAL_PORTABLE` means it held up somewhat but with a bigger drop or thinner sample; `EUROPE_ONLY_SO_FAR` means the non-Europe expansion lagged too much; `NEEDS_MORE_SAMPLE` means we should not over-read it yet."
        )
        portability_lines.append("")
        for market in TARGET_MARKETS:
            portability_lines.append(f"## {market}")
            market_df = portability_df[portability_df["market"].eq(market)].copy()
            if market_df.empty:
                portability_lines.append("No rows matched.")
                portability_lines.append("")
                continue
            for col in ["europe_hit_rate", "batch4_hit_rate", "hit_rate_gap"]:
                market_df[col] = pd.to_numeric(market_df[col], errors="coerce").round(3)
            portability_lines.extend(_markdown_table(market_df))
            portability_lines.append("")

        if portability_md:
            Path(portability_md).write_text("\n".join(portability_lines) + "\n")
        if portability_csv:
            portability_df.to_csv(portability_csv, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a market-specific cross-batch family leaderboard and portability split.")
    parser.add_argument("--batch-names", required=True)
    parser.add_argument("--family-market-csvs", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-csv")
    parser.add_argument("--portability-md")
    parser.add_argument("--portability-csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batch_names = [x.strip() for x in args.batch_names.split(",") if x.strip()]
    family_market_csvs = [x.strip() for x in args.family_market_csvs.split(",") if x.strip()]
    if len(batch_names) != len(family_market_csvs):
        raise SystemExit("batch-names and family-market-csvs must align")
    build_market_leaderboard(
        batch_names=batch_names,
        family_market_csvs=family_market_csvs,
        output_md=args.output_md,
        output_csv=args.output_csv,
        portability_md=args.portability_md,
        portability_csv=args.portability_csv,
    )
    print(f"WROTE: {args.output_md}")
