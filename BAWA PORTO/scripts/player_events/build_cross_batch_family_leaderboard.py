from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


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


def _best_fixture_text(master_board: pd.DataFrame, family: str) -> str:
    if master_board.empty:
        return "None"
    family_rows = master_board[
        master_board["source_family_tag"].astype(str).str.contains(family, regex=False, na=False)
    ].copy()
    if family_rows.empty:
        return "None"
    best_fixture = (
        family_rows.groupby(["fixture_key", "home_team_name", "away_team_name"], as_index=False)
        .agg(
            top_score=("market_score", "max"),
            rows=("market", "count"),
        )
        .sort_values(["top_score", "rows"], ascending=[False, False])
        .iloc[0]
    )
    return f"{best_fixture['fixture_key']} ({best_fixture['home_team_name']} vs {best_fixture['away_team_name']})"


def _strongest_markets_text(family_market: pd.DataFrame, family: str) -> str:
    market_slice = family_market[family_market["source_family"].eq(family)].sort_values(
        ["hit_rate", "rows"], ascending=[False, False]
    )
    if market_slice.empty:
        return "None"
    return ", ".join(
        f"{row.market} ({row.hit_rate:.3f})" for row in market_slice.head(2).itertuples(index=False)
    )


def _combine_family_summary(frames: list[pd.DataFrame]) -> pd.DataFrame:
    valid = [df for df in frames if not df.empty]
    if not valid:
        return pd.DataFrame()
    combined = pd.concat(valid, ignore_index=True)
    combined["weighted_hits"] = pd.to_numeric(combined["row_hit_rate"], errors="coerce").fillna(0.0) * pd.to_numeric(
        combined["rows"], errors="coerce"
    ).fillna(0.0)
    combined["weighted_quality"] = pd.to_numeric(combined["fixture_quality_avg"], errors="coerce").fillna(0.0) * pd.to_numeric(
        combined["rows"], errors="coerce"
    ).fillna(0.0)
    combined["weighted_market_score"] = pd.to_numeric(combined["market_score_avg"], errors="coerce").fillna(0.0) * pd.to_numeric(
        combined["rows"], errors="coerce"
    ).fillna(0.0)
    grouped = (
        combined.groupby("source_family", as_index=False)
        .agg(
            rows=("rows", "sum"),
            fixtures=("fixtures", "sum"),
            p1_rows=("p1_rows", "sum"),
            attack_rows=("attack_rows", "sum"),
            contact_rows=("contact_rows", "sum"),
            weighted_hits=("weighted_hits", "sum"),
            weighted_quality=("weighted_quality", "sum"),
            weighted_market_score=("weighted_market_score", "sum"),
        )
    )
    grouped["row_hit_rate"] = grouped["weighted_hits"] / grouped["rows"].replace(0, pd.NA)
    grouped["fixture_quality_avg"] = grouped["weighted_quality"] / grouped["rows"].replace(0, pd.NA)
    grouped["market_score_avg"] = grouped["weighted_market_score"] / grouped["rows"].replace(0, pd.NA)
    return grouped.drop(columns=["weighted_hits", "weighted_quality", "weighted_market_score"]).sort_values(
        ["row_hit_rate", "rows", "p1_rows"], ascending=[False, False, False]
    ).reset_index(drop=True)


def _combine_family_market(frames: list[pd.DataFrame]) -> pd.DataFrame:
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
        ["source_family", "hit_rate", "rows"], ascending=[True, False, False]
    ).reset_index(drop=True)


def build_leaderboard(
    batch_names: list[str],
    family_summary_csvs: list[str],
    family_market_csvs: list[str],
    master_board_csvs: list[str],
    output_md: str,
    output_csv: str | None = None,
    output_market_csv: str | None = None,
) -> None:
    summary_map: dict[str, pd.DataFrame] = {}
    market_map: dict[str, pd.DataFrame] = {}
    master_map: dict[str, pd.DataFrame] = {}
    for batch, summary_csv, market_csv, master_csv in zip(batch_names, family_summary_csvs, family_market_csvs, master_board_csvs):
        summary_map[batch] = _load_csv(summary_csv)
        market_map[batch] = _load_csv(market_csv)
        master_map[batch] = _load_csv(master_csv)

    combined_summary = _combine_family_summary(list(summary_map.values()))
    combined_market = _combine_family_market(list(market_map.values()))
    combined_master = pd.concat([df for df in master_map.values() if not df.empty], ignore_index=True) if any(
        not df.empty for df in master_map.values()
    ) else pd.DataFrame()

    summary_export_frames: list[pd.DataFrame] = []
    market_export_frames: list[pd.DataFrame] = []
    lines = ["# Cross-Batch Specialist Family Leaderboard", ""]

    def _render_scope(scope: str, family_summary: pd.DataFrame, family_market: pd.DataFrame, master_board: pd.DataFrame) -> None:
        lines.append(f"## {scope}")
        if family_summary.empty:
            lines.append("No rows matched.")
            lines.append("")
            return
        ranked = family_summary.sort_values(["row_hit_rate", "rows", "p1_rows"], ascending=[False, False, False]).reset_index(drop=True)
        for idx, row in ranked.iterrows():
            family = row["source_family"]
            strongest_markets = _strongest_markets_text(family_market, family)
            best_fixture = _best_fixture_text(master_board, family)
            lines.extend(
                [
                    f"### {idx + 1}. {family}",
                    f"- sample size: {int(row['rows'])} rows across {int(row['fixtures'])} fixtures",
                    f"- hit rate: {float(row['row_hit_rate']):.3f}",
                    f"- strongest markets: {strongest_markets}",
                    f"- best example fixture: {best_fixture}",
                    "",
                ]
            )
        lines.append("#### Family Hit Rates Table")
        family_print = ranked.copy()
        for col in ["row_hit_rate", "fixture_quality_avg", "market_score_avg"]:
            if col in family_print.columns:
                family_print[col] = pd.to_numeric(family_print[col], errors="coerce").round(3)
        lines.extend(_markdown_table(family_print))
        lines.append("")
        lines.append("#### Family x Market Hit Rates Table")
        market_print = family_market.copy().sort_values(["source_family", "hit_rate", "rows"], ascending=[True, False, False])
        if market_print.empty:
            lines.append("No family x market rows matched.")
        else:
            for col in ["hit_rate", "avg_score", "avg_fixture_quality"]:
                if col in market_print.columns:
                    market_print[col] = pd.to_numeric(market_print[col], errors="coerce").round(3)
            lines.extend(_markdown_table(market_print))
        lines.append("")

    for batch in batch_names:
        summary_df = summary_map[batch]
        market_df = market_map[batch]
        master_df = master_map[batch]
        if not summary_df.empty:
            export_df = summary_df.copy()
            export_df.insert(0, "scope", batch)
            summary_export_frames.append(export_df)
        if not market_df.empty:
            export_market_df = market_df.copy()
            export_market_df.insert(0, "scope", batch)
            market_export_frames.append(export_market_df)
        _render_scope(batch, summary_df, market_df, master_df)

    if not combined_summary.empty:
        export_df = combined_summary.copy()
        export_df.insert(0, "scope", "combined_overall")
        summary_export_frames.append(export_df)
    if not combined_market.empty:
        export_market_df = combined_market.copy()
        export_market_df.insert(0, "scope", "combined_overall")
        market_export_frames.append(export_market_df)
    _render_scope("combined_overall", combined_summary, combined_market, combined_master)

    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md).write_text("\n".join(lines) + "\n")

    if output_csv:
        pd.concat(summary_export_frames, ignore_index=True).to_csv(output_csv, index=False)
    if output_market_csv:
        pd.concat(market_export_frames, ignore_index=True).to_csv(output_market_csv, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare specialist family hit rates across multiple greenlist batches.")
    parser.add_argument("--batch-names", required=True, help="Comma-separated batch labels, in the same order as the csv inputs")
    parser.add_argument("--family-summary-csvs", required=True, help="Comma-separated family summary csv paths")
    parser.add_argument("--family-market-csvs", required=True, help="Comma-separated family market csv paths")
    parser.add_argument("--master-board-csvs", required=True, help="Comma-separated master specialist board csv paths")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-csv")
    parser.add_argument("--output-market-csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batch_names = [x.strip() for x in args.batch_names.split(",") if x.strip()]
    family_summary_csvs = [x.strip() for x in args.family_summary_csvs.split(",") if x.strip()]
    family_market_csvs = [x.strip() for x in args.family_market_csvs.split(",") if x.strip()]
    master_board_csvs = [x.strip() for x in args.master_board_csvs.split(",") if x.strip()]
    expected = len(batch_names)
    if not (len(family_summary_csvs) == len(family_market_csvs) == len(master_board_csvs) == expected):
        raise SystemExit("All batch-aligned input lists must have the same length.")
    build_leaderboard(
        batch_names=batch_names,
        family_summary_csvs=family_summary_csvs,
        family_market_csvs=family_market_csvs,
        master_board_csvs=master_board_csvs,
        output_md=args.output_md,
        output_csv=args.output_csv,
        output_market_csv=args.output_market_csv,
    )
    print(f"WROTE: {args.output_md}")
