#!/usr/bin/env python3
"""Build a research audit for API-Football as an intelligence overlay.

This audit intentionally separates API-Football from core pick generation.
FootyStats/model outputs remain the pick spine; API-Football is assessed for
context, lineup/injury/referee intelligence, and player-event specialist markets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_PLAYER_EVENTS_DIR = Path("data_sources/api_football/features/player_events")
DEFAULT_MASTER_BOARD = Path("reports/player_events/combined_boards/master_specialist_board.csv")
DEFAULT_BACKTEST_PACK = Path(
    "reports/player_events/backtests/player_events_3y_backtest__2026-05-04__tuned_cycle_6_recent_relaxed_contact"
)
DEFAULT_API_COVERAGE = Path("reports/2026-05-06/api_football_feature_coverage_audit/api_football_enrichment_readiness.csv")
DEFAULT_OUTDIR = Path("reports/2026-05-06/api_intelligence_overlay")

CORE_EVENT_MARKETS = {"yellow_cards", "fouls_committed", "tackles", "shots", "shots_on_target"}
PRIMARY_PLAYER_EVENT_LEAGUES = {
    "England Premier League",
    "Spain La Liga",
    "Italy Serie A",
    "Germany Bundesliga",
    "France Ligue 1",
    "UEFA Champions League",
    "UEFA Europa League",
    "Champions League",
    "Europa League",
}


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


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def inventory_player_event_files(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("*.csv")):
        stem = path.stem
        family = stem
        league = "UNKNOWN"
        season = "UNKNOWN"
        parts = stem.split("__")
        if len(parts) >= 3:
            family = parts[0]
            league = parts[1].replace("_", " ")
            season = parts[2]
        try:
            header = pd.read_csv(path, nrows=0)
            frame = pd.read_csv(path, low_memory=False)
            rows.append(
                {
                    "family": family,
                    "league": league,
                    "season": season,
                    "rows": len(frame),
                    "columns": len(header.columns),
                    "path": str(path),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "family": family,
                    "league": league,
                    "season": season,
                    "rows": 0,
                    "columns": 0,
                    "path": str(path),
                    "read_error": str(exc),
                }
            )
    return pd.DataFrame(rows)


def classify_overlay_readiness(row: pd.Series) -> str:
    fixture_input = int(row.get("player_events_fixture_input_rows", 0))
    referee = int(row.get("referee_profiles_rows", 0))
    style = int(row.get("fixture_style_overlay_rows", 0))
    goal = int(row.get("og_goal_environment_overlay_rows", 0))
    form = int(row.get("player_form_quality_overlay_rows", 0))
    primary = bool(row.get("primary_player_event_league", False))
    if primary and fixture_input > 0 and referee > 0 and style > 0 and form > 0:
        return "PLAYER_EVENT_CORE_RESEARCH"
    if fixture_input > 0 and (referee > 0 or style > 0) and (goal > 0 or form > 0):
        return "PLAYER_EVENT_WATCH"
    if fixture_input > 0:
        return "CONTEXT_ONLY"
    return "NO_PLAYER_EVENT_LAYER"


def build_readiness(inventory: pd.DataFrame, api_coverage: pd.DataFrame) -> pd.DataFrame:
    if inventory.empty:
        return pd.DataFrame()
    pivot = (
        inventory.pivot_table(index=["league", "season"], columns="family", values="rows", aggfunc="sum", fill_value=0)
        .reset_index()
        .rename_axis(None, axis=1)
    )
    known_families = [
        "player_events_fixture_input",
        "referee_profiles",
        "fixture_style_overlay",
        "og_goal_environment_overlay",
        "player_form_quality_overlay",
        "manual_side_enrichment",
    ]
    for family in known_families:
        if family in pivot.columns:
            pivot[f"{family}_rows"] = pivot[family].astype(int)
            pivot = pivot.drop(columns=[family])
        else:
            pivot[f"{family}_rows"] = 0
    if not api_coverage.empty and "league" in api_coverage.columns:
        cov = api_coverage[["league", "best_bucket", "max_populated_families"]].drop_duplicates("league")
        pivot = pivot.merge(cov, on="league", how="left")
    else:
        pivot["best_bucket"] = ""
        pivot["max_populated_families"] = pd.NA
    pivot["primary_player_event_league"] = pivot["league"].isin(PRIMARY_PLAYER_EVENT_LEAGUES)
    pivot["overlay_readiness"] = pivot.apply(classify_overlay_readiness, axis=1)
    return pivot.sort_values(["overlay_readiness", "league", "season"])


def summarize_master_board(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if master.empty:
        return pd.DataFrame(), pd.DataFrame()
    market_summary = (
        master.groupby(["market", "priority_bucket"], dropna=False)
        .agg(
            rows=("fixture_key", "size"),
            fixtures=("fixture_key", "nunique"),
            players=("player_name", "nunique"),
            avg_market_score=("market_score", "mean"),
            avg_fixture_quality=("fixture_quality_score", "mean"),
        )
        .reset_index()
        .sort_values(["priority_bucket", "rows"], ascending=[True, False])
    )
    league_summary = (
        master.groupby(["league", "market"], dropna=False)
        .agg(
            rows=("fixture_key", "size"),
            fixtures=("fixture_key", "nunique"),
            players=("player_name", "nunique"),
            avg_market_score=("market_score", "mean"),
            p1_rows=("priority_bucket", lambda s: int((s == "P1_SUPER_ELITE").sum())),
        )
        .reset_index()
        .sort_values(["p1_rows", "rows"], ascending=[False, False])
    )
    return market_summary, league_summary


def backtest_snapshot(backtest_dir: Path) -> pd.DataFrame:
    summary_path = backtest_dir / "player_events_3y_backtest_runner.csv"
    if not summary_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(summary_path)
    group_cols = [col for col in ["market", "source_family", "formation_matchup_label"] if col in df.columns]
    hit_col = "hit_flag" if "hit_flag" in df.columns else "observed_success_flag" if "observed_success_flag" in df.columns else ""
    if not group_cols or not hit_col:
        return df.head(0)
    return (
        df.groupby(group_cols, dropna=False)
        .agg(
            rows=(hit_col, "size"),
            fixtures=("fixture_key", "nunique"),
            observed_hit=(hit_col, "mean"),
            expected_hit_rate_3y=("expected_hit_rate_3y", "mean") if "expected_hit_rate_3y" in df.columns else (hit_col, "mean"),
            near_misses=("near_miss_flag", "sum") if "near_miss_flag" in df.columns else (hit_col, "sum"),
            missed_correct=("missed_correct_flag", "sum") if "missed_correct_flag" in df.columns else (hit_col, "sum"),
        )
        .reset_index()
        .sort_values(["observed_hit", "rows"], ascending=[False, False])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player-events-dir", default=str(DEFAULT_PLAYER_EVENTS_DIR))
    parser.add_argument("--master-board", default=str(DEFAULT_MASTER_BOARD))
    parser.add_argument("--backtest-pack", default=str(DEFAULT_BACKTEST_PACK))
    parser.add_argument("--api-coverage", default=str(DEFAULT_API_COVERAGE))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    inventory = inventory_player_event_files(Path(args.player_events_dir))
    api_coverage = read_csv_if_exists(Path(args.api_coverage))
    readiness = build_readiness(inventory, api_coverage)
    master = read_csv_if_exists(Path(args.master_board))
    market_summary, league_summary = summarize_master_board(master)
    bt_snapshot = backtest_snapshot(Path(args.backtest_pack))

    inventory.to_csv(outdir / "api_intelligence_player_event_file_inventory.csv", index=False)
    readiness.to_csv(outdir / "api_intelligence_overlay_readiness.csv", index=False)
    market_summary.to_csv(outdir / "api_intelligence_master_board_market_summary.csv", index=False)
    league_summary.to_csv(outdir / "api_intelligence_master_board_league_summary.csv", index=False)
    bt_snapshot.to_csv(outdir / "api_intelligence_player_event_backtest_snapshot.csv", index=False)

    readiness_counts = (
        readiness.groupby("overlay_readiness", dropna=False).size().reset_index(name="league_season_cells")
        if not readiness.empty
        else pd.DataFrame()
    )
    family_totals = (
        inventory.groupby("family", dropna=False)
        .agg(files=("path", "count"), rows=("rows", "sum"))
        .reset_index()
        .sort_values("rows", ascending=False)
        if not inventory.empty
        else pd.DataFrame()
    )
    report = [
        "# API Intelligence Overlay Audit",
        "",
        "Research-only posture: FootyStats/model outputs generate the core picks. "
        "API-Football supplies contextual intelligence, player-event specialist inputs, and manual review flags.",
        "",
        "## Doctrine",
        "- API-Football must not be routed as a parallel upstream deploy flow for FTR/BTTS/OU25.",
        "- API-Football can veto, annotate, or downgrade confidence when absences, lineups, rest, or referee context disagree.",
        "- API-Football is first-class for player-event research: cards, fouls, tackles, shots, and shots on target.",
        "- Player-event outputs remain beta/manual-review unless separately backtested and market coverage is proven.",
        "",
        "## Readiness Counts",
        markdown_table(readiness_counts),
        "",
        "## Player-Event File Families",
        markdown_table(family_totals),
        "",
        "## Overlay Readiness",
        markdown_table(
            readiness[
                [
                    "league",
                    "season",
                    "overlay_readiness",
                    "player_events_fixture_input_rows",
                    "referee_profiles_rows",
                    "fixture_style_overlay_rows",
                    "og_goal_environment_overlay_rows",
                    "player_form_quality_overlay_rows",
                    "best_bucket",
                ]
            ].head(60)
            if not readiness.empty
            else readiness
        ),
        "",
        "## Current Master Specialist Board By Market",
        markdown_table(market_summary.head(40)),
        "",
        "## Current Master Specialist Board By League",
        markdown_table(league_summary.head(40)),
        "",
        "## Backtest Snapshot",
        markdown_table(bt_snapshot.head(30)),
        "",
        "## Recommended Next Build",
        "- Build `API_CONTEXT_OVERLAY_BOARD`: fixture-level absence, lineup rotation, rest, referee, and tactical style flags joined to FootyStats picks as annotation only.",
        "- Build `PLAYER_EVENT_MARKET_COVERAGE_AUDIT`: bookmaker/league availability for cards, fouls, tackles, shots, and SOT.",
        "- Keep player-event boards as beta/manual-review outputs until market-specific backtests and Monday audits are boring.",
        "- Use API-Football to discover specialist markets, not to dilute the proven FootyStats core spine.",
    ]
    (outdir / "api_intelligence_overlay_audit.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"WROTE {outdir}")
    print(f"inventory_files={len(inventory)} master_rows={len(master)}")


if __name__ == "__main__":
    main()
