#!/usr/bin/env python3
"""Audit player-event market coverage from local API-Football/board evidence.

Separates bookmaker-priced coverage from internal research board coverage for
cards, fouls, tackles, shots, and shots on target.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


DEFAULT_OUTDIR = Path("reports/2026-05-06/player_event_market_coverage_audit")
NORMALIZED_ODDS_DIR = Path("data_sources/api_football/normalized")
PLAYER_EVENTS_DIR = Path("data_sources/api_football/features/player_events")
MASTER_BOARD = Path("reports/player_events/combined_boards/master_specialist_board.csv")

MARKET_KEYWORDS = {
    "yellow_cards": ["card", "booking", "yellow"],
    "fouls_committed": ["foul"],
    "tackles": ["tackle"],
    "shots": ["shots", "shot"],
    "shots_on_target": ["shots on target", "shot on target", "sot"],
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


def parse_league_year(path: Path, prefix: str) -> tuple[str, str]:
    stem = path.stem
    match = re.match(rf"{re.escape(prefix)}__(.+)__(20\d{{2}})$", stem)
    if not match:
        return "COMBINED_OR_UNKNOWN", "UNKNOWN"
    return match.group(1).replace("_", " "), match.group(2)


def classify_market(text: str) -> str:
    lower = str(text).lower()
    for market, keywords in MARKET_KEYWORDS.items():
        if any(keyword in lower for keyword in keywords):
            return market
    return ""


def odds_coverage() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    books = []
    for path in sorted(NORMALIZED_ODDS_DIR.glob("odds_prematch_long*.csv")):
        league, year = parse_league_year(path, "odds_prematch_long")
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        total_rows = int(len(df))
        if df.empty:
            rows.append(
                {
                    "league": league,
                    "season": year,
                    "market": "ANY_PLAYER_EVENT",
                    "priced_rows": 0,
                    "priced_fixtures": 0,
                    "bookmakers": 0,
                    "source_rows": total_rows,
                }
            )
            continue
        market_text = (
            df.get("market_name", pd.Series("", index=df.index)).astype(str)
            + " "
            + df.get("selection_name", pd.Series("", index=df.index)).astype(str)
            + " "
            + df.get("market_code", pd.Series("", index=df.index)).astype(str)
        )
        df = df.copy()
        df["player_event_market"] = market_text.map(classify_market)
        events = df[df["player_event_market"].ne("")].copy()
        if events.empty:
            rows.append(
                {
                    "league": league,
                    "season": year,
                    "market": "ANY_PLAYER_EVENT",
                    "priced_rows": 0,
                    "priced_fixtures": 0,
                    "bookmakers": 0,
                    "source_rows": total_rows,
                }
            )
            continue
        grouped = (
            events.groupby("player_event_market", dropna=False)
            .agg(
                priced_rows=("fixture_id", "size"),
                priced_fixtures=("fixture_id", "nunique"),
                bookmakers=("bookmaker_name", "nunique") if "bookmaker_name" in events.columns else ("fixture_id", "nunique"),
            )
            .reset_index()
            .rename(columns={"player_event_market": "market"})
        )
        grouped["league"] = league
        grouped["season"] = year
        grouped["source_rows"] = total_rows
        rows.extend(grouped.to_dict("records"))
        if "bookmaker_name" in events.columns:
            book_group = (
                events.groupby(["player_event_market", "bookmaker_name"], dropna=False)
                .agg(priced_rows=("fixture_id", "size"), priced_fixtures=("fixture_id", "nunique"))
                .reset_index()
                .rename(columns={"player_event_market": "market"})
            )
            book_group["league"] = league
            book_group["season"] = year
            books.extend(book_group.to_dict("records"))
    return pd.DataFrame(rows), pd.DataFrame(books)


def fixture_flag_coverage() -> pd.DataFrame:
    rows = []
    for path in sorted(PLAYER_EVENTS_DIR.glob("player_events_fixture_input__*.csv")):
        league, year = parse_league_year(path, "player_events_fixture_input")
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        fixture_count = int(df.get("fixture_key", pd.Series(dtype=str)).nunique())
        for market, col in [
            ("yellow_cards", "market_yellow_cards_available"),
            ("fouls_committed", "market_fouls_available"),
            ("team_cards", "market_team_cards_available"),
        ]:
            if col in df.columns:
                flagged = df[df[col].fillna(0).astype(float).gt(0)]
                rows.append(
                    {
                        "league": league,
                        "season": year,
                        "market": market,
                        "availability_flag_rows": int(len(flagged)),
                        "availability_flag_fixtures": int(flagged.get("fixture_key", pd.Series(dtype=str)).nunique()),
                        "fixture_input_fixtures": fixture_count,
                    }
                )
    return pd.DataFrame(rows)


def research_board_coverage() -> pd.DataFrame:
    if not MASTER_BOARD.exists():
        return pd.DataFrame()
    df = pd.read_csv(MASTER_BOARD)
    if df.empty:
        return pd.DataFrame()
    if "match_date" in df.columns:
        df["season"] = pd.to_datetime(df["match_date"], errors="coerce").dt.year.astype("Int64").astype(str)
    else:
        df["season"] = "UNKNOWN"
    return (
        df.groupby(["league", "season", "market"], dropna=False)
        .agg(
            research_rows=("fixture_key", "size"),
            research_fixtures=("fixture_key", "nunique"),
            research_players=("player_name", "nunique"),
            p1_rows=("priority_bucket", lambda s: int((s == "P1_SUPER_ELITE").sum()) if "priority_bucket" in df.columns else 0),
        )
        .reset_index()
    )


def build_matrix(odds: pd.DataFrame, flags: pd.DataFrame, research: pd.DataFrame) -> pd.DataFrame:
    keys = []
    for frame in [odds, flags, research]:
        if not frame.empty:
            keys.extend(frame[["league", "season", "market"]].drop_duplicates().to_dict("records"))
    base = pd.DataFrame(keys).drop_duplicates() if keys else pd.DataFrame(columns=["league", "season", "market"])
    if base.empty:
        return base
    out = base.merge(odds, on=["league", "season", "market"], how="left")
    out = out.merge(flags, on=["league", "season", "market"], how="left")
    out = out.merge(research, on=["league", "season", "market"], how="left")
    for col in [
        "priced_rows",
        "priced_fixtures",
        "bookmakers",
        "availability_flag_rows",
        "availability_flag_fixtures",
        "fixture_input_fixtures",
        "research_rows",
        "research_fixtures",
        "research_players",
        "p1_rows",
    ]:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    def status(row: pd.Series) -> str:
        if row["priced_rows"] > 0 or row["availability_flag_fixtures"] > 0:
            return "PRICED_MARKET_CONFIRMED"
        if row["research_rows"] > 0:
            return "RESEARCH_ONLY_NO_BOOK_COVERAGE"
        if row["fixture_input_fixtures"] > 0:
            return "FIXTURE_INPUT_NO_MARKET_FLAG"
        return "NO_EVIDENCE"

    out["coverage_status"] = out.apply(status, axis=1)
    return out.sort_values(["coverage_status", "league", "market"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    odds, books = odds_coverage()
    flags = fixture_flag_coverage()
    research = research_board_coverage()
    matrix = build_matrix(odds, flags, research)

    odds.to_csv(outdir / "player_event_market_odds_coverage.csv", index=False)
    books.to_csv(outdir / "player_event_market_bookmaker_coverage.csv", index=False)
    flags.to_csv(outdir / "player_event_market_fixture_flag_coverage.csv", index=False)
    research.to_csv(outdir / "player_event_market_research_board_coverage.csv", index=False)
    matrix.to_csv(outdir / "player_event_market_coverage_matrix.csv", index=False)

    status_counts = matrix.groupby("coverage_status", dropna=False).size().reset_index(name="cells") if not matrix.empty else pd.DataFrame()
    market_counts = (
        matrix.groupby(["market", "coverage_status"], dropna=False)
        .agg(cells=("league", "size"), priced_fixtures=("priced_fixtures", "sum"), research_fixtures=("research_fixtures", "sum"))
        .reset_index()
        if not matrix.empty
        else pd.DataFrame()
    )
    report = [
        "# Player Event Market Coverage Audit",
        "",
        "Audit of bookmaker-priced player-event market evidence versus internal research-board coverage.",
        "",
        "## Key Rule",
        "Research-board rows do not prove bookmaker availability. Priced coverage must be shown separately before app/web productization.",
        "",
        "## Coverage Status Counts",
        markdown_table(status_counts),
        "",
        "## Market Coverage Summary",
        markdown_table(market_counts),
        "",
        "## Coverage Matrix",
        markdown_table(matrix.head(80)),
        "",
        "## Operating Recommendation",
        "- Cards/fouls/tackles/shots/SOT should stay beta/manual-review until priced market coverage is populated.",
        "- Current local `odds_prematch_long` evidence is sparse/empty for player-event markets, so bookmaker coverage ingestion is the next bottleneck.",
        "- Use the specialist boards for discovery and analyst review, not as proof that the webapp can offer the market everywhere.",
    ]
    (outdir / "player_event_market_coverage_audit.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"WROTE {outdir}")
    print(f"matrix_rows={len(matrix)}")


if __name__ == "__main__":
    main()
