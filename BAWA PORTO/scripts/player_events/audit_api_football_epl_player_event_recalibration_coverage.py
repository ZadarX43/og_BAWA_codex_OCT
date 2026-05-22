from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_PLAYER_COLUMNS = {
    "shots": ["shots_total"],
    "sot": ["shots_on_target"],
    "fouls_committed": ["fouls_committed"],
    "fouls_drawn": ["fouls_drawn"],
    "tackles": ["tackles"],
    "cards_watchlist": ["yellow_cards", "red_cards"],
    "keeper_saves": ["saves"],
}


def _csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _market_rates(player_stats: pd.DataFrame) -> list[dict[str, object]]:
    if player_stats.empty:
        return []

    rows: list[dict[str, object]] = []
    played = player_stats[pd.to_numeric(player_stats.get("minutes", 0), errors="coerce").fillna(0) > 0].copy()
    starters = played[pd.to_numeric(played.get("started_flag", 0), errors="coerce").fillna(0).astype(int) == 1]

    market_exprs = {
        "shots_ge1": ("shots_total", 1, "all_played"),
        "sot_ge1": ("shots_on_target", 1, "all_played"),
        "fouls_committed_ge1": ("fouls_committed", 1, "all_played"),
        "fouls_drawn_ge1": ("fouls_drawn", 1, "all_played"),
        "tackles_ge1": ("tackles", 1, "all_played"),
        "card_any": ("card_any", 1, "all_played"),
        "keeper_saves_ge1": ("saves", 1, "keepers_only"),
    }

    if "card_any" not in played.columns:
        played["card_any"] = (
            pd.to_numeric(played.get("yellow_cards", 0), errors="coerce").fillna(0)
            + pd.to_numeric(played.get("red_cards", 0), errors="coerce").fillna(0)
        )
        starters = played[pd.to_numeric(played.get("started_flag", 0), errors="coerce").fillna(0).astype(int) == 1]

    for market, (column, threshold, scope) in market_exprs.items():
        scoped = played
        if scope == "keepers_only":
            scoped = played[played.get("position", "").astype(str).str.upper().str.startswith("G")]
        if scoped.empty or column not in scoped.columns:
            continue
        values = pd.to_numeric(scoped[column], errors="coerce").fillna(0)
        rows.append(
            {
                "market": market,
                "scope": scope,
                "rows": len(scoped),
                "fixtures": scoped["fixture_id"].nunique() if "fixture_id" in scoped.columns else 0,
                "hit_rate": float((values >= threshold).mean()),
                "avg_value": float(values.mean()),
                "p90_value": float(values.quantile(0.90)),
            }
        )

    for market, column in [
        ("starter_shots_ge1", "shots_total"),
        ("starter_sot_ge1", "shots_on_target"),
        ("starter_fouls_committed_ge1", "fouls_committed"),
        ("starter_fouls_drawn_ge1", "fouls_drawn"),
        ("starter_tackles_ge1", "tackles"),
    ]:
        if starters.empty or column not in starters.columns:
            continue
        values = pd.to_numeric(starters[column], errors="coerce").fillna(0)
        rows.append(
            {
                "market": market,
                "scope": "starters_only",
                "rows": len(starters),
                "fixtures": starters["fixture_id"].nunique() if "fixture_id" in starters.columns else 0,
                "hit_rate": float((values >= 1).mean()),
                "avg_value": float(values.mean()),
                "p90_value": float(values.quantile(0.90)),
            }
        )
    return rows


def _coverage_for_season(base: Path, league_tag: str, season: int) -> tuple[dict[str, object], list[dict[str, object]], pd.DataFrame]:
    fixtures_path = base / f"fixtures_master__{league_tag}__{season}.csv"
    player_path = base / f"match_player_stats__{league_tag}__{season}.csv"
    events_path = base / f"match_events__{league_tag}__{season}.csv"
    lineups_path = base / f"lineups__{league_tag}__{season}.csv"
    team_stats_path = base / f"match_team_stats__{league_tag}__{season}.csv"
    injuries_path = base / f"injuries__{league_tag}__{season}.csv"

    fixtures = _read_csv(fixtures_path)
    player_stats = _read_csv(player_path)
    events = _read_csv(events_path)
    lineups = _read_csv(lineups_path)

    fixture_ids = set(fixtures["fixture_id"].dropna().astype(str)) if "fixture_id" in fixtures.columns else set()
    player_fixture_ids = set(player_stats["fixture_id"].dropna().astype(str)) if "fixture_id" in player_stats.columns else set()
    event_fixture_ids = set(events["fixture_id"].dropna().astype(str)) if "fixture_id" in events.columns else set()
    lineup_fixture_ids = set(lineups["fixture_id"].dropna().astype(str)) if "fixture_id" in lineups.columns else set()

    required_status: dict[str, str] = {}
    for family, columns in REQUIRED_PLAYER_COLUMNS.items():
        missing = [column for column in columns if column not in player_stats.columns]
        required_status[family] = "MISSING" if missing else "AVAILABLE"

    if not player_stats.empty:
        for family, columns in REQUIRED_PLAYER_COLUMNS.items():
            if required_status[family] == "MISSING":
                continue
            numeric = player_stats[columns].apply(pd.to_numeric, errors="coerce")
            non_null = int(numeric.notna().any(axis=1).sum())
            required_status[family] = "AVAILABLE_NON_NULL" if non_null else "AVAILABLE_EMPTY"

    event_types = ""
    if not events.empty and "event_type" in events.columns:
        event_types = "|".join(sorted(events["event_type"].dropna().astype(str).unique()))

    summary = {
        "league_tag": league_tag,
        "season": season,
        "fixture_rows": len(fixtures),
        "fixture_date_from": fixtures["match_date"].min() if "match_date" in fixtures.columns and not fixtures.empty else "",
        "fixture_date_to": fixtures["match_date"].max() if "match_date" in fixtures.columns and not fixtures.empty else "",
        "player_rows": len(player_stats),
        "player_stat_fixtures": len(player_fixture_ids),
        "lineup_rows": len(lineups),
        "lineup_fixtures": len(lineup_fixture_ids),
        "event_rows": len(events),
        "event_fixtures": len(event_fixture_ids),
        "team_stat_rows": _csv_rows(team_stats_path),
        "injury_rows": _csv_rows(injuries_path),
        "player_fixture_coverage": round(len(player_fixture_ids & fixture_ids) / len(fixture_ids), 4) if fixture_ids else 0.0,
        "lineup_fixture_coverage": round(len(lineup_fixture_ids & fixture_ids) / len(fixture_ids), 4) if fixture_ids else 0.0,
        "event_fixture_coverage": round(len(event_fixture_ids & fixture_ids) / len(fixture_ids), 4) if fixture_ids else 0.0,
        "event_types": event_types,
        **{f"{family}_status": status for family, status in required_status.items()},
    }

    market_rows = _market_rates(player_stats)
    for row in market_rows:
        row["league_tag"] = league_tag
        row["season"] = season
    return summary, market_rows, player_stats


def build(league_tag: str, seasons: list[int], normalized_dir: Path, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    market_rows: list[dict[str, object]] = []
    for season in seasons:
        summary, markets, _ = _coverage_for_season(normalized_dir, league_tag, season)
        summaries.append(summary)
        market_rows.extend(markets)

    summary_df = pd.DataFrame(summaries)
    market_df = pd.DataFrame(market_rows)

    summary_csv = outdir / "api_football_epl_player_event_recalibration_coverage.csv"
    market_csv = outdir / "api_football_epl_player_event_market_base_rates.csv"
    summary_df.to_csv(summary_csv, index=False)
    market_df.to_csv(market_csv, index=False)

    lines = [
        "# API-Football EPL Player-Event Recalibration Coverage",
        "",
        "- Scope: research-only audit for player-event beta recalibration.",
        f"- League tag: `{league_tag}`",
        f"- Seasons: `{', '.join(map(str, seasons))}`",
        "- Target markets: shots, SOT, fouls committed, fouls drawn, tackles, cards watchlist, keeper saves.",
        "",
        "## Coverage",
        "",
    ]

    for _, row in summary_df.iterrows():
        status_bits = [
            f"shots={row['shots_status']}",
            f"sot={row['sot_status']}",
            f"fouls_committed={row['fouls_committed_status']}",
            f"fouls_drawn={row['fouls_drawn_status']}",
            f"tackles={row['tackles_status']}",
            f"cards={row['cards_watchlist_status']}",
            f"saves={row['keeper_saves_status']}",
        ]
        lines.append(
            f"- `{int(row['season'])}`: fixtures `{int(row['fixture_rows'])}`, player rows `{int(row['player_rows'])}`, "
            f"player fixture coverage `{row['player_fixture_coverage']:.1%}`, lineups `{row['lineup_fixture_coverage']:.1%}`, "
            f"events `{row['event_fixture_coverage']:.1%}`; {'; '.join(status_bits)}"
        )

    lines.extend(["", "## Base Rates", ""])
    if not market_df.empty:
        display = market_df.sort_values(["market", "season"])
        for market, group in display.groupby("market", sort=True):
            bits = [
                f"{int(row['season'])}: {row['hit_rate']:.1%} rows={int(row['rows'])}"
                for _, row in group.iterrows()
            ]
            lines.append(f"- `{market}`: " + " | ".join(bits))

    lines.extend(
        [
            "",
            "## Read",
            "",
            "- API-Football has the fields needed to recalibrate the Sunday player-event shapes for EPL.",
            "- 2022-2024 are the clean historical calibration block in the current local archive.",
            "- 2025 is usable as a current-season/live-form block, but should be refreshed after the final fixtures before being treated as complete.",
            "- Cards should remain a watchlist layer because card outcomes are much noisier than shots/SOT and contact volume.",
            "- Fouls committed and fouls drawn must remain separate canonical markets in every board and slip export.",
            "",
            "## Files",
            "",
            f"- `{summary_csv}`",
            f"- `{market_csv}`",
        ]
    )

    report_path = outdir / "API_FOOTBALL_EPL_PLAYER_EVENT_RECALIBRATION_COVERAGE.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote {outdir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit API-Football EPL player-event coverage for recalibration.")
    parser.add_argument("--league-tag", default="England_Premier_League")
    parser.add_argument("--seasons", default="2022,2023,2024,2025")
    parser.add_argument("--normalized-dir", default="data_sources/api_football/normalized")
    parser.add_argument("--outdir", default="reports/latest/api_football_epl_player_event_recalibration_coverage")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(
        league_tag=args.league_tag,
        seasons=[int(part.strip()) for part in args.seasons.split(",") if part.strip()],
        normalized_dir=Path(args.normalized_dir),
        outdir=Path(args.outdir),
    )
