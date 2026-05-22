#!/usr/bin/env python3
"""
Research-only EPL final-day player-event cluster audit.

This does not recalibrate live player-event rules. It asks whether historical
EPL final-day 10-match clusters behave differently from normal matchdays for
shots, SOT, fouls committed, fouls drawn, tackles, cards, and saves.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EVENT_MARKETS = {
    "shots_ge1": ("shots_total", 1),
    "sot_ge1": ("shots_on_target", 1),
    "fouls_committed_ge1": ("fouls_committed", 1),
    "fouls_drawn_ge1": ("fouls_drawn", 1),
    "tackles_ge1": ("tackles", 1),
    "card_any": ("card_any", 1),
    "keeper_saves_ge2": ("saves", 2),
}

COUNT_COLS = [
    "shots_total",
    "shots_on_target",
    "fouls_committed",
    "fouls_drawn",
    "tackles",
    "card_any",
    "saves",
]


def load_season(normalized_dir: Path, league_tag: str, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    fixtures_path = normalized_dir / f"fixtures_master__{league_tag}__{season}.csv"
    players_path = normalized_dir / f"match_player_stats__{league_tag}__{season}.csv"
    if not fixtures_path.exists() or not players_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    fixtures = pd.read_csv(fixtures_path, low_memory=False)
    players = pd.read_csv(players_path, low_memory=False)
    fixtures["season"] = season
    players["season"] = season
    fixtures["match_date"] = pd.to_datetime(fixtures["match_date"], errors="coerce")
    return fixtures, players


def build_base(normalized_dir: Path, league_tag: str, seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fixture_frames = []
    player_frames = []
    cluster_rows = []

    for season in seasons:
        fixtures, players = load_season(normalized_dir, league_tag, season)
        if fixtures.empty or players.empty:
            continue
        fixture_frames.append(fixtures)
        player_frames.append(players)

        date_counts = (
            fixtures.groupby("match_date", dropna=False)
            .agg(fixtures=("fixture_id", "nunique"))
            .reset_index()
            .sort_values(["fixtures", "match_date"], ascending=[False, True])
        )
        max_date = fixtures["match_date"].max()
        final_date_rows = date_counts[date_counts["match_date"].eq(max_date)]
        final_day_fixture_count = int(final_date_rows["fixtures"].iloc[0]) if not final_date_rows.empty else 0
        completed_season_flag = int(len(fixtures) >= 380 and final_day_fixture_count >= 10)
        for _, row in date_counts.iterrows():
            cluster_rows.append(
                {
                    "season": season,
                    "match_date": row["match_date"].date().isoformat() if pd.notna(row["match_date"]) else "",
                    "fixtures": int(row["fixtures"]),
                    "is_final_date": int(row["match_date"] == max_date),
                    "completed_season_flag": completed_season_flag,
                    "is_final_day_cluster": int(completed_season_flag and row["match_date"] == max_date),
                }
            )

    if not fixture_frames:
        raise FileNotFoundError("No requested EPL API-Football seasons were found.")

    fixtures_all = pd.concat(fixture_frames, ignore_index=True)
    players_all = pd.concat(player_frames, ignore_index=True)
    clusters = pd.DataFrame(cluster_rows)
    return fixtures_all, players_all, clusters


def enrich_players(players: pd.DataFrame, fixtures: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    keep = ["fixture_id", "season", "match_date", "home_team_name", "away_team_name"]
    df = players.merge(fixtures[keep], on=["fixture_id", "season"], how="left")
    cluster_flags = clusters[["season", "match_date", "is_final_day_cluster", "completed_season_flag"]].drop_duplicates()
    cluster_flags["match_date"] = pd.to_datetime(cluster_flags["match_date"], errors="coerce")
    df = df.merge(cluster_flags, on=["season", "match_date"], how="left")
    df["is_final_day_cluster"] = df["is_final_day_cluster"].fillna(0).astype(int)
    df["completed_season_flag"] = df["completed_season_flag"].fillna(0).astype(int)

    for col in COUNT_COLS + ["minutes", "started_flag"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["card_any"] = (
        pd.to_numeric(df.get("yellow_cards", 0), errors="coerce").fillna(0)
        + pd.to_numeric(df.get("red_cards", 0), errors="coerce").fillna(0)
    )
    df["position"] = df["position"].fillna("UNK").astype(str).str.upper().str[:1]
    return df[df["minutes"].gt(0)].copy()


def fixture_profile(df: pd.DataFrame) -> pd.DataFrame:
    fixture_totals = (
        df.groupby(["season", "fixture_id", "match_date", "home_team_name", "away_team_name", "is_final_day_cluster"], dropna=False)[COUNT_COLS]
        .sum()
        .reset_index()
    )
    rows = []
    for flag, group in fixture_totals.groupby("is_final_day_cluster"):
        label = "final_day_cluster" if int(flag) == 1 else "non_final_day"
        row = {
            "scope": label,
            "fixtures": group["fixture_id"].nunique(),
            "seasons": group["season"].nunique(),
        }
        for col in COUNT_COLS:
            row[f"{col}_per_fixture"] = float(group[col].mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out) == 2:
        final = out[out["scope"].eq("final_day_cluster")].iloc[0]
        base = out[out["scope"].eq("non_final_day")].iloc[0]
        lift = {"scope": "final_day_lift_vs_non_final", "fixtures": final["fixtures"], "seasons": final["seasons"]}
        for col in COUNT_COLS:
            denom = base[f"{col}_per_fixture"]
            lift[f"{col}_per_fixture"] = final[f"{col}_per_fixture"] / denom if denom else np.nan
        out = pd.concat([out, pd.DataFrame([lift])], ignore_index=True)
    return out


def player_market_rates(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope_name, scope_df in {
        "all_played": df,
        "starters_only": df[df["started_flag"].eq(1)],
        "keepers_only": df[df["position"].eq("G")],
    }.items():
        for final_flag, group in scope_df.groupby("is_final_day_cluster"):
            day_scope = "final_day_cluster" if int(final_flag) == 1 else "non_final_day"
            for market, (col, threshold) in EVENT_MARKETS.items():
                if market == "keeper_saves_ge2" and scope_name != "keepers_only":
                    continue
                if market != "keeper_saves_ge2" and scope_name == "keepers_only":
                    continue
                if len(group) == 0:
                    continue
                values = pd.to_numeric(group[col], errors="coerce").fillna(0)
                rows.append(
                    {
                        "scope": scope_name,
                        "day_scope": day_scope,
                        "market": market,
                        "rows": len(group),
                        "fixtures": group["fixture_id"].nunique(),
                        "hit_rate": float(values.ge(threshold).mean()),
                        "avg_value": float(values.mean()),
                    }
                )
    rates = pd.DataFrame(rows)
    lift_rows = []
    for (scope, market), group in rates.groupby(["scope", "market"], dropna=False):
        final = group[group["day_scope"].eq("final_day_cluster")]
        base = group[group["day_scope"].eq("non_final_day")]
        if final.empty or base.empty:
            continue
        f = final.iloc[0]
        b = base.iloc[0]
        lift_rows.append(
            {
                "scope": scope,
                "day_scope": "final_day_lift_vs_non_final",
                "market": market,
                "rows": int(f["rows"]),
                "fixtures": int(f["fixtures"]),
                "hit_rate": f["hit_rate"] / b["hit_rate"] if b["hit_rate"] else np.nan,
                "avg_value": f["avg_value"] / b["avg_value"] if b["avg_value"] else np.nan,
            }
        )
    return pd.concat([rates, pd.DataFrame(lift_rows)], ignore_index=True)


def system_label_final_day_rates(validation_rows_path: Path, clusters: pd.DataFrame) -> pd.DataFrame:
    if not validation_rows_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(validation_rows_path, low_memory=False)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    flags = clusters[["season", "match_date", "is_final_day_cluster"]].drop_duplicates()
    flags["match_date"] = pd.to_datetime(flags["match_date"], errors="coerce")
    df = df.merge(flags, on=["season", "match_date"], how="left", suffixes=("", "_cluster"))
    if "is_final_day_cluster_cluster" in df.columns:
        df["is_final_day_cluster"] = df["is_final_day_cluster_cluster"].fillna(df.get("is_final_day_cluster", 0)).fillna(0).astype(int)
    else:
        df["is_final_day_cluster"] = df["is_final_day_cluster"].fillna(0).astype(int)
    rows = []
    for market in EVENT_MARKETS:
        label_col = f"{market}_system_style_label"
        if market not in df.columns or label_col not in df.columns:
            continue
        if market == "keeper_saves_ge2":
            scoped = df[df["position"].astype(str).str.upper().str.startswith("G")].copy()
        else:
            scoped = df.copy()
        for (final_flag, label), group in scoped.groupby(["is_final_day_cluster", label_col], dropna=False):
            if len(group) < 20:
                continue
            rows.append(
                {
                    "market": market,
                    "day_scope": "final_day_cluster" if int(final_flag) == 1 else "non_final_day",
                    "label": label,
                    "rows": len(group),
                    "fixtures": group["fixture_id"].nunique(),
                    "hit_rate": float(pd.to_numeric(group[market], errors="coerce").fillna(0).mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["market", "day_scope", "hit_rate"], ascending=[True, True, False])


def write_report(outdir: Path, clusters: pd.DataFrame, fixture_rates: pd.DataFrame, player_rates: pd.DataFrame, label_rates: pd.DataFrame) -> None:
    lines = [
        "# API-Football EPL Final-Day Player-Event Cluster Audit",
        "",
        "- Scope: research only. No player-event thresholds, deploy routes, or Sunday board logic changed.",
        "- Question: do historical EPL final-day 10-match clusters show higher shots/contact/tackles than normal matchdays?",
        "- Completed seasons included as true final-day clusters: local EPL API-Football seasons with `380` fixtures and a final date with `10` fixtures.",
        "",
        "## Final-Day Clusters Found",
        "",
    ]
    finals = clusters[clusters["is_final_day_cluster"].eq(1)].sort_values(["season", "match_date"])
    for _, row in finals.iterrows():
        lines.append(f"- `{int(row['season'])}` final date `{row['match_date']}`: `{int(row['fixtures'])}` fixtures.")

    lines.extend(["", "## Fixture-Level Lift", ""])
    lift = fixture_rates[fixture_rates["scope"].eq("final_day_lift_vs_non_final")]
    if not lift.empty:
        row = lift.iloc[0]
        for col in COUNT_COLS:
            lines.append(f"- `{col}` per fixture final-day lift: `{row[f'{col}_per_fixture']:.3f}x`.")

    lines.extend(["", "## Player-Market Final-Day Hit Rates", ""])
    final_rates = player_rates[player_rates["day_scope"].eq("final_day_cluster")]
    for _, row in final_rates.sort_values(["scope", "market"]).iterrows():
        lines.append(
            f"- `{row['scope']}` `{row['market']}`: `{row['hit_rate']:.1%}` over `{int(row['rows'])}` rows / `{int(row['fixtures'])}` fixtures."
        )

    lines.extend(["", "## System-Style Proxy Labels On Final Day", ""])
    if not label_rates.empty:
        focus = label_rates[
            label_rates["day_scope"].eq("final_day_cluster")
            & label_rates["label"].isin(["SHADOW_CORE_PROXY", "STRONG_WATCH_PROXY", "ALT_WATCH_PROXY"])
        ]
        for _, row in focus.sort_values(["market", "hit_rate"], ascending=[True, False]).iterrows():
            lines.append(
                f"- `{row['market']}` `{row['label']}`: `{row['hit_rate']:.1%}` over `{int(row['rows'])}` rows."
            )
    else:
        lines.append("- Existing system-style validation rows not found; skipped label overlay.")

    lines.extend(
        [
            "",
            "## Read",
            "",
            "- Final-day EPL clusters exist cleanly in each completed local API season from 2022-2024.",
            "- The audit can identify event-rate movement, but it does not yet know exact team stakes such as title, Europe, or relegation pressure.",
            "- For Sunday, use this as a pressure-context check only. Confirmed lineups and exact player roles still matter more than broad final-day averages.",
            "",
            "## Files",
            "",
            f"- `{outdir / 'api_epl_final_day_clusters.csv'}`",
            f"- `{outdir / 'api_epl_final_day_fixture_event_rates.csv'}`",
            f"- `{outdir / 'api_epl_final_day_player_market_rates.csv'}`",
            f"- `{outdir / 'api_epl_final_day_system_label_rates.csv'}`",
        ]
    )
    (outdir / "API_FOOTBALL_EPL_FINAL_DAY_PLAYER_EVENT_CLUSTER_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(
    normalized_dir: Path,
    league_tag: str,
    seasons: list[int],
    validation_rows_path: Path,
    outdir: Path,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fixtures, players, clusters = build_base(normalized_dir, league_tag, seasons)
    enriched = enrich_players(players, fixtures, clusters)
    fixture_rates = fixture_profile(enriched[enriched["completed_season_flag"].eq(1)])
    player_rates = player_market_rates(enriched[enriched["completed_season_flag"].eq(1)])
    label_rates = system_label_final_day_rates(validation_rows_path, clusters)

    clusters.to_csv(outdir / "api_epl_final_day_clusters.csv", index=False)
    fixture_rates.to_csv(outdir / "api_epl_final_day_fixture_event_rates.csv", index=False)
    player_rates.to_csv(outdir / "api_epl_final_day_player_market_rates.csv", index=False)
    label_rates.to_csv(outdir / "api_epl_final_day_system_label_rates.csv", index=False)
    write_report(outdir, clusters, fixture_rates, player_rates, label_rates)
    print(f"[ok] final_day_clusters={int(clusters['is_final_day_cluster'].sum())}")
    print(f"[ok] wrote {outdir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit EPL API-Football final-day player-event clusters.")
    parser.add_argument("--normalized-dir", default="data_sources/api_football/normalized")
    parser.add_argument("--league-tag", default="England_Premier_League")
    parser.add_argument("--seasons", default="2022,2023,2024,2025")
    parser.add_argument(
        "--validation-rows",
        default="reports/latest/api_football_epl_player_event_system_validation/api_epl_player_event_validation_rows.csv",
    )
    parser.add_argument("--outdir", default="reports/latest/api_football_epl_final_day_player_event_cluster_audit")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(
        normalized_dir=Path(args.normalized_dir),
        league_tag=args.league_tag,
        seasons=[int(part.strip()) for part in args.seasons.split(",") if part.strip()],
        validation_rows_path=Path(args.validation_rows),
        outdir=Path(args.outdir),
    )
