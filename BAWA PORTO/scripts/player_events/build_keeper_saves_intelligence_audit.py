#!/usr/bin/env python3
"""Build a research-only keeper saves intelligence audit.

This joins leak-safe team SOT pressure profiles to settled goalkeeper saves.
Rows are framed as: attacking team SOT pressure -> opposing keeper saves.
No deploy routing, priced odds, or slips are changed.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
NORMALIZED_DIR = ROOT / "data_sources" / "api_football" / "normalized"
DEFAULT_TEAM_PROFILE = ROOT / "reports" / "2026-05-06" / "team_shots_profile" / "TEAM_SHOTS_PROFILE_ROWS.csv"
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-07" / "keeper_saves_intelligence_audit"

MARKETS = {
    "Keeper Saves 1.5+": 2,
    "Keeper Saves 2.5+": 3,
    "Keeper Saves 3.5+": 4,
}
SCORE_THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
TOP_N_PER_FIXTURE = [1, 2]


def num(values: Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def parse_csv_arg(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def parse_int_arg(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    work = df.head(max_rows).copy()
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
        else:
            work[col] = work[col].astype("string").fillna("")
    lines = ["| " + " | ".join(work.columns) + " |", "| " + " | ".join(["---"] * len(work.columns)) + " |"]
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "/") for col in work.columns) + " |")
    return "\n".join(lines)


def percentile(series: pd.Series) -> pd.Series:
    values = num(series)
    if values.notna().sum() <= 1:
        return pd.Series(0.0, index=series.index)
    return values.rank(pct=True).fillna(0.0)


def load_profiles(path: Path, leagues: set[str] | None, seasons: set[int] | None) -> pd.DataFrame:
    profiles = pd.read_csv(path, low_memory=False)
    if leagues:
        profiles = profiles[profiles["league_tag"].astype(str).isin(leagues)].copy()
    if seasons:
        profiles = profiles[profiles["season_tag"].astype(int).isin(seasons)].copy()
    return profiles


def load_goalkeepers(leagues: set[str] | None, seasons: set[int] | None) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for stats_path in sorted(NORMALIZED_DIR.glob("match_player_stats__*.csv")):
        match = re.match(r"match_player_stats__(.+)__(\d{4})\.csv$", stats_path.name)
        if not match:
            continue
        league_tag, season_tag = match.group(1), int(match.group(2))
        if leagues and league_tag not in leagues:
            continue
        if seasons and season_tag not in seasons:
            continue
        fixtures_path = NORMALIZED_DIR / f"fixtures_master__{league_tag}__{season_tag}.csv"
        if not fixtures_path.exists():
            continue
        stats = pd.read_csv(
            stats_path,
            low_memory=False,
            usecols=["fixture_id", "team_id", "player_name", "position", "minutes", "started_flag", "saves", "goals_conceded"],
        )
        fixtures = pd.read_csv(fixtures_path, low_memory=False, usecols=["fixture_id", "fixture_key", "match_date", "league"])
        stats["minutes"] = num(stats["minutes"])
        stats["saves"] = num(stats["saves"])
        gk = stats[stats["position"].astype(str).str.upper().eq("G") & stats["minutes"].fillna(0).gt(0)].copy()
        if gk.empty:
            continue
        gk = gk.sort_values(["fixture_id", "team_id", "minutes"], ascending=[True, True, False])
        gk = gk.drop_duplicates(["fixture_id", "team_id"], keep="first")
        gk = gk.merge(fixtures, on="fixture_id", how="left")
        gk["league_tag"] = league_tag
        gk["season_tag"] = season_tag
        rows.append(gk)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def add_scores(profiles: pd.DataFrame, keepers: pd.DataFrame) -> pd.DataFrame:
    rows = profiles.copy()
    for col in [
        "fixture_id",
        "opponent_team_id",
        "team_expected_sot",
        "keeper_save_pressure_score",
        "team_sot_for_l5",
        "team_sot_for_l10",
        "opponent_sot_allowed_l5",
        "opponent_sot_allowed_l10",
        "team_high_sot_volume_rate_l5",
        "team_expected_shots",
        "actual_shots_on_goal",
    ]:
        if col not in rows.columns:
            rows[col] = np.nan
        rows[col] = num(rows[col])
    rows = rows.merge(
        keepers[
            [
                "fixture_id",
                "team_id",
                "player_name",
                "minutes",
                "started_flag",
                "saves",
                "goals_conceded",
            ]
        ].rename(
            columns={
                "team_id": "opponent_team_id",
                "player_name": "keeper_name",
                "minutes": "keeper_minutes",
                "started_flag": "keeper_started_flag",
                "saves": "actual_keeper_saves",
                "goals_conceded": "actual_keeper_goals_conceded",
            }
        ),
        on=["fixture_id", "opponent_team_id"],
        how="left",
    )
    rows["profile_ready"] = rows.get("profile_ready", pd.Series(False, index=rows.index)).fillna(False).astype(bool)
    within = rows.groupby("league_tag", dropna=False)
    for col in [
        "keeper_save_pressure_score",
        "team_expected_sot",
        "team_sot_for_l5",
        "opponent_sot_allowed_l5",
        "team_high_sot_volume_rate_l5",
        "team_expected_shots",
    ]:
        rows[f"{col}_pct"] = within[col].transform(percentile)
    rows["keeper_saves_pressure_score"] = (
        0.42 * rows["keeper_save_pressure_score_pct"].fillna(0)
        + 0.24 * rows["team_expected_sot_pct"].fillna(0)
        + 0.14 * rows["team_sot_for_l5_pct"].fillna(0)
        + 0.10 * rows["opponent_sot_allowed_l5_pct"].fillna(0)
        + 0.06 * rows["team_high_sot_volume_rate_l5_pct"].fillna(0)
        + 0.04 * rows["team_expected_shots_pct"].fillna(0)
    ).clip(0.0, 1.0)
    for market, threshold in MARKETS.items():
        rows[f"actual_{market.lower().replace(' ', '_').replace('+', 'plus').replace('.', '_')}"] = (
            rows["actual_keeper_saves"].ge(threshold).astype(int)
        )
    return rows


def target_col(market: str) -> str:
    return f"actual_{market.lower().replace(' ', '_').replace('+', 'plus').replace('.', '_')}"


def evaluate(df: pd.DataFrame, market: str, label: str, mask: pd.Series) -> dict[str, Any]:
    target = target_col(market)
    cell = df[mask.fillna(False)].copy()
    graded = cell[cell["actual_keeper_saves"].notna()].copy()
    baseline_frame = df[df["actual_keeper_saves"].notna()].copy()
    hit_rate = float(graded[target].mean()) if len(graded) else np.nan
    baseline = float(baseline_frame[target].mean()) if len(baseline_frame) else np.nan
    months = []
    if len(graded):
        month_values = pd.to_datetime(graded["match_date"], errors="coerce").dt.to_period("M").astype(str)
        for _, group in graded.assign(_month=month_values).groupby("_month", dropna=False):
            if len(group) < 5:
                continue
            months.append(float(group[target].mean()))
    stable = float(np.mean([rate >= baseline for rate in months])) if months and pd.notna(baseline) else np.nan
    return {
        "market_display": market,
        "cell_label": label,
        "rows": int(len(cell)),
        "graded_rows": int(len(graded)),
        "fixtures": int(cell["fixture_key"].nunique()) if "fixture_key" in cell.columns else int(len(cell)),
        "keepers": int(cell["keeper_name"].nunique()) if "keeper_name" in cell.columns else 0,
        "hit_rate": hit_rate,
        "baseline_hit_rate": baseline,
        "lift_vs_baseline": hit_rate - baseline if pd.notna(hit_rate) and pd.notna(baseline) else np.nan,
        "stable_month_share_vs_baseline": stable,
        "months_with_min5": int(len(months)),
        "avg_score": float(cell["keeper_saves_pressure_score"].mean()) if len(cell) else np.nan,
        "avg_expected_sot_faced": float(cell["team_expected_sot"].mean()) if len(cell) else np.nan,
    }


def beta_label(row: pd.Series) -> str:
    rows = int(row.get("graded_rows", 0) or 0)
    lift = row.get("lift_vs_baseline", np.nan)
    stable = 0.0 if pd.isna(row.get("stable_month_share_vs_baseline", np.nan)) else float(row["stable_month_share_vs_baseline"])
    if rows >= 1200 and pd.notna(lift) and lift >= 0.10 and stable >= 0.85:
        return "CORE_WATCH"
    if rows >= 500 and pd.notna(lift) and lift >= 0.07 and stable >= 0.75:
        return "WATCH"
    if rows >= 150 and pd.notna(lift) and lift >= 0.035:
        return "RESEARCH_ONLY"
    return "DO_NOT_USE"


def build_cells(scored: pd.DataFrame) -> pd.DataFrame:
    work = scored[scored["profile_ready"] & scored["actual_keeper_saves"].notna()].copy()
    if work.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for market in MARKETS:
        for threshold in SCORE_THRESHOLDS:
            rows.append(evaluate(work, market, f"SCORE_GE_{threshold:.2f}", work["keeper_saves_pressure_score"].ge(threshold)))
        ranked = work.sort_values(["fixture_key", "keeper_saves_pressure_score"], ascending=[True, False]).copy()
        ranked["_fixture_rank"] = ranked.groupby("fixture_key").cumcount() + 1
        for top_n in TOP_N_PER_FIXTURE:
            mask = pd.Series(work.index.isin(ranked[ranked["_fixture_rank"].le(top_n)].index), index=work.index)
            rows.append(evaluate(work, market, f"TOP_{top_n}_KEEPER_PRESSURE_PER_FIXTURE", mask))
    cells = pd.DataFrame(rows)
    if cells.empty:
        return cells
    cells["recommended_beta_label"] = cells.apply(beta_label, axis=1)
    return cells.sort_values(
        ["recommended_beta_label", "lift_vs_baseline", "hit_rate", "graded_rows"],
        key=lambda series: series.map({"CORE_WATCH": 0, "WATCH": 1, "RESEARCH_ONLY": 2, "DO_NOT_USE": 3}).fillna(series)
        if series.name == "recommended_beta_label"
        else series,
        ascending=[True, False, False, False],
    ).reset_index(drop=True)


def breakdown(df: pd.DataFrame, by: list[str], target: str) -> pd.DataFrame:
    work = df[df["actual_keeper_saves"].notna()].copy()
    if work.empty:
        return pd.DataFrame()
    return (
        work.groupby(by, dropna=False)
        .agg(
            rows=("fixture_key", "count"),
            fixtures=("fixture_key", "nunique"),
            keepers=("keeper_name", "nunique"),
            hit_rate=(target, "mean"),
            avg_score=("keeper_saves_pressure_score", "mean"),
            avg_expected_sot_faced=("team_expected_sot", "mean"),
        )
        .reset_index()
        .sort_values(["hit_rate", "rows"], ascending=[False, False])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-profile", type=Path, default=DEFAULT_TEAM_PROFILE)
    parser.add_argument("--leagues", help="Comma-separated league tags.")
    parser.add_argument("--seasons", help="Comma-separated seasons.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    leagues = parse_csv_arg(args.leagues)
    seasons = parse_int_arg(args.seasons)
    profiles = load_profiles(args.team_profile, leagues, seasons)
    keepers = load_goalkeepers(leagues, seasons)
    scored = add_scores(profiles, keepers)
    cells = build_cells(scored)
    league_15 = breakdown(scored, ["league_tag"], target_col("Keeper Saves 1.5+"))
    league_25 = breakdown(scored, ["league_tag"], target_col("Keeper Saves 2.5+"))
    league_35 = breakdown(scored, ["league_tag"], target_col("Keeper Saves 3.5+"))

    scored.to_csv(args.outdir / "keeper_saves_intelligence_scored_rows.csv", index=False)
    cells.to_csv(args.outdir / "keeper_saves_intelligence_threshold_cells.csv", index=False)
    league_15.to_csv(args.outdir / "keeper_saves_15_league_breakdown.csv", index=False)
    league_25.to_csv(args.outdir / "keeper_saves_25_league_breakdown.csv", index=False)
    league_35.to_csv(args.outdir / "keeper_saves_35_league_breakdown.csv", index=False)

    counts = cells["recommended_beta_label"].value_counts().to_dict() if not cells.empty else {}
    report_cols = [
        "market_display",
        "cell_label",
        "recommended_beta_label",
        "graded_rows",
        "fixtures",
        "hit_rate",
        "baseline_hit_rate",
        "lift_vs_baseline",
        "stable_month_share_vs_baseline",
        "avg_expected_sot_faced",
    ]
    lines = [
        "# Keeper Saves Intelligence Audit",
        "",
        "Research-only audit for keeper saves from opponent SOT pressure. No deploy routing, priced odds, or slips are changed.",
        "",
        "## Summary",
        f"- profile_rows: {len(profiles)}",
        f"- keeper_actual_rows: {len(keepers)}",
        f"- scored_rows: {len(scored)}",
        f"- scored_with_keeper_actual: {int(scored['actual_keeper_saves'].notna().sum())}",
        f"- candidate_cell_counts: {counts}",
        "",
        "## Threshold Cells",
        markdown_table(cells[report_cols] if not cells.empty else cells),
        "",
        "## Keeper Saves 1.5+ League Breakdown",
        markdown_table(league_15),
        "",
        "## Keeper Saves 2.5+ League Breakdown",
        markdown_table(league_25),
        "",
        "## Keeper Saves 3.5+ League Breakdown",
        markdown_table(league_35),
        "",
        "## Recommendation",
        "- Use Keeper Saves Intelligence as a named/player context layer only after live shadow repeats.",
        "- 1.5+ should be the first dashboard watch candidate; 2.5+ and 3.5+ need stricter SOT pressure cells.",
        "- Keep this tied to team SOT pressure and expected keeper starter coverage.",
    ]
    (args.outdir / "KEEPER_SAVES_INTELLIGENCE_AUDIT.md").write_text("\n".join(lines) + "\n")
    print(f"WROTE: {args.outdir}")
    print(f"scored={len(scored)} with_keeper_actual={int(scored['actual_keeper_saves'].notna().sum())} cells={len(cells)} counts={counts}")
    if not cells.empty:
        print(cells[report_cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
