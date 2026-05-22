#!/usr/bin/env python3
"""Build a research-only corners intelligence audit.

The audit uses leak-safe team shots/corner profiles plus fixture context
(wide pressure, corner pressure, territorial stress) to test whether team and
match corner watch cells are stable enough for dashboard intelligence.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT / "data_sources" / "api_football" / "features" / "player_events"
DEFAULT_TEAM_PROFILE = ROOT / "reports" / "2026-05-06" / "team_shots_profile" / "TEAM_SHOTS_PROFILE_ROWS.csv"
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-07" / "corners_intelligence_audit"

TEAM_THRESHOLDS = {
    "Team Corners 4.5+": 5,
    "Team Corners 5.5+": 6,
}
MATCH_THRESHOLDS = {
    "Match Corners 8.5+": 9,
    "Match Corners 9.5+": 10,
}
SCORE_THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80]
TOP_N_PER_FIXTURE = [1, 2]

CONTEXT_COLS = [
    "fixture_corner_pressure_score",
    "fixture_attack_pressure_score",
    "fixture_territorial_stress_score",
    "fixture_wide_duel_score",
    "formation_wide_overload_flag",
    "formation_left_wide_overload_score",
    "formation_right_wide_overload_score",
    "lineup_xi_shot_power_delta",
    "lineup_formation_attack_delta",
    "h2h_total_corners_l5",
]


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


def percentile(series: pd.Series) -> pd.Series:
    values = num(series)
    if values.notna().sum() <= 1:
        return pd.Series(0.0, index=series.index)
    return values.rank(pct=True).fillna(0.0)


def parse_feature_input_tag(path: Path) -> tuple[str, int] | None:
    match = re.match(r"player_events_fixture_input__(.+)__(\d{4})\.csv$", path.name)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def load_fixture_context(leagues: set[str] | None, seasons: set[int] | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(FEATURE_DIR.glob("player_events_fixture_input__*.csv")):
        parsed = parse_feature_input_tag(path)
        if parsed is None:
            continue
        league_tag, season_tag = parsed
        if leagues and league_tag not in leagues:
            continue
        if seasons and season_tag not in seasons:
            continue
        header = pd.read_csv(path, nrows=0)
        usecols = [col for col in ["fixture_key", "league", "match_date"] + CONTEXT_COLS if col in header.columns]
        if "fixture_key" not in usecols:
            continue
        frame = pd.read_csv(path, usecols=usecols, low_memory=False)
        frame["league_tag"] = league_tag
        frame["season_tag"] = season_tag
        for col in CONTEXT_COLS:
            if col not in frame.columns:
                frame[col] = np.nan
            frame[col] = num(frame[col])
        grouped = (
            frame.groupby(["fixture_key", "league_tag", "season_tag"], dropna=False)
            .agg(
                league=("league", "first") if "league" in frame.columns else ("fixture_key", "size"),
                match_date=("match_date", "first") if "match_date" in frame.columns else ("fixture_key", "first"),
                fixture_corner_pressure_score=("fixture_corner_pressure_score", "max"),
                fixture_attack_pressure_score=("fixture_attack_pressure_score", "max"),
                fixture_territorial_stress_score=("fixture_territorial_stress_score", "max"),
                fixture_wide_duel_score=("fixture_wide_duel_score", "max"),
                formation_wide_overload_flag=("formation_wide_overload_flag", "max"),
                formation_left_wide_overload_score=("formation_left_wide_overload_score", "max"),
                formation_right_wide_overload_score=("formation_right_wide_overload_score", "max"),
                lineup_xi_shot_power_delta=("lineup_xi_shot_power_delta", "max"),
                lineup_formation_attack_delta=("lineup_formation_attack_delta", "max"),
                h2h_total_corners_l5=("h2h_total_corners_l5", "max"),
            )
            .reset_index()
        )
        frames.append(grouped)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def load_profiles(path: Path, leagues: set[str] | None, seasons: set[int] | None) -> pd.DataFrame:
    profiles = pd.read_csv(path, low_memory=False)
    if leagues:
        profiles = profiles[profiles["league_tag"].astype(str).isin(leagues)].copy()
    if seasons:
        profiles = profiles[profiles["season_tag"].astype(int).isin(seasons)].copy()
    return profiles


def add_scores(profiles: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    rows = profiles.copy()
    if not context.empty:
        rows = rows.merge(
            context.drop(columns=["league", "match_date"], errors="ignore"),
            on=["fixture_key", "league_tag", "season_tag"],
            how="left",
        )
    for col in [
        "team_expected_corners",
        "team_expected_corners_league_pct",
        "team_corner_5plus_rate_l5",
        "team_corners_for_l5",
        "team_corners_for_l10",
        "opponent_corners_allowed_l5",
        "opponent_corners_allowed_l10",
        "team_expected_shots",
        "team_expected_shots_league_pct",
        "team_expected_sot",
        "team_expected_sot_league_pct",
        "actual_corners_for",
    ] + CONTEXT_COLS:
        if col not in rows.columns:
            rows[col] = np.nan
        rows[col] = num(rows[col])

    within = rows.groupby("league_tag", dropna=False)
    for col in [
        "team_expected_corners",
        "team_corner_5plus_rate_l5",
        "opponent_corners_allowed_l5",
        "team_expected_shots",
        "team_expected_sot",
        "h2h_total_corners_l5",
    ]:
        rows[f"{col}_pct"] = within[col].transform(percentile)
    if rows["team_expected_corners_league_pct"].isna().all():
        rows["team_expected_corners_league_pct"] = rows["team_expected_corners_pct"]
    if rows["team_expected_shots_league_pct"].isna().all():
        rows["team_expected_shots_league_pct"] = rows["team_expected_shots_pct"]
    if rows["team_expected_sot_league_pct"].isna().all():
        rows["team_expected_sot_league_pct"] = rows["team_expected_sot_pct"]

    rows["wide_pressure_score"] = np.nanmax(
        rows[["fixture_wide_duel_score", "formation_left_wide_overload_score", "formation_right_wide_overload_score"]].fillna(0).to_numpy(),
        axis=1,
    )
    rows["territory_corner_context_score"] = (
        0.28 * rows["fixture_corner_pressure_score"].fillna(0)
        + 0.18 * rows["fixture_attack_pressure_score"].fillna(0)
        + 0.18 * rows["fixture_territorial_stress_score"].fillna(0)
        + 0.16 * rows["wide_pressure_score"].fillna(0)
        + 0.10 * rows["formation_wide_overload_flag"].fillna(0).clip(0, 1)
        + 0.10 * rows["h2h_total_corners_l5_pct"].fillna(0)
    ).clip(0.0, 1.0)
    rows["team_corner_pressure_score"] = (
        0.36 * rows["team_expected_corners_league_pct"].fillna(rows["team_expected_corners_pct"]).fillna(0)
        + 0.22 * rows["team_corner_5plus_rate_l5_pct"].fillna(0)
        + 0.18 * rows["opponent_corners_allowed_l5_pct"].fillna(0)
        + 0.10 * rows["team_expected_shots_league_pct"].fillna(rows["team_expected_shots_pct"]).fillna(0)
        + 0.06 * rows["team_expected_sot_league_pct"].fillna(rows["team_expected_sot_pct"]).fillna(0)
        + 0.08 * rows["territory_corner_context_score"].fillna(0)
    ).clip(0.0, 1.0)
    rows["actual_team_corners_4_5"] = rows["actual_corners_for"].ge(5).astype(int)
    rows["actual_team_corners_5_5"] = rows["actual_corners_for"].ge(6).astype(int)
    rows["profile_ready"] = rows.get("profile_ready", pd.Series(False, index=rows.index)).fillna(False).astype(bool)

    fixture = fixture_view(rows)
    rows = rows.merge(
        fixture[
            [
                "fixture_key",
                "match_corner_pressure_score",
                "match_expected_corners",
                "actual_match_corners",
                "actual_match_corners_8_5",
                "actual_match_corners_9_5",
            ]
        ],
        on="fixture_key",
        how="left",
    )
    return rows


def fixture_view(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        rows.groupby(["fixture_key", "league_tag", "season_tag"], dropna=False)
        .agg(
            league=("league", "first"),
            match_date=("match_date", "first"),
            home_team_name=("team_name", lambda s: s.iloc[0] if len(s) else ""),
            teams=("team_name", lambda s: " vs ".join(s.dropna().astype(str).tolist()[:2])),
            match_expected_corners=("team_expected_corners", "sum"),
            actual_match_corners=("actual_corners_for", "sum"),
            avg_team_corner_pressure_score=("team_corner_pressure_score", "mean"),
            max_team_corner_pressure_score=("team_corner_pressure_score", "max"),
            fixture_corner_pressure_score=("fixture_corner_pressure_score", "max"),
            fixture_attack_pressure_score=("fixture_attack_pressure_score", "max"),
            fixture_territorial_stress_score=("fixture_territorial_stress_score", "max"),
            fixture_wide_duel_score=("fixture_wide_duel_score", "max"),
            profile_ready=("profile_ready", "min"),
        )
        .reset_index()
    )
    for col in ["match_expected_corners"]:
        grouped[f"{col}_pct"] = grouped.groupby("league_tag")[col].transform(percentile)
    grouped["match_corner_pressure_score"] = (
        0.42 * grouped["match_expected_corners_pct"].fillna(0)
        + 0.22 * grouped["avg_team_corner_pressure_score"].fillna(0)
        + 0.16 * grouped["max_team_corner_pressure_score"].fillna(0)
        + 0.08 * grouped["fixture_corner_pressure_score"].fillna(0)
        + 0.06 * grouped["fixture_territorial_stress_score"].fillna(0)
        + 0.06 * grouped["fixture_wide_duel_score"].fillna(0)
    ).clip(0.0, 1.0)
    grouped["actual_match_corners_8_5"] = grouped["actual_match_corners"].ge(9).astype(int)
    grouped["actual_match_corners_9_5"] = grouped["actual_match_corners"].ge(10).astype(int)
    return grouped


def evaluate(df: pd.DataFrame, label: str, mask: pd.Series, target_col: str, market: str) -> dict[str, Any]:
    cell = df[mask.fillna(False)].copy()
    graded = cell[cell[target_col].notna()].copy()
    baseline_frame = df[df[target_col].notna()].copy()
    hit_rate = float(graded[target_col].mean()) if len(graded) else np.nan
    baseline = float(baseline_frame[target_col].mean()) if len(baseline_frame) else np.nan
    month_rates = []
    if len(graded) and "match_date" in graded.columns:
        months = pd.to_datetime(graded["match_date"], errors="coerce").dt.to_period("M").astype(str)
        for _, group in graded.assign(_month=months).groupby("_month", dropna=False):
            if len(group) < 5:
                continue
            month_rates.append(float(group[target_col].mean()))
    stable = float(np.mean([rate >= baseline for rate in month_rates])) if month_rates and pd.notna(baseline) else np.nan
    return {
        "market_display": market,
        "cell_label": label,
        "rows": int(len(cell)),
        "graded_rows": int(len(graded)),
        "fixtures": int(cell["fixture_key"].nunique()) if "fixture_key" in cell.columns else int(len(cell)),
        "hit_rate": hit_rate,
        "baseline_hit_rate": baseline,
        "lift_vs_baseline": hit_rate - baseline if pd.notna(hit_rate) and pd.notna(baseline) else np.nan,
        "stable_month_share_vs_baseline": stable,
        "months_with_min5": int(len(month_rates)),
        "avg_score": float(cell["team_corner_pressure_score"].mean()) if "team_corner_pressure_score" in cell.columns and len(cell) else float(cell["match_corner_pressure_score"].mean()) if "match_corner_pressure_score" in cell.columns and len(cell) else np.nan,
    }


def beta_label(row: pd.Series) -> str:
    rows = int(row.get("graded_rows", 0) or 0)
    hit = row.get("hit_rate", np.nan)
    lift = row.get("lift_vs_baseline", np.nan)
    stable = 0.0 if pd.isna(row.get("stable_month_share_vs_baseline", np.nan)) else float(row["stable_month_share_vs_baseline"])
    if rows >= 1200 and pd.notna(hit) and pd.notna(lift) and lift >= 0.10 and stable >= 0.85:
        return "CORE_WATCH"
    if rows >= 500 and pd.notna(hit) and pd.notna(lift) and lift >= 0.07 and stable >= 0.75:
        return "WATCH"
    if rows >= 150 and pd.notna(hit) and pd.notna(lift) and lift >= 0.035:
        return "RESEARCH_ONLY"
    return "DO_NOT_USE"


def build_cells(rows: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    out: list[dict[str, Any]] = []
    team = rows[rows["profile_ready"]].copy()
    for market, actual_min in TEAM_THRESHOLDS.items():
        target = f"actual_team_corners_{actual_min - 0.5:.1f}".replace(".", "_")
        if target not in team.columns:
            continue
        for threshold in SCORE_THRESHOLDS:
            out.append(evaluate(team, f"TEAM_SCORE_GE_{threshold:.2f}", team["team_corner_pressure_score"].ge(threshold), target, market))
        ranked = team.sort_values(["fixture_key", "team_corner_pressure_score"], ascending=[True, False]).copy()
        ranked["_fixture_rank"] = ranked.groupby("fixture_key").cumcount() + 1
        for top_n in TOP_N_PER_FIXTURE:
            top_index = ranked[ranked["_fixture_rank"].le(top_n)].index
            mask = pd.Series(team.index.isin(top_index), index=team.index)
            out.append(evaluate(team, f"TOP_{top_n}_TEAM_PER_FIXTURE", mask, target, market))

    match_rows = fixtures[fixtures["profile_ready"].fillna(False)].copy()
    for market, actual_min in MATCH_THRESHOLDS.items():
        target = f"actual_match_corners_{actual_min - 0.5:.1f}".replace(".", "_")
        if target not in match_rows.columns:
            continue
        for threshold in SCORE_THRESHOLDS:
            out.append(
                evaluate(
                    match_rows,
                    f"MATCH_SCORE_GE_{threshold:.2f}",
                    match_rows["match_corner_pressure_score"].ge(threshold),
                    target,
                    market,
                )
            )
    cells = pd.DataFrame(out)
    if cells.empty:
        return cells
    cells["recommended_beta_label"] = cells.apply(beta_label, axis=1)
    cells = cells.sort_values(
        ["recommended_beta_label", "lift_vs_baseline", "hit_rate", "graded_rows"],
        key=lambda series: series.map({"CORE_WATCH": 0, "WATCH": 1, "RESEARCH_ONLY": 2, "DO_NOT_USE": 3}).fillna(series)
        if series.name == "recommended_beta_label"
        else series,
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    return cells


def breakdown(df: pd.DataFrame, by: list[str], target: str, score: str) -> pd.DataFrame:
    work = df[df[target].notna()].copy()
    if work.empty:
        return pd.DataFrame()
    return (
        work.groupby(by, dropna=False)
        .agg(
            rows=("fixture_key", "count"),
            fixtures=("fixture_key", "nunique"),
            hit_rate=(target, "mean"),
            avg_score=(score, "mean"),
            avg_expected_corners=("team_expected_corners", "mean") if "team_expected_corners" in work.columns else (score, "mean"),
        )
        .reset_index()
        .sort_values(["hit_rate", "rows"], ascending=[False, False])
    )


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
    context = load_fixture_context(leagues, seasons)
    rows = add_scores(profiles, context)
    fixtures = fixture_view(rows)
    cells = build_cells(rows, fixtures)
    league_team = breakdown(rows[rows["profile_ready"]], ["league_tag"], "actual_team_corners_4_5", "team_corner_pressure_score")
    league_match = breakdown(fixtures[fixtures["profile_ready"]], ["league_tag"], "actual_match_corners_8_5", "match_corner_pressure_score")

    rows.to_csv(args.outdir / "corners_intelligence_team_scored_rows.csv", index=False)
    fixtures.to_csv(args.outdir / "corners_intelligence_fixture_scored_rows.csv", index=False)
    cells.to_csv(args.outdir / "corners_intelligence_threshold_cells.csv", index=False)
    league_team.to_csv(args.outdir / "corners_intelligence_team_league_breakdown.csv", index=False)
    league_match.to_csv(args.outdir / "corners_intelligence_match_league_breakdown.csv", index=False)

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
        "avg_score",
    ]
    lines = [
        "# Corners Intelligence Audit",
        "",
        "Research-only audit for team and match corner pressure. No deploy routing, priced odds, or slips are changed.",
        "",
        "## Summary",
        f"- team_rows: {len(rows)}",
        f"- profile_ready_team_rows: {int(rows['profile_ready'].sum())}",
        f"- fixture_rows: {len(fixtures)}",
        f"- candidate_cell_counts: {counts}",
        "",
        "## Threshold Cells",
        markdown_table(cells[report_cols] if not cells.empty else cells),
        "",
        "## Team 4.5+ League Breakdown",
        markdown_table(league_team),
        "",
        "## Match 8.5+ League Breakdown",
        markdown_table(league_match),
        "",
        "## Recommendation",
        "- Use corners as an intelligence/context layer first: team pressure, match pressure, and support for player shots/SOT and keeper-save reads.",
        "- Promote only stable cells to dashboard watch labels; do not treat corners as a deploy market yet.",
        "- If watch cells survive live shadow repeats, build current-board `CORNERS_INTELLIGENCE_LIVE_SHADOW` rows next.",
    ]
    (args.outdir / "CORNERS_INTELLIGENCE_AUDIT.md").write_text("\n".join(lines) + "\n")
    print(f"WROTE: {args.outdir}")
    print(f"team_rows={len(rows)} fixtures={len(fixtures)} cells={len(cells)} counts={counts}")
    if not cells.empty:
        print(cells[report_cols].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
