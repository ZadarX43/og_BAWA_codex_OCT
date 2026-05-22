#!/usr/bin/env python3
"""Build a research-only Player Cards 0.5+ hazard audit.

This is intentionally isolated from production deploy. It joins historical
player-event fixture features to settled yellow-card outcomes, builds a simple
interpretable hazard score, and reports whether any role/context cells deserve
dashboard watch status.
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
NORMALIZED_DIR = ROOT / "data_sources" / "api_football" / "normalized"
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-07" / "player_cards_hazard_audit"

DEFAULT_LEAGUES = [
    "England_Premier_League",
    "Spain_La_Liga",
    "Germany_Bundesliga",
    "France_Ligue_1",
    "Italy_Serie_A",
    "Netherlands_Eredivisie",
    "Portugal_Liga",
    "Belgium_Pro",
    "England_Championship",
]
DEFAULT_SEASONS = [2022, 2023, 2024]

CARD_FEATURES = [
    "yellow_cards_per90",
    "fouls_per90",
    "tackles_per90",
    "dribbles_faced_per90",
    "ground_duel_loss_rate",
    "ref_cards_per_match",
    "fixture_foul_density_score",
    "fixture_tackle_density_score",
    "fixture_midfield_grind_score",
    "fixture_wide_duel_score",
    "formation_pressure_score",
    "match_stakes_score",
    "opponent_possession_projection",
    "lineup_xi_card_risk_delta",
]

THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75]
TOP_N_PER_FIXTURE = [1, 2, 3, 5]


def num(values: Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def slug_list(value: str | None, default: list[str]) -> list[str]:
    if value is None or not value.strip():
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_name(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def role_cell(row: pd.Series) -> str:
    text = f"{row.get('tactical_role', '')} {row.get('position_group', '')}".lower()
    if any(term in text for term in ["holding midfielder", "defensive midfielder", "central midfielder"]):
        return "HOLDING_OR_CENTRAL_MID"
    if any(term in text for term in ["wide defender", "wing-back", "full-back", "fullback"]):
        return "WIDE_DEFENDER_WING_BACK"
    if any(term in text for term in ["centre-back", "center-back", "central defender", "enforcer"]):
        return "CENTRE_BACK_STRICT"
    if any(term in text for term in ["forward", "winger", "striker", "attacker"]):
        return "ATTACKER_PRESSER"
    return "OTHER"


def percentile(series: pd.Series) -> pd.Series:
    values = num(series)
    if values.notna().sum() <= 1:
        return pd.Series(0.0, index=series.index)
    return values.rank(pct=True).fillna(0.0)


def safe_flag(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=float)
    text = series.fillna("").astype(str).str.lower().str.strip()
    return text.isin(["1", "true", "yes", "y"]).astype(float)


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def load_joined(league: str, season: int) -> pd.DataFrame:
    feature_path = FEATURE_DIR / f"player_events_fixture_input__{league}__{season}.csv"
    fixtures_path = NORMALIZED_DIR / f"fixtures_master__{league}__{season}.csv"
    stats_path = NORMALIZED_DIR / f"match_player_stats__{league}__{season}.csv"
    if not feature_path.exists() or not fixtures_path.exists() or not stats_path.exists():
        return pd.DataFrame()

    features = pd.read_csv(feature_path, low_memory=False)
    fixtures = pd.read_csv(fixtures_path, low_memory=False, usecols=["fixture_id", "fixture_key", "match_date", "league"])
    stats = pd.read_csv(
        stats_path,
        low_memory=False,
        usecols=[
            "fixture_id",
            "player_name",
            "minutes",
            "started_flag",
            "fouls_committed",
            "tackles",
            "yellow_cards",
            "red_cards",
        ],
    )
    stats = stats.merge(fixtures, on="fixture_id", how="left")
    stats["_player_key"] = stats["player_name"].map(normalize_name)
    stats = stats.sort_values(["fixture_key", "_player_key", "minutes"], ascending=[True, True, False])
    stats = stats.drop_duplicates(["fixture_key", "_player_key"], keep="first")

    features["_player_key"] = features["player_name"].map(normalize_name)
    joined = features.merge(
        stats[
            [
                "fixture_key",
                "_player_key",
                "minutes",
                "started_flag",
                "fouls_committed",
                "tackles",
                "yellow_cards",
                "red_cards",
            ]
        ],
        on=["fixture_key", "_player_key"],
        how="left",
        suffixes=("", "_actual"),
    )
    joined["source_league_slug"] = league
    joined["source_season"] = season
    joined["actual_minutes"] = num(joined.get("minutes", np.nan))
    joined["actual_yellow_cards"] = num(joined.get("yellow_cards", np.nan)).fillna(0).astype(int)
    joined["actual_red_cards"] = num(joined.get("red_cards", np.nan)).fillna(0).astype(int)
    joined["actual_card_ge1"] = joined["actual_yellow_cards"].ge(1).astype(int)
    return joined


def add_hazard_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in CARD_FEATURES + ["expected_minutes", "support_score"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = num(out[col])

    out["role_cell"] = out.apply(role_cell, axis=1)
    out["eligible_pre_match"] = (
        out["expected_minutes"].fillna(0).ge(55)
        & out.get("expected_start_flag", pd.Series(1, index=out.index)).fillna(1).astype(str).str.lower().isin(["1", "true", "yes", "y"])
    )

    within = out.groupby(["source_league_slug", "source_season"], dropna=False)
    for col in CARD_FEATURES:
        out[f"{col}_pct"] = within[col].transform(percentile)

    out["role_risk_score"] = out["role_cell"].map(
        {
            "HOLDING_OR_CENTRAL_MID": 0.95,
            "WIDE_DEFENDER_WING_BACK": 0.88,
            "CENTRE_BACK_STRICT": 0.74,
            "ATTACKER_PRESSER": 0.42,
            "OTHER": 0.30,
        }
    ).fillna(0.30)
    out["player_contact_score"] = (
        0.32 * out["yellow_cards_per90_pct"]
        + 0.26 * out["fouls_per90_pct"]
        + 0.18 * out["tackles_per90_pct"]
        + 0.14 * out["dribbles_faced_per90_pct"]
        + 0.10 * out["ground_duel_loss_rate_pct"]
    )
    out["fixture_contact_score"] = (
        0.26 * out["ref_cards_per_match_pct"]
        + 0.21 * out["fixture_foul_density_score_pct"]
        + 0.18 * out["fixture_tackle_density_score_pct"]
        + 0.15 * out["fixture_midfield_grind_score_pct"]
        + 0.12 * out["fixture_wide_duel_score_pct"]
        + 0.08 * out["formation_pressure_score_pct"]
    )
    out["context_hazard_score"] = (
        0.24 * out["match_stakes_score_pct"]
        + 0.18 * out["opponent_possession_projection_pct"]
        + 0.18 * out["lineup_xi_card_risk_delta_pct"]
        + 0.20 * safe_flag(out.get("rivalry_flag", pd.Series("", index=out.index)))
        + 0.12 * safe_flag(out.get("temperament_flag", pd.Series("", index=out.index)))
        + 0.08 * safe_flag(out.get("suspension_risk_flag", pd.Series("", index=out.index)))
    ).clip(0.0, 1.0)
    out["cards_hazard_score"] = (
        0.30 * out["role_risk_score"]
        + 0.33 * out["player_contact_score"]
        + 0.27 * out["fixture_contact_score"]
        + 0.10 * out["context_hazard_score"]
    ).clip(0.0, 1.0)

    out["cards_hazard_label"] = np.select(
        [
            out["cards_hazard_score"].ge(0.75),
            out["cards_hazard_score"].ge(0.68),
            out["cards_hazard_score"].ge(0.60),
        ],
        ["CARDS_HAZARD_CORE", "CARDS_HAZARD_STRONG", "CARDS_HAZARD_WATCH"],
        default="CARDS_HAZARD_LOW",
    )
    out["month"] = pd.to_datetime(out.get("match_date", pd.Series("", index=out.index)), errors="coerce").dt.to_period("M").astype(str)
    return out


def evaluate_cell(df: pd.DataFrame, label: str, mask: pd.Series) -> dict[str, Any]:
    cell = df[mask].copy()
    graded = cell[cell["actual_minutes"].fillna(0).gt(0)]
    rows = len(cell)
    graded_rows = len(graded)
    hit_rate = float(graded["actual_card_ge1"].mean()) if graded_rows else np.nan
    baseline = float(df[df["actual_minutes"].fillna(0).gt(0)]["actual_card_ge1"].mean()) if len(df) else np.nan
    month_rows = []
    if graded_rows:
        for month, group in graded.groupby("month", dropna=False):
            if len(group) < 5:
                continue
            month_rows.append(float(group["actual_card_ge1"].mean()))
    stable_month_share = (
        float(np.mean([rate >= baseline for rate in month_rows])) if month_rows and pd.notna(baseline) else np.nan
    )
    return {
        "cell_label": label,
        "rows": int(rows),
        "graded_rows": int(graded_rows),
        "fixtures": int(cell["fixture_key"].nunique()) if "fixture_key" in cell.columns else 0,
        "hit_rate": hit_rate,
        "baseline_hit_rate": baseline,
        "lift_vs_baseline": hit_rate - baseline if pd.notna(hit_rate) and pd.notna(baseline) else np.nan,
        "stable_month_share_vs_baseline": stable_month_share,
        "months_with_min5": int(len(month_rows)),
        "avg_hazard_score": float(cell["cards_hazard_score"].mean()) if rows else np.nan,
    }


def beta_label(row: pd.Series) -> str:
    rows = int(row.get("graded_rows", 0) or 0)
    hit = row.get("hit_rate", np.nan)
    lift = row.get("lift_vs_baseline", np.nan)
    stable = row.get("stable_month_share_vs_baseline", np.nan)
    stable_value = 0.0 if pd.isna(stable) else float(stable)
    if rows >= 300 and pd.notna(hit) and hit >= 0.36 and pd.notna(lift) and lift >= 0.08 and stable_value >= 0.70:
        return "CORE_WATCH"
    if rows >= 150 and pd.notna(hit) and hit >= 0.30 and pd.notna(lift) and lift >= 0.05 and stable_value >= 0.60:
        return "WATCH"
    if rows >= 50 and pd.notna(hit) and hit >= 0.22 and pd.notna(lift) and lift >= 0.02:
        return "RESEARCH_ONLY"
    return "DO_NOT_USE"


def build_cells(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    eligible = scored[scored["eligible_pre_match"]].copy()
    if eligible.empty:
        return pd.DataFrame()

    for threshold in THRESHOLDS:
        rows.append(evaluate_cell(eligible, f"SCORE_GE_{threshold:.2f}", eligible["cards_hazard_score"].ge(threshold)))

    ranked = eligible.sort_values(["fixture_key", "cards_hazard_score"], ascending=[True, False]).copy()
    ranked["_fixture_rank"] = ranked.groupby("fixture_key").cumcount() + 1
    for top_n in TOP_N_PER_FIXTURE:
        top_index = ranked[ranked["_fixture_rank"].le(top_n)].index
        top_mask = pd.Series(eligible.index.isin(top_index), index=eligible.index)
        rows.append(evaluate_cell(eligible, f"TOP_{top_n}_PER_FIXTURE", top_mask))

    for role, group in eligible.groupby("role_cell", dropna=False):
        if len(group) < 20:
            continue
        for threshold in [0.60, 0.68, 0.75]:
            rows.append(evaluate_cell(group, f"{role}_SCORE_GE_{threshold:.2f}", group["cards_hazard_score"].ge(threshold)))

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["recommended_beta_label"] = out.apply(beta_label, axis=1)
    out = out.sort_values(
        ["recommended_beta_label", "lift_vs_baseline", "hit_rate", "graded_rows"],
        key=lambda series: series.map({"CORE_WATCH": 0, "WATCH": 1, "RESEARCH_ONLY": 2, "DO_NOT_USE": 3}).fillna(series)
        if series.name == "recommended_beta_label"
        else series,
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    return out


def breakdown(scored: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    eligible = scored[scored["eligible_pre_match"] & scored["actual_minutes"].fillna(0).gt(0)].copy()
    if eligible.empty:
        return pd.DataFrame()
    grouped = eligible.groupby(by, dropna=False)
    out = grouped.agg(
        rows=("player_name", "count"),
        fixtures=("fixture_key", "nunique"),
        hit_rate=("actual_card_ge1", "mean"),
        avg_hazard_score=("cards_hazard_score", "mean"),
        avg_player_contact_score=("player_contact_score", "mean"),
        avg_fixture_contact_score=("fixture_contact_score", "mean"),
    ).reset_index()
    return out.sort_values(["hit_rate", "rows"], ascending=[False, False])


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
    parser.add_argument("--leagues", default=",".join(DEFAULT_LEAGUES), help="Comma-separated league slugs.")
    parser.add_argument("--seasons", default=",".join(str(season) for season in DEFAULT_SEASONS), help="Comma-separated seasons.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--max-target-rows", type=int, default=0, help="Optional cap for smoke runs.")
    args = parser.parse_args()

    leagues = slug_list(args.leagues, DEFAULT_LEAGUES)
    seasons = [int(season) for season in slug_list(args.seasons, [str(season) for season in DEFAULT_SEASONS])]
    loaded: list[pd.DataFrame] = []
    load_rows: list[dict[str, Any]] = []
    for league in leagues:
        for season in seasons:
            frame = load_joined(league, season)
            load_rows.append({"league_slug": league, "season": season, "rows": len(frame), "loaded": not frame.empty})
            if not frame.empty:
                loaded.append(frame)

    args.outdir.mkdir(parents=True, exist_ok=True)
    load_audit = pd.DataFrame(load_rows)
    load_audit.to_csv(args.outdir / "player_cards_hazard_load_audit.csv", index=False)
    if not loaded:
        (args.outdir / "PLAYER_CARDS_HAZARD_AUDIT.md").write_text("# Player Cards Hazard Audit\n\nNo input rows loaded.\n")
        print(f"WROTE: {args.outdir}")
        print("No input rows loaded.")
        return

    raw = pd.concat(loaded, ignore_index=True, sort=False)
    if args.max_target_rows and len(raw) > args.max_target_rows:
        raw = raw.sort_values(["source_league_slug", "source_season", "fixture_key", "player_name"]).head(args.max_target_rows).copy()
    scored = add_hazard_features(raw)
    cells = build_cells(scored)
    role_breakdown = breakdown(scored, ["role_cell"])
    league_breakdown = breakdown(scored, ["source_league_slug"])
    league_role_breakdown = breakdown(scored, ["source_league_slug", "role_cell"])

    scored_out_cols = [
        "fixture_key",
        "match_date",
        "source_league_slug",
        "source_season",
        "league",
        "home_team_name",
        "away_team_name",
        "team_name",
        "player_name",
        "position_group",
        "tactical_role",
        "role_cell",
        "expected_minutes",
        "actual_minutes",
        "actual_card_ge1",
        "cards_hazard_score",
        "cards_hazard_label",
        "player_contact_score",
        "fixture_contact_score",
        "context_hazard_score",
        "referee_name",
        "ref_cards_per_match",
        "fixture_foul_density_score",
        "fixture_tackle_density_score",
        "fixture_midfield_grind_score",
        "fixture_wide_duel_score",
        "yellow_cards_per90",
        "fouls_per90",
        "tackles_per90",
        "temperament_flag",
        "suspension_risk_flag",
        "rivalry_flag",
    ]
    for col in scored_out_cols:
        if col not in scored.columns:
            scored[col] = np.nan
    scored[scored_out_cols].to_csv(args.outdir / "player_cards_hazard_scored_rows.csv", index=False)
    cells.to_csv(args.outdir / "player_cards_hazard_threshold_cells.csv", index=False)
    role_breakdown.to_csv(args.outdir / "player_cards_hazard_role_breakdown.csv", index=False)
    league_breakdown.to_csv(args.outdir / "player_cards_hazard_league_breakdown.csv", index=False)
    league_role_breakdown.to_csv(args.outdir / "player_cards_hazard_league_role_breakdown.csv", index=False)

    recommended_counts = cells["recommended_beta_label"].value_counts().to_dict() if not cells.empty else {}
    lines = [
        "# Player Cards 0.5+ Hazard Audit",
        "",
        "Research-only. No deploy tiers, priced odds, or slips are changed.",
        "",
        "## Summary",
        f"- loaded_rows: {len(raw)}",
        f"- eligible_pre_match_rows: {int(scored['eligible_pre_match'].sum())}",
        f"- graded_eligible_rows: {int((scored['eligible_pre_match'] & scored['actual_minutes'].fillna(0).gt(0)).sum())}",
        f"- candidate_cell_counts: {recommended_counts}",
        "",
        "## Threshold Cells",
        markdown_table(
            cells[
                [
                    "cell_label",
                    "recommended_beta_label",
                    "graded_rows",
                    "fixtures",
                    "hit_rate",
                    "baseline_hit_rate",
                    "lift_vs_baseline",
                    "stable_month_share_vs_baseline",
                    "avg_hazard_score",
                ]
            ]
            if not cells.empty
            else cells
        ),
        "",
        "## Role Breakdown",
        markdown_table(role_breakdown),
        "",
        "## League Breakdown",
        markdown_table(league_breakdown),
        "",
        "## Recommendation",
        "- Keep Player Cards 0.5+ out of dashboard prominence unless cells reach at least WATCH with stable historical lift.",
        "- Treat cards as a separate hazard model: contact role + referee strictness + fixture foul ecosystem + temperament/stakes.",
        "- Use this audit to decide whether a tiny cards watch drawer is justified; do not clone tackles thresholds into cards.",
    ]
    (args.outdir / "PLAYER_CARDS_HAZARD_AUDIT.md").write_text("\n".join(lines) + "\n")

    print(f"WROTE: {args.outdir}")
    print(f"loaded_rows={len(raw)} eligible={int(scored['eligible_pre_match'].sum())} cells={len(cells)}")
    if not cells.empty:
        print(cells[["cell_label", "recommended_beta_label", "graded_rows", "hit_rate", "lift_vs_baseline"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
