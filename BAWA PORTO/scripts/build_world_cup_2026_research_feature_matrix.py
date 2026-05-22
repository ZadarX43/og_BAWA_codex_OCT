#!/usr/bin/env python3
"""Assemble the World Cup 2026 research feature matrix.

This joins the existing macro/player overlay, FootyStats qualifier aggregates,
and Fjelstul historical World Cup priors into one fixture-level sidecar. It is
research-only and does not alter canonical training or deploy routing.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd


DEFAULT_CONTEXT = Path(
    "data_sources/footystats_world_cup/qualification_context_2026_footystats_enriched/"
    "world_cup_2026_model_ready_with_qualification_context.csv"
)
DEFAULT_FJELSTUL = Path(
    "data_sources/footystats_world_cup/fjelstul_historical_priors/"
    "world_cup_2026_team_historical_prior_sidecar.csv"
)
DEFAULT_ADDITIONS = Path(
    "data_sources/footystats_world_cup/additions_context_2026/"
    "world_cup_additions_context_sidecar.csv"
)
DEFAULT_RECENT_HISTORY = Path(
    "data_sources/footystats_world_cup/recent_history_h2h_2026/"
    "world_cup_2026_fixture_recent_history_sidecar.csv"
)
DEFAULT_LOCAL_H2H = Path(
    "data_sources/footystats_world_cup/recent_history_h2h_2026/"
    "world_cup_2026_local_h2h_sidecar.csv"
)
DEFAULT_VENUE_TRAVEL = Path(
    "data_sources/footystats_world_cup/venue_weather_travel_2026/"
    "world_cup_2026_fixture_venue_travel_weather_scaffold.csv"
)
DEFAULT_OUTDIR = Path("data_sources/footystats_world_cup/research_feature_matrix_2026")

TEAM_SLUG_ALIASES = {
    "cape_verde": "cape_verde",
    "cape_verde_islands": "cape_verde",
    "congo_dr": "dr_congo",
    "dr_congo": "dr_congo",
    "curacao": "curacao",
    "cura_ao": "curacao",
    "czechia": "czech_republic",
    "korea_republic": "south_korea",
    "south_korea": "south_korea",
    "cote_d_ivoire": "ivory_coast",
    "turkey": "turkiye",
    "turkiye": "turkiye",
    "t_rkiye": "turkiye",
    "united_states": "usa",
    "usa": "usa",
}

HISTORICAL_DELTA_COLS = [
    "all_wc_win_rate",
    "all_wc_draw_rate",
    "all_wc_goals_for_per_match",
    "all_wc_goals_against_per_match",
    "all_wc_goal_diff_per_match",
    "modern_wc_win_rate",
    "modern_wc_draw_rate",
    "modern_wc_goals_for_per_match",
    "modern_wc_goals_against_per_match",
    "modern_wc_goal_diff_per_match",
    "recent_wc_win_rate",
    "recent_wc_draw_rate",
    "recent_wc_goals_for_per_match",
    "recent_wc_goals_against_per_match",
    "recent_wc_goal_diff_per_match",
]

ADDITIONS_DELTA_COLS = [
    "additions_team_matches_played",
    "additions_team_weighted_ppg",
    "additions_team_goals_for_per_match",
    "additions_team_goals_against_per_match",
    "additions_team_goal_diff_per_match",
    "additions_team_xg_for_avg",
    "additions_team_xg_against_avg",
    "additions_team_btts_pct",
    "additions_team_over25_pct",
    "additions_team_clean_sheet_pct",
    "additions_team_fts_pct",
    "additions_player_rated_players",
    "additions_player_top11_avg_rating",
    "additions_player_top5_avg_rating",
    "additions_player_squad_market_value_proxy",
    "additions_player_goals",
    "additions_player_assists",
    "additions_player_xg",
    "additions_player_xa",
    "additions_match_goals_for_per_match",
    "additions_match_goals_against_per_match",
    "additions_match_shots_for_per_match",
    "additions_match_shots_against_per_match",
    "additions_match_sot_for_per_match",
    "additions_match_sot_against_per_match",
    "additions_match_cards_for_per_match",
    "additions_match_cards_against_per_match",
]


def slugify(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def canonical_team_slug(value: object) -> str:
    slug = slugify(value)
    return TEAM_SLUG_ALIASES.get(slug, slug)


def side_join(fixtures: pd.DataFrame, team_priors: pd.DataFrame, side: str) -> pd.DataFrame:
    out = fixtures.copy()
    name_col = f"api_{side}_team_name"
    out[f"{side}_team_slug"] = out[name_col].map(canonical_team_slug)
    side_priors = team_priors.rename(columns={c: f"{side}_{c}" for c in team_priors.columns if c != "team_slug"})
    side_priors = side_priors.rename(columns={"team_slug": f"{side}_team_slug"})
    return out.merge(side_priors, on=f"{side}_team_slug", how="left")


def add_historical_deltas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in HISTORICAL_DELTA_COLS:
        home = pd.to_numeric(out.get(f"home_{col}"), errors="coerce")
        away = pd.to_numeric(out.get(f"away_{col}"), errors="coerce")
        out[f"historical_{col}_delta"] = home - away
    out["fjelstul_historical_both_sides_flag"] = (
        pd.to_numeric(out["home_fjelstul_historical_prior_ready_flag"], errors="coerce").fillna(0).astype(int).eq(1)
        & pd.to_numeric(out["away_fjelstul_historical_prior_ready_flag"], errors="coerce").fillna(0).astype(int).eq(1)
    ).astype(int)
    out["fjelstul_recent_both_sides_flag"] = (
        pd.to_numeric(out["home_fjelstul_recent_prior_ready_flag"], errors="coerce").fillna(0).astype(int).eq(1)
        & pd.to_numeric(out["away_fjelstul_recent_prior_ready_flag"], errors="coerce").fillna(0).astype(int).eq(1)
    ).astype(int)
    out["world_cup_research_feature_readiness"] = "MACRO_PLAYER_QUALIFIER_HISTORY"
    out.loc[out["qualifier_form_both_sides_flag"].fillna(0).astype(int).eq(0), "world_cup_research_feature_readiness"] = (
        "MACRO_PLAYER_PARTIAL_QUALIFIER_HISTORY"
    )
    out.loc[out["fjelstul_historical_both_sides_flag"].eq(0), "world_cup_research_feature_readiness"] = (
        out["world_cup_research_feature_readiness"] + "_PARTIAL_WC_HISTORY"
    )
    return out


def add_additions_deltas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "home_additions_context_ready_flag" not in out.columns:
        out["additions_context_both_sides_flag"] = 0
        out["additions_player_context_both_sides_flag"] = 0
        return out
    out["additions_context_both_sides_flag"] = (
        pd.to_numeric(out["home_additions_context_ready_flag"], errors="coerce").fillna(0).astype(int).eq(1)
        & pd.to_numeric(out["away_additions_context_ready_flag"], errors="coerce").fillna(0).astype(int).eq(1)
    ).astype(int)
    out["additions_player_context_both_sides_flag"] = (
        pd.to_numeric(out["home_additions_player_context_ready_flag"], errors="coerce").fillna(0).astype(int).eq(1)
        & pd.to_numeric(out["away_additions_player_context_ready_flag"], errors="coerce").fillna(0).astype(int).eq(1)
    ).astype(int)
    out["host_additions_context_in_fixture_flag"] = (
        pd.to_numeric(out.get("home_additions_host_context_flag"), errors="coerce").fillna(0).astype(int).eq(1)
        | pd.to_numeric(out.get("away_additions_host_context_flag"), errors="coerce").fillna(0).astype(int).eq(1)
    ).astype(int)
    for col in ADDITIONS_DELTA_COLS:
        home = pd.to_numeric(out.get(f"home_{col}"), errors="coerce")
        away = pd.to_numeric(out.get(f"away_{col}"), errors="coerce")
        out[f"{col}_delta"] = home - away
    out["world_cup_research_feature_readiness_v2"] = out["world_cup_research_feature_readiness"]
    out.loc[
        out["additions_context_both_sides_flag"].eq(1),
        "world_cup_research_feature_readiness_v2",
    ] = out["world_cup_research_feature_readiness_v2"] + "_ADDITIONS_READY"
    out.loc[
        out["additions_context_both_sides_flag"].eq(0),
        "world_cup_research_feature_readiness_v2",
    ] = out["world_cup_research_feature_readiness_v2"] + "_ADDITIONS_PARTIAL"
    return out


def merge_fixture_sidecar(matrix: pd.DataFrame, sidecar_path: Path, label: str) -> pd.DataFrame:
    if not sidecar_path.exists():
        return matrix
    sidecar = pd.read_csv(sidecar_path, low_memory=False)
    if "api_fixture_id" not in sidecar.columns:
        return matrix
    keep = ["api_fixture_id"] + [c for c in sidecar.columns if c != "api_fixture_id" and c not in matrix.columns]
    out = matrix.merge(sidecar[keep], on="api_fixture_id", how="left")
    out[f"{label}_sidecar_joined_flag"] = out[keep[1:]].notna().any(axis=1).astype(int) if len(keep) > 1 else 0
    return out


def add_fixture_intelligence_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "home_recent_history_since_2023_matches" in out.columns and "away_recent_history_since_2023_matches" in out.columns:
        out["recent_history_since_2023_both_sides_flag"] = (
            pd.to_numeric(out["home_recent_history_since_2023_matches"], errors="coerce").gt(0)
            & pd.to_numeric(out["away_recent_history_since_2023_matches"], errors="coerce").gt(0)
        ).astype(int)
    if "local_h2h_match_count" in out.columns:
        out["local_h2h_available_flag"] = pd.to_numeric(out["local_h2h_match_count"], errors="coerce").fillna(0).gt(0).astype(int)
    if "venue_known_flag" in out.columns:
        out["venue_weather_travel_known_flag"] = pd.to_numeric(out["venue_known_flag"], errors="coerce").fillna(0).astype(int)
    if "weather_model_ready_flag" in out.columns:
        out["official_weather_truth_pending_flag"] = pd.to_numeric(out["weather_model_ready_flag"], errors="coerce").fillna(0).eq(0).astype(int)
    return out


def write_summary(outdir: Path, matrix: pd.DataFrame) -> None:
    readiness = matrix["world_cup_research_feature_readiness"].value_counts().rename_axis("status").reset_index(name="fixtures")
    readiness.to_csv(outdir / "world_cup_2026_research_feature_readiness_counts.csv", index=False)
    if "world_cup_research_feature_readiness_v2" in matrix.columns:
        matrix["world_cup_research_feature_readiness_v2"].value_counts().rename_axis("status").reset_index(name="fixtures").to_csv(
            outdir / "world_cup_2026_research_feature_readiness_v2_counts.csv", index=False
        )
    top = matrix.sort_values("qualifier_ppg_delta", ascending=False).head(10)
    top_lines = [
        f"- {r.api_home_team_name} vs {r.api_away_team_name}: qualifier PPG delta={r.qualifier_ppg_delta:.2f}, "
        f"macro FTR={r.macro_pick_ftr}, band={r.world_cup_ftr_research_band}"
        for r in top.itertuples()
        if pd.notna(getattr(r, "qualifier_ppg_delta", pd.NA))
    ]
    lines = [
        "# World Cup 2026 Research Feature Matrix",
        "",
        "Fixture-level research sidecar joining macro priors, player scaffold, qualification aggregates, and historical World Cup priors.",
        "",
        "## Outputs",
        "",
        f"- `{outdir / 'world_cup_2026_research_feature_matrix.csv'}`",
        f"- `{outdir / 'world_cup_2026_research_feature_readiness_counts.csv'}`",
        "",
        "## Coverage",
        "",
        f"- Fixtures: {len(matrix)}",
        f"- Both-side FootyStats qualifier coverage: {int(matrix['qualifier_form_both_sides_flag'].sum())} / {len(matrix)}",
        f"- Both-side Fjelstul historical coverage: {int(matrix['fjelstul_historical_both_sides_flag'].sum())} / {len(matrix)}",
        f"- Both-side Fjelstul recent 2010-2018 coverage: {int(matrix['fjelstul_recent_both_sides_flag'].sum())} / {len(matrix)}",
        f"- Both-side additions context coverage: {int(matrix.get('additions_context_both_sides_flag', pd.Series(dtype=int)).sum())} / {len(matrix)}",
        f"- Both-side additions player context coverage: {int(matrix.get('additions_player_context_both_sides_flag', pd.Series(dtype=int)).sum())} / {len(matrix)}",
        f"- Both-side recent-history coverage: {int(matrix.get('recent_history_since_2023_both_sides_flag', pd.Series(dtype=int)).sum())} / {len(matrix)}",
        f"- Local FootyStats H2H available: {int(matrix.get('local_h2h_available_flag', pd.Series(dtype=int)).sum())} / {len(matrix)}",
        f"- Venue/weather/travel scaffold known venue: {int(matrix.get('venue_weather_travel_known_flag', pd.Series(dtype=int)).sum())} / {len(matrix)}",
        "",
        "## Highest Qualifier PPG Home-Side Deltas",
        "",
        *top_lines,
        "",
        "## Notes",
        "",
        "- Research-only: this file does not update `Matches/__merged__/World_Cup__merged.csv` or deploy policy.",
        "- Host-auto teams can have weak qualifier coverage by design; treat those rows as partial qualifier contexts.",
        "- Fjelstul priors end at 2018 and need CC-BY-SA attribution review before product packaging.",
        "- Additions context uses regional tournaments/friendlies as pre-tournament proxy intelligence and keeps source flags attached.",
    ]
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-sidecar", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--fjelstul-sidecar", type=Path, default=DEFAULT_FJELSTUL)
    parser.add_argument("--additions-sidecar", type=Path, default=DEFAULT_ADDITIONS)
    parser.add_argument("--recent-history-sidecar", type=Path, default=DEFAULT_RECENT_HISTORY)
    parser.add_argument("--local-h2h-sidecar", type=Path, default=DEFAULT_LOCAL_H2H)
    parser.add_argument("--venue-travel-sidecar", type=Path, default=DEFAULT_VENUE_TRAVEL)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    context = pd.read_csv(args.context_sidecar, low_memory=False)
    fjelstul = pd.read_csv(args.fjelstul_sidecar, low_memory=False)
    matrix = side_join(context, fjelstul, "home")
    matrix = side_join(matrix, fjelstul, "away")
    matrix = add_historical_deltas(matrix)
    if args.additions_sidecar.exists():
        additions = pd.read_csv(args.additions_sidecar, low_memory=False)
        matrix = side_join(matrix, additions, "home")
        matrix = side_join(matrix, additions, "away")
        matrix = add_additions_deltas(matrix)
    matrix = merge_fixture_sidecar(matrix, args.recent_history_sidecar, "recent_history")
    matrix = merge_fixture_sidecar(matrix, args.local_h2h_sidecar, "local_h2h")
    matrix = merge_fixture_sidecar(matrix, args.venue_travel_sidecar, "venue_weather_travel")
    matrix = add_fixture_intelligence_flags(matrix)
    matrix.to_csv(args.outdir / "world_cup_2026_research_feature_matrix.csv", index=False)
    write_summary(args.outdir, matrix)
    print(f"[ok] matrix={len(matrix)}")
    print(f"[ok] wrote {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
