#!/usr/bin/env python3
"""
Build a World Cup 2026 qualification-context sidecar.

This captures the "road to the tournament" modelling layer:
- confederation/context priors
- host/direct/playoff/unknown route placeholders
- travel/climate/volatility priors
- fixture-level asymmetry and market-risk flags

The sidecar is research-only and intentionally conservative. It does not treat
unverified qualifying-path notes as facts; verified data can be supplied through
the generated override template.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd


DEFAULT_LAUNCH = Path("data_sources/footystats_world_cup/launch_2026/world_cup_2026_launch_scaffold.csv")
DEFAULT_OVERLAY = Path("data_sources/footystats_world_cup/intelligence_overlay_2026/world_cup_2026_model_ready_sidecar.csv")
DEFAULT_OUTDIR = Path("data_sources/footystats_world_cup/qualification_context_2026")


CONFEDERATION_BY_TEAM = {
    "algeria": "CAF",
    "argentina": "CONMEBOL",
    "australia": "AFC",
    "austria": "UEFA",
    "belgium": "UEFA",
    "bosnia_and_herzegovina": "UEFA",
    "brazil": "CONMEBOL",
    "canada": "CONCACAF",
    "cape_verde_islands": "CAF",
    "colombia": "CONMEBOL",
    "congo_dr": "CAF",
    "croatia": "UEFA",
    "curacao": "CONCACAF",
    "czech_republic": "UEFA",
    "ecuador": "CONMEBOL",
    "egypt": "CAF",
    "england": "UEFA",
    "france": "UEFA",
    "germany": "UEFA",
    "ghana": "CAF",
    "haiti": "CONCACAF",
    "iran": "AFC",
    "iraq": "AFC",
    "ivory_coast": "CAF",
    "japan": "AFC",
    "jordan": "AFC",
    "mexico": "CONCACAF",
    "morocco": "CAF",
    "netherlands": "UEFA",
    "new_zealand": "OFC",
    "norway": "UEFA",
    "panama": "CONCACAF",
    "paraguay": "CONMEBOL",
    "portugal": "UEFA",
    "qatar": "AFC",
    "saudi_arabia": "AFC",
    "scotland": "UEFA",
    "senegal": "CAF",
    "south_africa": "CAF",
    "south_korea": "AFC",
    "spain": "UEFA",
    "sweden": "UEFA",
    "switzerland": "UEFA",
    "tunisia": "CAF",
    "turkiye": "UEFA",
    "usa": "CONCACAF",
    "uruguay": "CONMEBOL",
    "uzbekistan": "AFC",
}

CONFEDERATION_BY_TEAM.update(
    {
        "cape_verde": "CAF",
        "cura_ao": "CONCACAF",
        "curacao": "CONCACAF",
        "dr_congo": "CAF",
        "t_rkiye": "UEFA",
        "turkiye": "UEFA",
    }
)


TEAM_SLUG_ALIASES = {
    "cape_verde": "cape_verde",
    "cape_verde_islands": "cape_verde",
    "congo_dr": "dr_congo",
    "dr_congo": "dr_congo",
    "curacao": "curacao",
    "cura_ao": "curacao",
    "turkey": "turkiye",
    "turkiye": "turkiye",
    "t_rkiye": "turkiye",
}


QUALIFIER_METRIC_COLS = [
    "qualifier_matches_played",
    "qualifier_ppg",
    "qualifier_goal_diff_per_match",
    "qualifier_goals_for_per_match",
    "qualifier_goals_against_per_match",
    "qualification_position",
    "qualification_goal_diff",
]


CONFEDERATION_PRIORS = {
    "UEFA": {
        "qualifier_structure": "GROUPS_PLUS_PLAYOFFS_NATIONS_LEAGUE_CONTEXT",
        "confed_volatility_prior": 0.42,
        "qualification_pressure_prior": 0.55,
        "travel_climate_complexity_prior": 0.35,
        "market_efficiency_risk_prior": 0.35,
        "notes": "Deep talent pool, playoff/Nations League context, fixture congestion.",
    },
    "CONMEBOL": {
        "qualifier_structure": "SINGLE_ROUND_ROBIN_HOME_AWAY",
        "confed_volatility_prior": 0.48,
        "qualification_pressure_prior": 0.72,
        "travel_climate_complexity_prior": 0.68,
        "market_efficiency_risk_prior": 0.42,
        "notes": "Clean opposition grid, high intensity, altitude/travel effects.",
    },
    "CONCACAF": {
        "qualifier_structure": "HOSTS_PLUS_MULTI_ROUND_QUALIFYING",
        "confed_volatility_prior": 0.62,
        "qualification_pressure_prior": 0.58,
        "travel_climate_complexity_prior": 0.78,
        "market_efficiency_risk_prior": 0.62,
        "notes": "Climate variance, island travel, host asymmetries.",
    },
    "CAF": {
        "qualifier_structure": "LARGE_GROUP_STAGE_PLUS_PLAYOFF_CONTEXT",
        "confed_volatility_prior": 0.74,
        "qualification_pressure_prior": 0.70,
        "travel_climate_complexity_prior": 0.82,
        "market_efficiency_risk_prior": 0.72,
        "notes": "Travel, infrastructure, heat, pitch and availability volatility.",
    },
    "AFC": {
        "qualifier_structure": "MULTI_ROUND_LONG_FORM_QUALIFYING",
        "confed_volatility_prior": 0.56,
        "qualification_pressure_prior": 0.62,
        "travel_climate_complexity_prior": 0.84,
        "market_efficiency_risk_prior": 0.55,
        "notes": "Long qualification path, timezone/climate/tactical diversity.",
    },
    "OFC": {
        "qualifier_structure": "DIRECT_SLOT_PLUS_SMALL_FIELD_CONTEXT",
        "confed_volatility_prior": 0.66,
        "qualification_pressure_prior": 0.50,
        "travel_climate_complexity_prior": 0.72,
        "market_efficiency_risk_prior": 0.70,
        "notes": "Small field, sparse market priors, unusual opponent profiles.",
    },
    "UNKNOWN": {
        "qualifier_structure": "UNKNOWN",
        "confed_volatility_prior": 0.50,
        "qualification_pressure_prior": 0.50,
        "travel_climate_complexity_prior": 0.50,
        "market_efficiency_risk_prior": 0.50,
        "notes": "Needs manual mapping.",
    },
}


HOST_TEAMS = {"canada", "mexico", "usa"}


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


def load_override(path: Path | None) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if "team_slug" not in df.columns:
        team_col = next((c for c in ["team_name", "team", "country", "nation"] if c in df.columns), None)
        if not team_col:
            raise SystemExit("Qualification override needs team_slug or team/country/nation column.")
        df["team_slug"] = df[team_col].map(canonical_team_slug)
    else:
        df["team_slug"] = df["team_slug"].map(canonical_team_slug)
    return df.drop_duplicates(subset=["team_slug"], keep="last")


def extract_teams(launch: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for side in ["home", "away"]:
        pieces.append(
            launch[[f"{side}_team_slug", f"{side}_team_name_latest", f"api_{side}_team_name", f"{side}_is_host"]].rename(
                columns={
                    f"{side}_team_slug": "team_slug",
                    f"{side}_team_name_latest": "team_name",
                    f"api_{side}_team_name": "api_team_name",
                    f"{side}_is_host": "is_host_from_launch",
                }
            )
        )
    teams = pd.concat(pieces, ignore_index=True).drop_duplicates(subset=["team_slug"]).reset_index(drop=True)
    teams["team_slug"] = teams["team_slug"].map(canonical_team_slug)
    teams["team_name"] = teams["team_name"].combine_first(teams["api_team_name"])
    teams = teams.drop(columns=["api_team_name"])
    return teams


def build_team_context(launch: pd.DataFrame, override: pd.DataFrame) -> pd.DataFrame:
    teams = extract_teams(launch)
    rows = []
    for row in teams.itertuples(index=False):
        confed = CONFEDERATION_BY_TEAM.get(row.team_slug, "UNKNOWN")
        prior = CONFEDERATION_PRIORS[confed]
        route = "HOST_AUTO" if row.team_slug in HOST_TEAMS else "QUALIFIED_ROUTE_TO_VERIFY"
        source = "STATIC_CONFEDERATION_PRIOR_UNVERIFIED_ROUTE"
        rows.append(
            {
                "team_slug": row.team_slug,
                "team_name": row.team_name,
                "confederation": confed,
                "is_host": int(row.team_slug in HOST_TEAMS),
                "qualification_route": route,
                "qualification_route_verified_flag": 1 if route == "HOST_AUTO" else 0,
                "qualifier_structure": prior["qualifier_structure"],
                "confed_volatility_prior": prior["confed_volatility_prior"],
                "qualification_pressure_prior": prior["qualification_pressure_prior"],
                "travel_climate_complexity_prior": prior["travel_climate_complexity_prior"],
                "market_efficiency_risk_prior": prior["market_efficiency_risk_prior"],
                "qualifier_context_notes": prior["notes"],
                "qualification_context_source_status": source,
            }
        )
    out = pd.DataFrame(rows)
    for col in QUALIFIER_METRIC_COLS:
        if col not in out.columns:
            out[col] = pd.NA
    if not override.empty:
        override_cols = [
            c
            for c in override.columns
            if c
            in {
                "team_slug",
                "confederation",
                "qualification_route",
                "qualification_route_verified_flag",
                "qualification_group",
                "qualification_position",
                "qualification_points",
                "qualification_goal_diff",
                "qualification_playoff_flag",
                "nations_league_playoff_context_flag",
                "intercontinental_playoff_flag",
                "qualifier_matches_played",
                "qualifier_ppg",
                "qualifier_goal_diff_per_match",
                "qualifier_goals_for_per_match",
                "qualifier_goals_against_per_match",
                "qualifier_context_notes",
                "qualification_context_source_status",
            }
        ]
        merged = out.merge(override[override_cols], on="team_slug", how="left", suffixes=("", "_override"))
        for col in override_cols:
            if col == "team_slug" or f"{col}_override" not in merged.columns:
                continue
            merged[col] = merged[f"{col}_override"].combine_first(merged[col])
            merged = merged.drop(columns=[f"{col}_override"])
        out = merged
    for col in QUALIFIER_METRIC_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["qualifier_form_ready_flag"] = (
        pd.to_numeric(out["qualifier_matches_played"], errors="coerce").fillna(0) > 0
    ).astype(int)
    for col in [
        "qualification_route_verified_flag",
        "qualification_playoff_flag",
        "nations_league_playoff_context_flag",
        "intercontinental_playoff_flag",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    out["qualification_context_readiness"] = out.apply(
        lambda r: "VERIFIED_ROUTE"
        if int(r.get("qualification_route_verified_flag") or 0) == 1
        else "HOST_AUTO"
        if r.get("qualification_route") == "HOST_AUTO"
        else "PRIOR_ONLY_NEEDS_QUALIFIER_DATA",
        axis=1,
    )
    return out.sort_values(["confederation", "team_slug"]).reset_index(drop=True)


def side_join(fixtures: pd.DataFrame, teams: pd.DataFrame, side: str) -> pd.DataFrame:
    key = f"{side}_team_slug"
    side_teams = teams.rename(columns={c: f"{side}_{c}" for c in teams.columns if c != "team_slug"})
    side_teams = side_teams.rename(columns={"team_slug": key})
    out = fixtures.copy()
    out[key] = out[key].map(canonical_team_slug)
    return out.merge(side_teams, on=key, how="left")


def numeric_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def build_fixture_context(launch: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    base = launch[
        [
            "season",
            "api_fixture_id",
            "api_date",
            "api_round",
            "api_home_team_name",
            "api_away_team_name",
            "home_team_slug",
            "away_team_slug",
        ]
    ].copy()
    out = side_join(base, teams, "home")
    out = side_join(out, teams, "away")
    out["cross_confederation_fixture_flag"] = (out["home_confederation"] != out["away_confederation"]).astype(int)
    out["host_in_fixture_flag"] = (
        pd.to_numeric(out.get("home_is_host"), errors="coerce").fillna(0)
        + pd.to_numeric(out.get("away_is_host"), errors="coerce").fillna(0)
        > 0
    ).astype(int)
    out["qualification_pressure_delta"] = (
        pd.to_numeric(out["home_qualification_pressure_prior"], errors="coerce").fillna(0.5)
        - pd.to_numeric(out["away_qualification_pressure_prior"], errors="coerce").fillna(0.5)
    )
    out["home_qualifier_form_ready_flag"] = numeric_col(out, "home_qualifier_form_ready_flag").fillna(0).astype(int)
    out["away_qualifier_form_ready_flag"] = numeric_col(out, "away_qualifier_form_ready_flag").fillna(0).astype(int)
    out["qualifier_form_any_side_flag"] = (
        (out["home_qualifier_form_ready_flag"] == 1) | (out["away_qualifier_form_ready_flag"] == 1)
    ).astype(int)
    out["qualifier_form_both_sides_flag"] = (
        (out["home_qualifier_form_ready_flag"] == 1) & (out["away_qualifier_form_ready_flag"] == 1)
    ).astype(int)
    out["qualifier_ppg_delta"] = numeric_col(out, "home_qualifier_ppg") - numeric_col(out, "away_qualifier_ppg")
    out["qualifier_goal_diff_per_match_delta"] = (
        numeric_col(out, "home_qualifier_goal_diff_per_match")
        - numeric_col(out, "away_qualifier_goal_diff_per_match")
    )
    out["qualifier_attack_delta"] = (
        numeric_col(out, "home_qualifier_goals_for_per_match")
        - numeric_col(out, "away_qualifier_goals_for_per_match")
    )
    out["qualifier_defence_delta"] = (
        numeric_col(out, "away_qualifier_goals_against_per_match")
        - numeric_col(out, "home_qualifier_goals_against_per_match")
    )
    out["confed_volatility_max"] = pd.concat(
        [
            pd.to_numeric(out["home_confed_volatility_prior"], errors="coerce").fillna(0.5),
            pd.to_numeric(out["away_confed_volatility_prior"], errors="coerce").fillna(0.5),
        ],
        axis=1,
    ).max(axis=1)
    out["travel_climate_complexity_max"] = pd.concat(
        [
            pd.to_numeric(out["home_travel_climate_complexity_prior"], errors="coerce").fillna(0.5),
            pd.to_numeric(out["away_travel_climate_complexity_prior"], errors="coerce").fillna(0.5),
        ],
        axis=1,
    ).max(axis=1)
    out["market_efficiency_risk_max"] = pd.concat(
        [
            pd.to_numeric(out["home_market_efficiency_risk_prior"], errors="coerce").fillna(0.5),
            pd.to_numeric(out["away_market_efficiency_risk_prior"], errors="coerce").fillna(0.5),
        ],
        axis=1,
    ).max(axis=1)
    out["qualification_context_risk_score"] = (
        0.35 * out["confed_volatility_max"]
        + 0.25 * out["travel_climate_complexity_max"]
        + 0.25 * out["market_efficiency_risk_max"]
        + 0.15 * out["cross_confederation_fixture_flag"]
    ).clip(0, 1)
    out["qualification_context_band"] = pd.cut(
        out["qualification_context_risk_score"],
        bins=[-0.01, 0.45, 0.60, 0.72, 1.01],
        labels=["LOW", "MEDIUM", "HIGH", "EXTREME"],
    ).astype(str)
    out["qualification_context_ready_flag"] = (
        (out["home_qualification_route_verified_flag"].fillna(0).astype(int) == 1)
        & (out["away_qualification_route_verified_flag"].fillna(0).astype(int) == 1)
    ).astype(int)
    return out


def write_template(path: Path, teams: pd.DataFrame) -> None:
    cols = [
        "team_slug",
        "team_name",
        "confederation",
        "qualification_route",
        "qualification_route_verified_flag",
        "qualification_group",
        "qualification_position",
        "qualification_points",
        "qualification_goal_diff",
        "qualification_playoff_flag",
        "nations_league_playoff_context_flag",
        "intercontinental_playoff_flag",
        "qualifier_matches_played",
        "qualifier_ppg",
        "qualifier_goal_diff_per_match",
        "qualifier_goals_for_per_match",
        "qualifier_goals_against_per_match",
        "qualification_context_source_status",
        "qualifier_context_notes",
    ]
    template = teams[["team_slug", "team_name", "confederation"]].copy()
    for col in cols:
        if col not in template.columns:
            template[col] = ""
    template[cols].to_csv(path, index=False)


def write_summary(outdir: Path, team_context: pd.DataFrame, fixture_context: pd.DataFrame) -> None:
    confed = team_context["confederation"].value_counts().rename_axis("confederation").reset_index(name="teams")
    bands = fixture_context["qualification_context_band"].value_counts().rename_axis("band").reset_index(name="fixtures")
    readiness = team_context["qualification_context_readiness"].value_counts().rename_axis("status").reset_index(name="teams")
    form_ready = team_context["qualifier_form_ready_flag"].value_counts().rename_axis("ready").reset_index(name="teams")
    fixture_form_ready = (
        fixture_context["qualifier_form_both_sides_flag"].value_counts().rename_axis("both_sides_ready").reset_index(name="fixtures")
    )
    confed.to_csv(outdir / "world_cup_2026_confederation_team_counts.csv", index=False)
    bands.to_csv(outdir / "world_cup_2026_qualification_context_band_counts.csv", index=False)
    readiness.to_csv(outdir / "world_cup_2026_qualification_context_readiness_counts.csv", index=False)
    form_ready.to_csv(outdir / "world_cup_2026_qualifier_form_team_coverage.csv", index=False)
    fixture_form_ready.to_csv(outdir / "world_cup_2026_qualifier_form_fixture_coverage.csv", index=False)

    def table(df: pd.DataFrame, left: str, right: str) -> list[str]:
        lines = [f"| {left} | {right} |", "|---|---:|"]
        lines.extend(f"| {getattr(r, left)} | {int(getattr(r, right))} |" for r in df.itertuples(index=False))
        return lines

    top = fixture_context.sort_values("qualification_context_risk_score", ascending=False).head(12)
    top_lines = [
        f"- {r.api_home_team_name} vs {r.api_away_team_name}: {r.qualification_context_band} "
        f"({r.qualification_context_risk_score:.3f}), {r.home_confederation}-{r.away_confederation}"
        for r in top.itertuples()
    ]
    lines = [
        "# World Cup 2026 Qualification Context Scaffold",
        "",
        "Research-only sidecar for road-to-tournament context.",
        "",
        "## Outputs",
        "",
        f"- `{outdir / 'world_cup_2026_team_qualification_context.csv'}`",
        f"- `{outdir / 'world_cup_2026_fixture_qualification_context_matrix.csv'}`",
        f"- `{outdir / 'world_cup_2026_model_ready_with_qualification_context.csv'}`",
        f"- `{outdir / 'world_cup_qualification_context_override_template.csv'}`",
        "",
        "## Team Confederation Coverage",
        "",
        *table(confed, "confederation", "teams"),
        "",
        "## Team Context Readiness",
        "",
        *table(readiness, "status", "teams"),
        "",
        "## FootyStats Qualifier-Form Coverage",
        "",
        f"- Teams with qualifier aggregate stats: {int(team_context['qualifier_form_ready_flag'].sum())} / {len(team_context)}",
        f"- Fixtures with both sides covered: {int(fixture_context['qualifier_form_both_sides_flag'].sum())} / {len(fixture_context)}",
        f"- Fixtures with at least one side covered: {int(fixture_context['qualifier_form_any_side_flag'].sum())} / {len(fixture_context)}",
        "",
        "## Fixture Qualification Context Bands",
        "",
        *table(bands, "band", "fixtures"),
        "",
        "## Highest Context-Risk Fixtures",
        "",
        *top_lines,
        "",
        "## Notes",
        "",
        "- Confederation priors are usable now as modelling priors.",
        "- Qualification route details are placeholders unless supplied via the override template.",
        "- This layer should inform risk bands and feature research, not live deployment gates.",
    ]
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-scaffold", type=Path, default=DEFAULT_LAUNCH)
    parser.add_argument("--overlay-sidecar", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--qualification-override", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    launch = pd.read_csv(args.launch_scaffold, low_memory=False)
    overlay = pd.read_csv(args.overlay_sidecar, low_memory=False) if args.overlay_sidecar.exists() else pd.DataFrame()
    override = load_override(args.qualification_override)

    team_context = build_team_context(launch, override)
    fixture_context = build_fixture_context(launch, team_context)
    write_template(args.outdir / "world_cup_qualification_context_override_template.csv", team_context)

    team_context.to_csv(args.outdir / "world_cup_2026_team_qualification_context.csv", index=False)
    fixture_context.to_csv(args.outdir / "world_cup_2026_fixture_qualification_context_matrix.csv", index=False)

    if not overlay.empty:
        keep = [
            "api_fixture_id",
            "home_confederation",
            "away_confederation",
            "home_qualification_context_source_status",
            "away_qualification_context_source_status",
            "home_qualifier_form_ready_flag",
            "away_qualifier_form_ready_flag",
            "qualifier_form_any_side_flag",
            "qualifier_form_both_sides_flag",
            "home_qualifier_matches_played",
            "away_qualifier_matches_played",
            "home_qualifier_ppg",
            "away_qualifier_ppg",
            "qualifier_ppg_delta",
            "home_qualifier_goal_diff_per_match",
            "away_qualifier_goal_diff_per_match",
            "qualifier_goal_diff_per_match_delta",
            "home_qualifier_goals_for_per_match",
            "away_qualifier_goals_for_per_match",
            "qualifier_attack_delta",
            "home_qualifier_goals_against_per_match",
            "away_qualifier_goals_against_per_match",
            "qualifier_defence_delta",
            "home_qualification_position",
            "away_qualification_position",
            "cross_confederation_fixture_flag",
            "host_in_fixture_flag",
            "qualification_pressure_delta",
            "confed_volatility_max",
            "travel_climate_complexity_max",
            "market_efficiency_risk_max",
            "qualification_context_risk_score",
            "qualification_context_band",
            "qualification_context_ready_flag",
        ]
        joined = overlay.merge(fixture_context[keep], on="api_fixture_id", how="left", validate="one_to_one")
        joined.to_csv(args.outdir / "world_cup_2026_model_ready_with_qualification_context.csv", index=False)

    write_summary(args.outdir, team_context, fixture_context)
    print(f"[ok] teams={len(team_context)} fixtures={len(fixture_context)}")
    print(f"[ok] wrote {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
