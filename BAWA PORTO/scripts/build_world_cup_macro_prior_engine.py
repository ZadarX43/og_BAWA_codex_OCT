#!/usr/bin/env python3
"""
Build a research-only World Cup macro prior engine.

This is the Klement-style baseline layer for Odds Genius:
- pre-tournament team strength
- optional external priors such as FIFA/Elo/squad value
- match-level FTR/BTTS/OU25 probability priors

It does not change production routing, training artifacts, or deploy gates.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_LAUNCH = Path("data_sources/footystats_world_cup/launch_2026/world_cup_2026_launch_scaffold.csv")
DEFAULT_OUTDIR = Path("data_sources/footystats_world_cup/macro_prior_engine")


SIDE_PREFIXES = ("home_", "away_")


TEAM_PRIOR_COLUMNS = {
    "wc_matches_2006_2022": "wc_matches",
    "wc_tournaments_2006_2022": "wc_tournaments",
    "wc_last_seen_year": "wc_last_seen_year",
    "wc_points_per_match": "wc_points_per_match",
    "wc_goal_diff_per_match": "wc_goal_diff_per_match",
    "wc_goals_for_per_match": "wc_goals_for_per_match",
    "wc_goals_against_per_match": "wc_goals_against_per_match",
    "wc_btts_rate": "wc_btts_rate",
    "wc_over25_rate": "wc_over25_rate",
    "wc_knockout_match_rate": "wc_knockout_match_rate",
    "wc_weighted_points_per_match": "wc_weighted_points_per_match",
    "wc_weighted_goal_diff_per_match": "wc_weighted_goal_diff_per_match",
    "wc_weighted_goals_for_per_match": "wc_weighted_goals_for_per_match",
    "wc_weighted_goals_against_per_match": "wc_weighted_goals_against_per_match",
    "last_wc_squad_year": "last_wc_squad_year",
    "last_wc_squad_players": "last_wc_squad_players",
    "last_wc_squad_avg_age": "last_wc_squad_avg_age",
    "last_wc_goalkeepers": "last_wc_goalkeepers",
    "last_wc_defenders": "last_wc_defenders",
    "last_wc_midfielders": "last_wc_midfielders",
    "last_wc_forwards": "last_wc_forwards",
}


EXTERNAL_NUMERIC_COLUMNS = {
    "fifa_rank": {"direction": -1, "weight": 0.20},
    "fifa_points": {"direction": 1, "weight": 0.20},
    "elo": {"direction": 1, "weight": 0.30},
    "elo_rating": {"direction": 1, "weight": 0.30},
    "squad_market_value_eur": {"direction": 1, "weight": 0.22},
    "squad_market_value": {"direction": 1, "weight": 0.22},
    "market_value_eur": {"direction": 1, "weight": 0.22},
    "squad_avg_rating": {"direction": 1, "weight": 0.26},
    "expected_xi_avg_rating": {"direction": 1, "weight": 0.30},
    "domestic_player_rating": {"direction": 1, "weight": 0.25},
    "gdp_per_capita": {"direction": 1, "weight": 0.05},
    "population": {"direction": 1, "weight": 0.04},
    "mean_temperature_c": {"direction": 0, "weight": 0.00},
}


def slugify(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.lower().strip()
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def zscore(series: pd.Series) -> pd.Series:
    nums = to_num(series)
    mean = nums.mean()
    std = nums.std(ddof=0)
    if not math.isfinite(float(std or 0)) or float(std or 0) == 0.0:
        return pd.Series(0.0, index=series.index)
    return (nums.fillna(mean) - mean) / std


def poisson_pmf(rate: float, max_goals: int) -> list[float]:
    rate = max(0.05, min(float(rate), 5.0))
    probs = []
    for goals in range(max_goals + 1):
        probs.append(math.exp(-rate) * (rate ** goals) / math.factorial(goals))
    total = sum(probs)
    if total > 0:
        probs = [p / total for p in probs]
    return probs


def goal_market_probs(home_lambda: float, away_lambda: float, max_goals: int = 10) -> dict[str, float]:
    hp = poisson_pmf(home_lambda, max_goals)
    ap = poisson_pmf(away_lambda, max_goals)
    home = draw = away = over25 = btts = 0.0
    for hg, hpv in enumerate(hp):
        for ag, apv in enumerate(ap):
            p = hpv * apv
            if hg > ag:
                home += p
            elif hg == ag:
                draw += p
            else:
                away += p
            if hg + ag >= 3:
                over25 += p
            if hg > 0 and ag > 0:
                btts += p
    return {
        "macro_prob_home": home,
        "macro_prob_draw": draw,
        "macro_prob_away": away,
        "macro_prob_over25": over25,
        "macro_prob_btts_yes": btts,
    }


def extract_team_table(launch: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for side in SIDE_PREFIXES:
        rename = {
            f"{side}team_slug": "team_slug",
            f"{side}team_name_latest": "team_name_latest",
        }
        for source, target in TEAM_PRIOR_COLUMNS.items():
            rename[f"{side}{source}"] = target
        cols = [c for c in rename if c in launch.columns]
        part = launch[cols].rename(columns={c: rename[c] for c in cols}).copy()
        part["team_slug"] = part["team_slug"].map(slugify)
        pieces.append(part)

    teams = pd.concat(pieces, ignore_index=True)
    teams = teams.drop_duplicates(subset=["team_slug"], keep="first").reset_index(drop=True)
    for col in teams.columns:
        if col not in {"team_slug", "team_name_latest"}:
            teams[col] = to_num(teams[col])
    teams["has_world_cup_prior"] = (teams.get("wc_matches", 0).fillna(0) > 0).astype(int)
    teams["has_2022_prior"] = (teams.get("wc_last_seen_year", 0).fillna(0) >= 2022).astype(int)
    return teams


def load_external_priors(path: Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    ext = pd.read_csv(path, low_memory=False)
    if "team_slug" not in ext.columns:
        name_col = next((c for c in ["team_name", "team", "country", "nation", "name"] if c in ext.columns), None)
        if not name_col:
            raise SystemExit(
                "External priors must include team_slug or one of: team_name, team, country, nation, name"
            )
        ext["team_slug"] = ext[name_col].map(slugify)
    else:
        ext["team_slug"] = ext["team_slug"].map(slugify)
    ext = ext.drop_duplicates(subset=["team_slug"], keep="first").copy()
    return ext


def add_external_priors(teams: pd.DataFrame, external: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if external.empty:
        teams["external_prior_joined_flag"] = 0
        return teams, []
    ext_cols = ["team_slug"] + [c for c in external.columns if c != "team_slug"]
    merged = teams.merge(external[ext_cols], on="team_slug", how="left")
    merged["external_prior_joined_flag"] = 0
    external_feature_cols = []
    for col in external.columns:
        if col == "team_slug":
            continue
        if col in EXTERNAL_NUMERIC_COLUMNS or pd.api.types.is_numeric_dtype(external[col]):
            merged[col] = to_num(merged[col])
            external_feature_cols.append(col)
            merged["external_prior_joined_flag"] = merged["external_prior_joined_flag"].where(
                merged[col].isna(), 1
            )
    return merged, external_feature_cols


def add_macro_scores(teams: pd.DataFrame, external_feature_cols: Iterable[str]) -> pd.DataFrame:
    out = teams.copy()
    out["wc_strength_core_z"] = (
        0.35 * zscore(out.get("wc_weighted_points_per_match", pd.Series(index=out.index, dtype=float)))
        + 0.25 * zscore(out.get("wc_weighted_goal_diff_per_match", pd.Series(index=out.index, dtype=float)))
        + 0.15 * zscore(out.get("wc_weighted_goals_for_per_match", pd.Series(index=out.index, dtype=float)))
        - 0.15 * zscore(out.get("wc_weighted_goals_against_per_match", pd.Series(index=out.index, dtype=float)))
        + 0.08 * zscore(out.get("wc_tournaments", pd.Series(index=out.index, dtype=float)))
        + 0.07 * zscore(out.get("wc_knockout_match_rate", pd.Series(index=out.index, dtype=float)))
    )
    last_seen = to_num(out.get("wc_last_seen_year", pd.Series(index=out.index, dtype=float))).fillna(0)
    out["wc_recency_score"] = ((last_seen - 2006) / (2022 - 2006)).clip(lower=0, upper=1)
    out["world_cup_prior_penalty"] = (1 - out["has_world_cup_prior"]) * -0.35
    out["macro_external_score_z"] = 0.0
    used_external = []
    for col in external_feature_cols:
        meta = EXTERNAL_NUMERIC_COLUMNS.get(col, {"direction": 1, "weight": 0.08})
        direction = float(meta.get("direction", 1))
        weight = float(meta.get("weight", 0.08))
        if direction == 0 or weight == 0:
            continue
        out[f"{col}_z_for_macro"] = direction * zscore(out[col])
        out["macro_external_score_z"] += weight * out[f"{col}_z_for_macro"]
        used_external.append(col)
    out["macro_prior_score"] = (
        out["wc_strength_core_z"]
        + 0.15 * out["wc_recency_score"]
        + out["world_cup_prior_penalty"]
        + out["macro_external_score_z"]
    )
    out["macro_prior_percentile"] = out["macro_prior_score"].rank(pct=True, method="average")
    out["macro_prior_band"] = pd.cut(
        out["macro_prior_percentile"],
        bins=[-0.01, 0.25, 0.50, 0.75, 1.01],
        labels=["LOW", "MID", "HIGH", "ELITE"],
    ).astype(str)
    out.attrs["used_external_features"] = used_external
    return out


def attach_side(launch: pd.DataFrame, teams: pd.DataFrame, side: str) -> pd.DataFrame:
    prefix = f"{side}_"
    keep = [
        "team_slug",
        "macro_prior_score",
        "macro_prior_percentile",
        "macro_prior_band",
        "has_world_cup_prior",
        "has_2022_prior",
        "external_prior_joined_flag",
        "wc_strength_core_z",
        "macro_external_score_z",
    ]
    side_teams = teams[[c for c in keep if c in teams.columns]].copy()
    side_teams = side_teams.rename(columns={c: f"{prefix}{c}" for c in side_teams.columns if c != "team_slug"})
    side_teams = side_teams.rename(columns={"team_slug": f"{prefix}team_slug"})
    out = launch.copy()
    out[f"{prefix}team_slug"] = out[f"{prefix}team_slug"].map(slugify)
    return out.merge(side_teams, on=f"{prefix}team_slug", how="left")


def build_fixture_matrix(launch: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    out = attach_side(launch, teams, "home")
    out = attach_side(out, teams, "away")
    home_host = to_num(out.get("home_is_host", pd.Series(0, index=out.index))).fillna(0)
    away_host = to_num(out.get("away_is_host", pd.Series(0, index=out.index))).fillna(0)
    out["macro_host_bonus"] = (home_host - away_host) * 0.18
    out["macro_score_diff"] = (
        to_num(out["home_macro_prior_score"]).fillna(0)
        - to_num(out["away_macro_prior_score"]).fillna(0)
        + out["macro_host_bonus"]
    )
    out["macro_prior_coverage_bucket"] = out.apply(
        lambda r: "BOTH_EXTERNAL"
        if r.get("home_external_prior_joined_flag", 0) == 1 and r.get("away_external_prior_joined_flag", 0) == 1
        else "BOTH_2022_PRIORS"
        if r.get("home_has_2022_prior", 0) == 1 and r.get("away_has_2022_prior", 0) == 1
        else "BOTH_WORLD_CUP_PRIORS"
        if r.get("home_has_world_cup_prior", 0) == 1 and r.get("away_has_world_cup_prior", 0) == 1
        else "ONE_SIDE_WORLD_CUP_PRIOR"
        if r.get("home_has_world_cup_prior", 0) == 1 or r.get("away_has_world_cup_prior", 0) == 1
        else "NO_WORLD_CUP_PRIOR",
        axis=1,
    )
    goal_cols = [
        "home_wc_weighted_goals_for_per_match",
        "home_wc_weighted_goals_against_per_match",
        "away_wc_weighted_goals_for_per_match",
        "away_wc_weighted_goals_against_per_match",
    ]
    goal_values = pd.concat([to_num(out[c]) for c in goal_cols if c in out.columns], ignore_index=True)
    global_goal_rate = float(goal_values.dropna().mean()) if not goal_values.dropna().empty else 1.16
    home_gf = to_num(out.get("home_wc_weighted_goals_for_per_match", pd.Series(index=out.index))).fillna(global_goal_rate)
    home_ga = to_num(out.get("home_wc_weighted_goals_against_per_match", pd.Series(index=out.index))).fillna(global_goal_rate)
    away_gf = to_num(out.get("away_wc_weighted_goals_for_per_match", pd.Series(index=out.index))).fillna(global_goal_rate)
    away_ga = to_num(out.get("away_wc_weighted_goals_against_per_match", pd.Series(index=out.index))).fillna(global_goal_rate)
    raw_home_goal = 0.60 * home_gf + 0.40 * away_ga
    raw_away_goal = 0.60 * away_gf + 0.40 * home_ga
    out["macro_home_xg_prior"] = (raw_home_goal * (1.0 + 0.13 * out["macro_score_diff"])).clip(0.35, 2.80)
    out["macro_away_xg_prior"] = (raw_away_goal * (1.0 - 0.13 * out["macro_score_diff"])).clip(0.35, 2.80)
    out["macro_total_goals_prior"] = out["macro_home_xg_prior"] + out["macro_away_xg_prior"]
    probs = out.apply(
        lambda r: goal_market_probs(r["macro_home_xg_prior"], r["macro_away_xg_prior"]),
        axis=1,
        result_type="expand",
    )
    out = pd.concat([out, probs], axis=1)
    out["macro_pick_ftr"] = out[["macro_prob_home", "macro_prob_draw", "macro_prob_away"]].idxmax(axis=1)
    out["macro_pick_ftr"] = out["macro_pick_ftr"].map(
        {
            "macro_prob_home": "HOME",
            "macro_prob_draw": "DRAW",
            "macro_prob_away": "AWAY",
        }
    )
    out["macro_pick_ou25"] = out["macro_prob_over25"].map(lambda p: "OVER_2_5" if p >= 0.50 else "UNDER_2_5")
    out["macro_pick_btts"] = out["macro_prob_btts_yes"].map(lambda p: "BTTS_YES" if p >= 0.50 else "BTTS_NO")
    out["macro_draw_stalemate_risk"] = (
        (out["macro_prob_draw"] >= 0.285)
        | ((out["macro_prob_home"] - out["macro_prob_away"]).abs() <= 0.055)
    ).astype(int)
    out["macro_ftr_confidence"] = out[["macro_prob_home", "macro_prob_draw", "macro_prob_away"]].max(axis=1)
    out["macro_ftr_risk_band"] = pd.cut(
        out["macro_ftr_confidence"],
        bins=[0.0, 0.39, 0.45, 0.52, 1.01],
        labels=["HIGH_RISK", "CAUTION", "LEAN", "STRONG"],
    ).astype(str)
    return out


def write_summary(
    outdir: Path,
    teams: pd.DataFrame,
    fixtures: pd.DataFrame,
    external_path: Path | None,
    used_external: list[str],
) -> None:
    coverage = fixtures["macro_prior_coverage_bucket"].value_counts(dropna=False).rename_axis("bucket").reset_index(name="fixtures")
    coverage.to_csv(outdir / "world_cup_2026_macro_prior_coverage.csv", index=False)
    coverage_md = ["| bucket | fixtures |", "|---|---:|"]
    coverage_md.extend(
        f"| {row.bucket} | {int(row.fixtures)} |" for row in coverage.itertuples()
    )

    probability_cols = [
        "api_fixture_id",
        "api_date",
        "api_round",
        "api_home_team_name",
        "api_away_team_name",
        "macro_prior_coverage_bucket",
        "home_macro_prior_band",
        "away_macro_prior_band",
        "macro_score_diff",
        "macro_home_xg_prior",
        "macro_away_xg_prior",
        "macro_prob_home",
        "macro_prob_draw",
        "macro_prob_away",
        "macro_prob_over25",
        "macro_prob_btts_yes",
        "macro_pick_ftr",
        "macro_ftr_confidence",
        "macro_ftr_risk_band",
        "macro_draw_stalemate_risk",
    ]
    fixtures[[c for c in probability_cols if c in fixtures.columns]].to_csv(
        outdir / "world_cup_2026_macro_probability_board.csv", index=False
    )

    top = fixtures.sort_values("macro_ftr_confidence", ascending=False).head(12)
    top_rows = [
        f"- {r.api_home_team_name} vs {r.api_away_team_name}: {r.macro_pick_ftr} "
        f"({r.macro_ftr_confidence:.3f}), coverage={r.macro_prior_coverage_bucket}"
        for r in top.itertuples()
    ]
    lines = [
        "# World Cup 2026 Macro Prior Engine",
        "",
        "Research-only Klement-style macro prior scaffold.",
        "",
        "## Outputs",
        "",
        f"- `{outdir / 'world_cup_team_macro_strength_2026.csv'}`",
        f"- `{outdir / 'world_cup_2026_macro_prior_fixture_matrix.csv'}`",
        f"- `{outdir / 'world_cup_2026_macro_probability_board.csv'}`",
        f"- `{outdir / 'world_cup_2026_macro_prior_coverage.csv'}`",
        "",
        "## Inputs",
        "",
        f"- Launch scaffold: `{DEFAULT_LAUNCH}`",
        f"- External priors: `{external_path}`" if external_path else "- External priors: not supplied",
        "",
        "## Coverage",
        "",
        *coverage_md,
        "",
        "## External Features Used",
        "",
        ", ".join(used_external) if used_external else "None yet. Add Kaggle/FIFA/Elo/squad-value CSV via `--external-priors`.",
        "",
        "## Top Macro FTR Leans",
        "",
        *top_rows,
        "",
        "## Research Boundary",
        "",
        "- This is not a deploy gate.",
        "- Full tournament simulation is held until group labels and knockout path are explicit.",
        "- Player, injury, and lineup intelligence should join after this macro prior layer.",
        "- All promoted features must remain timestamp-safe before model training or deployment.",
    ]
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-scaffold", type=Path, default=DEFAULT_LAUNCH)
    parser.add_argument("--external-priors", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.launch_scaffold.exists():
        raise SystemExit(f"Missing launch scaffold: {args.launch_scaffold}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    launch = pd.read_csv(args.launch_scaffold, low_memory=False)
    teams = extract_team_table(launch)
    external = load_external_priors(args.external_priors)
    teams, external_cols = add_external_priors(teams, external)
    teams = add_macro_scores(teams, external_cols)
    used_external = list(teams.attrs.get("used_external_features", []))

    fixtures = build_fixture_matrix(launch, teams)
    teams.to_csv(args.outdir / "world_cup_team_macro_strength_2026.csv", index=False)
    fixtures.to_csv(args.outdir / "world_cup_2026_macro_prior_fixture_matrix.csv", index=False)
    write_summary(args.outdir, teams, fixtures, args.external_priors, used_external)

    print(f"[ok] teams={len(teams)} fixtures={len(fixtures)}")
    print(f"[ok] wrote {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
