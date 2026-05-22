#!/usr/bin/env python3
"""Build FOULS_COMMITTED exact interaction proof tables.

Research-only beta/intelligence audit. This clones the fouled-player proof
shape for fouls committed: recent player foul rate x opponent/role foul
ecosystem x contact/referee context. It does not create priced player-prop
odds, deploy picks, slips, or production routing changes.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.api_football.team_name_map import normalize_team_name
from scripts.build_player_event_live_interaction_features import (
    aggregate_allowed,
    build_opponent_features,
    build_recent_features,
    load_historical_actuals,
    load_historical_roles,
    norm_text,
    role_group,
)


PLAYER_EVENTS_DIR = ROOT / "data_sources" / "api_football" / "features" / "player_events"
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-07" / "fouls_committed_exact_interaction_audit"

FOUNDATION_LEAGUES = (
    "Belgium_Pro",
    "Brazil_Serie_A",
    "England_Championship",
    "England_EFL_League_1",
    "England_Premier_League",
    "France_Ligue_1",
    "Germany_Bundesliga",
    "Italy_Serie_A",
    "Netherlands_Eredivisie",
    "Norway_Eliteserien",
    "Portugal_Liga",
    "Scotland_Premiership",
    "Spain_La_Liga",
    "USA_MLS",
)
DEFAULT_SEASONS = (2022, 2023, 2024)
RECENT_QUANTILES = (0.65, 0.75, 0.85)
OPPONENT_QUANTILES = (0.65, 0.75, 0.85)
CONTEXT_QUANTILES = (0.60, 0.70)
MIN_ROWS = 120
MIN_MONTH_ROWS = 15


@dataclass(frozen=True)
class MarketSpec:
    market: str
    display_market: str
    target_label: str
    hit_col: str
    threshold: int
    beta_core_min_hit: float
    research_ready_min_hit: float
    watch_min_hit: float


MARKETS = {
    "fouls_committed_ge1": MarketSpec(
        market="fouls_committed_ge1",
        display_market="Player Fouls Committed 0.5+",
        target_label="fouls_committed >= 1",
        hit_col="actual_fouls_committed_ge1",
        threshold=1,
        beta_core_min_hit=0.62,
        research_ready_min_hit=0.56,
        watch_min_hit=0.50,
    ),
    "fouls_committed_ge2": MarketSpec(
        market="fouls_committed_ge2",
        display_market="Player Fouls Committed 1.5+",
        target_label="fouls_committed >= 2",
        hit_col="actual_fouls_committed_ge2",
        threshold=2,
        beta_core_min_hit=0.38,
        research_ready_min_hit=0.34,
        watch_min_hit=0.30,
    ),
}

TARGET_COLS = [
    "fixture_key",
    "match_date",
    "competition",
    "league",
    "home_team_name",
    "away_team_name",
    "team_name",
    "player_name",
    "player_team_side",
    "position_group",
    "tactical_role",
    "expected_start_flag",
    "expected_minutes",
    "fixture_foul_density_score",
    "fixture_tackle_density_score",
    "fixture_midfield_grind_score",
    "fixture_wide_duel_score",
    "fixture_territorial_stress_score",
    "opponent_possession_projection",
    "ref_cards_per_match",
    "ref_foul_to_card_ratio",
    "match_stakes_score",
    "rivalry_flag",
    "fouls_per90",
    "tackles_per90",
    "dribbles_faced_per90",
]

RECENT_FEATURES = (
    "attacker_recent_fouls_committed_per90_l8",
    "attacker_recent_fouls_committed_per90_l5",
    "fouls_per90",
)

OPPONENT_FEATURES = (
    "opp_attack_allowed_role_fouls_committed_per_player_l10",
    "opp_attack_allowed_role_fouls_committed_per_match_l10",
    "opp_attack_allowed_attacker_any_fouls_committed_per_player_l10",
    "opp_attack_allowed_attacker_any_fouls_committed_per_match_l10",
)

CONTEXT_FEATURES = (
    "fixture_foul_density_score",
    "fixture_tackle_density_score",
    "fixture_midfield_grind_score",
    "fixture_wide_duel_score",
    "opponent_possession_projection",
    "ref_cards_per_match",
)


def num(values: Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def norm_team(value: Any, league_tag: Any = None) -> str:
    return norm_text(normalize_team_name(value, str(league_tag) if league_tag is not None else None))


def parse_csv_set(value: str, cast=str) -> set:
    return {cast(part.strip()) for part in value.split(",") if part.strip()}


def parse_tag(path: Path) -> tuple[str, int] | None:
    match = re.match(r"player_events_fixture_input__(.+)__(\d{4})\.csv$", path.name)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def read_csv_selected(path: Path, requested: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    usecols = [col for col in requested if col in header.columns]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    for col in requested:
        if col not in df.columns:
            df[col] = np.nan
    return df


def load_targets(input_dir: Path, leagues: set[str], seasons: set[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(input_dir.glob("player_events_fixture_input__*.csv")):
        parsed = parse_tag(path)
        if parsed is None:
            continue
        league_tag, season_tag = parsed
        if league_tag not in leagues or season_tag not in seasons:
            continue
        df = read_csv_selected(path, TARGET_COLS)
        if df.empty:
            continue
        df["league_tag"] = league_tag
        df["season_tag"] = season_tag
        df["source_file"] = path.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["match_date"] = pd.to_datetime(out["match_date"], errors="coerce")
    out["player_name_norm"] = out["player_name"].map(norm_text)
    out["team_name_norm"] = [norm_team(team, tag) for team, tag in zip(out["team_name"], out["league_tag"])]
    out["home_team_norm"] = [norm_team(team, tag) for team, tag in zip(out["home_team_name"], out["league_tag"])]
    out["away_team_norm"] = [norm_team(team, tag) for team, tag in zip(out["away_team_name"], out["league_tag"])]
    out["opponent_team_name"] = np.where(
        out["team_name_norm"].eq(out["home_team_norm"]),
        out["away_team_name"],
        out["home_team_name"],
    )
    out["opponent_team_norm"] = [norm_team(team, tag) for team, tag in zip(out["opponent_team_name"], out["league_tag"])]
    out["attack_role_group"] = [
        role_group(role, pos) for role, pos in zip(out.get("tactical_role", ""), out.get("position_group", ""))
    ]
    for col in TARGET_COLS:
        if col.endswith("_score") or col in {"expected_minutes", "fouls_per90", "tackles_per90", "dribbles_faced_per90", "ref_cards_per_match", "ref_foul_to_card_ratio", "opponent_possession_projection"}:
            out[col] = num(out[col])
    return out.dropna(subset=["fixture_key", "match_date", "team_name", "player_name"]).copy()


def join_actuals(targets: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "fixture_key",
        "team_name_norm",
        "player_name_norm",
        "minutes",
        "started_flag",
        "fouls_committed",
    ]
    actual_keep = actuals[[col for col in keep if col in actuals.columns]].copy()
    joined = targets.merge(actual_keep, on=["fixture_key", "team_name_norm", "player_name_norm"], how="left", suffixes=("", "_actual"))
    joined["actual_fouls_committed"] = num(joined.get("fouls_committed_actual", joined.get("fouls_committed", np.nan)))
    joined["actual_fouls_committed_ge1"] = joined["actual_fouls_committed"].ge(1).astype(float)
    joined["actual_fouls_committed_ge2"] = joined["actual_fouls_committed"].ge(2).astype(float)
    return joined.dropna(subset=["actual_fouls_committed"]).copy()


def feature_thresholds(
    df: pd.DataFrame,
    features: tuple[str, ...],
    quantiles: tuple[float, ...],
    min_value: float | None = None,
) -> list[tuple[str, float, float]]:
    rows: list[tuple[str, float, float]] = []
    for feature in features:
        if feature not in df.columns:
            continue
        series = num(df[feature]).replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            continue
        for q in quantiles:
            value = float(series.quantile(q))
            if min_value is not None and value <= min_value:
                continue
            if pd.notna(value):
                rows.append((feature, q, value))
    return rows


def summarize_cell(df: pd.DataFrame, mask: pd.Series, spec: MarketSpec) -> dict[str, Any]:
    group = df[mask].copy()
    rows = len(group)
    hits = float(group[spec.hit_col].sum()) if rows else 0.0
    hit_rate = hits / rows if rows else np.nan
    monthly = (
        group.assign(match_month=group["match_date"].dt.strftime("%Y-%m"))
        .groupby("match_month", dropna=False)
        .agg(rows=("fixture_key", "size"), hit_rate=(spec.hit_col, "mean"))
        .reset_index()
    )
    stable = monthly[monthly["rows"].ge(MIN_MONTH_ROWS)]
    stable_share = float(stable["hit_rate"].ge(spec.watch_min_hit).mean()) if not stable.empty else np.nan
    return {
        "rows": int(rows),
        "hits": int(hits),
        "hit_rate": float(hit_rate) if rows else np.nan,
        "months": int(monthly["match_month"].nunique()) if not monthly.empty else 0,
        "stable_months": int(len(stable)),
        "stable_month_share": stable_share,
    }


def label_cell(spec: MarketSpec, hit_rate: float, rows: int, stable_share: float) -> str:
    if rows < MIN_ROWS or pd.isna(hit_rate):
        return "DO_NOT_USE"
    stability_ok = pd.notna(stable_share) and stable_share >= 0.70
    if hit_rate >= spec.beta_core_min_hit and stability_ok:
        return "BETA_CORE"
    if hit_rate >= spec.research_ready_min_hit and (stability_ok or rows >= MIN_ROWS * 2):
        return "RESEARCH_READY"
    if hit_rate >= spec.watch_min_hit:
        return "WATCH"
    return "DO_NOT_USE"


def build_candidate_cells(df: pd.DataFrame) -> pd.DataFrame:
    recent_thresholds = feature_thresholds(df, RECENT_FEATURES, RECENT_QUANTILES, min_value=0.0)
    opponent_thresholds = feature_thresholds(df, OPPONENT_FEATURES, OPPONENT_QUANTILES, min_value=0.0)
    context_thresholds = feature_thresholds(df, CONTEXT_FEATURES, CONTEXT_QUANTILES, min_value=0.0)
    rows: list[dict[str, Any]] = []
    for spec in MARKETS.values():
        baseline = float(df[spec.hit_col].mean()) if spec.hit_col in df.columns and not df.empty else np.nan
        for recent_feature, recent_q, recent_t in recent_thresholds:
            recent_mask = num(df[recent_feature]).ge(recent_t)
            for opponent_feature, opponent_q, opponent_t in opponent_thresholds:
                base_mask = recent_mask & num(df[opponent_feature]).ge(opponent_t)
                if int(base_mask.sum()) < MIN_ROWS:
                    continue
                base_summary = summarize_cell(df, base_mask, spec)
                rows.append(
                    {
                        "market": spec.market,
                        "display_market": spec.display_market,
                        "cell_type": "RECENT_X_OPPONENT",
                        "recent_feature": recent_feature,
                        "recent_quantile": recent_q,
                        "recent_threshold": recent_t,
                        "opponent_feature": opponent_feature,
                        "opponent_quantile": opponent_q,
                        "opponent_threshold": opponent_t,
                        "context_feature": "",
                        "context_quantile": np.nan,
                        "context_threshold": np.nan,
                        "baseline_hit_rate": baseline,
                        **base_summary,
                    }
                )
                for context_feature, context_q, context_t in context_thresholds:
                    mask = base_mask & num(df[context_feature]).ge(context_t)
                    if int(mask.sum()) < MIN_ROWS:
                        continue
                    summary = summarize_cell(df, mask, spec)
                    rows.append(
                        {
                            "market": spec.market,
                            "display_market": spec.display_market,
                            "cell_type": "RECENT_X_OPPONENT_X_CONTEXT",
                            "recent_feature": recent_feature,
                            "recent_quantile": recent_q,
                            "recent_threshold": recent_t,
                            "opponent_feature": opponent_feature,
                            "opponent_quantile": opponent_q,
                            "opponent_threshold": opponent_t,
                            "context_feature": context_feature,
                            "context_quantile": context_q,
                            "context_threshold": context_t,
                            "baseline_hit_rate": baseline,
                            **summary,
                        }
                    )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["lift_vs_baseline"] = out["hit_rate"] - out["baseline_hit_rate"]
    out["beta_label"] = [
        label_cell(MARKETS[row["market"]], row["hit_rate"], int(row["rows"]), row["stable_month_share"])
        for _, row in out.iterrows()
    ]
    label_rank = {"BETA_CORE": 0, "RESEARCH_READY": 1, "WATCH": 2, "DO_NOT_USE": 3}
    out["_rank"] = out["beta_label"].map(label_rank).fillna(9)
    return out.sort_values(["_rank", "hit_rate", "rows"], ascending=[True, False, False]).drop(columns="_rank")


def group_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in MARKETS.values():
        for key, group in df.groupby(group_cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            rows.append(
                {
                    "market": spec.market,
                    **dict(zip(group_cols, key)),
                    "rows": int(len(group)),
                    "hit_rate": float(group[spec.hit_col].mean()) if len(group) else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["market", "rows"], ascending=[True, False])


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


def write_report(outdir: Path, df: pd.DataFrame, candidates: pd.DataFrame, league: pd.DataFrame, role: pd.DataFrame) -> None:
    lines = [
        "# Fouls Committed Exact Interaction Audit",
        "",
        "Research-only proof audit for fouls committed player-event intelligence.",
        "",
        "## Safety",
        "- No priced player-prop odds.",
        "- No deploy routing, tiers, slips, or production rulebook changes.",
        "- Candidate cells are proof labels only and require live shadow outcome tracking before dashboard prominence.",
        "",
        "## Overall",
        f"- proof rows: `{len(df)}`",
        f"- fixtures: `{df['fixture_key'].nunique() if not df.empty else 0}`",
        f"- players: `{df['player_name'].nunique() if not df.empty else 0}`",
        f"- leagues: `{df['league'].nunique() if 'league' in df.columns and not df.empty else 0}`",
        "",
        "## Top Candidate Cells",
        markdown_table(
            candidates[
                [
                    col
                    for col in [
                        "market",
                        "beta_label",
                        "cell_type",
                        "rows",
                        "hit_rate",
                        "baseline_hit_rate",
                        "lift_vs_baseline",
                        "stable_month_share",
                        "recent_feature",
                        "opponent_feature",
                        "context_feature",
                    ]
                    if col in candidates.columns
                ]
            ],
            max_rows=40,
        ),
        "",
        "## League Baseline",
        markdown_table(league, max_rows=80),
        "",
        "## Role Baseline",
        markdown_table(role, max_rows=80),
    ]
    (outdir / "FOULS_COMMITTED_EXACT_INTERACTION_AUDIT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PLAYER_EVENTS_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--leagues", default=",".join(FOUNDATION_LEAGUES))
    parser.add_argument("--seasons", default=",".join(str(season) for season in DEFAULT_SEASONS))
    parser.add_argument(
        "--max-target-rows",
        type=int,
        default=0,
        help="Optional smoke limiter for first-pass QA. 0 means no limit.",
    )
    args = parser.parse_args()

    leagues = parse_csv_set(args.leagues, str)
    seasons = parse_csv_set(args.seasons, int)
    args.outdir.mkdir(parents=True, exist_ok=True)

    targets = load_targets(args.input_dir, leagues, seasons)
    if targets.empty:
        raise SystemExit("No player-event fixture input rows found.")
    if args.max_target_rows > 0:
        targets = targets.sort_values(["league_tag", "season_tag", "match_date", "fixture_key"]).head(args.max_target_rows)
    actuals = load_historical_actuals(leagues, seasons)
    if actuals.empty:
        raise SystemExit("No historical API player actuals found.")

    history = load_historical_roles(targets, actuals)
    recent = build_recent_features(targets, history)
    opponent = build_opponent_features(targets, aggregate_allowed(history))
    proof = targets.merge(
        recent.drop(columns=["match_date"], errors="ignore"),
        on=["fixture_key", "league_tag", "season_tag", "team_name", "player_name", "player_team_side"],
        how="left",
    ).merge(
        opponent.drop(columns=["match_date", "opponent_team_name", "tactical_role"], errors="ignore"),
        on=["fixture_key", "league_tag", "season_tag", "team_name", "player_name"],
        how="left",
        suffixes=("", "_opp"),
    )
    proof = join_actuals(proof, actuals)
    candidates = build_candidate_cells(proof)
    league = group_summary(proof, ["league"])
    role = group_summary(proof, ["tactical_role"])

    proof.to_csv(args.outdir / "fouls_committed_exact_interaction_proof_rows.csv", index=False)
    candidates.to_csv(args.outdir / "fouls_committed_exact_interaction_candidate_cells.csv", index=False)
    league.to_csv(args.outdir / "fouls_committed_exact_interaction_league_summary.csv", index=False)
    role.to_csv(args.outdir / "fouls_committed_exact_interaction_role_summary.csv", index=False)
    write_report(args.outdir, proof, candidates, league, role)

    print(f"WROTE {args.outdir}")
    print(f"proof_rows={len(proof)} candidate_cells={len(candidates)}")
    if not candidates.empty:
        print(candidates.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
