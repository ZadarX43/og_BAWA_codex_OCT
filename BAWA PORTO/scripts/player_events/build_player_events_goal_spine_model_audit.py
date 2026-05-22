#!/usr/bin/env python3
"""
Audit whether stable goal-spine features improve player-event probabilities.

Phase 1 scope is tackles only:
- start from the leak-safe tackles NB proof predictions
- join restored goal-spine signals at fixture level
- run chronological out-of-fold calibration/stacking variants
- compare calibration/discrimination against raw NB and cohort baseline

Research only. This script does not train production artifacts or alter deploy.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TACKLES = ROOT / "reports" / "player_events" / "proof" / "tackles_nb_proof_predictions.csv"
DEFAULT_GOAL_SPINE = ROOT / "reports" / "2026-05-05" / "dominance_overlay_walkforward_audit_v2_fullweekly" / "dominance_overlay_walkforward_scored.csv"
DEFAULT_OUTDIR = ROOT / "reports" / "2026-05-06" / "player_events_goal_spine_model_audit"

EPS = 1e-6

GOAL_SPINE_COLS = [
    "p_meta_ou25",
    "cs_mass_over25",
    "mass_4plus_goals",
    "p_meta_btts",
    "cs_mass_btts_yes",
    "p00_est",
    "p_meta_ftr",
    "pick_side_margin_top3",
    "pick_side_mass_top3",
    "ftr_margin",
    "p_home_ge2",
    "p_away_ge2",
    "p_home_ge3",
    "p_away_ge3",
    "hw_hge2_combo_prob",
    "aw_age2_combo_prob",
]

LEGACY_GOAL_ENV_COLS = [
    "og_pre_match_xg_home",
    "og_pre_match_xg_away",
    "og_xg_total",
    "og_btts_pre",
    "og_over25_pre",
    "og_snap_over25_avg",
    "og_power_gap_abs",
    "og_balance_score",
    "og_goal_environment_score",
    "og_battle_on_score",
]

MATCH_STYLE_COLS = [
    "fixture_foul_density_score",
    "fixture_tackle_density_score",
    "fixture_midfield_grind_score",
    "fixture_wide_duel_score",
    "fixture_attack_pressure_score",
    "fixture_corner_pressure_score",
    "fixture_territorial_stress_score",
    "formation_pressure_score",
    "match_stakes_score",
    "ref_cards_per_match",
    "ref_foul_to_card_ratio",
    "ref_dissent_strictness",
    "ref_timewasting_strictness",
]

PLAYER_LAYER_COLS = [
    "nb_p_ge2",
    "player_tackles_per90_l5_shrunk",
    "player_tackles_per90_l10_shrunk",
    "player_tackles_per90_season_shrunk",
    "tackles_per90",
    "interceptions_per90",
    "duels_total_per90",
    "expected_minutes_proof",
    "actual_minutes",
    "opp_tackles_allowed_def_l10",
    "opp_tackles_allowed_mid_l10",
    "opp_tackles_allowed_pos_l10",
    "opp_possession_share_l10",
    "opp_dribble_attempts_l10",
    "player_form_rating_l5",
    "player_quality_score_l5",
    "starting_xi_quality_edge",
    "team_power_edge",
]

CATEGORICAL_COLS = [
    "league_tag",
    "position_group",
    "player_team_side",
    "tactical_role",
    "formation_matchup_label",
    "fixture_style_label",
    "fixture_attacking_style_label",
]


def norm_text(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    aliases = {
        "cf": "",
        "fc": "",
        "sc": "",
        "afc": "",
        "cd": "",
        "ud": "",
        "real betis balompie": "real betis",
        "atletico madrid": "atletico madrid",
        "athletic club bilbao": "athletic club",
        "granada cf": "granada",
    }
    text = " ".join(aliases.get(part, part) for part in text.split())
    return re.sub(r"\s+", " ", text).strip()


def make_join_key(df: pd.DataFrame) -> pd.Series:
    date = pd.to_datetime(df["match_date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    home = df["home_team_name"].map(norm_text)
    away = df["away_team_name"].map(norm_text)
    return date + "__" + home + "__" + away


def num(series: Any) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def logit(series: pd.Series) -> pd.Series:
    values = num(series).clip(EPS, 1 - EPS)
    return np.log(values / (1 - values))


def load_goal_spine(path: Path) -> pd.DataFrame:
    usecols = ["fixture_key", "match_date", "league", "home_team_name", "away_team_name", "market"]
    # Read header first because older files may not have every target column.
    header = pd.read_csv(path, nrows=0)
    available = [c for c in usecols + GOAL_SPINE_COLS if c in header.columns]
    raw = pd.read_csv(path, usecols=available, low_memory=False)
    raw["_join_key"] = make_join_key(raw)
    for col in GOAL_SPINE_COLS:
        if col not in raw.columns:
            raw[col] = np.nan
        raw[col] = num(raw[col])

    aggregations = {col: "max" for col in GOAL_SPINE_COLS}
    aggregations.update(
        {
            "fixture_key": "first",
            "match_date": "first",
            "league": "first",
            "home_team_name": "first",
            "away_team_name": "first",
            "market": lambda s: "|".join(sorted(set(s.dropna().astype(str)))),
        }
    )
    out = raw.groupby("_join_key", dropna=False).agg(aggregations).reset_index()
    out = out.rename(
        columns={
            "fixture_key": "goal_fixture_key",
            "league": "goal_league",
            "market": "goal_markets_present",
        }
    )
    out["goal_spine_feature_count"] = out[GOAL_SPINE_COLS].notna().sum(axis=1)
    return out


def load_tackles(path: Path, goal_spine: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df["_join_key"] = make_join_key(df)
    joined = df.merge(goal_spine, on="_join_key", how="left", suffixes=("", "_goal"))
    joined["goal_spine_matched"] = joined["goal_fixture_key"].notna().astype(int)
    joined["nb_logit_ge2"] = logit(joined["nb_p_ge2"])
    joined["cohort_logit_ge2"] = logit(joined["cohort_p_ge2"])
    joined["eval_month"] = joined["match_date"].dt.to_period("M").astype(str)
    return joined.sort_values(["match_date", "fixture_key", "team_name", "player_name"]).reset_index(drop=True)


def ece_score(y_true: pd.Series, y_prob: pd.Series, bins: int = 10) -> float:
    work = pd.DataFrame({"y": y_true, "p": y_prob}).dropna()
    if work.empty:
        return np.nan
    work["bin"] = pd.cut(work["p"].clip(0, 1), np.linspace(0, 1, bins + 1), include_lowest=True)
    total = len(work)
    ece = 0.0
    for _, group in work.groupby("bin", observed=False):
        if group.empty:
            continue
        ece += len(group) / total * abs(group["p"].mean() - group["y"].mean())
    return float(ece)


def top_decile_precision(y_true: pd.Series, y_prob: pd.Series) -> float:
    work = pd.DataFrame({"y": y_true, "p": y_prob}).dropna()
    if work.empty:
        return np.nan
    n = max(1, math.ceil(len(work) * 0.10))
    return float(work.nlargest(n, "p")["y"].mean())


def metric_row(name: str, y_true: pd.Series, y_prob: pd.Series) -> dict[str, Any]:
    work = pd.DataFrame({"y": y_true, "p": y_prob}).dropna()
    if work.empty or work["y"].nunique() < 2:
        return {
            "variant": name,
            "rows": len(work),
            "coverage": 0.0,
            "brier": np.nan,
            "logloss": np.nan,
            "ece_10bin": np.nan,
            "top_decile_precision": np.nan,
        }
    return {
        "variant": name,
        "rows": int(len(work)),
        "coverage": float(len(work) / len(y_true)),
        "brier": float(brier_score_loss(work["y"], work["p"].clip(EPS, 1 - EPS))),
        "logloss": float(log_loss(work["y"], work["p"].clip(EPS, 1 - EPS))),
        "ece_10bin": ece_score(work["y"], work["p"]),
        "top_decile_precision": top_decile_precision(work["y"], work["p"]),
    }


def build_feature_matrix(df: pd.DataFrame, feature_cols: list[str], categorical_cols: list[str]) -> pd.DataFrame:
    pieces = []
    numeric_cols = [c for c in feature_cols if c in df.columns]
    if numeric_cols:
        pieces.append(df[numeric_cols].apply(num))
    cats = [c for c in categorical_cols if c in df.columns and c in feature_cols]
    if cats:
        cat_df = pd.get_dummies(df[cats].fillna("UNKNOWN").astype(str), prefix=cats, dummy_na=False)
        pieces.append(cat_df)
    if not pieces:
        return pd.DataFrame(index=df.index)
    return pd.concat(pieces, axis=1)


def temporal_logistic_oof(
    df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str] | None = None,
    min_train_rows: int = 2500,
) -> pd.Series:
    categorical_cols = categorical_cols or []
    X_all = build_feature_matrix(df, feature_cols, categorical_cols)
    y = df["actual_hit_ge2"].astype(int)
    preds = pd.Series(np.nan, index=df.index, dtype=float)
    months = sorted(df["eval_month"].dropna().unique())

    for month in months:
        test_idx = df.index[df["eval_month"].eq(month)]
        train_idx = df.index[df["match_date"].lt(df.loc[test_idx, "match_date"].min())]
        train_idx = train_idx[y.loc[train_idx].notna()]
        if len(train_idx) < min_train_rows or y.loc[train_idx].nunique() < 2:
            continue
        x_train = X_all.loc[train_idx]
        x_test = X_all.loc[test_idx]
        non_empty_cols = x_train.columns[x_train.notna().any()]
        x_train = x_train[non_empty_cols]
        x_test = x_test[non_empty_cols]
        if x_train.shape[1] == 0:
            continue
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
                ("logit", LogisticRegression(max_iter=1000, C=0.5, solver="liblinear")),
            ]
        )
        try:
            model.fit(x_train, y.loc[train_idx])
            preds.loc[test_idx] = model.predict_proba(x_test)[:, 1]
        except Exception:
            continue
    return preds


def temporal_isotonic_oof(df: pd.DataFrame, prob_col: str, min_train_rows: int = 2500) -> pd.Series:
    y = df["actual_hit_ge2"].astype(int)
    p = num(df[prob_col]).clip(EPS, 1 - EPS)
    preds = pd.Series(np.nan, index=df.index, dtype=float)
    months = sorted(df["eval_month"].dropna().unique())
    for month in months:
        test_idx = df.index[df["eval_month"].eq(month)]
        first_test_date = df.loc[test_idx, "match_date"].min()
        train_idx = df.index[df["match_date"].lt(first_test_date)]
        train = pd.DataFrame({"p": p.loc[train_idx], "y": y.loc[train_idx]}).dropna()
        if len(train) < min_train_rows or train["y"].nunique() < 2:
            continue
        try:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(train["p"], train["y"])
            preds.loc[test_idx] = iso.predict(p.loc[test_idx])
        except Exception:
            continue
    return preds.clip(0, 1)


def family_coverage(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    families = {
        "stable_goal_spine": GOAL_SPINE_COLS,
        "legacy_goal_environment": LEGACY_GOAL_ENV_COLS,
        "match_style": MATCH_STYLE_COLS,
        "player_layer": PLAYER_LAYER_COLS,
    }
    for family, cols in families.items():
        available = [c for c in cols if c in joined.columns]
        rows.append(
            {
                "feature_family": family,
                "requested_features": len(cols),
                "available_features": len(available),
                "rows_with_any_feature": int(joined[available].notna().any(axis=1).sum()) if available else 0,
                "rows_with_all_features": int(joined[available].notna().all(axis=1).sum()) if available else 0,
                "coverage_any": float(joined[available].notna().any(axis=1).mean()) if available else 0.0,
                "coverage_all": float(joined[available].notna().all(axis=1).mean()) if available else 0.0,
                "available_feature_names": "|".join(available),
            }
        )
    return pd.DataFrame(rows)


def feature_screen(joined: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    y = joined["actual_hit_ge2"].astype(float)
    for col in cols:
        if col not in joined.columns:
            continue
        x = num(joined[col])
        valid = pd.DataFrame({"x": x, "y": y}).dropna()
        if len(valid) < 100:
            continue
        q80 = valid["x"].quantile(0.80)
        q20 = valid["x"].quantile(0.20)
        top = valid[valid["x"].ge(q80)]["y"].mean()
        bottom = valid[valid["x"].le(q20)]["y"].mean()
        rows.append(
            {
                "feature": col,
                "rows": int(len(valid)),
                "coverage": float(len(valid) / len(joined)),
                "corr_with_hit": float(valid["x"].corr(valid["y"])) if valid["x"].nunique() > 1 else np.nan,
                "bottom_quintile_hit": float(bottom),
                "top_quintile_hit": float(top),
                "top_minus_bottom_hit": float(top - bottom),
            }
        )
    return pd.DataFrame(rows).sort_values("top_minus_bottom_hit", ascending=False)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    if max_rows is not None:
        df = df.head(max_rows)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                value = round(value, 6)
            if pd.isna(value):
                value = ""
            values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    out_path: Path,
    joined: pd.DataFrame,
    metrics_all: pd.DataFrame,
    metrics_oof_common: pd.DataFrame,
    metrics_goal_matched: pd.DataFrame,
    coverage: pd.DataFrame,
    screen: pd.DataFrame,
    goal_spine_path: Path,
) -> None:
    matched = int(joined["goal_spine_matched"].sum())
    lines = [
        "# Player Events Goal-Spine Model Audit",
        "",
        "Research-only audit: tackles Phase 1 probability stack with restored goal-spine context.",
        "",
        "## Safety",
        "- No production model artifact written.",
        "- No deploy routing or tiers changed.",
        "- Player events remain beta/intelligence until calibration proof clears.",
        "",
        "## Join Summary",
        f"- tackles proof rows: `{len(joined)}`",
        f"- goal-spine matched rows: `{matched}`",
        f"- goal-spine match rate: `{matched / len(joined):.2%}`",
        f"- goal-spine source: `{goal_spine_path}`",
        "",
        "## Variant Metrics: Full Coverage View",
        markdown_table(metrics_all),
        "",
        "## Variant Metrics: Common OOF Rows",
        markdown_table(metrics_oof_common),
        "",
        "## Variant Metrics: Goal-Spine Matched Rows",
        markdown_table(metrics_goal_matched),
        "",
        "## Feature-Family Coverage",
        markdown_table(coverage),
        "",
        "## Goal-Spine Feature Screen",
        markdown_table(screen, max_rows=25),
        "",
        "## Read",
        "- `RAW_NB` is the original leak-safe tackles proof probability.",
        "- `NB_ISOTONIC_OOF` tests disciplined chronological calibration only.",
        "- `NB_PLUS_STABLE_GOAL_SPINE` tests whether restored Phase 8H goal-spine features add signal.",
        "- Treat positive movement in Brier/logloss/ECE/top-decile as a reason to promote this audit into the next proof loop.",
        "",
        "## Next Markets",
        "- Clone this exact audit shape for `fouls_committed` after tackles calibration is settled.",
        "- Then clone for `shots`, then `shots_on_target`.",
        "- Keep cards separate as a hazard model, not a count-rate copy.",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tackles-predictions", type=Path, default=DEFAULT_TACKLES)
    parser.add_argument("--goal-spine", type=Path, default=DEFAULT_GOAL_SPINE)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--min-train-rows", type=int, default=2500)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    goal_spine = load_goal_spine(args.goal_spine)
    joined = load_tackles(args.tackles_predictions, goal_spine)

    variants: dict[str, pd.Series] = {
        "RAW_NB": num(joined["nb_p_ge2"]),
        "COHORT_BASELINE": num(joined["cohort_p_ge2"]),
        "NB_ISOTONIC_OOF": temporal_isotonic_oof(joined, "nb_p_ge2", args.min_train_rows),
        "NB_LOGISTIC_CAL_OOF": temporal_logistic_oof(joined, ["nb_logit_ge2"], min_train_rows=args.min_train_rows),
        "NB_PLUS_LEGACY_GOAL_ENV": temporal_logistic_oof(
            joined,
            ["nb_logit_ge2"] + LEGACY_GOAL_ENV_COLS,
            min_train_rows=args.min_train_rows,
        ),
        "NB_PLUS_STABLE_GOAL_SPINE": temporal_logistic_oof(
            joined,
            ["nb_logit_ge2"] + GOAL_SPINE_COLS,
            min_train_rows=args.min_train_rows,
        ),
        "NB_PLUS_MATCH_STYLE": temporal_logistic_oof(
            joined,
            ["nb_logit_ge2"] + MATCH_STYLE_COLS,
            min_train_rows=args.min_train_rows,
        ),
        "NB_PLUS_ALL_CONTEXT": temporal_logistic_oof(
            joined,
            ["nb_logit_ge2"] + GOAL_SPINE_COLS + LEGACY_GOAL_ENV_COLS + MATCH_STYLE_COLS + PLAYER_LAYER_COLS + CATEGORICAL_COLS,
            categorical_cols=CATEGORICAL_COLS,
            min_train_rows=args.min_train_rows,
        ),
    }

    metrics_all = pd.DataFrame(
        [metric_row(name, joined["actual_hit_ge2"], pred) for name, pred in variants.items()]
    ).sort_values(["coverage", "brier"], ascending=[False, True])

    scored = joined.copy()
    for name, pred in variants.items():
        scored[f"pred_{name.lower()}"] = pred

    oof_mask = variants["NB_LOGISTIC_CAL_OOF"].notna()
    metrics_oof_common = pd.DataFrame(
        [metric_row(name, joined.loc[oof_mask, "actual_hit_ge2"], pred.loc[oof_mask]) for name, pred in variants.items()]
    ).sort_values("brier", ascending=True)

    goal_mask = joined["goal_spine_matched"].eq(1)
    metrics_goal_matched = pd.DataFrame(
        [metric_row(name, joined.loc[goal_mask, "actual_hit_ge2"], pred.loc[goal_mask]) for name, pred in variants.items()]
    ).sort_values(["coverage", "brier"], ascending=[False, True])

    coverage = family_coverage(joined)
    screen = feature_screen(joined, GOAL_SPINE_COLS + LEGACY_GOAL_ENV_COLS + MATCH_STYLE_COLS)

    scored_path = args.outdir / "player_events_goal_spine_tackles_joined_scored.csv"
    metrics_path = args.outdir / "player_events_goal_spine_tackles_variant_metrics.csv"
    metrics_oof_path = args.outdir / "player_events_goal_spine_tackles_variant_metrics_common_oof.csv"
    metrics_matched_path = args.outdir / "player_events_goal_spine_tackles_variant_metrics_goal_matched.csv"
    coverage_path = args.outdir / "player_events_goal_spine_feature_coverage.csv"
    screen_path = args.outdir / "player_events_goal_spine_feature_screen.csv"
    report_path = args.outdir / "PLAYER_EVENTS_GOAL_SPINE_MODEL_AUDIT.md"

    scored.to_csv(scored_path, index=False)
    metrics_all.to_csv(metrics_path, index=False)
    metrics_oof_common.to_csv(metrics_oof_path, index=False)
    metrics_goal_matched.to_csv(metrics_matched_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    screen.to_csv(screen_path, index=False)
    write_report(report_path, joined, metrics_all, metrics_oof_common, metrics_goal_matched, coverage, screen, args.goal_spine)

    print(f"WROTE {args.outdir}")
    print(f"rows={len(joined)} goal_spine_matches={int(joined['goal_spine_matched'].sum())}")
    print(metrics_all.to_string(index=False))


if __name__ == "__main__":
    main()
