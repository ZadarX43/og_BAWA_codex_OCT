#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SCRIPT_VERSION = "walkforward_cs_bucket_ablation_v1"
TIME_CANDIDATES: Tuple[str, ...] = (
    "match_date",
    "date_GMT",
    "date",
    "timestamp",
)

APPROVED_BTTS_CS_FEATURES: Tuple[str, ...] = (
    "cs_mass_btts_yes",
    "cs_mass_btts_no",
    "cs_mass_nil_nil",
    "cs_mass_one_side_nil",
    "cs_mass_1_1",
    "cs_mass_2_1_or_1_2",
    "cs_btts_yes_topk_share",
)

APPROVED_OU25_CS_FEATURES: Tuple[str, ...] = (
    "cs_mass_over25",
    "cs_mass_under25",
    "cs_mass_exact_2_goals",
    "cs_mass_exact_3_goals",
    "cs_mass_4plus_goals",
    "cs_topk_over25_share",
)

# These are the main product lanes we currently care about.
# BTTS is promotion-focused. OU25 is experimental / overlay-only for now.
MARKET_PROMOTION_POLICY: Dict[str, Dict[str, object]] = {
    "btts": {
        "mode": "promotion_candidate",
        "approved_cs_cols": list(APPROVED_BTTS_CS_FEATURES),
    },
    "ou25": {
        "mode": "experimental_overlay",
        "approved_cs_cols": list(APPROVED_OU25_CS_FEATURES),
    },
}

# Core baseline columns should stay aligned with walkforward_cs_bucket_promotion_test.py
BASELINE_BY_MARKET: Dict[str, List[str]] = {
    "btts": [
        "prob_btts",
        "prob_btts_v2",
        "odds_btts_yes",
        "odds_btts_no",
        "od_yes",
        "od_no",
        "bookie_implied",
        "bookie_implied_novig",
        "btts_rate_5_home",
        "btts_rate_5_away",
        "scored_rate_5_home",
        "scored_rate_5_away",
        "conceded_rate_5_home",
        "conceded_rate_5_away",
        "clean_sheet_rate_5_home",
        "clean_sheet_rate_5_away",
        "xg_for_avg_5_home",
        "xg_for_avg_5_away",
        "xg_against_avg_5_home",
        "xg_against_avg_5_away",
        "Home Team Pre-Match xG",
        "Away Team Pre-Match xG",
        "pre_match_xg_home",
        "pre_match_xg_away",
        "exp_goals_sum",
        "bookie_lambda_total_fit",
    ],
    "ou25": [
        "prob_over25",
        "prob_over25_v2",
        "odds_ft_over25",
        "odds_ft_under25",
        "od_over",
        "od_under",
        "bookie_implied",
        "bookie_implied_novig",
        "over25_rate_5_home",
        "over25_rate_5_away",
        "under25_rate_5_home",
        "under25_rate_5_away",
        "goaliness_avg_5_home",
        "goaliness_avg_5_away",
        "xg_for_avg_5_home",
        "xg_for_avg_5_away",
        "xg_against_avg_5_home",
        "xg_against_avg_5_away",
        "Home Team Pre-Match xG",
        "Away Team Pre-Match xG",
        "pre_match_xg_home",
        "pre_match_xg_away",
        "exp_goals_sum",
        "bookie_lambda_total_fit",
        "over_25_percentage_pre_match",
        "p_over25_novig",
        "p_under25_novig",
    ],
}

THRESHOLDS: Tuple[float, ...] = (0.55, 0.60, 0.65, 0.70)

# We want ablations that test exact features, leave-one-out, logical groupings,
# and broader bucket-mass structures rather than only exact-score atoms.
GROUP_DEFS: Dict[str, Dict[str, Sequence[str]]] = {
    "btts": {
        "BTTS_FULL_APPROVED": APPROVED_BTTS_CS_FEATURES,
        "BTTS_TOPK_STRUCTURAL": (
            "cs_mass_btts_yes",
            "cs_mass_btts_no",
            "cs_btts_yes_topk_share",
        ),
        "BTTS_EXACT_SCORE_ATOMS": (
            "cs_mass_nil_nil",
            "cs_mass_1_1",
            "cs_mass_2_1_or_1_2",
        ),
        "BTTS_NO_NIL_BUCKETS": (
            "cs_mass_btts_yes",
            "cs_mass_one_side_nil",
            "cs_mass_1_1",
            "cs_mass_2_1_or_1_2",
            "cs_btts_yes_topk_share",
        ),
        "BTTS_NON_EXACT_GENERAL_MASS": (
            "cs_mass_btts_yes",
            "cs_mass_btts_no",
            "cs_mass_one_side_nil",
            "cs_btts_yes_topk_share",
        ),
        "BTTS_MUTUAL_SCORING_SHAPE": (
            "cs_mass_btts_yes",
            "cs_mass_1_1",
            "cs_mass_2_1_or_1_2",
            "cs_btts_yes_topk_share",
        ),
    },
    "ou25": {
        "OU25_FULL_APPROVED": APPROVED_OU25_CS_FEATURES,
        "OU25_TOTAL_GOAL_MASS": (
            "cs_mass_over25",
            "cs_mass_under25",
            "cs_mass_exact_2_goals",
            "cs_mass_exact_3_goals",
            "cs_mass_4plus_goals",
            "cs_topk_over25_share",
        ),
        "OU25_ESCALATION_ONLY": (
            "cs_mass_over25",
            "cs_mass_exact_3_goals",
            "cs_mass_4plus_goals",
            "cs_topk_over25_share",
        ),
        "OU25_BALANCED_MASS": (
            "cs_mass_over25",
            "cs_mass_under25",
            "cs_mass_exact_2_goals",
            "cs_mass_exact_3_goals",
        ),
        "OU25_HIGH_SCORING_PRESSURE": (
            "cs_mass_exact_3_goals",
            "cs_mass_4plus_goals",
            "cs_topk_over25_share",
        ),
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward CS bucket ablation runner for BTTS / OU25 main product evaluation."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing scored cs-bucket files")
    parser.add_argument(
        "--glob",
        default="*/*__SCORED__cs_market_buckets.csv",
        help="Glob used under --input-dir",
    )
    parser.add_argument("--market", required=True, choices=["btts", "ou25"])
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--detail-csv", required=True)
    parser.add_argument(
        "--top-n-per-league",
        type=int,
        default=8,
        help="How many strongest ablations to keep per league in the printed summary",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=300,
        help="Minimum completed rows needed to evaluate a file",
    )
    return parser.parse_args()


def _discover_inputs(input_dir: str, glob_pat: str) -> List[Path]:
    root = Path(input_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")
    files = sorted([p.resolve() for p in root.glob(glob_pat) if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No files matched glob='{glob_pat}' in {root}")
    return files


def _league_name_from_df(df: pd.DataFrame, fallback: str) -> str:
    if "league" in df.columns:
        s = df["league"].dropna().astype(str)
        if not s.empty:
            return str(s.iloc[0]).strip()
    return fallback


def _safe_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _resolve_target(df: pd.DataFrame, market: str) -> pd.Series:
    home_goals = _safe_num(df, "home_goals")
    away_goals = _safe_num(df, "away_goals")
    valid = home_goals.notna() & away_goals.notna()

    target = pd.Series(np.nan, index=df.index, dtype="float64")
    if market == "btts":
        target.loc[valid] = ((home_goals.loc[valid] > 0) & (away_goals.loc[valid] > 0)).astype(float)
    elif market == "ou25":
        target.loc[valid] = ((home_goals.loc[valid] + away_goals.loc[valid]) >= 3).astype(float)
    else:
        raise ValueError(f"Unsupported market: {market}")
    return target


def _parse_time(df: pd.DataFrame) -> pd.Series:
    for col in TIME_CANDIDATES:
        if col not in df.columns:
            continue
        if col == "timestamp":
            return pd.to_datetime(df[col], errors="coerce", unit="s", utc=True)
        return pd.to_datetime(df[col], errors="coerce", utc=True, format="mixed")
    return pd.Series(pd.NaT, index=df.index)


def _prepare_frame(df: pd.DataFrame, market: str) -> pd.DataFrame:
    out = df.copy()
    out["__target__"] = _resolve_target(out, market)
    out["__sort_time__"] = _parse_time(out)
    out["__row_order__"] = np.arange(len(out))
    out = out.loc[out["__target__"].notna()].copy()
    out = out.sort_values(["__sort_time__", "__row_order__"], kind="mergesort")
    return out.reset_index(drop=True)


def _train_test_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cut = int(round(len(df) * 0.80))
    cut = min(max(cut, 1), len(df) - 1)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def _split_feature_types(df: pd.DataFrame, cols: Sequence[str]) -> Tuple[List[str], List[str]]:
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for col in cols:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols


def _build_pipeline(numeric_cols: Sequence[str], categorical_cols: Sequence[str]) -> Pipeline:
    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(numeric_cols),
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                list(categorical_cols),
            )
        )
    if not transformers:
        raise ValueError("No usable feature columns were available")

    return Pipeline(
        steps=[
            ("pre", ColumnTransformer(transformers=transformers, remainder="drop")),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )


def _score_metrics(y_true: pd.Series, proba: np.ndarray) -> Dict[str, float]:
    y = pd.to_numeric(y_true, errors="coerce")
    mask = y.notna()
    y = y.loc[mask].astype(int)
    p = pd.Series(proba, index=y_true.index).loc[mask].clip(1e-6, 1 - 1e-6)

    if y.nunique() < 2:
        return {"auc": np.nan, "logloss": np.nan, "brier": np.nan}

    return {
        "auc": float(roc_auc_score(y, p)),
        "logloss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def _threshold_stats(y_true: pd.Series, proba: pd.Series, thresholds: Sequence[float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    y = pd.to_numeric(y_true, errors="coerce")
    p = pd.to_numeric(proba, errors="coerce")
    valid = y.notna() & p.notna()
    y = y.loc[valid].astype(int)
    p = p.loc[valid]

    for thr in thresholds:
        deployed = p >= float(thr)
        deployed_rows = int(deployed.sum())
        coverage = float(deployed.mean()) if len(p) else 0.0
        if deployed_rows > 0:
            hit_rate = float(y.loc[deployed].mean())
        else:
            hit_rate = np.nan
        key = str(int(round(thr * 100)))
        out[f"hit_rate_{key}"] = hit_rate
        out[f"deployment_coverage_{key}"] = coverage
        out[f"deployed_rows_{key}"] = deployed_rows
    return out


def _fit_eval(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
) -> Dict[str, object]:
    usable_cols = [c for c in feature_cols if c in train_df.columns and c in test_df.columns]
    numeric_cols, categorical_cols = _split_feature_types(train_df, usable_cols)
    pipe = _build_pipeline(numeric_cols, categorical_cols)

    X_train = train_df[usable_cols].copy()
    y_train = pd.to_numeric(train_df[target_col], errors="coerce")
    train_mask = y_train.notna()
    X_train = X_train.loc[train_mask]
    y_train = y_train.loc[train_mask].astype(int)

    X_test = test_df[usable_cols].copy()
    y_test = pd.to_numeric(test_df[target_col], errors="coerce")
    test_mask = y_test.notna()
    X_test = X_test.loc[test_mask]
    y_test = y_test.loc[test_mask].astype(int)

    if len(X_train) == 0 or len(X_test) == 0 or y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError("Not enough valid train/test rows after filtering target")

    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    proba_s = pd.Series(proba, index=y_test.index)

    metrics = _score_metrics(y_test, proba)
    metrics.update(_threshold_stats(y_test, proba_s, THRESHOLDS))
    metrics["feature_count_total"] = len(usable_cols)
    metrics["numeric_feature_count"] = len(numeric_cols)
    metrics["categorical_feature_count"] = len(categorical_cols)
    metrics["usable_cols"] = " | ".join(usable_cols)
    return metrics


def _build_ablation_specs(market: str) -> List[Tuple[str, List[str]]]:
    approved = list(MARKET_PROMOTION_POLICY[market]["approved_cs_cols"])
    specs: List[Tuple[str, List[str]]] = []

    specs.append(("FULL_APPROVED", approved))

    for feature in approved:
        specs.append((f"SINGLE__{feature}", [feature]))

    for feature in approved:
        keep = [c for c in approved if c != feature]
        specs.append((f"ALL_MINUS__{feature}", keep))

    for group_name, cols in GROUP_DEFS.get(market, {}).items():
        cols_present = [c for c in cols if c in approved]
        if cols_present:
            specs.append((group_name, cols_present))

    deduped: List[Tuple[str, List[str]]] = []
    seen = set()
    for label, cols in specs:
        norm = tuple(dict.fromkeys(cols))
        key = (label, norm)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((label, list(norm)))
    return deduped


def _coverage_stats(df: pd.DataFrame, cols: Sequence[str]) -> Tuple[float, float, float]:
    usable = [c for c in cols if c in df.columns]
    if not usable or len(df) == 0:
        return (0.0, 0.0, 0.0)
    rates = [float(pd.to_numeric(df[c], errors="coerce").notna().mean()) if pd.api.types.is_numeric_dtype(df[c]) else float(df[c].notna().mean()) for c in usable]
    row_any = df[usable].notna().any(axis=1)
    fixture_level = float(row_any.mean()) if len(row_any) else 0.0
    return (fixture_level, float(min(rates)), float(max(rates)))


def main() -> None:
    args = _parse_args()
    files = _discover_inputs(args.input_dir, args.glob)
    market = args.market
    baseline_cols = list(dict.fromkeys(BASELINE_BY_MARKET[market]))
    approved_cols = list(MARKET_PROMOTION_POLICY[market]["approved_cs_cols"])
    ablation_specs = _build_ablation_specs(market)

    detail_rows: List[Dict[str, object]] = []

    for p in files:
        df = pd.read_csv(p, low_memory=False)
        league_fallback = p.stem.split("__", 1)[0].replace("_", " ")
        league = _league_name_from_df(df, league_fallback)
        prepared = _prepare_frame(df, market)
        if len(prepared) < args.min_rows:
            print(f"SKIP: {p} | too few usable rows | rows={len(prepared)}")
            continue

        train_df, test_df = _train_test_split(prepared)

        try:
            baseline_metrics = _fit_eval(train_df, test_df, baseline_cols, "__target__")
        except Exception as exc:
            print(f"SKIP: {p} | baseline failed | {exc}")
            continue

        present_approved = [c for c in approved_cols if c in prepared.columns]
        usable_approved = [c for c in present_approved if prepared[c].notna().any()]
        fixture_cov, min_cov, max_cov = _coverage_stats(prepared, usable_approved)

        for ablation_name, ablation_cols in ablation_specs:
            requested = list(ablation_cols)
            used = [c for c in requested if c in prepared.columns and prepared[c].notna().any()]
            if not used:
                continue

            feature_cols = list(dict.fromkeys(baseline_cols + used))
            try:
                promoted_metrics = _fit_eval(train_df, test_df, feature_cols, "__target__")
            except Exception as exc:
                print(f"SKIP ABLATION: {p} | {ablation_name} | {exc}")
                continue

            row: Dict[str, object] = {
                "league": league,
                "market": market,
                "promotion_mode": MARKET_PROMOTION_POLICY[market]["mode"],
                "input_csv": str(p),
                "rows_total": len(prepared),
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "ablation_name": ablation_name,
                "approved_cs_cols_requested": len(requested),
                "approved_cs_cols_used": len(used),
                "approved_cs_cols_list": " | ".join(used),
                "fixture_level_cs_coverage": fixture_cov,
                "approved_cs_coverage_min": min_cov,
                "approved_cs_coverage_max": max_cov,
                "baseline_auc": baseline_metrics["auc"],
                "baseline_logloss": baseline_metrics["logloss"],
                "baseline_brier": baseline_metrics["brier"],
                "promoted_auc": promoted_metrics["auc"],
                "promoted_logloss": promoted_metrics["logloss"],
                "promoted_brier": promoted_metrics["brier"],
                "delta_auc_vs_baseline": promoted_metrics["auc"] - baseline_metrics["auc"] if pd.notna(promoted_metrics["auc"]) and pd.notna(baseline_metrics["auc"]) else np.nan,
                "delta_logloss_vs_baseline": promoted_metrics["logloss"] - baseline_metrics["logloss"] if pd.notna(promoted_metrics["logloss"]) and pd.notna(baseline_metrics["logloss"]) else np.nan,
                "delta_brier_vs_baseline": promoted_metrics["brier"] - baseline_metrics["brier"] if pd.notna(promoted_metrics["brier"]) and pd.notna(baseline_metrics["brier"]) else np.nan,
                "baseline_feature_count_total": baseline_metrics["feature_count_total"],
                "promoted_feature_count_total": promoted_metrics["feature_count_total"],
                "baseline_numeric_feature_count": baseline_metrics["numeric_feature_count"],
                "baseline_categorical_feature_count": baseline_metrics["categorical_feature_count"],
                "promoted_numeric_feature_count": promoted_metrics["numeric_feature_count"],
                "promoted_categorical_feature_count": promoted_metrics["categorical_feature_count"],
                "baseline_usable_cols": baseline_metrics["usable_cols"],
                "promoted_usable_cols": promoted_metrics["usable_cols"],
            }

            for thr in THRESHOLDS:
                key = str(int(round(thr * 100)))
                row[f"baseline_hit_rate_{key}"] = baseline_metrics[f"hit_rate_{key}"]
                row[f"baseline_deployment_coverage_{key}"] = baseline_metrics[f"deployment_coverage_{key}"]
                row[f"baseline_deployed_rows_{key}"] = baseline_metrics[f"deployed_rows_{key}"]
                row[f"promoted_hit_rate_{key}"] = promoted_metrics[f"hit_rate_{key}"]
                row[f"promoted_deployment_coverage_{key}"] = promoted_metrics[f"deployment_coverage_{key}"]
                row[f"promoted_deployed_rows_{key}"] = promoted_metrics[f"deployed_rows_{key}"]
                row[f"delta_hit_rate_{key}"] = (
                    promoted_metrics[f"hit_rate_{key}"] - baseline_metrics[f"hit_rate_{key}"]
                    if pd.notna(promoted_metrics[f"hit_rate_{key}"]) and pd.notna(baseline_metrics[f"hit_rate_{key}"])
                    else np.nan
                )
                row[f"delta_deployment_coverage_{key}"] = (
                    promoted_metrics[f"deployment_coverage_{key}"] - baseline_metrics[f"deployment_coverage_{key}"]
                )

            detail_rows.append(row)

    if not detail_rows:
        print("No valid files were evaluated.")
        return

    detail_df = pd.DataFrame(detail_rows)
    detail_df = detail_df.sort_values(
        ["league", "delta_auc_vs_baseline", "delta_logloss_vs_baseline", "delta_brier_vs_baseline"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )

    summary_df = (
        detail_df.groupby("league", as_index=False, sort=True)
        .head(max(int(args.top_n_per_league), 1))
        .copy()
    )

    detail_csv = Path(args.detail_csv).expanduser().resolve()
    summary_csv = Path(args.summary_csv).expanduser().resolve()
    detail_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(detail_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print(f"SCRIPT_VERSION: {SCRIPT_VERSION}")
    print(f"MARKET: {market}")
    print(f"PROMOTION_MODE: {MARKET_PROMOTION_POLICY[market]['mode']}")
    print(f"FILES_EVALUATED: {detail_df['input_csv'].nunique()}")
    print(f"APPROVED_CS_FEATURES: {' | '.join(approved_cols)}")
    print("\nTOP ABLATIONS PER LEAGUE")
    with pd.option_context("display.max_rows", 300, "display.max_columns", None, "display.width", 320):
        print(summary_df.to_string(index=False))
    print(f"\nWROTE DETAIL CSV: {detail_csv}")
    print(f"WROTE SUMMARY CSV: {summary_csv}")


if __name__ == "__main__":
    main()