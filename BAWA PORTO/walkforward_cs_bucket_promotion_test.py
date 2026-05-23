

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SCRIPT_VERSION = "walkforward_cs_bucket_promotion_test_v1"

MARKET_CHOICES = ("ou25", "btts")
TIME_CANDIDATES = ("match_date", "date_GMT", "date", "timestamp")
LEAGUE_CANDIDATES = ("league", "competition", "division")
STATUS_CANDIDATES = ("status", "match_status", "fixture_status", "game_status", "state")

BASELINE_COLS: Dict[str, List[str]] = {
    "ou25": [
        "home_ppg",
        "away_ppg",
        "xg_for_avg_5_home",
        "xg_against_avg_5_home",
        "xg_for_avg_5_away",
        "xg_against_avg_5_away",
        "scored_rate_5_home",
        "conceded_rate_5_home",
        "scored_rate_5_away",
        "conceded_rate_5_away",
        "btts_rate_5_home",
        "btts_rate_5_away",
        "over25_rate_5_home",
        "over25_rate_5_away",
        "goaliness_avg_5_home",
        "goaliness_avg_5_away",
        "Home Team Pre-Match xG",
        "Away Team Pre-Match xG",
        "bookie_lambda_total_fit",
        "over_25_percentage_pre_match",
        "prob_over25",
        "prob_over25_v2",
        "p_over25_novig",
        "odds_ft_over25",
        "odds_ft_under25",
        "signal_over25",
        "exp_goals_sum",
        "exp_goals_sum_rm",
        "lambda_home",
        "lambda_away",
        "ou25_overround",
    ],
    "btts": [
        "home_ppg",
        "away_ppg",
        "xg_for_avg_5_home",
        "xg_against_avg_5_home",
        "xg_for_avg_5_away",
        "xg_against_avg_5_away",
        "scored_rate_5_home",
        "conceded_rate_5_home",
        "scored_rate_5_away",
        "conceded_rate_5_away",
        "btts_rate_5_home",
        "btts_rate_5_away",
        "over25_rate_5_home",
        "over25_rate_5_away",
        "goaliness_avg_5_home",
        "goaliness_avg_5_away",
        "Home Team Pre-Match xG",
        "Away Team Pre-Match xG",
        "prob_btts",
        "prob_btts_v2",
        "odds_btts_yes",
        "odds_btts_no",
        "signal_btts",
        "signal_btts_fixture",
        "signal_btts_runtime",
        "btts_alignment",
        "btts_fh_confidence",
        "btts_fh_confidence_rm",
        "p_home_fts",
        "p_away_fts",
        "p00_est",
        "p00_est_rm",
    ],
}

DEFAULT_APPROVED_CS_FEATURES: Dict[str, List[str]] = {
    "ou25": [
        "cs_mass_over25",
        "cs_mass_under25",
        "cs_mass_exact_2_goals",
        "cs_mass_exact_3_goals",
        "cs_mass_4plus_goals",
        "cs_topk_over25_share",
    ],
    "btts": [
        "cs_mass_btts_yes",
        "cs_mass_btts_no",
        "cs_mass_nil_nil",
        "cs_mass_one_side_nil",
        "cs_mass_1_1",
        "cs_mass_2_1_or_1_2",
        "cs_btts_yes_topk_share",
    ],
}

DEFAULT_THRESHOLDS = (0.55, 0.60, 0.65, 0.70)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward test for baseline vs baseline + approved CS bucket features."
    )
    parser.add_argument("--input-csv", default=None, help="Single __SCORED__cs_market_buckets.csv file")
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing scored CS bucket files",
    )
    parser.add_argument(
        "--glob",
        default="*__SCORED__cs_market_buckets.csv",
        help="Glob used with --input-dir",
    )
    parser.add_argument("--market", required=True, choices=MARKET_CHOICES)
    parser.add_argument(
        "--promotion-config",
        default=None,
        help="Optional JSON config with approved CS bucket features per market/league",
    )
    parser.add_argument(
        "--thresholds",
        default=",".join(f"{x:.2f}" for x in DEFAULT_THRESHOLDS),
        help="Comma-separated deployment thresholds, e.g. 0.55,0.60,0.65,0.70",
    )
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--detail-csv", default=None)
    parser.add_argument(
        "--min-train-rows",
        type=int,
        default=200,
        help="Minimum train rows required per file",
    )
    parser.add_argument(
        "--min-test-rows",
        type=int,
        default=50,
        help="Minimum test rows required per file",
    )
    args = parser.parse_args()

    if not args.input_csv and not args.input_dir:
        parser.error("Provide either --input-csv or --input-dir")
    if args.input_csv and args.input_dir:
        parser.error("Use only one of --input-csv or --input-dir")
    return args


def _parse_thresholds(raw: str) -> List[float]:
    vals: List[float] = []
    for part in str(raw).split(","):
        text = part.strip()
        if not text:
            continue
        vals.append(float(text))
    vals = sorted(set(vals))
    if not vals:
        raise ValueError("No valid thresholds supplied")
    return vals


def _discover_inputs(input_csv: str | None, input_dir: str | None, glob_pat: str) -> List[Path]:
    if input_csv:
        p = Path(input_csv).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Input CSV not found: {p}")
        return [p]

    d = Path(input_dir).expanduser().resolve()
    if not d.exists():
        raise FileNotFoundError(f"Input directory not found: {d}")
    files = sorted([p.resolve() for p in d.glob(glob_pat) if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No files matched glob='{glob_pat}' in {d}")
    return files


def _find_first(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _resolve_league_name(df: pd.DataFrame, path: Path) -> str:
    league_col = _find_first(df, LEAGUE_CANDIDATES)
    if league_col is not None:
        vals = df[league_col].dropna().astype(str).str.strip()
        if not vals.empty:
            return vals.iloc[0]
    stem = path.stem
    for suffix in ("__cs_market_buckets", "__SCORED", "__snapshot_proxy"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def _time_sort(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    time_col = _find_first(out, TIME_CANDIDATES)
    if time_col is None:
        out["__row_order__"] = np.arange(len(out))
        return out.sort_values(["__row_order__"], kind="mergesort").drop(columns=["__row_order__"])

    if time_col == "timestamp":
        parsed = pd.to_datetime(out[time_col], errors="coerce", unit="s", utc=True)
    else:
        parsed = pd.to_datetime(out[time_col], errors="coerce", utc=True, format="mixed")

    out["__sort_time__"] = parsed
    out["__row_order__"] = np.arange(len(out))
    out = out.sort_values(["__sort_time__", "__row_order__"], kind="mergesort")
    return out.drop(columns=["__sort_time__", "__row_order__"])


def _resolve_completed_mask(df: pd.DataFrame) -> pd.Series:
    status_col = _find_first(df, STATUS_CANDIDATES)
    if status_col is not None:
        s = df[status_col].astype(str).str.strip().str.lower()
        finished = {"complete", "completed", "finished", "ft", "full-time", "full time", "aet", "after penalties"}
        if s.isin(finished).any():
            return s.isin(finished)
    hg = _num(df, "home_goals")
    ag = _num(df, "away_goals")
    return hg.notna() & ag.notna()


def _resolve_target(df: pd.DataFrame, market: str) -> pd.Series:
    hg = _num(df, "home_goals")
    ag = _num(df, "away_goals")
    valid = hg.notna() & ag.notna()

    y = pd.Series(np.nan, index=df.index, dtype="float64")
    if market == "ou25":
        y.loc[valid] = ((hg.loc[valid] + ag.loc[valid]) >= 3).astype(float)
    elif market == "btts":
        y.loc[valid] = ((hg.loc[valid] >= 1) & (ag.loc[valid] >= 1)).astype(float)
    else:
        raise ValueError(f"Unsupported market: {market}")
    return y


def _prepare_frame(df: pd.DataFrame, market: str) -> pd.DataFrame:
    out = _time_sort(df)
    out["__target__"] = _resolve_target(out, market)
    completed_mask = _resolve_completed_mask(out)
    out = out.loc[completed_mask & out["__target__"].notna()].copy()
    out["__target__"] = out["__target__"].astype(int)
    return out


def _train_test_split_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    cut = int(round(n * 0.80))
    cut = min(max(cut, 1), n - 1)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def _safe_auc(y_true: pd.Series, proba: np.ndarray) -> float:
    y = pd.to_numeric(y_true, errors="coerce")
    mask = y.notna()
    y = y.loc[mask].astype(int)
    p = pd.Series(proba, index=y_true.index).loc[mask]
    if y.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _safe_logloss(y_true: pd.Series, proba: np.ndarray) -> float:
    y = pd.to_numeric(y_true, errors="coerce")
    mask = y.notna()
    y = y.loc[mask].astype(int)
    p = pd.Series(proba, index=y_true.index).loc[mask].clip(1e-6, 1 - 1e-6)
    if y.nunique() < 2:
        return float("nan")
    return float(log_loss(y, p))



def _safe_brier(y_true: pd.Series, proba: np.ndarray) -> float:
    y = pd.to_numeric(y_true, errors="coerce")
    mask = y.notna()
    y = y.loc[mask].astype(int)
    p = pd.Series(proba, index=y_true.index).loc[mask]
    if y.nunique() < 2:
        return float("nan")
    return float(brier_score_loss(y, p))


def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def _split_feature_types(df: pd.DataFrame, feature_cols: Sequence[str]) -> tuple[List[str], List[str]]:
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        if _is_numeric_series(df[col]):
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
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                list(numeric_cols),
            )
        )

    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                list(categorical_cols),
            )
        )

    if not transformers:
        raise ValueError("No usable feature columns were available for preprocessing")

    pre = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )
    return Pipeline([
        ("pre", pre),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def _evaluate_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    thresholds: Sequence[float],
) -> dict:
    usable_cols = [c for c in feature_cols if c in train_df.columns and c in test_df.columns]
    if not usable_cols:
        raise ValueError("No usable feature columns were available")

    numeric_cols, categorical_cols = _split_feature_types(train_df, usable_cols)

    X_train = train_df[usable_cols].copy()
    y_train = pd.to_numeric(train_df["__target__"], errors="coerce")
    X_test = test_df[usable_cols].copy()
    y_test = pd.to_numeric(test_df["__target__"], errors="coerce")

    train_mask = y_train.notna()
    test_mask = y_test.notna()
    X_train = X_train.loc[train_mask]
    y_train = y_train.loc[train_mask].astype(int)
    X_test = X_test.loc[test_mask]
    y_test = y_test.loc[test_mask].astype(int)

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError("Train/test split does not contain both classes")

    pipe = _build_pipeline(numeric_cols, categorical_cols)
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]

    out: dict = {
        "auc": _safe_auc(y_test, proba),
        "logloss": _safe_logloss(y_test, proba),
        "brier": _safe_brier(y_test, proba),
        "feature_count_total": len(usable_cols),
        "numeric_feature_count": len(numeric_cols),
        "categorical_feature_count": len(categorical_cols),
        "test_rows": int(len(y_test)),
    }

    y_test_arr = y_test.to_numpy(dtype=int)
    proba_arr = np.asarray(proba, dtype=float)

    for thr in thresholds:
        suffix = str(int(round(thr * 100)))
        deployed_mask = proba_arr >= thr
        deployed_rows = int(deployed_mask.sum())
        coverage = deployed_rows / len(proba_arr) if len(proba_arr) else float("nan")
        hit_rate = float(y_test_arr[deployed_mask].mean()) if deployed_rows > 0 else float("nan")
        out[f"hit_rate_{suffix}"] = hit_rate
        out[f"deployment_coverage_{suffix}"] = coverage
        out[f"deployed_rows_{suffix}"] = deployed_rows

    return out


def _load_promotion_config(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Promotion config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _approved_cs_cols_for_market_league(market: str, league: str, config: dict) -> List[str]:
    if not config:
        return list(DEFAULT_APPROVED_CS_FEATURES.get(market, []))

    market_cfg = config.get(market, {})
    league_map = market_cfg.get("league_whitelist", {})
    approved = league_map.get(league)
    if approved is None:
        approved = market_cfg.get("approved_proxy_columns", DEFAULT_APPROVED_CS_FEATURES.get(market, []))
    return list(approved or [])


def _fixture_level_coverage(df: pd.DataFrame, cols: Sequence[str]) -> tuple[float, float, int]:
    usable = [c for c in cols if c in df.columns]
    if not usable or df.empty:
        return (float("nan"), float("nan"), 0)
    cov = df[usable].notna().all(axis=1)
    fixture_coverage = float(cov.mean()) if len(cov) else float("nan")
    non_null_counts = [float(df[c].notna().mean()) for c in usable]
    return (fixture_coverage, float(min(non_null_counts)), len(usable))


def main() -> None:
    args = _parse_args()
    thresholds = _parse_thresholds(args.thresholds)
    inputs = _discover_inputs(args.input_csv, args.input_dir, args.glob)
    config = _load_promotion_config(args.promotion_config)

    if args.market not in BASELINE_COLS:
        raise ValueError(f"No baseline feature list configured for market={args.market}")

    detail_rows: List[Dict[str, object]] = []

    for p in inputs:
        raw_df = pd.read_csv(p, low_memory=False)
        league = _resolve_league_name(raw_df, p)
        approved_cs_cols = _approved_cs_cols_for_market_league(args.market, league, config)
        approved_cs_cols = list(dict.fromkeys(approved_cs_cols))

        df = _prepare_frame(raw_df, args.market)
        if len(df) < max(args.min_train_rows + args.min_test_rows, 20):
            print(f"SKIP: {p} | too few completed rows for market={args.market} | rows={len(df)}")
            continue

        train_df, test_df = _train_test_split_time(df)
        if len(train_df) < args.min_train_rows:
            print(f"SKIP: {p} | train rows below minimum | train_rows={len(train_df)}")
            continue
        if len(test_df) < args.min_test_rows:
            print(f"SKIP: {p} | test rows below minimum | test_rows={len(test_df)}")
            continue

        baseline_cols = [c for c in BASELINE_COLS[args.market] if c in df.columns]
        cs_cols_present = [c for c in approved_cs_cols if c in df.columns]
        cs_cols_usable = [c for c in cs_cols_present if train_df[c].notna().any() and test_df[c].notna().any()]

        fixture_cov, cs_cov_min, cs_count_usable = _fixture_level_coverage(df, cs_cols_usable)

        try:
            baseline_metrics = _evaluate_model(train_df, test_df, baseline_cols, thresholds)
        except Exception as exc:
            print(f"SKIP: {p} | baseline failed | {exc}")
            continue

        try:
            promoted_metrics = _evaluate_model(train_df, test_df, baseline_cols + cs_cols_usable, thresholds)
        except Exception as exc:
            print(f"SKIP: {p} | baseline+approved-cs failed | {exc}")
            continue

        row: Dict[str, object] = {
            "league": league,
            "market": args.market,
            "input_csv": str(p),
            "rows_total": int(len(df)),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "baseline_auc": baseline_metrics["auc"],
            "baseline_logloss": baseline_metrics["logloss"],
            "baseline_brier": baseline_metrics["brier"],
            "promoted_auc": promoted_metrics["auc"],
            "promoted_logloss": promoted_metrics["logloss"],
            "promoted_brier": promoted_metrics["brier"],
            "delta_auc": promoted_metrics["auc"] - baseline_metrics["auc"],
            "delta_logloss": promoted_metrics["logloss"] - baseline_metrics["logloss"],
            "delta_brier": promoted_metrics["brier"] - baseline_metrics["brier"],
            "baseline_feature_count": baseline_metrics["feature_count_total"],
            "promoted_feature_count": promoted_metrics["feature_count_total"],
            "baseline_numeric_feature_count": baseline_metrics["numeric_feature_count"],
            "baseline_categorical_feature_count": baseline_metrics["categorical_feature_count"],
            "promoted_numeric_feature_count": promoted_metrics["numeric_feature_count"],
            "promoted_categorical_feature_count": promoted_metrics["categorical_feature_count"],
            "approved_cs_cols_requested": len(approved_cs_cols),
            "approved_cs_cols_present": len(cs_cols_present),
            "approved_cs_cols_usable": cs_count_usable,
            "approved_cs_cols_list": " | ".join(cs_cols_usable),
            "fixture_level_cs_coverage": fixture_cov,
            "approved_cs_coverage_min": cs_cov_min,
            "approved_cs_coverage_max": float(max((float(df[c].notna().mean()) for c in cs_cols_usable), default=float("nan"))) if cs_cols_usable else float("nan"),
        }

        for thr in thresholds:
            suffix = str(int(round(thr * 100)))
            row[f"baseline_hit_rate_{suffix}"] = baseline_metrics[f"hit_rate_{suffix}"]
            row[f"baseline_deployment_coverage_{suffix}"] = baseline_metrics[f"deployment_coverage_{suffix}"]
            row[f"baseline_deployed_rows_{suffix}"] = baseline_metrics[f"deployed_rows_{suffix}"]
            row[f"promoted_hit_rate_{suffix}"] = promoted_metrics[f"hit_rate_{suffix}"]
            row[f"promoted_deployment_coverage_{suffix}"] = promoted_metrics[f"deployment_coverage_{suffix}"]
            row[f"promoted_deployed_rows_{suffix}"] = promoted_metrics[f"deployed_rows_{suffix}"]
            row[f"delta_hit_rate_{suffix}"] = (
                promoted_metrics[f"hit_rate_{suffix}"] - baseline_metrics[f"hit_rate_{suffix}"]
                if pd.notna(promoted_metrics[f"hit_rate_{suffix}"]) and pd.notna(baseline_metrics[f"hit_rate_{suffix}"])
                else float("nan")
            )
            row[f"delta_deployment_coverage_{suffix}"] = (
                promoted_metrics[f"deployment_coverage_{suffix}"] - baseline_metrics[f"deployment_coverage_{suffix}"]
                if pd.notna(promoted_metrics[f"deployment_coverage_{suffix}"]) and pd.notna(baseline_metrics[f"deployment_coverage_{suffix}"])
                else float("nan")
            )

        detail_rows.append(row)

    if not detail_rows:
        print("No valid files were evaluated.")
        return

    detail_df = pd.DataFrame(detail_rows)
    detail_df = detail_df.sort_values(
        ["delta_auc", "delta_logloss", "league"],
        ascending=[False, True, True],
        kind="mergesort",
    )

    print(f"SCRIPT_VERSION: {SCRIPT_VERSION}")
    print(f"MARKET: {args.market}")
    print(f"FILES_EVALUATED: {len(detail_df)}")
    print("\nBATCH SUMMARY")
    with pd.option_context("display.max_rows", 200, "display.max_columns", None, "display.width", 320):
        print(detail_df.to_string(index=False))

    if args.detail_csv:
        detail_path = Path(args.detail_csv).expanduser().resolve()
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        detail_df.to_csv(detail_path, index=False)
        print(f"\nWROTE DETAIL CSV: {detail_path}")

    if args.summary_csv:
        summary_cols = [
            "league",
            "market",
            "input_csv",
            "rows_total",
            "train_rows",
            "test_rows",
            "baseline_auc",
            "baseline_logloss",
            "baseline_brier",
            "promoted_auc",
            "promoted_logloss",
            "promoted_brier",
            "delta_auc",
            "delta_logloss",
            "delta_brier",
            "fixture_level_cs_coverage",
            "baseline_feature_count",
            "promoted_feature_count",
            "baseline_numeric_feature_count",
            "baseline_categorical_feature_count",
            "promoted_numeric_feature_count",
            "promoted_categorical_feature_count",
            "approved_cs_cols_requested",
            "approved_cs_cols_present",
            "approved_cs_cols_usable",
            "approved_cs_cols_list",
        ]
        for thr in thresholds:
            suffix = str(int(round(thr * 100)))
            summary_cols.extend([
                f"baseline_hit_rate_{suffix}",
                f"baseline_deployment_coverage_{suffix}",
                f"promoted_hit_rate_{suffix}",
                f"promoted_deployment_coverage_{suffix}",
                f"delta_hit_rate_{suffix}",
                f"delta_deployment_coverage_{suffix}",
            ])
        summary_df = detail_df[[c for c in summary_cols if c in detail_df.columns]].copy()
        summary_path = Path(args.summary_csv).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        print(f"WROTE SUMMARY CSV: {summary_path}")


if __name__ == "__main__":
    main()