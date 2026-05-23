#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_VERSION = "audit_ou25_fullgrid_signal_v1"
DEFAULT_GLOB = "*__SCORED__fullgrid_cs_market_buckets.csv"
DEFAULT_SAMPLE_ROWS = 8
DEFAULT_MAX_GOALS = 8
DEFAULT_TOPK_SCORELINES = 10
DEFAULT_MIN_ROWS = 80
DEFAULT_CORR_REDUNDANCY_THRESHOLD = 0.92
DEFAULT_TEST_FRACTION = 0.20
RANDOM_STATE = 42

FG_AUDIT_COLS: Sequence[str] = (
    "fg_cs_mass_over25",
    "fg_cs_mass_under25",
    "fg_cs_mass_exact_2_goals",
    "fg_cs_mass_exact_3_goals",
    "fg_cs_mass_4plus_goals",
    "fg_cs_topk_over25_share",
)

BASELINE_COLS: Sequence[str] = (
    "prob_over25",
    "prob_over25_v2",
    "over_25_percentage_pre_match",
    "exp_goals_sum",
    "bookie_lambda_total_fit",
)

OPTIONAL_BASELINE_COLS: Sequence[str] = (
    "snap_ou25_over_regime_blend",
    "snap_xg_total_pressure",
)

LAMBDA_HOME_CANDIDATES: Sequence[str] = (
    "lambda_home",
    "lambda_home_rm",
    "home_lambda",
    "home_lambda_rm",
)
LAMBDA_AWAY_CANDIDATES: Sequence[str] = (
    "lambda_away",
    "lambda_away_rm",
    "away_lambda",
    "away_lambda_rm",
)

TIME_COL_CANDIDATES: Sequence[str] = (
    "timestamp",
    "match_date",
    "date_GMT",
    "date",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit OU25 full-grid CS signal strength, redundancy, and incremental lift."
    )
    parser.add_argument("--input-csv", default=None, help="Single fullgrid CS scored CSV")
    parser.add_argument("--input-dir", default=None, help="Directory containing fullgrid CS scored CSVs")
    parser.add_argument("--glob", default=DEFAULT_GLOB, help="Glob used with --input-dir")
    parser.add_argument("--sample-rows", type=int, default=DEFAULT_SAMPLE_ROWS)
    parser.add_argument("--max-goals", type=int, default=DEFAULT_MAX_GOALS)
    parser.add_argument("--topk-scorelines", type=int, default=DEFAULT_TOPK_SCORELINES)
    parser.add_argument("--min-rows", type=int, default=DEFAULT_MIN_ROWS)
    parser.add_argument("--corr-threshold", type=float, default=DEFAULT_CORR_REDUNDANCY_THRESHOLD)
    parser.add_argument("--summary-csv", default=None, help="Optional summary CSV path")
    parser.add_argument("--corr-csv", default=None, help="Optional long-form correlation CSV path")
    parser.add_argument("--univariate-csv", default=None, help="Optional univariate stats CSV path")
    parser.add_argument("--residual-csv", default=None, help="Optional residual lift CSV path")
    args = parser.parse_args()

    if bool(args.input_csv) == bool(args.input_dir):
        parser.error("Provide exactly one of --input-csv or --input-dir")
    if args.sample_rows < 1:
        parser.error("--sample-rows must be >= 1")
    if args.max_goals < 1:
        parser.error("--max-goals must be >= 1")
    if args.topk_scorelines < 1:
        parser.error("--topk-scorelines must be >= 1")
    return args


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


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _find_first_present(df: pd.DataFrame, cols: Sequence[str]) -> str | None:
    for col in cols:
        if col in df.columns:
            return col
    return None


def _resolve_lambda_pair(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, str, str]:
    home_col = _find_first_present(df, LAMBDA_HOME_CANDIDATES)
    away_col = _find_first_present(df, LAMBDA_AWAY_CANDIDATES)
    home = _num(df, home_col) if home_col else pd.Series(np.nan, index=df.index, dtype="float64")
    away = _num(df, away_col) if away_col else pd.Series(np.nan, index=df.index, dtype="float64")
    return home, away, home_col or "", away_col or ""


def _resolve_time_series(df: pd.DataFrame) -> pd.Series:
    if "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        return pd.to_datetime(ts, errors="coerce", unit="s", utc=True)
    for col in ("match_date", "date_GMT", "date"):
        if col in df.columns:
            return pd.to_datetime(df[col], errors="coerce", utc=True, format="mixed")
    return pd.Series(pd.NaT, index=df.index)


def _resolve_target(df: pd.DataFrame) -> pd.Series:
    if "actual_ou25" in df.columns:
        txt = df["actual_ou25"].astype(str).str.strip().str.upper()
        y = pd.Series(np.nan, index=df.index, dtype="float64")
        y.loc[txt == "OVER25"] = 1.0
        y.loc[txt == "UNDER25"] = 0.0
        return y

    home_goals = _num(df, "home_goals")
    away_goals = _num(df, "away_goals")
    valid = home_goals.notna() & away_goals.notna()
    y = pd.Series(np.nan, index=df.index, dtype="float64")
    y.loc[valid] = ((home_goals.loc[valid] + away_goals.loc[valid]) >= 3).astype(float)
    return y


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["__target__"] = _resolve_target(out)
    out["__time__"] = _resolve_time_series(out)
    out["__lambda_home__"], out["__lambda_away__"], out["__lambda_home_source__"], out["__lambda_away_source__"] = _resolve_lambda_pair(out)
    out = out.loc[out["__target__"].notna()].copy()
    out = out.sort_values("__time__", kind="mergesort")
    return out


def _poisson_pmf(k: int, lam: float) -> float:
    if not np.isfinite(lam) or lam < 0:
        return float("nan")
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _build_score_grid(lam_home: float, lam_away: float, max_goals: int) -> pd.DataFrame:
    rows: List[Dict[str, float | int | str]] = []
    for hg in range(max_goals + 1):
        p_h = _poisson_pmf(hg, lam_home)
        for ag in range(max_goals + 1):
            p_a = _poisson_pmf(ag, lam_away)
            prob = p_h * p_a
            rows.append(
                {
                    "hg": hg,
                    "ag": ag,
                    "scoreline": f"{hg}-{ag}",
                    "prob": prob,
                    "total_goals": hg + ag,
                }
            )
    grid = pd.DataFrame(rows)
    total_mass = float(pd.to_numeric(grid["prob"], errors="coerce").sum())
    if total_mass > 0:
        grid["prob_norm"] = pd.to_numeric(grid["prob"], errors="coerce") / total_mass
    else:
        grid["prob_norm"] = np.nan
    return grid


def _grid_mass(grid: pd.DataFrame, mask: pd.Series) -> float:
    vals = pd.to_numeric(grid.loc[mask, "prob_norm"], errors="coerce")
    if vals.empty:
        return 0.0
    return float(vals.sum())


def _safe_auc(y_true: pd.Series, scores: pd.Series) -> float:
    mask = y_true.notna() & scores.notna()
    if mask.sum() < 2:
        return float("nan")
    yt = y_true.loc[mask].astype(int)
    sc = scores.loc[mask].astype(float)
    if yt.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(yt, sc))


def _safe_brier(y_true: pd.Series, scores: pd.Series) -> float:
    mask = y_true.notna() & scores.notna()
    if mask.sum() < 2:
        return float("nan")
    yt = y_true.loc[mask].astype(int)
    sc = scores.loc[mask].astype(float).clip(1e-6, 1 - 1e-6)
    return float(brier_score_loss(yt, sc))


def _normalized_for_brier(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return s
    smin = float(s.min())
    smax = float(s.max())
    if not np.isfinite(smin) or not np.isfinite(smax) or smax <= smin:
        return pd.Series(np.nan, index=s.index, dtype="float64")
    out = (s - smin) / (smax - smin)
    return out.clip(1e-6, 1 - 1e-6)


def _fit_small_model(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: Sequence[str]) -> Dict[str, float]:
    usable = [c for c in cols if c in train_df.columns and c in test_df.columns]
    usable = [c for c in usable if pd.to_numeric(train_df[c], errors="coerce").notna().sum() > 0]
    if not usable:
        raise ValueError("No usable numeric features")

    X_train = train_df[usable].apply(pd.to_numeric, errors="coerce")
    X_test = test_df[usable].apply(pd.to_numeric, errors="coerce")
    y_train = pd.to_numeric(train_df["__target__"], errors="coerce").astype(int)
    y_test = pd.to_numeric(test_df["__target__"], errors="coerce").astype(int)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    pipe.fit(X_train, y_train)
    proba = pd.Series(pipe.predict_proba(X_test)[:, 1], index=test_df.index).clip(1e-6, 1 - 1e-6)

    out = {
        "auc": float(roc_auc_score(y_test, proba)) if y_test.nunique() > 1 else float("nan"),
        "logloss": float(log_loss(y_test, proba, labels=[0, 1])),
        "brier": float(brier_score_loss(y_test, proba)),
        "n_test": int(len(test_df)),
        "feature_count": int(len(usable)),
    }
    return out


def _row_level_mass_sanity(df: pd.DataFrame, sample_rows: int, max_goals: int, topk_scorelines: int) -> List[Dict[str, object]]:
    sample_df = df.loc[
        df["__lambda_home__"].notna() & df["__lambda_away__"].notna() & df["fg_cs_mass_over25"].notna()
    ].copy()
    if sample_df.empty:
        return []

    sample_df = sample_df.head(sample_rows)
    rows: List[Dict[str, object]] = []
    for idx, row in sample_df.iterrows():
        grid = _build_score_grid(float(row["__lambda_home__"]), float(row["__lambda_away__"]), max_goals=max_goals)
        tg = pd.to_numeric(grid["total_goals"], errors="coerce")
        over25 = _grid_mass(grid, tg >= 3)
        under25 = _grid_mass(grid, tg <= 2)
        exact2 = _grid_mass(grid, tg == 2)
        exact3 = _grid_mass(grid, tg == 3)
        gte4 = _grid_mass(grid, tg >= 4)

        ordered = grid.sort_values("prob_norm", ascending=False, kind="mergesort").head(topk_scorelines)
        top10 = " | ".join(
            f"{r.scoreline}:{float(r.prob_norm):.4f}" for r in ordered.itertuples(index=False)
        )

        rows.append(
            {
                "row_index": int(idx),
                "fixture_key": row.get("fixture_key", ""),
                "home_team_name": row.get("home_team_name", ""),
                "away_team_name": row.get("away_team_name", ""),
                "lambda_home": float(row["__lambda_home__"]),
                "lambda_away": float(row["__lambda_away__"]),
                "p_total_le_2": under25,
                "p_total_eq_2": exact2,
                "p_total_eq_3": exact3,
                "p_total_ge_4": gte4,
                "p_total_ge_3": over25,
                "fg_cs_mass_over25_file": float(pd.to_numeric(pd.Series([row.get("fg_cs_mass_over25")]), errors="coerce").iloc[0]),
                "fg_cs_mass_under25_file": float(pd.to_numeric(pd.Series([row.get("fg_cs_mass_under25")]), errors="coerce").iloc[0]),
                "over25_abs_diff": abs(over25 - float(row.get("fg_cs_mass_over25", np.nan))),
                "under25_abs_diff": abs(under25 - float(row.get("fg_cs_mass_under25", np.nan))),
                "top_scorelines": top10,
            }
        )
    return rows


def _correlation_audit(df: pd.DataFrame, corr_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = [
        "fg_cs_mass_over25",
        "prob_over25",
        "prob_over25_v2",
        "over_25_percentage_pre_match",
        "exp_goals_sum",
        "bookie_lambda_total_fit",
    ]
    for opt in OPTIONAL_BASELINE_COLS:
        if opt in df.columns:
            cols.append(opt)

    usable = [c for c in cols if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().sum() > 0]
    num_df = df[usable].apply(pd.to_numeric, errors="coerce")

    pearson = num_df.corr(method="pearson")
    spearman = num_df.corr(method="spearman")

    redundancy_rows: List[Dict[str, object]] = []
    for i, c1 in enumerate(usable):
        for c2 in usable[i + 1 :]:
            val = pearson.loc[c1, c2]
            if pd.notna(val) and abs(float(val)) > corr_threshold:
                redundancy_rows.append(
                    {
                        "col_a": c1,
                        "col_b": c2,
                        "pearson_corr": float(val),
                        "flag": f"abs_corr_gt_{corr_threshold}",
                    }
                )
    redundancy_df = pd.DataFrame(redundancy_rows)
    return pearson, spearman, redundancy_df


def _univariate_strength(df: pd.DataFrame) -> pd.DataFrame:
    y = pd.to_numeric(df["__target__"], errors="coerce")
    rows: List[Dict[str, object]] = []
    cols = list(FG_AUDIT_COLS) + [c for c in BASELINE_COLS if c in df.columns] + [c for c in OPTIONAL_BASELINE_COLS if c in df.columns]
    seen: set[str] = set()
    for col in cols:
        if col in seen or col not in df.columns:
            continue
        seen.add(col)
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() == 0:
            continue
        over_mask = y == 1
        under_mask = y == 0
        rows.append(
            {
                "feature": col,
                "rows_non_null": int(s.notna().sum()),
                "auc": _safe_auc(y, s),
                "brier_raw_normalized": _safe_brier(y, _normalized_for_brier(s)),
                "mean_over25": float(s.loc[over_mask].mean()) if over_mask.any() else np.nan,
                "mean_under25": float(s.loc[under_mask].mean()) if under_mask.any() else np.nan,
                "mean_diff_over_minus_under": float(s.loc[over_mask].mean() - s.loc[under_mask].mean()) if over_mask.any() and under_mask.any() else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["auc", "mean_diff_over_minus_under"], ascending=[False, False], kind="mergesort")
    return out


def _residual_usefulness(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working = working.sort_values("__time__", kind="mergesort")
    n = len(working)
    split = max(int(round(n * (1 - DEFAULT_TEST_FRACTION))), 1)
    split = min(split, n - 1)
    train_df = working.iloc[:split].copy()
    test_df = working.iloc[split:].copy()
    if len(train_df) < 30 or len(test_df) < 20:
        return pd.DataFrame()

    baseline_cols = [c for c in BASELINE_COLS if c in working.columns]
    baseline_metrics = _fit_small_model(train_df, test_df, baseline_cols)

    rows: List[Dict[str, object]] = [
        {
            "model_name": "baseline_only",
            "added_fg_feature": "",
            "auc": baseline_metrics["auc"],
            "logloss": baseline_metrics["logloss"],
            "brier": baseline_metrics["brier"],
            "delta_auc_vs_baseline": 0.0,
            "delta_logloss_vs_baseline": 0.0,
            "delta_brier_vs_baseline": 0.0,
            "feature_count": baseline_metrics["feature_count"],
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
        }
    ]

    for fg_col in FG_AUDIT_COLS:
        if fg_col not in working.columns:
            continue
        try:
            metrics = _fit_small_model(train_df, test_df, list(baseline_cols) + [fg_col])
            rows.append(
                {
                    "model_name": f"baseline_plus__{fg_col}",
                    "added_fg_feature": fg_col,
                    "auc": metrics["auc"],
                    "logloss": metrics["logloss"],
                    "brier": metrics["brier"],
                    "delta_auc_vs_baseline": float(metrics["auc"] - baseline_metrics["auc"]) if pd.notna(metrics["auc"]) and pd.notna(baseline_metrics["auc"]) else np.nan,
                    "delta_logloss_vs_baseline": float(metrics["logloss"] - baseline_metrics["logloss"]),
                    "delta_brier_vs_baseline": float(metrics["brier"] - baseline_metrics["brier"]),
                    "feature_count": metrics["feature_count"],
                    "train_rows": int(len(train_df)),
                    "test_rows": int(len(test_df)),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "model_name": f"baseline_plus__{fg_col}",
                    "added_fg_feature": fg_col,
                    "auc": np.nan,
                    "logloss": np.nan,
                    "brier": np.nan,
                    "delta_auc_vs_baseline": np.nan,
                    "delta_logloss_vs_baseline": np.nan,
                    "delta_brier_vs_baseline": np.nan,
                    "feature_count": np.nan,
                    "train_rows": int(len(train_df)),
                    "test_rows": int(len(test_df)),
                    "status": f"error: {exc}",
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["model_name"], kind="mergesort")
    return out


def _write_optional(df: pd.DataFrame, path_text: str | None) -> None:
    if not path_text:
        return
    path = Path(path_text).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _melt_corr(name: str, corr_df: pd.DataFrame, input_csv: str) -> pd.DataFrame:
    if corr_df.empty:
        return pd.DataFrame(columns=["input_csv", "matrix", "col_a", "col_b", "corr"])
    out = corr_df.stack(dropna=False).reset_index()
    out.columns = ["col_a", "col_b", "corr"]
    out.insert(0, "matrix", name)
    out.insert(0, "input_csv", input_csv)
    return out


def main() -> None:
    args = _parse_args()
    inputs = _discover_inputs(args.input_csv, args.input_dir, args.glob)

    summary_rows: List[Dict[str, object]] = []
    corr_rows: List[pd.DataFrame] = []
    univariate_rows: List[pd.DataFrame] = []
    residual_rows: List[pd.DataFrame] = []

    for input_path in inputs:
        df = pd.read_csv(input_path, low_memory=False)
        df = _prepare_df(df)
        if len(df) < args.min_rows:
            print(f"SKIP: {input_path} | too few rows after target resolution | rows={len(df)}")
            continue

        print("=" * 120)
        print(f"SCRIPT_VERSION: {SCRIPT_VERSION}")
        print(f"INPUT_CSV: {input_path}")
        print(f"ROWS: {len(df)}")
        print(f"LAMBDA_HOME_SOURCE: {df['__lambda_home_source__'].iloc[0] if len(df) else ''}")
        print(f"LAMBDA_AWAY_SOURCE: {df['__lambda_away_source__'].iloc[0] if len(df) else ''}")

        sanity_rows = _row_level_mass_sanity(
            df=df,
            sample_rows=args.sample_rows,
            max_goals=args.max_goals,
            topk_scorelines=args.topk_scorelines,
        )
        sanity_df = pd.DataFrame(sanity_rows)
        print("\nA. ROW-LEVEL MASS SANITY CHECK")
        if sanity_df.empty:
            print("No usable rows for sanity check.")
        else:
            with pd.option_context("display.max_columns", None, "display.width", 260):
                print(sanity_df.to_string(index=False))

        pearson_df, spearman_df, redundancy_df = _correlation_audit(df, corr_threshold=args.corr_threshold)
        print("\nB. PEARSON CORRELATION MATRIX")
        with pd.option_context("display.max_columns", None, "display.width", 220):
            print(pearson_df.round(4).to_string())
        print("\nB. SPEARMAN CORRELATION MATRIX")
        with pd.option_context("display.max_columns", None, "display.width", 220):
            print(spearman_df.round(4).to_string())
        print("\nB. REDUNDANCY FLAGS")
        if redundancy_df.empty:
            print(f"No abs(Pearson) > {args.corr_threshold:.2f} pairs found.")
        else:
            with pd.option_context("display.max_columns", None, "display.width", 220):
                print(redundancy_df.to_string(index=False))

        univariate_df = _univariate_strength(df)
        print("\nC. PREDICTIVE UNIVARIATE STRENGTH")
        with pd.option_context("display.max_columns", None, "display.width", 220):
            print(univariate_df.round(6).to_string(index=False))

        residual_df = _residual_usefulness(df)
        print("\nD. RESIDUAL USEFULNESS TEST")
        if residual_df.empty:
            print("Not enough rows for residual usefulness test.")
        else:
            with pd.option_context("display.max_columns", None, "display.width", 220):
                print(residual_df.round(6).to_string(index=False))

        summary_rows.append(
            {
                "input_csv": str(input_path),
                "rows": int(len(df)),
                "target_over25_rate": float(pd.to_numeric(df["__target__"], errors="coerce").mean()),
                "lambda_home_source": df["__lambda_home_source__"].iloc[0] if len(df) else "",
                "lambda_away_source": df["__lambda_away_source__"].iloc[0] if len(df) else "",
                "sanity_rows_checked": int(len(sanity_df)),
                "max_over25_abs_diff": float(pd.to_numeric(sanity_df["over25_abs_diff"], errors="coerce").max()) if not sanity_df.empty else np.nan,
                "max_under25_abs_diff": float(pd.to_numeric(sanity_df["under25_abs_diff"], errors="coerce").max()) if not sanity_df.empty else np.nan,
                "fg_mass_over25_auc": float(univariate_df.loc[univariate_df["feature"] == "fg_cs_mass_over25", "auc"].iloc[0]) if (not univariate_df.empty and (univariate_df["feature"] == "fg_cs_mass_over25").any()) else np.nan,
                "fg_mass_over25_brier": float(univariate_df.loc[univariate_df["feature"] == "fg_cs_mass_over25", "brier_raw_normalized"].iloc[0]) if (not univariate_df.empty and (univariate_df["feature"] == "fg_cs_mass_over25").any()) else np.nan,
                "baseline_auc": float(residual_df.loc[residual_df["model_name"] == "baseline_only", "auc"].iloc[0]) if (not residual_df.empty and (residual_df["model_name"] == "baseline_only").any()) else np.nan,
                "best_fg_delta_auc": float(pd.to_numeric(residual_df.loc[residual_df["added_fg_feature"] != "", "delta_auc_vs_baseline"], errors="coerce").max()) if not residual_df.empty else np.nan,
                "best_fg_delta_logloss": float(pd.to_numeric(residual_df.loc[residual_df["added_fg_feature"] != "", "delta_logloss_vs_baseline"], errors="coerce").min()) if not residual_df.empty else np.nan,
            }
        )

        corr_rows.append(_melt_corr("pearson", pearson_df, str(input_path)))
        corr_rows.append(_melt_corr("spearman", spearman_df, str(input_path)))
        if not redundancy_df.empty:
            tmp = redundancy_df.copy()
            tmp.insert(0, "input_csv", str(input_path))
            corr_rows.append(tmp)
        if not univariate_df.empty:
            tmp = univariate_df.copy()
            tmp.insert(0, "input_csv", str(input_path))
            univariate_rows.append(tmp)
        if not residual_df.empty:
            tmp = residual_df.copy()
            tmp.insert(0, "input_csv", str(input_path))
            residual_rows.append(tmp)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        print("\n" + "=" * 120)
        print("SUMMARY")
        with pd.option_context("display.max_columns", None, "display.width", 220):
            print(summary_df.round(6).to_string(index=False))

    if args.summary_csv:
        _write_optional(summary_df, args.summary_csv)
    if args.corr_csv:
        corr_df = pd.concat(corr_rows, ignore_index=True) if corr_rows else pd.DataFrame()
        _write_optional(corr_df, args.corr_csv)
    if args.univariate_csv:
        uni_df = pd.concat(univariate_rows, ignore_index=True) if univariate_rows else pd.DataFrame()
        _write_optional(uni_df, args.univariate_csv)
    if args.residual_csv:
        res_df = pd.concat(residual_rows, ignore_index=True) if residual_rows else pd.DataFrame()
        _write_optional(res_df, args.residual_csv)


if __name__ == "__main__":
    main()