#!/usr/bin/env python3
"""Run trainer-native World Cup research ablations.

This is the proper research bridge toward the production training stack:

- canonical input: `Matches/__merged__/World_Cup__merged.csv`
- feature construction: imports `train_markets.build_features`
- targets: imports `train_markets._derive_targets`
- thresholds: imports `train_markets._best_f1_threshold`
- engines: CatBoost and/or XGBoost when installed
- output: research CSVs only, no ModelStore writes

The default split is deliberately simple and auditable:
2018 train/calibration split -> 2022 holdout test.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_markets import _best_f1_threshold, _derive_targets, build_features  # noqa: E402


DEFAULT_MERGED = Path("Matches/__merged__/World_Cup__merged.csv")
DEFAULT_BACKBUILT_SIDECAR = Path(
    "data_sources/footystats_world_cup/historical_full_stack_backbuild/"
    "world_cup_historical_backbuilt_fixture_intelligence_sidecar.csv"
)
DEFAULT_API_PLAYER_POWER_SIDECAR = Path(
    "data_sources/footystats_world_cup/api_lagged_player_power_backbuild/"
    "world_cup_api_lagged_player_power_fixture_sidecar.csv"
)
DEFAULT_OUTDIR = Path("data_sources/footystats_world_cup/trainer_native_research_runs")

MARKETS = ["ftr", "btts", "over25", "home_ge2", "away_ge2", "any_team_ge2"]
GROUPS = [
    "trainer_native_macro",
    "trainer_native_full_stack_backbuilt",
    "trainer_native_api_player_power",
    "trainer_native_full_stack_api_player_power",
]
IDENTITY_COLS = [
    "fixture_key",
    "season",
    "match_date",
    "home_team_name",
    "away_team_name",
    "home_team_goal_count",
    "away_team_goal_count",
]


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    text = df.copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            text[col] = text[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            text[col] = text[col].astype("string").fillna("")
    lines = [
        "| " + " | ".join(str(c) for c in text.columns) + " |",
        "| " + " | ".join(["---"] * len(text.columns)) + " |",
    ]
    for _, row in text.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in text.columns) + " |")
    return "\n".join(lines)


def engine_available(engine: str) -> bool:
    try:
        if engine == "catboost":
            return bool(importlib.util.find_spec("catboost"))
        if engine == "xgboost":
            return bool(importlib.util.find_spec("xgboost"))
    except (ImportError, ValueError):
        return False
    return False


def poisson_matrix(lambda_home: float, lambda_away: float, max_goals: int = 8) -> np.ndarray:
    def pmf(lam: float, k: int) -> float:
        return math.exp(-lam) * (lam**k) / math.factorial(k)

    home = np.array([pmf(lambda_home, k) for k in range(max_goals + 1)])
    away = np.array([pmf(lambda_away, k) for k in range(max_goals + 1)])
    mass = np.outer(home, away)
    return mass / mass.sum()


def add_goal_mass_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    base_home = pd.to_numeric(out.get("home_team_goal_count"), errors="coerce").mean()
    base_away = pd.to_numeric(out.get("away_team_goal_count"), errors="coerce").mean()

    lambda_home = []
    lambda_away = []
    p_home = []
    p_draw = []
    p_away = []
    p_home_ge2 = []
    p_away_ge2 = []
    p_btts = []
    p_over25 = []
    for row in out.itertuples(index=False):
        s = pd.Series(row._asdict())
        lh_parts = [
            s.get("home_prior_goals_for_per_match"),
            s.get("away_prior_goals_against_per_match"),
            s.get("home_backbuilt_recent_goals_for_per_match"),
            s.get("away_backbuilt_recent_goals_against_per_match"),
            base_home,
        ]
        la_parts = [
            s.get("away_prior_goals_for_per_match"),
            s.get("home_prior_goals_against_per_match"),
            s.get("away_backbuilt_recent_goals_for_per_match"),
            s.get("home_backbuilt_recent_goals_against_per_match"),
            base_away,
        ]
        lh = float(np.nanmean(pd.to_numeric(pd.Series(lh_parts), errors="coerce")))
        la = float(np.nanmean(pd.to_numeric(pd.Series(la_parts), errors="coerce")))
        lh = float(np.clip(lh, 0.15, 4.5))
        la = float(np.clip(la, 0.15, 4.5))
        mass = poisson_matrix(lh, la)
        lambda_home.append(lh)
        lambda_away.append(la)
        p_home.append(float(np.tril(mass, -1).sum()))
        p_draw.append(float(np.trace(mass)))
        p_away.append(float(np.triu(mass, 1).sum()))
        p_home_ge2.append(float(mass[2:, :].sum()))
        p_away_ge2.append(float(mass[:, 2:].sum()))
        p_btts.append(float(mass[1:, 1:].sum()))
        p_over25.append(float(sum(mass[i, j] for i in range(mass.shape[0]) for j in range(mass.shape[1]) if i + j >= 3)))

    out["lambda_home"] = lambda_home
    out["lambda_away"] = lambda_away
    out["exp_goals_sum"] = out["lambda_home"] + out["lambda_away"]
    out["p_home_pois"] = p_home
    out["p_draw_pois"] = p_draw
    out["p_away_pois"] = p_away
    out["prob_ftr_home"] = p_home
    out["prob_ftr_away"] = p_away
    out["home_ge2_confidence"] = p_home_ge2
    out["away_ge2_confidence"] = p_away_ge2
    out["btts_pois_prob"] = p_btts
    out["over25_pois_prob"] = p_over25
    out["ppg_diff"] = pd.to_numeric(out.get("Pre-Match PPG (Home)"), errors="coerce") - pd.to_numeric(
        out.get("Pre-Match PPG (Away)"), errors="coerce"
    )
    out["over25_rate_diff"] = pd.to_numeric(out.get("home_prior_over25_rate"), errors="coerce") - pd.to_numeric(
        out.get("away_prior_over25_rate"), errors="coerce"
    )
    out["btts_rate_diff"] = pd.to_numeric(out.get("home_prior_btts_rate"), errors="coerce") - pd.to_numeric(
        out.get("away_prior_btts_rate"), errors="coerce"
    )
    return out


def merge_research_sidecar(df: pd.DataFrame, sidecar_path: Path) -> pd.DataFrame:
    if not sidecar_path.exists():
        raise FileNotFoundError(f"research sidecar not found: {sidecar_path}")
    sidecar = pd.read_csv(sidecar_path, low_memory=False)
    keep = ["fixture_key"] + [c for c in sidecar.columns if c != "fixture_key" and c not in df.columns]
    return df.merge(sidecar[keep], on="fixture_key", how="left")


def load_group_dataset(
    merged_path: Path,
    sidecar_path: Path,
    api_player_power_sidecar_path: Path,
    group: str,
) -> pd.DataFrame:
    df = pd.read_csv(merged_path, low_memory=False)
    df = df[pd.to_numeric(df["season"], errors="coerce").isin([2018, 2022])].copy()
    if group in {"trainer_native_full_stack_backbuilt", "trainer_native_full_stack_api_player_power"}:
        df = merge_research_sidecar(df, sidecar_path)
    if group in {"trainer_native_api_player_power", "trainer_native_full_stack_api_player_power"}:
        df = merge_research_sidecar(df, api_player_power_sidecar_path)
    df = add_goal_mass_features(df)
    return df.sort_values(["season", "timestamp", "fixture_key"]).reset_index(drop=True)


def target_for_market(df_labels: pd.DataFrame, market: str) -> pd.Series:
    targets = _derive_targets(df_labels)
    if market in targets:
        return pd.to_numeric(targets[market], errors="coerce").astype("Int64")
    if market == "any_team_ge2":
        ht = pd.to_numeric(df_labels.get("home_team_goal_count"), errors="coerce")
        at = pd.to_numeric(df_labels.get("away_team_goal_count"), errors="coerce")
        return ((ht >= 2) | (at >= 2)).astype("Int64")
    raise ValueError(f"unsupported market: {market}")


def split_train_cal_test(df: pd.DataFrame, y: pd.Series, calibration_frac: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    train_all = df["season"].eq(2018) & y.notna()
    test = df["season"].eq(2022) & y.notna()
    train_idx = df[train_all].sort_values(["timestamp", "fixture_key"]).index
    cut = max(1, int(len(train_idx) * (1.0 - calibration_frac)))
    cut = min(cut, len(train_idx) - 1)
    fit = pd.Series(False, index=df.index)
    cal = pd.Series(False, index=df.index)
    fit.loc[train_idx[:cut]] = True
    cal.loc[train_idx[cut:]] = True
    return fit, cal, test


def prep_xgb(X: pd.DataFrame) -> np.ndarray:
    return X.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)


def train_predict_catboost(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    X_test: pd.DataFrame,
    *,
    market: str,
    seed: int,
    iterations: int,
    threads: int,
) -> tuple[np.ndarray, np.ndarray, float | None, str | None]:
    from catboost import CatBoostClassifier

    is_ftr = market == "ftr"
    model = CatBoostClassifier(
        loss_function="MultiClass" if is_ftr else "Logloss",
        eval_metric="Accuracy" if is_ftr else "AUC",
        iterations=int(iterations),
        learning_rate=0.05,
        depth=8 if is_ftr else 7,
        random_seed=int(seed),
        verbose=False,
        allow_writing_files=False,
        thread_count=int(max(1, threads)),
        od_type="Iter",
        od_wait=75,
    )
    model.fit(X_fit, y_fit, eval_set=(X_cal, y_cal), use_best_model=True)
    proba_cal = np.asarray(model.predict_proba(X_cal), dtype=float)
    proba_test = np.asarray(model.predict_proba(X_test), dtype=float)
    proba_cal = normalize_proba(proba_cal)
    proba_test = normalize_proba(proba_test)
    if is_ftr:
        return proba_test.argmax(axis=1), proba_test, None, None
    thr = _best_f1_threshold(y_cal, proba_cal[:, 1], market=market)
    pred = (proba_test[:, 1] >= float(thr)).astype(int)
    return pred, proba_test, float(thr), "train_markets_f1_bounded"


def train_predict_xgboost(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    X_test: pd.DataFrame,
    *,
    market: str,
    seed: int,
    iterations: int,
    threads: int,
) -> tuple[np.ndarray, np.ndarray, float | None, str | None]:
    from xgboost import XGBClassifier

    is_ftr = market == "ftr"
    params = dict(
        n_estimators=int(iterations),
        max_depth=6,
        learning_rate=0.05,
        subsample=0.80,
        colsample_bytree=0.80,
        min_child_weight=2.0,
        reg_lambda=1.0,
        objective="multi:softprob" if is_ftr else "binary:logistic",
        eval_metric="mlogloss" if is_ftr else "logloss",
        tree_method="hist",
        n_jobs=int(max(1, threads)),
        random_state=int(seed),
    )
    model = XGBClassifier(**params)
    model.fit(prep_xgb(X_fit), y_fit, eval_set=[(prep_xgb(X_cal), y_cal)], verbose=False)
    proba_cal = np.asarray(model.predict_proba(prep_xgb(X_cal)), dtype=float)
    proba_test = np.asarray(model.predict_proba(prep_xgb(X_test)), dtype=float)
    proba_cal = normalize_proba(proba_cal)
    proba_test = normalize_proba(proba_test)
    if is_ftr:
        return proba_test.argmax(axis=1), proba_test, None, None
    thr = _best_f1_threshold(y_cal, proba_cal[:, 1], market=market)
    pred = (proba_test[:, 1] >= float(thr)).astype(int)
    return pred, proba_test, float(thr), "train_markets_f1_bounded"


def normalize_proba(proba: np.ndarray) -> np.ndarray:
    arr = np.asarray(proba, dtype=float)
    arr = np.clip(arr, 0.0, 1.0)
    row_sum = arr.sum(axis=1)
    bad = ~np.isfinite(row_sum) | (row_sum <= 0)
    if bad.any():
        arr[bad, :] = 1.0 / arr.shape[1]
        row_sum = arr.sum(axis=1)
    return arr / row_sum[:, None]


def metric_row(y_test: pd.Series, pred: np.ndarray, proba: np.ndarray, market: str) -> dict:
    labels = [0, 1, 2] if market == "ftr" else [0, 1]
    proba = normalize_proba(proba)
    row = {
        "rows_test": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "log_loss": float(log_loss(y_test, proba, labels=labels)),
        "auc": np.nan,
        "brier": np.nan,
    }
    if market != "ftr":
        try:
            row["auc"] = float(roc_auc_score(y_test, proba[:, 1]))
        except Exception:
            pass
        row["brier"] = float(brier_score_loss(y_test, proba[:, 1]))
    return row


def delta_vs_macro(results: pd.DataFrame) -> pd.DataFrame:
    ok = results[results["status"].eq("OK")].copy() if "status" in results.columns else pd.DataFrame()
    if ok.empty:
        return pd.DataFrame()
    macro = ok[ok["ablation_group"].eq("trainer_native_macro")][
        ["engine", "market", "accuracy", "log_loss", "auc", "brier"]
    ].rename(
        columns={
            "accuracy": "macro_accuracy",
            "log_loss": "macro_log_loss",
            "auc": "macro_auc",
            "brier": "macro_brier",
        }
    )
    out = ok.merge(macro, on=["engine", "market"], how="left")
    out["accuracy_delta_vs_macro"] = out["accuracy"] - out["macro_accuracy"]
    out["log_loss_delta_vs_macro"] = out["log_loss"] - out["macro_log_loss"]
    out["auc_delta_vs_macro"] = out["auc"] - out["macro_auc"]
    out["brier_delta_vs_macro"] = out["brier"] - out["macro_brier"]
    return out.sort_values(["market", "engine", "ablation_group"]).reset_index(drop=True)


def run_one(df_raw: pd.DataFrame, group: str, market: str, engine: str, args: argparse.Namespace) -> tuple[dict, pd.DataFrame]:
    X, df_labels = build_features(df_raw, "World Cup")
    y = target_for_market(df_labels, market)
    valid = y.notna()
    X = X.loc[valid]
    df_labels = df_labels.loc[valid]
    y = y.loc[valid].astype(int)
    fit_mask, cal_mask, test_mask = split_train_cal_test(df_labels, y, args.calibration_frac)
    fit_mask = fit_mask.loc[valid]
    cal_mask = cal_mask.loc[valid]
    test_mask = test_mask.loc[valid]
    if int(fit_mask.sum()) < args.min_fit_rows or int(cal_mask.sum()) < 8 or int(test_mask.sum()) < 20:
        return {"status": "SKIP_LOW_ROWS"}, pd.DataFrame()
    if y.loc[fit_mask].nunique() < 2 or y.loc[cal_mask].nunique() < 2:
        return {"status": "SKIP_SINGLE_CLASS_SPLIT"}, pd.DataFrame()
    if not set(y.loc[test_mask].unique()).issubset(set(y.loc[fit_mask].unique())):
        return {"status": "SKIP_TEST_CLASS_NOT_IN_FIT"}, pd.DataFrame()
    if engine == "catboost":
        pred, proba, threshold, threshold_method = train_predict_catboost(
            X.loc[fit_mask],
            y.loc[fit_mask],
            X.loc[cal_mask],
            y.loc[cal_mask],
            X.loc[test_mask],
            market=market,
            seed=args.random_seed,
            iterations=args.iterations,
            threads=args.threads,
        )
    else:
        pred, proba, threshold, threshold_method = train_predict_xgboost(
            X.loc[fit_mask],
            y.loc[fit_mask],
            X.loc[cal_mask],
            y.loc[cal_mask],
            X.loc[test_mask],
            market=market,
            seed=args.random_seed,
            iterations=args.iterations,
            threads=args.threads,
        )
    metrics = metric_row(y.loc[test_mask], pred, proba, market)
    metrics.update(
        {
            "status": "OK",
            "engine": engine,
            "ablation_group": group,
            "market": market,
            "rows_fit": int(fit_mask.sum()),
            "rows_calibration": int(cal_mask.sum()),
            "n_features": int(X.shape[1]),
            "threshold": threshold,
            "threshold_method": threshold_method,
        }
    )
    pred_rows = df_labels.loc[test_mask, [c for c in IDENTITY_COLS if c in df_labels.columns]].copy()
    pred_rows["engine"] = engine
    pred_rows["ablation_group"] = group
    pred_rows["market"] = market
    pred_rows["target"] = y.loc[test_mask].values
    pred_rows["pred"] = pred
    pred_rows["hit"] = (pred_rows["target"] == pred_rows["pred"]).astype(int)
    for idx in range(proba.shape[1]):
        pred_rows[f"prob_class_{idx}"] = proba[:, idx]
    return metrics, pred_rows


def write_summary(outdir: Path, results: pd.DataFrame, engine_status: pd.DataFrame) -> None:
    ok = results[results["status"].eq("OK")].copy() if "status" in results.columns else pd.DataFrame()
    pivot = (
        ok.pivot_table(index=["market", "ablation_group"], columns="engine", values="accuracy", aggfunc="mean").reset_index()
        if not ok.empty
        else pd.DataFrame()
    )
    delta = delta_vs_macro(results)
    delta_view = delta[
        delta["ablation_group"].ne("trainer_native_macro")
    ][
        [
            "market",
            "engine",
            "ablation_group",
            "accuracy",
            "accuracy_delta_vs_macro",
            "log_loss_delta_vs_macro",
            "auc_delta_vs_macro",
        ]
    ] if not delta.empty else pd.DataFrame()
    lines = [
        "# Trainer-Native World Cup Research Run",
        "",
        "Research-only CatBoost/XGBoost runner using `train_markets.build_features`, `_derive_targets`, and `_best_f1_threshold`.",
        "",
        "## Outputs",
        "",
        f"- `{outdir / 'world_cup_trainer_native_results.csv'}`",
        f"- `{outdir / 'world_cup_trainer_native_predictions.csv'}`",
        f"- `{outdir / 'world_cup_trainer_native_engine_status.csv'}`",
        "",
        "## Engine Status",
        "",
        markdown_table(engine_status),
        "",
        "## Accuracy Pivot",
        "",
        markdown_table(pivot),
        "",
        "## Delta vs Macro",
        "",
        markdown_table(delta_view),
        "",
        "## Notes",
        "",
        "- No ModelStore artifacts are written.",
        "- Feature construction comes from `train_markets.build_features`.",
        "- Binary thresholds use `train_markets._best_f1_threshold` on the 2018 calibration split.",
        "- Default split: 2018 fit/calibration, 2022 holdout test.",
        "- API player-power groups are research-only because current fixture lineup membership is not yet proven by pre-kickoff lineup timestamps.",
    ]
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged", type=Path, default=DEFAULT_MERGED)
    parser.add_argument("--backbuilt-sidecar", type=Path, default=DEFAULT_BACKBUILT_SIDECAR)
    parser.add_argument("--api-player-power-sidecar", type=Path, default=DEFAULT_API_PLAYER_POWER_SIDECAR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--groups", default=",".join(GROUPS))
    parser.add_argument("--markets", default=",".join(MARKETS))
    parser.add_argument("--engines", default="catboost,xgboost")
    parser.add_argument("--iterations", type=int, default=350)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--calibration-frac", type=float, default=0.25)
    parser.add_argument("--min-fit-rows", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    groups = [x.strip() for x in args.groups.split(",") if x.strip()]
    markets = [x.strip().lower() for x in args.markets.split(",") if x.strip()]
    requested_engines = [x.strip().lower() for x in args.engines.split(",") if x.strip()]
    engine_status = pd.DataFrame(
        [{"engine": e, "available": engine_available(e), "used": engine_available(e)} for e in requested_engines]
    )
    engines = engine_status[engine_status["available"]]["engine"].tolist()
    if not engines:
        engine_status.to_csv(args.outdir / "world_cup_trainer_native_engine_status.csv", index=False)
        write_summary(args.outdir, pd.DataFrame(), engine_status)
        print("[warn] no requested engines available; install catboost/xgboost or run in the training env")
        return 0
    results = []
    prediction_frames = []
    for group in groups:
        df_raw = load_group_dataset(args.merged, args.backbuilt_sidecar, args.api_player_power_sidecar, group)
        for market in markets:
            for engine in engines:
                metrics, preds = run_one(df_raw, group, market, engine, args)
                metrics.update({"engine": engine, "ablation_group": group, "market": market})
                results.append(metrics)
                if not preds.empty:
                    prediction_frames.append(preds)
    results_df = pd.DataFrame(results)
    preds_df = pd.concat(prediction_frames, ignore_index=True, sort=False) if prediction_frames else pd.DataFrame()
    results_df.to_csv(args.outdir / "world_cup_trainer_native_results.csv", index=False)
    delta_vs_macro(results_df).to_csv(args.outdir / "world_cup_trainer_native_delta_vs_macro.csv", index=False)
    preds_df.to_csv(args.outdir / "world_cup_trainer_native_predictions.csv", index=False)
    engine_status.to_csv(args.outdir / "world_cup_trainer_native_engine_status.csv", index=False)
    write_summary(args.outdir, results_df, engine_status)
    print(f"[ok] results={len(results_df)} predictions={len(preds_df)} engines={','.join(engines)}")
    print(f"[ok] wrote {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
