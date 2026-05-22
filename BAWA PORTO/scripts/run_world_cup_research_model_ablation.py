#!/usr/bin/env python3
"""Run research-only World Cup model ablations.

This runner trains against the canonical World Cup merged adapter only:
`Matches/__merged__/World_Cup__merged.csv`. It does not write ModelStore
artifacts and does not touch deploy routing.

The default posture scores only currently leak-safe historical groups. CatBoost
and XGBoost are supported when installed, but the script will fall back to local
sklearn/LightGBM engines for smoke tests.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_MERGED = Path("Matches/__merged__/World_Cup__merged.csv")
DEFAULT_FJELSTUL_ARCHIVE = Path("/Users/hughwade/Desktop/FOOTYSTATS_DROP/archive")
DEFAULT_BACKBUILT_SIDECAR = Path(
    "data_sources/footystats_world_cup/historical_full_stack_backbuild/"
    "world_cup_historical_backbuilt_fixture_intelligence_sidecar.csv"
)
DEFAULT_OUTDIR = Path("data_sources/footystats_world_cup/model_ablation_runs")

MARKETS = ["FTR", "BTTS", "OU25", "HOME_TG15", "AWAY_TG15", "ANY_TEAM_TG15"]
LEAK_SAFE_GROUPS = ["macro_only", "macro_plus_fjelstul_history"]

BASE_ID_COLS = [
    "fixture_key",
    "season",
    "match_date",
    "home_team_name",
    "away_team_name",
    "actual_ftr_label",
    "actual_btts_label",
    "actual_over25_label",
]

MACRO_FEATURES = [
    "group_matchday",
    "neutral_venue_flag",
    "is_knockout_stage",
    "is_first_group_match",
    "odds_ft_home_team_win",
    "odds_ft_draw",
    "odds_ft_away_team_win",
    "odds_ft_over15",
    "odds_ft_over25",
    "odds_ft_under25",
    "odds_btts_yes",
    "odds_btts_no",
    "Pre-Match PPG (Home)",
    "Pre-Match PPG (Away)",
    "Home Team Pre-Match xG",
    "Away Team Pre-Match xG",
    "average_goals_per_match_pre_match",
    "btts_percentage_pre_match",
    "over_15_percentage_pre_match",
    "over_25_percentage_pre_match",
]

LAGGED_TOURNAMENT_FEATURES = [
    "home_prior_matches_played",
    "home_prior_points",
    "home_prior_points_per_match",
    "home_prior_goal_diff",
    "home_prior_goals_for_per_match",
    "home_prior_goals_against_per_match",
    "home_prior_btts_rate",
    "home_prior_over25_rate",
    "away_prior_matches_played",
    "away_prior_points",
    "away_prior_points_per_match",
    "away_prior_goal_diff",
    "away_prior_goals_for_per_match",
    "away_prior_goals_against_per_match",
    "away_prior_btts_rate",
    "away_prior_over25_rate",
]

FJELSTUL_LAGGED_FEATURES = [
    "home_fjelstul_prior_matches",
    "home_fjelstul_prior_win_rate",
    "home_fjelstul_prior_draw_rate",
    "home_fjelstul_prior_loss_rate",
    "home_fjelstul_prior_goals_for_per_match",
    "home_fjelstul_prior_goals_against_per_match",
    "home_fjelstul_prior_goal_diff_per_match",
    "home_fjelstul_prior_knockout_match_rate",
    "away_fjelstul_prior_matches",
    "away_fjelstul_prior_win_rate",
    "away_fjelstul_prior_draw_rate",
    "away_fjelstul_prior_loss_rate",
    "away_fjelstul_prior_goals_for_per_match",
    "away_fjelstul_prior_goals_against_per_match",
    "away_fjelstul_prior_goal_diff_per_match",
    "away_fjelstul_prior_knockout_match_rate",
    "fjelstul_prior_win_rate_delta",
    "fjelstul_prior_draw_rate_delta",
    "fjelstul_prior_goals_for_per_match_delta",
    "fjelstul_prior_goals_against_per_match_delta",
    "fjelstul_prior_goal_diff_per_match_delta",
]

BACKBUILT_PREFIXES = (
    "home_backbuilt_",
    "away_backbuilt_",
    "backbuilt_",
    "historical_local_h2h_",
    "historical_venue_",
)


def slugify(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


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
    if engine == "catboost":
        return bool(importlib.util.find_spec("catboost"))
    if engine == "xgboost":
        return bool(importlib.util.find_spec("xgboost"))
    if engine == "lightgbm":
        return bool(importlib.util.find_spec("lightgbm"))
    return True


def build_model(engine: str, n_classes: int, random_seed: int):
    if engine == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=180,
            learning_rate=0.04,
            depth=4,
            loss_function="MultiClass" if n_classes > 2 else "Logloss",
            random_seed=random_seed,
            verbose=False,
            allow_writing_files=False,
        )
    if engine == "xgboost":
        from xgboost import XGBClassifier

        objective = "multi:softprob" if n_classes > 2 else "binary:logistic"
        return XGBClassifier(
            n_estimators=180,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            objective=objective,
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
            random_state=random_seed,
        )
    if engine == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=180,
            learning_rate=0.04,
            max_depth=3,
            num_leaves=15,
            objective="multiclass" if n_classes > 2 else "binary",
            random_state=random_seed,
            verbose=-1,
        )
    if engine == "logistic":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=random_seed)),
            ]
        )
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=160,
        max_leaf_nodes=16,
        random_state=random_seed,
    )


def load_fjelstul_lagged_priors(archive: Path) -> pd.DataFrame:
    path = archive / "team_appearances.csv"
    if not path.exists():
        return pd.DataFrame()
    team = pd.read_csv(path, low_memory=False)
    team["season"] = team["tournament_name"].astype(str).str.extract(r"(\d{4})").astype(float)
    team["team_slug"] = team["team_name"].map(slugify)
    rows = []
    for season in [2018, 2022]:
        prior = team[pd.to_numeric(team["season"], errors="coerce").lt(season)].copy()
        grouped = prior.groupby("team_slug", dropna=False).agg(
            fjelstul_prior_matches=("team_slug", "size"),
            fjelstul_prior_win_rate=("win", "mean"),
            fjelstul_prior_draw_rate=("draw", "mean"),
            fjelstul_prior_loss_rate=("lose", "mean"),
            fjelstul_prior_goals_for_per_match=("goals_for", "mean"),
            fjelstul_prior_goals_against_per_match=("goals_against", "mean"),
            fjelstul_prior_goal_diff_per_match=("goal_differential", "mean"),
            fjelstul_prior_knockout_match_rate=("knockout_stage", "mean"),
        ).reset_index()
        grouped["season"] = season
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def side_join_priors(df: pd.DataFrame, priors: pd.DataFrame, side: str) -> pd.DataFrame:
    out = df.copy()
    out[f"{side}_team_slug"] = out[f"{side}_team_name"].map(slugify)
    side_priors = priors.rename(columns={c: f"{side}_{c}" for c in priors.columns if c not in {"team_slug", "season"}})
    side_priors = side_priors.rename(columns={"team_slug": f"{side}_team_slug"})
    return out.merge(side_priors, on=["season", f"{side}_team_slug"], how="left")


def add_fjelstul_lagged_features(df: pd.DataFrame, archive: Path) -> pd.DataFrame:
    priors = load_fjelstul_lagged_priors(archive)
    if priors.empty:
        return df
    out = side_join_priors(df, priors, "home")
    out = side_join_priors(out, priors, "away")
    for suffix in [
        "win_rate",
        "draw_rate",
        "goals_for_per_match",
        "goals_against_per_match",
        "goal_diff_per_match",
    ]:
        out[f"fjelstul_prior_{suffix}_delta"] = (
            pd.to_numeric(out.get(f"home_fjelstul_prior_{suffix}"), errors="coerce")
            - pd.to_numeric(out.get(f"away_fjelstul_prior_{suffix}"), errors="coerce")
        )
    return out


def load_dataset(merged_path: Path, fjelstul_archive: Path, backbuilt_sidecar: Path | None) -> pd.DataFrame:
    df = pd.read_csv(merged_path, low_memory=False)
    df = df[pd.to_numeric(df["season"], errors="coerce").isin([2018, 2022])].copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df["target_home_tg15_over"] = (pd.to_numeric(df["home_team_goal_count"], errors="coerce") >= 2).astype(int)
    df["target_away_tg15_over"] = (pd.to_numeric(df["away_team_goal_count"], errors="coerce") >= 2).astype(int)
    df["target_any_team_tg15_over"] = (
        df["target_home_tg15_over"].eq(1) | df["target_away_tg15_over"].eq(1)
    ).astype(int)
    df = add_fjelstul_lagged_features(df, fjelstul_archive)
    if backbuilt_sidecar and backbuilt_sidecar.exists():
        sidecar = pd.read_csv(backbuilt_sidecar, low_memory=False)
        if "fixture_key" in sidecar.columns:
            keep = ["fixture_key"] + [c for c in sidecar.columns if c != "fixture_key" and c not in df.columns]
            df = df.merge(sidecar[keep], on="fixture_key", how="left")
    return df.sort_values(["season", "match_date", "fixture_key"]).reset_index(drop=True)


def target_for_market(df: pd.DataFrame, market: str) -> tuple[pd.Series, list[str]]:
    if market == "FTR":
        return df["actual_ftr_label"].map({"HOME": 0, "DRAW": 1, "AWAY": 2}).astype("Int64"), ["HOME", "DRAW", "AWAY"]
    if market == "BTTS":
        return pd.to_numeric(df["actual_btts_label"], errors="coerce").astype("Int64"), ["NO", "YES"]
    if market == "OU25":
        return pd.to_numeric(df["actual_over25_label"], errors="coerce").astype("Int64"), ["UNDER", "OVER"]
    if market == "HOME_TG15":
        return pd.to_numeric(df["target_home_tg15_over"], errors="coerce").astype("Int64"), ["UNDER", "OVER"]
    if market == "AWAY_TG15":
        return pd.to_numeric(df["target_away_tg15_over"], errors="coerce").astype("Int64"), ["UNDER", "OVER"]
    return pd.to_numeric(df["target_any_team_tg15_over"], errors="coerce").astype("Int64"), ["UNDER", "OVER"]


def group_features(df: pd.DataFrame, group: str) -> list[str]:
    cols = []
    if group in {"macro_only", "macro_plus_fjelstul_history", "full_stack_backbuilt"}:
        cols.extend(MACRO_FEATURES)
    if group in {"macro_plus_fjelstul_history", "full_stack_backbuilt"}:
        cols.extend(LAGGED_TOURNAMENT_FEATURES)
        cols.extend(FJELSTUL_LAGGED_FEATURES)
    if group == "full_stack_backbuilt":
        cols.extend([c for c in df.columns if c.startswith(BACKBUILT_PREFIXES)])
    out = []
    seen = set()
    for col in cols:
        if col in df.columns and col not in seen:
            seen.add(col)
            out.append(col)
    return out


def prepare_X(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    X = df[list(cols)].copy()
    for col in X.columns:
        if X[col].dtype == "object" or str(X[col].dtype).startswith("string"):
            X[col] = X[col].astype("string").fillna("MISSING")
            X[col] = pd.factorize(X[col], sort=True)[0].astype(float)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X.replace([np.inf, -np.inf], np.nan)


def align_proba(model, proba: np.ndarray, labels: list[int]) -> np.ndarray:
    classes = list(getattr(model, "classes_", labels))
    if isinstance(model, Pipeline):
        classes = list(model.named_steps["model"].classes_)
    aligned = np.zeros((proba.shape[0], len(labels)))
    for idx, cls in enumerate(classes):
        if cls in labels:
            aligned[:, labels.index(cls)] = proba[:, idx]
    row_sum = aligned.sum(axis=1)
    missing = row_sum <= 0
    if missing.any():
        aligned[missing, :] = 1.0 / len(labels)
    aligned[~missing, :] = aligned[~missing, :] / row_sum[~missing, None]
    return aligned


def score_predictions(y_true: pd.Series, pred: np.ndarray, proba: np.ndarray, labels: list[int], market: str) -> dict:
    out = {
        "rows_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "log_loss": float(log_loss(y_true, proba, labels=labels)),
        "auc": np.nan,
        "brier": np.nan,
    }
    if market != "FTR" and 1 in labels:
        positive_idx = labels.index(1)
        try:
            out["auc"] = float(roc_auc_score(y_true, proba[:, positive_idx]))
        except Exception:
            out["auc"] = np.nan
        out["brier"] = float(brier_score_loss(y_true, proba[:, positive_idx]))
    return out


def run_model_score(df: pd.DataFrame, group: str, market: str, engine: str, random_seed: int) -> tuple[dict, pd.DataFrame]:
    y, label_names = target_for_market(df, market)
    labels = list(range(len(label_names))) if market == "FTR" else [0, 1]
    cols = group_features(df, group)
    valid = y.notna()
    data = df.loc[valid].copy()
    y = y.loc[valid].astype(int)
    train_mask = data["season"].eq(2018)
    test_mask = data["season"].eq(2022)
    if not cols:
        return {"status": "SKIP_NO_FEATURES", "n_features": 0}, pd.DataFrame()
    if int(train_mask.sum()) < 20 or int(test_mask.sum()) < 20:
        return {"status": "SKIP_LOW_ROWS", "n_features": len(cols)}, pd.DataFrame()
    if y.loc[train_mask].nunique() < 2:
        return {"status": "SKIP_SINGLE_CLASS_TRAIN", "n_features": len(cols)}, pd.DataFrame()
    if not set(y.loc[test_mask].unique()).issubset(set(y.loc[train_mask].unique())):
        return {"status": "SKIP_TEST_CLASS_NOT_IN_TRAIN", "n_features": len(cols)}, pd.DataFrame()
    X = prepare_X(data, cols)
    model = build_model(engine, len(labels), random_seed)
    model.fit(X.loc[train_mask], y.loc[train_mask])
    pred = model.predict(X.loc[test_mask])
    raw_proba = model.predict_proba(X.loc[test_mask])
    proba = align_proba(model, raw_proba, labels)
    metrics = score_predictions(y.loc[test_mask], pred, proba, labels, market)
    metrics.update(
        {
            "status": "OK",
            "engine": engine,
            "ablation_group": group,
            "market": market,
            "rows_train": int(train_mask.sum()),
            "n_features": int(len(cols)),
        }
    )
    pred_rows = data.loc[test_mask, BASE_ID_COLS].copy()
    pred_rows["market"] = market
    pred_rows["ablation_group"] = group
    pred_rows["engine"] = engine
    pred_rows["target_class"] = y.loc[test_mask].values
    pred_rows["pred_class"] = pred
    pred_rows["hit"] = (pred_rows["target_class"] == pred_rows["pred_class"]).astype(int)
    for idx, name in enumerate(label_names):
        pred_rows[f"prob_{name.lower()}"] = proba[:, idx]
    return metrics, pred_rows


def poisson_score_matrix(lambda_home: float, lambda_away: float, max_goals: int = 8) -> np.ndarray:
    def pmf(lam: float, k: int) -> float:
        return math.exp(-lam) * (lam**k) / math.factorial(k)

    home = np.array([pmf(lambda_home, k) for k in range(max_goals + 1)])
    away = np.array([pmf(lambda_away, k) for k in range(max_goals + 1)])
    mass = np.outer(home, away)
    return mass / mass.sum()


def row_lambdas(row: pd.Series, train_base_home: float, train_base_away: float) -> tuple[float, float]:
    home_for = pd.to_numeric(pd.Series([row.get("home_prior_goals_for_per_match")]), errors="coerce").iloc[0]
    away_against = pd.to_numeric(pd.Series([row.get("away_prior_goals_against_per_match")]), errors="coerce").iloc[0]
    away_for = pd.to_numeric(pd.Series([row.get("away_prior_goals_for_per_match")]), errors="coerce").iloc[0]
    home_against = pd.to_numeric(pd.Series([row.get("home_prior_goals_against_per_match")]), errors="coerce").iloc[0]
    lam_home = np.nanmean([home_for, away_against, train_base_home])
    lam_away = np.nanmean([away_for, home_against, train_base_away])
    return float(np.clip(lam_home, 0.15, 4.5)), float(np.clip(lam_away, 0.15, 4.5))


def run_goal_mass_baseline(df: pd.DataFrame, market: str) -> tuple[dict, pd.DataFrame]:
    y, label_names = target_for_market(df, market)
    valid = y.notna()
    data = df.loc[valid].copy()
    y = y.loc[valid].astype(int)
    train = data[data["season"].eq(2018)]
    test = data[data["season"].eq(2022)]
    if len(train) < 20 or len(test) < 20:
        return {"status": "SKIP_LOW_ROWS", "engine": "goal_mass"}, pd.DataFrame()
    base_home = pd.to_numeric(train["home_team_goal_count"], errors="coerce").mean()
    base_away = pd.to_numeric(train["away_team_goal_count"], errors="coerce").mean()
    probs = []
    preds = []
    labels = [0, 1, 2] if market == "FTR" else [0, 1]
    for row in test.itertuples(index=False):
        s = pd.Series(row._asdict())
        lh, la = row_lambdas(s, base_home, base_away)
        mass = poisson_score_matrix(lh, la)
        home_win = np.tril(mass, -1).sum()
        draw = np.trace(mass)
        away_win = np.triu(mass, 1).sum()
        btts_yes = mass[1:, 1:].sum()
        ou25_over = sum(mass[i, j] for i in range(mass.shape[0]) for j in range(mass.shape[1]) if i + j >= 3)
        home_tg15 = mass[2:, :].sum()
        away_tg15 = mass[:, 2:].sum()
        any_tg15 = home_tg15 + away_tg15 - mass[2:, 2:].sum()
        if market == "FTR":
            p = np.array([home_win, draw, away_win])
        elif market == "BTTS":
            p = np.array([1 - btts_yes, btts_yes])
        elif market == "OU25":
            p = np.array([1 - ou25_over, ou25_over])
        elif market == "HOME_TG15":
            p = np.array([1 - home_tg15, home_tg15])
        elif market == "AWAY_TG15":
            p = np.array([1 - away_tg15, away_tg15])
        else:
            p = np.array([1 - any_tg15, any_tg15])
        p = p / p.sum()
        probs.append(p)
        preds.append(int(np.argmax(p)))
    proba = np.vstack(probs)
    pred = np.array(preds)
    metrics = score_predictions(y.loc[test.index], pred, proba, labels, market)
    metrics.update(
        {
            "status": "OK",
            "engine": "goal_mass_poisson",
            "ablation_group": "goal_mass_lagged_team_state",
            "market": market,
            "rows_train": int(len(train)),
            "n_features": 4,
        }
    )
    pred_rows = test[BASE_ID_COLS].copy()
    pred_rows["market"] = market
    pred_rows["ablation_group"] = "goal_mass_lagged_team_state"
    pred_rows["engine"] = "goal_mass_poisson"
    pred_rows["target_class"] = y.loc[test.index].values
    pred_rows["pred_class"] = pred
    pred_rows["hit"] = (pred_rows["target_class"] == pred_rows["pred_class"]).astype(int)
    for idx, name in enumerate(label_names):
        pred_rows[f"prob_{name.lower()}"] = proba[:, idx]
    return metrics, pred_rows


def write_summary(outdir: Path, results: pd.DataFrame, engine_status: pd.DataFrame) -> None:
    ok = results[results["status"].eq("OK")].copy()
    pivot = (
        ok.pivot_table(
            index=["market", "ablation_group"],
            columns="engine",
            values="accuracy",
            aggfunc="mean",
        )
        .reset_index()
        if not ok.empty
        else pd.DataFrame()
    )
    lines = [
        "# World Cup Research Model Ablation",
        "",
        "Research-only model runner over canonical `Matches/__merged__/World_Cup__merged.csv`.",
        "",
        "## Outputs",
        "",
        f"- `{outdir / 'world_cup_research_model_ablation_results.csv'}`",
        f"- `{outdir / 'world_cup_research_model_ablation_predictions.csv'}`",
        f"- `{outdir / 'world_cup_research_engine_status.csv'}`",
        "",
        "## Engine Status",
        "",
        markdown_table(engine_status),
        "",
        "## Accuracy Pivot",
        "",
        markdown_table(pivot),
        "",
        "## Notes",
        "",
        "- Train/test split is 2018 World Cup train, 2022 World Cup test.",
        "- No ModelStore artifacts are written.",
        "- CatBoost/XGBoost rows appear only when those packages are installed locally.",
        "- Full-stack backbuilt groups should only be judged after timestamp-safe 2018/2022 sidecars exist.",
    ]
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged", type=Path, default=DEFAULT_MERGED)
    parser.add_argument("--fjelstul-archive", type=Path, default=DEFAULT_FJELSTUL_ARCHIVE)
    parser.add_argument("--backbuilt-sidecar", type=Path, default=DEFAULT_BACKBUILT_SIDECAR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--engines", default="auto", help="Comma list or auto. Options: catboost,xgboost,lightgbm,sklearn_hgb,logistic")
    parser.add_argument("--groups", default="leak_safe", help="Comma list, leak_safe, or all_available")
    parser.add_argument("--markets", default=",".join(MARKETS))
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.engines == "auto":
        requested_engines = ["catboost", "xgboost", "lightgbm", "sklearn_hgb", "logistic"]
    else:
        requested_engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    engine_status = pd.DataFrame(
        [
            {"engine": engine, "available": engine_available(engine), "used": engine_available(engine)}
            for engine in requested_engines
        ]
    )
    engines = engine_status[engine_status["available"]]["engine"].tolist()
    if args.groups == "leak_safe":
        groups = LEAK_SAFE_GROUPS
    elif args.groups == "all_available":
        groups = LEAK_SAFE_GROUPS + (["full_stack_backbuilt"] if args.backbuilt_sidecar.exists() else [])
    else:
        groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    markets = [m.strip().upper() for m in args.markets.split(",") if m.strip()]
    df = load_dataset(args.merged, args.fjelstul_archive, args.backbuilt_sidecar)
    result_rows = []
    prediction_frames = []
    for market in markets:
        metrics, preds = run_goal_mass_baseline(df, market)
        metrics.update({"market": market})
        result_rows.append(metrics)
        if not preds.empty:
            prediction_frames.append(preds)
        for group in groups:
            for engine in engines:
                metrics, preds = run_model_score(df, group, market, engine, args.random_seed)
                metrics.update({"engine": engine, "ablation_group": group, "market": market})
                result_rows.append(metrics)
                if not preds.empty:
                    prediction_frames.append(preds)
    results = pd.DataFrame(result_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True, sort=False) if prediction_frames else pd.DataFrame()
    results.to_csv(args.outdir / "world_cup_research_model_ablation_results.csv", index=False)
    predictions.to_csv(args.outdir / "world_cup_research_model_ablation_predictions.csv", index=False)
    engine_status.to_csv(args.outdir / "world_cup_research_engine_status.csv", index=False)
    write_summary(args.outdir, results, engine_status)
    print(f"[ok] results={len(results)} predictions={len(predictions)} engines={','.join(engines)}")
    print(f"[ok] wrote {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
