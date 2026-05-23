#!/usr/bin/env python3
"""Chronological walk-forward validation for promising API-Football ablation cells.

This is research-only. It consumes the offline ablation deltas, selects promising
league/market/family cells, and validates them by training on earlier seasons
and testing on later seasons. It does not write ModelStore artifacts or change
deploy behavior.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.api_football.feature_family_stacks import baseline_cols, family_cols  # noqa: E402


DEFAULT_DELTAS = Path(
    "reports/2026-05-06/api_football_foundation_safe_ablations/api_foundation_safe_ablation_deltas.csv"
)
DEFAULT_HYBRID_DIR = Path("data_sources/hybrid")
DEFAULT_OUTDIR = Path("reports/2026-05-06/api_football_foundation_safe_walkforward")

TARGETS = {
    "FTR": ("target_ftr_home", "target_ftr_draw", "target_ftr_away"),
    "BTTS": ("target_btts_yes",),
    "OU25": ("target_ou25_over",),
}


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


def league_tag(league: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", league).strip("_")


def select_candidates(
    deltas_path: Path,
    *,
    min_rows_test: int,
    min_accuracy_delta: float,
    require_logloss_improvement: bool,
    limit: int,
) -> pd.DataFrame:
    deltas = pd.read_csv(deltas_path)
    candidates = deltas[deltas["stack"].ne("baseline") & deltas["status"].eq("OK")].copy()
    candidates = candidates[candidates["rows_test"].ge(min_rows_test)]
    candidates = candidates[candidates["accuracy_delta_vs_baseline"].ge(min_accuracy_delta)]
    if require_logloss_improvement:
        candidates = candidates[candidates["log_loss_delta_vs_baseline"].le(0)]
    candidates = candidates.sort_values(
        ["accuracy_delta_vs_baseline", "log_loss_delta_vs_baseline", "rows_test"],
        ascending=[False, True, False],
    )
    candidates = candidates.groupby(["league", "market"], as_index=False).head(1)
    candidates = candidates.sort_values(
        ["accuracy_delta_vs_baseline", "log_loss_delta_vs_baseline", "rows_test"],
        ascending=[False, True, False],
    )
    if limit:
        candidates = candidates.head(limit)
    return candidates.reset_index(drop=True)


def load_league_training(hybrid_dir: Path, league: str) -> pd.DataFrame:
    tag = league_tag(league)
    files = sorted(hybrid_dir.glob(f"hybrid_match_training__{tag}__20*.csv"))
    frames = []
    for path in files:
        frame = pd.read_csv(path, low_memory=False)
        match = re.search(r"__(20\d{2})\.csv$", path.name)
        frame["__source_year"] = int(match.group(1)) if match else np.nan
        frame["__source_file"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    if "match_date" in df.columns:
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
        df["__year"] = df["match_date"].dt.year.fillna(df["__source_year"]).astype("Int64")
    else:
        df["__year"] = pd.to_numeric(df["__source_year"], errors="coerce").astype("Int64")
    df = df[df["__year"].notna()].copy()
    return df.sort_values(["__year", "match_date"] if "match_date" in df.columns else ["__year"]).reset_index(drop=True)


def stack_cols(df: pd.DataFrame, families: Sequence[str]) -> list[str]:
    cols = list(baseline_cols(df))
    for family in families:
        if family:
            cols.extend(family_cols(df, family))
    seen = set()
    out = []
    for col in cols:
        if col in seen or col not in df.columns:
            continue
        seen.add(col)
        out.append(col)
    return out


def build_target(df: pd.DataFrame, market: str) -> pd.Series:
    targets = TARGETS[market]
    if market == "FTR":
        return (
            df[list(targets)]
            .idxmax(axis=1)
            .map({"target_ftr_home": 0, "target_ftr_draw": 1, "target_ftr_away": 2})
            .astype("Int64")
        )
    return pd.to_numeric(df[targets[0]], errors="coerce").astype("Int64")


def prepare_features(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    X = df[list(cols)].copy()
    for col in X.columns:
        if X[col].dtype == "object" or str(X[col].dtype).startswith("string"):
            X[col] = X[col].astype("string").fillna("MISSING")
            X[col] = pd.factorize(X[col], sort=True)[0].astype(float)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X.replace([np.inf, -np.inf], np.nan)


def fit_score(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    market: str,
    families: Sequence[str],
    random_seed: int,
) -> dict:
    cols = stack_cols(train, families)
    target_cols = TARGETS[market]
    train_valid = build_target(train, market).notna()
    test_valid = build_target(test, market).notna()
    for target in target_cols:
        train_valid &= train[target].notna()
        test_valid &= test[target].notna()
    train = train.loc[train_valid].copy()
    test = test.loc[test_valid].copy()
    y_train = build_target(train, market).astype(int)
    y_test = build_target(test, market).astype(int)
    if len(train) < 80 or len(test) < 20 or y_train.nunique() < 2 or y_test.nunique() < 2:
        return {"status": "SKIP_LOW_OR_SINGLE_CLASS", "rows_train": len(train), "rows_test": len(test)}
    if not set(y_test.unique()).issubset(set(y_train.unique())):
        return {"status": "SKIP_TEST_CLASS_NOT_IN_TRAIN", "rows_train": len(train), "rows_test": len(test)}

    X_train = prepare_features(train, cols)
    X_test = prepare_features(test, cols)
    X_test = X_test.reindex(columns=X_train.columns)
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=120,
        max_leaf_nodes=24,
        random_state=random_seed,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    labels = sorted(model.classes_)
    row = {
        "status": "OK",
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "n_features": int(len(cols)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "log_loss": float(log_loss(y_test, proba, labels=labels)),
        "auc": np.nan,
    }
    if market != "FTR":
        try:
            positive_idx = list(model.classes_).index(1)
            row["auc"] = float(roc_auc_score(y_test, proba[:, positive_idx]))
        except Exception:
            row["auc"] = np.nan
    return row


def classify_policy(row: pd.Series) -> str:
    folds = int(row.get("folds_ok", 0))
    acc_delta = float(row.get("accuracy_delta_vs_baseline", np.nan))
    ll_delta = float(row.get("log_loss_delta_vs_baseline", np.nan))
    positive_folds = int(row.get("folds_accuracy_positive", 0))
    ll_positive = int(row.get("folds_logloss_improved", 0))
    if folds >= 2 and acc_delta >= 0.03 and ll_delta <= -0.02 and positive_folds == folds and ll_positive >= max(1, folds - 1):
        return "WALK_FORWARD_PROMISING"
    if folds >= 2 and acc_delta > 0 and ll_delta <= 0 and positive_folds >= max(1, folds - 1):
        return "WATCH_WITH_CONFIRM"
    if folds >= 1 and acc_delta > 0:
        return "MICRO_OR_NOISY"
    return "REJECT_FOR_NOW"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deltas", default=str(DEFAULT_DELTAS))
    parser.add_argument("--hybrid-dir", default=str(DEFAULT_HYBRID_DIR))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--min-rows-test", type=int, default=100)
    parser.add_argument("--min-accuracy-delta", type=float, default=0.02)
    parser.add_argument("--allow-positive-logloss", action="store_true")
    parser.add_argument("--limit-candidates", type=int, default=12)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    candidates = select_candidates(
        Path(args.deltas),
        min_rows_test=args.min_rows_test,
        min_accuracy_delta=args.min_accuracy_delta,
        require_logloss_improvement=not args.allow_positive_logloss,
        limit=args.limit_candidates,
    )
    candidates.to_csv(outdir / "api_walkforward_selected_candidates.csv", index=False)

    fold_rows = []
    for _, candidate in candidates.iterrows():
        league = str(candidate["league"])
        market = str(candidate["market"])
        families = tuple(str(candidate.get("families", "")).split("|")) if pd.notna(candidate.get("families")) else ()
        df = load_league_training(Path(args.hybrid_dir), league)
        if df.empty:
            fold_rows.append(
                {
                    "league": league,
                    "market": market,
                    "api_stack": candidate["stack"],
                    "test_year": np.nan,
                    "status": "MISSING_TRAINING",
                }
            )
            continue
        years = sorted(int(y) for y in df["__year"].dropna().unique())
        for test_year in years[1:]:
            train = df[df["__year"].lt(test_year)].copy()
            test = df[df["__year"].eq(test_year)].copy()
            base = fit_score(train, test, market=market, families=(), random_seed=args.random_seed)
            api = fit_score(train, test, market=market, families=families, random_seed=args.random_seed)
            row = {
                "league": league,
                "market": market,
                "api_stack": candidate["stack"],
                "families": "|".join(families),
                "test_year": test_year,
                "baseline_status": base.get("status"),
                "api_status": api.get("status"),
                "baseline_rows_train": base.get("rows_train", np.nan),
                "baseline_rows_test": base.get("rows_test", np.nan),
                "api_n_features": api.get("n_features", np.nan),
                "baseline_accuracy": base.get("accuracy", np.nan),
                "api_accuracy": api.get("accuracy", np.nan),
                "accuracy_delta_vs_baseline": api.get("accuracy", np.nan) - base.get("accuracy", np.nan),
                "baseline_log_loss": base.get("log_loss", np.nan),
                "api_log_loss": api.get("log_loss", np.nan),
                "log_loss_delta_vs_baseline": api.get("log_loss", np.nan) - base.get("log_loss", np.nan),
                "baseline_auc": base.get("auc", np.nan),
                "api_auc": api.get("auc", np.nan),
                "auc_delta_vs_baseline": api.get("auc", np.nan) - base.get("auc", np.nan),
            }
            row["fold_status"] = "OK" if row["baseline_status"] == "OK" and row["api_status"] == "OK" else "SKIP"
            fold_rows.append(row)

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(outdir / "api_walkforward_fold_results.csv", index=False)
    ok = folds[folds["fold_status"].eq("OK")].copy()
    if ok.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            ok.groupby(["league", "market", "api_stack", "families"], dropna=False)
            .agg(
                folds_ok=("test_year", "count"),
                rows_test=("baseline_rows_test", "sum"),
                baseline_accuracy=("baseline_accuracy", "mean"),
                api_accuracy=("api_accuracy", "mean"),
                accuracy_delta_vs_baseline=("accuracy_delta_vs_baseline", "mean"),
                baseline_log_loss=("baseline_log_loss", "mean"),
                api_log_loss=("api_log_loss", "mean"),
                log_loss_delta_vs_baseline=("log_loss_delta_vs_baseline", "mean"),
                baseline_auc=("baseline_auc", "mean"),
                api_auc=("api_auc", "mean"),
                auc_delta_vs_baseline=("auc_delta_vs_baseline", "mean"),
                folds_accuracy_positive=("accuracy_delta_vs_baseline", lambda s: int((s > 0).sum())),
                folds_logloss_improved=("log_loss_delta_vs_baseline", lambda s: int((s < 0).sum())),
            )
            .reset_index()
        )
        summary["walkforward_bucket"] = summary.apply(classify_policy, axis=1)
        summary = summary.sort_values(
            ["walkforward_bucket", "accuracy_delta_vs_baseline", "log_loss_delta_vs_baseline"],
            ascending=[True, False, True],
        )
    summary.to_csv(outdir / "api_walkforward_candidate_summary.csv", index=False)

    bucket_counts = (
        summary.groupby("walkforward_bucket", dropna=False).size().reset_index(name="cells")
        if not summary.empty
        else pd.DataFrame(columns=["walkforward_bucket", "cells"])
    )
    report = [
        "# API-Football Foundation-Safe Walk-Forward Validation",
        "",
        "Research-only chronological validation for promising API ablation cells.",
        "No ModelStore artifacts, production training, deploy gates, or API calls were used.",
        "",
        "## Selected Candidates",
        markdown_table(
            candidates[
                [
                    "league",
                    "market",
                    "stack",
                    "families",
                    "rows_test",
                    "accuracy_delta_vs_baseline",
                    "log_loss_delta_vs_baseline",
                    "auc_delta_vs_baseline",
                ]
            ]
        ),
        "",
        "## Walk-Forward Bucket Counts",
        markdown_table(bucket_counts),
        "",
        "## Candidate Summary",
        markdown_table(
            summary[
                [
                    "walkforward_bucket",
                    "league",
                    "market",
                    "api_stack",
                    "folds_ok",
                    "rows_test",
                    "baseline_accuracy",
                    "api_accuracy",
                    "accuracy_delta_vs_baseline",
                    "log_loss_delta_vs_baseline",
                    "auc_delta_vs_baseline",
                    "folds_accuracy_positive",
                    "folds_logloss_improved",
                ]
            ]
            if not summary.empty
            else summary
        ),
        "",
        "## Operating Decision",
        (
            "Only WALK_FORWARD_PROMISING cells should be considered for a later shadow feature flag. "
            "WATCH_WITH_CONFIRM cells need additional league-specific thresholding. "
            "MICRO_OR_NOISY and REJECT_FOR_NOW must stay out of live restoration."
        ),
    ]
    (outdir / "api_football_foundation_safe_walkforward_validation.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"WROTE {outdir}")
    print(f"candidates={len(candidates)} folds={len(folds)}")


if __name__ == "__main__":
    main()
