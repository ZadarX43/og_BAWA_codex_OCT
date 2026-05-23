#!/usr/bin/env python3
"""Run offline API-Football ablations on foundation-safe leagues only.

This consumes the API coverage readiness table and local hybrid training CSVs.
It does not fetch API data, train production artifacts, or change live routing.
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


DEFAULT_READINESS = Path("reports/2026-05-06/api_football_feature_coverage_audit/api_football_enrichment_readiness.csv")
DEFAULT_HYBRID_DIR = Path("data_sources/hybrid")
DEFAULT_OUTDIR = Path("reports/2026-05-06/api_football_foundation_safe_ablations")

SAFE_BUCKETS = {"SAFE_ENRICHMENT", "FOUNDATION_SAFE_NOISY_EXTRAS"}
STACKS = [
    ("baseline", ()),
    ("baseline_plus_team", ("team",)),
    ("baseline_plus_team_lineup", ("team", "lineup")),
    ("baseline_plus_team_lineup_injury", ("team", "lineup", "injury")),
    ("baseline_plus_team_lineup_injury_event", ("team", "lineup", "injury", "event")),
]
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


def load_foundation_safe_leagues(readiness_path: Path) -> list[str]:
    readiness = pd.read_csv(readiness_path)
    safe = readiness[readiness["best_bucket"].isin(SAFE_BUCKETS)].copy()
    return sorted(safe["league"].dropna().astype(str).unique())


def load_league_training(hybrid_dir: Path, league: str) -> pd.DataFrame:
    tag = league_tag(league)
    files = sorted(hybrid_dir.glob(f"hybrid_match_training__{tag}__20*.csv"))
    if not files:
        combined = hybrid_dir / f"hybrid_match_training__{tag}.csv"
        files = [combined] if combined.exists() else []
    frames = []
    for path in files:
        frame = pd.read_csv(path, low_memory=False)
        frame["__source_file"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    if "match_date" in df.columns:
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
        df = df.sort_values(["match_date"]).reset_index(drop=True)
    return df


def stack_cols(df: pd.DataFrame, families: Sequence[str]) -> list[str]:
    cols = list(baseline_cols(df))
    for family in families:
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
        y = (
            df[list(targets)]
            .idxmax(axis=1)
            .map({"target_ftr_home": 0, "target_ftr_draw": 1, "target_ftr_away": 2})
        )
        return y.astype("Int64")
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


def score_stack(
    df: pd.DataFrame,
    league: str,
    market: str,
    stack_name: str,
    families: Sequence[str],
    *,
    holdout_frac: float,
    min_rows: int,
    random_seed: int,
) -> dict:
    cols = stack_cols(df, families)
    y = build_target(df, market)
    valid = y.notna()
    for target in TARGETS[market]:
        valid &= df[target].notna()
    if int(valid.sum()) < min_rows:
        return {
            "league": league,
            "market": market,
            "stack": stack_name,
            "families": "|".join(families),
            "status": "SKIP_LOW_ROWS",
            "rows_total": int(valid.sum()),
        }
    data = df.loc[valid].copy()
    y = y.loc[valid].astype(int)
    if y.nunique() < 2:
        return {
            "league": league,
            "market": market,
            "stack": stack_name,
            "families": "|".join(families),
            "status": "SKIP_SINGLE_CLASS",
            "rows_total": int(len(data)),
        }

    X = prepare_features(data, cols)
    split_idx = max(1, int(len(X) * (1.0 - holdout_frac)))
    split_idx = min(split_idx, len(X) - 1)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        return {
            "league": league,
            "market": market,
            "stack": stack_name,
            "families": "|".join(families),
            "status": "SKIP_SPLIT_SINGLE_CLASS",
            "rows_total": int(len(data)),
            "rows_train": int(len(X_train)),
            "rows_test": int(len(X_test)),
        }
    if not set(y_test.unique()).issubset(set(y_train.unique())):
        return {
            "league": league,
            "market": market,
            "stack": stack_name,
            "families": "|".join(families),
            "status": "SKIP_TEST_CLASS_NOT_IN_TRAIN",
            "rows_total": int(len(data)),
            "rows_train": int(len(X_train)),
            "rows_test": int(len(X_test)),
        }

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
        "league": league,
        "market": market,
        "stack": stack_name,
        "families": "|".join(families),
        "status": "OK",
        "rows_total": int(len(data)),
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


def build_delta_table(results: pd.DataFrame) -> pd.DataFrame:
    ok = results[results["status"].eq("OK")].copy()
    if ok.empty:
        return pd.DataFrame()
    baseline = ok[ok["stack"].eq("baseline")][["league", "market", "accuracy", "log_loss", "auc"]].rename(
        columns={"accuracy": "baseline_accuracy", "log_loss": "baseline_log_loss", "auc": "baseline_auc"}
    )
    delta = ok.merge(baseline, on=["league", "market"], how="left")
    delta["accuracy_delta_vs_baseline"] = delta["accuracy"] - delta["baseline_accuracy"]
    delta["log_loss_delta_vs_baseline"] = delta["log_loss"] - delta["baseline_log_loss"]
    delta["auc_delta_vs_baseline"] = delta["auc"] - delta["baseline_auc"]
    return delta.sort_values(["market", "accuracy_delta_vs_baseline"], ascending=[True, False])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness", default=str(DEFAULT_READINESS))
    parser.add_argument("--hybrid-dir", default=str(DEFAULT_HYBRID_DIR))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--holdout-frac", type=float, default=0.25)
    parser.add_argument("--min-rows", type=int, default=80)
    parser.add_argument("--limit-leagues", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    leagues = load_foundation_safe_leagues(Path(args.readiness))
    if args.limit_leagues:
        leagues = leagues[: args.limit_leagues]

    manifest_rows = []
    result_rows = []
    for league in leagues:
        df = load_league_training(Path(args.hybrid_dir), league)
        manifest_rows.append(
            {
                "league": league,
                "rows": len(df),
                "source_files": "|".join(sorted(df["__source_file"].unique())) if "__source_file" in df.columns and not df.empty else "",
                "status": "FOUND" if not df.empty else "MISSING_TRAINING",
            }
        )
        if df.empty:
            continue
        for market in TARGETS:
            for stack_name, families in STACKS:
                result_rows.append(
                    score_stack(
                        df,
                        league,
                        market,
                        stack_name,
                        families,
                        holdout_frac=args.holdout_frac,
                        min_rows=args.min_rows,
                        random_seed=args.random_seed,
                    )
                )

    manifest = pd.DataFrame(manifest_rows)
    results = pd.DataFrame(result_rows)
    deltas = build_delta_table(results)

    manifest.to_csv(outdir / "api_foundation_safe_ablation_manifest.csv", index=False)
    results.to_csv(outdir / "api_foundation_safe_ablation_results.csv", index=False)
    deltas.to_csv(outdir / "api_foundation_safe_ablation_deltas.csv", index=False)

    top_positive = deltas[deltas["stack"].ne("baseline")].sort_values(
        ["accuracy_delta_vs_baseline", "log_loss_delta_vs_baseline"], ascending=[False, True]
    ).head(30)
    weak = deltas[deltas["stack"].ne("baseline")].sort_values(
        ["accuracy_delta_vs_baseline", "log_loss_delta_vs_baseline"], ascending=[True, False]
    ).head(30)

    status_counts = results.groupby(["status"], dropna=False).size().reset_index(name="rows") if not results.empty else pd.DataFrame()
    summary = [
        "# API-Football Foundation-Safe Ablations",
        "",
        "Offline ablation run restricted to coverage-audited foundation-safe leagues.",
        "No API calls, production training, deploy changes, or broad league restores were performed.",
        "",
        "## League Manifest",
        markdown_table(manifest),
        "",
        "## Result Status Counts",
        markdown_table(status_counts),
        "",
        "## Strongest API Stack Lifts",
        markdown_table(
            top_positive[
                [
                    "league",
                    "market",
                    "stack",
                    "rows_test",
                    "n_features",
                    "accuracy",
                    "baseline_accuracy",
                    "accuracy_delta_vs_baseline",
                    "log_loss_delta_vs_baseline",
                    "auc_delta_vs_baseline",
                ]
            ]
        ),
        "",
        "## Weakest API Stack Deltas",
        markdown_table(
            weak[
                [
                    "league",
                    "market",
                    "stack",
                    "rows_test",
                    "n_features",
                    "accuracy",
                    "baseline_accuracy",
                    "accuracy_delta_vs_baseline",
                    "log_loss_delta_vs_baseline",
                    "auc_delta_vs_baseline",
                ]
            ]
        ),
        "",
        "## Operating Decision",
        (
            "Use this as an ablation screen only. Positive league/market/family cells still need "
            "walk-forward validation before they can influence shadow gates. Negative cells are "
            "evidence to keep that API family out of restoration logic for that league."
        ),
    ]
    (outdir / "api_football_foundation_safe_ablations.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"WROTE {outdir}")
    print(f"leagues={len(leagues)} result_rows={len(results)}")


if __name__ == "__main__":
    main()
