from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_INPUT = Path("data_sources/legacy_euro_sqlite/player_event_architecture/legacy_euro_player_event_architecture_dataset.csv")
DEFAULT_OUTDIR = Path("reports/latest/legacy_euro_player_event_feature_shape_audit")

TARGETS = ["shots_ge1", "sot_ge1", "fouls_committed_ge1", "fouls_drawn_ge1", "card_any"]

ROLE_CONTEXT_NUMERIC = [
    "slot",
    "x_coord",
    "y_coord",
    "is_home",
]
ROLE_CONTEXT_CATEGORICAL = [
    "league_name",
    "position_bucket",
    "side",
]
PLAYER_ATTRS = [
    "height",
    "weight",
    "overall_rating",
    "potential",
    "crossing",
    "finishing",
    "heading_accuracy",
    "short_passing",
    "volleys",
    "dribbling",
    "curve",
    "free_kick_accuracy",
    "long_passing",
    "ball_control",
    "acceleration",
    "sprint_speed",
    "agility",
    "reactions",
    "balance",
    "shot_power",
    "jumping",
    "stamina",
    "strength",
    "long_shots",
    "aggression",
    "interceptions",
    "positioning",
    "vision",
    "penalties",
    "marking",
    "standing_tackle",
    "sliding_tackle",
    "gk_diving",
    "gk_handling",
    "gk_kicking",
    "gk_positioning",
    "gk_reflexes",
]
PLAYER_CATEGORICAL = [
    "preferred_foot",
    "attacking_work_rate",
    "defensive_work_rate",
]
TEAM_ATTRS = [
    "buildUpPlaySpeed",
    "buildUpPlayDribbling",
    "buildUpPlayPassing",
    "chanceCreationPassing",
    "chanceCreationCrossing",
    "chanceCreationShooting",
    "defencePressure",
    "defenceAggression",
    "defenceTeamWidth",
    "opp_buildUpPlaySpeed",
    "opp_buildUpPlayDribbling",
    "opp_buildUpPlayPassing",
    "opp_chanceCreationPassing",
    "opp_chanceCreationCrossing",
    "opp_chanceCreationShooting",
    "opp_defencePressure",
    "opp_defenceAggression",
    "opp_defenceTeamWidth",
]
MARKET_PRIORS = [
    "book_count_1x2",
    "team_consensus_win_prob",
    "opponent_consensus_win_prob",
    "consensus_draw_prob",
]

FEATURE_GROUPS = {
    "role_context": {
        "numeric": ROLE_CONTEXT_NUMERIC,
        "categorical": ROLE_CONTEXT_CATEGORICAL,
    },
    "role_player": {
        "numeric": ROLE_CONTEXT_NUMERIC + PLAYER_ATTRS,
        "categorical": ROLE_CONTEXT_CATEGORICAL + PLAYER_CATEGORICAL,
    },
    "role_team_market": {
        "numeric": ROLE_CONTEXT_NUMERIC + TEAM_ATTRS + MARKET_PRIORS,
        "categorical": ROLE_CONTEXT_CATEGORICAL,
    },
    "full_stack": {
        "numeric": ROLE_CONTEXT_NUMERIC + PLAYER_ATTRS + TEAM_ATTRS + MARKET_PRIORS,
        "categorical": ROLE_CONTEXT_CATEGORICAL + PLAYER_CATEGORICAL,
    },
}


def available(columns: list[str], df: pd.DataFrame) -> list[str]:
    return [c for c in columns if c in df.columns]


def make_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50)),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("pre", pre),
            (
                "model",
                LogisticRegression(
                    max_iter=300,
                    solver="lbfgs",
                    penalty="l2",
                    C=0.8,
                    random_state=42,
                ),
            ),
        ]
    )


def safe_auc(y: pd.Series, p: np.ndarray) -> float:
    if y.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def top_lift(y: pd.Series, p: np.ndarray, frac: float) -> tuple[float, float]:
    if len(y) == 0:
        return float("nan"), float("nan")
    n = max(1, int(round(len(y) * frac)))
    order = np.argsort(-p)[:n]
    base = float(np.mean(y))
    top = float(np.mean(np.asarray(y)[order]))
    lift = top / base if base > 0 else float("nan")
    return top, lift


def evaluate(y: pd.Series, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    top10_rate, top10_lift = top_lift(y, p, 0.10)
    top20_rate, top20_lift = top_lift(y, p, 0.20)
    return {
        "base_rate": round(float(np.mean(y)), 6),
        "roc_auc": round(safe_auc(y, p), 6),
        "average_precision": round(float(average_precision_score(y, p)), 6),
        "log_loss": round(float(log_loss(y, p, labels=[0, 1])), 6),
        "brier": round(float(brier_score_loss(y, p)), 6),
        "top10_hit_rate": round(top10_rate, 6),
        "top10_lift": round(top10_lift, 6),
        "top20_hit_rate": round(top20_rate, 6),
        "top20_lift": round(top20_lift, 6),
    }


def model_feature_weights(pipe: Pipeline, numeric_cols: list[str], categorical_cols: list[str], top_n: int = 30) -> pd.DataFrame:
    pre: ColumnTransformer = pipe.named_steps["pre"]
    model: LogisticRegression = pipe.named_steps["model"]
    try:
        names = list(pre.get_feature_names_out())
    except Exception:
        names = [f"feature_{i}" for i in range(model.coef_.shape[1])]
    coefs = model.coef_[0]
    usable = min(len(names), len(coefs))
    out = pd.DataFrame({"feature": names[:usable], "coef": coefs[:usable]})
    out["abs_coef"] = out["coef"].abs()
    return out.sort_values("abs_coef", ascending=False).head(top_n).reset_index(drop=True)


def univariate_lifts(train: pd.DataFrame, test: pd.DataFrame, target: str, features: list[str]) -> pd.DataFrame:
    rows = []
    base = float(test[target].mean())
    for feature in features:
        if feature not in train.columns or feature not in test.columns:
            continue
        train_series = pd.to_numeric(train[feature], errors="coerce")
        test_series = pd.to_numeric(test[feature], errors="coerce")
        if train_series.notna().sum() < 100 or train_series.nunique(dropna=True) < 4:
            continue
        try:
            cutoff = float(train_series.quantile(0.80))
        except Exception:
            continue
        mask = test_series.ge(cutoff)
        if mask.sum() < 100:
            continue
        hit_rate = float(test.loc[mask, target].mean())
        rows.append(
            {
                "target": target,
                "feature": feature,
                "train_p80_cutoff": round(cutoff, 6),
                "test_rows_above_cutoff": int(mask.sum()),
                "base_rate": round(base, 6),
                "hit_rate_above_cutoff": round(hit_rate, 6),
                "lift": round(hit_rate / base, 6) if base > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("lift", ascending=False).reset_index(drop=True)


def categorical_lifts(test: pd.DataFrame, target: str, features: list[str]) -> pd.DataFrame:
    rows = []
    base = float(test[target].mean())
    for feature in features:
        if feature not in test.columns:
            continue
        for value, group in test.groupby(feature, dropna=False):
            if len(group) < 500:
                continue
            hit_rate = float(group[target].mean())
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "value": str(value),
                    "rows": int(len(group)),
                    "base_rate": round(base, 6),
                    "hit_rate": round(hit_rate, 6),
                    "lift": round(hit_rate / base, 6) if base > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values("lift", ascending=False).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight feature-shape audits over the legacy Euro player-event dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--train-end-season", default="2013/2014")
    parser.add_argument("--test-start-season", default="2014/2015")
    parser.add_argument("--targets", default=",".join(TARGETS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    targets = [x.strip() for x in args.targets.split(",") if x.strip()]
    df = pd.read_csv(args.input, parse_dates=["match_date"], low_memory=False)
    df = df.sort_values(["match_date", "match_id", "player_api_id"]).reset_index(drop=True)
    train = df[df["season"].astype(str).le(args.train_end_season)].copy()
    test = df[df["season"].astype(str).ge(args.test_start_season)].copy()
    if train.empty or test.empty:
        raise SystemExit("Empty train/test split. Check season arguments.")

    metrics_rows = []
    coef_frames = []
    lift_frames = []
    cat_lift_frames = []

    all_numeric_for_lifts = ROLE_CONTEXT_NUMERIC + PLAYER_ATTRS + TEAM_ATTRS + MARKET_PRIORS
    all_cats_for_lifts = ROLE_CONTEXT_CATEGORICAL + PLAYER_CATEGORICAL

    for target in targets:
        if target not in df.columns:
            raise SystemExit(f"Missing target: {target}")
        y_train = train[target].astype(int)
        y_test = test[target].astype(int)
        lift_frames.append(univariate_lifts(train, test, target, all_numeric_for_lifts).head(30))
        cat_lift_frames.append(categorical_lifts(test, target, all_cats_for_lifts).head(30))

        for group_name, spec in FEATURE_GROUPS.items():
            numeric_cols = available(spec["numeric"], train)
            categorical_cols = available(spec["categorical"], train)
            pipe = make_pipeline(numeric_cols, categorical_cols)
            pipe.fit(train[numeric_cols + categorical_cols], y_train)
            probs = pipe.predict_proba(test[numeric_cols + categorical_cols])[:, 1]
            row = {
                "target": target,
                "feature_group": group_name,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_positive_rate": round(float(y_train.mean()), 6),
                "test_positive_rate": round(float(y_test.mean()), 6),
                "numeric_features": len(numeric_cols),
                "categorical_features": len(categorical_cols),
            }
            row.update(evaluate(y_test, probs))
            metrics_rows.append(row)

            if group_name == "full_stack":
                weights = model_feature_weights(pipe, numeric_cols, categorical_cols, top_n=40)
                weights.insert(0, "target", target)
                coef_frames.append(weights)

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(args.outdir / "legacy_player_event_model_ablation_metrics.csv", index=False)
    if coef_frames:
        pd.concat(coef_frames, ignore_index=True).to_csv(args.outdir / "legacy_player_event_full_stack_top_coefficients.csv", index=False)
    if lift_frames:
        pd.concat(lift_frames, ignore_index=True).to_csv(args.outdir / "legacy_player_event_univariate_numeric_lifts.csv", index=False)
    if cat_lift_frames:
        pd.concat(cat_lift_frames, ignore_index=True).to_csv(args.outdir / "legacy_player_event_categorical_lifts.csv", index=False)

    best = metrics.sort_values(["target", "roc_auc"], ascending=[True, False]).groupby("target", as_index=False).head(1)
    report = [
        "# Legacy Euro Player-Event Feature-Shape Audit",
        "",
        "Research-only audit over the legacy European SQLite player-event sidecar.",
        "",
        f"- Input: `{args.input}`",
        f"- Train seasons: <= `{args.train_end_season}`",
        f"- Test seasons: >= `{args.test_start_season}`",
        f"- Train rows: {len(train):,}",
        f"- Test rows: {len(test):,}",
        "",
        "## Best Feature Groups By Target",
        "",
        best[["target", "feature_group", "test_positive_rate", "roc_auc", "average_precision", "top10_hit_rate", "top10_lift", "brier"]].to_csv(index=False),
        "",
        "## Interpretation",
        "",
        "- `role_context` tests whether lineup role, side, and league alone carry signal.",
        "- `role_player` tests whether player attributes add signal.",
        "- `role_team_market` tests whether team tactical attributes and bookmaker consensus add signal.",
        "- `full_stack` tests whether the combined architecture survives chronologically.",
        "",
        "Use this only to validate feature shapes. Recalibrate all weights on modern API-Football before live use.",
    ]
    (args.outdir / "LEGACY_PLAYER_EVENT_FEATURE_SHAPE_AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"[ok] targets={len(targets)} metrics={len(metrics)}")
    print(f"[ok] wrote {args.outdir}")


if __name__ == "__main__":
    main()
