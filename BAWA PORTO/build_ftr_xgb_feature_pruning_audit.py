#!/usr/bin/env python3
"""Feature-pruning audit for XGB FTR models.

Trains XGB with top-k features based on current XGB bundle importances.
Outputs:
  - FTR_XGB_FEATURE_PRUNE__ALL.csv
  - FTR_XGB_FEATURE_PRUNE__BEST.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None  # type: ignore

import train_investor_leagues_v2 as tiv2


DEFAULT_LEAGUES = [
    "Australia A-League","Austria Bundesliga","Belgium Pro","Brazil Serie A","Champions League",
    "Czech First League","Denmark Superliga","England Championship","England FA Cup","England Premier League",
    "Europa Conference","Europa League","France Ligue 1","Germany Bundesliga","Germany Bundesliga 2",
    "Italy Serie A","Japan J1","Netherlands Eredivisie","Norway Eliteserien","Portugal Liga",
    "Saudi Pro League","Scotland Premiership","South Korea K League","Spain La Liga",
    "Swiss Super League","Turkey Super Lig","USA MLS",
]

TOP_K = [20, 40, 60, 80, 120]


def _load_merged(league: str, merged_root: Path, matches_root: Path) -> pd.DataFrame:
    tag = tiv2._league_tag(league)
    mp = merged_root / f"{tag}__merged.csv"
    if mp.exists():
        df = pd.read_csv(mp, low_memory=False)
        df["__src_csv"] = mp.name
        df = tiv2._apply_renames(df)
        df = tiv2._coerce_numbers(df)
        df = tiv2._ensure_match_date(df)
        return df
    return tiv2._load_all_matches_csvs(matches_root / league)


def _prep_frames(df: pd.DataFrame, val_frac: float = 0.2):
    df = tiv2._completed_only(df)
    y_btts, y_over, y_under, y_ftr = tiv2._build_targets(df)
    X_all, feats, cat_idx = tiv2._make_feature_frame(df)
    tr_idx, va_idx, _info = tiv2._compute_holdout_split(df, val_frac=val_frac)
    X_tr = X_all.loc[tr_idx.intersection(X_all.index)]
    X_va = X_all.loc[va_idx.intersection(X_all.index)]
    y_tr = y_ftr.loc[X_tr.index]
    y_va = y_ftr.loc[X_va.index]
    return X_tr, X_va, y_tr, y_va, feats, cat_idx


def _prep_xgb_arrays(X_tr, X_va, feats, cat_idx):
    X_tr_xgb, feats_xgb = tiv2._prep_xgb_frame(X_tr, feats, cat_idx, enable_categorical=False)
    X_va_xgb, feats_xgb = tiv2._prep_xgb_frame(X_va, feats_xgb, cat_idx=[], enable_categorical=False)
    X_tr_xgb = X_tr_xgb.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    X_va_xgb = X_va_xgb.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    return X_tr_xgb, X_va_xgb, feats_xgb


def _eval_model(model, X_va, y_va) -> float:
    proba = np.asarray(model.predict_proba(X_va))
    yhat = proba.argmax(axis=1)
    return float((yhat == y_va.to_numpy(dtype=int)).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description="XGB feature pruning audit (top-k features)")
    ap.add_argument("--leagues", default=",".join(DEFAULT_LEAGUES))
    ap.add_argument("--merged-root", default="Matches/__merged__")
    ap.add_argument("--matches-root", default="Matches")
    ap.add_argument("--modelstore", default="ModelStore")
    ap.add_argument("--outdir", default="predictions_output/walk_forward/_MASTER/XGB_TUNING")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    if XGBClassifier is None:
        raise SystemExit("xgboost not available in this environment")

    leagues = [s.strip() for s in str(args.leagues).split(",") if s.strip()]
    merged_root = Path(args.merged_root)
    matches_root = Path(args.matches_root)
    modelstore = Path(args.modelstore)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for lg in leagues:
        tag = tiv2._modelstore_tag(lg)
        p_xgb = modelstore / tag / "xgb" / "ftr_v2.pkl"
        if not p_xgb.exists():
            continue
        try:
            import joblib
            bundle = joblib.load(p_xgb)
        except Exception:
            continue

        feats = list(bundle.get("features", []))
        importances = np.asarray(getattr(bundle.get("model"), "feature_importances_", []))
        if len(importances) != len(feats):
            # fallback: skip if importance length mismatch
            continue
        order = np.argsort(importances)[::-1]
        feat_rank = [feats[i] for i in order]

        df = _load_merged(lg, merged_root, matches_root)
        if df is None or df.empty:
            continue
        X_tr, X_va, y_tr, y_va, all_feats, cat_idx = _prep_frames(df, val_frac=float(args.val_frac))

        # baseline accuracy
        base_acc = float(bundle.get("val_accuracy", np.nan))

        # only keep features that exist in the current frame
        feat_rank = [f for f in feat_rank if f in X_tr.columns]

        for k in TOP_K:
            topk = feat_rank[: min(k, len(feat_rank))]
            # filter frames to top-k (safe)
            X_tr_k = X_tr.loc[:, topk]
            X_va_k = X_va.loc[:, topk]
            # no categorical handling (already numeric-ified)
            X_tr_xgb = X_tr_k.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            X_va_xgb = X_va_k.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

            mdl = XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                n_estimators=600,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.80,
                colsample_bytree=0.80,
                min_child_weight=2.0,
                reg_lambda=1.0,
                n_jobs=int(args.threads),
                random_state=42,
            )
            mdl.fit(X_tr_xgb, y_tr.to_numpy(dtype=int))
            acc = _eval_model(mdl, X_va_xgb, y_va)

            rows.append({
                "league": lg,
                "k": int(k),
                "val_accuracy": acc,
                "base_xgb_val_accuracy": base_acc,
                "delta_vs_base_xgb": acc - base_acc if np.isfinite(base_acc) else np.nan,
            })

    all_df = pd.DataFrame(rows)
    all_df.to_csv(outdir / "FTR_XGB_FEATURE_PRUNE__ALL.csv", index=False)
    if not all_df.empty:
        best_df = (
            all_df.sort_values(["league", "val_accuracy"], ascending=[True, False])
            .groupby("league", as_index=False)
            .head(1)
        )
        best_df.to_csv(outdir / "FTR_XGB_FEATURE_PRUNE__BEST.csv", index=False)

    print(f"Wrote: {outdir / 'FTR_XGB_FEATURE_PRUNE__ALL.csv'}")


if __name__ == "__main__":
    main()
