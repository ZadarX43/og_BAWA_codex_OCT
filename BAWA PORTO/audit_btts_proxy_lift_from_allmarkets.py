#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_VERSION = "audit_btts_proxy_lift_from_allmarkets_v1"

BASELINE_FEATURES: Sequence[str] = (
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
    "clean_sheet_rate_5_home",
    "clean_sheet_rate_5_away",
    "scored_rate_5_home",
    "scored_rate_5_away",
    "conceded_rate_5_home",
    "conceded_rate_5_away",
    "goaliness_avg_5_home",
    "goaliness_avg_5_away",
    "xg_for_avg_5_home",
    "xg_for_avg_5_away",
    "xg_against_avg_5_home",
    "xg_against_avg_5_away",
    "pre_match_xg_home",
    "pre_match_xg_away",
    "exp_goals_sum",
    "bookie_lambda_total_fit",
    "over_25_percentage_pre_match",
    "p_over25_novig",
    "p_under25_novig",
    "model_p_for_bookie",
)

PROXY_FEATURES: Sequence[str] = (
    "snap_xg_total_pressure_proxy",
    "snap_style_chaos_index_proxy",
    "snap_ou25_over_regime_blend_proxy",
    "snapshot_ou25_support_score_proxy",
    "snap_timing_both_teams_late_risk_proxy",
    "snap_home_attack_vs_away_def_xg_proxy",
    "snap_home_attack_vs_away_def_goals_proxy",
)

FIXTURE_KEY_CANDIDATES: Sequence[str] = (
    "fixture_key",
    "fixture_key_ascii",
    "__fixture_key__",
    "match_id",
    "fixture_id",
)
MATCH_DATE_CANDIDATES: Sequence[str] = (
    "match_date",
    "date_GMT",
    "date",
    "timestamp",
)
HOME_TEAM_CANDIDATES: Sequence[str] = (
    "home_team_name",
    "home_team",
    "Home Team",
    "Home",
)
AWAY_TEAM_CANDIDATES: Sequence[str] = (
    "away_team_name",
    "away_team",
    "Away Team",
    "Away",
)
LEAGUE_CANDIDATES: Sequence[str] = (
    "league",
    "competition",
    "division",
)
DEPLOY_THRESHOLDS: Sequence[float] = (0.55, 0.60, 0.65, 0.70)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit BTTS proxy lift from a historical ALLMARKETS export."
    )
    parser.add_argument("--input-csv", required=True, help="Historical ALLMARKETS CSV with proxy columns attached")
    parser.add_argument(
        "--merged-dir",
        default="Matches/__merged__proxy_enriched",
        help="Directory containing league-level merged/proxy-enriched CSVs used to resolve historical BTTS truth",
    )
    parser.add_argument("--summary-csv", default=None, help="Optional summary CSV output path")
    parser.add_argument("--detail-csv", default=None, help="Optional detail CSV output path")
    parser.add_argument("--min-rows", type=int, default=120, help="Minimum BTTS rows per league")
    parser.add_argument("--min-train-rows", type=int, default=80, help="Minimum train rows")
    parser.add_argument("--min-test-rows", type=int, default=30, help="Minimum test rows")
    parser.add_argument(
        "--sort-col",
        default=None,
        help="Optional explicit sort column. Defaults to best available time column.",
    )
    return parser.parse_args()


def _find_first(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _norm_text(value: object) -> str:
    s = str(value or "").strip().lower()
    out = []
    prev_us = False
    for ch in s:
        if ch.isalnum():
            out.append(ch)
            prev_us = False
        else:
            if not prev_us:
                out.append("_")
                prev_us = True
    txt = "".join(out).strip("_")
    while "__" in txt:
        txt = txt.replace("__", "_")
    return txt


def _coerce_date_text(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=False, format="mixed")
    out = dt.dt.strftime("%Y-%m-%d")
    return out.fillna(series.astype(str).str.strip().str[:10])


def _ensure_join_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    fixture_col = _find_first(out, FIXTURE_KEY_CANDIDATES)
    if fixture_col is not None:
        out["__join_fixture_key__"] = out[fixture_col].fillna("").astype(str).str.strip()
        out["__join_fixture_key_norm__"] = out[fixture_col].fillna("").astype(str).map(_norm_text)
    else:
        out["__join_fixture_key__"] = ""
        out["__join_fixture_key_norm__"] = ""

    date_col = _find_first(out, MATCH_DATE_CANDIDATES)
    home_col = _find_first(out, HOME_TEAM_CANDIDATES)
    away_col = _find_first(out, AWAY_TEAM_CANDIDATES)

    if date_col is not None:
        date_text = _coerce_date_text(out[date_col])
    else:
        date_text = pd.Series("", index=out.index, dtype="object")

    if home_col is not None:
        home_text = out[home_col].fillna("").astype(str).map(_norm_text)
    else:
        home_text = pd.Series("", index=out.index, dtype="object")

    if away_col is not None:
        away_text = out[away_col].fillna("").astype(str).map(_norm_text)
    else:
        away_text = pd.Series("", index=out.index, dtype="object")

    out["__join_date_text__"] = date_text.fillna("").astype(str).str.strip()
    out["__join_home_norm__"] = home_text.fillna("")
    out["__join_away_norm__"] = away_text.fillna("")
    out["__join_composite_key__"] = (
        out["__join_date_text__"] + "__" + out["__join_home_norm__"] + "__" + out["__join_away_norm__"]
    )
    return out


def _merged_lookup_key_from_path(path: Path) -> str:
    stem = path.stem
    stem = stem.replace("__proxy_enriched", "")
    stem = stem.replace("__merged", "")
    return _norm_text(stem)


def _resolve_merged_lookup(merged_dir: Path) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    for p in sorted(merged_dir.glob("*.csv")):
        if not p.is_file():
            continue
        key = _merged_lookup_key_from_path(p)
        if key:
            lookup[key] = p
    return lookup


def _resolve_truth_from_selection(df: pd.DataFrame) -> pd.Series:
    for col in ("actual_btts_label", "btts_actual_label", "result_btts", "selection"):
        if col not in df.columns:
            continue
        s = df[col].fillna("").astype(str).str.strip().str.upper()
        yes = s.isin({"BTTS_YES", "YES", "Y", "BTTS YES", "BOTH_TEAMS_TO_SCORE_YES"})
        no = s.isin({"BTTS_NO", "NO", "N", "BTTS NO", "BOTH_TEAMS_TO_SCORE_NO"})
        out = pd.Series(np.nan, index=df.index, dtype="float64")
        out.loc[yes] = 1.0
        out.loc[no] = 0.0
        if out.notna().any():
            return out
    raise ValueError(
        "Could not resolve BTTS target. Available truth-related columns: "
        + ", ".join([c for c in ("selection", "bookie_pick", "actual_btts_label", "btts_actual_label", "result_btts") if c in df.columns])
    )


def _build_btts_truth_from_merged(merged_df: pd.DataFrame) -> pd.Series:
    home_goals = None
    away_goals = None
    for c in ("home_team_goal_count", "home_goals", "goals_home", "home_score", "team_a_goals"):
        if c in merged_df.columns:
            s = pd.to_numeric(merged_df[c], errors="coerce")
            if s.notna().any():
                home_goals = s
                break
    for c in ("away_team_goal_count", "away_goals", "goals_away", "away_score", "team_b_goals"):
        if c in merged_df.columns:
            s = pd.to_numeric(merged_df[c], errors="coerce")
            if s.notna().any():
                away_goals = s
                break
    if home_goals is None or away_goals is None:
        return pd.Series(np.nan, index=merged_df.index, dtype="float64")
    out = pd.Series(np.nan, index=merged_df.index, dtype="float64")
    mask = home_goals.notna() & away_goals.notna()
    out.loc[mask] = ((home_goals.loc[mask] >= 1) & (away_goals.loc[mask] >= 1)).astype(float)
    return out


def _resolve_target_from_merged(ou: pd.DataFrame, merged_dir: Path) -> pd.Series:
    merged_lookup = _resolve_merged_lookup(merged_dir)
    print(f"[TRUTH_RESOLVE] merged_lookup_keys={len(merged_lookup)} from {merged_dir}")
    out_target = pd.Series(np.nan, index=ou.index, dtype="float64")

    league_col = _find_first(ou, LEAGUE_CANDIDATES)
    if league_col is None:
        return out_target

    for league_name, sub_idx in ou.groupby(league_col).groups.items():
        league_key = _norm_text(str(league_name).replace("__merged", ""))
        merged_path = merged_lookup.get(league_key)
        if merged_path is None:
            print(f"[TRUTH_RESOLVE] league={league_name} | merged=<missing>")
            continue

        merged_raw = pd.read_csv(merged_path, low_memory=False)
        merged = _ensure_join_keys(merged_raw)
        merged_target = _build_btts_truth_from_merged(merged)
        merged["__target__"] = merged_target
        merged = merged.loc[merged["__target__"].notna()].copy()
        if merged.empty:
            print(f"[TRUTH_RESOLVE] league={league_name} | merged={merged_path.name} | no truth rows after goal resolution")
            continue

        pred = ou.loc[sub_idx].copy()
        raw_matches = 0
        norm_matches = 0
        comp_matches = 0

        raw_map = merged.loc[merged["__join_fixture_key__"].ne("")].drop_duplicates("__join_fixture_key__")
        pred = pred.merge(
            raw_map[["__join_fixture_key__", "__target__"]],
            on="__join_fixture_key__",
            how="left",
        )
        raw_matches = int(pred["__target__"].notna().sum())

        need = pred["__target__"].isna()
        if need.any():
            norm_map = merged.loc[merged["__join_fixture_key_norm__"].ne("")].drop_duplicates("__join_fixture_key_norm__")
            fb = pred.loc[need, ["__join_fixture_key_norm__"]].merge(
                norm_map[["__join_fixture_key_norm__", "__target__"]],
                on="__join_fixture_key_norm__",
                how="left",
            )
            pred.loc[need, "__target__"] = fb["__target__"].values
            norm_matches = int(pred.loc[need, "__target__"].notna().sum())

        need = pred["__target__"].isna()
        if need.any():
            comp_map = merged.loc[merged["__join_composite_key__"].ne("")].drop_duplicates("__join_composite_key__")
            fb = pred.loc[need, ["__join_composite_key__"]].merge(
                comp_map[["__join_composite_key__", "__target__"]],
                on="__join_composite_key__",
                how="left",
            )
            pred.loc[need, "__target__"] = fb["__target__"].values
            comp_matches = int(pred.loc[need, "__target__"].notna().sum())

        matched = int(pred["__target__"].notna().sum())
        out_target.loc[pred.index] = pred["__target__"]
        print(
            f"[TRUTH_RESOLVE] league={league_name} | merged={merged_path.name} | pred_rows={len(pred)} | "
            f"merged_truth_rows={len(merged)} | raw_matches={raw_matches} | norm_matches={norm_matches} | "
            f"comp_matches={comp_matches} | matched_truth={matched}"
        )

    return out_target


def _prepare_btts(raw: pd.DataFrame, merged_dir: Path, sort_col: Optional[str] = None) -> pd.DataFrame:
    out = raw.copy()
    market_col = _find_first(out, ("market",))
    if market_col is None:
        raise ValueError("market column missing")
    out = out.loc[out[market_col].fillna("").astype(str).str.lower().eq("btts")].copy()

    print(f"[PREPARE_BTTS] market_rows={len(out)}")

    out = _ensure_join_keys(out)
    try:
        out["__target__"] = _resolve_truth_from_selection(out)
    except Exception as exc:
        print(f"[PREPARE_BTTS] direct target resolution failed: {exc}")
        out["__target__"] = np.nan

    if not out["__target__"].notna().any():
        print("[PREPARE_BTTS] direct target rows were all null; retrying via merged truth")
        out["__target__"] = _resolve_target_from_merged(out, merged_dir)

    matched_target_rows = int(out["__target__"].notna().sum())
    print(f"[PREPARE_BTTS] matched_target_rows={matched_target_rows}")

    out = out.loc[out["__target__"].notna()].copy()
    if out.empty:
        raise ValueError("BTTS frame is empty after target resolution")

    if sort_col and sort_col in out.columns:
        out = out.sort_values(sort_col, kind="mergesort").reset_index(drop=True)
    else:
        dt_col = _find_first(out, ("match_date", "date_GMT", "date", "timestamp"))
        if dt_col is not None:
            out["__sort_dt__"] = pd.to_datetime(out[dt_col], errors="coerce", utc=True, format="mixed")
            out = out.sort_values(["__sort_dt__", "__join_fixture_key__"], kind="mergesort").reset_index(drop=True)
        else:
            out = out.reset_index(drop=True)

    print(f"[POST_PREP] btts_rows={len(out)} | leagues_present={out[_find_first(out, LEAGUE_CANDIDATES)].nunique()}")
    return out


def _usable_feature_cols(df: pd.DataFrame, features: Sequence[str]) -> List[str]:
    usable: List[str] = []
    for col in features:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() > 0:
            usable.append(col)
    return usable


def _fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    usable_cols = _usable_feature_cols(train_df, feature_cols)
    usable_cols = [c for c in usable_cols if c in test_df.columns]
    if not usable_cols:
        raise ValueError("No usable feature columns")

    X_train = train_df[usable_cols].apply(pd.to_numeric, errors="coerce")
    X_test = test_df[usable_cols].apply(pd.to_numeric, errors="coerce")
    y_train = train_df["__target__"].astype(int)

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )
    model.fit(X_train, y_train)
    pred = model.predict_proba(X_test)[:, 1]
    return pred, usable_cols


def _safe_auc(y_true: pd.Series, y_prob: np.ndarray) -> float:
    if pd.Series(y_true).nunique(dropna=True) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _safe_logloss(y_true: pd.Series, y_prob: np.ndarray) -> float:
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    return float(log_loss(y_true, y_prob, labels=[0, 1]))


def _safe_brier(y_true: pd.Series, y_prob: np.ndarray) -> float:
    return float(brier_score_loss(y_true, y_prob))


def _safe_accuracy(y_true: pd.Series, y_prob: np.ndarray) -> float:
    return float(accuracy_score(y_true, (np.asarray(y_prob) >= 0.5).astype(int)))


def _deploy_stats(y_true: pd.Series, y_prob: np.ndarray, thresholds: Sequence[float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    y = pd.Series(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    total = len(y)
    for thr in thresholds:
        m = p >= float(thr)
        deployed = int(m.sum())
        cov = (deployed / total) if total else 0.0
        hit = float(y.loc[m].mean()) if deployed else float("nan")
        suffix = str(int(round(thr * 100)))
        out[f"hit_rate_{suffix}"] = hit
        out[f"coverage_{suffix}"] = cov
        out[f"deployed_rows_{suffix}"] = deployed
    return out


def _evaluate_stack(
    league_df: pd.DataFrame,
    feature_cols: Sequence[str],
    min_train_rows: int,
    min_test_rows: int,
) -> Dict[str, object]:
    n = len(league_df)
    split_idx = int(np.floor(n * 0.8))
    split_idx = max(min_train_rows, split_idx)
    split_idx = min(split_idx, n - min_test_rows)
    if split_idx <= 0 or split_idx >= n:
        raise ValueError("Unable to create valid train/test split")

    train_df = league_df.iloc[:split_idx].copy()
    test_df = league_df.iloc[split_idx:].copy()
    if len(train_df) < min_train_rows or len(test_df) < min_test_rows:
        raise ValueError("Train/test rows below threshold")

    y_test = test_df["__target__"].astype(int)
    pred, usable_cols = _fit_predict(train_df, test_df, feature_cols)

    result: Dict[str, object] = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "auc": _safe_auc(y_test, pred),
        "logloss": _safe_logloss(y_test, pred),
        "brier": _safe_brier(y_test, pred),
        "accuracy": _safe_accuracy(y_test, pred),
        "feature_count": len(usable_cols),
        "cols_usable": len(usable_cols),
        "usable_cols": " | ".join(usable_cols),
    }
    result.update(_deploy_stats(y_test, pred, DEPLOY_THRESHOLDS))
    return result


def main() -> None:
    args = _parse_args()
    input_csv = Path(args.input_csv).expanduser().resolve()
    merged_dir = Path(args.merged_dir).expanduser().resolve()

    raw = pd.read_csv(input_csv, low_memory=False)
    print(f"[INPUT] raw_rows={len(raw)} | cols={len(raw.columns)}")
    lg_col = _find_first(raw, LEAGUE_CANDIDATES)
    if lg_col is not None:
        print(f"[INPUT] leagues_preview={raw[lg_col].astype(str).head(10).tolist()}")

    try:
        bt = _prepare_btts(raw, merged_dir=merged_dir, sort_col=args.sort_col)
    except Exception as exc:
        truth_related = [c for c in ("selection", "bookie_pick", "actual_btts_label", "btts_actual_label", "result_btts") if c in raw.columns]
        raise ValueError(
            f"Failed to prepare BTTS rows from {input_csv}. Reason: {exc}. "
            f"Truth-related columns present: {truth_related}. merged_dir={merged_dir}"
        ) from exc

    league_col = _find_first(bt, LEAGUE_CANDIDATES)
    if league_col is None:
        raise ValueError("No league column found after BTTS prepare")

    print(f"[GROUPS] league_groups={bt[league_col].nunique()}")

    summary_rows: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []

    for league_name, league_df in bt.groupby(league_col, sort=True):
        league_df = league_df.reset_index(drop=True)
        print(f"[LEAGUE] {league_name} | rows_total={len(league_df)}")
        if len(league_df) < args.min_rows:
            continue
        if league_df["__target__"].nunique(dropna=True) < 2:
            continue

        try:
            baseline = _evaluate_stack(
                league_df=league_df,
                feature_cols=BASELINE_FEATURES,
                min_train_rows=args.min_train_rows,
                min_test_rows=args.min_test_rows,
            )
            full_proxy = _evaluate_stack(
                league_df=league_df,
                feature_cols=list(BASELINE_FEATURES) + list(PROXY_FEATURES),
                min_train_rows=args.min_train_rows,
                min_test_rows=args.min_test_rows,
            )
        except Exception:
            continue

        delta_auc = float(full_proxy["auc"] - baseline["auc"])
        delta_logloss = float(full_proxy["logloss"] - baseline["logloss"])
        delta_brier = float(full_proxy["brier"] - baseline["brier"])
        delta_accuracy = float(full_proxy["accuracy"] - baseline["accuracy"])

        summary_row: Dict[str, object] = {
            "league": str(league_name),
            "rows_total": int(len(league_df)),
            "train_rows": baseline["train_rows"],
            "test_rows": baseline["test_rows"],
            "baseline_auc": baseline["auc"],
            "baseline_logloss": baseline["logloss"],
            "baseline_brier": baseline["brier"],
            "baseline_accuracy": baseline["accuracy"],
            "baseline_feature_count": baseline["feature_count"],
            "baseline_cols_usable": baseline["cols_usable"],
            "usable_baseline_cols": baseline["usable_cols"],
            "proxy_auc": full_proxy["auc"],
            "proxy_logloss": full_proxy["logloss"],
            "proxy_brier": full_proxy["brier"],
            "proxy_accuracy": full_proxy["accuracy"],
            "proxy_feature_count": full_proxy["feature_count"],
            "proxy_cols_usable": full_proxy["cols_usable"],
            "usable_proxy_cols": full_proxy["usable_cols"],
            "delta_auc_proxy_vs_baseline": delta_auc,
            "delta_logloss_proxy_vs_baseline": delta_logloss,
            "delta_brier_proxy_vs_baseline": delta_brier,
            "delta_accuracy_proxy_vs_baseline": delta_accuracy,
        }

        for thr in DEPLOY_THRESHOLDS:
            suffix = str(int(round(thr * 100)))
            summary_row[f"baseline_hit_rate_{suffix}"] = baseline[f"hit_rate_{suffix}"]
            summary_row[f"baseline_coverage_{suffix}"] = baseline[f"coverage_{suffix}"]
            summary_row[f"baseline_deployed_rows_{suffix}"] = baseline[f"deployed_rows_{suffix}"]
            summary_row[f"proxy_hit_rate_{suffix}"] = full_proxy[f"hit_rate_{suffix}"]
            summary_row[f"proxy_coverage_{suffix}"] = full_proxy[f"coverage_{suffix}"]
            summary_row[f"proxy_deployed_rows_{suffix}"] = full_proxy[f"deployed_rows_{suffix}"]

        summary_rows.append(summary_row)
        detail_rows.append(summary_row.copy())

        print(
            f"OK: {league_name} | rows={len(league_df)} | baseline_auc={baseline['auc']:.4f} | "
            f"proxy_auc={full_proxy['auc']:.4f} | delta_auc={delta_auc:.4f}"
        )

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)

    print(f"\nSCRIPT_VERSION: {SCRIPT_VERSION}")
    print(f"INPUT_CSV: {input_csv}")
    print(f"MERGED_DIR: {merged_dir}")
    print("\nSUMMARY")
    if summary_df.empty:
        print("No valid leagues evaluated.")
    else:
        with pd.option_context("display.max_rows", 200, "display.max_columns", None, "display.width", 260):
            print(summary_df.sort_values(["delta_auc_proxy_vs_baseline", "proxy_auc"], ascending=[False, False]).to_string(index=False))

    if args.summary_csv:
        summary_csv = Path(args.summary_csv).expanduser().resolve()
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_csv, index=False)
        print(f"\nWROTE SUMMARY CSV: {summary_csv}")

    if args.detail_csv:
        detail_csv = Path(args.detail_csv).expanduser().resolve()
        detail_csv.parent.mkdir(parents=True, exist_ok=True)
        detail_df.to_csv(detail_csv, index=False)
        print(f"WROTE DETAIL CSV: {detail_csv}")


if __name__ == "__main__":
    main()