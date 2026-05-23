#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_VERSION = "audit_btts_proxy_feature_ablation_from_allmarkets_v1"

BTTS_BASELINE_COLS: Sequence[str] = (
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
    "model_p_for_bookie",
)

PROXY_GROUPS: Dict[str, Sequence[str]] = {
    "score_proxy": (
        "snapshot_ou25_support_score_proxy",
    ),
    "regime_proxy": (
        "snap_xg_total_pressure_proxy",
        "snap_style_chaos_index_proxy",
        "snap_ou25_over_regime_blend_proxy",
    ),
    "matchup_proxy": (
        "snap_home_attack_vs_away_def_xg_proxy",
        "snap_home_attack_vs_away_def_goals_proxy",
    ),
    "timing_proxy": (
        "snap_timing_both_teams_late_risk_proxy",
    ),
}

VARIANT_ORDER: Sequence[str] = (
    "baseline",
    "score_proxy",
    "regime_proxy",
    "matchup_proxy",
    "timing_proxy",
    "full_proxy_bundle",
)

MATCH_DATE_CANDIDATES: Sequence[str] = (
    "match_date",
    "date_GMT",
    "date",
    "Date",
    "timestamp",
)

TARGET_CANDIDATES: Sequence[str] = (
    "actual_btts_label",
    "btts_actual_label",
    "actual_label",
    "label",
    "target",
    "is_win",
    "correct",
)

YES_TOKENS = {"YES", "BTTS_YES", "BOTH_TEAMS_TO_SCORE_YES", "1", "TRUE"}
NO_TOKENS = {"NO", "BTTS_NO", "BOTH_TEAMS_TO_SCORE_NO", "0", "FALSE"}
FIXTURE_KEY_CANDIDATES: Sequence[str] = (
    "fixture_key",
    "fixture_key_ascii",
    "__fixture_key__",
    "match_id",
    "fixture_id",
)

HOME_GOAL_CANDIDATES: Sequence[str] = (
    "home_team_goal_count",
    "home_goals",
    "goals_home_ft",
    "home_score",
    "home_goal_count",
)

AWAY_GOAL_CANDIDATES: Sequence[str] = (
    "away_team_goal_count",
    "away_goals",
    "goals_away_ft",
    "away_score",
    "away_goal_count",
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

def _normalize_text(value: object) -> str:
    text = str(value or "").strip().lower()
    for ch in (" ", "-", "/", "\\", ".", ",", ":", ";", "'", '"', "(", ")", "[", "]", "{", "}", "&", "+"):
        text = text.replace(ch, "_")
    text = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in text)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")

def _coerce_date_text(series: pd.Series) -> pd.Series:
    dt = _coerce_datetime(series)
    out = dt.dt.strftime("%Y-%m-%d")
    return out.fillna(series.astype(str).str.strip().str[:10])

def _normalize_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).map(_normalize_text)

def _ensure_join_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    fixture_col = _find_first(out, FIXTURE_KEY_CANDIDATES)
    if fixture_col:
        out["__join_fixture_key__"] = out[fixture_col].fillna("").astype(str).str.strip()
        out["__join_fixture_key_norm__"] = _normalize_series(out[fixture_col])
    else:
        out["__join_fixture_key__"] = ""
        out["__join_fixture_key_norm__"] = ""

    date_col = _find_first(out, MATCH_DATE_CANDIDATES)
    home_col = _find_first(out, HOME_TEAM_CANDIDATES)
    away_col = _find_first(out, AWAY_TEAM_CANDIDATES)

    if date_col:
        out["__join_date_text__"] = _coerce_date_text(out[date_col]).fillna("").astype(str).str.strip()
    else:
        out["__join_date_text__"] = ""

    if home_col:
        out["__join_home_norm__"] = _normalize_series(out[home_col])
    else:
        out["__join_home_norm__"] = ""

    if away_col:
        out["__join_away_norm__"] = _normalize_series(out[away_col])
    else:
        out["__join_away_norm__"] = ""

    out["__join_composite_key__"] = (
        out["__join_date_text__"]
        + "__"
        + out["__join_home_norm__"]
        + "__"
        + out["__join_away_norm__"]
    )
    return out

def _league_key(name: object) -> str:
    txt = _normalize_text(name)
    if txt.endswith("_proxy_enriched"):
        txt = txt[: -len("_proxy_enriched")]
    if txt.endswith("_merged"):
        txt = txt[: -len("_merged")]
    return txt

def _build_merged_lookup(merged_dir: Path) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    for p in sorted(merged_dir.glob("*.csv")):
        if not p.is_file():
            continue
        k = _league_key(p.stem)
        if k and k not in lookup:
            lookup[k] = p.resolve()
    return lookup

def _resolve_btts_target_from_merged(pred_df: pd.DataFrame, merged_dir: Path) -> pd.Series:
    out_target = pd.Series(np.nan, index=pred_df.index, dtype="float64")
    merged_lookup = _build_merged_lookup(merged_dir)
    print(f"[TRUTH_RESOLVE] merged_lookup_keys={len(merged_lookup)} from {merged_dir}")

    if not merged_lookup:
        return out_target

    pred_prepped = _ensure_join_keys(pred_df)

    for league_name, pred_lg in pred_prepped.groupby("league", dropna=False):
        lg = str(league_name).strip()
        key = _league_key(lg)
        merged_path = merged_lookup.get(key)
        if merged_path is None:
            print(f"[TRUTH_RESOLVE] league={lg} | merged=<missing>")
            continue

        merged_raw = pd.read_csv(merged_path, low_memory=False)
        merged_prepped = _ensure_join_keys(merged_raw)

        hg_col = _find_first(merged_prepped, HOME_GOAL_CANDIDATES)
        ag_col = _find_first(merged_prepped, AWAY_GOAL_CANDIDATES)
        if not hg_col or not ag_col:
            print(f"[TRUTH_RESOLVE] league={lg} | merged={merged_path.name} | no goal columns found")
            continue

        truth = merged_prepped[[
            "__join_fixture_key__",
            "__join_fixture_key_norm__",
            "__join_composite_key__",
            hg_col,
            ag_col,
        ]].copy()
        truth["__home_goals__"] = pd.to_numeric(truth[hg_col], errors="coerce")
        truth["__away_goals__"] = pd.to_numeric(truth[ag_col], errors="coerce")
        truth["__target__"] = ((truth["__home_goals__"] > 0) & (truth["__away_goals__"] > 0)).astype("float64")
        truth.loc[truth["__home_goals__"].isna() | truth["__away_goals__"].isna(), "__target__"] = np.nan
        truth = truth.loc[truth["__target__"].isin([0.0, 1.0])].copy()

        if truth.empty:
            print(f"[TRUTH_RESOLVE] league={lg} | merged={merged_path.name} | no truth rows after goal resolution")
            continue

        raw_truth = truth.loc[
            truth["__join_fixture_key__"].astype(str).str.strip() != "",
            ["__join_fixture_key__", "__target__"]
        ].drop_duplicates("__join_fixture_key__", keep="first")

        norm_truth = truth.loc[
            truth["__join_fixture_key_norm__"].astype(str).str.strip() != "",
            ["__join_fixture_key_norm__", "__target__"]
        ].drop_duplicates("__join_fixture_key_norm__", keep="first")

        comp_truth = truth.loc[
            truth["__join_composite_key__"].astype(str).str.strip() != "",
            ["__join_composite_key__", "__target__"]
        ].drop_duplicates("__join_composite_key__", keep="first")

        pred_idx = pred_lg.index
        pred_sub = pred_lg.copy()
        matched = pd.Series(np.nan, index=pred_idx, dtype="float64")

        raw_matches = 0
        norm_matches = 0
        comp_matches = 0

        if not raw_truth.empty:
            raw_merge = pred_sub[["__join_fixture_key__"]].merge(raw_truth, on="__join_fixture_key__", how="left")
            matched.loc[pred_idx] = raw_merge["__target__"].values
            raw_matches = int(pd.notna(raw_merge["__target__"]).sum())

        need_norm = matched.isna()
        if bool(need_norm.any()) and not norm_truth.empty:
            norm_merge = pred_sub.loc[need_norm, ["__join_fixture_key_norm__"]].merge(
                norm_truth, on="__join_fixture_key_norm__", how="left"
            )
            matched.loc[need_norm] = norm_merge["__target__"].values
            norm_matches = int(pd.notna(norm_merge["__target__"]).sum())

        need_comp = matched.isna()
        if bool(need_comp.any()) and not comp_truth.empty:
            comp_merge = pred_sub.loc[need_comp, ["__join_composite_key__"]].merge(
                comp_truth, on="__join_composite_key__", how="left"
            )
            matched.loc[need_comp] = comp_merge["__target__"].values
            comp_matches = int(pd.notna(comp_merge["__target__"]).sum())

        out_target.loc[pred_idx] = matched.values
        print(
            f"[TRUTH_RESOLVE] league={lg} | merged={merged_path.name} | pred_rows={len(pred_sub)} | "
            f"merged_truth_rows={len(truth)} | raw_matches={raw_matches} | norm_matches={norm_matches} | "
            f"comp_matches={comp_matches} | matched_truth={int(matched.notna().sum())}"
        )

    return out_target

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit BTTS proxy feature ablations from a historical ALLMARKETS export."
    )
    parser.add_argument("--input-csv", required=True, help="Historical ALLMARKETS CSV with proxy columns attached")
    parser.add_argument(
        "--merged-dir",
        default="Matches/__merged__proxy_enriched",
        help="Directory containing league-level merged/proxy-enriched CSVs used to resolve historical BTTS truth",
    )
    parser.add_argument("--summary-csv", default=None, help="Optional summary CSV output path")
    parser.add_argument("--detail-csv", default=None, help="Optional detail CSV output path")
    parser.add_argument("--min-rows", type=int, default=240, help="Minimum BTTS rows per league")
    parser.add_argument("--min-train-rows", type=int, default=150, help="Minimum train rows")
    parser.add_argument("--min-test-rows", type=int, default=60, help="Minimum test rows")
    parser.add_argument(
        "--sort-col",
        default=None,
        help="Optional explicit sort column. Defaults to best available time column.",
    )
    return parser.parse_args()


def _find_first(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return ""


def _best_sort_col(df: pd.DataFrame, explicit: str | None) -> str:
    if explicit and explicit in df.columns:
        return explicit
    return _find_first(df, MATCH_DATE_CANDIDATES)


def _coerce_datetime(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    except TypeError:
        return pd.to_datetime(series, errors="coerce", utc=True)
    except Exception:
        return pd.to_datetime(series.astype(str), errors="coerce", utc=True)


def _normalize_binary_label(series: pd.Series) -> pd.Series:
    s = series.astype("string").fillna("").str.strip().str.upper()
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out.loc[s.isin(YES_TOKENS)] = 1.0
    out.loc[s.isin(NO_TOKENS)] = 0.0
    return out


def _resolve_target(df: pd.DataFrame) -> pd.Series:
    for c in TARGET_CANDIDATES:
        if c not in df.columns:
            continue
        s = df[c]
        if c in {"is_win", "correct"}:
            num = pd.to_numeric(s, errors="coerce")
            if num.notna().any():
                return num.clip(0, 1)
        norm = _normalize_binary_label(s)
        if norm.notna().any():
            return norm

    yes_cols = [c for c in ("home_team_goal_count", "home_goals", "goals_home_ft") if c in df.columns]
    no_cols = [c for c in ("away_team_goal_count", "away_goals", "goals_away_ft") if c in df.columns]
    if yes_cols and no_cols:
        hg = pd.to_numeric(df[yes_cols[0]], errors="coerce")
        ag = pd.to_numeric(df[no_cols[0]], errors="coerce")
        tgt = ((hg > 0) & (ag > 0)).astype("float64")
        tgt[(hg.isna()) | (ag.isna())] = np.nan
        if tgt.notna().any():
            return tgt

    truth_related = [c for c in df.columns if any(tok in c.lower() for tok in ("label", "target", "goal", "btts", "win", "correct"))]
    raise ValueError(
        "Could not resolve BTTS target. Available truth-related columns: " + ", ".join(truth_related[:20])
    )


def _prepare_btts(raw: pd.DataFrame, sort_col: str | None, merged_dir: Optional[Path] = None) -> pd.DataFrame:
    out = raw.copy()
    if "market" not in out.columns:
        raise ValueError("input csv missing market column")

    out = out.loc[out["market"].astype("string").fillna("").str.lower().eq("btts")].copy()
    if out.empty:
        raise ValueError("no BTTS rows found in input csv")

    try:
        out["__target__"] = _resolve_target(out)
    except Exception as exc:
        print(f"[PREPARE_BTTS] direct target resolution failed: {exc}")
        out["__target__"] = np.nan

    if out["__target__"].notna().sum() == 0 and merged_dir is not None:
        out["__target__"] = _resolve_btts_target_from_merged(out, merged_dir)

    out = out.loc[out["__target__"].isin([0.0, 1.0])].copy()
    if out.empty:
        raise ValueError("BTTS frame is empty after target resolution")

    best_sort = _best_sort_col(out, sort_col)
    if best_sort:
        if best_sort == "timestamp":
            ts = pd.to_numeric(out[best_sort], errors="coerce")
            unit = "ms" if ts.dropna().gt(1e11).any() else "s"
            out["__sort_ts__"] = pd.to_datetime(ts, errors="coerce", utc=True, unit=unit)
        else:
            out["__sort_ts__"] = _coerce_datetime(out[best_sort])
    else:
        out["__sort_ts__"] = pd.NaT

    out["__sort_ts__"] = out["__sort_ts__"].fillna(pd.Timestamp("1900-01-01", tz="UTC"))
    return out


def _variant_features(variant: str) -> List[str]:
    cols = list(BTTS_BASELINE_COLS)
    if variant == "baseline":
        return cols
    if variant == "full_proxy_bundle":
        for group_cols in PROXY_GROUPS.values():
            cols.extend(group_cols)
        return list(dict.fromkeys(cols))
    if variant not in PROXY_GROUPS:
        raise ValueError(f"unknown variant: {variant}")
    cols.extend(PROXY_GROUPS[variant])
    return list(dict.fromkeys(cols))


def _usable_feature_cols(df: pd.DataFrame, feature_cols: Sequence[str]) -> List[str]:
    usable: List[str] = []
    for c in feature_cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0:
            usable.append(c)
    return usable


def _build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0)),
        ]
    )


def _safe_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, proba))


def _safe_logloss(y_true: np.ndarray, proba: np.ndarray) -> float:
    p = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    return float(log_loss(y_true, p, labels=[0, 1]))


def _threshold_stats(y_true: np.ndarray, proba: np.ndarray, thr: float) -> tuple[float, float, int]:
    mask = np.asarray(proba) >= thr
    deployed = int(mask.sum())
    coverage = float(deployed / len(proba)) if len(proba) else np.nan
    if deployed == 0:
        return np.nan, coverage, 0
    hit_rate = float(np.mean(np.asarray(y_true)[mask] == 1))
    return hit_rate, coverage, deployed


def _evaluate_variant(train_df: pd.DataFrame, test_df: pd.DataFrame, variant: str) -> Dict[str, object]:
    feature_cols = _variant_features(variant)
    usable_cols = _usable_feature_cols(train_df, feature_cols)
    usable_cols = [c for c in usable_cols if c in test_df.columns and pd.to_numeric(test_df[c], errors="coerce").notna().sum() > 0]
    if not usable_cols:
        raise ValueError(f"no usable features for variant={variant}")

    X_train = train_df[usable_cols].apply(pd.to_numeric, errors="coerce")
    X_test = test_df[usable_cols].apply(pd.to_numeric, errors="coerce")
    y_train = train_df["__target__"].astype(int).to_numpy()
    y_test = test_df["__target__"].astype(int).to_numpy()

    model = _build_model()
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    result: Dict[str, object] = {
        f"{variant}_auc": _safe_auc(y_test, proba),
        f"{variant}_logloss": _safe_logloss(y_test, proba),
        f"{variant}_brier": float(brier_score_loss(y_test, proba)),
        f"{variant}_accuracy": float(accuracy_score(y_test, pred)),
        f"{variant}_feature_count": len(usable_cols),
        f"{variant}_cols_usable": len(usable_cols),
        f"usable_{variant}_cols": " | ".join(usable_cols),
    }

    for thr in (0.55, 0.60, 0.65, 0.70):
        hr, cov, dep = _threshold_stats(y_test, proba, thr)
        suffix = str(int(round(thr * 100)))
        result[f"{variant}_hit_rate_{suffix}"] = hr
        result[f"{variant}_coverage_{suffix}"] = cov
        result[f"{variant}_deployed_rows_{suffix}"] = dep

    return result


def main() -> None:
    args = _parse_args()
    input_csv = Path(args.input_csv).expanduser().resolve()
    merged_dir = Path(args.merged_dir).expanduser().resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"input csv not found: {input_csv}")

    raw = pd.read_csv(input_csv, low_memory=False)
    print(f"[INPUT] raw_rows={len(raw)} | cols={len(raw.columns)}")
    if "league" in raw.columns:
        preview = raw["league"].astype("string").fillna("").head(10).tolist()
        print(f"[INPUT] leagues_preview={preview}")

    btts = _prepare_btts(raw, sort_col=args.sort_col, merged_dir=merged_dir)
    print(f"[PREPARE_BTTS] market_rows={int((raw['market'].astype('string').fillna('').str.lower() == 'btts').sum()) if 'market' in raw.columns else 0}")
    print(f"[PREPARE_BTTS] matched_target_rows={len(btts)}")
    print(f"[POST_PREP] btts_rows={len(btts)} | leagues_present={btts['league'].nunique() if 'league' in btts.columns else 0}")

    if "league" not in btts.columns:
        raise ValueError("prepared BTTS frame missing league column")

    summary_rows: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []

    grouped = list(btts.groupby("league", dropna=False))
    print(f"[GROUPS] league_groups={len(grouped)}")

    for league, df_league in grouped:
        lg = str(league).strip()
        print(f"[LEAGUE] {lg} | rows_total={len(df_league)}")
        if len(df_league) < int(args.min_rows):
            continue

        df_league = df_league.sort_values(["__sort_ts__", "fixture_key"] if "fixture_key" in df_league.columns else ["__sort_ts__"]).reset_index(drop=True)
        split_idx = int(round(len(df_league) * 0.80))
        train_df = df_league.iloc[:split_idx].copy()
        test_df = df_league.iloc[split_idx:].copy()

        if len(train_df) < int(args.min_train_rows) or len(test_df) < int(args.min_test_rows):
            continue
        if train_df["__target__"].nunique() < 2 or test_df["__target__"].nunique() < 2:
            continue

        row: Dict[str, object] = {
            "league": lg,
            "rows_total": len(df_league),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
        }

        variant_scores: Dict[str, Dict[str, object]] = {}
        try:
            for variant in VARIANT_ORDER:
                metrics = _evaluate_variant(train_df, test_df, variant)
                variant_scores[variant] = metrics
                row.update(metrics)

            for variant in VARIANT_ORDER[1:]:
                row[f"delta_auc_{variant}_vs_baseline"] = float(row[f"{variant}_auc"] - row["baseline_auc"])
                row[f"delta_logloss_{variant}_vs_baseline"] = float(row[f"{variant}_logloss"] - row["baseline_logloss"])
                row[f"delta_brier_{variant}_vs_baseline"] = float(row[f"{variant}_brier"] - row["baseline_brier"])
                row[f"delta_accuracy_{variant}_vs_baseline"] = float(row[f"{variant}_accuracy"] - row["baseline_accuracy"])

            summary_rows.append(row)

            for variant in VARIANT_ORDER:
                d = {
                    "league": lg,
                    "variant": variant,
                    "rows_total": len(df_league),
                    "train_rows": len(train_df),
                    "test_rows": len(test_df),
                    "auc": row[f"{variant}_auc"],
                    "logloss": row[f"{variant}_logloss"],
                    "brier": row[f"{variant}_brier"],
                    "accuracy": row[f"{variant}_accuracy"],
                    "feature_count": row[f"{variant}_feature_count"],
                    "cols_usable": row[f"{variant}_cols_usable"],
                    "usable_cols": row[f"usable_{variant}_cols"],
                }
                if variant != "baseline":
                    d["delta_auc_vs_baseline"] = row[f"delta_auc_{variant}_vs_baseline"]
                    d["delta_logloss_vs_baseline"] = row[f"delta_logloss_{variant}_vs_baseline"]
                    d["delta_brier_vs_baseline"] = row[f"delta_brier_{variant}_vs_baseline"]
                    d["delta_accuracy_vs_baseline"] = row[f"delta_accuracy_{variant}_vs_baseline"]
                for thr in (55, 60, 65, 70):
                    d[f"hit_rate_{thr}"] = row[f"{variant}_hit_rate_{thr}"]
                    d[f"coverage_{thr}"] = row[f"{variant}_coverage_{thr}"]
                    d[f"deployed_rows_{thr}"] = row[f"{variant}_deployed_rows_{thr}"]
                detail_rows.append(d)

            print(
                f"OK: {lg} | rows={len(df_league)} | baseline_auc={row['baseline_auc']:.4f} | "
                f"proxy_auc={row['full_proxy_bundle_auc']:.4f} | "
                f"delta_auc={row['delta_auc_full_proxy_bundle_vs_baseline']:.4f}"
            )
        except Exception as exc:
            print(f"SKIP: {lg} | {exc}")

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)

    print(f"\nSCRIPT_VERSION: {SCRIPT_VERSION}")
    print(f"INPUT_CSV: {input_csv}")
    print(f"MERGED_DIR: {merged_dir}")
    print("\nSUMMARY")
    if summary_df.empty:
        print("No valid leagues evaluated.")
    else:
        sort_col = "delta_auc_full_proxy_bundle_vs_baseline" if "delta_auc_full_proxy_bundle_vs_baseline" in summary_df.columns else "league"
        summary_df = summary_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
        with pd.option_context("display.max_rows", 200, "display.max_columns", None, "display.width", 260):
            print(summary_df.to_string(index=False))

    if args.summary_csv:
        summary_path = Path(args.summary_csv).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        print(f"\nWROTE SUMMARY CSV: {summary_path}")

    if args.detail_csv:
        detail_path = Path(args.detail_csv).expanduser().resolve()
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        detail_df.to_csv(detail_path, index=False)
        print(f"WROTE DETAIL CSV: {detail_path}")


if __name__ == "__main__":
    main()