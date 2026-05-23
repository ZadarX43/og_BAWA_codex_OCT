# audit_ou25_proxy_lift_from_allmarkets.py
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_VERSION = "audit_ou25_proxy_lift_from_allmarkets_v1"
THRESHOLDS: Sequence[int] = (55, 60, 65, 70)

BASELINE_COLS: Sequence[str] = (
    "prob_over25",
    "prob_over25_v2",
    "odds_ft_over25",
    "odds_ft_under25",
    "od_over",
    "od_under",
    "bookie_implied",
    "bookie_implied_novig",
    "over25_rate_5_home",
    "over25_rate_5_away",
    "under25_rate_5_home",
    "under25_rate_5_away",
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
)

PROXY_COLS: Sequence[str] = (
    "snap_xg_total_pressure_proxy",
    "snap_style_chaos_index_proxy",
    "snap_ou25_over_regime_blend_proxy",
    "snapshot_ou25_support_score_proxy",
    "snap_timing_both_teams_late_risk_proxy",
    "snap_home_attack_vs_away_def_xg_proxy",
    "snap_home_attack_vs_away_def_goals_proxy",
)

SIGNAL_COLS: Sequence[str] = (
    "model_p_for_bookie",
    "signal_over25",
    "ou25_policy_branch",
    "ou25_runtime_lane",
)

LEAGUE_COL = "league"
MARKET_COL = "market"
TARGET_PICK_COL = "actual_pick_result"

SELECTION_COL = "selection"
BOOKIE_PICK_COL = "bookie_pick"
IS_WIN_COL = "is_win"
ACTUAL_LABEL_CANDIDATES: Sequence[str] = (
    "actual_ou25_label",
    "actual_pick_result",
    "actual_label",
)
TOTAL_GOALS_CANDIDATES: Sequence[str] = (
    "total_goals",
    "total_goal_count",
    "FT_total_goals",
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
FIXTURE_KEY_CANDIDATES: Sequence[str] = (
    "fixture_key",
    "fixture_key_ascii",
    "__fixture_key__",
    "match_id",
    "fixture_id",
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Audit OU25 proxy lift on proxy-enriched ALLMARKETS history")
    ap.add_argument("--input-csv", required=True, help="Proxy-enriched ALLMARKETS historical CSV")
    ap.add_argument(
        "--merged-dir",
        default="Matches/__merged__proxy_enriched",
        help="Directory containing league-level merged truth files used to recover actual OU25 outcomes",
    )
    ap.add_argument(
        "--merged-glob",
        default="*__merged__proxy_enriched.csv",
        help="Glob used to discover merged truth files",
    )
    ap.add_argument("--summary-csv", default=None, help="Optional summary CSV output")
    ap.add_argument("--detail-csv", default=None, help="Optional detail CSV output")
    ap.add_argument("--min-rows", type=int, default=80, help="Minimum OU25 rows per league")
    ap.add_argument("--min-train-rows", type=int, default=50, help="Minimum train rows per league")
    ap.add_argument("--min-test-rows", type=int, default=20, help="Minimum test rows per league")
    return ap.parse_args()


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


# --- Helper functions for merged truth file matching ---
def _find_first(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _normalize_text(value: object) -> str:
    text = str(value or "").strip().lower()
    for ch in (" ", "-", "/", "\\", ".", ",", ":", ";", "'", '"', "(", ")", "[", "]", "{", "}", "&", "+"):
        text = text.replace(ch, "_")
    text = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in text)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def _coerce_date_text(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    out = dt.dt.strftime("%Y-%m-%d")
    return out.fillna(series.astype(str).str.strip().str[:10])


def _normalize_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).map(_normalize_text)


def _ensure_join_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    fixture_col = _find_first(out, FIXTURE_KEY_CANDIDATES)
    if fixture_col is not None:
        out["__join_fixture_key__"] = out[fixture_col].fillna("").astype(str).str.strip()
        out["__join_fixture_key_norm__"] = _normalize_series(out[fixture_col])
    else:
        out["__join_fixture_key__"] = ""
        out["__join_fixture_key_norm__"] = ""

    date_col = _find_first(out, MATCH_DATE_CANDIDATES)
    home_col = _find_first(out, HOME_TEAM_CANDIDATES)
    away_col = _find_first(out, AWAY_TEAM_CANDIDATES)

    date_text = _coerce_date_text(out[date_col]) if date_col is not None else pd.Series("", index=out.index, dtype="object")
    home_text = _normalize_series(out[home_col]) if home_col is not None else pd.Series("", index=out.index, dtype="object")
    away_text = _normalize_series(out[away_col]) if away_col is not None else pd.Series("", index=out.index, dtype="object")

    out["__join_date_text__"] = date_text.fillna("").astype(str).str.strip()
    out["__join_home_norm__"] = home_text.fillna("")
    out["__join_away_norm__"] = away_text.fillna("")
    out["__join_composite_key__"] = (
        out["__join_date_text__"] + "__" + out["__join_home_norm__"] + "__" + out["__join_away_norm__"]
    )
    return out


def _league_key_from_merged_path(p: Path) -> str:
    stem = p.stem
    if stem.endswith("__merged__proxy_enriched"):
        stem = stem[: -len("__merged__proxy_enriched")]
    elif stem.endswith("__merged"):
        stem = stem[: -len("__merged")]
    return _normalize_text(stem)


def _league_key_from_value(value: object) -> str:
    return _normalize_text(value)


def _discover_merged_files(merged_dir: Path, glob_pat: str) -> List[Path]:
    return sorted([p.resolve() for p in merged_dir.glob(glob_pat) if p.is_file()])


def _build_truth_lookup(merged_files: Sequence[Path]) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    for p in merged_files:
        key = _league_key_from_merged_path(p)
        if key and key not in lookup:
            lookup[key] = p
    return lookup


def _prepare_truth_frame(df: pd.DataFrame) -> pd.DataFrame:
    truth = _ensure_join_keys(df)

    y = pd.Series(np.nan, index=truth.index, dtype="float64")
    for col in ACTUAL_LABEL_CANDIDATES:
        if col not in truth.columns:
            continue
        s = truth[col].astype(str).str.strip().str.upper()
        y.loc[s == "OVER25"] = 1.0
        y.loc[s == "UNDER25"] = 0.0

    if y.notna().sum() == 0:
        for col in TOTAL_GOALS_CANDIDATES:
            if col not in truth.columns:
                continue
            tg = _num(truth, col)
            m = tg.notna()
            if m.any():
                y.loc[m] = (tg.loc[m] >= 3).astype(float)
                break

    if y.notna().sum() == 0 and {"home_goals", "away_goals"}.issubset(truth.columns):
        hg = _num(truth, "home_goals")
        ag = _num(truth, "away_goals")
        m = hg.notna() & ag.notna()
        if m.any():
            y.loc[m] = ((hg.loc[m] + ag.loc[m]) >= 3).astype(float)

    truth["__truth_target__"] = y
    truth = truth.loc[truth["__truth_target__"].notna()].copy()

    keep_cols = [
        "__join_fixture_key__",
        "__join_fixture_key_norm__",
        "__join_composite_key__",
        "__truth_target__",
    ]
    truth = truth[keep_cols].copy()

    if truth["__join_fixture_key__"].astype(str).str.strip().ne("").any():
        truth_raw = truth.loc[truth["__join_fixture_key__"].astype(str).str.strip().ne("")].drop_duplicates(
            subset=["__join_fixture_key__"], keep="first"
        )
    else:
        truth_raw = truth.iloc[0:0].copy()

    if truth["__join_fixture_key_norm__"].astype(str).str.strip().ne("").any():
        truth_norm = truth.loc[truth["__join_fixture_key_norm__"].astype(str).str.strip().ne("")].drop_duplicates(
            subset=["__join_fixture_key_norm__"], keep="first"
        )
    else:
        truth_norm = truth.iloc[0:0].copy()

    if truth["__join_composite_key__"].astype(str).str.strip().ne("").any():
        truth_comp = truth.loc[truth["__join_composite_key__"].astype(str).str.strip().ne("")].drop_duplicates(
            subset=["__join_composite_key__"], keep="first"
        )
    else:
        truth_comp = truth.iloc[0:0].copy()

    out = {
        "raw": truth_raw,
        "norm": truth_norm,
        "comp": truth_comp,
    }
    return out


def _attach_truth_from_merged(df: pd.DataFrame, merged_dir: Path, merged_glob: str) -> pd.DataFrame:
    out = _ensure_join_keys(df)
    merged_files = _discover_merged_files(merged_dir, merged_glob)
    if not merged_files:
        raise ValueError(f"No merged truth files matched glob='{merged_glob}' in {merged_dir}")

    truth_lookup = _build_truth_lookup(merged_files)
    if not truth_lookup:
        raise ValueError(f"Could not build truth lookup from merged files in {merged_dir}")

    out["__target__"] = np.nan

    if LEAGUE_COL not in out.columns:
        raise ValueError(f"Input CSV missing required league column: {LEAGUE_COL}")

    for league_value, idx in out.groupby(LEAGUE_COL, dropna=False).groups.items():
        league_key = _league_key_from_value(league_value)
        truth_path = truth_lookup.get(league_key)
        if truth_path is None:
            continue

        truth_df = pd.read_csv(truth_path, low_memory=False)
        truth_maps = _prepare_truth_frame(truth_df)
        block = out.loc[idx].copy()

        if not truth_maps["raw"].empty:
            merged = block[["__join_fixture_key__"]].merge(
                truth_maps["raw"][["__join_fixture_key__", "__truth_target__"]],
                on="__join_fixture_key__",
                how="left",
            )
            out.loc[idx, "__target__"] = merged["__truth_target__"].values

        need = out.loc[idx, "__target__"].isna()
        if need.any() and not truth_maps["norm"].empty:
            need_idx = out.loc[idx].index[need]
            merged = out.loc[need_idx, ["__join_fixture_key_norm__"]].merge(
                truth_maps["norm"][["__join_fixture_key_norm__", "__truth_target__"]],
                on="__join_fixture_key_norm__",
                how="left",
            )
            out.loc[need_idx, "__target__"] = merged["__truth_target__"].values

        need = out.loc[idx, "__target__"].isna()
        if need.any() and not truth_maps["comp"].empty:
            need_idx = out.loc[idx].index[need]
            merged = out.loc[need_idx, ["__join_composite_key__"]].merge(
                truth_maps["comp"][["__join_composite_key__", "__truth_target__"]],
                on="__join_composite_key__",
                how="left",
            )
            out.loc[need_idx, "__target__"] = merged["__truth_target__"].values

    return out


def _resolve_target(df: pd.DataFrame) -> pd.Series:
    y = pd.Series(np.nan, index=df.index, dtype="float64")

    # 1) Direct actual OU25 labels when present
    for col in ACTUAL_LABEL_CANDIDATES:
        if col not in df.columns:
            continue
        s = df[col].astype(str).str.strip().str.upper()
        y.loc[s == "OVER25"] = 1.0
        y.loc[s == "UNDER25"] = 0.0
        if y.notna().sum() > 0:
            return y

    # 2) Rebuild actual OU25 label from total-goals columns when present
    for col in TOTAL_GOALS_CANDIDATES:
        if col not in df.columns:
            continue
        tg = _num(df, col)
        m = tg.notna()
        if m.any():
            y.loc[m] = (tg.loc[m] >= 3).astype(float)
            return y

    # 3) Rebuild from home/away goals when present
    if {"home_goals", "away_goals"}.issubset(df.columns):
        hg = _num(df, "home_goals")
        ag = _num(df, "away_goals")
        m = hg.notna() & ag.notna()
        if m.any():
            y.loc[m] = ((hg.loc[m] + ag.loc[m]) >= 3).astype(float)
            return y

    # 4) Rebuild from selection/bookie_pick + is_win for already-scored files
    if IS_WIN_COL in df.columns:
        is_win = _num(df, IS_WIN_COL)
        pick_col = None
        if SELECTION_COL in df.columns:
            pick_col = SELECTION_COL
        elif BOOKIE_PICK_COL in df.columns:
            pick_col = BOOKIE_PICK_COL

        if pick_col is not None:
            pick = df[pick_col].astype(str).str.strip().str.upper()
            over_mask = pick.eq("OVER25")
            under_mask = pick.eq("UNDER25")

            m_over = is_win.notna() & over_mask
            m_under = is_win.notna() & under_mask

            if m_over.any() or m_under.any():
                y.loc[m_over] = is_win.loc[m_over]
                y.loc[m_under] = 1.0 - is_win.loc[m_under]
                if y.notna().sum() > 0:
                    return y

    available_truthish = [
        c
        for c in (
            list(ACTUAL_LABEL_CANDIDATES)
            + list(TOTAL_GOALS_CANDIDATES)
            + ["home_goals", "away_goals", IS_WIN_COL, SELECTION_COL, BOOKIE_PICK_COL]
        )
        if c in df.columns
    ]
    raise ValueError(
        "Could not resolve OU25 target. Available truth-related columns: "
        + (", ".join(available_truthish) if available_truthish else "<none>")
    )


def _resolve_time(df: pd.DataFrame) -> pd.Series:
    for col in ("match_date", "date_GMT", "date", "timestamp"):
        if col not in df.columns:
            continue
        if col == "timestamp":
            ts = pd.to_numeric(df[col], errors="coerce")
            unit = "ms" if ts.notna().any() and float(ts.max()) > 1e11 else "s"
            return pd.to_datetime(ts, errors="coerce", utc=True, unit=unit)
        return pd.to_datetime(df[col], errors="coerce", utc=True, format="mixed")
    return pd.Series(pd.NaT, index=df.index)


def _encode_signal_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "signal_over25" in out.columns:
        mapping = {
            "VERY_STRONG_UNDER": -3,
            "STRONG_UNDER": -2,
            "WEAK_UNDER": -1,
            "NEUTRAL": 0,
            "WEAK_OVER": 1,
            "STRONG_OVER": 2,
            "VERY_STRONG_OVER": 3,
        }
        s = out["signal_over25"].astype(str).str.strip().str.upper().map(mapping)
        out["signal_over25_ord"] = pd.to_numeric(s, errors="coerce")

    for raw_col, new_col in (
        ("ou25_policy_branch", "ou25_policy_branch_code"),
        ("ou25_runtime_lane", "ou25_runtime_lane_code"),
    ):
        if raw_col in out.columns:
            vals = out[raw_col].astype(str).str.strip().fillna("")
            vals = vals.where(vals.ne(""), "__missing__")
            codes, _ = pd.factorize(vals, sort=True)
            out[new_col] = pd.Series(codes, index=out.index).replace(-1, np.nan)

    return out


def _select_usable_columns(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    usable: List[str] = []
    for col in cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() == 0:
            continue
        usable.append(col)
    return usable


def _dedupe_cols(cols: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _threshold_metrics(y_true: pd.Series, proba: pd.Series) -> Dict[str, float]:
    out: Dict[str, float] = {}
    y = pd.to_numeric(y_true, errors="coerce").astype(int)
    p = pd.to_numeric(proba, errors="coerce")

    for threshold in THRESHOLDS:
        cut = threshold / 100.0
        deploy = (p >= cut) | (p <= (1.0 - cut))
        deployed = int(deploy.sum())
        coverage = float(deployed / len(p)) if len(p) else np.nan
        if deployed == 0:
            hit_rate = np.nan
        else:
            pred = (p.loc[deploy] >= 0.5).astype(int)
            hit_rate = float((pred == y.loc[deploy]).mean())

        out[f"hit_rate_{threshold}"] = hit_rate
        out[f"coverage_{threshold}"] = coverage
        out[f"deployed_rows_{threshold}"] = deployed

    return out


def _fit_lane(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: Sequence[str]) -> Dict[str, object]:
    usable = _dedupe_cols([c for c in cols if c in train_df.columns and c in test_df.columns])
    usable = [c for c in usable if pd.to_numeric(train_df[c], errors="coerce").notna().sum() > 0]
    usable = [c for c in usable if pd.to_numeric(test_df[c], errors="coerce").notna().sum() > 0]

    if not usable:
        raise ValueError("No usable columns for lane")

    X_train = train_df[usable].apply(pd.to_numeric, errors="coerce")
    X_test = test_df[usable].apply(pd.to_numeric, errors="coerce")
    y_train = pd.to_numeric(train_df["__target__"], errors="coerce").astype(int)
    y_test = pd.to_numeric(test_df["__target__"], errors="coerce").astype(int)

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    pipe.fit(X_train, y_train)

    proba = pd.Series(pipe.predict_proba(X_test)[:, 1], index=test_df.index).clip(1e-6, 1 - 1e-6)

    metrics: Dict[str, object] = {
        "usable_cols": usable,
        "feature_count": len(usable),
        "auc": float(roc_auc_score(y_test, proba)) if y_test.nunique() > 1 else np.nan,
        "logloss": float(log_loss(y_test, proba, labels=[0, 1])),
        "brier": float(brier_score_loss(y_test, proba)),
    }
    metrics.update(_threshold_metrics(y_test, proba))
    return metrics


def _prepare_ou25(df: pd.DataFrame, merged_dir: Path, merged_glob: str) -> pd.DataFrame:
    out = df.copy()
    out = out.loc[out[MARKET_COL].astype(str).str.lower().eq("ou25")].copy()
    out = _encode_signal_cols(out)

    try:
        out["__target__"] = _resolve_target(out)
    except Exception:
        out = _attach_truth_from_merged(out, merged_dir=merged_dir, merged_glob=merged_glob)

    if "__target__" not in out.columns or out["__target__"].notna().sum() == 0:
        raise ValueError("Could not resolve OU25 target from input CSV or merged truth files")

    out["__time__"] = _resolve_time(out)

    if "fixture_key" in out.columns:
        out = out.sort_values(["__time__", "fixture_key"], kind="mergesort")
        out = out.drop_duplicates(subset=["fixture_key"], keep="first")
    else:
        out = out.sort_values(["__time__"], kind="mergesort")

    out = out.loc[out["__target__"].notna()].copy()
    out = out.drop(
        columns=[
            "__join_fixture_key__",
            "__join_fixture_key_norm__",
            "__join_date_text__",
            "__join_home_norm__",
            "__join_away_norm__",
            "__join_composite_key__",
        ],
        errors="ignore",
    )
    return out


def _split_time_ordered(group: pd.DataFrame, min_train_rows: int, min_test_rows: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(group)
    split_idx = int(np.floor(n * 0.8))
    split_idx = max(split_idx, min_train_rows)
    split_idx = min(split_idx, n - min_test_rows)
    if split_idx <= 0 or split_idx >= n:
        raise ValueError("invalid train/test split")
    return group.iloc[:split_idx].copy(), group.iloc[split_idx:].copy()


def main() -> None:
    args = _parse_args()
    input_csv = Path(args.input_csv).expanduser().resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    raw = pd.read_csv(input_csv, low_memory=False)
    merged_dir = Path(args.merged_dir).expanduser().resolve()
    if not merged_dir.exists():
        raise FileNotFoundError(f"Merged truth directory not found: {merged_dir}")

    try:
        ou = _prepare_ou25(raw, merged_dir=merged_dir, merged_glob=args.merged_glob)
    except Exception as exc:
        truthish = [
            c
            for c in (
                list(ACTUAL_LABEL_CANDIDATES)
                + list(TOTAL_GOALS_CANDIDATES)
                + ["home_goals", "away_goals", IS_WIN_COL, SELECTION_COL, BOOKIE_PICK_COL]
            )
            if c in raw.columns
        ]
        raise ValueError(
            f"Failed to prepare OU25 rows from {input_csv}. "
            f"Reason: {exc}. Truth-related columns present: {truthish if truthish else ['<none>']}. "
            f"Merged truth dir used: {merged_dir} | merged_glob={args.merged_glob}"
        ) from exc

    baseline_pool = list(BASELINE_COLS) + ["signal_over25_ord", "ou25_policy_branch_code", "ou25_runtime_lane_code", "model_p_for_bookie"]
    proxy_pool = list(PROXY_COLS)

    summary_rows: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []

    for league, group in ou.groupby(LEAGUE_COL, dropna=False):
        group = group.sort_values(["__time__"], kind="mergesort").copy()
        rows_total = len(group)

        if rows_total < int(args.min_rows):
            print(f"SKIP: {league} | too few rows | rows={rows_total}")
            continue

        try:
            train_df, test_df = _split_time_ordered(
                group=group,
                min_train_rows=int(args.min_train_rows),
                min_test_rows=int(args.min_test_rows),
            )
        except Exception as exc:
            print(f"SKIP: {league} | {exc}")
            continue

        baseline_usable = _select_usable_columns(group, baseline_pool)
        proxy_usable = _select_usable_columns(group, proxy_pool)

        try:
            baseline_lane = _fit_lane(train_df, test_df, baseline_usable)
            proxy_lane = _fit_lane(train_df, test_df, baseline_usable + proxy_usable)
        except Exception as exc:
            print(f"SKIP: {league} | fit failed | {exc}")
            continue

        for lane_name, lane in (("baseline", baseline_lane), ("proxy", proxy_lane)):
            d: Dict[str, object] = {
                "league": league,
                "rows_total": rows_total,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "lane_name": lane_name,
                "feature_count": lane["feature_count"],
                "usable_cols": " | ".join(lane["usable_cols"]),
                "auc": lane["auc"],
                "logloss": lane["logloss"],
                "brier": lane["brier"],
            }
            for threshold in THRESHOLDS:
                d[f"hit_rate_{threshold}"] = lane[f"hit_rate_{threshold}"]
                d[f"coverage_{threshold}"] = lane[f"coverage_{threshold}"]
                d[f"deployed_rows_{threshold}"] = lane[f"deployed_rows_{threshold}"]
            detail_rows.append(d)

        s: Dict[str, object] = {
            "league": league,
            "rows_total": rows_total,
            "train_rows": len(train_df),
            "test_rows": len(test_df),

            "baseline_auc": baseline_lane["auc"],
            "baseline_logloss": baseline_lane["logloss"],
            "baseline_brier": baseline_lane["brier"],

            "proxy_auc": proxy_lane["auc"],
            "proxy_logloss": proxy_lane["logloss"],
            "proxy_brier": proxy_lane["brier"],

            "delta_auc_proxy_vs_baseline": proxy_lane["auc"] - baseline_lane["auc"],
            "delta_logloss_proxy_vs_baseline": proxy_lane["logloss"] - baseline_lane["logloss"],
            "delta_brier_proxy_vs_baseline": proxy_lane["brier"] - baseline_lane["brier"],

            "baseline_feature_count": baseline_lane["feature_count"],
            "proxy_feature_count": proxy_lane["feature_count"],

            "baseline_cols_usable": len(baseline_usable),
            "proxy_cols_usable": len(proxy_usable),

            "usable_baseline_cols": " | ".join(baseline_usable),
            "usable_proxy_cols": " | ".join(proxy_usable),
        }

        for threshold in THRESHOLDS:
            s[f"baseline_hit_rate_{threshold}"] = baseline_lane[f"hit_rate_{threshold}"]
            s[f"baseline_coverage_{threshold}"] = baseline_lane[f"coverage_{threshold}"]
            s[f"baseline_deployed_rows_{threshold}"] = baseline_lane[f"deployed_rows_{threshold}"]

            s[f"proxy_hit_rate_{threshold}"] = proxy_lane[f"hit_rate_{threshold}"]
            s[f"proxy_coverage_{threshold}"] = proxy_lane[f"coverage_{threshold}"]
            s[f"proxy_deployed_rows_{threshold}"] = proxy_lane[f"deployed_rows_{threshold}"]

        summary_rows.append(s)
        print(
            f"OK: {league} | rows={rows_total} "
            f"| baseline_auc={baseline_lane['auc']:.4f} "
            f"| proxy_auc={proxy_lane['auc']:.4f} "
            f"| delta_auc={proxy_lane['auc'] - baseline_lane['auc']:.4f}"
        )

    print(f"\nSCRIPT_VERSION: {SCRIPT_VERSION}")
    print(f"INPUT_CSV: {input_csv}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values("delta_auc_proxy_vs_baseline", ascending=False, kind="mergesort")
        print("\nSUMMARY")
        with pd.option_context("display.max_rows", 200, "display.max_columns", None, "display.width", 320):
            print(summary_df.to_string(index=False))

        if args.summary_csv:
            out_path = Path(args.summary_csv).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            summary_df.to_csv(out_path, index=False)
            print(f"\nWROTE SUMMARY CSV: {out_path}")
    else:
        print("No valid league groups evaluated.")

    if detail_rows and args.detail_csv:
        detail_df = pd.DataFrame(detail_rows)
        out_path = Path(args.detail_csv).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        detail_df.to_csv(out_path, index=False)
        print(f"WROTE DETAIL CSV: {out_path}")


if __name__ == "__main__":
    main()