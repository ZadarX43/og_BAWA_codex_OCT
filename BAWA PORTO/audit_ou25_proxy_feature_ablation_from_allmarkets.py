#!/usr/bin/env python3
from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_VERSION = "audit_ou25_proxy_feature_ablation_from_allmarkets_v1"
MARKET = "ou25"
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
    "signal_over25_ord",
    "ou25_policy_branch_code",
    "ou25_runtime_lane_code",
    "model_p_for_bookie",
)

PROXY_SCORE_COLS: Sequence[str] = (
    "snapshot_ou25_support_score_proxy",
)

PROXY_REGIME_COLS: Sequence[str] = (
    "snap_xg_total_pressure_proxy",
    "snap_style_chaos_index_proxy",
    "snap_ou25_over_regime_blend_proxy",
)

PROXY_MATCHUP_COLS: Sequence[str] = (
    "snap_home_attack_vs_away_def_xg_proxy",
    "snap_home_attack_vs_away_def_goals_proxy",
)

PROXY_TIMING_COLS: Sequence[str] = (
    "snap_timing_both_teams_late_risk_proxy",
)

PROXY_FULL_BUNDLE_COLS: Sequence[str] = (
    "snap_xg_total_pressure_proxy",
    "snap_style_chaos_index_proxy",
    "snap_ou25_over_regime_blend_proxy",
    "snapshot_ou25_support_score_proxy",
    "snap_timing_both_teams_late_risk_proxy",
    "snap_home_attack_vs_away_def_xg_proxy",
    "snap_home_attack_vs_away_def_goals_proxy",
)

LANE_SPECS: Sequence[tuple[str, Sequence[str]]] = (
    ("baseline", ()),
    ("baseline_plus_score_proxy", PROXY_SCORE_COLS),
    ("baseline_plus_regime_proxy", PROXY_REGIME_COLS),
    ("baseline_plus_matchup_proxy", PROXY_MATCHUP_COLS),
    ("baseline_plus_timing_proxy", PROXY_TIMING_COLS),
    ("baseline_plus_full_proxy_bundle", PROXY_FULL_BUNDLE_COLS),
)

TRUTH_LABEL_COLS: Sequence[str] = (
    "actual_pick_result",
    "actual_ou25_label",
    "actual_label",
    "result_label",
)

TIME_COL_CANDIDATES: Sequence[str] = (
    "match_date",
    "date_GMT",
    "date",
    "timestamp",
)

LEAGUE_COL_CANDIDATES: Sequence[str] = (
    "league",
    "competition",
    "division",
)

FIXTURE_KEY_CANDIDATES: Sequence[str] = (
    "fixture_key",
    "fixture_key_ascii",
    "__fixture_key__",
    "match_id",
    "fixture_id",
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

HOME_GOALS_CANDIDATES: Sequence[str] = (
    "home_goals",
    "home_team_goal_count",
    "FT_home_goals",
    "full_time_home_goals",
    "home_score",
    "goals_home",
)

AWAY_GOALS_CANDIDATES: Sequence[str] = (
    "away_goals",
    "away_team_goal_count",
    "FT_away_goals",
    "full_time_away_goals",
    "away_score",
    "goals_away",
)

TOTAL_GOALS_CANDIDATES: Sequence[str] = (
    "total_goals",
    "total_goal_count",
    "FT_total_goals",
    "full_time_total_goals",
    "goals_total",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit OU25 proxy feature ablations from a historical ALLMARKETS export."
    )
    parser.add_argument("--input-csv", required=True, help="Historical ALLMARKETS CSV with proxy columns attached")
    parser.add_argument(
        "--merged-dir",
        default="Matches/__merged__proxy_enriched",
        help="Directory containing league-level merged/proxy-enriched CSVs used to resolve historical OU25 truth",
    )
    parser.add_argument("--summary-csv", default=None, help="Optional summary CSV output path")
    parser.add_argument("--detail-csv", default=None, help="Optional detail CSV output path")
    parser.add_argument("--min-rows", type=int, default=120, help="Minimum OU25 rows per league")
    parser.add_argument("--min-train-rows", type=int, default=80, help="Minimum train rows")
    parser.add_argument("--min-test-rows", type=int, default=30, help="Minimum test rows")
    parser.add_argument(
        "--sort-col",
        default=None,
        help="Optional explicit sort column. Defaults to best available time column.",
    )
    return parser.parse_args()



def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _coalesce_numeric(df: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().sum() > 0:
                return s
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _dedupe_names(cols: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for col in cols:
        if col not in seen:
            seen.add(col)
            out.append(col)
    return out


def _find_first(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _normalize_text(value: object) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    for ch in (" ", "-", "/", "\\", ".", ",", ":", ";", "'", '"', "(", ")", "[", "]", "{", "}", "&", "+"):
        text = text.replace(ch, "_")
    text = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in text)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def _normalize_fixture_key_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).map(_normalize_text)


def _normalize_team_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).map(_normalize_text)


def _coerce_date_text(series: pd.Series) -> pd.Series:
    try:
        dt = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    except TypeError:
        dt = pd.to_datetime(series, errors="coerce", utc=True)
    out = dt.dt.strftime("%Y-%m-%d")
    return out.fillna(series.astype(str).str.strip().str[:10])


def _ensure_join_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    fixture_col = _find_first(out, FIXTURE_KEY_CANDIDATES)
    if fixture_col is not None:
        out["__join_fixture_key__"] = out[fixture_col].fillna("").astype(str).str.strip()
        out["__join_fixture_key_norm__"] = _normalize_fixture_key_series(out[fixture_col])
    else:
        out["__join_fixture_key__"] = ""
        out["__join_fixture_key_norm__"] = ""

    date_col = _find_first(out, TIME_COL_CANDIDATES)
    home_col = _find_first(out, HOME_TEAM_CANDIDATES)
    away_col = _find_first(out, AWAY_TEAM_CANDIDATES)

    if date_col is not None:
        date_text = _coerce_date_text(out[date_col])
    else:
        date_text = pd.Series("", index=out.index, dtype="object")

    if home_col is not None:
        home_text = _normalize_team_series(out[home_col])
    else:
        home_text = pd.Series("", index=out.index, dtype="object")

    if away_col is not None:
        away_text = _normalize_team_series(out[away_col])
    else:
        away_text = pd.Series("", index=out.index, dtype="object")

    out["__join_date_text__"] = date_text.fillna("").astype(str).str.strip()
    out["__join_home_norm__"] = home_text.fillna("")
    out["__join_away_norm__"] = away_text.fillna("")
    out["__join_composite_key__"] = (
        out["__join_date_text__"]
        + "__"
        + out["__join_home_norm__"]
        + "__"
        + out["__join_away_norm__"]
    )
    return out


def _resolve_league_tag(league_value: str) -> str:
    return _normalize_text(league_value)


# Helper function to enumerate all plausible merged file candidates for a league
def _league_file_candidates(league_value: str) -> List[str]:
    tag = _resolve_league_tag(league_value)
    candidates = [
        f"{tag}__merged__proxy_enriched.csv",
        f"{tag}__merged.csv",
    ]

    alias_map = {
        "england_premier_league": ["england_premier_league", "epl"],
        "england_championship": ["england_championship", "efl_championship"],
        "england_efl_league_1": ["england_efl_league_1", "efl_league_1", "england_league_1"],
        "france_ligue_1": ["france_ligue_1", "ligue_1"],
        "germany_bundesliga": ["germany_bundesliga", "bundesliga"],
        "italy_serie_a": ["italy_serie_a", "serie_a"],
        "spain_la_liga": ["spain_la_liga", "la_liga"],
        "netherlands_eredivisie": ["netherlands_eredivisie", "eredivisie"],
        "scotland_premiership": ["scotland_premiership", "scottish_premiership"],
        "usa_mls": ["usa_mls", "mls"],
        "brazil_serie_a": ["brazil_serie_a", "serie_a_brazil"],
        "belgium_pro": ["belgium_pro", "belgian_pro_league"],
        "portugal_liga": ["portugal_liga", "primeira_liga"],
    }

    for alias in alias_map.get(tag, []):
        candidates.append(f"{alias}__merged__proxy_enriched.csv")
        candidates.append(f"{alias}__merged.csv")

    seen: set[str] = set()
    out: List[str] = []
    for name in candidates:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


# --- Additional helper functions for improved merged lookup and resolution ---

def _build_merged_lookup(merged_dir: Path) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    if not merged_dir.exists():
        return lookup

    for p in sorted(merged_dir.glob("*.csv")):
        if not p.is_file():
            continue
        stem = p.stem
        norm_stem = _normalize_text(stem)
        candidates = [norm_stem]

        if norm_stem.endswith("_proxy_enriched"):
            candidates.append(norm_stem[: -len("_proxy_enriched")])
        if norm_stem.endswith("_merged"):
            candidates.append(norm_stem[: -len("_merged")])
        if norm_stem.endswith("_merged_proxy_enriched"):
            base = norm_stem[: -len("_merged_proxy_enriched")]
            candidates.append(base)
            candidates.append(f"{base}_merged")

        for key in candidates:
            if key and key not in lookup:
                lookup[key] = p

    return lookup


def _resolve_merged_path_for_league(league_value: str, merged_lookup: Dict[str, Path]) -> Optional[Path]:
    tag = _resolve_league_tag(league_value)
    candidates = _league_file_candidates(league_value)

    for name in candidates:
        stem_key = _normalize_text(Path(name).stem)
        if stem_key in merged_lookup:
            return merged_lookup[stem_key]

    if tag in merged_lookup:
        return merged_lookup[tag]
    if f"{tag}_merged" in merged_lookup:
        return merged_lookup[f"{tag}_merged"]
    if f"{tag}_merged_proxy_enriched" in merged_lookup:
        return merged_lookup[f"{tag}_merged_proxy_enriched"]

    for key, path in merged_lookup.items():
        if tag and (tag in key or key in tag):
            return path

    return None


def _resolve_truth_from_merged(pred_df: pd.DataFrame, merged_dir: Path) -> pd.Series:
    y = pd.Series(np.nan, index=pred_df.index, dtype="float64")
    if pred_df.empty:
        return y

    if "league" not in pred_df.columns:
        print("[TRUTH_RESOLVE] missing league column in prediction frame")
        return y

    merged_lookup = _build_merged_lookup(merged_dir)
    print(f"[TRUTH_RESOLVE] merged_lookup_keys={len(merged_lookup)} from {merged_dir}")

    pred = _ensure_join_keys(pred_df)

    for league_value, idx in pred.groupby("league", dropna=False).groups.items():
        league_text = str(league_value).strip()
        if not league_text:
            print("[TRUTH_RESOLVE] skip empty league label")
            continue

        merged_path = _resolve_merged_path_for_league(league_text, merged_lookup)
        if merged_path is None:
            print(f"[TRUTH_RESOLVE] league={league_text} | merged file not found")
            continue

        merged = pd.read_csv(merged_path, low_memory=False)
        merged = _ensure_join_keys(merged)

        home_goals = _coalesce_numeric(merged, HOME_GOALS_CANDIDATES)
        away_goals = _coalesce_numeric(merged, AWAY_GOALS_CANDIDATES)
        total_goals = _coalesce_numeric(merged, TOTAL_GOALS_CANDIDATES)
        target = pd.Series(np.nan, index=merged.index, dtype="float64")

        valid_pair = home_goals.notna() & away_goals.notna()
        target.loc[valid_pair] = ((home_goals.loc[valid_pair] + away_goals.loc[valid_pair]) >= 3).astype(float)

        valid_total = target.isna() & total_goals.notna()
        target.loc[valid_total] = (total_goals.loc[valid_total] >= 3).astype(float)

        merged["__truth_target__"] = target
        merged = merged.loc[merged["__truth_target__"].notna()].copy()
        if merged.empty:
            goal_cols_present = [
                c for c in [*HOME_GOALS_CANDIDATES, *AWAY_GOALS_CANDIDATES, *TOTAL_GOALS_CANDIDATES]
                if c in merged.columns
            ]
            print(
                f"[TRUTH_RESOLVE] league={league_text} | merged={merged_path.name} | "
                f"no truth rows after goal resolution | goal_cols_present={goal_cols_present}"
            )
            continue

        truth_cols = [
            "__join_fixture_key__",
            "__join_fixture_key_norm__",
            "__join_composite_key__",
            "__truth_target__",
        ]
        truth = merged[truth_cols].copy()

        truth_raw = truth.loc[truth["__join_fixture_key__"].astype(str).str.strip() != ""].drop_duplicates(
            subset=["__join_fixture_key__"], keep="last"
        )
        truth_norm = truth.loc[truth["__join_fixture_key_norm__"].astype(str).str.strip() != ""].drop_duplicates(
            subset=["__join_fixture_key_norm__"], keep="last"
        )
        truth_comp = truth.loc[truth["__join_composite_key__"].astype(str).str.strip() != ""].drop_duplicates(
            subset=["__join_composite_key__"], keep="last"
        )

        pred_sub = pred.loc[idx].copy()
        if pred_sub.empty:
            continue
        pred_target = pd.Series(np.nan, index=pred_sub.index, dtype="float64")

        raw_matches = 0
        norm_matches = 0
        comp_matches = 0

        if not truth_raw.empty:
            m = pred_sub[["__join_fixture_key__"]].merge(
                truth_raw[["__join_fixture_key__", "__truth_target__"]],
                on="__join_fixture_key__",
                how="left",
            )
            pred_target.loc[:] = pd.to_numeric(m["__truth_target__"].values, errors="coerce")
            raw_matches = int(pred_target.notna().sum())

        need = pred_target.isna()
        if need.any() and not truth_norm.empty:
            m = pred_sub.loc[need, ["__join_fixture_key_norm__"]].merge(
                truth_norm[["__join_fixture_key_norm__", "__truth_target__"]],
                on="__join_fixture_key_norm__",
                how="left",
            )
            pred_target.loc[need] = pd.to_numeric(m["__truth_target__"].values, errors="coerce")
            norm_matches = int(pred_target.notna().sum()) - raw_matches

        need = pred_target.isna()
        if need.any() and not truth_comp.empty:
            m = pred_sub.loc[need, ["__join_composite_key__"]].merge(
                truth_comp[["__join_composite_key__", "__truth_target__"]],
                on="__join_composite_key__",
                how="left",
            )
            pred_target.loc[need] = pd.to_numeric(m["__truth_target__"].values, errors="coerce")
            comp_matches = int(pred_target.notna().sum()) - raw_matches - norm_matches

        matched_n = int(pred_target.notna().sum())
        print(
            f"[TRUTH_RESOLVE] league={league_text} | merged={merged_path.name} | "
            f"pred_rows={len(pred_sub)} | merged_truth_rows={len(merged)} | "
            f"raw_matches={raw_matches} | norm_matches={norm_matches} | comp_matches={comp_matches} | "
            f"matched_truth={matched_n}"
        )

        y.loc[pred_sub.index] = pred_target

    return y


def _resolve_time_series(df: pd.DataFrame, explicit_col: str | None = None) -> pd.Series:
    if explicit_col and explicit_col in df.columns:
        col = explicit_col
        if col == "timestamp":
            ts = pd.to_numeric(df[col], errors="coerce")
            mx = float(ts.max()) if ts.notna().any() else float("nan")
            unit = "ms" if np.isfinite(mx) and mx > 1.0e11 else "s"
            return pd.to_datetime(ts, errors="coerce", utc=True, unit=unit)
        return pd.to_datetime(df[col], errors="coerce", utc=True, format="mixed")

    for col in TIME_COL_CANDIDATES:
        if col not in df.columns:
            continue
        if col == "timestamp":
            ts = pd.to_numeric(df[col], errors="coerce")
            mx = float(ts.max()) if ts.notna().any() else float("nan")
            unit = "ms" if np.isfinite(mx) and mx > 1.0e11 else "s"
            return pd.to_datetime(ts, errors="coerce", utc=True, unit=unit)
        try:
            dt = pd.to_datetime(df[col], errors="coerce", utc=True, format="mixed")
        except TypeError:
            dt = pd.to_datetime(df[col], errors="coerce", utc=True)
        if dt.notna().sum() > 0:
            return dt

    return pd.Series(pd.NaT, index=df.index)


def _resolve_target(df: pd.DataFrame) -> pd.Series:
    y = pd.Series(np.nan, index=df.index, dtype="float64")

    for col in TRUTH_LABEL_COLS:
        if col not in df.columns:
            continue
        s = df[col].astype(str).str.strip().str.upper()
        y.loc[s.eq("OVER25")] = 1.0
        y.loc[s.eq("UNDER25")] = 0.0
        if y.notna().sum() > 0:
            return y

    if "is_win" in df.columns:
        win = pd.to_numeric(df["is_win"], errors="coerce")
        if win.notna().sum() > 0:
            pick = pd.Series("", index=df.index, dtype="object")
            for col in ("selection", "bookie_pick", "pick", "prediction"):
                if col in df.columns:
                    pick = df[col].astype(str).str.strip().str.upper()
                    if pick.ne("").any():
                        break
            over_mask = pick.eq("OVER25")
            under_mask = pick.eq("UNDER25")
            y.loc[over_mask & win.eq(1)] = 1.0
            y.loc[over_mask & win.eq(0)] = 0.0
            y.loc[under_mask & win.eq(1)] = 0.0
            y.loc[under_mask & win.eq(0)] = 1.0
            if y.notna().sum() > 0:
                return y

    home_goals = _coalesce_numeric(df, HOME_GOALS_CANDIDATES)
    away_goals = _coalesce_numeric(df, AWAY_GOALS_CANDIDATES)
    valid = home_goals.notna() & away_goals.notna()
    if valid.any():
        y.loc[valid] = ((home_goals.loc[valid] + away_goals.loc[valid]) >= 3).astype(float)
        return y

    total_goals = _coalesce_numeric(df, TOTAL_GOALS_CANDIDATES)
    valid_total = total_goals.notna()
    if valid_total.any():
        y.loc[valid_total] = (total_goals.loc[valid_total] >= 3).astype(float)
        return y

    available_truth_cols = [c for c in [*TRUTH_LABEL_COLS, "selection", "bookie_pick", "is_win", *HOME_GOALS_CANDIDATES, *AWAY_GOALS_CANDIDATES, *TOTAL_GOALS_CANDIDATES] if c in df.columns]
    raise ValueError(
        "Could not resolve OU25 target. Available truth-related columns: " + ", ".join(available_truth_cols)
    )


def _prepare_ou25(df: pd.DataFrame, merged_dir: Path, sort_col: str | None = None) -> pd.DataFrame:
    out = df.copy()
    if "market" in out.columns:
        out = out.loc[out["market"].astype(str).str.strip().str.lower().eq(MARKET)].copy()
    try:
        out["__target__"] = _resolve_target(out)
    except Exception as exc:
        print(f"[PREPARE_OU25] direct target resolution failed: {exc}")
        out["__target__"] = _resolve_truth_from_merged(out, merged_dir=merged_dir)

    if out["__target__"].notna().sum() == 0:
        print("[PREPARE_OU25] direct target rows were all null; retrying via merged truth")
        out["__target__"] = _resolve_truth_from_merged(out, merged_dir=merged_dir)

    matched_target_rows = int(out["__target__"].notna().sum())
    print(f"[PREPARE_OU25] market_rows={len(out)} | matched_target_rows={matched_target_rows}")

    out = out.loc[out["__target__"].notna()].copy()
    if out.empty:
        raise ValueError("OU25 frame is empty after target resolution")

    out["__time__"] = _resolve_time_series(out, explicit_col=sort_col)
    if "fixture_key" in out.columns:
        out = out.sort_values(["__time__", "fixture_key"], kind="mergesort")
        out = out.drop_duplicates(subset=["fixture_key"], keep="last")
    else:
        out = out.sort_values(["__time__"], kind="mergesort")
    return out.reset_index(drop=True)


def _resolve_league_value(df: pd.DataFrame, fallback: str) -> str:
    for col in LEAGUE_COL_CANDIDATES:
        if col in df.columns:
            vals = df[col].dropna().astype(str).str.strip()
            vals = vals[vals != ""]
            if not vals.empty:
                return vals.iloc[0]
    return fallback


def _select_usable_columns(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    usable: List[str] = []
    for col in _dedupe_names(cols):
        if col not in train_df.columns or col not in test_df.columns:
            continue
        tr = pd.to_numeric(train_df[col], errors="coerce")
        te = pd.to_numeric(test_df[col], errors="coerce")
        if tr.notna().sum() == 0:
            continue
        if te.notna().sum() == 0:
            continue
        usable.append(col)
    return usable


def _compute_threshold_metrics(y_true: pd.Series, proba: pd.Series) -> Dict[str, float]:
    out: Dict[str, float] = {}
    y = pd.to_numeric(y_true, errors="coerce").astype(int)
    p = pd.to_numeric(proba, errors="coerce")

    for threshold in THRESHOLDS:
        cut = threshold / 100.0
        deploy_mask = (p >= cut) | (p <= (1.0 - cut))
        deployed = int(deploy_mask.sum())
        coverage = float(deployed / len(p)) if len(p) else np.nan
        if deployed == 0:
            hit_rate = np.nan
        else:
            pred = (p.loc[deploy_mask] >= 0.5).astype(int)
            hit_rate = float((pred == y.loc[deploy_mask]).mean())
        out[f"hit_rate_{threshold}"] = hit_rate
        out[f"coverage_{threshold}"] = coverage
        out[f"deployed_rows_{threshold}"] = deployed
    return out


def _fit_lane(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: Sequence[str]) -> Dict[str, object]:
    usable = _select_usable_columns(train_df, test_df, cols)
    if not usable:
        raise ValueError("No usable features for lane")

    X_train = train_df[usable].apply(pd.to_numeric, errors="coerce")
    X_test = test_df[usable].apply(pd.to_numeric, errors="coerce")
    y_train = pd.to_numeric(train_df["__target__"], errors="coerce").astype(int)
    y_test = pd.to_numeric(test_df["__target__"], errors="coerce").astype(int)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    pipe.fit(X_train, y_train)

    proba = pd.Series(pipe.predict_proba(X_test)[:, 1], index=test_df.index).clip(1e-6, 1 - 1e-6)
    hard_pred = (proba >= 0.5).astype(int)

    metrics: Dict[str, object] = {
        "usable_cols": usable,
        "feature_count": len(usable),
        "auc": float(roc_auc_score(y_test, proba)) if y_test.nunique() > 1 else np.nan,
        "logloss": float(log_loss(y_test, proba, labels=[0, 1])),
        "brier": float(brier_score_loss(y_test, proba)),
        "accuracy": float((hard_pred == y_test).mean()),
    }
    metrics.update(_compute_threshold_metrics(y_test, proba))
    return metrics


def _split_train_test(df: pd.DataFrame, min_train_rows: int, min_test_rows: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_total = len(df)
    split_idx = int(np.floor(rows_total * 0.8))
    split_idx = max(split_idx, min_train_rows)
    split_idx = min(split_idx, rows_total - min_test_rows)
    if split_idx <= 0 or split_idx >= rows_total:
        raise ValueError(f"Invalid train/test split for rows_total={rows_total}")
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def main() -> None:
    args = _parse_args()
    input_csv = Path(args.input_csv).expanduser().resolve()
    merged_dir = Path(args.merged_dir).expanduser().resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if not merged_dir.exists():
        raise FileNotFoundError(f"Merged directory not found: {merged_dir}")

    raw = pd.read_csv(input_csv, low_memory=False)
    print(f"[INPUT] raw_rows={len(raw)} | cols={len(raw.columns)}")
    if "league" in raw.columns:
        league_preview = raw["league"].dropna().astype(str).str.strip()
        league_preview = league_preview[league_preview != ""]
        print(f"[INPUT] leagues_preview={league_preview.head(10).tolist()}")
    try:
        ou = _prepare_ou25(raw, merged_dir=merged_dir, sort_col=args.sort_col)
    except Exception as exc:
        truth_cols_present = [c for c in [*TRUTH_LABEL_COLS, "selection", "bookie_pick", "is_win", "home_goals", "away_goals", "total_goals"] if c in raw.columns]
        raise ValueError(
            f"Failed to prepare OU25 rows from {input_csv}. Reason: {exc}. Truth-related columns present: {truth_cols_present}. merged_dir={merged_dir}"
        ) from exc
    print(f"[POST_PREP] ou_rows={len(ou)} | leagues_present={ou['league'].nunique(dropna=True) if 'league' in ou.columns else 0}")

    summary_rows: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []

    if "league" in ou.columns:
        league_groups = list(ou.groupby("league", dropna=False))
    else:
        league_groups = [("ALL", ou)]

    print(f"[GROUPS] league_groups={len(league_groups)}")

    for league_name, league_df in league_groups:
        league_text = _resolve_league_value(league_df, fallback=str(league_name))
        league_df = league_df.copy().reset_index(drop=True)
        rows_total = len(league_df)
        print(f"[LEAGUE] {league_text} | rows_total={rows_total}")
        if rows_total < int(args.min_rows):
            print(f"SKIP: {league_text} | too few rows={rows_total}")
            continue

        try:
            train_df, test_df = _split_train_test(
                league_df,
                min_train_rows=int(args.min_train_rows),
                min_test_rows=int(args.min_test_rows),
            )
        except Exception as exc:
            print(f"SKIP: {league_text} | {exc}")
            continue

        lane_results: Dict[str, Dict[str, object]] = {}
        for lane_name, extra_cols in LANE_SPECS:
            lane_cols = tuple(BASELINE_COLS) + tuple(extra_cols)
            lane_results[lane_name] = _fit_lane(train_df, test_df, lane_cols)

        baseline = lane_results["baseline"]
        full_bundle = lane_results["baseline_plus_full_proxy_bundle"]
        print(
            f"OK: {league_text} | rows={rows_total} | "
            f"baseline_auc={baseline['auc']:.4f} | "
            f"proxy_auc={full_bundle['auc']:.4f} | "
            f"delta_auc={float(full_bundle['auc']) - float(baseline['auc']):.4f}"
        )

        for lane_name, metrics in lane_results.items():
            detail_row: Dict[str, object] = {
                "league": league_text,
                "rows_total": rows_total,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "lane_name": lane_name,
                "feature_count": metrics["feature_count"],
                "usable_cols": " | ".join(metrics["usable_cols"]),
                "auc": metrics["auc"],
                "logloss": metrics["logloss"],
                "brier": metrics["brier"],
                "accuracy": metrics["accuracy"],
            }
            for threshold in THRESHOLDS:
                detail_row[f"hit_rate_{threshold}"] = metrics[f"hit_rate_{threshold}"]
                detail_row[f"coverage_{threshold}"] = metrics[f"coverage_{threshold}"]
                detail_row[f"deployed_rows_{threshold}"] = metrics[f"deployed_rows_{threshold}"]
            detail_rows.append(detail_row)

        summary_row: Dict[str, object] = {
            "league": league_text,
            "rows_total": rows_total,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
        }

        for lane_name, _extra_cols in LANE_SPECS:
            metrics = lane_results[lane_name]
            prefix = lane_name.replace("baseline_plus_", "").replace("proxy_bundle", "full_proxy_bundle")
            if lane_name == "baseline":
                prefix = "baseline"
            summary_row[f"{prefix}_auc"] = metrics["auc"]
            summary_row[f"{prefix}_logloss"] = metrics["logloss"]
            summary_row[f"{prefix}_brier"] = metrics["brier"]
            summary_row[f"{prefix}_accuracy"] = metrics["accuracy"]
            summary_row[f"{prefix}_feature_count"] = metrics["feature_count"]
            summary_row[f"{prefix}_cols_usable"] = len(metrics["usable_cols"])
            summary_row[f"usable_{prefix}_cols"] = " | ".join(metrics["usable_cols"])
            for threshold in THRESHOLDS:
                summary_row[f"{prefix}_hit_rate_{threshold}"] = metrics[f"hit_rate_{threshold}"]
                summary_row[f"{prefix}_coverage_{threshold}"] = metrics[f"coverage_{threshold}"]
                summary_row[f"{prefix}_deployed_rows_{threshold}"] = metrics[f"deployed_rows_{threshold}"]

        for lane_name, _extra_cols in LANE_SPECS:
            if lane_name == "baseline":
                continue
            metrics = lane_results[lane_name]
            prefix = lane_name.replace("baseline_plus_", "").replace("proxy_bundle", "full_proxy_bundle")
            summary_row[f"delta_auc_{prefix}_vs_baseline"] = float(metrics["auc"]) - float(baseline["auc"])
            summary_row[f"delta_logloss_{prefix}_vs_baseline"] = float(metrics["logloss"]) - float(baseline["logloss"])
            summary_row[f"delta_brier_{prefix}_vs_baseline"] = float(metrics["brier"]) - float(baseline["brier"])
            summary_row[f"delta_accuracy_{prefix}_vs_baseline"] = float(metrics["accuracy"]) - float(baseline["accuracy"])

        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)

    print(f"\nSCRIPT_VERSION: {SCRIPT_VERSION}")
    print(f"INPUT_CSV: {input_csv}")
    print(f"MERGED_DIR: {merged_dir}")
    print("\nSUMMARY")
    with pd.option_context("display.max_rows", 200, "display.max_columns", None, "display.width", 400):
        if not summary_df.empty:
            print(summary_df.to_string(index=False))
        else:
            print("No valid leagues evaluated.")

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