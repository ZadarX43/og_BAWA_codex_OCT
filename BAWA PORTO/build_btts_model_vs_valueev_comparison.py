#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

WALK_ROOT = Path("predictions_output/walk_forward")
OUT_MONTHLY = Path("btts_model_vs_valueev_monthly.csv")
OUT_LEAGUE = Path("btts_model_vs_valueev_league.csv")
OUT_WINNER = Path("btts_model_vs_valueev_winner_table.csv")
OUT_MD = Path("btts_model_vs_valueev_comparison.md")


MIN_MONTHS_REVIEW = 3
MIN_ROWS_REVIEW = 20
MIN_SHARED_MONTHS_VALID = 3
MIN_SHARED_ROWS_VALID = 20
MIN_SHARED_ROWS_SPARSE = 1
SPARSE_WINNER_ROI_MARGIN_MIN = 0.02
RECOMMENDED_LIVE_LANE_SORT = {
    "btts_model": 0,
    "btts_model_shadow_valueev": 1,
    "model_primary_watch_valueev": 2,
    "no_call": 3,
}
BAD_LEAGUES = {
    "England Premier League",
    "Scotland Premiership",
    "Spain La Liga",
    "Brazil Serie A",
    "England FA Cup",
    "England EFL League 1",
}


DEPLOY_MONTHS = {
    "2024-11",
    "2024-12",
    "2025-01",
    "2025-02",
    "2025-03",
    "2025-04",
}

# BTTS policy constants
BTTS_POLICY_PRIMARY_LIVE_LANE = "btts_model"
BTTS_POLICY_SECONDARY_LANE = "btts_valueev_shadow_watch"
BTTS_POLICY_PROMOTION_STATUS = "no_valueev_live_promotion"
BTTS_POLICY_REVIEW_TRIGGER = (
    "Revisit only when shared months and shared rows both increase, and BTTS valueEV "
    "continues to beat model with a wider ROI margin."
)


def _safe_num(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _weighted_mean(g: pd.DataFrame, value_col: str, weight_col: str = "rows") -> float:
    x = pd.to_numeric(g[value_col], errors="coerce")
    w = pd.to_numeric(g[weight_col], errors="coerce")
    m = x.notna() & w.notna()
    if not m.any():
        return float("nan")
    wsum = float(w[m].sum())
    if wsum == 0.0:
        return float("nan")
    return float((x[m] * w[m]).sum() / wsum)


def _month_sort_key(month_tag: str) -> tuple[int, int]:
    s = str(month_tag).strip()
    parts = s.split("-")
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return (int(parts[0]), int(parts[1]))
    return (9999, 9999)


def _max_drawdown_from_roi(roi_series: pd.Series) -> float:
    s = pd.to_numeric(roi_series, errors="coerce")
    s = s.dropna()
    if s.empty:
        return float("nan")
    equity = (1.0 + s).cumprod()
    running_peak = equity.cummax()
    drawdown = (equity / running_peak) - 1.0
    return float(drawdown.min())


def _league_bucket(league: str) -> str:
    return "bad_league" if str(league) in BAD_LEAGUES else "other_league"


def _to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows_"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def _pick_first_existing(patterns: list[Path]) -> Path | None:
    for p in patterns:
        if p.exists():
            return p
    return None


def _load_gate_rows(csv_path: Path, month_tag: str, source_label: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    if df.empty:
        return pd.DataFrame(columns=["month", "league", "rows", "hit", "roi", "avg_odds", "source"])

    required = {"league", "correct", "bookie_od"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"{csv_path} missing required columns: {missing}")

    d = df.copy()
    d["league"] = d["league"].astype(str)
    d["correct"] = pd.to_numeric(d["correct"], errors="coerce")
    d["bookie_od"] = pd.to_numeric(d["bookie_od"], errors="coerce")
    d = d[d["correct"].notna()].copy()

    if d.empty:
        return pd.DataFrame(columns=["month", "league", "rows", "hit", "roi", "avg_odds", "source"])

    grouped = []
    for league, g in d.groupby("league", dropna=False):
        roi_series = (g["correct"] * g["bookie_od"]) - 1.0
        grouped.append(
            {
                "month": month_tag,
                "league": str(league),
                "rows": int(len(g)),
                "hit": float(g["correct"].mean()),
                "roi": float(roi_series.mean()),
                "avg_odds": float(g["bookie_od"].mean()),
                "source": source_label,
            }
        )
    return pd.DataFrame(grouped)


# New helper function for summary rows
def _load_summary_rows(csv_path: Path, month_tag: str, source_label: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    if df.empty:
        return pd.DataFrame(columns=["month", "league", "rows", "hit", "roi", "avg_odds", "source"])

    n_col = None
    hit_col = None
    roi_col = None
    avg_od_col = None

    if {"league", "n_btts", "hit_btts", "roi_btts", "avg_od_btts"}.issubset(df.columns):
        n_col = "n_btts"
        hit_col = "hit_btts"
        roi_col = "roi_btts"
        avg_od_col = "avg_od_btts"
    elif {"league", "n", "hit", "roi", "avg_od"}.issubset(df.columns):
        n_col = "n"
        hit_col = "hit"
        roi_col = "roi"
        avg_od_col = "avg_od"
    else:
        raise SystemExit(
            f"{csv_path} missing required BTTS summary columns; found columns={list(df.columns)}"
        )

    d = df.copy()
    d["league"] = d["league"].astype(str)
    d[n_col] = pd.to_numeric(d[n_col], errors="coerce")
    d[hit_col] = pd.to_numeric(d[hit_col], errors="coerce")
    d[roi_col] = pd.to_numeric(d[roi_col], errors="coerce")
    d[avg_od_col] = pd.to_numeric(d[avg_od_col], errors="coerce")
    d = d[d["league"].astype(str).str.strip().ne("")].copy()
    d = d[d[n_col].fillna(0) > 0].copy()

    if d.empty:
        return pd.DataFrame(columns=["month", "league", "rows", "hit", "roi", "avg_odds", "source"])

    return pd.DataFrame(
        {
            "month": month_tag,
            "league": d["league"].astype(str),
            "rows": d[n_col].astype(int),
            "hit": d[hit_col].astype(float),
            "roi": d[roi_col].astype(float),
            "avg_odds": d[avg_od_col].astype(float),
            "source": source_label,
        }
    )


def _collect_monthly_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not WALK_ROOT.exists():
        raise SystemExit(f"walk-forward root not found: {WALK_ROOT}")

    valueev_rows: list[pd.DataFrame] = []
    model_rows: list[pd.DataFrame] = []

    month_dirs = sorted([p for p in WALK_ROOT.iterdir() if p.is_dir()], key=lambda p: _month_sort_key(p.name))
    for month_dir in month_dirs:
        month_tag = month_dir.name

        valueev_csv = _pick_first_existing(
            [
                month_dir / "BTTS_VALUEEV_GATED_ROWS.csv",
                month_dir / "btts_valueev_gated_rows.csv",
            ]
        )
        valueev_summary_csv = _pick_first_existing(
            [
                month_dir / "investor_table_btts_valueEV.csv",
                month_dir / "investor_table_btts_valueev.csv",
                month_dir / "investor_table_btts_valueEv.csv",
                month_dir / "investor_table_BTTS_valueEV.csv",
            ]
        )
        model_csv = _pick_first_existing(
            [
                month_dir / "BTTS_MODEL_GATED_ROWS.csv",
                month_dir / "btts_model_gated_rows.csv",
            ]
        )

        debug_valueev_source = "none"
        debug_model_source = "none"

        valueev_month_df = pd.DataFrame()
        if valueev_csv is not None:
            valueev_month_df = _load_gate_rows(valueev_csv, month_tag, "btts_valueev")
            if not valueev_month_df.empty:
                debug_valueev_source = f"gated:{valueev_csv.name}"
        if valueev_month_df.empty and valueev_summary_csv is not None:
            valueev_month_df = _load_summary_rows(valueev_summary_csv, month_tag, "btts_valueev")
            if not valueev_month_df.empty:
                debug_valueev_source = f"summary:{valueev_summary_csv.name}"
        if not valueev_month_df.empty:
            valueev_rows.append(valueev_month_df)

        model_month_df = pd.DataFrame()
        if model_csv is not None:
            model_month_df = _load_gate_rows(model_csv, month_tag, "btts_model")
            if not model_month_df.empty:
                debug_model_source = f"gated:{model_csv.name}"
        if not valueev_month_df.empty or not model_month_df.empty or month_tag in DEPLOY_MONTHS:
            print(
                f"BTTS_SCAN month={month_tag} "
                f"valueev_source={debug_valueev_source} valueev_rows={len(valueev_month_df)} "
                f"model_source={debug_model_source} model_rows={len(model_month_df)}"
            )
        if not model_month_df.empty:
            model_rows.append(model_month_df)

    if not valueev_rows:
        raise SystemExit("no BTTS valueEV rows found under predictions_output/walk_forward")
    if not model_rows:
        raise SystemExit("no BTTS model rows found under predictions_output/walk_forward")

    valueev_rows = [df for df in valueev_rows if df is not None and not df.empty]
    model_rows = [df for df in model_rows if df is not None and not df.empty]

    if not valueev_rows:
        raise SystemExit("no non-empty BTTS valueEV rows found under predictions_output/walk_forward")
    if not model_rows:
        raise SystemExit("no non-empty BTTS model rows found under predictions_output/walk_forward")

    valueev_df = pd.concat(valueev_rows, axis=0, ignore_index=True) if valueev_rows else pd.DataFrame()
    model_df = pd.concat(model_rows, axis=0, ignore_index=True) if model_rows else pd.DataFrame()
    return valueev_df, model_df



def _winner(a: float, b: float, a_label: str, b_label: str) -> str:
    if pd.isna(a) and pd.isna(b):
        return "tie"
    if pd.isna(a):
        return b_label
    if pd.isna(b):
        return a_label
    if a > b:
        return a_label
    if b > a:
        return b_label
    return "tie"


def _shared_month_count(g: pd.DataFrame) -> int:
    d = g.copy()
    value_mask = pd.to_numeric(d.get("valueev_rows"), errors="coerce").fillna(0).gt(0)
    model_mask = pd.to_numeric(d.get("model_rows"), errors="coerce").fillna(0).gt(0)
    both = d.loc[value_mask & model_mask, "month"].astype(str).nunique()
    return int(both)


def _evidence_state_from_shared(shared_months: int, shared_rows: int) -> str:
    if shared_months >= MIN_SHARED_MONTHS_VALID and shared_rows >= MIN_SHARED_ROWS_VALID:
        return "head_to_head_valid"
    if shared_rows >= MIN_SHARED_ROWS_SPARSE:
        return "sparse_shared_rows"
    return "insufficient_evidence"


# --- BEGIN new helper functions ---

def _head_to_head_call(winner_overall_by_roi: str, evidence_state: str) -> str:
    winner = str(winner_overall_by_roi).strip()
    evidence = str(evidence_state).strip()

    if evidence in {"head_to_head_valid", "sparse_shared_rows"}:
        if winner in {"btts_valueev", "btts_model", "tie"}:
            return winner
        return "no_call"

    return "no_call"


def _deployment_interpretation(
    *,
    winner_overall_by_roi: str,
    evidence_state: str,
    shared_valueev_rows: int,
    shared_model_rows: int,
) -> str:
    winner = str(winner_overall_by_roi).strip()
    evidence = str(evidence_state).strip()
    val_rows = int(shared_valueev_rows)
    model_rows = int(shared_model_rows)

    if evidence == "head_to_head_valid":
        if winner == "btts_model":
            return "model_primary"
        if winner == "btts_valueev":
            return "valueev_primary"
        if winner == "tie":
            return "co_primary"
        return "no_deploy_conclusion"

    if evidence == "sparse_shared_rows":
        if winner == "btts_model":
            return "model_primary_valueev_shadow"
        if winner == "btts_valueev":
            return "model_primary_valueev_watchlist"
        if winner == "tie":
            if model_rows >= val_rows:
                return "model_primary_valueev_shadow"
            return "model_primary_valueev_watchlist"
        if winner == "no_call":
            return "no_deploy_conclusion"
        return "no_deploy_conclusion"

    return "no_deploy_conclusion"


def _automatic_recommendation(
    *,
    deployment_interpretation: str,
    head_to_head_call: str,
    evidence_state: str,
) -> str:
    interp = str(deployment_interpretation).strip()
    h2h = str(head_to_head_call).strip()
    evidence = str(evidence_state).strip()

    if interp == "model_primary":
        return "use model only"
    if interp == "model_primary_valueev_shadow":
        return "use model + shadow valueEV"
    if interp in {"valueev_primary", "model_primary_valueev_watchlist"}:
        return "watch valueEV"
    if interp == "co_primary" and evidence == "head_to_head_valid":
        return "use model + shadow valueEV"
    if h2h in {"btts_valueev", "btts_model", "tie", "no_call"} and interp == "no_deploy_conclusion":
        return "no deploy conclusion"
    return "no deploy conclusion"

# --- BEGIN new helper function ---
def _recommended_live_lane_from_recommendation(automatic_recommendation: str) -> str:
    rec = str(automatic_recommendation).strip()
    if rec == "use model only":
        return "btts_model"
    if rec == "use model + shadow valueEV":
        return "btts_model_shadow_valueev"
    if rec == "watch valueEV":
        return "model_primary_watch_valueev"
    return "no_call"
# --- END new helper function ---

# --- BEGIN new helper functions ---

def _policy_summary_from_recommendation(automatic_recommendation: str) -> str:
    rec = str(automatic_recommendation).strip()
    if rec == "use model only":
        return "Model primary"
    if rec == "use model + shadow valueEV":
        return "Model primary; valueEV shadow only"
    if rec == "watch valueEV":
        return "Model primary; valueEV watchlist"
    return "No deploy conclusion"


def _comparison_vs_policy_status(
    *,
    winner_overall_by_roi: str,
    recommended_live_lane: str,
    evidence_state: str,
) -> str:
    winner = str(winner_overall_by_roi).strip()
    lane = str(recommended_live_lane).strip()
    evidence = str(evidence_state).strip()

    if evidence == "head_to_head_valid":
        if winner == "btts_model" and lane == "btts_model":
            return "model_won_and_deployable"
        if winner == "btts_valueev" and lane != "btts_model":
            return "valueev_won_and_deployable"
        if winner == "tie":
            return "shared_valid_but_tied"
        return "no_clear_edge"

    if evidence == "sparse_shared_rows":
        if winner == "btts_valueev" and lane == "model_primary_watch_valueev":
            return "valueev_won_but_not_deployable"
        if winner == "btts_model" and lane in {"btts_model", "btts_model_shadow_valueev"}:
            return "model_won_but_sparse"
        if winner in {"tie", "no_call"}:
            return "no_clear_edge"
        return "no_clear_edge"

    return "insufficient_shared_evidence"

# --- BEGIN new helper function ---
def _comparison_outcome(winner_overall_by_roi: str) -> str:
    winner = str(winner_overall_by_roi).strip()
    if winner == "btts_valueev":
        return "valueev_outperformed_model"
    if winner == "btts_model":
        return "model_outperformed_valueev"
    if winner == "tie":
        return "tied"
    return "no_clear_edge"

# --- BEGIN new helper function ---
def _policy_footer_text() -> str:
    return (
        "BTTS deployment decision: Model live; ValueEV shadow/watch only; "
        "no ValueEV live promotion at this time."
    )


# --- Policy markdown/csv helpers ---
def _policy_markdown_block() -> list[str]:
    return [
        "## BTTS deployment policy",
        "",
        f"- Primary live lane: `{BTTS_POLICY_PRIMARY_LIVE_LANE}`",
        f"- Secondary lane: `{BTTS_POLICY_SECONDARY_LANE}`",
        f"- Promotion status: `{BTTS_POLICY_PROMOTION_STATUS}`",
        f"- Decision: {_policy_footer_text()}",
        f"- Review trigger: {BTTS_POLICY_REVIEW_TRIGGER}",
        "",
    ]


def _policy_summary_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "policy_area": "BTTS",
                "primary_live_lane": BTTS_POLICY_PRIMARY_LIVE_LANE,
                "secondary_lane": BTTS_POLICY_SECONDARY_LANE,
                "promotion_status": BTTS_POLICY_PROMOTION_STATUS,
                "decision": _policy_footer_text(),
                "review_trigger": BTTS_POLICY_REVIEW_TRIGGER,
            }
        ]
    )


def _write_policy_summary_csv(df: pd.DataFrame) -> None:
    Path("btts_policy_decision_summary.csv").write_text(df.to_csv(index=False), encoding="utf-8")
# --- END new helper function ---


def _winner_priority_for_sort(winner: str) -> int:
    w = str(winner).strip()
    if w == "btts_model":
        return 0
    if w == "btts_valueev":
        return 1
    if w in {"tie", "no_call"}:
        return 2
    return 3


def _evidence_priority_for_sort(evidence_state: str) -> int:
    e = str(evidence_state).strip()
    if e == "head_to_head_valid":
        return 0
    if e == "sparse_shared_rows":
        return 1
    return 2


def _sparse_roi_margin_call(shared_present: pd.DataFrame) -> str:
    if shared_present.empty:
        return "not_comparable"

    valueev_roi = _weighted_mean(shared_present, "valueev_roi", "valueev_rows")
    model_roi = _weighted_mean(shared_present, "model_roi", "model_rows")

    if pd.isna(valueev_roi) or pd.isna(model_roi):
        return "not_comparable"

    margin = abs(float(valueev_roi) - float(model_roi))
    if margin < float(SPARSE_WINNER_ROI_MARGIN_MIN):
        return "no_call"

    return _winner(valueev_roi, model_roi, "btts_valueev", "btts_model")
# --- END new helper functions ---

# --- END new helper functions ---


def main() -> None:
    valueev_df, model_df = _collect_monthly_frames()

    valueev_df["month"] = valueev_df["month"].astype(str)
    model_df["month"] = model_df["month"].astype(str)

    # Restrict to deployment-grade months only.
    valueev_df = valueev_df[valueev_df["month"].isin(DEPLOY_MONTHS)].copy()
    model_df = model_df[model_df["month"].isin(DEPLOY_MONTHS)].copy()

    valueev_months_in_window = sorted(set(valueev_df["month"].astype(str)), key=_month_sort_key) if not valueev_df.empty else []
    model_months_in_window = sorted(set(model_df["month"].astype(str)), key=_month_sort_key) if not model_df.empty else []

    print(
        f"BTTS_DEBUG deployment_window valueev_months={valueev_months_in_window} "
        f"model_months={model_months_in_window}"
    )

    # Restrict to the true shared head-to-head window only.
    shared_months = sorted(
        set(valueev_months_in_window).intersection(set(model_months_in_window)),
        key=_month_sort_key,
    )

    if not shared_months:
        raise SystemExit(
            "no shared BTTS head-to-head months found in deployment window "
            f"| valueev_months={valueev_months_in_window} "
            f"| model_months={model_months_in_window} "
            f"| valueev_rows={len(valueev_df)} "
            f"| model_rows={len(model_df)}"
        )

    valueev_df = valueev_df[valueev_df["month"].astype(str).isin(shared_months)].copy()
    model_df = model_df[model_df["month"].astype(str).isin(shared_months)].copy()

    if valueev_df.empty or model_df.empty:
        raise SystemExit(
            "shared BTTS months were found but one side became empty after filtering "
            f"| shared_months={shared_months} "
            f"| valueev_rows={len(valueev_df)} "
            f"| model_rows={len(model_df)}"
        )

    monthly_key_frames = [
        # Keys come only from the already-shared deployment window.
        valueev_df[["month", "league"]].drop_duplicates(),
        model_df[["month", "league"]].drop_duplicates(),
    ]

    all_keys = pd.MultiIndex.from_frame(
        pd.concat(monthly_key_frames, axis=0, ignore_index=True).drop_duplicates()
    )

    valueev_m = valueev_df.rename(
        columns={
            "rows": "valueev_rows",
            "hit": "valueev_hit",
            "roi": "valueev_roi",
            "avg_odds": "valueev_avg_odds",
        }
    )[["month", "league", "valueev_rows", "valueev_hit", "valueev_roi", "valueev_avg_odds"]]

    model_m = model_df.rename(
        columns={
            "rows": "model_rows",
            "hit": "model_hit",
            "roi": "model_roi",
            "avg_odds": "model_avg_odds",
        }
    )[["month", "league", "model_rows", "model_hit", "model_roi", "model_avg_odds"]]

    monthly = pd.DataFrame(index=all_keys).reset_index()
    monthly = monthly.merge(valueev_m, on=["month", "league"], how="left")
    monthly = monthly.merge(model_m, on=["month", "league"], how="left")
    monthly["league_bucket"] = monthly["league"].map(_league_bucket)

    monthly["valueev_rows"] = pd.to_numeric(monthly["valueev_rows"], errors="coerce")
    monthly["model_rows"] = pd.to_numeric(monthly["model_rows"], errors="coerce")
    monthly["valueev_hit"] = pd.to_numeric(monthly["valueev_hit"], errors="coerce")
    monthly["model_hit"] = pd.to_numeric(monthly["model_hit"], errors="coerce")
    monthly["valueev_roi"] = pd.to_numeric(monthly["valueev_roi"], errors="coerce")
    monthly["model_roi"] = pd.to_numeric(monthly["model_roi"], errors="coerce")
    monthly["valueev_avg_odds"] = pd.to_numeric(monthly["valueev_avg_odds"], errors="coerce")
    monthly["model_avg_odds"] = pd.to_numeric(monthly["model_avg_odds"], errors="coerce")

    monthly["has_valueev"] = monthly["valueev_rows"].fillna(0).gt(0)
    monthly["has_model"] = monthly["model_rows"].fillna(0).gt(0)
    monthly["has_both"] = monthly["has_valueev"] & monthly["has_model"]

    monthly["winner_by_roi"] = monthly.apply(
        lambda r: _winner(r.get("valueev_roi"), r.get("model_roi"), "btts_valueev", "btts_model")
        if bool(r.get("has_both"))
        else "not_comparable",
        axis=1,
    )
    monthly["winner_by_hit"] = monthly.apply(
        lambda r: _winner(r.get("valueev_hit"), r.get("model_hit"), "btts_valueev", "btts_model")
        if bool(r.get("has_both"))
        else "not_comparable",
        axis=1,
    )
    monthly["winner_by_rows"] = monthly.apply(
        lambda r: _winner(r.get("valueev_rows"), r.get("model_rows"), "btts_valueev", "btts_model")
        if bool(r.get("has_both"))
        else "not_comparable",
        axis=1,
    )
    monthly = monthly.sort_values(["month", "league"]).reset_index(drop=True)

    league_rows: list[dict[str, Any]] = []
    winner_rows: list[dict[str, Any]] = []

    for league, g in monthly.groupby("league", dropna=False):
        g = g.sort_values("month").reset_index(drop=True)

        valueev_present = g[g["has_valueev"]].copy()
        model_present = g[g["has_model"]].copy()
        shared_present = g[g["has_both"]].copy()

        valueev_roi = pd.to_numeric(valueev_present["valueev_roi"], errors="coerce")
        model_roi = pd.to_numeric(model_present["model_roi"], errors="coerce")
        valueev_hit = pd.to_numeric(valueev_present["valueev_hit"], errors="coerce")
        model_hit = pd.to_numeric(model_present["model_hit"], errors="coerce")
        valueev_rows_series = pd.to_numeric(valueev_present["valueev_rows"], errors="coerce")
        model_rows_series = pd.to_numeric(model_present["model_rows"], errors="coerce")

        shared_valueev_rows_series = pd.to_numeric(shared_present["valueev_rows"], errors="coerce")
        shared_model_rows_series = pd.to_numeric(shared_present["model_rows"], errors="coerce")
        shared_months_present = _shared_month_count(g)
        shared_valueev_rows = int(shared_valueev_rows_series.fillna(0).sum()) if not shared_present.empty else 0
        shared_model_rows = int(shared_model_rows_series.fillna(0).sum()) if not shared_present.empty else 0
        shared_total_rows = min(shared_valueev_rows, shared_model_rows) if not shared_present.empty else 0

        valueev_best_idx = valueev_roi.idxmax() if not valueev_roi.dropna().empty else None
        valueev_worst_idx = valueev_roi.idxmin() if not valueev_roi.dropna().empty else None
        model_best_idx = model_roi.idxmax() if not model_roi.dropna().empty else None
        model_worst_idx = model_roi.idxmin() if not model_roi.dropna().empty else None

        roi_wins_valueev = int((shared_present["winner_by_roi"] == "btts_valueev").sum()) if not shared_present.empty else 0
        roi_wins_model = int((shared_present["winner_by_roi"] == "btts_model").sum()) if not shared_present.empty else 0

        winner_overall_by_roi_raw = _winner(
            _weighted_mean(shared_present, "valueev_roi", "valueev_rows") if not shared_present.empty else float("nan"),
            _weighted_mean(shared_present, "model_roi", "model_rows") if not shared_present.empty else float("nan"),
            "btts_valueev",
            "btts_model",
        ) if not shared_present.empty else "not_comparable"

        evidence_state = _evidence_state_from_shared(
            int(shared_months_present),
            int(shared_total_rows),
        )

        winner_overall_by_roi_final = winner_overall_by_roi_raw
        if evidence_state == "sparse_shared_rows":
            sparse_call = _sparse_roi_margin_call(shared_present)
            if sparse_call == "no_call":
                winner_overall_by_roi_final = "no_call"

        winner_overall_by_hit_final = _winner(
            _weighted_mean(shared_present, "valueev_hit", "valueev_rows") if not shared_present.empty else float("nan"),
            _weighted_mean(shared_present, "model_hit", "model_rows") if not shared_present.empty else float("nan"),
            "btts_valueev",
            "btts_model",
        ) if not shared_present.empty else "not_comparable"

        if winner_overall_by_roi_final == "no_call":
            winner_overall_by_hit_final = "no_call"

        league_row = {
            "league": str(league),
            "league_bucket": _league_bucket(str(league)),
            "months_present": int(g["month"].nunique()),
            "shared_months_present": int(shared_months_present),
            "shared_total_rows": int(shared_total_rows),
            "shared_valueev_rows": int(shared_valueev_rows),
            "shared_model_rows": int(shared_model_rows),
            "valueev_total_rows": int(valueev_rows_series.fillna(0).sum()),
            "valueev_weighted_hit": _weighted_mean(valueev_present, "valueev_hit", "valueev_rows") if not valueev_present.empty else float("nan"),
            "valueev_weighted_roi": _weighted_mean(valueev_present, "valueev_roi", "valueev_rows") if not valueev_present.empty else float("nan"),
            "valueev_weighted_avg_odds": _weighted_mean(valueev_present, "valueev_avg_odds", "valueev_rows") if not valueev_present.empty else float("nan"),
            "valueev_profitable_months": int((valueev_roi > 0).sum()) if not valueev_present.empty else 0,
            "valueev_worst_month": valueev_present.loc[valueev_worst_idx, "month"] if valueev_worst_idx is not None else "",
            "valueev_worst_roi": float(valueev_present.loc[valueev_worst_idx, "valueev_roi"]) if valueev_worst_idx is not None else float("nan"),
            "valueev_best_month": valueev_present.loc[valueev_best_idx, "month"] if valueev_best_idx is not None else "",
            "valueev_best_roi": float(valueev_present.loc[valueev_best_idx, "valueev_roi"]) if valueev_best_idx is not None else float("nan"),
            "valueev_max_drawdown": _max_drawdown_from_roi(valueev_roi),
            "model_total_rows": int(model_rows_series.fillna(0).sum()),
            "model_weighted_hit": _weighted_mean(model_present, "model_hit", "model_rows") if not model_present.empty else float("nan"),
            "model_weighted_roi": _weighted_mean(model_present, "model_roi", "model_rows") if not model_present.empty else float("nan"),
            "model_weighted_avg_odds": _weighted_mean(model_present, "model_avg_odds", "model_rows") if not model_present.empty else float("nan"),
            "model_profitable_months": int((model_roi > 0).sum()) if not model_present.empty else 0,
            "model_worst_month": model_present.loc[model_worst_idx, "month"] if model_worst_idx is not None else "",
            "model_worst_roi": float(model_present.loc[model_worst_idx, "model_roi"]) if model_worst_idx is not None else float("nan"),
            "model_best_month": model_present.loc[model_best_idx, "month"] if model_best_idx is not None else "",
            "model_best_roi": float(model_present.loc[model_best_idx, "model_roi"]) if model_best_idx is not None else float("nan"),
            "model_max_drawdown": _max_drawdown_from_roi(model_roi),
            "roi_month_wins_valueev": roi_wins_valueev,
            "roi_month_wins_model": roi_wins_model,
            "winner_overall_by_roi": winner_overall_by_roi_final,
            "winner_overall_by_hit": winner_overall_by_hit_final,
        }
        league_rows.append(league_row)

        winner_rows.append(
            {
                "league": str(league),
                "league_bucket": _league_bucket(str(league)),
                "shared_months_present": int(shared_months_present),
                "shared_total_rows": int(shared_total_rows),
                "shared_valueev_rows": int(shared_valueev_rows),
                "shared_model_rows": int(shared_model_rows),
                "winner_overall_by_roi": league_row["winner_overall_by_roi"],
                "winner_overall_by_hit": league_row["winner_overall_by_hit"],
                "roi_month_wins_valueev": roi_wins_valueev,
                "roi_month_wins_model": roi_wins_model,
                "head_to_head_call": _head_to_head_call(
                    league_row["winner_overall_by_roi"],
                    evidence_state,
                ),
                "deployment_interpretation": _deployment_interpretation(
                    winner_overall_by_roi=league_row["winner_overall_by_roi"],
                    evidence_state=evidence_state,
                    shared_valueev_rows=int(shared_valueev_rows),
                    shared_model_rows=int(shared_model_rows),
                ),
                "recommended_live_lane": "pending_recommendation_alignment",
                "evidence_state": evidence_state,
                "comparison_outcome": (
                    "no_clear_edge"
                    if str(league_row["winner_overall_by_roi"]).strip() == "no_call"
                    else _comparison_outcome(league_row["winner_overall_by_roi"])
                ),
            }
        )

    monthly_comparable = monthly[monthly["has_both"]].copy().reset_index(drop=True)

    league_df = pd.DataFrame(league_rows).sort_values(
        ["winner_overall_by_roi", "valueev_weighted_roi", "model_weighted_roi", "league"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)

    winner_df = pd.DataFrame(winner_rows)
    if not winner_df.empty:
        winner_df["automatic_recommendation"] = winner_df.apply(
            lambda r: _automatic_recommendation(
                deployment_interpretation=r.get("deployment_interpretation", ""),
                head_to_head_call=r.get("head_to_head_call", ""),
                evidence_state=r.get("evidence_state", ""),
            ),
            axis=1,
        )
        winner_df["recommended_live_lane"] = winner_df["automatic_recommendation"].apply(
            _recommended_live_lane_from_recommendation
        )
        winner_df["policy_summary"] = winner_df["automatic_recommendation"].apply(
            _policy_summary_from_recommendation
        )
        winner_df["comparison_vs_policy_status"] = winner_df.apply(
            lambda r: _comparison_vs_policy_status(
                winner_overall_by_roi=r.get("winner_overall_by_roi", ""),
                recommended_live_lane=r.get("recommended_live_lane", ""),
                evidence_state=r.get("evidence_state", ""),
            ),
            axis=1,
        )
    winner_df = winner_df[winner_df["shared_total_rows"].fillna(0).astype(int) > 0].copy()
    if not winner_df.empty:
        winner_df["recommended_live_lane_sort"] = winner_df["recommended_live_lane"].map(RECOMMENDED_LIVE_LANE_SORT).fillna(999).astype(int)
        winner_df["evidence_state_sort"] = winner_df["evidence_state"].apply(_evidence_priority_for_sort).astype(int)
        winner_df["winner_overall_sort"] = winner_df["winner_overall_by_roi"].apply(_winner_priority_for_sort).astype(int)
        winner_df = winner_df.sort_values(
            [
                "recommended_live_lane_sort",
                "evidence_state_sort",
                "winner_overall_sort",
                "shared_total_rows",
                "shared_months_present",
                "league",
            ],
            ascending=[True, True, True, False, False, True],
        ).drop(columns=["recommended_live_lane_sort", "evidence_state_sort", "winner_overall_sort"]).reset_index(drop=True)

    policy_df = _policy_summary_table()

    monthly_comparable.to_csv(OUT_MONTHLY, index=False)
    league_df.to_csv(OUT_LEAGUE, index=False)
    winner_df.to_csv(OUT_WINNER, index=False)
    _write_policy_summary_csv(policy_df)

    md_lines: list[str] = []
    md_lines.append("# BTTS Model vs ValueEV Comparison")
    md_lines.append("")
    md_lines.append(f"- Walk-forward root: `{WALK_ROOT}`")
    md_lines.append(f"- Deployment months requested: `{sorted(DEPLOY_MONTHS, key=_month_sort_key)}`")
    md_lines.append(f"- Shared months compared: `{shared_months}`")
    md_lines.append(f"- Months compared: `{monthly_comparable['month'].nunique() if not monthly_comparable.empty else 0}`")
    md_lines.append(f"- League rows: `{len(league_df)}`")
    md_lines.append("")
    md_lines.append("## League head-to-head")
    md_lines.append("")
    md_lines.append(
        _to_markdown(
            league_df[
                [
                    "league",
                    "league_bucket",
                    "months_present",
                    "shared_months_present",
                    "shared_total_rows",
                    "valueev_total_rows",
                    "valueev_weighted_hit",
                    "valueev_weighted_roi",
                    "model_total_rows",
                    "model_weighted_hit",
                    "model_weighted_roi",
                    "winner_overall_by_roi",
                    "winner_overall_by_hit",
                ]
            ]
        )
    )
    md_lines.append("")
    md_lines.append("## Final winner table")
    md_lines.append("")
    md_lines.append(
        _to_markdown(
            winner_df[
                [
                    "league",
                    "league_bucket",
                    "shared_months_present",
                    "shared_total_rows",
                    "shared_valueev_rows",
                    "shared_model_rows",
                    "winner_overall_by_roi",
                    "head_to_head_call",
                    "deployment_interpretation",
                    "automatic_recommendation",
                    "recommended_live_lane",
                    "comparison_outcome",
                    "comparison_vs_policy_status",
                    "policy_summary",
                    "evidence_state",
                ]
            ]
        )
    )
    md_lines.extend(_policy_markdown_block())
    md_lines.append("## Monthly head-to-head")
    md_lines.append("")
    md_lines.append(
        _to_markdown(
            monthly_comparable[
                [
                    "month",
                    "league",
                    "league_bucket",
                    "valueev_rows",
                    "valueev_hit",
                    "valueev_roi",
                    "model_rows",
                    "model_hit",
                    "model_roi",
                    "winner_by_roi",
                ]
            ]
        )
    )
    md_lines.append("")

    OUT_MD.write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")

    print("MONTHLY HEAD-TO-HEAD")
    print(monthly_comparable.to_string(index=False))
    print("\nLEAGUE HEAD-TO-HEAD")
    print(league_df.to_string(index=False))
    print("\nFINAL WINNER TABLE")
    print(winner_df.to_string(index=False))
    print("\nBTTS POLICY DECISION")
    print(policy_df.to_string(index=False))
    print(f"\nWROTE: {OUT_MONTHLY}")
    print(f"WROTE: {OUT_LEAGUE}")
    print(f"WROTE: {OUT_WINNER}")
    print("WROTE: btts_policy_decision_summary.csv")
    print(f"WROTE: {OUT_MD}")


if __name__ == "__main__":
    main()