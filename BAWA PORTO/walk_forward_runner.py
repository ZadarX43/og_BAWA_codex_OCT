#!/usr/bin/env python3
"""walk_forward_runner.py

Monthly walk-forward runner + summary generator.

Default behavior per test month:
  1) Run bookie_allmarkets.py for the test window
  2) Run backtest_deploy_csv.py to join truth
  3) Write walk_forward_summary.csv (league + market metrics)
  4) Write FTR "product tables" (ValueEV vs 68% Accuracy) per month (out-of-sample, using locked ValueEV gates)

Outputs per month:
  predictions_output/walk_forward/YYYY-MM/
    - BOOKIE_IMP*_ALLMARKETS_<from>_to_<to>.csv
    - ...__BACKTEST.csv
    - ...__BACKTEST_SUMMARY.csv
    - walk_forward_summary.csv
    - investor_table_ftr_valueEV_vs_accuracy68.csv
    - investor_table_ftr_valueEV_vs_accuracy68.meta.json
    - walk_forward_metadata.json

Notes:
  - Training is assumed to be handled separately (models already in ModelStore).
  - Minimum sample size can be overridden by league/market via a CSV.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import numpy as np


@dataclass
class Window:
    test_from: date
    test_to: date
    train_from: date
    train_to: date


def _month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def _month_end(d: date) -> date:
    if d.month == 12:
        return date(d.year, 12, 31)
    nxt = date(d.year, d.month + 1, 1)
    return nxt - timedelta(days=1)


def _add_months(d: date, months: int) -> date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    day = min(d.day, _month_end(date(y, m, 1)).day)
    return date(y, m, day)


def _parse_month(s: str) -> date:
    dt = datetime.strptime(s, "%Y-%m").date()
    return date(dt.year, dt.month, 1)


def _month_range(start_month: str, end_month: str) -> List[date]:
    start = _parse_month(start_month)
    end = _parse_month(end_month)
    months = []
    cur = start
    while cur <= end:
        months.append(cur)
        cur = _add_months(cur, 1)
    return months



def _run_cmd(cmd: List[str], dry_run: bool) -> None:
    print(">>>", " ".join(shlex.quote(x) for x in cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


# Helper: format a command template with placeholders and split to argv
def _format_cmd_template(template: str, **kwargs) -> List[str]:
    """Format a command template with placeholders and split to argv.

    Supported placeholders (recommended):
      {train_from}, {train_to}, {test_from}, {test_to},
      {matches_root}, {modelstore}, {leagues}, {run_dir}, {month}

    Notes:
      - The template is formatted with `str.format(**kwargs)`.
      - The result is split with `shlex.split` for safe argv handling.
    """
    tmpl = str(template or "").strip()
    if not tmpl:
        raise SystemExit("--train-cmd is required when --train-then-predict is set")
    try:
        cmd_str = tmpl.format(**kwargs)
    except KeyError as e:
        raise SystemExit(f"train-cmd missing placeholder value: {e}")
    return shlex.split(cmd_str)


def _find_allmarkets(run_dir: Path, date_from: str, date_to: str) -> Optional[Path]:
    patt = f"BOOKIE_IMP*_ALLMARKETS_{date_from}_to_{date_to}.csv"
    matches = sorted(run_dir.glob(patt))
    if not matches:
        return None
    return matches[-1]


def _load_min_n_overrides(p: Optional[Path]) -> Dict[Tuple[str, str], int]:
    if not p:
        return {}
    if not p.exists():
        raise SystemExit(f"min-n overrides file not found: {p}")
    df = pd.read_csv(p)
    required = {"league", "market", "min_n"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"min-n overrides missing columns: {sorted(missing)}")
    out: Dict[Tuple[str, str], int] = {}
    for _, r in df.iterrows():
        league = str(r["league"]).strip().lower()
        market = str(r["market"]).strip().lower()
        try:
            mn = int(r["min_n"])
        except Exception:
            continue
        out[(league, market)] = mn
    return out


def _compute_summary(
    backtest_csv: Path,
    min_n_default: int,
    min_n_overrides: Dict[Tuple[str, str], int],
    cov_min: float,
    cov_max: float,
) -> pd.DataFrame:
    df = pd.read_csv(backtest_csv, low_memory=False)

    for c in ["league", "market", "fixture_key", "correct", "bookie_od"]:
        if c not in df.columns:
            raise SystemExit(f"BACKTEST missing required column: {c}")

    df["league"] = df["league"].astype(str)
    df["market"] = df["market"].astype(str).str.lower().str.strip()

    scored = df[df["correct"].notna()].copy()
    roi_df = scored[scored["bookie_od"].notna()].copy()

    fixtures = (
        scored.groupby("league", dropna=False)["fixture_key"]
        .nunique()
        .rename("n_fixtures")
        .to_dict()
    )

    rows: List[Dict[str, object]] = []

    for (league, market), g_hit in scored.groupby(["league", "market"], dropna=False):
        g_roi = roi_df[(roi_df["league"] == league) & (roi_df["market"] == market)]
        n = int(len(g_hit))
        hit = float(g_hit["correct"].mean()) if n else float("nan")
        n_roi = int(len(g_roi))
        avg_od = float(g_roi["bookie_od"].mean()) if n_roi else float("nan")
        roi = float((g_roi["correct"] * g_roi["bookie_od"] - 1).mean()) if n_roi else float("nan")

        n_fix = int(fixtures.get(league, 0) or 0)
        coverage = (n / n_fix) if n_fix else float("nan")

        breakeven = (1.0 / avg_od) if avg_od and avg_od > 0 else float("nan")
        edge = (hit - breakeven) if pd.notna(hit) and pd.notna(breakeven) else float("nan")

        key = (str(league).strip().lower(), str(market).strip().lower())
        min_n_req = int(min_n_overrides.get(key, min_n_default))

        pass_roi = pd.notna(roi) and roi > 0
        pass_hit = pd.notna(hit) and pd.notna(breakeven) and hit > breakeven
        pass_n = n >= min_n_req
        pass_cov = pd.notna(coverage) and (coverage >= cov_min) and (coverage <= cov_max)
        pass_all = bool(pass_roi and pass_hit and pass_n and pass_cov)

        rows.append(
            {
                "league": league,
                "market": market,
                "n": n,
                "hit": hit,
                "avg_od": avg_od,
                "roi": roi,
                "breakeven_hit": breakeven,
                "edge_vs_be": edge,
                "n_fixtures": n_fix,
                "coverage": coverage,
                "min_n_required": min_n_req,
                "pass_roi": pass_roi,
                "pass_hit": pass_hit,
                "pass_n": pass_n,
                "pass_cov": pass_cov,
                "pass_all": pass_all,
            }
        )

    out = pd.DataFrame(rows).sort_values(["league", "market"])
    return out


# === FTR Product Table Helpers ===

def _safe_to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _pick_prob_from_heads(df: pd.DataFrame, *, mode: str) -> pd.Series:
    """Return per-row probability corresponding to df['bookie_pick'].

    mode:
      - 'confidence': uses confidence_home/draw/away
      - 'pois_norm': uses p_home_pois/p_draw_pois/p_away_pois normalised to sum=1
    """
    pick = df["bookie_pick"].astype(str).str.upper().str.strip()

    if mode == "confidence":
        ph = _safe_to_num(df.get("confidence_home"))
        pdw = _safe_to_num(df.get("confidence_draw"))
        pa = _safe_to_num(df.get("confidence_away"))
    else:
        ph = _safe_to_num(df.get("p_home_pois"))
        pdw = _safe_to_num(df.get("p_draw_pois"))
        pa = _safe_to_num(df.get("p_away_pois"))
        s = (ph + pdw + pa)
        s = s.where(s > 0, np.nan)
        ph = ph / s
        pdw = pdw / s
        pa = pa / s

    out = pd.Series(np.nan, index=df.index, dtype="float64")
    out.loc[pick.eq("HOME")] = ph.loc[pick.eq("HOME")]
    out.loc[pick.eq("DRAW")] = pdw.loc[pick.eq("DRAW")]
    out.loc[pick.eq("AWAY")] = pa.loc[pick.eq("AWAY")]
    return out


def _gate_ftr_valueev(
    df_ftr: pd.DataFrame,
    *,
    od_min: float,
    edge_min: float,
    p_mode: str,
) -> pd.DataFrame:
    """ValueEV gate: bookie_od >= od_min AND (bookie_od * p_bookie_pick) >= edge_min."""
    g = df_ftr.copy()
    g["bookie_od"] = _safe_to_num(g["bookie_od"])
    p = _pick_prob_from_heads(g, mode=p_mode)
    g["p_bookie_pick"] = _safe_to_num(p)
    g["edge"] = g["bookie_od"] * g["p_bookie_pick"]
    m = g["bookie_od"].ge(float(od_min)) & g["edge"].ge(float(edge_min))
    return g.loc[m].copy()



def _gate_ftr_accuracy68(
    df_ftr: pd.DataFrame,
    *,
    imp_min: float,
    top_q: Optional[float],
    top_q_scope: str,
    home_away_only: bool,
) -> pd.DataFrame:
    """Accuracy gate: bookie implied >= imp_min => bookie_od <= 1/imp_min.

    Optionally applies a score top-quantile gate. If top_q_scope is 'per_league',
    the quantile is computed per league to prevent league starvation.
    """
    g = df_ftr.copy()
    g["bookie_od"] = _safe_to_num(g["bookie_od"])

    max_od = 1.0 / float(imp_min)
    m = g["bookie_od"].le(max_od)

    if home_away_only:
        pick = g["bookie_pick"].astype(str).str.upper().str.strip()
        m = m & pick.isin(["HOME", "AWAY"])

    if top_q is not None:
        if "score" in g.columns:
            score = _safe_to_num(g["score"])
            if top_q_scope == "per_league":
                thr = (
                    g.assign(_score=score)
                    .groupby("league", dropna=False)["_score"]
                    .quantile(float(top_q))
                )
                m = m & score.ge(g["league"].map(thr))
            else:
                thr = float(score.dropna().quantile(float(top_q))) if score.notna().any() else None
                if thr is not None:
                    m = m & score.ge(thr)

    return g.loc[m].copy()


# --- OU25/BTTS product gates ---

def _gate_ou25_valueev(
    df_ou25: pd.DataFrame,
    *,
    od_min: float,
    edge_min: float,
) -> pd.DataFrame:
    """OU25 ValueEV gate using no-vig probabilities.

    Expects:
      - bookie_pick in {OVER25, UNDER25}
      - bookie_od populated
      - p_over25_novig, p_under25_novig populated (best-effort)

    Gate:
      bookie_od >= od_min AND (bookie_od * p_pick) >= edge_min
    """
    g = df_ou25.copy()
    g["bookie_od"] = _safe_to_num(g["bookie_od"])

    pick = g["bookie_pick"].astype(str).str.upper().str.strip()
    p_over = _safe_to_num(g.get("p_over25_novig"))
    p_under = _safe_to_num(g.get("p_under25_novig"))

    p_pick = pd.Series(np.nan, index=g.index, dtype="float64")
    p_pick.loc[pick.eq("OVER25")] = p_over.loc[pick.eq("OVER25")]
    p_pick.loc[pick.eq("UNDER25")] = p_under.loc[pick.eq("UNDER25")]

    g["p_pick"] = _safe_to_num(p_pick)
    g["edge"] = g["bookie_od"] * g["p_pick"]

    m = g["bookie_od"].ge(float(od_min)) & g["edge"].ge(float(edge_min))
    return g.loc[m].copy()


def _gate_btts_valueev(
    df_btts: pd.DataFrame,
    *,
    od_min: float,
    edge_min: float,
) -> pd.DataFrame:
    """BTTS ValueEV gate using prob_btts.

    Expects:
      - bookie_pick in {YES, NO}
      - bookie_od populated
      - prob_btts populated (YES probability)

    p_pick = prob_btts if YES else (1 - prob_btts)

    Gate:
      bookie_od >= od_min AND (bookie_od * p_pick) >= edge_min

    Also prints a forensic stage audit so we can see where BTTS rows are being
    filtered out.
    """
    g = df_btts.copy()

    if g.empty:
        print(
            f"BTTS_VALUEEV_AUDIT total_rows=0 valid_bookie_od=0 valid_prob_btts=0 "
            f"valid_yes_no_pick=0 yes_rows=0 no_rows=0 other_pick_rows=0 "
            f"rows_pass_od_min=0 rows_pass_edge_min=0 rows_pass_both=0 "
            f"rows_missing_prob_btts=0 rows_bad_pick_norm=0 rows_edge_nan=0 "
            f"od_min={float(od_min):.2f} edge_min={float(edge_min):.2f}"
        )
        return g

    g["bookie_od"] = _safe_to_num(g.get("bookie_od"))
    g["prob_btts"] = _safe_to_num(g.get("prob_btts"))

    raw_pick = g.get("bookie_pick")
    if raw_pick is None:
        pick = pd.Series("", index=g.index, dtype="object")
    else:
        pick = raw_pick.astype(str).str.upper().str.strip()

    pick_norm = pick.replace({
        "BTTS_YES": "YES",
        "BTTS_NO": "NO",
        "Y": "YES",
        "N": "NO",
    })

    p_yes = g["prob_btts"]

    p_pick = pd.Series(np.nan, index=g.index, dtype="float64")
    p_pick.loc[pick_norm.eq("YES")] = p_yes.loc[pick_norm.eq("YES")]
    p_pick.loc[pick_norm.eq("NO")] = (1.0 - p_yes).loc[pick_norm.eq("NO")]

    g["bookie_pick_norm"] = pick_norm
    g["p_pick"] = _safe_to_num(p_pick)
    g["edge"] = g["bookie_od"] * g["p_pick"]

    valid_od = g["bookie_od"].notna()
    valid_prob = g["prob_btts"].notna()
    valid_pick = g["bookie_pick_norm"].isin(["YES", "NO"])
    yes_rows = int(g["bookie_pick_norm"].eq("YES").sum())
    no_rows = int(g["bookie_pick_norm"].eq("NO").sum())
    other_pick_rows = int((~g["bookie_pick_norm"].isin(["YES", "NO"])).sum())

    pass_od = valid_od & g["bookie_od"].ge(float(od_min))
    pass_edge = g["edge"].notna() & g["edge"].ge(float(edge_min))
    pass_both = pass_od & pass_edge

    rows_missing_prob_btts = int((~valid_prob).sum())
    rows_bad_pick_norm = int((~valid_pick).sum())
    rows_edge_nan = int(g["edge"].isna().sum())

    print(
        "BTTS_VALUEEV_AUDIT "
        f"total_rows={int(len(g))} "
        f"valid_bookie_od={int(valid_od.sum())} "
        f"valid_prob_btts={int(valid_prob.sum())} "
        f"valid_yes_no_pick={int(valid_pick.sum())} "
        f"yes_rows={yes_rows} "
        f"no_rows={no_rows} "
        f"other_pick_rows={other_pick_rows} "
        f"rows_pass_od_min={int(pass_od.sum())} "
        f"rows_pass_edge_min={int(pass_edge.sum())} "
        f"rows_pass_both={int(pass_both.sum())} "
        f"rows_missing_prob_btts={rows_missing_prob_btts} "
        f"rows_bad_pick_norm={rows_bad_pick_norm} "
        f"rows_edge_nan={rows_edge_nan} "
        f"od_min={float(od_min):.2f} "
        f"edge_min={float(edge_min):.2f}"
    )

    return g.loc[pass_both].copy()



# === Dedicated model lanes for OU25/BTTS (threshold-driven, per league) ===

def _build_binary_model_lane(
    df_market: pd.DataFrame,
    *,
    prob_col: str,
    positive_pick: str,
    negative_pick: str,
    threshold: float,
    lane_name: str,
) -> pd.DataFrame:
    """Build a dedicated binary-model lane from scored backtest rows.

    The backtest already contains one row per selectable pick, so we only keep the
    side implied by the model threshold:
      - if prob >= threshold  -> keep positive_pick rows
      - else                  -> keep negative_pick rows

    Returned frame includes audit columns so downstream comparison scripts can
    identify the lane provenance.
    """
    g = df_market.copy()
    if g.empty:
        return g

    if prob_col not in g.columns:
        return g.iloc[0:0].copy()

    g["model_prob"] = _safe_to_num(g[prob_col])
    g["bookie_pick_norm"] = g["bookie_pick"].astype(str).str.upper().str.strip()

    pos = str(positive_pick).upper().strip()
    neg = str(negative_pick).upper().strip()
    thr = float(threshold)

    want_pick = np.where(g["model_prob"].ge(thr), pos, neg)
    g["model_threshold"] = thr
    g["model_lane"] = str(lane_name)
    g["model_pick_from_threshold"] = want_pick

    out = g.loc[g["bookie_pick_norm"].eq(g["model_pick_from_threshold"])].copy()
    return out


def _load_market_thresholds_for_league(modelstore_root: Path, league_name: str) -> Dict[str, float]:
    """Load per-league market thresholds from ModelStore/<tag>/market_thresholds.json."""
    tag = str(league_name).replace("/", "_").replace(" ", "_")
    path = modelstore_root / tag / "market_thresholds.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: Dict[str, float] = {}
    for k, v in payload.items():
        try:
            out[str(k).strip().lower()] = float(v)
        except Exception:
            continue
    return out


def _build_dedicated_over25_lane(df_ou25: pd.DataFrame, *, modelstore_root: Path) -> pd.DataFrame:
    """Build dedicated over25/under25 model lane using per-league thresholds.

    Prefers explicit over25 probability columns when present, else falls back to
    p_over25_novig.
    """
    if df_ou25.empty:
        return df_ou25.copy()

    prob_candidates = ["over25_prob", "p_over25", "model_prob_over25", "p_over25_novig"]
    prob_col = next((c for c in prob_candidates if c in df_ou25.columns), None)
    if prob_col is None:
        return df_ou25.iloc[0:0].copy()

    chunks: List[pd.DataFrame] = []
    for league, g in df_ou25.groupby("league", dropna=False):
        thr_map = _load_market_thresholds_for_league(modelstore_root, str(league))
        thr = thr_map.get("over25")
        if thr is None:
            continue
        lane = _build_binary_model_lane(
            g,
            prob_col=prob_col,
            positive_pick="OVER25",
            negative_pick="UNDER25",
            threshold=float(thr),
            lane_name="dedicated_over25_model",
        )
        if not lane.empty:
            lane["source_prob_col"] = prob_col
            chunks.append(lane)

    if not chunks:
        return df_ou25.iloc[0:0].copy()
    return pd.concat(chunks, axis=0, ignore_index=True)

# IMPORTANT BTTS RUNTIME POLICY
# -----------------------------
# BTTS live deployment is frozen to the dedicated BTTS model lane.
# BTTS valueEV is research/shadow/watch only and must not be promoted
# to live routing unless a future comparison review explicitly changes policy.
#
# Current policy:
#   live  -> BTTS_MODEL
#   shadow/watch -> BTTS_VALUEEV
#   valueEV live enabled -> False
def _build_dedicated_btts_lane(df_btts: pd.DataFrame, *, modelstore_root: Path) -> pd.DataFrame:
    """Build dedicated BTTS yes/no model lane using per-league thresholds."""
    if df_btts.empty:
        return df_btts.copy()

    prob_candidates = ["prob_btts", "btts_prob", "model_prob_btts", "p_btts"]
    prob_col = next((c for c in prob_candidates if c in df_btts.columns), None)
    if prob_col is None:
        return df_btts.iloc[0:0].copy()

    chunks: List[pd.DataFrame] = []
    for league, g in df_btts.groupby("league", dropna=False):
        thr_map = _load_market_thresholds_for_league(modelstore_root, str(league))
        thr_yes = thr_map.get("btts")
        if thr_yes is None:
            continue
        lane = _build_binary_model_lane(
            g,
            prob_col=prob_col,
            positive_pick="YES",
            negative_pick="NO",
            threshold=float(thr_yes),
            lane_name="dedicated_btts_model",
        )
        if not lane.empty:
            lane["source_prob_col"] = prob_col
            chunks.append(lane)

    if not chunks:
        return df_btts.iloc[0:0].copy()
    return pd.concat(chunks, axis=0, ignore_index=True)

def _agg_league_table(df: pd.DataFrame, *, label: str) -> pd.DataFrame:
    """Aggregate per-league hit/ROI/avg_od for an already-scored gated frame."""
    if df.empty:
        return pd.DataFrame(columns=["league", f"n_{label}", f"hit_{label}", f"roi_{label}", f"avg_od_{label}"])

    d = df.copy()
    d["bookie_od"] = _safe_to_num(d["bookie_od"])
    d["correct"] = _safe_to_num(d["correct"])

    g = d.groupby("league", dropna=False)
    out = g.agg(
        n=("correct", "size"),
        hit=("correct", "mean"),
        avg_od=("bookie_od", "mean"),
    ).reset_index()

    out["roi"] = g.apply(lambda x: float((x["correct"] * x["bookie_od"] - 1).mean())).values

    out = out.rename(
        columns={
            "n": f"n_{label}",
            "hit": f"hit_{label}",
            "roi": f"roi_{label}",
            "avg_od": f"avg_od_{label}",
        }
    )
    return out


def _agg_league_table_simple(df: pd.DataFrame, *, label: str) -> pd.DataFrame:
    """Aggregate per-league hit/ROI/avg_od for a scored gated frame (generic markets).

    Output columns:
      league, n_<label>, hit_<label>, roi_<label>, avg_od_<label>
    """
    if df.empty:
        return pd.DataFrame(columns=["league", f"n_{label}", f"hit_{label}", f"roi_{label}", f"avg_od_{label}"])

    d = df.copy()
    d["bookie_od"] = _safe_to_num(d.get("bookie_od"))
    d["correct"] = _safe_to_num(d.get("correct"))

    g = d.groupby("league", dropna=False)
    out = g.agg(
        n=("correct", "size"),
        hit=("correct", "mean"),
        avg_od=("bookie_od", "mean"),
    ).reset_index()

    out["roi"] = g.apply(lambda x: float((x["correct"] * x["bookie_od"] - 1).mean())).values

    out = out.rename(
        columns={
            "n": f"n_{label}",
            "hit": f"hit_{label}",
            "roi": f"roi_{label}",
            "avg_od": f"avg_od_{label}",
        }
    )
    return out

# === BTTS policy freeze helpers ===

BTTS_LIVE_POLICY = {
    "primary_live_lane": "BTTS_MODEL",
    "secondary_lane": "BTTS_VALUEEV",
    "secondary_mode": "shadow_watch_only",
    "valueev_live_enabled": False,
    "promotion_condition": (
        "Only revisit BTTS valueEV live promotion when shared months increase, "
        "shared rows increase, and valueEV continues to beat model with a wider ROI margin."
    ),
    "policy_footer": (
        "BTTS deployment decision: Model live; ValueEV shadow/watch only; "
        "no ValueEV live promotion at this time."
    ),
}


def _build_btts_policy_decision_df(
    *,
    btts_model_lane: pd.DataFrame,
    btts_valueev_lane: pd.DataFrame,
    month_tag: str,
) -> pd.DataFrame:
    """
    Build a tiny monthly BTTS policy decision table.

    This is intentionally policy-first, not a comparison engine.
    It records the frozen deployment stance so runtime/deploy artefacts
    cannot accidentally imply BTTS valueEV is promoted live.
    """
    model_rows = int(len(btts_model_lane)) if btts_model_lane is not None else 0
    valueev_rows = int(len(btts_valueev_lane)) if btts_valueev_lane is not None else 0

    model_hit = (
        float(pd.to_numeric(btts_model_lane.get("correct"), errors="coerce").mean())
        if model_rows > 0 else float("nan")
    )
    model_roi = (
        float(
            (
                pd.to_numeric(btts_model_lane.get("correct"), errors="coerce")
                * pd.to_numeric(btts_model_lane.get("bookie_od"), errors="coerce")
                - 1
            ).mean()
        )
        if model_rows > 0 else float("nan")
    )

    valueev_hit = (
        float(pd.to_numeric(btts_valueev_lane.get("correct"), errors="coerce").mean())
        if valueev_rows > 0 else float("nan")
    )
    valueev_roi = (
        float(
            (
                pd.to_numeric(btts_valueev_lane.get("correct"), errors="coerce")
                * pd.to_numeric(btts_valueev_lane.get("bookie_od"), errors="coerce")
                - 1
            ).mean()
        )
        if valueev_rows > 0 else float("nan")
    )

    rows = [
        {
            "month": str(month_tag),
            "market": "btts",
            "policy_state": "frozen",
            "primary_live_lane": BTTS_LIVE_POLICY["primary_live_lane"],
            "secondary_lane": BTTS_LIVE_POLICY["secondary_lane"],
            "secondary_mode": BTTS_LIVE_POLICY["secondary_mode"],
            "valueev_live_enabled": bool(BTTS_LIVE_POLICY["valueev_live_enabled"]),
            "live_route_target": "BTTS_MODEL",
            "shadow_route_target": "BTTS_VALUEEV",
            "btts_model_rows": model_rows,
            "btts_model_hit": model_hit,
            "btts_model_roi": model_roi,
            "btts_valueev_rows": valueev_rows,
            "btts_valueev_hit": valueev_hit,
            "btts_valueev_roi": valueev_roi,
            "promotion_condition": BTTS_LIVE_POLICY["promotion_condition"],
            "policy_footer": BTTS_LIVE_POLICY["policy_footer"],
        }
    ]
    return pd.DataFrame(rows)


def _write_btts_policy_decision_artifacts(
    *,
    out_dir: Path,
    month_tag: str,
    btts_model_lane: pd.DataFrame,
    btts_valueev_lane: pd.DataFrame,
) -> tuple[Path, Path]:
    """
    Write monthly BTTS policy freeze artefacts.

    Outputs:
      - BTTS_POLICY_DECISION.csv
      - BTTS_POLICY_DECISION.md
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    policy_df = _build_btts_policy_decision_df(
        btts_model_lane=btts_model_lane,
        btts_valueev_lane=btts_valueev_lane,
        month_tag=month_tag,
    )

    csv_path = out_dir / "BTTS_POLICY_DECISION.csv"
    md_path = out_dir / "BTTS_POLICY_DECISION.md"

    policy_df.to_csv(csv_path, index=False)

    row = policy_df.iloc[0].to_dict()
    md_lines = [
        "# BTTS Policy Decision",
        "",
        f"- Month: `{row['month']}`",
        f"- Policy state: `{row['policy_state']}`",
        f"- Primary live lane: `{row['primary_live_lane']}`",
        f"- Secondary lane: `{row['secondary_lane']}`",
        f"- Secondary mode: `{row['secondary_mode']}`",
        f"- ValueEV live enabled: `{row['valueev_live_enabled']}`",
        "",
        "## Monthly lane snapshot",
        "",
        f"- BTTS model rows: `{row['btts_model_rows']}`",
        f"- BTTS model hit: `{row['btts_model_hit']}`",
        f"- BTTS model ROI: `{row['btts_model_roi']}`",
        f"- BTTS valueEV rows: `{row['btts_valueev_rows']}`",
        f"- BTTS valueEV hit: `{row['btts_valueev_hit']}`",
        f"- BTTS valueEV ROI: `{row['btts_valueev_roi']}`",
        "",
        "## Deployment policy",
        "",
        f"- Live route target: `{row['live_route_target']}`",
        f"- Shadow route target: `{row['shadow_route_target']}`",
        f"- Promotion condition: {row['promotion_condition']}",
        "",
        f"- {row['policy_footer']}",
        "",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return csv_path, md_path


def _write_ftr_product_tables(
    backtest_csv: Path,
    out_dir: Path,
    *,
    month_tag: str,
    modelstore_root: Path,
    valueev_od_min: float,
    valueev_edge_min: float,
    valueev_p_mode: str,
    acc_imp_min: float,
    acc_top_q: Optional[float],
    acc_top_q_scope: str,
    acc_home_away_only: bool,
    ou25_od_min: float,
    ou25_edge_min: float,
    btts_od_min: float,
    btts_edge_min: float,
    min_n: int,
) -> Path:
    """Build and write per-month investor table: ValueEV vs Accuracy68 (FTR only, but also writes OU25/BTTS product tables)."""
    df = pd.read_csv(backtest_csv, low_memory=False)

    # Use scored-only rows (backtest should already be clean, but keep this defensive)
    df = df[df.get("correct").notna()].copy()

    # Focus FTR
    df["market"] = df["market"].astype(str).str.lower().str.strip()
    ftr = df[df["market"].eq("ftr")].copy()
    ou25 = df[df["market"].eq("ou25")].copy()
    btts = df[df["market"].eq("btts")].copy()

    over25_model_lane = _build_dedicated_over25_lane(ou25, modelstore_root=modelstore_root)
    btts_model_lane = _build_dedicated_btts_lane(btts, modelstore_root=modelstore_root)

    # Define headers for both output tables
    headers = [
        "league","n_val","hit_val","avg_od_val","roi_val",
        "n_acc","hit_acc","avg_od_acc","roi_acc"
    ]

    outp = out_dir / "investor_table_ftr_valueEV_vs_accuracy68.csv"
    outp_pres = out_dir / f"investor_table_ftr_valueEV_vs_accuracy68_PRESENTATION_MINN{min_n}.csv"

    if ftr.empty:
        # Write empty CSVs with headers for both files
        pd.DataFrame(columns=headers).to_csv(outp, index=False)
        pd.DataFrame(columns=headers).to_csv(outp_pres, index=False)
        # Still write meta JSON as before
        meta = {
            "valueev": {"od_min": valueev_od_min, "edge_min": valueev_edge_min, "p_mode": valueev_p_mode},
            "accuracy68": {"imp_min": acc_imp_min, "top_q": acc_top_q, "top_q_scope": acc_top_q_scope, "home_away_only": acc_home_away_only},
            "ou25_valueev": {"od_min": float(ou25_od_min), "edge_min": float(ou25_edge_min)},
            "btts_valueev": {"od_min": float(btts_od_min), "edge_min": float(btts_edge_min)},
            "min_n_valueev": int(min_n),
            "source_backtest": str(backtest_csv),
        }
        (out_dir / "investor_table_ftr_valueEV_vs_accuracy68.meta.json").write_text(json.dumps(meta, indent=2))
        return outp

    # Gates
    val = _gate_ftr_valueev(ftr, od_min=valueev_od_min, edge_min=valueev_edge_min, p_mode=valueev_p_mode)
    acc = _gate_ftr_accuracy68(
        ftr,
        imp_min=acc_imp_min,
        top_q=acc_top_q,
        top_q_scope=acc_top_q_scope,
        home_away_only=acc_home_away_only,
    )
    ou25_val = _gate_ou25_valueev(ou25, od_min=float(ou25_od_min), edge_min=float(ou25_edge_min))
    btts_val = _gate_btts_valueev(btts, od_min=float(btts_od_min), edge_min=float(btts_edge_min))

    # Write per-month gated row CSVs (forensic truth for ROI audits)
    val_out = out_dir / "FTR_VALUEEV_GATED_ROWS.csv"
    acc_out = out_dir / "FTR_ACCURACY68_GATED_ROWS.csv"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        val.to_csv(val_out, index=False)
        acc.to_csv(acc_out, index=False)
        print(f"WROTE: {val_out}")
        print(f"WROTE: {acc_out}")
    except Exception as e:
        print("❌ FAILED to write gated rows:", e)
        raise

    # Also write slim versions (presentation / quick inspection)
    slim_cols = [
        "league",
        "market",
        "fixture_key",
        "match_date",
        "home_team_name",
        "away_team_name",
        "bookie_pick",
        "bookie_od",
        "correct",
        "score",
        "p_home_pois",
        "p_draw_pois",
        "p_away_pois",
        "confidence_home",
        "confidence_draw",
        "confidence_away",
        "p_bookie_pick",
        "edge",
    ]
    val_slim = val[[c for c in slim_cols if c in val.columns]].copy()
    acc_slim = acc[[c for c in slim_cols if c in acc.columns]].copy()
    val_slim_out = out_dir / "FTR_VALUEEV_GATED_ROWS_SLIM.csv"
    acc_slim_out = out_dir / "FTR_ACCURACY68_GATED_ROWS_SLIM.csv"
    try:
        val_slim.to_csv(val_slim_out, index=False)
        acc_slim.to_csv(acc_slim_out, index=False)
        print(f"WROTE: {val_slim_out}")
        print(f"WROTE: {acc_slim_out}")
    except Exception as e:
        print("❌ FAILED to write slim gated rows:", e)
        raise

    # OU25/BTTS gated row CSVs (forensic)
    ou25_out = out_dir / "OU25_VALUEEV_GATED_ROWS.csv"
    btts_out = out_dir / "BTTS_VALUEEV_GATED_ROWS.csv"
    ou25_val.to_csv(ou25_out, index=False)
    btts_val.to_csv(btts_out, index=False)
    print(f"WROTE: {ou25_out}")
    print(f"WROTE: {btts_out}")

    # Slim versions (quick inspection)
    slim_cols_m = [
        "league",
        "market",
        "fixture_key",
        "match_date",
        "home_team_name",
        "away_team_name",
        "bookie_pick",
        "bookie_od",
        "correct",
        "score",
        "p_over25_novig",
        "p_under25_novig",
        "prob_btts",
        "p_pick",
        "edge",
    ]

    ou25_slim = ou25_val[[c for c in slim_cols_m if c in ou25_val.columns]].copy()
    btts_slim = btts_val[[c for c in slim_cols_m if c in btts_val.columns]].copy()

    ou25_slim_out = out_dir / "OU25_VALUEEV_GATED_ROWS_SLIM.csv"
    btts_slim_out = out_dir / "BTTS_VALUEEV_GATED_ROWS_SLIM.csv"
    ou25_slim.to_csv(ou25_slim_out, index=False)
    btts_slim.to_csv(btts_slim_out, index=False)
    print(f"WROTE: {ou25_slim_out}")
    print(f"WROTE: {btts_slim_out}")

    # Dedicated model lanes (threshold-driven, per league)
    over25_model_out = out_dir / "OVER25_MODEL_GATED_ROWS.csv"
    btts_model_out = out_dir / "BTTS_MODEL_GATED_ROWS.csv"
    over25_model_lane.to_csv(over25_model_out, index=False)
    btts_model_lane.to_csv(btts_model_out, index=False)
    print(f"WROTE: {over25_model_out}")
    print(f"WROTE: {btts_model_out}")

    over25_model_slim = over25_model_lane[[c for c in slim_cols_m + ["model_prob", "model_threshold", "model_lane", "model_pick_from_threshold", "source_prob_col"] if c in over25_model_lane.columns]].copy()
    btts_model_slim = btts_model_lane[[c for c in slim_cols_m + ["model_prob", "model_threshold", "model_lane", "model_pick_from_threshold", "source_prob_col"] if c in btts_model_lane.columns]].copy()

    over25_model_slim_out = out_dir / "OVER25_MODEL_GATED_ROWS_SLIM.csv"
    btts_model_slim_out = out_dir / "BTTS_MODEL_GATED_ROWS_SLIM.csv"
    over25_model_slim.to_csv(over25_model_slim_out, index=False)
    btts_model_slim.to_csv(btts_model_slim_out, index=False)
    print(f"WROTE: {over25_model_slim_out}")
    print(f"WROTE: {btts_model_slim_out}")

    V = _agg_league_table(val, label="val")
    A = _agg_league_table(acc, label="acc")

    M = V.merge(A, on="league", how="outer")

    # Write the unfiltered investor table (no min_n filter)
    M_out = M.copy()
    # Sort primarily by ValueEV ROI (for main table, optional)
    if "roi_val" in M_out.columns:
        M_out["_sort"] = M_out["roi_val"].fillna(-999)
        M_out = M_out.sort_values("_sort", ascending=False).drop(columns=["_sort"])
    # Ensure columns order and all headers present
    for col in headers:
        if col not in M_out.columns:
            M_out[col] = np.nan
    M_out = M_out[headers]
    M_out.to_csv(outp, index=False)

    # Build the presentation-filtered version (with min_n filter and sorting)
    M_pres = M.copy()
    if min_n and "n_val" in M_pres.columns:
        M_pres = M_pres[(M_pres["n_val"].fillna(0).astype(int) >= int(min_n)) | (M_pres["n_val"].isna())].copy()
    if "roi_val" in M_pres.columns:
        M_pres["_sort"] = M_pres["roi_val"].fillna(-999)
        M_pres = M_pres.sort_values("_sort", ascending=False).drop(columns=["_sort"])
    for col in headers:
        if col not in M_pres.columns:
            M_pres[col] = np.nan
    M_pres = M_pres[headers]
    M_pres.to_csv(outp_pres, index=False)

    # Monthly OU25/BTTS league tables (ValueEV)
    OU = _agg_league_table_simple(ou25_val, label="ou25")
    BT = _agg_league_table_simple(btts_val, label="btts")

    O25M = _agg_league_table_simple(over25_model_lane, label="over25_model")
    BTM = _agg_league_table_simple(btts_model_lane, label="btts_model")

    oup = out_dir / "investor_table_ou25_valueEV.csv"
    btp = out_dir / "investor_table_btts_valueEV.csv"

    o25mp = out_dir / "investor_table_over25_model.csv"
    btmp = out_dir / "investor_table_btts_model.csv"

    # Sort by ROI desc when present
    if "roi_ou25" in OU.columns:
        OU["_sort"] = OU["roi_ou25"].fillna(-999)
        OU = OU.sort_values("_sort", ascending=False).drop(columns=["_sort"])
    if "roi_btts" in BT.columns:
        BT["_sort"] = BT["roi_btts"].fillna(-999)
        BT = BT.sort_values("_sort", ascending=False).drop(columns=["_sort"])

    if "roi_over25_model" in O25M.columns:
        O25M["_sort"] = O25M["roi_over25_model"].fillna(-999)
        O25M = O25M.sort_values("_sort", ascending=False).drop(columns=["_sort"])
    if "roi_btts_model" in BTM.columns:
        BTM["_sort"] = BTM["roi_btts_model"].fillna(-999)
        BTM = BTM.sort_values("_sort", ascending=False).drop(columns=["_sort"])

    OU.to_csv(oup, index=False)
    BT.to_csv(btp, index=False)
    print(f"WROTE: {oup}")
    print(f"WROTE: {btp}")

    O25M.to_csv(o25mp, index=False)
    BTM.to_csv(btmp, index=False)
    print(f"WROTE: {o25mp}")
    print(f"WROTE: {btmp}")

    # Monthly BTTS policy freeze artefacts
    btts_policy_csv, btts_policy_md = _write_btts_policy_decision_artifacts(
        out_dir=out_dir,
        month_tag=month_tag,
        btts_model_lane=btts_model_lane,
        btts_valueev_lane=btts_val,
    )
    print(f"WROTE: {btts_policy_csv}")
    print(f"WROTE: {btts_policy_md}")

    # Tiny metadata JSON alongside (so we can prove gate params later)
    meta = {
        "valueev": {"od_min": valueev_od_min, "edge_min": valueev_edge_min, "p_mode": valueev_p_mode},
        "accuracy68": {"imp_min": acc_imp_min, "top_q": acc_top_q, "top_q_scope": acc_top_q_scope, "home_away_only": acc_home_away_only},
        "ou25_valueev": {"od_min": float(ou25_od_min), "edge_min": float(ou25_edge_min)},
        "btts_valueev": {"od_min": float(btts_od_min), "edge_min": float(btts_edge_min)},
        "btts_policy": BTTS_LIVE_POLICY,
        "min_n_valueev": int(min_n),
        "source_backtest": str(backtest_csv),
        "modelstore_root": str(modelstore_root),
        "dedicated_over25_model_rows": int(len(over25_model_lane)),
        "dedicated_btts_model_rows": int(len(btts_model_lane)),
    }
    (out_dir / "investor_table_ftr_valueEV_vs_accuracy68.meta.json").write_text(json.dumps(meta, indent=2))

    return outp


def _windows_for_months(months: Iterable[date], train_months: int) -> List[Window]:
    out: List[Window] = []
    for m in months:
        test_from = _month_start(m)
        test_to = _month_end(m)
        train_to = test_from - timedelta(days=1)
        train_from = _add_months(test_from, -train_months)
        out.append(Window(test_from, test_to, train_from, train_to))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-month", required=True, help="YYYY-MM (first test month)")
    ap.add_argument("--end-month", required=True, help="YYYY-MM (last test month)")
    ap.add_argument("--train-months", type=int, default=24, help="Months for rolling train window (default: 24)")

    ap.add_argument("--leagues", default=None, help="Comma-separated leagues (optional)")
    ap.add_argument("--leagues-file", default=None, help="File with one league per line (optional)")
    ap.add_argument("--markets", default="ftr,ou25,btts,tg15,tg25")

    ap.add_argument("--implied-min", type=float, default=0.68)
    ap.add_argument("--ftr-implied-min", type=float, default=None)
    ap.add_argument("--btts-implied-min", type=float, default=None)
    ap.add_argument("--ou25-implied-min", type=float, default=None)
    ap.add_argument("--tg15-pmin", type=float, default=None)
    ap.add_argument("--tg25-pmin", type=float, default=None)
    ap.add_argument("--tg-pois-ge2-min", type=float, default=None)
    ap.add_argument("--tg-pois-ge3-min", type=float, default=None)
    ap.add_argument("--tg-pois-gap-max-ge2", type=float, default=None)
    ap.add_argument("--tg-pois-gap-max-ge3", type=float, default=None)
    ap.add_argument("--modelstore", default="ModelStore")
    ap.add_argument("--matches-root", default="Matches")

    ap.add_argument("--bookie-cmd", default="python bookie_allmarkets.py")
    ap.add_argument("--backtest-cmd", default="python backtest_deploy_csv.py")

    # Optional: train models each month before generating predictions
    ap.add_argument(
        "--train-then-predict",
        action="store_true",
        help="If set, run --train-cmd for each month using the rolling train window before bookie generation",
    )
    ap.add_argument(
        "--train-cmd",
        default=None,
        help=(
            "Shell command template to train models. Supports placeholders: "
            "{train_from},{train_to},{test_from},{test_to},{matches_root},{modelstore},{leagues},{run_dir},{month}"
        ),
    )
    ap.add_argument("--outdir-root", default="predictions_output/walk_forward")
    ap.add_argument("--skip-bookie", action="store_true")
    ap.add_argument("--skip-backtest", action="store_true")
    ap.add_argument("--summary-only", action="store_true", help="Only generate summary from existing __BACKTEST.csv")
    ap.add_argument("--dry-run", action="store_true")

    ap.add_argument("--min-n-default", type=int, default=20)
    ap.add_argument("--min-n-overrides", default=None, help="CSV with columns: league,market,min_n")
    ap.add_argument("--coverage-min", type=float, default=0.10)
    ap.add_argument("--coverage-max", type=float, default=0.30)

    # FTR product table gates (OOS)
    ap.add_argument("--valueev-od-min", type=float, default=1.80, help="ValueEV gate: minimum bookie_od")
    ap.add_argument("--valueev-edge-min", type=float, default=1.08, help="ValueEV gate: minimum edge (bookie_od * p_bookie_pick)")
    ap.add_argument(
        "--valueev-p-mode",
        type=str,
        default="pois_norm",
        choices=["pois_norm", "confidence"],
        help="Which probability heads define p_bookie_pick for ValueEV (pois_norm recommended)",
    )

    ap.add_argument("--acc-imp-min", type=float, default=0.68, help="Accuracy68 gate: minimum implied prob (od <= 1/imp_min)")
    ap.add_argument("--acc-top-q", type=float, default=None, help="Optional score quantile gate (e.g. 0.70); None disables")
    ap.add_argument(
        "--acc-top-q-scope",
        type=str,
        default="per_league",
        choices=["global", "per_league"],
        help="Score quantile scope for accuracy gate (per_league prevents league starvation)",
    )
    ap.add_argument("--acc-home-away-only", action="store_true", help="Accuracy68 gate: keep only HOME/AWAY picks")

    ap.add_argument("--investor-min-n", type=int, default=50, help="Min n for ValueEV rows to appear in investor table")

    # OU25/BTTS ValueEV gates (OOS)
    ap.add_argument("--ou25-od-min", type=float, default=1.50, help="OU25 ValueEV gate: minimum bookie_od")
    ap.add_argument("--ou25-edge-min", type=float, default=1.03, help="OU25 ValueEV gate: minimum edge (bookie_od * p_pick)")
    ap.add_argument("--btts-od-min", type=float, default=1.33, help="BTTS ValueEV gate: minimum bookie_od")
    ap.add_argument("--btts-edge-min", type=float, default=1.03, help="BTTS ValueEV gate: minimum edge (bookie_od * p_pick)")

    args = ap.parse_args()

    leagues_list: List[str] = []
    if args.leagues:
        leagues_list = [x.strip() for x in args.leagues.split(",") if x.strip()]
    if args.leagues_file:
        p = Path(args.leagues_file)
        if not p.exists():
            raise SystemExit(f"leagues-file not found: {p}")
        leagues_list = [x.strip() for x in p.read_text().splitlines() if x.strip()]

    min_n_overrides = _load_min_n_overrides(Path(args.min_n_overrides) if args.min_n_overrides else None)

    months = _month_range(args.start_month, args.end_month)
    windows = _windows_for_months(months, args.train_months)

    out_root = Path(args.outdir_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for w in windows:
        month_tag = w.test_from.strftime("%Y-%m")
        run_dir = out_root / month_tag
        run_dir.mkdir(parents=True, exist_ok=True)

        date_from = w.test_from.isoformat()
        date_to = w.test_to.isoformat()

        metadata = {
            "test_from": date_from,
            "test_to": date_to,
            "train_from": w.train_from.isoformat(),
            "train_to": w.train_to.isoformat(),
            "leagues": leagues_list,
            "markets": args.markets,
        }
        (run_dir / "walk_forward_metadata.json").write_text(json.dumps(metadata, indent=2))

        allm_path: Optional[Path] = _find_allmarkets(run_dir, date_from, date_to)

        if not args.summary_only:
            # (Optional) Monthly train-then-predict step
            if args.train_then_predict:
                leagues_arg = ",".join(leagues_list) if leagues_list else ""
                train_cmd = _format_cmd_template(
                    args.train_cmd,
                    train_from=w.train_from.isoformat(),
                    train_to=w.train_to.isoformat(),
                    test_from=date_from,
                    test_to=date_to,
                    matches_root=str(args.matches_root),
                    modelstore=str(args.modelstore),
                    leagues=leagues_arg,
                    run_dir=str(run_dir),
                    month=month_tag,
                )
                _run_cmd(train_cmd, args.dry_run)
            if not args.skip_bookie:
                cmd = shlex.split(args.bookie_cmd)
                cmd += [
                    "--date-from",
                    date_from,
                    "--date-to",
                    date_to,
                    "--matches-root",
                    args.matches_root,
                    "--modelstore",
                    args.modelstore,
                    "--run-dir",
                    str(run_dir),
                    "--markets",
                    args.markets,
                    "--implied-min",
                    str(args.implied_min),
                ]
                if args.ftr_implied_min is not None:
                    cmd += ["--ftr-implied-min", str(args.ftr_implied_min)]
                if args.btts_implied_min is not None:
                    cmd += ["--btts-implied-min", str(args.btts_implied_min)]
                if args.ou25_implied_min is not None:
                    cmd += ["--ou25-implied-min", str(args.ou25_implied_min)]
                if args.tg15_pmin is not None:
                    cmd += ["--tg15-pmin", str(args.tg15_pmin)]
                if args.tg25_pmin is not None:
                    cmd += ["--tg25-pmin", str(args.tg25_pmin)]
                if args.tg_pois_ge2_min is not None:
                    cmd += ["--tg-pois-ge2-min", str(args.tg_pois_ge2_min)]
                if args.tg_pois_ge3_min is not None:
                    cmd += ["--tg-pois-ge3-min", str(args.tg_pois_ge3_min)]
                if args.tg_pois_gap_max_ge2 is not None:
                    cmd += ["--tg-pois-gap-max-ge2", str(args.tg_pois_gap_max_ge2)]
                if args.tg_pois_gap_max_ge3 is not None:
                    cmd += ["--tg-pois-gap-max-ge3", str(args.tg_pois_gap_max_ge3)]
                if leagues_list:
                    cmd += ["--leagues", ",".join(leagues_list)]
                _run_cmd(cmd, args.dry_run)

            # In dry-run mode we intentionally do not require files to exist.
            if args.dry_run:
                print(f"DRY-RUN: would look for ALLMARKETS in {run_dir} for {date_from}..{date_to}")
                continue

            allm_path = _find_allmarkets(run_dir, date_from, date_to)
            if not allm_path:
                raise SystemExit(f"No ALLMARKETS file found in {run_dir} for {date_from}..{date_to}")

            if not args.skip_backtest:
                cmd = shlex.split(args.backtest_cmd)
                cmd += [
                    "--deploy-csv",
                    str(allm_path),
                    "--matches-root",
                    args.matches_root,
                    "--outdir",
                    str(run_dir),
                ]
                _run_cmd(cmd, args.dry_run)

        # Summary generation
        if args.dry_run:
            continue

        backtest_csv = run_dir / f"{allm_path.stem if allm_path else 'BOOKIE'}__BACKTEST.csv"
        if not backtest_csv.exists():
            # Try to locate any __BACKTEST.csv in run_dir
            bt = sorted(run_dir.glob("*__BACKTEST.csv"))
            if bt:
                backtest_csv = bt[-1]
            else:
                raise SystemExit(f"No __BACKTEST.csv found in {run_dir}")

        summary = _compute_summary(
            backtest_csv=backtest_csv,
            min_n_default=args.min_n_default,
            min_n_overrides=min_n_overrides,
            cov_min=args.coverage_min,
            cov_max=args.coverage_max,
        )
        out_sum = run_dir / "walk_forward_summary.csv"
        summary.to_csv(out_sum, index=False)

        # Per-month investor table (FTR ValueEV vs 68% Accuracy)
        inv_path = _write_ftr_product_tables(
            backtest_csv=backtest_csv,
            out_dir=run_dir,
            month_tag=month_tag,
            modelstore_root=Path(args.modelstore),
            valueev_od_min=float(args.valueev_od_min),
            valueev_edge_min=float(args.valueev_edge_min),
            valueev_p_mode=str(args.valueev_p_mode),
            acc_imp_min=float(args.acc_imp_min),
            acc_top_q=(float(args.acc_top_q) if args.acc_top_q is not None else None),
            acc_top_q_scope=("per_league" if str(args.acc_top_q_scope) == "per_league" else "global"),
            acc_home_away_only=bool(args.acc_home_away_only),
            ou25_od_min=float(args.ou25_od_min),
            ou25_edge_min=float(args.ou25_edge_min),
            btts_od_min=float(args.btts_od_min),
            btts_edge_min=float(args.btts_edge_min),
            min_n=int(args.investor_min_n),
        )
        print(f"WROTE: {out_sum}")
        print(f"WROTE: {inv_path}")


if __name__ == "__main__":
    main()
