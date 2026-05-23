#!/usr/bin/env python3
"""ftr_consensus.py

Coverage-first consensus picker for FTR (1X2).

Purpose
- Produce a *coverage report* for ALL fixtures with a best-effort consensus pick (HOME/DRAW/AWAY)
  and a signal-strength tier.
- This is intentionally separate from deploy_rulebook.py so we do not disrupt the proven V2/V3
  deploy path.

Inputs
- A BOOKIE_IMP*_ALLMARKETS_<from>_to_<to>.csv produced by bookie_allmarkets.py

Outputs
- <input_stem>__FTR_CONSENSUS.csv (default) in the same folder as the input.

Design
- We route each fixture into a lane based on market microstructure:
    SIDE  : non-close / separated matches (mismatch/favourite)
    MID   : in-between matches (neither clear mismatch nor knife-edge)
    CLOSE : true close matches (tight cluster)
    AVOID : ultra-close + flat + contradictory (still produces a pick, but flagged avoid)

- For every fixture we compute consensus scores for HOME/DRAW/AWAY using layers:
    Model: confidence_home/draw/away
    Market: no-vig implieds (from odds)
    Align: power_diff, ppg_diff_pre, xg_diff
    Goal regime: exp_goals_sum, p00_est, (FTS / GE2 / GE3 heads when present)

This produces:
- consensus_pick
- consensus_confidence (softmax over outcome scores)
- consensus_margin (top - runner-up confidence)
- consensus_tier (ELITE/STRONG/MEDIUM/WEAK/AVOID)
- consensus_lane (SIDE/MID/CLOSE/AVOID)
- reason_codes (compact string)

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd


# -----------------------------
# Utils
# -----------------------------

def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    # clip for stability
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))

def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _softmax(v: np.ndarray, tau: float = 0.15) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if not np.isfinite(v).any():
        return np.array([1/3, 1/3, 1/3], dtype=float)
    tau = float(max(1e-6, tau))
    vv = (v - np.nanmax(v)) / tau
    ex = np.exp(np.clip(vv, -30.0, 30.0))
    s = ex.sum()
    if not np.isfinite(s) or s <= 0:
        return np.array([1/3, 1/3, 1/3], dtype=float)
    return ex / s


def _pick_latest_allmarkets(root: Path = Path("predictions_output")) -> Optional[Path]:
    if not root.exists():
        return None
    cands = list(root.rglob("BOOKIE_IMP*_ALLMARKETS_*.csv"))
    if not cands:
        return None
    cands = sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _coalesce_match_date(df: pd.DataFrame) -> pd.Series:
    for c in ("match_date", "date_GMT", "date", "Date", "timestamp"):
        if c not in df.columns:
            continue
        if c == "timestamp":
            ts = pd.to_numeric(df[c], errors="coerce")
            if ts.notna().any():
                # epoch seconds is typical
                dt = pd.to_datetime(ts, errors="coerce", unit="s", utc=True)
                return dt
        dt = pd.to_datetime(df[c], errors="coerce", utc=True)
        if dt.notna().any():
            return dt
    return pd.Series(pd.NaT, index=df.index)


def _novig_from_odds(od_h: float, od_d: float, od_a: float) -> Tuple[float, float, float]:
    # returns (ph, pd, pa)
    inv = np.array([1.0/od_h, 1.0/od_d, 1.0/od_a], dtype=float)
    s = float(np.nansum(inv))
    if not np.isfinite(s) or s <= 0:
        return (np.nan, np.nan, np.nan)
    inv = inv / s
    return (float(inv[0]), float(inv[1]), float(inv[2]))


def _tier_from(margin: float, lane: str) -> str:
    """Map consensus margin -> signal tier.

    Tiers are lane-aware because SIDE (mismatch/favourite) produces much larger margins than CLOSE/MID.

    The cutoffs below are tuned from observed margin quantiles in UCL/UEL windows:
      - SIDE margins are often 0.25–0.65
      - CLOSE/MID margins are often 0.10–0.26
    """
    m = float(margin) if np.isfinite(margin) else 0.0
    ln = str(lane).upper()

    if ln == "AVOID":
        return "AVOID"

    # SIDE: mismatch / separated market
    # (quantiles observed: p25~0.245, p50~0.503, p75~0.567, p90~0.651)
    if ln == "SIDE":
        if m >= 0.58:
            return "ELITE"
        if m >= 0.45:
            return "STRONG"
        if m >= 0.28:
            return "MEDIUM"
        return "WEAK"

    # MID: neither clearly separated nor knife-edge
    # (observed: ~0.10–0.21)
    if ln == "MID":
        if m >= 0.21:
            return "ELITE"
        if m >= 0.17:
            return "STRONG"
        if m >= 0.13:
            return "MEDIUM"
        return "WEAK"

    # CLOSE: true tight cluster
    # (observed: ~0.10–0.26)
    if m >= 0.24:
        return "ELITE"
    if m >= 0.18:
        return "STRONG"
    if m >= 0.13:
        return "MEDIUM"
    return "WEAK"


# -----------------------------
# Consensus scoring
# -----------------------------

def _xog_tier(xog_pick_score: float, xog_spread: float, lane: str) -> str:
    xs = float(xog_pick_score) if np.isfinite(xog_pick_score) else 0.0
    xd = float(xog_spread) if np.isfinite(xog_spread) else 0.0
    ln = str(lane).upper()

    if ln == "AVOID":
        return "AVOID"

    # SIDE distributions observed (p50~2.18, p75~2.25, p90~2.33) and spread (p50~0.95, p75~1.09, p90~1.24)
    if ln == "SIDE":
        if xs >= 2.30 and xd >= 1.10:
            return "ELITE"
        if xs >= 2.20 and xd >= 0.95:
            return "STRONG"
        if xs >= 2.05 and xd >= 0.75:
            return "MEDIUM"
        return "WEAK"

    # MID observed (p50~1.69, p90~1.70) and spread (p50~0.13, p75~0.17)
    if ln == "MID":
        if xs >= 1.70 and xd >= 0.20:
            return "ELITE"
        if xs >= 1.69 and xd >= 0.17:
            return "STRONG"
        if xs >= 1.67 and xd >= 0.12:
            return "MEDIUM"
        return "WEAK"

    # CLOSE observed (p50~1.71, p90~1.81) and spread (p50~0.12, p75~0.23)
    if xs >= 1.80 and xd >= 0.26:
        return "ELITE"
    if xs >= 1.74 and xd >= 0.20:
        return "STRONG"
    if xs >= 1.70 and xd >= 0.13:
        return "MEDIUM"
    return "WEAK"


# -----------------------------
# Consensus scoring
# -----------------------------

def _lane_for_row(
    *,
    bookie_spread: float,
    implied_prob_diff: float,
    power_diff: float,
    xg_diff_abs: float,
    ppg_diff_abs: float,
    ftr_margin: float,
    close_spread_max: float,
    close_ipd_max: float,
    side_spread_min: float,
    side_ipd_min: float,
) -> str:
    bs = float(bookie_spread) if np.isfinite(bookie_spread) else np.nan
    ipd = float(implied_prob_diff) if np.isfinite(implied_prob_diff) else np.nan

    is_close = (np.isfinite(bs) and np.isfinite(ipd) and bs <= close_spread_max and ipd <= close_ipd_max)
    is_side = (np.isfinite(bs) and bs >= side_spread_min) or (np.isfinite(ipd) and ipd >= side_ipd_min)

    # AVOID: ultra-close + flat model + flat structure
    pd_abs = abs(float(power_diff)) if np.isfinite(power_diff) else 0.0
    xg_abs = float(xg_diff_abs) if np.isfinite(xg_diff_abs) else 0.0
    pp_abs = float(ppg_diff_abs) if np.isfinite(ppg_diff_abs) else 0.0
    mgn = float(ftr_margin) if np.isfinite(ftr_margin) else 0.0

    if is_close and (mgn <= 0.02) and (pd_abs <= 3.0) and (xg_abs <= 0.15) and (pp_abs <= 0.15):
        return "AVOID"

    if is_close:
        return "CLOSE"

    if is_side:
        return "SIDE"

    return "MID"


def _score_outcomes(
    *,
    p_model: Tuple[float, float, float],
    p_bk: Tuple[float, float, float],
    power_diff: float,
    ppg_diff: float,
    xg_diff: float,
    exp_goals_sum: float,
    p00_est: float,
    p_home_fts: float,
    p_away_fts: float,
    home_ge2: float,
    away_ge2: float,
    home_ge3: float,
    away_ge3: float,
    bookie_spread: float,
    implied_prob_diff: float,
    # Optional Poisson 1X2 masses
    p_home_pois: float = np.nan,
    p_draw_pois: float = np.nan,
    p_away_pois: float = np.nan,
    # Optional rolling rates (small adjustments)
    scored_rate_5_home: float = np.nan,
    scored_rate_5_away: float = np.nan,
    conceded_rate_5_home: float = np.nan,
    conceded_rate_5_away: float = np.nan,
    goaliness_avg_5_home: float = np.nan,
    goaliness_avg_5_away: float = np.nan,
    h2h_goaliness_avg: float = np.nan,
    # Risk flags
    draw_risk_flag: float = np.nan,
    chaos_risk_flag: float = np.nan,
    draw_chaos_score: float = np.nan,
    not_glue_flag: float = np.nan,
    return_components: bool = False,
) -> Tuple[np.ndarray, str, Dict[str, float]]:
    """Return (scores[3], reason_codes, component_sums).

    Scores are built from multiple blocks:
      - model logits (primary)
      - market logits (prior)
      - edge (p_model - p_bk)
      - side tilt from power/ppg/xg
      - draw regime bonus from exp_goals_sum + p00 + poisson draw
      - specialist head bonuses (GE2/GE3/FTS)
      - small context tweaks (rolling rates / goaliness / h2h)
      - risk penalties (chaos/draw-risk)
    """

    pm = np.array(p_model, dtype=float)
    pb = np.array(p_bk, dtype=float)

    pm = np.where(np.isfinite(pm), pm, 1/3)
    pb = np.where(np.isfinite(pb), pb, 1/3)

    # --- Block weights (lane-agnostic v1; we can later make lane-aware) ---
    wM = 1.10  # model logit
    wB = 0.55  # market logit
    wG = 0.80  # edge (prob gap)
    wT = 0.55  # side tilt
    wP = 0.35  # poisson logit
    wD = 0.60  # draw regime
    wH = 0.45  # specialist heads
    wC = 0.15  # context (rolling/h2h)
    wR = 0.60  # risk penalty

    # --- Block 1: model logits ---
    mlog = np.tanh(_logit(pm) / 2.0)

    # --- Block 2: market logits ---
    blog = np.tanh(_logit(pb) / 2.0)

    # --- Edge ---
    gap = pm - pb

    # --- Block 3: side tilt (HOME vs AWAY) ---
    a_power = 0.5
    if np.isfinite(power_diff):
        a_power = float(_sigmoid(np.array([power_diff / 8.0]))[0])

    a_ppg = 0.5
    if np.isfinite(ppg_diff):
        a_ppg = float(_sigmoid(np.array([ppg_diff / 0.35]))[0])

    a_xg = 0.5
    if np.isfinite(xg_diff):
        a_xg = float(_sigmoid(np.array([xg_diff / 0.25]))[0])

    a_home = float(np.mean([a_power, a_ppg, a_xg]))
    a_away = 1.0 - a_home

    # --- Block 4: draw regime ---
    draw_reg = 0.0
    if np.isfinite(exp_goals_sum):
        draw_reg += float(np.clip((2.75 - exp_goals_sum) * 0.10, -0.20, 0.20))
    if np.isfinite(p00_est):
        draw_reg += float(np.clip((p00_est - 0.06) * 1.10, -0.15, 0.20))
    if np.isfinite(p_draw_pois):
        draw_reg += float(np.clip((p_draw_pois - (1/3)) * 0.60, -0.12, 0.12))

    # microstructure suppressors (prevent draw overuse when not truly close)
    if np.isfinite(bookie_spread):
        draw_reg -= float(np.clip((bookie_spread - 0.06) * 0.35, -0.12, 0.12))
    if np.isfinite(implied_prob_diff):
        draw_reg -= float(np.clip((implied_prob_diff - 0.08) * 0.45, -0.15, 0.15))

    # --- Block 5: specialist heads ---
    heads_home = 0.0
    heads_away = 0.0
    heads_draw = 0.0

    fts_max = np.nanmax([p_home_fts, p_away_fts]) if (np.isfinite(p_home_fts) or np.isfinite(p_away_fts)) else np.nan
    if np.isfinite(fts_max):
        # higher blank risk -> slightly more drawish, and suppress extreme win certainty
        heads_draw += float(np.clip((fts_max - 0.25) * 0.25, -0.05, 0.08))

    # If home GE2 high and away FTS high -> HOME bonus
    if np.isfinite(home_ge2) and np.isfinite(p_away_fts):
        heads_home += float(np.clip((home_ge2 - 0.45) * (p_away_fts - 0.20) * 1.20, -0.06, 0.10))

    # If away GE2 high and home FTS high -> AWAY bonus
    if np.isfinite(away_ge2) and np.isfinite(p_home_fts):
        heads_away += float(np.clip((away_ge2 - 0.45) * (p_home_fts - 0.20) * 1.20, -0.06, 0.10))

    # GE3 acts as stronger tail support
    if np.isfinite(home_ge3):
        heads_home += float(np.clip((home_ge3 - 0.12) * 0.20, -0.04, 0.06))
    if np.isfinite(away_ge3):
        heads_away += float(np.clip((away_ge3 - 0.12) * 0.20, -0.04, 0.06))

    # If both teams look capable (GE2 min high) -> reduce draw
    if np.isfinite(home_ge2) and np.isfinite(away_ge2):
        ge2_min = float(min(home_ge2, away_ge2))
        heads_draw -= float(np.clip((ge2_min - 0.45) * 0.20, -0.05, 0.05))

    # --- Block 6: rolling + h2h (small context) ---
    ctx_home = 0.0
    ctx_away = 0.0
    ctx_draw = 0.0

    if np.isfinite(scored_rate_5_home) and np.isfinite(scored_rate_5_away):
        # if one side is scoring much more, slight side tilt
        ctx_home += float(np.clip((scored_rate_5_home - scored_rate_5_away) * 0.06, -0.05, 0.05))
        ctx_away -= float(np.clip((scored_rate_5_home - scored_rate_5_away) * 0.06, -0.05, 0.05))

    if np.isfinite(conceded_rate_5_home) and np.isfinite(conceded_rate_5_away):
        # if home concedes more, away gets slight bump
        ctx_away += float(np.clip((conceded_rate_5_home - conceded_rate_5_away) * 0.05, -0.05, 0.05))
        ctx_home -= float(np.clip((conceded_rate_5_home - conceded_rate_5_away) * 0.05, -0.05, 0.05))

    if np.isfinite(goaliness_avg_5_home) and np.isfinite(goaliness_avg_5_away):
        g = 0.5 * (goaliness_avg_5_home + goaliness_avg_5_away)
        # higher goaliness -> slightly less draw
        ctx_draw -= float(np.clip((g - 2.7) * 0.04, -0.05, 0.05))

    if np.isfinite(h2h_goaliness_avg):
        ctx_draw -= float(np.clip((h2h_goaliness_avg - 2.7) * 0.03, -0.04, 0.04))

    # --- Block 7: risk penalties ---
    risk = 0.0
    if np.isfinite(draw_chaos_score):
        risk += float(np.clip((draw_chaos_score - 0.55) * 0.90, 0.0, 0.60))
    if np.isfinite(chaos_risk_flag) and int(chaos_risk_flag) == 1:
        risk += 0.25
    if np.isfinite(draw_risk_flag) and int(draw_risk_flag) == 1:
        risk += 0.15
    if np.isfinite(not_glue_flag) and int(not_glue_flag) == 1:
        risk += 0.10

    # Per-outcome poisson
    plog = np.array([0.0, 0.0, 0.0], dtype=float)
    if np.isfinite(p_home_pois) and np.isfinite(p_draw_pois) and np.isfinite(p_away_pois):
        plog = np.tanh(_logit(np.array([p_home_pois, p_draw_pois, p_away_pois], dtype=float)) / 2.0)

    # Compose
    s_home = wM*mlog[0] + wB*blog[0] + wG*gap[0] + wT*a_home + wP*plog[0] + wH*heads_home + wC*ctx_home - wR*risk
    s_draw = wM*mlog[1] + wB*blog[1] + wG*gap[1] + wD*draw_reg + wP*plog[1] + wH*heads_draw + wC*ctx_draw - wR*risk
    s_away = wM*mlog[2] + wB*blog[2] + wG*gap[2] + wT*a_away + wP*plog[2] + wH*heads_away + wC*ctx_away - wR*risk

    scores = np.array([s_home, s_draw, s_away], dtype=float)

    # reason codes
    rc: List[str] = []
    top = int(np.nanargmax(scores)) if np.isfinite(scores).any() else 1
    rc.append("PICK_HOME" if top==0 else ("PICK_AWAY" if top==2 else "PICK_DRAW"))
    if np.isfinite(power_diff) and abs(power_diff) >= 8:
        rc.append("POWER")
    if np.isfinite(ppg_diff) and abs(ppg_diff) >= 0.35:
        rc.append("PPG")
    if np.isfinite(xg_diff) and abs(xg_diff) >= 0.25:
        rc.append("XG")
    if np.isfinite(exp_goals_sum) and exp_goals_sum <= 2.6:
        rc.append("LOW_GOALS")
    if np.isfinite(exp_goals_sum) and exp_goals_sum >= 3.2:
        rc.append("HIGH_GOALS")
    if np.isfinite(p00_est) and p00_est >= 0.07:
        rc.append("P00")
    if np.isfinite(fts_max) and fts_max >= 0.32:
        rc.append("FTS")
    if np.isfinite(draw_chaos_score) and draw_chaos_score >= 0.60:
        rc.append("CHAOS")
    if np.isfinite(bookie_spread) and bookie_spread <= 0.10 and np.isfinite(implied_prob_diff) and implied_prob_diff <= 0.12:
        rc.append("TRUE_CLOSE")

    comps: Dict[str, float] = {}
    if return_components:
        comps = {
            "comp_model": float(wM*np.nanmean(mlog)),
            "comp_market": float(wB*np.nanmean(blog)),
            "comp_edge": float(wG*np.nanmean(gap)),
            "comp_tilt": float(wT*(a_home - 0.5)),
            "comp_draw": float(wD*draw_reg),
            "comp_pois": float(wP*np.nanmean(plog)),
            "comp_heads": float(wH*(heads_home + heads_draw + heads_away)),
            "comp_ctx": float(wC*(ctx_home + ctx_draw + ctx_away)),
            "comp_risk": float(-wR*risk),
        }

    return scores, "+".join(rc), comps


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Consensus FTR coverage report")
    ap.add_argument("--src", default=None, help="Path to BOOKIE_IMP*_ALLMARKETS_*.csv (default: latest under predictions_output)")
    ap.add_argument("--out", default=None, help="Output CSV path (default: alongside src with __FTR_CONSENSUS)")

    # Lane thresholds
    ap.add_argument("--close-spread-max", type=float, default=0.09)
    ap.add_argument("--close-ipd-max", type=float, default=0.11)
    ap.add_argument("--side-spread-min", type=float, default=0.16)
    ap.add_argument("--side-ipd-min", type=float, default=0.20)
    ap.add_argument("--tau", type=float, default=1.50, help="Softmax temperature (higher = less peaky).")
    ap.add_argument("--xog-k", type=float, default=0.25, help="XOG sensitivity (prob delta scale around 1/3).")
    ap.add_argument("--write-components", action="store_true", help="Write component columns (model/market/tilt/draw/heads/risk) for auditing.")
    ap.add_argument("--bankers-n", type=int, default=50, help="Write BANKERS file with top-N by xog_spread (default: 50)")
    ap.add_argument("--no-extra-outputs", action="store_true", help="Disable writing shortlist/bankers output CSVs")

    args = ap.parse_args()

    src = Path(args.src) if args.src else _pick_latest_allmarkets()
    if not src or not Path(src).exists():
        raise SystemExit("No ALLMARKETS source found. Provide --src.")

    df = pd.read_csv(src)
    if df is None or df.empty:
        raise SystemExit("src CSV empty")

    # Work only on FTR rows for coverage
    ftr = df[df.get("market", "").astype(str).str.lower().str.strip().eq("ftr")].copy()
    if ftr.empty:
        raise SystemExit("No FTR rows in input")

    # Coerce common numeric columns
    for c in [
        "confidence_home","confidence_draw","confidence_away",
        "model_strength","ftr_margin",
        "home_goals_pred","away_goals_pred","lambda_home","lambda_away","exp_goals_sum","p00_est",
        "pre_match_xg_home","pre_match_xg_away","xg_diff_abs",
        "bookie_spread","implied_prob_diff","odds_diff",
        "ppg_home_pre","ppg_away_pre","ppg_diff_pre",
        "home_power_rating","away_power_rating","power_diff",
        "home_ge2_confidence","away_ge2_confidence","home_ge3_confidence","away_ge3_confidence",
        "p_home_fts","p_away_fts",
        "od_home","od_draw","od_away",
        "imp_home","imp_draw","imp_away",
        # Additional fields for upgraded _score_outcomes
        "p_home_pois","p_draw_pois","p_away_pois",
        "scored_rate_5_home","scored_rate_5_away",
        "conceded_rate_5_home","conceded_rate_5_away",
        "goaliness_avg_5_home","goaliness_avg_5_away",
        "h2h_goaliness_avg",
        "draw_risk_flag","chaos_risk_flag","draw_chaos_score","not_glue_flag",
    ]:
        if c in ftr.columns:
            ftr[c] = _to_num(ftr[c])

    # Ensure fixture_key exists
    if "fixture_key" not in ftr.columns:
        # best-effort key
        ftr["fixture_key"] = (
            ftr.get("league", "").astype(str) + "||" +
            ftr.get("home_team_name", "").astype(str) + "||" +
            ftr.get("away_team_name", "").astype(str) + "||" +
            _coalesce_match_date(ftr).astype(str)
        )

    # One canonical row per fixture for model distribution + structure
    # Prefer GLUE/base rows (pool_tier empty/GLUE) else fall back to any.
    tier = ftr.get("pool_tier", "").astype(str).str.upper().fillna("")
    is_base = (tier.eq("") | tier.eq("GLUE"))

    base = ftr[is_base].copy()
    if base.empty:
        base = ftr.copy()

    # Deduplicate by fixture_key (keep highest model_strength if present)
    if "model_strength" in base.columns:
        base = base.sort_values(["fixture_key", "model_strength"], ascending=[True, False])
    base = base.drop_duplicates(subset=["fixture_key"], keep="first").copy()

    # Compute market implieds per fixture
    # If imp_* present use them, else compute from odds.
    def _get_probs(r: pd.Series) -> Tuple[float,float,float]:
        ih = float(r.get("imp_home", np.nan))
        idd = float(r.get("imp_draw", np.nan))
        ia = float(r.get("imp_away", np.nan))
        if np.isfinite(ih) and np.isfinite(idd) and np.isfinite(ia):
            s = ih + idd + ia
            if np.isfinite(s) and s > 0:
                return (ih/s, idd/s, ia/s)
        od_h = float(r.get("od_home", np.nan))
        od_d = float(r.get("od_draw", np.nan))
        od_a = float(r.get("od_away", np.nan))
        if np.isfinite(od_h) and np.isfinite(od_d) and np.isfinite(od_a) and od_h>1 and od_d>1 and od_a>1:
            return _novig_from_odds(od_h, od_d, od_a)
        return (np.nan,np.nan,np.nan)

    # Compute consensus per fixture
    rows: List[Dict[str, object]] = []

    md = _coalesce_match_date(base)
    base = base.assign(_match_dt=md)

    for _, r in base.iterrows():
        fixture_key = str(r.get("fixture_key", "")).strip()
        league = str(r.get("league", "")).strip()

        # Model probs
        p_model = (
            float(r.get("confidence_home", np.nan)),
            float(r.get("confidence_draw", np.nan)),
            float(r.get("confidence_away", np.nan)),
        )

        # Market novig implied
        p_bk = _get_probs(r)

        # Structure
        pre_xg_h = float(r.get("pre_match_xg_home", np.nan))
        pre_xg_a = float(r.get("pre_match_xg_away", np.nan))
        xg_diff = (pre_xg_h - pre_xg_a) if (np.isfinite(pre_xg_h) and np.isfinite(pre_xg_a)) else np.nan

        xg_abs = float(r.get("xg_diff_abs", np.nan))
        if not np.isfinite(xg_abs) and np.isfinite(xg_diff):
            xg_abs = abs(xg_diff)

        ppg_diff = float(r.get("ppg_diff_pre", np.nan))
        ppg_abs = abs(ppg_diff) if np.isfinite(ppg_diff) else np.nan

        # Lane
        lane = _lane_for_row(
            bookie_spread=float(r.get("bookie_spread", np.nan)),
            implied_prob_diff=float(r.get("implied_prob_diff", np.nan)),
            power_diff=float(r.get("power_diff", np.nan)),
            xg_diff_abs=xg_abs,
            ppg_diff_abs=float(ppg_abs) if np.isfinite(ppg_abs) else np.nan,
            ftr_margin=float(r.get("ftr_margin", np.nan)),
            close_spread_max=float(args.close_spread_max),
            close_ipd_max=float(args.close_ipd_max),
            side_spread_min=float(args.side_spread_min),
            side_ipd_min=float(args.side_ipd_min),
        )

        # Outcome scores
        scores, rc, comps = _score_outcomes(
            p_model=p_model,
            p_bk=p_bk,
            power_diff=float(r.get("power_diff", np.nan)),
            ppg_diff=ppg_diff,
            xg_diff=xg_diff,
            exp_goals_sum=float(r.get("exp_goals_sum", np.nan)),
            p00_est=float(r.get("p00_est", np.nan)),
            p_home_fts=float(r.get("p_home_fts", np.nan)),
            p_away_fts=float(r.get("p_away_fts", np.nan)),
            home_ge2=float(r.get("home_ge2_confidence", np.nan)),
            away_ge2=float(r.get("away_ge2_confidence", np.nan)),
            home_ge3=float(r.get("home_ge3_confidence", np.nan)),
            away_ge3=float(r.get("away_ge3_confidence", np.nan)),
            bookie_spread=float(r.get("bookie_spread", np.nan)),
            implied_prob_diff=float(r.get("implied_prob_diff", np.nan)),
            p_home_pois=float(r.get("p_home_pois", np.nan)),
            p_draw_pois=float(r.get("p_draw_pois", np.nan)),
            p_away_pois=float(r.get("p_away_pois", np.nan)),
            scored_rate_5_home=float(r.get("scored_rate_5_home", np.nan)),
            scored_rate_5_away=float(r.get("scored_rate_5_away", np.nan)),
            conceded_rate_5_home=float(r.get("conceded_rate_5_home", np.nan)),
            conceded_rate_5_away=float(r.get("conceded_rate_5_away", np.nan)),
            goaliness_avg_5_home=float(r.get("goaliness_avg_5_home", np.nan)),
            goaliness_avg_5_away=float(r.get("goaliness_avg_5_away", np.nan)),
            h2h_goaliness_avg=float(r.get("h2h_goaliness_avg", np.nan)),
            draw_risk_flag=float(r.get("draw_risk_flag", np.nan)),
            chaos_risk_flag=float(r.get("chaos_risk_flag", np.nan)),
            draw_chaos_score=float(r.get("draw_chaos_score", np.nan)),
            not_glue_flag=float(r.get("not_glue_flag", np.nan)),
            return_components=bool(args.write_components),
        )

        probs = _softmax(scores, tau=float(args.tau))
        top_i = int(np.argmax(probs))
        top_p = float(probs[top_i])
        second_p = float(np.sort(probs)[-2])
        margin = float(top_p - second_p)

        pick = "DRAW"
        if top_i == 0:
            pick = "HOME"
        elif top_i == 2:
            pick = "AWAY"

        tier = _tier_from(margin, lane)
        if lane == "AVOID":
            tier = "AVOID"

        # For completeness, compute simple pick-strength score
        strength = margin

        # XOG scores
        k = float(args.xog_k)
        xog_home = 3.0 * float(_sigmoid(np.array([(probs[0] - 1/3) / k]))[0])
        xog_draw = 3.0 * float(_sigmoid(np.array([(probs[1] - 1/3) / k]))[0])
        xog_away = 3.0 * float(_sigmoid(np.array([(probs[2] - 1/3) / k]))[0])
        xogs = np.array([xog_home, xog_draw, xog_away], dtype=float)
        i_best = int(np.argmax(xogs))
        xog_pick = "HOME" if i_best == 0 else ("AWAY" if i_best == 2 else "DRAW")
        x_sorted = np.sort(xogs)
        xog_pick_score = float(x_sorted[-1])
        xog_spread = float(x_sorted[-1] - x_sorted[-2])

        xog_tier = _xog_tier(xog_pick_score, xog_spread, lane)

        row = {
            "league": league,
            "fixture_key": fixture_key,
            "match_date": str(r.get("_match_dt", "")) if pd.notna(r.get("_match_dt")) else "",
            "home_team_name": str(r.get("home_team_name", "")),
            "away_team_name": str(r.get("away_team_name", "")),

            # Market
            "od_home": float(r.get("od_home", np.nan)),
            "od_draw": float(r.get("od_draw", np.nan)),
            "od_away": float(r.get("od_away", np.nan)),
            "bk_home": float(p_bk[0]) if np.isfinite(p_bk[0]) else np.nan,
            "bk_draw": float(p_bk[1]) if np.isfinite(p_bk[1]) else np.nan,
            "bk_away": float(p_bk[2]) if np.isfinite(p_bk[2]) else np.nan,
            "bookie_spread": float(r.get("bookie_spread", np.nan)),
            "implied_prob_diff": float(r.get("implied_prob_diff", np.nan)),
            "odds_diff": float(r.get("odds_diff", np.nan)),

            # Model
            "model_top_pick": str(r.get("model_top_pick", "")),
            "agree_model_vs_bookie": int(pd.to_numeric(r.get("agree_model_vs_bookie", 0), errors="coerce") or 0),
            "model_strength": float(r.get("model_strength", np.nan)),
            "confidence_home": float(p_model[0]) if np.isfinite(p_model[0]) else np.nan,
            "confidence_draw": float(p_model[1]) if np.isfinite(p_model[1]) else np.nan,
            "confidence_away": float(p_model[2]) if np.isfinite(p_model[2]) else np.nan,
            "ftr_margin": float(r.get("ftr_margin", np.nan)),

            # Goals/Poisson
            "home_goals_pred": float(r.get("home_goals_pred", np.nan)),
            "away_goals_pred": float(r.get("away_goals_pred", np.nan)),
            "lambda_home": float(r.get("lambda_home", np.nan)),
            "lambda_away": float(r.get("lambda_away", np.nan)),
            "exp_goals_sum": float(r.get("exp_goals_sum", np.nan)),
            "p00_est": float(r.get("p00_est", np.nan)),

            # Structure
            "pre_match_xg_home": float(r.get("pre_match_xg_home", np.nan)),
            "pre_match_xg_away": float(r.get("pre_match_xg_away", np.nan)),
            "xg_diff_abs": float(xg_abs) if np.isfinite(xg_abs) else np.nan,
            "ppg_home_pre": float(r.get("ppg_home_pre", np.nan)),
            "ppg_away_pre": float(r.get("ppg_away_pre", np.nan)),
            "ppg_diff_pre": float(ppg_diff) if np.isfinite(ppg_diff) else np.nan,

            # Power
            "home_power_rating": float(r.get("home_power_rating", np.nan)),
            "away_power_rating": float(r.get("away_power_rating", np.nan)),
            "power_diff": float(r.get("power_diff", np.nan)),

            # Specialist heads
            "home_ge2_confidence": float(r.get("home_ge2_confidence", np.nan)),
            "away_ge2_confidence": float(r.get("away_ge2_confidence", np.nan)),
            "home_ge3_confidence": float(r.get("home_ge3_confidence", np.nan)),
            "away_ge3_confidence": float(r.get("away_ge3_confidence", np.nan)),
            "p_home_fts": float(r.get("p_home_fts", np.nan)),
            "p_away_fts": float(r.get("p_away_fts", np.nan)),

            # Consensus
            "consensus_lane": lane,
            "consensus_pick": pick,
            "consensus_confidence": top_p,
            "consensus_margin": margin,
            "consensus_strength": strength,
            "consensus_tier": tier,
            "reason_codes": rc,
            # XOG
            "xog_home": xog_home,
            "xog_draw": xog_draw,
            "xog_away": xog_away,
            "xog_pick": xog_pick,
            "xog_pick_score": xog_pick_score,
            "xog_spread": xog_spread,
            "xog_tier": xog_tier,
        }
        if bool(args.write_components) and comps:
            row.update(comps)
        rows.append(row)

    out_df = pd.DataFrame(rows)

    # Stable sort
    try:
        out_df["_dt"] = pd.to_datetime(out_df["match_date"], errors="coerce", utc=True)
        out_df = out_df.sort_values(["league", "_dt", "consensus_tier", "consensus_confidence"], ascending=[True, True, True, False])
        out_df = out_df.drop(columns=["_dt"], errors="ignore")
    except Exception:
        pass

    out_path = Path(args.out) if args.out else (src.parent / f"{src.stem}__FTR_CONSENSUS.csv")
    out_df.to_csv(out_path, index=False)

    # ------------------------------------------------------------
    # Extra outputs: DEPLOY shortlist + BANKERS top-N by spread
    # ------------------------------------------------------------
    if not bool(getattr(args, "no_extra_outputs", False)):
        try:
            tmp = out_df.copy()
            tmp["consensus_lane"] = tmp.get("consensus_lane", "").astype(str).str.upper().str.strip()
            tmp["xog_tier"] = tmp.get("xog_tier", "").astype(str).str.upper().str.strip()
            tmp["xog_spread"] = pd.to_numeric(tmp.get("xog_spread", np.nan), errors="coerce")
            tmp["xog_pick_score"] = pd.to_numeric(tmp.get("xog_pick_score", np.nan), errors="coerce")

            # DEPLOY shortlist: SIDE + (ELITE/STRONG)
            short = tmp[(tmp["consensus_lane"] == "SIDE") & (tmp["xog_tier"].isin(["ELITE", "STRONG"]))].copy()
            short = short.sort_values(["xog_spread", "xog_pick_score"], ascending=[False, False])
            shortlist_path = out_path.with_name(out_path.name.replace("__FTR_CONSENSUS.csv", "__FTR_XOG_TIER_SHORTLIST.csv"))
            short.to_csv(shortlist_path, index=False)

            # BANKERS: top-N by xog_spread (exclude AVOID)
            n_bank_req = int(getattr(args, "bankers_n", 50) or 50)
            bank = tmp[tmp["consensus_lane"] != "AVOID"].copy()
            bank = bank[bank["xog_spread"].notna()].copy()
            bank = bank.sort_values(["xog_spread", "xog_pick_score"], ascending=[False, False]).head(max(0, n_bank_req))
            n_bank_eff = int(len(bank))
            bankers_path = out_path.with_name(out_path.name.replace("__FTR_CONSENSUS.csv", f"__FTR_BANKERS_TOP{n_bank_eff}.csv"))
            bank.to_csv(bankers_path, index=False)

            try:
                print("WROTE:", shortlist_path, "rows:", len(short))
                print("WROTE:", bankers_path, "rows:", len(bank), f"(requested={n_bank_req})")
            except Exception:
                pass
        except Exception as _e:
            print(f"⚠️ could not write extra outputs: {_e}")

    # Console summary
    try:
        print("SRC:", src)
        print("WROTE:", out_path)
        print("rows:", len(out_df))
        print("lane_counts:", out_df["consensus_lane"].value_counts().to_dict())
        print("tier_counts:", out_df["consensus_tier"].value_counts().to_dict())
        print("xog_tier_counts:", out_df["xog_tier"].value_counts().to_dict() if "xog_tier" in out_df.columns else {})
        if not bool(getattr(args, "no_extra_outputs", False)):
            try:
                dl = out_df[(out_df["consensus_lane"].astype(str).str.upper() == "SIDE") & (out_df.get("xog_tier", "").astype(str).str.upper().isin(["ELITE", "STRONG"]))]
                print("deploy_shortlist_rows:", int(len(dl)))
            except Exception:
                pass
    except Exception:
        pass


if __name__ == "__main__":
    main()