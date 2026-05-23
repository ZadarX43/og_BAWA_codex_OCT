# OU25 (Over 2.5 Goals) Prediction Deployments — Rulebook, Findings, and Production Decision Function

This document is the single source of truth for how we **evaluate**, **gate**, and **publish** OU25 (Over 2.5) picks in production.

It captures:

- the **two scoring lanes** (baseline vs deploy-quality),
- the **elite deploy profile** we validated,
- the exact **CLI command** used to produce the ~98.62% elite run,
- the **per-league stationary threshold philosophy** (TopQ + quantile signal banding),
- the **production deploy decision function** (`should_publish_over25(row) -> bool`) with an accompanying **human-readable UI reason** string,
- and a **drop-in wrapper** to create a deploy-ready CSV.

---

## Definitions

- **OU25 / Over 2.5**: total match goals ≥ 3.
- **model_p_for_bookie**: model probability aligned to the bookie side (here, OVER25 lane).
- **TopQ (per-league)**: keep only the top X% of `model_p_for_bookie` values within each league.
- **Signal strength (per-league)**:
  - `VERY_STRONG_OVER` = top percentile band within the league
  - `STRONG_OVER` = next percentile band within the league
  - `NEUTRAL` = everything else  
  Signals are **derived** (recomputed) from the same distribution you deploy against.

---

## Key Findings

### 1) There are two different truths: baseline vs deploy-quality

**Baseline lane (wide coverage, monitoring, not a premium product)**
- Scores **all OVER25 picks** across many leagues with minimal gating.
- Result observed in this validation slice: **85.66% accuracy** on **1834 picks**.
- Use: engine health monitoring, broad diagnostics, discovery.

**Deploy-quality lane (forward-facing “product”)**
- Uses:
  - **elite whitelist**
  - **per-league TopQ**
  - **per-league quantile banding** (derived signals)
  - strict signal gating (STRONG / VERY_STRONG only)
- Result observed: **98.62% accuracy** on **217 picks**.

**Key insight:**  
Signal labels must be **recomputed** from `model_p_for_bookie` (derived), not trusted from old artifacts.  
This prevents accidental evaluation of stale/incorrect `signal_over25`.

---

### 2) The deploy logic that works (and why)

To publish an Over 2.5 pick, you enforce **3 gates**:

#### Gate A — League eligibility (product lane)
League choice dominates reliability. For the validated “elite” lane:

**Elite keep-leagues**
- Germany Bundesliga
- Netherlands Eredivisie
- Italy Serie A
- Portugal Liga
- Champions League
- USA MLS
- France Ligue 1
- Belgium Pro *(kept, but treated as “weaker”)*

Everything else is excluded or treated as non-elite.

#### Gate B — TopQ per league
You keep only the **top 20%** of `model_p_for_bookie` values within each league.

- TopQ = 0.80 (per-league quantile threshold)
- This makes the gate **stationary by league**, avoiding fragile global thresholds.

#### Gate C — Signal strength per league (derived from the same distribution)
Signals are defined by **rank percentiles** within a league (tie-safe):

Default elite banding:
- strong_q = 0.80
- vstrong_q = 0.90

So:
- `VERY_STRONG_OVER` = top 10% of p inside league  
- `STRONG_OVER` = 80–90% band inside league  
- `NEUTRAL` = everything else  

Deploy (strict mode) keeps only:
- `STRONG_OVER` or `VERY_STRONG_OVER`

---

### 3) Belgium behavior (verified)
With the override `signals={"VERY_STRONG_OVER"}`, Belgium output should be **100% VERY_STRONG_OVER** post-filter.

Observed check:
- Belgium scored rows: 11
- Belgium signal dist: VERY_STRONG_OVER = 11

Meaning: the override is working as intended.

---

## Exact Command Used (Elite Deploy Validation)

This is the command used to produce the elite deploy run shown below.

```bash
python backtest_ou25_from_merged.py \
  --no-policy \
  --pred-path "predictions_output/ou25_frozen_compare/rulebook_ftr_validation_3yr_19lg_v1/BOOKIE_IMP40_ALLMARKETS_2022-01-01_to_2025-12-31__BACKTEST/ou25_combined_topq_080/BOOKIE_IMP40_ALLMARKETS_2022-01-01_to_2025-12-31__BACKTEST__OU25_COMBINED_TOPQ_080.csv" \
  --date-from 2022-01-01 --date-to 2025-09-30 \
  --autofix-misses --miss-autopsy \
  --deploy-v0 \
  --deploy-profile elite \
  --deploy-topq-q 0.80 \
  --deploy-signal-recompute always \
  --deploy-signal-banding quantile \
  --deploy-strong-q 0.80 \
  --deploy-vstrong-q 0.90 \
  --deploy-print-signal-dist
```

---

## Observed Output (Elite Deploy Validation)

### Truth coverage by league
All elite leagues in this run had **100% truth coverage** (scored_n == pred_n).

```
USA MLS                     43 / 43
Germany Bundesliga          35 / 35
Netherlands Eredivisie      35 / 35
France Ligue 1              28 / 28
Champions League            25 / 25
Portugal Liga               21 / 21
Italy Serie A               19 / 19
Belgium Pro                 11 / 11
```

### Accuracy
- Scored rows: **217 / 217**
- Accuracy: **98.62%**

### Accuracy by signal (derived)
- VERY_STRONG_OVER: 128 rows → 98.44%
- STRONG_OVER: 89 rows → 98.88%

### Accuracy by league (scored)
- USA MLS: 97.67%
- Germany Bundesliga: 100%
- Netherlands Eredivisie: 100%
- France Ligue 1: 96.43%
- Champions League: 100%
- Portugal Liga: 100%
- Italy Serie A: 100%
- Belgium Pro: 90.91%

---

## Production Rulebook for Over 2.5 Publishing

### “Forward-facing” publish rule (Elite lane)

A match gets an **Over 2.5** recommendation only if:

1) **League is in ELITE_KEEP_LEAGUES**  
2) `bookie_pick == OVER25` *(OVER-only lane gating)*  
3) `model_p_for_bookie` is in **TopQ** for that league (default TopQ=0.80)  
4) Derived signal (per-league banding) is:
   - `VERY_STRONG_OVER` OR `STRONG_OVER` (strict mode)  
5) Apply **overrides**:
   - Exclusions
   - Optional `min_p` floors
   - Per-league `strong_q` / `vstrong_q`
   - Per-league allowed signals (e.g., Belgium = VERY_STRONG only)

---

## Where do the “caches” come from in production?

In production, you must build caches from the **same distribution you are ranking against**, typically:

- “current league-model snapshot” predictions export, or
- a rolling window of recent predictions (e.g., last N matchdays), or
- a recent validated backtest window matching the current season state.

**Do not mix seasons silently.**  
If you change the distribution but reuse thresholds, you can drift the meaning of “TopQ” and signal bands.

---

## League Overrides (Current)

This is the per-league override map that powers exclusions and league-specific banding:

```python
LEAGUE_OVERRIDES_DEFAULT = {
    "Europa Conference": {"exclude": True},
    "Brazil Serie A": {"exclude": True},
    "Japan J1": {"exclude": True},

    "England Premier League": {"exclude": True},
    "Spain La Liga": {"exclude": True},
    "England EFL League 1": {"exclude": True},
    "England FA Cup": {"exclude": True},

    # Belgium: tighter banding + VERY_STRONG only
    "Belgium Pro": {"strong_q": 0.85, "vstrong_q": 0.93, "signals": {"VERY_STRONG_OVER"}},

    # Championship tightening (non-elite premium use)
    "England Championship": {"strong_q": 0.85, "vstrong_q": 0.93},
}
```

---

## Over 2.5 Deploy Decision Function

This is the production-ready decision function plus the cache builder and a wrapper to create deploy-ready outputs.

### What it does
- Builds **per-league TopQ thresholds** (numeric p cutoffs).
- Builds **per-league signal thresholds** (for UI/debug) and uses **rank-percentile** for tie-safe banding.
- Applies:
  - elite whitelist
  - TopQ gate
  - derived signal gate
  - overrides (exclude, min_p, strong_q/vstrong_q, allowed signals)
- Returns:
  - `publish_over25` (bool)
  - `publish_reason` (human-readable)
  - `deploy_signal_over25` (derived)

> Minimum required columns in `preds`:
> - `league`
> - `model_p_for_bookie`
> - `bookie_pick` (or mapped alias)

---

### Code (drop-in)

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple, Union
import math
import pandas as pd

Number = Union[int, float]


ELITE_KEEP_LEAGUES_DEFAULT = {
    "Germany Bundesliga",
    "Netherlands Eredivisie",
    "Italy Serie A",
    "Portugal Liga",
    "Champions League",
    "USA MLS",
    "France Ligue 1",
    "Belgium Pro",
}

LEAGUE_OVERRIDES_DEFAULT: Dict[str, Dict[str, Any]] = {
    "Europa Conference": {"exclude": True},
    "Brazil Serie A": {"exclude": True},
    "Japan J1": {"exclude": True},

    "England Premier League": {"exclude": True},
    "Spain La Liga": {"exclude": True},
    "England EFL League 1": {"exclude": True},
    "England FA Cup": {"exclude": True},

    "Belgium Pro": {"strong_q": 0.85, "vstrong_q": 0.93, "signals": {"VERY_STRONG_OVER"}},
    "England Championship": {"strong_q": 0.85, "vstrong_q": 0.93},
}


@dataclass
class Over25ThresholdCache:
    topq_thr_by_league: Dict[str, float] = field(default_factory=dict)
    signal_thr_by_league: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def get_topq_thr(self, league: str) -> Optional[float]:
        return self.topq_thr_by_league.get(str(league).strip())

    def get_signal_thr(self, league: str) -> Optional[Tuple[float, float]]:
        return self.signal_thr_by_league.get(str(league).strip())


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None

def _norm_str(x: Any) -> str:
    try:
        return str(x).strip()
    except Exception:
        return ""

def _compute_quantile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    q = float(q)
    q = 0.0 if q < 0.0 else 1.0 if q > 1.0 else q
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w

def _percentile_rank(values_sorted: list[float], x: float) -> float:
    """
    Tie-safe percentile rank in [0,1], "average" method:
      rank = (count(<x) + 0.5*count(==x)) / n
    """
    n = len(values_sorted)
    if n == 0:
        return 0.0
    import bisect
    left = bisect.bisect_left(values_sorted, x)
    right = bisect.bisect_right(values_sorted, x)
    less = left
    equal = right - left
    return (less + 0.5 * equal) / n

def _default_signals_strict() -> set[str]:
    return {"STRONG_OVER", "VERY_STRONG_OVER"}


def build_over25_threshold_cache_from_df(
    preds: pd.DataFrame,
    *,
    league_col: str = "league",
    p_col: str = "model_p_for_bookie",
    topq_q: float = 0.80,
    strong_q_default: float = 0.80,
    vstrong_q_default: float = 0.90,
    league_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[Over25ThresholdCache, Dict[str, list[float]]]:
    """
    Build per-league:
      - TopQ threshold (numeric p cutoff)
      - signal thresholds (numeric p cutoffs, for UI/debug)
      - plus: sorted p distribution per league for percentile-rank signals (authoritative)

    Returns:
      (cache, p_sorted_by_league)
    """
    league_overrides = league_overrides or {}

    p_by_lg: Dict[str, list[float]] = {}
    for lg, p in zip(preds.get(league_col, []), preds.get(p_col, [])):
        lg_s = _norm_str(lg)
        p_f = _safe_float(p)
        if not lg_s or p_f is None:
            continue
        p_by_lg.setdefault(lg_s, []).append(p_f)

    cache = Over25ThresholdCache()
    p_sorted_by_lg: Dict[str, list[float]] = {}

    for lg, ps in p_by_lg.items():
        if not ps:
            continue
        ps_sorted = sorted(ps)
        p_sorted_by_lg[lg] = ps_sorted

        topq_thr = _compute_quantile(ps_sorted, topq_q)
        if topq_thr is not None:
            cache.topq_thr_by_league[lg] = float(topq_thr)

        cfg = league_overrides.get(lg, {})
        strong_q = float(cfg.get("strong_q", strong_q_default))
        vstrong_q = float(cfg.get("vstrong_q", vstrong_q_default))
        if vstrong_q < strong_q:
            vstrong_q = strong_q

        s_thr = _compute_quantile(ps_sorted, strong_q)
        vs_thr = _compute_quantile(ps_sorted, vstrong_q)
        if s_thr is not None and vs_thr is not None:
            if vs_thr < s_thr:
                vs_thr = s_thr
            cache.signal_thr_by_league[lg] = (float(s_thr), float(vs_thr))

    return cache, p_sorted_by_lg


def should_publish_over25(
    row: Mapping[str, Any],
    *,
    cache: Over25ThresholdCache,
    p_sorted_by_league: Dict[str, list[float]],
    deploy_profile: str = "elite",
    elite_keep_leagues: Optional[set[str]] = None,
    league_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    league_col: str = "league",
    p_col: str = "model_p_for_bookie",
    bookie_pick_col: str = "bookie_pick",
    topq_q: float = 0.80,
    strong_q_default: float = 0.80,
    vstrong_q_default: float = 0.90,
    strict_signals: bool = True,
) -> Tuple[bool, str, str]:
    """
    Returns (publish_bool, reason_string, derived_signal_over25).
    """
    elite_keep_leagues = elite_keep_leagues or set(ELITE_KEEP_LEAGUES_DEFAULT)
    league_overrides = league_overrides or dict(LEAGUE_OVERRIDES_DEFAULT)

    lg = _norm_str(row.get(league_col))
    if not lg:
        return False, "Rejected: missing league", "NEUTRAL"

    if str(deploy_profile).strip().lower() == "elite" and lg not in elite_keep_leagues:
        return False, f"Rejected: league not in elite whitelist ({lg})", "NEUTRAL"

    cfg = league_overrides.get(lg, {})
    if bool(cfg.get("exclude", False)):
        return False, f"Rejected: league excluded by override ({lg})", "NEUTRAL"

    pick = _norm_str(row.get(bookie_pick_col)).upper().replace(" ", "")
    if pick != "OVER25":
        return False, f"Rejected: not OVER25 (pick={pick or 'n/a'})", "NEUTRAL"

    p = _safe_float(row.get(p_col))
    if p is None:
        return False, "Rejected: missing model_p_for_bookie", "NEUTRAL"

    min_p = _safe_float(cfg.get("min_p"))
    if min_p is not None and p < min_p:
        return False, f"Rejected: p<{min_p:.3f} min_p override ({p:.3f})", "NEUTRAL"

    topq_thr = cache.get_topq_thr(lg)
    if topq_thr is None:
        return False, f"Rejected: no TopQ threshold cached for league ({lg})", "NEUTRAL"
    if p < topq_thr:
        return False, f"Rejected: below TopQ@q={topq_q:.2f} (p={p:.4f} < thr={topq_thr:.4f})", "NEUTRAL"

    ps_sorted = p_sorted_by_league.get(lg)
    if not ps_sorted:
        return False, f"Rejected: no p-distribution cached for league ({lg})", "NEUTRAL"

    strong_q = float(cfg.get("strong_q", strong_q_default))
    vstrong_q = float(cfg.get("vstrong_q", vstrong_q_default))
    if vstrong_q < strong_q:
        vstrong_q = strong_q

    rp = _percentile_rank(ps_sorted, p)

    if rp >= vstrong_q:
        sig = "VERY_STRONG_OVER"
    elif rp >= strong_q:
        sig = "STRONG_OVER"
    else:
        sig = "NEUTRAL"

    if strict_signals:
        allowed = cfg.get("signals", _default_signals_strict())
        if isinstance(allowed, (list, set, tuple)):
            allowed_set = {str(x).strip().upper() for x in allowed}
        else:
            allowed_set = {str(allowed).strip().upper()}
        if sig not in allowed_set:
            return False, f"Rejected: signal {sig} not allowed (allowed={sorted(allowed_set)})", sig

    bits = []
    bits.append("Elite league" if str(deploy_profile).strip().lower() == "elite" else "Premium league")
    bits.append(f"TopQ@{topq_q:.2f}")
    bits.append(sig)

    sig_thr = cache.get_signal_thr(lg)
    if sig_thr:
        s_thr, vs_thr = sig_thr
        bits.append(f"(sig_thr≈{s_thr:.4f}/{vs_thr:.4f})")

    bits.append(f"p={p:.4f}")
    bits.append(f"rank_pct={rp:.3f}")

    return True, " + ".join(bits), sig


def build_deploy_ready_over25_csv(
    preds: pd.DataFrame,
    *,
    out_csv_path: str,
    deploy_profile: str = "elite",
    elite_keep_leagues: Optional[set[str]] = None,
    league_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    league_col: str = "league",
    p_col: str = "model_p_for_bookie",
    bookie_pick_col: str = "bookie_pick",
    topq_q: float = 0.80,
    strong_q_default: float = 0.80,
    vstrong_q_default: float = 0.90,
    strict_signals: bool = True,
    only_published: bool = False,
) -> pd.DataFrame:
    """
    Adds:
      - publish_over25 (bool)
      - publish_reason (str)
      - deploy_signal_over25 (derived)
      - topq_thr_league (float)
    Writes a deploy-ready CSV for UI.
    """
    elite_keep_leagues = elite_keep_leagues or set(ELITE_KEEP_LEAGUES_DEFAULT)
    league_overrides = league_overrides or dict(LEAGUE_OVERRIDES_DEFAULT)

    cache, p_sorted_by_lg = build_over25_threshold_cache_from_df(
        preds,
        league_col=league_col,
        p_col=p_col,
        topq_q=topq_q,
        strong_q_default=strong_q_default,
        vstrong_q_default=vstrong_q_default,
        league_overrides=league_overrides,
    )

    publish_flags = []
    reasons = []
    signals = []
    topq_thrs = []

    for r in preds.to_dict("records"):
        lg = _norm_str(r.get(league_col))
        topq_thrs.append(cache.get_topq_thr(lg))

        ok, reason, sig = should_publish_over25(
            r,
            cache=cache,
            p_sorted_by_league=p_sorted_by_lg,
            deploy_profile=deploy_profile,
            elite_keep_leagues=elite_keep_leagues,
            league_overrides=league_overrides,
            league_col=league_col,
            p_col=p_col,
            bookie_pick_col=bookie_pick_col,
            topq_q=topq_q,
            strong_q_default=strong_q_default,
            vstrong_q_default=vstrong_q_default,
            strict_signals=strict_signals,
        )
        publish_flags.append(bool(ok))
        reasons.append(str(reason))
        signals.append(str(sig))

    out = preds.copy()
    out["publish_over25"] = publish_flags
    out["publish_reason"] = reasons
    out["deploy_signal_over25"] = signals
    out["topq_thr_league"] = topq_thrs

    if only_published:
        out = out[out["publish_over25"]].copy()

    out.to_csv(out_csv_path, index=False)
    return out
```

---

## Recommended Production Workflow

1) Generate predictions export for the deploy snapshot (per league / across leagues).
2) Build caches from that snapshot (`build_over25_threshold_cache_from_df`).
3) Apply `should_publish_over25` to each row:
   - sets `publish_over25`, `publish_reason`, and `deploy_signal_over25`
4) Write deploy-ready CSV and feed it to the UI.

This guarantees your thresholds are always coherent with the distribution you are ranking against.

---

## Notes / Next Steps

- Belgium is the only elite league that underperformed in the demonstrated slice (11 picks → 90.91%).
  - Keep as-is (VERY_STRONG only), or
  - raise `min_p` for Belgium, or
  - remove Belgium from elite whitelist for a “purist” elite lane.

- If you later want a **premium (non-elite)** lane:
  - loosen the whitelist, but maintain per-league TopQ and per-league banding, and use overrides to park weak leagues.

---

## Appendix: Quick “Deploy-ready CSV” Example

```python
import pandas as pd

preds = pd.read_csv(
  "predictions_output/ou25_frozen_compare/.../BOOKIE_IMP40_...__OU25_COMBINED_TOPQ_080.csv",
  low_memory=False
)

df_deploy = build_deploy_ready_over25_csv(
  preds,
  out_csv_path="predictions_output/ou25_deploy_ready_elite.csv",
  deploy_profile="elite",
  topq_q=0.80,
  strong_q_default=0.80,
  vstrong_q_default=0.90,
  strict_signals=True,
  only_published=False,   # True if UI should only receive published rows
)

print(df_deploy["publish_over25"].value_counts(dropna=False))
```
