1) There are two different truths: baseline vs deploy-quality

Baseline lane (wide coverage, not a “premium” product)
	•	You scored all OVER25 picks across many leagues, no premium gating.
	•	Result: ~85.66% overall accuracy on 1834 picks.
	•	Reality: this is monitoring / broad engine health, not the product you sell.

Deploy-quality lane (what you actually want users to see)

Your best-performing deploy profile is:

Elite whitelist + per-league TopQ + per-league quantile banding signals (derived)
	•	Result: ~98.6% accuracy on ~217 picks (scored 217/217).
	•	This is the “forward-facing” lane.

Key insight:
Signal labels must be recomputed from model_p_for_bookie, not trusted from old artifacts.
(Otherwise you can accidentally evaluate stale/mis-tagged signals.)

⸻

2) The deploy logic that works (and why)

To publish an Over 2.5 pick, you need all 3 gates:

Gate A — League eligibility (your product “lane”)

You proved that league choice dominates outcome reliability.

Elite keep-leagues (hard whitelist):
	•	Germany Bundesliga
	•	Netherlands Eredivisie
	•	Italy Serie A
	•	Portugal Liga
	•	Champions League
	•	USA MLS
	•	France Ligue 1
	•	Belgium Pro (but it’s the weak link)

Everything else is either excluded or “watchlist/conditional”.

Gate B — TopQ by league (probability must be top-tier within that league)
	•	You’re using TopQ = 0.80 per league.
	•	Meaning: only keep the top ~20% of model_p_for_bookie values inside each league.

This makes the system stationary by league (no global threshold that breaks across leagues).

Gate C — Signal strength by league (STRONG / VERY_STRONG are relative to league)

You now define signals via per-league percentile rank (tie-safe), not absolute p-values.

Default elite banding:
	•	strong_q = 0.80
	•	vstrong_q = 0.90

So:
	•	VERY_STRONG_OVER = top 10% of p inside league
	•	STRONG_OVER = next 10% inside league
	•	NEUTRAL = everything else

And you deploy only:
STRONG_OVER or VERY_STRONG_OVER (strict mode)

⸻

3) What the numbers told us about reliability

Baseline (broad engine)
	•	85.66% is real, but it includes leagues where your edge is weak or the market is hostile.

Elite deploy (forward-facing)
	•	~98.6% is real under your chosen rules.
	•	But it’s achieved by:
	•	restricting to elite leagues
	•	taking only the top probability mass per league
	•	only publishing STRONG/VERY_STRONG signals

Belgium Pro special case (important learning)

Belgium is the only elite league that still drags.

You confirmed the override works:
	•	Belgium picks become VERY_STRONG only
	•	and you still got 90.9% in that slice (11 picks)

So forward-facing you have two options:
	•	Option 1 (purist elite): remove Belgium from elite whitelist.
	•	Option 2 (keep Belgium but stricter): add Belgium min_p (e.g. 0.90+) so it only appears when it’s absolutely nailed.

⸻

4) The “deployment thresholds” you should think in

When you deploy Over 2.5, don’t think “p must be 0.78 globally”.

Think in this hierarchy:

Level 0 — Baseline (internal only)
	•	Any league allowed
	•	No TopQ
	•	Accuracy ~85%

Level 1 — Premium (sellable, wider than elite)
	•	Exclude known-bad leagues
	•	TopQ per league (0.80)
	•	STRONG/VERY_STRONG per league
	•	Accuracy tends toward mid/high 90s depending on whitelist

Level 2 — Elite (front-facing “flagship”)
	•	Hard whitelist of elite leagues
	•	TopQ per league (0.80)
	•	Signal banding per league (0.80/0.90)
	•	Strict signals only
	•	Accuracy ~98–100% range in this validation window

⸻

5) The forward-facing deploy rulebook (simple)

A match gets an “Over 2.5” recommendation only if:
	1.	League is in ELITE_KEEP_LEAGUES
	2.	bookie_pick == OVER25 (as you’re evaluating OVER-only lane right now)
	3.	model_p_for_bookie is in Top 20% for that league
	4.	Signal recomputed per league and is either:
	•	VERY_STRONG_OVER (top 10%), or
	•	STRONG_OVER (80–90% band)
	5.	Apply overrides:
	•	Belgium: VERY_STRONG only (+ optional min_p)
	•	Championship: only if you’re running non-elite premium, and tighten strong_q/vstrong_q

That’s it. That’s your production standard.


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union
import math
import pandas as pd

Number = Union[int, float]


# ----------------------------
# Config (Elite default)
# ----------------------------

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

    # Belgium: tighten + VERY_STRONG only
    "Belgium Pro": {"strong_q": 0.85, "vstrong_q": 0.93, "signals": {"VERY_STRONG_OVER"}},

    # Championship tightening (only matters in non-elite profiles)
    "England Championship": {"strong_q": 0.85, "vstrong_q": 0.93},
}


# ----------------------------
# Threshold caches
# ----------------------------

@dataclass
class Over25ThresholdCache:
    topq_thr_by_league: Dict[str, float] = field(default_factory=dict)
    signal_thr_by_league: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def get_topq_thr(self, league: str) -> Optional[float]:
        return self.topq_thr_by_league.get(str(league).strip())

    def get_signal_thr(self, league: str) -> Optional[Tuple[float, float]]:
        return self.signal_thr_by_league.get(str(league).strip())


# ----------------------------
# Helpers
# ----------------------------

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


# ----------------------------
# Cache builder (from the SAME distribution you rank against)
# ----------------------------

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

    # Collect p values by league
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

        # TopQ cutoff
        topq_thr = _compute_quantile(ps_sorted, topq_q)
        if topq_thr is not None:
            cache.topq_thr_by_league[lg] = float(topq_thr)

        # Per-league banding overrides (defaults + overrides)
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


# ----------------------------
# Per-row decision + derived signal
# ----------------------------

def should_publish_over25(
    row: Mapping[str, Any],
    *,
    cache: Over25ThresholdCache,
    p_sorted_by_league: Dict[str, list[float]],
    deploy_profile: str = "elite",  # "elite" or "premium"
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

    Signals are derived via percentile rank within league (tie-safe),
    using p_sorted_by_league built from the same deployment distribution.
    """
    elite_keep_leagues = elite_keep_leagues or set(ELITE_KEEP_LEAGUES_DEFAULT)
    league_overrides = league_overrides or dict(LEAGUE_OVERRIDES_DEFAULT)

    lg = _norm_str(row.get(league_col))
    if not lg:
        return False, "Rejected: missing league", "NEUTRAL"

    # Elite whitelist
    if str(deploy_profile).strip().lower() == "elite" and lg not in elite_keep_leagues:
        return False, f"Rejected: league not in elite whitelist ({lg})", "NEUTRAL"

    # Overrides: exclude
    cfg = league_overrides.get(lg, {})
    if bool(cfg.get("exclude", False)):
        return False, f"Rejected: league excluded by override ({lg})", "NEUTRAL"

    # OVER-only lane
    pick = _norm_str(row.get(bookie_pick_col)).upper().replace(" ", "")
    if pick != "OVER25":
        return False, f"Rejected: not OVER25 (pick={pick or 'n/a'})", "NEUTRAL"

    # Probability required
    p = _safe_float(row.get(p_col))
    if p is None:
        return False, "Rejected: missing model_p_for_bookie", "NEUTRAL"

    # Overrides: min_p
    min_p = _safe_float(cfg.get("min_p"))
    if min_p is not None and p < min_p:
        return False, f"Rejected: p<{min_p:.3f} min_p override ({p:.3f})", "NEUTRAL"

    # TopQ gate
    topq_thr = cache.get_topq_thr(lg)
    if topq_thr is None:
        return False, f"Rejected: no TopQ threshold cached for league ({lg})", "NEUTRAL"
    if p < topq_thr:
        return False, f"Rejected: below TopQ@q={topq_q:.2f} (p={p:.4f} < thr={topq_thr:.4f})", "NEUTRAL"

    # Derived signal (percentile-rank stationary by league)
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

    # Strict signal gating
    if strict_signals:
        allowed = cfg.get("signals", _default_signals_strict())
        if isinstance(allowed, (list, set, tuple)):
            allowed_set = {str(x).strip().upper() for x in allowed}
        else:
            allowed_set = {str(allowed).strip().upper()}
        if sig not in allowed_set:
            return False, f"Rejected: signal {sig} not allowed (allowed={sorted(allowed_set)})", sig

    # Reason string (UI)
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


# ----------------------------
# Drop-in wrapper
# ----------------------------

def build_deploy_ready_over25_csv(
    preds: pd.DataFrame,
    *,
    out_csv_path: str,
    deploy_profile: str = "elite",
    elite_keep_leagues: Optional[set[str]] = None,
    league_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    # columns
    league_col: str = "league",
    p_col: str = "model_p_for_bookie",
    bookie_pick_col: str = "bookie_pick",
    # gates
    topq_q: float = 0.80,
    strong_q_default: float = 0.80,
    vstrong_q_default: float = 0.90,
    strict_signals: bool = True,
    # output controls
    only_published: bool = False,
) -> pd.DataFrame:
    """
    - takes a pandas DataFrame preds
    - builds the caches (from preds distribution)
    - adds columns:
        publish_over25 (bool)
        publish_reason (str)
        deploy_signal_over25 (str, derived)
        topq_thr_league (float)
    - optionally filters to only published
    - writes a deploy-ready CSV for the UI
    - returns the augmented DataFrame
    """
    if preds is None or preds.empty:
        raise ValueError("preds is empty")

    elite_keep_leagues = elite_keep_leagues or set(ELITE_KEEP_LEAGUES_DEFAULT)
    league_overrides = league_overrides or dict(LEAGUE_OVERRIDES_DEFAULT)

    # Build caches from THIS preds distribution (critical)
    cache, p_sorted_by_lg = build_over25_threshold_cache_from_df(
        preds,
        league_col=league_col,
        p_col=p_col,
        topq_q=topq_q,
        strong_q_default=strong_q_default,
        vstrong_q_default=vstrong_q_default,
        league_overrides=league_overrides,
    )

    # Apply decision per row (fast enough for a few k rows; if huge, we can vectorize later)
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


# ----------------------------
# Example CLI-style usage
# ----------------------------
"""
import pandas as pd

preds = pd.read_csv("predictions_output/.../BOOKIE_IMP40_...__OU25_COMBINED_TOPQ_080.csv", low_memory=False)

df_deploy = build_deploy_ready_over25_csv(
    preds,
    out_csv_path="predictions_output/ou25_deploy_ready_elite.csv",
    deploy_profile="elite",
    topq_q=0.80,
    strong_q_default=0.80,
    vstrong_q_default=0.90,
    strict_signals=True,
    only_published=False,   # set True if you want the UI file to contain only published rows
)

print(df_deploy["publish_over25"].value_counts(dropna=False))
print(df_deploy.loc[df_deploy["league"].eq("Belgium Pro"), "deploy_signal_over25"].value_counts(dropna=False))
"""


