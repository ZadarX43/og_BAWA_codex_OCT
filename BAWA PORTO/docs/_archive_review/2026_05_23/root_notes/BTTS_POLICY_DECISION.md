# BTTS Policy Decision

## Quick deploy note

- **Primary lane =** BTTS model live primary
- **Shadow lane =** explicit BTTS ValueEV rows, plus review leagues and approved near-miss aligned BTTS ValueEV rows tracked in shadow buckets
- **Promotion rule =** BTTS ValueEV is not broad-live; only explicit aligned high-conviction ValueEV rows may be promoted to `STANDARD`, and only when strict side-aware runtime whitelist conditions are met
- **Demotion rule =** any explicit BTTS ValueEV row not meeting whitelist-promotion conditions is forced out of live tiers into `OBSERVE`; league-level demotion still applies on refreshed audit if ROI / stability worsens

---

## Purpose

This document is the compact deploy-note counterpart to `BTTS_WALKFORWARD_SUMMARY.md`.

It locks the current BTTS routing posture into a deploy-legible policy that can be:
- enforced in `deploy_rulebook.py`
- expressed as routing artifacts (`btts_league_policy.json`, `btts_live_whitelist.csv`, `btts_shadow_review.csv`)
- monitored via shadow buckets without weakening the live lane

---

## One-line deploy posture

Keep **BTTS model as the live primary lane**. Route explicit BTTS ValueEV rows to **shadow/watch by default**, allow only the strict side-aware whitelist-promotion subset into **STANDARD** (not broad live), blacklist FA Cup, and hold insufficient-evidence leagues out until more walk-forward months exist.

---

## Policy buckets

### Live
Must satisfy (current rules):
- months_present >= 3
- total_rows >= 20
- weighted_roi >= 0.45
- losing_months == 0

### Review
Leagues that do not meet Live, Blacklist, or Insufficient Evidence.

### Blacklist
- weighted_roi < 0.15 with sufficient evidence

### Insufficient evidence
- months_present < 3 OR total_rows < 20

These policy buckets define the current BTTS **league-policy routing posture**. They do not, by themselves, grant broad BTTS ValueEV live routing in runtime.

---

## Live whitelist

BTTS ValueEV candidate whitelist (league-qualified for policy review, but **not** automatic live promotion):
- France Ligue 1
- Germany Bundesliga
- Portugal Liga
- Italy Serie A
- England Championship
- Norway Eliteserien
- Netherlands Eredivisie
- Japan J1
- Europa League
- Europa Conference

Notes:
- These leagues meet the current BTTS ValueEV **league-policy** Live rules from `btts_league_policy.json`.
- This does **not** create broad ValueEV live routing.
- The runtime code keeps `BTTS_VALUEEV_ALLOWED_LIVE = False` and blocks explicit BTTS ValueEV rows from live tiers by default.
- Only the strict whitelist-promotion subset may enter `STANDARD` at runtime.
- Live routing should remain league-aware; do not run BTTS as an all-league product.

---

## Review leagues

Shadow / reduced-trust routing (current locked list):
- USA MLS
- Belgium Pro
- Brazil Serie A
- Spain La Liga

Reasons these stay out of Live (current locked interpretation):
- USA MLS: strong ROI but has a losing month, fails the Live rule
- Belgium Pro: instability and losing months
- Brazil Serie A: weaker and more volatile
- Spain La Liga: weakest edge among review set

Routing:
- keep these leagues in ValueEV shadow/review only
- allow these leagues into OBSERVE tiers only
- track them in shadow buckets for continued month-by-month evidence

---

## Blacklist

- England FA Cup

Routing:
- block from live BTTS deployment
- allow only if explicitly running a research-only job

---

## Insufficient evidence

- Scotland Premiership
- Champions League
- England Premier League

Notes:
- EPL is not labeled “bad” here; it is under-sampled and unstable under current rules.

Routing:
- treat as holdout until thresholds are met (months_present >= 3 and total_rows >= 20)

---

## Runtime stamping rules

BTTS rows must carry runtime metadata after the BTTS split:
- `btts_alignment`
- `signal_btts_runtime`

Runtime interpretation:
- aligned rows (`selection == model_top_pick`) are stamped `ALIGNED`
- contrarian rows are stamped `CONTRARIAN`
- contrarian rows have `signal_btts_runtime = NEUTRAL`
- aligned rows keep the side-aware runtime label, preferring `signal_btts_side` and falling back to `signal_btts`

This rule is mandatory for clean tiering and explainable deployment output.

---

## BTTS YES / NO runtime calibration

The deploy code is side-specific. The current runtime BTTS **model** lane is calibrated as follows.

### BTTS YES runtime gates
Core runtime defaults in `deploy_rulebook.py`:
- `btts_implied_max = 0.72`
- `btts_yes_pmin_floor = 0.55`
- strong-label gate requires `STRONG_YES` or `VERY_STRONG_YES`
- FTS / blank-risk gating requires both sides below the YES blank-risk threshold
- explicit YES structural checks can use:
  - `btts_yes_scored_min = 0.70`
  - `btts_yes_cs_max = 0.35`
  - `btts_yes_conceded_min = 0.60`
  - `btts_yes_bttsrate_min = 0.55`
  - `btts_yes_xgfor_min = 1.10`
  - `btts_yes_xgagainst_min = 1.00`
- label-aware confirmation then decides whether the row survives as live model BTTS YES

### BTTS NO runtime gates
Core runtime defaults in `deploy_rulebook.py`:
- `btts_no_pmin_floor = 0.52`
- strong-label gate requires `STRONG_NO` or `VERY_STRONG_NO`
- blank-plausibility gate requires at least one meaningful FTS / blanking pathway
- explicit NO structural checks can use:
  - `btts_no_cs_min = 0.35`
  - `btts_no_scored_max = 0.70`
  - `btts_no_conceded_max` from current params
  - `btts_no_bttsrate_max` from current params
  - `btts_no_xgfor_max = 1.00`
  - `btts_no_xgagainst_max = 1.00`
- low-goaliness and clean-sheet / blanking confirmation then decide whether the row survives as live model BTTS NO

Interpretation:
- the live BTTS **model** lane is not driven by one generic BTTS threshold
- YES and NO are filtered differently
- any permanent BTTS policy doc should reflect the side split rather than describing BTTS as one monolithic gate

## ValueEV promotion policy

Goal: define the conditions under which explicit BTTS ValueEV rows may be promoted into `STANDARD`, without allowing unsafe ValueEV rows to leak into live tiers.

Important current lock from `deploy_rulebook.py`:
- `BTTS_LIVE_POLICY = "model_plus_whitelist_valueev"`
- `BTTS_VALUEEV_ALLOWED_LIVE = False`
- explicit BTTS ValueEV rows are blocked from live tiers by default
- eligible whitelist rows may still be promoted into `STANDARD`, not broad ELITE/live

This is the critical distinction:
1. `bookie_allmarkets.py` generates upstream BTTS ValueEV candidate rows.
2. `btts_league_policy.json` classifies leagues into live / review / blacklist / insufficient-evidence policy buckets.
3. `deploy_rulebook.py` still blocks explicit BTTS ValueEV rows from broad live tiers, and only promotes the strict whitelist subset into `STANDARD`.

Current whitelist-promotion conditions are materially stricter than the older walk-forward discovery note implied.

A BTTS ValueEV row is eligible for `STANDARD` whitelist promotion only when it is:
- `market == btts`
- explicitly detected as a BTTS ValueEV row
- `btts_alignment == ALIGNED`
- runtime label matches the side whitelist:
  - NO: `VERY_STRONG_NO`
  - YES: `VERY_STRONG_YES`
- side-specific promotion thresholds are met:
  - NO: `edge >= 0.21` and `model_p_for_bookie >= 0.71`
  - YES: `edge >= 0.24` and `model_p_for_bookie >= 0.74`

Interpretation:
- the old discovery note about `od_min = 1.33` and `edge_min = 1.02` belongs to the upstream BTTS ValueEV candidate-generation layer in `bookie_allmarkets.py`
- it is **not** the final live-promotion rule used in `deploy_rulebook.py`
- live documentation must distinguish:
  1. upstream BTTS ValueEV candidate generation in `bookie_allmarkets.py`
  2. runtime BTTS model filtering in `deploy_rulebook.py`
  3. runtime BTTS ValueEV whitelist promotion in `deploy_rulebook.py`

Until policy is explicitly upgraded, non-eligible BTTS ValueEV rows route to `OBSERVE` / shadow.

## Shadow bucket policy

Purpose:
- prevent profitable near-miss aligned BTTS ValueEV rows from being discarded
- measure profit leakage against the strict ValueEV whitelist-promotion baseline while the model remains the live primary lane
- only widen whitelist-promotion rules after shadow evidence is strong at meaningful sample size

Current operational rule:
- keep runtime BTTS ValueEV whitelist-promotion policy strict
- select near-miss aligned ValueEV rows into a shadow bucket
- audit shadow bucket hit/ROI vs the strict whitelist-promotion baseline
- do not confuse shadow-bucket success with automatic permission for broad live promotion

Locked evidence (March 2025 audit):
- shadow micro-bucket: 3 resolved / 3 wins (100.00%)
- live whitelist baseline: 53 resolved / 52 wins (98.11%)
- unresolved rows are settlement/normalization issues, not counted as losses

Truth source for deploy-audit settlement:
- build resolver from `Matches/__merged__/*__merged.csv`
- use `predictions_output/BTTS_TAG_TEST/BTTS_RESULTS_RESOLVER_FROM_MERGED_2025-03.csv` style outputs

---

## Promotion and demotion conditions

### Promotion to league-policy Live
A league can move from Review to Live in the league-policy artifact when it satisfies:
- months_present >= 3
- total_rows >= 20
- weighted_roi >= 0.45
- losing_months == 0

This is a **league-policy** promotion only. It does not, by itself, create broad BTTS ValueEV live routing.

### Promotion to runtime STANDARD whitelist
An explicit BTTS ValueEV row can be promoted into runtime `STANDARD` only when all of the following are true:
- the league is acceptable under current policy posture
- the row is explicitly ValueEV
- the row is `ALIGNED`
- the runtime label is side-whitelisted (`VERY_STRONG_YES` or `VERY_STRONG_NO`)
- side-specific `edge` and `model_p_for_bookie` thresholds are met

### Demotion from league-policy Live
A league must be demoted from league-policy Live to Review if either:
- it records any losing month in the locked window, OR
- weighted_roi falls below 0.45 on a refreshed audit, OR
- max drawdown breaches future explicit stability rules if/when formalised

### Runtime demotion / blocking
Any explicit BTTS ValueEV row that fails whitelist-promotion conditions is blocked from live tiers and routed to `OBSERVE`.

### Blacklist trigger
A league should be blacklisted when:
- it has sufficient evidence AND
- weighted_roi < 0.15

---

## Locked deployment recommendation

1) Keep BTTS model as the live primary lane
- do not claim broad BTTS ValueEV live promotion
- keep the policy aligned with runtime code: explicit ValueEV rows are blocked from live tiers unless they qualify for strict `STANDARD` whitelist promotion

2) Use league-aware routing for BTTS ValueEV policy posture
- whitelist leagues = strongest candidate pool for ValueEV policy review
- review leagues = shadow/research only
- blacklist FA Cup
- hold out insufficient-evidence leagues
- remember that league-policy whitelist status alone does not override runtime blocking of explicit ValueEV rows

3) Keep BTTS ValueEV promotion strict and side-aware
- only aligned `VERY_STRONG_YES` / `VERY_STRONG_NO` ValueEV rows can be promoted
- require side-specific `edge` and `model_p_for_bookie` thresholds
- promotion is currently into `STANDARD`, not broad ELITE/live
- route near-miss aligned rows into shadow buckets

4) Make truth-source settlement explicit in audits
- use merged match resolver for deploy audit settlement

5) Next validation sequence
- split BTTS YES vs BTTS NO formally in docs and audits
- rerun model-vs-filtered head-to-head at better sample size
- expand shadow buckets gradually before changing whitelist-promotion thresholds

---

## Current locked conclusion

- league-aware routing is required
- BTTS model remains the live primary lane at this time
- BTTS ValueEV league-policy whitelist identifies the strongest candidate pool, but does **not** grant broad live routing by itself
- explicit BTTS ValueEV rows are blocked from broad live tiers unless they pass strict aligned side-aware whitelist-promotion rules
- whitelist-passing ValueEV rows may be promoted into `STANDARD`, not broad ELITE/live
- review leagues remain shadow / research lanes
- blacklist league remains off
- shadow-bucket auditing is now part of the BTTS profit-leak detection process
- next priority is BTTS YES / NO split analysis and refreshed model-vs-filtered comparison

---

## Related files

- `BTTS_WALKFORWARD_SUMMARY.md`
- `deploy_rulebook.py`
- `bookie_allmarkets.py`
- `deploy_gates.py`
- `btts_league_policy.json`
- `btts_live_whitelist.csv`
- `btts_shadow_review.csv`
- `audit_btts_shadow_bucket.py`
- `predictions_output/BTTS_TAG_TEST/BTTS_SHADOW_AUDIT_SUMMARY.md`
- `predictions_output/BTTS_TAG_TEST/BTTS_RESULTS_RESOLVER_FROM_MERGED_2025-03.csv`