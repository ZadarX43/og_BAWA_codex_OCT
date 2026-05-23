BTTS_YES_POLICY_DECISION.md
# BTTS YES Policy Decision

## Quick deploy note

- **Primary lane =** BTTS model live primary
- **Shadow lane =** BTTS ValueEV YES candidate whitelist plus YES-side review leagues and approved near-miss aligned BTTS YES ValueEV rows
- **Promotion rule =** BTTS YES ValueEV stays shadow/watch unless refreshed head-to-head evidence supports live promotion and the league also passes whitelist rules; narrow STANDARD promotion may still occur for explicitly tagged aligned high-conviction YES rows
- **Demotion rule =** any losing month, sub-0.45 weighted ROI on refreshed YES-side audit, adverse head-to-head evidence, or future stability breach if max-drawdown rule is formalised

---

## Purpose

This document is the compact deploy-note counterpart to `BTTS_YES_WALKFORWARD_SUMMARY.md`.

It locks the current BTTS YES routing posture into a deploy-legible policy that can be:
- enforced alongside `deploy_rulebook.py`
- expressed through league-policy artifacts and directional BTTS audit outputs
- monitored via shadow buckets without weakening the live BTTS model lane

---

## One-line deploy posture

Keep **BTTS model as the live primary lane**. Treat BTTS YES ValueEV as a **candidate / shadow-watch lane**, using YES-side whitelist and review buckets as routing evidence rather than automatic live promotion; only narrowly promote explicitly tagged aligned high-conviction YES ValueEV rows into `STANDARD` when the runtime whitelist conditions are met.

---

## Policy buckets

### Live
Must satisfy current YES-side walk-forward routing rules:
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

---

## Live whitelist

BTTS YES ValueEV candidate whitelist (league-qualified, but **not** automatic live promotion):
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
- These leagues represent the strongest current YES-side BTTS ValueEV candidate pool where the broader BTTS policy file also supports whitelist status.
- This does **not** override the current live-policy lock in code: BTTS model remains live primary; BTTS ValueEV is blocked from live tiers by default and only eligible for narrow `STANDARD` promotion.
- Do not treat the YES whitelist as blanket ELITE/live approval.

---

## Review leagues

YES-side shadow / reduced-trust routing:
- USA MLS
- Belgium Pro
- Brazil Serie A
- Spain La Liga

Routing:
- keep these leagues in YES-side ValueEV shadow/review only
- allow them into OBSERVE tiers only unless future policy evidence upgrades them
- track them in shadow buckets for continued month-by-month evidence

---

## Blacklist

- England FA Cup

Routing:
- block from BTTS YES live deployment
- allow only in explicit research-only jobs

---

## Insufficient evidence

- Scotland Premiership
- Champions League
- England Premier League

Notes:
- England Premier League is under-sampled / unstable here, not automatically “bad”.
- Treat all insufficient-evidence YES leagues as holdout lanes until thresholds are met.

---

## Runtime stamping rules

BTTS YES rows must carry runtime metadata after the BTTS split:
- `btts_alignment`
- `signal_btts_runtime`

Runtime interpretation:
- aligned rows keep side-aware YES runtime strength
- contrarian rows are downgraded to `NEUTRAL`
- only aligned BTTS YES rows are even eligible for narrow ValueEV whitelist promotion

This is mandatory for clean YES-side tiering and explainable deployment output.

---

## Runtime YES thresholds reflected from code

Current runtime policy in `deploy_rulebook.py` relevant to BTTS YES:
- `btts_yes_pmin_floor = 0.55`
- `btts_implied_max = 0.72`
- default YES runtime labels required: `STRONG_YES` or `VERY_STRONG_YES`
- FTS safety gate: both sides should avoid high blank risk
- confirmation logic prefers strong structural support when YES-side rate / xG columns exist

Narrow BTTS YES ValueEV `STANDARD` promotion currently requires explicit whitelist-style conditions in code:
- runtime label in `{"VERY_STRONG_YES"}`
- `edge >= 0.24`
- `model_p_for_bookie >= 0.74`
- explicit BTTS ValueEV tagging
- `btts_alignment == ALIGNED`

This is stricter than the broad walk-forward league whitelist and must remain described separately.

---

## ValueEV promotion policy

Goal: define the conditions under which BTTS YES ValueEV can later be promoted without allowing unsafe YES rows to leak live prematurely.

Any future promotion into `STANDARD` is allowed only when the row is:
- market == btts
- YES-side selection
- explicitly tagged as BTTS ValueEV
- aligned (`btts_alignment == ALIGNED`)
- runtime label strong enough under the current whitelist promotion rule
- edge meets current YES promotion floor
- model probability meets current YES promotion floor
- league is in the YES-side candidate whitelist
- refreshed head-to-head / policy evidence supports maintaining that promotion posture

Until broader policy is explicitly upgraded, BTTS YES ValueEV remains shadow/watch by default.

---

## Shadow bucket policy

Purpose:
- prevent profitable near-miss aligned BTTS YES ValueEV rows from being discarded blindly
- measure profit leakage against the stricter YES-side candidate baseline while the BTTS model remains live primary
- only widen YES-side promotion rules after shadow evidence is strong at meaningful sample size

Current operational rule:
- keep live BTTS model policy strict
- keep BTTS YES ValueEV in shadow/watch by default
- route near-miss aligned YES rows into shadow buckets
- compare shadow performance with the candidate whitelist baseline before changing promotion logic

---

## Promotion and demotion conditions

### Promotion to stronger status
A YES-side league can move from Review to candidate-whitelist status when it satisfies:
- months_present >= 3
- total_rows >= 20
- weighted_roi >= 0.45
- losing_months == 0

A YES-side ValueEV row can move from shadow/watch into narrow `STANDARD` only when the explicit runtime whitelist conditions in code are satisfied.

### Demotion from candidate-whitelist status
A YES-side league should be demoted when either:
- it records any losing month in the locked window, OR
- weighted_roi falls below 0.45 on a refreshed YES-side audit, OR
- adverse head-to-head / policy evidence weakens the case for promotion, OR
- future max-drawdown stability rules are breached once formalised

### Blacklist trigger
A YES-side league should be blacklisted when:
- it has sufficient evidence AND
- weighted_roi < 0.15

---

## Locked deployment recommendation

1) Keep BTTS model as the live primary lane
- do not describe BTTS YES ValueEV as universally live
- keep the code-aligned policy language: model live primary, ValueEV shadow/watch by default

2) Use league-aware routing for BTTS YES ValueEV shadow/watch
- whitelist leagues = strongest YES-side candidate pool
- review leagues = shadow/research only
- blacklist FA Cup
- hold out insufficient-evidence leagues

3) Keep YES-side ValueEV promotion strict and conditional
- only consider narrow `STANDARD` promotion for aligned, explicitly tagged, high-conviction YES rows
- require both runtime whitelist thresholds and supporting policy evidence
- keep near-miss aligned YES rows in shadow buckets

4) Continue YES-side directional validation
- refresh YES-only walk-forward league audit
- expand YES-side shadow sample
- rerun model-vs-ValueEV comparison at larger sample size before any wider policy upgrade

---

## Current locked conclusion

- league-aware routing is required for BTTS YES ValueEV
- BTTS model remains the live primary lane
- BTTS YES ValueEV whitelist leagues are the strongest candidate / shadow-watch pool, not automatic live routing
- review leagues remain shadow / research lanes
- blacklist league remains off
- shadow-bucket auditing remains part of YES-side profit-leak detection
- wider YES-side promotion should only follow refreshed directional and head-to-head evidence

---

## Related files

- `BTTS_YES_WALKFORWARD_SUMMARY.md`
- `BTTS_POLICY_DECISION.md`
- `BTTS_MODEL_VS_VALUEEV_POLICY_NOTE.md`
- `btts_league_policy.json`
- `btts_live_whitelist.csv`
- `btts_shadow_review.csv`