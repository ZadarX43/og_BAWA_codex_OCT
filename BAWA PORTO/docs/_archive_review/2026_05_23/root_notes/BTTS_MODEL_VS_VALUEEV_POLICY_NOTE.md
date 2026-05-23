BTTS_MODEL_VS_VALUEEV_POLICY_NOTE.md
# BTTS Model vs ValueEV Policy Note

## Quick deploy note

- **Primary live lane =** BTTS model
- **ValueEV posture =** shadow/watch by default
- **Whitelist meaning =** strongest candidate pool for controlled ValueEV review, not automatic live approval
- **Promotion rule =** only explicitly tagged BTTS ValueEV rows that are aligned, side-qualified, and pass the stricter STANDARD promotion thresholds may be promoted into `STANDARD`
- **Blocked live rule =** explicit BTTS ValueEV rows are forced out of live tiers unless they satisfy the whitelist-style STANDARD promotion path

---

## Purpose

This note locks the current policy interpretation of the BTTS model-vs-ValueEV comparison work.

It is the compact counterpart to:
- `BTTS_WALKFORWARD_SUMMARY.md`
- `BTTS_POLICY_DECISION.md`
- `BTTS_YES_WALKFORWARD_SUMMARY.md`
- `BTTS_NO_WALKFORWARD_SUMMARY.md`

Its job is to make one point unambiguous:

**the current codebase does not support broad BTTS ValueEV live deployment.**

Instead, the current runtime posture is:
- BTTS model remains the live primary lane
- BTTS ValueEV remains shadow/watch by default
- only a narrow subset of aligned, high-conviction BTTS ValueEV rows can be promoted into `STANDARD`

---

## Locked policy position

The head-to-head comparison must now be interpreted in the context of the live runtime code, not just the walk-forward league-policy artifacts.

That means:
- `btts_league_policy.json` identifies league-quality routing context for BTTS ValueEV
- but `deploy_rulebook.py` still keeps explicit BTTS ValueEV rows out of live tiers by default
- the only exception is the narrow STANDARD-promotion path for aligned, high-conviction BTTS ValueEV rows

So the correct locked posture is:

- **BTTS model = live primary**
- **BTTS ValueEV = shadow/watch candidate lane**
- **BTTS ValueEV whitelist = routing context, not blanket live approval**

---

## What the comparison file is for

The model-vs-ValueEV comparison exists to answer a narrower question:

> In the leagues and months where both BTTS model and BTTS ValueEV are present, does ValueEV show enough evidence to justify future promotion?

That is a **research and promotion-governance** question.

It is **not** the same as saying:
- ValueEV is already the main BTTS live lane
- ValueEV should replace the BTTS model lane today
- all whitelist leagues should auto-route BTTS ValueEV live

Those statements are too strong for the present code state.

---

## Current head-to-head read

The existing comparison output should be read cautiously.

Observed pattern:
- some shared-month league slices show BTTS ValueEV outperforming the model on ROI
- several rows are `no_call`
- a large part of the table is still marked `not_comparable`
- evidence remains sparse in many league-month intersections

So the current comparison supports:
- continued BTTS ValueEV tracking
- selective whitelist/watch treatment
- future promotion research

It does **not** yet support:
- broad ValueEV live promotion
- replacing the BTTS model as the primary live lane
- treating the walk-forward whitelist as a direct live-routing permission set

---

## Runtime code alignment

### Live policy lock

Current runtime policy in `deploy_rulebook.py`:
- `BTTS_LIVE_POLICY = "model_plus_whitelist_valueev"`
- `BTTS_VALUEEV_ALLOWED_LIVE = False`
- `BTTS_VALUEEV_STANDARD_PROMOTION_ENABLED = True`

This means:
- explicit BTTS ValueEV rows are blocked from live tiers by default
- blocked ValueEV rows are routed to `OBSERVE`
- only a narrow qualifying subset may be re-promoted into `STANDARD`

### STANDARD promotion thresholds

Current strict promotion thresholds are side-specific:

#### BTTS YES ValueEV promotion
- runtime label must be `VERY_STRONG_YES`
- `edge >= 0.24`
- `model_p_for_bookie >= 0.74`
- row must be explicitly tagged as BTTS ValueEV
- row must be `ALIGNED`

#### BTTS NO ValueEV promotion
- runtime label must be `VERY_STRONG_NO`
- `edge >= 0.21`
- `model_p_for_bookie >= 0.71`
- row must be explicitly tagged as BTTS ValueEV
- row must be `ALIGNED`

This is the operative live rule.

So when this note talks about ValueEV “promotion,” it means:
- **narrow STANDARD promotion only**
- **not broad ELITE/live replacement**

---

## How to interpret the whitelist now

The BTTS ValueEV whitelist remains useful.

But its meaning must be phrased correctly.

It now means:
- these leagues are the strongest current candidate pool for BTTS ValueEV
- they are the best places to run shadow/watch tracking
- they are the best places to evaluate future ValueEV promotion
- they provide context for selective STANDARD promotion

It does **not** mean:
- all BTTS ValueEV rows in those leagues go live automatically
- BTTS ValueEV has already beaten the model strongly enough to replace it

---

## Current deployment interpretation

### BTTS model
- live primary lane
- remains the default BTTS production path
- not displaced by current ValueEV evidence

### BTTS ValueEV
- shadow/watch lane by default
- evaluated through league policy, directional audits, and head-to-head comparisons
- selectively promotable only through the stricter runtime whitelist-style STANDARD path

### Shadow buckets
- continue to capture profitable near-miss aligned ValueEV rows
- act as profit-leak detection rather than auto-promotion logic
- should be monitored before any widening of ValueEV live posture

---

## What the comparison currently proves

At this stage, the model-vs-ValueEV comparison proves:

1. BTTS ValueEV is worth continued tracking
2. BTTS ValueEV is not random noise
3. some league-month slices show promising ValueEV outperformance
4. sparse shared samples still limit strong replacement claims
5. the current code posture should remain model-live / ValueEV-shadow

---

## What it does not yet prove

The current comparison does **not** yet prove:
- BTTS ValueEV should become the new primary live lane
- BTTS ValueEV should be auto-live in all whitelist leagues
- BTTS model should be demoted
- the current narrow STANDARD promotion path is too strict

Those are future decisions, not current locked conclusions.

---

## Locked recommendation

Use the comparison layer as a governance tool, not a marketing overclaim.

Current locked policy:
- keep **BTTS model** as live primary
- keep **BTTS ValueEV** as shadow/watch by default
- use whitelist leagues as the strongest candidate pool for controlled evaluation
- allow only narrow `STANDARD` promotion for aligned, explicitly tagged, high-conviction BTTS ValueEV rows
- require stronger shared-sample evidence before any broader live promotion claim

---

## One-line policy footer

**BTTS deployment decision: model live primary; BTTS ValueEV shadow/watch by default; only aligned high-conviction ValueEV rows may be promoted to STANDARD under the stricter runtime thresholds.**

---

## Related files

- `BTTS_WALKFORWARD_SUMMARY.md`
- `BTTS_POLICY_DECISION.md`
- `BTTS_YES_WALKFORWARD_SUMMARY.md`
- `BTTS_NO_WALKFORWARD_SUMMARY.md`
- `deploy_rulebook.py`
- `btts_league_policy.json`
- `btts_live_whitelist.csv`
- `btts_shadow_review.csv`