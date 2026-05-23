BTTS_NO_WALKFORWARD_SUMMARY.md
# BTTS NO Walk-Forward Summary

## Product status

BTTS NO is now being locked as its own directional BTTS lane.

This document is the BTTS NO counterpart to `BTTS_YES_WALKFORWARD_SUMMARY.md` and is intended to mirror that structure closely while reflecting the actual current runtime policy in code.

The BTTS NO lane is no longer treated as just a generic subset of BTTS. It is now being documented as a distinct directional lane with:
- YES / NO separation at runtime
- side-aware signal handling
- NO-specific model-floor and structural gates
- league-aware walk-forward routing context
- model-live vs ValueEV-shadow policy split

This summary should be read together with:
- `BTTS_WALKFORWARD_SUMMARY.md`
- `BTTS_POLICY_DECISION.md`
- `deploy_rulebook.py`
- `btts_league_policy.json`

---

## Investor summary

BTTS NO is now strong enough to justify its own directional documentation layer, but the current deploy posture must reflect the actual runtime code rather than the older simplified ValueEV wording.

Current locked read:
- BTTS is **not** currently a blanket “ValueEV live in whitelist leagues” product
- the **base BTTS model lane remains the live primary lane**
- BTTS NO ValueEV rows are primarily **shadow / watch** unless they satisfy the stricter STANDARD-promotion policy already implemented in `deploy_rulebook.py`
- side-aware runtime labels, alignment status, and NO-side structural confirmation now matter materially for deploy posture

This means BTTS NO should be understood as:
- a real directional product lane
- a lane with deploy-grade runtime gating
- a lane whose ValueEV overlay is currently controlled and selective rather than broadly live

---

## Purpose

This document locks the current BTTS NO directional posture.

It records:
- BTTS NO as a distinct lane
- the NO-side runtime interpretation in `deploy_rulebook.py`
- the current relationship between BTTS NO model rows and BTTS NO ValueEV rows
- the walk-forward routing context inherited from the broader BTTS league policy
- recommended BTTS NO runtime label policy and probability-floor notes

This file is intended to be the NO-side directional companion to:
- `BTTS_YES_WALKFORWARD_SUMMARY.md`

---

## Directional lane definition

BTTS NO rows are the `market='btts'` rows where canonical selection / bookie pick is:
- `NO`

The runtime now treats BTTS as a split directional market.

Important current behavior:
- BTTS remains canonical as `market='btts'`
- selection is canonicalized to `YES` / `NO`
- runtime stamping adds:
  - `btts_alignment`
  - `signal_btts_runtime`
- contrarian BTTS rows are downgraded to `NEUTRAL`
- aligned BTTS rows keep side-aware runtime labels

So BTTS NO is not just “BTTS with a NO pick.”
It is now a side-aware deploy path.

---

## Runtime stamping rules

The BTTS runtime stamping logic currently does the following:

- if `selection == model_top_pick`:
  - `btts_alignment = ALIGNED`
  - keep side-aware runtime label from `signal_btts_side` (fallback to `signal_btts`)
- if `selection != model_top_pick`:
  - `btts_alignment = CONTRARIAN`
  - `signal_btts_runtime = NEUTRAL`

Locked interpretation:
- aligned NO rows retain meaningful NO-side signal strength
- contrarian NO rows are intentionally neutralized before tier routing
- this prevents contrarian BTTS NO rows from leaking upward through live tiers on raw label strength alone

---

## BTTS NO runtime gate notes from code

The current BTTS NO runtime path in `deploy_rulebook.py` applies the following main logic.

### 1) Strong-label gate

BTTS NO rows must first pass a NO-side signal gate:
- `STRONG_NO`
- `VERY_STRONG_NO`

Rows outside those labels are removed before deeper NO-side confirmation.

### 2) Basic blank-risk plausibility

A BTTS NO row is allowed through the first blank-risk stage if:
- at least one of `p_home_fts` or `p_away_fts` is meaningfully high enough, or
- the FTS heads are missing and the row fail-opens through that stage

This means the lane explicitly expects at least one plausible blanking pathway.

### 3) BTTS NO probability floor

Current code default:
- `btts_no_pmin_floor = 0.52`

Model floor logic is:
- `model_p_for_bookie >= max(btts_no_pmin_floor, bookie_implied_used - tol_btts)`

So BTTS NO is not using a raw fixed p-threshold only.
It is using a floor that can tighten against bookmaker implied probability.

### 4) Structural FTS confirmation

Current code default:
- `fts_no_min = 0.35`

If FTS heads are present, BTTS NO requires at least one side to show plausible blank risk.

### 5) Low-goaliness confirmation

BTTS NO also looks for low-goaliness style support, including when available:
- lower average goals context
- low scored-rate tendencies
- clean-sheet tendency
- low BTTS-rate context
- low rolling xG / low-conceding structure

### 6) Optional H2H gate

If configured and sufficient H2H sample exists, BTTS NO can also be filtered by:
- `h2h_n >= h2h_min_n`
- `h2h_btts_rate <= btts_no_h2h_max`

---

## Locked BTTS NO code thresholds

Current BTTS NO runtime defaults found in `deploy_rulebook.py`:

- `btts_no_pmin_floor = 0.52`
- `fts_no_min = 0.35`
- strong-label gate = `STRONG_NO` or `VERY_STRONG_NO`
- optional low-goaliness / clean-sheet / scored-rate / xG confirmation applies when those columns exist

Current BTTS ValueEV STANDARD-promotion constants for NO-side rows:
- allowed runtime label: `VERY_STRONG_NO`
- `BTTS_VALUEEV_STANDARD_NO_EDGE_MIN = 0.21`
- `BTTS_VALUEEV_STANDARD_NO_PMIN = 0.71`

This means NO-side ValueEV promotion is already stricter than the generic BTTS NO runtime gate.

---

## League routing context

BTTS NO does not yet have a fully locked NO-only league audit artifact in this file.

So the current league-routing context is inherited from the broader BTTS walk-forward policy until the dedicated NO-only audit is completed.

Current BTTS policy buckets from the broader BTTS audit are:

### Live-context whitelist
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

### Review-context leagues
- USA MLS
- Belgium Pro
- Brazil Serie A
- Spain La Liga

### Blacklist-context league
- England FA Cup

### Insufficient-evidence context
- Scotland Premiership
- Champions League
- England Premier League

Important policy clarification:
- these buckets do **not** mean BTTS NO ValueEV is broadly live in those leagues
- they define the current walk-forward routing context
- the actual code posture remains:
  - BTTS model live primary
  - BTTS ValueEV shadow/watch by default
  - only selected aligned high-conviction whitelist rows may be promoted to STANDARD

---

## Live vs shadow interpretation

This is the most important policy distinction to keep correct.

### Current live primary lane
The base BTTS model lane remains the live primary lane.

### Current BTTS NO ValueEV posture
BTTS NO ValueEV is **not** currently a blanket live lane.

Instead:
- explicit BTTS ValueEV rows are forced out of live tiers by policy lock
- they are moved to `OBSERVE` unless they satisfy the stricter whitelist-promotion rules
- selected NO-side rows may be promoted to `STANDARD`, not broad ELITE/live by default

So the real posture is:
- model live primary
- NO-side ValueEV shadow/watch primary
- selective STANDARD promotion for aligned, high-conviction whitelist candidates

---

## BTTS NO whitelist / review / blacklist posture

### BTTS NO whitelist posture
Use the current BTTS whitelist leagues as the best available directional routing context for BTTS NO until a dedicated NO-only league audit is locked.

Operational read:
- whitelist context supports monitoring and selective NO-side ValueEV STANDARD promotion
- whitelist context does **not** by itself justify full live promotion of BTTS NO ValueEV

### BTTS NO review posture
Review leagues remain:
- shadow / reduced-trust
- OBSERVE-only for NO-side ValueEV
- useful for continued directional evidence gathering

### BTTS NO blacklist posture
England FA Cup remains off for BTTS policy purposes unless explicitly running research-only work.

### BTTS NO insufficient-evidence posture
Scotland Premiership, Champions League, and England Premier League remain holdouts until more directional evidence exists.

---

## Recommended BTTS NO runtime label policy

Current locked practical label policy for BTTS NO:

### Base runtime eligibility
Allow BTTS NO rows only when runtime label is:
- `STRONG_NO`
- `VERY_STRONG_NO`

### Preferred high-conviction NO bucket
Treat `VERY_STRONG_NO` as the premium NO-side label for future promotion logic.

### Contrarian NO handling
If the row is contrarian:
- stamp `signal_btts_runtime = NEUTRAL`
- do not treat it as a premium NO-side row even if upstream labels looked strong

Locked recommendation:
- keep NO-side deployment explainable through alignment + runtime label + structural blank-risk support
- do not widen the NO-side lane using contrarian rows

---

## Recommended BTTS NO pmin notes

Current locked interpretation for BTTS NO probability handling:

### Base BTTS NO runtime floor
- `btts_no_pmin_floor = 0.52`

### Dynamic floor behavior
The actual floor is:
- `max(0.52, implied_used - tolerance)`

So in practice:
- short-priced NO rows can face a stricter effective floor
- weak model-over-implied cases should continue to be filtered out

### ValueEV STANDARD-promotion floor
For explicit NO-side ValueEV promotion into `STANDARD`, the current stricter floor is:
- `model_p_for_bookie >= 0.71`
- `edge >= 0.21`
- runtime label must be `VERY_STRONG_NO`
- row must be `ALIGNED`

This should remain the locked NO-side promotion note unless the code changes.

---

## Promotion policy for BTTS NO ValueEV

A BTTS NO ValueEV row is only eligible for current STANDARD promotion when it is:
- `market == btts`
- explicit BTTS ValueEV row
- `selection == NO`
- `btts_alignment == ALIGNED`
- `signal_btts_runtime == VERY_STRONG_NO`
- `edge >= 0.21`
- `model_p_for_bookie >= 0.71`
- inside the broader BTTS policy context that supports whitelist-based monitoring

Anything else should remain shadow/watch / OBSERVE.

This is the main policy distinction that the directional docs must preserve accurately.

---

## What has been proven

At this stage, the BTTS NO work has established the following:

1. BTTS NO is now a real directional runtime lane
2. BTTS NO rows are governed by side-aware runtime stamping
3. contrarian NO rows are intentionally neutralized
4. BTTS NO has its own model-floor and blank-risk logic in code
5. BTTS NO ValueEV promotion is selective and stricter than the base NO runtime gate

---

## What has not yet been proven

The current BTTS NO file does not yet fully prove:
- which leagues are best on a dedicated NO-only walk-forward basis
- whether BTTS NO outperforms BTTS YES at directional lane level
- whether current NO-side STANDARD-promotion thresholds are optimal
- whether some review leagues deserve separate NO-only promotion

Those require the dedicated NO-only league and month audits to be locked.

---

## Remaining work

The remaining work for BTTS NO is:
- build / lock a dedicated BTTS NO league audit
- build / lock a BTTS NO month-stability table
- compare BTTS NO model vs BTTS NO ValueEV at better sample size
- decide whether any NO-only league routing should diverge from the generic BTTS routing buckets
- integrate the final NO-side findings into investor-facing BTTS product framing

---

## Locked deployment recommendation

- keep BTTS model as the live primary lane
- keep BTTS NO ValueEV as shadow/watch by default
- allow selective NO-side STANDARD promotion only where current code thresholds are met
- use the broader BTTS whitelist as routing context, not as automatic NO-side live approval
- keep contrarian NO rows neutralized
- prioritize future NO-only audit work before widening policy

---

## Current locked conclusion

BTTS NO is now documented as a distinct directional lane.

Current locked state:
- BTTS model remains the live primary layer
- BTTS NO ValueEV is not broadly live
- BTTS NO ValueEV can be selectively promoted to STANDARD when aligned, whitelist-context, and high-conviction
- runtime alignment and side-aware labels are now central to NO-side deployment logic
- next priority is the dedicated NO-only league audit and month-stability lock

---

## Related files

- `BTTS_YES_WALKFORWARD_SUMMARY.md`
- `BTTS_WALKFORWARD_SUMMARY.md`
- `BTTS_POLICY_DECISION.md`
- `btts_league_policy.json`
- `btts_live_whitelist.csv`
- `btts_shadow_review.csv`
- `deploy_rulebook.py`