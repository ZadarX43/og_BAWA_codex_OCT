BTTS_YES_WALKFORWARD_SUMMARY.md
# BTTS YES Walk-Forward Summary

## Product status

BTTS YES is now being locked as its own directional lane inside the broader BTTS product family.

This document isolates the YES-side behaviour from the combined BTTS work so the project can answer the next important deployment questions cleanly:
- how BTTS YES behaves as a standalone lane
- whether YES-side edge is concentrated in specific leagues
- how stable the YES lane is month to month
- how YES-side runtime gates in `deploy_rulebook.py` should be described in policy documents

This is a directional audit document, not a claim that BTTS YES is already the automatic live primary lane.

The current repo-level BTTS policy remains:
- BTTS model = live primary lane
- BTTS ValueEV = shadow/watch by default
- selected aligned high-conviction BTTS ValueEV rows may be promoted only under the current STANDARD-promotion logic

---

## Purpose

This document locks the BTTS YES lane as a standalone walk-forward research and deployment artifact.

It is intended to record:
- YES-only league audit findings
- YES-only month stability findings
- YES-side whitelist / review / blacklist posture
- YES-side runtime label and probability rules
- the current policy split between BTTS model live routing and BTTS ValueEV shadow/watch routing

This document should be read alongside:
- `BTTS_WALKFORWARD_SUMMARY.md`
- `BTTS_POLICY_DECISION.md`
- future YES-only audit outputs / comparison tables

---

## Current policy position

At the current locked stage:
- BTTS YES is a real directional lane worth tracking independently
- BTTS YES runtime logic is strong enough to document explicitly
- YES-side ValueEV rows are not automatically live just because they come from a whitelist league
- YES-side promotions must still respect alignment, runtime label quality, probability floors, and the current BTTS live-policy lock in `deploy_rulebook.py`

So the correct framing is:
- **BTTS YES is deployment-relevant**
- **BTTS YES is not yet fully separated into an unrestricted live product lane**
- **BTTS YES promotions remain conditional**

---

## What this document is meant to lock

The BTTS YES lane should eventually answer four things clearly:

1. **League audit (YES-only)**
   - which leagues produce the strongest YES-side evidence
   - which leagues are unstable or weak

2. **Month stability (YES-only)**
   - whether YES-side performance is concentrated in one period or repeatable across months

3. **Routing posture (YES-only)**
   - which leagues should be treated as whitelist / review / blacklist / insufficient evidence for YES-side monitoring

4. **Runtime deployment rules**
   - which YES-side labels, pmin floors, and structural confirmations are actually used in `deploy_rulebook.py`

---

## YES-side league audit

This section is the YES-only league-audit lock point.

Populate here from the YES-only build artifact once generated.

### Required columns for the YES-only league audit table

| league | months_present | total_rows | weighted_hit | weighted_roi | weighted_avg_odds | profitable_months | losing_months | max_drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

### YES-only league audit table

_TODO: paste the YES-only league audit table here once the YES split artifact is built._

---

## YES-side month stability

This section records the month-by-month behaviour of the YES lane only.

The purpose is to answer:
- whether BTTS YES remains stable across the walk-forward window
- whether weak months cluster in particular leagues or periods
- whether YES-side routing should eventually be stricter than the combined BTTS posture

### Required columns for the YES-only monthly table

| month | rows | hit | roi | avg_odds |
|---|---:|---:|---:|---:|

### YES-only monthly table

_TODO: paste the YES-only monthly walk-forward table here once generated._

---

## YES-side policy buckets

These are the bucket definitions the YES-only audit should use unless the YES build explicitly introduces different directional policy rules.

### Live
- months_present >= 3
- total_rows >= 20
- weighted_roi >= 0.45
- losing_months == 0

### Review
- leagues that do not meet Live, Blacklist, or Insufficient Evidence rules

### Blacklist
- weighted_roi < 0.15 with sufficient evidence

### Insufficient evidence
- months_present < 3 OR total_rows < 20

Important note:
These are the current **league-policy bucket rules** inherited from the BTTS walk-forward policy file.
They do **not** by themselves override the repo-level live-policy lock for BTTS ValueEV rows.

---

## YES-side whitelist / review / blacklist

This section should be filled from the YES-only audit build.

### YES whitelist

Use this section to record leagues where BTTS YES has enough evidence to be trusted as the strongest directional candidate pool.

_TODO: insert YES-only whitelist once built._

### YES review lanes

Use this section to record leagues where BTTS YES is positive or interesting but not stable enough for directional trust.

_TODO: insert YES-only review list once built._

### YES blacklist

Use this section to record leagues where BTTS YES is directionally weak enough to avoid.

_TODO: insert YES-only blacklist once built._

### YES insufficient evidence

Use this section to record leagues where the YES sample remains too small to classify safely.

_TODO: insert YES-only insufficient-evidence list once built._

---

## YES runtime rules from deploy_rulebook.py

This is the most important code-truth section in this file.

The YES lane must be documented using the actual runtime behaviour in `deploy_rulebook.py`, not older shorthand phrasing.

### 1) Runtime stamping

BTTS rows are stamped with:
- `btts_alignment`
- `signal_btts_runtime`

Rules:
- aligned rows keep the side-aware runtime label
- contrarian rows are downgraded to `NEUTRAL`
- canonical deploy logic then uses the stamped runtime signal

This means BTTS YES policy is **alignment-aware**, not just raw label-aware.

### 2) YES strong-label gate

Inside `_apply_btts_yes_rules`, BTTS YES rows must first pass the YES runtime label gate:
- allowed labels by default: `STRONG_YES`, `VERY_STRONG_YES`

So weak YES labels are not part of the normal YES deploy lane.

### 3) YES implied-max gate

BTTS YES uses:
- `btts_implied_max = 0.72`

This is described in code as avoiding ultra-short YES selections.

### 4) YES model-probability floor

BTTS YES uses:
- `btts_yes_pmin_floor = 0.55`

And the row must satisfy:
- `model_p_for_bookie >= max(btts_yes_pmin_floor, bookie_implied_used - tol_btts_f)`

So the YES floor is not a single raw pmin only.
It is the maximum of:
- the hard minimum floor, and
- the implied-minus-tolerance floor

### 5) YES FTS risk controls

BTTS YES is filtered against strong blank-risk structures.

Current locked thresholds in `deploy_rulebook.py`:
- early FTS veto stage uses `< 0.30` on the max of `p_home_fts`, `p_away_fts`
- structural YES FTS rule uses:
  - `fts_yes_max = 0.25`
  - both sides must satisfy the YES-side blank-risk rule when those columns are available

Interpretation:
- BTTS YES is intentionally protected against one-team-clamp profiles

### 6) YES confirmation thresholds

Current YES-side defaults exposed in `deploy_rulebook.py` include:
- `btts_yes_scored_min = 0.70`
- `btts_yes_cs_max = 0.35`
- `btts_yes_conceded_min = 0.60`
- `btts_yes_bttsrate_min = 0.55`
- `btts_yes_xgfor_min = 1.10`
- `btts_yes_xgagainst_min = 1.00`

Where available, the YES lane uses these as confirmation-style gates.

### 7) YES label-aware structure gate

The YES path is not purely binary.

The code explicitly supports label-aware strictness:
- stronger YES labels can pass via a looser strong-structure path
- weaker acceptable labels rely on stricter confirmation logic

That is an important part of the YES deployment story:
- YES is not just “high probability”
- YES is “high probability plus structural confirmation plus signal strength”

### 8) Optional YES H2H gate

The file also supports an optional YES H2H gate when configured:
- `h2h_min_n = 3`
- `btts_yes_h2h_min = None` by default unless supplied

So H2H is optional, not part of the current locked mandatory YES posture unless explicitly enabled upstream.

---

## YES-side ValueEV promotion rules

The combined BTTS live policy matters here.

Even when a YES-side BTTS ValueEV row looks strong, it is not automatically live.

### Current code-truth live policy

In `deploy_rulebook.py`:
- `BTTS_LIVE_POLICY = "model_plus_whitelist_valueev"`
- `BTTS_VALUEEV_ALLOWED_LIVE = False`
- `BTTS_VALUEEV_STANDARD_PROMOTION_ENABLED = True`

This means:
- BTTS model rows can remain live
- explicit BTTS ValueEV rows are forced out of live tiers first
- selected BTTS ValueEV rows may then be promoted back into `STANDARD`, not broad unrestricted live status

### YES-side STANDARD promotion thresholds

Current YES-side ValueEV STANDARD promotion thresholds are:
- allowed runtime label set: `VERY_STRONG_YES`
- `edge >= 0.24`
- `model_p_for_bookie >= 0.74`
- row must be aligned
- row must be explicitly identified as BTTS ValueEV

So the YES-side promotion rule is much stricter than the broader YES runtime gate.

That difference must stay explicit in all docs:
- **runtime-eligible YES** is not the same thing as
- **ValueEV YES eligible for STANDARD promotion**

---

## Recommended YES-side policy wording

Use the following posture in future YES-lane docs unless code changes again:

- BTTS YES is a strong directional monitoring lane
- BTTS YES runtime selection is label-aware and structure-aware
- BTTS YES live behaviour is currently split:
  - model lane may remain live
  - ValueEV YES remains shadow/watch by default
  - only aligned, very-strong, high-edge YES ValueEV rows can be promoted into STANDARD under the current lock

This wording is consistent with the repo code and avoids overstating YES-side live status.

---

## What has been proven already

Even before the YES-only split table is pasted in, the current repo work already proves:

1. BTTS YES has explicit deploy-rule logic, not vague market-level heuristics
2. BTTS YES is alignment-aware through runtime stamping
3. BTTS YES uses real structure controls, not just a raw probability cut
4. BTTS YES ValueEV promotion is stricter than the general BTTS runtime gate
5. YES-side documentation must distinguish between:
   - model live routing
   - ValueEV shadow/watch routing
   - STANDARD-only whitelist promotion

---

## What remains to be filled from the YES-only build

To fully lock this file, the following artifacts still need to be inserted:
- YES-only league audit table
- YES-only monthly stability table
- YES-only whitelist / review / blacklist lists
- YES-only deployment interpretation by league
- YES-only comparison notes versus BTTS NO

---

## Current locked conclusion

BTTS YES is now documented as a standalone directional lane.

The current locked interpretation is:
- BTTS YES has real deployment value
- BTTS YES runtime logic is stronger and more specific than the older market-level summary implied
- BTTS YES should be described using the actual rulebook defaults and promotion thresholds
- BTTS YES is not yet a blanket automatic live ValueEV lane
- the next step is to paste in the YES-only walk-forward audit outputs and then compare YES vs NO directly

---

## Related files

- `BTTS_WALKFORWARD_SUMMARY.md`
- `BTTS_POLICY_DECISION.md`
- `BTTS_NO_WALKFORWARD_SUMMARY.md`
- `deploy_rulebook.py`
- future YES-only build outputs