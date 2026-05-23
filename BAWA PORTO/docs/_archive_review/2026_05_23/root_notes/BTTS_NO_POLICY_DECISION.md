BTTS_NO_POLICY_DECISION.md
# BTTS NO Policy Decision

## Quick deploy note

- **Primary lane =** BTTS model live primary
- **Shadow lane =** BTTS NO ValueEV whitelist candidates plus review leagues and approved high-conviction aligned BTTS NO ValueEV rows
- **Promotion rule =** BTTS NO ValueEV stays shadow/watch unless a refreshed head-to-head supports live promotion, the league also passes whitelist rules, and the row clears the explicit NO-side STANDARD promotion gate
- **Demotion rule =** any losing month, sub-0.45 weighted ROI on refreshed audit, adverse head-to-head evidence, or future stability breach if max-drawdown rule is formalised

---

## Purpose

This document is the compact deploy-note counterpart to the BTTS walk-forward summaries.

It locks the current BTTS NO routing posture into a deploy-legible policy that can be:
- enforced in `deploy_rulebook.py`
- expressed through BTTS league-policy artifacts
- monitored via shadow buckets without weakening the live BTTS model lane

---

## One-line deploy posture

Keep **BTTS model as the live primary lane**. Route BTTS NO ValueEV to **shadow/watch** by default, using whitelist and review buckets as routing evidence rather than automatic live promotion. Only allow narrow BTTS NO ValueEV promotion into `STANDARD` when the row is explicitly ValueEV, aligned, in-policy, and clears the locked NO-side runtime thresholds.

---

## Policy buckets

### Live
Must satisfy the current league-policy rules:
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

Important:
- these bucket rules describe **league quality for BTTS ValueEV monitoring and routing**
- they do **not** by themselves grant automatic live promotion over the BTTS model lane

---

## BTTS NO live candidate pool

Current BTTS ValueEV candidate whitelist leagues:
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

Interpretation:
- these leagues meet the current BTTS ValueEV league-policy Live rules
- they form the strongest candidate pool for BTTS NO ValueEV shadow/watch and selective promotion review
- this does **not** override the current global BTTS policy, which keeps the BTTS model as the live primary lane

---

## Review leagues

Current BTTS ValueEV review leagues:
- USA MLS
- Belgium Pro
- Brazil Serie A
- Spain La Liga

Routing:
- keep these leagues in shadow/review only for BTTS NO ValueEV
- do not treat them as automatic live candidates
- continue month-by-month monitoring and directional audit work before any promotion decision

---

## Blacklist / insufficient evidence

### Blacklist
- England FA Cup

Routing:
- block from live BTTS NO ValueEV deployment
- research-only if explicitly needed

### Insufficient evidence
- Scotland Premiership
- Champions League
- England Premier League

Routing:
- hold out from BTTS NO ValueEV promotion decisions until evidence thresholds are met

---

## Runtime stamping rules

BTTS rows must carry runtime metadata after the BTTS split:
- `btts_alignment`
- `signal_btts_runtime`

Locked runtime interpretation:
- aligned rows keep side-aware runtime labels
- contrarian rows are downgraded to `NEUTRAL`
- BTTS NO policy must use the stamped runtime signal, not the unstamped base label, for promotion logic

This matters because BTTS NO promotion is only safe when the row remains both:
- explicitly ValueEV
- explicitly aligned with the model side

---

## BTTS NO ValueEV promotion policy

Current code-aligned policy:
- `BTTS_LIVE_POLICY = "model_plus_whitelist_valueev"`
- `BTTS_VALUEEV_ALLOWED_LIVE = False`
- `BTTS_VALUEEV_STANDARD_PROMOTION_ENABLED = True`

This means:
- BTTS NO ValueEV is **not** broadly live
- explicit BTTS ValueEV rows are forced out of live tiers by default
- a narrow subset may still be promoted into `STANDARD`

### Locked BTTS NO STANDARD promotion gate

A BTTS NO ValueEV row is eligible for `STANDARD` promotion only when all of the following are true:
- `market == btts`
- row is explicitly detected as BTTS ValueEV
- `btts_alignment == ALIGNED`
- `signal_btts_runtime` is in `{"VERY_STRONG_NO"}`
- `edge >= 0.21`
- `model_p_for_bookie >= 0.71`

This is the current locked NO-side whitelist promotion gate from `deploy_rulebook.py`.

Important:
- this is **not** the older discovery phrasing of “BTTS ValueEV = live in whitelist leagues only”
- this is a narrower code-enforced promotion gate into `STANDARD`
- everything else remains shadow/watch or `OBSERVE`

---

## BTTS NO side runtime / structural notes

The BTTS NO branch in `deploy_rulebook.py` currently applies a directional safety stack including:
- strong-label gate: `STRONG_NO` / `VERY_STRONG_NO`
- blank plausibility via FTS structure
- BTTS NO model floor using `max(btts_no_pmin_floor, implied_used - tol_btts_f)`
- default `btts_no_pmin_floor = 0.52`
- low-goaliness confirmation
- clean-sheet / blanking-risk confirmation
- optional H2H NO gate when configured

Additional current defaults exposed in `deploy_rulebook.py` include:
- `--btts-no-pmin-floor` default `0.52`
- `--fts-no-min` default `0.35`
- `--btts-no-cs-min` default `0.35`
- `--btts-no-scored-max` default `0.70`
- `--btts-no-xgfor-max` default `1.00`
- `--btts-no-xgagainst-max` default `1.00`

Interpretation:
- BTTS NO is not just a league-policy lane
- it is also a directional runtime-filtered lane with a specific structural anti-chaos posture

---

## Shadow bucket policy

Purpose:
- prevent potentially profitable aligned BTTS NO ValueEV near-miss rows from disappearing without measurement
- compare strict promoted BTTS NO candidates against broader shadow-watch candidates
- only widen deployment after evidence is strong at meaningful sample size

Operational posture:
- keep the live BTTS model lane intact
- keep BTTS NO ValueEV outside broad live routing
- track promoted and near-miss BTTS NO ValueEV rows in shadow buckets
- only revise promotion rules after refreshed directional evidence and head-to-head support

---

## Promotion and demotion conditions

### Promotion to stronger candidate status
A BTTS NO ValueEV league can improve from Review to Live-candidate status when it satisfies:
- months_present >= 3
- total_rows >= 20
- weighted_roi >= 0.45
- losing_months == 0

### Promotion to STANDARD row status
A BTTS NO ValueEV row can only be promoted to `STANDARD` when it clears the explicit runtime gate:
- aligned
- explicit ValueEV
- `VERY_STRONG_NO`
- `edge >= 0.21`
- `model_p_for_bookie >= 0.71`
- league context remains acceptable under current policy

### Demotion
Demote back to shadow/watch when:
- losing month appears on refreshed league audit, or
- weighted ROI drops below live-candidate threshold, or
- head-to-head evidence turns adverse, or
- the row fails the locked runtime gate

### Blacklist trigger
Blacklist when:
- sufficient evidence exists AND
- weighted_roi < 0.15

---

## Locked deployment recommendation

1. Keep BTTS model as the live primary lane
- do not present BTTS NO ValueEV as a blanket live lane
- keep code and docs aligned on the model-primary posture

2. Treat BTTS NO ValueEV whitelist leagues as candidate/shadow-watch pool
- strongest leagues stay under active monitoring
- review leagues remain shadow/research only
- blacklist and insufficient-evidence leagues stay out

3. Only allow narrow BTTS NO ValueEV promotion into `STANDARD`
- require explicit ValueEV tagging
- require alignment
- require `VERY_STRONG_NO`
- require edge and probability thresholds from code

4. Keep directional BTTS NO work separate from generic BTTS claims
- BTTS NO should be judged on its own walk-forward and runtime behavior
- do not overstate league-policy quality as proof of broad live deployment

5. Next validation sequence
- complete BTTS NO directional walk-forward summary lock
- compare BTTS NO model vs BTTS NO ValueEV at better sample size
- extend shadow-bucket tracking before any wider promotion decision

---

## Current locked conclusion

- BTTS model remains the live primary lane
- BTTS NO ValueEV is a candidate / shadow-watch lane, not a blanket live lane
- league-aware routing still matters for BTTS NO monitoring
- narrow `STANDARD` promotion is allowed only for aligned, explicit, `VERY_STRONG_NO` ValueEV rows that clear the code thresholds
- broader live promotion is not yet justified
- next priority is continued directional audit and refreshed model-vs-ValueEV evidence

---

## Related files

- `BTTS_WALKFORWARD_SUMMARY.md`
- `BTTS_NO_WALKFORWARD_SUMMARY.md`
- `BTTS_MODEL_VS_VALUEEV_POLICY_NOTE.md`
- `btts_league_policy.json`
- `btts_live_whitelist.csv`
- `btts_shadow_review.csv`