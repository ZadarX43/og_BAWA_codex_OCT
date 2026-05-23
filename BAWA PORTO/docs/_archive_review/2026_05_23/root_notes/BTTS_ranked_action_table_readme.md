# BTTS Ranked Action Table — Proxy Feature Ablation Readme

## Scope
This readme summarizes the BTTS proxy-feature ablation results from:

- `audit_btts_proxy_feature_ablation_from_allmarkets.py`
- Input:
  - `predictions_output/2026-03-18/BOOKIE_IMP20_ALLMARKETS_2024-02-13_to_2026-02-17.csv`
- Truth source:
  - `Matches/__merged__proxy_enriched`

The goal is to decide, by league, whether BTTS should use:

- baseline only
- baseline + score proxy only
- baseline + regime proxy only
- baseline + matchup proxy only
- baseline + timing proxy only
- baseline + full proxy bundle

---

## BTTS proxy groups tested

### Baseline
Core BTTS stack only:
- `prob_btts`
- `prob_btts_v2`
- odds / implieds
- BTTS rates
- clean-sheet rates
- scored / conceded rates
- goaliness
- xG for / against
- pre-match xG
- expected-goals sum
- `bookie_lambda_total_fit`
- `over_25_percentage_pre_match`
- `model_p_for_bookie`

### Score proxy
- `snapshot_ou25_support_score_proxy`

### Regime proxy
- `snap_xg_total_pressure_proxy`
- `snap_style_chaos_index_proxy`
- `snap_ou25_over_regime_blend_proxy`

### Matchup proxy
- `snap_home_attack_vs_away_def_xg_proxy`
- `snap_home_attack_vs_away_def_goals_proxy`

### Timing proxy
- `snap_timing_both_teams_late_risk_proxy`

### Full proxy bundle
- all of the above proxy families together

---

# Ranked BTTS league action table

## Tier 1 — Build now

### 1) Brazil Serie A
**Label:** BUILD NOW  
**Deployment recommendation:** Dedicated BTTS proxy lane

#### Core result
- Baseline AUC: **0.5283**
- Full bundle AUC: **0.6614**
- Delta AUC: **+0.1331**
- Baseline accuracy: **0.5011**
- Full bundle accuracy: **0.5987**

#### Component view
- Score proxy only: **0.6540**
- Regime proxy only: **0.5325**
- Matchup proxy only: **0.5293**
- Timing proxy only: **0.5313**
- Full bundle: **0.6614**

#### Deploy-style threshold shape
- 55 hit rate: **49.7% → 60.6%**
- 60 hit rate: **54.4% → 67.1%**
- 65 hit rate: **71.4% → 62.5%** with much larger usable coverage
- 70 hit rate: baseline had almost no meaningful deploy lane; proxy bundle creates one

#### Action
Use:
- **baseline + score proxy**
- **baseline + full proxy bundle**

#### Interpretation
Brazil BTTS is the clearest winner. The score-support proxy is doing most of the heavy lifting, and the full bundle sharpens it further.

---

### 2) England Premier League
**Label:** BUILD NOW  
**Deployment recommendation:** Selective premium overlay, not global replacement

#### Core result
- Baseline AUC: **0.7659**
- Full bundle AUC: **0.7782**
- Delta AUC: **+0.0123**
- Baseline accuracy: **0.6973**
- Full bundle accuracy: **0.6919**

#### Component view
- Score proxy only: **0.7759**
- Regime proxy only: **0.7707**
- Matchup proxy only: **0.7670**
- Timing proxy only: **0.7664**
- Full bundle: **0.7782**

#### Deploy-style threshold shape
- 55 hit rate: **79.3% → 79.0%**
- 60 hit rate: **78.6% → 80.1%**
- 65 hit rate: **78.5% → 79.3%**
- 70 hit rate: **82.0% → 85.1%**

#### Action
Use:
- **baseline + score proxy**
- **baseline + full proxy bundle**
for high-threshold BTTS lanes only.

#### Interpretation
This is a ranking-quality improvement more than a raw 0.5 accuracy story. Best used as a premium threshold overlay.

---

### 3) Scotland Premiership
**Label:** BUILD NOW  
**Deployment recommendation:** Optional overlay lane

#### Core result
- Baseline AUC: **0.7213**
- Full bundle AUC: **0.7304**
- Delta AUC: **+0.0091**
- Baseline accuracy: **0.6739**
- Full bundle accuracy: **0.6920**

#### Component view
- Score proxy only: **0.7240**
- Regime proxy only: **0.7247**
- Matchup proxy only: **0.7221**
- Timing proxy only: **0.7214**
- Full bundle: **0.7304**

#### Action
Use:
- **baseline + full bundle**
or
- **baseline + regime + score**

#### Interpretation
Smaller than Brazil or EPL, but still real and usable.

---

## Tier 2 — Watch / test later

### 4) Spain La Liga
**Label:** WATCH  
**Deployment recommendation:** Soft watchlist only

#### Core result
- Baseline AUC: **0.7371**
- Full bundle AUC: **0.7423**
- Delta AUC: **+0.0052**
- Baseline accuracy: **0.6966**
- Full bundle accuracy: **0.6736**

#### Component view
- Score proxy only: **0.7374**
- Regime proxy only: **0.7356**
- Matchup proxy only: **0.7386**
- Timing proxy only: **0.7372**
- Full bundle: **0.7423**

#### Action
Keep as:
- future walk-forward candidate only

#### Interpretation
There is some AUC lift, but not enough to justify immediate live architecture changes.

---

### 5) Italy Serie A
**Label:** WATCH  
**Deployment recommendation:** Regime-first soft watch

#### Core result
- Baseline AUC: **0.8376**
- Full bundle AUC: **0.8408**
- Delta AUC: **+0.0031**
- Baseline accuracy: **0.7601**
- Full bundle accuracy: **0.7578**

#### Component view
- Score proxy only: **0.8388**
- Regime proxy only: **0.8412**
- Matchup proxy only: **0.8371**
- Timing proxy only: **0.8372**
- Full bundle: **0.8408**

#### Action
If revisiting:
- test **baseline + regime proxy only** first
- then compare to full bundle

#### Interpretation
Italy may like regime-style context slightly more than the whole bundle, but the edge is small.

---

### 6) USA MLS
**Label:** WATCH  
**Deployment recommendation:** Leave alone for now

#### Core result
- Baseline AUC: **0.7087**
- Full bundle AUC: **0.7105**
- Delta AUC: **+0.0018**
- Baseline accuracy: **0.6815**
- Full bundle accuracy: **0.6852**

#### Action
No build now. Revisit only if later walk-forward confirms.

---

### 7) Portugal Liga
**Label:** WATCH  
**Deployment recommendation:** Leave alone, possible future micro-overlay

#### Core result
- Baseline AUC: **0.7228**
- Full bundle AUC: **0.7239**
- Delta AUC: **+0.0011**
- Baseline accuracy: **0.6379**
- Full bundle accuracy: **0.6523**

#### Action
Do not prioritize. Possible future small overlay test.

---

## Tier 3 — Do not touch now

### 8) England Championship
**Label:** DO NOT TOUCH NOW  
- Baseline AUC: **0.7357**
- Full bundle AUC: **0.7318**
- Delta AUC: **-0.0038**

Action:
- Keep baseline only

---

### 9) England EFL League 1
**Label:** DO NOT TOUCH NOW  
- Baseline AUC: **0.7423**
- Full bundle AUC: **0.7412**
- Delta AUC: **-0.0011**

Action:
- Keep baseline only

---

### 10) Belgium Pro
**Label:** DO NOT TOUCH NOW  
- Baseline AUC: **0.7709**
- Full bundle AUC: **0.7628**
- Delta AUC: **-0.0081**

Action:
- No proxy overlay for BTTS

---

### 11) Germany Bundesliga
**Label:** DO NOT TOUCH NOW  
- Baseline AUC: **0.6130**
- Full bundle AUC: **0.6041**
- Delta AUC: **-0.0089**

Action:
- Do not use OU25 proxy family for BTTS Bundesliga

---

### 12) France Ligue 1
**Label:** DO NOT TOUCH NOW  
- Baseline AUC: **0.7415**
- Full bundle AUC: **0.7300**
- Delta AUC: **-0.0115**

Action:
- Baseline only

---

### 13) Netherlands Eredivisie
**Label:** DO NOT TOUCH NOW  
- Baseline AUC: **0.8017**
- Full bundle AUC: **0.7901**
- Delta AUC: **-0.0116**

Action:
- Baseline only

---

# BTTS deployment stack recommendations by league

## Build now
### Brazil Serie A
- Primary: **baseline + score proxy**
- Secondary: **baseline + full bundle**

### England Premier League
- Primary: **baseline + score proxy**
- Secondary: **baseline + full bundle**
- Use mainly in 60/65/70 threshold lanes

### Scotland Premiership
- Primary: **baseline + full bundle**
- Secondary: **baseline + regime + score**

---

## Watch / build later
### Spain La Liga
- Trial later: **baseline + full bundle**

### Italy Serie A
- Trial later: **baseline + regime proxy**
- Secondary: **baseline + full bundle**

### Portugal Liga
- Trial later: **baseline + full bundle** or **score proxy**

### USA MLS
- No immediate work
- Monitor only

---

## Do not touch now
- Belgium Pro
- Germany Bundesliga
- France Ligue 1
- Netherlands Eredivisie
- England Championship
- England EFL League 1

---

# What this means architecturally

## BTTS conclusion
The OU25-derived proxy family is **not** a universal BTTS solution.

Instead, it behaves like a:
- **league-specific overlay family**
- especially useful where BTTS is sensitive to wider score-context conditions

## Strongest reusable BTTS proxy component
The most reusable cross-league piece is:

- **`snapshot_ou25_support_score_proxy`**

That is the only proxy that repeatedly shows some positive effect in useful leagues.

## Weak BTTS proxy families
These are not broadly deploy-worthy for BTTS:
- regime stack
- matchup stack
- timing stack

They may help in isolated cases, but not enough to justify global inclusion.

---

# Final action summary

## BTTS build now
1. Brazil Serie A BTTS proxy lane
2. England Premier League BTTS premium overlay
3. Scotland Premiership optional overlay

## BTTS watch later
1. Italy Serie A
2. Spain La Liga
3. Portugal Liga
4. USA MLS

## BTTS do not touch now
1. Belgium Pro
2. Germany Bundesliga
3. France Ligue 1
4. Netherlands Eredivisie
5. England Championship
6. England EFL League 1

---

# Clean combined market takeaway

## OU25
Proxy family is a major upgrade path across many leagues.

## BTTS
Proxy family is selective and should only be used where proven.

## Next architecture direction
- **OU25:** broader proxy integration by league
- **BTTS:** selective overlay integration only
- **FTR:** leave until OU25 + BTTS architecture is locked

That gives a clean, non-Frankenstein build path into:
- wrapper integration
- market-specific walk-forward testing
- final deploy architecture for:
  - FTR
  - OU25
  - BTTS
