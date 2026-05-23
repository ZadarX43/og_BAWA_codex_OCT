# BTTS Walk-Forward Summary

## Product status

BTTS is now established as a major OG product family alongside FTR and OU25.

The current BTTS work has now moved beyond basic discovery and into deploy-grade walk-forward policy, league routing, runtime side-calibration, and live-vs-shadow audit logic.

This document locks the current BTTS position, including:
- league-level walk-forward routing policy
- candidate whitelist vs shadow/review lanes
- separation between candidate-generation thresholds and runtime live promotion
- BTTS YES / NO runtime calibration rules
- March 2025 deployment audit
- truth-source resolution method
- shadow micro-bucket findings

The practical outcome is that BTTS is no longer just a promising market lane. It is now a functioning, audited deployment family with explicit routing rules, side-aware runtime calibration, and a measurable shadow-bucket research path.

---

## Investor summary

BTTS ValueEV passed walk-forward league audit strongly enough to support deploy-grade league routing as a candidate-selection layer.

However, the runtime code path is stricter than the league-policy file alone.

Current locked read:
- BTTS ValueEV should be treated as a **league-filtered candidate lane**, not an all-league live lane
- the walk-forward whitelist is a **BTTS ValueEV candidate pool**, not blanket permission for live routing
- the current shadow-bucket audit shows profitable near-miss BTTS ValueEV rows are being left on the table and should be tracked formally
- the runtime deploy path remains **BTTS model live primary**, with explicit BTTS ValueEV live handling kept stricter and side-aware

Additional locked evidence from the March 2025 audit:
- shadow micro-bucket: **3 resolved / 3 wins = 100.00%**
- live whitelist baseline: **53 resolved / 52 wins = 98.11%**
- live coverage against merged truth resolver: **53 / 55**
- shadow coverage against merged truth resolver: **3 / 3**

This places BTTS in a similar strategic position to OU25:
- strong enough for controlled live use
- strong enough for investor-facing framing
- not yet finished, because YES / NO split work and filtered-vs-model comparisons still remain

---

## Purpose

This document records the BTTS walk-forward league audit and the resulting routing posture, while distinguishing league-policy candidate routing from stricter runtime live handling.

The goal is to convert the existing BTTS walk-forward evidence into deploy-grade routing artifacts while keeping the code-path distinctions explicit:
- `btts_league_policy.json`
- `btts_live_whitelist.csv`
- `btts_shadow_review.csv`
- `BTTS_WALKFORWARD_SUMMARY.md`

This document also records the March 2025 BTTS deploy audit, including:
- truth-source resolution from merged match files
- live whitelist baseline audit
- shadow micro-bucket audit
- unresolved fixture notes

---

## Three-layer interpretation

BTTS now needs to be read as three separate layers rather than one blended policy.

### 1) Walk-forward league policy layer
This is the league-routing layer derived from `btts_walkforward_league_audit.csv` and written into:
- `btts_league_policy.json`
- `btts_live_whitelist.csv`
- `btts_shadow_review.csv`

This layer decides which leagues are trusted as:
- candidate live-quality BTTS ValueEV leagues
- review / shadow leagues
- blacklist leagues
- insufficient-evidence leagues

### 2) Upstream BTTS ValueEV candidate-generation layer
In `bookie_allmarkets.py`, BTTS ValueEV candidate rows are created using the current upstream thresholds:
- `btts_valueev_od_min = 1.33`
- `btts_valueev_edge_min = 1.02`

Important implementation detail:
- the code checks `edge_value >= btts_valueev_edge_min / 100.0`
- so the effective raw edge gate in candidate generation is **edge >= 0.0102**

This is the layer behind the earlier deployment-window choke analysis.

### 3) Runtime live-tier enforcement layer
In `deploy_rulebook.py`, the runtime handling is stricter than the upstream candidate-generation layer.

Current locked runtime posture:
- `BTTS_LIVE_POLICY = "model_plus_whitelist_valueev"`
- `BTTS_VALUEEV_ALLOWED_LIVE = False`
- explicit BTTS ValueEV rows are blocked from live tiers by default
- only a narrow high-conviction subset can be promoted into `STANDARD`
- broad BTTS model rows remain the live primary lane

This distinction is critical.

The walk-forward whitelist does **not** mean:
- every BTTS ValueEV row in those leagues goes live

It means:
- those leagues form the strongest candidate pool for controlled BTTS ValueEV handling

---

## Policy rules

The current walk-forward league-policy rules are:
- **Live**: months_present >= 3, total_rows >= 20, weighted_roi >= 0.45, losing_months == 0
- **Review**: leagues not meeting live, blacklist, or insufficient-evidence rules
- **Blacklist**: weighted_roi < 0.15 with sufficient evidence
- **Insufficient evidence**: months_present < 3 or total_rows < 20

These rules define the current BTTS ValueEV league-routing posture.

They do **not** by themselves define final runtime live permission for BTTS ValueEV rows.

---

## Final leaderboard

                league         policy_bucket  months_present  total_rows  weighted_hit  weighted_roi  weighted_avg_odds  profitable_months  losing_months  max_drawdown
  Scotland Premiership insufficient_evidence               1           2      1.000000      0.845000           1.845000                  1              0      0.000000
        France Ligue 1                  live               8         120      0.966667      0.678917           1.737500                  8              0      0.000000
    Germany Bundesliga                  live               8          92      0.978261      0.615435           1.651304                  8              0      0.000000
         Portugal Liga                  live               8         120      0.916667      0.603250           1.746667                  8              0      0.000000
         Italy Serie A                  live               9         154      0.909091      0.595714           1.756104                  9              0      0.000000
      Champions League insufficient_evidence               6          17      0.882353      0.594706           1.805882                  6              0      0.000000
  England Championship                  live               8         192      0.901042      0.551875           1.725937                  8              0      0.000000
               USA MLS                review               8         164      0.939024      0.541768           1.641341                  7              1     -0.094000
    Norway Eliteserien                  live               8          89      0.932584      0.532472           1.643483                  8              0      0.000000
Netherlands Eredivisie                  live               8         129      0.899225      0.516047           1.685891                  8              0      0.000000
              Japan J1                  live               9         166      0.861446      0.498554           1.740843                  9              0      0.000000
         Europa League                  live               6          28      0.857143      0.492143           1.743214                  6              0      0.000000
     Europa Conference                  live               7         173      0.849711      0.455145           1.716243                  7              0      0.000000
England Premier League insufficient_evidence               8          18      0.777778      0.442778           1.867222                  6              2     -1.000000
           Belgium Pro                review               9         101      0.841584      0.398218           1.662376                  7              2     -1.000000
        Brazil Serie A                review               7          53      0.698113      0.247547           1.783774                  6              1     -0.317500
         Spain La Liga                review               8          96      0.666667      0.163229           1.752083                  6              2     -0.050000
        England FA Cup             blacklist               8         121      0.628099      0.073719           1.718926                  6              2     -0.173704

---

## What the leaderboard means

### Candidate live whitelist

These are currently trusted as the main BTTS ValueEV candidate-live-quality leagues:
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

Important clarification:
- these leagues pass the walk-forward BTTS ValueEV league-policy Live rules
- this does **not** mean all BTTS ValueEV rows from these leagues are runtime-live automatically

### Review / shadow lanes

These are not broken, but they are not yet strong enough to be treated as unconditional candidate live lanes:
- USA MLS
- Belgium Pro
- Brazil Serie A
- Spain La Liga

Interpretation:
- USA MLS is strong on headline ROI, but one losing month breaks the current live rule
- Belgium Pro has usable edge, but too much instability
- Brazil Serie A is positive but materially weaker than the live set
- Spain La Liga is the weakest of the current review set

### Blacklist

- England FA Cup

### Insufficient evidence

- Scotland Premiership
- Champions League
- England Premier League

Important subtlety:
- England Premier League is not being labeled as a bad BTTS lane
- it is being treated as under-sampled and unstable under the current rules
- so insufficient evidence is the correct state, not blacklist

---

## Live whitelist table

                league  months_present  total_rows  weighted_hit  weighted_roi  losing_months  max_drawdown
        France Ligue 1               8         120      0.966667      0.678917              0           0.0
    Germany Bundesliga               8          92      0.978261      0.615435              0           0.0
         Portugal Liga               8         120      0.916667      0.603250              0           0.0
         Italy Serie A               9         154      0.909091      0.595714              0           0.0
  England Championship               8         192      0.901042      0.551875              0           0.0
    Norway Eliteserien               8          89      0.932584      0.532472              0           0.0
Netherlands Eredivisie               8         129      0.899225      0.516047              0           0.0
              Japan J1               9         166      0.861446      0.498554              0           0.0
         Europa League               6          28      0.857143      0.492143              0           0.0
     Europa Conference               7         173      0.849711      0.455145              0           0.0

---

## Shadow / review lanes table

                league         policy_bucket  months_present  total_rows  weighted_hit  weighted_roi  losing_months worst_month_by_roi  worst_roi  max_drawdown
  Scotland Premiership insufficient_evidence               1           2      1.000000      0.845000              0            2024-02   0.845000        0.0000
      Champions League insufficient_evidence               6          17      0.882353      0.594706              0            2024-07   0.395556        0.0000
England Premier League insufficient_evidence               8          18      0.777778      0.442778              2            2024-04  -1.000000       -1.0000
               USA MLS                review               8         164      0.939024      0.541768              1            2024-09  -0.094000       -0.0940
           Belgium Pro                review               9         101      0.841584      0.398218              2            2024-07  -1.000000       -1.0000
        Brazil Serie A                review               7          53      0.698113      0.247547              1            2024-06  -0.317500       -0.3175
         Spain La Liga                review               8          96      0.666667      0.163229              2            2024-02  -0.050000       -0.0500

---

## Upstream candidate gate read

Current upstream BTTS ValueEV candidate-generation notes from `bookie_allmarkets.py`:
- `od_min = 1.33`
- `edge_min = 1.02`

Important implementation detail:
- `edge_min` is divided by 100.0 in the candidate-generation check
- so the effective raw candidate threshold is **edge >= 0.0102**

Deployment-window audit:
- 2024-11: 26 total -> 12 survive
- 2024-12: 22 total -> 2 survive
- 2025-01: 20 total -> 2 survive
- 2025-02: 18 total -> 2 survive
- 2025-03: 10 total -> 1 survives
- 2025-04: 6 total -> 2 survive

Total:
- 102 total BTTS rows
- 21 survivors
- survival rate = 20.6%

### What this means

This is now very clear at the upstream candidate layer:
- the odds floor is not the choke
- the candidate edge threshold is the choke

That matters because it changes the next optimization question.

The question is not:
- should odds-floor tuning be loosened?

The question is:
- should candidate edge thresholds become league-aware?
- should near-miss aligned rows be ranked more intelligently instead of simply dropped?
- should BTTS ValueEV promotion use a stricter whitelist plus shadow-bucket monitoring rather than one blunt candidate edge cut?

Locked interpretation:
- odds floor is not the choke
- candidate edge threshold is the choke
- the next refinement lever is edge / ranking policy, not odds-floor tuning

---

## Runtime BTTS YES / NO calibration

The runtime BTTS deploy logic is calibrated separately for YES and NO in `deploy_rulebook.py`.

### Runtime stamping
The BTTS deployment logic stamps:
- `btts_alignment`
- `signal_btts_runtime`

Runtime interpretation:
- aligned rows keep side-aware runtime strength
- contrarian rows are downgraded to `NEUTRAL`
- downstream BTTS runtime filtering uses the stamped runtime signal

### Core runtime BTTS YES rules
Current runtime defaults include:
- `btts_implied_max = 0.72`
- `btts_yes_pmin_floor = 0.55`
- YES strong-label requirement: `STRONG_YES` or `VERY_STRONG_YES`
- YES FTS safety: require no extreme blank-risk profile
- YES confirmation can use scored-rate / clean-sheet / conceded-rate / BTTS-rate / rolling xG checks when available

### Core runtime BTTS NO rules
Current runtime defaults include:
- `btts_no_pmin_floor = 0.52`
- NO strong-label requirement: `STRONG_NO` or `VERY_STRONG_NO`
- NO requires at least one plausible blank / clean-sheet-risk structure
- NO confirmation can use clean-sheet / low-goaliness / weak scoring / rolling xG checks when available

This is important because BTTS is not being deployed as one monolithic market gate.
It is already side-aware at runtime.

---

## Runtime BTTS ValueEV live policy

The runtime BTTS ValueEV handling is stricter than the upstream candidate-generation layer.

Current locked runtime policy constants in `deploy_rulebook.py`:
- `BTTS_LIVE_POLICY = "model_plus_whitelist_valueev"`
- `BTTS_VALUEEV_ALLOWED_LIVE = False`
- `BTTS_VALUEEV_STANDARD_PROMOTION_ENABLED = True`

This means:
- BTTS model rows remain the primary live lane
- explicit BTTS ValueEV rows are blocked from live tiers by default
- only a narrow high-conviction subset may be promoted into `STANDARD`

### Current BTTS ValueEV STANDARD promotion rules
For BTTS ValueEV rows to be eligible for `STANDARD`, they must be:
- explicit BTTS ValueEV rows
- `ALIGNED`
- side-approved by runtime label
- above side-specific edge floor
- above side-specific model-probability floor

Current side-specific promotion thresholds:

#### YES promotion
- runtime label in `{"VERY_STRONG_YES"}`
- `edge >= 0.24`
- `model_p_for_bookie >= 0.74`

#### NO promotion
- runtime label in `{"VERY_STRONG_NO"}`
- `edge >= 0.21`
- `model_p_for_bookie >= 0.71`

This is the real runtime live-promotion layer and should not be confused with the looser upstream ValueEV candidate-generation thresholds.

---

## March 2025 BTTS deploy audit

### Why this audit mattered

The current BTTS league audit was strong, but it did not yet answer a practical deployment question:

Are profitable aligned BTTS ValueEV rows being left behind by current live-routing rules?

To answer that, a shadow micro-bucket was compared against the current live whitelist baseline.

### Truth source

The initial scored backtest file did not contain the full settlement universe for the BTTS deploy artifacts.

The correct truth source was built from:
- `Matches/__merged__/*__merged.csv`

Resolver written to:
- `predictions_output/BTTS_TAG_TEST/BTTS_RESULTS_RESOLVER_FROM_MERGED_2025-03.csv`

This resolver includes:
- `league`
- `fixture_key`
- `market = btts`
- `home_team_goal_count`
- `away_team_goal_count`
- derived `actual_btts`

This resolved the earlier join failure against partial scored outputs.

### Audit files

- `predictions_output/BTTS_TAG_TEST/BTTS_SHADOW_MICRO_BUCKET_TEST_B.csv`
- `predictions_output/BTTS_TAG_TEST/BTTS_LIVE_WHITELIST_BASELINE.csv`
- `predictions_output/BTTS_TAG_TEST/BTTS_SHADOW_AUDIT_SUMMARY.md`
- `audit_btts_shadow_bucket.py`

### Locked audit result

- shadow fixture overlap with BTTS results: 3
- live fixture overlap with BTTS results: 53
- shadow merge hit rate: 1.0000
- live merge hit rate: 0.9636
- shadow rows: 3
- live rows: 55
- shadow resolved: 3 | hit_rate: 1.0
- live resolved: 53 | hit_rate: 0.9811

### Interpretation

This is the key deployment finding from today.

It tells us:
- the live whitelist baseline is elite on resolved rows
- the shadow micro-bucket also hit perfectly in this small sample
- the shadow concept is validated
- current BTTS live policy is safe, but may be slightly conservative

What it does not justify yet:
- blindly widening live policy
- auto-promoting every near-miss aligned BTTS ValueEV row

So the correct operational read is:
- keep the current candidate whitelist strict at the league-policy layer
- keep runtime live promotion stricter than candidate selection
- keep tracking near-miss aligned rows in shadow
- use the shadow process to measure profit leakage before changing promotion logic

### Unresolved live rows

These are unresolved in the merged truth source and should not be counted as losses:
- `Brazil Serie A | 2025_03_30_Palmeiras_Botafogo | btts`
- `Norway Eliteserien | 2025_03_30_Troms_Haugesund | btts`

Possible causes:
- missing settlement row in merged file at audit time
- fixture-key / team-name normalization issue
- accent handling (`Tromsø` vs `Troms`)

---

## What has been proven

At this stage, the BTTS work has established the following:

1. BTTS is strong enough to stand as its own deploy family
2. League-aware routing is required at the walk-forward policy layer
3. The BTTS ValueEV candidate whitelist is already real
4. The runtime BTTS deploy logic is side-aware and YES/NO-calibrated
5. Broad BTTS ValueEV live routing remains intentionally constrained
6. The shadow bucket concept is valid
7. Merged match files can act as the settlement truth layer for deploy audit

---

## What has not yet been proven

The current work does not yet fully prove:
- whether BTTS YES is stronger than BTTS NO under the filtered live policy
- whether the filtered BTTS ValueEV policy materially outperforms the base BTTS model lane in a stable way across all months
- whether some review leagues deserve promotion under league-specific thresholds
- whether near-miss aligned shadow rows remain strong enough at larger sample sizes to justify whitelist expansion
- whether current BTTS ValueEV STANDARD promotion thresholds are the final optimal runtime settings

---

## Remaining work

The remaining work is not rescue BTTS.

The remaining work is:
- BTTS YES audit
- BTTS NO audit
- filtered-vs-model head-to-head comparison
- league-specific promotion / demotion logic
- extended month-by-month shadow-bucket tracking
- investor-ready BTTS product summary and one-pager

---

## Deployment recommendation

### Walk-forward league-policy level
- BTTS ValueEV candidate routing = whitelist candidate pool only
- Review leagues = shadow / reduced trust
- Blacklist leagues = off
- Insufficient evidence leagues = holdout until more walk-forward data exists

### Runtime live level
- BTTS model remains the live primary lane
- BTTS ValueEV broad live routing remains off even inside whitelist leagues
- only aligned, high-conviction BTTS ValueEV rows may be promoted into `STANDARD` under the current side-specific runtime thresholds
- whitelist status supports candidate selection and controlled promotion review, not blanket live permission
- shadow micro-bucket tracking should continue
- merged match resolver remains the current canonical settlement source for deploy audit

---

## Current locked conclusion

BTTS is now established as a functioning, audited deploy family.

Current locked state:
- league-aware routing is required
- the BTTS ValueEV whitelist identifies the strongest candidate league pool, not blanket live permission
- BTTS model remains the live primary lane at runtime
- BTTS ValueEV rows are handled more strictly than the league-policy file alone implies
- runtime BTTS handling is YES / NO calibrated
- review leagues remain shadow / research lanes
- blacklist league remains off
- shadow-bucket auditing is now part of the BTTS profit-leak detection process
- next priority is BTTS YES / NO split analysis and refreshed model-vs-filtered comparison

---

## Source

- `btts_walkforward_league_audit.csv`
- `predictions_output/BTTS_TAG_TEST/BTTS_RESULTS_RESOLVER_FROM_MERGED_2025-03.csv`
- `predictions_output/BTTS_TAG_TEST/BTTS_SHADOW_AUDIT_SUMMARY.md`
- `bookie_allmarkets.py`
- `deploy_rulebook.py`

## Output files

- `btts_league_policy.json`
- `btts_live_whitelist.csv`
- `btts_shadow_review.csv`
- `BTTS_WALKFORWARD_SUMMARY.md`
- `predictions_output/BTTS_TAG_TEST/BTTS_RESULTS_RESOLVER_FROM_MERGED_2025-03.csv`
- `predictions_output/BTTS_TAG_TEST/BTTS_SHADOW_AUDIT_SUMMARY.md`

