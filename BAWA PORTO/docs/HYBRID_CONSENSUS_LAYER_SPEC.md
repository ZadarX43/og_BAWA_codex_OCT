# HYBRID_CONSENSUS_LAYER_SPEC

## Purpose
Define the first research-grade consensus layer for hybrid core and side-market outputs.

This layer should only be designed **after**:
- tuned hybrid OU25 artifact is locked
- side-market source winners are known
- league-specific thresholds are audited

## Current locked research winners (EPL prototype)
Expected current winners after threshold audit:
- OU25: tuned CatBoost using `baseline + team + lineup + injury`
- Home Goals O1.5: direct hybrid classifier
- Away Goals O1.5: direct hybrid classifier
- Home FTS: lambda-derived
- Away FTS: lambda-derived

## Consensus philosophy
Consensus is not simple averaging.
It should behave as a gated routing layer:
1. choose the best source per market
2. apply league-style threshold policies
3. only then expose deployable selections

## Inputs
### Core lane inputs
- FootyStats baseline probability
- hybrid CatBoost probability
- hybrid XGB probability where market-specific evidence supports it

### Goal-mass inputs
- `lambda_home`
- `lambda_away`
- team-goals O1.5 probabilities
- FTS probabilities
- BTTS first-half probabilities

### Market priors
- bookmaker implied probability
- no-vig probability where available
- value edge tier

## Layer order
### Stage 1: Source selection
Pick the winning source by market.

Example:
- `OU25` -> tuned CatBoost hybrid winner
- `home_goals_over15` -> direct classifier
- `home_fts` -> lambda-derived

### Stage 2: Threshold policy
Apply per-league threshold and coverage controls.

Fields to stamp:
- `threshold_used`
- `coverage_bucket`
- `style_gate_state`
- `value_edge_state`

### Stage 3: Agreement / conflict logic
Consensus can promote or demote selections based on agreement.

Examples:
- If direct team-goals and lambda-derived FTS strongly disagree, demote to review.
- If OU25 and BTTS structure agree with high value edge, promote confidence bucket.

## League-style gate concept
Each league gets a compact threshold profile such as:
- `ou25_threshold`
- `home_goals_over15_threshold`
- `away_goals_over15_threshold`
- `home_fts_threshold`
- `away_fts_threshold`
- `minimum_coverage`
- `minimum_value_edge`
- `style_gate_notes`

## Candidate consensus outputs
### Core
- `consensus_ou25_over_p`
- `consensus_ou25_source`
- `consensus_ou25_keep_flag`

### Team goals
- `consensus_home_goals_over15_p`
- `consensus_away_goals_over15_p`

### FTS
- `consensus_home_fts_p`
- `consensus_away_fts_p`

### Meta controls
- `consensus_agreement_score`
- `consensus_conflict_flag`
- `consensus_value_edge_tier`
- `consensus_style_gate_bucket`

## Promotion rule
No consensus layer should be promoted unless it improves at least one of:
- hit rate at deploy threshold
- ROI by value tier
- false-positive suppression
- league stability
- deployable volume at protected floors

without damaging the already locked Phase 8H value-layer protections.

## Immediate next build after spec
1. extend threshold audit to additional leagues
2. build `build_hybrid_consensus_inputs.py`
3. build `audit_hybrid_consensus_matrix.py`
4. only then implement live deploy routing
