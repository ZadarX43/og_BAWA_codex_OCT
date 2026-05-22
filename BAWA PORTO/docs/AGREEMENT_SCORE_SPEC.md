# Agreement Score Spec

## Purpose
`agreement_score` is the publish-safe cross-check between:
- model signal
- team intelligence
- lineup intelligence
- player-driver intelligence
- H2H support

It exists to answer one question:

**Do the supporting layers agree with the signal strongly enough to trust the read?**

This is not a raw model score and not a raw probability.
It is a judgement-quality score.

## Output
- range: `0-100`
- stable, deterministic
- safe for frontend display

## Interpretation Bands
- `80-100` → strong agreement
- `65-79` → moderate agreement
- `50-64` → mixed
- `35-49` → fragile
- `0-34` → avoid

## State Mapping
- `SUPPORTED`
  - usually `80+`
  - most major layers agree
- `MIXED`
  - usually `65-79`
  - signal is live, but not all structural layers align
- `FRAGILE`
  - usually `35-64`
  - edge exists but caution is too meaningful to ignore
- `AVOID`
  - usually `<35`
  - contradiction is stronger than support
- `WATCHLIST`
  - usually `58+` with non-deploy or observe conditions
  - structure is interesting, but not a clean pre-match deploy

## Layer Order
The score should be constructed in a ladder, not a flat pile.

### 1. Team layer
Core structural read.
Highest weighting.

Examples:
- `TEAM_POWER_ADVANTAGE`
- `BTTS_PRESSURE_SUPPORT`
- `GOAL_ENVIRONMENT_SUPPORT`
- `CONTROL_SUPPORT`

### 2. Home / away profile layer
Environment-specific support.

Examples:
- `HOME_FORTRESS_ADVANTAGE`
- `AWAY_THREAT_ADVANTAGE`
- `POWER_PARITY`

### 3. Lineup / unit layer
Second-order confirmation.
Important, but should not outweigh core team shape on its own.

Examples:
- `ATTACK_VS_DEFENCE_MISMATCH`
- `BOTH_ATTACK_UNITS_LIVE`
- `LINEUP_SUPPRESSION_SUPPORT`
- `LINEUP_DATA_MISSING`

### 4. Player-driver layer
Explanatory and confirmatory.
Should support the score, not dominate it.

### 5. H2H support layer
Supporting only.
Low weighting.
Absence is neutral-to-soft caution, not a hard contradiction.

Examples:
- `H2H_BTTS_SUPPORT`
- `H2H_OVER_SUPPORT`
- `H2H_UNAVAILABLE`

## Weighting Principle
Use weighted deterministic increments, not opaque nonlinear scoring.

Suggested pattern:
- strong structural support token: `+6`
- medium support token: `+4`
- strong caution token: `-6`
- soft caution token: `-2` to `-4`

Base score can start from a neutral midpoint such as `50`.

## Market-Specific Relevance
The score must use market-adaptive logic.

### FTR
Most relevant:
- `og_power_rating`
- `home_fortress_rating`
- `away_threat_rating`
- `defensive_lock_rating`
- `first_strike_rating`
- lineup attack vs defence mismatch

### BTTS
Most relevant:
- `btts_pressure_rating`
- `goal_heat_rating`
- `attack_flow_rating`
- `defensive_lock_rating`
- lineup attack access
- H2H BTTS regime

### OU25
Most relevant:
- `goal_heat_rating`
- `over25_heat_rating`
- `control_rating`
- `chaos_rating`
- lineup goal pressure / suppression
- H2H over regime

## Missing Data Rules
- Missing lineup data should reduce the score modestly, not kill it.
- Missing H2H should be a soft caution, not a contradiction.
- Missing both home and away team intelligence should force the score downward hard.

## Frontend Rule
The frontend should display:
- `agreement_score`
- `signal_state`
- support layers
- caution layers

The frontend should **not** recompute the score.

That logic belongs only in the reconciler.
