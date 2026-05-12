# Fixture Decision Reconciler Spec

## Objective
Build `fixture_decision_reconciler.py` as the canonical backend layer between:
- model / public fixture signal
- team intelligence
- player / squad intelligence
- lineup unit intelligence
- H2H support
- market suitability

The reconciler must produce a stable, deterministic, publish-safe fixture decision object for frontend consumption.

## Product Role
This is the layer that turns:
- prediction
- ratings
- lineup structure
- matchup context

into:
- one decision state
- one agreement score
- one support stack
- one caution stack
- one public-safe summary

It should answer:
1. What is the model signal?
2. Do the team ratings agree?
3. Do the home/away profiles agree?
4. Do the lineups agree?
5. Is there a meaningful unit mismatch?
6. Are there player-level drivers?
7. Is there a caution, trap, or avoid layer?

## Frontend Contract Rule
The frontend should not compose fixture intelligence by joining team, player, lineup, and H2H layers at render time.

It should read **one canonical reconciled fixture object** from the publish layer.

That keeps the page:
- fast
- deterministic
- consistent
- audit-friendly

## Canonical Output
Each fixture output must include:

1. `fixture_key`
2. `fixture`
3. `primary_signal`
4. `signal_state`
5. `agreement_score`
6. `confidence_band`
7. `supporting_layers`
8. `caution_layers`
9. `profile_tags`
10. `profile_narrative`
11. `team_faceoff_summary`
12. `unit_battle_summary`
13. `key_player_drivers`
14. `key_mismatches`
15. `h2h_context`
16. `market_suitability`
17. `market_intelligence`
18. `watchlist`
19. `preview`
20. `public_safe_summary`
21. `internal_reason_tokens`

## Signal States
Allowed states:
- `SUPPORTED`
- `MIXED`
- `FRAGILE`
- `AVOID`
- `WATCHLIST`

### Meaning
- `SUPPORTED`
  Most layers agree with the primary signal.
- `MIXED`
  Some layers support the signal, but one or more structural layers disagree.
- `FRAGILE`
  Signal exists but confidence is reduced by caution layers or missing alignment.
- `AVOID`
  Contradiction is too strong for a clean public signal.
- `WATCHLIST`
  Useful for live monitoring or in-play triggers, but not a clean pre-match deploy.

## Required Inputs

### Public fixture layer
- `frontend/public/data/fixture_intelligence_public.json`

### Team intelligence layer
- `frontend/public/data/team_intelligence/team_ratings_index.json`
- `frontend/public/data/team_intelligence/teams/<competition_key>/<season>/<team_slug>.json`

### Player / squad intelligence layer
- `frontend/public/data/player_intelligence/club_squad_ratings.json`
- `frontend/public/data/player_intelligence/clubs/<competition_key>/<season>/<club_slug>.json`

### Lineup intelligence layer
- `frontend/public/data/fixture_lineup_intelligence/<fixture_key>.json`

### H2H support layer
- `frontend/public/data/fixture_h2h_support/<fixture_key>.json`

## Deterministic Rules

### Public-safe only
- never expose raw provider stats
- never expose formulas
- never expose exact provider columns in public JSON

### Missing data
- if lineup data is missing, fall back to team + squad layers
- if H2H is missing, mark it unavailable rather than negative
- if both team layers are missing, degrade to `FRAGILE` or `AVOID`
- preview copy must be generated server-side during publish, never in-browser

### Caution discipline
- every signal must surface at least one support reason when available
- every signal must surface at least one caution reason when available
- contradiction downgrades state even if the model read is positive

## Agreement Score
`agreement_score` is a `0-100` reconciled confidence read across:
- team layers
- home/away layers
- lineup layers
- player driver layers
- H2H support

Suggested interpretation:
- `80-100` → strong agreement
- `65-79` → moderate agreement
- `50-64` → mixed
- `35-49` → fragile
- `<35` → avoid

## Preview Layer
Each published fixture decision object can be enriched after reconciliation with a deterministic `preview` object generated only from public-safe fields.

Required preview fields:
- `headline`
- `short_summary`
- `market_summary`
- `caution_line`
- `telegram_summary`
- `premium_summary`

Rules:
- use template logic first
- include both support and caution
- never expose raw provider stats
- never expose formulas
- keep tone calm, premium, analytical

## Market Intelligence
`market_intelligence` is the promoted per-market judgement object.

Each market entry should provide:
- `alignment_score`
- `state`
- `model_lean`
- `structural_support`
- `cautions`
- `public_summary`

This lets the frontend say:
- best aligned market
- secondary market
- weak market
- avoid market

without recomputing market reasoning in-browser.

## Watchlist
`WATCHLIST` remains a public-safe product state for structures worth monitoring without forcing pre-match deployment.

The published `watchlist` object should provide:
- `active`
- `summary`
- `trigger_signals`
- `public_state`

This is intentionally lightweight for now. It should support future live/in-play product work without exposing internal trigger logic or overbuilding the first version.

## Supporting Layers
Examples:
- `TEAM_POWER_ADVANTAGE`
- `HOME_FORTRESS_ADVANTAGE`
- `ATTACK_VS_DEFENCE_MISMATCH`
- `MIDFIELD_CONTROL_ADVANTAGE`
- `LOW_HOME_CHAOS`
- `GOAL_ENVIRONMENT_SUPPORT`
- `BTTS_PRESSURE_SUPPORT`

## Caution Layers
Examples:
- `AWAY_GOAL_HEAT_RISK`
- `BTTS_PARTIAL_RISK`
- `LINEUP_DATA_MISSING`
- `H2H_UNAVAILABLE`
- `HIGH_HOME_CHAOS`
- `OPPOSITION_ATTACK_ACCESS`
- `DISCIPLINE_HEAT_RISK`

## Team Faceoff Summary
Should contain:
- only signal-relevant ratings
- home value
- away value
- delta
- directional read

This should be adaptive by market:
- `FTR`
- `BTTS`
- `OU25`

## Unit Battle Summary
Should use lineup intelligence where available:
- `attack_unit`
- `midfield_control`
- `defensive_unit`
- `wide_threat`
- `central_threat`
- `discipline_risk`

The output should emphasize mismatch, not raw volume.

## Key Player Drivers
Must identify a small set of player-level reasons, not a squad dump.
Good examples:
- elite creator
- attack spearhead
- defensive anchor
- pressing engine
- booking risk driver

## Market Suitability
Required shape:
- `ftr`
- `btts`
- `ou25`
- `team_goals`
- `correct_score`
- `corners`
- `cards`

Each should emit:
- `rating`
- `label`
- `read`

## Build Sequence
1. define canonical schema in `docs/intelligence_schema.json`
2. build `fixture_decision_reconciler.py`
3. publish `frontend/public/data/fixture_decision_intelligence/<fixture_key>.json`
4. publish `frontend/public/data/fixture_decision_intelligence/index.json`
5. wire frontend fixture page to the reconciled object

## Refresh / Cache Policy
The reconciler should re-run when one of these changes:
- public fixture signal refresh
- team intelligence refresh
- player intelligence refresh
- lineup intelligence publish
- H2H support publish

The published reconciled object is the cache.
The website reads that object, not the underlying layers directly.

## Frontend Intent
The frontend should ultimately consume the reconciled object for:
- top verdict card
- decision companion / agreement stack
- team face-off
- unit battle
- player drivers
- market suitability
- caution framing

This layer is the brain between the models and the website.
