# Fixture Intelligence Publishing Plan

## Purpose

Define how Odds Genius should publish fixture-level intelligence beyond the current deploy-only board.

This plan covers:
- how `OBSERVE` files become website-safe fixture intelligence
- how covered-league fixtures are published even when there is no deployable pick
- what fields are allowed in each published fixture class
- what data comes from model output vs API-Football vs enrichment
- how followed-team and followed-fixture notifications should be generated
- how low-confidence model signal should be represented without pretending it is a deployable recommendation

This is not a plan to change:
- live model logic
- `deploy_rulebook.py`
- production routing gates
- the protected Python prediction spine

It is a plan to create a separate, safe publishing layer on top of those systems.

## Core Principle

Low signal is still intelligence.

The platform should not be binary:
- signal
- no signal

It should support four outcome classes:
- deployable signal
- observe signal
- contextual intelligence
- monitored fixture

That is how Odds Genius becomes:
- selective when confidence is high
- informative when confidence is weak
- observational when the market is unstable or inconclusive

## Final Platform Structure

The website and delivery system should operate across five layers:

1. Deploy
   - actionable routed signals
2. Observe
   - model-derived but non-deployable signal shape
3. Follow
   - personalised team / fixture / market monitoring
4. Delivery
   - website, Telegram, later app/email
5. Archive
   - transparency, grading, proof

This document focuses on Layers 1 to 4 at publish time.

## Fixture Classification System

Every covered fixture should eventually resolve into one of four published classes:

### 1. `DEPLOY`

Definition:
- high-confidence routed output approved by the live engine

Examples:
- ELITE BTTS
- ELITE OU25
- STANDARD FTR
- value-edge premium board entry
- acca-eligible deploy row

Characteristics:
- actionable
- confidence-tiered
- premium/public board eligible depending visibility rules

### 2. `OBSERVE`

Definition:
- model has identified a meaningful lean or pattern
- row is not deployable under the live rulebook

Examples:
- weak BTTS shape
- moderate totals lean
- unstable favourite
- draw-fragile spot
- chaotic derby profile
- team-goals tendency below deploy threshold

Characteristics:
- informative
- non-deployable
- should not be presented as a bet recommendation

### 3. `CONTEXT`

Definition:
- no strong publishable model lean is required
- fixture still has contextual intelligence worth surfacing

Examples:
- severe weather
- key injury
- late lineup disruption
- congestion / fatigue
- market drift
- formation uncertainty

Characteristics:
- informational
- can exist with or without deploy/observe output
- useful for followers and advanced users

### 4. `MONITOR`

Definition:
- covered fixture in a covered league with no strong current edge
- still worth surfacing as part of a user’s monitored environment

Examples:
- fixture card with odds, coverage, and basic intelligence context
- followed team with no deploy signal
- fixture with weak shape but no major alert

Characteristics:
- low-friction
- low-urgency
- completeness / awareness layer

### 5. `HIDDEN`

Definition:
- fixture should not be published into the website intelligence layer at all

Examples:
- unsupported league
- malformed fixture mapping
- missing identity joins
- non-public experimental rows
- sensitive internal-only feature cases

## Canonical Source Layers

The publishing architecture should be treated as a layered pipeline.

### Raw Source Layer — private

Never exposed directly.

Includes:
- API-Football raw fixture / lineup / odds / event payloads
- routed deploy CSVs
- OBSERVE rows
- model probabilities
- overlays
- xG / team-goals / score-shape internals
- enrichment families

### Processing Layer — private

Still not public.

Includes:
- fixture matching
- deploy / observe classification
- volatility tagging
- reason token compression
- alert generation
- safe wording generation
- field allowlisting

### Published Layer — website-safe

Only approved fields.

This should become the source for:
- fixture intelligence cards
- followed fixture pages
- Telegram intelligence alerts
- watchlist views
- future app delivery

## Publishable Artifact Model

Recommended output family:

- `public_predictions.json`
  - existing
- `premium_predictions.json`
  - existing
- `fixture_intelligence_public.json`
  - new
- `fixture_intelligence_premium.json`
  - optional, if richer premium context is needed
- `fixture_follow_digest.json`
  - optional later

The first important step is:
- `fixture_intelligence_public.json`

This should not replace deploy boards.
It should supplement them.

## Canonical Inputs For Fixture Intelligence

### Prediction / deploy inputs
- routed deploy CSV
- OBSERVE rows
- publish summary lineage

### Coverage / fixture inputs
- approved fixture identity layer
- approved competition / team identity mappings

### Enrichment inputs
- odds snapshots
- weather enrichment
- injury/news enrichment
- optional form / H2H / schedule-congestion summaries
- optional formation / lineup summaries when available

### Important rule

The publish step must use:
- approved normalized fields
- not raw upstream provider payloads

## Safe Publish Fields By Fixture Class

## `DEPLOY` Allowed Fields

Allowed:
- fixture id / key
- competition / league
- kickoff time
- teams
- team / league badges
- market
- pick
- confidence tier
- public-safe signal strength label
- bookmaker odds
- summary reason
- correct score shortlist if approved
- value-edge label if already public-safe

Do not expose:
- raw internal thresholds
- raw model feature vectors
- exact route gate internals
- hidden reason-token stacks

## `OBSERVE` Allowed Fields

Allowed:
- fixture metadata
- `observe_label`
  - e.g. `Observed BTTS lean`
  - `Observed scoring bias`
  - `Observed home-side pressure shape`
- public-safe shape summary
- volatility markers
- weather note
- injury / lineup caution note
- broad market context
- low / medium signal label

Do not expose:
- exact raw probabilities unless separately approved
- route threshold values
- detailed model disagreement internals
- anything that implies it is a deployable recommendation

## `CONTEXT` Allowed Fields

Allowed:
- fixture metadata
- competition / team identity
- odds summary
- weather note
- injury note
- lineup note
- formation note
- market movement note
- fatigue / congestion note
- public-safe H2H / form summary if approved

Do not expose:
- raw provider fields
- unvalidated long-form news dumps
- hidden model internals

## `MONITOR` Allowed Fields

Allowed:
- fixture metadata
- teams
- kickoff
- odds snapshot
- simple intelligence state
  - `No strong deploy signal`
  - `Monitor only`
  - `Awaiting lineup / weather / news`
- optional coverage tags

This class is mostly for:
- completeness
- followed teams
- followed fixtures
- league coverage awareness

## Field Provenance Rules

Every published field should have a known source family.

### From model output

Allowed examples:
- public-safe confidence label
- deploy tier
- observe label
- shape summary
- correct score shortlist if approved
- weak market lean wording

Not allowed directly:
- raw training features
- internal gate thresholds
- hidden route mechanics

### From API-Football or equivalent enrichment

Allowed only after normalization:
- fixture metadata
- league / team names
- kickoff time
- lineups / formations when normalized
- injury/news summaries when approved
- odds snapshots if normalized

Not allowed directly:
- raw JSON payloads
- unfiltered provider structures
- undocumented internal IDs that leak source complexity

### From enrichment systems

Allowed examples:
- weather summary
- fatigue / congestion flags
- form summary
- H2H headline summary
- odds movement summary

These should be:
- compressed
- normalized
- phrased consistently

## Representation Rules For Low-Confidence Signal

This is a core product rule.

Never represent low-confidence signal as:
- `Prediction`
- `Bet`
- `Pick`

Instead represent it as:
- `Model lean`
- `Observed shape`
- `Contextual signal`
- `Monitoring note`
- `Scoring profile`

Examples:

Bad:
- `BTTS YES`

Better:
- `Observed BTTS lean based on attacking convergence and defensive instability.`

Bad:
- `Over 2.5 prediction`

Better:
- `Goal-shape profile suggests elevated scoring potential.`

Bad:
- `Home win signal`

Better:
- `Observed home-side lean, but not strong enough for deployment.`

This protects:
- legal framing
- product clarity
- deploy / observe separation

## OBSERVE Publishing Strategy

Current OBSERVE rows already exist in the protected routing world.

The website should not consume them directly.

Instead:

1. read routed outputs with OBSERVE lineage
2. classify which OBSERVE rows are safe and useful for publishing
3. convert them into:
   - `observe_class`
   - `observe_summary`
   - `volatility_context`
4. publish only allowlisted fields

Recommended first safe OBSERVE outputs:
- market family
- observe label
- one summary sentence
- one to three context tags

## Covered-League Publishing Without Picks

This is the major unlock.

For competitions we already cover, the platform should be able to publish fixture intelligence even when there is no deployable pick.

Recommended rule:

If a fixture belongs to a currently covered / monitored competition, publish one of:
- `DEPLOY`
- `OBSERVE`
- `CONTEXT`
- `MONITOR`

That means a user following a team can still receive:
- weather
- injuries
- odds
- formation notes
- weak model lean
- goal-shape tendency

even when no premium board signal exists.

## Followed-Team And Followed-Fixture Notification Generation

This should be driven from the published intelligence layer, not raw private inputs.

### Followed team trigger examples
- new deploy involving followed team
- key injury on followed team
- major weather alert affecting followed team fixture
- scoring-shape summary on followed team fixture
- volatility spike on followed team fixture

### Followed fixture trigger examples
- fixture moved from monitor -> observe
- observe -> deploy
- new injury / lineup alert
- severe weather change
- market drift threshold crossed
- post-match result digest

### Important rule

Users should not receive every update.
They should receive:
- relevant updates
- filtered by preferences
- filtered by priority
- filtered by follow state

## Notification Generation Rules

### Immediate Telegram push

Good candidates:
- elite deployment
- high-priority observe shift on followed fixture
- severe weather on followed fixture
- major injury on followed team
- major market movement on followed fixture

### Digest-only by default

Good candidates:
- weaker observe summaries
- daily follow summaries
- weekend slate bundle
- followed team recap

### Website-only by default

Good candidates:
- low-urgency monitor context
- archive detail
- extended rationale

## Fixture Intelligence Data Model

Recommended top-level object:

- `fixture_id`
- `fixture_key`
- `publish_class`
  - `DEPLOY`
  - `OBSERVE`
  - `CONTEXT`
  - `MONITOR`
- `coverage_status`
  - `covered`
  - `follow_only`
  - `hidden`
- `kickoff_time`
- `league`
- `league_logo_url`
- `league_flag_url`
- `home_team`
- `home_team_logo_url`
- `away_team`
- `away_team_logo_url`
- `odds_summary`
- `signal_summary`
- `context_tags`
- `weather_summary`
- `injury_summary`
- `lineup_summary`
- `market_movement_summary`
- `correct_score_summary`
- `follow_relevance`
- `updated_at`

Not every class needs every field.

## Publish Safety Rules

The fixture intelligence layer must:
- use a strict allowlist
- remain deterministic
- avoid leaking raw provider payloads
- avoid leaking internal route mechanics
- avoid phrasing weak signal as a recommendation

It must not:
- override deploy gates
- backdoor OBSERVE rows into premium picks
- silently promote weak model output into signal language

## Relationship To Current Website

### Predictions page
- remains the board for deployable outputs

### Premium page
- remains the gated actionable surface

### Results page
- remains the proof and settlement layer

### Account / Telegram / future app
- become the delivery and follow-intelligence layer

### Future fixture intelligence page
- can be built on top of this publishing contract

## Recommended Implementation Order

### Phase 1
- define fixture publish classes
- define allowlisted fields
- build website-safe fixture intelligence JSON export

### Phase 2
- connect followed teams / fixtures to that published layer
- generate notification-ready alert records

### Phase 3
- build richer fixture pages
- add lineup / weather / injury summaries
- add low-signal / observe views

### Phase 4
- live updates
- push/app delivery
- broader market families

## Immediate Follow-On Tasks

1. Task 53 — Fixture Intelligence Schema Spec
2. Task 54 — OBSERVE To Public-Safe Export Rules
3. Task 55 — Followed Team / Fixture Notification Rules
4. Task 56 — Fixture Intelligence JSON Publisher

## Summary

Odds Genius should not publish only:
- bets

It should publish:
- deployable edge
- observed shape
- contextual intelligence
- monitored coverage

That lets the platform deliver value across:
- strong-signal periods
- weak-signal periods
- no-signal periods

And that is what makes it feel like:

`a football intelligence operating system`

rather than:

`a page of picks`
