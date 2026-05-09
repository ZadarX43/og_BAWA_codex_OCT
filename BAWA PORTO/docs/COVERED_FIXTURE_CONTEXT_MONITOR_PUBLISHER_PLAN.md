# Covered Fixture Intake And CONTEXT/MONITOR Publisher Plan

## Purpose

Define the next backend layer after the first routed fixture-intelligence exporter.

This plan covers the non-routed fixture lane:
- fixtures that do not become `DEPLOY`
- fixtures that do not become `OBSERVE`
- fixtures that still belong to leagues Odds Genius covers
- fixtures that still matter for followed teams, followed fixtures, and premium intelligence delivery

This is the bridge from:
- routed intelligence publishing

to:
- full covered-league fixture intelligence coverage

It does not change:
- `bookie_allmarkets.py`
- `deploy_rulebook.py`
- protected model routing semantics
- the production prediction spine

It defines a separate publish-safe layer that sits downstream of them.

## Core Product Principle

Low signal is still intelligence.

No routed signal is still intelligence.

The platform should not become silent simply because a fixture failed to:
- deploy
- observe
- qualify for live premium picks

For covered leagues, the system should still be able to say:
- what fixture is coming
- what context matters
- whether scoring shape is muted or elevated
- whether market context is stable or fragile
- whether weather, lineup, injuries, or player-intelligence overlays matter
- whether the fixture is worth monitoring for a followed user

That is how Odds Genius becomes:
- a prediction operating system
- a personalized football intelligence surface
- a useful product even when no official pick exists

## Why This Layer Matters

The routed intelligence layer currently gives us:
- `DEPLOY`
- `OBSERVE`

But not yet:
- `CONTEXT`
- `MONITOR`

Without `CONTEXT` and `MONITOR`, the product still has a hard ceiling:
- it only speaks when routed outputs exist
- it cannot fully serve followed teams
- it cannot fully serve followed fixtures
- it cannot surface intelligence in low-signal periods

With `CONTEXT` and `MONITOR`, the platform becomes:
- broader
- stickier
- more personalized
- more defensible

## Source-System Philosophy

The non-routed lane should be built from two distinct source families:

### 1. Stable goal-market / fixture-shape base

Primary base:
- FootyStats-era fixture and goal-shape outputs
- merged / canonical football context already trusted by the core stack
- all-markets and support fields where they exist

Why:
- this is still the most stable backbone for broad fixture-level goal intelligence
- it gives us the cleanest base for scoring shape, totals posture, BTTS posture, and fixture completeness

### 2. API-Football overlay and enrichment layer

Overlay / enrichment:
- fixtures master
- prematch odds snapshots
- lineups
- injuries
- team stats
- player stats
- match events
- later player-intelligence signals such as shots, tackles, fouls, cards, assists, key passes

Why:
- this is where advanced context and player intelligence live
- this is critical for premium intelligence depth
- this creates differentiation even when no deployable pick exists

## High-Level Architecture

The covered fixture lane should operate in four stages:

1. Coverage intake
   - identify all upcoming fixtures in covered leagues and date window

2. Safe intelligence shaping
   - derive normalized fixture context from approved source fields
   - compress weak/latent signal shape into non-recommendation language

3. Class decision
   - decide whether a non-routed fixture becomes:
     - `CONTEXT`
     - `MONITOR`
     - or `HIDDEN`

4. Publish-safe export
   - merge those fixtures into the website-safe fixture intelligence artifact family
   - support website cards, watchlists, and Telegram later

## Canonical Inputs For The Non-Routed Lane

The non-routed lane must not depend on raw API payloads at publish time.

It should consume normalized and approved inputs only.

### A. Coverage identity inputs

These define which fixtures exist in the window.

Preferred:
- `data_sources/api_football/normalized/fixtures_master*.csv`

Role:
- canonical upcoming fixture identity
- competition
- teams
- kickoff time
- fixture ids where available

### B. Stable market / shape base

These define whether a fixture has broad goal-market or result-shape intelligence even when not routed.

Preferred candidates:
- current-window all-markets files where available
- approved merged / support outputs already used by the production path
- FootyStats-era stable goal-market families
- `DEPLOY_CANDIDATES_RAW.csv`
- `DEPLOY_CANDIDATES_AFTER_GATES.csv`

Role:
- reveal weak or latent signal shape before / outside routed output
- provide low-signal FTR / BTTS / OU25 posture
- preserve the “poor signal is still signal” philosophy without promoting it into picks

### C. Odds and market context inputs

Preferred:
- `data_sources/api_football/normalized/odds_prematch_long*.csv`
- approved website-safe odds summary extracts

Role:
- compact market view
- odds availability
- broad price posture
- later drift / movement alerts once supported safely

### D. Overlay / enrichment inputs

Preferred:
- `data_sources/api_football/normalized/injuries*.csv`
- `data_sources/api_football/normalized/lineups*.csv`
- `data_sources/api_football/normalized/match_team_stats*.csv`
- approved normalized player / event overlays later

Role:
- injuries
- lineups / formations
- volatility context
- player-intelligence overlays
- later pre-match or live summary signals

### E. Routed layer references

Preferred:
- `__DEPLOY_TIER_ELITE__`
- `__DEPLOY_TIER_STANDARD__`
- `__DEPLOY_TIER_OBSERVE__`
- `fixture_intelligence_public.json`

Role:
- exclude already-handled fixtures from pure non-routed classification when needed
- prevent duplicate or contradictory publish states
- allow the covered-fixture lane to fill the gaps rather than redoing routed work

## Coverage Universe Definition

The first job of the `CONTEXT` / `MONITOR` lane is to define:

### “What is a covered fixture?”

A fixture should be considered covered if all are true:
- it belongs to a league currently inside the Odds Genius covered competition set
- it falls inside the active publish window
- it has valid identity fields:
  - competition
  - kickoff time
  - home team
  - away team
- it is not malformed or unsupported

### Covered window rules

The publish window should match the active website / weekend operator window.

Recommended operator definition:
- from earliest current publish date
- to latest current publish date

This should usually derive from:
- selected deploy/all-markets run window
- or current fixture snapshot window if running a context-only publish pass

## Fixture Intake Flow

Recommended intake flow:

1. build the candidate fixture universe from normalized fixtures master
2. restrict to covered leagues
3. restrict to publish window
4. attach identity and logo bridge
5. join any routed fixture keys already published
6. attach approved market / shape base where available
7. attach approved enrichment summaries
8. classify:
   - routed already handled
   - `CONTEXT`
   - `MONITOR`
   - `HIDDEN`

## Relationship To Routed Outputs

This lane is not meant to replace the routed exporter.

It should supplement it.

### Recommended separation

Routed exporter owns:
- `DEPLOY`
- safe `OBSERVE`

Covered-fixture lane owns:
- `CONTEXT`
- `MONITOR`
- optionally non-routed weak-shape context later

### Important rule

If a fixture already exists in routed output as:
- `DEPLOY`
- `OBSERVE`

the covered-fixture lane should not overwrite that primary state.

It may:
- enrich it later
- attach context later
- support notifications later

but should not downgrade or conflict with routed classification.

## Classification Rules For Non-Routed Fixtures

## `CONTEXT`

Use `CONTEXT` when:
- there is no clean routed deploy/observe row
- but there is meaningful intelligence worth surfacing
- and the main value is context rather than a model lean

Examples:
- key injury
- probable lineup disruption
- heavy weather
- notable formation uncertainty
- major market caution
- schedule congestion / fatigue
- player-intelligence overlay of real relevance

### Product meaning

`CONTEXT` means:
- there is no public deployable signal
- but there is real fixture-level information worth knowing

### Safe wording

Use:
- `Context alert`
- `Monitoring note`
- `Pre-match context remains relevant`
- `No deployable edge. Context remains relevant.`

Avoid:
- `prediction`
- `pick`
- `bet`
- `recommendation`

## `MONITOR`

Use `MONITOR` when:
- the fixture is covered
- no strong routed signal exists
- no major context alert exists
- but the fixture still deserves presence in the watch / follow environment

Examples:
- followed team fixture with no routed edge
- league-completeness card
- low-signal game with odds and basic shape only
- weak intelligence that is still worth retaining for completeness

### Product meaning

`MONITOR` means:
- covered fixture
- no strong deployable edge at current state
- no major contextual event strong enough to elevate to `CONTEXT`
- still relevant for awareness and follow systems

### Safe wording

Use:
- `Covered fixture. No strong deploy signal at current state.`
- `Monitoring fixture context ahead of kickoff.`
- `No strong public signal. Fixture remains covered for follow and context updates.`

## Minimum Required Fields For Non-Routed Publish

No non-routed fixture should be published unless it has:

- `fixture_key` or equivalent canonical fixture identity
- `kickoff_time`
- `league`
- `home_team`
- `away_team`
- logo join attempt / status
- `publish_class`
- `signal_summary.signal_state`
- `signal_summary.summary_text`
- `updated_at`

If any of those fail:
- demote to `HIDDEN`
- record it in operator reporting

## Allowed Field Families For CONTEXT/MONITOR

### Identity
- fixture id / key
- kickoff time
- league
- home team
- away team
- league/team badges

### Odds summary
- high-level approved bookmaker summary only
- no raw provider blobs

### Context summary
- weather note
- injury note
- lineup note
- formation note
- market movement note
- fatigue note
- form note
- H2H note
- volatility note

### Signal summary

For `CONTEXT`:
- `signal_state = "context_only"`
- `signal_strength = "none"`
- one safe summary sentence

For `MONITOR`:
- `signal_state = "monitor_only"`
- `signal_strength = "none"` or very low implicit stance
- one safe summary sentence

## Explicit Non-Goals

This layer should not:
- expose raw API-Football payloads
- expose raw FootyStats payloads
- expose route mechanics
- expose thresholds
- expose raw internal probabilities unless already approved by schema policy
- expose hidden player-feature engineering
- behave like a shadow picks feed

## How To Stamp Intelligence For Non-OBSERVE Fixtures

This is the key new behavior.

For fixtures that never make it into `OBSERVE`, we still need a stamping layer.

That stamping should not claim:
- actionable signal
- recommendation
- pick

It should stamp one of:

### 1. Weak shape stamp

Examples:
- `Muted scoring profile`
- `Low-event watch`
- `Home-side pressure tilt`
- `Away-side attacking caution`

Use only if supported by approved stable market / shape families.

### 2. Context stamp

Examples:
- `Weather caution`
- `Injury watch`
- `Lineup pending`
- `Congestion risk`
- `Volatility note`

### 3. Coverage stamp

Examples:
- `Covered fixture`
- `Monitoring kickoff context`
- `Follow-only fixture coverage`

This is how poor signal still becomes useful signal:
- not by promoting it to a pick
- but by turning it into a normalized awareness stamp

## Reasoning Generation Rules

Reasoning for non-routed fixtures must be:
- short
- plain English
- context-first
- non-recommendation
- deterministic from safe fields

### Good examples

- `No deployable edge. Heavy weather risk keeps this fixture in context view.`
- `Covered fixture. No strong public signal at current state, but lineup confirmation may matter later.`
- `Monitoring low-event scoring shape ahead of kickoff.`
- `No routed signal, but injury and fatigue context remain relevant.`

### Bad examples

- `Model predicts under`
- `Strong betting edge`
- `Back BTTS no`
- `Lock this in`

## How Followed Users Still Get Value

This layer is especially important for:
- followed teams
- followed fixtures
- followed leagues

Even without a routed signal, followed users should still be able to receive:

- pre-match context summary
- lineup pending note
- injury note
- weather caution
- weak shape note
- monitor card presence
- post-match recap later

### Example: followed team with no signal

User follows Liverpool.

Liverpool has:
- no deploy row
- no observe row

The system can still provide:
- fixture card
- odds summary
- injury note
- weather note
- fatigue note
- monitor summary

That is still real product value.

## Notification Behavior For CONTEXT/MONITOR

### `CONTEXT`

Immediate candidate only when:
- user follows the team or fixture
- context is material
- urgency is real

Examples:
- major injury
- severe weather
- major lineup disruption

Otherwise:
- digest or account surface

### `MONITOR`

Usually not immediate Telegram push.

Best uses:
- account watchlists
- daily digest
- pre-match followed team summary

The key rule:
- `CONTEXT` can notify
- `MONITOR` usually informs

## Recommended Output Strategy

Recommended near-term strategy:

### Phase 1

Keep one artifact:
- `fixture_intelligence_public.json`

Extend it so the combined publish family contains:
- routed `DEPLOY`
- routed `OBSERVE`
- non-routed `CONTEXT`
- non-routed `MONITOR`

### Phase 2

Optional split artifacts:
- `fixture_intelligence_context_monitor.json`
- `fixture_follow_digest.json`
- `fixture_alert_feed.json`

Phase 1 is simpler and safer:
- one combined artifact
- one validator family
- one frontend consumption path later

## Recommended Backend Components

Task 57 should lead into these implementation pieces:

1. covered fixture intake builder
   - identify covered upcoming fixtures by window and league

2. context/monitor join layer
   - attach safe odds/context/overlay summaries

3. context/monitor classifier
   - decide `CONTEXT` vs `MONITOR` vs `HIDDEN`

4. publish merger
   - merge routed and non-routed fixtures into one output family

5. validator extension
   - enforce non-routed class rules

## Suggested File Direction

Possible future implementation files:

- `publish_fixture_context_monitor.py`
- or extend `publish_fixture_intelligence.py` once the intake rules are stable

Recommended intermediate helpers:
- `build_covered_fixture_universe.py`
- `build_context_monitor_stamp_rules.py`
- `validate_fixture_context_monitor.py`

The safest path is:
- start with separate helper(s)
- merge into the main publisher only after the joins and stamps stabilize

## Risks

### 1. Raw provider leakage

Risk:
- accidentally exposing raw API-Football structure or hidden feature logic

Mitigation:
- strict allowlist only
- normalized summaries only

### 2. Noise inflation

Risk:
- publishing too many weak fixtures creates clutter

Mitigation:
- `CONTEXT` must require material value
- `MONITOR` should stay lightweight
- use follow relevance to prioritize user-facing surfaces

### 3. False “signal” presentation

Risk:
- weak shape gets interpreted as a recommendation

Mitigation:
- use monitor/context language only
- never use pick/recommendation vocabulary

### 4. Contradictory source families

Risk:
- FootyStats base and API-Football overlays disagree or arrive at different freshness levels

Mitigation:
- FootyStats-era goal outputs remain stable base
- API-Football remains overlay layer
- publish summaries should record freshness and conflicts later

## Implementation Order After This Plan

1. define the covered fixture universe builder
2. define the non-routed context/monitor stamp schema
3. implement a first standalone context/monitor exporter
4. merge it into the main fixture-intelligence publisher
5. extend frontend and follow notifications later

## Summary

Task 57 is the plan for the missing lane in Odds Genius:

- not picks
- not observe rows
- still meaningful

It formalizes how the platform can publish useful fixture intelligence for games that:
- never become deployable
- never become observe-worthy
- but still matter inside covered leagues and followed-user journeys

That is the bridge from:
- routed fixture intelligence

to:
- full covered-fixture intelligence coverage

And that is how Odds Genius becomes a true football intelligence operating system rather than a premium picks surface.
